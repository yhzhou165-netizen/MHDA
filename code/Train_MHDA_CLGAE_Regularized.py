"""MHDA cross-validation with fold-specific graph inputs.

No full-data association-derived similarities are loaded. Each fold uses only
its actual training positives for every association-derived model input.
"""
import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
import random

import numpy as np
import tensorflow.compat.v1 as tf

from Adj_preprocess import load_raw_data, build_fold_inputs
from Preprocessing import make_cv_splits, assert_fold_isolation
from Model_MHDA_CLGAE import MHDA_CLGAE
from Optimizer import OptimizerMetrics

tf.disable_v2_behavior()

MODEL_CNFIGS = {
    'baseline': {
        'use_multi_view': False,
        'view_fusion_mode': 'adaptive',
        'use_heterogeneous': False,
        'use_dual_stream': False,
        'use_contrastive': False,
        'lambda_contrast': 0.0,
        'hidden_dim': 28,
        'dropout': 0.42,
        'l2_weight': 1.5e-4,
        'learning_rate': 0.00028,
        'label_smoothing': 0.15
    },
    'full': {
        'use_multi_view': True,
        'view_fusion_mode': 'adaptive',
        'use_heterogeneous': True,
        'use_dual_stream': True,
        'use_contrastive': True,
        'lambda_contrast': 0.018,
        'hidden_dim': 28,
        'dropout': 0.42,
        'l2_weight': 1.5e-4,
        'learning_rate': 0.00028,
        'label_smoothing': 0.15
    },
    'mean_view': {
        'use_multi_view': True,
        'view_fusion_mode': 'mean',
        'use_heterogeneous': True,
        'use_dual_stream': True,
        'use_contrastive': True,
        'lambda_contrast': 0.018,
        'hidden_dim': 28,
        'dropout': 0.42,
        'l2_weight': 1.5e-4,
        'learning_rate': 0.00028,
        'label_smoothing': 0.15
    }
}

VIEW_NMES = ('jaccard', 'cosine', 'integrated')


def pack_edges(pos, neg):
    edges = np.vstack((pos, neg)).astype(np.int32)
    labels = np.concatenate((np.ones(len(pos)), np.zeros(len(neg))))
    return edges, labels.reshape(-1, 1).astype(np.float32)


def build_model_graph(n_circ, n_dis, config, seed):
    """Create a fresh graph, including optimizer state, for one fold."""
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    ph = {
        'features_circ': tf.placeholder(tf.float32, (n_circ, n_circ)),
        'features_dise': tf.placeholder(tf.float32, (n_dis, n_dis)),
        'assoc': tf.placeholder(tf.float32, (n_circ, n_dis)),
        'edges': tf.placeholder(tf.int32, (None, 2)),
        'labels': tf.placeholder(tf.float32, (None, 1)),
        'neg_edges': tf.placeholder(tf.int32, (None, 2)),
        'training': tf.placeholder(tf.bool),
        'learning_rate': tf.placeholder(tf.float32),
        'circ_views': {v: tf.placeholder(tf.float32, (n_circ, n_circ)) for v in VIEW_NAMES},
        'dise_views': {v: tf.placeholder(tf.float32, (n_dis, n_dis)) for v in VIEW_NAMES},
    }
    model = MHDA_CLGAE(
        n_circ=n_circ, n_dis=n_dis,
        circ_views_adj=ph['circ_views'], dise_views_adj=ph['dise_views'],
        assoc_matrix=ph['assoc'], hidden_dim=config['hidden_dim'],
        dropout=config['dropout'], use_multi_view=config['use_multi_view'],
        view_fusion_mode=config['view_fusion_mode'],
        use_heterogeneous=config['use_heterogeneous'],
        use_dual_stream=config['use_dual_stream'],
        use_contrastive=config['use_contrastive'])
    logits, emb_circ, emb_dise = model(
        ph['features_circ'], ph['features_dise'], ph['edges'], training=ph['training'])
    smoothed = ph['labels'] * (1 - config['label_smoothing']) + config['label_smoothing'] * 0.5
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=smoothed, logits=logits))
    if config['use_contrastive']:
        pos_edges = tf.boolean_mask(ph['edges'], tf.squeeze(ph['labels'] > 0.5, axis=1))
        contrast = model.compute_contrastive_loss(emb_circ, emb_dise, pos_edges, ph['neg_edges'])
        loss = loss + config['lambda_contrast'] * contrast
    loss = loss + config['l2_weight'] * tf.add_n(
        [tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name])
    train_op = tf.train.AdamOptimizer(ph['learning_rate']).minimize(loss)
    return ph, logits, loss, train_op


def make_graph_feed(ph, inputs):
    """A single immutable set of graph inputs shared by train/val/test."""
    feed = {
        ph['features_circ']: inputs['features_circ'],
        ph['features_dise']: inputs['features_dise'],
        ph['assoc']: inputs['assoc'],
    }
    for v in VIEW_NAMES:
        feed[ph['circ_views'][v]] = inputs['circ_views_norm'][v]
        feed[ph['dise_views'][v]] = inputs['dise_views_norm'][v]
    return feed


def predict_logits(sess, logits, ph, graph_feed, edges, batch_size=512):
    predictions = []
    for start in range(0, len(edges), batch_size):
        feed = dict(graph_feed)
        feed.update({ph['edges']: edges[start:start + batch_size], ph['training']: False})
        predictions.append(sess.run(logits, feed_dict=feed))
    return np.vstack(predictions)


def _array_hash(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def save_fold_inputs(folder, fold, inputs, seed):
    folder.mkdir(parents=True, exist_ok=False)
    arrays = {k: fold[k] for k in ('train_pos', 'train_neg', 'val_pos', 'val_neg', 'test_pos', 'test_neg')}
    arrays.update(A_fit=inputs['assoc'], features_circ=inputs['features_circ'], features_dise=inputs['features_dise'])
    for side in ('circ', 'dise'):
        for name in VIEW_NAMES:
            arrays[side + '_' + name] = inputs[side + '_views'][name]
            arrays[side + '_' + name + '_norm'] = inputs[side + '_views_norm'][name]
    np.savez_compressed(str(folder / 'inputs_and_split.npz'), **arrays)
    audit = {
        'fold': int(fold['fold']), 'model_seed': int(seed),
        'diagnostics': inputs['diagnostics'],
        'array_sha256': {k: _array_hash(v) for k, v in arrays.items()},
        'held_out_positive_edges_in_graph': 0,
        'graph_source': 'actual training positives only',
        'semantic_prior_provenance': 'Supplied D_SH.mat; independent MeSH provenance must be documented.',
    }
    (folder / 'input_audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')


def train_mhda_clgae(model_config_name='full', data_dir=None, output_dir=None,
                     num_epochs=130, split_seed=1024, model_seed=42):
    if model_config_name not in MODEL_CONFIGS:
        raise ValueError('Unknown model configuration: ' + str(model_config_name))
    if num_epochs < 1:
        raise ValueError('num_epochs must be positive')
    config = dict(MODEL_CONFIGS[model_config_name])
    A_full, semantic = load_raw_data(data_dir)
    folds = make_cv_splits(A_full, n_splits=5, seed=split_seed, neg_ratio=1.0, val_fraction=0.2)
    n_circ, n_dis = A_full.shape
    # A_full is deliberately not passed into the encoder/preprocessing path.
    del A_full
    root = Path(__file__).resolve().parent.parent
    output = Path(output_dir) if output_dir is not None else root / 'results' / (
        model_config_name + '_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    output.mkdir(parents=True, exist_ok=False)
    run_config = {
        'model_config_name': model_config_name, 'model_config': config,
        'num_epochs': num_epochs, 'batch_size': 512, 'split_seed': split_seed,
        'base_model_seed': model_seed, 'validation_fraction': 0.2,
        'validation_every_epochs': 10, 'early_stopping_patience_evaluations': 15,
        'checkpoint_selection': 'maximum validation AUC',
        'threshold_selection': 'maximum validation F1 in original 0.30:0.01:0.70 grid',
        'integration_rule': 'where(prior > 0, prior, fold_GIP)',
        'no_test_labels_used_for_selection': True,
    }
    (output / 'run_config.json').write_text(json.dumps(run_config, indent=2), encoding='utf-8')
    fold_results, fold_plot_data = [], []
    for fold in folds:
        assert_fold_isolation(fold)
        inputs = build_fold_inputs(fold['A_fit'], semantic)
        fold_seed = model_seed + int(fold['fold']) - 1
        random.seed(fold_seed)
        rng = np.random.RandomState(fold_seed)
        folder = output / ('fold_%02d' % fold['fold'])
        save_fold_inputs(folder, fold, inputs, fold_seed)
        ph, logits, loss, train_op = build_model_graph(n_circ, n_dis, config, fold_seed)
        graph_feed = make_graph_feed(ph, inputs)
        train_edges, train_labels = pack_edges(fold['train_pos'], fold['train_neg'])
        val_edges, val_labels = pack_edges(fold['val_pos'], fold['val_neg'])
        test_edges, test_labels = pack_edges(fold['test_pos'], fold['test_neg'])
        # Keep training minibatch order fixed, as in the original implementation.
        order = rng.permutation(len(train_edges))
        train_edges, train_labels = train_edges[order], train_labels[order]
        best_auc, best_epoch, patience = -np.inf, 0, 0
        saver = tf.train.Saver(max_to_keep=1)
        checkpoint = str(folder / 'best_validation_model')
        session_config = tf.ConfigProto()
        session_config.gpu_options.allow_growth = True
        history = []
        print('Fold %d: train=%d, val=%d, test=%d, graph positives=%d' % (
            fold['fold'], len(train_edges), len(val_edges), len(test_edges), np.count_nonzero(inputs['assoc'])))
        with tf.Session(config=session_config) as sess:
            sess.run(tf.global_variables_initializer())
            for epoch in range(num_epochs):
                lr = config['learning_rate'] * (0.95 ** (epoch // 15))
                losses = []
                for start in range(0, len(train_edges), 512):
                    edges = train_edges[start:start + 512]
                    labels = train_labels[start:start + 512]
                    # Contrastive negatives are drawn only from assigned TRAIN negatives.
                    n_neg = max(1, len(edges) // 2)
                    neg_ids = rng.choice(len(fold['train_neg']), size=n_neg,
                                         replace=n_neg > len(fold['train_neg']))
                    feed = dict(graph_feed)
                    feed.update({ph['edges']: edges, ph['labels']: labels,
                                 ph['neg_edges']: fold['train_neg'][neg_ids],
                                 ph['training']: True, ph['learning_rate']: lr})
                    _, loss_value = sess.run([train_op, loss], feed_dict=feed)
                    if not np.isfinite(loss_value):
                        raise FloatingPointError('Non-finite training loss')
                    losses.append(float(loss_value))
                if (epoch + 1) % 10 == 0 or epoch + 1 == num_epochs:
                    val_logits = predict_logits(sess, logits, ph, graph_feed, val_edges)
                    metrics = OptimizerMetrics.compute_metrics(val_labels, val_logits)
                    history.append({'epoch': epoch + 1, 'train_loss': float(np.mean(losses)),
                                    'validation_auc': float(metrics['auc'])})
                    if metrics['auc'] > best_auc:
                        best_auc, best_epoch, patience = metrics['auc'], epoch + 1, 0
                        saver.save(sess, checkpoint)
                    else:
                        patience += 1
                    print('  epoch=%d loss=%.6f val_AUC=%.4f' % (
                        epoch + 1, np.mean(losses), metrics['auc']))
                    if patience >= 15:
                        break
            saver.restore(sess, checkpoint)
            val_logits = predict_logits(sess, logits, ph, graph_feed, val_edges)
            selected = OptimizerMetrics.compute_metrics_with_optimal_threshold(val_labels, val_logits)
            threshold = float(selected['optimal_threshold'])
            # The held-out test labels enter only the final metric computation.
            test_logits = predict_logits(sess, logits, ph, graph_feed, test_edges)
            metrics = OptimizerMetrics.compute_metrics(test_labels, test_logits, threshold=threshold)
        metrics.update(optimal_threshold=threshold, best_epoch=best_epoch, fold=int(fold['fold']))
        fold_results.append(metrics)
        scores = OptimizerMetrics.probabilities(test_logits)
        fold_plot_data.append({'fold': fold['fold'], 'y_true': test_labels.flatten(),
                               'y_scores': scores, 'auc': metrics['auc'], 'aupr': metrics['aupr']})
        np.savez_compressed(str(folder / 'predictions.npz'), val_edges=val_edges,
                            val_labels=val_labels, val_logits=val_logits,
                            test_edges=test_edges, test_labels=test_labels, test_logits=test_logits)
        (folder / 'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
        (folder / 'history.json').write_text(json.dumps(history, indent=2), encoding='utf-8')
    from Plot_ROC_PR import ROCPRPlotter
    plot_path = ROCPRPlotter(save_dir=str(output / 'figs')).plot_5fold_curves(
        fold_plot_data, model_name=model_config_name, show_mean=True)
    summary = {'model_config': model_config_name, 'fold_results': fold_results,
               'metrics': {}, 'plot_path': str(plot_path), 'output_dir': str(output)}
    for metric in ('auc', 'aupr', 'f1', 'precision', 'recall', 'accuracy'):
        values = [item[metric] for item in fold_results]
        summary['metrics'][metric + '_mean'] = float(np.mean(values))
        summary['metrics'][metric + '_std'] = float(np.std(values))
    (output / ('results_' + model_config_name + '.json')).write_text(
        json.dumps(summary, indent=2), encoding='utf-8')
    print('Results saved to:', output)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', nargs='?', default='full', choices=sorted(MODEL_CONFIGS))
    parser.add_argument('--data-dir', default=None)
    parser.add_argument('--output-dir', default=None)
    parser.add_argument('--epochs', type=int, default=130)
    parser.add_argument('--split-seed', type=int, default=1024)
    parser.add_argument('--model-seed', type=int, default=42)
    args = parser.parse_args()
    train_mhda_clgae(args.config, data_dir=args.data_dir, output_dir=args.output_dir,
                     num_epochs=args.epochs, split_seed=args.split_seed, model_seed=args.model_seed)
