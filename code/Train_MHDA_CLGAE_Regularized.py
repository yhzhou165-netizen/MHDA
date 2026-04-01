import random
import numpy as np
import tensorflow.compat.v1 as tf
import os

random.seed(42)
np.random.seed(42)
tf.set_random_seed(42)
os.environ['PYTHONHASHSEED'] = '42'

import tensorflow.compat.v1 as tf
import numpy as np
from Model_MHDA_CLGAE import MHDA_CLGAE
from Adj_preprocess import adjacency_circRNA_disease_multi_view
from Preprocessing import split_train_test, preprocess_graph
from Optimizer import OptimizerMetrics
from Plot_ROC_PR import ROCPRPlotter
import scipy.sparse as sp
import time
import sys
import json

tf.disable_v2_behavior()


def train_mhda_clgae(model_config_name='full'):
    print("=" * 80)
    print("Training MHDA-CLGAE Model")
    print(f"Config: {model_config_name}")
    print("=" * 80)

    print("\n[Step 1/8] Loading data...")
    circ_views, dise_views, assoc_matrix = adjacency_circRNA_disease_multi_view()

    n_circ, n_dis = assoc_matrix.shape
    print(f"\n[Data dimensions]")
    print(f"  circRNA: {n_circ}, disease: {n_dis}")
    print(f"  Positive samples: {np.sum(assoc_matrix > 0)}")
    print(f"  Negative samples: {np.sum(assoc_matrix == 0)}")

    print("\n[Step 2/8] Preprocessing adjacency matrices...")
    circ_views_norm = {}
    for view_name, view_adj in circ_views.items():
        if not sp.issparse(view_adj):
            view_adj = sp.csr_matrix(view_adj)
        view_adj_norm = preprocess_graph(view_adj)
        circ_views_norm[view_name] = view_adj_norm.toarray().astype(np.float32)

    dise_views_norm = {}
    for view_name, view_adj in dise_views.items():
        if not sp.issparse(view_adj):
            view_adj = sp.csr_matrix(view_adj)
        view_adj_norm = preprocess_graph(view_adj)
        dise_views_norm[view_name] = view_adj_norm.toarray().astype(np.float32)

    print("\n[Step 3/8] Initializing node features...")
    features_circ_init = circ_views['jaccard'].astype(np.float32)
    features_dise_init = dise_views['jaccard'].astype(np.float32)

    print("\n[Step 4/8] Splitting train/test sets...")
    train_e_p_, train_e_n_, test_e_p_, test_e_n_, adj_dm_new_ = split_train_test(
        assoc_matrix, n_splits=5, seed=1024, neg_ratio=1.0
    )

    print("\n[Step 5/8] Configuring model...")
    config_dict = {
        'baseline': {
            'use_multi_view': False,
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

    model_config = config_dict.get(model_config_name, config_dict['full'])

    print(f"\n  Model configuration:")
    print(f"    Multi-view learning: {model_config['use_multi_view']}")
    print(f"    Heterogeneous attention: {model_config['use_heterogeneous']}")
    print(f"    Dual-stream decoder: {model_config['use_dual_stream']}")
    print(f"    Contrastive learning: {model_config['use_contrastive']}")
    print(f"    Hidden dim: {model_config['hidden_dim']}")
    print(f"    Dropout: {model_config['dropout']}")

    print("\n[Step 6/8] Building TensorFlow graph...")
    tf.reset_default_graph()

    ph_features_circ = tf.placeholder(tf.float32, shape=(n_circ, n_circ))
    ph_features_dise = tf.placeholder(tf.float32, shape=(n_dis, n_dis))

    ph_adj_circ = {}
    for view_name in circ_views_norm.keys():
        ph_adj_circ[view_name] = tf.placeholder(tf.float32, shape=(n_circ, n_circ))

    ph_adj_dise = {}
    for view_name in dise_views_norm.keys():
        ph_adj_dise[view_name] = tf.placeholder(tf.float32, shape=(n_dis, n_dis))

    ph_assoc = tf.placeholder(tf.float32, shape=(n_circ, n_dis))
    ph_edges = tf.placeholder(tf.int32, shape=(None, 2))
    ph_labels = tf.placeholder(tf.float32, shape=(None, 1))
    ph_training = tf.placeholder(tf.bool)
    ph_neg_edges = tf.placeholder(tf.int32, shape=(None, 2))
    ph_learning_rate = tf.placeholder(tf.float32)

    model = MHDA_CLGAE(
        n_circ=n_circ,
        n_dis=n_dis,
        circ_views_adj=ph_adj_circ,
        dise_views_adj=ph_adj_dise,
        assoc_matrix=ph_assoc,
        hidden_dim=model_config['hidden_dim'],
        dropout=model_config['dropout'],
        use_multi_view=model_config['use_multi_view'],
        use_heterogeneous=model_config['use_heterogeneous'],
        use_dual_stream=model_config['use_dual_stream'],
        use_contrastive=model_config['use_contrastive']
    )

    logits, emb_circ, emb_dise = model(
        ph_features_circ, ph_features_dise, ph_edges, training=ph_training
    )

    n_params = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])
    print(f"  Trainable parameters: {n_params:.0f}")

    print("\n[Step 7/8] Defining loss function...")

    label_smoothing = model_config['label_smoothing']
    smoothed_labels = ph_labels * (1 - label_smoothing) + label_smoothing * 0.5

    bce_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(labels=smoothed_labels, logits=logits)
    )

    if model_config['use_contrastive']:
        pos_mask = tf.squeeze(ph_labels > 0.5, axis=1)
        pos_edges = tf.boolean_mask(ph_edges, pos_mask)
        contrastive_loss = model.compute_contrastive_loss(
            emb_circ, emb_dise, pos_edges, ph_neg_edges
        )
        total_loss = bce_loss + model_config['lambda_contrast'] * contrastive_loss
    else:
        contrastive_loss = tf.constant(0.0)
        total_loss = bce_loss

    l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()
                        if 'bias' not in v.name]) * model_config['l2_weight']
    total_loss = total_loss + l2_loss

    optimizer = tf.train.AdamOptimizer(learning_rate=ph_learning_rate)
    train_op = optimizer.minimize(total_loss)

    print(f"  Loss components: BCE + L2 regularization" +
          (f" + Contrastive learning" if model_config['use_contrastive'] else ""))

    print("\n[Step 8/8] Starting training...")

    num_epochs = 130
    batch_size = 512
    early_stopping_patience = 15

    config_tf = tf.ConfigProto()
    config_tf.gpu_options.allow_growth = True

    fold_results = []
    fold_plot_data = []

    with tf.Session(config=config_tf) as sess:
        sess.run(tf.global_variables_initializer())

        for fold in range(5):
            print(f"\n{'=' * 80}")
            print(f"Fold {fold + 1}/5")
            print(f"{'=' * 80}")

            train_pos = train_e_p_[fold]
            train_neg = train_e_n_[fold]
            test_pos = test_e_p_[fold]
            test_neg = test_e_n_[fold]

            train_edges = np.vstack([train_pos, train_neg])
            train_labels = np.vstack([
                np.ones((len(train_pos), 1)),
                np.zeros((len(train_neg), 1))
            ])

            shuffle_idx = np.random.permutation(len(train_edges))
            train_edges = train_edges[shuffle_idx]
            train_labels = train_labels[shuffle_idx]

            n_train = len(train_edges)
            split_idx = int(n_train * 0.8)

            actual_train_edges = train_edges[:split_idx]
            actual_train_labels = train_labels[:split_idx]

            val_edges = train_edges[split_idx:]
            val_labels = train_labels[split_idx:]

            print(f"  Train samples: {len(actual_train_edges)}")
            print(f"  Val samples: {len(val_edges)}")
            print(f"  Test samples: {len(test_pos) + len(test_neg)} (pos:{len(test_pos)}, neg:{len(test_neg)})")

            best_val_auc = 0.0
            patience_counter = 0
            best_epoch = 0

            initial_lr = model_config['learning_rate']

            for epoch in range(num_epochs):
                current_lr = initial_lr * (0.95 ** (epoch // 15))

                epoch_loss = []

                for i in range(0, len(actual_train_edges), batch_size):
                    batch_edges = actual_train_edges[i:i + batch_size]
                    batch_labels = actual_train_labels[i:i + batch_size]

                    batch_size_actual = len(batch_edges)
                    neg_sample_size = batch_size_actual // 2

                    neg_circ_idx = np.random.randint(0, n_circ, neg_sample_size)
                    neg_dise_idx = np.random.randint(0, n_dis, neg_sample_size)
                    batch_neg_edges = np.column_stack([neg_circ_idx, neg_dise_idx])

                    feed_dict = {
                        ph_features_circ: features_circ_init,
                        ph_features_dise: features_dise_init,
                        ph_assoc: assoc_matrix,
                        ph_edges: batch_edges,
                        ph_labels: batch_labels,
                        ph_neg_edges: batch_neg_edges,
                        ph_training: True,
                        ph_learning_rate: current_lr
                    }

                    for view_name, view_adj in circ_views_norm.items():
                        feed_dict[ph_adj_circ[view_name]] = view_adj
                    for view_name, view_adj in dise_views_norm.items():
                        feed_dict[ph_adj_dise[view_name]] = view_adj

                    _, loss_val = sess.run([train_op, total_loss], feed_dict=feed_dict)
                    epoch_loss.append(loss_val)

                avg_loss = np.mean(epoch_loss)

                if (epoch + 1) % 10 == 0:
                    feed_dict_val = {
                        ph_features_circ: features_circ_init,
                        ph_features_dise: features_dise_init,
                        ph_assoc: assoc_matrix,
                        ph_edges: val_edges,
                        ph_labels: val_labels,
                        ph_training: False
                    }

                    for view_name, view_adj in circ_views_norm.items():
                        feed_dict_val[ph_adj_circ[view_name]] = view_adj
                    for view_name, view_adj in dise_views_norm.items():
                        feed_dict_val[ph_adj_dise[view_name]] = view_adj

                    val_logits = sess.run(logits, feed_dict=feed_dict_val)
                    val_metrics = OptimizerMetrics.compute_metrics(val_labels, val_logits)

                    print(f"  Epoch {epoch + 1:03d} | Loss: {avg_loss:.4f} | Val AUC: {val_metrics['auc']:.2f}%")

                    if val_metrics['auc'] > best_val_auc:
                        best_val_auc = val_metrics['auc']
                        patience_counter = 0
                        best_epoch = epoch + 1
                    else:
                        patience_counter += 1

                    if patience_counter >= early_stopping_patience:
                        print(f"\n  Early stopping at epoch {epoch + 1}")
                        break

            print(f"\n  [Test set evaluation]")
            test_edges = np.vstack([test_pos, test_neg])
            test_labels = np.vstack([
                np.ones((len(test_pos), 1)),
                np.zeros((len(test_neg), 1))
            ])

            test_logits_list = []
            for i in range(0, len(test_edges), batch_size):
                batch_edges = test_edges[i:i + batch_size]

                feed_dict_test = {
                    ph_features_circ: features_circ_init,
                    ph_features_dise: features_dise_init,
                    ph_assoc: assoc_matrix,
                    ph_edges: batch_edges,
                    ph_training: False
                }

                for view_name, view_adj in circ_views_norm.items():
                    feed_dict_test[ph_adj_circ[view_name]] = view_adj
                for view_name, view_adj in dise_views_norm.items():
                    feed_dict_test[ph_adj_dise[view_name]] = view_adj

                batch_logits = sess.run(logits, feed_dict=feed_dict_test)
                test_logits_list.append(batch_logits)

            test_logits = np.vstack(test_logits_list)

            test_metrics = OptimizerMetrics.compute_metrics_with_optimal_threshold(
                test_labels, test_logits
            )

            print(f"\n  Fold {fold + 1} Test Results:")
            print(f"    AUC:       {test_metrics['auc']:.2f}%")
            print(f"    AUPR:      {test_metrics['aupr']:.2f}%")
            print(f"    F1:        {test_metrics['f1']:.2f}%")
            print(f"    Precision: {test_metrics['precision']:.2f}%")
            print(f"    Recall:    {test_metrics['recall']:.2f}%")
            print(f"    Accuracy:  {test_metrics['accuracy']:.2f}%")

            fold_results.append(test_metrics)

            y_scores = 1.0 / (1.0 + np.exp(-test_logits.flatten()))
            fold_plot_data.append({
                'y_true': test_labels.flatten(),
                'y_scores': y_scores,
                'auc': test_metrics['auc'],
                'aupr': test_metrics['aupr'],
                'fold': fold + 1
            })

            if fold < 4:
                sess.run(tf.global_variables_initializer())

    print(f"\n{'=' * 80}")
    print(f"5-Fold Cross Validation Final Results - {model_config_name}")
    print(f"{'=' * 80}")

    avg_auc = np.mean([r['auc'] for r in fold_results])
    avg_aupr = np.mean([r['aupr'] for r in fold_results])
    avg_f1 = np.mean([r['f1'] for r in fold_results])
    avg_precision = np.mean([r['precision'] for r in fold_results])
    avg_recall = np.mean([r['recall'] for r in fold_results])
    avg_accuracy = np.mean([r['accuracy'] for r in fold_results])

    std_auc = np.std([r['auc'] for r in fold_results])
    std_aupr = np.std([r['aupr'] for r in fold_results])
    std_f1 = np.std([r['f1'] for r in fold_results])
    std_precision = np.std([r['precision'] for r in fold_results])
    std_recall = np.std([r['recall'] for r in fold_results])
    std_accuracy = np.std([r['accuracy'] for r in fold_results])

    print(f"\nAverage Performance (Mean +/- Std):")
    print(f"  AUC:       {avg_auc:.2f}% +/- {std_auc:.2f}%")
    print(f"  AUPR:      {avg_aupr:.2f}% +/- {std_aupr:.2f}%")
    print(f"  F1:        {avg_f1:.2f}% +/- {std_f1:.2f}%")
    print(f"  Precision: {avg_precision:.2f}% +/- {std_precision:.2f}%")
    print(f"  Recall:    {avg_recall:.2f}% +/- {std_recall:.2f}%")
    print(f"  Accuracy:  {avg_accuracy:.2f}% +/- {std_accuracy:.2f}%")

    print(f"\n{'=' * 80}")
    print("Plotting ROC/PR curves (with 5-fold mean curves)")
    print(f"{'=' * 80}")

    plotter = ROCPRPlotter(save_dir='figs')
    plot_path = plotter.plot_5fold_curves(
        fold_plot_data,
        model_name=model_config_name,
        show_mean=True
    )

    results_summary = {
        'model_config': model_config_name,
        'metrics': {
            'auc_mean': float(avg_auc),
            'auc_std': float(std_auc),
            'aupr_mean': float(avg_aupr),
            'aupr_std': float(std_aupr),
            'f1_mean': float(avg_f1),
            'f1_std': float(std_f1),
            'precision_mean': float(avg_precision),
            'precision_std': float(std_precision),
            'recall_mean': float(avg_recall),
            'recall_std': float(std_recall),
            'accuracy_mean': float(avg_accuracy),
            'accuracy_std': float(std_accuracy)
        },
        'fold_results': [
            {k: float(v) for k, v in r.items() if k != 'optimal_threshold'}
            for r in fold_results
        ],
        'plot_path': plot_path
    }

    output_file = f'results_{model_config_name}.json'
    with open(output_file, 'w') as f:
        json.dump(results_summary, f, indent=2)

    print(f"\nResults saved to: {output_file}")
    print(f"Figure saved to: {plot_path}")
    print(f"\n{'=' * 80}\n")

    return fold_results


if __name__ == '__main__':
    if len(sys.argv) > 1:
        config_name = sys.argv[1]
    else:
        config_name = 'full'

    valid_configs = ['baseline', 'full']

    if config_name not in valid_configs:
        print(f"Error: invalid config '{config_name}'")
        print(f"Valid configs: {valid_configs}")
        print(f"\nUsage:")
        print(f"  python Train_MHDA_CLGAE.py full      # train full model (recommended)")
        print(f"  python Train_MHDA_CLGAE.py baseline  # train baseline model")
        sys.exit(1)

    print(f"\nUsing config: {config_name}")
    start_time = time.time()

    results = train_mhda_clgae(model_config_name=config_name)

    elapsed = time.time() - start_time
    print(f"Total training time: {elapsed:.2f}s ({elapsed / 60:.2f}min)")