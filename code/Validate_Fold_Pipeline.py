"""Verify all fold inputs without fitting the neural network."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import KFold
from Adj_preprocess import load_raw_data, build_fold_inputs
from Preprocessing import make_cv_splits, assert_fold_isolation


def digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def validate_pipeline(data_dir=None):
    A, semantic = load_raw_data(data_dir)
    before_A, before_semantic = A.copy(), semantic.copy()
    folds = make_cv_splits(A, seed=1024)
    repeated = make_cv_splits(A, seed=1024)
    positions = np.argwhere(A > 0)
    expected = list(KFold(5, shuffle=True, random_state=1024).split(positions))
    report = {'status': 'passed', 'training_performed': False,
              'shape': list(A.shape), 'known_positive_count': int(np.count_nonzero(A)),
              'folds': [], 'semantic_prior_status':
              'Supplied D_SH.mat used as a fixed prior; independent MeSH provenance not verified.'}
    all_test_pos = []
    previous_inputs = []
    for fold, again, (_, reference_test) in zip(folds, repeated, expected):
        assert_fold_isolation(fold)
        for key in ('train_pos', 'train_neg', 'val_pos', 'val_neg', 'test_pos', 'test_neg', 'A_fit'):
            np.testing.assert_array_equal(fold[key], again[key])
        np.testing.assert_array_equal(fold['test_pos'], positions[reference_test])
        for key in ('train_pos', 'val_pos', 'test_pos'):
            edges = fold[key]
            assert np.all(A[edges[:, 0], edges[:, 1]] == 1)
        for key in ('train_neg', 'val_neg', 'test_neg'):
            edges = fold[key]
            assert np.all(A[edges[:, 0], edges[:, 1]] == 0)
        all_test_pos.extend(map(tuple, fold['test_pos'].tolist()))
        inputs = build_fold_inputs(fold['A_fit'], semantic)
        np.testing.assert_array_equal(inputs['assoc'], fold['A_fit'])
        np.testing.assert_array_equal(inputs['features_circ'], inputs['circ_views']['jaccard'])
        np.testing.assert_array_equal(inputs['features_dise'], inputs['dise_views']['jaccard'])
        hashes = {'assoc': digest(inputs['assoc'])}
        for side, n in (('circ', A.shape[0]), ('dise', A.shape[1])):
            for group in (side + '_views', side + '_views_norm'):
                for name, matrix in inputs[group].items():
                    assert matrix.shape == (n, n)
                    assert np.all(np.isfinite(matrix)) and np.all(matrix >= 0)
                    np.testing.assert_allclose(matrix, matrix.T, atol=1e-6)
                    hashes[group + '/' + name] = digest(matrix)
        # Rebuilding the same fold cannot depend on mutable state or a prior fold.
        repeat_inputs = build_fold_inputs(fold['A_fit'].copy(), semantic.copy())
        for side in ('circ', 'dise'):
            for view in ('jaccard', 'cosine', 'integrated'):
                np.testing.assert_array_equal(inputs[side + '_views'][view], repeat_inputs[side + '_views'][view])
        # A feature builder has no input channel for held-out labels.
        expected_fit = np.zeros_like(A)
        edges = fold['train_pos']
        expected_fit[edges[:, 0], edges[:, 1]] = 1
        np.testing.assert_array_equal(inputs['assoc'], expected_fit)
        entry = {'fold': int(fold['fold']),
                 'counts': {k: int(len(fold[k])) for k in
                            ('train_pos', 'train_neg', 'val_pos', 'val_neg', 'test_pos', 'test_neg')},
                 'held_out_positive_edges_in_graph': 0,
                 'diagnostics': inputs['diagnostics'], 'sha256': hashes}
        report['folds'].append(entry)
        previous_inputs.append(inputs['assoc'].copy())
    assert len(all_test_pos) == len(set(all_test_pos)) == len(positions)
    assert set(all_test_pos) == set(map(tuple, positions.tolist()))
    assert len({digest(a) for a in previous_inputs}) == 5
    np.testing.assert_array_equal(A, before_A)
    np.testing.assert_array_equal(semantic, before_semantic)
    report['checks'] = ['original KFold positive partitions retained', 'deterministic split and construction',
                        'train/validation/test pairs disjoint within each fold',
                        'all graph positives belong to actual training partition',
                        'all held-out positives absent from graph',
                        'all nodes and original matrix dimensions retained',
                        'finite symmetric similarity and normalized adjacency matrices',
                        'node features derive from fold-specific Jaccard views',
                        'different folds have different training graphs',
                        'input files/matrices not mutated']
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', default=None)
    parser.add_argument('--output', default=str(Path(__file__).resolve().parent.parent / 'checks' / 'pipeline_audit.json'))
    args = parser.parse_args()
    report = validate_pipeline(args.data_dir)
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print('PASS: 5-fold input isolation checks; no model training performed.')
    for fold in report['folds']:
        print('Fold %d: %s' % (fold['fold'], fold['counts']))
    print('Report:', target)
