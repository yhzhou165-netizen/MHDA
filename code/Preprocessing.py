"""Deterministic splits and graph normalization, without model dependencies.

Unknown entries are sampled as putative negatives, not biological negatives.
Within each fold the three sample partitions are disjoint; different folds may
reuse training data, as is usual in cross-validation.
"""

import numpy as np
import scipy.sparse as sp
from sklearn.model_selection import KFold, train_test_split


def validate_association_matrix(matrix, name="association matrix"):
    """Validate a finite binary matrix without modifying the caller's input."""
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2 or 0 in matrix.shape:
        raise ValueError(f"{name} must be a nonempty two-dimensional matrix.")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} contains non-finite values.")
    if not np.logical_or(matrix == 0.0, matrix == 1.0).all():
        raise ValueError(f"{name} must contain only binary values 0 and 1.")
    return matrix


def sparse_to_tuple(sparse_mx):
    sparse_mx = sp.coo_matrix(sparse_mx)
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).T
    return coords, sparse_mx.data, sparse_mx.shape


def preprocess_graph(adj):
    """Symmetric normalization D^-1/2 (adj + I) D^-1/2."""
    adj = sp.csr_matrix(adj, dtype=np.float64)
    if adj.shape[0] != adj.shape[1]:
        raise ValueError("Graph adjacency must be square.")
    if not np.isfinite(adj.data).all() or np.any(adj.data < 0):
        raise ValueError("Graph adjacency must be finite and nonnegative.")
    adj = adj + sp.eye(adj.shape[0], dtype=adj.dtype, format="csr")
    rowsum = np.asarray(adj.sum(axis=1)).ravel()
    d_inv_sqrt = np.zeros_like(rowsum)
    np.power(rowsum, -0.5, out=d_inv_sqrt, where=rowsum > 0)
    diagonal = sp.diags(d_inv_sqrt)
    return (diagonal @ adj @ diagonal).tocsr()


def _validated_edges(edges, shape, name):
    edges = np.asarray(edges)
    if edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError(f"{name} must have shape (n, 2).")
    if not np.issubdtype(edges.dtype, np.integer):
        raise ValueError(f"{name} must contain integer node indices.")
    if np.any(edges < 0) or np.any(edges[:, 0] >= shape[0]) or np.any(edges[:, 1] >= shape[1]):
        raise ValueError(f"{name} contains node indices outside the matrix.")
    flat = edges[:, 0] * shape[1] + edges[:, 1]
    if len(np.unique(flat)) != len(flat):
        raise ValueError(f"{name} contains duplicate pairs.")
    return edges.astype(np.int64, copy=False), flat


def assert_fold_isolation(fold):
    """Raise if pair partitions overlap or graph edges differ from train_pos."""
    A_fit = validate_association_matrix(fold["A_fit"], "A_fit")
    names = ("train_pos", "train_neg", "val_pos", "val_neg", "test_pos", "test_neg")
    encoded = {}
    edges = {}
    for name in names:
        edges[name], encoded[name] = _validated_edges(fold[name], A_fit.shape, name)
    if len(edges["train_pos"]) == 0:
        raise ValueError("A fold must contain at least one training positive association.")
    for index, left in enumerate(names):
        for right in names[index + 1:]:
            if np.intersect1d(encoded[left], encoded[right]).size:
                raise AssertionError(f"Sample overlap between {left} and {right}.")
    expected = np.zeros_like(A_fit)
    train_pos = edges["train_pos"]
    expected[train_pos[:, 0], train_pos[:, 1]] = 1.0
    if not np.array_equal(A_fit, expected):
        raise AssertionError("A_fit must contain exactly the training positive edges.")
    for name in names[1:]:
        pairs = edges[name]
        if pairs.size and np.any(A_fit[pairs[:, 0], pairs[:, 1]] != 0):
            raise AssertionError(f"{name} includes an edge visible in A_fit.")
    return True


def make_cv_splits(A_full, n_splits=5, seed=1024, neg_ratio=1.0, val_fraction=0.2):
    """Split labels first, then build each graph from training positives only.

    Outer positive folds reproduce KFold(shuffle=True, random_state=seed).
    One without-replacement negative sample is partitioned into outer-training
    and test pools per fold. The former pool is split jointly with positive
    pairs using stratification, retaining both classes in train and validation.
    neg_ratio applies to both outer-training and test samples (rounded to the
    nearest integer); integer allocation may slightly change inner ratios.
    """
    A_full = validate_association_matrix(A_full, "A_full")
    if not np.isfinite(neg_ratio) or neg_ratio <= 0:
        raise ValueError("neg_ratio must be finite and positive.")
    if not np.isfinite(val_fraction) or not 0 < val_fraction < 1:
        raise ValueError("val_fraction must lie strictly between 0 and 1.")
    positives = np.argwhere(A_full == 1).astype(np.int64)
    unknowns = np.argwhere(A_full == 0).astype(np.int64)
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    rng = np.random.RandomState(seed)
    folds = []
    for fold_number, (outer_train_ids, test_ids) in enumerate(kfold.split(positives), start=1):
        outer_pos = positives[outer_train_ids]
        test_pos = positives[test_ids]
        n_outer_neg = int(np.round(len(outer_pos) * float(neg_ratio)))
        n_test_neg = int(np.round(len(test_pos) * float(neg_ratio)))
        if n_outer_neg < 2 or n_test_neg < 1:
            raise ValueError("Too few sampled negatives for train/validation/test partitions.")
        n_requested = n_outer_neg + n_test_neg
        if n_requested > len(unknowns):
            raise ValueError("Insufficient unknown pairs for disjoint negative sampling.")
        sampled = unknowns[rng.choice(len(unknowns), size=n_requested, replace=False)]
        outer_neg, test_neg = sampled[:n_outer_neg], sampled[n_outer_neg:]
        outer_pairs = np.vstack((outer_pos, outer_neg))
        outer_labels = np.concatenate((np.ones(len(outer_pos), dtype=np.int8),
                                       np.zeros(len(outer_neg), dtype=np.int8)))
        try:
            train_ids, val_ids = train_test_split(
                np.arange(len(outer_pairs)), test_size=val_fraction,
                random_state=int(seed) + fold_number, stratify=outer_labels,
            )
        except ValueError as error:
            raise ValueError(f"Fold {fold_number} cannot form a stratified validation set: {error}") from error
        train_pairs, val_pairs = outer_pairs[train_ids], outer_pairs[val_ids]
        train_labels, val_labels = outer_labels[train_ids], outer_labels[val_ids]
        train_pos = train_pairs[train_labels == 1]
        train_neg = train_pairs[train_labels == 0]
        val_pos = val_pairs[val_labels == 1]
        val_neg = val_pairs[val_labels == 0]
        if min(len(train_pos), len(train_neg), len(val_pos), len(val_neg)) == 0:
            raise ValueError(f"Fold {fold_number} lacks a class in training or validation.")
        A_fit = np.zeros_like(A_full, dtype=np.float64)
        A_fit[train_pos[:, 0], train_pos[:, 1]] = 1.0
        fold = {
            "fold": fold_number, "train_pos": train_pos, "train_neg": train_neg,
            "val_pos": val_pos, "val_neg": val_neg, "test_pos": test_pos,
            "test_neg": test_neg, "A_fit": A_fit,
        }
        assert_fold_isolation(fold)
        all_positive_pairs = np.vstack((train_pos, val_pos, test_pos))
        if len(all_positive_pairs) != len(positives) or not np.all(
            A_full[all_positive_pairs[:, 0], all_positive_pairs[:, 1]] == 1
        ):
            raise AssertionError("Positive partitions must cover all known associations exactly once.")
        all_negative_pairs = np.vstack((train_neg, val_neg, test_neg))
        if not np.all(A_full[all_negative_pairs[:, 0], all_negative_pairs[:, 1]] == 0):
            raise AssertionError("Sampled negatives must come only from unknown pairs.")
        folds.append(fold)
    return folds
