"""Build every label-derived model input from an explicitly supplied A_fit.

Only the raw association table and externally supplied disease semantic prior
are loaded from disk. Full-data GIP/CFS/integrated similarity caches are never
loaded. This module has no TensorFlow dependency and does not write files.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio

from Preprocessing import preprocess_graph, validate_association_matrix


def _validate_disease_semantic(disease_semantic, n_diseases):
    semantic = np.asarray(disease_semantic, dtype=np.float64)
    if semantic.shape != (n_diseases, n_diseases):
        raise ValueError("Disease semantic similarity shape does not match disease nodes.")
    if not np.isfinite(semantic).all():
        raise ValueError("Disease semantic similarity contains non-finite values.")
    if np.any(semantic < 0) or np.any(semantic > 1):
        raise ValueError("Disease semantic similarities must lie in [0, 1].")
    if not np.allclose(semantic, semantic.T, atol=1e-8, rtol=1e-8):
        raise ValueError("Disease semantic similarity must be symmetric.")
    return semantic


def load_raw_data(data_dir=None):
    """Load A_full and the fixed semantic prior, without constructing views.

    Semantic similarity is treated as an external ontology-derived prior.
    Its upstream provenance must be independent of held-out association labels;
    this loader cannot establish that provenance from numeric values alone.
    """
    data_dir = Path(data_dir) if data_dir is not None else Path(__file__).resolve().parent.parent / "data"
    frame = pd.read_excel(data_dir / "Association Matrixs.xlsx", sheet_name=2, header=None)
    A_full = validate_association_matrix(frame.to_numpy(dtype=np.float64), "A_full")
    loaded = sio.loadmat(str(data_dir / "D_SH.mat"))
    key = "MeSHSemanticSimilarity"
    if key not in loaded:
        raise KeyError(f"D_SH.mat must contain {key!r}.")
    disease_semantic = _validate_disease_semantic(loaded[key], A_full.shape[1])
    return A_full.copy(), disease_semantic.copy()


def jaccard_rows_from_assoc(assoc_matrix):
    """Original binary Jaccard rule: two empty profiles have similarity zero."""
    matrix = validate_association_matrix(assoc_matrix)
    intersection = matrix @ matrix.T
    degree = matrix.sum(axis=1)
    union = degree[:, None] + degree[None, :] - intersection
    result = np.zeros_like(intersection)
    np.divide(intersection, union, out=result, where=union > 0)
    return result


def compute_cosine_similarity(matrix):
    """Retain the original cosine convention, including unit diagonal."""
    matrix = validate_association_matrix(matrix)
    norm = np.linalg.norm(matrix, axis=1, keepdims=True)
    normalized = np.zeros_like(matrix)
    np.divide(matrix, norm, out=normalized, where=norm > 0)
    similarity = normalized @ normalized.T
    np.fill_diagonal(similarity, 1.0)
    return np.clip(similarity, 0.0, 1.0)


def compute_gip_similarity(profiles, gamma0=1.0):
    """Return GIP and gamma fitted to supplied training interaction profiles.

    gamma = gamma0 / mean_i ||profile_i||^2, with no full-data statistics.
    """
    profiles = validate_association_matrix(profiles, "GIP profiles")
    if not np.isfinite(gamma0) or gamma0 <= 0:
        raise ValueError("gamma0 must be finite and positive.")
    squared_norm = np.sum(profiles * profiles, axis=1)
    mean_squared_norm = float(squared_norm.mean())
    if mean_squared_norm <= 0:
        raise ValueError("Cannot fit GIP bandwidth without training positive associations.")
    gamma = float(gamma0 / mean_squared_norm)
    squared_distance = squared_norm[:, None] + squared_norm[None, :] - 2.0 * profiles @ profiles.T
    similarity = np.exp(-gamma * np.maximum(squared_distance, 0.0))
    np.fill_diagonal(similarity, 1.0)
    return similarity, gamma


def compute_functional_similarity(A_fit, disease_semantic):
    """CFS from training-associated disease sets and a fixed semantic prior.

    For two nonempty disease sets, sum both directed best semantic matches and
    divide by their combined cardinality. If either set is empty, return zero.
    """
    matrix = validate_association_matrix(A_fit, "A_fit")
    semantic = _validate_disease_semantic(disease_semantic, matrix.shape[1])
    degree = matrix.sum(axis=1)
    best_match_to_set = np.zeros_like(matrix, dtype=np.float64)
    for row_index, row in enumerate(matrix):
        diseases = np.flatnonzero(row)
        if diseases.size:
            best_match_to_set[row_index] = semantic[:, diseases].max(axis=1)
    directed_sum = matrix @ best_match_to_set.T
    denominator = degree[:, None] + degree[None, :]
    nonempty_pair = (degree[:, None] > 0) & (degree[None, :] > 0)
    similarity = np.zeros_like(directed_sum)
    np.divide(directed_sum + directed_sum.T, denominator, out=similarity, where=nonempty_pair)
    return np.clip(similarity, 0.0, 1.0)


def build_fold_inputs(A_fit, disease_semantic):
    """Construct graph, features and normalized views exclusively from A_fit.

    A_fit must contain training positive edges only. Call assert_fold_isolation
    on the corresponding split before invoking this function. No full matrix,
    validation/test pairs, filesystem paths or global caches enter this builder.
    Integration retains the existing implementation: use a positive functional
    or semantic prior where available, otherwise use the fold-specific GIP.
    """
    matrix = validate_association_matrix(A_fit, "A_fit").copy()
    if not np.any(matrix):
        raise ValueError("A_fit must contain training positive associations.")
    semantic = _validate_disease_semantic(disease_semantic, matrix.shape[1]).copy()
    circ_gip, circ_gamma = compute_gip_similarity(matrix)
    dise_gip, dise_gamma = compute_gip_similarity(matrix.T)
    functional = compute_functional_similarity(matrix, semantic)
    circ_views = {
        "jaccard": jaccard_rows_from_assoc(matrix),
        "cosine": compute_cosine_similarity(matrix),
        "integrated": np.where(functional > 0.0, functional, circ_gip),
    }
    dise_views = {
        "jaccard": jaccard_rows_from_assoc(matrix.T),
        "cosine": compute_cosine_similarity(matrix.T),
        "integrated": np.where(semantic > 0.0, semantic, dise_gip),
    }
    for side, views in (("circRNA", circ_views), ("disease", dise_views)):
        for view_name, view in views.items():
            if not np.isfinite(view).all() or np.any(view < 0.0) or np.any(view > 1.0):
                raise AssertionError(f"Invalid {side} {view_name} similarity.")
    circ_views_norm = {name: preprocess_graph(view).toarray().astype(np.float32)
                       for name, view in circ_views.items()}
    dise_views_norm = {name: preprocess_graph(view).toarray().astype(np.float32)
                       for name, view in dise_views.items()}
    circ_views = {name: view.astype(np.float32) for name, view in circ_views.items()}
    dise_views = {name: view.astype(np.float32) for name, view in dise_views.items()}
    diagnostics = {
        "training_positive_count": int(matrix.sum()),
        "circ_gip_gamma": circ_gamma, "dise_gip_gamma": dise_gamma,
        "circ_nodes_without_training_edges": int(np.sum(matrix.sum(axis=1) == 0)),
        "dise_nodes_without_training_edges": int(np.sum(matrix.sum(axis=0) == 0)),
        "integration_rule": "where(prior > 0, prior, fold_GIP)",
        "fixed_prior": "D_SH.mat / MeSHSemanticSimilarity",
    }
    return {
        "assoc": matrix.astype(np.float32),
        "features_circ": circ_views["jaccard"].copy(),
        "features_dise": dise_views["jaccard"].copy(),
        "circ_views": circ_views, "dise_views": dise_views,
        "circ_views_norm": circ_views_norm, "dise_views_norm": dise_views_norm,
        "diagnostics": diagnostics,
    }
