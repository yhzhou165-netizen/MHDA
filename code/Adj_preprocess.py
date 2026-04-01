import random
import numpy as np
import tensorflow.compat.v1 as tf
import os

random.seed(42)
np.random.seed(42)
tf.set_random_seed(42)
os.environ['PYTHONHASHSEED'] = '42'

import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.io as sio


def read_data(path):
    if path.lower().endswith('.mat'):
        mat_data = sio.loadmat(path)
        print(f"Loaded .mat file: {path}")
        print("Keys in the .mat file:", mat_data.keys())
        if 'integrated_circ_sim' in mat_data:
            return mat_data['integrated_circ_sim']
        elif 'integrated_dise_sim' in mat_data:
            return mat_data['integrated_dise_sim']
        else:
            raise KeyError("No suitable key found, please check file content")
    else:
        return np.loadtxt(path)


def jaccard_rows_from_assoc(assoc_matrix):
    n_rows = assoc_matrix.shape[0]
    jaccard_matrix = np.zeros((n_rows, n_rows))

    for i in range(n_rows):
        for j in range(n_rows):
            intersection = np.sum(np.minimum(assoc_matrix[i, :], assoc_matrix[j, :]))
            union = np.sum(np.maximum(assoc_matrix[i, :], assoc_matrix[j, :]))
            jaccard_matrix[i, j] = intersection / float(union) if union != 0 else 0

    return jaccard_matrix


def compute_cosine_similarity(matrix):
    norm = np.linalg.norm(matrix, axis=1, keepdims=True)
    norm[norm == 0] = 1
    matrix_norm = matrix / norm
    cosine_sim = np.dot(matrix_norm, matrix_norm.T)
    np.fill_diagonal(cosine_sim, 1.0)
    cosine_sim = np.clip(cosine_sim, 0, 1)
    return cosine_sim


def compute_gaussian_similarity(matrix, gamma=1.0):
    n = matrix.shape[0]
    gaussian_sim = np.zeros((n, n))

    for i in range(n):
        for j in range(i, n):
            dist = np.linalg.norm(matrix[i] - matrix[j])
            sim = np.exp(-gamma * dist ** 2)
            gaussian_sim[i, j] = sim
            gaussian_sim[j, i] = sim

    return gaussian_sim


def adjacency_circRNA_disease():
    data_path = os.path.join(os.path.dirname(os.getcwd()), "data")

    excel_path = os.path.join(data_path, 'Association Matrixs.xlsx')
    assoc_df = pd.read_excel(excel_path, sheet_name=2, header=None)
    assoc_matrix = assoc_df.values.astype(float)
    n_circ, n_dis = assoc_matrix.shape

    c_mat = os.path.join(data_path, 'integrated_circ_sim.mat')
    c_txt = os.path.join(data_path, 'c-c.txt')
    d_mat = os.path.join(data_path, 'integrated_dise_sim.mat')
    d_txt = os.path.join(data_path, 'd-d.txt')

    if os.path.exists(c_mat):
        circ_sim = read_data(c_mat)
    elif os.path.exists(c_txt):
        circ_sim = read_data(c_txt)
    else:
        raise FileNotFoundError("c-c similarity file not found")

    if os.path.exists(d_mat):
        dise_sim = read_data(d_mat)
    elif os.path.exists(d_txt):
        dise_sim = read_data(d_txt)
    else:
        raise FileNotFoundError("d-d similarity file not found")

    circ_sim = np.array(circ_sim, dtype=float)
    dise_sim = np.array(dise_sim, dtype=float)

    A_bin = (assoc_matrix > 0).astype(float)
    jaccard_circ = jaccard_rows_from_assoc(A_bin)
    jaccard_dise = jaccard_rows_from_assoc(A_bin.T)

    adj_circ = sp.csr_matrix(circ_sim)
    adj_dise = sp.csr_matrix(dise_sim)

    print(f"[Adj] assoc_matrix shape: {assoc_matrix.shape}")
    print(f"[Adj] adj_circ shape: {adj_circ.shape}, adj_dise shape: {adj_dise.shape}")
    print(f"[Adj] jaccard_circ shape: {jaccard_circ.shape}, jaccard_dise shape: {jaccard_dise.shape}")

    return adj_circ, adj_dise, assoc_matrix, jaccard_circ, jaccard_dise


def adjacency_circRNA_disease_multi_view():
    data_path = os.path.join(os.path.dirname(os.getcwd()), "data")

    excel_path = os.path.join(data_path, 'Association Matrixs.xlsx')
    assoc_df = pd.read_excel(excel_path, sheet_name=2, header=None)
    assoc_matrix = assoc_df.values.astype(float)
    n_circ, n_dis = assoc_matrix.shape

    print(f"\n[Multi-View Data Loading]")
    print(f"  Association matrix shape: {assoc_matrix.shape}")
    print(f"  Number of positive associations: {np.sum(assoc_matrix > 0)}")

    c_mat = os.path.join(data_path, 'integrated_circ_sim.mat')
    d_mat = os.path.join(data_path, 'integrated_dise_sim.mat')

    circ_sim_integrated = read_data(c_mat)
    dise_sim_integrated = read_data(d_mat)

    A_bin = (assoc_matrix > 0).astype(float)

    print(f"\n[Computing Multi-View Similarities]")
    print("  (Gaussian view removed. Using 3 views: jaccard, cosine, integrated)")

    print("  CircRNA views:")
    print("    - Computing Jaccard similarity...")
    circ_view1_jaccard = jaccard_rows_from_assoc(A_bin)

    print("    - Computing Cosine similarity...")
    circ_view2_cosine = compute_cosine_similarity(A_bin)

    print("    - Loading Integrated similarity...")
    circ_view3_integrated = np.array(circ_sim_integrated, dtype=float)

    print("  Disease views:")
    print("    - Computing Jaccard similarity...")
    dise_view1_jaccard = jaccard_rows_from_assoc(A_bin.T)

    print("    - Computing Cosine similarity...")
    dise_view2_cosine = compute_cosine_similarity(A_bin.T)

    print("    - Loading Integrated similarity...")
    dise_view3_integrated = np.array(dise_sim_integrated, dtype=float)

    circ_views = {
        'jaccard': circ_view1_jaccard,
        'cosine': circ_view2_cosine,
        'integrated': circ_view3_integrated
    }

    dise_views = {
        'jaccard': dise_view1_jaccard,
        'cosine': dise_view2_cosine,
        'integrated': dise_view3_integrated
    }

    print(f"\n[Multi-View Statistics]")
    print("CircRNA views:")
    for view_name, view_data in circ_views.items():
        print(f"  {view_name:12s}: shape={view_data.shape}, "
              f"mean={np.mean(view_data):.4f}, "
              f"std={np.std(view_data):.4f}, "
              f"min={np.min(view_data):.4f}, "
              f"max={np.max(view_data):.4f}")

    print("Disease views:")
    for view_name, view_data in dise_views.items():
        print(f"  {view_name:12s}: shape={view_data.shape}, "
              f"mean={np.mean(view_data):.4f}, "
              f"std={np.std(view_data):.4f}, "
              f"min={np.min(view_data):.4f}, "
              f"max={np.max(view_data):.4f}")

    return circ_views, dise_views, assoc_matrix


if __name__ == '__main__':
    print("=" * 80)
    print("Testing multi-view data loading (3 views: jaccard/cosine/integrated)")
    print("=" * 80)

    circ_views, dise_views, assoc_matrix = adjacency_circRNA_disease_multi_view()

    print("\n[Test Passed] Multi-view data loading successful!")