import random
import numpy as np
import tensorflow.compat.v1 as tf
import os

random.seed(42)
np.random.seed(42)
tf.set_random_seed(42)
os.environ['PYTHONHASHSEED'] = '42'

import numpy as np
import scipy.sparse as sp
from sklearn.model_selection import KFold


def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).T
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape


def preprocess_graph(adj):
    if not sp.issparse(adj):
        adj = sp.csr_matrix(adj)
    adj = adj + sp.eye(adj.shape[0], dtype=adj.dtype, format='csr')
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv_sqrt = np.power(rowsum, -0.5, where=rowsum > 0)
    d_inv_sqrt[~np.isfinite(d_inv_sqrt)] = 0.0
    D_inv_sqrt = sp.diags(d_inv_sqrt)
    return (D_inv_sqrt @ adj @ D_inv_sqrt).tocsr()


def split_train_test(assoc_matrix, n_splits=5, seed=1024, neg_ratio=1.0):
    A = np.asarray(assoc_matrix, dtype=float)
    if A.ndim != 2:
        raise ValueError(f"assoc_matrix dimension should be 2, got {A.ndim}.")
    n_circ, n_dis = A.shape

    pos_idx = np.argwhere(A > 0.0)
    neg_idx = np.argwhere(A <= 0.0)

    rng = np.random.RandomState(seed)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    train_e_p_list, train_e_n_list = [], []
    test_e_p_list, test_e_n_list = [], []
    adj_dm_new_list = []

    for train_pos_idx, test_pos_idx in kf.split(pos_idx):
        train_pos = pos_idx[train_pos_idx]
        test_pos = pos_idx[test_pos_idx]

        n_train_pos = train_pos.shape[0]
        n_train_neg = int(np.round(n_train_pos * float(neg_ratio)))
        sample_ids = rng.choice(len(neg_idx), size=n_train_neg, replace=False)
        train_neg = neg_idx[sample_ids]

        n_test_neg = test_pos.shape[0]
        sample_ids_te = rng.choice(len(neg_idx), size=n_test_neg, replace=False)
        test_neg = neg_idx[sample_ids_te]

        A_train = A.copy()
        A_train[test_pos[:, 0], test_pos[:, 1]] = 0.0

        train_e_p_list.append(train_pos)
        train_e_n_list.append(train_neg)
        test_e_p_list.append(test_pos)
        test_e_n_list.append(test_neg)
        adj_dm_new_list.append(A_train)

    print(f"[split_train_test] assoc_matrix shape: {A.shape}")
    print(f"[split_train_test] adj_dm_new_list[0] shape: {adj_dm_new_list[0].shape}")

    return (train_e_p_list, train_e_n_list,
            test_e_p_list, test_e_n_list,
            adj_dm_new_list)