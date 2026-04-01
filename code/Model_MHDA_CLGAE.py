import random
import numpy as np
import tensorflow.compat.v1 as tf
import os

random.seed(42)
np.random.seed(42)
tf.set_random_seed(42)
os.environ['PYTHONHASHSEED'] = '42'

import tensorflow.compat.v1 as tf
from MultiView_Layers import MultiViewGAT
from Heterogeneous_Layers import HeterogeneousGATLayer
from Layers import DualStreamDecoder, MLPDecoder
from Contrastive_Loss import ContrastiveLoss


class MHDA_CLGAE:

    def __init__(self, n_circ, n_dis,
                 circ_views_adj, dise_views_adj, assoc_matrix,
                 hidden_dim=64, dropout=0.2,
                 use_multi_view=True,
                 use_heterogeneous=True,
                 use_dual_stream=True,
                 use_contrastive=True,
                 name='MHDA_CLGAE'):
        self.n_circ = n_circ
        self.n_dis = n_dis
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.name = name

        self.use_multi_view = use_multi_view
        self.use_heterogeneous = use_heterogeneous
        self.use_dual_stream = use_dual_stream
        self.use_contrastive = use_contrastive

        print(f"\n[Model Configuration]")
        print(f"  Multi-View Learning: {use_multi_view}")
        print(f"  Heterogeneous Attention: {use_heterogeneous}")
        print(f"  Dual-Stream Decoder: {use_dual_stream}")
        print(f"  Contrastive Learning: {use_contrastive}")

        with tf.variable_scope(self.name):
            if use_multi_view:
                n_views = len(circ_views_adj)
                print(f"  Number of views: {n_views}")

                self.mv_gat_circ = MultiViewGAT(
                    n_views=n_views,
                    input_dim=n_circ,
                    output_dim=hidden_dim,
                    adj_list=list(circ_views_adj.values()),
                    dropout=dropout,
                    name='mv_gat_circ'
                )
                self.mv_gat_dise = MultiViewGAT(
                    n_views=n_views,
                    input_dim=n_dis,
                    output_dim=hidden_dim,
                    adj_list=list(dise_views_adj.values()),
                    dropout=dropout,
                    name='mv_gat_dise'
                )
            else:
                from Layers import GraphAttentionLayer
                first_circ_adj = list(circ_views_adj.values())[0]
                first_dise_adj = list(dise_views_adj.values())[0]
                self.gat_circ = GraphAttentionLayer(
                    input_dim=n_circ,
                    output_dim=hidden_dim,
                    adj=first_circ_adj,
                    dropout=dropout,
                    name='gat_circ'
                )
                self.gat_dise = GraphAttentionLayer(
                    input_dim=n_dis,
                    output_dim=hidden_dim,
                    adj=first_dise_adj,
                    dropout=dropout,
                    name='gat_dise'
                )

            if use_heterogeneous:
                adj_circ_hetero = list(circ_views_adj.values())[0]
                adj_dise_hetero = list(dise_views_adj.values())[0]

                self.hetero_gat = HeterogeneousGATLayer(
                    input_dim_circ=hidden_dim,
                    input_dim_dise=hidden_dim,
                    output_dim=hidden_dim,
                    adj_circ=adj_circ_hetero,
                    adj_dise=adj_dise_hetero,
                    adj_cd=assoc_matrix,
                    dropout=dropout,
                    name='hetero_gat'
                )

            if use_dual_stream:
                self.decoder = DualStreamDecoder(
                    input_dim=hidden_dim,
                    hidden_dim=hidden_dim,
                    dropout=dropout,
                    name='dual_decoder'
                )
            else:
                self.decoder = MLPDecoder(
                    input_dim=hidden_dim * 2,
                    hidden_dim=hidden_dim,
                    dropout=dropout,
                    name='mlp_decoder'
                )

            if use_contrastive:
                self.contrastive_loss_fn = ContrastiveLoss(
                    temperature=0.5,
                    name='contrastive'
                )

    def encode(self, features_circ, features_dise, training=True):
        if self.use_multi_view:
            emb_circ, attn_weights_circ = self.mv_gat_circ(features_circ, training=training)
            emb_dise, attn_weights_dise = self.mv_gat_dise(features_dise, training=training)
        else:
            emb_circ = self.gat_circ(features_circ, training=training)
            emb_dise = self.gat_dise(features_dise, training=training)

        if self.use_heterogeneous:
            emb_circ, emb_dise = self.hetero_gat(emb_circ, emb_dise, training=training)

        return emb_circ, emb_dise

    def decode(self, emb_circ, emb_dise, edges, training=True):
        circ_indices = edges[:, 0]
        dise_indices = edges[:, 1]

        circ_embs = tf.gather(emb_circ, circ_indices)
        dise_embs = tf.gather(emb_dise, dise_indices)

        if self.use_dual_stream:
            scores = self.decoder(circ_embs, dise_embs, training=training)
        else:
            edge_embs = tf.concat([circ_embs, dise_embs], axis=-1)
            scores = self.decoder(edge_embs, training=training)

        return scores

    def compute_contrastive_loss(self, emb_circ, emb_dise, pos_edges, neg_edges):
        if not self.use_contrastive:
            return tf.constant(0.0)

        return self.contrastive_loss_fn.simplified_contrastive_loss(
            emb_circ, emb_dise, pos_edges, neg_edges
        )

    def __call__(self, features_circ, features_dise, edges, training=True):
        emb_circ, emb_dise = self.encode(features_circ, features_dise, training)
        scores = self.decode(emb_circ, emb_dise, edges, training)
        return scores, emb_circ, emb_dise