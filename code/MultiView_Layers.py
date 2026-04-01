import tensorflow.compat.v1 as tf
import numpy as np


def glorot_init(shape):
    init_range = np.sqrt(6.0 / (shape[0] + shape[1]))
    return tf.random_uniform(shape, minval=-init_range, maxval=init_range, dtype=tf.float32)


class MultiViewAttention:

    def __init__(self, n_views, hidden_dim, name='mv_attention'):
        self.n_views = n_views
        self.hidden_dim = hidden_dim
        self.name = name

        with tf.variable_scope(self.name):
            self.view_transforms = []
            for i in range(n_views):
                W = tf.Variable(
                    glorot_init([hidden_dim, hidden_dim]),
                    name=f'view_transform_{i}'
                )
                self.view_transforms.append(W)

            self.attention_w = tf.Variable(
                glorot_init([hidden_dim, 1]),
                name='attention_w'
            )

    def __call__(self, view_embeddings):
        transformed_views = []
        for i, emb in enumerate(view_embeddings):
            transformed = tf.matmul(emb, self.view_transforms[i])
            transformed_views.append(transformed)

        stacked = tf.stack(transformed_views, axis=1)

        n_nodes = tf.shape(stacked)[0]
        stacked_reshaped = tf.reshape(stacked, [-1, self.hidden_dim])

        attention_scores = tf.matmul(stacked_reshaped, self.attention_w)

        attention_scores = tf.reshape(attention_scores, [n_nodes, self.n_views, 1])

        attention_weights = tf.nn.softmax(attention_scores, axis=1)

        weighted_sum = tf.reduce_sum(stacked * attention_weights, axis=1)

        return weighted_sum, attention_weights


class MultiViewGAT:

    def __init__(self, n_views, input_dim, output_dim, adj_list,
                 dropout=0.0, name='mv_gat'):
        self.n_views = n_views
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.adj_list = adj_list
        self.dropout = dropout
        self.name = name

        with tf.variable_scope(self.name):
            from Layers import GraphAttentionLayer
            self.gat_layers = []
            for i in range(n_views):
                gat = GraphAttentionLayer(
                    input_dim=input_dim,
                    output_dim=output_dim,
                    adj=adj_list[i],
                    dropout=dropout,
                    name=f'gat_view_{i}'
                )
                self.gat_layers.append(gat)

            self.fusion = MultiViewAttention(
                n_views=n_views,
                hidden_dim=output_dim,
                name='fusion'
            )

    def __call__(self, features, training=None):
        view_embeddings = []
        for gat in self.gat_layers:
            emb = gat(features, training=training)
            view_embeddings.append(emb)

        fused_emb, attn_weights = self.fusion(view_embeddings)

        return fused_emb, attn_weights


if __name__ == '__main__':
    print("=" * 80)
    print("Testing multi-view graph attention layer")
    print("=" * 80)

    tf.reset_default_graph()

    n_nodes = 100
    n_views = 4
    input_dim = 100
    output_dim = 64

    adj_list = []
    for i in range(n_views):
        adj = np.random.rand(n_nodes, n_nodes).astype(np.float32)
        adj = (adj + adj.T) / 2
        adj_list.append(adj)

    ph_adj_list = []
    for i in range(n_views):
        ph_adj = tf.placeholder(tf.float32, [n_nodes, n_nodes], name=f'adj_{i}')
        ph_adj_list.append(ph_adj)

    ph_features = tf.placeholder(tf.float32, [n_nodes, input_dim], name='features')
    ph_training = tf.placeholder(tf.bool, name='training')

    mv_gat = MultiViewGAT(
        n_views=n_views,
        input_dim=input_dim,
        output_dim=output_dim,
        adj_list=ph_adj_list,
        dropout=0.1,
        name='test_mv_gat'
    )

    emb, attn = mv_gat(ph_features, training=ph_training)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        features = np.random.rand(n_nodes, input_dim).astype(np.float32)

        feed_dict = {ph_features: features, ph_training: False}
        for i, ph_adj in enumerate(ph_adj_list):
            feed_dict[ph_adj] = adj_list[i]

        emb_out, attn_out = sess.run([emb, attn], feed_dict=feed_dict)

        print(f"\n[Test Results]")
        print(f"  Output embedding shape: {emb_out.shape}")
        print(f"  Attention weights shape: {attn_out.shape}")
        print(f"  Attention weights sum per node (should be ~1.0): {np.mean(np.sum(attn_out, axis=1)):.4f}")
        print(f"  Mean attention per view: {np.mean(attn_out, axis=(0, 2))}")

        assert emb_out.shape == (n_nodes, output_dim), "Output shape mismatch"
        assert attn_out.shape == (n_nodes, n_views, 1), "Attention shape mismatch"
        assert not np.any(np.isnan(emb_out)), "NaN in output"

    print("\n[Passed] MultiView_Layers test passed!")