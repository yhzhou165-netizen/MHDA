import tensorflow.compat.v1 as tf
import numpy as np


def glorot_init(shape):
    init_range = np.sqrt(6.0 / (shape[0] + shape[1]))
    return tf.random_uniform(shape, minval=-init_range, maxval=init_range, dtype=tf.float32)


class HeterogeneousGATLayer:

    def __init__(self, input_dim_circ, input_dim_dise, output_dim,
                 adj_circ, adj_dise, adj_cd,
                 dropout=0.0, name='hetero_gat'):
        self.input_dim_circ = input_dim_circ
        self.input_dim_dise = input_dim_dise
        self.output_dim = output_dim
        self.adj_circ = adj_circ
        self.adj_dise = adj_dise
        self.adj_cd = adj_cd
        self.dropout = dropout
        self.name = name

        with tf.variable_scope(self.name):
            self.W_circ = tf.Variable(
                glorot_init([input_dim_circ, output_dim]),
                name='W_circ'
            )
            self.W_dise = tf.Variable(
                glorot_init([input_dim_dise, output_dim]),
                name='W_dise'
            )

            self.a_circ_circ = tf.Variable(
                glorot_init([2 * output_dim, 1]),
                name='attn_circ_circ'
            )
            self.a_dise_dise = tf.Variable(
                glorot_init([2 * output_dim, 1]),
                name='attn_dise_dise'
            )
            self.a_circ_dise = tf.Variable(
                glorot_init([2 * output_dim, 1]),
                name='attn_circ_dise'
            )

            self.alpha_circ = tf.Variable(0.7, dtype=tf.float32, name='alpha_circ')
            self.alpha_dise = tf.Variable(0.7, dtype=tf.float32, name='alpha_dise')

    def _compute_attention(self, h_src, h_dst, attn_param, adj_mask):
        n_src = tf.shape(h_src)[0]
        n_dst = tf.shape(h_dst)[0]

        h_src_expanded = tf.tile(tf.expand_dims(h_src, 1), [1, n_dst, 1])
        h_dst_expanded = tf.tile(tf.expand_dims(h_dst, 0), [n_src, 1, 1])

        concat = tf.concat([h_src_expanded, h_dst_expanded], axis=-1)

        concat_reshaped = tf.reshape(concat, [-1, 2 * self.output_dim])
        e = tf.matmul(concat_reshaped, attn_param)
        e = tf.reshape(e, [n_src, n_dst])

        e = tf.nn.leaky_relu(e, alpha=0.2)

        mask = tf.cast(adj_mask > 0, dtype=tf.float32)
        e = tf.where(mask > 0, e, tf.ones_like(e) * -1e9)

        attention = tf.nn.softmax(e, axis=1)

        return attention

    def __call__(self, features_circ, features_dise, training=None):
        h_circ = tf.matmul(features_circ, self.W_circ)
        h_dise = tf.matmul(features_dise, self.W_dise)

        attn_circ_circ = self._compute_attention(
            h_circ, h_circ, self.a_circ_circ, self.adj_circ
        )

        if self.dropout > 0 and training is not None:
            attn_circ_circ = tf.cond(
                training,
                lambda: tf.nn.dropout(attn_circ_circ, rate=self.dropout),
                lambda: attn_circ_circ
            )

        agg_circ_from_circ = tf.matmul(attn_circ_circ, h_circ)

        attn_circ_dise = self._compute_attention(
            h_circ, h_dise, self.a_circ_dise, self.adj_cd
        )

        if self.dropout > 0 and training is not None:
            attn_circ_dise = tf.cond(
                training,
                lambda: tf.nn.dropout(attn_circ_dise, rate=self.dropout),
                lambda: attn_circ_dise
            )

        agg_circ_from_dise = tf.matmul(attn_circ_dise, h_dise)

        alpha_c = tf.nn.sigmoid(self.alpha_circ)
        emb_circ = alpha_c * agg_circ_from_circ + (1 - alpha_c) * agg_circ_from_dise
        emb_circ = tf.nn.elu(emb_circ)

        attn_dise_dise = self._compute_attention(
            h_dise, h_dise, self.a_dise_dise, self.adj_dise
        )

        if self.dropout > 0 and training is not None:
            attn_dise_dise = tf.cond(
                training,
                lambda: tf.nn.dropout(attn_dise_dise, rate=self.dropout),
                lambda: attn_dise_dise
            )

        agg_dise_from_dise = tf.matmul(attn_dise_dise, h_dise)

        adj_dc = tf.transpose(self.adj_cd)
        attn_dise_circ = self._compute_attention(
            h_dise, h_circ, self.a_circ_dise, adj_dc
        )

        if self.dropout > 0 and training is not None:
            attn_dise_circ = tf.cond(
                training,
                lambda: tf.nn.dropout(attn_dise_circ, rate=self.dropout),
                lambda: attn_dise_circ
            )

        agg_dise_from_circ = tf.matmul(attn_dise_circ, h_circ)

        alpha_d = tf.nn.sigmoid(self.alpha_dise)
        emb_dise = alpha_d * agg_dise_from_dise + (1 - alpha_d) * agg_dise_from_circ
        emb_dise = tf.nn.elu(emb_dise)

        return emb_circ, emb_dise


if __name__ == '__main__':
    print("=" * 80)
    print("Testing heterogeneous graph attention layer")
    print("=" * 80)

    tf.reset_default_graph()

    n_circ = 561
    n_dis = 100
    hidden_dim = 64

    adj_circ = np.random.rand(n_circ, n_circ).astype(np.float32)
    adj_circ = (adj_circ > 0.5).astype(np.float32)

    adj_dise = np.random.rand(n_dis, n_dis).astype(np.float32)
    adj_dise = (adj_dise > 0.5).astype(np.float32)

    adj_cd = np.random.rand(n_circ, n_dis).astype(np.float32)
    adj_cd = (adj_cd > 0.9).astype(np.float32)

    print(f"\n[Test Data]")
    print(f"  adj_circ: {adj_circ.shape}, density={np.mean(adj_circ):.3f}")
    print(f"  adj_dise: {adj_dise.shape}, density={np.mean(adj_dise):.3f}")
    print(f"  adj_cd: {adj_cd.shape}, density={np.mean(adj_cd):.3f}")

    ph_adj_circ = tf.placeholder(tf.float32, [n_circ, n_circ])
    ph_adj_dise = tf.placeholder(tf.float32, [n_dis, n_dis])
    ph_adj_cd = tf.placeholder(tf.float32, [n_circ, n_dis])
    ph_feat_circ = tf.placeholder(tf.float32, [n_circ, hidden_dim])
    ph_feat_dise = tf.placeholder(tf.float32, [n_dis, hidden_dim])
    ph_training = tf.placeholder(tf.bool)

    hetero_layer = HeterogeneousGATLayer(
        input_dim_circ=hidden_dim,
        input_dim_dise=hidden_dim,
        output_dim=hidden_dim,
        adj_circ=ph_adj_circ,
        adj_dise=ph_adj_dise,
        adj_cd=ph_adj_cd,
        dropout=0.1,
        name='test_hetero'
    )

    emb_circ, emb_dise = hetero_layer(ph_feat_circ, ph_feat_dise, training=ph_training)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        feat_circ = np.random.randn(n_circ, hidden_dim).astype(np.float32)
        feat_dise = np.random.randn(n_dis, hidden_dim).astype(np.float32)

        out_circ_train, out_dise_train = sess.run(
            [emb_circ, emb_dise],
            feed_dict={
                ph_adj_circ: adj_circ,
                ph_adj_dise: adj_dise,
                ph_adj_cd: adj_cd,
                ph_feat_circ: feat_circ,
                ph_feat_dise: feat_dise,
                ph_training: True
            }
        )

        out_circ_test, out_dise_test = sess.run(
            [emb_circ, emb_dise],
            feed_dict={
                ph_adj_circ: adj_circ,
                ph_adj_dise: adj_dise,
                ph_adj_cd: adj_cd,
                ph_feat_circ: feat_circ,
                ph_feat_dise: feat_dise,
                ph_training: False
            }
        )

        alpha_c_val, alpha_d_val = sess.run([
            tf.nn.sigmoid(hetero_layer.alpha_circ),
            tf.nn.sigmoid(hetero_layer.alpha_dise)
        ])

        print(f"\n[Test Results]")
        print(f"  CircRNA embedding shape: {out_circ_train.shape}")
        print(f"  Disease embedding shape: {out_dise_train.shape}")
        print(f"  CircRNA fusion weight (alpha_circ): {alpha_c_val:.4f}")
        print(f"  Disease fusion weight (alpha_dise): {alpha_d_val:.4f}")
        print(f"  CircRNA embedding mean: {np.mean(out_circ_train):.4f}")
        print(f"  Disease embedding mean: {np.mean(out_dise_train):.4f}")
        print(f"  Difference between train/test mode:")
        print(f"    CircRNA: {np.mean(np.abs(out_circ_train - out_circ_test)):.6f}")
        print(f"    Disease: {np.mean(np.abs(out_dise_train - out_dise_test)):.6f}")

        assert out_circ_train.shape == (n_circ, hidden_dim), "CircRNA shape mismatch"
        assert out_dise_train.shape == (n_dis, hidden_dim), "Disease shape mismatch"
        assert not np.any(np.isnan(out_circ_train)), "NaN in CircRNA output"
        assert not np.any(np.isnan(out_dise_train)), "NaN in Disease output"
        assert not np.any(np.isinf(out_circ_train)), "Inf in CircRNA output"
        assert not np.any(np.isinf(out_dise_train)), "Inf in Disease output"

    print("\n[Passed] Heterogeneous_Layers test passed!")