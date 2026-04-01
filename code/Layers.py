import tensorflow.compat.v1 as tf
import numpy as np


def glorot_init(shape):
    init_range = np.sqrt(6.0 / (shape[0] + shape[1]))
    return tf.random_uniform(shape, minval=-init_range, maxval=init_range, dtype=tf.float32)


class GraphAttentionLayer:

    def __init__(self, input_dim, output_dim, adj, dropout=0.0, activation=tf.nn.elu, name='gat'):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.adj = adj
        self.dropout = dropout
        self.activation = activation
        self.name = name

        with tf.variable_scope(self.name):
            self.W = tf.Variable(glorot_init([input_dim, output_dim]), name='weights')
            self.a = tf.Variable(glorot_init([2 * output_dim, 1]), name='attention')

    def __call__(self, inputs, training=None):
        h = tf.matmul(inputs, self.W)
        n_nodes = tf.shape(h)[0]

        h_i = tf.tile(tf.expand_dims(h, 1), [1, n_nodes, 1])
        h_j = tf.tile(tf.expand_dims(h, 0), [n_nodes, 1, 1])
        concat = tf.concat([h_i, h_j], axis=-1)

        e = tf.squeeze(tf.matmul(concat, self.a), axis=-1)
        e = tf.nn.leaky_relu(e, alpha=0.2)

        mask = tf.cast(self.adj > 0, dtype=tf.float32)
        e = tf.where(mask > 0, e, tf.ones_like(e) * -1e9)

        attention = tf.nn.softmax(e, axis=1)

        if self.dropout > 0 and training is not None:
            attention = tf.cond(
                training,
                lambda: tf.nn.dropout(attention, rate=self.dropout),
                lambda: attention
            )

        h_out = tf.matmul(attention, h)

        if self.activation is not None:
            h_out = self.activation(h_out)

        return h_out


class MLPDecoder:

    def __init__(self, input_dim, hidden_dim=64, dropout=0.0, name='decoder'):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.name = name

        with tf.variable_scope(self.name):
            self.W1 = tf.Variable(glorot_init([input_dim, hidden_dim]), name='W1')
            self.b1 = tf.Variable(tf.zeros([hidden_dim]), name='b1')
            self.W2 = tf.Variable(glorot_init([hidden_dim, hidden_dim // 2]), name='W2')
            self.b2 = tf.Variable(tf.zeros([hidden_dim // 2]), name='b2')
            self.W3 = tf.Variable(glorot_init([hidden_dim // 2, 1]), name='W3')
            self.b3 = tf.Variable(tf.zeros([1]), name='b3')

    def __call__(self, inputs, training=None):
        h = tf.nn.relu(tf.matmul(inputs, self.W1) + self.b1)
        if self.dropout > 0 and training is not None:
            h = tf.cond(
                training,
                lambda: tf.nn.dropout(h, rate=self.dropout),
                lambda: h
            )

        h = tf.nn.relu(tf.matmul(h, self.W2) + self.b2)
        if self.dropout > 0 and training is not None:
            h = tf.cond(
                training,
                lambda: tf.nn.dropout(h, rate=self.dropout),
                lambda: h
            )

        out = tf.matmul(h, self.W3) + self.b3
        return out


class DualStreamDecoder:

    def __init__(self, input_dim, hidden_dim=64, dropout=0.0, name='dual_decoder'):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.name = name
        self.fused_dim = hidden_dim // 2

        with tf.variable_scope(self.name):
            self.W_struct_1 = tf.Variable(glorot_init([input_dim, hidden_dim]), name='W_struct_1')
            self.b_struct_1 = tf.Variable(tf.zeros([hidden_dim]), name='b_struct_1')
            self.W_struct_2 = tf.Variable(glorot_init([hidden_dim, self.fused_dim]), name='W_struct_2')
            self.b_struct_2 = tf.Variable(tf.zeros([self.fused_dim]), name='b_struct_2')

            self.W_semantic_1 = tf.Variable(glorot_init([input_dim * 2, hidden_dim]), name='W_semantic_1')
            self.b_semantic_1 = tf.Variable(tf.zeros([hidden_dim]), name='b_semantic_1')
            self.W_semantic_2 = tf.Variable(glorot_init([hidden_dim, self.fused_dim]), name='W_semantic_2')
            self.b_semantic_2 = tf.Variable(tf.zeros([self.fused_dim]), name='b_semantic_2')

            self.W_gate = tf.Variable(glorot_init([hidden_dim, self.fused_dim]), name='W_gate')
            self.b_gate = tf.Variable(tf.zeros([self.fused_dim]), name='b_gate')

            self.stream_weight = tf.Variable(0.0, dtype=tf.float32, name='stream_weight')

            self.W_fused = tf.Variable(glorot_init([self.fused_dim, self.fused_dim]), name='W_fused')
            self.b_fused = tf.Variable(tf.zeros([self.fused_dim]), name='b_fused')

            self.W_out = tf.Variable(glorot_init([self.fused_dim, 1]), name='W_out')
            self.b_out = tf.Variable(tf.zeros([1]), name='b_out')

    def __call__(self, emb_circ, emb_dise, training=None):
        edge_embs_struct = emb_circ * emb_dise
        edge_embs_semantic = tf.concat([emb_circ, emb_dise], axis=-1)

        h_struct = tf.nn.relu(tf.matmul(edge_embs_struct, self.W_struct_1) + self.b_struct_1)
        if self.dropout > 0 and training is not None:
            h_struct = tf.cond(
                training,
                lambda: tf.nn.dropout(h_struct, rate=self.dropout),
                lambda: h_struct
            )
        h_struct = tf.nn.relu(tf.matmul(h_struct, self.W_struct_2) + self.b_struct_2)

        h_semantic = tf.nn.relu(tf.matmul(edge_embs_semantic, self.W_semantic_1) + self.b_semantic_1)
        if self.dropout > 0 and training is not None:
            h_semantic = tf.cond(
                training,
                lambda: tf.nn.dropout(h_semantic, rate=self.dropout),
                lambda: h_semantic
            )
        h_semantic = tf.nn.relu(tf.matmul(h_semantic, self.W_semantic_2) + self.b_semantic_2)

        gate_input = tf.concat([h_struct, h_semantic], axis=-1)
        gate_logits = tf.matmul(gate_input, self.W_gate) + self.b_gate + self.stream_weight
        fusion_gate = tf.nn.sigmoid(gate_logits)

        h_fused = fusion_gate * h_struct + (1.0 - fusion_gate) * h_semantic

        h_fused = tf.nn.relu(tf.matmul(h_fused, self.W_fused) + self.b_fused)
        if self.dropout > 0 and training is not None:
            h_fused = tf.cond(
                training,
                lambda: tf.nn.dropout(h_fused, rate=self.dropout),
                lambda: h_fused
            )

        out = tf.matmul(h_fused, self.W_out) + self.b_out
        return out


if __name__ == '__main__':
    print("=" * 80)
    print("Testing DualStreamDecoder")
    print("=" * 80)

    tf.reset_default_graph()

    batch_size = 256
    input_dim = 64
    hidden_dim = 64

    ph_emb_circ = tf.placeholder(tf.float32, [None, input_dim], name='emb_circ')
    ph_emb_dise = tf.placeholder(tf.float32, [None, input_dim], name='emb_dise')
    ph_training = tf.placeholder(tf.bool, name='training')

    decoder = DualStreamDecoder(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        dropout=0.1,
        name='test_dual_decoder'
    )

    scores = decoder(ph_emb_circ, ph_emb_dise, training=ph_training)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        emb_circ = np.random.randn(batch_size, input_dim).astype(np.float32)
        emb_dise = np.random.randn(batch_size, input_dim).astype(np.float32)

        scores_train = sess.run(
            scores,
            feed_dict={
                ph_emb_circ: emb_circ,
                ph_emb_dise: emb_dise,
                ph_training: True
            }
        )

        scores_test = sess.run(
            scores,
            feed_dict={
                ph_emb_circ: emb_circ,
                ph_emb_dise: emb_dise,
                ph_training: False
            }
        )

        alpha_val = sess.run(tf.nn.sigmoid(decoder.stream_weight))

        print(f"\n[Test Results]")
        print(f"  Output scores shape: {scores_train.shape}")
        print(f"  Global fusion bias sigmoid(stream_weight): {alpha_val:.4f}")
        print(f"  Scores statistics (train mode):")
        print(f"    Mean: {np.mean(scores_train):.4f}")
        print(f"    Std: {np.std(scores_train):.4f}")
        print(f"    Min: {np.min(scores_train):.4f}")
        print(f"    Max: {np.max(scores_train):.4f}")
        print(f"  Difference between train/test: {np.mean(np.abs(scores_train - scores_test)):.6f}")

        assert scores_train.shape == (batch_size, 1), "Output shape mismatch"
        assert not np.any(np.isnan(scores_train)), "NaN in output"
        assert not np.any(np.isinf(scores_train)), "Inf in output"

    print("\n[Passed] DualStreamDecoder test passed!")