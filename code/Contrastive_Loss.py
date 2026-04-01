import tensorflow.compat.v1 as tf
import numpy as np


class ContrastiveLoss:

    def __init__(self, temperature=0.5, name='contrastive'):
        self.temperature = temperature
        self.name = name

    def compute_similarity(self, emb1, emb2):
        emb1_norm = tf.nn.l2_normalize(emb1, axis=1)
        emb2_norm = tf.nn.l2_normalize(emb2, axis=1)
        similarity = tf.reduce_sum(emb1_norm * emb2_norm, axis=1)
        return similarity

    def infoNCE_loss(self, emb_circ, emb_dise, pos_edges, neg_edges):
        pos_circ_indices = pos_edges[:, 0]
        pos_dise_indices = pos_edges[:, 1]
        pos_circ_emb = tf.gather(emb_circ, pos_circ_indices)
        pos_dise_emb = tf.gather(emb_dise, pos_dise_indices)

        neg_circ_indices = neg_edges[:, 0]
        neg_dise_indices = neg_edges[:, 1]
        neg_circ_emb = tf.gather(emb_circ, neg_circ_indices)
        neg_dise_emb = tf.gather(emb_dise, neg_dise_indices)

        pos_sim = self.compute_similarity(pos_circ_emb, pos_dise_emb)
        neg_sim = self.compute_similarity(neg_circ_emb, neg_dise_emb)

        pos_sim = pos_sim / self.temperature
        neg_sim = neg_sim / self.temperature

        n_pos = tf.shape(pos_sim)[0]
        n_neg = tf.shape(neg_sim)[0]

        pos_sim_expanded = tf.expand_dims(pos_sim, 1)
        neg_sim_expanded = tf.expand_dims(neg_sim, 0)

        logits = tf.concat([pos_sim_expanded,
                            tf.tile(neg_sim_expanded, [n_pos, 1])], axis=1)

        labels = tf.fill([n_pos], 0)

        loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
        )

        return loss

    def triplet_loss(self, emb_circ, emb_dise, pos_edges, neg_edges, margin=1.0):
        pos_circ_indices = pos_edges[:, 0]
        pos_dise_indices = pos_edges[:, 1]
        pos_circ_emb = tf.gather(emb_circ, pos_circ_indices)
        pos_dise_emb = tf.gather(emb_dise, pos_dise_indices)

        neg_circ_indices = neg_edges[:, 0]
        neg_dise_indices = neg_edges[:, 1]
        neg_circ_emb = tf.gather(emb_circ, neg_circ_indices)
        neg_dise_emb = tf.gather(emb_dise, neg_dise_indices)

        pos_dist = tf.reduce_sum(tf.square(pos_circ_emb - pos_dise_emb), axis=1)
        neg_dist = tf.reduce_sum(tf.square(neg_circ_emb - neg_dise_emb), axis=1)

        min_samples = tf.minimum(tf.shape(pos_dist)[0], tf.shape(neg_dist)[0])
        pos_dist_truncated = pos_dist[:min_samples]
        neg_dist_truncated = neg_dist[:min_samples]

        loss = tf.reduce_mean(tf.maximum(0.0, margin + pos_dist_truncated - neg_dist_truncated))

        return loss

    def simplified_contrastive_loss(self, emb_circ, emb_dise, pos_edges, neg_edges):
        pos_circ_emb = tf.gather(emb_circ, pos_edges[:, 0])
        pos_dise_emb = tf.gather(emb_dise, pos_edges[:, 1])
        neg_circ_emb = tf.gather(emb_circ, neg_edges[:, 0])
        neg_dise_emb = tf.gather(emb_dise, neg_edges[:, 1])

        pos_sim = self.compute_similarity(pos_circ_emb, pos_dise_emb)
        neg_sim = self.compute_similarity(neg_circ_emb, neg_dise_emb)

        pos_loss = tf.reduce_mean(tf.square(1.0 - pos_sim))
        neg_loss = tf.reduce_mean(tf.square(tf.maximum(0.0, neg_sim)))

        loss = pos_loss + neg_loss

        return loss


if __name__ == '__main__':
    print("=" * 80)
    print("Testing contrastive loss")
    print("=" * 80)

    tf.reset_default_graph()

    n_circ = 561
    n_dis = 100
    dim = 64
    n_pos = 100
    n_neg = 200

    ph_emb_circ = tf.placeholder(tf.float32, [n_circ, dim])
    ph_emb_dise = tf.placeholder(tf.float32, [n_dis, dim])
    ph_pos_edges = tf.placeholder(tf.int32, [None, 2])
    ph_neg_edges = tf.placeholder(tf.int32, [None, 2])

    contrastive = ContrastiveLoss(temperature=0.5)

    infoNCE = contrastive.infoNCE_loss(
        ph_emb_circ, ph_emb_dise, ph_pos_edges, ph_neg_edges
    )
    triplet = contrastive.triplet_loss(
        ph_emb_circ, ph_emb_dise, ph_pos_edges, ph_neg_edges, margin=1.0
    )
    simplified = contrastive.simplified_contrastive_loss(
        ph_emb_circ, ph_emb_dise, ph_pos_edges, ph_neg_edges
    )

    with tf.Session() as sess:
        emb_circ = np.random.randn(n_circ, dim).astype(np.float32)
        emb_dise = np.random.randn(n_dis, dim).astype(np.float32)

        pos_circ_idx = np.random.randint(0, n_circ, n_pos)
        pos_dise_idx = np.random.randint(0, n_dis, n_pos)
        pos_edges = np.column_stack([pos_circ_idx, pos_dise_idx])

        neg_circ_idx = np.random.randint(0, n_circ, n_neg)
        neg_dise_idx = np.random.randint(0, n_dis, n_neg)
        neg_edges = np.column_stack([neg_circ_idx, neg_dise_idx])

        infoNCE_val, triplet_val, simplified_val = sess.run(
            [infoNCE, triplet, simplified],
            feed_dict={
                ph_emb_circ: emb_circ,
                ph_emb_dise: emb_dise,
                ph_pos_edges: pos_edges,
                ph_neg_edges: neg_edges
            }
        )

        print(f"\n[Test Results]")
        print(f"  Positive samples: {n_pos}")
        print(f"  Negative samples: {n_neg}")
        print(f"  InfoNCE loss: {infoNCE_val:.4f}")
        print(f"  Triplet loss: {triplet_val:.4f}")
        print(f"  Simplified contrastive loss: {simplified_val:.4f}")

        assert not np.isnan(infoNCE_val), "NaN in InfoNCE loss"
        assert not np.isnan(triplet_val), "NaN in Triplet loss"
        assert not np.isnan(simplified_val), "NaN in Simplified loss"
        assert infoNCE_val >= 0, "InfoNCE loss should be non-negative"
        assert triplet_val >= 0, "Triplet loss should be non-negative"
        assert simplified_val >= 0, "Simplified loss should be non-negative"

    print("\n[Passed] Contrastive_Loss test passed!")
    print("\n[Note] Recommended to use simplified_contrastive_loss during training for stability")