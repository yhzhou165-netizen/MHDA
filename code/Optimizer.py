"""Metrics: validation threshold selection is separate from testing."""
import numpy as np
from scipy.special import expit
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score,
                             precision_score, recall_score, accuracy_score)


class OptimizerMetrics:
    @staticmethod
    def probabilities(logits):
        scores = np.asarray(logits, dtype=float).reshape(-1)
        if not np.all(np.isfinite(scores)):
            raise ValueError('Prediction logits must be finite')
        return expit(scores)

    @staticmethod
    def compute_metrics(y_true, y_pred_logits, threshold=0.5):
        """Evaluate at an already chosen threshold; never tune on these labels."""
        labels = np.asarray(y_true).reshape(-1)
        scores = OptimizerMetrics.probabilities(y_pred_logits)
        if len(labels) != len(scores) or len(labels) == 0:
            raise ValueError('Labels and predictions must have equal nonzero lengths')
        if not np.all(np.isin(labels, [0, 1])) or len(np.unique(labels)) != 2:
            raise ValueError('Evaluation requires both binary classes')
        if not 0 <= threshold <= 1:
            raise ValueError('Threshold must be in [0, 1]')
        predicted = (scores >= threshold).astype(int)
        return {
            'auc': float(roc_auc_score(labels, scores) * 100),
            'aupr': float(average_precision_score(labels, scores) * 100),
            'f1': float(f1_score(labels, predicted, zero_division=0) * 100),
            'precision': float(precision_score(labels, predicted, zero_division=0) * 100),
            'recall': float(recall_score(labels, predicted, zero_division=0) * 100),
            'accuracy': float(accuracy_score(labels, predicted) * 100),
        }

    @staticmethod
    def compute_metrics_with_optimal_threshold(y_true, y_pred_logits):
        """Select by VALIDATION F1 only. Never pass test labels to this method."""
        best = None
        for threshold in np.arange(0.30, 0.70, 0.01):
            candidate = OptimizerMetrics.compute_metrics(y_true, y_pred_logits, float(threshold))
            if best is None or candidate['f1'] > best['f1']:
                best = dict(candidate, optimal_threshold=float(threshold))
        return best
