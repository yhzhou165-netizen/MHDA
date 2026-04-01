import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score, \
    accuracy_score


class OptimizerMetrics:

    @staticmethod
    def compute_metrics(y_true, y_pred_logits):
        y_true = np.array(y_true).flatten()
        y_pred_logits = np.array(y_pred_logits).flatten()

        y_pred_prob = 1.0 / (1.0 + np.exp(-y_pred_logits))
        y_pred_binary = (y_pred_prob > 0.5).astype(int)

        try:
            auc = roc_auc_score(y_true, y_pred_prob) * 100
        except:
            auc = 0.0

        try:
            aupr = average_precision_score(y_true, y_pred_prob) * 100
        except:
            aupr = 0.0

        try:
            f1 = f1_score(y_true, y_pred_binary) * 100
        except:
            f1 = 0.0

        try:
            precision = precision_score(y_true, y_pred_binary, zero_division=0) * 100
        except:
            precision = 0.0

        try:
            recall = recall_score(y_true, y_pred_binary, zero_division=0) * 100
        except:
            recall = 0.0

        try:
            accuracy = accuracy_score(y_true, y_pred_binary) * 100
        except:
            accuracy = 0.0

        return {
            'auc': auc,
            'aupr': aupr,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'accuracy': accuracy
        }

    @staticmethod
    def compute_metrics_with_optimal_threshold(y_true, y_pred_logits):
        y_true = np.array(y_true).flatten()
        y_pred_logits = np.array(y_pred_logits).flatten()

        y_pred_prob = 1.0 / (1.0 + np.exp(-y_pred_logits))

        try:
            auc = roc_auc_score(y_true, y_pred_prob) * 100
        except:
            auc = 0.0

        try:
            aupr = average_precision_score(y_true, y_pred_prob) * 100
        except:
            aupr = 0.0

        best_f1 = 0.0
        best_threshold = 0.5
        best_precision = 0.0
        best_recall = 0.0
        best_accuracy = 0.0

        for threshold in np.arange(0.30, 0.70, 0.01):
            y_pred_binary = (y_pred_prob >= threshold).astype(int)
            try:
                f1 = f1_score(y_true, y_pred_binary) * 100
                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = threshold
                    best_precision = precision_score(y_true, y_pred_binary, zero_division=0) * 100
                    best_recall = recall_score(y_true, y_pred_binary, zero_division=0) * 100
                    best_accuracy = accuracy_score(y_true, y_pred_binary) * 100
            except:
                continue

        return {
            'auc': auc,
            'aupr': aupr,
            'f1': best_f1,
            'precision': best_precision,
            'recall': best_recall,
            'accuracy': best_accuracy,
            'optimal_threshold': best_threshold
        }