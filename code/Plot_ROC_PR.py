import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.font_manager import FontProperties
from sklearn.metrics import roc_curve, precision_recall_curve
import datetime


class ROCPRPlotter:

    FOLD_COLORS = ["#6c90c2", "#ebe974", "#609f8e", "#7de489", "#a7dcd8"]
    BASELINE_COLOR = "#94A3B8"
    MEAN_COLOR = "#FF6B35"

    def __init__(self, save_dir='figs'):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def _compute_mean_curve(self, fold_results, curve_type='roc'):
        if curve_type == 'roc':
            common_x = np.linspace(0, 1, 100)
            y_list = []
            metrics = []

            for result in fold_results:
                y_true = result['y_true']
                y_scores = result['y_scores']
                fpr, tpr, _ = roc_curve(y_true, y_scores)
                tpr_interp = np.interp(common_x, fpr, tpr)
                y_list.append(tpr_interp)
                metrics.append(result.get('auc', 0.0))

            mean_y = np.mean(y_list, axis=0)
            std_y = np.std(y_list, axis=0)
            mean_metric = np.mean(metrics)

            return common_x, mean_y, std_y, mean_metric

        else:
            common_x = np.linspace(0, 1, 100)
            y_list = []
            metrics = []

            for result in fold_results:
                y_true = result['y_true']
                y_scores = result['y_scores']
                precision, recall, _ = precision_recall_curve(y_true, y_scores)
                precision_interp = np.interp(common_x, recall[::-1], precision[::-1])
                y_list.append(precision_interp)
                metrics.append(result.get('aupr', 0.0))

            mean_y = np.mean(y_list, axis=0)
            std_y = np.std(y_list, axis=0)
            mean_metric = np.mean(metrics)

            return common_x, mean_y, std_y, mean_metric

    def plot_5fold_curves(self, fold_results, model_name='MVHD-CLGAE',
                          timestamp=None, show_mean=True):
        if timestamp is None:
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

        roc_curves = []
        pr_curves = []

        for result in fold_results:
            y_true = result['y_true']
            y_scores = result['y_scores']
            fold_id = result.get('fold', len(roc_curves) + 1)

            fpr, tpr, _ = roc_curve(y_true, y_scores)
            auc = result.get('auc', 0.0)
            roc_curves.append({
                'fpr': fpr,
                'tpr': tpr,
                'auc': auc,
                'fold': fold_id
            })

            precision, recall, _ = precision_recall_curve(y_true, y_scores)
            aupr = result.get('aupr', 0.0)
            pr_curves.append({
                'precision': precision,
                'recall': recall,
                'aupr': aupr,
                'fold': fold_id
            })

        if show_mean:
            mean_fpr, mean_tpr, std_tpr, mean_auc = self._compute_mean_curve(
                fold_results, curve_type='roc'
            )
            mean_recall, mean_precision, std_precision, mean_aupr = self._compute_mean_curve(
                fold_results, curve_type='pr'
            )

        fig, (ax_roc, ax_pr) = plt.subplots(
            1, 2, figsize=(12.6, 5.2), constrained_layout=True
        )
        ax_roc.set_box_aspect(1)
        ax_pr.set_box_aspect(1)

        for idx, d in enumerate(roc_curves):
            ax_roc.plot(d['fpr'], d['tpr'], lw=1.3, marker=None,
                        color=self.FOLD_COLORS[idx % len(self.FOLD_COLORS)],
                        alpha=0.6,
                        label=f"Fold {d['fold']} (AUC={d['auc']:.2f}%)")

        if show_mean:
            ax_roc.plot(mean_fpr, mean_tpr, lw=3.0,
                        color=self.MEAN_COLOR, linestyle='-',
                        label=f"Mean (AUC={mean_auc:.2f}%)",
                        zorder=10)

        ax_roc.set_xlabel('False Positive Rate')
        ax_roc.set_ylabel('True Positive Rate')
        ax_roc.set_title(f'ROC Curves (5-fold) - {model_name}')
        ax_roc.grid(True, ls='--', alpha=0.35)
        ax_roc.set_xlim(0, 1)
        ax_roc.set_ylim(0.0, 1.01)

        x1_roc, x2_roc = 0.00, 0.20
        y1_roc, y2_roc = 0.80, 1.00
        axins_roc = inset_axes(ax_roc, width="46%", height="46%",
                               loc="lower right", borderpad=2)
        for idx, d in enumerate(roc_curves):
            axins_roc.plot(d['fpr'], d['tpr'], lw=1.0,
                           color=self.FOLD_COLORS[idx % len(self.FOLD_COLORS)],
                           alpha=0.6)
        if show_mean:
            axins_roc.plot(mean_fpr, mean_tpr, lw=2.5,
                           color=self.MEAN_COLOR, zorder=10)
        axins_roc.set_xlim(x1_roc, x2_roc)
        axins_roc.set_ylim(y1_roc, y2_roc)
        axins_roc.grid(True, ls='--', alpha=0.30)

        ax_roc.add_patch(plt.Rectangle(
            (x1_roc, y1_roc), x2_roc - x1_roc, y2_roc - y1_roc,
            fill=False, lw=1.0, ls=(0, (3, 2)), ec=self.BASELINE_COLOR
        ))

        fig.add_artist(ConnectionPatch(
            xyA=(x2_roc, y1_roc), coordsA=ax_roc.transData,
            xyB=(0.0, 1.0), coordsB=axins_roc.transAxes,
            lw=1.0, ls=(0, (3, 2)), color=self.BASELINE_COLOR
        ))

        for idx, d in enumerate(pr_curves):
            ax_pr.plot(d['recall'], d['precision'], lw=1.3, marker=None,
                       color=self.FOLD_COLORS[idx % len(self.FOLD_COLORS)],
                       alpha=0.6,
                       label=f"Fold {d['fold']} (AUPR={d['aupr']:.2f}%)")

        if show_mean:
            ax_pr.plot(mean_recall, mean_precision, lw=3.0,
                       color=self.MEAN_COLOR, linestyle='-',
                       label=f"Mean (AUPR={mean_aupr:.2f}%)",
                       zorder=10)

        ax_pr.set_xlabel('Recall')
        ax_pr.set_ylabel('Precision')
        ax_pr.set_title(f'PR Curves (5-fold) - {model_name}')
        ax_pr.grid(True, ls='--', alpha=0.35)
        ax_pr.set_xlim(0, 1)
        ax_pr.set_ylim(0, 1.01)

        x1_pr, x2_pr = 0.80, 1.00
        y1_pr, y2_pr = 0.80, 1.01
        axins_pr = inset_axes(
            ax_pr, width="46%", height="46%",
            loc="lower left", borderpad=0,
            bbox_to_anchor=(0.10, 0.06, 1, 1),
            bbox_transform=ax_pr.transAxes
        )
        for idx, d in enumerate(pr_curves):
            axins_pr.plot(d['recall'], d['precision'], lw=1.0,
                          color=self.FOLD_COLORS[idx % len(self.FOLD_COLORS)],
                          alpha=0.6)
        if show_mean:
            axins_pr.plot(mean_recall, mean_precision, lw=2.5,
                          color=self.MEAN_COLOR, zorder=10)
        axins_pr.set_xlim(x1_pr, x2_pr)
        axins_pr.set_ylim(y1_pr, y2_pr)
        axins_pr.grid(True, ls='--', alpha=0.30)

        ax_pr.add_patch(plt.Rectangle(
            (x1_pr, y1_pr), x2_pr - x1_pr, y2_pr - y1_pr,
            fill=False, lw=1.0, ls=(0, (3, 2)), ec=self.BASELINE_COLOR
        ))

        fig.add_artist(ConnectionPatch(
            xyA=(x1_pr, y1_pr), coordsA=ax_pr.transData,
            xyB=(1.0, 1.0), coordsB=axins_pr.transAxes,
            lw=1.0, ls=(0, (3, 2)), color=self.BASELINE_COLOR
        ))

        Y_TICK_LEN = 6.0
        for ax in (ax_roc, ax_pr):
            ax.tick_params(axis='y', which='major', length=Y_TICK_LEN)
            ax.tick_params(axis='y', which='minor', length=Y_TICK_LEN)

        self._add_legend_with_tick_font(
            ax_roc,
            loc='upper right', bbox_to_anchor=(0.98, 0.86),
            handlelength=1.5, handletextpad=0.5, labelspacing=0.3,
            borderpad=0.3, framealpha=0.95, fontsize=9
        )

        self._add_legend_with_tick_font(
            ax_pr,
            loc='upper left', bbox_to_anchor=(0.02, 0.86),
            handlelength=1.5, handletextpad=0.5, labelspacing=0.3,
            borderpad=0.3, framealpha=0.95, fontsize=9
        )

        save_path = os.path.join(
            self.save_dir,
            f'ROC_PR_5fold_{model_name}_{timestamp}.png'
        )
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

        print(f"\n[Plot] Saved ROC+PR curves (with mean curve): {save_path}")

        return save_path

    def _add_legend_with_tick_font(self, ax, fontsize=None, **legend_kwargs):
        ticklabels = ax.get_yticklabels()
        if ticklabels:
            fp = ticklabels[0].get_fontproperties()
            size = fontsize if fontsize else ticklabels[0].get_fontsize()
        else:
            fp = ax.yaxis.label.get_fontproperties()
            size = fontsize if fontsize else ax.yaxis.label.get_fontsize()

        return ax.legend(
            prop=FontProperties(
                family=fp.get_family(),
                style=fp.get_style(),
                weight=fp.get_weight(),
                size=size
            ),
            **legend_kwargs
        )

    def plot_comparison_curves(self, results_dict, timestamp=None):
        if timestamp is None:
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

        fig, (ax_roc, ax_pr) = plt.subplots(
            1, 2, figsize=(12.6, 5.2), constrained_layout=True
        )
        ax_roc.set_box_aspect(1)
        ax_pr.set_box_aspect(1)

        colors = plt.cm.tab10(np.linspace(0, 1, len(results_dict)))

        for idx, (model_name, fold_results) in enumerate(results_dict.items()):
            all_fpr = np.linspace(0, 1, 100)
            tpr_list = []

            for result in fold_results:
                fpr, tpr, _ = roc_curve(result['y_true'], result['y_scores'])
                tpr_interp = np.interp(all_fpr, fpr, tpr)
                tpr_list.append(tpr_interp)

            mean_tpr = np.mean(tpr_list, axis=0)
            mean_auc = np.mean([r['auc'] for r in fold_results])

            ax_roc.plot(all_fpr, mean_tpr, lw=2, color=colors[idx],
                        label=f"{model_name} (AUC={mean_auc:.2f}%)")

            all_recall = np.linspace(0, 1, 100)
            precision_list = []

            for result in fold_results:
                precision, recall, _ = precision_recall_curve(
                    result['y_true'], result['y_scores']
                )
                precision_interp = np.interp(
                    all_recall, recall[::-1], precision[::-1]
                )
                precision_list.append(precision_interp)

            mean_precision = np.mean(precision_list, axis=0)
            mean_aupr = np.mean([r['aupr'] for r in fold_results])

            ax_pr.plot(all_recall, mean_precision, lw=2, color=colors[idx],
                       label=f"{model_name} (AUPR={mean_aupr:.2f}%)")

        ax_roc.set_xlabel('False Positive Rate')
        ax_roc.set_ylabel('True Positive Rate')
        ax_roc.set_title('ROC Curves (Model Comparison)')
        ax_roc.grid(True, ls='--', alpha=0.35)
        ax_roc.set_xlim(0, 1)
        ax_roc.set_ylim(0, 1.01)
        ax_roc.legend()

        ax_pr.set_xlabel('Recall')
        ax_pr.set_ylabel('Precision')
        ax_pr.set_title('PR Curves (Model Comparison)')
        ax_pr.grid(True, ls='--', alpha=0.35)
        ax_pr.set_xlim(0, 1)
        ax_pr.set_ylim(0, 1.01)
        ax_pr.legend()

        save_path = os.path.join(
            self.save_dir,
            f'ROC_PR_comparison_{timestamp}.png'
        )
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

        print(f"\n[Plot] Saved model comparison curves: {save_path}")

        return save_path


if __name__ == '__main__':
    print("=" * 80)
    print("Testing ROC/PR plotting module (with mean curve)")
    print("=" * 80)

    np.random.seed(42)
    fold_results = []

    for fold in range(1, 6):
        n_samples = 500
        y_true = np.random.binomial(1, 0.5, n_samples)
        y_scores = y_true + np.random.randn(n_samples) * 0.3
        y_scores = 1 / (1 + np.exp(-y_scores))

        from sklearn.metrics import roc_auc_score, average_precision_score

        auc = roc_auc_score(y_true, y_scores) * 100
        aupr = average_precision_score(y_true, y_scores) * 100

        fold_results.append({
            'y_true': y_true,
            'y_scores': y_scores,
            'auc': auc,
            'aupr': aupr,
            'fold': fold
        })

    plotter = ROCPRPlotter(save_dir='test_figs')
    save_path = plotter.plot_5fold_curves(
        fold_results,
        model_name='Test_Model',
        timestamp='test',
        show_mean=True
    )

    print(f"\n[Passed] Test passed!")
    print(f"  Figure saved: {save_path}")