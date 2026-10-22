"""One-click PyCharm entry for the controlled equal-view fusion ablation."""

import os
import time

from Train_MHDA_CLGAE_Regularized import train_mhda_clgae


def main():
    code_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(code_dir)

    print("=" * 80)
    print("DAFCDA ablation: fixed equal-weight multi-view fusion")
    print("Only the node-level view weights are changed to 1/3, 1/3, 1/3.")
    print("All other settings are identical to the full configuration.")
    print("=" * 80)

    start_time = time.time()
    summary = train_mhda_clgae(model_config_name='mean_view')
    elapsed = time.time() - start_time

    print("\nMEAN-VIEW ABLATION FINISHED")
    print("Result file: {}".format(
        os.path.join(summary['output_dir'], "results_mean_view.json")
    ))
    print("Elapsed time: {:.2f}s ({:.2f}min)".format(
        elapsed, elapsed / 60.0
    ))


if __name__ == '__main__':
    main()
