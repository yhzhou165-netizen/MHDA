# MHDA

Code for circRNA–disease association prediction.

## Code structure

```text
code/
├── Train_MHDA_CLGAE_Regularized.py  # Main training script and five-fold evaluation
├── Model_MHDA_CLGAE.py             # MHDA model
├── MultiView_Layers.py             # Multi-view graph learning and fusion
├── Heterogeneous_Layers.py         # Heterogeneous graph attention
├── Layers.py                      # Graph attention and decoder layers
├── Contrastive_Loss.py            # Contrastive learning objective
├── Adj_preprocess.py              # Fold-specific similarities and graph inputs
├── Preprocessing.py               # Data splitting and graph normalization
├── Optimizer.py                   # Evaluation metrics and threshold selection
├── Plot_ROC_PR.py                 # ROC and precision–recall curves
├── Run_MHDA_Mean_View_Ablation.py  # Mean-view fusion ablation
├── Validate_Fold_Pipeline.py      # Fold isolation checks
└── fig/                          # Stored figures
```

## Input data

The training script reads two files from `data/`:

- `Association Matrixs.xlsx`: circRNA–disease associations.
- `D_SH.mat`: disease semantic similarity.

## Run

From the repository root:

```bash
python code/Train_MHDA_CLGAE_Regularized.py full
```

By default, each run saves its results in a separate folder under `results/`, including evaluation metrics, predictions, model checkpoints, and ROC/PR figures.
