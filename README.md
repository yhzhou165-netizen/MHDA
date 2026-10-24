# MHDA

## Code structure

```text
code/
├── Train_MHDA_CLGAE_Regularized.py  # Training and five-fold evaluation
├── Model_MHDA_CLGAE.py             # MHDA model
├── MultiView_Layers.py             # Multi-view graph learning and fusion
├── Heterogeneous_Layers.py         # Heterogeneous graph attention
├── Layers.py                       # Graph attention and decoder layers
├── Contrastive_Loss.py             # Contrastive learning objective
├── Adj_preprocess.py               # Fold-specific similarities and graph inputs
├── Preprocessing.py                # Data splitting and graph normalization
├── Optimizer.py                    # Evaluation metrics and threshold selection
└── Plot_ROC_PR.py                  # ROC and precision–recall curves
```

Input files are read from `data/Association Matrixs.xlsx` and `data/D_SH.mat`.
Training outputs are saved in `results/`.
