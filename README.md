# MHDA: A Multi-View Heterogeneous Dual-Stream Graph Autoencoder for Predicting circRNA-Disease Associations

> **Paper**: *A Multi-View Heterogeneous Dual-Stream Graph Autoencoder for Predicting circRNA-Disease Associations*  
> **Authors**: Yu-Hao Zhao, Lei Wang, Zhu-Hong You, Chang-Qing Yu

---

## Overview

MHDA is a computational framework for predicting circRNA-disease associations (CDAs). It integrates four modules into a unified architecture:

1. **Multi-View Graph Attention Encoder** — parallel encoding across Jaccard, Cosine, and integrated similarity views with adaptive fusion
2. **Heterogeneous Attention Enhancement** — joint modeling of intra-type (circRNA–circRNA, disease–disease) and inter-type (circRNA–disease) interactions
3. **Dual-Stream Gated Decoder** — reconstructs associations from both structural (element-wise product) and semantic (concatenation) perspectives with a learned fusion gate
4. **Contrastive Learning Constraint** — improves consistency, discriminability, and robustness of learned embeddings

**Performance on CircR2Disease (5-fold CV):** AUC = 96.28%, AUPR = 96.08%

---

## Requirements

```
Python >= 3.7
tensorflow >= 2.x  (used in v1 compatibility mode)
numpy
scipy
pandas
scikit-learn
matplotlib
openpyxl
```

```bash
pip install tensorflow numpy scipy pandas scikit-learn matplotlib openpyxl
```

---

## Data Preparation

Place the following files under `../data/`:

| File | Description |
|---|---|
| `Association Matrixs.xlsx` | circRNA-disease association matrix (sheet index 2) |
| `integrated_circ_sim.mat` | Integrated circRNA similarity matrix |
| `integrated_dise_sim.mat` | Integrated disease similarity matrix |

The model additionally computes Jaccard and Cosine similarity views from the association matrix automatically.

---

## Usage

```bash
python Train_MHDA_CLGAE_Regularized.py
```

Runs 5-fold cross-validation and outputs AUC, AUPR, F1, Precision, Recall, Accuracy per fold and as mean ± std. ROC/PR curve figures are saved to `./figs/`.

---

## File Structure

```
├── Train_MHDA_CLGAE_Regularized.py   # Main training entry
├── Model_MHDA_CLGAE.py               # Model definition
├── MultiView_Layers.py               # Multi-view GAT + view attention fusion
├── Heterogeneous_Layers.py           # Heterogeneous GAT (circRNA ↔ disease)
├── Layers.py                         # Base GAT, MLP decoder, dual-stream decoder
├── Contrastive_Loss.py               # Contrastive loss functions
├── Adj_preprocess.py                 # Similarity matrix loading & multi-view construction
├── Preprocessing.py                  # Graph normalization & 5-fold splitting
├── Optimizer.py                      # Metric computation
└── Plot_ROC_PR.py                    # ROC/PR curve plotting
```

---

## Citation

If you use this code, please cite our paper (citation info to be updated upon publication).

---

## License

For academic research use only.