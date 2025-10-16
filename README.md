# DAGFormer

> **DAGFormer: A Graph-based Domain Adaptation Framework for Single-cell Cancer Drug Response Prediction**

## Abstract
Developing computational methods for single-cell drug response prediction deepens our understanding of tumor heterogeneity and uncovers resistance mechanisms critical to improving cancer therapy. However, current approaches struggle to fully capture intratumoral heterogeneity, as bulk RNA sequencing (bulk RNA-seq) obscures heterogeneity across individual cells, while single-cell RNA sequencing (scRNA-seq) remains constrained by limited throughput and high cost. Current approaches integrating bulk and scRNA-seq data frequently encounter batch effects, impairing robust knowledge transfer. Moreover, most existing methods overlook the role of intercellular interactions, treating cells as isolated entities. To overcome these limitations, we propose DAGFormer, a Graph-based Domain Adaptation framework that integrates bulk and scRNA-seq data for predicting single-cell drug responses. DAGFormer constructs cellular neighbor graphs using diverse topological strategies and employs Graph Domain Adaptation (GDA) to bridge graph-level distribution gaps between bulk and single-cell RNA-seq data. A dual-domain decoder further disentangles shared and modality-specific representations, preserving both general and unique biological signals. Benchmarking DAGFormer on ten independent scRNA-seq datasets demonstrated its superior performance compared to existing methods, underscoring its effectiveness and robustness in cancer drug response prediction.

![DAGFormer](./my_model.png)

## Dataset
The model utilizes two distinct domains for transfer learning: the **Bulk RNA-seq Source Domain ($\mathbf{D}_S$)** for training knowledge, and the **Single-cell RNA-seq Target Domain ($\mathbf{D}_T$)** for high-resolution prediction.

### 1. Source Domain ($\mathbf{D}_S$): Bulk $\text{RNA-seq}$ Data

* **Source:** **GDSC (Genomics of Drug Sensitivity in Cancer)** database.
* **Content:** Contains bulk gene expression profiles and pharmacological response data ($\text{IC}_{50}$ values) across numerous cancer cell lines.
* **Preprocessing & Labeling:** Continuous $\text{IC}_{50}$ values are transformed into **binary labels** ('sensitive' or 'resistant') using the **LOBIco** algorithm.

### 2. Target Domain ($\mathbf{D}_T$): Single-cell $\text{RNA-seq}$ Data

The target domain uses $\text{scRNA-seq}$ datasets (from $\text{CCLE}$, $\text{GSE149215}$, and $\text{GSE108383}$) for single-cell drug response prediction. Labels are derived based on the experimental context:

* **Post-treatment Data (Acquired Resistance):**
    * **Context:** Evaluates cells **after** drug exposure (e.g., Etoposide, PLX4720).
    * **Labeling:** Parental cells (untreated) are classified as **sensitive cells**; cells that survived drug exposure are categorized as **resistant cells**.
* **Pre-treatment Data (Inherent Resistance):**
    * **Context:** Evaluates cells **prior to** drug exposure (e.g., Gefitinib, Sorafenib).
    * **Labeling:** Cells are labeled based on inferred or established drug-resistance markers, assuming the existence of **pre-existing drug-resistant subpopulations**.

All datasets are preprocessed (including $\text{QC}$, normalization, and $\text{Z}$-score standardization) and filtered to retain **only genes shared between the source and target domains**.

The original and preprocessed datasets can be found at:
[Google Drive Dataset Link](https://drive.google.com/drive/folders/1y4_xWRmhIs1noyDmWz9CKL1oDWLGkO2Y?usp=drive_link)


## Requirements
- numpy>=1.24  
- scipy==1.7.3  
- scikit-learn>=1.3  
- torch-geometric==2.6.1  
- torch-scatter==2.1.2+pt24cu121  
- torch-sparse==0.6.18+pt24cu121  

## Computational environment
Experiments were conducted on a workstation equipped with one NVIDIA GeForce RTX 3090 GPU (24 GB VRAM), Intel Xeon 16-core CPU, and 128 GB RAM, running Ubuntu 22.04 LTS with CUDA 12.1 and cuDNN 8.9.
All models were implemented in Python 3.10 using PyTorch 2.4.1 (CUDA 11.8/12.1 compatible) and DGL 2.4.0+cu121.

## Parameter Settings(Default)

The following default hyperparameter values were used for all benchmarking experiments, as defined in `main_GT.py`. Note that $\lambda_{e}$ and the GRL rate utilize dynamic scheduling.

| Parameter (Argument)                   | Description                                                      | Value                                        |
|----------------------------------------|------------------------------------------------------------------|----------------------------------------------|
| Learning Rate (lr)                     | Optimizer initial learning rate                                   | 1e-2                                         |
| Weight Decay                            | L2 regularization applied to the optimizer                       | 5e-4                                         |
| Epochs (n_epoch)                       | Total number of training iterations                               | 300                                          |
| Hidden Dim (hidden)                    | Dimension of the main hidden layer in encoders                    | 512                                          |
| Feature Dim (gfeat)                    | Dimension of the final latent embedding                           | 128                                          |
| Dropout Rate (dropout)                 | Dropout rate used in encoders and decoders                        | 0.6                                          |
| λd (Weight for Domain Adaptation Loss) | Weight for Domain Adaptation Loss (Ldom)                          | 1.0                                          |
| λr (Weight for Reconstruction Loss)    | Weight for Reconstruction Loss (Lrec)                             | 0.3                                          |
| λf (Weight for Difference Loss)        | Weight for Difference Loss (Ldiff)                               | 0.0001                                       |
| Entropy Weight Schedule (λe)           | Weight for Entropy Loss (Lent)                                    | Schedule: epoch/n_epoch × 0.01               |
| GRL Rate Schedule                      | Gradient Reversal Layer (GRL) factor                              | Schedule: min((epoch+1)/n_epoch, 0.05)       |

## Training
To train DAGFormer on your dataset, you can run the following command:

```bash
python main_GT.py --cuda 0 --drug Gefitinib
