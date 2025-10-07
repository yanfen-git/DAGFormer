# DAGFormer

## Abstract
Developing computational methods for single-cell drug response prediction deepens our understanding of tumor heterogeneity and uncovers resistance mechanisms critical to improving cancer therapy. However, current approaches struggle to fully capture intratumoral heterogeneity, as bulk RNA sequencing (bulk RNA-seq) obscures heterogeneity across individual cells, while single-cell RNA sequencing (scRNA-seq) remains constrained by limited throughput and high cost. Current approaches integrating bulk and scRNA-seq data frequently encounter batch effects, impairing robust knowledge transfer. Moreover, most existing methods overlook the role of intercellular interactions, treating cells as isolated entities. To overcome these limitations, we propose DAGFormer, a Graph-based Domain Adaptation framework that integrates bulk and scRNA-seq data for predicting single-cell drug responses. DAGFormer constructs cellular neighbor graphs using diverse topological strategies and employs Graph Domain Adaptation (GDA) to bridge graph-level distribution gaps between bulk and single-cell RNA-seq data. A dual-domain decoder further disentangles shared and modality-specific representations, preserving both general and unique biological signals. Benchmarking DAGFormer on ten independent scRNA-seq datasets demonstrated its superior performance compared to existing methods, underscoring its effectiveness and robustness in cancer drug response prediction.

![DAGFormer](./my_model.png)

## Dataset
The data folder includes different drug data.

### Source Domain Data (Bulk RNA-seq)
- **GDSC (Genomics of Drug Sensitivity in Cancer)** database.
- Drug sensitivity and resistance data with IC50 values transformed into binary labels.

### Target Domain Data (scRNA-seq)
- Contains single-cell RNA-seq data for predicting drug responses at a high resolution.

The original and preprocessed datasets can be found at:  
[Google Drive Dataset Link](https://drive.google.com/drive/folders/1y4_xWRmhIs1noyDmWz9CKL1oDWLGkO2Y?usp=drive_link)

## Training
To train DAGFormer on your dataset, you can run the following command:

```bash
python main_GT.py --cuda 0 --drug Gefitinib

## Requirements
```bash
matplotlib==3.1.1
ipdb==0.12.3
torch==1.7.1
torchvision==0.8.2
numpy==1.21.0
scipy==1.7.3
texttable==1.6.2
networkx==2.4
numba==0.48.0
Pillow==7.0.0
scikit-learn==0.22.1
tensorboardX==2.0
tqdm==4.42.1
gensim==3.8.3
