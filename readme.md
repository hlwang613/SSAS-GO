# SSAS-GO: Structure-Sequence Adaptive Synergy for Protein Function Prediction

This repository contains the official PyTorch implementation of **SSAS-GO** (Structure-Sequence Adaptive Synergy for Gene Ontology).

SSAS-GO is a multimodal deep learning framework designed for large-scale protein function prediction. To address the "negative transfer" problem between structure-dependent tasks (e.g., Biological Process) and sequence-dependent tasks (e.g., Molecular Function), SSAS-Net introduces a **Task-Aware Cross-Modal Gating Mechanism**. It dynamically balances representations from a **Dual-Stream GNN** (capturing global and local structural topology) and a **Multi-Scale Motif Block** (capturing sequential motifs).

## Dependencies and Environment Setup

This project uses PyTorch, Deep Graph Library (DGL). We provide an `environment.yml` file for easy setup via Conda.

```bash
# Clone the repository
git clone https://github.com/yourusername/SSAS-GO.git
cd SSAS-GO

# Create the conda environment from the provided yaml file
conda env create -f environment.yml

# Activate the environment
conda activate SSAS

```

## Data Preparation

Data set can be downloaded from [HEAL](https://github.com/ZhonghuiGu/HEAL)

Before training, ensure your data is placed in the directories specified in `config.py`. The required files include:

1. **GO Annotations**:
* `nrPDB-GO_2019.06.18_annot.tsv` (for PDB dataset)
* `nrSwiss-Model-GO_annot.tsv` (for AF2/Swiss-Model augmented dataset)


2. **Preprocessed Graph Data**:
* `{split}_graph.pt` and `{split}_pdbch.pt` (where split is `train`, `val`, `test`, or `AF2test`). These should be placed in the `processed_data_dir` defined in `config.py`.


3. **Ontology & Evaluation Files**:
* `go-basic.obo` (Gene Ontology structure)
* `ic_count.pkl` (Information Content pre-calculated from the training set for Smin evaluation)



You can easily modify file paths, batch size, learning rates, and loss functions (BCE, Focal Loss, Asymmetric Loss) in `config.py`.

## Training

To train the SSAS-Net model, run `train.py`. The script automatically monitors validation metrics and saves the Top-K best model checkpoints.

**Arguments:**

* `--task`: GO branch to predict (`bp`, `mf`, `cc`)
* `--device`: GPU device ID (e.g., `0`)
* `--suffix`: Checkpoint naming suffix (e.g., `CLAF`)
* `--AF2model`: Whether to use AlphaFold2 augmented data for training (`True` or `False`)

**Example Commands:**

```bash
# Train on Molecular Function (MF) using only PDB data
python train.py --device 0 --task mf --batch_size 64 --suffix CL --AF2model False

# Train on Biological Process (BP) using PDB + AF2 augmented data
python train.py --device 0 --task bp --batch_size 64 --suffix CLAF --AF2model True

```

## Testing and Evaluation

To evaluate the trained models, run `test.py`. The script will automatically load the Top-K checkpoints, merge their predictions (ensemble), calculate CAFA-style metrics (Protein-centric , Macro/Micro AUPR, ), and save the raw prediction results to a `.pkl` file.

**Example Commands:**

```bash
# Test on the standard PDB test set for CC task
python test.py --task cc --model model_cc_CL --AF2test False --test_type test

# Test on the AF2 specific test set for MF task
python test.py --task mf --model model_mf_CLAF --AF2test True --test_type AF2test

```

Evaluation metrics are strictly implemented based on the CAFA challenge protocols, including label propagation upward to ancestors in the GO graph.

## Repository Structure

* `network.py`: Core model architecture including `SSAS_Net`, `GCN_Parallel`, `MultiScaleMotifBlock`, and `transformer_block`.
* `train.py`: Main training loop with early stopping, Top-K checkpoint saving, and loss function selection.
* `test.py`: Inference script supporting Top-K model ensembling and evaluation metric logging.
* `evaluation.py`: CAFA-style metrics calculation (, AUPR, ) and GO graph label propagation.
* `graph_data.py`: Custom `Dataset` class (`GoTermDataset`) handling label loading
* `config.py`: Global configuration and hyperparameter settings.
* `utils.py`: Helper functions, PDB/FASTA parsers, result merging.
