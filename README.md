# ProMeta

## About

This repository contains the code and resources of the following paper:

ProMeta: A meta-learning framework for robust disease diagnosis and prediction from plasma proteomics

Overview of the ProMeta framework

ProMeta is a few-shot meta-learning framework specifically designed for proteomics data analysis. Addressing the limitation of deep learning in data-scarce scenarios (rare diseases), ProMeta leverages a "learning-to-learn" paradigm to adapt to new disease tasks using as few as 4 patient samples.

<p align="center">
<img  src="ProMeta.png"> 
</p>

## 🚀 Features

* **Robust Few-Shot Adaptation**: Outperforms transfer learning and traditional ML baselines by ~24.6% in 4-shot scenarios (2 cases, 2 controls).
* **Knowledge-Guided Encoding**: Uses ConsensusPathDB (CPDB) to map proteins to biological pathways, creating robust functional tokens.
* **Dual-View Representation**: Handles the entire proteome by processing pathway-mapped proteins and unmapped proteins (via auxiliary tokens) in parallel streams.
* **Meta-SGD Optimization**: Learns both the model initialization and task-specific learning rates, enabling rapid convergence on novel diseases.

## 📂 Project Structure

```text
├── data_preprocess/
│   ├── preprocess_data_prevalent.ipynb       # Code for prevalent disease data preproocess
│   ├── preprocess_data_incident.ipynb        # Code for incident disease data preproocess
├── resource/
│   └── CPDB_pathways_genes.tab       # Pathway knowledge database
└── ProMeta/
    ├── main.py             # Entry point for training and evaluation
    ├── config.py           # Configuration and argument parsing
    ├── dataset.py          # Data loaders and Pathway Mask generation
    ├── model.py            # ProMeta model architecture and Loss functions
    ├── utils.py            # Metrics, logging, and helper functions
    └── run_ProMeta.sh      # Shell script to run experiments
```

## 🛠️ Setup Environment

Setup the required environment using `environment.yml` with Anaconda. While in the project directory run:
```
    conda env create
```
Activate the environment
```
    conda activate ProMeta
```

## 🏃 Run ProMeta

To reproduce the experiments described in the pap
er (e.g., 4-shot or 32-shot adaptation), navigate to the source directory and execute the run script:
```
cd ProMeta
bash run_ProMeta.sh
```

You can also run `main.py` directly with custom arguments:
```
python main.py \
    --data_dir "../data/out/" \
    --proteomics_csv "../data/proteomics.csv" \
    --cpdb_path "../resource/CPDB_pathways_genes.tab" \
    --support_size 4 \
    --batch_size 8 \
    --outer_lr 1e-4 \
    --inner_lr 0.005
```
