#!/bin/bash

# ==========================================
# ProMeta Running Script (Modular Version)
# ==========================================

DATA_DIR="./" #  The DATA_DIR variable in run_ProMeta.sh must point to a directory containing the following 6 pickle (.pkl) files: 
              #  1.term2pre_cases_train.pkl 2.term2pre_controls_train.pkl 3.term2pre_cases_valid.pkl 
              #  4.term2pre_controls_valid.pkl 5.term2pre_cases_test.pkl 6.term2pre_controls_test.pkl
PROTEOMICS_CSV="./preprocessed_proteomics_data.csv"
CPDB_FILE="../resource/CPDB_pathways_genes.tab"
OUTPUT_DIR="./experiments_output"

python main.py \
    --data_dir "$DATA_DIR" \
    --proteomics_csv "$PROTEOMICS_CSV" \
    --cpdb_path "$CPDB_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --gpu_id 2 \
    --random_seed 42 \
    --support_size 32 \
    --batch_size 8 \
    --outer_lr 1e-4 \
    --inner_lr 0.005