#!/bin/bash

# ==========================================
# ProMeta Running Script (Modular Version)
# ==========================================

DATA_DIR="/home/dataset-assist-0/yaosen/lihan/ght/Prophet-Meta-temp/data/out/"
PROTEOMICS_CSV="/home/dataset-assist-0/yaosen/lihan/ght/Prophet-Meta-temp/Prophet/data/preprocessed_proteomics_data.csv"
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