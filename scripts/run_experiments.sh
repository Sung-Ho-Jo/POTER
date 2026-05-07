#!/bin/bash
# POTER – Example experiment commands for all four datasets.
#
# Usage: edit DATA_DIR and FEATURE_DIR below, then run individual blocks.
#
# Step 1: extract features (run once per dataset)
# Step 2: run POTER

DATA_DIR=/path/to/data
FEATURE_DIR=/path/to/features

# ── Step 1: Feature extraction ────────────────────────────────────────────────

python extract_features.py \
    --dataset CUB --data_dir $DATA_DIR \
    --ref_per_group 0:0,1:466,2:133,3:0 \
    --output $FEATURE_DIR/cub_features.pt --gpu 0

python extract_features.py \
    --dataset CelebA --data_dir $DATA_DIR \
    --ref_per_group 0:8535,1:0,2:0,3:182 \
    --output $FEATURE_DIR/celeba_features.pt --gpu 0

python extract_features.py \
    --dataset CMNIST --data_dir $DATA_DIR \
    --ref_per_group 0:2591,1:0,2:0,3:2431 \
    --output $FEATURE_DIR/cmnist_features.pt --gpu 0

python extract_features.py \
    --dataset CivilComments --data_dir $DATA_DIR \
    --ref_per_group 0:0,1:5000,2:2111,3:0 \
    --output $FEATURE_DIR/civilcomments_features.pt --gpu 0

# ── Step 2: POTER training ────────────────────────────────────────────────────

# Waterbirds
python run_expt.py \
    --dataset CUB --data_dir $DATA_DIR \
    --ot_cache $FEATURE_DIR/cub_features.pt \
    --f_power 0.1 --clip_cap 20 \
    --n_epochs 300 --batch_size 128 --lr 1e-5 --weight_decay 1.0 \
    --augment_data \
    --seed 1 --gpu 0

# CelebA
python run_expt.py \
    --dataset CelebA --data_dir $DATA_DIR \
    --ot_cache $FEATURE_DIR/celeba_features.pt \
    --f_power 0.2 --clip_cap 20 \
    --n_epochs 50 --batch_size 128 --lr 1e-5 --weight_decay 1e-2 \
    --augment_data \
    --seed 1 --gpu 0

# ColoredMNIST
python run_expt.py \
    --dataset CMNIST --data_dir $DATA_DIR \
    --ot_cache $FEATURE_DIR/cmnist_features.pt \
    --f_power 0.6 --clip_cap 3 \
    --n_epochs 300 --batch_size 128 --lr 1e-2 --weight_decay 1e-2 \
    --seed 1 --gpu 0

# CivilComments
python run_expt.py \
    --dataset CivilComments --data_dir $DATA_DIR \
    --ot_cache $FEATURE_DIR/civilcomments_features.pt \
    --f_power 0.7 --clip_cap 2 \
    --n_epochs 3 --batch_size 32 --lr 1e-5 --weight_decay 1e-2 \
    --seed 1 --gpu 0
