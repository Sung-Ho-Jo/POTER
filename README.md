# POTER: Optimal Transport Reweighting for Robust Learning under Spurious Correlations and Label Noise

## Overview

POTER derives sample importance weights from the dual potentials of
class-conditioned optimal transport (OT) between the training distribution and
a reference distribution constructed from limited validation-set group annotations.
These weights reflect distributional alignment rather than losses from a
preliminary classifier, allowing POTER to emphasize bias-conflicting samples
and downweight strongly bias-aligned or mislabeled samples. It improves
robustness to spurious correlations and label noise without requiring training-set
group annotations. By constructing sample weights before model training, POTER
requires only a single standard ERM training stage, moving beyond pipelines that
rely on preliminary downstream-task training followed by full-model or last-layer
retraining.

## Requirements

```
torch >= 1.12
torchvision
numpy
scipy
pandas
tqdm
Pillow
scikit-learn
transformers
pytorch-transformers
```

## Dataset Preparation

Download and place datasets under a common root directory:

| Dataset       | Subdirectory    | Source |
|---------------|-----------------|--------|
| Waterbirds    | `cub/`          | [Sagawa et al., 2020](https://github.com/kohpangwei/group_DRO) |
| CelebA        | `celebA/`       | [Liu et al., 2015](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) |
| ColoredMNIST  | `CMNIST/`       | [Arjovsky et al., 2019](https://github.com/facebookresearch/InvariantRiskMinimization) |
| CivilComments | `civilcomments/`| [Borkan et al., 2019](https://wilds.stanford.edu/) |

## Reproducing Results

### Step 1: Extract Features

Feature extraction needs to be done once per dataset. `--ref_per_group g:n,...` specifies how many validation samples to draw from each group for the OT reference distribution (0 = exclude that group).

```bash
python extract_features.py --dataset CUB --data_dir /data \
    --ref_per_group 0:0,1:466,2:133,3:0 \
    --output features/cub_features.pt --gpu 0

python extract_features.py --dataset CelebA --data_dir /data \
    --ref_per_group 0:8535,1:0,2:0,3:182 \
    --output features/celeba_features.pt --gpu 0

python extract_features.py --dataset CMNIST --data_dir /data \
    --ref_per_group 0:2591,1:0,2:0,3:2431 \
    --output features/cmnist_features.pt --gpu 0

python extract_features.py --dataset CivilComments --data_dir /data \
    --ref_per_group 0:0,1:5000,2:2111,3:0 \
    --output features/civilcomments_features.pt --gpu 0
```

### Step 2: Run POTER

```bash
# Waterbirds
python run_expt.py \
    --dataset CUB --data_dir /data \
    --ot_cache features/cub_features.pt \
    --f_power 0.1 --clip_cap 20 \
    --n_epochs 300 --lr 1e-5 --weight_decay 1.0 --batch_size 128 \
    --augment_data \
    --seed 1 --gpu 0

# CelebA
python run_expt.py \
    --dataset CelebA --data_dir /data \
    --ot_cache features/celeba_features.pt \
    --f_power 0.2 --clip_cap 20 \
    --n_epochs 50 --lr 1e-5 --weight_decay 1e-2 --batch_size 128 \
    --augment_data \
    --seed 1 --gpu 0

# ColoredMNIST
python run_expt.py \
    --dataset CMNIST --data_dir /data \
    --ot_cache features/cmnist_features.pt \
    --f_power 0.6 --clip_cap 3 \
    --n_epochs 300 --lr 1e-2 --weight_decay 1e-2 --batch_size 128 \
    --seed 1 --gpu 0

# CivilComments
python run_expt.py \
    --dataset CivilComments --data_dir /data \
    --ot_cache features/civilcomments_features.pt \
    --f_power 0.7 --clip_cap 2 \
    --n_epochs 3 --lr 1e-5 --weight_decay 1e-2 --batch_size 32 \
    --seed 1 --gpu 0
```

To run all datasets at once, edit `DATA_DIR` and `FEATURE_DIR` in `scripts/run_experiments.sh` and execute it.

## Output

Each run creates a timestamped directory under `--log_dir/` containing:
- `log.txt` — training log
- `train.csv`, `val.csv`, `test.csv` — per-epoch group accuracies
- `val_16group.csv`, `test_16group.csv` — 16-group accuracy (CivilComments only)

## ERM Baseline

Omit `--ot_cache` to train without OT reweighting:

```bash
python run_expt.py --dataset CUB --data_dir /data \
    --n_epochs 300 --lr 1e-5 --weight_decay 1.0 --batch_size 128 --seed 1
```

## Label Noise

Two noise types are supported and can be enabled through `run_expt.py`.

For **symmetric noise**, re-extract features with both bias-aligned and
bias-conflicting groups represented within each class in the OT reference set.
For **subgroup noise**, use the same `--ref_per_group` setting as the clean
benchmark setting. The table below lists the reference-set configurations used
for the datasets in our label-noise experiments.

| Dataset       | Symmetric noise               |
|---------------|-------------------------------|
| Waterbirds    | `0:466,1:466,2:133,3:133`     |
| CelebA        | `0:8276,1:8276,2:182,3:182`   |
| CivilComments | `0:2500,1:2500,2:2111,3:2111` |

**Symmetric noise** — flip a fixed fraction of each class independently:

```bash
python run_expt.py \
    --dataset CUB --data_dir /data \
    --ot_cache features/cub_features.pt \
    --f_power 0.1 --clip_cap 20 \
    --n_epochs 300 --lr 1e-5 --weight_decay 1.0 --batch_size 128 \
    --augment_data \
    --noise_type symmetric --noise_fraction 0.2 --noise_seed 0 \
    --seed 1 --gpu 0
```

**Subgroup noise** — flip labels only within specified groups:

```bash
python run_expt.py \
    --dataset CUB --data_dir /data \
    --ot_cache features/cub_features.pt \
    --f_power 0.1 --clip_cap 20 \
    --n_epochs 300 --lr 1e-5 --weight_decay 1.0 --batch_size 128 \
    --augment_data \
    --noise_type subgroup --group_flip_fractions 1:0.2,2:0.2 --noise_seed 0 \
    --seed 1 --gpu 0
```

`--group_flip_fractions` format: `g0:f0,g1:f1,...` where `g` is group id and `f` is flip fraction (0–1).

## File Structure

```
POTER/
├── run_expt.py          # Main experiment script
├── extract_features.py  # Feature extraction (run once per dataset)
├── ot_reweight.py       # POTER core: OT potentials + sample weights
├── noise.py             # Label noise injection (symmetric / subgroup)
├── train.py             # Training loop
├── models.py            # Model attribute registry
├── loss.py              # Group-DRO loss computer
├── utils.py             # Logger, CSVBatchLogger, set_seed
├── data/
│   ├── data.py                  # Dataset dispatcher
│   ├── dro_dataset.py           # DRODataset wrapper (weighted sampling)
│   ├── confounder_dataset.py    # Base ConfounderDataset class
│   ├── confounder_utils.py      # prepare_confounder_data
│   ├── cub_dataset.py           # Waterbirds
│   ├── celebA_dataset.py        # CelebA
│   ├── cmnist_dataset.py        # ColoredMNIST
│   └── civilcomments_dataset.py # CivilComments
└── scripts/
    └── run_experiments.sh
```
