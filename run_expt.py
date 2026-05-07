#!/usr/bin/env python
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("PYTHONHASHSEED", "1")
"""
POTER: Optimal Transport Reweighting for Robust Learning
=========================================================
Main experiment script.

Basic usage
-----------
# With POTER reweighting (requires precomputed feature cache):
python run_expt.py \\
    --dataset      CUB \\
    --data_dir     /path/to/data \\
    --ot_cache     /path/to/features/cub_features.pt \\
    --f_power      0.1 \\
    --clip_cap     20 \\
    --n_epochs     300 \\
    --lr           1e-5 \\
    --weight_decay 1.0 \\
    --batch_size   128 \\
    --seed         1 \\
    --gpu          0

# ERM baseline (no --ot_cache):
python run_expt.py --dataset CUB --data_dir /path/to/data \\
    --n_epochs 300 --lr 1e-5 --weight_decay 1.0 --batch_size 128 --seed 1 --gpu 0

Dataset subdirectory layout expected under --data_dir:
    CUB          → <data_dir>/cub/
    CelebA       → <data_dir>/celebA/
    CMNIST       → <data_dir>/CMNIST/
    CivilComments→ <data_dir>/civilcomments/

Feature cache format (.pt file, loaded with torch.load):
    {
        'X_train': ndarray (N_train, d),
        'y_train': ndarray (N_train,),
        'X_val'  : ndarray (N_val,   d),
        'y_val'  : ndarray (N_val,   d),
    }
See extract_features.py to generate cache files.
"""

import os
import argparse
import datetime
import gc

import numpy as np
import torch
import torch.nn as nn
import torchvision

from models import model_attributes
from data.data import dataset_attributes, prepare_data, log_data
from utils import set_seed, Logger, CSVBatchLogger, log_args
from train import train
from ot_reweight import compute_ot_potentials, compute_sample_weights


# ─── Dataset-specific defaults ────────────────────────────────────────────────

_DATASET_CFG = {
    'CUB': {
        'target_name'    : 'waterbird_complete95',
        'confounder_names': ['forest2water2'],
        'model'          : 'resnet50',
        'data_subdir'    : 'cub',
    },
    'CelebA': {
        'target_name'    : 'Blond_Hair',
        'confounder_names': ['Male'],
        'model'          : 'resnet50',
        'data_subdir'    : 'celebA',
    },
    'CMNIST': {
        'target_name'    : '0-4',
        'confounder_names': ['isred'],
        'model'          : 'resnet50',
        'data_subdir'    : 'CMNIST',
    },
    'CivilComments': {
        'target_name'    : 'toxicity',
        'confounder_names': ['identity_any'],
        'model'          : 'bert',
        'data_subdir'    : 'civilcomments',
    },
}

_IDENTITY_COLS = [
    'male', 'female', 'LGBTQ', 'black', 'white',
    'christian', 'muslim', 'other_religions',
]

# ─── Argument parsing ─────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description='POTER: OT Reweighting for Robust Learning',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required
    parser.add_argument('--dataset', required=True, choices=list(_DATASET_CFG.keys()))
    parser.add_argument('--data_dir', required=True,
                        help='Root directory containing dataset subdirectories.')

    # POTER
    ot = parser.add_argument_group('POTER (OT reweighting)')
    ot.add_argument('--ot_cache', default=None,
                    help='Path to precomputed feature cache (.pt). '
                         'Omit to run plain ERM.')
    ot.add_argument('--f_power', type=float, default=0.5,
                    help='Reweighting sharpness α (τ = median(f*) · α).')
    ot.add_argument('--clip_cap', type=float, default=10.0,
                    help='Maximum sample weight after clipping.')
    ot.add_argument('--reg_scale', type=float, default=0.03,
                    help='OT regularization ε = reg_scale · mean(cost_matrix).')

    # Model
    parser.add_argument('--model', default=None,
                        help='Override default model for dataset.')

    # Data augmentation
    parser.add_argument('--augment_data', action='store_true', default=False,
                        help='Enable data augmentation (dataset-specific): '
                             'Waterbirds → crop + flip + color jitter; '
                             'CelebA → crop + flip + rotation.')

    # Label noise
    noise = parser.add_argument_group('Label noise')
    noise.add_argument('--noise_type', default=None, choices=['symmetric', 'subgroup'],
                       help='symmetric: class-conditional flip; '
                            'subgroup: per-group fractions via --group_flip_fractions.')
    noise.add_argument('--noise_fraction', type=float, default=0.0,
                       help='Flip fraction for symmetric noise.')
    noise.add_argument('--group_flip_fractions', default=None,
                       help='Per-group flip fractions for subgroup noise. '
                            'Format: g0:f0,g1:f1,...  e.g. 1:0.3,3:0.6')
    noise.add_argument('--noise_seed', type=int, default=0)

    # Optimization
    opt = parser.add_argument_group('Optimization')
    opt.add_argument('--n_epochs',      type=int,   default=3)
    opt.add_argument('--batch_size',    type=int,   default=32)
    opt.add_argument('--lr',            type=float, default=1e-5)
    opt.add_argument('--weight_decay',  type=float, default=0.01)

    # Misc
    parser.add_argument('--seed',     type=int, default=1)
    parser.add_argument('--gpu',      type=int, default=0)
    parser.add_argument('--log_dir',  default='./logs')
    parser.add_argument('--log_every', type=int, default=10000)

    args = parser.parse_args()

    cfg = _DATASET_CFG[args.dataset]
    if args.model is None:
        args.model = cfg['model']
    args.target_name       = cfg['target_name']
    args.confounder_names  = cfg['confounder_names']
    args.shift_type        = 'confounder'
    args.root_dir          = os.path.join(args.data_dir, cfg['data_subdir'])

    # Dataset-specific augmentation: --augment_data alone is sufficient
    args.augment_photometric = (args.augment_data and args.dataset == 'CUB')
    args.augment_rotate      = (args.augment_data and args.dataset == 'CelebA')

    # Fields required by prepare_data / LossComputer / train
    args.fraction                  = 1.0
    args.reweight_groups           = False
    args.val_fraction              = 0.1
    args.robust                    = False
    args.alpha                     = 0.2
    args.generalization_adjustment = '0.0'
    args.automatic_adjustment      = False
    args.robust_step_size          = 0.01
    args.use_normalized_loss       = False
    args.btl                       = False
    args.hinge                     = False
    args.train_from_scratch        = False
    args.scheduler                 = False
    args.gamma                     = 0.1
    args.minimum_variational_weight = 0
    args.show_progress             = False
    args.save_step                 = 10000   # effectively never
    args.save_best                 = False
    args.save_last                 = False
    args.minority_fraction         = None
    args.imbalance_ratio           = None
    args.resume                    = False

    if args.model == 'bert':
        args.max_grad_norm  = 1.0
        args.adam_epsilon   = 1e-8
        args.warmup_steps   = 0

    return args


# ─── Log directory naming ─────────────────────────────────────────────────────

def _pct_tag(fraction: float) -> str:
    return str(int(round(float(fraction) * 100)))


def _make_log_dir(args) -> str:
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M')
    if args.ot_cache:
        ot_tag = f"f{args.f_power}_cap{args.clip_cap}_reg{args.reg_scale}"
    else:
        ot_tag = 'erm'

    # Noise bucket
    if args.noise_type == 'symmetric':
        noise_bucket = f"Noise_Symmetric_{_pct_tag(args.noise_fraction)}"
    elif args.noise_type == 'subgroup':
        gff = {int(g): float(f)
               for token in args.group_flip_fractions.split(',')
               for g, f in [token.strip().split(':')]}
        gids = '-'.join(str(g) for g in sorted(gff))
        fracs = set(gff.values())
        if len(fracs) == 1:
            noise_bucket = f"Noise_SubG{gids}_{_pct_tag(fracs.pop())}"
        else:
            parts = '_'.join(f"g{g}f{_pct_tag(f)}" for g, f in sorted(gff.items()))
            noise_bucket = f"Noise_SubG_{parts}"
    else:
        noise_bucket = 'Clean'

    run_name = (
        f"POTER_{args.dataset}_{ot_tag}"
        f"_wd{args.weight_decay}_lr{args.lr}"
        f"_ep{args.n_epochs}_nseed{args.noise_seed}_seed{args.seed}_{ts}"
    )
    return os.path.join(args.log_dir, args.dataset, noise_bucket, run_name)


# ─── CivilComments 16-group evaluation ───────────────────────────────────────

def _load_civilcomments_split_dfs(root_dir: str):
    import pandas as pd
    csv_path = os.path.join(root_dir, 'data', 'all_data_with_identities.csv')
    meta = pd.read_csv(csv_path)
    val_df  = meta[meta['split'] == 'val' ].reset_index(drop=True)
    test_df = meta[meta['split'] == 'test'].reset_index(drop=True)
    return val_df, test_df


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # GPU
    torch.cuda.set_device(args.gpu)

    # Logging
    log_dir = _make_log_dir(args)
    os.makedirs(log_dir, exist_ok=True)
    args.log_dir = log_dir

    logger = Logger(os.path.join(log_dir, 'log.txt'), mode='w')
    log_args(args, logger)

    # Seed
    set_seed(args.seed, strict=(args.model == 'bert'), warn_only=False)

    # ── Dataset ───────────────────────────────────────────────────────────────
    train_data, val_data, test_data = prepare_data(args, train=True)

    # ── Label noise ───────────────────────────────────────────────────────────
    if args.noise_type is not None:
        from noise import apply_label_noise
        gff = None
        if args.group_flip_fractions:
            gff = {int(g): float(f)
                   for token in args.group_flip_fractions.split(',')
                   for g, f in [token.strip().split(':')]}
        train_data, flip_info = apply_label_noise(
            train_data,
            noise_type=args.noise_type,
            noise_fraction=args.noise_fraction,
            group_flip_fractions=gff,
            noise_seed=args.noise_seed,
        )
        logger.write(
            f'[Noise] type={args.noise_type}  '
            f'flipped={flip_info["n_flipped"]}/{flip_info["n_total"]}  '
            f'({flip_info["flip_fraction_applied"]:.3f})\n'
        )
        if flip_info['group_flip_applied']:
            for gid, info in flip_info['group_flip_applied'].items():
                logger.write(
                    f'[Noise]   group {gid}: '
                    f'{info["n_flipped"]}/{info["n_group"]} '
                    f'({info["flip_fraction_applied"]:.3f})\n'
                )

    # ── POTER reweighting ─────────────────────────────────────────────────────
    sample_weights = None
    if args.ot_cache is not None:
        logger.write(f'\n[POTER] Loading feature cache: {args.ot_cache}\n')
        try:
            cache = torch.load(args.ot_cache, map_location='cpu', weights_only=False)
        except TypeError:
            cache = torch.load(args.ot_cache, map_location='cpu')

        X_train = np.asarray(cache['X_train'])
        y_train = np.asarray(cache['y_train'])
        X_val   = np.asarray(cache['X_val'])
        y_val   = np.asarray(cache['y_val'])
        # When noise was applied, use noisy labels for OT so that the
        # transport plan reflects the corrupted label distribution.
        if args.noise_type is not None:
            y_train = np.asarray(train_data._y_array)
        logger.write(f'[POTER] Train features: {X_train.shape}, Val features: {X_val.shape}\n')

        logger.write(f'[POTER] Computing OT potentials (reg_scale={args.reg_scale})...\n')
        f_star = compute_ot_potentials(
            X_train, y_train, X_val, y_val,
            reg_scale=args.reg_scale,
            device=f'cuda:{args.gpu}',
        )

        logger.write(
            f'[POTER] Computing sample weights '
            f'(f_power={args.f_power}, clip_cap={args.clip_cap})...\n'
        )
        w_np = compute_sample_weights(
            f_star, y_train,
            f_power=args.f_power,
            clip_cap=args.clip_cap,
        )
        sample_weights = torch.from_numpy(w_np).float()
        logger.write(
            f'[POTER] Weight stats — mean: {sample_weights.mean():.3f}, '
            f'max: {sample_weights.max():.3f}, min: {sample_weights.min():.3f}\n\n'
        )
        del cache
        gc.collect()

    # ── Data loaders ──────────────────────────────────────────────────────────
    loader_kwargs = {'batch_size': args.batch_size, 'num_workers': 4,
                     'pin_memory': torch.cuda.is_available()}
    train_loader = train_data.get_loader(
        train=True, reweight_groups=False,
        sample_weights=sample_weights, **loader_kwargs
    )
    val_loader  = val_data.get_loader(train=False, reweight_groups=None, **loader_kwargs)
    test_loader = test_data.get_loader(train=False, reweight_groups=None, **loader_kwargs)

    data = {
        'train_loader'    : train_loader,
        'val_loader'      : val_loader,
        'test_loader'     : test_loader,
        'train_data'      : train_data,
        'val_data'        : val_data,
        'test_data'       : test_data,
        'sample_importance': None,   # weights go to sampler, not loss
    }
    log_data(data, logger)

    # ── Model ─────────────────────────────────────────────────────────────────
    n_classes = train_data.n_classes
    if args.model == 'bert':
        from pytorch_transformers import BertConfig, BertForSequenceClassification
        config = BertConfig.from_pretrained('bert-base-uncased', num_labels=n_classes)
        model = BertForSequenceClassification.from_pretrained(
            'bert-base-uncased', config=config
        )
    elif args.model == 'resnet50':
        model = torchvision.models.resnet50(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, n_classes)
    else:
        raise ValueError(f'Unsupported model: {args.model}')

    # ── 16-group loggers (CivilComments only) ─────────────────────────────────
    val_16group_csv = test_16group_csv = None
    val_split_df    = test_split_df    = None
    if args.dataset == 'CivilComments':
        val_split_df, test_split_df = _load_civilcomments_split_dfs(args.root_dir)
        val_16group_csv  = os.path.join(log_dir, 'val_16group.csv')
        test_16group_csv = os.path.join(log_dir, 'test_16group.csv')

    # ── Train ─────────────────────────────────────────────────────────────────
    criterion       = nn.CrossEntropyLoss(reduction='none')
    train_csv_logger = CSVBatchLogger(os.path.join(log_dir, 'train.csv'), train_data.n_groups)
    val_csv_logger   = CSVBatchLogger(os.path.join(log_dir, 'val.csv'),   train_data.n_groups)
    test_csv_logger  = CSVBatchLogger(os.path.join(log_dir, 'test.csv'),  train_data.n_groups)

    train(
        model, criterion, data, logger,
        train_csv_logger, val_csv_logger, test_csv_logger,
        args, epoch_offset=0,
        val_16group_csv=val_16group_csv,
        test_16group_csv=test_16group_csv,
        val_split_df=val_split_df,
        test_split_df=test_split_df,
    )

    train_csv_logger.close()
    val_csv_logger.close()
    test_csv_logger.close()
    logger.write('\nTraining complete.\n')


if __name__ == '__main__':
    main()
