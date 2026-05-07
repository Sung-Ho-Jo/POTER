"""
POTER – Label Noise Injection
==============================
Applies label flip noise to a training DRODataset.

Two noise types
---------------
Symmetric noise  : flip labels for a random fraction of samples,
                   applied class-conditionally (classwise=True, default)
                   so each class loses the same fraction of correct labels.

Subgroup noise   : flip labels only within specified subgroups,
                   with per-group fractions (group_flip_fractions dict).

Usage in run_expt.py
--------------------
    noisy_train_data, flip_info = apply_label_noise(
        train_data,
        noise_type='symmetric',
        noise_fraction=0.2,
        noise_seed=0,
    )

    noisy_train_data, flip_info = apply_label_noise(
        train_data,
        noise_type='subgroup',
        group_flip_fractions={1: 0.3, 3: 0.6},
        noise_seed=0,
    )
"""

import copy
import numpy as np
import torch
from torch.utils.data import Subset


def apply_label_noise(
    train_data,
    noise_type: str = 'symmetric',
    noise_fraction: float = 0.0,
    group_flip_fractions: dict = None,
    noise_seed: int = 0,
):
    """Apply label flip noise to a training DRODataset.

    Args:
        train_data           : DRODataset for the training split
        noise_type           : 'symmetric' or 'subgroup'
            'symmetric'  — flip `noise_fraction` of each class independently
            'subgroup'   — flip per-group fractions given by `group_flip_fractions`
        noise_fraction       : flip fraction for symmetric noise (ignored for subgroup)
        group_flip_fractions : {group_id: fraction} for subgroup noise
        noise_seed           : random seed

    Returns:
        noisy_train_data : deep-copied DRODataset with flipped labels
        flip_info        : dict with flip statistics
    """
    if noise_type == 'symmetric':
        return _apply_classwise(train_data, noise_fraction, noise_seed)
    elif noise_type == 'subgroup':
        if not group_flip_fractions:
            raise ValueError("group_flip_fractions must be provided for noise_type='subgroup'.")
        return _apply_group_dict(train_data, group_flip_fractions, noise_seed)
    else:
        raise ValueError(f"noise_type must be 'symmetric' or 'subgroup', got '{noise_type}'.")


# ─── Internal helpers ─────────────────────────────────────────────────────────

def _flip_dataset(train_data, flipped_local, subset_indices, base_ds, n_total,
                  classwise, selection_mode, per_group_applied, seed):
    """Shared mutation logic after candidate indices are chosen."""
    noisy_train = copy.deepcopy(train_data)
    ds = noisy_train.dataset
    base_ds_copy = ds.dataset if isinstance(ds, Subset) else ds

    n_flip = len(flipped_local)
    flipped_base = subset_indices[flipped_local]

    y_arr = np.asarray(base_ds_copy.y_array)
    original_labels = y_arr[flipped_base].copy()

    if n_flip > 0:
        unique_cls = np.unique(y_arr)
        if set(unique_cls.tolist()) == {0, 1}:
            y_arr[flipped_base] = 1 - y_arr[flipped_base]
        else:
            rng = np.random.RandomState(seed)
            for idx in flipped_base:
                old = y_arr[idx]
                y_arr[idx] = rng.choice(unique_cls[unique_cls != old])
        base_ds_copy.y_array = y_arr

    flipped_labels = y_arr[flipped_base].copy()

    # Keep group_array consistent with updated labels.
    groups_per_class = int(base_ds_copy.n_groups // base_ds_copy.n_classes)
    conf = np.asarray(base_ds_copy.confounder_array)
    base_ds_copy.group_array = (
        np.asarray(base_ds_copy.y_array) * groups_per_class + conf
    ).astype("int")

    new_y = np.asarray(base_ds_copy.y_array)[subset_indices]
    new_g = np.asarray(base_ds_copy.group_array)[subset_indices]
    noisy_train._y_array     = torch.as_tensor(new_y, dtype=torch.long)
    noisy_train._group_array = torch.as_tensor(new_g, dtype=torch.long)
    noisy_train._group_counts = (
        (torch.arange(noisy_train.n_groups).unsqueeze(1) == noisy_train._group_array)
        .sum(1).float()
    )
    noisy_train._y_counts = (
        (torch.arange(noisy_train.n_classes).unsqueeze(1) == noisy_train._y_array)
        .sum(1).float()
    )

    flip_mask = np.zeros(n_total, dtype=bool)
    flip_mask[flipped_local] = True
    noisy_train._flip_mask = torch.as_tensor(flip_mask, dtype=torch.bool)

    flip_info = {
        "n_total"              : int(n_total),
        "n_flipped"            : int(n_flip),
        "flip_fraction_applied": float(n_flip / max(n_total, 1)),
        "classwise"            : bool(classwise),
        "selection_mode"       : selection_mode,
        "group_flip_applied"   : per_group_applied,
        "seed"                 : int(seed),
        "original_labels"      : original_labels,
        "flipped_labels"       : flipped_labels,
        "flip_mask"            : flip_mask,
    }
    return noisy_train, flip_info


def _get_subset_info(train_data):
    ds = train_data.dataset
    if isinstance(ds, Subset):
        base_ds = ds.dataset
        subset_indices = np.asarray(ds.indices, dtype=np.int64)
    else:
        base_ds = ds
        subset_indices = np.arange(len(ds), dtype=np.int64)
    return base_ds, subset_indices


def _apply_classwise(train_data, flip_fraction, seed):
    base_ds, subset_indices = _get_subset_info(train_data)
    n_total = len(subset_indices)
    rng = np.random.RandomState(seed)
    y_subset = np.asarray(base_ds.y_array)[subset_indices]

    parts = []
    for cls in np.unique(y_subset):
        cls_local = np.where(y_subset == cls)[0]
        n_flip = int(np.round(len(cls_local) * float(flip_fraction)))
        if n_flip > 0:
            parts.append(rng.choice(cls_local, size=n_flip, replace=False))

    flipped_local = np.sort(np.concatenate(parts).astype(np.int64)) if parts else np.array([], dtype=np.int64)
    return _flip_dataset(train_data, flipped_local, subset_indices, base_ds,
                         n_total, classwise=True, selection_mode='classwise',
                         per_group_applied=None, seed=seed)


def _apply_group_dict(train_data, group_flip_fractions, seed):
    base_ds, subset_indices = _get_subset_info(train_data)
    n_total = len(subset_indices)
    rng = np.random.RandomState(seed)
    g_subset = np.asarray(base_ds.group_array)[subset_indices]

    parts = []
    per_group_applied = {}
    for gid_raw, frac_raw in group_flip_fractions.items():
        gid, frac = int(gid_raw), float(frac_raw)
        grp_local = np.where(g_subset == gid)[0]
        n_grp = len(grp_local)
        n_flip = int(np.round(n_grp * frac))
        if n_flip > 0:
            parts.append(rng.choice(grp_local, size=n_flip, replace=False))
        per_group_applied[gid] = {
            "n_group" : int(n_grp),
            "n_flipped": int(n_flip),
            "flip_fraction_applied": float(n_flip / max(n_grp, 1)),
        }

    flipped_local = np.sort(np.concatenate(parts).astype(np.int64)) if parts else np.array([], dtype=np.int64)
    return _flip_dataset(train_data, flipped_local, subset_indices, base_ds,
                         n_total, classwise=False, selection_mode='group_dict',
                         per_group_applied=per_group_applied, seed=seed)
