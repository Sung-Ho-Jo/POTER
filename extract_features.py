#!/usr/bin/env python
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("PYTHONHASHSEED", "1")
"""
POTER – Feature Extraction
===========================
Extracts and saves a feature cache (.pt file) for use by run_expt.py.

The cache contains {X_train, y_train, X_ref, y_ref} where X_ref is a
sampled subset of the validation split used as the reference distribution
for OT reweighting.  The reference set is constructed via --ref_per_group,
which specifies how many samples to draw from each group (0 = exclude).

Usage
-----
python extract_features.py \\
    --dataset CUB --data_dir /path/to/data \\
    --ref_per_group 0:0,1:466,2:133,3:0 \\
    --output features/cub_features.pt --gpu 0

python extract_features.py \\
    --dataset CelebA --data_dir /path/to/data \\
    --ref_per_group 0:0,1:1387,2:0,3:182 \\
    --output features/celeba_features.pt --gpu 0

python extract_features.py \\
    --dataset CMNIST --data_dir /path/to/data \\
    --ref_per_group 0:2591,1:0,2:0,3:2431 \\
    --output features/cmnist_features.pt --gpu 0

python extract_features.py \\
    --dataset CivilComments --data_dir /path/to/data \\
    --ref_per_group 0:0,1:5000,2:2111,3:0 \\
    --output features/civilcomments_features.pt --gpu 0

Default --ref_per_group per dataset (used when the flag is omitted):
    CUB           : 0:0,   1:466, 2:133, 3:0
    CelebA        : 0:0,   1:1387,2:0,   3:182
    CMNIST        : 0:2591,1:0,   2:0,   3:2431
    CivilComments : 0:0,   1:5000,2:2111,3:0
"""

import os
import argparse

import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm


_DATASET_CFG = {
    'CUB': {
        'target_name'         : 'waterbird_complete95',
        'confounder_names'    : ['forest2water2'],
        'model'               : 'resnet50',
        'data_subdir'         : 'cub',
        'default_ref_per_group': {0: 0, 1: 466, 2: 133, 3: 0},
    },
    'CelebA': {
        'target_name'         : 'Blond_Hair',
        'confounder_names'    : ['Male'],
        'model'               : 'resnet50',
        'data_subdir'         : 'celebA',
        'default_ref_per_group': {0: 8535, 1: 0, 2: 0, 3: 182},
    },
    'CMNIST': {
        'target_name'         : '0-4',
        'confounder_names'    : ['isred'],
        'model'               : 'resnet50',
        'data_subdir'         : 'CMNIST',
        'default_ref_per_group': {0: 2591, 1: 0, 2: 0, 3: 2431},
    },
    'CivilComments': {
        'target_name'         : 'toxicity',
        'confounder_names'    : ['identity_any'],
        'model'               : 'bert',
        'data_subdir'         : 'civilcomments',
        'default_ref_per_group': {0: 0, 1: 5000, 2: 2111, 3: 0},
    },
}


# ─── Argument parsing ─────────────────────────────────────────────────────────

def _parse_ref_per_group(s: str) -> dict:
    """Parse '0:0,1:466,2:133,3:0' into {0: 0, 1: 466, 2: 133, 3: 0}."""
    result = {}
    for token in s.split(','):
        g, n = token.strip().split(':')
        result[int(g)] = int(n)
    return result


def parse_args():
    parser = argparse.ArgumentParser(
        description='POTER feature extraction',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--dataset',  required=True, choices=list(_DATASET_CFG.keys()))
    parser.add_argument('--data_dir', required=True,
                        help='Root directory containing dataset subdirectories.')
    parser.add_argument('--output',   required=True,
                        help='Output .pt file path.')
    parser.add_argument('--ref_per_group', default=None,
                        help='Number of reference samples per group. '
                             'Format: g0:n0,g1:n1,... (0 = exclude group). '
                             'Default: dataset-specific setting (see module docstring).')
    parser.add_argument('--ref_seed', type=int, default=0,
                        help='Random seed for reference set sampling.')
    parser.add_argument('--batch_size',  type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=0,
                        help='DataLoader workers. 0 = main process only (required for '
                             'bit-for-bit reproducibility).')
    parser.add_argument('--gpu', type=int, default=0)
    return parser.parse_args()


# ─── Reference set construction ───────────────────────────────────────────────

def make_reference_subset(val_data, ref_per_group: dict, seed: int = 0):
    """Sample a fixed number of examples per group from the validation split.

    Args:
        val_data      : DRODataset for the validation split
        ref_per_group : {group_id: n_samples}; groups with n=0 are excluded
        seed          : random seed for reproducibility

    Returns:
        ref_subset    : Subset of val_data
        ref_group_array: numpy array of group labels for the subset
    """
    np.random.seed(seed)
    g_val = val_data._group_array.numpy()
    selected = []

    for gid, n in ref_per_group.items():
        idxs = np.where(g_val == gid)[0]
        if len(idxs) < n:
            raise ValueError(
                f'Group {gid} has only {len(idxs)} val samples, '
                f'but ref_per_group requests {n}.'
            )
        # Always call np.random.choice (even for n=0) to consume the same
        # random state, ensuring identical sample selection when re-extracting
        # features from scratch.
        chosen = np.random.choice(idxs, n, replace=False)
        selected.extend(chosen.tolist())

    ref_subset = Subset(val_data, selected)
    ref_group_array = g_val[selected]
    return ref_subset, ref_group_array


# ─── Feature extractors ───────────────────────────────────────────────────────

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD  = (0.229, 0.224, 0.225)


class _ResNet50FeatureExtractor(nn.Module):
    """ImageNet pretrained ResNet50 backbone (avgpool output, 2048-dim).

    Accepts images that are already ImageNet-normalised (as produced by the
    dataset transforms) and optionally resizes them to 224×224 via bicubic
    interpolation before the forward pass.  The resize step is a no-op when
    the spatial size is already 224×224 (e.g. Waterbirds, CelebA).
    """
    _TARGET = (224, 224)

    def __init__(self):
        super().__init__()
        try:
            weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V1
            base = torchvision.models.resnet50(weights=weights)
        except AttributeError:
            base = torchvision.models.resnet50(pretrained=True)
        self.backbone = nn.Sequential(*list(base.children())[:-1])

    def forward(self, x):
        import torch.nn.functional as F
        mean = x.new_tensor(_IMAGENET_MEAN).view(1, 3, 1, 1)
        std  = x.new_tensor(_IMAGENET_STD).view(1, 3, 1, 1)
        x = x * std + mean                                      # de-normalise
        if tuple(x.shape[-2:]) != self._TARGET:
            x = F.interpolate(x, size=self._TARGET,
                              mode='bicubic', align_corners=False)  # resize
        x = (x - mean) / std                                    # re-normalise
        feats = self.backbone(x)
        return feats.flatten(1)


def build_image_feature_extractor():
    return _ResNet50FeatureExtractor()


class BertFeatureExtractor(nn.Module):
    """Pretrained BERT-base-uncased [CLS] pooled representation (768-dim)."""
    def __init__(self):
        super().__init__()
        try:
            from transformers import BertModel
        except ImportError:
            from pytorch_transformers import BertModel
        self.bert = BertModel.from_pretrained('bert-base-uncased')

    def forward(self, x):
        output = self.bert(
            input_ids=x[:, :, 0],
            attention_mask=x[:, :, 1],
            token_type_ids=x[:, :, 2],
        )
        if hasattr(output, 'pooler_output'):
            return output.pooler_output  # transformers >= 4.x
        return output[1]                # pytorch_transformers


# ─── Dataset loading ──────────────────────────────────────────────────────────

def _make_args_namespace(dataset, data_dir):
    import types
    cfg = _DATASET_CFG[dataset]
    return types.SimpleNamespace(
        dataset=dataset,
        data_dir=data_dir,
        root_dir=os.path.join(data_dir, cfg['data_subdir']),
        target_name=cfg['target_name'],
        confounder_names=cfg['confounder_names'],
        model=cfg['model'],
        shift_type='confounder',
        fraction=1.0,
        reweight_groups=False,
        augment_data=False,
        augment_photometric=False,
        augment_rotate=False,
        val_fraction=0.1,
        minority_fraction=None,
        imbalance_ratio=None,
    )


# ─── Extraction ───────────────────────────────────────────────────────────────

@torch.no_grad()
def extract(model, loader, device):
    model.eval()
    model.to(device)
    features, labels = [], []
    for batch in tqdm(loader, desc='Extracting'):
        x, y = batch[0].to(device), batch[1]
        feats = model(x)
        features.append(feats.float().cpu())
        labels.append(y)
    return torch.cat(features).numpy(), torch.cat(labels).numpy()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)
    print(f'Device: {device}')

    cfg = _DATASET_CFG[args.dataset]
    is_bert = (cfg['model'] == 'bert')

    # Reference set configuration
    if args.ref_per_group is not None:
        ref_per_group = _parse_ref_per_group(args.ref_per_group)
    else:
        ref_per_group = cfg['default_ref_per_group']
    print(f'Reference set composition: {ref_per_group}')

    # Load dataset
    from data.data import prepare_data
    ns = _make_args_namespace(args.dataset, args.data_dir)
    train_data, val_data, _ = prepare_data(ns, train=True)

    loader_kw = dict(batch_size=args.batch_size, num_workers=args.num_workers,
                     pin_memory=True, shuffle=False)
    train_loader = DataLoader(train_data, **loader_kw)

    # Build reference subset
    ref_subset, ref_group_array = make_reference_subset(
        val_data, ref_per_group, seed=args.ref_seed
    )
    ref_loader = DataLoader(ref_subset, **loader_kw)
    total_ref = sum(n for n in ref_per_group.values())
    print(f'Reference set: {total_ref} samples  |  per group: {ref_per_group}')

    # Feature extractor
    model = BertFeatureExtractor() if is_bert else build_image_feature_extractor()

    print('Extracting train features...')
    X_train, y_train = extract(model, train_loader, device)

    print('Extracting reference features...')
    X_ref, y_ref = extract(model, ref_loader, device)

    print(f'X_train: {X_train.shape},  X_ref: {X_ref.shape}')

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    torch.save({
        'X_train'      : X_train,
        'y_train'      : y_train,
        'X_val'        : X_ref,      # key kept as X_val for run_expt.py compatibility
        'y_val'        : y_ref,
        'dataset'      : args.dataset,
        'ref_per_group': ref_per_group,
    }, args.output)
    print(f'Saved feature cache to: {args.output}')


if __name__ == '__main__':
    main()
