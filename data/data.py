import os
import torch
import numpy as np
from torch.utils.data import Subset

from data.confounder_utils import prepare_confounder_data

dataset_attributes = {
    'CelebA'       : {'root_dir': 'celebA'},
    'CUB'          : {'root_dir': 'cub'},
    'CMNIST'       : {'root_dir': 'CMNIST'},
    'CivilComments': {'root_dir': 'civilcomments'},
}

shift_types = ['confounder', 'label_shift_step']


def prepare_data(args, train, return_full_dataset=False):
    if args.root_dir is None:
        data_dir = getattr(args, 'data_dir', '.')
        args.root_dir = os.path.join(data_dir, dataset_attributes[args.dataset]['root_dir'])

    if args.shift_type == 'confounder':
        return prepare_confounder_data(args, train, return_full_dataset)
    else:
        raise ValueError(f"Unsupported shift_type: {args.shift_type}")


def log_data(data, logger):
    logger.write('Training Data...\n')
    for g in range(data['train_data'].n_groups):
        logger.write(f'    {data["train_data"].group_str(g)}: n = {data["train_data"].group_counts()[g]:.0f}\n')
    logger.write('Validation Data...\n')
    for g in range(data['val_data'].n_groups):
        logger.write(f'    {data["val_data"].group_str(g)}: n = {data["val_data"].group_counts()[g]:.0f}\n')
    if data.get('test_data') is not None:
        logger.write('Test Data...\n')
        for g in range(data['test_data'].n_groups):
            logger.write(f'    {data["test_data"].group_str(g)}: n = {data["test_data"].group_counts()[g]:.0f}\n')
