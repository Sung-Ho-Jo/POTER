import os
import torch
import pandas as pd
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from models import model_attributes
from torch.utils.data import Dataset, Subset
from data.confounder_dataset import ConfounderDataset

class CUBDataset(ConfounderDataset):
    """
    CUB dataset (already cropped and centered).
    Note: metadata_df is one-indexed.
    """

    def __init__(self, root_dir,
                 target_name, confounder_names,
                 augment_data=False,
                 augment_photometric=False,
                 augment_rotate=False,
                 model_type=None):
        self.root_dir = root_dir
        self.target_name = target_name
        self.confounder_names = confounder_names
        self.model_type = model_type
        self.augment_data = augment_data
        self.augment_photometric = augment_photometric
        self.augment_rotate = augment_rotate

        self.data_dir = os.path.join(
            self.root_dir,
            'data',
            '_'.join([self.target_name] + self.confounder_names))

        if not os.path.exists(self.data_dir):
            raise ValueError(
                f'{self.data_dir} does not exist yet. Please generate the dataset first.')

        # Read in metadata
        self.metadata_df = pd.read_csv(
            os.path.join(self.data_dir, 'metadata.csv'))

        # Get the y values
        self.y_array = self.metadata_df['y'].values
        self.n_classes = 2

        # We only support one confounder for CUB for now
        self.confounder_array = self.metadata_df['place'].values
        self.n_confounders = 1
        # Map to groups
        self.n_groups = pow(2, 2)
        self.group_array = (self.y_array*(self.n_groups/2) + self.confounder_array).astype('int')

        # Extract filenames and splits
        self.filename_array = self.metadata_df['img_filename'].values
        self.split_array = self.metadata_df['split'].values
        self.split_dict = {
            'train': 0,
            'val': 1,
            'test': 2
        }

        # Set transform
        if model_attributes[self.model_type]['feature_type']=='precomputed':
            self.features_mat = torch.from_numpy(np.load(
                os.path.join(root_dir, 'features', model_attributes[self.model_type]['feature_filename']))).float()
            self.train_transform = None
            self.eval_transform = None
        else:
            self.features_mat = None
            self.train_transform = get_transform_cub(
                self.model_type,
                train=True,
                augment_data=augment_data,
                augment_photometric=augment_photometric,
                augment_rotate=augment_rotate)
            self.eval_transform = get_transform_cub(
                self.model_type,
                train=False,
                augment_data=augment_data,
                augment_photometric=augment_photometric,
                augment_rotate=augment_rotate)


def get_transform_cub(
    model_type,
    train,
    augment_data,
    augment_photometric=False,
    augment_rotate=False,
):
    scale = 256.0/224.0
    target_resolution = model_attributes[model_type]['target_resolution']
    assert target_resolution is not None

    if (not train) or (not augment_data):
        # Resizes the image to a slightly larger square then crops the center.
        transform = transforms.Compose([
            transforms.Resize((int(target_resolution[0]*scale), int(target_resolution[1]*scale))),
            transforms.CenterCrop(target_resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        # v2 reference (disabled):
        # from torchvision.transforms import v2
        # transform = transforms.Compose([
        #     v2.RandomResizedCrop(size=target_resolution),
        #     v2.RandomPhotometricDistort(p=0.5),
        #     v2.RandomRotation((-15, 15)),
        #     v2.RandomHorizontalFlip(p=0.5),
        #     v2.ToImage(),
        #     v2.ToDtype(torch.float32, scale=True),
        #     v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        # ])
        aug_transforms = [
            transforms.RandomResizedCrop(
                target_resolution,
                scale=(0.7, 1.0),
                ratio=(1.0, 1.3333333333333333),
                interpolation=2),
        ]
        if augment_photometric:
            aug_transforms.append(
                transforms.RandomApply(
                    [transforms.ColorJitter(
                        brightness=0.125,
                        contrast=0.5,
                        saturation=0.5,
                        hue=0.05
                    )],
                    p=0.5
                )
            )
        if augment_rotate:
            aug_transforms.append(transforms.RandomRotation(degrees=(-15, 15)))
        aug_transforms += [
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]
        transform = transforms.Compose(aug_transforms)
    return transform
