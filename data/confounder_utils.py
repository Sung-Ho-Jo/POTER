from data.celebA_dataset import CelebADataset
from data.cub_dataset import CUBDataset
from data.dro_dataset import DRODataset
from data.civilcomments_dataset import CivilCommentsDataset
from data.cmnist_dataset import CMNISTDataset

confounder_settings = {
    'CelebA'       : {'constructor': CelebADataset},
    'CUB'          : {'constructor': CUBDataset},
    'CivilComments': {'constructor': CivilCommentsDataset},
    'CMNIST'       : {'constructor': CMNISTDataset},
}

########################
### DATA PREPARATION ###
########################
def prepare_confounder_data(args, train, return_full_dataset=False):
    constructor = confounder_settings[args.dataset]['constructor']
    extra_kwargs = {}
    if args.dataset in {'CUB', 'CelebA'}:
        extra_kwargs['augment_photometric'] = getattr(args, 'augment_photometric', False)
        extra_kwargs['augment_rotate'] = getattr(args, 'augment_rotate', False)

    full_dataset = constructor(
        root_dir=args.root_dir,
        target_name=args.target_name,
        confounder_names=args.confounder_names,
        model_type=args.model,
        augment_data=args.augment_data,
        **extra_kwargs)
    if return_full_dataset:
        return DRODataset(
            full_dataset,
            process_item_fn=None,
            n_groups=full_dataset.n_groups,
            n_classes=full_dataset.n_classes,
            group_str_fn=full_dataset.group_str)
    if train:
        splits = ['train', 'val', 'test']
    else:
        splits = ['test']
    subsets = full_dataset.get_splits(splits, train_frac=args.fraction)
    dro_subsets = [DRODataset(subsets[split], process_item_fn=None, n_groups=full_dataset.n_groups,
                              n_classes=full_dataset.n_classes, group_str_fn=full_dataset.group_str) \
                   for split in splits]
    return dro_subsets
