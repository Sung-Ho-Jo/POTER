import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, Subset
from tqdm import tqdm
from data.confounder_dataset import ConfounderDataset

# Identity columns used to derive identity_any
IDENTITY_COLUMNS = [
    'male', 'female', 'LGBTQ', 'christian', 'muslim',
    'other_religions', 'black', 'white',
]


class CivilCommentsDataset(ConfounderDataset):
    """
    CivilComments dataset (Borkan et al. 2019).

    Target (y):
        toxicity  — 1 if toxic (≥0.5), 0 if civil
    Confounder (d):
        identity_any — 1 if any minority-identity column ≥ 0.5, else 0

    Groups (g = y*2 + d):
        0: civil,  no identity
        1: civil,  has identity
        2: toxic,  no identity
        3: toxic,  has identity

    Expected data layout:
        <root_dir>/data/all_data_with_identities.csv   (WILDS format)
      OR
        <root_dir>/data/metadata.csv                   (pre-processed)

    WILDS CSV columns needed: comment_text, toxicity, split
      (+ at least one of the IDENTITY_COLUMNS, or a pre-computed identity_any column)

    Pre-processed CSV columns needed: text (or comment_text), toxicity,
      identity_any, split.

    BERT tokenizations are cached to:
        <root_dir>/data/cached_bert-base-uncased_128_civilcomments.pt
    """

    def __init__(self, root_dir, target_name, confounder_names,
                 augment_data=False, model_type=None):
        self.root_dir = root_dir
        self.target_name = target_name
        self.confounder_names = confounder_names
        self.model_type = model_type
        self.augment_data = augment_data

        assert target_name == 'toxicity', (
            f"CivilComments target_name must be 'toxicity', got '{target_name}'"
        )
        assert len(confounder_names) == 1 and confounder_names[0] == 'identity_any', (
            f"CivilComments confounder_names must be ['identity_any'], got {confounder_names}"
        )
        assert model_type == 'bert', (
            f"CivilComments requires model_type='bert', got '{model_type}'"
        )
        assert not augment_data, "Data augmentation is not supported for CivilComments."

        self.data_dir = os.path.join(root_dir, 'data')
        if not os.path.exists(self.data_dir):
            raise ValueError(
                f'{self.data_dir} does not exist. '
                'Please prepare the CivilComments dataset first.\n'
                'Download from https://wilds.stanford.edu/datasets/ '
                'and place the CSV under <root_dir>/data/.'
            )

        # ── Load metadata ────────────────────────────────────────────────────
        self.metadata_df = self._load_metadata()

        # ── Labels ──────────────────────────────────────────────────────────
        self.y_array = (self.metadata_df['toxicity'].values >= 0.5).astype(int)
        self.n_classes = 2

        # ── Confounder ──────────────────────────────────────────────────────
        self.confounder_array = self.metadata_df['identity_any'].values.astype(int)
        self.n_confounders = 1

        # ── Groups ──────────────────────────────────────────────────────────
        self.n_groups = 4
        self.group_array = (self.y_array * 2 + self.confounder_array).astype(int)

        # ── Splits ──────────────────────────────────────────────────────────
        self.split_dict = {'train': 0, 'val': 1, 'test': 2}
        split_col = self.metadata_df['split'].values
        if split_col.dtype == object or isinstance(split_col[0], str):
            split_map = {'train': 0, 'val': 1, 'test': 2}
            split_col = np.array([split_map[s] for s in split_col])
        self.split_array = split_col.astype(int)

        # ── BERT tokenization ────────────────────────────────────────────────
        self.x_array = self._load_or_tokenize(max_length=220)

    # ── Private helpers ──────────────────────────────────────────────────────

    def _load_metadata(self):
        """Load CSV and ensure identity_any column exists."""
        candidates = [
            os.path.join(self.data_dir, 'all_data_with_identities.csv'),
            os.path.join(self.data_dir, 'metadata.csv'),
        ]
        path = next((p for p in candidates if os.path.exists(p)), None)
        if path is None:
            raise ValueError(
                f'No metadata CSV found in {self.data_dir}. '
                'Expected "all_data_with_identities.csv" or "metadata.csv".'
            )

        df = pd.read_csv(path)

        # Normalise text column name
        if 'comment_text' in df.columns and 'text' not in df.columns:
            df = df.rename(columns={'comment_text': 'text'})

        # Derive identity_any from the 8 individual identity columns (binarized at 0.5)
        # This matches JTT/LISA/DFR convention: identity=1 if any column >= 0.5
        present_identity_cols = [c for c in IDENTITY_COLUMNS if c in df.columns]
        if not present_identity_cols:
            raise ValueError(
                'Could not find any of the individual identity '
                f'columns {IDENTITY_COLUMNS} in the CSV.'
            )
        df['identity_any'] = (
            df[present_identity_cols].fillna(0).ge(0.5).any(axis=1)
        ).astype(int)

        required = {'text', 'toxicity', 'identity_any', 'split'}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f'Metadata CSV is missing columns: {missing}')

        return df

    def _load_or_tokenize(self, max_length=128):
        """Return stacked BERT tokens (N, seq_len, 3) = [input_ids, mask, seg]."""
        cache_path = os.path.join(
            self.data_dir,
            f'cached_bert-base-uncased_{max_length}_civilcomments.pt'
        )

        if os.path.exists(cache_path):
            print(f'Loading cached BERT tokens from {cache_path}')
            cached = torch.load(cache_path)
            input_ids = cached['input_ids']
            input_masks = cached['input_masks']
            segment_ids = cached['segment_ids']
        else:
            print('Tokenizing CivilComments with bert-base-uncased (one-time)…')
            try:
                from transformers import BertTokenizer
            except ImportError:
                from pytorch_transformers import BertTokenizer

            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            texts = self.metadata_df['text'].tolist()

            all_ids, all_masks, all_segs = [], [], []
            for text in tqdm(texts, desc='Tokenising'):
                enc = tokenizer.encode_plus(
                    str(text),
                    add_special_tokens=True,
                    max_length=max_length,
                    padding='max_length',
                    truncation=True,
                    return_attention_mask=True,
                )
                all_ids.append(enc['input_ids'])
                all_masks.append(enc['attention_mask'])
                all_segs.append([0] * max_length)  # single-sentence: all zeros

            input_ids = torch.tensor(all_ids, dtype=torch.long)
            input_masks = torch.tensor(all_masks, dtype=torch.long)
            segment_ids = torch.tensor(all_segs, dtype=torch.long)

            torch.save(
                {'input_ids': input_ids, 'input_masks': input_masks, 'segment_ids': segment_ids},
                cache_path,
            )
            print(f'Saved tokenized features to {cache_path}')

        # Shape: (N, seq_len, 3)  — same layout as MultiNLI
        return torch.stack((input_ids, input_masks, segment_ids), dim=2)

    # ── Dataset interface ────────────────────────────────────────────────────

    def __len__(self):
        return len(self.y_array)

    def __getitem__(self, idx):
        return self.x_array[idx], self.y_array[idx], self.group_array[idx]

    def group_str(self, group_idx):
        y = group_idx // 2
        c = group_idx % 2
        return f'{self.target_name} = {int(y)}, {self.confounder_names[0]} = {int(c)}'
