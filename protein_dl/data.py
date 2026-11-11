"""Data loading for protein secondary-structure classification.

Everything a notebook needs to go from "nothing on disk" to PyTorch
DataLoaders lives here. Notebook 00 walks through each function step by step.

Dataset
-------
PS4 (Peracha et al., 2023, CC0 licence): 18,731 non-redundant protein chains
from the PDB, each with its amino-acid sequence and its DSSP 8-state secondary
structure. Train / validation split is the official one shipped with PS4.

CB513: the classic independent benchmark of 513 proteins (NetSurfP-2.0
version), used here as the test set. PS4 was built to have no overlap with it.

Task
----
Per-residue 3-class classification (Q3): every amino acid is labelled
H (helix), E (strand) or C (coil).
"""

import os
import random
import urllib.request

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# Where the raw files come from (pinned to one commit so results are stable)
# ---------------------------------------------------------------------------
PS4_COMMIT = "eb77a309c38824d920da6b959d543d6d16880d5a"
PS4_BASE = f"https://raw.githubusercontent.com/omarperacha/ps4-dataset/{PS4_COMMIT}/ps4_data/data"
FILES = {
    "data.csv": f"{PS4_BASE}/data.csv",
    "chain_ids.npz": f"{PS4_BASE}/chain_ids.npz",
    "CB513.csv": f"{PS4_BASE}/cb513/CB513_HHblits.csv",
}

# ---------------------------------------------------------------------------
# Vocabularies
# ---------------------------------------------------------------------------
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"  # the 20 standard amino acids
PAD_IDX = 0  # index 0 is reserved for padding
UNK_IDX = len(AMINO_ACIDS) + 1  # 21: any unknown / non-standard residue
VOCAB_SIZE = len(AMINO_ACIDS) + 2  # 22 = PAD + 20 amino acids + UNK
AA_TO_IDX = {aa: i + 1 for i, aa in enumerate(AMINO_ACIDS)}

Q3_CLASSES = ["C", "H", "E"]  # coil, helix, strand
Q3_NAMES = ["Coil", "Helix", "Strand"]
NUM_CLASSES = len(Q3_CLASSES)
IGNORE_INDEX = -100  # label for positions the loss and metrics must skip

# DSSP 8 states -> 3 states (the standard reduction)
#   H (alpha helix), G (3-10 helix), I (pi helix)        -> H
#   E (extended strand), B (isolated beta bridge)         -> E
#   T (turn), S (bend), C / '-' (coil), P (polyproline)   -> C
Q8_TO_Q3 = {"H": "H", "G": "H", "I": "H", "E": "E", "B": "E",
            "T": "C", "S": "C", "C": "C", "P": "C", "-": "C", " ": "C"}


def encode_sequence(seq):
    """'MKV...' -> list of integers (1..20, 21 for unknown).

    In PS4, lowercase letters mark cysteines that form disulfide bridges
    (a DSSP convention), so we map them to 'C'.
    """
    out = []
    for ch in seq:
        if ch.islower():
            ch = "C"
        out.append(AA_TO_IDX.get(ch, UNK_IDX))
    return out


def encode_q8_as_q3(ss8):
    """'HHHGGEEBTTS...' -> list of integers 0 (C), 1 (H), 2 (E)."""
    return [Q3_CLASSES.index(Q8_TO_Q3.get(ch, "C")) for ch in ss8]


# ---------------------------------------------------------------------------
# Download + read
# ---------------------------------------------------------------------------
def download_data(data_dir="data", verbose=True):
    """Download the three raw files once (about 10 MB) into `data_dir`."""
    os.makedirs(data_dir, exist_ok=True)
    for name, url in FILES.items():
        path = os.path.join(data_dir, name)
        if os.path.exists(path):
            if verbose:
                print(f"already here: {path}")
            continue
        if verbose:
            print(f"downloading {name} ...")
        urllib.request.urlretrieve(url, path)
    return data_dir


def load_raw_tables(data_dir="data"):
    """Return three pandas DataFrames: train, valid, test (CB513).

    Each has columns: id, sequence, ss8, mask  (mask is a string of 0/1,
    1 = residue is scored).
    """
    download_data(data_dir, verbose=False)
    ps4 = pd.read_csv(os.path.join(data_dir, "data.csv"))
    split = np.load(os.path.join(data_dir, "chain_ids.npz"))  # no pickles inside
    train_ids, valid_ids = set(split["train"]), set(split["valid"])

    ps4 = ps4.rename(columns={"chain_id": "id", "input": "sequence", "dssp8": "ss8"})
    ps4["mask"] = ps4["sequence"].str.len().apply(lambda n: "1" * n)
    cols = ["id", "sequence", "ss8", "mask"]
    train = ps4[ps4["id"].isin(train_ids)][cols].reset_index(drop=True)
    valid = ps4[ps4["id"].isin(valid_ids)][cols].reset_index(drop=True)

    cb = pd.read_csv(os.path.join(data_dir, "CB513.csv"), skipinitialspace=True)
    cb.columns = [c.strip() for c in cb.columns]
    test = pd.DataFrame({
        "id": [f"cb513_{i}" for i in range(len(cb))],
        "sequence": cb["input"].str.replace(" ", "", regex=False),
        "ss8": cb["dssp8"].str.replace(" ", "", regex=False),
        # CB513 mask: 1 = residue belongs to the original CB513 benchmark
        "mask": cb["cb513_mask"].apply(lambda s: "".join("1" if float(v) > 0.5 else "0" for v in s.split())),
    })
    return train, valid, test


# ---------------------------------------------------------------------------
# PyTorch Dataset and batching
# ---------------------------------------------------------------------------
class ProteinSSDataset(Dataset):
    """One item = one protein: (token tensor of shape [L], label tensor of shape [L]).

    If `max_len` is set, longer proteins are randomly cropped to `max_len`
    residues each time they are loaded (training only). This keeps GPU memory
    small and acts as light data augmentation.
    """

    def __init__(self, table, max_len=None):
        self.ids = list(table["id"])
        self.tokens = [encode_sequence(s) for s in table["sequence"]]
        labels = []
        for ss, mask in zip(table["ss8"], table["mask"]):
            y = encode_q8_as_q3(ss)
            y = [lab if m == "1" else IGNORE_INDEX for lab, m in zip(y, mask)]
            labels.append(y)
        self.labels = labels
        self.max_len = max_len

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, i):
        x, y = self.tokens[i], self.labels[i]
        if self.max_len is not None and len(x) > self.max_len:
            start = random.randint(0, len(x) - self.max_len)
            x = x[start:start + self.max_len]
            y = y[start:start + self.max_len]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


def collate_batch(batch):
    """Pad a list of proteins with different lengths into one rectangular batch.

    Returns
      tokens  [B, L_max]  long   (PAD_IDX where there is no residue)
      labels  [B, L_max]  long   (IGNORE_INDEX where there is no residue)
      lengths [B]         long   (true length of each protein)
    """
    xs, ys = zip(*batch)
    lengths = torch.tensor([len(x) for x in xs], dtype=torch.long)
    tokens = torch.nn.utils.rnn.pad_sequence(xs, batch_first=True, padding_value=PAD_IDX)
    labels = torch.nn.utils.rnn.pad_sequence(ys, batch_first=True, padding_value=IGNORE_INDEX)
    return tokens, labels, lengths


def make_dataloaders(data_dir="data", batch_size=32, max_len=512, train_subset=None, seed=42):
    """Build train / valid / test DataLoaders in one call.

    train_subset: use only the first N training proteins (for a quick try on CPU).
    """
    train, valid, test = load_raw_tables(data_dir)
    if train_subset is not None:
        train = train.sample(n=min(train_subset, len(train)), random_state=seed).reset_index(drop=True)
    g = torch.Generator().manual_seed(seed)
    train_dl = DataLoader(ProteinSSDataset(train, max_len=max_len), batch_size=batch_size,
                          shuffle=True, collate_fn=collate_batch, generator=g)
    # For evaluation we keep full-length proteins and sort by length so that
    # proteins of similar size share a batch (less padding = faster).
    valid = valid.assign(_n=valid["sequence"].str.len()).sort_values("_n").drop(columns="_n")
    test = test.assign(_n=test["sequence"].str.len()).sort_values("_n").drop(columns="_n")
    valid_dl = DataLoader(ProteinSSDataset(valid), batch_size=batch_size, shuffle=False, collate_fn=collate_batch)
    test_dl = DataLoader(ProteinSSDataset(test), batch_size=batch_size, shuffle=False, collate_fn=collate_batch)
    return train_dl, valid_dl, test_dl
