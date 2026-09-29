from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DATASETS, normalize_dataset
from .io_contract import array_hash, ordered_text_hash


def canonical_split_hash(smiles) -> str:
    from rdkit import Chem

    canonical = sorted(Chem.MolToSmiles(Chem.MolFromSmiles(str(value))) for value in smiles)
    return hashlib.md5("\n".join(canonical).encode("utf-8")).hexdigest()[:16]


def preprocess_raw(raw_csv: Path) -> tuple[np.ndarray, np.ndarray]:
    """Apply the exact RDKit validity, duplicate, and finite-target filters."""
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    frame = pd.read_csv(raw_csv)
    required = {"Smiles", "pIC50"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{raw_csv} is missing columns: {sorted(missing)}")
    frame = frame[frame["Smiles"].apply(lambda value: Chem.MolFromSmiles(str(value)) is not None)]
    frame = frame.drop_duplicates(subset=["Smiles"])
    smiles = frame["Smiles"].values
    targets = frame["pIC50"].values
    finite_mask = ~np.isinf(targets)
    smiles, targets = smiles[finite_mask], targets[finite_mask]
    return np.asarray(smiles), np.asarray(targets)


def reproduce_split(dataset: str, raw_csv: Path) -> dict:
    """Reproduce the original RDKit-cleaned qcut-10 60/20/20 split exactly."""
    from sklearn.model_selection import train_test_split

    ds = normalize_dataset(dataset)
    smiles, targets = preprocess_raw(raw_csv)
    bins = pd.qcut(targets, q=10, labels=False, duplicates="drop")
    train_smiles, temp_smiles, train_y, temp_y, _, temp_bins = train_test_split(
        smiles,
        targets,
        bins,
        test_size=0.40,
        random_state=0,
        stratify=bins,
    )
    val_smiles, test_smiles, val_y, test_y = train_test_split(
        temp_smiles,
        temp_y,
        test_size=0.50,
        random_state=0,
        stratify=temp_bins,
    )
    result = {
        "train": (np.asarray(train_smiles), np.asarray(train_y)),
        "val": (np.asarray(val_smiles), np.asarray(val_y)),
        "test": (np.asarray(test_smiles), np.asarray(test_y)),
    }
    spec = DATASETS[ds]
    if len(result["val"][0]) != spec.n_val or len(result["test"][0]) != spec.n_test:
        raise AssertionError(
            f"{ds}: split sizes differ from the research contract: "
            f"val={len(result['val'][0])}, test={len(result['test'][0])}"
        )
    for split, expected_hash in spec.canonical_split_md5.items():
        actual = canonical_split_hash(result[split][0])
        if actual != expected_hash:
            raise AssertionError(
                f"{ds}/{split}: canonical split MD5 {actual} != {expected_hash}"
            )
    for split in ("val", "test"):
        smiles_hash = ordered_text_hash(result[split][0])
        targets_hash = array_hash(result[split][1])
        if smiles_hash != spec.ordered_split_sha256[f"{split}_smiles"]:
            raise AssertionError(f"{ds}/{split}: ordered SMILES identity changed")
        if targets_hash != spec.ordered_split_sha256[f"{split}_targets"]:
            raise AssertionError(f"{ds}/{split}: ordered targets identity changed")
    return result
