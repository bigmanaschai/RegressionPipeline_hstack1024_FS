from __future__ import annotations

import hashlib
import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional

import joblib
import numpy as np
import pandas as pd

from .config import (
    DATASETS,
    FAMILY_ORDER,
    HSTACK_DIM,
    MODEL_NAME,
    baseline_workbook,
    normalize_dataset,
)


@dataclass(frozen=True)
class FeatureSplit:
    smiles: np.ndarray
    X: np.ndarray
    y: np.ndarray


@dataclass(frozen=True)
class SplitData:
    smiles: np.ndarray
    y: np.ndarray


def load_joblib(path: Path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return joblib.load(path)


def load_split_artifact(dataset: str, split: str, path: Path) -> SplitData:
    """Load and validate an output produced by reproduce-00-datasplit."""
    ds = normalize_dataset(dataset)
    if split not in {"train", "val", "test"}:
        raise ValueError(f"Unknown split: {split}")
    payload = load_joblib(path)
    missing = {"smiles", "targets"}.difference(payload)
    if missing:
        raise AssertionError(f"{path} is missing keys: {sorted(missing)}")
    smiles = np.asarray(payload["smiles"]).astype(str)
    targets = np.asarray(payload["targets"])
    if smiles.ndim != 1 or targets.ndim != 1 or len(smiles) != len(targets):
        raise AssertionError(f"{path}: invalid smiles/targets arrays")
    if not np.isfinite(targets).all():
        raise AssertionError(f"{path}: non-finite target value")
    expected_rows = {
        "train": None,
        "val": DATASETS[ds].n_val,
        "test": DATASETS[ds].n_test,
    }[split]
    if expected_rows is not None and len(targets) != expected_rows:
        raise AssertionError(
            f"{ds}/{split}: expected {expected_rows} rows, got {len(targets)}"
        )
    expected_identity = DATASETS[ds].ordered_split_sha256
    if split in {"val", "test"}:
        if ordered_text_hash(smiles) != expected_identity[f"{split}_smiles"]:
            raise AssertionError(f"{ds}/{split}: ordered SMILES identity changed")
        if array_hash(targets) != expected_identity[f"{split}_targets"]:
            raise AssertionError(f"{ds}/{split}: ordered target identity changed")
    return SplitData(smiles=smiles, y=targets)


def hstack_families(families: Dict[str, FeatureSplit]) -> FeatureSplit:
    first = families[FAMILY_ORDER[0]]
    for family in FAMILY_ORDER[1:]:
        current = families[family]
        if not np.array_equal(first.smiles, current.smiles):
            raise AssertionError(f"SMILES order differs for {family}")
        if not np.array_equal(first.y, current.y):
            raise AssertionError(f"Targets differ for {family}")
    X = np.hstack([families[name].X for name in FAMILY_ORDER])
    if X.shape != (len(first.y), HSTACK_DIM):
        raise AssertionError(f"Expected HStack shape (*, {HSTACK_DIM}), got {X.shape}")
    return FeatureSplit(first.smiles, X, first.y)


def load_reference_row(dataset: str, repo_root: Optional[Path] = None) -> dict:
    ds = normalize_dataset(dataset)
    path = baseline_workbook(repo_root)
    sheet = f"{ds}_HStack1024_baseline_MinMax"
    frame = pd.read_excel(path, sheet_name=sheet)
    row = frame.loc[frame["Model"].eq(MODEL_NAME)]
    if len(row) != 1:
        raise AssertionError(f"{path}/{sheet}: expected exactly one {MODEL_NAME} row")
    return row.iloc[0].to_dict()


def ordered_text_hash(values: Iterable[str]) -> str:
    text = "\n".join(str(value) for value in values)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def array_hash(array: np.ndarray) -> str:
    a = np.ascontiguousarray(array)
    h = hashlib.sha256()
    h.update(str(a.dtype).encode("ascii"))
    h.update(str(a.shape).encode("ascii"))
    h.update(a.tobytes())
    return h.hexdigest()


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
