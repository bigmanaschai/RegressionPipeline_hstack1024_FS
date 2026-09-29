"""Production frozen-regression inference with kNN applicability domain.

The new-data path intentionally contains no historical metric replay, split
oracle, or reference-result comparison. Frozen AR/ER/GR/PR artifacts are used
for inference, while the raw-derived training partition defines the professor's
kNN applicability domain for the supplied query compounds.
"""

from __future__ import annotations

import json
import platform
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors

from hstack1024_pipeline.config import FAMILY_ORDER, normalize_dataset, raw_csv_path
from hstack1024_pipeline.extractors import INFERENCE_DEVICE, extract_all
from hstack1024_pipeline.io_contract import FeatureSplit, file_hash, hstack_families, write_json
from hstack1024_pipeline.preprocessing import preprocess_raw

from .artifacts import (
    apply_frozen_minmax,
    apply_frozen_selector,
    load_frozen_bundle,
    predict_frozen_linear,
    selected_indices,
)
from .config import FS_METHODS, HSTACK_DIM, bundle_path, normalize_method


OOD_K_VALUES = tuple(range(3, 26))
OOD_PROTOCOL = "kNN mean distance <= training mean + 0.5 * training SD"


def _validated_matrix(name: str, values: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional matrix")
    if matrix.shape[1] != HSTACK_DIM:
        raise ValueError(
            f"{name} must contain {HSTACK_DIM} features, got {matrix.shape[1]}"
        )
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} contains a non-finite value")
    return matrix


def knn_applicability_domain(
    X_train: np.ndarray,
    X_query: np.ndarray,
    k: int = 6,
) -> tuple[np.ndarray, float]:
    """Return the professor-defined IND mask and distance threshold."""
    train = _validated_matrix("X_train", X_train)
    query = _validated_matrix("X_query", X_query)
    if not isinstance(k, (int, np.integer)) or int(k) < 1:
        raise ValueError("k must be a positive integer")
    k = int(k)
    if k > len(train):
        raise ValueError(f"k={k} exceeds the {len(train)} training compounds")

    neighbours = NearestNeighbors(n_neighbors=k)
    neighbours.fit(train)
    train_distances, _ = neighbours.kneighbors(train)
    train_mean_distance = train_distances.mean(axis=1)
    threshold = float(
        train_mean_distance.mean() + 0.5 * train_mean_distance.std()
    )
    query_distances, _ = neighbours.kneighbors(query)
    query_mean_distance = query_distances.mean(axis=1)
    return np.asarray(query_mean_distance <= threshold, dtype=bool), threshold


def analyze_query_ood(
    X_train: np.ndarray,
    X_query: np.ndarray,
    query_ids: Sequence[str],
    smiles: Sequence[str],
    *,
    k_values: Sequence[int] = OOD_K_VALUES,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create per-compound IND/OOD labels and per-k coverage summaries."""
    train = _validated_matrix("X_train", X_train)
    query = _validated_matrix("X_query", X_query)
    ids = np.asarray(query_ids).astype(str)
    smiles_array = np.asarray(smiles).astype(str)
    if not (len(query) == len(ids) == len(smiles_array)):
        raise ValueError("Query features, IDs, and SMILES row counts differ")
    requested_k = tuple(int(k) for k in k_values)
    if requested_k != OOD_K_VALUES:
        raise ValueError("The controlled OOD protocol requires every k from 3 through 25")

    result = pd.DataFrame(
        {
            "row_index": np.arange(len(query), dtype=int),
            "ID": ids,
            "Smiles": smiles_array,
        }
    )
    summaries = []
    for k in requested_k:
        inside, threshold = knn_applicability_domain(train, query, k=k)
        result[f"ADk{k}"] = np.where(inside, "IND", "OOD")
        summaries.append(
            {
                "k": k,
                "Threshold": threshold,
                "n_Total": int(len(query)),
                "n_IND": int(inside.sum()),
                "n_OOD": int((~inside).sum()),
                "IND_Coverage": float(inside.mean()),
            }
        )
    return result, pd.DataFrame(summaries)


def load_query_compounds(
    query_csv: Path,
    *,
    id_column: str = "ID",
    smiles_column: str = "Smiles",
) -> pd.DataFrame:
    """Load production query compounds and fail on unsafe schema/content."""
    from rdkit import Chem, RDLogger

    source = Path(query_csv)
    if not source.is_file():
        raise FileNotFoundError(source)
    frame = pd.read_csv(source)
    missing = {id_column, smiles_column}.difference(frame.columns)
    if missing:
        raise ValueError(f"{source} is missing columns: {sorted(missing)}")
    query = frame.loc[:, [id_column, smiles_column]].copy()
    if query[id_column].isna().any() or query[smiles_column].isna().any():
        raise ValueError(f"{source}: ID/SMILES contains a missing value")
    query[id_column] = query[id_column].astype(str).str.strip()
    query[smiles_column] = query[smiles_column].astype(str).str.strip()
    if query[id_column].eq("").any() or query[smiles_column].eq("").any():
        raise ValueError(f"{source}: ID/SMILES contains a blank value")
    if query[id_column].duplicated().any():
        raise ValueError(f"{source}: duplicate IDs are not allowed")
    RDLogger.DisableLog("rdApp.*")
    valid = query[smiles_column].map(lambda value: Chem.MolFromSmiles(value) is not None)
    if not valid.all():
        bad = query.loc[~valid, id_column].head(10).tolist()
        raise ValueError(f"{source}: invalid SMILES for IDs {bad}")
    return query.reset_index(drop=True)


def prepare_domain_training(raw_csv: Path) -> FeatureSplit:
    """Create the original 60% training partition without historical gates."""
    smiles, targets = preprocess_raw(Path(raw_csv))
    bins = pd.qcut(targets, q=10, labels=False, duplicates="drop")
    train_smiles, _, train_targets, _ = train_test_split(
        smiles,
        targets,
        test_size=0.40,
        random_state=0,
        stratify=bins,
    )
    return FeatureSplit(
        np.asarray(train_smiles).astype(str),
        np.empty((len(train_smiles), 0), dtype=np.float32),
        np.asarray(train_targets, dtype=float),
    )


def _extract_domain_and_query(
    dataset: str,
    training: FeatureSplit,
    query: pd.DataFrame,
    *,
    smiles_column: str,
    base_root: Optional[Path],
) -> tuple[FeatureSplit, FeatureSplit]:
    query_smiles = query[smiles_column].astype(str).to_numpy()
    combined_smiles = np.concatenate([training.smiles, query_smiles])
    combined_targets = np.concatenate(
        [training.y, np.zeros(len(query_smiles), dtype=float)]
    )
    extracted = extract_all(dataset, combined_smiles, combined_targets, base_root)
    n_train = len(training.smiles)
    training_families = {
        family: FeatureSplit(
            payload.smiles[:n_train], payload.X[:n_train], payload.y[:n_train]
        )
        for family, payload in extracted.items()
    }
    query_families = {
        family: FeatureSplit(
            payload.smiles[n_train:], payload.X[n_train:], payload.y[n_train:]
        )
        for family, payload in extracted.items()
    }
    return hstack_families(training_families), hstack_families(query_families)


def _run_production_variant(
    dataset: str,
    method: str,
    output_root: Path,
    training_hstack: FeatureSplit,
    query_hstack: FeatureSplit,
    query: pd.DataFrame,
    *,
    query_csv: Path,
    raw_csv: Path,
    id_column: str,
    smiles_column: str,
    fs_root: Optional[Path],
    repo_root: Optional[Path],
) -> pd.DataFrame:
    frozen_bundle_path = bundle_path(
        dataset, method, fs_root=fs_root, repo_root=repo_root
    )
    bundle = load_frozen_bundle(frozen_bundle_path)
    if str(bundle["dataset"]) != dataset or str(bundle["FS_Method"]) != method:
        raise AssertionError(f"{frozen_bundle_path}: variant identity mismatch")
    if int(bundle["feat_dim"]) != HSTACK_DIM:
        raise AssertionError(f"{frozen_bundle_path}: expected {HSTACK_DIM} inputs")

    train_scaled = apply_frozen_minmax(training_hstack.X, bundle["normalizer"])
    query_scaled = apply_frozen_minmax(query_hstack.X, bundle["normalizer"])
    query_selected = apply_frozen_selector(query_scaled, bundle)
    predictions = predict_frozen_linear(query_selected, bundle["model"])
    if not np.isfinite(predictions).all():
        raise AssertionError(f"{dataset}/{method}: non-finite prediction")

    result, summary = analyze_query_ood(
        train_scaled,
        query_scaled,
        query[id_column].astype(str).to_numpy(),
        query[smiles_column].astype(str).to_numpy(),
    )
    result.insert(0, "Predicted_pIC50", predictions)
    result.insert(0, "FS_k", int(bundle["FS_k"]))
    result.insert(0, "Model", str(bundle["model_name"]))
    result.insert(0, "FS_Method", method)
    result.insert(0, "Dataset", dataset)
    summary.insert(0, "FS_Method", method)
    summary.insert(0, "Dataset", dataset)

    output_dir = Path(output_root) / dataset / method
    output_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_dir / "production_predictions_ood.csv", index=False)
    summary.to_csv(output_dir / "ood_summary_k3_k25.csv", index=False)
    manifest = {
        "dataset": dataset,
        "feature_selection_method": method,
        "mode": "production new-compound inference with kNN applicability domain",
        "query_csv": str(query_csv),
        "query_rows": int(len(query)),
        "query_columns": {"id": id_column, "smiles": smiles_column},
        "query_sha256": file_hash(Path(query_csv)),
        "domain_reference_csv": str(raw_csv),
        "domain_reference_sha256": file_hash(Path(raw_csv)),
        "domain_reference": "raw-derived 60% training partition; no oracle comparison",
        "domain_reference_rows": int(len(training_hstack.y)),
        "feature_space": "frozen-MinMax-scaled HStack1024 before SelectKBest",
        "feature_dimension": HSTACK_DIM,
        "feature_order": list(FAMILY_ORDER),
        "ood_protocol_source": "standard_pipeline/OOD_ajPle/Readme.rtf",
        "ood_protocol": OOD_PROTOCOL,
        "k_values": list(OOD_K_VALUES),
        "labels": {"inside_domain": "IND", "outside_domain": "OOD"},
        "prediction_target": "pIC50",
        "ground_truth_available": False,
        "historical_reference_comparison_performed": False,
        "split_oracle_used": False,
        "regression_model_refitted": False,
        "normalizer_refitted": False,
        "selectkbest_refitted": False,
        "knn_index_fitted_for_domain_analysis": True,
        "frozen_bundle": str(frozen_bundle_path),
        "frozen_bundle_sha256": file_hash(frozen_bundle_path),
        "selected_feature_indices": selected_indices(bundle).tolist(),
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "inference_device": INFERENCE_DEVICE,
        },
        "outputs": {
            "predictions_and_ad": "production_predictions_ood.csv",
            "per_k_summary": "ood_summary_k3_k25.csv",
        },
    }
    write_json(output_dir / "production_manifest.json", manifest)
    return summary


def run_production_ood_dataset(
    dataset: str,
    query_csv: Path,
    output_root: Path,
    *,
    methods: Sequence[str] = FS_METHODS,
    base_root: Optional[Path] = None,
    fs_root: Optional[Path] = None,
    repo_root: Optional[Path] = None,
    raw_csv: Optional[Path] = None,
    id_column: str = "ID",
    smiles_column: str = "Smiles",
) -> pd.DataFrame:
    """Predict new compounds and assign IND/OOD for one receptor dataset."""
    ds = normalize_dataset(dataset)
    selected_methods = tuple(normalize_method(method) for method in methods)
    if not selected_methods:
        raise ValueError("At least one feature-selection method is required")
    raw_source = raw_csv_path(ds, base_root) if raw_csv is None else Path(raw_csv)
    query_source = Path(query_csv)
    query = load_query_compounds(
        query_source, id_column=id_column, smiles_column=smiles_column
    )
    training = prepare_domain_training(raw_source)
    training_hstack, query_hstack = _extract_domain_and_query(
        ds,
        training,
        query,
        smiles_column=smiles_column,
        base_root=base_root,
    )
    summaries = [
        _run_production_variant(
            ds,
            method,
            output_root,
            training_hstack,
            query_hstack,
            query,
            query_csv=query_source,
            raw_csv=raw_source,
            id_column=id_column,
            smiles_column=smiles_column,
            fs_root=fs_root,
            repo_root=repo_root,
        )
        for method in selected_methods
    ]
    report = pd.concat(summaries, ignore_index=True)
    report.to_csv(Path(output_root) / ds / "ood_summary.csv", index=False)
    return report


def run_production_ood_all(
    query_csv: Path,
    output_root: Path,
    *,
    datasets: Sequence[str],
    methods: Sequence[str] = FS_METHODS,
    base_root: Optional[Path] = None,
    fs_root: Optional[Path] = None,
    repo_root: Optional[Path] = None,
    raw_csv_paths: Mapping[str, Path],
    id_column: str = "ID",
    smiles_column: str = "Smiles",
) -> pd.DataFrame:
    """Run production prediction/OOD analysis for all requested receptors."""
    reports = [
        run_production_ood_dataset(
            dataset,
            query_csv,
            output_root,
            methods=methods,
            base_root=base_root,
            fs_root=fs_root,
            repo_root=repo_root,
            raw_csv=raw_csv_paths[dataset],
            id_column=id_column,
            smiles_column=smiles_column,
        )
        for dataset in datasets
    ]
    combined = pd.concat(reports, ignore_index=True)
    combined.to_csv(Path(output_root) / "ood_summary.csv", index=False)
    (Path(output_root) / "ood_summary.json").write_text(
        json.dumps(
            {
                "query_csv": str(query_csv),
                "ground_truth_available": False,
                "historical_reference_comparison_performed": False,
                "protocol": OOD_PROTOCOL,
                "k_values": list(OOD_K_VALUES),
                "rows": json.loads(combined.to_json(orient="records")),
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return combined
