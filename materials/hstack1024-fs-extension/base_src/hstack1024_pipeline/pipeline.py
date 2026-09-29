from __future__ import annotations

import json
import platform
from pathlib import Path
from typing import Dict, Mapping, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

from .config import (
    DATASETS,
    FAMILY_ORDER,
    HSTACK_DIM,
    MODEL_NAME,
    R2_TOLERANCE,
    RIDGE_ALPHA,
    baseline_workbook,
    checkpoint_path,
    fingerprint_transformer_path,
    normalize_dataset,
    raw_csv_path,
    ridge_path,
    scaler_path,
)
from .extractors import INFERENCE_DEVICE, extract_all
from .io_contract import (
    FeatureSplit,
    SplitData,
    array_hash,
    file_hash,
    hstack_families,
    load_joblib,
    load_reference_row,
    load_split_artifact,
    ordered_text_hash,
    write_json,
)
from .metrics import regression_metrics
from .preprocessing import canonical_split_hash, reproduce_split


def _validate_frozen_objects(dataset: str, scaler, ridge, val: FeatureSplit) -> dict:
    checks = {
        "model_class_is_ridge": ridge.__class__.__name__ == MODEL_NAME,
        "ridge_alpha_is_100": float(ridge.alpha) == RIDGE_ALPHA,
        "model_feature_dim_is_1024": int(ridge.n_features_in_) == HSTACK_DIM,
        "scaler_feature_dim_is_1024": int(scaler.n_features_in_) == HSTACK_DIM,
        "scaler_fit_row_count_matches_val": int(scaler.n_samples_seen_) == len(val.y),
        "scaler_data_min_matches_extracted_val": np.allclose(
            scaler.data_min_, val.X.min(axis=0), rtol=1e-6, atol=1e-5
        ),
        "scaler_data_max_matches_extracted_val": np.allclose(
            scaler.data_max_, val.X.max(axis=0), rtol=1e-6, atol=1e-5
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise AssertionError(f"{dataset}: frozen model/scaler contract failed: {failed}")
    return checks


def _apply_frozen_minmax(X: np.ndarray, scaler) -> np.ndarray:
    # Equivalent to the historical sklearn MinMaxScaler.transform.  Direct array
    # math avoids any accidental fitting and is stable when replaying a 1.2.2
    # artifact under a newer compatible sklearn runtime.
    transformed = np.asarray(X) * np.asarray(scaler.scale_) + np.asarray(scaler.min_)
    return transformed.astype(np.float32)


def _predict_frozen_ridge(X: np.ndarray, ridge) -> np.ndarray:
    # Equivalent to Ridge.predict for the fitted single-output estimator.
    # Accumulate in float64 to avoid platform BLAS warnings observed for the
    # historical float32 coefficients; the resulting R2 delta remains <1e-6.
    features = np.asarray(X, dtype=np.float64)
    coefficients = np.asarray(ridge.coef_, dtype=np.float64).reshape(1, -1)
    return np.sum(features * coefficients, axis=1) + np.asarray(
        ridge.intercept_, dtype=np.float64
    )


def _load_verified_splits(
    dataset: str,
    repo_root: Optional[Path],
    raw_csv: Optional[Path],
    split_paths: Optional[Mapping[str, Path]],
) -> Tuple[dict, dict]:
    if split_paths is None:
        raise ValueError(
            "split_paths is required: attach each reproduce-00-datasplit output "
            "and provide train/val/test .scl paths"
        )
    missing_keys = {"train", "val", "test", "manifest"}.difference(split_paths)
    if missing_keys:
        raise ValueError(f"Missing split paths: {sorted(missing_keys)}")
    source = raw_csv_path(dataset, repo_root) if raw_csv is None else Path(raw_csv)
    raw_derived_splits = {}
    checks = {}
    split_manifest = json.loads(
        Path(split_paths["manifest"]).read_text(encoding="utf-8")
    )
    if split_manifest.get("dataset") != dataset:
        raise AssertionError(f"{dataset}: reproduce-00 manifest dataset mismatch")
    generated_from_raw = reproduce_split(dataset, source)
    for split_name in ("train", "val", "test"):
        oracle = load_split_artifact(
            dataset, split_name, Path(split_paths[split_name])
        )
        count_matches = int(split_manifest["n"][split_name]) == len(oracle.y)
        md5_matches = (
            canonical_split_hash(oracle.smiles)
            == split_manifest["md5"][split_name]
        )
        checks[f"datasplit_{split_name}_count_matches_manifest"] = count_matches
        checks[f"datasplit_{split_name}_md5_matches_manifest"] = md5_matches
        if not count_matches or not md5_matches:
            raise AssertionError(
                f"{dataset}/{split_name}: split artifact differs from its "
                "reproduce-00 manifest"
            )

        # The execution split is recreated from the raw CSV.  The attached
        # reproduce-00 artifact is an oracle only and is never returned to the
        # feature extractors.
        generated_smiles, generated_targets = generated_from_raw[split_name]
        generated = SplitData(
            smiles=np.asarray(generated_smiles).astype(str),
            y=np.asarray(generated_targets),
        )
        smiles_match = np.array_equal(generated.smiles, oracle.smiles)
        targets_match = np.array_equal(generated.y, oracle.y)
        checks[
            f"raw_derived_{split_name}_smiles_match_reproduce00_oracle"
        ] = smiles_match
        checks[
            f"raw_derived_{split_name}_targets_match_reproduce00_oracle"
        ] = targets_match
        if not smiles_match or not targets_match:
            raise AssertionError(
                f"{dataset}/{split_name}: raw-derived split differs from the "
                "attached reproduce-00 validation oracle"
            )
        raw_derived_splits[split_name] = generated

    generated_smiles = np.concatenate(
        [raw_derived_splits[name].smiles for name in ("train", "val", "test")]
    )
    partitions_disjoint = len(np.unique(generated_smiles)) == len(generated_smiles)
    checks["raw_derived_partitions_are_disjoint"] = partitions_disjoint
    if not partitions_disjoint:
        raise AssertionError(f"{dataset}: molecule duplicated across raw-derived splits")
    return raw_derived_splits, checks


def _extract_from_splits(
    dataset: str,
    splits: dict,
    repo_root: Optional[Path],
) -> Tuple[Dict[str, FeatureSplit], Dict[str, FeatureSplit]]:
    val = splits["val"]
    test = splits["test"]
    # Extract in one model load per family, then restore the original split
    # boundary. No precomputed feature artifact is read here.
    combined_smiles = np.concatenate([val.smiles, test.smiles])
    combined_targets = np.concatenate([val.y, test.y])
    combined = extract_all(dataset, combined_smiles, combined_targets, repo_root)
    n_val = len(val.y)
    extracted_val = {
        family: FeatureSplit(
            bundle.smiles[:n_val], bundle.X[:n_val], bundle.y[:n_val]
        )
        for family, bundle in combined.items()
    }
    extracted_test = {
        family: FeatureSplit(
            bundle.smiles[n_val:], bundle.X[n_val:], bundle.y[n_val:]
        )
        for family, bundle in combined.items()
    }
    return extracted_val, extracted_test


def run_dataset(
    dataset: str,
    output_root: Path,
    *,
    feature_source: str = "extract",
    repo_root: Optional[Path] = None,
    raw_csv: Optional[Path] = None,
    split_paths: Optional[Mapping[str, Path]] = None,
    tolerance: float = R2_TOLERANCE,
) -> dict:
    ds = normalize_dataset(dataset)
    if feature_source != "extract":
        raise ValueError(
            "Precomputed feature input is disabled; feature_source must be 'extract'"
        )
    output_dir = Path(output_root) / ds
    output_dir.mkdir(parents=True, exist_ok=True)
    splits, split_checks = _load_verified_splits(
        ds, repo_root, raw_csv, split_paths
    )
    val_families, test_families = _extract_from_splits(
        ds, splits, repo_root
    )
    val = hstack_families(val_families)
    test = hstack_families(test_families)
    scaler = load_joblib(scaler_path(ds, repo_root))
    ridge = load_joblib(ridge_path(ds, repo_root))
    structural_checks = _validate_frozen_objects(ds, scaler, ridge, val)
    structural_checks.update(split_checks)
    X_test = _apply_frozen_minmax(test.X, scaler)
    predictions = _predict_frozen_ridge(X_test, ridge).reshape(-1)
    if not np.isfinite(predictions).all():
        raise AssertionError(f"{ds}: non-finite Ridge prediction")
    metrics = regression_metrics(test.y, predictions)
    reference = load_reference_row(ds, repo_root)
    reference_contract = {
        "reference_n_val_matches": int(reference["n_val"]) == len(val.y),
        "reference_n_test_matches": int(reference["n_test"]) == len(test.y),
        "reference_feature_dim_is_1024": int(reference["feat_dim"]) == HSTACK_DIM,
    }
    failed_reference_checks = [
        name for name, passed in reference_contract.items() if not passed
    ]
    if failed_reference_checks:
        raise AssertionError(
            f"{ds}: reference workbook contract failed: {failed_reference_checks}"
        )
    structural_checks.update(reference_contract)
    deltas = {name: metrics[name] - float(reference[name]) for name in metrics}
    r2_pass = abs(deltas["TSR2"]) <= tolerance

    prediction_frame = pd.DataFrame(
        {
            "Dataset": ds,
            "row_index": np.arange(len(test.y), dtype=int),
            "Smiles": test.smiles,
            "y_true": test.y,
            "y_pred": predictions,
            "residual": predictions - test.y,
        }
    )
    prediction_frame.to_csv(output_dir / "test_predictions.csv", index=False)
    report = {
        "Dataset": ds,
        "Model": MODEL_NAME,
        "feature_source": feature_source,
        "precomputed_features_consumed": False,
        "n_val": int(len(val.y)),
        "n_test": int(len(test.y)),
        "feat_dim": int(test.X.shape[1]),
        **metrics,
        "Reference_Test_R2": float(reference["TSR2"]),
        "Test_R2_delta": float(deltas["TSR2"]),
        "Tolerance": float(tolerance),
        "Status": "PASS" if r2_pass else "FAIL",
    }
    pd.DataFrame([report]).to_csv(output_dir / "reference_verification.csv", index=False)

    extracted_root = output_dir / "extracted_features"
    extracted_root.mkdir(parents=True, exist_ok=True)
    for split_name, families in (("val", val_families), ("test", test_families)):
        for family, bundle in families.items():
            joblib.dump(
                {"smiles": bundle.smiles, "X": bundle.X, "y": bundle.y},
                extracted_root / f"feat_{family}_{ds}_{split_name}.scl",
            )

    source_files = {
        "baseline_workbook": baseline_workbook(repo_root),
        "normalizer": scaler_path(ds, repo_root),
        "ridge": ridge_path(ds, repo_root),
        "raw_csv": raw_csv_path(ds, repo_root) if raw_csv is None else Path(raw_csv),
    }
    for split_name, path in split_paths.items():
        source_files[f"datasplit_{split_name}"] = Path(path)
    for family in FAMILY_ORDER:
        source_files[f"checkpoint_{family}"] = checkpoint_path(
            ds, family, repo_root
        )
    source_files["fingerprint_transformer"] = fingerprint_transformer_path(
        ds, repo_root
    )
    manifest = {
        "dataset": ds,
        "pipeline": "HStack1024_baseline_MinMax - Ridge",
        "training_performed": False,
        "precomputed_features_consumed": False,
        "feature_origin": "fresh extraction from raw-CSV-derived Validation/Test splits",
        "split_execution_source": "raw CSV -> RDKit cleaning -> qcut-10 stratified 60/20/20 split",
        "split_validation_oracle": "attached reproduce-00 train/validation/test outputs",
        "inference_device": INFERENCE_DEVICE,
        "feature_order": list(FAMILY_ORDER),
        "feature_dim_each": 256,
        "hstack_dim": HSTACK_DIM,
        "normalization": "frozen MinMaxScaler fitted on the historical Validation split",
        "model": {"class": MODEL_NAME, "alpha": float(ridge.alpha)},
        "test_policy": "the historical dataset-specific Test split is evaluated once",
        "quality_gate": {"metric": "Test_R2", "absolute_tolerance": tolerance},
        "structural_checks": structural_checks,
        "metrics": metrics,
        "reference_metrics": {name: float(reference[name]) for name in metrics},
        "metric_deltas": deltas,
        "status": report["Status"],
        "split_identity": {
            "val_smiles_sha256": ordered_text_hash(val.smiles),
            "test_smiles_sha256": ordered_text_hash(test.smiles),
            "val_targets_sha256": array_hash(val.y),
            "test_targets_sha256": array_hash(test.y),
        },
        "artifacts_sha256": {
            name: file_hash(path) for name, path in source_files.items()
        },
        "runtime": {
            "inference_device": INFERENCE_DEVICE,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "joblib": joblib.__version__,
        },
    }
    try:
        import sklearn

        manifest["runtime"]["scikit_learn"] = sklearn.__version__
    except Exception:
        pass
    write_json(output_dir / "run_manifest.json", manifest)
    write_json(output_dir / "metrics.json", report)
    if not r2_pass:
        raise AssertionError(
            f"{ds}: Test_R2={metrics['TSR2']:.12f}, reference={float(reference['TSR2']):.12f}, "
            f"delta={deltas['TSR2']:+.12f}, tolerance={tolerance}"
        )
    return report


def run_all(
    output_root: Path,
    *,
    feature_source: str = "extract",
    repo_root: Optional[Path] = None,
    raw_csv_paths: Optional[Mapping[str, Path]] = None,
    split_paths_by_dataset: Optional[Mapping[str, Mapping[str, Path]]] = None,
    tolerance: float = R2_TOLERANCE,
) -> pd.DataFrame:
    if feature_source != "extract":
        raise ValueError(
            "Precomputed feature input is disabled; feature_source must be 'extract'"
        )
    if split_paths_by_dataset is None:
        raise ValueError("split_paths_by_dataset is required")
    reports = [
        run_dataset(
            dataset,
            output_root,
            feature_source=feature_source,
            repo_root=repo_root,
            raw_csv=None if raw_csv_paths is None else raw_csv_paths[dataset],
            split_paths=split_paths_by_dataset[dataset],
            tolerance=tolerance,
        )
        for dataset in DATASETS
    ]
    frame = pd.DataFrame(reports)
    Path(output_root).mkdir(parents=True, exist_ok=True)
    frame.to_csv(Path(output_root) / "reference_verification.csv", index=False)
    write_json(
        Path(output_root) / "reference_verification.json",
        {
            "pipeline": "HStack1024_baseline_MinMax - Ridge",
            "training_performed": False,
            "precomputed_features_consumed": False,
            "all_pass": bool(frame["Status"].eq("PASS").all()),
            "datasets": reports,
        },
    )
    return frame
