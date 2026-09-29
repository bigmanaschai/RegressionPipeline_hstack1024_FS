from __future__ import annotations

import json
import platform
from pathlib import Path
from typing import Mapping, Optional

import joblib
import numpy as np
import pandas as pd

import hstack1024_pipeline
from hstack1024_pipeline.config import (
    DATASETS,
    FAMILY_ORDER,
    checkpoint_path,
    fingerprint_transformer_path,
    normalize_dataset,
    raw_csv_path,
)
from hstack1024_pipeline.extractors import INFERENCE_DEVICE
from hstack1024_pipeline.io_contract import (
    FeatureSplit,
    array_hash,
    file_hash,
    hstack_families,
    ordered_text_hash,
    write_json,
)
from hstack1024_pipeline.metrics import regression_metrics
from hstack1024_pipeline.pipeline import _extract_from_splits, _load_verified_splits

from .artifacts import (
    apply_frozen_minmax,
    apply_frozen_selector,
    load_frozen_bundle,
    load_reference_row,
    predict_frozen_linear,
    selected_indices,
    validate_frozen_bundle,
)
from .config import (
    BASE_CONTRACT_VERSION,
    FS_METHODS,
    HSTACK_DIM,
    R2_TOLERANCE,
    VariantSpec,
    bundle_path,
    get_variant,
    normalize_method,
    reference_path,
)


PIPELINE_NAME = "HStack1024-MinMax-SelectKBest frozen final regression"


def _validate_base_contract() -> None:
    actual = getattr(hstack1024_pipeline, "PIPELINE_CONTRACT_VERSION", None)
    if actual != BASE_CONTRACT_VERSION:
        raise RuntimeError(
            f"Base HStack1024 contract is {actual!r}; expected {BASE_CONTRACT_VERSION!r}"
        )
    if INFERENCE_DEVICE != "cpu":
        raise RuntimeError(f"Inference device must be CPU, got {INFERENCE_DEVICE!r}")


def _prepare_dataset(
    dataset: str,
    *,
    base_root: Optional[Path],
    raw_csv: Optional[Path],
    split_paths: Mapping[str, Path],
) -> tuple[FeatureSplit, FeatureSplit, dict, dict, dict, Path]:
    _validate_base_contract()
    ds = normalize_dataset(dataset)
    raw_source = raw_csv_path(ds, base_root) if raw_csv is None else Path(raw_csv)
    splits, split_checks = _load_verified_splits(
        ds, base_root, raw_source, split_paths
    )
    validation_families, test_families = _extract_from_splits(ds, splits, base_root)
    validation = hstack_families(validation_families)
    test = hstack_families(test_families)
    return (
        validation,
        test,
        validation_families,
        test_families,
        split_checks,
        raw_source,
    )


def _write_extracted_features(
    output_dir: Path,
    dataset: str,
    validation_families: dict,
    test_families: dict,
) -> None:
    extracted_root = output_dir / "extracted_features"
    extracted_root.mkdir(parents=True, exist_ok=True)
    for split_name, families in (
        ("val", validation_families),
        ("test", test_families),
    ):
        for family, payload in families.items():
            joblib.dump(
                {"smiles": payload.smiles, "X": payload.X, "y": payload.y},
                extracted_root / f"feat_{family}_{dataset}_{split_name}.scl",
            )


def _source_artifacts(
    spec: VariantSpec,
    *,
    base_root: Optional[Path],
    fs_root: Optional[Path],
    repo_root: Optional[Path],
    raw_source: Path,
    split_paths: Mapping[str, Path],
) -> dict[str, Path]:
    sources = {
        "frozen_final_bundle": bundle_path(
            spec.dataset, spec.method, fs_root=fs_root, repo_root=repo_root
        ),
        "reference_table": reference_path(
            spec.dataset, spec.method, fs_root=fs_root, repo_root=repo_root
        ),
        "raw_csv": raw_source,
        "fingerprint_transformer": fingerprint_transformer_path(
            spec.dataset, base_root
        ),
    }
    for family in FAMILY_ORDER:
        sources[f"checkpoint_{family}"] = checkpoint_path(
            spec.dataset, family, base_root
        )
    for name, path in split_paths.items():
        sources[f"datasplit_{name}"] = Path(path)
    return sources


def _evaluate_variant(
    spec: VariantSpec,
    output_root: Path,
    validation: FeatureSplit,
    test: FeatureSplit,
    validation_families: dict,
    test_families: dict,
    split_checks: dict,
    raw_source: Path,
    split_paths: Mapping[str, Path],
    *,
    base_root: Optional[Path],
    fs_root: Optional[Path],
    repo_root: Optional[Path],
    tolerance: float,
) -> dict:
    output_dir = Path(output_root) / spec.dataset / spec.method
    output_dir.mkdir(parents=True, exist_ok=True)
    model_bundle_path = bundle_path(
        spec.dataset, spec.method, fs_root=fs_root, repo_root=repo_root
    )
    result_reference_path = reference_path(
        spec.dataset, spec.method, fs_root=fs_root, repo_root=repo_root
    )
    bundle = load_frozen_bundle(model_bundle_path)
    reference = load_reference_row(result_reference_path, spec, bundle)
    structural_checks = validate_frozen_bundle(
        spec, bundle, reference, validation.X
    )
    structural_checks["reference_test_rows_match"] = int(reference["n_test"]) == len(test.y)
    if not structural_checks["reference_test_rows_match"]:
        raise AssertionError(f"{spec.variant_id}: reference Test row count mismatch")
    structural_checks.update(split_checks)
    structural_checks = {
        name: bool(passed) for name, passed in structural_checks.items()
    }

    validation_scaled = apply_frozen_minmax(validation.X, bundle["normalizer"])
    test_scaled = apply_frozen_minmax(test.X, bundle["normalizer"])
    validation_selected = apply_frozen_selector(validation_scaled, bundle)
    test_selected = apply_frozen_selector(test_scaled, bundle)
    indices = selected_indices(bundle)
    predictions = predict_frozen_linear(test_selected, bundle["model"])
    if not np.isfinite(validation_selected).all() or not np.isfinite(test_selected).all():
        raise AssertionError(f"{spec.variant_id}: non-finite selected feature value")
    if not np.isfinite(predictions).all():
        raise AssertionError(f"{spec.variant_id}: non-finite prediction")

    metrics = regression_metrics(test.y, predictions)
    reference_metrics = {
        name: float(reference[name])
        for name in ("TSR2", "TSRMSE", "TSMAE", "TSME", "TSPearson", "TSSpearman")
    }
    deltas = {name: metrics[name] - reference_metrics[name] for name in metrics}
    r2_pass = abs(deltas["TSR2"]) <= tolerance

    _write_extracted_features(
        output_dir,
        spec.dataset,
        validation_families,
        test_families,
    )
    selected_root = output_dir / "selected_features"
    selected_root.mkdir(parents=True, exist_ok=True)
    for split_name, payload, selected in (
        ("val", validation, validation_selected),
        ("test", test, test_selected),
    ):
        joblib.dump(
            {
                "smiles": payload.smiles,
                "X": selected,
                "y": payload.y,
                "idx": indices,
            },
            selected_root
            / f"feat_selectkbest_{spec.method}_{spec.dataset}_{split_name}_k{spec.selected_k}.scl",
        )
    pd.DataFrame(
        {
            "selected_rank": np.arange(1, len(indices) + 1, dtype=int),
            "feature_index": indices,
        }
    ).to_csv(output_dir / "selected_feature_indices.csv", index=False)
    scores = np.asarray(bundle["selector"].scores_, dtype=float)
    score_frame = pd.DataFrame(
        {"feature_index": np.arange(HSTACK_DIM, dtype=int), "score": scores}
    ).sort_values("score", ascending=False, na_position="last")
    score_frame.insert(0, "rank", np.arange(1, HSTACK_DIM + 1, dtype=int))
    score_frame.to_csv(output_dir / "feature_scores.csv", index=False)
    pd.DataFrame(
        {
            "Dataset": spec.dataset,
            "FS_Method": spec.method,
            "FS_k": spec.selected_k,
            "row_index": np.arange(len(test.y), dtype=int),
            "Smiles": test.smiles,
            "y_true": test.y,
            "y_pred": predictions,
            "residual": predictions - test.y,
        }
    ).to_csv(output_dir / "test_predictions.csv", index=False)

    report = {
        "Dataset": spec.dataset,
        "FS_Method": spec.method,
        "FS_k": spec.selected_k,
        "Model": spec.model_class,
        "Reference_Sheet": spec.reference_sheet,
        "Selection_Rule": spec.selection_rule,
        "feature_source": "extract",
        "precomputed_features_consumed": False,
        "training_performed": False,
        "n_val": int(len(validation.y)),
        "n_test": int(len(test.y)),
        "hstack_dim": int(test.X.shape[1]),
        "selected_dim": int(test_selected.shape[1]),
        **metrics,
        "Reference_Test_R2": reference_metrics["TSR2"],
        "Test_R2_delta": float(deltas["TSR2"]),
        "Tolerance": float(tolerance),
        "Status": "PASS" if r2_pass else "FAIL",
    }
    pd.DataFrame([report]).to_csv(
        output_dir / "reference_verification.csv", index=False
    )
    write_json(output_dir / "metrics.json", report)

    source_files = _source_artifacts(
        spec,
        base_root=base_root,
        fs_root=fs_root,
        repo_root=repo_root,
        raw_source=raw_source,
        split_paths=split_paths,
    )
    manifest = {
        "dataset": spec.dataset,
        "feature_selection_method": spec.method,
        "pipeline": PIPELINE_NAME,
        "base_contract": BASE_CONTRACT_VERSION,
        "training_performed": False,
        "precomputed_features_consumed": False,
        "feature_origin": "fresh extraction from raw-CSV-derived Validation/Test splits",
        "split_validation_oracle": "attached reproduce-00 outputs; never execution input",
        "inference_device": INFERENCE_DEVICE,
        "feature_order": list(FAMILY_ORDER),
        "feature_dim_each": 256,
        "hstack_dim": HSTACK_DIM,
        "normalization": "frozen Validation-fitted MinMaxScaler from final bundle",
        "feature_selection": {
            "class": "SelectKBest",
            "method": spec.method,
            "k": spec.selected_k,
            "selected_indices": indices.tolist(),
            "frozen_transform_only": True,
        },
        "model": {
            "class": spec.model_class,
            "parameters": spec.model_params,
            "selected_by": str(bundle.get("selected_by", "CVR2 on Validation")),
        },
        "quality_gate": {
            "metric": "Test_R2",
            "absolute_tolerance": float(tolerance),
        },
        "structural_checks": structural_checks,
        "metrics": metrics,
        "reference_metrics": reference_metrics,
        "reference_sheet": spec.reference_sheet,
        "selection_rule": spec.selection_rule,
        "reference_source_dataset_label": str(reference["Source_Dataset_Label"]),
        "metric_deltas": deltas,
        "status": report["Status"],
        "split_identity": {
            "val_smiles_sha256": ordered_text_hash(validation.smiles),
            "test_smiles_sha256": ordered_text_hash(test.smiles),
            "val_targets_sha256": array_hash(validation.y),
            "test_targets_sha256": array_hash(test.y),
        },
        "artifacts_sha256": {
            name: file_hash(path) for name, path in source_files.items()
        },
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "joblib": joblib.__version__,
            "inference_device": INFERENCE_DEVICE,
        },
    }
    try:
        import sklearn

        manifest["runtime"]["scikit_learn"] = sklearn.__version__
    except Exception:
        pass
    write_json(output_dir / "run_manifest.json", manifest)
    if not r2_pass:
        raise AssertionError(
            f"{spec.variant_id}: Test_R2={metrics['TSR2']:.12f}, "
            f"reference={reference_metrics['TSR2']:.12f}, "
            f"delta={deltas['TSR2']:+.12f}, tolerance={tolerance}"
        )
    return report


def run_variant(
    dataset: str,
    method: str,
    output_root: Path,
    *,
    feature_source: str = "extract",
    base_root: Optional[Path] = None,
    fs_root: Optional[Path] = None,
    repo_root: Optional[Path] = None,
    raw_csv: Optional[Path] = None,
    split_paths: Optional[Mapping[str, Path]] = None,
    tolerance: float = R2_TOLERANCE,
) -> dict:
    if feature_source != "extract":
        raise ValueError("Precomputed feature input is disabled; feature_source must be 'extract'")
    if split_paths is None:
        raise ValueError("split_paths is required")
    spec = get_variant(dataset, method)
    (
        validation,
        test,
        validation_families,
        test_families,
        split_checks,
        raw_source,
    ) = _prepare_dataset(
        spec.dataset,
        base_root=base_root,
        raw_csv=raw_csv,
        split_paths=split_paths,
    )
    return _evaluate_variant(
        spec,
        output_root,
        validation,
        test,
        validation_families,
        test_families,
        split_checks,
        raw_source,
        split_paths,
        base_root=base_root,
        fs_root=fs_root,
        repo_root=repo_root,
        tolerance=tolerance,
    )


def run_dataset(
    dataset: str,
    output_root: Path,
    *,
    methods: tuple[str, ...] = FS_METHODS,
    feature_source: str = "extract",
    base_root: Optional[Path] = None,
    fs_root: Optional[Path] = None,
    repo_root: Optional[Path] = None,
    raw_csv: Optional[Path] = None,
    split_paths: Optional[Mapping[str, Path]] = None,
    tolerance: float = R2_TOLERANCE,
) -> pd.DataFrame:
    if feature_source != "extract":
        raise ValueError("Precomputed feature input is disabled; feature_source must be 'extract'")
    if split_paths is None:
        raise ValueError("split_paths is required")
    ds = normalize_dataset(dataset)
    selected_methods = tuple(normalize_method(method) for method in methods)
    (
        validation,
        test,
        validation_families,
        test_families,
        split_checks,
        raw_source,
    ) = _prepare_dataset(
        ds,
        base_root=base_root,
        raw_csv=raw_csv,
        split_paths=split_paths,
    )
    reports = [
        _evaluate_variant(
            get_variant(ds, method),
            output_root,
            validation,
            test,
            validation_families,
            test_families,
            split_checks,
            raw_source,
            split_paths,
            base_root=base_root,
            fs_root=fs_root,
            repo_root=repo_root,
            tolerance=tolerance,
        )
        for method in selected_methods
    ]
    return pd.DataFrame(reports)


def run_all(
    output_root: Path,
    *,
    methods: tuple[str, ...] = FS_METHODS,
    feature_source: str = "extract",
    base_root: Optional[Path] = None,
    fs_root: Optional[Path] = None,
    repo_root: Optional[Path] = None,
    raw_csv_paths: Optional[Mapping[str, Path]] = None,
    split_paths_by_dataset: Optional[Mapping[str, Mapping[str, Path]]] = None,
    tolerance: float = R2_TOLERANCE,
) -> pd.DataFrame:
    if split_paths_by_dataset is None:
        raise ValueError("split_paths_by_dataset is required")
    frames = []
    for dataset in DATASETS:
        frames.append(
            run_dataset(
                dataset,
                output_root,
                methods=methods,
                feature_source=feature_source,
                base_root=base_root,
                fs_root=fs_root,
                repo_root=repo_root,
                raw_csv=None if raw_csv_paths is None else raw_csv_paths[dataset],
                split_paths=split_paths_by_dataset[dataset],
                tolerance=tolerance,
            )
        )
    result = pd.concat(frames, ignore_index=True)
    Path(output_root).mkdir(parents=True, exist_ok=True)
    result.to_csv(Path(output_root) / "reference_verification.csv", index=False)
    summary = {
        "pipeline": PIPELINE_NAME,
        "variant_count": int(len(result)),
        "training_performed": False,
        "precomputed_features_consumed": False,
        "all_pass": bool(result["Status"].eq("PASS").all()),
        "variants": json.loads(result.to_json(orient="records")),
    }
    write_json(Path(output_root) / "reference_verification.json", summary)
    return result
