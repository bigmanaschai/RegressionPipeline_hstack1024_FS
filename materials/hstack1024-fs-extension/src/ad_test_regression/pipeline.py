from __future__ import annotations

from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.neighbors import NearestNeighbors

from hstack1024_pipeline.extractors import INFERENCE_DEVICE, extract_all
from hstack1024_pipeline.io_contract import FeatureSplit, hstack_families
from hstack1024_pipeline.pipeline import _load_verified_splits
from hstack1024_fs_pipeline.artifacts import (
    apply_frozen_minmax,
    apply_frozen_selector,
    load_frozen_bundle,
    load_reference_row,
    predict_frozen_linear,
    validate_frozen_bundle,
)
from hstack1024_fs_pipeline.config import (
    R2_TOLERANCE,
    VariantSpec,
    bundle_path,
    get_variant,
    reference_path,
)


DEFAULT_AD_K_VALUES = tuple(range(3, 26))
DEFAULT_THRESHOLD_MULTIPLIERS = (0.5, 2.0)
DISTANCE_METRIC = "euclidean"
RESULT_COLUMNS = (
    "K",
    "Threshold_Multiplier",
    "AD_Distance_Threshold",
    "R2",
    "RMSE",
    "MAE",
    "ME",
    "Pearson",
    "Spearman",
    "N_Test",
    "INDs",
    "Coverage",
    "OODs",
)


def _validated_vector(name: str, values: np.ndarray) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64).reshape(-1)
    if vector.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.isfinite(vector).all():
        raise ValueError(f"{name} contains a non-finite value")
    return vector


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Return regression metrics using residual = prediction - observation."""
    observed = _validated_vector("y_true", y_true)
    predicted = _validated_vector("y_pred", y_pred)
    if observed.shape != predicted.shape:
        raise ValueError(
            f"y_true/y_pred shape mismatch: {observed.shape} != {predicted.shape}"
        )
    residual = predicted - observed
    ss_res = float(np.sum(residual * residual))
    centered = observed - float(np.mean(observed))
    ss_tot = float(np.sum(centered * centered))
    r2 = float("nan") if len(observed) < 2 or ss_tot == 0.0 else 1.0 - ss_res / ss_tot
    pearson = (
        float("nan")
        if len(observed) < 2 or np.std(observed) == 0.0 or np.std(predicted) == 0.0
        else float(pearsonr(observed, predicted)[0])
    )
    spearman = (
        float("nan")
        if len(observed) < 2 or np.std(observed) == 0.0 or np.std(predicted) == 0.0
        else float(spearmanr(observed, predicted)[0])
    )
    return {
        "R2": float(r2),
        "RMSE": float(np.sqrt(np.mean(residual * residual))),
        "MAE": float(np.mean(np.abs(residual))),
        "ME": float(np.mean(residual)),
        "Pearson": pearson,
        "Spearman": spearman,
    }


def _validate_controls(
    threshold_multipliers: Sequence[float],
    k_values: Sequence[int],
) -> tuple[tuple[float, ...], tuple[int, ...]]:
    multipliers = tuple(float(value) for value in threshold_multipliers)
    if not multipliers:
        raise ValueError("threshold_multipliers must contain at least one value")
    if len(set(multipliers)) != len(multipliers):
        raise ValueError("threshold_multipliers contains a duplicate value")
    if not np.isfinite(multipliers).all() or any(value < 0.0 for value in multipliers):
        raise ValueError("threshold multipliers must be finite and non-negative")

    neighbors = tuple(int(value) for value in k_values)
    if not neighbors:
        raise ValueError("k_values must contain at least one value")
    if len(set(neighbors)) != len(neighbors):
        raise ValueError("k_values contains a duplicate value")
    if any(value < 1 for value in neighbors):
        raise ValueError("all AD k values must be positive integers")
    return multipliers, neighbors


def _extract_all_partitions(
    dataset: str,
    splits: Mapping[str, object],
    base_root: Optional[Path],
) -> dict[str, FeatureSplit]:
    split_order = ("train", "val", "test")
    boundaries: dict[str, tuple[int, int]] = {}
    smiles_parts = []
    target_parts = []
    start = 0
    for split_name in split_order:
        payload = splits[split_name]
        stop = start + len(payload.y)
        boundaries[split_name] = (start, stop)
        smiles_parts.append(payload.smiles)
        target_parts.append(payload.y)
        start = stop

    combined_smiles = np.concatenate(smiles_parts).astype(str)
    combined_targets = np.concatenate(target_parts).astype(float)
    extracted = extract_all(dataset, combined_smiles, combined_targets, base_root)
    result: dict[str, FeatureSplit] = {}
    for split_name in split_order:
        left, right = boundaries[split_name]
        families = {
            family: FeatureSplit(
                bundle.smiles[left:right],
                bundle.X[left:right],
                bundle.y[left:right],
            )
            for family, bundle in extracted.items()
        }
        result[split_name] = hstack_families(families)
    return result


def _summary_row(
    label: str,
    multiplier: float,
    distance_threshold: float,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_test: int,
    n_ind: int,
) -> dict[str, float | int | str]:
    metrics = regression_metrics(y_true, y_pred)
    n_ood = int(n_test - n_ind)
    return {
        "K": label,
        "Threshold_Multiplier": float(multiplier),
        "AD_Distance_Threshold": float(distance_threshold),
        **metrics,
        "N_Test": int(n_test),
        "INDs": int(n_ind),
        "Coverage": float(n_ind / n_test),
        "OODs": n_ood,
    }


def build_ad_summaries(
    X_reference: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    *,
    threshold_multipliers: Sequence[float] = DEFAULT_THRESHOLD_MULTIPLIERS,
    k_values: Sequence[int] = DEFAULT_AD_K_VALUES,
) -> dict[float, pd.DataFrame]:
    """Build one No-AD/kNN summary table for each threshold multiplier."""
    multipliers, neighbors = _validate_controls(threshold_multipliers, k_values)
    reference = np.asarray(X_reference, dtype=np.float64)
    test = np.asarray(X_test, dtype=np.float64)
    if reference.ndim != 2 or test.ndim != 2:
        raise ValueError("X_reference and X_test must be two-dimensional")
    if reference.shape[1] != test.shape[1]:
        raise ValueError("Reference and Test feature dimensions differ")
    if not np.isfinite(reference).all() or not np.isfinite(test).all():
        raise ValueError("Reference/Test features contain a non-finite value")
    if max(neighbors) > len(reference):
        raise ValueError(
            f"Maximum AD k={max(neighbors)} exceeds {len(reference)} reference rows"
        )
    observed = _validated_vector("y_test", y_test)
    predicted = _validated_vector("y_pred", y_pred)
    if len(observed) != len(test) or observed.shape != predicted.shape:
        raise ValueError("Test features, observations, and predictions do not align")

    rows_by_multiplier: dict[float, list[dict]] = {value: [] for value in multipliers}
    no_ad_metrics = regression_metrics(observed, predicted)
    for multiplier in multipliers:
        rows_by_multiplier[multiplier].append(
            {
                "K": "No AD",
                "Threshold_Multiplier": float(multiplier),
                "AD_Distance_Threshold": float("nan"),
                **no_ad_metrics,
                "N_Test": int(len(observed)),
                "INDs": int(len(observed)),
                "Coverage": 1.0,
                "OODs": 0,
            }
        )

    for ad_k in neighbors:
        knn = NearestNeighbors(
            n_neighbors=ad_k,
            metric=DISTANCE_METRIC,
            algorithm="auto",
        )
        knn.fit(reference)
        reference_distances, _ = knn.kneighbors(reference)
        reference_mean_dist = reference_distances.mean(axis=1)
        test_distances, _ = knn.kneighbors(test)
        test_mean_dist = test_distances.mean(axis=1)
        center = float(reference_mean_dist.mean())
        spread = float(reference_mean_dist.std(ddof=0))

        previous_mask = None
        for multiplier in sorted(multipliers):
            threshold = center + float(multiplier) * spread
            ind_mask = np.asarray(test_mean_dist <= threshold, dtype=bool)
            if previous_mask is not None and np.any(previous_mask & ~ind_mask):
                raise AssertionError("IND membership decreased for a larger threshold")
            previous_mask = ind_mask
            if not ind_mask.any():
                raise ValueError(
                    f"No IND Test rows remain for AD k={ad_k}, multiplier={multiplier}"
                )
            rows_by_multiplier[multiplier].append(
                _summary_row(
                    f"k={ad_k}",
                    multiplier,
                    threshold,
                    observed[ind_mask],
                    predicted[ind_mask],
                    len(observed),
                    int(ind_mask.sum()),
                )
            )

    summaries = {
        multiplier: pd.DataFrame(rows, columns=RESULT_COLUMNS)
        for multiplier, rows in rows_by_multiplier.items()
    }
    expected_labels = ["No AD", *[f"k={value}" for value in neighbors]]
    for multiplier, frame in summaries.items():
        if frame["K"].tolist() != expected_labels:
            raise AssertionError(f"Unexpected K rows for multiplier={multiplier}")
        if not np.array_equal(frame["INDs"] + frame["OODs"], frame["N_Test"]):
            raise AssertionError("IND/OOD counts do not reconcile to N_Test")
    return summaries


def _threshold_token(multiplier: float) -> str:
    text = np.format_float_positional(float(multiplier), trim="-")
    if "." not in text:
        text += ".0"
    return text.replace("-", "m").replace(".", "p")


def _output_filename(spec: VariantSpec, multiplier: float) -> str:
    return (
        f"AD_Test_Regression_{spec.dataset}_{spec.method}_"
        f"threshold_{_threshold_token(multiplier)}.csv"
    )


def run_test_ad_variant(
    dataset: str,
    method: str,
    output_root: Path,
    *,
    base_root: Optional[Path],
    fs_root: Optional[Path],
    repo_root: Optional[Path],
    raw_csv: Path,
    split_paths: Mapping[str, Path],
    threshold_multipliers: Sequence[float] = DEFAULT_THRESHOLD_MULTIPLIERS,
    k_values: Sequence[int] = DEFAULT_AD_K_VALUES,
    tolerance: float = R2_TOLERANCE,
) -> dict[float, Path]:
    """Run one frozen regression variant and write one AD CSV per multiplier."""
    if INFERENCE_DEVICE != "cpu":
        raise RuntimeError(f"Inference device must be CPU, got {INFERENCE_DEVICE!r}")
    spec = get_variant(dataset, method)
    splits, _ = _load_verified_splits(
        spec.dataset,
        base_root,
        Path(raw_csv),
        split_paths,
    )
    extracted = _extract_all_partitions(spec.dataset, splits, base_root)
    train = extracted["train"]
    validation = extracted["val"]
    test = extracted["test"]

    model_path = bundle_path(
        spec.dataset,
        spec.method,
        fs_root=fs_root,
        repo_root=repo_root,
    )
    table_path = reference_path(
        spec.dataset,
        spec.method,
        fs_root=fs_root,
        repo_root=repo_root,
    )
    bundle = load_frozen_bundle(model_path)
    reference_row = load_reference_row(table_path, spec, bundle)
    validate_frozen_bundle(spec, bundle, reference_row, validation.X)

    train_selected = apply_frozen_selector(
        apply_frozen_minmax(train.X, bundle["normalizer"]), bundle
    )
    validation_selected = apply_frozen_selector(
        apply_frozen_minmax(validation.X, bundle["normalizer"]), bundle
    )
    test_selected = apply_frozen_selector(
        apply_frozen_minmax(test.X, bundle["normalizer"]), bundle
    )
    X_reference = np.vstack([train_selected, validation_selected])
    predictions = predict_frozen_linear(test_selected, bundle["model"])
    summaries = build_ad_summaries(
        X_reference,
        test_selected,
        test.y,
        predictions,
        threshold_multipliers=threshold_multipliers,
        k_values=k_values,
    )

    no_ad_r2 = float(next(iter(summaries.values())).iloc[0]["R2"])
    if abs(no_ad_r2 - float(spec.reference_test_r2)) > float(tolerance):
        raise AssertionError(
            f"{spec.variant_id}: No-AD Test R2={no_ad_r2:.12f}, "
            f"reference={spec.reference_test_r2:.12f}, tolerance={tolerance}"
        )

    output_dir = Path(output_root) / spec.dataset / spec.method
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[float, Path] = {}
    for multiplier, frame in summaries.items():
        output_path = output_dir / _output_filename(spec, multiplier)
        frame.to_csv(output_path, index=False)
        paths[float(multiplier)] = output_path
    return paths

