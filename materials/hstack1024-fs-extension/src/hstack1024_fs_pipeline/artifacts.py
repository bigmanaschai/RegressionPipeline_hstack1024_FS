from __future__ import annotations

import ast
import sys
import warnings
from contextlib import contextmanager
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .config import HSTACK_DIM, VariantSpec


def _corr_abs(A, b, method):
    """Historical absolute Pearson/Spearman score used by the source notebook."""
    if method == "spearman":
        from scipy.stats import rankdata

        A = np.apply_along_axis(rankdata, 0, A)
        b = rankdata(b)
    centered_y = b - b.mean()
    centered_X = A - A.mean(0)
    numerator = (centered_X * centered_y[:, None]).sum(0)
    denominator = np.sqrt((centered_X**2).sum(0)) * np.sqrt((centered_y**2).sum())
    return np.abs(numerator / (denominator + 1e-12))


@contextmanager
def _historical_pickle_symbols():
    """Expose the notebook-local Pearson function while loading its artifact."""
    main = sys.modules["__main__"]
    marker = object()
    previous = getattr(main, "_corr_abs", marker)
    main._corr_abs = _corr_abs
    try:
        yield
    finally:
        if previous is marker:
            delattr(main, "_corr_abs")
        else:
            main._corr_abs = previous


def load_frozen_bundle(path: Path) -> dict:
    with _historical_pickle_symbols(), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        payload = joblib.load(path)
    required = {
        "dataset",
        "model_name",
        "FS_Method",
        "FS_k",
        "model",
        "normalizer",
        "selector",
        "feature_order",
        "feat_dim",
        "TSR2",
    }
    missing = required.difference(payload)
    if missing:
        raise AssertionError(f"{path}: frozen bundle is missing keys {sorted(missing)}")
    return payload


def selected_indices(bundle: dict) -> np.ndarray:
    indices = np.asarray(bundle["selector"].get_support(indices=True), dtype=int)
    if indices.ndim != 1 or len(indices) != int(bundle["FS_k"]):
        raise AssertionError("Frozen selector support does not match FS_k")
    if len(np.unique(indices)) != len(indices) or np.any(indices < 0) or np.any(indices >= HSTACK_DIM):
        raise AssertionError("Frozen selector support contains invalid feature indices")
    return indices


def load_reference_row(path: Path, spec: VariantSpec, bundle: dict) -> dict:
    frame = pd.read_csv(path)
    selected = frame.loc[
        frame["Model"].astype(str).eq(str(bundle["model_name"]))
        & frame["FS_Method"].astype(str).eq(spec.method)
        & frame["FS_k"].astype(int).eq(spec.selected_k)
    ]
    if len(selected) != 1:
        raise AssertionError(
            f"{path}: expected one reference row for {spec.variant_id}, found {len(selected)}"
        )
    row = selected.iloc[0].to_dict()
    row["Source_Dataset_Label"] = str(row["Dataset"])
    return row


def parse_selected_features(value) -> np.ndarray:
    parsed = ast.literal_eval(str(value))
    return np.asarray(parsed, dtype=int)


def apply_frozen_minmax(X: np.ndarray, scaler) -> np.ndarray:
    transformed = np.asarray(X) * np.asarray(scaler.scale_) + np.asarray(scaler.min_)
    return transformed.astype(np.float32)


def apply_frozen_selector(X: np.ndarray, bundle: dict) -> np.ndarray:
    return np.asarray(X)[:, selected_indices(bundle)].astype(np.float32)


def predict_frozen_linear(X: np.ndarray, model) -> np.ndarray:
    features = np.asarray(X, dtype=np.float64)
    coefficients = np.asarray(model.coef_, dtype=np.float64).reshape(1, -1)
    intercept = np.asarray(model.intercept_, dtype=np.float64)
    return (np.sum(features * coefficients, axis=1) + intercept).reshape(-1)


def validate_frozen_bundle(
    spec: VariantSpec,
    bundle: dict,
    reference: dict,
    validation_features: np.ndarray,
) -> dict[str, bool]:
    scaler = bundle["normalizer"]
    selector = bundle["selector"]
    model = bundle["model"]
    support = selected_indices(bundle)
    reference_support = parse_selected_features(reference["Selected_Features"])
    checks = {
        "bundle_dataset_matches": str(bundle["dataset"]) == spec.dataset,
        "bundle_method_matches": str(bundle["FS_Method"]) == spec.method,
        "bundle_model_name_matches": str(bundle["model_name"]) == spec.model_class,
        "model_class_matches": model.__class__.__name__ == spec.model_class,
        "bundle_selected_k_matches": int(bundle["FS_k"]) == spec.selected_k,
        "bundle_input_dim_is_1024": int(bundle["feat_dim"]) == HSTACK_DIM,
        "scaler_input_dim_is_1024": int(scaler.n_features_in_) == HSTACK_DIM,
        "selector_input_dim_is_1024": int(selector.n_features_in_) == HSTACK_DIM,
        "selector_k_matches": int(selector.k) == spec.selected_k,
        "model_input_dim_matches_k": int(model.n_features_in_) == spec.selected_k,
        "feature_order_matches": list(bundle["feature_order"])
        == ["smiles", "selfies", "graph", "fingerprint"],
        "support_matches_reference": np.array_equal(support, reference_support),
        "reference_validation_rows_match": int(reference["n_val"])
        == len(validation_features),
        "reference_feature_dim_is_1024": int(reference["feat_dim"]) == HSTACK_DIM,
        "scaler_fit_rows_match_validation": int(scaler.n_samples_seen_)
        == len(validation_features),
        "scaler_min_matches_validation": np.allclose(
            scaler.data_min_, validation_features.min(axis=0), rtol=1e-6, atol=1e-5
        ),
        "scaler_max_matches_validation": np.allclose(
            scaler.data_max_, validation_features.max(axis=0), rtol=1e-6, atol=1e-5
        ),
        "bundle_reference_r2_matches": np.isclose(
            float(bundle["TSR2"]), float(reference["TSR2"]), rtol=0.0, atol=1e-12
        ),
    }
    # Both supplied ER tables contain a historical copy/paste label of "AR".
    # The user-designated ER paths, embedded bundle identity, and 505-row ER
    # contract are authoritative. Preserve the source label in provenance while
    # refusing any other mismatch.
    checks["reference_dataset_label_resolved"] = (
        str(reference["Source_Dataset_Label"]) == spec.dataset
        or (
            spec.dataset == "ER"
            and str(reference["Source_Dataset_Label"]) == "AR"
            and int(reference["n_val"]) == 505
            and int(reference["n_test"]) == 505
        )
    )
    for name, expected in spec.model_params.items():
        checks[f"model_parameter_{name}_matches"] = np.isclose(
            float(getattr(model, name)), float(expected), rtol=0.0, atol=1e-12
        )
    checks = {name: bool(passed) for name, passed in checks.items()}
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise AssertionError(f"{spec.variant_id}: frozen-artifact contract failed: {failed}")
    return checks
