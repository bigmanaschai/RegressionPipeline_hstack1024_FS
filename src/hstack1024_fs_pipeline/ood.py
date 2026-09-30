"""Production regression, professor-style activity reporting, and kNN OOD.

The query path never uses the held-out Test split. For each receptor, the
raw-derived Train and Validation partitions are combined into ``smiles_tr``
(80%), while every query SMILES is held in ``data``. Frozen HStack1024 assets
produce both matrices. OOD is calculated either in scaled HStack1024 space or
in the matching frozen-SelectKBest space.
"""

from __future__ import annotations

import json
import platform
from pathlib import Path
from typing import Mapping, Optional, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors

from hstack1024_pipeline.config import (
    FAMILY_ORDER,
    normalize_dataset,
    raw_csv_path,
    ridge_path,
    scaler_path,
)
from hstack1024_pipeline.extractors import INFERENCE_DEVICE, extract_all
from hstack1024_pipeline.io_contract import (
    FeatureSplit,
    file_hash,
    hstack_families,
    write_json,
)
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
OOD_FEATURE_SPACES = ("hstack1024", "selectkbest")
ACTIVITY_PIC50_THRESHOLD = 6.0
POSITIVE_LABEL = "Positive"
NEGATIVE_LABEL = "Negative"


def _validated_matrix(
    name: str,
    values: np.ndarray,
    *,
    expected_dim: Optional[int] = None,
) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional matrix")
    if matrix.shape[1] < 1:
        raise ValueError(f"{name} must contain at least one feature")
    if expected_dim is not None and matrix.shape[1] != expected_dim:
        raise ValueError(
            f"{name} must contain {expected_dim} features, got {matrix.shape[1]}"
        )
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} contains a non-finite value")
    return matrix


def knn_applicability_domain(
    X_train: np.ndarray,
    X_test: np.ndarray,
    k: int = 6,
) -> tuple[np.ndarray, float]:
    """Return the professor-defined IND mask and distance threshold."""
    train = _validated_matrix("X_train", X_train)
    test = _validated_matrix("X_test", X_test)
    if train.shape[1] != test.shape[1]:
        raise ValueError(
            "X_train and X_test feature dimensions differ: "
            f"{train.shape[1]} != {test.shape[1]}"
        )
    if not isinstance(k, (int, np.integer)) or int(k) < 1:
        raise ValueError("k must be a positive integer")
    k = int(k)
    if k > len(train):
        raise ValueError(f"k={k} exceeds the {len(train)} training compounds")

    nbrs = NearestNeighbors(n_neighbors=k)
    nbrs.fit(train)
    train_distances, _ = nbrs.kneighbors(train)
    train_mean_dist = train_distances.mean(axis=1)
    threshold = float(train_mean_dist.mean() + 0.5 * train_mean_dist.std())
    test_distances, _ = nbrs.kneighbors(test)
    test_mean_dist = test_distances.mean(axis=1)
    inside_domain = test_mean_dist <= threshold
    return np.asarray(inside_domain, dtype=bool), threshold


def professor_activity_prediction(
    X_train: np.ndarray,
    y_train: Sequence[float],
    X_test: np.ndarray,
    *,
    activity_threshold: float = ACTIVITY_PIC50_THRESHOLD,
) -> tuple[np.ndarray, np.ndarray, LinearDiscriminantAnalysis]:
    """Fit the professor-style LDA layer and return label/P(Positive)."""
    train = _validated_matrix("X_train", X_train)
    test = _validated_matrix("X_test", X_test)
    if train.shape[1] != test.shape[1]:
        raise ValueError("Activity train/test feature dimensions differ")
    targets = np.asarray(y_train, dtype=np.float64).reshape(-1)
    if len(targets) != len(train) or not np.isfinite(targets).all():
        raise ValueError("Training targets do not align with activity features")

    # Match the professor's label coding: class 0 = Positive, class 1 = Negative.
    y_activity = np.where(targets >= float(activity_threshold), 0, 1).astype(int)
    if set(np.unique(y_activity)) != {0, 1}:
        raise ValueError("Activity threshold must produce both Positive and Negative rows")
    finalclf = LinearDiscriminantAnalysis(tol=0.00001)
    finalclf.fit(train, y_activity)
    p_label = np.asarray(finalclf.predict(test), dtype=int)
    positive_column = int(np.flatnonzero(finalclf.classes_ == 0)[0])
    p_prob = np.asarray(finalclf.predict_proba(test)[:, positive_column], dtype=float)
    predicted = np.where(p_label == 0, POSITIVE_LABEL, NEGATIVE_LABEL)
    return predicted.astype(str), p_prob, finalclf


def analyze_query_ood(
    X_train: np.ndarray,
    X_test: np.ndarray,
    query_ids: Sequence[str],
    data: Sequence[str],
    *,
    k_values: Sequence[int] = OOD_K_VALUES,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create per-compound IND/OOD labels and per-k coverage summaries."""
    train = _validated_matrix("X_train", X_train)
    test = _validated_matrix("X_test", X_test)
    if train.shape[1] != test.shape[1]:
        raise ValueError("OOD train/test feature dimensions differ")
    ids = np.asarray(query_ids).astype(str)
    data = np.asarray(data).astype(str)
    if not (len(test) == len(ids) == len(data)):
        raise ValueError("Query features, IDs, and SMILES row counts differ")
    requested_k = tuple(int(k) for k in k_values)
    if requested_k != OOD_K_VALUES:
        raise ValueError("The controlled OOD protocol requires every k from 3 through 25")

    result = pd.DataFrame(
        {"row_index": np.arange(len(test), dtype=int), "ID": ids, "Smiles": data}
    )
    summaries = []
    for k in requested_k:
        inside_domain, threshold = knn_applicability_domain(train, test, k=k)
        result[f"ADk{k}"] = np.where(inside_domain, "IND", "OOD")
        summaries.append(
            {
                "k": k,
                "Threshold": threshold,
                "n_Total": int(len(test)),
                "n_IND": int(inside_domain.sum()),
                "n_OOD": int((~inside_domain).sum()),
                "IND_Coverage": float(inside_domain.mean()),
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
    """Create ``smiles_tr`` from Train+Validation (80%); never use Test."""
    smiles, targets = preprocess_raw(Path(raw_csv))
    bins = pd.qcut(targets, q=10, labels=False, duplicates="drop")
    train_smiles, temp_smiles, train_y, temp_y, _, temp_bins = train_test_split(
        smiles,
        targets,
        bins,
        test_size=0.40,
        random_state=0,
        stratify=bins,
    )
    val_smiles, _, val_y, _ = train_test_split(
        temp_smiles,
        temp_y,
        test_size=0.50,
        random_state=0,
        stratify=temp_bins,
    )
    smiles_tr = np.concatenate([train_smiles, val_smiles]).astype(str)
    y_tr = np.concatenate([train_y, val_y]).astype(float)
    return FeatureSplit(
        smiles_tr,
        np.empty((len(smiles_tr), 0), dtype=np.float32),
        y_tr,
    )


def _extract_domain_and_query(
    dataset: str,
    training: FeatureSplit,
    query: pd.DataFrame,
    *,
    smiles_column: str,
    base_root: Optional[Path],
) -> tuple[FeatureSplit, FeatureSplit]:
    smiles_tr = training.smiles
    data = query[smiles_column].astype(str).to_numpy()
    combined_smiles = np.concatenate([smiles_tr, data])
    combined_targets = np.concatenate([training.y, np.zeros(len(data), dtype=float)])
    extracted = extract_all(dataset, combined_smiles, combined_targets, base_root)
    n_train = len(smiles_tr)
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


def _professor_result(
    ood_result: pd.DataFrame,
    predicted: Sequence[str],
    probability: Sequence[float],
) -> pd.DataFrame:
    """Return the exact IND_Result layout, extended to ADk3..ADk25."""
    result = pd.DataFrame(
        {
            "Smiles": ood_result["Smiles"].astype(str),
            "Predicted": np.asarray(predicted).astype(str),
            "Probability": np.asarray(probability, dtype=float),
        }
    )
    for k in OOD_K_VALUES:
        result[f"ADk{k}"] = ood_result[f"ADk{k}"].to_numpy()
    return result


def _write_variant_outputs(
    *,
    dataset: str,
    variant_name: str,
    output_dir: Path,
    query: pd.DataFrame,
    ood_result: pd.DataFrame,
    summary: pd.DataFrame,
    predicted: np.ndarray,
    probability: np.ndarray,
    predicted_pic50: np.ndarray,
    regression_model_name: str,
    feature_space: str,
    feature_dimension: int,
    activity_classifier: LinearDiscriminantAnalysis,
    activity_threshold: float,
    query_csv: Path,
    raw_csv: Path,
    id_column: str,
    smiles_column: str,
    regression_artifact: Path,
    normalizer_artifact: Path,
    selected_feature_indices: Optional[Sequence[int]] = None,
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    ind_result = _professor_result(ood_result, predicted, probability)
    # Preserve the professor's unnamed leading CSV index column.
    ind_result.to_csv(output_dir / "IND_Result.csv", index=True)

    detailed = ood_result.copy()
    detailed.insert(3, "Predicted", predicted)
    detailed.insert(4, "Probability", probability)
    detailed.insert(5, "Predicted_pIC50", predicted_pic50)
    detailed.insert(0, "OOD_Feature_Space", feature_space)
    detailed.insert(0, "Regression_Model", regression_model_name)
    detailed.insert(0, "Variant", variant_name)
    detailed.insert(0, "Dataset", dataset)
    detailed.to_csv(output_dir / "production_predictions_ood.csv", index=False)

    summary.insert(0, "Feature_Dimension", int(feature_dimension))
    summary.insert(0, "OOD_Feature_Space", feature_space)
    summary.insert(0, "Variant", variant_name)
    summary.insert(0, "Dataset", dataset)
    summary.to_csv(output_dir / "ood_summary_k3_k25.csv", index=False)

    classifier_path = output_dir / "activity_classifier_lda.joblib"
    joblib.dump(activity_classifier, classifier_path, compress=3)
    manifest = {
        "dataset": dataset,
        "variant": variant_name,
        "mode": "production regression + professor-style activity reporting + kNN OOD",
        "query_csv": str(query_csv),
        "query_rows": int(len(query)),
        "query_columns": {"id": id_column, "smiles": smiles_column},
        "query_sha256": file_hash(Path(query_csv)),
        "domain_reference_csv": str(raw_csv),
        "domain_reference_sha256": file_hash(Path(raw_csv)),
        "domain_reference": "raw-derived Train+Validation 80%; Test excluded",
        "feature_space": feature_space,
        "feature_dimension": int(feature_dimension),
        "feature_order": list(FAMILY_ORDER),
        "selected_feature_indices": (
            None if selected_feature_indices is None else list(selected_feature_indices)
        ),
        "regression_model": regression_model_name,
        "regression_artifact": str(regression_artifact),
        "regression_artifact_sha256": file_hash(regression_artifact),
        "normalizer_artifact": str(normalizer_artifact),
        "normalizer_artifact_sha256": file_hash(normalizer_artifact),
        "activity_classifier": "LinearDiscriminantAnalysis(tol=0.00001)",
        "activity_label_rule": {
            "Positive": f"pIC50 >= {float(activity_threshold)}",
            "Negative": f"pIC50 < {float(activity_threshold)}",
        },
        "probability_definition": "LDA predict_proba for class 0 (Positive)",
        "activity_classifier_fitted_on": "Train+Validation 80%; Test excluded",
        "activity_classifier_artifact": classifier_path.name,
        "activity_classifier_sha256": file_hash(classifier_path),
        "ood_protocol_source": (
            "standard_pipeline/OOD_ajPle/"
            "ppar-2098c-ad-analyze-web-figure.ipynb"
        ),
        "ood_protocol": OOD_PROTOCOL,
        "k_values": list(OOD_K_VALUES),
        "labels": {"inside_domain": "IND", "outside_domain": "OOD"},
        "ground_truth_available": False,
        "historical_reference_comparison_performed": False,
        "test_split_used": False,
        "normalizer_refitted": False,
        "selectkbest_refitted": False,
        "regression_model_refitted": False,
        "activity_classifier_fitted": True,
        "knn_index_fitted_for_domain_analysis": True,
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "inference_device": INFERENCE_DEVICE,
        },
        "outputs": {
            "professor_style": "IND_Result.csv",
            "detailed_predictions_and_ad": "production_predictions_ood.csv",
            "per_k_summary": "ood_summary_k3_k25.csv",
        },
    }
    write_json(output_dir / "production_manifest.json", manifest)
    return summary


def _run_selectkbest_variant(
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
    activity_threshold: float,
) -> pd.DataFrame:
    frozen_bundle_path = bundle_path(
        dataset, method, fs_root=fs_root, repo_root=repo_root
    )
    bundle = load_frozen_bundle(frozen_bundle_path)
    train_scaled = apply_frozen_minmax(training_hstack.X, bundle["normalizer"])
    query_scaled = apply_frozen_minmax(query_hstack.X, bundle["normalizer"])
    X_train = apply_frozen_selector(train_scaled, bundle)
    X_test = apply_frozen_selector(query_scaled, bundle)
    predicted_pic50 = predict_frozen_linear(X_test, bundle["model"])
    predicted, probability, finalclf = professor_activity_prediction(
        X_train, training_hstack.y, X_test, activity_threshold=activity_threshold
    )
    ood_result, summary = analyze_query_ood(
        X_train,
        X_test,
        query[id_column].astype(str).to_numpy(),
        query[smiles_column].astype(str).to_numpy(),
    )
    variant_name = f"{dataset}_{method}"
    return _write_variant_outputs(
        dataset=dataset,
        variant_name=variant_name,
        output_dir=Path(output_root) / dataset / method,
        query=query,
        ood_result=ood_result,
        summary=summary,
        predicted=predicted,
        probability=probability,
        predicted_pic50=predicted_pic50,
        regression_model_name=str(bundle["model_name"]),
        feature_space="frozen-MinMax-scaled HStack1024 + frozen SelectKBest",
        feature_dimension=int(bundle["FS_k"]),
        activity_classifier=finalclf,
        activity_threshold=activity_threshold,
        query_csv=query_csv,
        raw_csv=raw_csv,
        id_column=id_column,
        smiles_column=smiles_column,
        regression_artifact=frozen_bundle_path,
        normalizer_artifact=frozen_bundle_path,
        selected_feature_indices=selected_indices(bundle).tolist(),
    )


def _run_hstack1024_variant(
    dataset: str,
    output_root: Path,
    training_hstack: FeatureSplit,
    query_hstack: FeatureSplit,
    query: pd.DataFrame,
    *,
    query_csv: Path,
    raw_csv: Path,
    id_column: str,
    smiles_column: str,
    base_root: Optional[Path],
    activity_threshold: float,
) -> pd.DataFrame:
    normalizer_path = scaler_path(dataset, base_root)
    regression_path = ridge_path(dataset, base_root)
    normalizer = joblib.load(normalizer_path)
    regression_model = joblib.load(regression_path)
    X_train = apply_frozen_minmax(training_hstack.X, normalizer)
    X_test = apply_frozen_minmax(query_hstack.X, normalizer)
    _validated_matrix("X_train", X_train, expected_dim=HSTACK_DIM)
    _validated_matrix("X_test", X_test, expected_dim=HSTACK_DIM)
    predicted_pic50 = predict_frozen_linear(X_test, regression_model)
    predicted, probability, finalclf = professor_activity_prediction(
        X_train, training_hstack.y, X_test, activity_threshold=activity_threshold
    )
    ood_result, summary = analyze_query_ood(
        X_train,
        X_test,
        query[id_column].astype(str).to_numpy(),
        query[smiles_column].astype(str).to_numpy(),
    )
    variant_name = f"{dataset}_hstack1024"
    return _write_variant_outputs(
        dataset=dataset,
        variant_name=variant_name,
        output_dir=Path(output_root) / dataset / "hstack1024",
        query=query,
        ood_result=ood_result,
        summary=summary,
        predicted=predicted,
        probability=probability,
        predicted_pic50=predicted_pic50,
        regression_model_name=regression_model.__class__.__name__,
        feature_space="frozen-MinMax-scaled HStack1024",
        feature_dimension=HSTACK_DIM,
        activity_classifier=finalclf,
        activity_threshold=activity_threshold,
        query_csv=query_csv,
        raw_csv=raw_csv,
        id_column=id_column,
        smiles_column=smiles_column,
        regression_artifact=regression_path,
        normalizer_artifact=normalizer_path,
    )


def run_production_ood_dataset(
    dataset: str,
    query_csv: Path,
    output_root: Path,
    *,
    feature_space: Optional[str] = None,
    method: Optional[str] = None,
    methods: Optional[Sequence[str]] = None,
    base_root: Optional[Path] = None,
    fs_root: Optional[Path] = None,
    repo_root: Optional[Path] = None,
    raw_csv: Optional[Path] = None,
    id_column: str = "ID",
    smiles_column: str = "Smiles",
    activity_threshold: float = ACTIVITY_PIC50_THRESHOLD,
) -> pd.DataFrame:
    """Run controlled endpoint/feature-space OOD variants.

    ``methods`` is retained for notebooks generated before the 12-variant API.
    Those notebooks only produced SelectKBest variants and may request one or
    both feature-selection methods in a single call. New callers should pass
    ``feature_space`` and, for SelectKBest, the singular ``method`` argument.
    """
    ds = normalize_dataset(dataset)
    if methods is not None:
        if method is not None:
            raise ValueError("Use either method or methods, not both")
        if feature_space not in (None, "selectkbest"):
            raise ValueError("Legacy methods is only valid for selectkbest OOD")
        selected_methods = tuple(normalize_method(item) for item in methods)
        if not selected_methods:
            raise ValueError("At least one feature-selection method is required")
        space = "selectkbest"
    else:
        if feature_space is None:
            raise ValueError("feature_space is required")
        space = str(feature_space).strip().lower()
        selected_methods = (
            (normalize_method(method),) if method is not None else tuple()
        )
    if space not in OOD_FEATURE_SPACES:
        raise ValueError(f"feature_space must be one of {OOD_FEATURE_SPACES}")
    if space == "selectkbest" and not selected_methods:
        raise ValueError("method is required for selectkbest OOD")
    if space == "hstack1024" and selected_methods:
        raise ValueError("method must be omitted for hstack1024 OOD")

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
    if space == "hstack1024":
        report = _run_hstack1024_variant(
            ds,
            output_root,
            training_hstack,
            query_hstack,
            query,
            query_csv=query_source,
            raw_csv=raw_source,
            id_column=id_column,
            smiles_column=smiles_column,
            base_root=base_root,
            activity_threshold=activity_threshold,
        )
    else:
        reports = [
            _run_selectkbest_variant(
                ds,
                selected_method,
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
                activity_threshold=activity_threshold,
            )
            for selected_method in selected_methods
        ]
        report = pd.concat(reports, ignore_index=True)
    selected_label = (
        selected_methods[0] if len(selected_methods) == 1 else "all"
    ) if space == "selectkbest" else "base"
    aggregate_dir = Path(output_root) / ds
    aggregate_dir.mkdir(parents=True, exist_ok=True)
    report.to_csv(
        aggregate_dir / f"ood_summary_{space}_{selected_label}.csv",
        index=False,
    )
    if methods is not None:
        report.to_csv(
            aggregate_dir / "ood_summary.csv",
            index=False,
        )
    return report


def run_production_ood_all(
    query_csv: Path,
    output_root: Path,
    *,
    datasets: Sequence[str],
    methods: Optional[Sequence[str]] = None,
    base_root: Optional[Path] = None,
    fs_root: Optional[Path] = None,
    repo_root: Optional[Path] = None,
    raw_csv_paths: Mapping[str, Path],
    id_column: str = "ID",
    smiles_column: str = "Smiles",
    activity_threshold: float = ACTIVITY_PIC50_THRESHOLD,
) -> pd.DataFrame:
    """Run all 12 variants, or legacy SelectKBest-only notebook requests."""
    reports = []
    for dataset in datasets:
        if methods is not None:
            reports.append(
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
                    activity_threshold=activity_threshold,
                )
            )
            continue
        reports.append(
            run_production_ood_dataset(
                dataset,
                query_csv,
                output_root,
                feature_space="hstack1024",
                base_root=base_root,
                fs_root=fs_root,
                repo_root=repo_root,
                raw_csv=raw_csv_paths[dataset],
                id_column=id_column,
                smiles_column=smiles_column,
                activity_threshold=activity_threshold,
            )
        )
        for method in FS_METHODS:
            reports.append(
                run_production_ood_dataset(
                    dataset,
                    query_csv,
                    output_root,
                    feature_space="selectkbest",
                    method=method,
                    base_root=base_root,
                    fs_root=fs_root,
                    repo_root=repo_root,
                    raw_csv=raw_csv_paths[dataset],
                    id_column=id_column,
                    smiles_column=smiles_column,
                    activity_threshold=activity_threshold,
                )
            )
    combined = pd.concat(reports, ignore_index=True)
    combined.to_csv(Path(output_root) / "ood_summary.csv", index=False)
    (Path(output_root) / "ood_summary.json").write_text(
        json.dumps(
            {
                "query_csv": str(query_csv),
                "ground_truth_available": False,
                "test_split_used": False,
                "activity_threshold": float(activity_threshold),
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
