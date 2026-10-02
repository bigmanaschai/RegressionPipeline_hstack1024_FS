"""Generate eight transparent Case-study OOD notebooks from AD Test outputs."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OOD_ROOT = ROOT / "kaggle_notebooks" / "OOD-Regression"
OUTPUT_DIR = OOD_ROOT / "01_run-08_run-OOD"
AD_BUILDER_PATH = ROOT / "AD_Test_Regression" / "build_kaggle_notebooks.py"


def _load_ad_builder():
    spec = importlib.util.spec_from_file_location("ad_notebook_builder", AD_BUILDER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load AD notebook builder: {AD_BUILDER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


AD_BUILDER = _load_ad_builder()

OOD_TARGETS = tuple(
    (
        f"{sequence}_run_{dataset}_{method}_OOD.ipynb",
        dataset,
        method,
        selected_k,
        model_class,
        reference_r2,
    )
    for sequence, dataset, method, selected_k, model_class, reference_r2
    in AD_BUILDER.VARIANTS
)


def _lines(text: str) -> list[str]:
    return text.strip("\n").splitlines(keepends=True)


def code_cell(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": _lines(text),
    }


def markdown_cell(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": _lines(text)}


README_CONTENT = """# Case-study OOD Regression notebooks

This directory contains exactly eight notebooks: AR, ER, GR, and PR crossed
with the frozen `mutual_info` and `pearson` variants.

The notebooks do not clone GitHub and do not import project `.py` files. They
consume the matching Kaggle output from `AD_Test_Regression` as an input:

- `01_raw_splits.npz` supplies Train+Validation pIC50 for the activity layer;
- `05_selected_features_k<K>.npz` verifies selected-feature identity;
- `07_ad_reference_train_plus_validation.npy` is the exact kNN domain reference;
- the two AD summary CSVs supply the accepted 0.5 and 2.0 distance thresholds;
- `run_metadata.json` verifies dataset, method, feature order, and AD controls.

Only `cleaned_Casestudy.csv` is freshly featurized. Every input row is retained.
Invalid SMILES are audited and marked `INVALID`; duplicate valid SMILES are
featurized once and mapped back to every original row. The current Case-study
file has columns `ID,Smiles` and no observed pIC50, so R2/RMSE/MAE/ME/Pearson/
Spearman are not calculated.

Each notebook reports k=3...25 separately for threshold multipliers 0.5 and
2.0, writes a compatibility `IND_Result.csv` for multiplier 0.5, and saves
query features, predictions, distances, IND/OOD labels, coverage summaries,
metadata, plot, and SHA-256 manifest.
"""

PARENT_README_CONTENT = """# OOD Regression

The active controlled notebooks are in `01_run-08_run-OOD/`. Only the eight
AR/ER/GR/PR × mutual_info/pearson Case-study OOD notebooks are maintained.
They consume the matching `AD_Test_Regression` Kaggle notebook output as input.
"""


def configuration_cell(
    dataset: str,
    method: str,
    selected_k: int,
    model_class: str,
    reference_r2: float,
) -> str:
    bundle_name = (
        f"{dataset}_{method}_bundle.json"
        if (dataset, method) == ("AR", "pearson")
        else f"{dataset}_{method}_best_model.pkl"
    )
    return f'''# STEP 0 — Configuration. Blank paths are auto-discovered under /kaggle/input.
RUN_DATASET = {dataset!r}
RUN_METHOD = {method!r}
EXPECTED_SELECTED_K = {selected_k}
EXPECTED_MODEL_CLASS = {model_class!r}
EXPECTED_REFERENCE_TEST_R2 = {reference_r2!r}
FROZEN_BUNDLE_FILENAME = {bundle_name!r}

AD_OUTPUT_DIR = ""             # matching AD_Test_Regression output directory
CASE_STUDY_CSV_PATH = ""       # cleaned_Casestudy.csv
SMILES_CHECKPOINT_PATH = ""
SELFIES_CHECKPOINT_PATH = ""
GRAPH_CHECKPOINT_PATH = ""
FINGERPRINT_CHECKPOINT_PATH = ""
FINGERPRINT_TRANSFORMER_PATH = ""

FROZEN_BUNDLE_PATH = ""
FROZEN_MODEL_PATH = ""
FROZEN_SCALER_PATH = ""
FROZEN_SELECTOR_PATH = ""
USE_EMBEDDED_FROZEN_ARTIFACTS = True

INPUT_SEARCH_ROOTS = ["/kaggle/input"]
OUTPUT_ROOT = f"/kaggle/working/OOD_Regression_{{RUN_DATASET}}_{{RUN_METHOD}}"
CASE_ID_COLUMN = "ID"
CASE_SMILES_COLUMN = "Smiles"
ACTIVITY_PIC50_THRESHOLD = 6.0
AD_THRESHOLD_MULTIPLIERS = (0.5, 2.0)
AD_NEIGHBOR_K_VALUES = tuple(range(3, 26))
GRAPH_QUERY_CHUNK_SIZE = 2000

AUTO_INSTALL_MISSING_DEPENDENCIES = True
SKFP_INSTALL_SPEC = "git+https://github.com/plenoi/scikit-finger-plenoi.git@master"

assert AD_THRESHOLD_MULTIPLIERS == (0.5, 2.0)
assert AD_NEIGHBOR_K_VALUES == tuple(range(3, 26))
print("OOD variant:", f"{{RUN_DATASET}} × {{RUN_METHOD}}")
print("Expected selected features:", EXPECTED_SELECTED_K)
print("Case-study output:", OUTPUT_ROOT)
'''


PATH_RESOLUTION = r'''# STEP 1 — Resolve Case study, deep checkpoints, matching AD output, and frozen final artifacts.
import hashlib
import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from IPython.display import display

OUTPUT_DIR = Path(OUTPUT_ROOT)
ARTIFACT_DIR = OUTPUT_DIR / "intermediate_artifacts"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

def configured_path(value):
    return Path(value).expanduser().resolve() if str(value).strip() else None

def discover_exact_file(label, configured, filenames, *, required=True):
    explicit = configured_path(configured)
    if explicit is not None:
        if not explicit.is_file():
            raise FileNotFoundError(f"{label} path does not exist: {explicit}")
        return explicit
    matches = []
    for root_value in INPUT_SEARCH_ROOTS:
        root = Path(root_value).expanduser()
        if root.is_dir():
            for filename in filenames:
                matches.extend(path.resolve() for path in root.rglob(filename) if path.is_file())
    matches = sorted(set(matches))
    if len(matches) == 1:
        return matches[0]
    if not matches and not required:
        return None
    if not matches:
        raise FileNotFoundError(
            f"Could not find {label}; set its path in STEP 0. Expected: {list(filenames)}"
        )
    preview = "\n".join(f"  - {path}" for path in matches[:20])
    raise RuntimeError(f"Multiple candidates for {label}; set the exact path in STEP 0:\n{preview}")

def discover_ad_output_dir(configured):
    explicit = configured_path(configured)
    if explicit is not None:
        metadata_paths = [explicit / "run_metadata.json"] if explicit.is_dir() else []
    else:
        metadata_paths = []
        for root_value in INPUT_SEARCH_ROOTS:
            root = Path(root_value).expanduser()
            if root.is_dir():
                metadata_paths.extend(root.rglob("run_metadata.json"))
    matches = []
    for metadata_path in metadata_paths:
        try:
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if (
            payload.get("dataset") == RUN_DATASET
            and payload.get("method") == RUN_METHOD
            and payload.get("ad_reference") == "Train + Validation selected features"
        ):
            matches.append(metadata_path.parent.resolve())
    matches = sorted(set(matches))
    if len(matches) != 1:
        preview = "\n".join(f"  - {path}" for path in matches[:20]) or "  (none)"
        raise RuntimeError(
            f"Expected exactly one matching AD output for {RUN_DATASET}/{RUN_METHOD}; "
            f"found {len(matches)}. Set AD_OUTPUT_DIR in STEP 0.\n{preview}"
        )
    return matches[0]

AD_INPUT_DIR = discover_ad_output_dir(AD_OUTPUT_DIR)
INPUT_PATHS = {
    "case_study_csv": discover_exact_file(
        "Case-study CSV", CASE_STUDY_CSV_PATH, ["cleaned_Casestudy.csv"]
    ),
    "smiles_checkpoint": discover_exact_file(
        "SMILES checkpoint", SMILES_CHECKPOINT_PATH, [f"model_smiles_{RUN_DATASET}.pt"]
    ),
    "selfies_checkpoint": discover_exact_file(
        "SELFIES checkpoint", SELFIES_CHECKPOINT_PATH, [f"model_selfies_{RUN_DATASET}.pt"]
    ),
    "graph_checkpoint": discover_exact_file(
        "Graph checkpoint", GRAPH_CHECKPOINT_PATH, [f"model_graph_{RUN_DATASET}.pt"]
    ),
    "fingerprint_checkpoint": discover_exact_file(
        "Fingerprint checkpoint", FINGERPRINT_CHECKPOINT_PATH,
        [f"model_fingerprint_{RUN_DATASET}.pt"],
    ),
    "fingerprint_transformer": discover_exact_file(
        "Fingerprint transformer", FINGERPRINT_TRANSFORMER_PATH,
        [f"ecfp_transformer_{RUN_DATASET}.pkl"],
    ),
}

separate_frozen_paths = all(
    str(value).strip()
    for value in (FROZEN_MODEL_PATH, FROZEN_SCALER_PATH, FROZEN_SELECTOR_PATH)
)
if separate_frozen_paths:
    INPUT_PATHS["frozen_model"] = discover_exact_file(
        "frozen model", FROZEN_MODEL_PATH, [Path(FROZEN_MODEL_PATH).name]
    )
    INPUT_PATHS["frozen_scaler"] = discover_exact_file(
        "frozen scaler", FROZEN_SCALER_PATH, [Path(FROZEN_SCALER_PATH).name]
    )
    INPUT_PATHS["frozen_selector"] = discover_exact_file(
        "frozen selector", FROZEN_SELECTOR_PATH, [Path(FROZEN_SELECTOR_PATH).name]
    )
    FROZEN_INPUT_SOURCE = "three explicit paths"
else:
    if any(str(value).strip() for value in (FROZEN_MODEL_PATH, FROZEN_SCALER_PATH, FROZEN_SELECTOR_PATH)):
        raise ValueError("Provide all three frozen component paths, or leave all three blank")
    discovered_bundle = discover_exact_file(
        "frozen bundle", FROZEN_BUNDLE_PATH, [FROZEN_BUNDLE_FILENAME], required=False
    )
    if discovered_bundle is not None:
        INPUT_PATHS["frozen_bundle"] = discovered_bundle
        FROZEN_INPUT_SOURCE = "external Kaggle input"
    elif USE_EMBEDDED_FROZEN_ARTIFACTS:
        INPUT_PATHS["frozen_bundle"] = EMBEDDED_FROZEN_PATHS[FROZEN_BUNDLE_FILENAME]
        FROZEN_INPUT_SOURCE = "embedded checksum-verified fallback"
    else:
        raise FileNotFoundError(
            f"Frozen bundle {FROZEN_BUNDLE_FILENAME!r} was not found and fallback is disabled"
        )

print("Matching AD output:", AD_INPUT_DIR)
print("Frozen final-artifact source:", FROZEN_INPUT_SOURCE)
display(pd.DataFrame([{"Input": key, "Resolved path": str(value)} for key, value in INPUT_PATHS.items()]))
'''


LOAD_AD_OUTPUT = r'''# STEP 3 — Load and validate the exact domain reference and thresholds from AD Test output.
ad_metadata_path = AD_INPUT_DIR / "run_metadata.json"
ad_manifest_path = AD_INPUT_DIR / "artifact_manifest.csv"
ad_artifact_dir = AD_INPUT_DIR / "intermediate_artifacts"
ad_raw_splits_path = ad_artifact_dir / "01_raw_splits.npz"
ad_selected_path = ad_artifact_dir / f"05_selected_features_k{EXPECTED_SELECTED_K}.npz"
ad_reference_path = ad_artifact_dir / "07_ad_reference_train_plus_validation.npy"

required_ad_files = [
    ad_metadata_path, ad_manifest_path, ad_raw_splits_path,
    ad_selected_path, ad_reference_path,
]
missing_ad_files = [path for path in required_ad_files if not path.is_file()]
if missing_ad_files:
    raise FileNotFoundError(f"AD output is incomplete: {missing_ad_files}")

ad_metadata = json.loads(ad_metadata_path.read_text(encoding="utf-8"))
if ad_metadata.get("dataset") != RUN_DATASET or ad_metadata.get("method") != RUN_METHOD:
    raise AssertionError("AD output dataset/method does not match this OOD notebook")
if int(ad_metadata.get("selected_feature_count")) != EXPECTED_SELECTED_K:
    raise AssertionError("AD output selected-feature count changed")
if tuple(ad_metadata.get("ad_k_values", [])) != AD_NEIGHBOR_K_VALUES:
    raise AssertionError("AD output k values changed")
if tuple(float(value) for value in ad_metadata.get("threshold_multipliers", [])) != AD_THRESHOLD_MULTIPLIERS:
    raise AssertionError("AD output threshold multipliers changed")

raw_splits = np.load(ad_raw_splits_path, allow_pickle=False)
ad_selected = np.load(ad_selected_path, allow_pickle=False)
X_reference = np.load(ad_reference_path, allow_pickle=False).astype(np.float32)
ad_selected_indices = np.asarray(ad_selected["selected_indices"], dtype=int)
y_reference = np.concatenate([raw_splits["train_y"], raw_splits["val_y"]]).astype(float)
expected_reference = np.vstack([ad_selected["train"], ad_selected["val"]]).astype(np.float32)

if not np.array_equal(ad_selected_indices, selected_indices):
    raise AssertionError("Frozen selector indices differ from AD Test output")
if X_reference.shape != expected_reference.shape or not np.array_equal(X_reference, expected_reference):
    raise AssertionError("AD reference differs from saved Train+Validation selected features")
if len(y_reference) != len(X_reference):
    raise AssertionError("AD reference rows and Train+Validation targets do not align")

ad_summaries = {}
for multiplier in AD_THRESHOLD_MULTIPLIERS:
    token = "0p5" if multiplier == 0.5 else "2p0"
    path = AD_INPUT_DIR / (
        f"AD_Test_Regression_{RUN_DATASET}_{RUN_METHOD}_threshold_{token}.csv"
    )
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    expected_labels = ["No AD"] + [f"k={k}" for k in AD_NEIGHBOR_K_VALUES]
    if frame["K"].tolist() != expected_labels:
        raise AssertionError(f"Unexpected AD summary rows: {path}")
    ad_summaries[multiplier] = frame

print("AD reference shape:", X_reference.shape)
print("Reference target shape:", y_reference.shape)
print("Selected indices verified:", ad_selected_indices.tolist())
display(pd.DataFrame({
    "Multiplier": list(ad_summaries),
    "Summary rows": [len(frame) for frame in ad_summaries.values()],
}))
'''


CASE_STUDY_PREP = r'''# STEP 4 — Validate every Case-study row, preserve duplicates, and isolate unique valid SMILES.
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")
case_raw = pd.read_csv(INPUT_PATHS["case_study_csv"])
required_case_columns = {CASE_ID_COLUMN, CASE_SMILES_COLUMN}
missing_case_columns = required_case_columns.difference(case_raw.columns)
if missing_case_columns:
    raise ValueError(f"Case-study CSV is missing columns: {sorted(missing_case_columns)}")

case_frame = case_raw.copy()
case_frame.insert(0, "Case_Row", np.arange(len(case_frame), dtype=int))
case_frame["Original_ID"] = case_frame[CASE_ID_COLUMN]
case_frame["Original_SMILES"] = case_frame[CASE_SMILES_COLUMN]
case_frame["Normalized_SMILES"] = case_frame[CASE_SMILES_COLUMN].fillna("").astype(str).str.strip()

def validate_case_smiles(value):
    if not value:
        return "MISSING_SMILES"
    return "VALID" if Chem.MolFromSmiles(value) is not None else "INVALID_SMILES"

case_frame["Input_Status"] = case_frame["Normalized_SMILES"].map(validate_case_smiles)
valid_mask = case_frame["Input_Status"].eq("VALID").to_numpy()
if not valid_mask.any():
    raise ValueError("Case study has no valid SMILES")

valid_smiles = case_frame.loc[valid_mask, "Normalized_SMILES"].to_numpy().astype(str)
unique_codes, unique_smiles = pd.factorize(valid_smiles, sort=False)
unique_smiles = np.asarray(unique_smiles).astype(str)
case_frame["Duplicate_Valid_SMILES"] = False
case_frame.loc[valid_mask, "Duplicate_Valid_SMILES"] = pd.Series(valid_smiles).duplicated(keep=False).to_numpy()

case_audit_path = ARTIFACT_DIR / "01_case_study_input_audit.csv"
unique_path = ARTIFACT_DIR / "02_unique_valid_case_smiles.csv"
case_frame.to_csv(case_audit_path, index=False)
pd.DataFrame({"Unique_Query_Row": np.arange(len(unique_smiles)), "Smiles": unique_smiles}).to_csv(
    unique_path, index=False
)

print("Case-study rows:", len(case_frame))
print("Valid rows:", int(valid_mask.sum()))
print("Invalid/missing rows:", int((~valid_mask).sum()))
print("Unique valid SMILES to featurize:", len(unique_smiles))
print("Duplicate valid rows:", int(case_frame["Duplicate_Valid_SMILES"].sum()))
print("Saved:", case_audit_path)
display(case_frame.head())
'''


FEATURE_PREP = r'''# STEP 6 — Prepare Case-study feature containers and reloadable family outputs.
query_family_features = {}

def save_query_family(family, matrix):
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.shape != (len(unique_smiles), 256):
        raise AssertionError(f"{family}: unexpected query feature shape {matrix.shape}")
    query_family_features[family] = matrix
    output_path = ARTIFACT_DIR / f"03_case_{family}_features_256.npy"
    np.save(output_path, matrix)
    print(f"{family}: shape={matrix.shape}, first five={matrix[0, :5]}")
    print("Saved:", output_path)
'''


def extract_family_cell(family: str) -> str:
    function = {
        "smiles": "extract_smiles",
        "selfies": "extract_selfies",
        "graph": "extract_graph",
        "fingerprint": "extract_fingerprint",
    }[family]
    if family == "graph":
        execution = '''graph_parts = []
for left in range(0, len(unique_smiles), GRAPH_QUERY_CHUNK_SIZE):
    right = min(left + GRAPH_QUERY_CHUNK_SIZE, len(unique_smiles))
    print(f"Graph chunk: rows {left}:{right} of {len(unique_smiles)}")
    graph_parts.append(extract_graph(RUN_DATASET, unique_smiles[left:right]))
graph_query = np.vstack(graph_parts).astype(np.float32)'''
    else:
        execution = f"{family}_query = {function}(RUN_DATASET, unique_smiles)"
    return f'''# STEP 7 — Extract fresh {family.upper()} Case-study features and save them.
{execution}
save_query_family({family!r}, {family}_query)
'''


QUERY_TRANSFORM = r'''# STEP 8 — HStack1024 → frozen MinMaxScaler → frozen selector for Case study.
FAMILY_ORDER = ("smiles", "selfies", "graph", "fingerprint")
X_query_hstack = np.hstack([query_family_features[name] for name in FAMILY_ORDER]).astype(np.float32)
if X_query_hstack.shape != (len(unique_smiles), 1024):
    raise AssertionError(f"Unexpected Case-study HStack shape: {X_query_hstack.shape}")

X_query_scaled = (
    X_query_hstack * np.asarray(frozen_scaler.scale_) + np.asarray(frozen_scaler.min_)
).astype(np.float32)
X_query_selected = X_query_scaled[:, selected_indices].astype(np.float32)
if X_query_selected.shape[1] != X_reference.shape[1]:
    raise AssertionError("Case-study and AD-reference feature dimensions differ")

hstack_path = ARTIFACT_DIR / "04_case_hstack1024.npy"
scaled_path = ARTIFACT_DIR / "05_case_minmax_scaled_hstack1024.npy"
selected_path = ARTIFACT_DIR / f"06_case_selected_features_k{EXPECTED_SELECTED_K}.npy"
np.save(hstack_path, X_query_hstack)
np.save(scaled_path, X_query_scaled)
np.save(selected_path, X_query_selected)

print("Case HStack1024:", X_query_hstack.shape)
print("Case scaled HStack1024:", X_query_scaled.shape)
print("Case selected features:", X_query_selected.shape)
print("AD reference:", X_reference.shape)
print("Saved:", selected_path)
'''


PREDICTION_ACTIVITY = r'''# STEP 9 — Frozen pIC50 prediction and Train+Validation-only LDA activity reporting.
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

def predict_frozen_linear(X, model):
    features = np.asarray(X, dtype=np.float64)
    coefficients = np.asarray(model.coef_, dtype=np.float64).reshape(1, -1)
    intercept = np.asarray(model.intercept_, dtype=np.float64)
    return (np.sum(features * coefficients, axis=1) + intercept).reshape(-1)

unique_predicted_pic50 = predict_frozen_linear(X_query_selected, frozen_model)
activity_y = np.where(y_reference >= ACTIVITY_PIC50_THRESHOLD, 0, 1)
if set(np.unique(activity_y)) != {0, 1}:
    raise ValueError("Train+Validation activity labels do not contain both classes")
activity_classifier = LinearDiscriminantAnalysis(tol=0.00001)
activity_classifier.fit(X_reference, activity_y)
unique_activity_class = activity_classifier.predict(X_query_selected)
positive_column = int(np.flatnonzero(activity_classifier.classes_ == 0)[0])
unique_probability = activity_classifier.predict_proba(X_query_selected)[:, positive_column]
unique_activity_label = np.where(unique_activity_class == 0, "Positive", "Negative")

joblib.dump(activity_classifier, OUTPUT_DIR / "activity_classifier_lda.joblib")
unique_predictions = pd.DataFrame({
    "Unique_Query_Row": np.arange(len(unique_smiles), dtype=int),
    "Smiles": unique_smiles,
    "Predicted_pIC50": unique_predicted_pic50,
    "Predicted": unique_activity_label,
    "Probability": unique_probability,
})
unique_prediction_path = ARTIFACT_DIR / "07_unique_case_predictions.csv"
unique_predictions.to_csv(unique_prediction_path, index=False)

full_predictions = case_frame.copy()
full_predictions["Predicted_pIC50"] = np.nan
full_predictions["Predicted"] = "INVALID"
full_predictions["Probability"] = np.nan
valid_positions = np.flatnonzero(valid_mask)
full_predictions.loc[valid_positions, "Predicted_pIC50"] = unique_predicted_pic50[unique_codes]
full_predictions.loc[valid_positions, "Predicted"] = unique_activity_label[unique_codes]
full_predictions.loc[valid_positions, "Probability"] = unique_probability[unique_codes]

prediction_path = ARTIFACT_DIR / "08_case_predictions_all_rows.csv"
full_predictions.to_csv(prediction_path, index=False)
print("LDA training rows:", len(X_reference))
print("Positive training rows:", int((activity_y == 0).sum()))
print("Negative training rows:", int((activity_y == 1).sum()))
print("Saved:", prediction_path)
display(full_predictions.head())
'''


OOD_EXECUTION = r'''# STEP 10 — Apply AD-Test thresholds to Case study for k=3...25 and both multipliers.
from sklearn.neighbors import NearestNeighbors

def threshold_token(multiplier):
    return "0p5" if float(multiplier) == 0.5 else "2p0"

summaries = []
detail_paths = {}
result_paths = {}
for multiplier in AD_THRESHOLD_MULTIPLIERS:
    stored_summary = ad_summaries[multiplier].set_index("K")
    detailed = full_predictions.copy()
    professor = pd.DataFrame({
        "Smiles": full_predictions["Original_SMILES"],
        "Predicted": full_predictions["Predicted"],
        "Probability": full_predictions["Probability"],
    })
    for ad_k in AD_NEIGHBOR_K_VALUES:
        knn = NearestNeighbors(n_neighbors=ad_k, metric="euclidean", algorithm="auto")
        knn.fit(X_reference)
        query_distances, _ = knn.kneighbors(X_query_selected)
        unique_mean_distance = query_distances.mean(axis=1)
        threshold = float(stored_summary.loc[f"k={ad_k}", "AD_Distance_Threshold"])
        unique_ind = unique_mean_distance <= threshold

        full_distance = np.full(len(case_frame), np.nan, dtype=float)
        full_label = np.full(len(case_frame), "INVALID", dtype=object)
        full_distance[valid_positions] = unique_mean_distance[unique_codes]
        full_label[valid_positions] = np.where(unique_ind[unique_codes], "IND", "OOD")
        detailed[f"k{ad_k}_Mean_Distance"] = full_distance
        detailed[f"k{ad_k}_Threshold"] = threshold
        detailed[f"ADk{ad_k}"] = full_label
        professor[f"ADk{ad_k}"] = full_label

        ind_count = int(np.sum(full_label == "IND"))
        ood_count = int(np.sum(full_label == "OOD"))
        invalid_count = int(np.sum(full_label == "INVALID"))
        summaries.append({
            "Dataset": RUN_DATASET,
            "FS_Method": RUN_METHOD,
            "Threshold_Multiplier": float(multiplier),
            "k": int(ad_k),
            "Distance_Threshold": threshold,
            "N_Case_Total": int(len(case_frame)),
            "N_Valid": int(valid_mask.sum()),
            "N_Invalid": invalid_count,
            "IND_Count": ind_count,
            "OOD_Count": ood_count,
            "IND_Coverage_Valid": float(ind_count / valid_mask.sum()),
            "IND_Coverage_All": float(ind_count / len(case_frame)),
        })

    token = threshold_token(multiplier)
    detail_path = OUTPUT_DIR / f"case_study_predictions_ood_threshold_{token}.csv"
    result_path = OUTPUT_DIR / f"IND_Result_threshold_{token}.csv"
    detailed.to_csv(detail_path, index=False)
    professor.to_csv(result_path, index=True)
    detail_paths[multiplier] = detail_path
    result_paths[multiplier] = result_path
    print(f"Multiplier={multiplier}: saved {detail_path.name} and {result_path.name}")

summary = pd.DataFrame(summaries)
summary_path = OUTPUT_DIR / "ood_summary_k3_k25.csv"
summary.to_csv(summary_path, index=False)

# Compatibility names use the historical multiplier 0.5.
full_05 = pd.read_csv(detail_paths[0.5])
ind_05 = pd.read_csv(result_paths[0.5], index_col=0)
full_05.to_csv(OUTPUT_DIR / "production_predictions_ood.csv", index=False)
ind_05.to_csv(OUTPUT_DIR / "IND_Result.csv", index=True)

print("Combined summary:", summary_path)
display(summary.head())
'''


PLOT_AND_MANIFEST = r'''# STEP 11 — Plot coverage and write provenance plus SHA-256 manifest.
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
for multiplier, group in summary.groupby("Threshold_Multiplier", sort=True):
    ax.plot(group["k"], group["IND_Coverage_Valid"], marker="o", label=f"multiplier={multiplier}")
ax.set(
    title=f"{RUN_DATASET} {RUN_METHOD}: Case-study IND coverage",
    xlabel="k",
    ylabel="IND coverage among valid Case-study rows",
)
ax.set_xticks(range(3, 26, 2))
ax.set_ylim(0.0, 1.05)
ax.grid(alpha=0.25)
ax.legend()
figure_path = OUTPUT_DIR / "ood_coverage_diagnostics.png"
fig.savefig(figure_path, dpi=180, bbox_inches="tight")
plt.show()

def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

metadata = {
    "dataset": RUN_DATASET,
    "method": RUN_METHOD,
    "case_study_csv": str(INPUT_PATHS["case_study_csv"]),
    "source_ad_output": str(AD_INPUT_DIR),
    "source_ad_metadata_sha256": sha256_file(ad_metadata_path),
    "ad_reference_sha256": sha256_file(ad_reference_path),
    "case_rows": int(len(case_frame)),
    "valid_case_rows": int(valid_mask.sum()),
    "invalid_case_rows": int((~valid_mask).sum()),
    "unique_valid_smiles": int(len(unique_smiles)),
    "selected_feature_count": int(EXPECTED_SELECTED_K),
    "selected_indices": selected_indices.tolist(),
    "threshold_multipliers": list(AD_THRESHOLD_MULTIPLIERS),
    "ad_k_values": list(AD_NEIGHBOR_K_VALUES),
    "observed_case_targets_available": False,
    "regression_metrics_reported": False,
    "activity_threshold_pIC50": ACTIVITY_PIC50_THRESHOLD,
    "invalid_policy": "retain every row; invalid/missing SMILES are marked INVALID",
    "duplicate_policy": "featurize each unique valid SMILES once; map back to every source row",
}
metadata_path = OUTPUT_DIR / "production_manifest.json"
metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

artifact_paths = sorted(
    path for path in OUTPUT_DIR.rglob("*")
    if path.is_file() and path.name != "artifact_manifest.csv"
)
manifest = pd.DataFrame([
    {
        "Relative_Path": str(path.relative_to(OUTPUT_DIR)),
        "Bytes": path.stat().st_size,
        "SHA256": sha256_file(path),
    }
    for path in artifact_paths
])
manifest_path = OUTPUT_DIR / "artifact_manifest.csv"
manifest.to_csv(manifest_path, index=False)

print("Coverage plot:", figure_path)
print("Production manifest:", metadata_path)
print("Artifact manifest:", manifest_path)
print("Total output files:", len(manifest) + 1)
display(manifest)
'''


def production_notebook(
    dataset: str,
    method: str,
    selected_k: int,
    model_class: str,
    reference_r2: float,
) -> dict:
    # Metadata are identical across generated AD notebooks; use the first one directly.
    template_path = ROOT / "AD_Test_Regression/kaggle_notebooks/01_run_AR_mutual_info_AD.ipynb"
    metadata = json.loads(template_path.read_text(encoding="utf-8"))["metadata"]

    extractor_source = AD_BUILDER._standalone_extractor_source().replace(
        "# STEP 4 — Complete frozen feature-extractor definitions.",
        "# STEP 5 — Complete frozen feature-extractor definitions for Case study.",
        1,
    )
    extractor_source = extractor_source.replace(
        "batch_size=max(1, len(graphs))",
        "batch_size=BATCH_SIZE",
    )
    cells = [
        markdown_cell(
            f"# {dataset} × {method} — Case-study OOD from AD Test output\n\n"
            "This notebook imports the matching `AD_Test_Regression` Kaggle output as "
            "its domain reference and accepted thresholds. It does not clone GitHub, "
            "does not import project `.py` files, and does not rebuild Train/Validation."
        ),
        markdown_cell(
            "## Controlled data flow\n\n"
            "AD output (Train+Validation selected reference + thresholds) + "
            "`cleaned_Casestudy.csv` → unique valid Case SMILES → fresh four-family "
            "features → HStack1024 → frozen scaling/selection/model → pIC50, activity, "
            "and IND/OOD for k=3...25 at multipliers 0.5 and 2.0.\n\n"
            "Case study has no observed pIC50, so regression accuracy metrics are not reported."
        ),
        code_cell(configuration_cell(dataset, method, selected_k, model_class, reference_r2)),
        code_cell(AD_BUILDER.DEPENDENCY_SETUP),
        code_cell(AD_BUILDER._embedded_frozen_cell(dataset, method)),
        code_cell(PATH_RESOLUTION),
        code_cell(AD_BUILDER.FROZEN_LOADING),
        code_cell(LOAD_AD_OUTPUT),
        code_cell(CASE_STUDY_PREP),
        markdown_cell(
            "## Fresh Case-study feature extraction\n\n"
            "The complete frozen extraction implementation is embedded below for audit. "
            "Only unique valid Case-study SMILES are computed; results are mapped back to all rows."
        ),
        code_cell(extractor_source),
        code_cell(FEATURE_PREP),
        code_cell(extract_family_cell("smiles")),
        code_cell(extract_family_cell("selfies")),
        code_cell(extract_family_cell("graph")),
        code_cell(extract_family_cell("fingerprint")),
        code_cell(QUERY_TRANSFORM),
        code_cell(PREDICTION_ACTIVITY),
        code_cell(OOD_EXECUTION),
        code_cell(PLOT_AND_MANIFEST),
        markdown_cell(
            "## Interpretation\n\n"
            "`Predicted_pIC50` is frozen regression output. `Predicted` and `Probability` "
            "come from the Train+Validation-only LDA activity layer. `ADk3`...`ADk25` "
            "are independent applicability warnings. `INVALID` means the source SMILES "
            "could not be featurized and was deliberately retained for audit."
        ),
    ]
    return {"cells": cells, "metadata": metadata, "nbformat": 4, "nbformat_minor": 5}


def rendered_notebooks() -> dict[str, str]:
    return {
        name: json.dumps(
            production_notebook(dataset, method, selected_k, model_class, reference_r2),
            indent=1,
            ensure_ascii=False,
        ) + "\n"
        for name, dataset, method, selected_k, model_class, reference_r2 in OOD_TARGETS
    }


def build() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    expected = rendered_notebooks()
    # Remove superseded 01-12 notebooks from the parent OOD directory. The
    # controlled deliverable now lives only in 01_run-08_run-OOD.
    for path in OOD_ROOT.glob("*.ipynb"):
        path.unlink()
    for path in OUTPUT_DIR.glob("*.ipynb"):
        if path.name not in expected:
            path.unlink()
    for name, content in expected.items():
        (OUTPUT_DIR / name).write_text(content, encoding="utf-8")
    (OUTPUT_DIR / "README.md").write_text(README_CONTENT, encoding="utf-8")
    (OOD_ROOT / "README.md").write_text(PARENT_README_CONTENT, encoding="utf-8")
    print(f"Eight AD-output-driven Case-study OOD notebooks are current: {OUTPUT_DIR}")


def check() -> None:
    expected = rendered_notebooks()
    for name, content in expected.items():
        path = OUTPUT_DIR / name
        if not path.is_file() or path.read_text(encoding="utf-8") != content:
            raise SystemExit(f"OOD notebook is stale or missing: {name}")
    extras = sorted(path.name for path in OUTPUT_DIR.glob("*.ipynb") if path.name not in expected)
    if extras:
        raise SystemExit(f"Unexpected OOD notebooks: {extras}")
    readme = OUTPUT_DIR / "README.md"
    if not readme.is_file() or readme.read_text(encoding="utf-8") != README_CONTENT:
        raise SystemExit(f"OOD README is stale: {readme}")
    parent_readme = OOD_ROOT / "README.md"
    if (
        not parent_readme.is_file()
        or parent_readme.read_text(encoding="utf-8") != PARENT_README_CONTENT
    ):
        raise SystemExit(f"Parent OOD README is stale: {parent_readme}")
    print("Eight Case-study OOD notebooks are complete and current")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    check() if args.check else build()


if __name__ == "__main__":
    main()
