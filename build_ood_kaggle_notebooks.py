"""Generate the 12 production regression + activity + OOD notebooks."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from build_kaggle_notebooks import code_cell, markdown_cell, notebook


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "kaggle_notebooks" / "OOD-Regression"
OOD_TARGETS = (
    ("01_run_AR_mutual_info_OOD.ipynb", "AR", "selectkbest", "mutual_info"),
    ("02_run_AR_pearson_OOD.ipynb", "AR", "selectkbest", "pearson"),
    ("03_run_ER_mutual_info_OOD.ipynb", "ER", "selectkbest", "mutual_info"),
    ("04_run_ER_pearson_OOD.ipynb", "ER", "selectkbest", "pearson"),
    ("05_run_GR_mutual_info_OOD.ipynb", "GR", "selectkbest", "mutual_info"),
    ("06_run_GR_pearson_OOD.ipynb", "GR", "selectkbest", "pearson"),
    ("07_run_PR_mutual_info_OOD.ipynb", "PR", "selectkbest", "mutual_info"),
    ("08_run_PR_pearson_OOD.ipynb", "PR", "selectkbest", "pearson"),
    ("09_run_AR_hstack1024_OOD.ipynb", "AR", "hstack1024", None),
    ("10_run_ER_hstack1024_OOD.ipynb", "ER", "hstack1024", None),
    ("11_run_GR_hstack1024_OOD.ipynb", "GR", "hstack1024", None),
    ("12_run_PR_hstack1024_OOD.ipynb", "PR", "hstack1024", None),
)

README_CONTENT = """# Production OOD Regression notebooks

This directory contains exactly 12 standalone Kaggle notebooks:

- `01`–`08`: AR/ER/GR/PR × mutual_info/pearson, with OOD calculated in the
  matching frozen-MinMax-scaled HStack1024 + frozen SelectKBest (`FS_k`) space;
- `09`–`12`: one notebook per endpoint, with OOD calculated in the full
  frozen-MinMax-scaled HStack1024 space (1024 d).

Every notebook recreates the deterministic 60/20/20 raw split, combines only
Train+Validation into `smiles_tr` (80%), and excludes Test from prediction,
activity classification, calibration, and OOD. All rows in
`/kaggle/input/datasets/manaschaiaonon/ood-regression-arergrpr/cleaned_Casestudy.csv`
become `data`.

The professor's reporting layer is reproduced with a new
`LinearDiscriminantAnalysis(tol=0.00001)` model fitted only on Train+Validation.
Class 0 is Positive (`pIC50 >= 6.0`), class 1 is Negative (`pIC50 < 6.0`), and
`Probability` is `predict_proba(...)[class 0]`. The kNN OOD calculation is
independent of this probability calculation.

Each notebook writes:

- `IND_Result.csv` with the professor's exact leading columns
  `[index],Smiles,Predicted,Probability`, followed by `ADk3` through `ADk25`;
- `production_predictions_ood.csv` with IDs, regression pIC50, and provenance;
- `ood_summary_k3_k25.csv`;
- `activity_classifier_lda.joblib`;
- `production_manifest.json`; and
- `ood_coverage_diagnostics.png`.

Required Kaggle inputs:

1. `manaschaiaonon/hstack1024-pipeline-libs` for frozen extractors, scalers,
   and the four baseline Ridge models;
2. `manaschaiaonon/ood-regression-arergrpr` for `cleaned_Casestudy.csv`; and
3. `plenoi/ar-er-gr-pr` for the four raw endpoint regression datasets.

MACCSFingerprint and reproduce-00 notebook outputs are not used.
The controlled OOD range covers every k from 3 through 25.
"""


def production_notebook(
    dataset: str,
    feature_space: str,
    method: str | None,
) -> dict:
    base = notebook(dataset, method or "mutual_info")
    clone_cell = copy.deepcopy(base["cells"][3])
    dependency_cell = copy.deepcopy(base["cells"][4])
    runtime_cell = copy.deepcopy(base["cells"][5])
    runtime_source = "".join(runtime_cell["source"])
    runtime_source += '''

# Require the production OOD runtime from the checked-out, manifest-verified source.
ood_source = FS_SRC / "hstack1024_fs_pipeline" / "ood.py"
if not ood_source.is_file():
    raise RuntimeError(
        "Production OOD runtime is absent from this checkout: "
        f"{ood_source}. Restart the session and Run All with the latest GITHUB_REF."
    )
ood_module = importlib.import_module("hstack1024_fs_pipeline.ood")
if Path(ood_module.__file__).resolve() != ood_source.resolve():
    raise RuntimeError(
        f"Imported production OOD runtime from {ood_module.__file__!r}; "
        f"expected {ood_source}"
    )
print("Production OOD runtime:", ood_module.__file__)
'''
    runtime_cell["source"] = runtime_source.splitlines(keepends=True)

    if feature_space == "selectkbest":
        title = f"{dataset} × {method}"
        feature_description = (
            "frozen-MinMax-scaled HStack1024 + frozen SelectKBest "
            "(the selected FS_k dimensions)"
        )
        prediction_description = "frozen SelectKBest + frozen Ridge/ElasticNet"
    else:
        title = f"{dataset} × full HStack1024"
        feature_description = "frozen-MinMax-scaled HStack1024 (1024 dimensions)"
        prediction_description = "frozen baseline Ridge"

    cells = [
        markdown_cell(
            f"""# {title} — production regression + activity reporting + OOD

This notebook uses all query rows and excludes Test completely. `smiles_tr` is
the endpoint-specific Train+Validation 80%; `data` is every SMILES in
`cleaned_Casestudy.csv`. OOD uses {feature_description}. MACCSFingerprint is
not used.
"""
        ),
        markdown_cell(
            f"""## Controlled workflow

| Step | Stage | Input | Process | Output |
|---:|---|---|---|---|
| 1 | Domain | Raw {dataset} CSV | Deterministic 60/20/20; combine Train+Validation only | `smiles_tr` (80%) |
| 2 | Query | `cleaned_Casestudy.csv` | Use every validated `Smiles` row | `data` |
| 3 | Extraction | `smiles_tr` + `data` | Frozen SMILES/SELFIES/Graph/ECFP extractors | HStack1024 |
| 4 | OOD space | HStack1024 | {feature_description} | `X_train`, `X_test` |
| 5 | Regression | Query feature vector | {prediction_description} | Predicted pIC50 |
| 6 | Activity layer | Train+Validation pIC50 | Professor-style LDA; Positive if pIC50 ≥ 6.0 | Predicted + Probability |
| 7 | OOD | `X_train`, `X_test` | Professor kNN equation for k=3…25 | ADk3…ADk25 |
| 8 | Export | Activity + OOD results | Professor-compatible column order | `IND_Result.csv` |

`Probability` is the LDA posterior probability of class 0 (Positive). It is not
generated by kNN. Test rows never enter the LDA, regression inference, or OOD.
"""
        ),
        code_cell(
            f'''# Production configuration
RUN_DATASET = {dataset!r}
RUN_FEATURE_SPACE = {feature_space!r}
RUN_METHOD = {method!r}

GITHUB_REPO_URL = "https://github.com/bigmanaschai/RegressionPipeline_hstack1024_FS.git"
GITHUB_REF = "main"
PROJECT_ROOT_OVERRIDE = ""
BASE_ROOT_OVERRIDE = ""
FS_ROOT_OVERRIDE = ""
QUERY_CSV = "/kaggle/input/datasets/manaschaiaonon/ood-regression-arergrpr/cleaned_Casestudy.csv"
QUERY_ID_COLUMN = "ID"
QUERY_SMILES_COLUMN = "Smiles"
OUTPUT_ROOT_OVERRIDE = ""
ACTIVITY_PIC50_THRESHOLD = 6.0
OOD_K_VALUES = tuple(range(3, 26))

assert RUN_DATASET in {{"AR", "ER", "GR", "PR"}}
assert RUN_FEATURE_SPACE in {{"hstack1024", "selectkbest"}}
assert (RUN_METHOD in {{"mutual_info", "pearson"}}) == (RUN_FEATURE_SPACE == "selectkbest")
assert OOD_K_VALUES == tuple(range(3, 26))
'''
        ),
        clone_cell,
        dependency_cell,
        runtime_cell,
        code_cell(
            '''# Resolve only the production inputs and frozen artifacts.
from hstack1024_pipeline.config import (
    FAMILY_ORDER, checkpoint_path, fingerprint_transformer_path, raw_csv_path,
    ridge_path, scaler_path,
)
from hstack1024_fs_pipeline.config import bundle_path, get_variant
from hstack1024_fs_pipeline.ood import load_query_compounds

raw_csv = raw_csv_path(RUN_DATASET, BASE_ROOT)
query_csv = Path(QUERY_CSV).expanduser()
if not raw_csv.is_file():
    raise FileNotFoundError(raw_csv)
if not query_csv.is_file():
    raise FileNotFoundError(f"Production query CSV not found: {query_csv}")
query_preview = load_query_compounds(
    query_csv, id_column=QUERY_ID_COLUMN, smiles_column=QUERY_SMILES_COLUMN
)
for family in FAMILY_ORDER:
    artifact = checkpoint_path(RUN_DATASET, family, BASE_ROOT)
    if not artifact.is_file():
        raise FileNotFoundError(artifact)
transformer = fingerprint_transformer_path(RUN_DATASET, BASE_ROOT)
if not transformer.is_file():
    raise FileNotFoundError(transformer)

if RUN_FEATURE_SPACE == "selectkbest":
    model_artifact = bundle_path(RUN_DATASET, RUN_METHOD, fs_root=FS_ROOT)
    spec = get_variant(RUN_DATASET, RUN_METHOD)
    print("Variant:", spec.variant_id, spec.model_class, f"FS_k={spec.selected_k}")
else:
    model_artifact = ridge_path(RUN_DATASET, BASE_ROOT)
    normalizer_artifact = scaler_path(RUN_DATASET, BASE_ROOT)
    if not normalizer_artifact.is_file():
        raise FileNotFoundError(normalizer_artifact)
if not model_artifact.is_file():
    raise FileNotFoundError(model_artifact)

print("Query CSV  :", query_csv)
print("Query rows :", len(query_preview))
print("Feature OOD:", RUN_FEATURE_SPACE)
print("Inputs: OK")
'''
        ),
        code_cell(
            '''# Run regression, professor-style LDA activity reporting, and OOD.
from hstack1024_fs_pipeline.ood import run_production_ood_dataset

variant_name = (
    f"{RUN_DATASET}_{RUN_METHOD}"
    if RUN_FEATURE_SPACE == "selectkbest"
    else f"{RUN_DATASET}_hstack1024"
)
default_output = Path("/kaggle/working") / f"production_ood_{variant_name}"
OUTPUT_ROOT = Path(OUTPUT_ROOT_OVERRIDE).expanduser() if OUTPUT_ROOT_OVERRIDE else default_output

ood_report = run_production_ood_dataset(
    RUN_DATASET,
    query_csv,
    OUTPUT_ROOT,
    feature_space=RUN_FEATURE_SPACE,
    method=RUN_METHOD,
    base_root=BASE_ROOT,
    fs_root=FS_ROOT,
    repo_root=PROJECT_ROOT,
    raw_csv=raw_csv,
    id_column=QUERY_ID_COLUMN,
    smiles_column=QUERY_SMILES_COLUMN,
    activity_threshold=ACTIVITY_PIC50_THRESHOLD,
)
print("Production output:", OUTPUT_ROOT)
ood_report
'''
        ),
        code_cell(
            '''# Plot IND coverage for all mandatory k values.
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
ax.plot(ood_report["k"], ood_report["IND_Coverage"], marker="o")
ax.set(title="New-compound applicability-domain coverage", xlabel="k", ylabel="IND coverage")
ax.set_xticks(range(3, 26, 2))
ax.set_ylim(0.0, 1.05)
ax.grid(alpha=0.25)
figure_path = OUTPUT_ROOT / "ood_coverage_diagnostics.png"
fig.savefig(figure_path, dpi=180, bbox_inches="tight")
plt.show()
print("Coverage figure:", figure_path)
'''
        ),
        code_cell(
            '''# Verify and display the professor-compatible report.
import pandas as pd

result_files = sorted(OUTPUT_ROOT.rglob("IND_Result.csv"))
if len(result_files) != 1:
    raise RuntimeError(f"Expected exactly one IND_Result.csv, found {len(result_files)}")
detailed_files = sorted(OUTPUT_ROOT.rglob("production_predictions_ood.csv"))
if len(detailed_files) != 1:
    raise RuntimeError(
        f"Expected exactly one production_predictions_ood.csv, found {len(detailed_files)}"
    )
result = pd.read_csv(result_files[0])
expected = ["Unnamed: 0", "Smiles", "Predicted", "Probability"] + [f"ADk{k}" for k in range(3, 26)]
if result.columns.tolist() != expected:
    raise AssertionError(f"IND_Result.csv columns changed: {result.columns.tolist()}")
if not set(result["Predicted"].unique()).issubset({"Positive", "Negative"}):
    raise AssertionError("Unexpected Predicted label")
if not result["Probability"].between(0.0, 1.0).all():
    raise AssertionError("Probability must be in [0,1]")
print("IND result:", result_files[0])
print("Rows:", len(result))
display(result.head())
for path in sorted(OUTPUT_ROOT.rglob("*")):
    if path.is_file():
        print(path.relative_to(OUTPUT_ROOT))
'''
        ),
        markdown_cell(
            """## Interpretation

- `Predicted` is Positive/Negative from the newly fitted LDA activity layer.
- `Probability` is the LDA probability for Positive, not a kNN probability.
- `ADk3`…`ADk25` independently report IND/OOD using the professor's equation.
- `Predicted_pIC50` remains available in `production_predictions_ood.csv`.
- OOD status is an applicability warning and does not replace activity class.
"""
        ),
    ]
    return {
        "cells": cells,
        "metadata": copy.deepcopy(base["metadata"]),
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def rendered_notebooks() -> dict[str, str]:
    return {
        name: json.dumps(
            production_notebook(dataset, feature_space, method),
            indent=1,
            ensure_ascii=False,
        )
        + "\n"
        for name, dataset, feature_space, method in OOD_TARGETS
    }


def build() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    expected = rendered_notebooks()
    for path in OUTPUT_DIR.glob("*.ipynb"):
        if path.name not in expected:
            path.unlink()
    for name, content in expected.items():
        (OUTPUT_DIR / name).write_text(content, encoding="utf-8")
    (OUTPUT_DIR / "README.md").write_text(README_CONTENT, encoding="utf-8")
    print(f"Production OOD notebooks are current: {OUTPUT_DIR}")


def check() -> None:
    expected = rendered_notebooks()
    for name, content in expected.items():
        path = OUTPUT_DIR / name
        if not path.is_file() or path.read_text(encoding="utf-8") != content:
            raise SystemExit(f"Production OOD notebook is stale or missing: {name}")
    readme = OUTPUT_DIR / "README.md"
    if not readme.is_file() or readme.read_text(encoding="utf-8") != README_CONTENT:
        raise SystemExit(f"Production OOD README is stale or missing: {readme}")
    extra = sorted(
        path.name for path in OUTPUT_DIR.glob("*.ipynb") if path.name not in expected
    )
    if extra:
        raise SystemExit(f"Unexpected OOD notebooks: {extra}")
    print("Production OOD notebooks are complete and current")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    check() if args.check else build()


if __name__ == "__main__":
    main()
