"""Generate clean production notebooks for new-data regression + OOD analysis."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from build_kaggle_notebooks import TARGETS, code_cell, markdown_cell, notebook


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "kaggle_notebooks" / "OOD-Regression"
OOD_TARGETS = tuple(
    (name.replace(".ipynb", "_OOD.ipynb"), dataset, method)
    for name, dataset, method in TARGETS
)

README_CONTENT = """# Production OOD Regression notebooks

These nine notebooks are clean new-compound inference pipelines for the eight
frozen AR/ER/GR/PR × mutual_info/pearson final models. They do **not** replay
historical test metrics, read reference CSV/workbook values, or use reproduce-00
split-oracle outputs.

New compounds are read from:

`/kaggle/input/datasets/manaschaiaonon/hstack1024-pipeline-libs/cleaned_Casestudy.csv`

The current file has `ID,Smiles` columns and no experimental pIC50, so outputs
contain frozen-model `Predicted_pIC50` plus `ADk3`…`ADk25`; no R2/RMSE/MAE is
calculated for the new compounds.

Production contract:

- domain reference: the deterministic 60% training partition recreated from
  the corresponding raw AR/ER/GR/PR dataset, without hash/oracle checks;
- query population: `cleaned_Casestudy.csv`;
- feature space: frozen-MinMax-scaled HStack1024 before SelectKBest (1024 d);
- prediction: frozen scaler → frozen selector → frozen Ridge/ElasticNet;
- OOD: kNN mean distance for every k from 3 through 25, threshold = training
  mean + 0.5 SD;
- no scaler, selector, or regression-model refitting.

Required Kaggle data inputs are only:

1. `manaschaiaonon/hstack1024-pipeline-libs` (deep checkpoints, fingerprint
   transformers, and `cleaned_Casestudy.csv`); and
2. `plenoi/ar-er-gr-pr` (raw endpoint datasets used to recreate the current
   training-domain reference).

The four reproduce-00 notebook outputs are not used and should be detached from
these production OOD notebooks. Historical reference tables/workbooks are also
not execution inputs.

Each variant writes:

- `production_predictions_ood.csv`;
- `ood_summary_k3_k25.csv`; and
- `production_manifest.json`.

The run also writes aggregate `ood_summary.csv`, `ood_summary.json` (run-all),
and `ood_coverage_diagnostics.png`. CPU inference over 17,855 query compounds is
substantial; the run-all notebook performs fresh extraction separately for all
four receptor checkpoints.
"""


def production_notebook(dataset: str, method: str) -> dict:
    base = notebook(dataset, method)
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
    title = (
        "AR/ER/GR/PR × mutual_info/pearson"
        if dataset == "ALL"
        else f"{dataset} × {method}"
    )
    cells = [
        markdown_cell(
            f"""# {title} — production HStack1024 regression + OOD

This notebook performs clean inference on new compounds from
`cleaned_Casestudy.csv`. It uses the frozen HStack1024 extractor, frozen
MinMaxScaler, frozen SelectKBest selector, and frozen final regressor. It does
not replay historical Test-R2, compare against reference results, or consume
reproduce-00 split-oracle outputs.
"""
        ),
        markdown_cell(
            f"""## Production workflow: Input → Prediction → OOD

Run target: **{title}**

| Step | Stage | Input | Process | Output |
|---:|---|---|---|---|
| 1 | Runtime | GitHub source + frozen Kaggle assets | Load versioned production runtime | Ready pipeline |
| 2 | Domain reference | Raw AR/ER/GR/PR CSV | Standard cleaning and deterministic 60% train partition; no oracle checks | Training SMILES |
| 3 | Query | `cleaned_Casestudy.csv` (`ID,Smiles`) | Schema and SMILES safety checks | New compounds |
| 4 | Extraction | Train + query SMILES | Fresh CPU SMILES/SELFIES/Graph/Fingerprint inference | HStack1024 |
| 5 | Prediction | Query HStack1024 | Frozen MinMax → frozen SelectKBest → frozen Ridge/ElasticNet | Predicted pIC50 |
| 6 | OOD | Scaled train/query HStack1024 | Professor kNN threshold for every k=3…25 | IND/OOD labels and coverage |
| 7 | Export | Predictions + AD labels | Write CSV/JSON provenance | Production artifacts |

The query file has no measured pIC50. Accordingly, this production run does not
calculate R2, RMSE, MAE, residuals, or historical-reference deltas.
"""
        ),
        code_cell(
            f'''# Production configuration
RUN_DATASET = {dataset!r}
RUN_METHOD = {method!r}

GITHUB_REPO_URL = "https://github.com/bigmanaschai/RegressionPipeline_hstack1024_FS.git"
GITHUB_REF = "main"
PROJECT_ROOT_OVERRIDE = ""
BASE_ROOT_OVERRIDE = ""
FS_ROOT_OVERRIDE = ""
QUERY_CSV = "/kaggle/input/datasets/manaschaiaonon/hstack1024-pipeline-libs/cleaned_Casestudy.csv"
QUERY_ID_COLUMN = "ID"
QUERY_SMILES_COLUMN = "Smiles"
OUTPUT_ROOT_OVERRIDE = ""
OOD_K_VALUES = tuple(range(3, 26))

assert RUN_DATASET in {{"AR", "ER", "GR", "PR", "ALL"}}
assert RUN_METHOD in {{"mutual_info", "pearson", "ALL"}}
assert OOD_K_VALUES == tuple(range(3, 26))
'''
        ),
        clone_cell,
        dependency_cell,
        runtime_cell,
        code_cell(
            '''# Resolve only the production inputs and frozen artifacts.
from hstack1024_pipeline.config import DATASETS, FAMILY_ORDER, checkpoint_path, fingerprint_transformer_path, raw_csv_path
from hstack1024_fs_pipeline.config import FS_METHODS, bundle_path, get_variant
from hstack1024_fs_pipeline.ood import load_query_compounds

datasets = tuple(DATASETS) if RUN_DATASET == "ALL" else (RUN_DATASET,)
methods = FS_METHODS if RUN_METHOD == "ALL" else (RUN_METHOD,)
raw_csv_paths = {dataset: raw_csv_path(dataset, BASE_ROOT) for dataset in datasets}
query_csv = Path(QUERY_CSV).expanduser()

if not query_csv.is_file():
    raise FileNotFoundError(f"Production query CSV not found: {query_csv}")
query_preview = load_query_compounds(
    query_csv,
    id_column=QUERY_ID_COLUMN,
    smiles_column=QUERY_SMILES_COLUMN,
)
for dataset in datasets:
    if not raw_csv_paths[dataset].is_file():
        raise FileNotFoundError(raw_csv_paths[dataset])
    for family in FAMILY_ORDER:
        artifact = checkpoint_path(dataset, family, BASE_ROOT)
        if not artifact.is_file():
            raise FileNotFoundError(artifact)
    transformer = fingerprint_transformer_path(dataset, BASE_ROOT)
    if not transformer.is_file():
        raise FileNotFoundError(transformer)
    for method in methods:
        model_bundle = bundle_path(dataset, method, fs_root=FS_ROOT)
        if not model_bundle.is_file():
            raise FileNotFoundError(model_bundle)
        spec = get_variant(dataset, method)
        print("Production variant:", spec.variant_id, spec.model_class, f"FS_k={spec.selected_k}")

print("Query CSV      :", query_csv)
print("Query rows     :", len(query_preview))
print("Query columns  :", list(query_preview.columns))
print("Production inputs: OK")
'''
        ),
        code_cell(
            '''# Run frozen prediction and professor-defined OOD analysis on new compounds.
from hstack1024_fs_pipeline.ood import run_production_ood_all, run_production_ood_dataset

target_name = "all" if RUN_DATASET == "ALL" else f"{RUN_DATASET}_{RUN_METHOD}"
default_output = Path("/kaggle/working") / f"hstack1024_production_ood_{target_name}"
OUTPUT_ROOT = Path(OUTPUT_ROOT_OVERRIDE).expanduser() if OUTPUT_ROOT_OVERRIDE else default_output

if RUN_DATASET == "ALL":
    ood_report = run_production_ood_all(
        query_csv,
        OUTPUT_ROOT,
        datasets=datasets,
        methods=methods,
        base_root=BASE_ROOT,
        fs_root=FS_ROOT,
        repo_root=PROJECT_ROOT,
        raw_csv_paths=raw_csv_paths,
        id_column=QUERY_ID_COLUMN,
        smiles_column=QUERY_SMILES_COLUMN,
    )
else:
    ood_report = run_production_ood_dataset(
        RUN_DATASET,
        query_csv,
        OUTPUT_ROOT,
        methods=methods,
        base_root=BASE_ROOT,
        fs_root=FS_ROOT,
        repo_root=PROJECT_ROOT,
        raw_csv=raw_csv_paths[RUN_DATASET],
        id_column=QUERY_ID_COLUMN,
        smiles_column=QUERY_SMILES_COLUMN,
    )

print("Production output:", OUTPUT_ROOT)
ood_report
'''
        ),
        code_cell(
            '''# Plot domain coverage across the mandatory k=3..25 range.
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
for (dataset, method), frame in ood_report.groupby(["Dataset", "FS_Method"], sort=True):
    ax.plot(
        frame["k"],
        frame["IND_Coverage"],
        marker="o",
        label=f"{dataset}-{method}",
    )
ax.set(title="New-compound applicability-domain coverage", xlabel="k", ylabel="IND coverage")
ax.set_xticks(range(3, 26, 2))
ax.set_ylim(0.0, 1.05)
ax.grid(alpha=0.25)
ax.legend(fontsize=8)
figure_path = OUTPUT_ROOT / "ood_coverage_diagnostics.png"
fig.savefig(figure_path, dpi=180, bbox_inches="tight")
plt.show()
print("Coverage figure:", figure_path)
'''
        ),
        code_cell(
            '''# Show one production output and list every saved artifact.
prediction_files = sorted(OUTPUT_ROOT.rglob("production_predictions_ood.csv"))
if not prediction_files:
    raise RuntimeError("No production prediction file was written")
sample_predictions = __import__("pandas").read_csv(prediction_files[0])
print("Prediction files:", len(prediction_files))
print("Example file:", prediction_files[0])
display(sample_predictions.head())
for path in sorted(OUTPUT_ROOT.rglob("*")):
    if path.is_file():
        print(path.relative_to(OUTPUT_ROOT))
'''
        ),
        markdown_cell(
            """## Production interpretation

- `Predicted_pIC50` is generated by the frozen final model selected for that
  dataset/method; it is not an experimentally observed value.
- `ADk3`…`ADk25` report `IND` or `OOD` independently at every controlled k.
- `IND_Coverage` summarizes how much of the new dataset lies inside the current
  training domain; OOD status is an applicability warning, not a class label.
- Because `cleaned_Casestudy.csv` has no measured pIC50, no performance metric
  should be inferred from these predictions.
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
            production_notebook(dataset, method), indent=1, ensure_ascii=False
        )
        + "\n"
        for name, dataset, method in OOD_TARGETS
    }


def build() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, content in rendered_notebooks().items():
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
