from __future__ import annotations

import copy
import json
from pathlib import Path


AD_ROOT = Path(__file__).resolve().parent
EXTENSION_ROOT = AD_ROOT.parent
TEMPLATE_ROOT = EXTENSION_ROOT / "kaggle_notebooks"
OUTPUT_ROOT = AD_ROOT / "kaggle_notebooks"

VARIANTS = (
    ("01", "AR", "mutual_info"),
    ("02", "AR", "pearson"),
    ("03", "ER", "mutual_info"),
    ("04", "ER", "pearson"),
    ("05", "GR", "mutual_info"),
    ("06", "GR", "pearson"),
    ("07", "PR", "mutual_info"),
    ("08", "PR", "pearson"),
)


def _lines(text: str) -> list[str]:
    return text.splitlines(keepends=True)


def _markdown(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": _lines(text),
    }


def _code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": _lines(text),
    }


def _configuration(dataset: str, method: str) -> str:
    return f'''# Configuration: edit AD_THRESHOLD_MULTIPLIERS here.
RUN_DATASET = {dataset!r}
RUN_METHOD = {method!r}
TOLERANCE = 0.001

# One CSV is created for each multiplier. Examples:
#   (0.5, 2.0) -> two CSV files
#   (1.0,)     -> one CSV file
AD_THRESHOLD_MULTIPLIERS = (0.5, 2.0)
AD_NEIGHBOR_K_VALUES = tuple(range(3, 26))

GITHUB_REPO_URL = "https://github.com/bigmanaschai/RegressionPipeline_hstack1024_FS.git"
GITHUB_REF = "main"    # branch, tag, or commit available to git clone
PROJECT_ROOT_OVERRIDE = ""  # existing checkout; takes priority over clone
BASE_ROOT_OVERRIDE = ""     # blank = standard Kaggle base-library path
FS_ROOT_OVERRIDE = ""       # blank = materials in the GitHub checkout
SPLIT_ROOT = "/kaggle/input"
OUTPUT_ROOT_OVERRIDE = ""

assert RUN_DATASET in {{"AR", "ER", "GR", "PR"}}
assert RUN_METHOD in {{"mutual_info", "pearson"}}
assert AD_THRESHOLD_MULTIPLIERS
assert all(float(value) >= 0.0 for value in AD_THRESHOLD_MULTIPLIERS)
assert tuple(AD_NEIGHBOR_K_VALUES) == tuple(range(3, 26))
'''


def _ad_import_cell() -> str:
    return '''# Load the manifest-verified AD runtime from the same FS source bundle.
ad_source = FS_SRC / "ad_test_regression" / "pipeline.py"
if not ad_source.is_file():
    raise RuntimeError(
        "Test-regression AD runtime is absent from the material bundle: "
        f"{ad_source}. Rebuild materials, push them, restart the session, and Run All."
    )

for module_name in list(sys.modules):
    if module_name == "ad_test_regression" or module_name.startswith("ad_test_regression."):
        del sys.modules[module_name]
importlib.invalidate_caches()

import ad_test_regression
imported_ad = Path(ad_test_regression.__file__).resolve().parent
expected_ad = (FS_SRC / "ad_test_regression").resolve()
if imported_ad != expected_ad:
    raise RuntimeError(f"Imported AD runtime from {imported_ad}; expected {expected_ad}")
print("Test AD runtime:", imported_ad)
'''


def _run_cell() -> str:
    return '''# Run Test-set regression AD and write one CSV per threshold multiplier.
import pandas as pd
from ad_test_regression.pipeline import run_test_ad_variant

target_name = f"{RUN_DATASET}_{RUN_METHOD}"
default_output = Path("/kaggle/working") / f"AD_Test_Regression_{target_name}"
OUTPUT_ROOT = Path(OUTPUT_ROOT_OVERRIDE).expanduser() if OUTPUT_ROOT_OVERRIDE else default_output

output_paths = run_test_ad_variant(
    RUN_DATASET,
    RUN_METHOD,
    OUTPUT_ROOT,
    base_root=BASE_ROOT,
    fs_root=FS_ROOT,
    repo_root=PROJECT_ROOT,
    raw_csv=raw_csv_paths[RUN_DATASET],
    split_paths=split_paths_by_dataset[RUN_DATASET],
    threshold_multipliers=AD_THRESHOLD_MULTIPLIERS,
    k_values=AD_NEIGHBOR_K_VALUES,
    tolerance=TOLERANCE,
)

print("Output root:", OUTPUT_ROOT)
for multiplier, output_path in output_paths.items():
    frame = pd.read_csv(output_path)
    print(f"threshold multiplier={multiplier}: {output_path}")
    display(frame)
'''


def build_notebook(sequence: str, dataset: str, method: str) -> Path:
    template_path = TEMPLATE_ROOT / f"{sequence}_run_{dataset}_{method}.ipynb"
    if not template_path.is_file():
        raise FileNotFoundError(template_path)
    template = json.loads(template_path.read_text(encoding="utf-8"))
    if len(template.get("cells", [])) < 8:
        raise RuntimeError(f"Unexpected template structure: {template_path}")

    title = f"{dataset} × {method} — Test regression applicability domain"
    cells = [
        _markdown(
            f"# {title}\n\n"
            "This notebook replays the frozen HStack1024 + SelectKBest regression "
            "variant, builds the applicability-domain reference from Train+Validation, "
            "and evaluates the locked Test split. It creates one summary CSV for each "
            "configured threshold multiplier. No model, scaler, or selector is fitted.\n"
        ),
        _markdown(
            "## Method\n\n"
            f"Run target: **{dataset} × {method}**\n\n"
            "1. Recreate and verify the locked 60/20/20 split.\n"
            "2. Extract fresh SMILES, SELFIES, Graph, and Fingerprint features.\n"
            "3. Apply the frozen MinMaxScaler and frozen SelectKBest selector.\n"
            "4. Combine Train+Validation selected features as the AD reference.\n"
            "5. Predict the locked Test split with the frozen final regressor.\n"
            "6. For AD k=3…25, classify Test rows as IND/OOD with Euclidean kNN.\n"
            "7. Report R2, RMSE, MAE, ME, Pearson, and Spearman on IND rows.\n"
            "8. Write one CSV per threshold multiplier.\n\n"
            "The `No AD` row uses every Test row. `ME` is prediction minus observation.\n"
        ),
        _code(_configuration(dataset, method)),
        copy.deepcopy(template["cells"][3]),
        copy.deepcopy(template["cells"][4]),
        copy.deepcopy(template["cells"][5]),
        _code(_ad_import_cell()),
        copy.deepcopy(template["cells"][6]),
        _code(_run_cell()),
        _markdown(
            "## Output interpretation\n\n"
            "Each CSV contains `No AD` followed by exactly one row for every AD k from "
            "3 through 25. `Coverage = INDs / N_Test` and `OODs = N_Test - INDs`. "
            "A larger threshold multiplier must not reduce Coverage for the same AD k.\n"
        ),
    ]
    notebook = {
        "cells": cells,
        "metadata": copy.deepcopy(template.get("metadata", {})),
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            cell["execution_count"] = None
            cell["outputs"] = []

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_ROOT / f"{sequence}_run_{dataset}_{method}_AD.ipynb"
    output_path.write_text(
        json.dumps(notebook, indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return output_path


def main() -> None:
    paths = [build_notebook(*variant) for variant in VARIANTS]
    if len(paths) != 8 or len(set(paths)) != 8:
        raise AssertionError("Expected exactly eight unique AD notebooks")
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
