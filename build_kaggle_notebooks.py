"""Generate nine synchronized Kaggle notebooks for the eight FS variants."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "kaggle_notebooks"

TARGETS = (
    ("00_run_all_hstack1024_fs.ipynb", "ALL", "ALL"),
    ("01_run_AR_mutual_info.ipynb", "AR", "mutual_info"),
    ("02_run_AR_pearson.ipynb", "AR", "pearson"),
    ("03_run_ER_mutual_info.ipynb", "ER", "mutual_info"),
    ("04_run_ER_pearson.ipynb", "ER", "pearson"),
    ("05_run_GR_mutual_info.ipynb", "GR", "mutual_info"),
    ("06_run_GR_pearson.ipynb", "GR", "pearson"),
    ("07_run_PR_mutual_info.ipynb", "PR", "mutual_info"),
    ("08_run_PR_pearson.ipynb", "PR", "pearson"),
)


def markdown_cell(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def notebook(dataset: str, method: str) -> dict:
    title = "AR/ER/GR/PR × mutual_info/pearson" if dataset == "ALL" else f"{dataset} × {method}"
    cells = [
        markdown_cell(
            f"""# {title} — frozen HStack1024 + SelectKBest pipeline

This notebook pulls the standard extension from GitHub (or uses a checked-out
copy), recreates raw-derived splits, extracts fresh frozen HStack1024 features,
then applies the frozen MinMaxScaler, frozen SelectKBest selector, and frozen
CV-selected final regressor. It performs no fitting, tuning, or cached-feature
input. Execution is CPU-only and Test R² must match the historical reference
within absolute tolerance 0.001.
"""
        ),
        markdown_cell(
            f"""## Pipeline workflow: Input → Process → Output

Run target: **{title}**

| Step | Stage | Input | Process | Output |
|---:|---|---|---|---|
| 1 | Source | GitHub repository/ref | Clone the immutable standard-pipeline source and small FS materials | Versioned runtime package |
| 2 | Preflight | Base frozen library + FS manifest | Validate package contracts, checksums, artifacts, and CPU device | Verified inputs |
| 3 | Raw preprocessing | Dataset raw CSV | RDKit validity, first-SMILES deduplication, finite-target filter | Clean population |
| 4 | Split | Clean population | qcut-10 stratified 60/20/20 with seed 0 | Raw-derived train/validation/test |
| 5 | Split validation | Raw-derived rows + reproduce-00 artifacts | Exact ordered SMILES/target, count, hash, and disjointness checks | Accepted raw-derived split |
| 6 | Deep extraction | Validation/Test SMILES + frozen checkpoints | Fresh CPU SMILES, SELFIES, Graph, and Fingerprint inference | Four aligned 256-d families |
| 7 | Integration | Four feature families | Fixed-order `smiles → selfies → graph → fingerprint` concatenation | HStack1024 |
| 8 | Normalization | HStack1024 + frozen scaler | Transform only | Scaled 1024-d features |
| 9 | Selection | Scaled features + frozen SelectKBest | Apply frozen `mutual_info` or `pearson` support | k=25/30/35 features |
| 10 | Prediction | Selected features + frozen final regressor | Frozen Ridge/ElasticNet inference | Test predictions |
| 11 | Acceptance | Predictions + variant reference CSV | Require `abs(Test_R2 − reference) ≤ 0.001` | PASS/FAIL, metrics, manifest, features |

`reproduce-00` arrays and historical feature files are validation evidence only;
they never become feature-extraction or prediction inputs.
"""
        ),
        code_cell(
            f'''# Configuration
RUN_DATASET = {dataset!r}
RUN_METHOD = {method!r}
TOLERANCE = 0.001

GITHUB_REPO_URL = "https://github.com/bigmanaschai/RegressionPipeline_hstack1024_FS.git"
GITHUB_REF = "main"    # branch, tag, or commit available to git clone
PROJECT_ROOT_OVERRIDE = ""  # existing checkout; takes priority over clone
BASE_ROOT_OVERRIDE = ""     # blank = standard Kaggle base-library path
FS_ROOT_OVERRIDE = ""       # blank = materials in the GitHub checkout
SPLIT_ROOT = "/kaggle/input"
OUTPUT_ROOT_OVERRIDE = ""

assert RUN_DATASET in {{"AR", "ER", "GR", "PR", "ALL"}}
assert RUN_METHOD in {{"mutual_info", "pearson", "ALL"}}
'''
        ),
        code_cell(
            '''# Resolve or clone the GitHub project without modifying Kaggle Inputs.
from pathlib import Path
import subprocess

EXPECTED_RELATIVE = Path("standard_pipeline/RegressionPipeline_hstack1024_FS")

def find_extension_root(project_root):
    """Support both the monorepo layout and this package's standalone GitHub repo."""
    nested = project_root / EXPECTED_RELATIVE
    if nested.is_dir():
        return nested
    if (
        (project_root / "src" / "hstack1024_fs_pipeline").is_dir()
        and (project_root / "materials" / "hstack1024-fs-extension").is_dir()
        and (project_root / "pyproject.toml").is_file()
    ):
        return project_root
    raise FileNotFoundError(
        f"RegressionPipeline_hstack1024_FS source not found under: {project_root}"
    )

if PROJECT_ROOT_OVERRIDE:
    PROJECT_ROOT = Path(PROJECT_ROOT_OVERRIDE).expanduser().resolve()
elif (Path.cwd() / EXPECTED_RELATIVE).is_dir():
    PROJECT_ROOT = Path.cwd().resolve()
else:
    if not GITHUB_REPO_URL:
        raise ValueError(
            "Set GITHUB_REPO_URL to the repository pushed from this materials folder, "
            "or set PROJECT_ROOT_OVERRIDE to an existing checkout."
        )
    checkout = Path("/kaggle/working/hstack1024-fs-repository")
    if not checkout.is_dir():
        subprocess.check_call(
            ["git", "clone", "--no-checkout", "--filter=blob:none", GITHUB_REPO_URL, str(checkout)]
        )
    subprocess.check_call(
        ["git", "-C", str(checkout), "fetch", "--depth", "1", "origin", GITHUB_REF]
    )
    subprocess.check_call(
        ["git", "-C", str(checkout), "checkout", "--detach", "FETCH_HEAD"]
    )
    PROJECT_ROOT = checkout.resolve()

EXTENSION_ROOT = find_extension_root(PROJECT_ROOT)
print("Project root:", PROJECT_ROOT)
print("Extension   :", EXTENSION_ROOT)
'''
        ),
        code_cell(
            '''# Install only missing optional deep-inference dependencies.
import importlib.util
import subprocess
import sys

required_modules = {
    "transformers": "transformers==4.48.3",
    "sentencepiece": "sentencepiece",
    "selfies": "selfies",
    "rdkit": "rdkit",
    "deepchem": "deepchem",
    "torch_geometric": "torch-geometric",
    "skfp": "git+https://github.com/plenoi/scikit-finger-plenoi.git@master",
}
missing = [package for module, package in required_modules.items() if importlib.util.find_spec(module) is None]
if missing:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing])

from transformers import AutoModel, AutoTokenizer

def pin_remote_code_revision(auto_class):
    original = auto_class.from_pretrained
    if getattr(original, "_hstack_revision_pinned", False):
        return
    def pinned(pretrained_model_name_or_path, *args, **kwargs):
        revision = kwargs.get("revision")
        if revision:
            kwargs.setdefault("code_revision", revision)
        return original(pretrained_model_name_or_path, *args, **kwargs)
    pinned._hstack_revision_pinned = True
    auto_class.from_pretrained = pinned

pin_remote_code_revision(AutoTokenizer)
pin_remote_code_revision(AutoModel)
print("Optional dependencies: OK")
'''
        ),
        code_cell(
            '''# Resolve packages and verify the small GitHub material bundle byte-for-byte.
import hashlib
import importlib
import json
import sys

BASE_KAGGLE_ROOT = Path("/kaggle/input/datasets/manaschaiaonon/hstack1024-pipeline-libs")
LOCAL_BASE_ROOT = PROJECT_ROOT / "standard_pipeline/frozen_hstack1024_ridge/kaggle_dataset/hstack1024-pipeline-libs"
BASE_ROOT = Path(BASE_ROOT_OVERRIDE).expanduser() if BASE_ROOT_OVERRIDE else (
    BASE_KAGGLE_ROOT if BASE_KAGGLE_ROOT.is_dir() else LOCAL_BASE_ROOT
)
FS_ROOT = Path(FS_ROOT_OVERRIDE).expanduser() if FS_ROOT_OVERRIDE else (
    EXTENSION_ROOT / "materials/hstack1024-fs-extension"
)
BASE_SRC = BASE_ROOT / "src"
FS_SRC = FS_ROOT / "src"
if not (BASE_SRC / "hstack1024_pipeline").is_dir():
    raise FileNotFoundError(f"Attach base Kaggle library; package not found: {BASE_SRC}")
if not (FS_SRC / "hstack1024_fs_pipeline").is_dir():
    raise FileNotFoundError(f"FS source package not found: {FS_SRC}")

manifest = json.loads((FS_ROOT / "MANIFEST.json").read_text(encoding="utf-8"))
def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
for relative, metadata in manifest["files"].items():
    path = FS_ROOT / relative
    if not path.is_file() or sha256(path) != metadata["sha256"]:
        raise RuntimeError(f"FS material missing or stale: {relative}")

for module_name in list(sys.modules):
    if (
        module_name == "hstack1024_pipeline"
        or module_name.startswith("hstack1024_pipeline.")
        or module_name == "hstack1024_fs_pipeline"
        or module_name.startswith("hstack1024_fs_pipeline.")
    ):
        del sys.modules[module_name]
sys.path[:0] = [str(FS_SRC), str(BASE_SRC)]
importlib.invalidate_caches()

import hstack1024_pipeline
import hstack1024_fs_pipeline

expected_imports = {
    "hstack1024_pipeline": (hstack1024_pipeline, BASE_SRC / "hstack1024_pipeline"),
    "hstack1024_fs_pipeline": (hstack1024_fs_pipeline, FS_SRC / "hstack1024_fs_pipeline"),
}
for package_name, (package, expected_dir) in expected_imports.items():
    imported_file = getattr(package, "__file__", None)
    if imported_file is None or Path(imported_file).resolve().parent != expected_dir.resolve():
        raise RuntimeError(
            f"Imported {package_name} from {imported_file!r}; expected {expected_dir}"
        )

base_contract = getattr(hstack1024_pipeline, "PIPELINE_CONTRACT_VERSION", None)
if base_contract != "fresh-raw-split-extraction-cpu-v2":
    raise RuntimeError(
        "Incompatible base HStack1024 contract: "
        f"{base_contract!r} from {hstack1024_pipeline.__file__}"
    )
fs_contract = getattr(hstack1024_fs_pipeline, "PIPELINE_CONTRACT_VERSION", None)
if fs_contract != "fresh-raw-split-extraction-selectkbest-cpu-v1":
    raise RuntimeError(
        "Incompatible HStack1024 FS contract: "
        f"{fs_contract!r} from {hstack1024_fs_pipeline.__file__}"
    )
print("Base root     :", BASE_ROOT)
print("FS materials  :", FS_ROOT)
print("Inference device: CPU")
'''
        ),
        code_cell(
            '''# Preflight raw, split-oracle, deep-model, final-model, and reference inputs.
from hstack1024_pipeline.config import DATASETS, FAMILY_ORDER, checkpoint_path, fingerprint_transformer_path, raw_csv_path
from hstack1024_fs_pipeline.cli import discover_split_paths
from hstack1024_fs_pipeline.config import FS_METHODS, bundle_path, reference_path

datasets = tuple(DATASETS) if RUN_DATASET == "ALL" else (RUN_DATASET,)
methods = FS_METHODS if RUN_METHOD == "ALL" else (RUN_METHOD,)
split_root = Path(SPLIT_ROOT)
split_paths_by_dataset = {dataset: discover_split_paths(dataset, split_root) for dataset in datasets}
raw_csv_paths = {dataset: raw_csv_path(dataset, BASE_ROOT) for dataset in datasets}

for dataset in datasets:
    if not raw_csv_paths[dataset].is_file():
        raise FileNotFoundError(raw_csv_paths[dataset])
    for family in FAMILY_ORDER:
        if not checkpoint_path(dataset, family, BASE_ROOT).is_file():
            raise FileNotFoundError(checkpoint_path(dataset, family, BASE_ROOT))
    if not fingerprint_transformer_path(dataset, BASE_ROOT).is_file():
        raise FileNotFoundError(fingerprint_transformer_path(dataset, BASE_ROOT))
    for method in methods:
        if not bundle_path(dataset, method, fs_root=FS_ROOT).is_file():
            raise FileNotFoundError(bundle_path(dataset, method, fs_root=FS_ROOT))
        if not reference_path(dataset, method, fs_root=FS_ROOT).is_file():
            raise FileNotFoundError(reference_path(dataset, method, fs_root=FS_ROOT))
print("Preflight: OK", datasets, methods)
'''
        ),
        code_cell(
            '''# Run fresh extraction and the frozen variant(s).
from hstack1024_fs_pipeline.pipeline import run_all, run_dataset, run_variant

target_name = "all" if RUN_DATASET == "ALL" else f"{RUN_DATASET}_{RUN_METHOD}"
default_output = Path("/kaggle/working") / f"hstack1024_fs_{target_name}"
OUTPUT_ROOT = Path(OUTPUT_ROOT_OVERRIDE).expanduser() if OUTPUT_ROOT_OVERRIDE else default_output

common = dict(
    feature_source="extract",
    base_root=BASE_ROOT,
    fs_root=FS_ROOT,
    repo_root=PROJECT_ROOT,
    tolerance=TOLERANCE,
)
if RUN_DATASET == "ALL":
    report = run_all(
        OUTPUT_ROOT,
        methods=methods,
        raw_csv_paths=raw_csv_paths,
        split_paths_by_dataset=split_paths_by_dataset,
        **common,
    )
elif RUN_METHOD == "ALL":
    report = run_dataset(
        RUN_DATASET,
        OUTPUT_ROOT,
        methods=methods,
        raw_csv=raw_csv_paths[RUN_DATASET],
        split_paths=split_paths_by_dataset[RUN_DATASET],
        **common,
    )
else:
    report = run_variant(
        RUN_DATASET,
        RUN_METHOD,
        OUTPUT_ROOT,
        raw_csv=raw_csv_paths[RUN_DATASET],
        split_paths=split_paths_by_dataset[RUN_DATASET],
        **common,
    )
print("Output root:", OUTPUT_ROOT)
report
'''
        ),
        markdown_cell(
            """## Result interpretation

`PASS` means the target was actually run and all structural checks plus the
Test-R² tolerance gate passed. A compiled notebook or successful preflight is
not a PASS. Keep `run_manifest.json`, `metrics.json`, predictions, selected
indices/scores, selected features, and freshly extracted family features as the
audit record.
"""
        ),
    ]
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10"},
            "kaggle": {"accelerator": "none", "isInternetEnabled": True},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def rendered_notebooks() -> dict[str, str]:
    return {
        name: json.dumps(notebook(dataset, method), indent=1, ensure_ascii=False) + "\n"
        for name, dataset, method in TARGETS
    }


def build() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, content in rendered_notebooks().items():
        (OUTPUT_DIR / name).write_text(content, encoding="utf-8")


def check() -> None:
    for name, content in rendered_notebooks().items():
        path = OUTPUT_DIR / name
        if not path.is_file() or path.read_text(encoding="utf-8") != content:
            raise SystemExit(f"Notebook is stale or missing: {name}")
    print("Kaggle notebooks are complete and current")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if not args.check:
        build()
    check()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
