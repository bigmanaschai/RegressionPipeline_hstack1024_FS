"""Build the small GitHub/Kaggle extension containing eight final FS bundles."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[1]
BASE_SRC = ROOT.parent / "frozen_hstack1024_ridge" / "src"
sys.path[:0] = [str(ROOT / "src"), str(BASE_SRC)]

from hstack1024_fs_pipeline.config import all_variants, original_result_dir  # noqa: E402


DESTINATION = ROOT / "materials" / "hstack1024-fs-extension"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact_sources() -> dict[Path, Path]:
    files: dict[Path, Path] = {}
    for spec in all_variants():
        result = original_result_dir(spec.dataset, spec.method, REPO_ROOT)
        files[Path("references") / f"{spec.variant_id}_reference.csv"] = (
            result / spec.reference_filename
        )
        if spec.variant_id == "AR_pearson":
            for role, source in component_sources(spec).items():
                files[Path("models") / f"{spec.variant_id}_{role}.pkl"] = source
        else:
            files[Path("models") / f"{spec.variant_id}_best_model.pkl"] = (
                result / spec.bundle_filename
            )
    return files


def component_sources(spec) -> dict[str, Path]:
    model_root = original_result_dir(
        spec.dataset, spec.method, REPO_ROOT
    ) / "models"
    if spec.method == "mutual_info":
        model_name = f"{spec.dataset}_{spec.model_class}_k{spec.selected_k}.pkl"
        selector_name = (
            f"{spec.dataset}_selectkbest_mutualinfo_k{spec.selected_k}.pkl"
        )
    else:
        model_name = (
            f"{spec.dataset}_{spec.model_class}_pearson_k{spec.selected_k}.pkl"
        )
        selector_name = (
            f"{spec.dataset}_selectkbest_correlation_pearson_k{spec.selected_k}.pkl"
        )
    sources = {
        "model": model_root / model_name,
        "selector": model_root / selector_name,
        "normalizer": model_root / f"{spec.dataset}_normalizer_minmax.pkl",
    }
    missing = [str(path) for path in sources.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing frozen component(s): {missing}")
    return sources


def selected_reference_row(spec) -> dict[str, str]:
    path = original_result_dir(
        spec.dataset, spec.method, REPO_ROOT
    ) / spec.reference_filename
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    selected = [
        row
        for row in rows
        if row["Model"] == spec.model_class
        and row["FS_Method"] == spec.method
        and int(row["FS_k"]) == spec.selected_k
    ]
    if len(selected) != 1:
        raise AssertionError(
            f"{path}: expected one row for {spec.variant_id}, found {len(selected)}"
        )
    row = selected[0]
    if abs(float(row["TSR2"]) - spec.reference_test_r2) > 1e-12:
        raise AssertionError(
            f"{path}: TSR2 differs from {spec.reference_sheet}/{spec.selection_rule}"
        )
    return row


def component_descriptor(spec) -> dict:
    row = selected_reference_row(spec)
    return {
        "format": "frozen_component_bundle_v1",
        "components": {
            role: f"{spec.variant_id}_{role}.pkl"
            for role in ("model", "selector", "normalizer")
        },
        "metadata": {
            "dataset": spec.dataset,
            "model_name": spec.model_class,
            "FS_Method": spec.method,
            "FS_k": spec.selected_k,
            "selector_type": "correlation_pearson",
            "selected_features": row["Selected_Features"],
            "feature_order": ["smiles", "selfies", "graph", "fingerprint"],
            "feat_dim": 1024,
            "selected_by": (
                "arergrpr-HStack1024 Feature Selection.xlsx/"
                f"{spec.reference_sheet}: {spec.selection_rule}"
            ),
            "best_params": row["Best_Params"],
            "CVR2": float(row["CVR2"]),
            "CVRMSE": float(row["CVRMSE"]),
            "CVMAE": float(row["CVMAE"]),
            "TSR2": float(row["TSR2"]),
            "TSRMSE": float(row["TSRMSE"]),
            "TSMAE": float(row["TSMAE"]),
            "sklearn_version": "1.2.2",
        },
    }


def patched_base_config() -> bytes:
    """Recognize Kaggle's base bundle from its required checkpoint directory."""
    source = (BASE_SRC / "hstack1024_pipeline" / "config.py").read_text(
        encoding="utf-8"
    )
    old = """    return (
        (root / \"arergrpr-hstack1024-baseline.xlsx\").is_file()
        and (root / \"src\" / \"hstack1024_pipeline\").is_dir()
    )
"""
    new = """    return (root / \"checkpoints\").is_dir()
"""
    if source.count(old) != 1:
        raise AssertionError("Unexpected base config: flat-bundle detector changed")
    return source.replace(old, new).encode("utf-8")


def generated_materials() -> dict[Path, bytes]:
    ar_pearson = next(spec for spec in all_variants() if spec.variant_id == "AR_pearson")
    selections = {
        "source_workbook": "arergrpr-HStack1024 Feature Selection.xlsx",
        "variants": [
            {
                "dataset": spec.dataset,
                "method": spec.method,
                "sheet": spec.reference_sheet,
                "selection_rule": spec.selection_rule,
                "model": spec.model_class,
                "FS_k": spec.selected_k,
                "Reference_Test_R2": spec.reference_test_r2,
            }
            for spec in all_variants()
        ],
    }
    return {
        Path("base_src/hstack1024_pipeline/config.py"): patched_base_config(),
        Path("models/AR_pearson_bundle.json"): (
            json.dumps(component_descriptor(ar_pearson), indent=2, ensure_ascii=False)
            + "\n"
        ).encode("utf-8"),
        Path("SELECTIONS.json"): (
            json.dumps(selections, indent=2, ensure_ascii=False) + "\n"
        ).encode("utf-8"),
    }


def source_tree_files() -> dict[Path, Path]:
    source_dir = ROOT / "src"
    return {
        Path("src") / path.relative_to(source_dir): path
        for path in sorted(source_dir.rglob("*.py"))
    }


def base_source_tree_files() -> dict[Path, Path]:
    """Vendor the small, exact base runtime; large artifacts remain in Kaggle Input."""
    return {
        Path("base_src") / path.relative_to(BASE_SRC): path
        for path in sorted(BASE_SRC.rglob("*.py"))
        if path.relative_to(BASE_SRC) != Path("hstack1024_pipeline/config.py")
    }


def contract_payload() -> dict:
    return {
        "contract_version": "fresh-raw-split-extraction-selectkbest-cpu-v1",
        "base_contract_version": "fresh-raw-split-extraction-cpu-v2",
        "base_kaggle_dataset": "manaschaiaonon/hstack1024-pipeline-libs",
        "training_performed": False,
        "precomputed_features_included": False,
        "inference_device": "cpu",
        "quality_gate": {"metric": "Test_R2", "absolute_tolerance": 0.001},
        "optional_post_analysis": {
            "name": "production regression, LDA activity reporting, and kNN applicability domain",
            "protocol_source": (
                "standard_pipeline/OOD_ajPle/"
                "ppar-2098c-ad-analyze-web-figure.ipynb"
            ),
            "default_query": (
                "/kaggle/input/datasets/manaschaiaonon/"
                "ood-regression-arergrpr/cleaned_Casestudy.csv"
            ),
            "notebook_count": 12,
            "feature_spaces": {
                "hstack1024": "4 endpoint notebooks at 1024 dimensions",
                "selectkbest": "8 endpoint/method notebooks at frozen FS_k",
            },
            "domain_reference": "raw-derived Train+Validation 80%; Test excluded",
            "k_values": list(range(3, 26)),
            "threshold": "training mean kNN distance + 0.5 * training SD",
            "activity_classifier": "LinearDiscriminantAnalysis(tol=0.00001)",
            "activity_threshold_pic50": 6.0,
            "probability": "predict_proba for class 0 (Positive)",
            "historical_reference_comparison_performed": False,
            "split_oracle_used": False,
            "test_split_used": False,
            "regression_model_refitted": False,
            "activity_classifier_fitted": True,
        },
        "variants": [
            {
                "dataset": spec.dataset,
                "method": spec.method,
                "model_class": spec.model_class,
                "selected_k": spec.selected_k,
                "model_params": spec.model_params,
                "source_stage": spec.source_stage,
                "reference_sheet": spec.reference_sheet,
                "selection_rule": spec.selection_rule,
                "reference_test_r2": spec.reference_test_r2,
            }
            for spec in all_variants()
        ],
    }


def expected_manifest() -> dict:
    sources = {**artifact_sources(), **source_tree_files(), **base_source_tree_files()}
    generated = generated_materials()
    return {
        "material_slug": "hstack1024-fs-extension",
        "mode": "fresh HStack1024 extraction plus frozen SelectKBest/final-model inference",
        "base_kaggle_dataset_required": "manaschaiaonon/hstack1024-pipeline-libs",
        "training_performed": False,
        "precomputed_features_included": False,
        "inference_device": "cpu",
        "variant_count": 8,
        "files": {
            relative.as_posix(): {
                "sha256": sha256(source),
                "bytes": source.stat().st_size,
            }
            for relative, source in sources.items()
        } | {
            relative.as_posix(): {
                "sha256": hashlib.sha256(content).hexdigest(),
                "bytes": len(content),
            }
            for relative, content in generated.items()
        },
    }


def build() -> None:
    DESTINATION.mkdir(parents=True, exist_ok=True)
    sources = {**artifact_sources(), **source_tree_files(), **base_source_tree_files()}
    for relative, source in sources.items():
        target = DESTINATION / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    for relative, content in generated_materials().items():
        target = DESTINATION / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    legacy = DESTINATION / "models" / "AR_pearson_best_model.pkl"
    if legacy.exists():
        legacy.unlink()
    (DESTINATION / "CONTRACT.json").write_text(
        json.dumps(contract_payload(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (DESTINATION / "MANIFEST.json").write_text(
        json.dumps(expected_manifest(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (DESTINATION / "README.txt").write_text(
        "HStack1024 SelectKBest extension materials (8 frozen final pipelines).\n"
        "Contains: exact base/FS/OOD runtime source, 8 frozen final bundles, 8 reference tables.\n"
        "Does not contain precomputed sample features or the 4.1 GB deep-feature assets.\n"
        "Standard replay requires the base library, raw AR/ER/GR/PR CSV input, "
        "and the four reproduce-00 split-oracle outputs.\n"
        "CPU inference only; no selector, scaler, or estimator training/refitting occurs.\n"
        "Production OOD uses Train+Validation 80%, excludes Test, fits the professor-style "
        "LDA activity layer plus a kNN domain index, and reports k=3..25.\n",
        encoding="utf-8",
    )


def check() -> None:
    manifest_path = DESTINATION / "MANIFEST.json"
    if not manifest_path.is_file():
        raise SystemExit("Materials are missing; run build_materials.py")
    recorded = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = expected_manifest()
    if recorded != expected:
        raise SystemExit("Materials manifest is stale")
    for relative, metadata in expected["files"].items():
        target = DESTINATION / relative
        if not target.is_file() or sha256(target) != metadata["sha256"]:
            raise SystemExit(f"Material file is stale or missing: {relative}")
    contract = json.loads((DESTINATION / "CONTRACT.json").read_text(encoding="utf-8"))
    if contract != contract_payload():
        raise SystemExit("CONTRACT.json is stale")
    if (DESTINATION / "models" / "AR_pearson_best_model.pkl").exists():
        raise SystemExit("Stale AR_pearson best_model bundle must be removed")
    print("HStack1024 FS materials are complete and current")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if not args.check:
        build()
    check()
    if not args.check:
        print("ready:", DESTINATION)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
