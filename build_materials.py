"""Build the small GitHub/Kaggle extension containing eight final FS bundles."""

from __future__ import annotations

import argparse
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
        files[Path("models") / f"{spec.variant_id}_best_model.pkl"] = (
            result / spec.bundle_filename
        )
        files[Path("references") / f"{spec.variant_id}_reference.csv"] = (
            result / spec.reference_filename
        )
    return files


def source_tree_files() -> dict[Path, Path]:
    source_dir = ROOT / "src"
    return {
        Path("src") / path.relative_to(source_dir): path
        for path in sorted(source_dir.rglob("*.py"))
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
        "variants": [
            {
                "dataset": spec.dataset,
                "method": spec.method,
                "model_class": spec.model_class,
                "selected_k": spec.selected_k,
                "model_params": spec.model_params,
                "source_stage": spec.source_stage,
            }
            for spec in all_variants()
        ],
    }


def expected_manifest() -> dict:
    sources = {**artifact_sources(), **source_tree_files()}
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
        },
    }


def build() -> None:
    DESTINATION.mkdir(parents=True, exist_ok=True)
    for relative, source in {**artifact_sources(), **source_tree_files()}.items():
        target = DESTINATION / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
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
        "Contains: source package, 8 frozen final bundles, 8 reference tables.\n"
        "Does not contain precomputed sample features or the 4.1 GB deep-feature assets.\n"
        "Kaggle also requires manaschaiaonon/hstack1024-pipeline-libs, raw AR/ER/GR/PR "
        "CSV input, and the four reproduce-00 split-oracle outputs.\n"
        "CPU inference only; no selector, scaler, or estimator training/refitting occurs.\n",
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
