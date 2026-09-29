from __future__ import annotations

import argparse
from pathlib import Path

from hstack1024_pipeline.config import DATASETS

from .config import FS_METHODS, R2_TOLERANCE
from .pipeline import run_all, run_dataset, run_variant


def discover_split_paths(dataset: str, root: Path) -> dict[str, Path]:
    paths = {}
    for split in ("train", "val", "test"):
        name = f"{dataset}_split_{split}.scl"
        hits = sorted(root.rglob(name))
        if len(hits) != 1:
            raise SystemExit(f"Expected exactly one {name} under {root}, found {len(hits)}")
        paths[split] = hits[0]
    manifest_name = f"{dataset}_split_manifest.json"
    hits = sorted(root.rglob(manifest_name))
    if len(hits) != 1:
        raise SystemExit(
            f"Expected exactly one {manifest_name} under {root}, found {len(hits)}"
        )
    paths["manifest"] = hits[0]
    return paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Replay the eight frozen HStack1024 + SelectKBest final pipelines."
    )
    parser.add_argument("--dataset", choices=[*DATASETS, "ALL"], default="ALL")
    parser.add_argument("--method", choices=[*FS_METHODS, "ALL"], default="ALL")
    parser.add_argument("--base-root", type=Path)
    parser.add_argument("--fs-root", type=Path)
    parser.add_argument("--repo-root", type=Path)
    parser.add_argument("--raw-csv", type=Path)
    parser.add_argument("--split-root", type=Path, default=Path("/kaggle/input"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "outputs" / "latest",
    )
    parser.add_argument("--tolerance", type=float, default=R2_TOLERANCE)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.raw_csv is not None and args.dataset == "ALL":
        raise SystemExit("--raw-csv can only be used with one --dataset")
    methods = FS_METHODS if args.method == "ALL" else (args.method,)
    common = {
        "feature_source": "extract",
        "base_root": args.base_root,
        "fs_root": args.fs_root,
        "repo_root": args.repo_root,
        "tolerance": args.tolerance,
    }
    if args.dataset == "ALL":
        report = run_all(
            args.output_dir,
            methods=methods,
            split_paths_by_dataset={
                dataset: discover_split_paths(dataset, args.split_root)
                for dataset in DATASETS
            },
            **common,
        )
    elif args.method == "ALL":
        report = run_dataset(
            args.dataset,
            args.output_dir,
            methods=methods,
            raw_csv=args.raw_csv,
            split_paths=discover_split_paths(args.dataset, args.split_root),
            **common,
        )
    else:
        report = run_variant(
            args.dataset,
            args.method,
            args.output_dir,
            raw_csv=args.raw_csv,
            split_paths=discover_split_paths(args.dataset, args.split_root),
            **common,
        )
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
