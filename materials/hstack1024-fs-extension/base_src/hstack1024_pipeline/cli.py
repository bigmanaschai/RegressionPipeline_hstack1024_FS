from __future__ import annotations

import argparse
from pathlib import Path

from .config import DATASETS, R2_TOLERANCE
from .pipeline import run_all, run_dataset


def discover_split_paths(dataset: str, root: Path) -> dict:
    paths = {}
    for split in ("train", "val", "test"):
        name = f"{dataset}_split_{split}.scl"
        hits = sorted(root.rglob(name))
        if len(hits) != 1:
            raise SystemExit(
                f"Expected exactly one {name} under {root}, found {len(hits)}"
            )
        paths[split] = hits[0]
    manifest_name = f"{dataset}_split_manifest.json"
    manifests = sorted(root.rglob(manifest_name))
    if len(manifests) != 1:
        raise SystemExit(
            f"Expected exactly one {manifest_name} under {root}, found {len(manifests)}"
        )
    paths["manifest"] = manifests[0]
    return paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Replay the frozen HStack1024-MinMax-Ridge research pipeline."
    )
    parser.add_argument("--dataset", choices=[*DATASETS, "ALL"], default="ALL")
    parser.add_argument("--raw-csv", type=Path)
    parser.add_argument(
        "--split-root",
        type=Path,
        default=Path("/kaggle/input"),
        help="Root containing outputs from the four reproduce-00-datasplit notebooks.",
    )
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
    if args.dataset == "ALL":
        split_paths = {
            dataset: discover_split_paths(dataset, args.split_root)
            for dataset in DATASETS
        }
        report = run_all(
            args.output_dir,
            feature_source="extract",
            split_paths_by_dataset=split_paths,
            tolerance=args.tolerance,
        )
    else:
        report = run_dataset(
            args.dataset,
            args.output_dir,
            feature_source="extract",
            raw_csv=args.raw_csv,
            split_paths=discover_split_paths(args.dataset, args.split_root),
            tolerance=args.tolerance,
        )
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
