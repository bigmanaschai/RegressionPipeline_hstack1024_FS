from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT.parent / "frozen_hstack1024_ridge" / "src")]
from hstack1024_fs_pipeline.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["--dataset", "PR", "--method", "pearson", *sys.argv[1:]]))
