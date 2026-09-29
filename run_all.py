from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
BASE_SRC = ROOT.parent / "frozen_hstack1024_ridge" / "src"
sys.path[:0] = [str(ROOT / "src"), str(BASE_SRC)]

from hstack1024_fs_pipeline.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
