"""Frozen HStack1024-MinMax-Ridge research replay pipeline."""

from .pipeline import run_all, run_dataset

__version__ = "2.1.0"
PIPELINE_CONTRACT_VERSION = "fresh-raw-split-extraction-cpu-v2"

__all__ = [
    "PIPELINE_CONTRACT_VERSION",
    "run_all",
    "run_dataset",
]
