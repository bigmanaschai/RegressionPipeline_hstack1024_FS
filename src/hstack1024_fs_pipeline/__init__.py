"""Frozen HStack1024 + SelectKBest regression replay pipelines."""

from .pipeline import run_all, run_dataset, run_variant

__version__ = "1.0.0"
PIPELINE_CONTRACT_VERSION = "fresh-raw-split-extraction-selectkbest-cpu-v1"

__all__ = [
    "PIPELINE_CONTRACT_VERSION",
    "run_all",
    "run_dataset",
    "run_variant",
]
