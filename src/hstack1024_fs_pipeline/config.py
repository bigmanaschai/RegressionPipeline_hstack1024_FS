from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from hstack1024_pipeline.config import DATASETS, normalize_dataset


FS_METHODS = ("mutual_info", "pearson")
HSTACK_DIM = 1024
R2_TOLERANCE = 0.001
BASE_CONTRACT_VERSION = "fresh-raw-split-extraction-cpu-v2"


@dataclass(frozen=True)
class VariantSpec:
    dataset: str
    method: str
    source_stage: str
    result_stage: str
    reference_filename: str
    model_class: str
    selected_k: int
    model_params: dict[str, float]
    reference_sheet: str
    reference_test_r2: float
    selection_rule: str = "Choose_Method == TRUE"

    @property
    def variant_id(self) -> str:
        return f"{self.dataset}_{self.method}"

    @property
    def bundle_filename(self) -> str:
        return f"best_model_{self.dataset}.pkl"


_SPECS = (
    VariantSpec(
        "AR", "mutual_info",
        "arergrpr-ar-hstack1024-selectkbest-mutualinfo-minm",
        "results-arergrpr-ar-hstack1024-selectkbest-mutualinfo-minm",
        "hstack_regression_AR_selectkbest_mutualinfo.csv",
        "ElasticNet", 30, {"alpha": 0.01, "l1_ratio": 0.1},
        "AR_selectkbest_mutualinfo", 0.605790202425591,
    ),
    VariantSpec(
        "AR", "pearson",
        "arergrpr-ar-hstack1024-kbest-corr-pearson-min",
        "results-arergrpr-ar-hstack1024-kbest-corr-pearson-min",
        "hstack_regression_AR_selectkbest_correlation.csv",
        "Ridge", 25, {"alpha": 1.0},
        "AR_selectkbest_correlation", 0.621969056290390,
    ),
    VariantSpec(
        "ER", "mutual_info",
        "arergrpr-er-hstack1024-selectkbest-mutualinfo-minm",
        "results-arergrpr-er-hstack1024-selectkbest-mutualinfo-minm",
        "hstack_regression_ER_selectkbest_mutualinfo.csv",
        "Ridge", 35, {"alpha": 10.0},
        "ER_selectkbest_mutualinfo", 0.769493031166217,
    ),
    VariantSpec(
        "ER", "pearson",
        "arergrpr-er-hstack1024-kbest-corr-pearson-min",
        "results-arergrpr-er-hstack1024-kbest-corr-pearson-min",
        "hstack_regression_ER_selectkbest_correlation.csv",
        "ElasticNet", 30, {"alpha": 0.01, "l1_ratio": 0.1},
        "ER_selectkbest_correlation", 0.768635542381417,
    ),
    VariantSpec(
        "GR", "mutual_info",
        "arergrpr-gr-hstack1024-selectkbest-mutuali",
        "results-arergrpr-gr-hstack1024-selectkbest-mutuali",
        "hstack_regression_GR_selectkbest_mutualinfo.csv",
        "ElasticNet", 30, {"alpha": 0.001, "l1_ratio": 0.1},
        "GR_selectkbest_mutualinfo", 0.589149117475332,
    ),
    VariantSpec(
        "GR", "pearson",
        "arergrpr-gr-hstack1024-kbest-corr-pearson",
        "results-arergrpr-gr-hstack1024-kbest-corr-pearson",
        "hstack_regression_GR_selectkbest_correlation.csv",
        "Ridge", 30, {"alpha": 10.0},
        "GR_selectkbest_correlation", 0.519489428395004,
    ),
    VariantSpec(
        "PR", "mutual_info",
        "arergrpr-pr-hstack1024-selectkbest-mutuali",
        "results-arergrpr-pr-hstack1024-selectkbest-mutuali",
        "hstack_regression_PR_selectkbest_mutualinfo.csv",
        "ElasticNet", 25, {"alpha": 0.01, "l1_ratio": 0.1},
        "PR_selectkbest_mutualinfo", 0.706200850113676,
    ),
    VariantSpec(
        "PR", "pearson",
        "arergrpr-pr-hstack1024-kbest-corr-pearson",
        "results-arergrpr-pr-hstack1024-kbest-corr-pearson",
        "hstack_regression_PR_selectkbest_correlation.csv",
        "Ridge", 25, {"alpha": 10.0},
        "PR_selectkbest_correlation", 0.705562571628553,
    ),
)

VARIANTS = {(spec.dataset, spec.method): spec for spec in _SPECS}


def normalize_method(method: str) -> str:
    value = method.strip().lower().replace("-", "_")
    aliases = {
        "mutuali": "mutual_info",
        "mutualinfo": "mutual_info",
        "mi": "mutual_info",
        "correlation": "pearson",
        "corr_pearson": "pearson",
    }
    value = aliases.get(value, value)
    if value not in FS_METHODS:
        raise ValueError(f"Feature-selection method must be one of {FS_METHODS}, got {method!r}")
    return value


def get_variant(dataset: str, method: str) -> VariantSpec:
    return VARIANTS[(normalize_dataset(dataset), normalize_method(method))]


def repository_root() -> Path:
    return Path(__file__).resolve().parents[4]


def method_source_root(repo_root: Optional[Path] = None) -> Path:
    root = repository_root() if repo_root is None else Path(repo_root)
    return root / "standard_pipeline" / "the_best_method_for_pipeline"


def default_material_root(repo_root: Optional[Path] = None) -> Path:
    root = repository_root() if repo_root is None else Path(repo_root)
    return (
        root
        / "standard_pipeline"
        / "RegressionPipeline_hstack1024_FS"
        / "materials"
        / "hstack1024-fs-extension"
    )


def is_material_bundle(path: Optional[Path]) -> bool:
    if path is None:
        return False
    root = Path(path)
    return (root / "MANIFEST.json").is_file() and (root / "models").is_dir()


def original_result_dir(
    dataset: str, method: str, repo_root: Optional[Path] = None
) -> Path:
    spec = get_variant(dataset, method)
    return method_source_root(repo_root) / spec.dataset / spec.source_stage / spec.result_stage


def bundle_path(
    dataset: str,
    method: str,
    *,
    fs_root: Optional[Path] = None,
    repo_root: Optional[Path] = None,
) -> Path:
    spec = get_variant(dataset, method)
    if is_material_bundle(fs_root):
        descriptor = Path(fs_root) / "models" / f"{spec.variant_id}_bundle.json"
        if descriptor.is_file():
            return descriptor
        return Path(fs_root) / "models" / f"{spec.variant_id}_best_model.pkl"
    local_material = default_material_root(repo_root)
    if is_material_bundle(local_material):
        descriptor = local_material / "models" / f"{spec.variant_id}_bundle.json"
        if descriptor.is_file():
            return descriptor
        return local_material / "models" / f"{spec.variant_id}_best_model.pkl"
    return original_result_dir(dataset, method, repo_root) / spec.bundle_filename


def reference_path(
    dataset: str,
    method: str,
    *,
    fs_root: Optional[Path] = None,
    repo_root: Optional[Path] = None,
) -> Path:
    spec = get_variant(dataset, method)
    if is_material_bundle(fs_root):
        return Path(fs_root) / "references" / f"{spec.variant_id}_reference.csv"
    local_material = default_material_root(repo_root)
    if is_material_bundle(local_material):
        return local_material / "references" / f"{spec.variant_id}_reference.csv"
    return original_result_dir(dataset, method, repo_root) / spec.reference_filename


def all_variants() -> tuple[VariantSpec, ...]:
    return tuple(VARIANTS[(dataset, method)] for dataset in DATASETS for method in FS_METHODS)
