from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional


FAMILY_ORDER = ("smiles", "selfies", "graph", "fingerprint")
FAMILY_DIM = 256
HSTACK_DIM = 1024
MODEL_NAME = "Ridge"
RIDGE_ALPHA = 100.0
R2_TOLERANCE = 0.001
KAGGLE_RAW_DATA_ROOT = Path("/kaggle/input/datasets/plenoi/ar-er-gr-pr")


@dataclass(frozen=True)
class DatasetSpec:
    dataset: str
    raw_csv_name: str
    n_val: int
    n_test: int
    canonical_split_md5: Dict[str, str]
    ordered_split_sha256: Dict[str, str]


DATASETS = {
    "AR": DatasetSpec(
        "AR",
        "AR_regression.csv",
        386,
        387,
        {
            "train": "9157a42bd7da0590",
            "val": "0c26403acd5ad627",
            "test": "45de4d5c02af07cf",
        },
        {
            "val_smiles": "550dcd541557c0f884fa39ee749ed91b976e251d8ace7508be83299aa83b4a5b",
            "test_smiles": "528c2191dc2698c5672418fb86e78be0dad9f099cfa664b0dcb9b288ad390c37",
            "val_targets": "b802b79633bb32e96369546d1537ab7e59cc2421875fd891b632200cf734eab3",
            "test_targets": "11d516c0d861455c399ce6a9674e329ca2a8fe3a1bc83bbb47e85de14c0d8fc5",
        },
    ),
    "ER": DatasetSpec(
        "ER",
        "ERalpha_regression.csv",
        505,
        505,
        {
            "train": "4cfe4cf00d1ed620",
            "val": "d3755b70a1815d8e",
            "test": "da4fc0c0979e2788",
        },
        {
            "val_smiles": "25ef64c862d31ec7106a3af30a84ff0ffbff278d29ae211964338d5f44fa07b8",
            "test_smiles": "d92f606a03ac63ee3118d08687520863d64aaecc014795098a3ea6660d9c8958",
            "val_targets": "d6a7ad1848f962a5d11223f2581ca33d628b8d7bc8fd9cec71d7e532dc629b43",
            "test_targets": "2bd7490750611dffbdb91bb9570377950a283b1e94186e0f1acdaa4aefbec0c9",
        },
    ),
    # The surviving GR/PR notebooks loaded their hashes from external manifests.
    # Those manifests are not present in this repository.  Ordered split identity
    # is therefore locked by the four matching feature artifacts instead.
    "GR": DatasetSpec(
        "GR",
        "GR_regression.csv",
        327,
        328,
        {},
        {
            "val_smiles": "7f81d003ba34737058a5895b2adcade1b3041cb66d135f16bf8c36a063007bdd",
            "test_smiles": "b41e1d7f042afc2104ac16c0eea29ead5b3a70f91a8f44a4e48b852c5b9e52b9",
            "val_targets": "180fd9c396ae21888a74bf3270701fdca1d570a79c241bfa409e83ea0ebca4c1",
            "test_targets": "8ac3296a4c04faeeee460d762238cda2d68185a3758171713c814f87f9866304",
        },
    ),
    "PR": DatasetSpec(
        "PR",
        "PR_regression.csv",
        271,
        272,
        {},
        {
            "val_smiles": "a55090e4477cfbdfd936290a4f1dd9192ec51e4323e27e6b33e643fda1bf73e5",
            "test_smiles": "ef88533b5408d8ecf32150a27662862afaa0a84440c7c086ea6536c975b22522",
            "val_targets": "f74322149eca6d4f6fe0f0abaea272950a98b197edfef5cc0558a0b8fbac764f",
            "test_targets": "9307f4af72ca85847a3a470bf7d839f4a71ef7012c5b20d29faefa1de614d6b3",
        },
    ),
}


def repository_root() -> Path:
    return Path(__file__).resolve().parents[4]


def source_root(repo_root: Optional[Path] = None) -> Path:
    root = repository_root() if repo_root is None else Path(repo_root)
    return root / "standard_pipeline" / "the_best_method_for_pipeline"


def is_flat_kaggle_bundle(repo_root: Optional[Path]) -> bool:
    if repo_root is None:
        return False
    root = Path(repo_root)
    return all(
        (root / directory).is_dir()
        for directory in ("checkpoints", "models", "transformers")
    )


def baseline_workbook(repo_root: Optional[Path] = None) -> Path:
    root = repository_root() if repo_root is None else Path(repo_root)
    if is_flat_kaggle_bundle(root):
        return root / "arergrpr-hstack1024-baseline.xlsx"
    return root / "standard_pipeline" / "OOD_ajPle" / "arergrpr-hstack1024-baseline.xlsx"


def hstack_result_dir(dataset: str, repo_root: Optional[Path] = None) -> Path:
    ds = normalize_dataset(dataset)
    stem = f"arergrpr-{ds.lower()}-hstack1024-baseline-minmax"
    return source_root(repo_root) / ds / stem / f"results-{stem}"


def checkpoint_path(
    dataset: str, family: str, repo_root: Optional[Path] = None
) -> Path:
    ds = normalize_dataset(dataset)
    lower = ds.lower()
    if is_flat_kaggle_bundle(repo_root):
        return Path(repo_root) / "checkpoints" / f"model_{family}_{ds}.pt"
    root = source_root(repo_root) / ds
    if family == "smiles":
        stage = f"reproduce-01-smiles-{lower}"
        return root / stage / f"results-{stage}" / f"model_smiles_{ds}.pt"
    if family == "selfies":
        stage = f"reproduce-02-selfies-{lower}"
        return root / stage / f"results-{stage}" / f"model_selfies_{ds}.pt"
    if family == "graph":
        stage = f"reproduce-03-graph-{lower}"
        return root / stage / f"results-{stage}" / f"model_graph_{ds}.pt"
    if family == "fingerprint":
        stage = (
            "reproduce-04b-fingerprint-mlp-extract"
            if ds == "AR"
            else f"{lower}-reproduce-04b-fingerprint-mlp-extract"
        )
        return root / stage / f"results-{stage}" / f"model_fingerprint_{ds}.pt"
    raise ValueError(f"Unknown feature family: {family}")


def fingerprint_transformer_path(
    dataset: str, repo_root: Optional[Path] = None
) -> Path:
    if is_flat_kaggle_bundle(repo_root):
        return Path(repo_root) / "transformers" / (
            f"ecfp_transformer_{normalize_dataset(dataset)}.pkl"
        )
    return checkpoint_path(dataset, "fingerprint", repo_root).with_name(
        f"ecfp_transformer_{normalize_dataset(dataset)}.pkl"
    )


def scaler_path(dataset: str, repo_root: Optional[Path] = None) -> Path:
    ds = normalize_dataset(dataset)
    if is_flat_kaggle_bundle(repo_root):
        return Path(repo_root) / "models" / f"{ds}_normalizer_minmax.pkl"
    return hstack_result_dir(ds, repo_root) / "models" / f"{ds}_normalizer_minmax.pkl"


def ridge_path(dataset: str, repo_root: Optional[Path] = None) -> Path:
    ds = normalize_dataset(dataset)
    if is_flat_kaggle_bundle(repo_root):
        return Path(repo_root) / "models" / f"{ds}_Ridge.pkl"
    return hstack_result_dir(ds, repo_root) / "models" / f"{ds}_Ridge.pkl"


def raw_csv_path(dataset: str, repo_root: Optional[Path] = None) -> Path:
    ds = normalize_dataset(dataset)
    kaggle_path = KAGGLE_RAW_DATA_ROOT / DATASETS[ds].raw_csv_name
    if kaggle_path.is_file():
        return kaggle_path
    if is_flat_kaggle_bundle(repo_root):
        return Path(repo_root) / "raw" / DATASETS[ds].raw_csv_name
    return source_root(repo_root) / "AR_ER_GR_PR" / DATASETS[ds].raw_csv_name


def normalize_dataset(dataset: str) -> str:
    ds = dataset.upper()
    if ds == "ERALPHA":
        ds = "ER"
    if ds not in DATASETS:
        raise ValueError(f"Dataset must be one of {tuple(DATASETS)}, got {dataset!r}")
    return ds
