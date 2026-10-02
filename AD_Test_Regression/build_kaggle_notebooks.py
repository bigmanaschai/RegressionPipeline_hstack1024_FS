from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path


AD_ROOT = Path(__file__).resolve().parent
EXTENSION_ROOT = AD_ROOT.parent
TEMPLATE_ROOT = EXTENSION_ROOT / "kaggle_notebooks"
OUTPUT_ROOT = AD_ROOT / "kaggle_notebooks"
EXTRACTOR_SOURCE = (
    EXTENSION_ROOT
    / "materials/hstack1024-fs-extension/base_src/hstack1024_pipeline/extractors.py"
)
FROZEN_MODEL_ROOT = EXTENSION_ROOT / "materials/hstack1024-fs-extension/models"

DATASET_CONTRACTS = {
    "AR": {
        "raw_csv": "AR_regression.csv",
        "n_val": 386,
        "n_test": 387,
        "canonical_hashes": {
            "train": "9157a42bd7da0590",
            "val": "0c26403acd5ad627",
            "test": "45de4d5c02af07cf",
        },
        "canonical_smiles_hashes": {
            "train": "3f94b98eaf5385700121212b7021730e42a33a62835e4c797461be96006539d5",
            "val": "7c16a541e552f5b98826667fcc4f5d5ffef90de696ae7c22a483290753daad40",
            "test": "cdd75f3a2a0467568b6b343953671718e50b189282918199a81ed94fed2bba6c",
        },
        "canonical_paired_hashes": {
            "train": "7a463c40c615ff782874cec00261a0e7b34dd8bd6869b2b91b95fbdfb46a784f",
            "val": "ba22efa20aa1d39f2d50dd677e35d00973609760df510ca73e9cb12e821e110a",
            "test": "430201a28be3bb3aa6ef72f7175da2d2eb75d5fbba39acd31fcd924b32cb1a84",
        },
    },
    "ER": {
        "raw_csv": "ERalpha_regression.csv",
        "n_val": 505,
        "n_test": 505,
        "canonical_hashes": {
            "train": "4cfe4cf00d1ed620",
            "val": "d3755b70a1815d8e",
            "test": "da4fc0c0979e2788",
        },
        "canonical_smiles_hashes": {
            "train": "56efcf1f07d653965f3236a04d3fb5c6255bda9e58f28ff6b4f2fcc89f249fe0",
            "val": "d5c427347d86eae1079a59f981b7eba5d855581d5a90d34aed5bd613d0458557",
            "test": "ee8a8870dc935bb759bb73b1e18ecb21af3b6c867fbb5a82f8a3dfb1c86bbf47",
        },
        "canonical_paired_hashes": {
            "train": "a30b7fe1f561a2fca7c0191b8be498b9e5979904c0c19cbd30ee24129a54f317",
            "val": "f7a809e3e176434ba3352e31efa36f3976d7aae44e203b4ddb7293db31ef61c3",
            "test": "c7641fa54b7889bfa905ae9338bec32ddeb209d401e74fcc6673b7e40950b2da",
        },
    },
    "GR": {
        "raw_csv": "GR_regression.csv",
        "n_val": 327,
        "n_test": 328,
        "canonical_hashes": {},
        "canonical_smiles_hashes": {
            "train": "4b69da7f4fac25e9d251f38136022e44143975a0c20c6652e77e0498fb076492",
            "val": "0af3bab48d9bd5865eabc3a8fb9dc20d0412b9f0b6beb4e616ef7d88e1d6f251",
            "test": "e73fd2bd0b5fc60a27380396b42370c73b3083ac0ac54329b077c04b38e845f9",
        },
        "canonical_paired_hashes": {
            "train": "c7e78e1e6ba4e1a4be1a3411c5f07351d415c62d73d708b9dc9dc7a5fc462cf4",
            "val": "d38eb41bdcfffd6b49e6002e85f21ab51294eeced8fe089416de6037cab68435",
            "test": "5f4afef9c3e9270728a4543b52776b7ae901dd7db549bb7f7f128c662f623be0",
        },
    },
    "PR": {
        "raw_csv": "PR_regression.csv",
        "n_val": 271,
        "n_test": 272,
        "canonical_hashes": {},
        "canonical_smiles_hashes": {
            "train": "1db2fa90a7e84d011fbda3623e280eaae8bb61f246499a1c62f7c949bfd4fc1c",
            "val": "446ee3ed481e403f85294d9eb103b64341b4c02171ef2add1d90fbedea184be5",
            "test": "e8df6d492f25d1a36d6789960a3f0d513af02fd2c9e86f1c09b3bc9c9aa19d0d",
        },
        "canonical_paired_hashes": {
            "train": "788159e020d538c59a58b1cb5d9865e61628b2e5436641ea48c1d2de097a9e22",
            "val": "cf60570e6219799ea413e2fd2a23988a9a98bfcc1cc3ab38486ef9e4591f9128",
            "test": "10df9cb17ea5bd862c70580d20fab3eab05389a8581310f58aeeaef8ab366937",
        },
    },
}

VARIANTS = (
    ("01", "AR", "mutual_info", 30, "ElasticNet", 0.605790202425591),
    ("02", "AR", "pearson", 25, "Ridge", 0.621969056290390),
    ("03", "ER", "mutual_info", 35, "Ridge", 0.769493031166217),
    ("04", "ER", "pearson", 30, "ElasticNet", 0.768635542381417),
    ("05", "GR", "mutual_info", 30, "ElasticNet", 0.589149117475332),
    ("06", "GR", "pearson", 30, "Ridge", 0.519489428395004),
    ("07", "PR", "mutual_info", 25, "ElasticNet", 0.706200850113676),
    ("08", "PR", "pearson", 25, "Ridge", 0.705562571628553),
)


def _lines(text: str) -> list[str]:
    return text.strip("\n").splitlines(keepends=True)


def _markdown(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": _lines(text)}


def _code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": _lines(text),
    }


def _configuration(
    dataset: str,
    method: str,
    selected_k: int,
    model_class: str,
    reference_r2: float,
) -> str:
    contract = DATASET_CONTRACTS[dataset]
    bundle_name = (
        f"{dataset}_{method}_bundle.json"
        if (dataset, method) == ("AR", "pearson")
        else f"{dataset}_{method}_best_model.pkl"
    )
    return f'''# STEP 0 — Configuration. Edit paths here when auto-discovery is ambiguous.
RUN_DATASET = {dataset!r}
RUN_METHOD = {method!r}
EXPECTED_SELECTED_K = {selected_k}
EXPECTED_MODEL_CLASS = {model_class!r}
EXPECTED_REFERENCE_TEST_R2 = {reference_r2!r}
EXPECTED_N_VAL = {contract["n_val"]}
EXPECTED_N_TEST = {contract["n_test"]}
RAW_CSV_FILENAME = {contract["raw_csv"]!r}
FROZEN_BUNDLE_FILENAME = {bundle_name!r}

# AD controls. One summary/detail CSV is written per multiplier.
AD_THRESHOLD_MULTIPLIERS = (0.5, 2.0)
AD_NEIGHBOR_K_VALUES = tuple(range(3, 26))
R2_TOLERANCE = 0.001

# Leave a path blank to auto-discover the exact filename under INPUT_SEARCH_ROOTS.
# If Kaggle finds multiple copies, paste the desired full path here.
RAW_CSV_PATH = ""
SMILES_CHECKPOINT_PATH = ""
SELFIES_CHECKPOINT_PATH = ""
GRAPH_CHECKPOINT_PATH = ""
FINGERPRINT_CHECKPOINT_PATH = ""
FINGERPRINT_TRANSFORMER_PATH = ""

# Preferred: one monolithic *.pkl bundle or a component-bundle *.json descriptor.
FROZEN_BUNDLE_PATH = ""
# Alternative: leave FROZEN_BUNDLE_PATH blank and provide all three paths below.
FROZEN_MODEL_PATH = ""
FROZEN_SCALER_PATH = ""
FROZEN_SELECTOR_PATH = ""

# Safe fallback for the small variant-specific final artifacts. External paths
# take precedence; set False to require an attached Kaggle dataset instead.
USE_EMBEDDED_FROZEN_ARTIFACTS = True

INPUT_SEARCH_ROOTS = ["/kaggle/input"]
OUTPUT_ROOT = f"/kaggle/working/AD_Test_Regression_{{RUN_DATASET}}_{{RUN_METHOD}}"

# Kaggle dependency setup. Replace SKFP_INSTALL_SPEC with an attached wheel path
# if you do not want pip to fetch the pinned third-party fingerprint package.
AUTO_INSTALL_MISSING_DEPENDENCIES = True
SKFP_INSTALL_SPEC = "git+https://github.com/plenoi/scikit-finger-plenoi.git@master"

EXPECTED_CANONICAL_HASHES = {contract["canonical_hashes"]!r}
EXPECTED_CANONICAL_SMILES_HASHES = {contract["canonical_smiles_hashes"]!r}
EXPECTED_CANONICAL_PAIRED_HASHES = {contract["canonical_paired_hashes"]!r}

assert RUN_DATASET in {{"AR", "ER", "GR", "PR"}}
assert RUN_METHOD in {{"mutual_info", "pearson"}}
assert AD_THRESHOLD_MULTIPLIERS
assert tuple(AD_NEIGHBOR_K_VALUES) == tuple(range(3, 26))
print("Variant:", f"{{RUN_DATASET}} × {{RUN_METHOD}}")
print("Expected frozen model:", EXPECTED_MODEL_CLASS)
print("Expected selected features:", EXPECTED_SELECTED_K)
print("Output root:", OUTPUT_ROOT)
'''


def _embedded_frozen_cell(dataset: str, method: str) -> str:
    bundle_name = (
        f"{dataset}_{method}_bundle.json"
        if (dataset, method) == ("AR", "pearson")
        else f"{dataset}_{method}_best_model.pkl"
    )
    filenames = [bundle_name]
    bundle_path = FROZEN_MODEL_ROOT / bundle_name
    if not bundle_path.is_file():
        raise FileNotFoundError(bundle_path)
    if bundle_path.suffix.lower() == ".json":
        descriptor = json.loads(bundle_path.read_text(encoding="utf-8"))
        filenames.extend(descriptor["components"].values())

    payloads = {}
    digests = {}
    for filename in filenames:
        data = (FROZEN_MODEL_ROOT / filename).read_bytes()
        payloads[filename] = base64.b64encode(data).decode("ascii")
        digests[filename] = hashlib.sha256(data).hexdigest()

    return f'''# STEP 0C — Materialize the small frozen final artifacts when no Kaggle input contains them.
# These payloads are binary sklearn/joblib artifacts, not executable project source.
import base64
import hashlib
from pathlib import Path

EMBEDDED_FROZEN_FILES_B64 = {payloads!r}
EMBEDDED_FROZEN_SHA256 = {digests!r}
EMBEDDED_FROZEN_ROOT = (
    Path(OUTPUT_ROOT).expanduser().parent
    / "ad_embedded_frozen_inputs"
    / f"{{RUN_DATASET}}_{{RUN_METHOD}}"
)

def materialize_embedded_frozen_files():
    EMBEDDED_FROZEN_ROOT.mkdir(parents=True, exist_ok=True)
    resolved = {{}}
    for filename, encoded in EMBEDDED_FROZEN_FILES_B64.items():
        data = base64.b64decode(encoded.encode("ascii"))
        actual = hashlib.sha256(data).hexdigest()
        expected = EMBEDDED_FROZEN_SHA256[filename]
        if actual != expected:
            raise RuntimeError(f"Embedded frozen artifact checksum failed: {{filename}}")
        path = EMBEDDED_FROZEN_ROOT / filename
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            path.write_bytes(data)
        resolved[filename] = path.resolve()
    return resolved

EMBEDDED_FROZEN_PATHS = (
    materialize_embedded_frozen_files()
    if USE_EMBEDDED_FROZEN_ARTIFACTS
    else {{}}
)
print("Embedded frozen fallback:", "ready" if EMBEDDED_FROZEN_PATHS else "disabled")
print("Embedded frozen SHA-256:", EMBEDDED_FROZEN_SHA256)
'''


DEPENDENCY_SETUP = r'''# STEP 0B — Install only missing third-party inference dependencies.
import importlib.util
import os
import subprocess
import sys

os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("WANDB_SILENT", "true")

required_modules = {
    "transformers": "transformers==4.48.3",
    "sentencepiece": "sentencepiece",
    "selfies": "selfies",
    "rdkit": "rdkit",
    "deepchem": "deepchem",
    "torch_geometric": "torch-geometric",
    "skfp": SKFP_INSTALL_SPEC,
}
missing_specs = [
    package_spec
    for module_name, package_spec in required_modules.items()
    if importlib.util.find_spec(module_name) is None
]
if missing_specs:
    if not AUTO_INSTALL_MISSING_DEPENDENCIES:
        raise ModuleNotFoundError(
            "Missing dependencies: " + ", ".join(missing_specs)
            + ". Install them or enable AUTO_INSTALL_MISSING_DEPENDENCIES."
        )
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing_specs])

from transformers import AutoModel, AutoTokenizer

def pin_remote_code_revision(auto_class):
    original = auto_class.from_pretrained
    if getattr(original, "_hstack_revision_pinned", False):
        return
    def pinned(pretrained_model_name_or_path, *args, **kwargs):
        revision = kwargs.get("revision")
        if revision:
            kwargs.setdefault("code_revision", revision)
        return original(pretrained_model_name_or_path, *args, **kwargs)
    pinned._hstack_revision_pinned = True
    auto_class.from_pretrained = pinned

pin_remote_code_revision(AutoTokenizer)
pin_remote_code_revision(AutoModel)
print("Third-party dependencies: OK")
print("Inference device contract: CPU")
'''


PATH_RESOLUTION = r'''# STEP 1 — Imports, output directories, and transparent input-path resolution.
import ast
import hashlib
import json
import random
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from IPython.display import display

OUTPUT_DIR = Path(OUTPUT_ROOT) / RUN_DATASET / RUN_METHOD
ARTIFACT_DIR = OUTPUT_DIR / "intermediate_artifacts"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

def _configured_path(value):
    return Path(value).expanduser().resolve() if str(value).strip() else None

def discover_exact_file(label, configured, filenames, *, required=True):
    explicit = _configured_path(configured)
    if explicit is not None:
        if not explicit.is_file():
            raise FileNotFoundError(f"{label} path does not exist: {explicit}")
        return explicit
    matches = []
    for root_value in INPUT_SEARCH_ROOTS:
        root = Path(root_value).expanduser()
        if not root.is_dir():
            continue
        for filename in filenames:
            matches.extend(path.resolve() for path in root.rglob(filename) if path.is_file())
    matches = sorted(set(matches))
    if len(matches) == 1:
        return matches[0]
    if not matches and not required:
        return None
    if not matches:
        raise FileNotFoundError(
            f"Could not find {label}. Set its *_PATH value in STEP 0. "
            f"Expected one of: {list(filenames)}"
        )
    preview = "\n".join(f"  - {path}" for path in matches[:20])
    raise RuntimeError(
        f"Found multiple candidates for {label}; set the exact path in STEP 0:\n{preview}"
    )

INPUT_PATHS = {
    "raw_csv": discover_exact_file("raw CSV", RAW_CSV_PATH, [RAW_CSV_FILENAME]),
    "smiles_checkpoint": discover_exact_file(
        "SMILES checkpoint", SMILES_CHECKPOINT_PATH, [f"model_smiles_{RUN_DATASET}.pt"]
    ),
    "selfies_checkpoint": discover_exact_file(
        "SELFIES checkpoint", SELFIES_CHECKPOINT_PATH, [f"model_selfies_{RUN_DATASET}.pt"]
    ),
    "graph_checkpoint": discover_exact_file(
        "Graph checkpoint", GRAPH_CHECKPOINT_PATH, [f"model_graph_{RUN_DATASET}.pt"]
    ),
    "fingerprint_checkpoint": discover_exact_file(
        "Fingerprint checkpoint", FINGERPRINT_CHECKPOINT_PATH,
        [f"model_fingerprint_{RUN_DATASET}.pt"],
    ),
    "fingerprint_transformer": discover_exact_file(
        "Fingerprint transformer", FINGERPRINT_TRANSFORMER_PATH,
        [f"ecfp_transformer_{RUN_DATASET}.pkl"],
    ),
}

separate_frozen_paths = all(
    str(value).strip()
    for value in (FROZEN_MODEL_PATH, FROZEN_SCALER_PATH, FROZEN_SELECTOR_PATH)
)
if separate_frozen_paths:
    INPUT_PATHS["frozen_model"] = discover_exact_file(
        "frozen model", FROZEN_MODEL_PATH, [Path(FROZEN_MODEL_PATH).name]
    )
    INPUT_PATHS["frozen_scaler"] = discover_exact_file(
        "frozen scaler", FROZEN_SCALER_PATH, [Path(FROZEN_SCALER_PATH).name]
    )
    INPUT_PATHS["frozen_selector"] = discover_exact_file(
        "frozen selector", FROZEN_SELECTOR_PATH, [Path(FROZEN_SELECTOR_PATH).name]
    )
else:
    if any(str(value).strip() for value in (FROZEN_MODEL_PATH, FROZEN_SCALER_PATH, FROZEN_SELECTOR_PATH)):
        raise ValueError("Provide all three separate frozen component paths, or leave all three blank")
    discovered_bundle = discover_exact_file(
        "frozen bundle", FROZEN_BUNDLE_PATH, [FROZEN_BUNDLE_FILENAME], required=False
    )
    if discovered_bundle is not None:
        INPUT_PATHS["frozen_bundle"] = discovered_bundle
        FROZEN_INPUT_SOURCE = "external Kaggle input"
    elif USE_EMBEDDED_FROZEN_ARTIFACTS:
        INPUT_PATHS["frozen_bundle"] = EMBEDDED_FROZEN_PATHS[FROZEN_BUNDLE_FILENAME]
        FROZEN_INPUT_SOURCE = "embedded checksum-verified fallback"
    else:
        raise FileNotFoundError(
            f"Could not find frozen bundle {FROZEN_BUNDLE_FILENAME!r}. "
            "Set FROZEN_BUNDLE_PATH or enable USE_EMBEDDED_FROZEN_ARTIFACTS."
        )

path_table = pd.DataFrame(
    [{"Input": key, "Resolved path": str(value)} for key, value in INPUT_PATHS.items()]
)
display(path_table)
print("Frozen final-artifact source:", FROZEN_INPUT_SOURCE if not separate_frozen_paths else "three explicit paths")
print("Output directory:", OUTPUT_DIR)
print("Intermediate artifact directory:", ARTIFACT_DIR)
'''


FROZEN_LOADING = r'''# STEP 2 — Load the frozen model, MinMaxScaler, and feature selector.
def _corr_abs(A, b, method):
    if method == "spearman":
        from scipy.stats import rankdata
        A = np.apply_along_axis(rankdata, 0, A)
        b = rankdata(b)
    centered_y = b - b.mean()
    centered_X = A - A.mean(0)
    numerator = (centered_X * centered_y[:, None]).sum(0)
    denominator = np.sqrt((centered_X**2).sum(0)) * np.sqrt((centered_y**2).sum())
    return np.abs(numerator / (denominator + 1e-12))

sys.modules["__main__"]._corr_abs = _corr_abs

def load_frozen_objects():
    if separate_frozen_paths:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = joblib.load(INPUT_PATHS["frozen_model"])
            scaler = joblib.load(INPUT_PATHS["frozen_scaler"])
            selector = joblib.load(INPUT_PATHS["frozen_selector"])
        metadata = {
            "dataset": RUN_DATASET, "FS_Method": RUN_METHOD,
            "FS_k": EXPECTED_SELECTED_K, "model_name": EXPECTED_MODEL_CLASS,
            "TSR2": EXPECTED_REFERENCE_TEST_R2,
        }
        return model, scaler, selector, metadata

    bundle_path = INPUT_PATHS["frozen_bundle"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if bundle_path.suffix.lower() == ".json":
            descriptor = json.loads(bundle_path.read_text(encoding="utf-8"))
            if descriptor.get("format") != "frozen_component_bundle_v1":
                raise AssertionError(f"Unsupported frozen descriptor: {bundle_path}")
            components = descriptor["components"]
            model = joblib.load(bundle_path.parent / components["model"])
            scaler = joblib.load(bundle_path.parent / components["normalizer"])
            selector = joblib.load(bundle_path.parent / components["selector"])
            metadata = dict(descriptor["metadata"])
        else:
            payload = joblib.load(bundle_path)
            model = payload["model"]
            scaler = payload["normalizer"]
            selector = payload["selector"]
            metadata = {key: value for key, value in payload.items() if key not in {"model", "normalizer", "selector"}}
    return model, scaler, selector, metadata

frozen_model, frozen_scaler, frozen_selector, frozen_metadata = load_frozen_objects()
selected_indices = np.asarray(frozen_selector.get_support(indices=True), dtype=int)

checks = {
    "model class": frozen_model.__class__.__name__ == EXPECTED_MODEL_CLASS,
    "selected feature count": len(selected_indices) == EXPECTED_SELECTED_K,
    "scaler input dimension": int(frozen_scaler.n_features_in_) == 1024,
    "selector input dimension": int(frozen_selector.n_features_in_) == 1024,
    "model input dimension": int(frozen_model.n_features_in_) == EXPECTED_SELECTED_K,
}
failed = [name for name, passed in checks.items() if not passed]
if failed:
    raise AssertionError(f"Frozen artifact contract failed: {failed}")

display(pd.DataFrame([{"Check": key, "Passed": value} for key, value in checks.items()]))
print("Model:", frozen_model)
print("Scaler:", frozen_scaler.__class__.__name__)
print("Selector:", frozen_selector.__class__.__name__)
print("Selected HStack1024 indices:", selected_indices.tolist())
'''


RAW_SPLIT = r'''# STEP 3 — Clean the raw receptor CSV and recreate the locked 60/20/20 split.
from rdkit import Chem, RDLogger
from sklearn.model_selection import train_test_split

RDLogger.DisableLog("rdApp.*")

def unordered_canonical_smiles_hash(values):
    """Hash canonical molecule membership without depending on row order."""
    ordered_values = sorted(str(value) for value in values)
    return hashlib.sha256("\n".join(ordered_values).encode("utf-8")).hexdigest()

def unordered_canonical_smiles_target_hash(smiles_values, target_values):
    """Hash canonical-SMILES/target pairs while ignoring only row order."""
    if len(smiles_values) != len(target_values):
        raise AssertionError("SMILES/target length mismatch")
    records = sorted(
        (str(smiles), np.float64(target).hex())
        for smiles, target in zip(smiles_values, target_values)
    )
    payload = "\n".join(f"{smiles}\t{target_hex}" for smiles, target_hex in records)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()

def canonical_split_hash(canonical_smiles):
    ordered_values = sorted(str(value) for value in canonical_smiles)
    return hashlib.md5("\n".join(ordered_values).encode("utf-8")).hexdigest()[:16]

raw_frame = pd.read_csv(INPUT_PATHS["raw_csv"])
required_columns = {"Smiles", "pIC50"}
missing_columns = required_columns.difference(raw_frame.columns)
if missing_columns:
    raise ValueError(f"Raw CSV is missing columns: {sorted(missing_columns)}")

valid_smiles_mask = raw_frame["Smiles"].apply(
    lambda value: Chem.MolFromSmiles(str(value)) is not None
)
clean_frame = raw_frame.loc[valid_smiles_mask].drop_duplicates(subset=["Smiles"]).copy()
clean_frame = clean_frame.loc[~np.isinf(clean_frame["pIC50"].to_numpy())].copy()
smiles = clean_frame["Smiles"].to_numpy().astype(str)
targets = clean_frame["pIC50"].to_numpy(dtype=float)
canonical_smiles = np.asarray([
    Chem.MolToSmiles(Chem.MolFromSmiles(value)) for value in smiles
]).astype(str)
canonical_by_raw_smiles = dict(zip(smiles.tolist(), canonical_smiles.tolist()))

bins = pd.qcut(targets, q=10, labels=False, duplicates="drop")
train_smiles, temp_smiles, train_y, temp_y, _, temp_bins = train_test_split(
    smiles, targets, bins, test_size=0.40, random_state=0, stratify=bins
)
val_smiles, test_smiles, val_y, test_y = train_test_split(
    temp_smiles, temp_y, test_size=0.50, random_state=0, stratify=temp_bins
)
splits = {
    "train": {"smiles": np.asarray(train_smiles).astype(str), "y": np.asarray(train_y, dtype=float)},
    "val": {"smiles": np.asarray(val_smiles).astype(str), "y": np.asarray(val_y, dtype=float)},
    "test": {"smiles": np.asarray(test_smiles).astype(str), "y": np.asarray(test_y, dtype=float)},
}
for payload in splits.values():
    payload["canonical_smiles"] = np.asarray([
        canonical_by_raw_smiles[value] for value in payload["smiles"]
    ]).astype(str)

if len(val_y) != EXPECTED_N_VAL or len(test_y) != EXPECTED_N_TEST:
    raise AssertionError(
        f"Split size changed: val={len(val_y)}, test={len(test_y)}; "
        f"expected val={EXPECTED_N_VAL}, test={EXPECTED_N_TEST}"
    )

# train_test_split can return the same stratified membership in a different row
# order across sklearn/NumPy versions. Validate canonical molecule membership
# and every canonical-SMILES/target pairing, but do not require a row order or
# one particular equivalent textual representation of a SMILES molecule.
split_smiles_sets = {
    split_name: set(payload["canonical_smiles"].tolist())
    for split_name, payload in splits.items()
}
for split_name, payload in splits.items():
    if len(split_smiles_sets[split_name]) != len(payload["canonical_smiles"]):
        raise AssertionError(f"{split_name} contains duplicate canonical molecules")
if split_smiles_sets["train"] & split_smiles_sets["val"]:
    raise AssertionError("Train/validation SMILES leakage detected")
if split_smiles_sets["train"] & split_smiles_sets["test"]:
    raise AssertionError("Train/test SMILES leakage detected")
if split_smiles_sets["val"] & split_smiles_sets["test"]:
    raise AssertionError("Validation/test SMILES leakage detected")
if set().union(*split_smiles_sets.values()) != set(canonical_smiles.tolist()):
    raise AssertionError("Split union does not reproduce the cleaned dataset")

for split_name, expected in EXPECTED_CANONICAL_HASHES.items():
    actual = canonical_split_hash(splits[split_name]["canonical_smiles"])
    if actual != expected:
        raise AssertionError(f"{split_name} canonical hash changed: {actual} != {expected}")

split_identity_audit = []
for split_name in ("train", "val", "test"):
    payload = splits[split_name]
    actual_smiles_hash = unordered_canonical_smiles_hash(payload["canonical_smiles"])
    actual_pair_hash = unordered_canonical_smiles_target_hash(
        payload["canonical_smiles"], payload["y"]
    )
    if actual_smiles_hash != EXPECTED_CANONICAL_SMILES_HASHES[split_name]:
        raise AssertionError(
            f"{split_name} canonical molecule membership changed: "
            f"{actual_smiles_hash} != {EXPECTED_CANONICAL_SMILES_HASHES[split_name]}"
        )
    if actual_pair_hash != EXPECTED_CANONICAL_PAIRED_HASHES[split_name]:
        raise AssertionError(
            f"{split_name} canonical-SMILES/target pairing changed: "
            f"{actual_pair_hash} != {EXPECTED_CANONICAL_PAIRED_HASHES[split_name]}"
        )
    split_identity_audit.append({
        "Split": split_name,
        "Rows": len(payload["y"]),
        "Unique_Canonical_Molecules": len(split_smiles_sets[split_name]),
        "Canonical_membership_hash": actual_smiles_hash,
        "Canonical_SMILES_target_pair_hash": actual_pair_hash,
    })

raw_split_path = ARTIFACT_DIR / "01_raw_splits.npz"
np.savez_compressed(
    raw_split_path,
    train_smiles=splits["train"]["smiles"], train_y=splits["train"]["y"],
    val_smiles=splits["val"]["smiles"], val_y=splits["val"]["y"],
    test_smiles=splits["test"]["smiles"], test_y=splits["test"]["y"],
    train_canonical_smiles=splits["train"]["canonical_smiles"],
    val_canonical_smiles=splits["val"]["canonical_smiles"],
    test_canonical_smiles=splits["test"]["canonical_smiles"],
)

print("Raw rows:", len(raw_frame))
print("Clean rows:", len(clean_frame))
print("Split sizes:", {name: len(payload["y"]) for name, payload in splits.items()})
print("Split identity verified without requiring version-specific row order.")
display(pd.DataFrame(split_identity_audit))
print("Saved:", raw_split_path)
display(pd.DataFrame({"test_smiles": test_smiles[:5], "test_pIC50": test_y[:5]}))
'''


def _standalone_extractor_source() -> str:
    source = EXTRACTOR_SOURCE.read_text(encoding="utf-8")
    config_import = '''from .config import (
    FAMILY_DIM,
    checkpoint_path,
    fingerprint_transformer_path,
    normalize_dataset,
)
from .io_contract import FeatureSplit, load_joblib
'''
    replacement = '''FAMILY_DIM = 256

def normalize_dataset(dataset):
    value = str(dataset).upper()
    return "ER" if value == "ERALPHA" else value

def checkpoint_path(dataset, family, repo_root=None):
    return INPUT_PATHS[f"{family}_checkpoint"]

def fingerprint_transformer_path(dataset, repo_root=None):
    return INPUT_PATHS["fingerprint_transformer"]

def load_joblib(path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return joblib.load(path)

@dataclass(frozen=True)
class FeatureSplit:
    smiles: np.ndarray
    X: np.ndarray
    y: np.ndarray
'''
    if config_import not in source:
        raise RuntimeError("Could not adapt the approved extractor source")
    source = source.replace(config_import, replacement)
    source = source.replace("from __future__ import annotations\n", "")
    source = source.replace(
        '"""Frozen feature extraction copied from the approved research notebooks.\n\n'
        "This module contains inference only.  It has no optimizer, loss, fit, or training\n"
        "loop. Heavy dependencies are imported lazily and every representation is\n"
        "computed from the Validation/Test splits recreated from the raw CSV. Attached\n"
        "reproduce-00 outputs are validation oracles and never feature inputs.\n"
        '"""\n\n',
        "",
    )
    return (
        "# STEP 4 — Complete frozen feature-extractor definitions.\n"
        "# This source is embedded in the notebook for direct inspection; it is not imported from GitHub.\n"
        "import warnings\nimport joblib\n\n"
        + source
    )


FEATURE_PREP = r'''# STEP 5 — Prepare one aligned Train+Validation+Test sequence for fresh extraction.
SPLIT_ORDER = ("train", "val", "test")
split_bounds = {}
all_smiles_parts = []
all_y_parts = []
start = 0
for split_name in SPLIT_ORDER:
    stop = start + len(splits[split_name]["y"])
    split_bounds[split_name] = (start, stop)
    all_smiles_parts.append(splits[split_name]["smiles"])
    all_y_parts.append(splits[split_name]["y"])
    start = stop

all_smiles = np.concatenate(all_smiles_parts).astype(str)
all_y = np.concatenate(all_y_parts).astype(float)
family_features = {}

def split_feature_matrix(matrix):
    return {
        split_name: np.asarray(matrix[left:right], dtype=np.float32)
        for split_name, (left, right) in split_bounds.items()
    }

def save_family_features(family):
    output_path = ARTIFACT_DIR / f"02_{family}_features_256.npz"
    np.savez_compressed(output_path, **family_features[family])
    print(f"{family}: shapes =", {key: value.shape for key, value in family_features[family].items()})
    print(f"{family}: test[0, :5] =", family_features[family]["test"][0, :5])
    print("Saved:", output_path)

print("Combined extraction rows:", len(all_smiles))
print("Boundaries:", split_bounds)
'''


def _extract_family_cell(family: str) -> str:
    function = {
        "smiles": "extract_smiles",
        "selfies": "extract_selfies",
        "graph": "extract_graph",
        "fingerprint": "extract_fingerprint",
    }[family]
    step = {"smiles": "6A", "selfies": "6B", "graph": "6C", "fingerprint": "6D"}[family]
    return f'''# STEP {step} — Extract fresh {family.upper()} features (256 dimensions) and save them.
{family}_all = {function}(RUN_DATASET, all_smiles)
family_features[{family!r}] = split_feature_matrix({family}_all)
save_family_features({family!r})
'''


HSTACK = r'''# STEP 7 — Concatenate four 256-dimensional families into HStack1024 and save it.
FAMILY_ORDER = ("smiles", "selfies", "graph", "fingerprint")
hstack = {}
for split_name in SPLIT_ORDER:
    matrices = [family_features[family][split_name] for family in FAMILY_ORDER]
    row_counts = {matrix.shape[0] for matrix in matrices}
    if row_counts != {len(splits[split_name]["y"])}:
        raise AssertionError(f"{split_name}: family row alignment failed")
    hstack[split_name] = np.hstack(matrices).astype(np.float32)
    expected_shape = (len(splits[split_name]["y"]), 1024)
    if hstack[split_name].shape != expected_shape:
        raise AssertionError(f"{split_name}: expected {expected_shape}, got {hstack[split_name].shape}")

hstack_path = ARTIFACT_DIR / "03_hstack1024.npz"
np.savez_compressed(hstack_path, **hstack)
print("HStack shapes:", {key: value.shape for key, value in hstack.items()})
print("Test HStack first five values:", hstack["test"][0, :5])
print("Saved:", hstack_path)
'''


SCALING = r'''# STEP 8 — Apply the frozen MinMaxScaler without fitting and save scaled HStack1024.
def apply_frozen_minmax(X, scaler):
    transformed = np.asarray(X) * np.asarray(scaler.scale_) + np.asarray(scaler.min_)
    return transformed.astype(np.float32)

scaled = {
    split_name: apply_frozen_minmax(hstack[split_name], frozen_scaler)
    for split_name in SPLIT_ORDER
}
scaled_path = ARTIFACT_DIR / "04_minmax_scaled_hstack1024.npz"
np.savez_compressed(scaled_path, **scaled)
print("Scaled shapes:", {key: value.shape for key, value in scaled.items()})
print("Test scaled range:", float(scaled["test"].min()), "to", float(scaled["test"].max()))
print("Test scaled first five values:", scaled["test"][0, :5])
print("Saved:", scaled_path)
'''


SELECTION = r'''# STEP 9 — Apply the frozen feature selector without fitting and save selected matrices.
selected = {
    split_name: np.asarray(scaled[split_name])[:, selected_indices].astype(np.float32)
    for split_name in SPLIT_ORDER
}
for split_name, matrix in selected.items():
    expected_shape = (len(splits[split_name]["y"]), EXPECTED_SELECTED_K)
    if matrix.shape != expected_shape:
        raise AssertionError(f"{split_name}: expected selected shape {expected_shape}, got {matrix.shape}")

selected_path = ARTIFACT_DIR / f"05_selected_features_k{EXPECTED_SELECTED_K}.npz"
np.savez_compressed(
    selected_path,
    selected_indices=selected_indices,
    train=selected["train"], val=selected["val"], test=selected["test"],
)
print("Selected HStack1024 indices:", selected_indices.tolist())
print("Selected shapes:", {key: value.shape for key, value in selected.items()})
print("Test selected first five values:", selected["test"][0, :5])
print("Saved:", selected_path)
'''


PREDICTION = r'''# STEP 10 — Predict the locked Test split with the frozen final regressor and save results.
def predict_frozen_linear(X, model):
    features = np.asarray(X, dtype=np.float64)
    coefficients = np.asarray(model.coef_, dtype=np.float64).reshape(1, -1)
    intercept = np.asarray(model.intercept_, dtype=np.float64)
    return (np.sum(features * coefficients, axis=1) + intercept).reshape(-1)

y_pred = predict_frozen_linear(selected["test"], frozen_model)
test_predictions = pd.DataFrame(
    {
        "Test_Row": np.arange(len(test_y), dtype=int),
        "Smiles": test_smiles,
        "y_true_pIC50": test_y,
        "y_pred_pIC50": y_pred,
        "Residual_Pred_minus_Obs": y_pred - test_y,
    }
)
predictions_path = ARTIFACT_DIR / "06_test_predictions.csv"
test_predictions.to_csv(predictions_path, index=False)
print("Prediction count:", len(y_pred))
print("Prediction range:", float(y_pred.min()), "to", float(y_pred.max()))
print("Saved:", predictions_path)
display(test_predictions.head())
'''


AD_DEFINITIONS = r'''# STEP 11 — Regression metrics and transparent kNN applicability-domain functions.
from scipy.stats import pearsonr, spearmanr
from sklearn.neighbors import NearestNeighbors

def regression_metrics(y_true, y_prediction):
    observed = np.asarray(y_true, dtype=np.float64).reshape(-1)
    predicted = np.asarray(y_prediction, dtype=np.float64).reshape(-1)
    residual = predicted - observed
    ss_res = float(np.sum(residual * residual))
    centered = observed - float(np.mean(observed))
    ss_tot = float(np.sum(centered * centered))
    r2 = float("nan") if len(observed) < 2 or ss_tot == 0.0 else 1.0 - ss_res / ss_tot
    pearson = (
        float("nan") if len(observed) < 2 or np.std(observed) == 0.0 or np.std(predicted) == 0.0
        else float(pearsonr(observed, predicted)[0])
    )
    spearman = (
        float("nan") if len(observed) < 2 or np.std(observed) == 0.0 or np.std(predicted) == 0.0
        else float(spearmanr(observed, predicted)[0])
    )
    return {
        "R2": r2,
        "RMSE": float(np.sqrt(np.mean(residual * residual))),
        "MAE": float(np.mean(np.abs(residual))),
        "ME": float(np.mean(residual)),
        "Pearson": pearson,
        "Spearman": spearman,
    }

def threshold_token(multiplier):
    text = np.format_float_positional(float(multiplier), trim="-")
    if "." not in text:
        text += ".0"
    return text.replace("-", "m").replace(".", "p")

def summary_row(label, multiplier, threshold, mask):
    metrics = regression_metrics(test_y[mask], y_pred[mask])
    n_ind = int(mask.sum())
    return {
        "K": label,
        "Threshold_Multiplier": float(multiplier),
        "AD_Distance_Threshold": float(threshold),
        **metrics,
        "N_Test": int(len(test_y)),
        "INDs": n_ind,
        "Coverage": float(n_ind / len(test_y)),
        "OODs": int(len(test_y) - n_ind),
    }

print("AD functions ready")
'''


AD_EXECUTION = r'''# STEP 12 — Build Train+Validation AD reference, evaluate No AD and k=3...25, save all outputs.
X_reference = np.vstack([selected["train"], selected["val"]]).astype(np.float32)
X_test_ad = selected["test"].astype(np.float32)
reference_path = ARTIFACT_DIR / "07_ad_reference_train_plus_validation.npy"
np.save(reference_path, X_reference)

no_ad_metrics = regression_metrics(test_y, y_pred)
if abs(float(no_ad_metrics["R2"]) - EXPECTED_REFERENCE_TEST_R2) > R2_TOLERANCE:
    raise AssertionError(
        f"No-AD Test R2={no_ad_metrics['R2']:.12f}, "
        f"reference={EXPECTED_REFERENCE_TEST_R2:.12f}, tolerance={R2_TOLERANCE}"
    )

rows_by_multiplier = {float(value): [] for value in AD_THRESHOLD_MULTIPLIERS}
details_by_multiplier = {
    float(value): test_predictions.copy() for value in AD_THRESHOLD_MULTIPLIERS
}
all_test_mask = np.ones(len(test_y), dtype=bool)
for multiplier in rows_by_multiplier:
    rows_by_multiplier[multiplier].append(
        summary_row("No AD", multiplier, float("nan"), all_test_mask)
    )

for ad_k in AD_NEIGHBOR_K_VALUES:
    knn = NearestNeighbors(n_neighbors=ad_k, metric="euclidean", algorithm="auto")
    knn.fit(X_reference)
    reference_distances, _ = knn.kneighbors(X_reference)
    test_distances, _ = knn.kneighbors(X_test_ad)
    reference_mean_distance = reference_distances.mean(axis=1)
    test_mean_distance = test_distances.mean(axis=1)
    center = float(reference_mean_distance.mean())
    spread = float(reference_mean_distance.std(ddof=0))
    for multiplier in sorted(rows_by_multiplier):
        threshold = center + multiplier * spread
        ind_mask = np.asarray(test_mean_distance <= threshold, dtype=bool)
        if not ind_mask.any():
            raise ValueError(f"No IND Test rows remain for k={ad_k}, multiplier={multiplier}")
        rows_by_multiplier[multiplier].append(
            summary_row(f"k={ad_k}", multiplier, threshold, ind_mask)
        )
        detail = details_by_multiplier[multiplier]
        detail[f"k{ad_k}_Mean_Distance"] = test_mean_distance
        detail[f"k{ad_k}_Threshold"] = threshold
        detail[f"k{ad_k}_IND"] = ind_mask

summary_paths = {}
detail_paths = {}
for multiplier, rows in rows_by_multiplier.items():
    token = threshold_token(multiplier)
    summary = pd.DataFrame(rows)
    summary_path = OUTPUT_DIR / (
        f"AD_Test_Regression_{RUN_DATASET}_{RUN_METHOD}_threshold_{token}.csv"
    )
    detail_path = ARTIFACT_DIR / f"08_test_ad_pointwise_threshold_{token}.csv"
    summary.to_csv(summary_path, index=False)
    details_by_multiplier[multiplier].to_csv(detail_path, index=False)
    summary_paths[multiplier] = summary_path
    detail_paths[multiplier] = detail_path
    print(f"\nThreshold multiplier={multiplier}")
    print("Summary saved:", summary_path)
    print("Pointwise AD saved:", detail_path)
    display(summary)

print("AD reference shape:", X_reference.shape)
print("Test AD query shape:", X_test_ad.shape)
print("AD reference saved:", reference_path)
'''


MANIFEST = r'''# STEP 13 — Write reproducibility metadata and an SHA-256 artifact manifest.
def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

metadata = {
    "dataset": RUN_DATASET,
    "method": RUN_METHOD,
    "raw_csv": str(INPUT_PATHS["raw_csv"]),
    "input_paths": {key: str(value) for key, value in INPUT_PATHS.items()},
    "split_rows": {key: int(len(value["y"])) for key, value in splits.items()},
    "feature_family_order": list(FAMILY_ORDER),
    "family_dimension": 256,
    "hstack_dimension": 1024,
    "selected_feature_count": int(EXPECTED_SELECTED_K),
    "selected_indices": selected_indices.tolist(),
    "model_class": frozen_model.__class__.__name__,
    "ad_reference": "Train + Validation selected features",
    "ad_query": "locked Test selected features",
    "ad_k_values": list(AD_NEIGHBOR_K_VALUES),
    "threshold_multipliers": list(AD_THRESHOLD_MULTIPLIERS),
    "threshold_formula": "mean(reference_mean_distance) + multiplier * std(reference_mean_distance, ddof=0)",
    "no_ad_metrics": no_ad_metrics,
}
metadata_path = OUTPUT_DIR / "run_metadata.json"
metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

artifact_paths = sorted(
    [path for path in OUTPUT_DIR.rglob("*") if path.is_file() and path.name != "artifact_manifest.csv"]
)
manifest_frame = pd.DataFrame(
    [
        {
            "Relative_Path": str(path.relative_to(OUTPUT_DIR)),
            "Bytes": path.stat().st_size,
            "SHA256": sha256_file(path),
        }
        for path in artifact_paths
    ]
)
manifest_path = OUTPUT_DIR / "artifact_manifest.csv"
manifest_frame.to_csv(manifest_path, index=False)

print("Metadata saved:", metadata_path)
print("Artifact manifest saved:", manifest_path)
print("Total saved files:", len(manifest_frame) + 1)
display(manifest_frame)
'''


def build_notebook(
    sequence: str,
    dataset: str,
    method: str,
    selected_k: int,
    model_class: str,
    reference_r2: float,
) -> Path:
    template_path = TEMPLATE_ROOT / f"{sequence}_run_{dataset}_{method}.ipynb"
    if not template_path.is_file():
        raise FileNotFoundError(template_path)
    template = json.loads(template_path.read_text(encoding="utf-8"))
    title = f"{dataset} × {method} — transparent Test regression AD"
    cells = [
        _markdown(
            f"# {title}\n\n"
            "This is a self-contained, cell-by-cell inference and applicability-domain "
            "notebook. It does not clone GitHub and does not import the project AD runtime. "
            "Every computational stage is visible, prints a compact audit, and saves a "
            "reloadable artifact. Frozen inputs are loaded only; nothing is fitted or trained."
        ),
        _markdown(
            "## Data lineage\n\n"
            "Raw receptor CSV → locked Train/Validation/Test → four fresh 256-D feature "
            "families → HStack1024 → frozen MinMaxScaler → frozen selector → frozen "
            "regressor + kNN AD.\n\n"
            "`No AD` uses every Test row. Each `k=` row reports regression metrics only "
            "for Test rows classified as IND. The Case-study dataset is not used here."
        ),
        _code(_configuration(dataset, method, selected_k, model_class, reference_r2)),
        _code(DEPENDENCY_SETUP),
        _code(_embedded_frozen_cell(dataset, method)),
        _code(PATH_RESOLUTION),
        _code(FROZEN_LOADING),
        _code(RAW_SPLIT),
        _markdown(
            "## Fresh 4-family feature extraction\n\n"
            "The complete frozen inference implementation is embedded in the following "
            "cell so the model architectures, pooling, checkpoints, and alignment rules "
            "can be inspected directly."
        ),
        _code(_standalone_extractor_source()),
        _code(FEATURE_PREP),
        _code(_extract_family_cell("smiles")),
        _code(_extract_family_cell("selfies")),
        _code(_extract_family_cell("graph")),
        _code(_extract_family_cell("fingerprint")),
        _code(HSTACK),
        _code(SCALING),
        _code(SELECTION),
        _code(PREDICTION),
        _code(AD_DEFINITIONS),
        _code(AD_EXECUTION),
        _code(MANIFEST),
        _markdown(
            "## Saved results\n\n"
            "The two original threshold summary CSVs remain at the variant output root. "
            "Reloadable split, feature, HStack, scaled, selected, prediction, AD-reference, "
            "and pointwise IND/OOD artifacts are under `intermediate_artifacts/`. "
            "`artifact_manifest.csv` records every saved file and SHA-256 digest."
        ),
    ]
    notebook = {
        "cells": cells,
        "metadata": template.get("metadata", {}),
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    output_path = OUTPUT_ROOT / f"{sequence}_run_{dataset}_{method}_AD.ipynb"
    output_path.write_text(
        json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return output_path


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    paths = [build_notebook(*variant) for variant in VARIANTS]
    if len(paths) != 8 or len(set(paths)) != 8:
        raise AssertionError("Expected exactly eight unique AD notebooks")
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
