"""Frozen feature extraction copied from the approved research notebooks.

This module contains inference only.  It has no optimizer, loss, fit, or training
loop. Heavy dependencies are imported lazily and every representation is
computed from the Validation/Test splits recreated from the raw CSV. Attached
reproduce-00 outputs are validation oracles and never feature inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional, Sequence

import numpy as np

from .config import (
    FAMILY_DIM,
    checkpoint_path,
    fingerprint_transformer_path,
    normalize_dataset,
)
from .io_contract import FeatureSplit, load_joblib


MOLFORMER_REPO = "ibm/MoLFormer-XL-both-10pct"
MOLFORMER_REVISION = "7b12d946c181a37f6012b9dc3b002275de070314"
BIOT5_REPO = "QizhiPei/biot5-plus-base"
BIOT5_REVISION = "622b6a04bf0fe266c67dd0927374b00bf21eb7de"
MAX_LENGTH = 128
BATCH_SIZE = 32
INFERENCE_DEVICE = "cpu"


def _seed_cpu() -> None:
    import random
    import torch

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _device():
    import torch

    # The research-replay contract is CPU-only. Never select CUDA implicitly:
    # different device kernels are outside the locked inference environment.
    return torch.device(INFERENCE_DEVICE)


def _require_256(name: str, features: np.ndarray, n_rows: int) -> np.ndarray:
    features = np.asarray(features, dtype=np.float32)
    if features.shape != (n_rows, FAMILY_DIM):
        raise AssertionError(
            f"{name}: expected {(n_rows, FAMILY_DIM)}, got {features.shape}"
        )
    if not np.isfinite(features).all():
        raise AssertionError(f"{name}: non-finite feature value")
    return features


def _transformer_dataset(sequences: Sequence[str], tokenizer):
    import torch
    from torch.utils.data import Dataset

    class SequenceDataset(Dataset):
        def __len__(self):
            return len(sequences)

        def __getitem__(self, index):
            encoded = tokenizer(
                str(sequences[index]),
                padding="max_length",
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            )
            return {key: value.squeeze(0) for key, value in encoded.items()}

    return SequenceDataset()


def _molformer_regressor(state_dict, dataset: str):
    import torch.nn as nn

    first_dim = int(state_dict["regressor.0.weight"].shape[0])
    dropout = 0.25 if normalize_dataset(dataset) == "ER" else 0.2
    return nn.Sequential(
        nn.Linear(768, first_dim),
        nn.LayerNorm(first_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(first_dim, 256),
        nn.LayerNorm(256),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(256, 1),
    )


def extract_smiles(
    dataset: str, smiles: Sequence[str], repo_root: Optional[Path] = None
) -> np.ndarray:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from transformers import AutoConfig, AutoModel, AutoTokenizer

    _seed_cpu()
    device = _device()
    state = torch.load(
        checkpoint_path(dataset, "smiles", repo_root),
        map_location="cpu",
        weights_only=True,
    )
    hub_args = {
        "trust_remote_code": True,
        "revision": MOLFORMER_REVISION,
        # Some repository auto_map entries explicitly name the source repo.
        # Without code_revision, Transformers can load remote Python from the
        # current HEAD while loading weights/config.json from revision, creating
        # two distinct MolformerConfig classes and a registration failure.
        "code_revision": MOLFORMER_REVISION,
    }
    config = AutoConfig.from_pretrained(
        MOLFORMER_REPO, deterministic_eval=True, **hub_args
    )
    tokenizer = AutoTokenizer.from_pretrained(MOLFORMER_REPO, **hub_args)
    base = AutoModel.from_pretrained(MOLFORMER_REPO, config=config, **hub_args)

    class FrozenMolFormer(nn.Module):
        def __init__(self):
            super().__init__()
            self.transformer = base
            self.regressor = _molformer_regressor(state, dataset)

    model = FrozenMolFormer()
    model.load_state_dict(state)
    model.to(device).eval()
    flags = [module.deterministic for module in model.modules() if hasattr(module, "deterministic")]
    if not flags or not all(flags):
        raise AssertionError("MoLFormer deterministic_eval is not active")
    loader = DataLoader(_transformer_dataset(smiles, tokenizer), batch_size=BATCH_SIZE)
    chunks = []
    with torch.no_grad():
        for batch in loader:
            output = model.transformer(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            )
            # Historical protocol: unmasked mean across all 128 token positions.
            pooled = output.last_hidden_state.mean(dim=1)
            chunks.append(model.regressor[:8](pooled).cpu().numpy())
    return _require_256("smiles", np.vstack(chunks), len(smiles))


def extract_selfies(
    dataset: str, smiles: Sequence[str], repo_root: Optional[Path] = None
) -> np.ndarray:
    import selfies as sf
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from transformers import AutoConfig, AutoModel, AutoTokenizer

    _seed_cpu()
    device = _device()
    sequences = []
    for index, value in enumerate(smiles):
        try:
            sequences.append(sf.encoder(str(value)))
        except Exception as exc:
            raise ValueError(f"SELFIES conversion failed at row {index}: {value!r}") from exc
    state = torch.load(
        checkpoint_path(dataset, "selfies", repo_root),
        map_location="cpu",
        weights_only=True,
    )
    hub_args = {
        "trust_remote_code": True,
        "revision": BIOT5_REVISION,
        "code_revision": BIOT5_REVISION,
    }
    config = AutoConfig.from_pretrained(BIOT5_REPO, **hub_args)
    tokenizer = AutoTokenizer.from_pretrained(BIOT5_REPO, **hub_args)
    base = AutoModel.from_pretrained(BIOT5_REPO, config=config, **hub_args)

    class FrozenBioT5(nn.Module):
        def __init__(self):
            super().__init__()
            self.transformer = base
            hidden = int(base.config.hidden_size)
            self.regressor = nn.Sequential(
                nn.Linear(hidden, 512),
                nn.LayerNorm(512),
                nn.GELU(),
                nn.Linear(512, 256),
                nn.LayerNorm(256),
                nn.GELU(),
                nn.Linear(256, 1),
            )

    model = FrozenBioT5()
    model.load_state_dict(state)
    model.to(device).eval()
    loader = DataLoader(_transformer_dataset(sequences, tokenizer), batch_size=BATCH_SIZE)
    chunks = []
    with torch.no_grad():
        for batch in loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            decoder_ids = torch.zeros(
                (ids.size(0), 1), dtype=torch.long, device=device
            )
            output = model.transformer(
                input_ids=ids,
                attention_mask=mask,
                decoder_input_ids=decoder_ids,
            )
            pooled = output.last_hidden_state.mean(dim=1)
            chunks.append(model.regressor[:6](pooled).cpu().numpy())
    return _require_256("selfies", np.vstack(chunks), len(smiles))


def extract_fingerprint(
    dataset: str, smiles: Sequence[str], repo_root: Optional[Path] = None
) -> np.ndarray:
    import torch
    import torch.nn as nn

    _seed_cpu()
    device = _device()
    transformer = load_joblib(fingerprint_transformer_path(dataset, repo_root))
    ecfp = np.asarray(transformer.transform(list(smiles)), dtype=np.float32)
    if ecfp.shape != (len(smiles), 2048):
        raise AssertionError(f"fingerprint: expected {(len(smiles), 2048)}, got {ecfp.shape}")

    class FrozenECFPRegressor(nn.Module):
        def __init__(self):
            super().__init__()
            self.regressor = nn.Sequential(
                nn.Linear(2048, 1024),
                nn.LayerNorm(1024),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(1024, 256),
                nn.LayerNorm(256),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(256, 1),
            )

    model = FrozenECFPRegressor()
    model.load_state_dict(
        torch.load(
            checkpoint_path(dataset, "fingerprint", repo_root),
            map_location="cpu",
            weights_only=True,
        )
    )
    model.to(device).eval()
    with torch.no_grad():
        features = model.regressor[:8](torch.from_numpy(ecfp).to(device)).cpu().numpy()
    return _require_256("fingerprint", features, len(smiles))


def extract_graph(
    dataset: str, smiles: Sequence[str], repo_root: Optional[Path] = None
) -> np.ndarray:
    import deepchem as dc
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from rdkit import Chem
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader
    from torch_geometric.nn import ChebConv, global_max_pool, global_mean_pool

    _seed_cpu()
    device = _device()
    featurizer = dc.feat.MolGraphConvFeaturizer(use_edges=True)
    graphs = []
    for index, value in enumerate(smiles):
        smi = str(value)
        try:
            if Chem.MolFromSmiles(smi) is None:
                raise ValueError("invalid RDKit SMILES")
            graph = featurizer.featurize([smi])[0]
            if graph.node_features is None or graph.num_nodes == 0:
                raise ValueError("empty graph")
            graphs.append(
                Data(
                    x=torch.tensor(graph.node_features, dtype=torch.float32),
                    edge_index=torch.tensor(graph.edge_index, dtype=torch.long),
                    edge_attr=torch.tensor(graph.edge_features, dtype=torch.float32),
                )
            )
        except Exception as exc:
            # The original notebooks silently skipped failed graphs.  That is not
            # safe for an aligned four-family pipeline, so fail without reordering.
            raise ValueError(f"Graph conversion failed at row {index}: {smi!r}") from exc

    class FrozenChebRegressorV3(nn.Module):
        def __init__(self):
            super().__init__()
            self.input_proj = nn.Linear(30, 256)
            self.convs = nn.ModuleList([ChebConv(256, 256, 3) for _ in range(3)])
            self.lns = nn.ModuleList([nn.LayerNorm(256) for _ in range(3)])
            self.fc = nn.Sequential(
                nn.Linear(512, 256), nn.LayerNorm(256), nn.GELU(), nn.Linear(256, 1)
            )

    model = FrozenChebRegressorV3()
    model.load_state_dict(
        torch.load(
            checkpoint_path(dataset, "graph", repo_root),
            map_location="cpu",
            weights_only=True,
        )
    )
    model.to(device).eval()
    chunks = []
    with torch.no_grad():
        for batch in DataLoader(graphs, batch_size=max(1, len(graphs)), shuffle=False):
            batch = batch.to(device)
            x = F.relu(model.input_proj(batch.x))
            for convolution, layer_norm in zip(model.convs, model.lns):
                x = x + F.relu(layer_norm(convolution(x, batch.edge_index)))
            mean = global_mean_pool(x, batch.batch)
            maximum = global_max_pool(x, batch.batch)
            chunks.append(
                model.fc[:3](torch.cat([mean, maximum], dim=-1)).cpu().numpy()
            )
    return _require_256("graph", np.vstack(chunks), len(smiles))


EXTRACTORS: Dict[str, Callable[..., np.ndarray]] = {
    "smiles": extract_smiles,
    "selfies": extract_selfies,
    "graph": extract_graph,
    "fingerprint": extract_fingerprint,
}


def extract_all(
    dataset: str,
    smiles: Sequence[str],
    targets: Sequence[float],
    repo_root: Optional[Path] = None,
) -> Dict[str, FeatureSplit]:
    ds = normalize_dataset(dataset)
    smi = np.asarray(smiles).astype(str)
    y = np.asarray(targets)
    return {
        family: FeatureSplit(smi, EXTRACTORS[family](ds, smi, repo_root), y)
        for family in ("smiles", "selfies", "graph", "fingerprint")
    }
