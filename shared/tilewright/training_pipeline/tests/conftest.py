# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import json
import sys
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence

import pytest

PIPELINE_DIR = Path(__file__).resolve().parent.parent
for _p in (PIPELINE_DIR, PIPELINE_DIR / "stages", PIPELINE_DIR / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


@pytest.fixture
def pipeline_dir() -> Path:
    return PIPELINE_DIR


@pytest.fixture
def config_env(monkeypatch, tmp_path):
    """The environment variables the shipped configs reference."""
    for name in (
        "TILEWRIGHT_BENCH_ROOT",
        "TILEWRIGHT_DEPLOY_ROOT",
        "TILEWRIGHT_DATASETS",
    ):
        monkeypatch.setenv(name, str(tmp_path / name.lower()))
    return tmp_path


def write_bundle(
    train_dir: Path,
    labels: Iterable[str],
    *,
    seed: int = 0,
    dims: Sequence[int] = (4, 6, 5),
    signatures: Optional[Dict[str, list]] = None,
    names: Optional[Dict[str, list]] = None,
) -> Path:
    """A small trained-bundle `models.pt` plus `cells.json` with random
    weights for `labels`."""
    import torch

    from lib import features as fs
    from lib import mlrec

    q = fs.query_feature_names()
    i = fs.item_feature_names()
    x = fs.interaction_feature_names()
    embed, hidden, inter = dims
    models = {}
    for n, label in enumerate(sorted(labels)):
        g = torch.Generator().manual_seed(seed * 1000 + n)
        shapes = mlrec.tensor_shapes(len(q), len(i), len(x), embed, hidden, inter)
        sd = {k: torch.randn(*s, generator=g) for k, s in shapes.items()}
        sd["temperature"] = torch.tensor([1.0])
        models[label] = {
            "state_dict": sd,
            "embed_dim": embed,
            "hidden_dim": hidden,
            "inter_hidden": inter,
            "feature_norms": {
                "q_mean": [0.0] * len(q),
                "q_std": [1.0] * len(q),
                "i_mean": [0.0] * len(i),
                "i_std": [1.0] * len(i),
                "x_mean": [0.0] * len(x),
                "x_std": [1.0] * len(x),
            },
            "smart_k_signatures": (signatures or {}).get(
                label, [[128, 128, 64, 16, 16, 128, 0, 0]]
            ),
        }
    bundle = {"q_names": q, "i_names": i, "x_names": x, "models": models}
    bundle.update(names or {})
    train_dir.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, train_dir / "models.pt")
    (train_dir / "cells.json").write_text(
        json.dumps({"feature_names_hash": fs.feature_names_hash(), "cells": []})
    )
    return train_dir / "models.pt"


@pytest.fixture
def make_bundle():
    return write_bundle
