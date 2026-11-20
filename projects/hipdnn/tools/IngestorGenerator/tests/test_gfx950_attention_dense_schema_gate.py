# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""hipkernel:Gfx950AttentionDense stays below engine API 1.3.0 until its matcher declines
ragged operands and a declared tensor alignment itself.

Its native graph_match checks neither a ragged_offset on Q/K/V/O nor a declared
alignment. hipDNN stamps a graph that uses either with min_required_engine_api_version
1.3.0, and GenericPlanBuilder::understandsGraph declines it only while the engine's
sdk_version is lower. The census suite TestGfx950AttentionDenseSchemaGate proves that
decline end to end, but it registers only where a build ships the engine for gfx950.
This pin runs in every build that runs the IngestorGenerator suite: the authored config,
its engine manifest and the committed UED all declare a version below 1.3.0.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

_ROOT = Path(__file__).resolve().parents[1]
_REPO = _ROOT.parents[3]
_CONFIGS = _ROOT / "configs"
_UED = (
    _REPO
    / "dnn-providers"
    / "hip-kernel-provider"
    / "src"
    / "engines"
    / "kernel_ingestor_engine"
    / "descriptors"
    / "rocKE"
    / "gfx950_attention_dense"
    / "gfx950_attention_dense.ued.json"
)

# The engine API level at which hipDNN stamps ragged and non-default-alignment graphs.
_RAGGED_AND_ALIGNMENT_API = (1, 3, 0)


def _version(text: str) -> tuple[int, ...]:
    return tuple(int(part) for part in str(text).split("."))


def _authored(name: str) -> str:
    return yaml.safe_load((_CONFIGS / name).read_text(encoding="utf-8"))["engine"][
        "sdk_version"
    ]


@pytest.mark.parametrize(
    "source",
    [
        "configs/gfx950_attention_dense.yaml",
        "configs/gfx950_attention_dense.engine.yaml",
        "gfx950_attention_dense.ued.json",
    ],
)
def test_sdk_version_stays_below_the_ragged_and_alignment_api(source):
    if source.endswith(".ued.json"):
        declared = json.loads(_UED.read_text(encoding="utf-8"))["sdk_version"]
    else:
        declared = _authored(Path(source).name)
    assert _version(declared) < _RAGGED_AND_ALIGNMENT_API, (
        f"{source} declares sdk_version {declared}: at 1.3.0 or above the schema gate "
        "no longer keeps ragged and non-default-alignment SDPA graphs away from "
        "hipkernel:Gfx950AttentionDense, whose graph_match checks neither. Raise it only "
        "with explicit declines in the matcher."
    )
