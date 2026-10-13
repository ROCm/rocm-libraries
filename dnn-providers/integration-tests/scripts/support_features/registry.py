# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""API node type -> the op module that catalogs its features.

Every member of the schema's ``NodeAttributes`` union is in exactly one of
``OPS`` (catalogued) or ``PENDING`` (not yet). ``test_support_features.py``
checks both against ``graph.fbs``, so a node type added to the API cannot go
uncounted: it fails the test until it is listed here. ``PENDING`` shrinks by
one entry per op commit and goes away when it is empty.
"""

from __future__ import annotations

import pathlib
from types import ModuleType

from support_features import (
    FeatureError,
    batchnorm_backward,
    batchnorm_inference,
    batchnorm_inference_variance_ext,
    batchnorm_training,
    block_scale_dequantize,
    convolution_bwd,
    convolution_fwd,
    convolution_wrw,
    layernorm,
    layernorm_backward,
    matmul,
    pointwise,
    rmsnorm,
    rmsnorm_backward,
    sdpa,
    sdpa_backward,
)

REPO_ROOT = pathlib.Path(__file__).resolve().parents[4]
SCHEMA_DIR = REPO_ROOT / "projects" / "hipdnn" / "flatbuffers_sdk" / "schemas"

# Catalogued op modules, keyed by their NODE_TYPE.
OPS: dict[str, ModuleType] = {
    m.NODE_TYPE: m
    for m in (
        batchnorm_backward,
        batchnorm_inference,
        batchnorm_inference_variance_ext,
        batchnorm_training,
        block_scale_dequantize,
        convolution_bwd,
        convolution_fwd,
        convolution_wrw,
        layernorm,
        layernorm_backward,
        matmul,
        pointwise,
        rmsnorm,
        rmsnorm_backward,
        sdpa,
        sdpa_backward,
    )
}

# Node types with no catalog yet. Their graphs are counted, but no feature %
# is reported for them.
PENDING = frozenset(
    {
        "BlockScaleQuantizeAttributes",
        "CustomOpAttributes",
        "MoeGroupedMatmulAttributes",
        "MoeGroupedMatmulBwdAttributes",
        "ReductionAttributes",
        "ResampleBwdAttributes",
        "ResampleFwdAttributes",
    }
)


def op_module(node_type: str) -> ModuleType | None:
    """The op module for ``node_type``; None while it is pending.

    A node type in neither list is not in the API this registry was written
    against, so guessing would be wrong either way: it raises.
    """
    module = OPS.get(node_type)
    if module is not None:
        return module
    if node_type in PENDING:
        return None
    raise FeatureError(f"unknown node type {node_type!r} (not in the API registry)")
