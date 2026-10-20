# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""API node type -> the op module that catalogs its features.

Every member of the schema's ``NodeAttributes`` union is in ``OPS``.
``test_support_features.py`` checks it against ``graph.fbs``, so a node type
added to the API cannot go uncounted: it fails the test until its catalog is
listed here.
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
    block_scale_quantize,
    convolution_bwd,
    convolution_fwd,
    convolution_wrw,
    custom_op,
    gdn,
    kda,
    layernorm,
    layernorm_backward,
    matmul,
    moe_grouped_matmul,
    moe_grouped_matmul_bwd,
    pointwise,
    reduction,
    resample_bwd,
    resample_fwd,
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
        block_scale_quantize,
        convolution_bwd,
        convolution_fwd,
        convolution_wrw,
        custom_op,
        gdn,
        kda,
        layernorm,
        layernorm_backward,
        matmul,
        moe_grouped_matmul,
        moe_grouped_matmul_bwd,
        pointwise,
        reduction,
        resample_bwd,
        resample_fwd,
        rmsnorm,
        rmsnorm_backward,
        sdpa,
        sdpa_backward,
    )
}


def op_module(node_type: str) -> ModuleType:
    """The op module for ``node_type``.

    A node type not listed is not in the API this registry was written
    against, so guessing would be wrong either way: it raises.
    """
    module = OPS.get(node_type)
    if module is not None:
        return module
    raise FeatureError(f"unknown node type {node_type!r} (not in the API registry)")
