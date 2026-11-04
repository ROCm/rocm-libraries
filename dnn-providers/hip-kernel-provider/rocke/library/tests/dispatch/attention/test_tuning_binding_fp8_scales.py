# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The explicit unified tuning binding forwards the FP8 K/V dequant scales."""

from __future__ import annotations

from dataclasses import replace

import pytest

import kernels.common.attention_unified as au
from dispatch.attention import AttentionRequest, bindings, dispatch_attention
from dispatch.attention.common import _problem


@pytest.mark.parametrize(
    "arch, spec_id",
    [
        ("gfx950", "gfx950_u3d_splitkv_seg64_t1xb"),
        ("gfx942", "gfx942_u3d_splitkv_seg8_t1xb"),
    ],
)
def test_tuning_binding_forwards_fp8_kv_scales(arch, spec_id, monkeypatch):
    req = AttentionRequest(
        batch=1,
        nhead_q=8,
        nhead_k=8,
        seqlen_q=1,
        seqlen_k=64,
        hdim_q=64,
        hdim_v=64,
        arch=arch,
        dtype="bf16",
        kv_block_size=16,
        use_fp8=True,
        fp8_fnuz=arch == "gfx942",
        num_cus=256,
    )
    result = dispatch_attention(
        replace(req, algorithm="unified_tuning", spec_id=spec_id)
    )
    assert result.spec.kernel_spec.kv_storage_dtype == "fp8e4m3"
    # Tensor structure is validated elsewhere; this checks the scalars only.
    monkeypatch.setattr(
        bindings, "validate_tuning_attention_tensors", lambda *a, **kw: 4
    )
    tensors = {
        name: object()
        for name in ("q", "k", "v", "out", "cu_seqlens_q", "seqused_k", "block_table")
    }
    tensors["problem"] = _problem(req)
    calls = []
    monkeypatch.setattr(
        au, "run_unified_attention_torch", lambda **kw: calls.append(kw)
    )
    binding = result.candidate.bind_torch(
        req, result.spec, tensors, k_scale=0.5, v_scale=0.25
    )
    binding.launch()
    binding.launch(k_scale=2.0)
    assert [(c["k_scale"], c["v_scale"]) for c in calls] == [(0.5, 0.25), (2.0, 0.25)]
