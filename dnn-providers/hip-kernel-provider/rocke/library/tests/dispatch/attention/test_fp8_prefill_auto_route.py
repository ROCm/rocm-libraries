# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Auto launch of FP8 K/V prefill with no tiled 2D kernel takes the 3D path."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import kernels.common.attention_unified as au


def _problem(arch: str) -> au.UnifiedAttentionProblem:
    return au.UnifiedAttentionProblem(
        total_q=2048,
        num_seqs=1,
        num_query_heads=64,
        num_kv_heads=8,
        head_size=128,
        block_size=64,
        max_seqlen_q=2048,
        max_seqlen_k=2048,
        dtype="bf16",
        use_fp8=True,
        fp8_fnuz=arch == "gfx942",
        num_cus=304,
        num_kv_blocks=32,
    )


def test_gfx942_fp8_prefill_auto_runs_3d(monkeypatch):
    problem = _problem("gfx942")
    assert problem.select_path() == "2d"
    assert not au._explicit_path_supported(problem, None, "2d", "gfx942")[0]
    assert au._explicit_path_supported(problem, None, "3d", "gfx942")[0]

    calls = []

    def fake_3d(**kwargs):
        calls.append(kwargs)
        return "3d"

    monkeypatch.setattr(au, "_resolve_attention_arch", lambda: "gfx942")
    monkeypatch.setattr(au, "_run_3d_tiled", fake_3d)
    out = au.run_unified_attention_torch(
        problem=problem,
        q=None,
        k=None,
        v=None,
        out=None,
        cu_seqlens_q=None,
        seqused_k=None,
        softmax_scale=0.1,
        block_table=SimpleNamespace(shape=(1, 32)),
        softcap=0.0,
        k_scale=0.5,
        v_scale=0.25,
    )
    assert out == "3d"
    assert [(c["k_scale"], c["v_scale"]) for c in calls] == [(0.5, 0.25)]


@pytest.mark.parametrize("arch", ["gfx950"])
def test_fp8_prefill_with_a_2d_kernel_keeps_2d(arch):
    # The 3D fallback only applies when no tiled 2D kernel serves the problem.
    problem = _problem(arch)
    assert problem.select_path() == "2d"
    assert au._explicit_path_supported(problem, None, "2d", arch)[0]
