"""The tiled 3D (split-KV) gate accepts num_queries_per_kv that does not divide
BLOCK_M=16, so decode with such a GQA ratio can take the split-KV path."""

from __future__ import annotations

import importlib

import pytest

ARCHES = ["gfx942", "gfx950"]


def _supports(arch: str, nqk: int):
    mod = importlib.import_module(f"kernels.{arch}.attention_tiled_3d")
    return mod.supports_tiled_3d(
        head_size=128,
        block_size=16,
        dtype="bf16",
        num_queries_per_kv=nqk,
        use_alibi=False,
        use_qq_bias=False,
        use_fp8=False,
        q_dtype="bf16",
        arch=arch,
    )


@pytest.mark.parametrize("arch", ARCHES)
@pytest.mark.parametrize("nqk", [1, 2, 3, 4, 5, 6, 7, 8, 12, 15, 16])
def test_accepts_nqk_in_range(arch, nqk):
    ok, reason = _supports(arch, nqk)
    assert ok, f"arch={arch} NQK={nqk} rejected: {reason}"


@pytest.mark.parametrize("arch", ARCHES)
@pytest.mark.parametrize("nqk", [0, 17, 32])
def test_rejects_nqk_out_of_range(arch, nqk):
    ok, _ = _supports(arch, nqk)
    assert not ok


@pytest.mark.parametrize("arch", ARCHES)
def test_decode_nonpow2_gqa_routes_to_3d(arch):
    from kernels.common import attention_unified as au

    problem = au.UnifiedAttentionProblem(
        total_q=1,
        num_seqs=1,
        num_query_heads=40,
        num_kv_heads=8,
        head_size=128,
        block_size=16,
        max_seqlen_q=1,
        max_seqlen_k=32768,
        dtype="bf16",
        num_cus=256,
        num_kv_blocks=2048,
    )
    assert problem.select_path() == "3d"
    ok, reason = au._explicit_path_supported(problem, None, "3d", arch)
    assert ok, reason
