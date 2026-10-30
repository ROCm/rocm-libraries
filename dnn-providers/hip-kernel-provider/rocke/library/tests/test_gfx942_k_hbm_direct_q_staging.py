# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Where the gfx942 tiled-2D kernel stages Q when K is read straight from HBM.

``use_k_hbm_direct`` reads the QK operand K straight from global memory, so the
kernel allocates ``K_lds`` as a one-element-per-buffer stub. Without K_lds
space, Q cannot use K_lds as its load scratch: a ``[BLOCK_M, HD]`` Q tile
written through a ``[bufs, 1, 1]`` stub lands element ``(row, col)`` at
``row + col``, so Q rows overwrite each other before the per-lane gather reads
them back and the kernel computes with a corrupted Q.

The kernel must therefore gather Q straight from global memory, exactly as the
explicit ``use_q_direct_global`` knob does. These tests check that in the
emitted LLVM IR for two K-HBM-direct configurations (bf16 head size 128 with V
also from HBM, and fp16 head size 64 with sinks and the causal-mask phase
split).

No GPU required.
"""

from __future__ import annotations

import dataclasses
import re

import pytest

from kernels.gfx942.attention_tiled_2d import (
    UnifiedAttention2DTiledSpec,
    build_unified_attention_2d_tiled,
)
from rocke.core.lower_llvm import lower_kernel_to_llvm

ARCH = "gfx942"

_BF16_D128_K_AND_V_FROM_HBM = UnifiedAttention2DTiledSpec(
    head_size=128,
    block_size=64,
    num_query_heads=32,
    num_kv_heads=8,
    dtype="bf16",
    use_sinks=False,
    sliding_window=0,
    has_softcap=False,
    num_seqs=1,
    num_warps=2,
    waves_per_eu=2,
    tile_size=64,
    block_m_per_warp=32,
    use_mfma_32x32x8=True,
    use_transposed_qk_32x32=True,
    use_transposed_scalar_state=True,
    use_transposed_invariant_hoist=True,
    use_transposed_mask_once=True,
    use_transposed_mask_limit=True,
    use_k_single_buffer=True,
    kv_cache_policy="all",
    use_k_hbm_direct=True,
    use_v_hbm_direct=True,
)

_FP16_D64_K_FROM_HBM_SINKS = UnifiedAttention2DTiledSpec(
    head_size=64,
    block_size=32,
    num_query_heads=32,
    num_kv_heads=8,
    dtype="fp16",
    use_sinks=True,
    sliding_window=0,
    has_softcap=False,
    num_warps=4,
    block_m_per_warp=32,
    tile_size=64,
    use_mfma_32x32x8=True,
    use_transposed_qk_32x32=True,
    use_k_hbm_direct=True,
    use_causal_mask_phase_split=True,
)

SPECS = pytest.mark.parametrize(
    "spec",
    [_BF16_D128_K_AND_V_FROM_HBM, _FP16_D64_K_FROM_HBM_SINKS],
    ids=["bf16_d128_k_and_v_from_hbm", "fp16_d64_k_from_hbm_sinks"],
)

# An address computed through the K stub's LDS type ``[bufs x [1 x [1 x T]]]``.
_K_STUB_ADDRESS_RE = re.compile(
    r"getelementptr[^\n]*\[\d+ x \[1 x \[1 x (?:half|bfloat)\]\]\]"
)


def _llvm_ir(spec: UnifiedAttention2DTiledSpec) -> tuple[str, str]:
    kernel = build_unified_attention_2d_tiled(spec, arch=ARCH)
    return kernel.name, lower_kernel_to_llvm(kernel, arch=ARCH)


@SPECS
def test_nothing_is_staged_through_the_k_stub(spec):
    # K comes from HBM and Q comes from global memory, so no LDS address may be
    # formed inside the one-element K stub. A Q tile staged through it would
    # show up here as one address per Q store.
    _, ir = _llvm_ir(spec)
    stub_addresses = _K_STUB_ADDRESS_RE.findall(ir)
    assert not stub_addresses, (
        f"{len(stub_addresses)} LDS addresses go through the one-element K stub; "
        "with use_k_hbm_direct the Q tile must be read from global memory"
    )


@SPECS
def test_k_from_hbm_reads_q_like_the_direct_q_knob(spec):
    # use_k_hbm_direct must produce the same kernel as also asking for direct Q
    # explicitly; only the kernel name (which records the explicit knob) differs.
    name, ir = _llvm_ir(spec)
    direct_q_name, direct_q_ir = _llvm_ir(
        dataclasses.replace(spec, use_q_direct_global=True)
    )
    assert name != direct_q_name
    assert ir.replace(name, "KERNEL") == direct_q_ir.replace(direct_q_name, "KERNEL")
