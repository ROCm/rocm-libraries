# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""gfx950 tiled 2D attention: Q is never staged in a padded K_lds.

``use_kq_lds_pad`` lays K_lds out as padded slabs (``[slots, T / slab_rows,
slab_rows * HD + pad]``). Q staged in K_lds is stored and read as plain
``[slot, row, col]``, which misaddresses that layout and gives wrong output, so
the spec refuses a padded K_lds with an aliased Q; Q re-read and the direct
register gather give Q its own storage and build. The K_lds layout facts live on
the spec, and the dispatch LDS model reads the same properties. A pad the
kernel lays out as no pad (here: K double-buffered) leaves Q aliased and the
kernel unchanged, and a pad width off the 8-half grid (zero and negative
included) is laid out as no pad.

No GPU required.
"""

from __future__ import annotations

import dataclasses

import pytest

from kernels.gfx950.attention_tiled_2d import (
    UnifiedAttention2DTiledSpec,
    build_unified_attention_2d_tiled,
)
from rocke.core.lower_llvm import lower_kernel_to_llvm

ARCH = "gfx950"

# Single-buffer K, so the pad is laid out: 16-row slabs of 4 * 128 + 16 halves.
_PADDED = UnifiedAttention2DTiledSpec(
    head_size=128,
    block_size=64,
    num_query_heads=32,
    num_kv_heads=8,
    dtype="bf16",
    use_sinks=False,
    sliding_window=0,
    has_softcap=False,
    num_warps=2,
    block_m_per_warp=32,
    tile_size=64,
    use_mfma_32x32=True,
    use_transposed_qk_32x32=True,
    use_k_single_buffer=True,
    use_kq_lds_pad=True,
    kq_lds_pad_halves=16,
    use_q_reread=True,
)
_PADDED_K_LDS_TYPE = "[1 x [16 x [528 x bfloat]]]"


def _llvm_ir(spec: UnifiedAttention2DTiledSpec) -> tuple[str, str]:
    kernel = build_unified_attention_2d_tiled(spec, arch=ARCH)
    return kernel.name, lower_kernel_to_llvm(kernel, arch=ARCH)


def test_padded_k_lds_refuses_an_aliased_q():
    with pytest.raises(ValueError) as refusal:
        dataclasses.replace(_PADDED, use_q_reread=False)
    assert str(refusal.value) == (
        "gfx950 tiled 2D spec failed - use_kq_lds_pad=True with Q staged "
        "in K_lds (use_q_reread=False, use_q_direct_reg=False), expected "
        "use_q_reread or use_q_direct_reg: the padded K_lds slabs cannot "
        "hold an aliased Q"
    )


@pytest.mark.parametrize(
    "q_storage",
    [{"use_q_reread": True}, {"use_q_reread": False, "use_q_direct_reg": True}],
    ids=["q_reread", "q_direct_reg"],
)
def test_padded_k_lds_builds_with_q_in_its_own_storage(q_storage):
    spec = dataclasses.replace(_PADDED, **q_storage)
    assert spec.kq_lds_pad_active and not spec.q_aliases_k_lds
    _, ir = _llvm_ir(spec)
    assert _PADDED_K_LDS_TYPE in ir


def test_a_pad_laid_out_as_none_keeps_q_aliased_and_the_kernel_unchanged():
    # Double-buffered K: the pad is not laid out, so Q aliases K_lds as it does
    # without the pad and the kernel differs only in its name.
    unpadded = dataclasses.replace(
        _PADDED, use_kq_lds_pad=False, use_q_reread=False, use_k_single_buffer=False
    )
    inert_pad = dataclasses.replace(unpadded, use_kq_lds_pad=True)
    assert inert_pad.kq_lds_pad_off_reason == (
        "the KQ LDS pad is only laid out for a single-K schedule"
    )
    assert inert_pad.q_aliases_k_lds and not inert_pad.kq_lds_pad_active
    name, ir = _llvm_ir(unpadded)
    inert_name, inert_ir = _llvm_ir(inert_pad)
    assert _PADDED_K_LDS_TYPE not in inert_ir
    assert inert_ir.replace(inert_name, "KERNEL") == ir.replace(name, "KERNEL")


@pytest.mark.parametrize("halves", [0, -8, 12])
def test_a_pad_width_off_the_8_half_grid_is_laid_out_as_none(halves):
    spec = dataclasses.replace(_PADDED, kq_lds_pad_halves=halves)
    assert spec.kq_lds_pad_off_reason == "the KQ LDS pad needs an aligned slab layout"
    assert not spec.kq_lds_pad_active
    _, ir = _llvm_ir(spec)
    assert "[528 x bfloat]" not in ir and f"[{512 + halves} x bfloat]" not in ir
