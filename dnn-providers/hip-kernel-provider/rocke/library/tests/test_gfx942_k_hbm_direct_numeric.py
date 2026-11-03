# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""gfx942 tiled 2D attention with K read straight from HBM, against an FP32
reference.

``use_k_hbm_direct`` reads the QK operand K from global memory and allocates
K_lds as a one-element-per-buffer stub, so Q must be read from global memory
too: a Q tile staged through the stub overwrites its own rows and the kernel
computes with a corrupted Q (``test_gfx942_k_hbm_direct_q_staging.py`` pins
the staging on the emitted IR without a GPU). These tests run the kernel on a
gfx942 GPU for a bf16 head size 128 and an fp16 head size 64 geometry, each with
V staged in LDS and with V also read from HBM, and compare every output with
the FP32 reference.

The knob is held out of the dispatch tuning space (``KNOWN_WRONG_KNOBS``), so
the tests take a dispatch variant's spec and turn the knob on directly. Q, K
and V are standard normal so the softmax is far from flat: a corrupted Q then
moves the output well past the tolerance, where a narrow fill could hide it.

The kernels run on ``rocke.runtime`` device buffers filled from numpy, so the
tests run in GPU lanes without torch.
"""

import dataclasses
import math

import numpy as np
import pytest

from dispatch.attention.gfx942_unified import GFX942_TUNING_VARIANTS
from dispatch.attention.unified_rules import canonicalize_tuning_spec
from kernels.common.attention_unified import UnifiedAttentionProblem
from rocke.runtime.hip_module import get_device_arch

from ._attention_paged_reference import paged_attention_reference, round_to_half
from ._attention_tiled_launch import launch_tiled_one_sequence

_ARCH = get_device_arch()
pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(_ARCH != "gfx942", reason=f"needs a gfx942 GPU, found {_ARCH}"),
]

_NUM_QUERY_HEADS, _NUM_KV_HEADS = 32, 8
_SEQ_LEN = 256

# (dtype, head size, KV page size, transposed-x8 dispatch variant). The bf16
# geometry has BLOCK_M = T = 64; the fp16 one has BLOCK_M = 128 over T = 64,
# where a staged Q would span both K_lds slots.
_GEOMETRIES = [
    ("bf16", 128, 64, "gfx942_u2d_transposed_x8_nw2_mw32_t1xb_llvm"),
    ("fp16", 64, 32, "gfx942_u2d_transposed_x8_nw4_mw32_t2xb_llvm"),
]

# Largest |O - O_ref| allowed: fp16 / bf16 rounding of the output and of P.
_TOLERANCE = {"fp16": 2e-2, "bf16": 4e-2}


@pytest.fixture(scope="module")
def rt():
    from rocke.runtime.hip_module import Runtime

    return Runtime()


def _k_from_hbm_spec(problem, variant_id, v_from_hbm):
    """``variant_id``'s spec for ``problem`` with K (and optionally V) read
    straight from HBM."""
    variant = next((v for v in GFX942_TUNING_VARIANTS if v.spec_id == variant_id), None)
    assert variant is not None, f"no gfx942 dispatch variant {variant_id!r}"
    base, why = canonicalize_tuning_spec(problem, variant, {})
    assert base is not None, f"{variant_id}: {why}"
    kernel_spec = dataclasses.replace(
        base.kernel_spec, use_k_hbm_direct=True, use_v_hbm_direct=v_from_hbm
    )
    return dataclasses.replace(base, kernel_spec=kernel_spec)


@pytest.mark.parametrize("v_from_hbm", [False, True], ids=["v_in_lds", "v_from_hbm"])
@pytest.mark.parametrize(
    "dtype,head_size,page_size,variant_id",
    _GEOMETRIES,
    ids=[f"{g[0]}_d{g[1]}" for g in _GEOMETRIES],
)
def test_k_from_hbm_matches_reference(
    rt, dtype, head_size, page_size, variant_id, v_from_hbm
):
    rng = np.random.default_rng(0)
    pages = _SEQ_LEN // page_size
    # Spare pages and a shuffled block table catch a misaddressed page.
    num_blocks = pages + 3
    cache_shape = (num_blocks, page_size, _NUM_KV_HEADS, head_size)
    q = round_to_half(
        rng.standard_normal((_SEQ_LEN, _NUM_QUERY_HEADS, head_size)), dtype
    )
    k_cache = round_to_half(rng.standard_normal(cache_shape), dtype)
    v_cache = round_to_half(rng.standard_normal(cache_shape), dtype)
    block_table = rng.permutation(num_blocks)[:pages].reshape(1, pages)
    block_table = block_table.astype(np.int32)
    scale = 1.0 / math.sqrt(head_size)

    problem = UnifiedAttentionProblem(
        total_q=_SEQ_LEN,
        num_seqs=1,
        num_query_heads=_NUM_QUERY_HEADS,
        num_kv_heads=_NUM_KV_HEADS,
        head_size=head_size,
        block_size=page_size,
        max_seqlen_q=_SEQ_LEN,
        max_seqlen_k=_SEQ_LEN,
        dtype=dtype,
        num_kv_blocks=num_blocks,
    )
    spec = _k_from_hbm_spec(problem, variant_id, v_from_hbm)

    o = launch_tiled_one_sequence(
        rt, problem, spec, q, k_cache, v_cache, block_table, scale, dtype
    )
    o_ref, _has_key = paged_attention_reference(
        q, k_cache, v_cache, _SEQ_LEN, block_table[0], scale
    )

    assert np.isfinite(o).all(), "non-finite output"
    error = np.abs(o - o_ref)
    assert error.max() < _TOLERANCE[dtype], (
        f"max |O - O_ref| {error.max():.3g} at (row, head, dim) "
        f"{np.unravel_index(error.argmax(), error.shape)}"
    )
