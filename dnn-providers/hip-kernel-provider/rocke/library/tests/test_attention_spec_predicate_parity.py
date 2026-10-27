# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The spec-shaped ``supports_tiled_{2d,3d}_spec`` must agree with the dispatcher gate.

``hkp_pack`` pre-flights a spec through ``supports_tiled_2d_spec`` /
``supports_tiled_3d_spec``; the dispatcher decides routing through
``supports_native_unified_attention_tiled`` / ``..._3d_tiled``. If a wrapper drops a
spec field the gate forwards (the fp8 K/V storage dtype, the tile size, the gfx942
wide-K / ring flags), pack time sizes a different kernel than the one dispatch builds
and can wrongly reject (or admit) a spec. This runs both on specs the dispatcher itself
produces, over fp8 / ALiBi / QQ-bias problems and over problems the gate rejects as
well as the ones it admits, and compares the whole verdict, reason included.

The 4-warp fast cohort (``_gfx942_4warp_fast``) is excluded from the 2D comparison:
its flag is not a spec field, so ``supports_tiled_2d_spec`` cannot reproduce the gate
for it. ``hkp_pack`` does alias that builder, ``build_gfx942_4warp_gqa``, to
``supports_gfx942_4warp_gqa_spec``, which fixes ``use_d256_fast=True`` from the builder;
that wrapper is covered by ``dispatch/attention/test_tiled_support_spec_wrappers.py``.
"""

from __future__ import annotations

import collections
import dataclasses
import importlib
import itertools

import pytest

import kernels.common.attention_unified as au
from kernels.common.attention_unified import (
    _tiled_3d_spec_from_problem,
    _tiled_spec_from_problem,
)


def _problem(**kw):
    base = dict(
        total_q=4096,
        num_seqs=1,
        num_query_heads=32,
        num_kv_heads=8,
        head_size=128,
        block_size=16,
        max_seqlen_q=4096,
        max_seqlen_k=4096,
        dtype="bf16",
        sliding_window=0,
    )
    base.update(kw)
    return au.UnifiedAttentionProblem(**base)


def _variant_fields(arch, variant):
    """Problem fields of one feature variant; fp8 is quantised to the arch's native
    format, which the dispatcher otherwise rejects before consulting the kernel gate."""
    if variant == "fp8":
        return dict(use_fp8=True, fp8_fnuz=arch == "gfx942")
    return {
        "plain": {},
        "alibi": dict(use_alibi=True),
        "qq_bias": dict(use_qq_bias=True),
    }[variant]


_VARIANTS = ("plain", "alibi", "qq_bias", "fp8")

# head_size 96 is outside every gate's envelope; 256 query heads over 8 KV heads is 32
# queries per KV head, over the cap of 16, and 24 over 8 is 3, which 3D cannot tile.
_HEAD_SIZES = (64, 96, 128, 256)
_NUM_HEADS = ((32, 8), (256, 8), (24, 8))


def _problems(arch, *, window_grid):
    for (
        head_size,
        dtype,
        block_size,
        window,
        (num_query_heads, num_kv_heads),
        variant,
    ) in itertools.product(
        _HEAD_SIZES,
        ("bf16", "fp16"),
        (16, 64),
        window_grid,
        _NUM_HEADS,
        _VARIANTS,
    ):
        yield variant, _problem(
            head_size=head_size,
            dtype=dtype,
            block_size=block_size,
            sliding_window=window,
            num_query_heads=num_query_heads,
            num_kv_heads=num_kv_heads,
            **_variant_fields(arch, variant),
        )


def _assert_both_directions_were_compared(arch, compared, *, fp8_buildable):
    """A grid that never reaches a gate-False (or fp8) verdict cannot catch a wrapper
    that is too permissive (or one that drops the fp8 storage dtype)."""
    assert compared[("plain", True)], f"{arch}: no accepted problem was comparable"
    assert compared[("plain", False)], f"{arch}: no rejected problem was comparable"
    assert bool(compared[("fp8", True)]) == fp8_buildable, (
        f"{arch}: fp8 problems "
        f"{'did not yield' if fp8_buildable else 'unexpectedly yielded'} "
        "an accepted comparable spec"
    )


@pytest.mark.parametrize("arch", ["gfx942", "gfx950"])
def test_2d_spec_wrapper_matches_dispatcher_gate(arch, monkeypatch):
    monkeypatch.setattr(au, "_RESOLVED_ATTENTION_ARCH", arch)
    mod = importlib.import_module(f"kernels.{arch}.attention_tiled_2d")
    compared = collections.Counter()
    for variant, p in _problems(arch, window_grid=(0, 4096)):
        if arch == "gfx942" and au._gfx942_4warp_fast(p, arch):
            continue
        gate = au.supports_native_unified_attention_tiled(p, arch)
        try:
            spec = _tiled_spec_from_problem(p, arch)
        except (RuntimeError, ValueError):
            # Over-budget or invalid spec shape; skip rather than fail. gfx942 has
            # no fp8 K/V cache, so no fp8 spec exists for its 2D wrapper to judge.
            continue
        wrapper = mod.supports_tiled_2d_spec(spec, arch=arch)
        assert wrapper == gate, (
            f"{arch} {variant} D{p.head_size} {p.dtype} bs{p.block_size} "
            f"sw{p.sliding_window} h{p.num_query_heads}/{p.num_kv_heads}: "
            f"gate={gate} wrapper={wrapper}"
        )
        compared[(variant, gate[0])] += 1
    _assert_both_directions_were_compared(
        arch, compared, fp8_buildable=arch == "gfx950"
    )


@pytest.mark.parametrize("arch", ["gfx942", "gfx950"])
def test_3d_spec_wrapper_matches_dispatcher_gate(arch, monkeypatch):
    monkeypatch.setattr(au, "_RESOLVED_ATTENTION_ARCH", arch)
    mod = importlib.import_module(f"kernels.{arch}.attention_tiled_3d")
    compared = collections.Counter()
    for variant, p in _problems(arch, window_grid=(0,)):
        gate = au.supports_native_unified_attention_3d_tiled(p, arch)
        try:
            spec = _tiled_3d_spec_from_problem(p, arch)
        except (RuntimeError, ValueError):
            continue
        wrapper = mod.supports_tiled_3d_spec(spec, arch=arch)
        assert wrapper == gate, (
            f"{arch} {variant} D{p.head_size} {p.dtype} bs{p.block_size} "
            f"h{p.num_query_heads}/{p.num_kv_heads}: gate={gate} wrapper={wrapper}"
        )
        compared[(variant, gate[0])] += 1
    _assert_both_directions_were_compared(arch, compared, fp8_buildable=True)


# The dispatcher only produces tile sizes the gate admits, and the spec rejects the
# rest of the structural tile rules itself, so no dispatcher-derived problem can tell a
# wrapper that forwards the spec's ``tile_size`` from one that leaves the gate on its
# default. These specs are hand-built so the tile size alone decides the verdict:
#   gfx950: 8 warps issue 8 * 64 * 8 halves per async-DMA call, and a 32-token tile of
#           head_size 64 holds only 2048, so the slab under-fills.
#   gfx942: a 128-token tile of head_size 128 stages 106,752 B of LDS against the
#           65,536 B budget, where the default one-block tile (16 tokens) fits.
_TILE_SIZE_SPECS = {
    "gfx950": dict(head_size=64, block_size=32, num_warps=8, tile_size=32),
    "gfx942": dict(head_size=128, block_size=16, num_warps=1, tile_size=128),
}


@pytest.mark.parametrize("arch", ["gfx942", "gfx950"])
def test_2d_spec_wrapper_judges_the_specs_tile_size(arch):
    mod = importlib.import_module(f"kernels.{arch}.attention_tiled_2d")
    spec = mod.UnifiedAttention2DTiledSpec(
        num_query_heads=32,
        num_kv_heads=8,
        dtype="bf16",
        use_sinks=False,
        sliding_window=0,
        has_softcap=False,
        **_TILE_SIZE_SPECS[arch],
    )

    assert mod.supports_tiled_2d_spec(dataclasses.replace(spec, tile_size=None))[0]
    assert not mod.supports_tiled_2d_spec(spec)[0], "the oversized tile was not judged"


# The gfx942 gate sizes LDS from the spec's flash flags, so each flag the wrapper
# forwards moves a boundary. Hand-derived for head_size 128, tile_size 128, 2-byte
# elements, against the 65,536 B budget (the transposed-x8 path of ``supports_tiled_2d``):
#   one K buffer:                 1*128*128*2 + 128*128*2        = 65,536  (at the limit)
#   one K buffer + V pad of 8:    1*128*128*2 + (128+8)*128*2    = 67,584
#   two K buffers:                2*128*128*2 + 128*128*2        = 98,304
#   K ring, depth 3, slice 32:    3*128*32*2  + 16*16*(64+8)*2   = 61,440
#   K ring, depth 2, slice 64:    2*128*64*2  + 16*16*(64+8)*2   = 69,632
# The ring cases start from the dispatcher's fp16 spec (which rings), the others from
# its bf16 spec (one K buffer, no ring).
_GFX942_FLASH_LDS_CASES = [
    ("one_k_buffer", "bf16", {}, True),
    ("one_k_buffer_v_pad", "bf16", {"use_conflict_free_v_store": True}, False),
    ("two_k_buffers", "bf16", {"use_k_single_buffer": False}, False),
    ("k_ring_depth3_slice32", "fp16", {"ring_depth": 3}, True),
    ("k_ring_depth2_slice64", "fp16", {"ring_depth": 2, "k_slice_hd": 64}, False),
]


@pytest.mark.parametrize(
    "dtype,overrides,fits",
    [case[1:] for case in _GFX942_FLASH_LDS_CASES],
    ids=[case[0] for case in _GFX942_FLASH_LDS_CASES],
)
def test_gfx942_2d_wrapper_sizes_lds_from_the_specs_flash_flags(
    dtype, overrides, fits, monkeypatch
):
    monkeypatch.setattr(au, "_RESOLVED_ATTENTION_ARCH", "gfx942")
    mod = importlib.import_module("kernels.gfx942.attention_tiled_2d")
    spec = dataclasses.replace(
        _tiled_spec_from_problem(_problem(dtype=dtype), "gfx942"),
        tile_size=128,
        **overrides,
    )
    assert mod.supports_tiled_2d_spec(spec)[0] is fits
