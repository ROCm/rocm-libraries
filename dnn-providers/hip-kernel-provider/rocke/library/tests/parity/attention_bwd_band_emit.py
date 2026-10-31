#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# tests/parity/attention_bwd_band_emit.py -- Python reference emitter for the
# attention backward band helpers (rocke.helpers.attention_band and
# kernels.common._attention_bwd_band). The helpers have no kernel of their own,
# so this builds a small probe kernel that calls every band entry point once per
# config; the C twin attention_bwd_band_emit.c builds the same kernel through
# the C++ helpers (helper_rocke.helpers.attention_band.h,
# helper_rocke.instances.common._attention_bwd_band.h) and the two .ll must match.
#
# A config is (arch, band shape, tile pair), decoded from argv[1] by
# divmod: index = arch_base + tile * len(SHAPES) + shape, with the tile list of
# the arch (the (block_m, block_n) pairs the shipped general, unified and dq
# tuned tables plan on that arch). The shapes cover the mask policies the
# backward kernels build (NoMaskKTail, RuntimeBand over runtime lengths and
# bounds as attention_bwd / _dq / _unified do, and over build-time ints as the
# dense kernels do, including mixed int / runtime sides).
#
# argv[2] selects the output: ll (default), ir, verify (see _emit_common.py).
from _emit_common import run_emit
from kernels.common._attention_bwd_band import (
    NoMaskKTail,
    RuntimeBand,
    dkv_store_rule,
    interior_q_range,
    is_edge_tile,
    q_tile_range,
    tile_class,
    tile_inside_band,
)
from rocke.core.ir import I32, IRBuilder, PtrType
from rocke.helpers.attention_band import AttnRuntimeBounds

# (block_m, block_n) pairs per arch: the tuned-table tiles of the general
# kernel (and the shipped dq tile) that reach the band helpers.
TILES = {
    "gfx942": ((16, 32), (16, 64), (16, 128), (32, 128)),
    "gfx950": ((16, 64), (16, 128), (16, 192), (32, 128), (32, 256), (128, 64)),
}

# Band shapes. kind "none": NoMaskKTail; "band": RuntimeBand. For "band",
# left / right / off are "rt" (the kernel parameter) or a build-time int; off
# "br" is the per-sequence select the backward kernels build (0 or
# len_kv - len_q by the bottom_right kernel argument); lengths are "rt" or ints.
SHAPES = (
    ("none", "rt", "rt", None, None, None),
    ("band", "rt", "rt", "br", "rt", "rt"),
    ("band", "rt", "rt", "br", -1, 0),
    ("band", 1024, 1024, 0, -1, 0),
    ("band", 512, 1024, 512, -1, 0),
    ("band", 777, 777, 0, 128, 0),
    ("band", 100, 200, 100, 64, 64),
    ("band", 640, 640, 0, 96, -1),
    ("none", 1000, 1300, None, None, None),
    ("band", "rt", "rt", "br", "rt", 0),
    ("band", "rt", "rt", 0, 32, "rt"),
    ("band", "rt", "rt", 64, "rt", 0),
    ("band", "rt", "rt", -32, 48, "rt"),
    ("band", 1024, "rt", "br", 0, "rt"),
    ("band", "rt", 2048, 16, "rt", 0),
)


def _decode(idx: int):
    base = 0
    for arch, tiles in TILES.items():
        span = len(tiles) * len(SHAPES)
        if idx < base + span:
            t, s = divmod(idx - base, len(SHAPES))
            return arch, tiles[t], SHAPES[s]
        base += span
    raise SystemExit(f"unknown config index {idx}")


def _spec(idx: int):
    arch, tile, shape = _decode(idx)
    return (tile, shape), arch


def _operand(x, rt):
    return rt if x == "rt" else x


def build_attention_bwd_band_kernel(spec, *, arch: str):
    """The probe kernel of one (tile, shape) config."""
    (km, kn), (kind, s_q, s_kv, off, left, right) = spec
    b = IRBuilder("attn_bwd_band_probe")
    len_q = b.param("len_q", I32)
    len_kv = b.param("len_kv", I32)
    bottom_right = b.param("bottom_right", I32)
    left_p = b.param("left", I32)
    right_p = b.param("right", I32)
    s_kv_max = b.param("s_kv_max", I32)
    out = b.param("out", PtrType(I32, "global"))
    tid = b.thread_id_x()
    bid = b.block_id_x()
    q_len = _operand(s_q, len_q)
    kv_len = _operand(s_kv, len_kv)
    if kind == "none":
        mask = NoMaskKTail(b, q_len, kv_len, k_m0=km, k_n0=kn)
    else:
        zero = b.const_i32(0)
        if off == "br":
            off = b.select(b.cmp_ne(bottom_right, zero), b.sub(len_kv, len_q), zero)
        bounds = AttnRuntimeBounds(
            q_len, kv_len, off, _operand(left, left_p), _operand(right, right_p)
        )
        mask = RuntimeBand(b, bounds, k_m0=km, k_n0=kn)
    band = mask.band
    res = []

    # rocke.helpers.attention_band: every BandEmitter entry point.
    rb = band.row_band(tid)
    res += [rb.row_in, rb.valid]
    res += [v for v in (rb.lo, rb.hi) if v is not None]
    col = band.col_in(bid)
    res.append(col)
    res.append(band.cell_keep(rb, bid, col))
    res += list(band.k_tile_range(tid, km, kn))
    res.append(band.k_tile_count_ceil(kn))

    # Policy entry points, as the backward kernels call them.
    qt_start, n_qt = mask.q_tile_range(b, bid)
    res += [qt_start, n_qt]
    res.append(mask.tile_class(b, tid, bid))
    res += list(mask.interior_q_range(b, bid, qt_start, n_qt))
    row = mask.row(b, tid)
    res.append(mask.keep(b, row, bid))
    res.append(mask.cell_keep(b, tid, bid))
    limits = mask.row_limits(b, tid)
    res += list(limits)
    ok = b.cmp_lt(bid, tid)
    limits_ok = mask.row_limits(b, tid, ok)
    res += list(limits_ok)
    res.append(mask.keep_in(b, limits, bid))
    res.append(mask.keep_in(b, limits_ok, bid))

    # Free functions with the flags the policies do not pass.
    qt0, n0 = q_tile_range(b, band, bid, k_n0=kn, k_m0=km, uniform=False)
    res += [qt0, n0]
    res.append(tile_inside_band(b, band, tid, bid, tid, bid))
    res.append(is_edge_tile(b, band, tid, bid, k_m0=km, k_n0=kn))
    res.append(is_edge_tile(b, band, tid, bid, k_m0=km, k_n0=kn, masked=False))
    res.append(tile_class(b, band, tid, bid, k_m0=km, k_n0=kn, uniform=False))
    res.append(
        tile_class(b, band, tid, bid, k_m0=km, k_n0=kn, masked=False, uniform=False)
    )
    res += list(
        interior_q_range(
            b, band, bid, qt0, n0, k_m0=km, k_n0=kn, masked=False, uniform=False
        )
    )
    res += list(
        interior_q_range(b, band, bid, qt0, n0, k_m0=km, k_n0=kn, uniform=False)
    )
    res += list(dkv_store_rule(b, tid, len_kv, seq_mode="batched", s_kv_max=s_kv_max))
    res += list(dkv_store_rule(b, tid, len_kv, seq_mode="batched", s_kv_max=4096))
    res += list(dkv_store_rule(b, tid, len_kv, seq_mode="thd"))

    # Keep every result live: one store per value.
    for i, v in enumerate(res):
        if v.type.name == "i1":
            v = b.zext(v, I32)
        b.global_store(out, b.const_i32(i), v)
    b.ret()
    return b.kernel


def main() -> int:
    return run_emit(
        _spec,
        build_attention_bwd_band_kernel,
        usage="usage: attention_bwd_band_emit.py <config_index> [ll|ir|verify]\n",
    )


if __name__ == "__main__":
    raise SystemExit(main())
