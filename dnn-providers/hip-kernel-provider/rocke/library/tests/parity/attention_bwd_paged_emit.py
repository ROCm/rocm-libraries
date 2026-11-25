#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# tests/parity/attention_bwd_paged_emit.py -- Python reference emitter for the
# paged address generator of the unified attention backward main kernel
# (kernels/common/_attention_bwd_paged.py). The module has no kernel of its
# own, so this builds a probe kernel per config that drives PagedKv the way
# attention_bwd_unified.py does (the kv sequence decoded with a SEQ_LENS_KV
# read, one PagedKv per kernel, K / V / dK / dV caches sharing its tiles, the
# page workspace rows of the atomic epilogue); the C twin
# attention_bwd_paged_emit.c builds the same kernel through
# helper_rocke.instances.common._attention_bwd_paged.h and the two .ll must
# match.
#
# Configs are decoded from argv[1] as an index into _CONFIGS, built by the same
# loops in the C twin (keep both in sync): every base tile (page class, kN0,
# page argument, block-id guard, block-table stride kind) on both arches with
# the cache access knobs (head, row clamp, dtype, stage_vec, stride kind,
# vector width, column) cycling over the combinations, then the rejects.
# Base tiles: the "multi" class at kN0 16 / 64 / 128 (the fallback, k64 and
# default tiles of attn_bwd_unified_tile_configs) with a runtime or build-time
# page, and the "sub" class at (kN0, sub_page) = (128, 16 / 32 / 64) and
# (256, 128) (SHIPPED_SUB_PAGES), each with and without page_rows (the guard)
# and with a runtime or build-time block-table stride. The probe calls
# live_blocks, PagedKv.tile (twice for one k0 and for a second tile),
# PagedKvTile.locate / checked / store_rule / workspace_row (and its row_slab
# closure: atomic_add and a store on the slab), PagedKv.cache,
# PagedCacheRows.slab, and PagedSlab.load_vec / store / store_vec.
#
# argv[2] selects the output: ll (default), ir, verify (see _emit_common.py).
from _emit_common import run_emit
from kernels.common._attention_bwd_addr import decode_batched
from kernels.common._attention_bwd_paged import PagedKv, live_blocks
from rocke.core.ir import BF16, F16, F32, I32, I64, IRBuilder, PtrType

ARCHES = ("gfx942", "gfx950")
MULTI_KNS = (16, 64, 128)
SUB_TILES = ((128, 16), (128, 32), (128, 64), (256, 128))
HEADS = ("zero", "int", "value")
DTYPES = ("f16", "bf16", "f32")
SVS = (8, 1)
VECS = (8, 16, 4, 2)
COLS = (0, 8, "value")
# (arch, kn, sub_page (0: None), page kind, guard, bt kind, head, clamp, dtype,
#  stage_vec, strides kind, vec width, column, workspace width)
_REJECTS = (
    # the sub page must be a multiple of 16 dividing kN0 and below it
    ("gfx950", 128, 24, "sub", 0, "value", "zero", 1, "f16", 8, "value", 8, 0, 64),
    ("gfx942", 128, 48, "sub", 1, "int", "value", 1, "bf16", 8, "int", 8, 0, 64),
    ("gfx950", 64, 64, "sub", 0, "value", "int", 0, "f16", 8, "value", 8, 8, 64),
    # the sub class takes the compile-time page
    ("gfx942", 128, 32, "value", 0, "value", "zero", 1, "f16", 8, "value", 8, 0, 64),
    ("gfx950", 128, 32, "int", 1, "int", "zero", 1, "f16", 8, "value", 8, 0, 64),
    # stage_vec, dtype and workspace width the slabs refuse
    ("gfx942", 64, 0, "value", 0, "value", "zero", 1, "f16", 4, "value", 8, 0, 64),
    ("gfx950", 64, 0, "value", 1, "value", "int", 1, "i64", 8, "int", 8, 0, 64),
    ("gfx942", 64, 0, "int", 0, "int", "value", 0, "bf16", 8, "int", 8, 8, 0),
    ("gfx950", 128, 64, "sub", 1, "int", "value", 1, "f16", 8, "int", 8, 0, 0),
)
# int strides 0 and 1 take the _scaled32 / _scaled64 shortcuts (kinds "unit",
# "zero"); also bt strides 0 and 1 are not reachable through the probe's bt kinds.
_STRIDES = (
    ("gfx942", 64, 0, "value", 0, "value", "int", 1, "f16", 8, "unit", 8, 8, 64),
    ("gfx950", 128, 0, "int", 1, "int", "value", 0, "bf16", 1, "zero", 4, 0, 64),
    ("gfx942", 128, 32, "sub", 0, "value", "zero", 1, "f16", 8, "unit", 2, "value", 64),
    ("gfx950", 64, 0, "value", 1, "value", "value", 0, "f32", 8, "zero", 8, 8, 64),
)


def _configs():
    bases = []
    for kn in MULTI_KNS:
        for guard in (0, 1):
            for pk in ("value", "int"):
                for bt in ("value", "int"):
                    bases.append((kn, 0, pk, guard, bt))
    for kn, sub in SUB_TILES:
        for guard in (0, 1):
            for bt in ("value", "int"):
                bases.append((kn, sub, "sub", guard, bt))
    out = []
    i = 0
    for arch in ARCHES:
        for base in bases:
            out.append(
                (
                    arch,
                    *base,
                    HEADS[i % 3],
                    (i // 3) % 2,
                    DTYPES[(i // 2) % 3],
                    SVS[(i // 5) % 2],
                    ("value", "int")[(i // 4) % 2],
                    VECS[(i // 3) % 4],
                    COLS[(i // 2) % 3],
                    64,
                )
            )
            i += 1
    out.extend(_REJECTS)
    out.extend(_STRIDES)
    return tuple(out)


_CONFIGS = _configs()


def _spec(idx: int):
    if not 0 <= idx < len(_CONFIGS):
        raise SystemExit(f"unknown config index {idx}")
    cfg = _CONFIGS[idx]
    return cfg, cfg[0]


_DT = {"f16": F16, "bf16": BF16, "f32": F32, "i64": I64}


def build_attention_bwd_paged_kernel(spec, *, arch: str):
    """The probe kernel of one config."""
    (_arch, kn, sub, page_kind, guard, bt_kind, head_kind, clamp, dt, sv,
     stride_kind, vec, col_kind, ws_width) = spec  # fmt: skip
    elem = _DT[dt]
    b = IRBuilder("attn_bwd_paged_probe")
    bt = b.param("bt", PtrType(I32, "global"))
    kc = b.param("kc", PtrType(elem, "global"))
    dkc = b.param("dkc", PtrType(elem, "global"))
    ws = b.param("ws", PtrType(F32, "global"))
    seq_lens = b.param("seq_lens", PtrType(I32, "global"))
    bt_stride = b.param("bt_stride", I32)
    s_max = b.param("s_max", I32)
    page = b.param("page", I32)
    page_rows = b.param("page_rows", I32)
    kc_blk = b.param("kc_blk", I32)
    kc_tok = b.param("kc_tok", I32)
    kc_h = b.param("kc_h", I32)
    hk_p = b.param("hk", I32)
    col_p = b.param("col", I32)
    val = b.param("val", F32)
    s_idx = b.block_id_y()
    seq_kv = decode_batched(b, s_idx, s_max=s_max, seq_ptr=seq_lens, has_len=1)
    sub_page = sub if sub else None
    page_arg = {"value": page, "int": 2 * kn, "sub": sub}[page_kind]
    paged = PagedKv(
        b,
        bt_ptr=bt,
        bt_stride=bt_stride if bt_kind == "value" else 64,
        seq_idx=s_idx,
        seq=seq_kv,
        k_n0=kn,
        page=page_arg,
        sub_page=sub_page,
        page_rows=page_rows if guard else None,
    )
    k0 = b.mul(b.block_id_x(), b.const_i32(kn))
    tile = paged.tile(k0)
    tile_again = paged.tile(k0)
    assert tile is tile_again
    tid = b.thread_id_x()
    lr = b.mod(tid, b.const_i32(kn))
    pid, slot = tile.locate(lr)
    aid, ok = tile.checked(pid)
    k = b.add(k0, lr)
    store, row_live = tile.store_rule(k)
    lb = live_blocks(b, seq_kv.length, page_arg)
    k1 = b.add(k0, b.const_i32(kn))
    tile2 = paged.tile(k1)
    pid2, slot2 = tile2.locate(lr)
    strides = {
        "value": (kc_blk, kc_tok, kc_h),
        "int": (16384, 128, 64),
        "unit": (1, 0, 1),
        "zero": (0, 1, 0),
    }[stride_kind]
    k_cache = paged.cache(kc, dtype=elem, strides=strides, stage_vec=sv)
    out_cache = paged.cache(dkc, dtype=elem, strides=strides, stage_vec=sv)
    head = {"zero": 0, "int": 3, "value": hk_p}[head_kind]
    col = col_p if col_kind == "value" else col_kind
    slab = k_cache.slab(head, k0, clamp=bool(clamp))
    row = b.add(k0, lr)
    v = slab.load_vec(row, col, vec)
    dst = out_cache.slab(head, k0, clamp=False)
    dst.store(row, col, b.vec_extract(v, 0))
    dst.store_vec(row, col, v)
    ws_pr = page_rows if guard else 8192
    row_slab = tile.workspace_row(ws, page_rows=ws_pr, width=ws_width, head=head)
    rs, rr = row_slab(lr)
    rs.atomic_add(rr, 0, val)
    rs.store(rr, col, val)
    rs2, rr2 = row_slab(b.add(lr, b.const_i32(1)))
    rs2.atomic_add(rr2, 8, val)
    # keep the scalar results live
    out = b.param("out", PtrType(I32, "global"))
    res = [slot, aid, k, pid2, slot2, lb, b.zext(store, I32), b.zext(row_live, I32)]
    if ok is not None:
        res.append(b.zext(ok, I32))
    for i, r in enumerate(res):
        b.global_store(out, b.const_i32(i), r)
    b.ret()
    return b.kernel


def main() -> int:
    return run_emit(
        _spec,
        build_attention_bwd_paged_kernel,
        usage="usage: attention_bwd_paged_emit.py <config_index> [ll|ir|verify]\n",
    )


if __name__ == "__main__":
    raise SystemExit(main())
