#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# tests/parity/attention_bwd_valu_emit.py -- Python reference emitter for the
# softmax-gradient VALU levers of the attention backward main kernel
# (kernels/common/_attention_bwd_valu.py). The module has no kernel of its
# own, so this builds a probe kernel per config that calls the entry points the
# way the backward body does; the C twin attention_bwd_valu_emit.c builds the
# same kernel through helper_rocke.instances.common._attention_bwd_valu.h and
# the two .ll must match.
#
# Configs are decoded from argv[1] as an index into _CONFIGS, built by the same
# loops in the C twin (keep both in sync). Sections, in order:
#   cast     (arch, n, elem, bf16_cast): cast_c_frag / rne_noguard_bf16 over the
#            C fragment lengths of the atoms (4, 16) and others, both I/O dtypes,
#            both casts, plus the rejects (rne_noguard on f16, an unknown cast,
#            an odd fragment)
#   softmax  (arch, bwd_exp2, stats_fma): softmax_p and stats_rows_neg, plus the
#            unknown bwd_exp2 reject
#   layout   (arch, op, width): check_stats_vec_layout over every f16 / bf16 MFMA
#            atom of the arch and the stats_vec widths (1, 2, 4) plus two widths
#            it rejects; the probe kernel is the same for every accepted config,
#            so this section checks that both engines accept and reject the
#            same (op, width) pairs
#   stats    (arch, width, runs, q0, n_heads, variant): batched_stats_rows over
#            the stats_vec widths 2 and 4 (width 1 never takes this path) with a build-time and a runtime q0 / n_heads,
#            plus the rejects (int ws_rows: the two workspaces do not share
#            their row model, token-major, wide workspace, distinct ws_rows or
#            row0, rows that do not split into runs)
# The shipped stats_vec is 4 with stats_fma on, bwd_exp2 "exact" / "fast" and
# bf16_cast "generic" (tuned tables of dispatch/attention_bwd); atoms are the
# MFMA f16 / bf16 atoms of gfx942 / gfx950.
#
# argv[2] selects the output: ll (default), ir, verify (see _emit_common.py).
from _emit_common import run_emit
from kernels.common._attention_bwd_addr import WorkspaceAddr
from kernels.common._attention_bwd_valu import (
    batched_stats_rows,
    cast_c_frag,
    check_stats_vec_layout,
    rne_noguard_bf16,
    softmax_p,
    stats_rows_neg,
)
from rocke.core.arch.target import ArchTarget
from rocke.core.ir import BF16, F16, F32, I32, I64, IRBuilder, PtrType

ARCHES = ("gfx942", "gfx950")
CAST_NS = (2, 4, 8, 16)
CAST_KINDS = (
    ("f16", "generic"),
    ("bf16", "generic"),
    ("bf16", "rne_noguard"),
    ("f16", "rne_noguard"),
)
EXP2S = ("exact", "fast", "bogus")
LAYOUT_WIDTHS = (1, 2, 4, 3, 8)
STATS_WIDTHS = (2, 4)
STATS_RUNS = (1, 2)
Q0S = ("zero", "int", "value")
N_HEADS = ("value", "int")


def _ops(arch):
    t = ArchTarget.from_gfx(arch)
    out = []
    for dt in ("f16", "bf16"):
        out += t.mma.enumerate(family="mma", a_dtype=dt, b_dtype=dt, c_dtype="fp32")
    return out


def _configs():
    out = []
    for arch in ARCHES:
        for n in CAST_NS:
            for elem, cast in CAST_KINDS:
                out.append(("cast", arch, n, elem, cast))
    out += [
        ("cast", "gfx950", 3, "bf16", "rne_noguard"),
        ("cast", "gfx942", 4, "bf16", "bogus"),
        ("cast", "gfx942", 12, "bf16", "rne_noguard"),
        ("cast", "gfx950", 6, "f16", "generic"),
    ]
    for arch in ARCHES:
        for exp2 in EXP2S:
            for fma in (0, 1):
                out.append(("softmax", arch, exp2, fma))
    for arch in ARCHES:
        for op in range(len(_ops(arch))):
            for w in LAYOUT_WIDTHS:
                out.append(("layout", arch, op, w))
    for arch in ARCHES:
        for w in STATS_WIDTHS:
            for runs in STATS_RUNS:
                for q0 in Q0S:
                    for nh in N_HEADS:
                        out.append(("stats", arch, w, runs, q0, nh, "ok"))
    for variant in (
        "int_ws_rows",
        "token_major",
        "wide_ws",
        "distinct_ws_rows",
        "distinct_row0",
        "bad_rows",
    ):
        out.append(("stats", "gfx950", 4, 2, "value", "value", variant))
        out.append(("stats", "gfx942", 2, 1, "int", "int", variant))
    return tuple(out)


_CONFIGS = _configs()


def _spec(idx: int):
    if not 0 <= idx < len(_CONFIGS):
        raise SystemExit(f"unknown config index {idx}")
    cfg = _CONFIGS[idx]
    return cfg, cfg[1]


def _build_cast(b, spec):
    _s, _arch, n, elem_name, cast = spec
    elem = F16 if elem_name == "f16" else BF16
    src = b.param("src", PtrType(F32, "global"))
    out = b.param("out", PtrType(elem, "global"))
    tid = b.thread_id_x()
    elems = [b.global_load_f32(src, b.add(tid, b.const_i32(i))) for i in range(n)]
    frag = b.vec_pack(elems, F32)
    r = cast_c_frag(b, frag, elem, cast)
    if cast == "rne_noguard" and elem == BF16:
        # the helper called directly, as cast_c_frag does under rne_noguard
        r2 = rne_noguard_bf16(b, frag)
        r = b.vec_concat(r, r2)
        n *= 2
    for i in range(n):
        b.global_store(out, b.const_i32(i), b.vec_extract(r, i))


def _build_softmax(b, spec):
    _s, _arch, exp2, fma = spec
    sv_p = b.param("sv", PtrType(F32, "global"))
    stats_p = b.param("stats", PtrType(F32, "global"))
    scale_log2 = b.param("scale_log2", F32)
    out = b.param("out", PtrType(F32, "global"))
    stats = {}
    for k in range(3):
        lse2 = b.global_load_f32(stats_p, b.const_i32(2 * k))
        dsum = b.global_load_f32(stats_p, b.const_i32(2 * k + 1))
        stats[k] = (lse2, dsum)
    neg = stats_rows_neg(b, stats) if fma else {}
    res = []
    for k in range(3):
        sv = b.global_load_f32(sv_p, b.const_i32(k))
        res.append(
            softmax_p(
                b, sv, scale_log2, stats[k][0], neg_lse2=neg.get(k), bwd_exp2=exp2
            )
        )
    for i, v in enumerate(res):
        b.global_store(out, b.const_i32(i), v)


def _build_layout(b, spec):
    _s, arch, op_i, width = spec
    out = b.param("out", PtrType(I32, "global"))
    check_stats_vec_layout(_ops(arch)[op_i], width)
    b.global_store(out, b.const_i32(0), b.const_i32(width))


def _build_stats(b, spec):
    _s, _arch, width, runs, q0_kind, nh_kind, variant = spec
    lse_p = b.param("lse", PtrType(F32, "global"))
    dsum_p = b.param("dsum", PtrType(F32, "global"))
    out = b.param("out", PtrType(F32, "global"))
    ws_rows = b.param("ws_rows", I32)
    ws_rows2 = b.param("ws_rows2", I32)
    row0 = b.param("row0", I64)
    row0_d = b.param("row0_d", I64)
    n_heads = b.param("n_heads", I32)
    h = b.param("h", I32)
    q0p = b.param("q0", I32)
    tid = b.thread_id_x()
    ws_arg = 1000 if variant == "int_ws_rows" else ws_rows
    layout = "token_major" if variant == "token_major" else "head_major"
    nh_arg = n_heads if nh_kind == "value" else 8
    lse = WorkspaceAddr(
        b, lse_p, ws_rows=ws_arg, width=1, row0=row0, layout=layout, n_heads=nh_arg
    )
    dsum = WorkspaceAddr(
        b,
        dsum_p,
        ws_rows=ws_rows2 if variant == "distinct_ws_rows" else ws_arg,
        width=2 if variant == "wide_ws" else 1,
        row0=row0_d if variant == "distinct_row0" else row0,
        layout=layout,
        n_heads=nh_arg,
    )
    base = b.mul(b.div(tid, b.const_i32(16)), b.const_i32(4))
    n_rows = runs * width + (1 if variant == "bad_rows" else 0)
    rows = [b.add(base, b.const_i32(i + 4 * (i // width))) for i in range(n_rows)]
    q0 = {"zero": 0, "int": 5, "value": q0p}[q0_kind]
    res = batched_stats_rows(b, lse, dsum, h, q0, rows, width=width, n_heads=nh_arg)
    neg = stats_rows_neg(b, dict(enumerate(res)))
    k = 0
    for i, (lse2, dsum_v) in enumerate(res):
        b.global_store(out, b.const_i32(k), lse2)
        b.global_store(out, b.const_i32(k + 1), dsum_v)
        b.global_store(out, b.const_i32(k + 2), neg[i])
        k += 3


_BUILD = {
    "cast": _build_cast,
    "softmax": _build_softmax,
    "layout": _build_layout,
    "stats": _build_stats,
}


def build_attention_bwd_valu_kernel(spec, *, arch: str):
    """The probe kernel of one config."""
    b = IRBuilder("attn_bwd_valu_probe")
    _BUILD[spec[0]](b, spec)
    b.ret()
    return b.kernel


def main() -> int:
    return run_emit(
        _spec,
        build_attention_bwd_valu_kernel,
        usage="usage: attention_bwd_valu_emit.py <config_index> [ll|ir|verify]\n",
    )


if __name__ == "__main__":
    raise SystemExit(main())
