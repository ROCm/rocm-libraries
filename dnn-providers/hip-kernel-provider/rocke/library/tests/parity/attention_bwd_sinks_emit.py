#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# tests/parity/attention_bwd_sinks_emit.py -- Python reference emitter for the
# dQ sinks of the attention backward tile step
# (kernels/common/_attention_bwd_sinks.py). The module has no kernel of its own,
# so this builds a probe kernel per config that drives the sinks the way the
# backward kernels do; the C twin attention_bwd_sinks_emit.c builds the same
# kernel through helper_rocke.instances.common._attention_bwd_sinks.h and the
# two .ll must match.
#
# Configs are decoded from argv[1] as an index into _CONFIGS, built by the same
# loops in the C twin (keep both in sync). Sections, in order:
#   atomic    (arch, sink, ws layout, seq mode, clamp, scale, slot rows, reuse,
#             head size): AtomicWorkspaceSink and PackedAtomicWorkspaceSink over
#             batched and THD sequences, both workspace layouts, int / SSA /
#             mixed slot rows and columns, with and without the scale and the
#             row clamp; reuse 0 consumes twice with one head, 1 with two heads
#             (a second rebase), 2 calls begin_head first (the caller that
#             hoists the rebase)
#   register  (arch, dtype, seq mode, scale, stage_vec, strides, rows, variant):
#             RegisterAccumulateSink (one or two consumes, state / set_state,
#             an empty fragment) plus the rejects (fragment shape, q block or
#             head changed between consumes, a state of the wrong length)
#   null      NullSink (consume rejects), SplitSliceSink (constructing rejects)
#             and split_slice_workspace_bytes (the value goes into the kernel as
#             a constant; a negative argument rejects)
#   pending   PendingDq: names, flat and from_flat through a loop-carried
#             partial (the names are the loop's iteration names) and a dQ sink
#             fed from the carried values, as the deferred sink of
#             sched = "stage_table_atomics" does; plus a from_flat of the wrong
#             length
# Head sizes, layouts, sinks and seq modes are those of the kernels that build
# sinks (attention_bwd.py, attention_bwd_unified.py: head_pack picks the packed
# sink, dq_mode "split" the NullSink); dtypes are the backward I/O dtypes.
#
# argv[2] selects the output: ll (default), ir, verify (see _emit_common.py).
from _emit_common import run_emit
from kernels.common._attention_bwd_addr import (
    PackedRows,
    StridedTensorAddr,
    WorkspaceAddr,
    decode_batched,
    decode_thd,
)
from kernels.common._attention_bwd_sinks import (
    AtomicWorkspaceSink,
    DqSlot,
    NullSink,
    PackedAtomicWorkspaceSink,
    PendingDq,
    RegisterAccumulateSink,
    SplitSliceSink,
    split_slice_workspace_bytes,
)
from rocke.core.ir import BF16, F16, F32, I32, I64, IRBuilder, PtrType

ARCHES = ("gfx942", "gfx950")
SINKS = ("atomic", "packed")
LAYOUTS = ("head_major", "token_major")
MODES = ("batched", "thd")
ROWS = ("int", "value", "mixed")
HEADS = (64, 128)
REG_DTYPES = ("f16", "bf16")
REG_STRIDES = ("int", "value")
REG_ROWS = ("int", "value")
REG_VARIANTS = ("one", "two", "state", "empty")
REG_REJECTS = ("shape_changed", "q0_changed", "head_changed", "state_len")
NULL_VARIANTS = (
    "consume",
    "ok",
    "split_slice",
    "bytes_ok",
    "bytes_round",
    "bytes_neg",
    "bytes_zero",
)
PENDING_NS = (4, 8)


def _configs():
    out = []
    k = 0
    for sink in SINKS:
        for layout in LAYOUTS:
            for mode in MODES:
                for clamp in (1, 0):
                    for scale in (0, 1):
                        for rows in ROWS:
                            out.append(
                                (
                                    "atomic",
                                    ARCHES[k % 2],
                                    sink,
                                    layout,
                                    mode,
                                    clamp,
                                    scale,
                                    rows,
                                    k % 3,
                                    HEADS[(k // 2) % 2],
                                )
                            )
                            k += 1
    k = 0
    for dt in REG_DTYPES:
        for mode in MODES:
            for scale in (0, 1):
                for sv in (8, 1):
                    for strides in REG_STRIDES:
                        for rows in REG_ROWS:
                            variant = REG_VARIANTS[k % len(REG_VARIANTS)]
                            out.append(
                                (
                                    "register",
                                    ARCHES[k % 2],
                                    dt,
                                    mode,
                                    scale,
                                    sv,
                                    strides,
                                    rows,
                                    variant,
                                )
                            )
                            k += 1
    for i, variant in enumerate(REG_REJECTS):
        out.append(
            (
                "register",
                ARCHES[i % 2],
                REG_DTYPES[i % 2],
                MODES[(i // 2) % 2],
                i % 2,
                8,
                "int",
                REG_ROWS[i % 2],
                variant,
            )
        )
    for i, variant in enumerate(NULL_VARIANTS):
        out.append(("null", ARCHES[i % 2], variant))
    for i, n in enumerate(PENDING_NS):
        for nh in (0, 1):
            out.append(("pending", ARCHES[(i + nh) % 2], n, nh, "ok"))
    out.append(("pending", "gfx950", 4, 1, "bad_len"))
    return tuple(out)


_CONFIGS = _configs()


def _spec(idx: int):
    if not 0 <= idx < len(_CONFIGS):
        raise SystemExit(f"unknown config index {idx}")
    cfg = _CONFIGS[idx]
    return cfg, cfg[1]


class _Params:
    """The kernel parameters every probe shares."""

    def __init__(self, b, elem=F32):
        self.dq = b.param("dq", PtrType(F32, "global"))
        self.cu = b.param("cu", PtrType(I32, "global"))
        self.src = b.param("src", PtrType(F32, "global"))
        self.outp = b.param("outp", PtrType(elem, "global"))
        self.s_max = b.param("s_max", I32)
        self.ws_rows = b.param("ws_rows", I32)
        self.n_heads = b.param("n_heads", I32)
        self.h = b.param("h", I32)
        self.h2 = b.param("h2", I32)
        self.scale = b.param("scale", F32)
        self.q0 = b.param("q0", I32)
        self.s_t = b.param("s_t", I32)
        self.pk_head0 = b.param("pk_head0", I32)
        self.pk_n = b.param("pk_n", I32)
        self.pk_rph = b.param("pk_rph", I32)
        self.pk_row0 = b.param("pk_row0", I32)


def _seq(b, mode, p):
    bid = b.block_id_x()
    if mode == "batched":
        return decode_batched(b, bid, s_max=p.s_max)
    return decode_thd(b, bid, s_max=p.s_max, off_ptr=p.cu)


def _frag(b, p, kind, shift, cols):
    """A dQ fragment: slot rows are build-time ints, SSA values or a mix."""
    tid = b.thread_id_x()
    col_v = b.mul(tid, b.const_i32(8))
    row_vals = {}
    slots = []
    n = 0
    for r in (0, 1, 4, 5):
        for c in cols:
            if kind == "int":
                row = r + shift
            elif kind == "value":
                if r not in row_vals:
                    row_vals[r] = b.add(tid, b.const_i32(r + shift))
                row = row_vals[r]
            else:
                if r % 2 == 0:
                    row = r + shift
                else:
                    if r not in row_vals:
                        row_vals[r] = b.add(tid, b.const_i32(r + shift))
                    row = row_vals[r]
            col = col_v if (kind != "int" and n == 3) else c
            v = b.global_load_f32(p.src, b.const_i32(n + 16 * shift))
            slots.append(DqSlot(row=row, col=col, value=v))
            n += 1
    return slots


def _build_atomic(b, spec):
    _s, _arch, sink_kind, layout, mode, clamp, scale, rows_kind, reuse, d = spec
    p = _Params(b)
    seq = _seq(b, mode, p)
    ws = WorkspaceAddr(
        b, p.dq, ws_rows=p.ws_rows, width=d, row0=seq.ws_row0, layout=layout,
        n_heads=p.n_heads,
    )  # fmt: skip
    cls = PackedAtomicWorkspaceSink if sink_kind == "packed" else AtomicWorkspaceSink
    sink = cls(b, ws, seq=seq, scale=p.scale if scale else None)
    cols = (0, 8, 16)
    frag = _frag(b, p, rows_kind, 0, cols)
    frag2 = _frag(b, p, rows_kind, 2, cols)
    if reuse == 2:
        sink.begin_head(b, p.h)
    if sink_kind == "packed":
        rows = PackedRows(p.pk_head0, p.pk_n, p.pk_rph, p.pk_row0)
        sink.consume(b, frag, rows, clamp=bool(clamp))
        sink.consume(b, frag2, rows, clamp=bool(clamp))
    else:
        h_b = p.h2 if reuse == 1 else p.h
        sink.consume(b, frag, p.q0, p.h, clamp=bool(clamp))
        sink.consume(b, frag2, p.q0, h_b, clamp=bool(clamp))
    sink.finish(b)


def _build_register(b, spec):
    _s, _arch, dt, mode, scale, sv, strides_kind, rows_kind, variant = spec
    elem = F16 if dt == "f16" else BF16
    p = _Params(b, elem)
    seq = _seq(b, mode, p)
    s_t = 512 if strides_kind == "int" else p.s_t
    out = StridedTensorAddr(
        b, p.outp, dtype=elem, strides=(65536, 64, s_t), seq=seq, stage_vec=sv,
    )  # fmt: skip
    sink = RegisterAccumulateSink(b, out, scale=p.scale if scale else None)
    cols = (0, 8)
    frag = _frag(b, p, rows_kind, 0, cols)
    # the same rows and columns (the sink keys them by identity), new values
    frag2 = [
        DqSlot(
            row=s.row,
            col=s.col,
            value=b.global_load_f32(p.src, b.const_i32(32 + i)),
        )
        for i, s in enumerate(frag)
    ]
    if variant == "empty":
        sink.consume(b, [], p.q0, p.h)
    else:
        sink.consume(b, frag, p.q0, p.h)
    if variant == "two":
        sink.consume(b, frag2, p.q0, p.h)
    elif variant == "state":
        st = sink.state()
        sink.set_state([b.fmul(v, p.scale) for v in st])
    elif variant == "shape_changed":
        sink.consume(b, frag2[:-1], p.q0, p.h)
    elif variant == "q0_changed":
        sink.consume(b, frag2, b.add(p.q0, b.const_i32(1)), p.h)
    elif variant == "head_changed":
        sink.consume(b, frag2, p.q0, p.h2)
    elif variant == "state_len":
        sink.set_state(sink.state()[:-1])
    sink.finish(b)


def _build_null(b, spec):
    _s, _arch, variant = spec
    out = b.param("out", PtrType(I64, "global"))
    if variant == "consume":
        sink = NullSink()
        sink.consume(b, [], b.const_i32(0), b.const_i32(0))
    elif variant == "ok":
        sink = NullSink()
        sink.begin_head(b, b.const_i32(0))
        sink.finish(b)
        b.global_store(out, b.const_i32(0), b.const_i64(int(sink.emits_g4)))
    elif variant == "split_slice":
        SplitSliceSink()
    elif variant == "bytes_ok":
        n = split_slice_workspace_bytes(
            n_split=3, batch=2, s_q=100, h_q=8, head_size=64
        )
        b.global_store(out, b.const_i32(0), b.const_i64(n))
    elif variant == "bytes_round":
        n = split_slice_workspace_bytes(n_split=1, batch=1, s_q=1, h_q=1, head_size=3)
        b.global_store(out, b.const_i32(0), b.const_i64(n))
    elif variant == "bytes_zero":
        # a zero factor after an enormous product is still 0 (Python ints do not overflow)
        n = split_slice_workspace_bytes(
            n_split=1 << 62, batch=4, s_q=0, h_q=8, head_size=64
        )
        b.global_store(out, b.const_i32(0), b.const_i64(n))
    else:
        split_slice_workspace_bytes(n_split=-1, batch=2, s_q=100, h_q=8, head_size=64)


def _build_pending(b, spec):
    _s, _arch, n, with_nh, variant = spec
    p = _Params(b)
    tid = b.thread_id_x()
    values = tuple(b.global_load_f32(p.src, b.const_i32(i)) for i in range(n))
    pd = PendingDq(
        values=values,
        q0=p.q0,
        h=p.h,
        n_heads=p.pk_n if with_nh else None,
        rows_per_head=p.pk_rph if with_nh else None,
    )
    zero = b.const_i32(0)
    trip = b.add(tid, b.const_i32(2))
    loop = b.scf_for_iter(
        zero, trip, b.const_i32(1), list(zip(pd.names("p_"), pd.flat())), iv_name="qi"
    )
    with loop as (_iv, cur):
        flat = list(cur)
        if variant == "bad_len":
            flat = flat[:-1]
        nxt = pd.from_flat(flat)
        b.scf_yield(*nxt.flat())
    done = pd.from_flat(list(loop.results))
    seq = decode_batched(b, b.block_id_x(), s_max=p.s_max)
    ws = WorkspaceAddr(
        b, p.dq, ws_rows=p.ws_rows, width=64, row0=seq.ws_row0, layout="head_major",
        n_heads=p.n_heads,
    )  # fmt: skip
    sink = AtomicWorkspaceSink(b, ws, seq=seq, scale=p.scale)
    slots = [
        DqSlot(row=i // 2, col=8 * (i % 2), value=v) for i, v in enumerate(done.values)
    ]
    sink.consume(b, slots, done.q0, done.h)
    sink.finish(b)


_BUILD = {
    "atomic": _build_atomic,
    "register": _build_register,
    "null": _build_null,
    "pending": _build_pending,
}


def build_attention_bwd_sinks_kernel(spec, *, arch: str):
    """The probe kernel of one config."""
    b = IRBuilder("attn_bwd_sinks_probe")
    _BUILD[spec[0]](b, spec)
    b.ret()
    return b.kernel


def main() -> int:
    return run_emit(
        _spec,
        build_attention_bwd_sinks_kernel,
        usage="usage: attention_bwd_sinks_emit.py <config_index> [ll|ir|verify]\n",
    )


if __name__ == "__main__":
    raise SystemExit(main())
