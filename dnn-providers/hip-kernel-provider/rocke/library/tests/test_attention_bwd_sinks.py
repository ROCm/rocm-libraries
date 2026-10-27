# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""dQ sinks of the attention backward.

Host evaluation (typed integer builder with numpy buffers) of the atomic
workspace sink (head- and token-major, clamped rows, ``ws_rows`` guard,
optional scale, THD bases past ``2**31`` elements), the register sink, the
null sink and the split-slice stub; IR lowering of an atomic probe at three
flavors (agent scope, monotonic, no branch around an atomic, i64 head
rebase); and the probe on the local device with conflicting atomics.
"""

from __future__ import annotations

import ctypes
import itertools
import re
import struct

import numpy as np
import pytest
from kernels.common import _attention_bwd_sinks as SK
from kernels.common._attention_bwd_addr import (
    StridedBatchedAddr,
    WorkspaceAddr,
    decode_batched,
    decode_thd,
)
from rocke.core.ir import F16, F32, I32, IRBuilder, PtrType

from ._attention_bwd_test_utils import device_arch
from ._attention_bwd_int_eval import IntEvalBuilder, VirtualTensor

ARCHS = ("gfx942", "gfx950", "gfx1151", "gfx1201")
FLAVORS = ("llvm20", "llvm22", "llvm23")


def _frag(b, vals, k_m0, D, *, runtime_rows=False):
    """A full k_m0 x D fragment with values vals[r, c]."""
    out = []
    for r, c in itertools.product(range(k_m0), range(D)):
        row = b.const_i32(r) if runtime_rows else r
        out.append(SK.DqSlot(row=row, col=c, value=b.const_f32(float(vals[r, c]))))
    return out


@pytest.mark.parametrize("layout", ["head_major", "token_major"])
@pytest.mark.parametrize("runtime_rows", [False, True])
def test_atomic_sink_accumulates_like_numpy(layout, runtime_rows):
    rng = np.random.default_rng(0)
    B, H, S, D, k_m0 = 2, 3, 40, 8, 16
    lens = [40, 21]
    ws_rows = B * S
    ws = np.zeros(H * ws_rows * D, np.float32)
    b = IntEvalBuilder()
    p = b.add_buffer("WS_DQ", ws)
    seqp = b.add_buffer("SEQ", np.array(lens, np.int32))
    want = np.zeros((H, ws_rows, D), np.float32)
    for bi in range(B):
        seq = decode_batched(b, b.const_i32(bi), s_max=S, seq_ptr=seqp, has_len=1)
        wa = WorkspaceAddr(
            b, p, ws_rows=ws_rows, width=D, row0=seq.ws_row0, layout=layout,
            n_heads=H if layout == "token_major" else None,
        )  # fmt: skip
        sink = SK.AtomicWorkspaceSink(b, wa, seq=seq)
        for h in range(H):
            hv = b.const_i32(h)
            sink.begin_head(b, hv)
            for qt in range((lens[bi] + k_m0 - 1) // k_m0):
                vals = rng.standard_normal((k_m0, D)).astype(np.float32)
                q = qt * k_m0 + np.arange(k_m0)
                vals[q >= lens[bi]] = 0.0  # rows past len_q carry exact zeros
                sink.consume(
                    b, _frag(b, vals, k_m0, D, runtime_rows=runtime_rows),
                    b.const_i32(qt * k_m0), hv,
                )  # fmt: skip
                for r in range(k_m0):
                    qq = min(qt * k_m0 + r, lens[bi] - 1)
                    want[h, bi * S + qq] += vals[r]
        sink.finish(b)
    got = (
        ws.reshape(H, ws_rows, D)
        if layout == "head_major"
        else (ws.reshape(ws_rows, H, D).transpose(1, 0, 2))
    )
    assert np.allclose(got, want, rtol=0, atol=1e-5)
    atomics = [e for e in b.log if e[0] == "atomic"]
    n_steps = sum((n + k_m0 - 1) // k_m0 for n in lens) * H
    assert len(atomics) == n_steps * k_m0 * D


def test_atomic_sink_scale_and_ws_rows_guard():
    D, k_m0 = 4, 16
    ws_rows = 10  # understated: rows 10.. are outside the workspace
    ws = np.zeros(ws_rows * D + 64, np.float32)  # canary tail
    ws[ws_rows * D :] = 7.0
    b = IntEvalBuilder()
    p = b.add_buffer("WS_DQ", ws)
    offs = np.array([0, 6, 30], np.int32)
    po = b.add_buffer("OFF", offs)
    seq = decode_thd(b, b.const_i32(1), s_max=32, off_ptr=po)  # tok0 = 6, len = 24
    wa = WorkspaceAddr(b, p, ws_rows=ws_rows, width=D, row0=seq.ws_row0)
    sink = SK.AtomicWorkspaceSink(b, wa, seq=seq, scale=b.const_f32(0.5))
    vals = np.ones((k_m0, D), np.float32)
    sink.consume(b, _frag(b, vals, k_m0, D), b.const_i32(0), b.const_i32(0))
    got = ws[: ws_rows * D].reshape(ws_rows, D)
    assert np.all(got[:6] == 0) and np.all(got[6:10] == 0.5)
    assert np.all(ws[ws_rows * D :] == 7.0)


def test_atomic_sink_thd_base_past_2_pow_31_elements():
    H, D, k_m0 = 4, 128, 16
    rows = (1 << 22) + 100  # H * rows * D > 2**31
    b = IntEvalBuilder()
    vt = VirtualTensor(H * rows * D, lambda e: 0.0)
    p = b.add_buffer("WS_DQ", vt, "f32")
    tok0 = rows - 40
    offs = np.array([tok0, tok0 + 40], np.int64).view(np.int32).copy()
    po = b.add_buffer("OFF", offs)
    seq = decode_thd(b, b.const_i32(0), s_max=64, off_ptr=po, off64=1)
    wa = WorkspaceAddr(b, p, ws_rows=b.const_i32(rows), width=D, row0=seq.ws_row0)
    sink = SK.AtomicWorkspaceSink(b, wa, seq=seq)
    vals = np.full((k_m0, D), 2.0, np.float32)
    sink.consume(b, _frag(b, vals, k_m0, D), b.const_i32(32), b.const_i32(H - 1))
    for r in range(k_m0):
        q = min(32 + r, 39)
        e = ((H - 1) * rows + tok0 + q) * D + 5
        assert e > 1 << 31
        assert vt[e] == 2.0 * (9 if q == 39 else 1)


@pytest.mark.parametrize(
    "layout,H,S,D",
    [
        ("head_major", 32, 1 << 20, 128),  # h * S_max * D past 2**31
        ("head_major", 40, (1 << 26) - 1, 32),  # h * S_max itself past 2**31
        ("token_major", 32, 1 << 20, 128),  # S_max * H * D past 2**31
    ],
)
def test_atomic_sink_large_workspace_index_stays_i64(layout, H, S, D):
    # S_max * D stays below 2**31 (the index-range contract); any i32 form of
    # the head or row rebase raises in the typed evaluator.
    k_m0 = 16
    b = IntEvalBuilder()
    vt = VirtualTensor(H * S * D, lambda e: 0.0)
    p = b.add_buffer("WS_DQ", vt, "f32")
    seq = decode_batched(b, b.const_i32(0), s_max=S)
    wa = WorkspaceAddr(
        b, p, ws_rows=b.const_i32(S), width=D, row0=seq.ws_row0, layout=layout,
        n_heads=H if layout == "token_major" else None,
    )  # fmt: skip
    sink = SK.AtomicWorkspaceSink(b, wa, seq=seq)
    h = b.const_i32(H - 1)
    sink.begin_head(b, h)
    q0 = S - 8  # the last step: rows past len_q - 1 clamp to it
    frag = [SK.DqSlot(row=r, col=D - 1, value=b.const_f32(1.0)) for r in range(k_m0)]
    sink.consume(b, frag, b.const_i32(q0), h)
    want = {}
    for r in range(k_m0):
        q = min(q0 + r, S - 1)
        if layout == "head_major":
            e = ((H - 1) * S + q) * D + D - 1
        else:
            e = (q * H + H - 1) * D + D - 1
        want[e] = want.get(e, 0.0) + 1.0
    assert min(want) >= 1 << 31
    assert vt.writes == want


def test_register_sink_accumulates_and_stores_valid_rows():
    B, H, S, D, k_m0 = 1, 2, 20, 8, 16
    out = np.full(B * S * H * D, np.float16(-3.0), np.float16)
    b = IntEvalBuilder()
    p = b.add_buffer("DQ", out)
    seq = decode_batched(b, b.const_i32(0), s_max=S)
    seq = type(seq)(  # valid length 13 of 20
        mode=seq.mode, batch=seq.batch, length=b.const_i32(13), last=b.const_i32(12),
        s_max=seq.s_max, ws_row0=seq.ws_row0,
    )  # fmt: skip
    addr = StridedBatchedAddr(
        b, p, dtype=F16, strides=(S * H * D, D, H * D), seq=seq, stage_vec=8
    )
    sink = SK.RegisterAccumulateSink(b, addr, scale=b.const_f32(2.0))
    h = b.const_i32(1)
    q0 = b.const_i32(4)
    total = np.zeros((k_m0, D), np.float32)
    for it in range(3):
        vals = np.full((k_m0, D), 0.25 * (it + 1), np.float32)
        total += vals
        sink.consume(b, _frag(b, vals, k_m0, D), q0, h)
    assert len(sink.state()) == k_m0 * D
    sink.finish(b)
    view = out.reshape(S, H, D)
    for r in range(k_m0):
        q = 4 + r
        if q < 13:
            assert np.all(view[q, 1] == np.float16(2.0 * total[r, 0]))
        elif q < S:
            assert np.all(view[q, 1] == np.float16(-3.0))
    assert np.all(view[:, 0] == np.float16(-3.0))
    # shape and identity checks
    with pytest.raises(ValueError):
        sink.consume(b, _frag(b, total, k_m0, D)[:-1], q0, h)
    with pytest.raises(ValueError):
        sink.consume(b, _frag(b, total, k_m0, D), b.const_i32(8), h)
    with pytest.raises(ValueError):
        sink.begin_head(b, b.const_i32(0))
    with pytest.raises(ValueError):
        sink.set_state([])


def test_null_and_split_slice_sinks():
    b = IntEvalBuilder()
    ns = SK.NullSink()
    assert ns.emits_g4 is False and SK.AtomicWorkspaceSink.emits_g4 is True
    ns.begin_head(b, b.const_i32(0))
    ns.finish(b)
    with pytest.raises(RuntimeError):
        ns.consume(b, [], b.const_i32(0), b.const_i32(0))
    with pytest.raises(NotImplementedError):
        SK.SplitSliceSink()
    assert (
        SK.split_slice_workspace_bytes(n_split=2, batch=2, s_q=16, h_q=8, head_size=64)
        == 4 * 2 * 2 * 16 * 8 * 64
    )
    assert (
        SK.split_slice_workspace_bytes(n_split=1, batch=1, s_q=1, h_q=1, head_size=32)
        == 256
    )
    with pytest.raises(ValueError):
        SK.split_slice_workspace_bytes(n_split=-1, batch=1, s_q=1, h_q=1, head_size=1)


def test_no_exported_helper_looks_like_a_builder():
    assert not [n for n in SK.__all__ if n.startswith("build_")]


# ---------------------------------------------------------------------------
# IR probe: one CTA per (head, batch, q tile, repeat); 16 x D fragment
# ---------------------------------------------------------------------------


def _probe_kernel(name, *, D, wave, layout="head_major", scale=False):
    b = IRBuilder(name)
    ws = b.param("WS_DQ", PtrType(F32, "global"))
    seqp = b.param("SEQ", PtrType(I32, "global"), readonly=True)
    s_max = b.param("S_max", I32)
    ws_rows = b.param("ws_rows", I32)
    n_heads = b.param("H", I32)
    n_qt = b.param("n_qt", I32)
    dq_mult = b.param("dq_mult", F32)
    h = b.block_id_x()
    bi = b.block_id_y()
    qt = b.mod(b.block_id_z(), n_qt)
    seq = decode_batched(b, bi, s_max=s_max, seq_ptr=seqp, has_len=1)
    wa = WorkspaceAddr(
        b, ws, ws_rows=ws_rows, width=D, row0=seq.ws_row0, layout=layout,
        n_heads=n_heads if layout == "token_major" else None,
    )  # fmt: skip
    sink = SK.AtomicWorkspaceSink(b, wa, seq=seq, scale=dq_mult if scale else None)
    sink.begin_head(b, h)
    lane = b.thread_id_x()
    row = b.mod(lane, b.const_i32(16))
    cgrp = b.div(lane, b.const_i32(16))
    per = wave // 16
    q0 = b.mul(qt, b.const_i32(16))
    q = b.add(q0, row)
    live = b.cmp_lt(q, seq.length)
    frag = []
    for j in range(D // per):
        col = b.add(cgrp, b.const_i32(per * j))
        # value: q * D + col + 1000 * h, zero on rows past len_q
        v = b.sitofp_f32(
            b.add(b.add(b.mul(q, b.const_i32(D)), col), b.mul(h, b.const_i32(1000)))
        )
        frag.append(
            SK.DqSlot(row=row, col=col, value=b.select(live, v, b.const_f32(0.0)))
        )
    sink.consume(b, frag, q0, h)
    sink.finish(b)
    b.ret()
    b.kernel.attrs["max_workgroup_size"] = 64
    return b.kernel


def _lower(kernel, arch, flavor):
    from rocke.core.lower_llvm import _lower_kernel_to_llvm_python as lower

    return lower(kernel, arch=arch, llvm_flavor=flavor)


@pytest.mark.parametrize("flavor", FLAVORS)
@pytest.mark.parametrize("arch", ARCHS)
@pytest.mark.parametrize("layout", ["head_major", "token_major"])
def test_atomic_probe_lowers(arch, flavor, layout):
    wave = 64 if arch in ("gfx942", "gfx950") else 32
    ir = _lower(
        _probe_kernel(f"bwd_sink_{layout}", D=32, wave=wave, layout=layout, scale=True),
        arch,
        flavor,
    )
    atomics = re.findall(r"atomicrmw fadd ptr addrspace\(1\) [^\n]*", ir)
    assert len(atomics) == 32 * 16 // wave
    assert all('syncscope("agent") monotonic' in a for a in atomics)
    assert "alloca" not in ir and "cmpxchg" not in ir
    body = ir.split("define amdgpu_kernel")[1]
    # branch-free: the only control flow is the 0/1-trip SEQ guard before the atomics
    first_atomic = body.index("atomicrmw")
    assert "br i1" not in body[first_atomic:]
    # the workspace pointer is a byte GEP with an i64 offset off the kernarg
    assert re.search(r"getelementptr inbounds i8, ptr addrspace\(1\) %WS_DQ, i64", ir)
    # the per-atomic index is i32
    assert re.findall(
        r"getelementptr inbounds float, ptr addrspace\(1\) [^,]+, i32", ir
    )


def _register_probe(name, *, D):
    """Two kv tiles' worth of fragments into a register sink, one store."""
    b = IRBuilder(name)
    dq = b.param("DQ", PtrType(F16, "global"))
    s_max = b.param("S_max", I32)
    seq = decode_batched(b, b.block_id_y(), s_max=s_max)
    addr = StridedBatchedAddr(
        b, dq, dtype=F16, strides=(b.param("s_b", I32), D, b.param("s_t", I32)),
        seq=seq, stage_vec=8,
    )  # fmt: skip
    sink = SK.RegisterAccumulateSink(b, addr, scale=b.param("scale", F32))
    h = b.block_id_x()
    lane = b.thread_id_x()
    q0 = b.const_i32(0)
    for kt in range(2):
        v = b.sitofp_f32(b.add(lane, b.const_i32(kt)))
        frag = [SK.DqSlot(row=lane, col=c, value=v) for c in range(0, D, 16)]
        sink.consume(b, frag, q0, h)
    sink.finish(b)
    b.ret()
    b.kernel.attrs["max_workgroup_size"] = 64
    return b.kernel


@pytest.mark.parametrize("flavor", FLAVORS)
@pytest.mark.parametrize("arch", ARCHS)
def test_register_probe_lowers(arch, flavor):
    ir = _lower(_register_probe("bwd_sink_reg", D=64), arch, flavor)
    assert "alloca" not in ir and "atomicrmw" not in ir
    # one fadd per slot after the first tile, one predicated half store per slot
    assert len(re.findall(r"= fadd float", ir)) == 64 // 16
    assert len(re.findall(r"store half", ir)) == 64 // 16


@pytest.mark.gpu
@pytest.mark.parametrize("layout", ["head_major", "token_major"])
def test_atomic_probe_on_device(layout):
    arch = device_arch()
    if arch not in ("gfx942", "gfx950", "gfx1151", "gfx1201"):
        pytest.skip(f"needs gfx942/gfx950/gfx1151/gfx1201; device is {arch}")
    from rocke.core.arch import ArchTarget
    from rocke.helpers.compile import compile_kernel
    from rocke.runtime.hip_module import Runtime

    wave = ArchTarget.from_gfx(arch).wave_size
    D, H, S, B, reps = 64, 3, 45, 3, 4
    lens = np.array([45, 20, 0], np.int32)
    ws_rows = B * S - 10  # the last batch's tail rows fall outside the workspace
    n_qt = (S + 15) // 16
    kern = _probe_kernel(
        f"bwd_sink_dev_{layout}", D=D, wave=wave, layout=layout, scale=True
    )
    art = compile_kernel(kern, arch=arch, capture_ir_text=False, backend="python")
    canary = 64
    ws = np.zeros(H * ws_rows * D + canary, np.float32)
    ws[-canary:] = 123.0
    rt = Runtime()
    mod = rt.load_module(art.hsaco)
    fn = mod.get_function(art.kernel_name)

    def up(a):
        a = np.ascontiguousarray(a)
        d = rt.alloc(max(a.nbytes, 16))
        rt.memcpy_h2d(d, (ctypes.c_uint8 * a.nbytes).from_buffer(a), a.nbytes)
        return d

    d_ws, d_seq = up(ws), up(lens)
    args = struct.pack("<QQiiiif", d_ws, d_seq, S, ws_rows, H, n_qt, 0.5)
    rt.launch(fn, (H, B, n_qt * reps), (wave, 1, 1), args)
    rt.sync()
    rt.memcpy_d2h((ctypes.c_uint8 * ws.nbytes).from_buffer(ws), d_ws, ws.nbytes)
    rt.free(d_ws)
    rt.free(d_seq)
    mod.unload()
    want = np.zeros((H, ws_rows, D), np.float64)
    for bi, h in itertools.product(range(B), range(H)):
        for q in range(int(lens[bi])):
            row = bi * S + q
            if row < ws_rows:
                want[h, row] = reps * 0.5 * (q * D + np.arange(D) + 1000 * h)
    body = ws[:-canary]
    got = (
        body.reshape(H, ws_rows, D)
        if layout == "head_major"
        else (body.reshape(ws_rows, H, D).transpose(1, 0, 2))
    )
    assert np.array_equal(got.astype(np.float64), want)
    assert np.all(ws[-canary:] == 123.0)


@pytest.mark.parametrize("layout", ["head_major", "token_major"])
def test_packed_sink_rebases_per_row_and_zeroes_dead_rows(layout):
    from kernels.common._attention_bwd_addr import PackedRows

    rng = np.random.default_rng(1)
    B, H, S, D, k_m0 = 2, 8, 3, 8, 32
    lens = [3, 2]
    ws_rows = B * S
    ws = np.zeros(H * ws_rows * D, np.float32)
    b = IntEvalBuilder()
    p = b.add_buffer("WS_DQ", ws)
    seqp = b.add_buffer("SEQ", np.array(lens, np.int32))
    want = np.zeros((H, ws_rows, D), np.float32)
    per = 6  # heads packed per q step: groups [0, 6) and [6, 8)
    for bi in range(B):
        seq = decode_batched(b, b.const_i32(bi), s_max=S, seq_ptr=seqp, has_len=1)
        wa = WorkspaceAddr(
            b, p, ws_rows=ws_rows, width=D, row0=seq.ws_row0, layout=layout,
            n_heads=H if layout == "token_major" else None,
        )  # fmt: skip
        sink = SK.PackedAtomicWorkspaceSink(b, wa, seq=seq, scale=b.const_f32(2.0))
        lq = lens[bi]
        for g0 in range(0, H, per):
            nh = min(per, H - g0)
            rows = PackedRows(
                head0=b.const_i32(g0), n_heads=b.const_i32(nh),
                rows_per_head=b.const_i32(lq), row0=b.const_i32(0),
            )  # fmt: skip
            sink.begin_head(b, b.const_i32(g0))
            vals = rng.standard_normal((k_m0, D)).astype(np.float32)
            sink.consume(b, _frag(b, vals, k_m0, D, runtime_rows=True), rows)
            for r in range(k_m0):
                hh, q = divmod(r, lq)
                if hh < nh:
                    want[g0 + hh, bi * S + q] += 2.0 * vals[r]
    if layout == "head_major":
        got = ws.reshape(H, ws_rows, D)
    else:
        got = ws.reshape(ws_rows, H, D).transpose(1, 0, 2)
    np.testing.assert_allclose(got, want, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("layout", ["head_major", "token_major"])
@pytest.mark.parametrize("packed", [False, True])
def test_unclamped_steps_inside_the_sequence_add_the_same_values(layout, packed):
    """``clamp=False`` (an interior q step: every row ``< len_q``) drops the
    ``min(q, len_q - 1)`` row clamps and adds the same values at the same
    workspace elements as the clamped sink."""
    from kernels.common._attention_bwd_addr import PackedRows

    B, H, S, D, k_m0 = 2, 2, 48, 4, 16
    lens = [48, 37]
    ws_rows = B * S
    runs = []
    for clamp in (True, False):
        rng = np.random.default_rng(3)
        ws = np.zeros(H * ws_rows * D, np.float32)
        b = IntEvalBuilder()
        n_smin = [0]
        smin = b.smin

        def counted(x, y, smin=smin, n=n_smin):
            n[0] += 1
            return smin(x, y)

        b.smin = counted
        p = b.add_buffer("WS_DQ", ws)
        seqp = b.add_buffer("SEQ", np.array(lens, np.int32))
        for bi in range(B):
            seq = decode_batched(b, b.const_i32(bi), s_max=S, seq_ptr=seqp, has_len=1)
            wa = WorkspaceAddr(
                b, p, ws_rows=ws_rows, width=D, row0=seq.ws_row0, layout=layout,
                n_heads=H if layout == "token_major" else None,
            )  # fmt: skip
            cls = SK.PackedAtomicWorkspaceSink if packed else SK.AtomicWorkspaceSink
            sink = cls(b, wa, seq=seq, scale=b.const_f32(0.5))
            for h in range(H):
                hv = b.const_i32(h)
                sink.begin_head(b, hv)
                for qt in range(lens[bi] // k_m0):  # interior: whole tiles only
                    q0 = b.const_i32(qt * k_m0)
                    rows = q0
                    if packed:  # packing off: the identity row map of one head
                        rows = PackedRows(
                            head0=hv, n_heads=b.const_i32(1),
                            rows_per_head=b.const_i32(k_m0), row0=q0,
                        )  # fmt: skip
                    vals = rng.standard_normal((k_m0, D)).astype(np.float32)
                    frag = _frag(b, vals, k_m0, D, runtime_rows=True)
                    sink.consume(b, frag, rows, hv, clamp=clamp)
        runs.append(([e for e in b.log if e[0] == "atomic"], ws, n_smin[0]))
    (log_c, ws_c, smin_c), (log_u, ws_u, smin_u) = runs
    assert log_c and log_c == log_u
    assert np.array_equal(ws_c, ws_u)
    assert smin_u < smin_c  # the row clamps are gone
