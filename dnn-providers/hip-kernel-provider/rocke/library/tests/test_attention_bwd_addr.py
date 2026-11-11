# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Address generators of the attention backward.

Host evaluation of the emitted address formulas (typed integer builder, so
i32 overflow and out-of-bounds accesses raise) against numpy strides: batched
and THD sequence decode (int32 / int64 offsets, multiplier, lengths), strided
user tensors (BSHD, BHSD, padded strides, packed QKV views, ``stage_vec`` 8 and
1, clamped rows), 64-bit rebases past ``2**31`` elements, the fp32 workspace
(head- and token-major) and paged caches.  Then IR lowering of a probe kernel
at three flavors and the probe on the local device.
"""

from __future__ import annotations

import ctypes
import itertools
import re
import struct

import numpy as np
import pytest
from kernels.common import _attention_bwd_addr as A
from rocke.core.ir import BF16, F16, F32, I32, IRBuilder, PtrType

from ._attention_bwd_test_utils import device_arch
from ._attention_bwd_int_eval import IntEvalBuilder, VirtualTensor

ARCHS = ("gfx942", "gfx950", "gfx1151", "gfx1201")
FLAVORS = ("llvm20", "llvm22", "llvm23")


def _f16_buffer(shape, seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal(shape).astype(np.float16)


# ---------------------------------------------------------------------------
# Sequence decode
# ---------------------------------------------------------------------------


def test_batched_decode_without_seq_never_reads_the_pointer():
    b = IntEvalBuilder()
    null = b.ptr(None)
    for has_len in (0, b.const_i32(0)):
        s = A.decode_batched(
            b, b.const_i32(3), s_max=b.const_i32(77), seq_ptr=null, has_len=has_len
        )
        assert s.length.v == 77 and s.last.v == 76 and s.ws_row0.v == 3 * 77
    assert not [e for e in b.log if e[0] == "load"]


def test_batched_decode_clamps_seq_lengths():
    b = IntEvalBuilder()
    seq = np.array([5, -3, 900, 0, 64, 1], np.int32)
    # strided (B, 1, 1, 1) tensor: one entry every len_stride elements
    stride = 3
    flat = np.full(len(seq) * stride, 12345, np.int32)
    flat[::stride] = seq
    p = b.add_buffer("SEQ", flat)
    for bi, raw in enumerate(seq):
        for has_len in (1, b.const_i32(1)):
            s = A.decode_batched(
                b, b.const_i32(bi), s_max=64, seq_ptr=p, has_len=has_len,
                len_stride=b.const_i32(stride),
            )  # fmt: skip
            want = min(max(int(raw), 0), 64)
            assert s.length.v == want
            assert s.last.v == max(want - 1, 0)
            assert s.ws_row0.v == bi * 64 and s.ws_row0.ty == "i64"


@pytest.mark.parametrize("off64", [0, 1, "runtime0", "runtime1"])
@pytest.mark.parametrize("mult,div", [(1, 1), (1, 384), (2, 4), (3, 6)])
def test_thd_decode_offsets(off64, mult, div):
    b = IntEvalBuilder()
    wide = off64 in (1, "runtime1")
    lens = [7, 0, 100, 1, 33, 0]  # zero-length middle and last segments
    tokens = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)
    base = (1 << 33) if wide else 1000  # int64 tables may exceed 2**31
    # caller offsets are in the units that off * mult / div maps to tokens
    offs = (tokens + base) * div // mult
    assert np.array_equal(offs * mult // div, tokens + base)
    table = offs.astype(np.int64).view(np.int32) if wide else offs.astype(np.int32)
    p = b.add_buffer("OFF", table.copy())
    flag = {0: 0, 1: 1, "runtime0": b.const_i32(0), "runtime1": b.const_i32(1)}[off64]
    for s, n in enumerate(lens):
        info = A.decode_thd(
            b, b.const_i32(s), s_max=b.const_i32(64), off_ptr=p, off64=flag,
            mult=b.const_i32(mult), div=b.const_i32(div),
        )  # fmt: skip
        assert info.tok0.v == int(tokens[s] + base) and info.tok0.ty == "i64"
        assert info.ws_row0 is info.tok0
        assert info.length.v == min(n, 64)
        assert info.last.v == max(min(n, 64) - 1, 0)
    # an int32 table is never read past B + 1 entries
    if not wide:
        idx = {e for kind, name, e, _ in b.log if kind == "load" and name == "OFF"}
        assert max(idx) <= len(lens)


@pytest.mark.parametrize("flag", [1, "runtime1", "runtime0"])
def test_thd_decode_slot_rows(flag):
    """``ws_seg`` nonzero: the workspace rows of sequence ``b`` start at its
    slot ``b * S_max`` (i64), whatever its storage token; 0 keeps ``tok0``."""
    b = IntEvalBuilder()
    offs = np.array([40, 47, 47, 147, 150], np.int32)  # lead 40, empty middle
    p = b.add_buffer("OFF", offs)
    seg = {1: 1, "runtime1": b.const_i32(1), "runtime0": b.const_i32(0)}[flag]
    for s in range(4):
        info = A.decode_thd(
            b, b.const_i32(s), s_max=b.const_i32(100), off_ptr=p, ws_seg=seg
        )
        assert info.tok0.v == int(offs[s])
        want = int(offs[s]) if flag == "runtime0" else s * 100
        assert info.ws_row0.v == want and info.ws_row0.ty == "i64"


def _storage(lens, *, lead, gap, tail):
    """Offsets of ``lens`` sequences in a storage with ``lead`` tokens before
    the first, ``gap`` tokens after every one and ``tail`` after the last."""
    st = [lead]
    for n in lens:
        st.append(st[-1] + n + gap)
    st[-1] += tail
    return np.array(st, np.int32)


def test_slot_rows_bound_every_storage_layout():
    """Rows by sequence slot (``ws_seg = 1``) stay inside ``B * S_max`` rows
    and never collide, for lead tokens, gaps, short SEQ_LEN counts, tails and
    empty (also last) segments; rows by storage token leave ``B * S_max`` as
    soon as the storage is not compact (why the plan uses slots without a
    caller bound)."""
    rng = np.random.default_rng(5)
    escaped = 0
    for trial in range(300):
        n_seq = int(rng.integers(1, 6))
        s_max = int(rng.integers(1, 40))
        lens = [int(rng.integers(0, s_max + 1)) for _ in range(n_seq)]
        if trial % 3 == 0:
            lens[-1] = 0  # empty last segment
        lead, gap, tail = (int(rng.integers(0, 9)) for _ in range(3))
        offs = _storage(lens, lead=lead, gap=gap, tail=tail)
        counts = np.array([max(0, n - int(rng.integers(0, 3))) for n in lens], np.int32)
        use_counts = bool(rng.integers(0, 2))
        b = IntEvalBuilder()
        po = b.add_buffer("OFF", offs.copy())
        ps = b.add_buffer("SEQ", counts.copy())
        rows_total = n_seq * s_max
        taken = np.zeros(rows_total, bool)
        for s in range(n_seq):
            kw = dict(seq_ptr=ps, has_len=b.const_i32(1)) if use_counts else {}
            slot = A.decode_thd(
                b, b.const_i32(s), s_max=s_max, off_ptr=po, ws_seg=1, **kw
            )
            tok = A.decode_thd(b, b.const_i32(s), s_max=s_max, off_ptr=po, **kw)
            n = slot.length.v
            lo, hi = slot.ws_row0.v, slot.ws_row0.v + n
            assert 0 <= lo and hi <= rows_total, (offs, lens, s)
            assert not taken[lo:hi].any(), "two sequences share a workspace row"
            taken[lo:hi] = True
            if n and tok.ws_row0.v + n > rows_total:
                escaped += 1
    assert escaped > 0  # the token rows do leave B * S_max on these storages


def test_thd_decode_with_seq_tensor_takes_the_minimum():
    b = IntEvalBuilder()
    offs = np.array([0, 10, 30, 31], np.int32)
    seq = np.array([4, 50, 0], np.int32)
    po = b.add_buffer("OFF", offs)
    ps = b.add_buffer("SEQ", seq)
    for s in range(3):
        info = A.decode_thd(
            b, b.const_i32(s), s_max=40, off_ptr=po, seq_ptr=ps,
            has_len=b.const_i32(1),
        )  # fmt: skip
        assert info.length.v == min(int(offs[s + 1] - offs[s]), int(seq[s]), 40)


def test_thd_decode_non_monotone_offsets_give_zero_length():
    b = IntEvalBuilder()
    p = b.add_buffer("OFF", np.array([10, 5, 5], np.int32))
    info = A.decode_thd(b, b.const_i32(0), s_max=64, off_ptr=p)
    assert info.length.v == 0 and info.last.v == 0


def test_offset_loader_reads_little_endian_words():
    b = IntEvalBuilder()
    vals = np.array([0, (1 << 40) + 5, -7, (1 << 31) + 3], np.int64)
    p = b.add_buffer("OFF", vals.view(np.int32).copy())
    for i, v in enumerate(vals):
        for flag in (1, b.const_i32(1)):
            assert A.load_offset(b, p, b.const_i32(i), flag).v == int(v)


# ---------------------------------------------------------------------------
# Strided user tensors
# ---------------------------------------------------------------------------


def _layouts(B, H, S, D):
    """(name, flat buffer, element strides (b, h, t), base element offset)."""
    out = []
    bshd = _f16_buffer(B * S * H * D, 1)
    out.append(("bshd", bshd, (S * H * D, D, H * D), 0))
    bhsd = _f16_buffer(B * H * S * D, 2)
    out.append(("bhsd", bhsd, (H * S * D, S * D, D), 0))
    # padded strides: every stride a multiple of 8 but not dense
    sb, sh, st = (S + 3) * H * (D + 16), D + 16, H * (D + 16)
    padded = _f16_buffer(B * sb, 3)
    out.append(("strided_bshd", padded, (sb, sh, st), 0))
    # packed QKV [B, S, 3, H, D]: K view starts at offset H*D
    qkv = _f16_buffer(B * S * 3 * H * D, 4)
    out.append(("packed_qkv_k", qkv, (S * 3 * H * D, D, 3 * H * D), H * D))
    # odd strides (stage_vec = 1 territory)
    sb, sh, st = S * H * (D + 3) + 5, D + 3, H * (D + 3)
    odd = _f16_buffer(B * sb + 8, 5)
    out.append(("odd_strides", odd, (sb, sh, st), 1))
    return out


@pytest.mark.parametrize("stage_vec", [8, 1])
def test_strided_batched_loads_match_numpy(stage_vec):
    B, H, S, D = 3, 2, 21, 32
    lens = [21, 9, 0]
    for name, flat, (sb, sh, st), base in _layouts(B, H, S, D):
        if stage_vec == 8 and name == "odd_strides":
            continue  # the plan never picks stage_vec = 8 for these
        b = IntEvalBuilder()
        p = b.add_buffer("X", flat)
        if base:
            p = b.global_ptr_add(p, b.const_i64(2 * base))
        view = np.lib.stride_tricks.as_strided(
            flat[base:], shape=(B, H, S, D), strides=(2 * sb, 2 * sh, 2 * st, 2)
        )
        seqp = b.add_buffer("SEQ", np.array(lens, np.int32))
        for bi in range(B):
            seq = A.decode_batched(b, b.const_i32(bi), s_max=S, seq_ptr=seqp, has_len=1)
            addr = A.StridedBatchedAddr(
                b, p, dtype=F16, strides=(b.const_i32(sb), sh, b.const_i32(st)),
                seq=seq, stage_vec=stage_vec,
            )  # fmt: skip
            last = max(lens[bi] - 1, 0)
            for h, row0 in itertools.product(range(H), (0, 8, 16)):
                slab = addr.slab(b.const_i32(h), b.const_i32(row0))
                for r, c0 in itertools.product(range(row0, row0 + 8), (0, 8, 24)):
                    got = slab.load_vec(b.const_i32(r), b.const_i32(c0))
                    want = view[bi, h, min(r, last), c0 : c0 + 8]
                    assert np.array_equal(np.array(got.v, np.float16), want), name
        vloads = [e for e in b.log if e[0] == "vload"]
        if stage_vec == 8:
            assert vloads and all(e[2] == 8 and e[3] == 16 for e in vloads)
        else:
            assert not vloads


def test_unclamped_slab_addresses_padding_rows_for_stores():
    B, H, S, D = 2, 3, 40, 16
    out = np.zeros(B * S * H * D, np.float16)
    b = IntEvalBuilder()
    p = b.add_buffer("DK", out)
    seqp = b.add_buffer("SEQ", np.array([13, 40], np.int32))
    view = out.reshape(B, S, H, D)
    for bi in range(B):
        seq = A.decode_batched(b, b.const_i32(bi), s_max=S, seq_ptr=seqp, has_len=1)
        addr = A.StridedBatchedAddr(
            b, p, dtype=F16, strides=(S * H * D, D, H * D), seq=seq, stage_vec=8
        )
        for h in range(H):
            slab = addr.slab(b.const_i32(h), b.const_i32(32), clamp=False)
            for r in range(32, 40):
                vec = b.vec_pack(
                    [b.cast_f32_to(b.const_f32(bi * 1000 + h * 100 + r), F16)] * 8,
                    F16,
                )
                slab.store_vec(b.const_i32(r), 8, vec)
    for bi, h, r in itertools.product(range(B), range(H), range(32, 40)):
        assert np.all(view[bi, r, h, 8:16] == np.float16(bi * 1000 + h * 100 + r))
    assert np.count_nonzero(view) == B * H * 8 * 8


@pytest.mark.parametrize("stage_vec", [8, 1])
@pytest.mark.parametrize("n", [8, 4, 2, 1])
def test_row_store_widths(n, stage_vec):
    """``store_vec`` of ``n`` elements (the dK / dV epilogue's row width):
    one aligned ``2 n``-byte vector store with ``stage_vec = 8``, element
    stores with ``stage_vec = 1``; the same elements land either way."""
    B, H, S, D = 1, 2, 8, 32
    out = np.zeros(B * S * H * D, np.float16)
    b = IntEvalBuilder()
    p = b.add_buffer("DK", out)
    seq = A.decode_batched(b, b.const_i32(0), s_max=S, seq_ptr=b.ptr(None), has_len=0)
    addr = A.StridedBatchedAddr(
        b, p, dtype=F16, strides=(S * H * D, D, H * D), seq=seq, stage_vec=stage_vec
    )
    view = out.reshape(S, H, D)
    slab = addr.slab(b.const_i32(1), b.const_i32(0), clamp=False)
    for r, c in itertools.product(range(S), range(0, D, n)):
        vals = [b.cast_f32_to(b.const_f32(r * 100 + c + i), F16) for i in range(n)]
        slab.store_vec(b.const_i32(r), b.const_i32(c), b.vec_pack(vals, F16))
    want = np.arange(D, dtype=np.float32)[None, :] + 100 * np.arange(S)[:, None]
    assert np.array_equal(view[:, 1, :], want.astype(np.float16))
    assert not view[:, 0, :].any()
    vstores = [e for e in b.log if e[0] == "vstore"]
    if stage_vec == 8 and n > 1:
        assert len(vstores) == S * D // n
        assert all(e[2] == n and e[3] == 2 * n for e in vstores)
    else:
        assert not vstores


def test_thd_tensor_rebases_by_token():
    H, D = 4, 64
    lens = [5, 0, 17, 3]
    T = sum(lens)
    x = _f16_buffer(T * H * D + 64, 7)
    b = IntEvalBuilder()
    p = b.add_buffer("Q", x)
    offs = np.concatenate([[0], np.cumsum(lens)]).astype(np.int32)
    po = b.add_buffer("OFF", offs)
    view = x[: T * H * D].reshape(T, H, D)
    for s in range(len(lens)):
        seq = A.decode_thd(b, b.const_i32(s), s_max=32, off_ptr=po)
        addr = A.ThdOffsetAddr(
            b, p, dtype=F16, strides=(0, D, H * D), seq=seq, stage_vec=8
        )
        last = max(lens[s] - 1, 0)
        for h in range(H):
            slab = addr.slab(b.const_i32(h), b.const_i32(0))
            for r in range(20):
                got = slab.load_vec(b.const_i32(r), b.const_i32(16))
                if lens[s] == 0:
                    # an empty last-but-one segment still reads a mapped row
                    continue
                want = view[offs[s] + min(r, last), h, 16:24]
                assert np.array_equal(np.array(got.v, np.float16), want)
    with pytest.raises(ValueError):
        A.StridedBatchedAddr(
            b, p, dtype=F16, strides=(0, D, H * D), seq=seq, stage_vec=8
        )


def test_rebase_past_2_pow_31_elements_uses_i64_only():
    """A batch / head base beyond 2**31 elements: no i32 overflow, exact element."""
    B, H, S, D = 64, 16, 1 << 16, 128  # 2**33 elements
    total = B * H * S * D
    b = IntEvalBuilder()
    p = b.add_buffer("Q", VirtualTensor(total, lambda e: float(e % 2039)), "f16")
    seq = A.decode_batched(b, b.const_i32(B - 1), s_max=b.const_i32(S))
    addr = A.StridedBatchedAddr(
        b, p, dtype=F16, strides=(b.const_i32(H * S * D), b.const_i32(S * D), D),
        seq=seq, stage_vec=8,
    )  # fmt: skip
    slab = addr.slab(b.const_i32(H - 1), b.const_i32(S - 16))
    got = slab.load_vec(b.const_i32(S - 1), b.const_i32(120))
    e0 = (((B - 1) * H + (H - 1)) * S + (S - 1)) * D + 120
    assert e0 >= 1 << 32
    assert got.v == tuple(float(e % 2039) for e in range(e0, e0 + 8))
    # negative control: the same element index formed in i32 overflows
    with pytest.raises(OverflowError):
        b.mul(b.const_i32(B - 1), b.const_i32(H * S * D))


def test_bf16_and_f32_scalar_access():
    b = IntEvalBuilder()
    lse = np.arange(2 * 3 * 10, dtype=np.float32)
    p = b.add_buffer("LSE", lse)
    seq = A.decode_batched(b, b.const_i32(1), s_max=10)
    addr = A.StridedBatchedAddr(b, p, dtype=F32, strides=(30, 10, 1), seq=seq)
    slab = addr.slab(b.const_i32(2), 4)
    assert slab.load(b.const_i32(7)).v == lse.reshape(2, 3, 10)[1, 2, 7]
    with pytest.raises(ValueError):
        A.StridedBatchedAddr(b, p, dtype=I32, strides=(1, 1, 1), seq=seq)
    with pytest.raises(ValueError):
        A.StridedBatchedAddr(b, p, dtype=BF16, strides=(1, 1, 1), seq=seq, stage_vec=4)


# ---------------------------------------------------------------------------
# Workspace and paged caches
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("layout", ["head_major", "token_major"])
def test_workspace_indexing_and_row_guards(layout):
    H, D = 3, 16
    ws_rows = 50
    ws = np.arange(H * ws_rows * D, dtype=np.float32)
    b = IntEvalBuilder()
    p = b.add_buffer("WS", ws)
    if layout == "head_major":
        view = ws.reshape(H, ws_rows, D)
    else:
        view = ws.reshape(ws_rows, H, D).transpose(1, 0, 2)
    for row0, tile0 in ((0, 0), (20, 8), (45, 0), (48, 4), (60, 0)):
        wa = A.WorkspaceAddr(
            b, p, ws_rows=b.const_i32(ws_rows), width=D, row0=b.const_i64(row0),
            layout=layout, n_heads=b.const_i32(H) if layout == "token_major" else None,
        )  # fmt: skip
        for h in range(H):
            slab = wa.slab(b.const_i32(h), b.const_i32(tile0))
            for r, x in itertools.product(range(8), (0, 5, 15)):
                row = row0 + tile0 + r
                got = slab.load(b.const_i32(r), b.const_i32(x), float("inf")).v
                want = view[h, row, x] if row < ws_rows else float("inf")
                assert got == want, (row0, tile0, h, r, x)


def test_workspace_base_past_a_single_row_workspace_stays_in_bounds():
    """ws_rows = 1 and a base past it: every access is clamped to row 0."""
    D = 4
    ws = np.arange(D, dtype=np.float32)
    b = IntEvalBuilder()
    p = b.add_buffer("WS", ws)
    wa = A.WorkspaceAddr(b, p, ws_rows=1, width=D, row0=b.const_i64(5))
    slab = wa.slab(b.const_i32(0))
    for r in range(4):
        assert slab.load(b.const_i32(r), b.const_i32(1), -1.0).v == -1.0
        slab.atomic_add(b.const_i32(r), b.const_i32(2), b.const_f32(3.0))
    assert np.array_equal(ws, np.arange(D, dtype=np.float32))


def test_workspace_atomics_mask_rows_past_ws_rows():
    H, D, ws_rows = 2, 8, 20
    ws = np.zeros(H * ws_rows * D, np.float32)
    b = IntEvalBuilder()
    p = b.add_buffer("WS", ws)
    wa = A.WorkspaceAddr(
        b, p, ws_rows=ws_rows, width=D, row0=b.const_i64(15), layout="head_major"
    )
    slab = wa.slab(b.const_i32(1))
    for r in range(10):
        for x in range(D):
            slab.atomic_add(b.const_i32(r), b.const_i32(x), b.const_f32(1.0))
    v = ws.reshape(H, ws_rows, D)
    assert np.all(v[1, 15:20] == 1.0)
    assert v.sum() == 5 * D  # rows 20.. added 0 at a clamped address
    atomics = [e for e in b.log if e[0] == "atomic"]
    assert len(atomics) == 10 * D


@pytest.mark.parametrize("layout", ["head_major", "token_major"])
def test_workspace_row_atomics_skip_rows_outside(layout):
    """``atomic_add_row``: rows inside the workspace (and ``ok``) get every
    value at ``(row, x)``; other rows issue no atomic at all."""
    H, D, ws_rows = 3, 8, 20
    ws = np.zeros(H * ws_rows * D, np.float32)
    b = IntEvalBuilder()
    p = b.add_buffer("WS", ws)
    wa = A.WorkspaceAddr(
        b, p, ws_rows=b.const_i32(ws_rows), width=D, row0=b.const_i64(12),
        layout=layout, n_heads=b.const_i32(H) if layout == "token_major" else None,
    )  # fmt: skip
    slab = wa.slab(b.const_i32(2), b.const_i32(3))  # rows 15 .. 19 inside
    for r in range(8):
        items = [(b.const_i32(x), b.const_f32(10 * r + x)) for x in (0, 3, 7)]
        ok = b.cmp_lt(b.const_i32(r), b.const_i32(4))  # e.g. k < len_kv
        slab.atomic_add_row(b.const_i32(r), items, ok)
    if layout == "head_major":
        v = ws.reshape(H, ws_rows, D)
    else:
        v = ws.reshape(ws_rows, H, D).transpose(1, 0, 2)
    want = np.zeros((H, ws_rows, D), np.float32)
    for r in range(4):
        for x in (0, 3, 7):
            want[2, 15 + r, x] = 10 * r + x
    assert np.array_equal(v, want)
    assert len([e for e in b.log if e[0] == "atomic"]) == 4 * 3
    # without ok, the workspace guard alone decides (rows 15 .. 19)
    b.log.clear()
    for r in range(8):
        slab.atomic_add_row(b.const_i32(r), [(0, b.const_f32(1.0))])
    assert len([e for e in b.log if e[0] == "atomic"]) == 5


@pytest.mark.parametrize("layout", ["head_major", "token_major"])
def test_workspace_row_atomics_select_guard_matches_the_branch(layout):
    """``atomic_add_row_select``: every pair is added, rows outside the
    workspace (or not ``ok``) add ``+0.0`` at a clamped row inside the
    head's slab, so the workspace is bit for bit the branch guard's (also
    with negative values and with -0.0 / +0.0 inputs)."""
    H, D, ws_rows = 3, 8, 20

    def run(method):
        ws = np.zeros(H * ws_rows * D, np.float32)
        ws[1::7] = -3.5  # untouched data elsewhere
        b = IntEvalBuilder()
        p = b.add_buffer("WS", ws)
        wa = A.WorkspaceAddr(
            b, p, ws_rows=b.const_i32(ws_rows), width=D, row0=b.const_i64(12),
            layout=layout, n_heads=b.const_i32(H) if layout == "token_major" else None,
        )  # fmt: skip
        slab = wa.slab(b.const_i32(2), b.const_i32(3))  # rows 15 .. 19 inside
        vals = (-0.0, 0.0, -1.25, 7.0)
        for r in range(8):
            items = [(x, b.const_f32(vals[(r + x) % 4] * (r + 1))) for x in (0, 3, 7)]
            ok = b.cmp_lt(b.const_i32(r), b.const_i32(4))
            getattr(slab, method)(b.const_i32(r), items, ok)
        atomics = [e for e in b.log if e[0] == "atomic"]
        return ws, atomics

    ws_branch, at_branch = run("atomic_add_row")
    ws_select, at_select = run("atomic_add_row_select")
    assert ws_branch.tobytes() == ws_select.tobytes()
    assert len(at_branch) == 4 * 3 and len(at_select) == 8 * 3
    # the extra atomics add +0.0 (never -0.0) inside the workspace
    extra = [e for e in at_select if e not in at_branch]
    assert extra and all(
        np.float32(v).tobytes() == np.float32(0.0).tobytes() for *_e, v in extra
    )


def test_workspace_head_rebase_past_2_pow_31_bytes():
    H, rows, D = 64, 1 << 20, 128  # 2**33 floats
    b = IntEvalBuilder()
    p = b.add_buffer("WS", VirtualTensor(H * rows * D, lambda e: float(e % 997)), "f32")
    wa = A.WorkspaceAddr(
        b, p, ws_rows=b.const_i32(rows), width=D, row0=b.const_i64(rows - 64)
    )
    slab = wa.slab(b.const_i32(H - 1))
    got = slab.load(b.const_i32(63), b.const_i32(127), 0.0).v
    e = ((H - 1) * rows + rows - 1) * D + 127
    assert got == float(e % 997)


@pytest.mark.parametrize(
    "H,rows,tile,width",
    [
        (32, 1 << 20, (1 << 20) - 16, 128),  # h * ws_rows * D past 2**31
        (40, (1 << 26) - 1, 1 << 25, 32),  # h * ws_rows itself past 2**31
        (40, (1 << 26) - 1, 1 << 25, 1),  # stats rows (X = 1)
    ],
)
def test_workspace_head_times_rows_past_2_pow_31_stays_i64(H, rows, tile, width):
    # ws_rows * X stays below 2**31 (the index-range contract); the typed
    # evaluator raises on any i32 form of the head rebase.
    b = IntEvalBuilder()
    vt = VirtualTensor(H * rows * width, lambda e: 0.0)
    p = b.add_buffer("WS", vt, "f32")
    wa = A.WorkspaceAddr(
        b, p, ws_rows=b.const_i32(rows), width=width, row0=b.const_i64(0)
    )
    h = H - 1
    slab = wa.slab(b.const_i32(h), b.const_i32(tile))  # per-CTA (prep / convert)
    slab.atomic_add(b.const_i32(5), b.const_i32(width - 1), b.const_f32(2.0))
    e = (h * rows + tile + 5) * width + width - 1
    assert e >= 1 << 31
    assert vt.writes == {e: 2.0}
    head = wa.slab(b.const_i32(h))  # per-head (dQ sink)
    assert head.load(b.const_i32(tile + 5), b.const_i32(width - 1), 0.0).v == 2.0


def test_workspace_rejects_bad_layouts():
    b = IntEvalBuilder()
    p = b.add_buffer("WS", np.zeros(4, np.float32))
    with pytest.raises(ValueError):
        A.WorkspaceAddr(b, p, ws_rows=1, width=1, row0=b.const_i64(0), layout="x")
    with pytest.raises(ValueError):
        A.WorkspaceAddr(
            b, p, ws_rows=1, width=1, row0=b.const_i64(0), layout="token_major"
        )


def test_paged_cache_and_block_table():
    nb, page, H, D = 9, 16, 2, 32
    cache = _f16_buffer(nb * page * H * D, 9)
    view = cache.reshape(nb, page, H, D)
    bt = np.array([[3, 7, 1, 0], [8, 2, 0, 0]], np.int32)
    b = IntEvalBuilder()
    pc = b.add_buffer("KC", cache)
    pb = b.add_buffer("BT", bt.reshape(-1).copy())
    addr = A.PagedCacheAddr(
        b, pc, dtype=F16, strides=(page * H * D, H * D, D), stage_vec=8
    )
    for s, j in itertools.product(range(2), range(3)):
        pid = A.block_table_entry(
            b, pb, b.const_i32(s), b.const_i32(j), bt_stride=b.const_i32(4)
        )
        assert pid.v == bt[s, j]
        for h, tok0 in itertools.product(range(H), (0, 8)):
            slab = addr.slab(pid, b.const_i32(h), b.const_i32(tok0))
            for r in range(tok0, tok0 + 8):
                got = slab.load_vec(b.const_i32(r), 8)
                assert np.array_equal(
                    np.array(got.v, np.float16), view[bt[s, j], r, h, 8:16]
                )


def test_no_exported_helper_looks_like_a_builder():
    assert not [n for n in A.__all__ if n.startswith("build_")]


# ---------------------------------------------------------------------------
# IR probe: copies rows of a strided / THD source into a dense output
# ---------------------------------------------------------------------------


def _probe_kernel(name, *, mode, stage_vec, dtype=F16, D=64):
    """One CTA per (batch, head); lane l copies row l (clamped) of the tile."""
    b = IRBuilder(name)
    src = b.param("SRC", PtrType(dtype, "global"), readonly=True)
    dst = b.param("DST", PtrType(dtype, "global"))
    seqp = b.param("SEQ", PtrType(I32, "global"), readonly=True)
    offp = b.param("OFF", PtrType(I32, "global"), readonly=True)
    names = ("s_b", "s_h", "s_t", "S_max", "has_len", "off64", "mult", "div", "row0")
    a = {n: b.param(n, I32) for n in names}
    bi = b.block_id_y()
    h = b.block_id_x()
    if mode == "thd":
        seq = A.decode_thd(
            b, bi, s_max=a["S_max"], off_ptr=offp, off64=a["off64"],
            mult=a["mult"], div=a["div"], seq_ptr=seqp, has_len=a["has_len"],
        )  # fmt: skip
        cls = A.ThdOffsetAddr
    else:
        seq = A.decode_batched(
            b, bi, s_max=a["S_max"], seq_ptr=seqp, has_len=a["has_len"]
        )
        cls = A.StridedBatchedAddr
    addr = cls(
        b, src, dtype=dtype, strides=(a["s_b"], a["s_h"], a["s_t"]), seq=seq,
        stage_vec=stage_vec,
    )  # fmt: skip
    slab = addr.slab(h, a["row0"])
    lane = b.thread_id_x()
    row = b.add(a["row0"], lane)
    # dense output [B, H, 64 rows, D]: row r of the tile, value of the clamped row
    gh = b.param("H", I32)
    out_row = b.add(b.mul(b.add(b.mul(bi, gh), h), b.const_i32(64)), lane)
    for c in range(0, D, 8):
        v = slab.load_vec(row, c)
        out_idx = b.add(b.mul(out_row, b.const_i32(D)), b.const_i32(c))
        b.global_store_vN(dst, out_idx, v, 8, align=2)
    b.ret()
    b.kernel.attrs["max_workgroup_size"] = 64
    return b.kernel


def _lower(kernel, arch, flavor):
    from rocke.core.lower_llvm import _lower_kernel_to_llvm_python as lower

    return lower(kernel, arch=arch, llvm_flavor=flavor)


@pytest.mark.parametrize("flavor", FLAVORS)
@pytest.mark.parametrize("arch", ARCHS)
@pytest.mark.parametrize("mode", ["batched", "thd"])
def test_probe_lowers_with_i64_rebase(arch, flavor, mode):
    for stage_vec in (8, 1):
        ir = _lower(
            _probe_kernel(
                f"bwd_addr_{mode}_{stage_vec}", mode=mode, stage_vec=stage_vec
            ),
            arch,
            flavor,
        )
        assert "alloca" not in ir
        # the tensor base is a byte GEP with an i64 offset
        assert re.search(r"getelementptr inbounds i8, ptr addrspace\(1\) %SRC, i64", ir)
        user_loads = re.findall(
            r"load (<\d+ x half>|half), ptr addrspace\(1\) [^,]+, align (\d+)", ir
        )
        assert user_loads
        if stage_vec == 8:
            assert all(t == "<8 x half>" and al == "16" for t, al in user_loads)
        else:
            assert all(t == "half" and al == "2" for t, al in user_loads)
        if mode == "thd":
            assert "sdiv i64" in ir


@pytest.mark.gpu
@pytest.mark.parametrize("mode", ["batched", "thd"])
@pytest.mark.parametrize("stage_vec", [8, 1])
def test_probe_on_device_matches_numpy(mode, stage_vec):
    arch = device_arch()
    if arch not in ("gfx942", "gfx950", "gfx1151", "gfx1201"):
        pytest.skip(f"needs gfx942/gfx950/gfx1151/gfx1201; device is {arch}")
    from rocke.helpers.compile import compile_kernel
    from rocke.runtime.hip_module import Runtime

    D, H, S = 64, 3, 50
    kern = _probe_kernel(
        f"bwd_addr_dev_{mode}_{stage_vec}", mode=mode, stage_vec=stage_vec
    )
    art = compile_kernel(kern, arch=arch, capture_ir_text=False, backend="python")
    rng = np.random.default_rng(3)
    if mode == "batched":
        B = 3
        lens = np.array([50, 17, 0], np.int32)
        pad_t = H * (D + 8) if stage_vec == 8 else H * (D + 3)
        sb, sh, st = S * pad_t + 8, (D + 8) if stage_vec == 8 else (D + 3), pad_t
        src = rng.standard_normal(B * sb + 64).astype(np.float16)
        offs = np.zeros(4, np.int32)
        off64, mult, div = 0, 1, 1
        has_len = 1
    else:
        B = 4
        lens = np.array([30, 0, 50, 0], np.int32)  # zero-length last segment
        sb, sh, st = 0, D, H * D
        T = int(lens.sum())
        # pad one token: a zero-length last segment clamps to row tok0 = T
        src = rng.standard_normal((T + 1) * H * D + 64).astype(np.float16)
        toks = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)
        offs = (toks * st).view(np.int32).copy()  # int64 element offsets
        off64, mult, div = 1, 1, st
        has_len = 0
    row0 = 16
    out = np.zeros(B * H * 64 * D, np.float16)
    rt = Runtime()
    mod = rt.load_module(art.hsaco)
    fn = mod.get_function(art.kernel_name)

    def up(a):
        a = np.ascontiguousarray(a)
        d = rt.alloc(max(a.nbytes, 16))
        rt.memcpy_h2d(d, (ctypes.c_uint8 * a.nbytes).from_buffer(a), a.nbytes)
        return d

    d_src, d_dst, d_seq, d_off = up(src), up(out), up(lens), up(offs)
    vals = [sb, sh, st, S, has_len, off64, mult, div, row0, H]
    args = struct.pack("<QQQQ" + "i" * len(vals), d_src, d_dst, d_seq, d_off, *vals)
    rt.launch(fn, (H, B, 1), (64, 1, 1), args)
    rt.sync()
    rt.memcpy_d2h((ctypes.c_uint8 * out.nbytes).from_buffer(out), d_dst, out.nbytes)
    for d in (d_src, d_dst, d_seq, d_off):
        rt.free(d)
    mod.unload()
    got = out.reshape(B, H, 64, D)
    tok = np.concatenate([[0], np.cumsum(lens)])
    for bi, h in itertools.product(range(B), range(H)):
        n = min(int(lens[bi]), S)
        if n == 0 and mode == "thd":
            continue  # a zero-length segment reads its neighbour's (mapped) row
        last = max(n - 1, 0)
        for lane in range(64):
            r = min(row0 + lane, last)
            if mode == "batched":
                e = bi * sb + h * sh + r * st
            else:
                e = (tok[bi] + r) * st + h * sh
            assert np.array_equal(got[bi, h, lane], src[e : e + D]), (bi, h, lane)


# ---------------------------------------------------------------------------
# packed rows
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("k_m0", [16, 32, 64])
def test_packed_rows_map_heads_and_rows(k_m0):
    b = IntEvalBuilder()
    for per, n_heads, len_q, head0 in itertools.product(
        (1, 2, 3, 4, 8), (1, 2, 3, 4, 8), (1, 2, 3, 5), (0, 7)
    ):
        if n_heads > per or per * len_q > k_m0:
            continue
        rows = A.PackedRows(
            head0=b.const_i32(head0),
            n_heads=b.const_i32(n_heads),
            rows_per_head=b.const_i32(len_q),
            row0=b.const_i32(0),
        )
        for r in range(k_m0):
            h, q, ok = rows.locate(b, b.const_i32(r))
            hh = r // len_q
            assert ok.v == (hh < n_heads)
            assert h.v == head0 + min(hh, n_heads - 1)
            assert q.v == r % len_q and 0 <= q.v < len_q


def test_packed_rows_identity_when_unpacked():
    # rows_per_head = k_m0 and one head: row = row0 + r on the same head
    b = IntEvalBuilder()
    rows = A.PackedRows(
        head0=b.const_i32(5),
        n_heads=b.const_i32(1),
        rows_per_head=b.const_i32(32),
        row0=b.const_i32(64),
    )
    for r in range(32):
        h, q, ok = rows.locate(b, r)
        assert (h.v, q.v, ok.v) == (5, 64 + r, True)
