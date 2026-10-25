# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Device numerics of the attention backward prep / convert / page roles.

Compared with numpy on poisoned buffers (NaN workspace, sentinel outputs,
canary tails), so a missing initialisation, a write outside its rows or an
out-of-range access shows up.  Skipped unless the visible device is gfx942,
gfx950, gfx1151 or gfx1201.
"""

from __future__ import annotations

import ctypes
import itertools
import struct

import numpy as np
import pytest
from kernels.common import attention_bwd_aux as X

_ARCH = None
try:
    from rocke.runtime.hip_module import get_device_arch

    _ARCH = get_device_arch(0)
except Exception:  # noqa: BLE001 - no runtime / no device
    _ARCH = None

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        _ARCH not in ("gfx942", "gfx950", "gfx1151", "gfx1201"),
        reason=f"needs gfx942/gfx950/gfx1151/gfx1201; device is {_ARCH}",
    ),
]

LOG2E = np.float32(1.4426950408889634)
CANARY = 256

_BUILD = {
    "prep": X.build_attn_bwd_prep,
    "convert": X.build_attn_bwd_convert,
    "page_zero": X.build_attn_bwd_page_zero,
    "page_convert": X.build_attn_bwd_page_convert,
}
_CACHE: dict = {}


def _bf16_bits(x):
    u = np.ascontiguousarray(x, np.float32).view(np.uint32).astype(np.uint64)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)


def _bf16_val(bits):
    return (bits.astype(np.uint32) << 16).view(np.float32)


def _enc(x, dtype):
    return x.astype(np.float16) if dtype == "fp16" else _bf16_bits(x)


def _dec(u, dtype):
    return u.astype(np.float32) if dtype == "fp16" else _bf16_val(u)


def _poison(dtype):
    return np.float16(np.nan) if dtype == "fp16" else np.uint16(0x7FC0)


def _run(stage, spec, grid, bufs, scalars):
    from rocke.helpers.compile import compile_kernel
    from rocke.runtime.hip_module import Runtime

    key = (stage, spec)
    if key not in _CACHE:
        kern = _BUILD[stage](spec, arch=_ARCH)
        _CACHE[key] = compile_kernel(
            kern, arch=_ARCH, capture_ir_text=False, backend="python"
        )
    art = _CACHE[key]
    rt = Runtime()
    mod = rt.load_module(art.hsaco)
    fn = mod.get_function(art.kernel_name)
    dev = {}
    for k, a in bufs.items():
        if a is None:
            dev[k] = 0
            continue
        dev[k] = rt.alloc(max(a.nbytes, 16))
        rt.memcpy_h2d(dev[k], (ctypes.c_uint8 * a.nbytes).from_buffer(a), a.nbytes)
    packed = b""
    for name, fmt in X.attn_bwd_aux_params(stage, spec):
        if fmt == "Q":
            packed += struct.pack("<Q", dev.get(name, 0))
        else:
            packed += struct.pack("<" + fmt, scalars.get(name, 0))
    rt.launch(fn, grid, (X.attn_bwd_aux_block_threads(spec, arch=_ARCH), 1, 1), packed)
    rt.sync()
    for k, a in bufs.items():
        if a is not None:
            rt.memcpy_d2h((ctypes.c_uint8 * a.nbytes).from_buffer(a), dev[k], a.nbytes)
            rt.free(dev[k])
    mod.unload()


def _nan_ws(n):
    return np.full(n + CANARY, np.nan, np.float32)


# ---------------------------------------------------------------------------
# prep
# ---------------------------------------------------------------------------


def _prep_inputs(rng, dtype, B, H, S, D, layout, dead_rows):
    """O, dO, LSE as flat buffers plus (b, h, t) strides; NaN O on dead rows."""
    if layout == "bshd":
        st = (S * H * D, D, H * D)
        n = B * S * H * D
    elif layout == "odd":
        st = (S * H * (D + 3) + 5, D + 3, H * (D + 3))
        n = B * st[0]
    else:
        st = (H * S * D, S * D, D)
        n = B * H * S * D
    o = np.full(n + 64, _poison(dtype))
    do = np.full(n + 64, _poison(dtype))
    ov = rng.standard_normal((B, H, S, D)).astype(np.float32)
    dov = rng.standard_normal((B, H, S, D)).astype(np.float32)
    for bi, h, s in itertools.product(range(B), range(H), range(S)):
        e = bi * st[0] + h * st[1] + s * st[2]
        o[e : e + D] = _enc(ov[bi, h, s], dtype)
        do[e : e + D] = _enc(dov[bi, h, s], dtype)
    lse = (rng.standard_normal((B, H, S)) * 3).astype(np.float32)
    for bi, h, s, kind in dead_rows:
        lse[bi, h, s] = {"ninf": -np.inf, "nan": np.nan, "pinf": np.inf}[kind]
        e = bi * st[0] + h * st[1] + s * st[2]
        o[e : e + D] = _poison(dtype)  # a dead row's O may be anything
    ovq = _dec(np.array([_enc(ov, dtype)])[0], dtype)
    dovq = _dec(np.array([_enc(dov, dtype)])[0], dtype)
    return o, do, lse, st, ovq, dovq


def _check_stats(lse2, dsum, h, row, lse_v, live, ov, dov):
    if live:
        assert lse2 == np.float32(lse_v) * LOG2E
        want = float(np.dot(ov.astype(np.float64), dov.astype(np.float64)))
        assert abs(float(dsum) - want) <= 1e-5 * max(1.0, abs(want)) + 1e-4
    else:
        assert np.isposinf(lse2) and dsum == 0.0


@pytest.mark.parametrize(
    "D,dtype,layout,stage_vec,ws_layout,rows_x",
    [
        (64, "fp16", "bshd", 8, "head_major", 1),
        (128, "bf16", "bhsd", 8, "head_major", 1),
        (32, "fp16", "odd", 1, "token_major", 1),
        (128, "bf16", "odd", 1, "head_major", 1),
        # rows_per_cta tuning values (2x / 4x the default rows per CTA)
        (64, "fp16", "bshd", 8, "head_major", 2),
        (128, "bf16", "odd", 1, "token_major", 4),
        (32, "fp16", "bhsd", 8, "token_major", 4),
    ],
)
def test_prep_batched(D, dtype, layout, stage_vec, ws_layout, rows_x):
    rng = np.random.default_rng(D)
    B, H_q, H_k, H_v = 3, 4, 2, 1
    S_q, S_kv = 37, 50
    lens_q = np.array([37, 13, 0], np.int32)
    lens_kv = np.array([50, 0, 7], np.int32)
    dead = [(0, 1, 5, "ninf"), (0, 2, 6, "nan"), (1, 0, 3, "pinf"), (0, 3, 36, "nan")]
    o, do, lse, st, ov, dov = _prep_inputs(rng, dtype, B, H_q, S_q, D, layout, dead)
    rows_q, rows_kv = B * S_q, B * S_kv
    ws = {
        "WS_LSE2": _nan_ws(H_q * rows_q),
        "WS_DSUM": _nan_ws(H_q * rows_q),
        "WS_DQ": _nan_ws(H_q * rows_q * D),
        "WS_DK": _nan_ws(H_k * rows_kv * D),
        "WS_DV": _nan_ws(H_v * rows_kv * D),
    }
    spec = X.AttnBwdAuxSpec(
        head_size=D, dtype=dtype, stage_vec=stage_vec, ws_layout=ws_layout
    )
    if rows_x != 1:
        rows = X.attn_bwd_aux_rows_per_cta(spec, arch=_ARCH) * rows_x
        spec = X.AttnBwdAuxSpec(
            head_size=D, dtype=dtype, stage_vec=stage_vec, ws_layout=ws_layout,
            rows_per_cta=rows,
        )  # fmt: skip
    bufs = dict(O=o, dO=do, LSE=lse.reshape(-1).copy(), SEQ_Q=lens_q, SEQ_KV=lens_kv,
                OFF_Q=None, OFF_KV=None, WORKLIST=None, **ws)  # fmt: skip
    sc = {
        "o_b": st[0], "o_h": st[1], "o_t": st[2], "do_b": st[0], "do_h": st[1], "do_t": st[2],
        "l_b": H_q * S_q, "l_h": S_q, "l_t": 1, "h_q": H_q, "h_k": H_k, "h_v": H_v,
        "S_q_max": S_q, "S_kv_max": S_kv, "has_len": 1, "len_stride": 1, "off64": 0,
        "q_mult": 1, "q_div": 1, "kv_mult": 1, "kv_div": 1, "ws_rows_q": rows_q,
        "ws_rows_kv": rows_kv, "zero_kv": 1, "zero_dq": 1,
    }  # fmt: skip
    grid = X.attn_bwd_prep_grid(
        spec, arch=_ARCH, s_q_max=S_q, s_kv_max=S_kv, h_q=H_q, h_k=H_k, h_v=H_v,
        batch=B, zero_kv=True,
    )  # fmt: skip
    _run("prep", spec, grid, bufs, sc)

    def ws_view(a, H, rows, X_):
        body = a[: H * rows * X_]
        if ws_layout == "head_major":
            return body.reshape(H, rows, X_)
        return body.reshape(rows, H, X_).transpose(1, 0, 2)

    lse2 = ws_view(ws["WS_LSE2"], H_q, rows_q, 1)[..., 0]
    dsum = ws_view(ws["WS_DSUM"], H_q, rows_q, 1)[..., 0]
    deadset = {(bi, h, s) for bi, h, s, _ in dead}
    for bi, h, q in itertools.product(range(B), range(H_q), range(S_q)):
        live = q < lens_q[bi] and (bi, h, q) not in deadset
        row = bi * S_q + q
        _check_stats(lse2[h, row], dsum[h, row], h, row, lse[bi, h, q], live,
                     ov[bi, h, q], dov[bi, h, q])  # fmt: skip
    assert np.all(ws_view(ws["WS_DQ"], H_q, rows_q, D) == 0.0)
    assert np.all(ws_view(ws["WS_DK"], H_k, rows_kv, D) == 0.0)
    assert np.all(ws_view(ws["WS_DV"], H_v, rows_kv, D) == 0.0)
    for a in ws.values():
        assert np.all(np.isnan(a[-CANARY:]))


@pytest.mark.parametrize("off64,mult", [(0, 1), (1, 2)])
def test_prep_thd_and_zero_dq_flag(off64, mult):
    rng = np.random.default_rng(7)
    D, H_q, H_k, H_v = 64, 2, 2, 2
    lens_q = [9, 0, 30, 1]
    lens_kv = [12, 5, 0, 40]
    Tq, Tkv = sum(lens_q), sum(lens_kv)
    S_max = 40
    o, do, lse, _st, ov, dov = _prep_inputs(
        rng, "fp16", 1, H_q, Tq, D, "bshd", [(0, 1, 9 + 4, "nan")]
    )
    # THD [T, H, D]: token stride H*D; offsets in elements, divided back by div
    t_stride = H_q * D
    div = t_stride
    tq = np.concatenate([[0], np.cumsum(lens_q)]).astype(np.int64)
    tk = np.concatenate([[0], np.cumsum(lens_kv)]).astype(np.int64)
    eq = tq * div // mult
    ek = tk * div // mult

    def table(e):
        return e.astype(np.int64).view(np.int32).copy() if off64 else e.astype(np.int32)

    rows_q, rows_kv = Tq, Tkv
    ws = {
        "WS_LSE2": _nan_ws(H_q * rows_q),
        "WS_DSUM": _nan_ws(H_q * rows_q),
        "WS_DQ": _nan_ws(H_q * rows_q * D),
        "WS_DK": _nan_ws(H_k * rows_kv * D),
        "WS_DV": _nan_ws(H_v * rows_kv * D),
    }
    spec = X.AttnBwdAuxSpec(head_size=D, seq_mode="thd")
    lse_thd = lse[0].transpose(1, 0).copy()  # [T, H]
    bufs = dict(O=o, dO=do, LSE=lse_thd.reshape(-1).copy(), SEQ_Q=None, SEQ_KV=None,
                OFF_Q=table(eq), OFF_KV=table(ek), WORKLIST=None, **ws)  # fmt: skip
    sc = {
        "o_b": 0, "o_h": D, "o_t": t_stride, "do_b": 0, "do_h": D, "do_t": t_stride,
        "l_b": 0, "l_h": 1, "l_t": H_q, "h_q": H_q, "h_k": H_k, "h_v": H_v, "S_q_max": S_max,
        "S_kv_max": S_max, "has_len": 0, "len_stride": 1, "off64": off64, "q_mult": mult,
        "q_div": div, "kv_mult": mult, "kv_div": div, "ws_rows_q": rows_q, "ws_rows_kv": rows_kv,
        "zero_kv": 1, "zero_dq": 0,
    }  # fmt: skip
    grid = X.attn_bwd_prep_grid(
        spec, arch=_ARCH, s_q_max=S_max, s_kv_max=S_max, h_q=H_q, h_k=H_k, h_v=H_v,
        batch=len(lens_q), zero_kv=True,
    )  # fmt: skip
    _run("prep", spec, grid, bufs, sc)
    lse2 = ws["WS_LSE2"][: H_q * rows_q].reshape(H_q, rows_q)
    dsum = ws["WS_DSUM"][: H_q * rows_q].reshape(H_q, rows_q)
    for h, t in itertools.product(range(H_q), range(Tq)):
        live = not (h == 1 and t == 13)
        _check_stats(lse2[h, t], dsum[h, t], h, t, lse[0, h, t], live,
                     ov[0, h, t], dov[0, h, t])  # fmt: skip
    assert np.all(np.isnan(ws["WS_DQ"]))  # zero_dq = 0: untouched
    assert np.all(ws["WS_DK"][: H_k * rows_kv * D] == 0.0)
    assert np.all(ws["WS_DV"][: H_v * rows_kv * D] == 0.0)
    for k in ("WS_LSE2", "WS_DSUM", "WS_DK", "WS_DV"):
        assert np.all(np.isnan(ws[k][-CANARY:]))


def test_prep_thd_leaves_rows_outside_segments_untouched():
    """Workspace rows beyond the last segment keep their poison (max_total slack)."""
    rng = np.random.default_rng(11)
    D, H = 32, 1
    lens = [5, 3]
    T = sum(lens)
    rows = T + 20  # caller's bound larger than the tokens used
    o, do, lse, _st, _ov, _dov = _prep_inputs(rng, "fp16", 1, H, T, D, "bshd", [])
    offs = np.array([0, 5, 8], np.int32)
    ws = {
        "WS_LSE2": _nan_ws(H * rows),
        "WS_DSUM": _nan_ws(H * rows),
        "WS_DQ": _nan_ws(H * rows * D),
    }
    spec = X.AttnBwdAuxSpec(head_size=D, seq_mode="thd")
    bufs = dict(O=o, dO=do, LSE=lse[0].T.copy().reshape(-1), SEQ_Q=None, SEQ_KV=None,
                OFF_Q=offs, OFF_KV=offs, WS_DK=None, WS_DV=None, WORKLIST=None, **ws)  # fmt: skip
    sc = {"o_b": 0, "o_h": D, "o_t": H * D, "do_b": 0, "do_h": D, "do_t": H * D, "l_b": 0, "l_h": 1,
              "l_t": H, "h_q": H, "h_k": H, "h_v": H, "S_q_max": 8, "S_kv_max": 8, "q_mult": 1, "q_div": 1,
              "kv_mult": 1, "kv_div": 1, "ws_rows_q": rows, "ws_rows_kv": rows, "zero_kv": 0,
              "zero_dq": 1}  # fmt: skip
    grid = X.attn_bwd_prep_grid(spec, arch=_ARCH, s_q_max=8, s_kv_max=8, h_q=H, h_k=H,
                                h_v=H, batch=2, zero_kv=False)  # fmt: skip
    _run("prep", spec, grid, bufs, sc)
    assert np.all(np.isfinite(ws["WS_DSUM"][:T])) and np.all(
        np.isnan(ws["WS_DSUM"][T:])
    )
    assert np.all(ws["WS_DQ"][: T * D] == 0) and np.all(np.isnan(ws["WS_DQ"][T * D :]))


# ---------------------------------------------------------------------------
# convert
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "D,dtype,stage_vec,ws_layout,vec,rows_x",
    [(64, "fp16", 8, "head_major", 8, 1), (128, "bf16", 8, "token_major", 8, 1),
     (32, "bf16", 1, "head_major", 8, 1), (128, "fp16", 1, "token_major", 8, 1),
     # convert_vec and rows_per_cta tuning values
     (64, "fp16", 8, "head_major", 4, 1), (128, "bf16", 8, "token_major", 16, 1),
     (32, "bf16", 8, "head_major", 16, 2), (128, "fp16", 1, "token_major", 4, 4),
     (32, "fp16", 1, "head_major", 16, 1), (64, "bf16", 8, "token_major", 8, 4)],
)  # fmt: skip
def test_convert_batched(D, dtype, stage_vec, ws_layout, vec, rows_x):
    rng = np.random.default_rng(D + stage_vec)
    B, H, S = 3, 3, 45
    lens = np.array([45, 20, 0], np.int32)
    ws_rows = B * S - 5  # the last rows are outside the workspace: read as 0
    src = rng.standard_normal(H * ws_rows * D).astype(np.float32)
    pad = 8 if stage_vec == 8 else 3
    st = (H * S * (D + pad) + (8 if stage_vec == 8 else 1), S * (D + pad), D + pad)
    n = B * st[0]
    sentinel = _enc(np.float32(-7.0), dtype)
    dst = np.full(n + 64, sentinel)
    spec = X.AttnBwdAuxSpec(head_size=D, dtype=dtype, stage_vec=stage_vec,
                            ws_layout=ws_layout, convert_vec=vec)  # fmt: skip
    if rows_x != 1:
        rows = X.attn_bwd_aux_rows_per_cta(spec, arch=_ARCH) * rows_x
        spec = X.AttnBwdAuxSpec(head_size=D, dtype=dtype, stage_vec=stage_vec,
                                ws_layout=ws_layout, convert_vec=vec,
                                rows_per_cta=rows)  # fmt: skip
    mult = 0.5
    grid = X.attn_bwd_convert_grid(spec, arch=_ARCH, s_max=S, heads=H, batch=B)
    _run("convert", spec, grid, {"SRC": src, "DST": dst, "SEQ": lens, "OFF": None},
         {"mult": mult, "H": H, "d_b": st[0], "d_h": st[1], "d_t": st[2], "S_max": S, "has_len": 1,
              "len_stride": 1, "off64": 0, "tok_mult": 1, "tok_div": 1, "ws_rows": ws_rows})  # fmt: skip
    wsv = (
        src.reshape(H, ws_rows, D)
        if ws_layout == "head_major"
        else (src.reshape(ws_rows, H, D).transpose(1, 0, 2))
    )
    written = np.zeros(dst.shape, bool)
    for bi, h, s in itertools.product(range(B), range(H), range(S)):
        e = bi * st[0] + h * st[1] + s * st[2]
        row = bi * S + s
        if s < lens[bi] and row < ws_rows:
            want = _enc(np.float32(mult) * wsv[h, row], dtype)
        else:
            want = _enc(np.zeros(D, np.float32), dtype)
        assert np.array_equal(dst[e : e + D], want), (bi, h, s)
        written[e : e + D] = True
    assert np.all(dst[~written] == sentinel)


@pytest.mark.parametrize("vec", [8, 4, 16])
@pytest.mark.parametrize("ws_layout", ["head_major", "token_major"])
def test_convert_thd_touches_segment_rows_only(ws_layout, vec):
    rng = np.random.default_rng(5)
    D, H = 64, 2
    lens = [10, 0, 33, 4]
    T = sum(lens)
    rows = T + 6
    src = rng.standard_normal(H * rows * D).astype(np.float32)
    offs = np.concatenate([[0], np.cumsum(lens)]).astype(np.int32)
    sentinel = np.float16(-7.0)
    dst = np.full((rows + 2) * H * D, sentinel, np.float16)
    spec = X.AttnBwdAuxSpec(
        head_size=D, seq_mode="thd", ws_layout=ws_layout, convert_vec=vec
    )
    grid = X.attn_bwd_convert_grid(spec, arch=_ARCH, s_max=40, heads=H, batch=len(lens))
    _run("convert", spec, grid, {"SRC": src, "DST": dst, "SEQ": None, "OFF": offs},
         {"mult": 2.0, "H": H, "d_b": 0, "d_h": D, "d_t": H * D, "S_max": 40, "has_len": 0,
              "len_stride": 1, "off64": 0, "tok_mult": 1, "tok_div": 1, "ws_rows": rows})  # fmt: skip
    wsv = (
        src.reshape(H, rows, D)
        if ws_layout == "head_major"
        else (src.reshape(rows, H, D).transpose(1, 0, 2))
    )
    out = dst.reshape(rows + 2, H, D)
    for t, h in itertools.product(range(rows + 2), range(H)):
        if t < T:
            assert np.array_equal(
                out[t, h], (np.float32(2.0) * wsv[h, t]).astype(np.float16)
            )
        else:
            assert np.all(out[t, h] == sentinel)


# ---------------------------------------------------------------------------
# paged workspace roles
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("D,dtype", [(64, "bf16"), (128, "fp16"), (32, "fp16")])
def test_page_zero_and_convert(D, dtype):
    rng = np.random.default_rng(D)
    spec = X.AttnBwdAuxSpec(head_size=D, dtype=dtype, kv_target="paged")
    rows = X.attn_bwd_aux_rows_per_cta(spec, arch=_ARCH)
    page = max(16, rows)
    nb, h_kv, mb = 10, 2, 4
    lens = np.array([3 * page + 1, 2 * page, 0], np.int32)  # seq 0 / 1 share page 4
    bt = np.array([[4, 7, 2, 9], [4, 1, 8, 8], [3, 3, 3, 3]], np.int32)
    bt_stride = 6  # padded rows
    bt_flat = np.full(3 * bt_stride, 9999, np.int32)  # poisoned past ceil(len/page)
    for s in range(3):
        n_live = -(-int(lens[s]) // page)
        bt_flat[s * bt_stride : s * bt_stride + n_live] = bt[s, :n_live]
    page_rows = nb * page
    pdk = _nan_ws(h_kv * page_rows * D)
    pdv = _nan_ws(h_kv * page_rows * D)
    grid = X.attn_bwd_page_grid(spec, arch=_ARCH, max_blocks_per_seq=mb, page=page,
                                h_kv=h_kv, num_seqs=3)  # fmt: skip
    common = {"h_kv": h_kv, "num_seqs": 3, "max_blocks_per_seq": mb, "bt_stride": bt_stride,
                  "page": page, "page_rows": page_rows}  # fmt: skip
    _run("page_zero", spec, grid,
         {"SEQ_LENS_KV": lens, "BLOCK_TABLE": bt_flat, "WS_PDK": pdk, "WS_PDV": pdv}, common)  # fmt: skip
    live_pages = {4, 7, 2, 9, 1}
    for a in (pdk, pdv):
        v = a[: h_kv * page_rows * D].reshape(h_kv, nb, page, D)
        for p in range(nb):
            if p in live_pages:
                assert np.all(v[:, p] == 0.0), p
            else:
                assert np.all(np.isnan(v[:, p])), p
        assert np.all(np.isnan(a[-CANARY:]))
    # page convert: every slot of every referenced page, unreferenced pages untouched
    src = rng.standard_normal(h_kv * page_rows * D).astype(np.float32)
    sentinel = _enc(np.float32(-7.0), dtype)
    s_blk, s_tok, s_h = page * h_kv * D, h_kv * D, D
    dst = np.full(nb * s_blk + 64, sentinel)
    _run("page_convert", spec, grid,
         {"SRC": src, "SEQ_LENS_KV": lens, "BLOCK_TABLE": bt_flat, "DST": dst},
         dict(mult=0.25, d_blk=s_blk, d_tok=s_tok, d_h=s_h, **common))  # fmt: skip
    sv = src.reshape(h_kv, nb, page, D)
    dv = dst[: nb * s_blk].reshape(nb, page, h_kv, D)
    for p in range(nb):
        if p in live_pages:
            want = _enc(np.float32(0.25) * sv[:, p].transpose(1, 0, 2), dtype)
            assert np.array_equal(dv[p], want), p
        else:
            assert np.all(dv[p] == sentinel), p
    assert np.all(dst[nb * s_blk :] == sentinel)


@pytest.mark.parametrize("D,dtype", [(64, "bf16"), (128, "fp16")])
@pytest.mark.parametrize(
    "bad", [(10,), (11, 13), (-1,), (2**31 - 1, -(2**31))],
    ids=["past_end", "canary", "minus_one", "extremes"],
)  # fmt: skip
def test_page_roles_never_dereference_out_of_range_block_ids(D, dtype, bad):
    """A live block-table entry outside [0, num_blocks) (past the cache, in
    the canary blocks allocated after it, negative, extreme) is skipped by
    page_zero and page_convert: no workspace slot and no gradient-cache slot
    of it is written (the canary blocks keep their poison / sentinel), every
    valid page is still zeroed / converted, and block 0 is not touched on
    behalf of a bad entry."""
    rng = np.random.default_rng(D + len(bad))
    spec = X.AttnBwdAuxSpec(head_size=D, dtype=dtype, kv_target="paged")
    rows = X.attn_bwd_aux_rows_per_cta(spec, arch=_ARCH)
    page = max(16, rows)
    nb, canary_blocks, h_kv, mb = 10, 4, 2, 4
    lens = np.array([3 * page + 1, 2 * page, 4 * page], np.int32)
    bt = np.array([[4, 7, 2, 9], [5, 1, 8, 8], [3, 6, 0, 0]], np.int32)
    bad_slots = [(0, 1), (2, 2), (1, 0), (2, 3)][: len(bad)]
    for (s, j), v in zip(bad_slots, bad):
        bt[s, j] = v
    live = set()
    for s in range(3):
        for j in range(-(-int(lens[s]) // page)):
            if 0 <= int(bt[s, j]) < nb:
                live.add(int(bt[s, j]))
    bt_flat = bt.reshape(-1).copy()
    rows_all = (nb + canary_blocks) * page
    page_rows = nb * page
    pdk = np.full(h_kv * rows_all * D, np.nan, np.float32)
    pdv = np.full(h_kv * rows_all * D, np.nan, np.float32)
    grid = X.attn_bwd_page_grid(spec, arch=_ARCH, max_blocks_per_seq=mb, page=page,
                                h_kv=h_kv, num_seqs=3)  # fmt: skip
    common = {"h_kv": h_kv, "num_seqs": 3, "max_blocks_per_seq": mb, "bt_stride": mb,
              "page": page, "page_rows": page_rows}  # fmt: skip
    _run("page_zero", spec, grid,
         {"SEQ_LENS_KV": lens, "BLOCK_TABLE": bt_flat, "WS_PDK": pdk, "WS_PDV": pdv}, common)  # fmt: skip
    for a in (pdk, pdv):
        v = a[: h_kv * page_rows * D].reshape(h_kv, nb, page, D)
        for p in range(nb):
            if p in live:
                assert np.all(v[:, p] == 0.0), p
            else:
                assert np.all(np.isnan(v[:, p])), p
        assert np.all(np.isnan(a[h_kv * page_rows * D :]))
    src = rng.standard_normal(h_kv * page_rows * D).astype(np.float32)
    src_all = np.concatenate(
        [src, np.full(h_kv * (rows_all - page_rows) * D, 5.0, np.float32)]
    )
    sentinel = _enc(np.float32(-7.0), dtype)
    s_blk, s_tok, s_h = page * h_kv * D, h_kv * D, D
    dst = np.full((nb + canary_blocks) * s_blk, sentinel)
    _run("page_convert", spec, grid,
         {"SRC": src_all, "SEQ_LENS_KV": lens, "BLOCK_TABLE": bt_flat, "DST": dst},
         dict(mult=0.25, d_blk=s_blk, d_tok=s_tok, d_h=s_h, **common))  # fmt: skip
    sv = src.reshape(h_kv, nb, page, D)
    dv = dst.reshape(nb + canary_blocks, page, h_kv, D)
    for p in range(nb + canary_blocks):
        if p in live:
            want = _enc(np.float32(0.25) * sv[:, p].transpose(1, 0, 2), dtype)
            assert np.array_equal(dv[p], want), p
        else:
            assert np.all(dv[p] == sentinel), p
