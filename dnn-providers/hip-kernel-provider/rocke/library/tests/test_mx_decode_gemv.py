# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""gfx950 MX decode GEMV (``kernels.gfx950.mx_decode_gemv``).

CPU: spec validation, deviation-only kernel names, the IR of every knob value
(instruction counts, Python-only bookkeeping), the host references (bf16 key
order, key folding) and the kernarg packing.

gfx950 (skips elsewhere; ``ROCKE_REQUIRE_GFX950=1`` fails instead):

* an exact fixture (every partial sum exact in f32): every knob value equals
  the f64 reference bit for bit, rows 1..rows_max, forward and reversed panel
  order, a K with a partial last load group;
* random data: every knob value that keeps the MFMA order (all but
  ``korder``) is bitwise equal to the default build, within a stated
  tolerance of f64, and each ``korder`` within the same tolerance;
* the argmax epilogues (``keys`` folded on the host, ``fused``) equal the
  host argmax of the same-order logits, with planted ties (lowest index wins)
  and ``-0`` / ``+0`` maxima; the ``fused`` arrival counter over many
  launches.
"""

import ctypes
import os
import struct
from pathlib import Path

import numpy as np
import pytest
from kernels.gfx950 import mx_decode_gemv as M
from rocke.core.ir import uses_python_only_ops
from rocke.core.lower_llvm import _lower_kernel_to_llvm_python

HEAD = dict(n=248320, k=2560)

# ----------------------------------------------------------------- CPU


def test_defaults_valid_and_named():
    sp = M.MxDecodeGemvSpec(**HEAD)
    assert M.is_valid_spec(sp) == (True, "")
    assert sp.kernel_name() == "mxgemv_gfx950_n248320_k2560_r4_bf16_logits_pyonly"
    assert M.MxDecodeGemvSpec(**HEAD, waves=8, dg=3).kernel_name().endswith("_w8_dg3_pyonly")
    names = {
        M.MxDecodeGemvSpec(**HEAD, **kw).kernel_name()
        for kw in (
            {},
            {"korder": "lane"},
            {"x_src": "global"},
            {"waves": 8},
            {"nt": 4},
            {"dg": 1},
            {"ctas": 512},
            {"cta_map": "blocked"},
            {"cpol": "nt"},
            {"xpad": 0},
            {"waves_per_eu": 2},
            {"setprio": 1},
            {"sched": "q"},
            {"sload": "bcast"},
            {"wload": "x1"},
            {"smask": True},
            {"epilogue": "keys"},
            {"rows_max": 16},
            {"nt": 2},
            {"fsync": "counter"},
        )
    }
    assert len(names) == 20


def test_balanced_ctas():
    # the head: 243 CTAs (15.97 panels a slot); a grid that fits: one panel a slot
    assert M.balanced_ctas(248320) == 243
    assert M.balanced_ctas(16384) == 256
    assert M.balanced_ctas(2560, waves=8, ksplit=8) == 160
    assert M.balanced_ctas(320, waves=8, ksplit=4) == 10
    g = M.balanced_ctas(13952, waves=8, ksplit=4)
    assert 231 <= g <= 256


@pytest.mark.parametrize(
    "kw, why",
    [
        ({"k": 2500}, "multiple of 128"),
        ({"n": 248320 + 16, "nt": 2}, "multiple of 16 * nt"),
        ({"rows_max": 8}, "rows_max"),
        ({"dg": 6}, "dg"),
        ({"korder": "x"}, "korder"),
        ({"epilogue": "fused", "ctas": 300, "fsync": "counter"}, "power-of-two"),
        ({"epilogue": "fused", "ctas": 512, "waves": 4}, "ctas <= threads"),
        ({"epilogue": "fused", "ctas": 512, "waves": 4, "fsync": "epoch"}, "ctas <= threads"),
        ({"fsync": "x"}, "fsync"),
        ({"pro": "x"}, "pro"),
        ({"rows_max": 16, "xpad": 8192}, "LDS"),
        ({"sched": "x"}, "sched"),
        ({"cpol": "x"}, "cpol"),
        ({"wload": "x1", "korder": "lane"}, "korder mono"),
    ],
)
def test_invalid_specs(kw, why):
    d = dict(HEAD)
    d.update(kw)
    ok, msg = M.is_valid_spec(M.MxDecodeGemvSpec(**d))
    assert not ok and why in msg
    with pytest.raises(ValueError):
        M.build_mx_decode_gemv(M.MxDecodeGemvSpec(**d))


@pytest.mark.parametrize(
    "kw",
    [
        {},
        {"korder": "lane"},
        {"x_src": "global"},
        {"nt": 4, "dg": 1},
        {"nt": 1, "dg": 5, "sched": "none"},
        {"epilogue": "keys", "cta_map": "blocked"},
        {"epilogue": "fused", "waves": 8, "ctas": 512, "rows_max": 16},
        {"setprio": 2, "cpol": "nt", "waves_per_eu": 2},
        {"sload": "x4"},
        {"wload": "x1", "sload": "x4"},
        {"korder": "lane", "sload": "x4"},
        {"ksplit": 2, "waves": 8, "epilogue": "fused"},
        {"epilogue": "fused", "fsync": "epoch", "ctas": 243},
        {"epilogue": "fused", "fsync": "counter", "ctas": 256},
        {"epilogue": "fused", "fsync": "counter64", "ctas": 243},
        {"sload": "bcast", "setprio": 0},
        {"pro": "w_first"},
        {"pro": "w_first", "epilogue": "keys", "rows_max": 16},
        {"sload": "bcast", "korder": "lane"},
    ],
)
def test_ir(kw):
    sp = M.MxDecodeGemvSpec(**HEAD, **kw)
    k = M.build_mx_decode_gemv(sp)
    assert uses_python_only_ops(k)  # always a Python-engine build
    ll = _lower_kernel_to_llvm_python(k, arch="gfx950", llvm_flavor="llvm22")
    mfma = ll.count("call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.bf16(")
    nq = sp.nqk  # 128-k chunks per K part
    assert mfma == 4 * nq * sp.nt  # one bf16 MFMA per 32 k per tile
    assert ll.count("call <2 x bfloat> @llvm.amdgcn.cvt.scalef32.pk.bf16.fp4(") == 16 * nq * sp.nt
    swaps = ll.count("call { i32, i32 } @llvm.amdgcn.permlane16.swap(")
    # one swap per scale broadcast per group and tile, two per 4 x 4 transpose
    want = (sp.ngroups * sp.nt if sp.sload == "bcast" else 0) + (
        2 * nq * sp.nt if sp.korder == "mono" and sp.wload == "x4" else 0
    )
    assert swaps == want
    if sp.setprio:
        assert "s.setprio" in ll


def test_bf16_key_order():
    vals = np.array([-np.inf, -3.0, -1.0, -0.0, 0.0, 1e-30, 1.0, 2.5, np.inf], np.float32)
    bits = (vals.view(np.uint32) >> 16).astype(np.uint32)
    keys = [M.key_of_bf16_bits(int(b) << 16) for b in bits]
    assert keys == sorted(keys)
    assert keys[3] == keys[4] == 0  # -0 == +0


def test_ref_fold_and_argmax():
    rng = np.random.default_rng(0)
    logits = rng.standard_normal((3, 1000)).astype(np.float32)
    logits[1, 17] = logits[1, 500] = 40.0  # tie: lowest index wins
    b16 = (logits.view(np.uint32) >> 16).astype(np.uint16)
    am = M.ref_argmax(b16)
    assert am[1] == 17
    ctas = 10
    keys = []
    for r in range(3):
        for c in range(ctas):
            sl = slice(c * 100, (c + 1) * 100)
            kk = np.array([M.key_of_bf16_bits(int(v) << 16) for v in b16[r, sl]])
            j = int(np.argmax(kk))
            keys.append((int(kk[j]) << 32) | (999 - (c * 100 + j)))
    assert np.array_equal(M.ref_fold_keys(np.array(keys, np.int64), 3, ctas, 1000), am)


def test_launch_args_layout():
    sp = M.MxDecodeGemvSpec(**HEAD, epilogue="fused")
    names = [n for n, _ in M.signature(sp)]
    assert names == ["x", "w", "s", "y", "ws", "m", "sxm", "sym", "rev"]
    raw = M.launch_args(sp, x=1, w=2, s=3, y=4, ws=5, m=6, sxm=7, sym=8, rev=9)
    assert raw == struct.pack("<QQQQQiiii", 1, 2, 3, 4, 5, 6, 7, 8, 9)


# ----------------------------------------------------------------- gfx950


def _require_gfx950():
    strict = os.environ.get("ROCKE_REQUIRE_GFX950") == "1"
    if not Path("/dev/kfd").exists():
        if strict:
            pytest.fail("requires a gfx950 GPU")
        pytest.skip("requires an AMD GPU")
    from rocke.runtime.hip_module import get_device_arch

    arch = get_device_arch(0)
    if arch != "gfx950":
        if strict:
            pytest.fail(f"requires gfx950, found {arch}")
        pytest.skip(f"requires gfx950, found {arch}")


_ART = {}


def _compile(sp):
    from rocke.helpers import compile_kernel

    name = sp.kernel_name()
    if name not in _ART:
        _ART[name] = compile_kernel(M.build_mx_decode_gemv(sp), arch="gfx950", backend="python")
    return _ART[name]


class _Dev:
    def __init__(self):
        from rocke.runtime.hip_module import Runtime

        self.rt = Runtime()
        self.bufs = []

    def put(self, arr):
        arr = np.ascontiguousarray(arr)
        p = self.rt.alloc(max(arr.nbytes, 4))
        self.bufs.append(p)
        self.rt.memcpy_h2d(p, (ctypes.c_uint8 * arr.nbytes).from_buffer_copy(arr.tobytes()), arr.nbytes)
        return p

    def zeros(self, nbytes):
        p = self.rt.alloc(nbytes)
        self.bufs.append(p)
        self.rt.memset(p, 0, nbytes)
        return p

    def get(self, p, dtype, shape):
        nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
        host = (ctypes.c_uint8 * nbytes)()
        self.rt.memcpy_d2h(host, p, nbytes)
        return np.frombuffer(bytes(host), dtype).reshape(shape)

    def free(self):
        for p in self.bufs:
            self.rt.free(p)


def _run(dev, sp, x_bits, codes, scales, rev=0, ws=None, launches=1, xs=None):
    """Launch ``sp``; returns bf16 logits bits, int64 keys or int32 argmax.
    ``act="mxfp8"``: ``x_bits`` are the E4M3 codes and ``xs`` the E8M0 scales."""
    art = _compile(sp)
    m = x_bits.shape[0]
    n = codes.shape[0]
    px, pw, ps = dev.put(x_bits), dev.put(codes), dev.put(scales)
    pxs = dev.put(xs) if xs is not None else 0
    if sp.epilogue == "logits":
        py, out = dev.zeros(m * n * 2), (np.uint16, (m, n))
    elif sp.epilogue == "keys":
        py, out = dev.zeros(m * sp.ctas * 8), (np.int64, (m * sp.ctas,))
    else:
        py, out = dev.zeros(m * 4), (np.int32, (m,))
        if ws is None:
            ws = dev.zeros(M.fused_workspace_bytes(sp))
    module = dev.rt.load_module(art.hsaco)
    try:
        fn = module.get_function(art.kernel_name)
        g, blk = M.grid(sp)
        args = M.launch_args(
            sp, x=px, w=pw, s=ps, xs=pxs, y=py, ws=ws or 0, m=m, sxm=x_bits.shape[1], sym=n, rev=rev
        )
        for _ in range(launches):
            dev.rt.launch_blocking(fn, g, blk, args)
        res = dev.get(py, *out)
    finally:
        module.unload()
    return res, ws


def _bf16_bits(x):
    """f32 -> bf16 bits, round to nearest even (finite values)."""
    u = np.asarray(x, np.float32).view(np.uint32).astype(np.uint64)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)


def _exact_case(rng, m, n, k):
    """Every partial sum exact in f32: x in small multiples of 1/4, weights
    E2M1 times 2^0..2^2, |sum| < 2^20."""
    x = rng.integers(-8, 9, (m, k)).astype(np.float32) / 4
    codes = rng.integers(0, 256, (n, k // 2), dtype=np.uint8)
    scales = rng.integers(127, 130, (n, k // 32), dtype=np.uint8)
    return _bf16_bits(x), x, codes, scales


SMALL = dict(n=2048, ctas=8)


@pytest.mark.parametrize("k", [2560, 384])
@pytest.mark.parametrize(
    "kw",
    [
        {},
        {"korder": "lane"},
        {"x_src": "global"},
        {"nt": 4, "dg": 1},
        {"nt": 1, "dg": 3, "sched": "none"},
        {"cta_map": "blocked", "waves": 8, "sched": "q"},
        {"rows_max": 16},
        {"cpol": "nt", "setprio": 1, "xpad": 0},
        {"sload": "x4", "wload": "x1"},
        {"korder": "lane", "sload": "x4"},
        {"smask": True},
        {"smask": True, "korder": "lane"},
        {"ksplit": 2, "waves": 8},
        {"ksplit": 4, "waves": 4, "dg": 1, "cta_map": "blocked"},
        {"sload": "bcast", "setprio": 0, "nt": 2, "ctas": 256},
        {"sload": "bcast", "korder": "lane"},
        {"pro": "w_first"},
        {"pro": "w_first", "rows_max": 16, "ksplit": 2, "waves": 8},
    ],
)
def test_gfx950_exact(kw, k):
    _require_gfx950()
    d = dict(SMALL, k=k)
    d.update(kw)
    ks = d.get("ksplit", 1)
    if (k // 128) % ks:
        pytest.skip("k does not split into ksplit parts")
    if d.get("dg", 2) > -(-(k // 128 // ks) // 4):
        d["dg"] = -(-(k // 128 // ks) // 4)
    sp = M.MxDecodeGemvSpec(**d)
    rng = np.random.default_rng(1)
    dev = _Dev()
    try:
        for m in sorted({1, 3, sp.rows_max}):
            xb, x, codes, scales = _exact_case(rng, m, sp.n, k)
            ref = M.ref_logits_f32(x, codes, scales)
            for rev in (0, 1):
                got, _ = _run(dev, sp, xb, codes, scales, rev=rev)
                assert np.array_equal(got, _bf16_bits(ref.astype(np.float32))), (m, rev)
    finally:
        dev.free()


ORDER_KNOBS = [
    {"x_src": "global"},
    {"nt": 4, "dg": 1},
    {"nt": 1, "dg": 5, "sched": "none"},
    {"cta_map": "blocked", "waves": 8},
    {"ctas": 3, "sched": "q"},
    {"rows_max": 16},
    {"sload": "x4"},
    {"wload": "x1"},
]
#: knobs that change the f32 summation order (within the tolerance, not bitwise)
ORDER_CHANGING = [{"korder": "lane"}, {"ksplit": 2, "waves": 8}, {"ksplit": 4, "waves": 4}]


def test_gfx950_random_order_invariance():
    _require_gfx950()
    rng = np.random.default_rng(2)
    n, k, m = 4096, 2560, 4
    x = (rng.standard_normal((m, k)) * 0.5).astype(np.float32)
    xb = _bf16_bits(x)
    xf = (xb.astype(np.uint32) << 16).view(np.float32)
    codes = rng.integers(0, 256, (n, k // 2), dtype=np.uint8)
    scales = rng.integers(112, 124, (n, k // 32), dtype=np.uint8)
    ref = M.ref_logits_f32(xf, codes, scales)
    tol = 1e-5 * np.abs(ref).max(axis=1, keepdims=True)
    dev = _Dev()
    try:
        base, _ = _run(dev, M.MxDecodeGemvSpec(n=n, k=k), xb, codes, scales)
        others = [_run(dev, M.MxDecodeGemvSpec(n=n, k=k, **kw), xb, codes, scales)[0] for kw in ORDER_CHANGING]
        for got in (base, *others):
            f = (got.astype(np.uint32) << 16).view(np.float32)
            # bf16 output: half an ulp of the output plus the f32 summation
            assert np.all(np.abs(f - ref) <= np.abs(ref) * 2.0**-8 + tol)
        for kw in ORDER_KNOBS:
            got, _ = _run(dev, M.MxDecodeGemvSpec(n=n, k=k, **kw), xb, codes, scales, rev=1)
            assert np.array_equal(got, base), kw
    finally:
        dev.free()


@pytest.mark.parametrize("ctas", [16, 13])
@pytest.mark.parametrize("ksplit", [1, 2])
@pytest.mark.parametrize("korder", ["mono", "lane"])
@pytest.mark.parametrize("rows_max", [4, 16])
def test_gfx950_argmax(korder, rows_max, ksplit, ctas):
    _require_gfx950()
    rng = np.random.default_rng(3)
    n, k = 4096, 512
    # a power-of-two grid takes either fused synchronization, any other only the epoch
    fsync = "counter" if ctas & (ctas - 1) == 0 else "epoch"
    codes = rng.integers(0, 256, (n, k // 2), dtype=np.uint8)
    scales = rng.integers(115, 125, (n, k // 32), dtype=np.uint8)
    # planted ties: column 3000 duplicates column 700 (lowest index must win)
    codes[3000] = codes[700]
    scales[3000] = scales[700]
    dev = _Dev()
    try:
        common = dict(n=n, k=k, rows_max=rows_max, korder=korder, ctas=ctas, dg=1, ksplit=ksplit)
        for m in sorted({1, 2, 4, rows_max}):
            x = (rng.standard_normal((m, k)) * 0.5).astype(np.float32)
            # row 0 aims at the duplicated column: x = that column's weights
            vals = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6])
            c = np.stack([codes[700] & 15, codes[700] >> 4], -1).reshape(-1)
            x[0] = vals[c] * 4
            xb = _bf16_bits(x)
            logits, _ = _run(dev, M.MxDecodeGemvSpec(**common), xb, codes, scales)
            want = M.ref_argmax(logits)
            assert want[0] == 700
            for rev in (0, 1):
                keys, _ = _run(dev, M.MxDecodeGemvSpec(**common, epilogue="keys"), xb, codes, scales, rev=rev)
                assert np.array_equal(M.ref_fold_keys(keys, m, ctas, n), want), (m, rev)
                for fs in sorted({fsync, "epoch", "counter64"}):
                    sp = M.MxDecodeGemvSpec(**common, epilogue="fused", fsync=fs)
                    fused, _ = _run(dev, sp, xb, codes, scales, rev=rev)
                    assert np.array_equal(fused, want), (m, rev, fs)
        # all-zero logits (x = 0, +0 and -0 products): the answer is column 0
        xb = _bf16_bits(np.zeros((1, k), np.float32))
        xb[0, ::2] = 0x8000  # -0
        fused, _ = _run(dev, M.MxDecodeGemvSpec(**common, epilogue="fused", fsync=fsync), xb, codes, scales)
        assert fused[0] == 0
    finally:
        dev.free()


@pytest.mark.parametrize(
    "fsync, ctas", [("counter", 32), ("epoch", 32), ("epoch", 24), ("counter64", 32), ("counter64", 24)]
)
def test_gfx950_fused_counter(fsync, ctas):
    _require_gfx950()
    rng = np.random.default_rng(4)
    n, k, launches = 4096, 512, 64
    sp = M.MxDecodeGemvSpec(n=n, k=k, ctas=ctas, dg=1, epilogue="fused", fsync=fsync)
    codes = rng.integers(0, 256, (n, k // 2), dtype=np.uint8)
    scales = rng.integers(115, 125, (n, k // 32), dtype=np.uint8)
    xb = _bf16_bits((rng.standard_normal((3, k)) * 0.5).astype(np.float32))
    dev = _Dev()
    try:
        logits, _ = _run(dev, M.MxDecodeGemvSpec(n=n, k=k, dg=1), xb, codes, scales)
        want = M.ref_argmax(logits)
        got, ws = _run(dev, sp, xb, codes, scales, launches=launches)
        assert np.array_equal(got, want)
        words = dev.get(ws, np.uint32, (2,))
        if fsync == "counter":  # never reset: every arrival counted
            assert int(words[0]) == launches * sp.ctas
        elif fsync == "counter64":  # the same in 64 bits (low word, high word)
            assert (int(words[0]), int(words[1])) == (launches * sp.ctas, 0)
        else:  # reset by every launch's last CTA, which advances the epoch
            assert (int(words[0]), int(words[1])) == (0, launches)
    finally:
        dev.free()


@pytest.mark.parametrize(
    "ctas, epilogue, fsync",
    [(384, "keys", None), (485, "keys", None), (243, "fused", "epoch"), (243, "fused", "counter64")],
)
def test_gfx950_argmax_uneven_waves(ctas, epilogue, fsync):
    """Waves of a CTA leave the panel loop at different times (unequal trip counts):
    the argmax's LDS reduction must not overlay the x rows a slower wave still reads
    (regression: ``red`` reused ``xs``'s LDS and gave wrong keys at ctas = 384 / 485)."""
    _require_gfx950()
    rng = np.random.default_rng(9)
    n, k, m = 124160, 2560, 4
    codes = rng.integers(0, 256, (n, k // 2), dtype=np.uint8)
    scales = rng.integers(112, 124, (n, k // 32), dtype=np.uint8)
    xb = _bf16_bits((rng.standard_normal((m, k)) * 0.5).astype(np.float32))
    common = dict(n=n, k=k, nt=1, ctas=ctas)
    dev = _Dev()
    try:
        logits, _ = _run(dev, M.MxDecodeGemvSpec(**common), xb, codes, scales)
        want = M.ref_argmax(logits)
        fs = {"fsync": fsync} if fsync else {}
        sp = M.MxDecodeGemvSpec(**common, epilogue=epilogue, **fs)
        for rev in (0, 1):
            for _ in range(3):
                got, _ = _run(dev, sp, xb, codes, scales, rev=rev)
                if epilogue == "keys":
                    got = M.ref_fold_keys(got, m, ctas, n)
                assert np.array_equal(got, want), (rev, got, want)
    finally:
        dev.free()


_E4M3_SMALL = np.array([0x00, 0x30, 0x38, 0x3C, 0x40, 0xB0, 0xB8, 0xBC, 0xC0], np.uint8)


@pytest.mark.parametrize(
    "kw",
    [
        {},
        {"x_src": "global"},
        {"nt": 4, "dg": 1, "sload": "x4"},
        {"rows_max": 16, "sched": "q"},
        {"epilogue": "fused", "ctas": 8},
        {"pro": "w_first"},
    ],
)
def test_gfx950_mxfp8_exact(kw):
    """``act="mxfp8"``: every partial sum exact in f32, so equal to f64 bit for bit."""
    _require_gfx950()
    d = dict(SMALL, k=2560, act="mxfp8")
    d.update(kw)
    sp = M.MxDecodeGemvSpec(**d)
    rng = np.random.default_rng(5)
    dev = _Dev()
    try:
        for m in sorted({1, 4, sp.rows_max}):
            xq = _E4M3_SMALL[rng.integers(0, len(_E4M3_SMALL), (m, sp.k))]
            xs = rng.integers(125, 129, (m, sp.k // 32), dtype=np.uint8)
            codes = rng.integers(0, 256, (sp.n, sp.k // 2), dtype=np.uint8)
            scales = rng.integers(127, 130, (sp.n, sp.k // 32), dtype=np.uint8)
            ref = M.ref_logits_mxfp8(xq, xs, codes, scales).astype(np.float32)
            for rev in (0, 1):
                got, _ = _run(dev, sp, xq, codes, scales, rev=rev, xs=xs)
                if sp.epilogue == "logits":
                    assert np.array_equal(got, _bf16_bits(ref)), (m, rev)
                else:
                    assert np.array_equal(got, M.ref_argmax(_bf16_bits(ref))), (m, rev)
    finally:
        dev.free()


def test_mxfp8_ir():
    sp = M.MxDecodeGemvSpec(**HEAD, act="mxfp8")
    ll = _lower_kernel_to_llvm_python(M.build_mx_decode_gemv(sp), arch="gfx950", llvm_flavor="llvm22")
    calls = ll.count("call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v8i32.v4i32(")
    assert calls == 20 * sp.nt  # one scaled MFMA per 128 k per tile
    assert "cvt.scalef32.pk.bf16.fp4(" not in ll
    assert [n for n, _ in M.signature(sp)][:5] == ["x", "w", "s", "xs", "y"]
    assert not M.is_valid_spec(M.MxDecodeGemvSpec(**HEAD, act="mxfp8", korder="lane"))[0]


def test_ref_fp8e4m3():
    assert M.ref_fp8e4m3(np.array([0x38, 0x30, 0xC0, 0x7E, 0x01]))[:5].tolist()[:4] == [1.0, 0.5, -2.0, 448.0]
    assert M.ref_fp8e4m3(np.array([0x01]))[0] == 2.0**-9


@pytest.mark.parametrize("kw", [{}, {"rows_max": 16, "epilogue": "fused", "ctas": 8}, {"ksplit": 2, "waves": 8}])
def test_gfx950_bf16q8(kw):
    """``act="bf16q8"``: x quantized in the kernel by the ``mx_quant`` rule. Values whose
    quantization is exact give the f64 product bit for bit; random values give the bits
    of the ``act="mxfp8"`` build fed the host quantizer's codes and scales."""
    _require_gfx950()
    from rocke.helpers import mx_quant as MQ

    d = dict(SMALL, k=2560, act="bf16q8")
    d.update(kw)
    sp = M.MxDecodeGemvSpec(**d)
    rng = np.random.default_rng(6)
    codes = rng.integers(0, 256, (sp.n, sp.k // 2), dtype=np.uint8)
    dev = _Dev()
    try:
        for m in sorted({1, 4, sp.rows_max}):
            x = rng.choice(np.array([0, 0.5, 1, 1.5, 2, -0.5, -1, -1.5, -2], np.float32), (m, sp.k))
            scales = rng.integers(127, 130, (sp.n, sp.k // 32), dtype=np.uint8)
            ref = M.ref_logits_f32(x, codes, scales).astype(np.float32)
            got, _ = _run(dev, sp, _bf16_bits(x), codes, scales)
            want = _bf16_bits(ref) if sp.epilogue == "logits" else M.ref_argmax(_bf16_bits(ref))
            assert np.array_equal(got, want), m
            if sp.epilogue != "logits":
                continue
            xr = (rng.standard_normal((m, sp.k)) * 0.5).astype(np.float32)
            xb = _bf16_bits(xr)
            xf = (xb.astype(np.uint32) << 16).view(np.float32)
            q, e = MQ.ref_quantize_mxfp8(xf.reshape(m, sp.k // 32, 32))
            q = q.reshape(m, sp.k)
            scales = rng.integers(112, 124, (sp.n, sp.k // 32), dtype=np.uint8)
            got, _ = _run(dev, sp, xb, codes, scales)
            # the in-kernel quantizer is the host's: the same build fed the host's codes and
            # scales (act="mxfp8") gives the same bits. Against the f64 product of the quantized
            # operands: half a bf16 ulp plus 2^-16 of sum |x w| (the scaled MFMA truncates small
            # products inside a chunk; see the module docstring), not an f32-order bound.
            sp8 = M.MxDecodeGemvSpec(**dict(d, act="mxfp8"))
            want8, _ = _run(dev, sp8, q, codes, scales, xs=e)
            assert np.array_equal(got, want8), m
            ref = M.ref_logits_mxfp8(q, e, codes, scales)
            f = (got.astype(np.uint32) << 16).view(np.float32)
            xa = np.abs(M.ref_fp8e4m3(q)) * np.exp2(e.astype(np.float64) - 127.0).repeat(32, axis=1)
            mag = M.ref_logits_f32(xa, codes & 0x77, scales)  # sum |x w| (E2M1 sign bits cleared)
            assert np.all(np.abs(f - ref) <= 2.0**-8 * np.abs(ref) + 2.0**-16 * mag), m
    finally:
        dev.free()
