# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""CPU tests of the gfx950 MXFP6 pack / quantize kernels (``kernels/gfx950/mxfp6_gemm/pack.py``).

The host references (``pack.quantize_ref`` = ``layout.quantize_bf16`` for bf16, its f32 twin) are checked against an
independent float64 OCP oracle on every corner of ``pack.edge_activations``; the software encoder's one-add RNE
trick is checked exhaustively in a numpy model; the builders' validation, names and lowering (hardware conversion
present, nt metadata) are checked without a GPU. The on-GPU byte identity is ``test_mxfp6_pack_gpu.py``.
"""

from __future__ import annotations

import numpy as np
import pytest
from kernels.gfx950.mxfp6_gemm import layout as L
from kernels.gfx950.mxfp6_gemm import pack as P

FMTS = ("e2m3", "e3m2")
SRCS = ("bf16", "f32")


# ----------------------------------------------------------------- independent oracle


def oracle_quantize(x, fmt: str, src: str):
    """Independent OCP MX oracle (float64, no code shared with layout.py / pack.py): per 32-block, a block with an
    inf / NaN -> byte 0xFF and zero codes; amax = max |x|; zero block -> byte 0; otherwise the shared exponent
    floor(log2 amax) - emax (frexp, exact), stored byte clamped to 0..254 and that clamped exponent used; codes =
    nearest of the 32 levels (definition: E2M3 = m/8 | 2^(e-1)(1+m/8), E3M2 = m/16 | 2^(e-3)(1+m/4)) of
    min(|x| 2^-se, max), ties to the even code, sign from the input's sign bit. Returns (codes [rows, K], scales)."""
    if src == "bf16":
        bits = np.asarray(x, np.uint16).astype(np.uint32) << 16
    else:
        bits = np.asarray(x, np.float32).view(np.uint32)
    v = bits.view(np.float32).astype(np.float64)
    neg = (bits >> 31).astype(bool)
    rows, k = v.shape
    vb = v.reshape(rows, k // 32, 32)
    bad = ~np.isfinite(vb).all(axis=-1)
    a = np.abs(np.where(np.isfinite(vb), vb, 0.0))
    amax = a.max(axis=-1)
    emax = {"e2m3": 2, "e3m2": 4}[fmt]
    e = np.frexp(np.where(amax > 0, amax, 1.0))[1] - 1
    byte = np.clip(e - emax + 127, 0, 254)
    byte = np.where(amax == 0, 0, byte)
    byte = np.where(bad, 255, byte).astype(np.uint8)
    se = np.where(bad | (amax == 0), 0, byte.astype(np.int64) - 127)
    if fmt == "e2m3":
        lv = np.array([m / 8 for m in range(8)] + [2.0 ** (ee - 1) * (1 + m / 8) for ee in range(1, 4)
                                                    for m in range(8)])
    else:
        lv = np.array([m / 16 for m in range(4)] + [2.0 ** (ee - 3) * (1 + m / 4) for ee in range(1, 8)
                                                     for m in range(4)])
    s = np.minimum(a * np.ldexp(1.0, -se)[..., None], lv[-1])
    d = np.abs(s[..., None] - lv)
    best = d.min(axis=-1, keepdims=True)
    cand = d == best
    even = cand & (np.arange(32) % 2 == 0)
    code = np.where(even.any(axis=-1), even.argmax(axis=-1), cand.argmax(axis=-1))
    code = code | np.where(neg.reshape(vb.shape), 32, 0)
    code = np.where(bad[..., None], 0, code)
    return code.reshape(rows, k).astype(np.uint8), byte


def _check_ref(x, fmt, src):
    q = P.quantize_ref(x, fmt, src)
    codes, scales = oracle_quantize(x, fmt, src)
    assert np.array_equal(q.scales, scales), f"{int((q.scales != scales).sum())} scale bytes differ"
    got = L.unpack_codes(q.codes, x.shape[1])
    assert np.array_equal(got, codes), f"{int((got != codes).sum())} codes differ"
    return q


@pytest.mark.parametrize("src", SRCS)
@pytest.mark.parametrize("fmt", FMTS)
def test_reference_matches_independent_oracle(fmt, src):
    """quantize_ref == the float64 oracle on every corner of edge_activations (and Gaussian rows over 2^-150..2^120)."""
    for seed in (3, 4):
        _check_ref(P.edge_activations(256, 1024, fmt=fmt, src=src, seed=seed), fmt, src)


@pytest.mark.parametrize("fmt", FMTS)
def test_bf16_reference_is_layout_quantizer_and_f32_agrees_on_bf16_values(fmt):
    xb = P.edge_activations(128, 512, fmt=fmt, src="bf16")
    q = P.quantize_ref(xb, fmt, "bf16")
    q2 = L.quantize_bf16(xb, fmt)
    assert np.array_equal(q.codes, q2.codes) and np.array_equal(q.scales, q2.scales)
    q3 = P.quantize_ref(L.bf16_bits_to_f32(xb), fmt, "f32")
    assert np.array_equal(q.codes, q3.codes) and np.array_equal(q.scales, q3.scales)


@pytest.mark.parametrize("src", SRCS)
@pytest.mark.parametrize("fmt", FMTS)
def test_ocp_scale_rule_and_corners(fmt, src):
    """Explicit corners: byte = floor(log2 amax) - emax + 127 (emax 2 / 4), zero / -0 block -> 0 with codes 0 / 32,
    NaN / inf blocks -> 0xFF with zero codes, saturation, ties to even, subnormal amax -> byte 0."""
    emax = {"e2m3": 2, "e3m2": 4}[fmt]
    mx = P.fmt_levels(fmt)[-1]
    x = np.zeros((1, 32 * 8), np.float32)
    x[0, 0:32] = 0.0
    x[0, 32:64] = -0.0
    x[0, 64] = np.nan
    x[0, 96] = -np.inf
    x[0, 128:160] = np.linspace(-3, 5, 32) * 2.0 ** 10  # amax 5 * 2^10 -> floor(log2) = 12
    x[0, 160:192] = mx * 1.03  # saturation: a in (max, 2^(emax+1))
    x[0, 192] = mx
    x[0, 193:224] = (P.fmt_levels(fmt)[8] + P.fmt_levels(fmt)[9]) / 2  # tie between codes 8 and 9 -> 8
    x[0, 224:256] = 2.0 ** -140 if src == "f32" else 2.0 ** -133
    xin = x if src == "f32" else L.f32_to_bf16_bits(x)
    q = _check_ref(xin, fmt, src)
    codes = L.unpack_codes(q.codes, 256)
    sc = q.scales[0]
    assert sc[0] == 0 and (codes[0, 0:32] == 0).all()
    assert sc[1] == 0 and (codes[0, 32:64] == 32).all()
    assert sc[2] == 255 and (codes[0, 64:96] == 0).all()
    assert sc[3] == 255 and (codes[0, 96:128] == 0).all()
    assert sc[4] == 12 - emax + 127
    assert (codes[0, 160:192] == 31).all() and sc[5] == int(np.floor(np.log2(mx * 1.03))) - emax + 127
    assert codes[0, 192] == 31 and (codes[0, 193:224] == 8).all()
    assert sc[7] == 0  # subnormal amax: floor(log2) - emax < -127


@pytest.mark.parametrize("fmt", FMTS)
def test_sw_encoder_one_add_rne_model_is_exact(fmt):
    """numpy model of pack._codes_sw (f32 ops, the same bit tricks) == layout.fmt_encode for every bf16 value at
    every shared exponent the kernels use (-127..125) -- the formula, independent of the IR. Domain: the scale rule
    bounds every scaled value of a block by |a| < 2^(emax + 1) (also when the byte clamps at 0); the test also
    checks that each exponent's domain is non-empty."""
    bias, mbits, emax = P._FMT_GEOM[fmt]
    lo = 128 - bias
    xb = np.arange(65536, dtype=np.uint32)
    xb = xb[((xb >> 7) & 0xFF) != 0xFF]  # finite (NaN / inf blocks are masked)
    xf = (xb << 16).view(np.float32)
    for se in range(-127, 126):
        mult = np.uint32((127 - se) << 23).view(np.float32)
        with np.errstate(under="ignore", over="ignore"):
            a = (xf * mult).astype(np.float32)
        abits = a.view(np.uint32) & np.uint32(0x7FFFFFFF)
        jc = np.maximum((abits >> 23).astype(np.int64), lo)
        cbits = ((jc + 23 - mbits) << 23).astype(np.uint32)
        s = (abits.view(np.float32) + cbits.view(np.float32)).astype(np.float32)
        code = s.view(np.uint32).astype(np.int64) - cbits.astype(np.int64) + ((jc - lo) << mbits)
        code = np.minimum(code, 31) | (((xb << 16) >> 26) & 32)  # the kernel takes the sign from the f32 bits
        want = L.fmt_encode(a, fmt).astype(np.int64)
        dom = np.abs(a) < 2.0 ** (emax + 1)
        assert dom.sum() > 500
        bad = (code != want) & dom
        assert not bad.any(), (se, int(bad.sum()), hex(int(xb[bad][0])))


def test_pack_reference_is_a_permutation_and_matches_layout():
    op = L.generate(512, 1024, L.DIST_UNIFORM_CODES, seed=5)
    img, sc = P.pack_ref(op)
    assert np.array_equal(img, L.pack_operand_c0c1(op, 256))
    assert np.array_equal(np.sort(img), np.sort(op.codes.reshape(-1)))
    assert np.array_equal(np.sort(sc), np.sort(op.scales.reshape(-1)))
    # inverse of the c0c1 image (slot o = 64 rb + 16 kg + r16): the canonical rows come back
    R, nt = 256, 1024 // 128
    x = img.reshape(512 // R, nt, R * 96)
    c0 = x[..., : 64 * R].reshape(512 // R, nt, R // 16, 4, 16, 16)
    c1 = x[..., 64 * R:].reshape(512 // R, nt, R // 16, 4, 16, 8)
    chunk = np.concatenate([c0, c1], axis=-1)  # blk s rb kg r16 24
    rows = chunk.transpose(0, 2, 4, 1, 3, 5).reshape(512, 1024 * 3 // 4)
    assert np.array_equal(rows, op.codes)


def test_edge_activations_cover_the_corners():
    for fmt in FMTS:
        for src in SRCS:
            x = P.edge_activations(64, 512, fmt=fmt, src=src)
            q = P.quantize_ref(x, fmt, src)
            s = q.scales
            assert (s == 255).sum() >= 3 and (s == 0).sum() >= 4  # NaN / inf blocks; zero / subnormal blocks
            codes = L.unpack_codes(q.codes, 512)
            assert (codes == 31).any() and (codes == 63).any() and (codes == 32).any()
            assert x.dtype == (np.uint16 if src == "bf16" else np.float32)


# ----------------------------------------------------------------- builders (no GPU)


def _lower(kfn):
    from rocke.core.lower_llvm import _lower_kernel_to_llvm_python

    return _lower_kernel_to_llvm_python(kfn, arch="gfx950", llvm_flavor="llvm22")


@pytest.mark.parametrize("bad", [
    dict(rows=500), dict(k=4000), dict(wg_rows=96), dict(wg_tiles=3), dict(order="zz"), dict(mutant="x"),
    dict(wg_rows=64, wg_tiles=1),  # 64 x 1 tile: 384 chunks, not a whole number of 128-row scale groups
])
def test_pack_operand_validation(bad):
    kw = dict(rows=512, k=4096)
    kw.update(bad)
    with pytest.raises(ValueError):
        P.build_pack_operand(kw.pop("rows"), kw.pop("k"), **kw)


@pytest.mark.parametrize("bad", [
    dict(fmt="e5m2"), dict(src="f16"), dict(cvt="x"), dict(out="tile"), dict(scale_out="kstep"), dict(stage="x"),
    dict(lane_order="x"), dict(wg_rows=96), dict(wg_rows=512), dict(rows=320), dict(k=4000), dict(mutant="x"),
])
def test_quant_validation(bad):
    kw = dict(rows=512, k=4096)
    kw.update(bad)
    with pytest.raises(ValueError):
        P.build_quant_mx(kw.pop("rows"), kw.pop("k"), **kw)


def test_kernel_names_are_unique_per_knob_set():
    names = set()
    n = 0
    for fmt in FMTS:
        for src in SRCS:
            for cvt in ("hw", "sw"):
                for st in ("direct", "lds"):
                    for nt in (False, True):
                        names.add(P.build_quant_mx(256, 512, fmt=fmt, src=src, cvt=cvt, stage=st, nt_store=nt).name)
                        n += 1
    for wr, t in ((64, 2), (64, 4), (128, 1), (128, 4), (256, 1)):
        for order in P.PACK_ORDERS:
            for c1w in (False, True):
                names.add(P.build_pack_operand(512, 4096, wg_rows=wr, wg_tiles=t, order=order, c1_wide=c1w).name)
                n += 1
    assert len(names) == n


@pytest.mark.parametrize("src", SRCS)
@pytest.mark.parametrize("fmt", FMTS)
def test_hw_conversion_is_emitted_only_for_cvt_hw(fmt, src):
    ins = {("bf16", "e2m3"): "v_cvt_scalef32_pk32_fp6_bf16", ("bf16", "e3m2"): "v_cvt_scalef32_pk32_bf6_bf16",
           ("f32", "e2m3"): "v_cvt_scalef32_2xpk16_fp6_f32", ("f32", "e3m2"): "v_cvt_scalef32_2xpk16_bf6_f32"}
    hw = _lower(P.build_quant_mx(256, 512, fmt=fmt, src=src, cvt="hw"))
    sw = _lower(P.build_quant_mx(256, 512, fmt=fmt, src=src, cvt="sw"))
    assert ins[(src, fmt)] in hw and "v_cvt_scalef32" not in sw
    for other in set(ins.values()) - {ins[(src, fmt)]}:
        assert other not in hw
    assert "=&v" in hw  # early-clobber output (the conversion must not overwrite its own sources)


def test_streaming_hints_lower_to_nontemporal():
    a = _lower(P.build_pack_operand(512, 4096, nt_load=True, nt_store=True))
    b = _lower(P.build_pack_operand(512, 4096, nt_store=False))
    assert "!nontemporal" in a and "!nontemporal" not in b
    c = _lower(P.build_copy(1 << 20, nt_store=True))
    assert c.count("!nontemporal") >= 4 and "load <4 x i32>" in c


def test_grids():
    assert P.pack_operand_grid(4096, 65536, 64, 4) == 64 * 128 and P.pack_operand_grid(4096, 65536) == 32 * 128
    assert P.quant_mx_grid(4096, 65536) == 64 * 512
    assert P.pack_operand_grid(4096, 65536, 128, 1) == 32 * 512
    assert P.quant_mx_grid(4096, 65536, 128) == 32 * 512
    assert P.copy_grid(1 << 24, 4) == (1 << 24) // 16384


# ----------------------------------------------------------------- scale pack


def _perm_np(src0, src1, sel):
    """numpy model of v_perm_b32 (the IR doc): result byte i = byte sel_i of [src1 bytes 0..3, src0 bytes 4..7]."""
    cat = np.concatenate([np.frombuffer(np.uint32(src1).tobytes(), np.uint8),
                          np.frombuffer(np.uint32(src0).tobytes(), np.uint8)])
    sb = np.frombuffer(np.uint32(sel).tobytes(), np.uint8)
    return int(np.frombuffer(cat[sb].tobytes(), np.uint32)[0])


def test_transpose4x4_selectors():
    """The 8 v_perm_b32 selectors of pack._transpose4x4 transpose a 4x4 byte matrix (perm modelled per the IR doc)."""
    rng = np.random.default_rng(0)
    for _ in range(50):
        d = [int(x) for x in rng.integers(0, 2 ** 32, size=4, dtype=np.uint64)]
        t0, t1 = _perm_np(d[1], d[0], 0x05010400), _perm_np(d[1], d[0], 0x07030602)
        u0, u1 = _perm_np(d[3], d[2], 0x05010400), _perm_np(d[3], d[2], 0x07030602)
        out = [_perm_np(u0, t0, 0x05040100), _perm_np(u0, t0, 0x07060302), _perm_np(u1, t1, 0x05040100),
               _perm_np(u1, t1, 0x07060302)]
        m = np.array([[(x >> (8 * j)) & 0xFF for j in range(4)] for x in d])
        o = np.array([[(x >> (8 * j)) & 0xFF for j in range(4)] for x in out])
        assert np.array_equal(o, m.T)


@pytest.mark.parametrize("bad", [dict(rows_a=100), dict(rows_b=64), dict(k=100), dict(atom="x"), dict(order="row"),
                                 dict(mutant="x"), dict(rows_a=0)])
def test_pack_scales_validation(bad):
    kw = dict(rows_a=512, k=4096)
    kw.update(bad)
    with pytest.raises(ValueError):
        P.build_pack_scales(kw.pop("rows_a"), kw.pop("k"), **kw)


def test_pack_scales_grid_and_tiles_per_wg():
    assert P._scale_tiles_per_wg(65536) == 32 and P._scale_tiles_per_wg(128) == 1 and P._scale_tiles_per_wg(384) == 1
    assert P._scale_tiles_per_wg(1024) == 8
    assert P.pack_scales_grid(4096, 65536, 4096) == 2 * 32 * 16
    assert P.pack_scales_grid(128, 128) == 1
    lo = _lower(P.build_pack_scales(4096, 65536, 4096))
    assert lo.count("llvm.amdgcn.perm") >= 16 and "store <4 x i32>" in lo
