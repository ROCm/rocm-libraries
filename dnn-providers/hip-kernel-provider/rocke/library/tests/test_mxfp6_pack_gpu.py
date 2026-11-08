# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""gfx950 GPU tests of the MXFP6 pack / quantize kernels (``kernels/gfx950/mxfp6_gemm/pack.py``); skipped elsewhere.

Every kernel output must be byte-identical to the host reference (``layout.pack_operand_c0c1`` /
``layout.pack_scales_tile`` for the packs; ``pack.quantize_ref`` -- the OCP rule, checked against an independent
oracle in ``test_mxfp6_pack.py`` -- followed by the same packers for the quantizers). Each launch runs twice over
different garbage (0xAB / 0x54): an unwritten byte or a race shows as a difference. Every must-fail mutant must
differ from the reference.
"""

from __future__ import annotations

import ctypes
import struct

import numpy as np
import pytest
from kernels.gfx950.mxfp6_gemm import layout as L
from kernels.gfx950.mxfp6_gemm import pack as P


def _gpu_arch():
    try:
        from rocke.runtime.hip_module import get_device_arch

        return get_device_arch()
    except Exception:  # noqa: BLE001 - no HIP runtime / device
        return None


pytestmark = pytest.mark.skipif(_gpu_arch() != "gfx950", reason="needs a gfx950 GPU")

FMTS = ("e2m3", "e3m2")
SRCS = ("bf16", "f32")
_MODS: dict = {}


@pytest.fixture(scope="module")
def runtime():
    from rocke.runtime.hip_module import Runtime

    rt = Runtime()
    yield rt
    for m in _MODS.values():
        m.unload()
    _MODS.clear()


def _fn(rt, kfn):
    from rocke.helpers.compile import compile_kernel

    if kfn.name not in _MODS:
        _MODS[kfn.name] = rt.load_module(compile_kernel(kfn, arch="gfx950", backend="python").hsaco)
    return _MODS[kfn.name].get_function(kfn.name)


def _u8(a):
    a = np.ascontiguousarray(a)
    return a, (ctypes.c_ubyte * a.nbytes).from_buffer(a)


def _launch(rt, kfn, grid, ins, out_sizes, fills=(0xAB, 0x54)):
    """Launch ``kfn`` (params: inputs then outputs) once per garbage fill; -> [[output bytes] per fill]."""
    fn = _fn(rt, kfn)
    pin = []
    for a in ins:
        a, buf = _u8(a)
        p = rt.alloc(a.nbytes)
        rt.memcpy_h2d(p, buf, a.nbytes)
        pin.append(p)
    pout = [rt.alloc(n) for n in out_sizes]
    res = []
    for fill in fills:
        for p, n in zip(pout, out_sizes):
            rt.memset(p, fill, n)
        rt.launch(fn, (grid, 1, 1), (256, 1, 1), struct.pack("<" + "Q" * (len(pin) + len(pout)), *pin, *pout))
        rt.wait_stream(0)
        outs = []
        for p, n in zip(pout, out_sizes):
            got = np.empty(n, np.uint8)
            rt.memcpy_d2h((ctypes.c_ubyte * n).from_buffer(got), p, n)
            outs.append(got)
        res.append(outs)
    for p in pin + pout:
        rt.free(p)
    return res


def _diff(got, want):
    want = np.ascontiguousarray(want).view(np.uint8).reshape(-1)
    bad = np.flatnonzero(got != want)
    return f"{bad.size} of {want.size} bytes differ" + (f" (first at {bad[0]}: {got[bad[0]]:#x} vs "
                                                         f"{want[bad[0]]:#x})" if bad.size else "")


def _assert_same(res, wants, what):
    for fi, outs in enumerate(res):
        for oi, (got, want) in enumerate(zip(outs, wants)):
            w = np.ascontiguousarray(want).view(np.uint8).reshape(-1)
            assert np.array_equal(got, w), f"{what} fill {fi} output {oi}: {_diff(got, w)}"


def _operands(rows, k, fmt="e2m3"):
    gen = L.generate(rows, k, L.DIST_GAUSS, seed=11, tag=1, fmt=fmt)
    rng = np.random.default_rng(12)  # arbitrary bytes: every code bit pattern, NaN (0xFF) / 0 / 254 scales
    rnd = L.Operand(rows, k, rng.integers(0, 256, size=gen.codes.shape, dtype=np.uint8),
                    rng.integers(0, 256, size=gen.scales.shape, dtype=np.uint8), fmt)
    return gen, rnd


# ----------------------------------------------------------------- copy roofline kernel


@pytest.mark.parametrize("unroll,ntl,nts,order", [(1, False, False, "k"), (4, False, False, "k"),
                                                  (4, True, True, "xcd"), (8, False, True, "xcd")])
def test_copy_kernel(runtime, unroll, ntl, nts, order):
    n = 1 << 21
    src = np.random.default_rng(1).integers(0, 256, size=n, dtype=np.uint8)
    k = P.build_copy(n, unroll=unroll, nt_load=ntl, nt_store=nts, order=order)
    _assert_same(_launch(runtime, k, P.copy_grid(n, unroll), [src], [n]), [src], k.name)


# ----------------------------------------------------------------- packs


@pytest.mark.parametrize("fmt", FMTS)
def test_legacy_pack_kernels_match_the_host_packers(runtime, fmt):
    """build_pack_c0c1 / build_pack_scales_tile (the bench's "_pack" arms): byte-identical to layout.py."""
    rows, k = 512, 4096
    g_ops, g_sc = P.pack_grids(rows, k)
    for op in _operands(rows, k, fmt):
        img, sc = P.pack_ref(op)
        k1 = P.build_pack_c0c1(rows, k)
        _assert_same(_launch(runtime, k1, g_ops, [op.codes], [img.nbytes]), [img], k1.name)
        k2 = P.build_pack_scales_tile(rows, k)
        _assert_same(_launch(runtime, k2, g_sc, [op.scales], [sc.nbytes]), [sc], k2.name)


_PACK_CFGS = [  # (rows, k, knobs)
    *[(512, 4096, dict(wg_rows=wr, wg_tiles=t)) for wr, t in ((64, 2), (64, 4), (64, 8), (128, 1), (128, 2),
                                                               (128, 4), (256, 1), (256, 2))],
    *[(512, 4096, dict(wg_rows=64, wg_tiles=4, order=o, c1_wide=c)) for o in P.PACK_ORDERS for c in (False, True)],
    *[(512, 4096, dict(wg_rows=128, wg_tiles=1, order=o, c1_wide=c)) for o in P.PACK_ORDERS for c in (False, True)],
    (512, 4096, dict(nt_load=True)), (512, 4096, dict(nt_store=True)), (512, 4096, dict(nt_load=True, nt_store=True)),
    # boundary shapes: one row block / a single tile group / 3 row blocks, K = one work-group of tiles
    (256, 512, dict(wg_rows=64, wg_tiles=4)), (256, 128, dict(wg_rows=128, wg_tiles=1)),
    (768, 1024, dict(wg_rows=64, wg_tiles=8)), (768, 1024, dict(wg_rows=128, wg_tiles=2, order="xcd")),
    (512, 4096, dict(scales=False)),
    (512, 4096, dict()),  # the defaults (128 x 4 tiles, XCD walk, nt stores)
    (256, 512, dict(order="xcd")),  # 2 work-groups: the XCD walk falls back to identity
]


@pytest.mark.parametrize("rows,k,kw", _PACK_CFGS, ids=[f"{r}x{k}-" + "-".join(f"{a}{b}" for a, b in kw.items())
                                                        for r, k, kw in _PACK_CFGS])
def test_pack_operand_matches_host_packers(runtime, rows, k, kw):
    kfn = P.build_pack_operand(rows, k, **kw)
    grid = P.pack_operand_grid(rows, k, kw.get("wg_rows", 128), kw.get("wg_tiles", 4))
    for fmt in FMTS:
        for op in _operands(rows, k, fmt):
            img, sc = P.pack_ref(op)
            res = _launch(runtime, kfn, grid, [op.codes, op.scales], [img.nbytes, sc.nbytes])
            if kw.get("scales", True):
                _assert_same(res, [img, sc], kfn.name)
            else:  # codes only; the scale output must be untouched (still the fill)
                _assert_same([[r[0]] for r in res], [img], kfn.name)
                assert (res[0][1] == 0xAB).all() and (res[1][1] == 0x54).all()


@pytest.mark.parametrize("mutant", [m for m in P.PACK_MUTANTS if m != "none"])
def test_pack_mutants_fail(runtime, mutant):
    rows, k = 512, 4096
    kfn = P.build_pack_operand(rows, k, mutant=mutant)
    op = _operands(rows, k)[1]
    img, sc = P.pack_ref(op)
    got = _launch(runtime, kfn, P.pack_operand_grid(rows, k), [op.codes, op.scales], [img.nbytes, sc.nbytes],
                  fills=(0,))[0]
    assert not (np.array_equal(got[0], img) and np.array_equal(got[1], sc)), f"mutant {mutant} passed"


# ----------------------------------------------------------------- quantizers


def _quant_want(x, fmt, src, out, scale_out, block_rows=256):
    q = P.quantize_ref(x, fmt, src)
    wc = L.pack_operand_c0c1(q, block_rows) if out == "c0c1" else np.ascontiguousarray(q.codes).reshape(-1)
    ws = (L.pack_scales_tile(q.scales, "16x16x128").view(np.uint8).reshape(-1) if scale_out == "tile"
          else np.ascontiguousarray(q.scales).reshape(-1))
    return q, wc, ws


def _run_quant(rt, rows, k, fmt, src, x=None, fills=(0xAB, 0x54), **kw):
    x = P.edge_activations(rows, k, fmt=fmt, src=src) if x is None else x
    kfn = P.build_quant_mx(rows, k, fmt=fmt, src=src, **kw)
    _, wc, ws = _quant_want(x, fmt, src, kw.get("out", "c0c1"), kw.get("scale_out", "tile"))
    res = _launch(rt, kfn, P.quant_mx_grid(rows, k, kw.get("wg_rows", 64)), [x], [wc.nbytes, ws.nbytes],
                  fills=fills)
    return kfn, res, wc, ws


@pytest.mark.parametrize("cvt", ["hw", "sw"])
@pytest.mark.parametrize("src", SRCS)
@pytest.mark.parametrize("fmt", FMTS)
def test_quant_matches_reference(runtime, fmt, src, cvt):
    """bf16 / f32 -> E2M3 / E3M2 c0c1 + tile scales, hardware conversion (with the tiny-block fallback) or software
    encoder, on every OCP corner (edge_activations) at two seeds."""
    for seed in (3, 4):
        x = P.edge_activations(512, 1024, fmt=fmt, src=src, seed=seed)
        kfn, res, wc, ws = _run_quant(runtime, 512, 1024, fmt, src, x=x, cvt=cvt)
        _assert_same(res, [wc, ws], f"{kfn.name} seed {seed}")


_LAYOUTS = [
    dict(stage="direct", lane_order="kg", out="c0c1", scale_out="tile", wg_rows=128, nt_load=False),
    dict(stage="direct", lane_order="row", out="c0c1", scale_out="tile", wg_rows=64, nt_load=False),
    dict(stage="direct", lane_order="kg", out="canonical", scale_out="raw", wg_rows=256),
    dict(stage="lds", lane_order="kg", out="c0c1", scale_out="tile", wg_rows=128, nt_load=False),
    dict(stage="lds", lane_order="row", out="c0c1", scale_out="raw", wg_rows=64),
    dict(stage="lds", lane_order="kg", out="canonical", scale_out="tile", wg_rows=256),
    dict(stage="direct", lane_order="row", out="canonical", scale_out="raw", wg_rows=128),
    dict(stage="direct", lane_order="kg", out="c0c1", scale_out="tile", wg_rows=256, nt_load=True, nt_store=True),
    dict(stage="lds", lane_order="kg", out="c0c1", scale_out="tile", wg_rows=64, nt_store=True),
]


@pytest.mark.parametrize("li", range(len(_LAYOUTS)), ids=["-".join(f"{v}" for v in d.values()) for d in _LAYOUTS])
@pytest.mark.parametrize("fmt,src", [("e2m3", "bf16"), ("e3m2", "f32"), ("e3m2", "bf16"), ("e2m3", "f32")])
def test_quant_layouts_match_reference(runtime, fmt, src, li):
    kfn, res, wc, ws = _run_quant(runtime, 512, 1024, fmt, src, **_LAYOUTS[li])
    _assert_same(res, [wc, ws], kfn.name)


@pytest.mark.parametrize("rows,k", [(256, 128), (256, 256), (768, 384)])
def test_quant_boundary_shapes(runtime, rows, k):
    """Smallest shapes (one row block, one / two K tiles, three row blocks): Gaussian data over 2^-30..2^30."""
    rng = np.random.default_rng(rows + k)
    x = (rng.standard_normal((rows, k)) * np.ldexp(1.0, rng.integers(-30, 31, size=(rows, 1)))).astype(np.float32)
    for src in SRCS:
        xi = x if src == "f32" else L.f32_to_bf16_bits(x)
        for fmt in FMTS:
            kfn, res, wc, ws = _run_quant(runtime, rows, k, fmt, src, x=xi)
            _assert_same(res, [wc, ws], kfn.name)


@pytest.mark.parametrize("fallback", [-1, 8, 254])
@pytest.mark.parametrize("fmt,src", [("e2m3", "bf16"), ("e3m2", "bf16"), ("e2m3", "f32"), ("e3m2", "f32")])
def test_hw_conversion_every_block(runtime, fmt, src, fallback, capsys):
    """The hardware conversion alone (fallback -1, the default) must equal the OCP reference on every block -- tiny /
    subnormal / zero blocks (scale byte 0 -> the f32 scale is +0.0 or subnormal), ties, saturation, -0 included;
    fallback 8 / 254 (software encoder for small / all finite scale bytes, divergent branch) must too. Per-block
    mismatch counts are printed for the record (canonical / raw layout: blocks map 1:1)."""
    rows, k = 512, 1024
    x = P.edge_activations(rows, k, fmt=fmt, src=src)
    kfn, res, wc, ws = _run_quant(runtime, rows, k, fmt, src, x=x, fallback=fallback, out="canonical",
                                  scale_out="raw", fills=(0xAB,))
    q = P.quantize_ref(x, fmt, src)
    got = L.unpack_codes(res[0][0].reshape(rows, k * 3 // 4), k).reshape(rows, k // 32, 32)
    want = L.unpack_codes(q.codes, k).reshape(rows, k // 32, 32)
    blk_bad = (got != want).any(-1)
    small = q.scales <= 8
    with capsys.disabled():
        print(f"\n[hw-semantics] {fmt} {src} fallback {fallback}: differing blocks {int(blk_bad.sum())} "
              f"(scale byte <= 8: {int((blk_bad & small).sum())} of {int(small.sum())})")
        for r, c in list(zip(*np.nonzero(blk_bad)))[:3]:
            print(f"  row {r} blk {c} byte {q.scales[r, c]} got {got[r, c][:8].tolist()} want {want[r, c][:8].tolist()}")
    assert np.array_equal(res[0][1], q.scales.reshape(-1)), _diff(res[0][1], q.scales)
    assert not blk_bad.any() and int(small.sum()) > 100


@pytest.mark.parametrize("mutant,cvt", [("emax", "hw"), ("nanmask", "hw"), ("word", "hw"), ("scalepos", "hw"),
                                        ("emax", "sw"), ("nanmask", "sw")])
def test_quant_mutants_fail(runtime, mutant, cvt):
    kfn, res, wc, ws = _run_quant(runtime, 512, 1024, "e2m3", "bf16", fills=(0,), mutant=mutant, cvt=cvt)
    assert not (np.array_equal(res[0][0], wc) and np.array_equal(res[0][1], ws)), f"mutant {mutant} passed"


def test_quant_e3m2_kernel_on_e2m3_data_differs(runtime):
    """Format guard: the E3M2 kernel's output on the same activations differs from the E2M3 reference."""
    x = P.edge_activations(512, 1024, fmt="e2m3", src="bf16")
    kfn = P.build_quant_mx(512, 1024, fmt="e3m2")
    _, wc, ws = _quant_want(x, "e2m3", "bf16", "c0c1", "tile")
    got = _launch(runtime, kfn, P.quant_mx_grid(512, 1024), [x], [wc.nbytes, ws.nbytes], fills=(0,))[0]
    assert not np.array_equal(got[0], wc) and not np.array_equal(got[1], ws)


# ----------------------------------------------------------------- legacy quantizer (bench "_quant" arms)


@pytest.mark.parametrize("stage,wg_rows", [("lds", 256), ("lds", 128), ("lds", 64), ("slot", 256)])
@pytest.mark.parametrize("out,scale_out", [("c0c1", "tile"), ("c0c1", "raw"), ("canonical", "tile"),
                                           ("canonical", "raw")])
def test_legacy_quantize_kernel_matches_host_quantizer(runtime, out, scale_out, stage, wg_rows):
    """build_quant_e2m3: bf16 -> E2M3 codes + scales, byte-identical to layout.quantize_bf16 + the packers."""
    rows, k = 512, 4096
    xb = P.edge_activations(rows, k, fmt="e2m3", src="bf16")
    _, wc, ws = _quant_want(xb, "e2m3", "bf16", out, scale_out)
    kfn = P.build_quant_e2m3(rows, k, out, scale_out, stage=stage, wg_rows=wg_rows)
    res = _launch(runtime, kfn, P.quant_grid(rows, k, stage=stage, wg_rows=wg_rows), [xb], [wc.nbytes, ws.nbytes])
    _assert_same(res, [wc, ws], kfn.name)


# ----------------------------------------------------------------- scale pack (raw -> tile), one launch for A and B

_SC_CFGS = [  # (rows_a, rows_b, k, knobs)
    (512, 512, 4096, {}), (512, 0, 4096, {}), (4096, 4096, 65536, {}), (4096, 4096, 65536, dict(order="xcd")),
    (512, 384, 4096, dict(nt_load=True, nt_store=True)), (256, 768, 1024, {}), (128, 128, 128, {}),
    (256, 128, 384, {}), (384, 0, 512, dict(order="xcd")), (1024, 2048, 8192, {}),
]


@pytest.mark.parametrize("atom", ["16x16x128", "32x32x64"])
@pytest.mark.parametrize("ra,rb,k,kw", _SC_CFGS, ids=[f"{a}_{b}x{k}-" + "-".join(f"{x}{y}" for x, y in kw.items())
                                                      for a, b, k, kw in _SC_CFGS])
def test_pack_scales_matches_host(runtime, ra, rb, k, kw, atom):
    """raw E8M0 [rows][K/32] -> layout.pack_scales_tile(atom), A and B in one launch; random bytes incl. 0xFF / 0."""
    rng = np.random.default_rng(ra + rb + k)
    sa = rng.integers(0, 256, size=(ra, k // 32), dtype=np.uint8)
    sa[0, :4] = 255
    sa[1, :4] = 0
    sb = rng.integers(0, 256, size=(max(rb, 128), k // 32), dtype=np.uint8)
    kfn = P.build_pack_scales(ra, k, rb, atom=atom, **kw)
    wa = L.pack_scales_tile(sa, atom).view(np.uint8).reshape(-1)
    wb = L.pack_scales_tile(sb, atom).view(np.uint8).reshape(-1)
    res = _launch(runtime, kfn, P.pack_scales_grid(ra, k, rb), [sa, sb], [wa.nbytes, wb.nbytes])
    if rb:
        _assert_same(res, [wa, wb], kfn.name)
    else:  # B untouched
        _assert_same([[r[0]] for r in res], [wa], kfn.name)
        assert (res[0][1] == 0xAB).all() and (res[1][1] == 0x54).all()


@pytest.mark.parametrize("atom", ["16x16x128", "32x32x64"])
@pytest.mark.parametrize("mutant", [m for m in P.SCALE_MUTANTS if m != "none"])
def test_pack_scales_mutants_fail(runtime, mutant, atom):
    rng = np.random.default_rng(5)
    s = rng.integers(0, 256, size=(512, 128), dtype=np.uint8)
    w = L.pack_scales_tile(s, atom).view(np.uint8).reshape(-1)
    kfn = P.build_pack_scales(512, 4096, 512, atom=atom, mutant=mutant)
    got = _launch(runtime, kfn, P.pack_scales_grid(512, 4096, 512), [s, s], [w.nbytes, w.nbytes], fills=(0,))[0]
    assert not np.array_equal(got[0], w) and not np.array_equal(got[1], w), f"mutant {mutant} passed"
