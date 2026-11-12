# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""gfx950 numerics of the MXFP6 GEMM tile-config library (needs a gfx950 GPU and the rocKE direct compiler with
the ``trtag`` lever, ``ROCKE_RUST_COMPILE``; skipped elsewhere).

Per structural family (tiny / tall / wide / square / large / DepthU 512), both activation layouts (K-contiguous
``ak``, M-contiguous ``am``) and both formats: full-output checks of a runtime-shape build at a batched problem
whose shape differs from the build's reference shape -- the exact-integer set bit for bit, the Gaussian /
uniform-code sets within the scaled-MFMA tolerance, every launch identical; the same code object at a second
problem shape; the edge suite (subnormals kept, OCP NaN / overflow convention); must-fail mutants built on each
family (incl. a batch-index swap and the M-contiguous slot swizzle). ``test_tr_b6_semantics`` pins the
``ds_read_b96_tr_b6`` model the CPU suite's data-path tests rely on.
"""

from __future__ import annotations

import ctypes
import os
import struct
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from kernels.gfx950.mxfp6_gemm import edge_cases as E
from kernels.gfx950.mxfp6_gemm import gemm as G
from kernels.gfx950.mxfp6_gemm import layout as L
from kernels.gfx950.mxfp6_gemm import library as LB


def _gpu_arch():
    try:
        from rocke.runtime.hip_module import get_device_arch

        return get_device_arch()
    except Exception:  # noqa: BLE001 - no HIP runtime / device
        return None


pytestmark = [pytest.mark.skipif(_gpu_arch() != "gfx950", reason="needs a gfx950 GPU"),
              pytest.mark.skipif(not os.environ.get("ROCKE_RUST_COMPILE"),
                                 reason="needs the rocKE direct compiler (ROCKE_RUST_COMPILE)")]

#: (MT0, MT1, DepthU) per structural family
FAMILY_CASES = {
    "tiny-64x64": (64, 64, 256), "tiny-128x32": (128, 32, 256), "tiny-32x128": (32, 128, 256),
    "tall-512x32": (512, 32, 256), "tall-384x96": (384, 96, 256),
    "wide-32x512": (32, 512, 256), "wide-64x384": (64, 384, 256),
    "square-128x128": (128, 128, 256), "square-256x256": (256, 256, 256), "square-96x128": (96, 128, 256),
    "square-224x256": (224, 256, 256), "square-160x384": (160, 384, 256),
    "large-256x288": (256, 288, 256), "large-384x192": (384, 192, 256),
    "du512-32x32": (32, 32, 512), "du512-128x160": (128, 160, 512), "du512-192x64": (192, 64, 512),
}
#: knob variants on a family default (the structures the searches start from)
VARIANTS = {
    "du256-barrier": ((64, 64, 256), {"depth_u": 256, "stages": 4, "occupancy": 2}),
    "du512-barrier": ((32, 32, 512), {"depth_u": 512, "stages": 8, "unroll": 1, "occupancy": 2}),
    "switch-tail": ((128, 128, 256), {"rt_tail": "switch"}),
    "vgpr-scales": ((128, 64, 256), {"scale_src": "vgpr"}),
    "linear-order": ((128, 128, 256), {"cta_order": "linear"}),
    "jit-4w": ((128, 128, 256), {"frag_mode": "jit", "acq_slot": -1}),
    "deep-ring-s8": ((64, 64, 256), {"stages": 8, "occupancy": 1}),
    "line2m-256": ((256, 256, 256), LB.LINE2M),
    "line2m-ring-256": ((256, 256, 256), {"ring": "line2m"}),
    "line2m-ring-128": ((128, 128, 256), {"ring": "line2m"}),
    "line2m-ring-32x128": ((32, 128, 256), {"ring": "line2m"}),
    "line2m-ring-448x128": ((448, 128, 256), {"ring": "line2m"}),
    "line2m-ring-96x384": ((96, 384, 256), {"ring": "line2m"}),
    "line2m-ring-160x128x512": ((160, 128, 512), {"ring": "line2m"}),
    "b128b64-128x128": ((128, 128, 256), {"canon_read": "b128b64"}),
    "b128b64-64x64": ((64, 64, 256), {"canon_read": "b128b64"}),
    "b128b64-96x128-rot48off": ((96, 128, 256), {"canon_read": "b128b64", "swizzle": "none"}),
    # the stage ring where the family default is line2m (ring="auto")
    "stage-ring-256": ((256, 256, 256), {"ring": "stage"}),
    "stage-ring-128": ((128, 128, 256), {"ring": "stage"}),
    "stage-ring-32x128": ((32, 128, 256), {"ring": "stage"}),
    # persistent schedule (default grid: the 3x2 / 4x5 tile test problems are not multiples of 8 work units)
    "persistent-128": ((128, 128, 256), {"schedule": "persistent"}),
    "persistent-256": ((256, 256, 256), {"schedule": "persistent"}),
    "persistent-64x64": ((64, 64, 256), {"schedule": "persistent"}),
    "persistent-du512-32x32": ((32, 32, 512), {"schedule": "persistent"}),
    "persistent-256x288": ((256, 288, 256), {"schedule": "persistent"}),
    "persist-ov-256": ((256, 256, 256), {"schedule": "persistent", "persist_overlap": True}),
    "persist-ov-128": ((128, 128, 256), {"schedule": "persistent", "persist_overlap": True}),
    "persist-ov-64x64": ((64, 64, 256), {"schedule": "persistent", "persist_overlap": True}),
    "persist-ov-du512-32x32": ((32, 32, 512), {"schedule": "persistent", "persist_overlap": True}),
    # 4-wave layouts with accumulators past 256 per lane (acc_vgpr_spill), jit fragments
    "w4spill-192x384": ((192, 384, 256), {"waves": (2, 2), "frag_mode": "jit", "acq_slot": -1, "ring": "stage"}),
    "w4spill-384x192-bdir": ((384, 192, 256), {"waves": (1, 4), "frag_mode": "jit", "acq_slot": -1, "ring": "stage",
                                               "b_direct": True, "stages": 4}),
    "mfast-32x128": ((32, 128, 256), {"cta_order": "mfast"}),
    # line DTL block order: 384 line rows on 4 waves (12 chunks per wave, 3 row-class blocks)
    "line2m-block-64x384": ((64, 384, 256), {"line_dtl_order": "block"}),
    # C staged through LDS, dwordx4 stores: line2m 2x2 waves (fn 8), and persistent + nt on a stage ring (fn 2)
    "epi-lds-256x256": ((256, 256, 256), {"epi_lds": True}),
    # direct B beside the line2m A ring (B straight from the c0c1 image to VGPRs)
    "line2m-bdir-256x256": ((256, 256, 256), {"b_direct": True, "ring": "line2m"}),
    "line2m-bdir3-256x256": ((256, 256, 256), {"b_direct": True, "ring": "line2m", "b_depth": 3}),
    "line2m-merge-256x256": ((256, 256, 256), {"line_merge": True}),
    # canonical A, its c0c1 stage image built by the DMAs (a_fill): dp and persistent + edge (library specs)
    "afill-256x256": ((256, 256, 256), {"ring": "stage", "a_fill": "c0c1"}),
    "afill-128x128-pers": ((128, 128, 256), {"ring": "stage", "a_fill": "c0c1", "schedule": "persistent"}),
    "line2m-merge-256x256-pers-epi": ((256, 256, 256), {"line_merge": True, "schedule": "persistent", "read_every": 3,
                                                        "epi_lds": True}),
    "line2m-bdir1-256x256-pers": ((256, 256, 256), {"b_direct": True, "ring": "line2m", "b_depth": 1,
                                                    "schedule": "persistent", "read_every": 3}),
    "epi-lds-64x64-pers-nt": ((64, 64, 256), {"epi_lds": True, "schedule": "persistent", "store_nt": True}),
    "mfast-pers-64x448": ((64, 448, 256), {"cta_order": "mfast", "schedule": "persistent", "edge_tiles": True}),
    # direct-to-VGPR A (a_direct): canonical rows straight to VGPRs, B through LDS
    "adir-160x384-s6": ((160, 384, 256), {"a_direct": True, "ring": "stage", "stages": 6}),
    "adir-128x384-s6-jit": ((128, 384, 256), {"a_direct": True, "ring": "stage", "stages": 6, "frag_mode": "jit",
                                              "acq_slot": -1}),
    "adir-edge-pers-96x384-s7": ((96, 384, 256), {"a_direct": True, "ring": "stage", "stages": 7, "edge_tiles": True,
                                                  "schedule": "persistent"}),
    # direct-to-VGPR B (b_direct) in library mode: prefetch fragments, A through LDS
    "bdir-448x128": ((448, 128, 256), {"b_direct": True}),
    "bdir-448x128-s6": ((448, 128, 256), {"b_direct": True, "stages": 6}),
    "bdir-384x128-s4-d2": ((384, 128, 256), {"b_direct": True, "stages": 4, "b_depth": 2}),
    "bdir-128x128": ((128, 128, 256), {"b_direct": True}),
    "bdir-128x128-jit": ((128, 128, 256), {"b_direct": True, "frag_mode": "jit", "acq_slot": -1}),
    "bdir-edge-pers-320x128": ((320, 128, 256), {"b_direct": True, "edge_tiles": True, "schedule": "persistent"}),
    # B fragment pairs (c0c1p: 3 b128 per 2 fragments) under canonical / M-contiguous A
    "c0c1p-256": ((256, 256, 256), {"operand_layout": "c0c1p"}),
    "c0c1p-128": ((128, 128, 256), {"operand_layout": "c0c1p"}),
    "c0c1p-64x128": ((64, 128, 256), {"operand_layout": "c0c1p"}),
    "c0c1p-192x256": ((192, 256, 256), {"operand_layout": "c0c1p"}),
    "c0c1p-256x288": ((256, 288, 256), {"operand_layout": "c0c1p"}),
    "c0c1p-edge-pers-128": ((128, 128, 256), {"operand_layout": "c0c1p", "edge_tiles": True,
                                              "schedule": "persistent"}),
    # raw canonical scales in VGPRs (permlane transpose)
    "raw-64x64": ((64, 64, 256), {"scale_src": "vgpr", "scale_raw": True}),
    "raw-128x128": ((128, 128, 256), {"scale_src": "vgpr", "scale_raw": True}),
    "raw-256x256": ((256, 256, 256), {"scale_src": "vgpr", "scale_raw": True}),
    "raw-32x128": ((32, 128, 256), {"scale_src": "vgpr", "scale_raw": True}),
    "raw-32x32x512": ((32, 32, 512), {"scale_src": "vgpr", "scale_raw": True}),
    "raw-256x288": ((256, 288, 256), {"scale_src": "vgpr", "scale_raw": True}),
    "raw-128x32": ((128, 32, 256), {"scale_src": "vgpr", "scale_raw": True}),
    # in-tree ports: host magic divisors for the CTA map, kernarg preload
    "magic-128": ((128, 128, 256), {"magic_div": True}),
    "magic-linear-128": ((128, 128, 256), {"magic_div": True, "cta_order": "linear"}),
    "magic-64x64-g2": ((64, 64, 256), {"magic_div": True, "group_m": 2}),
    "magic-256": ((256, 256, 256), {"magic_div": True}),
    "magic-persistent-128": ((128, 128, 256), {"magic_div": True, "schedule": "persistent"}),
    "preload-128": ((128, 128, 256), {"kernarg_preload": True}),
    "preload-256": ((256, 256, 256), {"kernarg_preload": True}),
    "preload-magic-pov-64x64": ((64, 64, 256), {"kernarg_preload": True, "magic_div": True, "schedule": "persistent",
                                                "persist_overlap": True}),
    # VGPR wtile scales loaded per period of wtile_period K tiles
    "wper4-64x64": ((64, 64, 256), {"scale_src": "vgpr", "wtile_period": 4}),
    "wper2-64x64": ((64, 64, 256), {"scale_src": "vgpr", "wtile_period": 2}),
    "wper4-32x32x512": ((32, 32, 512), {"scale_src": "vgpr", "wtile_period": 4}),
    "wper4-128x32": ((128, 32, 256), {"scale_src": "vgpr", "wtile_period": 4}),
    "wper4-32x128": ((32, 128, 256), {"scale_src": "vgpr", "wtile_period": 4}),
    "wper2-128x128": ((128, 128, 256), {"scale_src": "vgpr", "wtile_period": 2}),
    "wper2-64x256": ((64, 256, 256), {"scale_src": "vgpr", "wtile_period": 2}),
    "wper4-128x32x512": ((128, 32, 512), {"scale_src": "vgpr", "wtile_period": 4}),
}


@pytest.fixture(scope="module")
def runtime():
    from rocke.runtime.hip_module import Runtime

    return Runtime()


def _hsaco(spec) -> bytes:
    from benchmarks.gfx950.mxfp6_gemm.bench import _compile_rust

    old = os.environ.get("ROCKE_RUST_LEVERS")
    os.environ["ROCKE_RUST_LEVERS"] = LB.LEVERS
    try:
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "k.hsaco"
            _compile_rust(G.build_mxfp6_gemm(spec, arch="gfx950"), out)
            return out.read_bytes()
    finally:
        if old is None:
            os.environ.pop("ROCKE_RUST_LEVERS", None)
        else:
            os.environ["ROCKE_RUST_LEVERS"] = old


def _images(spec, entries):
    """Device images of a batch: per-entry packed A / B / scale images back to back (M-contiguous A: the zero tail
    pad once at the end)."""
    la = G.operand_layout_of(spec, "A")
    if la == "mcontig":
        a_img = np.concatenate([L.mcontig_codes(a).reshape(-1) for a, _ in entries] + [np.zeros(L.MCONTIG_PAD, np.uint8)])
    else:
        a_img = np.concatenate([L.pack_operand(a, la, spec.tile_m) for a, _ in entries])
    b_img = np.concatenate([L.pack_operand(b, G.operand_layout_of(spec, "B"), spec.tile_n) for _, b in entries])
    sa = np.concatenate([L.pack_scales_for(a.scales, spec.atom, G.scale_layout_of(spec, "A")).view(np.uint8)
                         for a, _ in entries])
    sb = np.concatenate([L.pack_scales_for(b.scales, spec.atom, G.scale_layout_of(spec, "B")).view(np.uint8)
                         for _, b in entries])
    return a_img, b_img, sa, sb


class _Kernel:
    def __init__(self, rt, spec):
        self.rt, self.spec = rt, spec
        self.mod = rt.load_module(_hsaco(spec))
        self.fn = self.mod.get_function(G.kernel_name(spec))

    def run(self, entries, launches=3, ngrid=0):
        rt, spec = self.rt, self.spec
        a0, b0 = entries[0]
        m, n, k, nb = a0.rows, b0.rows, a0.k, len(entries)
        ok, why = G.runtime_problem_ok(spec, m, n, k, nb)
        assert ok, why

        def put(x):
            x = np.ascontiguousarray(x).view(np.uint8).reshape(-1)
            p = rt.alloc(x.nbytes)
            rt.memcpy_h2d(p, (ctypes.c_ubyte * x.nbytes).from_buffer(x), x.nbytes)
            return p

        ptrs = [put(x) for x in _images(spec, entries)]
        out = np.empty((nb, m, n), dtype=np.uint16)
        c = rt.alloc(out.nbytes)
        grid, block = G.grid_for(spec, m, n, nb)
        if ngrid:
            grid = (ngrid, 1, 1)
        ka = G.runtime_kargs(spec, (*ptrs, c), m, n, k, nb, ngrid=ngrid)
        res = []
        for _ in range(launches):
            rt.memset(c, 0xFF, out.nbytes)  # NaN-fill: an unwritten output fails every check
            rt.launch(self.fn, grid, block, ka)
            rt.wait_stream(0)
            rt.memcpy_d2h((ctypes.c_ubyte * out.nbytes).from_buffer(out), c, out.nbytes)
            res.append(out.copy())
        for p in ptrs + [c]:
            rt.free(p)
        return res

    def close(self):
        self.mod.unload()


def _shape(spec, j=1, gm=3, gn=2):
    """A problem that differs from the build's reference shape: gm x gn tiles (not a multiple of 8 CTAs: the
    grouped order's fallback), K = k_ref + j loop bodies."""
    U = G.loop_plan(spec)[0]
    return gm * spec.tile_m, gn * spec.tile_n, spec.k + 128 * U * j


def _entries(m, n, k, dist, fmt, nb=2, seed=11):
    return [(L.generate(m, k, dist, seed=seed + 7 * e, tag=1, fmt=fmt), L.generate(n, k, dist, seed=seed + 7 * e, tag=2,
                                                                                    fmt=fmt)) for e in range(nb)]


def _check_all(kern, spec, m, n, k, fmt, nb=2):
    for name, dist in (("int", L.DIST_INT), ("gauss", L.DIST_GAUSS), ("uniform", L.DIST_UNIFORM_CODES)):
        ents = _entries(m, n, k, dist, fmt, nb)
        outs = kern.run(ents)
        assert all(np.array_equal(outs[0], o) for o in outs[1:]), f"{name}: launches differ (race)"
        for e, (a, b) in enumerate(ents):
            ref, s_abs = L.reference_f64(a, b)
            r = L.check_against_reference(outs[0][e], ref, s_abs, exact=(name == "int"))
            assert r["pass"], f"{name} entry {e} ({m}x{n}x{k}): {r}"


@pytest.mark.parametrize("fmt", ["e2m3", "e3m2"])
@pytest.mark.parametrize("fam", ["ak", "am"])
@pytest.mark.parametrize("case", list(FAMILY_CASES))
def test_family_full_output(runtime, case, fam, fmt):
    spec = LB.family_spec(*FAMILY_CASES[case], fam, fmt)
    kern = _Kernel(runtime, spec)
    try:
        m, n, k = _shape(spec)
        _check_all(kern, spec, m, n, k, fmt)
        # the same code object at another problem: more tiles, more K, one entry
        m2, n2, k2 = _shape(spec, j=2, gm=4, gn=5)
        _check_all(kern, spec, m2, n2, k2, fmt, nb=1)
    finally:
        kern.close()


@pytest.mark.parametrize("var", list(VARIANTS))
@pytest.mark.parametrize("fam", ["ak", "am"])
def test_variant_full_output(runtime, var, fam):
    cfg, over = VARIANTS[var]
    if (var.startswith("line2m") or var.startswith("b128b64") or var.startswith("adir")) and fam != "ak":
        pytest.skip("the line rings hold K-contiguous (canonical) activations")
    spec = LB.family_spec(*cfg, fam, "e2m3", **over)
    kern = _Kernel(runtime, spec)
    try:
        m, n, k = _shape(spec)
        _check_all(kern, spec, m, n, k, "e2m3")
        if spec.rt_tail == "switch":  # every K residue of the loop body
            U = G.loop_plan(spec)[0]
            for r in range(1, U):
                _check_all(kern, spec, m, n, spec.k + 128 * (U + r), "e2m3", nb=1)
    finally:
        kern.close()


@pytest.mark.parametrize("fam", ["ak", "am"])
@pytest.mark.parametrize("case", ["tiny-64x64", "square-128x128", "square-224x256", "large-256x288", "du512-128x160",
                                  "tall-512x32", "wide-32x512"])
def test_family_edge_suite(runtime, case, fam):
    """The edge suite (extreme / NaN scales, zero blocks, max-magnitude codes, subnormal results kept, -0) through a
    family build, both formats, OCP / IEEE convention (``edge_cases.check_ieee``)."""
    for fmt in ("e2m3", "e3m2"):
        spec = LB.family_spec(*FAMILY_CASES[case], fam, fmt)
        m = -(-512 // spec.tile_m) * spec.tile_m
        n = -(-512 // spec.tile_n) * spec.tile_n
        U = G.loop_plan(spec)[0]
        k = spec.k + 128 * U * max(0, -(-(1024 - spec.k) // (128 * U)))
        es = E.edge_operands(m, n, k, fmt=fmt)
        kern = _Kernel(runtime, spec)
        try:
            out = kern.run([(es.a, es.b)], launches=2)
        finally:
            kern.close()
        assert np.array_equal(out[0], out[1])
        ref32, s_abs = E.reference_ieee(es.a, es.b)
        r = E.check_ieee(out[0][0], ref32, s_abs, cases=es.cases)
        assert r["bad"] == 0, f"{case} {fam} {fmt}: {r}"


MUTANT_CASES = ["tiny-64x64", "tall-512x32", "wide-32x512", "square-128x128", "square-256x256", "large-256x288",
                "du512-32x32", "du512-128x160"]


@pytest.mark.parametrize("over", [{"mutant": "scale_sel"}, {"mutant": "scale_lane"}, {"mutant": "k_order"},
                                  {"mutant": "drop_kstep"}, {"mutant": "wrong_fmt"}, {"mutant": "batch_swap"},
                                  {"probe": "l2m_nocopy"}, {"probe": "l2m_lead4"}])
@pytest.mark.parametrize("cfg", [(256, 256, 256), (128, 128, 256), (32, 128, 256)])
def test_line2m_runtime_mutants_fail(runtime, over, cfg):
    """Must-fail mutants of the runtime-shape line2m family variant (incl. the line-ring hazards) on a 64-, 16- and
    4-MFMA-per-tile tile."""
    spec = LB.family_spec(*cfg, "ak", "e2m3", ring="line2m", **over)
    m, n, k = _shape(spec)
    ents = _entries(m, n, k, L.DIST_GAUSS, "e2m3", nb=2)
    kern = _Kernel(runtime, spec)
    try:
        out = kern.run(ents, launches=1)[0]
    finally:
        kern.close()
    passes = []
    for e, (a, b) in enumerate(ents):
        ref, s_abs = L.reference_f64(a, b)
        passes.append(L.check_against_reference(out[e], ref, s_abs)["pass"])
    assert not all(passes), f"line2m + {over} passed"


@pytest.mark.parametrize("mutant", ["scale_sel", "scale_lane", "k_order", "drop_kstep", "wrong_fmt", "batch_swap",
                                    "mc_noswz"])
@pytest.mark.parametrize("fam", ["ak", "am"])
@pytest.mark.parametrize("case", MUTANT_CASES)
def test_family_mutants_fail(runtime, case, fam, mutant):
    """Must-fail mutants built on each family default: every one must fail the Gaussian check on some entry
    (batch_swap: entry z multiplies entry z ^ 1's A; mc_noswz: the M-contiguous reads ignore the slot swizzle)."""
    if mutant == "mc_noswz" and fam != "am":
        pytest.skip("M-contiguous family only")
    spec = LB.family_spec(*FAMILY_CASES[case], fam, "e2m3", mutant=mutant)
    if mutant == "scale_sel" and spec.tile_m // spec.waves_m == 16 and spec.tile_n // spec.waves_n == 16:
        pytest.skip("one MFMA tile per wave: op_sel is always 0 (scale_lane covers the scales)")
    m, n, k = _shape(spec)
    ents = _entries(m, n, k, L.DIST_GAUSS, "e2m3", nb=2)
    kern = _Kernel(runtime, spec)
    try:
        out = kern.run(ents, launches=1)[0]
    finally:
        kern.close()
    passes = []
    for e, (a, b) in enumerate(ents):
        ref, s_abs = L.reference_f64(a, b)
        passes.append(L.check_against_reference(out[e], ref, s_abs)["pass"])
    assert not all(passes), f"{case} {fam} + {mutant} passed"


def test_tr_b6_semantics(runtime):
    """ds_read_b96_tr_b6 on the GPU equals the model the CPU data-path tests use: per 16-lane group, lane t receives
    element t of the 96-bit rows of lanes j = 0..15 (element j at bits 6 j). Built with the direct compiler (the
    kernels' path, trtag lever) and with LLVM (the asm waits for itself)."""
    from rocke.core.ir import I32, I64, IRBuilder, PtrType, VectorType
    from rocke.helpers.compile import compile_kernel
    tr_b6 = L.tr_b6_model

    rng = np.random.default_rng(2026)
    inp = rng.integers(0, 2**32, size=(64, 4), dtype=np.uint64).astype(np.uint32)
    perm = rng.permutation(64).astype(np.uint32)  # lane l reads slot perm[l]
    want = np.zeros((64, 3), np.uint32)
    slots = inp.view(np.uint8).reshape(64, 16)
    for g in range(4):
        rows = np.stack([slots[perm[16 * g + j], :12] for j in range(16)])
        want[16 * g:16 * g + 16] = tr_b6(rows).reshape(16, 12).view(np.uint32)
    for backend in ("rust", "llvm"):
        b = IRBuilder(f"tr_b6_probe_{backend}")
        b.kernel.attrs["max_workgroup_size"] = 64
        p_in = b.param("inp", PtrType(I32, "global"))
        p_perm = b.param("perm", PtrType(I32, "global"))
        p_out = b.param("out", PtrType(I32, "global"))
        lane = b.lane_id()
        lds = b.smem_alloc(I32, [256], name_hint="trlds")
        rs_in = b.buffer_rsrc(p_in, b.const_i32(1024))
        v = b.buffer_load_vN(rs_in, b.mul(lane, b.const_i32(16)), b.const_i32(0), I32, 4)
        b.smem_store_vN(lds, [b.mul(lane, b.const_i32(4))], v, 4)
        b.sync()
        rs_p = b.buffer_rsrc(p_perm, b.const_i32(256))
        slot = b.vec_extract(b.buffer_load_vN(rs_p, b.mul(lane, b.const_i32(4)), b.const_i32(0), I32, 1), 0)
        addr = b.trunc(b.smem_ptr_add(b.smem_addr_of(lds), b.zext(b.mul(slot, b.const_i32(16)), I64)), I32)
        tmpl = "ds_read_b96_tr_b6 $0, $1" + ("" if backend == "rust" else "\n\ts_waitcnt lgkmcnt(0)")
        r = b.inline_asm(tmpl, "=v,v", [addr], VectorType(I32, 3), sideeffect=True)
        rs_o = b.buffer_rsrc(p_out, b.const_i32(1024))
        r4 = b.vec_pack([b.vec_extract(r, 0), b.vec_extract(r, 1), b.vec_extract(r, 2), b.const_i32(0)], I32)
        b.buffer_store_vN(rs_o, b.mul(lane, b.const_i32(16)), b.const_i32(0), r4)
        b.ret()
        k = b.kernel
        if backend == "rust":
            from benchmarks.gfx950.mxfp6_gemm.bench import _compile_rust

            with tempfile.TemporaryDirectory() as td:
                out = Path(td) / "k.hsaco"
                old = os.environ.get("ROCKE_RUST_LEVERS")
                os.environ["ROCKE_RUST_LEVERS"] = "dsb96,trtag"
                try:
                    _compile_rust(k, out)
                finally:
                    if old is None:
                        os.environ.pop("ROCKE_RUST_LEVERS", None)
                    else:
                        os.environ["ROCKE_RUST_LEVERS"] = old
                hs = out.read_bytes()
        else:
            hs = compile_kernel(k, arch="gfx950", backend="python").hsaco
        mod = runtime.load_module(hs)
        fn = mod.get_function(f"tr_b6_probe_{backend}")

        def put(x):
            x = np.ascontiguousarray(x)
            p = runtime.alloc(x.nbytes)
            runtime.memcpy_h2d(p, (ctypes.c_ubyte * x.nbytes).from_buffer(x), x.nbytes)
            return p

        pi, pp = put(inp), put(perm)
        po = runtime.alloc(64 * 16)
        runtime.launch(fn, (1, 1, 1), (64, 1, 1), struct.pack("<QQQ", pi, pp, po))
        runtime.wait_stream(0)
        got4 = np.empty((64, 4), np.uint32)
        runtime.memcpy_d2h((ctypes.c_ubyte * got4.nbytes).from_buffer(got4), po, got4.nbytes)
        got = got4[:, :3]
        for p in (pi, pp, po):
            runtime.free(p)
        mod.unload()
        assert np.array_equal(got, want), f"{backend}: tr_b6 differs from the model (first bad lane " \
                                          f"{int(np.argmax((got != want).any(axis=1)))})"


@pytest.mark.parametrize("fam", ["ak", "am"])
def test_kbug1_32x32_large_grid_batch1(runtime, fam):
    """K-BUG-1 regression (harness lib-base1): 32x32x512 at the S problem 1 x 2048 x 2048 x 8192 (4096 CTAs, batch
    1) -- the reported fault was the batch_swap mutant at batch 1 reading past A (now refused by the host contract);
    the candidate itself must be bit-exact on the integer set and within tolerance on Gaussian data."""
    spec = LB.family_spec(32, 32, 512, fam, "e2m3")
    kern = _Kernel(runtime, spec)
    try:
        for name, dist in (("int", L.DIST_INT), ("gauss", L.DIST_GAUSS)):
            ents = _entries(2048, 2048, 8192, dist, "e2m3", nb=1)
            outs = kern.run(ents, launches=2)
            assert np.array_equal(outs[0], outs[1])
            ref, s_abs = L.reference_f64(*ents[0])
            r = L.check_against_reference(outs[0][0], ref, s_abs, exact=(name == "int"))
            assert r["pass"], (fam, name, r)
    finally:
        kern.close()


#: (MT0, MT1, DepthU) x grids: the even (XCD-range) walk at 4 x 4 tiles x 2 entries = 32 units -- fewer CTAs than
#: units (several trips), uneven trips (24), exactly one unit each, more CTAs than units; the odd walk at 3 x 3 x 1
PERSIST_CASES = [(128, 128, 256), (256, 256, 256), (64, 64, 256), (32, 128, 256), (256, 288, 256), (32, 32, 512)]


@pytest.mark.parametrize("ov", [False, True])
@pytest.mark.parametrize("fam", ["ak", "am"])
@pytest.mark.parametrize("cfg", PERSIST_CASES)
def test_persistent_grids(runtime, cfg, fam, ov):
    spec = LB.family_spec(*cfg, fam, "e2m3", schedule="persistent", persist_overlap=ov)
    kern = _Kernel(runtime, spec)
    try:
        U = G.loop_plan(spec)[0]
        for gm, gn, nb, grids in ((4, 4, 2, (8, 16, 24, 32, 48)), (3, 3, 1, (1, 4, 7, 9, 16))):
            m, n, k = gm * spec.tile_m, gn * spec.tile_n, spec.k + 128 * U
            ents = _entries(m, n, k, L.DIST_INT, "e2m3", nb)
            for ngrid in grids:
                outs = kern.run(ents, launches=2, ngrid=ngrid)
                assert np.array_equal(outs[0], outs[1]), (ngrid, "launches differ")
                for e, (a, b) in enumerate(ents):
                    ref, s_abs = L.reference_f64(a, b)
                    r = L.check_against_reference(outs[0][e], ref, s_abs, exact=True)
                    assert r["pass"], (cfg, fam, gm, gn, nb, ngrid, e, r)
    finally:
        kern.close()


@pytest.mark.parametrize("ov", [False, True])
@pytest.mark.parametrize("fam", ["ak", "am"])
@pytest.mark.parametrize("cfg", [(128, 128, 256), (64, 64, 256), (256, 256, 256)])
def test_persist_drop_mutant_fails(runtime, cfg, fam, ov):
    spec = LB.family_spec(*cfg, fam, "e2m3", schedule="persistent", mutant="persist_drop", persist_overlap=ov)
    kern = _Kernel(runtime, spec)
    try:
        m, n, k = 4 * spec.tile_m, 4 * spec.tile_n, spec.k
        ents = _entries(m, n, k, L.DIST_INT, "e2m3", 2)
        out = kern.run(ents, launches=1, ngrid=16)[0]
        bad = any(not L.check_against_reference(out[e], *L.reference_f64(a, b), exact=True)["pass"]
                  for e, (a, b) in enumerate(ents))
        assert bad, "persist_drop passed"
    finally:
        kern.close()


@pytest.mark.parametrize("fam", ["ak", "am"])
@pytest.mark.parametrize("cfg,P", [((64, 64, 256), 4), ((64, 64, 256), 2), ((32, 32, 512), 4), ((128, 128, 256), 2)])
def test_wper_rot_mutant_fails(runtime, cfg, P, fam):
    """wtile_period: reading the next K tile's dwords of the period load must fail."""
    spec = LB.family_spec(*cfg, fam, "e2m3", scale_src="vgpr", wtile_period=P, mutant="wper_rot")
    kern = _Kernel(runtime, spec)
    try:
        m, n, k = _shape(spec)
        ents = _entries(m, n, k, L.DIST_GAUSS, "e2m3", 1)  # varying scales (the integer set's are constant)
        out = kern.run(ents, launches=1)[0]
        assert not L.check_against_reference(out[0], *L.reference_f64(*ents[0]))["pass"]
    finally:
        kern.close()


def test_raw_scale_transpose_semantics(runtime):
    """gemm.raw_scale_transpose (v_permlane32_swap + v_perm, v_permlane16_swap + v_perm) on the GPU equals
    layout.raw_transpose_model: lane 16 g + r gets byte g of the words of lanes 16 i + r as its byte i."""
    from rocke.core.ir import I32, IRBuilder, PtrType
    from benchmarks.gfx950.mxfp6_gemm.bench import _compile_rust

    rng = np.random.default_rng(77)
    inp = rng.integers(0, 2**32, size=64, dtype=np.uint64).astype(np.uint32)
    b = IRBuilder("raw_tr_probe")
    b.kernel.attrs["max_workgroup_size"] = 64
    p_in = b.param("inp", PtrType(I32, "global"))
    p_out = b.param("out", PtrType(I32, "global"))
    lane = b.lane_id()
    w = b.vec_extract(b.buffer_load_vN(b.buffer_rsrc(p_in, b.const_i32(256)), b.mul(lane, b.const_i32(4)),
                                       b.const_i32(0), I32, 1), 0)
    r = G.raw_scale_transpose(b, w, *G.raw_transpose_selectors(b, lane))
    b.buffer_store_vN(b.buffer_rsrc(p_out, b.const_i32(256)), b.mul(lane, b.const_i32(4)), b.const_i32(0),
                      b.vec_pack([r], I32))
    b.ret()
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "k.hsaco"
        _compile_rust(b.kernel, out)
        hs = out.read_bytes()
    mod = runtime.load_module(hs)
    fn = mod.get_function("raw_tr_probe")
    pi = runtime.alloc(256)
    runtime.memcpy_h2d(pi, (ctypes.c_ubyte * 256).from_buffer(inp), 256)
    po = runtime.alloc(256)
    runtime.launch(fn, (1, 1, 1), (64, 1, 1), struct.pack("<QQ", pi, po))
    runtime.wait_stream(0)
    got = np.empty(64, np.uint32)
    runtime.memcpy_d2h((ctypes.c_ubyte * 256).from_buffer(got), po, 256)
    runtime.free(pi)
    runtime.free(po)
    mod.unload()
    want = L.raw_transpose_model(inp)
    assert np.array_equal(got, want), f"first bad lane {int(np.argmax(got != want))}"


EDGE_CASES = [(64, 64, 256), (128, 128, 256), (256, 256, 256), (32, 128, 256), (128, 32, 256), (192, 256, 256),
              (32, 32, 512), (256, 288, 256)]


@pytest.mark.parametrize("fam", ["ak", "am"])
@pytest.mark.parametrize("cfg", EDGE_CASES)
def test_edge_tiles_full_output(runtime, cfg, fam):
    """edge_tiles: M / N not tile multiples (the last tile shifted), every output element against the reference
    (int bit-exact, Gaussian / uniform), launches identical; plus a tile-multiple shape on the same code object."""
    spec = LB.family_spec(*cfg, fam, "e2m3", edge_tiles=True)
    kern = _Kernel(runtime, spec)
    try:
        U = G.loop_plan(spec)[0]
        tm, tn = spec.tile_m, spec.tile_n
        for m, n in ((2 * tm + 16, tn + 48), (tm + 48, 2 * tn + 16), (tm, tn + 32), (3 * tm, 2 * tn)):
            _check_all(kern, spec, m, n, spec.k + 128 * U, "e2m3")
    finally:
        kern.close()


CONFIG_CASES = [(32, 128, 256), (64, 64, 256), (128, 32, 256), (32, 32, 512), (64, 512, 256), (512, 64, 256),
                (128, 128, 256), (192, 128, 256), (256, 128, 256), (96, 384, 256)]


@pytest.mark.parametrize("fam", ["ak", "am"])
@pytest.mark.parametrize("cfg", CONFIG_CASES)
def test_config_spec_full_output(runtime, cfg, fam):
    """config_spec (multi-tile work group + edge_tiles) at shapes of the config's own granularity (not multiples
    of the work-group tile), every output element, plus a WG-tile multiple."""
    spec = LB.config_spec(*cfg, fam, "e2m3")
    kern = _Kernel(runtime, spec)
    try:
        U = G.loop_plan(spec)[0]
        mt0, mt1 = cfg[0], cfg[1]
        tm, tn = spec.tile_m, spec.tile_n
        m1 = max(tm, 3 * mt1) + (mt1 if tm > mt1 else 16)
        n1 = max(tn, 3 * mt0) + (mt0 if tn > mt0 else 16)
        for m, n in ((m1, n1), (2 * tm, 2 * tn)):
            _check_all(kern, spec, m, n, spec.k + 128 * U, "e2m3")
    finally:
        kern.close()


def test_buffer_range_check_includes_soffset(runtime):
    """Hardware fact the K-tail / m_mask edges rely on: a raw buffer load (stride 0) is out of range -- reads 0 --
    when voffset + soffset reaches num_records (not only voffset). Lane l loads dword at voffset 4 l with soffset 64 B
    from a 128-byte descriptor over a 256-byte buffer of ones: lanes 0..15 in range, 16..63 must read 0."""
    from rocke.core.ir import I32, IRBuilder, PtrType
    from benchmarks.gfx950.mxfp6_gemm.bench import _compile_rust

    b = IRBuilder("range_probe")
    b.kernel.attrs["max_workgroup_size"] = 64
    p_in = b.param("inp", PtrType(I32, "global"))
    p_out = b.param("out", PtrType(I32, "global"))
    lane = b.lane_id()
    v = b.buffer_load_vN(b.buffer_rsrc(p_in, b.const_i32(128)), b.mul(lane, b.const_i32(4)), b.const_i32(64), I32, 1)
    b.buffer_store_vN(b.buffer_rsrc(p_out, b.const_i32(256)), b.mul(lane, b.const_i32(4)), b.const_i32(0), v)
    b.ret()
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "k.hsaco"
        _compile_rust(b.kernel, out)
        hs = out.read_bytes()
    mod = runtime.load_module(hs)
    fn = mod.get_function("range_probe")
    ones = np.arange(1, 65, dtype=np.uint32)
    pi = runtime.alloc(256)
    runtime.memcpy_h2d(pi, (ctypes.c_ubyte * 256).from_buffer(ones), 256)
    po = runtime.alloc(256)
    runtime.launch(fn, (1, 1, 1), (64, 1, 1), struct.pack("<QQ", pi, po))
    runtime.wait_stream(0)
    got = np.empty(64, np.uint32)
    runtime.memcpy_d2h((ctypes.c_ubyte * 256).from_buffer(got), po, 256)
    for p_ in (pi, po):
        runtime.free(p_)
    mod.unload()
    # in range: dword 16 + l for l < 16 (bytes 64 + 4 l < 128); beyond: 0 if soffset counts, else ones[16 + l]
    assert list(got[:16]) == list(ones[16:32]), got[:16]
    assert not got[16:].any(), f"soffset not in the range check: {got[16:24]}"


@pytest.mark.parametrize("fam", ["ak", "am"])
@pytest.mark.parametrize("cfg", [(128, 128, 256), (64, 64, 256), (256, 256, 256), (32, 128, 256)])
def test_k_tail_full_output(runtime, cfg, fam):
    """k_tail: K a multiple of 32 but not of 128 (ceil(K / 128) tiles). B and both scale images padded with zeros
    (layout.pad_k), A as produced; every output element against the fp64 reference of the unpadded K."""
    over = {"k_tail": True, "edge_tiles": True}
    if fam == "ak":
        over["ring"] = "stage"
    spec = LB.family_spec(*cfg, fam, "e2m3", **over)
    U = G.loop_plan(spec)[0]
    kern = _Kernel(runtime, spec)
    try:
        for short in (32, 64, 96):
            k = spec.k + 128 * U - short
            m, n = 2 * spec.tile_m, spec.tile_n + 16
            ok, why = G.runtime_problem_ok(spec, m, n, k, 1)
            assert ok, why
            for dist, exact in ((L.DIST_INT, True), (L.DIST_GAUSS, False)):
                a = L.generate(m, k, dist, seed=3, tag=1, fmt="e2m3")
                bo = L.generate(n, k, dist, seed=3, tag=2, fmt="e2m3")
                ap, bp = L.pad_k(a), L.pad_k(bo)
                la = G.operand_layout_of(spec, "A")
                a_img = L.mcontig_codes(a).reshape(-1) if la == "mcontig" else L.pack_operand(a, la, spec.tile_m)
                b_img = L.pack_operand(bp, G.operand_layout_of(spec, "B"), spec.tile_n)
                sa = L.pack_scales_for(ap.scales, spec.atom, G.scale_layout_of(spec, "A")).view(np.uint8)
                sb = L.pack_scales_for(bp.scales, spec.atom, G.scale_layout_of(spec, "B")).view(np.uint8)
                out = _launch_images(runtime, kern, spec, (a_img, b_img, sa, sb), m, n, k)
                ref, s_abs = L.reference_f64(a, bo)
                r = L.check_against_reference(out, ref, s_abs, exact=exact)
                assert r["pass"], (cfg, fam, k, dist, r)
    finally:
        kern.close()


def _launch_images(runtime, kern, spec, imgs, m, n, k):
    """One launch of ``kern`` on prebuilt device images (one entry); C NaN-filled first."""
    def put(x):
        x = np.ascontiguousarray(x).view(np.uint8).reshape(-1)
        p = runtime.alloc(x.nbytes)
        runtime.memcpy_h2d(p, (ctypes.c_ubyte * x.nbytes).from_buffer(x), x.nbytes)
        return p

    ptrs = [put(x) for x in imgs]
    out = np.empty((m, n), np.uint16)
    c = runtime.alloc(out.nbytes)
    runtime.memset(c, 0xFF, out.nbytes)
    grid, block = G.grid_for(spec, m, n, 1)
    runtime.launch(kern.fn, grid, block, G.runtime_kargs(spec, (*ptrs, c), m, n, k, 1))
    runtime.wait_stream(0)
    runtime.memcpy_d2h((ctypes.c_ubyte * out.nbytes).from_buffer(out), c, out.nbytes)
    for p in ptrs + [c]:
        runtime.free(p)
    return out
