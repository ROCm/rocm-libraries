# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""gfx950 numerics of the MXFP6 GEMM (needs a gfx950 GPU; skipped elsewhere).

Full-output checks at small shapes against the float64 reference: the exact
integer set must match bit for bit, the Gaussian / uniform-code sets must stay
within the scaled-MFMA tolerance, three launches must be identical (no races),
and every must-fail mutant must fail. The parametrized configurations include
knob combinations that once raced (regression guards).
"""

from __future__ import annotations

import ctypes
import struct
from dataclasses import replace

import numpy as np
import pytest
from kernels.gfx950.mxfp6_gemm import layout as L
from kernels.gfx950.mxfp6_gemm.gemm import Mxfp6GemmSpec, build_mxfp6_gemm, grid_of, kernel_name


def _gpu_arch():
    try:
        from rocke.runtime.hip_module import get_device_arch

        return get_device_arch()
    except Exception:  # noqa: BLE001 - no HIP runtime / device
        return None


pytestmark = pytest.mark.skipif(_gpu_arch() != "gfx950", reason="needs a gfx950 GPU")

SHAPE = {"m": 512, "n": 512, "k": 4096, "xcd_block": "none"}
FULL_GRID = {"m": 4096, "n": 4096, "k": 1024, "xcd_block": "4x8"}  # 256 CTAs: the XCD block orders apply
FULL_GRID_K8 = {**FULL_GRID, "k": 8192}  # line ring + k_rot: the per-XCD K offset a multiple of 8 tiles
#: knob combinations (on top of the defaults) that must be correct
CONFIGS = {
    "default": {},
    "e3m2": {"elem_fmt": "e3m2"},
    "e3m2-prefetch": {"elem_fmt": "e3m2", "frag_mode": "prefetch", "acq_slot": 8, "read_every": 3},
    "aiter-like-burst": {"read_lead": 64},
    "vgpr-scales": {"scale_src": "vgpr", "scale_slot": 2},
    "carry-first-row": {"carry": 9},
    "carry-lead6": {"carry": 4, "read_lead": 6, "carry_slot": 52},
    "paired-planes": {"operand_layout": "c0c1p", "dtl_imm": True},
    "rsrc-once": {"rsrc_once": True},
    "mixed-a-canonical": {"a_layout": "canonical"},
    "canonical-interleave": {"operand_layout": "canonical", "canon_dtl": "interleave"},
    "mixed-a-canonical-interleave": {"a_layout": "canonical", "canon_dtl": "interleave"},
    "raw-scales": {"scale_layout": "raw"},
    "canonical-line2": {"operand_layout": "canonical", "canon_ring": "line2", "frag_mode": "prefetch", "stages": 4,
                        "unroll": 2, "read_every": 1, "acq_slot": 2, "read_delay": 4},
    "mixed-line2-s2": {"a_layout": "canonical", "canon_ring": "line2", "frag_mode": "prefetch", "stages": 2,
                       "unroll": 4, "read_every": 1, "acq_slot": 2, "read_delay": 4},
    "mixed-line2-s3": {"a_layout": "canonical", "canon_ring": "line2", "frag_mode": "prefetch", "stages": 3,
                       "unroll": 8, "read_every": 1, "acq_slot": 2, "read_delay": 4, "canon_rsrc_once": True},
    "canonical-line2-early": {"operand_layout": "canonical", "canon_ring": "line2", "frag_mode": "prefetch",
                              "stages": 4, "unroll": 2, "read_every": 1, "acq_slot": 2, "read_delay": 0,
                              "line_release": 20, "line_early": True, "line_spread": 2},
    "mixed-line2-reads-first": {"a_layout": "canonical", "canon_ring": "line2", "frag_mode": "prefetch", "stages": 3,
                                "unroll": 8, "read_every": 3, "acq_slot": 2, "read_delay": 4,
                                "line_reads_first": True, "line_release": 30, "line_spread": 1},
    "canonical-line2-A-reads-first": {"operand_layout": "canonical", "canon_ring": "line2", "ring_ops": "A",
                                      "canon_dtl": "interleave", "frag_mode": "prefetch", "stages": 3, "unroll": 8,
                                      "read_every": 2, "acq_slot": 2, "read_delay": 0, "line_reads_first": True,
                                      "line_release": 19, "line_spread": 1, "dtl_delay": 0},
    "mixed-line3": {"a_layout": "canonical", "canon_ring": "line3", "frag_mode": "prefetch", "stages": 2,
                    "unroll": 2, "read_every": 1, "acq_slot": 2, "read_delay": 4},
    "canonical-line3-A": {"operand_layout": "canonical", "canon_ring": "line3", "ring_ops": "A",
                          "canon_dtl": "interleave", "frag_mode": "prefetch", "stages": 2, "unroll": 2,
                          "read_every": 3, "acq_slot": 2, "read_delay": 4},
    "canonical-line2-raw": {"operand_layout": "canonical", "canon_ring": "line2", "frag_mode": "prefetch",
                            "stages": 4, "unroll": 2, "read_every": 1, "acq_slot": 2, "read_delay": 4,
                            "scale_layout": "raw"},
    "canonical-raw-scales": {"operand_layout": "canonical", "canon_dtl": "interleave", "scale_layout": "raw"},
    "mixed-raw-scales-best": {"a_layout": "canonical", "canon_dtl": "interleave", "scale_layout": "raw",
                              "frag_mode": "prefetch", "acq_slot": 2, "read_every": 3, "read_delay": 4, "dtl_delay": 1,
                              "unroll": 4, "tile_lgkm0": True, "dtl_imm": True, "rsrc_once": True, "swap_ab": False},
    "raw-scales-carry": {"scale_layout": "raw", "carry": 2},
    "mixed-a-canonical-best": {"a_layout": "canonical", "frag_mode": "prefetch", "acq_slot": 2, "read_every": 3,
                               "read_delay": 4, "dtl_delay": 1, "unroll": 4, "tile_lgkm0": True, "dtl_imm": True,
                               "rsrc_once": True, "swap_ab": False},
    "mma-pairs": {"mma_order": "pairs"},
    "swap-ab-false": {"swap_ab": False},
    "store-nt": {"store_nt": True},
    "mma-blk2-prefetch": {"mma_order": "blk2", "frag_mode": "prefetch", "acq_slot": 2, "read_every": 3,
                          "read_delay": 4, "dtl_delay": 1, "unroll": 2, "operand_layout": "c0c1p"},
    "scale-x4w": {"scale_dtl": "x4w"},
    "scale-x4w-prefetch": {"scale_dtl": "x4w", "frag_mode": "prefetch", "acq_slot": 4, "read_every": 3,
                           "read_delay": 2, "unroll": 3, "dtl_imm": True, "rsrc_once": True},
    # the structure of PRESETS["best"] (built here with LLVM: correct, only slow)
    "preset-best-structure": {"frag_mode": "prefetch", "acq_slot": 4, "read_every": 3, "read_delay": 2, "unroll": 3,
                              "dtl_imm": True, "rsrc_once": True},
    "rsrc-once-imm-planes-prefetch": {"rsrc_once": True, "dtl_imm": True, "operand_layout": "c0c1p",
                                      "frag_mode": "prefetch", "acq_slot": 4, "read_every": 3, "read_delay": 2,
                                      "unroll": 2},
    "paired-planes-prefetch": {"operand_layout": "c0c1p", "frag_mode": "prefetch", "acq_slot": 8, "read_every": 3},
    "prefetch-lds-scales": {"frag_mode": "prefetch", "acq_slot": 8, "read_every": 3},
    "prefetch-c0c1": {"frag_mode": "prefetch", "scale_src": "vgpr", "scale_slot": 2, "acq_slot": 8, "read_every": 3},
    "eight-waves": {"waves_n": 4, "scale_src": "vgpr", "scale_slot": 2},
    "m1-layout": {"frag_mode": "prefetch", "stages": 2, "operand_layout": "canonical", "scale_layout": "kstep",
                  "scale_src": "vgpr", "read_lead": 8, "dtl_delay": 1},
    # once raced: DTLs clamped to the last MFMA slot were reordered behind the scale loads (search bug 2)
    "regress-dtl-spread-32x32": {"frag_mode": "prefetch", "stages": 2, "operand_layout": "canonical",
                                 "scale_layout": "kstep", "scale_src": "vgpr", "read_lead": 4, "dtl_delay": 1,
                                 "atom": "32x32x64", "dtl_spread": 5, "peel_first": True, "read_delay": 8,
                                 "scale_slot": 56, "setprio": 3},
    # once faulted / raced (search r4): direct-B loads past K under the clamped tail; a direct-B load sharing an
    # MFMA slot with a DTL was issued out of the static vmcnt model's order
    "regress-bdirect-clamp": {"b_direct": True, "dtl_spread": 2, "m_outer": False, "peel_first": True,
                              "release_slot": 24, "serpentine": False, "setprio": 3, "tail_mode": "clamp",
                              "tile_lgkm0": True},
    "regress-bdirect-slot-order": {"b_direct": True, "acc_pin": True, "dtl_aux": 3, "dtl_delay": 4, "dtl_spread": 2,
                                   "lds_alias_scopes": False, "m_outer": False, "read_lead": 64, "scale_src": "vgpr",
                                   "serpentine": False, "setprio": 3, "stages": 2, "tile_lgkm0": True},
    "regress-bdirect-depth2": {"b_direct": True, "dtl_delay": 4, "b_depth": 2, "scale_src": "vgpr", "scale_slot": 2},
    # once raced: no MFMA pinning lets the scheduler reorder the VM issue (search bug 4)
    "regress-nopin-late-acquire": {"frag_mode": "prefetch", "stages": 2, "operand_layout": "canonical",
                                   "scale_layout": "kstep", "scale_src": "vgpr", "acq_slot": 4, "atom": "32x32x64",
                                   "dtl_delay": 4, "dtl_spread": 5, "m_outer": False, "pin_mfma": False,
                                   "read_every": 4, "scale_slot": 56, "serpentine": False, "tile_lgkm0": True},
}


@pytest.fixture(scope="module")
def runtime():
    from rocke.runtime.hip_module import Runtime

    return Runtime()


def _hsaco(spec):
    """LLVM build, or the rocKE direct compiler for a spec that requires it (when ``ROCKE_RUST_COMPILE`` is set)."""
    import os
    import tempfile
    from pathlib import Path

    if spec.required_compiler == "rust":
        if not os.environ.get("ROCKE_RUST_COMPILE"):
            pytest.skip("spec requires the rocKE direct compiler (ROCKE_RUST_COMPILE not set)")
        from benchmarks.gfx950.mxfp6_gemm.bench import _compile_rust

        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "k.hsaco"
            _compile_rust(build_mxfp6_gemm(spec, arch="gfx950"), out)
            return out.read_bytes()
    from rocke.helpers.compile import compile_kernel

    return compile_kernel(build_mxfp6_gemm(spec, arch="gfx950"), arch="gfx950", backend="python").hsaco


def _run(rt, spec, a, b, launches=3):
    mod = rt.load_module(_hsaco(spec))
    fn = mod.get_function(kernel_name(spec))

    def put(x):
        x = np.ascontiguousarray(x)
        p = rt.alloc(x.nbytes)
        rt.memcpy_h2d(p, (ctypes.c_ubyte * x.nbytes).from_buffer(x), x.nbytes)
        return p

    from kernels.gfx950.mxfp6_gemm.gemm import operand_layout_of

    ptrs = [put(L.pack_operand(a, operand_layout_of(spec, "A"), spec.tile_m)),
            put(L.pack_operand(b, operand_layout_of(spec, "B"), spec.tile_n)),
            put(L.pack_scales_for(a.scales, spec.atom, spec.scale_layout)),
            put(L.pack_scales_for(b.scales, spec.atom, spec.scale_layout))]
    c = rt.alloc(spec.m * spec.n * 2)
    out = np.empty((spec.m, spec.n), dtype=np.uint16)
    grid, block = grid_of(spec)
    res = []
    for _ in range(launches):
        rt.memset(c, 0xFF, out.nbytes)
        rt.launch(fn, grid, block, struct.pack("<QQQQQ", *ptrs, c))
        rt.wait_stream(0)
        rt.memcpy_d2h((ctypes.c_ubyte * out.nbytes).from_buffer(out), c, out.nbytes)
        res.append(out.copy())
    for p in ptrs + [c]:
        rt.free(p)
    mod.unload()
    return res


@pytest.fixture(scope="module")
def data():
    """(element format) -> dist -> (a, b, (ref, abs_sum)): the data are generated in the spec's format."""
    m, n, k = SHAPE["m"], SHAPE["n"], SHAPE["k"]
    out = {}
    for fmt in L.FMTS:
        out[fmt] = {}
        for name, dist in (("int", L.DIST_INT), ("gauss", L.DIST_GAUSS), ("uniform", L.DIST_UNIFORM_CODES)):
            a = L.generate(m, k, dist, seed=42, tag=1, fmt=fmt)
            b = L.generate(n, k, dist, seed=42, tag=2, fmt=fmt)
            out[fmt][name] = (a, b, L.reference_f64(a, b))
    return out


@pytest.mark.parametrize("cfg", list(CONFIGS))
def test_full_output(runtime, data, cfg):
    spec = replace(Mxfp6GemmSpec(), **SHAPE, **CONFIGS[cfg])
    for name, (a, b, (ref, abs_sum)) in data[spec.elem_fmt].items():
        outs = _run(runtime, spec, a, b)
        assert all(np.array_equal(outs[0], o) for o in outs[1:]), f"{cfg}/{name}: launches differ (race)"
        r = L.check_against_reference(outs[0], ref, abs_sum, exact=(name == "int"))
        assert r["pass"], f"{cfg}/{name}: {r}"


@pytest.mark.parametrize("fmt", ["e2m3", "e3m2"])
@pytest.mark.parametrize("mutant", ["scale_sel", "k_order", "drop_kstep", "wrong_fmt"])
def test_mutants_fail(runtime, data, mutant, fmt):
    spec = replace(Mxfp6GemmSpec(), **SHAPE, mutant=mutant, elem_fmt=fmt)
    a, b, (ref, abs_sum) = data[fmt]["gauss"]
    r = L.check_against_reference(_run(runtime, spec, a, b, launches=1)[0], ref, abs_sum)
    assert not r["pass"], f"mutant {mutant} ({fmt}) passed"


@pytest.mark.parametrize("fmt", ["e2m3", "e3m2"])
def test_wrong_format_data_fails(runtime, data, fmt):
    """Data of one format through the kernel built for the other must not pass (format is really used)."""
    other = "e3m2" if fmt == "e2m3" else "e2m3"
    spec = replace(Mxfp6GemmSpec(), **SHAPE, elem_fmt=other)
    a, b, (ref, abs_sum) = data[fmt]["int"]
    r = L.check_against_reference(_run(runtime, spec, a, b, launches=1)[0], ref, abs_sum, exact=True)
    assert not r["pass"]


# ----------------------------------------------------------------- edge cases (OCP MX / IEEE reference convention)

_EDGE_SPECS = {
    "default": {},
    "e3m2": {"elem_fmt": "e3m2"},
    "swap-ab-false": {"swap_ab": False},
    "best": "best",
    "best_v3": "best_v3",
    "best_v4": "best_v4",
    "best-e3m2": ("best", {"elem_fmt": "e3m2"}),
    "best-swap-ab-false": ("best", {"swap_ab": False}),
    "best-swap-ab-false-e3m2": ("best", {"swap_ab": False, "elem_fmt": "e3m2"}),
    "best-raw-scales": ("best", {"scale_layout": "raw"}),
    "best-mixed-raw-scales": ("best", {"a_layout": "canonical", "canon_dtl": "interleave", "scale_layout": "raw"}),
    "canonical-raw-scales": {"operand_layout": "canonical", "canon_dtl": "interleave", "scale_layout": "raw"},
    "best_canonical": ("best_canonical", {"k_rot": 0}),
    "best_canonical_stage": ("best_canonical_stage", {"k_rot": 0}),
    "best_canonical-rsrc-once": ("best_canonical", {"k_rot": 0, "canon_rsrc_once": True}),
    "best_mixed-rsrc-once": ("best_mixed", {"canon_rsrc_once": True}),
    "best_mixed": "best_mixed",
    "best_mixed_stage": "best_mixed_stage",
    "best_mixed_line2": "best_mixed_line2",
    "best_canonical-e3m2": ("best_canonical", {"k_rot": 0, "elem_fmt": "e3m2"}),
    "best_mixed-e3m2": ("best_mixed", {"elem_fmt": "e3m2"}),
    "canonical-line2": ("best_canonical", {"k_rot": 0, "canon_ring": "line2", "ring_ops": "all", "stages": 4,
                                           "unroll": 2, "read_every": 1, "line_reads_first": False,
                                           "line_release": 24}),
    "mixed-line2": ("best_mixed", {"canon_ring": "line2", "stages": 2, "unroll": 4, "read_every": 1}),
    "mixed-line3": ("best_mixed", {"canon_ring": "line3", "stages": 2, "unroll": 2, "read_every": 1}),
    "mixed-line2m": ("best_mixed", {"canon_ring": "line2m"}),
    "mixed-line3-breg": ("best_mixed", {"canon_ring": "line3", "stages": 2, "unroll": 2, "b_reg": True}),
    "mixed-line3-breg-full-grid-krot": ("best_mixed", {"canon_ring": "line3", "stages": 2, "unroll": 2, "b_reg": True,
                                                       "k_rot": 1}, FULL_GRID_K8),
    "v6-breg": ("best_v6", {"stages": 2, "b_reg": True}),
    "v6-breg-u2": ("best_v6", {"stages": 2, "b_reg": True, "unroll": 2}),
    "canonical-line2m": ("best_canonical", {"k_rot": 0, "canon_ring": "line2m"}),
    "canonical-line2m-all": ("best_canonical", {"k_rot": 0, "canon_ring": "line2m", "ring_ops": "all", "stages": 4,
                                                "unroll": 2, "line_release": 20}),
    "mixed-line2m-full-grid-krot": ("best_mixed", {"canon_ring": "line2m", "k_rot": 1}, FULL_GRID_K8),
    "mixed-line2-32x32-kstep": ("best_mixed_line2", {"atom": "32x32x64", "swap_ab": True, "scale_layout": "kstep",
                                                "scale_src": "vgpr", "scale_slot": 32, "read_split": False,
                                                "unroll": 4}),
    "mixed-line2-32x32-tile": ("best_mixed_line2", {"atom": "32x32x64", "swap_ab": True, "unroll": 8}),
    # K = 8192 (a 2-trip loop): at K = 4096 the direct compiler flattens the single-trip 24-tile body and runs out
    # of VGPRs (canonical 32x32 + VGPR kstep scales)
    "canonical-line2-32x32-kstep": ("best_canonical", {"k_rot": 0, "atom": "32x32x64", "swap_ab": True,
                                                       "scale_layout": "kstep", "scale_src": "vgpr",
                                                       "scale_slot": 32, "read_split": False, "unroll": 4},
                                    FULL_GRID_K8),
    "canonical-line3-A": ("best_canonical_stage", {"k_rot": 0, "canon_ring": "line3", "ring_ops": "A",
                                                   "stages": 2, "unroll": 2}),
    "canonical-line2-full-grid-krot": ("best_canonical", {"canon_ring": "line2", "ring_ops": "all", "stages": 4,
                                                          "unroll": 2, "read_every": 1, "line_reads_first": False,
                                                          "line_release": 24}, FULL_GRID_K8),
    # the full CTA grid (an XCD block order: k_rot needs it), short K
    "best_canonical-full-grid": ("best_canonical", {}, FULL_GRID_K8),
    "best_canonical_stage-full-grid": ("best_canonical_stage", {}, FULL_GRID),
    "best_mixed-full-grid-krot": ("best_mixed", {"k_rot": 1}, FULL_GRID_K8),
    "best_mixed_stage-full-grid-krot": ("best_mixed_stage", {"k_rot": 1}, FULL_GRID),
}


@pytest.mark.parametrize("name", list(_EDGE_SPECS))
def test_edge_cases(runtime, name):
    """E8M0 0x00 / 0x01 / 0x7F / 0xFE / 0xFF (NaN), zero blocks, max-magnitude codes at the bf16 / fp32 overflow,
    subnormal codes, -0, shuffled scales and mixed extreme rows at tile / K-tile boundaries (edge_cases.py)."""
    from kernels.gfx950.mxfp6_gemm import edge_cases as E
    from kernels.gfx950.mxfp6_gemm.gemm import preset

    v = _EDGE_SPECS[name]
    if isinstance(v, str):
        spec = preset(v, **SHAPE)
    elif isinstance(v, tuple):
        spec = preset(v[0], **(v[2] if len(v) > 2 else SHAPE), **v[1])
    else:
        spec = replace(Mxfp6GemmSpec(), **SHAPE, **v)
    es = E.edge_operands(spec.m, spec.n, spec.k, fmt=spec.elem_fmt)
    ref32, abs_sum = E.reference_ieee(es.a, es.b)
    outs = _run(runtime, spec, es.a, es.b, launches=2)
    assert np.array_equal(outs[0], outs[1]), f"{name}: launches differ"
    r = E.check_ieee(outs[0], ref32, abs_sum, es.cases)
    assert r["bad"] == 0, f"{name}: {r}"


@pytest.mark.parametrize("name", ["default", "best", "best-e3m2", "best_canonical", "best_mixed", "best_canonical-e3m2",
                                  "best_mixed-e3m2"])
def test_subnormal_behaviour(runtime, name, capsys):
    """Outputs in the fp32 subnormal range follow one convention (all flushed or all kept); the result is printed and
    recorded in the notes (the reference accepts either)."""
    from kernels.gfx950.mxfp6_gemm import edge_cases as E
    from kernels.gfx950.mxfp6_gemm.gemm import preset

    v = _EDGE_SPECS[name]
    spec = preset(v, **SHAPE) if isinstance(v, str) else (preset(v[0], **SHAPE, **v[1]) if isinstance(v, tuple)
                                                        else replace(Mxfp6GemmSpec(), **SHAPE, **v))
    es = E.edge_operands(spec.m, spec.n, spec.k, fmt=spec.elem_fmt)
    ref32, _ = E.reference_ieee(es.a, es.b)
    out = _run(runtime, spec, es.a, es.b, launches=1)[0]
    rows, cols = es.cases["fp32_subnormal"]
    beh = E.subnormal_behaviour(out, ref32, rows, cols)
    with capsys.disabled():
        print(f"[subnormal] {name}: {beh}")
    assert beh == "keep"  # documented convention: subnormal results are kept (edge_cases.py)


# The GPU pack / quantize tests (pack.py) moved to test_mxfp6_pack_gpu.py (pack agent, 2026-11-07).
