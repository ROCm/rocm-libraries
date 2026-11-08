# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""gfx950 numerics of the wide-tile MXFP6 GEMM (``gemm_wide``; needs a gfx950 GPU and the rocKE direct compiler
``ROCKE_RUST_COMPILE``; skipped elsewhere).

Full-output checks at small shapes (wide + narrow tiles in one grid) against the float64 reference: the exact integer
set bit for bit, Gaussian / uniform-code sets within the scaled-MFMA tolerance, three launches identical, every
must-fail mutant fails, and the OCP / IEEE edge suite (``edge_cases.py``) incl. the kept fp32 subnormals."""

from __future__ import annotations

import ctypes
import os
import struct
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from kernels.gfx950.mxfp6_gemm import gemm_wide as W
from kernels.gfx950.mxfp6_gemm import layout as L


def _gpu_arch():
    try:
        from rocke.runtime.hip_module import get_device_arch

        return get_device_arch()
    except Exception:  # noqa: BLE001 - no HIP runtime / device
        return None


pytestmark = [pytest.mark.skipif(_gpu_arch() != "gfx950", reason="needs a gfx950 GPU"),
              pytest.mark.skipif(not os.environ.get("ROCKE_RUST_COMPILE"), reason="needs ROCKE_RUST_COMPILE")]

MIX = {"m": 512, "n": 640, "k": 4096, "xcd_map": "none"}  # one wide + one narrow column, 2 tile rows
BAND = {"m": 2048, "n": 640, "k": 2048, "xcd_map": "band"}  # the band XCD map (16 CTAs)
CONFIGS = {
    "default": {},
    "band": BAND,
    "two-wide-one-narrow": {"n": 1024},
    "all-wide": {"n": 768},
    "narrow-only": {"tile_n": 256, "n": 512},
    "stages-a2": {"stages_a": 2},
    "no-release": {"release": False},
    "no-release-a2": {"release": False, "stages_a": 2},
    "unroll2": {"unroll": 2},
    "no-dtl-imm": {"dtl_imm": False},
    "early-acquire": {"acq_delay": -4, "release_delay": 4},
    "b-dtl-to-end": {"b_dtl_end": "end", "dtl_delay": 0},
    "a-delay": {"a_delay": 4},
    "head-rows-2": {"head_rows": 2},
    "a-dtl-spread": {"a_dtl": "spread"},
    "a-dtl-spread-end": {"a_dtl": "spread", "a_dtl_end": "end"},
    "narrow-one-barrier": {"tile_n": 256, "n": 512, "stages_b": 3, "release": False, "b_dtl": "spread"},
    "scale-b128": {"sc_exact": False},
    "setprio": {"setprio": 1},
    "e3m2": {"elem_fmt": "e3m2"},
}


def _spec(over):
    return replace(W.WideGemmSpec(), **{**MIX, **over})


def _hsaco(spec):
    from benchmarks.gfx950.mxfp6_gemm.bench import _compile_rust

    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "k.hsaco"
        _compile_rust(W.build_mxfp6_gemm_wide(spec), out)
        return out.read_bytes()


@pytest.fixture(scope="module")
def runtime():
    from rocke.runtime.hip_module import Runtime

    return Runtime()


def _run(rt, spec, a, b, launches=3):
    mod = rt.load_module(_hsaco(spec))
    fn = mod.get_function(W.kernel_name(spec))

    def put(x):
        x = np.ascontiguousarray(x)
        p = rt.alloc(x.nbytes)
        rt.memcpy_h2d(p, (ctypes.c_ubyte * x.nbytes).from_buffer(x), x.nbytes)
        return p

    ptrs = [put(x) for x in W.host_images(spec, a, b)]
    c = rt.alloc(spec.m * spec.n * 2)
    extra = []
    if spec.stream_k:  # partial workspace + flags (zeroed once; every launch leaves them zero)
        wsb, flb = W.workspace_bytes(spec)
        extra = [rt.alloc(wsb), rt.alloc(flb)]
        rt.memset(extra[1], 0, flb)
    out = np.empty((spec.m, spec.n), dtype=np.uint16)
    grid, block = W.grid_of(spec)
    res = []
    for _ in range(launches):
        rt.memset(c, 0xFF, out.nbytes)
        rt.launch(fn, grid, block, struct.pack("<" + "Q" * (5 + len(extra)), *ptrs, c, *extra))
        rt.wait_stream(0)
        rt.memcpy_d2h((ctypes.c_ubyte * out.nbytes).from_buffer(out), c, out.nbytes)
        res.append(out.copy())
    if extra:  # the protocol left every flag reset
        fl = np.empty(W.workspace_bytes(spec)[1] // 4, dtype=np.int32)
        rt.memcpy_d2h((ctypes.c_ubyte * fl.nbytes).from_buffer(fl), extra[1], fl.nbytes)
        assert not fl.any(), "stream-K flags not reset"
    for p in ptrs + [c] + extra:
        rt.free(p)
    mod.unload()
    return res


_DATA = {}


def _data(m, n, k, fmt):
    key = (m, n, k, fmt)
    if key not in _DATA:
        out = {}
        for name, dist in (("int", L.DIST_INT), ("gauss", L.DIST_GAUSS), ("uniform", L.DIST_UNIFORM_CODES)):
            a = L.generate(m, k, dist, seed=42, tag=1, fmt=fmt)
            b = L.generate(n, k, dist, seed=42, tag=2, fmt=fmt)
            out[name] = (a, b, L.reference_f64(a, b))
        _DATA[key] = out
    return _DATA[key]


@pytest.mark.parametrize("cfg", list(CONFIGS))
def test_full_output(runtime, cfg):
    spec = _spec(CONFIGS[cfg])
    for name, (a, b, (ref, abs_sum)) in _data(spec.m, spec.n, spec.k, spec.elem_fmt).items():
        outs = _run(runtime, spec, a, b)
        assert all(np.array_equal(outs[0], o) for o in outs[1:]), f"{cfg}/{name}: launches differ (race)"
        r = L.check_against_reference(outs[0], ref, abs_sum, exact=(name == "int"))
        assert r["pass"], f"{cfg}/{name}: {r}"


@pytest.mark.parametrize("mutant", ["scale_sel", "k_order", "drop_kstep", "wrong_fmt"])
def test_mutants_fail(runtime, mutant):
    spec = _spec({"mutant": mutant})
    a, b, (ref, abs_sum) = _data(spec.m, spec.n, spec.k, spec.elem_fmt)["gauss"]
    r = L.check_against_reference(_run(runtime, spec, a, b, launches=1)[0], ref, abs_sum)
    assert not r["pass"], f"mutant {mutant} passed"


@pytest.mark.parametrize("cfg", ["default", "two-wide-one-narrow", "e3m2", "no-release-a2"])
def test_edge_cases(runtime, cfg):
    from kernels.gfx950.mxfp6_gemm import edge_cases as E

    spec = _spec(CONFIGS[cfg])
    es = E.edge_operands(spec.m, spec.n, spec.k, fmt=spec.elem_fmt)
    ref32, abs_sum = E.reference_ieee(es.a, es.b)
    outs = _run(runtime, spec, es.a, es.b, launches=2)
    assert np.array_equal(outs[0], outs[1]), f"{cfg}: launches differ"
    r = E.check_ieee(outs[0], ref32, abs_sum, es.cases)
    assert r["bad"] == 0, f"{cfg}: {r}"
    rows, cols = es.cases["fp32_subnormal"]
    assert E.subnormal_behaviour(outs[0], ref32, rows, cols) == "keep"


# ----------------------------------------------------------------- stream-K (4096 x 4096 only; short K)

SK = {"m": 4096, "n": 4096, "k": 4096, "xcd_map": "band", "stream_k": True}


@pytest.mark.parametrize("over", [{}, {"head_rows": 2, "a_dtl": "spread"}, {"k_head": 12}])
def test_stream_k_full_output(runtime, over):
    """Heads, tails (two segments), narrow tiles and the partial handoff: int bitwise, Gaussian tolerance, 3 launches
    identical, flags reset after every launch."""
    spec = _spec({**SK, **over})
    for name in ("int", "gauss"):
        a, b, (ref, abs_sum) = _data(spec.m, spec.n, spec.k, spec.elem_fmt)[name]
        outs = _run(runtime, spec, a, b)
        assert all(np.array_equal(outs[0], o) for o in outs[1:]), f"{over}/{name}: launches differ"
        r = L.check_against_reference(outs[0], ref, abs_sum, exact=(name == "int"))
        assert r["pass"], f"{over}/{name}: {r}"


@pytest.mark.parametrize("mutant", ["k_order", "drop_kstep"])
def test_stream_k_mutants_fail(runtime, mutant):
    spec = _spec({**SK, "mutant": mutant})
    a, b, (ref, abs_sum) = _data(spec.m, spec.n, spec.k, spec.elem_fmt)["gauss"]
    r = L.check_against_reference(_run(runtime, spec, a, b, launches=1)[0], ref, abs_sum)
    assert not r["pass"], f"stream-K mutant {mutant} passed"
