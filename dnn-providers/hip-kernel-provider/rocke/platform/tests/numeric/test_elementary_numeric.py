# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""f64 elementary functions (``rocke.helpers.elementary``) on the device.

f64: every output equals the host model bit for bit (``tests/helpers/test_elementary.py`` bounds
the model against a high-precision reference), through both engines when the native one is
built; the inputs include every exponent (each ``v_trig_preop_f64`` window position) and the
arguments closest to multiples of pi/2. f32 (evaluated in f64, rounded once): every output is the
exact value correctly rounded, or within 2^-20 ulp of a rounding boundary, and the specials
(NaN, infinities, signed zeros, subnormals, domain errors) are exact.

Any AMD GPU runs this (f64 is VALU on every target).
"""

from __future__ import annotations

import ctypes
import importlib.util
import math
import random
import struct
import sys
from pathlib import Path

import numpy as np
import pytest
from rocke.core.ir import F32, F64, I32, IRBuilder, PtrType
from rocke.core.ir_serialize import serialize
from rocke.core.lower_llvm import _lower_kernel_to_llvm_python
from rocke.helpers import elementary as el
from rocke.runtime.hip_module import get_device_arch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "helpers"))
from test_elementary import HARD_TRIG, samples  # noqa: E402

BLOCK = 256


def _arch():
    arch = (get_device_arch() or "").split(":")[0]
    if not arch:
        pytest.skip("requires an AMD GPU")
    return arch


def _engines():
    if importlib.util.find_spec("rocke_engine") is None:
        return ("python",)
    return ("python", "cpp")


def _kernel(name: str, ty):
    b = IRBuilder(f"el_{name}_{ty.name}")
    xp = b.param("X", PtrType(ty, "global"), align=16)
    yp = b.param("Y", PtrType(ty, "global"), align=16)
    n = b.param("n", I32)
    i = b.add(b.mul(b.block_id_x(), b.const_i32(BLOCK)), b.thread_id_x())
    nb = 8 if ty == F64 else 4
    with b.scf_if(b.cmp_lt(i, n)):
        x = b.global_load(xp, i, ty, align=nb)
        b.global_store(yp, i, el.emit(b, name, x), align=nb)
    b.ret()
    b.kernel.attrs["max_workgroup_size"] = BLOCK
    return b.kernel


def _llvm(kernel, engine, arch):
    if engine == "python":
        return _lower_kernel_to_llvm_python(kernel, arch=arch)
    import rocke_engine as native

    return native.lower_serialized_ir(serialize(kernel), arch=arch)


def _run(ll, name, x: np.ndarray, arch) -> np.ndarray:
    from rocke.runtime.comgr import build_hsaco_from_llvm_ir
    from rocke.runtime.hip_module import Runtime

    blob, _ = build_hsaco_from_llvm_ir(ll, isa=f"amdgcn-amd-amdhsa--{arch}")
    rt = Runtime()
    module = rt.load_module(blob)
    n = x.size
    y = np.zeros_like(x)
    ptrs = []
    try:
        for arr in (x, y):
            p = rt.alloc(arr.nbytes)
            ptrs.append(p)
            rt.memcpy_h2d(p, (ctypes.c_ubyte * arr.nbytes).from_buffer(arr), arr.nbytes)
        rt.launch_blocking(
            module.get_function(name),
            ((n + BLOCK - 1) // BLOCK, 1, 1),
            (BLOCK, 1, 1),
            struct.pack("2Qi4x", ptrs[0], ptrs[1], n),
        )
        rt.memcpy_d2h((ctypes.c_ubyte * y.nbytes).from_buffer(y), ptrs[1], y.nbytes)
        return y
    finally:
        for p in ptrs:
            rt.free(p)
        module.unload()


SPECIALS = [0.0, -0.0, math.inf, -math.inf, math.nan, 5e-324, -5e-324, 2.2250738585072014e-308]
SPECIALS += [1.0, -1.0, 0.5, -0.5, 2.0, 1.7976931348623157e308, -1.7976931348623157e308]


def _f64_inputs(name: str):
    rnd = random.Random(sum(map(ord, name)) + 7)
    xs = list(SPECIALS) + samples(name, 600, seed=sum(map(ord, name)) + 1)
    # every exponent (each trig_preop window position, every log / scale path)
    xs += [rnd.choice((-1.0, 1.0)) * math.ldexp(rnd.uniform(1, 2), e) for e in range(-1074, 1024)]
    if name in ("sin", "cos", "tan"):
        xs += list(HARD_TRIG) + [-x for x in HARD_TRIG]
    return xs


@pytest.mark.parametrize("name", el.FUNCTIONS)
def test_f64_matches_host_model(name):
    arch = _arch()
    xs = _f64_inputs(name)
    x = np.array(xs, dtype=np.float64)
    want = np.array([el.host(name, v) for v in xs])
    kernel = _kernel(name, F64)
    for engine in _engines():
        got = _run(_llvm(kernel, engine, arch), kernel.name, x.copy(), arch)
        gb, wb = got.view(np.int64), want.view(np.int64)
        bad = [
            (xs[i], got[i], want[i])
            for i in range(len(xs))
            if gb[i] != wb[i] and not (math.isnan(got[i]) and math.isnan(want[i]))
        ]
        assert not bad, f"{name} [{engine}]: {len(bad)} differ from the host model: {bad[:4]}"


def _f32(v: float) -> float:
    with np.errstate(over="ignore"):
        return float(np.float32(v))


def _f32_check(name: str, x: float, got: float):
    """None if ``got`` is ``name(x)`` correctly rounded to f32 (or within 2^-20 ulp of the
    boundary between two f32 values), else a description."""
    mp = pytest.importorskip("mpmath")
    if math.isnan(x):
        return None if math.isnan(got) else "NaN in, not NaN out"
    with mp.workprec(256):
        try:
            v = getattr(mp, name)(mp.mpf(x))
        except (ValueError, ZeroDivisionError):
            v = mp.nan
        if isinstance(v, mp.mpc) or mp.isnan(v):
            if name == "atanh" and abs(x) == 1:
                v = mp.inf if x > 0 else -mp.inf
            elif name == "acosh" and x == 1:
                v = mp.mpf(0)
            else:
                return None if math.isnan(got) else f"want NaN, got {got!r}"
        if mp.isinf(v):
            return None if got == float(v) else f"want {float(v)}, got {got!r}"
        if v == 0:
            sign = math.copysign(1.0, x) if name not in ("cos", "cosh", "acos", "acosh") else 1.0
            if name in ("floor", "ceil"):
                sign = math.copysign(1.0, x)
            want = math.copysign(0.0, sign)
            ok = got == 0 and math.copysign(1.0, got) == math.copysign(1.0, want)
            return None if ok else f"want {want!r}, got {got!r}"
        if abs(v) >= mp.mpf(2) ** 128 * (1 - mp.mpf(2) ** -25):
            return None if math.isinf(got) and (got > 0) == (v > 0) else f"want inf, got {got!r}"
        e = max(int(mp.floor(mp.log(abs(v), 2))) - 23, -149)
        err = abs(mp.mpf(got) - v) / mp.mpf(2) ** e
        if err <= mp.mpf(0.5) + mp.mpf(2) ** -20:
            return None
        return f"{float(err):.6f} ulp (want {_f32(float(v))!r}, got {got!r})"


def _f32_inputs(name: str):
    rnd = random.Random(sum(map(ord, name)) + 3)
    xs = [_f32(v) for v in SPECIALS[:7]]
    xs += [1.0, -1.0, 0.5, -0.5, 2.0, 3.4028234663852886e38, -3.4028234663852886e38]
    xs += [1e-45, -1e-45, 1.1754943508222875e-38, 1.17549421e-38]  # subnormals
    xs += [_f32(v) for v in samples(name, 800, seed=sum(map(ord, name)) + 2)]
    xs += [
        _f32(rnd.choice((-1.0, 1.0)) * math.ldexp(rnd.uniform(1, 2), e))
        for e in range(-149, 128)
    ]
    return [x for x in xs if math.isnan(x) or abs(x) < math.inf or x in (math.inf, -math.inf)]


@pytest.mark.parametrize("name", el.FUNCTIONS)
def test_f32_correctly_rounded(name):
    arch = _arch()
    xs = _f32_inputs(name)
    x = np.array(xs, dtype=np.float32)
    kernel = _kernel(name, F32)
    got = _run(_llvm(kernel, "python", arch), kernel.name, x.copy(), arch)
    bad = []
    for xi, gi in zip(x.tolist(), got.tolist()):
        why = _f32_check(name, xi, gi)
        if why:
            bad.append((xi, why))
    assert not bad, f"{name}: {len(bad)} bad f32 results: {bad[:4]}"
