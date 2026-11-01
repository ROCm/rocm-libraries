# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""f64 math ops on the device: every output equals the host model bit for bit.

``rocke.core.f64_math.evaluate`` runs the emitted instruction program in IEEE
f64 with an exact fma; ``tests/core/test_f64_math.py`` bounds that model against
a 200-bit reference. Here the device must reproduce the model exactly (the
programs use only correctly rounded operations: fma, add, mul, div, sqrt,
rint, ldexp, frexp), through both engines when the native one is built.

Any AMD GPU runs this (f64 is VALU on every target, not MFMA).
"""

from __future__ import annotations

import ctypes
import importlib.util
import math
import struct
import sys
from pathlib import Path

import numpy as np
import pytest
from rocke.core import f64_math as fm
from rocke.core.ir import F64, IRBuilder, PtrType
from rocke.core.ir_serialize import serialize
from rocke.core.lower_llvm import _lower_kernel_to_llvm_python
from rocke.runtime.hip_module import get_device_arch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core"))
from test_f64_math import SPECIALS, accuracy_samples  # noqa: E402

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


def _kernel(op: str):
    b = IRBuilder(f"f64_{op}")
    x_ptr = b.param("X", PtrType(F64, "global"), align=16)
    y_ptr = b.param("Y", PtrType(F64, "global"), align=16)
    i = b.add(b.mul(b.block_id_x(), b.const_i32(BLOCK)), b.thread_id_x())
    x = b.global_load(x_ptr, i, F64, align=8)
    b.global_store(y_ptr, i, getattr(b, op)(x), align=8)
    b.ret()
    b.kernel.attrs["max_workgroup_size"] = BLOCK
    return b.kernel


def _llvm(kernel, engine, arch):
    if engine == "python":
        return _lower_kernel_to_llvm_python(kernel, arch=arch)
    import rocke_engine as native

    return native.lower_serialized_ir(serialize(kernel), arch=arch)


def _run(ll, name, x, arch):
    from rocke.runtime.comgr import build_hsaco_from_llvm_ir
    from rocke.runtime.hip_module import Runtime

    blob, _ = build_hsaco_from_llvm_ir(ll, isa=f"amdgcn-amd-amdhsa--{arch}")
    rt = Runtime()
    module = rt.load_module(blob)
    n = x.size
    y = np.full(n, np.nan)
    ptrs = []
    try:
        for arr in (x, y):
            p = rt.alloc(arr.nbytes)
            ptrs.append(p)
            rt.memcpy_h2d(p, (ctypes.c_ubyte * arr.nbytes).from_buffer(arr), arr.nbytes)
        rt.launch_blocking(
            module.get_function(name),
            (n // BLOCK, 1, 1),
            (BLOCK, 1, 1),
            struct.pack("2Q", *ptrs),
        )
        rt.memcpy_d2h((ctypes.c_ubyte * y.nbytes).from_buffer(y), ptrs[1], y.nbytes)
        return y
    finally:
        for p in ptrs:
            rt.free(p)
        module.unload()


def _bits(v):
    return struct.unpack("<q", struct.pack("<d", v))[0]


@pytest.mark.parametrize("op", ["exp", "exp2", "log", "log2", "tanh", "sqrt", "rsqrt"])
def test_f64_math_matches_host_model(op):
    arch = _arch()
    name = f"math.{op}"
    xs = list(SPECIALS) + accuracy_samples(name, 600, seed=11)
    xs += [-x for x in xs[: len(SPECIALS)]]
    pad = (-len(xs)) % BLOCK
    xs += [1.0] * pad
    x = np.array(xs, dtype=np.float64)
    want = np.array([fm.evaluate(fm.PROGRAMS[name], float(v)) for v in xs])
    kernel = _kernel(op)
    for engine in _engines():
        got = _run(_llvm(kernel, engine, arch), kernel.name, x.copy(), arch)
        bad = [
            (xs[i], got[i], want[i])
            for i in range(len(xs))
            if not (
                (math.isnan(got[i]) and math.isnan(want[i]))
                or _bits(float(got[i])) == _bits(float(want[i]))
            )
        ]
        assert (
            not bad
        ), f"{engine} {op} on {arch}: {len(bad)} mismatches, e.g. {bad[:4]}"
