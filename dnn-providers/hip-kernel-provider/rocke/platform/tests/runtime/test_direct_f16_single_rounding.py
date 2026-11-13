# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""An f32 FMA or product converted to f16 is rounded once, as COMGR's code does.

LLVM selects ``fptrunc (fma a, b, c)`` and ``fptrunc (fmul a, b)`` to f16 as
``v_fma_mixlo_f16``, which rounds the exact result to f16.  ``v_fma_f32`` then
``v_cvt_f16_f32`` rounds it to f32 first, which lands on an f16 tie for about 1
exact result in 10^4 and rounds the wrong way for half of those.  The direct
compiler's code must agree with COMGR's, bit for bit, on those inputs: they are
what made 1 element in 10^5 of fp16 depthwise convolutions and unified-attention
reductions differ by one f16 ulp.

The expected bits are numpy's: the exact float64 result converted to f16 once.
The inputs are search results (the pairs where one rounding and two differ), so
a compiler that rounds twice fails on many elements, not by chance.  Needs a
gfx950 GPU and the direct compiler's library; numpy only.
"""

from __future__ import annotations

import ctypes

import numpy as np
import pytest
from rocke.core.ir import F16, F32, IRBuilder, PtrType
from rocke.helpers.compile import compile_kernel
from rocke.runtime import code_object

N = 2048
BLOCK = 64


def _gpu_arch() -> str:
    try:
        from rocke.runtime import get_device_info

        return get_device_info().base_arch
    except (ImportError, OSError, RuntimeError) as e:  # pragma: no cover
        pytest.skip(f"no visible GPU: {e}")


def _build(op: str):
    """``D[i] = f16(a[i] * b[i])`` or ``f16(fma(a[i], b[i], c[i]))``, one conversion and one store."""
    b = IRBuilder(f"f16_single_rounding_{op}")
    b.kernel.attrs["max_workgroup_size"] = BLOCK
    A = b.param("A", PtrType(F32, "global"), noalias=True, readonly=True, align=4)
    B = b.param("B", PtrType(F32, "global"), noalias=True, readonly=True, align=4)
    C = b.param("C", PtrType(F32, "global"), noalias=True, readonly=True, align=4)
    D = b.param("D", PtrType(F16, "global"), noalias=True, writeonly=True, align=2)
    i = b.add(b.mul(b.block_id_x(), b.const_i32(BLOCK)), b.thread_id_x())
    a = b.global_load(A, i, F32, align=4)
    bv = b.global_load(B, i, F32, align=4)
    if op == "fma":
        m = b.fma(a, bv, b.global_load(C, i, F32, align=4))
    else:
        m = b.fmul(a, bv)
    b.global_store(D, i, b.trunc_f32_to_f16(m), align=2)
    b.ret()
    return b.kernel


def _significand(rng: np.random.Generator, n: int, bits: int) -> np.ndarray:
    """Random f32 in [1, 2) with `bits` fraction bits and a random sign and a small exponent."""
    m = 1.0 + rng.integers(0, 1 << bits, n) / float(1 << bits)
    return (m * rng.choice([-1.0, 1.0], n) * 2.0 ** rng.integers(-4, 5, n)).astype(
        np.float32
    )


def _two_roundings_differ(exact: np.ndarray) -> np.ndarray:
    return np.float16(exact) != np.float16(np.float32(exact))


def _inputs(op: str):
    """N triples `(a, b, c)`: half of them where rounding the exact result to f32 and then f16 differs from rounding it once."""
    rng = np.random.default_rng(20261112)
    found = []
    while sum(len(f[0]) for f in found) < N // 2:
        a = _significand(rng, 1 << 22, 16)
        b = _significand(rng, 1 << 22, 16)
        c = np.zeros_like(a)
        if op == "fma":
            # the addend sits a few bits above the product's, so the sum needs more than 24 bits
            c = _significand(rng, 1 << 22, 12) * np.float32(2.0**-3)
        exact = a.astype(np.float64) * b.astype(np.float64) + c.astype(np.float64)
        hard = np.nonzero(_two_roundings_differ(exact))[0]
        found.append((a[hard], b[hard], c[hard], exact[hard]))
    hard = [np.concatenate(x)[: N // 2] for x in zip(*found)]
    # and as many ordinary ones: random operands, the first of the last draw
    easy = [x[: N // 2] for x in (a, b, c, exact)]
    return tuple(np.concatenate([h, e]) for h, e in zip(hard, easy))


def _launch(hsaco: bytes, name: str, a, b, c) -> np.ndarray:
    from rocke.runtime import KernelLauncher, LaunchConfig, Runtime

    rt = Runtime()
    out = np.zeros(N, dtype=np.uint16)

    def u8(x):
        return (ctypes.c_uint8 * x.nbytes).from_address(x.ctypes.data)

    host = [np.ascontiguousarray(x, dtype=np.float32) for x in (a, b, c)]
    dev = [rt.alloc(h.nbytes) for h in host] + [rt.alloc(out.nbytes)]
    for d, h in zip(dev, host):
        rt.memcpy_h2d(d, u8(h), h.nbytes)
    sig = [{"name": n, "type": "ptr<f32, global>", "size_bytes": 8} for n in "ABC"]
    sig.append({"name": "D", "type": "ptr<f16, global>", "size_bytes": 8})
    launcher = KernelLauncher(hsaco=hsaco, kernel_name=name, signature=sig)
    launcher(
        dict(zip("ABCD", dev)),
        config=LaunchConfig(grid=(N // BLOCK, 1, 1), block=(BLOCK, 1, 1)),
    )
    rt.sync()
    rt.memcpy_d2h(u8(out), dev[3], out.nbytes)
    for d in dev:
        rt.free(d)
    return out


@pytest.mark.parametrize("op", ["fma", "fmul"])
def test_an_f16_conversion_of_an_fma_or_product_rounds_once(op):
    if _gpu_arch() != "gfx950":
        pytest.skip("needs gfx950")
    if code_object._direct_library_path() is None:
        pytest.skip("build the Rust cdylib or set ROCKE_CODEOBJ_DIRECT_LIB")
    a, b, c, exact = _inputs(op)
    once = np.float16(exact).view(np.uint16)
    twice = np.float16(np.float32(exact)).view(np.uint16)
    assert np.count_nonzero(once != twice) >= N // 2

    kernel = _build(op)
    for backend in ("direct", "comgr"):
        art = compile_kernel(
            kernel, arch="gfx950", code_object_backend=backend, capture_ir_text=False
        )
        got = _launch(art.hsaco, art.kernel_name, a, b, c)
        bad = np.nonzero(got != once)[0]
        assert not len(bad), (
            f"{backend}: {len(bad)} of {N} f16 results are not the exact result "
            f"rounded once (first: a={a[bad[0]]!r} b={b[bad[0]]!r} c={c[bad[0]]!r} "
            f"got {got[bad[0]]:#06x}, once {once[bad[0]]:#06x}, twice {twice[bad[0]]:#06x})"
        )
