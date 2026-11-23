# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The fp6 and fp4 decode tables follow the OCP MX spec.

``unpack_fp6_bytes_to_quad_f32`` decodes the OCP MX E3M2 format: bit 5 is the
sign, bits 4..2 the exponent (bias 3), bits 1..0 the mantissa. Exponent field 0
is zero or subnormal, so code 0 is +0.0, code 32 is -0.0, and the other codes
with exponent field 0 are the subnormals ``2^(1-3) * (m / 4)``. Exponent field
1..7 is normal, ``2^(e-3) * (1 + m/4)``.

The tests compare every code against a decoder written here from the bit
fields with ``math.ldexp`` (independent of the helper's table), check the sign
of both zeros with ``math.copysign``, and run all 64 codes through the emitted
select chain on the GPU, comparing the f32 bits so that -0.0 is told apart from
+0.0. The fp4 (E2M1) table is checked against its spec decoder the same way.
"""

import ctypes
import importlib.util
import math
import struct

import numpy as np
import pytest

from rocke.core.ir import F32, I8, IRBuilder, PtrType
from rocke.core.lower_llvm import lower_kernel_to_llvm
from rocke.core.passes import optimize_kernel
from rocke.helpers import compile_kernel
from rocke.helpers.i4_dequant import (
    _FP4_CODEBOOK,
    _FP6_CODEBOOK,
    unpack_fp6_bytes_to_quad_f32,
)

_FP6_CODES = 64


def _spec_e3m2(code):
    """OCP MX E3M2 value of a 6-bit code, from its bit fields."""
    sign = -1.0 if code & 0b100000 else 1.0
    exponent = (code >> 2) & 0b111
    mantissa = code & 0b11
    if exponent == 0:
        magnitude = math.ldexp(mantissa, 1 - 3 - 2)
    else:
        magnitude = math.ldexp(4 + mantissa, exponent - 3 - 2)
    return math.copysign(magnitude, sign)


def _spec_e2m1(code):
    """OCP MX E2M1 value of a 4-bit code, from its bit fields."""
    sign = -1.0 if code & 0b1000 else 1.0
    exponent = (code >> 1) & 0b11
    mantissa = code & 0b1
    if exponent == 0:
        magnitude = math.ldexp(mantissa, 1 - 1 - 1)
    else:
        magnitude = math.ldexp(2 + mantissa, exponent - 1 - 1)
    return math.copysign(magnitude, sign)


def _f32_bits(value):
    return int(np.float32(value).view(np.uint32))


def test_fp6_table_matches_the_spec_for_every_code():
    assert len(_FP6_CODEBOOK) == _FP6_CODES
    for code in range(_FP6_CODES):
        want = _spec_e3m2(code)
        got = _FP6_CODEBOOK[code]
        assert got == want, (code, got, want)
        assert math.copysign(1.0, got) == math.copysign(1.0, want), code


def test_fp6_zeros_are_signed_zeros():
    assert _FP6_CODEBOOK[0] == 0.0 and math.copysign(1.0, _FP6_CODEBOOK[0]) == 1.0
    assert _FP6_CODEBOOK[32] == 0.0 and math.copysign(1.0, _FP6_CODEBOOK[32]) == -1.0


def test_fp6_subnormals_are_not_flushed():
    # Exponent field 0 with mantissa m is m / 16.
    assert _FP6_CODEBOOK[1:4] == (0.0625, 0.125, 0.1875)
    assert _FP6_CODEBOOK[33:36] == (-0.0625, -0.125, -0.1875)


def test_fp4_table_matches_the_spec_for_every_code():
    assert len(_FP4_CODEBOOK) == 16
    for code in range(16):
        want = _spec_e2m1(code)
        got = _FP4_CODEBOOK[code]
        assert got == want, (code, got, want)
        assert math.copysign(1.0, got) == math.copysign(1.0, want), code


def _decode_kernel():
    """Thread t decodes code t from the first two slots of a packed word.

    ``Codes`` holds two bytes per thread: the low and high byte of
    ``t | (t << 6)``, so slot 0 (bits 0..5) and slot 1 (bits 6..11) both hold
    code t. ``Out[2t]`` and ``Out[2t + 1]`` receive the two decoded values.
    """
    b = IRBuilder("fp6_decode")
    codes = b.param("Codes", PtrType(I8, "global"))
    out = b.param("Out", PtrType(F32, "global"))
    tid = b.thread_id_x()
    two_t = b.mul(tid, b.const_i32(2))
    two_t_plus_1 = b.add(two_t, b.const_i32(1))
    packed_lo = b.global_load_i8(codes, two_t)
    packed_hi = b.global_load_i8(codes, two_t_plus_1)
    v0, v1, _, _ = unpack_fp6_bytes_to_quad_f32(b, packed_lo, packed_hi)
    b.global_store(out, two_t, v0, align=4)
    b.global_store(out, two_t_plus_1, v1, align=4)
    b.ret()
    return b.kernel


def _packed_codes():
    words = [code | (code << 6) for code in range(_FP6_CODES)]
    return np.array(
        [byte for word in words for byte in (word & 0xFF, (word >> 8) & 0xFF)],
        dtype=np.uint8,
    )


@pytest.fixture(scope="module")
def gpu_arch():
    from rocke.runtime.hip_module import get_device_arch

    try:
        arch = get_device_arch(0)
    except Exception as exc:  # no HIP runtime on this host
        pytest.skip(f"needs a GPU: {exc}")
    if not arch:
        pytest.skip("needs a GPU")
    return arch


def _decode_on_gpu(arch, optimize_ir):
    """Run all 64 codes through the select chain; return the f32 results as bits."""
    from rocke.runtime.hip_module import Runtime

    # With the C++ engine present, backend="both" lowers through both engines
    # and requires identical LLVM IR.
    backend = "both" if importlib.util.find_spec("rocke_engine") else "python"
    artifact = compile_kernel(
        _decode_kernel(), arch=arch, backend=backend, optimize_ir=optimize_ir
    )
    codes = _packed_codes()
    out_bytes = 2 * _FP6_CODES * 4
    rt = Runtime()
    module = rt.load_module(artifact.hsaco)
    src = rt.alloc(codes.nbytes)
    dst = rt.alloc(out_bytes)
    try:
        rt.memcpy_h2d(
            src, (ctypes.c_uint8 * codes.nbytes).from_buffer_copy(codes), codes.nbytes
        )
        rt.launch(
            module.get_function(artifact.kernel_name),
            (1, 1, 1),
            (_FP6_CODES, 1, 1),
            struct.pack("<QQ", src, dst),
        )
        rt.sync()
        out = (ctypes.c_uint8 * out_bytes)()
        rt.memcpy_d2h(out, dst, out_bytes)
        return np.frombuffer(bytes(out), dtype=np.uint32).reshape(_FP6_CODES, 2)
    finally:
        rt.free(src)
        rt.free(dst)
        module.unload()


def test_optimized_decode_keeps_negative_zero():
    # Code 32 decodes to -0.0, a different constant from code 0's +0.0, so the
    # optimizer must not merge them (Python's 0.0 == -0.0).
    kernel = _decode_kernel()
    optimize_kernel(kernel)
    assert "0x8000000000000000" in lower_kernel_to_llvm(kernel, arch="gfx942")


# The optimizer runs only when asked; run the codes through both paths.
@pytest.mark.parametrize("optimize_ir", [False, True], ids=["plain", "optimized"])
def test_all_fp6_codes_decode_to_the_spec_value_on_gpu(gpu_arch, optimize_ir):
    got = _decode_on_gpu(gpu_arch, optimize_ir)
    for code in range(_FP6_CODES):
        want = _f32_bits(_spec_e3m2(code))
        for slot in range(2):
            assert int(got[code, slot]) == want, (code, slot, hex(int(got[code, slot])))
    # The sign of zero survives the GPU path: +0.0 and -0.0 as f32 bits.
    assert int(got[0, 0]) == 0x00000000
    assert int(got[32, 0]) == 0x80000000
