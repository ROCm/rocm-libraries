# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The E8M0 scale decoder follows the OCP MX spec.

An E8M0 scale byte is an unsigned biased exponent (bias 127, no sign, no
mantissa). Codes 0..254 decode to exactly ``2^(code - 127)``; code 0 is
``2^-127``, an f32 subnormal, not zero. Code 255 decodes to NaN.

The tests below check that contract four ways: every one of the 256 codes run
through the decoder on the GPU against an independent reference; the decoder's
lowered text (it zero-extends the byte, never sign-extends it, so the signed
int-to-float conversion that follows sees 0..255) in LLVM IR and in HIP; the two engines lowering the
decoder to the same LLVM IR; and the MX GEMM reading its scale bytes the same
unsigned way in both engines. They also check the scaled-dot helper (its
refusals and its ldexp form) and the largest code the scaled MFMA takes
directly. The MX GEMM's numerics across the scale range are checked in
tests/instances/test_mx_gemm_scale_range.py.
"""

import ctypes
import importlib.util
import math
import re
import struct

import numpy as np
import pytest

from rocke.core.backend import lower_mx_gemm
from rocke.core.ir import F32, I8, IRBuilder, PtrType
from rocke.core.ir_serialize import serialize
from rocke.core.lower_hip import lower_kernel_to_hip
from rocke.core.lower_llvm import LLVM_FLAVORS, _lower_kernel_to_llvm_python
from rocke.helpers import compile_kernel
from rocke.helpers.mx_scale import (
    BF8_E5M2_FNUZ,
    BF8_E5M2_OCP,
    FP8_E4M3_FNUZ,
    FP8_E4M3_OCP,
    add_mx_scaled_dot,
    decode_mx_scale_e8m0,
    largest_safe_mx_scale_code,
)
from rocke.instances.common.mx_gemm import MxGemmSpec, build_mx_gemm

_CODES = 256

# The decoder's lines in the kernel built by ``_decode_kernel``. The C++
# engine's test (test_mx_scale_decode.cpp) pins the same LLVM lines.
_EXPECTED_LLVM = (
    "  %zx3 = zext i8 %gl2 to i32\n"
    "  %eq5 = icmp eq i32 %zx3, 255\n"
    "  %sitof6 = sitofp i32 %zx3 to float\n"
    "  %fsub8 = fsub float %sitof6, 0x405FC00000000000\n"
    "  %exp29 = call float @llvm.exp2.f32(float %fsub8)\n"
    "  %sel11 = select i1 %eq5, float 0x7FF8000000000000, float %exp29\n"
)

_EXPECTED_HIP = (
    " int zx3 = (int)(uint8_t)gl2;\n"
    " int c4 = 255;\n"
    " bool eq5 = zx3 == c4;\n"
    " float sitof6 = (float)zx3;\n"
    " float c7 = 127.0f;\n"
    " float fsub8 = sitof6 - c7;\n"
    " float exp29 = exp2f(fsub8);\n"
    " float c10 = ((float)NAN);\n"
    " float sel11 = eq5 ? c10 : exp29;\n"
)


def _has_cpp_engine():
    return importlib.util.find_spec("rocke_engine") is not None


def _decode_kernel():
    """``Out[tid] = decode_mx_scale_e8m0(Codes[tid])``, one thread per code."""
    b = IRBuilder("mx_scale_decode")
    codes = b.param("Codes", PtrType(I8, "global"))
    out = b.param("Out", PtrType(F32, "global"))
    tid = b.thread_id_x()
    code = b.global_load_i8(codes, tid)
    b.global_store(out, tid, decode_mx_scale_e8m0(b, code), align=4)
    b.ret()
    return b.kernel


def _reference_bits(code):
    """The spec value of one code as f32 bits, computed without the decoder."""
    if code == 255:
        return None  # NaN; any NaN payload is accepted
    value = np.float32(math.ldexp(1.0, code - 127))
    assert float(value) == math.ldexp(1.0, code - 127)  # exact in f32
    return int(value.view(np.uint32))


def _mx_gemm_spec():
    return MxGemmSpec(
        M=16, N=16, K=64, block_tile_m=32, block_tile_n=128, mantissa_dtype="fp8e4m3"
    )


def test_decoder_llvm_reads_the_byte_unsigned():
    ll = _lower_kernel_to_llvm_python(
        _decode_kernel(), arch="gfx90a", llvm_flavor="llvm20"
    )
    assert _EXPECTED_LLVM in ll
    assert "sext" not in ll


def test_decoder_hip_reads_the_byte_unsigned():
    hip = lower_kernel_to_hip(_decode_kernel(), arch="gfx90a", include_prologue=False)
    assert _EXPECTED_HIP in hip


@pytest.mark.parametrize("flavor", LLVM_FLAVORS)
def test_engines_lower_the_decoder_identically(flavor):
    if not _has_cpp_engine():
        pytest.skip("needs the rocke_engine binding")
    import rocke_engine

    kernel = _decode_kernel()
    python_ll = _lower_kernel_to_llvm_python(kernel, arch="gfx90a", llvm_flavor=flavor)
    cpp_ll = rocke_engine.lower_serialized_ir(
        serialize(kernel), arch="gfx90a", flavor=flavor
    )
    assert cpp_ll == python_ll


def test_non_byte_input_is_rejected():
    b = IRBuilder("mx_scale_decode_i32")
    with pytest.raises(
        ValueError, match="decode_mx_scale_e8m0 expects i8 input, got i32"
    ):
        decode_mx_scale_e8m0(b, b.const_i32(0))


def test_mx_gemm_reads_the_scale_bytes_unsigned_in_both_engines():
    # The C++ MX GEMM builder must emit the same LLVM IR as the Python one;
    # backend="both" lowers through each and fails on any byte difference.
    if not _has_cpp_engine():
        pytest.skip("needs the rocke_engine binding")
    ll = lower_mx_gemm(_mx_gemm_spec(), arch="gfx950", backend="both").llvm_text
    assert ll.count("zext i8") >= 2  # the A and the B scale bytes
    assert "sext i8" not in ll
    assert ll.count("icmp eq i32") >= 2  # each byte checked for the NaN code
    assert "float 0x7FF8000000000000" in ll


@pytest.mark.parametrize(
    "case, message",
    [
        ("scalar dot", "add_mx_scaled_dot expects an f32 vector dot, got f32"),
        (
            "mismatched accumulator",
            (
                "add_mx_scaled_dot expects the accumulator type to match the dot type "
                "vec<f32x4>, got vec<f32x2>"
            ),
        ),
        (
            "wrong code count",
            "add_mx_scaled_dot expects 4 A and 4 B scale bytes, got 3 and 4",
        ),
        ("i32 scale code", "add_mx_scaled_dot expects i8 input, got i32"),
    ],
)
def test_scaled_dot_refuses_wrong_types(case, message):
    """The C++ engine's test (test_mx_scale_decode.cpp) checks the same texts."""
    b = IRBuilder("mx_scaled_dot_refusal")
    dot = b.vector_splat(b.const_f32(1.0), 4)
    acc = dot
    codes = b.param("Codes", PtrType(I8, "global"))
    code = b.global_load_i8(codes, b.const_i32(0))
    a_codes, b_codes = [code] * 4, [code] * 4
    if case == "scalar dot":
        dot = b.const_f32(1.0)
    elif case == "mismatched accumulator":
        acc = b.vector_splat(b.const_f32(1.0), 2)
    elif case == "wrong code count":
        a_codes = a_codes[:3]
    elif case == "i32 scale code":
        b_codes = [b.const_i32(0)] * 4
    with pytest.raises(ValueError, match=re.escape(message)):
        add_mx_scaled_dot(b, acc, dot, a_codes, b_codes)


def test_scaled_dot_scales_with_ldexp_and_selects_nan():
    """Each element is ``acc + select(nan, NaN, ldexp(dot, ca + cb - 254))``;
    a byte shared by several elements is read once."""
    b = IRBuilder("mx_scaled_dot_shape")
    codes = b.param("Codes", PtrType(I8, "global"))
    out = b.param("Out", PtrType(F32, "global"))
    a_code = b.global_load_i8(codes, b.const_i32(0))
    b_code = b.global_load_i8(codes, b.const_i32(1))
    dot = b.vector_splat(b.const_f32(3.0), 2)
    acc = b.vector_splat(b.const_f32(1.0), 2)
    total = add_mx_scaled_dot(b, acc, dot, [a_code, a_code], [b_code, b_code])
    b.global_store(out, b.const_i32(0), b.vec_extract(total, 0), align=4)
    b.ret()
    ll = _lower_kernel_to_llvm_python(b.kernel, arch="gfx950", llvm_flavor="llvm20")
    assert ll.count("zext i8") == 2  # each distinct byte read once
    assert ll.count("call float @llvm.ldexp.f32.i32(") == 2
    assert ll.count("float 0x7FF8000000000000") >= 1
    assert "fmul" not in ll


# The largest code each fp8 / bf8 encoding the matrix cores read may take on
# both sides of the scaled MFMA (64 products per output): 64 * largest^2 is
# below 2^24 for OCP E4M3, 2^22 for E4M3 FNUZ and 2^38 for both E5M2
# encodings, and 2c - 254 + that power is at most 127.
_SAFE_CODES = (
    (FP8_E4M3_OCP, 178),
    (BF8_E5M2_OCP, 171),
    (FP8_E4M3_FNUZ, 179),
    (BF8_E5M2_FNUZ, 171),
)


@pytest.mark.parametrize(
    "element_format, code", _SAFE_CODES, ids=[f.name for f, _ in _SAFE_CODES]
)
def test_safe_code_for_the_mfma_formats(element_format, code):
    """The C++ engine's test (test_mx_scale_decode.cpp) pins the same codes."""
    assert largest_safe_mx_scale_code(element_format, element_format, 64) == code


@pytest.mark.parametrize(
    "element_format, code", _SAFE_CODES, ids=[f.name for f, _ in _SAFE_CODES]
)
def test_safe_code_keeps_the_largest_sum_a_binade_below_overflow(element_format, code):
    """At the code, the largest sum of 64 scaled products is at most 2^127;
    one code higher it is not."""
    largest_sum = 64 * element_format.largest_value**2
    assert largest_sum * 2.0 ** (2 * code - 254) <= 2.0**127
    assert largest_sum * 2.0 ** (2 * (code + 1) - 254) > 2.0**127


def test_safe_code_refuses_a_count_that_is_not_positive():
    with pytest.raises(
        ValueError,
        match=re.escape("largest_safe_mx_scale_code: products must be positive, got 0"),
    ):
        largest_safe_mx_scale_code(FP8_E4M3_OCP, FP8_E4M3_OCP, 0)


def _skip_unless_comgr_targets(isa: str) -> None:
    """Skip when COMGR cannot be loaded or cannot target ``isa``.

    Only a missing or unloadable COMGR, or one that does not list ``isa``,
    skips; a compile error after this check fails the test.
    """
    try:
        from rocke.runtime import comgr
    except ImportError as exc:  # environment-dependent
        pytest.skip(f"comgr unavailable: {exc}")
    try:
        isas = comgr.supported_isas()
    except comgr.ComgrError as exc:  # environment-dependent
        pytest.skip(f"comgr unavailable: {exc}")
    if isa not in isas:
        pytest.skip(f"the loaded comgr cannot target {isa}")


def test_mx_gemm_compiles_for_gfx950():
    # Compile-checked only: MX GEMM numerics need a gfx950 GPU.
    _skip_unless_comgr_targets("amdgcn-amd-amdhsa--gfx950")
    backend = "both" if _has_cpp_engine() else "python"
    artifact = compile_kernel(
        build_mx_gemm(_mx_gemm_spec(), arch="gfx950"), arch="gfx950", backend=backend
    )
    assert artifact.hsaco


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


def _decode_on_gpu(arch):
    """Run all 256 codes through the decoder; return the f32 results as bits."""
    from rocke.runtime.hip_module import Runtime

    # With the C++ engine present, backend="both" lowers through both engines
    # and requires identical LLVM IR.
    backend = "both" if _has_cpp_engine() else "python"
    artifact = compile_kernel(_decode_kernel(), arch=arch, backend=backend)
    codes = np.arange(_CODES, dtype=np.uint8)
    rt = Runtime()
    module = rt.load_module(artifact.hsaco)
    src = rt.alloc(codes.nbytes)
    dst = rt.alloc(_CODES * 4)
    try:
        rt.memcpy_h2d(
            src, (ctypes.c_uint8 * codes.nbytes).from_buffer_copy(codes), codes.nbytes
        )
        rt.launch(
            module.get_function(artifact.kernel_name),
            (1, 1, 1),
            (_CODES, 1, 1),
            struct.pack("<QQ", src, dst),
        )
        rt.sync()
        out = (ctypes.c_uint8 * (_CODES * 4))()
        rt.memcpy_d2h(out, dst, _CODES * 4)
        return np.frombuffer(bytes(out), dtype=np.uint32)
    finally:
        rt.free(src)
        rt.free(dst)
        module.unload()


def test_all_codes_decode_to_the_spec_value_on_gpu(gpu_arch):
    got = _decode_on_gpu(gpu_arch)
    for code in range(_CODES):
        want = _reference_bits(code)
        if want is None:
            assert math.isnan(got[code : code + 1].view(np.float32)[0]), code
        else:
            assert int(got[code]) == want, (code, hex(int(got[code])), hex(want))

    # Recorded GPU behavior for code 0: the decoder keeps the f32 subnormal
    # 2^-127 (bits 0x00400000); the GPU does not flush it to zero.
    assert int(got[0]) == 0x00400000
