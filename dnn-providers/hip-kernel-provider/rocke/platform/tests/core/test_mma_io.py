# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Mirrored authoring and bit-pattern placement, independent of MMA support."""

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from rocke.core.arch.wmma_scale import scaled_matrix_layout
from rocke.core.ir import F16, FP8E4M3, I8, I32, I64, IRBuilder, PtrType, VectorType
from rocke.core.ir_serialize import parse, serialize
from rocke.core.lower_llvm import lower_kernel_to_llvm
from rocke.core.lower_hip import lower_kernel_to_hip
from rocke.core.storage import (
    BitPacking,
    FragmentPacking,
    MatrixFragmentLayout,
)
from rocke.helpers.mma_io import (
    load_matrix_fragment,
    pack_fragment_bits,
    storage_ir_type,
)


CASES = [
    "load96_i8",
    "load96_f16",
    "load96_i32",
    "fp4",
    "fp6",
    "fp6_padded",
    "bf6",
    "fp8",
    "e4m3",
    "bf8",
    "f16",
    "bf16",
    "f16_padded",
    "bf16_padded",
    "pack_fp6_cross_word",
    "pack_scale_bytes_i32",
    "pack_scale_bytes_i64",
]


def build_transport(dtype):
    b = IRBuilder("transport")
    if dtype.startswith("load96_"):
        unit, n = {
            "load96_i8": (I8, 12),
            "load96_f16": (F16, 6),
            "load96_i32": (I32, 3),
        }[dtype]
        a = b.param("A", PtrType(unit, "global"))
        o = b.param("O", PtrType(unit, "global"))
        one = b.const_i32(1)
        value = b.global_load_vN(a, one, unit, n)
        smem = b.smem_alloc(unit, (n + 1,), name_hint="payload")
        for j in range(n):
            b.smem_store_vN(smem, [b.const_i32(j + 1)], b.vec_extract(value, j), 1)
        b.s_barrier_bare()
        value = b.smem_load_vN(smem, one, dtype=unit, n=n)
        aligned = b.global_load_vN(a, b.const_i32(0), unit, n, align=16)
        for k, vector in enumerate((value, aligned)):
            for j in range(n):
                b.global_store(
                    o, b.const_i32(k * n + j), b.vec_extract(vector, j), align=12 // n
                )
        return b.kernel
    padded = dtype.endswith("_padded")
    dtype = dtype.removesuffix("_padded")
    patterns = dtype.startswith("pack_")
    unit = I8 if patterns else storage_ir_type(dtype)
    typed = dtype in ("f16", "bf16")
    carrier = unit if typed else I64 if dtype == "pack_scale_bytes_i64" else I32
    a = b.param("A", PtrType(unit, "global"))
    o = b.param("O", PtrType(carrier, "global"))
    if patterns:
        bits = 6 if dtype == "pack_fp6_cross_word" else 8
        count = 16 if bits == 6 else 8 if dtype == "pack_scale_bytes_i64" else 4
        words = 3 if bits == 6 else 1
        fragment = FragmentPacking(
            BitPacking(bits), count, 64 if carrier == I64 else 32, words
        )
        values = pack_fragment_bits(
            b, lambda j: b.global_load(a, b.const_i32(j), I8, align=1), fragment
        )
        for j, value in enumerate(values):
            b.global_store(o, b.const_i32(j), value, align=4)
    else:
        layout = (
            MatrixFragmentLayout(FragmentPacking(BitPacking(16), 32, 16, 32), 16, 2, 16)
            if typed
            else scaled_matrix_layout(dtype, 16)
        )
        base = b.const_i32((129 if typed else 97) if padded else 0)
        lane = b.mod(b.thread_id_x(), b.const_i32(32))
        group = b.div(lane, b.const_i32(16))
        value = load_matrix_fragment(
            b,
            a,
            base,
            group,
            0,
            dtype=dtype,
            layout=layout,
            carrier_type=carrier,
            alignment_bytes=(2 if typed else 1) if padded else 16,
        )
        for j in range(value.type.count):
            index = b.const_i32(j)
            element = b.vec_extract(value, j)
            b.global_store(o, index, element, align=2 if typed else 4)
    return b.kernel


@pytest.mark.parametrize("dtype", CASES)
def test_descriptor_authoring_lowers_in_both_backends(dtype):
    kernel = build_transport(dtype)
    assert "define" in lower_kernel_to_llvm(
        kernel, arch="gfx1250", llvm_flavor="llvm23"
    )
    assert "transport" in lower_kernel_to_hip(kernel, arch="gfx1250")


@pytest.mark.parametrize("dtype", CASES)
@pytest.mark.parametrize("route", ["ir", "hip"])
def test_native_helper_emits_identical_ir(dtype, route):
    executable = os.environ.get("ROCKE_STORAGE_TEST")
    if not executable:
        pytest.skip("set ROCKE_STORAGE_TEST to the built native storage test")
    assert Path(executable).is_file()
    actual = subprocess.run(
        [executable, "--hip" if route == "hip" else "--emit", dtype],
        check=True,
        capture_output=True,
        text=True,
    )
    kernel = build_transport(dtype)
    expected = (
        serialize(kernel)
        if route == "ir"
        else lower_kernel_to_hip(kernel, arch="gfx1250")
    )
    assert actual.stdout == expected


def test_reject_bit_origin():
    b = IRBuilder("invalid_transport")
    ptr = b.param("A", PtrType(I8, "global"))
    zero = b.const_i32(0)
    with pytest.raises(ValueError, match="aligned to pointer storage units"):
        load_matrix_fragment(
            b,
            ptr,
            zero,
            zero,
            1,
            dtype="fp4",
            layout=scaled_matrix_layout("fp4", 16),
        )


@pytest.mark.parametrize(
    "dtype,alignment,error",
    [("fp6", 3, "positive power of two"), ("fp4", 1, "packing width mismatch")],
)
def test_fragment_input_contract(dtype, alignment, error):
    b = IRBuilder("invalid_fragment")
    ptr = b.param("A", PtrType(I8, "global"))
    zero = b.const_i32(0)
    before = serialize(b.kernel)
    with pytest.raises(ValueError, match=error):
        load_matrix_fragment(
            b,
            ptr,
            zero,
            zero,
            0,
            dtype=dtype,
            layout=scaled_matrix_layout("fp6", 16),
            alignment_bytes=alignment,
        )
    assert serialize(b.kernel) == before


@pytest.mark.parametrize("dtype", ["e8m0", "fp6", "f16"])
@pytest.mark.parametrize(
    "count,groups,k0,valid",
    [
        (64, 100_000_000, 0, False),
        (64, 1 << 25, 0, True),
        (64, 1 << 25, 1, False),
        (64, 2, (1 << 31) - 128, True),
        (64, 2, (1 << 31) - 127, False),
        (16, 100_000_000, 1_500_000_000, False),
        (64, 0x7FFFFFFF, 0, False),
    ],
)
def test_fragment_offset_bounds(dtype, count, groups, k0, valid):
    # Padded FP6 and E8M0 use byte units; FP16 uses two-byte pointer units.
    bits = 16 if dtype == "f16" else 6 if dtype == "fp6" else 8
    packing = BitPacking(bits, 16 if dtype == "f16" else 8)
    layout = MatrixFragmentLayout(
        FragmentPacking(packing, count, 32, count * packing.slot_bits // 32),
        16,
        groups,
        1,
    )
    b = IRBuilder("offset_bounds")
    ptr = b.param("A", PtrType(storage_ir_type(dtype), "global"))
    zero = b.const_i32(0)
    before = serialize(b.kernel)
    if valid:
        assert (
            load_matrix_fragment(b, ptr, zero, zero, k0, dtype=dtype, layout=layout)
            is not None
        )
    else:
        with pytest.raises(ValueError, match="offset exceeds i32 range"):
            load_matrix_fragment(b, ptr, zero, zero, k0, dtype=dtype, layout=layout)
        assert serialize(b.kernel) == before


def test_hip_declares_only_encountered_missing_vector_widths():
    source = lower_kernel_to_hip(build_transport("fp6"), arch="gfx1250")
    for name in ("i8x24", "i8x40", "i32x12"):
        assert source.count(f"using {name} =") == 1
    assert "using f16x32 =" not in source
    source = lower_kernel_to_hip(build_transport("f16"), arch="gfx1250")
    assert source.count("using f16x32 =") == 1
    assert "using i8x24 =" not in source


@pytest.mark.parametrize("dtype", ["fp8", "bf8"])
def test_byte_width_does_not_erase_pointer_identity(dtype):
    assert storage_ir_type(dtype) != I8
    b = IRBuilder("nominal_pointer")
    ptr = b.param("A", PtrType(I8, "global"))
    zero = b.const_i32(0)
    with pytest.raises(ValueError, match="pointer storage type mismatch"):
        load_matrix_fragment(
            b,
            ptr,
            zero,
            zero,
            0,
            dtype=dtype,
            layout=scaled_matrix_layout(dtype, 16),
        )


@pytest.mark.parametrize("elem_type", ["unknown", "fp4e2m1"])
def test_hip_rejects_unsupported_vector_element(elem_type):
    b = IRBuilder("invalid_element")
    ptr = b.param("A", PtrType(I8, "global"))
    value = b.global_load_vN(ptr, b.const_i32(0), I8, 16)
    value.op.attrs["elem_type"] = elem_type
    with pytest.raises(NotImplementedError, match="unsupported element type"):
        lower_kernel_to_hip(b.kernel, arch="gfx1250")


@pytest.mark.parametrize(
    "dtype,copy_bytes,alignment",
    [("fp6", 16, 8), ("fp6_padded", 16, 1), ("f16", 32, 16)],
)
def test_hip_compiler_preserves_fragment_load_alignment(dtype, copy_bytes, alignment):
    """Inspect compiler IR: source parity cannot detect a shared alignment bug."""
    if not shutil.which("hipcc"):
        pytest.skip("hipcc not in PATH")
    from rocke.helpers.compile import emit_device_llvm_ir_via_hipcc

    llvm = emit_device_llvm_ir_via_hipcc(
        build_transport(dtype), arch="gfx950", extra_flags=["-O0"]
    )
    source_alignments = [
        int(align)
        for align, size in re.findall(
            r"@llvm\.memcpy[^\n]*?\([^,]+,\s*ptr[^,]*\balign (\d+) [^,]+,\s*i64 (\d+)",
            llvm,
        )
        if int(size) == copy_bytes
    ]
    assert source_alignments, "expected unaligned-safe fragment payload copies"
    assert all(value <= alignment for value in source_alignments), source_alignments


@pytest.mark.parametrize("alignment", [0, -1, -16, 3, 24])
def test_hip_rejects_invalid_vector_load_alignment(alignment):
    b = IRBuilder("invalid_alignment")
    ptr = b.param("A", PtrType(I8, "global"))
    value = b.global_load_vN(ptr, b.const_i32(0), I8, 16)
    # Serialized/custom IR can bypass the builder's default alignment handling.
    value.op.attrs["align"] = alignment
    with pytest.raises(ValueError, match="alignment must be a positive power of two"):
        lower_kernel_to_hip(b.kernel, arch="gfx1250")


def test_e4m3_alias_serialization_matches_native():
    executable = os.environ.get("ROCKE_STORAGE_TEST")
    if not executable:
        pytest.skip("set ROCKE_STORAGE_TEST to the built native storage test")
    b = IRBuilder("canonical_e4m3")
    b.param("scalar", FP8E4M3)
    b.param("pointer", PtrType(FP8E4M3, "global"))
    b.param("vector", VectorType(FP8E4M3, 4))
    canonical = serialize(b.kernel)
    alias = canonical.replace("fp8e4m3", "e4m3")
    assert serialize(parse(alias)) == canonical
    result = subprocess.run(
        [executable, "--parse", alias], check=True, capture_output=True, text=True
    )
    assert result.stdout == canonical


def test_quant_type_diagnostic_matches_native():
    from rocke.helpers.quant import quant_ir_type

    executable = os.environ.get("ROCKE_STORAGE_TEST")
    if not executable:
        pytest.skip("set ROCKE_STORAGE_TEST to the built native storage test")
    with pytest.raises(ValueError) as error:
        quant_ir_type("invalid")
    result = subprocess.run(
        [executable, "--quant-error", "invalid"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout == str(error.value)


def test_reject_wide_encoded_fields_before_emission():
    b = IRBuilder("unsupported_packing")
    fragment = FragmentPacking(BitPacking(40), 2, 64, 2)
    before = len(b.kernel.body.ops)
    with pytest.raises(ValueError, match="encoded fields of at most 32 bits"):
        pack_fragment_bits(b, lambda j: b.const_i64(1 << 35), fragment)
    assert len(b.kernel.body.ops) == before


@pytest.mark.parametrize("dtype", ["load96_i8", "load96_f16", "load96_i32"])
def test_96bit_loads_copy_exact_payload(dtype):
    """Offset and over-aligned global loads plus an LDS tail must not read padding."""
    kernel = build_transport(dtype)
    llvm = lower_kernel_to_llvm(kernel, arch="gfx1250", llvm_flavor="llvm23")
    n, elem, alignment = {
        "load96_i8": (12, "i8", 1),
        "load96_f16": (6, "half", 2),
        "load96_i32": (3, "i32", 4),
    }[dtype]
    for space in (1, 3):
        assert re.search(
            rf"load <{n} x {elem}>, ptr addrspace\({space}\).*align {alignment}\b", llvm
        )
    hip = lower_kernel_to_hip(kernel, arch="gfx1250")
    assert len(re.findall(r"__builtin_memcpy\([^\n]*, 12\);", hip)) == 3
    if not shutil.which("hipcc"):
        pytest.skip("hipcc not in PATH")
    from rocke.helpers.compile import emit_device_llvm_ir_via_hipcc

    compiled = emit_device_llvm_ir_via_hipcc(kernel, arch="gfx950", extra_flags=["-O0"])
    # Clang's vector object is 16 bytes; the source copy must remain 12 bytes.
    compiled = compiled.split("define protected amdgpu_kernel void @transport(", 1)[
        1
    ].split("\n}", 1)[0]
    source_copies = re.findall(
        r"@llvm\.memcpy[^\n]*?\([^,]+,\s*ptr[^,]*\balign (\d+) [^,]+,\s*i64 (\d+)",
        compiled,
    )
    assert len(source_copies) == 3
    for (actual_align, size), max_align in zip(
        source_copies, (alignment, alignment, 16)
    ):
        assert int(size) == 12
        assert int(actual_align) <= max_align
