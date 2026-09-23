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
from rocke.core.ir import I8, I32, I64, IRBuilder, PtrType
from rocke.core.ir_serialize import serialize
from rocke.core.lower_llvm import lower_kernel_to_llvm
from rocke.core.lower_hip import lower_kernel_to_hip
from rocke.core.storage import (
    BitPacking,
    FragmentPacking,
    MatrixFragmentLayout,
    TensorStorage,
)
from rocke.helpers.mma_io import (
    load_matrix_fragment,
    pack_fragment_bits,
    storage_ir_type,
)


CASES = [
    "fp4",
    "fp6",
    "fp6_padded",
    "bf6",
    "fp8",
    "bf8",
    "f16",
    "bf16",
    "pack_fp6_cross_word",
    "pack_scale_bytes_i32",
    "pack_scale_bytes_i64",
]


def build_transport(dtype):
    b = IRBuilder("transport")
    padded = dtype == "fp6_padded"
    dtype = "fp6" if padded else dtype
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
        storage = TensorStorage(
            dtype,
            (16, 128),
            row_stride_bytes=97 if padded else None,
            alignment_bytes=16,
        )
        layout = (
            MatrixFragmentLayout(FragmentPacking(BitPacking(16), 32, 16, 32), 16, 2, 16)
            if typed
            else scaled_matrix_layout(dtype, 16)
        )
        base = b.const_i32(0)
        lane = b.mod(b.thread_id_x(), b.const_i32(32))
        group = b.div(lane, b.const_i32(16))
        value = load_matrix_fragment(
            b, a, base, group, 0, storage=storage, layout=layout, carrier_type=carrier
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


def test_reject_partial_fragment_and_wrong_pointer():
    b = IRBuilder("invalid_transport")
    ptr = b.param("A", PtrType(I8, "global"))
    zero = b.const_i32(0)
    with pytest.raises(ValueError, match="exceeds"):
        load_matrix_fragment(
            b,
            ptr,
            zero,
            zero,
            1,
            storage=TensorStorage("fp4", (16, 128)),
            layout=scaled_matrix_layout("fp4", 16),
        )
    with pytest.raises(ValueError, match="pointer storage type"):
        load_matrix_fragment(
            b,
            ptr,
            zero,
            zero,
            0,
            storage=TensorStorage("fp8", (16, 128)),
            layout=scaled_matrix_layout("fp8", 16),
        )


def test_hip_declares_only_encountered_missing_vector_widths():
    source = lower_kernel_to_hip(build_transport("fp6"), arch="gfx1250")
    for name in ("i8x24", "i8x40", "i32x12"):
        assert source.count(f"using {name} =") == 1
    assert "using f16x32 =" not in source
    source = lower_kernel_to_hip(build_transport("f16"), arch="gfx1250")
    assert source.count("using f16x32 =") == 1
    assert "using i8x24 =" not in source


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
