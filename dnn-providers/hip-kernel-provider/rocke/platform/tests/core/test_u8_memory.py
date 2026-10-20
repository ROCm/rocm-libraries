# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""U8 in memory, and the element list the LDS vector builders admit.

A u8 element takes one byte, like i8, in both engines' LLVM and HIP lowerings.
``smem_load_vN`` and ``smem_store_vN`` admit one shared element list, which
leaves u8 out, so u8 vectors are refused by the LDS pair as by the global pair.
The C++ builders are reached through the ``rocke_ir_types`` test binary
(``ROCKE_IR_TYPES_TEST``), because the serialized-IR seam lowers kernels the
Python builder built.
"""

import os
import re
import shutil
import subprocess

import pytest
from rocke.core.ir import (
    I8,
    U8,
    IRBuilder,
    KernelDef,
    PtrType,
    Type,
    dtype_to_ir_type,
)
from rocke.core.lower_hip import lower_kernel_to_hip
from rocke.tests.engines import lower_hip_with_both_engines, lower_with_both_engines
from rocke.tests.ir_validity import assert_target_accepts

_ADMITTED = "f16 / bf16 / i16 / f32 / i32 / tf32 / f64 / fp8e4m3 / bf8e5m2 / i8"


def _byte_copy_kernel(dtype: Type) -> KernelDef:
    """Copy one ``dtype`` element per thread from SRC to DST."""
    b = IRBuilder("byte_copy")
    src = b.param("SRC", PtrType(dtype, "global"))
    dst = b.param("DST", PtrType(dtype, "global"))
    tid = b.thread_id_x()
    b.global_store(dst, tid, b.global_load(src, tid, dtype))
    return b.kernel


def _lds_byte_store_kernel(dtype: Type) -> KernelDef:
    """Store one ``dtype`` element per thread from SRC into a 64-element LDS tile."""
    b = IRBuilder("lds_byte_store")
    src = b.param("SRC", PtrType(dtype, "global"))
    tid = b.thread_id_x()
    tile = b.smem_alloc(dtype, [64], name_hint="tile")
    b.smem_store_vN(tile, [tid], b.global_load(src, tid, dtype), 1)
    return b.kernel


def _python_smem_vector_access(direction: str, elem: str, n: int) -> str:
    """Build one LDS vector access with the Python builder; ``ok`` or ``error: <text>``.

    The twin of ``rocke_ir_types --smem-vector``: an LDS tile of ``elem``, then
    ``smem_load_vN``, or ``smem_store_vN`` of ``vector_splat(<loaded elem>, n)``.
    A refusal must leave the kernel without a new op.
    """
    dtype = dtype_to_ir_type(elem)
    b = IRBuilder("smem_vector_access")
    tile = b.smem_alloc(dtype, [64], name_hint="tile")
    zero = b.const_i32(0)
    value = None
    if direction == "store":
        src = b.param("SRC", PtrType(dtype, "global"))
        value = b.vector_splat(b.global_load(src, zero, dtype), n)
    ops_before = len(b.kernel.body.ops)
    try:
        if direction == "store":
            b.smem_store_vN(tile, [zero], value, n)
        else:
            b.smem_load_vN(tile, zero, dtype=dtype, n=n)
    except ValueError as error:
        assert len(b.kernel.body.ops) == ops_before
        return f"error: {error}"
    return "ok"


def test_u8_lds_store_is_laid_out_like_i8():
    # A u8 LDS element is one byte, so the tile and the store match the i8 kernel.
    u8_ir = lower_with_both_engines(_lds_byte_store_kernel(U8), "gfx90a")
    assert u8_ir == lower_with_both_engines(_lds_byte_store_kernel(I8), "gfx90a")
    assert "[64 x i8]" in u8_ir
    assert re.search(r"store i8 \S+, ptr addrspace\(3\) \S+, align 1\n", u8_ir)
    assert_target_accepts(u8_ir, "gfx90a")
    hip = lower_hip_with_both_engines(_lds_byte_store_kernel(U8), "gfx90a")
    assert "__shared__ uint8_t" in hip


@pytest.mark.parametrize("direction", ["load", "store"])
def test_lds_vector_access_refuses_u8(direction):
    assert _python_smem_vector_access(direction, "u8", 4) == (
        f"error: smem_{direction}_vN supports {_ADMITTED}, got u8"
    )
    assert _python_smem_vector_access(direction, "i8", 16) == "ok"


@pytest.mark.parametrize("direction", ["load", "store"])
@pytest.mark.parametrize("elem", ["u8", "i8", "i16"])
@pytest.mark.parametrize("n", [4, 16])
def test_native_lds_vector_access_matches_the_python_builder(direction, elem, n):
    executable = os.environ.get("ROCKE_IR_TYPES_TEST")
    if not executable:
        pytest.skip("set ROCKE_IR_TYPES_TEST to the built rocke_ir_types test")
    native = subprocess.run(
        [executable, "--smem-vector", direction, elem, str(n)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.rstrip("\n")
    python = _python_smem_vector_access(direction, elem, n)
    if elem == "u8":
        # The element refusal is one text in both builders.
        assert native == python
    else:
        # The width refusals word the allowed widths differently in the two
        # builders; compare the outcome only.
        assert native.split(":")[0] == python.split(":")[0]


@pytest.mark.parametrize("kernel_builder", [_byte_copy_kernel, _lds_byte_store_kernel])
def test_u8_hip_source_compiles(kernel_builder, tmp_path):
    if not shutil.which("hipcc"):
        pytest.skip("hipcc not in PATH")
    path = tmp_path / "kernel.hip"
    path.write_text(lower_kernel_to_hip(kernel_builder(U8), arch="gfx90a"))
    subprocess.run(
        [
            "hipcc",
            "--offload-arch=gfx90a",
            "--cuda-device-only",
            "-fsyntax-only",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
