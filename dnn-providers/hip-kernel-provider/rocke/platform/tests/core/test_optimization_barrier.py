# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Value barriers preserve types and survive serialization through both engines."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from rocke.core.ir import (
    BF16,
    BF8E5M2,
    F16,
    F32,
    FP8E4M3,
    I1,
    I8,
    I16,
    I32,
    I64,
    IRBuilder,
    PtrType,
    Type,
    VectorType,
)
from rocke.core.ir_serialize import parse, serialize
from rocke.core.lower_hip import lower_kernel_to_hip
from rocke.core.lower_llvm import lower_kernel_to_llvm


@pytest.mark.parametrize("idx", range(8))
def test_hip_barrier_paired_emitters(idx):
    """Independent builders must agree on the FP32 barrier before output conversion."""
    executable = os.environ.get("ROCKE_OPTIMIZATION_BARRIER_EMIT")
    if not executable:
        pytest.skip("set ROCKE_OPTIMIZATION_BARRIER_EMIT to the built barrier emitter")
    native = Path(executable).resolve()
    assert native.is_file(), f"barrier emitter is not a file: {native}"
    python = (
        Path(__file__).resolve().parents[1]
        / "instances/parity/optimization_barrier_emit.py"
    )
    c_hip = subprocess.run(
        [str(native), str(idx), "hip"], check=True, capture_output=True, timeout=30
    ).stdout
    p_hip = subprocess.run(
        [sys.executable, str(python), str(idx), "hip"],
        check=True,
        capture_output=True,
        timeout=30,
    ).stdout
    assert c_hip == p_hip
    assert b'asm ("" : "=v"' in c_hip
    assert b"asm volatile" not in c_hip
    assert b"factor_a * factor_b" in c_hip


@pytest.mark.parametrize(
    "dtype", [I1, I8, I16, I32, I64, BF16, F16, F32, FP8E4M3, BF8E5M2]
)
def test_scalar_barrier_roundtrip(dtype):
    b = IRBuilder("barrier")
    ptr = b.param("p", PtrType(dtype, "global"))
    tid = b.thread_id_x()
    value = b.global_load(ptr, tid, dtype)
    result = b.optimization_barrier(value)
    assert result.type == value.type
    b.global_store(ptr, tid, result)
    b.ret()
    copy = parse(serialize(b.kernel))
    for arch in ("gfx950", "gfx1250"):
        llvm = lower_kernel_to_llvm(b.kernel, arch=arch, llvm_flavor="llvm23")
        assert llvm == lower_kernel_to_llvm(copy, arch=arch, llvm_flavor="llvm23")
        assert 'asm "", "=v,0"' in llvm
        assert "asm sideeffect" not in llvm


@pytest.mark.parametrize(
    "dtype", [PtrType(F32, "global"), VectorType(F32, 2), Type("unknown")]
)
def test_rejects_non_numeric_scalar(dtype):
    b = IRBuilder("invalid_barrier")
    value = b.param("value", dtype)
    before = serialize(b.kernel)
    with pytest.raises(ValueError, match="directly lowerable scalar"):
        b.optimization_barrier(value)
    assert serialize(b.kernel) == before


@pytest.mark.parametrize("dtype", ["fp4", "fp6", "bf6", "e8m0", "e5m3", "tf32"])
def test_logical_types_use_storage_barriers(dtype):
    from rocke.core.ir import dtype_to_ir_type
    from rocke.helpers.mma_io import storage_ir_type

    logical = dtype_to_ir_type(dtype)
    b = IRBuilder("logical_barrier")
    value = b.param("value", logical)
    before = serialize(b.kernel)
    with pytest.raises(ValueError, match="directly lowerable scalar"):
        b.optimization_barrier(value)
    assert serialize(b.kernel) == before
    storage = storage_ir_type(dtype)
    assert storage == (I32 if dtype == "tf32" else I8)
    assert b.optimization_barrier(b.param("bits", storage)).type == storage


@pytest.mark.parametrize("arch", ["gfx950", "gfx1250"])
def test_predicate_producer(arch):
    b = IRBuilder("barrier_predicate")
    ptr = b.param("p", PtrType(I32, "global"))
    tid = b.thread_id_x()
    value = b.global_load(ptr, tid, I32)
    predicate = b.cmp_lt(value, b.const_i32(0))
    result = b.optimization_barrier(predicate)
    b.global_store(ptr, tid, b.zext(result, I32))
    b.ret()
    copy = parse(serialize(b.kernel))
    llvm = lower_kernel_to_llvm(b.kernel, arch=arch, llvm_flavor="llvm23")
    assert llvm == lower_kernel_to_llvm(copy, arch=arch, llvm_flavor="llvm23")
    assert 'asm "", "=v,0"' in llvm


@pytest.mark.parametrize("arch", ["gfx950", "gfx1250"])
@pytest.mark.parametrize("roundtrip", [False, True])
@pytest.mark.parametrize("clobber", [None, ""])
def test_hip_barrier_valid_clobber(arch, roundtrip, clobber):
    b = IRBuilder("barrier_clobber")
    result = b.optimization_barrier(b.param("value", F32))
    baseline = lower_kernel_to_hip(b.kernel, arch=arch)
    if clobber is not None:
        result.op.attrs["clobber"] = clobber
    kernel = parse(serialize(b.kernel)) if roundtrip else b.kernel
    assert lower_kernel_to_hip(kernel, arch=arch) == baseline
    assert 'asm ("" : "=v"' in baseline
    assert "asm volatile" not in baseline


@pytest.mark.parametrize("roundtrip", [False, True])
@pytest.mark.parametrize("clobber", [False, True, 0, 1, 0.0, 1.0, "memory"])
def test_hip_barrier_rejects_invalid_clobber(roundtrip, clobber):
    b = IRBuilder("invalid_barrier_clobber")
    result = b.optimization_barrier(b.param("value", F32))
    result.op.attrs["clobber"] = clobber
    kernel = parse(serialize(b.kernel)) if roundtrip else b.kernel
    with pytest.raises(ValueError, match="clobber must be an empty string"):
        lower_kernel_to_hip(kernel, arch="gfx1250")


@pytest.mark.parametrize("roundtrip", [False, True])
@pytest.mark.parametrize("sideeffect", [0, 1, 0.0, 1.0, "", "false"])
def test_hip_inline_asm_rejects_non_bool_sideeffect(roundtrip, sideeffect):
    b = IRBuilder("invalid_asm_sideeffect")
    result = b.optimization_barrier(b.param("value", F32))
    result.op.attrs["sideeffect"] = sideeffect
    kernel = parse(serialize(b.kernel)) if roundtrip else b.kernel
    with pytest.raises(ValueError, match="sideeffect must be a boolean"):
        lower_kernel_to_hip(kernel, arch="gfx1250")


@pytest.mark.parametrize("roundtrip", [False, True])
@pytest.mark.parametrize("sideeffect", [None, False, True])
@pytest.mark.parametrize("template", ["", "v_mov_b32 $0, $1"])
def test_hip_inline_asm_preserves_bool_and_default_sideeffect(
    roundtrip, sideeffect, template
):
    b = IRBuilder("asm_sideeffect")
    result = b.inline_asm(template, "=v,0", [b.param("value", F32)], F32)
    if sideeffect is None:
        del result.op.attrs["sideeffect"]
    else:
        result.op.attrs["sideeffect"] = sideeffect
    kernel = parse(serialize(b.kernel)) if roundtrip else b.kernel
    hip = lower_kernel_to_hip(kernel, arch="gfx1250")
    assert ("asm volatile" in hip) == (sideeffect is not False)
    assert ('"memory"' in hip) == (sideeffect is not False)
    if template:
        assert "v_mov_b32 %0, %1" in hip


def test_hip_general_pure_asm_preserves_explicit_memory_clobber():
    b = IRBuilder("asm_clobber")
    result = b.inline_asm(
        "v_mov_b32 $0, $1", "=v,0", [b.param("value", F32)], F32, sideeffect=False
    )
    result.op.attrs["clobber"] = "memory"
    hip = lower_kernel_to_hip(parse(serialize(b.kernel)), arch="gfx1250")
    assert 'asm ("v_mov_b32 %0, %1"' in hip
    assert '"memory"' in hip
