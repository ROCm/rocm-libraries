# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The dtype-generic scalar buffer load (``tile.buffer_load``) in both engines.

``IRBuilder.buffer_load`` admits f16 / bf16 (16-bit intrinsic) and f32 / i32
(32-bit intrinsic). Both the LLVM and the HIP-source lowerings of the two
engines lower every admitted element type to the same text, and refuse any
other element type (reachable only through hand-built or deserialized IR)
with the same message.
"""

import pytest
from rocke.core.ir import BF16, F16, F32, I8, I32, IRBuilder, KernelDef, PtrType
from rocke.tests.engines import (
    assert_both_engines_reject,
    assert_both_engines_reject_hip,
    lower_hip_with_both_engines,
    lower_with_both_engines,
)

_ARCH = "gfx1250"


def _load_kernel(dtype, *, elem_override: str | None = None) -> KernelDef:
    """One scalar buffer load of ``dtype`` per thread, stored to DST."""
    b = IRBuilder("buffer_load_scalar")
    src = b.param("SRC", PtrType(dtype, "global"), readonly=True)
    dst = b.param("DST", PtrType(dtype, "global"))
    nbytes = b.param("nbytes", I32)
    rs = b.buffer_rsrc(src, nbytes)
    tid = b.thread_id_x()
    off = b.mul(tid, b.const_i32(4))
    value = b.buffer_load(rs, off, b.const_i32(0), dtype)
    if elem_override is not None:
        # Hand-built IR: an element type the builder does not admit.
        b.kernel.body.ops[-1].attrs["elem_type"] = elem_override
    b.global_store(dst, tid, value)
    return b.kernel


@pytest.mark.parametrize("dtype", [F16, BF16, F32, I32], ids=lambda t: t.name)
def test_both_engines_lower_every_element_type(dtype):
    ir = lower_with_both_engines(_load_kernel(dtype), _ARCH)
    width = "i16" if dtype.name in ("f16", "bf16") else "i32"
    assert f"@llvm.amdgcn.raw.ptr.buffer.load.{width}(" in ir
    hip = lower_hip_with_both_engines(_load_kernel(dtype), _ARCH)
    builtin = "b16" if dtype.name in ("f16", "bf16") else "b32"
    assert f"__builtin_amdgcn_raw_buffer_load_{builtin}(" in hip


def test_builder_refuses_other_element_types():
    b = IRBuilder("buffer_load_i8")
    src = b.param("SRC", PtrType(I8, "global"))
    rs = b.buffer_rsrc(src, b.const_i32(64))
    with pytest.raises(ValueError, match="unsupported dtype 'i8'"):
        b.buffer_load(rs, b.const_i32(0), b.const_i32(0), I8)


def test_both_engines_refuse_other_element_types():
    kernel = _load_kernel(I32, elem_override="i8")
    message = "tile.buffer_load: unsupported elem_type 'i8'"
    assert_both_engines_reject(kernel, _ARCH, ValueError, message)
    assert_both_engines_reject_hip(kernel, _ARCH, ValueError, message)
