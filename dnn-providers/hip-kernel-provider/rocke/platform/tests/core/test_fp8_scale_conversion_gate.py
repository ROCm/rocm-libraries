# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The fused scale-and-convert ops lower only for a target with the instruction.

``cvt_scalef32_pk_f32_fp8x4`` / ``cvt_scalef32_pk_f32_bf8x4`` wrap
``v_cvt_scalef32_pk_f32_fp8`` / ``_bf8``, which only gfx950 has. The builder
has no target, so the check is made at lowering, from the ISA backend's
``has_fp8_scale_conversion``. It cannot be left to LLVM: for gfx942 LLVM
selects the intrinsic and writes an encoding the GPU does not have.

Both engines must refuse with the same text and still lower the op for gfx950.
The C++ half calls the engine directly so its own message is what is compared;
``tests/core/test_fp8_scale_conversion_gate.cpp`` pins the same text natively.
"""

from __future__ import annotations

import pytest
from rocke.core.backend import BackendError, _lower_via_cpp_engine
from rocke.core.ir import BF8E5M2, F32, FP8E4M3, IRBuilder, PtrType
from rocke.core.ir_serialize import serialize
from rocke.core.isa.backend import backend_for, wired_arches
from rocke.core.lower_llvm import _lower_kernel_to_llvm_python

_FLAVOR = "llvm20"

# (element type, builder method, op name, instruction, fallback builder method)
_VARIANTS = (
    (
        FP8E4M3,
        "cvt_scalef32_pk_f32_fp8x4",
        "arith.cvt_scalef32_pk_f32_fp8",
        "v_cvt_scalef32_pk_f32_fp8",
        "cvt_pk_f32_fp8x4",
    ),
    (
        BF8E5M2,
        "cvt_scalef32_pk_f32_bf8x4",
        "arith.cvt_scalef32_pk_f32_bf8",
        "v_cvt_scalef32_pk_f32_bf8",
        "cvt_pk_f32_bf8x4",
    ),
)

# Every wired target other than gfx950, read from the registry so a newly
# wired target is checked too.
_REFUSED = tuple(arch for arch in wired_arches() if arch != "gfx950")


def _kernel(elem, method: str):
    """D[0:4] = convert(S[0:4]) * scale, through the fused op."""
    b = IRBuilder("fp8_scale_conversion_gate")
    src = b.param("S", PtrType(elem, "global"))
    dst = b.param("D", PtrType(F32, "global"))
    scale = b.param("scale", F32)
    idx = b.const_i32(0)
    v = b.global_load_vN(src, idx, elem, 4)
    b.global_store_vN(dst, idx, getattr(b, method)(v, scale), 4)
    b.ret()
    return b.kernel


def _lower_cpp(kernel, arch: str) -> str:
    try:
        return _lower_via_cpp_engine(serialize(kernel), arch, _FLAVOR)
    except BackendError as e:
        pytest.skip(f"C++ engine not importable: {str(e)[:200]}")


def test_only_gfx950_backend_has_the_instruction():
    assert backend_for("gfx950").has_fp8_scale_conversion
    for arch in _REFUSED:
        assert not backend_for(arch).has_fp8_scale_conversion, arch


@pytest.mark.parametrize("elem,method,op,instruction,fallback", _VARIANTS)
def test_gfx950_lowers_the_fused_op_in_both_engines(
    elem, method, op, instruction, fallback
):
    kernel = _kernel(elem, method)
    py = _lower_kernel_to_llvm_python(kernel, arch="gfx950", llvm_flavor=_FLAVOR)
    intrinsic = "@llvm.amdgcn." + op.removeprefix("arith.").replace("_", ".")
    assert py.count(f"call <2 x float> {intrinsic}(") == 2  # one per lane pair
    assert _lower_cpp(kernel, "gfx950") == py


@pytest.mark.parametrize("arch", _REFUSED)
@pytest.mark.parametrize("elem,method,op,instruction,fallback", _VARIANTS)
def test_other_targets_are_refused_with_one_message(
    elem, method, op, instruction, fallback, arch
):
    kernel = _kernel(elem, method)
    want = (
        f"{op} needs {instruction}, which {arch} does not have (gfx950 does); "
        f"convert with {fallback} and multiply by the scale instead"
    )
    with pytest.raises(ValueError) as py_err:
        _lower_kernel_to_llvm_python(kernel, arch=arch, llvm_flavor=_FLAVOR)
    assert str(py_err.value) == want
    try:
        _lower_cpp(kernel, arch)
    except RuntimeError as cpp_err:
        # The binding prefixes its own context; the engine's text follows.
        assert str(cpp_err).endswith(f"): {want}")
    else:
        pytest.fail(f"the C++ engine lowered {op} for {arch}")
