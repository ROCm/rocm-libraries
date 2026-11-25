# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Both engines refuse the same img2col problems: a grouped problem and a 3-D
problem in their spec checks, each with the same reason in both (img2col
covers single-group 2-D convolution only), and a problem field set to None."""

from __future__ import annotations

import pytest
from rocke.tests.lanes import BUILD_VARIANT, import_or_skip_until

from kernels.common.conv_implicit_gemm import ConvProblem
from kernels.common.img2col import Img2ColSpec, build_img2col, is_valid_spec
from kernels.conv.engine_bridge import img2col_spec_to_dict

_GROUPED = Img2ColSpec(
    problem=ConvProblem(N=2, Hi=16, Wi=16, C=32, K=32, Y=3, X=3, groups=2),
    block_tile_m=8,
    block_tile_k=128,
    vec_k=4,
)
_REASON = "groups 2 unsupported (img2col covers single-group convolution only)"
_THREE_D = Img2ColSpec(
    problem=ConvProblem(
        N=1, Hi=8, Wi=8, C=16, K=16, Y=3, X=3, Di=4, Z=3, sD=1, pD=0, dD=1
    ),
    block_tile_m=4,
    block_tile_k=64,
    vec_k=1,
)
_THREE_D_REASON = (
    "img2col failed - is_3d=True, expected a 2-D problem "
    "(img2col covers 2-D convolution only)"
)


def test_python_refuses_a_grouped_problem():
    assert is_valid_spec(_GROUPED, arch="gfx950") == (False, _REASON)
    with pytest.raises(ValueError, match=r"groups 2 unsupported \(img2col covers"):
        build_img2col(_GROUPED, arch="gfx950")


def test_the_cpp_entry_refuses_a_grouped_problem_like_python():
    engine = import_or_skip_until("rocke_library_engine", BUILD_VARIANT)
    with pytest.raises(RuntimeError) as refusal:
        engine.img2col_lower_llvm(img2col_spec_to_dict(_GROUPED), arch="gfx950")
    assert _REASON in str(refusal.value)


def test_python_refuses_a_3d_problem():
    assert is_valid_spec(_THREE_D, arch="gfx950") == (False, _THREE_D_REASON)
    with pytest.raises(ValueError) as refusal:
        build_img2col(_THREE_D, arch="gfx950")
    assert str(refusal.value) == f"invalid img2col spec for gfx950: {_THREE_D_REASON}"


def test_the_cpp_entry_refuses_a_3d_problem_like_python():
    engine = import_or_skip_until("rocke_library_engine", BINDING_LANE)
    with pytest.raises(RuntimeError) as refusal:
        engine.img2col_lower_llvm(img2col_spec_to_dict(_THREE_D), arch="gfx950")
    assert _THREE_D_REASON in str(refusal.value)


def _with_padding_none() -> Img2ColSpec:
    return Img2ColSpec(
        problem=ConvProblem(N=1, Hi=8, Wi=8, C=16, K=16, Y=3, X=3, pH=None),
        block_tile_m=4,
        block_tile_k=64,
        vec_k=1,
    )


def test_both_engines_refuse_a_problem_field_set_to_none():
    with pytest.raises(TypeError, match="NoneType"):
        build_img2col(_with_padding_none(), arch="gfx950")
    engine = import_or_skip_until("rocke_library_engine", BUILD_VARIANT)
    with pytest.raises(RuntimeError, match="Unable to cast Python instance"):
        engine.img2col_lower_llvm(
            img2col_spec_to_dict(_with_padding_none()), arch="gfx950"
        )
