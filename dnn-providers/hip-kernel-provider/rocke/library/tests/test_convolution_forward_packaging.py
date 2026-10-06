# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""CPU tests for descriptor hydration and the unmodified convolution builder ABI.

Both packaged families are covered: implicit GEMM and rocKE's direct depthwise
kernels. Direct kernels are lowered with the Python backend here.
"""

from __future__ import annotations

import inspect
import json
import math
import typing
from dataclasses import asdict, fields, replace
from pathlib import Path

import pytest
from builders.common.convolution_forward import (
    KERNEL_FAMILY_DIRECT_DEPTHWISE,
    KERNEL_FAMILY_IMPLICIT_GEMM,
    Gfx950ConvFwdSpec,
    build_gfx950_conv_fwd,
    gfx950_conv_fwd_direct_spec_for_request,
    gfx950_conv_fwd_spec_for_request,
    supports_gfx950_conv_fwd,
)
from dispatch.grouped_convolution import ConvGroupedRequest, dispatch_conv_grouped
from rocke.core.ir import BF16, F16, I32, PtrType
from rocke.core.lower_llvm import _lower_kernel_to_llvm_python
from kernels.common.conv_abi import (
    conv_arg_names,
    conv_desc_names,
    conv_direct_arg_names,
)
from kernels.common.conv_args import ConvArgs
from kernels.common.conv_direct_grouped import (
    DirectConvProblem,
    DirectDepthwiseSpatialSpec,
    DirectDepthwiseSpec,
    build_direct_depthwise,
    build_direct_depthwise_spatial,
)
from kernels.common.conv_implicit_gemm import (
    ConvProblem,
    build_implicit_gemm_conv,
)


def _request(**overrides) -> ConvGroupedRequest:
    values = {
        "N": 2,
        "Hi": 14,
        "Wi": 14,
        "C": 32,
        "K": 32,
        "Y": 3,
        "X": 3,
        "pad_h": 1,
        "pad_w": 1,
        "arch": "gfx950",
    }
    values.update(overrides)
    return ConvGroupedRequest(**values)


def _assert_aot_abi(packaged, spec: Gfx950ConvFwdSpec, dtype: str) -> None:
    """The packaged kernel declares rocKE's AOT ABI for its family, in order."""
    abi = conv_direct_arg_names() if spec.is_direct else conv_arg_names()
    assert [p.name for p in packaged.params] == conv_desc_names(abi)
    assert [p.name for p in packaged.params][:6] == [
        "A",
        "B",
        "D",
        "A_bytes",
        "B_bytes",
        "D_bytes",
    ]
    element = F16 if dtype == "fp16" else BF16
    assert [p.type for p in packaged.params] == [PtrType(element, "global")] * 3 + [
        I32
    ] * (len(abi) - 3)
    assert all(p.attrs["align"] == 16 for p in packaged.params[:3])
    signature = spec.launch_signature()
    assert [arg["name"] for arg in signature] == conv_desc_names(abi)
    assert set(spec.launch_values(0, 0, 0, 0, 0, 0)) == set(conv_desc_names(abi))


def _original_problem(request: ConvGroupedRequest) -> ConvProblem:
    return ConvProblem(
        N=request.N,
        Hi=request.Hi,
        Wi=request.Wi,
        C=request.C,
        K=request.K,
        Y=request.Y,
        X=request.X,
        sH=request.stride_h,
        sW=request.stride_w,
        pH=request.pad_h,
        pW=request.pad_w,
        dH=request.dilation_h,
        dW=request.dilation_w,
        groups=request.G,
    )


@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
def test_factory_preserves_dispatcher_defaults(dtype):
    request = _request(dtype=dtype)
    packaged = gfx950_conv_fwd_spec_for_request(request)
    original = dispatch_conv_grouped(request).spec.to_fwd_spec(
        _original_problem(request)
    )
    actual = packaged.to_instance_spec()
    assert replace(actual, name=original.name) == original
    assert packaged.epilogue == "cshuffle"
    assert packaged.tile_k == 64
    assert packaged.grid() == (1, 7, 1)
    assert packaged.block() == (256, 1, 1)


@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
@pytest.mark.parametrize("tile_k", [64, 128])
@pytest.mark.parametrize("llvm_flavor", ["llvm20", "llvm22"])
def test_adapter_emits_original_llvm_and_aot_abi(dtype, tile_k, llvm_flavor):
    request = _request(dtype=dtype)
    spec = gfx950_conv_fwd_spec_for_request(request, tile_k=tile_k)
    original_spec = dispatch_conv_grouped(request).spec.to_fwd_spec(
        _original_problem(request)
    )
    original_spec = replace(
        original_spec, tile_k=tile_k, name=spec.to_instance_spec().name
    )
    packaged = build_gfx950_conv_fwd(spec, arch="gfx950")
    original = build_implicit_gemm_conv(original_spec, arch="gfx950")
    assert _lower_kernel_to_llvm_python(
        packaged, arch="gfx950", llvm_flavor=llvm_flavor
    ) == _lower_kernel_to_llvm_python(original, arch="gfx950", llvm_flavor=llvm_flavor)
    assert packaged.name == spec.kernel_name()
    _assert_aot_abi(packaged, spec, dtype)
    assert packaged.max_workgroup_size == 256


@pytest.mark.parametrize(
    "overrides",
    [
        {"Y": 1, "X": 1, "pad_h": 0, "pad_w": 0},
        {"stride_h": 2, "stride_w": 2},
        {"dilation_h": 2, "dilation_w": 2, "pad_h": 2, "pad_w": 2},
        {"Hi": 15, "Wi": 19, "Y": 3, "X": 5, "pad_h": 1, "pad_w": 2, "stride_w": 2},
    ],
)
def test_geometry_round_trip_for_extended_2d_matrix(overrides):
    request = _request(**overrides)
    spec = gfx950_conv_fwd_spec_for_request(request)
    assert supports_gfx950_conv_fwd(spec) == (True, "ok")
    assert spec.to_problem() == _original_problem(request)
    assert spec.grid() == dispatch_conv_grouped(request).grid


def test_serialized_spec_and_typed_builder_are_packager_compatible():
    spec = gfx950_conv_fwd_spec_for_request(_request(), tile_k=128)
    values = json.loads(json.dumps(asdict(spec)))
    assert Gfx950ConvFwdSpec(**values) == spec
    assert all(type(values[field.name]) in (int, str) for field in fields(spec))
    signature = inspect.signature(build_gfx950_conv_fwd)
    assert list(signature.parameters) == ["spec", "arch"]
    assert signature.parameters["arch"].kind is inspect.Parameter.KEYWORD_ONLY
    assert typing.get_type_hints(build_gfx950_conv_fwd)["spec"] is Gfx950ConvFwdSpec


@pytest.mark.parametrize("pipeline", ["compv3", "compv4", "basic"])
def test_swept_pipelines_package_with_the_dispatchers_launch(pipeline):
    # Only the schedule differs: block and grid are what the native pack computes.
    spec = replace(gfx950_conv_fwd_spec_for_request(_request()), pipeline=pipeline)
    assert supports_gfx950_conv_fwd(spec) == (True, "ok")
    assert spec.block() == (256, 1, 1)
    assert spec.grid() == gfx950_conv_fwd_spec_for_request(_request()).grid()
    assert build_gfx950_conv_fwd(spec, arch="gfx950").name == spec.kernel_name()


@pytest.mark.parametrize(
    "override",
    [
        {"dtype": "bf16"},
        {"sH": 2},
        {"sW": 2},
        {"pH": 2},
        {"pW": 2},
        {"dH": 2},
        {"dW": 2},
        {"tile_k": 128},
    ],
)
def test_symbols_include_attributes_missing_from_original_shape_name(override):
    spec = gfx950_conv_fwd_spec_for_request(_request())
    changed = replace(spec, **override)
    assert changed.kernel_name() != spec.kernel_name()
    restored = Gfx950ConvFwdSpec(**json.loads(json.dumps(asdict(changed))))
    assert restored.kernel_name() == changed.kernel_name()


@pytest.mark.parametrize(
    ("override", "reason"),
    [
        ({"N": 0}, "N must be"),
        ({"C": True}, "C must be"),
        ({"Hi": 14.0}, "Hi must be"),
        ({"sH": 0}, "sH must be"),
        ({"dW": -1}, "dW must be"),
        ({"pH": -1}, "pH must be"),
        ({"groups": 0}, "groups must be"),
        ({"groups": True}, "groups must be"),
        ({"groups": 65536, "C": 65536, "K": 65536}, "groups must be"),
        ({"groups": 3}, "C=32 is not divisible by groups=3"),
        ({"groups": 2, "K": 31}, "K=31 is not divisible by groups=2"),
        ({"groups": 2, "Y": 1, "X": 1, "pH": 0, "pW": 0}, "grouped pointwise"),
        ({"groups": 32, "Y": 1, "X": 1, "pH": 0, "pW": 0}, "grouped pointwise"),
        ({"layout": "NCHW"}, "NHWC"),
        ({"dtype": "fp32"}, "fp16 or bf16"),
        ({"tile_k": 0}, "tile_k must be"),
        ({"warp_m": 0}, "warp_m must be"),
        ({"tile_m": 96}, "not divisible"),
        ({"wave_size": 32}, "wave_size"),
        ({"pipeline": "wavelet"}, "supports pipelines"),
        ({"epilogue": "unknown"}, "epilogue must be"),
        ({"epilogue": "default"}, "vector size c"),
        ({"Hi": 1, "Wi": 1, "Y": 7, "X": 7}, "nonpositive output"),
        ({"N": 33554432, "Hi": 1, "Wi": 1}, "input byte count"),
        ({"dH": (1 << 31) - 1}, "effective filter height"),
        (
            {
                "N": 1,
                "Hi": 4096,
                "Wi": 4096,
                "C": 1,
                "K": 1,
                "Y": 1,
                "X": 1,
                "pH": 0,
                "pW": 0,
            },
            "grid_m",
        ),
    ],
)
def test_invalid_specs_decline_before_emission(override, reason):
    spec = replace(gfx950_conv_fwd_spec_for_request(_request()), **override)
    ok, explanation = supports_gfx950_conv_fwd(spec)
    assert not ok
    assert reason in explanation
    with pytest.raises(ValueError, match="invalid packaged forward convolution"):
        build_gfx950_conv_fwd(spec, arch="gfx950")


@pytest.mark.parametrize(
    ("override", "reason"),
    [
        ({"direction": "wgrad"}, "forward convolution only"),
        ({"G": 3}, "C=32 is not divisible by groups=3"),
        ({"G": 2, "K": 31}, "K=31 is not divisible by groups=2"),
        ({"G": 2, "Y": 1, "X": 1, "pad_h": 0, "pad_w": 0}, "grouped pointwise"),
        ({"layout": "NCHW"}, "NHWC"),
        ({"arch": "gfx942"}, "gfx950"),
        ({"Di": 4, "Z": 1, "stride_d": 1, "pad_d": 0, "dilation_d": 1}, "2D"),
        ({"vec_size_c": 1}, "vector defaults"),
        ({"stride_w": 0}, "sW must be"),
    ],
)
def test_request_factory_declines_outside_integration_contract(override, reason):
    with pytest.raises(ValueError, match=reason):
        gfx950_conv_fwd_spec_for_request(_request(**override))


def test_arch_is_checked_at_build_time():
    spec = gfx950_conv_fwd_spec_for_request(_request())
    assert not supports_gfx950_conv_fwd(spec, arch="gfx942")[0]
    with pytest.raises(ValueError, match="requires arch=gfx950"):
        build_gfx950_conv_fwd(spec, arch="gfx942")


def test_odd_output_channels_obtain_dispatcher_scalar_epilogue():
    spec = gfx950_conv_fwd_spec_for_request(_request(K=31))
    assert spec.epilogue == "default"
    assert supports_gfx950_conv_fwd(spec) == (True, "ok")


@pytest.mark.parametrize(
    "overrides",
    [
        {"G": 2},
        {"G": 32},
        {"G": 5, "C": 5, "K": 5, "Y": 7, "X": 7, "pad_h": 3, "pad_w": 3},
        {"G": 4, "C": 12, "K": 24},
        {"G": 2, "C": 16, "K": 16, "Y": 5, "X": 5, "dilation_h": 2, "dilation_w": 2},
        {"G": 32, "stride_h": 2, "stride_w": 2},
        # 1x1 with stride 2 is not the pointwise shortcut, so it stays grouped.
        {"G": 2, "Y": 1, "X": 1, "pad_h": 0, "pad_w": 0, "stride_h": 2},
    ],
)
def test_grouped_requests_are_accepted_with_group_on_grid_z(overrides):
    request = _request(**overrides)
    spec = gfx950_conv_fwd_spec_for_request(request)
    assert spec.groups == request.G
    assert supports_gfx950_conv_fwd(spec) == (True, "ok")
    assert spec.to_problem() == _original_problem(request)
    assert spec.grid() == dispatch_conv_grouped(request).grid
    problem = spec.to_problem()
    assert spec.grid() == (
        -(-problem.kpg // spec.tile_n),
        -(-problem.M // spec.tile_m),
        request.G,
    )


@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
@pytest.mark.parametrize("tile_k", [None, 128])
def test_depthwise_spec_emits_original_kernel(dtype, tile_k):
    request = _request(dtype=dtype, G=32)
    spec = gfx950_conv_fwd_spec_for_request(request, tile_k=tile_k)
    original_spec = dispatch_conv_grouped(request).spec.to_fwd_spec(
        _original_problem(request)
    )
    original_spec = replace(
        original_spec,
        tile_k=spec.tile_k,
        name=spec.to_instance_spec().name,
    )
    packaged = build_gfx950_conv_fwd(spec, arch="gfx950")
    original = build_implicit_gemm_conv(original_spec, arch="gfx950")
    assert _lower_kernel_to_llvm_python(
        packaged, arch="gfx950", llvm_flavor="llvm22"
    ) == _lower_kernel_to_llvm_python(original, arch="gfx950", llvm_flavor="llvm22")
    assert packaged.name == spec.kernel_name()
    _assert_aot_abi(packaged, spec, dtype)
    assert spec.grid()[2] == spec.groups == 32
    assert spec.block() == (256, 1, 1)


# ---------------------------------------------------------------------------
# Direct depthwise family (kernel_family=1)
# ---------------------------------------------------------------------------


def _dw_request(**overrides) -> ConvGroupedRequest:
    values = {"G": 32}
    values.update(overrides)
    return _request(**values)


def _rocke_direct_spec(spec: Gfx950ConvFwdSpec):
    problem = DirectConvProblem(
        N=spec.N,
        H=spec.Hi,
        W=spec.Wi,
        groups=spec.groups,
        cpg=1,
        kpg=1,
        KH=spec.Y,
        KW=spec.X,
        PAD=spec.pH,
        stride=spec.sH,
        dtype=spec.dtype,
    )
    name = spec.to_direct_spec().name
    if spec.direct_variant == "spatial":
        return DirectDepthwiseSpatialSpec(
            problem=problem, name=name, block_waves=spec.block_waves
        )
    return DirectDepthwiseSpec(
        problem=problem, name=name, block_w=spec.block_w, block_waves=spec.block_waves
    )


def test_family_fields_trail_with_implicit_gemm_defaults():
    names = [field.name for field in fields(Gfx950ConvFwdSpec)]
    assert names[-4:] == ["kernel_family", "direct_variant", "block_w", "block_waves"]
    # The KMD declares these defaults; hkp_pack proves agreement per kernel.
    defaults = {field.name: field.default for field in fields(Gfx950ConvFwdSpec)}
    assert (
        defaults["kernel_family"],
        defaults["direct_variant"],
        defaults["block_w"],
        defaults["block_waves"],
    ) == (KERNEL_FAMILY_IMPLICIT_GEMM, "none", 0, 0)


# Symbols packaged before the family fields existed (computed at the parent
# revision). Implicit-GEMM specs must keep them so shipped code objects and
# descriptors stay byte-identical.
@pytest.mark.parametrize(
    ("overrides", "tile_k", "symbol"),
    [
        (
            {},
            None,
            "hkp_conv_fwd_gfx950_b76e8195f39e3a99a1d921c4_N2H14W14C32_K32Y3X3_"
            "t64x64x64_w2x2_a32x32x16_mem_cshuffle",
        ),
        (
            {},
            128,
            "hkp_conv_fwd_gfx950_a0a804fb8cd438305a73f21a_N2H14W14C32_K32Y3X3_"
            "t64x64x128_w2x2_a32x32x16_mem_cshuffle",
        ),
        (
            {"dtype": "bf16", "G": 32},
            None,
            "hkp_conv_fwd_gfx950_dd539cc5f515bb894b5c7c4e_N2H14W14C32_K32Y3X3G32_"
            "t64x64x64_w2x2_a32x32x16_mem_cshuffle",
        ),
        (
            {
                "G": 5,
                "C": 5,
                "K": 5,
                "Hi": 9,
                "Wi": 9,
                "Y": 7,
                "X": 7,
                "pad_h": 3,
                "pad_w": 3,
            },
            None,
            "hkp_conv_fwd_gfx950_29ccf81e7bdbb6b5b03b37e5_N2H9W9C5_K5Y7X7G5_"
            "t64x64x64_w2x2_a32x32x16_mem_default",
        ),
        (
            {"stride_h": 2, "stride_w": 2, "Hi": 17, "Wi": 19, "K": 64},
            128,
            "hkp_conv_fwd_gfx950_038a332bb2fb0a359dff3c7b_N2H17W19C32_K64Y3X3_"
            "t64x64x128_w2x2_a32x32x16_mem_cshuffle",
        ),
    ],
)
def test_existing_implicit_gemm_symbols_are_unchanged(overrides, tile_k, symbol):
    spec = gfx950_conv_fwd_spec_for_request(_request(**overrides), tile_k=tile_k)
    assert spec.kernel_name() == symbol
    # A descriptor written before the family fields existed hydrates unchanged.
    legacy = {
        key: value
        for key, value in asdict(spec).items()
        if key not in ("kernel_family", "direct_variant", "block_w", "block_waves")
    }
    assert Gfx950ConvFwdSpec(**legacy).kernel_name() == symbol


_KDP_PATH = (
    Path(__file__).resolve().parents[3]
    / "src/engines/kernel_ingestor_engine/descriptors/rocKE/gfx950_conv_fwd"
    / "gfx950_conv_fwd.kdp.json"
)
_FAMILY_FIELDS = ("kernel_family", "direct_variant", "block_w", "block_waves")

# The first shipped catalog, in descriptor order: every implicit-GEMM descriptor
# that existed before the direct family, with the launch symbol of the code
# object it was first packaged as. Extending the catalog must leave these
# descriptors, their order and their symbols unchanged.
_FIRST_SHIPPED_SYMBOLS = (
    (
        "gfx950_conv_fwd_dtypefp16_N2_Hi14_Wi14_C32_K32_Y3_X3_sH1_sW1_pH1_pW1_dH1_dW1_tile_k64",
        "hkp_conv_fwd_gfx950_b76e8195f39e3a99a1d921c4_N2H14W14C32_K32Y3X3_t64x64x64_w2x2_a32x32x16_mem_cshuffle",
    ),
    (
        "gfx950_conv_fwd_dtypefp16_N2_Hi14_Wi14_C32_K32_Y3_X3_sH1_sW1_pH1_pW1_dH1_dW1_tile_k128",
        "hkp_conv_fwd_gfx950_a0a804fb8cd438305a73f21a_N2H14W14C32_K32Y3X3_t64x64x128_w2x2_a32x32x16_mem_cshuffle",
    ),
    (
        "gfx950_conv_fwd_dtypebf16_N2_Hi14_Wi14_C32_K32_Y3_X3_sH1_sW1_pH1_pW1_dH1_dW1_tile_k64",
        "hkp_conv_fwd_gfx950_0d2d3007bba99153602afdd6_N2H14W14C32_K32Y3X3_t64x64x64_w2x2_a32x32x16_mem_cshuffle",
    ),
    (
        "gfx950_conv_fwd_dtypebf16_N2_Hi14_Wi14_C32_K32_Y3_X3_sH1_sW1_pH1_pW1_dH1_dW1_tile_k128",
        "hkp_conv_fwd_gfx950_a8f588256e9441d929105762_N2H14W14C32_K32Y3X3_t64x64x128_w2x2_a32x32x16_mem_cshuffle",
    ),
    (
        "gfx950_conv_fwd_dtypefp16_N1_Hi8_Wi8_C32_K64_Y1_X1_sH1_sW1_pH0_pW0_dH1_dW1_tile_k64",
        "hkp_conv_fwd_gfx950_3f4ec2ee8a1817edb33ec4a9_N1H8W8C32_K64Y1X1_t64x64x64_w2x2_a32x32x16_mem_cshuffle",
    ),
    (
        "gfx950_conv_fwd_dtypefp16_N1_Hi8_Wi8_C32_K64_Y1_X1_sH1_sW1_pH0_pW0_dH1_dW1_tile_k128",
        "hkp_conv_fwd_gfx950_7571ff6d15d9b530335ebead_N1H8W8C32_K64Y1X1_t64x64x128_w2x2_a32x32x16_mem_cshuffle",
    ),
    (
        "gfx950_conv_fwd_dtypebf16_N1_Hi8_Wi8_C32_K64_Y1_X1_sH1_sW1_pH0_pW0_dH1_dW1_tile_k64",
        "hkp_conv_fwd_gfx950_cc679850ce950efb5ddbf192_N1H8W8C32_K64Y1X1_t64x64x64_w2x2_a32x32x16_mem_cshuffle",
    ),
    (
        "gfx950_conv_fwd_dtypebf16_N1_Hi8_Wi8_C32_K64_Y1_X1_sH1_sW1_pH0_pW0_dH1_dW1_tile_k128",
        "hkp_conv_fwd_gfx950_035740b38b42787624008330_N1H8W8C32_K64Y1X1_t64x64x128_w2x2_a32x32x16_mem_cshuffle",
    ),
    (
        "gfx950_conv_fwd_dtypefp16_N2_Hi17_Wi19_C32_K64_Y3_X3_sH2_sW2_pH1_pW1_dH1_dW1_tile_k64",
        "hkp_conv_fwd_gfx950_c471019d8d559e50053c541b_N2H17W19C32_K64Y3X3_t64x64x64_w2x2_a32x32x16_mem_cshuffle",
    ),
    (
        "gfx950_conv_fwd_dtypefp16_N2_Hi17_Wi19_C32_K64_Y3_X3_sH2_sW2_pH1_pW1_dH1_dW1_tile_k128",
        "hkp_conv_fwd_gfx950_038a332bb2fb0a359dff3c7b_N2H17W19C32_K64Y3X3_t64x64x128_w2x2_a32x32x16_mem_cshuffle",
    ),
    (
        "gfx950_conv_fwd_dtypebf16_N2_Hi17_Wi19_C32_K64_Y3_X3_sH2_sW2_pH1_pW1_dH1_dW1_tile_k64",
        "hkp_conv_fwd_gfx950_c98ccc30b19e15f49d140083_N2H17W19C32_K64Y3X3_t64x64x64_w2x2_a32x32x16_mem_cshuffle",
    ),
    (
        "gfx950_conv_fwd_dtypebf16_N2_Hi17_Wi19_C32_K64_Y3_X3_sH2_sW2_pH1_pW1_dH1_dW1_tile_k128",
        "hkp_conv_fwd_gfx950_27bfb9aa129132faae125099_N2H17W19C32_K64Y3X3_t64x64x128_w2x2_a32x32x16_mem_cshuffle",
    ),
    (
        "gfx950_conv_fwd_dtypefp16_N1_Hi15_Wi17_C32_K32_Y3_X3_sH1_sW1_pH2_pW2_dH2_dW2_tile_k64",
        "hkp_conv_fwd_gfx950_eaa0f52c429df4468173ee7f_N1H15W17C32_K32Y3X3_t64x64x64_w2x2_a32x32x16_mem_cshuffle",
    ),
    (
        "gfx950_conv_fwd_dtypefp16_N1_Hi15_Wi17_C32_K32_Y3_X3_sH1_sW1_pH2_pW2_dH2_dW2_tile_k128",
        "hkp_conv_fwd_gfx950_a57f2c033c3b1fc80e88379c_N1H15W17C32_K32Y3X3_t64x64x128_w2x2_a32x32x16_mem_cshuffle",
    ),
    (
        "gfx950_conv_fwd_dtypebf16_N1_Hi15_Wi17_C32_K32_Y3_X3_sH1_sW1_pH2_pW2_dH2_dW2_tile_k64",
        "hkp_conv_fwd_gfx950_8fb0c109c146787879a31af6_N1H15W17C32_K32Y3X3_t64x64x64_w2x2_a32x32x16_mem_cshuffle",
    ),
    (
        "gfx950_conv_fwd_dtypebf16_N1_Hi15_Wi17_C32_K32_Y3_X3_sH1_sW1_pH2_pW2_dH2_dW2_tile_k128",
        "hkp_conv_fwd_gfx950_c4a088fd3cbb94e2e595964e_N1H15W17C32_K32Y3X3_t64x64x128_w2x2_a32x32x16_mem_cshuffle",
    ),
    (
        "gfx950_conv_fwd_dtypefp16_N2_Hi11_Wi17_C32_K64_Y3_X5_sH1_sW2_pH1_pW2_dH1_dW1_tile_k64",
        "hkp_conv_fwd_gfx950_b58efed7b3e75fe2cf9b0a63_N2H11W17C32_K64Y3X5_t64x64x64_w2x2_a32x32x16_mem_cshuffle",
    ),
    (
        "gfx950_conv_fwd_dtypefp16_N2_Hi11_Wi17_C32_K64_Y3_X5_sH1_sW2_pH1_pW2_dH1_dW1_tile_k128",
        "hkp_conv_fwd_gfx950_1d78aa67b2a7235d515ebef5_N2H11W17C32_K64Y3X5_t64x64x128_w2x2_a32x32x16_mem_cshuffle",
    ),
    (
        "gfx950_conv_fwd_dtypebf16_N2_Hi11_Wi17_C32_K64_Y3_X5_sH1_sW2_pH1_pW2_dH1_dW1_tile_k64",
        "hkp_conv_fwd_gfx950_19f0fa8a1afc7d6a52b72f47_N2H11W17C32_K64Y3X5_t64x64x64_w2x2_a32x32x16_mem_cshuffle",
    ),
    (
        "gfx950_conv_fwd_dtypebf16_N2_Hi11_Wi17_C32_K64_Y3_X5_sH1_sW2_pH1_pW2_dH1_dW1_tile_k128",
        "hkp_conv_fwd_gfx950_ea172dc4410f154b421fa10e_N2H11W17C32_K64Y3X5_t64x64x128_w2x2_a32x32x16_mem_cshuffle",
    ),
    (
        "gfx950_conv_fwd_dtypefp16_G2_N2_Hi13_Wi13_C16_K16_Y3_X3_sH1_sW1_pH1_pW1_dH1_dW1_tile_k64",
        "hkp_conv_fwd_gfx950_593b9a4ad49fbcd61f792ebf_N2H13W13C16_K16Y3X3G2_t64x64x64_w2x2_a32x32x16_mem_cshuffle",
    ),
    (
        "gfx950_conv_fwd_dtypefp16_G2_N2_Hi13_Wi13_C16_K16_Y3_X3_sH1_sW1_pH1_pW1_dH1_dW1_tile_k128",
        "hkp_conv_fwd_gfx950_c6a957ccd905d98003817c1a_N2H13W13C16_K16Y3X3G2_t64x64x128_w2x2_a32x32x16_mem_cshuffle",
    ),
    (
        "gfx950_conv_fwd_dtypebf16_G2_N2_Hi13_Wi13_C16_K16_Y3_X3_sH1_sW1_pH1_pW1_dH1_dW1_tile_k64",
        "hkp_conv_fwd_gfx950_932a20ef2b1a31255e7a7ba9_N2H13W13C16_K16Y3X3G2_t64x64x64_w2x2_a32x32x16_mem_cshuffle",
    ),
    (
        "gfx950_conv_fwd_dtypebf16_G2_N2_Hi13_Wi13_C16_K16_Y3_X3_sH1_sW1_pH1_pW1_dH1_dW1_tile_k128",
        "hkp_conv_fwd_gfx950_706998c4cb7b030706373c2f_N2H13W13C16_K16Y3X3G2_t64x64x128_w2x2_a32x32x16_mem_cshuffle",
    ),
    (
        "gfx950_conv_fwd_dtypefp16_G4_N2_Hi11_Wi11_C12_K24_Y3_X3_sH1_sW1_pH1_pW1_dH1_dW1_tile_k64",
        "hkp_conv_fwd_gfx950_572e6f6a777ad4f691490acc_N2H11W11C12_K24Y3X3G4_t64x64x64_w2x2_a32x32x16_mem_cshuffle",
    ),
    (
        "gfx950_conv_fwd_dtypefp16_G4_N2_Hi11_Wi11_C12_K24_Y3_X3_sH1_sW1_pH1_pW1_dH1_dW1_tile_k128",
        "hkp_conv_fwd_gfx950_7bc6b830fabee987dbe2ced5_N2H11W11C12_K24Y3X3G4_t64x64x128_w2x2_a32x32x16_mem_cshuffle",
    ),
    (
        "gfx950_conv_fwd_dtypebf16_G4_N2_Hi11_Wi11_C12_K24_Y3_X3_sH1_sW1_pH1_pW1_dH1_dW1_tile_k64",
        "hkp_conv_fwd_gfx950_93e233b4ef78c48534c02a2c_N2H11W11C12_K24Y3X3G4_t64x64x64_w2x2_a32x32x16_mem_cshuffle",
    ),
    (
        "gfx950_conv_fwd_dtypebf16_G4_N2_Hi11_Wi11_C12_K24_Y3_X3_sH1_sW1_pH1_pW1_dH1_dW1_tile_k128",
        "hkp_conv_fwd_gfx950_38e90f695e6711943e282e85_N2H11W11C12_K24Y3X3G4_t64x64x128_w2x2_a32x32x16_mem_cshuffle",
    ),
    (
        "gfx950_conv_fwd_dtypefp16_G32_N2_Hi14_Wi14_C32_K32_Y3_X3_sH1_sW1_pH1_pW1_dH1_dW1_tile_k64",
        "hkp_conv_fwd_gfx950_699336feb194d7d73d6b48ff_N2H14W14C32_K32Y3X3G32_t64x64x64_w2x2_a32x32x16_mem_cshuffle",
    ),
    (
        "gfx950_conv_fwd_dtypefp16_G32_N2_Hi14_Wi14_C32_K32_Y3_X3_sH1_sW1_pH1_pW1_dH1_dW1_tile_k128",
        "hkp_conv_fwd_gfx950_ea88891c9ac08b99e0b1a376_N2H14W14C32_K32Y3X3G32_t64x64x128_w2x2_a32x32x16_mem_cshuffle",
    ),
    (
        "gfx950_conv_fwd_dtypebf16_G32_N2_Hi14_Wi14_C32_K32_Y3_X3_sH1_sW1_pH1_pW1_dH1_dW1_tile_k64",
        "hkp_conv_fwd_gfx950_dd539cc5f515bb894b5c7c4e_N2H14W14C32_K32Y3X3G32_t64x64x64_w2x2_a32x32x16_mem_cshuffle",
    ),
    (
        "gfx950_conv_fwd_dtypebf16_G32_N2_Hi14_Wi14_C32_K32_Y3_X3_sH1_sW1_pH1_pW1_dH1_dW1_tile_k128",
        "hkp_conv_fwd_gfx950_4bef76c9324837313aab7fc3_N2H14W14C32_K32Y3X3G32_t64x64x128_w2x2_a32x32x16_mem_cshuffle",
    ),
    (
        "gfx950_conv_fwd_dtypefp16_G32_N2_Hi17_Wi17_C32_K32_Y3_X3_sH2_sW2_pH1_pW1_dH1_dW1_tile_k64",
        "hkp_conv_fwd_gfx950_9deb014d7a44fe1990a3e64f_N2H17W17C32_K32Y3X3G32_t64x64x64_w2x2_a32x32x16_mem_cshuffle",
    ),
    (
        "gfx950_conv_fwd_dtypefp16_G32_N2_Hi17_Wi17_C32_K32_Y3_X3_sH2_sW2_pH1_pW1_dH1_dW1_tile_k128",
        "hkp_conv_fwd_gfx950_33bf9c192368a03dfcbfaad5_N2H17W17C32_K32Y3X3G32_t64x64x128_w2x2_a32x32x16_mem_cshuffle",
    ),
    (
        "gfx950_conv_fwd_dtypebf16_G32_N2_Hi17_Wi17_C32_K32_Y3_X3_sH2_sW2_pH1_pW1_dH1_dW1_tile_k64",
        "hkp_conv_fwd_gfx950_2f3ec3ae42bf5969c2f16e6c_N2H17W17C32_K32Y3X3G32_t64x64x64_w2x2_a32x32x16_mem_cshuffle",
    ),
    (
        "gfx950_conv_fwd_dtypebf16_G32_N2_Hi17_Wi17_C32_K32_Y3_X3_sH2_sW2_pH1_pW1_dH1_dW1_tile_k128",
        "hkp_conv_fwd_gfx950_fc5a22f33b709890a0180f45_N2H17W17C32_K32Y3X3G32_t64x64x128_w2x2_a32x32x16_mem_cshuffle",
    ),
    (
        "gfx950_conv_fwd_dtypefp16_G5_N2_Hi9_Wi9_C5_K5_Y7_X7_sH1_sW1_pH3_pW3_dH1_dW1_tile_k64",
        "hkp_conv_fwd_gfx950_29ccf81e7bdbb6b5b03b37e5_N2H9W9C5_K5Y7X7G5_t64x64x64_w2x2_a32x32x16_mem_default",
    ),
    (
        "gfx950_conv_fwd_dtypefp16_G5_N2_Hi9_Wi9_C5_K5_Y7_X7_sH1_sW1_pH3_pW3_dH1_dW1_tile_k128",
        "hkp_conv_fwd_gfx950_5549632cc171722aa6f97d95_N2H9W9C5_K5Y7X7G5_t64x64x128_w2x2_a32x32x16_mem_default",
    ),
    (
        "gfx950_conv_fwd_dtypebf16_G5_N2_Hi9_Wi9_C5_K5_Y7_X7_sH1_sW1_pH3_pW3_dH1_dW1_tile_k64",
        "hkp_conv_fwd_gfx950_badaa56b858c09558a63f43a_N2H9W9C5_K5Y7X7G5_t64x64x64_w2x2_a32x32x16_mem_default",
    ),
    (
        "gfx950_conv_fwd_dtypebf16_G5_N2_Hi9_Wi9_C5_K5_Y7_X7_sH1_sW1_pH3_pW3_dH1_dW1_tile_k128",
        "hkp_conv_fwd_gfx950_50445ab7434ebf2d84a206fd_N2H9W9C5_K5Y7X7G5_t64x64x128_w2x2_a32x32x16_mem_default",
    ),
    (
        "gfx950_conv_fwd_dtypefp16_G16_N2_Hi12_Wi12_C16_K32_Y3_X3_sH1_sW1_pH1_pW1_dH1_dW1_tile_k64",
        "hkp_conv_fwd_gfx950_af2c0e74a45d755eb1837c77_N2H12W12C16_K32Y3X3G16_t64x64x64_w2x2_a32x32x16_mem_cshuffle",
    ),
    (
        "gfx950_conv_fwd_dtypefp16_G16_N2_Hi12_Wi12_C16_K32_Y3_X3_sH1_sW1_pH1_pW1_dH1_dW1_tile_k128",
        "hkp_conv_fwd_gfx950_aa4759d2f99f17d0bfedd305_N2H12W12C16_K32Y3X3G16_t64x64x128_w2x2_a32x32x16_mem_cshuffle",
    ),
    (
        "gfx950_conv_fwd_dtypebf16_G16_N2_Hi12_Wi12_C16_K32_Y3_X3_sH1_sW1_pH1_pW1_dH1_dW1_tile_k64",
        "hkp_conv_fwd_gfx950_e7115fa5daeed2c1cb3f3b3f_N2H12W12C16_K32Y3X3G16_t64x64x64_w2x2_a32x32x16_mem_cshuffle",
    ),
    (
        "gfx950_conv_fwd_dtypebf16_G16_N2_Hi12_Wi12_C16_K32_Y3_X3_sH1_sW1_pH1_pW1_dH1_dW1_tile_k128",
        "hkp_conv_fwd_gfx950_c1fee444660ea083c3c26825_N2H12W12C16_K32Y3X3G16_t64x64x128_w2x2_a32x32x16_mem_cshuffle",
    ),
    (
        "gfx950_conv_fwd_dtypefp16_G2_N2_Hi15_Wi15_C16_K16_Y5_X5_sH1_sW1_pH4_pW4_dH2_dW2_tile_k64",
        "hkp_conv_fwd_gfx950_7a3fad912909cea55e77329e_N2H15W15C16_K16Y5X5G2_t64x64x64_w2x2_a32x32x16_mem_cshuffle",
    ),
    (
        "gfx950_conv_fwd_dtypefp16_G2_N2_Hi15_Wi15_C16_K16_Y5_X5_sH1_sW1_pH4_pW4_dH2_dW2_tile_k128",
        "hkp_conv_fwd_gfx950_fdccfed395de2b9a62563a0f_N2H15W15C16_K16Y5X5G2_t64x64x128_w2x2_a32x32x16_mem_cshuffle",
    ),
    (
        "gfx950_conv_fwd_dtypebf16_G2_N2_Hi15_Wi15_C16_K16_Y5_X5_sH1_sW1_pH4_pW4_dH2_dW2_tile_k64",
        "hkp_conv_fwd_gfx950_25583cd94a7698e572d3024b_N2H15W15C16_K16Y5X5G2_t64x64x64_w2x2_a32x32x16_mem_cshuffle",
    ),
    (
        "gfx950_conv_fwd_dtypebf16_G2_N2_Hi15_Wi15_C16_K16_Y5_X5_sH1_sW1_pH4_pW4_dH2_dW2_tile_k128",
        "hkp_conv_fwd_gfx950_9f166bb6d6460141c741bf90_N2H15W15C16_K16Y5X5G2_t64x64x128_w2x2_a32x32x16_mem_cshuffle",
    ),
)


def _committed_kernel_descriptors() -> list:
    if not _KDP_PATH.is_file():
        pytest.skip("the hip-kernel-provider descriptor tree is not beside rocKE")
    return json.loads(_KDP_PATH.read_text())["kernelDescriptors"]


def test_first_shipped_descriptors_keep_order_spec_and_symbol():
    kernels = _committed_kernel_descriptors()
    assert [k["name"] for k in kernels[: len(_FIRST_SHIPPED_SYMBOLS)]] == [
        name for name, _ in _FIRST_SHIPPED_SYMBOLS
    ]
    for kernel, (name, symbol) in zip(kernels, _FIRST_SHIPPED_SYMBOLS):
        spec_fields = kernel["kernel_source"]["spec"]
        # Family fields stay off these descriptors; the KMD default_value
        # supplies them, so the descriptor bytes are as first shipped.
        assert not set(_FAMILY_FIELDS) & set(spec_fields), name
        assert not set(_FAMILY_FIELDS) & set(kernel["metadata"]), name
        spec = Gfx950ConvFwdSpec(**spec_fields)
        assert spec.kernel_family == KERNEL_FAMILY_IMPLICIT_GEMM
        assert spec.kernel_name() == symbol, name
        assert build_gfx950_conv_fwd(spec, arch="gfx950").name == symbol, name


def test_family_fields_appear_only_on_direct_descriptors():
    for kernel in _committed_kernel_descriptors():
        spec_fields = kernel["kernel_source"]["spec"]
        family = kernel["metadata"].get("kernel_family", KERNEL_FAMILY_IMPLICIT_GEMM)
        if family == KERNEL_FAMILY_IMPLICIT_GEMM:
            assert not set(_FAMILY_FIELDS) & set(spec_fields), kernel["name"]
            assert not set(_FAMILY_FIELDS) & set(kernel["metadata"]), kernel["name"]
        else:
            assert family == KERNEL_FAMILY_DIRECT_DEPTHWISE, kernel["name"]
            assert spec_fields["kernel_family"] == family, kernel["name"]
            spec = Gfx950ConvFwdSpec(**spec_fields)
            assert supports_gfx950_conv_fwd(spec) == (True, "ok"), kernel["name"]


@pytest.mark.parametrize(
    ("overrides", "variant", "block_w", "block_waves"),
    [
        ({"G": 32}, "spatial", 0, 1),
        ({"G": 63, "C": 63, "K": 63}, "spatial", 0, 1),
        ({"G": 64, "C": 64, "K": 64}, "std", 4, 1),
        ({"G": 96, "C": 96, "K": 96}, "std", 4, 1),
    ],
)
def test_direct_default_policy(overrides, variant, block_w, block_waves):
    spec = gfx950_conv_fwd_direct_spec_for_request(_dw_request(**overrides))
    assert (spec.kernel_family, spec.direct_variant) == (
        KERNEL_FAMILY_DIRECT_DEPTHWISE,
        variant,
    )
    assert (spec.block_w, spec.block_waves) == (block_w, block_waves)
    # Implicit-GEMM fields carry fixed placeholders; tile_k 0 keeps a forced
    # tile_k=64/128 knob off every direct kernel.
    assert (spec.tile_m, spec.tile_n, spec.tile_k) == (0, 0, 0)
    assert (spec.warp_m, spec.warp_n) == (0, 0)
    assert (spec.warp_tile_m, spec.warp_tile_n, spec.warp_tile_k) == (0, 0, 0)
    assert (spec.wave_size, spec.pipeline, spec.epilogue) == (64, "none", "none")
    assert supports_gfx950_conv_fwd(spec) == (True, "ok")


def test_direct_block_w_selects_the_variant():
    request = _dw_request()
    assert (
        gfx950_conv_fwd_direct_spec_for_request(request, block_w=0).direct_variant
        == "spatial"
    )
    std = gfx950_conv_fwd_direct_spec_for_request(request, block_w=3, block_waves=2)
    assert (std.direct_variant, std.block_w, std.block_waves) == ("std", 3, 2)


def test_direct_spec_round_trips_through_json():
    spec = gfx950_conv_fwd_direct_spec_for_request(
        _dw_request(G=96, C=96, K=96), block_w=3, block_waves=2
    )
    values = json.loads(json.dumps(asdict(spec)))
    assert Gfx950ConvFwdSpec(**values) == spec
    assert all(type(values[field.name]) in (int, str) for field in fields(spec))
    assert Gfx950ConvFwdSpec(**values).kernel_name() == spec.kernel_name()


# Accepted direct shapes: (request overrides, block_w, block_waves). Covers the
# masked lanes of a partial channel tile, stride 2 with odd and even extents,
# large filters, and tall images.
_DIRECT_ACCEPTED = [
    ({}, None, None),
    ({}, 4, 1),
    ({"G": 96, "C": 96, "K": 96}, 3, 2),
    ({"Hi": 17, "Wi": 17, "stride_h": 2, "stride_w": 2}, None, None),
    (
        {"G": 96, "C": 96, "K": 96, "Hi": 16, "Wi": 18, "stride_h": 2, "stride_w": 2},
        4,
        1,
    ),
    (
        {
            "G": 5,
            "C": 5,
            "K": 5,
            "Hi": 9,
            "Wi": 9,
            "Y": 7,
            "X": 7,
            "pad_h": 3,
            "pad_w": 3,
        },
        0,
        1,
    ),
    ({"G": 64, "C": 64, "K": 64, "Hi": 70, "Wi": 40}, 32, 1),
    (
        {"G": 64, "C": 64, "K": 64, "Hi": 70, "Wi": 40, "stride_h": 2, "stride_w": 2},
        32,
        1,
    ),
    (
        {
            "G": 5,
            "C": 5,
            "K": 5,
            "Hi": 56,
            "Wi": 24,
            "Y": 17,
            "X": 17,
            "pad_h": 8,
            "pad_w": 8,
        },
        0,
        1,
    ),
    ({"G": 32}, 4, 16),
]


@pytest.mark.parametrize(("overrides", "block_w", "block_waves"), _DIRECT_ACCEPTED)
def test_direct_accepted_shapes(overrides, block_w, block_waves):
    spec = gfx950_conv_fwd_direct_spec_for_request(
        _dw_request(**overrides), block_w=block_w, block_waves=block_waves
    )
    assert supports_gfx950_conv_fwd(spec) == (True, "ok")


# Shapes rocKE's direct kernels accept but compute wrong (pad above (K-1)/2
# leaves the last rows unwritten; pad below (K-1)/2 makes the runtime H loop
# write into the next image), plus shapes the kernels cannot express. All of
# them stay on implicit GEMM.
_DIRECT_DECLINED = [
    ({"pad_h": 2, "pad_w": 2}, None, "not covered"),
    (
        {"pad_h": 2, "pad_w": 2, "G": 64, "C": 64, "K": 64, "Hi": 70, "Wi": 40},
        32,
        "not covered",
    ),
    ({"pad_h": 0, "pad_w": 0}, 4, "not covered"),
    (
        {"pad_h": 0, "pad_w": 0, "G": 64, "C": 64, "K": 64, "Hi": 70, "Wi": 40},
        32,
        "not covered",
    ),
    (
        {
            "pad_h": 0,
            "pad_w": 0,
            "G": 5,
            "C": 5,
            "K": 5,
            "Hi": 72,
            "Wi": 24,
            "Y": 17,
            "X": 17,
        },
        0,
        "not covered",
    ),
    (
        {
            "G": 5,
            "C": 5,
            "K": 5,
            "Hi": 56,
            "Wi": 24,
            "Y": 9,
            "X": 9,
            "pad_h": 5,
            "pad_w": 5,
        },
        0,
        "not covered",
    ),
    (
        {"Hi": 17, "Wi": 17, "pad_h": 0, "pad_w": 0, "stride_h": 2, "stride_w": 2},
        4,
        "not covered",
    ),
    ({"Y": 3, "X": 5}, 4, "output width is not covered"),
    ({"G": 16, "C": 16, "K": 32}, 4, "groups == C == K"),
    ({"G": 2}, 4, "groups == C == K"),
    ({"stride_h": 2, "stride_w": 1}, 4, "sH == sW"),
    ({"pad_h": 1, "pad_w": 0}, 4, "pH == pW"),
    ({"dilation_h": 2, "dilation_w": 2, "pad_h": 2, "pad_w": 2}, 4, "dH == dW == 1"),
    ({"G": 64, "C": 64, "K": 64}, 0, "groups < 64"),
    ({}, None, "block_waves must be"),
    ({"N": 65536, "Hi": 3, "Wi": 3, "G": 2, "C": 2, "K": 2}, 4, "grid z"),
    ({"N": 1, "Hi": 3, "Wi": 70000, "G": 2, "C": 2, "K": 2}, 1, "grid x"),
    # Every row and column covered, but rocKE's own forward_padding_reason refuses
    # anything except an odd filter with "same" padding.
    (
        {"Y": 4, "X": 4, "Hi": 16, "Wi": 16, "stride_h": 2, "stride_w": 2},
        4,
        "odd filter extents",
    ),
    (
        {"Y": 2, "X": 2, "pad_h": 0, "pad_w": 0, "stride_h": 2, "stride_w": 2},
        4,
        "odd filter extents",
    ),
    (
        {
            "Y": 3,
            "X": 3,
            "pad_h": 0,
            "pad_w": 0,
            "Hi": 9,
            "Wi": 9,
            "stride_h": 3,
            "stride_w": 3,
        },
        0,
        "'same' padding",
    ),
]


@pytest.mark.parametrize(("overrides", "block_w", "reason"), _DIRECT_DECLINED)
def test_direct_declines_outside_the_guarded_contract(overrides, block_w, reason):
    block_waves = 0 if reason == "block_waves must be" else None
    request = _dw_request(**overrides)
    with pytest.raises(ValueError, match=reason):
        gfx950_conv_fwd_direct_spec_for_request(
            request, block_w=block_w, block_waves=block_waves
        )


def test_shapes_rocke_refuses_for_direct_stay_on_implicit_gemm():
    for overrides, _, reason in _DIRECT_DECLINED[-3:]:
        assert reason in ("odd filter extents", "'same' padding")
        spec = gfx950_conv_fwd_spec_for_request(_dw_request(**overrides))
        assert supports_gfx950_conv_fwd(spec) == (True, "ok")


@pytest.mark.parametrize(
    "request_",
    [
        _request(),
        _request(G=4, C=32, K=64, stride_h=2, stride_w=2, Hi=15, Wi=19),
        _request(N=3, Y=1, X=1, pad_h=0, pad_w=0, C=48, K=80),
    ],
)
@pytest.mark.parametrize("tile_k", [None, 128])
def test_implicit_gemm_launch_values_follow_the_packaged_grid(request_, tile_k):
    # The kernel decodes its workgroup id against p_num_pid_m/n, so they must be the
    # tile counts of the grid the native pack launches.
    spec = gfx950_conv_fwd_spec_for_request(request_, tile_k=tile_k)
    values = spec.launch_values(0, 0, 0, 1, 2, 3)
    assert (values["p_num_pid_n"], values["p_num_pid_m"], spec.groups) == spec.grid()
    expected = ConvArgs.from_problem(
        spec.to_problem(), tile_m=spec.tile_m, tile_n=spec.tile_n
    ).to_launch_values(0, 0, 0, 1, 2, 3)
    assert values == expected
    assert (values["A_bytes"], values["B_bytes"], values["D_bytes"]) == (1, 2, 3)


def test_direct_launch_values_are_rockes_direct_conv_args():
    spec = gfx950_conv_fwd_direct_spec_for_request(
        _dw_request(G=96, C=96, K=96, Hi=16, Wi=18, stride_h=2, stride_w=2), block_w=4
    )
    values = spec.launch_values(0, 0, 0, 1, 2, 3)
    assert values == ConvArgs.from_problem(spec.to_direct_problem()).to_launch_values(
        0, 0, 0, 1, 2, 3
    )
    assert (values["p_Ho"], values["p_Wo"]) == (8, 9)


def test_implicit_gemm_refuses_a_reduction_past_the_24_bit_limit():
    spec = Gfx950ConvFwdSpec(N=1, Hi=8, Wi=8, C=1 << 17, K=8, Y=8, X=8)
    ok, reason = supports_gfx950_conv_fwd(spec)
    assert not ok and "2**23" in reason
    ok, reason = supports_gfx950_conv_fwd(replace(spec, C=(1 << 17) - 8))
    assert "2**23" not in reason


def test_declined_direct_shapes_stay_on_implicit_gemm():
    for overrides, _, reason in _DIRECT_DECLINED[:7]:
        assert reason == "not covered"
        spec = gfx950_conv_fwd_spec_for_request(_dw_request(**overrides))
        assert supports_gfx950_conv_fwd(spec) == (True, "ok")


@pytest.mark.parametrize(
    ("override", "reason"),
    [
        ({"kernel_family": 2}, "kernel_family must be"),
        ({"kernel_family": True}, "kernel_family must be"),
        ({"direct_variant": "none"}, "direct_variant must be"),
        ({"direct_variant": "unknown"}, "direct_variant must be"),
        ({"block_waves": 0}, "block_waves must be"),
        ({"block_waves": 17}, "block_waves must be"),
        ({"block_w": 0}, "block_w must be"),
        ({"block_w": -1}, "block_w must be"),
        ({"direct_variant": "spatial", "block_w": 4}, "derives block_w"),
        ({"tile_k": 64}, "must set tile_k to 0"),
        ({"tile_m": 64}, "must set tile_m to 0"),
        ({"warp_tile_k": 16}, "must set warp_tile_k to 0"),
        ({"wave_size": 32}, "must set wave_size to 64"),
        ({"pipeline": "mem"}, "must set pipeline to 'none'"),
        ({"epilogue": "cshuffle"}, "must set epilogue to 'none'"),
    ],
)
def test_direct_spec_consistency_is_validated(override, reason):
    spec = replace(
        gfx950_conv_fwd_direct_spec_for_request(_dw_request(), block_w=4), **override
    )
    ok, explanation = supports_gfx950_conv_fwd(spec)
    assert not ok
    assert reason in explanation
    with pytest.raises(ValueError, match="invalid packaged forward convolution"):
        build_gfx950_conv_fwd(spec, arch="gfx950")


@pytest.mark.parametrize(
    "override",
    [
        {"direct_variant": "std"},
        {"direct_variant": "spatial"},
        {"block_w": 4},
        {"block_waves": 1},
    ],
)
def test_implicit_gemm_spec_must_leave_direct_fields_at_defaults(override):
    spec = replace(gfx950_conv_fwd_spec_for_request(_request()), **override)
    assert not supports_gfx950_conv_fwd(spec)[0]


@pytest.mark.parametrize(
    "override",
    [
        {"sH": 2, "sW": 2},
        {"pH": 0, "pW": 0, "Hi": 9, "Wi": 9, "Y": 1, "X": 1, "sH": 2, "sW": 2},
        {"Y": 5, "X": 5, "pH": 2, "pW": 2},
        {"Y": 7, "X": 7, "pH": 3, "pW": 3, "sH": 2, "sW": 2, "Hi": 16, "Wi": 16},
        {"block_w": 5},
        {"block_waves": 2},
        {"direct_variant": "spatial", "block_w": 0},
        {"dtype": "bf16"},
        {"N": 3},
    ],
)
def test_direct_symbols_hash_every_field(override):
    # rocKE's own direct names omit filter size, padding and stride.
    spec = gfx950_conv_fwd_direct_spec_for_request(_dw_request(), block_w=4)
    changed = replace(spec, **override)
    assert supports_gfx950_conv_fwd(changed) == (True, "ok"), override
    assert changed.kernel_name() != spec.kernel_name()


def test_rocke_direct_names_collide_without_the_hash():
    spec = gfx950_conv_fwd_direct_spec_for_request(_dw_request(), block_w=4)
    other = replace(spec, Y=5, X=5, pH=2, pW=2)
    plain = [
        DirectDepthwiseSpec(
            problem=_rocke_direct_spec(s).problem, block_w=4
        ).kernel_name()
        for s in (spec, other)
    ]
    assert plain[0] == plain[1]
    assert spec.kernel_name() != other.kernel_name()
    assert spec.kernel_name().startswith("hkp_conv_fwd_dw_gfx950_")


@pytest.mark.parametrize(("overrides", "block_w", "block_waves"), _DIRECT_ACCEPTED)
def test_direct_grid_and_block_match_rocke(overrides, block_w, block_waves):
    spec = gfx950_conv_fwd_direct_spec_for_request(
        _dw_request(**overrides), block_w=block_w, block_waves=block_waves
    )
    rocke_spec = _rocke_direct_spec(spec)
    problem = rocke_spec.problem
    # benchmarks/common/benchmark_direct_conv.py's launch.
    if spec.direct_variant == "spatial":
        expected = (math.ceil(problem.Wo / rocke_spec.block_w), 1, problem.N)
    else:
        expected = (
            math.ceil(problem.Wo / rocke_spec.block_w),
            math.ceil(problem.groups / rocke_spec.block_ch),
            problem.N,
        )
    assert spec.grid() == expected
    assert spec.block() == (rocke_spec.threads_per_block, 1, 1)


@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
@pytest.mark.parametrize(
    ("overrides", "block_w", "block_waves"),
    [
        ({}, 0, 1),
        ({}, 4, 1),
        (
            {
                "G": 96,
                "C": 96,
                "K": 96,
                "Hi": 17,
                "Wi": 17,
                "stride_h": 2,
                "stride_w": 2,
            },
            3,
            2,
        ),
        (
            {
                "G": 64,
                "C": 64,
                "K": 64,
                "Hi": 70,
                "Wi": 40,
                "stride_h": 2,
                "stride_w": 2,
            },
            32,
            1,
        ),
    ],
)
def test_direct_spec_emits_original_kernel(dtype, overrides, block_w, block_waves):
    spec = gfx950_conv_fwd_direct_spec_for_request(
        _dw_request(dtype=dtype, **overrides), block_w=block_w, block_waves=block_waves
    )
    rocke_spec = _rocke_direct_spec(spec)
    packaged = build_gfx950_conv_fwd(spec, arch="gfx950")
    if spec.direct_variant == "spatial":
        original = build_direct_depthwise_spatial(rocke_spec, arch="gfx950")
    else:
        original = build_direct_depthwise(rocke_spec, arch="gfx950")
    assert _lower_kernel_to_llvm_python(
        packaged, arch="gfx950", llvm_flavor="llvm22"
    ) == _lower_kernel_to_llvm_python(original, arch="gfx950", llvm_flavor="llvm22")
    assert packaged.name == spec.kernel_name()
    _assert_aot_abi(packaged, spec, dtype)
    assert packaged.max_workgroup_size == spec.block()[0]


def test_direct_spec_has_no_implicit_gemm_instance_and_vice_versa():
    direct = gfx950_conv_fwd_direct_spec_for_request(_dw_request())
    with pytest.raises(ValueError, match="no implicit-GEMM instance"):
        direct.to_instance_spec()
    with pytest.raises(ValueError, match="no direct depthwise instance"):
        gfx950_conv_fwd_spec_for_request(_dw_request()).to_direct_spec()
