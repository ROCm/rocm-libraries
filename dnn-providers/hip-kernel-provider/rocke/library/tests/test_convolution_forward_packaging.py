# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""CPU tests for descriptor hydration and the unmodified convolution builder ABI."""

from __future__ import annotations

import inspect
import json
import typing
from dataclasses import asdict, fields, replace

import pytest
from builders.common.convolution_forward import (
    Gfx950ConvFwdSpec,
    build_gfx950_conv_fwd,
    gfx950_conv_fwd_spec_for_request,
    supports_gfx950_conv_fwd,
)
from dispatch.grouped_convolution import ConvGroupedRequest, dispatch_conv_grouped
from rocke.core.ir import BF16, F16, I32, PtrType
from rocke.core.lower_llvm import _lower_kernel_to_llvm_python
from rocke.instances.common.conv_implicit_gemm import (
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
def test_adapter_emits_original_llvm_and_six_argument_abi(dtype, tile_k, llvm_flavor):
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
    assert [p.name for p in packaged.params] == [
        "A",
        "B",
        "D",
        "A_bytes",
        "B_bytes",
        "D_bytes",
    ]
    element = F16 if dtype == "fp16" else BF16
    expected_types = [PtrType(element, "global")] * 3 + [I32] * 3
    assert [p.type for p in packaged.params] == expected_types
    assert all(p.attrs["align"] == 16 for p in packaged.params[:3])
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
        ({"groups": 2}, "groups=1"),
        ({"layout": "NCHW"}, "NHWC"),
        ({"dtype": "fp32"}, "fp16 or bf16"),
        ({"tile_k": 0}, "tile_k must be"),
        ({"warp_m": 0}, "warp_m must be"),
        ({"tile_m": 96}, "not divisible"),
        ({"wave_size": 32}, "wave_size"),
        ({"pipeline": "wavelet"}, "pipeline='mem'"),
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
        ({"G": 2}, "groups=1"),
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
