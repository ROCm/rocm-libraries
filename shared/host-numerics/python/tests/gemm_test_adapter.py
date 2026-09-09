# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Test-only access to the internal fused GEMM implementation.

Public users compose ``matmul`` with tensor and epilogue operations. These
helpers retain coverage of the fused implementation until numerical and
performance evidence makes it safe to remove internally as well.
"""

import roc_host_numerics as hn
from roc_host_numerics import _roc_host_numerics as _native


def _broadcast_vector(scale, axis):
    return scale.expand_dims(axis) if len(scale.shape) == 1 else scale


def _options(
    accumulator_type,
    alpha,
    beta,
    scale_c,
    compute_type_a,
    compute_type_b,
    math_mode,
    activation,
    activation_parameter0,
    activation_parameter1,
    output_selection,
    block_scale_a,
    block_scale_b,
    block_size_a,
    block_size_b,
    pre_quantization_scales_a,
    pre_quantization_scales_b,
    bias,
    scale_alpha,
    scale_a,
    scale_b,
    output_scale,
    output_conversion,
    accumulation_rounding,
    conjugate_a,
    conjugate_b,
):
    options = _native._GemmOptions(accumulator_type)
    options.accumulation_rounding = accumulation_rounding
    options.math_mode = math_mode
    options.compute_type_a = compute_type_a
    options.compute_type_b = compute_type_b
    options.pre_quantization_scales_a = [
        _broadcast_vector(scale, 1)
        for scale in (
            [] if pre_quantization_scales_a is None else pre_quantization_scales_a
        )
    ]
    options.pre_quantization_scales_b = [
        _broadcast_vector(scale, 0)
        for scale in (
            [] if pre_quantization_scales_b is None else pre_quantization_scales_b
        )
    ]
    options.block_scale_a = block_scale_a
    options.block_scale_b = block_scale_b
    options.block_size_a = block_size_a
    options.block_size_b = block_size_b
    options.conjugate_a = conjugate_a
    options.conjugate_b = conjugate_b
    options.alpha = alpha
    options.beta = beta
    options.scale_c = scale_c
    options.bias = bias
    options.scale_alpha = scale_alpha
    options.scale_a = None if scale_a is None else _broadcast_vector(scale_a, 1)
    options.scale_b = None if scale_b is None else _broadcast_vector(scale_b, 0)
    options.output_scale = output_scale
    options.output_conversion = output_conversion
    options.activation = activation
    options.activation_parameter0 = activation_parameter0
    options.activation_parameter1 = activation_parameter1
    options.output_selection = (
        hn.OutputSelection.all() if output_selection is None else output_selection
    )
    return options


def reference_gemm(
    a,
    b,
    c,
    output_type,
    accumulator_type,
    alpha=1.0,
    beta=0.0,
    scale_c=1.0,
    compute_type_a=None,
    compute_type_b=None,
    math_mode=hn.MathMode.Default,
    activation=hn.Activation.None_,
    activation_parameter0=0.0,
    activation_parameter1=0.0,
    output_selection=None,
    backend=hn.GemmBackend.Automatic,
    block_scale_a=None,
    block_scale_b=None,
    block_size_a=0,
    block_size_b=0,
    pre_quantization_scales_a=None,
    pre_quantization_scales_b=None,
    bias=None,
    scale_alpha=None,
    scale_a=None,
    scale_b=None,
    output_scale=1.0,
    output_conversion=hn.OutputConversion.Default,
    accumulation_rounding=hn.AccumulationRounding.TypeDefault,
    conjugate_a=False,
    conjugate_b=False,
    output_layout=None,
):
    return _native._reference_gemm(
        a,
        b,
        c,
        output_type,
        _options(
            accumulator_type,
            alpha,
            beta,
            scale_c,
            compute_type_a,
            compute_type_b,
            math_mode,
            activation,
            activation_parameter0,
            activation_parameter1,
            output_selection,
            block_scale_a,
            block_scale_b,
            block_size_a,
            block_size_b,
            pre_quantization_scales_a,
            pre_quantization_scales_b,
            bias,
            scale_alpha,
            scale_a,
            scale_b,
            output_scale,
            output_conversion,
            accumulation_rounding,
            conjugate_a,
            conjugate_b,
        ),
        output_layout,
        backend,
    )


def reference_gemm_into(
    a,
    b,
    c,
    d,
    accumulator_type=hn.ScalarType.Float32,
    alpha=1.0,
    beta=0.0,
    scale_c=1.0,
    compute_type_a=None,
    compute_type_b=None,
    math_mode=hn.MathMode.Default,
    activation=hn.Activation.None_,
    activation_parameter0=0.0,
    activation_parameter1=0.0,
    output_selection=None,
    backend=hn.GemmBackend.Automatic,
    block_scale_a=None,
    block_scale_b=None,
    block_size_a=0,
    block_size_b=0,
    pre_quantization_scales_a=None,
    pre_quantization_scales_b=None,
    bias=None,
    scale_alpha=None,
    scale_a=None,
    scale_b=None,
    output_scale=1.0,
    output_conversion=hn.OutputConversion.Default,
    accumulation_rounding=hn.AccumulationRounding.TypeDefault,
    conjugate_a=False,
    conjugate_b=False,
):
    return _native._reference_gemm_into(
        a,
        b,
        c,
        d,
        _options(
            accumulator_type,
            alpha,
            beta,
            scale_c,
            compute_type_a,
            compute_type_b,
            math_mode,
            activation,
            activation_parameter0,
            activation_parameter1,
            output_selection,
            block_scale_a,
            block_scale_b,
            block_size_a,
            block_size_b,
            pre_quantization_scales_a,
            pre_quantization_scales_b,
            bias,
            scale_alpha,
            scale_a,
            scale_b,
            output_scale,
            output_conversion,
            accumulation_rounding,
            conjugate_a,
            conjugate_b,
        ),
        backend,
    )
