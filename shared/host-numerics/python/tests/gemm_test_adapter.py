# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Test utility implemented with public tensor operations.

The helper contains no fused GEMM implementation. Numerical oracle tests use
``matmul``, elementwise arithmetic, and ``reference_epilogue_into`` as separate
operations.
"""

import numpy as np

import roc_host_numerics as hn


def _broadcast_vector(scale, axis):
    return scale.expand_dims(axis) if len(scale.shape) == 1 else scale


def _scalar_value(value):
    return value.item() if isinstance(value, hn.Tensor) else value


def _scalar_tensor(value, scalar_type):
    if isinstance(value, hn.Tensor):
        return value
    return hn.from_numpy(np.asarray(value)).to(scalar_type)


def _product(
    a,
    b,
    accumulator_type,
    *,
    alpha,
    compute_type_a,
    compute_type_b,
    math_mode,
    block_scale_a,
    block_scale_b,
    block_size_a,
    block_size_b,
    pre_quantization_scales_a,
    pre_quantization_scales_b,
    accumulation_rounding,
    conjugate_a,
    conjugate_b,
    scale_alpha,
    scale_a,
    scale_b,
):
    output_shape = hn.Shape([a.shape[0], b.shape[1]])
    product = hn.Tensor(accumulator_type, output_shape)
    if _scalar_value(alpha) == 0 or a.shape[1] == 0:
        return product

    product = hn.matmul(
        a,
        b,
        accumulator_type,
        accumulator_type,
        compute_type_a=compute_type_a,
        compute_type_b=compute_type_b,
        math_mode=math_mode,
        block_scale_a=block_scale_a,
        block_scale_b=block_scale_b,
        block_size_a=block_size_a,
        block_size_b=block_size_b,
        pre_quantization_scales_a=pre_quantization_scales_a,
        pre_quantization_scales_b=pre_quantization_scales_b,
        accumulation_rounding=accumulation_rounding,
        conjugate_a=conjugate_a,
        conjugate_b=conjugate_b,
    )
    product = hn.multiply(
        product,
        alpha,
        output_type=accumulator_type,
        compute_type=accumulator_type,
    )
    for scale, axis in ((scale_a, 1), (scale_b, 0), (scale_alpha, None)):
        if scale is not None:
            if axis is not None:
                scale = _broadcast_vector(scale, axis)
            product = hn.multiply(
                product,
                scale,
                output_type=accumulator_type,
                compute_type=accumulator_type,
            )
    return product


def _combined_input(product, c, accumulator_type, beta, scale_c):
    if _scalar_value(beta) == 0:
        return product
    coefficient = hn.multiply(
        _scalar_tensor(beta, accumulator_type),
        scale_c,
        output_type=accumulator_type,
        compute_type=accumulator_type,
    )
    addend = hn.multiply(
        c,
        coefficient,
        output_type=accumulator_type,
        compute_type=accumulator_type,
    )
    return hn.add(
        product,
        addend,
        output_type=accumulator_type,
        compute_type=accumulator_type,
    )


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
    output = hn.Tensor(
        output_type,
        output_layout
        if output_layout is not None
        else hn.Shape([a.shape[0], b.shape[1]]),
    )
    reference_gemm_into(
        a,
        b,
        c,
        output,
        accumulator_type,
        alpha,
        beta,
        scale_c,
        compute_type_a,
        compute_type_b,
        math_mode,
        output_selection,
        backend,
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
    )
    return output


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
    if backend not in (hn.GemmBackend.Automatic, hn.GemmBackend.Blocked):
        raise ValueError("Python matmul exposes the built-in blocked backend")
    selection = hn.OutputSelection.all() if output_selection is None else output_selection
    product = _product(
        a,
        b,
        accumulator_type,
        alpha=alpha,
        compute_type_a=compute_type_a,
        compute_type_b=compute_type_b,
        math_mode=math_mode,
        block_scale_a=block_scale_a,
        block_scale_b=block_scale_b,
        block_size_a=block_size_a,
        block_size_b=block_size_b,
        pre_quantization_scales_a=pre_quantization_scales_a,
        pre_quantization_scales_b=pre_quantization_scales_b,
        accumulation_rounding=accumulation_rounding,
        conjugate_a=conjugate_a,
        conjugate_b=conjugate_b,
        scale_alpha=scale_alpha,
        scale_a=scale_a,
        scale_b=scale_b,
    )
    combined = _combined_input(product, c, accumulator_type, beta, scale_c)
    hn.reference_epilogue_into(
        combined,
        d,
        accumulator_type,
        bias=bias,
        output_scale=output_scale,
        output_conversion=output_conversion,
        output_selection=selection,
    )
