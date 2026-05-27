# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""FLOP handlers for the three convolution directions.

All three directions do the same total scalar work; they only differ
in which tensor is being produced. The per-direction handlers each
resolve the right UID-keyed tensor as ``x`` (activation-shaped input),
``w`` (weight), ``y`` (activation-shaped output) and dispatch to a
single arithmetic implementation.

MIOpen formula (matches ``conv_driver.hpp:1750-1751``)::

    2 * N * C_in * R * S * K * H_out * W_out / group
"""

from typing import Any, Dict, Optional


def _conv_flops_impl(
    x: Optional[Dict[str, Any]],
    w: Optional[Dict[str, Any]],
    y: Optional[Dict[str, Any]],
    params: Dict[str, Any],
) -> Optional[int]:
    """Compute conv FLOPs given resolved input / weight / output tensors."""
    if not x or not w or not y:
        return None
    x_dims = x.get("dims") or []
    w_dims = w.get("dims") or []
    y_dims = y.get("dims") or []
    if len(x_dims) < 4 or len(w_dims) < 4 or len(y_dims) < 4:
        return None
    spatial_w = w_dims[2:]
    spatial_y = y_dims[2:]
    if not spatial_w or not spatial_y:
        return None

    # NCHW / NCDHW: dim 0 = N, dim 1 = C; for weight K = dim 0, C/g = dim 1.
    n = int(x_dims[0])
    c_in = int(x_dims[1])
    k = int(w_dims[0])

    weight_spatial = 1
    for d in spatial_w:
        weight_spatial *= int(d)
    output_spatial = 1
    for d in spatial_y:
        output_spatial *= int(d)

    group_count = int(params.get("group_count", 1)) or 1

    return 2 * n * c_in * weight_spatial * k * output_spatial // group_count


# TODO: lift the string-keyed UID lookups in the per-direction handlers
# below into a structural pass over the parsed graph once a Graph/Node
# abstraction is available. Today every handler reaches into the raw
# JSON and probes hipDNN-specific input/output names by string, which
# couples us to the exact key strings emitted by hipDNN's frontend.
#
# Cannot use ``or`` chains for UID lookups: hipDNN tensor UIDs start at
# 0 and ``0 or fallback`` evaluates to fallback, masking the real UID.


def conv_fwd_flops(
    node: Dict[str, Any], tensors_by_uid: Dict[int, Dict[str, Any]]
) -> Optional[int]:
    """ConvolutionFwdAttributes: ``y = conv(x, w)``."""
    inputs = node.get("inputs", {}) or {}
    outputs = node.get("outputs", {}) or {}
    x_uid = inputs.get("x_tensor_uid")
    w_uid = inputs.get("w_tensor_uid")
    y_uid = outputs.get("y_tensor_uid")
    if x_uid is None or w_uid is None or y_uid is None:
        return None
    return _conv_flops_impl(
        x=tensors_by_uid.get(int(x_uid)),
        w=tensors_by_uid.get(int(w_uid)),
        y=tensors_by_uid.get(int(y_uid)),
        params=node.get("parameters", {}) or {},
    )


def conv_dgrad_flops(
    node: Dict[str, Any], tensors_by_uid: Dict[int, Dict[str, Any]]
) -> Optional[int]:
    """ConvolutionBwdAttributes (dgrad): ``dx = conv_transposed(dy, w)``.

    The output ``dx`` has the activation-input shape ``[N, C, H_in, W_in]``
    and ``dy`` has the activation-output shape ``[N, K, H_out, W_out]``.
    Map them onto the fwd-shaped impl: dx -> x, dy -> y.
    """
    inputs = node.get("inputs", {}) or {}
    outputs = node.get("outputs", {}) or {}
    dy_uid = inputs.get("dy_tensor_uid")
    w_uid = inputs.get("w_tensor_uid")
    dx_uid = outputs.get("dx_tensor_uid")
    if dy_uid is None or w_uid is None or dx_uid is None:
        return None
    return _conv_flops_impl(
        x=tensors_by_uid.get(int(dx_uid)),
        w=tensors_by_uid.get(int(w_uid)),
        y=tensors_by_uid.get(int(dy_uid)),
        params=node.get("parameters", {}) or {},
    )


def conv_wgrad_flops(
    node: Dict[str, Any], tensors_by_uid: Dict[int, Dict[str, Any]]
) -> Optional[int]:
    """ConvolutionWrwAttributes (wgrad): ``dw = conv(x, dy)``.

    Output ``dw`` has the weight shape ``[K, C/G, R, S]`` and ``dy``
    has the activation-output shape. Map onto the fwd-shaped impl:
    x -> x, dw -> w, dy -> y.
    """
    inputs = node.get("inputs", {}) or {}
    outputs = node.get("outputs", {}) or {}
    x_uid = inputs.get("x_tensor_uid")
    dy_uid = inputs.get("dy_tensor_uid")
    dw_uid = outputs.get("dw_tensor_uid")
    if x_uid is None or dy_uid is None or dw_uid is None:
        return None
    return _conv_flops_impl(
        x=tensors_by_uid.get(int(x_uid)),
        w=tensors_by_uid.get(int(dw_uid)),
        y=tensors_by_uid.get(int(dy_uid)),
        params=node.get("parameters", {}) or {},
    )
