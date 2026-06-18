# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Layernorm and RMSNorm reference handlers."""

from typing import Any, Dict, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from .._common import *  # noqa: F401,F403
from .._registry import register_handler


def _shape_is_channel_affine(
    scale_shape: Sequence[int], x_shape: Sequence[int]
) -> bool:
    scale = tuple(int(dim) for dim in scale_shape)
    x = tuple(int(dim) for dim in x_shape)
    if len(x) < 3 or len(scale) <= 1:
        return False
    if _numel(scale) != x[1]:
        return False

    non_singletons = [idx for idx, dim in enumerate(scale) if dim != 1]
    if len(non_singletons) != 1:
        return False

    idx = non_singletons[0]
    if len(scale) == len(x):
        return idx == 1
    if len(scale) == len(x) - 1:
        return idx == 0
    return False


def _infer_trailing_normalized_count(
    x: torch.Tensor,
    *affine_tensors: Optional[torch.Tensor],
) -> int:
    for tensor in affine_tensors:
        if tensor is None:
            continue
        stripped = _strip_leading_singletons(tensor.shape)
        if (
            stripped
            and len(stripped) <= x.ndim
            and tuple(x.shape[-len(stripped) :]) == stripped
        ):
            return len(stripped)

        elements = tensor.numel()
        for count in range(1, x.ndim + 1):
            if _numel(x.shape[-count:]) == elements:
                return count

    raise ValueError("Unable to infer normalized dimensions from affine tensors")


def _layernorm_normalized_shape(
    node: Dict[str, Any],
    x: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor,
) -> Tuple[int, ...]:
    count = int(_node_param(node, "normalized_dim_count", 0) or 0)
    if count <= 0:
        count = _infer_trailing_normalized_count(x, scale, bias)
    if count < 1 or count > x.ndim:
        raise ValueError(
            f"Layernorm normalized_dim_count={count} is invalid for rank {x.ndim}"
        )
    return tuple(int(dim) for dim in x.shape[-count:])


def _reshape_affine_for_normalized_shape(
    tensor: torch.Tensor,
    normalized_shape: Sequence[int],
    x: torch.Tensor,
    name: str,
) -> torch.Tensor:
    shape = tuple(int(dim) for dim in normalized_shape)
    value = tensor.to(dtype=torch.float32, device=x.device)
    if tuple(value.shape) == shape:
        return value
    if _strip_leading_singletons(value.shape) == shape:
        return value.reshape(shape)
    if value.numel() == _numel(shape):
        return value.reshape(shape)
    raise ValueError(
        f"{name} tensor shape {tuple(tensor.shape)} is not compatible with "
        f"normalized shape {shape}"
    )


def _reshape_affine_for_broadcast(
    tensor: torch.Tensor,
    broadcast_shape: Sequence[int],
    x: torch.Tensor,
    name: str,
) -> torch.Tensor:
    shape = tuple(int(dim) for dim in broadcast_shape)
    value = tensor.to(dtype=torch.float32, device=x.device)
    if tuple(value.shape) == shape:
        return value
    if value.numel() == _numel(shape):
        return value.reshape(shape)
    try:
        return torch.broadcast_to(value, shape)
    except RuntimeError as e:
        raise ValueError(
            f"{name} tensor shape {tuple(tensor.shape)} is not broadcastable to {shape}"
        ) from e


def _rmsnorm_layout_from_shapes(
    x_shape: Sequence[int], scale_shape: Sequence[int]
) -> Tuple[str, Tuple[int, ...], Tuple[int, ...], Optional[Tuple[int, ...]]]:
    x = tuple(int(dim) for dim in x_shape)
    scale = tuple(int(dim) for dim in scale_shape)

    if _shape_is_channel_affine(scale, x):
        broadcast_shape = (1, x[1], *([1] * (len(x) - 2)))
        reduce_dims = tuple(range(2, len(x)))
        return "channel", reduce_dims, broadcast_shape, None

    stripped = _strip_leading_singletons(scale)
    if stripped and len(stripped) <= len(x) and tuple(x[-len(stripped) :]) == stripped:
        reduce_dims = tuple(range(len(x) - len(stripped), len(x)))
        broadcast_shape = (*([1] * (len(x) - len(stripped))), *stripped)
        return "trailing", reduce_dims, broadcast_shape, stripped

    elements = _numel(scale)
    for count in range(1, len(x) + 1):
        trailing = x[-count:]
        if _numel(trailing) == elements:
            reduce_dims = tuple(range(len(x) - count, len(x)))
            broadcast_shape = (*([1] * (len(x) - count)), *trailing)
            return "trailing", reduce_dims, broadcast_shape, trailing

    raise ValueError(
        f"RMSNorm scale shape {scale} is not compatible with input shape {x}"
    )


def _rmsnorm_graph_can_use_builtin(
    node: Dict[str, Any], graph_json: Dict[str, Any]
) -> bool:
    if not hasattr(F, "rms_norm"):
        return False
    x_uid = _node_uid(node, "x_tensor_uid", ("inputs",), required=False)
    scale_uid = _node_uid(node, "scale_tensor_uid", ("inputs",), required=False)
    if x_uid is None or scale_uid is None:
        return False
    x_shape = _tensor_shape(graph_json, int(x_uid))
    scale_shape = _tensor_shape(graph_json, int(scale_uid))
    if x_shape is None or scale_shape is None:
        return False
    try:
        layout, _dims, _broadcast_shape, normalized_shape = _rmsnorm_layout_from_shapes(
            x_shape, scale_shape
        )
    except ValueError:
        return False
    return layout == "trailing" and normalized_shape is not None


def _rmsnorm_layout(
    x: torch.Tensor, scale: torch.Tensor
) -> Tuple[str, Tuple[int, ...], Tuple[int, ...], Optional[Tuple[int, ...]]]:
    return _rmsnorm_layout_from_shapes(x.shape, scale.shape)


@register_handler("LayernormAttributes")
def handle_layernorm(
    node: Dict[str, Any],
    tensors: Dict[int, torch.Tensor],
    graph_json: Dict[str, Any],
) -> None:
    """Handle layer normalization over trailing normalized dimensions."""
    x_uid = _required_input_uid(node, "x_tensor_uid")
    scale_uid = _required_input_uid(node, "scale_tensor_uid")
    bias_uid = _required_input_uid(node, "bias_tensor_uid")
    epsilon_uid = _required_input_uid(node, "epsilon_tensor_uid")
    y_uid = _required_output_uid(node, "y_tensor_uid")

    x = _tensor(tensors, x_uid, node)
    x_float = x.to(dtype=torch.float32)
    scale = _tensor(tensors, scale_uid, node)
    bias = _tensor(tensors, bias_uid, node)
    epsilon = _scalar_value(tensors, epsilon_uid, node)

    normalized_shape = _layernorm_normalized_shape(node, x, scale, bias)
    weight = _reshape_affine_for_normalized_shape(
        scale, normalized_shape, x, "Layernorm scale"
    )
    bias_value = _reshape_affine_for_normalized_shape(
        bias, normalized_shape, x, "Layernorm bias"
    )

    y = F.layer_norm(
        x_float,
        normalized_shape,
        weight=weight,
        bias=bias_value,
        eps=epsilon,
    ).to(dtype=x.dtype)
    _store_tensor_for_uid(tensors, graph_json, y_uid, y)

    reduce_dims = tuple(range(x.ndim - len(normalized_shape), x.ndim))
    mean_uid = _optional_uid(node, "mean_tensor_uid")
    inv_uid = _optional_uid(node, "inv_variance_tensor_uid")
    if mean_uid is not None or inv_uid is not None:
        mean = x_float.mean(dim=reduce_dims, keepdim=True)
        variance = x_float.var(dim=reduce_dims, unbiased=False, keepdim=True)
        if mean_uid is not None:
            _store_tensor_for_uid(tensors, graph_json, int(mean_uid), mean)
        if inv_uid is not None:
            _store_tensor_for_uid(
                tensors,
                graph_json,
                int(inv_uid),
                torch.rsqrt(variance + epsilon),
            )


@register_handler("RMSNormAttributes")
def handle_rmsnorm(
    node: Dict[str, Any],
    tensors: Dict[int, torch.Tensor],
    graph_json: Dict[str, Any],
) -> None:
    """Handle RMSNorm forward with trailing or per-channel affine layout."""
    x_uid = _required_input_uid(node, "x_tensor_uid")
    scale_uid = _required_input_uid(node, "scale_tensor_uid")
    epsilon_uid = _required_input_uid(node, "epsilon_tensor_uid")
    y_uid = _required_output_uid(node, "y_tensor_uid")

    x = _tensor(tensors, x_uid, node)
    x_float = x.to(dtype=torch.float32)
    scale = _tensor(tensors, scale_uid, node)
    epsilon = _scalar_value(tensors, epsilon_uid, node)
    layout, reduce_dims, broadcast_shape, normalized_shape = _rmsnorm_layout(x, scale)

    use_builtin = (
        layout == "trailing" and normalized_shape is not None and hasattr(F, "rms_norm")
    )
    if use_builtin:
        weight = _reshape_affine_for_normalized_shape(
            scale, normalized_shape, x, "RMSNorm scale"
        )
        y_float = F.rms_norm(x_float, normalized_shape, weight=weight, eps=epsilon)
        inv_rms = None
    else:
        scale_b = _reshape_affine_for_broadcast(
            scale, broadcast_shape, x, "RMSNorm scale"
        )
        inv_rms = torch.rsqrt(
            x_float.square().mean(dim=reduce_dims, keepdim=True) + epsilon
        )
        y_float = x_float * inv_rms * scale_b

    bias_uid = _optional_uid(node, "bias_tensor_uid")
    if bias_uid is not None:
        bias = _tensor(tensors, int(bias_uid), node)
        y_float = y_float + _reshape_affine_for_broadcast(
            bias, broadcast_shape, x, "RMSNorm bias"
        )

    y = y_float.to(dtype=x.dtype)
    _store_tensor_for_uid(tensors, graph_json, y_uid, y)

    inv_uid = _optional_uid(node, "inv_rms_tensor_uid")
    if inv_uid is not None:
        if inv_rms is None:
            inv_rms = torch.rsqrt(
                x_float.square().mean(dim=reduce_dims, keepdim=True) + epsilon
            )
        _store_tensor_for_uid(tensors, graph_json, int(inv_uid), inv_rms)


@register_handler("RMSNormBackwardAttributes")
def handle_rmsnorm_backward(
    node: Dict[str, Any],
    tensors: Dict[int, torch.Tensor],
    graph_json: Dict[str, Any],
) -> None:
    """Handle RMSNorm backward using the saved inverse RMS tensor."""
    dy_uid = _required_input_uid(node, "dy_tensor_uid")
    x_uid = _required_input_uid(node, "x_tensor_uid")
    scale_uid = _required_input_uid(node, "scale_tensor_uid")
    inv_uid = _required_input_uid(node, "inv_rms_tensor_uid")
    dx_uid = _required_output_uid(node, "dx_tensor_uid")
    dscale_uid = _required_output_uid(node, "dscale_tensor_uid")

    dy = _tensor(tensors, dy_uid, node).to(dtype=torch.float32)
    x = _tensor(tensors, x_uid, node)
    x_float = x.to(dtype=torch.float32)
    scale = _tensor(tensors, scale_uid, node)
    inv_rms = _tensor(tensors, inv_uid, node).to(dtype=torch.float32, device=x.device)

    _layout, reduce_dims, broadcast_shape, _normalized_shape = _rmsnorm_layout(x, scale)
    scale_b = _reshape_affine_for_broadcast(scale, broadcast_shape, x, "RMSNorm scale")
    weighted_dy = dy * scale_b
    if reduce_dims:
        dot = (weighted_dy * x_float).sum(dim=reduce_dims, keepdim=True)
        elements = _numel([x.shape[dim] for dim in reduce_dims])
    else:
        dot = weighted_dy * x_float
        elements = 1

    dx = (weighted_dy * inv_rms - x_float * inv_rms.pow(3) * dot / float(elements)).to(
        dtype=x.dtype
    )
    _store_tensor_for_uid(tensors, graph_json, dx_uid, dx)

    dscale = _sum_to_shape(dy * x_float * inv_rms, scale.shape).to(dtype=scale.dtype)
    _store_tensor_for_uid(tensors, graph_json, dscale_uid, dscale)

    dbias_uid = _optional_uid(node, "dbias_tensor_uid")
    if dbias_uid is not None:
        dbias_shape = _stored_tensor_shape(tensors, graph_json, int(dbias_uid))
        if dbias_shape is None:
            dbias_shape = tuple(int(dim) for dim in scale.shape)
        dbias = _sum_to_shape(dy, dbias_shape).to(dtype=scale.dtype)
        _store_tensor_for_uid(tensors, graph_json, int(dbias_uid), dbias)
