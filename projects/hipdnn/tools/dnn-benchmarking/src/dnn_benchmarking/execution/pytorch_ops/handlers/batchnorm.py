# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Batch-norm reference handlers."""

from typing import Any, Dict, Tuple

import torch
import torch.nn.functional as F

from .._common import *  # noqa: F401,F403
from .._registry import register_handler


def _bn_reduce_dims(x: torch.Tensor) -> Tuple[int, ...]:
    if x.ndim < 2:
        raise ValueError("Batchnorm requires at least 2D tensor (batch and channel)")
    return tuple(dim for dim in range(x.ndim) if dim != 1)


def _bn_mean_var(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    x_float = x.to(dtype=torch.float32)
    reduce_dims = _bn_reduce_dims(x)
    mean = x_float.mean(dim=reduce_dims)
    mean_sq = (x_float * x_float).mean(dim=reduce_dims)
    var = mean_sq - mean * mean
    return mean, var


def _bn_affine(
    x: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor,
    mean: torch.Tensor,
    inv_variance: torch.Tensor,
) -> torch.Tensor:
    x_float = x.to(dtype=torch.float32)
    scale_b = _channel_broadcast(scale, x_float)
    bias_b = _channel_broadcast(bias, x_float)
    mean_b = _channel_broadcast(mean, x_float)
    inv_b = _channel_broadcast(inv_variance, x_float)
    return (scale_b * ((x_float - mean_b) * inv_b) + bias_b).to(dtype=x.dtype)


@register_handler("BatchnormInferenceAttributes")
def handle_batchnorm_inference(
    node: Dict[str, Any],
    tensors: Dict[int, torch.Tensor],
    graph_json: Dict[str, Any],
) -> None:
    """Handle batchnorm inference with precomputed inverse variance."""
    x_uid = _required_input_uid(node, "x_tensor_uid")
    mean_uid = _required_input_uid(node, "mean_tensor_uid")
    inv_uid = _required_input_uid(node, "inv_variance_tensor_uid")
    scale_uid = _required_input_uid(node, "scale_tensor_uid")
    bias_uid = _required_input_uid(node, "bias_tensor_uid")
    y_uid = _required_output_uid(node, "y_tensor_uid")

    x = _tensor(tensors, x_uid, node)
    y = _bn_affine(
        x,
        _channel_values(_tensor(tensors, scale_uid, node), x),
        _channel_values(_tensor(tensors, bias_uid, node), x),
        _channel_values(_tensor(tensors, mean_uid, node), x),
        _channel_values(_tensor(tensors, inv_uid, node), x),
    )
    _store_tensor(tensors, y_uid, y)


@register_handler("BatchnormInferenceAttributesVarianceExt")
def handle_batchnorm_inference_variance(
    node: Dict[str, Any],
    tensors: Dict[int, torch.Tensor],
    graph_json: Dict[str, Any],
) -> None:
    """Handle batchnorm inference with variance and epsilon.

    Maps directly onto ``torch.nn.functional.batch_norm`` in eval mode, so the
    timed reference row measures the PyTorch batchnorm primitive rather than
    hand-rolled elementwise glue.
    """
    x_uid = _required_input_uid(node, "x_tensor_uid")
    mean_uid = _required_input_uid(node, "mean_tensor_uid")
    variance_uid = _required_input_uid(node, "variance_tensor_uid")
    scale_uid = _required_input_uid(node, "scale_tensor_uid")
    bias_uid = _required_input_uid(node, "bias_tensor_uid")
    epsilon_uid = _required_input_uid(node, "epsilon_tensor_uid")
    y_uid = _required_output_uid(node, "y_tensor_uid")

    x = _tensor(tensors, x_uid, node)
    x_float = x.to(dtype=torch.float32)
    running_mean = _channel_values(_tensor(tensors, mean_uid, node), x).to(
        torch.float32
    )
    running_var = _channel_values(_tensor(tensors, variance_uid, node), x).to(
        torch.float32
    )
    weight = _channel_values(_tensor(tensors, scale_uid, node), x).to(torch.float32)
    bias = _channel_values(_tensor(tensors, bias_uid, node), x).to(torch.float32)
    epsilon = _scalar_value(tensors, epsilon_uid, node)

    y = F.batch_norm(
        x_float,
        running_mean,
        running_var,
        weight=weight,
        bias=bias,
        training=False,
        eps=epsilon,
    ).to(dtype=x.dtype)
    _store_tensor(tensors, y_uid, y)


@register_handler("BatchnormAttributes")
def handle_batchnorm_training(
    node: Dict[str, Any],
    tensors: Dict[int, torch.Tensor],
    graph_json: Dict[str, Any],
) -> None:
    """Handle batchnorm forward training."""
    _reject_peer_stats(node, "Batchnorm forward training")
    x_uid = _required_input_uid(node, "x_tensor_uid")
    scale_uid = _required_input_uid(node, "scale_tensor_uid")
    bias_uid = _required_input_uid(node, "bias_tensor_uid")
    epsilon_uid = _required_input_uid(node, "epsilon_tensor_uid")
    y_uid = _required_output_uid(node, "y_tensor_uid")

    x = _tensor(tensors, x_uid, node)
    scale = _channel_values(_tensor(tensors, scale_uid, node), x)
    bias = _channel_values(_tensor(tensors, bias_uid, node), x)
    epsilon = _scalar_value(tensors, epsilon_uid, node)
    mean, variance = _bn_mean_var(x)
    inv_variance = torch.rsqrt(variance + epsilon)
    y = _bn_affine(x, scale, bias, mean, inv_variance)
    _store_tensor(tensors, y_uid, y)

    _store_channel_tensor(tensors, _optional_uid(node, "mean_tensor_uid"), mean, x.ndim)
    _store_channel_tensor(
        tensors,
        _optional_uid(node, "inv_variance_tensor_uid"),
        inv_variance,
        x.ndim,
    )

    prev_mean_uid = _optional_uid(node, "prev_running_mean_tensor_uid")
    prev_var_uid = _optional_uid(node, "prev_running_variance_tensor_uid")
    next_mean_uid = _optional_uid(node, "next_running_mean_tensor_uid")
    next_var_uid = _optional_uid(node, "next_running_variance_tensor_uid")
    momentum_uid = _optional_uid(node, "momentum_tensor_uid")
    running_present = [
        prev_mean_uid,
        prev_var_uid,
        next_mean_uid,
        next_var_uid,
        momentum_uid,
    ]
    if any(uid is not None for uid in running_present):
        if not all(uid is not None for uid in running_present):
            raise ValueError(
                "Batchnorm running-stat update requires prev mean/var, next mean/var, and momentum"
            )
        momentum = _scalar_value(tensors, int(momentum_uid), node)
        prev_mean = _channel_values(_tensor(tensors, int(prev_mean_uid), node), x)
        prev_var = _channel_values(_tensor(tensors, int(prev_var_uid), node), x)
        elements_per_channel = x.numel() // x.shape[1]
        if elements_per_channel == 1:
            adjusted_variance = variance
        else:
            adjusted_variance = variance * (
                elements_per_channel / (elements_per_channel - 1)
            )
        next_mean = (1.0 - momentum) * prev_mean + momentum * mean
        next_var = (1.0 - momentum) * prev_var + momentum * adjusted_variance
        _store_channel_tensor(tensors, next_mean_uid, next_mean, x.ndim)
        _store_channel_tensor(tensors, next_var_uid, next_var, x.ndim)


@register_handler("BatchnormBackwardAttributes")
def handle_batchnorm_backward(
    node: Dict[str, Any],
    tensors: Dict[int, torch.Tensor],
    graph_json: Dict[str, Any],
) -> None:
    """Handle batchnorm backward."""
    _reject_peer_stats(node, "Batchnorm backward")
    dy_uid = _required_input_uid(node, "dy_tensor_uid")
    x_uid = _required_input_uid(node, "x_tensor_uid")
    scale_uid = _required_input_uid(node, "scale_tensor_uid")
    dx_uid = _required_output_uid(node, "dx_tensor_uid")
    dscale_uid = _required_output_uid(node, "dscale_tensor_uid")
    dbias_uid = _required_output_uid(node, "dbias_tensor_uid")

    dy = _tensor(tensors, dy_uid, node).to(dtype=torch.float32)
    x = _tensor(tensors, x_uid, node)
    x_float = x.to(dtype=torch.float32)
    scale = _channel_values(_tensor(tensors, scale_uid, node), x)
    mean_uid = _optional_uid(node, "mean_tensor_uid")
    inv_uid = _optional_uid(node, "inv_variance_tensor_uid")
    if (mean_uid is None) != (inv_uid is None):
        raise ValueError(
            "Batchnorm backward requires both mean and inv variance, or neither"
        )
    if mean_uid is None:
        mean, variance = _bn_mean_var(x)
        inv_variance = torch.rsqrt(variance + 1e-5)
    else:
        mean = _channel_values(_tensor(tensors, int(mean_uid), node), x)
        inv_variance = _channel_values(_tensor(tensors, int(inv_uid), node), x)

    x_hat = (x_float - _channel_broadcast(mean, x_float)) * _channel_broadcast(
        inv_variance, x_float
    )
    reduce_dims = _bn_reduce_dims(x)
    dscale = (x_hat * dy).sum(dim=reduce_dims)
    dbias = dy.sum(dim=reduce_dims)
    elements_per_channel = x.numel() // x.shape[1]
    mean_dy = dbias / elements_per_channel
    mean_dy_xhat = dscale / elements_per_channel
    dx = (
        (
            dy
            - _channel_broadcast(mean_dy, x_float)
            - x_hat * _channel_broadcast(mean_dy_xhat, x_float)
        )
        * _channel_broadcast(scale * inv_variance, x_float)
    ).to(dtype=x.dtype)

    _store_tensor(tensors, dx_uid, dx)
    _store_channel_tensor(tensors, dscale_uid, dscale.to(dtype=scale.dtype), x.ndim)
    _store_channel_tensor(tensors, dbias_uid, dbias.to(dtype=scale.dtype), x.ndim)
