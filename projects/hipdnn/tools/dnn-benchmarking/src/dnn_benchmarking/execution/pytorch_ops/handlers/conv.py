# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Convolution forward/backward/wgrad reference handlers."""

from typing import Any, Dict, Sequence, Tuple

import torch
import torch.nn.functional as F

from .._common import *  # noqa: F401,F403
from .._registry import register_handler


def _validate_cross_correlation(node: Dict[str, Any]) -> None:
    conv_mode = _node_param(node, "conv_mode", "CROSS_CORRELATION")
    if conv_mode != "CROSS_CORRELATION":
        raise ValueError(
            f"Unsupported convolution mode {conv_mode!r}; PyTorch reference only supports CROSS_CORRELATION"
        )


def _conv_padding(node: Dict[str, Any]) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    pre = _as_tuple(_node_param(node, "pre_padding", [0, 0]), [0, 0])
    post = _as_tuple(_node_param(node, "post_padding", pre), pre)
    if len(pre) != 2 or len(post) != 2:
        raise ValueError("Only 2D convolution padding is supported")
    return (pre[0], pre[1]), (post[0], post[1])


def _conv_stride_dilation(
    node: Dict[str, Any],
) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    stride = _as_tuple(_node_param(node, "stride", [1, 1]), [1, 1])
    dilation = _as_tuple(_node_param(node, "dilation", [1, 1]), [1, 1])
    if len(stride) != 2 or len(dilation) != 2:
        raise ValueError("Only 2D convolution stride/dilation is supported")
    return (stride[0], stride[1]), (dilation[0], dilation[1])


def _conv_group_count(input_shape: Sequence[int], weight_shape: Sequence[int]) -> int:
    """Infer grouped convolution count from hipDNN tensor shapes."""
    if len(input_shape) < 2:
        raise ValueError(
            "Convolution input tensor must have at least 2 dimensions, "
            f"got {len(input_shape)}"
        )
    if len(weight_shape) < 2:
        raise ValueError(
            "Convolution weight tensor must have at least 2 dimensions, "
            f"got {len(weight_shape)}"
        )

    input_channels = int(input_shape[1])
    weight_channels_per_group = int(weight_shape[1])
    output_channels = int(weight_shape[0])
    if input_channels <= 0:
        raise ValueError(
            f"Convolution input channels must be positive, got {input_channels}"
        )
    if weight_channels_per_group <= 0:
        raise ValueError(
            "Convolution weight channels per group must be positive, "
            f"got {weight_channels_per_group}"
        )
    if output_channels <= 0:
        raise ValueError(
            f"Convolution weight output channels must be positive, got {output_channels}"
        )
    if input_channels % weight_channels_per_group != 0:
        raise ValueError(
            f"Convolution input channels ({input_channels}) must be evenly divisible "
            f"by weight channels per group ({weight_channels_per_group})"
        )

    groups = input_channels // weight_channels_per_group
    if output_channels % groups != 0:
        raise ValueError(
            f"Convolution weight output channels ({output_channels}) must be evenly "
            f"divisible by inferred group count ({groups})"
        )
    return groups


def _pad_conv_input(
    x: torch.Tensor, pre: Tuple[int, int], post: Tuple[int, int]
) -> torch.Tensor:
    if pre == (0, 0) and post == (0, 0):
        return x
    return F.pad(x, (pre[1], post[1], pre[0], post[0]))


def _conv2d_forward(
    node: Dict[str, Any], x: torch.Tensor, w: torch.Tensor
) -> torch.Tensor:
    _validate_cross_correlation(node)
    pre, post = _conv_padding(node)
    stride, dilation = _conv_stride_dilation(node)
    padded_x = _pad_conv_input(x, pre, post)
    return F.conv2d(
        padded_x,
        w,
        stride=stride,
        dilation=dilation,
        groups=_conv_group_count(x.shape, w.shape),
    )


def _conv_padding_is_symmetric(node: Dict[str, Any]) -> bool:
    pre, post = _conv_padding(node)
    return pre == post


@register_handler("ConvolutionFwdAttributes")
def handle_conv_fwd(
    node: Dict[str, Any],
    tensors: Dict[int, torch.Tensor],
    graph_json: Dict[str, Any],
) -> None:
    """Handle ConvolutionFwdAttributes (2D convolution forward pass)."""
    x_uid = _required_input_uid(node, "x_tensor_uid")
    w_uid = _required_input_uid(node, "w_tensor_uid")
    y_uid = _required_output_uid(node, "y_tensor_uid")

    y = _conv2d_forward(
        node, _tensor(tensors, x_uid, node), _tensor(tensors, w_uid, node)
    )
    _store_tensor(tensors, y_uid, y)


@register_handler("ConvolutionBwdAttributes")
def handle_conv_bwd(
    node: Dict[str, Any],
    tensors: Dict[int, torch.Tensor],
    graph_json: Dict[str, Any],
) -> None:
    """Handle ConvolutionBwdAttributes (gradient with respect to input)."""
    _validate_cross_correlation(node)
    dy_uid = _required_input_uid(node, "dy_tensor_uid")
    w_uid = _required_input_uid(node, "w_tensor_uid")
    dx_uid = _required_output_uid(node, "dx_tensor_uid")

    dy = _tensor(tensors, dy_uid, node)
    w = _tensor(tensors, w_uid, node)
    input_size = _tensor_shape(graph_json, dx_uid)
    if input_size is None:
        raise ValueError(
            f"ConvolutionBwdAttributes missing dx tensor shape for UID {dx_uid}"
        )

    stride, dilation = _conv_stride_dilation(node)
    pre, post = _conv_padding(node)
    groups = _conv_group_count(input_size, w.shape)
    if _conv_padding_is_symmetric(node):
        dx = torch.nn.grad.conv2d_input(
            input_size,
            w,
            dy,
            stride=stride,
            padding=pre,
            dilation=dilation,
            groups=groups,
        )
    else:
        with torch.enable_grad():
            x = torch.zeros(
                input_size, dtype=dy.dtype, device=dy.device, requires_grad=True
            )
            y = _conv2d_forward(node, x, w.detach())
            y.backward(dy)
            dx = x.grad.detach()
    _store_tensor(tensors, dx_uid, dx)


@register_handler("ConvolutionWrwAttributes")
def handle_conv_wrw(
    node: Dict[str, Any],
    tensors: Dict[int, torch.Tensor],
    graph_json: Dict[str, Any],
) -> None:
    """Handle ConvolutionWrwAttributes (gradient with respect to weights)."""
    _validate_cross_correlation(node)
    x_uid = _required_input_uid(node, "x_tensor_uid")
    dy_uid = _required_input_uid(node, "dy_tensor_uid")
    dw_uid = _required_output_uid(node, "dw_tensor_uid")

    x = _tensor(tensors, x_uid, node)
    dy = _tensor(tensors, dy_uid, node)
    weight_size = _tensor_shape(graph_json, dw_uid)
    if weight_size is None:
        raise ValueError(
            f"ConvolutionWrwAttributes missing dw tensor shape for UID {dw_uid}"
        )

    stride, dilation = _conv_stride_dilation(node)
    pre, _post = _conv_padding(node)
    groups = _conv_group_count(x.shape, weight_size)
    if _conv_padding_is_symmetric(node):
        dw = torch.nn.grad.conv2d_weight(
            x,
            weight_size,
            dy,
            stride=stride,
            padding=pre,
            dilation=dilation,
            groups=groups,
        )
    else:
        with torch.enable_grad():
            w = torch.zeros(
                weight_size, dtype=x.dtype, device=x.device, requires_grad=True
            )
            y = _conv2d_forward(node, x.detach(), w)
            y.backward(dy)
            dw = w.grad.detach()
    _store_tensor(tensors, dw_uid, dw)
