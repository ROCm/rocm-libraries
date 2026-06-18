# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Resample (pooling) reference handler."""

from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from .._common import *  # noqa: F401,F403
from .._registry import register_handler


_RESAMPLE_MODE_BY_VALUE = {
    1: "MAXPOOL",
    2: "AVGPOOL_EXCLUDE_PADDING",
    3: "AVGPOOL_INCLUDE_PADDING",
}


def _resample_mode_name(value: Any) -> str:
    if isinstance(value, str):
        return value.upper()
    return _RESAMPLE_MODE_BY_VALUE.get(int(value), "NOT_SET")


_PADDING_MODE_BY_VALUE = {
    1: "NEG_INF_PAD",
    2: "ZERO_PAD",
}


def _padding_mode_name(value: Any) -> str:
    if isinstance(value, str):
        mode = value.upper()
        return "PADDING_NOT_SET" if mode == "NOT_SET" else mode
    return _PADDING_MODE_BY_VALUE.get(int(value), "PADDING_NOT_SET")


def _spatial_tuple(
    node: Dict[str, Any],
    graph_json: Dict[str, Any],
    x_uid: int,
    key: str,
    default_value: int,
    x_shape_override: Optional[Sequence[int]] = None,
) -> Tuple[int, ...]:
    x_shape = (
        tuple(int(dim) for dim in x_shape_override)
        if x_shape_override is not None
        else _tensor_shape(graph_json, x_uid)
    )
    if x_shape is None:
        raise ValueError(f"ResampleFwdAttributes missing input shape for UID {x_uid}")
    spatial_rank = len(x_shape) - 2
    if spatial_rank < 1 or spatial_rank > 3:
        raise ValueError(
            f"ResampleFwdAttributes supports rank 3/4/5 tensors, got rank {len(x_shape)}"
        )
    values = _node_param(node, key, None)
    if values is None:
        return (default_value,) * spatial_rank
    result = tuple(int(v) for v in values)
    if len(result) != spatial_rank:
        raise ValueError(
            f"ResampleFwdAttributes {key} length {len(result)} does not match "
            f"spatial rank {spatial_rank}"
        )
    return result


def _resample_has_asymmetric_padding(
    node: Dict[str, Any], graph_json: Dict[str, Any]
) -> bool:
    x_uid = _node_uid(node, "x_tensor_uid", ("inputs",), required=False)
    if x_uid is None:
        return False
    try:
        pre = _spatial_tuple(node, graph_json, int(x_uid), "pre_padding", 0)
        post = _spatial_tuple(node, graph_json, int(x_uid), "post_padding", 0)
    except ValueError:
        return False
    return pre != post


def _pad_spatial(
    x: torch.Tensor,
    pre: Sequence[int],
    post: Sequence[int],
    value: float,
) -> torch.Tensor:
    if all(v == 0 for v in pre) and all(v == 0 for v in post):
        return x
    pad = []
    for before, after in reversed(tuple(zip(pre, post))):
        pad.extend([int(before), int(after)])
    return F.pad(x, tuple(pad), value=value)


def _pool_function(mode: str, spatial_rank: int) -> Callable[..., Any]:
    if mode == "MAXPOOL":
        return (F.max_pool1d, F.max_pool2d, F.max_pool3d)[spatial_rank - 1]
    return (F.avg_pool1d, F.avg_pool2d, F.avg_pool3d)[spatial_rank - 1]


@register_handler("ResampleFwdAttributes")
def handle_resample_fwd(
    node: Dict[str, Any],
    tensors: Dict[int, torch.Tensor],
    graph_json: Dict[str, Any],
) -> None:
    """Handle resample forward as max/average pooling."""
    x_uid = _required_input_uid(node, "x_tensor_uid")
    y_uid = _required_output_uid(node, "y_tensor_uid")
    index_uid = _node_uid(node, "index_tensor_uid", ("outputs",), required=False)

    x = _tensor(tensors, x_uid, node)
    spatial_rank = x.ndim - 2
    if spatial_rank < 1 or spatial_rank > 3:
        raise ValueError(
            f"ResampleFwdAttributes supports rank 3/4/5 tensors, got rank {x.ndim}"
        )

    pre = _spatial_tuple(node, graph_json, x_uid, "pre_padding", 0, x.shape)
    post = _spatial_tuple(node, graph_json, x_uid, "post_padding", 0, x.shape)
    stride = _spatial_tuple(node, graph_json, x_uid, "stride", 1, x.shape)
    window = _spatial_tuple(node, graph_json, x_uid, "window", 1, x.shape)
    if any(v <= 0 for v in (*stride, *window)):
        raise ValueError("ResampleFwdAttributes stride/window values must be positive")

    mode = _resample_mode_name(_node_param(node, "resample_mode", "NOT_SET"))
    padding_mode = _padding_mode_name(
        _node_param(node, "padding_mode", "PADDING_NOT_SET")
    )
    pool = _pool_function(mode, spatial_rank)

    if mode == "MAXPOOL":
        return_indices = index_uid is not None
        use_builtin_padding = pre == post and padding_mode != "ZERO_PAD"
        if use_builtin_padding:
            pooled = pool(
                x,
                kernel_size=window,
                stride=stride,
                padding=pre,
                return_indices=return_indices,
            )
        else:
            pad_value = 0.0 if padding_mode == "ZERO_PAD" else float("-inf")
            padded = _pad_spatial(x, pre, post, pad_value)
            pooled = pool(
                padded,
                kernel_size=window,
                stride=stride,
                padding=0,
                return_indices=return_indices,
            )
        if return_indices:
            y, indices = pooled
            _store_tensor_for_uid(tensors, graph_json, y_uid, y)
            _store_tensor_for_uid(tensors, graph_json, int(index_uid), indices)
        else:
            _store_tensor_for_uid(tensors, graph_json, y_uid, pooled)
        return

    if mode not in ("AVGPOOL_EXCLUDE_PADDING", "AVGPOOL_INCLUDE_PADDING"):
        raise ValueError(f"Unsupported resample mode: {mode}")
    if index_uid is not None:
        raise ValueError("Average pooling resample does not produce indices")
    if padding_mode not in ("PADDING_NOT_SET", "ZERO_PAD"):
        raise ValueError(f"{mode} requires ZERO_PAD padding, got {padding_mode}")

    count_include_pad = mode == "AVGPOOL_INCLUDE_PADDING"
    if pre == post:
        y = pool(
            x,
            kernel_size=window,
            stride=stride,
            padding=pre,
            count_include_pad=count_include_pad,
        )
    else:
        padded = _pad_spatial(x, pre, post, 0.0)
        if count_include_pad:
            y = pool(
                padded,
                kernel_size=window,
                stride=stride,
                padding=0,
                count_include_pad=True,
            )
        else:
            window_elements = float(_numel(window))
            sums = (
                pool(
                    padded,
                    kernel_size=window,
                    stride=stride,
                    padding=0,
                    count_include_pad=True,
                )
                * window_elements
            )
            mask = torch.ones_like(x, dtype=torch.float32)
            counts = (
                pool(
                    _pad_spatial(mask, pre, post, 0.0),
                    kernel_size=window,
                    stride=stride,
                    padding=0,
                    count_include_pad=True,
                )
                * window_elements
            )
            y = sums / counts.clamp_min(1.0).to(dtype=sums.dtype)

    _store_tensor_for_uid(tensors, graph_json, y_uid, y)
