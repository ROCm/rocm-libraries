# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Reduction reference handler."""

from typing import Any, Dict, List, Optional, Tuple

import torch

from .._common import *  # noqa: F401,F403
from .._registry import register_handler


_REDUCTION_MODE_BY_VALUE = {
    1: "ADD",
    2: "MUL",
    3: "MIN",
    4: "MAX",
    5: "AMAX",
    6: "AVG",
    7: "NORM1",
    8: "NORM2",
    9: "MUL_NO_ZEROS",
}


def _reduction_mode_name(value: Any) -> str:
    if isinstance(value, str):
        mode = value.upper()
        return {"MIN_OP": "MIN", "MAX_OP": "MAX"}.get(mode, mode)
    return _REDUCTION_MODE_BY_VALUE.get(int(value), "NOT_SET")


def _reduce_prod(
    value: torch.Tensor, dims: Tuple[int, ...], keepdim: bool
) -> torch.Tensor:
    if not dims:
        return value
    result = value
    for dim in sorted(dims, reverse=True):
        result = result.prod(dim=dim, keepdim=keepdim)
    return result


def _reduction_dims_for_output(
    x: torch.Tensor,
    out_shape: Optional[Tuple[int, ...]],
) -> Tuple[Tuple[int, ...], bool]:
    if out_shape is None or _numel(out_shape) == 1:
        return tuple(range(x.ndim)), False

    if len(out_shape) == x.ndim:
        dims = tuple(
            dim
            for dim, (input_extent, output_extent) in enumerate(zip(x.shape, out_shape))
            if int(output_extent) == 1 and int(input_extent) != 1
        )
        return dims, True

    matched = 0
    dims_list: List[int] = []
    for dim, input_extent in enumerate(x.shape):
        if matched < len(out_shape) and int(out_shape[matched]) == int(input_extent):
            matched += 1
        else:
            dims_list.append(dim)

    if matched == len(out_shape):
        return tuple(dims_list), False

    raise ValueError(
        f"Reduction output shape {out_shape} is not compatible with input shape "
        f"{tuple(x.shape)}"
    )


@register_handler("ReductionAttributes")
def handle_reduction(
    node: Dict[str, Any],
    tensors: Dict[int, torch.Tensor],
    graph_json: Dict[str, Any],
) -> None:
    """Handle hipDNN reduction attributes with PyTorch reductions."""
    in_uid = _node_uid(node, "in_tensor_uid", ("inputs",), required=False)
    if in_uid is None:
        in_uid = _node_uid(node, "x_tensor_uid", ("inputs",), required=True)
    out_uid = _node_uid(node, "out_tensor_uid", ("outputs",), required=False)
    if out_uid is None:
        out_uid = _node_uid(node, "y_tensor_uid", ("outputs",), required=True)

    x = _tensor(tensors, int(in_uid), node)
    out_shape = _stored_tensor_shape(tensors, graph_json, int(out_uid))
    dims, keepdim = _reduction_dims_for_output(x, out_shape)
    mode = _reduction_mode_name(_node_param(node, "mode", "NOT_SET"))

    if mode == "ADD":
        result = x.sum(dim=dims, keepdim=keepdim) if dims else x
    elif mode == "MUL":
        result = _reduce_prod(x, dims, keepdim)
    elif mode == "MIN":
        result = torch.amin(x, dim=dims, keepdim=keepdim) if dims else x
    elif mode == "MAX":
        result = torch.amax(x, dim=dims, keepdim=keepdim) if dims else x
    elif mode == "AMAX":
        result = (
            torch.amax(torch.abs(x), dim=dims, keepdim=keepdim)
            if dims
            else torch.abs(x)
        )
    elif mode == "AVG":
        result = x.mean(dim=dims, keepdim=keepdim) if dims else x
    elif mode == "NORM1":
        result = torch.abs(x).sum(dim=dims, keepdim=keepdim) if dims else torch.abs(x)
    elif mode == "NORM2":
        result = (
            torch.linalg.vector_norm(x, ord=2, dim=dims, keepdim=keepdim)
            if dims
            else torch.abs(x)
        )
    elif mode == "MUL_NO_ZEROS":
        nonzero = torch.where(x == 0, torch.ones((), dtype=x.dtype, device=x.device), x)
        result = _reduce_prod(nonzero, dims, keepdim)
    else:
        raise ValueError(f"Unsupported reduction mode: {mode}")

    _store_tensor_for_uid(tensors, graph_json, int(out_uid), result)
