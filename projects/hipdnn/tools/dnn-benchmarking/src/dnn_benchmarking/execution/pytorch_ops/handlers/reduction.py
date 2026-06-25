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

    # Assume float32 accumulation: promote low-precision floating inputs so the
    # reduction accumulates in fp32, then cast the result back to the graph dtype
    # on store. Integer inputs are reduced in their own dtype.
    promote = x.is_floating_point() and x.dtype not in (torch.float32, torch.float64)
    xc = x.to(dtype=torch.float32) if promote else x

    if mode == "ADD":
        result = xc.sum(dim=dims, keepdim=keepdim) if dims else xc
    elif mode == "MUL":
        result = _reduce_prod(xc, dims, keepdim)
    elif mode == "MIN":
        result = torch.amin(xc, dim=dims, keepdim=keepdim) if dims else xc
    elif mode == "MAX":
        result = torch.amax(xc, dim=dims, keepdim=keepdim) if dims else xc
    elif mode == "AMAX":
        result = (
            torch.amax(torch.abs(xc), dim=dims, keepdim=keepdim)
            if dims
            else torch.abs(xc)
        )
    elif mode == "AVG":
        result = xc.mean(dim=dims, keepdim=keepdim) if dims else xc
    elif mode == "NORM1":
        result = torch.abs(xc).sum(dim=dims, keepdim=keepdim) if dims else torch.abs(xc)
    elif mode == "NORM2":
        result = (
            torch.linalg.vector_norm(xc, ord=2, dim=dims, keepdim=keepdim)
            if dims
            else torch.abs(xc)
        )
    elif mode == "MUL_NO_ZEROS":
        nonzero = torch.where(
            xc == 0, torch.ones((), dtype=xc.dtype, device=xc.device), xc
        )
        result = _reduce_prod(nonzero, dims, keepdim)
    else:
        raise ValueError(f"Unsupported reduction mode: {mode}")

    if promote:
        result = result.to(dtype=x.dtype)
    _store_tensor_for_uid(tensors, graph_json, int(out_uid), result)
