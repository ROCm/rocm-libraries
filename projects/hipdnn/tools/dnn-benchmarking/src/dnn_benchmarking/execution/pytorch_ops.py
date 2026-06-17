# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""PyTorch operation implementations for graph execution.

These handlers execute on the device of the input tensors (CPU or CUDA).
Used by both PyTorchReferenceProvider (CPU) and PyTorchCudaExecutor (CUDA).
"""

from math import prod, sqrt
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import torch
import torch.nn.functional as F

# Type alias for operation handlers
OpHandler = Callable[[Dict[str, Any], Dict[int, torch.Tensor], Dict[str, Any]], None]

# Registry of operation handlers
_OP_HANDLERS: Dict[str, OpHandler] = {}


def register_handler(op_type: str) -> Callable[[OpHandler], OpHandler]:
    """Decorator to register an operation handler.

    Args:
        op_type: The node type string to handle (e.g., "ConvolutionFwdAttributes").

    Returns:
        Decorator function.
    """

    def decorator(func: OpHandler) -> OpHandler:
        _OP_HANDLERS[op_type] = func
        return func

    return decorator


def get_handler(op_type: str) -> Optional[OpHandler]:
    """Get handler for operation type.

    Args:
        op_type: The node type string.

    Returns:
        Handler function or None if not found.
    """
    return _OP_HANDLERS.get(op_type)


def get_supported_operations() -> Set[str]:
    """Get set of supported operation types.

    Returns:
        Set of operation type strings that have handlers.
    """
    return set(_OP_HANDLERS.keys())


def supports_graph(graph_json: Dict[str, Any]) -> bool:
    """Check if all graph operations are supported.

    Args:
        graph_json: The graph as a parsed JSON dictionary.

    Returns:
        True if all node types have handlers.
    """
    for node in graph_json.get("nodes", []):
        if node.get("type") not in _OP_HANDLERS:
            return False
    return True


def get_unsupported_operations(graph_json: Dict[str, Any]) -> List[str]:
    """Get list of unsupported operation types in graph.

    Args:
        graph_json: The graph as a parsed JSON dictionary.

    Returns:
        List of unsupported operation type strings.
    """
    unsupported = []
    for node in graph_json.get("nodes", []):
        op_type = node.get("type")
        if op_type not in _OP_HANDLERS:
            unsupported.append(op_type)
    return unsupported


def get_reference_warnings(graph_json: Dict[str, Any]) -> List[str]:
    """Describe manual/non-built-in portions of the PyTorch reference graph.

    The timed PyTorch reference row is useful as a baseline only when users can
    tell whether it is measuring a public PyTorch primitive or local reference
    glue.  This helper is intentionally static: it only inspects graph metadata
    and reports paths whose handler is not solely a built-in PyTorch operator.
    """

    warnings: List[str] = []
    for node in graph_json.get("nodes", []):
        op_type = str(node.get("type", ""))
        name = str(node.get("name") or op_type)

        if op_type == "LayernormAttributes":
            if (
                _node_uid(node, "mean_tensor_uid", ("outputs",), required=False)
                is not None
                or _node_uid(
                    node, "inv_variance_tensor_uid", ("outputs",), required=False
                )
                is not None
            ):
                warnings.append(
                    f"{name}: LayernormAttributes uses torch.nn.functional.layer_norm "
                    "for y but computes requested mean/inv-variance outputs manually; "
                    "PyTorch reference timing is not solely built-in PyTorch operator time."
                )

        elif op_type == "RMSNormAttributes":
            reasons: List[str] = []
            if not hasattr(F, "rms_norm"):
                reasons.append("torch.nn.functional.rms_norm is unavailable")
            if (
                _node_uid(node, "bias_tensor_uid", ("inputs",), required=False)
                is not None
            ):
                reasons.append("optional bias is applied manually")
            if (
                _node_uid(node, "inv_rms_tensor_uid", ("outputs",), required=False)
                is not None
            ):
                reasons.append("requested inv_rms output is computed manually")
            if not _rmsnorm_graph_can_use_builtin(node, graph_json):
                reasons.append("per-channel layout uses a manual RMSNorm formula")
            if reasons:
                warnings.append(
                    f"{name}: RMSNormAttributes includes manual reference work "
                    f"({'; '.join(dict.fromkeys(reasons))}); PyTorch reference timing "
                    "is not solely built-in PyTorch operator time."
                )

        elif op_type == "RMSNormBackwardAttributes":
            warnings.append(
                f"{name}: RMSNormBackwardAttributes uses a manual RMSNorm backward "
                "formula because PyTorch has no public operator matching hipDNN's "
                "saved-inv_rms backward node; PyTorch reference timing is not solely "
                "built-in PyTorch operator time."
            )

        elif op_type == "SdpaBackwardAttributes":
            warnings.append(
                f"{name}: SdpaBackwardAttributes uses a manual flash-attention "
                "backward formula that consumes hipDNN's saved stats (log-sum-exp); "
                "torch.nn.functional.scaled_dot_product_attention autograd cannot "
                "consume external stats, so PyTorch reference timing is not solely "
                "built-in PyTorch operator time."
            )

        elif op_type == "ReductionAttributes":
            if (
                _reduction_mode_name(_node_param(node, "mode", "NOT_SET"))
                == "MUL_NO_ZEROS"
            ):
                warnings.append(
                    f"{name}: ReductionAttributes mode MUL_NO_ZEROS uses a manual "
                    "masked product; PyTorch reference timing is not solely built-in "
                    "PyTorch operator time."
                )

        elif op_type == "ResampleFwdAttributes":
            mode = _resample_mode_name(_node_param(node, "resample_mode", "NOT_SET"))
            if mode == "AVGPOOL_EXCLUDE_PADDING" and _resample_has_asymmetric_padding(
                node, graph_json
            ):
                warnings.append(
                    f"{name}: ResampleFwdAttributes AVGPOOL_EXCLUDE_PADDING with "
                    "asymmetric padding uses manual valid-count correction around "
                    "torch.nn.functional.avg_pool; PyTorch reference timing is not "
                    "solely built-in PyTorch operator time."
                )

    return warnings


def execute_graph(
    graph_json: Dict[str, Any],
    tensors: Dict[int, torch.Tensor],
) -> None:
    """Execute all graph operations in order.

    Args:
        graph_json: The graph as a parsed JSON dictionary.
        tensors: Mapping of tensor UID to torch.Tensor.

    Raises:
        ValueError: If graph contains unsupported operations.
    """
    for node in graph_json.get("nodes", []):
        op_type = node.get("type")
        handler = _OP_HANDLERS.get(op_type)
        if handler:
            handler(node, tensors, graph_json)
        else:
            raise ValueError(f"Unsupported operation type: {op_type}")


# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------


def _as_tuple(
    values: Optional[Sequence[Any]], default: Sequence[int]
) -> Tuple[int, ...]:
    if values is None:
        return tuple(int(v) for v in default)
    return tuple(int(v) for v in values)


def _node_section(node: Dict[str, Any], section: str) -> Dict[str, Any]:
    value = node.get(section, {})
    return value if isinstance(value, dict) else {}


def _node_param(node: Dict[str, Any], key: str, default: Any = None) -> Any:
    for section_name in ("parameters", "attributes", "inputs", "outputs"):
        section = _node_section(node, section_name)
        if key in section:
            return section[key]
    return node.get(key, default)


def _node_uid(
    node: Dict[str, Any],
    key: str,
    sections: Iterable[str],
    required: bool = True,
) -> Optional[int]:
    for section_name in sections:
        section = _node_section(node, section_name)
        if key in section and section[key] is not None:
            return int(section[key])
    attrs = _node_section(node, "attributes")
    if key in attrs and attrs[key] is not None:
        return int(attrs[key])
    if key in node and node[key] is not None:
        return int(node[key])
    if required:
        raise ValueError(
            f"{node.get('type', 'Node')} missing required tensor UIDs ({key}): {node}"
        )
    return None


def _required_input_uid(node: Dict[str, Any], key: str) -> int:
    return int(_node_uid(node, key, ("inputs",), required=True))


def _required_output_uid(node: Dict[str, Any], key: str) -> int:
    return int(_node_uid(node, key, ("outputs",), required=True))


def _optional_uid(node: Dict[str, Any], key: str) -> Optional[int]:
    return _node_uid(node, key, ("inputs", "outputs"), required=False)


def _tensor(
    tensors: Dict[int, torch.Tensor], uid: int, node: Dict[str, Any]
) -> torch.Tensor:
    try:
        return tensors[uid]
    except KeyError as e:
        raise ValueError(
            f"{node.get('type', 'Node')} references missing tensor UID {uid}"
        ) from e


def _tensor_shape(graph_json: Dict[str, Any], uid: int) -> Optional[Tuple[int, ...]]:
    for tensor_json in graph_json.get("tensors", []):
        if tensor_json.get("uid") == uid:
            return tuple(int(dim) for dim in tensor_json.get("dims", []))
    return None


def _store_tensor(
    tensors: Dict[int, torch.Tensor], uid: int, value: torch.Tensor
) -> None:
    existing = tensors.get(uid)
    if existing is not None and tuple(existing.shape) == tuple(value.shape):
        existing.copy_(value.to(dtype=existing.dtype, device=existing.device))
        tensors[uid] = existing
    else:
        tensors[uid] = value


def _store_channel_tensor(
    tensors: Dict[int, torch.Tensor],
    uid: Optional[int],
    values: torch.Tensor,
    fallback_ndim: int,
) -> None:
    if uid is None:
        return
    existing = tensors.get(uid)
    if existing is not None:
        shaped = values.reshape(existing.shape)
    else:
        shaped = values.reshape([1, values.numel()] + [1] * max(fallback_ndim - 2, 0))
    _store_tensor(tensors, uid, shaped)


def _channel_values(tensor: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    values = tensor.reshape(-1).to(dtype=torch.float32)
    if x.ndim < 2:
        raise ValueError("Batchnorm tensors require at least 2 dimensions")
    if values.numel() != x.shape[1]:
        raise ValueError(
            f"Batchnorm channel tensor has {values.numel()} elements, expected {x.shape[1]}"
        )
    return values


def _reject_peer_stats(node: Dict[str, Any], operation: str) -> None:
    peer_stats = _node_param(node, "peer_stats_tensor_uid", None)
    if peer_stats is None:
        return
    if isinstance(peer_stats, (list, tuple)) and len(peer_stats) == 0:
        return
    raise ValueError(f"{operation} does not support peer statistics")


def _channel_broadcast(values: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return values.reshape([1, values.numel()] + [1] * (x.ndim - 2)).to(device=x.device)


def _scalar_value(
    tensors: Dict[int, torch.Tensor], uid: int, node: Dict[str, Any]
) -> float:
    tensor = _tensor(tensors, uid, node)
    if tensor.numel() < 1:
        raise ValueError(f"Scalar tensor UID {uid} is empty")
    return float(tensor.detach().reshape(-1)[0].item())


def _numel(shape: Sequence[int]) -> int:
    return prod(int(dim) for dim in shape)


def _stored_tensor_shape(
    tensors: Dict[int, torch.Tensor], graph_json: Dict[str, Any], uid: int
) -> Optional[Tuple[int, ...]]:
    existing = tensors.get(uid)
    if existing is not None:
        return tuple(int(dim) for dim in existing.shape)
    return _tensor_shape(graph_json, uid)


def _store_tensor_for_uid(
    tensors: Dict[int, torch.Tensor],
    graph_json: Dict[str, Any],
    uid: int,
    value: torch.Tensor,
) -> None:
    shape = _stored_tensor_shape(tensors, graph_json, uid)
    if shape is not None and tuple(value.shape) != shape:
        if _numel(shape) != value.numel():
            raise ValueError(
                f"Cannot store tensor UID {uid} with shape {tuple(value.shape)} "
                f"as graph shape {shape}"
            )
        value = value.reshape(shape)
    _store_tensor(tensors, uid, value)


def _strip_leading_singletons(shape: Sequence[int]) -> Tuple[int, ...]:
    values = tuple(int(dim) for dim in shape)
    index = 0
    while index < len(values) and values[index] == 1:
        index += 1
    return values[index:]


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


def _sum_to_shape(value: torch.Tensor, target_shape: Sequence[int]) -> torch.Tensor:
    shape = tuple(int(dim) for dim in target_shape)
    result = value
    while result.ndim > len(shape):
        result = result.sum(dim=0)

    if result.ndim != len(shape):
        raise ValueError(
            f"Cannot reduce tensor with shape {tuple(value.shape)} to shape {shape}"
        )

    for dim, target in enumerate(shape):
        current = int(result.shape[dim])
        if current == target:
            continue
        if target != 1:
            raise ValueError(
                f"Cannot reduce tensor with shape {tuple(value.shape)} to shape {shape}"
            )
        result = result.sum(dim=dim, keepdim=True)

    return result.reshape(shape)


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


def _sdpa_bool(node: Dict[str, Any], key: str, default: bool = False) -> bool:
    return bool(_node_param(node, key, default))


def _sdpa_unsupported_if_present(node: Dict[str, Any], keys: Sequence[str]) -> None:
    for key in keys:
        if _optional_uid(node, key) is not None:
            raise ValueError(
                f"Unsupported SDPA optional tensor '{key}' in PyTorch reference"
            )


def _sdpa_scale(
    node: Dict[str, Any], tensors: Dict[int, torch.Tensor]
) -> Optional[float]:
    scale_uid = _optional_uid(node, "scale_tensor_uid")
    if scale_uid is not None:
        return _scalar_value(tensors, scale_uid, node)
    value = _node_param(node, "attn_scale_value", None)
    return None if value is None else float(value)


def _sdpa_head_repeat(q_heads: int, kv_heads: int, label: str) -> int:
    """Validate and return the per-query-head repeat factor for K or V.

    hipDNN allows independent K and V head counts; each must divide the query
    head count (frontend SdpaBwdNode validation), and the CPU reference maps K
    and V with separate ratios.
    """
    if kv_heads <= 0 or q_heads % kv_heads != 0:
        raise ValueError(
            f"Unsupported SDPA {label} head count: q_heads={q_heads}, "
            f"{label.lower()}_heads={kv_heads}"
        )
    return q_heads // kv_heads


def _sdpa_common(
    node: Dict[str, Any],
    tensors: Dict[int, torch.Tensor],
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> Tuple[Optional[torch.Tensor], float, bool, Optional[float], int, int]:
    unsupported = [
        "seq_len_q_tensor_uid",
        "seq_len_kv_tensor_uid",
        "seed_tensor_uid",
        "offset_tensor_uid",
        "dropout_mask_tensor_uid",
        "dropout_scale_tensor_uid",
        "page_table_k_tensor_uid",
        "page_table_v_tensor_uid",
        "block_mask_tensor_uid",
        "sink_token_tensor_uid",
        "descale_q_tensor_uid",
        "descale_k_tensor_uid",
        "descale_v_tensor_uid",
        "descale_s_tensor_uid",
        "scale_s_tensor_uid",
        "scale_o_tensor_uid",
    ]
    _sdpa_unsupported_if_present(node, unsupported)

    if _sdpa_bool(node, "alibi_mask") or _sdpa_bool(node, "padding_mask"):
        raise ValueError(
            "SDPA alibi/padding masks are not supported by the PyTorch reference"
        )
    if _sdpa_bool(node, "causal_mask_bottom_right"):
        raise ValueError(
            "SDPA bottom-right causal mask is not supported by the PyTorch reference"
        )
    diagonal_alignment = _node_param(node, "diagonal_alignment", "TOP_LEFT")
    if diagonal_alignment not in ("TOP_LEFT", 0, None):
        raise ValueError("Only TOP_LEFT SDPA diagonal alignment is supported")
    if (
        _node_param(node, "left_bound", None) is not None
        or _node_param(node, "right_bound", None) is not None
    ):
        raise ValueError(
            "SDPA sliding-window bounds are not supported by the PyTorch reference"
        )

    dropout_probability = _node_param(node, "dropout_probability", 0.0)
    dropout_p = 0.0 if dropout_probability is None else float(dropout_probability)
    if dropout_p != 0.0:
        raise ValueError(
            "Nonzero SDPA dropout cannot be exactly validated against PyTorch"
        )

    mask_uid = _optional_uid(node, "attn_mask_tensor_uid")
    attn_mask = _tensor(tensors, mask_uid, node) if mask_uid is not None else None
    is_causal = _sdpa_bool(node, "causal_mask")
    if attn_mask is not None and is_causal:
        raise ValueError(
            "PyTorch SDPA reference does not support both attn_mask and causal_mask"
        )

    scale = _sdpa_scale(node, tensors)
    if q.ndim < 3 or k.ndim < 3 or v.ndim < 3:
        raise ValueError("SDPA expects q/k/v tensors with head and matrix dimensions")
    q_heads = int(q.shape[-3])
    rep_k = _sdpa_head_repeat(q_heads, int(k.shape[-3]), "K")
    rep_v = _sdpa_head_repeat(q_heads, int(v.shape[-3]), "V")
    return attn_mask, dropout_p, is_causal, scale, rep_k, rep_v


def _call_sdpa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask: Optional[torch.Tensor],
    dropout_p: float,
    is_causal: bool,
    scale: Optional[float],
    rep_k: int,
    rep_v: int,
) -> torch.Tensor:
    kwargs: Dict[str, Any] = {}
    if scale is not None:
        kwargs["scale"] = scale
    # Expand K and V independently to the query head count. PyTorch's
    # enable_gqa only models equal K/V head counts, so explicit repeat is the
    # only correct path when Hk != Hv.
    if rep_k > 1:
        k = k.repeat_interleave(rep_k, dim=-3)
    if rep_v > 1:
        v = v.repeat_interleave(rep_v, dim=-3)
    return F.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=is_causal,
        **kwargs,
    )


def _sdpa_stats(
    q: torch.Tensor,
    k: torch.Tensor,
    attn_mask: Optional[torch.Tensor],
    is_causal: bool,
    scale: Optional[float],
    rep_k: int,
) -> torch.Tensor:
    q_float = q.to(dtype=torch.float32)
    k_float = k.to(dtype=torch.float32)
    if rep_k > 1:
        k_float = k_float.repeat_interleave(rep_k, dim=-3)
    scale_value = (1.0 / sqrt(float(q.shape[-1]))) if scale is None else scale
    scores = torch.matmul(q_float, k_float.transpose(-2, -1)) * scale_value
    if attn_mask is not None:
        scores = scores + attn_mask.to(dtype=torch.float32)
    if is_causal:
        length_q = scores.shape[-2]
        length_k = scores.shape[-1]
        causal = torch.ones(
            length_q,
            length_k,
            dtype=torch.bool,
            device=scores.device,
        ).tril()
        scores = scores.masked_fill(~causal, float("-inf"))
    return torch.logsumexp(scores, dim=-1, keepdim=True)


# -----------------------------------------------------------------------------
# Operation Handlers
# -----------------------------------------------------------------------------


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


@register_handler("MatmulAttributes")
def handle_matmul(
    node: Dict[str, Any],
    tensors: Dict[int, torch.Tensor],
    graph_json: Dict[str, Any],
) -> None:
    """Handle MatmulAttributes (matrix multiplication)."""
    a_uid = _required_input_uid(node, "a_tensor_uid")
    b_uid = _required_input_uid(node, "b_tensor_uid")
    c_uid = _required_output_uid(node, "c_tensor_uid")

    c = torch.matmul(_tensor(tensors, a_uid, node), _tensor(tensors, b_uid, node))
    _store_tensor(tensors, c_uid, c)


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


@register_handler("SdpaAttributes")
def handle_sdpa(
    node: Dict[str, Any],
    tensors: Dict[int, torch.Tensor],
    graph_json: Dict[str, Any],
) -> None:
    """Handle scaled dot-product attention forward."""
    _sdpa_unsupported_if_present(
        node,
        [
            "max_tensor_uid",
            "sum_exp_tensor_uid",
            "rng_dump_tensor_uid",
            "amax_s_tensor_uid",
            "amax_o_tensor_uid",
        ],
    )
    q_uid = _required_input_uid(node, "q_tensor_uid")
    k_uid = _required_input_uid(node, "k_tensor_uid")
    v_uid = _required_input_uid(node, "v_tensor_uid")
    o_uid = _required_output_uid(node, "o_tensor_uid")

    q = _tensor(tensors, q_uid, node)
    k = _tensor(tensors, k_uid, node)
    v = _tensor(tensors, v_uid, node)
    attn_mask, dropout_p, is_causal, scale, rep_k, rep_v = _sdpa_common(
        node, tensors, q, k, v
    )
    o = _call_sdpa(q, k, v, attn_mask, dropout_p, is_causal, scale, rep_k, rep_v)
    _store_tensor(tensors, o_uid, o)

    stats_uid = _optional_uid(node, "stats_tensor_uid")
    if stats_uid is not None:
        _store_tensor(
            tensors,
            stats_uid,
            _sdpa_stats(q, k, attn_mask, is_causal, scale, rep_k),
        )


@register_handler("SdpaBackwardAttributes")
def handle_sdpa_backward(
    node: Dict[str, Any],
    tensors: Dict[int, torch.Tensor],
    graph_json: Dict[str, Any],
) -> None:
    """Handle scaled dot-product attention backward.

    Mirrors hipDNN's CPU reference (CpuFpReferenceSdpa::backward): the saved
    softmax statistics ``stats`` (forward log-sum-exp) are consumed directly to
    recompute probabilities as ``P = exp(scores - stats)`` without
    renormalization.  PyTorch's built-in SDPA autograd cannot consume an
    external ``stats`` tensor and always renormalizes its own softmax, so it
    would diverge from hipDNN whenever ``stats`` is not the exact, consistent
    forward LSE.  This handler therefore implements the gradient manually.
    """
    _sdpa_unsupported_if_present(node, ["dropout_scale_inv_tensor_uid"])
    if _optional_uid(node, "dbias_tensor_uid") is not None:
        raise ValueError(
            "SDPA backward dBias gradient is not supported by the PyTorch reference"
        )

    q_uid = _required_input_uid(node, "q_tensor_uid")
    k_uid = _required_input_uid(node, "k_tensor_uid")
    v_uid = _required_input_uid(node, "v_tensor_uid")
    o_uid = _required_input_uid(node, "o_tensor_uid")
    do_uid = _required_input_uid(node, "do_tensor_uid")
    stats_uid = _required_input_uid(node, "stats_tensor_uid")
    dq_uid = _required_output_uid(node, "dq_tensor_uid")
    dk_uid = _required_output_uid(node, "dk_tensor_uid")
    dv_uid = _required_output_uid(node, "dv_tensor_uid")

    q = _tensor(tensors, q_uid, node)
    k = _tensor(tensors, k_uid, node)
    v = _tensor(tensors, v_uid, node)
    o = _tensor(tensors, o_uid, node)
    do = _tensor(tensors, do_uid, node)
    stats = _tensor(tensors, stats_uid, node)
    attn_mask, _dropout_p, is_causal, scale, rep_k, rep_v = _sdpa_common(
        node, tensors, q, k, v
    )

    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("SDPA backward expects rank-4 q/k/v tensors [B, H, S, D]")

    q_f = q.to(dtype=torch.float32)
    k_f = k.to(dtype=torch.float32)
    v_f = v.to(dtype=torch.float32)
    o_f = o.to(dtype=torch.float32)
    do_f = do.to(dtype=torch.float32)
    stats_f = stats.to(dtype=torch.float32)

    head_dim = int(q.shape[-1])
    scale_value = (1.0 / sqrt(float(head_dim))) if scale is None else float(scale)
    k_heads = int(k.shape[1])
    v_heads = int(v.shape[1])
    if rep_k > 1:
        k_f = k_f.repeat_interleave(rep_k, dim=1)
    if rep_v > 1:
        v_f = v_f.repeat_interleave(rep_v, dim=1)

    scores = torch.matmul(q_f, k_f.transpose(-2, -1)) * scale_value
    if attn_mask is not None:
        scores = scores + attn_mask.to(dtype=torch.float32)
    if is_causal:
        causal = torch.ones(
            scores.shape[-2],
            scores.shape[-1],
            dtype=torch.bool,
            device=scores.device,
        ).tril()
        scores = scores.masked_fill(~causal, float("-inf"))

    probs = torch.exp(scores - stats_f)
    row_dot = (do_f * o_f).sum(dim=-1, keepdim=True)
    d_probs = torch.matmul(do_f, v_f.transpose(-2, -1))
    d_scores = probs * (d_probs - row_dot)
    d_scores_scaled = d_scores * scale_value

    dq = torch.matmul(d_scores_scaled, k_f)
    dk_full = torch.matmul(d_scores_scaled.transpose(-2, -1), q_f)
    dv_full = torch.matmul(probs.transpose(-2, -1), do_f)

    batch, seq_kv = dk_full.shape[0], dk_full.shape[2]
    if rep_k > 1:
        dk_f = dk_full.view(batch, k_heads, rep_k, seq_kv, head_dim).sum(dim=2)
    else:
        dk_f = dk_full
    if rep_v > 1:
        dv_f = dv_full.view(batch, v_heads, rep_v, seq_kv, int(v.shape[-1])).sum(dim=2)
    else:
        dv_f = dv_full

    _store_tensor(tensors, dq_uid, dq.to(dtype=q.dtype))
    _store_tensor(tensors, dk_uid, dk_f.to(dtype=k.dtype))
    _store_tensor(tensors, dv_uid, dv_f.to(dtype=v.dtype))


@register_handler("PointwiseAttributes")
def handle_pointwise(
    node: Dict[str, Any],
    tensors: Dict[int, torch.Tensor],
    graph_json: Dict[str, Any],
) -> None:
    """Handle PointwiseAttributes (element-wise operations).

    Supports: relu_fwd, add, mul, sub, div, sqrt, abs, neg, exp, log, tanh_fwd, sigmoid_fwd
    """
    inputs = node.get("inputs", {})
    outputs = node.get("outputs", {})

    operation = inputs.get("operation", "")
    in0_uid = inputs.get("in_0_tensor_uid")
    in1_uid = inputs.get("in_1_tensor_uid")
    out_uid = outputs.get("out_0_tensor_uid")

    if in0_uid is None or out_uid is None:
        raise ValueError(f"Pointwise node missing required tensor UIDs: {node}")

    in0 = tensors[in0_uid]
    in1 = tensors.get(in1_uid) if in1_uid is not None else None

    # Map operation to PyTorch equivalent
    if operation == "relu_fwd":
        # Check for clipping bounds (ReLU6-style)
        lower_clip = inputs.get("relu_lower_clip", 0.0)
        upper_clip = inputs.get("relu_upper_clip", float("inf"))

        if upper_clip == float("inf") or upper_clip >= 1e30:
            # Standard ReLU
            out = F.relu(in0)
        else:
            # Clipped ReLU (e.g., ReLU6)
            out = torch.clamp(in0, min=lower_clip, max=upper_clip)

    elif operation == "add":
        if in1 is None:
            raise ValueError("Add operation requires two inputs")
        out = in0 + in1

    elif operation == "mul":
        if in1 is None:
            raise ValueError("Mul operation requires two inputs")
        out = in0 * in1

    elif operation == "sub":
        if in1 is None:
            raise ValueError("Sub operation requires two inputs")
        out = in0 - in1

    elif operation == "div":
        if in1 is None:
            raise ValueError("Div operation requires two inputs")
        out = in0 / in1

    elif operation == "sqrt":
        out = torch.sqrt(in0)

    elif operation == "abs":
        out = torch.abs(in0)

    elif operation == "neg":
        out = -in0

    elif operation == "exp":
        out = torch.exp(in0)

    elif operation == "log":
        out = torch.log(in0)

    elif operation == "tanh_fwd":
        out = torch.tanh(in0)

    elif operation == "sigmoid_fwd":
        out = torch.sigmoid(in0)

    else:
        raise ValueError(f"Unsupported pointwise operation: {operation}")

    _store_tensor(tensors, out_uid, out)
