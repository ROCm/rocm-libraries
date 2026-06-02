# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""PyTorch operation implementations for graph execution.

These handlers execute on the device of the input tensors (CPU or CUDA).
Used by both PyTorchReferenceProvider (CPU) and PyTorchCudaExecutor (CUDA).
"""

from math import sqrt
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


def _channel_broadcast(values: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return values.reshape([1, values.numel()] + [1] * (x.ndim - 2)).to(device=x.device)


def _scalar_value(
    tensors: Dict[int, torch.Tensor], uid: int, node: Dict[str, Any]
) -> float:
    tensor = _tensor(tensors, uid, node)
    if tensor.numel() < 1:
        raise ValueError(f"Scalar tensor UID {uid} is empty")
    return float(tensor.detach().reshape(-1)[0].item())


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
    return F.conv2d(padded_x, w, stride=stride, dilation=dilation)


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


def _sdpa_common(
    node: Dict[str, Any],
    tensors: Dict[int, torch.Tensor],
    q: torch.Tensor,
    k: torch.Tensor,
) -> Tuple[Optional[torch.Tensor], float, bool, Optional[float], bool]:
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
    if q.ndim < 3 or k.ndim < 3:
        raise ValueError("SDPA expects q/k/v tensors with head and matrix dimensions")
    q_heads = int(q.shape[-3])
    kv_heads = int(k.shape[-3])
    enable_gqa = q_heads != kv_heads
    if enable_gqa and (kv_heads == 0 or q_heads % kv_heads != 0):
        raise ValueError(
            f"Unsupported GQA head counts: q_heads={q_heads}, kv_heads={kv_heads}"
        )
    return attn_mask, dropout_p, is_causal, scale, enable_gqa


def _call_sdpa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask: Optional[torch.Tensor],
    dropout_p: float,
    is_causal: bool,
    scale: Optional[float],
    enable_gqa: bool,
) -> torch.Tensor:
    kwargs: Dict[str, Any] = {}
    if scale is not None:
        kwargs["scale"] = scale
    if enable_gqa:
        kwargs["enable_gqa"] = True
    try:
        return F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
            **kwargs,
        )
    except TypeError:
        if not enable_gqa:
            raise
        repeat = q.shape[-3] // k.shape[-3]
        return F.scaled_dot_product_attention(
            q,
            k.repeat_interleave(repeat, dim=-3),
            v.repeat_interleave(repeat, dim=-3),
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
            **{key: value for key, value in kwargs.items() if key != "enable_gqa"},
        )


def _sdpa_stats(
    q: torch.Tensor,
    k: torch.Tensor,
    attn_mask: Optional[torch.Tensor],
    is_causal: bool,
    scale: Optional[float],
    enable_gqa: bool,
) -> torch.Tensor:
    q_float = q.to(dtype=torch.float32)
    k_float = k.to(dtype=torch.float32)
    if enable_gqa:
        repeat = q.shape[-3] // k.shape[-3]
        k_float = k_float.repeat_interleave(repeat, dim=-3)
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
    return torch.logsumexp(scores, dim=-1)


def _sdpa_consistency_tolerances(dtype: torch.dtype) -> Tuple[float, float]:
    if dtype == torch.bfloat16:
        return 6e-3, 3e-3
    if dtype == torch.float16:
        return 1e-3, 1e-3
    return 1e-5, 1e-6


def _assert_sdpa_consistent(
    actual: torch.Tensor,
    expected: torch.Tensor,
    name: str,
    dtype: torch.dtype,
) -> None:
    rtol, atol = _sdpa_consistency_tolerances(dtype)
    if not torch.allclose(
        actual.to(dtype=torch.float32),
        expected.to(dtype=torch.float32),
        rtol=rtol,
        atol=atol,
    ):
        diff = (actual.to(dtype=torch.float32) - expected.to(dtype=torch.float32)).abs()
        raise ValueError(
            f"SDPA backward input '{name}' is inconsistent with q/k/v "
            f"(max_abs_diff={float(diff.max())}, rtol={rtol}, atol={atol})"
        )


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
    if _conv_padding_is_symmetric(node):
        dx = torch.nn.grad.conv2d_input(
            input_size,
            w,
            dy,
            stride=stride,
            padding=pre,
            dilation=dilation,
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
    if _conv_padding_is_symmetric(node):
        dw = torch.nn.grad.conv2d_weight(
            x,
            weight_size,
            dy,
            stride=stride,
            padding=pre,
            dilation=dilation,
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
    """Handle batchnorm inference with variance and epsilon."""
    x_uid = _required_input_uid(node, "x_tensor_uid")
    mean_uid = _required_input_uid(node, "mean_tensor_uid")
    variance_uid = _required_input_uid(node, "variance_tensor_uid")
    scale_uid = _required_input_uid(node, "scale_tensor_uid")
    bias_uid = _required_input_uid(node, "bias_tensor_uid")
    epsilon_uid = _required_input_uid(node, "epsilon_tensor_uid")
    y_uid = _required_output_uid(node, "y_tensor_uid")

    x = _tensor(tensors, x_uid, node)
    variance = _channel_values(_tensor(tensors, variance_uid, node), x)
    epsilon = _scalar_value(tensors, epsilon_uid, node)
    inv_variance = torch.rsqrt(variance + epsilon)
    y = _bn_affine(
        x,
        _channel_values(_tensor(tensors, scale_uid, node), x),
        _channel_values(_tensor(tensors, bias_uid, node), x),
        _channel_values(_tensor(tensors, mean_uid, node), x),
        inv_variance,
    )
    _store_tensor(tensors, y_uid, y)


@register_handler("BatchnormAttributes")
def handle_batchnorm_training(
    node: Dict[str, Any],
    tensors: Dict[int, torch.Tensor],
    graph_json: Dict[str, Any],
) -> None:
    """Handle batchnorm forward training."""
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


@register_handler("SdpaAttributes")
def handle_sdpa(
    node: Dict[str, Any],
    tensors: Dict[int, torch.Tensor],
    graph_json: Dict[str, Any],
) -> None:
    """Handle scaled dot-product attention forward."""
    q_uid = _required_input_uid(node, "q_tensor_uid")
    k_uid = _required_input_uid(node, "k_tensor_uid")
    v_uid = _required_input_uid(node, "v_tensor_uid")
    o_uid = _required_output_uid(node, "o_tensor_uid")

    q = _tensor(tensors, q_uid, node)
    k = _tensor(tensors, k_uid, node)
    v = _tensor(tensors, v_uid, node)
    attn_mask, dropout_p, is_causal, scale, enable_gqa = _sdpa_common(
        node, tensors, q, k
    )
    o = _call_sdpa(q, k, v, attn_mask, dropout_p, is_causal, scale, enable_gqa)
    _store_tensor(tensors, o_uid, o)

    stats_uid = _optional_uid(node, "stats_tensor_uid")
    if stats_uid is not None:
        _store_tensor(
            tensors,
            stats_uid,
            _sdpa_stats(q, k, attn_mask, is_causal, scale, enable_gqa),
        )


@register_handler("SdpaBackwardAttributes")
def handle_sdpa_backward(
    node: Dict[str, Any],
    tensors: Dict[int, torch.Tensor],
    graph_json: Dict[str, Any],
) -> None:
    """Handle scaled dot-product attention backward for dropout-free graphs."""
    unsupported = [
        "seq_len_q_tensor_uid",
        "seq_len_kv_tensor_uid",
        "seed_tensor_uid",
        "offset_tensor_uid",
        "dropout_mask_tensor_uid",
        "dropout_scale_tensor_uid",
        "dropout_scale_inv_tensor_uid",
    ]
    _sdpa_unsupported_if_present(node, unsupported)

    q_uid = _required_input_uid(node, "q_tensor_uid")
    k_uid = _required_input_uid(node, "k_tensor_uid")
    v_uid = _required_input_uid(node, "v_tensor_uid")
    o_uid = _required_input_uid(node, "o_tensor_uid")
    do_uid = _required_input_uid(node, "do_tensor_uid")
    stats_uid = _required_input_uid(node, "stats_tensor_uid")
    dq_uid = _required_output_uid(node, "dq_tensor_uid")
    dk_uid = _required_output_uid(node, "dk_tensor_uid")
    dv_uid = _required_output_uid(node, "dv_tensor_uid")

    q_base = _tensor(tensors, q_uid, node)
    k_base = _tensor(tensors, k_uid, node)
    v_base = _tensor(tensors, v_uid, node)
    o = _tensor(tensors, o_uid, node)
    do = _tensor(tensors, do_uid, node)
    stats = _tensor(tensors, stats_uid, node)
    attn_mask, dropout_p, is_causal, scale, enable_gqa = _sdpa_common(
        node, tensors, q_base, k_base
    )
    if dropout_p != 0.0:
        raise ValueError(
            "Nonzero SDPA dropout cannot be exactly validated against PyTorch"
        )

    q = q_base.to(dtype=torch.float32)
    k = k_base.to(dtype=torch.float32)
    v = v_base.to(dtype=torch.float32)
    if enable_gqa:
        repeat = q.shape[-3] // k.shape[-3]
        k_for_attn = k.repeat_interleave(repeat, dim=-3)
        v_for_attn = v.repeat_interleave(repeat, dim=-3)
    else:
        repeat = 1
        k_for_attn = k
        v_for_attn = v

    scale_value = (1.0 / sqrt(float(q.shape[-1]))) if scale is None else float(scale)
    scores = torch.matmul(q, k_for_attn.transpose(-2, -1)) * scale_value
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

    expected_stats = torch.logsumexp(scores, dim=-1)
    _assert_sdpa_consistent(stats, expected_stats, "stats_tensor_uid", q_base.dtype)

    probs = torch.exp(scores - stats.to(dtype=torch.float32).unsqueeze(-1))
    expected_o = torch.matmul(probs, v_for_attn)
    _assert_sdpa_consistent(o, expected_o, "o_tensor_uid", q_base.dtype)

    do_f32 = do.to(dtype=torch.float32)
    o_f32 = o.to(dtype=torch.float32)
    d = (do_f32 * o_f32).sum(dim=-1)
    dp = torch.matmul(do_f32, v_for_attn.transpose(-2, -1))
    ds = probs * (dp - d.unsqueeze(-1))

    dq = torch.matmul(ds, k_for_attn) * scale_value
    dk = torch.matmul(ds.transpose(-2, -1), q) * scale_value
    dv = torch.matmul(probs.transpose(-2, -1), do_f32)

    if enable_gqa:
        kv_heads = k.shape[-3]
        prefix = dk.shape[:-3]
        dk = dk.reshape(*prefix, kv_heads, repeat, dk.shape[-2], dk.shape[-1]).sum(
            dim=-3
        )
        dv = dv.reshape(*prefix, kv_heads, repeat, dv.shape[-2], dv.shape[-1]).sum(
            dim=-3
        )

    _store_tensor(tensors, dq_uid, dq.to(dtype=q_base.dtype))
    _store_tensor(tensors, dk_uid, dk.to(dtype=k_base.dtype))
    _store_tensor(tensors, dv_uid, dv.to(dtype=v_base.dtype))

    dbias_uid = _optional_uid(node, "dbias_tensor_uid")
    if dbias_uid is not None:
        raise ValueError("SDPA dbias output is not supported by the PyTorch reference")


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
