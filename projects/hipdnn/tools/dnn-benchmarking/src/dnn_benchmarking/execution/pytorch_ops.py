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


def _store_tensor(
    tensors: Dict[int, torch.Tensor], uid: int, value: torch.Tensor
) -> None:
    existing = tensors.get(uid)
    if existing is not None and tuple(existing.shape) == tuple(value.shape):
        existing.copy_(value.to(dtype=existing.dtype, device=existing.device))
        tensors[uid] = existing
    else:
        tensors[uid] = value


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
