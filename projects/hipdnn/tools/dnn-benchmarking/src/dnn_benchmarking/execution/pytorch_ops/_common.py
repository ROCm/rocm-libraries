# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Shared tensor and graph-node helpers for PyTorch reference handlers."""

from math import prod
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import torch

from ...common.exceptions import UnsupportedGraphError

__all__ = [
    "_as_tuple",
    "_node_section",
    "_node_param",
    "_node_uid",
    "_required_input_uid",
    "_required_output_uid",
    "_optional_uid",
    "_tensor",
    "_tensor_shape",
    "_store_tensor",
    "_store_channel_tensor",
    "_channel_values",
    "_effective_compute_type",
    "_is_float32_compute",
    "_reject_peer_stats",
    "_require_fp32_stat",
    "_channel_broadcast",
    "_scalar_value",
    "_numel",
    "_store_planned",
    "_strip_leading_singletons",
    "_sum_to_shape",
    "ReplayTensors",
]


class ReplayTensors(dict):
    """Tensor map that memoizes host-resolved scalar reads.

    Reading a scalar input (e.g. batchnorm/layernorm epsilon, SDPA scale) calls
    ``Tensor.item()``, a device->host copy that synchronizes the stream. During
    a stalled-queue benchmark the work stream is gated, so any such sync inside
    the timed region would deadlock. Because the benchmark replays the same
    inputs every iteration, those scalars are constant: resolve each once and
    cache it here so timed iterations are pure asynchronous enqueue.

    Plain ``dict`` inputs (the one-shot reference path) carry no cache, so
    ``_scalar_value`` resolves normally with no behavior change.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._scalar_cache: Dict[int, float] = {}


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
    # Replace the dict entry with the op's own output rather than copying into a
    # pre-allocated buffer. The buffer manager shares this dict by reference
    # (get_tensors) and reads outputs back device->host only after the timed
    # region, so the timed execution avoids an unnecessary device-to-device copy
    # every iteration. Coerce only the declared dtype/device, which is a no-op
    # (no copy, no kernel) when the op already produced them -- the common case.
    existing = tensors.get(uid)
    if (
        existing is not None
        and existing is not value
        and (value.dtype != existing.dtype or value.device != existing.device)
    ):
        value = value.to(dtype=existing.dtype, device=existing.device)
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


def _channel_values(
    tensor: torch.Tensor,
    x: torch.Tensor,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    values = tensor.reshape(-1)
    if dtype is not None:
        values = values.to(dtype=dtype)
    if x.ndim < 2:
        raise ValueError("Batchnorm tensors require at least 2 dimensions")
    if values.numel() != x.shape[1]:
        raise ValueError(
            f"Batchnorm channel tensor has {values.numel()} elements, expected {x.shape[1]}"
        )
    return values


def _effective_compute_type(node: Dict[str, Any], graph_json: Dict[str, Any]) -> str:
    """Resolve a node's compute_data_type: node value, else graph-level, else 'float'."""
    cdt = node.get("compute_data_type")
    if not cdt or str(cdt).lower() == "unset":
        cdt = graph_json.get("compute_data_type", "float")
    return str(cdt)


def _is_float32_compute(node: Dict[str, Any], graph_json: Dict[str, Any]) -> bool:
    """True only for hipDNN's canonical float32 token: DataType::FLOAT serializes
    to exactly the lowercase string "float" (no "fp32"/"float32" JSON alias)."""
    return _effective_compute_type(node, graph_json) == "float"


def _require_fp32_stat(values: torch.Tensor, name: str) -> torch.Tensor:
    """Saved-statistic tensors (batchnorm mean/variance/inv_variance, RMSNorm
    inv_rms, SDPA log-sum-exp) are produced and consumed in float32 by this
    reference. A graph that declares such a stat tensor in a lower precision
    cannot be represented faithfully — the fp32 value the reference computes is
    not what the graph carries — so the reference declares the graph inapplicable
    rather than silently promoting the stat to float32."""
    if values.dtype != torch.float32:
        raise UnsupportedGraphError(
            f"reference requires a float32 {name} stat tensor; graph declares "
            f"{values.dtype}"
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
    # Reuse a previously resolved value when the caller supplies a caching
    # tensor map (ReplayTensors); this avoids the .item() device->host sync on
    # every replay so a stalled-queue benchmark stays a pure asynchronous
    # enqueue. Plain dicts have no cache and resolve normally.
    cache = getattr(tensors, "_scalar_cache", None)
    if cache is not None and uid in cache:
        return cache[uid]
    tensor = _tensor(tensors, uid, node)
    if tensor.numel() < 1:
        raise ValueError(f"Scalar tensor UID {uid} is empty")
    value = float(tensor.detach().reshape(-1)[0].item())
    if cache is not None:
        cache[uid] = value
    return value


def _numel(shape: Sequence[int]) -> int:
    return prod(int(dim) for dim in shape)


def _store_planned(
    tensors: Dict[int, torch.Tensor],
    uid: int,
    value: torch.Tensor,
    declared_shape: Optional[Tuple[int, ...]],
) -> None:
    """Store ``value`` for ``uid``, reshaping to the shape declared at plan time.

    A live tensor already present in ``tensors`` wins over ``declared_shape``.
    Reshape is guarded by a numel-equality check that raises on a true mismatch.
    """
    existing = tensors.get(uid)
    shape = (
        tuple(int(dim) for dim in existing.shape)
        if existing is not None
        else declared_shape
    )
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
