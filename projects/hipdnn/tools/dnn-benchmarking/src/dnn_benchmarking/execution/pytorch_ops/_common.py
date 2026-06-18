# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Shared tensor and graph-node helpers for PyTorch reference handlers."""

from math import prod
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import torch

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
    "_reject_peer_stats",
    "_channel_broadcast",
    "_scalar_value",
    "_numel",
    "_stored_tensor_shape",
    "_store_tensor_for_uid",
    "_strip_leading_singletons",
    "_sum_to_shape",
]


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
