# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""hipDNN SDPA forward oracle driven by a ``dispatch.sdpa.SdpaRequest``.

An adapter over :func:`rocke.numeric.sdpa_reference.sdpa_reference`: it
decodes what that array oracle does not model (flat strided buffers, packed
ragged buffers, the scale source, FP8 descales) and delegates the attention
math, so there is one SDPA forward definition.

The request is read by attribute rather than imported, because ``platform``
must not depend on ``library``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np

from rocke.numeric.sdpa_reference import sdpa_reference

__all__ = ["sdpa_reference_from_request"]


def _strided_view(
    name: str, buffer: np.ndarray, base: int, shape: tuple, strides: tuple
) -> np.ndarray:
    """Read ``shape`` from a flat buffer at element offset ``base`` with element strides."""
    if any(s < 0 for s in strides):
        raise ValueError(f"{name}: negative strides {strides} are not supported")
    if 0 in shape:
        return np.zeros(shape, dtype=buffer.dtype)
    last = base + sum((n - 1) * s for n, s in zip(shape, strides))
    if base < 0 or last >= buffer.size:
        raise ValueError(
            f"{name}: dims {shape} with strides {strides} at offset {base} need "
            f"{last + 1} elements, buffer has {buffer.size}"
        )
    item = buffer.itemsize
    return np.lib.stride_tricks.as_strided(
        buffer[base:], shape=shape, strides=tuple(s * item for s in strides)
    )


def _dense_operand(name: str, array: Any, tensor: Any) -> np.ndarray:
    """An operand as its logical array.

    ``array`` is either already logical (its shape is ``tensor.dims``, or
    reshapes to it, as a lower-rank bias does), or a 1-D flat buffer read
    through ``tensor.strides``.
    """
    dims = tuple(tensor.dims)
    arr = np.asarray(array)
    if arr.shape == dims:
        return arr
    if arr.ndim == 1:
        flat = np.ascontiguousarray(arr)
        return _strided_view(name, flat, 0, dims, tuple(tensor.strides))
    if arr.size == math.prod(dims):
        return arr.reshape(dims)
    raise ValueError(
        f"{name}: array shape {arr.shape} is neither the logical dims {dims} nor a "
        "flat buffer"
    )


def _ragged_operand(
    name: str, array: Any, offsets: Any, tensor: Any
) -> tuple[np.ndarray, np.ndarray]:
    """A packed ragged operand as ``(logical BHSD array, per-batch lengths)``.

    Batch ``b`` starts at element ``ragged_offset[b] * ragged_offset_multiplier``
    and spans ``(offset[b + 1] - offset[b]) / seq_stride`` sequence rows; the
    batch stride is not used. Rows past a batch's length are zero.
    """
    batch, heads, s_max, d = tensor.dims
    _, s_h, s_s, s_d = tensor.strides
    flat = np.ascontiguousarray(np.asarray(array).reshape(-1))
    elems = np.asarray(offsets).reshape(-1).astype(np.int64)
    elems = elems * int(tensor.ragged_offset_multiplier)
    if elems.size != batch + 1:
        raise ValueError(
            f"{name}: ragged offsets need B + 1 = {batch + 1} entries, got {elems.size}"
        )
    if s_s <= 0 or np.any(elems % s_s):
        raise ValueError(
            f"{name}: ragged offsets {elems.tolist()} are not whole sequence rows "
            f"(sequence stride {s_s})"
        )
    lengths = np.diff(elems) // s_s
    if np.any(lengths < 0) or np.any(lengths > s_max):
        raise ValueError(
            f"{name}: ragged lengths {lengths.tolist()} must be in [0, S_max={s_max}]"
        )
    out = np.zeros(tensor.dims, dtype=flat.dtype)
    for b in range(batch):
        n = int(lengths[b])
        out[b, :, :n, :] = _strided_view(
            f"{name}[{b}]", flat, int(elems[b]), (heads, n, d), (s_h, s_s, s_d)
        )
    return out, lengths


def _seq_lengths(name: str, array: Any, batch: int, extent: int) -> np.ndarray:
    lengths = np.asarray(array).reshape(-1).astype(np.int64)
    if lengths.size != batch:
        raise ValueError(
            f"{name}: need one length per batch ({batch}), got {lengths.size}"
        )
    if np.any(lengths < 0) or np.any(lengths > extent):
        raise ValueError(f"{name}: lengths {lengths.tolist()} must be in [0, {extent}]")
    return lengths


def _descale(
    name: str, tensors: Mapping, present: bool, batch: int, heads_kv: int, h_q: int
) -> np.ndarray:
    """Per ``(batch, query head)`` descale factor; 1 when the descale is absent.

    hipDNN indexes a ``(B, H_kv, 1, 1)`` descale by the KV head the query head
    reads, so the Q descale follows K's grouping, not Q's own head.
    """
    if not present:
        return np.ones((batch, h_q), dtype=np.float64)
    values = np.asarray(tensors[name], dtype=np.float64).reshape(-1)
    if values.size == 1:
        return np.full((batch, h_q), values[0], dtype=np.float64)
    if values.size != batch * heads_kv:
        raise ValueError(
            f"{name}: must be a scalar or (B, H_kv, 1, 1) = ({batch}, {heads_kv}, 1, 1), "
            f"got {values.size} values"
        )
    return values.reshape(batch, heads_kv)[:, np.arange(h_q) // (h_q // heads_kv)]


def _unsupported_sdpa_features(req: Any) -> list[str]:
    fp8 = getattr(req, "fp8", None)
    s_o_fields = ("descale_s", "scale_s", "scale_o", "amax_s", "amax_o")
    checks = {
        "dropout": req.dropout is not None,
        "paged": req.paged is not None,
        "block_mask": req.block_mask is not None,
        "sinks": req.sink_token is not None,
        "alibi": bool(req.alibi),
        "fp8_s_o": fp8 is not None
        and any(getattr(fp8, f) is not None for f in s_o_fields),
        "max_sum_exp": req.max is not None or req.sum_exp is not None,
    }
    return [name for name, active in checks.items() if active]


def sdpa_reference_from_request(
    req: Any, tensors: Mapping[str, Any]
) -> dict[str, np.ndarray]:
    """hipDNN SDPA forward for a ``dispatch.sdpa.SdpaRequest``.

    The pipeline is hipDNN's (``CpuFpReferenceSdpa::forward``)::

        s     = (Q . K^T) * scale * descale_q * descale_k
        s    += bias[broadcast(b, h_q, q, k)]       # unscaled, by query head
        s     = -inf where not keep(q, k)
        p     = softmax(s)                          # fully masked row -> p = 0
        O     = (p . V) * descale_v
        stats = max + ln(sum_exp)                   # -inf on a fully masked row

    The scale is always the request's (hipDNN's default is 1.0, not
    ``1/sqrt(d)``). The descales are folded into Q and O, and the rest is
    :func:`rocke.numeric.sdpa_reference.sdpa_reference` in float64.

    ``tensors`` maps operand names to arrays: ``q``, ``k``, ``v`` always;
    ``scale`` when ``req.scale_source == "tensor"``; ``bias``; ``seq_len_q`` /
    ``seq_len_kv``; ``descale_q`` / ``descale_k`` / ``descale_v``; and
    ``ragged_offset_q`` / ``ragged_offset_k`` / ``ragged_offset_v`` (V defaults to
    K's) for ragged operands. A dense operand is either its logical array or a
    1-D buffer read through the request's strides; a ragged one is the packed
    buffer. Pass fp8 and bf16 data as float32 values.

    Per batch, the effective lengths come from ``seq_len_*`` when present, else
    from the ragged offsets, else the full extent; the diagonal offset uses the
    effective lengths. Returns float32 logical arrays: ``o`` ``(B, H_q, S_q, D_v)``,
    ``stats`` ``(B, H_q, S_q, 1)``, and ``row_valid`` ``(B, S_q)``. Rows past a
    batch's ``s_q`` hold 0 in ``o`` and ``-inf`` in ``stats``, as in hipDNN's
    golden generator; kernels may leave them unwritten, so compare only
    ``row_valid`` rows.

    Raises ``NotImplementedError`` naming the feature for dropout, paged KV,
    block masks, sinks, ALiBi, FP8 softmax/output scaling, and max/sum_exp
    outputs, which hipDNN's CPU reference rejects too.
    """
    unsupported = _unsupported_sdpa_features(req)
    if unsupported:
        raise NotImplementedError(
            f"sdpa_reference_from_request does not implement {', '.join(unsupported)}"
        )
    batch, h_q, s_q, _ = req.q.dims
    h_k, s_kv = req.k.dims[1], req.k.dims[2]
    h_v = req.v.dims[1]

    def operand(name: str, offsets_key: str) -> tuple[np.ndarray, np.ndarray | None]:
        tensor = getattr(req, name)
        if not tensor.ragged:
            return _dense_operand(name, tensors[name], tensor), None
        if offsets_key not in tensors:
            raise KeyError(f"ragged operand {name!r} needs tensors[{offsets_key!r}]")
        return _ragged_operand(name, tensors[name], tensors[offsets_key], tensor)

    v_offsets = "ragged_offset_v" if "ragged_offset_v" in tensors else "ragged_offset_k"
    q, q_lengths = operand("q", "ragged_offset_q")
    k, k_lengths = operand("k", "ragged_offset_k")
    v, _ = operand("v", v_offsets)

    def lengths(key: str, ragged: np.ndarray | None, extent: int) -> np.ndarray:
        if getattr(req, key) is not None:
            chosen = _seq_lengths(key, tensors[key], batch, extent)
            if ragged is not None and np.any(chosen > ragged):
                raise ValueError(
                    f"{key} {chosen.tolist()} exceeds the ragged lengths {ragged.tolist()}"
                )
            return chosen
        return ragged if ragged is not None else np.full(batch, extent, np.int64)

    q_eff = lengths("seq_len_q", q_lengths, s_q)
    kv_eff = lengths("seq_len_kv", k_lengths, s_kv)

    if req.scale_source == "tensor":
        scale = float(np.asarray(tensors["scale"], dtype=np.float32).reshape(-1)[0])
    else:
        scale = float(req.scale)
    fp8 = getattr(req, "fp8", None)

    def has(field: str) -> bool:
        return fp8 is not None and getattr(fp8, field) is not None

    descale_qk = _descale(
        "descale_q", tensors, has("descale_q"), batch, h_k, h_q
    ) * _descale("descale_k", tensors, has("descale_k"), batch, h_k, h_q)
    descale_v = _descale("descale_v", tensors, has("descale_v"), batch, h_v, h_q)

    bias = None
    if req.bias is not None:
        bias = _dense_operand("bias", tensors["bias"], req.bias.tensor)

    out, lse = sdpa_reference(
        np.asarray(q, dtype=np.float64) * descale_qk[:, :, None, None],
        k,
        v,
        scale=scale,
        bias=bias,
        left_bound=req.left_bound,
        right_bound=req.right_bound,
        top_left=req.diagonal == "top_left",
        seq_len_q=q_eff,
        seq_len_kv=kv_eff,
    )
    out = out * descale_v[:, :, None, None]
    row_valid = np.arange(s_q)[None, :] < q_eff[:, None]
    return {
        "o": out.astype(np.float32),
        "stats": lse[..., None].astype(np.float32),
        "row_valid": row_valid,
    }
