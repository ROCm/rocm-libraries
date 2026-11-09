# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""hipDNN SDPA forward oracle driven by a ``dispatch.sdpa.SdpaRequest``.

An adapter over :func:`rocke.numeric.sdpa_reference.sdpa_reference`: it
decodes what that array oracle does not model (flat strided buffers, packed
ragged buffers, the scale source, FP8 descales, hipDNN's left bound) and
delegates the attention math, so there is one SDPA forward definition. It
lives beside the SDPA tests rather than in ``platform``, so it reads the
request's own facts (:meth:`SdpaRequest.features`) instead of re-deriving them.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np

from dispatch.sdpa import SdpaRequest
from rocke.numeric.sdpa_reference import hipdnn_left_to_window, sdpa_reference
from .helpers import strided_view

__all__ = ["sdpa_reference_from_request"]


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
        return strided_view(name, flat, 0, dims, tuple(tensor.strides))
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
        out[b, :, :n, :] = strided_view(
            f"{name}[{b}]", flat, int(elems[b]), (heads, n, d), (s_h, s_s, s_d)
        )
    return out, lengths


def _seq_lengths(name: str, array: Any, batch: int, extent: int) -> np.ndarray:
    # SdpaRequest fixes the shape at [B, 1, 1, 1], so there is one per batch.
    lengths = np.asarray(array).reshape(-1).astype(np.int64)
    if np.any(lengths < 0) or np.any(lengths > extent):
        raise ValueError(f"{name}: lengths {lengths.tolist()} must be in [0, {extent}]")
    return lengths


def _descale(
    name: str, tensors: Mapping, tensor: Any, batch: int, heads_kv: int, h_q: int
) -> np.ndarray:
    """Per ``(batch, query head)`` descale factor; 1 when ``tensor`` is ``None``.

    hipDNN indexes a ``(B, H_kv, 1, 1)`` descale by the KV head the query head
    reads, so the Q descale follows K's grouping, not Q's own head. The array
    is decoded through ``tensor``'s strides like any other dense operand.
    """
    if tensor is None:
        return np.ones((batch, h_q), dtype=np.float64)
    logical = _dense_operand(name, tensors[name], tensor)
    values = np.asarray(logical, dtype=np.float64).reshape(-1)
    if values.size == 1:
        return np.full((batch, h_q), values[0], dtype=np.float64)
    if values.size != batch * heads_kv:
        raise ValueError(
            f"{name}: must be a scalar or (B, H_kv, 1, 1) = ({batch}, {heads_kv}, 1, 1), "
            f"got {values.size} values"
        )
    return values.reshape(batch, heads_kv)[:, np.arange(h_q) // (h_q // heads_kv)]


# Features this oracle does not model yet, by SdpaRequest.features() name. hipDNN's
# CPU reference implements them; a kernel test that needs one compares against it.
UNSUPPORTED_FEATURES = (
    "dropout",
    "paged",
    "block_mask",
    "sinks",
    "alibi",
    "fp8_s_o",
    "max_sum_exp",
)


def sdpa_reference_from_request(
    req: SdpaRequest, tensors: Mapping[str, Any]
) -> dict[str, np.ndarray]:
    """hipDNN SDPA forward for a ``dispatch.sdpa.SdpaRequest``.

    The pipeline is hipDNN's (``CpuFpReferenceSdpa::forward``)::

        s     = (Q . K^T) * scale * descale_q * descale_k
        s    += bias[broadcast(b, h_q, q, k)]       # unscaled, by query head
        s     = -inf where not keep(q, k)
        p     = softmax(s)                          # fully masked row -> p = 0
        O     = (p . V) * descale_v
        stats = max + ln(sum_exp)                   # -inf on a fully masked row

    ``keep`` is hipDNN's band: ``left_bound = L`` keeps the ``L`` keys ending at
    the diagonal, converted once to the array oracle's window ``L - 1``. The
    scale is always the request's (hipDNN's default is 1.0, not
    ``1/sqrt(d)``). The descales are folded into Q and O, and the rest is
    :func:`rocke.numeric.sdpa_reference.sdpa_reference` in float64.

    ``tensors`` maps operand names to arrays: ``q``, ``k``, ``v`` always;
    ``scale`` when ``req.scale_source == "tensor"``; ``bias``; ``seq_len_q`` /
    ``seq_len_kv``; ``descale_q`` / ``descale_k`` / ``descale_v``; and
    ``ragged_offset_q`` / ``ragged_offset_k`` / ``ragged_offset_v`` (V defaults to
    K's) for ragged operands. A dense operand, ``seq_len_*`` and the descales
    included, is either its logical array or a 1-D buffer read through the
    request's strides; a ragged one is the packed buffer, and ragged K and V
    must hold the same lengths. Pass fp8 and bf16 data as float32 values.

    Per batch, the effective lengths come from ``seq_len_*`` when present, else
    from the ragged offsets, else the full extent; the diagonal offset uses the
    effective lengths. Returns float32 logical arrays: ``o`` ``(B, H_q, S_q, D_v)``,
    ``stats`` ``(B, H_q, S_q, 1)``, and ``row_valid`` ``(B, S_q)``. Rows past a
    batch's ``s_q`` hold 0 in ``o`` and ``-inf`` in ``stats``, as in hipDNN's
    golden generator; kernels may leave them unwritten, so compare only
    ``row_valid`` rows.

    Raises ``NotImplementedError`` naming the feature for dropout, paged KV,
    block masks, sinks, ALiBi, FP8 softmax/output scaling, and max/sum_exp
    outputs, which this oracle does not model yet (hipDNN's CPU reference does).
    """
    features = req.features()
    unsupported = [name for name in UNSUPPORTED_FEATURES if name in features]
    if unsupported:
        raise NotImplementedError(
            f"sdpa_reference_from_request does not implement {', '.join(unsupported)}"
        )
    batch, h_q, s_q, _ = req.q.dims
    h_k, s_kv = req.k.dims[1], req.s_kv
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
    v, v_lengths = operand("v", v_offsets)
    # As CpuFpReferenceSdpaRagged: K and V must pack the same tokens per batch,
    # or a key would be paired with a zero-filled value row.
    if (
        k_lengths is not None
        and v_lengths is not None
        and not np.array_equal(k_lengths, v_lengths)
    ):
        raise ValueError(
            f"ragged K lengths {k_lengths.tolist()} and V lengths "
            f"{v_lengths.tolist()} differ"
        )

    def lengths(key: str, ragged: np.ndarray | None, extent: int) -> np.ndarray:
        tensor = getattr(req, key)
        if tensor is not None:
            logical = _dense_operand(key, tensors[key], tensor)
            chosen = _seq_lengths(key, logical, batch, extent)
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
    fp8 = req.fp8

    def descriptor(field: str) -> Any:
        return None if fp8 is None else getattr(fp8, field)

    descale_qk = _descale(
        "descale_q", tensors, descriptor("descale_q"), batch, h_k, h_q
    ) * _descale("descale_k", tensors, descriptor("descale_k"), batch, h_k, h_q)
    descale_v = _descale("descale_v", tensors, descriptor("descale_v"), batch, h_v, h_q)

    bias = None
    if req.bias is not None:
        bias = _dense_operand("bias", tensors["bias"], req.bias.tensor)

    out, lse = sdpa_reference(
        np.asarray(q, dtype=np.float64) * descale_qk[:, :, None, None],
        k,
        v,
        scale=scale,
        bias=bias,
        left_bound=hipdnn_left_to_window(req.left_bound),
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
