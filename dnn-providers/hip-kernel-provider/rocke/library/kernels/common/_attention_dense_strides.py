# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Launch-time Q/K/V/O strides of the dense attention kernels.

Every dense attention kernel (gfx942 and gfx950, every body) takes twelve
trailing ``i32`` kernel parameters: one ``(batch, head, token)`` element stride
triple per tensor Q, K, V and O, declared after every other parameter in the
order of :data:`DENSE_STRIDE_NAMES`. The head-dim stride is 1 and is not a
parameter. Tensors are logical ``[B, S, H, D]`` views (``[B, Skv, Hkv, D]`` for
K and V) of any storage.

This module is the single source of:

* the names (:data:`DENSE_STRIDE_NAMES`), their kernel parameters
  (:func:`declare_dense_stride_params`) and launcher signature
  (:func:`append_dense_stride_signature`);
* the packed values (:func:`packed_dense_strides`), the values read off tensor
  views (:func:`view_dense_strides`) and their canonical form
  (:func:`canonical_dense_strides`);
* the bytes a K/V slab load may reach (:func:`dense_slab_bytes`, one function for
  the host predicate and the kernel's buffer-resource bound);
* the host admission predicate (:func:`dense_strides_admit`) with the decline
  reasons of :data:`DENSE_STRIDE_REASONS`, and the runner's use of it
  (:func:`admit_dense_launch_strides`).

The predicate has a C++ twin in the hipDNN engines; both are pinned to one
vectors file (see ``library/tests/test_attention_dense_strided_admission.py``). Pure host
code apart from the two IR helpers: no torch import, no GPU.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import NamedTuple

from kernels.common._sdpa_fwd_strides import contiguous_strides
from kernels.common.attention_dense_spec import INT32_LIMIT

# The tensors in parameter order, and the three axes of each, in order.
DENSE_STRIDE_TENSORS = ("q", "k", "v", "o")
DENSE_STRIDE_AXES = ("batch", "head", "token")

# The 12 trailing i32 kernel parameters, in declaration order.
DENSE_STRIDE_NAMES = tuple(
    f"stride_{t}_{axis}" for t in DENSE_STRIDE_TENSORS for axis in DENSE_STRIDE_AXES
)

# Bytes per element of every dense tensor (fp16 / bf16). K and V of an fp8 K/V-cache
# spec are 1 byte per element: see :func:`dense_kv_elem_bytes`.
DENSE_ELEM_BYTES = 2
# Every stride of an axis with extent > 1 is a multiple of this many elements
# (16 bytes), and every base address a multiple of DENSE_POINTER_ALIGN bytes. One
# conservative rule for every body: the kernels tolerate less on some operands.
DENSE_STRIDE_MULTIPLE = 8
DENSE_POINTER_ALIGN = 16
# K/V loads may reach this many tiles past the padded length: an empty KV segment
# still loads tiles n_upper and n_upper + 1.
DENSE_SLAB_SLACK_TILES = 2
# Byte offsets of a batch/head slab base are 64-bit.
_INT64_LIMIT = 2**63

# Decline reasons of dense_strides_admit, in the order the checks run.
DENSE_STRIDE_REASONS = (
    "operand",
    "head_dim_stride",
    "stride_alignment",
    "overlap",
    "pointer_alignment",
    "size_limit",
)


# --------------------------------------------------------------------------- #
# Names and kernel ABI
# --------------------------------------------------------------------------- #


def declare_dense_stride_params(b) -> dict:
    """Declare the 12 stride parameters on IR builder ``b`` (``i32``, in
    :data:`DENSE_STRIDE_NAMES` order) and return them by name. Call it after
    every other parameter of the kernel: the strides are the constant tail of
    the dense ABI."""
    from rocke.core.ir import I32

    return {name: b.param(name, I32) for name in DENSE_STRIDE_NAMES}


def append_dense_stride_signature(sig):
    """``sig`` (a :class:`rocke.helpers.spec.SignatureBuilder`) with the 12
    ``i32`` stride scalars appended, mirroring
    :func:`declare_dense_stride_params`."""
    for name in DENSE_STRIDE_NAMES:
        sig = sig.scalar(name, "i32")
    return sig


# --------------------------------------------------------------------------- #
# Values
# --------------------------------------------------------------------------- #


def _tensor_dims(spec, tensor: str) -> tuple[int, int]:
    """``(seqlen, heads)`` of one tensor of ``spec``."""
    if tensor in ("q", "o"):
        return spec.seqlen_q, spec.num_query_heads
    return spec.seqlen_kv, spec.num_kv_heads


def dense_operand_dims(spec, tensor: str) -> tuple[int, int, int, int]:
    """Logical ``(B, H, S, D)`` extents of ``tensor`` (``"q"``, ``"k"``, ``"v"``
    or ``"o"``) for ``spec``: the axis order of :data:`DENSE_STRIDE_AXES` plus
    the head dim."""
    seqlen, heads = _tensor_dims(spec, tensor)
    return (spec.batch, heads, seqlen, spec.head_size)


def packed_dense_strides(spec) -> dict[str, int]:
    """The 12 stride values (element units) of packed ``[B, S, H, D]`` tensors
    of ``spec``, keyed by :data:`DENSE_STRIDE_NAMES` in that order."""
    out: dict[str, int] = {}
    for t in DENSE_STRIDE_TENSORS:
        seqlen, heads = _tensor_dims(spec, t)
        triple = contiguous_strides(
            "bshd", heads=heads, seqlen=seqlen, head_size=spec.head_size
        )
        for axis, value in zip(DENSE_STRIDE_AXES, triple):
            out[f"stride_{t}_{axis}"] = value
    return out


def canonical_dense_strides(spec, strides: Mapping[str, int]) -> dict[str, int]:
    """``strides`` (the 12 values) with every extent-1 axis canonicalized:
    batch and head to 0, token to the head size. An extent-1 axis is never
    stepped, so its stride carries no information and may be anything."""
    out = {}
    D = spec.head_size
    for t in DENSE_STRIDE_TENSORS:
        dims = dense_operand_dims(spec, t)
        for i, axis in enumerate(DENSE_STRIDE_AXES):
            name = f"stride_{t}_{axis}"
            if dims[i] == 1:
                out[name] = D if axis == "token" else 0
            else:
                out[name] = int(strides[name])
    return out


def packed_launch_strides(spec) -> dict[str, int]:
    """The 12 values a kernel of ``spec`` is launched with for packed
    ``[B, S, H, D]`` tensors: :func:`packed_dense_strides` in canonical form.
    Every launch path (``attention_dense_runtime_args``, the runners, the SDPA
    replay export) passes canonical strides, so a frozen launch replays the
    values the runner gave."""
    return canonical_dense_strides(spec, packed_dense_strides(spec))


@dataclass(frozen=True)
class DenseOperand:
    """One dense tensor as the predicate sees it.

    ``dims`` and ``strides`` are in logical ``(B, H, S, D)`` order (strides in
    elements; ``strides[3]`` is the head-dim stride). ``align`` is a power of
    two the base address is a multiple of, in bytes. ``virtual`` and ``ragged``
    (a ragged offset tensor) are graph properties the kernels cannot serve."""

    dims: tuple[int, ...]
    strides: tuple[int, ...]
    align: int = DENSE_POINTER_ALIGN
    virtual: bool = False
    ragged: bool = False


def _address_align(ptr: int) -> int:
    """The largest power of two (capped at 256) dividing address ``ptr``."""
    ptr = int(ptr)
    return 256 if ptr % 256 == 0 else ptr & -ptr


def dense_operand_from_view(view) -> DenseOperand:
    """:class:`DenseOperand` of a logical ``[B, S, H, D]`` tensor view (anything
    with ``.shape`` and ``.stride()``; ``.data_ptr()`` gives the alignment when
    present, otherwise :data:`DENSE_POINTER_ALIGN` is assumed). A view of another
    rank keeps its own dims, which the predicate then declines."""
    shape = tuple(int(n) for n in view.shape)
    st = tuple(int(s) for s in view.stride())
    ptr = getattr(view, "data_ptr", None)
    align = _address_align(ptr()) if callable(ptr) else DENSE_POINTER_ALIGN
    if len(shape) != 4 or len(st) != 4:
        return DenseOperand(dims=shape, strides=st, align=align)
    return DenseOperand(
        dims=(shape[0], shape[2], shape[1], shape[3]),
        strides=(st[0], st[2], st[1], st[3]),
        align=align,
    )


def packed_dense_operand(spec, tensor: str) -> DenseOperand:
    """:class:`DenseOperand` of a packed ``[B, S, H, D]`` tensor of ``spec``."""
    packed = packed_dense_strides(spec)
    return DenseOperand(
        dims=dense_operand_dims(spec, tensor),
        strides=tuple(packed[f"stride_{tensor}_{a}"] for a in DENSE_STRIDE_AXES) + (1,),
    )


def _has_strides(view) -> bool:
    return callable(getattr(view, "stride", None)) and hasattr(view, "shape")


def view_dense_strides(spec, *, q, k, v, o) -> dict[str, int]:
    """The 12 stride values (element units, as given: not validated, not
    canonicalized) of four logical ``[B, S, H, D]`` tensor views. A tensor
    without ``.stride()`` (a raw address, a ``DeviceMem``) is taken as packed."""
    packed = packed_dense_strides(spec)
    out: dict[str, int] = {}
    for t, view in zip(DENSE_STRIDE_TENSORS, (q, k, v, o)):
        if not _has_strides(view):
            out.update(
                {
                    f"stride_{t}_{a}": packed[f"stride_{t}_{a}"]
                    for a in DENSE_STRIDE_AXES
                }
            )
            continue
        op = dense_operand_from_view(view)
        if len(op.strides) != 4:
            raise ValueError(
                f"{t}: a [B, S, H, D] view has 4 strides, got {len(op.strides)}"
            )
        for i, axis in enumerate(DENSE_STRIDE_AXES):
            out[f"stride_{t}_{axis}"] = op.strides[i]
    return out


# --------------------------------------------------------------------------- #
# The bytes a load may reach (one rule for the predicate and the kernels)
# --------------------------------------------------------------------------- #


def dense_slab_bytes(rows, token_stride, *, head_size: int, elem_bytes: int, b=None):
    """Extent of the first ``rows`` rows of one (batch, head) slab:
    ``rows > 0 ? elem_bytes * ((rows - 1) * token_stride + head_size) : 0``.

    With ``b`` None, ``rows`` and ``token_stride`` are Python ints and so is the
    result (``elem_bytes=1`` gives elements). With an IR builder ``b`` they are
    ``i32`` values and the result is the ``i32`` ``num_records`` of a K/V slab
    buffer resource, with ``rows`` the valid rows (at most ``seqlen_kv``).
    :func:`dense_strides_admit` holds every offset a load forms, up to the
    furthest element of :func:`dense_kv_reach_rows` (``seqlen_kv`` plus the slack),
    below 2^31 bytes, so that record count is below 2^31 too.
    """
    if b is None:
        rows, token_stride = int(rows), int(token_stride)
        if rows <= 0:
            return 0
        return elem_bytes * ((rows - 1) * token_stride + head_size)
    last = b.add(
        b.mul(b.sub(rows, b.const_i32(1)), token_stride), b.const_i32(head_size)
    )
    return b.select(
        b.cmp_gt(rows, b.const_i32(0)),
        b.mul(last, b.const_i32(elem_bytes)),
        b.const_i32(0),
    )


def dense_kv_elem_bytes(spec) -> int:
    """Bytes per stored K/V element of ``spec``: its ``kv_elem_bytes`` where it has
    one (1 on the gfx950 fp8 K/V-cache spec), else :data:`DENSE_ELEM_BYTES`. Q and
    O are always :data:`DENSE_ELEM_BYTES`."""
    return int(getattr(spec, "kv_elem_bytes", DENSE_ELEM_BYTES))


def dense_kv_reach_rows(seqlen_kv: int, block_n: int) -> int:
    """Rows of a K/V slab a load may address: the padded length plus the slack of
    :data:`DENSE_SLAB_SLACK_TILES` tiles of ``block_n`` rows."""
    padded = -(-int(seqlen_kv) // block_n) * block_n
    return padded + DENSE_SLAB_SLACK_TILES * block_n


def dense_slab_fits(spec, tensor: str, token_stride: int) -> bool:
    """Whether every in-slab offset of ``tensor`` fits the kernel's 32-bit index.

    K and V (buffer resources, byte offsets): the furthest element a load may
    address, row ``dense_kv_reach_rows - 1``, starts below 2^31 bytes. Q and O
    (64-bit slab base, then a 32-bit element index): ``(S - 1) * sS + D < 2^31``
    elements."""
    D = spec.head_size
    if tensor in ("k", "v"):
        rows = dense_kv_reach_rows(spec.seqlen_kv, spec.block_n)
        elem = dense_kv_elem_bytes(spec)
        reach = dense_slab_bytes(rows, token_stride, head_size=D, elem_bytes=elem)
        return reach - elem < INT32_LIMIT
    return (
        dense_slab_bytes(spec.seqlen_q, token_stride, head_size=D, elem_bytes=1)
        < INT32_LIMIT
    )


# --------------------------------------------------------------------------- #
# Admission
# --------------------------------------------------------------------------- #


class DenseStrideVerdict(NamedTuple):
    """``ok`` with the canonical 12 strides to launch with, or a decline: one of
    :data:`DENSE_STRIDE_REASONS` plus a sentence naming the operand."""

    ok: bool
    reason: str = ""
    detail: str = ""
    strides: Mapping[str, int] | None = None


def _decline(reason: str, detail: str) -> DenseStrideVerdict:
    return DenseStrideVerdict(False, reason, detail)


def _overlap_free(dims, strides) -> bool:
    """No two index tuples of the view share an element: walking the axes of
    extent > 1 by increasing stride, each stride clears the span of the axes
    before it."""
    axes = sorted((s, n) for n, s in zip(dims, strides) if n > 1)
    span = 1
    for stride, extent in axes:
        if stride < span:
            return False
        span += (extent - 1) * stride
    return True


def dense_strides_admit(
    spec, operands: Mapping[str, DenseOperand]
) -> DenseStrideVerdict:
    """Admit four dense operands (keys ``"q"``, ``"k"``, ``"v"``, ``"o"``) for
    ``spec``, or decline with the first failing check.

    ``spec`` supplies ``batch``, ``seqlen_q``, ``seqlen_kv``, ``num_query_heads``,
    ``num_kv_heads``, ``head_size`` and ``block_n``. The checks run in the order
    of :data:`DENSE_STRIDE_REASONS`; inside a check the operands go q, k, v, o.

    * ``operand``: rank 4, dims equal to the spec's ``(B, H, S, D)``, not
      virtual, no ragged offset;
    * ``head_dim_stride``: the head-dim stride is exactly 1;
    * ``stride_alignment``: on every axis of extent > 1 the stride is positive
      and a multiple of :data:`DENSE_STRIDE_MULTIPLE` elements, and the token
      stride is at least D (zero and negative strides are declined);
    * ``overlap``: no operand overlaps itself;
    * ``pointer_alignment``: the base is a multiple of
      :data:`DENSE_POINTER_ALIGN` bytes;
    * ``size_limit``: every canonical stride is below 2^31, the slab offsets fit
      (:func:`dense_slab_fits`), and the byte offset of the last slab's base plus
      its ``S`` rows, ``2 * ((B - 1) * sB + (H - 1) * sH) + 2 * ((S - 1) * sS + D)``,
      is below 2^63.

    Extent-1 axes are exempt and canonicalized (:func:`canonical_dense_strides`).
    The whole-tensor 2^31 preflight of the spec stays a separate check.
    """
    names = DENSE_STRIDE_TENSORS
    want = {t: dense_operand_dims(spec, t) for t in names}
    D = spec.head_size

    for t in names:
        op = operands[t]
        if len(op.dims) != 4 or len(op.strides) != 4:
            return _decline("operand", f"{t} has rank {len(op.dims)}, not 4")
        if op.virtual:
            return _decline("operand", f"{t} is a virtual tensor")
        if op.ragged:
            return _decline("operand", f"{t} carries a ragged offset")
        if any(n <= 0 for n in op.dims):
            return _decline("operand", f"{t} dims {tuple(op.dims)} are not positive")
        if tuple(op.dims) != want[t]:
            return _decline(
                "operand",
                f"{t} dims (B, H, S, D) {tuple(op.dims)} != spec {want[t]}",
            )
    for t in names:
        if operands[t].strides[3] != 1:
            return _decline(
                "head_dim_stride",
                f"{t} head-dim stride is {operands[t].strides[3]}, not 1",
            )
    for t in names:
        op = operands[t]
        for i, axis in enumerate(DENSE_STRIDE_AXES):
            n, s = op.dims[i], op.strides[i]
            if n == 1:
                continue
            if s <= 0:
                return _decline(
                    "stride_alignment",
                    f"{t} {axis} stride {s} is not positive (extent {n})",
                )
            if s % DENSE_STRIDE_MULTIPLE:
                return _decline(
                    "stride_alignment",
                    f"{t} {axis} stride {s} is not a multiple of "
                    f"{DENSE_STRIDE_MULTIPLE} elements",
                )
            if axis == "token" and s < D:
                return _decline(
                    "stride_alignment",
                    f"{t} token stride {s} is below the head size {D}",
                )
    for t in names:
        op = operands[t]
        if not _overlap_free(op.dims, op.strides):
            return _decline(
                "overlap",
                f"{t} dims {tuple(op.dims)} with strides {tuple(op.strides)} "
                "overlap themselves",
            )
    for t in names:
        if operands[t].align % DENSE_POINTER_ALIGN:
            return _decline(
                "pointer_alignment",
                f"{t} base is {operands[t].align}-byte aligned, needs "
                f"{DENSE_POINTER_ALIGN}",
            )
    raw = {
        f"stride_{t}_{axis}": operands[t].strides[i]
        for t in names
        for i, axis in enumerate(DENSE_STRIDE_AXES)
    }
    canon = canonical_dense_strides(spec, raw)
    for t in names:
        sB, sH, sS = (canon[f"stride_{t}_{a}"] for a in DENSE_STRIDE_AXES)
        for axis, s in zip(DENSE_STRIDE_AXES, (sB, sH, sS)):
            if s >= INT32_LIMIT:
                return _decline(
                    "size_limit", f"{t} {axis} stride {s} is not below 2^31"
                )
        if not dense_slab_fits(spec, t, sS):
            return _decline(
                "size_limit",
                f"{t} slab with token stride {sS} passes the 32-bit offset limit",
            )
        B, H = want[t][0], want[t][1]
        elem = dense_kv_elem_bytes(spec) if t in ("k", "v") else DENSE_ELEM_BYTES
        base = elem * ((B - 1) * sB + (H - 1) * sH)
        slab = dense_slab_bytes(want[t][2], sS, head_size=D, elem_bytes=elem)
        if base + slab >= _INT64_LIMIT:
            return _decline("size_limit", f"{t} slab base offset passes 64 bits")
    return DenseStrideVerdict(True, strides=canon)


def admit_dense_launch_strides(
    spec, *, q, k, v, o, serves_strides: bool
) -> dict[str, int]:
    """The 12 stride values a runner launches ``spec`` with, or ``ValueError``
    before anything is launched.

    Varlen (packed ``[total_tokens, H, D]``) and paged (a K/V page cache) bodies
    read packed storage only: every tensor that has strides must be contiguous,
    and the kernel gets :func:`packed_launch_strides` (it does not read them).
    Every other
    body: the four ``[B, S, H, D]`` views go through :func:`dense_strides_admit`
    (a decline raises with its reason); a body that does not serve strides
    (``serves_strides`` False) also requires the canonical strides to equal the
    packed ones (``not packed``), and with ``kv_segments > 1`` O must be packed
    (the combine kernel writes packed O). A tensor without ``.stride()`` is taken
    as packed."""
    views = dict(zip(DENSE_STRIDE_TENSORS, (q, k, v, o)))
    if getattr(spec, "varlen", False) or getattr(spec, "paged", False):
        for t, view in views.items():
            contiguous = getattr(view, "is_contiguous", None)
            if callable(contiguous) and not contiguous():
                raise ValueError(
                    f"dense attention: {t} is not packed: a varlen or paged body "
                    "reads contiguous storage only"
                )
        return packed_launch_strides(spec)
    operands = {
        t: (
            dense_operand_from_view(view)
            if _has_strides(view)
            else packed_dense_operand(spec, t)
        )
        for t, view in views.items()
    }
    verdict = dense_strides_admit(spec, operands)
    if not verdict.ok:
        raise ValueError(
            f"dense attention strides declined ({verdict.reason}): {verdict.detail}"
        )
    packed = packed_launch_strides(spec)
    for t in DENSE_STRIDE_TENSORS:
        keys = [f"stride_{t}_{a}" for a in DENSE_STRIDE_AXES]
        if all(verdict.strides[key] == packed[key] for key in keys):
            continue
        if not serves_strides:
            got = tuple(verdict.strides[key] for key in keys)
            want = tuple(packed[key] for key in keys)
            raise ValueError(
                f"dense attention: {t} is not packed: (batch, head, token) strides "
                f"{got} != packed {want}; this kernel reads packed [B, S, H, D] only"
            )
        if t == "o" and getattr(spec, "kv_segments", 1) > 1:
            raise ValueError(
                "dense attention: o is not packed: kv_segments > 1 writes O through "
                "the combine kernel, which writes packed [B, S, H, D]"
            )
    return dict(verdict.strides)


__all__ = [
    "DENSE_ELEM_BYTES",
    "DENSE_POINTER_ALIGN",
    "DENSE_SLAB_SLACK_TILES",
    "DENSE_STRIDE_AXES",
    "DENSE_STRIDE_MULTIPLE",
    "DENSE_STRIDE_NAMES",
    "DENSE_STRIDE_REASONS",
    "DENSE_STRIDE_TENSORS",
    "DenseOperand",
    "DenseStrideVerdict",
    "admit_dense_launch_strides",
    "append_dense_stride_signature",
    "canonical_dense_strides",
    "declare_dense_stride_params",
    "dense_kv_elem_bytes",
    "dense_kv_reach_rows",
    "dense_operand_dims",
    "dense_operand_from_view",
    "dense_slab_bytes",
    "dense_slab_fits",
    "dense_strides_admit",
    "packed_dense_operand",
    "packed_dense_strides",
    "packed_launch_strides",
    "view_dense_strides",
]
