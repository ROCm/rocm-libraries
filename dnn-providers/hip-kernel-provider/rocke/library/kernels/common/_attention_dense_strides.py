# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Launch-time Q/K/V/O strides of the dense attention kernels.

Every dense attention kernel (gfx942 and gfx950, every body) takes twelve
trailing ``i32`` kernel parameters: one ``(batch, head, token)`` element stride
triple per tensor Q, K, V and O, declared after every other parameter in the
order of :data:`DENSE_STRIDE_NAMES`. The head-dim stride is 1 and is not a
parameter. A caller hands the runner logical ``[B, S, H, D]`` views
(``[B, Skv, Hkv, D]`` for K and V) of any storage, in torch's axis order. The
predicate works in ``(B, H, S, D)`` order, hipDNN's graph order and the order of
its C++ twin; :func:`dense_operand_from_view` reorders a view into it.

Which function to call:

* a kernel builder: :func:`declare_dense_stride_params` for the parameters,
  :func:`emit_dense_slab_ptr` for a pointer moved to its (batch, head) slab and
  :func:`dense_slab_bytes` with ``b`` for the ``num_records`` of a K/V slab; its
  launcher signature: :func:`append_dense_stride_signature`;
* a runner, before it launches: :func:`admit_dense_launch_strides`, which reads
  the four views, applies every check below and returns the 12 values to pass;
* a launch path without views (the runtime-args helpers, the SDPA replay
  export): :func:`packed_launch_strides`;
* a caller that describes operands without views (the table of
  ``test_attention_dense_strided_admission.py`` behind the vectors file, and the
  hipDNN IngestorGenerator test ``tests/test_ground_truth_layouts.py``, which
  checks that it admits every layout the ground-truth tool writes):
  :func:`dense_strides_admit` on :class:`DenseOperand`.

The other functions are building blocks of these.

The predicate has a C++ twin,
``src/engines/kernel_ingestor_engine/packs/AttentionDenseStrides.hpp`` in the
hip-kernel-provider. Both are pinned to one vectors file,
``platform/tests/golden/attention_dense_stride_vectors.json``, which
``library/tests/test_attention_dense_strided_admission.py`` renders from its
hand-written table. Python measures a view's alignment from its address; the
twin takes the alignment the graph declares. The design and its measurements are
in ``dense_attention_launch_time_strides_case_study.md`` under
``library/builders/gfx950/attention/prefill/``.

Related checks elsewhere look alike but stay separate, because their units and
rules differ. The strided split-KV decode path (``_attention_strided_kv.py``,
checked by ``attention_kv_cache.KvTensorLayout``) takes byte strides, 64-bit
batch and head strides and a span the host computes. The backward plan
(``attention_bwd_plan``) also admits interleaved views through an exact overlap
test, and ``_attention_bwd_addr.StridedTensorAddr`` scales element strides with a
clamp. Dense takes ``i32`` element strides, bounds the K/V reach inside the
kernel and is held to its C++ twin by the shared vectors, so a change to the
dense rules does not carry over to those families, nor theirs to dense.
``_sdpa_fwd_strides.contiguous_strides`` gives the packed values used here.

The kernels form these addresses with plain IR builder arithmetic, not the
platform ``rocke.helpers.tensor_view`` layer, because that layer lacks three
things these bodies need: vector loads through a buffer view serve f16 only
(dense also reads bf16), its global vector loads take no ``align=`` (dense loads
claim 8 and 16 bytes), and a descriptor offset is one ``i32`` element offset,
with no 64-bit (batch, head) slab rebase.

Pure host code, apart from the IR helpers that take a builder ``b``
(:func:`declare_dense_stride_params`, :func:`emit_dense_slab_ptr` and
:func:`dense_slab_bytes`): no torch import, no GPU.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, NamedTuple, Protocol

from kernels.common._sdpa_fwd_strides import contiguous_strides
from kernels.common.attention_dense_spec import INT32_LIMIT

if TYPE_CHECKING:
    from rocke.core.ir import IRBuilder, Value
    from rocke.helpers.spec import SignatureBuilder

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
# K/V tiles a load may reach past the padded length. This widens only the host's
# 2^31 check (:func:`dense_kv_reach_rows`): it is the furthest offset a load can
# FORM, not a memory bound. The K/V buffer resource of a slab is bounded at the last
# valid row (:func:`dense_slab_bytes` at ``seqlen_kv``), so a load into the padded
# tail or the slack returns 0 and reads no memory past the view.
#
# The gfx950 grid, KV-segment and persistent bodies prime a K/V double buffer with
# tiles start_tile and start_tile + 1 before their loop, and an empty KV segment
# starts at start_tile = n_upper (the padded tile count), so the furthest tile they
# load is n_upper + 1. The gfx950 sliding-window bodies stop one tile past the padded
# length; the gfx942 bodies load tile j inside the loop and stop at n_upper - 1.
# test_kv_loads_stay_inside_the_admitted_reach and
# test_the_slack_is_what_the_deepest_prefetch_needs (in
# tests/test_attention_dense_gfx950_varlen_kv_bound.py) walk the IR of every strided
# body to pin this value to the builders. A body that loads further ahead must
# raise it, and the C++ twin's DENSE_SLAB_SLACK_TILES with it (the vectors file's
# header pins the two equal).
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


class DenseProblem(Protocol):
    """The spec fields the predicate reads: an attention dense spec, or any object
    that has them (the vectors table and the hipDNN IngestorGenerator test pass a
    namespace)."""

    @property
    def batch(self) -> int: ...

    @property
    def seqlen_q(self) -> int: ...

    @property
    def seqlen_kv(self) -> int: ...

    @property
    def num_query_heads(self) -> int: ...

    @property
    def num_kv_heads(self) -> int: ...

    @property
    def head_size(self) -> int: ...

    @property
    def block_n(self) -> int: ...


# --------------------------------------------------------------------------- #
# Names and kernel ABI
# --------------------------------------------------------------------------- #


def declare_dense_stride_params(b: IRBuilder) -> dict[str, Value]:
    """Declare the 12 stride parameters on IR builder ``b`` (``i32``, in
    :data:`DENSE_STRIDE_NAMES` order) and return them by name. Call it after
    every other parameter of the kernel: the strides are the constant tail of
    the dense ABI."""
    from rocke.core.ir import I32

    return {name: b.param(name, I32) for name in DENSE_STRIDE_NAMES}


def append_dense_stride_signature(sig: SignatureBuilder) -> SignatureBuilder:
    """``sig`` with the 12 ``i32`` stride scalars appended, mirroring
    :func:`declare_dense_stride_params`."""
    for name in DENSE_STRIDE_NAMES:
        sig = sig.scalar(name, "i32")
    return sig


def emit_dense_slab_ptr(
    b: IRBuilder,
    strides: Mapping[str, Value],
    ptr: Value,
    batch: Value,
    head: Value,
    tensor: str,
) -> Value:
    """``ptr`` moved to the (``batch``, ``head``) slab of ``tensor`` (``"q"``,
    ``"k"``, ``"v"`` or ``"o"``): one 64-bit pointer add of
    ``DENSE_ELEM_BYTES * (batch * stride_<t>_batch + head * stride_<t>_head)``
    bytes. ``strides`` maps the stride parameter names to their ``i32`` values
    (:func:`declare_dense_stride_params`). The host admits only non-negative
    strides below 2^31 and holds the slab base below 2^63 bytes (``size_limit``),
    so each operand is zero-extended to ``i64`` before the multiply and the
    offset cannot wrap at 2^31. Offsets inside the slab stay 32-bit."""
    from rocke.core.ir import I64

    elems = b.add(
        b.mul(b.zext(batch, I64), b.zext(strides[f"stride_{tensor}_batch"], I64)),
        b.mul(b.zext(head, I64), b.zext(strides[f"stride_{tensor}_head"], I64)),
    )
    return b.global_ptr_add(ptr, b.mul(elems, b.const_i64(DENSE_ELEM_BYTES)))


# --------------------------------------------------------------------------- #
# Values
# --------------------------------------------------------------------------- #


def _tensor_dims(spec: DenseProblem, tensor: str) -> tuple[int, int]:
    """``(seqlen, heads)`` of one tensor of ``spec``."""
    if tensor in ("q", "o"):
        return spec.seqlen_q, spec.num_query_heads
    return spec.seqlen_kv, spec.num_kv_heads


def dense_operand_dims(spec: DenseProblem, tensor: str) -> tuple[int, int, int, int]:
    """Logical ``(B, H, S, D)`` extents of ``tensor`` (``"q"``, ``"k"``, ``"v"``
    or ``"o"``) for ``spec``: the axis order of :data:`DENSE_STRIDE_AXES` plus
    the head dim."""
    seqlen, heads = _tensor_dims(spec, tensor)
    return (spec.batch, heads, seqlen, spec.head_size)


def packed_dense_strides(spec: DenseProblem) -> dict[str, int]:
    """The 12 stride values (element units) of packed ``[B, S, H, D]`` tensors
    of ``spec``, keyed by :data:`DENSE_STRIDE_NAMES` in that order. Not
    canonical (an extent-1 axis keeps its packed stride), so a launch must not
    pass these: it passes :func:`packed_launch_strides`."""
    out: dict[str, int] = {}
    for t in DENSE_STRIDE_TENSORS:
        seqlen, heads = _tensor_dims(spec, t)
        triple = contiguous_strides(
            "bshd", heads=heads, seqlen=seqlen, head_size=spec.head_size
        )
        for axis, value in zip(DENSE_STRIDE_AXES, triple):
            out[f"stride_{t}_{axis}"] = value
    return out


def canonical_dense_strides(
    spec: DenseProblem, strides: Mapping[str, int]
) -> dict[str, int]:
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


def packed_launch_strides(spec: DenseProblem) -> dict[str, int]:
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
    elements; ``strides[3]`` is the head-dim stride): hipDNN's graph order and
    the C++ twin's. Torch views are ``[B, S, H, D]``;
    :func:`dense_operand_from_view` reorders them. ``align`` is a power of two
    the base address is a multiple of, in bytes; here it is measured from the
    address, while the C++ twin takes the alignment the graph declares.
    ``virtual`` and ``ragged`` (a ragged offset tensor) are graph properties the
    kernels cannot serve."""

    dims: tuple[int, ...]
    strides: tuple[int, ...]
    align: int = DENSE_POINTER_ALIGN
    virtual: bool = False
    ragged: bool = False


def _address_align(ptr: int) -> int:
    """The largest power of two (capped at 256) dividing address ``ptr``."""
    ptr = int(ptr)
    return 256 if ptr % 256 == 0 else ptr & -ptr


def _base_address(view: Any) -> int | None:
    """The device address ``view`` stands for, in the forms the launcher packs
    (``rocke.runtime.packing.device_pointer``): an int, ``data_ptr()`` or a
    ``DeviceMem``'s ``ptr()``. None when it has none of them."""
    if isinstance(view, int):
        return view
    for name in ("data_ptr", "ptr"):
        method = getattr(view, name, None)
        if callable(method):
            return int(method())
    return None


def _align_of(view: Any) -> int:
    """The alignment of ``view``'s address, or :data:`DENSE_POINTER_ALIGN`
    when it has no address to measure."""
    addr = _base_address(view)
    return DENSE_POINTER_ALIGN if addr is None else _address_align(addr)


def dense_operand_from_view(view: Any) -> DenseOperand:
    """:class:`DenseOperand` of a logical ``[B, S, H, D]`` tensor view (anything
    with ``.shape`` and ``.stride()``). Its address (``data_ptr()`` or ``ptr()``)
    gives the alignment; without one, :data:`DENSE_POINTER_ALIGN` is assumed. A
    view of another rank keeps its own dims, which the predicate then declines."""
    shape = tuple(int(n) for n in view.shape)
    st = tuple(int(s) for s in view.stride())
    align = _align_of(view)
    if len(shape) != 4 or len(st) != 4:
        return DenseOperand(dims=shape, strides=st, align=align)
    return DenseOperand(
        dims=(shape[0], shape[2], shape[1], shape[3]),
        strides=(st[0], st[2], st[1], st[3]),
        align=align,
    )


def packed_dense_operand(
    spec: DenseProblem, tensor: str, *, align: int = DENSE_POINTER_ALIGN
) -> DenseOperand:
    """:class:`DenseOperand` of a packed ``[B, S, H, D]`` tensor of ``spec``
    whose base address is a multiple of ``align`` bytes."""
    packed = packed_dense_strides(spec)
    return DenseOperand(
        dims=dense_operand_dims(spec, tensor),
        strides=tuple(packed[f"stride_{tensor}_{a}"] for a in DENSE_STRIDE_AXES) + (1,),
        align=align,
    )


def _has_strides(view: Any) -> bool:
    """Whether ``view`` describes its layout (``shape`` and ``stride()``)."""
    return callable(getattr(view, "stride", None)) and hasattr(view, "shape")


def _is_row_major(shape: Sequence[int], strides: Sequence[int]) -> bool:
    """Whether ``strides`` are the packed row-major strides of ``shape``. An
    extent-1 axis is never stepped, so its stride is ignored, and an empty view
    is packed: the rule of torch's ``is_contiguous()``."""
    if len(shape) != len(strides):
        return False
    if 0 in shape:
        return True
    want = 1
    for n, s in zip(reversed(shape), reversed(strides)):
        if n != 1:
            if s != want:
                return False
            want *= n
    return True


def _view_is_packed(view: Any) -> bool:
    """Whether ``view`` is packed storage: judged by its strides when it has
    ``shape`` and ``stride()`` (``is_contiguous()`` is not needed), else by its
    ``is_contiguous()``; a view with neither (a raw address) is taken as packed."""
    if _has_strides(view):
        return _is_row_major(
            tuple(int(n) for n in view.shape), tuple(int(s) for s in view.stride())
        )
    contiguous = getattr(view, "is_contiguous", None)
    return not callable(contiguous) or bool(contiguous())


# --------------------------------------------------------------------------- #
# Slab extents: one formula, two row counts (the kernels' bound, the host's reach)
# --------------------------------------------------------------------------- #


def dense_slab_bytes(
    rows: int | Value,
    token_stride: int | Value,
    *,
    head_size: int,
    elem_bytes: int,
    b: IRBuilder | None = None,
) -> int | Value:
    """Extent of the first ``rows`` rows of one (batch, head) slab:
    ``rows > 0 ? elem_bytes * ((rows - 1) * token_stride + head_size) : 0``.

    With ``b`` None, ``rows`` and ``token_stride`` are Python ints and so is the
    result (``elem_bytes=1`` gives elements). With an IR builder ``b`` they are
    ``i32`` values and the result is the ``i32`` ``num_records`` of a K/V slab
    buffer resource.

    The two callers pass different row counts, on purpose:

    * a kernel passes the valid rows, ``seqlen_kv`` itself: the slab's record count
      ends at the last element of the last valid row, never at the padding to
      ``block_n`` and never at the slack. For a K/V view that ends at its
      allocation (a packed ``[B, Skv, Hkv, D]`` tensor, say) the last
      (batch, head) slab therefore ends exactly at the allocation's end, and a load
      past it returns 0 with no fault;
    * the host admission passes :func:`dense_kv_reach_rows` (``seqlen_kv`` padded
      to ``block_n`` plus the slack) to bound the furthest offset a load can form
      below 2^31 bytes. That number is never a record count. It is at least the
      record count, so the record count is below 2^31 too.
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


def dense_kv_elem_bytes(spec: DenseProblem) -> int:
    """Bytes per stored K/V element of ``spec``: its ``kv_elem_bytes`` where it has
    one (1 on the gfx950 fp8 K/V-cache spec), else :data:`DENSE_ELEM_BYTES`. Q and
    O are always :data:`DENSE_ELEM_BYTES`."""
    return int(getattr(spec, "kv_elem_bytes", DENSE_ELEM_BYTES))


def _elem_bytes(spec: DenseProblem, tensor: str) -> int:
    """Bytes per stored element of ``tensor`` of ``spec``."""
    return dense_kv_elem_bytes(spec) if tensor in ("k", "v") else DENSE_ELEM_BYTES


def dense_kv_reach_rows(seqlen_kv: int, block_n: int) -> int:
    """Rows of a K/V slab a load may address: the padded length plus the slack of
    :data:`DENSE_SLAB_SLACK_TILES` tiles of ``block_n`` rows."""
    padded = -(-int(seqlen_kv) // block_n) * block_n
    return padded + DENSE_SLAB_SLACK_TILES * block_n


def dense_slab_fits(spec: DenseProblem, tensor: str, token_stride: int) -> bool:
    """Whether every in-slab offset of ``tensor`` fits the kernel's 32-bit index.

    K and V (buffer resources, byte offsets): the furthest element a load may
    address, row ``dense_kv_reach_rows - 1``, starts below 2^31 bytes. Q and O
    (64-bit slab base, then a 32-bit element index):
    ``(S - 1) * token_stride + D < 2^31`` elements."""
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
    """A declined verdict."""
    return DenseStrideVerdict(False, reason, detail)


def _misaligned_base(tensor: str, align: int) -> DenseStrideVerdict | None:
    """The ``pointer_alignment`` decline of ``tensor`` whose base is a multiple of
    ``align`` bytes, or None when that is a multiple of
    :data:`DENSE_POINTER_ALIGN`. The predicate and the varlen/paged launch path
    share it, so both say the same thing."""
    if align % DENSE_POINTER_ALIGN == 0:
        return None
    return _decline(
        "pointer_alignment",
        f"{tensor} base is {align}-byte aligned, needs {DENSE_POINTER_ALIGN}",
    )


def _raise_declined(verdict: DenseStrideVerdict) -> None:
    """Raise the ``ValueError`` a runner gives for a declined ``verdict``."""
    raise ValueError(
        f"dense attention strides declined ({verdict.reason}): {verdict.detail}"
    )


def _overlap_free(dims: Sequence[int], strides: Sequence[int]) -> bool:
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
    spec: DenseProblem, operands: Mapping[str, DenseOperand]
) -> DenseStrideVerdict:
    """Admit four dense operands (keys ``"q"``, ``"k"``, ``"v"``, ``"o"``) for
    ``spec``, or decline with the first failing check.

    ``spec`` supplies ``batch``, ``seqlen_q``, ``seqlen_kv``, ``num_query_heads``,
    ``num_kv_heads``, ``head_size`` and ``block_n``, and optionally
    ``kv_elem_bytes`` (:func:`dense_kv_elem_bytes`). The checks run in the order
    of :data:`DENSE_STRIDE_REASONS`; inside a check the operands go q, k, v, o.

    * ``operand``: rank 4, not virtual, no ragged offset, every extent positive,
      and dims equal to the spec's ``(B, H, S, D)``;
    * ``head_dim_stride``: the head-dim stride is exactly 1;
    * ``stride_alignment``: on every axis of extent > 1 the stride is positive
      and a multiple of :data:`DENSE_STRIDE_MULTIPLE` elements, and the token
      stride is at least D (zero and negative strides are declined);
    * ``overlap``: no operand overlaps itself;
    * ``pointer_alignment``: the base is a multiple of
      :data:`DENSE_POINTER_ALIGN` bytes;
    * ``size_limit``: every canonical stride is below 2^31, the slab offsets fit
      (:func:`dense_slab_fits`), and the byte offset of the last slab's base plus
      its ``S`` rows,
      ``2 * ((B - 1) * s_b + (H - 1) * s_h) + 2 * ((S - 1) * s_s + D)``, is below
      2^63.

    Extent-1 axes are exempt and canonicalized (:func:`canonical_dense_strides`).
    The whole-tensor 2^31 preflight of the spec stays a separate check, and so
    does the check of O against the inputs (:func:`admit_dense_launch_strides`).
    """
    names = DENSE_STRIDE_TENSORS
    want = {t: dense_operand_dims(spec, t) for t in names}
    D = spec.head_size

    for t in names:
        op = operands[t]
        if len(op.dims) != 4:
            return _decline("operand", f"{t} has rank {len(op.dims)}, not 4")
        if len(op.strides) != 4:
            return _decline(
                "operand", f"{t} has {len(op.strides)} strides for its 4 dims, not 4"
            )
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
        misaligned = _misaligned_base(t, operands[t].align)
        if misaligned is not None:
            return misaligned
    raw = {
        f"stride_{t}_{axis}": operands[t].strides[i]
        for t in names
        for i, axis in enumerate(DENSE_STRIDE_AXES)
    }
    canon = canonical_dense_strides(spec, raw)
    for t in names:
        s_b, s_h, s_s = (canon[f"stride_{t}_{a}"] for a in DENSE_STRIDE_AXES)
        for axis, s in zip(DENSE_STRIDE_AXES, (s_b, s_h, s_s)):
            if s >= INT32_LIMIT:
                return _decline(
                    "size_limit", f"{t} {axis} stride {s} is not below 2^31"
                )
        if not dense_slab_fits(spec, t, s_s):
            return _decline(
                "size_limit",
                f"{t} slab with token stride {s_s} passes the 32-bit offset limit",
            )
        B, H = want[t][0], want[t][1]
        elem = _elem_bytes(spec, t)
        base = elem * ((B - 1) * s_b + (H - 1) * s_h)
        slab = dense_slab_bytes(want[t][2], s_s, head_size=D, elem_bytes=elem)
        if base + slab >= _INT64_LIMIT:
            return _decline("size_limit", f"{t} slab base offset passes 64 bits")
    return DenseStrideVerdict(True, strides=canon)


# --------------------------------------------------------------------------- #
# Launch
# --------------------------------------------------------------------------- #


def _byte_span(
    addr: int | None, dims: Sequence[int], strides: Sequence[int], elem_bytes: int
) -> tuple[int, int] | None:
    """``[first, end)`` bytes of the elements of a view at base address ``addr``:
    ``addr`` plus ``elem_bytes * (1 + sum((n - 1) * s))``. None when the address
    is 0 or unknown or the view is empty. The launch calls it only on admitted
    views, whose axes of extent > 1 all have positive strides
    (:func:`dense_strides_admit` or the packed check refused the rest), so the
    first element is at ``addr``."""
    if not addr or any(n <= 0 for n in dims):
        return None
    last = sum((n - 1) * s for n, s in zip(dims, strides))
    return addr, addr + elem_bytes * (last + 1)


def _refuse_output_overlap(spans: Mapping[str, tuple[int, int] | None]) -> None:
    """Raise ``ValueError`` when O's byte span meets the span of Q, K or V.

    The kernels declare O ``noalias`` and write it while other workgroups still
    read the inputs, so an O that shares storage with one (say, a view into a
    fused QKV buffer) gives wrong results with no error. It is a span test: an O
    interleaved with an input without sharing an element is refused too. A
    tensor whose span is None (``_byte_span``) is not checked.

    The LSE output has a check of its own (``_lse_store._check_lse_buffer``) with
    the same span formula. The two stay apart because their inputs differ: that
    one reads ``data_ptr()`` and ``element_size()`` off torch-like tensors and
    holds LSE against every input of the launch, the bias and the sinks included.
    This one takes the address forms the launcher packs (an int, ``data_ptr()``
    or ``ptr()``), counts each tensor at the spec's element size, skips address 0
    (synthetic host launches), and holds O against Q, K and V only: the runner
    passes it no other tensor, and the C++ launch check
    (``denseOutputOverlapsInput``) has the same scope."""
    o = spans.get("o")
    if o is None:
        return
    for t in ("q", "k", "v"):
        span = spans.get(t)
        if span is not None and span[0] < o[1] and o[0] < span[1]:
            raise ValueError(
                f"dense attention: o overlaps {t}: o bytes [{o[0]:#x}, {o[1]:#x}) "
                f"meet {t} bytes [{span[0]:#x}, {span[1]:#x}); the kernel writes O "
                f"while it still reads {t}, so O needs storage of its own"
            )


def _launch_operand(spec: DenseProblem, tensor: str, view: Any) -> DenseOperand:
    """:class:`DenseOperand` of one runner argument: read off the view when it
    has ``shape`` and ``stride()``, else packed with the alignment of its
    address. A view that says it is not contiguous but has no ``stride()`` is
    refused: its layout cannot be read."""
    if _has_strides(view):
        return dense_operand_from_view(view)
    if not _view_is_packed(view):
        raise ValueError(
            f"dense attention: {tensor} is not packed and has no stride() to read "
            "its layout from"
        )
    return packed_dense_operand(spec, tensor, align=_align_of(view))


def admit_dense_launch_strides(
    spec: DenseProblem,
    *,
    q: Any,
    k: Any,
    v: Any,
    o: Any,
    serves_strides: bool,
) -> dict[str, int]:
    """The 12 stride values a runner launches ``spec`` with, or ``ValueError``
    before anything is launched.

    ``q``, ``k``, ``v`` and ``o`` are torch tensors, duck-typed views (``shape``
    and ``stride()``, with ``data_ptr()`` or ``ptr()`` for the address) or raw
    device addresses. ``serves_strides`` is ``body_serves_strides(spec)`` of the
    arch's ``attention_dense`` module. ``spec`` may also carry ``varlen``,
    ``paged`` and ``kv_segments``; each defaults to off.

    * Every body: each base address is a multiple of :data:`DENSE_POINTER_ALIGN`
      bytes (``pointer_alignment``), the alignment measured from the address
      itself, whether it is an int, ``data_ptr()`` or ``ptr()``. An object with
      none of these is taken as aligned, and so is address 0.
    * Varlen (packed ``[total_tokens, H, D]``) and paged (a K/V page cache)
      bodies read packed storage only: every view must be packed (judged by its
      strides when it has them, else by ``is_contiguous()``; a raw address is
      taken as packed), and the kernel gets :func:`packed_launch_strides`. The
      extents are not checked here: they are not the spec's ``[B, S, H, D]``.
    * Every other body: the four ``[B, S, H, D]`` views go through
      :func:`dense_strides_admit`, and a decline raises with its reason. A raw
      address, or a view without ``stride()``, is a packed operand. When
      ``serves_strides`` is False the canonical strides must equal the packed
      ones (``not packed``), and with ``kv_segments > 1`` O must be packed (the
      combine kernel writes packed O).
    * Every body: O's bytes may not meet those of Q, K or V
      (:func:`_refuse_output_overlap`). Spans come from each view's dims and
      strides; a raw address of a varlen or paged body, or an address of 0, is
      not checked.
    """
    views = dict(zip(DENSE_STRIDE_TENSORS, (q, k, v, o)))
    if getattr(spec, "varlen", False) or getattr(spec, "paged", False):
        for t, view in views.items():
            if not _view_is_packed(view):
                raise ValueError(
                    f"dense attention: {t} is not packed: a varlen or paged body "
                    "reads contiguous storage only"
                )
        # The kernels claim 16-byte aligned Q/K/V/O pointers on these bodies too.
        for t, view in views.items():
            misaligned = _misaligned_base(t, _align_of(view))
            if misaligned is not None:
                _raise_declined(misaligned)
        _refuse_output_overlap(
            {
                t: (
                    _byte_span(
                        _base_address(view),
                        tuple(int(n) for n in view.shape),
                        tuple(int(s) for s in view.stride()),
                        _elem_bytes(spec, t),
                    )
                    if _has_strides(view)
                    else None
                )
                for t, view in views.items()
            }
        )
        return packed_launch_strides(spec)
    operands = {t: _launch_operand(spec, t, view) for t, view in views.items()}
    verdict = dense_strides_admit(spec, operands)
    if not verdict.ok:
        _raise_declined(verdict)
    _refuse_output_overlap(
        {
            t: _byte_span(
                _base_address(views[t]),
                operands[t].dims,
                operands[t].strides,
                _elem_bytes(spec, t),
            )
            for t in DENSE_STRIDE_TENSORS
        }
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
    "admit_dense_launch_strides",
    "append_dense_stride_signature",
    "canonical_dense_strides",
    "declare_dense_stride_params",
    "dense_kv_reach_rows",
    "dense_operand_dims",
    "dense_operand_from_view",
    "dense_slab_bytes",
    "dense_strides_admit",
    "emit_dense_slab_ptr",
    "packed_dense_operand",
    "packed_launch_strides",
]
