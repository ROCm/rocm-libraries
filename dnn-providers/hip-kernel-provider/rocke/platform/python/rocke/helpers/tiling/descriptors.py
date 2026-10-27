# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The memory-layout value objects: an unpositioned :class:`TensorDesc` and the positioned
:class:`TensorWindow` cut from it.

Every mature library carries a lengths+strides+dtype descriptor SEPARATELY from the thread
mapping, then composes them: cuBLASLt `cublasLtMatrixLayout_t`, cuTENSOR
`cutensorTensorDescriptor`, CUTLASS/CuTe `Layout=Shape:Stride`, CK `TensorDescriptor` +
`tile_distribution`. Unlike the C-API handles, ours is a TRANSPARENT value object -- the point
of this layer is that you can read it.

This module is the MEMORY side of the surface -- it is pure data (no IRBuilder, no dtype
casting). It says WHERE a tensor sits and WHERE a tile is placed on it; the IR verbs in
:mod:`rocke.helpers.tiling.emit` turn a window into actual loads/stores.

- :class:`TensorDesc` -- the pure, ptr-free memory layout (lengths + strides + dtype). A length or a
  stride is a compile-time int or an i32 ``Value`` whose facts are stated: a kernel parameter
  declared with them (``b.param("M", I32, ge=1, le=4096)``, checked on the host before each launch
  through ``KernelLauncher.from_artifact``) or a Value ``b.gpu_checked_value`` returned (checked on
  the GPU). The facts are read with
  ``rocke.core.facts.stated_facts_of``; nothing here restates them.
- :class:`TensorWindow` -- a `TensorDesc` positioned at an `origin` (no length of its own: the
  valid region is the tensor's): the 'where' handed to load/store.

One axis convention, no axis labels. A tensor is written batch axes first and its reduction axis
last, with the memory layout only in the strides. A rank-``r`` tile covers the tensor's LAST ``r``
axes; each leading axis is positioned by the window origin alone (an axis of tile extent 1). A
batched ``(G, M, K)`` tensor read with a rank-2 ``(M, K)`` tile is windowed at ``(g, m0, k0)``. Another
axis order goes through :meth:`TensorDesc.permute` (a strides view); nothing reads what an axis means.

GEMM tensors are written in one axis order -- A (M, K), B (N, K), C (M, N), batch axes first --
the order their tile descriptors use, with the memory layout only in the strides: a B stored
N-contiguous (stride 1 on N) is ``make_tensor_desc((N, K), (1, ldb), dtype)``, a K-contiguous B is
``make_tensor_desc((N, K), (ldb, 1), dtype)``. :func:`check_gemm_shapes` checks at build time that
the A, B and C of one GEMM agree on their extents in that reading.
"""

from __future__ import annotations

import numbers
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from rocke.core.facts import has_stated_facts, stated_facts_of
from rocke.core.ir import GPU_CHECKED_VALUE_WRITER, I32, Value, marker_facts
from rocke.core.param_facts import LOWER_WORDS
from rocke.core.value_sources import ASSUME_RANGE_OP, value_text

from .tensor_transforms.walk import wave_uniform

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = [
    "TensorDesc",
    "make_tensor_desc",
    "TensorWindow",
    "make_window",
    "check_gemm_shapes",
]


@dataclass(frozen=True)
class TensorDesc:
    """Pure, ptr-free memory-layout descriptor: per-axis `lengths` + `strides` + element `dtype`.

    The `lengths` are the valid extent (a window has no lengths of its own) and `strides` are
    the physical layout (so a stride-1 axis is the memory-contiguous one); the ptr binds later, at
    load/store. The same holds in global memory and in LDS: an element's index is
    ``sum(position * stride)`` in both, and an LDS buffer is a flat allocation sized by
    :attr:`element_span`. Axes are positional: batch axes first, the reduction axis last.

    A length or a stride is a compile-time int or an i32 ``Value`` with stated facts, used as it
    is: the same Value as a length and a stride (``make_tensor_desc((M, K), (K, 1), F16)``) says
    the rows are packed. A runtime length is the bound every check of a window over this tensor
    compares against, by the identity of that Value. The contiguous axis keeps the compile-time
    stride 1.

    ``begin`` is internal: a slice's start in its parent's coordinates, one int or per-wave i32
    value per axis, all 0 (the default) for a tensor that is not a slice.
    """

    lengths: tuple[int | Value, ...]
    strides: tuple[int | Value, ...]
    dtype: Any
    begin: tuple[int | Value, ...] | None = None

    def __post_init__(self) -> None:
        if len(self.lengths) != len(self.strides):
            raise ValueError(
                f"lengths rank {len(self.lengths)} != strides rank {len(self.strides)}"
            )
        for axis, length in enumerate(self.lengths):
            _check_length(axis, length)
        for axis, stride in enumerate(self.strides):
            _check_stride(axis, stride)
        _check_contiguous_axis(self.strides)
        if self.begin is None:
            object.__setattr__(self, "begin", (0,) * len(self.lengths))
        elif len(self.begin) != len(self.lengths):
            raise ValueError(
                f"TensorDesc begin has {len(self.begin)} entries but the tensor has rank "
                f"{len(self.lengths)} -- one begin per axis"
            )

    @property
    def rank(self) -> int:
        return len(self.lengths)

    def _runtime_extent(self) -> tuple[str, int, Value] | None:
        """The first runtime length or stride, as ``("lengths" or "strides", axis, Value)``, or
        ``None`` for a description whose lengths and strides are all ints."""
        for name, extents in (("lengths", self.lengths), ("strides", self.strides)):
            for axis, extent in enumerate(extents):
                if isinstance(extent, Value):
                    return name, axis, extent
        return None

    def _require_static(self, reader: str) -> None:
        """Refuse a runtime length or stride for a reader that needs the whole layout at build
        time (an LDS buffer is sized at build)."""
        runtime = self._runtime_extent()
        if runtime is not None:
            name, axis, extent = runtime
            raise ValueError(
                f"{reader} is static -- {name}[{axis}] is a runtime value "
                f"({value_text(extent)}); LDS tensors are sized at build"
            )

    @property
    def element_span(self) -> int:
        """Number of elements from the first to the last element this description can address:
        ``1 + sum((length - 1) * stride)`` over every axis. It is the size an LDS buffer needs --
        ``b.smem_alloc(td.dtype, [td.element_span])`` -- so the descriptor stays the one home of the
        buffer's layout. Padding (a larger outer stride) is counted up to the last element; the dead
        space after it is not. Static: a runtime length or stride is refused. The LDS verbs' built-in
        swizzle reaches further; size a swizzled buffer with ``lds_element_span``.
        """
        self._require_static("TensorDesc.element_span")
        for axis, stride in enumerate(self.strides):
            if stride < 0:
                raise ValueError(
                    f"TensorDesc.element_span needs non-negative strides -- strides[{axis}]={stride}"
                )
        return 1 + sum(
            (length - 1) * stride for length, stride in zip(self.lengths, self.strides)
        )

    def permute(self, order: tuple[int, ...]) -> "TensorDesc":
        """A permuted VIEW (new axis ``i`` = old axis ``order[i]``): the same memory, presented in
        another axis order. A tensor needs no permute to match its memory layout -- the layout lives
        in the strides, so a GEMM's B stored N-contiguous is described ``(N, K)`` with strides
        ``(1, ldb)`` directly. Permute when a consumer wants the same bytes in another axis order: on
        that B, ``permute([1, 0])`` gives its ``(K, N)`` view, strides ``(ldb, 1)``."""
        if sorted(order) != list(range(self.rank)):
            raise ValueError(
                f"permute order {order!r} must be a permutation of 0..{self.rank - 1}"
            )
        return TensorDesc(
            tuple(self.lengths[i] for i in order),
            tuple(self.strides[i] for i in order),
            self.dtype,
            begin=tuple(self.begin[i] for i in order),
        )


def make_tensor_desc(
    lengths: tuple[int | Value, ...],
    strides: tuple[int | Value, ...],
    dtype: Any,
) -> TensorDesc:
    """Free factory: a pure, ptr-free `TensorDesc` (lengths + strides + dtype), axes written batch
    first and the reduction axis last, the memory layout in the strides. Each length and stride is
    a compile-time int or an i32 Value with stated facts (a kernel parameter declared with them, or
    a Value ``b.gpu_checked_value`` returned). A runtime length needs a stated lower bound of at
    least 0 and a stated upper bound; a runtime stride a stated lower bound of at least 1.
    """
    return TensorDesc(tuple(lengths), tuple(strides), dtype)


def _check_length(axis: int, length: Any) -> None:
    """A length is a positive compile-time int, or a runtime Value stated ``>= 0`` with an upper
    bound."""
    if isinstance(length, Value):
        facts = _check_runtime_extent("lengths", axis, length)
        if facts.lower is None or facts.lower < 0:
            raise ValueError(
                f"TensorDesc lengths[{axis}] is {value_text(length)}, which could be negative "
                f"({_lower_text(length, facts.lower)}).\nFix: {_fix_text(length, 'ge=0')}"
            )
        if facts.upper is None:
            raise ValueError(
                f"TensorDesc lengths[{axis}] is {value_text(length)}, which has no upper bound: "
                "state le= or lt=, the tightest bound it can reach"
            )
        return
    if not (_is_compile_time_integer(length) and length > 0):
        raise ValueError(
            f"TensorDesc lengths[{axis}] = {length!r} -- a length is a positive int or an i32 "
            "Value with stated facts"
        )


def _check_stride(axis: int, stride: Any) -> None:
    """A stride is a compile-time int, or a runtime Value stated ``>= 1``."""
    if isinstance(stride, Value):
        facts = _check_runtime_extent("strides", axis, stride)
        if facts.lower is None or facts.lower < 1:
            raise ValueError(
                f"TensorDesc strides[{axis}] is {value_text(stride)}, which could be below 1 "
                f"({_lower_text(stride, facts.lower)}).\nFix: {_fix_text(stride, 'ge=1')}, or "
                "ge= a length it is at least"
            )
        return
    if not _is_compile_time_integer(stride):
        raise ValueError(
            f"TensorDesc strides[{axis}] = {stride!r} -- a stride is an int or an i32 Value with "
            "stated facts"
        )


def _check_runtime_extent(name: str, axis: int, value: Value):
    """The stated facts of a runtime length or stride, after the rules both share: an i32 Value,
    with stated facts, the same for the whole wave."""
    where = f"TensorDesc {name}[{axis}] is {value_text(value)}"
    if value.type != I32:
        raise TypeError(
            f"{where}, a {value.type.name} Value -- a runtime length or stride is an i32 Value"
        )
    if not has_stated_facts(value):
        raise ValueError(
            f"{where}, which has no stated facts. State them on its b.param declaration, or "
            "check it with b.gpu_checked_value; if you checked it with gpu_checked_value, pass "
            "the Value gpu_checked_value returned"
        )
    if not wave_uniform(value):
        raise ValueError(
            f"{where}, which can differ between the lanes of a wave (it depends on a thread id "
            "or a per-lane load) -- a length or a stride is the same for the whole wave; pass a "
            "value loaded from memory through b.readfirstlane before checking it"
        )
    return stated_facts_of(value)


def _lower_text(value: Value, lower: int | None) -> str:
    """Why a runtime extent's stated lower bound is too low, in the words of its statement."""
    stated = "declared" if value.param is not None else "checked"
    if lower is None and not _states_lower_bound(value):
        return f"{stated} with no lower bound"
    if lower is None:
        return f"{stated} with a lower bound that does not resolve to a number"
    return f"{stated} with the lower bound {lower}"


def _states_lower_bound(value: Value) -> bool:
    """Whether the statement of ``value``'s facts (a declaration or a range marker) states a lower
    bound, resolved to a number or not."""
    if value.param is not None:
        return any(word in value.param.attrs for word in ("eq", *LOWER_WORDS))
    return marker_facts(value.op).lower is not None


def _fix_text(value: Value, word: str) -> str:
    """The statement that fixes a missing lower bound, where the value's facts are stated."""
    if value.param is not None:
        return f'b.param("{value.param.name}", I32, {word}, ...)'
    return f"b.gpu_checked_value(..., {word}, ...)"


def _check_contiguous_axis(strides: tuple[Any, ...]) -> None:
    """A tensor with a runtime stride keeps its contiguous axis at the compile-time stride 1.

    A runtime stride on the contiguous axis would make every access a gather. A tensor with no
    int stride of 1 and a runtime stride has it on the axis that must be contiguous; the last such
    axis is named (the convention puts the reduction axis, usually the contiguous one, last).
    """
    runtime_axes = [a for a, s in enumerate(strides) if isinstance(s, Value)]
    if not runtime_axes:
        return
    if any(_is_compile_time_integer(s) and s == 1 for s in strides):
        return
    raise ValueError(
        f"TensorDesc strides[{runtime_axes[-1]}] is a runtime value on the stride-1 axis -- the "
        "contiguous axis must have the compile-time stride 1"
    )


# The kinds of value a window origin may be, named in the origin type error.
_ORIGIN_KINDS = "an int or an i32 IR value"


def _origin_type_text(origin: Any) -> str:
    """How an origin of the wrong kind is named in its error: its IR type for an IR value."""
    if isinstance(origin, Value):
        return f"Value of type {origin.type.name}"
    return type(origin).__name__


def _checked_origin(where: str, origin: Any) -> int | Value:
    """One axis's origin as the window keeps it: an int, or an i32 IR value. ``where`` names the
    argument in the type error."""
    if _is_compile_time_integer(origin):
        return int(origin)
    if isinstance(origin, Value) and origin.type == I32:
        return origin
    raise TypeError(
        f"{where} is a {_origin_type_text(origin)} -- an origin is {_ORIGIN_KINDS}"
    )


@dataclass(frozen=True)
class TensorWindow:
    """A :class:`TensorDesc` positioned at an `origin` -- the 'where' handed to load / store.

    A window has one job: place a tile on a tensor. It holds no length (the valid region is the
    tensor's ``lengths``, the one home of a length), no pointer (the verbs take it) and no memory
    choice. Build it with :func:`make_window`; move it along one axis with :meth:`with_origin`.

    The window has one origin per tensor axis: an int or an i32 IR value. A rank-``r`` tile covers its last ``r`` axes (tile axis ``i`` is
    tensor axis ``rank - r + i``); each leading axis sits at its origin, so its address term is
    ``origin * stride``. An origin may be negative: a convolution's halo is a window placed
    before the tensor start, whose elements outside the tensor read 0 and are not written.
    """

    tensor: TensorDesc
    origin: tuple[int | Value, ...]

    def __post_init__(self) -> None:
        rank = self.tensor.rank
        if len(self.origin) != rank:
            raise ValueError(
                f"make_window got {len(self.origin)} origins for a rank-{rank} tensor -- give one "
                "origin per tensor axis (a rank-t tile covers the last t axes; leading axes are "
                "positioned by their origin)"
            )
        origin = tuple(
            _checked_origin(f"make_window origin[{axis}]", value)
            for axis, value in enumerate(self.origin)
        )
        object.__setattr__(self, "origin", origin)

    def with_origin(self, axis: int, origin: int | Value) -> TensorWindow:
        """The same window with tensor axis ``axis``'s origin replaced by ``origin``; every other
        axis keeps its origin. Moves a window along one axis, e.g. the K step of a loop::

            a_k = a_window.with_origin(1, k)      # A (M, K): the same M origin, K at k
        """
        rank = self.tensor.rank
        if not (_is_compile_time_integer(axis) and 0 <= axis < rank):
            raise ValueError(
                f"with_origin: axis {axis} does not exist on a rank-{rank} window"
            )
        moved = _checked_origin(f"with_origin origin for axis {axis}", origin)
        return TensorWindow(
            self.tensor, self.origin[:axis] + (moved,) + self.origin[axis + 1 :]
        )


def make_window(tensor: TensorDesc, origin: Sequence[Any]) -> TensorWindow:
    """Free factory (ck_tile's ``make_tile_window`` shape): `tensor` positioned at `origin`, one
    origin per tensor axis -- an int or an i32 IR value, negative allowed. A rank-t tile covers the last t axes; each leading axis is positioned by its origin.

    The valid region is the tensor's own lengths: where an element's position ``origin + coord``
    falls outside ``[0, length)`` on an axis, load reads 0 and store drops it. To narrow the
    region, describe a narrower tensor.
    """
    return TensorWindow(tensor, tuple(origin))


# The reading every check_gemm_shapes extent message ends with: how GEMM tensors are read, and how to
# describe the operand most often written the other way round.
_GEMM_READING = (
    "GEMM tensors are read as A (M, K), B (N, K), C (M, N), memory order in the strides; "
    "a B stored N-contiguous (stride 1 on N) is make_tensor_desc((N, K), (1, ldb), dtype)"
)


def _is_compile_time_integer(value: Any) -> bool:
    """Whether an extent is a compile-time integer: a Python or numpy integer, not a bool."""
    return isinstance(value, numbers.Integral) and not isinstance(value, bool)


def _same_extent(x: Any, y: Any) -> bool:
    """Whether two axis extents are the same.

    Two compile-time integers agree by value, so ``64`` and ``numpy.int64(64)`` agree. Any other
    extent (a runtime value) agrees only with itself: equal-looking runtime values may differ when
    the kernel runs, so only one and the same object is known to match.
    """
    if x is y:
        return True
    return _is_compile_time_integer(x) and _is_compile_time_integer(y) and x == y


def _extent_text(extent: Any) -> str:
    """An extent as a message names it: an int, or a runtime Value by its parameter's name."""
    return value_text(extent) if isinstance(extent, Value) else str(extent)


def _checked_twice_text(x: Any, y: Any) -> str | None:
    """The advice for two extents that differ only because one value was checked twice: two
    Values that two ``gpu_checked_value`` calls returned for the same IR value. ``None`` for any
    other pair, whose mismatch is a real one."""
    raws = []
    for extent in (x, y):
        op = extent.op if isinstance(extent, Value) else None
        if (
            op is None
            or op.name != ASSUME_RANGE_OP
            or op.attrs.get("by") != GPU_CHECKED_VALUE_WRITER
        ):
            return None
        raws.append(op.operands[0])
    if raws[0] is not raws[1]:
        return None
    return (
        f"both check {value_text(raws[0])} with their own gpu_checked_value call; check it once "
        "and pass the one Value it returned to every tensor"
    )


def check_gemm_shapes(
    a: TensorDesc, b: TensorDesc, c: TensorDesc | None = None
) -> None:
    """Check at build time that the tensors of one GEMM agree on their extents. Emits no IR.

    Reads its tensors positionally as A ``(batch..., M, K)``, B ``(batch..., N, K)`` and, when
    given, C ``(batch..., M, N)``; the memory order lives in the strides and is not read. A and B
    have equal ranks of at least 2, the same extent on every leading batch axis, and A's K (last
    axis) is B's K (last axis). With ``c``: C has the same rank and batch extents, A's M (axis -2)
    is C's M (axis -2), and B's N (axis -2) is C's N (last axis). Two extents agree when they are
    equal integers (Python or numpy), or one and the same extent object. ``c=None`` checks the A / B pair alone -- the ``Q * K^T`` of a fused chain whose C
    is never a tensor, or a GEMM that reduces over a batch axis.

    Its domain: GEMMs whose A, B and C are tensors in that shape family, their batch axes shared one
    to one. Do not call it on tensors outside that family -- operands reached through a transform
    graph (an implicit-GEMM convolution, whose K is ``Y * X * C``), or the C of a GEMM that sums over
    a batch axis (pass only its A and B). Nothing reads what an axis means, so it cannot tell such a
    tensor from a mistake.

    It catches a transposed or mis-sized operand whenever the swapped extents differ, and every
    batch-extent mismatch. A square transposed operand (M = N = K) passes; bit-exact tests stay the
    backstop. Call it on the logical tensors: on the ``(M, N)`` C the verbs see, and for a kernel
    that crosses A and B, on the C before the crossed view.
    """
    named = [("A", a, "(batch..., M, K)"), ("B", b, "(batch..., N, K)")]
    if c is not None:
        named.append(("C", c, "(batch..., M, N)"))
    for name, tensor, shape in named:
        if tensor.rank < 2:
            raise ValueError(
                f"check_gemm_shapes: {name} has rank {tensor.rank} -- a GEMM operand has rank >= 2: "
                f"{shape}"
            )
    ranks = [tensor.rank for _, tensor, _ in named]
    if len(set(ranks)) != 1:
        names = "A, B and C" if c is not None else "A and B"
        raise ValueError(
            f"check_gemm_shapes: {names} have ranks {', '.join(str(r) for r in ranks)} -- batched "
            "GEMM tensors carry the same leading batch axes: (G, M, K), (G, N, K), (G, M, N)"
        )
    rank = a.rank
    across = "A, B and C" if c is not None else "A and B"
    for axis in range(rank - 2):
        for name, tensor, _ in named[1:]:
            first_extent, second_extent = a.lengths[axis], tensor.lengths[axis]
            if not _same_extent(first_extent, second_extent):
                advice = _checked_twice_text(first_extent, second_extent) or (
                    f"every batch axis has one extent across {across}"
                )
                raise ValueError(
                    f"check_gemm_shapes: batch axis {axis} is {_extent_text(first_extent)} in A "
                    f"but {_extent_text(second_extent)} in {name} -- {advice}"
                )
    pairs = [("A's K", a, rank - 1, "B's K", b, rank - 1)]
    if c is not None:
        pairs.append(("A's M", a, rank - 2, "C's M", c, rank - 2))
        pairs.append(("B's N", b, rank - 2, "C's N", c, rank - 1))
    for first, first_tensor, first_axis, second, second_tensor, second_axis in pairs:
        first_extent = first_tensor.lengths[first_axis]
        second_extent = second_tensor.lengths[second_axis]
        if not _same_extent(first_extent, second_extent):
            advice = _checked_twice_text(first_extent, second_extent) or _GEMM_READING
            raise ValueError(
                f"check_gemm_shapes: {first} (axis {first_axis}) is {_extent_text(first_extent)} "
                f"but {second} (axis {second_axis}) is {_extent_text(second_extent)} -- {advice}"
            )
