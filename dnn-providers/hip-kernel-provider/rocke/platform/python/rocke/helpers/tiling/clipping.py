# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Which bound checks a window's accesses need, decided once, by the shared bound-check rule.

A global access is safe only when every element it touches lies inside its tensor:
``0 <= position < length`` on every tensor axis. Whether a side of that test can fail is a
build-time question with one answer, the tensor_transforms rule ``required_checks``: a side is
checked unless the ranges the IR proves for the window's positions show it cannot fail. Nothing
is trusted -- an origin the IR says nothing about keeps both of its sides checked, and a tile
alignment is not a proof.

``explain_bounds_checks(b, ptr, window, tile_desc)`` answers that question for one access of
``load_fragment`` / ``store_fragment`` without emitting anything, as a :class:`BoundsCheckPlan`;
the verbs take the same decision from the same function, so the plan is what they emit. Print it
for one line per tensor axis and side.

How a window's positions are known. On each tensor axis ``a`` an element's position is
``origin[a] + coordinate``; the IR range walk splits the origin into a constant, per-wave parts and
a per-lane part (a wave offset), each with its proven range. A tile axis adds the tile's
coordinate, ``[0, extent)`` over the tile (the layout's per-lane term plus a register constant); a
leading axis (one the tile does not cover) is its origin alone. These ranges, against the bounds
``[0, length)`` of the tensor's own lengths, decide every side.

What each value of the plan means, in plain words:

- :class:`SideCheck` (one per side of each tensor axis):

  - ``NOT_NEEDED`` -- proven at build time that the side cannot fail; no code;
  - ``BY_BUFFER`` -- the buffer instruction's range check enforces it (upper sides only; not used
    until accesses go through buffer resources);
  - ``SCALAR_CHECK`` -- the side's position is the same for every lane (a leading axis, a tile
    axis of extent 1): one scalar compare per access;
  - ``PER_LANE_CHECK`` -- one compare per distinct lane value.

- :class:`VectorValidity` (how a vector of ``vector_width`` elements along ``vector_axis`` is
  decided): ``NOT_CHECKED`` (nothing is checked), ``FIRST_ELEMENT`` (every check has one answer
  for the whole vector) or ``FIRST_AND_LAST_ELEMENT`` (the vector may cross a bound: its first and
  last element decide it, and the valid elements form one span).

- :class:`AccessPath` -- the memory instructions: ``GLOBAL`` (global loads and stores) or
  ``BUFFER``. Every access is a global access today: a proven window reads and writes unmasked,
  a checked one masks every access by its validity (an element outside the tensor reads 0 and is
  not written, and no instruction touches it).

- :class:`GlobalForm` -- how a global access forms its addresses (the ``addresses`` line of
  ``print(plan)``): one 32-bit element index, a placed address, or a placed address under the
  access's validity.

One code for every tile: the decision is made per window, never per tile at run time, so a kernel
built for ragged shapes runs its checked code on interior tiles too (they pass every compare). The
fast path for shapes that are tile multiples is a second kernel, built with its lengths stated as
multiples of the tile and picked at launch with ``rocke.runtime.first_supported``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import TYPE_CHECKING, Any

from rocke.core.dtypes import dtype_info
from rocke.core.facts import stated_facts_of
from rocke.core.ir import GPU_CHECKED_VALUE_WRITER, Value

from .descriptors import TensorWindow
from .encoding import lane_terms, register_coord_offsets
from .fragments import TileDesc
from .tensor_transforms.graph import TransformGraph, identity_graph
from .tensor_transforms.ranges import (
    AffineCoord,
    AxisBound,
    BoundCheck,
    CheckSide,
    Interval,
    LoopStopWithoutMultipleOf,
    VectorRun,
    required_checks_within,
)
from .tensor_transforms.runs import validity_uniform_over_run
from .tensor_transforms.walk import split_coordinate

if TYPE_CHECKING:
    from rocke.core.ir import IRBuilder

    from .descriptors import TensorDesc

__all__ = [
    "AccessPath",
    "AxisChecks",
    "BoundsCheckPlan",
    "DomainChecks",
    "GlobalForm",
    "SideCheck",
    "VectorValidity",
    "check_tile_rank",
    "contiguous_run",
    "element_bytes",
    "explain_bounds_checks",
    "global_store_width",
    "is_lds",
    "leading_rank",
    "plan_window",
]


class SideCheck(Enum):
    """How one side of one coordinate's bound is enforced."""

    NOT_NEEDED = "not needed (proven at build time)"
    BY_BUFFER = "checked by the buffer instruction"
    SCALAR_CHECK = "one scalar compare, the same for every lane"
    PER_LANE_CHECK = "one compare per lane"


class VectorValidity(Enum):
    """How the validity of one vector access is decided."""

    NOT_CHECKED = "not checked"
    FIRST_ELEMENT = "decided by its first element"
    FIRST_AND_LAST_ELEMENT = (
        "decided by its first and last element (it may cross a bound)"
    )


class AccessPath(Enum):
    """The memory instructions an access uses."""

    GLOBAL = "global memory instructions"
    BUFFER = "buffer memory instructions"


class GlobalForm(Enum):
    """How a global access forms its addresses; the verbs emit the form its plan names.

    - ``INDEX32``: every side proven, a plain window with int strides, no begin offset and every
      element offset inside i32 -- one i32 element index from the pointer.
    - ``PLACED``: every side proven otherwise -- the per-wave part of the address on the 64-bit
      pointer, the per-lane part as a zero-extended 32-bit byte offset.
    - ``MASKED``: a side is checked -- the placed address with every access under its validity, so
      no instruction touches an element outside the tensor.
    """

    INDEX32 = "one 32-bit element index from the pointer"
    PLACED = (
        "the per-wave part on the 64-bit pointer, the per-lane part a 32-bit offset"
    )
    MASKED = (
        "the per-wave part on the 64-bit pointer, the per-lane part a 32-bit offset, each "
        "access under its validity"
    )


@dataclass(frozen=True)
class AxisChecks:
    """How the two sides ``0 <= position`` and ``position < length`` of one axis are enforced."""

    lower: SideCheck
    upper: SideCheck


@dataclass(frozen=True)
class DomainChecks:
    """How the two sides of one graph coordinate's read domain are enforced, by the name the graph
    was built with."""

    coordinate: str
    lower: SideCheck
    upper: SideCheck


@dataclass(frozen=True)
class BoundsCheckPlan:
    """The checks one window access needs, and how they are enforced (built, never constructed by
    an author: :func:`explain_bounds_checks` returns it).

    - ``axes``: one :class:`AxisChecks` per TENSOR axis, leading axes included.
    - ``inputs`` / ``read_domains``: a transform-graph window's input extents and read domains;
      ``()`` for a plain :class:`~rocke.helpers.tiling.descriptors.TensorWindow`, whose input
      extents are its tensor bounds, reported once in ``axes``.
    - ``vector_axis``: the WINDOW axis each vector access runs along (``None`` for single-element
      accesses); ``vector_width``: the elements per access (a store on the ``INDEX32`` form writes
      one element per register, see :func:`global_store_width`); ``vector_validity``: how a
      vector's validity is decided.
    - ``edge_vectors_load_whole``: a vector crossing a bound is still one wide load, its elements
      outside replaced by a select (a buffer-path form; ``False`` on the global path).
    - ``path``: the memory instructions; ``form``: how a global access forms its addresses.
    """

    axes: tuple[AxisChecks, ...]
    inputs: tuple[AxisChecks, ...]
    read_domains: tuple[DomainChecks, ...]
    vector_axis: int | None
    vector_width: int
    vector_validity: VectorValidity
    edge_vectors_load_whole: bool
    path: AccessPath
    form: GlobalForm
    # Internal (not part of the public API, but compared like the public fields): the upper
    # sides the buffer instruction checks (none on the global path), and per tensor axis the
    # loop stops whose missing multiple-of fact is the only reason its upper side is checked
    # (empty for every other axis; ``str`` prints the fix).
    _buffer_checked_sides: frozenset[CheckSide] = field(default=frozenset(), repr=False)
    _upper_side_loop_stops: tuple[tuple[LoopStopWithoutMultipleOf, ...], ...] = field(
        default=(), repr=False
    )

    def __str__(self) -> str:
        lines = [f"bounds checks ({self.path.value}):"]
        for axis, (checks, stops) in enumerate(
            zip(self.axes, self._upper_side_loop_stops, strict=True)
        ):
            lines.append(f"  tensor axis {axis}, lower side: {checks.lower.value}")
            lines.append(
                f"  tensor axis {axis}, upper side: {checks.upper.value}"
                + _multiple_of_fix_text(stops)
            )
        for i, checks in enumerate(self.inputs):
            lines.append(f"  graph input {i}, lower side: {checks.lower.value}")
            lines.append(f"  graph input {i}, upper side: {checks.upper.value}")
        for domain in self.read_domains:
            lines.append(
                f"  read domain of '{domain.coordinate}', lower side: {domain.lower.value}"
            )
            lines.append(
                f"  read domain of '{domain.coordinate}', upper side: {domain.upper.value}"
            )
        if self.vector_axis is None:
            lines.append(
                f"  each access is one element, validity {self.vector_validity.value}"
            )
        else:
            lines.append(
                f"  each access is {self.vector_width} elements along window axis "
                f"{self.vector_axis}, validity {self.vector_validity.value}"
            )
        return "\n".join(lines)


def _multiple_of_fix_text(stops: tuple[LoopStopWithoutMultipleOf, ...]) -> str:
    """The fix ``str(plan)`` appends to an upper side checked only because of ``stops`` (``""``
    when there are none): the one home of that text.

    For each loop length, the multiple to state is the least common multiple of the loop step and
    of the multiple the length already states, stated where its facts are stated: a kernel
    parameter's ``b.param`` declaration, or the ``b.gpu_checked_value`` call that returned it. A
    length whose facts the author does not state (a marker another helper wrote, or one with a
    non-zero residue) has no such fix, and then no part of the fix is printed.
    """
    fixes = [_state_multiple_of(stop.length, stop.step) for stop in stops]
    return "" if None in fixes else "".join(fixes)


def _state_multiple_of(length: Value, step: int) -> str | None:
    """`` -- state multiple_of=N ...`` for one loop length and the step it must be a multiple of,
    or ``None`` when the author does not state that length's facts."""
    facts = stated_facts_of(length)
    if facts.residue != 0:
        return None
    multiple = math.lcm(step, facts.multiple_of)
    if length.param is not None:
        return f" -- state multiple_of={multiple} on {length.param.name}'s b.param declaration"
    if length.op is not None and length.op.attrs.get("by") == GPU_CHECKED_VALUE_WRITER:
        return (
            f" -- state multiple_of={multiple} in the b.gpu_checked_value call that returned "
            f"{length.name}"
        )
    return None


def explain_bounds_checks(
    b: IRBuilder, ptr: Value, window: TensorWindow, tile_desc: TileDesc
) -> BoundsCheckPlan:
    """The bound checks ``load_fragment`` / ``store_fragment`` apply to an access of ``tile_desc``
    through ``window`` and ``ptr`` -- which sides are proven, which are checked and how, how a
    vector's validity is decided and which memory instructions are used. Emits no IR.

    Give it the builder, pointer, window and tile the verb will be given: it reads the same
    origins through the same IR range walk (which may record the workgroup size a thread-id
    origin was walked with, exactly as the verb does), so it reports exactly what the verb emits::

        plan = explain_bounds_checks(b, a_ptr, a_win, a_desc)
        print(plan)          # one line per tensor axis and side

    ``ptr`` is a global pointer: LDS accesses are never bound-checked (the buffer holds the
    descriptor's whole span), so an LDS buffer is refused. A pointer parameter declared
    ``addr_space="constant"`` is refused when the access needs 64-bit pointer arithmetic (every
    form but the one-i32-index form), exactly as the verbs refuse it.
    """
    if is_lds(ptr):
        raise ValueError(
            "explain_bounds_checks: ptr is an LDS buffer -- LDS accesses are not bound-checked "
            "(the buffer holds the descriptor's whole span); explain an access through a global "
            "pointer"
        )
    plan, _ = plan_window("explain_bounds_checks", b, ptr, window, tile_desc)
    return plan


def global_store_width(plan: BoundsCheckPlan) -> int:
    """The elements one global store instruction of ``store_fragment`` writes for an access ``plan``
    describes: 1 on the one-32-bit-index form, whose per-register stores the backend's store
    vectorizer may merge, else the plan's ``vector_width``. (Every global load is
    ``vector_width`` wide.)"""
    return 1 if plan.form is GlobalForm.INDEX32 else plan.vector_width


# --- the decision (shared with the verbs) -----------------------------------------------------------


def plan_window(
    verb: str, b: IRBuilder, ptr: Value, window: TensorWindow, tile_desc: TileDesc
) -> tuple[BoundsCheckPlan, tuple[AffineCoord, ...]]:
    """The plan of a global access through ``ptr``, and each tensor axis's origin as the IR range
    walk splits it (the verbs build their access coordinates from these splits). Refuses a
    pointer the plan's form cannot address (:func:`_require_pointer_for_form`)."""
    if not isinstance(window, TensorWindow):
        raise TypeError(
            f"{verb} takes a TensorWindow -- got a {type(window).__name__}; build one with "
            "make_window(tensor, origin)"
        )
    check_tile_rank(verb, window, tile_desc)
    tensor = window.tensor
    lead = leading_rank(window, len(tile_desc.shape))
    origins = tuple(split_coordinate(b, origin) for origin in window.origin)
    lanes = lane_terms(tile_desc.layout)

    tile_extents = (None,) * lead + tuple(tile_desc.shape)
    ranges = [
        _positions(origin, extent)
        for origin, extent in zip(origins, tile_extents, strict=True)
    ]
    per_lane = [
        origin.lane is not None or (axis >= lead and lanes[axis - lead].extent > 1)
        for axis, origin in enumerate(origins)
    ]

    graph = identity_graph(tensor.lengths)
    bounds = tuple(AxisBound.of(0, length) for length in tensor.lengths)
    checks = required_checks_within(graph, tuple(ranges), bounds)
    by_name = {check.coordinate: check for check in checks}
    axes = tuple(
        _axis_checks(by_name.get(name), per_lane[axis])
        for axis, name in enumerate(graph.outputs)
    )
    upper_side_loop_stops = _upper_side_loop_stops(
        graph, bounds, origins, tile_extents, by_name
    )

    width = contiguous_run(tile_desc, window, tensor.dtype)
    vector_axis = None
    if width > 1:
        vector_axis = lead + tile_desc.layout.register_to_rh_major[-1] - 1
    checked = any(
        side is not SideCheck.NOT_NEEDED
        for axis in axes
        for side in (axis.lower, axis.upper)
    )
    if not checked:
        validity = VectorValidity.NOT_CHECKED
    elif width == 1 or _uniform_over_every_vector(
        graph, checks, tile_desc, origins, lead, vector_axis, width
    ):
        validity = VectorValidity.FIRST_ELEMENT
    else:
        validity = VectorValidity.FIRST_AND_LAST_ELEMENT

    if checked:
        form = GlobalForm.MASKED
    elif _fits_index32(tensor, ranges):
        form = GlobalForm.INDEX32
    else:
        form = GlobalForm.PLACED
    plan = BoundsCheckPlan(
        axes=axes,
        inputs=(),
        read_domains=(),
        vector_axis=vector_axis,
        vector_width=width,
        vector_validity=validity,
        edge_vectors_load_whole=False,
        path=AccessPath.GLOBAL,
        form=form,
        _upper_side_loop_stops=upper_side_loop_stops,
    )
    _require_pointer_for_form(verb, ptr, form)
    return plan, origins


def _positions(origin: AffineCoord, tile_extent: int | None) -> Interval:
    """The positions one tensor axis takes: its origin's range, plus ``[0, tile_extent)`` on an
    axis the tile covers (``tile_extent`` is ``None`` on a leading axis: its origin alone).
    """
    if tile_extent is None:
        return origin.range
    return origin.range.plus(Interval(0, tile_extent))


def _upper_side_loop_stops(
    graph: TransformGraph,
    bounds: tuple[AxisBound, ...],
    origins: tuple[AffineCoord, ...],
    tile_extents: tuple[int | None, ...],
    checks_by_name: dict[str, BoundCheck],
) -> tuple[tuple[LoopStopWithoutMultipleOf, ...], ...]:
    """Per tensor axis, the loop stops whose missing multiple-of fact is the only reason its
    upper side is checked.

    They are the stops the IR range walk recorded on the axis origin's loop counters, kept when
    ``required_checks_within`` proves that upper side once each such counter has the range
    stating the fact would give it. Empty for an axis whose upper side is proven, has no such
    counter, or stays checked for another reason. The record reaches a counter that is a summand of
    the origin on its own (``k``, ``k + 16``, ``k`` plus a per-lane offset); a counter folded into a
    larger wave-uniform value (``k_base + k``) carries none.
    """
    ranges = [
        _positions(origin, extent)
        for origin, extent in zip(origins, tile_extents, strict=True)
    ]
    per_axis: list[tuple[LoopStopWithoutMultipleOf, ...]] = []
    for axis, name in enumerate(graph.outputs):
        check = checks_by_name.get(name)
        parts = origins[axis].uniform
        stops = tuple(
            part.loop_stop_without_multiple_of
            for part in parts
            if part.loop_stop_without_multiple_of is not None
        )
        if check is None or check.hi is None or not stops:
            per_axis.append(())
            continue
        stated_parts = tuple(
            (
                part
                if part.loop_stop_without_multiple_of is None
                else replace(
                    part, range=part.loop_stop_without_multiple_of.range_if_stated
                )
            )
            for part in parts
        )
        stated_ranges = list(ranges)
        stated_ranges[axis] = _positions(
            replace(origins[axis], uniform=stated_parts), tile_extents[axis]
        )
        still_checked = any(
            stated.coordinate == name and stated.hi is not None
            for stated in required_checks_within(graph, tuple(stated_ranges), bounds)
        )
        per_axis.append(() if still_checked else stops)
    return tuple(per_axis)


def _require_pointer_for_form(verb: str, ptr: Value, form: GlobalForm) -> None:
    """Refuse a constant-space pointer parameter for every form but ``INDEX32``.

    A pointer parameter declared ``addr_space="constant"`` lowers to ``ptr addrspace(4)``. The
    placed and masked forms move the per-wave part of the address onto the 64-bit pointer with
    core's ``global_ptr_*`` ops, which take only a global pointer and refuse this one at lowering,
    with no word about the window; so they refuse it here, at build, before any IR, naming the
    parameter -- a ``TypeError``, as core's refusal of a pointer in the wrong space is.

    The one-i32-index form stays exempt on purpose: it hands the pointer itself to core's global
    loads and stores, and relies on their lowering to handle a constant-space pointer -- to read
    through it in the constant address space and to refuse a store through it. The tiling layer
    adds no rule of its own there. Until core's lowering does both, such an access lowers to IR
    that names the pointer in the global address space, which LLVM rejects.
    """
    if form is GlobalForm.INDEX32:
        return
    param = ptr.param
    if param is not None and param.attrs.get("addr_space") == "constant":
        raise TypeError(
            f'{verb}: ptr is the parameter {param.name}, declared addr_space="constant"; a '
            "bound-checked or 64-bit-addressed access needs a global pointer -- declare "
            f"{param.name} without addr_space"
        )


def _axis_checks(check: BoundCheck | None, per_lane: bool) -> AxisChecks:
    """One tensor axis's two sides: ``NOT_NEEDED`` where the rule proved them, else the compare
    its position needs (a scalar one when the position is the same for every lane)."""
    kind = SideCheck.PER_LANE_CHECK if per_lane else SideCheck.SCALAR_CHECK
    if check is None:
        return AxisChecks(SideCheck.NOT_NEEDED, SideCheck.NOT_NEEDED)
    return AxisChecks(
        SideCheck.NOT_NEEDED if check.lo is None else kind,
        SideCheck.NOT_NEEDED if check.hi is None else kind,
    )


def _uniform_over_every_vector(
    graph: TransformGraph,
    checks: tuple[BoundCheck, ...],
    tile_desc: TileDesc,
    origins: tuple[AffineCoord, ...],
    lead: int,
    vector_axis: int,
    width: int,
) -> bool:
    """Every check has one answer for each whole vector, asked of the tensor_transforms run query
    for the first element of every vector.

    The first element is the origin plus the register offset of the vector's first register. The
    layout's per-lane term is left out: a vector runs along the innermost level of its axis, so
    every other level -- each lane bucket included -- moves the coordinate by a multiple of the
    run length, which the width divides; it cannot change the residue the query reads.
    """
    run = VectorRun(vector_axis, width)
    for register in range(0, tile_desc.register_count, width):
        offsets = register_coord_offsets(tile_desc.layout, register)
        coords = tuple(
            origin + AffineCoord.of_int(offsets[axis - lead] if axis >= lead else 0)
            for axis, origin in enumerate(origins)
        )
        if not validity_uniform_over_run(graph, checks, run, coords):
            return False
    return True


_I32_LO, _I32_END = -(1 << 31), 1 << 31


def _fits_index32(tensor: TensorDesc, ranges: list[Interval]) -> bool:
    """Whether the one-i32-element-index form can address every element: int strides, no begin
    offset, and ``sum(position * stride)`` inside i32 over the positions' numeric ranges.
    """
    if any(isinstance(stride, Value) for stride in tensor.strides):
        return False
    if any(not (isinstance(begin, int) and begin == 0) for begin in tensor.begin):
        return False
    least = most = 0
    for positions, stride in zip(ranges, tensor.strides):
        lo, last = positions.numeric_ends()
        if lo is None or last is None:
            return False
        least += min(lo * stride, last * stride)
        most += max(lo * stride, last * stride)
    return _I32_LO <= least and most < _I32_END


# --- the layout facts and the vector width -----------------------------------------------------------


def leading_rank(window: TensorWindow, tile_rank: int) -> int:
    """How many leading tensor axes a rank-``tile_rank`` tile leaves to the window origin. A tile
    covers the window's LAST axes, so tile axis ``i`` is tensor axis ``leading_rank + i``.
    """
    return window.tensor.rank - tile_rank


def check_tile_rank(verb: str, window: TensorWindow, tile_desc: TileDesc) -> None:
    """Fail fast unless the tile fits the window: a rank-t tile covers the window's last t axes."""
    tile_rank, window_rank = len(tile_desc.shape), window.tensor.rank
    if tile_rank > window_rank:
        raise ValueError(
            f"{verb}: the tile has rank {tile_rank} but the window has rank {window_rank} -- a "
            f"rank-t tile covers the window's last t axes, so t <= r"
        )


def is_lds(source: Any) -> bool:
    """True when `source` is an LDS buffer (an ``smem<...>`` value from ``smem_alloc``) rather
    than a global/constant ``ptr<...>``. Lets one verb serve both spaces -- the load/store op is
    chosen from the SOURCE, so the caller just hands the buffer (global ptr OR LDS smem) it has.
    """
    name = getattr(getattr(source, "type", None), "name", "")
    return isinstance(name, str) and name.startswith("smem")


def element_bytes(dtype: str) -> int:
    """The bytes one element of the dtype named ``dtype`` (a ``Type.name``) occupies, from its
    encoded width in ``rocke.core.dtypes.dtype_info``. It drives the access width and alignment, so
    no dtype is baked into the verbs.

    Raises ``ValueError`` for a dtype that is not a whole number of bytes (the verbs move whole
    elements) and, from ``dtype_info``, for a name that is no dtype.
    """
    info = dtype_info(dtype)
    if info.encoded_bits % 8:
        raise ValueError(
            f"element_bytes failed - dtype={info.name!r} is {info.encoded_bits} bits, expected "
            "a whole number of bytes; the tiling verbs move whole elements"
        )
    return info.encoded_bits // 8


# The one home of the dtypes whose runs the verbs access as one vector (``global_load_vN`` /
# ``smem_load_vN`` and their stores); every other dtype is accessed one element at a time.
_WIDE_OK = frozenset({"f16", "bf16", "f32", "i32"})


def contiguous_run(tile_desc: TileDesc, window: TensorWindow, dtype: Any) -> int:
    """Width of the innermost stride-1 register run -> a single WIDE access; else 1 (scalar).

    The innermost register bucket varies fastest in the register index. Its registers are
    consecutive positions on their axis only when the bucket is that axis's INNERMOST hierarchical
    level (levels of length 1 inside it do not count). A bucket on an outer level steps the position
    by the product of the levels inside it -- an interleaved layout holds positions 0, 4, 8, 12 in
    four consecutive registers -- so it gives no run. When the bucket is the innermost level of the
    tensor's stride-1 axis, its registers are memory-contiguous and can be one wide load / store.
    Width is the largest supported vector size for the dtype (8/4/2) dividing the run, at most 16
    bytes. The rule is the same in global memory and LDS: both address an element by
    ``sum(position * stride)``. Whether a wide global access needs a bound check is a separate
    question, answered by the access's :class:`BoundsCheckPlan` (a checked vector is masked by its
    validity, never narrowed).

    The width must also divide every OTHER stride, leading axes included: a run then starts at a
    multiple of the width whatever the positions on the other axes, so the ``align`` the access
    claims holds (an odd batch stride would otherwise start a wide run mid-vector). A runtime stride
    on another axis has no compile-time residue here, so the access stays scalar."""
    layout = tile_desc.layout
    if dtype.name not in _WIDE_OK:
        return 1
    majors, minors = layout.register_to_rh_major, layout.register_to_rh_minor
    if not majors:
        return 1
    lead = leading_rank(window, len(tile_desc.shape))
    tile_axis = majors[-1] - 1  # innermost register bucket's X-dim -> tile axis
    strides = window.tensor.strides
    run_axis = lead + tile_axis
    if tile_axis < 0 or strides[run_axis] != 1:
        return 1
    inner_levels = layout.hierarchical_lengths[tile_axis][minors[-1] + 1 :]
    if math.prod(inner_levels) != 1:
        return 1  # an outer level of its axis: consecutive registers are not consecutive positions
    other_strides = [stride for axis, stride in enumerate(strides) if axis != run_axis]
    if any(isinstance(stride, Value) for stride in other_strides):
        return 1
    length = layout.bucket_length(majors[-1], minors[-1])
    # One access is at most a dwordx4 (16 bytes); a longer run takes several.
    max_vw = 16 // element_bytes(dtype.name)
    for vw in (16, 8, 4, 2):
        if (
            vw <= max_vw
            and length % vw == 0
            and all(stride % vw == 0 for stride in other_strides)
        ):
            return vw
    return 1
