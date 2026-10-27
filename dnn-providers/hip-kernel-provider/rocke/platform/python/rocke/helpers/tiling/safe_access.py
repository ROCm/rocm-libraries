# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Safe access: deliver a window's accesses to global memory so that no instruction touches a byte
outside the tensor (internal; reached only through ``load_fragment`` / ``store_fragment``).

The verbs hand over one coordinate tuple per access (one split coordinate per tensor axis, the
vector's first element) and the access's :class:`~rocke.helpers.tiling.clipping.BoundsCheckPlan`.
This module asks tensor_transforms for each access's split element address and validity
(``emit_accesses``) and places them; it adds no check of its own.

Placement (every side proven, and every checked access before it is masked):

- the per-wave part of the address (every uniform term, a slice's begin and a runtime-stride
  delta) is added to the 64-bit pointer, once per distinct set of terms;
- the per-lane part, which starts at 0 by construction, becomes a byte offset in 32 bits,
  zero-extended to 64 and applied as its own offset (the pointer-plus-32-bit-offset form of the
  global instructions); its largest value, plus one vector, must stay below ``2^31`` bytes;
- the constant is a trailing element offset the instruction can fold.

A proven window may use in-bounds pointer arithmetic for a per-wave part proven non-negative. A
checked window never does: its invalid lanes compute addresses too, so every part that can move an
address outside the tensor is an offset that makes no in-bounds promise, and nothing is
dereferenced by an invalid lane.

The masked form of a checked access (the same on every target; it reads no buffer fact):

- validity from the first element (every check has one answer for the whole vector): the wide
  access runs only where it is valid; a load yields zeros elsewhere;
- validity from the first and last element (the vector may cross a bound; ``valid_span`` gives
  the valid elements ``[first, end)``): the wide access runs where the whole vector is valid; where
  it is partly valid ("straddles"), each valid element is accessed alone; a load yields zeros for
  every other element.

No access is clamped to a safe address: a tensor of runtime length 0 has no element to clamp to.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from functools import partial
from math import gcd
from typing import TYPE_CHECKING

from rocke.core.facts import stated_facts_of
from rocke.core.ir import I64, Type, Value, VectorType

from .clipping import BoundsCheckPlan, GlobalForm, element_bytes
from .encoding import lane_terms, register_coord_offsets
from .tensor_transforms.emit import (
    Access,
    ElementAddress,
    emit_accesses,
    emit_accesses_except_buffer_checked,
)
from .tensor_transforms.graph import identity_graph
from .tensor_transforms.ranges import AffineCoord, Congruence, Interval
from .tensor_transforms.walk import split_coordinate

if TYPE_CHECKING:
    from rocke.core.ir import IRBuilder

    from .descriptors import TensorDesc
    from .fragments import TileDesc

__all__: list[str] = []

# The largest byte offset the zero-extended 32-bit per-lane part may reach, one vector included.
_LANE_BYTE_LIMIT = 1 << 31


def load_accesses(
    b: IRBuilder,
    ptr: Value,
    tensor: TensorDesc,
    accesses: Sequence[tuple[AffineCoord, ...]],
    plan: BoundsCheckPlan,
    align: int,
) -> list[Value]:
    """Load every access: one value per access (a ``vector_width`` vector, or a scalar for one
    element), 0 in every element outside the tensor. ``align`` is the alignment every access may
    claim (:func:`access_alignment`)."""
    width = plan.vector_width
    dtype = tensor.dtype
    placer = _Placer(b, ptr, width, masked=_masked(plan))
    loaded = []
    for access in _emit(b, tensor, accesses, plan):
        pointer, index = placer.place(access.address)
        wide = partial(_load, b, pointer, index, dtype, width, align)
        if access.valid is None:
            loaded.append(wide())
        elif access.valid_span is None:
            loaded.append(
                _yielding_if(
                    b,
                    access.valid,
                    _vector_type(dtype, width),
                    wide,
                    _zero(b, dtype, width),
                )
            )
        else:
            loaded.append(
                _load_straddling(b, access, pointer, index, dtype, width, align, wide)
            )
    return loaded


def store_accesses(
    b: IRBuilder,
    ptr: Value,
    tensor: TensorDesc,
    accesses: Sequence[tuple[AffineCoord, ...]],
    values: Sequence[Value],
    plan: BoundsCheckPlan,
    align: int,
) -> None:
    """Store ``values[i]`` (a ``vector_width`` vector, or a scalar) at access ``i``; no element
    outside the tensor is written. ``align`` is the alignment every access may claim
    (:func:`access_alignment`)."""
    width = plan.vector_width
    placer = _Placer(b, ptr, width, masked=_masked(plan))
    element_align = _element_claim(align, tensor.dtype)
    for access, value in zip(_emit(b, tensor, accesses, plan), values, strict=True):
        pointer, index = placer.place(access.address)
        if access.valid is None:
            _store(b, pointer, index, value, width, align)
        elif access.valid_span is None:
            with b.scf_if(access.valid):
                _store(b, pointer, index, value, width, align)
        else:
            whole, straddle, in_span = _span_tests(b, access, width)
            with b.scf_if(whole):
                _store(b, pointer, index, value, width, align)
            with b.scf_if(straddle):
                for element in range(width):
                    with b.scf_if(in_span[element]):
                        _store(
                            b,
                            pointer,
                            index + element,
                            b.vec_extract(value, element),
                            1,
                            element_align,
                        )


# --- the alignment an access claims ---------------------------------------------------------------

# The largest alignment any access claims: one 16-byte vector. Every residue below is kept modulo
# this many elements, which decides divisibility by every power of two up to it.
_MOST_ALIGN = 16


def access_alignment(
    b: IRBuilder,
    ptr: Value,
    tensor: TensorDesc,
    origins: Sequence[AffineCoord],
    tile_desc: TileDesc,
    width: int,
) -> int:
    """The alignment, in bytes, every access of a ``width``-element run of ``tile_desc`` through
    ``tensor`` at ``origins`` may claim: the largest power of two the facts prove divides the
    address of every access's first element, never more than the pointer's alignment nor the
    access's own byte width.

    - The pointer's alignment is the ``align=`` its kernel parameter declares, else (a parameter
      without one, or a pointer that is not a parameter) the element's byte size.
    - The element offset is ``sum((position + begin) * stride)`` over the tensor's axes, its
      residue combined exactly from the residue of each part: an int by its value, a value by the
      multiple-of fact the IR range walk proves for it, a runtime stride by its stated
      ``multiple_of``. A position is the origin plus any lane's and any run's offset in the
      tile: its residue is the origin's, known only modulo the gcd of those offsets.

    A claim above the truth is a wrong answer, not a slow one: a uniform address claiming 4 bytes
    or more may be read by a scalar load, which ignores the low address bits. So an f16 access at
    an odd offset claims 2 bytes, whatever its width::

        A (M, K) f16, strides (lda, 1), pointer align=16, 8-wide K runs
        origin K = block_id * 32, lda = 64           -> 16 bytes
        origin K = a kernel parameter (no fact)       -> 2 bytes
        origin K = 1, begin K = 1                     -> 4 bytes (the offset is 2 elements)
        lda stated multiple_of=8, pointer align=4     -> 4 bytes
    """
    elem = element_bytes(tensor.dtype.name)
    layout = tile_desc.layout
    lead = tensor.rank - len(tile_desc.shape)
    lanes = lane_terms(layout)
    run_offsets = [
        register_coord_offsets(layout, register)
        for register in range(0, tile_desc.register_count, width)
    ]
    offset = _exact(0)
    for axis, (origin, begin, stride) in enumerate(
        zip(origins, tensor.begin, tensor.strides)
    ):
        position = _coordinate_residue(origin).plus(
            _coordinate_residue(split_coordinate(b, begin))
        )
        if axis >= lead:
            tile_axis = axis - lead
            spread = 0
            if lanes[tile_axis].extent > 1:
                spread = lanes[tile_axis].step
            for offsets in run_offsets:
                spread = gcd(spread, offsets[tile_axis])
            position = position.plus(_kept(Congruence(spread or _MOST_ALIGN, 0)))
        offset = offset.plus(_product(position, _stride_residue(stride)))
    divides = gcd(offset.modulus, offset.residue) * elem
    return min(
        _pointer_alignment(b, ptr, elem),
        width * elem,
        divides & -divides,
        _MOST_ALIGN,
    )


def _pointer_alignment(b: IRBuilder, ptr: Value, elem: int) -> int:
    """The ``align=`` declared on the kernel parameter ``ptr`` is; the element's byte size for a
    parameter without one and for a pointer that is not a parameter."""
    for param in b.kernel.params:
        if b.get_param(param.name) is ptr:
            declared = param.attrs.get("align")
            if not declared:
                return elem
            declared = int(declared)
            return declared & -declared
    return elem


def _kept(congruence: Congruence) -> Congruence:
    """The residue modulo ``_MOST_ALIGN`` elements (the power-of-two part of the modulus)."""
    return Congruence(gcd(congruence.modulus, _MOST_ALIGN), congruence.residue)


def _exact(n: int) -> Congruence:
    return Congruence(_MOST_ALIGN, n)


def _product(x: Congruence, y: Congruence) -> Congruence:
    """The residue of ``x * y``: with ``x = r + m * i`` and ``y = s + n * j`` the product is
    ``r * s`` plus a multiple of ``gcd(r * n, s * m, m * n)``."""
    m, r, n, s = x.modulus, x.residue, y.modulus, y.residue
    return _kept(Congruence(gcd(r * n, s * m, m * n), r * s))


def _coordinate_residue(coord: AffineCoord) -> Congruence:
    """A split coordinate's residue: an exact constant by its value, else the walk's fact."""
    if not coord.uniform and coord.lane is None:
        return _exact(coord.const)
    return _kept(coord.congruence)


def _stride_residue(stride: int | Value) -> Congruence:
    """A stride's residue: an int by its value, a runtime stride by its stated multiple-of
    fact."""
    if isinstance(stride, Value):
        facts = stated_facts_of(stride)
        return _kept(Congruence(facts.multiple_of, facts.residue))
    return _exact(stride)


def _element_claim(align: int, dtype: Type) -> int:
    """The claim of one element of an access that claims ``align``: never above the element's
    own byte size."""
    return min(align, element_bytes(dtype.name))


# --- the accesses -------------------------------------------------------------------------------


def _masked(plan: BoundsCheckPlan) -> bool:
    return plan.form is GlobalForm.MASKED


def _emit(
    b: IRBuilder,
    tensor: TensorDesc,
    accesses: Sequence[tuple[AffineCoord, ...]],
    plan: BoundsCheckPlan,
) -> tuple[Access, ...]:
    """Each access's split address and validity, from tensor_transforms over the identity graph of
    the tensor (a plain window's coordinates are its tensor positions)."""
    graph = identity_graph(tensor.lengths)
    vector = {}
    if plan.vector_width > 1:
        vector = {"vector_axis": plan.vector_axis, "vector_width": plan.vector_width}
    if plan._buffer_checked_sides:
        return emit_accesses_except_buffer_checked(
            b,
            graph,
            tensor,
            list(accesses),
            buffer_checked_sides=plan._buffer_checked_sides,
            **vector,
        )
    return emit_accesses(b, graph, tensor, list(accesses), **vector)


class _Placer:
    """Turns split element addresses into a pointer and a trailing element index, sharing the
    per-wave base and the per-lane byte offset between the accesses that have them in common.
    """

    def __init__(self, b: IRBuilder, ptr: Value, width: int, *, masked: bool) -> None:
        self._b = b
        self._ptr = ptr
        self._width = width
        self._masked = masked
        self._bases: dict[tuple[int, ...], Value] = {}
        self._lane_offsets: dict[int, Value] = {}

    def place(self, address: ElementAddress) -> tuple[Value, int]:
        """The pointer of ``address`` without its constant, and the constant as the element index
        the instruction adds (in the masked form the constant is applied to the pointer too, and
        the index is 0, so no in-bounds promise covers an address an invalid lane computes).
        """
        b = self._b
        elem_bytes = address.elem_bits // 8
        pointer = self._base(address, elem_bytes)
        if address.lane is not None:
            pointer = b.global_ptr_offset(
                pointer, self._lane_bytes(address, elem_bytes)
            )
        if not self._masked:
            return pointer, address.const
        if address.const:
            pointer = b.global_ptr_offset(
                pointer, b.const_i64(address.const * elem_bytes)
            )
        return pointer, 0

    def _base(self, address: ElementAddress, elem_bytes: int) -> Value:
        """``ptr`` plus every per-wave term of the address, in bytes, once per distinct set."""
        terms = [(term.value, term.range) for term in address.uniform]
        if address.uniform_delta is not None:
            terms.append((address.uniform_delta, address.uniform_delta_range))
        if not terms:
            return self._ptr
        key = tuple(id(value) for value, _ in terms)
        if key not in self._bases:
            b = self._b
            total = None
            for value, _ in terms:
                wide = value if value.type == I64 else b.sext(value, I64)
                total = wide if total is None else b.add(total, wide)
            if elem_bytes != 1:
                total = b.mul(total, b.const_i64(elem_bytes))
            non_negative = all(
                Interval.from_pair(term_range).within(0, None)
                for _, term_range in terms
            )
            if non_negative and not self._masked:
                self._bases[key] = b.global_ptr_add(self._ptr, total)
            else:
                self._bases[key] = b.global_ptr_offset(self._ptr, total)
        return self._bases[key]

    def _lane_bytes(self, address: ElementAddress, elem_bytes: int) -> Value:
        """The per-lane part in bytes, zero-extended to 64 bits (it starts at 0 by construction)."""
        lane = address.lane
        if id(lane) not in self._lane_offsets:
            lane_max = address.lane_range[1] - 1
            vector_bytes = self._width * elem_bytes
            reach = lane_max * elem_bytes + vector_bytes
            if reach >= _LANE_BYTE_LIMIT:
                raise ValueError(
                    f"lane offset can reach {reach} bytes (lane {lane_max} elements x "
                    f"{address.elem_bits} bits + {vector_bytes} B vector + soffset 0 B), past "
                    "the 2^31-byte buffer limit -- move a larger part of the address into the "
                    "uniform (per-wave) coordinate"
                )
            b = self._b
            in_bytes = lane if elem_bytes == 1 else b.mul(lane, b.const_i32(elem_bytes))
            self._lane_offsets[id(lane)] = b.zext(in_bytes, I64)
        return self._lane_offsets[id(lane)]


# --- loads and stores ---------------------------------------------------------------------------


def _vector_type(dtype: Type, width: int) -> Type:
    return VectorType(dtype, width) if width > 1 else dtype


def _zero(b: IRBuilder, dtype: Type, width: int) -> Callable[[], Value]:
    """A builder of the zero an invalid access reads (emitted where it is used)."""
    if width > 1:
        return lambda: b.zero_vec(dtype, width)
    return lambda: b.vec_extract(b.zero_vec(dtype, 1), 0)


def _load(
    b: IRBuilder, pointer: Value, index: int, dtype: Type, width: int, align: int
) -> Value:
    if width > 1:
        return b.global_load_vN(pointer, b.const_i32(index), dtype, width, align=align)
    return b.global_load(pointer, b.const_i32(index), dtype, align=align)


def _store(
    b: IRBuilder, pointer: Value, index: int, value: Value, width: int, align: int
) -> None:
    if width > 1:
        b.global_store_vN(pointer, b.const_i32(index), value, width, align=align)
    else:
        b.global_store(pointer, b.const_i32(index), value, align=align)


def _yielding_if(
    b: IRBuilder,
    cond: Value,
    result_type: Type,
    then_value: Callable[[], Value],
    else_value: Callable[[], Value],
) -> Value:
    """``then_value()`` where ``cond``, else ``else_value()``; each side is emitted in its own
    branch, so an access in ``then_value`` runs only in the lanes where ``cond`` holds.
    """
    branch = b.scf_if_else(cond, result_types=[result_type])
    with branch as (then_side, else_side):
        with then_side:
            b.scf_yield(then_value())
        with else_side:
            b.scf_yield(else_value())
    return branch.results[0]


def _span_tests(
    b: IRBuilder, access: Access, width: int
) -> tuple[Value, Value, list[Value]]:
    """For a vector that may cross a bound: whether it is valid whole, whether it is partly valid
    ("straddles"), and per element whether it lies in the valid span ``[first, end)``.
    """
    whole = access.valid
    first, end = access.valid_span
    none = b.cmp_le(end, first)
    straddle = b.land(b.lnot(whole), b.lnot(none))
    in_span = [
        b.land(
            b.cmp_le(first, b.const_i32(element)), b.cmp_lt(b.const_i32(element), end)
        )
        for element in range(width)
    ]
    return whole, straddle, in_span


def _load_straddling(
    b: IRBuilder,
    access: Access,
    pointer: Value,
    index: int,
    dtype: Type,
    width: int,
    align: int,
    wide: Callable[[], Value],
) -> Value:
    """A vector that may cross a bound: the wide load where it is valid whole; where it straddles,
    each valid element loaded alone; zeros everywhere else."""
    zero = _zero(b, dtype, width)
    whole, straddle, in_span = _span_tests(b, access, width)
    whole_value = _yielding_if(b, whole, _vector_type(dtype, width), wide, zero)

    def elements() -> Value:
        vector = zero()
        for element in range(width):
            loaded = _yielding_if(
                b,
                in_span[element],
                dtype,
                partial(
                    _load,
                    b,
                    pointer,
                    index + element,
                    dtype,
                    1,
                    _element_claim(align, dtype),
                ),
                _zero(b, dtype, 1),
            )
            vector = b.vec_insert(vector, loaded, element)
        return vector

    part_value = _yielding_if(b, straddle, _vector_type(dtype, width), elements, zero)
    return b.select(whole, whole_value, part_value)
