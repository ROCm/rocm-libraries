# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The register-side types: a :class:`TileDesc` (logical layout) and the :class:`Fragment` (its
realized per-lane registers).

This is the REGISTER side of the surface, the counterpart to
:mod:`rocke.helpers.tiling.descriptors` (the memory side). It never emits IR (no IRBuilder). A
`TileDesc` is a frozen value saying *where each element lives* in lanes/registers (a shape + a
:class:`~rocke.helpers.tiling.encoding.WarpDistributionEncoding`). A `Fragment` binds that layout
to an element `dtype` and holds the SSA `value` with the registers; `value` is its only mutable
field, and every write to it is checked against the layout and dtype. The IR verbs in
:mod:`rocke.helpers.tiling.emit` fill / load / store the `Fragment`.

Terminology: a *register* here is one logical vector slot holding one element of the fragment's
`dtype` (``register_count`` slots per lane), not a physical 32-bit VGPR. LLVM packs the slots into
VGPRs: a gfx90a MFMA A operand of ``vec<f16x4>`` occupies two 32-bit registers, and a 64-bit
element occupies two. Sub-byte formats, which LLVM cannot pack, are not supported here.
"""

from __future__ import annotations

from dataclasses import dataclass

from ...core.ir import Type, Value, VectorType
from .encoding import WarpDistributionEncoding
from .register_mapper import RegisterMapper

__all__ = ["TileDesc", "Fragment", "make_fragment", "fragment_length"]


def fragment_length(encoding: WarpDistributionEncoding) -> int:
    """Per-lane register count for an encoding (= num_vector_items)."""
    return RegisterMapper(encoding).num_vector_items


@dataclass(frozen=True)
class TileDesc:
    """A logical-matrix -> per-lane-register layout descriptor.

    DTYPE-FREE and MEMORY-FREE: it says *where each element lives* (shape + the warp
    distribution `layout`), never *what type it is* or *which buffer*. The same TileDesc is
    reusable across dtypes -- the type is bound only when a `Fragment` is realized (load) or
    written (store). `shape` is the logical (rows, cols) of the tile.
    """

    shape: tuple[int, ...]
    layout: WarpDistributionEncoding

    @property
    def register_count(self) -> int:
        """Per-lane register count implied by the layout."""
        return fragment_length(self.layout)

    def swap_dims(self, i: int, j: int) -> "TileDesc":
        """Swap two X-dims -- a free-symmetry REPOSITION (coordinate transpose), register-identity
        and label-invariant. The same elements on the same lanes, addressed in transposed axis order
        (e.g. index a free-innermost memref in (K, free) order while the MMA side keeps (free, K)).

        Rank-generic: ``i``/``j`` are X-dim indices. An ``rh_major`` value ``v`` names replication
        (``v==0``) or X-dim ``v-1``, so swapping X-dims ``i`` and ``j`` remaps majors ``i+1<->j+1``
        and leaves replication + all minors untouched.
        """
        e = self.layout
        n = len(e.hierarchical_lengths)
        if not (0 <= i < n and 0 <= j < n):
            raise ValueError(f"swap_dims axes out of range -- i={i}, j={j}, rank={n}")

        def remap(v: int) -> int:
            if v == i + 1:
                return j + 1
            if v == j + 1:
                return i + 1
            return v

        hl = list(e.hierarchical_lengths)
        hl[i], hl[j] = hl[j], hl[i]
        shape = list(self.shape)
        shape[i], shape[j] = shape[j], shape[i]
        return TileDesc(
            shape=tuple(shape),
            layout=WarpDistributionEncoding(
                replication_lengths=e.replication_lengths,
                hierarchical_lengths=tuple(hl),
                lane_to_rh_major=tuple(
                    tuple(remap(m) for m in row) for row in e.lane_to_rh_major
                ),
                lane_to_rh_minor=e.lane_to_rh_minor,
                register_to_rh_major=tuple(remap(m) for m in e.register_to_rh_major),
                register_to_rh_minor=e.register_to_rh_minor,
            ),
        )

    def reorder_registers(self, order: tuple[int, ...]) -> "TileDesc":
        """Permute the REGISTER-bucket significance (outer = most-significant slot), keeping the lane
        map and every label identical -- the same elements on the same lanes, in a different register
        order. ``order`` is a permutation of the existing register buckets; callers name their buckets
        and compute the permutation (a bucket that collapses to extent 1 is absent from both the
        encoding and the permutation, so index math stays stable). ``make_tile_desc`` hardcodes
        ``block_repeat`` major / ``thread_tile`` minor, which cannot express the MMA-ready or the
        free-dim-fastest LDS-read order; this is the minimal escape.
        """
        e = self.layout
        if sorted(order) != list(range(len(e.register_to_rh_major))):
            raise ValueError(
                f"reorder_registers order must be a permutation of the "
                f"{len(e.register_to_rh_major)} register buckets -- got {order!r}"
            )
        return TileDesc(
            shape=self.shape,
            layout=WarpDistributionEncoding(
                replication_lengths=e.replication_lengths,
                hierarchical_lengths=e.hierarchical_lengths,
                lane_to_rh_major=e.lane_to_rh_major,
                lane_to_rh_minor=e.lane_to_rh_minor,
                register_to_rh_major=tuple(e.register_to_rh_major[i] for i in order),
                register_to_rh_minor=tuple(e.register_to_rh_minor[i] for i in order),
            ),
        )


def _fragment_dtype_ok(dtype: object) -> tuple[bool, str]:
    """Whether `dtype` can be a fragment's element type: a scalar IR `Type` (``F16``, ``F32``, ...),
    not a string token and not a composite (vector / pointer / LDS) type."""
    # Exact type, not isinstance: every scalar dtype is a bare `Type`, and its only subclasses
    # (VectorType, PtrType, SmemType) are composites. A future scalar subclass would need an
    # isinstance check that excludes those three instead.
    if type(dtype) is Type:
        return True, "ok"
    return False, (
        f"Fragment dtype must be a scalar IR Type (e.g. F16, F32) -- got {dtype!r} "
        f"({type(dtype).__name__})"
    )


def _fragment_value_type_ok(
    tile_desc: TileDesc, dtype: Type, value: object
) -> tuple[bool, str]:
    """Whether `value` can hold the registers of a `tile_desc` fragment at element `dtype`: an IR
    `Value` of type ``vec<dtype x register_count>``.

    One element per register slot -- packed sub-byte representations are not supported. A
    single-register fragment is a ``vec<dtype x 1>`` vector, never a bare scalar. This checks the
    register TYPE only: a value of the right type whose registers are in the wrong order, or an SSA
    value used outside the scope that defines it, is not caught here.
    """
    expected = VectorType(dtype, tile_desc.register_count)
    where = f"for TileDesc(shape={tile_desc.shape})"
    if not isinstance(value, Value):
        return False, (
            f"Fragment value must be an IR Value of type {expected.name} {where} -- got "
            f"{type(value).__name__} {value!r}"
        )
    got = f"Fragment value expected {expected.name} {where}, got"
    if not isinstance(value.type, VectorType):
        if type(value.type) is not Type:
            return False, (
                f"{got} non-vector {value.type.name} -- a fragment holds registers, not a memory "
                "handle; read memory into one with load_fragment"
            )
        if tile_desc.register_count == 1:
            fix = f"wrap the single register as a vec<{dtype.name}x1>"
        else:
            fix = (
                f"this TileDesc lays out {tile_desc.register_count} registers per lane; pass "
                "them as one vector"
            )
        if value.type != dtype:
            fix += (
                f"; and the register is {value.type.name}, not {dtype.name}; build the Fragment "
                f"at {value.type.name}"
            )
        return False, f"{got} scalar {value.type.name} -- {fix}"
    if value.type == expected:
        return True, "ok"
    fixes: list[str] = []
    if value.type.count != expected.count:
        fixes.append(
            f"the value holds {value.type.count} registers per lane but this TileDesc lays out "
            f"{expected.count}; wrap it in a new Fragment using the TileDesc that produced "
            "these registers, or, if they are one MMA atom's result, drive the whole tile "
            "through TileMma instead of raw b.mma"
        )
    if value.type.elem != dtype:
        fixes.append(
            f"the registers are {value.type.elem.name}, not {dtype.name}; build the Fragment at "
            f"{value.type.elem.name}"
        )
    return False, f"{got} {value.type.name} -- " + "; and ".join(fixes)


class Fragment:
    """Per-lane register data for a tile: the `tile_desc` that lays it out, its element `dtype`,
    and the SSA `value` holding the registers.

    `tile_desc` and `dtype` are fixed at construction (read-only); `dtype` must be a scalar IR
    `Type` (``F16``, ``F32``, ...) -- a string token or a vector / pointer / LDS type raises
    ``ValueError``; so does a `tile_desc` that is not a :class:`TileDesc`. `value` is the one
    mutable field, because a kernel re-binds it every K step (``accumulator.value = b.mma(...)`` for
    a single-atom tile -- wave shape == atom shape, one ``b.mma``; a multi-atom tile is driven
    through `TileMma`). Every write, construction included, must be a
    ``VectorType(dtype, tile_desc.register_count)``; anything else raises ``ValueError`` naming the
    expected and actual type. A too-short vector would otherwise still lower, and every register
    slice past its end would extract out-of-range vector elements as LLVM poison -- a silent wrong
    answer; a too-long one would silently drop its extra registers. A wrong element type mislabels
    every register.

    A fragment constructed without a value is unfilled: reading `value` raises ``ValueError``
    until registers are written (`fill_fragment`, or a checked ``value =`` write), and assigning
    ``None`` later is rejected. `load_fragment`, `transform_fragment`, and `TileMma` return a new,
    filled `Fragment`. Two fragments compare equal only if they are the same object.
    """

    __slots__ = ("_tile_desc", "_dtype", "_value")

    def __init__(
        self, tile_desc: TileDesc, dtype: Type, value: Value | None = None
    ) -> None:
        if not isinstance(tile_desc, TileDesc):
            raise ValueError(
                "Fragment tile_desc must be a TileDesc (from make_tile_desc, or a TileMma's "
                f"a_desc / b_desc / c_desc) -- got {type(tile_desc).__name__} {tile_desc!r}"
            )
        ok, why = _fragment_dtype_ok(dtype)
        if not ok:
            raise ValueError(why)
        self._tile_desc = tile_desc
        self._dtype = dtype
        self._value: Value | None = None
        if value is not None:
            self.value = value

    @property
    def tile_desc(self) -> TileDesc:
        """The register layout; fixed at construction."""
        return self._tile_desc

    @property
    def dtype(self) -> Type:
        """The element type; fixed at construction."""
        return self._dtype

    @property
    def value(self) -> Value:
        """The SSA ``vec<dtype x register_count>`` vector holding this lane's registers."""
        if self._value is None:
            raise ValueError(
                f"Fragment(shape={self._tile_desc.shape}, dtype={self._dtype.name}) not filled -- "
                "call fill_fragment(b, fragment, 0), or use the Fragment load_fragment returns"
            )
        return self._value

    @value.setter
    def value(self, value: Value) -> None:
        if value is None:
            expected = VectorType(self._dtype, self._tile_desc.register_count)
            raise ValueError(
                f"Fragment value cannot be None -- assign a {expected.name}, or build a new "
                "unfilled Fragment with make_fragment"
            )
        ok, why = _fragment_value_type_ok(self._tile_desc, self._dtype, value)
        if not ok:
            raise ValueError(why)
        self._value = value

    def __repr__(self) -> str:
        registers = (
            "<unfilled>"
            if self._value is None
            else f"{self._value!r}: {self._value.type.name}"
        )
        return (
            f"Fragment(shape={self._tile_desc.shape}, dtype={self._dtype.name}, "
            f"value={registers})"
        )


def make_fragment(
    tile_desc: TileDesc, dtype: Type, value: Value | None = None
) -> Fragment:
    """Free factory: a `Fragment` for `tile_desc` at element `dtype`. Without `value` it is
    unfilled until its first write (`fill_fragment`, or a checked ``value =`` write); a given
    `value` is checked like any other write (see `Fragment`). `load_fragment` builds and returns a
    filled fragment itself."""
    return Fragment(tile_desc, dtype, value)
