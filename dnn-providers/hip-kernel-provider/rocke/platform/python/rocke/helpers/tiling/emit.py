# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""IR-emitting tiling verbs -- the author surface's lowering layer.

This module is EVERYTHING that threads the IRBuilder (``b``) -- the lowering vessel. It drives
ALL addressing from rocke.helpers.tiling's own :class:`~rocke.helpers.tiling.encoding.WarpDistributionEncoding`
via an IR-emitting ``calculate_x`` (:func:`emit_tensor_coordinates`), then reads/writes memory
through the :class:`~rocke.helpers.tiling.descriptors.TensorWindow` (memory side) into a
:class:`~rocke.helpers.tiling.fragments.Fragment` (register side). It bottoms out ONLY at raw IRBuilder
ops (``global_load`` / ``global_store`` / integer arithmetic) -- it does **not** use rocke's
``mfma_gemm_inner`` loaders/k-loop/epilogue. This is the isolated new-API surface.

``b`` is the only rocke dependency here (the IRBuilder + a couple of element types). Everything
about *where each element lives* comes from our encodings; the pure value objects live in
:mod:`rocke.helpers.tiling.descriptors` and :mod:`rocke.helpers.tiling.fragments`.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any, Callable

from ...core.dtypes import dtype_info
from ...core.ir import I8, I32, U8, VectorType
from ...core.storage import BitPacking
from .descriptors import TensorDesc, TensorWindow
from .encoding import WarpDistributionEncoding
from .fragments import (
    Fragment,
    TileDesc,
    _fragment_storage_type,
    _is_packed,
    _storage_element,
)

__all__ = [
    "emit_tensor_coordinates",
    "fill_fragment",
    "lds_element_span",
    "load_fragment",
    "store_fragment",
]

# Sub-byte elements are packed into 32-bit words; one word is the smallest unit a lane accesses.
_WORD_BITS = 32


def _element_bits(dtype: Any) -> int:
    """Bits of one logical element of `dtype` (``core.dtypes.dtype_info``)."""
    return dtype_info(dtype.name).encoded_bits


def _unit_bits(dtype: Any) -> int:
    """Bits of one storage unit of `dtype` -- what one memory access moves per element of its
    vector: the fragment storage element (``i32`` for tf32 and for packed sub-byte dtypes, the dtype
    itself otherwise). Also the access's natural alignment, in bits."""
    return dtype_info(_storage_element(dtype).name).encoded_bits


def _unit_count(dtype: Any, elements: int) -> int:
    """How many storage units hold `elements` logical elements of `dtype` (whole units)."""
    return elements * _element_bits(dtype) // _unit_bits(dtype)


def _unit_index(b: Any, element_index: Any, dtype: Any) -> Any:
    """Convert a flat element index into the storage-unit index an access is addressed by,
    ``element_index * bits / unit_bits``. Emits nothing when one element is one unit (every
    byte-or-wider dtype), so their IR is unchanged. A packed index is divided by the elements of one
    whole-word group before it is multiplied by that group's words, ``(element_index / 16) * 3`` for
    a 6-bit dtype: exact, because every packed access starts on a whole word and so on a whole group
    (:func:`_check_packed_runs`), and it never overflows i32 where the element index does not, as
    multiplying first would past a third of the i32 range for a 6-bit dtype."""
    bits, unit = _element_bits(dtype), _unit_bits(dtype)
    if bits == unit:
        return element_index
    common = math.gcd(bits, unit)
    groups = b.div(element_index, b.const_i32(unit // common))
    return groups if bits == common else b.mul(groups, b.const_i32(bits // common))


def _packed_group(dtype: Any) -> tuple[int, int]:
    """The smallest whole-word group of a packed `dtype`: ``(elements, words)`` -- (8, 1) for a
    4-bit dtype, (16, 3) for a 6-bit one."""
    return BitPacking(_element_bits(dtype)).group(_WORD_BITS)


def _cast_element(b: Any, value: Any, src: Any, dst: Any) -> Any:
    """Honest, fail-fast element cast: identity, or the proven f32->{f16,bf16} accumulator
    narrowing. Any other conversion raises rather than silently doing the wrong thing.
    """
    if src.name == dst.name:
        return value
    if src.name == "f32":
        return b.cast_f32_to(value, dst)  # validates the target itself
    raise NotImplementedError(
        f"unsupported fragment cast on store -- src={src.name!r}, dst={dst.name!r}, "
        f"expected src==dst or src=='f32'"
    )


def emit_lane_contributors(
    b: Any, encoding: WarpDistributionEncoding, thread: Any
) -> dict[tuple[int, int], Any]:
    """The thread-only partition decomposition, factored out of :func:`emit_tensor_coordinates` so a
    per-register loop can emit it ONCE. It depends on ``thread`` alone (never the register index), so
    hoisting it keeps each store/load address ``base + <compile-time constant>`` -- which the LLVM
    LoadStoreVectorizer needs to see to merge consecutive scalar stores into a wide ``dwordx4``:
    re-deriving the lane ``mod``/``div`` per register instead makes each address a structurally
    distinct expression the vectorizer cannot relate.
    Decomposes across ALL partition buckets, wave partition outer -> lane inner (last fastest):
    ``thread % wave_size`` falls out as the lane, ``thread // wave_size`` as the wave.
    """
    contributor: dict[tuple[int, int], Any] = {}
    partition_buckets: list[tuple[int, int]] = []
    for majors, minors in zip(encoding.lane_to_rh_major, encoding.lane_to_rh_minor):
        partition_buckets.extend(zip(majors, minors))
    suffix = 1
    for position in reversed(range(len(partition_buckets))):
        length = encoding.bucket_length(*partition_buckets[position])
        if length == 1:
            contributor[partition_buckets[position]] = b.const_i32(0)
        else:
            divided = thread if suffix == 1 else b.div(thread, b.const_i32(suffix))
            contributor[partition_buckets[position]] = b.mod(
                divided, b.const_i32(length)
            )
        suffix *= length
    return contributor


def emit_coordinates_for_register(
    b: Any,
    encoding: WarpDistributionEncoding,
    lane_contributors: dict[tuple[int, int], Any],
    register_index: int,
) -> tuple[Any, ...]:
    """Place the compile-time ``register_index`` and reconstruct each X coordinate innermost-stride-1,
    reusing the hoisted ``lane_contributors`` (:func:`emit_lane_contributors`). The lane terms are
    SHARED SSA across registers; only the register terms (compile-time constants) vary -- so
    consecutive registers' coordinates, and thus addresses, differ by a constant. Returns ONE SSA
    coordinate per X-dim (2 for an MMA fragment, N for a data tile).
    """
    register_buckets = list(
        zip(encoding.register_to_rh_major, encoding.register_to_rh_minor)
    )
    register_lengths = [encoding.bucket_length(*bucket) for bucket in register_buckets]

    contributor = dict(lane_contributors)  # copy -- never mutate the hoisted lane map
    remainder = register_index
    register_values = [0] * len(register_lengths)
    for position in reversed(range(len(register_lengths))):
        register_values[position] = remainder % register_lengths[position]
        remainder //= register_lengths[position]
    for bucket, value in zip(register_buckets, register_values):
        contributor[bucket] = b.const_i32(value)

    coordinates: list[Any] = []
    for x_dim, levels in enumerate(encoding.hierarchical_lengths):
        coordinate: Any | None = None
        stride = 1
        for level in reversed(range(len(levels))):
            source = contributor.get((x_dim + 1, level))
            if source is not None:
                term = source if stride == 1 else b.mul(source, b.const_i32(stride))
                coordinate = term if coordinate is None else b.add(coordinate, term)
            stride *= levels[level]
        coordinates.append(coordinate if coordinate is not None else b.const_i32(0))
    return tuple(coordinates)


def emit_tensor_coordinates(
    b: Any, encoding: WarpDistributionEncoding, thread: Any, register_index: int
) -> tuple[Any, ...]:
    """IR-emitting ``calculate_x``: (runtime ``thread`` id SSA, compile-time register) -> coords.

    Thin composition of :func:`emit_lane_contributors` (the thread-only decomposition) + a single
    :func:`emit_coordinates_for_register`. A loop over registers should instead call
    ``emit_lane_contributors`` ONCE and ``emit_coordinates_for_register`` per register, so the lane
    ``mod``/``div`` is emitted once and the addresses stay ``base + <constant>`` (which the store
    vectorizer needs to widen a strided global store). Single-wave (NDimP=1) and block-wide ``wave_dist``
    (NDimP=2) are the same code, driven by ``thread`` alone. Returns ONE SSA coordinate per X-dim.
    """
    return emit_coordinates_for_register(
        b, encoding, emit_lane_contributors(b, encoding, thread), register_index
    )


def _refuse_u8_fragment(verb: str, dtype: Any) -> None:
    """Refuse a u8 fragment in a verb that builds the register vector from a zero vector of the
    storage type: core's ``zero_vec`` has no u8 form. The i8 zero vector reinterpreted is the same
    bits, so the message names that route.
    """
    if dtype == U8:
        raise NotImplementedError(
            f"{verb} cannot build a u8 fragment: core's zero_vec has no u8 form -- TODO: not "
            f"implemented yet; build an I8 fragment with {verb} and b.bitcast its value to "
            f"vec<u8 x n>"
        )


def _move_registers(
    b: Any, vector: Any, moves: Sequence[tuple[int, int]], count: int
) -> Any:
    """A new ``count``-element register vector of ``vector``'s element type holding, for each
    ``(src, dst)`` of `moves` in order, element ``src`` of `vector` at element ``dst``: a
    compile-time register shuffle (``vec_extract`` / ``vec_insert`` into a zero vector). Elements no
    move writes stay zero.

    A u8 vector is moved as the i8 vector of the same bits, since core's ``zero_vec`` has no u8
    form, and the result reinterpreted back to u8.
    """
    elem = vector.type.elem
    if elem == U8:
        as_i8 = b.bitcast(vector, VectorType(I8, vector.type.count))
        return b.bitcast(_move_registers(b, as_i8, moves, count), VectorType(U8, count))
    out = b.zero_vec(elem, count)
    for src, dst in moves:
        out = b.vec_insert(out, b.vec_extract(vector, src), dst)
    return out


def fill_fragment(b: Any, fragment: Fragment, scalar: Any) -> None:
    """Set every register of `fragment` to `scalar`, element-wise (no layout, no addressing). A zero
    fill is the zero of the fragment's storage type, so a tf32 or packed fragment is all-zero words.

    Only a compile-time zero fill is implemented today. A nonzero or runtime `scalar` is REJECTED
    rather than silently zero-filled, so a caller asking for ones gets a loud failure instead of a
    wrong fragment. A u8 fragment is refused (:func:`_refuse_u8_fragment`).
    """
    _refuse_u8_fragment("fill_fragment", fragment.dtype)
    if not (isinstance(scalar, int) and scalar == 0):
        raise NotImplementedError(
            f"fill_fragment supports only a compile-time zero fill -- got scalar={scalar!r}; "
            "nonzero/runtime scalar splat is not implemented"
        )
    storage = fragment.storage_type
    fragment.value = b.zero_vec(storage.elem, storage.count)


def _as_value(b: Any, x: Any) -> Any:
    """An int becomes a const_i32; an SSA value passes through unchanged."""
    return b.const_i32(x) if isinstance(x, int) else x


def _leading_rank(window: TensorWindow, tile_rank: int) -> int:
    """How many leading tensor axes a rank-``tile_rank`` tile leaves to the window origin. A tile
    covers the window's LAST axes, so tile axis ``i`` is tensor axis ``_leading_rank + i``.
    """
    return window.tensor.rank - tile_rank


def _check_tile_rank(verb: str, window: TensorWindow, tile_desc: TileDesc) -> None:
    """Fail fast unless the tile fits the window: a rank-t tile covers the window's last t axes."""
    tile_rank, window_rank = len(tile_desc.shape), window.tensor.rank
    if tile_rank > window_rank:
        raise ValueError(
            f"{verb}: the tile has rank {tile_rank} but the window has rank {window_rank} -- a "
            f"rank-t tile covers the window's last t axes, so t <= r"
        )


def _check_leading_bounds(verb: str, window: TensorWindow, tile_desc: TileDesc) -> None:
    """Fail fast on a clip bound for a leading axis. A leading axis has tile extent 1, so a tile is
    wholly inside or wholly outside any bound on it; the clip only masks a tile that partly overhangs,
    and keeping a tile off positions wholly outside is the caller's job on every axis. A bound there
    would mask nothing, so it is refused rather than ignored."""
    if window.bounds is None:
        return
    lead = _leading_rank(window, len(tile_desc.shape))
    clipped = [axis for axis in range(lead) if window.bounds[axis] is not None]
    if clipped:
        raise ValueError(
            f"{verb}: the window bounds axis {clipped[0]}, a leading axis of this rank-"
            f"{len(tile_desc.shape)} tile -- a leading axis is placed by its origin and is never "
            f"clipped; pass None for it and keep the origin inside the tensor (guard the index "
            f"before the access if it can fall outside)"
        )


def _positions(b: Any, window: TensorWindow, coords: tuple[Any, ...]) -> list[Any]:
    """Per-tile-axis GLOBAL position = origin + coord (the basis of both the address and the clip).
    Tile axis ``i`` lies on tensor axis ``_leading_rank + i``, so it takes that axis's origin.
    """
    lead = _leading_rank(window, len(coords))
    return [
        b.add(_as_value(b, window.origin[lead + axis]), coord)
        for axis, coord in enumerate(coords)
    ]


def _address(
    b: Any,
    window: TensorWindow,
    positions: list[Any],
    leading_positions: list[Any] | None = None,
) -> Any:
    """The strided element index from the tile axes' precomputed positions: ``sum(position *
    stride)`` over the tile axes in axis order, then ``position * stride`` of each leading axis in
    axis order. A leading axis's position is its origin (the tile covers one index of it) unless
    ``leading_positions`` gives it (a swizzle may move it, see :func:`_lds_index`).
    """
    lead = _leading_rank(window, len(positions))
    if leading_positions is None:
        leading_positions = [_as_value(b, window.origin[axis]) for axis in range(lead)]
    strides = window.tensor.strides
    address: Any = None
    for axis, position in enumerate(positions):
        stride = strides[lead + axis]
        term = position if stride == 1 else b.mul(position, b.const_i32(stride))
        address = term if address is None else b.add(address, term)
    for axis, position in enumerate(leading_positions):
        stride = strides[axis]
        term = position if stride == 1 else b.mul(position, b.const_i32(stride))
        address = term if address is None else b.add(address, term)
    return address


def _lds_storage_element(dtype: Any) -> Any:
    """The element type of an LDS buffer holding a `dtype` tensor: ``i32`` words for a packed
    sub-byte dtype, the tensor dtype itself otherwise."""
    return I32 if _is_packed(dtype) else dtype


def _check_lds_buffer(
    verb: str,
    buffer: Any,
    tensor: TensorDesc,
    lds_swizzle: bool | Callable[[Any, list], list],
) -> None:
    """Fail fast unless ``buffer`` can hold ``tensor``: an LDS buffer is flat (rank 1), holds the
    tensor's LDS storage element (:func:`_lds_storage_element`), and holds the storage units of at
    least :func:`lds_element_span` elements (the descriptor's span, or the built-in swizzle's reach).
    The descriptor is the one home of the buffer's layout; the allocation only gives it room, so a
    buffer that disagrees with it is refused here rather than read or written out of range (LDS
    accesses are unclipped). Window origins are trusted, as for every LDS access. A packed tensor is
    addressed in whole words, which no swizzle keeps together, so a swizzle is refused for it.
    """
    buffer_type = buffer.type
    dtype = tensor.dtype
    element = _lds_storage_element(dtype)
    span = lds_element_span(tensor, lds_swizzle)
    units = -(-span * _element_bits(dtype) // _element_bits(element))
    if _is_packed(dtype) and lds_swizzle:
        raise NotImplementedError(
            f"{verb}: an LDS swizzle of packed {dtype.name} data -- a swizzle moves elements, and "
            f"{dtype.name} elements are addressed in whole 32-bit words -- TODO: not implemented yet"
        )
    if len(buffer_type.shape) != 1:
        raise ValueError(
            f"{verb}: the LDS buffer {buffer_type.name} has rank {len(buffer_type.shape)} -- an LDS "
            f"buffer is flat; its layout lives in the tensor descriptor. Allocate "
            f"smem_alloc({element.name}, [{units}])"
        )
    if buffer_type.elem.name != element.name:
        if _is_packed(dtype):
            raise ValueError(
                f"LDS buffer for {dtype.name} fragments must hold its storage words -- allocate "
                f"smem_alloc(I32, [words]) sized from the descriptor; got {buffer_type.name}"
            )
        raise ValueError(
            f"{verb}: LDS buffer element {buffer_type.elem.name} != tensor dtype {dtype.name}"
        )
    if units > buffer_type.shape[0]:
        if _is_packed(dtype):
            raise ValueError(
                f"{verb}: the LDS tensor spans {span} {dtype.name} elements, {units} i32 words, "
                f"but the buffer holds {buffer_type.shape[0]} -- size the allocation as "
                f"ceil(lds_element_span * {_element_bits(dtype)} / 32) words"
            )
        raise ValueError(
            f"{verb}: the LDS tensor spans {span} elements but the buffer holds "
            f"{buffer_type.shape[0]} -- size the allocation with lds_element_span"
        )


def _register_coord_offsets(
    encoding: WarpDistributionEncoding, register_index: int
) -> tuple[int, ...]:
    """The register's COMPILE-TIME contribution to each X coordinate (lane buckets contribute 0) --
    the same value :func:`emit_coordinates_for_register` places, computed in pure Python. Lets the
    global store emit ``address = base + <constant>``: the lane base-address is computed ONCE and this
    fixed per-register offset (linearized by the tensor strides) is added, so consecutive registers'
    addresses differ by a constant and the store vectorizer merges them into ``dwordx4``. It is
    bit-exact with the per-register reconstruction by linearity of ``_address``."""
    register_buckets = list(
        zip(encoding.register_to_rh_major, encoding.register_to_rh_minor)
    )
    register_lengths = [encoding.bucket_length(*bucket) for bucket in register_buckets]
    remainder = register_index
    register_values = [0] * len(register_lengths)
    for position in reversed(range(len(register_lengths))):
        register_values[position] = remainder % register_lengths[position]
        remainder //= register_lengths[position]
    reg_const = dict(zip(register_buckets, register_values))
    offsets: list[int] = []
    for x_dim, levels in enumerate(encoding.hierarchical_lengths):
        coord = 0
        stride = 1
        for level in reversed(range(len(levels))):
            value = reg_const.get((x_dim + 1, level))
            if value is not None:
                coord += value * stride
            stride *= levels[level]
        offsets.append(coord)
    return tuple(offsets)


def _lane_base_index(
    b: Any, window: TensorWindow, tile_desc: TileDesc, thread: Any
) -> tuple[list[Any], Any]:
    """Register 0's positions and element index -- the lane-only part of every register's address,
    emitted ONCE per access. Returns ``(positions, index)``: the positions feed the per-register
    positions (for the clip) and the index feeds :func:`_register_index`."""
    coords = emit_coordinates_for_register(
        b, tile_desc.layout, emit_lane_contributors(b, tile_desc.layout, thread), 0
    )
    positions = _positions(b, window, coords)
    return positions, _address(b, window, positions)


def _register_positions(
    b: Any, tile_desc: TileDesc, base_positions: list[Any], register: int
) -> list[Any]:
    """The register's per-axis positions: the lane base positions plus the register's compile-time
    coordinate offset on each axis (an axis with no offset reuses the base value)."""
    deltas = _register_coord_offsets(tile_desc.layout, register)
    return [
        position if delta == 0 else b.add(position, b.const_i32(delta))
        for position, delta in zip(base_positions, deltas, strict=True)
    ]


def _register_index(
    b: Any,
    window: TensorWindow,
    tile_desc: TileDesc,
    base_index: Any,
    register: int,
    dtype: Any,
) -> Any:
    """The register's storage-unit index: the lane base index (already in storage units, see
    :func:`_unit_index`) plus the register's compile-time coordinate offset linearized by the tensor
    strides and converted to units. Consecutive registers' indices then differ by a constant, which
    the backend vectorizer can merge; it is bit-exact with :func:`_address` of the register's
    positions by linearity. For every byte-or-wider dtype a unit is an element."""
    deltas = _register_coord_offsets(tile_desc.layout, register)
    lead = _leading_rank(window, len(deltas))
    offset = _unit_count(
        dtype,
        sum(
            delta * window.tensor.strides[lead + axis]
            for axis, delta in enumerate(deltas)
        ),
    )
    return base_index if offset == 0 else b.add(base_index, b.const_i32(offset))


def _origin_misaligned(origin: Any, extent: int) -> bool:
    """True when `origin` is a COMPILE-TIME int NOT on the tile grid (`origin % extent != 0`): a tile
    placed there straddles a tile boundary and can overhang the tensor's far edge even when the extent
    is a tile multiple. A runtime (SSA) origin is TRUSTED grid-aligned -- every current caller positions
    tiles at ``block_id * tile``. PARTIAL GUARD: an SSA mid-tile origin (attention's sliding window) is
    NOT caught here; full mid-tile-origin handling is DEFERRED with attention support.
    """
    return isinstance(origin, int) and origin % extent != 0


def _clip_mask(
    b: Any, window: TensorWindow, positions: list[Any], tile_shape: tuple[int, ...]
) -> Any:
    """In-bounds predicate (i1) for an element at the tile axes' `positions`, or `None` when no axis
    needs a compare. The per-axis upper bound defaults to ``window.tensor.lengths`` and is overridden
    by ``window.bounds`` (a `None` entry keeps the length). Predicate = AND over checked axes of
    ``position < bound``. A tile-aligned compile-time bound AT a grid-aligned origin can NEVER overhang,
    so it is SKIPPED at build time -- an aligned kernel emits no compare and stays byte-identical to
    no-clip; a compile-time MID-TILE origin is NOT skipped (it could straddle the boundary). Only the
    tile axes are checked: a leading axis (tile extent 1, positioned by its origin) is not clipped.
    """
    lead = _leading_rank(window, len(positions))
    mask: Any = None
    for tile_axis, position in enumerate(positions):
        axis = lead + tile_axis
        bound = window.bounds[axis] if window.bounds is not None else None
        if bound is None:
            bound = window.tensor.lengths[axis]
        extent = tile_shape[tile_axis]
        if (
            isinstance(bound, int)
            and bound % extent == 0
            and not _origin_misaligned(window.origin[axis], extent)
        ):
            continue
        in_axis = b.cmp_lt(position, _as_value(b, bound))
        mask = in_axis if mask is None else b.land(mask, in_axis)
    return mask


def _zero_scalar(b: Any, dtype: Any) -> Any:
    """A scalar 0 of `dtype`'s storage unit -- the zero-pad value handed to ``masked_global_load``."""
    return b.vec_extract(b.zero_vec(_storage_element(dtype), 1), 0)


# Storage elements moved with vector accesses: global loads, LDS loads and LDS stores
# (global_load_vN, smem_load_vN, smem_store_vN). tf32 is held and moved as i32.
_WIDE_OK = {
    "f16",
    "bf16",
    "f32",
    "i32",
    "fp8e4m3",
    "bf8e5m2",
    "i8",
}


def _register_run(tile_desc: TileDesc, window: TensorWindow) -> tuple[int, int] | None:
    """``(tensor axis, length)`` of the innermost register bucket when its registers are consecutive
    elements in memory, else ``None``.

    The innermost register bucket varies fastest in the register index. Its registers are
    consecutive positions on their axis only when the bucket is that axis's INNERMOST hierarchical
    level (levels of length 1 inside it do not count). A bucket on an outer level steps the position
    by the product of the levels inside it -- an interleaved layout holds positions 0, 4, 8, 12 in
    four consecutive registers -- so it gives no run. When the bucket is the innermost level of the
    tensor's stride-1 axis, its registers are memory-contiguous. Because both spaces address an
    element by ``sum(position * stride)``, that holds in LDS exactly as in global memory.
    """
    layout = tile_desc.layout
    majors, minors = layout.register_to_rh_major, layout.register_to_rh_minor
    if not majors:
        return None
    lead = _leading_rank(window, len(tile_desc.shape))
    tile_axis = majors[-1] - 1  # innermost register bucket's X-dim -> tile axis
    run_axis = lead + tile_axis
    if tile_axis < 0 or window.tensor.strides[run_axis] != 1:
        return None
    inner_levels = layout.hierarchical_lengths[tile_axis][minors[-1] + 1 :]
    if math.prod(inner_levels) != 1:
        return None  # an outer level of its axis: consecutive registers are not consecutive positions
    return run_axis, layout.bucket_length(majors[-1], minors[-1])


def _tile_in_bounds(tile_desc: TileDesc, window: TensorWindow) -> bool:
    """True when no tile of this window can overhang an axis it spans, so a GLOBAL access needs no
    mask: every tile axis's effective clip (``window.bounds[axis]`` or the tensor length) is a
    COMPILE-TIME multiple of the tile extent, and the tile origin is on the tile grid.

    Not just the stride-1 run axis: a wide K-run at a fixed but OOB M row (odd M, aligned K) is just as
    out of bounds as an overhanging run. A COMPILE-TIME mid-tile origin is caught (it could straddle
    the boundary and overhang); a runtime (SSA) origin is TRUSTED grid-aligned (every caller:
    block_id*tile). This is a PARTIAL guard -- an SSA mid-tile origin (attention) is not caught; that
    is DEFERRED with attention support (see _origin_misaligned). Only the tile axes are checked: a
    leading axis is positioned by its origin and not clipped, exactly as in _clip_mask.
    """
    lead = _leading_rank(window, len(tile_desc.shape))
    for tile_axis, extent in enumerate(tile_desc.shape):
        ax = lead + tile_axis
        clip = (
            window.bounds[ax]
            if window.bounds is not None and window.bounds[ax] is not None
            else window.tensor.lengths[ax]
        )
        if not (isinstance(clip, int) and clip % extent == 0):
            return False
        if _origin_misaligned(window.origin[ax], extent):
            return False
    return True


def _check_packed_runs(
    verb: str, tile_desc: TileDesc, window: TensorWindow, dtype: Any, is_lds: bool
) -> int:
    """Fail fast unless every access of a packed `dtype` moves whole 32-bit words, and return the
    run length in elements.

    A lane cannot write part of a word another lane owns, so each lane's registers must form runs of
    whole words along the tensor's stride-1 axis (:func:`_register_run`; a multiple of the dtype's
    whole-word group, 8 elements for 4 bits, 16 for 6 bits), and each run must start on a group
    boundary: every other stride and a compile-time origin on the run axis are multiples of the
    group. A clipped GLOBAL window masks whole groups, so its compile-time bound along the stride-1
    axis must be a multiple of the group too (a runtime bound is trusted aligned, as runtime origins
    are). LDS accesses are unclipped."""
    group, _ = _packed_group(dtype)
    run = _register_run(tile_desc, window)
    length = 1 if run is None else run[1]
    # "an 8-element boundary", "a 16-element boundary"
    article = "an" if str(group).startswith("8") else "a"
    need = (
        f"{verb} of {dtype.name} needs each lane's registers to form runs of whole 32-bit words "
        f"along the tensor's stride-1 axis (multiples of {group} elements, starting on {article} "
        f"{group}-element boundary)"
    )
    if run is None or length % group:
        raise ValueError(
            f"{need}; this layout gives runs of {length} "
            f"{'element' if length == 1 else 'elements'} -- choose a layout whose innermost "
            "register run lies along the stride-1 axis"
        )
    run_axis = run[0]
    for axis, stride in enumerate(window.tensor.strides):
        if axis != run_axis and stride % group:
            raise ValueError(
                f"{need}; tensor axis {axis} steps by {stride} elements, so its runs start off "
                f"that boundary -- pad that stride to a multiple of {group}"
            )
    origin = window.origin[run_axis]
    if isinstance(origin, int) and origin % group:
        raise ValueError(
            f"{need}; the window origin on the stride-1 axis is {origin} -- place the window on a "
            f"multiple of {group}"
        )
    if not is_lds:
        bound = (
            window.bounds[run_axis]
            if window.bounds is not None and window.bounds[run_axis] is not None
            else window.tensor.lengths[run_axis]
        )
        if isinstance(bound, int) and bound % group:
            raise ValueError(
                f"{verb} of a clipped {dtype.name} window needs its bound along the stride-1 axis "
                f"to be a multiple of {group} elements (whole 32-bit words) -- got {bound}"
            )
    return length


def _contiguous_run(
    tile_desc: TileDesc,
    window: TensorWindow,
    dtype: Any,
    is_lds: bool,
    *,
    verb: str = "load_fragment / store_fragment",
) -> int:
    """Storage units one access moves: the widest vector over the innermost stride-1 register run
    (:func:`_register_run`) -> a single WIDE load; else 1 (one element). `dtype` is the logical
    dtype; the width is counted in its storage units (:func:`_unit_bits`), which are its elements for
    every byte-or-wider dtype. Width is the largest supported vector size (16/8/4/2) of at most 16
    bytes dividing the run.

    A wide GLOBAL load is UNMASKED, so it is emitted only for a tile that is FULLY IN BOUNDS
    (:func:`_tile_in_bounds`). A ragged/odd tensor, or a runtime clip that can't be proven aligned,
    falls to the SCALAR path (which masks against the length) -- never an out-of-bounds wide read (the
    page-fault risk Principle 4 kills). LDS is EXEMPT: the verbs check that the flat buffer holds the
    descriptor's whole span (:func:`_check_lds_buffer`) and LDS accesses are unclipped, so the gate
    applies to global only.

    The width must also divide every OTHER stride, leading axes included: a run then starts at a
    multiple of the width whatever the positions on the other axes, so the ``align`` the access claims
    holds (an odd batch stride would otherwise start a wide run mid-vector).

    A packed sub-byte dtype has no one-element access: its runs must be whole words
    (:func:`_check_packed_runs`, which raises with `verb` otherwise). A global tile that may overhang
    moves one whole-word group per access, masked as a unit; any other packed access is the widest
    whole-group vector of words dividing the run."""
    if _is_packed(dtype):
        return _packed_access_units(verb, tile_desc, window, dtype, is_lds)
    if _storage_element(dtype).name not in _WIDE_OK:
        return 1
    run = _register_run(tile_desc, window)
    if run is None:
        return 1
    run_axis, length = run
    if not is_lds and not _tile_in_bounds(tile_desc, window):
        return 1
    other_strides = [
        stride for axis, stride in enumerate(window.tensor.strides) if axis != run_axis
    ]
    # One load = dwordx4 = 16 bytes; longer runs -> N loads.
    max_vw = 128 // _unit_bits(dtype)
    for vw in (16, 8, 4, 2):
        if (
            vw <= max_vw
            and length % vw == 0
            and all(stride % vw == 0 for stride in other_strides)
        ):
            return vw
    return 1


def _packed_access_units(
    verb: str, tile_desc: TileDesc, window: TensorWindow, dtype: Any, is_lds: bool
) -> int:
    """The words one access of a packed `dtype` moves (see :func:`_contiguous_run`): one whole-word
    group for a global tile that may overhang (masked as a unit), else the widest whole-group count
    of at most 4 words (16 bytes) that divides the run and keeps every run start on a multiple of
    itself."""
    length = _check_packed_runs(verb, tile_desc, window, dtype, is_lds)
    _, group_words = _packed_group(dtype)
    if not is_lds and not _tile_in_bounds(tile_desc, window):
        return group_words
    run_axis, _ = _register_run(tile_desc, window)
    bits = _element_bits(dtype)
    run_words = _unit_count(dtype, length)
    other_strides = [
        stride for axis, stride in enumerate(window.tensor.strides) if axis != run_axis
    ]
    for words in (4, 3, 2, 1):
        if (
            words % group_words == 0
            and run_words % words == 0
            and all(
                stride * bits % (words * _WORD_BITS) == 0 for stride in other_strides
            )
        ):
            return words
    raise AssertionError("a whole-word group always divides a checked packed run")


def _access_align(units: int, dtype: Any) -> int:
    """The alignment (bytes) an access of `units` storage units claims: the access size when it is a
    power of two, else the largest power of two dividing it (a 3-word access claims one word).
    """
    return (_unit_bits(dtype) // 8) * (units & -units)


def _is_lds(source: Any) -> bool:
    """True when `source` is an LDS buffer (an ``smem<...>`` value from ``smem_alloc``) rather
    than a global/constant ``ptr<...>``. Lets one verb serve both spaces -- the load/store op is
    chosen from the SOURCE, so the caller just hands the buffer (global ptr OR LDS smem) it has.
    """
    name = getattr(getattr(source, "type", None), "name", "")
    return isinstance(name, str) and name.startswith("smem")


def _swizzle_lds_positions(b: Any, positions: list) -> list:
    """XOR-swizzle the last LDS position by the one before it, at b128-block (8-f16) granularity:
    ``m -> ((m>>3) ^ (outer & 7)) << 3 | (m & 7)``. Keeps every 8-f16 (b128) run contiguous while
    de-aliasing LDS banks across the outer (K) dim -> conflict-free WITHOUT padding (which breaks
    alignment). A bijection of the physical ``(outer, m)``, so applying it identically to the store and
    the read leaves correctness intact regardless of the two fragments' layouts. Requires each access to
    be naturally vw-aligned within an 8-f16 block (true for our b128 store / b64 read), and the last
    axis to be the buffer's stride-1 axis (checked by :func:`_check_builtin_swizzle`): the swizzled
    positions are then linearised by the descriptor strides like any other LDS position.
    """
    if len(positions) < 2:
        return positions
    outer, m = positions[-2], positions[-1]
    # m ^ ((outer & 7) << 3): flips m's block bits [3:6] by (K&7), leaving the low-3 (8-f16 b128 run)
    # intact -> identical bank mapping to a block XOR, in 3 VALU ops (mod, shl, xor) not 6.
    m_swz = b.xor(m, b.shl(b.mod(outer, b.const_i32(8)), b.const_i32(3)))
    return positions[:-1] + [m_swz]


# The built-in swizzle flips bits [3:6) of the last position (8-element blocks, 8 of them), so an
# element moves only within its aligned group of 64 elements.
_SWIZZLE_GROUP = 64


def _swizzled_row_length(tensor: TensorDesc) -> int:
    """How far a row of the last axis reaches under the built-in swizzle: its length rounded up to
    the swizzle's 64-element group. A row of 48 f16 reaches element 63 of its group::

        position (outer = 7)   0..7    8..15   ...  40..47
        swizzled               56..63  48..55  ...  16..23     <- 56..63 lies past the 48
    """
    length = tensor.lengths[-1]
    return -(-length // _SWIZZLE_GROUP) * _SWIZZLE_GROUP


def lds_element_span(
    tensor: TensorDesc, lds_swizzle: bool | Callable[[Any, list], list] = False
) -> int:
    """The elements an LDS buffer for ``tensor`` needs:
    ``b.smem_alloc(td.dtype, [lds_element_span(td, lds_swizzle)])``, with the ``lds_swizzle`` the
    verbs are given.

    Without a swizzle it is ``tensor.element_span``. The built-in swizzle (``lds_swizzle=True``)
    moves an element of the last axis within its 64-element group, so the last row reaches the end
    of its group: the span counts the last axis at :func:`_swizzled_row_length`. A row of length
    48 with a padded stride of 64, 8 rows, spans 496 elements but needs 512 swizzled. A swizzle
    callable must keep every position inside the descriptor's lengths; its span is the
    descriptor's.
    """
    if lds_swizzle is not True:
        return tensor.element_span
    lengths = tensor.lengths[:-1] + (_swizzled_row_length(tensor),)
    return TensorDesc(lengths, tensor.strides, tensor.dtype).element_span


def _check_builtin_swizzle(tensor: TensorDesc) -> None:
    """The built-in swizzle (``lds_swizzle=True``) permutes the LAST position inside 8-element
    blocks, which keeps each block contiguous only when the last axis is the buffer's stride-1 axis,
    and moves it within a 64-element group, so every other axis must step past a whole swizzled
    row (:func:`_swizzled_row_length`) or two rows would share elements. Refuse any other layout
    instead of scattering a vector run or overwriting a neighbouring row; a swizzle callable states
    its own rule."""
    strides = tensor.strides
    if strides[-1] != 1:
        where = (
            f"put it at axis {strides.index(1)}"
            if 1 in strides
            else "have no stride-1 axis"
        )
        raise ValueError(
            f"lds_swizzle=True needs the stride-1 axis last -- strides {tuple(strides)} {where}; "
            f"describe the buffer with its contiguous axis last or pass a swizzle callable"
        )
    row = _swizzled_row_length(tensor)
    for axis, (length, stride) in enumerate(zip(tensor.lengths[:-1], strides[:-1])):
        if length > 1 and stride < row:
            raise ValueError(
                f"lds_swizzle=True moves elements of the last axis within groups of "
                f"{_SWIZZLE_GROUP}, so a row of {tensor.lengths[-1]} reaches {row} elements, but "
                f"axis {axis} steps by {stride} -- rows would overlap; pad that stride to at least "
                f"{row} or pass a swizzle callable"
            )


def _lds_index(
    b: Any,
    window: TensorWindow,
    tile_desc: TileDesc,
    thread: Any,
    register: int,
    lds_swizzle: bool | Callable[[Any, list], list],
    base_index: Any,
    dtype: Any,
) -> Any:
    """The flat LDS storage-unit index of `register`'s run start: ``sum(position * stride)`` over
    the window's descriptor -- the same address rule as global memory -- in units of `dtype`'s LDS
    storage element (:func:`_unit_index`; the element index for every byte-or-wider dtype).

    Without a swizzle it is the lane base index (``base_index``, from :func:`_lane_base_index` in
    storage units, emitted once per access) plus the register's compile-time offset: consecutive runs
    differ by a constant, which the backend folds into the instruction offset and pairs into
    ``ds_read2`` / ``ds_write2``. A swizzle is not linear, so a swizzled run rebuilds its positions,
    swizzles them and linearises the result (``base_index`` is unused then).

    The swizzle sees the position on EVERY tensor axis, leading axes (their origins) first, so an
    element's swizzled address does not depend on the rank of the tile that reaches it.
    """
    if not lds_swizzle:
        return _register_index(b, window, tile_desc, base_index, register, dtype)
    coords = emit_tensor_coordinates(b, tile_desc.layout, thread, register)
    lead = _leading_rank(window, len(coords))
    leading = [_as_value(b, window.origin[axis]) for axis in range(lead)]
    swizzle = _swizzle_lds_positions if lds_swizzle is True else lds_swizzle
    swizzled = swizzle(b, leading + _positions(b, window, coords))
    return _unit_index(b, _address(b, window, swizzled[lead:], swizzled[:lead]), dtype)


def _checked_lds_base_index(
    verb: str,
    b: Any,
    buffer: Any,
    window: TensorWindow,
    tile_desc: TileDesc,
    thread: Any,
    lds_swizzle: bool | Callable[[Any, list], list],
) -> Any:
    """Check the LDS buffer (and the built-in swizzle's layout rule), then return the lane base
    index, in storage units, the unswizzled runs add their offsets to (``None`` when a swizzle is
    active).
    """
    if lds_swizzle is True:
        _check_builtin_swizzle(window.tensor)
    _check_lds_buffer(verb, buffer, window.tensor, lds_swizzle)
    if lds_swizzle:
        return None
    base_index = _lane_base_index(b, window, tile_desc, thread)[1]
    return _unit_index(b, base_index, window.tensor.dtype)


def _swizzle_vw(lds_swizzle: Any, dist_vw: int, align: int) -> int:
    """Resolve the LDS access width when a swizzle is active, with a fail-fast range check.

    A callable swizzle relocates whole units of its granularity, so the access must not span wider than
    that unit. The callable declares its granularity via a ``vw_elems`` attribute (default = one dword);
    the built-in bool swizzle preserves the natural run. The chosen width is checked to satisfy
    ``1 <= vw <= dist_vw`` (the distribution's natural contiguous run) -- wider is unrepresentable.
    """
    if not callable(lds_swizzle):
        return dist_vw
    want = getattr(lds_swizzle, "vw_elems", max(1, 4 // align))
    if want < 1:
        raise ValueError(f"lds_swizzle vw_elems={want} invalid -- must be >= 1")
    # Cap at the distribution's natural run: the access can never be wider than the layout is
    # contiguous (a scalar-run distribution stays scalar; the swizzle then applies per element).
    # Result is guaranteed in [1, dist_vw].
    return min(want, dist_vw)


def load_fragment(
    b: Any,
    ptr: Any,
    window: TensorWindow,
    tile_desc: TileDesc,
    thread: Any,
    *,
    pad: Any = 0,
    coherency: int = 0,
    lds_swizzle: bool | Callable[[Any, list], list] = False,
) -> Fragment:
    """Generic per-lane load: memory (`ptr` + `window`) -> `Fragment`, driven by `tile_desc.layout`.

    ONE verb for A, B, C (and later index/scale) -- the operand difference is the window's
    TensorDesc (strides) + the tile_desc. `ptr` is the SOURCE the fragment is loaded from: a global
    ``ptr<...>`` (VMEM) OR an LDS ``smem<...>`` buffer from ``smem_alloc`` -- the load op is chosen
    from the source (:func:`_is_lds`), so the same verb stages global<->register and LDS<->register.
    The fragment's dtype IS ``window.tensor.dtype`` (nothing is baked in); its registers are read as
    that dtype's storage units (:attr:`Fragment.storage_type`: i32 for tf32, whole i32 words for a
    packed sub-byte dtype, whose runs must be whole words). Where the window clips (an
    element past its bound), OOB elements are filled with `pad` (`masked_global_load`); otherwise the
    plain `global_load` fast path (byte-identical to no-clip). In both spaces an element's index is
    ``sum(position * stride)`` over the window's descriptor. An LDS buffer is flat
    (``b.smem_alloc(td.dtype, [lds_element_span(td, lds_swizzle)])``) and the descriptor is its
    layout: the buffer must be rank 1, hold the tensor's dtype (i32 words for a packed dtype) and
    hold that span (refused otherwise); LDS loads are unclipped.
    `pad` default `0`; a non-zero `constant(value)` fill is reserved. `coherency` reserved.

    A rank-t tile covers the window's last t axes (t <= the window's rank, refused otherwise); each
    leading axis is read at its origin. A u8 tensor is refused (:func:`_refuse_u8_fragment`).
    """
    _refuse_u8_fragment("load_fragment", window.tensor.dtype)
    _check_tile_rank("load_fragment", window, tile_desc)
    _check_leading_bounds("load_fragment", window, tile_desc)
    dtype = window.tensor.dtype
    lds = _is_lds(ptr)
    # Storage units per access (>1 -> one wide load per run) and the registers they hold.
    vw = _contiguous_run(tile_desc, window, dtype, lds, verb="load_fragment")
    # A custom swizzle relocates whole units of its granularity (vw_elems), capped + range-checked to
    # [1, natural run]. The built-in bool swizzle preserves its run.
    if lds:
        vw = _swizzle_vw(lds_swizzle, vw, _unit_bits(dtype) // 8)
    step = vw * _unit_bits(dtype) // _element_bits(dtype)
    storage = _fragment_storage_type(dtype, tile_desc.register_count)
    unit = storage.elem
    value = b.zero_vec(unit, storage.count)
    if lds:
        # LDS source: one flat storage-unit index per run (wide when contiguous), lane base once.
        base_index = _checked_lds_base_index(
            "load_fragment", b, ptr, window, tile_desc, thread, lds_swizzle
        )
        for register in range(0, tile_desc.register_count, step):
            index = _lds_index(
                b, window, tile_desc, thread, register, lds_swizzle, base_index, dtype
            )
            loaded = b.smem_load_vN(ptr, index, dtype=unit, n=vw)
            slot = _unit_count(dtype, register)
            for i in range(vw):
                value = b.vec_insert(value, b.vec_extract(loaded, i), slot + i)
        return Fragment(tile_desc, dtype, value)
    align = _unit_bits(dtype) // 8
    zero: Any = None  # the pad scalar, emitted lazily on the first clipped access
    for register in range(0, tile_desc.register_count, step):
        coords = emit_tensor_coordinates(b, tile_desc.layout, thread, register)
        positions = _positions(b, window, coords)
        address = _unit_index(b, _address(b, window, positions), dtype)
        slot = _unit_count(dtype, register)
        # A wide access is only chosen for a tile that cannot overhang, so it never needs a mask; a
        # packed whole-word group may, and is masked as a unit (its bound is whole groups).
        mask = _clip_mask(b, window, positions, tile_desc.shape)
        if mask is None and vw > 1:
            # Contiguous stride-1 run -> single vectorised global load (global_load_dwordxN).
            loaded = b.global_load_vN(
                ptr, address, unit, vw, align=_access_align(vw, dtype)
            )
            for i in range(vw):
                value = b.vec_insert(value, b.vec_extract(loaded, i), slot + i)
            continue
        if mask is not None:
            if pad != 0:
                # Only a zero clip pad is implemented; a per-stage pad (e.g. -inf for a max) is
                # reserved for the reduction stages.
                raise NotImplementedError(
                    f"non-zero clip pad (non-GEMM stage identity) is reserved for the reduction "
                    f"stages -- pad={pad!r}"
                )
            if zero is None:
                zero = _zero_scalar(b, dtype)
        for i in range(vw):
            unit_address = address if i == 0 else b.add(address, b.const_i32(i))
            if mask is None:
                loaded = b.global_load(ptr, unit_address, unit, align=align)
            else:
                loaded = b.masked_global_load(
                    ptr, unit_address, mask, zero, unit, align=align
                )
            value = b.vec_insert(value, loaded, slot + i)
    return Fragment(tile_desc, dtype, value)


def _stored_unit(b: Any, fragment: Fragment, slot: int, out_dtype: Any) -> Any:
    """Storage register `slot` of `fragment`, cast to `out_dtype` (:func:`_cast_element`)."""
    return _cast_element(
        b, b.vec_extract(fragment.value, slot), fragment.dtype, out_dtype
    )


def _global_store_units(
    b: Any, ptr: Any, address: Any, units: list[Any], align: int
) -> None:
    """Store `units` at consecutive storage-unit addresses from `address`, one scalar store each."""
    for i, value in enumerate(units):
        unit_address = address if i == 0 else b.add(address, b.const_i32(i))
        b.global_store(ptr, unit_address, value, align=align)


def store_fragment(
    b: Any,
    ptr: Any,
    window: TensorWindow,
    fragment: Fragment,
    thread: Any,
    *,
    coherency: int = 0,
    lds_swizzle: bool | Callable[[Any, list], list] = False,
) -> None:
    """Generic per-lane store: `Fragment` -> memory (`ptr` + `window`), casting the fragment
    dtype to ``window.tensor.dtype`` only along the honest path (identity or f32->{f16,bf16}).

    M1 keeps the explicit per-element scatter + cast; the general MMA->memory relayout is
    reserved -- this is not a transparent memcpy. `ptr` is the DEST: a global ``ptr<...>`` OR an LDS
    ``smem<...>`` buffer -- the store op is chosen from the dest (:func:`_is_lds`). If the window has
    a clip, OOB global writes are **dropped** (guarded by ``scf_if``); C2's branchless buffer
    ``_OOB_SENTINEL`` drop is the reserved efficient path. LDS stores address the flat buffer through
    the window's strides, exactly like global stores, after the same buffer checks as
    :func:`load_fragment` (unclipped). A rank-t fragment covers the window's last t axes, as in
    :func:`load_fragment`.
    """
    _check_tile_rank("store_fragment", window, fragment.tile_desc)
    _check_leading_bounds("store_fragment", window, fragment.tile_desc)
    tile_desc = fragment.tile_desc
    out_dtype = window.tensor.dtype
    align = _unit_bits(out_dtype) // 8
    lds = _is_lds(ptr)
    # Wide LDS stores when the innermost register run is contiguous (ds_write_b{32,64,128}); the
    # global store stays scalar (the C epilogue is one-shot + may clip/cast): one element, or one
    # whole-word group of a packed dtype, per access.
    if lds:
        vw = _contiguous_run(tile_desc, window, out_dtype, True, verb="store_fragment")
        # A custom swizzle relocates whole units of its granularity (vw_elems) -> the access is
        # capped and range-checked to [1, natural run]. The built-in bool swizzle preserves its run.
        vw = _swizzle_vw(lds_swizzle, vw, align)
    elif _is_packed(out_dtype):
        _check_packed_runs("store_fragment", tile_desc, window, out_dtype, False)
        vw = _packed_group(out_dtype)[1]
    else:
        vw = 1
    step = vw * _unit_bits(out_dtype) // _element_bits(out_dtype)
    unit = _storage_element(out_dtype)

    # Global path: compute the LANE base-address ONCE (register 0 = the thread-only address, no register
    # offset), then emit each store as base + <compile-time constant> -- the fixed per-register coordinate
    # offset linearized by the strides. Consecutive registers' addresses then differ by a constant, which
    # the backend store vectorizer merges into wide dwordx4 transactions; re-deriving the full address per
    # register (the lane mod/div rebuilt each time) instead leaves every store a distinct expression it
    # cannot relate, so ANY strided global store this verb emits stays scalar/narrow -- the C epilogue is
    # just the visible case. Bit-exact with the per-register reconstruction by linearity of _address. The
    # LDS path uses the same base-plus-constant form per wide run (see _lds_index).
    if lds:
        lds_base_index = _checked_lds_base_index(
            "store_fragment", b, ptr, window, tile_desc, thread, lds_swizzle
        )
        for register in range(0, tile_desc.register_count, step):
            index = _lds_index(
                b,
                window,
                tile_desc,
                thread,
                register,
                lds_swizzle,
                lds_base_index,
                out_dtype,
            )
            slot = _unit_count(fragment.dtype, register)
            if vw == 1:
                b.smem_store_vN(
                    ptr, [index], _stored_unit(b, fragment, slot, out_dtype), 1
                )
            elif vw & (vw - 1):
                # No LDS vector store of this width (a 3-word packed group): one store per word.
                for i in range(vw):
                    el = _stored_unit(b, fragment, slot + i, out_dtype)
                    b.smem_store_vN(ptr, [b.add(index, b.const_i32(i))], el, 1)
            else:
                vec = b.zero_vec(unit, vw)
                for i in range(vw):
                    el = _stored_unit(b, fragment, slot + i, out_dtype)
                    vec = b.vec_insert(vec, el, i)
                b.smem_store_vN(ptr, [index], vec, vw)
        return
    base_positions, base_index = _lane_base_index(b, window, tile_desc, thread)
    base_index = _unit_index(b, base_index, out_dtype)
    for register in range(0, tile_desc.register_count, step):
        positions = _register_positions(b, tile_desc, base_positions, register)
        address = _register_index(b, window, tile_desc, base_index, register, out_dtype)
        slot = _unit_count(fragment.dtype, register)
        units = [_stored_unit(b, fragment, slot + i, out_dtype) for i in range(vw)]
        mask = _clip_mask(b, window, positions, tile_desc.shape)
        if mask is None:
            _global_store_units(b, ptr, address, units, align)
        else:
            with b.scf_if(mask):
                _global_store_units(b, ptr, address, units, align)
