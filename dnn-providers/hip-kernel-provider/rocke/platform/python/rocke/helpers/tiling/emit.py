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
from typing import Any, Callable

from .descriptors import TensorDesc, TensorWindow
from .encoding import WarpDistributionEncoding
from .fragments import Fragment, TileDesc

__all__ = [
    "emit_tensor_coordinates",
    "fill_fragment",
    "lds_element_span",
    "load_fragment",
    "store_fragment",
]

# Element byte-widths, keyed by the rocke ``ir.Type`` name. Drives the load/store
# alignment so NO dtype is baked into the verb -- the width follows the fragment's type.
_BYTE_WIDTH = {
    "i8": 1,
    "fp8e4m3": 1,
    "bf8e5m2": 1,
    "i16": 2,
    "f16": 2,
    "bf16": 2,
    "i32": 4,
    "f32": 4,
    "i64": 8,
    "f64": 8,
}


def _align_of(dtype: Any) -> int:
    """Natural alignment (bytes) for an ``ir.Type`` -- the element's own width."""
    try:
        return _BYTE_WIDTH[dtype.name]
    except KeyError as exc:
        raise NotImplementedError(
            f"no alignment known for dtype -- dtype={dtype.name!r}, "
            f"expected one of {sorted(_BYTE_WIDTH)}"
        ) from exc


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


def fill_fragment(b: Any, fragment: Fragment, scalar: Any) -> None:
    """Set every register of `fragment` to `scalar`, element-wise (no layout, no addressing).

    Only a compile-time zero fill is implemented today. A nonzero or runtime `scalar` is REJECTED
    rather than silently zero-filled, so a caller asking for ones gets a loud failure instead of a
    wrong fragment.
    """
    if not (isinstance(scalar, int) and scalar == 0):
        raise NotImplementedError(
            f"fill_fragment supports only a compile-time zero fill -- got scalar={scalar!r}; "
            "nonzero/runtime scalar splat is not implemented"
        )
    fragment.value = b.zero_vec(fragment.dtype, fragment.tile_desc.register_count)


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


def _check_lds_buffer(
    verb: str,
    buffer: Any,
    tensor: TensorDesc,
    lds_swizzle: bool | Callable[[Any, list], list],
) -> None:
    """Fail fast unless ``buffer`` can hold ``tensor``: an LDS buffer is flat (rank 1), holds the
    tensor's dtype, and holds at least :func:`lds_element_span` elements (the descriptor's span, or
    the built-in swizzle's reach). The descriptor is the one home of the buffer's layout; the
    allocation only gives it room, so a buffer that disagrees with it is refused here rather than
    read or written out of range (LDS accesses are unclipped). Window origins are trusted, as for
    every LDS access."""
    buffer_type = buffer.type
    span = lds_element_span(tensor, lds_swizzle)
    if len(buffer_type.shape) != 1:
        raise ValueError(
            f"{verb}: the LDS buffer {buffer_type.name} has rank {len(buffer_type.shape)} -- an LDS "
            f"buffer is flat; its layout lives in the tensor descriptor. Allocate "
            f"smem_alloc({tensor.dtype.name}, [{span}])"
        )
    if buffer_type.elem.name != tensor.dtype.name:
        raise ValueError(
            f"{verb}: LDS buffer element {buffer_type.elem.name} != tensor dtype {tensor.dtype.name}"
        )
    if span > buffer_type.shape[0]:
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
    b: Any, window: TensorWindow, tile_desc: TileDesc, base_index: Any, register: int
) -> Any:
    """The register's element index: the lane base index plus the register's compile-time coordinate
    offset linearized by the tensor strides. Consecutive registers' indices then differ by a constant,
    which the backend vectorizer can merge; it is bit-exact with :func:`_address` of the register's
    positions by linearity."""
    deltas = _register_coord_offsets(tile_desc.layout, register)
    lead = _leading_rank(window, len(deltas))
    offset = sum(
        delta * window.tensor.strides[lead + axis] for axis, delta in enumerate(deltas)
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
    """A scalar 0 of `dtype` -- the zero-pad value handed to ``masked_global_load``."""
    return b.vec_extract(b.zero_vec(dtype, 1), 0)


_WIDE_OK = {
    "f16",
    "bf16",
    "f32",
    "i32",
}  # dtypes global_load_vN / smem_load_vN vectorize


def _contiguous_run(
    tile_desc: TileDesc, window: TensorWindow, dtype: Any, is_lds: bool
) -> int:
    """Width of the innermost stride-1 register run -> a single WIDE load; else 1 (scalar).

    The innermost register bucket varies fastest in the register index. Its registers are
    consecutive positions on their axis only when the bucket is that axis's INNERMOST hierarchical
    level (levels of length 1 inside it do not count). A bucket on an outer level steps the position
    by the product of the levels inside it -- an interleaved layout holds positions 0, 4, 8, 12 in
    four consecutive registers -- so it gives no run. When the bucket is the innermost level of the
    tensor's stride-1 axis, its registers are memory-contiguous and can be one `global_load_vN` /
    `smem_load_vN`. Width is the largest supported vector size for the dtype (8/4/2) dividing the run.

    A wide GLOBAL load is UNMASKED, so it is emitted only for a tile that is FULLY IN BOUNDS on the
    stride-1 run axis: the effective clip there (`window.bounds[axis]` or the tensor length) must be a
    COMPILE-TIME multiple of the tile extent, so no tile overhangs the tensor. A ragged/odd tensor, or
    a runtime clip that can't be proven aligned, falls to the SCALAR path (which masks against the
    length) -- never an out-of-bounds wide read (the page-fault risk Principle 4 kills). LDS is EXEMPT:
    the verbs check that the flat buffer holds the descriptor's whole span (:func:`_check_lds_buffer`)
    and LDS accesses are unclipped, so the gate applies to global only. Because both spaces address
    an element by ``sum(position * stride)``, consecutive registers of a run on a stride-1 axis are
    consecutive elements in LDS exactly as in global memory.

    The width must also divide every OTHER stride, leading axes included: a run then starts at a
    multiple of the width whatever the positions on the other axes, so the ``align`` the access claims
    holds (an odd batch stride would otherwise start a wide run mid-vector)."""
    layout = tile_desc.layout
    if dtype.name not in _WIDE_OK:
        return 1
    majors, minors = layout.register_to_rh_major, layout.register_to_rh_minor
    if not majors:
        return 1
    lead = _leading_rank(window, len(tile_desc.shape))
    tile_axis = majors[-1] - 1  # innermost register bucket's X-dim -> tile axis
    strides = window.tensor.strides
    run_axis = lead + tile_axis
    if tile_axis < 0 or strides[run_axis] != 1:
        return 1
    inner_levels = layout.hierarchical_lengths[tile_axis][minors[-1] + 1 :]
    if math.prod(inner_levels) != 1:
        return 1  # an outer level of its axis: consecutive registers are not consecutive positions
    other_strides = [stride for axis, stride in enumerate(strides) if axis != run_axis]
    if not is_lds:
        # Fully-in-bounds gate (global only): a wide load is UNMASKED, so it is safe only if the tile
        # overhangs NO axis it spans -- not just the stride-1 run axis. A wide K-run at a fixed but
        # OOB M row (odd M, aligned K) is just as out of bounds as an overhanging run. So EVERY axis's
        # effective clip must be a compile-time multiple of the tile extent on that axis; otherwise
        # some tile overhangs and the unmasked read faults -> scalarize (the masked path is safe).
        # Every axis's effective clip must be a compile-time tile multiple AND the tile origin must be on
        # the tile grid: a COMPILE-TIME mid-tile origin is caught (it could straddle the boundary and
        # overhang); a runtime (SSA) origin is TRUSTED grid-aligned (every caller: block_id*tile). This
        # is a PARTIAL guard -- an SSA mid-tile origin (attention) is not caught; that is DEFERRED with
        # attention support (see _origin_misaligned). Only the tile axes are checked: a leading axis
        # is positioned by its origin and not clipped, exactly as in _clip_mask.
        for tile_axis, extent in enumerate(tile_desc.shape):
            ax = lead + tile_axis
            clip = (
                window.bounds[ax]
                if window.bounds is not None and window.bounds[ax] is not None
                else window.tensor.lengths[ax]
            )
            if not (isinstance(clip, int) and clip % extent == 0):
                return 1
            if _origin_misaligned(window.origin[ax], extent):
                return 1
    length = layout.bucket_length(majors[-1], minors[-1])
    max_vw = (
        16 // _BYTE_WIDTH[dtype.name]
    )  # one load = dwordx4 = 16 bytes; longer runs -> N loads
    for vw in (16, 8, 4, 2):
        if (
            vw <= max_vw
            and length % vw == 0
            and all(stride % vw == 0 for stride in other_strides)
        ):
            return vw
    return 1


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
) -> Any:
    """The flat LDS element index of `register`'s run start, ``sum(position * stride)`` over the
    window's descriptor -- the same address rule as global memory.

    Without a swizzle it is the lane base index (``base_index``, from :func:`_lane_base_index`,
    emitted once per access) plus the register's compile-time offset: consecutive runs differ by a
    constant, which the backend folds into the instruction offset and pairs into ``ds_read2`` /
    ``ds_write2``. A swizzle is not linear, so a swizzled run rebuilds its positions, swizzles them
    and linearises the result (``base_index`` is unused then).

    The swizzle sees the position on EVERY tensor axis, leading axes (their origins) first, so an
    element's swizzled address does not depend on the rank of the tile that reaches it.
    """
    if not lds_swizzle:
        return _register_index(b, window, tile_desc, base_index, register)
    coords = emit_tensor_coordinates(b, tile_desc.layout, thread, register)
    lead = _leading_rank(window, len(coords))
    leading = [_as_value(b, window.origin[axis]) for axis in range(lead)]
    swizzle = _swizzle_lds_positions if lds_swizzle is True else lds_swizzle
    swizzled = swizzle(b, leading + _positions(b, window, coords))
    return _address(b, window, swizzled[lead:], swizzled[:lead])


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
    index the unswizzled runs add their offsets to (``None`` when a swizzle is active).
    """
    if lds_swizzle is True:
        _check_builtin_swizzle(window.tensor)
    _check_lds_buffer(verb, buffer, window.tensor, lds_swizzle)
    if lds_swizzle:
        return None
    return _lane_base_index(b, window, tile_desc, thread)[1]


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
    The fragment's dtype IS ``window.tensor.dtype`` (nothing is baked in). Where the window clips (an
    element past its bound), OOB elements are filled with `pad` (`masked_global_load`); otherwise the
    plain `global_load` fast path (byte-identical to no-clip). In both spaces an element's index is
    ``sum(position * stride)`` over the window's descriptor. An LDS buffer is flat
    (``b.smem_alloc(td.dtype, [lds_element_span(td, lds_swizzle)])``) and the descriptor is its
    layout: the buffer must be rank 1, hold the tensor's dtype and hold that span (refused
    otherwise); LDS loads are unclipped.
    `pad` default `0`; a non-zero `constant(value)` fill is reserved. `coherency` reserved.

    A rank-t tile covers the window's last t axes (t <= the window's rank, refused otherwise); each
    leading axis is read at its origin.
    """
    _check_tile_rank("load_fragment", window, tile_desc)
    _check_leading_bounds("load_fragment", window, tile_desc)
    dtype = window.tensor.dtype
    align = _align_of(dtype)
    lds = _is_lds(ptr)
    vw = _contiguous_run(tile_desc, window, dtype, lds)  # >1 -> one wide load per run
    # A custom swizzle relocates whole units of its granularity (vw_elems), capped + range-checked to
    # [1, natural run]. The built-in bool swizzle preserves its run.
    if lds:
        vw = _swizzle_vw(lds_swizzle, vw, align)
    value = b.zero_vec(dtype, tile_desc.register_count)
    if lds:
        # LDS source: one flat element index per run (wide when contiguous), lane base once.
        base_index = _checked_lds_base_index(
            "load_fragment", b, ptr, window, tile_desc, thread, lds_swizzle
        )
        for register in range(0, tile_desc.register_count, vw):
            index = _lds_index(
                b, window, tile_desc, thread, register, lds_swizzle, base_index
            )
            loaded = b.smem_load_vN(ptr, index, dtype=dtype, n=vw)
            for i in range(vw):
                value = b.vec_insert(value, b.vec_extract(loaded, i), register + i)
        return Fragment(tile_desc, dtype, value)
    zero: Any = None  # the pad scalar, emitted lazily on the first clipped element
    for register in range(0, tile_desc.register_count, vw):
        coords = emit_tensor_coordinates(b, tile_desc.layout, thread, register)
        positions = _positions(b, window, coords)
        address = _address(b, window, positions)
        if vw > 1:
            # Contiguous stride-1 run -> single vectorised global load (global_load_dwordxN).
            loaded = b.global_load_vN(ptr, address, dtype, vw, align=vw * align)
            for i in range(vw):
                value = b.vec_insert(value, b.vec_extract(loaded, i), register + i)
            continue
        mask = _clip_mask(b, window, positions, tile_desc.shape)
        if mask is None:
            loaded = b.global_load(ptr, address, dtype, align=align)
        else:
            if pad != 0:
                # Only a zero clip pad is implemented; a per-stage pad (e.g. -inf for a max) is
                # reserved for the reduction stages.
                raise NotImplementedError(
                    f"non-zero clip pad (non-GEMM stage identity) is reserved for the reduction "
                    f"stages -- pad={pad!r}"
                )
            if zero is None:
                zero = _zero_scalar(b, dtype)
            loaded = b.masked_global_load(ptr, address, mask, zero, dtype, align=align)
        value = b.vec_insert(value, loaded, register)
    return Fragment(tile_desc, dtype, value)


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
    out_dtype = window.tensor.dtype
    align = _align_of(out_dtype)
    lds = _is_lds(ptr)
    # Wide LDS stores when the innermost register run is contiguous (ds_write_b{32,64,128}); the
    # global store stays scalar (the C epilogue is one-shot + may clip/cast).
    vw = (
        _contiguous_run(fragment.tile_desc, window, out_dtype, is_lds=True)
        if lds
        else 1
    )
    # A custom swizzle relocates whole units of its granularity (vw_elems) -> the access is capped and
    # range-checked to [1, natural run]. The built-in bool swizzle preserves its run.
    if lds:
        vw = _swizzle_vw(lds_swizzle, vw, align)
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
            "store_fragment", b, ptr, window, fragment.tile_desc, thread, lds_swizzle
        )
        for register in range(0, fragment.tile_desc.register_count, vw):
            index = _lds_index(
                b,
                window,
                fragment.tile_desc,
                thread,
                register,
                lds_swizzle,
                lds_base_index,
            )
            if vw == 1:
                el = _cast_element(
                    b,
                    b.vec_extract(fragment.value, register),
                    fragment.dtype,
                    out_dtype,
                )
                b.smem_store_vN(ptr, [index], el, 1)
            else:
                vec = b.zero_vec(out_dtype, vw)
                for i in range(vw):
                    el = _cast_element(
                        b,
                        b.vec_extract(fragment.value, register + i),
                        fragment.dtype,
                        out_dtype,
                    )
                    vec = b.vec_insert(vec, el, i)
                b.smem_store_vN(ptr, [index], vec, vw)
        return
    base_positions, base_index = _lane_base_index(b, window, fragment.tile_desc, thread)
    for register in range(0, fragment.tile_desc.register_count, vw):
        positions = _register_positions(b, fragment.tile_desc, base_positions, register)
        address = _register_index(b, window, fragment.tile_desc, base_index, register)
        element = b.vec_extract(fragment.value, register)
        value = _cast_element(b, element, fragment.dtype, out_dtype)
        mask = _clip_mask(b, window, positions, fragment.tile_desc.shape)
        if mask is None:
            b.global_store(ptr, address, value, align=align)
        else:
            with b.scf_if(mask):
                b.global_store(ptr, address, value, align=align)
