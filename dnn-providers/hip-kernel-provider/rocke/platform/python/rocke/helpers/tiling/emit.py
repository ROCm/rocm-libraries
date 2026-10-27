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

Which bound checks a global access needs is decided once, by the shared rule in
:mod:`rocke.helpers.tiling.clipping` (the same decision ``explain_bounds_checks`` reports). A window
proven inside its tensor with int strides is addressed here by one i32 element index; every other
global access is handed, one split coordinate per tensor axis, to
:mod:`rocke.helpers.tiling.safe_access`, which places its address and masks it by its validity.
"""

from __future__ import annotations

from typing import Any, Callable

from rocke.core.ir import Value

from .clipping import (
    GlobalForm,
    check_tile_rank,
    contiguous_run,
    element_bytes,
    global_store_width,
    is_lds,
    leading_rank,
    plan_window,
)
from .descriptors import TensorDesc, TensorWindow
from .encoding import WarpDistributionEncoding, lane_terms, register_coord_offsets
from .fragments import Fragment, TileDesc
from .safe_access import access_alignment, load_accesses, store_accesses
from .tensor_transforms.ranges import AffineCoord, Congruence, Interval

__all__ = [
    "emit_tensor_coordinates",
    "fill_fragment",
    "lds_element_span",
    "load_fragment",
    "store_fragment",
]


def _refuse_nonzero_pad(verb: str, pad: Any) -> None:
    """Refuse a ``pad`` other than 0: an element outside the tensor loads as 0 and no other fill
    is implemented, so a non-zero ``pad`` would be ignored. An IR value is no pad at all.
    """
    if isinstance(pad, Value):
        raise TypeError(
            f"{verb}: pad is the IR value {pad.name} -- pad is a Python number, and only pad=0 "
            "is implemented; leave pad at its default"
        )
    if pad != 0:
        raise NotImplementedError(
            f"{verb}: pad={pad!r} -- only pad=0 is implemented (an element outside the tensor "
            "loads as 0); leave pad at its default"
        )


def _refuse_coherency(verb: str, coherency: Any) -> None:
    """Refuse a ``coherency`` other than 0: no cache-coherency control is implemented, so any
    other value would be ignored. An IR value is no coherency at all."""
    if isinstance(coherency, Value):
        raise TypeError(
            f"{verb}: coherency is the IR value {coherency.name} -- coherency is a Python int, and "
            "only its default 0 is implemented; leave coherency at its default"
        )
    if coherency != 0:
        raise NotImplementedError(
            f"{verb}: coherency={coherency!r} -- no cache-coherency control is implemented; every "
            "access uses the default cache behaviour, so leave coherency at its default (0)"
        )


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


def _scaled(b: Any, position: Any, stride: Any) -> Any:
    """``position * stride`` as IR: an int stride 1 adds nothing, another int stride is a
    constant, a runtime stride is its Value."""
    if isinstance(stride, Value):
        return b.mul(position, stride)
    return position if stride == 1 else b.mul(position, b.const_i32(stride))


def _positions(b: Any, window: TensorWindow, coords: tuple[Any, ...]) -> list[Any]:
    """Per-tile-axis position = origin + coord (the basis of the one-index address).
    Tile axis ``i`` lies on tensor axis ``leading_rank + i``, so it takes that axis's origin.
    """
    lead = leading_rank(window, len(coords))
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
    lead = leading_rank(window, len(positions))
    if leading_positions is None:
        leading_positions = [_as_value(b, window.origin[axis]) for axis in range(lead)]
    strides = window.tensor.strides
    address: Any = None
    for axis, position in enumerate(positions):
        term = _scaled(b, position, strides[lead + axis])
        address = term if address is None else b.add(address, term)
    for axis, position in enumerate(leading_positions):
        term = _scaled(b, position, strides[axis])
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


def _lane_base_index(
    b: Any, window: TensorWindow, tile_desc: TileDesc, thread: Any
) -> tuple[list[Any], Any]:
    """Register 0's positions and element index -- the lane-only part of every register's address,
    emitted ONCE per access. Returns ``(positions, index)``: the positions feed the per-register
    positions of the bank-conflict address map and the index feeds :func:`_register_index`.
    """
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
    deltas = register_coord_offsets(tile_desc.layout, register)
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
    positions by linearity. An offset on a runtime-stride axis is a runtime value, added after the
    compile-time part."""
    deltas = register_coord_offsets(tile_desc.layout, register)
    lead = leading_rank(window, len(deltas))
    strides = [window.tensor.strides[lead + axis] for axis in range(len(deltas))]
    offset = sum(
        delta * stride
        for delta, stride in zip(deltas, strides)
        if not isinstance(stride, Value)
    )
    index = base_index if offset == 0 else b.add(base_index, b.const_i32(offset))
    for delta, stride in zip(deltas, strides):
        if isinstance(stride, Value) and delta != 0:
            index = b.add(index, b.mul(b.const_i32(delta), stride))
    return index


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
    tensor._require_static_for_lds("the built-in LDS swizzle")
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
    lead = leading_rank(window, len(coords))
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


def _access_coordinates(
    b: Any,
    window: TensorWindow,
    tile_desc: TileDesc,
    thread: Any,
    origins: tuple[AffineCoord, ...],
    width: int,
) -> list[tuple[AffineCoord, ...]]:
    """One coordinate tuple per access (``width`` registers from register ``0, width, ...``), one
    split coordinate per tensor axis, for tensor_transforms' ``emit_accesses``.

    A leading axis is its origin. A tile axis is its origin plus the layout's per-lane term
    (emitted once per call; its exact range comes from the layout) plus the first register's
    compile-time offset. An origin's own per-lane part (a wave offset) is added to the layout's,
    once per call, so each coordinate keeps one per-lane part."""
    layout = tile_desc.layout
    lead = leading_rank(window, len(tile_desc.shape))
    lanes = lane_terms(layout)
    lane_values = emit_coordinates_for_register(
        b, layout, emit_lane_contributors(b, layout, thread), 0
    )
    bases = list(origins)
    for tile_axis, term in enumerate(lanes):
        if term.extent == 1:
            continue
        axis = lead + tile_axis
        origin = origins[axis]
        lane, lane_range = lane_values[tile_axis], Interval(0, term.extent)
        congruence = Congruence(term.step, 0)
        if origin.lane is not None:
            lane = b.add(origin.lane, lane)
            lane_range = origin.lane_range.plus(lane_range)
            congruence = origin.lane_congruence.plus(congruence)
        bases[axis] = AffineCoord(
            origin.const, origin.uniform, lane, lane_range, congruence
        )
    accesses = []
    for register in range(0, tile_desc.register_count, width):
        offsets = register_coord_offsets(layout, register)
        accesses.append(
            tuple(
                (
                    base + AffineCoord.of_int(offsets[axis - lead])
                    if axis >= lead
                    else base
                )
                for axis, base in enumerate(bases)
            )
        )
    return accesses


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
    from the source (:func:`is_lds`), so the same verb stages global<->register and LDS<->register.
    The fragment's dtype IS ``window.tensor.dtype`` (nothing is baked in). In both spaces an
    element's index is ``sum(position * stride)`` over the window's descriptor.

    A global load reads 0 for every element outside the tensor's ``[0, length)`` on some axis, and
    never touches it. Which sides need a run-time check is decided at build time by the shared
    bound-check rule (:func:`~rocke.helpers.tiling.clipping.explain_bounds_checks` reports the same
    decision): a window proven inside the tensor loads unmasked, a checked one masks each access by
    its validity. An LDS buffer is flat (``b.smem_alloc(td.dtype, [lds_element_span(td,
    lds_swizzle)])``) and the descriptor is its layout, so it takes int lengths and strides only
    (rocKE allocates LDS statically, with a fixed size); the buffer must be rank 1, hold the
    tensor's dtype and hold that span (refused otherwise); LDS loads are unclipped.
    `pad` is the fill of an element outside the tensor: only 0 is implemented, and any other
    value is refused (``NotImplementedError``). `coherency` is reserved: any value but its default
    0 is refused (``NotImplementedError``).

    A rank-t tile covers the window's last t axes (t <= the window's rank, refused otherwise); each
    leading axis is read at its origin.
    """
    _refuse_nonzero_pad("load_fragment", pad)
    _refuse_coherency("load_fragment", coherency)
    check_tile_rank("load_fragment", window, tile_desc)
    dtype = window.tensor.dtype
    if is_lds(ptr):
        window.tensor._require_static_for_lds("load_fragment")
        # A custom swizzle relocates whole units of its granularity (vw_elems), capped + range-
        # checked to [1, natural run]. The built-in bool swizzle preserves its run.
        vw = _swizzle_vw(
            lds_swizzle,
            contiguous_run(tile_desc, window, dtype),
            element_bytes(dtype.name),
        )
        value = b.zero_vec(dtype, tile_desc.register_count)
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
    plan, origins = plan_window("load_fragment", b, ptr, window, tile_desc)
    vw = plan.vector_width
    claim = access_alignment(b, ptr, window.tensor, origins, tile_desc, vw)
    value = b.zero_vec(dtype, tile_desc.register_count)
    if plan.form is GlobalForm.INDEX32:
        # Every side proven and every offset inside i32: one i32 element index from the pointer.
        for register in range(0, tile_desc.register_count, vw):
            coords = emit_tensor_coordinates(b, tile_desc.layout, thread, register)
            address = _address(b, window, _positions(b, window, coords))
            if vw > 1:
                loaded = b.global_load_vN(ptr, address, dtype, vw, align=claim)
                for i in range(vw):
                    value = b.vec_insert(value, b.vec_extract(loaded, i), register + i)
            else:
                value = b.vec_insert(
                    value, b.global_load(ptr, address, dtype, align=claim), register
                )
        return Fragment(tile_desc, dtype, value)
    accesses = _access_coordinates(b, window, tile_desc, thread, origins, vw)
    loads = load_accesses(b, ptr, window.tensor, accesses, plan, claim)
    for register, loaded in zip(range(0, tile_desc.register_count, vw), loads):
        if vw == 1:
            value = b.vec_insert(value, loaded, register)
            continue
        for i in range(vw):
            value = b.vec_insert(value, b.vec_extract(loaded, i), register + i)
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
    ``smem<...>`` buffer -- the store op is chosen from the dest (:func:`is_lds`). A global store
    writes no element outside the tensor's ``[0, length)`` on some axis, and touches none; the
    checks are decided as for :func:`load_fragment`. LDS stores address the flat buffer through the
    window's strides, exactly like global stores, after the same buffer checks as
    :func:`load_fragment` (unclipped). A rank-t fragment covers the window's last t axes, as in
    :func:`load_fragment`. `coherency` is reserved: any value but its default 0 is refused
    (``NotImplementedError``).
    """
    _refuse_coherency("store_fragment", coherency)
    tile_desc = fragment.tile_desc
    check_tile_rank("store_fragment", window, tile_desc)
    out_dtype = window.tensor.dtype

    def element(register: int) -> Any:
        return _cast_element(
            b, b.vec_extract(fragment.value, register), fragment.dtype, out_dtype
        )

    if is_lds(ptr):
        window.tensor._require_static_for_lds("store_fragment")
        # Wide LDS stores when the innermost register run is contiguous (ds_write_b{32,64,128}). A
        # custom swizzle relocates whole units of its granularity (vw_elems) -> the access is
        # capped and range-checked to [1, natural run]. The built-in bool swizzle preserves its run.
        vw = _swizzle_vw(
            lds_swizzle,
            contiguous_run(tile_desc, window, out_dtype),
            element_bytes(out_dtype.name),
        )
        lds_base_index = _checked_lds_base_index(
            "store_fragment", b, ptr, window, tile_desc, thread, lds_swizzle
        )
        for register in range(0, tile_desc.register_count, vw):
            index = _lds_index(
                b, window, tile_desc, thread, register, lds_swizzle, lds_base_index
            )
            if vw == 1:
                b.smem_store_vN(ptr, [index], element(register), 1)
            else:
                vec = b.zero_vec(out_dtype, vw)
                for i in range(vw):
                    vec = b.vec_insert(vec, element(register + i), i)
                b.smem_store_vN(ptr, [index], vec, vw)
        return
    plan, origins = plan_window("store_fragment", b, ptr, window, tile_desc)
    vw = global_store_width(plan)
    claim = access_alignment(b, ptr, window.tensor, origins, tile_desc, vw)
    if plan.form is GlobalForm.INDEX32:
        # Every side proven and every offset inside i32: compute the LANE base index ONCE (register
        # 0 = the thread-only index), then emit each store as base + <compile-time constant> -- the
        # fixed per-register coordinate offset linearized by the strides. Consecutive registers'
        # addresses then differ by a constant, which the backend store vectorizer merges into wide
        # dwordx4 transactions; re-deriving the full address per register (the lane mod/div rebuilt
        # each time) leaves every store a distinct expression it cannot relate. Bit-exact with the
        # per-register reconstruction by linearity of _address. (``vw`` is 1 on this form.)
        _, base_index = _lane_base_index(b, window, tile_desc, thread)
        for register in range(tile_desc.register_count):
            address = _register_index(b, window, tile_desc, base_index, register)
            b.global_store(ptr, address, element(register), align=claim)
        return
    accesses = _access_coordinates(b, window, tile_desc, thread, origins, vw)
    values = []
    for register in range(0, tile_desc.register_count, vw):
        if vw == 1:
            values.append(element(register))
            continue
        vec = b.zero_vec(out_dtype, vw)
        for i in range(vw):
            vec = b.vec_insert(vec, element(register + i), i)
        values.append(vec)
    store_accesses(b, ptr, window.tensor, accesses, values, plan, claim)
