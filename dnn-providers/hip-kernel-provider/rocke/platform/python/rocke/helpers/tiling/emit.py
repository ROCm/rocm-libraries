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

from typing import Any, Callable

from .descriptors import TensorWindow
from .encoding import WarpDistributionEncoding
from .fragments import Fragment, TileDesc

__all__ = [
    "emit_tensor_coordinates",
    "fill_fragment",
    "load_fragment",
    "store_fragment",
]

# Element byte-widths, keyed by the rocke ``ir.Type`` name. Drives the load/store
# alignment so NO dtype is baked into the verb -- the width follows the fragment's type.
_BYTE_WIDTH = {
    "i8": 1, "fp8e4m3": 1, "bf8e5m2": 1,
    "i16": 2, "f16": 2, "bf16": 2,
    "i32": 4, "f32": 4,
    "i64": 8, "f64": 8,
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
    narrowing. Any other conversion raises rather than silently doing the wrong thing."""
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
            contributor[partition_buckets[position]] = b.mod(divided, b.const_i32(length))
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
    register_buckets = list(zip(encoding.register_to_rh_major, encoding.register_to_rh_minor))
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
    M1: scalar is 0."""
    fragment.value = b.zero_vec(fragment.dtype, fragment.tile_desc.register_count)

def _as_value(b: Any, x: Any) -> Any:
    """An int becomes a const_i32; an SSA value passes through unchanged."""
    return b.const_i32(x) if isinstance(x, int) else x

def _positions(b: Any, window: TensorWindow, coords: tuple[Any, ...]) -> list[Any]:
    """Per-axis GLOBAL position = origin + coord (the basis of both the address and the clip)."""
    return [b.add(_as_value(b, window.origin[axis]), coord) for axis, coord in enumerate(coords)]

def _address(b: Any, window: TensorWindow, positions: list[Any]) -> Any:
    """The strided element address from precomputed positions: sum(position * stride), plus the
    fixed offset of any batch axes already reduced away by ``TensorWindow.at_index`` (``pinned``)."""
    address: Any = None
    for axis, position in enumerate(positions):
        stride = window.tensor.strides[axis]
        term = position if stride == 1 else b.mul(position, b.const_i32(stride))
        address = term if address is None else b.add(address, term)
    for index, stride in window.pinned:
        pos = _as_value(b, index)
        term = pos if stride == 1 else b.mul(pos, b.const_i32(stride))
        address = term if address is None else b.add(address, term)
    return address

def _register_coord_offsets(
    encoding: WarpDistributionEncoding, register_index: int
) -> tuple[int, ...]:
    """The register's COMPILE-TIME contribution to each X coordinate (lane buckets contribute 0) --
    the same value :func:`emit_coordinates_for_register` places, computed in pure Python. Lets the
    global store emit ``address = base + <constant>``: the lane base-address is computed ONCE and this
    fixed per-register offset (linearized by the tensor strides) is added, so consecutive registers'
    addresses differ by a constant and the store vectorizer merges them into ``dwordx4``. It is
    bit-exact with the per-register reconstruction by linearity of ``_address``."""
    register_buckets = list(zip(encoding.register_to_rh_major, encoding.register_to_rh_minor))
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


def _origin_misaligned(origin: Any, extent: int) -> bool:
    """True when `origin` is a COMPILE-TIME int NOT on the tile grid (`origin % extent != 0`): a tile
    placed there straddles a tile boundary and can overhang the tensor's far edge even when the extent
    is a tile multiple. A runtime (SSA) origin is TRUSTED grid-aligned -- every current caller positions
    tiles at ``block_id * tile``. PARTIAL GUARD: an SSA mid-tile origin (attention's sliding window) is
    NOT caught here; full mid-tile-origin handling is DEFERRED with attention support."""
    return isinstance(origin, int) and origin % extent != 0

def _loads_as_operand(tensor: Any) -> bool:
    """True when `tensor` is loaded as a TYPED MMA OPERAND -- it declares axis roles AND carries a
    CONTRACTION axis. A positional tensor (no roles) or an OUTPUT tensor whose roles are all `free`
    (a C reload, (M, N)) is NOT an operand, so the rank-2 (free, contraction) gate must not fire for it."""
    return tensor.axis_roles is not None and "contraction" in tensor.axis_roles

def _clip_mask(
    b: Any, window: TensorWindow, positions: list[Any], tile_shape: tuple[int, ...]
) -> Any:
    """In-bounds predicate (i1) for an element at `positions`, or `None` when no axis needs a
    compare. The per-axis upper bound defaults to ``window.tensor.lengths`` and is overridden by
    ``window.bounds`` (a `None` entry keeps the length). Predicate = AND over checked axes of
    ``position < bound``. A tile-aligned compile-time bound AT a grid-aligned origin can NEVER overhang,
    so it is SKIPPED at build time -- an aligned kernel emits no compare and stays byte-identical to
    no-clip; a compile-time MID-TILE origin is NOT skipped (it could straddle the boundary)."""
    mask: Any = None
    for axis, position in enumerate(positions):
        bound = window.bounds[axis] if window.bounds is not None else None
        if bound is None:
            bound = window.tensor.lengths[axis]
        if isinstance(bound, int) and bound % tile_shape[axis] == 0 \
                and not _origin_misaligned(window.origin[axis], tile_shape[axis]):
            continue
        in_axis = b.cmp_lt(position, _as_value(b, bound))
        mask = in_axis if mask is None else b.land(mask, in_axis)
    return mask

def _zero_scalar(b: Any, dtype: Any) -> Any:
    """A scalar 0 of `dtype` -- the zero-pad value handed to ``masked_global_load``."""
    return b.vec_extract(b.zero_vec(dtype, 1), 0)

_WIDE_OK = {"f16", "bf16", "f32", "i32"}   # dtypes global_load_vN / smem_load_vN vectorize

def _contiguous_run(
    tile_desc: TileDesc, window: TensorWindow, dtype: Any, is_lds: bool
) -> int:
    """Width of the innermost stride-1 register run -> a single WIDE load; else 1 (scalar).

    The innermost register bucket varies fastest in the register index; if it maps to the tensor's
    stride-1 axis then those registers are memory-contiguous and can be one `global_load_vN` /
    `smem_load_vN`. Width is the largest supported vector size for the dtype (8/4/2) dividing the run.

    A wide GLOBAL load is UNMASKED, so it is emitted only for a tile that is FULLY IN BOUNDS on the
    stride-1 run axis: the effective clip there (`window.bounds[axis]` or the tensor length) must be a
    COMPILE-TIME multiple of the tile extent, so no tile overhangs the tensor. A ragged/odd tensor, or
    a runtime clip that can't be proven aligned, falls to the SCALAR path (which masks against the
    length) -- never an out-of-bounds wide read (the page-fault risk Principle 4 kills). LDS is EXEMPT:
    its allocation is the full tile by construction (seam invariant a), so a wide LDS access never
    overhangs; the gate applies to global only."""
    layout = tile_desc.layout
    if dtype.name not in _WIDE_OK:
        return 1
    majors, minors = layout.register_to_rh_major, layout.register_to_rh_minor
    if not majors:
        return 1
    axis = majors[-1] - 1                      # innermost register bucket's X-dim -> tensor axis
    if axis < 0 or window.tensor.strides[axis] != 1:
        return 1
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
        # attention support (see _origin_misaligned).
        if len(tile_desc.shape) != len(window.tensor.lengths):
            raise ValueError(
                f"tile rank {len(tile_desc.shape)} != tensor rank {len(window.tensor.lengths)} -- "
                f"the wide-load bounds check needs matching ranks (reduce an N-D tensor with at_index)"
            )
        for ax in range(len(window.tensor.lengths)):
            clip = window.bounds[ax] if window.bounds is not None and window.bounds[ax] is not None \
                else window.tensor.lengths[ax]
            if not (isinstance(clip, int) and clip % tile_desc.shape[ax] == 0):
                return 1
            if _origin_misaligned(window.origin[ax], tile_desc.shape[ax]):
                return 1
    length = layout.bucket_length(majors[-1], minors[-1])
    max_vw = 16 // _BYTE_WIDTH[dtype.name]   # one load = dwordx4 = 16 bytes; longer runs -> N loads
    for vw in (16, 8, 4, 2):
        if vw <= max_vw and length % vw == 0:
            return vw
    return 1

def _is_lds(source: Any) -> bool:
    """True when `source` is an LDS buffer (an ``smem<...>`` value from ``smem_alloc``) rather
    than a global/constant ``ptr<...>``. Lets one verb serve both spaces -- the load/store op is
    chosen from the SOURCE, so the caller just hands the buffer (global ptr OR LDS smem) it has."""
    name = getattr(getattr(source, "type", None), "name", "")
    return isinstance(name, str) and name.startswith("smem")

def _swizzle_lds_positions(b: Any, positions: list) -> list:
    """XOR-swizzle the innermost LDS index by the outer index, at b128-block (8-f16) granularity:
    ``m -> ((m>>3) ^ (outer & 7)) << 3 | (m & 7)``. Keeps every 8-f16 (b128) run contiguous while
    de-aliasing LDS banks across the outer (K) dim -> conflict-free WITHOUT padding (which breaks
    alignment). A bijection of the physical ``(outer, m)``, so applying it identically to the store and
    the read leaves correctness intact regardless of the two fragments' layouts. Requires each access to
    be naturally vw-aligned within an 8-f16 block (true for our b128 store / b64 read)."""
    if len(positions) < 2:
        return positions
    outer, m = positions[-2], positions[-1]
    # m ^ ((outer & 7) << 3): flips m's block bits [3:6] by (K&7), leaving the low-3 (8-f16 b128 run)
    # intact -> identical bank mapping to a block XOR, in 3 VALU ops (mod, shl, xor) not 6.
    m_swz = b.xor(m, b.shl(b.mod(outer, b.const_i32(8)), b.const_i32(3)))
    return positions[:-1] + [m_swz]


def _swizzle_vw(lds_swizzle: Any, dist_vw: int, align: int) -> int:
    """Resolve the LDS access width when a swizzle is active, with a fail-fast range check.

    A callable swizzle relocates whole units of its granularity, so the access must not span wider than
    that unit. The callable declares its granularity via a ``vw_elems`` attribute (default = one dword);
    the built-in bool swizzle preserves the natural run. The chosen width is checked to satisfy
    ``1 <= vw <= dist_vw`` (the distribution's natural contiguous run) -- wider is unrepresentable."""
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
    plain `global_load` fast path (byte-identical to no-clip). LDS loads are unclipped (the buffer is
    exactly sized) and index the smem memref directly by the layout's positions -- the window's
    strides are unused there. `pad` default `0`; a non-zero `constant(value)` fill is reserved.
    `coherency` reserved.
    """
    dtype = window.tensor.dtype
    # N-D boundary gate: a tensor loaded as a typed MMA OPERAND must be the rank-2 one-free-one-
    # contraction (EITHER order -- A is (M,K), B is (K,N)) an operand requires -- reduce batch axes
    # (at_index / squeeze) BEFORE loading. A positional rank-2 tensor (no roles) and an OUTPUT tensor
    # (roles all "free", a C reload) are NOT operands, so the gate does not fire for them.
    if _loads_as_operand(window.tensor):
        window.tensor.assert_mma_operand()
    align = _align_of(dtype)
    lds = _is_lds(ptr)
    vw = _contiguous_run(tile_desc, window, dtype, lds)   # >1 -> one wide load per run
    # A custom swizzle relocates whole units of its granularity (vw_elems), capped + range-checked to
    # [1, natural run]. The built-in bool swizzle preserves its run.
    if lds:
        vw = _swizzle_vw(lds_swizzle, vw, align)
    value = b.zero_vec(dtype, tile_desc.register_count)
    zero: Any = None  # the pad scalar, emitted lazily on the first clipped element
    for register in range(0, tile_desc.register_count, vw):
        coords = emit_tensor_coordinates(b, tile_desc.layout, thread, register)
        positions = _positions(b, window, coords)
        if lds:
            if lds_swizzle:
                swz = _swizzle_lds_positions if lds_swizzle is True else lds_swizzle
                positions = swz(b, positions)
            # LDS source: index the smem memref by the layout positions (wide when contiguous).
            loaded = b.smem_load_vN(ptr, *positions, dtype=dtype, n=vw)
            for i in range(vw):
                value = b.vec_insert(value, b.vec_extract(loaded, i), register + i)
            continue
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
                # The clip pad must be the compute stage's additive IDENTITY. On a CONTRACTION-axis
                # operand (a MAC input) that identity is 0 -- a non-zero pad would add phantom terms
                # to the sum. Forbid it outright; a general per-stage pad (e.g. -inf for max) is
                # reserved for the reduction stages (Step 7).
                roles = window.tensor.axis_roles
                if roles is not None and "contraction" in roles:
                    raise ValueError(
                        f"non-zero clip pad on a contraction-axis operand corrupts the MAC -- the "
                        f"additive identity is 0 (pad={pad!r})"
                    )
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
    ``_OOB_SENTINEL`` drop is the reserved efficient path. LDS stores index the smem memref by the
    layout positions (unclipped; strides unused).
    """
    out_dtype = window.tensor.dtype
    align = _align_of(out_dtype)
    lds = _is_lds(ptr)
    # Wide LDS stores when the innermost register run is contiguous (ds_write_b{32,64,128}); the
    # global store stays scalar (the C epilogue is one-shot + may clip/cast).
    vw = _contiguous_run(fragment.tile_desc, window, out_dtype, is_lds=True) if lds else 1
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
    # LDS path keeps the plain wrapper (its wide ds_write already vectorizes).
    base_positions: list[Any] = []
    base_address: Any = None
    if not lds:
        base_coords = emit_coordinates_for_register(
            b, fragment.tile_desc.layout, emit_lane_contributors(b, fragment.tile_desc.layout, thread), 0
        )
        base_positions = _positions(b, window, base_coords)
        base_address = _address(b, window, base_positions)
    for register in range(0, fragment.tile_desc.register_count, vw):
        if lds:
            coords = emit_tensor_coordinates(b, fragment.tile_desc.layout, thread, register)
            positions = _positions(b, window, coords)
            if lds_swizzle:
                swz = _swizzle_lds_positions if lds_swizzle is True else lds_swizzle
                positions = swz(b, positions)
            if vw == 1:
                el = _cast_element(b, b.vec_extract(fragment.value, register), fragment.dtype, out_dtype)
                b.smem_store_vN(ptr, positions, el, 1)
            else:
                vec = b.zero_vec(out_dtype, vw)
                for i in range(vw):
                    el = _cast_element(
                        b, b.vec_extract(fragment.value, register + i), fragment.dtype, out_dtype
                    )
                    vec = b.vec_insert(vec, el, i)
                b.smem_store_vN(ptr, positions, vec, vw)
            continue
        deltas = _register_coord_offsets(fragment.tile_desc.layout, register)
        positions = [
            base_positions[axis] if d == 0 else b.add(base_positions[axis], b.const_i32(d))
            for axis, d in enumerate(deltas)
        ]
        offset = sum(d * window.tensor.strides[axis] for axis, d in enumerate(deltas))
        address = base_address if offset == 0 else b.add(base_address, b.const_i32(offset))
        element = b.vec_extract(fragment.value, register)
        value = _cast_element(b, element, fragment.dtype, out_dtype)
        mask = _clip_mask(b, window, positions, fragment.tile_desc.shape)
        if mask is None:
            b.global_store(ptr, address, value, align=align)
        else:
            with b.scf_if(mask):
                b.global_store(ptr, address, value, align=align)
