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

from typing import Any

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

def emit_tensor_coordinates(
    b: Any, encoding: WarpDistributionEncoding, lane: Any, register_index: int
) -> tuple[Any, ...]:
    """IR-emitting ``calculate_x``: (runtime ``lane`` SSA, compile-time register) -> coords.

    Emits the same mixed-radix arithmetic the pure-int register mapper computes, but on the
    runtime lane value: decompose ``lane`` across its contributing buckets (last listed =
    fastest) via ``div``/``mod``, place the compile-time register indices, then reconstruct
    each X coordinate innermost-stride-1. Returns ONE SSA coordinate per X-dim of the encoding
    -- so it is N-D: 2 coords for an MMA fragment (``(m, k)`` for A, ``(row, col)`` for C), N
    for an N-D data tile. The verbs iterate these against the window's per-axis origin/strides.
    """
    lane_buckets = list(zip(encoding.lane_to_rh_major[0], encoding.lane_to_rh_minor[0]))
    lane_lengths = [encoding.bucket_length(*bucket) for bucket in lane_buckets]
    register_buckets = list(
        zip(encoding.register_to_rh_major, encoding.register_to_rh_minor)
    )
    register_lengths = [encoding.bucket_length(*bucket) for bucket in register_buckets]

    contributor: dict[tuple[int, int], Any] = {}

    # Runtime lane -> per-bucket contributor (last bucket changes fastest).
    suffix = 1
    for position in reversed(range(len(lane_lengths))):
        length = lane_lengths[position]
        if length == 1:
            contributor[lane_buckets[position]] = b.const_i32(0)
        else:
            divided = lane if suffix == 1 else b.div(lane, b.const_i32(suffix))
            contributor[lane_buckets[position]] = b.mod(divided, b.const_i32(length))
        suffix *= length

    # Compile-time register index -> per-bucket contributor (last fastest).
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
    """The strided element address from precomputed positions: sum(position * stride)."""
    address: Any = None
    for axis, position in enumerate(positions):
        stride = window.tensor.strides[axis]
        term = position if stride == 1 else b.mul(position, b.const_i32(stride))
        address = term if address is None else b.add(address, term)
    return address

def _clip_mask(
    b: Any, window: TensorWindow, positions: list[Any], tile_shape: tuple[int, ...]
) -> Any:
    """In-bounds predicate (i1) for an element at `positions`, or `None` when no axis needs a
    compare. The per-axis upper bound defaults to ``window.tensor.lengths`` and is overridden by
    ``window.bounds`` (a `None` entry keeps the length). Predicate = AND over checked axes of
    ``position < bound``. A tile-aligned compile-time bound can NEVER overhang, so it is SKIPPED
    at build time -- an aligned kernel emits no compare and stays byte-identical to no-clip."""
    mask: Any = None
    for axis, position in enumerate(positions):
        bound = window.bounds[axis] if window.bounds is not None else None
        if bound is None:
            bound = window.tensor.lengths[axis]
        if isinstance(bound, int) and bound % tile_shape[axis] == 0:
            continue
        in_axis = b.cmp_lt(position, _as_value(b, bound))
        mask = in_axis if mask is None else b.land(mask, in_axis)
    return mask

def _zero_scalar(b: Any, dtype: Any) -> Any:
    """A scalar 0 of `dtype` -- the zero-pad value handed to ``masked_global_load``."""
    return b.vec_extract(b.zero_vec(dtype, 1), 0)

def load_fragment(
    b: Any,
    ptr: Any,
    window: TensorWindow,
    tile_desc: TileDesc,
    lane: Any,
    *,
    pad: Any = 0,
    coherency: int = 0,
) -> Fragment:
    """Generic per-lane load: memory (`ptr` + `window`) -> `Fragment`, driven by `tile_desc.layout`.

    ONE verb for A, B, C (and later index/scale) -- the operand difference is the window's
    TensorDesc (strides) + the tile_desc. `ptr` is the typed base pointer; the fragment's dtype
    IS ``window.tensor.dtype`` (nothing is baked in). Where the window clips (an element past its
    bound), OOB elements are filled with `pad` (`masked_global_load`); otherwise the plain
    `global_load` fast path (byte-identical to no-clip). `pad` default `0` (zero-pad / `mask`); a
    non-zero `constant(value)` fill is reserved. `coherency` reserved.
    """
    dtype = window.tensor.dtype
    align = _align_of(dtype)
    value = b.zero_vec(dtype, tile_desc.register_count)
    zero: Any = None  # the pad scalar, emitted lazily on the first clipped element
    for register in range(tile_desc.register_count):
        coords = emit_tensor_coordinates(b, tile_desc.layout, lane, register)
        positions = _positions(b, window, coords)
        address = _address(b, window, positions)
        mask = _clip_mask(b, window, positions, tile_desc.shape)
        if mask is None:
            loaded = b.global_load(ptr, address, dtype, align=align)
        else:
            if pad != 0:
                raise NotImplementedError(
                    f"non-zero pad (constant-value fill) is reserved -- pad={pad!r}"
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
    lane: Any,
    *,
    coherency: int = 0,
) -> None:
    """Generic per-lane store: `Fragment` -> memory (`ptr` + `window`), casting the fragment
    dtype to ``window.tensor.dtype`` only along the honest path (identity or f32->{f16,bf16}).

    M1 keeps the explicit per-element scatter + cast; the general MMA->memory relayout is
    reserved -- this is not a transparent memcpy. If the window has a clip, OOB writes are
    **dropped** (guarded by ``scf_if`` on the in-bounds predicate); C2's branchless buffer
    ``_OOB_SENTINEL`` drop is the reserved efficient path.
    """
    out_dtype = window.tensor.dtype
    align = _align_of(out_dtype)
    for register in range(fragment.tile_desc.register_count):
        coords = emit_tensor_coordinates(b, fragment.tile_desc.layout, lane, register)
        positions = _positions(b, window, coords)
        address = _address(b, window, positions)
        element = b.vec_extract(fragment.value, register)
        value = _cast_element(b, element, fragment.dtype, out_dtype)
        mask = _clip_mask(b, window, positions, fragment.tile_desc.shape)
        if mask is None:
            b.global_store(ptr, address, value, align=align)
        else:
            with b.scf_if(mask):
                b.global_store(ptr, address, value, align=align)
