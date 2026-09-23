# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The cooperative global->LDS memory bridge -- the style-agnostic primitives a layout style COMPOSES.

A layout style (``mma/styles/``) owns the *recipe* (which descriptors, in what order); this module
owns the *correctness envelope* the style cannot reach past (rev-7 Principle 8, seam invariants):

- ``cooperative_load_desc`` / ``cooperative_load_width`` -- the block-wide, free-dim-contiguous global
  load tile (the wide, coalesced direction). Rank-2-enforcing.
- ``lds_tile_alloc`` -- the FULL-tile LDS allocation shape, DERIVED from ``desc_extents`` of the
  (K, free) store descriptor (invariant a). The extent is ``bounds``-independent: a style never sizes
  LDS, so it cannot clip-size the allocation (which would let a full-width unmasked read index a
  neighbour workgroup's LDS).
- ``lds_store`` / ``lds_read`` -- the LDS-facing verbs. Their signature has NO ``bounds`` parameter, so
  masking an LDS access is unrepresentable (invariant b -- the cooperative store/read stay full-width
  and unmasked; zeros, if any, flow through LDS and contribute nothing to the MAC). Each runs the
  ``desc_extents``-vs-allocation fit guard UNCONDITIONALLY before emitting (invariant c).
"""

from __future__ import annotations

from typing import Any

from .descriptors import make_window
from .emit import load_fragment, store_fragment
from .fragments import Fragment, TileDesc
from .layouts import make_tile_desc
from .lds_conflict import desc_extents

__all__ = [
    "cooperative_load_desc",
    "cooperative_load_width",
    "lds_tile_alloc",
    "lds_store",
    "lds_read",
]


def cooperative_load_width(
    tile_free: int, tile_k: int, n_waves: int, *, wave_size: int = 64
) -> int | None:
    """Widest per-lane global vector width (in elements) a cooperative load of this operand can use,
    or ``None`` if none works. DERIVED, and PER OPERAND -- A and B have independent free extents so may
    want different widths.

    Two arithmetic constraints: the free axis splits into whole lanes (``tile_free % vw == 0`` and
    ``tile_free/vw`` divides the wave), and the waves split K evenly
    (``tile_k % ((wave/free_lanes) * n_waves) == 0``). The binding one at small tiles is per-thread
    SUPPLY -- a ``tile_free*tile_k`` tile shared by ``wave*n_waves`` threads cannot give every thread a
    whole ``vw``-run once ``vw`` exceeds that quotient; narrowing ``vw`` is the only fix.
    """
    for vw in (8, 4, 2, 1):
        if tile_free % vw:
            continue
        free_lanes = tile_free // vw
        if free_lanes > wave_size or wave_size % free_lanes:
            continue
        if tile_k % ((wave_size // free_lanes) * n_waves) == 0:
            return vw
    return None


def cooperative_load_desc(
    tile_free: int, tile_k: int, n_waves: int, *, vw: int, wave_size: int = 64
) -> TileDesc:
    """Block-wide cooperative global-load tile, in (free, K) axis order.

    The lane takes ``vw`` CONTIGUOUS free-dim elements -- the free dim is the tensor's stride-1 axis, so
    this is the wide, coalesced direction. ``thread_order`` puts the free axis LAST (fastest), so
    consecutive lanes step the free dim by ``vw``; the register order falls out as [K major, free minor]
    -> the free dim is innermost -> the global load AND the LDS store are both wide with NO reorder
    between them. ``vw`` is REQUIRED (derive it with :func:`cooperative_load_width`, or pin it).

    CAUTION (served-group property): the LDS store's "one K per served group" property (what keeps the
    row stride out of its bank map) holds only while the free-axis lane count reaches a served group
    (arch-specific; 32 on gfx90a). This descriptor is free-major, so that count is ``tile_free/vw``; the
    identity is NOT general -- dump the address map rather than assume it for another config.
    """
    if vw < 1 or tile_free % vw:
        raise ValueError(f"cooperative load: tile_free={tile_free} not divisible by vw={vw}")
    free_lanes = tile_free // vw
    if free_lanes > wave_size or wave_size % free_lanes:
        raise ValueError(
            f"cooperative load: tile_free/vw = {free_lanes} does not divide wave {wave_size} "
            f"(tile_free={tile_free}, vw={vw})"
        )
    k_lanes = wave_size // free_lanes
    if tile_k % (k_lanes * n_waves):
        raise ValueError(
            f"cooperative load: tile_k={tile_k} not divisible by k_lanes*n_waves = "
            f"{k_lanes}*{n_waves} = {k_lanes * n_waves}"
        )
    k_repeat = tile_k // (k_lanes * n_waves)
    # `thread_order` names exactly the axes that CARRY lanes: make_tile_desc drops a lane bucket whose
    # thread_dist is 1, and a hardcoded order would then name an axis that is not there. K first, free
    # LAST so the free dim stays the fastest lane axis (the coalescing property); when k_lanes == 1 the
    # K entry drops out.
    lanes_per_axis = [free_lanes, k_lanes]  # axis 0 = free, axis 1 = K
    thread_order = [axis for axis in (1, 0) if lanes_per_axis[axis] > 1]
    desc = make_tile_desc(
        shape=[tile_free, tile_k],
        thread_tile=[vw, 1],
        thread_dist=[free_lanes, k_lanes],
        thread_order=thread_order,  # free dim fastest -> coalesced across lanes
        block_repeat=[1, k_repeat],
        wave_dist=[1, n_waves],  # the waves split K; each wave loads the full free extent
        wave_size=wave_size,
    )
    if len(desc.shape) != 2:
        raise ValueError(
            f"cooperative_load_desc must be rank-2 (free, K) -- got shape {desc.shape!r}"
        )
    return desc


def lds_tile_alloc(
    coop_desc: TileDesc, *, buffers: int, n_waves: int, wave_size: int = 64
) -> tuple[tuple[int, int], tuple[int, int]]:
    """The FULL-tile LDS allocation ``(shape, strides)`` for a cooperative-load operand -- DERIVED from
    ``desc_extents`` of the (K, free) store descriptor, so it is ``bounds``-independent (seam invariant
    a). ``buffers`` is the double-buffer count; the free axis is that multiple of the store's free
    extent, and the two buffer halves are free-dim-adjacent (row stride = ``buffers * free_extent``).

    A style supplies the coop descriptor; the SEAM sizes the allocation. A style can never clip-size it.
    """
    store_desc = coop_desc.swap_dims(0, 1)  # (free, K) -> (K, free): the LDS store order
    block_lanes = wave_size * n_waves
    ext = desc_extents(store_desc, block_lanes)
    if len(ext) != 2:
        raise ValueError(f"LDS store descriptor must be rank-2 (K, free) -- extents {ext!r}")
    k_ext, free_ext = ext
    shape = (k_ext, buffers * free_ext)
    strides = (buffers * free_ext, 1)
    return shape, strides


def _fit_or_raise(what: str, tile_desc: TileDesc, n_lanes: int, alloc: tuple[int, ...]) -> None:
    """Raise unless every axis of ``tile_desc`` fits the LDS ``alloc`` (seam invariant c). LDS accesses
    are UNCLIPPED by design, so an out-of-range index is not masked -- it reads another workgroup's
    memory. Sizing to the DESCRIPTOR, per axis (a product check hides compensating per-axis errors).
    This is an EXTENT-vs-alloc check (`origin` unchecked): it polices full-tile SIZING, not a
    within-alloc buffer-index bug -- the double-buffer geometry closes that by construction."""
    ext = desc_extents(tile_desc, n_lanes)
    if len(ext) != len(alloc) or any(e > a for e, a in zip(ext, alloc)):
        raise ValueError(
            f"{what} addresses {ext} but the LDS allocation is {tuple(alloc)} -- "
            f"the cooperative LDS access would run outside the allocation (unclipped by design). "
            f"Size the allocation to the descriptor, per axis."
        )


def lds_store(
    b: Any,
    ptr: Any,
    lds_tensor_desc: Any,
    origin: tuple[Any, ...],
    fragment: Fragment,
    thread: Any,
    *,
    alloc: tuple[int, ...],
    n_lanes: int,
) -> None:
    """Full-width, UNMASKED cooperative store into LDS (seam invariant b: NO ``bounds`` parameter, so an
    LDS mask is unrepresentable). Runs the fit guard against ``alloc`` first (invariant c). The store
    width follows the fragment's contiguous run (``ds_write_b{32,64,128}``)."""
    _fit_or_raise("cooperative LDS store", fragment.tile_desc, n_lanes, alloc)
    window = make_window(lds_tensor_desc, origin)  # no bounds -- masking LDS is unrepresentable
    store_fragment(b, ptr, window, fragment, thread, lds_swizzle=False)


def lds_read(
    b: Any,
    ptr: Any,
    lds_tensor_desc: Any,
    origin: tuple[Any, ...],
    tile_desc: TileDesc,
    thread: Any,
    *,
    alloc: tuple[int, ...],
    n_lanes: int,
) -> Fragment:
    """Full-width, UNMASKED cooperative read from LDS (seam invariant b: NO ``bounds`` parameter). Runs
    the fit guard against ``alloc`` first (invariant c). Returns the loaded fragment; the caller applies
    any MMA-ready register reorder (that is a register transform, not a memory access)."""
    _fit_or_raise("cooperative LDS read", tile_desc, n_lanes, alloc)
    window = make_window(lds_tensor_desc, origin)  # no bounds
    return load_fragment(b, ptr, window, tile_desc, thread)
