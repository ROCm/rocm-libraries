# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The cooperative global->LDS memory bridge -- the style-agnostic primitives a layout style COMPOSES.

A layout style (``mma/styles/``) owns the *recipe* (which descriptors, in what order); this module
owns the *correctness envelope* the style cannot reach past (rev-7 Principle 8, seam invariants):

- ``cooperative_load_desc`` / ``cooperative_load_width`` -- the block-wide, free-dim-contiguous global
  load tile (the wide, coalesced direction). Rank-2-enforcing.
- ``lds_tile_desc`` -- the FULL-tile LDS tensor descriptor, DERIVED from ``desc_extents`` of the
  cooperative descriptor (invariant a) and written in the convention order ``(free, K)``. The extent is
  ``bounds``-independent: a style never sizes LDS, so it cannot clip-size the buffer (which would let a
  full-width unmasked read index a neighbour workgroup's LDS). The buffer is flat and sized from the
  descriptor: ``b.smem_alloc(td.dtype, [td.element_span])``.
- ``lds_store`` / ``lds_read`` -- the LDS-facing verbs. Their signature has NO ``bounds`` parameter, so
  masking an LDS access is unrepresentable (invariant b -- the cooperative store/read stay full-width
  and unmasked; zeros, if any, flow through LDS and contribute nothing to the MAC). Each runs the
  ``desc_extents``-vs-descriptor-lengths fit guard UNCONDITIONALLY before emitting (invariant c); the
  verbs they call then check the flat buffer holds the descriptor's span.
"""

from __future__ import annotations

from typing import Any

from .descriptors import TensorDesc, make_tensor_desc, make_window
from .emit import load_fragment, store_fragment
from .fragments import Fragment, TileDesc
from .layouts import make_tile_desc
from .lds_conflict import desc_extents

__all__ = [
    "cooperative_load_desc",
    "cooperative_load_width",
    "lds_tile_desc",
    "lds_store",
    "lds_read",
]


def cooperative_load_width(
    tile_free: int, tile_k: int, n_waves: int, *, wave_size: int = 64
) -> int | None:
    """Widest GEOMETRY-PERMITTED per-lane free-run (in elements) for a cooperative load of this operand,
    or ``None`` if none works. This is the distribution ceiling only -- the dtype / 128-bit HARDWARE ceiling
    is applied separately in ``emit``: ``_contiguous_run`` caps a vector-eligible dtype at 16 B (an f32
    8-run -> 2x ``dwordx4``) and scalarizes an ineligible one (f64) outright, so a wide result here need not
    be a single global access. DERIVED, and PER OPERAND -- A and B have independent free extents so may want
    different widths.

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

    The lane takes ``vw`` CONTIGUOUS free-dim elements. This ASSUMES the operand's free axis is stride-1
    (an M-contiguous A, an N-contiguous B) -- then the free dim is the wide, coalesced direction;
    for a K-contiguous operand the coalesced axis is K instead. ``thread_order`` puts the free axis LAST
    (fastest), so consecutive lanes step the free dim by ``vw``; the register order falls out as [K major, free minor]
    -> the free dim is innermost -> the global load AND the LDS store are both wide with NO reorder
    between them. ``vw`` is REQUIRED (derive it with :func:`cooperative_load_width`, or pin it).

    CAUTION (served-group property): the LDS store's "one K per served group" property (what keeps the
    row stride out of its bank map) holds only while the free-axis lane count reaches a served group
    (arch-specific; 32 on gfx90a). This descriptor is free-major, so that count is ``tile_free/vw``; the
    identity is NOT general -- dump the address map rather than assume it for another config.
    """
    if vw < 1 or tile_free % vw:
        raise ValueError(
            f"cooperative load: tile_free={tile_free} not divisible by vw={vw}"
        )
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
        wave_dist=[
            1,
            n_waves,
        ],  # the waves split K; each wave loads the full free extent
        wave_size=wave_size,
    )
    if len(desc.shape) != 2:
        raise ValueError(
            f"cooperative_load_desc must be rank-2 (free, K) -- got shape {desc.shape!r}"
        )
    return desc


def lds_tile_desc(
    coop_desc: TileDesc,
    dtype: Any,
    *,
    buffers: int,
    n_waves: int,
    wave_size: int = 64,
) -> TensorDesc:
    """The FULL-tile LDS tensor descriptor for a cooperative-load operand, in the convention order
    ``(free, K)`` -- DERIVED from ``desc_extents`` of ``coop_desc`` itself, so it is
    ``bounds``-independent (seam invariant a). A style supplies the coop descriptor; the SEAM sizes the
    buffer. A style can never clip-size it.

    ``buffers`` is the multi-buffer count. The buffers sit side by side on the free axis -- buffer ``i``
    starts at free position ``i * free_ext`` -- so the lengths are ``(buffers * free_ext, k_ext)`` and
    the strides ``(1, buffers * free_ext)``: element ``(f, k)`` lives at ``f + k * buffers * free_ext``,
    a ``[K][buffers * free]`` buffer. Allocate it flat from the descriptor,
    ``b.smem_alloc(td.dtype, [td.element_span])``, and address it with the cooperative and read tiles
    as they are (``(free, K)``, unswapped) at origins ``(free_position, k)``.

    The stride-1 axis is axis 0 (the free axis), so the built-in ``lds_swizzle=True`` refuses this
    descriptor (it needs the stride-1 axis last); a swizzle on it is a swizzle callable. Vector width
    is unaffected: the cooperative and read tiles keep the free axis innermost in their registers.
    """
    ext = desc_extents(coop_desc, wave_size * n_waves)
    if len(ext) != 2:
        raise ValueError(
            f"cooperative LDS descriptor must be rank-2 (free, K) -- extents {ext!r}"
        )
    free_ext, k_ext = ext
    row = buffers * free_ext
    return make_tensor_desc((row, k_ext), (1, row), dtype)


def _fit_or_raise(
    what: str, tile_desc: TileDesc, n_lanes: int, lds_tensor_desc: TensorDesc
) -> None:
    """Raise unless every axis of ``tile_desc`` fits the LDS descriptor's lengths (seam invariant c).
    LDS accesses are UNCLIPPED by design, so an out-of-range index is not masked -- it reads another
    workgroup's memory. Checked per axis (a product check hides compensating per-axis errors). This is
    an EXTENT-vs-lengths check (`origin` unchecked): it polices full-tile SIZING, not a within-buffer
    buffer-index bug -- the multi-buffer geometry closes that by construction. The tile covers the
    descriptor's LAST axes (a staged buffer's leading stage axis is positioned by the origin), so its
    extents are checked against those axes' lengths.
    """
    ext = desc_extents(tile_desc, n_lanes)
    lengths = tuple(lds_tensor_desc.lengths)
    covered = lengths[len(lengths) - len(ext) :]
    if len(ext) > len(lengths) or any(e > n for e, n in zip(ext, covered)):
        raise ValueError(
            f"{what} addresses {ext} but the LDS tensor descriptor has lengths {lengths} -- "
            f"the cooperative LDS access would run outside the buffer (unclipped by design). "
            f"Describe the buffer with lengths that cover the tile, per axis."
        )


def lds_store(
    b: Any,
    ptr: Any,
    lds_tensor_desc: TensorDesc,
    origin: tuple[Any, ...],
    fragment: Fragment,
    thread: Any,
    *,
    n_lanes: int,
) -> None:
    """Full-width, UNMASKED cooperative store into LDS (seam invariant b: NO ``bounds`` parameter, so an
    LDS mask is unrepresentable). Runs the fit guard against the descriptor's lengths first (invariant
    c). The store width follows the fragment's contiguous run (``ds_write_b{32,64,128}``).
    """
    _fit_or_raise("cooperative LDS store", fragment.tile_desc, n_lanes, lds_tensor_desc)
    window = make_window(
        lds_tensor_desc, origin
    )  # no bounds -- masking LDS is unrepresentable
    if window.bounds is not None:  # invariant b: the LDS window can never carry a clip
        raise ValueError(
            "cooperative LDS access window carries a clip -- invariant (b) violated: masking LDS is "
            "unrepresentable (the full-width store/read stays unmasked). Enforced here (survives -O) as "
            "the sibling extent guard _fit_or_raise is."
        )
    store_fragment(b, ptr, window, fragment, thread, lds_swizzle=False)


def lds_read(
    b: Any,
    ptr: Any,
    lds_tensor_desc: TensorDesc,
    origin: tuple[Any, ...],
    tile_desc: TileDesc,
    thread: Any,
    *,
    n_lanes: int,
) -> Fragment:
    """Full-width, UNMASKED cooperative read from LDS (seam invariant b: NO ``bounds`` parameter). Runs
    the fit guard against the descriptor's lengths first (invariant c). Returns the loaded fragment; the
    caller applies any MMA-ready register reorder (that is a register transform, not a memory access).
    """
    _fit_or_raise("cooperative LDS read", tile_desc, n_lanes, lds_tensor_desc)
    window = make_window(lds_tensor_desc, origin)  # no bounds
    if window.bounds is not None:  # invariant b: the LDS window can never carry a clip
        raise ValueError(
            "cooperative LDS access window carries a clip -- invariant (b) violated: masking LDS is "
            "unrepresentable (the full-width store/read stays unmasked). Enforced here (survives -O) as "
            "the sibling extent guard _fit_or_raise is."
        )
    return load_fragment(b, ptr, window, tile_desc, thread)
