# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Low-level drawing primitives for the layout-viz renderer.

The matplotlib-facing gestures every view repeats -- fill a cell, outline a box, place a centered
label, set grid limits/ticks, pick a shade, draw a bank grid, render the grouped-cell overlay, place a
tick ruler, and the standalone-figure scaffold -- live here ONCE so `layout_render.py` (the domain
views) stays focused and consistent. This module is domain-agnostic and imports nothing from
`layout_render` (it owns the colour model so there is no upward dependency).

Colour model (the house convention, unchanged): COLOUR = thread/lane identity (8 accents, `lane % 8`);
TINT = visit/phase order (t0 = full accent, later steps blend toward white). `shade()` adds the
`first8` gating (only lanes 0..7 coloured, the rest grey) used across the views.
"""
from __future__ import annotations

from math import log2 as _log2  # noqa: F401  (kept for parity with layout_render imports)


def _plt():
    """Lazy matplotlib (Agg) -- keeps the module importable without a display / matplotlib."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


# ---------------------------------------------------------------- colour model
# 8 pleasing accents (6 Excel "Themed Cell Styles" + a pastel red + a pastel yellow), ordered by hue.
_HEX = ["E67C73",   # pastel red
        "E97132",   # orange
        "F1C232",   # pastel yellow
        "4EA72E",   # green
        "196B24",   # deep green
        "156082",   # teal-blue
        "0F9ED5",   # light blue
        "A02B93"]   # purple
ACCENTS = [tuple(int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)) for h in _HEX]
NACC = len(ACCENTS)
GREY = (0.92, 0.92, 0.92)          # non-first-cycle cell fill
EMPTY = (0.93, 0.93, 0.93)         # unoccupied bank/grid cell


def accent_tint(ai, tstep, ntsteps):
    """Accent ``ai`` at visit-order ``tstep``: t0 = full accent, later steps blend toward white (MONOTONIC:
    darkest = first, per §9.6 shade = transaction time order)."""
    base = ACCENTS[ai % NACC]
    p = 1.0 if ntsteps <= 1 else 1.0 - 0.72 * (tstep / (ntsteps - 1))   # 1.0(full) .. 0.28(pale)
    return tuple(base[k] * p + (1 - p) for k in range(3))


def cell_rgb(lane, tstep, ntsteps):
    """Thread colour = ACCENTS[lane % 8] (cycled); tint = visit order."""
    return accent_tint(lane % NACC, tstep, ntsteps)


def shade(lane, step, nsteps, *, color_mode="first8"):
    """Cell colour with the first-cycle gating: ``full`` -> every lane%8 coloured; ``first8`` (default)
    -> only lanes 0..7 coloured (tinted by ``step``), the rest neutral grey."""
    if color_mode == "full" or lane < NACC:
        return cell_rgb(lane, step, nsteps)
    return GREY


# ---------------------------------------------------------------- cell-unit primitives (ax-based)
def fill(ax, x, y, color, *, w=1, h=1, edge="white", lw=0.2, zorder=1):
    """Fill one grid cell (a solid rectangle)."""
    from matplotlib.patches import Rectangle
    ax.add_patch(Rectangle((x, y), w, h, facecolor=color,
                           edgecolor=edge if edge else "none", lw=lw, zorder=zorder))


def box(ax, x, y, w, h, *, color="black", lw=1.8, zorder=4):
    """Outline a region (an unfilled rectangle) -- group borders, anchors, red conflict boxes, highlights."""
    from matplotlib.patches import Rectangle
    ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor=color, lw=lw, zorder=zorder))


def fit_fs(s, cell_wh, fs_max, *, floor=4.5, pad=0.90):
    """The LARGEST font (pt, capped at ``fs_max``) at which label ``s`` fits inside a ``cell_wh`` (inches)
    cell -- so a per-cell DETAILED label in a dense grid scales DOWN to fit instead of overflowing into an
    unreadable overlap. ~0.60*fs per char wide, ~1.0*fs per line tall; ``pad`` leaves a hair of margin. A
    2-line label (``\\n``) uses the widest line for width + line count for height. Never below ``floor``."""
    cw, ch = cell_wh
    lines = s.split("\n")
    longest = max((len(t) for t in lines), default=1)
    w_fs = (cw * 72.0 * pad) / max(1.0, 0.60 * longest)
    h_fs = (ch * 72.0 * pad) / max(1, len(lines))
    return max(floor, min(fs_max, w_fs, h_fs))


def label(ax, x, y, s, *, fs, weight="normal", style="normal", color="0.0", ha="center", va="center",
          family=None, rotation=0, zorder=None):
    """Place text (centered by default). ``family='monospace'`` for the register/logical cell labels."""
    kw = dict(ha=ha, va=va, fontsize=fs, weight=weight, style=style, color=color, rotation=rotation)
    if family is not None:
        kw["family"] = family
    if zorder is not None:
        kw["zorder"] = zorder
    ax.text(x, y, s, **kw)


def grid_limits(ax, ncols, nrows, *, invert_y=True, aspect="equal"):
    """Set a cell grid's limits (origin top-left when ``invert_y``) and aspect."""
    ax.set_xlim(0, ncols)
    ax.set_ylim(nrows, 0) if invert_y else ax.set_ylim(0, nrows)
    if aspect is not None:
        ax.set_aspect(aspect)


def sparse_ticks(ax, ncols, nrows, *, div=8):
    """Coarse integer ticks (~``div`` per axis) for a dense grid."""
    ax.set_xticks(range(0, ncols + 1, max(1, ncols // div)))
    ax.set_yticks(range(0, nrows + 1, max(1, nrows // div)))


# ---------------------------------------------------------------- geometry helper
def grid_components(cells):
    """4-connected components of a set of ``(gx, gy)`` grid cells -> list of cell-sets. Lets a group whose
    members are non-contiguous in the grid be bordered as several contiguous runs."""
    cells = set(cells); comps = []
    while cells:
        stack = [cells.pop()]; comp = set(stack)
        while stack:
            x, y = stack.pop()
            for nb in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                if nb in cells:
                    cells.discard(nb); comp.add(nb); stack.append(nb)
        comps.append(comp)
    return comps


# ---------------------------------------------------------------- tick rulers
def edge_ticks(vals, step=1):
    """Edge-anchored ticks for a per-cell axis (one item per cell): the tick for a value sits on its cell's
    OUTER edge -- value 0 lands on the edge closest to it, so descending order anchors automatically."""
    asc = len(vals) < 2 or vals[0] <= vals[-1]
    return [(i if asc else i + 1, str(v)) for i, v in enumerate(vals) if v % step == 0]


_NICE_STEPS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048)   # powers of 2 -- aligns to tile dims


def auto_tick_step(nvals, cell_in, *, min_gap_in=0.42):
    """A power-of-2 tick step (in cells) so ticks land ~``min_gap_in`` INCHES apart at the actual render
    size -- ticks stay visible and never crowd, whatever the cell size, AND fall on tile-dim boundaries
    (…8, 16, 32, 64…) since tiles are powers of 2. ``cell_in`` = inches per cell along the axis
    (``cell_w``/``cell_h`` standalone, or the pipeline ``scale``)."""
    if nvals <= 1 or cell_in <= 0:
        return 1
    raw = max(1.0, min_gap_in / cell_in)                     # cells per tick to clear the gap
    for s in _NICE_STEPS:
        if s >= raw:
            return s
    return int(raw)


def reg_ticks(vals, dtype_bits):
    """Edge-anchored ticks for the vreg axis in PHYSICAL 32-bit registers. fp16 (2 elem/reg) -> one tick
    per 2-cell run; fp32 -> 1 tick/cell; fp64 -> each cell labelled with its 2-register span."""
    eb = dtype_bits
    asc = len(vals) < 2 or vals[0] <= vals[-1]
    edge = lambda i, j: i if asc else j                   # low-vreg boundary of the run [i, j)
    ticks = []
    if eb < 32:
        per = 32 // eb
        i = 0
        while i < len(vals):
            reg = vals[i] // per; j = i
            while j < len(vals) and vals[j] // per == reg:
                j += 1
            ticks.append((edge(i, j), str(reg))); i = j
    elif eb == 32:
        ticks = [(edge(i, i + 1), str(v)) for i, v in enumerate(vals)]
    else:
        span = eb // 32
        ticks = [(edge(i, i + 1), f"{v * span}-{v * span + span - 1}") for i, v in enumerate(vals)]
    return ticks


def set_axis_ticks(ax, ticks, *, axis, origin_off, name, fs, side):
    """Set matplotlib ticks/labels/name for a standalone (non-composed) axes. ``axis`` = 'x'|'y';
    ``side`` = bottom|top (x) or left|right (y)."""
    pos = [origin_off + p for p, _ in ticks]
    labs = [lab for _, lab in ticks]
    if axis == "x":
        ax.set_xticks(pos); ax.set_xticklabels(labs, fontsize=fs); ax.set_xlabel(name)
        s = "top" if side == "top" else "bottom"
        ax.xaxis.set_ticks_position(s); ax.xaxis.set_label_position(s)
    else:
        ax.set_yticks(pos); ax.set_yticklabels(labs, fontsize=fs); ax.set_ylabel(name)
        s = "right" if side == "right" else "left"
        ax.yaxis.set_ticks_position(s); ax.yaxis.set_label_position(s)


def axis_marks(ax, ticks, *, horizontal, side, origin, along, cross, name, fs, mark=0.25):
    """Draw real tick MARKS + labels + a centered axis name relative to ``origin`` (for composition on a
    shared axes). ``along`` = grid extent along this axis, ``cross`` = extent across it; ``side`` picks the
    edge (bottom|top for horizontal, left|right for vertical) so a wing's ruler can face outward."""
    ox, oy = origin
    if horizontal:
        top = side == "top"
        base = oy if top else oy + cross
        d = -mark if top else mark
        va = "bottom" if top else "top"
        for p, lab in ticks:
            ax.plot([ox + p, ox + p], [base, base + d], color="0.2", lw=0.7, zorder=8)
            ax.text(ox + p, base + d + (-0.2 if top else 0.2), lab, ha="center", va=va, fontsize=fs,
                    color="0.1")
        ax.text(ox + along / 2, base + d + (-1.05 if top else 1.05), name, ha="center", va=va,
                fontsize=fs + 1, weight="bold")
    else:
        right = side == "right"
        base = ox + cross if right else ox
        d = mark if right else -mark
        ha = "left" if right else "right"
        for p, lab in ticks:
            ax.plot([base, base + d], [oy + p, oy + p], color="0.2", lw=0.7, zorder=8)
            ax.text(base + d + (0.2 if right else -0.2), oy + p, lab, ha=ha, va="center", fontsize=fs,
                    color="0.1")
        ax.text(base + d + (1.2 if right else -1.2), oy + along / 2, name, ha=ha, va="center",
                rotation=90, fontsize=fs + 1, weight="bold")


# ---------------------------------------------------------------- structural composites
def render_cells_and_groups(ax, cells, *, pos_of, lane_of, shade_of, nsteps, color_mode, groups,
                            detailed, label_of, summary_of, highlight_cells, highlight_color,
                            origin=(0.0, 0.0), fs=7.0, cell_wh=(1.0, 1.0), palette=None,
                            detail_label_of=None):
    """The shared cell-grid + group-overlay used by every cell-field view. Fills every cell (hue =
    ``lane_of``, tint = ``shade_of`` under ``color_mode``), labels the ``detailed`` cells, borders each
    group's contiguous runs, adds each group's overlay by ``detail``, and outlines ``highlight_cells``.

    Group detail levels (the SAME machinery renders every scope):
      ``detailed`` -- cells already labelled above; border only.
      ``grouped``  -- a bordered ANCHOR cell stamped with its real coord + one derived summary label
                      (the WAVE view: one atom shown in full, the rest extrapolated from it).
      ``block``    -- a SOLID block: no inner grid lines, no anchor, just ONE centred name label
                      (the MACRO/macro view: a whole wave's tile at a glance).
      ``plain``    -- border only, NO labels/anchor (the macro LDS bank grid -- the wave is read from HUE).

    ``cells`` = iterable of cell keys; ``pos_of(cell)->(gx,gy)``; ``label_of(cell)->str``;
    ``summary_of(members)->str``; ``groups`` have ``.members``, ``.detail``, ``.name``."""
    ox, oy = origin
    block_members = set().union(*(frozenset(g.members) for g in groups if g.detail == "block")) \
        if any(g.detail == "block" for g in groups) else set()
    for cell in cells:
        gx, gy = pos_of(cell)
        # ``palette`` (a lane/wave -> RGB map) OVERRIDES the accent+tint model -- used by macro scope to give
        # each wave its OWN distinct colour instead of the 8-hue cycle.
        col = (palette(lane_of(cell)) if palette is not None
               else shade(lane_of(cell), shade_of.get(cell, 0), nsteps, color_mode=color_mode))
        # block cells fuse into a seamless solid (no inner grid); every other cell keeps the hairline grid
        fill(ax, ox + gx, oy + gy, col, edge=("none" if cell in block_members else "white"), lw=0.2)
        # Every cell-field label is FIT to the space it occupies (``fit_fs``) -- one consistent rule for all
        # detailed views (the MMA tee keeps its own sizing). A DETAILED per-cell label fits ONE cell, so it
        # scales down in a dense grid instead of overflowing into an unreadable overlap.
        if cell in detailed:
            lbl = (detail_label_of or label_of)(cell)          # detail labels may be 2-line (d0 / d1) to fit
            label(ax, ox + gx + 0.5, oy + gy + 0.5, lbl, fs=fit_fs(lbl, cell_wh, fs))
    cw, ch = cell_wh
    for g in groups:
        pts = {pos_of(c) for c in g.members}
        for comp in grid_components(pts):
            cxs = [p[0] for p in comp]; cys = [p[1] for p in comp]
            box(ax, ox + min(cxs), oy + min(cys), max(cxs) - min(cxs) + 1, max(cys) - min(cys) + 1,
                lw=1.8, zorder=4)
        if g.detail == "plain":                            # border only -- no labels, no anchor
            continue
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        x0, x1, y0, y1 = min(xs), max(xs) + 1, min(ys), max(ys) + 1
        bw, bh = (x1 - x0) * cw, (y1 - y0) * ch             # the box's PHYSICAL span (inches)
        if g.detail == "block":                            # solid block: ONE centred name, no anchor/grid
            rot = 90 if bh > 1.2 * bw else 0               # rotate the label to run ALONG a tall-thin band
            txt = g.name or summary_of(g.members)
            avail = (bh, bw) if rot else (bw, ch * 1.6)    # rotated -> text runs along the height; one line tall
            label(ax, ox + (x0 + x1) / 2, oy + (y0 + y1) / 2, txt, fs=fit_fs(txt, avail, fs + 2),
                  style="italic", weight="bold", zorder=5, rotation=rot)
        elif g.detail == "grouped":
            summ = summary_of(g.members)
            # A 1D box (a single row OR single column) is fully described by its range summary -- the anchor
            # cell would just repeat one endpoint, so skip it. Only a >=2D box keeps the anchor as its key coord.
            is_1d = (y1 - y0 == 1) or (x1 - x0 == 1)
            if not (is_1d and summ):
                anchor = min(g.members)                    # data anchor = lowest key, order-independent
                agx, agy = pos_of(anchor)
                box(ax, ox + agx, oy + agy, 1, 1, lw=2.4, zorder=6)
                albl = label_of(anchor)
                label(ax, ox + agx + 0.5, oy + agy + 0.5, albl, fs=fit_fs(albl, cell_wh, fs),
                      weight="bold", zorder=7)
            label(ax, ox + (x0 + x1) / 2, oy + (y0 + y1) / 2, summ, fs=fit_fs(summ, (bw, ch * 1.6), fs + 2),
                  style="italic", weight="bold", zorder=5)
    for cell in highlight_cells:
        gx, gy = pos_of(cell)
        box(ax, ox + gx, oy + gy, 1, 1, color=highlight_color, lw=3.4, zorder=12)


def bank_phase_grid(ax, phases, *, nbanks, color_of, y_top_first=True, box_lw=2.4, box_zorder=5,
                    text_zorder=6):
    """Draw a bank grid: for each served phase (a ``{bank: [items...]}`` dict, phase 0 first), fill each
    bank by its first item's colour (``color_of(item)->rgb``) and red-box + count any bank hit by >1 item.
    ``y_top_first`` puts phase 0 on the top row. ``box_lw``/``box_zorder``/``text_zorder`` tune the red
    conflict marker (the LDS-view and the conflict-view use slightly different weights). Returns the
    worst-case conflict multiplicity."""
    nphase = len(phases)
    worst = 1
    for p, occ in enumerate(phases):
        y = (nphase - 1 - p) if y_top_first else p
        for bank in range(nbanks):
            who = occ.get(bank, [])
            if not who:
                fill(ax, bank, y, EMPTY, edge="white", lw=0.3)
                continue
            fill(ax, bank, y, color_of(who[0]), edge="white", lw=0.3)
            if len(who) > 1:
                box(ax, bank, y, 1, 1, color="red", lw=box_lw, zorder=box_zorder)
                label(ax, bank + 0.5, y + 0.5, str(len(who)), fs=7, color="red", weight="bold",
                      zorder=text_zorder)
                worst = max(worst, len(who))
    return worst


def component_figure(comp, path, dpi=150):
    """Standalone-render scaffold shared by RegisterFileComponent / LogicalTileComponent: a right-sized
    figure with one axes, the component drawn + its ticks, saved. Duck-typed on the component's public
    surface (``grid_size``/``size``/``draw``/``_ticks``/``origin``/``cell_w``/``cell_h``/``title``/``font_size``)."""
    plt = _plt()
    nc, nr = comp.grid_size()
    w_in, h_in = comp.size()
    fig = plt.figure(figsize=(w_in + 1.4, h_in + 1.2))
    ax = fig.add_axes([0.10, 0.08, 0.86, 0.82])
    comp.draw(ax)
    ax.set_xlim(comp.origin[0], comp.origin[0] + nc)
    ax.set_ylim(comp.origin[1] + nr, comp.origin[1])       # origin top-left
    ax.set_aspect(comp.cell_h / comp.cell_w)
    comp._ticks(ax)
    if comp.title:
        ax.set_title(comp.title, fontsize=comp.font_size + 2)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path
