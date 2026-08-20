# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Graphical layout renderer -- one consistent visual language for tile/MMA layouts.

The graphical sibling of :mod:`layout_visualizer` (which is ASCII). A layout is a
:class:`~rocke.helpers.tiling.encoding.WarpDistributionEncoding`; every picture is DERIVED from it
through :class:`~rocke.helpers.tiling.register_mapper.RegisterMapper` (a proven bijection), so what is
drawn *is* the encoding. Two visual channels:

  * COLOUR = thread/lane identity (8 accents, cycled: ``lane % 8``).
  * TINT   = visit order (one shade per *vectorized transaction* = a maximal contiguous-address run).

Views: a **logical** matrix map (row down, col right), a **register** map (lane down, 32-bit VGPR
right, datatype-aware packing), a **macro** tile (dense full gamut), an **LDS bank** map (32 banks x
commit phases, conflicts flagged), and the encoding + flat level-major table as text.

matplotlib is imported lazily inside the drawing functions so importing this module (or the ASCII
reflection package) never requires matplotlib.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from math import log2
from types import SimpleNamespace

from ..encoding import WarpDistributionEncoding
from ..register_mapper import RegisterMapper
from ..transforms import (as_forward_map, derive_c_distribution, diagnose_k_match, mma_pair_compatible,
                          classify_transform)
from . import _canvas as cv
from ._canvas import (  # colour model + low-level helpers (moved to _canvas; re-exported here)
    ACCENTS, NACC, accent_tint, cell_rgb, _plt,
    grid_components as _grid_components, edge_ticks as _edge_ticks,
)

__all__ = [
    "ACCENTS", "NACC", "NLANES", "REG_CELL_WH",
    "accent_tint", "cell_rgb", "transactions",
    "derive_encoding", "flat_table", "flat_levels", "map_from_encoding",
    "invert", "wave_boxes", "compress", "clamp_aspect", "eff_ratio",
    "draw_logical", "draw_register", "draw_macro", "draw_lds_banks",
    "draw_encoding_ref", "draw_legend", "render_views",
    "draw_conflict_regfile", "draw_conflict_arrows", "draw_conflict_bankgrid",
    "draw_phase_legend", "render_conflict_dataflow",
    "CellFieldMixin", "CellGroup",
    "RegisterFileComponent", "RegGroup", "LogicalTileComponent", "LogicalGroup", "LdsBankView", "MmaTee",
    "FlowStage", "Pipeline", "WaveStrip", "transform_note", "flow_mem_to_register",
    "flow_lds_to_register", "flow_wave_mma", "flow_kloop_operand",
]

BLOCK_HUES = [(0.90, 0.36, 0.33), (0.06, 0.62, 0.83), (0.31, 0.65, 0.18),
              (0.63, 0.17, 0.58), (0.95, 0.62, 0.20), (0.09, 0.42, 0.14)]  # per output-block tint base

NLANES = 64
WAVE_MARKERS = ["o", "^", "s", "D", "P", "X", "*", "v"]
REG_CELL_WH = 3.4          # register-map cell width:height (cells wider than tall so labels are legible)

# Colour model (ACCENTS/NACC/accent_tint/cell_rgb) + _plt live in _canvas.py; imported above.

# ---------------------------------------------------------------- transactions (time channel)
def transactions(mp, addr_fn, order_by="reg"):
    """A 'vectorized transaction' = a maximal run of CONTIGUOUS addresses a lane touches (independent of
    register numbering). Ordered in TIME: ``order_by='reg'`` = issue order (smallest register first, how
    a LOAD fills regs); ``order_by='addr'`` = proper store order (lowest address first, how a coalescing
    store reorders the native accumulator). Returns ``({(lane,reg): tstep}, max_tsteps)``."""
    by_lane = {}
    for (lane, reg), (r, c) in mp.items():
        by_lane.setdefault(lane, []).append((addr_fn(r, c), reg))
    ts, maxt = {}, 1
    for lane, items in by_lane.items():
        items.sort()                                   # by address
        runs, cur = [], [items[0]]
        for a, reg in items[1:]:
            if a == cur[-1][0] + 1:
                cur.append((a, reg))
            else:
                runs.append(cur); cur = [(a, reg)]
        runs.append(cur)
        key = ((lambda i: min(a for a, _reg in runs[i])) if order_by == "addr"
               else (lambda i: min(reg for _a, reg in runs[i])))
        order = sorted(range(len(runs)), key=key)
        rank = {i: k for k, i in enumerate(order)}
        for i, run in enumerate(runs):
            for _a, reg in run:
                ts[(lane, reg)] = rank[i]
        maxt = max(maxt, len(runs))
    return ts, maxt

# ---------------------------------------------------------------- encoding <-> map
def _source_buckets(src_gens):
    """One source's bit-generators (each ``[src,p,dim,stride]``) -> merged buckets in P0/Y order
    (thread/register index MSB->LSB). Adjacent bits of the SAME dim with 2x-consecutive strides merge
    into one level; the merge is a contiguous bitfield so the decode is unchanged."""
    buckets = []
    for g in sorted(src_gens, key=lambda g: -g[1]):            # high bit -> low bit
        if buckets:
            b = buckets[-1]
            if b["dim"] == g[2] and g[3] * 2 == b["base_stride"]:
                b["base_stride"] = g[3]; b["length"] *= 2; b["pmin"] = min(b["pmin"], g[1]); continue
        buckets.append(dict(dim=g[2], base_stride=g[3], length=2, pmin=g[1]))
    return buckets

def derive_encoding(mp_tid):
    """EXACT ``{(tid,reg)->(row,col)}`` map -> ``WarpDistributionEncoding``. Assumes a linear (bijective,
    power-of-2) distribution rooted at (0,0): each thread/register bit shifts ONE matrix dim by a
    power-of-2 stride. Verified by reproduction via RegisterMapper. Returns ``(enc, Lbuckets, Ybuckets)``."""
    ntid = max(t for t, _ in mp_tid) + 1
    nreg = max(r for _, r in mp_tid) + 1
    if mp_tid[(0, 0)] != (0, 0):
        raise ValueError(f"layout not rooted at origin: (0,0)->{mp_tid[(0,0)]}")
    gens = []
    for p in range(int(log2(ntid)) if ntid > 1 else 0):
        dr, dc = mp_tid[(1 << p, 0)]
        if (dr == 0) == (dc == 0):
            raise ValueError(f"thread bit {p} is not single-dim (drow={dr}, dcol={dc}) -- non-linear layout")
        gens.append(["L", p, 0 if dr else 1, dr or dc])
    for p in range(int(log2(nreg)) if nreg > 1 else 0):
        dr, dc = mp_tid[(0, 1 << p)]
        if (dr == 0) == (dc == 0):
            raise ValueError(f"register bit {p} is not single-dim (drow={dr}, dcol={dc}) -- non-linear layout")
        gens.append(["Y", p, 0 if dr else 1, dr or dc])
    Lb = _source_buckets([g for g in gens if g[0] == "L"])
    Yb = _source_buckets([g for g in gens if g[0] == "Y"])
    Hs, loc = [[], []], {}
    for d in (0, 1):
        db = sorted([("L", i, b) for i, b in enumerate(Lb) if b["dim"] == d]
                    + [("Y", i, b) for i, b in enumerate(Yb) if b["dim"] == d],
                    key=lambda t: -t[2]["base_stride"])                       # outer(=big stride) -> inner
        for minor, (s, i, _b) in enumerate(db):
            Hs[d].append(db[minor][2]["length"]); loc[(s, i)] = (d + 1, minor)
    enc = WarpDistributionEncoding(
        replication_lengths=(),
        hierarchical_lengths=(tuple(Hs[0]), tuple(Hs[1])),
        lane_to_rh_major=(tuple(loc[("L", i)][0] for i in range(len(Lb))),),
        lane_to_rh_minor=(tuple(loc[("L", i)][1] for i in range(len(Lb))),),
        register_to_rh_major=tuple(loc[("Y", i)][0] for i in range(len(Yb))),
        register_to_rh_minor=tuple(loc[("Y", i)][1] for i in range(len(Yb))))
    rm = RegisterMapper(enc)
    for (t, rg), rc in mp_tid.items():
        if rm.matrix_coordinates(t, rg) != rc:
            raise ValueError(f"derived encoding does not reproduce (tid={t},reg={rg})")
    return enc, Lb, Yb

def _addr_stride(addr_fn, dim, st):
    base = [0, 0]; base[dim] = st
    return addr_fn(base[0], base[1]) - addr_fn(0, 0)

def flat_table(Lb, Yb, addr_fn):
    """Flat LEVEL-MAJOR table from :func:`derive_encoding` bucket lists. Each level = a positional
    ``(row_len, col_len)`` tuple (1 = level skips that axis) tagged with its consumer. Register level:
    addr stride 1 = ``vector`` else ``steps``. Thread level: thread bit < 64 = ``lane`` else ``wave``.
    Ordered ``vector < steps < lane < wave``."""
    rows = []
    for b in Yb:
        L = (b["length"], 1) if b["dim"] == 0 else (1, b["length"])
        st = _addr_stride(addr_fn, b["dim"], b["base_stride"])
        rows.append(("vector" if abs(st) == 1 else "steps", L, st))
    for b in Lb:
        L = (b["length"], 1) if b["dim"] == 0 else (1, b["length"])
        st = _addr_stride(addr_fn, b["dim"], b["base_stride"])
        rows.append(("wave" if (1 << b["pmin"]) >= NLANES else "lane", L, st))
    rank = {"vector": 0, "steps": 1, "lane": 2, "wave": 3, "broadcast": 4}
    rows.sort(key=lambda r: rank[r[0]])
    return rows

def flat_levels(enc, addr_fn):
    """Flat LEVEL-MAJOR table directly from ANY ``WarpDistributionEncoding`` (walks the P/Y buckets),
    so the encoding panel works for TileMma-produced encodings, not just :func:`derive_encoding` output.
    Same row schema and ordering as :func:`flat_table`."""
    def coord_stride(dim, minor):
        levels = enc.hierarchical_lengths[dim]
        s = 1
        for lvl in range(minor + 1, len(levels)):
            s *= levels[lvl]
        return s
    rows = []
    for maj, mn in zip(enc.register_to_rh_major, enc.register_to_rh_minor):
        if maj == 0:                                             # replication read into a register (rare)
            rows.append(("broadcast", (1, 1), 0)); continue
        dim = maj - 1; length = enc.hierarchical_lengths[dim][mn]
        st = _addr_stride(addr_fn, dim, coord_stride(dim, mn))
        L = (length, 1) if dim == 0 else (1, length)
        rows.append(("vector" if abs(st) == 1 else "steps", L, st))
    p_major = enc.lane_to_rh_major[0] if enc.lane_to_rh_major else ()
    p_minor = enc.lane_to_rh_minor[0] if enc.lane_to_rh_minor else ()
    lane_lengths = [enc.bucket_length(maj, mn) for maj, mn in zip(p_major, p_minor)]
    for j, (maj, mn) in enumerate(zip(p_major, p_minor)):
        place = 1
        for k in range(j + 1, len(lane_lengths)):
            place *= lane_lengths[k]
        if maj == 0:
            rows.append(("broadcast", (1, 1), 0)); continue
        dim = maj - 1; length = enc.hierarchical_lengths[dim][mn]
        st = _addr_stride(addr_fn, dim, coord_stride(dim, mn))
        L = (length, 1) if dim == 0 else (1, length)
        rows.append(("wave" if place >= NLANES else "lane", L, st))
    rank = {"vector": 0, "steps": 1, "lane": 2, "wave": 3, "broadcast": 4}
    rows.sort(key=lambda r: rank[r[0]])
    return rows

def map_from_encoding(enc, gm=1, gn=1):
    """``WarpDistributionEncoding`` -> ``{(lane, reg, wave): (row, col)}``. tid splits into wave=tid//64,
    lane=tid%64 (cooperative encodings span >64 threads); a per-wave encoding may be replicated across a
    ``gm x gn`` wave grid (each wave the same internal layout at its tile origin)."""
    rm = RegisterMapper(enc)
    base = {}
    for tid in range(rm.num_lanes):
        for reg in range(rm.num_vector_items):
            base[(tid % NLANES, reg, tid // NLANES)] = rm.matrix_coordinates(tid, reg)
    if (gm, gn) == (1, 1):
        return base
    tile_r = max(r for r, _ in base.values()) + 1
    tile_c = max(c for _, c in base.values()) + 1
    nw = len({w for (_l, _r, w) in base})
    out = {}
    for (l, r, w), (rr, cc) in base.items():
        for wm in range(gm):
            for wn in range(gn):
                out[(l, r, (wm * gn + wn) * nw + w)] = (rr + wm * tile_r, cc + wn * tile_c)
    return out

# ---------------------------------------------------------------- small helpers
def invert(mp):
    """``(lane,reg,wave)->(row,col)`` ==> ``(row,col)->(lane,reg,wave)``."""
    return {rc: (l, r, w) for (l, r, w), rc in mp.items()}

def compress(nums):
    """``[0,1,2,3,16,17]`` -> ``'0-3,16-17'``."""
    nums = sorted(set(nums)); out = []; i = 0
    while i < len(nums):
        j = i
        while j + 1 < len(nums) and nums[j + 1] == nums[j] + 1:
            j += 1
        out.append(str(nums[i]) if i == j else f"{nums[i]}-{nums[j]}")
        i = j + 1
    return ",".join(out)

def _rng(vals):
    """Coord span as a compact string: '5' (single) or '0-3' (a run)."""
    lo, hi = min(vals), max(vals)
    return f"{lo}" if lo == hi else f"{lo}-{hi}"

def clamp_aspect(ax, ny, nx, maxr=6.0):
    """Proportional (equal) unless taller than ``maxr:1`` -- then clamp so cells/labels stay legible."""
    ax.set_aspect(1.0 if ny <= maxr * nx else maxr * nx / ny)

def eff_ratio(nx, ny, maxr=6.0):
    """Displayed width:height a panel takes under :func:`clamp_aspect` -- used to size panel columns so
    content fills its box (no wasted inter-panel whitespace)."""
    a = 1.0 if ny <= maxr * nx else maxr * nx / ny
    return nx / (ny * a)

def wave_boxes(inv):
    """wave -> ``[row_min, row_max, col_min, col_max]`` bounding box of that wave's footprint."""
    box = {}
    for (r, c), (l, reg, w) in inv.items():
        b = box.setdefault(w, [r, r, c, c])
        b[0] = min(b[0], r); b[1] = max(b[1], r); b[2] = min(b[2], c); b[3] = max(b[3], c)
    return box

def logical_cells(mp, addr_fn, max_rows, max_cols, order_by="reg"):
    """``{(lane,reg,wave):(r,c)}`` -> draw cells ``(col,row,lane,tstep,ntsteps,wave)`` + max tsteps."""
    ts, maxt = transactions({(l, r): rc for (l, r, w), rc in mp.items()}, addr_fn, order_by=order_by)
    cells = []
    for (lane, reg, wave), (row, col) in mp.items():
        if row < max_rows and col < max_cols:
            cells.append((col, row, lane, ts[(lane, reg)], maxt, wave))
    return cells, maxt

# ---------------------------------------------------------------- panel drawers (ax-based)
def _draw_grid(ax, cells, ncols, nrows, xlabel, ylabel, title, edge=True):
    for x, y, lane, tstep, ntsteps, _wave in cells:
        cv.fill(ax, x, y, cell_rgb(lane, tstep, ntsteps),
                edge="white" if edge else "none", lw=0.4 if edge else 0)
    cv.grid_limits(ax, ncols, nrows)               # invert y -> origin top-left, aspect equal
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title, fontsize=9)
    cv.sparse_ticks(ax, ncols, nrows)

def draw_logical(ax, inv, ts, maxt, r0, r1, c0, c1, title, dims=("M", "K"), fontsize=8):
    """One WAVE TILE, zoomed. Each thread's vectorized transaction = ONE joined cell, labelled ``T{n}``
    (or ``T{n}[{step}]``), left-aligned monospace. Only the FIRST COLOUR CYCLE (threads 0..7) is
    colour-coded (tinted by step); other threads stay neutral. Full gamut lives in the macro view."""
    groups = {}
    for r in range(r0, r1):
        for c in range(c0, c1):
            if (r, c) in inv:
                lane, reg, _ = inv[(r, c)]
                groups.setdefault((lane, ts[(lane, reg)]), []).append((r, c))
    for (lane, t), cs in groups.items():
        rs = [r for r, _ in cs]; ks = [c for _, c in cs]
        rr0, rr1, kk0, kk1 = min(rs), max(rs) + 1, min(ks), max(ks) + 1
        hl = lane < NACC                                   # first colour cycle only
        cv.fill(ax, kk0 - c0, rr0 - r0, cv.shade(lane, t, maxt), w=kk1 - kk0, h=rr1 - rr0,
                edge=("black" if hl else "white"), lw=1.4 if hl else 1.0)
        lbl = f"T{lane}" if maxt <= 1 else f"T{lane}[{t}]"
        cv.label(ax, kk0 - c0 + 0.06, (rr0 + rr1) / 2 - r0, lbl, fs=fontsize, ha="left",
                 family="monospace", weight="bold" if hl else "normal")
    ax.set_xlim(0, c1 - c0); ax.set_ylim(r1 - r0, 0)
    clamp_aspect(ax, r1 - r0, c1 - c0)
    ax.set_title(title + "\nT# = thread, [#] = step; first cycle T0-7 coloured", fontsize=9)
    ax.set_xlabel(f"col = {dims[1]}"); ax.set_ylabel(f"row = {dims[0]}")
    ax.set_xticks(range(0, c1 - c0 + 1, max(1, (c1 - c0) // 4)))
    ax.set_yticks(range(0, r1 - r0 + 1, max(1, (r1 - r0) // 8)))

def draw_macro(ax, mp, addr_fn, rows, cols, title, order_by="reg", dims=("M", "K")):
    """Dense full-resolution render: colour = ACCENTS[lane % 8], tint = visit order; each WAVE bordered."""
    cells, _maxt = logical_cells(mp, addr_fn, rows, cols, order_by=order_by)
    _draw_grid(ax, cells, cols, rows, f"col = {dims[1]}", f"row = {dims[0]}", title, edge=False)
    for _w, b in wave_boxes(invert(mp)).items():
        cv.box(ax, b[2], b[0], b[3] - b[2] + 1, b[1] - b[0] + 1, lw=1.8)
    clamp_aspect(ax, rows, cols)

def draw_register(ax, mp_wave, ts, maxt, title, dims=("M", "K"), dtype_bits=16, fontsize=None):
    """WAVE-TILE -> REGISTER map: x = 32-bit VGPR (vector/issue order), y = lane. Registers grouped by
    VECTORIZED TRANSACTION (a contiguous-memory run), NOT raw register index, so a store vector scattered
    across non-adjacent native-ACC registers still reads as one block. Block width in VGPRs is set by the
    datatype pack (f16 -> 2 elem/VGPR). Only the first colour cycle (lanes 0..7) is colour-coded."""
    lanes = sorted({l for l, _ in mp_wave})
    nreg = max(r for _, r in mp_wave) + 1
    ew = dtype_bits / 32.0                                  # element width in VGPRs
    nvgpr = nreg * ew
    fs = fontsize if fontsize is not None else max(5.0, min(9.0, 42.0 / max(1.0, nvgpr) ** 0.5))
    blocks = []                                            # (lane, x, w, t, hl, rstr, cstr)
    for lane in lanes:
        hl = lane < NACC
        groups = {}                                        # tstep -> [reg,...] (one vectorized transaction)
        for reg in range(nreg):
            if (lane, reg) in mp_wave:
                groups.setdefault(ts[(lane, reg)], []).append(reg)
        x = 0.0
        for t in sorted(groups):
            coords = [mp_wave[(lane, r)] for r in sorted(groups[t], key=lambda r: mp_wave[(lane, r)])]
            w = len(coords) * ew
            blocks.append((lane, x, w, t, hl, _rng([c[0] for c in coords]), _rng([c[1] for c in coords])))
            x += w
    rw = max(len(b[5]) for b in blocks); cw = max(len(b[6]) for b in blocks)
    for lane, x, w, t, hl, rstr, cstr in blocks:
        cv.fill(ax, x, lane, cv.shade(lane, t, maxt), w=w, h=1, edge="white", lw=0.4)
        cv.label(ax, x + w / 2, lane + 0.5, f"{dims[0]}{rstr:<{rw}} {dims[1]}{cstr:<{cw}}",
                 fs=fs, family="monospace", weight="bold" if hl else "normal")
    for v in range(1, int(round(nvgpr))):
        ax.axvline(v, color="0.6", lw=0.25, zorder=0)
    ax.set_xlim(0, nvgpr); ax.set_ylim(len(lanes), 0)
    ax.set_aspect(1.0 / REG_CELL_WH)
    pack = max(1, int(round(1.0 / ew)))
    ax.set_title(title + f"\n{dtype_bits}-bit, {pack} elem/VGPR", fontsize=9)
    ax.set_xlabel("32-bit VGPR (vector / issue order)"); ax.set_ylabel("lane")
    nv = int(round(nvgpr))
    ax.set_xticks(range(0, nv + 1, max(1, nv // 16)))
    ax.set_yticks(range(0, len(lanes) + 1, max(1, len(lanes) // 8)))

def draw_lds_banks(ax, mp_wave, addr_fn, title, dtype_bits=16):
    """LDS BANK map: 32 banks (x, 4-byte words) x commit phases (y, 32 dwords each). Store order is
    lane-major; ``bank = dword_addr % 32`` with ``dword_addr = elem_addr * dtype_bits/32``. A cell is
    coloured by its (first) writing lane, tinted by register issue order; a bank hit by >1 write in the
    same phase is red-bordered with the conflict count. Returns the worst-case conflict way."""
    lanes = sorted({l for l, _ in mp_wave})
    nreg = max(r for _, r in mp_wave) + 1
    dpe = dtype_bits / 32.0
    flat = []                                              # (bank, lane, reg)
    for lane in lanes:
        for reg in range(nreg):
            if (lane, reg) in mp_wave:
                r, c = mp_wave[(lane, reg)]
                flat.append((int(addr_fn(r, c) * dpe) % 32, lane, reg))
    nphase = max(1, (len(flat) + 31) // 32)
    phases = []                                            # per-phase {bank: [(lane,reg),...]}
    for p in range(nphase):
        occ = {}
        for bank, lane, reg in flat[p * 32:(p + 1) * 32]:
            occ.setdefault(bank, []).append((lane, reg))
        phases.append(occ)
    worst = cv.bank_phase_grid(ax, phases, nbanks=32,
                               color_of=lambda lr: cell_rgb(lr[0], lr[1], nreg),
                               box_lw=1.6, box_zorder=1, text_zorder=3)
    ax.set_xlim(0, 32); ax.set_ylim(0, nphase)
    clamp_aspect(ax, nphase, 32)
    ax.set_title(title + f"\nworst {worst}-way conflict (red)", fontsize=9)
    ax.set_xlabel("LDS bank (0..31)"); ax.set_ylabel("commit phase (32 dwords)")
    ax.set_xticks(range(0, 33, 4)); ax.set_yticks([])
    return worst

# ---------------------------------------------------------------- text panels
def draw_encoding_ref(ax, enc, flat, dim_names=("row", "col")):
    """Text panel: the COMPLETE ``WarpDistributionEncoding`` (raw six fields, no derived items) plus the
    flat level-major table -- the source-of-truth reference the pictures are rendered from."""
    ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    mono = dict(family="monospace", fontsize=8.5, va="top")
    ax.text(0.0, 1.0, "COMPLETE STATIC TILE DISTRIBUTION  (WarpDistributionEncoding -- raw fields)",
            weight="bold", fontsize=9.5, va="top")
    lines = [
        f"replication_lengths   Rs = {enc.replication_lengths}",
        f"hierarchical_lengths  Hs = {enc.hierarchical_lengths}",
        f"lane_to_rh_major   Ps.maj = {enc.lane_to_rh_major}",
        f"lane_to_rh_minor   Ps.min = {enc.lane_to_rh_minor}",
        f"register_to_rh_major Y.maj = {enc.register_to_rh_major}",
        f"register_to_rh_minor Y.min = {enc.register_to_rh_minor}",
    ]
    for i, ln in enumerate(lines):
        ax.text(0.0, 0.90 - i * 0.075, ln, **mono)
    ax.text(0.0, 0.40, "flat level-major table  (level = (row_len, col_len); 1 = axis skipped)",
            weight="bold", fontsize=9, va="top")
    ax.text(0.0, 0.335, f"{'consumer':10s}{'(%s,%s) len' % dim_names:>14s}{'addr stride':>14s}",
            **dict(mono, weight="bold"))
    for i, (cons, L, astr) in enumerate(flat):
        ax.text(0.0, 0.275 - i * 0.052, f"{cons:10s}{str(L):>14s}{astr:>14d}", **mono)

def draw_legend(ax, ntsteps):
    """Stacked vertically (COLOUR ramp / TINT ramp / notes) so headers never collide in a narrow panel."""
    from matplotlib.patches import Rectangle
    ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.text(0.0, 0.94, "COLOUR = lane % 8", fontsize=9, weight="bold")
    for ai in range(NACC):
        ax.add_patch(Rectangle((0.02 + 0.058 * ai, 0.78), 0.05, 0.10, facecolor=ACCENTS[ai],
                               edgecolor="white", lw=0.5))
        ax.text(0.02 + 0.058 * ai + 0.025, 0.73, str(ai), ha="center", fontsize=7)
    ax.text(0.0, 0.60, "TINT = visit order  (dark = 1st  ->  pale = later)", fontsize=9, weight="bold")
    demo = max(4, ntsteps)
    for t in range(demo):
        ax.add_patch(Rectangle((0.02 + 0.30 * t / demo, 0.46), 0.30 / demo, 0.10,
                               facecolor=accent_tint(0, t, demo), lw=0.3, edgecolor="white"))
    ax.text(0.02, 0.40, "t0", fontsize=7); ax.text(0.30, 0.40, "later", fontsize=7)
    ax.text(0.0, 0.26, "notes", fontsize=9, weight="bold")
    ax.text(0.0, 0.17, "* details: only the first colour cycle (lanes 0-7) is coloured", fontsize=7.5)
    ax.text(0.0, 0.09, "* macro: full lane%8 gamut, each wave bordered", fontsize=7.5)
    ax.text(0.0, 0.01, "* register block = one vectorized transaction (its coord span)", fontsize=7.5)

# ---------------------------------------------------------------- Part 1: per-distribution views
_VIEW_TITLES = {"logical": "LOGICAL DETAIL", "macro": "MACRO TILE",
                "register": "REGISTER DETAIL", "lds": "LDS BANKS"}

def render_views(enc, axes=("M", "K"), *, views=("logical", "register", "lds"), dtype_bits=16,
                 layout="col", order_by="reg", replicate=(1, 1), out_dir=".", name="layout",
                 combined=True, title=None, note=""):
    """Render a static distribution in the requested views (``logical``/``macro``/``register``/``lds``),
    each as its own PNG and (default) a combined figure with the encoding + legend on top. ``axes`` labels
    the two matrix dims (A=("M","K"), B=("N","K"), C=("M","N")); ``layout`` ('col'|'row') sets the memory
    order used for load/store ordering + LDS banking. Returns the list of written PNG paths."""
    plt = _plt()
    os.makedirs(out_dir, exist_ok=True)
    gm, gn = replicate
    mp = map_from_encoding(enc, gm, gn)
    rows = max(r for r, _ in mp.values()) + 1
    cols = max(c for _, c in mp.values()) + 1
    addr = (lambda r, c: r + c * rows) if layout == "col" else (lambda r, c: r * cols + c)
    inv = invert(mp)
    ts, maxt = transactions({(l, r): rc for (l, r, w), rc in mp.items()}, addr, order_by=order_by)
    flat = flat_levels(enc, addr)
    dwave = min(wave_boxes(inv)); d = wave_boxes(inv)[dwave]
    mp_wave = {(l, r): rc for (l, r, w), rc in mp.items() if w == dwave}
    ts_p, maxt_p = transactions(mp_wave, addr, order_by=order_by)
    nlanes_w = len({l for l, _ in mp_wave})
    nreg_w = max(r for _, r in mp_wave) + 1
    ew = dtype_bits / 32.0

    def panel(ax, view):
        if view == "logical":
            draw_logical(ax, inv, ts, maxt, d[0], d[1] + 1, d[2], d[3] + 1,
                         _VIEW_TITLES[view], dims=axes)
        elif view == "macro":
            draw_macro(ax, mp, addr, rows, cols, _VIEW_TITLES[view], order_by=order_by, dims=axes)
        elif view == "register":
            draw_register(ax, mp_wave, ts_p, maxt_p, _VIEW_TITLES[view], dims=axes, dtype_bits=dtype_bits)
        elif view == "lds":
            draw_lds_banks(ax, mp_wave, addr, _VIEW_TITLES[view], dtype_bits=dtype_bits)
        else:
            raise ValueError(f"unknown view {view!r} (choose from {sorted(_VIEW_TITLES)})")

    def view_ratio(view):
        if view == "logical":
            return eff_ratio(d[3] - d[2] + 1, d[1] - d[0] + 1)
        if view == "macro":
            return eff_ratio(cols, rows)
        if view == "register":
            return nreg_w * ew * REG_CELL_WH / nlanes_w
        if view == "lds":
            nphase = max(1, (len(mp_wave) + 31) // 32)
            return eff_ratio(32, nphase)
        return 1.0

    paths = []
    for view in views:                                   # individual PNGs
        fig, ax = plt.subplots(figsize=(9, 9))
        panel(ax, view)
        p = os.path.join(out_dir, f"{name}_{view}.png")
        fig.savefig(p, dpi=200, bbox_inches="tight"); plt.close(fig); paths.append(p)

    if combined:                                         # combined figure (encoding + legend on top)
        wr = [view_ratio(v) for v in views]
        fig_h = 14.0
        panel_h = fig_h * 10.0 / 12.4
        fig_w = max(15.0, sum(wr) * panel_h + 2.0)
        fig = plt.figure(figsize=(fig_w, fig_h))
        outer = fig.add_gridspec(2, 1, height_ratios=[2.4, 10], hspace=0.08)
        top = outer[0].subgridspec(1, 2, width_ratios=[1.5, 1.0], wspace=0.10)
        bot = outer[1].subgridspec(1, len(views), width_ratios=wr, wspace=0.06)
        draw_encoding_ref(fig.add_subplot(top[0]), enc, flat, dim_names=axes)
        draw_legend(fig.add_subplot(top[1]), ntsteps=maxt)
        for i, v in enumerate(views):
            ax = fig.add_subplot(bot[i])
            panel(ax, v)
            ax.set_title(f"{i + 1}) " + ax.get_title(), fontsize=9)
        fig.suptitle(title or f"{name} -- " + " + ".join(views), fontsize=12, weight="bold", y=0.995)
        if note:
            fig.text(0.5, 0.965, note, ha="center", fontsize=8.5, style="italic")
        p = os.path.join(out_dir, f"{name}_combined.png")
        fig.savefig(p, dpi=200, bbox_inches="tight"); plt.close(fig); paths.append(p)
    return paths

# ---------------------------------------------------------------- Part 2: MMA tee sheet
def _reg_fs(nx, ny):
    return max(4.5, min(11.0, 82.0 / max(nx, ny) ** 0.6))

def _owner2(enc):
    """``{(coord0, coord1): (lane, reg)}`` (first owner) + (extent0, extent1) over the whole wave."""
    rm = RegisterMapper(enc); owner = {}; e0 = e1 = 0
    for lane in range(rm.num_lanes):
        for reg in range(rm.num_vector_items):
            c = rm.matrix_coordinates(lane, reg)
            owner.setdefault((c[0], c[1]), (lane, reg))
            e0 = max(e0, c[0] + 1); e1 = max(e1, c[1] + 1)
    return owner, e0, e1, rm.num_vector_items

def _issue_seq(m_iter, n_iter, issue_order):
    """Ordinal -> (mi, nj) tile in MMA issue order. A-major = M outer (issue a full C row first);
    B-major = N outer (issue a full C column first)."""
    if issue_order == "B":
        return [(mi, nj) for nj in range(n_iter) for mi in range(m_iter)]
    return [(mi, nj) for mi in range(m_iter) for nj in range(n_iter)]

def _atom_c_fragment(c_rm, atom_m, atom_n):
    """Single-atom ACC fragment read from the FIRST C tile: ``{(lm, ln): (lane, atom_reg)}`` where atom_reg
    is the rank of the owning register within that tile (per lane). Also returns registers-per-atom. This is
    the hardware-invariant atom shape; the wave layout is DERIVED by tiling it in issue order."""
    by_lane = {}
    for lane in range(c_rm.num_lanes):
        for reg in range(c_rm.num_vector_items):
            m, n = c_rm.matrix_coordinates(lane, reg)
            if m < atom_m and n < atom_n:
                by_lane.setdefault(lane, []).append((reg, m, n))
    frag = {}
    for lane, lst in by_lane.items():
        for ar, (_reg, m, n) in enumerate(sorted(lst)):
            frag[(m, n)] = (lane, ar)
    return frag, max((len(v) for v in by_lane.values()), default=1)


# ================================================================================================
# COMPOSABLE VISUAL COMPONENTS (new layer -- migration in progress; the drawers/composites above will
# be recomposed on top of these). Each component is self-sizing, renders standalone OR draws onto a
# shared axes at ``origin`` for composition. The distribution (WarpDistributionEncoding) is the single
# source of truth; every cell is filled via RegisterMapper (the proven bijection).
# ================================================================================================

def _compact_1d(vals):
    """Compact a set of ints to a formula from its CONTENTS: 'a' (singleton) | 'a-b' (contiguous) |
    'a..b/d' (constant stride d) | '[a-b]xc/S' (c contiguous width-(b-a+1) blocks at block-stride S --
    the 2-level pattern interleaved atoms produce) | None (no closed form -> caller renders CUSTOM)."""
    s = sorted(set(vals))
    if len(s) == 1:
        return str(s[0])
    if s == list(range(s[0], s[-1] + 1)):
        return f"{s[0]}-{s[-1]}"
    d = s[1] - s[0]
    if d > 0 and all(s[i] - s[i - 1] == d for i in range(1, len(s))):
        return f"{s[0]}..{s[-1]}/{d}"
    # 2-level: c equal-width contiguous blocks at a uniform block-stride (interleaved atom owners/coords)
    w = 1
    while w < len(s) and s[w] == s[0] + w:
        w += 1
    if w > 1 and len(s) % w == 0:
        c = len(s) // w
        stride = s[w] - s[0]
        if stride > w and all(
            s[i * w + j] == s[0] + i * stride + j for i in range(c) for j in range(w)
        ):
            return f"[{s[0]}-{s[0] + w - 1}]x{c}/{stride}"
    return None


# _edge_ticks and _grid_components moved to _canvas.py (imported above as _edge_ticks/_grid_components).


# ---------------------------------------------------------------- LDS bank-conflict dataflow
# The 3-panel register->LDS conflict figure lives HERE (the one visual language), not in the
# analysis module -- so the drawing is machine/model-independent. The analysis module
# (`lds_conflict.py`) prepares the VALIDATED data + gates it, then calls `render_conflict_dataflow`.
#
# Data contract: `datum` = {(lane, phase) -> (K, free, dword, bank)} for a store (see
# `lds_conflict.store_datum`); `highlight` = the lanes piling on one bank (the located collision);
# `fix_bank_fn(lane) -> bank` = the conflict-free permutation for the FIXED row.

def draw_conflict_regfile(ax, datum, *, half, nreg, highlight, wtag):
    """Panel 1: register file (lane x reg), cell hue = lane, the colliding lanes' reg0 boxed red."""
    for lane in range(half):
        for reg in range(nreg):
            K = datum[(lane, reg)][0]
            hl = lane in highlight and reg == 0
            cv.fill(ax, reg, half - 1 - lane, cell_rgb(lane, reg, nreg),
                    edge="red" if hl else "white", lw=2.2 if hl else 0.3, zorder=3 if hl else 1)
            if nreg <= 4:
                cv.label(ax, reg + 0.5, half - 1 - lane + 0.5, f"K{K}", fs=4.5, color="white")
    ax.set_xlim(0, nreg)
    ax.set_ylim(0, half)
    ax.set_xlabel("reg (dword in op)")
    ax.set_ylabel(f"lane (half-wave 0: T0..T{half - 1})")
    ax.set_title(f"(1) register file  {wtag} store\nred = the {len(highlight)} lanes that collide",
                 fontsize=8)
    ax.set_xticks([x + 0.5 for x in range(nreg)])
    ax.set_xticklabels(range(nreg), fontsize=6)
    ax.set_yticks([0.5, half / 2 - 0.5, half - 0.5])
    ax.set_yticklabels([f"T{half - 1}", f"T{half // 2}", "T0"], fontsize=6)


def draw_conflict_arrows(ax, datum, *, shown_lanes, nbanks, fixed, fix_bank_fn, subject_bank=0,
                         cpa=None):
    """Panel 2: funnel arrows T{lane}R0 -> dword -> bank for the SAME `shown_lanes` in both rows.
    Conflicted: those lanes pile onto a handful of banks (several arrows converge per bank). Fixed:
    the same lanes fan out to DISTINCT banks (conflict-free). Bank labels are drawn once per bank."""
    from matplotlib.patches import FancyArrowPatch
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 34)
    ax.axis("off")
    ax.text(1.2, 33, "T{lane}R0", fontsize=7, ha="center", weight="bold")
    ax.text(8.6, 33, "LDS bank", fontsize=7, ha="center", weight="bold")
    # Arrows live in y=[TOP..BOT]; the 2-line caption sits BELOW that band so it never overlaps the
    # bottom-most arrow.
    TOP, BOT = 31.0, 7.0

    def dest(lane):
        return fix_bank_fn(lane) if fixed else datum[(lane, 0)][3]

    # order sources by destination bank so converging groups sit together (fewer crossings)
    lanes = sorted(shown_lanes, key=lambda l: (dest(l), l))
    n = len(lanes)
    ybank = {}
    for i, lane in enumerate(lanes):
        b = dest(lane)
        d = datum[(lane, 0)][2]
        y0 = TOP - i * ((TOP - BOT) / (n - 1)) if n > 1 else (TOP + BOT) / 2
        yb = TOP - (b / nbanks) * (TOP - BOT)
        ybank[b] = yb
        col = cell_rgb(lane, 0, 1)
        ax.text(1.0, y0, f"T{lane}", fontsize=6.0, ha="center", va="center", color=col, weight="bold")
        ax.text(4.6, y0, f"d={d}", fontsize=5.0, ha="center", va="center", color="#444")
        ax.add_patch(FancyArrowPatch((1.7, y0), (7.7, yb), arrowstyle="-|>", mutation_scale=6,
                                     lw=1.0, color=col, alpha=0.8, zorder=2))
    for b, yb in ybank.items():
        ax.text(8.6, yb, f"bank {b}", fontsize=6, ha="center", va="center",
                color="#1a7" if fixed else "red", weight="bold")
    nbank = len(ybank)
    nway = 1 if fixed else max(sum(1 for l in lanes if dest(l) == b) for b in ybank)
    cost = f" -> {cpa:.2f} conflicts/access" if cpa is not None else ""
    if fixed:
        ax.text(5, 2.2, f"same {n} threads -> {nbank} DISTINCT banks{cost or '  ->  0-way'}",
                ha="center", fontsize=7, color="#1a7", weight="bold")
    else:
        ax.text(5, 2.2, f"{n} threads pile {nway} deep onto {nbank} bank(s){cost}",
                ha="center", fontsize=7, color="red", weight="bold")
    ax.set_title("(2) T{lane}R{reg} -> dword -> bank", fontsize=8)


def draw_conflict_bankgrid(ax, datum, *, half, nbanks, nreg, fixed, fix_bank_fn, subject_bank=0,
                           cpa=None):
    """Panel 3: LDS bank grid (x = bank, one served group). Cell hue = first lane landing there; a
    RED BOX + N-way count on any bank >1 lane hits (conflicted row)."""
    occ = {}
    for lane in range(half):
        bank = fix_bank_fn(lane) if fixed else datum[(lane, 0)][3]
        occ.setdefault(bank, []).append(lane)
    cv.bank_phase_grid(ax, [occ], nbanks=nbanks, color_of=lambda l: cell_rgb(l, 0, nreg))
    ax.set_xlim(0, nbanks)
    ax.set_ylim(-0.5, 1.5)
    ax.set_xlabel(f"LDS bank (0..{nbanks - 1})")
    ax.set_yticks([])
    nway = max((len(v) for v in occ.values()), default=1)
    banks_used = len(occ)
    cost = f" -> {cpa:.2f} conflicts/access" if cpa is not None else ""
    lab = ("FIXED: banks all distinct -> 0 conflicts/access" if fixed
           else f"CONFLICTED: {nway}-deep piles across {banks_used} banks{cost}")
    ax.set_title(f"(3) LDS bank grid (half-wave 0, phase 0)\n{lab}", fontsize=8)
    ax.set_xticks(range(0, nbanks + 1, 4))


def draw_phase_legend(fig, nreg, *, base_accent=5):
    """Figure-level legend for the register-file SHADE channel: shade = store phase (the dword-in-op
    issue order), darkest = phase 0 = issued first. `nreg` swatches, one representative accent tinted
    across phases (the same tint schedule every lane uses)."""
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=accent_tint(base_accent, k, nreg), edgecolor="0.4",
                     label=f"phase {k} (d{k})") for k in range(nreg)]
    fig.legend(handles=handles, loc="lower center", ncol=nreg, fontsize=7, frameon=True,
               bbox_to_anchor=(0.5, -0.015),
               title="register-file shade = store phase within the op "
                     "(dwords issued in order; darkest = first)", title_fontsize=7)


def render_conflict_dataflow(out_path, *, datum, shown_lanes, half, nreg, nbanks, fix_bank_fn, wtag,
                             suptitle, subject_bank=0, cpa=None):
    """Compose the two-row (conflicted | fixed) x 3-panel register->LDS conflict figure and save it.
    Pure drawing: the caller (`lds_conflict.render_conflict_3panel`) supplies VALIDATED, gated data.
    `shown_lanes` is the SAME representative thread set drawn in both rows (piled on top, fanned out
    conflict-free on the bottom); `subject_bank` is the representative colliding bank. Returns out_path."""
    plt = _plt()
    fig = plt.figure(figsize=(15, 8.6))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.1, 1.3, 2.2], height_ratios=[1, 1],
                          hspace=0.42, wspace=0.32)
    draw_conflict_regfile(fig.add_subplot(gs[0, 0]), datum, half=half, nreg=nreg,
                          highlight=shown_lanes, wtag=wtag)
    draw_conflict_arrows(fig.add_subplot(gs[0, 1]), datum, shown_lanes=shown_lanes, nbanks=nbanks,
                         fixed=False, fix_bank_fn=fix_bank_fn, subject_bank=subject_bank, cpa=cpa)
    draw_conflict_bankgrid(fig.add_subplot(gs[0, 2]), datum, half=half, nbanks=nbanks, nreg=nreg,
                           fixed=False, fix_bank_fn=fix_bank_fn, subject_bank=subject_bank, cpa=cpa)
    draw_conflict_regfile(fig.add_subplot(gs[1, 0]), datum, half=half, nreg=nreg, highlight=[],
                          wtag=wtag)
    draw_conflict_arrows(fig.add_subplot(gs[1, 1]), datum, shown_lanes=shown_lanes, nbanks=nbanks,
                         fixed=True, fix_bank_fn=fix_bank_fn, subject_bank=subject_bank, cpa=0.0)
    draw_conflict_bankgrid(fig.add_subplot(gs[1, 2]), datum, half=half, nbanks=nbanks, nreg=nreg,
                           fixed=True, fix_bank_fn=fix_bank_fn, subject_bank=subject_bank, cpa=0.0)
    fig.suptitle(suptitle, fontsize=9.5, y=0.99)
    draw_phase_legend(fig, nreg)
    return _save_fig(fig, out_path, 150, tight=True)


@dataclass
class CellGroup:
    """A set of cell-field cells drawn as one group (cells are ``(tid, vreg)`` for a register file or
    ``(c0, c1)`` logical coords for a tile -- the group machinery is identical either way). ``detail``:
    ``'detailed'`` -> border + a label in every cell; ``'grouped'`` -> border + one DERIVED summary label
    + the anchor cell bordered and stamped with its real coordinate (the WAVE view: one atom detailed);
    ``'block'`` -> a solid block, no inner grid / anchor, just the ``name`` centred (the MACRO/WG view);
    ``'plain'`` -> border only, no labels (the WG LDS bank grid, wave read from hue)."""
    members: frozenset
    detail: str = "grouped"          # "detailed" | "grouped" | "block" | "plain"
    name: str = ""


# RegGroup / LogicalGroup are the SAME structure -- kept as names for back-compat.
RegGroup = CellGroup


class CellFieldMixin:
    """Shared behaviour for every cell-field view (register file, logical tile, and -- later -- LDS
    banks): a grid of labelled/bordered/shaded cells. The drawing is already in ``_canvas``; this mixin
    houses the per-view GLUE that was copy-pasted between the components (grouping/shade/detail/highlight
    computation, tick placement, figure scaffold). Each view supplies a few hooks for the ONLY parts
    that differ:

      ``_cells()``            iterable of cell keys.
      ``_pos(*cell)``         cell key -> (grid_x, grid_y).
      ``_lane_of(cell)``      cell key -> owning lane (hue).
      ``_cell_label(*cell)``  cell key -> its text label.
      ``summary_label(members)`` one derived label for a grouped group.
      ``_highlight_cells()``  the set of cell keys to outline (provenance/follow trace).
      ``_grid_vals()``        (col_vals, row_vals) -- the drawn axis orders.
      ``_tick_spec()``        ((col_vals, kind, name), (row_vals, kind, name)); kind in {reg, edge}.

    It reads the shared dataclass fields (``groups``/``shade_map``/``color_mode``/``highlight_color``/
    ``col_ticks_side``/``row_ticks_side``/``origin``/``font_size``/``cell_w``/``cell_h``/``title``)."""

    wg_detail = "block"                  # WG-scope collapse style: "block" (tile/register) | "plain" (LDS)

    def _wave_of(self, cell):
        """The WAVE a cell belongs to = its owning lane // ``lanes_per_wave`` (total lanes may span many
        waves when fed a cooperative macro map). The WG scope groups + colours by this."""
        return self._lane_of(cell) // max(1, getattr(self, "lanes_per_wave", 64))

    def _wg_groups(self):
        """MACRO/WG scope: collapse detail to ONE group per WAVE. ``block`` detail (tile/register) borders +
        names each wave; ``plain`` detail (the LDS bank grid) keeps ONE group over all slots -- the wave is
        read from HUE, not borders. Axes/orientation are untouched (this only changes the overlay)."""
        if self.wg_detail == "plain":
            return [CellGroup(frozenset(self._cells()), "plain", "")]
        by_wave = {}
        for c in self._cells():
            by_wave.setdefault(self._wave_of(c), []).append(c)
        return [CellGroup(frozenset(cs), self.wg_detail, f"wave {w}") for w, cs in sorted(by_wave.items())]

    def _effective_groups(self):
        if getattr(self, "scope", "wave") == "wg":
            return self._wg_groups()
        return list(self.groups) if self.groups else [
            CellGroup(frozenset(self._cells()), detail="detailed")]

    def grid_size(self):
        col_vals, row_vals = self._grid_vals()
        return len(col_vals), len(row_vals)

    def size(self):
        nc, nr = self.grid_size()
        return nc * self.cell_w, nr * self.cell_h

    def draw(self, ax):
        # WAVE scope: hue = lane % 8; SHADE = group (feed step) unless a ``shade_map`` decouples it.
        # WG scope: hue = WAVE (all coloured), SHADE = wave // 8 (a 2nd cycle so waves 8..15 stay distinct
        # from 0..7). Borders/labels come from the groups (detailed=label all, grouped=anchor+summary,
        # block=one name, plain=none).
        groups = self._effective_groups()
        if getattr(self, "scope", "wave") == "wg":
            lane_of = self._wave_of
            shade_of = {c: self._wave_of(c) // NACC for c in self._cells()}
            nsteps = max(1, max(shade_of.values(), default=0) + 1)
            cmode = "full"                                   # every wave coloured (not just the first 8 lanes)
        else:
            lane_of = self._lane_of
            step_of = {cell: i for i, g in enumerate(groups) for cell in g.members}
            if self.shade_map is not None:
                shade_of = self.shade_map; nsteps = max(1, max(self.shade_map.values()) + 1)
            else:
                shade_of = step_of; nsteps = max(1, len(groups))
            cmode = self.color_mode
        detailed = set().union(*(g.members for g in groups if g.detail == "detailed")) \
            if any(g.detail == "detailed" for g in groups) else set()
        cv.render_cells_and_groups(
            ax, self._cells(), pos_of=lambda c: self._pos(*c), lane_of=lane_of,
            shade_of=shade_of, nsteps=nsteps, color_mode=cmode, groups=groups,
            detailed=detailed, label_of=lambda c: self._cell_label(*c), summary_of=self.summary_label,
            highlight_cells=self._highlight_cells(), highlight_color=self.highlight_color,
            origin=self.origin, fs=self.font_size, cell_wh=(self.cell_w, self.cell_h))

    def _ticks(self, ax):
        ox, oy = self.origin
        (cvals, ckind, cname), (rvals, rkind, rname) = self._tick_spec()
        cstep = cv.auto_tick_step(len(cvals), self.cell_w)   # ~0.42in apart at the real cell size (no crowd)
        rstep = cv.auto_tick_step(len(rvals), self.cell_h)
        xt = cv.reg_ticks(cvals, self.dtype_bits) if ckind == "reg" else cv.edge_ticks(cvals, cstep)
        cv.set_axis_ticks(ax, xt, axis="x", origin_off=ox, fs=self.font_size, name=cname,
                          side=self.col_ticks_side)
        yt = cv.reg_ticks(rvals, self.dtype_bits) if rkind == "reg" else cv.edge_ticks(rvals, rstep)
        cv.set_axis_ticks(ax, yt, axis="y", origin_off=oy, fs=self.font_size, name=rname,
                          side=self.row_ticks_side)

    def draw_ticks(self, ax, step=None, mark=0.25, cell_in=0.2):
        """Draw real tick MARKS + labels + axis names relative to ``origin`` (for composition on a shared
        axes). A register axis is an edge-anchored 32-bit-register ruler; other axes AUTO-TUNE their tick
        density to ~0.42in at the render scale (``cell_in`` = inches/cell, e.g. the pipeline ``scale``) so
        ticks stay visible and never crowd -- unless an explicit ``step`` overrides. ``col_ticks_side`` /
        ``row_ticks_side`` pick which edge each faces."""
        nc, nr = self.grid_size()
        (cvals, ckind, cname), (rvals, rkind, rname) = self._tick_spec()
        cstep = step or cv.auto_tick_step(len(cvals), cell_in)
        rstep = step or cv.auto_tick_step(len(rvals), cell_in)
        xt = cv.reg_ticks(cvals, self.dtype_bits) if ckind == "reg" else cv.edge_ticks(cvals, cstep)
        cv.axis_marks(ax, xt, horizontal=True, side=self.col_ticks_side, origin=self.origin,
                      along=nc, cross=nr, name=cname, fs=self.font_size, mark=mark)
        yt = cv.reg_ticks(rvals, self.dtype_bits) if rkind == "reg" else cv.edge_ticks(rvals, rstep)
        cv.axis_marks(ax, yt, horizontal=False, side=self.row_ticks_side, origin=self.origin,
                      along=nr, cross=nc, name=rname, fs=self.font_size, mark=mark)

    def render(self, path, dpi=200):
        return cv.component_figure(self, path, dpi)


@dataclass
class RegisterFileComponent(CellFieldMixin):
    """Register-file view: a grid of ``tid x vreg`` cells filled from a static distribution. ONE
    component renders both MMA-tee wings by changing knobs only. Colour hue = lane%8; SHADE = group
    (feed step) so a lane's cells in one group share a shade (== used together).

    Fed EITHER by a ``dist`` (encoding -> coords via ``RegisterMapper``) OR by an explicit ``fwd_map``
    (``{(lane,reg)->(c0,c1)}``) -- the latter lets a DERIVED distribution (e.g. the tee's machine-derived
    C register file) drive the same component even when it is not a representable encoding."""
    dist: WarpDistributionEncoding | None = None
    dims: tuple = ("M", "K")                  # names for the two logical coords (c0, c1)
    row_axis: str = "tid"                     # which axis is vertical: "tid" | "vreg"
    col_axis: str = "vreg"                    # which axis is horizontal
    row_order: str = "asc"                    # "asc" | "desc" (ignored if explicit *_values given)
    col_order: str = "asc"
    tid_values: tuple | None = None           # explicit tid order (e.g. group-major); None -> range
    vreg_values: tuple | None = None          # explicit vreg order
    groups: tuple = ()                        # of RegGroup; empty -> one detailed group over all cells
    color_mode: str = "first8"                # "first8" (lanes 0-7 coloured, rest grey) | "full" (all lane%8)
    col_ticks_side: str = "top"               # register files default to axes on TOP (vreg) + LEFT (tid)
    row_ticks_side: str = "left"              # where the vertical-axis ruler sits: "left" | "right"
    fwd_map: dict | None = None               # {(lane,reg)->(c0,c1)}; overrides ``dist`` (derived layouts)
    shade_map: dict | None = None             # {(lane,reg)->step}; SHADE (feed/iteration order) DECOUPLED
                                              # from ``groups`` (which drive borders). None -> shade by group.
    highlight: frozenset = frozenset()         # logical coords (c0,c1) to outline (provenance/follow trace)
    highlight_color: str = "#d81b60"           # outline colour for highlighted cells
    dtype_bits: int = 32                       # element width; sets vreg->physical 32-bit register indexing
    wave_size: int | None = None               # lanes; None -> take from the distribution (NOT assumed 64)
    scope: str = "wave"                        # "wave" (per-cell detail) | "wg" (macro: one block per wave)
    lanes_per_wave: int = 64                   # lanes per wave for WG grouping (total lanes may span many)
    cell_w: float = 0.62                      # inches per cell (standalone render)
    cell_h: float = 0.30
    font_size: float = 7.0
    origin: tuple = (0.0, 0.0)                # (x, y) placement in cell units (composition)
    title: str = ""

    def __post_init__(self):
        if self.fwd_map is not None:                   # derived distribution: (lane,reg) -> coord given directly
            self._fwd = dict(self.fwd_map)
            self._nreg = max((r for _, r in self._fwd), default=-1) + 1
            self._wave = self.wave_size or (max((l for l, _ in self._fwd), default=-1) + 1)
            return
        if self.dist is None:
            raise ValueError("RegisterFileComponent needs either dist or fwd_map")
        self._rm = RegisterMapper(self.dist)
        nl = self._rm.num_lanes
        if self.wave_size is not None and self.wave_size != nl:
            raise ValueError(f"wave_size={self.wave_size} != distribution lanes={nl}")
        self._wave = self.wave_size or nl        # wave size from the distribution; wave32/wave64 both work
        self._nreg = self._rm.num_vector_items

    # -- geometry -------------------------------------------------------------
    def _axis_vals(self, which):
        n = self._wave if which == "tid" else self._nreg
        explicit = self.tid_values if which == "tid" else self.vreg_values
        if explicit is not None:
            return list(explicit)
        order = self.row_order if which == self.row_axis else self.col_order
        vals = list(range(n))
        return vals if order == "asc" else vals[::-1]

    def _pos(self, tid, vreg):
        cvv = self._axis_vals(self.col_axis); rvv = self._axis_vals(self.row_axis)
        val = {"tid": tid, "vreg": vreg}
        return cvv.index(val[self.col_axis]), rvv.index(val[self.row_axis])

    def _coord(self, tid, vreg):
        if self.fwd_map is not None:
            return self._fwd[(tid, vreg)]
        return self._rm.matrix_coordinates(tid, vreg)

    def summary_label(self, members):
        """Derive ONE label from the group's cell contents: range -> affine -> CUSTOM."""
        c0 = [self._coord(t, v)[0] for (t, v) in members]
        c1 = [self._coord(t, v)[1] for (t, v) in members]
        a, b = _compact_1d(c0), _compact_1d(c1)
        return "CUSTOM" if a is None or b is None else f"{self.dims[0]}{a} {self.dims[1]}{b}"

    # -- CellFieldMixin hooks -------------------------------------------------
    def _cells(self):
        return [(t, v) for t in range(self._wave) for v in range(self._nreg)]

    def _lane_of(self, cell):
        return cell[0]

    def _cell_label(self, tid, vreg):
        c0, c1 = self._coord(tid, vreg)
        return f"{self.dims[0]}{c0}{self.dims[1]}{c1}"

    def _highlight_cells(self):
        if not self.highlight:
            return set()
        return {c for c in self._cells() if self._coord(*c) in self.highlight}

    def _grid_vals(self):
        return self._axis_vals(self.col_axis), self._axis_vals(self.row_axis)

    def _tick_spec(self):
        def spec(axis):
            kind = "reg" if axis == "vreg" else "edge"
            name = "vreg (32-bit reg)" if axis == "vreg" else "tid"
            return (self._axis_vals(axis), kind, name)
        return spec(self.col_axis), spec(self.row_axis)


# LogicalGroup is the same structure as RegGroup/CellGroup -- kept as a name for back-compat.
LogicalGroup = CellGroup


@dataclass
class LogicalTileComponent(CellFieldMixin):
    """Logical-tile view: the DUAL of :class:`RegisterFileComponent`. The grid is the logical matrix (the
    two coords the distribution deploys, axis labels supplied by the caller); each cell's CONTENT is either
    its register owner ``T{lane}R{reg}`` or its logical coord (``label_coords``). Colour hue = owner
    lane%8; SHADE = group (feed step).

    Fed EITHER by a ``dist`` (``WarpDistributionEncoding`` -> owner via ``RegisterMapper``) OR by an explicit
    ``owner_map`` (``{(c0,c1) -> (lane,reg)}``) -- the latter lets a DERIVED distribution (e.g. the tee's
    machine-derived C, which may not be a representable encoding) drive the same component."""
    dist: WarpDistributionEncoding | None = None
    dims: tuple = ("M", "K")                   # axis labels for the two logical coords (c0, c1) -- caller-given
    row_coord: int = 0                         # which logical coord (0|1) is vertical
    row_order: str = "asc"                     # "asc" | "desc"
    col_order: str = "asc"
    mode: str = "layout"                       # view preset: "layout" (a) | "thread_tile" (b) |
                                               # "coalescing" (c). Auto-derives borders (+ shade for
                                               # b/c) UNLESS ``groups``/``shade_map`` are given explicitly.
    atom: tuple | None = None                  # (atom0, atom1) -> layout-mode tile borders; None -> whole tile
    addr_fn: object | None = None              # (c0,c1)->addr: memory order for vectorization shade (b/c)
                                               # + physical-vector borders (c); None -> coord (row-major) order
    detail_first: bool = True                  # first auto-group detailed, the rest grouped
    groups: tuple = ()                         # of LogicalGroup; empty -> one detailed group over all cells
    color_mode: str = "first8"                 # "first8" (owner lanes 0-7 coloured) | "full"
    label_coords: str = "logical"              # DEFAULT: the logical coord {dim0}{c0}{dim1}{c1}. Set
                                               # "register" ONLY when asked -> owner T{lane}R{reg}.
    col_ticks_side: str = "top"                # logical tiles default to axes on TOP (cols) + LEFT (rows)
    row_ticks_side: str = "left"               # where the vertical-axis ruler sits: "left" | "right"
    owner_map: dict | None = None              # {(c0,c1)->(lane,reg)}; overrides ``dist`` (derived layouts)
    shade_map: dict | None = None              # {(c0,c1)->step}; SHADE (feed/iteration order) DECOUPLED from
                                               # ``groups`` (borders). None -> shade by group.
    text_map: dict | None = None               # {(c0,c1)->(v0,v1)} shown-label overriding the grid coord:
                                               # lets the grid be POSITION while cells show the FLOWED label
    highlight: frozenset = frozenset()         # logical coords (c0,c1) to outline (provenance/follow trace)
    highlight_color: str = "#d81b60"           # outline colour for highlighted cells
    wave_size: int | None = None
    scope: str = "wave"                        # "wave" (per-cell detail) | "wg" (macro: one block per wave)
    lanes_per_wave: int = 64                   # lanes per wave for WG grouping (total lanes may span many)
    cell_w: float = 0.42
    cell_h: float = 0.42
    font_size: float = 7.0
    origin: tuple = (0.0, 0.0)
    title: str = ""

    def __post_init__(self):
        if self.owner_map is not None:                   # derived distribution: coord -> (lane, reg) given directly
            self._owner = dict(self.owner_map)
            lanes = [lr[0] for lr in self._owner.values()]
            self._wave = self.wave_size or (max(lanes) + 1 if lanes else 1)
            e0 = max((c[0] for c in self._owner), default=-1) + 1
            e1 = max((c[1] for c in self._owner), default=-1) + 1
            self._ext = (e0, e1)
        else:
            if self.dist is None:
                raise ValueError("LogicalTileComponent needs either dist or owner_map")
            self._rm = RegisterMapper(self.dist)
            nl = self._rm.num_lanes
            if self.wave_size is not None and self.wave_size != nl:
                raise ValueError(f"wave_size={self.wave_size} != distribution lanes={nl}")
            self._wave = self.wave_size or nl
            self._owner = {}; e0 = e1 = 0                # coord -> (lane, reg); bijective
            for lane in range(self._wave):
                for reg in range(self._rm.num_vector_items):
                    c0, c1 = self._rm.matrix_coordinates(lane, reg)
                    self._owner[(c0, c1)] = (lane, reg)
                    e0 = max(e0, c0 + 1); e1 = max(e1, c1 + 1)
            self._ext = (e0, e1)
        self._apply_mode()

    # -- view-mode presets (additive: only act when groups/shade_map not given explicitly) -----
    def _apply_mode(self):
        if not self.groups:
            auto = self._auto_groups()
            if auto:
                self.groups = auto
        if self.shade_map is None and self.mode in ("thread_tile", "coalescing"):
            self.shade_map = self._vector_shade()

    def _vector_runs(self):
        """Contiguous-memory runs per lane (a physical vector = a maximal run of adjacent addresses),
        ordered lane-major. ``addr_fn`` gives the memory order; default is coord (row-major)."""
        addr = self.addr_fn or (lambda c0, c1: c0 * (self._ext[1] + 1) + c1)
        by_lane = {}
        for coord, (lane, _r) in self._owner.items():
            by_lane.setdefault(lane, []).append(coord)
        out = []
        for lane in sorted(by_lane):
            cs = sorted(by_lane[lane], key=lambda c: addr(*c))
            cur = [cs[0]]
            for c in cs[1:]:
                if addr(*c) == addr(*cur[-1]) + 1:
                    cur.append(c)
                else:
                    out.append((f"T{lane}", cur)); cur = [c]
            out.append((f"T{lane}", cur))
        return out

    def _auto_groups(self):
        """Borders for the view mode: ``layout`` -> atom tiles (or nothing -> default single group);
        ``thread_tile`` -> one group per owning lane; ``coalescing`` -> one group per physical-vector
        run. First group detailed / rest grouped when ``detail_first``."""
        def tag(i):
            return "detailed" if (self.detail_first and i == 0) else "grouped"
        if self.mode == "thread_tile":
            by_lane = {}
            for coord, (lane, _r) in self._owner.items():
                by_lane.setdefault(lane, []).append(coord)
            return tuple(CellGroup(frozenset(by_lane[l]), tag(i), f"T{l}")
                         for i, l in enumerate(sorted(by_lane)))
        if self.mode == "coalescing":
            return tuple(CellGroup(frozenset(run), tag(i), name)
                         for i, (name, run) in enumerate(self._vector_runs()))
        if self.atom is not None:                        # layout mode with atom-tile borders
            a0, a1 = self.atom
            by_tile = {}
            for coord in self._owner:
                by_tile.setdefault((coord[0] // a0, coord[1] // a1), []).append(coord)
            return tuple(CellGroup(frozenset(by_tile[k]), tag(i), str(k))
                         for i, k in enumerate(sorted(by_tile)))
        return ()

    def _vector_shade(self):
        """SHADE = vectorization/time order: each thread's contiguous-memory run is one time step
        (darker earlier). Derived via ``transactions`` over the memory order (``addr_fn`` or coords)."""
        mp = {(lane, reg): coord for coord, (lane, reg) in self._owner.items()}
        addr = self.addr_fn or (lambda c0, c1: c0 * (self._ext[1] + 1) + c1)
        ts, _ = transactions(mp, addr)
        return {coord: ts[(lane, reg)] for coord, (lane, reg) in self._owner.items()}

    # -- geometry -------------------------------------------------------------
    def _col_coord(self):
        return 1 - self.row_coord

    def _coord_vals(self, ci):
        order = self.row_order if ci == self.row_coord else self.col_order
        vals = list(range(self._ext[ci]))
        return vals if order == "asc" else vals[::-1]

    def _pos(self, c0, c1):
        cc = self._col_coord()
        cvv = self._coord_vals(cc); rvv = self._coord_vals(self.row_coord)
        coord = (c0, c1)
        return cvv.index(coord[cc]), rvv.index(coord[self.row_coord])

    def _coord_label(self):                                      # cell text = logical coord, not register owner
        return self.label_coords == "logical"

    def summary_label(self, members):
        """Derive ONE label from the group. ``register`` labelling summarises the OWNERS (T<lanes> R<regs>);
        ``logical`` labelling summarises the logical COORDS (M<range> N<range>). range -> affine -> CUSTOM."""
        if self.text_map is not None:                            # summarise the FLOWED labels, not the positions
            a = _compact_1d([self.text_map[m][0] for m in members])
            b = _compact_1d([self.text_map[m][1] for m in members])
            return "CUSTOM" if a is None or b is None else f"{self.dims[0]}{a} {self.dims[1]}{b}"
        if self._coord_label():
            a = _compact_1d([m[0] for m in members]); b = _compact_1d([m[1] for m in members])
            return "CUSTOM" if a is None or b is None else f"{self.dims[0]}{a} {self.dims[1]}{b}"
        lanes = [self._owner[m][0] for m in members]; regs = [self._owner[m][1] for m in members]
        a, b = _compact_1d(lanes), _compact_1d(regs)
        return "CUSTOM" if a is None or b is None else f"T{a} R{b}"

    # -- CellFieldMixin hooks -------------------------------------------------
    def _cells(self):
        return list(self._owner)

    def _lane_of(self, cell):
        return self._owner[cell][0]

    def _cell_label(self, c0, c1):
        if self.text_map is not None:                            # grid = POSITION, cell text = FLOWED label
            v0, v1 = self.text_map[(c0, c1)]
            return f"{self.dims[0]}{v0}{self.dims[1]}{v1}"
        if self._coord_label():                                  # logical coord of this cell
            return f"{self.dims[0]}{c0}{self.dims[1]}{c1}"
        lane, reg = self._owner[(c0, c1)]                        # register: destination register owner
        return f"T{lane}R{reg}"

    def _highlight_cells(self):
        return {c for c in self._owner if c in self.highlight} if self.highlight else set()

    def _grid_vals(self):
        return self._coord_vals(self._col_coord()), self._coord_vals(self.row_coord)

    def _tick_spec(self):
        cc = self._col_coord()
        return ((self._coord_vals(cc), "edge", self.dims[cc]),
                (self._coord_vals(self.row_coord), "edge", self.dims[self.row_coord]))


@dataclass
class LdsBankView(CellFieldMixin):
    """LDS data-placement view (the third cell field): a ``depth x banks`` grid showing WHERE a wave's
    store puts each element in LDS. A cell is an occupied LDS slot at ``(depth, bank) = (addr // NB,
    addr % NB)``; origin ``(depth 0, bank 0)`` is top-left. Label = the flowed logical datum (default)
    or the writing thread ``T{lane}R{reg}`` (``label_by``). Hue = writing lane; SHADE = vectorization /
    time order (one contiguous-address store run = one shade, darker = issued earlier). Placement is
    HW-fixed; phase / half-wave / conflict zoom is the bank-conflict skill's job, not this view's.

    Fed by the wave register map ``mp`` (``{(lane,reg)->(row,col)}``) + an ``addr_fn`` giving each
    element's LDS address; an optional ``flow_map`` supplies the logical labels to carry (default: the
    ``mp`` coords themselves)."""
    mp: dict = None                            # {(lane,reg)->(row,col)} the wave's register map
    addr_fn: object = None                     # (row,col)->int LDS element address (memory order)
    nbanks: int = 32                           # arch bank count (gfx90a/CDNA = 32)
    label_by: str = "flow"                     # "flow" (logical datum) | "thread" (T{lane}R{reg})
    flow_map: dict | None = None               # {(lane,reg)->(v0,v1)} logical labels; None -> use mp coords
    dims: tuple = ("K", "free")                # names for the flowed logical label (v0, v1)
    color_mode: str = "first8"
    row_order: str = "asc"                     # depth order
    col_order: str = "asc"                     # bank order
    groups: tuple = ()                         # of CellGroup; empty -> one detailed group over all slots
    shade_map: dict | None = None              # {(depth,bank)->step}; None -> vectorization/time order
    col_ticks_side: str = "top"                # bank axis ruler defaults to the TOP (HW convention)
    row_ticks_side: str = "left"               # depth axis ruler
    highlight: frozenset = frozenset()         # (depth,bank) slots to outline (provenance/follow trace)
    highlight_color: str = "#d81b60"
    order_by: str = "reg"                      # transaction ordering for the shade ("reg" issue | "addr")
    compact_rows: bool = False                 # drop EMPTY depth rows (reindex occupied depths consecutive):
                                               # a wide macro store occupies few of many physical depths, so
                                               # this keeps the grid the height of the OCCUPIED rows only.
    scope: str = "wave"                        # "wave" (per-lane detail) | "wg" (macro: colour by wave)
    lanes_per_wave: int = 64                   # lanes per wave for WG colouring (total lanes may span many)
    cell_w: float = 0.42
    cell_h: float = 0.42
    font_size: float = 6.0
    origin: tuple = (0.0, 0.0)
    title: str = ""
    wg_detail = "plain"                        # WG scope: keep the bank grid, colour by wave (no labels)

    def __post_init__(self):
        if self.mp is None or self.addr_fn is None:
            raise ValueError("LdsBankView needs mp and addr_fn")
        ts, maxt = transactions(self.mp, self.addr_fn, order_by=self.order_by)
        self._owner = {}                       # (depth,bank) -> (lane,reg)
        self._flow = {}                        # (depth,bank) -> (v0,v1) flowed logical label
        auto_shade = {}                        # (depth,bank) -> vectorization/time step
        self._row_depth = {}                   # drawn-row -> real LDS depth (identity unless compacted)
        for (lane, reg), (r, c) in self.mp.items():
            addr = int(self.addr_fn(r, c))
            slot = (addr // self.nbanks, addr % self.nbanks)
            self._owner[slot] = (lane, reg)
            self._flow[slot] = self.flow_map[(lane, reg)] if self.flow_map else (r, c)
            auto_shade[slot] = ts[(lane, reg)]
        if self.compact_rows:                  # reindex OCCUPIED depths to consecutive rows (drop empties)
            occ = sorted({d for d, _b in self._owner})
            idx = {d: i for i, d in enumerate(occ)}
            self._row_depth = {i: d for d, i in idx.items()}
            self._owner = {(idx[d], b): v for (d, b), v in self._owner.items()}
            self._flow = {(idx[d], b): v for (d, b), v in self._flow.items()}
            auto_shade = {(idx[d], b): v for (d, b), v in auto_shade.items()}
            rows = len(occ)
        else:
            rows = (max((d for d, _b in self._owner), default=-1) + 1)
            self._row_depth = {i: i for i in range(rows)}
        if self.shade_map is None:
            self.shade_map = auto_shade
        self._ext = (rows, self.nbanks)            # (rows = drawn depth, cols = banks)

    # -- geometry -------------------------------------------------------------
    def _axis_vals(self, ci):
        order = self.row_order if ci == 0 else self.col_order
        vals = list(range(self._ext[ci]))
        return vals if order == "asc" else vals[::-1]

    def _pos(self, depth, bank):
        rvv = self._axis_vals(0); cvv = self._axis_vals(1)
        return cvv.index(bank), rvv.index(depth)

    def summary_label(self, members):
        if self.label_by == "thread":
            lanes = [self._owner[m][0] for m in members]; regs = [self._owner[m][1] for m in members]
            a, b = _compact_1d(lanes), _compact_1d(regs)
            return "CUSTOM" if a is None or b is None else f"T{a} R{b}"
        a = _compact_1d([self._flow[m][0] for m in members])
        b = _compact_1d([self._flow[m][1] for m in members])
        return "CUSTOM" if a is None or b is None else f"{self.dims[0]}{a} {self.dims[1]}{b}"

    # -- CellFieldMixin hooks -------------------------------------------------
    def _cells(self):
        return list(self._owner)

    def _lane_of(self, cell):
        return self._owner[cell][0]                        # RAW writing lane (WG colours by wave via _wave_of)

    def _cell_label(self, depth, bank):
        if self.label_by == "thread":
            lane, reg = self._owner[(depth, bank)]
            return f"T{lane}R{reg}"
        v0, v1 = self._flow[(depth, bank)]
        return f"{self.dims[0]}{v0}{self.dims[1]}{v1}"

    def _highlight_cells(self):
        return {c for c in self._owner if c in self.highlight} if self.highlight else set()

    def _grid_vals(self):
        return self._axis_vals(1), self._axis_vals(0)      # (col=bank vals, row=depth vals)

    def _tick_spec(self):
        depth_name = "LDS depth (occupied rows)" if self.compact_rows else "LDS depth"
        return ((self._axis_vals(1), "edge", f"bank (0..{self.nbanks - 1})"),
                (self._axis_vals(0), "edge", depth_name))


def _cellfield_legend(fig, nsteps=4):
    """Figure-level legend for the cell-field channels: HUE = thread (lane % 8), SHADE = vectorization /
    time order (darkest = issued first). Two stacked rows at the bottom."""
    from matplotlib.patches import Patch
    hue = [Patch(facecolor=ACCENTS[i], edgecolor="0.4", label=f"T{i}") for i in range(NACC)]
    shade = [Patch(facecolor=accent_tint(5, k, nsteps), edgecolor="0.4",
                   label=("first" if k == 0 else "last" if k == nsteps - 1 else str(k)))
             for k in range(nsteps)]
    leg = fig.legend(handles=hue, loc="lower center", ncol=NACC, fontsize=6, frameon=True,
                     bbox_to_anchor=(0.5, -0.02), title="hue = thread (lane % 8)", title_fontsize=6)
    fig.add_artist(leg)
    fig.legend(handles=shade, loc="lower center", ncol=nsteps, fontsize=6, frameon=True,
               bbox_to_anchor=(0.5, -0.10),
               title="shade = vectorization / time order (darkest = first)", title_fontsize=6)


def _legend_in_axes(ax, x, y, nsteps=4, sw=1.4, row=1.0):
    """Draw the hue + shade legend in DATA coords, growing down from (x, y) -- so it sits directly under
    the distribution box in the header (not floating at the figure edge)."""
    from matplotlib.patches import Rectangle
    ax.text(x, y, "hue = thread (lane % 8)", ha="left", va="top", fontsize=6.8, family="monospace",
            zorder=6)
    for i in range(NACC):
        ax.add_patch(Rectangle((x + i * sw, y + 0.5), sw * 0.85, row, facecolor=ACCENTS[i],
                               edgecolor="0.4", lw=0.5, zorder=6))
        ax.text(x + i * sw + sw * 0.42, y + 0.5 + row / 2, f"T{i}", ha="center", va="center",
                fontsize=5.2, zorder=7)
    y2 = y + 0.5 + row + 0.7
    ax.text(x, y2, "shade = vectorization / time order (darkest = first)", ha="left", va="top",
            fontsize=6.8, family="monospace", zorder=6)
    for k in range(nsteps):
        ax.add_patch(Rectangle((x + k * sw, y2 + 0.5), sw * 0.85, row,
                               facecolor=accent_tint(5, k, nsteps), edgecolor="0.4", lw=0.5, zorder=6))
        ax.text(x + k * sw + sw * 0.42, y2 + 0.5 + row / 2,
                ("first" if k == 0 else "last" if k == nsteps - 1 else str(k)), ha="center",
                va="center", fontsize=4.6, zorder=7)


@dataclass
class FlowStage:
    """One stage of a data-flow pipeline: a cell-field ``component`` (register / logical / LDS view) whose
    cells each carry a global-memory ORIGIN label, so a datum is traceable stage-to-stage. ``transform``
    names the operation that produced this stage from the previous one (e.g. "global load", "LDS store",
    "wave read", "MMA", "C-shuffle"). ``origin`` maps each cell key -> the global (o0, o1) it holds; when
    omitted it is taken from the component (a register/LDS view's flowed coord, a logical view's coord)."""
    name: str
    component: object
    transform: str = ""
    origin: dict | None = None          # {cell_key -> (o0,o1)}; None -> derive from the component

    def _origin(self):
        if self.origin is not None:
            return self.origin
        comp = self.component
        if isinstance(comp, LogicalTileComponent):          # coord grid: origin = the coord (or flowed)
            if comp.text_map is not None:
                return dict(comp.text_map)
            return {c: c for c in comp._owner}
        if isinstance(comp, LdsBankView):                   # slot -> flowed logical datum
            return dict(comp._flow)
        if isinstance(comp, RegisterFileComponent):         # (lane,reg) -> its (flowed) coord
            return {c: comp._coord(*c) for c in comp._cells()}
        return {}

    def cells_for(self, origin_coord):
        """The cell keys in this stage whose ORIGIN is ``origin_coord`` (what to highlight when tracing)."""
        return {cell for cell, o in self._origin().items() if o == origin_coord}

    def summary(self):
        """The full static tile distribution (Rs/Hs/Ps/Ys) + effective vector width, as text lines --
        the 'distribution dump' shown under each stage. Falls back to a grid summary for views without
        a representable encoding (LDS / explicit maps)."""
        comp = self.component
        dist = getattr(comp, "dist", None)
        if dist is None:
            nc, nr = comp.grid_size()
            return [f"{nc} x {nr} cells (no encoding)"]
        rm = RegisterMapper(dist)
        mp = {(l, r): rm.matrix_coordinates(l, r)
              for l in range(rm.num_lanes) for r in range(rm.num_vector_items)}
        ncols = max((c for _, c in mp.values()), default=0) + 1
        _ts, maxt = transactions(mp, lambda r, c: r * ncols + c)
        vw = rm.num_vector_items // max(1, maxt)
        return [
            f"lanes={rm.num_lanes}  regs/lane={rm.num_vector_items}  eff VW={vw}",
            f"Rs={dist.replication_lengths}  Hs={dist.hierarchical_lengths}",
            f"Ps maj={dist.lane_to_rh_major} min={dist.lane_to_rh_minor}",
            f"Ys maj={dist.register_to_rh_major} min={dist.register_to_rh_minor}",
        ]


def _save_fig(fig, out_path, dpi, *, tight=False):
    """Shared figure finish: ensure the dir exists, save (optionally bbox-tight), close, return the path."""
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=dpi, **({"bbox_inches": "tight"} if tight else {}))
    _plt().close(fig)
    return out_path


def _flow_arrow(ax, x_tail, x_head, y, label, *, fs=8, lw=1.7, label_dy=0.0):
    """A left-to-right stage-transition arrow + its italic transform label, drawn in whatever coordinate
    space ``ax`` uses (data coords for the shared-axes render, figure-fraction for the panel render)."""
    ax.annotate("", xy=(x_head, y), xytext=(x_tail, y),
                arrowprops=dict(arrowstyle="-|>", lw=lw, color="#333"))
    if label:
        ax.text((x_tail + x_head) / 2.0, y + label_dy, label, ha="center", va="bottom",
                fontsize=fs, style="italic", color="#333")


@dataclass
class Pipeline:
    """A traceable data-flow: an ordered list of :class:`FlowStage`. ``trace`` lights one global-memory
    datum across every stage (global -> registers -> LDS -> ... -> C); ``render`` lays the stages
    left-to-right with the transform between each. Built on the cell-field views, so each stage is just a
    configured component -- the pipeline adds the traceability + staged composition."""
    stages: tuple = ()
    title: str = ""

    def trace(self, origin_coord, color="#d81b60"):
        """Highlight ``origin_coord`` (a global (o0,o1)) in every stage that carries it. The register
        view highlights by LOGICAL COORD; the logical / LDS views highlight by CELL KEY -- so each stage
        gets the value in its own highlight vocabulary."""
        for s in self.stages:
            comp = s.component
            if isinstance(comp, (RegisterFileComponent, WaveStrip)):
                comp.highlight = frozenset({origin_coord})     # register views (incl. per-wave strip) key by coord
            else:
                comp.highlight = frozenset(s.cells_for(origin_coord))
            comp.highlight_color = color

    def render(self, out_path, dpi=200, gap=4.0, ticks=True, scale=None, show_info=True,
              show_legend=True, max_in=16.0):
        plt = _plt()
        sizes = [(s, *s.component.grid_size()) for s in self.stages]
        maxnr = max((nr for _, _nc, nr in sizes), default=1)
        minnr = min((nr for _, _nc, nr in sizes), default=1)
        if scale is None:                                   # auto: bound the TALL axis (e.g. tid=1024 in WG)
            scale = min(0.52, max_in / max(1.0, maxnr + 8))  # to ~max_in inches; never upscale past 0.52
        placed = []
        x = 0.0
        for s, nc, nr in sizes:
            s.component.origin = (x, 0.0)                    # TOP-ALIGN all stages (grid tops share y=0)
            placed.append((s, x, nc, nr))
            x += nc + gap
        total_w = max(1.0, x - gap)
        # header = ONE box per UNIQUE distribution (stages sharing a layout are not repeated) + the
        # hue/shade legend directly beneath it, sitting COMPACTLY just above the tile tops.
        uniq = []
        for s in self.stages:
            key = tuple(s.summary())
            for u in uniq:
                if u[0] == key:
                    u[1].append(s.name); break
            else:
                uniq.append((key, [s.name]))
        BOXGAP, LINE = 0.7, 1.15                             # tight inter-box gap + line pitch
        header_h = 3.0                                       # name row + margin above the grids
        if show_info:
            header_h += sum((len(k) + 1) * LINE + BOXGAP for k, _ in uniq)
        if show_legend:
            header_h += 5.2
        fig = plt.figure(figsize=(max(7.0, total_w * scale + 2.4),
                                  max(5.0, (maxnr + header_h + 1) * scale + 2.0)))
        ax = fig.add_axes([0.03, 0.05, 0.94, 0.86])
        ax.axis("off")
        ax.set_aspect("equal")
        arrow_y = minnr / 2.0
        for i, (s, x0, nc, nr) in enumerate(placed):
            s.component.draw(ax)
            if ticks:
                s.component.draw_ticks(ax, cell_in=scale)    # auto tick density at the actual render scale
            ax.text(x0 + nc / 2.0, -1.0, s.name, ha="center", va="bottom", fontsize=8.5, weight="bold")
            if i > 0 and s.transform:
                _flow_arrow(ax, x0 - gap + 0.4, x0 - 0.4, arrow_y, s.transform, fs=7, lw=1.6, label_dy=-0.6)
        # -- header: distribution box(es) then legend, growing DOWN, ending just above the name row --
        y = -(header_h - 0.5)                                # top anchor; block bottoms out near the tiles
        if show_info:
            for key, names in uniq:
                lines = [f"static tile distribution  [{', '.join(names)}]"] + list(key)
                ax.text(0.0, y, "\n".join(lines), ha="left", va="top", fontsize=7.0,
                        family="monospace", linespacing=1.3, zorder=6,
                        bbox=dict(boxstyle="round,pad=0.4", facecolor="#f4f4f4", edgecolor="0.3", lw=1.0))
                y += len(lines) * LINE + BOXGAP              # legend sits right under the box (tight)
        if show_legend:
            _legend_in_axes(ax, 0.0, y)
        ax.set_xlim(-2.0, total_w + 2.0)
        ax.set_ylim(maxnr + 2.0, -(header_h + 1.5))         # inverted (origin top-left); tight header room
        fig.suptitle(self.title or "pipeline dataflow", fontsize=11, y=0.995)
        return _save_fig(fig, out_path, dpi, tight=True)

    def render_panels(self, out_path, dpi=200, panel_h_in=9.0, pw_min=2.0, pw_max=5.0, gap_in=1.2,
                      title=None, lpad=0.85, tpad=1.15, bpad=0.7, rpad=0.4):
        """Compose the stages as a ROW OF PANELS at a COMMON HEIGHT -- each stage gets its OWN axes at the
        same height, so a tall M×K tile and a 1024-tall register file sit side by side with their wave bands
        ALIGNED, instead of being crammed into one shared-aspect grid that ribbons. Each panel's WIDTH tracks
        its aspect (``panel_h * nc/nr``) clamped to ``[pw_min, pw_max]`` -- so a WIDE K×N tile gets room for
        its (vertical) wave bands while tall panels stay narrow; the block label auto-rotates to run along a
        tall-thin band. Ticks auto-tune to each panel's own cell size. Use this for MACRO/WG overviews; the
        shared-axes ``render`` stays for detailed wave-scope flows where stage heights already match."""
        plt = _plt()
        stages = list(self.stages)
        n = max(1, len(stages))
        geom = []                                            # (stage, nc, nr, panel_w) -- width tracks aspect
        for s in stages:
            nc, nr = s.component.grid_size()
            pw = min(pw_max, max(pw_min, panel_h_in * nc / max(1, nr)))
            geom.append((s, nc, nr, pw))
        fig_w = lpad + rpad + sum(g[3] for g in geom) + gap_in * (n - 1)
        fig_h = tpad + panel_h_in + bpad
        fig = plt.figure(figsize=(fig_w, fig_h))
        ov = fig.add_axes([0, 0, 1, 1]); ov.axis("off"); ov.set_xlim(0, 1); ov.set_ylim(0, 1)
        ymid = (bpad + panel_h_in / 2) / fig_h
        x = lpad; prev_r = None
        for s, nc, nr, pw in geom:
            s.component.cell_w = pw / max(1, nc); s.component.cell_h = panel_h_in / max(1, nr)
            s.component.origin = (0.0, 0.0)
            ax = fig.add_axes([x / fig_w, bpad / fig_h, pw / fig_w, panel_h_in / fig_h])
            s.component.draw(ax)
            ax.set_xlim(0, nc); ax.set_ylim(nr, 0); ax.set_aspect("auto")   # non-square cells fill the panel
            s.component._ticks(ax)
            ov.text((x + pw / 2) / fig_w, (bpad - 0.45) / fig_h, s.name, ha="center", va="top",
                    fontsize=10, weight="bold")
            if prev_r is not None and s.transform:                          # arrow + transform name in the gap
                _flow_arrow(ov, prev_r / fig_w, x / fig_w, ymid, s.transform, fs=8, lw=1.8, label_dy=0.015)
            prev_r = x + pw; x += pw + gap_in
        fig.suptitle(title or self.title or "pipeline dataflow", fontsize=12, y=0.99)
        return _save_fig(fig, out_path, dpi)


@dataclass
class WaveStrip:
    """WG-scope register stage: N per-wave 64-lane register files SIDE BY SIDE (one panel per wave), each
    carrying its wave's flowed data (absolute coords -- the wave's tile offset is baked in). A drop-in
    :class:`Pipeline` stage component (duck-typed ``grid_size``/``origin``/``draw``/``draw_ticks`` + ``dist``
    for the header + ``highlight`` forwarded to every panel for the trace). This is the answer to 'show the
    whole workgroup' WITHOUT a 1024-lane monolith: every wave keeps its own 64-lane file (tid vertical / reg
    horizontal, the normal register-file orientation), laid out left to right. Build with
    :meth:`from_fwd_map`."""
    panels: tuple = ()                          # of RegisterFileComponent (one per wave), lane-local 0..63
    gap: float = 3.0
    origin: tuple = (0.0, 0.0)
    highlight: frozenset = frozenset()
    highlight_color: str = "#d81b60"
    dist: object = None                         # shared per-wave encoding (for FlowStage.summary); may be None
    font_size: float = 5.0

    @classmethod
    def from_fwd_map(cls, fwd_map, *, dims, wave_size=64, dist=None, font_size=5.0, gap=3.0,
                     dtype_bits=16, shade_addr=None):
        """Split a full ``{(tid,reg)->coord}`` map into one lane-local 64-lane ``RegisterFileComponent`` per
        wave (``tid // wave_size``), each drawn in the NORMAL orientation (tid rows / reg cols). ``shade_addr``
        (a ``(c0,c1)->addr`` memory order) shades each panel by vectorization TIME (one contiguous run = one
        shade), computed per wave via :func:`transactions`; without it the shade falls back to the group."""
        waves = sorted({tid // wave_size for tid, _ in fwd_map})
        panels = []
        for w in waves:
            local = {(tid - w * wave_size, reg): coord
                     for (tid, reg), coord in fwd_map.items() if tid // wave_size == w}
            shade = transactions(local, shade_addr)[0] if shade_addr is not None else None
            panels.append(RegisterFileComponent(fwd_map=local, dims=dims, col_ticks_side="top",
                                                font_size=font_size, dtype_bits=dtype_bits, shade_map=shade))
        return cls(panels=tuple(panels), gap=gap, dist=dist, font_size=font_size)

    def grid_size(self):
        ws = [p.grid_size() for p in self.panels]
        return (sum(w for w, _ in ws) + self.gap * max(0, len(ws) - 1),
                max((h for _, h in ws), default=1))

    def _place(self):
        ox, oy = self.origin
        x = ox
        for p in self.panels:
            p.origin = (x, oy)
            x += p.grid_size()[0] + self.gap

    def draw(self, ax):
        self._place()
        for i, p in enumerate(self.panels):
            p.highlight = self.highlight
            p.highlight_color = self.highlight_color
            p.draw(ax)
            px, py = p.origin
            pw, _ = p.grid_size()
            ax.text(px + pw / 2.0, py - 0.9, f"wave {i}", ha="center", va="bottom",
                    fontsize=self.font_size, weight="bold", color="#444")

    def draw_ticks(self, ax, cell_in=0.2):
        self._place()
        for p in self.panels:
            p.draw_ticks(ax, cell_in=cell_in)


# ---------------------------------------------------------------- workflow recipes (compose on the spine)
def transform_note(src, tgt):
    """One-line description of the register transform from ``src`` to ``tgt`` (each a
    ``{(lane,reg)->coord}`` forward map or a ``WarpDistributionEncoding``): ``identity`` (free relabel,
    no data movement) | ``in-register reorder`` (a lane-uniform register permutation -- e.g. the
    interleave transpose) | ``cross-lane (LDS / DPP)`` (needs ds_bpermute / LDS round-trip). Uses the
    validated ``classify_transform`` so the workflow arrows can name the cost of each hop."""
    s = as_forward_map(src); t = as_forward_map(tgt)
    if s == t:
        return "identity (free relabel)"
    try:
        tier = classify_transform(s, t).tier
    except Exception:
        return "transform"
    return {"reorder": "in-register reorder", "cross_lane": "cross-lane (LDS / DPP)"}.get(tier, tier)


def flow_mem_to_register(dist, dims=("M", "K"), *, title=""):
    """Workflow 1 -- 'show the memory -> register mapping of A': the logical tile (global) beside its
    register file (after load), sharing the origin so any datum is traceable across both. Call
    ``.trace((m,k))`` then ``.render(path)``."""
    return Pipeline(stages=(
        FlowStage(f"{dims[0]}x{dims[1]} logical (global)",
                  LogicalTileComponent(dist=dist, dims=dims, label_coords="logical", font_size=6.0)),
        FlowStage("register file", RegisterFileComponent(dist=dist, dims=dims, font_size=6.0,
                                                         col_ticks_side="top"),
                  transform="global load"),
    ), title=title or f"{dims[0]} x {dims[1]}: memory -> register mapping")


def flow_lds_to_register(mp, addr_fn, read_dist, dims=("M", "K"), *, nbanks=32, title=""):
    """Workflow 2 -- 'show the LDS -> register mapping of a tile': the LDS bank placement beside the
    register file after the wave read. ``mp`` is the store's register map ({(lane,reg)->(row,col)});
    ``read_dist`` maps (lane,reg) -> the global datum the wave read brings into each register."""
    return Pipeline(stages=(
        FlowStage("LDS (depth x banks)",
                  LdsBankView(mp=mp, addr_fn=addr_fn, nbanks=nbanks, dims=dims, font_size=6.0)),
        FlowStage("register file", RegisterFileComponent(dist=read_dist, dims=dims, font_size=6.0,
                                                         col_ticks_side="top"),
                  transform="wave read"),
    ), title=title or f"{dims[0]} x {dims[1]}: LDS -> register mapping")


def flow_kloop_operand(load_dist, store_mp, store_addr, read_dist, dims=("M", "K"), *, nbanks=32,
                       name="A", title=""):
    """Workflow 5 -- 'show the whole K-loop dataflow' for one operand, as one wide traceable strip:
    global tile -> registers (global load) -> LDS (store) -> registers (wave read = the MMA operand).
    Every stage carries the ORIGINAL global datum, so ``.trace((m,k))`` lights it end-to-end (up to the
    MMA, where it is consumed into the whole C row/col). ``load_dist`` = the global-load distribution;
    ``store_mp`` ({(lane,reg)->(row,col)}) + ``store_addr`` = the LDS store; ``read_dist`` = the wave-read
    (MMA-operand) distribution. Compose two of these (A + B) with :func:`flow_wave_mma` for the full loop."""
    return Pipeline(stages=(
        FlowStage(f"{name} global tile",
                  LogicalTileComponent(dist=load_dist, dims=dims, label_coords="logical")),
        FlowStage(f"{name} regs (load)",
                  RegisterFileComponent(dist=load_dist, dims=dims, col_ticks_side="top"),
                  transform="global load"),
        FlowStage(f"{name} in LDS",
                  LdsBankView(mp=store_mp, addr_fn=store_addr, nbanks=nbanks, dims=dims),
                  transform="LDS store"),
        FlowStage(f"{name} regs (MMA operand)",
                  RegisterFileComponent(dist=read_dist, dims=dims, col_ticks_side="top"),
                  transform=f"wave read\n[{transform_note(load_dist, read_dist)}]"),
    ), title=title or f"{name} K-loop dataflow: global -> regs -> LDS -> MMA-operand regs")


def flow_wave_mma(mma, **overrides):
    """Workflow 3 -- 'show the wave-tile MMA': the A/B register-file wings feeding the machine-derived C
    (logical body + C register file), with each operand's boxed static distribution (A at A's top-left,
    B above B, C under the C register file). This IS the :class:`MmaTee`; exposed as a named workflow.
    Returns an ``MmaTee`` (render via ``.render(out_dir, name=...)``)."""
    overrides.setdefault("show_static", True)
    return MmaTee.from_mma(mma, **overrides)


def _dtype_bits(name):
    """Element bit width from a dtype tag ('f16'->16, 'bf16'->16, 'f32'->32, 'f64'->64, 'i8'->8)."""
    digits = "".join(c for c in str(name) if c.isdigit())
    return int(digits) if digits else 32


@dataclass
class MmaTee:
    """Automated MMA-tee generator: composes the two components (A/B ``RegisterFileComponent`` wings +
    C ``LogicalTileComponent`` body) into a tee, deriving groups / detail / orientation / placement /
    sizing from the ATOM + distributions. Drive it two ways: pass everything explicitly, or unpack a
    :class:`TileMma` via :meth:`from_mma` (TileMma is just a wrapper that fills these same fields)."""
    a_enc: object                              # SUPPLIED A: a WarpDistributionEncoding OR a pre-populated
    b_enc: object                              # register map {(lane,reg)->(coord)} from ANOTHER STAGE.
    c_enc: object                              # canonical C (fallback only; C is DERIVED when canon refs given)
    atom_shape: tuple                          # (atom_m, atom_n, atom_k)
    a_canon: WarpDistributionEncoding | None = None   # canonical refs = the fixed machine coupling; C is
    b_canon: WarpDistributionEncoding | None = None   # derived by flowing the SUPPLIED A/B labels through them
    c_canon: WarpDistributionEncoding | None = None   # (docs/mma_is_machinery.md). None -> trust c_enc.
    a_dtype_bits: int = 16
    b_dtype_bits: int = 16
    c_dtype_bits: int = 32
    in_dtype: str = ""                         # input (A/B) dtype name, e.g. "f16"; blank -> shown as bits
    out_dtype: str = ""                        # output (C) dtype name, e.g. "f32"
    dims_a: tuple = ("M", "K")
    dims_b: tuple = ("N", "K")
    dims_c: tuple = ("M", "N")
    issue_order: str = "A"                     # "A" (M-outer) | "B" (N-outer): sets the MMA tile sequence
    full_detail: tuple = (0,)                  # issue-order ordinals of the TILES drawn detailed
    op_id: str = ""
    wave_shape: tuple | None = None
    color_mode: str = "first8"
    cell: float = 0.40                         # inches per grid cell (sized so ~6-char labels fit)
    font_size: float = 5.0
    gap: int = 2
    show_logical_inputs: bool = False          # logical A left of A regfile (arrow ->), logical B above B (arrow v)
    show_static: bool = False                  # print the full static distribution + tile sizes (A left, B above)
    trace_a: tuple | None = None               # (m, k): FOLLOW this A element across the tee (provenance)
    trace_b: tuple | None = None               # (n, k): matched-K B element -> lights the chain -> C[m, n]
    show_diagnostics: bool = True              # run the K-match DIAGNOSTIC (observer) and annotate warn/error
    show_legend: bool = True                   # draw the shade legend (hue=thread, shade=vectorized-load order)
    vec_a: str = ""                            # preferred vectorized axis for A (expert-reflected); "" -> derived
    vec_b: str = ""                            # preferred vectorized axis for B (expert-reflected); "" -> derived
    vec_c: str = ""                            # preferred vectorized axis for C store (expert); "" -> derived

    @classmethod
    def from_mma(cls, mma, **overrides):
        """Build from a ``TileMma`` wrapper -- pulls the A/B/C distributions, atom shape, dtypes, op_id,
        wave shape, and the canonical machine refs (so C is derived by flowing the A/B labels through the
        machine). Supply a different ``a_enc``/``b_enc`` via overrides to feed non-canonical distributions."""
        t = mma.traits
        base = dict(a_enc=mma.a_layout, b_enc=mma.b_layout, c_enc=mma.c_layout, atom_shape=mma.atom_shape,
                    a_canon=mma.a_layout, b_canon=mma.b_layout, c_canon=mma.c_layout,
                    a_dtype_bits=_dtype_bits(t.input_dtype), b_dtype_bits=_dtype_bits(t.input_dtype),
                    c_dtype_bits=_dtype_bits(t.output_dtype), in_dtype=t.input_dtype, out_dtype=t.output_dtype,
                    op_id=mma.op_id, wave_shape=mma.shape)
        base.update(overrides)
        return cls(**base)

    # -- group derivation (from the atom + the label maps) --------------------
    def _reg_groups(self, canon, fwd, free_atom, detailed_frees):
        """MMA-input register groups = WHICH physical register slots feed WHICH MMA atom. This is a property
        of the MACHINE (``canon``), NOT of the supplied labels: partition every physical ``(lane,reg)`` slot
        by its canonical atom ``(free // free_atom, K // atom_k)``. Grouping from the machine (not tid0's
        labels) keeps the borders STABLE and register-aligned regardless of what labels (interleaved,
        corrupted) sit in the slots. Falls back to the supplied ``fwd`` only when no machine ref is given.
        A subtile is 'detailed' iff its free-atom is fed by a detailed tile."""
        atom_k = self.atom_shape[2]
        if canon is not None:
            cm = RegisterMapper(canon)
            keyed = {(l, r): cm.matrix_coordinates(l, r)[:2]
                     for l in range(cm.num_lanes) for r in range(cm.num_vector_items)}
        else:
            keyed = {s: tuple(fwd[s][:2]) for s in fwd}
        g: dict[tuple[int, int], list[tuple[int, int]]] = {}
        for slot, (f, k) in keyed.items():
            g.setdefault((f // free_atom, k // atom_k), []).append(slot)
        return tuple(RegGroup(frozenset(cells),
                              "detailed" if key[0] in detailed_frees else "grouped", str(key))
                     for key, cells in sorted(g.items()))

    @staticmethod
    def _tile_order(keys, seq):
        """Order C atom-tile keys by the MMA ISSUE sequence (so the shade step == issue order); tiles not
        in seq trail at the end. Falls back to sorted() when no seq is given."""
        if not seq:
            return sorted(keys)
        rank = {k: i for i, k in enumerate(seq)}
        return sorted(keys, key=lambda k: (rank.get(k, len(seq)), k))

    def _c_groups_from_coords(self, coords, detailed_tiles, seq=None):
        """C atom-tile groups from logical (m, n) coords (works for the derived owner map too). Grouped by
        (m//atom_m, n//atom_n) and ordered by the MMA issue sequence."""
        atom_m, atom_n, _ = self.atom_shape; g = {}
        for c0, c1 in coords:
            g.setdefault((c0 // atom_m, c1 // atom_n), []).append((c0, c1))
        return tuple(LogicalGroup(frozenset(g[key]), "detailed" if key in detailed_tiles else "grouped", str(key))
                     for key in self._tile_order(g.keys(), seq))

    def _c_reg_groups(self, canon, fwd, detailed_tiles, seq=None):
        """C REGISTER-FILE groups = WHICH physical C register slots belong to WHICH MMA atom tile. Like the
        A/B wings, this is a property of the MACHINE (``canon``): partition every physical ``(lane,reg)`` C
        slot by its canonical atom ``(Mc//atom_m, Nc//atom_n)`` -- stable and register-aligned regardless of
        the flowed labels. Issue-ordered. Falls back to the supplied ``fwd`` labels only when no machine ref."""
        atom_m, atom_n, _ = self.atom_shape
        if canon is not None:
            cm = RegisterMapper(canon)
            keyed = {(l, r): cm.matrix_coordinates(l, r)[:2]
                     for l in range(cm.num_lanes) for r in range(cm.num_vector_items)}
        else:
            keyed = {s: tuple(fwd[s][:2]) for s in fwd}
        g: dict[tuple[int, int], list[tuple[int, int]]] = {}
        for slot, (c0, c1) in keyed.items():
            g.setdefault((c0 // atom_m, c1 // atom_n), []).append(slot)
        return tuple(RegGroup(frozenset(g[key]), "detailed" if key in detailed_tiles else "grouped", str(key))
                     for key in self._tile_order(g.keys(), seq))

    def _logi_free_groups(self, fwd, free_atom, detailed_frees):
        """Logical-tile groups matching the register groups, from a forward map: bucket (free, K) cells by
        (free-sub, K-sub); detailed iff the free index is fed by a detailed tile."""
        atom_k = self.atom_shape[2]; g = {}
        for (lane, reg), (c0, c1) in fwd.items():
            g.setdefault((c0 // free_atom, c1 // atom_k), []).append((c0, c1))
        return tuple(LogicalGroup(frozenset(v), "detailed" if key[0] in detailed_frees else "grouped", str(key))
                     for key, v in sorted(g.items()))

    def _static_lines(self, role, fwd, dims, dbits, enc=None):
        """Static distribution stats from a forward map (tile size, dtype, lanes/regs); when an ``enc`` is
        available (the input was a distribution, not a raw stage map) also list its Rs/Hs/Ps/Ys spaces."""
        lanes = sorted({l for l, _ in fwd}); nreg = max(r for _, r in fwd) + 1
        e0 = max(c[0] for c in fwd.values()) + 1; e1 = max(c[1] for c in fwd.values()) + 1
        pack = (f"{max(1, 32 // dbits)} elem/VGPR" if dbits <= 32 else f"{dbits // 32} VGPR/elem")
        lines = [f"{role} static distribution",
                 f"  {dims[0]}x{dims[1]} tile = {e0} x {e1}",
                 f"  atom (MxNxK) = {self.atom_shape[0]}x{self.atom_shape[1]}x{self.atom_shape[2]}",
                 f"  dtype = {dbits}-bit ({pack})",
                 f"  lanes={len(lanes)}  vregs/lane={nreg}"]
        if enc is not None:
            ps = tuple((maj, mn) for maj, mn in zip(enc.lane_to_rh_major, enc.lane_to_rh_minor))
            ys = tuple(zip(enc.register_to_rh_major, enc.register_to_rh_minor))
            lines += [f"  Rs (replicate) = {tuple(enc.replication_lengths)}",
                      f"  Hs (hier)      = {tuple(enc.hierarchical_lengths)}",
                      f"  Ps (lane->rh)  = {ps}",
                      f"  Ys (reg->rh)   = {ys}"]
        else:
            lines.append("  (stage map input -- no encoding spaces)")
        return lines

    def c_mapping(self):
        """EXPORT: the derived C register mapping WITH LABELS -- ``{(lane, reg) -> (m, n)}`` -- for handing
        to another stage. Flows the supplied A/B labels through the canonical machine (docs/mma_is_machinery
        .md). Requires the canonical machine refs (from :meth:`from_mma`)."""
        if self.a_canon is None or self.b_canon is None or self.c_canon is None:
            return as_forward_map(self.c_enc)
        return derive_c_distribution(self.a_enc, self.b_enc,
                                     a_canon=self.a_canon, b_canon=self.b_canon, c_canon=self.c_canon)

    def _build_core(self):
        """Build the tee's core components (A/B register-file wings, derived C body, C register file) at
        their tee origins, plus the maps/metadata both ``render`` and the exploded view need. Extracted
        so the exploded workflow can reuse the EXACT wings/C (no convention drift)."""
        atom_m, atom_n, _ = self.atom_shape
        # A/B may be a distribution (encoding) OR a pre-populated register mapping+labels from ANOTHER STAGE.
        # Normalize to forward maps; the tee flows these labels the same way regardless of their source.
        a_map = as_forward_map(self.a_enc); b_map = as_forward_map(self.b_enc)
        a_enc_obj = None if isinstance(self.a_enc, dict) else self.a_enc     # encoding present -> static spaces
        b_enc_obj = None if isinstance(self.b_enc, dict) else self.b_enc
        c_map = self.c_mapping()                                             # DERIVED C register mapping+labels
        e0 = max(c[0] for c in c_map.values()) + 1; e1 = max(c[1] for c in c_map.values()) + 1
        m_iter, n_iter = max(1, e0 // atom_m), max(1, e1 // atom_n)
        seq = _issue_seq(m_iter, n_iter, self.issue_order)
        full_tiles = {seq[o] for o in self.full_detail if 0 <= o < len(seq)}
        det_m = {mi for mi, _ in full_tiles}; det_n = {nj for _, nj in full_tiles}
        # provenance/follow trace: light A(m,k) and B(n,k) in their panels, C[m,n] in the C panels.
        ha = frozenset({tuple(self.trace_a)}) if self.trace_a else frozenset()
        hb = frozenset({tuple(self.trace_b)}) if self.trace_b else frozenset()
        hc = frozenset({(self.trace_a[0], self.trace_b[0])}) if (self.trace_a and self.trace_b) else frozenset()

        nra = max(r for _, r in a_map) + 1; nrb = max(r for _, r in b_map) + 1
        x0, y0 = nra + self.gap, nrb + self.gap
        # SHADE = per-thread vectorized-load / feed-iteration order (contiguous register runs), DECOUPLED from
        # the atom borders: run i = registers [i*run .. ), one shade per load. run = regs / free-dim atoms
        # (m_iter for A, n_iter for B) so an interleaved thread shows its N loads as N shades (not 1).
        run_a = max(1, nra // m_iter); run_b = max(1, nrb // n_iter)
        shade_a = {(l, r): r // run_a for (l, r) in a_map}
        shade_b = {(l, r): r // run_b for (l, r) in b_map}
        # wing conventions baked in: A reg0 at RIGHT (col desc), B reg0 at BOTTOM (row desc); C logical coords.
        # RULERS face OUTWARD, away from the C body (bottom-right): A left of C -> vreg TOP / tid LEFT;
        # B above C -> tid TOP / vreg RIGHT; C -> N BOTTOM / M RIGHT.
        A = RegisterFileComponent(fwd_map=a_map, dims=self.dims_a, row_axis="tid", col_axis="vreg",
                                  col_order="desc", dtype_bits=self.a_dtype_bits, color_mode=self.color_mode,
                                  col_ticks_side="top", row_ticks_side="left", highlight=ha, shade_map=shade_a,
                                  font_size=self.font_size, groups=self._reg_groups(self.a_canon, a_map, atom_m, det_m),
                                  origin=(0, y0))
        B = RegisterFileComponent(fwd_map=b_map, dims=self.dims_b, row_axis="vreg", col_axis="tid",
                                  row_order="desc", dtype_bits=self.b_dtype_bits, color_mode=self.color_mode,
                                  col_ticks_side="top", row_ticks_side="left", highlight=hb, shade_map=shade_b,
                                  font_size=self.font_size, groups=self._reg_groups(self.b_canon, b_map, atom_n, det_n),
                                  origin=(x0, 0))
        # C is DERIVED by flowing the SUPPLIED A/B labels through the canonical machine (docs/mma_is_machinery.md).
        # POSITION != LABEL: the C body grid is the machine's fixed canonical POSITION (Mc,Nc); each position's
        # cell text is the FLOWED (M,N) LABEL that landed there (canonical pos [1,1] labeled M4N2, never "C[M4,N2]").
        if self.c_canon is not None:
            ccm = RegisterMapper(self.c_canon)
            pos_of = {(l, r): tuple(ccm.matrix_coordinates(l, r)[:2])
                      for l in range(ccm.num_lanes) for r in range(ccm.num_vector_items)}
            c_pos_owner = {pos_of[s]: s for s in c_map}              # canonical POSITION -> physical slot
            c_pos_text = {pos_of[s]: tuple(c_map[s]) for s in c_map}  # canonical POSITION -> FLOWED (M,N) label
            # C output orientation: CANONICAL iff every position's flowed label == its position (store-ready as
            # is); else SWIZZLED -- classify the C-shuffle to a COALESCED store built from C's OWN ownership
            # (the interleaved store, the useful target), NOT to canonical. Interleaved C owns a rectangular
            # per-lane tile, so that shuffle is a cheap in-register reorder; only forcing strict canonical
            # (a different per-lane element set) is the drastic cross-lane move.
            if all(p == lab for p, lab in c_pos_text.items()):
                c_out_note = "CANONICAL (store-ready)"
            else:
                _lanes: dict[int, list] = {}
                for (l, r), coord in c_map.items():
                    _lanes.setdefault(l, []).append(coord)
                store_tgt = {}                                # same ownership, registers ordered for a wide store
                for l, cells in _lanes.items():
                    for r, coord in enumerate(sorted(cells, key=lambda mn: (mn[1], mn[0]))):
                        store_tgt[(l, r)] = coord
                tier = classify_transform(c_map, store_tgt).tier
                c_out_note = ("SWIZZLED -> store via in-reg reorder" if tier == "reorder"
                              else "SWIZZLED -> store needs cross-lane")
            hc_pos = frozenset(p for p, lab in c_pos_text.items() if lab in hc)   # trace by flowed label -> position
            C = LogicalTileComponent(owner_map=c_pos_owner, text_map=c_pos_text, dims=self.dims_c, row_coord=0,
                                     color_mode=self.color_mode, font_size=self.font_size,
                                     col_ticks_side="top", row_ticks_side="left", highlight=hc_pos,
                                     groups=self._c_groups_from_coords(c_pos_owner.keys(), full_tiles, seq),
                                     origin=(x0, y0))
        else:                                                       # no machine reference -> logical-matrix fallback
            c_out_note = "unknown (no canonical reference)"
            c_owner = {coord: slot for slot, coord in c_map.items()}
            C = LogicalTileComponent(owner_map=c_owner, dims=self.dims_c, row_coord=0, label_coords="logical",
                                     color_mode=self.color_mode, font_size=self.font_size,
                                     col_ticks_side="top", row_ticks_side="left", highlight=hc,
                                     groups=self._c_groups_from_coords(c_owner.keys(), full_tiles, seq), origin=(x0, y0))

        aw, ah = A.grid_size(); bw, _bh = B.grid_size(); cw, ch = C.grid_size()
        # C REGISTER FILE beneath the C logical tiles: vreg vertical / tid horizontal, coloured to MATCH the
        # C blocks (hue = lane%8, shade = issue-order group). Groups = the order the MMAs are performed.
        CR = RegisterFileComponent(fwd_map=c_map, dims=self.dims_c, row_axis="vreg", col_axis="tid",
                                   dtype_bits=self.c_dtype_bits, color_mode=self.color_mode,
                                   font_size=self.font_size, col_ticks_side="top", row_ticks_side="left",
                                   highlight=hc, groups=self._c_reg_groups(self.c_canon, c_map, full_tiles, seq),
                                   origin=(x0, y0 + ch + self.gap + 2))
        crw, crh = CR.grid_size(); cr_y = y0 + ch + self.gap + 2
        return SimpleNamespace(
            A=A, B=B, C=C, CR=CR, x0=x0, y0=y0, ch=ch, cr_y=cr_y, aw=aw, ah=ah, bw=bw, cw=cw,
            crw=crw, crh=crh, a_map=a_map, b_map=b_map, c_map=c_map, c_out_note=c_out_note, seq=seq,
            full_tiles=full_tiles, atom_m=atom_m, atom_n=atom_n, det_m=det_m, det_n=det_n, ha=ha,
            hb=hb, hc=hc, shade_a=shade_a, shade_b=shade_b, run_a=run_a, run_b=run_b,
            m_iter=m_iter, n_iter=n_iter, a_enc_obj=a_enc_obj, b_enc_obj=b_enc_obj)

    def render(self, out_dir=".", name="mma_tee", title=None, dpi=200):
        plt = _plt(); os.makedirs(out_dir, exist_ok=True)
        co = self._build_core()
        A, B, C, CR = co.A, co.B, co.C, co.CR
        x0, y0, ch, cr_y = co.x0, co.y0, co.ch, co.cr_y
        aw, ah, bw, cw, crw, crh = co.aw, co.ah, co.bw, co.cw, co.crw, co.crh
        a_map, b_map, c_map = co.a_map, co.b_map, co.c_map
        c_out_note, seq, full_tiles = co.c_out_note, co.seq, co.full_tiles
        atom_m, atom_n, det_m, det_n = co.atom_m, co.atom_n, co.det_m, co.det_n
        ha, hb, hc, run_a, run_b = co.ha, co.hb, co.hc, co.run_a, co.run_b
        m_iter, n_iter = co.m_iter, co.n_iter
        a_enc_obj, b_enc_obj = co.a_enc_obj, co.b_enc_obj
        comps = [A, B, C, CR]; arrows = []; texts = []; pt = self.font_size + 5
        titles = [(aw / 2, y0 - 1.8, "A Register File"), (x0 + bw / 2, -1.8, "B Register File"),
                  (x0 + cw / 2, y0 - 1.8, "C Logical Tiles"), (x0 + crw / 2, cr_y - 1.8, "C Register File")]
        # C static distribution stats UNDER the C register file (machine C = canonical spaces).
        cbot = cr_y + crh
        texts.append((x0, cbot + 2.5, "left", "top",
                      self._static_lines("C", c_map, self.dims_c, self.c_dtype_bits, enc=self.c_canon)))
        left = top = 0.0; ARROW = 3.0; cstat_h = 12

        if self.show_logical_inputs:                          # logical A LEFT (arrow ->), logical B ABOVE (arrow v)
            AL = LogicalTileComponent(owner_map={c: s for s, c in a_map.items()}, dims=self.dims_a, row_coord=0,
                                      label_coords="logical", color_mode=self.color_mode,
                                      font_size=self.font_size, highlight=ha,
                                      shade_map={c: r // run_a for (l, r), c in a_map.items()},
                                      groups=self._logi_free_groups(a_map, atom_m, det_m))
            alc, alr = AL.grid_size(); AL.origin = (-(alc + ARROW), y0); comps.append(AL)
            my = y0 + min(alr, ah) / 2.0; arrows.append(((-ARROW, my), (0, my)))
            titles.append((AL.origin[0] + alc / 2, y0 - 1.8, "A Logical Tile")); left = AL.origin[0]
            BL = LogicalTileComponent(owner_map={c: s for s, c in b_map.items()}, dims=self.dims_b, row_coord=1,
                                      label_coords="logical", color_mode=self.color_mode,
                                      font_size=self.font_size, highlight=hb,
                                      shade_map={c: r // run_b for (l, r), c in b_map.items()},
                                      groups=self._logi_free_groups(b_map, atom_n, det_n))
            blc, blr = BL.grid_size(); BL.origin = (x0, -(blr + ARROW)); comps.append(BL)
            mx = x0 + min(blc, bw) / 2.0; arrows.append(((mx, -ARROW), (mx, 0)))
            titles.append((x0 + blc / 2, BL.origin[1] - 1.8, "B Logical Tile")); top = BL.origin[1]

        if self.show_static:                                  # full static distribution + tile sizes: A left, B above
            a_lines = self._static_lines("A", a_map, self.dims_a, self.a_dtype_bits, enc=a_enc_obj)
            b_lines = self._static_lines("B", b_map, self.dims_b, self.b_dtype_bits, enc=b_enc_obj)
            axx = (left if self.show_logical_inputs else 0) - 3          # A static LEFT of A logical/regfile
            texts.append((axx, y0, "right", "top", a_lines))
            left = min(left, axx - 40)
            byy = (top if self.show_logical_inputs else 0) - 3          # B static ABOVE B logical/regfile (grows up)
            texts.append((x0, byy, "left", "bottom", b_lines))
            top = min(top, byy - (len(b_lines) + 3))

        xmax = x0 + max(cw, bw, crw); ymax = max(y0 + max(ah, ch), cbot + cstat_h)
        fig = plt.figure(figsize=((xmax - left + 12) * self.cell, (ymax - top + 12) * self.cell))
        ax = fig.add_subplot(111)
        for comp in comps:
            comp.draw(ax); comp.draw_ticks(ax)
        for (sx, sy), (ex, ey) in arrows:
            ax.annotate("", xy=(ex, ey), xytext=(sx, sy), arrowprops=dict(arrowstyle="-|>", lw=2.4, color="black"))
        for tx, ty, ha, va, lines in texts:                   # each static distribution in its OWN box
            ax.text(tx, ty, "\n".join(lines), ha=ha, va=va, fontsize=self.font_size + 1,
                    family="monospace", zorder=6,
                    bbox=dict(boxstyle="round,pad=0.4", facecolor="#f4f4f4", edgecolor="0.3", lw=1.0))
        for tx, ty, lab in titles:
            ax.text(tx, ty, lab, ha="center", va="bottom", fontsize=pt, weight="bold")
        if self.show_diagnostics:                              # DIAGNOSTIC only -- observes labels, never mutates C
            # full soundness + K-match when the machine refs are present; else K-match alone.
            if self.a_canon is not None and self.b_canon is not None:
                diag = mma_pair_compatible(self.a_enc, self.b_enc, a_canon=self.a_canon, b_canon=self.b_canon)
                tag = "MMA-compatible"
            else:
                diag = diagnose_k_match(self.a_enc, self.b_enc); tag = "K-match"
            if diag.severity != "ok":
                col = "#c00000" if diag.severity == "error" else "#c07000"
                ax.text(x0 + cw / 2, ymax + 2.2, f"{diag.severity.upper()} ({tag}): {diag.message}",
                        ha="center", va="top", fontsize=pt, weight="bold", color=col,
                        bbox=dict(boxstyle="round", facecolor="white", edgecolor=col, lw=1.5))
        # INFO PANEL (top-left corner): atom/wave/tile/dtype/issue/vec/C-out kept IN VIEW when zoomed into the
        # data flow. Box is sized to the ACTUAL rendered text extent (measured, not estimated) so it always
        # fits; the shade legend sits below it.
        from matplotlib.patches import Rectangle as _Rect
        nlanes = max((l for l, _ in a_map), default=0) + 1
        am, an, ak = self.atom_shape
        tm, tn, tk = (self.wave_shape if self.wave_shape else (0, 0, 0))
        in_dt = self.in_dtype or f"{self.a_dtype_bits}b"; out_dt = self.out_dtype or f"{self.c_dtype_bits}b"

        def _vec_axis(fwd, dims):                            # preferred vectorized axis = the register stride-1
            c0 = fwd[(0, 0)]; c1 = fwd.get((0, 1), c0)       # axis (which coord advances between consecutive
            return dims[1] if c1[1] != c0[1] else dims[0]    # registers): K-inner -> K, free-inner -> M/N
        vec_a = self.vec_a or _vec_axis(a_map, self.dims_a)
        vec_b = self.vec_b or _vec_axis(b_map, self.dims_b)
        vec_c = self.vec_c or _vec_axis(c_map, self.dims_c)
        info = [title or f"MMA tee: {self.op_id or 'raw'}",
                f"atom   {am}x{an}x{ak}", f"wave   {nlanes} lanes",
                f"tile   {tm}x{tn}x{tk}", f"dtype  in {in_dt}  out {out_dt}",
                f"issue  {self.issue_order}-major", f"vec    A:{vec_a}  B:{vec_b}  C:{vec_c}",
                f"C out  {c_out_note}"]
        if self.trace_a and self.trace_b:
            m, ka = self.trace_a; n, kb = self.trace_b
            info.append(f"follow {self.dims_a[0]}{m}{self.dims_a[1]}{ka} x {self.dims_b[0]}{n}{self.dims_b[1]}{kb}"
                        + ("" if ka == kb else "  [K MISMATCH]"))

        ax.set_xlim(left - 3, xmax + 1); ax.set_ylim(ymax + 4, top - 4); ax.set_aspect("equal"); ax.axis("off")
        fs = 9                                               # compact; the box is fit to the measured extent
        # Anchor the panel ABOVE the content (top-left margin) and grow it UPWARD, so it never overlaps the
        # A/B wings (which start at y>=0). The shade legend sits ABOVE the box, further into the margin.
        panel_bottom = top - 1.5
        txt = ax.text(1.0, panel_bottom, "\n".join(info), ha="left", va="bottom", zorder=5,
                      fontsize=fs, family="monospace", weight="bold", linespacing=1.3)
        fig.canvas.draw()                                    # realize text so we can measure its true extent
        bb = txt.get_window_extent(fig.canvas.get_renderer()).transformed(ax.transData.inverted())
        x0b, x1b = min(bb.x0, bb.x1), max(bb.x0, bb.x1); y0b, y1b = min(bb.y0, bb.y1), max(bb.y0, bb.y1)
        pad = 0.6
        ax.add_patch(_Rect((x0b - pad, y0b - pad), (x1b - x0b) + 2 * pad, (y1b - y0b) + 2 * pad,
                           facecolor="#f2f2f2", edgecolor="0.2", lw=1.4, zorder=3))
        panel_extent_top = y0b - pad                         # highest (smallest-y) point of the info box
        if self.show_legend:                                 # shade = vectorized-load issue order; hue = thread
            nload = max(2, m_iter); sw = 1.6; ly = y0b - pad - 1.7   # swatches ABOVE the info box
            for i in range(nload):
                ax.add_patch(_Rect((x0b - pad + i * sw, ly), sw * 0.85, 1.1, facecolor=cell_rgb(0, i, nload),
                                   edgecolor="0.4", lw=0.6, zorder=4))
                ax.text(x0b - pad + i * sw + sw * 0.425, ly + 0.55, f"load{i}", ha="center", va="center",
                        fontsize=fs - 2, zorder=6)
            ax.text(x0b - pad, ly - 0.5, "shade = load issue order (0..N/thread);  hue = thread",
                    ha="left", va="bottom", fontsize=fs - 1, family="monospace", zorder=5)
            panel_extent_top = ly - 1.6
        ax.set_ylim(ymax + 4, min(top - 4, panel_extent_top - 1))   # include the panel above the content
        p = os.path.join(out_dir, f"{name}.png")
        fig.savefig(p, dpi=dpi, bbox_inches="tight"); plt.close(fig)
        return p
