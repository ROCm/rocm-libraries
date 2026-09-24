#!/usr/bin/env python3
# ##############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
# ##############################################################################
"""Visualize hipBLASLt workgroup-mapping (WGM) instrumentation output.

This is the companion tool for the per-kernel ``EnableWGMDebug`` option.

Workflow
--------
1. Set ``EnableWGMDebug: 1`` on the solutions to inspect and build hipBLASLt::

       ./install.sh -c --skip_rocroller -a gfx950

   Non-StreamK kernels overwrite the top-left of every macro-tile:

       tile[0..3] = [ original 1D WG id,
                      (post-WGM WorkGroup0 << 16) | post-WGM WorkGroup1,
                      XCC id,
                      0 ]

   StreamK kernels cannot afford persistent debug SGPRs and have no unique
   workgroup owner per tile. They park the raw launch id in AddressC and replay
   the mapping code once per tile the persistent workgroup visits, writing a
   record at ``D + (wg0 + wg1 * nwg0)*16`` after each DefaultWGM:

       record[0..3] = [ raw launch id of the last workgroup to map the tile,
                        (post-WGM WorkGroup0 << 16) | WorkGroup1,
                        0x534B0000 | XCC id,
                        packed WGM word ]

2. Run any GEMM through hipblaslt-bench (Debug build) with validation enabled
   and the dump environment variable set::

       HIPBLASLT_DEBUG_WGM_DUMP=d.bin ./hipblaslt-bench \
           --precision f32_r --transA N --transB N \
           -m 256 -n 256 -k 32 --unit_check 1

   This writes ``d.bin`` (raw D output + a small header).

3. Plot the mapping::

       python3 plot_wgm.py d.bin

Binary format written by the client (see debug_wgm_dump_d in testing_matmul.hpp):
    int32 magic == 0x57474D44 ('WGMD')
    int32 M
    int32 N
    int32 ldd                (leading dimension of D, in elements)
    int32 bytesPerElement
    <ldd * N * bytesPerElement raw bytes of the column-major D output>
"""

import argparse
import math
import struct
import sys

import numpy as np

MAGIC = 0x57474D44  # 'WGMD'
STREAMK_RECORD_MAGIC = 0x534B0000  # 'SK' in dword2[31:16]
# MI300/MI355: 8 XCDs. DP and StreamK plots share this palette so XCC k is the
# same color on every image (tab10[k]), independent of which XCDs a dump uses.
XCC_COUNT = 8
# Every cell is labelled, so the figure is sized per macro-tile and then capped
# so a large lattice (e.g. 90x128 tiles) still renders to a usable image.
CELL_INCHES_MAX = 2.0
FIG_INCHES_MAX = 110.0


def read_dump(path):
    """Read the binary D dump and return (raw_bytes, m, n, ldd, bpe).

    Data-type agnostic: the WGM instrumentation emits a dedicated raw 16-byte
    (dwordx4) store to each tile's top-left element, so the 4 diagnostic dwords
    are the first 16 bytes at that element's byte offset regardless of the
    output element size (bpe).
    """
    with open(path, "rb") as f:
        header = f.read(20)
        if len(header) < 20:
            sys.exit(f"{path}: file too small to contain a WGM dump header")
        magic, m, n, ldd, bpe = struct.unpack("<5i", header)
        if magic != MAGIC:
            sys.exit(
                f"{path}: bad magic 0x{magic:08X} (expected 0x{MAGIC:08X}); "
                "was this produced by an EnableWGMDebug build with HIPBLASLT_DEBUG_WGM_DUMP set?"
            )
        raw = f.read(ldd * n * bpe)
    if len(raw) < ldd * n * bpe:
        sys.exit(f"{path}: truncated payload ({len(raw)} < {ldd * n * bpe} bytes)")
    return raw, m, n, ldd, bpe


def build_lattice(raw, m, n, ldd, bpe, mt0, mt1):
    """Decode the per-workgroup WGM diagnostics from each tile's top-left element.

    D is column-major MxN (leading dim ldd, in elements). Element (row=m, col=n)
    starts at byte (col*ldd + row)*bpe. The instrumentation wrote 16 contiguous
    bytes (4 uint32) there. Returns lattice[wgN][wgM] = (orig_wg, (new_wg1,
    new_wg0), xcc), the XCC grid, the packed WGM value, and the walk order.
    """
    n_wg_m = math.ceil(m / mt0)
    n_wg_n = math.ceil(n / mt1)

    lattice = np.empty((n_wg_n, n_wg_m), dtype=object)
    lattice_str = np.empty((n_wg_n, n_wg_m), dtype=object)
    xcc_grid = np.zeros((n_wg_n, n_wg_m), dtype=int)
    wgm_value = None

    def read4(row, col):
        off = (col * ldd + row) * bpe
        return struct.unpack("<4I", raw[off:off + 16])

    for wg_n, col in enumerate(range(0, n, mt1)):
        for wg_m, row in enumerate(range(0, m, mt0)):
            orig_wg, packed_new, xcc, wgm = read4(row, col)

            new_wg0 = (packed_new >> 16) & 0xFFFF
            new_wg1 = packed_new & 0xFFFF
            if wgm == 0xDEADBEEF or wgm == 0xBEEFBEEF:
                sys.exit(
                    "Sentinel value found in WGM slot -- the instrumented store path "
                    "was not the one taken at runtime. Try a GEMM whose D store uses the "
                    "regular gwvw>=4 path (e.g. f32_r, beta==0)."
                )
            if wgm_value is None:
                wgm_value = wgm
            elif wgm != wgm_value:
                print(
                    f"warning: WGM value differs across tiles: 0x{wgm:08X} != 0x{wgm_value:08X}",
                    file=sys.stderr,
                )

            lattice[wg_n][wg_m] = (orig_wg, (new_wg1, new_wg0), xcc)
            lattice_str[wg_n][wg_m] = f"{orig_wg}->({new_wg1},{new_wg0})(XCC:{xcc})"
            xcc_grid[wg_n][wg_m] = xcc

    grid_size = n_wg_m * n_wg_n
    order = [None] * grid_size
    for iy, ix in np.ndindex(lattice.shape):
        orig_wg, (new_wg1, new_wg0), _ = lattice[iy][ix]
        if 0 <= orig_wg < grid_size:
            # Plot uses matrix convention: M-tile (new_wg0) on the vertical axis,
            # N-tile (new_wg1) on the horizontal axis. Store arrow points as
            # (x=new_wg1, y=new_wg0) to match the cell-text placement.
            order[orig_wg] = (new_wg1, new_wg0)

    return lattice, lattice_str, xcc_grid, wgm_value, order, (n_wg_m, n_wg_n)


def read_streamk_records(raw, n_wg_m, n_wg_n):
    """Decode StreamK per-tile records written at D + tile*16.

    Each record is:
      [writer_raw_id, (m<<16)|n, 0x534B0000 | xcc_id, packed_wgm]
    stored at the flattened tile index ``m + n * n_wg_m``, so a persistent
    workgroup leaves one record per tile it visits. Slots whose record is
    missing (tile never mapped) or inconsistent with their own index are
    skipped. Return ``None`` when the SK marker is absent anywhere, so legacy
    tile-origin dumps fall back to ``build_lattice``.
    """
    records = []
    n_slots = min(len(raw) // 16, n_wg_m * n_wg_n)
    for tile in range(n_slots):
        writer, packed_tile, tagged_xcc, packed_wgm = struct.unpack_from("<4I", raw, tile * 16)
        if (tagged_xcc & 0xFFFFFF00) != STREAMK_RECORD_MAGIC:
            continue
        wg0 = (packed_tile >> 16) & 0xFFFF
        wg1 = packed_tile & 0xFFFF
        if wg0 + wg1 * n_wg_m != tile:
            continue
        records.append((writer, wg0, wg1, tagged_xcc & 0xFF, packed_wgm))
    return records or None


def streamk_records_to_lattice(records, n_wg_m, n_wg_n):
    """Turn StreamK per-tile records into the same lattice to_plot consumes."""
    lattice = np.full((n_wg_n, n_wg_m), None, dtype=object)
    xcc_grid = np.full((n_wg_n, n_wg_m), np.nan)
    wgm_value = records[0][4]
    for writer, wg0, wg1, xcc, packed_wgm in records:
        lattice[wg1][wg0] = (writer, (wg1, wg0), xcc)
        xcc_grid[wg1][wg0] = xcc
        wgm_value = packed_wgm
    covered = len(records)
    # A tile's record names the last workgroup to map it, and a persistent
    # workgroup owns many tiles, so the walk uses one representative point per
    # launch id: the lowest-numbered tile that workgroup is seen to own.
    first_tile = {}
    for writer, wg0, wg1, _, _ in records:
        flat = wg0 + wg1 * n_wg_m
        if writer not in first_tile or flat < first_tile[writer][0]:
            first_tile[writer] = (flat, (wg1, wg0))
    order = [None] * (max(first_tile) + 1) if first_tile else []
    for writer, (_, point) in first_tile.items():
        order[writer] = point
    return lattice, xcc_grid, wgm_value, order, (n_wg_m, n_wg_n), covered


def decode_streamk_packed_wgm(packed_wgm):
    wgm_bits = packed_wgm & 0x3FF
    signed_wgm = wgm_bits - 0x400 if wgm_bits & 0x200 else wgm_bits
    return {
        "wgm": signed_wgm,
        "wgmxcc": (packed_wgm >> 10) & 0xF,
        "chunk": (packed_wgm >> 14) & 0xFF,
        "splitk": (packed_wgm >> 22) & 0x3FF,
    }


def to_plot(lattice, xcc_grid, order, wgm_value, out_path, meta=None):
    from matplotlib import patches
    from matplotlib import pyplot as plt

    meta = meta or {}
    invalid = bool(meta.get("invalid"))

    def to_text(orig_wg, new_wg0, new_wg1, xcc):
        # Launch id, this cell's macro-tile coordinate, and its physical XCD.
        return f"WG{orig_wg}\nM{new_wg0} N{new_wg1}\nXCC{xcc}"

    n_wg_n, n_wg_m = lattice.shape  # (N-tiles, M-tiles)

    # Matrix convention: M-tiles on the vertical axis (rows), N-tiles on the
    # horizontal axis (columns). xcc_grid is [wg_n][wg_m]; transpose it so rows
    # index M and columns index N.
    n_cells = n_wg_n * n_wg_m
    # Every cell carries text, so size the figure per cell: 2 inches each for
    # small grids, shrinking so a large StreamK lattice stays a sane image.
    cell_in = min(CELL_INCHES_MAX, FIG_INCHES_MAX / max(n_wg_m, n_wg_n, 1))
    if invalid:
        figsize = (min(24, max(10, n_wg_n * 0.12)), min(16, max(6, n_wg_m * 0.25)))
    else:
        figsize = (max(8, n_wg_n * cell_in), max(8, n_wg_m * cell_in))
    fig, ax = plt.subplots(figsize=figsize, dpi=80)
    plt.tight_layout()
    plt.xticks([], [])
    plt.yticks([], [])
    ax.set_xlabel("N-tiles (columns)")
    ax.set_ylabel("M-tiles (rows)")

    if invalid:
        # No valid WGM dump (e.g. uninstrumented StreamK kernel): draw only the
        # correctly-sized/oriented empty grid. No XCD colors, cell text, or arrows.
        ax.imshow(np.zeros((n_wg_m, n_wg_n)), cmap="Greys", vmin=0, vmax=1)
    else:
        from matplotlib.colors import BoundaryNorm, ListedColormap
        shown = np.asarray(xcc_grid, dtype=float).T
        cmap = ListedColormap(plt.get_cmap("tab10").colors[:XCC_COUNT])
        norm = BoundaryNorm(np.arange(-0.5, XCC_COUNT + 0.5, 1.0), cmap.N)
        image = ax.imshow(np.ma.masked_invalid(shown), cmap=cmap, norm=norm,
                          interpolation="nearest")
        bar = fig.colorbar(image, ax=ax, ticks=list(range(XCC_COUNT)),
                           fraction=0.025, pad=0.01)
        bar.set_label("XCD (HW_REG_XCC_ID)")

        # One dashed line per macro-tile edge so each cell is visibly one MT.
        ax.set_xticks(np.arange(-0.5, n_wg_n, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, n_wg_m, 1), minor=True)
        ax.tick_params(which="minor", bottom=False, left=False, length=0)
        ax.grid(which="minor", linestyle="--", linewidth=0.35, color="0.15",
                alpha=0.55)
        ax.set_axisbelow(False)
        ax.set_xlim(-0.5, n_wg_n - 0.5)
        ax.set_ylim(n_wg_m - 0.5, -0.5)

        # Cell text: launch id, macro-tile coordinate, physical XCD. Font scales
        # with the cell so it stays inside the tile on large grids.
        font_pt = max(2.5, min(12.0, 6.0 * cell_in))
        for _, z in np.ndenumerate(lattice):
            if z is None:
                continue
            orig_wg, (new_wg1, new_wg0), xcc = z
            # x = N-tile (new_wg1), y = M-tile (new_wg0)
            ax.text(new_wg1, new_wg0, to_text(orig_wg, new_wg0, new_wg1, xcc),
                    ha="center", va="center", size=font_pt, linespacing=1.0,
                    zorder=5)

        # Draw the walk order (arrows between consecutive launch ids). StreamK
        # records name the last workgroup to map a tile, so each launch id is
        # represented by the lowest-numbered tile it owns.
        n_arrows = sum(1 for p in order if p is not None)
        if n_arrows:
            # Long-range hops (StreamK, large WGM) pile up, so keep the walk a
            # thin translucent layer under the cell labels.
            dense = n_arrows > 64
            tail = max(0.04, (0.08 if dense else 0.25) * cell_in)
            head = max(1.5, (2.0 if dense else 4.0) * cell_in)
            style = f"Simple, tail_width={tail}, head_width={head}, head_length={head}"
            for i in range(1, len(order)):
                if order[i] is None or order[i - 1] is None:
                    continue
                p = patches.FancyArrowPatch(
                    (order[i - 1][0], order[i - 1][1]),
                    (order[i][0], order[i][1]),
                    connectionstyle="arc3,rad=.1",
                    arrowstyle=style,
                    color="k",
                    alpha=0.35 if dense else 0.7,
                    zorder=3,
                )
                ax.add_patch(p)

    # --- Title: WGM value + macro-tile / grid / StreamK / GSU / LSU ---
    total_wgs = n_wg_m * n_wg_n
    mt0 = meta.get("mt0")
    mt1 = meta.get("mt1")
    depthu = meta.get("depthu")
    mt_str = None
    if mt0 is not None and mt1 is not None:
        mt_str = f"{mt0}x{mt1}" + (f"x{depthu}" if depthu is not None else "")

    line1 = "WGM workgroup mapping"
    if wgm_value is not None:
        wgm_dec = meta.get("wgm")
        line1 += f"  (WGM=0x{wgm_value:08X}" + (f", WGM={wgm_dec}" if wgm_dec is not None else "") + ")"

    parts = []
    if mt_str is not None:
        parts.append(f"MacroTile {mt_str}")
    parts.append(f"Grid {n_wg_m}x{n_wg_n} ({total_wgs} WGs)")
    for key, label in (("streamk", "StreamK"), ("gsu", "GSU"), ("lsu", "LSU"),
                       ("wgm", "WGM"), ("wgmxcc", "WGMXCC")):
        if meta.get(key) is not None:
            parts.append(f"{label}={meta[key]}")
    line2 = "  |  ".join(parts)

    if not invalid:
        walk = ("arrows = launch-id walk (StreamK: each WG at the lowest tile "
                "it owns)" if meta.get("per_tile")
                else "arrows = launch-id walk")
        line2 += ("\nCell = 1 macro-tile (dashed border): WG<pre-WGM launch id>, "
                  "M<M-tile row> N<N-tile col>, XCC<physical XCD>;  " + walk)

    # --- Per-XCD operand-load reuse stats ---
    # Each output tile lives at (M-tile = wg_m along x, N-tile = wg_n along y) and
    # loads one A row-block (its M-tile) and one B column-block (its N-tile). For
    # each XCD (XCC id), count the tiles it owns and how many DISTINCT M-tiles
    # (rows) and N-tiles (cols) it touches. Share% = redundant (reused) fraction
    # of that operand's block loads = (total - unique) * 100 / total; higher =
    # more L2 reuse of that operand within the XCD.
    if invalid:
        stats_table = ("Per-XCD operand-load reuse: N/A -- StreamK kernel is NOT "
                       "instrumented (no valid WGM dump).")
    else:
        from collections import defaultdict
        rows_by_xcd = defaultdict(list)  # M-tile indices (rows)
        cols_by_xcd = defaultdict(list)  # N-tile indices (cols)
        for (i, j), x in np.ndenumerate(xcc_grid):
            if x is None or (isinstance(x, float) and math.isnan(x)):
                continue
            rows_by_xcd[int(x)].append(j)   # wg_m -> M-tile (row)
            cols_by_xcd[int(x)].append(i)   # wg_n -> N-tile (col)
        stat_lines = ["Per-XCD operand-load reuse (Rows=M-tiles, Cols=N-tiles). "
                      "TotShare% = 100 - (uRows+uCols)/(2*tiles):",
                      "XCD  tiles  uRows  uCols  RowShare%  ColShare%  TotShare%"]
        for x in sorted(rows_by_xcd):
            tot = len(rows_by_xcd[x])
            ur = len(set(rows_by_xcd[x]))
            uc = len(set(cols_by_xcd[x]))
            rs = (tot - ur) * 100.0 / tot if tot else 0.0
            cs = (tot - uc) * 100.0 / tot if tot else 0.0
            ts = (2 * tot - ur - uc) * 100.0 / (2 * tot) if tot else 0.0
            stat_lines.append(f"{x:3d}  {tot:5d}  {ur:5d}  {uc:5d}  {rs:8.1f}  {cs:8.1f}  {ts:8.1f}")
        stats_table = "\n".join(stat_lines)

    # --- Full names at the top of the chart (solution + kernel) ---
    import textwrap
    solname = meta.get("solname")
    name = meta.get("name")
    header_lines = []
    if solname:
        header_lines.append("Solution: " + textwrap.fill(solname, 150, subsequent_indent="          "))
    if name:
        header_lines.append("Kernel:   " + textwrap.fill(name, 150, subsequent_indent="          "))
    header = ("\n".join(header_lines) + "\n\n") if header_lines else ""

    if invalid:
        line1 = ("*** WGM DATA INVALID -- StreamK kernel (uninstrumented): the grid "
                 "size/name/config below are real, but XCD colors, tile mapping, walk "
                 "order and per-XCD stats are NOT captured. ***\n") + line1

    title = header + line1 + "\n" + line2 + "\n\n" + stats_table
    ax.set_title(title, fontsize=8, family="monospace", loc="left",
                 color=("red" if invalid else "black"))

    plt.savefig(out_path, bbox_inches="tight")
    print(f"wrote {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dump", help="binary D dump produced via HIPBLASLT_DEBUG_WGM_DUMP")
    parser.add_argument("--mt0", type=int, default=16, help="MacroTile0 / M tile size (default 16)")
    parser.add_argument("--mt1", type=int, default=16, help="MacroTile1 / N tile size (default 16)")
    parser.add_argument("-o", "--output", default=None,
                        help="output image path (default: <dump>_wgm.jpg)")
    # Optional metadata annotated onto the plot (from the kernel name / solution).
    parser.add_argument("--depthu", type=int, default=None, help="DepthU (unroll K) for MacroTile label")
    parser.add_argument("--streamk", default=None, help="StreamK value")
    parser.add_argument("--gsu", default=None, help="GlobalSplitU (GSU) value")
    parser.add_argument("--lsu", default=None, help="LocalSplitU (LSU) value")
    parser.add_argument("--wgm", default=None, help="WorkGroupMapping (WGM) value")
    parser.add_argument("--wgmxcc", default=None, help="WorkGroupMappingXCC value")
    parser.add_argument("--name", default=None, help="kernel name to show at the top of the plot")
    parser.add_argument("--solname", default=None, help="full solution name to show at the top of the plot")
    parser.add_argument("--invalid", action="store_true",
                        help="mark the dump as invalid (e.g. uninstrumented StreamK): show names/config/grid only")
    args = parser.parse_args()

    raw, m, n, ldd, bpe = read_dump(args.dump)
    print(f"D output: {m}x{n} (ldd={ldd}, {bpe}B/elem)")

    meta = {
        "mt0": args.mt0, "mt1": args.mt1, "depthu": args.depthu,
        "streamk": args.streamk, "gsu": args.gsu, "lsu": args.lsu,
        "wgm": args.wgm, "wgmxcc": args.wgmxcc, "name": args.name,
        "solname": args.solname, "invalid": args.invalid,
    }
    out_path = args.output or (args.dump + "_wgm.jpg")

    sk_n_wg_m = math.ceil(m / args.mt0)
    sk_n_wg_n = math.ceil(n / args.mt1)
    streamk_records = read_streamk_records(raw, sk_n_wg_m, sk_n_wg_n)
    if streamk_records is not None:
        print(f"StreamK per-tile records: {len(streamk_records)}")
        packed = decode_streamk_packed_wgm(streamk_records[0][4])
        meta = dict(meta)
        meta.setdefault("wgm", packed["wgm"])
        meta.setdefault("wgmxcc", packed["wgmxcc"])
        meta["streamk"] = meta.get("streamk") or "per-tile"
        meta["per_tile"] = True
        lattice, xcc_grid, wgm_value, order, (n_wg_m, n_wg_n), covered = (
            streamk_records_to_lattice(streamk_records, sk_n_wg_m, sk_n_wg_n)
        )
        total_tiles = n_wg_m * n_wg_n
        print(f"Workgroup grid: {n_wg_m} x {n_wg_n} (MT {args.mt0}x{args.mt1}); "
              f"tiles covered={covered}/{total_tiles} "
              f"({100.0 * covered / total_tiles:.1f}%)")
        print(f"WGM value: 0x{wgm_value:08X} (WGM={packed['wgm']} WGMXCC={packed['wgmxcc']} "
              f"chunk={packed['chunk']} K={packed['splitk']})")
        to_plot(lattice, xcc_grid, order, wgm_value, out_path, meta=meta)
        return

    lattice, lattice_str, xcc_grid, wgm_value, order, (n_wg_m, n_wg_n) = build_lattice(
        raw, m, n, ldd, bpe, args.mt0, args.mt1
    )
    print(f"Workgroup grid: {n_wg_m} x {n_wg_n} (MT {args.mt0}x{args.mt1})")
    if wgm_value is not None:
        print(f"WGM value: 0x{wgm_value:08X}")

    to_plot(lattice, xcc_grid, order, wgm_value, out_path, meta=meta)


if __name__ == "__main__":
    main()
