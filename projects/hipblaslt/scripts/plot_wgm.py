#!/usr/bin/env python3
# ##############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
# ##############################################################################
"""Visualize hipBLASLt workgroup-mapping (WGM) instrumentation output.

This is the companion tool for the debug-only ``--debug-wgm`` build option.

Workflow
--------
1. Build hipBLASLt in Debug with WGM instrumentation::

       ./install.sh --debug --debug-wgm -c        # or: invoke build --debug --debug-wgm --clients

   With this option the generated GEMM kernels overwrite the top-left element
   of every workgroup's 16x16(-ish) output tile with workgroup-mapping
   diagnostics *instead of the real result*:

       tile[0..3] = [ original 1D WG id,
                      (post-WGM WorkGroup0 << 16) | post-WGM WorkGroup1,
                      XCC id,
                      packed WGM sgpr value ]

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
            # Store as (x=new_wg0, y=new_wg1) to match the cell-text placement
            # ax.text(new_wg0, new_wg1, ...); otherwise arrows are transposed and
            # only cover part of the grid.
            order[orig_wg] = (new_wg0, new_wg1)

    return lattice, lattice_str, xcc_grid, wgm_value, order, (n_wg_m, n_wg_n)


def to_plot(lattice, xcc_grid, order, wgm_value, out_path, meta=None):
    from matplotlib import patches
    from matplotlib import pyplot as plt

    meta = meta or {}

    def to_text(orig_wg, new_wg0, new_wg1, xcc):
        return f"{orig_wg}->({new_wg1},{new_wg0})\n(XCC:{xcc})"

    n_wg_n, n_wg_m = lattice.shape

    fig, ax = plt.subplots(figsize=(max(8, n_wg_m * 2), max(8, n_wg_n * 2)), dpi=80)
    plt.tight_layout()
    plt.xticks([], [])
    plt.yticks([], [])

    ax.imshow(xcc_grid)

    for (i, j), z in np.ndenumerate(lattice):
        orig_wg, (new_wg1, new_wg0), xcc = z
        ax.text(new_wg0, new_wg1, to_text(orig_wg, new_wg0, new_wg1, xcc),
                ha="center", va="center", size=12)

    # Draw the space-filling walk order (arrows between consecutive WG ids).
    style = "Simple, tail_width=0.5, head_width=8, head_length=8"
    for i in range(1, len(order)):
        if order[i] is None or order[i - 1] is None:
            continue
        p = patches.FancyArrowPatch(
            (order[i - 1][0], order[i - 1][1]),
            (order[i][0], order[i][1]),
            connectionstyle="arc3,rad=.1",
            arrowstyle=style,
            color="k",
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

    ax.set_title(line1 + "\n" + line2, fontsize=14)

    # Full kernel/solution name as a small caption under the figure.
    name = meta.get("name")
    if name:
        fig.text(0.5, 0.005, name, ha="center", va="bottom", fontsize=6,
                 family="monospace", wrap=True)

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
    parser.add_argument("--name", default=None, help="kernel/solution name to caption on the plot")
    args = parser.parse_args()

    raw, m, n, ldd, bpe = read_dump(args.dump)
    print(f"D output: {m}x{n} (ldd={ldd}, {bpe}B/elem)")

    lattice, lattice_str, xcc_grid, wgm_value, order, (n_wg_m, n_wg_n) = build_lattice(
        raw, m, n, ldd, bpe, args.mt0, args.mt1
    )
    print(f"Workgroup grid: {n_wg_m} x {n_wg_n} (MT {args.mt0}x{args.mt1})")
    if wgm_value is not None:
        print(f"WGM value: 0x{wgm_value:08X}")

    meta = {
        "mt0": args.mt0, "mt1": args.mt1, "depthu": args.depthu,
        "streamk": args.streamk, "gsu": args.gsu, "lsu": args.lsu,
        "wgm": args.wgm, "wgmxcc": args.wgmxcc, "name": args.name,
    }
    out_path = args.output or (args.dump + "_wgm.jpg")
    to_plot(lattice, xcc_grid, order, wgm_value, out_path, meta=meta)


if __name__ == "__main__":
    main()
