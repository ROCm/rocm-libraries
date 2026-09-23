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
    """Read the binary D dump and return a (N, ldd) uint32 array (raw bits)."""
    with open(path, "rb") as f:
        header = f.read(20)
        if len(header) < 20:
            sys.exit(f"{path}: file too small to contain a WGM dump header")
        magic, m, n, ldd, bpe = struct.unpack("<5i", header)
        if magic != MAGIC:
            sys.exit(
                f"{path}: bad magic 0x{magic:08X} (expected 0x{MAGIC:08X}); "
                "was this produced by a --debug-wgm build with HIPBLASLT_DEBUG_WGM_DUMP set?"
            )
        if bpe != 4:
            sys.exit(
                f"{path}: bytesPerElement={bpe}, but WGM visualization only supports "
                "4-byte (f32_r) output. Rerun the GEMM with --precision f32_r."
            )
        raw = f.read(ldd * n * bpe)

    # D is column-major MxN with leading dimension ldd. Interpreting the raw
    # bytes as uint32 and reshaping to (N, ldd) gives grid[n][m] == element(m, n)
    # (the same orientation the original WriteTensor-based tool used: one text
    # "row" per N index, columns walking M).
    grid = np.frombuffer(raw, dtype=np.uint32)
    if grid.size < ldd * n:
        sys.exit(f"{path}: truncated payload ({grid.size} < {ldd * n} words)")
    grid = grid[: ldd * n].reshape(n, ldd)
    return grid, m, n, ldd


def build_lattice(grid, m, n, mt0, mt1):
    """Decode the per-workgroup WGM diagnostics from the tile top-left elements.

    Returns lattice[wgN][wgM] = (orig_wg, (new_wg1, new_wg0), xcc), the XCC grid,
    the packed WGM value (should be identical across all tiles), and the walk
    order indexed by original WG id.
    """
    n_wg_m = math.ceil(m / mt0)
    n_wg_n = math.ceil(n / mt1)

    lattice = np.empty((n_wg_n, n_wg_m), dtype=object)
    lattice_str = np.empty((n_wg_n, n_wg_m), dtype=object)
    xcc_grid = np.zeros((n_wg_n, n_wg_m), dtype=int)
    wgm_value = None

    # grid[n][m]: n over columns (N, step mt1), m over rows (M, step mt0).
    for wg_n, col in enumerate(range(0, n, mt1)):
        for wg_m, row in enumerate(range(0, m, mt0)):
            orig_wg = int(grid[col][row + 0])

            packed_new = int(grid[col][row + 1])
            new_wg0 = (packed_new >> 16) & 0xFFFF
            new_wg1 = packed_new & 0xFFFF

            xcc = int(grid[col][row + 2])

            wgm = int(grid[col][row + 3])
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
            order[orig_wg] = (new_wg1, new_wg0)

    return lattice, lattice_str, xcc_grid, wgm_value, order, (n_wg_m, n_wg_n)


def to_plot(lattice, xcc_grid, order, wgm_value, out_path):
    from matplotlib import patches
    from matplotlib import pyplot as plt

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

    title = "WGM workgroup mapping"
    if wgm_value is not None:
        title += f"  (WGM=0x{wgm_value:08X})"
    ax.set_title(title)

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
    args = parser.parse_args()

    grid, m, n, ldd = read_dump(args.dump)
    print(f"D output: {m}x{n} (ldd={ldd})")

    lattice, lattice_str, xcc_grid, wgm_value, order, (n_wg_m, n_wg_n) = build_lattice(
        grid, m, n, args.mt0, args.mt1
    )
    print(f"Workgroup grid: {n_wg_m} x {n_wg_n} (MT {args.mt0}x{args.mt1})")
    if wgm_value is not None:
        print(f"WGM value: 0x{wgm_value:08X}")

    out_path = args.output or (args.dump + "_wgm.jpg")
    to_plot(lattice, xcc_grid, order, wgm_value, out_path)


if __name__ == "__main__":
    main()
