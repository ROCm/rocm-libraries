#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors

ENTRY_PATTERN = re.compile(r"^[ABCD]\[(\d+)\]\[(\d+)\](?:\.B(\d+))?$", re.IGNORECASE)
MATRIX_LABEL_PATTERN = re.compile(r"_([ABCD])_matrix_layout$", re.IGNORECASE)


def determine_matrix_type(path: Path) -> str:
    stem = path.stem
    match = MATRIX_LABEL_PATTERN.search(stem)
    if not match:
        raise ValueError(
            f"Unable to determine matrix type from filename '{path.name}'. "
            "Expected pattern '*_<A|B|C|D>_matrix_layout'."
        )
    return match.group(1).upper()


def determine_axis_labels(matrix_type: str) -> Tuple[str, str]:
    matrix_type = matrix_type.upper()
    if matrix_type == "A":
        return "k", "m"
    if matrix_type == "B":
        return "n", "k"
    if matrix_type in ("C", "D"):
        return "n", "m"
    raise ValueError(f"Unsupported matrix type '{matrix_type}' for axis labels.")


def parse_layout_csv(path: Path) -> Tuple[np.ndarray, int]:
    with path.open(newline="") as f:
        reader = csv.reader(f)
        try:
            next(reader)  # Architecture line
            next(reader)  # Instruction line
            header = next(reader)
        except StopIteration:
            raise ValueError(f"{path} does not contain required header rows.")

        if not header or header[0].strip().lower() != "lane":
            raise ValueError(f"{path} missing 'lane' header row.")

        entries: Dict[Tuple[int, int, int], int] = {}
        max_row = max_col = max_block = max_lane = -1

        for row in reader:
            if not row:
                continue
            lane_str, *items = row
            lane = int(lane_str.strip())
            max_lane = max(max_lane, lane)
            for item in items:
                item = item.strip()
                if not item:
                    continue
                match = ENTRY_PATTERN.match(item)
                if not match:
                    raise ValueError(f"Unrecognized entry '{item}' in {path}.")
                row_idx = int(match.group(1))
                col_idx = int(match.group(2))
                block_idx = int(match.group(3)) if match.group(3) else 0

                key = (block_idx, row_idx, col_idx)
                if key in entries:
                    raise ValueError(
                        f"Duplicate mapping for block={block_idx}, row={row_idx}, col={col_idx}."
                    )
                entries[key] = lane
                max_row = max(max_row, row_idx)
                max_col = max(max_col, col_idx)
                max_block = max(max_block, block_idx)

    if not entries:
        raise ValueError(f"{path} contains no layout entries.")

    blocks = max_block + 1
    rows = max_row + 1
    cols = max_col + 1
    data = np.full((blocks, rows, cols), -1, dtype=int)

    for (block, row_idx, col_idx), lane in entries.items():
        data[block, row_idx, col_idx] = lane

    if np.any(data < 0):
        raise ValueError("Layout is incomplete; some matrix entries were not assigned.")

    return data, max_lane


def plot_layout(
    data: np.ndarray,
    max_lane: int,
    output_base: Path,
    title: str,
    show: bool,
    matrix_type: str,
) -> None:
    blocks, rows, cols = data.shape
    stack_vertical = cols >= rows

    if stack_vertical:
        min_block_height = max(rows * 0.28, 3.6)
        fig_height = min_block_height * max(blocks, 1)
        per_block_height = min_block_height
        axis_width = max(
            per_block_height * (cols / max(rows, 1)),
            cols * 0.35,
        )
        colorbar_width = 0.8
        fig_width = max(axis_width + colorbar_width + 2.0, 9.0)

        fig = plt.figure(figsize=(fig_width, fig_height))
        gs = fig.add_gridspec(blocks, 2, width_ratios=[22, 1], wspace=0.24, hspace=0.35)
        axis_positions = [gs[block, 0] for block in range(blocks)]
        cax_spec = gs[:, 1]
    else:
        min_block_width = max(cols * 0.28, 3.6)
        fig_width = min_block_width * max(blocks, 1)
        per_block_width = min_block_width
        axis_height = max(
            per_block_width * (rows / max(cols, 1)),
            rows * 0.35,
        )
        fig_height = max(axis_height + 1.5, 5.0)

        fig = plt.figure(figsize=(fig_width, fig_height))
        width_ratios = [20] * blocks + [1]
        gs = fig.add_gridspec(1, blocks + 1, width_ratios=width_ratios, wspace=0.24)
        axis_positions = [gs[0, block] for block in range(blocks)]
        cax_spec = gs[0, -1]

    cmap = plt.cm.viridis
    norm = colors.Normalize(vmin=0, vmax=max_lane)
    axes = []
    im = None

    x_label, y_label = determine_axis_labels(matrix_type)

    for block in range(blocks):
        ax = fig.add_subplot(axis_positions[block])
        axes.append(ax)
        im = ax.imshow(
            data[block],
            cmap=cmap,
            norm=norm,
            origin="upper",
            interpolation="nearest",
            aspect="equal",
        )
        ax.set_title(f"Block {block}")
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label, rotation=0, labelpad=15)

        x_positions = np.arange(cols)
        ax.set_xticks(x_positions)
        ax.set_yticks(np.arange(rows))
        if cols > 16:
            ax.set_xticklabels([])
        else:
            if cols > 12:
                labels = [str(idx) if idx % 2 == 0 else "" for idx in x_positions]
                ax.tick_params(axis="x", labelrotation=45)
            else:
                labels = [str(idx) for idx in x_positions]
            ax.set_xticklabels(labels)
        if rows > 16:
            ax.set_yticklabels([])

        ax.set_xticks(np.arange(-0.5, cols, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, rows, 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=0.8)
        ax.tick_params(which="both", length=0)

    if im is None:
        raise RuntimeError("No image was generated for plotting.")

    cax = fig.add_subplot(cax_spec)
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("Lane")

    fig.suptitle(title, fontsize=14)
    png_path = output_base.with_suffix(".png")
    fig.savefig(png_path, dpi=300)
    print(f"Saved plot to {png_path}")
    if show:
        plt.show()
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Plot an MFMA matrix layout CSV as a colored grid."
    )
    parser.add_argument("csv", type=Path, help="Path to layout CSV file.")
    parser.add_argument(
        "--output",
        type=Path,
        help="Output image path (defaults to CSV path with .png).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the plot interactively in addition to saving.",
    )
    args = parser.parse_args()

    try:
        matrix_type = determine_matrix_type(args.csv)
        data, max_lane = parse_layout_csv(args.csv)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if args.output:
        output_base = args.output.with_suffix("")
    else:
        output_base = args.csv.with_name(f"{args.csv.stem}_matrix_thread_layouts")

    title = args.csv.stem
    plot_layout(data, max_lane, output_base, title, args.show, matrix_type)
    return 0


if __name__ == "__main__":
    sys.exit(main())
