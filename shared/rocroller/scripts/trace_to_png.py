#!/usr/bin/env python3

################################################################################
#
# MIT License
#
# Copyright 2024-2025 AMD ROCm(TM) Software
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
# ies of the Software, and to permit persons to whom the Software is furnished
# to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
# PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
# CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
################################################################################

"""Utility to render the output of 'trace_memory'."""

import argparse
import pathlib

import numpy as np
import pandas as pd

from PIL import Image


def crop_to_bbox(img):
    """Crop `img` to smallest bounding box."""

    flat = np.any(img, axis=2)
    rows = np.any(flat, axis=1)
    cols = np.any(flat, axis=0)
    rmin, rmax = np.nonzero(rows)[0][[0, -1]]
    cmin, cmax = np.nonzero(cols)[0][[0, -1]]

    return img[rmin : rmax + 1, cmin : cmax + 1, :]


def trace_to_png(
    image_file, trace, m, n, element_size=4, instruction_width=16, crop=True, theme="wg"
):
    """Visualize memory access pattern from output of 'trace_memory'.

    The matrix size is M x N elements; and addressing is column-major.

    Colouring is based on the `theme`:
    - "wg", colour based on workgroup-x (red), workgroup-y (blue), workitem (green)
    - "wf", colour based on lane (red) and wavefront (blue)
    """

    wgx_max = max(1, trace["WorkgroupX"].max())
    wgy_max = max(1, trace["WorkgroupY"].max())
    wi_max = max(1, trace["Workitem"].max())

    # we assume traced matrix is column-major (row-index is fastest
    # moving), and therefore we align bytes with the row-index.
    img = np.zeros([m * element_size, n, 3], dtype=np.uint8, order="F")

    for _, row in trace.iterrows():
        address, wgx, wgy, wi = (
            row["Address"],
            row["WorkgroupX"],
            row["WorkgroupY"],
            row["Workitem"],
        )

        i, j = address % (m * element_size), address // (m * element_size)

        if theme == "wg":
            img[i : (i + instruction_width), j, 0] = int((wgx / wgx_max) * 255)
            img[i : (i + instruction_width), j, 1] = int((wi / wi_max) * 255)
            img[i : (i + instruction_width), j, 2] = int((wgy / wgy_max) * 255)
        elif theme == "wf":
            img[i : (i + instruction_width), j, 0] = int((wi // 64) * 64 + 63)
            img[i : (i + instruction_width), j, 1] = 0
            img[i : (i + instruction_width), j, 2] = int((wi % 64) * 4 + 3)
        else:
            raise ValueError(f"Invalid theme '{theme}'.  Choices are 'wg' and 'wf'.")

    if crop:
        img = crop_to_bbox(img)

    with Image.fromarray(img) as png:
        png.save(image_file)
        print(f"Wrote {image_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert CSV trace from trace_memory to a PNG."
    )
    parser.add_argument("filename")
    parser.add_argument(
        "-m",
        "--m",
        dest="M",
        type=int,
        required=True,
        help="Matrix M size (in elements)",
    )
    parser.add_argument(
        "-n",
        "--n",
        dest="N",
        type=int,
        required=True,
        help="Matrix N size (in elements)",
    )
    parser.add_argument(
        "-s",
        "--element_size",
        type=int,
        default=2,
        help="Number of bytes per matrix element",
    )
    parser.add_argument(
        "-w",
        "--instruction_width",
        type=int,
        default=16,
        help="Number of bytes per load/store instruction",
    )
    parser.add_argument(
        "-c",
        "--crop",
        default=False,
        action="store_true",
        help="Crop image to smalled bounding-box.",
    )
    parser.add_argument(
        "-t", "--theme", type=str, default="wg", help="Colour theme: 'wg' or 'wf'."
    )

    args = parser.parse_args()

    trace_fname = pathlib.Path(args.filename)
    if not trace_fname.exists():
        print(f"Trace file {trace_fname} not found.")
        return

    png_fname = trace_fname.with_suffix(".png")

    trace = pd.read_csv(str(trace_fname))
    trace_to_png(
        str(png_fname),
        trace,
        args.M,
        args.N,
        element_size=args.element_size,
        instruction_width=args.instruction_width,
        crop=args.crop,
        theme=args.theme,
    )


if __name__ == "__main__":
    main()
