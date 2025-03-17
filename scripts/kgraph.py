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

"""Load kernel graph from the meta data of an assembly file and render it.

Usage:

  kgraph.py path/to/assembly.s

this creates a PDF file at path/to/assembly.pdf.  To automatically
open it (via xdg-open), pass '-x'.

"""

import argparse
import pathlib
import re
import subprocess

import yaml
from dot_diff import diff_dots


def extract_normalised_asm(path: pathlib.Path, strip_metadata: bool = True):
    """Extract normalized ASM from assembly file."""
    source = path.read_text()

    meta = False

    asm = []
    for line in source.splitlines():
        double_slash = line.find("//")
        if double_slash != -1:
            line = line[:double_slash]
        line = line.rstrip()
        if "kernel_graph_dot" in line:
            continue
        if ".amdgpu_metadata" in line:
            meta = True
        if line and not meta:
            asm.append(line)
        if ".end_amdgpu_metadata" in line:
            meta = False

    return "\n".join(asm)


def extract_asm_dot(path: pathlib.Path):
    """Extract .kernel_graph meta data from assembly file."""
    source = path.read_text()
    beginMatch = re.search(r"^---$", source, re.MULTILINE)
    endMatch = re.search(r"^\.\.\.$", source, re.MULTILINE)

    assert beginMatch is not None
    assert endMatch is not None

    beginPos = beginMatch.span()[1]
    endPos = endMatch.span()[0]
    assert beginPos < endPos

    meta = yaml.load(source[beginPos:endPos], Loader=yaml.CSafeLoader)
    kernel = meta["amdhsa.kernels"][0]
    if ".kernel_graph_dot" in kernel:
        return kernel[".name"], kernel[".kernel_graph_dot"], kernel[".kernel_graph"]
    return kernel[".name"], kernel[".kernel_graph"]


def extract_log_dots(path: pathlib.Path):
    source = path.read_text()
    dots = []
    for graph in re.finditer("digraph {", source):
        dots.append(source[graph.start() :].partition("\n\n")[0])
    return dots


def write_dot(dot: str, fname: pathlib.Path):
    out_fname = fname.with_suffix(".dot")
    out_fname.write_text(dot)
    print(f"Wrote {out_fname}")
    return out_fname


def render_dot(fname: pathlib.Path, dot: str):
    """Render graph."""
    with fname.open("w") as out:
        subprocess.run(["dot", "-Tpdf"], input=dot.encode(), stdout=out)
    print(f"Wrote {fname}")


def open_dot(fname):
    subprocess.call(["xdg-open", str(fname)])


def open_code(fname):
    subprocess.call(["code", str(fname)])


def process_dot(
    dot: str,
    out_path: pathlib.Path,
    code_open: bool,
    dot_only: bool,
    xdg_open: bool,
):
    dotfile = write_dot(dot, out_path)
    if code_open:
        open_code(dotfile)
    if not dot_only:
        rendered_fname = out_path.with_suffix(".pdf")
        render_dot(rendered_fname, dot)
        if xdg_open:
            open_dot(rendered_fname)


def process_serialized_graph(graph: dict, out_path: pathlib.Path):
    with out_path.with_suffix(".yaml").open("w") as f:
        yaml.dump(graph, f, default_flow_style=None, Dumper=yaml.CSafeDumper)
        print(f"Wrote {f.name}")


def process_asm(path: pathlib.Path):
    asm = pathlib.Path(path.stem + "_normalized.s")
    asm.write_text(extract_normalised_asm(path))
    print(f"Wrote {asm}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract and render kernel graph.")
    parser.add_argument("fname", type=str, help="Assembly or log file name")
    parser.add_argument("-d", "--dot-only", default=False, action="store_true")
    parser.add_argument(
        "-x",
        "--xdg-open",
        default=False,
        action="store_true",
        help="Open with xdg-open after generating",
    )
    parser.add_argument(
        "-c",
        "--code-open",
        default=False,
        action="store_true",
        help="Open .dot with VSCode after generating",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Output file base name",
    )
    parser.add_argument(
        "--omit_diff",
        default=False,
        action="store_true",
        help="Omit dot diff coloring",
    )

    args = parser.parse_args()
    path = pathlib.Path(args.fname)

    foutput = path
    if args.output is not None:
        foutput = pathlib.Path(args.output)

    if path.suffix == ".s":
        _, dot, graph = extract_asm_dot(path)
        process_dot(
            dot,
            foutput,
            args.code_open,
            args.dot_only,
            args.xdg_open,
        )
        if graph is not None and not args.dot_only:
            process_serialized_graph(graph, foutput)

        process_asm(path)

    elif path.suffix == ".log":
        dots = extract_log_dots(path)
        if not args.omit_diff:
            dots = diff_dots(dots)
        for i, dot in enumerate(dots):
            serial_str = f"_{i:04d}"
            foutput_serial = pathlib.Path(str(foutput) + serial_str)
            process_dot(
                dot,
                foutput_serial,
                args.code_open,
                args.dot_only,
                args.xdg_open,
            )
    else:
        print("Unknown file extension")
