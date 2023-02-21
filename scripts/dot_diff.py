#!/usr/bin/env python3

import argparse
import difflib
import pathlib
import subprocess
from typing import List


def color_node(line: str, is_addition: bool):
    if is_addition:
        color = "blue"
    else:
        color = "red"
        if "blue" in line:
            color = "yellow"
    if "]" in line and "invis" not in line:
        idx = line.index("]")
        return line[:idx] + f",color={color}" + line[idx:]
    return line


def diff_dots(dots: List[str]) -> List[str]:
    dots_diff = []
    for i, dot in enumerate(dots):
        if i + 1 == len(dots):
            continue
        dot_next = dots[i + 1]
        if i == 0:
            dots_diff.append(dot.split("\n"))
        dots_diff.append(dot_next.split("\n"))

        index = 0
        index_next = 0
        for diff in difflib.unified_diff(
            dot.split("\n"),
            dot_next.split("\n"),
            n=0,
            lineterm="",
        ):
            if "@@" in diff:
                index = int(diff.split(" ")[1].split(",")[0].strip("-")) - 1
                index_next = int(diff.split(" ")[2].split(",")[0]) - 1
            else:
                if diff[0] == "+":
                    dots_diff[i + 1][index_next] = color_node(
                        dots_diff[i + 1][index_next],
                        True,
                    )
                    index_next += 1
                elif diff[0] == "-":
                    dots_diff[i][index] = color_node(
                        dots_diff[i][index],
                        False,
                    )
                    index += 1

    return ["\n".join(d) for d in dots_diff]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract and render kernel graph.")
    parser.add_argument(
        "fnames",
        type=str,
        help="Dot files to compare",
        nargs="+",
    )
    parser.add_argument(
        "-o", "--output", type=str, help="Output file base name", default="dot_output"
    )
    parser.add_argument("-d", "--dot-only", default=False, action="store_true")

    args = parser.parse_args()
    dots_in = []
    for fname in args.fnames:
        dots_in.append(pathlib.Path(fname).read_text())

    dots = diff_dots(dots_in)
    for i, dot in enumerate(dots):
        out_fname = pathlib.Path(args.output + f"_{i:04d}")
        out_fname.with_suffix(".dot").write_text(dot)
        if not args.dot_only:
            with out_fname.with_suffix(".pdf").open("w") as out:
                subprocess.run(["dot", "-Tpdf"], input=dot.encode(), stdout=out)
