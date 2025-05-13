#!/usr/bin/env python3

################################################################################
#
# MIT License
#
# Copyright 2025 AMD ROCm(TM) Software
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


import argparse
import json


def filter_sarif(input_file, output_file, exclude_paths):
    with open(input_file, "r") as f:
        sarif = json.load(f)

    filtered_results = []
    for run in sarif["runs"]:
        for result in run["results"]:
            analyzed_file_path = result["locations"][0]["physicalLocation"][
                "artifactLocation"
            ]["uri"]
            if not any(analyzed_file_path.startswith(path) for path in exclude_paths):
                filtered_results.append(result)
            else:
                print(f"  excluding '{analyzed_file_path}' file.")
        run["results"] = filtered_results

    with open(output_file, "w") as f:
        json.dump(sarif, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter sarif file")

    parser.add_argument("input", type=str, help="Input sarif file")
    parser.add_argument(
        "exclude_paths_file",
        type=str,
        help="File with paths to exclude. One path per line",
    )
    parser.add_argument(
        "-o", "--output", type=str, help="Output sarif file (default: <input>.filtered)"
    )

    args = parser.parse_args()

    if not args.output:
        args.output = args.input + ".filtered"

    print(f"Filtering sarif file '{args.input}' and saving it to '{args.output}' file.")

    with open(args.exclude_paths_file, "r") as f:
        exclude_paths = [line.strip() for line in f.readlines()]

    filter_sarif(args.input, args.output, exclude_paths)
