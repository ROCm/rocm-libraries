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
import os
import sys
import yaml

DESCRIPTION = """
Format a YAML file.
"""


def format_yaml(input_path, output_path, force=False):
    if not os.path.exists(input_path):
        print(f"Error: The path '{input_path}' does not exist.", file=sys.stderr)
        sys.exit(1)
    if os.path.isdir(input_path):
        print(
            f"Error: The path '{input_path}' is a directory, not a file.",
            file=sys.stderr,
        )
        sys.exit(1)

    if output_path and os.path.exists(output_path) and not force:
        confirmation = input(
            f"Warning: The file '{output_path}' already exists. Overwrite? (y/n) "
        )
        if confirmation.lower() != "y":
            print("Cancelled. No file written. Exiting.")
            sys.exit(0)

    with open(input_path, "r", encoding="utf-8") as file:
        try:
            data = yaml.safe_load(file)
        except yaml.YAMLError as e:
            print(
                f"Error: Failed to parse YAML from '{input_path}'.\n{e}",
                file=sys.stderr,
            )
            sys.exit(1)

    formatted_yaml = yaml.dump(
        data, sort_keys=True, default_flow_style=False, allow_unicode=True
    )

    if output_path:
        with open(output_path, "w", encoding="utf-8") as file:
            file.write(formatted_yaml)
        print(f"YAML from '{input_path}' formatted and written to '{output_path}'")
    else:
        print(formatted_yaml)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument("input", type=str, help="Path to the input YAML file")
    parser.add_argument(
        "output",
        type=str,
        nargs="?",
        help="Optional output file (default: print to stdout)",
    )
    parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Overwrite output file without confirmation",
    )
    args = parser.parse_args()

    format_yaml(args.input, args.output, args.force)
