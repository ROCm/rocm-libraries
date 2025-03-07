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


import argparse
import pathlib
import re
import subprocess
from datetime import datetime


def runAndLog(args):
    fname = pathlib.Path(args.output_file)
    output = subprocess.check_output(args.command, cwd=args.working_directory).decode()
    print(output)
    if args.regex:
        regex = "|".join(["(?:{})".format(r) for r in args.regex])
        output = ", ".join(re.findall(regex, output))
    if not args.exclude_commit:
        commit = subprocess.check_output(
            "git rev-parse --short HEAD".split(), cwd=args.git_directory
        ).decode()
        output = ", ".join(["Git Commit: {}".format(commit.strip()), output])
    output = ", ".join([str(datetime.now()), output])
    print(output)
    mode = "w" if args.overwrite else "a"
    with fname.open(mode) as log:
        log.write(output)
        log.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""Run a program and log its output to a file along
        with current git commit."""
    )
    parser.add_argument(
        "--output_file", type=str, default="run.log", help="Target log file."
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output file instead of appending.",
    )
    parser.add_argument(
        "--exclude_commit", action="store_true", help="Don\t include commit ID in log."
    )
    parser.add_argument(
        "--git_directory",
        type=str,
        default="./",
        help="Directory to query for git commit.",
    )
    parser.add_argument(
        "--working_directory",
        type=str,
        default="./",
        help="Directory to run script in.",
    )
    parser.add_argument(
        "-r",
        "--regex",
        default=[],
        type=str,
        action="append",
        help="Regex for what to capture and log.",
    )
    parser.add_argument(
        "command", nargs=argparse.REMAINDER, help="Command to run and log."
    )
    args = parser.parse_args()

    runAndLog(args)
