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

"""
rocRoller performance tracking suite command line interface.

New, modular way to add subcommands to rrperf:

Make a top-level module inside rrperf. It must implement get_args() and run().

- get_args() should take an argparse.ArgumentParser and add all command-line arguments
  to it.
- run() should take the parsed arguments object and run the relevant command.

The docstring for the module will be used as the description for the subcommand (under
`rrperf <command> --help`).  The docstring for the `run` function will be used as the
 help text for the subcommand (under `rrperf --help`).

"""

import argparse
import inspect
import rrperf


def main():
    parser = argparse.ArgumentParser(
        description="rocRoller performance tracking suite."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    for name, mod in rrperf.__dict__.items():
        if hasattr(mod, "get_args") and hasattr(mod, "run"):
            mod_parser = subparsers.add_parser(
                name,
                help=inspect.getdoc(mod.run),
                description=inspect.getdoc(mod),
                formatter_class=argparse.RawDescriptionHelpFormatter,
            )
            mod.get_args(mod_parser)

    args = parser.parse_args()

    if (
        "group_results" in args.__dict__
        and args.group_results
        and not args.exclude_boxplot
    ):
        parser.error("--group_results cannot be used without --exclude_boxplot")

    getattr(rrperf, args.command).run(args)
