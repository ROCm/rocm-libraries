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

"""List benchmark suites."""

import argparse
import functools
import os

import rrperf.args as args
import rrperf.rrsuites
import rrperf.utils as utils


@functools.cache
def hline():
    width = os.get_terminal_size().columns
    return "-" * width


def get_args(parser: argparse.ArgumentParser):
    args.suite(parser)


def run(args):
    """List benchmarks."""

    suite = args.suite
    if suite is None:
        if utils.rocm_gfx().startswith("gfx120"):
            suite = "all_gfx120X"
        else:
            suite = "all"

    generator = getattr(rrperf.rrsuites, suite)
    for x in generator():
        print(hline())
        print("-- rocRoller benchmark run")
        print("--")
        print("Token: ", x)
        print("")
        print(utils.sjoin(x.command()))
        print("")
