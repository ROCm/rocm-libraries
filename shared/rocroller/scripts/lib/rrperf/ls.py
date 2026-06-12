# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""List benchmark suites."""

import argparse
import functools
import shutil

import rrperf.args as args
import rrperf.utils as utils


@functools.cache
def hline():
    width = shutil.get_terminal_size(fallback=(80, 20)).columns
    return "-" * width


def get_args(parser: argparse.ArgumentParser):
    args.suite(parser)


def run(args):
    """List benchmarks."""

    suite = args.suite
    if suite is None:
        if utils.rocm_gfx().startswith("gfx120"):
            suite = "all_gfx120X"
        if utils.rocm_gfx().startswith("gfx1250"):
            suite = "all_gfx1250"
        else:
            suite = "all"

    generator = utils.load_suite(suite)
    for x in generator:
        print(hline())
        print("-- rocRoller benchmark run")
        print("--")
        print("Id: ", x.id)
        print("Token: ", x)
        print("")
        print(utils.sjoin(x.command()))
        print("")
