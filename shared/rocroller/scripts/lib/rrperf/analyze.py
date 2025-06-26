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

"""Result reporting routines."""

import argparse
import numpy as np
import pathlib
import yaml

from typing import List

import rrperf.problems

from rrperf.problems import GEMMResult


def get_args(parser: argparse.ArgumentParser):
    parser.add_argument("directory", type=pathlib.Path)


def run(args):
    analyze(args.directory)


def analyze(directory: pathlib.Path):
    data = read_data(directory)

    infos = sorted(map(info, data), key=lambda x: x[2])
    for (dim, value, time) in infos:
        print(f"{dim}, {value}, {time}")


def info(res: GEMMResult):
    dim, value = res.workgroupMapping
    time = np.median(res.kernelExecute)
    return (dim, value, time)


def read_data(directory: pathlib.Path) -> List[GEMMResult]:
    import itertools
    return itertools.chain(*map(rrperf.problems.load_results, directory.glob("*.yaml")))


def read_file(file: pathlib.Path):
    with file.open("r") as f:
        GEMMResult()
        return yaml.safe_load(f)
