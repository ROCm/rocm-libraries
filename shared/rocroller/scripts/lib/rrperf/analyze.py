# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Result reporting routines."""

import argparse
import pathlib
from typing import List

import numpy as np
import rrperf.problems
import yaml
from rrperf.problems import GEMMResult


def get_args(parser: argparse.ArgumentParser):
    parser.add_argument("directory", type=pathlib.Path)


def run(args):
    analyze(args.directory)


def analyze(directory: pathlib.Path):
    data = read_data(directory)

    infos = sorted(map(info, data), key=lambda x: x[2])
    for dim, value, time in infos:
        print(f"{dim}, {value}, {time}")


def info(res: GEMMResult):
    dim = res.workgroupMappingDim
    value = res.workgroupMappingValue
    time = np.median(res.kernelExecute)
    return (dim, value, time)


def read_data(directory: pathlib.Path) -> List[GEMMResult]:
    import itertools

    return itertools.chain(*map(rrperf.problems.load_results, directory.glob("*.yaml")))


def read_file(file: pathlib.Path):
    with file.open("r") as f:
        GEMMResult()
        return yaml.safe_load(f)
