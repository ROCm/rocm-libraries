# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Result reporting routines."""

import argparse
from pathlib import Path

import numpy as np
import rrperf.problems
import yaml
from rrperf.problems import GEMMResult


def get_args(parser: argparse.ArgumentParser):
    parser.add_argument("directory", type=Path)


def run(args):
    analyze(args.directory)


def analyze(directory: Path):
    data = read_data(directory)

    infos = sorted(map(info, data), key=lambda x: x[2])
    for dim, value, time in infos:
        print(f"{dim}, {value}, {time}")


def info(res: GEMMResult):
    dim = res.workgroupMappingDim
    value = res.workgroupMappingValue
    time = np.median(res.kernelExecute)
    return (dim, value, time)


def read_data(directory: Path) -> list[GEMMResult]:
    import itertools

    return itertools.chain(*map(rrperf.problems.load_results, directory.glob("*.yaml")))


def read_file(file: Path):
    with file.open("r") as f:
        GEMMResult()
        return yaml.safe_load(f)
