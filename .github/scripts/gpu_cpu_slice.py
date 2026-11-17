#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Run a command on one GPU's share of the CPUs this process may use.

A multi-GPU host runs one GitHub runner per GPU, and the job containers have no CPU
limit, so every job sees all host CPUs. Tools that size their worker pools from the
CPUs they can run on (Python's os.process_cpu_count, joblib/loky, TensileLite's
process pools) then start one process per host CPU in every job at once. This script
gives the command the GPU's slice of the affinity mask instead: with N GPUs per host,
GPU g gets the g-th of N equal, contiguous slices. It also exports LOKY_MAX_CPU_COUNT
so loky-based pools that read it see the same count.

Usage (the CI step does the same):
    GPUS_PER_HOST=4 ROCR_VISIBLE_DEVICES=1 python3 .github/scripts/gpu_cpu_slice.py -- CMD [ARGS...]

GPU index: --gpu, else the first index in ROCR_VISIBLE_DEVICES, else 0.
GPUs per host: --gpus-per-host, else GPUS_PER_HOST, else 1 (the whole mask).
"""

import argparse
import os
import sys


def cpu_slice(cpus, gpu, gpus_per_host):
    """The CPUs of GPU ``gpu``: its 1/``gpus_per_host`` of ``cpus`` (sorted), at least one CPU."""
    cpus = sorted(cpus)
    gpus_per_host = max(1, gpus_per_host)
    per = max(1, len(cpus) // gpus_per_host)
    start = (gpu % gpus_per_host) * per
    return cpus[start : start + per] or cpus[:1]


def cpu_ranges(cpus):
    """Compact text for a CPU list, e.g. [0, 1, 2, 5] -> '0-2,5'."""
    ranges, start = [], None
    for i, cpu in enumerate(cpus):
        if start is None:
            start = cpu
        if i + 1 == len(cpus) or cpus[i + 1] != cpu + 1:
            ranges.append(str(start) if start == cpu else "%d-%d" % (start, cpu))
            start = None
    return ",".join(ranges)


def parse_args(argv, environ):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--gpu", type=int, help="GPU index (default: ROCR_VISIBLE_DEVICES)"
    )
    parser.add_argument(
        "--gpus-per-host", type=int, help="runners/GPUs sharing the host CPUs"
    )
    parser.add_argument("command", nargs=argparse.REMAINDER, help="-- CMD [ARGS...]")
    args = parser.parse_args(argv)
    if args.command[:1] == ["--"]:
        args.command = args.command[1:]
    if not args.command:
        parser.error("no command given")
    if args.gpu is None:
        first = environ.get("ROCR_VISIBLE_DEVICES", "").split(",")[0].strip()
        args.gpu = int(first) if first.isdigit() else 0
    if args.gpus_per_host is None:
        value = environ.get("GPUS_PER_HOST", "1")
        args.gpus_per_host = int(value) if value.isdigit() else 1
    return args


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv, os.environ)
    allowed = sorted(os.sched_getaffinity(0))
    mine = cpu_slice(allowed, args.gpu, args.gpus_per_host)
    print(
        "CPU slice for GPU %d (%d per host): %d of %d CPUs: %s"
        % (args.gpu, args.gpus_per_host, len(mine), len(allowed), cpu_ranges(mine)),
        flush=True,
    )
    os.sched_setaffinity(0, mine)
    os.environ["LOKY_MAX_CPU_COUNT"] = str(len(mine))
    os.execvp(args.command[0], args.command)


if __name__ == "__main__":
    main()
