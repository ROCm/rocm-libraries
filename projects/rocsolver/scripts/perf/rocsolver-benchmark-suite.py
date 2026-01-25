#!/usr/bin/env python3
# ##########################################################################
# Copyright (C) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
# OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
# OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
# SUCH DAMAGE.
# ##########################################################################

"""
Automated benchmark and graphing suite for rocSOLVER.

This script automates the process of running benchmarks and generating
performance graphs by calling rocsolver-perfoptim-suite.py and rocsolver-graph.py.
"""

import argparse
import os
import subprocess
import sys

from rocsolver_suites import SUITES


def run_command(cmd, description):
    """
    Run a command and handle errors.

    Args:
        cmd: Command to run as a list of arguments
        description: Description of what the command does (for error messages)
    """
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")
    print(f"Running: {' '.join(cmd)}\n")

    result = subprocess.run(cmd)
    if result.returncode != 0:
        sys.exit(f"Error: {description} failed with exit code {result.returncode}")


def main():
    parser = argparse.ArgumentParser(
        prog='rocsolver-benchmark-suite',
        description='Run rocSOLVER benchmarks and automatically generate performance graphs.')
    parser.add_argument('-v','--verbose',
            action='store_true',
            help='display more information about operations being performed')
    parser.add_argument('--exe',
            default='../../build/release/clients/staging/rocsolver-bench',
            help='the benchmark executable to run')
    parser.add_argument('-o', '--output',
            dest='output_base',
            default=None,
            help='base name for output files (default: suite_precision_case)')
    parser.add_argument('--no-graph',
            action='store_true',
            help='skip graph generation')
    parser.add_argument('suite',
            choices=SUITES.keys(),
            help='the set of benchmarks to run')
    parser.add_argument('precision',
            choices=['s', 'd', 'c' , 'z'],
            help='the precision to use for the benchmarks')
    parser.add_argument('case',
            choices=['small', 'medium', 'large'],
            help='the size case to use for the benchmarks')

    args = parser.parse_args()

    # Determine output base name
    if args.output_base:
        output_base = args.output_base
    else:
        output_base = f"{args.suite}_{args.precision}_{args.case}"

    csv_path = f"{output_base}.csv"
    graph_path = f"{output_base}.png"

    # Find the directory containing this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    perfoptim_script = os.path.join(script_dir, 'rocsolver-perfoptim-suite.py')
    graph_script = os.path.join(script_dir, 'rocsolver-graph.py')

    # Step 1: Run benchmarks
    benchmark_cmd = [
        sys.executable,  # Use the same Python interpreter
        perfoptim_script,
        '-o', csv_path,
        '--exe', args.exe
    ]
    if args.verbose:
        benchmark_cmd.append('-v')

    benchmark_cmd.extend([args.suite, args.precision, args.case])

    run_command(benchmark_cmd, "Running benchmarks")

    # Step 2: Generate graph (unless disabled)
    if not args.no_graph:
        if not os.path.exists(csv_path):
            sys.exit(f"Error: Benchmark CSV not found: {csv_path}")

        graph_cmd = [
            sys.executable,
            graph_script,
            csv_path,
            '-o', graph_path
        ]

        run_command(graph_cmd, "Generating performance graph")

        print(f"\n{'='*60}")
        print("SUCCESS")
        print(f"{'='*60}")
        print(f"Benchmark results: {csv_path}")
        print(f"Performance graph: {graph_path}")
    else:
        print(f"\n{'='*60}")
        print("SUCCESS")
        print(f"{'='*60}")
        print(f"Benchmark results: {csv_path}")


if __name__ == '__main__':
    main()
