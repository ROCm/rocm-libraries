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
Profiling script for rocSOLVER benchmarks using rocprofv3.

This script executes selected benchmark suites with rocprofv3 to collect
detailed profiling information including kernel traces and statistics.
"""

import argparse
import collections
import os
import shlex
import sys
from subprocess import Popen, PIPE

from rocsolver_suites import SUITES, get_size_configurations


#################################################
############## Helper functions #################
#################################################

def setup_vprint(args):
    """
    SETUP_VPRINT defines the function vprint as the normal print function when
    verbose output is enabled, or alternatively as a function that does nothing.
    """
    global vprint
    vprint = print if args.verbose else lambda *a, **k: None


def call_rocsolver_bench(bench_executable, *args):
    """
    CALL_ROCSOLVER_BENCH executes system call to the benchmark
    client executable with the given list of arguments
    """
    cmd = [bench_executable]
    for arg in args:
        if isinstance(arg, str):
            cmd.extend(shlex.split(arg, False, False))
        elif isinstance(arg, collections.Sequence):
            cmd.extend(arg)
        else:
            cmd.push(str(arg))
    process = Popen(cmd, stdout=PIPE, stderr=PIPE)
    vprint('executing {}'.format(' '.join(cmd)))
    stdout, stderr = process.communicate()
    return (str(stdout, encoding='utf-8', errors='surrogateescape'),
            str(stderr, encoding='utf-8', errors='surrogateescape'),
            process.returncode)


def profile_benchmarks(suite, precision, case, bench_executable, output_dir, graph):
    """
    PROFILE_BENCHMARKS runs the benchmark suite with rocprofv3 to collect profiling data
    """
    benchmark_generator = SUITES[suite]
    sizenormal, sizebatch = get_size_configurations(case)

    # Create output directory if it doesn't exist
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for row, n, bench_args in benchmark_generator(suite=suite, precision=precision,
                                                    sizenormal=sizenormal, sizebatch=sizebatch):
        # Construct benchmark identifier string
        benchmark_string = "_".join([str(x) for x in list(row.values())])

        # Determine output path for this benchmark's profiling data
        if output_dir:
            profile_output = os.path.join(output_dir, benchmark_string)
        else:
            profile_output = benchmark_string

        # Run with rocprofv3
        command_args = (f'--kernel-trace --stats=ON -d {profile_output} '
                        f'-o {benchmark_string} -f csv -- {bench_executable} {bench_args}')

        print(f"Profiling: {benchmark_string}")
        out_rocprof, err_rocprof, exitcode_rocprof = call_rocsolver_bench("rocprofv3", command_args)

        if exitcode_rocprof != 0:
            print(f"Warning: rocprofv3 call failed for {benchmark_string}: {err_rocprof}", file=sys.stderr)
            continue

        rocprof_time = float(out_rocprof)
        print(f'  Timing (with profiler overhead): {rocprof_time} us')

        # Generate graph if requested
        if graph:
            try:
                from rocsolver_graph import generate_rocprof_graph
                generate_rocprof_graph(profile_output)
            except Exception as e:
                print(f"Warning: Failed to generate graph for {benchmark_string}: {e}", file=sys.stderr)


#################################################
######### Main functions ########################
#################################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='rocsolver-profile',
        description='Profile rocSOLVER benchmarks using rocprofv3.')
    parser.add_argument('-v','--verbose',
            action='store_true',
            help='display more information about operations being performed')
    parser.add_argument('--exe',
            default='../../build/release/clients/staging/rocsolver-bench',
            help='the benchmark executable to run')
    parser.add_argument('-o', '--output-dir',
            dest='output_dir',
            default=None,
            help='directory to store profiling results (default: current directory)')
    parser.add_argument('--graph',
            action='store_true',
            help='generate graphs of profiling results using matplotlib')
    parser.add_argument('suite',
            choices=SUITES.keys(),
            help='the set of benchmarks to profile')
    parser.add_argument('precision',
            choices=['s', 'd', 'c' , 'z'],
            help='the precision to use for the benchmarks')
    parser.add_argument('case',
            choices=['small', 'medium', 'large'],
            help='the size case to use for the benchmarks')

    args = parser.parse_args()
    setup_vprint(args)

    profile_benchmarks(args.suite, args.precision, args.case, args.exe,
                       args.output_dir, args.graph)
