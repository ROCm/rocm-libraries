#!/usr/bin/env python3

# Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import argparse
from collections import namedtuple
import os
import re
import stat
import subprocess
import sys

BenchmarkContext = namedtuple('BenchmarkContext', ['gpu_architecture', 'benchmark_output_dir', 'benchmark_dir', 'benchmark_filename_regex', 'benchmark_filter_regex'])

def run_benchmarks(benchmark_context):
    def is_benchmark_executable(filename):
        if not re.match(benchmark_context.benchmark_filename_regex, filename):
            return False
        path = os.path.join(benchmark_context.benchmark_dir, filename)
        st_mode = os.stat(path).st_mode

        # we are not interested in permissions, just whether there is any execution flag set
        # and it is a regular file (S_IFREG)
        return (st_mode & (stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)) and (st_mode & stat.S_IFREG)

    success = True
    benchmark_names = [name for name in os.listdir(benchmark_context.benchmark_dir) if is_benchmark_executable(name)]
    print('The following benchmarks will be ran:\n{}'.format('\n'.join(benchmark_names)), file=sys.stderr, flush=True)
    for benchmark_name in benchmark_names:
        results_json_name = f'{benchmark_name}_{benchmark_context.gpu_architecture}.json'

        benchmark_path = os.path.join(benchmark_context.benchmark_dir, benchmark_name)
        results_json_path = os.path.join(benchmark_context.benchmark_output_dir, results_json_name)
        args = [
            benchmark_path,
            '--name_format',
            'json',
            '--benchmark_out_format=json',
            f'--benchmark_out={results_json_path}',
            f'--benchmark_filter={benchmark_context.benchmark_filter_regex}'
        ]
        try:
            subprocess.check_call(args)
        except subprocess.CalledProcessError as error:
            print(f'Could not run benchmark at {benchmark_path}. Error: "{error}"', file=sys.stderr, flush=True)
            success = False
    return success



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmark_dir',
        help='The local directory that contains the benchmark executables',
        required=True)
    parser.add_argument('--benchmark_gpu_architecture',
        help='The architecture of the currently enabled GPU',
        required=True)
    parser.add_argument('--benchmark_output_dir',
        help='The directory to write the benchmarks to',
        required=True)
    parser.add_argument('--benchmark_filename_regex',
        help='Regular expression that controls the list of benchmark executables to run',
        default=r'^benchmark',
        required=False)
    parser.add_argument('--benchmark_filter_regex',
        help='Regular expression that controls the list of benchmarks to run in each benchmark executable',
        default='',
        required=False)

    args = parser.parse_args()

    benchmark_context = BenchmarkContext(
        args.benchmark_gpu_architecture,
        args.benchmark_output_dir,
        args.benchmark_dir,
        args.benchmark_filename_regex,
        args.benchmark_filter_regex)

    benchmark_run_successful = run_benchmarks(benchmark_context)

    return benchmark_run_successful


if __name__ == '__main__':
    success = main()
    if success:
        exit(0)
    else:
        exit(1)
