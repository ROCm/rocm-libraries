# ##########################################################################
# Copyright (C) 2022 Advanced Micro Devices, Inc. All rights reserved.
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

import argparse
import collections
import csv
import math
import os
import re
import shlex
import sys
from itertools import chain, repeat
from subprocess import Popen, PIPE

def setup_vprint(args):
    """
    Defines the function vprint as the normal print function when verbose output
    is enabled, or alternatively as a function that does nothing.
    """
    global vprint
    vprint = print if args.verbose else lambda *a, **k: None

def call_rocsolver_bench(bench_executable, *args):
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

class ParseError(Exception):
    pass

def parse_arguments(bench_output):
    m = re.search(r"\n=+\nArguments:\s*\n=+\n(?P<arg_names>.*)\n(?P<arg_values>.*)\n",
            bench_output, re.MULTILINE)
    if not m:
        raise ParseError("Failed to parse arguments")
    arg_names = m.group('arg_names').split()
    arg_values = m.group('arg_values').split()
    if len(arg_names) != len(arg_values):
        raise ParseError("Mismatched argument labels and values")
    return dict(zip(arg_names, arg_values))

def parse_results(bench_output):
    m = re.search(r"\n=+\nResults:\s*\n=+\n(?P<arg_names>.*)\n(?P<arg_values>.*)\n",
            bench_output, re.MULTILINE)
    if not m:
        raise ParseError("Failed to parse results")
    arg_names = m.group('arg_names').split()
    arg_values = m.group('arg_values').split()
    if len(arg_names) != len(arg_values):
        raise ParseError("Mismatched result labels and values")
    return dict(zip(arg_names, arg_values))

def getrf_suite(*, precision):
    for fn in ['getrf', 'getrf_npvt']:
        for m in chain(range(2, 64, 8),
                       range(64, 256, 32),
                       range(256, 2048, 64),
                       range(2048, 4096, 128),
                       range(4096, 8193, 256)):
            yield (fn, f'-f {fn} -r {precision} -m {m} --iters 10')

def getrf_strided_batched_suite(*, precision):
    for fn in ['getrf_strided_batched', 'getrf_npvt_strided_batched']:
        for m, bc in chain(zip(range(2, 64, 1), repeat(5000)),
                           zip(range(64, 256, 8), repeat(2500)),
                           zip(range(256, 384, 16), repeat(1000)),
                           zip(range(384, 512, 32), repeat(750)),
                           zip(range(512, 640, 32), repeat(500)),
                           zip(range(640, 1025, 64), repeat(50))):
            yield (fn, f'-f {fn} -r {precision} -m {m} --iters 10 --batch_count {bc}')

        yield (fn, f'-f {fn} -r {precision} -m 20 --iters 10 --batch_count 4096')
        yield (fn, f'-f {fn} -r {precision} -m 20 --iters 10 --batch_count 32768')

        yield (fn, f'-f {fn} -r {precision} -m 50 --iters 10 --batch_count 4096')
        yield (fn, f'-f {fn} -r {precision} -m 50 --iters 10 --batch_count 32768')

        yield (fn, f'-f {fn} -r {precision} -m 64 --iters 10 --batch_count 1024')
        yield (fn, f'-f {fn} -r {precision} -m 64 --iters 10 --batch_count 2048')
        yield (fn, f'-f {fn} -r {precision} -m 64 --iters 10 --batch_count 4096')

        yield (fn, f'-f {fn} -r {precision} -m 80 --iters 10 --batch_count 4096')
        yield (fn, f'-f {fn} -r {precision} -m 80 --iters 10 --batch_count 32768')

        yield (fn, f'-f {fn} -r {precision} -m 161 --iters 10 --batch_count 1024')
        yield (fn, f'-f {fn} -r {precision} -m 161 --iters 10 --batch_count 2048')
        yield (fn, f'-f {fn} -r {precision} -m 161 --iters 10 --batch_count 4096')

def getri_suite(*, precision):
    for fn in ['getri', 'getri_npvt']:
        for n in chain(range(2, 64, 8),
                       range(64, 256, 32),
                       range(256, 1024, 64),
                       range(1024, 2048, 128),
                       range(2048, 4096, 256),
                       range(4096, 8193, 512)):
            yield (fn, f'-f {fn} -r {precision} -n {n} --iters 10')


def getri_strided_batched_suite(*, precision):
    for fn in ['getri_strided_batched', 'getri_npvt_strided_batched']:
        for n, bc in chain(zip(range(2, 64, 1), repeat(5000)),
                           zip(range(64, 256, 8), repeat(2500)),
                           zip(range(256, 384, 16), repeat(1000)),
                           zip(range(384, 512, 32), repeat(750)),
                           zip(range(512, 640, 32), repeat(500)),
                           zip(range(640, 1025, 64), repeat(50))):
            if precision == 'z' and 232 <= n:
                continue # TODO: fix crash in rocsolver-bench at these sizes
            yield (fn, f'-f {fn} -r {precision} -n {n} --iters 10 --batch_count {bc}')

def geqrf_suite(*, precision):
    for fn in ['geqrf']:
        for m in chain(range(2, 64, 8),
                       range(64, 256, 32),
                       range(256, 1024, 64),
                       range(1024, 2048, 128),
                       range(2048, 4096, 256),
                       range(4096, 8193, 512)):
            yield (fn, f'-f {fn} -r {precision} -m {m} --iters 10')

def geqrf_strided_batched_suite(*, precision):
    for fn in ['geqrf_strided_batched']:
        for m, bc in chain(zip(range(2, 64, 1), repeat(5000)),
                           zip(range(64, 256, 8), repeat(2500)),
                           zip(range(256, 384, 16), repeat(1000)),
                           zip(range(384, 512, 32), repeat(750)),
                           zip(range(512, 640, 32), repeat(500)),
                           zip(range(640, 1025, 64), repeat(50))):
            yield (fn, f'-f {fn} -r {precision} -m {m} --iters 10 --batch_count {bc}')

suites = {
  'geqrf': geqrf_suite,
  'geqrf_strided_batched': geqrf_strided_batched_suite,
  'getrf': getrf_suite,
  'getrf_strided_batched': getrf_strided_batched_suite,
  'getri': getri_suite,
  'getri_strided_batched': getri_strided_batched_suite,
}

def execute_benchmarks(output_file, precision, suite, bench_executable):
    init = False
    benchmark_generator = suites[suite];
    for fn, bench_args in benchmark_generator(precision=precision):
        out, err, exitcode = call_rocsolver_bench(bench_executable, bench_args)
        if exitcode != 0:
            sys.exit("rocsolver-bench call failure: {}".format(err))
        args = parse_arguments(out)
        perf = parse_results(out)
        row = { 'function': fn, 'precision': precision, **args, **perf }
        if not init:
            results = csv.DictWriter(output_file, fieldnames=row.keys(), extrasaction='raise', dialect='excel')
            results.writeheader()
            init = True
        results.writerow(row)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='rocsolver-bench-suite',
            description='Executes a selected suite of benchmarks and collates the results.')
    parser.add_argument('-v','--verbose',
            action='store_true',
            help='display more information about operations being performed')
    parser.add_argument('--exe',
            default='rocsolver-bench',
            help='the benchmark executable to run')
    parser.add_argument('-o',
            dest='output_path',
            default=None,
            help='the output file name for the benchmark results')
    parser.add_argument('precision',
            choices=['s', 'd', 'c' , 'z'],
            help='the precision to use for the benchmark')
    parser.add_argument('suite',
            choices=suites.keys(),
            help='the set of benchmarks to run')
    args = parser.parse_args()
    setup_vprint(args)

    if args.output_path is not None:
        with open(args.output_path, 'w', buffering=1, encoding='utf-8') as output_file:
            execute_benchmarks(output_file, args.precision, args.suite, args.exe)
    else:
        execute_benchmarks(sys.stdout, args.precision, args.suite, args.exe)
