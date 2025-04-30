# ##########################################################################
# Copyright (C) 2024 Advanced Micro Devices, Inc. All rights reserved.
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


#################################################
######### Benchmark suites definitions ##########
#################################################
common = '--iters 3 --perf 1' #always do 3 iterations in perf mode

"""
SYEVD tests are run, for the given precision and sizes, with vectors and without vectors
"""
def syevd_heevd_suite(*, suite, precision, sizenormal, sizebatch):
    fn = 'syevd' if precision == 's' or precision == 'd' else 'heevd'
    size = sizenormal
    for v in ['V', 'N']:
        if v == 'V': vv = 'yes'
        else: vv = 'no'
        for s in size:
            row = {'name': precision+suite, 'name_test': suite, 'function': fn, 'precision': precision, 'evect': vv, 'n': s}
            yield (row, s, f'-f {fn} -r {precision} --evect {v} -n {s} {common}')

"""
SYEVDX tests are run, for the given precision and sizes, with vectors and without vectors and
computing 20, 60 and 100 percent of the eigenvalues
"""
def syevdx_heevdx_suite(*, suite, precision, sizenormal, sizebatch):
    fn = 'syevdx' if precision == 's' or precision == 'd' else 'heevdx'
    size=sizenormal
    for per in [20, 60, 100]:
        for v in ['V', 'N']:
            if v == 'V': vv = 'yes'
            else: vv = 'no'
            for s in size:
                p = int(s * per / 100)
                if p == 0: p = 1
                row = {'name': precision+suite, 'name_test': suite, 'function': fn, 'precision': precision, 'range': per, 'evect': vv, 'n': s}
                yield (row, s, f'-f {fn} -r {precision} --erange I --il 1 --iu {p} --evect {v} -n {s} {common}')

"""
SYEVJ tests are run, for the given precision and sizes, with vectors and without vectors
"""
def syevj_heevj_suite(*, suite, precision, sizenormal, sizebatch):
    fn = 'syevj' if precision == 's' or precision == 'd' else 'heevj'
    size = sizenormal
    for v in ['V', 'N']:
        if v == 'V': vv = 'yes'
        else: vv = 'no'
        for s in size:
            row = {'name': precision+suite, 'name_test': suite, 'function': fn, 'precision': precision, 'evect': vv, 'n': s}
            yield (row, s, f'-f {fn} -r {precision} --evect {v} -n {s} {common}')

"""
SYEVJBATCH tests are run, for the given precision and sizes, with vectors and without vectors
"""
def syevj_heevjBatch_suite(*, suite, precision, sizenormal, sizebatch):
    fn = 'syevj_strided_batched' if precision == 's' or precision == 'd' else 'heevj_strided_batched'
    size = sizebatch
    for v in ['V', 'N']:
        if v == 'V': vv = 'yes'
        else: vv = 'no'
        for s, bc in size:
            row = {'name': precision+suite, 'name_test': suite, 'function': fn, 'precision': precision, 'evect': vv, 'batch_count': bc, 'n': s}
            yield (row, s, f'-f {fn} -r {precision} --evect {v} --batch_count {bc} -n {s} {common}')

"""
GESVD tests are run, for the given precision and sizes, with vectors and without vectors
"""
def gesvd_suite(*, suite, precision, sizenormal, sizebatch):
    fn = 'gesvd'
    size = sizenormal
    for v in ['V', 'N']:
        if v == 'V': vv = 'yes'
        else: vv = 'no'
        for s in size:
            row = {'name': precision+suite, 'name_test': suite, 'function': fn, 'precision': precision, 'svect': vv, 'n': s}
            yield (row, s, f'-f {fn} -r {precision} --left_svect {v} --right_svect {v} -m {s} {common}')

"""
GESVDJ tests are run, for the given precision and sizes, with vectors and without vectors
"""
def gesvdj_suite(*, suite, precision, sizenormal, sizebatch):
    fn = 'gesvdj'
    size = sizenormal
    for v in ['V', 'N']:
        if v == 'V': vv = 'yes'
        else: vv = 'no'
        for s in size:
            row = {'name': precision+suite, 'name_test': suite, 'function': fn, 'precision': precision, 'svect': vv, 'n': s}
            yield (row, s, f'-f {fn} -r {precision} --left_svect {v} --right_svect {v} -m {s} {common}')

"""
GESVDJBATCH tests are run, for the given precision and sizes, with vectors and without vectors
"""
def gesvdjBatch_suite(*, suite, precision, sizenormal, sizebatch):
    fn = 'gesvdj_strided_batched'
    size = sizebatch
    for v in ['V', 'N']:
        if v == 'V': vv = 'yes'
        else: vv = 'no'
        for s, bc in size:
            row = {'name': precision+suite, 'name_test': suite, 'function': fn, 'precision': precision, 'evect': vv, 'batch_count': bc, 'n': s}
            yield (row, s, f'-f {fn} -r {precision} --left_svect {v} --right_svect {v} --batch_count {bc} -m {s} {common}')

"""
POTRF tests are run with the given precision and sizes
"""
def potrf_suite(*, suite, precision, sizenormal, sizebatch):
    fn = 'potrf'
    size = sizenormal
    for s in size:
        row = {'name': precision+suite, 'name_test': suite, 'function': fn, 'precision': precision, 'n': s}
        yield (row, s, f'-f {fn} -r {precision} -n {s} {common}')

"""
POTRFBATCH tests are run with the given precision and sizes
"""
def potrfBatch_suite(*, suite, precision, sizenormal, sizebatch):
    fn = 'potrf_batched'
    size = sizebatch
    for s, bc in size:
        row = {'name': precision+suite, 'name_test': suite, 'function': fn, 'precision': precision, 'batch_count': bc, 'n': s}
        yield (row, s, f'-f {fn} -r {precision} --batch_count {bc} -n {s} {common}')

"""
GEQRF tests are run, for the given precision and number of rows,
with 160, 576, 1088, 2176, and 4352 columns and also for the square case (#rows = #columns)
"""
def geqrf_suite(*, suite, precision, sizenormal, sizebatch):
    fn = 'geqrf'
    size=sizenormal
    for nc in [0, 160, 576, 1088, 2176, 4352]:
        if nc == 0: nn = 'sq'
        else: nn = nc
        for s in size:
            if nc == 0: n = s
            else: n = nc
            row = {'name': precision+suite, 'name_test': suite, 'function': fn, 'precision': precision, 'cols': nn, 'n': s}
            yield (row, s, f'-f {fn} -r {precision} -n {n} -m {s} {common}')

suites = {
  'syevd': syevd_heevd_suite,
  'syevdx': syevdx_heevdx_suite,
  'syevj': syevj_heevj_suite,
  'syevjBatch': syevj_heevjBatch_suite,
  'gesvd': gesvd_suite,
  'gesvdj': gesvdj_suite,
  'gesvdjBatch': gesvdjBatch_suite,
  'potrf': potrf_suite,
  'potrfBatch': potrfBatch_suite,
  'geqrf': geqrf_suite}


#################################################
############## Helper functions #################
#################################################
"""
SETUP_VPRINT defines the function vprint as the normal print function when
verbose output is enabled, or alternatively as a function that does nothing.
"""
def setup_vprint(args):
    global vprint
    vprint = print if args.verbose else lambda *a, **k: None

"""
CALL_ROCSOLVER_BENCH executes system call to the benchmark
client executable with the given list of arguments
"""
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

"""
EXECUTE_BENCHMARKS collects the arguments for the benchmark client, calls
the client, gets the resulting time, and put everything in file or screen
"""
def execute_benchmarks(output_file, suite, precision, case, bench_executable):
    init = False
    benchmark_generator = suites[suite];
    sizenormal = list(chain(range(2, 64, 8), range(64, 256, 32), range(256, 1024, 64)))
    sizebatch = list(chain(zip(range(2, 64, 4), repeat(5000)), zip(range(72, 164, 8), repeat(2500))))
    if case == 'medium' or case == 'large':
        sizenormal += list(chain(range(1024, 2048, 64), range(2048, 4096, 128)))
        sizebatch += list(chain(zip(range(168, 260, 8), repeat(2500)), zip(range(272, 520, 16), repeat(1000))))
    if case == 'large':
        sizenormal += list(chain(range(4096, 8192, 256), range(8192, 12300, 512)))
        sizebatch += list(chain(zip(range(544, 1050, 32), repeat(500)), zip(range(1088, 2050, 64), repeat(50))))

    for row, n, bench_args in benchmark_generator(suite=suite, precision=precision, sizenormal=sizenormal, sizebatch=sizebatch):
        out, err, exitcode = call_rocsolver_bench(bench_executable, bench_args)
        if exitcode != 0:
            sys.exit("rocsolver-bench call failure: {}".format(err))
        time = float(out)
        row['gpu_time_us'] = time
        row['log_n'] = math.log10(n)
        row['log_gpu_time_us'] = math.log10(time)
        if not init:
            results = csv.DictWriter(output_file, fieldnames=row.keys(), extrasaction='raise', dialect='excel')
            results.writeheader()
            init = True
        results.writerow(row)


#################################################
######### Main functions ########################
#################################################
if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='rocsolver-perfoptim-suite',
            description='Executes a selected suite of benchmarks and collates the results.')
    parser.add_argument('-v','--verbose',
            action='store_true',
            help='display more information about operations being performed')
    parser.add_argument('--exe',
            default='../../build/release/clients/staging/rocsolver-bench',
            help='the benchmark executable to run')
    parser.add_argument('-o',
            dest='output_path',
            default=None,
            help='the output file name for the benchmark results')
    parser.add_argument('suite',
            choices=suites.keys(),
            help='the set of benchmarks to run')
    parser.add_argument('precision',
            choices=['s', 'd', 'c' , 'z'],
            help='the precision to use for the benchmarks')
    parser.add_argument('case',
            choices=['small', 'medium', 'large'],
            help='the size case to use for the benchmarks')
    args = parser.parse_args()
    setup_vprint(args)

    if args.output_path is not None:
        with open(args.output_path, 'w', buffering=1, encoding='utf-8') as output_file:
            execute_benchmarks(output_file, args.suite, args.precision, args.case, args.exe)
    else:
        execute_benchmarks(sys.stdout, args.suite, args.precision, args.case, args.exe)


