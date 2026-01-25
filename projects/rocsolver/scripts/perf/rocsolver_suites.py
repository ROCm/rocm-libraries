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
Shared module containing benchmark suite definitions for rocSOLVER.

This module provides:
- Test suite generator functions for various rocSOLVER routines
- Common benchmark parameters
- Size configurations for different test cases
"""

from itertools import chain, repeat

# Common benchmark arguments - always do 3 iterations in perf mode
COMMON_ARGS = '--iters 3 --perf 1'


def get_size_configurations(case):
    """
    Get size configurations for normal and batched tests.

    Args:
        case: One of 'small', 'medium', or 'large'

    Returns:
        tuple: (sizenormal, sizebatch) lists
    """
    sizenormal = list(chain(range(2, 64, 8), range(64, 256, 32), range(256, 1024, 64)))
    sizebatch = list(chain(zip(range(2, 64, 4), repeat(5000)), zip(range(72, 164, 8), repeat(2500))))

    if case == 'medium' or case == 'large':
        sizenormal += list(chain(range(1024, 2048, 64), range(2048, 4096, 128)))
        sizebatch += list(chain(zip(range(168, 260, 8), repeat(2500)), zip(range(272, 520, 16), repeat(1000))))

    if case == 'large':
        sizenormal += list(chain(range(4096, 8192, 256), range(8192, 12300, 512)))
        sizebatch += list(chain(zip(range(544, 1050, 32), repeat(500)), zip(range(1088, 2050, 64), repeat(50))))

    return sizenormal, sizebatch


def syevd_heevd_suite(*, suite, precision, sizenormal, sizebatch):
    """
    SYEVD tests are run, for the given precision and sizes, with vectors and without vectors
    """
    fn = 'syevd' if precision == 's' or precision == 'd' else 'heevd'
    size = sizenormal
    for v in ['V', 'N']:
        if v == 'V': vv = 'yes'
        else: vv = 'no'
        for s in size:
            row = {'name': precision+suite, 'name_test': suite, 'function': fn, 'precision': precision, 'evect': vv, 'n': s}
            yield (row, s, f'-f {fn} -r {precision} --evect {v} -n {s} {COMMON_ARGS}')


def syevdx_heevdx_suite(*, suite, precision, sizenormal, sizebatch):
    """
    SYEVDX tests are run, for the given precision and sizes, with vectors and without vectors and
    computing 20, 60 and 100 percent of the eigenvalues
    """
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
                yield (row, s, f'-f {fn} -r {precision} --erange I --il 1 --iu {p} --evect {v} -n {s} {COMMON_ARGS}')


def syevj_heevj_suite(*, suite, precision, sizenormal, sizebatch):
    """
    SYEVJ tests are run, for the given precision and sizes, with vectors and without vectors
    """
    fn = 'syevj' if precision == 's' or precision == 'd' else 'heevj'
    size = sizenormal
    for v in ['V', 'N']:
        if v == 'V': vv = 'yes'
        else: vv = 'no'
        for s in size:
            row = {'name': precision+suite, 'name_test': suite, 'function': fn, 'precision': precision, 'evect': vv, 'n': s}
            yield (row, s, f'-f {fn} -r {precision} --evect {v} -n {s} {COMMON_ARGS}')


def syevj_heevjBatch_suite(*, suite, precision, sizenormal, sizebatch):
    """
    SYEVJBATCH tests are run, for the given precision and sizes, with vectors and without vectors
    """
    fn = 'syevj_strided_batched' if precision == 's' or precision == 'd' else 'heevj_strided_batched'
    size = sizebatch
    for v in ['V', 'N']:
        if v == 'V': vv = 'yes'
        else: vv = 'no'
        for s, bc in size:
            row = {'name': precision+suite, 'name_test': suite, 'function': fn, 'precision': precision, 'evect': vv, 'batch_count': bc, 'n': s}
            yield (row, s, f'-f {fn} -r {precision} --evect {v} --batch_count {bc} -n {s} {COMMON_ARGS}')


def gesvd_suite(*, suite, precision, sizenormal, sizebatch):
    """
    GESVD tests are run, for the given precision and sizes, with vectors and without vectors
    """
    fn = 'gesvd'
    size = sizenormal
    for v in ['V', 'N']:
        if v == 'V': vv = 'yes'
        else: vv = 'no'
        for s in size:
            row = {'name': precision+suite, 'name_test': suite, 'function': fn, 'precision': precision, 'svect': vv, 'n': s}
            yield (row, s, f'-f {fn} -r {precision} --left_svect {v} --right_svect {v} -m {s} {COMMON_ARGS}')


def gesvdj_suite(*, suite, precision, sizenormal, sizebatch):
    """
    GESVDJ tests are run, for the given precision and sizes, with vectors and without vectors
    """
    fn = 'gesvdj'
    size = sizenormal
    for v in ['V', 'N']:
        if v == 'V': vv = 'yes'
        else: vv = 'no'
        for s in size:
            row = {'name': precision+suite, 'name_test': suite, 'function': fn, 'precision': precision, 'svect': vv, 'n': s}
            yield (row, s, f'-f {fn} -r {precision} --left_svect {v} --right_svect {v} -m {s} {COMMON_ARGS}')


def gesvdjBatch_suite(*, suite, precision, sizenormal, sizebatch):
    """
    GESVDJBATCH tests are run, for the given precision and sizes, with vectors and without vectors
    """
    fn = 'gesvdj_strided_batched'
    size = sizebatch
    for v in ['V', 'N']:
        if v == 'V': vv = 'yes'
        else: vv = 'no'
        for s, bc in size:
            row = {'name': precision+suite, 'name_test': suite, 'function': fn, 'precision': precision, 'evect': vv, 'batch_count': bc, 'n': s}
            yield (row, s, f'-f {fn} -r {precision} --left_svect {v} --right_svect {v} --batch_count {bc} -m {s} {COMMON_ARGS}')


def potrf_suite(*, suite, precision, sizenormal, sizebatch):
    """
    POTRF tests are run with the given precision and sizes
    """
    fn = 'potrf'
    size = sizenormal
    for s in size:
        row = {'name': precision+suite, 'name_test': suite, 'function': fn, 'precision': precision, 'n': s}
        yield (row, s, f'-f {fn} -r {precision} -n {s} {COMMON_ARGS}')


def potrfBatch_suite(*, suite, precision, sizenormal, sizebatch):
    """
    POTRFBATCH tests are run with the given precision and sizes
    """
    fn = 'potrf_batched'
    size = sizebatch
    for s, bc in size:
        row = {'name': precision+suite, 'name_test': suite, 'function': fn, 'precision': precision, 'batch_count': bc, 'n': s}
        yield (row, s, f'-f {fn} -r {precision} --batch_count {bc} -n {s} {COMMON_ARGS}')


def geqrf_suite(*, suite, precision, sizenormal, sizebatch):
    """
    GEQRF tests are run, for the given precision and number of rows,
    with 160, 576, 1088, 2176, and 4352 columns and also for the square case (#rows = #columns)
    """
    fn = 'geqrf'
    size=sizenormal
    for nc in [0, 160, 576, 1088, 2176, 4352]:
        if nc == 0: nn = 'sq'
        else: nn = nc
        for s in size:
            if nc == 0: n = s
            else: n = nc
            row = {'name': precision+suite, 'name_test': suite, 'function': fn, 'precision': precision, 'cols': nn, 'n': s}
            yield (row, s, f'-f {fn} -r {precision} -n {n} -m {s} {COMMON_ARGS}')


# Registry of all available benchmark suites
SUITES = {
    'syevd': syevd_heevd_suite,
    'syevdx': syevdx_heevdx_suite,
    'syevj': syevj_heevj_suite,
    'syevjBatch': syevj_heevjBatch_suite,
    'gesvd': gesvd_suite,
    'gesvdj': gesvdj_suite,
    'gesvdjBatch': gesvdjBatch_suite,
    'potrf': potrf_suite,
    'potrfBatch': potrfBatch_suite,
    'geqrf': geqrf_suite
}
