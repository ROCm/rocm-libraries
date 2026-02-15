# ##########################################################################
# Copyright (C) 2024-2026 Advanced Micro Devices, Inc. All rights reserved.
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


def potrs_suite(*, suite, precision, sizenormal, sizebatch):
    """
    POTRS tests are run with the given precision and sizes, and with 1, n/2 and n right-hand-vectors
    """
    fn = 'potrs'
    size = sizenormal
    for nv in ['one', 'half_n', 'n']:
        nrhs = 1
        for s in size:
            if nv == 'half_n': nrhs = s/2
            elif nv == 'n': nrhs = s 
            row = {'name': precision+suite, 'name_test': suite, 'function': fn, 'precision': precision, 'nrhs': nv, 'n': s}
            yield (row, s, f'-f {fn} -r {precision} --nrhs {nrhs} -n {s} {COMMON_ARGS}')


def potrsBatch_suite(*, suite, precision, sizenormal, sizebatch):
    """
    POTRSBATCH tests are run with the given precision and sizes, and with 1, n/2 and n right-hand-vectors
    """
    fn = 'potrs_batched'
    size = sizebatch
    for nv in ['one', 'half_n', 'n']:
        nrhs = 1
        for s, bc in size:
            if nv == 'half_n': nrhs = s/2
            elif nv == 'n': nrhs = s
            row = {'name': precision+suite, 'name_test': suite, 'function': fn, 'precision': precision, 'batch_count': bc, 'nrhs': nv, 'n': s}
            yield (row, s, f'-f {fn} -r {precision} --batch_count {bc} --nrhs {nrhs} -n {s} {COMMON_ARGS}')


def potri_suite(*, suite, precision, sizenormal, sizebatch):
    """
    POTRI tests are run with the given precision and sizes
    """
    fn = 'potri'
    size = sizenormal
    for s in size:
        row = {'name': precision+suite, 'name_test': suite, 'function': fn, 'precision': precision, 'n': s}
        yield (row, s, f'-f {fn} -r {precision} -n {s} {COMMON_ARGS}')


def sytrf_suite(*, suite, precision, sizenormal, sizebatch):
    """
    SYTRF tests are run with the given precision and sizes
    """
    fn = 'sytrf'
    size = sizenormal
    for s in size:
        row = {'name': precision+suite, 'name_test': suite, 'function': fn, 'precision': precision, 'n': s}
        yield (row, s, f'-f {fn} -r {precision} -n {s} {COMMON_ARGS}')


def sytrs_suite(*, suite, precision, sizenormal, sizebatch):
#(TODO: this function needs to be added to rocsolver)
    """
    SYTRS tests are run with the given precision and sizes, and with 1, n/2 and n right-hand-vectors
    """
    fn = 'sytrs'
    size = sizenormal
    for nv in ['one', 'half_n', 'n']:
        nrhs = 1
        for s in size:
            if nv == 'half_n': nrhs = s/2
            elif nv == 'n': nrhs = s
            row = {'name': precision+suite, 'name_test': suite, 'function': fn, 'precision': precision, 'nrhs': nv, 'n': s}
            yield (row, s, f'-f {fn} -r {precision} --nrhs {nrhs} -n {s} {COMMON_ARGS}')


def getrf_suite(*, suite, precision, sizenormal, sizebatch):
    """
    GETRF tests are run with the given precision and sizes (only square case)
    """
    fn = 'getrf'
    size = sizenormal
    for s in size:
        row = {'name': precision+suite, 'name_test': suite, 'function': fn, 'precision': precision, 'n': s}
        yield (row, s, f'-f {fn} -r {precision} -m {s} {COMMON_ARGS}')


def getrfBatch_suite(*, suite, precision, sizenormal, sizebatch):
    """
    GETRFBATCH tests are run with the given precision and sizes (only square case)
    """
    fn = 'getrf_batched'
    size = sizebatch
    for s, bc in size:
        row = {'name': precision+suite, 'name_test': suite, 'function': fn, 'precision': precision, 'batch_count': bc, 'n': s}
        yield (row, s, f'-f {fn} -r {precision} --batch_count {bc} -m {s} {COMMON_ARGS}')


def getrfNpvt_suite(*, suite, precision, sizenormal, sizebatch):
    """
    GETRFNPVT tests are run with the given precision and sizes (only square case)
    """
    fn = 'getrf_npvt'
    size = sizenormal
    for s in size:
        row = {'name': precision+suite, 'name_test': suite, 'function': fn, 'precision': precision, 'n': s}
        yield (row, s, f'-f {fn} -r {precision} -m {s} {COMMON_ARGS}')


def getrfNpvtBatch_suite(*, suite, precision, sizenormal, sizebatch):
    """
    GETRFNPVTBATCH tests are run with the given precision and sizes (only square case)
    """
    fn = 'getrf_npvt_batched'
    size = sizebatch
    for s, bc in size:
        row = {'name': precision+suite, 'name_test': suite, 'function': fn, 'precision': precision, 'batch_count': bc, 'n': s}
        yield (row, s, f'-f {fn} -r {precision} --batch_count {bc} -m {s} {COMMON_ARGS}')


def getrs_suite(*, suite, precision, sizenormal, sizebatch):
    """
    GETRS tests are run with the given precision and sizes, and with 1, n/2 and n right-hand-vectors
    """
    fn = 'getrs'
    size = sizenormal
    for nv in ['one', 'half_n', 'n']:
        nrhs = 1
        for s in size:
            if nv == 'half_n': nrhs = s/2
            elif nv == 'n': nrhs = s 
            row = {'name': precision+suite, 'name_test': suite, 'function': fn, 'precision': precision, 'nrhs': nv, 'n': s}
            yield (row, s, f'-f {fn} -r {precision} --nrhs {nrhs} -n {s} {COMMON_ARGS}')


def getrsBatch_suite(*, suite, precision, sizenormal, sizebatch):
    """
    GETRSBATCH tests are run with the given precision and sizes, and with 1, n/2 and n right-hand-vectors
    """
    fn = 'getrs_batched'
    size = sizebatch
    for nv in ['one', 'half_n', 'n']:
        nrhs = 1
        for s, bc in size:
            if nv == 'half_n': nrhs = s/2
            elif nv == 'n': nrhs = s
            row = {'name': precision+suite, 'name_test': suite, 'function': fn, 'precision': precision, 'batch_count': bc, 'nrhs': nv, 'n': s}
            yield (row, s, f'-f {fn} -r {precision} --batch_count {bc} --nrhs {nrhs} -n {s} {COMMON_ARGS}')


def getrsNpvt_suite(*, suite, precision, sizenormal, sizebatch):
#(TODO: this function needs to be added to rocsolver)
    """
    GETRSNPVT tests are run with the given precision and sizes, and with 1, n/2 and n right-hand-vectors
    """
    fn = 'getrs_npvt'
    size = sizenormal
    for nv in ['one', 'half_n', 'n']:
        nrhs = 1
        for s in size:
            if nv == 'half_n': nrhs = s/2
            elif nv == 'n': nrhs = s
            row = {'name': precision+suite, 'name_test': suite, 'function': fn, 'precision': precision, 'nrhs': nv, 'n': s}
            yield (row, s, f'-f {fn} -r {precision} --nrhs {nrhs} -n {s} {COMMON_ARGS}')


def getrsNpvtBatch_suite(*, suite, precision, sizenormal, sizebatch):
#(TODO: this function needs to be added to rocsolver)
    """
    GETRSNPVTBATCH tests are run with the given precision and sizes, and with 1, n/2 and n right-hand-vectors
    """
    fn = 'getrs_npvt_batched'
    size = sizebatch
    for nv in ['one', 'half_n', 'n']:
        nrhs = 1
        for s, bc in size:
            if nv == 'half_n': nrhs = s/2
            elif nv == 'n': nrhs = s
            row = {'name': precision+suite, 'name_test': suite, 'function': fn, 'precision': precision, 'batch_count': bc, 'nrhs': nv, 'n': s}
            yield (row, s, f'-f {fn} -r {precision} --batch_count {bc} --nrhs {nrhs} -n {s} {COMMON_ARGS}')


def getriBatch_suite(*, suite, precision, sizenormal, sizebatch):
    """
    GETRIBATCH tests are run with the given precision and sizes
    """
    fn = 'getri_batched'
    size = sizebatch
    for s, bc in size:
        row = {'name': precision+suite, 'name_test': suite, 'function': fn, 'precision': precision, 'batch_count': bc, 'n': s}
        yield (row, s, f'-f {fn} -r {precision} --batch_count {bc} -n {s} {COMMON_ARGS}')


def getriOOPBatch_suite(*, suite, precision, sizenormal, sizebatch):
    """
    GETRIOOPBATCH tests are run with the given precision and sizes
    """
    fn = 'getri_outofplace_batched'
    size = sizebatch
    for s, bc in size:
        row = {'name': precision+suite, 'name_test': suite, 'function': fn, 'precision': precision, 'batch_count': bc, 'n': s}
        yield (row, s, f'-f {fn} -r {precision} --batch_count {bc} -n {s} {COMMON_ARGS}')


def trtri_suite(*, suite, precision, sizenormal, sizebatch):
    """
    TRTRI tests are run with the given precision and sizes
    """
    fn = 'trtri'
    size = sizenormal
    for s in size:
        row = {'name': precision+suite, 'name_test': suite, 'function': fn, 'precision': precision, 'n': s}
        yield (row, s, f'-f {fn} -r {precision} -n {s} {COMMON_ARGS}')


def geqrf_suite(*, suite, precision, sizenormal, sizebatch):
    """
    GEQRF tests are run, for the given precision and number of rows,
    with 160 columns and also for the square case (#rows = #columns)
    """
    fn = 'geqrf'
    size=sizenormal
    for nc in [0, 160]:
        if nc == 0: nn = 'sq'
        else: nn = nc
        for s in size:
            if nc == 0: n = s
            else: n = nc
            if s >= n:
                row = {'name': precision+suite, 'name_test': suite, 'function': fn, 'precision': precision, 'cols': nn, 'n': s}
                yield (row, s, f'-f {fn} -r {precision} -n {n} -m {s} {COMMON_ARGS}')


def geqrfBatch_suite(*, suite, precision, sizenormal, sizebatch):
    """
    GEQRFBATCH tests are run, for the given precision and number of rows,
    with 26 columns and also for the square case (#rows = #columns)
    """
    fn = 'geqrf_batched'
    size = sizebatch
    for nc in [0, 26]:
        if nc == 0: nn = 'sq'
        else: nn = nc
        for s, bc in size:
            if nc == 0: n = s
            else: n = nc
            if s >= n:
                row = {'name': precision+suite, 'name_test': suite, 'function': fn, 'precision': precision, 'batch_count': bc, 'cols': nn, 'n': s}
                yield (row, s, f'-f {fn} -r {precision} --batch_count {bc} -n {n} -m {s} {COMMON_ARGS}')


def cholqr_suite(*, suite, precision, sizenormal, sizebatch):
#(TODO: this function needs to be added to rocsolver)
    """
    CHOLQR tests are run, for the given precision and number of rows,
    with 160 columns and also for the square case (#rows = #columns).
    Tests run for cholqr1 and cholqr2 variants.
    """
    fn = 'cholqr'
    size=sizenormal
    for nc in [0, 160]:
        if nc == 0: nn = 'sq'
        else: nn = nc
        for alg in [1, 2]:
            for s in size:
                if nc == 0: n = s
                else: n = nc
                if s >= n:
                    row = {'name': precision+suite, 'name_test': suite, 'function': fn, 'precision': precision, 'cols': nn, 'algo': alg, 'n': s}
                    yield (row, s, f'-f {fn} -r {precision} -n {n} --cholqr_algo {alg} -m {s} {COMMON_ARGS}')


def cholqrBatch_suite(*, suite, precision, sizenormal, sizebatch):
#(TODO: this function needs to be added to rocsolver)
    """
    CHOLQRBATCH tests are run, for the given precision and number of rows,
    with 26 columns and also for the square case (#rows = #columns)
    Tests run for cholqr1 and cholqr2 variants.
    """
    fn = 'cholqr_batched'
    size = sizebatch
    for nc in [0, 26]:
        if nc == 0: nn = 'sq'
        else: nn = nc
        for alg in [1, 2]:
            for s, bc in size:
                if nc == 0: n = s
                else: n = nc
                if s >= n:
                    row = {'name': precision+suite, 'name_test': suite, 'function': fn, 'precision': precision, 'batch_count': bc, 'cols': nn, 'algo': alg, 'n': s}
                    yield (row, s, f'-f {fn} -r {precision} --batch_count {bc} -n {n} --cholqr_algo {alg} -m {s} {COMMON_ARGS}')


def gels_suite(*, suite, precision, sizenormal, sizebatch):
    """
    GELS tests are run, for the given precision and number of rows, with 160 columns and with 1, 
    n/2 and n right-hand-vectors. Gels is based on QR only when m > n.
    """
    fn = 'gels'
    size = sizenormal
    for nv in ['one', 'half_n', 'n']:
        nrhs = 1
        for s in size:
            if nv == 'half_n': nrhs = s/2
            elif nv == 'n': nrhs = s
            if s >= 160:
                row = {'name': precision+suite, 'name_test': suite, 'function': fn, 'precision': precision, 'nrhs': nv, 'n': s}
                yield (row, s, f'-f {fn} -r {precision} -n 160 --nrhs {nrhs} -m {s} {COMMON_ARGS}')


def gelsBatch_suite(*, suite, precision, sizenormal, sizebatch):
    """
    GELSBATCH tests are run, for the given precision and number of rows, with 26 columns and with 1, 
    n/2 and n right-hand-vectors. Gels is based on QR only when m > n.
    """
    fn = 'gels_batched'
    size = sizebatch
    for nv in ['one', 'half_n', 'n']:
        nrhs = 1
        for s, bc in size:
            if nv == 'half_n': nrhs = s/2
            elif nv == 'n': nrhs = s
            if s >= 26:
                row = {'name': precision+suite, 'name_test': suite, 'function': fn, 'precision': precision, 'batch_count': bc, 'nrhs': nv, 'n': s}
                yield (row, s, f'-f {fn} -r {precision} --batch_count {bc} -n 26 --nrhs {nrhs} -m {s} {COMMON_ARGS}')


def xxgqr_suite(*, suite, precision, sizenormal, sizebatch):
    """
    XXGQR (ORGQR or UNGQR) tests are run, for the given precision and number of rows,
    with 160 columns and also for the square case (#rows = #columns)
    """
    fn = 'orgqr' if precision == 's' or precision == 'd' else 'ungqr'
    size=sizenormal
    for nc in [0, 160]:
        if nc == 0: nn = 'sq'
        else: nn = nc
        for s in size:
            if nc == 0: n = s
            else: n = nc
            if s >= n:
                row = {'name': precision+suite, 'name_test': suite, 'function': fn, 'precision': precision, 'cols': nn, 'n': s}
                yield (row, s, f'-f {fn} -r {precision} -n {n} -m {s} {COMMON_ARGS}')


def xxmqr_suite(*, suite, precision, sizenormal, sizebatch):
    """
    XXMQR (ORMQR or UNMQR) tests are run with the given precision and sizes (only square case), from the left.
    """
    fn = 'ormqr' if precision == 's' or precision == 'd' else 'unmqr'
    size = sizenormal
    for s in size:
        row = {'name': precision+suite, 'name_test': suite, 'function': fn, 'precision': precision, 'n': s}
        yield (row, s, f'-f {fn} -r {precision} --side L -m {s} {COMMON_ARGS}')


def larft_suite(*, suite, precision, sizenormal, sizebatch):
    """
    LARFT tests are run with the given precision and sizes, columns-wise and
    forward direction. Tests use 1, n/2 and n Householder vectors.
    """
    fn = 'larft'
    size = sizenormal
    for nk in ['one', 'half_n', 'n']:
        k = 1
        for s in size:
            if nk == 'half_n': k = s/2
            elif nk == 'n': k = s
            row = {'name': precision+suite, 'name_test': suite, 'function': fn, 'precision': precision, 'nk': nk, 'n': s}
            yield (row, s, f'-f {fn} -r {precision} --storev C -k {k} -n {s} {COMMON_ARGS}')


def xxtrd_suite(*, suite, precision, sizenormal, sizebatch):
    """
    XXTRD (SYTRD or HETRD) tests are run with the given precision and sizes
    """
    fn = 'sytrd' if precision == 's' or precision == 'd' else 'hetrd'
    size = sizenormal
    for s in size:
        row = {'name': precision+suite, 'name_test': suite, 'function': fn, 'precision': precision, 'n': s}
        yield (row, s, f'-f {fn} -r {precision} -n {s} {COMMON_ARGS}')


def xxgtr_suite(*, suite, precision, sizenormal, sizebatch):
    """
    XXGTR (ORGTR or UNGTR) tests are run with the given precision and sizes.
    Always upper to actually use orgql/ungql.
    """
    fn = 'orgtr' if precision == 's' or precision == 'd' else 'ungtr'
    size = sizenormal
    for s in size:
        row = {'name': precision+suite, 'name_test': suite, 'function': fn, 'precision': precision, 'n': s}
        yield (row, s, f'-f {fn} -r {precision} --uplo U -n {s} {COMMON_ARGS}')


def xxmtr_suite(*, suite, precision, sizenormal, sizebatch):
    """
    XXMTR (ORMTR or UNMTR) tests are run with the given precision and sizes (only square case), from the left.
    Always upper to actually use ormql/unmql.
    """
    fn = 'ormtr' if precision == 's' or precision == 'd' else 'unmtr'
    size = sizenormal
    for s in size:
        row = {'name': precision+suite, 'name_test': suite, 'function': fn, 'precision': precision, 'n': s}
        yield (row, s, f'-f {fn} -r {precision} --uplo U --side L -m {s} {COMMON_ARGS}')


def gebrd_suite(*, suite, precision, sizenormal, sizebatch):
    """
    GEBRD tests are run with the given precision and sizes (only square case)
    """
    fn = 'gebrd'
    size = sizenormal
    for s in size:
        row = {'name': precision+suite, 'name_test': suite, 'function': fn, 'precision': precision, 'n': s}
        yield (row, s, f'-f {fn} -r {precision} -m {s} {COMMON_ARGS}')


def xxgbr_suite(*, suite, precision, sizenormal, sizebatch):
    """
    XXGBR (ORGBR or UNGBR) tests are run with the given precision and sizes (only square case). 
    Always row-wise to actually use orglq/unglq.
    """
    fn = 'orgbr' if precision == 's' or precision == 'd' else 'ungbr'
    size = sizenormal
    for s in size:
        row = {'name': precision+suite, 'name_test': suite, 'function': fn, 'precision': precision, 'n': s}
        yield (row, s, f'-f {fn} -r {precision} --storev R -m {s} {COMMON_ARGS}')


def xxevd_suite(*, suite, precision, sizenormal, sizebatch):
    """
    XXEVD (SYEVD or HEEVD) tests are run, for the given precision and sizes, with vectors and without vectors
    """
    fn = 'syevd' if precision == 's' or precision == 'd' else 'heevd'
    size = sizenormal
    for v in ['V', 'N']:
        if v == 'V': vv = 'yes'
        else: vv = 'no'
        for s in size:
            row = {'name': precision+suite, 'name_test': suite, 'function': fn, 'precision': precision, 'evect': vv, 'n': s}
            yield (row, s, f'-f {fn} -r {precision} --evect {v} -n {s} {COMMON_ARGS}')


def xxgvd_suite(*, suite, precision, sizenormal, sizebatch):
    """
    XXGVD (SYGVD or HEGVD) tests are run, for the given precision and sizes, with vectors and without vectors
    """
    fn = 'sygvd' if precision == 's' or precision == 'd' else 'hegvd'
    size = sizenormal
    for v in ['V', 'N']:
        if v == 'V': vv = 'yes'
        else: vv = 'no'
        for s in size:
            row = {'name': precision+suite, 'name_test': suite, 'function': fn, 'precision': precision, 'evect': vv, 'n': s}
            yield (row, s, f'-f {fn} -r {precision} --evect {v} -n {s} {COMMON_ARGS}')


def xxevdBatch_suite(*, suite, precision, sizenormal, sizebatch):
    """
    XXEVDBATCH (SYEVDBATCH or HEEVDBATCH) tests are run, for the given precision and sizes, with vectors and without vectors
    """
    fn = 'syevd_strided_batched' if precision == 's' or precision == 'd' else 'heevd_strided_batched'
    size = sizebatch
    for v in ['V', 'N']:
        if v == 'V': vv = 'yes'
        else: vv = 'no'
        for s, bc in size:
            row = {'name': precision+suite, 'name_test': suite, 'function': fn, 'precision': precision, 'evect': vv, 'batch_count': bc, 'n': s}
            yield (row, s, f'-f {fn} -r {precision} --evect {v} --batch_count {bc} -n {s} {COMMON_ARGS}')


def xxevBatch_suite(*, suite, precision, sizenormal, sizebatch):
    """
    XXEVBATCH (SYEVBATCH or HEEVBATCH) tests are run, for the given precision and sizes, with vectors and without vectors
    """
    fn = 'syev_strided_batched' if precision == 's' or precision == 'd' else 'heev_strided_batched'
    size = sizebatch
    for v in ['V', 'N']:
        if v == 'V': vv = 'yes'
        else: vv = 'no'
        for s, bc in size:
            row = {'name': precision+suite, 'name_test': suite, 'function': fn, 'precision': precision, 'evect': vv, 'batch_count': bc, 'n': s}
            yield (row, s, f'-f {fn} -r {precision} --evect {v} --batch_count {bc} -n {s} {COMMON_ARGS}')


def xxevdx_suite(*, suite, precision, sizenormal, sizebatch):
    """
    XXEVDX (SYEVDX or HEEVDX) tests are run, for the given precision and sizes, with vectors and without vectors and
    computing 20 and 60 percent of the eigenvalues
    """
    fn = 'syevdx' if precision == 's' or precision == 'd' else 'heevdx'
    size=sizenormal
    for per in [20, 60]:
        for v in ['V', 'N']:
            if v == 'V': vv = 'yes'
            else: vv = 'no'
            for s in size:
                p = int(s * per / 100)
                if p == 0: p = 1
                row = {'name': precision+suite, 'name_test': suite, 'function': fn, 'precision': precision, 'range': per, 'evect': vv, 'n': s}
                yield (row, s, f'-f {fn} -r {precision} --erange I --il 1 --iu {p} --evect {v} -n {s} {COMMON_ARGS}')


def xxgvdx_suite(*, suite, precision, sizenormal, sizebatch):
    """
    XXGVDX (SYGVDX or HEGVDX) tests are run, for the given precision and sizes, with vectors and without vectors and
    computing 20 and 60 percent of the eigenvalues
    """
    fn = 'sygvdx' if precision == 's' or precision == 'd' else 'hegvdx'
    size=sizenormal
    for per in [20, 60]:
        for v in ['V', 'N']:
            if v == 'V': vv = 'yes'
            else: vv = 'no'
            for s in size:
                p = int(s * per / 100)
                if p == 0: p = 1
                row = {'name': precision+suite, 'name_test': suite, 'function': fn, 'precision': precision, 'range': per, 'evect': vv, 'n': s}
                yield (row, s, f'-f {fn} -r {precision} --erange I --il 1 --iu {p} --evect {v} -n {s} {COMMON_ARGS}')


def xxevj_suite(*, suite, precision, sizenormal, sizebatch):
    """
    XXEVJ (SYEVJ or HEEVJ) tests are run, for the given precision and sizes, with vectors and without vectors
    """
    fn = 'syevj' if precision == 's' or precision == 'd' else 'heevj'
    size = sizenormal
    for v in ['V', 'N']:
        if v == 'V': vv = 'yes'
        else: vv = 'no'
        for s in size:
            row = {'name': precision+suite, 'name_test': suite, 'function': fn, 'precision': precision, 'evect': vv, 'n': s}
            yield (row, s, f'-f {fn} -r {precision} --evect {v} -n {s} {COMMON_ARGS}')


def xxgvj_suite(*, suite, precision, sizenormal, sizebatch):
    """
    XXGVJ (SYGVJ or HEGVJ) tests are run, for the given precision and sizes, with vectors and without vectors
    """
    fn = 'sygvj' if precision == 's' or precision == 'd' else 'hegvj'
    size = sizenormal
    for v in ['V', 'N']:
        if v == 'V': vv = 'yes'
        else: vv = 'no'
        for s in size:
            row = {'name': precision+suite, 'name_test': suite, 'function': fn, 'precision': precision, 'evect': vv, 'n': s}
            yield (row, s, f'-f {fn} -r {precision} --evect {v} -n {s} {COMMON_ARGS}')


def xxevjBatch_suite(*, suite, precision, sizenormal, sizebatch):
    """
    XXEVJBATCH (SYEVJBATCH or HEEVJBATCH) tests are run, for the given precision and sizes, with vectors and without vectors
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
    GESVD tests are run, for the given precision and sizes, with vectors and without vectors (only square case).
    Tests are run wityh the hybrid approach as well. 
    """
    fn = 'gesvd'
    size = sizenormal
    for alg in [1 ,0]:
        if alg == 0: hyb = 'no'
        else: hyb = 'yes'
        for v in ['V', 'N']:
            if v == 'V': vv = 'yes'
            else: vv = 'no'
            for s in size:
                row = {'name': precision+suite, 'name_test': suite, 'function': fn, 'precision': precision, 'hybrid': hyb, 'svect': vv, 'n': s}
                yield (row, s, f'-f {fn} -r {precision} --alg_mode {alg} --left_svect {v} --right_svect {v} -m {s} {COMMON_ARGS}')


def gesdd_suite(*, suite, precision, sizenormal, sizebatch):
    """
    GESDD tests are run, for the given precision and sizes, with vectors and without vectors (only square case).
    """
    fn = 'gesdd'
    size = sizenormal
    for v in ['V', 'N']:
        if v == 'V': vv = 'yes'
        else: vv = 'no'
        for s in size:
            row = {'name': precision+suite, 'name_test': suite, 'function': fn, 'precision': precision, 'svect': vv, 'n': s}
            yield (row, s, f'-f {fn} -r {precision} --left_svect {v} --right_svect {v} -m {s} {COMMON_ARGS}')


def gesvdj_suite(*, suite, precision, sizenormal, sizebatch):
    """
    GESVDJ tests are run, for the given precision and sizes, with vectors and without vectors (only square case).
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
    GESVDJBATCH tests are run, for the given precision and sizes, with vectors and without vectors (only square case).
    """
    fn = 'gesvdj_strided_batched'
    size = sizebatch
    for v in ['V', 'N']:
        if v == 'V': vv = 'yes'
        else: vv = 'no'
        for s, bc in size:
            row = {'name': precision+suite, 'name_test': suite, 'function': fn, 'precision': precision, 'evect': vv, 'batch_count': bc, 'n': s}
            yield (row, s, f'-f {fn} -r {precision} --left_svect {v} --right_svect {v} --batch_count {bc} -m {s} {COMMON_ARGS}')






# Registry of all available benchmark suites
SUITES = {
    # Symmetric linear systems
    'potrf': potrf_suite,
    'potrfBatch': potrfBatch_suite,
    'potrs': potrs_suite,
    'potrsBatch': potrsBatch_suite,
    'potri': potri_suite,
    'sytrf': sytrf_suite,
    'sytrs': sytrs_suite,                       #(TODO: needs to be added to rocsolver)
    
    # General linear systems
    'getrf': getrf_suite,
    'getrfBatch': getrfBatch_suite,
    'getrfNpvt': getrfNpvt_suite,
    'getrfNpvtBatch': getrfNpvtBatch_suite,
    'getrs': getrs_suite,
    'getrsBatch': getrsBatch_suite,
    'getrsNpvt': getrsNpvt_suite,               #(TODO: needs to be added to rocsolver)
    'getrsNpvtBatch': getrsNpvtBatch_suite,     #(TODO: needs to be added to rocsolver)
    'getriBatch': getriBatch_suite,
    'getriOOPBatch': getriOOPBatch_suite,
    'trtri': trtri_suite,

    # Over-determined linear systems (least-squares)
    'geqrf': geqrf_suite,
    'geqrfBatch': geqrfBatch_suite,
    'cholqr': cholqr_suite,                     #(TODO: needs to be added to rocsolver)
    'cholqrBatch': cholqrBatch_suite,           #(TODO: needs to be added to rocsolver)
    'gels': gels_suite,                          
    'gelsBatch': gelsBatch_suite,               
    'xxgqr': xxgqr_suite,
    'xxmqr': xxmqr_suite,
    'larft': larft_suite,

    # Matrix reductions (tridiagonalization, bidiagonalization)
    'xxtrd': xxtrd_suite, 
    'xxgtr': xxgtr_suite,           
    'xxmtr': xxmtr_suite,           
    'gebrd': gebrd_suite,
    'xxgbr': xxgbr_suite,           
 
    # Symmetric Eigenvalue problem
    'xxevd': xxevd_suite,
    'xxgvd': xxgvd_suite,
    'xxevdBatch': xxevdBatch_suite,
    'xxevBatch': xxevBatch_suite,
    'xxevdx': xxevdx_suite,
    'xxgvdx': xxgvdx_suite,
    'xxevj': xxevj_suite,
    'xxgvj': xxgvj_suite,
    'xxevjBatch': xxevjBatch_suite,

    # Singular value decomposition
    'gesvd': gesvd_suite,
    'gesdd': gesdd_suite,
    'gesvdj': gesvdj_suite,
    'gesvdjBatch': gesvdjBatch_suite,
}
