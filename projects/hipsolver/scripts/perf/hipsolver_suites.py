# ##########################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
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
Shared module containing benchmark suite definitions for hipSOLVER.

This module provides:
- Test suite generator functions for various hipSOLVER routines
- Common benchmark parameters
- Size configurations for different test cases
"""

from itertools import chain, repeat

# Common benchmark arguments - always do 5 iterations in perf mode
COMMON_ARGS = '--iters 5 --perf 1'


# Common helpers
###########################################

def get_ld(s):
    """
    Gets leading dimension depending on the size. 
    All the used sizes "n" are even. Relatively better performance is observed when the leading dimension "ld" is 
    not exaclty equal to the size. Based on observations, we are taking ld = n + 1 if n < 4000, and ld = n + 64 otherwise.
    This could be revisited and changed in the future   
    """
    if s < 4000: ld = s + 1
    else: ld = s + 64
    return ld


def get_uplo(s_uplo):
    """
    Gets uplo (default is upper U)
    """
    if s_uplo == 'lower': uplo = 'L'
    else: uplo = 'U'
    return uplo


def get_nrhs(s_nrhs, s):
    """
    Gets nrhs (default is 1)
    """
    if s_nrhs == 'n': nrhs = s
    elif s_nrhs == 'half_n': nrhs = s//2
    else: nrhs = 1
    return nrhs 


def get_mn(s_shape, s, mode):
    """
    Gets the number of columns and rows depending on the shape (default is square-normal)
    """
    if mode == 'batched': mn = 26
    else: mn = 160
    if s_shape == 'skinny' or s_shape == 'overdet':
        m = s
        n = mn
    elif s_shape == 'underdet': 
        n = s
        m = mn
    else:
        m = s
        n = s
    return m,n


def get_size_configurations(case):
    """
    Get size configurations for normal and batched tests.
    Args: a list with one or more of 'small', 'medium', 'large' or 'huge'
    Returns: (sizenormal, sizebatch) lists
    """
    sizenormal = []
    sizebatch = []
    for c in case:
        if c == 'small':
            sizenormal += list(chain(range(2, 64, 8), range(64, 256, 32), range(256, 1024, 64)))
            sizebatch += list(chain(zip(range(2, 64, 4), repeat(5000)), zip(range(72, 164, 8), repeat(2500))))
        elif c == 'medium':
            sizenormal += list(chain(range(1024, 2048, 64), range(2048, 4096, 128)))
            sizebatch += list(chain(zip(range(168, 260, 8), repeat(2500)), zip(range(272, 520, 16), repeat(1000))))
        elif c == 'large': 
            sizenormal += list(chain(range(4096, 8192, 256), range(8192, 12800, 512)))
            sizebatch += list(chain(zip(range(544, 1050, 32), repeat(500)), zip(range(1088, 2050, 64), repeat(50))))
        elif c == 'huge': # huge == large for batch cases
            sizenormal += list(chain(range(12800, 23040, 2048), range(23040, 32768, 4096)))
            if 'large' not in case:
                sizebatch += list(chain(zip(range(544, 1050, 32), repeat(500)), zip(range(1088, 2050, 64), repeat(50))))
    return sizenormal, sizebatch


# Benchmark suites
########################################

def potrf_suite(*, suite, precision, sizenormal, sizebatch):
    """
    POTRF tests are run with the given precision and sizes, and for upper and lower cases
    Upper case uses:            | Lower case uses:  
    trsm_upper_left_transposed  | trsm_lower_right_transposed 
    syrk_upper_transposed       | syrk_lower_none
    gemv_transposed             | gemv_none
    <potf2_small_upper>         | <potf2_small_lower>
    """
    fn = 'potrf'
    size = sizenormal
    for s_uplo in ['upper', 'lower']:
        uplo = get_uplo(s_uplo)
        for s in size:
            ld = get_ld(s)
            row = {'name': precision+suite, 'name_test': suite, 'function': fn, 'precision': precision, 'uplo': s_uplo, 'n': s}
            yield (row, s, f'{COMMON_ARGS} -f {fn} -r {precision} --uplo {uplo} -n {s} --lda {ld}')


def potrfBatch_suite(*, suite, precision, sizenormal, sizebatch):
    """
    POTRFBATCH tests are run with the given precision and sizes, and for upper and lower cases
    Upper case uses:            | Lower case uses:  
    trsm_upper_left_transposed  | trsm_lower_right_transposed 
    syrk_upper_transposed       | syrk_lower_none
    gemv_transposed             | gemv_none
    <potf2_small_upper>         | <potf2_small_lower>
    """
    fn = 'potrf_batched'
    size = sizebatch
    for s_uplo in ['upper', 'lower']:
        uplo = get_uplo(s_uplo)
        for s, bc in size:
            ld = get_ld(s)
            row = {'name': precision+suite, 'name_test': suite, 'function': fn, 'precision': precision, 'batch_count': bc, 'uplo': s_uplo, 'n': s}
            yield (row, s, f'{COMMON_ARGS} -f {fn} -r {precision} --batch_count {bc} --uplo {uplo} -n {s} --lda {ld}')


def potrs_suite(*, suite, precision, sizenormal, sizebatch):
    """
    POTRS tests are run with the given precision and sizes, and with 1, n/2 and n right-hand-vectors.
    Tests run upper and lower cases.
    Upper case uses:            | Lower case uses: 
    trsm_upper_left_transposed  | trsm_lower_left_transposed 
    trsm_upper_left_none        | trsm_lower_left_none
    """
    fn = 'potrs'
    size = sizenormal
    for s_uplo in ['upper', 'lower']:
        uplo = get_uplo(s_uplo)
        for s_nrhs in ['one', 'half_n', 'n']:
            for s in size:
                nrhs = get_nrhs(s_nrhs, s)
                ld = get_ld(s)
                row = {'name': precision+suite, 'name_test': suite, 'function': fn, 'precision': precision, 'uplo': s_uplo, 'nrhs': s_nrhs, 'n': s}
                yield (row, s, f'{COMMON_ARGS} -f {fn} -r {precision} --uplo {uplo} --nrhs {nrhs} -n {s} --lda {ld} --ldb {ld}')


def potrsBatch_suite(*, suite, precision, sizenormal, sizebatch):
    """
    POTRSBATCH tests are run with the given precision and sizes, and with 1, n/2 and n right-hand-vectors
    Tests run upper and lower cases.
    Upper case uses:            | Lower case uses: 
    trsm_upper_left_transposed  | trsm_lower_left_transposed 
    trsm_upper_left_none        | trsm_lower_left_none
    """
    fn = 'potrs_batched'
    size = sizebatch
    for s_uplo in ['upper', 'lower']:
        uplo = get_uplo(s_uplo)
        for s_nrhs in ['one']: #['one', 'half_n', 'n'] cuda currently only supports 1 rhs 
            for s, bc in size:
                nrhs = get_nrhs(s_nrhs, s)
                ld = get_ld(s)
                row = {'name': precision+suite, 'name_test': suite, 'function': fn, 'precision': precision, 'batch_count': bc, 'uplo': s_uplo, 'nrhs': s_nrhs, 'n': s}
                yield (row, s, f'{COMMON_ARGS} -f {fn} -r {precision} --batch_count {bc} --uplo {uplo} --nrhs {nrhs} -n {s} --lda {ld} --ldb {ld}')


def potri_suite(*, suite, precision, sizenormal, sizebatch):
    """
    POTRI tests are run with the given precision and sizes, and for upper and lower cases.
    Upper case uses:            | Lower case uses:
    hipsolver_trtri_upper       | hipsolver_trtri_lower
    trmm_upper_right_transposed | trmm_lower_left_transposed 
    """
    fn = 'potri'
    size = sizenormal
    for s_uplo in ['upper', 'lower']:
        uplo = get_uplo(s_uplo)
        for s in size:
            ld = get_ld(s)
            row = {'name': precision+suite, 'name_test': suite, 'function': fn, 'precision': precision, 'uplo': s_uplo, 'n': s}
            yield (row, s, f'{COMMON_ARGS} -f {fn} -r {precision} --uplo {uplo} -n {s} --lda {ld}')


def sytrf_suite(*, suite, precision, sizenormal, sizebatch):
    """
    SYTRF tests are run with the given precision and sizes, and for upper and lower cases.
    Upper or lower test different kernels.
    """
    fn = 'sytrf'
    size = sizenormal
    for s_uplo in ['upper', 'lower']:
        uplo = get_uplo(s_uplo)
        for s in size:
            ld = get_ld(s)
            row = {'name': precision+suite, 'name_test': suite, 'function': fn, 'precision': precision, 'uplo': s_uplo, 'n': s}
            yield (row, s, f'{COMMON_ARGS} -f {fn} -r {precision} --uplo {uplo} -n {s} --lda {ld}')


def sytrs_suite(*, suite, precision, sizenormal, sizebatch):
    """
    SYTRS tests are run with the given precision and sizes, and with 1, n/2 and n right-hand-vectors.
    Tests run upper and lower cases. Upper or lower test different kernels.
    """
    fn = 'sytrs_64'
    size = sizenormal
    for s_uplo in ['upper', 'lower']:
        uplo = get_uplo(s_uplo)
        for s_nrhs in ['one', 'half_n', 'n']:
            for s in size:
                nrhs = get_nrhs(s_nrhs, s)
                ld = get_ld(s)
                row = {'name': precision+suite, 'name_test': suite, 'function': fn, 'precision': precision, 'uplo': s_uplo, 'nrhs': s_nrhs, 'n': s}
                yield (row, s, f'{COMMON_ARGS} -f {fn} -r {precision} --uplo {uplo} --nrhs {nrhs} -n {s} --lda {ld} --ldb {ld}')


def getrf_suite(*, suite, precision, sizenormal, sizebatch):
    """
    GETRF tests are run with the given precision and sizes (only square case)
    """
    fn = 'getrf'
    size = sizenormal
    for s in size:
        ld = get_ld(s)
        row = {'name': precision+suite, 'name_test': suite, 'function': fn, 'precision': precision, 'n': s}
        yield (row, s, f'{COMMON_ARGS} -f {fn} -r {precision} -m {s} --lda {ld}')


#def getrfBatch_suite(*, suite, precision, sizenormal, sizebatch):
#    """
#    GETRFBATCH tests are run with the given precision and sizes (only square case)
#    """
#    fn = 'getrf_batched'
#    size = sizebatch
#    for s, bc in size:
#        ld = get_ld(s)
#        row = {'name': precision+suite, 'name_test': suite, 'function': fn, 'precision': precision, 'batch_count': bc, 'n': s}
#        yield (row, s, f'{COMMON_ARGS} -f {fn} -r {precision} --batch_count {bc} -m {s} --lda {ld}')


def getrfNpvt_suite(*, suite, precision, sizenormal, sizebatch):
    """
    GETRFNPVT tests are run with the given precision and sizes (only square case)
    """
    fn = 'getrf_npvt'
    size = sizenormal
    for s in size:
        ld = get_ld(s)
        row = {'name': precision+suite, 'name_test': suite, 'function': fn, 'precision': precision, 'n': s}
        yield (row, s, f'{COMMON_ARGS} -f {fn} -r {precision} -m {s} --lda {ld}')


#def getrfNpvtBatch_suite(*, suite, precision, sizenormal, sizebatch):
#    """
#    GETRFNPVTBATCH tests are run with the given precision and sizes (only square case)
#    """
#    fn = 'getrf_npvt_batched'
#    size = sizebatch
#    for s, bc in size:
#        ld = get_ld(s)
#        row = {'name': precision+suite, 'name_test': suite, 'function': fn, 'precision': precision, 'batch_count': bc, 'n': s}
#        yield (row, s, f'{COMMON_ARGS} -f {fn} -r {precision} --batch_count {bc} -m {s} --lda {ld}')


def getrs_suite(*, suite, precision, sizenormal, sizebatch):
    """
    GETRS tests are run with the given precision and sizes, and with 1, n/2 and n right-hand-vectors
    The operation argument does not test any new path.
    """
    fn = 'getrs'
    size = sizenormal
    for s_nrhs in ['one', 'half_n', 'n']:
        for s in size:
            nrhs = get_nrhs(s_nrhs, s)
            ld = get_ld(s)
            row = {'name': precision+suite, 'name_test': suite, 'function': fn, 'precision': precision, 'nrhs': s_nrhs, 'n': s}
            yield (row, s, f'{COMMON_ARGS} -f {fn} -r {precision} --nrhs {nrhs} -n {s} --lda {ld} --ldb {ld}')


#def getrsBatch_suite(*, suite, precision, sizenormal, sizebatch):
#    """
#    GETRSBATCH tests are run with the given precision and sizes, and with 1, n/2 and n right-hand-vectors
#    The operation argument does not test any new path.
#    """
#    fn = 'getrs_batched'
#    size = sizebatch
#    for s_nrhs in ['one', 'half_n', 'n']:
#        for s, bc in size:
#            nrhs = get_nrhs(s_nrhs, s)
#            ld = get_ld(s)
#            row = {'name': precision+suite, 'name_test': suite, 'function': fn, 'precision': precision, 'batch_count': bc, 'nrhs': s_nrhs, 'n': s}
#            yield (row, s, f'{COMMON_ARGS} -f {fn} -r {precision} --batch_count {bc} --nrhs {nrhs} -n {s} --lda {ld} --ldb {ld}')


#def getrsNpvt_suite(*, suite, precision, sizenormal, sizebatch):
#    """
#    GETRSNPVT tests are run with the given precision and sizes, and with 1, n/2 and n right-hand-vectors
#    The operation argument does not test any new path.
#    """
#    fn = 'getrs_npvt'
#    size = sizenormal
#    for s_nrhs in ['one', 'half_n', 'n']:
#        for s in size:
#            nrhs = get_nrhs(s_nrhs, s)
#            ld = get_ld(s)
#            row = {'name': precision+suite, 'name_test': suite, 'function': fn, 'precision': precision, 'nrhs': s_nrhs, 'n': s}
#            yield (row, s, f'{COMMON_ARGS} -f {fn} -r {precision} --nrhs {nrhs} -n {s} --lda {ld} --ldb {ld}')


#def getrsNpvtBatch_suite(*, suite, precision, sizenormal, sizebatch):
#    """
#    GETRSNPVTBATCH tests are run with the given precision and sizes, and with 1, n/2 and n right-hand-vectors
#    The operation argument does not test any new path.
#    """
#    fn = 'getrs_npvt_batched'
#    size = sizebatch
#    for s_nrhs in ['one', 'half_n', 'n']:
#        for s, bc in size:
#            nrhs = get_nrhs(s_nrhs, s)
#            ld = get_ld(s)
#            row = {'name': precision+suite, 'name_test': suite, 'function': fn, 'precision': precision, 'batch_count': bc, 'nrhs': s_nrhs, 'n': s}
#            yield (row, s, f'{COMMON_ARGS} -f {fn} -r {precision} --batch_count {bc} --nrhs {nrhs} -n {s} --lda {ld} --ldb {ld}')


#def getriBatch_suite(*, suite, precision, sizenormal, sizebatch):
#    """
#    GETRIBATCH tests are run with the given precision and sizes
#    """
#    fn = 'getri_batched'
#    size = sizebatch
#    for s, bc in size:
#        ld = get_ld(s)
#        row = {'name': precision+suite, 'name_test': suite, 'function': fn, 'precision': precision, 'batch_count': bc, 'n': s}
#        yield (row, s, f'{COMMON_ARGS} -f {fn} -r {precision} --batch_count {bc} -n {s} --lda {ld} --ldc {ld}')


#def trtri_suite(*, suite, precision, sizenormal, sizebatch):
#    """
#    TRTRI tests are run with the given precision and sizes, and for upper and lower cases.
#    Upper case uses:        | Lower case uses:   
#    trtri_upper             | trtri_lower
#    trmv_upper_none         | trmv_lower_none
#    trmm_upper_left_none    | trmm_lower_left_none
#    trsm_upper_right_none   | trsm_lower_right_none
#    <trti2_small_upper>     | <trti2_small_lower>
#    """
#    fn = 'trtri'
#    size = sizenormal
#    for s_uplo in ['upper', 'lower']:
#        uplo = get_uplo(s_uplo)
#        for s in size:
#            ld = get_ld(s)
#            row = {'name': precision+suite, 'name_test': suite, 'function': fn, 'precision': precision, 'uplo': s_uplo, 'n': s}
#            yield (row, s, f'{COMMON_ARGS} -f {fn} -r {precision} --uplo {uplo} -n {s} --lda {ld}')


def geqrf_suite(*, suite, precision, sizenormal, sizebatch):
    """
    GEQRF tests are run, for the given precision and number of rows,
    with 160 columns and also for the square case (#rows = #columns)
    geqrf uses: 
    larft_forward_column
    larfb_forward_column_letf_transposed
    """
    fn = 'geqrf'
    size = sizenormal
    for s_shape in ['square', 'skinny']:
        for s in size:
            m,n = get_mn(s_shape, s, 'normal')
            ld = get_ld(s)
            if m >= n:
                row = {'name': precision+suite, 'name_test': suite, 'function': fn, 'precision': precision, 'shape': s_shape, 'n': s}
                yield (row, s, f'{COMMON_ARGS} -f {fn} -r {precision} -n {n} -m {m} --lda {ld}')


#def geqrfBatch_suite(*, suite, precision, sizenormal, sizebatch):
#    """
#    GEQRFBATCH tests are run, for the given precision and number of rows,
#    with 26 columns and also for the square case (#rows = #columns)
#    geqrf uses: 
#    larft_forward_column
#    larfb_forward_column_letf_transposed
#    """
#    fn = 'geqrf_batched'
#    size = sizebatch
#    for s_shape in ['square', 'skinny']:
#        for s, bc in size:
#            m,n = get_mn(s_shape, s, 'batched')
#            ld = get_ld(s)
#            if m >= n:
#                row = {'name': precision+suite, 'name_test': suite, 'function': fn, 'precision': precision, 'batch_count': bc, 'shape': s_shape, 'n': s}
#                yield (row, s, f'{COMMON_ARGS} -f {fn} -r {precision} --batch_count {bc} -n {n} -m {m} --lda {ld}')


def gels_suite(*, suite, precision, sizenormal, sizebatch):
    """
    GELS tests are run, for the given precision and number of rows (columns), with 160 columns (rows) and with 1, 
    n/2 and n right-hand-vectors. We want the overdetermined case m >= n, but also the underdetermined m < n 
    to actually test gelqf and ormlq/unmlq. 
    gelqf uses:
    larft_forward_row
    larfb_forward_row_right_none
    ormlq uses:           
    larft_forward_row           
    larfb_forward_row_left_none
    """
    fn = 'gels'
    size = sizenormal
    for s_shape in ['overdet']: #['overdet', 'underdet'] underdetermined systems are not currently supported in cuda
        for s_nrhs in ['one', 'half_n', 'n']:
            for s in size:
                nrhs = get_nrhs(s_nrhs, s)
                m,n = get_mn(s_shape, s, 'normal')
                ld_b = get_ld(s)
                ld_a = get_ld(m)
                ld_x = get_ld(n)
                if (s_shape == 'overdet' and m >= n) or (s_shape == 'underdet' and m < n):
                    row = {'name': precision+suite, 'name_test': suite, 'function': fn, 'precision': precision, 'shape': s_shape, 'nrhs': s_nrhs, 'n': s}
                    yield (row, s, f'{COMMON_ARGS} -f {fn} -r {precision} -m {m} -n {n} --nrhs {nrhs} --lda {ld_a} --ldb {ld_b} --ldx {ld_x}')


#def gelsBatch_suite(*, suite, precision, sizenormal, sizebatch):
#    """
#    GELSBATCH tests are run, for the given precision and number of rows (columns), with 160 columns (rows) and with 1,
#    n/2 and n right-hand-vectors. We want the overdetermined case m >= n, but also the underdetermined m < n
#    to actually test gelqf and ormlq/unmlq.
#    gelqf uses:
#    larft_forward_row
#    larfb_forward_row_right_none
#    ormlq uses:
#    larft_forward_row
#    larfb_forward_row_left_none
#    """
#    fn = 'gels_batched'
#    size = sizebatch
#    for s_shape in ['overdet']: #['overdet', 'underdet'] underdetermined systems are not currently supported in cuda
#        for s_nrhs in ['one', 'half_n', 'n']:
#            for s in size:
#                nrhs = get_nrhs(s_nrhs, s)
#                m,n = get_mn(s_shape, s, 'batched')
#                ld_b = get_ld(s)
#                ld_a = get_ld(m)
#                ld_x = get_ld(n)
#                if (s_shape == 'overdet' and m >= n) or (s_shape == 'underdet' and m < n):
#                    row = {'name': precision+suite, 'name_test': suite, 'function': fn, 'precision': precision, 'batch_count': bc, 'shape': s_shape, 'nrhs': s_nrhs, 'n': s}
#                    yield (row, s, f'{COMMON_ARGS} -f {fn} -r {precision} --batch_count {bc} -m {m} -n {n} --nrhs {nrhs} --lda {ld_a} --ldb {ld_b} --ldx {ld_x}')


def xxgqr_suite(*, suite, precision, sizenormal, sizebatch):
    """
    XXGQR (ORGQR or UNGQR) tests are run, for the given precision and number of rows,
    with 160 columns and also for the square case (#rows = #columns)
    orgqr uses:
    larft_forward_column
    larfb_forward_column_left_none
    """
    fn = 'orgqr' if precision == 's' or precision == 'd' else 'ungqr'
    size=sizenormal
    for nc in [0, 160]:
        if nc == 0: nn = 'sq'
        else: nn = nc
        for s in size:
            if s < 4000: ld = s + 1
            else: ld = s + 64
            if nc == 0: n = s
            else: n = nc
            if s >= n:
                row = {'name': precision+suite, 'name_test': suite, 'function': fn, 'precision': precision, 'cols': nn, 'n': s}
                yield (row, s, f'{COMMON_ARGS} -f {fn} -r {precision} -n {n} -m {s} --lda {ld}')


def xxmqr_suite(*, suite, precision, sizenormal, sizebatch):
    """
    XXMQR (ORMQR or UNMQR) tests are run with the given precision and sizes (only square case), from the right.
    Tests run ops = {transposed, none} cases.
    ormqr uses:             
    larft_forward_column
    larfb_forward_column_right_<ops>
    """
    fn = 'ormqr' if precision == 's' or precision == 'd' else 'unmqr'
    tr = 'T' if precision == 's' or precision == 'd' else 'C'
    size = sizenormal
    for ops in ['none', 'trans']:
        if ops == 'none': op = 'N'
        else: op = tr
        for s in size:
            if s < 4000: ld = s + 1
            else: ld = s + 64
            row = {'name': precision+suite, 'name_test': suite, 'function': fn, 'precision': precision, 'trans': ops, 'n': s}
            yield (row, s, f'{COMMON_ARGS} -f {fn} -r {precision} --side R --trans {op} -n {s} --lda {ld} --ldc {ld}')


#def larft_suite(*, suite, precision, sizenormal, sizebatch):
#    """
#    LARFT tests are run with the given precision and sizes, backward direction, and column-wise and
#    row-wise. Tests use 1, n/2 and n Householder vectors.
#    """
#    fn = 'larft'
#    size = sizenormal
#    for stor in ['colwise']: #['colwise', 'rowwise'] rowwise is not currently supported in cuda.
#        if stor == 'colwise': sto = 'C'
#        else: sto = 'R'
#        for nk in ['one', 'half_n', 'n']:
#            k = 1
#            for s in size:
#                if nk == 'half_n': k = s//2
#                elif nk == 'n': k = s
#                if s < 4000: ld1 = s + 1
#                else: ld1 = s + 64
#                if k < 4000: ld2 = k + 1
#                else: ld2 = k + 64
#                row = {'name': precision+suite, 'name_test': suite, 'function': fn, 'precision': precision, 'storev': stor, 'nk': nk, 'n': s}
#                yield (row, s, f'{COMMON_ARGS} -f {fn} -r {precision} --direct B --storev {sto} -k {k} -n {s} --ldv {ld1} --ldt {ld2}')


def xxtrd_suite(*, suite, precision, sizenormal, sizebatch):
    """
    XXTRD (SYTRD or HETRD) tests are run with the given precision and sizes.
    Tests run upper and lower cases. Upper or lower test different kernels.
    """
    fn = 'sytrd' if precision == 's' or precision == 'd' else 'hetrd'
    size = sizenormal
    for s_uplo in ['upper', 'lower']:
        if s_uplo == 'upper': upl = 'U'
        else: upl = 'L'
        for s in size:
            if s < 4000: ld = s + 1
            else: ld = s + 64
            row = {'name': precision+suite, 'name_test': suite, 'function': fn, 'precision': precision, 'uplo': s_uplo, 'n': s}
            yield (row, s, f'{COMMON_ARGS} -f {fn} -r {precision} --uplo {uplo} -n {s} --lda {ld}')


def xxgtr_suite(*, suite, precision, sizenormal, sizebatch):
    """
    XXGTR (ORGTR or UNGTR) tests are run with the given precision and sizes.
    Always upper to actually use orgql/ungql.
    orgql uses:
    larft_backward_column
    larfb_backward_column_left_none
    """
    fn = 'orgtr' if precision == 's' or precision == 'd' else 'ungtr'
    size = sizenormal
    for s in size:
        if s < 4000: ld = s + 1
        else: ld = s + 64
        row = {'name': precision+suite, 'name_test': suite, 'function': fn, 'precision': precision, 'n': s}
        yield (row, s, f'{COMMON_ARGS} -f {fn} -r {precision} --uplo U -n {s} --lda {ld}')


def xxmtr_suite(*, suite, precision, sizenormal, sizebatch):
    """
    XXMTR (ORMTR or UNMTR) tests are run with the given precision and sizes (only square case).
    Always upper to actually use ormql/unmql.
    Tests run side = left with ops = transposed, 
    and side = right with ops = {none, transposed} cases.
    ormql uses:
    larft_backward_column
    larfb_backward_column_<side>_<ops>    
    """
    fn = 'ormtr' if precision == 's' or precision == 'd' else 'unmtr'
    tr = 'T' if precision == 's' or precision == 'd' else 'C'
    size = sizenormal
    for slr in ['left', 'right']:
        if slr == 'left': lr = 'L'
        else: lr = 'R'
        for ops in ['none', 'trans']:
            if ops == 'none': op = 'N'
            else: op = tr
            for s in size:
                if s < 4000: ld = s + 1
                else: ld = s + 64
                row = {'name': precision+suite, 'name_test': suite, 'function': fn, 'precision': precision, 'side': slr, 'trans': ops, 'n': s}
                if slr == 'right':
                    yield (row, s, f'{COMMON_ARGS} -f {fn} -r {precision} --uplo U --side {lr} --trans {op} -n {s} --lda {ld} --ldc {ld}')
                if slr == 'left' and ops == 'trans':
                    yield (row, s, f'{COMMON_ARGS} -f {fn} -r {precision} --uplo U --side {lr} --trans {op} -m {s} --lda {ld} --ldc {ld}')


def gebrd_suite(*, suite, precision, sizenormal, sizebatch):
    """
    GEBRD tests are run with the given precision and sizes (only square case)
    """
    fn = 'gebrd'
    size = sizenormal
    for s in size:
        if s < 4000: ld = s + 1
        else: ld = s + 64
        row = {'name': precision+suite, 'name_test': suite, 'function': fn, 'precision': precision, 'n': s}
        yield (row, s, f'{COMMON_ARGS} -f {fn} -r {precision} -m {s} --lda {ld}')


def xxgbr_suite(*, suite, precision, sizenormal, sizebatch):
    """
    XXGBR (ORGBR or UNGBR) tests are run with the given precision and sizes (only square case). 
    Always form the right (row-wise) to actually test orglq/unglq.
    orglq uses:
    larft_forward_row
    larfb_forward_row_right_transposed
    """
    fn = 'orgbr' if precision == 's' or precision == 'd' else 'ungbr'
    size = sizenormal
    for s in size:
        if s < 4000: ld = s + 1
        else: ld = s + 64
        row = {'name': precision+suite, 'name_test': suite, 'function': fn, 'precision': precision, 'n': s}
        yield (row, s, f'{COMMON_ARGS} -f {fn} -r {precision} --side R -m {s} --lda {ld}')


#def stedc_suite(*, suite, precision, sizenormal, sizebatch):
#    """
#    STEDC tests are run, for the given precision and sizes, with vectors and without vectors
#    """
#    fn = 'stedc' 
#    size = sizenormal
#    for v in ['I', 'N']:
#        if v == 'I': vv = 'vect'
#        else: vv = 'novect'
#        for s in size:
#            if s < 4000: ld = s + 1
#            else: ld = s + 64
#            row = {'name': precision+suite, 'name_test': suite, 'function': fn, 'precision': precision, 'evect': vv, 'n': s}
#            yield (row, s, f'{COMMON_ARGS} -f {fn} -r {precision} --compz {v} -n {s} --ldc {ld}')


def xxevd_suite(*, suite, precision, sizenormal, sizebatch):
    """
    XXEVD (SYEVD or HEEVD) tests are run, for the given precision and sizes, with vectors and without vectors. Upper case.
    """
    fn = 'syevd' if precision == 's' or precision == 'd' else 'heevd'
    size = sizenormal
    for v in ['V', 'N']:
        if v == 'V': vv = 'vect'
        else: vv = 'novect'
        for s in size:
            if s < 4000: ld = s + 1
            else: ld = s + 64
            row = {'name': precision+suite, 'name_test': suite, 'function': fn, 'precision': precision, 'evect': vv, 'n': s}
            yield (row, s, f'{COMMON_ARGS} -f {fn} -r {precision} --jobz {v} -n {s} --lda {ld}')


def xxgvd_suite(*, suite, precision, sizenormal, sizebatch):
    """
    XXGVD (SYGVD or HEGVD) tests are run, for the given precision and sizes, with vectors.
    Tests run upper and lower case with AX and BAX forms. 
    """
    fn = 'sygvd' if precision == 's' or precision == 'd' else 'hegvd'
    size = sizenormal
    for s_uplo in ['upper', 'lower']:
        if s_uplo == 'upper': upl = 'U'
        else: upl = 'L'
        for ty in ['AX', 'BAX']:
            if ty == 'AX': ity = 1
            else: ity = 3
            for s in size:
                if s < 4000: ld = s + 1
                else: ld = s + 64
                row = {'name': precision+suite, 'name_test': suite, 'function': fn, 'precision': precision, 'uplo': s_uplo, 'type': ty, 'n': s}
                yield (row, s, f'{COMMON_ARGS} -f {fn} -r {precision} --jobz V --uplo {uplo} --itype {ity} -n {s} --lda {ld} --ldb {ld}')


def xxevBatch_suite(*, suite, precision, sizenormal, sizebatch):
    """
    XXEVBATCH (SYEVBATCH or HEEVBATCH) tests are run, for the given precision and sizes, with vectors and without vectors
    """
    fn = 'syev_batched_64' if precision == 's' or precision == 'd' else 'heev_batched_64'
    size = sizebatch
    for v in ['V', 'N']:
        if v == 'V': vv = 'vect'
        else: vv = 'novect'
        for s, bc in size:
            if s < 4000: ld = s + 1
            else: ld = s + 64
            row = {'name': precision+suite, 'name_test': suite, 'function': fn, 'precision': precision, 'batch_count': bc, 'evect': vv, 'n': s}
            yield (row, s, f'{COMMON_ARGS} -f {fn} -r {precision} --batch_count {bc} --jobz {v} -n {s} --lda {ld}')


def xxevdx_suite(*, suite, precision, sizenormal, sizebatch):
    """
    XXEVDX (SYEVDX or HEEVDX) tests are run, for the given precision and sizes, with vectors and 
    computing 20 and 60 percent of the eigenvalues. Upper case.
    """
    fn = 'syevdx' if precision == 's' or precision == 'd' else 'heevdx'
    size=sizenormal
    for per in [20, 60]:
        for s in size:
            p = int(s * per / 100)
            if p == 0: p = 1
            if s < 4000: ld = s + 1
            else: ld = s + 64
            row = {'name': precision+suite, 'name_test': suite, 'function': fn, 'precision': precision, 'range': per, 'n': s}
            yield (row, s, f'{COMMON_ARGS} -f {fn} -r {precision} --jobz V --range I --il 1 --iu {p} -n {s} --lda {ld}')


def xxgvdx_suite(*, suite, precision, sizenormal, sizebatch):
    """
    XXGVDX (SYGVDX or HEGVDX) tests are run, for the given precision and sizes, with vectors and 
    computing 20 and 60 percent of the eigenvalues. Upper case, AX form. 
    """
    fn = 'sygvdx' if precision == 's' or precision == 'd' else 'hegvdx'
    size=sizenormal
    for per in [20, 60]:
        for s in size:
            p = int(s * per / 100)
            if p == 0: p = 1
            if s < 4000: ld = s + 1
            else: ld = s + 64
            row = {'name': precision+suite, 'name_test': suite, 'function': fn, 'precision': precision, 'range': per, 'n': s}
            yield (row, s, f'{COMMON_ARGS} -f {fn} -r {precision} --jobz V --range I --il 1 --iu {p} -n {s} --lda {ld}')


def xxevj_suite(*, suite, precision, sizenormal, sizebatch):
    """
    XXEVJ (SYEVJ or HEEVJ) tests are run, for the given precision and sizes, with vectors and without vectors. Upper case.
    """
    fn = 'syevj' if precision == 's' or precision == 'd' else 'heevj'
    size = sizenormal
    for v in ['V', 'N']:
        if v == 'V': vv = 'vect'
        else: vv = 'novect'
        for s in size:
            if s < 4000: ld = s + 1
            else: ld = s + 64
            row = {'name': precision+suite, 'name_test': suite, 'function': fn, 'precision': precision, 'evect': vv, 'n': s}
            yield (row, s, f'{COMMON_ARGS} -f {fn} -r {precision} --jobz {v} -n {s} --lda {ld}')


def xxgvj_suite(*, suite, precision, sizenormal, sizebatch):
    """
    XXGVJ (SYGVJ or HEGVJ) tests are run, for the given precision and sizes, with vectors. Upper case, AX form.
    """
    fn = 'sygvj' if precision == 's' or precision == 'd' else 'hegvj'
    size = sizenormal
    for s in size:
        if s < 4000: ld = s + 1
        else: ld = s + 64
        row = {'name': precision+suite, 'name_test': suite, 'function': fn, 'precision': precision, 'n': s}
        yield (row, s, f'{COMMON_ARGS} -f {fn} -r {precision} --jobz V -n {s} --lda {ld}')


def xxevjBatch_suite(*, suite, precision, sizenormal, sizebatch):
    """
    XXEVJBATCH (SYEVJBATCH or HEEVJBATCH) tests are run, for the given precision and sizes, with vectors and without vectors. Upper case.
    """
    fn = 'syevj_batched' if precision == 's' or precision == 'd' else 'heevj_batched'
    size = sizebatch
    for v in ['V', 'N']:
        if v == 'V': vv = 'vect'
        else: vv = 'novect'
        for s, bc in size:
            if s < 4000: ld = s + 1
            else: ld = s + 64
            row = {'name': precision+suite, 'name_test': suite, 'function': fn, 'precision': precision, 'batch_count': bc, 'evect': vv, 'n': s}
            yield (row, s, f'{COMMON_ARGS} -f {fn} -r {precision} --batch_count {bc} --jobz {v} -n {s} --lda {ld}')


def gesvd_suite(*, suite, precision, sizenormal, sizebatch):
    """
    GESVD tests are run, for the given precision and sizes, with vectors and without vectors (only square case).
    """
    fn = 'gesvd'
    size = sizenormal
    for v in ['S', 'N']:
        if v == 'S': vv = 'vect'
        else: vv = 'novect'
        for s in size:
            if s < 4000: ld = s + 1
            else: ld = s + 64
            row = {'name': precision+suite, 'name_test': suite, 'function': fn, 'precision': precision, 'svect': vv, 'n': s}
            yield (row, s, f'{COMMON_ARGS} -f {fn} -r {precision} --jobu {v} --jobv {v} -m {s} --lda {ld} --ldu {ld} --ldv {ld}')


def gesvdj_suite(*, suite, precision, sizenormal, sizebatch):
    """
    GESVDJ tests are run, for the given precision and sizes, with vectors and without vectors (only square case).
    """
    fn = 'gesvdj'
    size = sizenormal
    for v in ['V', 'N']:
        if v == 'V': vv = 'vect'
        else: vv = 'novect'
        for s in size:
            if s < 4000: ld = s + 1
            else: ld = s + 64
            row = {'name': precision+suite, 'name_test': suite, 'function': fn, 'precision': precision, 'svect': vv, 'n': s}
            yield (row, s, f'{COMMON_ARGS} -f {fn} -r {precision} --jobz {v} -m {s} --lda {ld} --ldu {ld} --ldv {ld}')


def gesvdjBatch_suite(*, suite, precision, sizenormal, sizebatch):
    """
    GESVDJBATCH tests are run, for the given precision and sizes, with vectors and without vectors (only square case).
    """
    fn = 'gesvdj_batched'
    size = sizebatch
    for v in ['V', 'N']:
        if v == 'V': vv = 'vect'
        else: vv = 'novect'
        for s, bc in size:
            if s < 33: # only sizes n <= 32 are currently supportted by cuda
                if s < 4000: ld = s + 1
                else: ld = s + 64
                row = {'name': precision+suite, 'name_test': suite, 'function': fn, 'precision': precision, 'batch_count': bc, 'evect': vv, 'n': s}
                yield (row, s, f'{COMMON_ARGS} -f {fn} -r {precision} --batch_count {bc} --jobz {v} -m {s} --lda {ld} --ldu {ld} --ldv {ld}')


# Registry of all available benchmark suites
#####################################################
#### TODO: add back missing functions when they become available in hipsolver ####

SUITES = {
    # Symmetric linear systems
    'potrf': potrf_suite,
    'potrfBatch': potrfBatch_suite,
    'potrs': potrs_suite,
    'potrsBatch': potrsBatch_suite,
    'potri': potri_suite,
    'sytrf': sytrf_suite,
    'sytrs': sytrs_suite,                       
    
    # General linear systems
    'getrf': getrf_suite,
#    'getrfBatch': getrfBatch_suite,
    'getrfNpvt': getrfNpvt_suite,
#    'getrfNpvtBatch': getrfNpvtBatch_suite,
    'getrs': getrs_suite,
#    'getrsBatch': getrsBatch_suite,
#    'getrsNpvt': getrsNpvt_suite,               
#    'getrsNpvtBatch': getrsNpvtBatch_suite,     
#    'getriBatch': getriBatch_suite,
#    'trtri': trtri_suite,

    # Over-determined linear systems (least-squares)
    'geqrf': geqrf_suite,
#    'geqrfBatch': geqrfBatch_suite,
    'gels': gels_suite,                          
#    'gelsBatch': gelsBatch_suite,               
    'xxgqr': xxgqr_suite,
    'xxmqr': xxmqr_suite,
#    'larft': larft_suite,

    # Matrix reductions (tridiagonalization, bidiagonalization)
    'xxtrd': xxtrd_suite, 
    'xxgtr': xxgtr_suite,           
    'xxmtr': xxmtr_suite,           
    'gebrd': gebrd_suite,
    'xxgbr': xxgbr_suite,           
 
    # Symmetric Eigenvalue problem
#    'stedc': stedc_suite,
    'xxevd': xxevd_suite,
    'xxgvd': xxgvd_suite,
    'xxevBatch': xxevBatch_suite,
    'xxevdx': xxevdx_suite,
    'xxgvdx': xxgvdx_suite,
    'xxevj': xxevj_suite,
    'xxgvj': xxgvj_suite,
    'xxevjBatch': xxevjBatch_suite,

    # Singular value decomposition
    'gesvd': gesvd_suite,
    'gesvdj': gesvdj_suite,
    'gesvdjBatch': gesvdjBatch_suite,
}
