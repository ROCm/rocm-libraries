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
Shared module containing benchmark suite definitions for rocSOLVER.

This module provides:
- Test suite generator functions for various rocSOLVER routines
- Common benchmark parameters
- Size configurations for different test cases
"""

from itertools import chain, repeat

# Common benchmark arguments - always do 5 iterations in perf mode
COMMON_ARGS = '--iters 5 --perf 1'


# Common helpers
######################################

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


def get_ops(s_ops, precision):
    """
    Gets the operation type -transposed or none (default is none)
    """
    tr = 'T' if precision == 's' or precision == 'd' else 'C'
    if s_ops == 'trans': ops = tr
    else: ops = 'N'
    return ops


def get_storev(s_storev):
    """
    Gets storev: column-wise or row-wise (default is colwise)
    """
    if s_storev == 'rowwise': storev = 'R'
    else: storev = 'C'
    return storev


def get_side(s_side):
    """
    Gets side: left or right (default is right)
    """
    if s_side == 'left': side = 'L'
    else: side = 'R'
    return side


def get_evect(s_evect, matvect):
    """
    Gets evect: with-vectors or without-vectors (default is with-vectors original matrix)
    """
    if s_evect == 'novect': evect = 'N'
    else: evect = 'I' if matvect == 'tridiag' else 'V'
    return evect


def get_itype(s_itype):
    """
    Gets itype: AX, ABX, BAX (default is AX)
    """
    if s_itype == 'ABX': itype = 2
    elif s_itype == 'BAX': itype = 3
    else: itype = 1
    return itype


def get_upperid(per, s):
    """
    Gets upper index for the eigenvalue range depending on the percentage
    """
    iu = int(s * per / 100)
    if iu == 0: iu = 1
    return iu


def get_alg(s_alg):
    """
    Gets algorithm mode: GPU or hybrid (default is GPU)
    """
    if s_alg == 'hybrid': alg = 1
    else: alg = 0
    return alg


def get_size_configurations(case, mode):
    """
    Get size configurations for normal and batched tests.
    Args: a list with one or more of 'small', 'medium', 'large' or 'huge' and the mode (default is normal)
    Returns: sizenormal or sizebatch
    """
    size = []
    for c in case:
        if c == 'small':
            if mode == 'batched':
                size += list(chain(zip(range(2, 64, 4), repeat(5000)), zip(range(72, 164, 8), repeat(2500))))
            else:
                size += list(chain(range(2, 64, 8), range(64, 256, 32), range(256, 1024, 64)))
        elif c == 'medium':
            if mode == 'batched':
                size += list(chain(zip(range(168, 260, 8), repeat(2500)), zip(range(272, 520, 16), repeat(1000))))
            else:
                size += list(chain(range(1024, 2048, 64), range(2048, 4096, 128)))
        elif c == 'large':
            if mode == 'batched':
                size += list(chain(zip(range(544, 1050, 32), repeat(500)), zip(range(1088, 2050, 64), repeat(50))))
            else:
                size += list(chain(range(4096, 8192, 256), range(8192, 12800, 512)))
        elif c == 'huge': # huge == large for batch cases
            if mode == 'batched' and 'large' not in case:
                size += list(chain(zip(range(544, 1050, 32), repeat(500)), zip(range(1088, 2050, 64), repeat(50))))
            else:
                size += list(chain(range(12800, 23040, 2048), range(23040, 32768, 4096)))
    return size


# Benchmark suites
########################################

def potrf_suite(*, suite, precision, case):
    """
    POTRF tests are run with the given precision and sizes, and for upper and lower cases
    Upper case uses:            | Lower case uses:  
    trsm_upper_left_transposed  | trsm_lower_right_transposed 
    syrk_upper_transposed       | syrk_lower_none
    gemv_transposed             | gemv_none
    <potf2_small_upper>         | <potf2_small_lower>
    """
    fn = 'potrf'
    mode = 'normal'
    size = get_size_configurations(case, mode)
    for s_uplo in ['upper', 'lower']:
        uplo = get_uplo(s_uplo)
        for s in size:
            ld = get_ld(s)
            row = {'name': precision+suite, 'name_test': suite, 'function': fn, 'precision': precision, 'uplo': s_uplo, 'n': s}
            yield (row, s, f'{COMMON_ARGS} -f {fn} -r {precision} --uplo {uplo} -n {s} --lda {ld}')


def potrfBatch_suite(*, suite, precision, case):
    """
    POTRFBATCH tests are run with the given precision and sizes, and for upper and lower cases
    Upper case uses:            | Lower case uses:  
    trsm_upper_left_transposed  | trsm_lower_right_transposed 
    syrk_upper_transposed       | syrk_lower_none
    gemv_transposed             | gemv_none
    <potf2_small_upper>         | <potf2_small_lower>
    """
    fn = 'potrf_batched'
    mode = 'batched'
    size = get_size_configurations(case, mode)
    for s_uplo in ['upper', 'lower']:
        uplo = get_uplo(s_uplo)
        for s, bc in size:
            ld = get_ld(s)
            row = {'name': precision+suite, 'name_test': suite, 'function': fn, 'precision': precision, 'batch_count': bc, 'uplo': s_uplo, 'n': s}
            yield (row, s, f'{COMMON_ARGS} -f {fn} -r {precision} --batch_count {bc} --uplo {uplo} -n {s} --lda {ld}')


def potrs_suite(*, suite, precision, case):
    """
    POTRS tests are run with the given precision and sizes, and with 1, n/2 and n right-hand-vectors.
    Tests run upper and lower cases.
    Upper case uses:            | Lower case uses: 
    trsm_upper_left_transposed  | trsm_lower_left_transposed 
    trsm_upper_left_none        | trsm_lower_left_none
    """
    fn = 'potrs'
    mode = 'normal'
    size = get_size_configurations(case, mode)
    for s_uplo in ['upper', 'lower']:
        uplo = get_uplo(s_uplo)
        for s_nrhs in ['one', 'half_n', 'n']:
            for s in size:
                nrhs = get_nrhs(s_nrhs, s)
                ld = get_ld(s)
                row = {'name': precision+suite, 'name_test': suite, 'function': fn, 'precision': precision, 'uplo': s_uplo, 'nrhs': s_nrhs, 'n': s}
                yield (row, s, f'{COMMON_ARGS} -f {fn} -r {precision} --uplo {uplo} --nrhs {nrhs} -n {s} --lda {ld} --ldb {ld}')


def potrsBatch_suite(*, suite, precision, case):
    """
    POTRSBATCH tests are run with the given precision and sizes, and with 1, n/2 and n right-hand-vectors
    Tests run upper and lower cases.
    Upper case uses:            | Lower case uses: 
    trsm_upper_left_transposed  | trsm_lower_left_transposed 
    trsm_upper_left_none        | trsm_lower_left_none
    """
    fn = 'potrs_batched'
    mode = 'batched'
    size = get_size_configurations(case, mode)
    for s_uplo in ['upper', 'lower']:
        uplo = get_uplo(s_uplo)
        for s_nrhs in ['one', 'half_n', 'n']:
            for s, bc in size:
                nrhs = get_nrhs(s_nrhs, s)
                ld = get_ld(s)
                row = {'name': precision+suite, 'name_test': suite, 'function': fn, 'precision': precision, 'batch_count': bc, 'uplo': s_uplo, 'nrhs': s_nrhs, 'n': s}
                yield (row, s, f'{COMMON_ARGS} -f {fn} -r {precision} --batch_count {bc} --uplo {uplo} --nrhs {nrhs} -n {s} --lda {ld} --ldb {ld}')


def potri_suite(*, suite, precision, case):
    """
    POTRI tests are run with the given precision and sizes, and for upper and lower cases.
    Upper case uses:            | Lower case uses:
    rocsolver_trtri_upper       | rocsolver_trtri_lower
    trmm_upper_right_transposed | trmm_lower_left_transposed 
    """
    fn = 'potri'
    mode = 'normal'
    size = get_size_configurations(case, mode)
    for s_uplo in ['upper', 'lower']:
        uplo = get_uplo(s_uplo)
        for s in size:
            ld = get_ld(s)
            row = {'name': precision+suite, 'name_test': suite, 'function': fn, 'precision': precision, 'uplo': s_uplo, 'n': s}
            yield (row, s, f'{COMMON_ARGS} -f {fn} -r {precision} --uplo {uplo} -n {s} --lda {ld}')


def sytrf_suite(*, suite, precision, case):
    """
    SYTRF tests are run with the given precision and sizes, and for upper and lower cases.
    Upper or lower test different kernels.
    """
    fn = 'sytrf'
    mode = 'normal'
    size = get_size_configurations(case, mode)
    for s_uplo in ['upper', 'lower']:
        uplo = get_uplo(s_uplo)
        for s in size:
            ld = get_ld(s)
            row = {'name': precision+suite, 'name_test': suite, 'function': fn, 'precision': precision, 'uplo': s_uplo, 'n': s}
            yield (row, s, f'{COMMON_ARGS} -f {fn} -r {precision} --uplo {uplo} -n {s} --lda {ld}')


def sytrs_suite(*, suite, precision, case):
    """
    SYTRS tests are run with the given precision and sizes, and with 1, n/2 and n right-hand-vectors.
    Tests run upper and lower cases. Upper or lower test different kernels.
    """
    fn = 'sytrs'
    mode = 'normal'
    size = get_size_configurations(case, mode)
    for s_uplo in ['upper', 'lower']:
        uplo = get_uplo(s_uplo)
        for s_nrhs in ['one', 'half_n', 'n']:
            for s in size:
                nrhs = get_nrhs(s_nrhs, s)
                ld = get_ld(s)
                row = {'name': precision+suite, 'name_test': suite, 'function': fn, 'precision': precision, 'uplo': s_uplo, 'nrhs': s_nrhs, 'n': s}
                yield (row, s, f'{COMMON_ARGS} -f {fn} -r {precision} --uplo {uplo} --nrhs {nrhs} -n {s} --lda {ld} --ldb {ld}')


def getrf_suite(*, suite, precision, case):
    """
    GETRF tests are run with the given precision and sizes (only square case)
    """
    fn = 'getrf'
    mode = 'normal'
    size = get_size_configurations(case, mode)
    for s in size:
        ld = get_ld(s)
        row = {'name': precision+suite, 'name_test': suite, 'function': fn, 'precision': precision, 'n': s}
        yield (row, s, f'{COMMON_ARGS} -f {fn} -r {precision} -m {s} --lda {ld}')


def getrfBatch_suite(*, suite, precision, case):
    """
    GETRFBATCH tests are run with the given precision and sizes (only square case)
    """
    fn = 'getrf_batched'
    mode = 'batched'
    size = get_size_configurations(case, mode)
    for s, bc in size:
        ld = get_ld(s)
        row = {'name': precision+suite, 'name_test': suite, 'function': fn, 'precision': precision, 'batch_count': bc, 'n': s}
        yield (row, s, f'{COMMON_ARGS} -f {fn} -r {precision} --batch_count {bc} -m {s} --lda {ld}')


def getrfNpvt_suite(*, suite, precision, case):
    """
    GETRFNPVT tests are run with the given precision and sizes (only square case)
    """
    fn = 'getrf_npvt'
    mode = 'normal'
    size = get_size_configurations(case, mode)
    for s in size:
        ld = get_ld(s)
        row = {'name': precision+suite, 'name_test': suite, 'function': fn, 'precision': precision, 'n': s}
        yield (row, s, f'{COMMON_ARGS} -f {fn} -r {precision} -m {s} --lda {ld}')


def getrfNpvtBatch_suite(*, suite, precision, case):
    """
    GETRFNPVTBATCH tests are run with the given precision and sizes (only square case)
    """
    fn = 'getrf_npvt_batched'
    mode = 'batched'
    size = get_size_configurations(case, mode)
    for s, bc in size:
        ld = get_ld(s)
        row = {'name': precision+suite, 'name_test': suite, 'function': fn, 'precision': precision, 'batch_count': bc, 'n': s}
        yield (row, s, f'{COMMON_ARGS} -f {fn} -r {precision} --batch_count {bc} -m {s} --lda {ld}')


def getrs_suite(*, suite, precision, case):
    """
    GETRS tests are run with the given precision and sizes, and with 1, n/2 and n right-hand-vectors
    The operation argument does not test any new path.
    """
    fn = 'getrs'
    mode = 'normal'
    size = get_size_configurations(case, mode)
    for s_nrhs in ['one', 'half_n', 'n']:
        for s in size:
            nrhs = get_nrhs(s_nrhs, s)
            ld = get_ld(s)
            row = {'name': precision+suite, 'name_test': suite, 'function': fn, 'precision': precision, 'nrhs': s_nrhs, 'n': s}
            yield (row, s, f'{COMMON_ARGS} -f {fn} -r {precision} --nrhs {nrhs} -n {s} --lda {ld} --ldb {ld}')


def getrsBatch_suite(*, suite, precision, case):
    """
    GETRSBATCH tests are run with the given precision and sizes, and with 1, n/2 and n right-hand-vectors
    The operation argument does not test any new path.
    """
    fn = 'getrs_batched'
    mode = 'batched'
    size = get_size_configurations(case, mode)
    for s_nrhs in ['one', 'half_n', 'n']:
        for s, bc in size:
            nrhs = get_nrhs(s_nrhs, s)
            ld = get_ld(s)
            row = {'name': precision+suite, 'name_test': suite, 'function': fn, 'precision': precision, 'batch_count': bc, 'nrhs': s_nrhs, 'n': s}
            yield (row, s, f'{COMMON_ARGS} -f {fn} -r {precision} --batch_count {bc} --nrhs {nrhs} -n {s} --lda {ld} --ldb {ld}')


def getrsNpvt_suite(*, suite, precision, case):
    """
    GETRSNPVT tests are run with the given precision and sizes, and with 1, n/2 and n right-hand-vectors
    The operation argument does not test any new path.
    """
    fn = 'getrs_npvt'
    mode = 'normal'
    size = get_size_configurations(case, mode)
    for s_nrhs in ['one', 'half_n', 'n']:
        for s in size:
            nrhs = get_nrhs(s_nrhs, s)
            ld = get_ld(s)
            row = {'name': precision+suite, 'name_test': suite, 'function': fn, 'precision': precision, 'nrhs': s_nrhs, 'n': s}
            yield (row, s, f'{COMMON_ARGS} -f {fn} -r {precision} --nrhs {nrhs} -n {s} --lda {ld} --ldb {ld}')


def getrsNpvtBatch_suite(*, suite, precision, case):
    """
    GETRSNPVTBATCH tests are run with the given precision and sizes, and with 1, n/2 and n right-hand-vectors
    The operation argument does not test any new path.
    """
    fn = 'getrs_npvt_batched'
    mode = 'batched'
    size = get_size_configurations(case, mode)
    for s_nrhs in ['one', 'half_n', 'n']:
        for s, bc in size:
            nrhs = get_nrhs(s_nrhs, s)
            ld = get_ld(s)
            row = {'name': precision+suite, 'name_test': suite, 'function': fn, 'precision': precision, 'batch_count': bc, 'nrhs': s_nrhs, 'n': s}
            yield (row, s, f'{COMMON_ARGS} -f {fn} -r {precision} --batch_count {bc} --nrhs {nrhs} -n {s} --lda {ld} --ldb {ld}')


def getriBatch_suite(*, suite, precision, case):
    """
    GETRIBATCH tests are run with the given precision and sizes
    """
    fn = 'getri_batched'
    mode = 'batched'
    size = get_size_configurations(case, mode)
    for s, bc in size:
        ld = get_ld(s)
        row = {'name': precision+suite, 'name_test': suite, 'function': fn, 'precision': precision, 'batch_count': bc, 'n': s}
        yield (row, s, f'{COMMON_ARGS} -f {fn} -r {precision} --batch_count {bc} -n {s} --lda {ld}')


def getriOOPBatch_suite(*, suite, precision, case):
    """
    GETRIOOPBATCH tests are run with the given precision and sizes
    """
    fn = 'getri_outofplace_batched'
    mode = 'batched'
    size = get_size_configurations(case, mode)
    for s, bc in size:
        ld = get_ld(s)
        row = {'name': precision+suite, 'name_test': suite, 'function': fn, 'precision': precision, 'batch_count': bc, 'n': s}
        yield (row, s, f'{COMMON_ARGS} -f {fn} -r {precision} --batch_count {bc} -n {s} --lda {ld} --ldc {ld}')


def trtri_suite(*, suite, precision, case):
    """
    TRTRI tests are run with the given precision and sizes, and for upper and lower cases.
    Upper case uses:        | Lower case uses:   
    trtri_upper             | trtri_lower
    trmv_upper_none         | trmv_lower_none
    trmm_upper_left_none    | trmm_lower_left_none
    trsm_upper_right_none   | trsm_lower_right_none
    <trti2_small_upper>     | <trti2_small_lower>
    """
    fn = 'trtri'
    mode = 'normal'
    size = get_size_configurations(case, mode)
    for s_uplo in ['upper', 'lower']:
        uplo = get_uplo(s_uplo)
        for s in size:
            ld = get_ld(s)
            row = {'name': precision+suite, 'name_test': suite, 'function': fn, 'precision': precision, 'uplo': s_uplo, 'n': s}
            yield (row, s, f'{COMMON_ARGS} -f {fn} -r {precision} --uplo {uplo} -n {s} --lda {ld}')


def geqrf_suite(*, suite, precision, case):
    """
    GEQRF tests are run, for the given precision and number of rows,
    with 160 columns and also for the square case (#rows = #columns)
    geqrf uses: 
    larft_forward_column
    larfb_forward_column_letf_transposed
    """
    fn = 'geqrf'
    mode = 'normal'
    size = get_size_configurations(case, mode)
    for s_shape in ['square', 'skinny']:
        for s in size:
            m,n = get_mn(s_shape, s, mode)
            ld = get_ld(s)
            if m >= n:
                row = {'name': precision+suite, 'name_test': suite, 'function': fn, 'precision': precision, 'shape': s_shape, 'n': s}
                yield (row, s, f'{COMMON_ARGS} -f {fn} -r {precision} -n {n} -m {m} --lda {ld}')


def geqrfBatch_suite(*, suite, precision, case):
    """
    GEQRFBATCH tests are run, for the given precision and number of rows,
    with 26 columns and also for the square case (#rows = #columns)
    geqrf uses: 
    larft_forward_column
    larfb_forward_column_letf_transposed
    """
    fn = 'geqrf_batched'
    mode = 'batched'
    size = get_size_configurations(case, mode)
    for s_shape in ['square', 'skinny']:
        for s, bc in size:
            m,n = get_mn(s_shape, s, mode)
            ld = get_ld(s)
            if m >= n:
                row = {'name': precision+suite, 'name_test': suite, 'function': fn, 'precision': precision, 'batch_count': bc, 'shape': s_shape, 'n': s}
                yield (row, s, f'{COMMON_ARGS} -f {fn} -r {precision} --batch_count {bc} -n {n} -m {m} --lda {ld}')


def cholqr_suite(*, suite, precision, case):
    """
    CHOLQR tests are run, for the given precision and number of rows,
    with 160 columns and also for the square case (#rows = #columns).
    Tests run for cholqr1 and cholqr2 variants.
    """
    fn = 'cholqr'
    mode = 'normal'
    cshift = 'N'
    size=get_size_configurations(case, mode)
    for s_shape in ['square', 'skinny']:
        for alg in [1, 2]:
            for s in size:
                m,n = get_mn(s_shape, s, mode)
                ld = get_ld(s)
                if m >= n:
                    row = {'name': precision+suite, 'name_test': suite, 'function': fn, 'precision': precision, 'shape': s_shape, 'algo': alg, 'n': s}
                    yield (row, s, f'{COMMON_ARGS} -f {fn} -r {precision} --cholshift {cshift} --cholnum {alg} -n {n} -m {m} --lda {ld}')


def cholqrBatch_suite(*, suite, precision, case):
    """
    CHOLQRBATCH tests are run, for the given precision and number of rows,
    with 26 columns and also for the square case (#rows = #columns)
    Tests run for cholqr1 and cholqr2 variants.
    """
    fn = 'cholqr_batched'
    mode = 'batched'
    cshift = 'N'
    size = get_size_configurations(case, mode)
    for s_shape in ['square', 'skinny']:
        for alg in [1, 2]:
            for s, bc in size:
                m,n = get_mn(s_shape, s, mode)
                ld = get_ld(s)
                if m >= n:
                    row = {'name': precision+suite, 'name_test': suite, 'function': fn, 'precision': precision, 'batch_count': bc, 'shape': s_shape, 'algo': alg, 'n': s}
                    yield (row, s, f'{COMMON_ARGS} -f {fn} -r {precision} --batch_count {bc} --cholshift {cshift} --cholnum {alg} -n {n} -m {m} --lda {ld}')


def gels_suite(*, suite, precision, case):
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
    mode = 'normal'
    size = get_size_configurations(case, mode)
    for s_shape in ['overdet', 'underdet']: 
        for s_nrhs in ['one', 'half_n', 'n']:
            for s in size:
                nrhs = get_nrhs(s_nrhs, s)
                m,n = get_mn(s_shape, s, mode)
                ld_b = get_ld(s)
                ld_a = get_ld(m)
                ld_x = get_ld(n)
                if (s_shape == 'overdet' and m >= n) or (s_shape == 'underdet' and m < n):
                    row = {'name': precision+suite, 'name_test': suite, 'function': fn, 'precision': precision, 'shape': s_shape, 'nrhs': s_nrhs, 'n': s}
                    yield (row, s, f'{COMMON_ARGS} -f {fn} -r {precision} -m {m} -n {n} --nrhs {nrhs} --lda {ld_a} --ldb {ld_b} --ldx {ld_x}')


def gelsBatch_suite(*, suite, precision, case):
    """
    GELSBATCH tests are run, for the given precision and number of rows (columns), with 26 columns (rows) and with 1,
    n/2 and n right-hand-vectors. We want the overdetermined case m >= n, but also the underdetermined m < n
    to actually test gelqf and ormlq/unmlq.
    gelqf uses:
    larft_forward_row
    larfb_forward_row_right_none
    ormlq uses:
    larft_forward_row
    larfb_forward_row_left_none
    """
    fn = 'gels_batched'
    mode = 'batched'
    size = get_size_configurations(case, mode)
    for s_shape in ['overdet', 'underdet']:
        for s_nrhs in ['one', 'half_n', 'n']:
            for s in size:
                nrhs = get_nrhs(s_nrhs, s)
                m,n = get_mn(s_shape, s, mode)
                ld_b = get_ld(s)
                ld_a = get_ld(m)
                ld_x = get_ld(n)
                if (s_shape == 'overdet' and m >= n) or (s_shape == 'underdet' and m < n):
                    row = {'name': precision+suite, 'name_test': suite, 'function': fn, 'precision': precision, 'batch_count': bc, 'shape': s_shape, 'nrhs': s_nrhs, 'n': s}
                    yield (row, s, f'{COMMON_ARGS} -f {fn} -r {precision} --batch_count {bc} -m {m} -n {n} --nrhs {nrhs} --lda {ld_a} --ldb {ld_b} --ldx {ld_x}')


def xxgqr_suite(*, suite, precision, case):
    """
    XXGQR (ORGQR or UNGQR) tests are run, for the given precision and number of rows,
    with 160 columns and also for the square case (#rows = #columns)
    orgqr uses:
    larft_forward_column
    larfb_forward_column_left_none
    """
    fn = 'orgqr' if precision == 's' or precision == 'd' else 'ungqr'
    mode = 'normal'
    size=get_size_configurations(case, mode)
    for s_shape in ['square', 'skinny']:
        for s in size:
            m,n = get_mn(s_shape, s, mode)
            ld = get_ld(s)
            if m >= n:
                row = {'name': precision+suite, 'name_test': suite, 'function': fn, 'precision': precision, 'shape': s_shape, 'n': s}
                yield (row, s, f'{COMMON_ARGS} -f {fn} -r {precision} -n {n} -m {m} --lda {ld}')


def xxmqr_suite(*, suite, precision, case):
    """
    XXMQR (ORMQR or UNMQR) tests are run with the given precision and sizes (only square case), from the right.
    Tests run ops = {transposed, none} cases.
    ormqr uses:             
    larft_forward_column
    larfb_forward_column_right_<ops>
    """
    fn = 'ormqr' if precision == 's' or precision == 'd' else 'unmqr'
    mode = 'normal'
    size = get_size_configurations(case, mode)
    for s_ops in ['none', 'trans']:
        ops = get_ops(s_ops, precision)
        for s in size:
            ld = get_ld(s)
            row = {'name': precision+suite, 'name_test': suite, 'function': fn, 'precision': precision, 'ops': s_ops, 'n': s}
            yield (row, s, f'{COMMON_ARGS} -f {fn} -r {precision} --side R --trans {ops} -n {s} --lda {ld} --ldc {ld}')


def larft_suite(*, suite, precision, case):
    """
    LARFT tests are run with the given precision and sizes, row-wise and col-wise in
    backward direction. Tests use 1, n/2 and n Householder vectors.
    """
    fn = 'larft'
    mode = 'normal'
    size = get_size_configurations(case, mode)
    for s_storev in ['colwise', 'rowwise']:
        storev = get_storev(s_storev)
        for s_nk in ['one', 'half_n', 'n']:
            for s in size:
                nk = get_nrhs(s_nk, s)
                ld1 = get_ld(s)
                ld2 = get_ld(nk)
                row = {'name': precision+suite, 'name_test': suite, 'function': fn, 'precision': precision, 'storev': s_storev, 'nk': s_nk, 'n': s}
                yield (row, s, f'{COMMON_ARGS} -f {fn} -r {precision} --direct B --storev {storev} -k {nk} -n {s} --ldv {ld1} --ldt {ld2}')


def xxtrd_suite(*, suite, precision, case):
    """
    XXTRD (SYTRD or HETRD) tests are run with the given precision and sizes.
    Tests run upper and lower cases. Upper or lower test different kernels.
    """
    fn = 'sytrd' if precision == 's' or precision == 'd' else 'hetrd'
    mode = 'normal'
    size = get_size_configurations(case, mode)
    for s_uplo in ['upper', 'lower']:
        uplo = get_uplo(s_uplo)
        for s in size:
            ld = get_ld(s)
            row = {'name': precision+suite, 'name_test': suite, 'function': fn, 'precision': precision, 'uplo': s_uplo, 'n': s}
            yield (row, s, f'{COMMON_ARGS} -f {fn} -r {precision} --uplo {uplo} -n {s} --lda {ld}')


def xxgtr_suite(*, suite, precision, case):
    """
    XXGTR (ORGTR or UNGTR) tests are run with the given precision and sizes.
    Always upper to actually use orgql/ungql.
    orgql uses:
    larft_backward_column
    larfb_backward_column_left_none
    """
    fn = 'orgtr' if precision == 's' or precision == 'd' else 'ungtr'
    mode = 'normal'
    size = get_size_configurations(case, mode)
    for s in size:
        ld = get_ld(s)
        row = {'name': precision+suite, 'name_test': suite, 'function': fn, 'precision': precision, 'n': s}
        yield (row, s, f'{COMMON_ARGS} -f {fn} -r {precision} --uplo U -n {s} --lda {ld}')


def xxmtr_suite(*, suite, precision, case):
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
    mode = 'normal'
    size = get_size_configurations(case, mode)
    for s_side in ['left', 'right']:
        side = get_side(s_side)
        for s_ops in ['none', 'trans']:
            ops = get_ops(s_ops, precision)
            for s in size:
                ld = get_ld(s)
                row = {'name': precision+suite, 'name_test': suite, 'function': fn, 'precision': precision, 'side': s_side, 'ops': s_ops, 'n': s}
                if s_side == 'right':
                    yield (row, s, f'{COMMON_ARGS} -f {fn} -r {precision} --uplo U --side {side} --trans {ops} -n {s} --lda {ld} --ldc {ld}')
                if s_side == 'left' and s_ops == 'trans':
                    yield (row, s, f'{COMMON_ARGS} -f {fn} -r {precision} --uplo U --side {side} --trans {ops} -m {s} --lda {ld} --ldc {ld}')


def gebrd_suite(*, suite, precision, case):
    """
    GEBRD tests are run with the given precision and sizes (only square case)
    """
    fn = 'gebrd'
    mode = 'normal'
    size = get_size_configurations(case, mode)
    for s in size:
        ld = get_ld(s)
        row = {'name': precision+suite, 'name_test': suite, 'function': fn, 'precision': precision, 'n': s}
        yield (row, s, f'{COMMON_ARGS} -f {fn} -r {precision} -m {s} --lda {ld}')


def xxgbr_suite(*, suite, precision, case):
    """
    XXGBR (ORGBR or UNGBR) tests are run with the given precision and sizes (only square case). 
    Always row-wise to actually test orglq/unglq.
    orglq uses:
    larft_forward_row
    larfb_forward_row_right_transposed
    """
    fn = 'orgbr' if precision == 's' or precision == 'd' else 'ungbr'
    mode = 'normal'
    size = get_size_configurations(case, mode)
    for s in size:
        ld = get_ld(s)
        row = {'name': precision+suite, 'name_test': suite, 'function': fn, 'precision': precision, 'n': s}
        yield (row, s, f'{COMMON_ARGS} -f {fn} -r {precision} --storev R -m {s} --lda {ld}')


def stedc_suite(*, suite, precision, case):
    """
    STEDC tests are run, for the given precision and sizes, with vectors and without vectors
    """
    fn = 'stedc' 
    mode = 'normal'
    size = get_size_configurations(case, mode)
    for s_evect in ['vect', 'novect']:
        evect = get_evect(s_evect, 'tridiag')
        for s in size:
            ld = get_ld(s)
            row = {'name': precision+suite, 'name_test': suite, 'function': fn, 'precision': precision, 'evect': s_evect, 'n': s}
            yield (row, s, f'{COMMON_ARGS} -f {fn} -r {precision} --evect {evect} -n {s} --ldc {ld}')


def xxevd_suite(*, suite, precision, case):
    """
    XXEVD (SYEVD or HEEVD) tests are run, for the given precision and sizes, with vectors and without vectors. Upper case.
    """
    fn = 'syevd' if precision == 's' or precision == 'd' else 'heevd'
    mode = 'normal'
    size = get_size_configurations(case, mode)
    for s_evect in ['vect', 'novect']:
        evect = get_evect(s_evect, 'orig')
        for s in size:
            ld = get_ld(s)
            row = {'name': precision+suite, 'name_test': suite, 'function': fn, 'precision': precision, 'evect': s_evect, 'n': s}
            yield (row, s, f'{COMMON_ARGS} -f {fn} -r {precision} --evect {evect} -n {s} --lda {ld}')


def xxgvd_suite(*, suite, precision, case):
    """
    XXGVD (SYGVD or HEGVD) tests are run, for the given precision and sizes, with vectors.
    Tests run upper and lower case with AX and BAX forms. 
    """
    fn = 'sygvd' if precision == 's' or precision == 'd' else 'hegvd'
    mode = 'normal'
    size = get_size_configurations(case, mode)
    for s_uplo in ['upper', 'lower']:
        uplo = get_uplo(s_uplo)
        for s_itype in ['AX', 'BAX']:
            itype = get_itype(s_itype)
            for s in size:
                ld = get_ld(s)
                row = {'name': precision+suite, 'name_test': suite, 'function': fn, 'precision': precision, 'uplo': s_uplo, 'itype': s_itype, 'n': s}
                yield (row, s, f'{COMMON_ARGS} -f {fn} -r {precision} --evect V --uplo {uplo} --itype {itype} -n {s} --lda {ld} --ldb {ld}')


def xxevdBatch_suite(*, suite, precision, case):
    """
    XXEVDBATCH (SYEVDBATCH or HEEVDBATCH) tests are run, for the given precision and sizes, with vectors and without vectors
    """
    fn = 'syevd_strided_batched' if precision == 's' or precision == 'd' else 'heevd_strided_batched'
    mode = 'batched'
    size = get_size_configurations(case, mode)
    for s_evect in ['vect', 'novect']:
        evect = get_evect(s_evect, 'orig')
        for s, bc in size:
            ld = get_ld(s)
            row = {'name': precision+suite, 'name_test': suite, 'function': fn, 'precision': precision, 'batch_count': bc, 'evect': s_evect, 'n': s}
            yield (row, s, f'{COMMON_ARGS} -f {fn} -r {precision} --batch_count {bc} --evect {evect} -n {s} --lda {ld}')


def xxevBatch_suite(*, suite, precision, case):
    """
    XXEVBATCH (SYEVBATCH or HEEVBATCH) tests are run, for the given precision and sizes, with vectors and without vectors
    """
    fn = 'syev_strided_batched' if precision == 's' or precision == 'd' else 'heev_strided_batched'
    mode = 'batched'
    size = get_size_configurations(case, mode)
    for s_evect in ['vect', 'novect']:
        evect = get_evect(s_evect, 'orig')
        for s, bc in size:
            ld = get_ld(s)
            row = {'name': precision+suite, 'name_test': suite, 'function': fn, 'precision': precision, 'batch_count': bc, 'evect': s_evect, 'n': s}
            yield (row, s, f'{COMMON_ARGS} -f {fn} -r {precision} --batch_count {bc} --evect {evect} -n {s} --lda {ld}')


def xxevdx_suite(*, suite, precision, case):
    """
    XXEVDX (SYEVDX or HEEVDX) tests are run, for the given precision and sizes, with vectors and 
    computing 20 and 60 percent of the eigenvalues. Upper case.
    """
    fn = 'syevdx' if precision == 's' or precision == 'd' else 'heevdx'
    mode = 'normal'
    size=get_size_configurations(case, mode)
    for per in [20, 60]:
        for s in size:
            iu = get_upperid(per, s)
            ld = get_ld(s)
            row = {'name': precision+suite, 'name_test': suite, 'function': fn, 'precision': precision, 'range': per, 'n': s}
            yield (row, s, f'{COMMON_ARGS} -f {fn} -r {precision} --evect V --erange I --il 1 --iu {iu} -n {s} --lda {ld} --ldz {ld}')


def xxgvdx_suite(*, suite, precision, case):
    """
    XXGVDX (SYGVDX or HEGVDX) tests are run, for the given precision and sizes, with vectors and 
    computing 20 and 60 percent of the eigenvalues. Upper case, AX form. 
    """
    fn = 'sygvdx' if precision == 's' or precision == 'd' else 'hegvdx'
    mode = 'normal'
    size=get_size_configurations(case, mode)
    for per in [20, 60]:
        for s in size:
            iu = get_upperid(per, s)
            ld = get_ld(s)
            row = {'name': precision+suite, 'name_test': suite, 'function': fn, 'precision': precision, 'range': per, 'n': s}
            yield (row, s, f'{COMMON_ARGS} -f {fn} -r {precision} --evect V --erange I --il 1 --iu {iu} -n {s} --lda {ld} --ldz {ld}')


def xxevj_suite(*, suite, precision, case):
    """
    XXEVJ (SYEVJ or HEEVJ) tests are run, for the given precision and sizes, with vectors and without vectors. Upper case.
    """
    fn = 'syevj' if precision == 's' or precision == 'd' else 'heevj'
    mode = 'normal'
    size = get_size_configurations(case, mode)
    for s_evect in ['vect', 'novect']:
        evect = get_evect(s_evect, 'orig')
        for s in size:
            ld = get_ld(s)
            row = {'name': precision+suite, 'name_test': suite, 'function': fn, 'precision': precision, 'evect': s_evect, 'n': s}
            yield (row, s, f'{COMMON_ARGS} -f {fn} -r {precision} --evect {evect} -n {s} --lda {ld}')


def xxgvj_suite(*, suite, precision, case):
    """
    XXGVJ (SYGVJ or HEGVJ) tests are run, for the given precision and sizes, with vectors. Upper case, AX form.
    """
    fn = 'sygvj' if precision == 's' or precision == 'd' else 'hegvj'
    mode = 'normal'
    size = get_size_configurations(case, mode)
    for s in size:
        ld = get_ld(s)
        row = {'name': precision+suite, 'name_test': suite, 'function': fn, 'precision': precision, 'n': s}
        yield (row, s, f'{COMMON_ARGS} -f {fn} -r {precision} --evect V -n {s} --lda {ld}')


def xxevjBatch_suite(*, suite, precision, case):
    """
    XXEVJBATCH (SYEVJBATCH or HEEVJBATCH) tests are run, for the given precision and sizes, with vectors and without vectors. Upper case.
    """
    fn = 'syevj_strided_batched' if precision == 's' or precision == 'd' else 'heevj_strided_batched'
    mode = 'batched'
    size = get_size_configurations(case, mode)
    for s_evect in ['vect', 'novect']:
        evect = get_evect(s_evect, 'orig')
        for s, bc in size:
            ld = get_ld(s)
            row = {'name': precision+suite, 'name_test': suite, 'function': fn, 'precision': precision, 'batch_count': bc, 'evect': s_evect, 'n': s}
            yield (row, s, f'{COMMON_ARGS} -f {fn} -r {precision} --batch_count {bc} --evect {evect} -n {s} --lda {ld}')


def gesvd_suite(*, suite, precision, case):
    """
    GESVD tests are run, for the given precision and sizes, with vectors and without vectors (only square case).
    Tests are run with the hybrid approach as well. 
    """
    fn = 'gesvd'
    mode = 'normal'
    size = get_size_configurations(case, mode)
    for s_alg in ['gpu' ,'hybrid']:
        alg = get_alg(s_alg)
        for s_evect in ['vect', 'novect']:
            evect = get_evect(s_evect, 'orig')
            for s in size:
                ld = get_ld(s)
                row = {'name': precision+suite, 'name_test': suite, 'function': fn, 'precision': precision, 'algmode': s_alg, 'svect': s_evect, 'n': s}
                yield (row, s, f'{COMMON_ARGS} -f {fn} -r {precision} --alg_mode {alg} --left_svect {evect} --right_svect {evect} -m {s} --lda {ld} --ldu {ld} --ldv {ld}')


def gesdd_suite(*, suite, precision, case):
    """
    GESDD tests are run, for the given precision and sizes, with vectors and without vectors (only square case).
    """
    fn = 'gesdd'
    mode = 'normal'
    size = get_size_configurations(case, mode)
    for s_evect in ['vect', 'novect']:
        evect = get_evect(s_evect, 'orig')
        for s in size:
            ld = get_ld(s)
            row = {'name': precision+suite, 'name_test': suite, 'function': fn, 'precision': precision, 'svect': s_evect, 'n': s}
            yield (row, s, f'{COMMON_ARGS} -f {fn} -r {precision} --left_svect {evect} --right_svect {evect} -m {s} --lda {ld} --ldu {ld} --ldv {ld}')


def gesvdj_suite(*, suite, precision, case):
    """
    GESVDJ tests are run, for the given precision and sizes, with vectors and without vectors (only square case).
    """
    fn = 'gesvdj'
    mode = 'normal'
    size = get_size_configurations(case, mode)
    for s_evect in ['vect', 'novect']:
        evect = get_evect(s_evect, 'orig')
        for s in size:
            ld = get_ld(s)
            row = {'name': precision+suite, 'name_test': suite, 'function': fn, 'precision': precision, 'svect': s_evect, 'n': s}
            yield (row, s, f'{COMMON_ARGS} -f {fn} -r {precision} --left_svect {evect} --right_svect {evect} -m {s} --lda {ld} --ldu {ld} --ldv {ld}')


def gesvdjBatch_suite(*, suite, precision, case):
    """
    GESVDJBATCH tests are run, for the given precision and sizes, with vectors and without vectors (only square case).
    """
    fn = 'gesvdj_strided_batched'
    mode = 'batched'
    size = get_size_configurations(case, mode)
    for s_evect in ['vect', 'novect']:
        evect = get_evect(s_evect, 'orig')
        for s, bc in size:
            ld = get_ld(s)
            row = {'name': precision+suite, 'name_test': suite, 'function': fn, 'precision': precision, 'batch_count': bc, 'evect': s_evect, 'n': s}
            yield (row, s, f'{COMMON_ARGS} -f {fn} -r {precision} --batch_count {bc} --left_svect {evect} --right_svect {evect} -m {s} --lda {ld} --ldu {ld} --ldv {ld}')


# Registry of all available benchmark suites
#########################################################

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
    'getrfBatch': getrfBatch_suite,
    'getrfNpvt': getrfNpvt_suite,
    'getrfNpvtBatch': getrfNpvtBatch_suite,
    'getrs': getrs_suite,
    'getrsBatch': getrsBatch_suite,
    'getrsNpvt': getrsNpvt_suite,               
    'getrsNpvtBatch': getrsNpvtBatch_suite,     
    'getriBatch': getriBatch_suite,
    'getriOOPBatch': getriOOPBatch_suite,
    'trtri': trtri_suite,

    # Over-determined linear systems (least-squares)
    'geqrf': geqrf_suite,
    'geqrfBatch': geqrfBatch_suite,
    'cholqr': cholqr_suite,                     
    'cholqrBatch': cholqrBatch_suite,           
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
    'stedc': stedc_suite,
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
