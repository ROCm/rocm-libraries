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

# Common benchmark arguments - always do 7 iterations in perf mode
COMMON_ARGS = '--iters 7 --perf 1'


def get_size_configurations(case):
    """
    Get size configurations for normal and batched tests.

    Args:
        case: One of 'small', 'medium', 'large' or 'huge'

    Returns:
        tuple: (sizenormal, sizebatch) lists
    """
    sizenormal = list(chain(range(2, 64, 8), range(64, 256, 32), range(256, 1024, 64)))
    sizebatch = list(chain(zip(range(2, 64, 4), repeat(5000)), zip(range(72, 164, 8), repeat(2500))))

    if case == 'medium' or case == 'large' or case == 'huge':
        sizenormal += list(chain(range(1024, 2048, 64), range(2048, 4096, 128)))
        sizebatch += list(chain(zip(range(168, 260, 8), repeat(2500)), zip(range(272, 520, 16), repeat(1000))))

    if case == 'large' or case == 'huge':
        sizenormal += list(chain(range(4096, 8192, 256), range(8192, 12800, 512)))
        sizebatch += list(chain(zip(range(544, 1050, 32), repeat(500)), zip(range(1088, 2050, 64), repeat(50))))

    if case == 'huge':
        sizenormal += list(chain(range(12800, 23040, 2048), range(23040, 32768, 4096)))

    return sizenormal, sizebatch


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
    for shape in ['upper', 'lower']:
        if shape == 'upper': upl = 'U'
        else: upl = 'L' 
        for s in size:
            if s < 4000: ld = s + 1
            else: ld = s + 64
            row = {'name': precision+suite, 'name_test': suite, 'function': fn, 'precision': precision, 'uplo': shape, 'n': s}
            yield (row, s, f'{COMMON_ARGS} -f {fn} -r {precision} --uplo {upl} -n {s} --lda {ld}')


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
    for shape in ['upper', 'lower']:
        if shape == 'upper': upl = 'U'
        else: upl = 'L'
        for s, bc in size:
            if s < 4000: ld = s + 1
            else: ld = s + 64
            row = {'name': precision+suite, 'name_test': suite, 'function': fn, 'precision': precision, 'batch_count': bc, 'uplo': shape, 'n': s}
            yield (row, s, f'{COMMON_ARGS} -f {fn} -r {precision} --batch_count {bc} --uplo {upl} -n {s} --lda {ld}')


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
    for shape in ['upper', 'lower']:
        if shape == 'upper': upl = 'U'
        else: upl = 'L'
        for nv in ['one', 'half_n', 'n']:
            nrhs = 1
            for s in size:
                if s < 4000: ld = s + 1
                else: ld = s + 64
                if nv == 'half_n': nrhs = s//2
                elif nv == 'n': nrhs = s 
                row = {'name': precision+suite, 'name_test': suite, 'function': fn, 'precision': precision, 'uplo': shape, 'nrhs': nv, 'n': s}
                yield (row, s, f'{COMMON_ARGS} -f {fn} -r {precision} --uplo {upl} --nrhs {nrhs} -n {s} --lda {ld} --ldb {ld}')


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
    for shape in ['upper', 'lower']:
        if shape == 'upper': upl = 'U'
        else: upl = 'L'
        for nv in ['one', 'half_n', 'n']:
            nrhs = 1
            for s, bc in size:
                if s < 4000: ld = s + 1
                else: ld = s + 64
                if nv == 'half_n': nrhs = s//2
                elif nv == 'n': nrhs = s
                row = {'name': precision+suite, 'name_test': suite, 'function': fn, 'precision': precision, 'batch_count': bc, 'uplo': shape, 'nrhs': nv, 'n': s}
                yield (row, s, f'{COMMON_ARGS} -f {fn} -r {precision} --batch_count {bc} --uplo {upl} --nrhs {nrhs} -n {s} --lda {ld} --ldb {ld}')


def potri_suite(*, suite, precision, sizenormal, sizebatch):
    """
    POTRI tests are run with the given precision and sizes, and for upper and lower cases.
    Upper case uses:            | Lower case uses:
    rocsolver_trtri_upper       | rocsolver_trtri_lower
    trmm_upper_right_transposed | trmm_lower_left_transposed 
    """
    fn = 'potri'
    size = sizenormal
    for shape in ['upper', 'lower']:
        if shape == 'upper': upl = 'U'
        else: upl = 'L'
        for s in size:
            if s < 4000: ld = s + 1
            else: ld = s + 64
            row = {'name': precision+suite, 'name_test': suite, 'function': fn, 'precision': precision, 'uplo': shape, 'n': s}
            yield (row, s, f'{COMMON_ARGS} -f {fn} -r {precision} --uplo {upl} -n {s} --lda {ld}')


def sytrf_suite(*, suite, precision, sizenormal, sizebatch):
    """
    SYTRF tests are run with the given precision and sizes, and for upper and lower cases.
    Upper or lower test different kernels.
    """
    fn = 'sytrf'
    size = sizenormal
    for shape in ['upper', 'lower']:
        if shape == 'upper': upl = 'U'
        else: upl = 'L'
        for s in size:
            if s < 4000: ld = s + 1
            else: ld = s + 64
            row = {'name': precision+suite, 'name_test': suite, 'function': fn, 'precision': precision, 'uplo': shape, 'n': s}
            yield (row, s, f'{COMMON_ARGS} -f {fn} -r {precision} --uplo {upl} -n {s} --lda {ld}')


def sytrs_suite(*, suite, precision, sizenormal, sizebatch):
    """
    SYTRS tests are run with the given precision and sizes, and with 1, n/2 and n right-hand-vectors.
    Tests run upper and lower cases. Upper or lower test different kernels.
    """
    fn = 'sytrs'
    size = sizenormal
    for shape in ['upper', 'lower']:
        if shape == 'upper': upl = 'U'
        else: upl = 'L'
        for nv in ['one', 'half_n', 'n']:
            nrhs = 1
            for s in size:
                if s < 4000: ld = s + 1
                else: ld = s + 64
                if nv == 'half_n': nrhs = s//2
                elif nv == 'n': nrhs = s
                row = {'name': precision+suite, 'name_test': suite, 'function': fn, 'precision': precision, 'uplo': shape, 'nrhs': nv, 'n': s}
                yield (row, s, f'{COMMON_ARGS} -f {fn} -r {precision} --uplo {upl} --nrhs {nrhs} -n {s} --lda {ld} --ldb {ld}')


def getrf_suite(*, suite, precision, sizenormal, sizebatch):
    """
    GETRF tests are run with the given precision and sizes (only square case)
    """
    fn = 'getrf'
    size = sizenormal
    for s in size:
        if s < 4000: ld = s + 1
        else: ld = s + 64
        row = {'name': precision+suite, 'name_test': suite, 'function': fn, 'precision': precision, 'n': s}
        yield (row, s, f'{COMMON_ARGS} -f {fn} -r {precision} -m {s} --lda {ld}')


def getrfBatch_suite(*, suite, precision, sizenormal, sizebatch):
    """
    GETRFBATCH tests are run with the given precision and sizes (only square case)
    """
    fn = 'getrf_batched'
    size = sizebatch
    for s, bc in size:
        if s < 4000: ld = s + 1
        else: ld = s + 64
        row = {'name': precision+suite, 'name_test': suite, 'function': fn, 'precision': precision, 'batch_count': bc, 'n': s}
        yield (row, s, f'{COMMON_ARGS} -f {fn} -r {precision} --batch_count {bc} -m {s} --lda {ld}')


def getrfNpvt_suite(*, suite, precision, sizenormal, sizebatch):
    """
    GETRFNPVT tests are run with the given precision and sizes (only square case)
    """
    fn = 'getrf_npvt'
    size = sizenormal
    for s in size:
        if s < 4000: ld = s + 1
        else: ld = s + 64
        row = {'name': precision+suite, 'name_test': suite, 'function': fn, 'precision': precision, 'n': s}
        yield (row, s, f'{COMMON_ARGS} -f {fn} -r {precision} -m {s} --lda {ld}')


def getrfNpvtBatch_suite(*, suite, precision, sizenormal, sizebatch):
    """
    GETRFNPVTBATCH tests are run with the given precision and sizes (only square case)
    """
    fn = 'getrf_npvt_batched'
    size = sizebatch
    for s, bc in size:
        if s < 4000: ld = s + 1
        else: ld = s + 64
        row = {'name': precision+suite, 'name_test': suite, 'function': fn, 'precision': precision, 'batch_count': bc, 'n': s}
        yield (row, s, f'{COMMON_ARGS} -f {fn} -r {precision} --batch_count {bc} -m {s} --lda {ld}')


def getrs_suite(*, suite, precision, sizenormal, sizebatch):
    """
    GETRS tests are run with the given precision and sizes, and with 1, n/2 and n right-hand-vectors
    The operation argument does not test any new path.
    """
    fn = 'getrs'
    size = sizenormal
    for nv in ['one', 'half_n', 'n']:
        nrhs = 1
        for s in size:
            if s < 4000: ld = s + 1
            else: ld = s + 64
            if nv == 'half_n': nrhs = s//2
            elif nv == 'n': nrhs = s 
            row = {'name': precision+suite, 'name_test': suite, 'function': fn, 'precision': precision, 'nrhs': nv, 'n': s}
            yield (row, s, f'{COMMON_ARGS} -f {fn} -r {precision} --nrhs {nrhs} -n {s} --lda {ld} --ldb {ld}')


def getrsBatch_suite(*, suite, precision, sizenormal, sizebatch):
    """
    GETRSBATCH tests are run with the given precision and sizes, and with 1, n/2 and n right-hand-vectors
    The operation argument does not test any new path.
    """
    fn = 'getrs_batched'
    size = sizebatch
    for nv in ['one', 'half_n', 'n']:
        nrhs = 1
        for s, bc in size:
            if s < 4000: ld = s + 1
            else: ld = s + 64
            if nv == 'half_n': nrhs = s//2
            elif nv == 'n': nrhs = s
            row = {'name': precision+suite, 'name_test': suite, 'function': fn, 'precision': precision, 'batch_count': bc, 'nrhs': nv, 'n': s}
            yield (row, s, f'{COMMON_ARGS} -f {fn} -r {precision} --batch_count {bc} --nrhs {nrhs} -n {s} --lda {ld} --ldb {ld}')


def getrsNpvt_suite(*, suite, precision, sizenormal, sizebatch):
    """
    GETRSNPVT tests are run with the given precision and sizes, and with 1, n/2 and n right-hand-vectors
    The operation argument does not test any new path.
    """
    fn = 'getrs_npvt'
    size = sizenormal
    for nv in ['one', 'half_n', 'n']:
        nrhs = 1
        for s in size:
            if s < 4000: ld = s + 1
            else: ld = s + 64
            if nv == 'half_n': nrhs = s//2
            elif nv == 'n': nrhs = s
            row = {'name': precision+suite, 'name_test': suite, 'function': fn, 'precision': precision, 'nrhs': nv, 'n': s}
            yield (row, s, f'{COMMON_ARGS} -f {fn} -r {precision} --nrhs {nrhs} -n {s} --lda {ld} --ldb {ld}')


def getrsNpvtBatch_suite(*, suite, precision, sizenormal, sizebatch):
    """
    GETRSNPVTBATCH tests are run with the given precision and sizes, and with 1, n/2 and n right-hand-vectors
    The operation argument does not test any new path.
    """
    fn = 'getrs_npvt_batched'
    size = sizebatch
    for nv in ['one', 'half_n', 'n']:
        nrhs = 1
        for s, bc in size:
            if s < 4000: ld = s + 1
            else: ld = s + 64
            if nv == 'half_n': nrhs = s//2
            elif nv == 'n': nrhs = s
            row = {'name': precision+suite, 'name_test': suite, 'function': fn, 'precision': precision, 'batch_count': bc, 'nrhs': nv, 'n': s}
            yield (row, s, f'{COMMON_ARGS} -f {fn} -r {precision} --batch_count {bc} --nrhs {nrhs} -n {s} --lda {ld} --ldb {ld}')


def getriBatch_suite(*, suite, precision, sizenormal, sizebatch):
    """
    GETRIBATCH tests are run with the given precision and sizes
    """
    fn = 'getri_batched'
    size = sizebatch
    for s, bc in size:
        if s < 4000: ld = s + 1
        else: ld = s + 64
        row = {'name': precision+suite, 'name_test': suite, 'function': fn, 'precision': precision, 'batch_count': bc, 'n': s}
        yield (row, s, f'{COMMON_ARGS} -f {fn} -r {precision} --batch_count {bc} -n {s} --lda {ld}')


def getriOOPBatch_suite(*, suite, precision, sizenormal, sizebatch):
    """
    GETRIOOPBATCH tests are run with the given precision and sizes
    """
    fn = 'getri_outofplace_batched'
    size = sizebatch
    for s, bc in size:
        if s < 4000: ld = s + 1
        else: ld = s + 64
        row = {'name': precision+suite, 'name_test': suite, 'function': fn, 'precision': precision, 'batch_count': bc, 'n': s}
        yield (row, s, f'{COMMON_ARGS} -f {fn} -r {precision} --batch_count {bc} -n {s} --lda {ld} --ldc {ld}')


def trtri_suite(*, suite, precision, sizenormal, sizebatch):
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
    size = sizenormal
    for shape in ['upper', 'lower']:
        if shape == 'upper': upl = 'U'
        else: upl = 'L'
        for s in size:
            if s < 4000: ld = s + 1
            else: ld = s + 64
            row = {'name': precision+suite, 'name_test': suite, 'function': fn, 'precision': precision, 'uplo': shape, 'n': s}
            yield (row, s, f'{COMMON_ARGS} -f {fn} -r {precision} --uplo {upl} -n {s} --lda {ld}')


def geqrf_suite(*, suite, precision, sizenormal, sizebatch):
    """
    GEQRF tests are run, for the given precision and number of rows,
    with 160 columns and also for the square case (#rows = #columns)
    geqrf uses: 
    larft_forward_column
    larfb_forward_column_letf_transposed
    """
    fn = 'geqrf'
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


def geqrfBatch_suite(*, suite, precision, sizenormal, sizebatch):
    """
    GEQRFBATCH tests are run, for the given precision and number of rows,
    with 26 columns and also for the square case (#rows = #columns)
    geqrf uses: 
    larft_forward_column
    larfb_forward_column_letf_transposed
    """
    fn = 'geqrf_batched'
    size = sizebatch
    for nc in [0, 26]:
        if nc == 0: nn = 'sq'
        else: nn = nc
        for s, bc in size:
            if s < 4000: ld = s + 1
            else: ld = s + 64
            if nc == 0: n = s
            else: n = nc
            if s >= n:
                row = {'name': precision+suite, 'name_test': suite, 'function': fn, 'precision': precision, 'batch_count': bc, 'cols': nn, 'n': s}
                yield (row, s, f'{COMMON_ARGS} -f {fn} -r {precision} --batch_count {bc} -n {n} -m {s} --lda {ld}')


def cholqr_suite(*, suite, precision, sizenormal, sizebatch):
    """
    CHOLQR tests are run, for the given precision and number of rows,
    with 160 columns and also for the square case (#rows = #columns).
    Tests run for cholqr1 and cholqr2 variants.
    """
    fn = 'cholqr'
    cshift = 'N'
    size=sizenormal
    for nc in [0, 160]:
        if nc == 0: nn = 'sq'
        else: nn = nc
        for alg in [1, 2]:
            for s in size:
                if s < 4000: ld = s + 1
                else: ld = s + 64
                if nc == 0: n = s
                else: n = nc
                if s >= n:
                    row = {'name': precision+suite, 'name_test': suite, 'function': fn, 'precision': precision, 'cols': nn, 'algo': alg, 'n': s}
                    yield (row, s, f'{COMMON_ARGS} -f {fn} -r {precision} --cholshift {cshift} --cholnum {alg} -n {n} -m {s} --lda {ld}')


def cholqrBatch_suite(*, suite, precision, sizenormal, sizebatch):
    """
    CHOLQRBATCH tests are run, for the given precision and number of rows,
    with 26 columns and also for the square case (#rows = #columns)
    Tests run for cholqr1 and cholqr2 variants.
    """
    fn = 'cholqr_batched'
    cshift = 'N'
    size = sizebatch
    for nc in [0, 26]:
        if nc == 0: nn = 'sq'
        else: nn = nc
        for alg in [1, 2]:
            for s, bc in size:
                if s < 4000: ld = s + 1
                else: ld = s + 64
                if nc == 0: n = s
                else: n = nc
                if s >= n:
                    row = {'name': precision+suite, 'name_test': suite, 'function': fn, 'precision': precision, 'batch_count': bc, 'cols': nn, 'algo': alg, 'n': s}
                    yield (row, s, f'{COMMON_ARGS} -f {fn} -r {precision} --batch_count {bc} --cholshift {cshift} --cholnum {alg} -n {n} -m {s} --lda {ld}')


def gels_suite(*, suite, precision, sizenormal, sizebatch):
    """
    GELS tests are run, for the given precision and number of rows, with 160 columns and with 1, 
    n/2 and n right-hand-vectors. Only with m < n to actually test gelqf and ormlq/unmlq. 
    Tests run ops = {none, transposed} cases.
    gelqf uses:
    larft_forward_row
    larfb_forward_row_right_none
    ormlq uses:           
    larft_forward_row           
    larfb_forward_row_left_<ops>
    """
    fn = 'gels'
    tr = 'T' if precision == 's' or precision == 'd' else 'C'
    size = sizenormal
    for ops in ['none', 'trans']:
        if ops == 'none': op = 'N'
        else: op = tr
        for nv in ['one', 'half_n', 'n']:
            nrhs = 1
            for s in size:
                if s < 4000: ld = s + 1
                else: ld = s + 64
                if nv == 'half_n': nrhs = s//2
                elif nv == 'n': nrhs = s
                if s >= 160:
                    row = {'name': precision+suite, 'name_test': suite, 'function': fn, 'precision': precision, 'trans': ops, 'nrhs': nv, 'n': s}
                    yield (row, s, f'{COMMON_ARGS} -f {fn} -r {precision} -m 160 --trans {op} --nrhs {nrhs} -n {s} --lda 161 --ldb {ld}')


def gelsBatch_suite(*, suite, precision, sizenormal, sizebatch):
    """
    GELSBATCH tests are run, for the given precision and number of rows, with 26 columns and with 1, 
    n/2 and n right-hand-vectors. Only with m < n to actually test gelqf and ormlq/unmlq.
    Tests run ops = {none, transposed} cases.
    gelqf uses:
    larft_forward_row
    larfb_forward_row_right_none
    ormlq uses:           
    larft_forward_row           
    larfb_forward_row_left_<ops>
    """
    fn = 'gels_batched'
    tr = 'T' if precision == 's' or precision == 'd' else 'C'
    size = sizebatch
    for ops in ['none', 'trans']:
        if ops == 'none': op = 'N'
        else: op = tr
        for nv in ['one', 'half_n', 'n']:
            nrhs = 1
            for s, bc in size:
                if s < 4000: ld = s + 1
                else: ld = s + 64
                if nv == 'half_n': nrhs = s//2
                elif nv == 'n': nrhs = s
                if s >= 26:
                    row = {'name': precision+suite, 'name_test': suite, 'function': fn, 'precision': precision, 'batch_count': bc, 'trans': ops, 'nrhs': nv, 'n': s}
                    yield (row, s, f'{COMMON_ARGS} -f {fn} -r {precision} --batch_count {bc} -m 26 --trans {op} --nrhs {nrhs} -n {s} --lda 27 --ldb {ld}')


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


def larft_suite(*, suite, precision, sizenormal, sizebatch):
    """
    LARFT tests are run with the given precision and sizes, row-wise and
    backward direction. Tests use 1, n/2 and n Householder vectors.
    """
    fn = 'larft'
    size = sizenormal
    for nk in ['one', 'half_n', 'n']:
        k = 1
        for s in size:
            if nk == 'half_n': k = s//2
            elif nk == 'n': k = s
            if s < 4000: ld1 = s + 1
            else: ld1 = s + 64
            if k < 4000: ld2 = k + 1
            else: ld2 = k + 64
            row = {'name': precision+suite, 'name_test': suite, 'function': fn, 'precision': precision, 'nk': nk, 'n': s}
            yield (row, s, f'{COMMON_ARGS} -f {fn} -r {precision} --storev R --direct B -k {k} -n {s} --ldv {ld1} --ldt {ld2}')


def xxtrd_suite(*, suite, precision, sizenormal, sizebatch):
    """
    XXTRD (SYTRD or HETRD) tests are run with the given precision and sizes.
    Tests run upper and lower cases. Upper or lower test different kernels.
    """
    fn = 'sytrd' if precision == 's' or precision == 'd' else 'hetrd'
    size = sizenormal
    for shape in ['upper', 'lower']:
        if shape == 'upper': upl = 'U'
        else: upl = 'L'
        for s in size:
            if s < 4000: ld = s + 1
            else: ld = s + 64
            row = {'name': precision+suite, 'name_test': suite, 'function': fn, 'precision': precision, 'uplo': shape, 'n': s}
            yield (row, s, f'{COMMON_ARGS} -f {fn} -r {precision} --uplo {upl} -n {s} --lda {ld}')


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
    Always row-wise to actually test orglq/unglq.
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
        yield (row, s, f'{COMMON_ARGS} -f {fn} -r {precision} --storev R -m {s} --lda {ld}')


def stedc_suite(*, suite, precision, sizenormal, sizebatch):
    """
    STEDC tests are run, for the given precision and sizes, with vectors and without vectors
    """
    fn = 'stedc' 
    size = sizenormal
    for v in ['V', 'N']:
        if v == 'V': vv = 'vect'
        else: vv = 'novect'
        for s in size:
            if s < 4000: ld = s + 1
            else: ld = s + 64
            row = {'name': precision+suite, 'name_test': suite, 'function': fn, 'precision': precision, 'evect': vv, 'n': s}
            yield (row, s, f'{COMMON_ARGS} -f {fn} -r {precision} --evect {v} -n {s} --ldc {ld}')


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
            yield (row, s, f'{COMMON_ARGS} -f {fn} -r {precision} --evect {v} -n {s} --lda {ld}')


def xxgvd_suite(*, suite, precision, sizenormal, sizebatch):
    """
    XXGVD (SYGVD or HEGVD) tests are run, for the given precision and sizes, with vectors.
    Tests run upper and lower case with AX and BAX forms. 
    """
    fn = 'sygvd' if precision == 's' or precision == 'd' else 'hegvd'
    size = sizenormal
    for shape in ['upper', 'lower']:
        if shape == 'upper': upl = 'U'
        else: upl = 'L'
        for ty in ['AX', 'BAX']:
            if ty == 'AX': ity = 1
            else: ity = 3
            for s in size:
                if s < 4000: ld = s + 1
                else: ld = s + 64
                row = {'name': precision+suite, 'name_test': suite, 'function': fn, 'precision': precision, 'uplo': shape, 'type': ty, 'n': s}
                yield (row, s, f'{COMMON_ARGS} -f {fn} -r {precision} --evect V --uplo {upl} --itype {ity} -n {s} --lda {ld} --ldb {ld}')


def xxevdBatch_suite(*, suite, precision, sizenormal, sizebatch):
    """
    XXEVDBATCH (SYEVDBATCH or HEEVDBATCH) tests are run, for the given precision and sizes, with vectors and without vectors
    """
    fn = 'syevd_strided_batched' if precision == 's' or precision == 'd' else 'heevd_strided_batched'
    size = sizebatch
    for v in ['V', 'N']:
        if v == 'V': vv = 'vect'
        else: vv = 'novect'
        for s, bc in size:
            if s < 4000: ld = s + 1
            else: ld = s + 64
            row = {'name': precision+suite, 'name_test': suite, 'function': fn, 'precision': precision, 'batch_count': bc, 'evect': vv, 'n': s}
            yield (row, s, f'{COMMON_ARGS} -f {fn} -r {precision} --batch_count {bc} --evect {v} -n {s} --lda {ld}')


def xxevBatch_suite(*, suite, precision, sizenormal, sizebatch):
    """
    XXEVBATCH (SYEVBATCH or HEEVBATCH) tests are run, for the given precision and sizes, with vectors and without vectors
    """
    fn = 'syev_strided_batched' if precision == 's' or precision == 'd' else 'heev_strided_batched'
    size = sizebatch
    for v in ['V', 'N']:
        if v == 'V': vv = 'vect'
        else: vv = 'novect'
        for s, bc in size:
            if s < 4000: ld = s + 1
            else: ld = s + 64
            row = {'name': precision+suite, 'name_test': suite, 'function': fn, 'precision': precision, 'batch_count': bc, 'evect': vv, 'n': s}
            yield (row, s, f'{COMMON_ARGS} -f {fn} -r {precision} --batch_count {bc} --evect {v} -n {s} --lda {ld}')


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
            yield (row, s, f'{COMMON_ARGS} -f {fn} -r {precision} --evect V --erange I --il 1 --iu {p} -n {s} --lda {ld} --ldz {ld}')


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
            yield (row, s, f'{COMMON_ARGS} -f {fn} -r {precision} --evect V --erange I --il 1 --iu {p} -n {s} --lda {ld} --ldz {ld}')


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
            yield (row, s, f'{COMMON_ARGS} -f {fn} -r {precision} --evect {v} -n {s} --lda {ld}')


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
        yield (row, s, f'{COMMON_ARGS} -f {fn} -r {precision} --evect V -n {s} --lda {ld}')


def xxevjBatch_suite(*, suite, precision, sizenormal, sizebatch):
    """
    XXEVJBATCH (SYEVJBATCH or HEEVJBATCH) tests are run, for the given precision and sizes, with vectors and without vectors. Upper case.
    """
    fn = 'syevj_strided_batched' if precision == 's' or precision == 'd' else 'heevj_strided_batched'
    size = sizebatch
    for v in ['V', 'N']:
        if v == 'V': vv = 'vect'
        else: vv = 'novect'
        for s, bc in size:
            if s < 4000: ld = s + 1
            else: ld = s + 64
            row = {'name': precision+suite, 'name_test': suite, 'function': fn, 'precision': precision, 'batch_count': bc, 'evect': vv, 'n': s}
            yield (row, s, f'{COMMON_ARGS} -f {fn} -r {precision} --batch_count {bc} --evect {v} -n {s} --lda {ld}')


def gesvd_suite(*, suite, precision, sizenormal, sizebatch):
    """
    GESVD tests are run, for the given precision and sizes, with vectors and without vectors (only square case).
    Tests are run with the hybrid approach as well. 
    """
    fn = 'gesvd'
    size = sizenormal
    for alg in [1 ,0]:
        if alg == 0: hyb = 'normal'
        else: hyb = 'hybrid'
        for v in ['V', 'N']:
            if v == 'V': vv = 'vect'
            else: vv = 'novect'
            for s in size:
                if s < 4000: ld = s + 1
                else: ld = s + 64
                row = {'name': precision+suite, 'name_test': suite, 'function': fn, 'precision': precision, 'hybrid': hyb, 'svect': vv, 'n': s}
                yield (row, s, f'{COMMON_ARGS} -f {fn} -r {precision} --alg_mode {alg} --left_svect {v} --right_svect {v} -m {s} --lda {ld} --ldu {ld} --ldv {ld}')


def gesdd_suite(*, suite, precision, sizenormal, sizebatch):
    """
    GESDD tests are run, for the given precision and sizes, with vectors and without vectors (only square case).
    """
    fn = 'gesdd'
    size = sizenormal
    for v in ['V', 'N']:
        if v == 'V': vv = 'vect'
        else: vv = 'novect'
        for s in size:
            if s < 4000: ld = s + 1
            else: ld = s + 64
            row = {'name': precision+suite, 'name_test': suite, 'function': fn, 'precision': precision, 'svect': vv, 'n': s}
            yield (row, s, f'{COMMON_ARGS} -f {fn} -r {precision} --left_svect {v} --right_svect {v} -m {s} --lda {ld} --ldu {ld} --ldv {ld}')


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
            yield (row, s, f'{COMMON_ARGS} -f {fn} -r {precision} --left_svect {v} --right_svect {v} -m {s} --lda {ld} --ldu {ld} --ldv {ld}')


def gesvdjBatch_suite(*, suite, precision, sizenormal, sizebatch):
    """
    GESVDJBATCH tests are run, for the given precision and sizes, with vectors and without vectors (only square case).
    """
    fn = 'gesvdj_strided_batched'
    size = sizebatch
    for v in ['V', 'N']:
        if v == 'V': vv = 'vect'
        else: vv = 'novect'
        for s, bc in size:
            if s < 4000: ld = s + 1
            else: ld = s + 64
            row = {'name': precision+suite, 'name_test': suite, 'function': fn, 'precision': precision, 'batch_count': bc, 'evect': vv, 'n': s}
            yield (row, s, f'{COMMON_ARGS} -f {fn} -r {precision} --batch_count {bc} --left_svect {v} --right_svect {v} -m {s} --lda {ld} --ldu {ld} --ldv {ld}')


# Registry of all available benchmark suites
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
