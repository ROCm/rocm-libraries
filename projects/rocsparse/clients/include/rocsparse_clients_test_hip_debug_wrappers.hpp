/*! \file */
/* ************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */
#pragma once
#ifdef GOOGLE_TEST

#include "rocsparse-auxiliary.h"
#include "rocsparse-debugging.h"
#include "rocsparse-functions.h"
#include "rocsparse_clients_test_hip_debug.hpp"

#define ROCSPARSE_CLIENTS_TEST_WRAP(NAME)                                           \
    inline auto rocsparse_real_##NAME##_ptr = &::rocsparse_##NAME;                  \
    struct rocsparse_wrap_##NAME##_t                                                \
    {                                                                               \
        template <typename... P>                                                    \
        static inline rocsparse_status apply(rocsparse_handle handle, P... p)       \
        {                                                                           \
            rocsparse_status status;                                                \
            if(rocsparse_clients_test::hip_debug_t::instance().enabled())           \
            {                                                                       \
                status = rocsparse_hip_debug_start(handle, nullptr);                \
                if(status != rocsparse_status_success)                              \
                    return status;                                                  \
            }                                                                       \
            status = rocsparse_real_##NAME##_ptr(handle, p...);                     \
            if(status != rocsparse_status_success)                                  \
                return status;                                                      \
            if(rocsparse_clients_test::hip_debug_t::instance().enabled())           \
            {                                                                       \
                rocsparse_clients_test::hip_debug_check_api(handle, #NAME);         \
            }                                                                       \
            return status;                                                          \
        }                                                                           \
        template <typename... P>                                                    \
        inline rocsparse_status operator()(rocsparse_handle handle, P&&... p) const \
        {                                                                           \
            rocsparse_status status;                                                \
            if(rocsparse_clients_test::hip_debug_t::instance().enabled())           \
            {                                                                       \
                status = rocsparse_hip_debug_start(handle, nullptr);                \
                if(status != rocsparse_status_success)                              \
                    return status;                                                  \
            }                                                                       \
            status = rocsparse_real_##NAME##_ptr(handle, p...);                     \
            if(status != rocsparse_status_success)                                  \
                return status;                                                      \
            if(rocsparse_clients_test::hip_debug_t::instance().enabled())           \
            {                                                                       \
                rocsparse_clients_test::hip_debug_check_api(handle, #NAME);         \
            }                                                                       \
            return status;                                                          \
        }                                                                           \
    };                                                                              \
    inline rocsparse_wrap_##NAME##_t rocsparse_test_wrap_##NAME

#define ROCSPARSE_CLIENTS_TEST_WRAP_NO_HANDLE(NAME)                          \
    inline auto rocsparse_real_##NAME##_ptr = &::rocsparse_##NAME;           \
    struct rocsparse_wrap_##NAME##_t                                         \
    {                                                                        \
        template <typename... P>                                             \
        static inline rocsparse_status apply(P... p)                         \
        {                                                                    \
            rocsparse_status status;                                         \
            if(rocsparse_clients_test::hip_debug_t::instance().enabled())    \
            {                                                                \
                status = rocsparse_hip_debug_start(nullptr, nullptr);        \
                if(status != rocsparse_status_success)                       \
                    return status;                                           \
            }                                                                \
            status = rocsparse_real_##NAME##_ptr(p...);                      \
            if(status != rocsparse_status_success)                           \
                return status;                                               \
            if(rocsparse_clients_test::hip_debug_t::instance().enabled())    \
            {                                                                \
                rocsparse_clients_test::hip_debug_check_api(nullptr, #NAME); \
            }                                                                \
            return status;                                                   \
        }                                                                    \
        template <typename... P>                                             \
        inline rocsparse_status operator()(P&&... p) const                   \
        {                                                                    \
            rocsparse_status status;                                         \
            if(rocsparse_clients_test::hip_debug_t::instance().enabled())    \
            {                                                                \
                status = rocsparse_hip_debug_start(nullptr, nullptr);        \
                if(status != rocsparse_status_success)                       \
                    return status;                                           \
            }                                                                \
            status = rocsparse_real_##NAME##_ptr(p...);                      \
            if(status != rocsparse_status_success)                           \
                return status;                                               \
            if(rocsparse_clients_test::hip_debug_t::instance().enabled())    \
            {                                                                \
                rocsparse_clients_test::hip_debug_check_api(nullptr, #NAME); \
            }                                                                \
            return status;                                                   \
        }                                                                    \
    };                                                                       \
    inline rocsparse_wrap_##NAME##_t rocsparse_test_wrap_##NAME

ROCSPARSE_CLIENTS_TEST_WRAP(axpby);
ROCSPARSE_CLIENTS_TEST_WRAP(bsrgeam_nnzb);
ROCSPARSE_CLIENTS_TEST_WRAP(bsrgemm_nnzb);
ROCSPARSE_CLIENTS_TEST_WRAP(bsric0_clear);
ROCSPARSE_CLIENTS_TEST_WRAP(bsric0_zero_pivot);
ROCSPARSE_CLIENTS_TEST_WRAP(bsrilu0_clear);
ROCSPARSE_CLIENTS_TEST_WRAP(bsrilu0_zero_pivot);
ROCSPARSE_CLIENTS_TEST_WRAP(bsrmv_clear);
ROCSPARSE_CLIENTS_TEST_WRAP(bsrsm_clear);
ROCSPARSE_CLIENTS_TEST_WRAP(bsrsm_zero_pivot);
ROCSPARSE_CLIENTS_TEST_WRAP(bsrsv_clear);
ROCSPARSE_CLIENTS_TEST_WRAP(bsrsv_zero_pivot);
ROCSPARSE_CLIENTS_TEST_WRAP(caxpyi);
ROCSPARSE_CLIENTS_TEST_WRAP(cbsr2csr);
ROCSPARSE_CLIENTS_TEST_WRAP(cbsrgeam);
ROCSPARSE_CLIENTS_TEST_WRAP(cbsrgemm);
ROCSPARSE_CLIENTS_TEST_WRAP(cbsrgemm_buffer_size);
ROCSPARSE_CLIENTS_TEST_WRAP(cbsric0);
ROCSPARSE_CLIENTS_TEST_WRAP(cbsric0_analysis);
ROCSPARSE_CLIENTS_TEST_WRAP(cbsric0_buffer_size);
ROCSPARSE_CLIENTS_TEST_WRAP(cbsrilu0);
ROCSPARSE_CLIENTS_TEST_WRAP(cbsrilu0_analysis);
ROCSPARSE_CLIENTS_TEST_WRAP(cbsrilu0_buffer_size);
ROCSPARSE_CLIENTS_TEST_WRAP(cbsrilu0_numeric_boost);
ROCSPARSE_CLIENTS_TEST_WRAP(cbsrmm);
ROCSPARSE_CLIENTS_TEST_WRAP(cbsrmv);
ROCSPARSE_CLIENTS_TEST_WRAP(cbsrmv_analysis);
ROCSPARSE_CLIENTS_TEST_WRAP(cbsrpad_value);
ROCSPARSE_CLIENTS_TEST_WRAP(cbsrsm_analysis);
ROCSPARSE_CLIENTS_TEST_WRAP(cbsrsm_buffer_size);
ROCSPARSE_CLIENTS_TEST_WRAP(cbsrsm_solve);
ROCSPARSE_CLIENTS_TEST_WRAP(cbsrsv_analysis);
ROCSPARSE_CLIENTS_TEST_WRAP(cbsrsv_buffer_size);
ROCSPARSE_CLIENTS_TEST_WRAP(cbsrsv_solve);
ROCSPARSE_CLIENTS_TEST_WRAP(cbsrxmv);
ROCSPARSE_CLIENTS_TEST_WRAP(ccheck_matrix_coo);
ROCSPARSE_CLIENTS_TEST_WRAP(ccheck_matrix_coo_buffer_size);
ROCSPARSE_CLIENTS_TEST_WRAP(ccheck_matrix_csc);
ROCSPARSE_CLIENTS_TEST_WRAP(ccheck_matrix_csc_buffer_size);
ROCSPARSE_CLIENTS_TEST_WRAP(ccheck_matrix_csr);
ROCSPARSE_CLIENTS_TEST_WRAP(ccheck_matrix_csr_buffer_size);
ROCSPARSE_CLIENTS_TEST_WRAP(ccheck_matrix_ell);
ROCSPARSE_CLIENTS_TEST_WRAP(ccheck_matrix_ell_buffer_size);
ROCSPARSE_CLIENTS_TEST_WRAP(ccheck_matrix_gebsc);
ROCSPARSE_CLIENTS_TEST_WRAP(ccheck_matrix_gebsc_buffer_size);
ROCSPARSE_CLIENTS_TEST_WRAP(ccheck_matrix_gebsr);
ROCSPARSE_CLIENTS_TEST_WRAP(ccheck_matrix_gebsr_buffer_size);
ROCSPARSE_CLIENTS_TEST_WRAP(ccoo2dense);
ROCSPARSE_CLIENTS_TEST_WRAP(ccoomv);
ROCSPARSE_CLIENTS_TEST_WRAP(ccsc2dense);
ROCSPARSE_CLIENTS_TEST_WRAP(ccsr2bsr);
ROCSPARSE_CLIENTS_TEST_WRAP(ccsr2csc);
ROCSPARSE_CLIENTS_TEST_WRAP(ccsr2csr_compress);
ROCSPARSE_CLIENTS_TEST_WRAP(ccsr2dense);
ROCSPARSE_CLIENTS_TEST_WRAP(ccsr2ell);
ROCSPARSE_CLIENTS_TEST_WRAP(ccsr2gebsr);
ROCSPARSE_CLIENTS_TEST_WRAP(ccsr2gebsr_buffer_size);
ROCSPARSE_CLIENTS_TEST_WRAP(ccsr2hyb);
ROCSPARSE_CLIENTS_TEST_WRAP(ccsrcolor);
ROCSPARSE_CLIENTS_TEST_WRAP(ccsrgeam);
ROCSPARSE_CLIENTS_TEST_WRAP(ccsrgemm);
ROCSPARSE_CLIENTS_TEST_WRAP(ccsrgemm_buffer_size);
ROCSPARSE_CLIENTS_TEST_WRAP(ccsrgemm_numeric);
ROCSPARSE_CLIENTS_TEST_WRAP(ccsric0);
ROCSPARSE_CLIENTS_TEST_WRAP(ccsric0_analysis);
ROCSPARSE_CLIENTS_TEST_WRAP(ccsric0_buffer_size);
ROCSPARSE_CLIENTS_TEST_WRAP(ccsrilu0);
ROCSPARSE_CLIENTS_TEST_WRAP(ccsrilu0_analysis);
ROCSPARSE_CLIENTS_TEST_WRAP(ccsrilu0_buffer_size);
ROCSPARSE_CLIENTS_TEST_WRAP(ccsrilu0_numeric_boost);
ROCSPARSE_CLIENTS_TEST_WRAP(ccsritilu0_compute);
ROCSPARSE_CLIENTS_TEST_WRAP(ccsritilu0_compute_ex);
ROCSPARSE_CLIENTS_TEST_WRAP(ccsritilu0_history);
ROCSPARSE_CLIENTS_TEST_WRAP(ccsritsv_analysis);
ROCSPARSE_CLIENTS_TEST_WRAP(ccsritsv_buffer_size);
ROCSPARSE_CLIENTS_TEST_WRAP(ccsritsv_solve);
ROCSPARSE_CLIENTS_TEST_WRAP(ccsritsv_solve_ex);
ROCSPARSE_CLIENTS_TEST_WRAP(ccsrmm);
ROCSPARSE_CLIENTS_TEST_WRAP(ccsrmv);
ROCSPARSE_CLIENTS_TEST_WRAP(ccsrmv_analysis);
ROCSPARSE_CLIENTS_TEST_WRAP(ccsrsm_analysis);
ROCSPARSE_CLIENTS_TEST_WRAP(ccsrsm_buffer_size);
ROCSPARSE_CLIENTS_TEST_WRAP(ccsrsm_solve);
ROCSPARSE_CLIENTS_TEST_WRAP(ccsrsv_analysis);
ROCSPARSE_CLIENTS_TEST_WRAP(ccsrsv_buffer_size);
ROCSPARSE_CLIENTS_TEST_WRAP(ccsrsv_solve);
ROCSPARSE_CLIENTS_TEST_WRAP(cdense2coo);
ROCSPARSE_CLIENTS_TEST_WRAP(cdense2csc);
ROCSPARSE_CLIENTS_TEST_WRAP(cdense2csr);
ROCSPARSE_CLIENTS_TEST_WRAP(cdotci);
ROCSPARSE_CLIENTS_TEST_WRAP(cdoti);
ROCSPARSE_CLIENTS_TEST_WRAP(cell2csr);
ROCSPARSE_CLIENTS_TEST_WRAP(cellmv);
ROCSPARSE_CLIENTS_TEST_WRAP(cgebsr2csr);
ROCSPARSE_CLIENTS_TEST_WRAP(cgebsr2gebsc);
ROCSPARSE_CLIENTS_TEST_WRAP(cgebsr2gebsc_buffer_size);
ROCSPARSE_CLIENTS_TEST_WRAP(cgebsr2gebsr);
ROCSPARSE_CLIENTS_TEST_WRAP(cgebsr2gebsr_buffer_size);
ROCSPARSE_CLIENTS_TEST_WRAP(cgebsrmm);
ROCSPARSE_CLIENTS_TEST_WRAP(cgebsrmv);
ROCSPARSE_CLIENTS_TEST_WRAP(cgemmi);
ROCSPARSE_CLIENTS_TEST_WRAP(cgemvi);
ROCSPARSE_CLIENTS_TEST_WRAP(cgemvi_buffer_size);
ROCSPARSE_CLIENTS_TEST_WRAP(cgpsv_interleaved_batch);
ROCSPARSE_CLIENTS_TEST_WRAP(cgpsv_interleaved_batch_buffer_size);
ROCSPARSE_CLIENTS_TEST_WRAP(cgthr);
ROCSPARSE_CLIENTS_TEST_WRAP(cgthrz);
ROCSPARSE_CLIENTS_TEST_WRAP(cgtsv);
ROCSPARSE_CLIENTS_TEST_WRAP(cgtsv_buffer_size);
ROCSPARSE_CLIENTS_TEST_WRAP(cgtsv_interleaved_batch);
ROCSPARSE_CLIENTS_TEST_WRAP(cgtsv_interleaved_batch_buffer_size);
ROCSPARSE_CLIENTS_TEST_WRAP(cgtsv_no_pivot);
ROCSPARSE_CLIENTS_TEST_WRAP(cgtsv_no_pivot_buffer_size);
ROCSPARSE_CLIENTS_TEST_WRAP(cgtsv_no_pivot_strided_batch);
ROCSPARSE_CLIENTS_TEST_WRAP(cgtsv_no_pivot_strided_batch_buffer_size);
ROCSPARSE_CLIENTS_TEST_WRAP(check_matrix_hyb);
ROCSPARSE_CLIENTS_TEST_WRAP(check_matrix_hyb_buffer_size);
ROCSPARSE_CLIENTS_TEST_WRAP(check_spmat);
ROCSPARSE_CLIENTS_TEST_WRAP(chyb2csr);
ROCSPARSE_CLIENTS_TEST_WRAP(chybmv);
ROCSPARSE_CLIENTS_TEST_WRAP(cnnz);
ROCSPARSE_CLIENTS_TEST_WRAP(cnnz_compress);
ROCSPARSE_CLIENTS_TEST_WRAP(coo2csr);
ROCSPARSE_CLIENTS_TEST_WRAP(coosort_buffer_size);
ROCSPARSE_CLIENTS_TEST_WRAP(coosort_by_column);
ROCSPARSE_CLIENTS_TEST_WRAP(coosort_by_row);
ROCSPARSE_CLIENTS_TEST_WRAP(create_identity_permutation);
ROCSPARSE_CLIENTS_TEST_WRAP(cscsort);
ROCSPARSE_CLIENTS_TEST_WRAP(cscsort_buffer_size);
ROCSPARSE_CLIENTS_TEST_WRAP(csctr);
ROCSPARSE_CLIENTS_TEST_WRAP(csr2bsr_nnz);
ROCSPARSE_CLIENTS_TEST_WRAP(csr2coo);
ROCSPARSE_CLIENTS_TEST_WRAP(csr2csc_buffer_size);
ROCSPARSE_CLIENTS_TEST_WRAP(csr2ell_width);
ROCSPARSE_CLIENTS_TEST_WRAP(csr2gebsr_nnz);
ROCSPARSE_CLIENTS_TEST_WRAP(csrgeam_nnz);
ROCSPARSE_CLIENTS_TEST_WRAP(csrgemm_nnz);
ROCSPARSE_CLIENTS_TEST_WRAP(csrgemm_symbolic);
ROCSPARSE_CLIENTS_TEST_WRAP(csric0_clear);
ROCSPARSE_CLIENTS_TEST_WRAP(csric0_get_tolerance);
ROCSPARSE_CLIENTS_TEST_WRAP(csric0_set_tolerance);
ROCSPARSE_CLIENTS_TEST_WRAP(csric0_singular_pivot);
ROCSPARSE_CLIENTS_TEST_WRAP(csric0_zero_pivot);
ROCSPARSE_CLIENTS_TEST_WRAP(csrilu0_clear);
ROCSPARSE_CLIENTS_TEST_WRAP(csrilu0_get_tolerance);
ROCSPARSE_CLIENTS_TEST_WRAP(csrilu0_set_tolerance);
ROCSPARSE_CLIENTS_TEST_WRAP(csrilu0_singular_pivot);
ROCSPARSE_CLIENTS_TEST_WRAP(csrilu0_zero_pivot);
ROCSPARSE_CLIENTS_TEST_WRAP(csritilu0_buffer_size);
ROCSPARSE_CLIENTS_TEST_WRAP(csritilu0_preprocess);
ROCSPARSE_CLIENTS_TEST_WRAP(csritsv_clear);
ROCSPARSE_CLIENTS_TEST_WRAP(csritsv_zero_pivot);
ROCSPARSE_CLIENTS_TEST_WRAP(csrmv_clear);
ROCSPARSE_CLIENTS_TEST_WRAP(csrsm_clear);
ROCSPARSE_CLIENTS_TEST_WRAP(csrsm_zero_pivot);
ROCSPARSE_CLIENTS_TEST_WRAP(csrsort);
ROCSPARSE_CLIENTS_TEST_WRAP(csrsort_buffer_size);
ROCSPARSE_CLIENTS_TEST_WRAP(csrsv_clear);
ROCSPARSE_CLIENTS_TEST_WRAP(csrsv_zero_pivot);
ROCSPARSE_CLIENTS_TEST_WRAP(daxpyi);
ROCSPARSE_CLIENTS_TEST_WRAP(dbsr2csr);
ROCSPARSE_CLIENTS_TEST_WRAP(dbsrgeam);
ROCSPARSE_CLIENTS_TEST_WRAP(dbsrgemm);
ROCSPARSE_CLIENTS_TEST_WRAP(dbsrgemm_buffer_size);
ROCSPARSE_CLIENTS_TEST_WRAP(dbsric0);
ROCSPARSE_CLIENTS_TEST_WRAP(dbsric0_analysis);
ROCSPARSE_CLIENTS_TEST_WRAP(dbsric0_buffer_size);
ROCSPARSE_CLIENTS_TEST_WRAP(dbsrilu0);
ROCSPARSE_CLIENTS_TEST_WRAP(dbsrilu0_analysis);
ROCSPARSE_CLIENTS_TEST_WRAP(dbsrilu0_buffer_size);
ROCSPARSE_CLIENTS_TEST_WRAP(dbsrilu0_numeric_boost);
ROCSPARSE_CLIENTS_TEST_WRAP(dbsrmm);
ROCSPARSE_CLIENTS_TEST_WRAP(dbsrmv);
ROCSPARSE_CLIENTS_TEST_WRAP(dbsrmv_analysis);
ROCSPARSE_CLIENTS_TEST_WRAP(dbsrpad_value);
ROCSPARSE_CLIENTS_TEST_WRAP(dbsrsm_analysis);
ROCSPARSE_CLIENTS_TEST_WRAP(dbsrsm_buffer_size);
ROCSPARSE_CLIENTS_TEST_WRAP(dbsrsm_solve);
ROCSPARSE_CLIENTS_TEST_WRAP(dbsrsv_analysis);
ROCSPARSE_CLIENTS_TEST_WRAP(dbsrsv_buffer_size);
ROCSPARSE_CLIENTS_TEST_WRAP(dbsrsv_solve);
ROCSPARSE_CLIENTS_TEST_WRAP(dbsrxmv);
ROCSPARSE_CLIENTS_TEST_WRAP(dcbsrilu0_numeric_boost);
ROCSPARSE_CLIENTS_TEST_WRAP(dccsrilu0_numeric_boost);
ROCSPARSE_CLIENTS_TEST_WRAP(dcheck_matrix_coo);
ROCSPARSE_CLIENTS_TEST_WRAP(dcheck_matrix_coo_buffer_size);
ROCSPARSE_CLIENTS_TEST_WRAP(dcheck_matrix_csc);
ROCSPARSE_CLIENTS_TEST_WRAP(dcheck_matrix_csc_buffer_size);
ROCSPARSE_CLIENTS_TEST_WRAP(dcheck_matrix_csr);
ROCSPARSE_CLIENTS_TEST_WRAP(dcheck_matrix_csr_buffer_size);
ROCSPARSE_CLIENTS_TEST_WRAP(dcheck_matrix_ell);
ROCSPARSE_CLIENTS_TEST_WRAP(dcheck_matrix_ell_buffer_size);
ROCSPARSE_CLIENTS_TEST_WRAP(dcheck_matrix_gebsc);
ROCSPARSE_CLIENTS_TEST_WRAP(dcheck_matrix_gebsc_buffer_size);
ROCSPARSE_CLIENTS_TEST_WRAP(dcheck_matrix_gebsr);
ROCSPARSE_CLIENTS_TEST_WRAP(dcheck_matrix_gebsr_buffer_size);
ROCSPARSE_CLIENTS_TEST_WRAP(dcoo2dense);
ROCSPARSE_CLIENTS_TEST_WRAP(dcoomv);
ROCSPARSE_CLIENTS_TEST_WRAP(dcsc2dense);
ROCSPARSE_CLIENTS_TEST_WRAP(dcsr2bsr);
ROCSPARSE_CLIENTS_TEST_WRAP(dcsr2csc);
ROCSPARSE_CLIENTS_TEST_WRAP(dcsr2csr_compress);
ROCSPARSE_CLIENTS_TEST_WRAP(dcsr2dense);
ROCSPARSE_CLIENTS_TEST_WRAP(dcsr2ell);
ROCSPARSE_CLIENTS_TEST_WRAP(dcsr2gebsr);
ROCSPARSE_CLIENTS_TEST_WRAP(dcsr2gebsr_buffer_size);
ROCSPARSE_CLIENTS_TEST_WRAP(dcsr2hyb);
ROCSPARSE_CLIENTS_TEST_WRAP(dcsrcolor);
ROCSPARSE_CLIENTS_TEST_WRAP(dcsrgeam);
ROCSPARSE_CLIENTS_TEST_WRAP(dcsrgemm);
ROCSPARSE_CLIENTS_TEST_WRAP(dcsrgemm_buffer_size);
ROCSPARSE_CLIENTS_TEST_WRAP(dcsrgemm_numeric);
ROCSPARSE_CLIENTS_TEST_WRAP(dcsric0);
ROCSPARSE_CLIENTS_TEST_WRAP(dcsric0_analysis);
ROCSPARSE_CLIENTS_TEST_WRAP(dcsric0_buffer_size);
ROCSPARSE_CLIENTS_TEST_WRAP(dcsrilu0);
ROCSPARSE_CLIENTS_TEST_WRAP(dcsrilu0_analysis);
ROCSPARSE_CLIENTS_TEST_WRAP(dcsrilu0_buffer_size);
ROCSPARSE_CLIENTS_TEST_WRAP(dcsrilu0_numeric_boost);
ROCSPARSE_CLIENTS_TEST_WRAP(dcsritilu0_compute);
ROCSPARSE_CLIENTS_TEST_WRAP(dcsritilu0_compute_ex);
ROCSPARSE_CLIENTS_TEST_WRAP(dcsritilu0_history);
ROCSPARSE_CLIENTS_TEST_WRAP(dcsritsv_analysis);
ROCSPARSE_CLIENTS_TEST_WRAP(dcsritsv_buffer_size);
ROCSPARSE_CLIENTS_TEST_WRAP(dcsritsv_solve);
ROCSPARSE_CLIENTS_TEST_WRAP(dcsritsv_solve_ex);
ROCSPARSE_CLIENTS_TEST_WRAP(dcsrmm);
ROCSPARSE_CLIENTS_TEST_WRAP(dcsrmv);
ROCSPARSE_CLIENTS_TEST_WRAP(dcsrmv_analysis);
ROCSPARSE_CLIENTS_TEST_WRAP(dcsrsm_analysis);
ROCSPARSE_CLIENTS_TEST_WRAP(dcsrsm_buffer_size);
ROCSPARSE_CLIENTS_TEST_WRAP(dcsrsm_solve);
ROCSPARSE_CLIENTS_TEST_WRAP(dcsrsv_analysis);
ROCSPARSE_CLIENTS_TEST_WRAP(dcsrsv_buffer_size);
ROCSPARSE_CLIENTS_TEST_WRAP(dcsrsv_solve);
ROCSPARSE_CLIENTS_TEST_WRAP(ddense2coo);
ROCSPARSE_CLIENTS_TEST_WRAP(ddense2csc);
ROCSPARSE_CLIENTS_TEST_WRAP(ddense2csr);
ROCSPARSE_CLIENTS_TEST_WRAP(ddoti);
ROCSPARSE_CLIENTS_TEST_WRAP(dell2csr);
ROCSPARSE_CLIENTS_TEST_WRAP(dellmv);
ROCSPARSE_CLIENTS_TEST_WRAP(dense_to_sparse);
ROCSPARSE_CLIENTS_TEST_WRAP(dgebsr2csr);
ROCSPARSE_CLIENTS_TEST_WRAP(dgebsr2gebsc);
ROCSPARSE_CLIENTS_TEST_WRAP(dgebsr2gebsc_buffer_size);
ROCSPARSE_CLIENTS_TEST_WRAP(dgebsr2gebsr);
ROCSPARSE_CLIENTS_TEST_WRAP(dgebsr2gebsr_buffer_size);
ROCSPARSE_CLIENTS_TEST_WRAP(dgebsrmm);
ROCSPARSE_CLIENTS_TEST_WRAP(dgebsrmv);
ROCSPARSE_CLIENTS_TEST_WRAP(dgemmi);
ROCSPARSE_CLIENTS_TEST_WRAP(dgemvi);
ROCSPARSE_CLIENTS_TEST_WRAP(dgemvi_buffer_size);
ROCSPARSE_CLIENTS_TEST_WRAP(dgpsv_interleaved_batch);
ROCSPARSE_CLIENTS_TEST_WRAP(dgpsv_interleaved_batch_buffer_size);
ROCSPARSE_CLIENTS_TEST_WRAP(dgthr);
ROCSPARSE_CLIENTS_TEST_WRAP(dgthrz);
ROCSPARSE_CLIENTS_TEST_WRAP(dgtsv);
ROCSPARSE_CLIENTS_TEST_WRAP(dgtsv_buffer_size);
ROCSPARSE_CLIENTS_TEST_WRAP(dgtsv_interleaved_batch);
ROCSPARSE_CLIENTS_TEST_WRAP(dgtsv_interleaved_batch_buffer_size);
ROCSPARSE_CLIENTS_TEST_WRAP(dgtsv_no_pivot);
ROCSPARSE_CLIENTS_TEST_WRAP(dgtsv_no_pivot_buffer_size);
ROCSPARSE_CLIENTS_TEST_WRAP(dgtsv_no_pivot_strided_batch);
ROCSPARSE_CLIENTS_TEST_WRAP(dgtsv_no_pivot_strided_batch_buffer_size);
ROCSPARSE_CLIENTS_TEST_WRAP(dhyb2csr);
ROCSPARSE_CLIENTS_TEST_WRAP(dhybmv);
ROCSPARSE_CLIENTS_TEST_WRAP(dnnz);
ROCSPARSE_CLIENTS_TEST_WRAP(dnnz_compress);
ROCSPARSE_CLIENTS_TEST_WRAP(dprune_csr2csr);
ROCSPARSE_CLIENTS_TEST_WRAP(dprune_csr2csr_buffer_size);
ROCSPARSE_CLIENTS_TEST_WRAP(dprune_csr2csr_by_percentage);
ROCSPARSE_CLIENTS_TEST_WRAP(dprune_csr2csr_by_percentage_buffer_size);
ROCSPARSE_CLIENTS_TEST_WRAP(dprune_csr2csr_nnz);
ROCSPARSE_CLIENTS_TEST_WRAP(dprune_csr2csr_nnz_by_percentage);
ROCSPARSE_CLIENTS_TEST_WRAP(dprune_dense2csr);
ROCSPARSE_CLIENTS_TEST_WRAP(dprune_dense2csr_buffer_size);
ROCSPARSE_CLIENTS_TEST_WRAP(dprune_dense2csr_by_percentage);
ROCSPARSE_CLIENTS_TEST_WRAP(dprune_dense2csr_by_percentage_buffer_size);
ROCSPARSE_CLIENTS_TEST_WRAP(dprune_dense2csr_nnz);
ROCSPARSE_CLIENTS_TEST_WRAP(dprune_dense2csr_nnz_by_percentage);
ROCSPARSE_CLIENTS_TEST_WRAP(droti);
ROCSPARSE_CLIENTS_TEST_WRAP(dsbsrilu0_numeric_boost);
ROCSPARSE_CLIENTS_TEST_WRAP(dscsrilu0_numeric_boost);
ROCSPARSE_CLIENTS_TEST_WRAP(dsctr);
ROCSPARSE_CLIENTS_TEST_WRAP(ell2csr_nnz);
ROCSPARSE_CLIENTS_TEST_WRAP(extract);
ROCSPARSE_CLIENTS_TEST_WRAP(extract_buffer_size);
ROCSPARSE_CLIENTS_TEST_WRAP(extract_nnz);
ROCSPARSE_CLIENTS_TEST_WRAP(gather);
ROCSPARSE_CLIENTS_TEST_WRAP(gebsr2gebsr_nnz);
ROCSPARSE_CLIENTS_TEST_WRAP(get_git_rev);
ROCSPARSE_CLIENTS_TEST_WRAP(get_pointer_mode);
ROCSPARSE_CLIENTS_TEST_WRAP(get_stream);
ROCSPARSE_CLIENTS_TEST_WRAP(get_version);
ROCSPARSE_CLIENTS_TEST_WRAP(hyb2csr_buffer_size);
ROCSPARSE_CLIENTS_TEST_WRAP(inverse_permutation);
ROCSPARSE_CLIENTS_TEST_WRAP(isctr);
ROCSPARSE_CLIENTS_TEST_WRAP(rot);
ROCSPARSE_CLIENTS_TEST_WRAP(saxpyi);
ROCSPARSE_CLIENTS_TEST_WRAP(sbsr2csr);
ROCSPARSE_CLIENTS_TEST_WRAP(sbsrgeam);
ROCSPARSE_CLIENTS_TEST_WRAP(sbsrgemm);
ROCSPARSE_CLIENTS_TEST_WRAP(sbsrgemm_buffer_size);
ROCSPARSE_CLIENTS_TEST_WRAP(sbsric0);
ROCSPARSE_CLIENTS_TEST_WRAP(sbsric0_analysis);
ROCSPARSE_CLIENTS_TEST_WRAP(sbsric0_buffer_size);
ROCSPARSE_CLIENTS_TEST_WRAP(sbsrilu0);
ROCSPARSE_CLIENTS_TEST_WRAP(sbsrilu0_analysis);
ROCSPARSE_CLIENTS_TEST_WRAP(sbsrilu0_buffer_size);
ROCSPARSE_CLIENTS_TEST_WRAP(sbsrilu0_numeric_boost);
ROCSPARSE_CLIENTS_TEST_WRAP(sbsrmm);
ROCSPARSE_CLIENTS_TEST_WRAP(sbsrmv);
ROCSPARSE_CLIENTS_TEST_WRAP(sbsrmv_analysis);
ROCSPARSE_CLIENTS_TEST_WRAP(sbsrpad_value);
ROCSPARSE_CLIENTS_TEST_WRAP(sbsrsm_analysis);
ROCSPARSE_CLIENTS_TEST_WRAP(sbsrsm_buffer_size);
ROCSPARSE_CLIENTS_TEST_WRAP(sbsrsm_solve);
ROCSPARSE_CLIENTS_TEST_WRAP(sbsrsv_analysis);
ROCSPARSE_CLIENTS_TEST_WRAP(sbsrsv_buffer_size);
ROCSPARSE_CLIENTS_TEST_WRAP(sbsrsv_solve);
ROCSPARSE_CLIENTS_TEST_WRAP(sbsrxmv);
ROCSPARSE_CLIENTS_TEST_WRAP(scatter);
ROCSPARSE_CLIENTS_TEST_WRAP(scheck_matrix_coo);
ROCSPARSE_CLIENTS_TEST_WRAP(scheck_matrix_coo_buffer_size);
ROCSPARSE_CLIENTS_TEST_WRAP(scheck_matrix_csc);
ROCSPARSE_CLIENTS_TEST_WRAP(scheck_matrix_csc_buffer_size);
ROCSPARSE_CLIENTS_TEST_WRAP(scheck_matrix_csr);
ROCSPARSE_CLIENTS_TEST_WRAP(scheck_matrix_csr_buffer_size);
ROCSPARSE_CLIENTS_TEST_WRAP(scheck_matrix_ell);
ROCSPARSE_CLIENTS_TEST_WRAP(scheck_matrix_ell_buffer_size);
ROCSPARSE_CLIENTS_TEST_WRAP(scheck_matrix_gebsc);
ROCSPARSE_CLIENTS_TEST_WRAP(scheck_matrix_gebsc_buffer_size);
ROCSPARSE_CLIENTS_TEST_WRAP(scheck_matrix_gebsr);
ROCSPARSE_CLIENTS_TEST_WRAP(scheck_matrix_gebsr_buffer_size);
ROCSPARSE_CLIENTS_TEST_WRAP(scoo2dense);
ROCSPARSE_CLIENTS_TEST_WRAP(scoomv);
ROCSPARSE_CLIENTS_TEST_WRAP(scsc2dense);
ROCSPARSE_CLIENTS_TEST_WRAP(scsr2bsr);
ROCSPARSE_CLIENTS_TEST_WRAP(scsr2csc);
ROCSPARSE_CLIENTS_TEST_WRAP(scsr2csr_compress);
ROCSPARSE_CLIENTS_TEST_WRAP(scsr2dense);
ROCSPARSE_CLIENTS_TEST_WRAP(scsr2ell);
ROCSPARSE_CLIENTS_TEST_WRAP(scsr2gebsr);
ROCSPARSE_CLIENTS_TEST_WRAP(scsr2gebsr_buffer_size);
ROCSPARSE_CLIENTS_TEST_WRAP(scsr2hyb);
ROCSPARSE_CLIENTS_TEST_WRAP(scsrcolor);
ROCSPARSE_CLIENTS_TEST_WRAP(scsrgeam);
ROCSPARSE_CLIENTS_TEST_WRAP(scsrgemm);
ROCSPARSE_CLIENTS_TEST_WRAP(scsrgemm_buffer_size);
ROCSPARSE_CLIENTS_TEST_WRAP(scsrgemm_numeric);
ROCSPARSE_CLIENTS_TEST_WRAP(scsric0);
ROCSPARSE_CLIENTS_TEST_WRAP(scsric0_analysis);
ROCSPARSE_CLIENTS_TEST_WRAP(scsric0_buffer_size);
ROCSPARSE_CLIENTS_TEST_WRAP(scsrilu0);
ROCSPARSE_CLIENTS_TEST_WRAP(scsrilu0_analysis);
ROCSPARSE_CLIENTS_TEST_WRAP(scsrilu0_buffer_size);
ROCSPARSE_CLIENTS_TEST_WRAP(scsrilu0_numeric_boost);
ROCSPARSE_CLIENTS_TEST_WRAP(scsritilu0_compute);
ROCSPARSE_CLIENTS_TEST_WRAP(scsritilu0_compute_ex);
ROCSPARSE_CLIENTS_TEST_WRAP(scsritilu0_history);
ROCSPARSE_CLIENTS_TEST_WRAP(scsritsv_analysis);
ROCSPARSE_CLIENTS_TEST_WRAP(scsritsv_buffer_size);
ROCSPARSE_CLIENTS_TEST_WRAP(scsritsv_solve);
ROCSPARSE_CLIENTS_TEST_WRAP(scsritsv_solve_ex);
ROCSPARSE_CLIENTS_TEST_WRAP(scsrmm);
ROCSPARSE_CLIENTS_TEST_WRAP(scsrmv);
ROCSPARSE_CLIENTS_TEST_WRAP(scsrmv_analysis);
ROCSPARSE_CLIENTS_TEST_WRAP(scsrsm_analysis);
ROCSPARSE_CLIENTS_TEST_WRAP(scsrsm_buffer_size);
ROCSPARSE_CLIENTS_TEST_WRAP(scsrsm_solve);
ROCSPARSE_CLIENTS_TEST_WRAP(scsrsv_analysis);
ROCSPARSE_CLIENTS_TEST_WRAP(scsrsv_buffer_size);
ROCSPARSE_CLIENTS_TEST_WRAP(scsrsv_solve);
ROCSPARSE_CLIENTS_TEST_WRAP(sddmm);
ROCSPARSE_CLIENTS_TEST_WRAP(sddmm_buffer_size);
ROCSPARSE_CLIENTS_TEST_WRAP(sddmm_preprocess);
ROCSPARSE_CLIENTS_TEST_WRAP(sdense2coo);
ROCSPARSE_CLIENTS_TEST_WRAP(sdense2csc);
ROCSPARSE_CLIENTS_TEST_WRAP(sdense2csr);
ROCSPARSE_CLIENTS_TEST_WRAP(sdoti);
ROCSPARSE_CLIENTS_TEST_WRAP(sell2csr);
ROCSPARSE_CLIENTS_TEST_WRAP(sellmv);
ROCSPARSE_CLIENTS_TEST_WRAP(set_identity_permutation);
ROCSPARSE_CLIENTS_TEST_WRAP(set_pointer_mode);
ROCSPARSE_CLIENTS_TEST_WRAP(set_stream);
ROCSPARSE_CLIENTS_TEST_WRAP(sgebsr2csr);
ROCSPARSE_CLIENTS_TEST_WRAP(sgebsr2gebsc);
ROCSPARSE_CLIENTS_TEST_WRAP(sgebsr2gebsc_buffer_size);
ROCSPARSE_CLIENTS_TEST_WRAP(sgebsr2gebsr);
ROCSPARSE_CLIENTS_TEST_WRAP(sgebsr2gebsr_buffer_size);
ROCSPARSE_CLIENTS_TEST_WRAP(sgebsrmm);
ROCSPARSE_CLIENTS_TEST_WRAP(sgebsrmv);
ROCSPARSE_CLIENTS_TEST_WRAP(sgemmi);
ROCSPARSE_CLIENTS_TEST_WRAP(sgemvi);
ROCSPARSE_CLIENTS_TEST_WRAP(sgemvi_buffer_size);
ROCSPARSE_CLIENTS_TEST_WRAP(sgpsv_interleaved_batch);
ROCSPARSE_CLIENTS_TEST_WRAP(sgpsv_interleaved_batch_buffer_size);
ROCSPARSE_CLIENTS_TEST_WRAP(sgthr);
ROCSPARSE_CLIENTS_TEST_WRAP(sgthrz);
ROCSPARSE_CLIENTS_TEST_WRAP(sgtsv);
ROCSPARSE_CLIENTS_TEST_WRAP(sgtsv_buffer_size);
ROCSPARSE_CLIENTS_TEST_WRAP(sgtsv_interleaved_batch);
ROCSPARSE_CLIENTS_TEST_WRAP(sgtsv_interleaved_batch_buffer_size);
ROCSPARSE_CLIENTS_TEST_WRAP(sgtsv_no_pivot);
ROCSPARSE_CLIENTS_TEST_WRAP(sgtsv_no_pivot_buffer_size);
ROCSPARSE_CLIENTS_TEST_WRAP(sgtsv_no_pivot_strided_batch);
ROCSPARSE_CLIENTS_TEST_WRAP(sgtsv_no_pivot_strided_batch_buffer_size);
ROCSPARSE_CLIENTS_TEST_WRAP(shyb2csr);
ROCSPARSE_CLIENTS_TEST_WRAP(shybmv);
ROCSPARSE_CLIENTS_TEST_WRAP(snnz);
ROCSPARSE_CLIENTS_TEST_WRAP(snnz_compress);
ROCSPARSE_CLIENTS_TEST_WRAP(sparse_to_dense);
ROCSPARSE_CLIENTS_TEST_WRAP(sparse_to_sparse);
ROCSPARSE_CLIENTS_TEST_WRAP(sparse_to_sparse_buffer_size);
ROCSPARSE_CLIENTS_TEST_WRAP(spgeam);
ROCSPARSE_CLIENTS_TEST_WRAP(spgeam_buffer_size);
ROCSPARSE_CLIENTS_TEST_WRAP(spgeam_get_output);
ROCSPARSE_CLIENTS_TEST_WRAP(spgeam_set_input);
ROCSPARSE_CLIENTS_TEST_WRAP(spgemm);
ROCSPARSE_CLIENTS_TEST_WRAP(spic0);
ROCSPARSE_CLIENTS_TEST_WRAP(spic0_buffer_size);
ROCSPARSE_CLIENTS_TEST_WRAP(spic0_descr_create);
ROCSPARSE_CLIENTS_TEST_WRAP(spic0_descr_destroy);
ROCSPARSE_CLIENTS_TEST_WRAP(spic0_get_output);
ROCSPARSE_CLIENTS_TEST_WRAP(spic0_set_input);
ROCSPARSE_CLIENTS_TEST_WRAP(spilu0);
ROCSPARSE_CLIENTS_TEST_WRAP(spilu0_buffer_size);
ROCSPARSE_CLIENTS_TEST_WRAP(spilu0_descr_create);
ROCSPARSE_CLIENTS_TEST_WRAP(spilu0_descr_destroy);
ROCSPARSE_CLIENTS_TEST_WRAP(spilu0_get_output);
ROCSPARSE_CLIENTS_TEST_WRAP(spilu0_set_input);
ROCSPARSE_CLIENTS_TEST_WRAP(spitsv);
ROCSPARSE_CLIENTS_TEST_WRAP(spmm);
ROCSPARSE_CLIENTS_TEST_WRAP(spmv);
ROCSPARSE_CLIENTS_TEST_WRAP(spmv_clear_extra);
ROCSPARSE_CLIENTS_TEST_WRAP(spmv_set_extra);
ROCSPARSE_CLIENTS_TEST_WRAP(spmv_set_input);
ROCSPARSE_CLIENTS_TEST_WRAP(sprune_csr2csr);
ROCSPARSE_CLIENTS_TEST_WRAP(sprune_csr2csr_buffer_size);
ROCSPARSE_CLIENTS_TEST_WRAP(sprune_csr2csr_by_percentage);
ROCSPARSE_CLIENTS_TEST_WRAP(sprune_csr2csr_by_percentage_buffer_size);
ROCSPARSE_CLIENTS_TEST_WRAP(sprune_csr2csr_nnz);
ROCSPARSE_CLIENTS_TEST_WRAP(sprune_csr2csr_nnz_by_percentage);
ROCSPARSE_CLIENTS_TEST_WRAP(sprune_dense2csr);
ROCSPARSE_CLIENTS_TEST_WRAP(sprune_dense2csr_buffer_size);
ROCSPARSE_CLIENTS_TEST_WRAP(sprune_dense2csr_by_percentage);
ROCSPARSE_CLIENTS_TEST_WRAP(sprune_dense2csr_by_percentage_buffer_size);
ROCSPARSE_CLIENTS_TEST_WRAP(sprune_dense2csr_nnz);
ROCSPARSE_CLIENTS_TEST_WRAP(sprune_dense2csr_nnz_by_percentage);
ROCSPARSE_CLIENTS_TEST_WRAP(spsm);
ROCSPARSE_CLIENTS_TEST_WRAP(spsv);
ROCSPARSE_CLIENTS_TEST_WRAP(sptrsm);
ROCSPARSE_CLIENTS_TEST_WRAP(sptrsm_buffer_size);
ROCSPARSE_CLIENTS_TEST_WRAP(sptrsm_get_output);
ROCSPARSE_CLIENTS_TEST_WRAP(sptrsm_set_input);
ROCSPARSE_CLIENTS_TEST_WRAP(sptrsv);
ROCSPARSE_CLIENTS_TEST_WRAP(sptrsv_buffer_size);
ROCSPARSE_CLIENTS_TEST_WRAP(sptrsv_descr_create);
ROCSPARSE_CLIENTS_TEST_WRAP(sptrsv_descr_destroy);
ROCSPARSE_CLIENTS_TEST_WRAP(sptrsv_get_output);
ROCSPARSE_CLIENTS_TEST_WRAP(sptrsv_set_input);
ROCSPARSE_CLIENTS_TEST_WRAP(spvv);
ROCSPARSE_CLIENTS_TEST_WRAP(sroti);
ROCSPARSE_CLIENTS_TEST_WRAP(ssctr);
ROCSPARSE_CLIENTS_TEST_WRAP(v2_spmv);
ROCSPARSE_CLIENTS_TEST_WRAP(v2_spmv_buffer_size);
ROCSPARSE_CLIENTS_TEST_WRAP(zaxpyi);
ROCSPARSE_CLIENTS_TEST_WRAP(zbsr2csr);
ROCSPARSE_CLIENTS_TEST_WRAP(zbsrgeam);
ROCSPARSE_CLIENTS_TEST_WRAP(zbsrgemm);
ROCSPARSE_CLIENTS_TEST_WRAP(zbsrgemm_buffer_size);
ROCSPARSE_CLIENTS_TEST_WRAP(zbsric0);
ROCSPARSE_CLIENTS_TEST_WRAP(zbsric0_analysis);
ROCSPARSE_CLIENTS_TEST_WRAP(zbsric0_buffer_size);
ROCSPARSE_CLIENTS_TEST_WRAP(zbsrilu0);
ROCSPARSE_CLIENTS_TEST_WRAP(zbsrilu0_analysis);
ROCSPARSE_CLIENTS_TEST_WRAP(zbsrilu0_buffer_size);
ROCSPARSE_CLIENTS_TEST_WRAP(zbsrilu0_numeric_boost);
ROCSPARSE_CLIENTS_TEST_WRAP(zbsrmm);
ROCSPARSE_CLIENTS_TEST_WRAP(zbsrmv);
ROCSPARSE_CLIENTS_TEST_WRAP(zbsrmv_analysis);
ROCSPARSE_CLIENTS_TEST_WRAP(zbsrpad_value);
ROCSPARSE_CLIENTS_TEST_WRAP(zbsrsm_analysis);
ROCSPARSE_CLIENTS_TEST_WRAP(zbsrsm_buffer_size);
ROCSPARSE_CLIENTS_TEST_WRAP(zbsrsm_solve);
ROCSPARSE_CLIENTS_TEST_WRAP(zbsrsv_analysis);
ROCSPARSE_CLIENTS_TEST_WRAP(zbsrsv_buffer_size);
ROCSPARSE_CLIENTS_TEST_WRAP(zbsrsv_solve);
ROCSPARSE_CLIENTS_TEST_WRAP(zbsrxmv);
ROCSPARSE_CLIENTS_TEST_WRAP(zcheck_matrix_coo);
ROCSPARSE_CLIENTS_TEST_WRAP(zcheck_matrix_coo_buffer_size);
ROCSPARSE_CLIENTS_TEST_WRAP(zcheck_matrix_csc);
ROCSPARSE_CLIENTS_TEST_WRAP(zcheck_matrix_csc_buffer_size);
ROCSPARSE_CLIENTS_TEST_WRAP(zcheck_matrix_csr);
ROCSPARSE_CLIENTS_TEST_WRAP(zcheck_matrix_csr_buffer_size);
ROCSPARSE_CLIENTS_TEST_WRAP(zcheck_matrix_ell);
ROCSPARSE_CLIENTS_TEST_WRAP(zcheck_matrix_ell_buffer_size);
ROCSPARSE_CLIENTS_TEST_WRAP(zcheck_matrix_gebsc);
ROCSPARSE_CLIENTS_TEST_WRAP(zcheck_matrix_gebsc_buffer_size);
ROCSPARSE_CLIENTS_TEST_WRAP(zcheck_matrix_gebsr);
ROCSPARSE_CLIENTS_TEST_WRAP(zcheck_matrix_gebsr_buffer_size);
ROCSPARSE_CLIENTS_TEST_WRAP(zcoo2dense);
ROCSPARSE_CLIENTS_TEST_WRAP(zcoomv);
ROCSPARSE_CLIENTS_TEST_WRAP(zcsc2dense);
ROCSPARSE_CLIENTS_TEST_WRAP(zcsr2bsr);
ROCSPARSE_CLIENTS_TEST_WRAP(zcsr2csc);
ROCSPARSE_CLIENTS_TEST_WRAP(zcsr2csr_compress);
ROCSPARSE_CLIENTS_TEST_WRAP(zcsr2dense);
ROCSPARSE_CLIENTS_TEST_WRAP(zcsr2ell);
ROCSPARSE_CLIENTS_TEST_WRAP(zcsr2gebsr);
ROCSPARSE_CLIENTS_TEST_WRAP(zcsr2gebsr_buffer_size);
ROCSPARSE_CLIENTS_TEST_WRAP(zcsr2hyb);
ROCSPARSE_CLIENTS_TEST_WRAP(zcsrcolor);
ROCSPARSE_CLIENTS_TEST_WRAP(zcsrgeam);
ROCSPARSE_CLIENTS_TEST_WRAP(zcsrgemm);
ROCSPARSE_CLIENTS_TEST_WRAP(zcsrgemm_buffer_size);
ROCSPARSE_CLIENTS_TEST_WRAP(zcsrgemm_numeric);
ROCSPARSE_CLIENTS_TEST_WRAP(zcsric0);
ROCSPARSE_CLIENTS_TEST_WRAP(zcsric0_analysis);
ROCSPARSE_CLIENTS_TEST_WRAP(zcsric0_buffer_size);
ROCSPARSE_CLIENTS_TEST_WRAP(zcsrilu0);
ROCSPARSE_CLIENTS_TEST_WRAP(zcsrilu0_analysis);
ROCSPARSE_CLIENTS_TEST_WRAP(zcsrilu0_buffer_size);
ROCSPARSE_CLIENTS_TEST_WRAP(zcsrilu0_numeric_boost);
ROCSPARSE_CLIENTS_TEST_WRAP(zcsritilu0_compute);
ROCSPARSE_CLIENTS_TEST_WRAP(zcsritilu0_compute_ex);
ROCSPARSE_CLIENTS_TEST_WRAP(zcsritilu0_history);
ROCSPARSE_CLIENTS_TEST_WRAP(zcsritsv_analysis);
ROCSPARSE_CLIENTS_TEST_WRAP(zcsritsv_buffer_size);
ROCSPARSE_CLIENTS_TEST_WRAP(zcsritsv_solve);
ROCSPARSE_CLIENTS_TEST_WRAP(zcsritsv_solve_ex);
ROCSPARSE_CLIENTS_TEST_WRAP(zcsrmm);
ROCSPARSE_CLIENTS_TEST_WRAP(zcsrmv);
ROCSPARSE_CLIENTS_TEST_WRAP(zcsrmv_analysis);
ROCSPARSE_CLIENTS_TEST_WRAP(zcsrsm_analysis);
ROCSPARSE_CLIENTS_TEST_WRAP(zcsrsm_buffer_size);
ROCSPARSE_CLIENTS_TEST_WRAP(zcsrsm_solve);
ROCSPARSE_CLIENTS_TEST_WRAP(zcsrsv_analysis);
ROCSPARSE_CLIENTS_TEST_WRAP(zcsrsv_buffer_size);
ROCSPARSE_CLIENTS_TEST_WRAP(zcsrsv_solve);
ROCSPARSE_CLIENTS_TEST_WRAP(zdense2coo);
ROCSPARSE_CLIENTS_TEST_WRAP(zdense2csc);
ROCSPARSE_CLIENTS_TEST_WRAP(zdense2csr);
ROCSPARSE_CLIENTS_TEST_WRAP(zdotci);
ROCSPARSE_CLIENTS_TEST_WRAP(zdoti);
ROCSPARSE_CLIENTS_TEST_WRAP(zell2csr);
ROCSPARSE_CLIENTS_TEST_WRAP(zellmv);
ROCSPARSE_CLIENTS_TEST_WRAP(zgebsr2csr);
ROCSPARSE_CLIENTS_TEST_WRAP(zgebsr2gebsc);
ROCSPARSE_CLIENTS_TEST_WRAP(zgebsr2gebsc_buffer_size);
ROCSPARSE_CLIENTS_TEST_WRAP(zgebsr2gebsr);
ROCSPARSE_CLIENTS_TEST_WRAP(zgebsr2gebsr_buffer_size);
ROCSPARSE_CLIENTS_TEST_WRAP(zgebsrmm);
ROCSPARSE_CLIENTS_TEST_WRAP(zgebsrmv);
ROCSPARSE_CLIENTS_TEST_WRAP(zgemmi);
ROCSPARSE_CLIENTS_TEST_WRAP(zgemvi);
ROCSPARSE_CLIENTS_TEST_WRAP(zgemvi_buffer_size);
ROCSPARSE_CLIENTS_TEST_WRAP(zgpsv_interleaved_batch);
ROCSPARSE_CLIENTS_TEST_WRAP(zgpsv_interleaved_batch_buffer_size);
ROCSPARSE_CLIENTS_TEST_WRAP(zgthr);
ROCSPARSE_CLIENTS_TEST_WRAP(zgthrz);
ROCSPARSE_CLIENTS_TEST_WRAP(zgtsv);
ROCSPARSE_CLIENTS_TEST_WRAP(zgtsv_buffer_size);
ROCSPARSE_CLIENTS_TEST_WRAP(zgtsv_interleaved_batch);
ROCSPARSE_CLIENTS_TEST_WRAP(zgtsv_interleaved_batch_buffer_size);
ROCSPARSE_CLIENTS_TEST_WRAP(zgtsv_no_pivot);
ROCSPARSE_CLIENTS_TEST_WRAP(zgtsv_no_pivot_buffer_size);
ROCSPARSE_CLIENTS_TEST_WRAP(zgtsv_no_pivot_strided_batch);
ROCSPARSE_CLIENTS_TEST_WRAP(zgtsv_no_pivot_strided_batch_buffer_size);
ROCSPARSE_CLIENTS_TEST_WRAP(zhyb2csr);
ROCSPARSE_CLIENTS_TEST_WRAP(zhybmv);
ROCSPARSE_CLIENTS_TEST_WRAP(znnz);
ROCSPARSE_CLIENTS_TEST_WRAP(znnz_compress);
ROCSPARSE_CLIENTS_TEST_WRAP(zsctr);
ROCSPARSE_CLIENTS_TEST_WRAP_NO_HANDLE(bell_get);
ROCSPARSE_CLIENTS_TEST_WRAP_NO_HANDLE(bsr_get);
ROCSPARSE_CLIENTS_TEST_WRAP_NO_HANDLE(bsr_set_pointers);
ROCSPARSE_CLIENTS_TEST_WRAP_NO_HANDLE(const_bell_get);
ROCSPARSE_CLIENTS_TEST_WRAP_NO_HANDLE(const_bsr_get);
ROCSPARSE_CLIENTS_TEST_WRAP_NO_HANDLE(const_coo_aos_get);
ROCSPARSE_CLIENTS_TEST_WRAP_NO_HANDLE(const_coo_get);
ROCSPARSE_CLIENTS_TEST_WRAP_NO_HANDLE(const_csc_get);
ROCSPARSE_CLIENTS_TEST_WRAP_NO_HANDLE(const_csr_get);
ROCSPARSE_CLIENTS_TEST_WRAP_NO_HANDLE(const_dnmat_get);
ROCSPARSE_CLIENTS_TEST_WRAP_NO_HANDLE(const_dnmat_get_values);
ROCSPARSE_CLIENTS_TEST_WRAP_NO_HANDLE(const_dnvec_get);
ROCSPARSE_CLIENTS_TEST_WRAP_NO_HANDLE(const_dnvec_get_values);
ROCSPARSE_CLIENTS_TEST_WRAP_NO_HANDLE(const_ell_get);
ROCSPARSE_CLIENTS_TEST_WRAP_NO_HANDLE(const_sell_get);
ROCSPARSE_CLIENTS_TEST_WRAP_NO_HANDLE(const_spmat_get_values);
ROCSPARSE_CLIENTS_TEST_WRAP_NO_HANDLE(const_spvec_get);
ROCSPARSE_CLIENTS_TEST_WRAP_NO_HANDLE(const_spvec_get_values);
ROCSPARSE_CLIENTS_TEST_WRAP_NO_HANDLE(coo_aos_get);
ROCSPARSE_CLIENTS_TEST_WRAP_NO_HANDLE(coo_aos_set_pointers);
ROCSPARSE_CLIENTS_TEST_WRAP_NO_HANDLE(coo_get);
ROCSPARSE_CLIENTS_TEST_WRAP_NO_HANDLE(coo_set_pointers);
ROCSPARSE_CLIENTS_TEST_WRAP_NO_HANDLE(coo_set_strided_batch);
ROCSPARSE_CLIENTS_TEST_WRAP_NO_HANDLE(copy_color_info);
ROCSPARSE_CLIENTS_TEST_WRAP_NO_HANDLE(copy_hyb_mat);
ROCSPARSE_CLIENTS_TEST_WRAP_NO_HANDLE(copy_mat_descr);
ROCSPARSE_CLIENTS_TEST_WRAP_NO_HANDLE(copy_mat_info);
ROCSPARSE_CLIENTS_TEST_WRAP_NO_HANDLE(create_bell_descr);
ROCSPARSE_CLIENTS_TEST_WRAP_NO_HANDLE(create_bsr_descr);
ROCSPARSE_CLIENTS_TEST_WRAP_NO_HANDLE(create_color_info);
ROCSPARSE_CLIENTS_TEST_WRAP_NO_HANDLE(create_const_bell_descr);
ROCSPARSE_CLIENTS_TEST_WRAP_NO_HANDLE(create_const_coo_descr);
ROCSPARSE_CLIENTS_TEST_WRAP_NO_HANDLE(create_const_csc_descr);
ROCSPARSE_CLIENTS_TEST_WRAP_NO_HANDLE(create_const_csr_descr);
ROCSPARSE_CLIENTS_TEST_WRAP_NO_HANDLE(create_const_dnmat_descr);
ROCSPARSE_CLIENTS_TEST_WRAP_NO_HANDLE(create_const_dnvec_descr);
ROCSPARSE_CLIENTS_TEST_WRAP_NO_HANDLE(create_const_sell_descr);
ROCSPARSE_CLIENTS_TEST_WRAP_NO_HANDLE(create_const_spvec_descr);
ROCSPARSE_CLIENTS_TEST_WRAP_NO_HANDLE(create_coo_aos_descr);
ROCSPARSE_CLIENTS_TEST_WRAP_NO_HANDLE(create_coo_descr);
ROCSPARSE_CLIENTS_TEST_WRAP_NO_HANDLE(create_csc_descr);
ROCSPARSE_CLIENTS_TEST_WRAP_NO_HANDLE(create_csr_descr);
ROCSPARSE_CLIENTS_TEST_WRAP_NO_HANDLE(create_dnmat_descr);
ROCSPARSE_CLIENTS_TEST_WRAP_NO_HANDLE(create_dnvec_descr);
ROCSPARSE_CLIENTS_TEST_WRAP_NO_HANDLE(create_ell_descr);
ROCSPARSE_CLIENTS_TEST_WRAP_NO_HANDLE(create_extract_descr);
ROCSPARSE_CLIENTS_TEST_WRAP_NO_HANDLE(create_handle);
ROCSPARSE_CLIENTS_TEST_WRAP_NO_HANDLE(create_hyb_mat);
ROCSPARSE_CLIENTS_TEST_WRAP_NO_HANDLE(create_mat_descr);
ROCSPARSE_CLIENTS_TEST_WRAP_NO_HANDLE(create_mat_info);
ROCSPARSE_CLIENTS_TEST_WRAP_NO_HANDLE(create_sell_descr);
ROCSPARSE_CLIENTS_TEST_WRAP_NO_HANDLE(create_sparse_to_sparse_descr);
ROCSPARSE_CLIENTS_TEST_WRAP_NO_HANDLE(create_spgeam_descr);
ROCSPARSE_CLIENTS_TEST_WRAP_NO_HANDLE(create_spmv_descr);
ROCSPARSE_CLIENTS_TEST_WRAP_NO_HANDLE(create_sptrsm_descr);
ROCSPARSE_CLIENTS_TEST_WRAP_NO_HANDLE(create_sptrsv_descr);
ROCSPARSE_CLIENTS_TEST_WRAP_NO_HANDLE(create_spvec_descr);
ROCSPARSE_CLIENTS_TEST_WRAP_NO_HANDLE(csc_get);
ROCSPARSE_CLIENTS_TEST_WRAP_NO_HANDLE(csc_set_pointers);
ROCSPARSE_CLIENTS_TEST_WRAP_NO_HANDLE(csc_set_strided_batch);
ROCSPARSE_CLIENTS_TEST_WRAP_NO_HANDLE(csr_get);
ROCSPARSE_CLIENTS_TEST_WRAP_NO_HANDLE(csr_set_pointers);
ROCSPARSE_CLIENTS_TEST_WRAP_NO_HANDLE(csr_set_strided_batch);
ROCSPARSE_CLIENTS_TEST_WRAP_NO_HANDLE(destroy_color_info);
ROCSPARSE_CLIENTS_TEST_WRAP_NO_HANDLE(destroy_dnmat_descr);
ROCSPARSE_CLIENTS_TEST_WRAP_NO_HANDLE(destroy_dnvec_descr);
ROCSPARSE_CLIENTS_TEST_WRAP_NO_HANDLE(destroy_error);
ROCSPARSE_CLIENTS_TEST_WRAP_NO_HANDLE(destroy_extract_descr);
ROCSPARSE_CLIENTS_TEST_WRAP_NO_HANDLE(destroy_handle);
ROCSPARSE_CLIENTS_TEST_WRAP_NO_HANDLE(destroy_hyb_mat);
ROCSPARSE_CLIENTS_TEST_WRAP_NO_HANDLE(destroy_mat_descr);
ROCSPARSE_CLIENTS_TEST_WRAP_NO_HANDLE(destroy_mat_info);
ROCSPARSE_CLIENTS_TEST_WRAP_NO_HANDLE(destroy_sparse_to_sparse_descr);
ROCSPARSE_CLIENTS_TEST_WRAP_NO_HANDLE(destroy_spgeam_descr);
ROCSPARSE_CLIENTS_TEST_WRAP_NO_HANDLE(destroy_spmat_descr);
ROCSPARSE_CLIENTS_TEST_WRAP_NO_HANDLE(destroy_spmv_descr);
ROCSPARSE_CLIENTS_TEST_WRAP_NO_HANDLE(destroy_sptrsm_descr);
ROCSPARSE_CLIENTS_TEST_WRAP_NO_HANDLE(destroy_sptrsv_descr);
ROCSPARSE_CLIENTS_TEST_WRAP_NO_HANDLE(destroy_spvec_descr);
ROCSPARSE_CLIENTS_TEST_WRAP_NO_HANDLE(dnmat_get);
ROCSPARSE_CLIENTS_TEST_WRAP_NO_HANDLE(dnmat_get_strided_batch);
ROCSPARSE_CLIENTS_TEST_WRAP_NO_HANDLE(dnmat_get_values);
ROCSPARSE_CLIENTS_TEST_WRAP_NO_HANDLE(dnmat_set_strided_batch);
ROCSPARSE_CLIENTS_TEST_WRAP_NO_HANDLE(dnmat_set_values);
ROCSPARSE_CLIENTS_TEST_WRAP_NO_HANDLE(dnvec_get);
ROCSPARSE_CLIENTS_TEST_WRAP_NO_HANDLE(dnvec_get_strided_batch);
ROCSPARSE_CLIENTS_TEST_WRAP_NO_HANDLE(dnvec_get_values);
ROCSPARSE_CLIENTS_TEST_WRAP_NO_HANDLE(dnvec_set_strided_batch);
ROCSPARSE_CLIENTS_TEST_WRAP_NO_HANDLE(dnvec_set_values);
ROCSPARSE_CLIENTS_TEST_WRAP_NO_HANDLE(ell_get);
ROCSPARSE_CLIENTS_TEST_WRAP_NO_HANDLE(ell_set_pointers);
ROCSPARSE_CLIENTS_TEST_WRAP_NO_HANDLE(sell_get);
ROCSPARSE_CLIENTS_TEST_WRAP_NO_HANDLE(set_mat_diag_type);
ROCSPARSE_CLIENTS_TEST_WRAP_NO_HANDLE(set_mat_fill_mode);
ROCSPARSE_CLIENTS_TEST_WRAP_NO_HANDLE(set_mat_index_base);
ROCSPARSE_CLIENTS_TEST_WRAP_NO_HANDLE(set_mat_storage_mode);
ROCSPARSE_CLIENTS_TEST_WRAP_NO_HANDLE(set_mat_type);
ROCSPARSE_CLIENTS_TEST_WRAP_NO_HANDLE(sparse_to_sparse_permissive);
ROCSPARSE_CLIENTS_TEST_WRAP_NO_HANDLE(spmat_get_attribute);
ROCSPARSE_CLIENTS_TEST_WRAP_NO_HANDLE(spmat_get_format);
ROCSPARSE_CLIENTS_TEST_WRAP_NO_HANDLE(spmat_get_index_base);
ROCSPARSE_CLIENTS_TEST_WRAP_NO_HANDLE(spmat_get_nnz);
ROCSPARSE_CLIENTS_TEST_WRAP_NO_HANDLE(spmat_get_size);
ROCSPARSE_CLIENTS_TEST_WRAP_NO_HANDLE(spmat_get_strided_batch);
ROCSPARSE_CLIENTS_TEST_WRAP_NO_HANDLE(spmat_get_values);
ROCSPARSE_CLIENTS_TEST_WRAP_NO_HANDLE(spmat_set_attribute);
ROCSPARSE_CLIENTS_TEST_WRAP_NO_HANDLE(spmat_set_nnz);
ROCSPARSE_CLIENTS_TEST_WRAP_NO_HANDLE(spmat_set_strided_batch);
ROCSPARSE_CLIENTS_TEST_WRAP_NO_HANDLE(spmat_set_values);
ROCSPARSE_CLIENTS_TEST_WRAP_NO_HANDLE(spvec_get);
ROCSPARSE_CLIENTS_TEST_WRAP_NO_HANDLE(spvec_get_index_base);
ROCSPARSE_CLIENTS_TEST_WRAP_NO_HANDLE(spvec_get_values);
ROCSPARSE_CLIENTS_TEST_WRAP_NO_HANDLE(spvec_set_values);

//
// Redirect rocsparse C API names to test wrappers
//
#define rocsparse_axpby rocsparse_test_wrap_axpby
#define rocsparse_bsrgeam_nnzb rocsparse_test_wrap_bsrgeam_nnzb
#define rocsparse_bsrgemm_nnzb rocsparse_test_wrap_bsrgemm_nnzb
#define rocsparse_bsric0_clear rocsparse_test_wrap_bsric0_clear
#define rocsparse_bsric0_zero_pivot rocsparse_test_wrap_bsric0_zero_pivot
#define rocsparse_bsrilu0_clear rocsparse_test_wrap_bsrilu0_clear
#define rocsparse_bsrilu0_zero_pivot rocsparse_test_wrap_bsrilu0_zero_pivot
#define rocsparse_bsrmv_clear rocsparse_test_wrap_bsrmv_clear
#define rocsparse_bsrsm_clear rocsparse_test_wrap_bsrsm_clear
#define rocsparse_bsrsm_zero_pivot rocsparse_test_wrap_bsrsm_zero_pivot
#define rocsparse_bsrsv_clear rocsparse_test_wrap_bsrsv_clear
#define rocsparse_bsrsv_zero_pivot rocsparse_test_wrap_bsrsv_zero_pivot
#define rocsparse_caxpyi rocsparse_test_wrap_caxpyi
#define rocsparse_cbsr2csr rocsparse_test_wrap_cbsr2csr
#define rocsparse_cbsrgeam rocsparse_test_wrap_cbsrgeam
#define rocsparse_cbsrgemm rocsparse_test_wrap_cbsrgemm
#define rocsparse_cbsrgemm_buffer_size rocsparse_test_wrap_cbsrgemm_buffer_size
#define rocsparse_cbsric0 rocsparse_test_wrap_cbsric0
#define rocsparse_cbsric0_analysis rocsparse_test_wrap_cbsric0_analysis
#define rocsparse_cbsric0_buffer_size rocsparse_test_wrap_cbsric0_buffer_size
#define rocsparse_cbsrilu0 rocsparse_test_wrap_cbsrilu0
#define rocsparse_cbsrilu0_analysis rocsparse_test_wrap_cbsrilu0_analysis
#define rocsparse_cbsrilu0_buffer_size rocsparse_test_wrap_cbsrilu0_buffer_size
#define rocsparse_cbsrilu0_numeric_boost rocsparse_test_wrap_cbsrilu0_numeric_boost
#define rocsparse_cbsrmm rocsparse_test_wrap_cbsrmm
#define rocsparse_cbsrmv rocsparse_test_wrap_cbsrmv
#define rocsparse_cbsrmv_analysis rocsparse_test_wrap_cbsrmv_analysis
#define rocsparse_cbsrpad_value rocsparse_test_wrap_cbsrpad_value
#define rocsparse_cbsrsm_analysis rocsparse_test_wrap_cbsrsm_analysis
#define rocsparse_cbsrsm_buffer_size rocsparse_test_wrap_cbsrsm_buffer_size
#define rocsparse_cbsrsm_solve rocsparse_test_wrap_cbsrsm_solve
#define rocsparse_cbsrsv_analysis rocsparse_test_wrap_cbsrsv_analysis
#define rocsparse_cbsrsv_buffer_size rocsparse_test_wrap_cbsrsv_buffer_size
#define rocsparse_cbsrsv_solve rocsparse_test_wrap_cbsrsv_solve
#define rocsparse_cbsrxmv rocsparse_test_wrap_cbsrxmv
#define rocsparse_ccheck_matrix_coo rocsparse_test_wrap_ccheck_matrix_coo
#define rocsparse_ccheck_matrix_coo_buffer_size rocsparse_test_wrap_ccheck_matrix_coo_buffer_size
#define rocsparse_ccheck_matrix_csc rocsparse_test_wrap_ccheck_matrix_csc
#define rocsparse_ccheck_matrix_csc_buffer_size rocsparse_test_wrap_ccheck_matrix_csc_buffer_size
#define rocsparse_ccheck_matrix_csr rocsparse_test_wrap_ccheck_matrix_csr
#define rocsparse_ccheck_matrix_csr_buffer_size rocsparse_test_wrap_ccheck_matrix_csr_buffer_size
#define rocsparse_ccheck_matrix_ell rocsparse_test_wrap_ccheck_matrix_ell
#define rocsparse_ccheck_matrix_ell_buffer_size rocsparse_test_wrap_ccheck_matrix_ell_buffer_size
#define rocsparse_ccheck_matrix_gebsc rocsparse_test_wrap_ccheck_matrix_gebsc
#define rocsparse_ccheck_matrix_gebsc_buffer_size \
    rocsparse_test_wrap_ccheck_matrix_gebsc_buffer_size
#define rocsparse_ccheck_matrix_gebsr rocsparse_test_wrap_ccheck_matrix_gebsr
#define rocsparse_ccheck_matrix_gebsr_buffer_size \
    rocsparse_test_wrap_ccheck_matrix_gebsr_buffer_size
#define rocsparse_ccoo2dense rocsparse_test_wrap_ccoo2dense
#define rocsparse_ccoomv rocsparse_test_wrap_ccoomv
#define rocsparse_ccsc2dense rocsparse_test_wrap_ccsc2dense
#define rocsparse_ccsr2bsr rocsparse_test_wrap_ccsr2bsr
#define rocsparse_ccsr2csc rocsparse_test_wrap_ccsr2csc
#define rocsparse_ccsr2csr_compress rocsparse_test_wrap_ccsr2csr_compress
#define rocsparse_ccsr2dense rocsparse_test_wrap_ccsr2dense
#define rocsparse_ccsr2ell rocsparse_test_wrap_ccsr2ell
#define rocsparse_ccsr2gebsr rocsparse_test_wrap_ccsr2gebsr
#define rocsparse_ccsr2gebsr_buffer_size rocsparse_test_wrap_ccsr2gebsr_buffer_size
#define rocsparse_ccsr2hyb rocsparse_test_wrap_ccsr2hyb
#define rocsparse_ccsrcolor rocsparse_test_wrap_ccsrcolor
#define rocsparse_ccsrgeam rocsparse_test_wrap_ccsrgeam
#define rocsparse_ccsrgemm rocsparse_test_wrap_ccsrgemm
#define rocsparse_ccsrgemm_buffer_size rocsparse_test_wrap_ccsrgemm_buffer_size
#define rocsparse_ccsrgemm_numeric rocsparse_test_wrap_ccsrgemm_numeric
#define rocsparse_ccsric0 rocsparse_test_wrap_ccsric0
#define rocsparse_ccsric0_analysis rocsparse_test_wrap_ccsric0_analysis
#define rocsparse_ccsric0_buffer_size rocsparse_test_wrap_ccsric0_buffer_size
#define rocsparse_ccsrilu0 rocsparse_test_wrap_ccsrilu0
#define rocsparse_ccsrilu0_analysis rocsparse_test_wrap_ccsrilu0_analysis
#define rocsparse_ccsrilu0_buffer_size rocsparse_test_wrap_ccsrilu0_buffer_size
#define rocsparse_ccsrilu0_numeric_boost rocsparse_test_wrap_ccsrilu0_numeric_boost
#define rocsparse_ccsritilu0_compute rocsparse_test_wrap_ccsritilu0_compute
#define rocsparse_ccsritilu0_compute_ex rocsparse_test_wrap_ccsritilu0_compute_ex
#define rocsparse_ccsritilu0_history rocsparse_test_wrap_ccsritilu0_history
#define rocsparse_ccsritsv_analysis rocsparse_test_wrap_ccsritsv_analysis
#define rocsparse_ccsritsv_buffer_size rocsparse_test_wrap_ccsritsv_buffer_size
#define rocsparse_ccsritsv_solve rocsparse_test_wrap_ccsritsv_solve
#define rocsparse_ccsritsv_solve_ex rocsparse_test_wrap_ccsritsv_solve_ex
#define rocsparse_ccsrmm rocsparse_test_wrap_ccsrmm
#define rocsparse_ccsrmv rocsparse_test_wrap_ccsrmv
#define rocsparse_ccsrmv_analysis rocsparse_test_wrap_ccsrmv_analysis
#define rocsparse_ccsrsm_analysis rocsparse_test_wrap_ccsrsm_analysis
#define rocsparse_ccsrsm_buffer_size rocsparse_test_wrap_ccsrsm_buffer_size
#define rocsparse_ccsrsm_solve rocsparse_test_wrap_ccsrsm_solve
#define rocsparse_ccsrsv_analysis rocsparse_test_wrap_ccsrsv_analysis
#define rocsparse_ccsrsv_buffer_size rocsparse_test_wrap_ccsrsv_buffer_size
#define rocsparse_ccsrsv_solve rocsparse_test_wrap_ccsrsv_solve
#define rocsparse_cdense2coo rocsparse_test_wrap_cdense2coo
#define rocsparse_cdense2csc rocsparse_test_wrap_cdense2csc
#define rocsparse_cdense2csr rocsparse_test_wrap_cdense2csr
#define rocsparse_cdotci rocsparse_test_wrap_cdotci
#define rocsparse_cdoti rocsparse_test_wrap_cdoti
#define rocsparse_cell2csr rocsparse_test_wrap_cell2csr
#define rocsparse_cellmv rocsparse_test_wrap_cellmv
#define rocsparse_cgebsr2csr rocsparse_test_wrap_cgebsr2csr
#define rocsparse_cgebsr2gebsc rocsparse_test_wrap_cgebsr2gebsc
#define rocsparse_cgebsr2gebsc_buffer_size rocsparse_test_wrap_cgebsr2gebsc_buffer_size
#define rocsparse_cgebsr2gebsr rocsparse_test_wrap_cgebsr2gebsr
#define rocsparse_cgebsr2gebsr_buffer_size rocsparse_test_wrap_cgebsr2gebsr_buffer_size
#define rocsparse_cgebsrmm rocsparse_test_wrap_cgebsrmm
#define rocsparse_cgebsrmv rocsparse_test_wrap_cgebsrmv
#define rocsparse_cgemmi rocsparse_test_wrap_cgemmi
#define rocsparse_cgemvi rocsparse_test_wrap_cgemvi
#define rocsparse_cgemvi_buffer_size rocsparse_test_wrap_cgemvi_buffer_size
#define rocsparse_cgpsv_interleaved_batch rocsparse_test_wrap_cgpsv_interleaved_batch
#define rocsparse_cgpsv_interleaved_batch_buffer_size \
    rocsparse_test_wrap_cgpsv_interleaved_batch_buffer_size
#define rocsparse_cgthr rocsparse_test_wrap_cgthr
#define rocsparse_cgthrz rocsparse_test_wrap_cgthrz
#define rocsparse_cgtsv rocsparse_test_wrap_cgtsv
#define rocsparse_cgtsv_buffer_size rocsparse_test_wrap_cgtsv_buffer_size
#define rocsparse_cgtsv_interleaved_batch rocsparse_test_wrap_cgtsv_interleaved_batch
#define rocsparse_cgtsv_interleaved_batch_buffer_size \
    rocsparse_test_wrap_cgtsv_interleaved_batch_buffer_size
#define rocsparse_cgtsv_no_pivot rocsparse_test_wrap_cgtsv_no_pivot
#define rocsparse_cgtsv_no_pivot_buffer_size rocsparse_test_wrap_cgtsv_no_pivot_buffer_size
#define rocsparse_cgtsv_no_pivot_strided_batch rocsparse_test_wrap_cgtsv_no_pivot_strided_batch
#define rocsparse_cgtsv_no_pivot_strided_batch_buffer_size \
    rocsparse_test_wrap_cgtsv_no_pivot_strided_batch_buffer_size
#define rocsparse_check_matrix_hyb rocsparse_test_wrap_check_matrix_hyb
#define rocsparse_check_matrix_hyb_buffer_size rocsparse_test_wrap_check_matrix_hyb_buffer_size
#define rocsparse_check_spmat rocsparse_test_wrap_check_spmat
#define rocsparse_chyb2csr rocsparse_test_wrap_chyb2csr
#define rocsparse_chybmv rocsparse_test_wrap_chybmv
#define rocsparse_cnnz rocsparse_test_wrap_cnnz
#define rocsparse_cnnz_compress rocsparse_test_wrap_cnnz_compress
#define rocsparse_coo2csr rocsparse_test_wrap_coo2csr
#define rocsparse_coosort_buffer_size rocsparse_test_wrap_coosort_buffer_size
#define rocsparse_coosort_by_column rocsparse_test_wrap_coosort_by_column
#define rocsparse_coosort_by_row rocsparse_test_wrap_coosort_by_row
#define rocsparse_create_identity_permutation rocsparse_test_wrap_create_identity_permutation
#define rocsparse_cscsort rocsparse_test_wrap_cscsort
#define rocsparse_cscsort_buffer_size rocsparse_test_wrap_cscsort_buffer_size
#define rocsparse_csctr rocsparse_test_wrap_csctr
#define rocsparse_csr2bsr_nnz rocsparse_test_wrap_csr2bsr_nnz
#define rocsparse_csr2coo rocsparse_test_wrap_csr2coo
#define rocsparse_csr2csc_buffer_size rocsparse_test_wrap_csr2csc_buffer_size
#define rocsparse_csr2ell_width rocsparse_test_wrap_csr2ell_width
#define rocsparse_csr2gebsr_nnz rocsparse_test_wrap_csr2gebsr_nnz
#define rocsparse_csrgeam_nnz rocsparse_test_wrap_csrgeam_nnz
#define rocsparse_csrgemm_nnz rocsparse_test_wrap_csrgemm_nnz
#define rocsparse_csrgemm_symbolic rocsparse_test_wrap_csrgemm_symbolic
#define rocsparse_csric0_clear rocsparse_test_wrap_csric0_clear
#define rocsparse_csric0_get_tolerance rocsparse_test_wrap_csric0_get_tolerance
#define rocsparse_csric0_set_tolerance rocsparse_test_wrap_csric0_set_tolerance
#define rocsparse_csric0_singular_pivot rocsparse_test_wrap_csric0_singular_pivot
#define rocsparse_csric0_zero_pivot rocsparse_test_wrap_csric0_zero_pivot
#define rocsparse_csrilu0_clear rocsparse_test_wrap_csrilu0_clear
#define rocsparse_csrilu0_get_tolerance rocsparse_test_wrap_csrilu0_get_tolerance
#define rocsparse_csrilu0_set_tolerance rocsparse_test_wrap_csrilu0_set_tolerance
#define rocsparse_csrilu0_singular_pivot rocsparse_test_wrap_csrilu0_singular_pivot
#define rocsparse_csrilu0_zero_pivot rocsparse_test_wrap_csrilu0_zero_pivot
#define rocsparse_csritilu0_buffer_size rocsparse_test_wrap_csritilu0_buffer_size
#define rocsparse_csritilu0_preprocess rocsparse_test_wrap_csritilu0_preprocess
#define rocsparse_csritsv_clear rocsparse_test_wrap_csritsv_clear
#define rocsparse_csritsv_zero_pivot rocsparse_test_wrap_csritsv_zero_pivot
#define rocsparse_csrmv_clear rocsparse_test_wrap_csrmv_clear
#define rocsparse_csrsm_clear rocsparse_test_wrap_csrsm_clear
#define rocsparse_csrsm_zero_pivot rocsparse_test_wrap_csrsm_zero_pivot
#define rocsparse_csrsort rocsparse_test_wrap_csrsort
#define rocsparse_csrsort_buffer_size rocsparse_test_wrap_csrsort_buffer_size
#define rocsparse_csrsv_clear rocsparse_test_wrap_csrsv_clear
#define rocsparse_csrsv_zero_pivot rocsparse_test_wrap_csrsv_zero_pivot
#define rocsparse_daxpyi rocsparse_test_wrap_daxpyi
#define rocsparse_dbsr2csr rocsparse_test_wrap_dbsr2csr
#define rocsparse_dbsrgeam rocsparse_test_wrap_dbsrgeam
#define rocsparse_dbsrgemm rocsparse_test_wrap_dbsrgemm
#define rocsparse_dbsrgemm_buffer_size rocsparse_test_wrap_dbsrgemm_buffer_size
#define rocsparse_dbsric0 rocsparse_test_wrap_dbsric0
#define rocsparse_dbsric0_analysis rocsparse_test_wrap_dbsric0_analysis
#define rocsparse_dbsric0_buffer_size rocsparse_test_wrap_dbsric0_buffer_size
#define rocsparse_dbsrilu0 rocsparse_test_wrap_dbsrilu0
#define rocsparse_dbsrilu0_analysis rocsparse_test_wrap_dbsrilu0_analysis
#define rocsparse_dbsrilu0_buffer_size rocsparse_test_wrap_dbsrilu0_buffer_size
#define rocsparse_dbsrilu0_numeric_boost rocsparse_test_wrap_dbsrilu0_numeric_boost
#define rocsparse_dbsrmm rocsparse_test_wrap_dbsrmm
#define rocsparse_dbsrmv rocsparse_test_wrap_dbsrmv
#define rocsparse_dbsrmv_analysis rocsparse_test_wrap_dbsrmv_analysis
#define rocsparse_dbsrpad_value rocsparse_test_wrap_dbsrpad_value
#define rocsparse_dbsrsm_analysis rocsparse_test_wrap_dbsrsm_analysis
#define rocsparse_dbsrsm_buffer_size rocsparse_test_wrap_dbsrsm_buffer_size
#define rocsparse_dbsrsm_solve rocsparse_test_wrap_dbsrsm_solve
#define rocsparse_dbsrsv_analysis rocsparse_test_wrap_dbsrsv_analysis
#define rocsparse_dbsrsv_buffer_size rocsparse_test_wrap_dbsrsv_buffer_size
#define rocsparse_dbsrsv_solve rocsparse_test_wrap_dbsrsv_solve
#define rocsparse_dbsrxmv rocsparse_test_wrap_dbsrxmv
#define rocsparse_dcbsrilu0_numeric_boost rocsparse_test_wrap_dcbsrilu0_numeric_boost
#define rocsparse_dccsrilu0_numeric_boost rocsparse_test_wrap_dccsrilu0_numeric_boost
#define rocsparse_dcheck_matrix_coo rocsparse_test_wrap_dcheck_matrix_coo
#define rocsparse_dcheck_matrix_coo_buffer_size rocsparse_test_wrap_dcheck_matrix_coo_buffer_size
#define rocsparse_dcheck_matrix_csc rocsparse_test_wrap_dcheck_matrix_csc
#define rocsparse_dcheck_matrix_csc_buffer_size rocsparse_test_wrap_dcheck_matrix_csc_buffer_size
#define rocsparse_dcheck_matrix_csr rocsparse_test_wrap_dcheck_matrix_csr
#define rocsparse_dcheck_matrix_csr_buffer_size rocsparse_test_wrap_dcheck_matrix_csr_buffer_size
#define rocsparse_dcheck_matrix_ell rocsparse_test_wrap_dcheck_matrix_ell
#define rocsparse_dcheck_matrix_ell_buffer_size rocsparse_test_wrap_dcheck_matrix_ell_buffer_size
#define rocsparse_dcheck_matrix_gebsc rocsparse_test_wrap_dcheck_matrix_gebsc
#define rocsparse_dcheck_matrix_gebsc_buffer_size \
    rocsparse_test_wrap_dcheck_matrix_gebsc_buffer_size
#define rocsparse_dcheck_matrix_gebsr rocsparse_test_wrap_dcheck_matrix_gebsr
#define rocsparse_dcheck_matrix_gebsr_buffer_size \
    rocsparse_test_wrap_dcheck_matrix_gebsr_buffer_size
#define rocsparse_dcoo2dense rocsparse_test_wrap_dcoo2dense
#define rocsparse_dcoomv rocsparse_test_wrap_dcoomv
#define rocsparse_dcsc2dense rocsparse_test_wrap_dcsc2dense
#define rocsparse_dcsr2bsr rocsparse_test_wrap_dcsr2bsr
#define rocsparse_dcsr2csc rocsparse_test_wrap_dcsr2csc
#define rocsparse_dcsr2csr_compress rocsparse_test_wrap_dcsr2csr_compress
#define rocsparse_dcsr2dense rocsparse_test_wrap_dcsr2dense
#define rocsparse_dcsr2ell rocsparse_test_wrap_dcsr2ell
#define rocsparse_dcsr2gebsr rocsparse_test_wrap_dcsr2gebsr
#define rocsparse_dcsr2gebsr_buffer_size rocsparse_test_wrap_dcsr2gebsr_buffer_size
#define rocsparse_dcsr2hyb rocsparse_test_wrap_dcsr2hyb
#define rocsparse_dcsrcolor rocsparse_test_wrap_dcsrcolor
#define rocsparse_dcsrgeam rocsparse_test_wrap_dcsrgeam
#define rocsparse_dcsrgemm rocsparse_test_wrap_dcsrgemm
#define rocsparse_dcsrgemm_buffer_size rocsparse_test_wrap_dcsrgemm_buffer_size
#define rocsparse_dcsrgemm_numeric rocsparse_test_wrap_dcsrgemm_numeric
#define rocsparse_dcsric0 rocsparse_test_wrap_dcsric0
#define rocsparse_dcsric0_analysis rocsparse_test_wrap_dcsric0_analysis
#define rocsparse_dcsric0_buffer_size rocsparse_test_wrap_dcsric0_buffer_size
#define rocsparse_dcsrilu0 rocsparse_test_wrap_dcsrilu0
#define rocsparse_dcsrilu0_analysis rocsparse_test_wrap_dcsrilu0_analysis
#define rocsparse_dcsrilu0_buffer_size rocsparse_test_wrap_dcsrilu0_buffer_size
#define rocsparse_dcsrilu0_numeric_boost rocsparse_test_wrap_dcsrilu0_numeric_boost
#define rocsparse_dcsritilu0_compute rocsparse_test_wrap_dcsritilu0_compute
#define rocsparse_dcsritilu0_compute_ex rocsparse_test_wrap_dcsritilu0_compute_ex
#define rocsparse_dcsritilu0_history rocsparse_test_wrap_dcsritilu0_history
#define rocsparse_dcsritsv_analysis rocsparse_test_wrap_dcsritsv_analysis
#define rocsparse_dcsritsv_buffer_size rocsparse_test_wrap_dcsritsv_buffer_size
#define rocsparse_dcsritsv_solve rocsparse_test_wrap_dcsritsv_solve
#define rocsparse_dcsritsv_solve_ex rocsparse_test_wrap_dcsritsv_solve_ex
#define rocsparse_dcsrmm rocsparse_test_wrap_dcsrmm
#define rocsparse_dcsrmv rocsparse_test_wrap_dcsrmv
#define rocsparse_dcsrmv_analysis rocsparse_test_wrap_dcsrmv_analysis
#define rocsparse_dcsrsm_analysis rocsparse_test_wrap_dcsrsm_analysis
#define rocsparse_dcsrsm_buffer_size rocsparse_test_wrap_dcsrsm_buffer_size
#define rocsparse_dcsrsm_solve rocsparse_test_wrap_dcsrsm_solve
#define rocsparse_dcsrsv_analysis rocsparse_test_wrap_dcsrsv_analysis
#define rocsparse_dcsrsv_buffer_size rocsparse_test_wrap_dcsrsv_buffer_size
#define rocsparse_dcsrsv_solve rocsparse_test_wrap_dcsrsv_solve
#define rocsparse_ddense2coo rocsparse_test_wrap_ddense2coo
#define rocsparse_ddense2csc rocsparse_test_wrap_ddense2csc
#define rocsparse_ddense2csr rocsparse_test_wrap_ddense2csr
#define rocsparse_ddoti rocsparse_test_wrap_ddoti
#define rocsparse_dell2csr rocsparse_test_wrap_dell2csr
#define rocsparse_dellmv rocsparse_test_wrap_dellmv
#define rocsparse_dense_to_sparse rocsparse_test_wrap_dense_to_sparse
#define rocsparse_dgebsr2csr rocsparse_test_wrap_dgebsr2csr
#define rocsparse_dgebsr2gebsc rocsparse_test_wrap_dgebsr2gebsc
#define rocsparse_dgebsr2gebsc_buffer_size rocsparse_test_wrap_dgebsr2gebsc_buffer_size
#define rocsparse_dgebsr2gebsr rocsparse_test_wrap_dgebsr2gebsr
#define rocsparse_dgebsr2gebsr_buffer_size rocsparse_test_wrap_dgebsr2gebsr_buffer_size
#define rocsparse_dgebsrmm rocsparse_test_wrap_dgebsrmm
#define rocsparse_dgebsrmv rocsparse_test_wrap_dgebsrmv
#define rocsparse_dgemmi rocsparse_test_wrap_dgemmi
#define rocsparse_dgemvi rocsparse_test_wrap_dgemvi
#define rocsparse_dgemvi_buffer_size rocsparse_test_wrap_dgemvi_buffer_size
#define rocsparse_dgpsv_interleaved_batch rocsparse_test_wrap_dgpsv_interleaved_batch
#define rocsparse_dgpsv_interleaved_batch_buffer_size \
    rocsparse_test_wrap_dgpsv_interleaved_batch_buffer_size
#define rocsparse_dgthr rocsparse_test_wrap_dgthr
#define rocsparse_dgthrz rocsparse_test_wrap_dgthrz
#define rocsparse_dgtsv rocsparse_test_wrap_dgtsv
#define rocsparse_dgtsv_buffer_size rocsparse_test_wrap_dgtsv_buffer_size
#define rocsparse_dgtsv_interleaved_batch rocsparse_test_wrap_dgtsv_interleaved_batch
#define rocsparse_dgtsv_interleaved_batch_buffer_size \
    rocsparse_test_wrap_dgtsv_interleaved_batch_buffer_size
#define rocsparse_dgtsv_no_pivot rocsparse_test_wrap_dgtsv_no_pivot
#define rocsparse_dgtsv_no_pivot_buffer_size rocsparse_test_wrap_dgtsv_no_pivot_buffer_size
#define rocsparse_dgtsv_no_pivot_strided_batch rocsparse_test_wrap_dgtsv_no_pivot_strided_batch
#define rocsparse_dgtsv_no_pivot_strided_batch_buffer_size \
    rocsparse_test_wrap_dgtsv_no_pivot_strided_batch_buffer_size
#define rocsparse_dhyb2csr rocsparse_test_wrap_dhyb2csr
#define rocsparse_dhybmv rocsparse_test_wrap_dhybmv
#define rocsparse_dnnz rocsparse_test_wrap_dnnz
#define rocsparse_dnnz_compress rocsparse_test_wrap_dnnz_compress
#define rocsparse_dprune_csr2csr rocsparse_test_wrap_dprune_csr2csr
#define rocsparse_dprune_csr2csr_buffer_size rocsparse_test_wrap_dprune_csr2csr_buffer_size
#define rocsparse_dprune_csr2csr_by_percentage rocsparse_test_wrap_dprune_csr2csr_by_percentage
#define rocsparse_dprune_csr2csr_by_percentage_buffer_size \
    rocsparse_test_wrap_dprune_csr2csr_by_percentage_buffer_size
#define rocsparse_dprune_csr2csr_nnz rocsparse_test_wrap_dprune_csr2csr_nnz
#define rocsparse_dprune_csr2csr_nnz_by_percentage \
    rocsparse_test_wrap_dprune_csr2csr_nnz_by_percentage
#define rocsparse_dprune_dense2csr rocsparse_test_wrap_dprune_dense2csr
#define rocsparse_dprune_dense2csr_buffer_size rocsparse_test_wrap_dprune_dense2csr_buffer_size
#define rocsparse_dprune_dense2csr_by_percentage rocsparse_test_wrap_dprune_dense2csr_by_percentage
#define rocsparse_dprune_dense2csr_by_percentage_buffer_size \
    rocsparse_test_wrap_dprune_dense2csr_by_percentage_buffer_size
#define rocsparse_dprune_dense2csr_nnz rocsparse_test_wrap_dprune_dense2csr_nnz
#define rocsparse_dprune_dense2csr_nnz_by_percentage \
    rocsparse_test_wrap_dprune_dense2csr_nnz_by_percentage
#define rocsparse_droti rocsparse_test_wrap_droti
#define rocsparse_dsbsrilu0_numeric_boost rocsparse_test_wrap_dsbsrilu0_numeric_boost
#define rocsparse_dscsrilu0_numeric_boost rocsparse_test_wrap_dscsrilu0_numeric_boost
#define rocsparse_dsctr rocsparse_test_wrap_dsctr
#define rocsparse_ell2csr_nnz rocsparse_test_wrap_ell2csr_nnz
#define rocsparse_extract rocsparse_test_wrap_extract
#define rocsparse_extract_buffer_size rocsparse_test_wrap_extract_buffer_size
#define rocsparse_extract_nnz rocsparse_test_wrap_extract_nnz
#define rocsparse_gather rocsparse_test_wrap_gather
#define rocsparse_gebsr2gebsr_nnz rocsparse_test_wrap_gebsr2gebsr_nnz
#define rocsparse_get_git_rev rocsparse_test_wrap_get_git_rev
#define rocsparse_get_pointer_mode rocsparse_test_wrap_get_pointer_mode
#define rocsparse_get_stream rocsparse_test_wrap_get_stream
#define rocsparse_get_version rocsparse_test_wrap_get_version
#define rocsparse_hyb2csr_buffer_size rocsparse_test_wrap_hyb2csr_buffer_size
#define rocsparse_inverse_permutation rocsparse_test_wrap_inverse_permutation
#define rocsparse_isctr rocsparse_test_wrap_isctr
#define rocsparse_rot rocsparse_test_wrap_rot
#define rocsparse_saxpyi rocsparse_test_wrap_saxpyi
#define rocsparse_sbsr2csr rocsparse_test_wrap_sbsr2csr
#define rocsparse_sbsrgeam rocsparse_test_wrap_sbsrgeam
#define rocsparse_sbsrgemm rocsparse_test_wrap_sbsrgemm
#define rocsparse_sbsrgemm_buffer_size rocsparse_test_wrap_sbsrgemm_buffer_size
#define rocsparse_sbsric0 rocsparse_test_wrap_sbsric0
#define rocsparse_sbsric0_analysis rocsparse_test_wrap_sbsric0_analysis
#define rocsparse_sbsric0_buffer_size rocsparse_test_wrap_sbsric0_buffer_size
#define rocsparse_sbsrilu0 rocsparse_test_wrap_sbsrilu0
#define rocsparse_sbsrilu0_analysis rocsparse_test_wrap_sbsrilu0_analysis
#define rocsparse_sbsrilu0_buffer_size rocsparse_test_wrap_sbsrilu0_buffer_size
#define rocsparse_sbsrilu0_numeric_boost rocsparse_test_wrap_sbsrilu0_numeric_boost
#define rocsparse_sbsrmm rocsparse_test_wrap_sbsrmm
#define rocsparse_sbsrmv rocsparse_test_wrap_sbsrmv
#define rocsparse_sbsrmv_analysis rocsparse_test_wrap_sbsrmv_analysis
#define rocsparse_sbsrpad_value rocsparse_test_wrap_sbsrpad_value
#define rocsparse_sbsrsm_analysis rocsparse_test_wrap_sbsrsm_analysis
#define rocsparse_sbsrsm_buffer_size rocsparse_test_wrap_sbsrsm_buffer_size
#define rocsparse_sbsrsm_solve rocsparse_test_wrap_sbsrsm_solve
#define rocsparse_sbsrsv_analysis rocsparse_test_wrap_sbsrsv_analysis
#define rocsparse_sbsrsv_buffer_size rocsparse_test_wrap_sbsrsv_buffer_size
#define rocsparse_sbsrsv_solve rocsparse_test_wrap_sbsrsv_solve
#define rocsparse_sbsrxmv rocsparse_test_wrap_sbsrxmv
#define rocsparse_scatter rocsparse_test_wrap_scatter
#define rocsparse_scheck_matrix_coo rocsparse_test_wrap_scheck_matrix_coo
#define rocsparse_scheck_matrix_coo_buffer_size rocsparse_test_wrap_scheck_matrix_coo_buffer_size
#define rocsparse_scheck_matrix_csc rocsparse_test_wrap_scheck_matrix_csc
#define rocsparse_scheck_matrix_csc_buffer_size rocsparse_test_wrap_scheck_matrix_csc_buffer_size
#define rocsparse_scheck_matrix_csr rocsparse_test_wrap_scheck_matrix_csr
#define rocsparse_scheck_matrix_csr_buffer_size rocsparse_test_wrap_scheck_matrix_csr_buffer_size
#define rocsparse_scheck_matrix_ell rocsparse_test_wrap_scheck_matrix_ell
#define rocsparse_scheck_matrix_ell_buffer_size rocsparse_test_wrap_scheck_matrix_ell_buffer_size
#define rocsparse_scheck_matrix_gebsc rocsparse_test_wrap_scheck_matrix_gebsc
#define rocsparse_scheck_matrix_gebsc_buffer_size \
    rocsparse_test_wrap_scheck_matrix_gebsc_buffer_size
#define rocsparse_scheck_matrix_gebsr rocsparse_test_wrap_scheck_matrix_gebsr
#define rocsparse_scheck_matrix_gebsr_buffer_size \
    rocsparse_test_wrap_scheck_matrix_gebsr_buffer_size
#define rocsparse_scoo2dense rocsparse_test_wrap_scoo2dense
#define rocsparse_scoomv rocsparse_test_wrap_scoomv
#define rocsparse_scsc2dense rocsparse_test_wrap_scsc2dense
#define rocsparse_scsr2bsr rocsparse_test_wrap_scsr2bsr
#define rocsparse_scsr2csc rocsparse_test_wrap_scsr2csc
#define rocsparse_scsr2csr_compress rocsparse_test_wrap_scsr2csr_compress
#define rocsparse_scsr2dense rocsparse_test_wrap_scsr2dense
#define rocsparse_scsr2ell rocsparse_test_wrap_scsr2ell
#define rocsparse_scsr2gebsr rocsparse_test_wrap_scsr2gebsr
#define rocsparse_scsr2gebsr_buffer_size rocsparse_test_wrap_scsr2gebsr_buffer_size
#define rocsparse_scsr2hyb rocsparse_test_wrap_scsr2hyb
#define rocsparse_scsrcolor rocsparse_test_wrap_scsrcolor
#define rocsparse_scsrgeam rocsparse_test_wrap_scsrgeam
#define rocsparse_scsrgemm rocsparse_test_wrap_scsrgemm
#define rocsparse_scsrgemm_buffer_size rocsparse_test_wrap_scsrgemm_buffer_size
#define rocsparse_scsrgemm_numeric rocsparse_test_wrap_scsrgemm_numeric
#define rocsparse_scsric0 rocsparse_test_wrap_scsric0
#define rocsparse_scsric0_analysis rocsparse_test_wrap_scsric0_analysis
#define rocsparse_scsric0_buffer_size rocsparse_test_wrap_scsric0_buffer_size
#define rocsparse_scsrilu0 rocsparse_test_wrap_scsrilu0
#define rocsparse_scsrilu0_analysis rocsparse_test_wrap_scsrilu0_analysis
#define rocsparse_scsrilu0_buffer_size rocsparse_test_wrap_scsrilu0_buffer_size
#define rocsparse_scsrilu0_numeric_boost rocsparse_test_wrap_scsrilu0_numeric_boost
#define rocsparse_scsritilu0_compute rocsparse_test_wrap_scsritilu0_compute
#define rocsparse_scsritilu0_compute_ex rocsparse_test_wrap_scsritilu0_compute_ex
#define rocsparse_scsritilu0_history rocsparse_test_wrap_scsritilu0_history
#define rocsparse_scsritsv_analysis rocsparse_test_wrap_scsritsv_analysis
#define rocsparse_scsritsv_buffer_size rocsparse_test_wrap_scsritsv_buffer_size
#define rocsparse_scsritsv_solve rocsparse_test_wrap_scsritsv_solve
#define rocsparse_scsritsv_solve_ex rocsparse_test_wrap_scsritsv_solve_ex
#define rocsparse_scsrmm rocsparse_test_wrap_scsrmm
#define rocsparse_scsrmv rocsparse_test_wrap_scsrmv
#define rocsparse_scsrmv_analysis rocsparse_test_wrap_scsrmv_analysis
#define rocsparse_scsrsm_analysis rocsparse_test_wrap_scsrsm_analysis
#define rocsparse_scsrsm_buffer_size rocsparse_test_wrap_scsrsm_buffer_size
#define rocsparse_scsrsm_solve rocsparse_test_wrap_scsrsm_solve
#define rocsparse_scsrsv_analysis rocsparse_test_wrap_scsrsv_analysis
#define rocsparse_scsrsv_buffer_size rocsparse_test_wrap_scsrsv_buffer_size
#define rocsparse_scsrsv_solve rocsparse_test_wrap_scsrsv_solve
#define rocsparse_sddmm rocsparse_test_wrap_sddmm
#define rocsparse_sddmm_buffer_size rocsparse_test_wrap_sddmm_buffer_size
#define rocsparse_sddmm_preprocess rocsparse_test_wrap_sddmm_preprocess
#define rocsparse_sdense2coo rocsparse_test_wrap_sdense2coo
#define rocsparse_sdense2csc rocsparse_test_wrap_sdense2csc
#define rocsparse_sdense2csr rocsparse_test_wrap_sdense2csr
#define rocsparse_sdoti rocsparse_test_wrap_sdoti
#define rocsparse_sell2csr rocsparse_test_wrap_sell2csr
#define rocsparse_sellmv rocsparse_test_wrap_sellmv
#define rocsparse_set_identity_permutation rocsparse_test_wrap_set_identity_permutation
#define rocsparse_set_pointer_mode rocsparse_test_wrap_set_pointer_mode
#define rocsparse_set_stream rocsparse_test_wrap_set_stream
#define rocsparse_sgebsr2csr rocsparse_test_wrap_sgebsr2csr
#define rocsparse_sgebsr2gebsc rocsparse_test_wrap_sgebsr2gebsc
#define rocsparse_sgebsr2gebsc_buffer_size rocsparse_test_wrap_sgebsr2gebsc_buffer_size
#define rocsparse_sgebsr2gebsr rocsparse_test_wrap_sgebsr2gebsr
#define rocsparse_sgebsr2gebsr_buffer_size rocsparse_test_wrap_sgebsr2gebsr_buffer_size
#define rocsparse_sgebsrmm rocsparse_test_wrap_sgebsrmm
#define rocsparse_sgebsrmv rocsparse_test_wrap_sgebsrmv
#define rocsparse_sgemmi rocsparse_test_wrap_sgemmi
#define rocsparse_sgemvi rocsparse_test_wrap_sgemvi
#define rocsparse_sgemvi_buffer_size rocsparse_test_wrap_sgemvi_buffer_size
#define rocsparse_sgpsv_interleaved_batch rocsparse_test_wrap_sgpsv_interleaved_batch
#define rocsparse_sgpsv_interleaved_batch_buffer_size \
    rocsparse_test_wrap_sgpsv_interleaved_batch_buffer_size
#define rocsparse_sgthr rocsparse_test_wrap_sgthr
#define rocsparse_sgthrz rocsparse_test_wrap_sgthrz
#define rocsparse_sgtsv rocsparse_test_wrap_sgtsv
#define rocsparse_sgtsv_buffer_size rocsparse_test_wrap_sgtsv_buffer_size
#define rocsparse_sgtsv_interleaved_batch rocsparse_test_wrap_sgtsv_interleaved_batch
#define rocsparse_sgtsv_interleaved_batch_buffer_size \
    rocsparse_test_wrap_sgtsv_interleaved_batch_buffer_size
#define rocsparse_sgtsv_no_pivot rocsparse_test_wrap_sgtsv_no_pivot
#define rocsparse_sgtsv_no_pivot_buffer_size rocsparse_test_wrap_sgtsv_no_pivot_buffer_size
#define rocsparse_sgtsv_no_pivot_strided_batch rocsparse_test_wrap_sgtsv_no_pivot_strided_batch
#define rocsparse_sgtsv_no_pivot_strided_batch_buffer_size \
    rocsparse_test_wrap_sgtsv_no_pivot_strided_batch_buffer_size
#define rocsparse_shyb2csr rocsparse_test_wrap_shyb2csr
#define rocsparse_shybmv rocsparse_test_wrap_shybmv
#define rocsparse_snnz rocsparse_test_wrap_snnz
#define rocsparse_snnz_compress rocsparse_test_wrap_snnz_compress
#define rocsparse_sparse_to_dense rocsparse_test_wrap_sparse_to_dense
#define rocsparse_sparse_to_sparse rocsparse_test_wrap_sparse_to_sparse
#define rocsparse_sparse_to_sparse_buffer_size rocsparse_test_wrap_sparse_to_sparse_buffer_size
#define rocsparse_spgeam rocsparse_test_wrap_spgeam
#define rocsparse_spgeam_buffer_size rocsparse_test_wrap_spgeam_buffer_size
#define rocsparse_spgeam_get_output rocsparse_test_wrap_spgeam_get_output
#define rocsparse_spgeam_set_input rocsparse_test_wrap_spgeam_set_input
#define rocsparse_spgemm rocsparse_test_wrap_spgemm
#define rocsparse_spic0 rocsparse_test_wrap_spic0
#define rocsparse_spic0_buffer_size rocsparse_test_wrap_spic0_buffer_size
#define rocsparse_spic0_descr_create rocsparse_test_wrap_spic0_descr_create
#define rocsparse_spic0_descr_destroy rocsparse_test_wrap_spic0_descr_destroy
#define rocsparse_spic0_get_output rocsparse_test_wrap_spic0_get_output
#define rocsparse_spic0_set_input rocsparse_test_wrap_spic0_set_input
#define rocsparse_spilu0 rocsparse_test_wrap_spilu0
#define rocsparse_spilu0_buffer_size rocsparse_test_wrap_spilu0_buffer_size
#define rocsparse_spilu0_descr_create rocsparse_test_wrap_spilu0_descr_create
#define rocsparse_spilu0_descr_destroy rocsparse_test_wrap_spilu0_descr_destroy
#define rocsparse_spilu0_get_output rocsparse_test_wrap_spilu0_get_output
#define rocsparse_spilu0_set_input rocsparse_test_wrap_spilu0_set_input
#define rocsparse_spitsv rocsparse_test_wrap_spitsv
#define rocsparse_spmm rocsparse_test_wrap_spmm
#define rocsparse_spmv rocsparse_test_wrap_spmv
#define rocsparse_spmv_clear_extra rocsparse_test_wrap_spmv_clear_extra
#define rocsparse_spmv_set_extra rocsparse_test_wrap_spmv_set_extra
#define rocsparse_spmv_set_input rocsparse_test_wrap_spmv_set_input
#define rocsparse_sprune_csr2csr rocsparse_test_wrap_sprune_csr2csr
#define rocsparse_sprune_csr2csr_buffer_size rocsparse_test_wrap_sprune_csr2csr_buffer_size
#define rocsparse_sprune_csr2csr_by_percentage rocsparse_test_wrap_sprune_csr2csr_by_percentage
#define rocsparse_sprune_csr2csr_by_percentage_buffer_size \
    rocsparse_test_wrap_sprune_csr2csr_by_percentage_buffer_size
#define rocsparse_sprune_csr2csr_nnz rocsparse_test_wrap_sprune_csr2csr_nnz
#define rocsparse_sprune_csr2csr_nnz_by_percentage \
    rocsparse_test_wrap_sprune_csr2csr_nnz_by_percentage
#define rocsparse_sprune_dense2csr rocsparse_test_wrap_sprune_dense2csr
#define rocsparse_sprune_dense2csr_buffer_size rocsparse_test_wrap_sprune_dense2csr_buffer_size
#define rocsparse_sprune_dense2csr_by_percentage rocsparse_test_wrap_sprune_dense2csr_by_percentage
#define rocsparse_sprune_dense2csr_by_percentage_buffer_size \
    rocsparse_test_wrap_sprune_dense2csr_by_percentage_buffer_size
#define rocsparse_sprune_dense2csr_nnz rocsparse_test_wrap_sprune_dense2csr_nnz
#define rocsparse_sprune_dense2csr_nnz_by_percentage \
    rocsparse_test_wrap_sprune_dense2csr_nnz_by_percentage
#define rocsparse_spsm rocsparse_test_wrap_spsm
#define rocsparse_spsv rocsparse_test_wrap_spsv
#define rocsparse_sptrsm rocsparse_test_wrap_sptrsm
#define rocsparse_sptrsm_buffer_size rocsparse_test_wrap_sptrsm_buffer_size
#define rocsparse_sptrsm_get_output rocsparse_test_wrap_sptrsm_get_output
#define rocsparse_sptrsm_set_input rocsparse_test_wrap_sptrsm_set_input
#define rocsparse_sptrsv rocsparse_test_wrap_sptrsv
#define rocsparse_sptrsv_buffer_size rocsparse_test_wrap_sptrsv_buffer_size
#define rocsparse_sptrsv_descr_create rocsparse_test_wrap_sptrsv_descr_create
#define rocsparse_sptrsv_descr_destroy rocsparse_test_wrap_sptrsv_descr_destroy
#define rocsparse_sptrsv_get_output rocsparse_test_wrap_sptrsv_get_output
#define rocsparse_sptrsv_set_input rocsparse_test_wrap_sptrsv_set_input
#define rocsparse_spvv rocsparse_test_wrap_spvv
#define rocsparse_sroti rocsparse_test_wrap_sroti
#define rocsparse_ssctr rocsparse_test_wrap_ssctr
#define rocsparse_v2_spmv rocsparse_test_wrap_v2_spmv
#define rocsparse_v2_spmv_buffer_size rocsparse_test_wrap_v2_spmv_buffer_size
#define rocsparse_zaxpyi rocsparse_test_wrap_zaxpyi
#define rocsparse_zbsr2csr rocsparse_test_wrap_zbsr2csr
#define rocsparse_zbsrgeam rocsparse_test_wrap_zbsrgeam
#define rocsparse_zbsrgemm rocsparse_test_wrap_zbsrgemm
#define rocsparse_zbsrgemm_buffer_size rocsparse_test_wrap_zbsrgemm_buffer_size
#define rocsparse_zbsric0 rocsparse_test_wrap_zbsric0
#define rocsparse_zbsric0_analysis rocsparse_test_wrap_zbsric0_analysis
#define rocsparse_zbsric0_buffer_size rocsparse_test_wrap_zbsric0_buffer_size
#define rocsparse_zbsrilu0 rocsparse_test_wrap_zbsrilu0
#define rocsparse_zbsrilu0_analysis rocsparse_test_wrap_zbsrilu0_analysis
#define rocsparse_zbsrilu0_buffer_size rocsparse_test_wrap_zbsrilu0_buffer_size
#define rocsparse_zbsrilu0_numeric_boost rocsparse_test_wrap_zbsrilu0_numeric_boost
#define rocsparse_zbsrmm rocsparse_test_wrap_zbsrmm
#define rocsparse_zbsrmv rocsparse_test_wrap_zbsrmv
#define rocsparse_zbsrmv_analysis rocsparse_test_wrap_zbsrmv_analysis
#define rocsparse_zbsrpad_value rocsparse_test_wrap_zbsrpad_value
#define rocsparse_zbsrsm_analysis rocsparse_test_wrap_zbsrsm_analysis
#define rocsparse_zbsrsm_buffer_size rocsparse_test_wrap_zbsrsm_buffer_size
#define rocsparse_zbsrsm_solve rocsparse_test_wrap_zbsrsm_solve
#define rocsparse_zbsrsv_analysis rocsparse_test_wrap_zbsrsv_analysis
#define rocsparse_zbsrsv_buffer_size rocsparse_test_wrap_zbsrsv_buffer_size
#define rocsparse_zbsrsv_solve rocsparse_test_wrap_zbsrsv_solve
#define rocsparse_zbsrxmv rocsparse_test_wrap_zbsrxmv
#define rocsparse_zcheck_matrix_coo rocsparse_test_wrap_zcheck_matrix_coo
#define rocsparse_zcheck_matrix_coo_buffer_size rocsparse_test_wrap_zcheck_matrix_coo_buffer_size
#define rocsparse_zcheck_matrix_csc rocsparse_test_wrap_zcheck_matrix_csc
#define rocsparse_zcheck_matrix_csc_buffer_size rocsparse_test_wrap_zcheck_matrix_csc_buffer_size
#define rocsparse_zcheck_matrix_csr rocsparse_test_wrap_zcheck_matrix_csr
#define rocsparse_zcheck_matrix_csr_buffer_size rocsparse_test_wrap_zcheck_matrix_csr_buffer_size
#define rocsparse_zcheck_matrix_ell rocsparse_test_wrap_zcheck_matrix_ell
#define rocsparse_zcheck_matrix_ell_buffer_size rocsparse_test_wrap_zcheck_matrix_ell_buffer_size
#define rocsparse_zcheck_matrix_gebsc rocsparse_test_wrap_zcheck_matrix_gebsc
#define rocsparse_zcheck_matrix_gebsc_buffer_size \
    rocsparse_test_wrap_zcheck_matrix_gebsc_buffer_size
#define rocsparse_zcheck_matrix_gebsr rocsparse_test_wrap_zcheck_matrix_gebsr
#define rocsparse_zcheck_matrix_gebsr_buffer_size \
    rocsparse_test_wrap_zcheck_matrix_gebsr_buffer_size
#define rocsparse_zcoo2dense rocsparse_test_wrap_zcoo2dense
#define rocsparse_zcoomv rocsparse_test_wrap_zcoomv
#define rocsparse_zcsc2dense rocsparse_test_wrap_zcsc2dense
#define rocsparse_zcsr2bsr rocsparse_test_wrap_zcsr2bsr
#define rocsparse_zcsr2csc rocsparse_test_wrap_zcsr2csc
#define rocsparse_zcsr2csr_compress rocsparse_test_wrap_zcsr2csr_compress
#define rocsparse_zcsr2dense rocsparse_test_wrap_zcsr2dense
#define rocsparse_zcsr2ell rocsparse_test_wrap_zcsr2ell
#define rocsparse_zcsr2gebsr rocsparse_test_wrap_zcsr2gebsr
#define rocsparse_zcsr2gebsr_buffer_size rocsparse_test_wrap_zcsr2gebsr_buffer_size
#define rocsparse_zcsr2hyb rocsparse_test_wrap_zcsr2hyb
#define rocsparse_zcsrcolor rocsparse_test_wrap_zcsrcolor
#define rocsparse_zcsrgeam rocsparse_test_wrap_zcsrgeam
#define rocsparse_zcsrgemm rocsparse_test_wrap_zcsrgemm
#define rocsparse_zcsrgemm_buffer_size rocsparse_test_wrap_zcsrgemm_buffer_size
#define rocsparse_zcsrgemm_numeric rocsparse_test_wrap_zcsrgemm_numeric
#define rocsparse_zcsric0 rocsparse_test_wrap_zcsric0
#define rocsparse_zcsric0_analysis rocsparse_test_wrap_zcsric0_analysis
#define rocsparse_zcsric0_buffer_size rocsparse_test_wrap_zcsric0_buffer_size
#define rocsparse_zcsrilu0 rocsparse_test_wrap_zcsrilu0
#define rocsparse_zcsrilu0_analysis rocsparse_test_wrap_zcsrilu0_analysis
#define rocsparse_zcsrilu0_buffer_size rocsparse_test_wrap_zcsrilu0_buffer_size
#define rocsparse_zcsrilu0_numeric_boost rocsparse_test_wrap_zcsrilu0_numeric_boost
#define rocsparse_zcsritilu0_compute rocsparse_test_wrap_zcsritilu0_compute
#define rocsparse_zcsritilu0_compute_ex rocsparse_test_wrap_zcsritilu0_compute_ex
#define rocsparse_zcsritilu0_history rocsparse_test_wrap_zcsritilu0_history
#define rocsparse_zcsritsv_analysis rocsparse_test_wrap_zcsritsv_analysis
#define rocsparse_zcsritsv_buffer_size rocsparse_test_wrap_zcsritsv_buffer_size
#define rocsparse_zcsritsv_solve rocsparse_test_wrap_zcsritsv_solve
#define rocsparse_zcsritsv_solve_ex rocsparse_test_wrap_zcsritsv_solve_ex
#define rocsparse_zcsrmm rocsparse_test_wrap_zcsrmm
#define rocsparse_zcsrmv rocsparse_test_wrap_zcsrmv
#define rocsparse_zcsrmv_analysis rocsparse_test_wrap_zcsrmv_analysis
#define rocsparse_zcsrsm_analysis rocsparse_test_wrap_zcsrsm_analysis
#define rocsparse_zcsrsm_buffer_size rocsparse_test_wrap_zcsrsm_buffer_size
#define rocsparse_zcsrsm_solve rocsparse_test_wrap_zcsrsm_solve
#define rocsparse_zcsrsv_analysis rocsparse_test_wrap_zcsrsv_analysis
#define rocsparse_zcsrsv_buffer_size rocsparse_test_wrap_zcsrsv_buffer_size
#define rocsparse_zcsrsv_solve rocsparse_test_wrap_zcsrsv_solve
#define rocsparse_zdense2coo rocsparse_test_wrap_zdense2coo
#define rocsparse_zdense2csc rocsparse_test_wrap_zdense2csc
#define rocsparse_zdense2csr rocsparse_test_wrap_zdense2csr
#define rocsparse_zdotci rocsparse_test_wrap_zdotci
#define rocsparse_zdoti rocsparse_test_wrap_zdoti
#define rocsparse_zell2csr rocsparse_test_wrap_zell2csr
#define rocsparse_zellmv rocsparse_test_wrap_zellmv
#define rocsparse_zgebsr2csr rocsparse_test_wrap_zgebsr2csr
#define rocsparse_zgebsr2gebsc rocsparse_test_wrap_zgebsr2gebsc
#define rocsparse_zgebsr2gebsc_buffer_size rocsparse_test_wrap_zgebsr2gebsc_buffer_size
#define rocsparse_zgebsr2gebsr rocsparse_test_wrap_zgebsr2gebsr
#define rocsparse_zgebsr2gebsr_buffer_size rocsparse_test_wrap_zgebsr2gebsr_buffer_size
#define rocsparse_zgebsrmm rocsparse_test_wrap_zgebsrmm
#define rocsparse_zgebsrmv rocsparse_test_wrap_zgebsrmv
#define rocsparse_zgemmi rocsparse_test_wrap_zgemmi
#define rocsparse_zgemvi rocsparse_test_wrap_zgemvi
#define rocsparse_zgemvi_buffer_size rocsparse_test_wrap_zgemvi_buffer_size
#define rocsparse_zgpsv_interleaved_batch rocsparse_test_wrap_zgpsv_interleaved_batch
#define rocsparse_zgpsv_interleaved_batch_buffer_size \
    rocsparse_test_wrap_zgpsv_interleaved_batch_buffer_size
#define rocsparse_zgthr rocsparse_test_wrap_zgthr
#define rocsparse_zgthrz rocsparse_test_wrap_zgthrz
#define rocsparse_zgtsv rocsparse_test_wrap_zgtsv
#define rocsparse_zgtsv_buffer_size rocsparse_test_wrap_zgtsv_buffer_size
#define rocsparse_zgtsv_interleaved_batch rocsparse_test_wrap_zgtsv_interleaved_batch
#define rocsparse_zgtsv_interleaved_batch_buffer_size \
    rocsparse_test_wrap_zgtsv_interleaved_batch_buffer_size
#define rocsparse_zgtsv_no_pivot rocsparse_test_wrap_zgtsv_no_pivot
#define rocsparse_zgtsv_no_pivot_buffer_size rocsparse_test_wrap_zgtsv_no_pivot_buffer_size
#define rocsparse_zgtsv_no_pivot_strided_batch rocsparse_test_wrap_zgtsv_no_pivot_strided_batch
#define rocsparse_zgtsv_no_pivot_strided_batch_buffer_size \
    rocsparse_test_wrap_zgtsv_no_pivot_strided_batch_buffer_size
#define rocsparse_zhyb2csr rocsparse_test_wrap_zhyb2csr
#define rocsparse_zhybmv rocsparse_test_wrap_zhybmv
#define rocsparse_znnz rocsparse_test_wrap_znnz
#define rocsparse_znnz_compress rocsparse_test_wrap_znnz_compress
#define rocsparse_zsctr rocsparse_test_wrap_zsctr
#define rocsparse_bell_get rocsparse_test_wrap_bell_get
#define rocsparse_bsr_get rocsparse_test_wrap_bsr_get
#define rocsparse_bsr_set_pointers rocsparse_test_wrap_bsr_set_pointers
#define rocsparse_const_bell_get rocsparse_test_wrap_const_bell_get
#define rocsparse_const_bsr_get rocsparse_test_wrap_const_bsr_get
#define rocsparse_const_coo_aos_get rocsparse_test_wrap_const_coo_aos_get
#define rocsparse_const_coo_get rocsparse_test_wrap_const_coo_get
#define rocsparse_const_csc_get rocsparse_test_wrap_const_csc_get
#define rocsparse_const_csr_get rocsparse_test_wrap_const_csr_get
#define rocsparse_const_dnmat_get rocsparse_test_wrap_const_dnmat_get
#define rocsparse_const_dnmat_get_values rocsparse_test_wrap_const_dnmat_get_values
#define rocsparse_const_dnvec_get rocsparse_test_wrap_const_dnvec_get
#define rocsparse_const_dnvec_get_values rocsparse_test_wrap_const_dnvec_get_values
#define rocsparse_const_ell_get rocsparse_test_wrap_const_ell_get
#define rocsparse_const_sell_get rocsparse_test_wrap_const_sell_get
#define rocsparse_const_spmat_get_values rocsparse_test_wrap_const_spmat_get_values
#define rocsparse_const_spvec_get rocsparse_test_wrap_const_spvec_get
#define rocsparse_const_spvec_get_values rocsparse_test_wrap_const_spvec_get_values
#define rocsparse_coo_aos_get rocsparse_test_wrap_coo_aos_get
#define rocsparse_coo_aos_set_pointers rocsparse_test_wrap_coo_aos_set_pointers
#define rocsparse_coo_get rocsparse_test_wrap_coo_get
#define rocsparse_coo_set_pointers rocsparse_test_wrap_coo_set_pointers
#define rocsparse_coo_set_strided_batch rocsparse_test_wrap_coo_set_strided_batch
#define rocsparse_copy_color_info rocsparse_test_wrap_copy_color_info
#define rocsparse_copy_hyb_mat rocsparse_test_wrap_copy_hyb_mat
#define rocsparse_copy_mat_descr rocsparse_test_wrap_copy_mat_descr
#define rocsparse_copy_mat_info rocsparse_test_wrap_copy_mat_info
#define rocsparse_create_bell_descr rocsparse_test_wrap_create_bell_descr
#define rocsparse_create_bsr_descr rocsparse_test_wrap_create_bsr_descr
#define rocsparse_create_color_info rocsparse_test_wrap_create_color_info
#define rocsparse_create_const_bell_descr rocsparse_test_wrap_create_const_bell_descr
#define rocsparse_create_const_coo_descr rocsparse_test_wrap_create_const_coo_descr
#define rocsparse_create_const_csc_descr rocsparse_test_wrap_create_const_csc_descr
#define rocsparse_create_const_csr_descr rocsparse_test_wrap_create_const_csr_descr
#define rocsparse_create_const_dnmat_descr rocsparse_test_wrap_create_const_dnmat_descr
#define rocsparse_create_const_dnvec_descr rocsparse_test_wrap_create_const_dnvec_descr
#define rocsparse_create_const_sell_descr rocsparse_test_wrap_create_const_sell_descr
#define rocsparse_create_const_spvec_descr rocsparse_test_wrap_create_const_spvec_descr
#define rocsparse_create_coo_aos_descr rocsparse_test_wrap_create_coo_aos_descr
#define rocsparse_create_coo_descr rocsparse_test_wrap_create_coo_descr
#define rocsparse_create_csc_descr rocsparse_test_wrap_create_csc_descr
#define rocsparse_create_csr_descr rocsparse_test_wrap_create_csr_descr
#define rocsparse_create_dnmat_descr rocsparse_test_wrap_create_dnmat_descr
#define rocsparse_create_dnvec_descr rocsparse_test_wrap_create_dnvec_descr
#define rocsparse_create_ell_descr rocsparse_test_wrap_create_ell_descr
#define rocsparse_create_extract_descr rocsparse_test_wrap_create_extract_descr
#define rocsparse_create_handle rocsparse_test_wrap_create_handle
#define rocsparse_create_hyb_mat rocsparse_test_wrap_create_hyb_mat
#define rocsparse_create_mat_descr rocsparse_test_wrap_create_mat_descr
#define rocsparse_create_mat_info rocsparse_test_wrap_create_mat_info
#define rocsparse_create_sell_descr rocsparse_test_wrap_create_sell_descr
#define rocsparse_create_sparse_to_sparse_descr rocsparse_test_wrap_create_sparse_to_sparse_descr
#define rocsparse_create_spgeam_descr rocsparse_test_wrap_create_spgeam_descr
#define rocsparse_create_spmv_descr rocsparse_test_wrap_create_spmv_descr
#define rocsparse_create_sptrsm_descr rocsparse_test_wrap_create_sptrsm_descr
#define rocsparse_create_sptrsv_descr rocsparse_test_wrap_create_sptrsv_descr
#define rocsparse_create_spvec_descr rocsparse_test_wrap_create_spvec_descr
#define rocsparse_csc_get rocsparse_test_wrap_csc_get
#define rocsparse_csc_set_pointers rocsparse_test_wrap_csc_set_pointers
#define rocsparse_csc_set_strided_batch rocsparse_test_wrap_csc_set_strided_batch
#define rocsparse_csr_get rocsparse_test_wrap_csr_get
#define rocsparse_csr_set_pointers rocsparse_test_wrap_csr_set_pointers
#define rocsparse_csr_set_strided_batch rocsparse_test_wrap_csr_set_strided_batch
#define rocsparse_destroy_color_info rocsparse_test_wrap_destroy_color_info
#define rocsparse_destroy_dnmat_descr rocsparse_test_wrap_destroy_dnmat_descr
#define rocsparse_destroy_dnvec_descr rocsparse_test_wrap_destroy_dnvec_descr
#define rocsparse_destroy_error rocsparse_test_wrap_destroy_error
#define rocsparse_destroy_extract_descr rocsparse_test_wrap_destroy_extract_descr
#define rocsparse_destroy_handle rocsparse_test_wrap_destroy_handle
#define rocsparse_destroy_hyb_mat rocsparse_test_wrap_destroy_hyb_mat
#define rocsparse_destroy_mat_descr rocsparse_test_wrap_destroy_mat_descr
#define rocsparse_destroy_mat_info rocsparse_test_wrap_destroy_mat_info
#define rocsparse_destroy_sparse_to_sparse_descr rocsparse_test_wrap_destroy_sparse_to_sparse_descr
#define rocsparse_destroy_spgeam_descr rocsparse_test_wrap_destroy_spgeam_descr
#define rocsparse_destroy_spmat_descr rocsparse_test_wrap_destroy_spmat_descr
#define rocsparse_destroy_spmv_descr rocsparse_test_wrap_destroy_spmv_descr
#define rocsparse_destroy_sptrsm_descr rocsparse_test_wrap_destroy_sptrsm_descr
#define rocsparse_destroy_sptrsv_descr rocsparse_test_wrap_destroy_sptrsv_descr
#define rocsparse_destroy_spvec_descr rocsparse_test_wrap_destroy_spvec_descr
#define rocsparse_dnmat_get rocsparse_test_wrap_dnmat_get
#define rocsparse_dnmat_get_strided_batch rocsparse_test_wrap_dnmat_get_strided_batch
#define rocsparse_dnmat_get_values rocsparse_test_wrap_dnmat_get_values
#define rocsparse_dnmat_set_strided_batch rocsparse_test_wrap_dnmat_set_strided_batch
#define rocsparse_dnmat_set_values rocsparse_test_wrap_dnmat_set_values
#define rocsparse_dnvec_get rocsparse_test_wrap_dnvec_get
#define rocsparse_dnvec_get_strided_batch rocsparse_test_wrap_dnvec_get_strided_batch
#define rocsparse_dnvec_get_values rocsparse_test_wrap_dnvec_get_values
#define rocsparse_dnvec_set_strided_batch rocsparse_test_wrap_dnvec_set_strided_batch
#define rocsparse_dnvec_set_values rocsparse_test_wrap_dnvec_set_values
#define rocsparse_ell_get rocsparse_test_wrap_ell_get
#define rocsparse_ell_set_pointers rocsparse_test_wrap_ell_set_pointers
#define rocsparse_sell_get rocsparse_test_wrap_sell_get
#define rocsparse_set_mat_diag_type rocsparse_test_wrap_set_mat_diag_type
#define rocsparse_set_mat_fill_mode rocsparse_test_wrap_set_mat_fill_mode
#define rocsparse_set_mat_index_base rocsparse_test_wrap_set_mat_index_base
#define rocsparse_set_mat_storage_mode rocsparse_test_wrap_set_mat_storage_mode
#define rocsparse_set_mat_type rocsparse_test_wrap_set_mat_type
#define rocsparse_sparse_to_sparse_permissive rocsparse_test_wrap_sparse_to_sparse_permissive
#define rocsparse_spmat_get_attribute rocsparse_test_wrap_spmat_get_attribute
#define rocsparse_spmat_get_format rocsparse_test_wrap_spmat_get_format
#define rocsparse_spmat_get_index_base rocsparse_test_wrap_spmat_get_index_base
#define rocsparse_spmat_get_nnz rocsparse_test_wrap_spmat_get_nnz
#define rocsparse_spmat_get_size rocsparse_test_wrap_spmat_get_size
#define rocsparse_spmat_get_strided_batch rocsparse_test_wrap_spmat_get_strided_batch
#define rocsparse_spmat_get_values rocsparse_test_wrap_spmat_get_values
#define rocsparse_spmat_set_attribute rocsparse_test_wrap_spmat_set_attribute
#define rocsparse_spmat_set_nnz rocsparse_test_wrap_spmat_set_nnz
#define rocsparse_spmat_set_strided_batch rocsparse_test_wrap_spmat_set_strided_batch
#define rocsparse_spmat_set_values rocsparse_test_wrap_spmat_set_values
#define rocsparse_spvec_get rocsparse_test_wrap_spvec_get
#define rocsparse_spvec_get_index_base rocsparse_test_wrap_spvec_get_index_base
#define rocsparse_spvec_get_values rocsparse_test_wrap_spvec_get_values
#define rocsparse_spvec_set_values rocsparse_test_wrap_spvec_set_values

#else

#include "rocsparse.h"

#endif
