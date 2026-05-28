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
#ifdef GOOGLE_TEST
#include <cstring>

#include "rocsparse_clients_test_hip_debug.hpp"
#include <fstream>
#include <gtest/gtest.h>
#include <iostream>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
namespace rocsparse_clients_test
{

#define NONE rocsparse_hip_debug_api_history_none
#define SYNC rocsparse_hip_debug_api_history_sync
#define PSYNC rocsparse_hip_debug_api_history_psync
#define ASYNC rocsparse_hip_debug_api_history_async

#define SYNC_ONLY SYNC
#define PSYNC_ONLY PSYNC
#define ASYNC_ONLY ASYNC

#define SYNC_OR_ASYNC (SYNC | ASYNC)
#define NONE_OR_SYNC (NONE | SYNC)
#define NONE_OR_SYNC_OR_ASYNC (NONE | SYNC | ASYNC)
#define NONE_OR_SYNC_OR_PSYNC (NONE | SYNC | PSYNC)
#define NONE_OR_SYNC_OR_PSYNC_OR_ASYNC (NONE | SYNC | PSYNC | ASYNC)
#define NONE_OR_PSYNC (NONE | PSYNC)
#define NONE_OR_PSYNC_OR_ASYNC (NONE | PSYNC | ASYNC)
#define NONE_OR_ASYNC (NONE | ASYNC)
#define SYNC_OR_PSYNC (SYNC | PSYNC)
#define SYNC_OR_PSYNC_OR_ASYNC (SYNC | PSYNC | ASYNC)
#define PSYNC_OR_ASYNC (PSYNC | ASYNC)

    std::map<std::string, hip_debug_api_history_t> hip_debug_t::s_map{
        {"axpby", {NONE_OR_ASYNC}},
        {"bsrgeam_nnzb", {SYNC_OR_ASYNC}},
        {"bsrgemm_nnzb", {NONE_OR_SYNC_OR_PSYNC_OR_ASYNC}},
        {"bsric0_clear", {SYNC}},
        {"bsric0_zero_pivot", {SYNC_ONLY}},
        {"bsrilu0_clear", {SYNC}},
        {"bsrilu0_zero_pivot", {SYNC_ONLY}},
        {"bsrmv_clear", {ASYNC_ONLY}},
        {"bsrsm_clear", {SYNC}},
        {"bsrsm_zero_pivot", {SYNC_ONLY}},
        {"bsrsv_clear", {NONE_OR_SYNC}},
        {"bsrsv_zero_pivot", {SYNC_ONLY}},
        {"caxpyi", {NONE_OR_ASYNC}},
        {"cbsr2csr", {ASYNC_ONLY}},
        {"cbsrgeam", {ASYNC_ONLY}},
        {"cbsrgemm", {NONE_OR_PSYNC_OR_ASYNC}},
        {"cbsrgemm_buffer_size", {NONE}},
        {"cbsric0", {NONE_OR_ASYNC}},
        {"cbsric0_analysis", {NONE_OR_SYNC}},
        {"cbsric0_buffer_size", {NONE}},
        {"cbsrilu0", {NONE_OR_ASYNC}},
        {"cbsrilu0_analysis", {NONE_OR_SYNC}},
        {"cbsrilu0_buffer_size", {NONE}},
        {"cbsrilu0_numeric_boost", {NONE}},
        {"cbsrmm", {ASYNC_ONLY}},
        {"cbsrmv", {ASYNC_ONLY}},
        {"cbsrmv_analysis", {NONE_OR_SYNC}},
        {"cbsrpad_value", {NONE_OR_ASYNC}},
        {"cbsrsm_analysis", {NONE_OR_SYNC}},
        {"cbsrsm_buffer_size", {NONE}},
        {"cbsrsm_solve", {NONE_OR_ASYNC}},
        {"cbsrsv_analysis", {NONE_OR_SYNC}},
        {"cbsrsv_buffer_size", {NONE}},
        {"cbsrsv_solve", {NONE_OR_ASYNC}},
        {"cbsrxmv", {NONE_OR_ASYNC}},
        {"ccheck_matrix_coo", {NONE_OR_SYNC}},
        {"ccheck_matrix_coo_buffer_size", {NONE}},
        {"ccheck_matrix_csc", {SYNC_ONLY}},
        {"ccheck_matrix_csc_buffer_size", {NONE}},
        {"ccheck_matrix_csr", {SYNC_ONLY}},
        {"ccheck_matrix_csr_buffer_size", {NONE}},
        {"ccheck_matrix_ell", {SYNC_ONLY}},
        {"ccheck_matrix_ell_buffer_size", {NONE}},
        {"ccheck_matrix_gebsc", {SYNC_ONLY}},
        {"ccheck_matrix_gebsc_buffer_size", {NONE}},
        {"ccheck_matrix_gebsr", {SYNC_ONLY}},
        {"ccheck_matrix_gebsr_buffer_size", {NONE}},
        {"ccoo2dense", {NONE_OR_ASYNC}},
        {"ccoomv", {ASYNC_ONLY}},
        {"ccsc2dense", {NONE_OR_ASYNC}},
        {"ccsr2bsr", {NONE_OR_SYNC_OR_PSYNC}},
        {"ccsr2csc", {NONE_OR_ASYNC}},
        {"ccsr2csr_compress", {NONE_OR_PSYNC_OR_ASYNC}},
        {"ccsr2dense", {NONE_OR_ASYNC}},
        {"ccsr2ell", {ASYNC_ONLY}},
        {"ccsr2gebsr", {NONE_OR_SYNC_OR_PSYNC}},
        {"ccsr2gebsr_buffer_size", {NONE_OR_SYNC}},
        {"ccsr2hyb", {NONE_OR_PSYNC}},
        {"ccsrcolor", {PSYNC_ONLY}},
        {"ccsrgeam", {ASYNC_ONLY}},
        {"ccsrgemm", {NONE_OR_PSYNC_OR_ASYNC}},
        {"ccsrgemm_buffer_size", {NONE}},
        {"ccsrgemm_numeric", {NONE_OR_PSYNC_OR_ASYNC}},
        {"ccsric0", {NONE_OR_ASYNC}},
        {"ccsric0_analysis", {NONE_OR_SYNC}},
        {"ccsric0_buffer_size", {NONE}},
        {"ccsrilu0", {NONE_OR_ASYNC}},
        {"ccsrilu0_analysis", {NONE_OR_SYNC}},
        {"ccsrilu0_buffer_size", {NONE}},
        {"ccsrilu0_numeric_boost", {NONE}},
        {"ccsritilu0_compute", {NONE_OR_PSYNC}},
        {"ccsritilu0_compute_ex", {NONE_OR_PSYNC}},
        {"ccsritilu0_history", {SYNC_ONLY}},
        {"ccsritsv_analysis", {NONE_OR_SYNC_OR_PSYNC_OR_ASYNC}},
        {"ccsritsv_buffer_size", {NONE}},
        {"ccsritsv_solve", {NONE_OR_SYNC_OR_PSYNC_OR_ASYNC}},
        {"ccsritsv_solve_ex", {NONE_OR_SYNC_OR_PSYNC_OR_ASYNC}},
        {"ccsrmm", {ASYNC_ONLY}},
        {"ccsrmv", {NONE_OR_ASYNC}},
        {"ccsrmv_analysis", {NONE_OR_SYNC}},
        {"ccsrsm_analysis", {NONE_OR_SYNC}},
        {"ccsrsm_buffer_size", {NONE}},
        {"ccsrsm_solve", {NONE_OR_ASYNC}},
        {"ccsrsv_analysis", {NONE_OR_SYNC}},
        {"ccsrsv_buffer_size", {NONE}},
        {"ccsrsv_solve", {ASYNC_ONLY}},
        {"cdense2coo", {PSYNC_ONLY}},
        {"cdense2csc", {ASYNC_ONLY}},
        {"cdense2csr", {ASYNC_ONLY}},
        {"cdotci", {NONE_OR_ASYNC}},
        {"cdoti", {NONE_OR_ASYNC}},
        {"cell2csr", {NONE_OR_ASYNC}},
        {"cellmv", {NONE_OR_ASYNC}},
        {"cgebsr2csr", {PSYNC_OR_ASYNC}},
        {"cgebsr2gebsc", {ASYNC_ONLY}},
        {"cgebsr2gebsc_buffer_size", {NONE}},
        {"cgebsr2gebsr", {PSYNC_ONLY}},
        {"cgebsr2gebsr_buffer_size", {NONE}},
        {"cgebsrmm", {ASYNC_ONLY}},
        {"cgebsrmv", {ASYNC_ONLY}},
        {"cgemmi", {ASYNC_ONLY}},
        {"cgemvi", {NONE_OR_ASYNC}},
        {"cgemvi_buffer_size", {NONE}},
        {"cgpsv_interleaved_batch", {NONE_OR_ASYNC}},
        {"cgpsv_interleaved_batch_buffer_size", {NONE}},
        {"cgthr", {NONE_OR_ASYNC}},
        {"cgthrz", {NONE_OR_ASYNC}},
        {"cgtsv", {ASYNC_ONLY}},
        {"cgtsv_buffer_size", {NONE}},
        {"cgtsv_interleaved_batch", {NONE_OR_ASYNC}},
        {"cgtsv_interleaved_batch_buffer_size", {NONE}},
        {"cgtsv_no_pivot", {ASYNC_ONLY}},
        {"cgtsv_no_pivot_buffer_size", {NONE}},
        {"cgtsv_no_pivot_strided_batch", {ASYNC_ONLY}},
        {"cgtsv_no_pivot_strided_batch_buffer_size", {NONE}},
        {"check_matrix_hyb", {SYNC_ONLY}},
        {"check_matrix_hyb_buffer_size", {NONE}},
        {"check_spmat", {NONE_OR_SYNC}},
        {"chyb2csr", {ASYNC_ONLY}},
        {"chybmv", {NONE_OR_ASYNC}},
        {"cnnz", {NONE_OR_SYNC_OR_ASYNC}},
        {"cnnz_compress", {NONE_OR_PSYNC}},
        {"coo2csr", {ASYNC_ONLY}},
        {"coosort_buffer_size", {NONE}},
        {"coosort_by_column", {NONE_OR_SYNC_OR_PSYNC}},
        {"coosort_by_row", {NONE_OR_SYNC_OR_PSYNC}},
        {"create_identity_permutation", {NONE_OR_ASYNC}},
        {"cscsort", {NONE_OR_ASYNC}},
        {"cscsort_buffer_size", {NONE}},
        {"csctr", {NONE_OR_ASYNC}},
        {"csr2bsr_nnz", {NONE_OR_SYNC_OR_PSYNC_OR_ASYNC}},
        {"csr2coo", {NONE_OR_ASYNC}},
        {"csr2csc_buffer_size", {NONE}},
        {"csr2ell_width", {ASYNC_ONLY}},
        {"csr2gebsr_nnz", {NONE_OR_SYNC_OR_PSYNC_OR_ASYNC}},
        {"csrgeam_nnz", {SYNC_OR_ASYNC}},
        {"csrgemm_nnz", {SYNC_OR_PSYNC_OR_ASYNC}},
        {"csrgemm_symbolic", {NONE_OR_PSYNC_OR_ASYNC}},
        {"csric0_clear", {SYNC}},
        {"csric0_get_tolerance", {NONE}},
        {"csric0_set_tolerance", {NONE}},
        {"csric0_singular_pivot", {SYNC_ONLY}},
        {"csric0_zero_pivot", {SYNC_ONLY}},
        {"csrilu0_clear", {SYNC}},
        {"csrilu0_get_tolerance", {NONE}},
        {"csrilu0_set_tolerance", {NONE}},
        {"csrilu0_singular_pivot", {SYNC_ONLY}},
        {"csrilu0_zero_pivot", {SYNC_ONLY}},
        {"csritilu0_buffer_size", {NONE_OR_SYNC}},
        {"csritilu0_preprocess", {NONE_OR_SYNC}},
        {"csritsv_clear", {NONE_OR_SYNC}},
        {"csritsv_zero_pivot", {SYNC_ONLY}},
        {"csrmv_clear", {NONE_OR_SYNC}},
        {"csrsm_clear", {SYNC}},
        {"csrsm_zero_pivot", {SYNC_ONLY}},
        {"csrsort", {NONE_OR_ASYNC}},
        {"csrsort_buffer_size", {NONE}},
        {"csrsv_clear", {NONE_OR_SYNC}},
        {"csrsv_zero_pivot", {SYNC_ONLY}},
        {"daxpyi", {NONE_OR_ASYNC}},
        {"dbsr2csr", {ASYNC_ONLY}},
        {"dbsrgeam", {ASYNC_ONLY}},
        {"dbsrgemm", {NONE_OR_PSYNC_OR_ASYNC}},
        {"dbsrgemm_buffer_size", {NONE}},
        {"dbsric0", {NONE_OR_ASYNC}},
        {"dbsric0_analysis", {NONE_OR_SYNC}},
        {"dbsric0_buffer_size", {NONE}},
        {"dbsrilu0", {NONE_OR_ASYNC}},
        {"dbsrilu0_analysis", {NONE_OR_SYNC}},
        {"dbsrilu0_buffer_size", {NONE}},
        {"dbsrilu0_numeric_boost", {NONE}},
        {"dbsrmm", {ASYNC_ONLY}},
        {"dbsrmv", {ASYNC_ONLY}},
        {"dbsrmv_analysis", {NONE_OR_SYNC}},
        {"dbsrpad_value", {NONE_OR_ASYNC}},
        {"dbsrsm_analysis", {NONE_OR_SYNC}},
        {"dbsrsm_buffer_size", {NONE}},
        {"dbsrsm_solve", {NONE_OR_ASYNC}},
        {"dbsrsv_analysis", {NONE_OR_SYNC}},
        {"dbsrsv_buffer_size", {NONE}},
        {"dbsrsv_solve", {NONE_OR_ASYNC}},
        {"dbsrxmv", {NONE_OR_ASYNC}},
        {"dcbsrilu0_numeric_boost", {ASYNC_ONLY}},
        {"dccsrilu0_numeric_boost", {ASYNC_ONLY}},
        {"dcheck_matrix_coo", {NONE_OR_SYNC}},
        {"dcheck_matrix_coo_buffer_size", {NONE}},
        {"dcheck_matrix_csc", {SYNC_ONLY}},
        {"dcheck_matrix_csc_buffer_size", {NONE}},
        {"dcheck_matrix_csr", {SYNC_ONLY}},
        {"dcheck_matrix_csr_buffer_size", {NONE}},
        {"dcheck_matrix_ell", {SYNC_ONLY}},
        {"dcheck_matrix_ell_buffer_size", {NONE}},
        {"dcheck_matrix_gebsc", {SYNC_ONLY}},
        {"dcheck_matrix_gebsc_buffer_size", {NONE}},
        {"dcheck_matrix_gebsr", {SYNC_ONLY}},
        {"dcheck_matrix_gebsr_buffer_size", {NONE}},
        {"dcoo2dense", {NONE_OR_ASYNC}},
        {"dcoomv", {ASYNC_ONLY}},
        {"dcsc2dense", {NONE_OR_ASYNC}},
        {"dcsr2bsr", {NONE_OR_SYNC_OR_PSYNC}},
        {"dcsr2csc", {NONE_OR_ASYNC}},
        {"dcsr2csr_compress", {NONE_OR_PSYNC_OR_ASYNC}},
        {"dcsr2dense", {NONE_OR_ASYNC}},
        {"dcsr2ell", {ASYNC_ONLY}},
        {"dcsr2gebsr", {NONE_OR_SYNC_OR_PSYNC}},
        {"dcsr2gebsr_buffer_size", {NONE_OR_SYNC}},
        {"dcsr2hyb", {NONE_OR_PSYNC}},
        {"dcsrcolor", {PSYNC_ONLY}},
        {"dcsrgeam", {ASYNC_ONLY}},
        {"dcsrgemm", {NONE_OR_PSYNC_OR_ASYNC}},
        {"dcsrgemm_buffer_size", {NONE}},
        {"dcsrgemm_numeric", {NONE_OR_PSYNC_OR_ASYNC}},
        {"dcsric0", {NONE_OR_ASYNC}},
        {"dcsric0_analysis", {NONE_OR_SYNC}},
        {"dcsric0_buffer_size", {NONE}},
        {"dcsrilu0", {NONE_OR_ASYNC}},
        {"dcsrilu0_analysis", {NONE_OR_SYNC}},
        {"dcsrilu0_buffer_size", {NONE}},
        {"dcsrilu0_numeric_boost", {NONE}},
        {"dcsritilu0_compute", {NONE_OR_PSYNC}},
        {"dcsritilu0_compute_ex", {NONE_OR_PSYNC}},
        {"dcsritilu0_history", {SYNC_ONLY}},
        {"dcsritsv_analysis", {NONE_OR_SYNC_OR_PSYNC_OR_ASYNC}},
        {"dcsritsv_buffer_size", {NONE}},
        {"dcsritsv_solve", {NONE_OR_SYNC_OR_PSYNC_OR_ASYNC}},
        {"dcsritsv_solve_ex", {NONE_OR_SYNC_OR_PSYNC_OR_ASYNC}},
        {"dcsrmm", {ASYNC_ONLY}},
        {"dcsrmv", {NONE_OR_ASYNC}},
        {"dcsrmv_analysis", {NONE_OR_SYNC}},
        {"dcsrsm_analysis", {NONE_OR_SYNC}},
        {"dcsrsm_buffer_size", {NONE}},
        {"dcsrsm_solve", {NONE_OR_ASYNC}},
        {"dcsrsv_analysis", {NONE_OR_SYNC}},
        {"dcsrsv_buffer_size", {NONE}},
        {"dcsrsv_solve", {ASYNC_ONLY}},
        {"ddense2coo", {PSYNC_ONLY}},
        {"ddense2csc", {ASYNC_ONLY}},
        {"ddense2csr", {ASYNC_ONLY}},
        {"ddoti", {NONE_OR_ASYNC}},
        {"dell2csr", {NONE_OR_ASYNC}},
        {"dellmv", {NONE_OR_ASYNC}},
        {"dense_to_sparse", {NONE_OR_SYNC_OR_PSYNC_OR_ASYNC}},
        {"dgebsr2csr", {PSYNC_OR_ASYNC}},
        {"dgebsr2gebsc", {ASYNC_ONLY}},
        {"dgebsr2gebsc_buffer_size", {NONE}},
        {"dgebsr2gebsr", {PSYNC_ONLY}},
        {"dgebsr2gebsr_buffer_size", {NONE}},
        {"dgebsrmm", {ASYNC_ONLY}},
        {"dgebsrmv", {ASYNC_ONLY}},
        {"dgemmi", {ASYNC_ONLY}},
        {"dgemvi", {NONE_OR_ASYNC}},
        {"dgemvi_buffer_size", {NONE}},
        {"dgpsv_interleaved_batch", {NONE_OR_ASYNC}},
        {"dgpsv_interleaved_batch_buffer_size", {NONE}},
        {"dgthr", {NONE_OR_ASYNC}},
        {"dgthrz", {NONE_OR_ASYNC}},
        {"dgtsv", {ASYNC_ONLY}},
        {"dgtsv_buffer_size", {NONE}},
        {"dgtsv_interleaved_batch", {NONE_OR_ASYNC}},
        {"dgtsv_interleaved_batch_buffer_size", {NONE}},
        {"dgtsv_no_pivot", {ASYNC_ONLY}},
        {"dgtsv_no_pivot_buffer_size", {NONE}},
        {"dgtsv_no_pivot_strided_batch", {ASYNC_ONLY}},
        {"dgtsv_no_pivot_strided_batch_buffer_size", {NONE}},
        {"dhyb2csr", {ASYNC_ONLY}},
        {"dhybmv", {NONE_OR_ASYNC}},
        {"dnnz", {NONE_OR_SYNC_OR_ASYNC}},
        {"dnnz_compress", {NONE_OR_PSYNC}},
        {"dprune_csr2csr", {NONE_OR_PSYNC_OR_ASYNC}},
        {"dprune_csr2csr_buffer_size", {NONE}},
        {"dprune_csr2csr_by_percentage", {NONE_OR_PSYNC}},
        {"dprune_csr2csr_by_percentage_buffer_size", {NONE}},
        {"dprune_csr2csr_nnz", {PSYNC_OR_ASYNC}},
        {"dprune_csr2csr_nnz_by_percentage", {PSYNC_OR_ASYNC}},
        {"dprune_dense2csr", {ASYNC_ONLY}},
        {"dprune_dense2csr_buffer_size", {NONE}},
        {"dprune_dense2csr_by_percentage", {PSYNC_OR_ASYNC}},
        {"dprune_dense2csr_by_percentage_buffer_size", {NONE}},
        {"dprune_dense2csr_nnz", {SYNC_OR_PSYNC}},
        {"dprune_dense2csr_nnz_by_percentage", {SYNC_OR_PSYNC}},
        {"droti", {NONE_OR_ASYNC}},
        {"dsbsrilu0_numeric_boost", {ASYNC_ONLY}},
        {"dscsrilu0_numeric_boost", {ASYNC_ONLY}},
        {"dsctr", {NONE_OR_ASYNC}},
        {"ell2csr_nnz", {NONE_OR_SYNC}},
        {"extract", {ASYNC_ONLY}},
        {"extract_buffer_size", {NONE}},
        {"extract_nnz", {ASYNC_ONLY}},
        {"gather", {NONE_OR_ASYNC}},
        {"gebsr2gebsr_nnz", {SYNC_OR_ASYNC}},
        {"get_git_rev", {NONE}},
        {"get_pointer_mode", {ASYNC_ONLY}},
        {"get_stream", {NONE}},
        {"get_version", {NONE}},
        {"hyb2csr_buffer_size", {NONE}},
        {"inverse_permutation", {ASYNC_ONLY}},
        {"isctr", {NONE_OR_ASYNC}},
        {"rot", {NONE_OR_ASYNC}},
        {"saxpyi", {NONE_OR_ASYNC}},
        {"sbsr2csr", {ASYNC_ONLY}},
        {"sbsrgeam", {ASYNC_ONLY}},
        {"sbsrgemm", {NONE_OR_PSYNC_OR_ASYNC}},
        {"sbsrgemm_buffer_size", {NONE}},
        {"sbsric0", {NONE_OR_ASYNC}},
        {"sbsric0_analysis", {NONE_OR_SYNC}},
        {"sbsric0_buffer_size", {NONE}},
        {"sbsrilu0", {NONE_OR_ASYNC}},
        {"sbsrilu0_analysis", {NONE_OR_SYNC}},
        {"sbsrilu0_buffer_size", {NONE}},
        {"sbsrilu0_numeric_boost", {NONE}},
        {"sbsrmm", {ASYNC_ONLY}},
        {"sbsrmv", {ASYNC_ONLY}},
        {"sbsrmv_analysis", {NONE_OR_SYNC}},
        {"sbsrpad_value", {NONE_OR_ASYNC}},
        {"sbsrsm_analysis", {NONE_OR_SYNC}},
        {"sbsrsm_buffer_size", {NONE}},
        {"sbsrsm_solve", {NONE_OR_ASYNC}},
        {"sbsrsv_analysis", {NONE_OR_SYNC}},
        {"sbsrsv_buffer_size", {NONE}},
        {"sbsrsv_solve", {NONE_OR_ASYNC}},
        {"sbsrxmv", {NONE_OR_ASYNC}},
        {"scatter", {NONE_OR_ASYNC}},
        {"scheck_matrix_coo", {NONE_OR_SYNC}},
        {"scheck_matrix_coo_buffer_size", {NONE}},
        {"scheck_matrix_csc", {SYNC_ONLY}},
        {"scheck_matrix_csc_buffer_size", {NONE}},
        {"scheck_matrix_csr", {SYNC_ONLY}},
        {"scheck_matrix_csr_buffer_size", {NONE}},
        {"scheck_matrix_ell", {SYNC_ONLY}},
        {"scheck_matrix_ell_buffer_size", {NONE}},
        {"scheck_matrix_gebsc", {SYNC_ONLY}},
        {"scheck_matrix_gebsc_buffer_size", {NONE}},
        {"scheck_matrix_gebsr", {SYNC_ONLY}},
        {"scheck_matrix_gebsr_buffer_size", {NONE}},
        {"scoo2dense", {NONE_OR_ASYNC}},
        {"scoomv", {ASYNC_ONLY}},
        {"scsc2dense", {NONE_OR_ASYNC}},
        {"scsr2bsr", {NONE_OR_SYNC_OR_PSYNC}},
        {"scsr2csc", {NONE_OR_ASYNC}},
        {"scsr2csr_compress", {NONE_OR_PSYNC_OR_ASYNC}},
        {"scsr2dense", {NONE_OR_ASYNC}},
        {"scsr2ell", {ASYNC_ONLY}},
        {"scsr2gebsr", {NONE_OR_SYNC_OR_PSYNC}},
        {"scsr2gebsr_buffer_size", {NONE_OR_SYNC}},
        {"scsr2hyb", {NONE_OR_PSYNC}},
        {"scsrcolor", {PSYNC_ONLY}},
        {"scsrgeam", {ASYNC_ONLY}},
        {"scsrgemm", {NONE_OR_PSYNC_OR_ASYNC}},
        {"scsrgemm_buffer_size", {NONE}},
        {"scsrgemm_numeric", {NONE_OR_PSYNC_OR_ASYNC}},
        {"scsric0", {NONE_OR_ASYNC}},
        {"scsric0_analysis", {NONE_OR_SYNC}},
        {"scsric0_buffer_size", {NONE}},
        {"scsrilu0", {NONE_OR_ASYNC}},
        {"scsrilu0_analysis", {NONE_OR_SYNC}},
        {"scsrilu0_buffer_size", {NONE}},
        {"scsrilu0_numeric_boost", {NONE}},
        {"scsritilu0_compute", {NONE_OR_PSYNC}},
        {"scsritilu0_compute_ex", {NONE_OR_PSYNC}},
        {"scsritilu0_history", {SYNC_ONLY}},
        {"scsritsv_analysis", {NONE_OR_SYNC_OR_PSYNC_OR_ASYNC}},
        {"scsritsv_buffer_size", {NONE}},
        {"scsritsv_solve", {NONE_OR_SYNC_OR_PSYNC_OR_ASYNC}},
        {"scsritsv_solve_ex", {NONE_OR_SYNC_OR_PSYNC_OR_ASYNC}},
        {"scsrmm", {ASYNC_ONLY}},
        {"scsrmv", {NONE_OR_ASYNC}},
        {"scsrmv_analysis", {NONE_OR_SYNC}},
        {"scsrsm_analysis", {NONE_OR_SYNC}},
        {"scsrsm_buffer_size", {NONE}},
        {"scsrsm_solve", {NONE_OR_ASYNC}},
        {"scsrsv_analysis", {NONE_OR_SYNC}},
        {"scsrsv_buffer_size", {NONE}},
        {"scsrsv_solve", {ASYNC_ONLY}},
        {"sddmm", {ASYNC_ONLY}},
        {"sddmm_buffer_size", {NONE}},
        {"sddmm_preprocess", {NONE}},
        {"sdense2coo", {PSYNC_ONLY}},
        {"sdense2csc", {ASYNC_ONLY}},
        {"sdense2csr", {ASYNC_ONLY}},
        {"sdoti", {NONE_OR_ASYNC}},
        {"sell2csr", {NONE_OR_ASYNC}},
        {"sellmv", {NONE_OR_ASYNC}},
        {"set_identity_permutation", {ASYNC_ONLY}},
        {"set_pointer_mode", {NONE}},
        {"set_stream", {NONE}},
        {"sgebsr2csr", {PSYNC_OR_ASYNC}},
        {"sgebsr2gebsc", {ASYNC_ONLY}},
        {"sgebsr2gebsc_buffer_size", {NONE}},
        {"sgebsr2gebsr", {PSYNC_ONLY}},
        {"sgebsr2gebsr_buffer_size", {NONE}},
        {"sgebsrmm", {ASYNC_ONLY}},
        {"sgebsrmv", {ASYNC_ONLY}},
        {"sgemmi", {ASYNC_ONLY}},
        {"sgemvi", {NONE_OR_ASYNC}},
        {"sgemvi_buffer_size", {NONE}},
        {"sgpsv_interleaved_batch", {NONE_OR_ASYNC}},
        {"sgpsv_interleaved_batch_buffer_size", {NONE}},
        {"sgthr", {NONE_OR_ASYNC}},
        {"sgthrz", {NONE_OR_ASYNC}},
        {"sgtsv", {ASYNC_ONLY}},
        {"sgtsv_buffer_size", {NONE}},
        {"sgtsv_interleaved_batch", {NONE_OR_ASYNC}},
        {"sgtsv_interleaved_batch_buffer_size", {NONE}},
        {"sgtsv_no_pivot", {ASYNC_ONLY}},
        {"sgtsv_no_pivot_buffer_size", {NONE}},
        {"sgtsv_no_pivot_strided_batch", {ASYNC_ONLY}},
        {"sgtsv_no_pivot_strided_batch_buffer_size", {NONE}},
        {"shyb2csr", {ASYNC_ONLY}},
        {"shybmv", {NONE_OR_ASYNC}},
        {"snnz", {NONE_OR_SYNC_OR_ASYNC}},
        {"snnz_compress", {NONE_OR_PSYNC}},
        {"sparse_to_dense", {NONE_OR_ASYNC}},
        {"sparse_to_sparse", {NONE_OR_SYNC_OR_PSYNC_OR_ASYNC}},
        {"sparse_to_sparse_buffer_size", {NONE_OR_SYNC}},
        {"spgeam", {NONE_OR_SYNC_OR_PSYNC_OR_ASYNC}},
        {"spgeam_buffer_size", {NONE}},
        {"spgeam_get_output", {NONE}},
        {"spgeam_set_input", {NONE}},
        {"spgemm", {NONE_OR_SYNC_OR_PSYNC_OR_ASYNC}},
        {"spic0", {NONE_OR_SYNC_OR_ASYNC}},
        {"spic0_buffer_size", {NONE}},
        {"spic0_descr_create", {NONE}},
        {"spic0_descr_destroy", {NONE_OR_SYNC}},
        {"spic0_get_output", {ASYNC_ONLY}},
        {"spic0_set_input", {NONE}},
        {"spilu0", {NONE_OR_SYNC_OR_ASYNC}},
        {"spilu0_buffer_size", {NONE}},
        {"spilu0_descr_create", {NONE}},
        {"spilu0_descr_destroy", {NONE_OR_SYNC}},
        {"spilu0_get_output", {ASYNC_ONLY}},
        {"spilu0_set_input", {NONE}},
        {"spitsv", {NONE_OR_SYNC_OR_PSYNC_OR_ASYNC}},
        {"spmm", {NONE_OR_ASYNC}},
        {"spmv", {NONE_OR_SYNC_OR_PSYNC_OR_ASYNC}},
        {"spmv_clear_extra", {SYNC_ONLY}},
        {"spmv_set_extra", {PSYNC_ONLY}},
        {"spmv_set_input", {NONE}},
        {"sprune_csr2csr", {NONE_OR_PSYNC_OR_ASYNC}},
        {"sprune_csr2csr_buffer_size", {NONE}},
        {"sprune_csr2csr_by_percentage", {NONE_OR_PSYNC}},
        {"sprune_csr2csr_by_percentage_buffer_size", {NONE}},
        {"sprune_csr2csr_nnz", {PSYNC_OR_ASYNC}},
        {"sprune_csr2csr_nnz_by_percentage", {PSYNC_OR_ASYNC}},
        {"sprune_dense2csr", {ASYNC_ONLY}},
        {"sprune_dense2csr_buffer_size", {NONE}},
        {"sprune_dense2csr_by_percentage", {PSYNC_OR_ASYNC}},
        {"sprune_dense2csr_by_percentage_buffer_size", {NONE}},
        {"sprune_dense2csr_nnz", {SYNC_OR_PSYNC}},
        {"sprune_dense2csr_nnz_by_percentage", {SYNC_OR_PSYNC}},
        {"spsm", {NONE_OR_SYNC_OR_ASYNC}},
        {"spsv", {NONE_OR_SYNC_OR_ASYNC}},
        {"sptrsm", {NONE_OR_SYNC_OR_ASYNC}},
        {"sptrsm_buffer_size", {NONE}},
        {"sptrsm_get_output", {ASYNC_ONLY}},
        {"sptrsm_set_input", {NONE}},
        {"sptrsv", {SYNC_OR_ASYNC}},
        {"sptrsv_buffer_size", {NONE}},
        {"sptrsv_descr_create", {NONE}},
        {"sptrsv_descr_destroy", {NONE_OR_SYNC}},
        {"sptrsv_get_output", {SYNC_OR_ASYNC}},
        {"sptrsv_set_input", {NONE}},
        {"spvv", {NONE_OR_ASYNC}},
        {"sroti", {NONE_OR_ASYNC}},
        {"ssctr", {NONE_OR_ASYNC}},
        {"v2_spmv", {NONE_OR_SYNC_OR_PSYNC_OR_ASYNC}},
        {"v2_spmv_buffer_size", {NONE}},
        {"zaxpyi", {NONE_OR_ASYNC}},
        {"zbsr2csr", {ASYNC_ONLY}},
        {"zbsrgeam", {ASYNC_ONLY}},
        {"zbsrgemm", {NONE_OR_PSYNC_OR_ASYNC}},
        {"zbsrgemm_buffer_size", {NONE}},
        {"zbsric0", {NONE_OR_ASYNC}},
        {"zbsric0_analysis", {NONE_OR_SYNC}},
        {"zbsric0_buffer_size", {NONE}},
        {"zbsrilu0", {NONE_OR_ASYNC}},
        {"zbsrilu0_analysis", {NONE_OR_SYNC}},
        {"zbsrilu0_buffer_size", {NONE}},
        {"zbsrilu0_numeric_boost", {NONE}},
        {"zbsrmm", {ASYNC_ONLY}},
        {"zbsrmv", {ASYNC_ONLY}},
        {"zbsrmv_analysis", {NONE_OR_SYNC}},
        {"zbsrpad_value", {NONE_OR_ASYNC}},
        {"zbsrsm_analysis", {NONE_OR_SYNC}},
        {"zbsrsm_buffer_size", {NONE}},
        {"zbsrsm_solve", {NONE_OR_ASYNC}},
        {"zbsrsv_analysis", {NONE_OR_SYNC}},
        {"zbsrsv_buffer_size", {NONE}},
        {"zbsrsv_solve", {NONE_OR_ASYNC}},
        {"zbsrxmv", {NONE_OR_ASYNC}},
        {"zcheck_matrix_coo", {NONE_OR_SYNC}},
        {"zcheck_matrix_coo_buffer_size", {NONE}},
        {"zcheck_matrix_csc", {SYNC_ONLY}},
        {"zcheck_matrix_csc_buffer_size", {NONE}},
        {"zcheck_matrix_csr", {SYNC_ONLY}},
        {"zcheck_matrix_csr_buffer_size", {NONE}},
        {"zcheck_matrix_ell", {SYNC_ONLY}},
        {"zcheck_matrix_ell_buffer_size", {NONE}},
        {"zcheck_matrix_gebsc", {SYNC_ONLY}},
        {"zcheck_matrix_gebsc_buffer_size", {NONE}},
        {"zcheck_matrix_gebsr", {SYNC_ONLY}},
        {"zcheck_matrix_gebsr_buffer_size", {NONE}},
        {"zcoo2dense", {NONE_OR_ASYNC}},
        {"zcoomv", {ASYNC_ONLY}},
        {"zcsc2dense", {NONE_OR_ASYNC}},
        {"zcsr2bsr", {NONE_OR_SYNC_OR_PSYNC}},
        {"zcsr2csc", {NONE_OR_ASYNC}},
        {"zcsr2csr_compress", {NONE_OR_PSYNC_OR_ASYNC}},
        {"zcsr2dense", {NONE_OR_ASYNC}},
        {"zcsr2ell", {ASYNC_ONLY}},
        {"zcsr2gebsr", {NONE_OR_SYNC_OR_PSYNC}},
        {"zcsr2gebsr_buffer_size", {NONE_OR_SYNC}},
        {"zcsr2hyb", {NONE_OR_PSYNC}},
        {"zcsrcolor", {PSYNC_ONLY}},
        {"zcsrgeam", {ASYNC_ONLY}},
        {"zcsrgemm", {NONE_OR_PSYNC_OR_ASYNC}},
        {"zcsrgemm_buffer_size", {NONE}},
        {"zcsrgemm_numeric", {NONE_OR_PSYNC_OR_ASYNC}},
        {"zcsric0", {NONE_OR_ASYNC}},
        {"zcsric0_analysis", {NONE_OR_SYNC}},
        {"zcsric0_buffer_size", {NONE}},
        {"zcsrilu0", {NONE_OR_ASYNC}},
        {"zcsrilu0_analysis", {NONE_OR_SYNC}},
        {"zcsrilu0_buffer_size", {NONE}},
        {"zcsrilu0_numeric_boost", {NONE}},
        {"zcsritilu0_compute", {NONE_OR_PSYNC}},
        {"zcsritilu0_compute_ex", {NONE_OR_PSYNC}},
        {"zcsritilu0_history", {SYNC_ONLY}},
        {"zcsritsv_analysis", {NONE_OR_SYNC_OR_PSYNC_OR_ASYNC}},
        {"zcsritsv_buffer_size", {NONE}},
        {"zcsritsv_solve", {NONE_OR_SYNC_OR_PSYNC_OR_ASYNC}},
        {"zcsritsv_solve_ex", {NONE_OR_SYNC_OR_PSYNC_OR_ASYNC}},
        {"zcsrmm", {ASYNC_ONLY}},
        {"zcsrmv", {NONE_OR_ASYNC}},
        {"zcsrmv_analysis", {NONE_OR_SYNC}},
        {"zcsrsm_analysis", {NONE_OR_SYNC}},
        {"zcsrsm_buffer_size", {NONE}},
        {"zcsrsm_solve", {NONE_OR_ASYNC}},
        {"zcsrsv_analysis", {NONE_OR_SYNC}},
        {"zcsrsv_buffer_size", {NONE}},
        {"zcsrsv_solve", {ASYNC_ONLY}},
        {"zdense2coo", {PSYNC_ONLY}},
        {"zdense2csc", {ASYNC_ONLY}},
        {"zdense2csr", {ASYNC_ONLY}},
        {"zdotci", {NONE_OR_ASYNC}},
        {"zdoti", {NONE_OR_ASYNC}},
        {"zell2csr", {NONE_OR_ASYNC}},
        {"zellmv", {NONE_OR_ASYNC}},
        {"zgebsr2csr", {PSYNC_OR_ASYNC}},
        {"zgebsr2gebsc", {ASYNC_ONLY}},
        {"zgebsr2gebsc_buffer_size", {NONE}},
        {"zgebsr2gebsr", {PSYNC_ONLY}},
        {"zgebsr2gebsr_buffer_size", {NONE}},
        {"zgebsrmm", {ASYNC_ONLY}},
        {"zgebsrmv", {ASYNC_ONLY}},
        {"zgemmi", {ASYNC_ONLY}},
        {"zgemvi", {NONE_OR_ASYNC}},
        {"zgemvi_buffer_size", {NONE}},
        {"zgpsv_interleaved_batch", {NONE_OR_ASYNC}},
        {"zgpsv_interleaved_batch_buffer_size", {NONE}},
        {"zgthr", {NONE_OR_ASYNC}},
        {"zgthrz", {NONE_OR_ASYNC}},
        {"zgtsv", {ASYNC_ONLY}},
        {"zgtsv_buffer_size", {NONE}},
        {"zgtsv_interleaved_batch", {NONE_OR_ASYNC}},
        {"zgtsv_interleaved_batch_buffer_size", {NONE}},
        {"zgtsv_no_pivot", {ASYNC_ONLY}},
        {"zgtsv_no_pivot_buffer_size", {NONE}},
        {"zgtsv_no_pivot_strided_batch", {ASYNC_ONLY}},
        {"zgtsv_no_pivot_strided_batch_buffer_size", {NONE}},
        {"zhyb2csr", {ASYNC_ONLY}},
        {"zhybmv", {NONE_OR_ASYNC}},
        {"znnz", {NONE_OR_SYNC_OR_ASYNC}},
        {"znnz_compress", {NONE_OR_PSYNC}},
        {"zsctr", {NONE_OR_ASYNC}},
        {"bell_get", {NONE}},
        {"bsr_get", {NONE}},
        {"bsr_set_pointers", {NONE}},
        {"const_bell_get", {NONE}},
        {"const_bsr_get", {NONE}},
        {"const_coo_aos_get", {NONE}},
        {"const_coo_get", {NONE}},
        {"const_csc_get", {NONE}},
        {"const_csr_get", {NONE}},
        {"const_dnmat_get", {NONE}},
        {"const_dnmat_get_values", {NONE}},
        {"const_dnvec_get", {NONE}},
        {"const_dnvec_get_values", {NONE}},
        {"const_ell_get", {NONE}},
        {"const_sell_get", {NONE}},
        {"const_spmat_get_values", {NONE}},
        {"const_spvec_get", {NONE}},
        {"const_spvec_get_values", {NONE}},
        {"coo_aos_get", {NONE}},
        {"coo_aos_set_pointers", {NONE}},
        {"coo_get", {NONE}},
        {"coo_set_pointers", {NONE}},
        {"coo_set_strided_batch", {NONE}},
        {"copy_color_info", {NONE}},
        {"copy_hyb_mat", {SYNC_ONLY}},
        {"copy_mat_descr", {NONE}},
        {"copy_mat_info", {NONE_OR_SYNC}},
        {"create_bell_descr", {NONE}},
        {"create_bsr_descr", {NONE}},
        {"create_color_info", {NONE}},
        {"create_const_bell_descr", {NONE}},
        {"create_const_coo_descr", {NONE}},
        {"create_const_csc_descr", {NONE}},
        {"create_const_csr_descr", {NONE}},
        {"create_const_dnmat_descr", {NONE}},
        {"create_const_dnvec_descr", {NONE}},
        {"create_const_sell_descr", {NONE}},
        {"create_const_spvec_descr", {NONE}},
        {"create_coo_aos_descr", {NONE}},
        {"create_coo_descr", {NONE}},
        {"create_csc_descr", {NONE}},
        {"create_csr_descr", {NONE}},
        {"create_dnmat_descr", {NONE}},
        {"create_dnvec_descr", {NONE}},
        {"create_ell_descr", {NONE}},
        {"create_extract_descr", {SYNC_ONLY}},
        {"create_handle", {SYNC_ONLY}},
        {"create_hyb_mat", {NONE}},
        {"create_mat_descr", {NONE}},
        {"create_mat_info", {NONE}},
        {"create_sell_descr", {NONE}},
        {"create_sparse_to_sparse_descr", {NONE}},
        {"create_spgeam_descr", {NONE}},
        {"create_spmv_descr", {NONE}},
        {"create_sptrsm_descr", {NONE}},
        {"create_sptrsv_descr", {NONE}},
        {"create_spvec_descr", {NONE}},
        {"csc_get", {NONE}},
        {"csc_set_pointers", {NONE}},
        {"csc_set_strided_batch", {NONE}},
        {"csr_get", {NONE}},
        {"csr_set_pointers", {NONE}},
        {"csr_set_strided_batch", {NONE}},
        {"destroy_color_info", {NONE}},
        {"destroy_dnmat_descr", {NONE}},
        {"destroy_dnvec_descr", {NONE}},
        {"destroy_error", {NONE}},
        {"destroy_extract_descr", {SYNC_ONLY}},
        {"destroy_handle", {SYNC_ONLY}},
        {"destroy_hyb_mat", {SYNC_ONLY}},
        {"destroy_mat_descr", {NONE}},
        {"destroy_mat_info", {SYNC_ONLY}},
        {"destroy_sparse_to_sparse_descr", {NONE_OR_SYNC}},
        {"destroy_spgeam_descr", {SYNC_ONLY}},
        {"destroy_spmat_descr", {SYNC_OR_PSYNC}},
        {"destroy_spmv_descr", {NONE_OR_SYNC}},
        {"destroy_sptrsm_descr", {NONE}},
        {"destroy_sptrsv_descr", {NONE_OR_SYNC}},
        {"destroy_spvec_descr", {NONE}},
        {"dnmat_get", {NONE}},
        {"dnmat_get_strided_batch", {NONE}},
        {"dnmat_get_values", {NONE}},
        {"dnmat_set_strided_batch", {NONE}},
        {"dnmat_set_values", {NONE}},
        {"dnvec_get", {NONE}},
        {"dnvec_get_strided_batch", {NONE}},
        {"dnvec_get_values", {NONE}},
        {"dnvec_set_strided_batch", {NONE}},
        {"dnvec_set_values", {NONE}},
        {"ell_get", {NONE}},
        {"ell_set_pointers", {NONE}},
        {"sell_get", {NONE}},
        {"set_mat_diag_type", {NONE}},
        {"set_mat_fill_mode", {NONE}},
        {"set_mat_index_base", {NONE}},
        {"set_mat_storage_mode", {NONE}},
        {"set_mat_type", {NONE}},
        {"sparse_to_sparse_permissive", {NONE}},
        {"spmat_get_attribute", {NONE}},
        {"spmat_get_format", {NONE}},
        {"spmat_get_index_base", {NONE}},
        {"spmat_get_nnz", {NONE}},
        {"spmat_get_size", {NONE}},
        {"spmat_get_strided_batch", {NONE}},
        {"spmat_get_values", {NONE}},
        {"spmat_set_attribute", {NONE}},
        {"spmat_set_nnz", {NONE}},
        {"spmat_set_strided_batch", {NONE}},
        {"spmat_set_values", {NONE}},
        {"spvec_get", {NONE}},
        {"spvec_get_index_base", {NONE}},
        {"spvec_get_values", {NONE}},
        {"spvec_set_values", {NONE}}};

    void hip_debug_check_api(rocsparse_handle handle, const char* name)
    {

        if(hip_debug_t::instance().enabled())
        {
            auto&         info       = hip_debug_t::instance().get_hip_debug_api_history(name);
            const int32_t sync_value = info.get_api_value();

            rocsparse_hip_debug_api_history api;
            const rocsparse_hip_debug_info  hip_debug_info = rocsparse_hip_debug_info_api;
            rocsparse_error*                p_error{};
            rocsparse_status                status
                = rocsparse_hip_debug_info_get(handle, hip_debug_info, &api, sizeof(api), p_error);
            if(status != rocsparse_status_success)
            {
                FAIL() << "rocsparse_hip_debug_info_get failed";
            }
            switch(api)
            {
            case rocsparse_hip_debug_api_history_async:
            {
                info.add_call(ASYNC);
                if((sync_value & rocsparse_hip_debug_api_history_async) == 0)
                {
                    FAIL() << "Error: rocsparse_" << name << " is declared '"
                           << hip_debug_api_t2string(sync_value)
                           << "' but production code returns 'asynchronous'";
                }
                return;
            }
            case rocsparse_hip_debug_api_history_sync:
            {
                info.add_call(SYNC);
                if((sync_value & rocsparse_hip_debug_api_history_sync) == 0)
                {
                    FAIL() << "Error: rocsparse_" << name << " is declared '"
                           << hip_debug_api_t2string(sync_value)
                           << "' but production code returns 'synchronous'";
                }
                return;
            }

            case rocsparse_hip_debug_api_history_psync:
            {
                info.add_call(PSYNC);
                if((sync_value & rocsparse_hip_debug_api_history_psync) == 0)
                {
                    FAIL() << "Error: rocsparse_" << name << " is declared '"
                           << hip_debug_api_t2string(sync_value)
                           << "' but production code returns 'partially_synchronous'";
                }
                return;
            }
            case rocsparse_hip_debug_api_history_none:
            {
                info.add_call(NONE);
                if((sync_value & rocsparse_hip_debug_api_history_none) == 0)
                {
                    FAIL() << "Error: rocsparse_" << name << " is declared '"
                           << hip_debug_api_t2string(sync_value)
                           << "' but production code returns 'host'";
                }
                return;
            }
            }
        }
    }

    hip_debug_t& hip_debug_t::instance()
    {
        static hip_debug_t s_function_properties{};
        return s_function_properties;
    }

    bool hip_debug_t::enabled() const
    {
        return this->m_enabled;
    }

    void hip_debug_t::enable()
    {
        this->m_enabled = true;
        rocsparse_hip_debug_enable();
    }

    void hip_debug_t::disable()
    {
        this->m_enabled = false;
        rocsparse_hip_debug_disable();
    }

    void hip_debug_t::set_hip_debug_report_filename(const char* value)
    {
        this->m_filename = value;
    }

    hip_debug_api_history_t& hip_debug_t::get_hip_debug_api_history(const char* name)
    {
        // The catalog is fully populated at static initialization; missing
        // names should never occur in practice. Guard the lookup with a
        // mutex anyway so concurrent threads cannot race a tree mutation
        // (operator[] inserts on a missing key).
        static std::mutex           s_map_mutex;
        std::lock_guard<std::mutex> lock(s_map_mutex);
        auto                        it = hip_debug_t::s_map.find(name);
        if(it == hip_debug_t::s_map.end())
        {
            it = hip_debug_t::s_map.emplace(name, hip_debug_api_history_t{}).first;
        }
        return it->second;
    }

    bool hip_debug_t::get_non_permissive() const
    {
        return this->m_non_permissive;
    }
    void hip_debug_t::set_non_permissive(bool value)
    {
        this->m_non_permissive = value;
    };

    void hip_debug_t::report(rocsparse_handle handle, std::ostream& out) const
    {
        out << "[" << std::endl;
        int64_t count = 0;
        for(const auto& p : hip_debug_t::s_map)
        {
            const auto& info = p.second;
            if(info.get_ncalls() == 0)
            {
                continue;
            }
            const std::string& name                         = p.first;
            const int32_t      sync_value                   = info.get_api_value();
            const uint64_t     ncalls_synchronous           = info.get_calls(SYNC);
            const uint64_t     ncalls_asynchronous          = info.get_calls(ASYNC);
            const uint64_t     ncalls_partially_synchronous = info.get_calls(PSYNC);
            const uint64_t     ncalls_none                  = info.get_calls(NONE);

            if(count > 0)
            {
                out << ", " << std::endl;
            }

            out << "{\"name\": \"rocsparse_" << name << "\"," << std::endl;

            out << " \"api calls\": { \"classification\": \"" << hip_debug_api_t2string(sync_value)
                << "\"," << std::endl;
            out << "                   \"classification value\": \"" << sync_value << "\","
                << std::endl;
            out << "                   \"stats\": { \"none\": " << ncalls_none << "," << std::endl;
            out << "                              \"sync\": " << ncalls_synchronous << ","
                << std::endl;
            out << "                              \"psync\": " << ncalls_partially_synchronous
                << "," << std::endl;
            out << "                              \"async\": " << ncalls_asynchronous << "}}}";
            ++count;
        }

        out << std::endl << "]" << std::endl;
    }

    const std::string& hip_debug_t::get_filename() const
    {
        return this->m_filename;
    }

    static void init()
    {
        //
        // Look  for inconsistencies in the table.
        //
        int32_t index  = 0;
        bool    failed = false;
        for(const auto& p : hip_debug_t::s_map)
        {
            const auto&   name       = p.first;
            const auto&   info       = p.second;
            const int32_t sync_value = info.get_api_value();
            for(const auto& q : hip_debug_t::s_map)
            {
                const auto& tname = q.first;
                if(name != tname)
                {
                    const char* legacy_name  = name.c_str() + 1;
                    const char* tlegacy_name = tname.c_str() + 1;

                    if((name[0] == 's' || name[0] == 'd' || name[0] == 'c' || name[0] == 'z')
                       && (tname[0] == 's' || tname[0] == 'd' || tname[0] == 'c' || tname[0] == 'z')
                       && !strcmp(legacy_name, tlegacy_name))
                    {
                        const auto&   tinfo       = q.second;
                        const int32_t tsync_value = tinfo.get_api_value();
                        if(tsync_value != sync_value)
                        {
                            std::cout << "> " << index << ". : rocsparse_" << name
                                      << " and rocsparse_" << tname << std::endl;
                            std::cout << " rocsparse_" << name << ": "
                                      << hip_debug_api_t2string(sync_value) << std::endl;
                            std::cout << " rocsparse_" << tname << ": "
                                      << hip_debug_api_t2string(tsync_value) << std::endl;
                            failed = true;
                        }
                    }
                }
            }
        }
        if(failed)
        {
            std::cerr << "// rocsparse_clients_test_hip_debug error: Consistency check failed."
                      << std::endl;
            exit(1);
        }
    }

    hip_debug_t::hip_debug_t()
    {
        init();
    }

    rocsparse_status
        hip_debug_t::check(rocsparse_handle handle, bool non_permissive, std::ostream& out) const
    {

        bool failed = false;
        for(const auto& p : hip_debug_t::s_map)
        {
            const auto& info = p.second;
            if(info.get_ncalls() == 0)
            {
                continue;
            }

            const std::string& name       = p.first;
            const int32_t      sync_value = info.get_api_value();

            const uint64_t ncalls_sync  = info.get_calls(SYNC);
            const uint64_t ncalls_async = info.get_calls(ASYNC);
            const uint64_t ncalls_psync = info.get_calls(PSYNC);
            const uint64_t ncalls_none  = info.get_calls(NONE);

            if(non_permissive)
            {
                bool inconsistent = (sync_value == 0);
                inconsistent |= ((((sync_value & rocsparse_hip_debug_api_history_none) != 0)
                                  && (ncalls_none == 0))
                                 || (((sync_value & rocsparse_hip_debug_api_history_none) == 0)
                                     && (ncalls_none > 0)));

                inconsistent |= ((((sync_value & rocsparse_hip_debug_api_history_sync) != 0)
                                  && (ncalls_sync == 0))
                                 || (((sync_value & rocsparse_hip_debug_api_history_sync) == 0)
                                     && (ncalls_sync > 0)));

                inconsistent |= ((((sync_value & rocsparse_hip_debug_api_history_psync) != 0)
                                  && (ncalls_psync == 0))
                                 || (((sync_value & rocsparse_hip_debug_api_history_psync) == 0)
                                     && (ncalls_psync > 0)));

                inconsistent |= ((((sync_value & rocsparse_hip_debug_api_history_async) != 0)
                                  && (ncalls_async == 0))
                                 || (((sync_value & rocsparse_hip_debug_api_history_async) == 0)
                                     && (ncalls_async > 0)));

                if(inconsistent)
                {
                    out << "Error: rocsparse_" << name << " is declared '"
                        << hip_debug_api_t2string(sync_value)
                        << "' but production code returns:" << std::endl;
                    out << "   ncalls_none                  : " << ncalls_none << std::endl;
                    out << "   ncalls_synchronous           : " << ncalls_sync << std::endl;
                    out << "   ncalls_partially_synchronous : " << ncalls_psync << std::endl;
                    out << "   ncalls_asynchronous          : " << ncalls_async << std::endl;
                }

                failed |= inconsistent;
            }
            else
            {
                bool inconsistent = (sync_value == 0);
                if(ncalls_none > 0)
                {
                    inconsistent |= !(sync_value & rocsparse_hip_debug_api_history_none);
                }

                if(ncalls_sync > 0)
                {
                    inconsistent |= !(sync_value & rocsparse_hip_debug_api_history_sync);
                }

                if(ncalls_psync > 0)
                {
                    inconsistent |= !(sync_value & rocsparse_hip_debug_api_history_psync);
                }

                if(ncalls_async > 0)
                {
                    inconsistent |= !(sync_value & rocsparse_hip_debug_api_history_async);
                }

                if(inconsistent)
                {
                    out << "Error: rocsparse_" << name << " is declared '"
                        << hip_debug_api_t2string(sync_value)
                        << "' but production code returns:" << std::endl;
                    out << "   ncalls_none                  : " << ncalls_none << std::endl;
                    out << "   ncalls_synchronous           : " << ncalls_sync << std::endl;
                    out << "   ncalls_partially_synchronous : " << ncalls_psync << std::endl;
                    out << "   ncalls_asynchronous          : " << ncalls_async << std::endl;
                }

                failed |= inconsistent;
            }
        }
        return (failed) ? rocsparse_status_internal_error : rocsparse_status_success;
    }

}

#endif
