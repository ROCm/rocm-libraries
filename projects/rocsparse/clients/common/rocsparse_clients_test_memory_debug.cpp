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

#include "rocsparse_clients_test_memory_debug.hpp"
#include <fstream>
#include <gtest/gtest.h>
#include <iostream>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
namespace rocsparse_clients_test
{

#define HOST rocsparse_memory_debug_synchronicity_host
#define SYNC rocsparse_memory_debug_synchronicity_sync
#define PSYNC rocsparse_memory_debug_synchronicity_psync
#define ASYNC rocsparse_memory_debug_synchronicity_async

#define HOST_ONLY HOST
#define SYNC_ONLY SYNC
#define PSYNC_ONLY PSYNC
#define ASYNC_ONLY ASYNC
#define SYNC_OR_ASYNC SYNC | ASYNC
#define HOST_OR_SYNC HOST | SYNC
#define HOST_OR_SYNC_OR_ASYNC HOST | SYNC | ASYNC
#define HOST_OR_SYNC_OR_PSYNC HOST | SYNC | PSYNC
#define HOST_OR_SYNC_OR_PSYNC_OR_ASYNC HOST | SYNC | PSYNC | ASYNC
#define HOST_OR_PSYNC HOST | PSYNC
#define HOST_OR_PSYNC_OR_ASYNC HOST | PSYNC | ASYNC
#define HOST_OR_ASYNC HOST | ASYNC
#define SYNC_OR_PSYNC SYNC | PSYNC
#define SYNC_OR_PSYNC_OR_ASYNC SYNC | PSYNC | ASYNC
#define PSYNC_OR_ASYNC PSYNC | ASYNC

    std::map<std::string, memory_debug_synchronicity_info_t> memory_debug_t::s_map{
        {"axpby", {HOST_OR_ASYNC}},
        {"bsrgeam_nnzb", {SYNC_OR_ASYNC}},
        {"bsrgemm_nnzb", {HOST_OR_SYNC_OR_PSYNC_OR_ASYNC}},
        {"bsric0_clear", {HOST_OR_SYNC}},
        {"bsric0_zero_pivot", {SYNC_ONLY}},
        {"bsrilu0_clear", {HOST_OR_SYNC}},
        {"bsrilu0_zero_pivot", {SYNC_ONLY}},
        {"bsrmv_clear", {ASYNC_ONLY}},
        {"bsrsm_clear", {HOST_OR_SYNC}},
        {"bsrsm_zero_pivot", {SYNC_ONLY}},
        {"bsrsv_clear", {HOST_OR_SYNC}},
        {"bsrsv_zero_pivot", {SYNC_ONLY}},
        {"caxpyi", {HOST_OR_ASYNC}},
        {"cbsr2csr", {ASYNC_ONLY}},
        {"cbsrgeam", {ASYNC_ONLY}},
        {"cbsrgemm", {HOST_OR_PSYNC_OR_ASYNC}},
        {"cbsrgemm_buffer_size", {HOST_ONLY}},
        {"cbsric0", {HOST_OR_ASYNC}},
        {"cbsric0_analysis", {HOST_OR_SYNC}},
        {"cbsric0_buffer_size", {HOST_ONLY}},
        {"cbsrilu0", {HOST_OR_ASYNC}},
        {"cbsrilu0_analysis", {HOST_OR_SYNC}},
        {"cbsrilu0_buffer_size", {HOST_ONLY}},
        {"cbsrilu0_numeric_boost", {HOST_ONLY}},
        {"cbsrmm", {ASYNC_ONLY}},
        {"cbsrmv", {ASYNC_ONLY}},
        {"cbsrmv_analysis", {HOST_OR_SYNC}},
        {"cbsrpad_value", {HOST_OR_ASYNC}},
        {"cbsrsm_analysis", {HOST_OR_SYNC}},
        {"cbsrsm_buffer_size", {HOST_ONLY}},
        {"cbsrsm_solve", {HOST_OR_ASYNC}},
        {"cbsrsv_analysis", {HOST_OR_SYNC}},
        {"cbsrsv_buffer_size", {HOST_ONLY}},
        {"cbsrsv_solve", {HOST_OR_ASYNC}},
        {"cbsrxmv", {ASYNC_ONLY}},
        {"ccheck_matrix_coo", {HOST_OR_SYNC}},
        {"ccheck_matrix_coo_buffer_size", {HOST_ONLY}},
        {"ccheck_matrix_csc", {SYNC_ONLY}},
        {"ccheck_matrix_csc_buffer_size", {HOST_ONLY}},
        {"ccheck_matrix_csr", {SYNC_ONLY}},
        {"ccheck_matrix_csr_buffer_size", {HOST_ONLY}},
        {"ccheck_matrix_ell", {SYNC_ONLY}},
        {"ccheck_matrix_ell_buffer_size", {HOST_ONLY}},
        {"ccheck_matrix_gebsc", {SYNC_ONLY}},
        {"ccheck_matrix_gebsc_buffer_size", {HOST_ONLY}},
        {"ccheck_matrix_gebsr", {SYNC_ONLY}},
        {"ccheck_matrix_gebsr_buffer_size", {HOST_ONLY}},
        {"ccoo2dense", {HOST_OR_ASYNC}},
        {"ccoomv", {ASYNC_ONLY}},
        {"ccsc2dense", {HOST_OR_ASYNC}},
        {"ccsr2bsr", {HOST_OR_SYNC_OR_PSYNC}},
        {"ccsr2csc", {HOST_OR_ASYNC}},
        {"ccsr2csr_compress", {HOST_OR_PSYNC_OR_ASYNC}},
        {"ccsr2dense", {HOST_OR_ASYNC}},
        {"ccsr2ell", {ASYNC_ONLY}},
        {"ccsr2gebsr", {HOST_OR_SYNC_OR_PSYNC}},
        {"ccsr2gebsr_buffer_size", {HOST_OR_SYNC}},
        {"ccsr2hyb", {HOST_OR_PSYNC}},
        {"ccsrcolor", {PSYNC_ONLY}},
        {"ccsrgeam", {ASYNC_ONLY}},
        {"ccsrgemm", {HOST_OR_PSYNC_OR_ASYNC}},
        {"ccsrgemm_buffer_size", {HOST_ONLY}},
        {"ccsrgemm_numeric", {HOST_OR_PSYNC_OR_ASYNC}},
        {"ccsric0", {ASYNC_ONLY}},
        {"ccsric0_analysis", {HOST_OR_SYNC}},
        {"ccsric0_buffer_size", {HOST_ONLY}},
        {"ccsrilu0", {HOST_OR_ASYNC}},
        {"ccsrilu0_analysis", {HOST_OR_SYNC}},
        {"ccsrilu0_buffer_size", {HOST_ONLY}},
        {"ccsrilu0_numeric_boost", {HOST_ONLY}},
        {"ccsritilu0_compute", {HOST_OR_PSYNC}},
        {"ccsritilu0_compute_ex", {HOST_OR_PSYNC}},
        {"ccsritilu0_history", {SYNC_ONLY}},
        {"ccsritsv_analysis", {SYNC_OR_PSYNC_OR_ASYNC}},
        {"ccsritsv_buffer_size", {HOST_ONLY}},
        {"ccsritsv_solve", {SYNC_ONLY}},
        {"ccsritsv_solve_ex", {SYNC_ONLY}},
        {"ccsrmm", {ASYNC_ONLY}},
        {"ccsrmv", {HOST_OR_ASYNC}},
        {"ccsrmv_analysis", {HOST_OR_SYNC}},
        {"ccsrsm_analysis", {HOST_OR_SYNC}},
        {"ccsrsm_buffer_size", {HOST_ONLY}},
        {"ccsrsm_solve", {ASYNC_ONLY}},
        {"ccsrsv_analysis", {HOST_OR_SYNC}},
        {"ccsrsv_buffer_size", {HOST_ONLY}},
        {"ccsrsv_solve", {ASYNC_ONLY}},
        {"cdense2coo", {PSYNC_ONLY}},
        {"cdense2csc", {ASYNC_ONLY}},
        {"cdense2csr", {ASYNC_ONLY}},
        {"cdotci", {HOST_OR_ASYNC}},
        {"cdoti", {HOST_OR_ASYNC}},
        {"cell2csr", {HOST_OR_ASYNC}},
        {"cellmv", {HOST_OR_ASYNC}},
        {"cgebsr2csr", {PSYNC_OR_ASYNC}},
        {"cgebsr2gebsc", {ASYNC_ONLY}},
        {"cgebsr2gebsc_buffer_size", {HOST_ONLY}},
        {"cgebsr2gebsr", {PSYNC_ONLY}},
        {"cgebsr2gebsr_buffer_size", {HOST_ONLY}},
        {"cgebsrmm", {ASYNC_ONLY}},
        {"cgebsrmv", {ASYNC_ONLY}},
        {"cgemmi", {ASYNC_ONLY}},
        {"cgemvi", {ASYNC_ONLY}},
        {"cgemvi_buffer_size", {HOST_ONLY}},
        {"cgpsv_interleaved_batch", {HOST_OR_ASYNC}},
        {"cgpsv_interleaved_batch_buffer_size", {HOST_ONLY}},
        {"cgthr", {HOST_OR_ASYNC}},
        {"cgthrz", {HOST_OR_ASYNC}},
        {"cgtsv", {ASYNC_ONLY}},
        {"cgtsv_buffer_size", {HOST_ONLY}},
        {"cgtsv_interleaved_batch", {HOST_OR_ASYNC}},
        {"cgtsv_interleaved_batch_buffer_size", {HOST_ONLY}},
        {"cgtsv_no_pivot", {ASYNC_ONLY}},
        {"cgtsv_no_pivot_buffer_size", {HOST_ONLY}},
        {"cgtsv_no_pivot_strided_batch", {ASYNC_ONLY}},
        {"cgtsv_no_pivot_strided_batch_buffer_size", {HOST_ONLY}},
        {"check_matrix_hyb", {SYNC_ONLY}},
        {"check_matrix_hyb_buffer_size", {HOST_ONLY}},
        {"check_spmat", {HOST_OR_SYNC}},
        {"chyb2csr", {ASYNC_ONLY}},
        {"chybmv", {ASYNC_ONLY}},
        {"cnnz", {HOST_OR_SYNC_OR_ASYNC}},
        {"cnnz_compress", {HOST_OR_PSYNC}},
        {"coo2csr", {ASYNC_ONLY}},
        {"coosort_buffer_size", {HOST_ONLY}},
        {"coosort_by_column", {HOST_OR_SYNC_OR_PSYNC}},
        {"coosort_by_row", {HOST_OR_SYNC_OR_PSYNC}},
        {"create_identity_permutation", {HOST_OR_ASYNC}},
        {"cscsort", {HOST_OR_ASYNC}},
        {"cscsort_buffer_size", {HOST_ONLY}},
        {"csctr", {HOST_OR_ASYNC}},
        {"csr2bsr_nnz", {HOST_OR_SYNC_OR_PSYNC_OR_ASYNC}},
        {"csr2coo", {HOST_OR_ASYNC}},
        {"csr2csc_buffer_size", {HOST_ONLY}},
        {"csr2ell_width", {ASYNC_ONLY}},
        {"csr2gebsr_nnz", {HOST_OR_SYNC_OR_PSYNC_OR_ASYNC}},
        {"csrgeam_nnz", {SYNC_OR_ASYNC}},
        {"csrgemm_nnz", {SYNC_OR_PSYNC_OR_ASYNC}},
        {"csrgemm_symbolic", {HOST_OR_PSYNC_OR_ASYNC}},
        {"csric0_clear", {HOST_OR_SYNC}},
        {"csric0_get_tolerance", {HOST_ONLY}},
        {"csric0_set_tolerance", {HOST_ONLY}},
        {"csric0_singular_pivot", {SYNC_ONLY}},
        {"csric0_zero_pivot", {SYNC_ONLY}},
        {"csrilu0_clear", {HOST_OR_SYNC}},
        {"csrilu0_get_tolerance", {HOST_ONLY}},
        {"csrilu0_set_tolerance", {HOST_ONLY}},
        {"csrilu0_singular_pivot", {SYNC_ONLY}},
        {"csrilu0_zero_pivot", {SYNC_ONLY}},
        {"csritilu0_buffer_size", {HOST_OR_SYNC}},
        {"csritilu0_preprocess", {HOST_OR_SYNC}},
        {"csritsv_clear", {HOST_OR_SYNC}},
        {"csritsv_zero_pivot", {SYNC_ONLY}},
        {"csrmv_clear", {HOST_OR_SYNC}},
        {"csrsm_clear", {HOST_OR_SYNC}},
        {"csrsm_zero_pivot", {SYNC_ONLY}},
        {"csrsort", {HOST_OR_ASYNC}},
        {"csrsort_buffer_size", {HOST_ONLY}},
        {"csrsv_clear", {HOST_OR_SYNC}},
        {"csrsv_zero_pivot", {SYNC_ONLY}},
        {"daxpyi", {HOST_OR_ASYNC}},
        {"dbsr2csr", {ASYNC_ONLY}},
        {"dbsrgeam", {ASYNC_ONLY}},
        {"dbsrgemm", {HOST_OR_PSYNC_OR_ASYNC}},
        {"dbsrgemm_buffer_size", {HOST_ONLY}},
        {"dbsric0", {HOST_OR_ASYNC}},
        {"dbsric0_analysis", {HOST_OR_SYNC}},
        {"dbsric0_buffer_size", {HOST_ONLY}},
        {"dbsrilu0", {HOST_OR_ASYNC}},
        {"dbsrilu0_analysis", {HOST_OR_SYNC}},
        {"dbsrilu0_buffer_size", {HOST_ONLY}},
        {"dbsrilu0_numeric_boost", {HOST_ONLY}},
        {"dbsrmm", {ASYNC_ONLY}},
        {"dbsrmv", {ASYNC_ONLY}},
        {"dbsrmv_analysis", {HOST_OR_SYNC}},
        {"dbsrpad_value", {HOST_OR_ASYNC}},
        {"dbsrsm_analysis", {HOST_OR_SYNC}},
        {"dbsrsm_buffer_size", {HOST_ONLY}},
        {"dbsrsm_solve", {HOST_OR_ASYNC}},
        {"dbsrsv_analysis", {HOST_OR_SYNC}},
        {"dbsrsv_buffer_size", {HOST_ONLY}},
        {"dbsrsv_solve", {HOST_OR_ASYNC}},
        {"dbsrxmv", {HOST_OR_ASYNC}},
        {"dcbsrilu0_numeric_boost", {ASYNC_ONLY}},
        {"dccsrilu0_numeric_boost", {ASYNC_ONLY}},
        {"dcheck_matrix_coo", {HOST_OR_SYNC}},
        {"dcheck_matrix_coo_buffer_size", {HOST_ONLY}},
        {"dcheck_matrix_csc", {SYNC_ONLY}},
        {"dcheck_matrix_csc_buffer_size", {HOST_ONLY}},
        {"dcheck_matrix_csr", {SYNC_ONLY}},
        {"dcheck_matrix_csr_buffer_size", {HOST_ONLY}},
        {"dcheck_matrix_ell", {SYNC_ONLY}},
        {"dcheck_matrix_ell_buffer_size", {HOST_ONLY}},
        {"dcheck_matrix_gebsc", {SYNC_ONLY}},
        {"dcheck_matrix_gebsc_buffer_size", {HOST_ONLY}},
        {"dcheck_matrix_gebsr", {SYNC_ONLY}},
        {"dcheck_matrix_gebsr_buffer_size", {HOST_ONLY}},
        {"dcoo2dense", {HOST_OR_ASYNC}},
        {"dcoomv", {ASYNC_ONLY}},
        {"dcsc2dense", {HOST_OR_ASYNC}},
        {"dcsr2bsr", {HOST_OR_SYNC_OR_PSYNC}},
        {"dcsr2csc", {HOST_OR_ASYNC}},
        {"dcsr2csr_compress", {HOST_OR_PSYNC_OR_ASYNC}},
        {"dcsr2dense", {HOST_OR_ASYNC}},
        {"dcsr2ell", {ASYNC_ONLY}},
        {"dcsr2gebsr", {HOST_OR_SYNC_OR_PSYNC}},
        {"dcsr2gebsr_buffer_size", {HOST_OR_SYNC}},
        {"dcsr2hyb", {HOST_OR_PSYNC}},
        {"dcsrcolor", {PSYNC_ONLY}},
        {"dcsrgeam", {ASYNC_ONLY}},
        {"dcsrgemm", {HOST_OR_PSYNC_OR_ASYNC}},
        {"dcsrgemm_buffer_size", {HOST_ONLY}},
        {"dcsrgemm_numeric", {HOST_OR_PSYNC_OR_ASYNC}},
        {"dcsric0", {ASYNC_ONLY}},
        {"dcsric0_analysis", {HOST_OR_SYNC}},
        {"dcsric0_buffer_size", {HOST_ONLY}},
        {"dcsrilu0", {HOST_OR_ASYNC}},
        {"dcsrilu0_analysis", {HOST_OR_SYNC}},
        {"dcsrilu0_buffer_size", {HOST_ONLY}},
        {"dcsrilu0_numeric_boost", {HOST_ONLY}},
        {"dcsritilu0_compute", {HOST_OR_PSYNC}},
        {"dcsritilu0_compute_ex", {HOST_OR_PSYNC}},
        {"dcsritilu0_history", {SYNC_ONLY}},
        {"dcsritsv_analysis", {HOST_OR_SYNC_OR_ASYNC}},
        {"dcsritsv_buffer_size", {HOST_ONLY}},
        {"dcsritsv_solve", {HOST_OR_SYNC_OR_ASYNC}},
        {"dcsritsv_solve_ex", {HOST_OR_SYNC_OR_ASYNC}},
        {"dcsrmm", {ASYNC_ONLY}},
        {"dcsrmv", {HOST_OR_ASYNC}},
        {"dcsrmv_analysis", {HOST_OR_SYNC}},
        {"dcsrsm_analysis", {HOST_OR_SYNC}},
        {"dcsrsm_buffer_size", {HOST_ONLY}},
        {"dcsrsm_solve", {HOST_OR_ASYNC}},
        {"dcsrsv_analysis", {HOST_OR_SYNC}},
        {"dcsrsv_buffer_size", {HOST_ONLY}},
        {"dcsrsv_solve", {ASYNC_ONLY}},
        {"ddense2coo", {PSYNC_ONLY}},
        {"ddense2csc", {ASYNC_ONLY}},
        {"ddense2csr", {ASYNC_ONLY}},
        {"ddoti", {HOST_OR_ASYNC}},
        {"dell2csr", {HOST_OR_ASYNC}},
        {"dellmv", {HOST_OR_ASYNC}},
        {"dense_to_sparse", {HOST_OR_SYNC_OR_PSYNC_OR_ASYNC}},
        {"dgebsr2csr", {PSYNC_OR_ASYNC}},
        {"dgebsr2gebsc", {ASYNC_ONLY}},
        {"dgebsr2gebsc_buffer_size", {HOST_ONLY}},
        {"dgebsr2gebsr", {PSYNC_ONLY}},
        {"dgebsr2gebsr_buffer_size", {HOST_ONLY}},
        {"dgebsrmm", {ASYNC_ONLY}},
        {"dgebsrmv", {ASYNC_ONLY}},
        {"dgemmi", {ASYNC_ONLY}},
        {"dgemvi", {HOST_OR_ASYNC}},
        {"dgemvi_buffer_size", {HOST_ONLY}},
        {"dgpsv_interleaved_batch", {HOST_OR_ASYNC}},
        {"dgpsv_interleaved_batch_buffer_size", {HOST_ONLY}},
        {"dgthr", {HOST_OR_ASYNC}},
        {"dgthrz", {HOST_OR_ASYNC}},
        {"dgtsv", {ASYNC_ONLY}},
        {"dgtsv_buffer_size", {HOST_ONLY}},
        {"dgtsv_interleaved_batch", {HOST_OR_ASYNC}},
        {"dgtsv_interleaved_batch_buffer_size", {HOST_ONLY}},
        {"dgtsv_no_pivot", {ASYNC_ONLY}},
        {"dgtsv_no_pivot_buffer_size", {HOST_ONLY}},
        {"dgtsv_no_pivot_strided_batch", {ASYNC_ONLY}},
        {"dgtsv_no_pivot_strided_batch_buffer_size", {HOST_ONLY}},
        {"dhyb2csr", {ASYNC_ONLY}},
        {"dhybmv", {HOST_OR_ASYNC}},
        {"dnnz", {HOST_OR_SYNC_OR_ASYNC}},
        {"dnnz_compress", {HOST_OR_PSYNC}},
        {"dprune_csr2csr", {HOST_OR_PSYNC_OR_ASYNC}},
        {"dprune_csr2csr_buffer_size", {HOST_ONLY}},
        {"dprune_csr2csr_by_percentage", {HOST_OR_PSYNC}},
        {"dprune_csr2csr_by_percentage_buffer_size", {HOST_ONLY}},
        {"dprune_csr2csr_nnz", {PSYNC_OR_ASYNC}},
        {"dprune_csr2csr_nnz_by_percentage", {PSYNC_OR_ASYNC}},
        {"dprune_dense2csr", {ASYNC_ONLY}},
        {"dprune_dense2csr_buffer_size", {HOST_ONLY}},
        {"dprune_dense2csr_by_percentage", {PSYNC_OR_ASYNC}},
        {"dprune_dense2csr_by_percentage_buffer_size", {HOST_ONLY}},
        {"dprune_dense2csr_nnz", {SYNC_OR_PSYNC}},
        {"dprune_dense2csr_nnz_by_percentage", {SYNC_OR_PSYNC}},
        {"droti", {HOST_OR_ASYNC}},
        {"dsbsrilu0_numeric_boost", {ASYNC_ONLY}},
        {"dscsrilu0_numeric_boost", {ASYNC_ONLY}},
        {"dsctr", {HOST_OR_ASYNC}},
        {"ell2csr_nnz", {HOST_OR_SYNC}},
        {"extract", {ASYNC_ONLY}},
        {"extract_buffer_size", {HOST_ONLY}},
        {"extract_nnz", {ASYNC_ONLY}},
        {"gather", {HOST_OR_ASYNC}},
        {"gebsr2gebsr_nnz", {SYNC_OR_ASYNC}},
        {"get_git_rev", {HOST_ONLY}},
        {"get_pointer_mode", {ASYNC_ONLY}},
        {"get_stream", {HOST_ONLY}},
        {"get_version", {HOST_ONLY}},
        {"hyb2csr_buffer_size", {HOST_ONLY}},
        {"inverse_permutation", {ASYNC_ONLY}},
        {"isctr", {HOST_OR_ASYNC}},
        {"rot", {HOST_OR_ASYNC}},
        {"saxpyi", {HOST_OR_ASYNC}},
        {"sbsr2csr", {ASYNC_ONLY}},
        {"sbsrgeam", {ASYNC_ONLY}},
        {"sbsrgemm", {HOST_OR_PSYNC_OR_ASYNC}},
        {"sbsrgemm_buffer_size", {HOST_ONLY}},
        {"sbsric0", {HOST_OR_ASYNC}},
        {"sbsric0_analysis", {HOST_OR_SYNC}},
        {"sbsric0_buffer_size", {HOST_ONLY}},
        {"sbsrilu0", {HOST_OR_ASYNC}},
        {"sbsrilu0_analysis", {HOST_OR_SYNC}},
        {"sbsrilu0_buffer_size", {HOST_ONLY}},
        {"sbsrilu0_numeric_boost", {HOST_ONLY}},
        {"sbsrmm", {ASYNC_ONLY}},
        {"sbsrmv", {ASYNC_ONLY}},
        {"sbsrmv_analysis", {HOST_OR_SYNC}},
        {"sbsrpad_value", {HOST_OR_ASYNC}},
        {"sbsrsm_analysis", {HOST_OR_SYNC}},
        {"sbsrsm_buffer_size", {HOST_ONLY}},
        {"sbsrsm_solve", {HOST_OR_ASYNC}},
        {"sbsrsv_analysis", {HOST_OR_SYNC}},
        {"sbsrsv_buffer_size", {HOST_ONLY}},
        {"sbsrsv_solve", {HOST_OR_ASYNC}},
        {"sbsrxmv", {HOST_OR_ASYNC}},
        {"scatter", {HOST_OR_ASYNC}},
        {"scheck_matrix_coo", {HOST_OR_SYNC}},
        {"scheck_matrix_coo_buffer_size", {HOST_ONLY}},
        {"scheck_matrix_csc", {SYNC_ONLY}},
        {"scheck_matrix_csc_buffer_size", {HOST_ONLY}},
        {"scheck_matrix_csr", {SYNC_ONLY}},
        {"scheck_matrix_csr_buffer_size", {HOST_ONLY}},
        {"scheck_matrix_ell", {SYNC_ONLY}},
        {"scheck_matrix_ell_buffer_size", {HOST_ONLY}},
        {"scheck_matrix_gebsc", {SYNC_ONLY}},
        {"scheck_matrix_gebsc_buffer_size", {HOST_ONLY}},
        {"scheck_matrix_gebsr", {SYNC_ONLY}},
        {"scheck_matrix_gebsr_buffer_size", {HOST_ONLY}},
        {"scoo2dense", {HOST_OR_ASYNC}},
        {"scoomv", {ASYNC_ONLY}},
        {"scsc2dense", {HOST_OR_ASYNC}},
        {"scsr2bsr", {HOST_OR_SYNC_OR_PSYNC}},
        {"scsr2csc", {HOST_OR_ASYNC}},
        {"scsr2csr_compress", {HOST_OR_PSYNC_OR_ASYNC}},
        {"scsr2dense", {HOST_OR_ASYNC}},
        {"scsr2ell", {ASYNC_ONLY}},
        {"scsr2gebsr", {HOST_OR_SYNC_OR_PSYNC}},
        {"scsr2gebsr_buffer_size", {HOST_OR_SYNC}},
        {"scsr2hyb", {HOST_OR_PSYNC}},
        {"scsrcolor", {PSYNC_ONLY}},
        {"scsrgeam", {ASYNC_ONLY}},
        {"scsrgemm", {HOST_OR_PSYNC_OR_ASYNC}},
        {"scsrgemm_buffer_size", {HOST_ONLY}},
        {"scsrgemm_numeric", {HOST_OR_PSYNC_OR_ASYNC}},
        {"scsric0", {ASYNC_ONLY}},
        {"scsric0_analysis", {HOST_OR_SYNC}},
        {"scsric0_buffer_size", {HOST_ONLY}},
        {"scsrilu0", {HOST_OR_ASYNC}},
        {"scsrilu0_analysis", {HOST_OR_SYNC}},
        {"scsrilu0_buffer_size", {HOST_ONLY}},
        {"scsrilu0_numeric_boost", {HOST_ONLY}},
        {"scsritilu0_compute", {HOST_OR_PSYNC}},
        {"scsritilu0_compute_ex", {HOST_OR_PSYNC}},
        {"scsritilu0_history", {SYNC_ONLY}},
        {"scsritsv_analysis", {HOST_OR_SYNC_OR_ASYNC}},
        {"scsritsv_buffer_size", {HOST_ONLY}},
        {"scsritsv_solve", {HOST_OR_SYNC_OR_ASYNC}},
        {"scsritsv_solve_ex", {HOST_OR_SYNC_OR_ASYNC}},
        {"scsrmm", {ASYNC_ONLY}},
        {"scsrmv", {HOST_OR_ASYNC}},
        {"scsrmv_analysis", {HOST_OR_SYNC}},
        {"scsrsm_analysis", {HOST_OR_SYNC}},
        {"scsrsm_buffer_size", {HOST_ONLY}},
        {"scsrsm_solve", {HOST_OR_ASYNC}},
        {"scsrsv_analysis", {HOST_OR_SYNC}},
        {"scsrsv_buffer_size", {HOST_ONLY}},
        {"scsrsv_solve", {ASYNC_ONLY}},
        {"sddmm", {ASYNC_ONLY}},
        {"sddmm_buffer_size", {HOST_ONLY}},
        {"sddmm_preprocess", {HOST_ONLY}},
        {"sdense2coo", {PSYNC_ONLY}},
        {"sdense2csc", {ASYNC_ONLY}},
        {"sdense2csr", {ASYNC_ONLY}},
        {"sdoti", {HOST_OR_ASYNC}},
        {"sell2csr", {HOST_OR_ASYNC}},
        {"sellmv", {HOST_OR_ASYNC}},
        {"set_identity_permutation", {ASYNC_ONLY}},
        {"set_pointer_mode", {HOST_ONLY}},
        {"set_stream", {HOST_ONLY}},
        {"sgebsr2csr", {PSYNC_OR_ASYNC}},
        {"sgebsr2gebsc", {ASYNC_ONLY}},
        {"sgebsr2gebsc_buffer_size", {HOST_ONLY}},
        {"sgebsr2gebsr", {PSYNC_ONLY}},
        {"sgebsr2gebsr_buffer_size", {HOST_ONLY}},
        {"sgebsrmm", {ASYNC_ONLY}},
        {"sgebsrmv", {ASYNC_ONLY}},
        {"sgemmi", {ASYNC_ONLY}},
        {"sgemvi", {HOST_OR_ASYNC}},
        {"sgemvi_buffer_size", {HOST_ONLY}},
        {"sgpsv_interleaved_batch", {HOST_OR_ASYNC}},
        {"sgpsv_interleaved_batch_buffer_size", {HOST_ONLY}},
        {"sgthr", {HOST_OR_ASYNC}},
        {"sgthrz", {HOST_OR_ASYNC}},
        {"sgtsv", {ASYNC_ONLY}},
        {"sgtsv_buffer_size", {HOST_ONLY}},
        {"sgtsv_interleaved_batch", {HOST_OR_ASYNC}},
        {"sgtsv_interleaved_batch_buffer_size", {HOST_ONLY}},
        {"sgtsv_no_pivot", {ASYNC_ONLY}},
        {"sgtsv_no_pivot_buffer_size", {HOST_ONLY}},
        {"sgtsv_no_pivot_strided_batch", {ASYNC_ONLY}},
        {"sgtsv_no_pivot_strided_batch_buffer_size", {HOST_ONLY}},
        {"shyb2csr", {ASYNC_ONLY}},
        {"shybmv", {HOST_OR_ASYNC}},
        {"snnz", {HOST_OR_SYNC_OR_ASYNC}},
        {"snnz_compress", {HOST_OR_PSYNC}},
        {"sparse_to_dense", {HOST_OR_ASYNC}},
        {"sparse_to_sparse", {HOST_OR_SYNC_OR_PSYNC_OR_ASYNC}},
        {"sparse_to_sparse_buffer_size", {HOST_OR_SYNC_OR_PSYNC}},
        {"spgeam", {HOST_OR_SYNC_OR_PSYNC_OR_ASYNC}},
        {"spgeam_buffer_size", {HOST_ONLY}},
        {"spgeam_get_output", {HOST_ONLY}},
        {"spgeam_set_input", {HOST_ONLY}},
        {"spgemm", {HOST_OR_SYNC_OR_PSYNC_OR_ASYNC}},
        {"spic0", {HOST_OR_SYNC_OR_ASYNC}},
        {"spic0_buffer_size", {HOST_ONLY}},
        {"spic0_descr_create", {HOST_ONLY}},
        {"spic0_descr_destroy", {HOST_OR_SYNC}},
        {"spic0_get_output", {ASYNC_ONLY}},
        {"spic0_set_input", {HOST_ONLY}},
        {"spilu0", {HOST_OR_SYNC_OR_ASYNC}},
        {"spilu0_buffer_size", {HOST_ONLY}},
        {"spilu0_descr_create", {HOST_ONLY}},
        {"spilu0_descr_destroy", {HOST_OR_SYNC}},
        {"spilu0_get_output", {ASYNC_ONLY}},
        {"spilu0_set_input", {HOST_ONLY}},
        {"spitsv", {HOST_OR_SYNC_OR_PSYNC_OR_ASYNC}},
        {"spmm", {HOST_OR_ASYNC}},
        {"spmv", {HOST_OR_SYNC_OR_PSYNC_OR_ASYNC}},
        {"spmv_clear_extra", {SYNC_ONLY}},
        {"spmv_set_extra", {PSYNC_ONLY}},
        {"spmv_set_input", {HOST_ONLY}},
        {"sprune_csr2csr", {HOST_OR_PSYNC_OR_ASYNC}},
        {"sprune_csr2csr_buffer_size", {HOST_ONLY}},
        {"sprune_csr2csr_by_percentage", {HOST_OR_PSYNC}},
        {"sprune_csr2csr_by_percentage_buffer_size", {HOST_ONLY}},
        {"sprune_csr2csr_nnz", {PSYNC_OR_ASYNC}},
        {"sprune_csr2csr_nnz_by_percentage", {PSYNC_OR_ASYNC}},
        {"sprune_dense2csr", {ASYNC_ONLY}},
        {"sprune_dense2csr_buffer_size", {HOST_ONLY}},
        {"sprune_dense2csr_by_percentage", {PSYNC_OR_ASYNC}},
        {"sprune_dense2csr_by_percentage_buffer_size", {HOST_ONLY}},
        {"sprune_dense2csr_nnz", {SYNC_OR_PSYNC}},
        {"sprune_dense2csr_nnz_by_percentage", {SYNC_OR_PSYNC}},
        {"spsm", {HOST_OR_SYNC_OR_ASYNC}},
        {"spsv", {HOST_OR_SYNC_OR_ASYNC}},
        {"sptrsm", {HOST_OR_SYNC_OR_ASYNC}},
        {"sptrsm_buffer_size", {HOST_ONLY}},
        {"sptrsm_get_output", {ASYNC_ONLY}},
        {"sptrsm_set_input", {HOST_ONLY}},
        {"sptrsv", {SYNC_OR_ASYNC}},
        {"sptrsv_buffer_size", {HOST_ONLY}},
        {"sptrsv_descr_create", {HOST_ONLY}},
        {"sptrsv_descr_destroy", {HOST_OR_SYNC}},
        {"sptrsv_get_output", {SYNC_OR_ASYNC}},
        {"sptrsv_set_input", {HOST_ONLY}},
        {"spvv", {HOST_OR_ASYNC}},
        {"sroti", {HOST_OR_ASYNC}},
        {"ssctr", {HOST_OR_ASYNC}},
        {"v2_spmv", {HOST_OR_SYNC_OR_PSYNC_OR_ASYNC}},
        {"v2_spmv_buffer_size", {HOST_ONLY}},
        {"zaxpyi", {HOST_OR_ASYNC}},
        {"zbsr2csr", {ASYNC_ONLY}},
        {"zbsrgeam", {ASYNC_ONLY}},
        {"zbsrgemm", {HOST_OR_PSYNC_OR_ASYNC}},
        {"zbsrgemm_buffer_size", {HOST_ONLY}},
        {"zbsric0", {HOST_OR_ASYNC}},
        {"zbsric0_analysis", {HOST_OR_SYNC}},
        {"zbsric0_buffer_size", {HOST_ONLY}},
        {"zbsrilu0", {HOST_OR_ASYNC}},
        {"zbsrilu0_analysis", {HOST_OR_SYNC}},
        {"zbsrilu0_buffer_size", {HOST_ONLY}},
        {"zbsrilu0_numeric_boost", {HOST_ONLY}},
        {"zbsrmm", {ASYNC_ONLY}},
        {"zbsrmv", {ASYNC_ONLY}},
        {"zbsrmv_analysis", {HOST_OR_SYNC}},
        {"zbsrpad_value", {HOST_OR_ASYNC}},
        {"zbsrsm_analysis", {HOST_OR_SYNC}},
        {"zbsrsm_buffer_size", {HOST_ONLY}},
        {"zbsrsm_solve", {HOST_OR_ASYNC}},
        {"zbsrsv_analysis", {HOST_OR_SYNC}},
        {"zbsrsv_buffer_size", {HOST_ONLY}},
        {"zbsrsv_solve", {HOST_OR_ASYNC}},
        {"zbsrxmv", {ASYNC_ONLY}},
        {"zcheck_matrix_coo", {HOST_OR_SYNC}},
        {"zcheck_matrix_coo_buffer_size", {HOST_ONLY}},
        {"zcheck_matrix_csc", {SYNC_ONLY}},
        {"zcheck_matrix_csc_buffer_size", {HOST_ONLY}},
        {"zcheck_matrix_csr", {SYNC_ONLY}},
        {"zcheck_matrix_csr_buffer_size", {HOST_ONLY}},
        {"zcheck_matrix_ell", {SYNC_ONLY}},
        {"zcheck_matrix_ell_buffer_size", {HOST_ONLY}},
        {"zcheck_matrix_gebsc", {SYNC_ONLY}},
        {"zcheck_matrix_gebsc_buffer_size", {HOST_ONLY}},
        {"zcheck_matrix_gebsr", {SYNC_ONLY}},
        {"zcheck_matrix_gebsr_buffer_size", {HOST_ONLY}},
        {"zcoo2dense", {HOST_OR_ASYNC}},
        {"zcoomv", {ASYNC_ONLY}},
        {"zcsc2dense", {HOST_OR_ASYNC}},
        {"zcsr2bsr", {HOST_OR_SYNC_OR_PSYNC}},
        {"zcsr2csc", {HOST_OR_ASYNC}},
        {"zcsr2csr_compress", {HOST_OR_PSYNC_OR_ASYNC}},
        {"zcsr2dense", {HOST_OR_ASYNC}},
        {"zcsr2ell", {ASYNC_ONLY}},
        {"zcsr2gebsr", {HOST_OR_SYNC_OR_PSYNC}},
        {"zcsr2gebsr_buffer_size", {HOST_OR_SYNC}},
        {"zcsr2hyb", {HOST_OR_PSYNC}},
        {"zcsrcolor", {PSYNC_ONLY}},
        {"zcsrgeam", {ASYNC_ONLY}},
        {"zcsrgemm", {HOST_OR_PSYNC_OR_ASYNC}},
        {"zcsrgemm_buffer_size", {HOST_ONLY}},
        {"zcsrgemm_numeric", {HOST_OR_PSYNC_OR_ASYNC}},
        {"zcsric0", {HOST_OR_ASYNC}},
        {"zcsric0_analysis", {HOST_OR_SYNC}},
        {"zcsric0_buffer_size", {HOST_ONLY}},
        {"zcsrilu0", {HOST_OR_ASYNC}},
        {"zcsrilu0_analysis", {HOST_OR_SYNC}},
        {"zcsrilu0_buffer_size", {HOST_ONLY}},
        {"zcsrilu0_numeric_boost", {HOST_ONLY}},
        {"zcsritilu0_compute", {HOST_OR_PSYNC}},
        {"zcsritilu0_compute_ex", {HOST_OR_PSYNC}},
        {"zcsritilu0_history", {SYNC_ONLY}},
        {"zcsritsv_analysis", {SYNC_OR_PSYNC_OR_ASYNC}},
        {"zcsritsv_buffer_size", {HOST_ONLY}},
        {"zcsritsv_solve", {SYNC_OR_PSYNC}},
        {"zcsritsv_solve_ex", {SYNC_OR_PSYNC}},
        {"zcsrmm", {ASYNC_ONLY}},
        {"zcsrmv", {HOST_OR_ASYNC}},
        {"zcsrmv_analysis", {HOST_OR_SYNC}},
        {"zcsrsm_analysis", {HOST_OR_SYNC}},
        {"zcsrsm_buffer_size", {HOST_ONLY}},
        {"zcsrsm_solve", {ASYNC_ONLY}},
        {"zcsrsv_analysis", {HOST_OR_SYNC}},
        {"zcsrsv_buffer_size", {HOST_ONLY}},
        {"zcsrsv_solve", {ASYNC_ONLY}},
        {"zdense2coo", {PSYNC_ONLY}},
        {"zdense2csc", {ASYNC_ONLY}},
        {"zdense2csr", {ASYNC_ONLY}},
        {"zdotci", {HOST_OR_ASYNC}},
        {"zdoti", {HOST_OR_ASYNC}},
        {"zell2csr", {HOST_OR_ASYNC}},
        {"zellmv", {HOST_OR_ASYNC}},
        {"zgebsr2csr", {PSYNC_OR_ASYNC}},
        {"zgebsr2gebsc", {ASYNC_ONLY}},
        {"zgebsr2gebsc_buffer_size", {HOST_ONLY}},
        {"zgebsr2gebsr", {PSYNC_ONLY}},
        {"zgebsr2gebsr_buffer_size", {HOST_ONLY}},
        {"zgebsrmm", {ASYNC_ONLY}},
        {"zgebsrmv", {ASYNC_ONLY}},
        {"zgemmi", {ASYNC_ONLY}},
        {"zgemvi", {ASYNC_ONLY}},
        {"zgemvi_buffer_size", {HOST_ONLY}},
        {"zgpsv_interleaved_batch", {HOST_OR_ASYNC}},
        {"zgpsv_interleaved_batch_buffer_size", {HOST_ONLY}},
        {"zgthr", {HOST_OR_ASYNC}},
        {"zgthrz", {HOST_OR_ASYNC}},
        {"zgtsv", {ASYNC_ONLY}},
        {"zgtsv_buffer_size", {HOST_ONLY}},
        {"zgtsv_interleaved_batch", {HOST_OR_ASYNC}},
        {"zgtsv_interleaved_batch_buffer_size", {HOST_ONLY}},
        {"zgtsv_no_pivot", {ASYNC_ONLY}},
        {"zgtsv_no_pivot_buffer_size", {HOST_ONLY}},
        {"zgtsv_no_pivot_strided_batch", {ASYNC_ONLY}},
        {"zgtsv_no_pivot_strided_batch_buffer_size", {HOST_ONLY}},
        {"zhyb2csr", {ASYNC_ONLY}},
        {"zhybmv", {ASYNC_ONLY}},
        {"znnz", {HOST_OR_SYNC_OR_ASYNC}},
        {"znnz_compress", {HOST_OR_PSYNC}},
        {"zsctr", {HOST_OR_ASYNC}},
        {"bell_get", {HOST_ONLY}},
        {"bsr_get", {HOST_ONLY}},
        {"bsr_set_pointers", {HOST_ONLY}},
        {"const_bell_get", {HOST_ONLY}},
        {"const_bsr_get", {HOST_ONLY}},
        {"const_coo_aos_get", {HOST_ONLY}},
        {"const_coo_get", {HOST_ONLY}},
        {"const_csc_get", {HOST_ONLY}},
        {"const_csr_get", {HOST_ONLY}},
        {"const_dnmat_get", {HOST_ONLY}},
        {"const_dnmat_get_values", {HOST_ONLY}},
        {"const_dnvec_get", {HOST_ONLY}},
        {"const_dnvec_get_values", {HOST_ONLY}},
        {"const_ell_get", {HOST_ONLY}},
        {"const_sell_get", {HOST_ONLY}},
        {"const_spmat_get_values", {HOST_ONLY}},
        {"const_spvec_get", {HOST_ONLY}},
        {"const_spvec_get_values", {HOST_ONLY}},
        {"coo_aos_get", {HOST_ONLY}},
        {"coo_aos_set_pointers", {HOST_ONLY}},
        {"coo_get", {HOST_ONLY}},
        {"coo_set_pointers", {HOST_ONLY}},
        {"coo_set_strided_batch", {HOST_ONLY}},
        {"copy_color_info", {HOST_ONLY}},
        {"copy_hyb_mat", {SYNC_ONLY}},
        {"copy_mat_descr", {HOST_ONLY}},
        {"copy_mat_info", {HOST_OR_SYNC}},
        {"create_bell_descr", {HOST_ONLY}},
        {"create_bsr_descr", {HOST_ONLY}},
        {"create_color_info", {HOST_ONLY}},
        {"create_const_bell_descr", {HOST_ONLY}},
        {"create_const_coo_descr", {HOST_ONLY}},
        {"create_const_csc_descr", {HOST_ONLY}},
        {"create_const_csr_descr", {HOST_ONLY}},
        {"create_const_dnmat_descr", {HOST_ONLY}},
        {"create_const_dnvec_descr", {HOST_ONLY}},
        {"create_const_sell_descr", {HOST_ONLY}},
        {"create_const_spvec_descr", {HOST_ONLY}},
        {"create_coo_aos_descr", {HOST_ONLY}},
        {"create_coo_descr", {HOST_ONLY}},
        {"create_csc_descr", {HOST_ONLY}},
        {"create_csr_descr", {HOST_ONLY}},
        {"create_dnmat_descr", {HOST_ONLY}},
        {"create_dnvec_descr", {HOST_ONLY}},
        {"create_ell_descr", {HOST_ONLY}},
        {"create_extract_descr", {SYNC_ONLY}},
        {"create_handle", {SYNC_ONLY}},
        {"create_hyb_mat", {HOST_ONLY}},
        {"create_mat_descr", {HOST_ONLY}},
        {"create_mat_info", {HOST_ONLY}},
        {"create_sell_descr", {HOST_ONLY}},
        {"create_sparse_to_sparse_descr", {HOST_ONLY}},
        {"create_spgeam_descr", {HOST_ONLY}},
        {"create_spmv_descr", {HOST_ONLY}},
        {"create_sptrsm_descr", {HOST_ONLY}},
        {"create_sptrsv_descr", {HOST_ONLY}},
        {"create_spvec_descr", {HOST_ONLY}},
        {"csc_get", {HOST_ONLY}},
        {"csc_set_pointers", {HOST_ONLY}},
        {"csc_set_strided_batch", {HOST_ONLY}},
        {"csr_get", {HOST_ONLY}},
        {"csr_set_pointers", {HOST_ONLY}},
        {"csr_set_strided_batch", {HOST_ONLY}},
        {"destroy_color_info", {HOST_ONLY}},
        {"destroy_dnmat_descr", {HOST_ONLY}},
        {"destroy_dnvec_descr", {HOST_ONLY}},
        {"destroy_error", {HOST_ONLY}},
        {"destroy_extract_descr", {SYNC_ONLY}},
        {"destroy_handle", {SYNC_ONLY}},
        {"destroy_hyb_mat", {SYNC_ONLY}},
        {"destroy_mat_descr", {HOST_ONLY}},
        {"destroy_mat_info", {SYNC_ONLY}},
        {"destroy_sparse_to_sparse_descr", {HOST_OR_SYNC}},
        {"destroy_spgeam_descr", {SYNC_ONLY}},
        {"destroy_spmat_descr", {SYNC_OR_PSYNC}},
        {"destroy_spmv_descr", {HOST_OR_SYNC}},
        {"destroy_sptrsm_descr", {HOST_ONLY}},
        {"destroy_sptrsv_descr", {HOST_OR_SYNC}},
        {"destroy_spvec_descr", {HOST_ONLY}},
        {"dnmat_get", {HOST_ONLY}},
        {"dnmat_get_strided_batch", {HOST_ONLY}},
        {"dnmat_get_values", {HOST_ONLY}},
        {"dnmat_set_strided_batch", {HOST_ONLY}},
        {"dnmat_set_values", {HOST_ONLY}},
        {"dnvec_get", {HOST_ONLY}},
        {"dnvec_get_strided_batch", {HOST_ONLY}},
        {"dnvec_get_values", {HOST_ONLY}},
        {"dnvec_set_strided_batch", {HOST_ONLY}},
        {"dnvec_set_values", {HOST_ONLY}},
        {"ell_get", {HOST_ONLY}},
        {"ell_set_pointers", {HOST_ONLY}},
        {"sell_get", {HOST_ONLY}},
        {"set_mat_diag_type", {HOST_ONLY}},
        {"set_mat_fill_mode", {HOST_ONLY}},
        {"set_mat_index_base", {HOST_ONLY}},
        {"set_mat_storage_mode", {HOST_ONLY}},
        {"set_mat_type", {HOST_ONLY}},
        {"sparse_to_sparse_permissive", {HOST_ONLY}},
        {"spmat_get_attribute", {HOST_ONLY}},
        {"spmat_get_format", {HOST_ONLY}},
        {"spmat_get_index_base", {HOST_ONLY}},
        {"spmat_get_nnz", {HOST_ONLY}},
        {"spmat_get_size", {HOST_ONLY}},
        {"spmat_get_strided_batch", {HOST_ONLY}},
        {"spmat_get_values", {HOST_ONLY}},
        {"spmat_set_attribute", {HOST_ONLY}},
        {"spmat_set_nnz", {HOST_ONLY}},
        {"spmat_set_strided_batch", {HOST_ONLY}},
        {"spmat_set_values", {HOST_ONLY}},
        {"spvec_get", {HOST_ONLY}},
        {"spvec_get_index_base", {HOST_ONLY}},
        {"spvec_get_values", {HOST_ONLY}},
        {"spvec_set_values", {HOST_ONLY}}};

    void memory_debug_check_synchronicity(rocsparse_handle handle, const char* name)
    {

        if(memory_debug_t::instance().enabled())
        {
            auto& info = memory_debug_t::instance().get_memory_debug_synchronicity_info(name);
            const int32_t sync_value = info.get_synchronicity_value();

            rocsparse_memory_debug_synchronicity synchronicity;
            const rocsparse_memory_debug_info    memory_debug_info
                = rocsparse_memory_debug_info_synchronicity;
            rocsparse_status status = rocsparse_memory_debug_info_get(
                handle, memory_debug_info, &synchronicity, sizeof(synchronicity));
            if(status != rocsparse_status_success)
            {
                FAIL() << "rocsparse_memory_debug_info_get failed";
            }
            switch(synchronicity)
            {
            case rocsparse_memory_debug_synchronicity_async:
            {
                info.add_call(ASYNC);
                if((sync_value & rocsparse_memory_debug_synchronicity_async) == 0)
                {
                    FAIL() << "Error: rocsparse_" << name << " is declared '"
                           << memory_debug_synchronicity_t2string(sync_value)
                           << "' but production code returns 'asynchronous'";
                }
                return;
            }
            case rocsparse_memory_debug_synchronicity_sync:
            {
                info.add_call(SYNC);
                if((sync_value & rocsparse_memory_debug_synchronicity_sync) == 0)
                {
                    FAIL() << "Error: rocsparse_" << name << " is declared '"
                           << memory_debug_synchronicity_t2string(sync_value)
                           << "' but production code returns 'synchronous'";
                }
                return;
            }

            case rocsparse_memory_debug_synchronicity_psync:
            {
                info.add_call(PSYNC);
                if((sync_value & rocsparse_memory_debug_synchronicity_psync) == 0)
                {
                    FAIL() << "Error: rocsparse_" << name << " is declared '"
                           << memory_debug_synchronicity_t2string(sync_value)
                           << "' but production code returns 'partially_synchronous'";
                }
                return;
            }
            case rocsparse_memory_debug_synchronicity_host:
            {
                info.add_call(HOST);
                if((sync_value & rocsparse_memory_debug_synchronicity_host) == 0)
                {
                    FAIL() << "Error: rocsparse_" << name << " is declared '"
                           << memory_debug_synchronicity_t2string(sync_value)
                           << "' but production code returns 'host'";
                }
                return;
            }
            }
        }
    }

    memory_debug_t& memory_debug_t::instance()
    {
        static memory_debug_t s_function_properties{};
        return s_function_properties;
    }

    bool memory_debug_t::enabled() const
    {
        return this->m_enabled;
    }

    void memory_debug_t::enable()
    {
        this->m_enabled = true;
    }

    void memory_debug_t::disable()
    {
        this->m_enabled = false;
    }

    void memory_debug_t::set_sync_report_filename(const char* value)
    {
        this->m_filename = value;
    }

    memory_debug_synchronicity_info_t&
        memory_debug_t::get_memory_debug_synchronicity_info(const char* name)
    {
        // The catalog is fully populated at static initialization; missing
        // names should never occur in practice. Guard the lookup with a
        // mutex anyway so concurrent threads cannot race a tree mutation
        // (operator[] inserts on a missing key).
        static std::mutex           s_map_mutex;
        std::lock_guard<std::mutex> lock(s_map_mutex);
        auto                        it = memory_debug_t::s_map.find(name);
        if(it == memory_debug_t::s_map.end())
        {
            it = memory_debug_t::s_map.emplace(name, memory_debug_synchronicity_info_t{}).first;
        }
        return it->second;
    }

    bool memory_debug_t::get_non_permissive() const
    {
        return this->m_non_permissive;
    }
    void memory_debug_t::set_non_permissive(bool value)
    {
        this->m_non_permissive = value;
    };

    void memory_debug_t::report(rocsparse_handle handle, std::ostream& out) const
    {
        out << "[" << std::endl;
        int64_t count = 0;
        for(const auto& p : memory_debug_t::s_map)
        {
            const auto& info = p.second;
            if(info.get_ncalls() == 0)
            {
                continue;
            }
            const std::string& name                         = p.first;
            const int32_t      sync_value                   = info.get_synchronicity_value();
            const uint64_t     ncalls_synchronous           = info.get_calls(SYNC);
            const uint64_t     ncalls_asynchronous          = info.get_calls(ASYNC);
            const uint64_t     ncalls_partially_synchronous = info.get_calls(PSYNC);
            const uint64_t     ncalls_host                  = info.get_calls(HOST);

            if(count > 0)
            {
                out << ", " << std::endl;
            }

            out << "{\"name\": \"rocsparse_" << name << "\"," << std::endl;

            out << " \"synchronicity\": { \"name\" : \""
                << memory_debug_synchronicity_t2string(sync_value) << "\"," << std::endl;
            out << "                     \"value\" : \"" << sync_value << "\"," << std::endl;

            out << "                     \"calls\": { \"host\": " << ncalls_host << ","
                << std::endl;
            out << "                                \"sync\": " << ncalls_synchronous << ","
                << std::endl;
            out << "                                \"psync\": " << ncalls_partially_synchronous
                << "," << std::endl;
            out << "                                \"async\": " << ncalls_asynchronous << "}}}";
            ++count;
        }

        out << std::endl << "]" << std::endl;
    }

    const std::string& memory_debug_t::get_filename() const
    {
        return this->m_filename;
    }

    rocsparse_status
        memory_debug_t::check(rocsparse_handle handle, bool non_permissive, std::ostream& out) const
    {
        bool failed = false;
        for(const auto& p : memory_debug_t::s_map)
        {
            const auto& info = p.second;
            if(info.get_ncalls() == 0)
            {
                continue;
            }

            const std::string& name       = p.first;
            const int32_t      sync_value = info.get_synchronicity_value();

            const uint64_t ncalls_sync  = info.get_calls(SYNC);
            const uint64_t ncalls_async = info.get_calls(ASYNC);
            const uint64_t ncalls_psync = info.get_calls(PSYNC);
            const uint64_t ncalls_host  = info.get_calls(HOST);

            if(non_permissive)
            {
                bool inconsistent = (sync_value == 0);
                inconsistent |= ((((sync_value & rocsparse_memory_debug_synchronicity_host) != 0)
                                  && (ncalls_host == 0))
                                 || (((sync_value & rocsparse_memory_debug_synchronicity_host) == 0)
                                     && (ncalls_host > 0)));

                inconsistent |= ((((sync_value & rocsparse_memory_debug_synchronicity_sync) != 0)
                                  && (ncalls_sync == 0))
                                 || (((sync_value & rocsparse_memory_debug_synchronicity_sync) == 0)
                                     && (ncalls_sync > 0)));

                inconsistent
                    |= ((((sync_value & rocsparse_memory_debug_synchronicity_psync) != 0)
                         && (ncalls_psync == 0))
                        || (((sync_value & rocsparse_memory_debug_synchronicity_psync) == 0)
                            && (ncalls_psync > 0)));

                inconsistent
                    |= ((((sync_value & rocsparse_memory_debug_synchronicity_async) != 0)
                         && (ncalls_async == 0))
                        || (((sync_value & rocsparse_memory_debug_synchronicity_async) == 0)
                            && (ncalls_async > 0)));

                if(inconsistent)
                {
                    out << "Error: rocsparse_" << name << " is declared '"
                        << memory_debug_synchronicity_t2string(sync_value)
                        << "' but production code returns:" << std::endl;
                    out << "   ncalls_synchronous           : " << ncalls_sync << std::endl;
                    out << "   ncalls_asynchronous          : " << ncalls_async << std::endl;
                    out << "   ncalls_partially_synchronous : " << ncalls_psync << std::endl;
                    out << "   ncalls_host                  : " << ncalls_host << std::endl;
                }

                failed |= inconsistent;
            }
            else
            {
                bool inconsistent = (sync_value == 0);
                if(ncalls_host > 0)
                {
                    inconsistent |= !(sync_value & rocsparse_memory_debug_synchronicity_host);
                }

                if(ncalls_sync > 0)
                {
                    inconsistent |= !(sync_value & rocsparse_memory_debug_synchronicity_sync);
                }

                if(ncalls_psync > 0)
                {
                    inconsistent |= !(sync_value & rocsparse_memory_debug_synchronicity_psync);
                }

                if(ncalls_async > 0)
                {
                    inconsistent |= !(sync_value & rocsparse_memory_debug_synchronicity_async);
                }

                if(inconsistent)
                {
                    out << "Error: rocsparse_" << name << " is declared '"
                        << memory_debug_synchronicity_t2string(sync_value)
                        << "' but production code returns:" << std::endl;
                    out << "   ncalls_synchronous           : " << ncalls_sync << std::endl;
                    out << "   ncalls_asynchronous          : " << ncalls_async << std::endl;
                    out << "   ncalls_partially_synchronous : " << ncalls_psync << std::endl;
                    out << "   ncalls_host                  : " << ncalls_host << std::endl;
                }

                failed |= inconsistent;
            }
        }
        return (failed) ? rocsparse_status_internal_error : rocsparse_status_success;
    }

}

#endif
