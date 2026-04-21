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
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>

namespace rocsparse_clients_test
{

    static std::map<std::string, memory_debug_synchronicity_info_t> s_map{
        {"axpby", {memory_debug_synchronicity_t::host_or_asynchronous}},
        {"bsrgeam_nnzb", {memory_debug_synchronicity_t::depends}},
        {"bsrgemm_nnzb", {memory_debug_synchronicity_t::depends}},
        {"bsric0_clear", {memory_debug_synchronicity_t::host_or_synchronous}},
        {"bsric0_zero_pivot", {memory_debug_synchronicity_t::synchronous}},
        {"bsrilu0_clear", {memory_debug_synchronicity_t::host_or_synchronous}},
        {"bsrilu0_zero_pivot", {memory_debug_synchronicity_t::synchronous}},
        {"bsrmv_clear", {memory_debug_synchronicity_t::asynchronous}},
        {"bsrsm_clear", {memory_debug_synchronicity_t::host_or_synchronous}},
        {"bsrsm_zero_pivot", {memory_debug_synchronicity_t::synchronous}},
        {"bsrsv_clear", {memory_debug_synchronicity_t::host_or_synchronous}},
        {"bsrsv_zero_pivot", {memory_debug_synchronicity_t::synchronous}},
        {"caxpyi", {memory_debug_synchronicity_t::asynchronous}},
        {"cbsr2csr", {memory_debug_synchronicity_t::asynchronous}},
        {"cbsrgeam", {memory_debug_synchronicity_t::synchronous}},
        {"cbsrgemm", {memory_debug_synchronicity_t::synchronous}},
        {"cbsrgemm_buffer_size", {memory_debug_synchronicity_t::synchronous}},
        {"cbsric0", {memory_debug_synchronicity_t::asynchronous}},
        {"cbsric0_analysis", {memory_debug_synchronicity_t::synchronous}},
        {"cbsric0_buffer_size", {memory_debug_synchronicity_t::asynchronous}},
        {"cbsrilu0", {memory_debug_synchronicity_t::asynchronous}},
        {"cbsrilu0_analysis", {memory_debug_synchronicity_t::synchronous}},
        {"cbsrilu0_buffer_size", {memory_debug_synchronicity_t::asynchronous}},
        {"cbsrilu0_numeric_boost", {memory_debug_synchronicity_t::asynchronous}},
        {"cbsrmm", {memory_debug_synchronicity_t::asynchronous}},
        {"cbsrmv", {memory_debug_synchronicity_t::asynchronous}},
        {"cbsrmv_analysis", {memory_debug_synchronicity_t::asynchronous}},
        {"cbsrpad_value", {memory_debug_synchronicity_t::asynchronous}},
        {"cbsrsm_analysis", {memory_debug_synchronicity_t::synchronous}},
        {"cbsrsm_buffer_size", {memory_debug_synchronicity_t::asynchronous}},
        {"cbsrsm_solve", {memory_debug_synchronicity_t::asynchronous}},
        {"cbsrsv_analysis", {memory_debug_synchronicity_t::synchronous}},
        {"cbsrsv_buffer_size", {memory_debug_synchronicity_t::asynchronous}},
        {"cbsrsv_solve", {memory_debug_synchronicity_t::asynchronous}},
        {"cbsrxmv", {memory_debug_synchronicity_t::asynchronous}},
        {"ccheck_matrix_coo", {memory_debug_synchronicity_t::asynchronous}},
        {"ccheck_matrix_coo_buffer_size", {memory_debug_synchronicity_t::asynchronous}},
        {"ccheck_matrix_csc", {memory_debug_synchronicity_t::asynchronous}},
        {"ccheck_matrix_csc_buffer_size", {memory_debug_synchronicity_t::asynchronous}},
        {"ccheck_matrix_csr", {memory_debug_synchronicity_t::asynchronous}},
        {"ccheck_matrix_csr_buffer_size", {memory_debug_synchronicity_t::asynchronous}},
        {"ccheck_matrix_ell", {memory_debug_synchronicity_t::asynchronous}},
        {"ccheck_matrix_ell_buffer_size", {memory_debug_synchronicity_t::asynchronous}},
        {"ccheck_matrix_gebsc", {memory_debug_synchronicity_t::asynchronous}},
        {"ccheck_matrix_gebsc_buffer_size", {memory_debug_synchronicity_t::asynchronous}},
        {"ccheck_matrix_gebsr", {memory_debug_synchronicity_t::asynchronous}},
        {"ccheck_matrix_gebsr_buffer_size", {memory_debug_synchronicity_t::asynchronous}},
        {"ccoo2dense", {memory_debug_synchronicity_t::asynchronous}},
        {"ccoomv", {memory_debug_synchronicity_t::asynchronous}},
        {"ccsc2dense", {memory_debug_synchronicity_t::asynchronous}},
        {"ccsr2bsr", {memory_debug_synchronicity_t::synchronous}},
        {"ccsr2csc", {memory_debug_synchronicity_t::asynchronous}},
        {"ccsr2csr_compress", {memory_debug_synchronicity_t::synchronous}},
        {"ccsr2dense", {memory_debug_synchronicity_t::asynchronous}},
        {"ccsr2ell", {memory_debug_synchronicity_t::asynchronous}},
        {"ccsr2gebsr", {memory_debug_synchronicity_t::synchronous}},
        {"ccsr2gebsr_buffer_size", {memory_debug_synchronicity_t::asynchronous}},
        {"ccsr2hyb", {memory_debug_synchronicity_t::synchronous}},
        {"ccsrcolor", {memory_debug_synchronicity_t::synchronous}},
        {"ccsrgeam", {memory_debug_synchronicity_t::synchronous}},
        {"ccsrgemm", {memory_debug_synchronicity_t::asynchronous}},
        {"ccsrgemm_buffer_size", {memory_debug_synchronicity_t::asynchronous}},
        {"ccsrgemm_numeric", {memory_debug_synchronicity_t::asynchronous}},
        {"ccsric0", {memory_debug_synchronicity_t::asynchronous}},
        {"ccsric0_analysis", {memory_debug_synchronicity_t::synchronous}},
        {"ccsric0_buffer_size", {memory_debug_synchronicity_t::asynchronous}},
        {"ccsrilu0", {memory_debug_synchronicity_t::asynchronous}},
        {"ccsrilu0_analysis", {memory_debug_synchronicity_t::synchronous}},
        {"ccsrilu0_buffer_size", {memory_debug_synchronicity_t::asynchronous}},
        {"ccsrilu0_numeric_boost", {memory_debug_synchronicity_t::asynchronous}},
        {"ccsritilu0_compute", {memory_debug_synchronicity_t::synchronous}},
        {"ccsritilu0_compute_ex", {memory_debug_synchronicity_t::synchronous}},
        {"ccsritilu0_history", {memory_debug_synchronicity_t::synchronous}},
        {"ccsritsv_analysis", {memory_debug_synchronicity_t::synchronous}},
        {"ccsritsv_buffer_size", {memory_debug_synchronicity_t::asynchronous}},
        {"ccsritsv_solve", {memory_debug_synchronicity_t::asynchronous}},
        {"ccsritsv_solve_ex", {memory_debug_synchronicity_t::asynchronous}},
        {"ccsrmm", {memory_debug_synchronicity_t::asynchronous}},
        {"ccsrmv", {memory_debug_synchronicity_t::host_or_asynchronous}},
        {"ccsrmv_analysis", {memory_debug_synchronicity_t::host_or_synchronous}},
        {"ccsrsm_analysis", {memory_debug_synchronicity_t::synchronous}},
        {"ccsrsm_buffer_size", {memory_debug_synchronicity_t::asynchronous}},
        {"ccsrsm_solve", {memory_debug_synchronicity_t::asynchronous}},
        {"ccsrsv_analysis", {memory_debug_synchronicity_t::synchronous}},
        {"ccsrsv_buffer_size", {memory_debug_synchronicity_t::asynchronous}},
        {"ccsrsv_solve", {memory_debug_synchronicity_t::asynchronous}},
        {"cdense2coo", {memory_debug_synchronicity_t::synchronous}},
        {"cdense2csc", {memory_debug_synchronicity_t::synchronous}},
        {"cdense2csr", {memory_debug_synchronicity_t::synchronous}},
        {"cdotci", {memory_debug_synchronicity_t::asynchronous}},
        {"cdoti", {memory_debug_synchronicity_t::asynchronous}},
        {"cell2csr", {memory_debug_synchronicity_t::asynchronous}},
        {"cellmv", {memory_debug_synchronicity_t::asynchronous}},
        {"cgebsr2csr", {memory_debug_synchronicity_t::asynchronous}},
        {"cgebsr2gebsc", {memory_debug_synchronicity_t::asynchronous}},
        {"cgebsr2gebsc_buffer_size", {memory_debug_synchronicity_t::asynchronous}},
        {"cgebsr2gebsr", {memory_debug_synchronicity_t::synchronous}},
        {"cgebsr2gebsr_buffer_size", {memory_debug_synchronicity_t::asynchronous}},
        {"cgebsrmm", {memory_debug_synchronicity_t::asynchronous}},
        {"cgebsrmv", {memory_debug_synchronicity_t::asynchronous}},
        {"cgemmi", {memory_debug_synchronicity_t::asynchronous}},
        {"cgemvi", {memory_debug_synchronicity_t::asynchronous}},
        {"cgemvi_buffer_size", {memory_debug_synchronicity_t::asynchronous}},
        {"cgpsv_interleaved_batch", {memory_debug_synchronicity_t::asynchronous}},
        {"cgpsv_interleaved_batch_buffer_size", {memory_debug_synchronicity_t::asynchronous}},
        {"cgthr", {memory_debug_synchronicity_t::asynchronous}},
        {"cgthrz", {memory_debug_synchronicity_t::asynchronous}},
        {"cgtsv", {memory_debug_synchronicity_t::asynchronous}},
        {"cgtsv_buffer_size", {memory_debug_synchronicity_t::asynchronous}},
        {"cgtsv_interleaved_batch", {memory_debug_synchronicity_t::asynchronous}},
        {"cgtsv_interleaved_batch_buffer_size", {memory_debug_synchronicity_t::asynchronous}},
        {"cgtsv_no_pivot", {memory_debug_synchronicity_t::asynchronous}},
        {"cgtsv_no_pivot_buffer_size", {memory_debug_synchronicity_t::asynchronous}},
        {"cgtsv_no_pivot_strided_batch", {memory_debug_synchronicity_t::asynchronous}},
        {"cgtsv_no_pivot_strided_batch_buffer_size", {memory_debug_synchronicity_t::asynchronous}},
        {"check_matrix_hyb", {memory_debug_synchronicity_t::synchronous}},
        {"check_matrix_hyb_buffer_size", {memory_debug_synchronicity_t::host_or_asynchronous}},
        {"check_spmat", {memory_debug_synchronicity_t::depends}},
        {"chyb2csr", {memory_debug_synchronicity_t::asynchronous}},
        {"chybmv", {memory_debug_synchronicity_t::asynchronous}},
        {"cnnz", {memory_debug_synchronicity_t::asynchronous}},
        {"cnnz_compress", {memory_debug_synchronicity_t::synchronous}},
        {"coo2csr", {memory_debug_synchronicity_t::asynchronous}},
        {"coosort_buffer_size", {memory_debug_synchronicity_t::host}},
        {"coosort_by_column", {memory_debug_synchronicity_t::depends}},
        {"coosort_by_row", {memory_debug_synchronicity_t::depends}},
        {"create_identity_permutation", {memory_debug_synchronicity_t::host_or_asynchronous}},
        {"cscsort", {memory_debug_synchronicity_t::host_or_asynchronous}},
        {"cscsort_buffer_size", {memory_debug_synchronicity_t::host}},
        {"csctr", {memory_debug_synchronicity_t::asynchronous}},
        {"csr2bsr_nnz", {memory_debug_synchronicity_t::depends}},
        {"csr2coo", {memory_debug_synchronicity_t::host_or_asynchronous}},
        {"csr2csc_buffer_size", {memory_debug_synchronicity_t::host}},
        {"csr2ell_width", {memory_debug_synchronicity_t::asynchronous}},
        {"csr2gebsr_nnz", {memory_debug_synchronicity_t::depends}},
        {"csrgeam_nnz", {memory_debug_synchronicity_t::depends}},
        {"csrgemm_nnz", {memory_debug_synchronicity_t::depends}},
        {"csrgemm_symbolic", {memory_debug_synchronicity_t::depends}},
        {"csric0_clear", {memory_debug_synchronicity_t::host_or_synchronous}},
        {"csric0_get_tolerance", {memory_debug_synchronicity_t::host}},
        {"csric0_set_tolerance", {memory_debug_synchronicity_t::host}},
        {"csric0_singular_pivot", {memory_debug_synchronicity_t::synchronous}},
        {"csric0_zero_pivot", {memory_debug_synchronicity_t::synchronous}},
        {"csrilu0_clear", {memory_debug_synchronicity_t::host_or_synchronous}},
        {"csrilu0_get_tolerance", {memory_debug_synchronicity_t::host}},
        {"csrilu0_set_tolerance", {memory_debug_synchronicity_t::host}},
        {"csrilu0_singular_pivot", {memory_debug_synchronicity_t::synchronous}},
        {"csrilu0_zero_pivot", {memory_debug_synchronicity_t::synchronous}},
        {"csritilu0_buffer_size", {memory_debug_synchronicity_t::host_or_synchronous}},
        {"csritilu0_preprocess", {memory_debug_synchronicity_t::host_or_synchronous}},
        {"csritsv_clear", {memory_debug_synchronicity_t::host_or_synchronous}},
        {"csritsv_zero_pivot", {memory_debug_synchronicity_t::synchronous}},
        {"csrmv_clear", {memory_debug_synchronicity_t::host_or_synchronous}},
        {"csrsm_clear", {memory_debug_synchronicity_t::host_or_synchronous}},
        {"csrsm_zero_pivot", {memory_debug_synchronicity_t::synchronous}},
        {"csrsort", {memory_debug_synchronicity_t::host_or_asynchronous}},
        {"csrsort_buffer_size", {memory_debug_synchronicity_t::host}},
        {"csrsv_clear", {memory_debug_synchronicity_t::host_or_synchronous}},
        {"csrsv_zero_pivot", {memory_debug_synchronicity_t::synchronous}},
        {"daxpyi", {memory_debug_synchronicity_t::asynchronous}},
        {"dbsr2csr", {memory_debug_synchronicity_t::asynchronous}},
        {"dbsrgeam", {memory_debug_synchronicity_t::synchronous}},
        {"dbsrgemm", {memory_debug_synchronicity_t::synchronous}},
        {"dbsrgemm_buffer_size", {memory_debug_synchronicity_t::synchronous}},
        {"dbsric0", {memory_debug_synchronicity_t::asynchronous}},
        {"dbsric0_analysis", {memory_debug_synchronicity_t::synchronous}},
        {"dbsric0_buffer_size", {memory_debug_synchronicity_t::asynchronous}},
        {"dbsrilu0", {memory_debug_synchronicity_t::asynchronous}},
        {"dbsrilu0_analysis", {memory_debug_synchronicity_t::synchronous}},
        {"dbsrilu0_buffer_size", {memory_debug_synchronicity_t::asynchronous}},
        {"dbsrilu0_numeric_boost", {memory_debug_synchronicity_t::asynchronous}},
        {"dbsrmm", {memory_debug_synchronicity_t::asynchronous}},
        {"dbsrmv", {memory_debug_synchronicity_t::asynchronous}},
        {"dbsrmv_analysis", {memory_debug_synchronicity_t::asynchronous}},
        {"dbsrpad_value", {memory_debug_synchronicity_t::asynchronous}},
        {"dbsrsm_analysis", {memory_debug_synchronicity_t::synchronous}},
        {"dbsrsm_buffer_size", {memory_debug_synchronicity_t::asynchronous}},
        {"dbsrsm_solve", {memory_debug_synchronicity_t::asynchronous}},
        {"dbsrsv_analysis", {memory_debug_synchronicity_t::synchronous}},
        {"dbsrsv_buffer_size", {memory_debug_synchronicity_t::asynchronous}},
        {"dbsrsv_solve", {memory_debug_synchronicity_t::asynchronous}},
        {"dbsrxmv", {memory_debug_synchronicity_t::asynchronous}},
        {"dcbsrilu0_numeric_boost", {memory_debug_synchronicity_t::asynchronous}},
        {"dccsrilu0_numeric_boost", {memory_debug_synchronicity_t::asynchronous}},
        {"dcheck_matrix_coo", {memory_debug_synchronicity_t::asynchronous}},
        {"dcheck_matrix_coo_buffer_size", {memory_debug_synchronicity_t::asynchronous}},
        {"dcheck_matrix_csc", {memory_debug_synchronicity_t::asynchronous}},
        {"dcheck_matrix_csc_buffer_size", {memory_debug_synchronicity_t::asynchronous}},
        {"dcheck_matrix_csr", {memory_debug_synchronicity_t::asynchronous}},
        {"dcheck_matrix_csr_buffer_size", {memory_debug_synchronicity_t::asynchronous}},
        {"dcheck_matrix_ell", {memory_debug_synchronicity_t::asynchronous}},
        {"dcheck_matrix_ell_buffer_size", {memory_debug_synchronicity_t::asynchronous}},
        {"dcheck_matrix_gebsc", {memory_debug_synchronicity_t::asynchronous}},
        {"dcheck_matrix_gebsc_buffer_size", {memory_debug_synchronicity_t::asynchronous}},
        {"dcheck_matrix_gebsr", {memory_debug_synchronicity_t::asynchronous}},
        {"dcheck_matrix_gebsr_buffer_size", {memory_debug_synchronicity_t::asynchronous}},
        {"dcoo2dense", {memory_debug_synchronicity_t::asynchronous}},
        {"dcoomv", {memory_debug_synchronicity_t::asynchronous}},
        {"dcsc2dense", {memory_debug_synchronicity_t::asynchronous}},
        {"dcsr2bsr", {memory_debug_synchronicity_t::synchronous}},
        {"dcsr2csc", {memory_debug_synchronicity_t::asynchronous}},
        {"dcsr2csr_compress", {memory_debug_synchronicity_t::synchronous}},
        {"dcsr2dense", {memory_debug_synchronicity_t::asynchronous}},
        {"dcsr2ell", {memory_debug_synchronicity_t::asynchronous}},
        {"dcsr2gebsr", {memory_debug_synchronicity_t::synchronous}},
        {"dcsr2gebsr_buffer_size", {memory_debug_synchronicity_t::asynchronous}},
        {"dcsr2hyb", {memory_debug_synchronicity_t::synchronous}},
        {"dcsrcolor", {memory_debug_synchronicity_t::synchronous}},
        {"dcsrgeam", {memory_debug_synchronicity_t::synchronous}},
        {"dcsrgemm", {memory_debug_synchronicity_t::asynchronous}},
        {"dcsrgemm_buffer_size", {memory_debug_synchronicity_t::asynchronous}},
        {"dcsrgemm_numeric", {memory_debug_synchronicity_t::asynchronous}},
        {"dcsric0", {memory_debug_synchronicity_t::asynchronous}},
        {"dcsric0_analysis", {memory_debug_synchronicity_t::synchronous}},
        {"dcsric0_buffer_size", {memory_debug_synchronicity_t::asynchronous}},
        {"dcsrilu0", {memory_debug_synchronicity_t::asynchronous}},
        {"dcsrilu0_analysis", {memory_debug_synchronicity_t::synchronous}},
        {"dcsrilu0_buffer_size", {memory_debug_synchronicity_t::asynchronous}},
        {"dcsrilu0_numeric_boost", {memory_debug_synchronicity_t::asynchronous}},
        {"dcsritilu0_compute", {memory_debug_synchronicity_t::synchronous}},
        {"dcsritilu0_compute_ex", {memory_debug_synchronicity_t::synchronous}},
        {"dcsritilu0_history", {memory_debug_synchronicity_t::synchronous}},
        {"dcsritsv_analysis", {memory_debug_synchronicity_t::synchronous}},
        {"dcsritsv_buffer_size", {memory_debug_synchronicity_t::asynchronous}},
        {"dcsritsv_solve", {memory_debug_synchronicity_t::asynchronous}},
        {"dcsritsv_solve_ex", {memory_debug_synchronicity_t::asynchronous}},
        {"dcsrmm", {memory_debug_synchronicity_t::asynchronous}},
        {"dcsrmv", {memory_debug_synchronicity_t::host_or_asynchronous}},
        {"dcsrmv_analysis", {memory_debug_synchronicity_t::host_or_synchronous}},
        {"dcsrsm_analysis", {memory_debug_synchronicity_t::synchronous}},
        {"dcsrsm_buffer_size", {memory_debug_synchronicity_t::asynchronous}},
        {"dcsrsm_solve", {memory_debug_synchronicity_t::asynchronous}},
        {"dcsrsv_analysis", {memory_debug_synchronicity_t::synchronous}},
        {"dcsrsv_buffer_size", {memory_debug_synchronicity_t::asynchronous}},
        {"dcsrsv_solve", {memory_debug_synchronicity_t::asynchronous}},
        {"ddense2coo", {memory_debug_synchronicity_t::synchronous}},
        {"ddense2csc", {memory_debug_synchronicity_t::synchronous}},
        {"ddense2csr", {memory_debug_synchronicity_t::synchronous}},
        {"ddoti", {memory_debug_synchronicity_t::asynchronous}},
        {"dell2csr", {memory_debug_synchronicity_t::asynchronous}},
        {"dellmv", {memory_debug_synchronicity_t::asynchronous}},
        {"dense_to_sparse", {memory_debug_synchronicity_t::depends}},
        {"dgebsr2csr", {memory_debug_synchronicity_t::asynchronous}},
        {"dgebsr2gebsc", {memory_debug_synchronicity_t::asynchronous}},
        {"dgebsr2gebsc_buffer_size", {memory_debug_synchronicity_t::asynchronous}},
        {"dgebsr2gebsr", {memory_debug_synchronicity_t::synchronous}},
        {"dgebsr2gebsr_buffer_size", {memory_debug_synchronicity_t::asynchronous}},
        {"dgebsrmm", {memory_debug_synchronicity_t::asynchronous}},
        {"dgebsrmv", {memory_debug_synchronicity_t::asynchronous}},
        {"dgemmi", {memory_debug_synchronicity_t::asynchronous}},
        {"dgemvi", {memory_debug_synchronicity_t::asynchronous}},
        {"dgemvi_buffer_size", {memory_debug_synchronicity_t::asynchronous}},
        {"dgpsv_interleaved_batch", {memory_debug_synchronicity_t::asynchronous}},
        {"dgpsv_interleaved_batch_buffer_size", {memory_debug_synchronicity_t::asynchronous}},
        {"dgthr", {memory_debug_synchronicity_t::asynchronous}},
        {"dgthrz", {memory_debug_synchronicity_t::asynchronous}},
        {"dgtsv", {memory_debug_synchronicity_t::asynchronous}},
        {"dgtsv_buffer_size", {memory_debug_synchronicity_t::asynchronous}},
        {"dgtsv_interleaved_batch", {memory_debug_synchronicity_t::asynchronous}},
        {"dgtsv_interleaved_batch_buffer_size", {memory_debug_synchronicity_t::asynchronous}},
        {"dgtsv_no_pivot", {memory_debug_synchronicity_t::asynchronous}},
        {"dgtsv_no_pivot_buffer_size", {memory_debug_synchronicity_t::asynchronous}},
        {"dgtsv_no_pivot_strided_batch", {memory_debug_synchronicity_t::asynchronous}},
        {"dgtsv_no_pivot_strided_batch_buffer_size", {memory_debug_synchronicity_t::asynchronous}},
        {"dhyb2csr", {memory_debug_synchronicity_t::asynchronous}},
        {"dhybmv", {memory_debug_synchronicity_t::asynchronous}},
        {"dnnz", {memory_debug_synchronicity_t::asynchronous}},
        {"dnnz_compress", {memory_debug_synchronicity_t::synchronous}},
        {"dprune_csr2csr", {memory_debug_synchronicity_t::synchronous}},
        {"dprune_csr2csr_buffer_size", {memory_debug_synchronicity_t::asynchronous}},
        {"dprune_csr2csr_by_percentage", {memory_debug_synchronicity_t::synchronous}},
        {"dprune_csr2csr_by_percentage_buffer_size", {memory_debug_synchronicity_t::asynchronous}},
        {"dprune_csr2csr_nnz", {memory_debug_synchronicity_t::asynchronous}},
        {"dprune_csr2csr_nnz_by_percentage", {memory_debug_synchronicity_t::asynchronous}},
        {"dprune_dense2csr", {memory_debug_synchronicity_t::synchronous}},
        {"dprune_dense2csr_buffer_size", {memory_debug_synchronicity_t::asynchronous}},
        {"dprune_dense2csr_by_percentage", {memory_debug_synchronicity_t::synchronous}},
        {"dprune_dense2csr_by_percentage_buffer_size",
         {memory_debug_synchronicity_t::asynchronous}},
        {"dprune_dense2csr_nnz", {memory_debug_synchronicity_t::asynchronous}},
        {"dprune_dense2csr_nnz_by_percentage", {memory_debug_synchronicity_t::synchronous}},
        {"droti", {memory_debug_synchronicity_t::asynchronous}},
        {"dsbsrilu0_numeric_boost", {memory_debug_synchronicity_t::asynchronous}},
        {"dscsrilu0_numeric_boost", {memory_debug_synchronicity_t::asynchronous}},
        {"dsctr", {memory_debug_synchronicity_t::asynchronous}},
        {"ell2csr_nnz", {memory_debug_synchronicity_t::host_or_synchronous}},
        {"extract", {memory_debug_synchronicity_t::asynchronous}},
        {"extract_buffer_size", {memory_debug_synchronicity_t::host}},
        {"extract_nnz", {memory_debug_synchronicity_t::host}},
        {"gather", {memory_debug_synchronicity_t::host_or_asynchronous}},
        {"gebsr2gebsr_nnz", {memory_debug_synchronicity_t::depends}},
        {"get_git_rev", {memory_debug_synchronicity_t::host}},
        {"get_pointer_mode", {memory_debug_synchronicity_t::asynchronous}},
        {"get_stream", {memory_debug_synchronicity_t::host}},
        {"get_version", {memory_debug_synchronicity_t::host}},
        {"hyb2csr_buffer_size", {memory_debug_synchronicity_t::host}},
        {"inverse_permutation", {memory_debug_synchronicity_t::asynchronous}},
        {"isctr", {memory_debug_synchronicity_t::asynchronous}},
        {"rot", {memory_debug_synchronicity_t::host_or_asynchronous}},
        {"saxpyi", {memory_debug_synchronicity_t::asynchronous}},
        {"sbsr2csr", {memory_debug_synchronicity_t::asynchronous}},
        {"sbsrgeam", {memory_debug_synchronicity_t::synchronous}},
        {"sbsrgemm", {memory_debug_synchronicity_t::synchronous}},
        {"sbsrgemm_buffer_size", {memory_debug_synchronicity_t::synchronous}},
        {"sbsric0", {memory_debug_synchronicity_t::asynchronous}},
        {"sbsric0_analysis", {memory_debug_synchronicity_t::synchronous}},
        {"sbsric0_buffer_size", {memory_debug_synchronicity_t::asynchronous}},
        {"sbsrilu0", {memory_debug_synchronicity_t::asynchronous}},
        {"sbsrilu0_analysis", {memory_debug_synchronicity_t::synchronous}},
        {"sbsrilu0_buffer_size", {memory_debug_synchronicity_t::asynchronous}},
        {"sbsrilu0_numeric_boost", {memory_debug_synchronicity_t::asynchronous}},
        {"sbsrmm", {memory_debug_synchronicity_t::asynchronous}},
        {"sbsrmv", {memory_debug_synchronicity_t::asynchronous}},
        {"sbsrmv_analysis", {memory_debug_synchronicity_t::asynchronous}},
        {"sbsrpad_value", {memory_debug_synchronicity_t::asynchronous}},
        {"sbsrsm_analysis", {memory_debug_synchronicity_t::synchronous}},
        {"sbsrsm_buffer_size", {memory_debug_synchronicity_t::asynchronous}},
        {"sbsrsm_solve", {memory_debug_synchronicity_t::asynchronous}},
        {"sbsrsv_analysis", {memory_debug_synchronicity_t::synchronous}},
        {"sbsrsv_buffer_size", {memory_debug_synchronicity_t::asynchronous}},
        {"sbsrsv_solve", {memory_debug_synchronicity_t::asynchronous}},
        {"sbsrxmv", {memory_debug_synchronicity_t::asynchronous}},
        {"scatter", {memory_debug_synchronicity_t::host_or_asynchronous}},
        {"scheck_matrix_coo", {memory_debug_synchronicity_t::asynchronous}},
        {"scheck_matrix_coo_buffer_size", {memory_debug_synchronicity_t::asynchronous}},
        {"scheck_matrix_csc", {memory_debug_synchronicity_t::asynchronous}},
        {"scheck_matrix_csc_buffer_size", {memory_debug_synchronicity_t::asynchronous}},
        {"scheck_matrix_csr", {memory_debug_synchronicity_t::asynchronous}},
        {"scheck_matrix_csr_buffer_size", {memory_debug_synchronicity_t::asynchronous}},
        {"scheck_matrix_ell", {memory_debug_synchronicity_t::asynchronous}},
        {"scheck_matrix_ell_buffer_size", {memory_debug_synchronicity_t::asynchronous}},
        {"scheck_matrix_gebsc", {memory_debug_synchronicity_t::asynchronous}},
        {"scheck_matrix_gebsc_buffer_size", {memory_debug_synchronicity_t::asynchronous}},
        {"scheck_matrix_gebsr", {memory_debug_synchronicity_t::asynchronous}},
        {"scheck_matrix_gebsr_buffer_size", {memory_debug_synchronicity_t::asynchronous}},
        {"scoo2dense", {memory_debug_synchronicity_t::asynchronous}},
        {"scoomv", {memory_debug_synchronicity_t::asynchronous}},
        {"scsc2dense", {memory_debug_synchronicity_t::asynchronous}},
        {"scsr2bsr", {memory_debug_synchronicity_t::synchronous}},
        {"scsr2csc", {memory_debug_synchronicity_t::asynchronous}},
        {"scsr2csr_compress", {memory_debug_synchronicity_t::synchronous}},
        {"scsr2dense", {memory_debug_synchronicity_t::asynchronous}},
        {"scsr2ell", {memory_debug_synchronicity_t::asynchronous}},
        {"scsr2gebsr", {memory_debug_synchronicity_t::synchronous}},
        {"scsr2gebsr_buffer_size", {memory_debug_synchronicity_t::asynchronous}},
        {"scsr2hyb", {memory_debug_synchronicity_t::synchronous}},
        {"scsrcolor", {memory_debug_synchronicity_t::synchronous}},
        {"scsrgeam", {memory_debug_synchronicity_t::synchronous}},
        {"scsrgemm", {memory_debug_synchronicity_t::asynchronous}},
        {"scsrgemm_buffer_size", {memory_debug_synchronicity_t::asynchronous}},
        {"scsrgemm_numeric", {memory_debug_synchronicity_t::asynchronous}},
        {"scsric0", {memory_debug_synchronicity_t::asynchronous}},
        {"scsric0_analysis", {memory_debug_synchronicity_t::synchronous}},
        {"scsric0_buffer_size", {memory_debug_synchronicity_t::asynchronous}},
        {"scsrilu0", {memory_debug_synchronicity_t::asynchronous}},
        {"scsrilu0_analysis", {memory_debug_synchronicity_t::synchronous}},
        {"scsrilu0_buffer_size", {memory_debug_synchronicity_t::asynchronous}},
        {"scsrilu0_numeric_boost", {memory_debug_synchronicity_t::asynchronous}},
        {"scsritilu0_compute", {memory_debug_synchronicity_t::synchronous}},
        {"scsritilu0_compute_ex", {memory_debug_synchronicity_t::synchronous}},
        {"scsritilu0_history", {memory_debug_synchronicity_t::synchronous}},
        {"scsritsv_analysis", {memory_debug_synchronicity_t::synchronous}},
        {"scsritsv_buffer_size", {memory_debug_synchronicity_t::asynchronous}},
        {"scsritsv_solve", {memory_debug_synchronicity_t::asynchronous}},
        {"scsritsv_solve_ex", {memory_debug_synchronicity_t::asynchronous}},
        {"scsrmm", {memory_debug_synchronicity_t::asynchronous}},
        {"scsrmv", {memory_debug_synchronicity_t::host_or_asynchronous}},
        {"scsrmv_analysis", {memory_debug_synchronicity_t::host_or_synchronous}},
        {"scsrsm_analysis", {memory_debug_synchronicity_t::synchronous}},
        {"scsrsm_buffer_size", {memory_debug_synchronicity_t::asynchronous}},
        {"scsrsm_solve", {memory_debug_synchronicity_t::asynchronous}},
        {"scsrsv_analysis", {memory_debug_synchronicity_t::synchronous}},
        {"scsrsv_buffer_size", {memory_debug_synchronicity_t::asynchronous}},
        {"scsrsv_solve", {memory_debug_synchronicity_t::asynchronous}},
        {"sddmm", {memory_debug_synchronicity_t::asynchronous}},
        {"sddmm_buffer_size", {memory_debug_synchronicity_t::host}},
        {"sddmm_preprocess", {memory_debug_synchronicity_t::host_or_asynchronous}},
        {"sdense2coo", {memory_debug_synchronicity_t::synchronous}},
        {"sdense2csc", {memory_debug_synchronicity_t::synchronous}},
        {"sdense2csr", {memory_debug_synchronicity_t::synchronous}},
        {"sdoti", {memory_debug_synchronicity_t::asynchronous}},
        {"sell2csr", {memory_debug_synchronicity_t::asynchronous}},
        {"sellmv", {memory_debug_synchronicity_t::asynchronous}},
        {"set_identity_permutation", {memory_debug_synchronicity_t::asynchronous}},
        {"set_pointer_mode", {memory_debug_synchronicity_t::host}},
        {"set_stream", {memory_debug_synchronicity_t::host}},
        {"sgebsr2csr", {memory_debug_synchronicity_t::asynchronous}},
        {"sgebsr2gebsc", {memory_debug_synchronicity_t::asynchronous}},
        {"sgebsr2gebsc_buffer_size", {memory_debug_synchronicity_t::asynchronous}},
        {"sgebsr2gebsr", {memory_debug_synchronicity_t::synchronous}},
        {"sgebsr2gebsr_buffer_size", {memory_debug_synchronicity_t::asynchronous}},
        {"sgebsrmm", {memory_debug_synchronicity_t::asynchronous}},
        {"sgebsrmv", {memory_debug_synchronicity_t::asynchronous}},
        {"sgemmi", {memory_debug_synchronicity_t::asynchronous}},
        {"sgemvi", {memory_debug_synchronicity_t::asynchronous}},
        {"sgemvi_buffer_size", {memory_debug_synchronicity_t::asynchronous}},
        {"sgpsv_interleaved_batch", {memory_debug_synchronicity_t::asynchronous}},
        {"sgpsv_interleaved_batch_buffer_size", {memory_debug_synchronicity_t::asynchronous}},
        {"sgthr", {memory_debug_synchronicity_t::asynchronous}},
        {"sgthrz", {memory_debug_synchronicity_t::asynchronous}},
        {"sgtsv", {memory_debug_synchronicity_t::asynchronous}},
        {"sgtsv_buffer_size", {memory_debug_synchronicity_t::asynchronous}},
        {"sgtsv_interleaved_batch", {memory_debug_synchronicity_t::asynchronous}},
        {"sgtsv_interleaved_batch_buffer_size", {memory_debug_synchronicity_t::asynchronous}},
        {"sgtsv_no_pivot", {memory_debug_synchronicity_t::asynchronous}},
        {"sgtsv_no_pivot_buffer_size", {memory_debug_synchronicity_t::asynchronous}},
        {"sgtsv_no_pivot_strided_batch", {memory_debug_synchronicity_t::asynchronous}},
        {"sgtsv_no_pivot_strided_batch_buffer_size", {memory_debug_synchronicity_t::asynchronous}},
        {"shyb2csr", {memory_debug_synchronicity_t::asynchronous}},
        {"shybmv", {memory_debug_synchronicity_t::asynchronous}},
        {"snnz", {memory_debug_synchronicity_t::asynchronous}},
        {"snnz_compress", {memory_debug_synchronicity_t::synchronous}},
        {"sparse_to_dense", {memory_debug_synchronicity_t::host_or_asynchronous}},
        {"sparse_to_sparse", {memory_debug_synchronicity_t::depends}},
        {"sparse_to_sparse_buffer_size", {memory_debug_synchronicity_t::depends}},
        {"spgeam", {memory_debug_synchronicity_t::depends}},
        {"spgeam_buffer_size", {memory_debug_synchronicity_t::host}},
        {"spgeam_get_output", {memory_debug_synchronicity_t::host_or_asynchronous}},
        {"spgeam_set_input", {memory_debug_synchronicity_t::host}},
        {"spgemm", {memory_debug_synchronicity_t::depends}},
        {"spic0", {memory_debug_synchronicity_t::depends}},
        {"spic0_buffer_size", {memory_debug_synchronicity_t::host}},
        {"spic0_descr_create", {memory_debug_synchronicity_t::host}},
        {"spic0_descr_destroy", {memory_debug_synchronicity_t::host_or_synchronous}},
        {"spic0_get_output", {memory_debug_synchronicity_t::asynchronous}},
        {"spic0_set_input", {memory_debug_synchronicity_t::host}},
        {"spilu0", {memory_debug_synchronicity_t::depends}},
        {"spilu0_buffer_size", {memory_debug_synchronicity_t::host}},
        {"spilu0_descr_create", {memory_debug_synchronicity_t::host}},
        {"spilu0_descr_destroy", {memory_debug_synchronicity_t::host_or_synchronous}},
        {"spilu0_get_output", {memory_debug_synchronicity_t::asynchronous}},
        {"spilu0_set_input", {memory_debug_synchronicity_t::host}},
        {"spitsv", {memory_debug_synchronicity_t::depends}},
        {"spmm", {memory_debug_synchronicity_t::host_or_asynchronous}},
        {"spmv", {memory_debug_synchronicity_t::depends}},
        {"spmv_clear_extra", {memory_debug_synchronicity_t::synchronous}},
        {"spmv_set_extra", {memory_debug_synchronicity_t::partially_synchronous}},
        {"spmv_set_input", {memory_debug_synchronicity_t::host}},
        {"sprune_csr2csr", {memory_debug_synchronicity_t::synchronous}},
        {"sprune_csr2csr_buffer_size", {memory_debug_synchronicity_t::asynchronous}},
        {"sprune_csr2csr_by_percentage", {memory_debug_synchronicity_t::synchronous}},
        {"sprune_csr2csr_by_percentage_buffer_size", {memory_debug_synchronicity_t::asynchronous}},
        {"sprune_csr2csr_nnz", {memory_debug_synchronicity_t::asynchronous}},
        {"sprune_csr2csr_nnz_by_percentage", {memory_debug_synchronicity_t::asynchronous}},
        {"sprune_dense2csr", {memory_debug_synchronicity_t::synchronous}},
        {"sprune_dense2csr_buffer_size", {memory_debug_synchronicity_t::asynchronous}},
        {"sprune_dense2csr_by_percentage", {memory_debug_synchronicity_t::synchronous}},
        {"sprune_dense2csr_by_percentage_buffer_size",
         {memory_debug_synchronicity_t::asynchronous}},
        {"sprune_dense2csr_nnz", {memory_debug_synchronicity_t::asynchronous}},
        {"sprune_dense2csr_nnz_by_percentage", {memory_debug_synchronicity_t::synchronous}},
        {"spsm", {memory_debug_synchronicity_t::depends}},
        {"spsv", {memory_debug_synchronicity_t::depends}},
        {"sptrsm", {memory_debug_synchronicity_t::depends}},
        {"sptrsm_buffer_size", {memory_debug_synchronicity_t::host}},
        {"sptrsm_get_output", {memory_debug_synchronicity_t::asynchronous}},
        {"sptrsm_set_input", {memory_debug_synchronicity_t::host}},
        {"sptrsv", {memory_debug_synchronicity_t::depends}},
        {"sptrsv_buffer_size", {memory_debug_synchronicity_t::host}},
        {"sptrsv_descr_create", {memory_debug_synchronicity_t::host}},
        {"sptrsv_descr_destroy", {memory_debug_synchronicity_t::host_or_synchronous}},
        {"sptrsv_get_output", {memory_debug_synchronicity_t::depends}},
        {"sptrsv_set_input", {memory_debug_synchronicity_t::host}},
        {"spvv", {memory_debug_synchronicity_t::host_or_asynchronous}},
        {"sroti", {memory_debug_synchronicity_t::asynchronous}},
        {"ssctr", {memory_debug_synchronicity_t::asynchronous}},
        {"v2_spmv", {memory_debug_synchronicity_t::depends}},
        {"v2_spmv_buffer_size", {memory_debug_synchronicity_t::host}},
        {"zaxpyi", {memory_debug_synchronicity_t::asynchronous}},
        {"zbsr2csr", {memory_debug_synchronicity_t::asynchronous}},
        {"zbsrgeam", {memory_debug_synchronicity_t::synchronous}},
        {"zbsrgemm", {memory_debug_synchronicity_t::synchronous}},
        {"zbsrgemm_buffer_size", {memory_debug_synchronicity_t::synchronous}},
        {"zbsric0", {memory_debug_synchronicity_t::asynchronous}},
        {"zbsric0_analysis", {memory_debug_synchronicity_t::synchronous}},
        {"zbsric0_buffer_size", {memory_debug_synchronicity_t::asynchronous}},
        {"zbsrilu0", {memory_debug_synchronicity_t::asynchronous}},
        {"zbsrilu0_analysis", {memory_debug_synchronicity_t::synchronous}},
        {"zbsrilu0_buffer_size", {memory_debug_synchronicity_t::asynchronous}},
        {"zbsrilu0_numeric_boost", {memory_debug_synchronicity_t::asynchronous}},
        {"zbsrmm", {memory_debug_synchronicity_t::asynchronous}},
        {"zbsrmv", {memory_debug_synchronicity_t::asynchronous}},
        {"zbsrmv_analysis", {memory_debug_synchronicity_t::asynchronous}},
        {"zbsrpad_value", {memory_debug_synchronicity_t::asynchronous}},
        {"zbsrsm_analysis", {memory_debug_synchronicity_t::synchronous}},
        {"zbsrsm_buffer_size", {memory_debug_synchronicity_t::asynchronous}},
        {"zbsrsm_solve", {memory_debug_synchronicity_t::asynchronous}},
        {"zbsrsv_analysis", {memory_debug_synchronicity_t::synchronous}},
        {"zbsrsv_buffer_size", {memory_debug_synchronicity_t::asynchronous}},
        {"zbsrsv_solve", {memory_debug_synchronicity_t::asynchronous}},
        {"zbsrxmv", {memory_debug_synchronicity_t::asynchronous}},
        {"zcheck_matrix_coo", {memory_debug_synchronicity_t::asynchronous}},
        {"zcheck_matrix_coo_buffer_size", {memory_debug_synchronicity_t::asynchronous}},
        {"zcheck_matrix_csc", {memory_debug_synchronicity_t::asynchronous}},
        {"zcheck_matrix_csc_buffer_size", {memory_debug_synchronicity_t::asynchronous}},
        {"zcheck_matrix_csr", {memory_debug_synchronicity_t::asynchronous}},
        {"zcheck_matrix_csr_buffer_size", {memory_debug_synchronicity_t::asynchronous}},
        {"zcheck_matrix_ell", {memory_debug_synchronicity_t::asynchronous}},
        {"zcheck_matrix_ell_buffer_size", {memory_debug_synchronicity_t::asynchronous}},
        {"zcheck_matrix_gebsc", {memory_debug_synchronicity_t::asynchronous}},
        {"zcheck_matrix_gebsc_buffer_size", {memory_debug_synchronicity_t::asynchronous}},
        {"zcheck_matrix_gebsr", {memory_debug_synchronicity_t::asynchronous}},
        {"zcheck_matrix_gebsr_buffer_size", {memory_debug_synchronicity_t::asynchronous}},
        {"zcoo2dense", {memory_debug_synchronicity_t::asynchronous}},
        {"zcoomv", {memory_debug_synchronicity_t::asynchronous}},
        {"zcsc2dense", {memory_debug_synchronicity_t::asynchronous}},
        {"zcsr2bsr", {memory_debug_synchronicity_t::synchronous}},
        {"zcsr2csc", {memory_debug_synchronicity_t::asynchronous}},
        {"zcsr2csr_compress", {memory_debug_synchronicity_t::synchronous}},
        {"zcsr2dense", {memory_debug_synchronicity_t::asynchronous}},
        {"zcsr2ell", {memory_debug_synchronicity_t::asynchronous}},
        {"zcsr2gebsr", {memory_debug_synchronicity_t::synchronous}},
        {"zcsr2gebsr_buffer_size", {memory_debug_synchronicity_t::asynchronous}},
        {"zcsr2hyb", {memory_debug_synchronicity_t::synchronous}},
        {"zcsrcolor", {memory_debug_synchronicity_t::synchronous}},
        {"zcsrgeam", {memory_debug_synchronicity_t::synchronous}},
        {"zcsrgemm", {memory_debug_synchronicity_t::asynchronous}},
        {"zcsrgemm_buffer_size", {memory_debug_synchronicity_t::asynchronous}},
        {"zcsrgemm_numeric", {memory_debug_synchronicity_t::asynchronous}},
        {"zcsric0", {memory_debug_synchronicity_t::asynchronous}},
        {"zcsric0_analysis", {memory_debug_synchronicity_t::synchronous}},
        {"zcsric0_buffer_size", {memory_debug_synchronicity_t::asynchronous}},
        {"zcsrilu0", {memory_debug_synchronicity_t::asynchronous}},
        {"zcsrilu0_analysis", {memory_debug_synchronicity_t::synchronous}},
        {"zcsrilu0_buffer_size", {memory_debug_synchronicity_t::asynchronous}},
        {"zcsrilu0_numeric_boost", {memory_debug_synchronicity_t::asynchronous}},
        {"zcsritilu0_compute", {memory_debug_synchronicity_t::synchronous}},
        {"zcsritilu0_compute_ex", {memory_debug_synchronicity_t::synchronous}},
        {"zcsritilu0_history", {memory_debug_synchronicity_t::synchronous}},
        {"zcsritsv_analysis", {memory_debug_synchronicity_t::synchronous}},
        {"zcsritsv_buffer_size", {memory_debug_synchronicity_t::asynchronous}},
        {"zcsritsv_solve", {memory_debug_synchronicity_t::asynchronous}},
        {"zcsritsv_solve_ex", {memory_debug_synchronicity_t::asynchronous}},
        {"zcsrmm", {memory_debug_synchronicity_t::asynchronous}},
        {"zcsrmv", {memory_debug_synchronicity_t::host_or_asynchronous}},
        {"zcsrmv_analysis", {memory_debug_synchronicity_t::host_or_synchronous}},
        {"zcsrsm_analysis", {memory_debug_synchronicity_t::synchronous}},
        {"zcsrsm_buffer_size", {memory_debug_synchronicity_t::asynchronous}},
        {"zcsrsm_solve", {memory_debug_synchronicity_t::asynchronous}},
        {"zcsrsv_analysis", {memory_debug_synchronicity_t::synchronous}},
        {"zcsrsv_buffer_size", {memory_debug_synchronicity_t::asynchronous}},
        {"zcsrsv_solve", {memory_debug_synchronicity_t::asynchronous}},
        {"zdense2coo", {memory_debug_synchronicity_t::synchronous}},
        {"zdense2csc", {memory_debug_synchronicity_t::synchronous}},
        {"zdense2csr", {memory_debug_synchronicity_t::synchronous}},
        {"zdotci", {memory_debug_synchronicity_t::asynchronous}},
        {"zdoti", {memory_debug_synchronicity_t::asynchronous}},
        {"zell2csr", {memory_debug_synchronicity_t::asynchronous}},
        {"zellmv", {memory_debug_synchronicity_t::asynchronous}},
        {"zgebsr2csr", {memory_debug_synchronicity_t::asynchronous}},
        {"zgebsr2gebsc", {memory_debug_synchronicity_t::asynchronous}},
        {"zgebsr2gebsc_buffer_size", {memory_debug_synchronicity_t::asynchronous}},
        {"zgebsr2gebsr", {memory_debug_synchronicity_t::synchronous}},
        {"zgebsr2gebsr_buffer_size", {memory_debug_synchronicity_t::asynchronous}},
        {"zgebsrmm", {memory_debug_synchronicity_t::asynchronous}},
        {"zgebsrmv", {memory_debug_synchronicity_t::asynchronous}},
        {"zgemmi", {memory_debug_synchronicity_t::asynchronous}},
        {"zgemvi", {memory_debug_synchronicity_t::asynchronous}},
        {"zgemvi_buffer_size", {memory_debug_synchronicity_t::asynchronous}},
        {"zgpsv_interleaved_batch", {memory_debug_synchronicity_t::asynchronous}},
        {"zgpsv_interleaved_batch_buffer_size", {memory_debug_synchronicity_t::asynchronous}},
        {"zgthr", {memory_debug_synchronicity_t::asynchronous}},
        {"zgthrz", {memory_debug_synchronicity_t::asynchronous}},
        {"zgtsv", {memory_debug_synchronicity_t::asynchronous}},
        {"zgtsv_buffer_size", {memory_debug_synchronicity_t::asynchronous}},
        {"zgtsv_interleaved_batch", {memory_debug_synchronicity_t::asynchronous}},
        {"zgtsv_interleaved_batch_buffer_size", {memory_debug_synchronicity_t::asynchronous}},
        {"zgtsv_no_pivot", {memory_debug_synchronicity_t::asynchronous}},
        {"zgtsv_no_pivot_buffer_size", {memory_debug_synchronicity_t::asynchronous}},
        {"zgtsv_no_pivot_strided_batch", {memory_debug_synchronicity_t::asynchronous}},
        {"zgtsv_no_pivot_strided_batch_buffer_size", {memory_debug_synchronicity_t::asynchronous}},
        {"zhyb2csr", {memory_debug_synchronicity_t::asynchronous}},
        {"zhybmv", {memory_debug_synchronicity_t::asynchronous}},
        {"znnz", {memory_debug_synchronicity_t::asynchronous}},
        {"znnz_compress", {memory_debug_synchronicity_t::synchronous}},
        {"zsctr", {memory_debug_synchronicity_t::asynchronous}},
        {"bell_get", {memory_debug_synchronicity_t::host}},
        {"bsr_get", {memory_debug_synchronicity_t::host}},
        {"bsr_set_pointers", {memory_debug_synchronicity_t::host}},
        {"const_bell_get", {memory_debug_synchronicity_t::host}},
        {"const_bsr_get", {memory_debug_synchronicity_t::host}},
        {"const_coo_aos_get", {memory_debug_synchronicity_t::host}},
        {"const_coo_get", {memory_debug_synchronicity_t::host}},
        {"const_csc_get", {memory_debug_synchronicity_t::host}},
        {"const_csr_get", {memory_debug_synchronicity_t::host}},
        {"const_dnmat_get", {memory_debug_synchronicity_t::host}},
        {"const_dnmat_get_values", {memory_debug_synchronicity_t::host}},
        {"const_dnvec_get", {memory_debug_synchronicity_t::host}},
        {"const_dnvec_get_values", {memory_debug_synchronicity_t::host}},
        {"const_ell_get", {memory_debug_synchronicity_t::host}},
        {"const_sell_get", {memory_debug_synchronicity_t::host}},
        {"const_spmat_get_values", {memory_debug_synchronicity_t::host}},
        {"const_spvec_get", {memory_debug_synchronicity_t::host}},
        {"const_spvec_get_values", {memory_debug_synchronicity_t::host}},
        {"coo_aos_get", {memory_debug_synchronicity_t::host}},
        {"coo_aos_set_pointers", {memory_debug_synchronicity_t::host}},
        {"coo_get", {memory_debug_synchronicity_t::host}},
        {"coo_set_pointers", {memory_debug_synchronicity_t::host}},
        {"coo_set_strided_batch", {memory_debug_synchronicity_t::host}},
        {"copy_color_info", {memory_debug_synchronicity_t::host}},
        {"copy_hyb_mat", {memory_debug_synchronicity_t::host_or_synchronous}},
        {"copy_mat_descr", {memory_debug_synchronicity_t::host}},
        {"copy_mat_info", {memory_debug_synchronicity_t::host_or_synchronous}},
        {"create_bell_descr", {memory_debug_synchronicity_t::host}},
        {"create_bsr_descr", {memory_debug_synchronicity_t::host}},
        {"create_color_info", {memory_debug_synchronicity_t::host}},
        {"create_const_bell_descr", {memory_debug_synchronicity_t::host}},
        {"create_const_coo_descr", {memory_debug_synchronicity_t::host}},
        {"create_const_csc_descr", {memory_debug_synchronicity_t::host}},
        {"create_const_csr_descr", {memory_debug_synchronicity_t::host}},
        {"create_const_dnmat_descr", {memory_debug_synchronicity_t::host}},
        {"create_const_dnvec_descr", {memory_debug_synchronicity_t::host}},
        {"create_const_sell_descr", {memory_debug_synchronicity_t::host}},
        {"create_const_spvec_descr", {memory_debug_synchronicity_t::host}},
        {"create_coo_aos_descr", {memory_debug_synchronicity_t::host}},
        {"create_coo_descr", {memory_debug_synchronicity_t::host}},
        {"create_csc_descr", {memory_debug_synchronicity_t::host}},
        {"create_csr_descr", {memory_debug_synchronicity_t::host}},
        {"create_dnmat_descr", {memory_debug_synchronicity_t::host}},
        {"create_dnvec_descr", {memory_debug_synchronicity_t::host}},
        {"create_ell_descr", {memory_debug_synchronicity_t::host}},
        {"create_extract_descr", {memory_debug_synchronicity_t::synchronous}},
        {"create_handle", {memory_debug_synchronicity_t::synchronous}},
        {"create_hyb_mat", {memory_debug_synchronicity_t::host}},
        {"create_mat_descr", {memory_debug_synchronicity_t::host}},
        {"create_mat_info", {memory_debug_synchronicity_t::host}},
        {"create_sell_descr", {memory_debug_synchronicity_t::host}},
        {"create_sparse_to_sparse_descr", {memory_debug_synchronicity_t::host}},
        {"create_spgeam_descr", {memory_debug_synchronicity_t::host}},
        {"create_spmv_descr", {memory_debug_synchronicity_t::host}},
        {"create_sptrsm_descr", {memory_debug_synchronicity_t::host}},
        {"create_sptrsv_descr", {memory_debug_synchronicity_t::host}},
        {"create_spvec_descr", {memory_debug_synchronicity_t::host}},
        {"csc_get", {memory_debug_synchronicity_t::host}},
        {"csc_set_pointers", {memory_debug_synchronicity_t::host}},
        {"csc_set_strided_batch", {memory_debug_synchronicity_t::host}},
        {"csr_get", {memory_debug_synchronicity_t::host}},
        {"csr_set_pointers", {memory_debug_synchronicity_t::host}},
        {"csr_set_strided_batch", {memory_debug_synchronicity_t::host}},
        {"destroy_color_info", {memory_debug_synchronicity_t::host}},
        {"destroy_dnmat_descr", {memory_debug_synchronicity_t::host}},
        {"destroy_dnvec_descr", {memory_debug_synchronicity_t::host}},
        {"destroy_error", {memory_debug_synchronicity_t::host}},
        {"destroy_extract_descr", {memory_debug_synchronicity_t::synchronous}},
        {"destroy_handle", {memory_debug_synchronicity_t::synchronous}},
        {"destroy_hyb_mat", {memory_debug_synchronicity_t::synchronous}},
        {"destroy_mat_descr", {memory_debug_synchronicity_t::host}},
        {"destroy_mat_info", {memory_debug_synchronicity_t::synchronous}},
        {"destroy_sparse_to_sparse_descr", {memory_debug_synchronicity_t::host_or_synchronous}},
        {"destroy_spgeam_descr", {memory_debug_synchronicity_t::host_or_synchronous}},
        {"destroy_spmat_descr", {memory_debug_synchronicity_t::depends}},
        {"destroy_spmv_descr", {memory_debug_synchronicity_t::host_or_synchronous}},
        {"destroy_sptrsm_descr", {memory_debug_synchronicity_t::host_or_synchronous}},
        {"destroy_sptrsv_descr", {memory_debug_synchronicity_t::host_or_synchronous}},
        {"destroy_spvec_descr", {memory_debug_synchronicity_t::host}},
        {"dnmat_get", {memory_debug_synchronicity_t::host}},
        {"dnmat_get_strided_batch", {memory_debug_synchronicity_t::host}},
        {"dnmat_get_values", {memory_debug_synchronicity_t::host}},
        {"dnmat_set_strided_batch", {memory_debug_synchronicity_t::host}},
        {"dnmat_set_values", {memory_debug_synchronicity_t::host}},
        {"dnvec_get", {memory_debug_synchronicity_t::host}},
        {"dnvec_get_strided_batch", {memory_debug_synchronicity_t::host}},
        {"dnvec_get_values", {memory_debug_synchronicity_t::host}},
        {"dnvec_set_strided_batch", {memory_debug_synchronicity_t::host}},
        {"dnvec_set_values", {memory_debug_synchronicity_t::host}},
        {"ell_get", {memory_debug_synchronicity_t::host}},
        {"ell_set_pointers", {memory_debug_synchronicity_t::host}},
        {"sell_get", {memory_debug_synchronicity_t::host}},
        {"set_mat_diag_type", {memory_debug_synchronicity_t::host}},
        {"set_mat_fill_mode", {memory_debug_synchronicity_t::host}},
        {"set_mat_index_base", {memory_debug_synchronicity_t::host}},
        {"set_mat_storage_mode", {memory_debug_synchronicity_t::host}},
        {"set_mat_type", {memory_debug_synchronicity_t::host}},
        {"sparse_to_sparse_permissive", {memory_debug_synchronicity_t::host}},
        {"spmat_get_attribute", {memory_debug_synchronicity_t::host}},
        {"spmat_get_format", {memory_debug_synchronicity_t::host}},
        {"spmat_get_index_base", {memory_debug_synchronicity_t::host}},
        {"spmat_get_nnz", {memory_debug_synchronicity_t::host}},
        {"spmat_get_size", {memory_debug_synchronicity_t::host}},
        {"spmat_get_strided_batch", {memory_debug_synchronicity_t::host}},
        {"spmat_get_values", {memory_debug_synchronicity_t::host}},
        {"spmat_set_attribute", {memory_debug_synchronicity_t::host}},
        {"spmat_set_nnz", {memory_debug_synchronicity_t::host}},
        {"spmat_set_strided_batch", {memory_debug_synchronicity_t::host}},
        {"spmat_set_values", {memory_debug_synchronicity_t::host}},
        {"spvec_get", {memory_debug_synchronicity_t::host}},
        {"spvec_get_index_base", {memory_debug_synchronicity_t::host}},
        {"spvec_get_values", {memory_debug_synchronicity_t::host}},
        {"spvec_set_values", {memory_debug_synchronicity_t::host}}};

    void memory_debug_check_synchronicity(rocsparse_handle handle, const char* name)
    {
        if(memory_debug_t::instance().enabled())
        {
            auto&      info = memory_debug_t::instance().get_memory_debug_synchronicity_info(name);
            const auto sync = info.get_sync();

            rocsparse_memory_debug_synchronicity synchronicity;
            const rocsparse_memory_debug_info    memory_debug_info
                = rocsparse_memory_debug_info_synchronicity;
            rocsparse_status status = rocsparse_memory_debug_info_get(
                handle, memory_debug_info, &synchronicity, sizeof(synchronicity));
            if(status != rocsparse_status_success)
            {
                throw std::runtime_error(std::string("rocsparse_memory_debug_info_get failed"));
            }
            switch(synchronicity)
            {
            case rocsparse_memory_debug_synchronicity_async:
            {
                info.add_call(memory_debug_synchronicity_t::asynchronous);
                if((sync != memory_debug_synchronicity_t::asynchronous)
                   && (sync != memory_debug_synchronicity_t::host_or_asynchronous)
                   && (sync != memory_debug_synchronicity_t::depends))
                {
                    std::cerr << "Error: rocsparse_" << name << " is declared '"
                              << memory_debug_synchronicity_t2string(sync)
                              << "' but production code returns 'asynchronous'" << std::endl;
                    throw std::runtime_error(std::string("rocsparse_") + name
                                             + ": sync property mismatch (asynchronous)");
                }
                return;
            }
            case rocsparse_memory_debug_synchronicity_sync:
            {
                info.add_call(memory_debug_synchronicity_t::synchronous);
                if((sync != memory_debug_synchronicity_t::synchronous)
                   && (sync != memory_debug_synchronicity_t::host_or_synchronous)
                   && (sync != memory_debug_synchronicity_t::depends))
                {
                    std::cerr << "Error: rocsparse_" << name << " is declared '"
                              << memory_debug_synchronicity_t2string(sync)
                              << "' but production code returns 'synchronous'" << std::endl;
                    throw std::runtime_error(std::string("rocsparse_") + name
                                             + ": sync property mismatch (synchronous)");
                }
                return;
            }

            case rocsparse_memory_debug_synchronicity_psync:
            {
                info.add_call(memory_debug_synchronicity_t::partially_synchronous);
                if((sync != memory_debug_synchronicity_t::partially_synchronous)
                   && (sync != memory_debug_synchronicity_t::host_or_partially_synchronous)
                   && (sync != memory_debug_synchronicity_t::depends))
                {
                    std::cerr << "Error: rocsparse_" << name << " is declared '"
                              << memory_debug_synchronicity_t2string(sync)
                              << "' but production code returns 'partially_synchronous'"
                              << std::endl;
                    throw std::runtime_error(std::string("rocsparse_") + name
                                             + ": sync property mismatch (partially_synchronous)");
                }
                return;
            }
            case rocsparse_memory_debug_synchronicity_host:
            {

                info.add_call(memory_debug_synchronicity_t::host);
                if((sync != memory_debug_synchronicity_t::host)
                   && (sync != memory_debug_synchronicity_t::host_or_asynchronous)
                   && (sync != memory_debug_synchronicity_t::host_or_synchronous)
                   && (sync != memory_debug_synchronicity_t::depends))
                {
                    std::cerr << "Error: rocsparse_" << name << " is declared '"
                              << memory_debug_synchronicity_t2string(sync)
                              << "' but production code returns 'host'" << std::endl;
                    throw std::runtime_error(std::string("rocsparse_") + name
                                             + ": sync property mismatch (host)");
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
        return s_map[name];
    }

    void memory_debug_t::report(rocsparse_handle handle) const
    {
        std::ofstream out(this->get_filename());
        out << "[" << std::endl;
        int64_t count = 0;
        for(const auto& p : s_map)
        {
            const auto& info = p.second;
            if(info.get_ncalls() == 0)
            {
                continue;
            }
            const std::string& name = p.first;
            const auto         sync = info.get_sync();
            const uint64_t     ncalls_synchronous
                = info.get_calls(memory_debug_synchronicity_t::synchronous);
            const uint64_t ncalls_asynchronous
                = info.get_calls(memory_debug_synchronicity_t::asynchronous);
            const uint64_t ncalls_partially_synchronous
                = info.get_calls(memory_debug_synchronicity_t::partially_synchronous);
            const uint64_t ncalls_host = info.get_calls(memory_debug_synchronicity_t::host);

            if(count > 0)
            {
                out << ", " << std::endl;
            }

            out << "{\"name\": \"rocsparse_" << name << "\"," << std::endl;
            out << " \"sync\": \"" << memory_debug_synchronicity_t2string(sync) << "\","
                << std::endl;
            out << " \"calls\": {\"sync\": " << ncalls_synchronous << "," << std::endl;
            out << "            \"async\": " << ncalls_asynchronous << "," << std::endl;
            out << "            \"partialsync\": " << ncalls_partially_synchronous << ","
                << std::endl;
            out << "            \"host\": " << ncalls_host << "}}";
            ++count;
        }

        out << std::endl << "]" << std::endl;
    }

    const std::string& memory_debug_t::get_filename() const
    {
        return this->m_filename;
    }

    rocsparse_status memory_debug_t::check(rocsparse_handle handle) const
    {
        bool failed = false;
        for(const auto& p : s_map)
        {
            const auto& info = p.second;
            if(info.get_ncalls() == 0)
            {
                continue;
            }
            const std::string& name = p.first;
            const auto         sync = info.get_sync();
            const uint64_t     ncalls_synchronous
                = info.get_calls(memory_debug_synchronicity_t::synchronous);
            const uint64_t ncalls_asynchronous
                = info.get_calls(memory_debug_synchronicity_t::asynchronous);
            const uint64_t ncalls_partially_synchronous
                = info.get_calls(memory_debug_synchronicity_t::partially_synchronous);
            const uint64_t ncalls_host = info.get_calls(memory_debug_synchronicity_t::host);

            switch(sync)
            {
            case memory_debug_synchronicity_t::synchronous:
            {
                if((ncalls_asynchronous > 0) || //
                   (ncalls_partially_synchronous > 0) || //
                   (ncalls_host > 0))
                {
                    break;
                }
                continue;
            }

            case memory_debug_synchronicity_t::host_or_synchronous:
            {
                if((ncalls_asynchronous > 0) || //
                   (ncalls_partially_synchronous > 0))
                {
                    break;
                }
                continue;
            }

            case memory_debug_synchronicity_t::asynchronous:
            {
                if((ncalls_synchronous > 0) || //
                   (ncalls_partially_synchronous > 0) || //
                   (ncalls_host > 0))
                {
                    break;
                }
                continue;
            }

            case memory_debug_synchronicity_t::host_or_asynchronous:
            {
                if((ncalls_synchronous > 0) || //
                   (ncalls_partially_synchronous > 0))
                {
                    break;
                }
                continue;
            }

            case memory_debug_synchronicity_t::partially_synchronous:
            {
                if((ncalls_synchronous > 0) || //
                   (ncalls_asynchronous > 0) || //
                   (ncalls_host > 0))
                {
                    break;
                }
                continue;
            }

            case memory_debug_synchronicity_t::host_or_partially_synchronous:
            {
                if((ncalls_synchronous > 0) || //
                   (ncalls_asynchronous > 0))
                {
                    break;
                }
                continue;
            }

            case memory_debug_synchronicity_t::host:
            {
                if((ncalls_synchronous > 0) || //
                   (ncalls_asynchronous > 0) || //
                   (ncalls_partially_synchronous > 0))
                {
                    break;
                }
                continue;
            }

            case memory_debug_synchronicity_t::unknown:
            case memory_debug_synchronicity_t::depends:
            {
                continue;
            }
            }

            std::cerr << "Error: rocsparse_" << name << " is declared '"
                      << memory_debug_synchronicity_t2string(sync)
                      << "' but production code returns:" << std::endl;
            std::cerr << "   ncalls_synchronous           : " << ncalls_synchronous << std::endl;
            std::cerr << "   ncalls_asynchronous          : " << ncalls_asynchronous << std::endl;
            std::cerr << "   ncalls_partially_synchronous : " << ncalls_partially_synchronous
                      << std::endl;
            std::cerr << "   ncalls_host                  : " << ncalls_host << std::endl;
            failed = true;
        }

        return (failed) ? rocsparse_status_internal_error : rocsparse_status_success;
    }

}

#endif
