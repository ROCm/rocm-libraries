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

// Pull in the declarations from the header (enum, struct, class).
// We only need the rocsparse_clients_test namespace types, not the WRAP macros,
// but including the full header is harmless for a single TU.
#include "../tests/rocsparse_test.hpp"
using namespace rocsparse_clients_test;
#include <fstream>
#include <iostream>
#include <map>

void rocsparse_execution_record_property(rocsparse_handle handle, const char* name)
{
    if(rocsparse_clients_test::function_properties_t::instance().enabled())
    {
        auto& info = rocsparse_clients_test::function_properties_t::instance().get_info(name);

        if(rocsparse_execution_is_synchronous(handle))
        {
            info.add_call(rocsparse_clients_test::sync_property::synchronous);
        }
        else if(rocsparse_execution_is_asynchronous(handle))
        {
            info.add_call(rocsparse_clients_test::sync_property::asynchronous);
        }
        else if(rocsparse_execution_is_partially_synchronous(handle))
        {
            info.add_call(rocsparse_clients_test::sync_property::partially_synchronous);
        }
        else if(rocsparse_execution_is_host(handle))
        {
            info.add_call(rocsparse_clients_test::sync_property::host);
        }
    }
}

void rocsparse_execution_check_sync_property(rocsparse_handle handle, const char* name)
{
    if(rocsparse_clients_test::function_properties_t::instance().enabled())
    {
        const auto info = rocsparse_clients_test::function_properties_t::instance().get_info(name);
        const auto sync = info.get_sync();

        rocsparse_execution_record_property(handle, name);

        const int32_t is_synchronous = rocsparse_execution_is_synchronous(handle);
        if(is_synchronous)
        {

            if((sync != rocsparse_clients_test::sync_property::synchronous)
               && (sync != rocsparse_clients_test::sync_property::host_or_synchronous)
               && (sync != rocsparse_clients_test::sync_property::depends))
            {
                std::cerr << "Error: rocsparse_" << name << " is declared '"
                          << rocsparse_clients_test::sync_property2string(sync)
                          << "' but production code returns 'synchronous'" << std::endl;
                throw rocsparse_status_internal_error;
            }

            return;
        }

        const int32_t is_asynchronous = rocsparse_execution_is_asynchronous(handle);
        if(is_asynchronous)
        {

            if((sync != rocsparse_clients_test::sync_property::asynchronous)
               && (sync != rocsparse_clients_test::sync_property::host_or_asynchronous)
               && (sync != rocsparse_clients_test::sync_property::depends))
            {
                std::cerr << "Error: rocsparse_" << name << " is declared '"
                          << rocsparse_clients_test::sync_property2string(sync)
                          << "' but production code returns 'asynchronous'" << std::endl;
                throw rocsparse_status_internal_error;
            }
            return;
        }

        const int32_t is_partially_synchronous
            = rocsparse_execution_is_partially_synchronous(handle);
        if(is_partially_synchronous)
        {
            if((sync != rocsparse_clients_test::sync_property::partially_synchronous)
               && (sync != rocsparse_clients_test::sync_property::host_or_partially_synchronous)
               && (sync != rocsparse_clients_test::sync_property::depends))
            {
                std::cerr << "Error: rocsparse_" << name << " is declared '"
                          << rocsparse_clients_test::sync_property2string(sync)
                          << "' but production code returns 'partially_synchronous'" << std::endl;
                throw rocsparse_status_internal_error;
            }
            return;
        }

        const int32_t is_host = rocsparse_execution_is_host(handle);
        if(is_host)
        {

            if((sync != rocsparse_clients_test::sync_property::host)
               && (sync != rocsparse_clients_test::sync_property::host_or_asynchronous)
               && (sync != rocsparse_clients_test::sync_property::host_or_synchronous)
               && (sync != rocsparse_clients_test::sync_property::depends))
            {
                std::cerr << "Error: rocsparse_" << name << " is declared '"
                          << rocsparse_clients_test::sync_property2string(sync)
                          << "' but production code returns 'host'" << std::endl;
                throw rocsparse_status_internal_error;
            }
            return;
        }
    }
}

static std::map<const char*, function_info> s_map{
    {"axpby", {sync_property::host_or_asynchronous}},
    {"bsrgeam_nnzb", {sync_property::depends}},
    {"bsrgemm_nnzb", {sync_property::depends}},
    {"bsric0_clear", {sync_property::host_or_synchronous}},
    {"bsric0_zero_pivot", {sync_property::synchronous}},
    {"bsrilu0_clear", {sync_property::host_or_synchronous}},
    {"bsrilu0_zero_pivot", {sync_property::synchronous}},
    {"bsrmv_clear", {sync_property::asynchronous}},
    {"bsrsm_clear", {sync_property::host_or_synchronous}},
    {"bsrsm_zero_pivot", {sync_property::synchronous}},
    {"bsrsv_clear", {sync_property::host_or_synchronous}},
    {"bsrsv_zero_pivot", {sync_property::synchronous}},
    {"caxpyi", {sync_property::asynchronous}},
    {"cbsr2csr", {sync_property::asynchronous}},
    {"cbsrgeam", {sync_property::synchronous}},
    {"cbsrgemm", {sync_property::synchronous}},
    {"cbsrgemm_buffer_size", {sync_property::synchronous}},
    {"cbsric0", {sync_property::asynchronous}},
    {"cbsric0_analysis", {sync_property::synchronous}},
    {"cbsric0_buffer_size", {sync_property::asynchronous}},
    {"cbsrilu0", {sync_property::asynchronous}},
    {"cbsrilu0_analysis", {sync_property::synchronous}},
    {"cbsrilu0_buffer_size", {sync_property::asynchronous}},
    {"cbsrilu0_numeric_boost", {sync_property::asynchronous}},
    {"cbsrmm", {sync_property::asynchronous}},
    {"cbsrmv", {sync_property::asynchronous}},
    {"cbsrmv_analysis", {sync_property::asynchronous}},
    {"cbsrpad_value", {sync_property::asynchronous}},
    {"cbsrsm_analysis", {sync_property::synchronous}},
    {"cbsrsm_buffer_size", {sync_property::asynchronous}},
    {"cbsrsm_solve", {sync_property::asynchronous}},
    {"cbsrsv_analysis", {sync_property::synchronous}},
    {"cbsrsv_buffer_size", {sync_property::asynchronous}},
    {"cbsrsv_solve", {sync_property::asynchronous}},
    {"cbsrxmv", {sync_property::asynchronous}},
    {"ccheck_matrix_coo", {sync_property::asynchronous}},
    {"ccheck_matrix_coo_buffer_size", {sync_property::asynchronous}},
    {"ccheck_matrix_csc", {sync_property::asynchronous}},
    {"ccheck_matrix_csc_buffer_size", {sync_property::asynchronous}},
    {"ccheck_matrix_csr", {sync_property::asynchronous}},
    {"ccheck_matrix_csr_buffer_size", {sync_property::asynchronous}},
    {"ccheck_matrix_ell", {sync_property::asynchronous}},
    {"ccheck_matrix_ell_buffer_size", {sync_property::asynchronous}},
    {"ccheck_matrix_gebsc", {sync_property::asynchronous}},
    {"ccheck_matrix_gebsc_buffer_size", {sync_property::asynchronous}},
    {"ccheck_matrix_gebsr", {sync_property::asynchronous}},
    {"ccheck_matrix_gebsr_buffer_size", {sync_property::asynchronous}},
    {"ccoo2dense", {sync_property::asynchronous}},
    {"ccoomv", {sync_property::asynchronous}},
    {"ccsc2dense", {sync_property::asynchronous}},
    {"ccsr2bsr", {sync_property::synchronous}},
    {"ccsr2csc", {sync_property::asynchronous}},
    {"ccsr2csr_compress", {sync_property::synchronous}},
    {"ccsr2dense", {sync_property::asynchronous}},
    {"ccsr2ell", {sync_property::asynchronous}},
    {"ccsr2gebsr", {sync_property::synchronous}},
    {"ccsr2gebsr_buffer_size", {sync_property::asynchronous}},
    {"ccsr2hyb", {sync_property::synchronous}},
    {"ccsrcolor", {sync_property::synchronous}},
    {"ccsrgeam", {sync_property::synchronous}},
    {"ccsrgemm", {sync_property::asynchronous}},
    {"ccsrgemm_buffer_size", {sync_property::asynchronous}},
    {"ccsrgemm_numeric", {sync_property::asynchronous}},
    {"ccsric0", {sync_property::asynchronous}},
    {"ccsric0_analysis", {sync_property::synchronous}},
    {"ccsric0_buffer_size", {sync_property::asynchronous}},
    {"ccsrilu0", {sync_property::asynchronous}},
    {"ccsrilu0_analysis", {sync_property::synchronous}},
    {"ccsrilu0_buffer_size", {sync_property::asynchronous}},
    {"ccsrilu0_numeric_boost", {sync_property::asynchronous}},
    {"ccsritilu0_compute", {sync_property::synchronous}},
    {"ccsritilu0_compute_ex", {sync_property::synchronous}},
    {"ccsritilu0_history", {sync_property::synchronous}},
    {"ccsritsv_analysis", {sync_property::synchronous}},
    {"ccsritsv_buffer_size", {sync_property::asynchronous}},
    {"ccsritsv_solve", {sync_property::asynchronous}},
    {"ccsritsv_solve_ex", {sync_property::asynchronous}},
    {"ccsrmm", {sync_property::asynchronous}},
    {"ccsrmv", {sync_property::asynchronous}},
    {"ccsrmv_analysis", {sync_property::synchronous}},
    {"ccsrsm_analysis", {sync_property::synchronous}},
    {"ccsrsm_buffer_size", {sync_property::asynchronous}},
    {"ccsrsm_solve", {sync_property::asynchronous}},
    {"ccsrsv_analysis", {sync_property::synchronous}},
    {"ccsrsv_buffer_size", {sync_property::asynchronous}},
    {"ccsrsv_solve", {sync_property::asynchronous}},
    {"cdense2coo", {sync_property::synchronous}},
    {"cdense2csc", {sync_property::synchronous}},
    {"cdense2csr", {sync_property::synchronous}},
    {"cdotci", {sync_property::asynchronous}},
    {"cdoti", {sync_property::asynchronous}},
    {"cell2csr", {sync_property::asynchronous}},
    {"cellmv", {sync_property::asynchronous}},
    {"cgebsr2csr", {sync_property::asynchronous}},
    {"cgebsr2gebsc", {sync_property::asynchronous}},
    {"cgebsr2gebsc_buffer_size", {sync_property::asynchronous}},
    {"cgebsr2gebsr", {sync_property::synchronous}},
    {"cgebsr2gebsr_buffer_size", {sync_property::asynchronous}},
    {"cgebsrmm", {sync_property::asynchronous}},
    {"cgebsrmv", {sync_property::asynchronous}},
    {"cgemmi", {sync_property::asynchronous}},
    {"cgemvi", {sync_property::asynchronous}},
    {"cgemvi_buffer_size", {sync_property::asynchronous}},
    {"cgpsv_interleaved_batch", {sync_property::asynchronous}},
    {"cgpsv_interleaved_batch_buffer_size", {sync_property::asynchronous}},
    {"cgthr", {sync_property::asynchronous}},
    {"cgthrz", {sync_property::asynchronous}},
    {"cgtsv", {sync_property::asynchronous}},
    {"cgtsv_buffer_size", {sync_property::asynchronous}},
    {"cgtsv_interleaved_batch", {sync_property::asynchronous}},
    {"cgtsv_interleaved_batch_buffer_size", {sync_property::asynchronous}},
    {"cgtsv_no_pivot", {sync_property::asynchronous}},
    {"cgtsv_no_pivot_buffer_size", {sync_property::asynchronous}},
    {"cgtsv_no_pivot_strided_batch", {sync_property::asynchronous}},
    {"cgtsv_no_pivot_strided_batch_buffer_size", {sync_property::asynchronous}},
    {"check_matrix_hyb", {sync_property::synchronous}},
    {"check_matrix_hyb_buffer_size", {sync_property::host_or_asynchronous}},
    {"check_spmat", {sync_property::depends}},
    {"chyb2csr", {sync_property::asynchronous}},
    {"chybmv", {sync_property::asynchronous}},
    {"cnnz", {sync_property::asynchronous}},
    {"cnnz_compress", {sync_property::synchronous}},
    {"coo2csr", {sync_property::asynchronous}},
    {"coosort_buffer_size", {sync_property::host}},
    {"coosort_by_column", {sync_property::depends}},
    {"coosort_by_row", {sync_property::depends}},
    {"create_identity_permutation", {sync_property::host_or_asynchronous}},
    {"cscsort", {sync_property::host_or_asynchronous}},
    {"cscsort_buffer_size", {sync_property::host}},
    {"csctr", {sync_property::asynchronous}},
    {"csr2bsr_nnz", {sync_property::depends}},
    {"csr2coo", {sync_property::host_or_asynchronous}},
    {"csr2csc_buffer_size", {sync_property::host}},
    {"csr2ell_width", {sync_property::asynchronous}},
    {"csr2gebsr_nnz", {sync_property::depends}},
    {"csrgeam_nnz", {sync_property::depends}},
    {"csrgemm_nnz", {sync_property::depends}},
    {"csrgemm_symbolic", {sync_property::depends}},
    {"csric0_clear", {sync_property::host_or_synchronous}},
    {"csric0_get_tolerance", {sync_property::host}},
    {"csric0_set_tolerance", {sync_property::host}},
    {"csric0_singular_pivot", {sync_property::synchronous}},
    {"csric0_zero_pivot", {sync_property::synchronous}},
    {"csrilu0_clear", {sync_property::host_or_synchronous}},
    {"csrilu0_get_tolerance", {sync_property::host}},
    {"csrilu0_set_tolerance", {sync_property::host}},
    {"csrilu0_singular_pivot", {sync_property::synchronous}},
    {"csrilu0_zero_pivot", {sync_property::synchronous}},
    {"csritilu0_buffer_size", {sync_property::host_or_synchronous}},
    {"csritilu0_preprocess", {sync_property::host_or_synchronous}},
    {"csritsv_clear", {sync_property::host_or_synchronous}},
    {"csritsv_zero_pivot", {sync_property::synchronous}},
    {"csrmv_clear", {sync_property::host_or_synchronous}},
    {"csrsm_clear", {sync_property::host_or_synchronous}},
    {"csrsm_zero_pivot", {sync_property::synchronous}},
    {"csrsort", {sync_property::host_or_asynchronous}},
    {"csrsort_buffer_size", {sync_property::host}},
    {"csrsv_clear", {sync_property::host_or_synchronous}},
    {"csrsv_zero_pivot", {sync_property::synchronous}},
    {"daxpyi", {sync_property::asynchronous}},
    {"dbsr2csr", {sync_property::asynchronous}},
    {"dbsrgeam", {sync_property::synchronous}},
    {"dbsrgemm", {sync_property::synchronous}},
    {"dbsrgemm_buffer_size", {sync_property::synchronous}},
    {"dbsric0", {sync_property::asynchronous}},
    {"dbsric0_analysis", {sync_property::synchronous}},
    {"dbsric0_buffer_size", {sync_property::asynchronous}},
    {"dbsrilu0", {sync_property::asynchronous}},
    {"dbsrilu0_analysis", {sync_property::synchronous}},
    {"dbsrilu0_buffer_size", {sync_property::asynchronous}},
    {"dbsrilu0_numeric_boost", {sync_property::asynchronous}},
    {"dbsrmm", {sync_property::asynchronous}},
    {"dbsrmv", {sync_property::asynchronous}},
    {"dbsrmv_analysis", {sync_property::asynchronous}},
    {"dbsrpad_value", {sync_property::asynchronous}},
    {"dbsrsm_analysis", {sync_property::synchronous}},
    {"dbsrsm_buffer_size", {sync_property::asynchronous}},
    {"dbsrsm_solve", {sync_property::asynchronous}},
    {"dbsrsv_analysis", {sync_property::synchronous}},
    {"dbsrsv_buffer_size", {sync_property::asynchronous}},
    {"dbsrsv_solve", {sync_property::asynchronous}},
    {"dbsrxmv", {sync_property::asynchronous}},
    {"dcbsrilu0_numeric_boost", {sync_property::asynchronous}},
    {"dccsrilu0_numeric_boost", {sync_property::asynchronous}},
    {"dcheck_matrix_coo", {sync_property::asynchronous}},
    {"dcheck_matrix_coo_buffer_size", {sync_property::asynchronous}},
    {"dcheck_matrix_csc", {sync_property::asynchronous}},
    {"dcheck_matrix_csc_buffer_size", {sync_property::asynchronous}},
    {"dcheck_matrix_csr", {sync_property::asynchronous}},
    {"dcheck_matrix_csr_buffer_size", {sync_property::asynchronous}},
    {"dcheck_matrix_ell", {sync_property::asynchronous}},
    {"dcheck_matrix_ell_buffer_size", {sync_property::asynchronous}},
    {"dcheck_matrix_gebsc", {sync_property::asynchronous}},
    {"dcheck_matrix_gebsc_buffer_size", {sync_property::asynchronous}},
    {"dcheck_matrix_gebsr", {sync_property::asynchronous}},
    {"dcheck_matrix_gebsr_buffer_size", {sync_property::asynchronous}},
    {"dcoo2dense", {sync_property::asynchronous}},
    {"dcoomv", {sync_property::asynchronous}},
    {"dcsc2dense", {sync_property::asynchronous}},
    {"dcsr2bsr", {sync_property::synchronous}},
    {"dcsr2csc", {sync_property::asynchronous}},
    {"dcsr2csr_compress", {sync_property::synchronous}},
    {"dcsr2dense", {sync_property::asynchronous}},
    {"dcsr2ell", {sync_property::asynchronous}},
    {"dcsr2gebsr", {sync_property::synchronous}},
    {"dcsr2gebsr_buffer_size", {sync_property::asynchronous}},
    {"dcsr2hyb", {sync_property::synchronous}},
    {"dcsrcolor", {sync_property::synchronous}},
    {"dcsrgeam", {sync_property::synchronous}},
    {"dcsrgemm", {sync_property::asynchronous}},
    {"dcsrgemm_buffer_size", {sync_property::asynchronous}},
    {"dcsrgemm_numeric", {sync_property::asynchronous}},
    {"dcsric0", {sync_property::asynchronous}},
    {"dcsric0_analysis", {sync_property::synchronous}},
    {"dcsric0_buffer_size", {sync_property::asynchronous}},
    {"dcsrilu0", {sync_property::asynchronous}},
    {"dcsrilu0_analysis", {sync_property::synchronous}},
    {"dcsrilu0_buffer_size", {sync_property::asynchronous}},
    {"dcsrilu0_numeric_boost", {sync_property::asynchronous}},
    {"dcsritilu0_compute", {sync_property::synchronous}},
    {"dcsritilu0_compute_ex", {sync_property::synchronous}},
    {"dcsritilu0_history", {sync_property::synchronous}},
    {"dcsritsv_analysis", {sync_property::synchronous}},
    {"dcsritsv_buffer_size", {sync_property::asynchronous}},
    {"dcsritsv_solve", {sync_property::asynchronous}},
    {"dcsritsv_solve_ex", {sync_property::asynchronous}},
    {"dcsrmm", {sync_property::asynchronous}},
    {"dcsrmv", {sync_property::asynchronous}},
    {"dcsrmv_analysis", {sync_property::synchronous}},
    {"dcsrsm_analysis", {sync_property::synchronous}},
    {"dcsrsm_buffer_size", {sync_property::asynchronous}},
    {"dcsrsm_solve", {sync_property::asynchronous}},
    {"dcsrsv_analysis", {sync_property::synchronous}},
    {"dcsrsv_buffer_size", {sync_property::asynchronous}},
    {"dcsrsv_solve", {sync_property::asynchronous}},
    {"ddense2coo", {sync_property::synchronous}},
    {"ddense2csc", {sync_property::synchronous}},
    {"ddense2csr", {sync_property::synchronous}},
    {"ddoti", {sync_property::asynchronous}},
    {"dell2csr", {sync_property::asynchronous}},
    {"dellmv", {sync_property::asynchronous}},
    {"dense_to_sparse", {sync_property::depends}},
    {"dgebsr2csr", {sync_property::asynchronous}},
    {"dgebsr2gebsc", {sync_property::asynchronous}},
    {"dgebsr2gebsc_buffer_size", {sync_property::asynchronous}},
    {"dgebsr2gebsr", {sync_property::synchronous}},
    {"dgebsr2gebsr_buffer_size", {sync_property::asynchronous}},
    {"dgebsrmm", {sync_property::asynchronous}},
    {"dgebsrmv", {sync_property::asynchronous}},
    {"dgemmi", {sync_property::asynchronous}},
    {"dgemvi", {sync_property::asynchronous}},
    {"dgemvi_buffer_size", {sync_property::asynchronous}},
    {"dgpsv_interleaved_batch", {sync_property::asynchronous}},
    {"dgpsv_interleaved_batch_buffer_size", {sync_property::asynchronous}},
    {"dgthr", {sync_property::asynchronous}},
    {"dgthrz", {sync_property::asynchronous}},
    {"dgtsv", {sync_property::asynchronous}},
    {"dgtsv_buffer_size", {sync_property::asynchronous}},
    {"dgtsv_interleaved_batch", {sync_property::asynchronous}},
    {"dgtsv_interleaved_batch_buffer_size", {sync_property::asynchronous}},
    {"dgtsv_no_pivot", {sync_property::asynchronous}},
    {"dgtsv_no_pivot_buffer_size", {sync_property::asynchronous}},
    {"dgtsv_no_pivot_strided_batch", {sync_property::asynchronous}},
    {"dgtsv_no_pivot_strided_batch_buffer_size", {sync_property::asynchronous}},
    {"dhyb2csr", {sync_property::asynchronous}},
    {"dhybmv", {sync_property::asynchronous}},
    {"dnnz", {sync_property::asynchronous}},
    {"dnnz_compress", {sync_property::synchronous}},
    {"dprune_csr2csr", {sync_property::synchronous}},
    {"dprune_csr2csr_buffer_size", {sync_property::asynchronous}},
    {"dprune_csr2csr_by_percentage", {sync_property::synchronous}},
    {"dprune_csr2csr_by_percentage_buffer_size", {sync_property::asynchronous}},
    {"dprune_csr2csr_nnz", {sync_property::asynchronous}},
    {"dprune_csr2csr_nnz_by_percentage", {sync_property::asynchronous}},
    {"dprune_dense2csr", {sync_property::synchronous}},
    {"dprune_dense2csr_buffer_size", {sync_property::asynchronous}},
    {"dprune_dense2csr_by_percentage", {sync_property::synchronous}},
    {"dprune_dense2csr_by_percentage_buffer_size", {sync_property::asynchronous}},
    {"dprune_dense2csr_nnz", {sync_property::asynchronous}},
    {"dprune_dense2csr_nnz_by_percentage", {sync_property::synchronous}},
    {"droti", {sync_property::asynchronous}},
    {"dsbsrilu0_numeric_boost", {sync_property::asynchronous}},
    {"dscsrilu0_numeric_boost", {sync_property::asynchronous}},
    {"dsctr", {sync_property::asynchronous}},
    {"ell2csr_nnz", {sync_property::host_or_synchronous}},
    {"extract", {sync_property::asynchronous}},
    {"extract_buffer_size", {sync_property::host}},
    {"extract_nnz", {sync_property::host}},
    {"gather", {sync_property::host_or_asynchronous}},
    {"gebsr2gebsr_nnz", {sync_property::depends}},
    {"get_git_rev", {sync_property::host}},
    {"get_pointer_mode", {sync_property::asynchronous}},
    {"get_stream", {sync_property::host}},
    {"get_version", {sync_property::host}},
    {"hyb2csr_buffer_size", {sync_property::host}},
    {"inverse_permutation", {sync_property::asynchronous}},
    {"isctr", {sync_property::asynchronous}},
    {"rot", {sync_property::host_or_asynchronous}},
    {"saxpyi", {sync_property::asynchronous}},
    {"sbsr2csr", {sync_property::asynchronous}},
    {"sbsrgeam", {sync_property::synchronous}},
    {"sbsrgemm", {sync_property::synchronous}},
    {"sbsrgemm_buffer_size", {sync_property::synchronous}},
    {"sbsric0", {sync_property::asynchronous}},
    {"sbsric0_analysis", {sync_property::synchronous}},
    {"sbsric0_buffer_size", {sync_property::asynchronous}},
    {"sbsrilu0", {sync_property::asynchronous}},
    {"sbsrilu0_analysis", {sync_property::synchronous}},
    {"sbsrilu0_buffer_size", {sync_property::asynchronous}},
    {"sbsrilu0_numeric_boost", {sync_property::asynchronous}},
    {"sbsrmm", {sync_property::asynchronous}},
    {"sbsrmv", {sync_property::asynchronous}},
    {"sbsrmv_analysis", {sync_property::asynchronous}},
    {"sbsrpad_value", {sync_property::asynchronous}},
    {"sbsrsm_analysis", {sync_property::synchronous}},
    {"sbsrsm_buffer_size", {sync_property::asynchronous}},
    {"sbsrsm_solve", {sync_property::asynchronous}},
    {"sbsrsv_analysis", {sync_property::synchronous}},
    {"sbsrsv_buffer_size", {sync_property::asynchronous}},
    {"sbsrsv_solve", {sync_property::asynchronous}},
    {"sbsrxmv", {sync_property::asynchronous}},
    {"scatter", {sync_property::host_or_asynchronous}},
    {"scheck_matrix_coo", {sync_property::asynchronous}},
    {"scheck_matrix_coo_buffer_size", {sync_property::asynchronous}},
    {"scheck_matrix_csc", {sync_property::asynchronous}},
    {"scheck_matrix_csc_buffer_size", {sync_property::asynchronous}},
    {"scheck_matrix_csr", {sync_property::asynchronous}},
    {"scheck_matrix_csr_buffer_size", {sync_property::asynchronous}},
    {"scheck_matrix_ell", {sync_property::asynchronous}},
    {"scheck_matrix_ell_buffer_size", {sync_property::asynchronous}},
    {"scheck_matrix_gebsc", {sync_property::asynchronous}},
    {"scheck_matrix_gebsc_buffer_size", {sync_property::asynchronous}},
    {"scheck_matrix_gebsr", {sync_property::asynchronous}},
    {"scheck_matrix_gebsr_buffer_size", {sync_property::asynchronous}},
    {"scoo2dense", {sync_property::asynchronous}},
    {"scoomv", {sync_property::asynchronous}},
    {"scsc2dense", {sync_property::asynchronous}},
    {"scsr2bsr", {sync_property::synchronous}},
    {"scsr2csc", {sync_property::asynchronous}},
    {"scsr2csr_compress", {sync_property::synchronous}},
    {"scsr2dense", {sync_property::asynchronous}},
    {"scsr2ell", {sync_property::asynchronous}},
    {"scsr2gebsr", {sync_property::synchronous}},
    {"scsr2gebsr_buffer_size", {sync_property::asynchronous}},
    {"scsr2hyb", {sync_property::synchronous}},
    {"scsrcolor", {sync_property::synchronous}},
    {"scsrgeam", {sync_property::synchronous}},
    {"scsrgemm", {sync_property::asynchronous}},
    {"scsrgemm_buffer_size", {sync_property::asynchronous}},
    {"scsrgemm_numeric", {sync_property::asynchronous}},
    {"scsric0", {sync_property::asynchronous}},
    {"scsric0_analysis", {sync_property::synchronous}},
    {"scsric0_buffer_size", {sync_property::asynchronous}},
    {"scsrilu0", {sync_property::asynchronous}},
    {"scsrilu0_analysis", {sync_property::synchronous}},
    {"scsrilu0_buffer_size", {sync_property::asynchronous}},
    {"scsrilu0_numeric_boost", {sync_property::asynchronous}},
    {"scsritilu0_compute", {sync_property::synchronous}},
    {"scsritilu0_compute_ex", {sync_property::synchronous}},
    {"scsritilu0_history", {sync_property::synchronous}},
    {"scsritsv_analysis", {sync_property::synchronous}},
    {"scsritsv_buffer_size", {sync_property::asynchronous}},
    {"scsritsv_solve", {sync_property::asynchronous}},
    {"scsritsv_solve_ex", {sync_property::asynchronous}},
    {"scsrmm", {sync_property::asynchronous}},
    {"scsrmv", {sync_property::asynchronous}},
    {"scsrmv_analysis", {sync_property::synchronous}},
    {"scsrsm_analysis", {sync_property::synchronous}},
    {"scsrsm_buffer_size", {sync_property::asynchronous}},
    {"scsrsm_solve", {sync_property::asynchronous}},
    {"scsrsv_analysis", {sync_property::synchronous}},
    {"scsrsv_buffer_size", {sync_property::asynchronous}},
    {"scsrsv_solve", {sync_property::asynchronous}},
    {"sddmm", {sync_property::asynchronous}},
    {"sddmm_buffer_size", {sync_property::host}},
    {"sddmm_preprocess", {sync_property::host_or_asynchronous}},
    {"sdense2coo", {sync_property::synchronous}},
    {"sdense2csc", {sync_property::synchronous}},
    {"sdense2csr", {sync_property::synchronous}},
    {"sdoti", {sync_property::asynchronous}},
    {"sell2csr", {sync_property::asynchronous}},
    {"sellmv", {sync_property::asynchronous}},
    {"set_identity_permutation", {sync_property::asynchronous}},
    {"set_pointer_mode", {sync_property::host}},
    {"set_stream", {sync_property::host}},
    {"sgebsr2csr", {sync_property::asynchronous}},
    {"sgebsr2gebsc", {sync_property::asynchronous}},
    {"sgebsr2gebsc_buffer_size", {sync_property::asynchronous}},
    {"sgebsr2gebsr", {sync_property::synchronous}},
    {"sgebsr2gebsr_buffer_size", {sync_property::asynchronous}},
    {"sgebsrmm", {sync_property::asynchronous}},
    {"sgebsrmv", {sync_property::asynchronous}},
    {"sgemmi", {sync_property::asynchronous}},
    {"sgemvi", {sync_property::asynchronous}},
    {"sgemvi_buffer_size", {sync_property::asynchronous}},
    {"sgpsv_interleaved_batch", {sync_property::asynchronous}},
    {"sgpsv_interleaved_batch_buffer_size", {sync_property::asynchronous}},
    {"sgthr", {sync_property::asynchronous}},
    {"sgthrz", {sync_property::asynchronous}},
    {"sgtsv", {sync_property::asynchronous}},
    {"sgtsv_buffer_size", {sync_property::asynchronous}},
    {"sgtsv_interleaved_batch", {sync_property::asynchronous}},
    {"sgtsv_interleaved_batch_buffer_size", {sync_property::asynchronous}},
    {"sgtsv_no_pivot", {sync_property::asynchronous}},
    {"sgtsv_no_pivot_buffer_size", {sync_property::asynchronous}},
    {"sgtsv_no_pivot_strided_batch", {sync_property::asynchronous}},
    {"sgtsv_no_pivot_strided_batch_buffer_size", {sync_property::asynchronous}},
    {"shyb2csr", {sync_property::asynchronous}},
    {"shybmv", {sync_property::asynchronous}},
    {"snnz", {sync_property::asynchronous}},
    {"snnz_compress", {sync_property::synchronous}},
    {"sparse_to_dense", {sync_property::host_or_asynchronous}},
    {"sparse_to_sparse", {sync_property::depends}},
    {"sparse_to_sparse_buffer_size", {sync_property::depends}},
    {"spgeam", {sync_property::depends}},
    {"spgeam_buffer_size", {sync_property::host}},
    {"spgeam_get_output", {sync_property::host_or_asynchronous}},
    {"spgeam_set_input", {sync_property::host}},
    {"spgemm", {sync_property::depends}},
    {"spic0", {sync_property::depends}},
    {"spic0_buffer_size", {sync_property::host}},
    {"spic0_descr_create", {sync_property::host}},
    {"spic0_descr_destroy", {sync_property::host_or_synchronous}},
    {"spic0_get_output", {sync_property::asynchronous}},
    {"spic0_set_input", {sync_property::host}},
    {"spilu0", {sync_property::depends}},
    {"spilu0_buffer_size", {sync_property::host}},
    {"spilu0_descr_create", {sync_property::host}},
    {"spilu0_descr_destroy", {sync_property::host_or_synchronous}},
    {"spilu0_get_output", {sync_property::asynchronous}},
    {"spilu0_set_input", {sync_property::host}},
    {"spitsv", {sync_property::depends}},
    {"spmm", {sync_property::host_or_asynchronous}},
    {"spmv", {sync_property::depends}},
    {"spmv_clear_extra", {sync_property::synchronous}},
    {"spmv_set_extra", {sync_property::synchronous}},
    {"spmv_set_input", {sync_property::host}},
    {"sprune_csr2csr", {sync_property::synchronous}},
    {"sprune_csr2csr_buffer_size", {sync_property::asynchronous}},
    {"sprune_csr2csr_by_percentage", {sync_property::synchronous}},
    {"sprune_csr2csr_by_percentage_buffer_size", {sync_property::asynchronous}},
    {"sprune_csr2csr_nnz", {sync_property::asynchronous}},
    {"sprune_csr2csr_nnz_by_percentage", {sync_property::asynchronous}},
    {"sprune_dense2csr", {sync_property::synchronous}},
    {"sprune_dense2csr_buffer_size", {sync_property::asynchronous}},
    {"sprune_dense2csr_by_percentage", {sync_property::synchronous}},
    {"sprune_dense2csr_by_percentage_buffer_size", {sync_property::asynchronous}},
    {"sprune_dense2csr_nnz", {sync_property::asynchronous}},
    {"sprune_dense2csr_nnz_by_percentage", {sync_property::synchronous}},
    {"spsm", {sync_property::depends}},
    {"spsv", {sync_property::depends}},
    {"sptrsm", {sync_property::depends}},
    {"sptrsm_buffer_size", {sync_property::host}},
    {"sptrsm_get_output", {sync_property::asynchronous}},
    {"sptrsm_set_input", {sync_property::host}},
    {"sptrsv", {sync_property::depends}},
    {"sptrsv_buffer_size", {sync_property::host}},
    {"sptrsv_descr_create", {sync_property::host}},
    {"sptrsv_descr_destroy", {sync_property::host_or_synchronous}},
    {"sptrsv_get_output", {sync_property::depends}},
    {"sptrsv_set_input", {sync_property::host}},
    {"spvv", {sync_property::host_or_asynchronous}},
    {"sroti", {sync_property::asynchronous}},
    {"ssctr", {sync_property::asynchronous}},
    {"v2_spmv", {sync_property::depends}},
    {"v2_spmv_buffer_size", {sync_property::host}},
    {"zaxpyi", {sync_property::asynchronous}},
    {"zbsr2csr", {sync_property::asynchronous}},
    {"zbsrgeam", {sync_property::synchronous}},
    {"zbsrgemm", {sync_property::synchronous}},
    {"zbsrgemm_buffer_size", {sync_property::synchronous}},
    {"zbsric0", {sync_property::asynchronous}},
    {"zbsric0_analysis", {sync_property::synchronous}},
    {"zbsric0_buffer_size", {sync_property::asynchronous}},
    {"zbsrilu0", {sync_property::asynchronous}},
    {"zbsrilu0_analysis", {sync_property::synchronous}},
    {"zbsrilu0_buffer_size", {sync_property::asynchronous}},
    {"zbsrilu0_numeric_boost", {sync_property::asynchronous}},
    {"zbsrmm", {sync_property::asynchronous}},
    {"zbsrmv", {sync_property::asynchronous}},
    {"zbsrmv_analysis", {sync_property::asynchronous}},
    {"zbsrpad_value", {sync_property::asynchronous}},
    {"zbsrsm_analysis", {sync_property::synchronous}},
    {"zbsrsm_buffer_size", {sync_property::asynchronous}},
    {"zbsrsm_solve", {sync_property::asynchronous}},
    {"zbsrsv_analysis", {sync_property::synchronous}},
    {"zbsrsv_buffer_size", {sync_property::asynchronous}},
    {"zbsrsv_solve", {sync_property::asynchronous}},
    {"zbsrxmv", {sync_property::asynchronous}},
    {"zcheck_matrix_coo", {sync_property::asynchronous}},
    {"zcheck_matrix_coo_buffer_size", {sync_property::asynchronous}},
    {"zcheck_matrix_csc", {sync_property::asynchronous}},
    {"zcheck_matrix_csc_buffer_size", {sync_property::asynchronous}},
    {"zcheck_matrix_csr", {sync_property::asynchronous}},
    {"zcheck_matrix_csr_buffer_size", {sync_property::asynchronous}},
    {"zcheck_matrix_ell", {sync_property::asynchronous}},
    {"zcheck_matrix_ell_buffer_size", {sync_property::asynchronous}},
    {"zcheck_matrix_gebsc", {sync_property::asynchronous}},
    {"zcheck_matrix_gebsc_buffer_size", {sync_property::asynchronous}},
    {"zcheck_matrix_gebsr", {sync_property::asynchronous}},
    {"zcheck_matrix_gebsr_buffer_size", {sync_property::asynchronous}},
    {"zcoo2dense", {sync_property::asynchronous}},
    {"zcoomv", {sync_property::asynchronous}},
    {"zcsc2dense", {sync_property::asynchronous}},
    {"zcsr2bsr", {sync_property::synchronous}},
    {"zcsr2csc", {sync_property::asynchronous}},
    {"zcsr2csr_compress", {sync_property::synchronous}},
    {"zcsr2dense", {sync_property::asynchronous}},
    {"zcsr2ell", {sync_property::asynchronous}},
    {"zcsr2gebsr", {sync_property::synchronous}},
    {"zcsr2gebsr_buffer_size", {sync_property::asynchronous}},
    {"zcsr2hyb", {sync_property::synchronous}},
    {"zcsrcolor", {sync_property::synchronous}},
    {"zcsrgeam", {sync_property::synchronous}},
    {"zcsrgemm", {sync_property::asynchronous}},
    {"zcsrgemm_buffer_size", {sync_property::asynchronous}},
    {"zcsrgemm_numeric", {sync_property::asynchronous}},
    {"zcsric0", {sync_property::asynchronous}},
    {"zcsric0_analysis", {sync_property::synchronous}},
    {"zcsric0_buffer_size", {sync_property::asynchronous}},
    {"zcsrilu0", {sync_property::asynchronous}},
    {"zcsrilu0_analysis", {sync_property::synchronous}},
    {"zcsrilu0_buffer_size", {sync_property::asynchronous}},
    {"zcsrilu0_numeric_boost", {sync_property::asynchronous}},
    {"zcsritilu0_compute", {sync_property::synchronous}},
    {"zcsritilu0_compute_ex", {sync_property::synchronous}},
    {"zcsritilu0_history", {sync_property::synchronous}},
    {"zcsritsv_analysis", {sync_property::synchronous}},
    {"zcsritsv_buffer_size", {sync_property::asynchronous}},
    {"zcsritsv_solve", {sync_property::asynchronous}},
    {"zcsritsv_solve_ex", {sync_property::asynchronous}},
    {"zcsrmm", {sync_property::asynchronous}},
    {"zcsrmv", {sync_property::asynchronous}},
    {"zcsrmv_analysis", {sync_property::synchronous}},
    {"zcsrsm_analysis", {sync_property::synchronous}},
    {"zcsrsm_buffer_size", {sync_property::asynchronous}},
    {"zcsrsm_solve", {sync_property::asynchronous}},
    {"zcsrsv_analysis", {sync_property::synchronous}},
    {"zcsrsv_buffer_size", {sync_property::asynchronous}},
    {"zcsrsv_solve", {sync_property::asynchronous}},
    {"zdense2coo", {sync_property::synchronous}},
    {"zdense2csc", {sync_property::synchronous}},
    {"zdense2csr", {sync_property::synchronous}},
    {"zdotci", {sync_property::asynchronous}},
    {"zdoti", {sync_property::asynchronous}},
    {"zell2csr", {sync_property::asynchronous}},
    {"zellmv", {sync_property::asynchronous}},
    {"zgebsr2csr", {sync_property::asynchronous}},
    {"zgebsr2gebsc", {sync_property::asynchronous}},
    {"zgebsr2gebsc_buffer_size", {sync_property::asynchronous}},
    {"zgebsr2gebsr", {sync_property::synchronous}},
    {"zgebsr2gebsr_buffer_size", {sync_property::asynchronous}},
    {"zgebsrmm", {sync_property::asynchronous}},
    {"zgebsrmv", {sync_property::asynchronous}},
    {"zgemmi", {sync_property::asynchronous}},
    {"zgemvi", {sync_property::asynchronous}},
    {"zgemvi_buffer_size", {sync_property::asynchronous}},
    {"zgpsv_interleaved_batch", {sync_property::asynchronous}},
    {"zgpsv_interleaved_batch_buffer_size", {sync_property::asynchronous}},
    {"zgthr", {sync_property::asynchronous}},
    {"zgthrz", {sync_property::asynchronous}},
    {"zgtsv", {sync_property::asynchronous}},
    {"zgtsv_buffer_size", {sync_property::asynchronous}},
    {"zgtsv_interleaved_batch", {sync_property::asynchronous}},
    {"zgtsv_interleaved_batch_buffer_size", {sync_property::asynchronous}},
    {"zgtsv_no_pivot", {sync_property::asynchronous}},
    {"zgtsv_no_pivot_buffer_size", {sync_property::asynchronous}},
    {"zgtsv_no_pivot_strided_batch", {sync_property::asynchronous}},
    {"zgtsv_no_pivot_strided_batch_buffer_size", {sync_property::asynchronous}},
    {"zhyb2csr", {sync_property::asynchronous}},
    {"zhybmv", {sync_property::asynchronous}},
    {"znnz", {sync_property::asynchronous}},
    {"znnz_compress", {sync_property::synchronous}},
    {"zsctr", {sync_property::asynchronous}},
    {"bell_get", {sync_property::host}},
    {"bsr_get", {sync_property::host}},
    {"bsr_set_pointers", {sync_property::host}},
    {"const_bell_get", {sync_property::host}},
    {"const_bsr_get", {sync_property::host}},
    {"const_coo_aos_get", {sync_property::host}},
    {"const_coo_get", {sync_property::host}},
    {"const_csc_get", {sync_property::host}},
    {"const_csr_get", {sync_property::host}},
    {"const_dnmat_get", {sync_property::host}},
    {"const_dnmat_get_values", {sync_property::host}},
    {"const_dnvec_get", {sync_property::host}},
    {"const_dnvec_get_values", {sync_property::host}},
    {"const_ell_get", {sync_property::host}},
    {"const_sell_get", {sync_property::host}},
    {"const_spmat_get_values", {sync_property::host}},
    {"const_spvec_get", {sync_property::host}},
    {"const_spvec_get_values", {sync_property::host}},
    {"coo_aos_get", {sync_property::host}},
    {"coo_aos_set_pointers", {sync_property::host}},
    {"coo_get", {sync_property::host}},
    {"coo_set_pointers", {sync_property::host}},
    {"coo_set_strided_batch", {sync_property::host}},
    {"copy_color_info", {sync_property::host}},
    {"copy_hyb_mat", {sync_property::host_or_synchronous}},
    {"copy_mat_descr", {sync_property::host}},
    {"copy_mat_info", {sync_property::host_or_synchronous}},
    {"create_bell_descr", {sync_property::host}},
    {"create_bsr_descr", {sync_property::host}},
    {"create_color_info", {sync_property::host}},
    {"create_const_bell_descr", {sync_property::host}},
    {"create_const_coo_descr", {sync_property::host}},
    {"create_const_csc_descr", {sync_property::host}},
    {"create_const_csr_descr", {sync_property::host}},
    {"create_const_dnmat_descr", {sync_property::host}},
    {"create_const_dnvec_descr", {sync_property::host}},
    {"create_const_sell_descr", {sync_property::host}},
    {"create_const_spvec_descr", {sync_property::host}},
    {"create_coo_aos_descr", {sync_property::host}},
    {"create_coo_descr", {sync_property::host}},
    {"create_csc_descr", {sync_property::host}},
    {"create_csr_descr", {sync_property::host}},
    {"create_dnmat_descr", {sync_property::host}},
    {"create_dnvec_descr", {sync_property::host}},
    {"create_ell_descr", {sync_property::host}},
    {"create_extract_descr", {sync_property::synchronous}},
    {"create_handle", {sync_property::synchronous}},
    {"create_hyb_mat", {sync_property::host}},
    {"create_mat_descr", {sync_property::host}},
    {"create_mat_info", {sync_property::host}},
    {"create_sell_descr", {sync_property::host}},
    {"create_sparse_to_sparse_descr", {sync_property::host}},
    {"create_spgeam_descr", {sync_property::host}},
    {"create_spmv_descr", {sync_property::host}},
    {"create_sptrsm_descr", {sync_property::host}},
    {"create_sptrsv_descr", {sync_property::host}},
    {"create_spvec_descr", {sync_property::host}},
    {"csc_get", {sync_property::host}},
    {"csc_set_pointers", {sync_property::host}},
    {"csc_set_strided_batch", {sync_property::host}},
    {"csr_get", {sync_property::host}},
    {"csr_set_pointers", {sync_property::host}},
    {"csr_set_strided_batch", {sync_property::host}},
    {"destroy_color_info", {sync_property::host}},
    {"destroy_dnmat_descr", {sync_property::host}},
    {"destroy_dnvec_descr", {sync_property::host}},
    {"destroy_error", {sync_property::host}},
    {"destroy_extract_descr", {sync_property::synchronous}},
    {"destroy_handle", {sync_property::synchronous}},
    {"destroy_hyb_mat", {sync_property::synchronous}},
    {"destroy_mat_descr", {sync_property::host}},
    {"destroy_mat_info", {sync_property::synchronous}},
    {"destroy_sparse_to_sparse_descr", {sync_property::host_or_synchronous}},
    {"destroy_spgeam_descr", {sync_property::host_or_synchronous}},
    {"destroy_spmat_descr", {sync_property::depends}},
    {"destroy_spmv_descr", {sync_property::host_or_synchronous}},
    {"destroy_sptrsm_descr", {sync_property::host_or_synchronous}},
    {"destroy_sptrsv_descr", {sync_property::host_or_synchronous}},
    {"destroy_spvec_descr", {sync_property::host}},
    {"dnmat_get", {sync_property::host}},
    {"dnmat_get_strided_batch", {sync_property::host}},
    {"dnmat_get_values", {sync_property::host}},
    {"dnmat_set_strided_batch", {sync_property::host}},
    {"dnmat_set_values", {sync_property::host}},
    {"dnvec_get", {sync_property::host}},
    {"dnvec_get_strided_batch", {sync_property::host}},
    {"dnvec_get_values", {sync_property::host}},
    {"dnvec_set_strided_batch", {sync_property::host}},
    {"dnvec_set_values", {sync_property::host}},
    {"ell_get", {sync_property::host}},
    {"ell_set_pointers", {sync_property::host}},
    {"sell_get", {sync_property::host}},
    {"set_mat_diag_type", {sync_property::host}},
    {"set_mat_fill_mode", {sync_property::host}},
    {"set_mat_index_base", {sync_property::host}},
    {"set_mat_storage_mode", {sync_property::host}},
    {"set_mat_type", {sync_property::host}},
    {"sparse_to_sparse_permissive", {sync_property::host}},
    {"spmat_get_attribute", {sync_property::host}},
    {"spmat_get_format", {sync_property::host}},
    {"spmat_get_index_base", {sync_property::host}},
    {"spmat_get_nnz", {sync_property::host}},
    {"spmat_get_size", {sync_property::host}},
    {"spmat_get_strided_batch", {sync_property::host}},
    {"spmat_get_values", {sync_property::host}},
    {"spmat_set_attribute", {sync_property::host}},
    {"spmat_set_nnz", {sync_property::host}},
    {"spmat_set_strided_batch", {sync_property::host}},
    {"spmat_set_values", {sync_property::host}},
    {"spvec_get", {sync_property::host}},
    {"spvec_get_index_base", {sync_property::host}},
    {"spvec_get_values", {sync_property::host}},
    {"spvec_set_values", {sync_property::host}}};

sync_property function_info::get_sync() const
{
    return this->sync;
}
uint64_t function_info::get_ncalls() const
{
    return this->ncalls;
}
uint64_t function_info::get_calls(sync_property sync) const
{
    return this->histo_calls[(int)sync];
}
void function_info::add_call(sync_property sync)
{
    this->histo_calls[(int)sync] += 1;
    ++this->ncalls;
}

function_info::function_info(sync_property s)
    : sync(s)
{
}

function_properties_t& function_properties_t::instance()
{
    static function_properties_t s_function_properties{};
    return s_function_properties;
}

bool function_properties_t::enabled() const
{
    return this->m_enabled;
}

void function_properties_t::enable()
{
    this->m_enabled = true;
}

void function_properties_t::disable()
{
    this->m_enabled = false;
}

void function_properties_t::set_sync_report_filename(const char* value)
{
    this->filename = value;
}

const function_info& function_properties_t::get_info(const char* name) const
{
    return s_map[name];
}

function_info& function_properties_t::get_info(const char* name)
{
    return s_map[name];
}

void function_properties_t::report(rocsparse_handle handle) const
{
    std::ofstream out(this->filename);
    out << "# rocsparse-test: summary of called functions " << std::endl;
    out << "{" << std::endl;
    int64_t count = 0;
    for(const auto& p : s_map)
    {
        const auto& info = p.second;
        if(info.get_ncalls() == 0)
        {
            continue;
        }
        const char*    name = p.first;
        const auto     sync = info.get_sync();
        const uint64_t ncalls_synchronous
            = info.get_calls(rocsparse_clients_test::sync_property::synchronous);
        const uint64_t ncalls_asynchronous
            = info.get_calls(rocsparse_clients_test::sync_property::asynchronous);
        const uint64_t ncalls_partially_synchronous
            = info.get_calls(rocsparse_clients_test::sync_property::partially_synchronous);
        const uint64_t ncalls_host = info.get_calls(rocsparse_clients_test::sync_property::host);

        if(count > 0)
        {
            out << ", " << std::endl;
        }

        out << "{'name': 'rocsparse_" << name << "'," << std::endl;
        out << " 'sync': '" << rocsparse_clients_test::sync_property2string(sync) << "'}"
            << std::endl;
        out << " 'calls': [ 'sync': '" << ncalls_synchronous << "'," << std::endl;
        out << "            'async':      '" << ncalls_asynchronous << "'," << std::endl;
        out << "            'partialsync: '" << ncalls_partially_synchronous << "'," << std::endl;
        out << "            'host':       '" << ncalls_host << "' ]";
        ++count;
    }

    out << std::endl << "}" << std::endl;
    out << "# end report " << std::endl;
}

const std::string& function_properties_t::get_filename() const
{
    return this->filename;
}

rocsparse_status function_properties_t::check(rocsparse_handle handle) const
{
    for(const auto& p : s_map)
    {
        const auto& info = p.second;
        if(info.get_ncalls() == 0)
        {
            continue;
        }
        const char*    name = p.first;
        const auto     sync = info.get_sync();
        const uint64_t ncalls_synchronous
            = info.get_calls(rocsparse_clients_test::sync_property::synchronous);
        const uint64_t ncalls_asynchronous
            = info.get_calls(rocsparse_clients_test::sync_property::asynchronous);
        const uint64_t ncalls_partially_synchronous
            = info.get_calls(rocsparse_clients_test::sync_property::partially_synchronous);
        const uint64_t ncalls_host = info.get_calls(rocsparse_clients_test::sync_property::host);

        switch(sync)
        {
        case rocsparse_clients_test::sync_property::synchronous:
        {
            if((ncalls_asynchronous > 0) || (ncalls_partially_synchronous > 0) || (ncalls_host > 0))
            {
                break;
            }
            continue;
        }

        case rocsparse_clients_test::sync_property::host_or_synchronous:
        {
            if((ncalls_asynchronous > 0) || (ncalls_partially_synchronous > 0))
            {
                break;
            }
            continue;
        }

        case rocsparse_clients_test::sync_property::asynchronous:
        {
            if((ncalls_synchronous > 0) || (ncalls_partially_synchronous > 0) || (ncalls_host > 0))
            {
                break;
            }
            continue;
        }

        case rocsparse_clients_test::sync_property::host_or_asynchronous:
        {
            if((ncalls_synchronous > 0) || (ncalls_partially_synchronous > 0))
            {
                break;
            }
            continue;
        }

        case rocsparse_clients_test::sync_property::partially_synchronous:
        {
            if((ncalls_synchronous > 0) || (ncalls_asynchronous > 0) || (ncalls_host > 0))
            {
                break;
            }
            continue;
        }

        case rocsparse_clients_test::sync_property::host_or_partially_synchronous:
        {
            if((ncalls_synchronous > 0) || (ncalls_asynchronous > 0))
            {
                break;
            }
            continue;
        }

        case rocsparse_clients_test::sync_property::host:
        {
            if((ncalls_synchronous > 0) || (ncalls_asynchronous > 0)
               || (ncalls_partially_synchronous > 0))
            {
                break;
            }
            continue;
        }

        case rocsparse_clients_test::sync_property::unknown:
        case rocsparse_clients_test::sync_property::depends:
        {
            continue;
        }
        }

        std::cerr << "Error: rocsparse_" << name << " is declared '"
                  << rocsparse_clients_test::sync_property2string(sync)
                  << "' but production code returns:" << std::endl;
        std::cerr << "   ncalls_synchronous           : " << ncalls_synchronous << std::endl;
        std::cerr << "   ncalls_asynchronous          : " << ncalls_asynchronous << std::endl;
        std::cerr << "   ncalls_partially_synchronous : " << ncalls_partially_synchronous
                  << std::endl;
        std::cerr << "   ncalls_host                  : " << ncalls_host << std::endl;
        return rocsparse_status_internal_error;
    }
    return rocsparse_status_success;
}
#endif
