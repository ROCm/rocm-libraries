/*! \file */
/* ************************************************************************
 * Copyright (C) 2020-2025 Advanced Micro Devices, Inc. All rights Reserved.
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

/*! \file
 *  \brief rocsparse_graph.hpp exposes C++ templated rocSPARSE routine wrappers
 *  that determine if the routine is ran on a hipgraph context
 */
#ifndef ROCSPARSE_GRAPH_HPP
#define ROCSPARSE_GRAPH_HPP

#include "rocsparse.hpp"
#include "utility.hpp"
#include <utility>
#include <vector>

#if HIP_VERSION >= 50300000
#define GRAPH_TEST 1
#else
#define GRAPH_TEST 0
#endif

#if GRAPH_TEST
#define BEGIN_GRAPH_CAPTURE() handle.rocsparse_stream_begin_capture()
#define END_GRAPH_CAPTURE() handle.rocsparse_stream_end_capture()
#else
#define BEGIN_GRAPH_CAPTURE()
#define END_GRAPH_CAPTURE()
#endif

#define TESTING_TEMPLATE(NAME_)                                                  \
    template <typename... P>                                                     \
    rocsparse_status rocsparse_##NAME_(rocsparse_local_handle& handle, P&&... p) \
    {                                                                            \
        rocsparse_status status;                                                 \
        BEGIN_GRAPH_CAPTURE();                                                   \
                                                                                 \
        status = ::rocsparse_##NAME_(handle, std::forward<P>(p)...);             \
                                                                                 \
        END_GRAPH_CAPTURE();                                                     \
                                                                                 \
        return status;                                                           \
    };

#define TESTING_COMPUTE_TEMPLATE(NAME_)                                          \
    template <typename T, typename... P>                                         \
    rocsparse_status rocsparse_##NAME_(rocsparse_local_handle& handle, P&&... p) \
    {                                                                            \
        rocsparse_status status;                                                 \
        BEGIN_GRAPH_CAPTURE();                                                   \
                                                                                 \
        status = ::rocsparse_##NAME_<T>(handle, std::forward<P>(p)...);          \
                                                                                 \
        END_GRAPH_CAPTURE();                                                     \
                                                                                 \
        return status;                                                           \
    };

namespace testing
{
    /*
    * ===========================================================================
    *    utility SPARSE
    * ===========================================================================
    */

    TESTING_COMPUTE_TEMPLATE(check_matrix_csr_buffer_size)
    TESTING_COMPUTE_TEMPLATE(check_matrix_csr)
    TESTING_COMPUTE_TEMPLATE(check_matrix_coo_buffer_size)
    TESTING_COMPUTE_TEMPLATE(check_matrix_coo)
    TESTING_COMPUTE_TEMPLATE(check_matrix_gebsr_buffer_size)
    TESTING_COMPUTE_TEMPLATE(check_matrix_gebsr)
    TESTING_COMPUTE_TEMPLATE(check_matrix_gebsc_buffer_size)
    TESTING_COMPUTE_TEMPLATE(check_matrix_gebsc)
    TESTING_COMPUTE_TEMPLATE(check_matrix_csc_buffer_size)
    TESTING_COMPUTE_TEMPLATE(check_matrix_csc)
    TESTING_COMPUTE_TEMPLATE(check_matrix_ell_buffer_size)
    TESTING_COMPUTE_TEMPLATE(check_matrix_ell)
    TESTING_TEMPLATE(check_matrix_hyb_buffer_size)
    TESTING_TEMPLATE(check_matrix_hyb)

    /*
    * ===========================================================================
    *    level 1 SPARSE
    * ===========================================================================
    */

    TESTING_COMPUTE_TEMPLATE(axpyi)
    TESTING_COMPUTE_TEMPLATE(doti)
    TESTING_COMPUTE_TEMPLATE(dotci)
    TESTING_COMPUTE_TEMPLATE(gthr)
    TESTING_COMPUTE_TEMPLATE(gthrz)
    TESTING_COMPUTE_TEMPLATE(roti)
    TESTING_COMPUTE_TEMPLATE(sctr)
    TESTING_TEMPLATE(isctr)

    /*
    * ===========================================================================
    *    level 2 SPARSE
    * ===========================================================================
    */

    TESTING_COMPUTE_TEMPLATE(bsrmv_analysis)
    TESTING_COMPUTE_TEMPLATE(bsrmv)
    TESTING_COMPUTE_TEMPLATE(bsrxmv)
    TESTING_TEMPLATE(bsrsv_zero_pivot)
    TESTING_COMPUTE_TEMPLATE(bsrsv_buffer_size)
    TESTING_COMPUTE_TEMPLATE(bsrsv_analysis)
    TESTING_TEMPLATE(bsrmv_clear)
    TESTING_TEMPLATE(bsrsv_clear)
    TESTING_COMPUTE_TEMPLATE(bsrsv_solve)
    TESTING_COMPUTE_TEMPLATE(coomv)
    TESTING_COMPUTE_TEMPLATE(csrmv_analysis)
    TESTING_TEMPLATE(csrmv_clear)
    TESTING_COMPUTE_TEMPLATE(csrmv)
    TESTING_TEMPLATE(csrsv_zero_pivot)
    TESTING_COMPUTE_TEMPLATE(csrsv_buffer_size)
    TESTING_COMPUTE_TEMPLATE(csrsv_analysis)
    TESTING_TEMPLATE(csrsv_clear)
    TESTING_COMPUTE_TEMPLATE(csrsv_solve)
    TESTING_COMPUTE_TEMPLATE(ellmv)
    TESTING_COMPUTE_TEMPLATE(hybmv)
    TESTING_COMPUTE_TEMPLATE(gebsrmv)
    TESTING_COMPUTE_TEMPLATE(gemvi_buffer_size)
    TESTING_COMPUTE_TEMPLATE(gemvi)

    /*
    * ===========================================================================
    *    level 3 SPARSE
    * ===========================================================================
    */

    TESTING_COMPUTE_TEMPLATE(bsrmm)
    TESTING_COMPUTE_TEMPLATE(gebsrmm)
    TESTING_COMPUTE_TEMPLATE(csrmm)
    TESTING_TEMPLATE(csrsm_zero_pivot)
    TESTING_COMPUTE_TEMPLATE(csrsm_buffer_size)
    TESTING_COMPUTE_TEMPLATE(csrsm_analysis)
    TESTING_TEMPLATE(csrsm_clear)
    TESTING_COMPUTE_TEMPLATE(csrsm_solve)
    TESTING_TEMPLATE(bsrsm_zero_pivot)
    TESTING_COMPUTE_TEMPLATE(bsrsm_buffer_size)
    TESTING_COMPUTE_TEMPLATE(bsrsm_analysis)
    TESTING_TEMPLATE(bsrsm_clear)
    TESTING_COMPUTE_TEMPLATE(bsrsm_solve)
    TESTING_COMPUTE_TEMPLATE(gemmi)

    /*
    * ===========================================================================
    *    extra SPARSE
    * ===========================================================================
    */

    TESTING_TEMPLATE(csrgeam_nnz)
    TESTING_COMPUTE_TEMPLATE(csrgeam)
    TESTING_COMPUTE_TEMPLATE(csrgemm_buffer_size)
    TESTING_TEMPLATE(csrgemm_nnz)
    TESTING_COMPUTE_TEMPLATE(csrgemm)
    TESTING_TEMPLATE(csrgemm_symbolic)
    TESTING_COMPUTE_TEMPLATE(csrgemm_numeric)

    /*
    * ===========================================================================
    *    preconditioner SPARSE
    * ===========================================================================
    */

    TESTING_TEMPLATE(bsric0_zero_pivot)
    TESTING_COMPUTE_TEMPLATE(bsric0_buffer_size)
    TESTING_COMPUTE_TEMPLATE(bsric0_analysis)
    TESTING_TEMPLATE(bsric0_clear)
    TESTING_COMPUTE_TEMPLATE(bsric0)
    TESTING_TEMPLATE(bsrilu0_zero_pivot)
    TESTING_COMPUTE_TEMPLATE(bsrilu0_numeric_boost)
    TESTING_TEMPLATE(dsbsrilu0_numeric_boost)
    TESTING_TEMPLATE(dcbsrilu0_numeric_boost)
    TESTING_COMPUTE_TEMPLATE(bsrilu0_buffer_size)
    TESTING_COMPUTE_TEMPLATE(bsrilu0_analysis)
    TESTING_TEMPLATE(bsrilu0_clear)
    TESTING_COMPUTE_TEMPLATE(bsrilu0)
    TESTING_TEMPLATE(csric0_zero_pivot)
    TESTING_TEMPLATE(csric0_singular_pivot)
    TESTING_COMPUTE_TEMPLATE(csric0_buffer_size)
    TESTING_COMPUTE_TEMPLATE(csric0_analysis)
    TESTING_TEMPLATE(csric0_clear)
    TESTING_COMPUTE_TEMPLATE(csric0)
    TESTING_TEMPLATE(csrilu0_zero_pivot)
    TESTING_TEMPLATE(csrilu0_singular_pivot)
    TESTING_COMPUTE_TEMPLATE(csrilu0_numeric_boost)
    TESTING_TEMPLATE(dscsrilu0_numeric_boost)
    TESTING_TEMPLATE(dccsrilu0_numeric_boost)
    TESTING_COMPUTE_TEMPLATE(csrilu0_buffer_size)
    TESTING_COMPUTE_TEMPLATE(csrilu0_analysis)
    TESTING_TEMPLATE(csrilu0_clear)
    TESTING_COMPUTE_TEMPLATE(csrilu0)
    TESTING_COMPUTE_TEMPLATE(gtsv_buffer_size)
    TESTING_COMPUTE_TEMPLATE(gtsv)
    TESTING_COMPUTE_TEMPLATE(gtsv_no_pivot_buffer_size)
    TESTING_COMPUTE_TEMPLATE(gtsv_no_pivot)
    TESTING_COMPUTE_TEMPLATE(gtsv_no_pivot_strided_batch_buffer_size)
    TESTING_COMPUTE_TEMPLATE(gtsv_no_pivot_strided_batch)
    TESTING_COMPUTE_TEMPLATE(gtsv_interleaved_batch_buffer_size)
    TESTING_COMPUTE_TEMPLATE(gtsv_interleaved_batch)
    TESTING_COMPUTE_TEMPLATE(gpsv_interleaved_batch_buffer_size)
    TESTING_COMPUTE_TEMPLATE(gpsv_interleaved_batch)

    /*
    * ===========================================================================
    *    conversion SPARSE
    * ===========================================================================
    */

    TESTING_COMPUTE_TEMPLATE(nnz)
    TESTING_COMPUTE_TEMPLATE(dense2csr)
    TESTING_COMPUTE_TEMPLATE(prune_dense2csr_buffer_size)
    TESTING_COMPUTE_TEMPLATE(prune_dense2csr_nnz)
    TESTING_COMPUTE_TEMPLATE(prune_dense2csr)
    TESTING_COMPUTE_TEMPLATE(prune_dense2csr_by_percentage_buffer_size)
    TESTING_COMPUTE_TEMPLATE(prune_dense2csr_nnz_by_percentage)
    TESTING_COMPUTE_TEMPLATE(prune_dense2csr_by_percentage)
    TESTING_COMPUTE_TEMPLATE(dense2csc)
    TESTING_COMPUTE_TEMPLATE(dense2coo)
    TESTING_COMPUTE_TEMPLATE(csr2dense)
    TESTING_COMPUTE_TEMPLATE(csc2dense)
    TESTING_COMPUTE_TEMPLATE(coo2dense)
    TESTING_COMPUTE_TEMPLATE(nnz_compress)
    TESTING_TEMPLATE(csr2coo)
    TESTING_TEMPLATE(csr2csc_buffer_size)
    TESTING_COMPUTE_TEMPLATE(csr2csc)
    TESTING_COMPUTE_TEMPLATE(gebsr2gebsc_buffer_size)
    TESTING_COMPUTE_TEMPLATE(gebsr2gebsc)
    TESTING_TEMPLATE(csr2ell_width)
    TESTING_COMPUTE_TEMPLATE(csr2ell)
    TESTING_COMPUTE_TEMPLATE(csr2hyb)
    TESTING_TEMPLATE(csr2bsr_nnz)
    TESTING_COMPUTE_TEMPLATE(csr2bsr)
    TESTING_COMPUTE_TEMPLATE(bsrpad_value)
    TESTING_COMPUTE_TEMPLATE(csr2gebsr_buffer_size)
    TESTING_TEMPLATE(csr2gebsr_nnz)
    TESTING_COMPUTE_TEMPLATE(csr2gebsr)
    TESTING_COMPUTE_TEMPLATE(csr2csr_compress)
    TESTING_COMPUTE_TEMPLATE(prune_csr2csr_buffer_size)
    TESTING_COMPUTE_TEMPLATE(prune_csr2csr_nnz)
    TESTING_COMPUTE_TEMPLATE(prune_csr2csr)
    TESTING_COMPUTE_TEMPLATE(prune_csr2csr_by_percentage_buffer_size)
    TESTING_COMPUTE_TEMPLATE(prune_csr2csr_nnz_by_percentage)
    TESTING_COMPUTE_TEMPLATE(prune_csr2csr_by_percentage)
    TESTING_TEMPLATE(coo2csr)
    TESTING_TEMPLATE(ell2csr_nnz)
    TESTING_COMPUTE_TEMPLATE(ell2csr)
    TESTING_TEMPLATE(hyb2csr_buffer_size)
    TESTING_COMPUTE_TEMPLATE(hyb2csr)
    TESTING_TEMPLATE(create_identity_permutation)
    TESTING_TEMPLATE(csrsort_buffer_size)
    TESTING_TEMPLATE(csrsort)
    TESTING_TEMPLATE(cscsort_buffer_size)
    TESTING_TEMPLATE(cscsort)
    TESTING_TEMPLATE(coosort_buffer_size)
    TESTING_TEMPLATE(coosort_by_row)
    TESTING_TEMPLATE(coosort_by_column)
    TESTING_COMPUTE_TEMPLATE(bsr2csr)
    TESTING_COMPUTE_TEMPLATE(gebsr2csr)
    TESTING_COMPUTE_TEMPLATE(gebsr2gebsr_buffer_size)
    TESTING_TEMPLATE(gebsr2gebsr_nnz)
    TESTING_COMPUTE_TEMPLATE(gebsr2gebsr)
    TESTING_TEMPLATE(inverse_permutation)

    /*
    * ===========================================================================
    *    generic SPARSE
    * ===========================================================================
    */

    TESTING_TEMPLATE(axpby)
    TESTING_TEMPLATE(gather)
    TESTING_TEMPLATE(scatter)
    TESTING_TEMPLATE(rot)
    TESTING_TEMPLATE(sparse_to_dense)
    TESTING_TEMPLATE(dense_to_sparse)
    TESTING_TEMPLATE(spvv)
    TESTING_TEMPLATE(spmv)
    TESTING_TEMPLATE(v2_spmv)
    TESTING_TEMPLATE(spsv)
    TESTING_TEMPLATE(spsm)
    TESTING_TEMPLATE(spmm)
    TESTING_TEMPLATE(spgemm)
    TESTING_TEMPLATE(spgeam)
    TESTING_TEMPLATE(sddmm)
    TESTING_TEMPLATE(sddmm_buffer_size)
    TESTING_TEMPLATE(sddmm_preprocess)

    /*
    * ===========================================================================
    *    reordering SPARSE
    * ===========================================================================
    */
    TESTING_COMPUTE_TEMPLATE(csrcolor)
}

#endif
