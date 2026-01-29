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

/*! \file
 *  \brief hipsparse_graph.hpp exposes C++ templated hipSPARSE routine wrappers
 *  that determine if the routine is ran on a hipgraph context
 */
#pragma once

#include <utility>
#include <vector>

#include "hipsparse.hpp"
#include "hipsparse_local_handle.hpp"

#if HIP_VERSION >= 50300000
#define GRAPH_TEST 1
#else
#define GRAPH_TEST 0
#endif

#if GRAPH_TEST
#define BEGIN_GRAPH_CAPTURE() handle.hipsparseStreamBeginCapture()
#define END_GRAPH_CAPTURE() handle.hipsparseStreamEndCapture()
#else
#define BEGIN_GRAPH_CAPTURE()
#define END_GRAPH_CAPTURE()
#endif

#define TESTING_TEMPLATE(NAME_)                                                  \
    template <typename... P>                                                     \
    hipsparseStatus_t hipsparse##NAME_(hipsparseLocalHandle_t& handle, P&&... p) \
    {                                                                            \
        hipsparseStatus_t status;                                                \
        BEGIN_GRAPH_CAPTURE();                                                   \
                                                                                 \
        status = ::hipsparse##NAME_(handle, std::forward<P>(p)...);              \
                                                                                 \
        END_GRAPH_CAPTURE();                                                     \
                                                                                 \
        return status;                                                           \
    };

#define TESTING_COMPUTE_TEMPLATE(NAME_)                                           \
    template <typename T, typename... P>                                          \
    hipsparseStatus_t hipsparseX##NAME_(hipsparseLocalHandle_t& handle, P&&... p) \
    {                                                                             \
        hipsparseStatus_t status;                                                 \
        BEGIN_GRAPH_CAPTURE();                                                    \
                                                                                  \
        status = hipsparse::hipsparseX##NAME_<T>(handle, std::forward<P>(p)...);  \
                                                                                  \
        END_GRAPH_CAPTURE();                                                      \
                                                                                  \
        return status;                                                            \
    };

namespace testing
{
    /*
    * ===========================================================================
    *    level 1 SPARSE
    * ===========================================================================
    */
#if(!defined(CUDART_VERSION) || CUDART_VERSION < 12000)
    TESTING_COMPUTE_TEMPLATE(axpyi)
    TESTING_COMPUTE_TEMPLATE(gthr)
    TESTING_COMPUTE_TEMPLATE(gthrz)
    TESTING_COMPUTE_TEMPLATE(roti)
    TESTING_COMPUTE_TEMPLATE(sctr)
#endif

    /*
    * ===========================================================================
    *    level 2 SPARSE
    * ===========================================================================
    */
    TESTING_COMPUTE_TEMPLATE(bsrmv)
    TESTING_COMPUTE_TEMPLATE(bsrsv2_bufferSizeExt)
    TESTING_COMPUTE_TEMPLATE(csrsv2_bufferSizeExt)
#if(!defined(CUDART_VERSION) || CUDART_VERSION < 13000)
    TESTING_COMPUTE_TEMPLATE(bsrsv2_bufferSize)
    TESTING_COMPUTE_TEMPLATE(bsrsv2_solve)
    TESTING_COMPUTE_TEMPLATE(bsrxmv)
#endif
#if(!defined(CUDART_VERSION) || CUDART_VERSION < 12000)
    TESTING_COMPUTE_TEMPLATE(csrsv2_bufferSize)
    TESTING_COMPUTE_TEMPLATE(csrsv2_solve)
    TESTING_COMPUTE_TEMPLATE(gemvi_bufferSize)
    TESTING_COMPUTE_TEMPLATE(gemvi)
#endif
#if(!defined(CUDART_VERSION) || CUDART_VERSION < 11000)
    TESTING_COMPUTE_TEMPLATE(csrmv)
    TESTING_COMPUTE_TEMPLATE(hybmv)
#endif

    /*
    * ===========================================================================
    *    level 3 SPARSE
    * ===========================================================================
    */
    TESTING_COMPUTE_TEMPLATE(bsrmm)
#if(!defined(CUDART_VERSION) || CUDART_VERSION < 13000)
    TESTING_COMPUTE_TEMPLATE(bsrsm2_bufferSize)
    TESTING_COMPUTE_TEMPLATE(bsrsm2_solve)
#endif
#if(!defined(CUDART_VERSION) || CUDART_VERSION < 12000)
    TESTING_COMPUTE_TEMPLATE(csrsm2_bufferSizeExt)
    TESTING_COMPUTE_TEMPLATE(csrsm2_solve)
    TESTING_COMPUTE_TEMPLATE(gemmi)
#endif
#if(!defined(CUDART_VERSION) || CUDART_VERSION < 11000)
    // TESTING_COMPUTE_TEMPLATE(csrmm)
    TESTING_COMPUTE_TEMPLATE(csrmm2)
#endif

    /*
    * ===========================================================================
    *    generic SPARSE
    * ===========================================================================
    */

    TESTING_TEMPLATE(Axpby)
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 11000)
    TESTING_TEMPLATE(Gather)
    TESTING_TEMPLATE(Scatter)
#endif
#if(!defined(CUDART_VERSION) || (CUDART_VERSION >= 11000 && CUDART_VERSION < 13000))
    TESTING_TEMPLATE(Rot)
#endif
#if(!defined(CUDART_VERSION) || CUDART_VERSION > 10010 \
    || (CUDART_VERSION == 10010 && CUDART_10_1_UPDATE_VERSION == 1))
    TESTING_TEMPLATE(SpVV_bufferSize)
    TESTING_TEMPLATE(SpVV)
#endif
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 11000)
    TESTING_TEMPLATE(SpMM_bufferSize)
    TESTING_TEMPLATE(SpMM_preprocess)
    TESTING_TEMPLATE(SpMM)
#endif
#if(!defined(CUDART_VERSION))
    TESTING_TEMPLATE(SDDMM)
    TESTING_TEMPLATE(SDDMM_bufferSize)
    TESTING_TEMPLATE(SDDMM_preprocess)
#endif

    /*
    * ===========================================================================
    *    conversion SPARSE
    * ===========================================================================
    */

    // coo2csr and csr2coo are not templated on T
    TESTING_TEMPLATE(Xcoo2csr)
    TESTING_TEMPLATE(Xcsr2coo)

#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 10010)
    // csr2csc_ex2 available from CUDA 10.1
    TESTING_TEMPLATE(Csr2cscEx2_bufferSize)
    TESTING_TEMPLATE(Csr2cscEx2)
#endif

    // bsr2csr
    TESTING_COMPUTE_TEMPLATE(bsr2csr)

    // csr2gebsr and gebsr2csr
    TESTING_TEMPLATE(Xcsr2gebsrNnz)
    TESTING_COMPUTE_TEMPLATE(csr2gebsr)
    TESTING_COMPUTE_TEMPLATE(gebsr2csr)

    // gebsr2gebsr and gebsr2gebsc
    TESTING_TEMPLATE(Xgebsr2gebsrNnz)
    TESTING_COMPUTE_TEMPLATE(gebsr2gebsr)
    TESTING_COMPUTE_TEMPLATE(gebsr2gebsc)

    // csr2csr_compress
#if(!defined(CUDART_VERSION) || CUDART_VERSION < 13000)
    TESTING_COMPUTE_TEMPLATE(nnz_compress)
#endif
    TESTING_COMPUTE_TEMPLATE(csr2csr_compress)

    // dense to sparse
#if(!defined(CUDART_VERSION) || CUDART_VERSION < 12000)
    TESTING_COMPUTE_TEMPLATE(dense2csr)
    TESTING_COMPUTE_TEMPLATE(dense2csc)
    TESTING_COMPUTE_TEMPLATE(csr2dense)
    TESTING_COMPUTE_TEMPLATE(csc2dense)
#endif

    // prune routines
#if(!defined(CUDART_VERSION) || CUDART_VERSION < 13000)
    TESTING_COMPUTE_TEMPLATE(pruneDense2csr_bufferSize)
    TESTING_COMPUTE_TEMPLATE(pruneDense2csrNnz)
    TESTING_COMPUTE_TEMPLATE(pruneDense2csr)
    TESTING_COMPUTE_TEMPLATE(pruneCsr2csr_bufferSize)
    TESTING_COMPUTE_TEMPLATE(pruneCsr2csrNnz)
    TESTING_COMPUTE_TEMPLATE(pruneCsr2csr)
    TESTING_COMPUTE_TEMPLATE(pruneDense2csr_bufferSizeExt)
    TESTING_COMPUTE_TEMPLATE(pruneCsr2csr_bufferSizeExt)
    TESTING_COMPUTE_TEMPLATE(pruneDense2csrByPercentage_bufferSize)
    TESTING_COMPUTE_TEMPLATE(pruneDense2csrByPercentage_bufferSizeExt)
    TESTING_COMPUTE_TEMPLATE(pruneDense2csrNnzByPercentage)
    TESTING_COMPUTE_TEMPLATE(pruneDense2csrByPercentage)
    TESTING_COMPUTE_TEMPLATE(pruneCsr2csrByPercentage_bufferSize)
    TESTING_COMPUTE_TEMPLATE(pruneCsr2csrByPercentage_bufferSizeExt)
    TESTING_COMPUTE_TEMPLATE(pruneCsr2csrNnzByPercentage)
    TESTING_COMPUTE_TEMPLATE(pruneCsr2csrByPercentage)
#endif

    // nnz routines
    TESTING_COMPUTE_TEMPLATE(nnz)
}
