/* ************************************************************************
 * Copyright (C) 2023-2024 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
 * ies of the Software, and to permit persons to whom the Software is furnished
 * to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
 * PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
 * CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 *
 * ************************************************************************ */

#pragma once

#include "hipsolver.h"
#include "hipsolver.hpp"
#include "hipsparse/hipsparse.h"

// Most functions within this file exist to provide a consistent interface for our templated tests.
// Function overloading is used to select between the float, double, rocblas_float_complex
// and rocblas_double_complex variants, and to distinguish the batched case (T**) from the normal
// and strided_batched cases (T*).
//
// The normal and strided_batched cases are distinguished from each other by passing a boolean
// parameter, STRIDED. Variants such as the blocked and unblocked versions of algorithms, may be
// provided in similar ways.

/* ============================================================================================
 */
/*! \brief  local matrix descriptor which is automatically created and destroyed  */
class hipsparse_local_mat_descr
{
    hipsparseMatDescr_t m_info;

public:
    hipsparse_local_mat_descr()
    {
        if(hipsparseCreateMatDescr(&m_info) != HIPSPARSE_STATUS_SUCCESS)
            throw std::runtime_error("ERROR: Could not create hipsparseMatDescr_t");
    }
    ~hipsparse_local_mat_descr()
    {
        hipsparseDestroyMatDescr(m_info);
    }

    hipsparse_local_mat_descr(const hipsparse_local_mat_descr&) = delete;

    hipsparse_local_mat_descr(hipsparse_local_mat_descr&&) = delete;

    hipsparse_local_mat_descr& operator=(const hipsparse_local_mat_descr&) = delete;

    hipsparse_local_mat_descr& operator=(hipsparse_local_mat_descr&&) = delete;

    // Allow hipsparse_local_mat_descr to be used anywhere hipsparseMatDescr_t is expected
    operator hipsparseMatDescr_t&()
    {
        return m_info;
    }
    operator const hipsparseMatDescr_t&() const
    {
        return m_info;
    }
};

/******************** CSRLSVCHOL ********************/
// normal and strided_batched
inline hipsolverStatus_t hipsolver_csrlsvchol(bool                      HOST,
                                              hipsolverSpHandle_t       handle,
                                              int                       n,
                                              int                       nnz,
                                              const hipsparseMatDescr_t descrA,
                                              const float*              csrVal,
                                              const int*                csrRowPtr,
                                              const int*                csrColInd,
                                              const float*              b,
                                              float                     tol,
                                              int                       reorder,
                                              float*                    x,
                                              int*                      singularity)
{
    if(!HOST)
        return hipsolverSpScsrlsvchol(
            handle, n, nnz, descrA, csrVal, csrRowPtr, csrColInd, b, tol, reorder, x, singularity);
    else
        return hipsolverSpScsrlsvcholHost(
            handle, n, nnz, descrA, csrVal, csrRowPtr, csrColInd, b, tol, reorder, x, singularity);
}

inline hipsolverStatus_t hipsolver_csrlsvchol(bool                      HOST,
                                              hipsolverSpHandle_t       handle,
                                              int                       n,
                                              int                       nnz,
                                              const hipsparseMatDescr_t descrA,
                                              const double*             csrVal,
                                              const int*                csrRowPtr,
                                              const int*                csrColInd,
                                              const double*             b,
                                              double                    tol,
                                              int                       reorder,
                                              double*                   x,
                                              int*                      singularity)
{
    if(!HOST)
        return hipsolverSpDcsrlsvchol(
            handle, n, nnz, descrA, csrVal, csrRowPtr, csrColInd, b, tol, reorder, x, singularity);
    else
        return hipsolverSpDcsrlsvcholHost(
            handle, n, nnz, descrA, csrVal, csrRowPtr, csrColInd, b, tol, reorder, x, singularity);
}

/*inline hipsolverStatus_t hipsolver_csrlsvchol(bool                      HOST,
                                              hipsolverSpHandle_t       handle,
                                              int                       n,
                                              int                       nnz,
                                              const hipsparseMatDescr_t descrA,
                                              const hipsolverComplex*   csrVal,
                                              const int*                csrRowPtr,
                                              const int*                csrColInd,
                                              const hipsolverComplex*   b,
                                              float                     tol,
                                              int                       reorder,
                                              hipsolverComplex*         x,
                                              int*                      singularity)
{
    if(!HOST)
        return hipsolverSpCcsrlsvchol(handle,
                                      n,
                                      nnz,
                                      descrA,
                                      (hipFloatComplex*)csrVal,
                                      csrRowPtr,
                                      csrColInd,
                                      (hipFloatComplex*)b,
                                      tol,
                                      reorder,
                                      (hipFloatComplex*)x,
                                      singularity);
    else
        return hipsolverSpCcsrlsvcholHost(handle,
                                          n,
                                          nnz,
                                          descrA,
                                          (hipFloatComplex*)csrVal,
                                          csrRowPtr,
                                          csrColInd,
                                          (hipFloatComplex*)b,
                                          tol,
                                          reorder,
                                          (hipFloatComplex*)x,
                                          singularity);
}

inline hipsolverStatus_t hipsolver_csrlsvchol(bool                          HOST,
                                              hipsolverSpHandle_t           handle,
                                              int                           n,
                                              int                           nnz,
                                              const hipsparseMatDescr_t     descrA,
                                              const hipsolverDoubleComplex* csrVal,
                                              const int*                    csrRowPtr,
                                              const int*                    csrColInd,
                                              const hipsolverDoubleComplex* b,
                                              double                        tol,
                                              int                           reorder,
                                              hipsolverDoubleComplex*       x,
                                              int*                          singularity)
{
    if(!HOST)
        return hipsolverSpZcsrlsvchol(handle,
                                      n,
                                      nnz,
                                      descrA,
                                      (hipDoubleComplex*)csrVal,
                                      csrRowPtr,
                                      csrColInd,
                                      (hipDoubleComplex*)b,
                                      tol,
                                      reorder,
                                      (hipDoubleComplex*)x,
                                      singularity);
    else
        return hipsolverSpZcsrlsvcholHost(handle,
                                          n,
                                          nnz,
                                          descrA,
                                          (hipDoubleComplex*)csrVal,
                                          csrRowPtr,
                                          csrColInd,
                                          (hipDoubleComplex*)b,
                                          tol,
                                          reorder,
                                          (hipDoubleComplex*)x,
                                          singularity);
}*/
/********************************************************/

/******************** CSRLSVQR ********************/
// normal and strided_batched
inline hipsolverStatus_t hipsolver_csrlsvqr(bool                      HOST,
                                            hipsolverSpHandle_t       handle,
                                            int                       n,
                                            int                       nnz,
                                            const hipsparseMatDescr_t descrA,
                                            const float*              csrVal,
                                            const int*                csrRowPtr,
                                            const int*                csrColInd,
                                            const float*              b,
                                            double                    tol,
                                            int                       reorder,
                                            float*                    x,
                                            int*                      singularity)
{
    if(!HOST)
        return hipsolverSpScsrlsvqr(
            handle, n, nnz, descrA, csrVal, csrRowPtr, csrColInd, b, tol, reorder, x, singularity);
    else
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
}

inline hipsolverStatus_t hipsolver_csrlsvqr(bool                      HOST,
                                            hipsolverSpHandle_t       handle,
                                            int                       n,
                                            int                       nnz,
                                            const hipsparseMatDescr_t descrA,
                                            const double*             csrVal,
                                            const int*                csrRowPtr,
                                            const int*                csrColInd,
                                            const double*             b,
                                            double                    tol,
                                            int                       reorder,
                                            double*                   x,
                                            int*                      singularity)
{
    if(!HOST)
        return hipsolverSpDcsrlsvqr(
            handle, n, nnz, descrA, csrVal, csrRowPtr, csrColInd, b, tol, reorder, x, singularity);
    else
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
}

/*inline hipsolverStatus_t hipsolver_csrlsvqr(bool                      HOST,
                                            hipsolverSpHandle_t       handle,
                                            int                       n,
                                            int                       nnz,
                                            const hipsparseMatDescr_t descrA,
                                            const hipsolverComplex*   csrVal,
                                            const int*                csrRowPtr,
                                            const int*                csrColInd,
                                            const hipsolverComplex*   b,
                                            double                    tol,
                                            int                       reorder,
                                            hipsolverComplex*         x,
                                            int*                      singularity)
{
    if(!HOST)
        return hipsolverSpCcsrlsvqr(handle,
                                    n,
                                    nnz,
                                    descrA,
                                    (hipFloatComplex*)csrVal,
                                    csrRowPtr,
                                    csrColInd,
                                    (hipFloatComplex*)b,
                                    tol,
                                    reorder,
                                    (hipFloatComplex*)x,
                                    singularity);
    else
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
}

inline hipsolverStatus_t hipsolver_csrlsvqr(bool                          HOST,
                                            hipsolverSpHandle_t           handle,
                                            int                           n,
                                            int                           nnz,
                                            const hipsparseMatDescr_t     descrA,
                                            const hipsolverDoubleComplex* csrVal,
                                            const int*                    csrRowPtr,
                                            const int*                    csrColInd,
                                            const hipsolverDoubleComplex* b,
                                            double                        tol,
                                            int                           reorder,
                                            hipsolverDoubleComplex*       x,
                                            int*                          singularity)
{
    if(!HOST)
        return hipsolverSpZcsrlsvqr(handle,
                                    n,
                                    nnz,
                                    descrA,
                                    (hipDoubleComplex*)csrVal,
                                    csrRowPtr,
                                    csrColInd,
                                    (hipDoubleComplex*)b,
                                    tol,
                                    reorder,
                                    (hipDoubleComplex*)x,
                                    singularity);
    else
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
}*/
/********************************************************/
