/* ************************************************************************
* Copyright (C) 2025 Advanced Micro Devices, Inc. All rights Reserved.
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
#include "hipsparse.h"

#include <cuda_runtime_api.h>
#include <cusparse_v2.h>
#include <hip/hip_runtime_api.h>

#include "../utility.h"

#if CUDART_VERSION < 13000
hipsparseStatus_t
    hipsparseXcsrilu02_zeroPivot(hipsparseHandle_t handle, csrilu02Info_t info, int* position)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseXcsrilu02_zeroPivot((cusparseHandle_t)handle, (csrilu02Info_t)info, position));
}
#endif

#if CUDART_VERSION < 13000
hipsparseStatus_t hipsparseScsrilu02_numericBoost(
    hipsparseHandle_t handle, csrilu02Info_t info, int enable_boost, double* tol, float* boost_val)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(cusparseScsrilu02_numericBoost(
        (cusparseHandle_t)handle, (csrilu02Info_t)info, enable_boost, tol, boost_val));
}

hipsparseStatus_t hipsparseDcsrilu02_numericBoost(
    hipsparseHandle_t handle, csrilu02Info_t info, int enable_boost, double* tol, double* boost_val)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(cusparseDcsrilu02_numericBoost(
        (cusparseHandle_t)handle, (csrilu02Info_t)info, enable_boost, tol, boost_val));
}

hipsparseStatus_t hipsparseCcsrilu02_numericBoost(hipsparseHandle_t handle,
                                                  csrilu02Info_t    info,
                                                  int               enable_boost,
                                                  double*           tol,
                                                  hipComplex*       boost_val)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(cusparseCcsrilu02_numericBoost(
        (cusparseHandle_t)handle, (csrilu02Info_t)info, enable_boost, tol, (cuComplex*)boost_val));
}

hipsparseStatus_t hipsparseZcsrilu02_numericBoost(hipsparseHandle_t handle,
                                                  csrilu02Info_t    info,
                                                  int               enable_boost,
                                                  double*           tol,
                                                  hipDoubleComplex* boost_val)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseZcsrilu02_numericBoost((cusparseHandle_t)handle,
                                       (csrilu02Info_t)info,
                                       enable_boost,
                                       tol,
                                       (cuDoubleComplex*)boost_val));
}
#endif

#if CUDART_VERSION < 13000
hipsparseStatus_t hipsparseScsrilu02_bufferSize(hipsparseHandle_t         handle,
                                                int                       m,
                                                int                       nnz,
                                                const hipsparseMatDescr_t descrA,
                                                float*                    csrSortedValA,
                                                const int*                csrSortedRowPtrA,
                                                const int*                csrSortedColIndA,
                                                csrilu02Info_t            info,
                                                int*                      pBufferSizeInBytes)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseScsrilu02_bufferSize((cusparseHandle_t)handle,
                                     m,
                                     nnz,
                                     (cusparseMatDescr_t)descrA,
                                     csrSortedValA,
                                     csrSortedRowPtrA,
                                     csrSortedColIndA,
                                     (csrilu02Info_t)info,
                                     pBufferSizeInBytes));
}

hipsparseStatus_t hipsparseDcsrilu02_bufferSize(hipsparseHandle_t         handle,
                                                int                       m,
                                                int                       nnz,
                                                const hipsparseMatDescr_t descrA,
                                                double*                   csrSortedValA,
                                                const int*                csrSortedRowPtrA,
                                                const int*                csrSortedColIndA,
                                                csrilu02Info_t            info,
                                                int*                      pBufferSizeInBytes)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseDcsrilu02_bufferSize((cusparseHandle_t)handle,
                                     m,
                                     nnz,
                                     (cusparseMatDescr_t)descrA,
                                     csrSortedValA,
                                     csrSortedRowPtrA,
                                     csrSortedColIndA,
                                     (csrilu02Info_t)info,
                                     pBufferSizeInBytes));
}

hipsparseStatus_t hipsparseCcsrilu02_bufferSize(hipsparseHandle_t         handle,
                                                int                       m,
                                                int                       nnz,
                                                const hipsparseMatDescr_t descrA,
                                                hipComplex*               csrSortedValA,
                                                const int*                csrSortedRowPtrA,
                                                const int*                csrSortedColIndA,
                                                csrilu02Info_t            info,
                                                int*                      pBufferSizeInBytes)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseCcsrilu02_bufferSize((cusparseHandle_t)handle,
                                     m,
                                     nnz,
                                     (cusparseMatDescr_t)descrA,
                                     (cuComplex*)csrSortedValA,
                                     csrSortedRowPtrA,
                                     csrSortedColIndA,
                                     (csrilu02Info_t)info,
                                     pBufferSizeInBytes));
}

hipsparseStatus_t hipsparseZcsrilu02_bufferSize(hipsparseHandle_t         handle,
                                                int                       m,
                                                int                       nnz,
                                                const hipsparseMatDescr_t descrA,
                                                hipDoubleComplex*         csrSortedValA,
                                                const int*                csrSortedRowPtrA,
                                                const int*                csrSortedColIndA,
                                                csrilu02Info_t            info,
                                                int*                      pBufferSizeInBytes)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseZcsrilu02_bufferSize((cusparseHandle_t)handle,
                                     m,
                                     nnz,
                                     (cusparseMatDescr_t)descrA,
                                     (cuDoubleComplex*)csrSortedValA,
                                     csrSortedRowPtrA,
                                     csrSortedColIndA,
                                     (csrilu02Info_t)info,
                                     pBufferSizeInBytes));
}
#endif

hipsparseStatus_t hipsparseScsrilu02_bufferSizeExt(hipsparseHandle_t         handle,
                                                   int                       m,
                                                   int                       nnz,
                                                   const hipsparseMatDescr_t descrA,
                                                   float*                    csrSortedValA,
                                                   const int*                csrSortedRowPtrA,
                                                   const int*                csrSortedColIndA,
                                                   csrilu02Info_t            info,
                                                   size_t*                   pBufferSizeInBytes)
{
    return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

hipsparseStatus_t hipsparseDcsrilu02_bufferSizeExt(hipsparseHandle_t         handle,
                                                   int                       m,
                                                   int                       nnz,
                                                   const hipsparseMatDescr_t descrA,
                                                   double*                   csrSortedValA,
                                                   const int*                csrSortedRowPtrA,
                                                   const int*                csrSortedColIndA,
                                                   csrilu02Info_t            info,
                                                   size_t*                   pBufferSizeInBytes)
{
    return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

hipsparseStatus_t hipsparseCcsrilu02_bufferSizeExt(hipsparseHandle_t         handle,
                                                   int                       m,
                                                   int                       nnz,
                                                   const hipsparseMatDescr_t descrA,
                                                   hipComplex*               csrSortedValA,
                                                   const int*                csrSortedRowPtrA,
                                                   const int*                csrSortedColIndA,
                                                   csrilu02Info_t            info,
                                                   size_t*                   pBufferSizeInBytes)
{
    return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

hipsparseStatus_t hipsparseZcsrilu02_bufferSizeExt(hipsparseHandle_t         handle,
                                                   int                       m,
                                                   int                       nnz,
                                                   const hipsparseMatDescr_t descrA,
                                                   hipDoubleComplex*         csrSortedValA,
                                                   const int*                csrSortedRowPtrA,
                                                   const int*                csrSortedColIndA,
                                                   csrilu02Info_t            info,
                                                   size_t*                   pBufferSizeInBytes)
{
    return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

#if CUDART_VERSION < 13000
hipsparseStatus_t hipsparseScsrilu02_analysis(hipsparseHandle_t         handle,
                                              int                       m,
                                              int                       nnz,
                                              const hipsparseMatDescr_t descrA,
                                              const float*              csrSortedValA,
                                              const int*                csrSortedRowPtrA,
                                              const int*                csrSortedColIndA,
                                              csrilu02Info_t            info,
                                              hipsparseSolvePolicy_t    policy,
                                              void*                     pBuffer)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseScsrilu02_analysis((cusparseHandle_t)handle,
                                   m,
                                   nnz,
                                   (cusparseMatDescr_t)descrA,
                                   csrSortedValA,
                                   csrSortedRowPtrA,
                                   csrSortedColIndA,
                                   (csrilu02Info_t)info,
                                   hipsparse::hipPolicyToCudaPolicy(policy),
                                   pBuffer));
}

hipsparseStatus_t hipsparseDcsrilu02_analysis(hipsparseHandle_t         handle,
                                              int                       m,
                                              int                       nnz,
                                              const hipsparseMatDescr_t descrA,
                                              const double*             csrSortedValA,
                                              const int*                csrSortedRowPtrA,
                                              const int*                csrSortedColIndA,
                                              csrilu02Info_t            info,
                                              hipsparseSolvePolicy_t    policy,
                                              void*                     pBuffer)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseDcsrilu02_analysis((cusparseHandle_t)handle,
                                   m,
                                   nnz,
                                   (cusparseMatDescr_t)descrA,
                                   csrSortedValA,
                                   csrSortedRowPtrA,
                                   csrSortedColIndA,
                                   (csrilu02Info_t)info,
                                   hipsparse::hipPolicyToCudaPolicy(policy),
                                   pBuffer));
}

hipsparseStatus_t hipsparseCcsrilu02_analysis(hipsparseHandle_t         handle,
                                              int                       m,
                                              int                       nnz,
                                              const hipsparseMatDescr_t descrA,
                                              const hipComplex*         csrSortedValA,
                                              const int*                csrSortedRowPtrA,
                                              const int*                csrSortedColIndA,
                                              csrilu02Info_t            info,
                                              hipsparseSolvePolicy_t    policy,
                                              void*                     pBuffer)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseCcsrilu02_analysis((cusparseHandle_t)handle,
                                   m,
                                   nnz,
                                   (cusparseMatDescr_t)descrA,
                                   (const cuComplex*)csrSortedValA,
                                   csrSortedRowPtrA,
                                   csrSortedColIndA,
                                   (csrilu02Info_t)info,
                                   hipsparse::hipPolicyToCudaPolicy(policy),
                                   pBuffer));
}

hipsparseStatus_t hipsparseZcsrilu02_analysis(hipsparseHandle_t         handle,
                                              int                       m,
                                              int                       nnz,
                                              const hipsparseMatDescr_t descrA,
                                              const hipDoubleComplex*   csrSortedValA,
                                              const int*                csrSortedRowPtrA,
                                              const int*                csrSortedColIndA,
                                              csrilu02Info_t            info,
                                              hipsparseSolvePolicy_t    policy,
                                              void*                     pBuffer)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseZcsrilu02_analysis((cusparseHandle_t)handle,
                                   m,
                                   nnz,
                                   (cusparseMatDescr_t)descrA,
                                   (const cuDoubleComplex*)csrSortedValA,
                                   csrSortedRowPtrA,
                                   csrSortedColIndA,
                                   (csrilu02Info_t)info,
                                   hipsparse::hipPolicyToCudaPolicy(policy),
                                   pBuffer));
}
#endif

#if CUDART_VERSION < 13000
hipsparseStatus_t hipsparseScsrilu02(hipsparseHandle_t         handle,
                                     int                       m,
                                     int                       nnz,
                                     const hipsparseMatDescr_t descrA,
                                     float*                    csrSortedValA_valM,
                                     /* matrix A values are updated inplace
                                        to be the preconditioner M values */
                                     const int*             csrSortedRowPtrA,
                                     const int*             csrSortedColIndA,
                                     csrilu02Info_t         info,
                                     hipsparseSolvePolicy_t policy,
                                     void*                  pBuffer)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseScsrilu02((cusparseHandle_t)handle,
                          m,
                          nnz,
                          (cusparseMatDescr_t)descrA,
                          csrSortedValA_valM,
                          csrSortedRowPtrA,
                          csrSortedColIndA,
                          (csrilu02Info_t)info,
                          hipsparse::hipPolicyToCudaPolicy(policy),
                          pBuffer));
}

hipsparseStatus_t hipsparseDcsrilu02(hipsparseHandle_t         handle,
                                     int                       m,
                                     int                       nnz,
                                     const hipsparseMatDescr_t descrA,
                                     double*                   csrSortedValA_valM,
                                     /* matrix A values are updated inplace
                                        to be the preconditioner M values */
                                     const int*             csrSortedRowPtrA,
                                     const int*             csrSortedColIndA,
                                     csrilu02Info_t         info,
                                     hipsparseSolvePolicy_t policy,
                                     void*                  pBuffer)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseDcsrilu02((cusparseHandle_t)handle,
                          m,
                          nnz,
                          (cusparseMatDescr_t)descrA,
                          csrSortedValA_valM,
                          csrSortedRowPtrA,
                          csrSortedColIndA,
                          (csrilu02Info_t)info,
                          hipsparse::hipPolicyToCudaPolicy(policy),
                          pBuffer));
}

hipsparseStatus_t hipsparseCcsrilu02(hipsparseHandle_t         handle,
                                     int                       m,
                                     int                       nnz,
                                     const hipsparseMatDescr_t descrA,
                                     hipComplex*               csrSortedValA_valM,
                                     /* matrix A values are updated inplace
                                        to be the preconditioner M values */
                                     const int*             csrSortedRowPtrA,
                                     const int*             csrSortedColIndA,
                                     csrilu02Info_t         info,
                                     hipsparseSolvePolicy_t policy,
                                     void*                  pBuffer)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseCcsrilu02((cusparseHandle_t)handle,
                          m,
                          nnz,
                          (cusparseMatDescr_t)descrA,
                          (cuComplex*)csrSortedValA_valM,
                          csrSortedRowPtrA,
                          csrSortedColIndA,
                          (csrilu02Info_t)info,
                          hipsparse::hipPolicyToCudaPolicy(policy),
                          pBuffer));
}

hipsparseStatus_t hipsparseZcsrilu02(hipsparseHandle_t         handle,
                                     int                       m,
                                     int                       nnz,
                                     const hipsparseMatDescr_t descrA,
                                     hipDoubleComplex*         csrSortedValA_valM,
                                     /* matrix A values are updated inplace
                                        to be the preconditioner M values */
                                     const int*             csrSortedRowPtrA,
                                     const int*             csrSortedColIndA,
                                     csrilu02Info_t         info,
                                     hipsparseSolvePolicy_t policy,
                                     void*                  pBuffer)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseZcsrilu02((cusparseHandle_t)handle,
                          m,
                          nnz,
                          (cusparseMatDescr_t)descrA,
                          (cuDoubleComplex*)csrSortedValA_valM,
                          csrSortedRowPtrA,
                          csrSortedColIndA,
                          (csrilu02Info_t)info,
                          hipsparse::hipPolicyToCudaPolicy(policy),
                          pBuffer));
}
#endif
