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
    hipsparseXcsric02_zeroPivot(hipsparseHandle_t handle, csric02Info_t info, int* position)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseXcsric02_zeroPivot((cusparseHandle_t)handle, (csric02Info_t)info, position));
}
#endif

#if CUDART_VERSION < 13000
hipsparseStatus_t hipsparseScsric02_bufferSize(hipsparseHandle_t         handle,
                                               int                       m,
                                               int                       nnz,
                                               const hipsparseMatDescr_t descrA,
                                               float*                    csrSortedValA,
                                               const int*                csrSortedRowPtrA,
                                               const int*                csrSortedColIndA,
                                               csric02Info_t             info,
                                               int*                      pBufferSizeInBytes)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseScsric02_bufferSize((cusparseHandle_t)handle,
                                    m,
                                    nnz,
                                    (cusparseMatDescr_t)descrA,
                                    csrSortedValA,
                                    csrSortedRowPtrA,
                                    csrSortedColIndA,
                                    (csric02Info_t)info,
                                    pBufferSizeInBytes));
}

hipsparseStatus_t hipsparseDcsric02_bufferSize(hipsparseHandle_t         handle,
                                               int                       m,
                                               int                       nnz,
                                               const hipsparseMatDescr_t descrA,
                                               double*                   csrSortedValA,
                                               const int*                csrSortedRowPtrA,
                                               const int*                csrSortedColIndA,
                                               csric02Info_t             info,
                                               int*                      pBufferSizeInBytes)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseDcsric02_bufferSize((cusparseHandle_t)handle,
                                    m,
                                    nnz,
                                    (cusparseMatDescr_t)descrA,
                                    csrSortedValA,
                                    csrSortedRowPtrA,
                                    csrSortedColIndA,
                                    (csric02Info_t)info,
                                    pBufferSizeInBytes));
}

hipsparseStatus_t hipsparseCcsric02_bufferSize(hipsparseHandle_t         handle,
                                               int                       m,
                                               int                       nnz,
                                               const hipsparseMatDescr_t descrA,
                                               hipComplex*               csrSortedValA,
                                               const int*                csrSortedRowPtrA,
                                               const int*                csrSortedColIndA,
                                               csric02Info_t             info,
                                               int*                      pBufferSizeInBytes)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseCcsric02_bufferSize((cusparseHandle_t)handle,
                                    m,
                                    nnz,
                                    (cusparseMatDescr_t)descrA,
                                    (cuComplex*)csrSortedValA,
                                    csrSortedRowPtrA,
                                    csrSortedColIndA,
                                    (csric02Info_t)info,
                                    pBufferSizeInBytes));
}

hipsparseStatus_t hipsparseZcsric02_bufferSize(hipsparseHandle_t         handle,
                                               int                       m,
                                               int                       nnz,
                                               const hipsparseMatDescr_t descrA,
                                               hipDoubleComplex*         csrSortedValA,
                                               const int*                csrSortedRowPtrA,
                                               const int*                csrSortedColIndA,
                                               csric02Info_t             info,
                                               int*                      pBufferSizeInBytes)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseZcsric02_bufferSize((cusparseHandle_t)handle,
                                    m,
                                    nnz,
                                    (cusparseMatDescr_t)descrA,
                                    (cuDoubleComplex*)csrSortedValA,
                                    csrSortedRowPtrA,
                                    csrSortedColIndA,
                                    (csric02Info_t)info,
                                    pBufferSizeInBytes));
}
#endif

hipsparseStatus_t hipsparseScsric02_bufferSizeExt(hipsparseHandle_t         handle,
                                                  int                       m,
                                                  int                       nnz,
                                                  const hipsparseMatDescr_t descrA,
                                                  float*                    csrSortedValA,
                                                  const int*                csrSortedRowPtrA,
                                                  const int*                csrSortedColIndA,
                                                  csric02Info_t             info,
                                                  size_t*                   pBufferSizeInBytes)
{
    return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

hipsparseStatus_t hipsparseDcsric02_bufferSizeExt(hipsparseHandle_t         handle,
                                                  int                       m,
                                                  int                       nnz,
                                                  const hipsparseMatDescr_t descrA,
                                                  double*                   csrSortedValA,
                                                  const int*                csrSortedRowPtrA,
                                                  const int*                csrSortedColIndA,
                                                  csric02Info_t             info,
                                                  size_t*                   pBufferSizeInBytes)
{
    return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

hipsparseStatus_t hipsparseCcsric02_bufferSizeExt(hipsparseHandle_t         handle,
                                                  int                       m,
                                                  int                       nnz,
                                                  const hipsparseMatDescr_t descrA,
                                                  hipComplex*               csrSortedValA,
                                                  const int*                csrSortedRowPtrA,
                                                  const int*                csrSortedColIndA,
                                                  csric02Info_t             info,
                                                  size_t*                   pBufferSizeInBytes)
{
    return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

hipsparseStatus_t hipsparseZcsric02_bufferSizeExt(hipsparseHandle_t         handle,
                                                  int                       m,
                                                  int                       nnz,
                                                  const hipsparseMatDescr_t descrA,
                                                  hipDoubleComplex*         csrSortedValA,
                                                  const int*                csrSortedRowPtrA,
                                                  const int*                csrSortedColIndA,
                                                  csric02Info_t             info,
                                                  size_t*                   pBufferSizeInBytes)
{
    return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

#if CUDART_VERSION < 13000
hipsparseStatus_t hipsparseScsric02_analysis(hipsparseHandle_t         handle,
                                             int                       m,
                                             int                       nnz,
                                             const hipsparseMatDescr_t descrA,
                                             const float*              csrSortedValA,
                                             const int*                csrSortedRowPtrA,
                                             const int*                csrSortedColIndA,
                                             csric02Info_t             info,
                                             hipsparseSolvePolicy_t    policy,
                                             void*                     pBuffer)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseScsric02_analysis((cusparseHandle_t)handle,
                                  m,
                                  nnz,
                                  (cusparseMatDescr_t)descrA,
                                  csrSortedValA,
                                  csrSortedRowPtrA,
                                  csrSortedColIndA,
                                  (csric02Info_t)info,
                                  hipsparse::hipPolicyToCudaPolicy(policy),
                                  pBuffer));
}

hipsparseStatus_t hipsparseDcsric02_analysis(hipsparseHandle_t         handle,
                                             int                       m,
                                             int                       nnz,
                                             const hipsparseMatDescr_t descrA,
                                             const double*             csrSortedValA,
                                             const int*                csrSortedRowPtrA,
                                             const int*                csrSortedColIndA,
                                             csric02Info_t             info,
                                             hipsparseSolvePolicy_t    policy,
                                             void*                     pBuffer)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseDcsric02_analysis((cusparseHandle_t)handle,
                                  m,
                                  nnz,
                                  (cusparseMatDescr_t)descrA,
                                  csrSortedValA,
                                  csrSortedRowPtrA,
                                  csrSortedColIndA,
                                  (csric02Info_t)info,
                                  hipsparse::hipPolicyToCudaPolicy(policy),
                                  pBuffer));
}

hipsparseStatus_t hipsparseCcsric02_analysis(hipsparseHandle_t         handle,
                                             int                       m,
                                             int                       nnz,
                                             const hipsparseMatDescr_t descrA,
                                             const hipComplex*         csrSortedValA,
                                             const int*                csrSortedRowPtrA,
                                             const int*                csrSortedColIndA,
                                             csric02Info_t             info,
                                             hipsparseSolvePolicy_t    policy,
                                             void*                     pBuffer)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseCcsric02_analysis((cusparseHandle_t)handle,
                                  m,
                                  nnz,
                                  (cusparseMatDescr_t)descrA,
                                  (const cuComplex*)csrSortedValA,
                                  csrSortedRowPtrA,
                                  csrSortedColIndA,
                                  (csric02Info_t)info,
                                  hipsparse::hipPolicyToCudaPolicy(policy),
                                  pBuffer));
}

hipsparseStatus_t hipsparseZcsric02_analysis(hipsparseHandle_t         handle,
                                             int                       m,
                                             int                       nnz,
                                             const hipsparseMatDescr_t descrA,
                                             const hipDoubleComplex*   csrSortedValA,
                                             const int*                csrSortedRowPtrA,
                                             const int*                csrSortedColIndA,
                                             csric02Info_t             info,
                                             hipsparseSolvePolicy_t    policy,
                                             void*                     pBuffer)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseZcsric02_analysis((cusparseHandle_t)handle,
                                  m,
                                  nnz,
                                  (cusparseMatDescr_t)descrA,
                                  (const cuDoubleComplex*)csrSortedValA,
                                  csrSortedRowPtrA,
                                  csrSortedColIndA,
                                  (csric02Info_t)info,
                                  hipsparse::hipPolicyToCudaPolicy(policy),
                                  pBuffer));
}
#endif

#if CUDART_VERSION < 13000
hipsparseStatus_t hipsparseScsric02(hipsparseHandle_t         handle,
                                    int                       m,
                                    int                       nnz,
                                    const hipsparseMatDescr_t descrA,
                                    float*                    csrSortedValA_valM,
                                    /* matrix A values are updated inplace
                                        to be the preconditioner M values */
                                    const int*             csrSortedRowPtrA,
                                    const int*             csrSortedColIndA,
                                    csric02Info_t          info,
                                    hipsparseSolvePolicy_t policy,
                                    void*                  pBuffer)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseScsric02((cusparseHandle_t)handle,
                         m,
                         nnz,
                         (cusparseMatDescr_t)descrA,
                         csrSortedValA_valM,
                         csrSortedRowPtrA,
                         csrSortedColIndA,
                         (csric02Info_t)info,
                         hipsparse::hipPolicyToCudaPolicy(policy),
                         pBuffer));
}

hipsparseStatus_t hipsparseDcsric02(hipsparseHandle_t         handle,
                                    int                       m,
                                    int                       nnz,
                                    const hipsparseMatDescr_t descrA,
                                    double*                   csrSortedValA_valM,
                                    /* matrix A values are updated inplace
                                        to be the preconditioner M values */
                                    const int*             csrSortedRowPtrA,
                                    const int*             csrSortedColIndA,
                                    csric02Info_t          info,
                                    hipsparseSolvePolicy_t policy,
                                    void*                  pBuffer)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseDcsric02((cusparseHandle_t)handle,
                         m,
                         nnz,
                         (cusparseMatDescr_t)descrA,
                         csrSortedValA_valM,
                         csrSortedRowPtrA,
                         csrSortedColIndA,
                         (csric02Info_t)info,
                         hipsparse::hipPolicyToCudaPolicy(policy),
                         pBuffer));
}

hipsparseStatus_t hipsparseCcsric02(hipsparseHandle_t         handle,
                                    int                       m,
                                    int                       nnz,
                                    const hipsparseMatDescr_t descrA,
                                    hipComplex*               csrSortedValA_valM,
                                    /* matrix A values are updated inplace
                                        to be the preconditioner M values */
                                    const int*             csrSortedRowPtrA,
                                    const int*             csrSortedColIndA,
                                    csric02Info_t          info,
                                    hipsparseSolvePolicy_t policy,
                                    void*                  pBuffer)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseCcsric02((cusparseHandle_t)handle,
                         m,
                         nnz,
                         (cusparseMatDescr_t)descrA,
                         (cuComplex*)csrSortedValA_valM,
                         csrSortedRowPtrA,
                         csrSortedColIndA,
                         (csric02Info_t)info,
                         hipsparse::hipPolicyToCudaPolicy(policy),
                         pBuffer));
}

hipsparseStatus_t hipsparseZcsric02(hipsparseHandle_t         handle,
                                    int                       m,
                                    int                       nnz,
                                    const hipsparseMatDescr_t descrA,
                                    hipDoubleComplex*         csrSortedValA_valM,
                                    /* matrix A values are updated inplace
                                        to be the preconditioner M values */
                                    const int*             csrSortedRowPtrA,
                                    const int*             csrSortedColIndA,
                                    csric02Info_t          info,
                                    hipsparseSolvePolicy_t policy,
                                    void*                  pBuffer)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseZcsric02((cusparseHandle_t)handle,
                         m,
                         nnz,
                         (cusparseMatDescr_t)descrA,
                         (cuDoubleComplex*)csrSortedValA_valM,
                         csrSortedRowPtrA,
                         csrSortedColIndA,
                         (csric02Info_t)info,
                         hipsparse::hipPolicyToCudaPolicy(policy),
                         pBuffer));
}
#endif
