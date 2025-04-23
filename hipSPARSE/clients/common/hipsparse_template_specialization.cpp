/* ************************************************************************
* Copyright (C) 2018-2021 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "hipsparse.hpp"

#include <hipsparse.h>

namespace hipsparse
{
#if(!defined(CUDART_VERSION) || CUDART_VERSION < 12000)
    template <>
    hipsparseStatus_t hipsparseXaxpyi(hipsparseHandle_t    handle,
                                      int                  nnz,
                                      const float*         alpha,
                                      const float*         xVal,
                                      const int*           xInd,
                                      float*               y,
                                      hipsparseIndexBase_t idxBase)
    {
        return hipsparseSaxpyi(handle, nnz, alpha, xVal, xInd, y, idxBase);
    }

    template <>
    hipsparseStatus_t hipsparseXaxpyi(hipsparseHandle_t    handle,
                                      int                  nnz,
                                      const double*        alpha,
                                      const double*        xVal,
                                      const int*           xInd,
                                      double*              y,
                                      hipsparseIndexBase_t idxBase)
    {
        return hipsparseDaxpyi(handle, nnz, alpha, xVal, xInd, y, idxBase);
    }

    template <>
    hipsparseStatus_t hipsparseXaxpyi(hipsparseHandle_t    handle,
                                      int                  nnz,
                                      const hipComplex*    alpha,
                                      const hipComplex*    xVal,
                                      const int*           xInd,
                                      hipComplex*          y,
                                      hipsparseIndexBase_t idxBase)
    {
        return hipsparseCaxpyi(handle, nnz, alpha, xVal, xInd, y, idxBase);
    }

    template <>
    hipsparseStatus_t hipsparseXaxpyi(hipsparseHandle_t       handle,
                                      int                     nnz,
                                      const hipDoubleComplex* alpha,
                                      const hipDoubleComplex* xVal,
                                      const int*              xInd,
                                      hipDoubleComplex*       y,
                                      hipsparseIndexBase_t    idxBase)
    {
        return hipsparseZaxpyi(handle, nnz, alpha, xVal, xInd, y, idxBase);
    }
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 11000)
    template <>
    hipsparseStatus_t hipsparseXdoti(hipsparseHandle_t    handle,
                                     int                  nnz,
                                     const float*         xVal,
                                     const int*           xInd,
                                     const float*         y,
                                     float*               result,
                                     hipsparseIndexBase_t idxBase)
    {
        return hipsparseSdoti(handle, nnz, xVal, xInd, y, result, idxBase);
    }

    template <>
    hipsparseStatus_t hipsparseXdoti(hipsparseHandle_t    handle,
                                     int                  nnz,
                                     const double*        xVal,
                                     const int*           xInd,
                                     const double*        y,
                                     double*              result,
                                     hipsparseIndexBase_t idxBase)
    {
        return hipsparseDdoti(handle, nnz, xVal, xInd, y, result, idxBase);
    }

    template <>
    hipsparseStatus_t hipsparseXdoti(hipsparseHandle_t    handle,
                                     int                  nnz,
                                     const hipComplex*    xVal,
                                     const int*           xInd,
                                     const hipComplex*    y,
                                     hipComplex*          result,
                                     hipsparseIndexBase_t idxBase)
    {
        return hipsparseCdoti(handle, nnz, xVal, xInd, y, result, idxBase);
    }

    template <>
    hipsparseStatus_t hipsparseXdoti(hipsparseHandle_t       handle,
                                     int                     nnz,
                                     const hipDoubleComplex* xVal,
                                     const int*              xInd,
                                     const hipDoubleComplex* y,
                                     hipDoubleComplex*       result,
                                     hipsparseIndexBase_t    idxBase)
    {
        return hipsparseZdoti(handle, nnz, xVal, xInd, y, result, idxBase);
    }
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 11000)
    template <>
    hipsparseStatus_t hipsparseXdotci(hipsparseHandle_t    handle,
                                      int                  nnz,
                                      const hipComplex*    xVal,
                                      const int*           xInd,
                                      const hipComplex*    y,
                                      hipComplex*          result,
                                      hipsparseIndexBase_t idxBase)
    {
        return hipsparseCdotci(handle, nnz, xVal, xInd, y, result, idxBase);
    }

    template <>
    hipsparseStatus_t hipsparseXdotci(hipsparseHandle_t       handle,
                                      int                     nnz,
                                      const hipDoubleComplex* xVal,
                                      const int*              xInd,
                                      const hipDoubleComplex* y,
                                      hipDoubleComplex*       result,
                                      hipsparseIndexBase_t    idxBase)
    {
        return hipsparseZdotci(handle, nnz, xVal, xInd, y, result, idxBase);
    }
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 12000)
    template <>
    hipsparseStatus_t hipsparseXgthr(hipsparseHandle_t    handle,
                                     int                  nnz,
                                     const float*         y,
                                     float*               xVal,
                                     const int*           xInd,
                                     hipsparseIndexBase_t idxBase)
    {
        return hipsparseSgthr(handle, nnz, y, xVal, xInd, idxBase);
    }

    template <>
    hipsparseStatus_t hipsparseXgthr(hipsparseHandle_t    handle,
                                     int                  nnz,
                                     const double*        y,
                                     double*              xVal,
                                     const int*           xInd,
                                     hipsparseIndexBase_t idxBase)
    {
        return hipsparseDgthr(handle, nnz, y, xVal, xInd, idxBase);
    }

    template <>
    hipsparseStatus_t hipsparseXgthr(hipsparseHandle_t    handle,
                                     int                  nnz,
                                     const hipComplex*    y,
                                     hipComplex*          xVal,
                                     const int*           xInd,
                                     hipsparseIndexBase_t idxBase)
    {
        return hipsparseCgthr(handle, nnz, y, xVal, xInd, idxBase);
    }

    template <>
    hipsparseStatus_t hipsparseXgthr(hipsparseHandle_t       handle,
                                     int                     nnz,
                                     const hipDoubleComplex* y,
                                     hipDoubleComplex*       xVal,
                                     const int*              xInd,
                                     hipsparseIndexBase_t    idxBase)
    {
        return hipsparseZgthr(handle, nnz, y, xVal, xInd, idxBase);
    }
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 12000)
    template <>
    hipsparseStatus_t hipsparseXgthrz(hipsparseHandle_t    handle,
                                      int                  nnz,
                                      float*               y,
                                      float*               xVal,
                                      const int*           xInd,
                                      hipsparseIndexBase_t idxBase)
    {
        return hipsparseSgthrz(handle, nnz, y, xVal, xInd, idxBase);
    }

    template <>
    hipsparseStatus_t hipsparseXgthrz(hipsparseHandle_t    handle,
                                      int                  nnz,
                                      double*              y,
                                      double*              xVal,
                                      const int*           xInd,
                                      hipsparseIndexBase_t idxBase)
    {
        return hipsparseDgthrz(handle, nnz, y, xVal, xInd, idxBase);
    }

    template <>
    hipsparseStatus_t hipsparseXgthrz(hipsparseHandle_t    handle,
                                      int                  nnz,
                                      hipComplex*          y,
                                      hipComplex*          xVal,
                                      const int*           xInd,
                                      hipsparseIndexBase_t idxBase)
    {
        return hipsparseCgthrz(handle, nnz, y, xVal, xInd, idxBase);
    }

    template <>
    hipsparseStatus_t hipsparseXgthrz(hipsparseHandle_t    handle,
                                      int                  nnz,
                                      hipDoubleComplex*    y,
                                      hipDoubleComplex*    xVal,
                                      const int*           xInd,
                                      hipsparseIndexBase_t idxBase)
    {
        return hipsparseZgthrz(handle, nnz, y, xVal, xInd, idxBase);
    }
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 12000)
    template <>
    hipsparseStatus_t hipsparseXroti(hipsparseHandle_t    handle,
                                     int                  nnz,
                                     float*               xVal,
                                     const int*           xInd,
                                     float*               y,
                                     const float*         c,
                                     const float*         s,
                                     hipsparseIndexBase_t idxBase)
    {
        return hipsparseSroti(handle, nnz, xVal, xInd, y, c, s, idxBase);
    }

    template <>
    hipsparseStatus_t hipsparseXroti(hipsparseHandle_t    handle,
                                     int                  nnz,
                                     double*              xVal,
                                     const int*           xInd,
                                     double*              y,
                                     const double*        c,
                                     const double*        s,
                                     hipsparseIndexBase_t idxBase)
    {
        return hipsparseDroti(handle, nnz, xVal, xInd, y, c, s, idxBase);
    }
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 12000)
    template <>
    hipsparseStatus_t hipsparseXsctr(hipsparseHandle_t    handle,
                                     int                  nnz,
                                     const float*         xVal,
                                     const int*           xInd,
                                     float*               y,
                                     hipsparseIndexBase_t idxBase)
    {
        return hipsparseSsctr(handle, nnz, xVal, xInd, y, idxBase);
    }

    template <>
    hipsparseStatus_t hipsparseXsctr(hipsparseHandle_t    handle,
                                     int                  nnz,
                                     const double*        xVal,
                                     const int*           xInd,
                                     double*              y,
                                     hipsparseIndexBase_t idxBase)
    {
        return hipsparseDsctr(handle, nnz, xVal, xInd, y, idxBase);
    }

    template <>
    hipsparseStatus_t hipsparseXsctr(hipsparseHandle_t    handle,
                                     int                  nnz,
                                     const hipComplex*    xVal,
                                     const int*           xInd,
                                     hipComplex*          y,
                                     hipsparseIndexBase_t idxBase)
    {
        return hipsparseCsctr(handle, nnz, xVal, xInd, y, idxBase);
    }

    template <>
    hipsparseStatus_t hipsparseXsctr(hipsparseHandle_t       handle,
                                     int                     nnz,
                                     const hipDoubleComplex* xVal,
                                     const int*              xInd,
                                     hipDoubleComplex*       y,
                                     hipsparseIndexBase_t    idxBase)
    {
        return hipsparseZsctr(handle, nnz, xVal, xInd, y, idxBase);
    }
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 11000)
    template <>
    hipsparseStatus_t hipsparseXcsrmv(hipsparseHandle_t         handle,
                                      hipsparseOperation_t      trans,
                                      int                       m,
                                      int                       n,
                                      int                       nnz,
                                      const float*              alpha,
                                      const hipsparseMatDescr_t descr,
                                      const float*              csrVal,
                                      const int*                csrRowPtr,
                                      const int*                csrColInd,
                                      const float*              x,
                                      const float*              beta,
                                      float*                    y)
    {
        return hipsparseScsrmv(
            handle, trans, m, n, nnz, alpha, descr, csrVal, csrRowPtr, csrColInd, x, beta, y);
    }

    template <>
    hipsparseStatus_t hipsparseXcsrmv(hipsparseHandle_t         handle,
                                      hipsparseOperation_t      trans,
                                      int                       m,
                                      int                       n,
                                      int                       nnz,
                                      const double*             alpha,
                                      const hipsparseMatDescr_t descr,
                                      const double*             csrVal,
                                      const int*                csrRowPtr,
                                      const int*                csrColInd,
                                      const double*             x,
                                      const double*             beta,
                                      double*                   y)
    {
        return hipsparseDcsrmv(
            handle, trans, m, n, nnz, alpha, descr, csrVal, csrRowPtr, csrColInd, x, beta, y);
    }

    template <>
    hipsparseStatus_t hipsparseXcsrmv(hipsparseHandle_t         handle,
                                      hipsparseOperation_t      trans,
                                      int                       m,
                                      int                       n,
                                      int                       nnz,
                                      const hipComplex*         alpha,
                                      const hipsparseMatDescr_t descr,
                                      const hipComplex*         csrVal,
                                      const int*                csrRowPtr,
                                      const int*                csrColInd,
                                      const hipComplex*         x,
                                      const hipComplex*         beta,
                                      hipComplex*               y)
    {
        return hipsparseCcsrmv(
            handle, trans, m, n, nnz, alpha, descr, csrVal, csrRowPtr, csrColInd, x, beta, y);
    }

    template <>
    hipsparseStatus_t hipsparseXcsrmv(hipsparseHandle_t         handle,
                                      hipsparseOperation_t      trans,
                                      int                       m,
                                      int                       n,
                                      int                       nnz,
                                      const hipDoubleComplex*   alpha,
                                      const hipsparseMatDescr_t descr,
                                      const hipDoubleComplex*   csrVal,
                                      const int*                csrRowPtr,
                                      const int*                csrColInd,
                                      const hipDoubleComplex*   x,
                                      const hipDoubleComplex*   beta,
                                      hipDoubleComplex*         y)
    {
        return hipsparseZcsrmv(
            handle, trans, m, n, nnz, alpha, descr, csrVal, csrRowPtr, csrColInd, x, beta, y);
    }
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 12000)
    template <>
    hipsparseStatus_t hipsparseXcsrsv2_bufferSize(hipsparseHandle_t         handle,
                                                  hipsparseOperation_t      transA,
                                                  int                       m,
                                                  int                       nnz,
                                                  const hipsparseMatDescr_t descrA,
                                                  float*                    csrSortedValA,
                                                  const int*                csrSortedRowPtrA,
                                                  const int*                csrSortedColIndA,
                                                  csrsv2Info_t              info,
                                                  int*                      pBufferSizeInBytes)
    {
        return hipsparseScsrsv2_bufferSize(handle,
                                           transA,
                                           m,
                                           nnz,
                                           descrA,
                                           csrSortedValA,
                                           csrSortedRowPtrA,
                                           csrSortedColIndA,
                                           info,
                                           pBufferSizeInBytes);
    }

    template <>
    hipsparseStatus_t hipsparseXcsrsv2_bufferSize(hipsparseHandle_t         handle,
                                                  hipsparseOperation_t      transA,
                                                  int                       m,
                                                  int                       nnz,
                                                  const hipsparseMatDescr_t descrA,
                                                  double*                   csrSortedValA,
                                                  const int*                csrSortedRowPtrA,
                                                  const int*                csrSortedColIndA,
                                                  csrsv2Info_t              info,
                                                  int*                      pBufferSizeInBytes)
    {
        return hipsparseDcsrsv2_bufferSize(handle,
                                           transA,
                                           m,
                                           nnz,
                                           descrA,
                                           csrSortedValA,
                                           csrSortedRowPtrA,
                                           csrSortedColIndA,
                                           info,
                                           pBufferSizeInBytes);
    }

    template <>
    hipsparseStatus_t hipsparseXcsrsv2_bufferSize(hipsparseHandle_t         handle,
                                                  hipsparseOperation_t      transA,
                                                  int                       m,
                                                  int                       nnz,
                                                  const hipsparseMatDescr_t descrA,
                                                  hipComplex*               csrSortedValA,
                                                  const int*                csrSortedRowPtrA,
                                                  const int*                csrSortedColIndA,
                                                  csrsv2Info_t              info,
                                                  int*                      pBufferSizeInBytes)
    {
        return hipsparseCcsrsv2_bufferSize(handle,
                                           transA,
                                           m,
                                           nnz,
                                           descrA,
                                           csrSortedValA,
                                           csrSortedRowPtrA,
                                           csrSortedColIndA,
                                           info,
                                           pBufferSizeInBytes);
    }

    template <>
    hipsparseStatus_t hipsparseXcsrsv2_bufferSize(hipsparseHandle_t         handle,
                                                  hipsparseOperation_t      transA,
                                                  int                       m,
                                                  int                       nnz,
                                                  const hipsparseMatDescr_t descrA,
                                                  hipDoubleComplex*         csrSortedValA,
                                                  const int*                csrSortedRowPtrA,
                                                  const int*                csrSortedColIndA,
                                                  csrsv2Info_t              info,
                                                  int*                      pBufferSizeInBytes)
    {
        return hipsparseZcsrsv2_bufferSize(handle,
                                           transA,
                                           m,
                                           nnz,
                                           descrA,
                                           csrSortedValA,
                                           csrSortedRowPtrA,
                                           csrSortedColIndA,
                                           info,
                                           pBufferSizeInBytes);
    }
#endif

    template <>
    hipsparseStatus_t hipsparseXcsrsv2_bufferSizeExt(hipsparseHandle_t         handle,
                                                     hipsparseOperation_t      transA,
                                                     int                       m,
                                                     int                       nnz,
                                                     const hipsparseMatDescr_t descrA,
                                                     float*                    csrSortedValA,
                                                     const int*                csrSortedRowPtrA,
                                                     const int*                csrSortedColIndA,
                                                     csrsv2Info_t              info,
                                                     size_t*                   pBufferSizeInBytes)
    {
        return hipsparseScsrsv2_bufferSizeExt(handle,
                                              transA,
                                              m,
                                              nnz,
                                              descrA,
                                              csrSortedValA,
                                              csrSortedRowPtrA,
                                              csrSortedColIndA,
                                              info,
                                              pBufferSizeInBytes);
    }

    template <>
    hipsparseStatus_t hipsparseXcsrsv2_bufferSizeExt(hipsparseHandle_t         handle,
                                                     hipsparseOperation_t      transA,
                                                     int                       m,
                                                     int                       nnz,
                                                     const hipsparseMatDescr_t descrA,
                                                     double*                   csrSortedValA,
                                                     const int*                csrSortedRowPtrA,
                                                     const int*                csrSortedColIndA,
                                                     csrsv2Info_t              info,
                                                     size_t*                   pBufferSizeInBytes)
    {
        return hipsparseDcsrsv2_bufferSizeExt(handle,
                                              transA,
                                              m,
                                              nnz,
                                              descrA,
                                              csrSortedValA,
                                              csrSortedRowPtrA,
                                              csrSortedColIndA,
                                              info,
                                              pBufferSizeInBytes);
    }

    template <>
    hipsparseStatus_t hipsparseXcsrsv2_bufferSizeExt(hipsparseHandle_t         handle,
                                                     hipsparseOperation_t      transA,
                                                     int                       m,
                                                     int                       nnz,
                                                     const hipsparseMatDescr_t descrA,
                                                     hipComplex*               csrSortedValA,
                                                     const int*                csrSortedRowPtrA,
                                                     const int*                csrSortedColIndA,
                                                     csrsv2Info_t              info,
                                                     size_t*                   pBufferSizeInBytes)
    {
        return hipsparseCcsrsv2_bufferSizeExt(handle,
                                              transA,
                                              m,
                                              nnz,
                                              descrA,
                                              csrSortedValA,
                                              csrSortedRowPtrA,
                                              csrSortedColIndA,
                                              info,
                                              pBufferSizeInBytes);
    }

    template <>
    hipsparseStatus_t hipsparseXcsrsv2_bufferSizeExt(hipsparseHandle_t         handle,
                                                     hipsparseOperation_t      transA,
                                                     int                       m,
                                                     int                       nnz,
                                                     const hipsparseMatDescr_t descrA,
                                                     hipDoubleComplex*         csrSortedValA,
                                                     const int*                csrSortedRowPtrA,
                                                     const int*                csrSortedColIndA,
                                                     csrsv2Info_t              info,
                                                     size_t*                   pBufferSizeInBytes)
    {
        return hipsparseZcsrsv2_bufferSizeExt(handle,
                                              transA,
                                              m,
                                              nnz,
                                              descrA,
                                              csrSortedValA,
                                              csrSortedRowPtrA,
                                              csrSortedColIndA,
                                              info,
                                              pBufferSizeInBytes);
    }

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 12000)
    template <>
    hipsparseStatus_t hipsparseXcsrsv2_analysis(hipsparseHandle_t         handle,
                                                hipsparseOperation_t      transA,
                                                int                       m,
                                                int                       nnz,
                                                const hipsparseMatDescr_t descrA,
                                                const float*              csrSortedValA,
                                                const int*                csrSortedRowPtrA,
                                                const int*                csrSortedColIndA,
                                                csrsv2Info_t              info,
                                                hipsparseSolvePolicy_t    policy,
                                                void*                     pBuffer)
    {
        return hipsparseScsrsv2_analysis(handle,
                                         transA,
                                         m,
                                         nnz,
                                         descrA,
                                         csrSortedValA,
                                         csrSortedRowPtrA,
                                         csrSortedColIndA,
                                         info,
                                         policy,
                                         pBuffer);
    }

    template <>
    hipsparseStatus_t hipsparseXcsrsv2_analysis(hipsparseHandle_t         handle,
                                                hipsparseOperation_t      transA,
                                                int                       m,
                                                int                       nnz,
                                                const hipsparseMatDescr_t descrA,
                                                const double*             csrSortedValA,
                                                const int*                csrSortedRowPtrA,
                                                const int*                csrSortedColIndA,
                                                csrsv2Info_t              info,
                                                hipsparseSolvePolicy_t    policy,
                                                void*                     pBuffer)
    {
        return hipsparseDcsrsv2_analysis(handle,
                                         transA,
                                         m,
                                         nnz,
                                         descrA,
                                         csrSortedValA,
                                         csrSortedRowPtrA,
                                         csrSortedColIndA,
                                         info,
                                         policy,
                                         pBuffer);
    }

    template <>
    hipsparseStatus_t hipsparseXcsrsv2_analysis(hipsparseHandle_t         handle,
                                                hipsparseOperation_t      transA,
                                                int                       m,
                                                int                       nnz,
                                                const hipsparseMatDescr_t descrA,
                                                const hipComplex*         csrSortedValA,
                                                const int*                csrSortedRowPtrA,
                                                const int*                csrSortedColIndA,
                                                csrsv2Info_t              info,
                                                hipsparseSolvePolicy_t    policy,
                                                void*                     pBuffer)
    {
        return hipsparseCcsrsv2_analysis(handle,
                                         transA,
                                         m,
                                         nnz,
                                         descrA,
                                         csrSortedValA,
                                         csrSortedRowPtrA,
                                         csrSortedColIndA,
                                         info,
                                         policy,
                                         pBuffer);
    }

    template <>
    hipsparseStatus_t hipsparseXcsrsv2_analysis(hipsparseHandle_t         handle,
                                                hipsparseOperation_t      transA,
                                                int                       m,
                                                int                       nnz,
                                                const hipsparseMatDescr_t descrA,
                                                const hipDoubleComplex*   csrSortedValA,
                                                const int*                csrSortedRowPtrA,
                                                const int*                csrSortedColIndA,
                                                csrsv2Info_t              info,
                                                hipsparseSolvePolicy_t    policy,
                                                void*                     pBuffer)
    {
        return hipsparseZcsrsv2_analysis(handle,
                                         transA,
                                         m,
                                         nnz,
                                         descrA,
                                         csrSortedValA,
                                         csrSortedRowPtrA,
                                         csrSortedColIndA,
                                         info,
                                         policy,
                                         pBuffer);
    }
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 12000)
    template <>
    hipsparseStatus_t hipsparseXcsrsv2_solve(hipsparseHandle_t         handle,
                                             hipsparseOperation_t      transA,
                                             int                       m,
                                             int                       nnz,
                                             const float*              alpha,
                                             const hipsparseMatDescr_t descrA,
                                             const float*              csrSortedValA,
                                             const int*                csrSortedRowPtrA,
                                             const int*                csrSortedColIndA,
                                             csrsv2Info_t              info,
                                             const float*              f,
                                             float*                    x,
                                             hipsparseSolvePolicy_t    policy,
                                             void*                     pBuffer)
    {
        return hipsparseScsrsv2_solve(handle,
                                      transA,
                                      m,
                                      nnz,
                                      alpha,
                                      descrA,
                                      csrSortedValA,
                                      csrSortedRowPtrA,
                                      csrSortedColIndA,
                                      info,
                                      f,
                                      x,
                                      policy,
                                      pBuffer);
    }

    template <>
    hipsparseStatus_t hipsparseXcsrsv2_solve(hipsparseHandle_t         handle,
                                             hipsparseOperation_t      transA,
                                             int                       m,
                                             int                       nnz,
                                             const double*             alpha,
                                             const hipsparseMatDescr_t descrA,
                                             const double*             csrSortedValA,
                                             const int*                csrSortedRowPtrA,
                                             const int*                csrSortedColIndA,
                                             csrsv2Info_t              info,
                                             const double*             f,
                                             double*                   x,
                                             hipsparseSolvePolicy_t    policy,
                                             void*                     pBuffer)
    {
        return hipsparseDcsrsv2_solve(handle,
                                      transA,
                                      m,
                                      nnz,
                                      alpha,
                                      descrA,
                                      csrSortedValA,
                                      csrSortedRowPtrA,
                                      csrSortedColIndA,
                                      info,
                                      f,
                                      x,
                                      policy,
                                      pBuffer);
    }

    template <>
    hipsparseStatus_t hipsparseXcsrsv2_solve(hipsparseHandle_t         handle,
                                             hipsparseOperation_t      transA,
                                             int                       m,
                                             int                       nnz,
                                             const hipComplex*         alpha,
                                             const hipsparseMatDescr_t descrA,
                                             const hipComplex*         csrSortedValA,
                                             const int*                csrSortedRowPtrA,
                                             const int*                csrSortedColIndA,
                                             csrsv2Info_t              info,
                                             const hipComplex*         f,
                                             hipComplex*               x,
                                             hipsparseSolvePolicy_t    policy,
                                             void*                     pBuffer)
    {
        return hipsparseCcsrsv2_solve(handle,
                                      transA,
                                      m,
                                      nnz,
                                      alpha,
                                      descrA,
                                      csrSortedValA,
                                      csrSortedRowPtrA,
                                      csrSortedColIndA,
                                      info,
                                      f,
                                      x,
                                      policy,
                                      pBuffer);
    }

    template <>
    hipsparseStatus_t hipsparseXcsrsv2_solve(hipsparseHandle_t         handle,
                                             hipsparseOperation_t      transA,
                                             int                       m,
                                             int                       nnz,
                                             const hipDoubleComplex*   alpha,
                                             const hipsparseMatDescr_t descrA,
                                             const hipDoubleComplex*   csrSortedValA,
                                             const int*                csrSortedRowPtrA,
                                             const int*                csrSortedColIndA,
                                             csrsv2Info_t              info,
                                             const hipDoubleComplex*   f,
                                             hipDoubleComplex*         x,
                                             hipsparseSolvePolicy_t    policy,
                                             void*                     pBuffer)
    {
        return hipsparseZcsrsv2_solve(handle,
                                      transA,
                                      m,
                                      nnz,
                                      alpha,
                                      descrA,
                                      csrSortedValA,
                                      csrSortedRowPtrA,
                                      csrSortedColIndA,
                                      info,
                                      f,
                                      x,
                                      policy,
                                      pBuffer);
    }
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 11000)
    template <>
    hipsparseStatus_t hipsparseXhybmv(hipsparseHandle_t         handle,
                                      hipsparseOperation_t      trans,
                                      const float*              alpha,
                                      const hipsparseMatDescr_t descr,
                                      const hipsparseHybMat_t   hyb,
                                      const float*              x,
                                      const float*              beta,
                                      float*                    y)
    {
        return hipsparseShybmv(handle, trans, alpha, descr, hyb, x, beta, y);
    }

    template <>
    hipsparseStatus_t hipsparseXhybmv(hipsparseHandle_t         handle,
                                      hipsparseOperation_t      trans,
                                      const double*             alpha,
                                      const hipsparseMatDescr_t descr,
                                      const hipsparseHybMat_t   hyb,
                                      const double*             x,
                                      const double*             beta,
                                      double*                   y)
    {
        return hipsparseDhybmv(handle, trans, alpha, descr, hyb, x, beta, y);
    }

    template <>
    hipsparseStatus_t hipsparseXhybmv(hipsparseHandle_t         handle,
                                      hipsparseOperation_t      trans,
                                      const hipComplex*         alpha,
                                      const hipsparseMatDescr_t descr,
                                      const hipsparseHybMat_t   hyb,
                                      const hipComplex*         x,
                                      const hipComplex*         beta,
                                      hipComplex*               y)
    {
        return hipsparseChybmv(handle, trans, alpha, descr, hyb, x, beta, y);
    }

    template <>
    hipsparseStatus_t hipsparseXhybmv(hipsparseHandle_t         handle,
                                      hipsparseOperation_t      trans,
                                      const hipDoubleComplex*   alpha,
                                      const hipsparseMatDescr_t descr,
                                      const hipsparseHybMat_t   hyb,
                                      const hipDoubleComplex*   x,
                                      const hipDoubleComplex*   beta,
                                      hipDoubleComplex*         y)
    {
        return hipsparseZhybmv(handle, trans, alpha, descr, hyb, x, beta, y);
    }
#endif

    template <>
    hipsparseStatus_t hipsparseXbsrmv(hipsparseHandle_t         handle,
                                      hipsparseDirection_t      dirA,
                                      hipsparseOperation_t      transA,
                                      int                       mb,
                                      int                       nb,
                                      int                       nnzb,
                                      const float*              alpha,
                                      const hipsparseMatDescr_t descrA,
                                      const float*              bsrSortedValA,
                                      const int*                bsrSortedRowPtrA,
                                      const int*                bsrSortedColIndA,
                                      int                       blockDim,
                                      const float*              x,
                                      const float*              beta,
                                      float*                    y)
    {
        return hipsparseSbsrmv(handle,
                               dirA,
                               transA,
                               mb,
                               nb,
                               nnzb,
                               alpha,
                               descrA,
                               bsrSortedValA,
                               bsrSortedRowPtrA,
                               bsrSortedColIndA,
                               blockDim,
                               x,
                               beta,
                               y);
    }

    template <>
    hipsparseStatus_t hipsparseXbsrmv(hipsparseHandle_t         handle,
                                      hipsparseDirection_t      dirA,
                                      hipsparseOperation_t      transA,
                                      int                       mb,
                                      int                       nb,
                                      int                       nnzb,
                                      const double*             alpha,
                                      const hipsparseMatDescr_t descrA,
                                      const double*             bsrSortedValA,
                                      const int*                bsrSortedRowPtrA,
                                      const int*                bsrSortedColIndA,
                                      int                       blockDim,
                                      const double*             x,
                                      const double*             beta,
                                      double*                   y)
    {
        return hipsparseDbsrmv(handle,
                               dirA,
                               transA,
                               mb,
                               nb,
                               nnzb,
                               alpha,
                               descrA,
                               bsrSortedValA,
                               bsrSortedRowPtrA,
                               bsrSortedColIndA,
                               blockDim,
                               x,
                               beta,
                               y);
    }

    template <>
    hipsparseStatus_t hipsparseXbsrmv(hipsparseHandle_t         handle,
                                      hipsparseDirection_t      dirA,
                                      hipsparseOperation_t      transA,
                                      int                       mb,
                                      int                       nb,
                                      int                       nnzb,
                                      const hipComplex*         alpha,
                                      const hipsparseMatDescr_t descrA,
                                      const hipComplex*         bsrSortedValA,
                                      const int*                bsrSortedRowPtrA,
                                      const int*                bsrSortedColIndA,
                                      int                       blockDim,
                                      const hipComplex*         x,
                                      const hipComplex*         beta,
                                      hipComplex*               y)
    {
        return hipsparseCbsrmv(handle,
                               dirA,
                               transA,
                               mb,
                               nb,
                               nnzb,
                               alpha,
                               descrA,
                               bsrSortedValA,
                               bsrSortedRowPtrA,
                               bsrSortedColIndA,
                               blockDim,
                               x,
                               beta,
                               y);
    }

    template <>
    hipsparseStatus_t hipsparseXbsrmv(hipsparseHandle_t         handle,
                                      hipsparseDirection_t      dirA,
                                      hipsparseOperation_t      transA,
                                      int                       mb,
                                      int                       nb,
                                      int                       nnzb,
                                      const hipDoubleComplex*   alpha,
                                      const hipsparseMatDescr_t descrA,
                                      const hipDoubleComplex*   bsrSortedValA,
                                      const int*                bsrSortedRowPtrA,
                                      const int*                bsrSortedColIndA,
                                      int                       blockDim,
                                      const hipDoubleComplex*   x,
                                      const hipDoubleComplex*   beta,
                                      hipDoubleComplex*         y)
    {
        return hipsparseZbsrmv(handle,
                               dirA,
                               transA,
                               mb,
                               nb,
                               nnzb,
                               alpha,
                               descrA,
                               bsrSortedValA,
                               bsrSortedRowPtrA,
                               bsrSortedColIndA,
                               blockDim,
                               x,
                               beta,
                               y);
    }

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 13000)
    template <>
    hipsparseStatus_t hipsparseXbsrxmv(hipsparseHandle_t         handle,
                                       hipsparseDirection_t      dir,
                                       hipsparseOperation_t      trans,
                                       int                       sizeOfMask,
                                       int                       mb,
                                       int                       nb,
                                       int                       nnzb,
                                       const float*              alpha,
                                       const hipsparseMatDescr_t descr,
                                       const float*              bsrVal,
                                       const int*                bsrMaskPtr,
                                       const int*                bsrRowPtr,
                                       const int*                bsrEndPtr,
                                       const int*                bsrColInd,
                                       int                       blockDim,
                                       const float*              x,
                                       const float*              beta,
                                       float*                    y)
    {
        return hipsparseSbsrxmv(handle,
                                dir,
                                trans,
                                sizeOfMask,
                                mb,
                                nb,
                                nnzb,
                                alpha,
                                descr,
                                bsrVal,
                                bsrMaskPtr,
                                bsrRowPtr,
                                bsrEndPtr,
                                bsrColInd,
                                blockDim,
                                x,
                                beta,
                                y);
    }

    template <>
    hipsparseStatus_t hipsparseXbsrxmv(hipsparseHandle_t         handle,
                                       hipsparseDirection_t      dir,
                                       hipsparseOperation_t      trans,
                                       int                       sizeOfMask,
                                       int                       mb,
                                       int                       nb,
                                       int                       nnzb,
                                       const double*             alpha,
                                       const hipsparseMatDescr_t descr,
                                       const double*             bsrVal,
                                       const int*                bsrMaskPtr,
                                       const int*                bsrRowPtr,
                                       const int*                bsrEndPtr,

                                       const int*    bsrColInd,
                                       int           blockDim,
                                       const double* x,
                                       const double* beta,
                                       double*       y)
    {
        return hipsparseDbsrxmv(handle,
                                dir,
                                trans,
                                sizeOfMask,
                                mb,
                                nb,
                                nnzb,
                                alpha,
                                descr,
                                bsrVal,
                                bsrMaskPtr,
                                bsrRowPtr,
                                bsrEndPtr,
                                bsrColInd,
                                blockDim,
                                x,
                                beta,
                                y);
    }

    template <>
    hipsparseStatus_t hipsparseXbsrxmv(hipsparseHandle_t         handle,
                                       hipsparseDirection_t      dir,
                                       hipsparseOperation_t      trans,
                                       int                       sizeOfMask,
                                       int                       mb,
                                       int                       nb,
                                       int                       nnzb,
                                       const hipComplex*         alpha,
                                       const hipsparseMatDescr_t descr,
                                       const hipComplex*         bsrVal,
                                       const int*                bsrMaskPtr,
                                       const int*                bsrRowPtr,
                                       const int*                bsrEndPtr,

                                       const int*        bsrColInd,
                                       int               blockDim,
                                       const hipComplex* x,
                                       const hipComplex* beta,
                                       hipComplex*       y)
    {
        return hipsparseCbsrxmv(handle,
                                dir,
                                trans,
                                sizeOfMask,
                                mb,
                                nb,
                                nnzb,
                                alpha,
                                descr,
                                bsrVal,
                                bsrMaskPtr,
                                bsrRowPtr,
                                bsrEndPtr,
                                bsrColInd,
                                blockDim,
                                x,
                                beta,
                                y);
    }

    template <>
    hipsparseStatus_t hipsparseXbsrxmv(hipsparseHandle_t         handle,
                                       hipsparseDirection_t      dir,
                                       hipsparseOperation_t      trans,
                                       int                       sizeOfMask,
                                       int                       mb,
                                       int                       nb,
                                       int                       nnzb,
                                       const hipDoubleComplex*   alpha,
                                       const hipsparseMatDescr_t descr,
                                       const hipDoubleComplex*   bsrVal,
                                       const int*                bsrMaskPtr,
                                       const int*                bsrRowPtr,
                                       const int*                bsrEndPtr,

                                       const int*              bsrColInd,
                                       int                     blockDim,
                                       const hipDoubleComplex* x,
                                       const hipDoubleComplex* beta,
                                       hipDoubleComplex*       y)
    {
        return hipsparseZbsrxmv(handle,
                                dir,
                                trans,
                                sizeOfMask,
                                mb,
                                nb,
                                nnzb,
                                alpha,
                                descr,
                                bsrVal,
                                bsrMaskPtr,
                                bsrRowPtr,
                                bsrEndPtr,
                                bsrColInd,
                                blockDim,
                                x,
                                beta,
                                y);
    }
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 13000)
    template <>
    hipsparseStatus_t hipsparseXbsrsv2_bufferSize(hipsparseHandle_t         handle,
                                                  hipsparseDirection_t      dir,
                                                  hipsparseOperation_t      transA,
                                                  int                       mb,
                                                  int                       nnzb,
                                                  const hipsparseMatDescr_t descrA,
                                                  float*                    bsrSortedValA,
                                                  const int*                bsrSortedRowPtrA,
                                                  const int*                bsrSortedColIndA,
                                                  int                       blockDim,
                                                  bsrsv2Info_t              info,
                                                  int*                      pBufferSizeInBytes)
    {
        return hipsparseSbsrsv2_bufferSize(handle,
                                           dir,
                                           transA,
                                           mb,
                                           nnzb,
                                           descrA,
                                           bsrSortedValA,
                                           bsrSortedRowPtrA,
                                           bsrSortedColIndA,
                                           blockDim,
                                           info,
                                           pBufferSizeInBytes);
    }

    template <>
    hipsparseStatus_t hipsparseXbsrsv2_bufferSize(hipsparseHandle_t         handle,
                                                  hipsparseDirection_t      dir,
                                                  hipsparseOperation_t      transA,
                                                  int                       mb,
                                                  int                       nnzb,
                                                  const hipsparseMatDescr_t descrA,
                                                  double*                   bsrSortedValA,
                                                  const int*                bsrSortedRowPtrA,
                                                  const int*                bsrSortedColIndA,
                                                  int                       blockDim,
                                                  bsrsv2Info_t              info,
                                                  int*                      pBufferSizeInBytes)
    {
        return hipsparseDbsrsv2_bufferSize(handle,
                                           dir,
                                           transA,
                                           mb,
                                           nnzb,
                                           descrA,
                                           bsrSortedValA,
                                           bsrSortedRowPtrA,
                                           bsrSortedColIndA,
                                           blockDim,
                                           info,
                                           pBufferSizeInBytes);
    }

    template <>
    hipsparseStatus_t hipsparseXbsrsv2_bufferSize(hipsparseHandle_t         handle,
                                                  hipsparseDirection_t      dir,
                                                  hipsparseOperation_t      transA,
                                                  int                       mb,
                                                  int                       nnzb,
                                                  const hipsparseMatDescr_t descrA,
                                                  hipComplex*               bsrSortedValA,
                                                  const int*                bsrSortedRowPtrA,
                                                  const int*                bsrSortedColIndA,
                                                  int                       blockDim,
                                                  bsrsv2Info_t              info,
                                                  int*                      pBufferSizeInBytes)
    {
        return hipsparseCbsrsv2_bufferSize(handle,
                                           dir,
                                           transA,
                                           mb,
                                           nnzb,
                                           descrA,
                                           bsrSortedValA,
                                           bsrSortedRowPtrA,
                                           bsrSortedColIndA,
                                           blockDim,
                                           info,
                                           pBufferSizeInBytes);
    }

    template <>
    hipsparseStatus_t hipsparseXbsrsv2_bufferSize(hipsparseHandle_t         handle,
                                                  hipsparseDirection_t      dir,
                                                  hipsparseOperation_t      transA,
                                                  int                       mb,
                                                  int                       nnzb,
                                                  const hipsparseMatDescr_t descrA,
                                                  hipDoubleComplex*         bsrSortedValA,
                                                  const int*                bsrSortedRowPtrA,
                                                  const int*                bsrSortedColIndA,
                                                  int                       blockDim,
                                                  bsrsv2Info_t              info,
                                                  int*                      pBufferSizeInBytes)
    {
        return hipsparseZbsrsv2_bufferSize(handle,
                                           dir,
                                           transA,
                                           mb,
                                           nnzb,
                                           descrA,
                                           bsrSortedValA,
                                           bsrSortedRowPtrA,
                                           bsrSortedColIndA,
                                           blockDim,
                                           info,
                                           pBufferSizeInBytes);
    }
#endif

    template <>
    hipsparseStatus_t hipsparseXbsrsv2_bufferSizeExt(hipsparseHandle_t         handle,
                                                     hipsparseDirection_t      dir,
                                                     hipsparseOperation_t      transA,
                                                     int                       mb,
                                                     int                       nnzb,
                                                     const hipsparseMatDescr_t descrA,
                                                     float*                    bsrSortedValA,
                                                     const int*                bsrSortedRowPtrA,
                                                     const int*                bsrSortedColIndA,
                                                     int                       blockDim,
                                                     bsrsv2Info_t              info,
                                                     size_t*                   pBufferSizeInBytes)
    {
        return hipsparseSbsrsv2_bufferSizeExt(handle,
                                              dir,
                                              transA,
                                              mb,
                                              nnzb,
                                              descrA,
                                              bsrSortedValA,
                                              bsrSortedRowPtrA,
                                              bsrSortedColIndA,
                                              blockDim,
                                              info,
                                              pBufferSizeInBytes);
    }

    template <>
    hipsparseStatus_t hipsparseXbsrsv2_bufferSizeExt(hipsparseHandle_t         handle,
                                                     hipsparseDirection_t      dir,
                                                     hipsparseOperation_t      transA,
                                                     int                       mb,
                                                     int                       nnzb,
                                                     const hipsparseMatDescr_t descrA,
                                                     double*                   bsrSortedValA,
                                                     const int*                bsrSortedRowPtrA,
                                                     const int*                bsrSortedColIndA,
                                                     int                       blockDim,
                                                     bsrsv2Info_t              info,
                                                     size_t*                   pBufferSizeInBytes)
    {
        return hipsparseDbsrsv2_bufferSizeExt(handle,
                                              dir,
                                              transA,
                                              mb,
                                              nnzb,
                                              descrA,
                                              bsrSortedValA,
                                              bsrSortedRowPtrA,
                                              bsrSortedColIndA,
                                              blockDim,
                                              info,
                                              pBufferSizeInBytes);
    }

    template <>
    hipsparseStatus_t hipsparseXbsrsv2_bufferSizeExt(hipsparseHandle_t         handle,
                                                     hipsparseDirection_t      dir,
                                                     hipsparseOperation_t      transA,
                                                     int                       mb,
                                                     int                       nnzb,
                                                     const hipsparseMatDescr_t descrA,
                                                     hipComplex*               bsrSortedValA,
                                                     const int*                bsrSortedRowPtrA,
                                                     const int*                bsrSortedColIndA,
                                                     int                       blockDim,
                                                     bsrsv2Info_t              info,
                                                     size_t*                   pBufferSizeInBytes)
    {
        return hipsparseCbsrsv2_bufferSizeExt(handle,
                                              dir,
                                              transA,
                                              mb,
                                              nnzb,
                                              descrA,
                                              bsrSortedValA,
                                              bsrSortedRowPtrA,
                                              bsrSortedColIndA,
                                              blockDim,
                                              info,
                                              pBufferSizeInBytes);
    }

    template <>
    hipsparseStatus_t hipsparseXbsrsv2_bufferSizeExt(hipsparseHandle_t         handle,
                                                     hipsparseDirection_t      dir,
                                                     hipsparseOperation_t      transA,
                                                     int                       mb,
                                                     int                       nnzb,
                                                     const hipsparseMatDescr_t descrA,
                                                     hipDoubleComplex*         bsrSortedValA,
                                                     const int*                bsrSortedRowPtrA,
                                                     const int*                bsrSortedColIndA,
                                                     int                       blockDim,
                                                     bsrsv2Info_t              info,
                                                     size_t*                   pBufferSizeInBytes)
    {
        return hipsparseZbsrsv2_bufferSizeExt(handle,
                                              dir,
                                              transA,
                                              mb,
                                              nnzb,
                                              descrA,
                                              bsrSortedValA,
                                              bsrSortedRowPtrA,
                                              bsrSortedColIndA,
                                              blockDim,
                                              info,
                                              pBufferSizeInBytes);
    }

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 13000)
    template <>
    hipsparseStatus_t hipsparseXbsrsv2_analysis(hipsparseHandle_t         handle,
                                                hipsparseDirection_t      dir,
                                                hipsparseOperation_t      transA,
                                                int                       mb,
                                                int                       nnzb,
                                                const hipsparseMatDescr_t descrA,
                                                const float*              bsrSortedValA,
                                                const int*                bsrSortedRowPtrA,
                                                const int*                bsrSortedColIndA,
                                                int                       blockDim,
                                                bsrsv2Info_t              info,
                                                hipsparseSolvePolicy_t    policy,
                                                void*                     pBuffer)
    {
        return hipsparseSbsrsv2_analysis(handle,
                                         dir,
                                         transA,
                                         mb,
                                         nnzb,
                                         descrA,
                                         bsrSortedValA,
                                         bsrSortedRowPtrA,
                                         bsrSortedColIndA,
                                         blockDim,
                                         info,
                                         policy,
                                         pBuffer);
    }

    template <>
    hipsparseStatus_t hipsparseXbsrsv2_analysis(hipsparseHandle_t         handle,
                                                hipsparseDirection_t      dir,
                                                hipsparseOperation_t      transA,
                                                int                       mb,
                                                int                       nnzb,
                                                const hipsparseMatDescr_t descrA,
                                                const double*             bsrSortedValA,
                                                const int*                bsrSortedRowPtrA,
                                                const int*                bsrSortedColIndA,
                                                int                       blockDim,
                                                bsrsv2Info_t              info,
                                                hipsparseSolvePolicy_t    policy,
                                                void*                     pBuffer)
    {
        return hipsparseDbsrsv2_analysis(handle,
                                         dir,
                                         transA,
                                         mb,
                                         nnzb,
                                         descrA,
                                         bsrSortedValA,
                                         bsrSortedRowPtrA,
                                         bsrSortedColIndA,
                                         blockDim,
                                         info,
                                         policy,
                                         pBuffer);
    }

    template <>
    hipsparseStatus_t hipsparseXbsrsv2_analysis(hipsparseHandle_t         handle,
                                                hipsparseDirection_t      dir,
                                                hipsparseOperation_t      transA,
                                                int                       mb,
                                                int                       nnzb,
                                                const hipsparseMatDescr_t descrA,
                                                const hipComplex*         bsrSortedValA,
                                                const int*                bsrSortedRowPtrA,
                                                const int*                bsrSortedColIndA,
                                                int                       blockDim,
                                                bsrsv2Info_t              info,
                                                hipsparseSolvePolicy_t    policy,
                                                void*                     pBuffer)
    {
        return hipsparseCbsrsv2_analysis(handle,
                                         dir,
                                         transA,
                                         mb,
                                         nnzb,
                                         descrA,
                                         bsrSortedValA,
                                         bsrSortedRowPtrA,
                                         bsrSortedColIndA,
                                         blockDim,
                                         info,
                                         policy,
                                         pBuffer);
    }

    template <>
    hipsparseStatus_t hipsparseXbsrsv2_analysis(hipsparseHandle_t         handle,
                                                hipsparseDirection_t      dir,
                                                hipsparseOperation_t      transA,
                                                int                       mb,
                                                int                       nnzb,
                                                const hipsparseMatDescr_t descrA,
                                                const hipDoubleComplex*   bsrSortedValA,
                                                const int*                bsrSortedRowPtrA,
                                                const int*                bsrSortedColIndA,
                                                int                       blockDim,
                                                bsrsv2Info_t              info,
                                                hipsparseSolvePolicy_t    policy,
                                                void*                     pBuffer)
    {
        return hipsparseZbsrsv2_analysis(handle,
                                         dir,
                                         transA,
                                         mb,
                                         nnzb,
                                         descrA,
                                         bsrSortedValA,
                                         bsrSortedRowPtrA,
                                         bsrSortedColIndA,
                                         blockDim,
                                         info,
                                         policy,
                                         pBuffer);
    }
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 13000)
    template <>
    hipsparseStatus_t hipsparseXbsrsv2_solve(hipsparseHandle_t         handle,
                                             hipsparseDirection_t      dir,
                                             hipsparseOperation_t      transA,
                                             int                       mb,
                                             int                       nnzb,
                                             const float*              alpha,
                                             const hipsparseMatDescr_t descrA,
                                             const float*              bsrSortedValA,
                                             const int*                bsrSortedRowPtrA,
                                             const int*                bsrSortedColIndA,
                                             int                       blockDim,
                                             bsrsv2Info_t              info,
                                             const float*              f,
                                             float*                    x,
                                             hipsparseSolvePolicy_t    policy,
                                             void*                     pBuffer)
    {
        return hipsparseSbsrsv2_solve(handle,
                                      dir,
                                      transA,
                                      mb,
                                      nnzb,
                                      alpha,
                                      descrA,
                                      bsrSortedValA,
                                      bsrSortedRowPtrA,
                                      bsrSortedColIndA,
                                      blockDim,
                                      info,
                                      f,
                                      x,
                                      policy,
                                      pBuffer);
    }

    template <>
    hipsparseStatus_t hipsparseXbsrsv2_solve(hipsparseHandle_t         handle,
                                             hipsparseDirection_t      dir,
                                             hipsparseOperation_t      transA,
                                             int                       mb,
                                             int                       nnzb,
                                             const double*             alpha,
                                             const hipsparseMatDescr_t descrA,
                                             const double*             bsrSortedValA,
                                             const int*                bsrSortedRowPtrA,
                                             const int*                bsrSortedColIndA,
                                             int                       blockDim,
                                             bsrsv2Info_t              info,
                                             const double*             f,
                                             double*                   x,
                                             hipsparseSolvePolicy_t    policy,
                                             void*                     pBuffer)
    {
        return hipsparseDbsrsv2_solve(handle,
                                      dir,
                                      transA,
                                      mb,
                                      nnzb,
                                      alpha,
                                      descrA,
                                      bsrSortedValA,
                                      bsrSortedRowPtrA,
                                      bsrSortedColIndA,
                                      blockDim,
                                      info,
                                      f,
                                      x,
                                      policy,
                                      pBuffer);
    }

    template <>
    hipsparseStatus_t hipsparseXbsrsv2_solve(hipsparseHandle_t         handle,
                                             hipsparseDirection_t      dir,
                                             hipsparseOperation_t      transA,
                                             int                       mb,
                                             int                       nnzb,
                                             const hipComplex*         alpha,
                                             const hipsparseMatDescr_t descrA,
                                             const hipComplex*         bsrSortedValA,
                                             const int*                bsrSortedRowPtrA,
                                             const int*                bsrSortedColIndA,
                                             int                       blockDim,
                                             bsrsv2Info_t              info,
                                             const hipComplex*         f,
                                             hipComplex*               x,
                                             hipsparseSolvePolicy_t    policy,
                                             void*                     pBuffer)
    {
        return hipsparseCbsrsv2_solve(handle,
                                      dir,
                                      transA,
                                      mb,
                                      nnzb,
                                      alpha,
                                      descrA,
                                      bsrSortedValA,
                                      bsrSortedRowPtrA,
                                      bsrSortedColIndA,
                                      blockDim,
                                      info,
                                      f,
                                      x,
                                      policy,
                                      pBuffer);
    }

    template <>
    hipsparseStatus_t hipsparseXbsrsv2_solve(hipsparseHandle_t         handle,
                                             hipsparseDirection_t      dir,
                                             hipsparseOperation_t      transA,
                                             int                       mb,
                                             int                       nnzb,
                                             const hipDoubleComplex*   alpha,
                                             const hipsparseMatDescr_t descrA,
                                             const hipDoubleComplex*   bsrSortedValA,
                                             const int*                bsrSortedRowPtrA,
                                             const int*                bsrSortedColIndA,
                                             int                       blockDim,
                                             bsrsv2Info_t              info,
                                             const hipDoubleComplex*   f,
                                             hipDoubleComplex*         x,
                                             hipsparseSolvePolicy_t    policy,
                                             void*                     pBuffer)
    {
        return hipsparseZbsrsv2_solve(handle,
                                      dir,
                                      transA,
                                      mb,
                                      nnzb,
                                      alpha,
                                      descrA,
                                      bsrSortedValA,
                                      bsrSortedRowPtrA,
                                      bsrSortedColIndA,
                                      blockDim,
                                      info,
                                      f,
                                      x,
                                      policy,
                                      pBuffer);
    }
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 12000)
    template <>
    hipsparseStatus_t hipsparseXgemvi_bufferSize<float>(hipsparseHandle_t    handle,
                                                        hipsparseOperation_t transA,
                                                        int                  m,
                                                        int                  n,
                                                        int                  nnz,
                                                        int*                 pBufferSizeInBytes)
    {
        return hipsparseSgemvi_bufferSize(handle, transA, m, n, nnz, pBufferSizeInBytes);
    }

    template <>
    hipsparseStatus_t hipsparseXgemvi_bufferSize<double>(hipsparseHandle_t    handle,
                                                         hipsparseOperation_t transA,
                                                         int                  m,
                                                         int                  n,
                                                         int                  nnz,
                                                         int*                 pBufferSizeInBytes)
    {
        return hipsparseDgemvi_bufferSize(handle, transA, m, n, nnz, pBufferSizeInBytes);
    }

    template <>
    hipsparseStatus_t hipsparseXgemvi_bufferSize<hipComplex>(hipsparseHandle_t    handle,
                                                             hipsparseOperation_t transA,
                                                             int                  m,
                                                             int                  n,
                                                             int                  nnz,
                                                             int* pBufferSizeInBytes)
    {
        return hipsparseCgemvi_bufferSize(handle, transA, m, n, nnz, pBufferSizeInBytes);
    }

    template <>
    hipsparseStatus_t hipsparseXgemvi_bufferSize<hipDoubleComplex>(hipsparseHandle_t    handle,
                                                                   hipsparseOperation_t transA,
                                                                   int                  m,
                                                                   int                  n,
                                                                   int                  nnz,
                                                                   int* pBufferSizeInBytes)
    {
        return hipsparseZgemvi_bufferSize(handle, transA, m, n, nnz, pBufferSizeInBytes);
    }
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 12000)
    template <>
    hipsparseStatus_t hipsparseXgemvi(hipsparseHandle_t    handle,
                                      hipsparseOperation_t transA,
                                      int                  m,
                                      int                  n,
                                      const float*         alpha,
                                      const float*         A,
                                      int                  lda,
                                      int                  nnz,
                                      const float*         x,
                                      const int*           xInd,
                                      const float*         beta,
                                      float*               y,
                                      hipsparseIndexBase_t idxBase,
                                      void*                pBuffer)
    {
        return hipsparseSgemvi(
            handle, transA, m, n, alpha, A, lda, nnz, x, xInd, beta, y, idxBase, pBuffer);
    }

    template <>
    hipsparseStatus_t hipsparseXgemvi(hipsparseHandle_t    handle,
                                      hipsparseOperation_t transA,
                                      int                  m,
                                      int                  n,
                                      const double*        alpha,
                                      const double*        A,
                                      int                  lda,
                                      int                  nnz,
                                      const double*        x,
                                      const int*           xInd,
                                      const double*        beta,
                                      double*              y,
                                      hipsparseIndexBase_t idxBase,
                                      void*                pBuffer)
    {
        return hipsparseDgemvi(
            handle, transA, m, n, alpha, A, lda, nnz, x, xInd, beta, y, idxBase, pBuffer);
    }

    template <>
    hipsparseStatus_t hipsparseXgemvi(hipsparseHandle_t    handle,
                                      hipsparseOperation_t transA,
                                      int                  m,
                                      int                  n,
                                      const hipComplex*    alpha,
                                      const hipComplex*    A,
                                      int                  lda,
                                      int                  nnz,
                                      const hipComplex*    x,
                                      const int*           xInd,
                                      const hipComplex*    beta,
                                      hipComplex*          y,
                                      hipsparseIndexBase_t idxBase,
                                      void*                pBuffer)
    {
        return hipsparseCgemvi(
            handle, transA, m, n, alpha, A, lda, nnz, x, xInd, beta, y, idxBase, pBuffer);
    }

    template <>
    hipsparseStatus_t hipsparseXgemvi(hipsparseHandle_t       handle,
                                      hipsparseOperation_t    transA,
                                      int                     m,
                                      int                     n,
                                      const hipDoubleComplex* alpha,
                                      const hipDoubleComplex* A,
                                      int                     lda,
                                      int                     nnz,
                                      const hipDoubleComplex* x,
                                      const int*              xInd,
                                      const hipDoubleComplex* beta,
                                      hipDoubleComplex*       y,
                                      hipsparseIndexBase_t    idxBase,
                                      void*                   pBuffer)
    {
        return hipsparseZgemvi(
            handle, transA, m, n, alpha, A, lda, nnz, x, xInd, beta, y, idxBase, pBuffer);
    }
#endif

    template <>
    hipsparseStatus_t hipsparseXbsrmm(hipsparseHandle_t         handle,
                                      hipsparseDirection_t      dirA,
                                      hipsparseOperation_t      transA,
                                      hipsparseOperation_t      transB,
                                      int                       mb,
                                      int                       n,
                                      int                       kb,
                                      int                       nnzb,
                                      const float*              alpha,
                                      const hipsparseMatDescr_t descrA,
                                      const float*              bsrValA,
                                      const int*                bsrRowPtrA,
                                      const int*                bsrColIndA,
                                      int                       blockDim,
                                      const float*              B,
                                      int                       ldb,
                                      const float*              beta,
                                      float*                    C,
                                      int                       ldc)
    {
        return hipsparseSbsrmm(handle,
                               dirA,
                               transA,
                               transB,
                               mb,
                               n,
                               kb,
                               nnzb,
                               alpha,
                               descrA,
                               bsrValA,
                               bsrRowPtrA,
                               bsrColIndA,
                               blockDim,
                               B,
                               ldb,
                               beta,
                               C,
                               ldc);
    }

    template <>
    hipsparseStatus_t hipsparseXbsrmm(hipsparseHandle_t         handle,
                                      hipsparseDirection_t      dirA,
                                      hipsparseOperation_t      transA,
                                      hipsparseOperation_t      transB,
                                      int                       mb,
                                      int                       n,
                                      int                       kb,
                                      int                       nnzb,
                                      const double*             alpha,
                                      const hipsparseMatDescr_t descrA,
                                      const double*             bsrValA,
                                      const int*                bsrRowPtrA,
                                      const int*                bsrColIndA,
                                      int                       blockDim,
                                      const double*             B,
                                      int                       ldb,
                                      const double*             beta,
                                      double*                   C,
                                      int                       ldc)
    {
        return hipsparseDbsrmm(handle,
                               dirA,
                               transA,
                               transB,
                               mb,
                               n,
                               kb,
                               nnzb,
                               alpha,
                               descrA,
                               bsrValA,
                               bsrRowPtrA,
                               bsrColIndA,
                               blockDim,
                               B,
                               ldb,
                               beta,
                               C,
                               ldc);
    }

    template <>
    hipsparseStatus_t hipsparseXbsrmm(hipsparseHandle_t         handle,
                                      hipsparseDirection_t      dirA,
                                      hipsparseOperation_t      transA,
                                      hipsparseOperation_t      transB,
                                      int                       mb,
                                      int                       n,
                                      int                       kb,
                                      int                       nnzb,
                                      const hipComplex*         alpha,
                                      const hipsparseMatDescr_t descrA,
                                      const hipComplex*         bsrValA,
                                      const int*                bsrRowPtrA,
                                      const int*                bsrColIndA,
                                      int                       blockDim,
                                      const hipComplex*         B,
                                      int                       ldb,
                                      const hipComplex*         beta,
                                      hipComplex*               C,
                                      int                       ldc)
    {
        return hipsparseCbsrmm(handle,
                               dirA,
                               transA,
                               transB,
                               mb,
                               n,
                               kb,
                               nnzb,
                               alpha,
                               descrA,
                               bsrValA,
                               bsrRowPtrA,
                               bsrColIndA,
                               blockDim,
                               B,
                               ldb,
                               beta,
                               C,
                               ldc);
    }

    template <>
    hipsparseStatus_t hipsparseXbsrmm(hipsparseHandle_t         handle,
                                      hipsparseDirection_t      dirA,
                                      hipsparseOperation_t      transA,
                                      hipsparseOperation_t      transB,
                                      int                       mb,
                                      int                       n,
                                      int                       kb,
                                      int                       nnzb,
                                      const hipDoubleComplex*   alpha,
                                      const hipsparseMatDescr_t descrA,
                                      const hipDoubleComplex*   bsrValA,
                                      const int*                bsrRowPtrA,
                                      const int*                bsrColIndA,
                                      int                       blockDim,
                                      const hipDoubleComplex*   B,
                                      int                       ldb,
                                      const hipDoubleComplex*   beta,
                                      hipDoubleComplex*         C,
                                      int                       ldc)
    {
        return hipsparseZbsrmm(handle,
                               dirA,
                               transA,
                               transB,
                               mb,
                               n,
                               kb,
                               nnzb,
                               alpha,
                               descrA,
                               bsrValA,
                               bsrRowPtrA,
                               bsrColIndA,
                               blockDim,
                               B,
                               ldb,
                               beta,
                               C,
                               ldc);
    }

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 11000)
    template <>
    hipsparseStatus_t hipsparseXcsrmm2(hipsparseHandle_t         handle,
                                       hipsparseOperation_t      transA,
                                       hipsparseOperation_t      transB,
                                       int                       m,
                                       int                       n,
                                       int                       k,
                                       int                       nnz,
                                       const float*              alpha,
                                       const hipsparseMatDescr_t descr,
                                       const float*              csrVal,
                                       const int*                csrRowPtr,
                                       const int*                csrColInd,
                                       const float*              B,
                                       int                       ldb,
                                       const float*              beta,
                                       float*                    C,
                                       int                       ldc)
    {
        return hipsparseScsrmm2(handle,
                                transA,
                                transB,
                                m,
                                n,
                                k,
                                nnz,
                                alpha,
                                descr,
                                csrVal,
                                csrRowPtr,
                                csrColInd,
                                B,
                                ldb,
                                beta,
                                C,
                                ldc);
    }

    template <>
    hipsparseStatus_t hipsparseXcsrmm2(hipsparseHandle_t         handle,
                                       hipsparseOperation_t      transA,
                                       hipsparseOperation_t      transB,
                                       int                       m,
                                       int                       n,
                                       int                       k,
                                       int                       nnz,
                                       const double*             alpha,
                                       const hipsparseMatDescr_t descr,
                                       const double*             csrVal,
                                       const int*                csrRowPtr,
                                       const int*                csrColInd,
                                       const double*             B,
                                       int                       ldb,
                                       const double*             beta,
                                       double*                   C,
                                       int                       ldc)
    {
        return hipsparseDcsrmm2(handle,
                                transA,
                                transB,
                                m,
                                n,
                                k,
                                nnz,
                                alpha,
                                descr,
                                csrVal,
                                csrRowPtr,
                                csrColInd,
                                B,
                                ldb,
                                beta,
                                C,
                                ldc);
    }

    template <>
    hipsparseStatus_t hipsparseXcsrmm2(hipsparseHandle_t         handle,
                                       hipsparseOperation_t      transA,
                                       hipsparseOperation_t      transB,
                                       int                       m,
                                       int                       n,
                                       int                       k,
                                       int                       nnz,
                                       const hipComplex*         alpha,
                                       const hipsparseMatDescr_t descr,
                                       const hipComplex*         csrVal,
                                       const int*                csrRowPtr,
                                       const int*                csrColInd,
                                       const hipComplex*         B,
                                       int                       ldb,
                                       const hipComplex*         beta,
                                       hipComplex*               C,
                                       int                       ldc)
    {
        return hipsparseCcsrmm2(handle,
                                transA,
                                transB,
                                m,
                                n,
                                k,
                                nnz,
                                alpha,
                                descr,
                                csrVal,
                                csrRowPtr,
                                csrColInd,
                                B,
                                ldb,
                                beta,
                                C,
                                ldc);
    }

    template <>
    hipsparseStatus_t hipsparseXcsrmm2(hipsparseHandle_t         handle,
                                       hipsparseOperation_t      transA,
                                       hipsparseOperation_t      transB,
                                       int                       m,
                                       int                       n,
                                       int                       k,
                                       int                       nnz,
                                       const hipDoubleComplex*   alpha,
                                       const hipsparseMatDescr_t descr,
                                       const hipDoubleComplex*   csrVal,
                                       const int*                csrRowPtr,
                                       const int*                csrColInd,
                                       const hipDoubleComplex*   B,
                                       int                       ldb,
                                       const hipDoubleComplex*   beta,
                                       hipDoubleComplex*         C,
                                       int                       ldc)
    {
        return hipsparseZcsrmm2(handle,
                                transA,
                                transB,
                                m,
                                n,
                                k,
                                nnz,
                                alpha,
                                descr,
                                csrVal,
                                csrRowPtr,
                                csrColInd,
                                B,
                                ldb,
                                beta,
                                C,
                                ldc);
    }
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 13000)
    template <>
    hipsparseStatus_t hipsparseXbsrsm2_bufferSize(hipsparseHandle_t         handle,
                                                  hipsparseDirection_t      dirA,
                                                  hipsparseOperation_t      transA,
                                                  hipsparseOperation_t      transX,
                                                  int                       mb,
                                                  int                       nrhs,
                                                  int                       nnzb,
                                                  const hipsparseMatDescr_t descrA,
                                                  float*                    bsrSortedValA,
                                                  const int*                bsrSortedRowPtrA,
                                                  const int*                bsrSortedColIndA,
                                                  int                       blockDim,
                                                  bsrsm2Info_t              info,
                                                  int*                      pBufferSizeInBytes)
    {
        return hipsparseSbsrsm2_bufferSize(handle,
                                           dirA,
                                           transA,
                                           transX,
                                           mb,
                                           nrhs,
                                           nnzb,
                                           descrA,
                                           bsrSortedValA,
                                           bsrSortedRowPtrA,
                                           bsrSortedColIndA,
                                           blockDim,
                                           info,
                                           pBufferSizeInBytes);
    }

    template <>
    hipsparseStatus_t hipsparseXbsrsm2_bufferSize(hipsparseHandle_t         handle,
                                                  hipsparseDirection_t      dirA,
                                                  hipsparseOperation_t      transA,
                                                  hipsparseOperation_t      transX,
                                                  int                       mb,
                                                  int                       nrhs,
                                                  int                       nnzb,
                                                  const hipsparseMatDescr_t descrA,
                                                  double*                   bsrSortedValA,
                                                  const int*                bsrSortedRowPtrA,
                                                  const int*                bsrSortedColIndA,
                                                  int                       blockDim,
                                                  bsrsm2Info_t              info,
                                                  int*                      pBufferSizeInBytes)
    {
        return hipsparseDbsrsm2_bufferSize(handle,
                                           dirA,
                                           transA,
                                           transX,
                                           mb,
                                           nrhs,
                                           nnzb,
                                           descrA,
                                           bsrSortedValA,
                                           bsrSortedRowPtrA,
                                           bsrSortedColIndA,
                                           blockDim,
                                           info,
                                           pBufferSizeInBytes);
    }

    template <>
    hipsparseStatus_t hipsparseXbsrsm2_bufferSize(hipsparseHandle_t         handle,
                                                  hipsparseDirection_t      dirA,
                                                  hipsparseOperation_t      transA,
                                                  hipsparseOperation_t      transX,
                                                  int                       mb,
                                                  int                       nrhs,
                                                  int                       nnzb,
                                                  const hipsparseMatDescr_t descrA,
                                                  hipComplex*               bsrSortedValA,
                                                  const int*                bsrSortedRowPtrA,
                                                  const int*                bsrSortedColIndA,
                                                  int                       blockDim,
                                                  bsrsm2Info_t              info,
                                                  int*                      pBufferSizeInBytes)
    {
        return hipsparseCbsrsm2_bufferSize(handle,
                                           dirA,
                                           transA,
                                           transX,
                                           mb,
                                           nrhs,
                                           nnzb,
                                           descrA,
                                           bsrSortedValA,
                                           bsrSortedRowPtrA,
                                           bsrSortedColIndA,
                                           blockDim,
                                           info,
                                           pBufferSizeInBytes);
    }

    template <>
    hipsparseStatus_t hipsparseXbsrsm2_bufferSize(hipsparseHandle_t         handle,
                                                  hipsparseDirection_t      dirA,
                                                  hipsparseOperation_t      transA,
                                                  hipsparseOperation_t      transX,
                                                  int                       mb,
                                                  int                       nrhs,
                                                  int                       nnzb,
                                                  const hipsparseMatDescr_t descrA,
                                                  hipDoubleComplex*         bsrSortedValA,
                                                  const int*                bsrSortedRowPtrA,
                                                  const int*                bsrSortedColIndA,
                                                  int                       blockDim,
                                                  bsrsm2Info_t              info,
                                                  int*                      pBufferSizeInBytes)
    {
        return hipsparseZbsrsm2_bufferSize(handle,
                                           dirA,
                                           transA,
                                           transX,
                                           mb,
                                           nrhs,
                                           nnzb,
                                           descrA,
                                           bsrSortedValA,
                                           bsrSortedRowPtrA,
                                           bsrSortedColIndA,
                                           blockDim,
                                           info,
                                           pBufferSizeInBytes);
    }
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 13000)
    template <>
    hipsparseStatus_t hipsparseXbsrsm2_analysis(hipsparseHandle_t         handle,
                                                hipsparseDirection_t      dirA,
                                                hipsparseOperation_t      transA,
                                                hipsparseOperation_t      transX,
                                                int                       mb,
                                                int                       nrhs,
                                                int                       nnzb,
                                                const hipsparseMatDescr_t descrA,
                                                const float*              bsrSortedValA,
                                                const int*                bsrSortedRowPtrA,
                                                const int*                bsrSortedColIndA,
                                                int                       blockDim,
                                                bsrsm2Info_t              info,
                                                hipsparseSolvePolicy_t    policy,
                                                void*                     pBuffer)
    {
        return hipsparseSbsrsm2_analysis(handle,
                                         dirA,
                                         transA,
                                         transX,
                                         mb,
                                         nrhs,
                                         nnzb,
                                         descrA,
                                         bsrSortedValA,
                                         bsrSortedRowPtrA,
                                         bsrSortedColIndA,
                                         blockDim,
                                         info,
                                         policy,
                                         pBuffer);
    }

    template <>
    hipsparseStatus_t hipsparseXbsrsm2_analysis(hipsparseHandle_t         handle,
                                                hipsparseDirection_t      dirA,
                                                hipsparseOperation_t      transA,
                                                hipsparseOperation_t      transX,
                                                int                       mb,
                                                int                       nrhs,
                                                int                       nnzb,
                                                const hipsparseMatDescr_t descrA,
                                                const double*             bsrSortedValA,
                                                const int*                bsrSortedRowPtrA,
                                                const int*                bsrSortedColIndA,
                                                int                       blockDim,
                                                bsrsm2Info_t              info,
                                                hipsparseSolvePolicy_t    policy,
                                                void*                     pBuffer)
    {
        return hipsparseDbsrsm2_analysis(handle,
                                         dirA,
                                         transA,
                                         transX,
                                         mb,
                                         nrhs,
                                         nnzb,
                                         descrA,
                                         bsrSortedValA,
                                         bsrSortedRowPtrA,
                                         bsrSortedColIndA,
                                         blockDim,
                                         info,
                                         policy,
                                         pBuffer);
    }

    template <>
    hipsparseStatus_t hipsparseXbsrsm2_analysis(hipsparseHandle_t         handle,
                                                hipsparseDirection_t      dirA,
                                                hipsparseOperation_t      transA,
                                                hipsparseOperation_t      transX,
                                                int                       mb,
                                                int                       nrhs,
                                                int                       nnzb,
                                                const hipsparseMatDescr_t descrA,
                                                const hipComplex*         bsrSortedValA,
                                                const int*                bsrSortedRowPtrA,
                                                const int*                bsrSortedColIndA,
                                                int                       blockDim,
                                                bsrsm2Info_t              info,
                                                hipsparseSolvePolicy_t    policy,
                                                void*                     pBuffer)
    {
        return hipsparseCbsrsm2_analysis(handle,
                                         dirA,
                                         transA,
                                         transX,
                                         mb,
                                         nrhs,
                                         nnzb,
                                         descrA,
                                         bsrSortedValA,
                                         bsrSortedRowPtrA,
                                         bsrSortedColIndA,
                                         blockDim,
                                         info,
                                         policy,
                                         pBuffer);
    }

    template <>
    hipsparseStatus_t hipsparseXbsrsm2_analysis(hipsparseHandle_t         handle,
                                                hipsparseDirection_t      dirA,
                                                hipsparseOperation_t      transA,
                                                hipsparseOperation_t      transX,
                                                int                       mb,
                                                int                       nrhs,
                                                int                       nnzb,
                                                const hipsparseMatDescr_t descrA,
                                                const hipDoubleComplex*   bsrSortedValA,
                                                const int*                bsrSortedRowPtrA,
                                                const int*                bsrSortedColIndA,
                                                int                       blockDim,
                                                bsrsm2Info_t              info,
                                                hipsparseSolvePolicy_t    policy,
                                                void*                     pBuffer)
    {
        return hipsparseZbsrsm2_analysis(handle,
                                         dirA,
                                         transA,
                                         transX,
                                         mb,
                                         nrhs,
                                         nnzb,
                                         descrA,
                                         bsrSortedValA,
                                         bsrSortedRowPtrA,
                                         bsrSortedColIndA,
                                         blockDim,
                                         info,
                                         policy,
                                         pBuffer);
    }
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 13000)
    template <>
    hipsparseStatus_t hipsparseXbsrsm2_solve(hipsparseHandle_t         handle,
                                             hipsparseDirection_t      dirA,
                                             hipsparseOperation_t      transA,
                                             hipsparseOperation_t      transX,
                                             int                       mb,
                                             int                       nrhs,
                                             int                       nnzb,
                                             const float*              alpha,
                                             const hipsparseMatDescr_t descrA,
                                             const float*              bsrSortedValA,
                                             const int*                bsrSortedRowPtrA,
                                             const int*                bsrSortedColIndA,
                                             int                       blockDim,
                                             bsrsm2Info_t              info,
                                             const float*              B,
                                             int                       ldb,
                                             float*                    X,
                                             int                       ldx,
                                             hipsparseSolvePolicy_t    policy,
                                             void*                     pBuffer)
    {
        return hipsparseSbsrsm2_solve(handle,
                                      dirA,
                                      transA,
                                      transX,
                                      mb,
                                      nrhs,
                                      nnzb,
                                      alpha,
                                      descrA,
                                      bsrSortedValA,
                                      bsrSortedRowPtrA,
                                      bsrSortedColIndA,
                                      blockDim,
                                      info,
                                      B,
                                      ldb,
                                      X,
                                      ldx,
                                      policy,
                                      pBuffer);
    }

    template <>
    hipsparseStatus_t hipsparseXbsrsm2_solve(hipsparseHandle_t         handle,
                                             hipsparseDirection_t      dirA,
                                             hipsparseOperation_t      transA,
                                             hipsparseOperation_t      transX,
                                             int                       mb,
                                             int                       nrhs,
                                             int                       nnzb,
                                             const double*             alpha,
                                             const hipsparseMatDescr_t descrA,
                                             const double*             bsrSortedValA,
                                             const int*                bsrSortedRowPtrA,
                                             const int*                bsrSortedColIndA,
                                             int                       blockDim,
                                             bsrsm2Info_t              info,
                                             const double*             B,
                                             int                       ldb,
                                             double*                   X,
                                             int                       ldx,
                                             hipsparseSolvePolicy_t    policy,
                                             void*                     pBuffer)
    {
        return hipsparseDbsrsm2_solve(handle,
                                      dirA,
                                      transA,
                                      transX,
                                      mb,
                                      nrhs,
                                      nnzb,
                                      alpha,
                                      descrA,
                                      bsrSortedValA,
                                      bsrSortedRowPtrA,
                                      bsrSortedColIndA,
                                      blockDim,
                                      info,
                                      B,
                                      ldb,
                                      X,
                                      ldx,
                                      policy,
                                      pBuffer);
    }

    template <>
    hipsparseStatus_t hipsparseXbsrsm2_solve(hipsparseHandle_t         handle,
                                             hipsparseDirection_t      dirA,
                                             hipsparseOperation_t      transA,
                                             hipsparseOperation_t      transX,
                                             int                       mb,
                                             int                       nrhs,
                                             int                       nnzb,
                                             const hipComplex*         alpha,
                                             const hipsparseMatDescr_t descrA,
                                             const hipComplex*         bsrSortedValA,
                                             const int*                bsrSortedRowPtrA,
                                             const int*                bsrSortedColIndA,
                                             int                       blockDim,
                                             bsrsm2Info_t              info,
                                             const hipComplex*         B,
                                             int                       ldb,
                                             hipComplex*               X,
                                             int                       ldx,
                                             hipsparseSolvePolicy_t    policy,
                                             void*                     pBuffer)
    {
        return hipsparseCbsrsm2_solve(handle,
                                      dirA,
                                      transA,
                                      transX,
                                      mb,
                                      nrhs,
                                      nnzb,
                                      alpha,
                                      descrA,
                                      bsrSortedValA,
                                      bsrSortedRowPtrA,
                                      bsrSortedColIndA,
                                      blockDim,
                                      info,
                                      B,
                                      ldb,
                                      X,
                                      ldx,
                                      policy,
                                      pBuffer);
    }

    template <>
    hipsparseStatus_t hipsparseXbsrsm2_solve(hipsparseHandle_t         handle,
                                             hipsparseDirection_t      dirA,
                                             hipsparseOperation_t      transA,
                                             hipsparseOperation_t      transX,
                                             int                       mb,
                                             int                       nrhs,
                                             int                       nnzb,
                                             const hipDoubleComplex*   alpha,
                                             const hipsparseMatDescr_t descrA,
                                             const hipDoubleComplex*   bsrSortedValA,
                                             const int*                bsrSortedRowPtrA,
                                             const int*                bsrSortedColIndA,
                                             int                       blockDim,
                                             bsrsm2Info_t              info,
                                             const hipDoubleComplex*   B,
                                             int                       ldb,
                                             hipDoubleComplex*         X,
                                             int                       ldx,
                                             hipsparseSolvePolicy_t    policy,
                                             void*                     pBuffer)
    {
        return hipsparseZbsrsm2_solve(handle,
                                      dirA,
                                      transA,
                                      transX,
                                      mb,
                                      nrhs,
                                      nnzb,
                                      alpha,
                                      descrA,
                                      bsrSortedValA,
                                      bsrSortedRowPtrA,
                                      bsrSortedColIndA,
                                      blockDim,
                                      info,
                                      B,
                                      ldb,
                                      X,
                                      ldx,
                                      policy,
                                      pBuffer);
    }
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 12000)
    template <>
    hipsparseStatus_t hipsparseXcsrsm2_bufferSizeExt(hipsparseHandle_t         handle,
                                                     int                       algo,
                                                     hipsparseOperation_t      transA,
                                                     hipsparseOperation_t      transB,
                                                     int                       m,
                                                     int                       nrhs,
                                                     int                       nnz,
                                                     const float*              alpha,
                                                     const hipsparseMatDescr_t descrA,
                                                     const float*              csrSortedValA,
                                                     const int*                csrSortedRowPtrA,
                                                     const int*                csrSortedColIndA,
                                                     const float*              B,
                                                     int                       ldb,
                                                     csrsm2Info_t              info,
                                                     hipsparseSolvePolicy_t    policy,
                                                     size_t*                   pBufferSizeInBytes)
    {
        return hipsparseScsrsm2_bufferSizeExt(handle,
                                              algo,
                                              transA,
                                              transB,
                                              m,
                                              nrhs,
                                              nnz,
                                              alpha,
                                              descrA,
                                              csrSortedValA,
                                              csrSortedRowPtrA,
                                              csrSortedColIndA,
                                              B,
                                              ldb,
                                              info,
                                              policy,
                                              pBufferSizeInBytes);
    }

    template <>
    hipsparseStatus_t hipsparseXcsrsm2_bufferSizeExt(hipsparseHandle_t         handle,
                                                     int                       algo,
                                                     hipsparseOperation_t      transA,
                                                     hipsparseOperation_t      transB,
                                                     int                       m,
                                                     int                       nrhs,
                                                     int                       nnz,
                                                     const double*             alpha,
                                                     const hipsparseMatDescr_t descrA,
                                                     const double*             csrSortedValA,
                                                     const int*                csrSortedRowPtrA,
                                                     const int*                csrSortedColIndA,
                                                     const double*             B,
                                                     int                       ldb,
                                                     csrsm2Info_t              info,
                                                     hipsparseSolvePolicy_t    policy,
                                                     size_t*                   pBufferSizeInBytes)
    {
        return hipsparseDcsrsm2_bufferSizeExt(handle,
                                              algo,
                                              transA,
                                              transB,
                                              m,
                                              nrhs,
                                              nnz,
                                              alpha,
                                              descrA,
                                              csrSortedValA,
                                              csrSortedRowPtrA,
                                              csrSortedColIndA,
                                              B,
                                              ldb,
                                              info,
                                              policy,
                                              pBufferSizeInBytes);
    }

    template <>
    hipsparseStatus_t hipsparseXcsrsm2_bufferSizeExt(hipsparseHandle_t         handle,
                                                     int                       algo,
                                                     hipsparseOperation_t      transA,
                                                     hipsparseOperation_t      transB,
                                                     int                       m,
                                                     int                       nrhs,
                                                     int                       nnz,
                                                     const hipComplex*         alpha,
                                                     const hipsparseMatDescr_t descrA,
                                                     const hipComplex*         csrSortedValA,
                                                     const int*                csrSortedRowPtrA,
                                                     const int*                csrSortedColIndA,
                                                     const hipComplex*         B,
                                                     int                       ldb,
                                                     csrsm2Info_t              info,
                                                     hipsparseSolvePolicy_t    policy,
                                                     size_t*                   pBufferSizeInBytes)
    {
        return hipsparseCcsrsm2_bufferSizeExt(handle,
                                              algo,
                                              transA,
                                              transB,
                                              m,
                                              nrhs,
                                              nnz,
                                              alpha,
                                              descrA,
                                              csrSortedValA,
                                              csrSortedRowPtrA,
                                              csrSortedColIndA,
                                              B,
                                              ldb,
                                              info,
                                              policy,
                                              pBufferSizeInBytes);
    }

    template <>
    hipsparseStatus_t hipsparseXcsrsm2_bufferSizeExt(hipsparseHandle_t         handle,
                                                     int                       algo,
                                                     hipsparseOperation_t      transA,
                                                     hipsparseOperation_t      transB,
                                                     int                       m,
                                                     int                       nrhs,
                                                     int                       nnz,
                                                     const hipDoubleComplex*   alpha,
                                                     const hipsparseMatDescr_t descrA,
                                                     const hipDoubleComplex*   csrSortedValA,
                                                     const int*                csrSortedRowPtrA,
                                                     const int*                csrSortedColIndA,
                                                     const hipDoubleComplex*   B,
                                                     int                       ldb,
                                                     csrsm2Info_t              info,
                                                     hipsparseSolvePolicy_t    policy,
                                                     size_t*                   pBufferSizeInBytes)
    {
        return hipsparseZcsrsm2_bufferSizeExt(handle,
                                              algo,
                                              transA,
                                              transB,
                                              m,
                                              nrhs,
                                              nnz,
                                              alpha,
                                              descrA,
                                              csrSortedValA,
                                              csrSortedRowPtrA,
                                              csrSortedColIndA,
                                              B,
                                              ldb,
                                              info,
                                              policy,
                                              pBufferSizeInBytes);
    }
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 12000)
    template <>
    hipsparseStatus_t hipsparseXcsrsm2_analysis(hipsparseHandle_t         handle,
                                                int                       algo,
                                                hipsparseOperation_t      transA,
                                                hipsparseOperation_t      transB,
                                                int                       m,
                                                int                       nrhs,
                                                int                       nnz,
                                                const float*              alpha,
                                                const hipsparseMatDescr_t descrA,
                                                const float*              csrSortedValA,
                                                const int*                csrSortedRowPtrA,
                                                const int*                csrSortedColIndA,
                                                const float*              B,
                                                int                       ldb,
                                                csrsm2Info_t              info,
                                                hipsparseSolvePolicy_t    policy,
                                                void*                     pBuffer)
    {
        return hipsparseScsrsm2_analysis(handle,
                                         algo,
                                         transA,
                                         transB,
                                         m,
                                         nrhs,
                                         nnz,
                                         alpha,
                                         descrA,
                                         csrSortedValA,
                                         csrSortedRowPtrA,
                                         csrSortedColIndA,
                                         B,
                                         ldb,
                                         info,
                                         policy,
                                         pBuffer);
    }

    template <>
    hipsparseStatus_t hipsparseXcsrsm2_analysis(hipsparseHandle_t         handle,
                                                int                       algo,
                                                hipsparseOperation_t      transA,
                                                hipsparseOperation_t      transB,
                                                int                       m,
                                                int                       nrhs,
                                                int                       nnz,
                                                const double*             alpha,
                                                const hipsparseMatDescr_t descrA,
                                                const double*             csrSortedValA,
                                                const int*                csrSortedRowPtrA,
                                                const int*                csrSortedColIndA,
                                                const double*             B,
                                                int                       ldb,
                                                csrsm2Info_t              info,
                                                hipsparseSolvePolicy_t    policy,
                                                void*                     pBuffer)
    {
        return hipsparseDcsrsm2_analysis(handle,
                                         algo,
                                         transA,
                                         transB,
                                         m,
                                         nrhs,
                                         nnz,
                                         alpha,
                                         descrA,
                                         csrSortedValA,
                                         csrSortedRowPtrA,
                                         csrSortedColIndA,
                                         B,
                                         ldb,
                                         info,
                                         policy,
                                         pBuffer);
    }

    template <>
    hipsparseStatus_t hipsparseXcsrsm2_analysis(hipsparseHandle_t         handle,
                                                int                       algo,
                                                hipsparseOperation_t      transA,
                                                hipsparseOperation_t      transB,
                                                int                       m,
                                                int                       nrhs,
                                                int                       nnz,
                                                const hipComplex*         alpha,
                                                const hipsparseMatDescr_t descrA,
                                                const hipComplex*         csrSortedValA,
                                                const int*                csrSortedRowPtrA,
                                                const int*                csrSortedColIndA,
                                                const hipComplex*         B,
                                                int                       ldb,
                                                csrsm2Info_t              info,
                                                hipsparseSolvePolicy_t    policy,
                                                void*                     pBuffer)
    {
        return hipsparseCcsrsm2_analysis(handle,
                                         algo,
                                         transA,
                                         transB,
                                         m,
                                         nrhs,
                                         nnz,
                                         alpha,
                                         descrA,
                                         csrSortedValA,
                                         csrSortedRowPtrA,
                                         csrSortedColIndA,
                                         B,
                                         ldb,
                                         info,
                                         policy,
                                         pBuffer);
    }

    template <>
    hipsparseStatus_t hipsparseXcsrsm2_analysis(hipsparseHandle_t         handle,
                                                int                       algo,
                                                hipsparseOperation_t      transA,
                                                hipsparseOperation_t      transB,
                                                int                       m,
                                                int                       nrhs,
                                                int                       nnz,
                                                const hipDoubleComplex*   alpha,
                                                const hipsparseMatDescr_t descrA,
                                                const hipDoubleComplex*   csrSortedValA,
                                                const int*                csrSortedRowPtrA,
                                                const int*                csrSortedColIndA,
                                                const hipDoubleComplex*   B,
                                                int                       ldb,
                                                csrsm2Info_t              info,
                                                hipsparseSolvePolicy_t    policy,
                                                void*                     pBuffer)
    {
        return hipsparseZcsrsm2_analysis(handle,
                                         algo,
                                         transA,
                                         transB,
                                         m,
                                         nrhs,
                                         nnz,
                                         alpha,
                                         descrA,
                                         csrSortedValA,
                                         csrSortedRowPtrA,
                                         csrSortedColIndA,
                                         B,
                                         ldb,
                                         info,
                                         policy,
                                         pBuffer);
    }
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 12000)
    template <>
    hipsparseStatus_t hipsparseXcsrsm2_solve(hipsparseHandle_t         handle,
                                             int                       algo,
                                             hipsparseOperation_t      transA,
                                             hipsparseOperation_t      transB,
                                             int                       m,
                                             int                       nrhs,
                                             int                       nnz,
                                             const float*              alpha,
                                             const hipsparseMatDescr_t descrA,
                                             const float*              csrSortedValA,
                                             const int*                csrSortedRowPtrA,
                                             const int*                csrSortedColIndA,
                                             float*                    B,
                                             int                       ldb,
                                             csrsm2Info_t              info,
                                             hipsparseSolvePolicy_t    policy,
                                             void*                     pBuffer)
    {
        return hipsparseScsrsm2_solve(handle,
                                      algo,
                                      transA,
                                      transB,
                                      m,
                                      nrhs,
                                      nnz,
                                      alpha,
                                      descrA,
                                      csrSortedValA,
                                      csrSortedRowPtrA,
                                      csrSortedColIndA,
                                      B,
                                      ldb,
                                      info,
                                      policy,
                                      pBuffer);
    }

    template <>
    hipsparseStatus_t hipsparseXcsrsm2_solve(hipsparseHandle_t         handle,
                                             int                       algo,
                                             hipsparseOperation_t      transA,
                                             hipsparseOperation_t      transB,
                                             int                       m,
                                             int                       nrhs,
                                             int                       nnz,
                                             const double*             alpha,
                                             const hipsparseMatDescr_t descrA,
                                             const double*             csrSortedValA,
                                             const int*                csrSortedRowPtrA,
                                             const int*                csrSortedColIndA,
                                             double*                   B,
                                             int                       ldb,
                                             csrsm2Info_t              info,
                                             hipsparseSolvePolicy_t    policy,
                                             void*                     pBuffer)
    {
        return hipsparseDcsrsm2_solve(handle,
                                      algo,
                                      transA,
                                      transB,
                                      m,
                                      nrhs,
                                      nnz,
                                      alpha,
                                      descrA,
                                      csrSortedValA,
                                      csrSortedRowPtrA,
                                      csrSortedColIndA,
                                      B,
                                      ldb,
                                      info,
                                      policy,
                                      pBuffer);
    }

    template <>
    hipsparseStatus_t hipsparseXcsrsm2_solve(hipsparseHandle_t         handle,
                                             int                       algo,
                                             hipsparseOperation_t      transA,
                                             hipsparseOperation_t      transB,
                                             int                       m,
                                             int                       nrhs,
                                             int                       nnz,
                                             const hipComplex*         alpha,
                                             const hipsparseMatDescr_t descrA,
                                             const hipComplex*         csrSortedValA,
                                             const int*                csrSortedRowPtrA,
                                             const int*                csrSortedColIndA,
                                             hipComplex*               B,
                                             int                       ldb,
                                             csrsm2Info_t              info,
                                             hipsparseSolvePolicy_t    policy,
                                             void*                     pBuffer)
    {
        return hipsparseCcsrsm2_solve(handle,
                                      algo,
                                      transA,
                                      transB,
                                      m,
                                      nrhs,
                                      nnz,
                                      alpha,
                                      descrA,
                                      csrSortedValA,
                                      csrSortedRowPtrA,
                                      csrSortedColIndA,
                                      B,
                                      ldb,
                                      info,
                                      policy,
                                      pBuffer);
    }

    template <>
    hipsparseStatus_t hipsparseXcsrsm2_solve(hipsparseHandle_t         handle,
                                             int                       algo,
                                             hipsparseOperation_t      transA,
                                             hipsparseOperation_t      transB,
                                             int                       m,
                                             int                       nrhs,
                                             int                       nnz,
                                             const hipDoubleComplex*   alpha,
                                             const hipsparseMatDescr_t descrA,
                                             const hipDoubleComplex*   csrSortedValA,
                                             const int*                csrSortedRowPtrA,
                                             const int*                csrSortedColIndA,
                                             hipDoubleComplex*         B,
                                             int                       ldb,
                                             csrsm2Info_t              info,
                                             hipsparseSolvePolicy_t    policy,
                                             void*                     pBuffer)
    {
        return hipsparseZcsrsm2_solve(handle,
                                      algo,
                                      transA,
                                      transB,
                                      m,
                                      nrhs,
                                      nnz,
                                      alpha,
                                      descrA,
                                      csrSortedValA,
                                      csrSortedRowPtrA,
                                      csrSortedColIndA,
                                      B,
                                      ldb,
                                      info,
                                      policy,
                                      pBuffer);
    }
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 12000)
    template <>
    hipsparseStatus_t hipsparseXgemmi(hipsparseHandle_t handle,
                                      int               m,
                                      int               n,
                                      int               k,
                                      int               nnz,
                                      const float*      alpha,
                                      const float*      A,
                                      int               lda,
                                      const float*      cscValB,
                                      const int*        cscColPtrB,
                                      const int*        cscRowIndB,
                                      const float*      beta,
                                      float*            C,
                                      int               ldc)
    {
        return hipsparseSgemmi(
            handle, m, n, k, nnz, alpha, A, lda, cscValB, cscColPtrB, cscRowIndB, beta, C, ldc);
    }

    template <>
    hipsparseStatus_t hipsparseXgemmi(hipsparseHandle_t handle,
                                      int               m,
                                      int               n,
                                      int               k,
                                      int               nnz,
                                      const double*     alpha,
                                      const double*     A,
                                      int               lda,
                                      const double*     cscValB,
                                      const int*        cscColPtrB,
                                      const int*        cscRowIndB,
                                      const double*     beta,
                                      double*           C,
                                      int               ldc)
    {
        return hipsparseDgemmi(
            handle, m, n, k, nnz, alpha, A, lda, cscValB, cscColPtrB, cscRowIndB, beta, C, ldc);
    }

    template <>
    hipsparseStatus_t hipsparseXgemmi(hipsparseHandle_t handle,
                                      int               m,
                                      int               n,
                                      int               k,
                                      int               nnz,
                                      const hipComplex* alpha,
                                      const hipComplex* A,
                                      int               lda,
                                      const hipComplex* cscValB,
                                      const int*        cscColPtrB,
                                      const int*        cscRowIndB,
                                      const hipComplex* beta,
                                      hipComplex*       C,
                                      int               ldc)
    {
        return hipsparseCgemmi(
            handle, m, n, k, nnz, alpha, A, lda, cscValB, cscColPtrB, cscRowIndB, beta, C, ldc);
    }

    template <>
    hipsparseStatus_t hipsparseXgemmi(hipsparseHandle_t       handle,
                                      int                     m,
                                      int                     n,
                                      int                     k,
                                      int                     nnz,
                                      const hipDoubleComplex* alpha,
                                      const hipDoubleComplex* A,
                                      int                     lda,
                                      const hipDoubleComplex* cscValB,
                                      const int*              cscColPtrB,
                                      const int*              cscRowIndB,
                                      const hipDoubleComplex* beta,
                                      hipDoubleComplex*       C,
                                      int                     ldc)
    {
        return hipsparseZgemmi(
            handle, m, n, k, nnz, alpha, A, lda, cscValB, cscColPtrB, cscRowIndB, beta, C, ldc);
    }
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 11000)
    template <>
    hipsparseStatus_t hipsparseXcsrgeam(hipsparseHandle_t         handle,
                                        int                       m,
                                        int                       n,
                                        const float*              alpha,
                                        const hipsparseMatDescr_t descrA,
                                        int                       nnzA,
                                        const float*              csrSortedValA,
                                        const int*                csrSortedRowPtrA,
                                        const int*                csrSortedColIndA,
                                        const float*              beta,
                                        const hipsparseMatDescr_t descrB,
                                        int                       nnzB,
                                        const float*              csrSortedValB,
                                        const int*                csrSortedRowPtrB,
                                        const int*                csrSortedColIndB,
                                        const hipsparseMatDescr_t descrC,
                                        float*                    csrSortedValC,
                                        int*                      csrSortedRowPtrC,
                                        int*                      csrSortedColIndC)
    {
        return hipsparseScsrgeam(handle,
                                 m,
                                 n,
                                 alpha,
                                 descrA,
                                 nnzA,
                                 csrSortedValA,
                                 csrSortedRowPtrA,
                                 csrSortedColIndA,
                                 beta,
                                 descrB,
                                 nnzB,
                                 csrSortedValB,
                                 csrSortedRowPtrB,
                                 csrSortedColIndB,
                                 descrC,
                                 csrSortedValC,
                                 csrSortedRowPtrC,
                                 csrSortedColIndC);
    }

    template <>
    hipsparseStatus_t hipsparseXcsrgeam(hipsparseHandle_t         handle,
                                        int                       m,
                                        int                       n,
                                        const double*             alpha,
                                        const hipsparseMatDescr_t descrA,
                                        int                       nnzA,
                                        const double*             csrSortedValA,
                                        const int*                csrSortedRowPtrA,
                                        const int*                csrSortedColIndA,
                                        const double*             beta,
                                        const hipsparseMatDescr_t descrB,
                                        int                       nnzB,
                                        const double*             csrSortedValB,
                                        const int*                csrSortedRowPtrB,
                                        const int*                csrSortedColIndB,
                                        const hipsparseMatDescr_t descrC,
                                        double*                   csrSortedValC,
                                        int*                      csrSortedRowPtrC,
                                        int*                      csrSortedColIndC)
    {
        return hipsparseDcsrgeam(handle,
                                 m,
                                 n,
                                 alpha,
                                 descrA,
                                 nnzA,
                                 csrSortedValA,
                                 csrSortedRowPtrA,
                                 csrSortedColIndA,
                                 beta,
                                 descrB,
                                 nnzB,
                                 csrSortedValB,
                                 csrSortedRowPtrB,
                                 csrSortedColIndB,
                                 descrC,
                                 csrSortedValC,
                                 csrSortedRowPtrC,
                                 csrSortedColIndC);
    }

    template <>
    hipsparseStatus_t hipsparseXcsrgeam(hipsparseHandle_t         handle,
                                        int                       m,
                                        int                       n,
                                        const hipComplex*         alpha,
                                        const hipsparseMatDescr_t descrA,
                                        int                       nnzA,
                                        const hipComplex*         csrSortedValA,
                                        const int*                csrSortedRowPtrA,
                                        const int*                csrSortedColIndA,
                                        const hipComplex*         beta,
                                        const hipsparseMatDescr_t descrB,
                                        int                       nnzB,
                                        const hipComplex*         csrSortedValB,
                                        const int*                csrSortedRowPtrB,
                                        const int*                csrSortedColIndB,
                                        const hipsparseMatDescr_t descrC,
                                        hipComplex*               csrSortedValC,
                                        int*                      csrSortedRowPtrC,
                                        int*                      csrSortedColIndC)
    {
        return hipsparseCcsrgeam(handle,
                                 m,
                                 n,
                                 alpha,
                                 descrA,
                                 nnzA,
                                 csrSortedValA,
                                 csrSortedRowPtrA,
                                 csrSortedColIndA,
                                 beta,
                                 descrB,
                                 nnzB,
                                 csrSortedValB,
                                 csrSortedRowPtrB,
                                 csrSortedColIndB,
                                 descrC,
                                 csrSortedValC,
                                 csrSortedRowPtrC,
                                 csrSortedColIndC);
    }

    template <>
    hipsparseStatus_t hipsparseXcsrgeam(hipsparseHandle_t         handle,
                                        int                       m,
                                        int                       n,
                                        const hipDoubleComplex*   alpha,
                                        const hipsparseMatDescr_t descrA,
                                        int                       nnzA,
                                        const hipDoubleComplex*   csrSortedValA,
                                        const int*                csrSortedRowPtrA,
                                        const int*                csrSortedColIndA,
                                        const hipDoubleComplex*   beta,
                                        const hipsparseMatDescr_t descrB,
                                        int                       nnzB,
                                        const hipDoubleComplex*   csrSortedValB,
                                        const int*                csrSortedRowPtrB,
                                        const int*                csrSortedColIndB,
                                        const hipsparseMatDescr_t descrC,
                                        hipDoubleComplex*         csrSortedValC,
                                        int*                      csrSortedRowPtrC,
                                        int*                      csrSortedColIndC)
    {
        return hipsparseZcsrgeam(handle,
                                 m,
                                 n,
                                 alpha,
                                 descrA,
                                 nnzA,
                                 csrSortedValA,
                                 csrSortedRowPtrA,
                                 csrSortedColIndA,
                                 beta,
                                 descrB,
                                 nnzB,
                                 csrSortedValB,
                                 csrSortedRowPtrB,
                                 csrSortedColIndB,
                                 descrC,
                                 csrSortedValC,
                                 csrSortedRowPtrC,
                                 csrSortedColIndC);
    }
#endif

    template <>
    hipsparseStatus_t hipsparseXcsrgeam2_bufferSizeExt(hipsparseHandle_t         handle,
                                                       int                       m,
                                                       int                       n,
                                                       const float*              alpha,
                                                       const hipsparseMatDescr_t descrA,
                                                       int                       nnzA,
                                                       const float*              csrSortedValA,
                                                       const int*                csrSortedRowPtrA,
                                                       const int*                csrSortedColIndA,
                                                       const float*              beta,
                                                       const hipsparseMatDescr_t descrB,
                                                       int                       nnzB,
                                                       const float*              csrSortedValB,
                                                       const int*                csrSortedRowPtrB,
                                                       const int*                csrSortedColIndB,
                                                       const hipsparseMatDescr_t descrC,
                                                       const float*              csrSortedValC,
                                                       const int*                csrSortedRowPtrC,
                                                       const int*                csrSortedColIndC,
                                                       size_t*                   pBufferSizeInBytes)
    {
        return hipsparseScsrgeam2_bufferSizeExt(handle,
                                                m,
                                                n,
                                                alpha,
                                                descrA,
                                                nnzA,
                                                csrSortedValA,
                                                csrSortedRowPtrA,
                                                csrSortedColIndA,
                                                beta,
                                                descrB,
                                                nnzB,
                                                csrSortedValB,
                                                csrSortedRowPtrB,
                                                csrSortedColIndB,
                                                descrC,
                                                csrSortedValC,
                                                csrSortedRowPtrC,
                                                csrSortedColIndC,
                                                pBufferSizeInBytes);
    }

    template <>
    hipsparseStatus_t hipsparseXcsrgeam2_bufferSizeExt(hipsparseHandle_t         handle,
                                                       int                       m,
                                                       int                       n,
                                                       const double*             alpha,
                                                       const hipsparseMatDescr_t descrA,
                                                       int                       nnzA,
                                                       const double*             csrSortedValA,
                                                       const int*                csrSortedRowPtrA,
                                                       const int*                csrSortedColIndA,
                                                       const double*             beta,
                                                       const hipsparseMatDescr_t descrB,
                                                       int                       nnzB,
                                                       const double*             csrSortedValB,
                                                       const int*                csrSortedRowPtrB,
                                                       const int*                csrSortedColIndB,
                                                       const hipsparseMatDescr_t descrC,
                                                       const double*             csrSortedValC,
                                                       const int*                csrSortedRowPtrC,
                                                       const int*                csrSortedColIndC,
                                                       size_t*                   pBufferSizeInBytes)
    {
        return hipsparseDcsrgeam2_bufferSizeExt(handle,
                                                m,
                                                n,
                                                alpha,
                                                descrA,
                                                nnzA,
                                                csrSortedValA,
                                                csrSortedRowPtrA,
                                                csrSortedColIndA,
                                                beta,
                                                descrB,
                                                nnzB,
                                                csrSortedValB,
                                                csrSortedRowPtrB,
                                                csrSortedColIndB,
                                                descrC,
                                                csrSortedValC,
                                                csrSortedRowPtrC,
                                                csrSortedColIndC,
                                                pBufferSizeInBytes);
    }

    template <>
    hipsparseStatus_t hipsparseXcsrgeam2_bufferSizeExt(hipsparseHandle_t         handle,
                                                       int                       m,
                                                       int                       n,
                                                       const hipComplex*         alpha,
                                                       const hipsparseMatDescr_t descrA,
                                                       int                       nnzA,
                                                       const hipComplex*         csrSortedValA,
                                                       const int*                csrSortedRowPtrA,
                                                       const int*                csrSortedColIndA,
                                                       const hipComplex*         beta,
                                                       const hipsparseMatDescr_t descrB,
                                                       int                       nnzB,
                                                       const hipComplex*         csrSortedValB,
                                                       const int*                csrSortedRowPtrB,
                                                       const int*                csrSortedColIndB,
                                                       const hipsparseMatDescr_t descrC,
                                                       const hipComplex*         csrSortedValC,
                                                       const int*                csrSortedRowPtrC,
                                                       const int*                csrSortedColIndC,
                                                       size_t*                   pBufferSizeInBytes)
    {
        return hipsparseCcsrgeam2_bufferSizeExt(handle,
                                                m,
                                                n,
                                                alpha,
                                                descrA,
                                                nnzA,
                                                csrSortedValA,
                                                csrSortedRowPtrA,
                                                csrSortedColIndA,
                                                beta,
                                                descrB,
                                                nnzB,
                                                csrSortedValB,
                                                csrSortedRowPtrB,
                                                csrSortedColIndB,
                                                descrC,
                                                csrSortedValC,
                                                csrSortedRowPtrC,
                                                csrSortedColIndC,
                                                pBufferSizeInBytes);
    }

    template <>
    hipsparseStatus_t hipsparseXcsrgeam2_bufferSizeExt(hipsparseHandle_t         handle,
                                                       int                       m,
                                                       int                       n,
                                                       const hipDoubleComplex*   alpha,
                                                       const hipsparseMatDescr_t descrA,
                                                       int                       nnzA,
                                                       const hipDoubleComplex*   csrSortedValA,
                                                       const int*                csrSortedRowPtrA,
                                                       const int*                csrSortedColIndA,
                                                       const hipDoubleComplex*   beta,
                                                       const hipsparseMatDescr_t descrB,
                                                       int                       nnzB,
                                                       const hipDoubleComplex*   csrSortedValB,
                                                       const int*                csrSortedRowPtrB,
                                                       const int*                csrSortedColIndB,
                                                       const hipsparseMatDescr_t descrC,
                                                       const hipDoubleComplex*   csrSortedValC,
                                                       const int*                csrSortedRowPtrC,
                                                       const int*                csrSortedColIndC,
                                                       size_t*                   pBufferSizeInBytes)
    {
        return hipsparseZcsrgeam2_bufferSizeExt(handle,
                                                m,
                                                n,
                                                alpha,
                                                descrA,
                                                nnzA,
                                                csrSortedValA,
                                                csrSortedRowPtrA,
                                                csrSortedColIndA,
                                                beta,
                                                descrB,
                                                nnzB,
                                                csrSortedValB,
                                                csrSortedRowPtrB,
                                                csrSortedColIndB,
                                                descrC,
                                                csrSortedValC,
                                                csrSortedRowPtrC,
                                                csrSortedColIndC,
                                                pBufferSizeInBytes);
    }

    template <>
    hipsparseStatus_t hipsparseXcsrgeam2(hipsparseHandle_t         handle,
                                         int                       m,
                                         int                       n,
                                         const float*              alpha,
                                         const hipsparseMatDescr_t descrA,
                                         int                       nnzA,
                                         const float*              csrSortedValA,
                                         const int*                csrSortedRowPtrA,
                                         const int*                csrSortedColIndA,
                                         const float*              beta,
                                         const hipsparseMatDescr_t descrB,
                                         int                       nnzB,
                                         const float*              csrSortedValB,
                                         const int*                csrSortedRowPtrB,
                                         const int*                csrSortedColIndB,
                                         const hipsparseMatDescr_t descrC,
                                         float*                    csrSortedValC,
                                         int*                      csrSortedRowPtrC,
                                         int*                      csrSortedColIndC,
                                         void*                     pBuffer)
    {
        return hipsparseScsrgeam2(handle,
                                  m,
                                  n,
                                  alpha,
                                  descrA,
                                  nnzA,
                                  csrSortedValA,
                                  csrSortedRowPtrA,
                                  csrSortedColIndA,
                                  beta,
                                  descrB,
                                  nnzB,
                                  csrSortedValB,
                                  csrSortedRowPtrB,
                                  csrSortedColIndB,
                                  descrC,
                                  csrSortedValC,
                                  csrSortedRowPtrC,
                                  csrSortedColIndC,
                                  pBuffer);
    }

    template <>
    hipsparseStatus_t hipsparseXcsrgeam2(hipsparseHandle_t         handle,
                                         int                       m,
                                         int                       n,
                                         const double*             alpha,
                                         const hipsparseMatDescr_t descrA,
                                         int                       nnzA,
                                         const double*             csrSortedValA,
                                         const int*                csrSortedRowPtrA,
                                         const int*                csrSortedColIndA,
                                         const double*             beta,
                                         const hipsparseMatDescr_t descrB,
                                         int                       nnzB,
                                         const double*             csrSortedValB,
                                         const int*                csrSortedRowPtrB,
                                         const int*                csrSortedColIndB,
                                         const hipsparseMatDescr_t descrC,
                                         double*                   csrSortedValC,
                                         int*                      csrSortedRowPtrC,
                                         int*                      csrSortedColIndC,
                                         void*                     pBuffer)
    {
        return hipsparseDcsrgeam2(handle,
                                  m,
                                  n,
                                  alpha,
                                  descrA,
                                  nnzA,
                                  csrSortedValA,
                                  csrSortedRowPtrA,
                                  csrSortedColIndA,
                                  beta,
                                  descrB,
                                  nnzB,
                                  csrSortedValB,
                                  csrSortedRowPtrB,
                                  csrSortedColIndB,
                                  descrC,
                                  csrSortedValC,
                                  csrSortedRowPtrC,
                                  csrSortedColIndC,
                                  pBuffer);
    }

    template <>
    hipsparseStatus_t hipsparseXcsrgeam2(hipsparseHandle_t         handle,
                                         int                       m,
                                         int                       n,
                                         const hipComplex*         alpha,
                                         const hipsparseMatDescr_t descrA,
                                         int                       nnzA,
                                         const hipComplex*         csrSortedValA,
                                         const int*                csrSortedRowPtrA,
                                         const int*                csrSortedColIndA,
                                         const hipComplex*         beta,
                                         const hipsparseMatDescr_t descrB,
                                         int                       nnzB,
                                         const hipComplex*         csrSortedValB,
                                         const int*                csrSortedRowPtrB,
                                         const int*                csrSortedColIndB,
                                         const hipsparseMatDescr_t descrC,
                                         hipComplex*               csrSortedValC,
                                         int*                      csrSortedRowPtrC,
                                         int*                      csrSortedColIndC,
                                         void*                     pBuffer)
    {
        return hipsparseCcsrgeam2(handle,
                                  m,
                                  n,
                                  alpha,
                                  descrA,
                                  nnzA,
                                  csrSortedValA,
                                  csrSortedRowPtrA,
                                  csrSortedColIndA,
                                  beta,
                                  descrB,
                                  nnzB,
                                  csrSortedValB,
                                  csrSortedRowPtrB,
                                  csrSortedColIndB,
                                  descrC,
                                  csrSortedValC,
                                  csrSortedRowPtrC,
                                  csrSortedColIndC,
                                  pBuffer);
    }

    template <>
    hipsparseStatus_t hipsparseXcsrgeam2(hipsparseHandle_t         handle,
                                         int                       m,
                                         int                       n,
                                         const hipDoubleComplex*   alpha,
                                         const hipsparseMatDescr_t descrA,
                                         int                       nnzA,
                                         const hipDoubleComplex*   csrSortedValA,
                                         const int*                csrSortedRowPtrA,
                                         const int*                csrSortedColIndA,
                                         const hipDoubleComplex*   beta,
                                         const hipsparseMatDescr_t descrB,
                                         int                       nnzB,
                                         const hipDoubleComplex*   csrSortedValB,
                                         const int*                csrSortedRowPtrB,
                                         const int*                csrSortedColIndB,
                                         const hipsparseMatDescr_t descrC,
                                         hipDoubleComplex*         csrSortedValC,
                                         int*                      csrSortedRowPtrC,
                                         int*                      csrSortedColIndC,
                                         void*                     pBuffer)
    {
        return hipsparseZcsrgeam2(handle,
                                  m,
                                  n,
                                  alpha,
                                  descrA,
                                  nnzA,
                                  csrSortedValA,
                                  csrSortedRowPtrA,
                                  csrSortedColIndA,
                                  beta,
                                  descrB,
                                  nnzB,
                                  csrSortedValB,
                                  csrSortedRowPtrB,
                                  csrSortedColIndB,
                                  descrC,
                                  csrSortedValC,
                                  csrSortedRowPtrC,
                                  csrSortedColIndC,
                                  pBuffer);
    }

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 11000)
    template <>
    hipsparseStatus_t hipsparseXcsrgemm(hipsparseHandle_t         handle,
                                        hipsparseOperation_t      transA,
                                        hipsparseOperation_t      transB,
                                        int                       m,
                                        int                       n,
                                        int                       k,
                                        const hipsparseMatDescr_t descrA,
                                        int                       nnzA,
                                        const float*              csrValA,
                                        const int*                csrRowPtrA,
                                        const int*                csrColIndA,
                                        const hipsparseMatDescr_t descrB,
                                        int                       nnzB,
                                        const float*              csrValB,
                                        const int*                csrRowPtrB,
                                        const int*                csrColIndB,
                                        const hipsparseMatDescr_t descrC,
                                        float*                    csrValC,
                                        const int*                csrRowPtrC,
                                        int*                      csrColIndC)
    {
        return hipsparseScsrgemm(handle,
                                 transA,
                                 transB,
                                 m,
                                 n,
                                 k,
                                 descrA,
                                 nnzA,
                                 csrValA,
                                 csrRowPtrA,
                                 csrColIndA,
                                 descrB,
                                 nnzB,
                                 csrValB,
                                 csrRowPtrB,
                                 csrColIndB,
                                 descrC,
                                 csrValC,
                                 csrRowPtrC,
                                 csrColIndC);
    }

    template <>
    hipsparseStatus_t hipsparseXcsrgemm(hipsparseHandle_t         handle,
                                        hipsparseOperation_t      transA,
                                        hipsparseOperation_t      transB,
                                        int                       m,
                                        int                       n,
                                        int                       k,
                                        const hipsparseMatDescr_t descrA,
                                        int                       nnzA,
                                        const double*             csrValA,
                                        const int*                csrRowPtrA,
                                        const int*                csrColIndA,
                                        const hipsparseMatDescr_t descrB,
                                        int                       nnzB,
                                        const double*             csrValB,
                                        const int*                csrRowPtrB,
                                        const int*                csrColIndB,
                                        const hipsparseMatDescr_t descrC,
                                        double*                   csrValC,
                                        const int*                csrRowPtrC,
                                        int*                      csrColIndC)
    {
        return hipsparseDcsrgemm(handle,
                                 transA,
                                 transB,
                                 m,
                                 n,
                                 k,
                                 descrA,
                                 nnzA,
                                 csrValA,
                                 csrRowPtrA,
                                 csrColIndA,
                                 descrB,
                                 nnzB,
                                 csrValB,
                                 csrRowPtrB,
                                 csrColIndB,
                                 descrC,
                                 csrValC,
                                 csrRowPtrC,
                                 csrColIndC);
    }

    template <>
    hipsparseStatus_t hipsparseXcsrgemm(hipsparseHandle_t         handle,
                                        hipsparseOperation_t      transA,
                                        hipsparseOperation_t      transB,
                                        int                       m,
                                        int                       n,
                                        int                       k,
                                        const hipsparseMatDescr_t descrA,
                                        int                       nnzA,
                                        const hipComplex*         csrValA,
                                        const int*                csrRowPtrA,
                                        const int*                csrColIndA,
                                        const hipsparseMatDescr_t descrB,
                                        int                       nnzB,
                                        const hipComplex*         csrValB,
                                        const int*                csrRowPtrB,
                                        const int*                csrColIndB,
                                        const hipsparseMatDescr_t descrC,
                                        hipComplex*               csrValC,
                                        const int*                csrRowPtrC,
                                        int*                      csrColIndC)
    {
        return hipsparseCcsrgemm(handle,
                                 transA,
                                 transB,
                                 m,
                                 n,
                                 k,
                                 descrA,
                                 nnzA,
                                 csrValA,
                                 csrRowPtrA,
                                 csrColIndA,
                                 descrB,
                                 nnzB,
                                 csrValB,
                                 csrRowPtrB,
                                 csrColIndB,
                                 descrC,
                                 csrValC,
                                 csrRowPtrC,
                                 csrColIndC);
    }

    template <>
    hipsparseStatus_t hipsparseXcsrgemm(hipsparseHandle_t         handle,
                                        hipsparseOperation_t      transA,
                                        hipsparseOperation_t      transB,
                                        int                       m,
                                        int                       n,
                                        int                       k,
                                        const hipsparseMatDescr_t descrA,
                                        int                       nnzA,
                                        const hipDoubleComplex*   csrValA,
                                        const int*                csrRowPtrA,
                                        const int*                csrColIndA,
                                        const hipsparseMatDescr_t descrB,
                                        int                       nnzB,
                                        const hipDoubleComplex*   csrValB,
                                        const int*                csrRowPtrB,
                                        const int*                csrColIndB,
                                        const hipsparseMatDescr_t descrC,
                                        hipDoubleComplex*         csrValC,
                                        const int*                csrRowPtrC,
                                        int*                      csrColIndC)
    {
        return hipsparseZcsrgemm(handle,
                                 transA,
                                 transB,
                                 m,
                                 n,
                                 k,
                                 descrA,
                                 nnzA,
                                 csrValA,
                                 csrRowPtrA,
                                 csrColIndA,
                                 descrB,
                                 nnzB,
                                 csrValB,
                                 csrRowPtrB,
                                 csrColIndB,
                                 descrC,
                                 csrValC,
                                 csrRowPtrC,
                                 csrColIndC);
    }
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 12000)
    template <>
    hipsparseStatus_t hipsparseXcsrgemm2_bufferSizeExt(hipsparseHandle_t         handle,
                                                       int                       m,
                                                       int                       n,
                                                       int                       k,
                                                       const float*              alpha,
                                                       const hipsparseMatDescr_t descrA,
                                                       int                       nnzA,
                                                       const int*                csrRowPtrA,
                                                       const int*                csrColIndA,
                                                       const hipsparseMatDescr_t descrB,
                                                       int                       nnzB,
                                                       const int*                csrRowPtrB,
                                                       const int*                csrColIndB,
                                                       const float*              beta,
                                                       const hipsparseMatDescr_t descrD,
                                                       int                       nnzD,
                                                       const int*                csrRowPtrD,
                                                       const int*                csrColIndD,
                                                       csrgemm2Info_t            info,
                                                       size_t*                   pBufferSizeInBytes)
    {
        return hipsparseScsrgemm2_bufferSizeExt(handle,
                                                m,
                                                n,
                                                k,
                                                alpha,
                                                descrA,
                                                nnzA,
                                                csrRowPtrA,
                                                csrColIndA,
                                                descrB,
                                                nnzB,
                                                csrRowPtrB,
                                                csrColIndB,
                                                beta,
                                                descrD,
                                                nnzD,
                                                csrRowPtrD,
                                                csrColIndD,
                                                info,
                                                pBufferSizeInBytes);
    }

    template <>
    hipsparseStatus_t hipsparseXcsrgemm2_bufferSizeExt(hipsparseHandle_t         handle,
                                                       int                       m,
                                                       int                       n,
                                                       int                       k,
                                                       const double*             alpha,
                                                       const hipsparseMatDescr_t descrA,
                                                       int                       nnzA,
                                                       const int*                csrRowPtrA,
                                                       const int*                csrColIndA,
                                                       const hipsparseMatDescr_t descrB,
                                                       int                       nnzB,
                                                       const int*                csrRowPtrB,
                                                       const int*                csrColIndB,
                                                       const double*             beta,
                                                       const hipsparseMatDescr_t descrD,
                                                       int                       nnzD,
                                                       const int*                csrRowPtrD,
                                                       const int*                csrColIndD,
                                                       csrgemm2Info_t            info,
                                                       size_t*                   pBufferSizeInBytes)
    {
        return hipsparseDcsrgemm2_bufferSizeExt(handle,
                                                m,
                                                n,
                                                k,
                                                alpha,
                                                descrA,
                                                nnzA,
                                                csrRowPtrA,
                                                csrColIndA,
                                                descrB,
                                                nnzB,
                                                csrRowPtrB,
                                                csrColIndB,
                                                beta,
                                                descrD,
                                                nnzD,
                                                csrRowPtrD,
                                                csrColIndD,
                                                info,
                                                pBufferSizeInBytes);
    }

    template <>
    hipsparseStatus_t hipsparseXcsrgemm2_bufferSizeExt(hipsparseHandle_t         handle,
                                                       int                       m,
                                                       int                       n,
                                                       int                       k,
                                                       const hipComplex*         alpha,
                                                       const hipsparseMatDescr_t descrA,
                                                       int                       nnzA,
                                                       const int*                csrRowPtrA,
                                                       const int*                csrColIndA,
                                                       const hipsparseMatDescr_t descrB,
                                                       int                       nnzB,
                                                       const int*                csrRowPtrB,
                                                       const int*                csrColIndB,
                                                       const hipComplex*         beta,
                                                       const hipsparseMatDescr_t descrD,
                                                       int                       nnzD,
                                                       const int*                csrRowPtrD,
                                                       const int*                csrColIndD,
                                                       csrgemm2Info_t            info,
                                                       size_t*                   pBufferSizeInBytes)
    {
        return hipsparseCcsrgemm2_bufferSizeExt(handle,
                                                m,
                                                n,
                                                k,
                                                alpha,
                                                descrA,
                                                nnzA,
                                                csrRowPtrA,
                                                csrColIndA,
                                                descrB,
                                                nnzB,
                                                csrRowPtrB,
                                                csrColIndB,
                                                beta,
                                                descrD,
                                                nnzD,
                                                csrRowPtrD,
                                                csrColIndD,
                                                info,
                                                pBufferSizeInBytes);
    }

    template <>
    hipsparseStatus_t hipsparseXcsrgemm2_bufferSizeExt(hipsparseHandle_t         handle,
                                                       int                       m,
                                                       int                       n,
                                                       int                       k,
                                                       const hipDoubleComplex*   alpha,
                                                       const hipsparseMatDescr_t descrA,
                                                       int                       nnzA,
                                                       const int*                csrRowPtrA,
                                                       const int*                csrColIndA,
                                                       const hipsparseMatDescr_t descrB,
                                                       int                       nnzB,
                                                       const int*                csrRowPtrB,
                                                       const int*                csrColIndB,
                                                       const hipDoubleComplex*   beta,
                                                       const hipsparseMatDescr_t descrD,
                                                       int                       nnzD,
                                                       const int*                csrRowPtrD,
                                                       const int*                csrColIndD,
                                                       csrgemm2Info_t            info,
                                                       size_t*                   pBufferSizeInBytes)
    {
        return hipsparseZcsrgemm2_bufferSizeExt(handle,
                                                m,
                                                n,
                                                k,
                                                alpha,
                                                descrA,
                                                nnzA,
                                                csrRowPtrA,
                                                csrColIndA,
                                                descrB,
                                                nnzB,
                                                csrRowPtrB,
                                                csrColIndB,
                                                beta,
                                                descrD,
                                                nnzD,
                                                csrRowPtrD,
                                                csrColIndD,
                                                info,
                                                pBufferSizeInBytes);
    }

    template <>
    hipsparseStatus_t hipsparseXcsrgemm2(hipsparseHandle_t         handle,
                                         int                       m,
                                         int                       n,
                                         int                       k,
                                         const float*              alpha,
                                         const hipsparseMatDescr_t descrA,
                                         int                       nnzA,
                                         const float*              csrValA,
                                         const int*                csrRowPtrA,
                                         const int*                csrColIndA,
                                         const hipsparseMatDescr_t descrB,
                                         int                       nnzB,
                                         const float*              csrValB,
                                         const int*                csrRowPtrB,
                                         const int*                csrColIndB,
                                         const float*              beta,
                                         const hipsparseMatDescr_t descrD,
                                         int                       nnzD,
                                         const float*              csrValD,
                                         const int*                csrRowPtrD,
                                         const int*                csrColIndD,
                                         const hipsparseMatDescr_t descrC,
                                         float*                    csrValC,
                                         const int*                csrRowPtrC,
                                         int*                      csrColIndC,
                                         const csrgemm2Info_t      info,
                                         void*                     pBuffer)
    {
        return hipsparseScsrgemm2(handle,
                                  m,
                                  n,
                                  k,
                                  alpha,
                                  descrA,
                                  nnzA,
                                  csrValA,
                                  csrRowPtrA,
                                  csrColIndA,
                                  descrB,
                                  nnzB,
                                  csrValB,
                                  csrRowPtrB,
                                  csrColIndB,
                                  beta,
                                  descrD,
                                  nnzD,
                                  csrValD,
                                  csrRowPtrD,
                                  csrColIndD,
                                  descrC,
                                  csrValC,
                                  csrRowPtrC,
                                  csrColIndC,
                                  info,
                                  pBuffer);
    }

    template <>
    hipsparseStatus_t hipsparseXcsrgemm2(hipsparseHandle_t         handle,
                                         int                       m,
                                         int                       n,
                                         int                       k,
                                         const double*             alpha,
                                         const hipsparseMatDescr_t descrA,
                                         int                       nnzA,
                                         const double*             csrValA,
                                         const int*                csrRowPtrA,
                                         const int*                csrColIndA,
                                         const hipsparseMatDescr_t descrB,
                                         int                       nnzB,
                                         const double*             csrValB,
                                         const int*                csrRowPtrB,
                                         const int*                csrColIndB,
                                         const double*             beta,
                                         const hipsparseMatDescr_t descrD,
                                         int                       nnzD,
                                         const double*             csrValD,
                                         const int*                csrRowPtrD,
                                         const int*                csrColIndD,
                                         const hipsparseMatDescr_t descrC,
                                         double*                   csrValC,
                                         const int*                csrRowPtrC,
                                         int*                      csrColIndC,
                                         const csrgemm2Info_t      info,
                                         void*                     pBuffer)
    {
        return hipsparseDcsrgemm2(handle,
                                  m,
                                  n,
                                  k,
                                  alpha,
                                  descrA,
                                  nnzA,
                                  csrValA,
                                  csrRowPtrA,
                                  csrColIndA,
                                  descrB,
                                  nnzB,
                                  csrValB,
                                  csrRowPtrB,
                                  csrColIndB,
                                  beta,
                                  descrD,
                                  nnzD,
                                  csrValD,
                                  csrRowPtrD,
                                  csrColIndD,
                                  descrC,
                                  csrValC,
                                  csrRowPtrC,
                                  csrColIndC,
                                  info,
                                  pBuffer);
    }

    template <>
    hipsparseStatus_t hipsparseXcsrgemm2(hipsparseHandle_t         handle,
                                         int                       m,
                                         int                       n,
                                         int                       k,
                                         const hipComplex*         alpha,
                                         const hipsparseMatDescr_t descrA,
                                         int                       nnzA,
                                         const hipComplex*         csrValA,
                                         const int*                csrRowPtrA,
                                         const int*                csrColIndA,
                                         const hipsparseMatDescr_t descrB,
                                         int                       nnzB,
                                         const hipComplex*         csrValB,
                                         const int*                csrRowPtrB,
                                         const int*                csrColIndB,
                                         const hipComplex*         beta,
                                         const hipsparseMatDescr_t descrD,
                                         int                       nnzD,
                                         const hipComplex*         csrValD,
                                         const int*                csrRowPtrD,
                                         const int*                csrColIndD,
                                         const hipsparseMatDescr_t descrC,
                                         hipComplex*               csrValC,
                                         const int*                csrRowPtrC,
                                         int*                      csrColIndC,
                                         const csrgemm2Info_t      info,
                                         void*                     pBuffer)
    {
        return hipsparseCcsrgemm2(handle,
                                  m,
                                  n,
                                  k,
                                  alpha,
                                  descrA,
                                  nnzA,
                                  csrValA,
                                  csrRowPtrA,
                                  csrColIndA,
                                  descrB,
                                  nnzB,
                                  csrValB,
                                  csrRowPtrB,
                                  csrColIndB,
                                  beta,
                                  descrD,
                                  nnzD,
                                  csrValD,
                                  csrRowPtrD,
                                  csrColIndD,
                                  descrC,
                                  csrValC,
                                  csrRowPtrC,
                                  csrColIndC,
                                  info,
                                  pBuffer);
    }

    template <>
    hipsparseStatus_t hipsparseXcsrgemm2(hipsparseHandle_t         handle,
                                         int                       m,
                                         int                       n,
                                         int                       k,
                                         const hipDoubleComplex*   alpha,
                                         const hipsparseMatDescr_t descrA,
                                         int                       nnzA,
                                         const hipDoubleComplex*   csrValA,
                                         const int*                csrRowPtrA,
                                         const int*                csrColIndA,
                                         const hipsparseMatDescr_t descrB,
                                         int                       nnzB,
                                         const hipDoubleComplex*   csrValB,
                                         const int*                csrRowPtrB,
                                         const int*                csrColIndB,
                                         const hipDoubleComplex*   beta,
                                         const hipsparseMatDescr_t descrD,
                                         int                       nnzD,
                                         const hipDoubleComplex*   csrValD,
                                         const int*                csrRowPtrD,
                                         const int*                csrColIndD,
                                         const hipsparseMatDescr_t descrC,
                                         hipDoubleComplex*         csrValC,
                                         const int*                csrRowPtrC,
                                         int*                      csrColIndC,
                                         const csrgemm2Info_t      info,
                                         void*                     pBuffer)
    {
        return hipsparseZcsrgemm2(handle,
                                  m,
                                  n,
                                  k,
                                  alpha,
                                  descrA,
                                  nnzA,
                                  csrValA,
                                  csrRowPtrA,
                                  csrColIndA,
                                  descrB,
                                  nnzB,
                                  csrValB,
                                  csrRowPtrB,
                                  csrColIndB,
                                  beta,
                                  descrD,
                                  nnzD,
                                  csrValD,
                                  csrRowPtrD,
                                  csrColIndD,
                                  descrC,
                                  csrValC,
                                  csrRowPtrC,
                                  csrColIndC,
                                  info,
                                  pBuffer);
    }
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 13000)
    template <>
    hipsparseStatus_t hipsparseXbsrilu02_numericBoost(hipsparseHandle_t handle,
                                                      bsrilu02Info_t    info,
                                                      int               enable_boost,
                                                      double*           tol,
                                                      float*            boost_val)
    {
        return hipsparseSbsrilu02_numericBoost(handle, info, enable_boost, tol, boost_val);
    }

    template <>
    hipsparseStatus_t hipsparseXbsrilu02_numericBoost(hipsparseHandle_t handle,
                                                      bsrilu02Info_t    info,
                                                      int               enable_boost,
                                                      double*           tol,
                                                      double*           boost_val)
    {
        return hipsparseDbsrilu02_numericBoost(handle, info, enable_boost, tol, boost_val);
    }

    template <>
    hipsparseStatus_t hipsparseXbsrilu02_numericBoost(hipsparseHandle_t handle,
                                                      bsrilu02Info_t    info,
                                                      int               enable_boost,
                                                      double*           tol,
                                                      hipComplex*       boost_val)
    {
        return hipsparseCbsrilu02_numericBoost(handle, info, enable_boost, tol, boost_val);
    }

    template <>
    hipsparseStatus_t hipsparseXbsrilu02_numericBoost(hipsparseHandle_t handle,
                                                      bsrilu02Info_t    info,
                                                      int               enable_boost,
                                                      double*           tol,
                                                      hipDoubleComplex* boost_val)
    {
        return hipsparseZbsrilu02_numericBoost(handle, info, enable_boost, tol, boost_val);
    }
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 13000)
    template <>
    hipsparseStatus_t hipsparseXbsrilu02_bufferSize(hipsparseHandle_t         handle,
                                                    hipsparseDirection_t      dirA,
                                                    int                       mb,
                                                    int                       nnzb,
                                                    const hipsparseMatDescr_t descrA,
                                                    float*                    bsrValA,
                                                    const int*                bsrRowPtrA,
                                                    const int*                bsrColIndA,
                                                    int                       blockDim,
                                                    bsrilu02Info_t            info,
                                                    int*                      pBufferSizeInBytes)
    {
        return hipsparseSbsrilu02_bufferSize(handle,
                                             dirA,
                                             mb,
                                             nnzb,
                                             descrA,
                                             bsrValA,
                                             bsrRowPtrA,
                                             bsrColIndA,
                                             blockDim,
                                             info,
                                             pBufferSizeInBytes);
    }

    template <>
    hipsparseStatus_t hipsparseXbsrilu02_bufferSize(hipsparseHandle_t         handle,
                                                    hipsparseDirection_t      dirA,
                                                    int                       mb,
                                                    int                       nnzb,
                                                    const hipsparseMatDescr_t descrA,
                                                    double*                   bsrValA,
                                                    const int*                bsrRowPtrA,
                                                    const int*                bsrColIndA,
                                                    int                       blockDim,
                                                    bsrilu02Info_t            info,
                                                    int*                      pBufferSizeInBytes)
    {
        return hipsparseDbsrilu02_bufferSize(handle,
                                             dirA,
                                             mb,
                                             nnzb,
                                             descrA,
                                             bsrValA,
                                             bsrRowPtrA,
                                             bsrColIndA,
                                             blockDim,
                                             info,
                                             pBufferSizeInBytes);
    }

    template <>
    hipsparseStatus_t hipsparseXbsrilu02_bufferSize(hipsparseHandle_t         handle,
                                                    hipsparseDirection_t      dirA,
                                                    int                       mb,
                                                    int                       nnzb,
                                                    const hipsparseMatDescr_t descrA,
                                                    hipComplex*               bsrValA,
                                                    const int*                bsrRowPtrA,
                                                    const int*                bsrColIndA,
                                                    int                       blockDim,
                                                    bsrilu02Info_t            info,
                                                    int*                      pBufferSizeInBytes)
    {
        return hipsparseCbsrilu02_bufferSize(handle,
                                             dirA,
                                             mb,
                                             nnzb,
                                             descrA,
                                             bsrValA,
                                             bsrRowPtrA,
                                             bsrColIndA,
                                             blockDim,
                                             info,
                                             pBufferSizeInBytes);
    }

    template <>
    hipsparseStatus_t hipsparseXbsrilu02_bufferSize(hipsparseHandle_t         handle,
                                                    hipsparseDirection_t      dirA,
                                                    int                       mb,
                                                    int                       nnzb,
                                                    const hipsparseMatDescr_t descrA,
                                                    hipDoubleComplex*         bsrValA,
                                                    const int*                bsrRowPtrA,
                                                    const int*                bsrColIndA,
                                                    int                       blockDim,
                                                    bsrilu02Info_t            info,
                                                    int*                      pBufferSizeInBytes)
    {
        return hipsparseZbsrilu02_bufferSize(handle,
                                             dirA,
                                             mb,
                                             nnzb,
                                             descrA,
                                             bsrValA,
                                             bsrRowPtrA,
                                             bsrColIndA,
                                             blockDim,
                                             info,
                                             pBufferSizeInBytes);
    }
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 13000)
    template <>
    hipsparseStatus_t hipsparseXbsrilu02_analysis(hipsparseHandle_t         handle,
                                                  hipsparseDirection_t      dirA,
                                                  int                       mb,
                                                  int                       nnzb,
                                                  const hipsparseMatDescr_t descrA,
                                                  float*                    bsrValA,
                                                  const int*                bsrRowPtrA,
                                                  const int*                bsrColIndA,
                                                  int                       blockDim,
                                                  bsrilu02Info_t            info,
                                                  hipsparseSolvePolicy_t    policy,
                                                  void*                     pBuffer)
    {
        return hipsparseSbsrilu02_analysis(handle,
                                           dirA,
                                           mb,
                                           nnzb,
                                           descrA,
                                           bsrValA,
                                           bsrRowPtrA,
                                           bsrColIndA,
                                           blockDim,
                                           info,
                                           policy,
                                           pBuffer);
    }

    template <>
    hipsparseStatus_t hipsparseXbsrilu02_analysis(hipsparseHandle_t         handle,
                                                  hipsparseDirection_t      dirA,
                                                  int                       mb,
                                                  int                       nnzb,
                                                  const hipsparseMatDescr_t descrA,
                                                  double*                   bsrValA,
                                                  const int*                bsrRowPtrA,
                                                  const int*                bsrColIndA,
                                                  int                       blockDim,
                                                  bsrilu02Info_t            info,
                                                  hipsparseSolvePolicy_t    policy,
                                                  void*                     pBuffer)
    {
        return hipsparseDbsrilu02_analysis(handle,
                                           dirA,
                                           mb,
                                           nnzb,
                                           descrA,
                                           bsrValA,
                                           bsrRowPtrA,
                                           bsrColIndA,
                                           blockDim,
                                           info,
                                           policy,
                                           pBuffer);
    }

    template <>
    hipsparseStatus_t hipsparseXbsrilu02_analysis(hipsparseHandle_t         handle,
                                                  hipsparseDirection_t      dirA,
                                                  int                       mb,
                                                  int                       nnzb,
                                                  const hipsparseMatDescr_t descrA,
                                                  hipComplex*               bsrValA,
                                                  const int*                bsrRowPtrA,
                                                  const int*                bsrColIndA,
                                                  int                       blockDim,
                                                  bsrilu02Info_t            info,
                                                  hipsparseSolvePolicy_t    policy,
                                                  void*                     pBuffer)
    {
        return hipsparseCbsrilu02_analysis(handle,
                                           dirA,
                                           mb,
                                           nnzb,
                                           descrA,
                                           bsrValA,
                                           bsrRowPtrA,
                                           bsrColIndA,
                                           blockDim,
                                           info,
                                           policy,
                                           pBuffer);
    }

    template <>
    hipsparseStatus_t hipsparseXbsrilu02_analysis(hipsparseHandle_t         handle,
                                                  hipsparseDirection_t      dirA,
                                                  int                       mb,
                                                  int                       nnzb,
                                                  const hipsparseMatDescr_t descrA,
                                                  hipDoubleComplex*         bsrValA,
                                                  const int*                bsrRowPtrA,
                                                  const int*                bsrColIndA,
                                                  int                       blockDim,
                                                  bsrilu02Info_t            info,
                                                  hipsparseSolvePolicy_t    policy,
                                                  void*                     pBuffer)
    {
        return hipsparseZbsrilu02_analysis(handle,
                                           dirA,
                                           mb,
                                           nnzb,
                                           descrA,
                                           bsrValA,
                                           bsrRowPtrA,
                                           bsrColIndA,
                                           blockDim,
                                           info,
                                           policy,
                                           pBuffer);
    }
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 13000)
    template <>
    hipsparseStatus_t hipsparseXbsrilu02(hipsparseHandle_t         handle,
                                         hipsparseDirection_t      dirA,
                                         int                       mb,
                                         int                       nnzb,
                                         const hipsparseMatDescr_t descrA,
                                         float*                    bsrValA,
                                         const int*                bsrRowPtrA,
                                         const int*                bsrColIndA,
                                         int                       blockDim,
                                         bsrilu02Info_t            info,
                                         hipsparseSolvePolicy_t    policy,
                                         void*                     pBuffer)
    {
        return hipsparseSbsrilu02(handle,
                                  dirA,
                                  mb,
                                  nnzb,
                                  descrA,
                                  bsrValA,
                                  bsrRowPtrA,
                                  bsrColIndA,
                                  blockDim,
                                  info,
                                  policy,
                                  pBuffer);
    }

    template <>
    hipsparseStatus_t hipsparseXbsrilu02(hipsparseHandle_t         handle,
                                         hipsparseDirection_t      dirA,
                                         int                       mb,
                                         int                       nnzb,
                                         const hipsparseMatDescr_t descrA,
                                         double*                   bsrValA,
                                         const int*                bsrRowPtrA,
                                         const int*                bsrColIndA,
                                         int                       blockDim,
                                         bsrilu02Info_t            info,
                                         hipsparseSolvePolicy_t    policy,
                                         void*                     pBuffer)
    {
        return hipsparseDbsrilu02(handle,
                                  dirA,
                                  mb,
                                  nnzb,
                                  descrA,
                                  bsrValA,
                                  bsrRowPtrA,
                                  bsrColIndA,
                                  blockDim,
                                  info,
                                  policy,
                                  pBuffer);
    }

    template <>
    hipsparseStatus_t hipsparseXbsrilu02(hipsparseHandle_t         handle,
                                         hipsparseDirection_t      dirA,
                                         int                       mb,
                                         int                       nnzb,
                                         const hipsparseMatDescr_t descrA,
                                         hipComplex*               bsrValA,
                                         const int*                bsrRowPtrA,
                                         const int*                bsrColIndA,
                                         int                       blockDim,
                                         bsrilu02Info_t            info,
                                         hipsparseSolvePolicy_t    policy,
                                         void*                     pBuffer)
    {
        return hipsparseCbsrilu02(handle,
                                  dirA,
                                  mb,
                                  nnzb,
                                  descrA,
                                  bsrValA,
                                  bsrRowPtrA,
                                  bsrColIndA,
                                  blockDim,
                                  info,
                                  policy,
                                  pBuffer);
    }

    template <>
    hipsparseStatus_t hipsparseXbsrilu02(hipsparseHandle_t         handle,
                                         hipsparseDirection_t      dirA,
                                         int                       mb,
                                         int                       nnzb,
                                         const hipsparseMatDescr_t descrA,
                                         hipDoubleComplex*         bsrValA,
                                         const int*                bsrRowPtrA,
                                         const int*                bsrColIndA,
                                         int                       blockDim,
                                         bsrilu02Info_t            info,
                                         hipsparseSolvePolicy_t    policy,
                                         void*                     pBuffer)
    {
        return hipsparseZbsrilu02(handle,
                                  dirA,
                                  mb,
                                  nnzb,
                                  descrA,
                                  bsrValA,
                                  bsrRowPtrA,
                                  bsrColIndA,
                                  blockDim,
                                  info,
                                  policy,
                                  pBuffer);
    }
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 13000)
    template <>
    hipsparseStatus_t hipsparseXcsrilu02_numericBoost(hipsparseHandle_t handle,
                                                      csrilu02Info_t    info,
                                                      int               enable_boost,
                                                      double*           tol,
                                                      float*            boost_val)
    {
        return hipsparseScsrilu02_numericBoost(handle, info, enable_boost, tol, boost_val);
    }

    template <>
    hipsparseStatus_t hipsparseXcsrilu02_numericBoost(hipsparseHandle_t handle,
                                                      csrilu02Info_t    info,
                                                      int               enable_boost,
                                                      double*           tol,
                                                      double*           boost_val)
    {
        return hipsparseDcsrilu02_numericBoost(handle, info, enable_boost, tol, boost_val);
    }

    template <>
    hipsparseStatus_t hipsparseXcsrilu02_numericBoost(hipsparseHandle_t handle,
                                                      csrilu02Info_t    info,
                                                      int               enable_boost,
                                                      double*           tol,
                                                      hipComplex*       boost_val)
    {
        return hipsparseCcsrilu02_numericBoost(handle, info, enable_boost, tol, boost_val);
    }

    template <>
    hipsparseStatus_t hipsparseXcsrilu02_numericBoost(hipsparseHandle_t handle,
                                                      csrilu02Info_t    info,
                                                      int               enable_boost,
                                                      double*           tol,
                                                      hipDoubleComplex* boost_val)
    {
        return hipsparseZcsrilu02_numericBoost(handle, info, enable_boost, tol, boost_val);
    }
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 13000)
    template <>
    hipsparseStatus_t hipsparseXcsrilu02_bufferSize(hipsparseHandle_t         handle,
                                                    int                       m,
                                                    int                       nnz,
                                                    const hipsparseMatDescr_t descrA,
                                                    float*                    csrSortedValA,
                                                    const int*                csrSortedRowPtrA,
                                                    const int*                csrSortedColIndA,
                                                    csrilu02Info_t            info,
                                                    int*                      pBufferSizeInBytes)
    {
        return hipsparseScsrilu02_bufferSize(handle,
                                             m,
                                             nnz,
                                             descrA,
                                             csrSortedValA,
                                             csrSortedRowPtrA,
                                             csrSortedColIndA,
                                             info,
                                             pBufferSizeInBytes);
    }

    template <>
    hipsparseStatus_t hipsparseXcsrilu02_bufferSize(hipsparseHandle_t         handle,
                                                    int                       m,
                                                    int                       nnz,
                                                    const hipsparseMatDescr_t descrA,
                                                    double*                   csrSortedValA,
                                                    const int*                csrSortedRowPtrA,
                                                    const int*                csrSortedColIndA,
                                                    csrilu02Info_t            info,
                                                    int*                      pBufferSizeInBytes)
    {
        return hipsparseDcsrilu02_bufferSize(handle,
                                             m,
                                             nnz,
                                             descrA,
                                             csrSortedValA,
                                             csrSortedRowPtrA,
                                             csrSortedColIndA,
                                             info,
                                             pBufferSizeInBytes);
    }

    template <>
    hipsparseStatus_t hipsparseXcsrilu02_bufferSize(hipsparseHandle_t         handle,
                                                    int                       m,
                                                    int                       nnz,
                                                    const hipsparseMatDescr_t descrA,
                                                    hipComplex*               csrSortedValA,
                                                    const int*                csrSortedRowPtrA,
                                                    const int*                csrSortedColIndA,
                                                    csrilu02Info_t            info,
                                                    int*                      pBufferSizeInBytes)
    {
        return hipsparseCcsrilu02_bufferSize(handle,
                                             m,
                                             nnz,
                                             descrA,
                                             csrSortedValA,
                                             csrSortedRowPtrA,
                                             csrSortedColIndA,
                                             info,
                                             pBufferSizeInBytes);
    }

    template <>
    hipsparseStatus_t hipsparseXcsrilu02_bufferSize(hipsparseHandle_t         handle,
                                                    int                       m,
                                                    int                       nnz,
                                                    const hipsparseMatDescr_t descrA,
                                                    hipDoubleComplex*         csrSortedValA,
                                                    const int*                csrSortedRowPtrA,
                                                    const int*                csrSortedColIndA,
                                                    csrilu02Info_t            info,
                                                    int*                      pBufferSizeInBytes)
    {
        return hipsparseZcsrilu02_bufferSize(handle,
                                             m,
                                             nnz,
                                             descrA,
                                             csrSortedValA,
                                             csrSortedRowPtrA,
                                             csrSortedColIndA,
                                             info,
                                             pBufferSizeInBytes);
    }
#endif

    template <>
    hipsparseStatus_t hipsparseXcsrilu02_bufferSizeExt(hipsparseHandle_t         handle,
                                                       int                       m,
                                                       int                       nnz,
                                                       const hipsparseMatDescr_t descrA,
                                                       float*                    csrSortedValA,
                                                       const int*                csrSortedRowPtrA,
                                                       const int*                csrSortedColIndA,
                                                       csrilu02Info_t            info,
                                                       size_t*                   pBufferSizeInBytes)
    {
        return hipsparseScsrilu02_bufferSizeExt(handle,
                                                m,
                                                nnz,
                                                descrA,
                                                csrSortedValA,
                                                csrSortedRowPtrA,
                                                csrSortedColIndA,
                                                info,
                                                pBufferSizeInBytes);
    }

    template <>
    hipsparseStatus_t hipsparseXcsrilu02_bufferSizeExt(hipsparseHandle_t         handle,
                                                       int                       m,
                                                       int                       nnz,
                                                       const hipsparseMatDescr_t descrA,
                                                       double*                   csrSortedValA,
                                                       const int*                csrSortedRowPtrA,
                                                       const int*                csrSortedColIndA,
                                                       csrilu02Info_t            info,
                                                       size_t*                   pBufferSizeInBytes)
    {
        return hipsparseDcsrilu02_bufferSizeExt(handle,
                                                m,
                                                nnz,
                                                descrA,
                                                csrSortedValA,
                                                csrSortedRowPtrA,
                                                csrSortedColIndA,
                                                info,
                                                pBufferSizeInBytes);
    }

    template <>
    hipsparseStatus_t hipsparseXcsrilu02_bufferSizeExt(hipsparseHandle_t         handle,
                                                       int                       m,
                                                       int                       nnz,
                                                       const hipsparseMatDescr_t descrA,
                                                       hipComplex*               csrSortedValA,
                                                       const int*                csrSortedRowPtrA,
                                                       const int*                csrSortedColIndA,
                                                       csrilu02Info_t            info,
                                                       size_t*                   pBufferSizeInBytes)
    {
        return hipsparseCcsrilu02_bufferSizeExt(handle,
                                                m,
                                                nnz,
                                                descrA,
                                                csrSortedValA,
                                                csrSortedRowPtrA,
                                                csrSortedColIndA,
                                                info,
                                                pBufferSizeInBytes);
    }

    template <>
    hipsparseStatus_t hipsparseXcsrilu02_bufferSizeExt(hipsparseHandle_t         handle,
                                                       int                       m,
                                                       int                       nnz,
                                                       const hipsparseMatDescr_t descrA,
                                                       hipDoubleComplex*         csrSortedValA,
                                                       const int*                csrSortedRowPtrA,
                                                       const int*                csrSortedColIndA,
                                                       csrilu02Info_t            info,
                                                       size_t*                   pBufferSizeInBytes)
    {
        return hipsparseZcsrilu02_bufferSizeExt(handle,
                                                m,
                                                nnz,
                                                descrA,
                                                csrSortedValA,
                                                csrSortedRowPtrA,
                                                csrSortedColIndA,
                                                info,
                                                pBufferSizeInBytes);
    }

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 13000)
    template <>
    hipsparseStatus_t hipsparseXcsrilu02_analysis(hipsparseHandle_t         handle,
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
        return hipsparseScsrilu02_analysis(handle,
                                           m,
                                           nnz,
                                           descrA,
                                           csrSortedValA,
                                           csrSortedRowPtrA,
                                           csrSortedColIndA,
                                           info,
                                           policy,
                                           pBuffer);
    }

    template <>
    hipsparseStatus_t hipsparseXcsrilu02_analysis(hipsparseHandle_t         handle,
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
        return hipsparseDcsrilu02_analysis(handle,
                                           m,
                                           nnz,
                                           descrA,
                                           csrSortedValA,
                                           csrSortedRowPtrA,
                                           csrSortedColIndA,
                                           info,
                                           policy,
                                           pBuffer);
    }

    template <>
    hipsparseStatus_t hipsparseXcsrilu02_analysis(hipsparseHandle_t         handle,
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
        return hipsparseCcsrilu02_analysis(handle,
                                           m,
                                           nnz,
                                           descrA,
                                           csrSortedValA,
                                           csrSortedRowPtrA,
                                           csrSortedColIndA,
                                           info,
                                           policy,
                                           pBuffer);
    }

    template <>
    hipsparseStatus_t hipsparseXcsrilu02_analysis(hipsparseHandle_t         handle,
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
        return hipsparseZcsrilu02_analysis(handle,
                                           m,
                                           nnz,
                                           descrA,
                                           csrSortedValA,
                                           csrSortedRowPtrA,
                                           csrSortedColIndA,
                                           info,
                                           policy,
                                           pBuffer);
    }
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 13000)
    template <>
    hipsparseStatus_t hipsparseXcsrilu02(hipsparseHandle_t         handle,
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
        return hipsparseScsrilu02(handle,
                                  m,
                                  nnz,
                                  descrA,
                                  csrSortedValA_valM,
                                  csrSortedRowPtrA,
                                  csrSortedColIndA,
                                  info,
                                  policy,
                                  pBuffer);
    }

    template <>
    hipsparseStatus_t hipsparseXcsrilu02(hipsparseHandle_t         handle,
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
        return hipsparseDcsrilu02(handle,
                                  m,
                                  nnz,
                                  descrA,
                                  csrSortedValA_valM,
                                  csrSortedRowPtrA,
                                  csrSortedColIndA,
                                  info,
                                  policy,
                                  pBuffer);
    }

    template <>
    hipsparseStatus_t hipsparseXcsrilu02(hipsparseHandle_t         handle,
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
        return hipsparseCcsrilu02(handle,
                                  m,
                                  nnz,
                                  descrA,
                                  csrSortedValA_valM,
                                  csrSortedRowPtrA,
                                  csrSortedColIndA,
                                  info,
                                  policy,
                                  pBuffer);
    }

    template <>
    hipsparseStatus_t hipsparseXcsrilu02(hipsparseHandle_t         handle,
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
        return hipsparseZcsrilu02(handle,
                                  m,
                                  nnz,
                                  descrA,
                                  csrSortedValA_valM,
                                  csrSortedRowPtrA,
                                  csrSortedColIndA,
                                  info,
                                  policy,
                                  pBuffer);
    }
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 13000)
    template <>
    hipsparseStatus_t hipsparseXbsric02_bufferSize(hipsparseHandle_t         handle,
                                                   hipsparseDirection_t      dirA,
                                                   int                       mb,
                                                   int                       nnzb,
                                                   const hipsparseMatDescr_t descrA,
                                                   float*                    bsrValA,
                                                   const int*                bsrRowPtrA,
                                                   const int*                bsrColIndA,
                                                   int                       blockDim,
                                                   bsric02Info_t             info,
                                                   int*                      pBufferSizeInBytes)
    {
        return hipsparseSbsric02_bufferSize(handle,
                                            dirA,
                                            mb,
                                            nnzb,
                                            descrA,
                                            bsrValA,
                                            bsrRowPtrA,
                                            bsrColIndA,
                                            blockDim,
                                            info,
                                            pBufferSizeInBytes);
    }

    template <>
    hipsparseStatus_t hipsparseXbsric02_bufferSize(hipsparseHandle_t         handle,
                                                   hipsparseDirection_t      dirA,
                                                   int                       mb,
                                                   int                       nnzb,
                                                   const hipsparseMatDescr_t descrA,
                                                   double*                   bsrValA,
                                                   const int*                bsrRowPtrA,
                                                   const int*                bsrColIndA,
                                                   int                       blockDim,
                                                   bsric02Info_t             info,
                                                   int*                      pBufferSizeInBytes)
    {
        return hipsparseDbsric02_bufferSize(handle,
                                            dirA,
                                            mb,
                                            nnzb,
                                            descrA,
                                            bsrValA,
                                            bsrRowPtrA,
                                            bsrColIndA,
                                            blockDim,
                                            info,
                                            pBufferSizeInBytes);
    }

    template <>
    hipsparseStatus_t hipsparseXbsric02_bufferSize(hipsparseHandle_t         handle,
                                                   hipsparseDirection_t      dirA,
                                                   int                       mb,
                                                   int                       nnzb,
                                                   const hipsparseMatDescr_t descrA,
                                                   hipComplex*               bsrValA,
                                                   const int*                bsrRowPtrA,
                                                   const int*                bsrColIndA,
                                                   int                       blockDim,
                                                   bsric02Info_t             info,
                                                   int*                      pBufferSizeInBytes)
    {
        return hipsparseCbsric02_bufferSize(handle,
                                            dirA,
                                            mb,
                                            nnzb,
                                            descrA,
                                            bsrValA,
                                            bsrRowPtrA,
                                            bsrColIndA,
                                            blockDim,
                                            info,
                                            pBufferSizeInBytes);
    }

    template <>
    hipsparseStatus_t hipsparseXbsric02_bufferSize(hipsparseHandle_t         handle,
                                                   hipsparseDirection_t      dirA,
                                                   int                       mb,
                                                   int                       nnzb,
                                                   const hipsparseMatDescr_t descrA,
                                                   hipDoubleComplex*         bsrValA,
                                                   const int*                bsrRowPtrA,
                                                   const int*                bsrColIndA,
                                                   int                       blockDim,
                                                   bsric02Info_t             info,
                                                   int*                      pBufferSizeInBytes)
    {
        return hipsparseZbsric02_bufferSize(handle,
                                            dirA,
                                            mb,
                                            nnzb,
                                            descrA,
                                            bsrValA,
                                            bsrRowPtrA,
                                            bsrColIndA,
                                            blockDim,
                                            info,
                                            pBufferSizeInBytes);
    }
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 13000)
    template <>
    hipsparseStatus_t hipsparseXbsric02_analysis(hipsparseHandle_t         handle,
                                                 hipsparseDirection_t      dirA,
                                                 int                       mb,
                                                 int                       nnzb,
                                                 const hipsparseMatDescr_t descrA,
                                                 const float*              bsrValA,
                                                 const int*                bsrRowPtrA,
                                                 const int*                bsrColIndA,
                                                 int                       blockDim,
                                                 bsric02Info_t             info,
                                                 hipsparseSolvePolicy_t    policy,
                                                 void*                     pBuffer)
    {
        return hipsparseSbsric02_analysis(handle,
                                          dirA,
                                          mb,
                                          nnzb,
                                          descrA,
                                          bsrValA,
                                          bsrRowPtrA,
                                          bsrColIndA,
                                          blockDim,
                                          info,
                                          policy,
                                          pBuffer);
    }

    template <>
    hipsparseStatus_t hipsparseXbsric02_analysis(hipsparseHandle_t         handle,
                                                 hipsparseDirection_t      dirA,
                                                 int                       mb,
                                                 int                       nnzb,
                                                 const hipsparseMatDescr_t descrA,
                                                 const double*             bsrValA,
                                                 const int*                bsrRowPtrA,
                                                 const int*                bsrColIndA,
                                                 int                       blockDim,
                                                 bsric02Info_t             info,
                                                 hipsparseSolvePolicy_t    policy,
                                                 void*                     pBuffer)
    {
        return hipsparseDbsric02_analysis(handle,
                                          dirA,
                                          mb,
                                          nnzb,
                                          descrA,
                                          bsrValA,
                                          bsrRowPtrA,
                                          bsrColIndA,
                                          blockDim,
                                          info,
                                          policy,
                                          pBuffer);
    }

    template <>
    hipsparseStatus_t hipsparseXbsric02_analysis(hipsparseHandle_t         handle,
                                                 hipsparseDirection_t      dirA,
                                                 int                       mb,
                                                 int                       nnzb,
                                                 const hipsparseMatDescr_t descrA,
                                                 const hipComplex*         bsrValA,
                                                 const int*                bsrRowPtrA,
                                                 const int*                bsrColIndA,
                                                 int                       blockDim,
                                                 bsric02Info_t             info,
                                                 hipsparseSolvePolicy_t    policy,
                                                 void*                     pBuffer)
    {
        return hipsparseCbsric02_analysis(handle,
                                          dirA,
                                          mb,
                                          nnzb,
                                          descrA,
                                          bsrValA,
                                          bsrRowPtrA,
                                          bsrColIndA,
                                          blockDim,
                                          info,
                                          policy,
                                          pBuffer);
    }

    template <>
    hipsparseStatus_t hipsparseXbsric02_analysis(hipsparseHandle_t         handle,
                                                 hipsparseDirection_t      dirA,
                                                 int                       mb,
                                                 int                       nnzb,
                                                 const hipsparseMatDescr_t descrA,
                                                 const hipDoubleComplex*   bsrValA,
                                                 const int*                bsrRowPtrA,
                                                 const int*                bsrColIndA,
                                                 int                       blockDim,
                                                 bsric02Info_t             info,
                                                 hipsparseSolvePolicy_t    policy,
                                                 void*                     pBuffer)
    {
        return hipsparseZbsric02_analysis(handle,
                                          dirA,
                                          mb,
                                          nnzb,
                                          descrA,
                                          bsrValA,
                                          bsrRowPtrA,
                                          bsrColIndA,
                                          blockDim,
                                          info,
                                          policy,
                                          pBuffer);
    }
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 13000)
    template <>
    hipsparseStatus_t hipsparseXbsric02(hipsparseHandle_t         handle,
                                        hipsparseDirection_t      dirA,
                                        int                       mb,
                                        int                       nnzb,
                                        const hipsparseMatDescr_t descrA,
                                        float*                    bsrValA,
                                        const int*                bsrRowPtrA,
                                        const int*                bsrColIndA,
                                        int                       blockDim,
                                        bsric02Info_t             info,
                                        hipsparseSolvePolicy_t    policy,
                                        void*                     pBuffer)
    {
        return hipsparseSbsric02(handle,
                                 dirA,
                                 mb,
                                 nnzb,
                                 descrA,
                                 bsrValA,
                                 bsrRowPtrA,
                                 bsrColIndA,
                                 blockDim,
                                 info,
                                 policy,
                                 pBuffer);
    }

    template <>
    hipsparseStatus_t hipsparseXbsric02(hipsparseHandle_t         handle,
                                        hipsparseDirection_t      dirA,
                                        int                       mb,
                                        int                       nnzb,
                                        const hipsparseMatDescr_t descrA,
                                        double*                   bsrValA,
                                        const int*                bsrRowPtrA,
                                        const int*                bsrColIndA,
                                        int                       blockDim,
                                        bsric02Info_t             info,
                                        hipsparseSolvePolicy_t    policy,
                                        void*                     pBuffer)
    {
        return hipsparseDbsric02(handle,
                                 dirA,
                                 mb,
                                 nnzb,
                                 descrA,
                                 bsrValA,
                                 bsrRowPtrA,
                                 bsrColIndA,
                                 blockDim,
                                 info,
                                 policy,
                                 pBuffer);
    }

    template <>
    hipsparseStatus_t hipsparseXbsric02(hipsparseHandle_t         handle,
                                        hipsparseDirection_t      dirA,
                                        int                       mb,
                                        int                       nnzb,
                                        const hipsparseMatDescr_t descrA,
                                        hipComplex*               bsrValA,
                                        const int*                bsrRowPtrA,
                                        const int*                bsrColIndA,
                                        int                       blockDim,
                                        bsric02Info_t             info,
                                        hipsparseSolvePolicy_t    policy,
                                        void*                     pBuffer)
    {
        return hipsparseCbsric02(handle,
                                 dirA,
                                 mb,
                                 nnzb,
                                 descrA,
                                 bsrValA,
                                 bsrRowPtrA,
                                 bsrColIndA,
                                 blockDim,
                                 info,
                                 policy,
                                 pBuffer);
    }

    template <>
    hipsparseStatus_t hipsparseXbsric02(hipsparseHandle_t         handle,
                                        hipsparseDirection_t      dirA,
                                        int                       mb,
                                        int                       nnzb,
                                        const hipsparseMatDescr_t descrA,
                                        hipDoubleComplex*         bsrValA,
                                        const int*                bsrRowPtrA,
                                        const int*                bsrColIndA,
                                        int                       blockDim,
                                        bsric02Info_t             info,
                                        hipsparseSolvePolicy_t    policy,
                                        void*                     pBuffer)
    {
        return hipsparseZbsric02(handle,
                                 dirA,
                                 mb,
                                 nnzb,
                                 descrA,
                                 bsrValA,
                                 bsrRowPtrA,
                                 bsrColIndA,
                                 blockDim,
                                 info,
                                 policy,
                                 pBuffer);
    }
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 13000)
    template <>
    hipsparseStatus_t hipsparseXcsric02_bufferSize(hipsparseHandle_t         handle,
                                                   int                       m,
                                                   int                       nnz,
                                                   const hipsparseMatDescr_t descrA,
                                                   float*                    csrSortedValA,
                                                   const int*                csrSortedRowPtrA,
                                                   const int*                csrSortedColIndA,
                                                   csric02Info_t             info,
                                                   int*                      pBufferSizeInBytes)
    {
        return hipsparseScsric02_bufferSize(handle,
                                            m,
                                            nnz,
                                            descrA,
                                            csrSortedValA,
                                            csrSortedRowPtrA,
                                            csrSortedColIndA,
                                            info,
                                            pBufferSizeInBytes);
    }

    template <>
    hipsparseStatus_t hipsparseXcsric02_bufferSize(hipsparseHandle_t         handle,
                                                   int                       m,
                                                   int                       nnz,
                                                   const hipsparseMatDescr_t descrA,
                                                   double*                   csrSortedValA,
                                                   const int*                csrSortedRowPtrA,
                                                   const int*                csrSortedColIndA,
                                                   csric02Info_t             info,
                                                   int*                      pBufferSizeInBytes)
    {
        return hipsparseDcsric02_bufferSize(handle,
                                            m,
                                            nnz,
                                            descrA,
                                            csrSortedValA,
                                            csrSortedRowPtrA,
                                            csrSortedColIndA,
                                            info,
                                            pBufferSizeInBytes);
    }

    template <>
    hipsparseStatus_t hipsparseXcsric02_bufferSize(hipsparseHandle_t         handle,
                                                   int                       m,
                                                   int                       nnz,
                                                   const hipsparseMatDescr_t descrA,
                                                   hipComplex*               csrSortedValA,
                                                   const int*                csrSortedRowPtrA,
                                                   const int*                csrSortedColIndA,
                                                   csric02Info_t             info,
                                                   int*                      pBufferSizeInBytes)
    {
        return hipsparseCcsric02_bufferSize(handle,
                                            m,
                                            nnz,
                                            descrA,
                                            csrSortedValA,
                                            csrSortedRowPtrA,
                                            csrSortedColIndA,
                                            info,
                                            pBufferSizeInBytes);
    }

    template <>
    hipsparseStatus_t hipsparseXcsric02_bufferSize(hipsparseHandle_t         handle,
                                                   int                       m,
                                                   int                       nnz,
                                                   const hipsparseMatDescr_t descrA,
                                                   hipDoubleComplex*         csrSortedValA,
                                                   const int*                csrSortedRowPtrA,
                                                   const int*                csrSortedColIndA,
                                                   csric02Info_t             info,
                                                   int*                      pBufferSizeInBytes)
    {
        return hipsparseZcsric02_bufferSize(handle,
                                            m,
                                            nnz,
                                            descrA,
                                            csrSortedValA,
                                            csrSortedRowPtrA,
                                            csrSortedColIndA,
                                            info,
                                            pBufferSizeInBytes);
    }
#endif

    template <>
    hipsparseStatus_t hipsparseXcsric02_bufferSizeExt(hipsparseHandle_t         handle,
                                                      int                       m,
                                                      int                       nnz,
                                                      const hipsparseMatDescr_t descrA,
                                                      float*                    csrSortedValA,
                                                      const int*                csrSortedRowPtrA,
                                                      const int*                csrSortedColIndA,
                                                      csric02Info_t             info,
                                                      size_t*                   pBufferSizeInBytes)
    {
        return hipsparseScsric02_bufferSizeExt(handle,
                                               m,
                                               nnz,
                                               descrA,
                                               csrSortedValA,
                                               csrSortedRowPtrA,
                                               csrSortedColIndA,
                                               info,
                                               pBufferSizeInBytes);
    }

    template <>
    hipsparseStatus_t hipsparseXcsric02_bufferSizeExt(hipsparseHandle_t         handle,
                                                      int                       m,
                                                      int                       nnz,
                                                      const hipsparseMatDescr_t descrA,
                                                      double*                   csrSortedValA,
                                                      const int*                csrSortedRowPtrA,
                                                      const int*                csrSortedColIndA,
                                                      csric02Info_t             info,
                                                      size_t*                   pBufferSizeInBytes)
    {
        return hipsparseDcsric02_bufferSizeExt(handle,
                                               m,
                                               nnz,
                                               descrA,
                                               csrSortedValA,
                                               csrSortedRowPtrA,
                                               csrSortedColIndA,
                                               info,
                                               pBufferSizeInBytes);
    }

    template <>
    hipsparseStatus_t hipsparseXcsric02_bufferSizeExt(hipsparseHandle_t         handle,
                                                      int                       m,
                                                      int                       nnz,
                                                      const hipsparseMatDescr_t descrA,
                                                      hipComplex*               csrSortedValA,
                                                      const int*                csrSortedRowPtrA,
                                                      const int*                csrSortedColIndA,
                                                      csric02Info_t             info,
                                                      size_t*                   pBufferSizeInBytes)
    {
        return hipsparseCcsric02_bufferSizeExt(handle,
                                               m,
                                               nnz,
                                               descrA,
                                               csrSortedValA,
                                               csrSortedRowPtrA,
                                               csrSortedColIndA,
                                               info,
                                               pBufferSizeInBytes);
    }

    template <>
    hipsparseStatus_t hipsparseXcsric02_bufferSizeExt(hipsparseHandle_t         handle,
                                                      int                       m,
                                                      int                       nnz,
                                                      const hipsparseMatDescr_t descrA,
                                                      hipDoubleComplex*         csrSortedValA,
                                                      const int*                csrSortedRowPtrA,
                                                      const int*                csrSortedColIndA,
                                                      csric02Info_t             info,
                                                      size_t*                   pBufferSizeInBytes)
    {
        return hipsparseZcsric02_bufferSizeExt(handle,
                                               m,
                                               nnz,
                                               descrA,
                                               csrSortedValA,
                                               csrSortedRowPtrA,
                                               csrSortedColIndA,
                                               info,
                                               pBufferSizeInBytes);
    }

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 13000)
    template <>
    hipsparseStatus_t hipsparseXcsric02_analysis(hipsparseHandle_t         handle,
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
        return hipsparseScsric02_analysis(handle,
                                          m,
                                          nnz,
                                          descrA,
                                          csrSortedValA,
                                          csrSortedRowPtrA,
                                          csrSortedColIndA,
                                          info,
                                          policy,
                                          pBuffer);
    }

    template <>
    hipsparseStatus_t hipsparseXcsric02_analysis(hipsparseHandle_t         handle,
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
        return hipsparseDcsric02_analysis(handle,
                                          m,
                                          nnz,
                                          descrA,
                                          csrSortedValA,
                                          csrSortedRowPtrA,
                                          csrSortedColIndA,
                                          info,
                                          policy,
                                          pBuffer);
    }

    template <>
    hipsparseStatus_t hipsparseXcsric02_analysis(hipsparseHandle_t         handle,
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
        return hipsparseCcsric02_analysis(handle,
                                          m,
                                          nnz,
                                          descrA,
                                          csrSortedValA,
                                          csrSortedRowPtrA,
                                          csrSortedColIndA,
                                          info,
                                          policy,
                                          pBuffer);
    }

    template <>
    hipsparseStatus_t hipsparseXcsric02_analysis(hipsparseHandle_t         handle,
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
        return hipsparseZcsric02_analysis(handle,
                                          m,
                                          nnz,
                                          descrA,
                                          csrSortedValA,
                                          csrSortedRowPtrA,
                                          csrSortedColIndA,
                                          info,
                                          policy,
                                          pBuffer);
    }
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 13000)
    template <>
    hipsparseStatus_t hipsparseXcsric02(hipsparseHandle_t         handle,
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
        return hipsparseScsric02(handle,
                                 m,
                                 nnz,
                                 descrA,
                                 csrSortedValA_valM,
                                 csrSortedRowPtrA,
                                 csrSortedColIndA,
                                 info,
                                 policy,
                                 pBuffer);
    }

    template <>
    hipsparseStatus_t hipsparseXcsric02(hipsparseHandle_t         handle,
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
        return hipsparseDcsric02(handle,
                                 m,
                                 nnz,
                                 descrA,
                                 csrSortedValA_valM,
                                 csrSortedRowPtrA,
                                 csrSortedColIndA,
                                 info,
                                 policy,
                                 pBuffer);
    }

    template <>
    hipsparseStatus_t hipsparseXcsric02(hipsparseHandle_t         handle,
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
        return hipsparseCcsric02(handle,
                                 m,
                                 nnz,
                                 descrA,
                                 csrSortedValA_valM,
                                 csrSortedRowPtrA,
                                 csrSortedColIndA,
                                 info,
                                 policy,
                                 pBuffer);
    }

    template <>
    hipsparseStatus_t hipsparseXcsric02(hipsparseHandle_t         handle,
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
        return hipsparseZcsric02(handle,
                                 m,
                                 nnz,
                                 descrA,
                                 csrSortedValA_valM,
                                 csrSortedRowPtrA,
                                 csrSortedColIndA,
                                 info,
                                 policy,
                                 pBuffer);
    }
#endif

    template <>
    hipsparseStatus_t hipsparseXnnz(hipsparseHandle_t         handle,
                                    hipsparseDirection_t      dirA,
                                    int                       m,
                                    int                       n,
                                    const hipsparseMatDescr_t descrA,
                                    const float*              A,
                                    int                       lda,
                                    int*                      nnzPerRowColumn,
                                    int*                      nnzTotalDevHostPtr)
    {
        return hipsparseSnnz(
            handle, dirA, m, n, descrA, A, lda, nnzPerRowColumn, nnzTotalDevHostPtr);
    }

    template <>
    hipsparseStatus_t hipsparseXnnz(hipsparseHandle_t         handle,
                                    hipsparseDirection_t      dirA,
                                    int                       m,
                                    int                       n,
                                    const hipsparseMatDescr_t descrA,
                                    const double*             A,
                                    int                       lda,
                                    int*                      nnzPerRowColumn,
                                    int*                      nnzTotalDevHostPtr)
    {
        return hipsparseDnnz(
            handle, dirA, m, n, descrA, A, lda, nnzPerRowColumn, nnzTotalDevHostPtr);
    }

    template <>
    hipsparseStatus_t hipsparseXnnz(hipsparseHandle_t         handle,
                                    hipsparseDirection_t      dirA,
                                    int                       m,
                                    int                       n,
                                    const hipsparseMatDescr_t descrA,
                                    const hipComplex*         A,
                                    int                       lda,
                                    int*                      nnzPerRowColumn,
                                    int*                      nnzTotalDevHostPtr)
    {
        return hipsparseCnnz(
            handle, dirA, m, n, descrA, A, lda, nnzPerRowColumn, nnzTotalDevHostPtr);
    }

    template <>
    hipsparseStatus_t hipsparseXnnz(hipsparseHandle_t         handle,
                                    hipsparseDirection_t      dirA,
                                    int                       m,
                                    int                       n,
                                    const hipsparseMatDescr_t descrA,
                                    const hipDoubleComplex*   A,
                                    int                       lda,
                                    int*                      nnzPerRowColumn,
                                    int*                      nnzTotalDevHostPtr)
    {
        return hipsparseZnnz(
            handle, dirA, m, n, descrA, A, lda, nnzPerRowColumn, nnzTotalDevHostPtr);
    }

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 13000)
    template <>
    hipsparseStatus_t hipsparseXnnz_compress(hipsparseHandle_t         handle,
                                             int                       m,
                                             const hipsparseMatDescr_t descrA,
                                             const float*              csrValA,
                                             const int*                csrRowPtrA,
                                             int*                      nnzPerRow,
                                             int*                      nnzC,
                                             float                     tol)
    {
        return hipsparseSnnz_compress(handle, m, descrA, csrValA, csrRowPtrA, nnzPerRow, nnzC, tol);
    }

    template <>
    hipsparseStatus_t hipsparseXnnz_compress(hipsparseHandle_t         handle,
                                             int                       m,
                                             const hipsparseMatDescr_t descrA,
                                             const double*             csrValA,
                                             const int*                csrRowPtrA,
                                             int*                      nnzPerRow,
                                             int*                      nnzC,
                                             double                    tol)
    {
        return hipsparseDnnz_compress(handle, m, descrA, csrValA, csrRowPtrA, nnzPerRow, nnzC, tol);
    }

    template <>
    hipsparseStatus_t hipsparseXnnz_compress(hipsparseHandle_t         handle,
                                             int                       m,
                                             const hipsparseMatDescr_t descrA,
                                             const hipComplex*         csrValA,
                                             const int*                csrRowPtrA,
                                             int*                      nnzPerRow,
                                             int*                      nnzC,
                                             hipComplex                tol)
    {
        return hipsparseCnnz_compress(handle, m, descrA, csrValA, csrRowPtrA, nnzPerRow, nnzC, tol);
    }

    template <>
    hipsparseStatus_t hipsparseXnnz_compress(hipsparseHandle_t         handle,
                                             int                       m,
                                             const hipsparseMatDescr_t descrA,
                                             const hipDoubleComplex*   csrValA,
                                             const int*                csrRowPtrA,
                                             int*                      nnzPerRow,
                                             int*                      nnzC,
                                             hipDoubleComplex          tol)
    {
        return hipsparseZnnz_compress(handle, m, descrA, csrValA, csrRowPtrA, nnzPerRow, nnzC, tol);
    }
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 12000)
    template <>
    hipsparseStatus_t hipsparseXdense2csr(hipsparseHandle_t         handle,
                                          int                       m,
                                          int                       n,
                                          const hipsparseMatDescr_t descr,
                                          const float*              A,
                                          int                       ld,
                                          const int*                nnzPerRow,
                                          float*                    csrVal,
                                          int*                      csrRowPtr,
                                          int*                      csrColInd)
    {
        return hipsparseSdense2csr(
            handle, m, n, descr, A, ld, nnzPerRow, csrVal, csrRowPtr, csrColInd);
    }

    template <>
    hipsparseStatus_t hipsparseXdense2csr(hipsparseHandle_t         handle,
                                          int                       m,
                                          int                       n,
                                          const hipsparseMatDescr_t descr,
                                          const double*             A,
                                          int                       ld,
                                          const int*                nnzPerRow,
                                          double*                   csrVal,
                                          int*                      csrRowPtr,
                                          int*                      csrColInd)
    {
        return hipsparseDdense2csr(
            handle, m, n, descr, A, ld, nnzPerRow, csrVal, csrRowPtr, csrColInd);
    }

    template <>
    hipsparseStatus_t hipsparseXdense2csr(hipsparseHandle_t         handle,
                                          int                       m,
                                          int                       n,
                                          const hipsparseMatDescr_t descr,
                                          const hipComplex*         A,
                                          int                       ld,
                                          const int*                nnzPerRow,
                                          hipComplex*               csrVal,
                                          int*                      csrRowPtr,
                                          int*                      csrColInd)
    {
        return hipsparseCdense2csr(
            handle, m, n, descr, A, ld, nnzPerRow, csrVal, csrRowPtr, csrColInd);
    }

    template <>
    hipsparseStatus_t hipsparseXdense2csr(hipsparseHandle_t         handle,
                                          int                       m,
                                          int                       n,
                                          const hipsparseMatDescr_t descr,
                                          const hipDoubleComplex*   A,
                                          int                       ld,
                                          const int*                nnzPerRow,
                                          hipDoubleComplex*         csrVal,
                                          int*                      csrRowPtr,
                                          int*                      csrColInd)
    {
        return hipsparseZdense2csr(
            handle, m, n, descr, A, ld, nnzPerRow, csrVal, csrRowPtr, csrColInd);
    }
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 13000)
    template <>
    hipsparseStatus_t hipsparseXpruneDense2csr_bufferSize(hipsparseHandle_t         handle,
                                                          int                       m,
                                                          int                       n,
                                                          const float*              A,
                                                          int                       lda,
                                                          const float*              threshold,
                                                          const hipsparseMatDescr_t descr,
                                                          const float*              csrVal,
                                                          const int*                csrRowPtr,
                                                          const int*                csrColInd,
                                                          size_t* pBufferSizeInBytes)
    {
        return hipsparseSpruneDense2csr_bufferSize(handle,
                                                   m,
                                                   n,
                                                   A,
                                                   lda,
                                                   threshold,
                                                   descr,
                                                   csrVal,
                                                   csrRowPtr,
                                                   csrColInd,
                                                   pBufferSizeInBytes);
    }

    template <>
    hipsparseStatus_t hipsparseXpruneDense2csr_bufferSize(hipsparseHandle_t         handle,
                                                          int                       m,
                                                          int                       n,
                                                          const double*             A,
                                                          int                       lda,
                                                          const double*             threshold,
                                                          const hipsparseMatDescr_t descr,
                                                          const double*             csrVal,
                                                          const int*                csrRowPtr,
                                                          const int*                csrColInd,
                                                          size_t* pBufferSizeInBytes)
    {
        return hipsparseDpruneDense2csr_bufferSize(handle,
                                                   m,
                                                   n,
                                                   A,
                                                   lda,
                                                   threshold,
                                                   descr,
                                                   csrVal,
                                                   csrRowPtr,
                                                   csrColInd,
                                                   pBufferSizeInBytes);
    }
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 13000)
    template <>
    hipsparseStatus_t hipsparseXpruneDense2csr_bufferSizeExt(hipsparseHandle_t         handle,
                                                             int                       m,
                                                             int                       n,
                                                             const float*              A,
                                                             int                       lda,
                                                             const float*              threshold,
                                                             const hipsparseMatDescr_t descr,
                                                             const float*              csrVal,
                                                             const int*                csrRowPtr,
                                                             const int*                csrColInd,
                                                             size_t* pBufferSizeInBytes)
    {
        return hipsparseSpruneDense2csr_bufferSizeExt(handle,
                                                      m,
                                                      n,
                                                      A,
                                                      lda,
                                                      threshold,
                                                      descr,
                                                      csrVal,
                                                      csrRowPtr,
                                                      csrColInd,
                                                      pBufferSizeInBytes);
    }

    template <>
    hipsparseStatus_t hipsparseXpruneDense2csr_bufferSizeExt(hipsparseHandle_t         handle,
                                                             int                       m,
                                                             int                       n,
                                                             const double*             A,
                                                             int                       lda,
                                                             const double*             threshold,
                                                             const hipsparseMatDescr_t descr,
                                                             const double*             csrVal,
                                                             const int*                csrRowPtr,
                                                             const int*                csrColInd,
                                                             size_t* pBufferSizeInBytes)
    {
        return hipsparseDpruneDense2csr_bufferSizeExt(handle,
                                                      m,
                                                      n,
                                                      A,
                                                      lda,
                                                      threshold,
                                                      descr,
                                                      csrVal,
                                                      csrRowPtr,
                                                      csrColInd,
                                                      pBufferSizeInBytes);
    }
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 13000)
    template <>
    hipsparseStatus_t hipsparseXpruneDense2csrNnz(hipsparseHandle_t         handle,
                                                  int                       m,
                                                  int                       n,
                                                  const float*              A,
                                                  int                       lda,
                                                  const float*              threshold,
                                                  const hipsparseMatDescr_t descr,
                                                  int*                      csrRowPtr,
                                                  int*                      nnzTotalDevHostPtr,
                                                  void*                     buffer)
    {
        return hipsparseSpruneDense2csrNnz(
            handle, m, n, A, lda, threshold, descr, csrRowPtr, nnzTotalDevHostPtr, buffer);
    }

    template <>
    hipsparseStatus_t hipsparseXpruneDense2csrNnz(hipsparseHandle_t         handle,
                                                  int                       m,
                                                  int                       n,
                                                  const double*             A,
                                                  int                       lda,
                                                  const double*             threshold,
                                                  const hipsparseMatDescr_t descr,
                                                  int*                      csrRowPtr,
                                                  int*                      nnzTotalDevHostPtr,
                                                  void*                     buffer)
    {
        return hipsparseDpruneDense2csrNnz(
            handle, m, n, A, lda, threshold, descr, csrRowPtr, nnzTotalDevHostPtr, buffer);
    }
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 13000)
    template <>
    hipsparseStatus_t hipsparseXpruneDense2csr(hipsparseHandle_t         handle,
                                               int                       m,
                                               int                       n,
                                               const float*              A,
                                               int                       lda,
                                               const float*              threshold,
                                               const hipsparseMatDescr_t descr,
                                               float*                    csrVal,
                                               const int*                csrRowPtr,
                                               int*                      csrColInd,
                                               void*                     buffer)
    {
        return hipsparseSpruneDense2csr(
            handle, m, n, A, lda, threshold, descr, csrVal, csrRowPtr, csrColInd, buffer);
    }

    template <>
    hipsparseStatus_t hipsparseXpruneDense2csr(hipsparseHandle_t         handle,
                                               int                       m,
                                               int                       n,
                                               const double*             A,
                                               int                       lda,
                                               const double*             threshold,
                                               const hipsparseMatDescr_t descr,
                                               double*                   csrVal,
                                               const int*                csrRowPtr,
                                               int*                      csrColInd,
                                               void*                     buffer)
    {
        return hipsparseDpruneDense2csr(
            handle, m, n, A, lda, threshold, descr, csrVal, csrRowPtr, csrColInd, buffer);
    }
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 13000)
    template <>
    hipsparseStatus_t
        hipsparseXpruneDense2csrByPercentage_bufferSize(hipsparseHandle_t         handle,
                                                        int                       m,
                                                        int                       n,
                                                        const float*              A,
                                                        int                       lda,
                                                        float                     percentage,
                                                        const hipsparseMatDescr_t descr,
                                                        const float*              csrVal,
                                                        const int*                csrRowPtr,
                                                        const int*                csrColInd,
                                                        pruneInfo_t               info,
                                                        size_t* pBufferSizeInBytes)
    {
        return hipsparseSpruneDense2csrByPercentage_bufferSize(handle,
                                                               m,
                                                               n,
                                                               A,
                                                               lda,
                                                               percentage,
                                                               descr,
                                                               csrVal,
                                                               csrRowPtr,
                                                               csrColInd,
                                                               info,
                                                               pBufferSizeInBytes);
    }

    template <>
    hipsparseStatus_t
        hipsparseXpruneDense2csrByPercentage_bufferSize(hipsparseHandle_t         handle,
                                                        int                       m,
                                                        int                       n,
                                                        const double*             A,
                                                        int                       lda,
                                                        double                    percentage,
                                                        const hipsparseMatDescr_t descr,
                                                        const double*             csrVal,
                                                        const int*                csrRowPtr,
                                                        const int*                csrColInd,
                                                        pruneInfo_t               info,
                                                        size_t* pBufferSizeInBytes)
    {
        return hipsparseDpruneDense2csrByPercentage_bufferSize(handle,
                                                               m,
                                                               n,
                                                               A,
                                                               lda,
                                                               percentage,
                                                               descr,
                                                               csrVal,
                                                               csrRowPtr,
                                                               csrColInd,
                                                               info,
                                                               pBufferSizeInBytes);
    }
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 13000)
    template <>
    hipsparseStatus_t
        hipsparseXpruneDense2csrByPercentage_bufferSizeExt(hipsparseHandle_t         handle,
                                                           int                       m,
                                                           int                       n,
                                                           const float*              A,
                                                           int                       lda,
                                                           float                     percentage,
                                                           const hipsparseMatDescr_t descr,
                                                           const float*              csrVal,
                                                           const int*                csrRowPtr,
                                                           const int*                csrColInd,
                                                           pruneInfo_t               info,
                                                           size_t* pBufferSizeInBytes)
    {
        return hipsparseSpruneDense2csrByPercentage_bufferSizeExt(handle,
                                                                  m,
                                                                  n,
                                                                  A,
                                                                  lda,
                                                                  percentage,
                                                                  descr,
                                                                  csrVal,
                                                                  csrRowPtr,
                                                                  csrColInd,
                                                                  info,
                                                                  pBufferSizeInBytes);
    }

    template <>
    hipsparseStatus_t
        hipsparseXpruneDense2csrByPercentage_bufferSizeExt(hipsparseHandle_t         handle,
                                                           int                       m,
                                                           int                       n,
                                                           const double*             A,
                                                           int                       lda,
                                                           double                    percentage,
                                                           const hipsparseMatDescr_t descr,
                                                           const double*             csrVal,
                                                           const int*                csrRowPtr,
                                                           const int*                csrColInd,
                                                           pruneInfo_t               info,
                                                           size_t* pBufferSizeInBytes)
    {
        return hipsparseDpruneDense2csrByPercentage_bufferSizeExt(handle,
                                                                  m,
                                                                  n,
                                                                  A,
                                                                  lda,
                                                                  percentage,
                                                                  descr,
                                                                  csrVal,
                                                                  csrRowPtr,
                                                                  csrColInd,
                                                                  info,
                                                                  pBufferSizeInBytes);
    }
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 13000)
    template <>
    hipsparseStatus_t hipsparseXpruneDense2csrNnzByPercentage(hipsparseHandle_t         handle,
                                                              int                       m,
                                                              int                       n,
                                                              const float*              A,
                                                              int                       lda,
                                                              float                     percentage,
                                                              const hipsparseMatDescr_t descr,
                                                              int*                      csrRowPtr,
                                                              int*        nnzTotalDevHostPtr,
                                                              pruneInfo_t info,
                                                              void*       buffer)
    {
        return hipsparseSpruneDense2csrNnzByPercentage(
            handle, m, n, A, lda, percentage, descr, csrRowPtr, nnzTotalDevHostPtr, info, buffer);
    }

    template <>
    hipsparseStatus_t hipsparseXpruneDense2csrNnzByPercentage(hipsparseHandle_t         handle,
                                                              int                       m,
                                                              int                       n,
                                                              const double*             A,
                                                              int                       lda,
                                                              double                    percentage,
                                                              const hipsparseMatDescr_t descr,
                                                              int*                      csrRowPtr,
                                                              int*        nnzTotalDevHostPtr,
                                                              pruneInfo_t info,
                                                              void*       buffer)
    {
        return hipsparseDpruneDense2csrNnzByPercentage(
            handle, m, n, A, lda, percentage, descr, csrRowPtr, nnzTotalDevHostPtr, info, buffer);
    }
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 13000)
    template <>
    hipsparseStatus_t hipsparseXpruneDense2csrByPercentage(hipsparseHandle_t         handle,
                                                           int                       m,
                                                           int                       n,
                                                           const float*              A,
                                                           int                       lda,
                                                           float                     percentage,
                                                           const hipsparseMatDescr_t descr,
                                                           float*                    csrVal,
                                                           const int*                csrRowPtr,
                                                           int*                      csrColInd,
                                                           pruneInfo_t               info,
                                                           void*                     buffer)
    {
        return hipsparseSpruneDense2csrByPercentage(
            handle, m, n, A, lda, percentage, descr, csrVal, csrRowPtr, csrColInd, info, buffer);
    }

    template <>
    hipsparseStatus_t hipsparseXpruneDense2csrByPercentage(hipsparseHandle_t         handle,
                                                           int                       m,
                                                           int                       n,
                                                           const double*             A,
                                                           int                       lda,
                                                           double                    percentage,
                                                           const hipsparseMatDescr_t descr,
                                                           double*                   csrVal,
                                                           const int*                csrRowPtr,
                                                           int*                      csrColInd,
                                                           pruneInfo_t               info,
                                                           void*                     buffer)
    {
        return hipsparseDpruneDense2csrByPercentage(
            handle, m, n, A, lda, percentage, descr, csrVal, csrRowPtr, csrColInd, info, buffer);
    }
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 12000)
    template <>
    hipsparseStatus_t hipsparseXdense2csc(hipsparseHandle_t         handle,
                                          int                       m,
                                          int                       n,
                                          const hipsparseMatDescr_t descr,
                                          const float*              A,
                                          int                       ld,
                                          const int*                nnzPerColumn,
                                          float*                    cscVal,
                                          int*                      cscRowInd,
                                          int*                      cscColPtr)
    {
        return hipsparseSdense2csc(
            handle, m, n, descr, A, ld, nnzPerColumn, cscVal, cscRowInd, cscColPtr);
    }
    template <>
    hipsparseStatus_t hipsparseXdense2csc(hipsparseHandle_t         handle,
                                          int                       m,
                                          int                       n,
                                          const hipsparseMatDescr_t descr,
                                          const double*             A,
                                          int                       ld,
                                          const int*                nnzPerColumn,
                                          double*                   cscVal,
                                          int*                      cscRowInd,
                                          int*                      cscColPtr)
    {
        return hipsparseDdense2csc(
            handle, m, n, descr, A, ld, nnzPerColumn, cscVal, cscRowInd, cscColPtr);
    }
    template <>
    hipsparseStatus_t hipsparseXdense2csc(hipsparseHandle_t         handle,
                                          int                       m,
                                          int                       n,
                                          const hipsparseMatDescr_t descr,
                                          const hipComplex*         A,
                                          int                       ld,
                                          const int*                nnzPerColumn,
                                          hipComplex*               cscVal,
                                          int*                      cscRowInd,
                                          int*                      cscColPtr)
    {
        return hipsparseCdense2csc(
            handle, m, n, descr, A, ld, nnzPerColumn, cscVal, cscRowInd, cscColPtr);
    }
    template <>
    hipsparseStatus_t hipsparseXdense2csc(hipsparseHandle_t         handle,
                                          int                       m,
                                          int                       n,
                                          const hipsparseMatDescr_t descr,
                                          const hipDoubleComplex*   A,
                                          int                       ld,
                                          const int*                nnzPerColumn,
                                          hipDoubleComplex*         cscVal,
                                          int*                      cscRowInd,
                                          int*                      cscColPtr)
    {
        return hipsparseZdense2csc(
            handle, m, n, descr, A, ld, nnzPerColumn, cscVal, cscRowInd, cscColPtr);
    }
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 12000)
    template <>
    hipsparseStatus_t hipsparseXcsr2dense(hipsparseHandle_t         handle,
                                          int                       m,
                                          int                       n,
                                          const hipsparseMatDescr_t descr,
                                          const float*              csrVal,
                                          const int*                csrRowPtr,
                                          const int*                csrColInd,
                                          float*                    A,
                                          int                       ld)
    {
        return hipsparseScsr2dense(handle, m, n, descr, csrVal, csrRowPtr, csrColInd, A, ld);
    }
    template <>
    hipsparseStatus_t hipsparseXcsr2dense(hipsparseHandle_t         handle,
                                          int                       m,
                                          int                       n,
                                          const hipsparseMatDescr_t descr,
                                          const double*             csrVal,
                                          const int*                csrRowPtr,
                                          const int*                csrColInd,
                                          double*                   A,
                                          int                       ld)
    {
        return hipsparseDcsr2dense(handle, m, n, descr, csrVal, csrRowPtr, csrColInd, A, ld);
    }
    template <>
    hipsparseStatus_t hipsparseXcsr2dense(hipsparseHandle_t         handle,
                                          int                       m,
                                          int                       n,
                                          const hipsparseMatDescr_t descr,
                                          const hipComplex*         csrVal,
                                          const int*                csrRowPtr,
                                          const int*                csrColInd,
                                          hipComplex*               A,
                                          int                       ld)
    {
        return hipsparseCcsr2dense(handle, m, n, descr, csrVal, csrRowPtr, csrColInd, A, ld);
    }
    template <>
    hipsparseStatus_t hipsparseXcsr2dense(hipsparseHandle_t         handle,
                                          int                       m,
                                          int                       n,
                                          const hipsparseMatDescr_t descr,
                                          const hipDoubleComplex*   csrVal,
                                          const int*                csrRowPtr,
                                          const int*                csrColInd,
                                          hipDoubleComplex*         A,
                                          int                       ld)
    {
        return hipsparseZcsr2dense(handle, m, n, descr, csrVal, csrRowPtr, csrColInd, A, ld);
    }
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 12000)
    template <>
    hipsparseStatus_t hipsparseXcsc2dense(hipsparseHandle_t         handle,
                                          int                       m,
                                          int                       n,
                                          const hipsparseMatDescr_t descr,
                                          const float*              cscVal,
                                          const int*                cscRowInd,
                                          const int*                cscColPtr,
                                          float*                    A,
                                          int                       ld)
    {
        return hipsparseScsc2dense(handle, m, n, descr, cscVal, cscRowInd, cscColPtr, A, ld);
    }
    template <>
    hipsparseStatus_t hipsparseXcsc2dense(hipsparseHandle_t         handle,
                                          int                       m,
                                          int                       n,
                                          const hipsparseMatDescr_t descr,
                                          const double*             cscVal,
                                          const int*                cscRowInd,
                                          const int*                cscColPtr,
                                          double*                   A,
                                          int                       ld)
    {
        return hipsparseDcsc2dense(handle, m, n, descr, cscVal, cscRowInd, cscColPtr, A, ld);
    }
    template <>
    hipsparseStatus_t hipsparseXcsc2dense(hipsparseHandle_t         handle,
                                          int                       m,
                                          int                       n,
                                          const hipsparseMatDescr_t descr,
                                          const hipComplex*         cscVal,
                                          const int*                cscRowInd,
                                          const int*                cscColPtr,
                                          hipComplex*               A,
                                          int                       ld)
    {
        return hipsparseCcsc2dense(handle, m, n, descr, cscVal, cscRowInd, cscColPtr, A, ld);
    }
    template <>
    hipsparseStatus_t hipsparseXcsc2dense(hipsparseHandle_t         handle,
                                          int                       m,
                                          int                       n,
                                          const hipsparseMatDescr_t descr,
                                          const hipDoubleComplex*   cscVal,
                                          const int*                cscRowInd,
                                          const int*                cscColPtr,
                                          hipDoubleComplex*         A,
                                          int                       ld)
    {
        return hipsparseZcsc2dense(handle, m, n, descr, cscVal, cscRowInd, cscColPtr, A, ld);
    }
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 11000)
    template <>
    hipsparseStatus_t hipsparseXcsr2csc(hipsparseHandle_t    handle,
                                        int                  m,
                                        int                  n,
                                        int                  nnz,
                                        const float*         csrVal,
                                        const int*           csrRowPtr,
                                        const int*           csrColInd,
                                        float*               csc_val,
                                        int*                 cscRowInd,
                                        int*                 cscColPtr,
                                        hipsparseAction_t    copyValues,
                                        hipsparseIndexBase_t idxBase)
    {
        return hipsparseScsr2csc(handle,
                                 m,
                                 n,
                                 nnz,
                                 csrVal,
                                 csrRowPtr,
                                 csrColInd,
                                 csc_val,
                                 cscRowInd,
                                 cscColPtr,
                                 copyValues,
                                 idxBase);
    }

    template <>
    hipsparseStatus_t hipsparseXcsr2csc(hipsparseHandle_t    handle,
                                        int                  m,
                                        int                  n,
                                        int                  nnz,
                                        const double*        csrVal,
                                        const int*           csrRowPtr,
                                        const int*           csrColInd,
                                        double*              csc_val,
                                        int*                 cscRowInd,
                                        int*                 cscColPtr,
                                        hipsparseAction_t    copyValues,
                                        hipsparseIndexBase_t idxBase)
    {
        return hipsparseDcsr2csc(handle,
                                 m,
                                 n,
                                 nnz,
                                 csrVal,
                                 csrRowPtr,
                                 csrColInd,
                                 csc_val,
                                 cscRowInd,
                                 cscColPtr,
                                 copyValues,
                                 idxBase);
    }

    template <>
    hipsparseStatus_t hipsparseXcsr2csc(hipsparseHandle_t    handle,
                                        int                  m,
                                        int                  n,
                                        int                  nnz,
                                        const hipComplex*    csrVal,
                                        const int*           csrRowPtr,
                                        const int*           csrColInd,
                                        hipComplex*          csc_val,
                                        int*                 cscRowInd,
                                        int*                 cscColPtr,
                                        hipsparseAction_t    copyValues,
                                        hipsparseIndexBase_t idxBase)
    {
        return hipsparseCcsr2csc(handle,
                                 m,
                                 n,
                                 nnz,
                                 csrVal,
                                 csrRowPtr,
                                 csrColInd,
                                 csc_val,
                                 cscRowInd,
                                 cscColPtr,
                                 copyValues,
                                 idxBase);
    }

    template <>
    hipsparseStatus_t hipsparseXcsr2csc(hipsparseHandle_t       handle,
                                        int                     m,
                                        int                     n,
                                        int                     nnz,
                                        const hipDoubleComplex* csrVal,
                                        const int*              csrRowPtr,
                                        const int*              csrColInd,
                                        hipDoubleComplex*       csc_val,
                                        int*                    cscRowInd,
                                        int*                    cscColPtr,
                                        hipsparseAction_t       copyValues,
                                        hipsparseIndexBase_t    idxBase)
    {
        return hipsparseZcsr2csc(handle,
                                 m,
                                 n,
                                 nnz,
                                 csrVal,
                                 csrRowPtr,
                                 csrColInd,
                                 csc_val,
                                 cscRowInd,
                                 cscColPtr,
                                 copyValues,
                                 idxBase);
    }
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 11000)
    template <>
    hipsparseStatus_t hipsparseXcsr2hyb(hipsparseHandle_t         handle,
                                        int                       m,
                                        int                       n,
                                        const hipsparseMatDescr_t descr,
                                        const float*              csrVal,
                                        const int*                csrRowPtr,
                                        const int*                csrColInd,
                                        hipsparseHybMat_t         hyb,
                                        int                       user_ell_width,
                                        hipsparseHybPartition_t   partition_type)
    {
        return hipsparseScsr2hyb(
            handle, m, n, descr, csrVal, csrRowPtr, csrColInd, hyb, user_ell_width, partition_type);
    }

    template <>
    hipsparseStatus_t hipsparseXcsr2hyb(hipsparseHandle_t         handle,
                                        int                       m,
                                        int                       n,
                                        const hipsparseMatDescr_t descr,
                                        const double*             csrVal,
                                        const int*                csrRowPtr,
                                        const int*                csrColInd,
                                        hipsparseHybMat_t         hyb,
                                        int                       user_ell_width,
                                        hipsparseHybPartition_t   partition_type)
    {
        return hipsparseDcsr2hyb(
            handle, m, n, descr, csrVal, csrRowPtr, csrColInd, hyb, user_ell_width, partition_type);
    }

    template <>
    hipsparseStatus_t hipsparseXcsr2hyb(hipsparseHandle_t         handle,
                                        int                       m,
                                        int                       n,
                                        const hipsparseMatDescr_t descr,
                                        const hipComplex*         csrVal,
                                        const int*                csrRowPtr,
                                        const int*                csrColInd,
                                        hipsparseHybMat_t         hyb,
                                        int                       user_ell_width,
                                        hipsparseHybPartition_t   partition_type)
    {
        return hipsparseCcsr2hyb(
            handle, m, n, descr, csrVal, csrRowPtr, csrColInd, hyb, user_ell_width, partition_type);
    }

    template <>
    hipsparseStatus_t hipsparseXcsr2hyb(hipsparseHandle_t         handle,
                                        int                       m,
                                        int                       n,
                                        const hipsparseMatDescr_t descr,
                                        const hipDoubleComplex*   csrVal,
                                        const int*                csrRowPtr,
                                        const int*                csrColInd,
                                        hipsparseHybMat_t         hyb,
                                        int                       user_ell_width,
                                        hipsparseHybPartition_t   partition_type)
    {
        return hipsparseZcsr2hyb(
            handle, m, n, descr, csrVal, csrRowPtr, csrColInd, hyb, user_ell_width, partition_type);
    }
#endif

    template <>
    hipsparseStatus_t hipsparseXgebsr2gebsc_bufferSize(hipsparseHandle_t handle,
                                                       int               mb,
                                                       int               nb,
                                                       int               nnzb,
                                                       const float*      bsrVal,
                                                       const int*        bsrRowPtr,
                                                       const int*        bsrColInd,
                                                       int               rowBlockDim,
                                                       int               colBlockDim,
                                                       size_t*           pBufferSizeInBytes)
    {
        return hipsparseSgebsr2gebsc_bufferSize(handle,
                                                mb,
                                                nb,
                                                nnzb,
                                                bsrVal,
                                                bsrRowPtr,
                                                bsrColInd,
                                                rowBlockDim,
                                                colBlockDim,
                                                pBufferSizeInBytes);
    }

    template <>
    hipsparseStatus_t hipsparseXgebsr2gebsc_bufferSize(hipsparseHandle_t handle,
                                                       int               mb,
                                                       int               nb,
                                                       int               nnzb,
                                                       const double*     bsrVal,
                                                       const int*        bsrRowPtr,
                                                       const int*        bsrColInd,
                                                       int               rowBlockDim,
                                                       int               colBlockDim,
                                                       size_t*           pBufferSizeInBytes)
    {
        return hipsparseDgebsr2gebsc_bufferSize(handle,
                                                mb,
                                                nb,
                                                nnzb,
                                                bsrVal,
                                                bsrRowPtr,
                                                bsrColInd,
                                                rowBlockDim,
                                                colBlockDim,
                                                pBufferSizeInBytes);
    }

    template <>
    hipsparseStatus_t hipsparseXgebsr2gebsc_bufferSize(hipsparseHandle_t handle,
                                                       int               mb,
                                                       int               nb,
                                                       int               nnzb,
                                                       const hipComplex* bsrVal,
                                                       const int*        bsrRowPtr,
                                                       const int*        bsrColInd,
                                                       int               rowBlockDim,
                                                       int               colBlockDim,
                                                       size_t*           pBufferSizeInBytes)
    {
        return hipsparseCgebsr2gebsc_bufferSize(handle,
                                                mb,
                                                nb,
                                                nnzb,
                                                bsrVal,
                                                bsrRowPtr,
                                                bsrColInd,
                                                rowBlockDim,
                                                colBlockDim,
                                                pBufferSizeInBytes);
    }

    template <>
    hipsparseStatus_t hipsparseXgebsr2gebsc_bufferSize(hipsparseHandle_t       handle,
                                                       int                     mb,
                                                       int                     nb,
                                                       int                     nnzb,
                                                       const hipDoubleComplex* bsrVal,
                                                       const int*              bsrRowPtr,
                                                       const int*              bsrColInd,
                                                       int                     rowBlockDim,
                                                       int                     colBlockDim,
                                                       size_t*                 pBufferSizeInBytes)
    {
        return hipsparseZgebsr2gebsc_bufferSize(handle,
                                                mb,
                                                nb,
                                                nnzb,
                                                bsrVal,
                                                bsrRowPtr,
                                                bsrColInd,
                                                rowBlockDim,
                                                colBlockDim,
                                                pBufferSizeInBytes);
    }

    template <>
    hipsparseStatus_t hipsparseXgebsr2gebsc(hipsparseHandle_t    handle,
                                            int                  mb,
                                            int                  nb,
                                            int                  nnzb,
                                            const float*         bsrVal,
                                            const int*           bsrRowPtr,
                                            const int*           bsrColInd,
                                            int                  rowBlockDim,
                                            int                  colBlockDim,
                                            float*               bscVal,
                                            int*                 bscRowInd,
                                            int*                 bscColPtr,
                                            hipsparseAction_t    copyValues,
                                            hipsparseIndexBase_t idxBase,
                                            void*                temp_buffer)
    {
        return hipsparseSgebsr2gebsc(handle,
                                     mb,
                                     nb,
                                     nnzb,
                                     bsrVal,
                                     bsrRowPtr,
                                     bsrColInd,
                                     rowBlockDim,
                                     colBlockDim,
                                     bscVal,
                                     bscRowInd,
                                     bscColPtr,
                                     copyValues,
                                     idxBase,
                                     temp_buffer);
    }

    template <>
    hipsparseStatus_t hipsparseXgebsr2gebsc(hipsparseHandle_t    handle,
                                            int                  mb,
                                            int                  nb,
                                            int                  nnzb,
                                            const double*        bsrVal,
                                            const int*           bsrRowPtr,
                                            const int*           bsrColInd,
                                            int                  rowBlockDim,
                                            int                  colBlockDim,
                                            double*              bscVal,
                                            int*                 bscRowInd,
                                            int*                 bscColPtr,
                                            hipsparseAction_t    copyValues,
                                            hipsparseIndexBase_t idxBase,
                                            void*                temp_buffer)
    {
        return hipsparseDgebsr2gebsc(handle,
                                     mb,
                                     nb,
                                     nnzb,
                                     bsrVal,
                                     bsrRowPtr,
                                     bsrColInd,
                                     rowBlockDim,
                                     colBlockDim,
                                     bscVal,
                                     bscRowInd,
                                     bscColPtr,
                                     copyValues,
                                     idxBase,
                                     temp_buffer);
    }

    template <>
    hipsparseStatus_t hipsparseXgebsr2gebsc(hipsparseHandle_t    handle,
                                            int                  mb,
                                            int                  nb,
                                            int                  nnzb,
                                            const hipComplex*    bsrVal,
                                            const int*           bsrRowPtr,
                                            const int*           bsrColInd,
                                            int                  rowBlockDim,
                                            int                  colBlockDim,
                                            hipComplex*          bscVal,
                                            int*                 bscRowInd,
                                            int*                 bscColPtr,
                                            hipsparseAction_t    copyValues,
                                            hipsparseIndexBase_t idxBase,
                                            void*                temp_buffer)
    {
        return hipsparseCgebsr2gebsc(handle,
                                     mb,
                                     nb,
                                     nnzb,
                                     bsrVal,
                                     bsrRowPtr,
                                     bsrColInd,
                                     rowBlockDim,
                                     colBlockDim,
                                     bscVal,
                                     bscRowInd,
                                     bscColPtr,
                                     copyValues,
                                     idxBase,
                                     temp_buffer);
    }

    template <>
    hipsparseStatus_t hipsparseXgebsr2gebsc(hipsparseHandle_t       handle,
                                            int                     mb,
                                            int                     nb,
                                            int                     nnzb,
                                            const hipDoubleComplex* bsrVal,
                                            const int*              bsrRowPtr,
                                            const int*              bsrColInd,
                                            int                     rowBlockDim,
                                            int                     colBlockDim,
                                            hipDoubleComplex*       bscVal,
                                            int*                    bscRowInd,
                                            int*                    bscColPtr,
                                            hipsparseAction_t       copyValues,
                                            hipsparseIndexBase_t    idxBase,
                                            void*                   temp_buffer)
    {
        return hipsparseZgebsr2gebsc(handle,
                                     mb,
                                     nb,
                                     nnzb,
                                     bsrVal,
                                     bsrRowPtr,
                                     bsrColInd,
                                     rowBlockDim,
                                     colBlockDim,
                                     bscVal,
                                     bscRowInd,
                                     bscColPtr,
                                     copyValues,
                                     idxBase,
                                     temp_buffer);
    }

    template <>
    hipsparseStatus_t hipsparseXcsr2gebsr_bufferSize(hipsparseHandle_t         handle,
                                                     hipsparseDirection_t      dir,
                                                     int                       m,
                                                     int                       n,
                                                     const hipsparseMatDescr_t csr_descr,
                                                     const hipDoubleComplex*   csrVal,
                                                     const int*                csrRowPtr,
                                                     const int*                csrColInd,
                                                     int                       rowBlockDim,
                                                     int                       colBlockDim,
                                                     size_t*                   pBufferSizeInBytes)
    {
        return hipsparseZcsr2gebsr_bufferSize(handle,
                                              dir,
                                              m,
                                              n,
                                              csr_descr,
                                              csrVal,
                                              csrRowPtr,
                                              csrColInd,
                                              rowBlockDim,
                                              colBlockDim,
                                              pBufferSizeInBytes);
    }

    template <>
    hipsparseStatus_t hipsparseXcsr2gebsr_bufferSize(hipsparseHandle_t         handle,
                                                     hipsparseDirection_t      dir,
                                                     int                       m,
                                                     int                       n,
                                                     const hipsparseMatDescr_t csr_descr,
                                                     const float*              csrVal,
                                                     const int*                csrRowPtr,
                                                     const int*                csrColInd,
                                                     int                       rowBlockDim,
                                                     int                       colBlockDim,
                                                     size_t*                   pBufferSizeInBytes)
    {
        return hipsparseScsr2gebsr_bufferSize(handle,
                                              dir,
                                              m,
                                              n,
                                              csr_descr,
                                              csrVal,
                                              csrRowPtr,
                                              csrColInd,
                                              rowBlockDim,
                                              colBlockDim,
                                              pBufferSizeInBytes);
    }

    template <>
    hipsparseStatus_t hipsparseXcsr2gebsr_bufferSize(hipsparseHandle_t         handle,
                                                     hipsparseDirection_t      dir,
                                                     int                       m,
                                                     int                       n,
                                                     const hipsparseMatDescr_t csr_descr,
                                                     const double*             csrVal,
                                                     const int*                csrRowPtr,
                                                     const int*                csrColInd,
                                                     int                       rowBlockDim,
                                                     int                       colBlockDim,
                                                     size_t*                   pBufferSizeInBytes)
    {
        return hipsparseDcsr2gebsr_bufferSize(handle,
                                              dir,
                                              m,
                                              n,
                                              csr_descr,
                                              csrVal,
                                              csrRowPtr,
                                              csrColInd,
                                              rowBlockDim,
                                              colBlockDim,
                                              pBufferSizeInBytes);
    }

    template <>
    hipsparseStatus_t hipsparseXcsr2gebsr_bufferSize(hipsparseHandle_t         handle,
                                                     hipsparseDirection_t      dir,
                                                     int                       m,
                                                     int                       n,
                                                     const hipsparseMatDescr_t csr_descr,
                                                     const hipComplex*         csrVal,
                                                     const int*                csrRowPtr,
                                                     const int*                csrColInd,
                                                     int                       rowBlockDim,
                                                     int                       colBlockDim,
                                                     size_t*                   pBufferSizeInBytes)
    {
        return hipsparseCcsr2gebsr_bufferSize(handle,
                                              dir,
                                              m,
                                              n,
                                              csr_descr,
                                              csrVal,
                                              csrRowPtr,
                                              csrColInd,
                                              rowBlockDim,
                                              colBlockDim,
                                              pBufferSizeInBytes);
    }

    template <>
    hipsparseStatus_t hipsparseXcsr2gebsr(hipsparseHandle_t         handle,
                                          hipsparseDirection_t      dir,
                                          int                       m,
                                          int                       n,
                                          const hipsparseMatDescr_t csr_descr,
                                          const float*              csrVal,
                                          const int*                csrRowPtr,
                                          const int*                csrColInd,
                                          const hipsparseMatDescr_t bsr_descr,
                                          float*                    bsrVal,
                                          int*                      bsrRowPtr,
                                          int*                      bsrColInd,
                                          int                       rowBlockDim,
                                          int                       colBlockDim,
                                          void*                     pbuffer)
    {
        return hipsparseScsr2gebsr(handle,
                                   dir,
                                   m,
                                   n,
                                   csr_descr,
                                   csrVal,
                                   csrRowPtr,
                                   csrColInd,
                                   bsr_descr,
                                   bsrVal,
                                   bsrRowPtr,
                                   bsrColInd,
                                   rowBlockDim,
                                   colBlockDim,
                                   pbuffer);
    }

    template <>
    hipsparseStatus_t hipsparseXcsr2gebsr(hipsparseHandle_t         handle,
                                          hipsparseDirection_t      dir,
                                          int                       m,
                                          int                       n,
                                          const hipsparseMatDescr_t csr_descr,
                                          const double*             csrVal,
                                          const int*                csrRowPtr,
                                          const int*                csrColInd,
                                          const hipsparseMatDescr_t bsr_descr,
                                          double*                   bsrVal,
                                          int*                      bsrRowPtr,
                                          int*                      bsrColInd,
                                          int                       rowBlockDim,
                                          int                       colBlockDim,
                                          void*                     pbuffer)
    {
        return hipsparseDcsr2gebsr(handle,
                                   dir,
                                   m,
                                   n,
                                   csr_descr,
                                   csrVal,
                                   csrRowPtr,
                                   csrColInd,
                                   bsr_descr,
                                   bsrVal,
                                   bsrRowPtr,
                                   bsrColInd,
                                   rowBlockDim,
                                   colBlockDim,
                                   pbuffer);
    }

    template <>
    hipsparseStatus_t hipsparseXcsr2gebsr(hipsparseHandle_t         handle,
                                          hipsparseDirection_t      dir,
                                          int                       m,
                                          int                       n,
                                          const hipsparseMatDescr_t csr_descr,
                                          const hipComplex*         csrVal,
                                          const int*                csrRowPtr,
                                          const int*                csrColInd,
                                          const hipsparseMatDescr_t bsr_descr,
                                          hipComplex*               bsrVal,
                                          int*                      bsrRowPtr,
                                          int*                      bsrColInd,
                                          int                       rowBlockDim,
                                          int                       colBlockDim,
                                          void*                     pbuffer)
    {
        return hipsparseCcsr2gebsr(handle,
                                   dir,
                                   m,
                                   n,
                                   csr_descr,
                                   csrVal,
                                   csrRowPtr,
                                   csrColInd,
                                   bsr_descr,
                                   bsrVal,
                                   bsrRowPtr,
                                   bsrColInd,
                                   rowBlockDim,
                                   colBlockDim,
                                   pbuffer);
    }

    template <>
    hipsparseStatus_t hipsparseXcsr2gebsr(hipsparseHandle_t         handle,
                                          hipsparseDirection_t      dir,
                                          int                       m,
                                          int                       n,
                                          const hipsparseMatDescr_t csr_descr,
                                          const hipDoubleComplex*   csrVal,
                                          const int*                csrRowPtr,
                                          const int*                csrColInd,
                                          const hipsparseMatDescr_t bsr_descr,
                                          hipDoubleComplex*         bsrVal,
                                          int*                      bsrRowPtr,
                                          int*                      bsrColInd,
                                          int                       rowBlockDim,
                                          int                       colBlockDim,
                                          void*                     pbuffer)
    {
        return hipsparseZcsr2gebsr(handle,
                                   dir,
                                   m,
                                   n,
                                   csr_descr,
                                   csrVal,
                                   csrRowPtr,
                                   csrColInd,
                                   bsr_descr,
                                   bsrVal,
                                   bsrRowPtr,
                                   bsrColInd,
                                   rowBlockDim,
                                   colBlockDim,
                                   pbuffer);
    }

    template <>
    hipsparseStatus_t hipsparseXcsr2bsr(hipsparseHandle_t         handle,
                                        hipsparseDirection_t      dirA,
                                        int                       m,
                                        int                       n,
                                        const hipsparseMatDescr_t descrA,
                                        const float*              csrValA,
                                        const int*                csrRowPtrA,
                                        const int*                csrColIndA,
                                        int                       blockDim,
                                        const hipsparseMatDescr_t descrC,
                                        float*                    bsrValC,
                                        int*                      bsrRowPtrC,
                                        int*                      bsrColIndC)
    {
        return hipsparseScsr2bsr(handle,
                                 dirA,
                                 m,
                                 n,
                                 descrA,
                                 csrValA,
                                 csrRowPtrA,
                                 csrColIndA,
                                 blockDim,
                                 descrC,
                                 bsrValC,
                                 bsrRowPtrC,
                                 bsrColIndC);
    }

    template <>
    hipsparseStatus_t hipsparseXcsr2bsr(hipsparseHandle_t         handle,
                                        hipsparseDirection_t      dirA,
                                        int                       m,
                                        int                       n,
                                        const hipsparseMatDescr_t descrA,
                                        const double*             csrValA,
                                        const int*                csrRowPtrA,
                                        const int*                csrColIndA,
                                        int                       blockDim,
                                        const hipsparseMatDescr_t descrC,
                                        double*                   bsrValC,
                                        int*                      bsrRowPtrC,
                                        int*                      bsrColIndC)
    {
        return hipsparseDcsr2bsr(handle,
                                 dirA,
                                 m,
                                 n,
                                 descrA,
                                 csrValA,
                                 csrRowPtrA,
                                 csrColIndA,
                                 blockDim,
                                 descrC,
                                 bsrValC,
                                 bsrRowPtrC,
                                 bsrColIndC);
    }

    template <>
    hipsparseStatus_t hipsparseXcsr2bsr(hipsparseHandle_t         handle,
                                        hipsparseDirection_t      dirA,
                                        int                       m,
                                        int                       n,
                                        const hipsparseMatDescr_t descrA,
                                        const hipComplex*         csrValA,
                                        const int*                csrRowPtrA,
                                        const int*                csrColIndA,
                                        int                       blockDim,
                                        const hipsparseMatDescr_t descrC,
                                        hipComplex*               bsrValC,
                                        int*                      bsrRowPtrC,
                                        int*                      bsrColIndC)
    {
        return hipsparseCcsr2bsr(handle,
                                 dirA,
                                 m,
                                 n,
                                 descrA,
                                 csrValA,
                                 csrRowPtrA,
                                 csrColIndA,
                                 blockDim,
                                 descrC,
                                 bsrValC,
                                 bsrRowPtrC,
                                 bsrColIndC);
    }

    template <>
    hipsparseStatus_t hipsparseXcsr2bsr(hipsparseHandle_t         handle,
                                        hipsparseDirection_t      dirA,
                                        int                       m,
                                        int                       n,
                                        const hipsparseMatDescr_t descrA,
                                        const hipDoubleComplex*   csrValA,
                                        const int*                csrRowPtrA,
                                        const int*                csrColIndA,
                                        int                       blockDim,
                                        const hipsparseMatDescr_t descrC,
                                        hipDoubleComplex*         bsrValC,
                                        int*                      bsrRowPtrC,
                                        int*                      bsrColIndC)
    {
        return hipsparseZcsr2bsr(handle,
                                 dirA,
                                 m,
                                 n,
                                 descrA,
                                 csrValA,
                                 csrRowPtrA,
                                 csrColIndA,
                                 blockDim,
                                 descrC,
                                 bsrValC,
                                 bsrRowPtrC,
                                 bsrColIndC);
    }

    template <>
    hipsparseStatus_t hipsparseXbsr2csr(hipsparseHandle_t         handle,
                                        hipsparseDirection_t      dirA,
                                        int                       mb,
                                        int                       nb,
                                        const hipsparseMatDescr_t descrA,
                                        const float*              bsrValA,
                                        const int*                bsrRowPtrA,
                                        const int*                bsrColIndA,
                                        int                       blockDim,
                                        const hipsparseMatDescr_t descrC,
                                        float*                    csrValC,
                                        int*                      csrRowPtrC,
                                        int*                      csrColIndC)
    {
        return hipsparseSbsr2csr(handle,
                                 dirA,
                                 mb,
                                 nb,
                                 descrA,
                                 bsrValA,
                                 bsrRowPtrA,
                                 bsrColIndA,
                                 blockDim,
                                 descrC,
                                 csrValC,
                                 csrRowPtrC,
                                 csrColIndC);
    }

    template <>
    hipsparseStatus_t hipsparseXbsr2csr(hipsparseHandle_t         handle,
                                        hipsparseDirection_t      dirA,
                                        int                       mb,
                                        int                       nb,
                                        const hipsparseMatDescr_t descrA,
                                        const double*             bsrValA,
                                        const int*                bsrRowPtrA,
                                        const int*                bsrColIndA,
                                        int                       blockDim,
                                        const hipsparseMatDescr_t descrC,
                                        double*                   csrValC,
                                        int*                      csrRowPtrC,
                                        int*                      csrColIndC)
    {
        return hipsparseDbsr2csr(handle,
                                 dirA,
                                 mb,
                                 nb,
                                 descrA,
                                 bsrValA,
                                 bsrRowPtrA,
                                 bsrColIndA,
                                 blockDim,
                                 descrC,
                                 csrValC,
                                 csrRowPtrC,
                                 csrColIndC);
    }

    template <>
    hipsparseStatus_t hipsparseXbsr2csr(hipsparseHandle_t         handle,
                                        hipsparseDirection_t      dirA,
                                        int                       mb,
                                        int                       nb,
                                        const hipsparseMatDescr_t descrA,
                                        const hipComplex*         bsrValA,
                                        const int*                bsrRowPtrA,
                                        const int*                bsrColIndA,
                                        int                       blockDim,
                                        const hipsparseMatDescr_t descrC,
                                        hipComplex*               csrValC,
                                        int*                      csrRowPtrC,
                                        int*                      csrColIndC)
    {
        return hipsparseCbsr2csr(handle,
                                 dirA,
                                 mb,
                                 nb,
                                 descrA,
                                 bsrValA,
                                 bsrRowPtrA,
                                 bsrColIndA,
                                 blockDim,
                                 descrC,
                                 csrValC,
                                 csrRowPtrC,
                                 csrColIndC);
    }

    template <>
    hipsparseStatus_t hipsparseXbsr2csr(hipsparseHandle_t         handle,
                                        hipsparseDirection_t      dirA,
                                        int                       mb,
                                        int                       nb,
                                        const hipsparseMatDescr_t descrA,
                                        const hipDoubleComplex*   bsrValA,
                                        const int*                bsrRowPtrA,
                                        const int*                bsrColIndA,
                                        int                       blockDim,
                                        const hipsparseMatDescr_t descrC,
                                        hipDoubleComplex*         csrValC,
                                        int*                      csrRowPtrC,
                                        int*                      csrColIndC)
    {
        return hipsparseZbsr2csr(handle,
                                 dirA,
                                 mb,
                                 nb,
                                 descrA,
                                 bsrValA,
                                 bsrRowPtrA,
                                 bsrColIndA,
                                 blockDim,
                                 descrC,
                                 csrValC,
                                 csrRowPtrC,
                                 csrColIndC);
    }

    template <>
    hipsparseStatus_t hipsparseXgebsr2csr(hipsparseHandle_t         handle,
                                          hipsparseDirection_t      dirA,
                                          int                       mb,
                                          int                       nb,
                                          const hipsparseMatDescr_t descrA,
                                          const float*              bsrValA,
                                          const int*                bsrRowPtrA,
                                          const int*                bsrColIndA,
                                          int                       rowBlockDim,
                                          int                       colBlockDim,
                                          const hipsparseMatDescr_t descrC,
                                          float*                    csrValC,
                                          int*                      csrRowPtrC,
                                          int*                      csrColIndC)
    {
        return hipsparseSgebsr2csr(handle,
                                   dirA,
                                   mb,
                                   nb,
                                   descrA,
                                   bsrValA,
                                   bsrRowPtrA,
                                   bsrColIndA,
                                   rowBlockDim,
                                   colBlockDim,
                                   descrC,
                                   csrValC,
                                   csrRowPtrC,
                                   csrColIndC);
    }

    template <>
    hipsparseStatus_t hipsparseXgebsr2csr(hipsparseHandle_t         handle,
                                          hipsparseDirection_t      dirA,
                                          int                       mb,
                                          int                       nb,
                                          const hipsparseMatDescr_t descrA,
                                          const double*             bsrValA,
                                          const int*                bsrRowPtrA,
                                          const int*                bsrColIndA,
                                          int                       rowBlockDim,
                                          int                       colBlockDim,
                                          const hipsparseMatDescr_t descrC,
                                          double*                   csrValC,
                                          int*                      csrRowPtrC,
                                          int*                      csrColIndC)
    {
        return hipsparseDgebsr2csr(handle,
                                   dirA,
                                   mb,
                                   nb,
                                   descrA,
                                   bsrValA,
                                   bsrRowPtrA,
                                   bsrColIndA,
                                   rowBlockDim,
                                   colBlockDim,
                                   descrC,
                                   csrValC,
                                   csrRowPtrC,
                                   csrColIndC);
    }

    template <>
    hipsparseStatus_t hipsparseXgebsr2csr(hipsparseHandle_t         handle,
                                          hipsparseDirection_t      dirA,
                                          int                       mb,
                                          int                       nb,
                                          const hipsparseMatDescr_t descrA,
                                          const hipComplex*         bsrValA,
                                          const int*                bsrRowPtrA,
                                          const int*                bsrColIndA,
                                          int                       rowBlockDim,
                                          int                       colBlockDim,
                                          const hipsparseMatDescr_t descrC,
                                          hipComplex*               csrValC,
                                          int*                      csrRowPtrC,
                                          int*                      csrColIndC)
    {
        return hipsparseCgebsr2csr(handle,
                                   dirA,
                                   mb,
                                   nb,
                                   descrA,
                                   bsrValA,
                                   bsrRowPtrA,
                                   bsrColIndA,
                                   rowBlockDim,
                                   colBlockDim,
                                   descrC,
                                   csrValC,
                                   csrRowPtrC,
                                   csrColIndC);
    }

    template <>
    hipsparseStatus_t hipsparseXgebsr2csr(hipsparseHandle_t         handle,
                                          hipsparseDirection_t      dirA,
                                          int                       mb,
                                          int                       nb,
                                          const hipsparseMatDescr_t descrA,
                                          const hipDoubleComplex*   bsrValA,
                                          const int*                bsrRowPtrA,
                                          const int*                bsrColIndA,
                                          int                       rowBlockDim,
                                          int                       colBlockDim,
                                          const hipsparseMatDescr_t descrC,
                                          hipDoubleComplex*         csrValC,
                                          int*                      csrRowPtrC,
                                          int*                      csrColIndC)
    {
        return hipsparseZgebsr2csr(handle,
                                   dirA,
                                   mb,
                                   nb,
                                   descrA,
                                   bsrValA,
                                   bsrRowPtrA,
                                   bsrColIndA,
                                   rowBlockDim,
                                   colBlockDim,
                                   descrC,
                                   csrValC,
                                   csrRowPtrC,
                                   csrColIndC);
    }

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 11000)
    template <>
    hipsparseStatus_t hipsparseXhyb2csr(hipsparseHandle_t         handle,
                                        const hipsparseMatDescr_t descrA,
                                        const hipsparseHybMat_t   hybA,
                                        float*                    csrValA,
                                        int*                      csrRowPtrA,
                                        int*                      csrColIndA)
    {
        return hipsparseShyb2csr(handle, descrA, hybA, csrValA, csrRowPtrA, csrColIndA);
    }

    template <>
    hipsparseStatus_t hipsparseXhyb2csr(hipsparseHandle_t         handle,
                                        const hipsparseMatDescr_t descrA,
                                        const hipsparseHybMat_t   hybA,
                                        double*                   csrValA,
                                        int*                      csrRowPtrA,
                                        int*                      csrColIndA)
    {
        return hipsparseDhyb2csr(handle, descrA, hybA, csrValA, csrRowPtrA, csrColIndA);
    }

    template <>
    hipsparseStatus_t hipsparseXhyb2csr(hipsparseHandle_t         handle,
                                        const hipsparseMatDescr_t descrA,
                                        const hipsparseHybMat_t   hybA,
                                        hipComplex*               csrValA,
                                        int*                      csrRowPtrA,
                                        int*                      csrColIndA)
    {
        return hipsparseChyb2csr(handle, descrA, hybA, csrValA, csrRowPtrA, csrColIndA);
    }

    template <>
    hipsparseStatus_t hipsparseXhyb2csr(hipsparseHandle_t         handle,
                                        const hipsparseMatDescr_t descrA,
                                        const hipsparseHybMat_t   hybA,
                                        hipDoubleComplex*         csrValA,
                                        int*                      csrRowPtrA,
                                        int*                      csrColIndA)
    {
        return hipsparseZhyb2csr(handle, descrA, hybA, csrValA, csrRowPtrA, csrColIndA);
    }
#endif

    template <>
    hipsparseStatus_t hipsparseXcsr2csr_compress(hipsparseHandle_t         handle,
                                                 int                       m,
                                                 int                       n,
                                                 const hipsparseMatDescr_t descrA,
                                                 const float*              csrValA,
                                                 const int*                csrColIndA,
                                                 const int*                csrRowPtrA,
                                                 int                       nnzA,
                                                 const int*                nnzPerRow,
                                                 float*                    csrValC,
                                                 int*                      csrColIndC,
                                                 int*                      csrRowPtrC,
                                                 float                     tol)
    {
        return hipsparseScsr2csr_compress(handle,
                                          m,
                                          n,
                                          descrA,
                                          csrValA,
                                          csrColIndA,
                                          csrRowPtrA,
                                          nnzA,
                                          nnzPerRow,
                                          csrValC,
                                          csrColIndC,
                                          csrRowPtrC,
                                          tol);
    }

    template <>
    hipsparseStatus_t hipsparseXcsr2csr_compress(hipsparseHandle_t         handle,
                                                 int                       m,
                                                 int                       n,
                                                 const hipsparseMatDescr_t descrA,
                                                 const double*             csrValA,
                                                 const int*                csrColIndA,
                                                 const int*                csrRowPtrA,
                                                 int                       nnzA,
                                                 const int*                nnzPerRow,
                                                 double*                   csrValC,
                                                 int*                      csrColIndC,
                                                 int*                      csrRowPtrC,
                                                 double                    tol)
    {
        return hipsparseDcsr2csr_compress(handle,
                                          m,
                                          n,
                                          descrA,
                                          csrValA,
                                          csrColIndA,
                                          csrRowPtrA,
                                          nnzA,
                                          nnzPerRow,
                                          csrValC,
                                          csrColIndC,
                                          csrRowPtrC,
                                          tol);
    }

    template <>
    hipsparseStatus_t hipsparseXcsr2csr_compress(hipsparseHandle_t         handle,
                                                 int                       m,
                                                 int                       n,
                                                 const hipsparseMatDescr_t descrA,
                                                 const hipComplex*         csrValA,
                                                 const int*                csrColIndA,
                                                 const int*                csrRowPtrA,
                                                 int                       nnzA,
                                                 const int*                nnzPerRow,
                                                 hipComplex*               csrValC,
                                                 int*                      csrColIndC,
                                                 int*                      csrRowPtrC,
                                                 hipComplex                tol)
    {
        return hipsparseCcsr2csr_compress(handle,
                                          m,
                                          n,
                                          descrA,
                                          csrValA,
                                          csrColIndA,
                                          csrRowPtrA,
                                          nnzA,
                                          nnzPerRow,
                                          csrValC,
                                          csrColIndC,
                                          csrRowPtrC,
                                          tol);
    }

    template <>
    hipsparseStatus_t hipsparseXcsr2csr_compress(hipsparseHandle_t         handle,
                                                 int                       m,
                                                 int                       n,
                                                 const hipsparseMatDescr_t descrA,
                                                 const hipDoubleComplex*   csrValA,
                                                 const int*                csrColIndA,
                                                 const int*                csrRowPtrA,
                                                 int                       nnzA,
                                                 const int*                nnzPerRow,
                                                 hipDoubleComplex*         csrValC,
                                                 int*                      csrColIndC,
                                                 int*                      csrRowPtrC,
                                                 hipDoubleComplex          tol)
    {
        return hipsparseZcsr2csr_compress(handle,
                                          m,
                                          n,
                                          descrA,
                                          csrValA,
                                          csrColIndA,
                                          csrRowPtrA,
                                          nnzA,
                                          nnzPerRow,
                                          csrValC,
                                          csrColIndC,
                                          csrRowPtrC,
                                          tol);
    }

    template <>
    hipsparseStatus_t hipsparseXpruneCsr2csr_bufferSize(hipsparseHandle_t         handle,
                                                        int                       m,
                                                        int                       n,
                                                        int                       nnzA,
                                                        const hipsparseMatDescr_t descrA,
                                                        const float*              csrValA,
                                                        const int*                csrRowPtrA,
                                                        const int*                csrColIndA,
                                                        const float*              threshold,
                                                        const hipsparseMatDescr_t descrC,
                                                        const float*              csrValC,
                                                        const int*                csrRowPtrC,
                                                        const int*                csrColIndC,
                                                        size_t* pBufferSizeInBytes)
    {
        return hipsparseSpruneCsr2csr_bufferSize(handle,
                                                 m,
                                                 n,
                                                 nnzA,
                                                 descrA,
                                                 csrValA,
                                                 csrRowPtrA,
                                                 csrColIndA,
                                                 threshold,
                                                 descrC,
                                                 csrValC,
                                                 csrRowPtrC,
                                                 csrColIndC,
                                                 pBufferSizeInBytes);
    }

    template <>
    hipsparseStatus_t hipsparseXpruneCsr2csr_bufferSize(hipsparseHandle_t         handle,
                                                        int                       m,
                                                        int                       n,
                                                        int                       nnzA,
                                                        const hipsparseMatDescr_t descrA,
                                                        const double*             csrValA,
                                                        const int*                csrRowPtrA,
                                                        const int*                csrColIndA,
                                                        const double*             threshold,
                                                        const hipsparseMatDescr_t descrC,
                                                        const double*             csrValC,
                                                        const int*                csrRowPtrC,
                                                        const int*                csrColIndC,
                                                        size_t* pBufferSizeInBytes)
    {
        return hipsparseDpruneCsr2csr_bufferSize(handle,
                                                 m,
                                                 n,
                                                 nnzA,
                                                 descrA,
                                                 csrValA,
                                                 csrRowPtrA,
                                                 csrColIndA,
                                                 threshold,
                                                 descrC,
                                                 csrValC,
                                                 csrRowPtrC,
                                                 csrColIndC,
                                                 pBufferSizeInBytes);
    }

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 13000)
    template <>
    hipsparseStatus_t hipsparseXpruneCsr2csr_bufferSizeExt(hipsparseHandle_t         handle,
                                                           int                       m,
                                                           int                       n,
                                                           int                       nnzA,
                                                           const hipsparseMatDescr_t descrA,
                                                           const float*              csrValA,
                                                           const int*                csrRowPtrA,
                                                           const int*                csrColIndA,
                                                           const float*              threshold,
                                                           const hipsparseMatDescr_t descrC,
                                                           const float*              csrValC,
                                                           const int*                csrRowPtrC,
                                                           const int*                csrColIndC,
                                                           size_t* pBufferSizeInBytes)
    {
        return hipsparseSpruneCsr2csr_bufferSizeExt(handle,
                                                    m,
                                                    n,
                                                    nnzA,
                                                    descrA,
                                                    csrValA,
                                                    csrRowPtrA,
                                                    csrColIndA,
                                                    threshold,
                                                    descrC,
                                                    csrValC,
                                                    csrRowPtrC,
                                                    csrColIndC,
                                                    pBufferSizeInBytes);
    }

    template <>
    hipsparseStatus_t hipsparseXpruneCsr2csr_bufferSizeExt(hipsparseHandle_t         handle,
                                                           int                       m,
                                                           int                       n,
                                                           int                       nnzA,
                                                           const hipsparseMatDescr_t descrA,
                                                           const double*             csrValA,
                                                           const int*                csrRowPtrA,
                                                           const int*                csrColIndA,
                                                           const double*             threshold,
                                                           const hipsparseMatDescr_t descrC,
                                                           const double*             csrValC,
                                                           const int*                csrRowPtrC,
                                                           const int*                csrColIndC,
                                                           size_t* pBufferSizeInBytes)
    {
        return hipsparseDpruneCsr2csr_bufferSizeExt(handle,
                                                    m,
                                                    n,
                                                    nnzA,
                                                    descrA,
                                                    csrValA,
                                                    csrRowPtrA,
                                                    csrColIndA,
                                                    threshold,
                                                    descrC,
                                                    csrValC,
                                                    csrRowPtrC,
                                                    csrColIndC,
                                                    pBufferSizeInBytes);
    }
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 13000)
    template <>
    hipsparseStatus_t hipsparseXpruneCsr2csrNnz(hipsparseHandle_t         handle,
                                                int                       m,
                                                int                       n,
                                                int                       nnzA,
                                                const hipsparseMatDescr_t descrA,
                                                const float*              csrValA,
                                                const int*                csrRowPtrA,
                                                const int*                csrColIndA,
                                                const float*              threshold,
                                                const hipsparseMatDescr_t descrC,
                                                int*                      csrRowPtrC,
                                                int*                      nnzTotalDevHostPtr,
                                                void*                     buffer)
    {
        return hipsparseSpruneCsr2csrNnz(handle,
                                         m,
                                         n,
                                         nnzA,
                                         descrA,
                                         csrValA,
                                         csrRowPtrA,
                                         csrColIndA,
                                         threshold,
                                         descrC,
                                         csrRowPtrC,
                                         nnzTotalDevHostPtr,
                                         buffer);
    }

    template <>
    hipsparseStatus_t hipsparseXpruneCsr2csrNnz(hipsparseHandle_t         handle,
                                                int                       m,
                                                int                       n,
                                                int                       nnzA,
                                                const hipsparseMatDescr_t descrA,
                                                const double*             csrValA,
                                                const int*                csrRowPtrA,
                                                const int*                csrColIndA,
                                                const double*             threshold,
                                                const hipsparseMatDescr_t descrC,
                                                int*                      csrRowPtrC,
                                                int*                      nnzTotalDevHostPtr,
                                                void*                     buffer)
    {
        return hipsparseDpruneCsr2csrNnz(handle,
                                         m,
                                         n,
                                         nnzA,
                                         descrA,
                                         csrValA,
                                         csrRowPtrA,
                                         csrColIndA,
                                         threshold,
                                         descrC,
                                         csrRowPtrC,
                                         nnzTotalDevHostPtr,
                                         buffer);
    }
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 13000)
    template <>
    hipsparseStatus_t hipsparseXpruneCsr2csr(hipsparseHandle_t         handle,
                                             int                       m,
                                             int                       n,
                                             int                       nnzA,
                                             const hipsparseMatDescr_t descrA,
                                             const float*              csrValA,
                                             const int*                csrRowPtrA,
                                             const int*                csrColIndA,
                                             const float*              threshold,
                                             const hipsparseMatDescr_t descrC,
                                             float*                    csrValC,
                                             const int*                csrRowPtrC,
                                             int*                      csrColIndC,
                                             void*                     buffer)
    {
        return hipsparseSpruneCsr2csr(handle,
                                      m,
                                      n,
                                      nnzA,
                                      descrA,
                                      csrValA,
                                      csrRowPtrA,
                                      csrColIndA,
                                      threshold,
                                      descrC,
                                      csrValC,
                                      csrRowPtrC,
                                      csrColIndC,
                                      buffer);
    }

    template <>
    hipsparseStatus_t hipsparseXpruneCsr2csr(hipsparseHandle_t         handle,
                                             int                       m,
                                             int                       n,
                                             int                       nnzA,
                                             const hipsparseMatDescr_t descrA,
                                             const double*             csrValA,
                                             const int*                csrRowPtrA,
                                             const int*                csrColIndA,
                                             const double*             threshold,
                                             const hipsparseMatDescr_t descrC,
                                             double*                   csrValC,
                                             const int*                csrRowPtrC,
                                             int*                      csrColIndC,
                                             void*                     buffer)
    {
        return hipsparseDpruneCsr2csr(handle,
                                      m,
                                      n,
                                      nnzA,
                                      descrA,
                                      csrValA,
                                      csrRowPtrA,
                                      csrColIndA,
                                      threshold,
                                      descrC,
                                      csrValC,
                                      csrRowPtrC,
                                      csrColIndC,
                                      buffer);
    }
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 13000)
    template <>
    hipsparseStatus_t
        hipsparseXpruneCsr2csrByPercentage_bufferSize(hipsparseHandle_t         handle,
                                                      int                       m,
                                                      int                       n,
                                                      int                       nnzA,
                                                      const hipsparseMatDescr_t descrA,
                                                      const float*              csrValA,
                                                      const int*                csrRowPtrA,
                                                      const int*                csrColIndA,
                                                      float                     percentage,
                                                      const hipsparseMatDescr_t descrC,
                                                      const float*              csrValC,
                                                      const int*                csrRowPtrC,
                                                      const int*                csrColIndC,
                                                      pruneInfo_t               info,
                                                      size_t*                   pBufferSizeInBytes)
    {
        return hipsparseSpruneCsr2csrByPercentage_bufferSize(handle,
                                                             m,
                                                             n,
                                                             nnzA,
                                                             descrA,
                                                             csrValA,
                                                             csrRowPtrA,
                                                             csrColIndA,
                                                             percentage,
                                                             descrC,
                                                             csrValC,
                                                             csrRowPtrC,
                                                             csrColIndC,
                                                             info,
                                                             pBufferSizeInBytes);
    }

    template <>
    hipsparseStatus_t
        hipsparseXpruneCsr2csrByPercentage_bufferSize(hipsparseHandle_t         handle,
                                                      int                       m,
                                                      int                       n,
                                                      int                       nnzA,
                                                      const hipsparseMatDescr_t descrA,
                                                      const double*             csrValA,
                                                      const int*                csrRowPtrA,
                                                      const int*                csrColIndA,
                                                      double                    percentage,
                                                      const hipsparseMatDescr_t descrC,
                                                      const double*             csrValC,
                                                      const int*                csrRowPtrC,
                                                      const int*                csrColIndC,
                                                      pruneInfo_t               info,
                                                      size_t*                   pBufferSizeInBytes)
    {
        return hipsparseDpruneCsr2csrByPercentage_bufferSize(handle,
                                                             m,
                                                             n,
                                                             nnzA,
                                                             descrA,
                                                             csrValA,
                                                             csrRowPtrA,
                                                             csrColIndA,
                                                             percentage,
                                                             descrC,
                                                             csrValC,
                                                             csrRowPtrC,
                                                             csrColIndC,
                                                             info,
                                                             pBufferSizeInBytes);
    }
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 13000)
    template <>
    hipsparseStatus_t
        hipsparseXpruneCsr2csrByPercentage_bufferSizeExt(hipsparseHandle_t         handle,
                                                         int                       m,
                                                         int                       n,
                                                         int                       nnzA,
                                                         const hipsparseMatDescr_t descrA,
                                                         const float*              csrValA,
                                                         const int*                csrRowPtrA,
                                                         const int*                csrColIndA,
                                                         float                     percentage,
                                                         const hipsparseMatDescr_t descrC,
                                                         const float*              csrValC,
                                                         const int*                csrRowPtrC,
                                                         const int*                csrColIndC,
                                                         pruneInfo_t               info,
                                                         size_t* pBufferSizeInBytes)
    {
        return hipsparseSpruneCsr2csrByPercentage_bufferSizeExt(handle,
                                                                m,
                                                                n,
                                                                nnzA,
                                                                descrA,
                                                                csrValA,
                                                                csrRowPtrA,
                                                                csrColIndA,
                                                                percentage,
                                                                descrC,
                                                                csrValC,
                                                                csrRowPtrC,
                                                                csrColIndC,
                                                                info,
                                                                pBufferSizeInBytes);
    }

    template <>
    hipsparseStatus_t
        hipsparseXpruneCsr2csrByPercentage_bufferSizeExt(hipsparseHandle_t         handle,
                                                         int                       m,
                                                         int                       n,
                                                         int                       nnzA,
                                                         const hipsparseMatDescr_t descrA,
                                                         const double*             csrValA,
                                                         const int*                csrRowPtrA,
                                                         const int*                csrColIndA,
                                                         double                    percentage,
                                                         const hipsparseMatDescr_t descrC,
                                                         const double*             csrValC,
                                                         const int*                csrRowPtrC,
                                                         const int*                csrColIndC,
                                                         pruneInfo_t               info,
                                                         size_t* pBufferSizeInBytes)
    {
        return hipsparseDpruneCsr2csrByPercentage_bufferSizeExt(handle,
                                                                m,
                                                                n,
                                                                nnzA,
                                                                descrA,
                                                                csrValA,
                                                                csrRowPtrA,
                                                                csrColIndA,
                                                                percentage,
                                                                descrC,
                                                                csrValC,
                                                                csrRowPtrC,
                                                                csrColIndC,
                                                                info,
                                                                pBufferSizeInBytes);
    }
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 13000)
    template <>
    hipsparseStatus_t hipsparseXpruneCsr2csrNnzByPercentage(hipsparseHandle_t         handle,
                                                            int                       m,
                                                            int                       n,
                                                            int                       nnzA,
                                                            const hipsparseMatDescr_t descrA,
                                                            const float*              csrValA,
                                                            const int*                csrRowPtrA,
                                                            const int*                csrColIndA,
                                                            float                     percentage,
                                                            const hipsparseMatDescr_t descrC,
                                                            int*                      csrRowPtrC,
                                                            int*        nnzTotalDevHostPtr,
                                                            pruneInfo_t info,
                                                            void*       buffer)
    {
        return hipsparseSpruneCsr2csrNnzByPercentage(handle,
                                                     m,
                                                     n,
                                                     nnzA,
                                                     descrA,
                                                     csrValA,
                                                     csrRowPtrA,
                                                     csrColIndA,
                                                     percentage,
                                                     descrC,
                                                     csrRowPtrC,
                                                     nnzTotalDevHostPtr,
                                                     info,
                                                     buffer);
    }

    template <>
    hipsparseStatus_t hipsparseXpruneCsr2csrNnzByPercentage(hipsparseHandle_t         handle,
                                                            int                       m,
                                                            int                       n,
                                                            int                       nnzA,
                                                            const hipsparseMatDescr_t descrA,
                                                            const double*             csrValA,
                                                            const int*                csrRowPtrA,
                                                            const int*                csrColIndA,
                                                            double                    percentage,
                                                            const hipsparseMatDescr_t descrC,
                                                            int*                      csrRowPtrC,
                                                            int*        nnzTotalDevHostPtr,
                                                            pruneInfo_t info,
                                                            void*       buffer)
    {
        return hipsparseDpruneCsr2csrNnzByPercentage(handle,
                                                     m,
                                                     n,
                                                     nnzA,
                                                     descrA,
                                                     csrValA,
                                                     csrRowPtrA,
                                                     csrColIndA,
                                                     percentage,
                                                     descrC,
                                                     csrRowPtrC,
                                                     nnzTotalDevHostPtr,
                                                     info,
                                                     buffer);
    }
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 13000)
    template <>
    hipsparseStatus_t hipsparseXpruneCsr2csrByPercentage(hipsparseHandle_t         handle,
                                                         int                       m,
                                                         int                       n,
                                                         int                       nnzA,
                                                         const hipsparseMatDescr_t descrA,
                                                         const float*              csrValA,
                                                         const int*                csrRowPtrA,
                                                         const int*                csrColIndA,
                                                         float                     percentage,
                                                         const hipsparseMatDescr_t descrC,
                                                         float*                    csrValC,
                                                         const int*                csrRowPtrC,
                                                         int*                      csrColIndC,
                                                         pruneInfo_t               info,
                                                         void*                     buffer)
    {
        return hipsparseSpruneCsr2csrByPercentage(handle,
                                                  m,
                                                  n,
                                                  nnzA,
                                                  descrA,
                                                  csrValA,
                                                  csrRowPtrA,
                                                  csrColIndA,
                                                  percentage,
                                                  descrC,
                                                  csrValC,
                                                  csrRowPtrC,
                                                  csrColIndC,
                                                  info,
                                                  buffer);
    }

    template <>
    hipsparseStatus_t hipsparseXpruneCsr2csrByPercentage(hipsparseHandle_t         handle,
                                                         int                       m,
                                                         int                       n,
                                                         int                       nnzA,
                                                         const hipsparseMatDescr_t descrA,
                                                         const double*             csrValA,
                                                         const int*                csrRowPtrA,
                                                         const int*                csrColIndA,
                                                         double                    percentage,
                                                         const hipsparseMatDescr_t descrC,
                                                         double*                   csrValC,
                                                         const int*                csrRowPtrC,
                                                         int*                      csrColIndC,
                                                         pruneInfo_t               info,
                                                         void*                     buffer)
    {
        return hipsparseDpruneCsr2csrByPercentage(handle,
                                                  m,
                                                  n,
                                                  nnzA,
                                                  descrA,
                                                  csrValA,
                                                  csrRowPtrA,
                                                  csrColIndA,
                                                  percentage,
                                                  descrC,
                                                  csrValC,
                                                  csrRowPtrC,
                                                  csrColIndC,
                                                  info,
                                                  buffer);
    }
#endif

    template <>
    hipsparseStatus_t hipsparseXgebsr2gebsr_bufferSize(hipsparseHandle_t         handle,
                                                       hipsparseDirection_t      dirA,
                                                       int                       mb,
                                                       int                       nb,
                                                       int                       nnzb,
                                                       const hipsparseMatDescr_t descrA,
                                                       const float*              bsrValA,
                                                       const int*                bsrRowPtrA,
                                                       const int*                bsrColIndA,
                                                       int                       rowBlockDimA,
                                                       int                       colBlockDimA,
                                                       int                       rowBlockDimC,
                                                       int                       colBlockDimC,
                                                       int*                      pBufferSizeInBytes)
    {
        return hipsparseSgebsr2gebsr_bufferSize(handle,
                                                dirA,
                                                mb,
                                                nb,
                                                nnzb,
                                                descrA,
                                                bsrValA,
                                                bsrRowPtrA,
                                                bsrColIndA,
                                                rowBlockDimA,
                                                colBlockDimA,
                                                rowBlockDimC,
                                                colBlockDimC,
                                                pBufferSizeInBytes);
    }

    template <>
    hipsparseStatus_t hipsparseXgebsr2gebsr_bufferSize(hipsparseHandle_t         handle,
                                                       hipsparseDirection_t      dirA,
                                                       int                       mb,
                                                       int                       nb,
                                                       int                       nnzb,
                                                       const hipsparseMatDescr_t descrA,
                                                       const double*             bsrValA,
                                                       const int*                bsrRowPtrA,
                                                       const int*                bsrColIndA,
                                                       int                       rowBlockDimA,
                                                       int                       colBlockDimA,
                                                       int                       rowBlockDimC,
                                                       int                       colBlockDimC,
                                                       int*                      pBufferSizeInBytes)
    {
        return hipsparseDgebsr2gebsr_bufferSize(handle,
                                                dirA,
                                                mb,
                                                nb,
                                                nnzb,
                                                descrA,
                                                bsrValA,
                                                bsrRowPtrA,
                                                bsrColIndA,
                                                rowBlockDimA,
                                                colBlockDimA,
                                                rowBlockDimC,
                                                colBlockDimC,
                                                pBufferSizeInBytes);
    }

    template <>
    hipsparseStatus_t hipsparseXgebsr2gebsr_bufferSize(hipsparseHandle_t         handle,
                                                       hipsparseDirection_t      dirA,
                                                       int                       mb,
                                                       int                       nb,
                                                       int                       nnzb,
                                                       const hipsparseMatDescr_t descrA,
                                                       const hipComplex*         bsrValA,
                                                       const int*                bsrRowPtrA,
                                                       const int*                bsrColIndA,
                                                       int                       rowBlockDimA,
                                                       int                       colBlockDimA,
                                                       int                       rowBlockDimC,
                                                       int                       colBlockDimC,
                                                       int*                      pBufferSizeInBytes)
    {
        return hipsparseCgebsr2gebsr_bufferSize(handle,
                                                dirA,
                                                mb,
                                                nb,
                                                nnzb,
                                                descrA,
                                                bsrValA,
                                                bsrRowPtrA,
                                                bsrColIndA,
                                                rowBlockDimA,
                                                colBlockDimA,
                                                rowBlockDimC,
                                                colBlockDimC,
                                                pBufferSizeInBytes);
    }

    template <>
    hipsparseStatus_t hipsparseXgebsr2gebsr_bufferSize(hipsparseHandle_t         handle,
                                                       hipsparseDirection_t      dirA,
                                                       int                       mb,
                                                       int                       nb,
                                                       int                       nnzb,
                                                       const hipsparseMatDescr_t descrA,
                                                       const hipDoubleComplex*   bsrValA,
                                                       const int*                bsrRowPtrA,
                                                       const int*                bsrColIndA,
                                                       int                       rowBlockDimA,
                                                       int                       colBlockDimA,
                                                       int                       rowBlockDimC,
                                                       int                       colBlockDimC,
                                                       int*                      pBufferSizeInBytes)
    {
        return hipsparseZgebsr2gebsr_bufferSize(handle,
                                                dirA,
                                                mb,
                                                nb,
                                                nnzb,
                                                descrA,
                                                bsrValA,
                                                bsrRowPtrA,
                                                bsrColIndA,
                                                rowBlockDimA,
                                                colBlockDimA,
                                                rowBlockDimC,
                                                colBlockDimC,
                                                pBufferSizeInBytes);
    }

    template <>
    hipsparseStatus_t hipsparseXgebsr2gebsr(hipsparseHandle_t         handle,
                                            hipsparseDirection_t      dirA,
                                            int                       mb,
                                            int                       nb,
                                            int                       nnzb,
                                            const hipsparseMatDescr_t descrA,
                                            const float*              bsrValA,
                                            const int*                bsrRowPtrA,
                                            const int*                bsrColIndA,
                                            int                       rowBlockDimA,
                                            int                       colBlockDimA,
                                            const hipsparseMatDescr_t descrC,
                                            float*                    bsrValC,
                                            int*                      bsrRowPtrC,
                                            int*                      bsrColIndC,
                                            int                       rowBlockDimC,
                                            int                       colBlockDimC,
                                            void*                     buffer)
    {
        return hipsparseSgebsr2gebsr(handle,
                                     dirA,
                                     mb,
                                     nb,
                                     nnzb,
                                     descrA,
                                     bsrValA,
                                     bsrRowPtrA,
                                     bsrColIndA,
                                     rowBlockDimA,
                                     colBlockDimA,
                                     descrC,
                                     bsrValC,
                                     bsrRowPtrC,
                                     bsrColIndC,
                                     rowBlockDimC,
                                     colBlockDimC,
                                     buffer);
    }

    template <>
    hipsparseStatus_t hipsparseXgebsr2gebsr(hipsparseHandle_t         handle,
                                            hipsparseDirection_t      dirA,
                                            int                       mb,
                                            int                       nb,
                                            int                       nnzb,
                                            const hipsparseMatDescr_t descrA,
                                            const double*             bsrValA,
                                            const int*                bsrRowPtrA,
                                            const int*                bsrColIndA,
                                            int                       rowBlockDimA,
                                            int                       colBlockDimA,
                                            const hipsparseMatDescr_t descrC,
                                            double*                   bsrValC,
                                            int*                      bsrRowPtrC,
                                            int*                      bsrColIndC,
                                            int                       rowBlockDimC,
                                            int                       colBlockDimC,
                                            void*                     buffer)
    {
        return hipsparseDgebsr2gebsr(handle,
                                     dirA,
                                     mb,
                                     nb,
                                     nnzb,
                                     descrA,
                                     bsrValA,
                                     bsrRowPtrA,
                                     bsrColIndA,
                                     rowBlockDimA,
                                     colBlockDimA,
                                     descrC,
                                     bsrValC,
                                     bsrRowPtrC,
                                     bsrColIndC,
                                     rowBlockDimC,
                                     colBlockDimC,
                                     buffer);
    }

    template <>
    hipsparseStatus_t hipsparseXgebsr2gebsr(hipsparseHandle_t         handle,
                                            hipsparseDirection_t      dirA,
                                            int                       mb,
                                            int                       nb,
                                            int                       nnzb,
                                            const hipsparseMatDescr_t descrA,
                                            const hipComplex*         bsrValA,
                                            const int*                bsrRowPtrA,
                                            const int*                bsrColIndA,
                                            int                       rowBlockDimA,
                                            int                       colBlockDimA,
                                            const hipsparseMatDescr_t descrC,
                                            hipComplex*               bsrValC,
                                            int*                      bsrRowPtrC,
                                            int*                      bsrColIndC,
                                            int                       rowBlockDimC,
                                            int                       colBlockDimC,
                                            void*                     buffer)
    {
        return hipsparseCgebsr2gebsr(handle,
                                     dirA,
                                     mb,
                                     nb,
                                     nnzb,
                                     descrA,
                                     bsrValA,
                                     bsrRowPtrA,
                                     bsrColIndA,
                                     rowBlockDimA,
                                     colBlockDimA,
                                     descrC,
                                     bsrValC,
                                     bsrRowPtrC,
                                     bsrColIndC,
                                     rowBlockDimC,
                                     colBlockDimC,
                                     buffer);
    }

    template <>
    hipsparseStatus_t hipsparseXgebsr2gebsr(hipsparseHandle_t         handle,
                                            hipsparseDirection_t      dirA,
                                            int                       mb,
                                            int                       nb,
                                            int                       nnzb,
                                            const hipsparseMatDescr_t descrA,
                                            const hipDoubleComplex*   bsrValA,
                                            const int*                bsrRowPtrA,
                                            const int*                bsrColIndA,
                                            int                       rowBlockDimA,
                                            int                       colBlockDimA,
                                            const hipsparseMatDescr_t descrC,
                                            hipDoubleComplex*         bsrValC,
                                            int*                      bsrRowPtrC,
                                            int*                      bsrColIndC,
                                            int                       rowBlockDimC,
                                            int                       colBlockDimC,
                                            void*                     buffer)
    {
        return hipsparseZgebsr2gebsr(handle,
                                     dirA,
                                     mb,
                                     nb,
                                     nnzb,
                                     descrA,
                                     bsrValA,
                                     bsrRowPtrA,
                                     bsrColIndA,
                                     rowBlockDimA,
                                     colBlockDimA,
                                     descrC,
                                     bsrValC,
                                     bsrRowPtrC,
                                     bsrColIndC,
                                     rowBlockDimC,
                                     colBlockDimC,
                                     buffer);
    }

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 13000)
    template <>
    hipsparseStatus_t hipsparseXcsru2csr_bufferSizeExt(hipsparseHandle_t handle,
                                                       int               m,
                                                       int               n,
                                                       int               nnz,
                                                       float*            csrVal,
                                                       const int*        csrRowPtr,
                                                       int*              csrColInd,
                                                       csru2csrInfo_t    info,
                                                       size_t*           pBufferSizeInBytes)
    {
        return hipsparseScsru2csr_bufferSizeExt(
            handle, m, n, nnz, csrVal, csrRowPtr, csrColInd, info, pBufferSizeInBytes);
    }

    template <>
    hipsparseStatus_t hipsparseXcsru2csr_bufferSizeExt(hipsparseHandle_t handle,
                                                       int               m,
                                                       int               n,
                                                       int               nnz,
                                                       double*           csrVal,
                                                       const int*        csrRowPtr,
                                                       int*              csrColInd,
                                                       csru2csrInfo_t    info,
                                                       size_t*           pBufferSizeInBytes)
    {
        return hipsparseDcsru2csr_bufferSizeExt(
            handle, m, n, nnz, csrVal, csrRowPtr, csrColInd, info, pBufferSizeInBytes);
    }

    template <>
    hipsparseStatus_t hipsparseXcsru2csr_bufferSizeExt(hipsparseHandle_t handle,
                                                       int               m,
                                                       int               n,
                                                       int               nnz,
                                                       hipComplex*       csrVal,
                                                       const int*        csrRowPtr,
                                                       int*              csrColInd,
                                                       csru2csrInfo_t    info,
                                                       size_t*           pBufferSizeInBytes)
    {
        return hipsparseCcsru2csr_bufferSizeExt(
            handle, m, n, nnz, csrVal, csrRowPtr, csrColInd, info, pBufferSizeInBytes);
    }

    template <>
    hipsparseStatus_t hipsparseXcsru2csr_bufferSizeExt(hipsparseHandle_t handle,
                                                       int               m,
                                                       int               n,
                                                       int               nnz,
                                                       hipDoubleComplex* csrVal,
                                                       const int*        csrRowPtr,
                                                       int*              csrColInd,
                                                       csru2csrInfo_t    info,
                                                       size_t*           pBufferSizeInBytes)
    {
        return hipsparseZcsru2csr_bufferSizeExt(
            handle, m, n, nnz, csrVal, csrRowPtr, csrColInd, info, pBufferSizeInBytes);
    }
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 13000)
    template <>
    hipsparseStatus_t hipsparseXcsru2csr(hipsparseHandle_t         handle,
                                         int                       m,
                                         int                       n,
                                         int                       nnz,
                                         const hipsparseMatDescr_t descrA,
                                         float*                    csrVal,
                                         const int*                csrRowPtr,
                                         int*                      csrColInd,
                                         csru2csrInfo_t            info,
                                         void*                     pBuffer)
    {
        return hipsparseScsru2csr(
            handle, m, n, nnz, descrA, csrVal, csrRowPtr, csrColInd, info, pBuffer);
    }

    template <>
    hipsparseStatus_t hipsparseXcsru2csr(hipsparseHandle_t         handle,
                                         int                       m,
                                         int                       n,
                                         int                       nnz,
                                         const hipsparseMatDescr_t descrA,
                                         double*                   csrVal,
                                         const int*                csrRowPtr,
                                         int*                      csrColInd,
                                         csru2csrInfo_t            info,
                                         void*                     pBuffer)
    {
        return hipsparseDcsru2csr(
            handle, m, n, nnz, descrA, csrVal, csrRowPtr, csrColInd, info, pBuffer);
    }

    template <>
    hipsparseStatus_t hipsparseXcsru2csr(hipsparseHandle_t         handle,
                                         int                       m,
                                         int                       n,
                                         int                       nnz,
                                         const hipsparseMatDescr_t descrA,
                                         hipComplex*               csrVal,
                                         const int*                csrRowPtr,
                                         int*                      csrColInd,
                                         csru2csrInfo_t            info,
                                         void*                     pBuffer)
    {
        return hipsparseCcsru2csr(
            handle, m, n, nnz, descrA, csrVal, csrRowPtr, csrColInd, info, pBuffer);
    }

    template <>
    hipsparseStatus_t hipsparseXcsru2csr(hipsparseHandle_t         handle,
                                         int                       m,
                                         int                       n,
                                         int                       nnz,
                                         const hipsparseMatDescr_t descrA,
                                         hipDoubleComplex*         csrVal,
                                         const int*                csrRowPtr,
                                         int*                      csrColInd,
                                         csru2csrInfo_t            info,
                                         void*                     pBuffer)
    {
        return hipsparseZcsru2csr(
            handle, m, n, nnz, descrA, csrVal, csrRowPtr, csrColInd, info, pBuffer);
    }
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 13000)
    template <>
    hipsparseStatus_t hipsparseXcsr2csru(hipsparseHandle_t         handle,
                                         int                       m,
                                         int                       n,
                                         int                       nnz,
                                         const hipsparseMatDescr_t descrA,
                                         float*                    csrVal,
                                         const int*                csrRowPtr,
                                         int*                      csrColInd,
                                         csru2csrInfo_t            info,
                                         void*                     pBuffer)
    {
        return hipsparseScsr2csru(
            handle, m, n, nnz, descrA, csrVal, csrRowPtr, csrColInd, info, pBuffer);
    }

    template <>
    hipsparseStatus_t hipsparseXcsr2csru(hipsparseHandle_t         handle,
                                         int                       m,
                                         int                       n,
                                         int                       nnz,
                                         const hipsparseMatDescr_t descrA,
                                         double*                   csrVal,
                                         const int*                csrRowPtr,
                                         int*                      csrColInd,
                                         csru2csrInfo_t            info,
                                         void*                     pBuffer)
    {
        return hipsparseDcsr2csru(
            handle, m, n, nnz, descrA, csrVal, csrRowPtr, csrColInd, info, pBuffer);
    }

    template <>
    hipsparseStatus_t hipsparseXcsr2csru(hipsparseHandle_t         handle,
                                         int                       m,
                                         int                       n,
                                         int                       nnz,
                                         const hipsparseMatDescr_t descrA,
                                         hipComplex*               csrVal,
                                         const int*                csrRowPtr,
                                         int*                      csrColInd,
                                         csru2csrInfo_t            info,
                                         void*                     pBuffer)
    {
        return hipsparseCcsr2csru(
            handle, m, n, nnz, descrA, csrVal, csrRowPtr, csrColInd, info, pBuffer);
    }

    template <>
    hipsparseStatus_t hipsparseXcsr2csru(hipsparseHandle_t         handle,
                                         int                       m,
                                         int                       n,
                                         int                       nnz,
                                         const hipsparseMatDescr_t descrA,
                                         hipDoubleComplex*         csrVal,
                                         const int*                csrRowPtr,
                                         int*                      csrColInd,
                                         csru2csrInfo_t            info,
                                         void*                     pBuffer)
    {
        return hipsparseZcsr2csru(
            handle, m, n, nnz, descrA, csrVal, csrRowPtr, csrColInd, info, pBuffer);
    }
#endif

    template <>
    hipsparseStatus_t hipsparseXgpsvInterleavedBatch_bufferSizeExt(hipsparseHandle_t handle,
                                                                   int               algo,
                                                                   int               m,
                                                                   const float*      ds,
                                                                   const float*      dl,
                                                                   const float*      d,
                                                                   const float*      du,
                                                                   const float*      dw,
                                                                   const float*      x,
                                                                   int               batchCount,
                                                                   size_t* pBufferSizeInBytes)
    {
        return hipsparseSgpsvInterleavedBatch_bufferSizeExt(
            handle, algo, m, ds, dl, d, du, dw, x, batchCount, pBufferSizeInBytes);
    }

    template <>
    hipsparseStatus_t hipsparseXgpsvInterleavedBatch_bufferSizeExt(hipsparseHandle_t handle,
                                                                   int               algo,
                                                                   int               m,
                                                                   const double*     ds,
                                                                   const double*     dl,
                                                                   const double*     d,
                                                                   const double*     du,
                                                                   const double*     dw,
                                                                   const double*     x,
                                                                   int               batchCount,
                                                                   size_t* pBufferSizeInBytes)
    {
        return hipsparseDgpsvInterleavedBatch_bufferSizeExt(
            handle, algo, m, ds, dl, d, du, dw, x, batchCount, pBufferSizeInBytes);
    }

    template <>
    hipsparseStatus_t hipsparseXgpsvInterleavedBatch_bufferSizeExt(hipsparseHandle_t handle,
                                                                   int               algo,
                                                                   int               m,
                                                                   const hipComplex* ds,
                                                                   const hipComplex* dl,
                                                                   const hipComplex* d,
                                                                   const hipComplex* du,
                                                                   const hipComplex* dw,
                                                                   const hipComplex* x,
                                                                   int               batchCount,
                                                                   size_t* pBufferSizeInBytes)
    {
        return hipsparseCgpsvInterleavedBatch_bufferSizeExt(
            handle, algo, m, ds, dl, d, du, dw, x, batchCount, pBufferSizeInBytes);
    }

    template <>
    hipsparseStatus_t hipsparseXgpsvInterleavedBatch_bufferSizeExt(hipsparseHandle_t       handle,
                                                                   int                     algo,
                                                                   int                     m,
                                                                   const hipDoubleComplex* ds,
                                                                   const hipDoubleComplex* dl,
                                                                   const hipDoubleComplex* d,
                                                                   const hipDoubleComplex* du,
                                                                   const hipDoubleComplex* dw,
                                                                   const hipDoubleComplex* x,
                                                                   int     batchCount,
                                                                   size_t* pBufferSizeInBytes)
    {
        return hipsparseZgpsvInterleavedBatch_bufferSizeExt(
            handle, algo, m, ds, dl, d, du, dw, x, batchCount, pBufferSizeInBytes);
    }

    template <>
    hipsparseStatus_t hipsparseXgpsvInterleavedBatch(hipsparseHandle_t handle,
                                                     int               algo,
                                                     int               m,
                                                     float*            ds,
                                                     float*            dl,
                                                     float*            d,
                                                     float*            du,
                                                     float*            dw,
                                                     float*            x,
                                                     int               batchCount,
                                                     void*             pBuffer)
    {
        return hipsparseSgpsvInterleavedBatch(
            handle, algo, m, ds, dl, d, du, dw, x, batchCount, pBuffer);
    }

    template <>
    hipsparseStatus_t hipsparseXgpsvInterleavedBatch(hipsparseHandle_t handle,
                                                     int               algo,
                                                     int               m,
                                                     double*           ds,
                                                     double*           dl,
                                                     double*           d,
                                                     double*           du,
                                                     double*           dw,
                                                     double*           x,
                                                     int               batchCount,
                                                     void*             pBuffer)
    {
        return hipsparseDgpsvInterleavedBatch(
            handle, algo, m, ds, dl, d, du, dw, x, batchCount, pBuffer);
    }

    template <>
    hipsparseStatus_t hipsparseXgpsvInterleavedBatch(hipsparseHandle_t handle,
                                                     int               algo,
                                                     int               m,
                                                     hipComplex*       ds,
                                                     hipComplex*       dl,
                                                     hipComplex*       d,
                                                     hipComplex*       du,
                                                     hipComplex*       dw,
                                                     hipComplex*       x,
                                                     int               batchCount,
                                                     void*             pBuffer)
    {
        return hipsparseCgpsvInterleavedBatch(
            handle, algo, m, ds, dl, d, du, dw, x, batchCount, pBuffer);
    }

    template <>
    hipsparseStatus_t hipsparseXgpsvInterleavedBatch(hipsparseHandle_t handle,
                                                     int               algo,
                                                     int               m,
                                                     hipDoubleComplex* ds,
                                                     hipDoubleComplex* dl,
                                                     hipDoubleComplex* d,
                                                     hipDoubleComplex* du,
                                                     hipDoubleComplex* dw,
                                                     hipDoubleComplex* x,
                                                     int               batchCount,
                                                     void*             pBuffer)
    {
        return hipsparseZgpsvInterleavedBatch(
            handle, algo, m, ds, dl, d, du, dw, x, batchCount, pBuffer);
    }

    template <>
    hipsparseStatus_t hipsparseXgtsv2StridedBatch_bufferSizeExt(hipsparseHandle_t handle,
                                                                int               m,
                                                                const float*      dl,
                                                                const float*      d,
                                                                const float*      du,
                                                                const float*      x,
                                                                int               batchCount,
                                                                int               batchStride,
                                                                size_t* pBufferSizeInBytes)
    {
        return hipsparseSgtsv2StridedBatch_bufferSizeExt(
            handle, m, dl, d, du, x, batchCount, batchStride, pBufferSizeInBytes);
    }

    template <>
    hipsparseStatus_t hipsparseXgtsv2StridedBatch_bufferSizeExt(hipsparseHandle_t handle,
                                                                int               m,
                                                                const double*     dl,
                                                                const double*     d,
                                                                const double*     du,
                                                                const double*     x,
                                                                int               batchCount,
                                                                int               batchStride,
                                                                size_t* pBufferSizeInBytes)
    {
        return hipsparseDgtsv2StridedBatch_bufferSizeExt(
            handle, m, dl, d, du, x, batchCount, batchStride, pBufferSizeInBytes);
    }

    template <>
    hipsparseStatus_t hipsparseXgtsv2StridedBatch_bufferSizeExt(hipsparseHandle_t handle,
                                                                int               m,
                                                                const hipComplex* dl,
                                                                const hipComplex* d,
                                                                const hipComplex* du,
                                                                const hipComplex* x,
                                                                int               batchCount,
                                                                int               batchStride,
                                                                size_t* pBufferSizeInBytes)
    {
        return hipsparseCgtsv2StridedBatch_bufferSizeExt(
            handle, m, dl, d, du, x, batchCount, batchStride, pBufferSizeInBytes);
    }

    template <>
    hipsparseStatus_t hipsparseXgtsv2StridedBatch_bufferSizeExt(hipsparseHandle_t       handle,
                                                                int                     m,
                                                                const hipDoubleComplex* dl,
                                                                const hipDoubleComplex* d,
                                                                const hipDoubleComplex* du,
                                                                const hipDoubleComplex* x,
                                                                int                     batchCount,
                                                                int                     batchStride,
                                                                size_t* pBufferSizeInBytes)
    {
        return hipsparseZgtsv2StridedBatch_bufferSizeExt(
            handle, m, dl, d, du, x, batchCount, batchStride, pBufferSizeInBytes);
    }

    template <>
    hipsparseStatus_t hipsparseXgtsv2StridedBatch(hipsparseHandle_t handle,
                                                  int               m,
                                                  const float*      dl,
                                                  const float*      d,
                                                  const float*      du,
                                                  float*            x,
                                                  int               batchCount,
                                                  int               batchStride,
                                                  void*             pBuffer)
    {
        return hipsparseSgtsv2StridedBatch(
            handle, m, dl, d, du, x, batchCount, batchStride, pBuffer);
    }

    template <>
    hipsparseStatus_t hipsparseXgtsv2StridedBatch(hipsparseHandle_t handle,
                                                  int               m,
                                                  const double*     dl,
                                                  const double*     d,
                                                  const double*     du,
                                                  double*           x,
                                                  int               batchCount,
                                                  int               batchStride,
                                                  void*             pBuffer)
    {
        return hipsparseDgtsv2StridedBatch(
            handle, m, dl, d, du, x, batchCount, batchStride, pBuffer);
    }

    template <>
    hipsparseStatus_t hipsparseXgtsv2StridedBatch(hipsparseHandle_t handle,
                                                  int               m,
                                                  const hipComplex* dl,
                                                  const hipComplex* d,
                                                  const hipComplex* du,
                                                  hipComplex*       x,
                                                  int               batchCount,
                                                  int               batchStride,
                                                  void*             pBuffer)
    {
        return hipsparseCgtsv2StridedBatch(
            handle, m, dl, d, du, x, batchCount, batchStride, pBuffer);
    }

    template <>
    hipsparseStatus_t hipsparseXgtsv2StridedBatch(hipsparseHandle_t       handle,
                                                  int                     m,
                                                  const hipDoubleComplex* dl,
                                                  const hipDoubleComplex* d,
                                                  const hipDoubleComplex* du,
                                                  hipDoubleComplex*       x,
                                                  int                     batchCount,
                                                  int                     batchStride,
                                                  void*                   pBuffer)
    {
        return hipsparseZgtsv2StridedBatch(
            handle, m, dl, d, du, x, batchCount, batchStride, pBuffer);
    }

    template <>
    hipsparseStatus_t hipsparseXgtsv2_bufferSizeExt(hipsparseHandle_t handle,
                                                    int               m,
                                                    int               n,
                                                    const float*      dl,
                                                    const float*      d,
                                                    const float*      du,
                                                    const float*      B,
                                                    int               ldb,
                                                    size_t*           pBufferSizeInBytes)
    {
        return hipsparseSgtsv2_bufferSizeExt(handle, m, n, dl, d, du, B, ldb, pBufferSizeInBytes);
    }

    template <>
    hipsparseStatus_t hipsparseXgtsv2_bufferSizeExt(hipsparseHandle_t handle,
                                                    int               m,
                                                    int               n,
                                                    const double*     dl,
                                                    const double*     d,
                                                    const double*     du,
                                                    const double*     B,
                                                    int               ldb,
                                                    size_t*           pBufferSizeInBytes)
    {
        return hipsparseDgtsv2_bufferSizeExt(handle, m, n, dl, d, du, B, ldb, pBufferSizeInBytes);
    }

    template <>
    hipsparseStatus_t hipsparseXgtsv2_bufferSizeExt(hipsparseHandle_t handle,
                                                    int               m,
                                                    int               n,
                                                    const hipComplex* dl,
                                                    const hipComplex* d,
                                                    const hipComplex* du,
                                                    const hipComplex* B,
                                                    int               ldb,
                                                    size_t*           pBufferSizeInBytes)
    {
        return hipsparseCgtsv2_bufferSizeExt(handle, m, n, dl, d, du, B, ldb, pBufferSizeInBytes);
    }

    template <>
    hipsparseStatus_t hipsparseXgtsv2_bufferSizeExt(hipsparseHandle_t       handle,
                                                    int                     m,
                                                    int                     n,
                                                    const hipDoubleComplex* dl,
                                                    const hipDoubleComplex* d,
                                                    const hipDoubleComplex* du,
                                                    const hipDoubleComplex* B,
                                                    int                     ldb,
                                                    size_t*                 pBufferSizeInBytes)
    {
        return hipsparseZgtsv2_bufferSizeExt(handle, m, n, dl, d, du, B, ldb, pBufferSizeInBytes);
    }

    template <>
    hipsparseStatus_t hipsparseXgtsv2(hipsparseHandle_t handle,
                                      int               m,
                                      int               n,
                                      const float*      dl,
                                      const float*      d,
                                      const float*      du,
                                      float*            B,
                                      int               ldb,
                                      void*             pBuffer)
    {
        return hipsparseSgtsv2(handle, m, n, dl, d, du, B, ldb, pBuffer);
    }

    template <>
    hipsparseStatus_t hipsparseXgtsv2(hipsparseHandle_t handle,
                                      int               m,
                                      int               n,
                                      const double*     dl,
                                      const double*     d,
                                      const double*     du,
                                      double*           B,
                                      int               ldb,
                                      void*             pBuffer)
    {
        return hipsparseDgtsv2(handle, m, n, dl, d, du, B, ldb, pBuffer);
    }

    template <>
    hipsparseStatus_t hipsparseXgtsv2(hipsparseHandle_t handle,
                                      int               m,
                                      int               n,
                                      const hipComplex* dl,
                                      const hipComplex* d,
                                      const hipComplex* du,
                                      hipComplex*       B,
                                      int               ldb,
                                      void*             pBuffer)
    {
        return hipsparseCgtsv2(handle, m, n, dl, d, du, B, ldb, pBuffer);
    }

    template <>
    hipsparseStatus_t hipsparseXgtsv2(hipsparseHandle_t       handle,
                                      int                     m,
                                      int                     n,
                                      const hipDoubleComplex* dl,
                                      const hipDoubleComplex* d,
                                      const hipDoubleComplex* du,
                                      hipDoubleComplex*       B,
                                      int                     ldb,
                                      void*                   pBuffer)
    {
        return hipsparseZgtsv2(handle, m, n, dl, d, du, B, ldb, pBuffer);
    }

    template <>
    hipsparseStatus_t hipsparseXgtsv2_nopivot_bufferSizeExt(hipsparseHandle_t handle,
                                                            int               m,
                                                            int               n,
                                                            const float*      dl,
                                                            const float*      d,
                                                            const float*      du,
                                                            const float*      B,
                                                            int               ldb,
                                                            size_t*           pBufferSizeInBytes)
    {
        return hipsparseSgtsv2_nopivot_bufferSizeExt(
            handle, m, n, dl, d, du, B, ldb, pBufferSizeInBytes);
    }

    template <>
    hipsparseStatus_t hipsparseXgtsv2_nopivot_bufferSizeExt(hipsparseHandle_t handle,
                                                            int               m,
                                                            int               n,
                                                            const double*     dl,
                                                            const double*     d,
                                                            const double*     du,
                                                            const double*     B,
                                                            int               ldb,
                                                            size_t*           pBufferSizeInBytes)
    {
        return hipsparseDgtsv2_nopivot_bufferSizeExt(
            handle, m, n, dl, d, du, B, ldb, pBufferSizeInBytes);
    }

    template <>
    hipsparseStatus_t hipsparseXgtsv2_nopivot_bufferSizeExt(hipsparseHandle_t handle,
                                                            int               m,
                                                            int               n,
                                                            const hipComplex* dl,
                                                            const hipComplex* d,
                                                            const hipComplex* du,
                                                            const hipComplex* B,
                                                            int               ldb,
                                                            size_t*           pBufferSizeInBytes)
    {
        return hipsparseCgtsv2_nopivot_bufferSizeExt(
            handle, m, n, dl, d, du, B, ldb, pBufferSizeInBytes);
    }

    template <>
    hipsparseStatus_t hipsparseXgtsv2_nopivot_bufferSizeExt(hipsparseHandle_t       handle,
                                                            int                     m,
                                                            int                     n,
                                                            const hipDoubleComplex* dl,
                                                            const hipDoubleComplex* d,
                                                            const hipDoubleComplex* du,
                                                            const hipDoubleComplex* B,
                                                            int                     ldb,
                                                            size_t* pBufferSizeInBytes)
    {
        return hipsparseZgtsv2_nopivot_bufferSizeExt(
            handle, m, n, dl, d, du, B, ldb, pBufferSizeInBytes);
    }

    template <>
    hipsparseStatus_t hipsparseXgtsv2_nopivot(hipsparseHandle_t handle,
                                              int               m,
                                              int               n,
                                              const float*      dl,
                                              const float*      d,
                                              const float*      du,
                                              float*            B,
                                              int               ldb,
                                              void*             pBuffer)
    {
        return hipsparseSgtsv2_nopivot(handle, m, n, dl, d, du, B, ldb, pBuffer);
    }

    template <>
    hipsparseStatus_t hipsparseXgtsv2_nopivot(hipsparseHandle_t handle,
                                              int               m,
                                              int               n,
                                              const double*     dl,
                                              const double*     d,
                                              const double*     du,
                                              double*           B,
                                              int               ldb,
                                              void*             pBuffer)
    {
        return hipsparseDgtsv2_nopivot(handle, m, n, dl, d, du, B, ldb, pBuffer);
    }

    template <>
    hipsparseStatus_t hipsparseXgtsv2_nopivot(hipsparseHandle_t handle,
                                              int               m,
                                              int               n,
                                              const hipComplex* dl,
                                              const hipComplex* d,
                                              const hipComplex* du,
                                              hipComplex*       B,
                                              int               ldb,
                                              void*             pBuffer)
    {
        return hipsparseCgtsv2_nopivot(handle, m, n, dl, d, du, B, ldb, pBuffer);
    }

    template <>
    hipsparseStatus_t hipsparseXgtsv2_nopivot(hipsparseHandle_t       handle,
                                              int                     m,
                                              int                     n,
                                              const hipDoubleComplex* dl,
                                              const hipDoubleComplex* d,
                                              const hipDoubleComplex* du,
                                              hipDoubleComplex*       B,
                                              int                     ldb,
                                              void*                   pBuffer)
    {
        return hipsparseZgtsv2_nopivot(handle, m, n, dl, d, du, B, ldb, pBuffer);
    }

    template <>
    hipsparseStatus_t hipsparseXgtsvInterleavedBatch_bufferSizeExt(hipsparseHandle_t handle,
                                                                   int               algo,
                                                                   int               m,
                                                                   const float*      dl,
                                                                   const float*      d,
                                                                   const float*      du,
                                                                   const float*      x,
                                                                   int               batchCount,
                                                                   size_t* pBufferSizeInBytes)
    {
        return hipsparseSgtsvInterleavedBatch_bufferSizeExt(
            handle, algo, m, dl, d, du, x, batchCount, pBufferSizeInBytes);
    }

    template <>
    hipsparseStatus_t hipsparseXgtsvInterleavedBatch_bufferSizeExt(hipsparseHandle_t handle,
                                                                   int               algo,
                                                                   int               m,
                                                                   const double*     dl,
                                                                   const double*     d,
                                                                   const double*     du,
                                                                   const double*     x,
                                                                   int               batchCount,
                                                                   size_t* pBufferSizeInBytes)
    {
        return hipsparseDgtsvInterleavedBatch_bufferSizeExt(
            handle, algo, m, dl, d, du, x, batchCount, pBufferSizeInBytes);
    }

    template <>
    hipsparseStatus_t hipsparseXgtsvInterleavedBatch_bufferSizeExt(hipsparseHandle_t handle,
                                                                   int               algo,
                                                                   int               m,
                                                                   const hipComplex* dl,
                                                                   const hipComplex* d,
                                                                   const hipComplex* du,
                                                                   const hipComplex* x,
                                                                   int               batchCount,
                                                                   size_t* pBufferSizeInBytes)
    {
        return hipsparseCgtsvInterleavedBatch_bufferSizeExt(
            handle, algo, m, dl, d, du, x, batchCount, pBufferSizeInBytes);
    }

    template <>
    hipsparseStatus_t hipsparseXgtsvInterleavedBatch_bufferSizeExt(hipsparseHandle_t       handle,
                                                                   int                     algo,
                                                                   int                     m,
                                                                   const hipDoubleComplex* dl,
                                                                   const hipDoubleComplex* d,
                                                                   const hipDoubleComplex* du,
                                                                   const hipDoubleComplex* x,
                                                                   int     batchCount,
                                                                   size_t* pBufferSizeInBytes)
    {
        return hipsparseZgtsvInterleavedBatch_bufferSizeExt(
            handle, algo, m, dl, d, du, x, batchCount, pBufferSizeInBytes);
    }

    template <>
    hipsparseStatus_t hipsparseXgtsvInterleavedBatch(hipsparseHandle_t handle,
                                                     int               algo,
                                                     int               m,
                                                     float*            dl,
                                                     float*            d,
                                                     float*            du,
                                                     float*            x,
                                                     int               batchCount,
                                                     void*             pBuffer)
    {
        return hipsparseSgtsvInterleavedBatch(handle, algo, m, dl, d, du, x, batchCount, pBuffer);
    }

    template <>
    hipsparseStatus_t hipsparseXgtsvInterleavedBatch(hipsparseHandle_t handle,
                                                     int               algo,
                                                     int               m,
                                                     double*           dl,
                                                     double*           d,
                                                     double*           du,
                                                     double*           x,
                                                     int               batchCount,
                                                     void*             pBuffer)
    {
        return hipsparseDgtsvInterleavedBatch(handle, algo, m, dl, d, du, x, batchCount, pBuffer);
    }

    template <>
    hipsparseStatus_t hipsparseXgtsvInterleavedBatch(hipsparseHandle_t handle,
                                                     int               algo,
                                                     int               m,
                                                     hipComplex*       dl,
                                                     hipComplex*       d,
                                                     hipComplex*       du,
                                                     hipComplex*       x,
                                                     int               batchCount,
                                                     void*             pBuffer)
    {
        return hipsparseCgtsvInterleavedBatch(handle, algo, m, dl, d, du, x, batchCount, pBuffer);
    }

    template <>
    hipsparseStatus_t hipsparseXgtsvInterleavedBatch(hipsparseHandle_t handle,
                                                     int               algo,
                                                     int               m,
                                                     hipDoubleComplex* dl,
                                                     hipDoubleComplex* d,
                                                     hipDoubleComplex* du,
                                                     hipDoubleComplex* x,
                                                     int               batchCount,
                                                     void*             pBuffer)
    {
        return hipsparseZgtsvInterleavedBatch(handle, algo, m, dl, d, du, x, batchCount, pBuffer);
    }

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 13000)
    template <>
    hipsparseStatus_t hipsparseXcsrcolor(hipsparseHandle_t         handle,
                                         int                       m,
                                         int                       nnz,
                                         const hipsparseMatDescr_t descrA,
                                         const float*              csrValA,
                                         const int*                csrRowPtrA,
                                         const int*                csrColIndA,
                                         const float*              fractionToColor,
                                         int*                      ncolors,
                                         int*                      coloring,
                                         int*                      reordering,
                                         hipsparseColorInfo_t      info)
    {
        return hipsparseScsrcolor(handle,
                                  m,
                                  nnz,
                                  descrA,
                                  csrValA,
                                  csrRowPtrA,
                                  csrColIndA,
                                  fractionToColor,
                                  ncolors,
                                  coloring,
                                  reordering,
                                  info);
    }

    template <>
    hipsparseStatus_t hipsparseXcsrcolor(hipsparseHandle_t         handle,
                                         int                       m,
                                         int                       nnz,
                                         const hipsparseMatDescr_t descrA,
                                         const double*             csrValA,
                                         const int*                csrRowPtrA,
                                         const int*                csrColIndA,
                                         const double*             fractionToColor,
                                         int*                      ncolors,
                                         int*                      coloring,
                                         int*                      reordering,
                                         hipsparseColorInfo_t      info)
    {
        return hipsparseDcsrcolor(handle,
                                  m,
                                  nnz,
                                  descrA,
                                  csrValA,
                                  csrRowPtrA,
                                  csrColIndA,
                                  fractionToColor,
                                  ncolors,
                                  coloring,
                                  reordering,
                                  info);
    }

    template <>
    hipsparseStatus_t hipsparseXcsrcolor(hipsparseHandle_t         handle,
                                         int                       m,
                                         int                       nnz,
                                         const hipsparseMatDescr_t descrA,
                                         const hipComplex*         csrValA,
                                         const int*                csrRowPtrA,
                                         const int*                csrColIndA,
                                         const float*              fractionToColor,
                                         int*                      ncolors,
                                         int*                      coloring,
                                         int*                      reordering,
                                         hipsparseColorInfo_t      info)
    {
        return hipsparseCcsrcolor(handle,
                                  m,
                                  nnz,
                                  descrA,
                                  csrValA,
                                  csrRowPtrA,
                                  csrColIndA,
                                  fractionToColor,
                                  ncolors,
                                  coloring,
                                  reordering,
                                  info);
    }

    template <>
    hipsparseStatus_t hipsparseXcsrcolor(hipsparseHandle_t         handle,
                                         int                       m,
                                         int                       nnz,
                                         const hipsparseMatDescr_t descrA,
                                         const hipDoubleComplex*   csrValA,
                                         const int*                csrRowPtrA,
                                         const int*                csrColIndA,
                                         const double*             fractionToColor,
                                         int*                      ncolors,
                                         int*                      coloring,
                                         int*                      reordering,
                                         hipsparseColorInfo_t      info)
    {
        return hipsparseZcsrcolor(handle,
                                  m,
                                  nnz,
                                  descrA,
                                  csrValA,
                                  csrRowPtrA,
                                  csrColIndA,
                                  fractionToColor,
                                  ncolors,
                                  coloring,
                                  reordering,
                                  info);
    }
#endif

} // namespace hipsparse
