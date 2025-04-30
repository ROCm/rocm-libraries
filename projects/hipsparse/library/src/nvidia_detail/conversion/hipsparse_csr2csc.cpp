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

#if CUDART_VERSION < 11000
hipsparseStatus_t hipsparseScsr2csc(hipsparseHandle_t    handle,
                                    int                  m,
                                    int                  n,
                                    int                  nnz,
                                    const float*         csrSortedVal,
                                    const int*           csrSortedRowPtr,
                                    const int*           csrSortedColInd,
                                    float*               cscSortedVal,
                                    int*                 cscSortedRowInd,
                                    int*                 cscSortedColPtr,
                                    hipsparseAction_t    copyValues,
                                    hipsparseIndexBase_t idxBase)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseScsr2csc((cusparseHandle_t)handle,
                         m,
                         n,
                         nnz,
                         csrSortedVal,
                         csrSortedRowPtr,
                         csrSortedColInd,
                         cscSortedVal,
                         cscSortedRowInd,
                         cscSortedColPtr,
                         hipsparse::hipActionToCudaAction(copyValues),
                         hipsparse::hipIndexBaseToCudaIndexBase(idxBase)));
}

hipsparseStatus_t hipsparseDcsr2csc(hipsparseHandle_t    handle,
                                    int                  m,
                                    int                  n,
                                    int                  nnz,
                                    const double*        csrSortedVal,
                                    const int*           csrSortedRowPtr,
                                    const int*           csrSortedColInd,
                                    double*              cscSortedVal,
                                    int*                 cscSortedRowInd,
                                    int*                 cscSortedColPtr,
                                    hipsparseAction_t    copyValues,
                                    hipsparseIndexBase_t idxBase)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseDcsr2csc((cusparseHandle_t)handle,
                         m,
                         n,
                         nnz,
                         csrSortedVal,
                         csrSortedRowPtr,
                         csrSortedColInd,
                         cscSortedVal,
                         cscSortedRowInd,
                         cscSortedColPtr,
                         hipsparse::hipActionToCudaAction(copyValues),
                         hipsparse::hipIndexBaseToCudaIndexBase(idxBase)));
}

hipsparseStatus_t hipsparseCcsr2csc(hipsparseHandle_t    handle,
                                    int                  m,
                                    int                  n,
                                    int                  nnz,
                                    const hipComplex*    csrSortedVal,
                                    const int*           csrSortedRowPtr,
                                    const int*           csrSortedColInd,
                                    hipComplex*          cscSortedVal,
                                    int*                 cscSortedRowInd,
                                    int*                 cscSortedColPtr,
                                    hipsparseAction_t    copyValues,
                                    hipsparseIndexBase_t idxBase)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseCcsr2csc((cusparseHandle_t)handle,
                         m,
                         n,
                         nnz,
                         (const cuComplex*)csrSortedVal,
                         csrSortedRowPtr,
                         csrSortedColInd,
                         (cuComplex*)cscSortedVal,
                         cscSortedRowInd,
                         cscSortedColPtr,
                         hipsparse::hipActionToCudaAction(copyValues),
                         hipsparse::hipIndexBaseToCudaIndexBase(idxBase)));
}

hipsparseStatus_t hipsparseZcsr2csc(hipsparseHandle_t       handle,
                                    int                     m,
                                    int                     n,
                                    int                     nnz,
                                    const hipDoubleComplex* csrSortedVal,
                                    const int*              csrSortedRowPtr,
                                    const int*              csrSortedColInd,
                                    hipDoubleComplex*       cscSortedVal,
                                    int*                    cscSortedRowInd,
                                    int*                    cscSortedColPtr,
                                    hipsparseAction_t       copyValues,
                                    hipsparseIndexBase_t    idxBase)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseZcsr2csc((cusparseHandle_t)handle,
                         m,
                         n,
                         nnz,
                         (const cuDoubleComplex*)csrSortedVal,
                         csrSortedRowPtr,
                         csrSortedColInd,
                         (cuDoubleComplex*)cscSortedVal,
                         cscSortedRowInd,
                         cscSortedColPtr,
                         hipsparse::hipActionToCudaAction(copyValues),
                         hipsparse::hipIndexBaseToCudaIndexBase(idxBase)));
}
#endif

#if(CUDART_VERSION >= 10010)
hipsparseStatus_t hipsparseCsr2cscEx2_bufferSize(hipsparseHandle_t     handle,
                                                 int                   m,
                                                 int                   n,
                                                 int                   nnz,
                                                 const void*           csrVal,
                                                 const int*            csrRowPtr,
                                                 const int*            csrColInd,
                                                 void*                 cscVal,
                                                 int*                  cscColPtr,
                                                 int*                  cscRowInd,
                                                 hipDataType           valType,
                                                 hipsparseAction_t     copyValues,
                                                 hipsparseIndexBase_t  idxBase,
                                                 hipsparseCsr2CscAlg_t alg,
                                                 size_t*               pBufferSizeInBytes)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseCsr2cscEx2_bufferSize((cusparseHandle_t)handle,
                                      m,
                                      n,
                                      nnz,
                                      csrVal,
                                      csrRowPtr,
                                      csrColInd,
                                      cscVal,
                                      cscColPtr,
                                      cscRowInd,
                                      hipsparse::hipDataTypeToCudaDataType(valType),
                                      hipsparse::hipActionToCudaAction(copyValues),
                                      hipsparse::hipIndexBaseToCudaIndexBase(idxBase),
                                      hipsparse::hipCsr2CscAlgToCudaCsr2CscAlg(alg),
                                      pBufferSizeInBytes));
}

hipsparseStatus_t hipsparseCsr2cscEx2(hipsparseHandle_t     handle,
                                      int                   m,
                                      int                   n,
                                      int                   nnz,
                                      const void*           csrVal,
                                      const int*            csrRowPtr,
                                      const int*            csrColInd,
                                      void*                 cscVal,
                                      int*                  cscColPtr,
                                      int*                  cscRowInd,
                                      hipDataType           valType,
                                      hipsparseAction_t     copyValues,
                                      hipsparseIndexBase_t  idxBase,
                                      hipsparseCsr2CscAlg_t alg,
                                      void*                 buffer)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseCsr2cscEx2((cusparseHandle_t)handle,
                           m,
                           n,
                           nnz,
                           csrVal,
                           csrRowPtr,
                           csrColInd,
                           cscVal,
                           cscColPtr,
                           cscRowInd,
                           hipsparse::hipDataTypeToCudaDataType(valType),
                           hipsparse::hipActionToCudaAction(copyValues),
                           hipsparse::hipIndexBaseToCudaIndexBase(idxBase),
                           hipsparse::hipCsr2CscAlgToCudaCsr2CscAlg(alg),
                           buffer));
}
#endif
