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

#if(CUDART_VERSION >= 12000)
hipsparseStatus_t hipsparseSpGEMMreuse_workEstimation(hipsparseHandle_t          handle,
                                                      hipsparseOperation_t       opA,
                                                      hipsparseOperation_t       opB,
                                                      hipsparseConstSpMatDescr_t matA,
                                                      hipsparseConstSpMatDescr_t matB,
                                                      hipsparseSpMatDescr_t      matC,
                                                      hipsparseSpGEMMAlg_t       alg,
                                                      hipsparseSpGEMMDescr_t     spgemmDescr,
                                                      size_t*                    bufferSize1,
                                                      void*                      externalBuffer1)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseSpGEMMreuse_workEstimation((cusparseHandle_t)handle,
                                           hipsparse::hipOperationToCudaOperation(opA),
                                           hipsparse::hipOperationToCudaOperation(opB),
                                           (cusparseConstSpMatDescr_t)matA,
                                           (cusparseConstSpMatDescr_t)matB,
                                           (cusparseSpMatDescr_t)matC,
                                           hipsparse::hipSpGEMMAlgToCudaSpGEMMAlg(alg),
                                           (cusparseSpGEMMDescr_t)spgemmDescr,
                                           bufferSize1,
                                           externalBuffer1));
}
#elif(CUDART_VERSION >= 11031)
hipsparseStatus_t hipsparseSpGEMMreuse_workEstimation(hipsparseHandle_t      handle,
                                                      hipsparseOperation_t   opA,
                                                      hipsparseOperation_t   opB,
                                                      hipsparseSpMatDescr_t  matA,
                                                      hipsparseSpMatDescr_t  matB,
                                                      hipsparseSpMatDescr_t  matC,
                                                      hipsparseSpGEMMAlg_t   alg,
                                                      hipsparseSpGEMMDescr_t spgemmDescr,
                                                      size_t*                bufferSize1,
                                                      void*                  externalBuffer1)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseSpGEMMreuse_workEstimation((cusparseHandle_t)handle,
                                           hipsparse::hipOperationToCudaOperation(opA),
                                           hipsparse::hipOperationToCudaOperation(opB),
                                           (cusparseSpMatDescr_t)matA,
                                           (cusparseSpMatDescr_t)matB,
                                           (cusparseSpMatDescr_t)matC,
                                           hipsparse::hipSpGEMMAlgToCudaSpGEMMAlg(alg),
                                           (cusparseSpGEMMDescr_t)spgemmDescr,
                                           bufferSize1,
                                           externalBuffer1));
}
#endif

#if(CUDART_VERSION >= 12000)
hipsparseStatus_t hipsparseSpGEMMreuse_nnz(hipsparseHandle_t          handle,
                                           hipsparseOperation_t       opA,
                                           hipsparseOperation_t       opB,
                                           hipsparseConstSpMatDescr_t matA,
                                           hipsparseConstSpMatDescr_t matB,
                                           hipsparseSpMatDescr_t      matC,
                                           hipsparseSpGEMMAlg_t       alg,
                                           hipsparseSpGEMMDescr_t     spgemmDescr,
                                           size_t*                    bufferSize2,
                                           void*                      externalBuffer2,
                                           size_t*                    bufferSize3,
                                           void*                      externalBuffer3,
                                           size_t*                    bufferSize4,
                                           void*                      externalBuffer4)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseSpGEMMreuse_nnz((cusparseHandle_t)handle,
                                hipsparse::hipOperationToCudaOperation(opA),
                                hipsparse::hipOperationToCudaOperation(opB),
                                (cusparseConstSpMatDescr_t)matA,
                                (cusparseConstSpMatDescr_t)matB,
                                (cusparseSpMatDescr_t)matC,
                                hipsparse::hipSpGEMMAlgToCudaSpGEMMAlg(alg),
                                (cusparseSpGEMMDescr_t)spgemmDescr,
                                bufferSize2,
                                externalBuffer2,
                                bufferSize3,
                                externalBuffer3,
                                bufferSize4,
                                externalBuffer4));
}
#elif(CUDART_VERSION >= 11031)
hipsparseStatus_t hipsparseSpGEMMreuse_nnz(hipsparseHandle_t      handle,
                                           hipsparseOperation_t   opA,
                                           hipsparseOperation_t   opB,
                                           hipsparseSpMatDescr_t  matA,
                                           hipsparseSpMatDescr_t  matB,
                                           hipsparseSpMatDescr_t  matC,
                                           hipsparseSpGEMMAlg_t   alg,
                                           hipsparseSpGEMMDescr_t spgemmDescr,
                                           size_t*                bufferSize2,
                                           void*                  externalBuffer2,
                                           size_t*                bufferSize3,
                                           void*                  externalBuffer3,
                                           size_t*                bufferSize4,
                                           void*                  externalBuffer4)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseSpGEMMreuse_nnz((cusparseHandle_t)handle,
                                hipsparse::hipOperationToCudaOperation(opA),
                                hipsparse::hipOperationToCudaOperation(opB),
                                (cusparseSpMatDescr_t)matA,
                                (cusparseSpMatDescr_t)matB,
                                (cusparseSpMatDescr_t)matC,
                                hipsparse::hipSpGEMMAlgToCudaSpGEMMAlg(alg),
                                (cusparseSpGEMMDescr_t)spgemmDescr,
                                bufferSize2,
                                externalBuffer2,
                                bufferSize3,
                                externalBuffer3,
                                bufferSize4,
                                externalBuffer4));
}

#endif

#if(CUDART_VERSION >= 12000)
hipsparseStatus_t hipsparseSpGEMMreuse_compute(hipsparseHandle_t          handle,
                                               hipsparseOperation_t       opA,
                                               hipsparseOperation_t       opB,
                                               const void*                alpha,
                                               hipsparseConstSpMatDescr_t matA,
                                               hipsparseConstSpMatDescr_t matB,
                                               const void*                beta,
                                               hipsparseSpMatDescr_t      matC,
                                               hipDataType                computeType,
                                               hipsparseSpGEMMAlg_t       alg,
                                               hipsparseSpGEMMDescr_t     spgemmDescr)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseSpGEMMreuse_compute((cusparseHandle_t)handle,
                                    hipsparse::hipOperationToCudaOperation(opA),
                                    hipsparse::hipOperationToCudaOperation(opB),
                                    alpha,
                                    (cusparseConstSpMatDescr_t)matA,
                                    (cusparseConstSpMatDescr_t)matB,
                                    beta,
                                    (cusparseSpMatDescr_t)matC,
                                    computeType,
                                    hipsparse::hipSpGEMMAlgToCudaSpGEMMAlg(alg),
                                    (cusparseSpGEMMDescr_t)spgemmDescr));
}
#elif(CUDART_VERSION >= 11031)
hipsparseStatus_t hipsparseSpGEMMreuse_compute(hipsparseHandle_t      handle,
                                               hipsparseOperation_t   opA,
                                               hipsparseOperation_t   opB,
                                               const void*            alpha,
                                               hipsparseSpMatDescr_t  matA,
                                               hipsparseSpMatDescr_t  matB,
                                               const void*            beta,
                                               hipsparseSpMatDescr_t  matC,
                                               hipDataType            computeType,
                                               hipsparseSpGEMMAlg_t   alg,
                                               hipsparseSpGEMMDescr_t spgemmDescr)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseSpGEMMreuse_compute((cusparseHandle_t)handle,
                                    hipsparse::hipOperationToCudaOperation(opA),
                                    hipsparse::hipOperationToCudaOperation(opB),
                                    alpha,
                                    (cusparseSpMatDescr_t)matA,
                                    (cusparseSpMatDescr_t)matB,
                                    beta,
                                    (cusparseSpMatDescr_t)matC,
                                    computeType,
                                    hipsparse::hipSpGEMMAlgToCudaSpGEMMAlg(alg),
                                    (cusparseSpGEMMDescr_t)spgemmDescr));
}
#endif

#if(CUDART_VERSION >= 12000)
hipsparseStatus_t hipsparseSpGEMMreuse_copy(hipsparseHandle_t          handle,
                                            hipsparseOperation_t       opA,
                                            hipsparseOperation_t       opB,
                                            hipsparseConstSpMatDescr_t matA,
                                            hipsparseConstSpMatDescr_t matB,
                                            hipsparseSpMatDescr_t      matC,
                                            hipsparseSpGEMMAlg_t       alg,
                                            hipsparseSpGEMMDescr_t     spgemmDescr,
                                            size_t*                    bufferSize5,
                                            void*                      externalBuffer5)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseSpGEMMreuse_copy((cusparseHandle_t)handle,
                                 hipsparse::hipOperationToCudaOperation(opA),
                                 hipsparse::hipOperationToCudaOperation(opB),
                                 (cusparseConstSpMatDescr_t)matA,
                                 (cusparseConstSpMatDescr_t)matB,
                                 (cusparseSpMatDescr_t)matC,
                                 hipsparse::hipSpGEMMAlgToCudaSpGEMMAlg(alg),
                                 (cusparseSpGEMMDescr_t)spgemmDescr,
                                 bufferSize5,
                                 externalBuffer5));
}
#elif(CUDART_VERSION >= 11031)
hipsparseStatus_t hipsparseSpGEMMreuse_copy(hipsparseHandle_t      handle,
                                            hipsparseOperation_t   opA,
                                            hipsparseOperation_t   opB,
                                            hipsparseSpMatDescr_t  matA,
                                            hipsparseSpMatDescr_t  matB,
                                            hipsparseSpMatDescr_t  matC,
                                            hipsparseSpGEMMAlg_t   alg,
                                            hipsparseSpGEMMDescr_t spgemmDescr,
                                            size_t*                bufferSize5,
                                            void*                  externalBuffer5)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseSpGEMMreuse_copy((cusparseHandle_t)handle,
                                 hipsparse::hipOperationToCudaOperation(opA),
                                 hipsparse::hipOperationToCudaOperation(opB),
                                 (cusparseSpMatDescr_t)matA,
                                 (cusparseSpMatDescr_t)matB,
                                 (cusparseSpMatDescr_t)matC,
                                 hipsparse::hipSpGEMMAlgToCudaSpGEMMAlg(alg),
                                 (cusparseSpGEMMDescr_t)spgemmDescr,
                                 bufferSize5,
                                 externalBuffer5));
}

#endif
