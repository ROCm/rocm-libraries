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
#include "hipsparse.h"

#include <cuda_runtime_api.h>
#include <cusparse_v2.h>
#include <hip/hip_runtime_api.h>

#include "../utility.h"

#if(CUDART_VERSION >= 13030)
hipsparseStatus_t hipsparseSpGEAM_createDescr(hipsparseSpGEAMDescr_t* descr)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseSpGEAM_createDescr((cusparseSpGEAMDescr_t*)descr));
}
#endif

#if(CUDART_VERSION >= 13030)
hipsparseStatus_t hipsparseSpGEAM_destroyDescr(hipsparseSpGEAMDescr_t descr)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseSpGEAM_destroyDescr((cusparseSpGEAMDescr_t)descr));
}
#endif

#if(CUDART_VERSION >= 13030)
hipsparseStatus_t hipsparseSpGEAM_bufferSize(hipsparseHandle_t          handle,
                                             hipsparseOperation_t       opA,
                                             hipsparseOperation_t       opB,
                                             const void*                alpha,
                                             hipsparseConstSpMatDescr_t matA,
                                             const void*                beta,
                                             hipsparseConstSpMatDescr_t matB,
                                             hipsparseSpMatDescr_t      matC,
                                             hipDataType                computeType,
                                             hipsparseSpGEAMAlg_t       alg,
                                             hipsparseSpGEAMDescr_t     spgeamDescr,
                                             size_t*                    bufferSize)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseSpGEAM_bufferSize((cusparseHandle_t)handle,
                                  hipsparse::hipOperationToCudaOperation(opA),
                                  hipsparse::hipOperationToCudaOperation(opB),
                                  alpha,
                                  (cusparseConstSpMatDescr_t)matA,
                                  beta,
                                  (cusparseConstSpMatDescr_t)matB,
                                  (cusparseSpMatDescr_t)matC,
                                  computeType,
                                  hipsparse::hipSpGEAMAlgToCudaSpGEAMAlg(alg),
                                  (cusparseSpGEAMDescr_t)spgeamDescr,
                                  bufferSize));
}
#endif

#if(CUDART_VERSION >= 13030)
hipsparseStatus_t hipsparseSpGEAM_nnz(hipsparseHandle_t          handle,
                                      hipsparseOperation_t       opA,
                                      hipsparseOperation_t       opB,
                                      const void*                alpha,
                                      hipsparseConstSpMatDescr_t matA,
                                      const void*                beta,
                                      hipsparseConstSpMatDescr_t matB,
                                      hipsparseSpMatDescr_t      matC,
                                      hipDataType                computeType,
                                      hipsparseSpGEAMAlg_t       alg,
                                      hipsparseSpGEAMDescr_t     spgeamDescr,
                                      void*                      externalBuffer)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseSpGEAM_nnz((cusparseHandle_t)handle,
                           hipsparse::hipOperationToCudaOperation(opA),
                           hipsparse::hipOperationToCudaOperation(opB),
                           alpha,
                           (cusparseConstSpMatDescr_t)matA,
                           beta,
                           (cusparseConstSpMatDescr_t)matB,
                           (cusparseSpMatDescr_t)matC,
                           computeType,
                           hipsparse::hipSpGEAMAlgToCudaSpGEAMAlg(alg),
                           (cusparseSpGEAMDescr_t)spgeamDescr,
                           externalBuffer));
}
#endif

#if(CUDART_VERSION >= 13030)
hipsparseStatus_t hipsparseSpGEAM(hipsparseHandle_t          handle,
                                  hipsparseOperation_t       opA,
                                  hipsparseOperation_t       opB,
                                  const void*                alpha,
                                  hipsparseConstSpMatDescr_t matA,
                                  const void*                beta,
                                  hipsparseConstSpMatDescr_t matB,
                                  hipsparseSpMatDescr_t      matC,
                                  hipDataType                computeType,
                                  hipsparseSpGEAMAlg_t       alg,
                                  hipsparseSpGEAMDescr_t     spgeamDescr,
                                  void*                      externalBuffer)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseSpGEAM((cusparseHandle_t)handle,
                       hipsparse::hipOperationToCudaOperation(opA),
                       hipsparse::hipOperationToCudaOperation(opB),
                       alpha,
                       (cusparseConstSpMatDescr_t)matA,
                       beta,
                       (cusparseConstSpMatDescr_t)matB,
                       (cusparseSpMatDescr_t)matC,
                       computeType,
                       hipsparse::hipSpGEAMAlgToCudaSpGEAMAlg(alg),
                       (cusparseSpGEAMDescr_t)spgeamDescr,
                       externalBuffer));
}
#endif
