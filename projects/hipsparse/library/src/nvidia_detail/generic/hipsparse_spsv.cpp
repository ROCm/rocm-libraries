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

#if(CUDART_VERSION >= 11030)
hipsparseStatus_t hipsparseSpSV_createDescr(hipsparseSpSVDescr_t* descr)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseSpSV_createDescr((cusparseSpSVDescr_t*)descr));
}
#endif

#if(CUDART_VERSION >= 11030)
hipsparseStatus_t hipsparseSpSV_destroyDescr(hipsparseSpSVDescr_t descr)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseSpSV_destroyDescr((cusparseSpSVDescr_t)descr));
}
#endif

#if(CUDART_VERSION >= 12000)
hipsparseStatus_t hipsparseSpSV_bufferSize(hipsparseHandle_t           handle,
                                           hipsparseOperation_t        opA,
                                           const void*                 alpha,
                                           hipsparseConstSpMatDescr_t  matA,
                                           hipsparseConstDnVecDescr_t  x,
                                           const hipsparseDnVecDescr_t y,
                                           hipDataType                 computeType,
                                           hipsparseSpSVAlg_t          alg,
                                           hipsparseSpSVDescr_t        spsvDescr,
                                           size_t*                     pBufferSizeInBytes)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseSpSV_bufferSize((cusparseHandle_t)handle,
                                hipsparse::hipOperationToCudaOperation(opA),
                                alpha,
                                (cusparseConstSpMatDescr_t)matA,
                                (cusparseConstDnVecDescr_t)x,
                                (const cusparseDnVecDescr_t)y,
                                hipsparse::hipDataTypeToCudaDataType(computeType),
                                hipsparse::hipSpSVAlgToCudaSpSVAlg(alg),
                                (cusparseSpSVDescr_t)spsvDescr,
                                pBufferSizeInBytes));
}
#elif(CUDART_VERSION >= 11030)
hipsparseStatus_t hipsparseSpSV_bufferSize(hipsparseHandle_t           handle,
                                           hipsparseOperation_t        opA,
                                           const void*                 alpha,
                                           const hipsparseSpMatDescr_t matA,
                                           const hipsparseDnVecDescr_t x,
                                           const hipsparseDnVecDescr_t y,
                                           hipDataType                 computeType,
                                           hipsparseSpSVAlg_t          alg,
                                           hipsparseSpSVDescr_t        spsvDescr,
                                           size_t*                     pBufferSizeInBytes)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseSpSV_bufferSize((cusparseHandle_t)handle,
                                hipsparse::hipOperationToCudaOperation(opA),
                                alpha,
                                (const cusparseSpMatDescr_t)matA,
                                (const cusparseDnVecDescr_t)x,
                                (const cusparseDnVecDescr_t)y,
                                hipsparse::hipDataTypeToCudaDataType(computeType),
                                hipsparse::hipSpSVAlgToCudaSpSVAlg(alg),
                                (cusparseSpSVDescr_t)spsvDescr,
                                pBufferSizeInBytes));
}
#endif

#if(CUDART_VERSION >= 12000)
hipsparseStatus_t hipsparseSpSV_analysis(hipsparseHandle_t           handle,
                                         hipsparseOperation_t        opA,
                                         const void*                 alpha,
                                         hipsparseConstSpMatDescr_t  matA,
                                         hipsparseConstDnVecDescr_t  x,
                                         const hipsparseDnVecDescr_t y,
                                         hipDataType                 computeType,
                                         hipsparseSpSVAlg_t          alg,
                                         hipsparseSpSVDescr_t        spsvDescr,
                                         void*                       externalBuffer)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseSpSV_analysis((cusparseHandle_t)handle,
                              hipsparse::hipOperationToCudaOperation(opA),
                              alpha,
                              (cusparseConstSpMatDescr_t)matA,
                              (cusparseConstDnVecDescr_t)x,
                              (const cusparseDnVecDescr_t)y,
                              hipsparse::hipDataTypeToCudaDataType(computeType),
                              hipsparse::hipSpSVAlgToCudaSpSVAlg(alg),
                              (cusparseSpSVDescr_t)spsvDescr,
                              externalBuffer));
}
#elif(CUDART_VERSION >= 11030)
hipsparseStatus_t hipsparseSpSV_analysis(hipsparseHandle_t           handle,
                                         hipsparseOperation_t        opA,
                                         const void*                 alpha,
                                         const hipsparseSpMatDescr_t matA,
                                         const hipsparseDnVecDescr_t x,
                                         const hipsparseDnVecDescr_t y,
                                         hipDataType                 computeType,
                                         hipsparseSpSVAlg_t          alg,
                                         hipsparseSpSVDescr_t        spsvDescr,
                                         void*                       externalBuffer)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseSpSV_analysis((cusparseHandle_t)handle,
                              hipsparse::hipOperationToCudaOperation(opA),
                              alpha,
                              (const cusparseSpMatDescr_t)matA,
                              (const cusparseDnVecDescr_t)x,
                              (const cusparseDnVecDescr_t)y,
                              hipsparse::hipDataTypeToCudaDataType(computeType),
                              hipsparse::hipSpSVAlgToCudaSpSVAlg(alg),
                              (cusparseSpSVDescr_t)spsvDescr,
                              externalBuffer));
}
#endif

#if(CUDART_VERSION >= 12000)
hipsparseStatus_t hipsparseSpSV_solve(hipsparseHandle_t           handle,
                                      hipsparseOperation_t        opA,
                                      const void*                 alpha,
                                      hipsparseConstSpMatDescr_t  matA,
                                      hipsparseConstDnVecDescr_t  x,
                                      const hipsparseDnVecDescr_t y,
                                      hipDataType                 computeType,
                                      hipsparseSpSVAlg_t          alg,
                                      hipsparseSpSVDescr_t        spsvDescr)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseSpSV_solve((cusparseHandle_t)handle,
                           hipsparse::hipOperationToCudaOperation(opA),
                           alpha,
                           (cusparseConstSpMatDescr_t)matA,
                           (cusparseConstDnVecDescr_t)x,
                           (const cusparseDnVecDescr_t)y,
                           hipsparse::hipDataTypeToCudaDataType(computeType),
                           hipsparse::hipSpSVAlgToCudaSpSVAlg(alg),
                           (cusparseSpSVDescr_t)spsvDescr));
}
#elif(CUDART_VERSION >= 11030)
hipsparseStatus_t hipsparseSpSV_solve(hipsparseHandle_t           handle,
                                      hipsparseOperation_t        opA,
                                      const void*                 alpha,
                                      const hipsparseSpMatDescr_t matA,
                                      const hipsparseDnVecDescr_t x,
                                      const hipsparseDnVecDescr_t y,
                                      hipDataType                 computeType,
                                      hipsparseSpSVAlg_t          alg,
                                      hipsparseSpSVDescr_t        spsvDescr)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseSpSV_solve((cusparseHandle_t)handle,
                           hipsparse::hipOperationToCudaOperation(opA),
                           alpha,
                           (const cusparseSpMatDescr_t)matA,
                           (const cusparseDnVecDescr_t)x,
                           (const cusparseDnVecDescr_t)y,
                           hipsparse::hipDataTypeToCudaDataType(computeType),
                           hipsparse::hipSpSVAlgToCudaSpSVAlg(alg),
                           (cusparseSpSVDescr_t)spsvDescr));
}
#endif
