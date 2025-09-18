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
hipsparseStatus_t hipsparseSpMM_bufferSize(hipsparseHandle_t           handle,
                                           hipsparseOperation_t        opA,
                                           hipsparseOperation_t        opB,
                                           const void*                 alpha,
                                           hipsparseConstSpMatDescr_t  matA,
                                           hipsparseConstDnMatDescr_t  matB,
                                           const void*                 beta,
                                           const hipsparseDnMatDescr_t matC,
                                           hipDataType                 computeType,
                                           hipsparseSpMMAlg_t          alg,
                                           size_t*                     pBufferSizeInBytes)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseSpMM_bufferSize((cusparseHandle_t)handle,
                                hipsparse::hipOperationToCudaOperation(opA),
                                hipsparse::hipOperationToCudaOperation(opB),
                                alpha,
                                (cusparseConstSpMatDescr_t)matA,
                                (cusparseConstDnMatDescr_t)matB,
                                beta,
                                (const cusparseDnMatDescr_t)matC,
                                hipsparse::hipDataTypeToCudaDataType(computeType),
                                hipsparse::hipSpMMAlgToCudaSpMMAlg(alg),
                                pBufferSizeInBytes));
}
#elif(CUDART_VERSION >= 10010)
hipsparseStatus_t hipsparseSpMM_bufferSize(hipsparseHandle_t           handle,
                                           hipsparseOperation_t        opA,
                                           hipsparseOperation_t        opB,
                                           const void*                 alpha,
                                           const hipsparseSpMatDescr_t matA,
                                           const hipsparseDnMatDescr_t matB,
                                           const void*                 beta,
                                           const hipsparseDnMatDescr_t matC,
                                           hipDataType                 computeType,
                                           hipsparseSpMMAlg_t          alg,
                                           size_t*                     pBufferSizeInBytes)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseSpMM_bufferSize((cusparseHandle_t)handle,
                                hipsparse::hipOperationToCudaOperation(opA),
                                hipsparse::hipOperationToCudaOperation(opB),
                                alpha,
                                (const cusparseSpMatDescr_t)matA,
                                (const cusparseDnMatDescr_t)matB,
                                beta,
                                (const cusparseDnMatDescr_t)matC,
                                hipsparse::hipDataTypeToCudaDataType(computeType),
                                hipsparse::hipSpMMAlgToCudaSpMMAlg(alg),
                                pBufferSizeInBytes));
}
#endif

#if(CUDART_VERSION >= 12000)
hipsparseStatus_t hipsparseSpMM_preprocess(hipsparseHandle_t           handle,
                                           hipsparseOperation_t        opA,
                                           hipsparseOperation_t        opB,
                                           const void*                 alpha,
                                           hipsparseConstSpMatDescr_t  matA,
                                           hipsparseConstDnMatDescr_t  matB,
                                           const void*                 beta,
                                           const hipsparseDnMatDescr_t matC,
                                           hipDataType                 computeType,
                                           hipsparseSpMMAlg_t          alg,
                                           void*                       externalBuffer)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseSpMM_preprocess((cusparseHandle_t)handle,
                                hipsparse::hipOperationToCudaOperation(opA),
                                hipsparse::hipOperationToCudaOperation(opB),
                                alpha,
                                (cusparseConstSpMatDescr_t)matA,
                                (cusparseConstDnMatDescr_t)matB,
                                beta,
                                (const cusparseDnMatDescr_t)matC,
                                hipsparse::hipDataTypeToCudaDataType(computeType),
                                hipsparse::hipSpMMAlgToCudaSpMMAlg(alg),
                                externalBuffer));
}
#elif(CUDART_VERSION >= 11021)
hipsparseStatus_t hipsparseSpMM_preprocess(hipsparseHandle_t           handle,
                                           hipsparseOperation_t        opA,
                                           hipsparseOperation_t        opB,
                                           const void*                 alpha,
                                           const hipsparseSpMatDescr_t matA,
                                           const hipsparseDnMatDescr_t matB,
                                           const void*                 beta,
                                           const hipsparseDnMatDescr_t matC,
                                           hipDataType                 computeType,
                                           hipsparseSpMMAlg_t          alg,
                                           void*                       externalBuffer)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseSpMM_preprocess((cusparseHandle_t)handle,
                                hipsparse::hipOperationToCudaOperation(opA),
                                hipsparse::hipOperationToCudaOperation(opB),
                                alpha,
                                (const cusparseSpMatDescr_t)matA,
                                (const cusparseDnMatDescr_t)matB,
                                beta,
                                (const cusparseDnMatDescr_t)matC,
                                hipsparse::hipDataTypeToCudaDataType(computeType),
                                hipsparse::hipSpMMAlgToCudaSpMMAlg(alg),
                                externalBuffer));
}
#endif

#if(CUDART_VERSION >= 12000)
hipsparseStatus_t hipsparseSpMM(hipsparseHandle_t           handle,
                                hipsparseOperation_t        opA,
                                hipsparseOperation_t        opB,
                                const void*                 alpha,
                                hipsparseConstSpMatDescr_t  matA,
                                hipsparseConstDnMatDescr_t  matB,
                                const void*                 beta,
                                const hipsparseDnMatDescr_t matC,
                                hipDataType                 computeType,
                                hipsparseSpMMAlg_t          alg,
                                void*                       externalBuffer)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseSpMM((cusparseHandle_t)handle,
                     hipsparse::hipOperationToCudaOperation(opA),
                     hipsparse::hipOperationToCudaOperation(opB),
                     alpha,
                     (cusparseConstSpMatDescr_t)matA,
                     (cusparseConstDnMatDescr_t)matB,
                     beta,
                     (const cusparseDnMatDescr_t)matC,
                     hipsparse::hipDataTypeToCudaDataType(computeType),
                     hipsparse::hipSpMMAlgToCudaSpMMAlg(alg),
                     externalBuffer));
}
#elif(CUDART_VERSION >= 10010)
hipsparseStatus_t hipsparseSpMM(hipsparseHandle_t           handle,
                                hipsparseOperation_t        opA,
                                hipsparseOperation_t        opB,
                                const void*                 alpha,
                                const hipsparseSpMatDescr_t matA,
                                const hipsparseDnMatDescr_t matB,
                                const void*                 beta,
                                const hipsparseDnMatDescr_t matC,
                                hipDataType                 computeType,
                                hipsparseSpMMAlg_t          alg,
                                void*                       externalBuffer)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseSpMM((cusparseHandle_t)handle,
                     hipsparse::hipOperationToCudaOperation(opA),
                     hipsparse::hipOperationToCudaOperation(opB),
                     alpha,
                     (const cusparseSpMatDescr_t)matA,
                     (const cusparseDnMatDescr_t)matB,
                     beta,
                     (const cusparseDnMatDescr_t)matC,
                     hipsparse::hipDataTypeToCudaDataType(computeType),
                     hipsparse::hipSpMMAlgToCudaSpMMAlg(alg),
                     externalBuffer));
}
#endif
