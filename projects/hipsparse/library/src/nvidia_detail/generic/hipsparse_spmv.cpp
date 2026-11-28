/* ************************************************************************
* Copyright (C) 2025-2026 Advanced Micro Devices, Inc. All rights Reserved.
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
hipsparseStatus_t hipsparseSpMV_bufferSize(hipsparseHandle_t           handle,
                                           hipsparseOperation_t        opA,
                                           const void*                 alpha,
                                           hipsparseConstSpMatDescr_t  matA,
                                           hipsparseConstDnVecDescr_t  vecX,
                                           const void*                 beta,
                                           const hipsparseDnVecDescr_t vecY,
                                           hipDataType                 computeType,
                                           hipsparseSpMVAlg_t          alg,
                                           size_t*                     pBufferSizeInBytes)
{
    cusparseOperation_t backend_opA{};
    RETURN_IF_HIPSPARSE_ERROR(hipsparse::hipOperationToCudaOperation(opA, &backend_opA));

    cudaDataType backend_data_type{};
    RETURN_IF_HIPSPARSE_ERROR(
        hipsparse::hipDataTypeToCudaDataType(computeType, &backend_data_type));

    cusparseSpMVAlg_t backend_alg{};
    RETURN_IF_HIPSPARSE_ERROR(hipsparse::hipSpMVAlgToCudaSpMVAlg(alg, &backend_alg));

    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseSpMV_bufferSize((cusparseHandle_t)handle,
                                backend_opA,
                                alpha,
                                (cusparseConstSpMatDescr_t)matA,
                                (cusparseConstDnVecDescr_t)vecX,
                                beta,
                                (const cusparseDnVecDescr_t)vecY,
                                backend_data_type,
                                backend_alg,
                                pBufferSizeInBytes));
}
#elif(CUDART_VERSION > 10010 || (CUDART_VERSION == 10010 && CUDART_10_1_UPDATE_VERSION == 1))
hipsparseStatus_t hipsparseSpMV_bufferSize(hipsparseHandle_t           handle,
                                           hipsparseOperation_t        opA,
                                           const void*                 alpha,
                                           const hipsparseSpMatDescr_t matA,
                                           const hipsparseDnVecDescr_t vecX,
                                           const void*                 beta,
                                           const hipsparseDnVecDescr_t vecY,
                                           hipDataType                 computeType,
                                           hipsparseSpMVAlg_t          alg,
                                           size_t*                     pBufferSizeInBytes)
{
    cusparseOperation_t backend_opA{};
    RETURN_IF_HIPSPARSE_ERROR(hipsparse::hipOperationToCudaOperation(opA, &backend_opA));

    cudaDataType backend_data_type{};
    RETURN_IF_HIPSPARSE_ERROR(
        hipsparse::hipDataTypeToCudaDataType(computeType, &backend_data_type));

    cusparseSpMVAlg_t backend_alg{};
    RETURN_IF_HIPSPARSE_ERROR(hipsparse::hipSpMVAlgToCudaSpMVAlg(alg, &backend_alg));

    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseSpMV_bufferSize((cusparseHandle_t)handle,
                                backend_opA,
                                alpha,
                                (const cusparseSpMatDescr_t)matA,
                                (const cusparseDnVecDescr_t)vecX,
                                beta,
                                (const cusparseDnVecDescr_t)vecY,
                                backend_data_type,
                                backend_alg,
                                pBufferSizeInBytes));
}
#endif

#if(CUDART_VERSION >= 12040)
hipsparseStatus_t hipsparseSpMV_preprocess(hipsparseHandle_t           handle,
                                           hipsparseOperation_t        opA,
                                           const void*                 alpha,
                                           hipsparseConstSpMatDescr_t  matA,
                                           hipsparseConstDnVecDescr_t  vecX,
                                           const void*                 beta,
                                           const hipsparseDnVecDescr_t vecY,
                                           hipDataType                 computeType,
                                           hipsparseSpMVAlg_t          alg,
                                           void*                       externalBuffer)
{
    cusparseOperation_t backend_opA{};
    RETURN_IF_HIPSPARSE_ERROR(hipsparse::hipOperationToCudaOperation(opA, &backend_opA));

    cudaDataType backend_data_type{};
    RETURN_IF_HIPSPARSE_ERROR(
        hipsparse::hipDataTypeToCudaDataType(computeType, &backend_data_type));

    cusparseSpMVAlg_t backend_alg{};
    RETURN_IF_HIPSPARSE_ERROR(hipsparse::hipSpMVAlgToCudaSpMVAlg(alg, &backend_alg));

    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseSpMV_preprocess((cusparseHandle_t)handle,
                                backend_opA,
                                alpha,
                                (const cusparseSpMatDescr_t)matA,
                                (const cusparseDnVecDescr_t)vecX,
                                beta,
                                (const cusparseDnVecDescr_t)vecY,
                                backend_data_type,
                                backend_alg,
                                externalBuffer));
}
#elif(CUDART_VERSION >= 12000)
hipsparseStatus_t hipsparseSpMV_preprocess(hipsparseHandle_t           handle,
                                           hipsparseOperation_t        opA,
                                           const void*                 alpha,
                                           hipsparseConstSpMatDescr_t  matA,
                                           hipsparseConstDnVecDescr_t  vecX,
                                           const void*                 beta,
                                           const hipsparseDnVecDescr_t vecY,
                                           hipDataType                 computeType,
                                           hipsparseSpMVAlg_t          alg,
                                           void*                       externalBuffer)
{
    if(handle == nullptr)
    {
        return HIPSPARSE_STATUS_INVALID_VALUE;
    }

    cusparseOperation_t backend_opA{};
    RETURN_IF_HIPSPARSE_ERROR(hipsparse::hipOperationToCudaOperation(opA, &backend_opA));

    if(matA == nullptr)
    {
        return HIPSPARSE_STATUS_INVALID_VALUE;
    }

    if(vecX == nullptr)
    {
        return HIPSPARSE_STATUS_INVALID_VALUE;
    }

    if(vecY == nullptr)
    {
        return HIPSPARSE_STATUS_INVALID_VALUE;
    }

    if(alpha == nullptr)
    {
        return HIPSPARSE_STATUS_INVALID_VALUE;
    }

    if(beta == nullptr)
    {
        return HIPSPARSE_STATUS_INVALID_VALUE;
    }

    // cuSPARSE has no preprocess stage before CUDA 12.4. Validate computeType anyway so
    // this routine rejects the same values as the other SpMV routines.
    cudaDataType backend_data_type{};
    RETURN_IF_HIPSPARSE_ERROR(
        hipsparse::hipDataTypeToCudaDataType(computeType, &backend_data_type));

    cusparseSpMVAlg_t backend_alg{};
    RETURN_IF_HIPSPARSE_ERROR(hipsparse::hipSpMVAlgToCudaSpMVAlg(alg, &backend_alg));

    if(externalBuffer == nullptr)
    {
        return HIPSPARSE_STATUS_INVALID_VALUE;
    }

    return HIPSPARSE_STATUS_SUCCESS;
}
#elif(CUDART_VERSION > 10010 || (CUDART_VERSION == 10010 && CUDART_10_1_UPDATE_VERSION == 1))
hipsparseStatus_t hipsparseSpMV_preprocess(hipsparseHandle_t           handle,
                                           hipsparseOperation_t        opA,
                                           const void*                 alpha,
                                           const hipsparseSpMatDescr_t matA,
                                           const hipsparseDnVecDescr_t vecX,
                                           const void*                 beta,
                                           const hipsparseDnVecDescr_t vecY,
                                           hipDataType                 computeType,
                                           hipsparseSpMVAlg_t          alg,
                                           void*                       externalBuffer)
{
    if(handle == nullptr)
    {
        return HIPSPARSE_STATUS_INVALID_VALUE;
    }

    cusparseOperation_t backend_opA{};
    RETURN_IF_HIPSPARSE_ERROR(hipsparse::hipOperationToCudaOperation(opA, &backend_opA));

    if(matA == nullptr)
    {
        return HIPSPARSE_STATUS_INVALID_VALUE;
    }

    if(vecX == nullptr)
    {
        return HIPSPARSE_STATUS_INVALID_VALUE;
    }

    if(vecY == nullptr)
    {
        return HIPSPARSE_STATUS_INVALID_VALUE;
    }

    if(alpha == nullptr)
    {
        return HIPSPARSE_STATUS_INVALID_VALUE;
    }

    if(beta == nullptr)
    {
        return HIPSPARSE_STATUS_INVALID_VALUE;
    }

    // cuSPARSE has no preprocess stage before CUDA 12.4. Validate computeType anyway so
    // this routine rejects the same values as the other SpMV routines.
    cudaDataType backend_data_type{};
    RETURN_IF_HIPSPARSE_ERROR(
        hipsparse::hipDataTypeToCudaDataType(computeType, &backend_data_type));

    cusparseSpMVAlg_t backend_alg{};
    RETURN_IF_HIPSPARSE_ERROR(hipsparse::hipSpMVAlgToCudaSpMVAlg(alg, &backend_alg));

    if(externalBuffer == nullptr)
    {
        return HIPSPARSE_STATUS_INVALID_VALUE;
    }

    return HIPSPARSE_STATUS_SUCCESS;
}
#endif

#if(CUDART_VERSION >= 12000)
hipsparseStatus_t hipsparseSpMV(hipsparseHandle_t           handle,
                                hipsparseOperation_t        opA,
                                const void*                 alpha,
                                hipsparseConstSpMatDescr_t  matA,
                                hipsparseConstDnVecDescr_t  vecX,
                                const void*                 beta,
                                const hipsparseDnVecDescr_t vecY,
                                hipDataType                 computeType,
                                hipsparseSpMVAlg_t          alg,
                                void*                       externalBuffer)
{
    cusparseOperation_t backend_opA{};
    RETURN_IF_HIPSPARSE_ERROR(hipsparse::hipOperationToCudaOperation(opA, &backend_opA));

    cudaDataType backend_data_type{};
    RETURN_IF_HIPSPARSE_ERROR(
        hipsparse::hipDataTypeToCudaDataType(computeType, &backend_data_type));

    cusparseSpMVAlg_t backend_alg{};
    RETURN_IF_HIPSPARSE_ERROR(hipsparse::hipSpMVAlgToCudaSpMVAlg(alg, &backend_alg));

    return hipsparse::hipCUSPARSEStatusToHIPStatus(cusparseSpMV((cusparseHandle_t)handle,
                                                                backend_opA,
                                                                alpha,
                                                                (cusparseConstSpMatDescr_t)matA,
                                                                (cusparseConstDnVecDescr_t)vecX,
                                                                beta,
                                                                (const cusparseDnVecDescr_t)vecY,
                                                                backend_data_type,
                                                                backend_alg,
                                                                externalBuffer));
}
#elif(CUDART_VERSION > 10010 || (CUDART_VERSION == 10010 && CUDART_10_1_UPDATE_VERSION == 1))
hipsparseStatus_t hipsparseSpMV(hipsparseHandle_t           handle,
                                hipsparseOperation_t        opA,
                                const void*                 alpha,
                                const hipsparseSpMatDescr_t matA,
                                const hipsparseDnVecDescr_t vecX,
                                const void*                 beta,
                                const hipsparseDnVecDescr_t vecY,
                                hipDataType                 computeType,
                                hipsparseSpMVAlg_t          alg,
                                void*                       externalBuffer)
{
    cusparseOperation_t backend_opA{};
    RETURN_IF_HIPSPARSE_ERROR(hipsparse::hipOperationToCudaOperation(opA, &backend_opA));

    cudaDataType backend_data_type{};
    RETURN_IF_HIPSPARSE_ERROR(
        hipsparse::hipDataTypeToCudaDataType(computeType, &backend_data_type));

    cusparseSpMVAlg_t backend_alg{};
    RETURN_IF_HIPSPARSE_ERROR(hipsparse::hipSpMVAlgToCudaSpMVAlg(alg, &backend_alg));

    return hipsparse::hipCUSPARSEStatusToHIPStatus(cusparseSpMV((cusparseHandle_t)handle,
                                                                backend_opA,
                                                                alpha,
                                                                (const cusparseSpMatDescr_t)matA,
                                                                (const cusparseDnVecDescr_t)vecX,
                                                                beta,
                                                                (const cusparseDnVecDescr_t)vecY,
                                                                backend_data_type,
                                                                backend_alg,
                                                                externalBuffer));
}
#endif
