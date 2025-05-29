/*! \file */
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

#include <hip/hip_complex.h>
#include <hip/hip_runtime_api.h>
#include <rocsparse/rocsparse.h>

#include "../utility.h"

struct hipsparseSpGEMMDescr
{
    size_t bufferSize1{};
    size_t bufferSize2{};
    size_t bufferSize3{};
    size_t bufferSize4{};
    size_t bufferSize5{};

    void* externalBuffer1{};
    void* externalBuffer2{};
    void* externalBuffer3{};
    void* externalBuffer4{};
    void* externalBuffer5{};
};

namespace hipsparse
{
    static hipsparseStatus_t getIndexTypeSize(hipsparseIndexType_t indexType, size_t& size)
    {
        switch(indexType)
        {
        case HIPSPARSE_INDEX_16U:
        {
            size = sizeof(uint16_t);
            return HIPSPARSE_STATUS_SUCCESS;
        }
        case HIPSPARSE_INDEX_32I:
        {
            size = sizeof(int32_t);
            return HIPSPARSE_STATUS_SUCCESS;
        }
        case HIPSPARSE_INDEX_64I:
        {
            size = sizeof(int64_t);
            return HIPSPARSE_STATUS_SUCCESS;
        }
        }

        size = 0;
        return HIPSPARSE_STATUS_INVALID_VALUE;
    }

    static hipsparseStatus_t getDataTypeSize(hipDataType dataType, size_t& size)
    {
        switch(dataType)
        {
        case HIP_R_32F:
        {
            size = sizeof(float);
            return HIPSPARSE_STATUS_SUCCESS;
        }
        case HIP_R_64F:
        {
            size = sizeof(double);
            return HIPSPARSE_STATUS_SUCCESS;
        }
        case HIP_C_32F:
        {
            size = sizeof(hipComplex);
            return HIPSPARSE_STATUS_SUCCESS;
        }
        case HIP_C_64F:
        {
            size = sizeof(hipDoubleComplex);
            return HIPSPARSE_STATUS_SUCCESS;
        }
        default:
        {
            size = 0;
            return HIPSPARSE_STATUS_INVALID_VALUE;
        }
        }

        size = 0;
        return HIPSPARSE_STATUS_INVALID_VALUE;
    }
}

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
    const void* alpha = (const void*)0x4;
    const void* beta  = (const void*)0x4;

    // Get data stored in C matrix
    int64_t              rowsC, colsC, nnzC;
    void*                csrRowOffsetsC;
    void*                csrColIndC;
    void*                csrValuesC;
    hipsparseIndexType_t csrRowOffsetsTypeC;
    hipsparseIndexType_t csrColIndTypeC;
    hipsparseIndexBase_t idxBaseC;
    hipDataType          csrValueTypeC;
    RETURN_IF_HIPSPARSE_ERROR(hipsparseCsrGet(matC,
                                              &rowsC,
                                              &colsC,
                                              &nnzC,
                                              &csrRowOffsetsC,
                                              &csrColIndC,
                                              &csrValuesC,
                                              &csrRowOffsetsTypeC,
                                              &csrColIndTypeC,
                                              &idxBaseC,
                                              &csrValueTypeC));

    hipDataType computeType = csrValueTypeC;

    return hipsparseSpGEMM_workEstimation(handle,
                                          opA,
                                          opB,
                                          alpha,
                                          matA,
                                          matB,
                                          beta,
                                          matC,
                                          computeType,
                                          alg,
                                          spgemmDescr,
                                          bufferSize1,
                                          externalBuffer1);
}

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
    // Match cusparse error handling
    if(handle == nullptr || matA == nullptr || matB == nullptr || matC == nullptr
       || spgemmDescr == nullptr || bufferSize2 == nullptr || bufferSize3 == nullptr
       || bufferSize3 == nullptr || bufferSize4 == nullptr)
    {
        return HIPSPARSE_STATUS_INVALID_VALUE;
    }

    // If any external buffer is nullptr, they must all be nullptr.
    bool allBuffersNull = (externalBuffer2 == nullptr) & (externalBuffer3 == nullptr)
                          & (externalBuffer4 == nullptr);
    bool anyBuffersNull = (externalBuffer2 == nullptr) | (externalBuffer3 == nullptr)
                          | (externalBuffer4 == nullptr);
    if(anyBuffersNull && !allBuffersNull)
    {
        return HIPSPARSE_STATUS_INVALID_VALUE;
    }

    const void* alpha = (const void*)0x4;

    // Get data stored in C matrix
    int64_t              rowsC, colsC, nnzC;
    void*                csrRowOffsetsC;
    void*                csrColIndC;
    void*                csrValuesC;
    hipsparseIndexType_t csrRowOffsetsTypeC;
    hipsparseIndexType_t csrColIndTypeC;
    hipsparseIndexBase_t idxBaseC;
    hipDataType          csrValueTypeC;
    RETURN_IF_HIPSPARSE_ERROR(hipsparseCsrGet(matC,
                                              &rowsC,
                                              &colsC,
                                              &nnzC,
                                              &csrRowOffsetsC,
                                              &csrColIndC,
                                              &csrValuesC,
                                              &csrRowOffsetsTypeC,
                                              &csrColIndTypeC,
                                              &idxBaseC,
                                              &csrValueTypeC));

    hipDataType computeType = csrValueTypeC;

    size_t csrRowOffsetsTypeSizeC;
    size_t csrColIndTypeSizeC;
    size_t csrValueTypeSizeC;
    RETURN_IF_HIPSPARSE_ERROR(
        hipsparse::getIndexTypeSize(csrRowOffsetsTypeC, csrRowOffsetsTypeSizeC));
    RETURN_IF_HIPSPARSE_ERROR(hipsparse::getIndexTypeSize(csrColIndTypeC, csrColIndTypeSizeC));
    RETURN_IF_HIPSPARSE_ERROR(hipsparse::getDataTypeSize(csrValueTypeC, csrValueTypeSizeC));

    if(allBuffersNull)
    {
        *bufferSize2 = 4;
        *bufferSize3 = 4;
        *bufferSize4 = spgemmDescr->bufferSize1;

        *bufferSize3 += ((csrColIndTypeSizeC * nnzC - 1) / 256 + 1) * 256;
        *bufferSize3 += ((csrValueTypeSizeC * nnzC - 1) / 256 + 1) * 256;

        spgemmDescr->bufferSize2 = *bufferSize2;
        spgemmDescr->bufferSize3 = *bufferSize3;
        spgemmDescr->bufferSize4 = *bufferSize4;
    }
    else
    {
        hipStream_t stream;
        RETURN_IF_HIPSPARSE_ERROR(hipsparseGetStream(handle, &stream));

        spgemmDescr->externalBuffer2 = externalBuffer2;
        spgemmDescr->externalBuffer3 = externalBuffer3; // stores C column indices and values
        spgemmDescr->externalBuffer4
            = externalBuffer4; // stores C row pointers array + spgemm buffer

        size_t byteOffset3 = ((csrColIndTypeSizeC * nnzC - 1) / 256 + 1) * 256;
        size_t byteOffset4 = ((csrRowOffsetsTypeSizeC * (rowsC + 1) - 1) / 256 + 1) * 256;

        // Transfer external buffer 1 (which stores C row pointer array) to external buffer 4 because
        // buffer 1 can be deleted after this function
        RETURN_IF_HIP_ERROR(hipMemcpyAsync(spgemmDescr->externalBuffer4,
                                           spgemmDescr->externalBuffer1,
                                           *bufferSize4,
                                           hipMemcpyDeviceToDevice,
                                           stream));

        // Temporarily set in C matrix in order to compute C row pointer array (stored in externalBuffer1)
        RETURN_IF_HIPSPARSE_ERROR(hipsparseCsrSetPointers(
            matC,
            spgemmDescr->externalBuffer4,
            spgemmDescr->externalBuffer3,
            static_cast<char*>(spgemmDescr->externalBuffer3) + byteOffset3));

        size_t bufferSize = (spgemmDescr->bufferSize4 - byteOffset4);
        RETURN_IF_ROCSPARSE_ERROR(
            rocsparse_spgemm((rocsparse_handle)handle,
                             hipsparse::hipOperationToHCCOperation(opA),
                             hipsparse::hipOperationToHCCOperation(opB),
                             alpha,
                             to_rocsparse_const_spmat_descr(matA),
                             to_rocsparse_const_spmat_descr(matB),
                             nullptr,
                             to_rocsparse_const_spmat_descr(matC),
                             to_rocsparse_spmat_descr(matC),
                             hipsparse::hipDataTypeToHCCDataType(computeType),
                             hipsparse::hipSpGEMMAlgToHCCSpGEMMAlg(alg),
                             rocsparse_spgemm_stage_symbolic,
                             &bufferSize,
                             static_cast<char*>(spgemmDescr->externalBuffer4) + byteOffset4));
    }

    return HIPSPARSE_STATUS_SUCCESS;
}

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
    if(handle == nullptr || matA == nullptr || matB == nullptr || matC == nullptr
       || bufferSize5 == nullptr || spgemmDescr == nullptr)
    {
        return HIPSPARSE_STATUS_INVALID_VALUE;
    }

    // Get data stored in C matrix
    int64_t              rowsC, colsC, nnzC;
    void*                csrRowOffsetsC;
    void*                csrColIndC;
    void*                csrValuesC;
    hipsparseIndexType_t csrRowOffsetsTypeC;
    hipsparseIndexType_t csrColIndTypeC;
    hipsparseIndexBase_t idxBaseC;
    hipDataType          csrValueTypeC;
    RETURN_IF_HIPSPARSE_ERROR(hipsparseCsrGet(matC,
                                              &rowsC,
                                              &colsC,
                                              &nnzC,
                                              &csrRowOffsetsC,
                                              &csrColIndC,
                                              &csrValuesC,
                                              &csrRowOffsetsTypeC,
                                              &csrColIndTypeC,
                                              &idxBaseC,
                                              &csrValueTypeC));

    size_t csrRowOffsetsTypeSizeC;
    size_t csrColIndTypeSizeC;
    size_t csrValueTypeSizeC;
    RETURN_IF_HIPSPARSE_ERROR(
        hipsparse::getIndexTypeSize(csrRowOffsetsTypeC, csrRowOffsetsTypeSizeC));
    RETURN_IF_HIPSPARSE_ERROR(hipsparse::getIndexTypeSize(csrColIndTypeC, csrColIndTypeSizeC));
    RETURN_IF_HIPSPARSE_ERROR(hipsparse::getDataTypeSize(csrValueTypeC, csrValueTypeSizeC));

    if(externalBuffer5 == nullptr)
    {
        *bufferSize5 = 0;

        // Need to store temporary space for values array
        *bufferSize5 += ((csrValueTypeSizeC * nnzC - 1) / 256 + 1) * 256;

        // Need to store temporary space for indices array used in hipsparseSpGEMM_copy Axpby
        *bufferSize5 += ((csrColIndTypeSizeC * nnzC - 1) / 256 + 1) * 256;

        // Need to store temporary space for host/device 1 value used in hipsparseSpGEMM_copy Axpby
        *bufferSize5 += ((16 - 1) / 256 + 1) * 256;

        spgemmDescr->bufferSize5 = *bufferSize5;
    }
    else
    {
        spgemmDescr->externalBuffer5 = externalBuffer5;

        hipStream_t stream;
        RETURN_IF_HIPSPARSE_ERROR(hipsparseGetStream(handle, &stream));

        RETURN_IF_HIP_ERROR(hipMemcpyAsync(csrRowOffsetsC,
                                           spgemmDescr->externalBuffer4,
                                           csrRowOffsetsTypeSizeC * (rowsC + 1),
                                           hipMemcpyDeviceToDevice,
                                           stream));
        RETURN_IF_HIP_ERROR(hipMemcpyAsync(csrColIndC,
                                           spgemmDescr->externalBuffer3,
                                           csrColIndTypeSizeC * nnzC,
                                           hipMemcpyDeviceToDevice,
                                           stream));

        RETURN_IF_HIPSPARSE_ERROR(
            hipsparseCsrSetPointers(matC, csrRowOffsetsC, csrColIndC, csrValuesC));
    }

    return HIPSPARSE_STATUS_SUCCESS;
}

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
    if(handle == nullptr || alpha == nullptr || beta == nullptr || spgemmDescr == nullptr)
    {
        return HIPSPARSE_STATUS_INVALID_VALUE;
    }

    // Get data stored in C matrix
    int64_t              rowsC, colsC, nnzC;
    void*                csrRowOffsetsC;
    void*                csrColIndC;
    void*                csrValuesC;
    hipsparseIndexType_t csrRowOffsetsTypeC;
    hipsparseIndexType_t csrColIndTypeC;
    hipsparseIndexBase_t idxBaseC;
    hipDataType          csrValueTypeC;
    RETURN_IF_HIPSPARSE_ERROR(hipsparseCsrGet(matC,
                                              &rowsC,
                                              &colsC,
                                              &nnzC,
                                              &csrRowOffsetsC,
                                              &csrColIndC,
                                              &csrValuesC,
                                              &csrRowOffsetsTypeC,
                                              &csrColIndTypeC,
                                              &idxBaseC,
                                              &csrValueTypeC));

    size_t csrRowOffsetsTypeSizeC;
    size_t csrColIndTypeSizeC;
    size_t csrValueTypeSizeC;
    RETURN_IF_HIPSPARSE_ERROR(
        hipsparse::getIndexTypeSize(csrRowOffsetsTypeC, csrRowOffsetsTypeSizeC));
    RETURN_IF_HIPSPARSE_ERROR(hipsparse::getIndexTypeSize(csrColIndTypeC, csrColIndTypeSizeC));
    RETURN_IF_HIPSPARSE_ERROR(hipsparse::getDataTypeSize(csrValueTypeC, csrValueTypeSizeC));

    size_t byteOffset4 = ((csrRowOffsetsTypeSizeC * (rowsC + 1) - 1) / 256 + 1) * 256;
    size_t byteOffset5 = 0;

    void* csrValuesCFromBuffer5 = spgemmDescr->externalBuffer5;
    byteOffset5 += ((csrValueTypeSizeC * nnzC - 1) / 256 + 1) * 256;

    void* indicesArray = (static_cast<char*>(spgemmDescr->externalBuffer5) + byteOffset5);
    byteOffset5 += ((csrColIndTypeSizeC * nnzC - 1) / 256 + 1) * 256;

    void* device_one = (static_cast<char*>(spgemmDescr->externalBuffer5) + byteOffset5);

    // Use external buffer for values array as the original values array may have data in it
    // that must be accounted for when multiplying by beta. See below.
    RETURN_IF_HIPSPARSE_ERROR(
        hipsparseCsrSetPointers(matC, csrRowOffsetsC, csrColIndC, csrValuesCFromBuffer5));

    size_t bufferSize = (spgemmDescr->bufferSize4 - byteOffset4);
    RETURN_IF_ROCSPARSE_ERROR(
        rocsparse_spgemm((rocsparse_handle)handle,
                         hipsparse::hipOperationToHCCOperation(opA),
                         hipsparse::hipOperationToHCCOperation(opB),
                         alpha,
                         to_rocsparse_const_spmat_descr(matA),
                         to_rocsparse_const_spmat_descr(matB),
                         nullptr,
                         to_rocsparse_const_spmat_descr(matC),
                         to_rocsparse_spmat_descr(matC),
                         hipsparse::hipDataTypeToHCCDataType(computeType),
                         hipsparse::hipSpGEMMAlgToHCCSpGEMMAlg(alg),
                         rocsparse_spgemm_stage_numeric,
                         &bufferSize,
                         static_cast<char*>(spgemmDescr->externalBuffer4) + byteOffset4));

    // Get pointer mode
    hipsparsePointerMode_t pointer_mode;
    RETURN_IF_HIPSPARSE_ERROR(hipsparseGetPointerMode(handle, &pointer_mode));

    hipStream_t stream;
    RETURN_IF_HIPSPARSE_ERROR(hipsparseGetStream(handle, &stream));

    float            host_sone = 1.0f;
    double           host_done = 1.0f;
    hipComplex       host_cone = make_hipComplex(1.0f, 0.0f);
    hipDoubleComplex host_zone = make_hipDoubleComplex(1.0, 0.0);

    void* one = nullptr;
    if(pointer_mode == HIPSPARSE_POINTER_MODE_HOST)
    {
        if(computeType == HIP_R_32F)
            one = &host_sone;
        if(computeType == HIP_R_64F)
            one = &host_done;
        if(computeType == HIP_C_32F)
            one = &host_cone;
        if(computeType == HIP_C_64F)
            one = &host_zone;
    }
    else
    {
        if(computeType == HIP_R_32F)
        {
            RETURN_IF_HIP_ERROR(hipMemcpyAsync(
                device_one, &host_sone, sizeof(float), hipMemcpyHostToDevice, stream));
            one = device_one;
        }
        if(computeType == HIP_R_64F)
        {
            RETURN_IF_HIP_ERROR(hipMemcpyAsync(
                device_one, &host_done, sizeof(double), hipMemcpyHostToDevice, stream));
            one = device_one;
        }
        if(computeType == HIP_C_32F)
        {
            RETURN_IF_HIP_ERROR(hipMemcpyAsync(
                device_one, &host_cone, sizeof(hipComplex), hipMemcpyHostToDevice, stream));
            one = device_one;
        }
        if(computeType == HIP_C_64F)
        {
            RETURN_IF_HIP_ERROR(hipMemcpyAsync(
                device_one, &host_zone, sizeof(hipDoubleComplex), hipMemcpyHostToDevice, stream));
            one = device_one;
        }
    }

    if(csrColIndTypeC == HIPSPARSE_INDEX_32I)
    {
        RETURN_IF_ROCSPARSE_ERROR(
            rocsparse_set_identity_permutation((rocsparse_handle)handle,
                                               nnzC,
                                               static_cast<int32_t*>(indicesArray),
                                               rocsparse_indextype_i32));
    }
    else if(csrColIndTypeC == HIPSPARSE_INDEX_64I)
    {
        RETURN_IF_ROCSPARSE_ERROR(
            rocsparse_set_identity_permutation((rocsparse_handle)handle,
                                               nnzC,
                                               static_cast<int64_t*>(indicesArray),
                                               rocsparse_indextype_i64));
    }
    else
    {
        return HIPSPARSE_STATUS_NOT_SUPPORTED;
    }

    hipsparseConstSpVecDescr_t vecX;
    RETURN_IF_HIPSPARSE_ERROR(hipsparseCreateConstSpVec(&vecX,
                                                        nnzC,
                                                        nnzC,
                                                        indicesArray,
                                                        csrValuesCFromBuffer5,
                                                        csrColIndTypeC,
                                                        HIPSPARSE_INDEX_BASE_ZERO,
                                                        csrValueTypeC));

    hipsparseDnVecDescr_t vecY;
    RETURN_IF_HIPSPARSE_ERROR(hipsparseCreateDnVec(&vecY, nnzC, csrValuesC, csrValueTypeC));

    // Axpby computes: Y = alpha * X + beta * Y
    // What we want to compute: csrValuesC = 1.0 * csrValuesCFromBuffer5 + beta * csrValuesC
    RETURN_IF_HIPSPARSE_ERROR(hipsparseAxpby(handle, one, vecX, beta, vecY));

    RETURN_IF_HIPSPARSE_ERROR(hipsparseDestroySpVec(vecX));
    RETURN_IF_HIPSPARSE_ERROR(hipsparseDestroyDnVec(vecY));

    // Finally, update C matrix
    RETURN_IF_HIPSPARSE_ERROR(
        hipsparseCsrSetPointers(matC, csrRowOffsetsC, csrColIndC, csrValuesC));

    return HIPSPARSE_STATUS_SUCCESS;
}
