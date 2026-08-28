// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// Product-private hipBLASLt reference GEMM adapter.

#include "datatype_interface.hpp"

#include <hipblaslt/hipblaslt.h>

namespace hipblaslt::host_numerics
{
    struct HipblasltReferenceGemmRequest
    {
        hipblasOperation_t   operationA = HIPBLAS_OP_N;
        hipblasOperation_t   operationB = HIPBLAS_OP_N;
        int64_t              rows       = 0;
        int64_t              columns    = 0;
        int64_t              reduction  = 0;
        computeTypeInterface alpha{};
        computeTypeInterface beta{};
        const void*          a                 = nullptr;
        const void*          b                 = nullptr;
        const void*          c                 = nullptr;
        void*                d                 = nullptr;
        int64_t              leadingDimensionA = 0;
        int64_t              leadingDimensionB = 0;
        int64_t              leadingDimensionC = 0;
        int64_t              leadingDimensionD = 0;
        const void*          alphaVector       = nullptr;
        const void*          scaleA            = nullptr;
        const void*          scaleB            = nullptr;
        const void*          scaleC            = nullptr;
        const void*          scaleD            = nullptr;
        bool                 scaleAIsVector    = false;
        bool                 scaleBIsVector    = false;
        hipDataType          typeA             = HIP_R_32F;
        hipDataType          typeB             = HIP_R_32F;
        hipDataType          typeC             = HIP_R_32F;
        hipDataType          typeD             = HIP_R_32F;
        hipDataType          coefficientType   = HIP_R_32F;
        hipDataType          computeInputTypeA = HIP_R_32F;
        hipDataType          computeInputTypeB = HIP_R_32F;
        bool                 scaleAIsMx        = false;
        bool                 scaleBIsMx        = false;
    };

    void hipblaslt_reference_gemm(const HipblasltReferenceGemmRequest& request);
} // namespace hipblaslt::host_numerics
