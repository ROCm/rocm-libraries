// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// Product-private hipBLASLt reference GEMM adapter.

#include "datatype_interface.hpp"

#include <hipblaslt/hipblaslt.h>
#include <hipblaslt/client/MatmulPreparation.hpp>
#include <roc/host_numerics/gemm.hpp>

#include <optional>
#include <utility>

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

    struct MatmulReferenceInputs
    {
        MatmulReferenceInputs(roc::host_numerics::Tensor aTensor,
                              roc::host_numerics::Tensor bTensor,
                              roc::host_numerics::Tensor cTensor,
                              roc::host_numerics::Tensor dTensor)
            : a(std::move(aTensor))
            , b(std::move(bTensor))
            , c(std::move(cTensor))
            , d(std::move(dTensor))
        {
        }

        roc::host_numerics::Tensor                a;
        roc::host_numerics::Tensor                b;
        roc::host_numerics::Tensor                c;
        roc::host_numerics::Tensor                d;
        std::optional<roc::host_numerics::Tensor> alphaVector;
        std::optional<roc::host_numerics::Tensor> scaleA;
        std::optional<roc::host_numerics::Tensor> scaleB;
        std::optional<roc::host_numerics::Scalar> scaleC;
        std::optional<roc::host_numerics::Scalar> scaleD;
    };

    roc::host_numerics::GemmRunInfo
        referenceMatmulGemm(const hipblaslt::client::MatmulProblem&         problem,
                            const hipblaslt::client::MatmulDataTypes&       dataTypes,
                            const hipblaslt::client::PreparedMatmulProblem& preparation,
                            MatmulReferenceInputs                            inputs,
                            hipblaslt_scaling_format                         scaleAMode,
                            hipblaslt_scaling_format                         scaleBMode);
} // namespace hipblaslt::host_numerics
