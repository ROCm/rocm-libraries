// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "datatype_interface.hpp"
#include "hipblaslt_arguments.hpp"
#include <hipblaslt/client/MatmulTestCase.hpp>

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace hipblaslt::client
{
    struct MatmulSwizzleParameters
    {
        size_t innerBlock;
        size_t vectorWidth;
        size_t packingFactor;
    };

    struct PreparedMatmulOperand
    {
        size_t  elements                       = 0;
        int64_t batchStride                    = 0;
        size_t  scaleElements                  = 0;
        bool    replacedUnsupportedBatchStride = false;
    };

    struct PreparedMatmulCase
    {
        PreparedMatmulOperand a;
        PreparedMatmulOperand b;
        size_t                outputCopyElements = 0;
        size_t                biasElements       = 0;
        size_t                scaleAlphaElements = 0;
        hipblasLtEpilogue_t   epilogue           = HIPBLASLT_EPILOGUE_DEFAULT;
        bool                  epilogueEnabled    = false;
        float                 activation0        = 0.0f;
        float                 activation1        = 0.0f;
        computeTypeInterface  alpha{};
        computeTypeInterface  beta{};
    };

    struct MatmulPreparation
    {
        std::vector<PreparedMatmulCase> cases;
        int64_t                         rotatingBytes = 0;
    };

    bool supportsMatmulSwizzle(hipDataType dataType);

    bool usesRocrollerMxLayout();

    hipblasLtOrder_t matmulOrderForDataType(hipDataType dataType);

    hipblasLtMatmulMatrixScale_t matmulScaleMode(hipblaslt_scaling_format format);

    hipblasLtEpilogue_t matmulEpilogue(const Arguments& arguments);

    MatmulSwizzleParameters matmulSwizzleParameters(hipDataType          dataType,
                                                    hipblasComputeType_t computeType);

    MatmulPreparation prepareMatmulCases(const Arguments&                arguments,
                                         std::span<const MatmulTestCase> matmulCases,
                                         hipDataType                     inputTypeA,
                                         hipDataType                     inputTypeB,
                                         hipDataType                     inputTypeC,
                                         hipDataType                     outputType,
                                         hipDataType                     computeScalarType,
                                         hipDataType                     coefficientType,
                                         hipDataType                     biasType,
                                         bool                            swizzleA,
                                         bool                            swizzleB,
                                         bool                            useRocrollerMxLayout);
} // namespace hipblaslt::client
