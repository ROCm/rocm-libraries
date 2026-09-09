// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "datatype_interface.hpp"
#include "hipblaslt_arguments.hpp"
#include <hipblaslt/client/MatmulProblem.hpp>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <roc/host_numerics/amd_gpu_layout/mx.hpp>
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
        // Device allocation and binding facts after optional operand swizzling.
        size_t  elements                       = 0;
        int64_t batchStride                    = 0;
        size_t  scaleElements                  = 0;
        // Present only for block scaling. The host-numerics plan is the single
        // authority for natural shape, physical layout, and exact byte counts.
        std::optional<roc::host_numerics::amd_gpu_layout::MxScaleStoragePlan> mxScaleStorage;
        bool    replacedUnsupportedBatchStride = false;
    };

    // Product execution state derived from a checked MatmulProblem and the
    // API-facing Arguments. It contains device allocation/descriptor policy;
    // reference arithmetic consumes Tensors directly rather than this plan.
    struct PreparedMatmulProblem
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
        std::vector<PreparedMatmulProblem> problems;
        int64_t                            rotatingBytes = 0;
    };

    bool supportsMatmulSwizzle(hipDataType dataType);

    hipblasLtOrder_t matmulOrderForDataType(hipDataType dataType);

    hipblasLtMatmulMatrixScale_t matmulScaleMode(hipblaslt_scaling_format format);

    hipblasLtEpilogue_t matmulEpilogue(const Arguments& arguments);

    MatmulSwizzleParameters matmulSwizzleParameters(hipDataType          dataType,
                                                    hipblasComputeType_t computeType);

    // A and B accept separate physical scale layouts because their independent
    // scaling formats may select different kernel ABIs. Both are values from
    // the one host-numerics MxScaleStorageLayout model above.
    MatmulPreparation prepareMatmulProblems(
        const Arguments&                                         arguments,
        std::span<const MatmulProblem>                           matmulProblems,
        hipDataType                                              inputTypeA,
        hipDataType                                              inputTypeB,
        hipDataType                                              inputTypeC,
        hipDataType                                              outputType,
        hipDataType                                              computeScalarType,
        hipDataType                                              coefficientType,
        hipDataType                                              biasType,
        bool                                                     swizzleA,
        bool                                                     swizzleB,
        roc::host_numerics::amd_gpu_layout::MxScaleStorageLayout scaleLayoutA,
        roc::host_numerics::amd_gpu_layout::MxScaleStorageLayout scaleLayoutB);
} // namespace hipblaslt::client
