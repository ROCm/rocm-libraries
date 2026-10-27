// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hipblaslt/hostnumerics/Types.hpp>
#include <optional>
#include <roc/hostnumerics/comparison.hpp>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace hipblaslt::hostnumerics
{
    struct MatmulValidationOptions
    {
        bool                          compareAllClose = false;
        bool                          compareNorm     = false;
        bool                          searchAllClose  = false;
        bool                          computeUlp      = false;
        bool                          assertNorm      = false;
        hipblasComputeType_t          computeType     = HIPBLAS_COMPUTE_32F;
        roc::hostnumerics::ScalarType inputTypeA      = roc::hostnumerics::ScalarType::Float32;
        roc::hostnumerics::ScalarType inputTypeB      = roc::hostnumerics::ScalarType::Float32;
    };

    struct MatmulValidationCase
    {
        using TensorPair = std::pair<roc::hostnumerics::Tensor, roc::hostnumerics::Tensor>;

        std::vector<TensorPair> outputs;
        struct SideOutput
        {
            TensorPair selected;
            TensorPair norm;
            bool       useComputeNormPolicy = false;
        };
        std::optional<SideOutput>              maximum;
        std::optional<SideOutput>              auxiliary;
        std::optional<SideOutput>              bias;
        roc::hostnumerics::ComparisonTolerance allCloseTolerance;
    };

    struct MatmulAllCloseFailure
    {
        // Complex real and imaginary components are counted separately.
        size_t badCount      = 0;
        size_t comparedCount = 0;
        size_t gemm          = 0;
        size_t batch         = 0;
        size_t row           = 0;
        size_t column        = 0;
        bool   imaginary     = false;
        double observed      = 0.0;
        double expected      = 0.0;
        double error         = 0.0;
    };

    [[nodiscard]] std::string formatAllCloseFailure(const MatmulAllCloseFailure& failure);

    struct MatmulValidationSummary
    {
        bool                                 passed                 = true;
        double                               relativeFrobeniusError = 0.0;
        double                               absoluteTolerance      = 0.0;
        double                               relativeTolerance      = 0.0;
        double                               maximumUlp             = 0.0;
        double                               averageUlp             = 0.0;
        std::optional<MatmulAllCloseFailure> allCloseFailure;
    };

    [[nodiscard]] MatmulValidationSummary
        validateMatmulOutputs(const MatmulValidationOptions&        options,
                              std::span<const MatmulValidationCase> cases);
} // namespace hipblaslt::hostnumerics
