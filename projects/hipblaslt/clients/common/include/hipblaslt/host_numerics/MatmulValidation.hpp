// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hipblaslt/host_numerics/Types.hpp>
#include <optional>
#include <roc/host_numerics/comparison.hpp>
#include <span>
#include <utility>
#include <vector>

namespace hipblaslt::host_numerics
{
    struct MatmulValidationOptions
    {
        bool                 compareAllClose  = false;
        bool                compareNorm      = false;
        bool                searchAllClose   = false;
        bool                computeUlp       = false;
        bool                assertNorm       = false;
        hipblasComputeType_t computeType      = HIPBLAS_COMPUTE_32F;
        roc::host_numerics::ScalarType inputTypeA       = roc::host_numerics::ScalarType::Float32;
        roc::host_numerics::ScalarType inputTypeB       = roc::host_numerics::ScalarType::Float32;
    };

    struct MatmulValidationCase
    {
        using TensorPair = std::pair<roc::host_numerics::Tensor, roc::host_numerics::Tensor>;

        std::vector<TensorPair> outputs;
        struct SideOutput
        {
            TensorPair selected;
            TensorPair norm;
            bool       useComputeNormPolicy = false;
        };
        std::optional<SideOutput> maximum;
        std::optional<SideOutput> auxiliary;
        std::optional<SideOutput> bias;
        roc::host_numerics::ComparisonTolerance allCloseTolerance;
    };

    struct MatmulValidationMetrics
    {
        double& relativeFrobeniusError;
        double& absoluteTolerance;
        double& relativeTolerance;
        double& maximumUlp;
        double& averageUlp;
    };

    bool validateMatmulOutputs(const MatmulValidationOptions&        options,
                               std::span<const MatmulValidationCase> cases,
                               MatmulValidationMetrics               metrics);
} // namespace hipblaslt::host_numerics
