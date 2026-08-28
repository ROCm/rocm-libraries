// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hipblaslt/host_numerics/HostComparison.hpp>
#include <optional>
#include <span>
#include <vector>

namespace hipblaslt::host_numerics
{
    struct MatmulValidationOptions
    {
        bool                comparePointwise = false;
        bool                compareNorm      = false;
        bool                searchAllClose   = false;
        bool                computeUlp       = false;
        bool                assertNorm       = false;
        hipblasComputeType_t computeType      = HIPBLAS_COMPUTE_32F;
        hipDataType          inputTypeA       = HIP_R_32F;
        hipDataType          inputTypeB       = HIP_R_32F;
    };

    struct MatmulValidationCase
    {
        struct PointwiseTolerance
        {
            double absolute          = 0.0;
            double symmetricRelative = 0.0;
        };

        std::vector<HostComparisonRequest> outputs;
        struct SideOutput
        {
            HostComparisonRequest pointwise;
            HostComparisonRequest norm;
            bool useComputeNormPolicy = false;
        };
        std::optional<SideOutput> maximum;
        std::optional<SideOutput> auxiliary;
        std::optional<SideOutput> bias;
        PointwiseTolerance pointwiseTolerance;
    };

    struct MatmulValidationMetrics
    {
        double& relativeFrobeniusError;
        double& absoluteTolerance;
        double& relativeTolerance;
        double& maximumUlp;
        double& averageUlp;
    };

    struct MatmulValidationRequest
    {
        MatmulValidationOptions options;
        std::span<const MatmulValidationCase> cases;
        MatmulValidationMetrics metrics;
    };

    struct MatmulValidationResult
    {
        size_t failedChecks = 0;

        bool passed() const
        {
            return failedChecks == 0;
        }
    };

    MatmulValidationResult validateMatmulOutputs(const MatmulValidationRequest& request);
} // namespace hipblaslt::host_numerics
