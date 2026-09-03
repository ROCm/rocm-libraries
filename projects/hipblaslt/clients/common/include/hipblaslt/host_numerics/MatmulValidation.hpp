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
        bool                 compareAllClose  = false;
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
        struct AllCloseTolerance
        {
            double absolute          = 0.0;
            double symmetricRelative = 0.0;
        };

        std::vector<HostComparisonRequest> outputs;
        struct SideOutput
        {
            HostComparisonRequest selected;
            HostComparisonRequest norm;
            bool useComputeNormPolicy = false;
        };
        std::optional<SideOutput> maximum;
        std::optional<SideOutput> auxiliary;
        std::optional<SideOutput> bias;
        AllCloseTolerance         allCloseTolerance;
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
