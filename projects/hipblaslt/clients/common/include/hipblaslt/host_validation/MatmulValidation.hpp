// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "hipblaslt_test.hpp"
#include "norm.hpp"
#include <algorithm>
#include <cmath>
#include <hipblaslt/host_validation/HostComparison.hpp>
#include <optional>
#include <span>
#include <vector>

namespace hipblaslt::host_validation
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

    inline HostComparisonReport compareMatmulOutput(const HostComparisonRequest&   source,
                                                    const MatmulValidationCase&    validationCase,
                                                    const MatmulValidationOptions& options,
                                                    bool specialValueConsistency,
                                                    bool searchAllClose,
                                                    bool computeUlp)
    {
        HostComparisonRequest request          = source;
        request.requireSpecialValueConsistency = specialValueConsistency;
        request.computeRelativeFrobeniusError  = options.compareNorm;
        request.findAllCloseTolerance          = searchAllClose;
        request.computeUnitsInLastPlace        = computeUlp;

        if(options.comparePointwise)
        {
            if(validationCase.pointwiseTolerance.symmetricRelative != 0)
            {
                request.pointwise = HostPointwiseComparison::SymmetricRelative;
                request.symmetricRelativeTolerance
                    = validationCase.pointwiseTolerance.symmetricRelative;
            }
            else
            {
                request.pointwise         = validationCase.pointwiseTolerance.absolute != 0
                                                ? HostPointwiseComparison::Near
                                                : HostPointwiseComparison::Unit;
                request.absoluteTolerance = validationCase.pointwiseTolerance.absolute;
            }
        }
        return compareHost(request);
    }

    inline void validateMatmulOutputs(const MatmulValidationRequest& request)
    {
        double ulpSum   = 0.0;
        size_t ulpCount = 0;

        for(const auto& validationCase : request.cases)
        {
            std::vector<HostComparisonReport> outputReports;
            outputReports.reserve(validationCase.outputs.size());
            for(const auto& output : validationCase.outputs)
            {
                outputReports.push_back(compareMatmulOutput(output,
                                                            validationCase,
                                                            request.options,
                                                            request.options.comparePointwise
                                                                || request.options.compareNorm,
                                                            request.options.searchAllClose,
                                                            request.options.computeUlp));
            }

            if(request.options.comparePointwise || request.options.compareNorm)
            {
                for(const auto& report : outputReports)
                    CHECK_SUCCESS(report.comparison.nonFiniteMismatches == 0);
            }
            if(request.options.comparePointwise)
            {
                for(const auto& report : outputReports)
                    CHECK_SUCCESS(report.comparison.passed());
            }

            if(request.options.compareNorm)
            {
                for(const auto& report : outputReports)
                {
                    const double normError = std::abs(report.relativeFrobeniusError);
                    request.metrics.relativeFrobeniusError += normError;
                    if(request.options.assertNorm)
                    {
                        CHECK_SUCCESS(norm_check(normError,
                                                 validationCase.outputs.front().type,
                                                 request.options.computeType,
                                                 request.options.inputTypeA,
                                                 request.options.inputTypeB));
                    }
                }
            }

            if(request.options.searchAllClose)
            {
                for(const auto& report : outputReports)
                {
                    if(report.allCloseTolerance)
                    {
                        request.metrics.absoluteTolerance = report.allCloseTolerance->absolute;
                        request.metrics.relativeTolerance = report.allCloseTolerance->relative;
                    }
                    else if(report.comparison.compared != 0)
                    {
                        request.metrics.absoluteTolerance = 1.0;
                        request.metrics.relativeTolerance = 1.0;
                    }
                }
            }

            if(request.options.computeUlp)
            {
                for(const auto& report : outputReports)
                {
                    request.metrics.maximumUlp = std::max(
                        request.metrics.maximumUlp, report.unitsInLastPlaceComparison.maximumUlp);
                    ulpSum += report.unitsInLastPlaceComparison.sumUlp;
                    ulpCount += report.unitsInLastPlaceComparison.ulpCompared;
                }
            }

            auto validateSideOutput
                = [&](const std::optional<MatmulValidationCase::SideOutput>& output) {
                      if(!output)
                          return;
                      if(request.options.comparePointwise)
                      {
                          auto pointwiseOptions        = request.options;
                          pointwiseOptions.compareNorm = false;
                          const auto report            = compareMatmulOutput(output->pointwise,
                                                                             validationCase,
                                                                             pointwiseOptions,
                                                                             false,
                                                                             false,
                                                                             false);
                          CHECK_SUCCESS(report.comparison.passed());
                      }
                      if(request.options.compareNorm)
                      {
                          auto normOptions             = request.options;
                          normOptions.comparePointwise = false;
                          const auto report            = compareMatmulOutput(
                              output->norm, validationCase, normOptions, false, false, false);
                          const double normError = std::abs(report.relativeFrobeniusError);
                          request.metrics.relativeFrobeniusError += normError;
                          if(request.options.assertNorm)
                          {
                              CHECK_SUCCESS(output->useComputeNormPolicy
                                                ? norm_check(normError,
                                                             output->norm.type,
                                                             request.options.computeType,
                                                             request.options.inputTypeA,
                                                             request.options.inputTypeB)
                                                : norm_check(normError, output->norm.type));
                          }
                      }
                  };

            validateSideOutput(validationCase.maximum);
            validateSideOutput(validationCase.auxiliary);
            validateSideOutput(validationCase.bias);
        }

        if(request.options.computeUlp && ulpCount != 0)
            request.metrics.averageUlp = ulpSum / ulpCount;
    }
} // namespace hipblaslt::host_validation
