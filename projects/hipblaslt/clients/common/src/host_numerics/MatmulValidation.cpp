// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <hipblaslt/host_numerics/MatmulValidation.hpp>
#include <hipblaslt/host_numerics/norm.hpp>

#include <algorithm>
#include <cmath>
#include <optional>
#include <utility>
#include <vector>

namespace hipblaslt::host_numerics
{
    namespace
    {
        HostComparisonReport compareMatmulOutput(const HostComparisonRequest&   source,
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
                    request.pointwise = validationCase.pointwiseTolerance.absolute != 0
                                            ? HostPointwiseComparison::Near
                                            : HostPointwiseComparison::Unit;
                    request.absoluteTolerance = validationCase.pointwiseTolerance.absolute;
                }
            }
            return compareHost(request);
        }
    } // namespace

    MatmulValidationResult validateMatmulOutputs(const MatmulValidationRequest& request)
    {
        MatmulValidationResult result;
        const auto             record = [&](bool passed) {
            result.failedChecks += static_cast<size_t>(!passed);
        };
        double ulpSum           = 0.0;
        size_t ulpCount         = 0;
        // Each output is searched independently. The componentwise maxima form
        // a tolerance that accepts every successful output; any failed search
        // dominates the aggregate regardless of output order.
        bool   allCloseCompared = false;
        bool   allCloseFailed   = false;
        double requiredAbsolute = 0.0;
        double requiredRelative = 0.0;

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
                    record(report.comparison.nonFiniteMismatches == 0);
            }
            if(request.options.comparePointwise)
            {
                for(const auto& report : outputReports)
                    record(report.comparison.passed());
            }

            if(request.options.compareNorm)
            {
                for(const auto& report : outputReports)
                {
                    const double normError = std::abs(report.relativeFrobeniusError);
                    request.metrics.relativeFrobeniusError += normError;
                    if(request.options.assertNorm)
                    {
                        record(norm_check(normError,
                                          validationCase.outputs.front().type,
                                          request.options.computeType,
                                          request.options.inputTypeA,
                                          request.options.inputTypeB));
                    }
                }
            }

            if(request.options.searchAllClose)
            {
                for(size_t outputIndex = 0; outputIndex < outputReports.size(); ++outputIndex)
                {
                    const auto& output = validationCase.outputs[outputIndex];
                    const auto& report = outputReports[outputIndex];
                    if(output.rows == 0 || output.columns == 0 || output.batchCount == 0)
                        continue;

                    allCloseCompared = true;
                    if(report.allCloseTolerance)
                    {
                        requiredAbsolute
                            = std::max(requiredAbsolute, report.allCloseTolerance->absolute);
                        requiredRelative
                            = std::max(requiredRelative, report.allCloseTolerance->relative);
                    }
                    else
                        allCloseFailed = true;
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

            const auto validateSideOutput
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
                          record(report.comparison.passed());
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
                              record(output->useComputeNormPolicy
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

        if(request.options.searchAllClose && allCloseCompared)
        {
            request.metrics.absoluteTolerance = allCloseFailed ? 1.0 : requiredAbsolute;
            request.metrics.relativeTolerance = allCloseFailed ? 1.0 : requiredRelative;
        }
        if(request.options.computeUlp && ulpCount != 0)
            request.metrics.averageUlp = ulpSum / ulpCount;
        return result;
    }
} // namespace hipblaslt::host_numerics
