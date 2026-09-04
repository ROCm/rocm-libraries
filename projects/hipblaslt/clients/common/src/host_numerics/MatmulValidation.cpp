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

            if(options.compareAllClose)
            {
                if(validationCase.allCloseTolerance.symmetricRelative != 0)
                {
                    request.allCloseMode = HostAllCloseMode::SymmetricRelative;
                    request.symmetricRelativeTolerance
                        = validationCase.allCloseTolerance.symmetricRelative;
                }
                else
                {
                    request.allCloseMode      = validationCase.allCloseTolerance.absolute != 0
                                                    ? HostAllCloseMode::Near
                                                    : HostAllCloseMode::Unit;
                    request.absoluteTolerance = validationCase.allCloseTolerance.absolute;
                }
            }
            return compareHost(request);
        }
    } // namespace

    bool validateMatmulOutputs(const MatmulValidationOptions&        options,
                               std::span<const MatmulValidationCase> cases,
                               MatmulValidationMetrics               metrics)
    {
        size_t     failedChecks = 0;
        const auto record = [&](bool passed) { failedChecks += static_cast<size_t>(!passed); };
        double ulpSum           = 0.0;
        size_t ulpCount         = 0;
        // Each output is searched independently. The componentwise maxima form
        // a tolerance that accepts every successful output; any failed search
        // dominates the aggregate regardless of output order.
        bool   allCloseCompared = false;
        bool   allCloseFailed   = false;
        double requiredAbsolute = 0.0;
        double requiredRelative = 0.0;

        for(const auto& validationCase : cases)
        {
            std::vector<HostComparisonReport> outputReports;
            outputReports.reserve(validationCase.outputs.size());
            for(const auto& output : validationCase.outputs)
            {
                outputReports.push_back(
                    compareMatmulOutput(output,
                                        validationCase,
                                        options,
                                        options.compareAllClose || options.compareNorm,
                                        options.searchAllClose,
                                        options.computeUlp));
            }

            if(options.compareAllClose || options.compareNorm)
            {
                for(const auto& report : outputReports)
                    record(report.comparison.nonFiniteMismatches == 0);
            }
            if(options.compareAllClose)
            {
                for(const auto& report : outputReports)
                    record(report.comparison.passed());
            }

            if(options.compareNorm)
            {
                for(const auto& report : outputReports)
                {
                    const double normError = std::abs(report.relativeFrobeniusError);
                    metrics.relativeFrobeniusError += normError;
                    if(options.assertNorm)
                    {
                        record(norm_check(normError,
                                          validationCase.outputs.front().type,
                                          options.computeType,
                                          options.inputTypeA,
                                          options.inputTypeB));
                    }
                }
            }

            if(options.searchAllClose)
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

            if(options.computeUlp)
            {
                for(const auto& report : outputReports)
                {
                    metrics.maximumUlp = std::max(metrics.maximumUlp,
                                                  report.unitsInLastPlaceComparison.maximumUlp);
                    ulpSum += report.unitsInLastPlaceComparison.sumUlp;
                    ulpCount += report.unitsInLastPlaceComparison.ulpCompared;
                }
            }

            const auto validateSideOutput
                = [&](const std::optional<MatmulValidationCase::SideOutput>& output) {
                      if(!output)
                          return;
                      if(options.compareAllClose)
                      {
                          auto selectedOptions        = options;
                          selectedOptions.compareNorm = false;
                          const auto report           = compareMatmulOutput(output->selected,
                                                                  validationCase,
                                                                  selectedOptions,
                                                                  false,
                                                                  false,
                                                                  false);
                          record(report.comparison.passed());
                      }
                      if(options.compareNorm)
                      {
                          auto normOptions             = options;
                          normOptions.compareAllClose  = false;
                          const auto report            = compareMatmulOutput(
                              output->norm, validationCase, normOptions, false, false, false);
                          const double normError = std::abs(report.relativeFrobeniusError);
                          metrics.relativeFrobeniusError += normError;
                          if(options.assertNorm)
                          {
                              record(output->useComputeNormPolicy
                                         ? norm_check(normError,
                                                      output->norm.type,
                                                      options.computeType,
                                                      options.inputTypeA,
                                                      options.inputTypeB)
                                         : norm_check(normError, output->norm.type));
                          }
                      }
                  };

            validateSideOutput(validationCase.maximum);
            validateSideOutput(validationCase.auxiliary);
            validateSideOutput(validationCase.bias);
        }

        if(options.searchAllClose && allCloseCompared)
        {
            metrics.absoluteTolerance = allCloseFailed ? 1.0 : requiredAbsolute;
            metrics.relativeTolerance = allCloseFailed ? 1.0 : requiredRelative;
        }
        if(options.computeUlp && ulpCount != 0)
            metrics.averageUlp = ulpSum / ulpCount;
        return failedChecks == 0;
    }
} // namespace hipblaslt::host_numerics
