// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <hipblaslt/host_numerics/MatmulValidation.hpp>
#include <hipblaslt/host_numerics/norm.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <optional>
#include <utility>
#include <vector>

namespace hipblaslt::host_numerics
{
    using namespace roc::host_numerics;

    namespace
    {
        struct ComparisonEvidence
        {
            ComparisonReport                   comparison;
            ComparisonReport                   unitsInLastPlace;
            double                             relativeFrobeniusError = 0.0;
            std::optional<ComparisonTolerance> allCloseTolerance;
        };

        ComparisonOptions unitComparisonOptions(ScalarType type)
        {
            ComparisonOptions options;
            options.equalNaNs                    = true;
            options.computeElementwiseStatistics = false;
            options.computeFrobenius             = false;
            options.maxReportedMismatches        = 10;

            if(type == ScalarType::Float32 || type == ScalarType::Float64
               || type == ScalarType::ComplexFloat32 || type == ScalarType::ComplexFloat64)
            {
                options.allClose            = false;
                options.computeUlp          = true;
                options.ulpType             = type;
                options.maximumUlpTolerance = 4.0;
            }
            return options;
        }

        Tensor batchView(const Tensor& tensor, size_t batch)
        {
            if(tensor.shape().rank() != 3)
                return tensor;
            const std::array<size_t, 3> origin{0, 0, batch};
            return tensor.shareStorageWithLayout(
                Layout(Shape{tensor.shape()[0], tensor.shape()[1]},
                       {tensor.layout().stride(0), tensor.layout().stride(1)},
                       tensor.layout().elementOffset(origin)));
        }

        ComparisonEvidence compareMatmulOutput(const MatmulValidationCase::TensorPair& tensors,
                                               const ComparisonTolerance&              tolerance,
                                               const MatmulValidationOptions&          validation,
                                               bool specialValueConsistency,
                                               bool searchAllClose,
                                               bool computeUlp)
        {
            const auto& [expected, observed] = tensors;
            ComparisonEvidence result;

            if(validation.compareAllClose || specialValueConsistency)
            {
                ComparisonOptions options;
                if(validation.compareAllClose)
                    options = tolerance.absolute != 0.0 || tolerance.relative != 0.0
                                  ? allCloseComparisonOptions(
                                        tolerance.absolute, tolerance.relative, true)
                                  : unitComparisonOptions(expected.type());
                else
                    options.allClose = false;
                options.computeElementwiseStatistics = specialValueConsistency;
                options.computeFrobenius             = false;
                options.maxReportedMismatches        = validation.compareAllClose ? 10 : 0;
                options.selection = OutputSelection::all(IndexOrder::FirstDimensionFastest);
                result.comparison = compare(observed, expected, options);
            }

            if(computeUlp)
            {
                ComparisonOptions options;
                options.allClose                     = false;
                options.computeElementwiseStatistics = false;
                options.computeFrobenius             = false;
                options.computeUlp                   = true;
                options.ulpType                      = expected.type();
                options.maxReportedMismatches        = 0;
                options.selection       = OutputSelection::all(IndexOrder::FirstDimensionFastest);
                result.unitsInLastPlace = compare(observed, expected, options);
            }

            if(validation.compareNorm && expected.elementCount() != 0)
            {
                ComparisonOptions options;
                options.allClose                     = false;
                options.equalNaNs                    = true;
                options.computeElementwiseStatistics = false;
                options.computeFrobenius             = true;
                options.maxReportedMismatches        = 0;
                options.selection = OutputSelection::all(IndexOrder::FirstDimensionFastest);

                const size_t batches
                    = expected.shape().rank() == 3 ? expected.shape()[2] : size_t{1};
                for(size_t batch = 0; batch < batches; ++batch)
                {
                    const ComparisonReport batchReport
                        = compare(batchView(observed, batch), batchView(expected, batch), options);
                    result.relativeFrobeniusError += batchReport.relativeFrobeniusError;
                }
            }

            if(searchAllClose && expected.elementCount() != 0)
            {
                ComparisonOptions options            = allCloseComparisonOptions();
                options.computeElementwiseStatistics = false;
                options.computeFrobenius             = false;
                options.maxReportedMismatches        = 0;
                options.selection = OutputSelection::all(IndexOrder::FirstDimensionFastest);

                constexpr std::array<double, 6> candidates{
                    1e-6,
                    1e-5,
                    1e-4,
                    1e-3,
                    1e-2,
                    1e-1,
                };
                result.allCloseTolerance
                    = findAllCloseTolerance(observed, expected, candidates, candidates, options);
            }
            return result;
        }
    } // namespace

    MatmulValidationSummary validateMatmulOutputs(const MatmulValidationOptions&        options,
                                                  std::span<const MatmulValidationCase> cases)
    {
        MatmulValidationSummary summary;
        size_t                  failedChecks = 0;
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
            std::vector<ComparisonEvidence> outputReports;
            outputReports.reserve(validationCase.outputs.size());
            for(const auto& output : validationCase.outputs)
            {
                outputReports.push_back(
                    compareMatmulOutput(output,
                                        validationCase.allCloseTolerance,
                                        options,
                                        options.compareAllClose || options.compareNorm,
                                        options.searchAllClose,
                                        options.computeUlp));
            }

            if(options.compareAllClose || options.compareNorm)
                for(const auto& report : outputReports)
                    record(report.comparison.nonFiniteMismatches == 0);
            if(options.compareAllClose)
                for(const auto& report : outputReports)
                    record(report.comparison.passed());

            if(options.compareNorm)
            {
                for(const auto& report : outputReports)
                {
                    const double normError = std::abs(report.relativeFrobeniusError);
                    summary.relativeFrobeniusError += normError;
                    if(options.assertNorm)
                        record(norm_check(normError,
                                          validationCase.outputs.front().first.type(),
                                          options.computeType,
                                          options.inputTypeA,
                                          options.inputTypeB));
                }
            }

            if(options.searchAllClose)
            {
                for(size_t outputIndex = 0; outputIndex < outputReports.size(); ++outputIndex)
                {
                    const auto& expected = validationCase.outputs[outputIndex].first;
                    const auto& report   = outputReports[outputIndex];
                    if(expected.elementCount() == 0)
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
                    summary.maximumUlp
                        = std::max(summary.maximumUlp, report.unitsInLastPlace.maximumUlp);
                    ulpSum += report.unitsInLastPlace.sumUlp;
                    ulpCount += report.unitsInLastPlace.ulpCompared;
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
                                                                  validationCase.allCloseTolerance,
                                                                  selectedOptions,
                                                                  false,
                                                                  false,
                                                                  false);
                          record(report.comparison.passed());
                      }
                      if(options.compareNorm)
                      {
                          auto normOptions            = options;
                          normOptions.compareAllClose = false;
                          const auto   report         = compareMatmulOutput(output->norm,
                                                                  validationCase.allCloseTolerance,
                                                                  normOptions,
                                                                  false,
                                                                  false,
                                                                  false);
                          const double normError = std::abs(report.relativeFrobeniusError);
                          summary.relativeFrobeniusError += normError;
                          if(options.assertNorm)
                              record(output->useComputeNormPolicy
                                         ? norm_check(normError,
                                                      output->norm.first.type(),
                                                      options.computeType,
                                                      options.inputTypeA,
                                                      options.inputTypeB)
                                         : norm_check(normError, output->norm.first.type()));
                      }
                  };

            validateSideOutput(validationCase.maximum);
            validateSideOutput(validationCase.auxiliary);
            validateSideOutput(validationCase.bias);
        }

        if(options.searchAllClose && allCloseCompared)
        {
            summary.absoluteTolerance = allCloseFailed ? 1.0 : requiredAbsolute;
            summary.relativeTolerance = allCloseFailed ? 1.0 : requiredRelative;
        }
        if(options.computeUlp && ulpCount != 0)
            summary.averageUlp = ulpSum / ulpCount;
        summary.passed = failedChecks == 0;
        return summary;
    }
} // namespace hipblaslt::host_numerics
