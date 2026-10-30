// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <hipblaslt/hostnumerics/MatmulValidation.hpp>
#include <hipblaslt/hostnumerics/norm.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <iomanip>
#include <optional>
#include <sstream>
#include <utility>
#include <vector>

namespace hipblaslt::hostnumerics
{
    using roc::hostnumerics::allCloseComparisonOptions;
    using roc::hostnumerics::compare;
    using roc::hostnumerics::ComparisonOptions;
    using roc::hostnumerics::ComparisonReport;
    using roc::hostnumerics::ComparisonTolerance;
    using roc::hostnumerics::findAllCloseTolerance;

    using roc::hostnumerics::Layout;
    using roc::hostnumerics::ScalarType;
    using roc::hostnumerics::Shape;
    using roc::hostnumerics::Tensor;

    namespace
    {
        struct ComparisonEvidence
        {
            ComparisonReport                     comparison;
            ComparisonReport                     unitsInLastPlace;
            double                               relativeFrobeniusError   = 0.0;
            double                               worstBatchFrobeniusError = 0.0;
            size_t                               worstBatch               = 0;
            std::optional<ComparisonTolerance>   allCloseTolerance;
            std::optional<MatmulAllCloseFailure> allCloseFailure;
        };

        bool worseError(double candidate, double current)
        {
            return (std::isnan(candidate) && !std::isnan(current)) || candidate > current;
        }

        MatmulAllCloseFailure describeAllCloseFailure(const Tensor&                         expected,
                                                      const Tensor&                         observed,
                                                      const roc::hostnumerics::Permutation& order)
        {
            const bool complex = roc::hostnumerics::scalarTypeInfo(expected.type()).category
                                 == roc::hostnumerics::ScalarCategory::Complex;
            const size_t          components = complex ? 2 : 1;
            MatmulAllCloseFailure failure;
            failure.comparedCount                = expected.elementCount() * components;
            auto options                         = allCloseComparisonOptions(0.1, 0.1, false);
            options.computeElementwiseStatistics = false;
            options.computeFrobenius             = false;

            // Bound diagnostic storage even when every component of a large output fails.
            constexpr size_t chunkSize = 2048;
            for(size_t first = 0; first < expected.elementCount();)
            {
                const size_t        count = std::min(chunkSize, expected.elementCount() - first);
                std::vector<double> reference(count * components), result(count * components);
                expected.shape().forEachCoordinate(
                    roc::hostnumerics::IndexRange{first, first + count, 1},
                    order,
                    [&](size_t index, std::span<const size_t> coordinates) {
                        const size_t offset = (index - first) * components;
                        if(complex)
                        {
                            const auto want   = expected.loadAs<std::complex<double>>(coordinates);
                            const auto got    = observed.loadAs<std::complex<double>>(coordinates);
                            reference[offset] = want.real();
                            reference[offset + 1] = want.imag();
                            result[offset]        = got.real();
                            result[offset + 1]    = got.imag();
                        }
                        else
                        {
                            reference[offset] = expected.loadAs<double>(coordinates);
                            result[offset]    = observed.loadAs<double>(coordinates);
                        }
                    });
                options.maxReportedMismatches = reference.size();
                // The benchmark's relative tolerance is scaled by the GPU result.
                const auto report = compare(Tensor::copyNativeStorage(std::span(reference)),
                                            Tensor::copyNativeStorage(std::span(result)),
                                            options);
                for(const auto& sample : report.reportedMismatches)
                {
                    const double error = std::abs(sample.observed - sample.expected);
                    if(failure.badCount == 0 || worseError(error, failure.error))
                    {
                        const auto coordinates = expected.shape().coordinates(
                            first + sample.index / components, order);
                        failure.row       = coordinates.empty() ? 0 : coordinates[0];
                        failure.column    = coordinates.size() < 2 ? 0 : coordinates[1];
                        failure.batch     = coordinates.size() < 3 ? 0 : coordinates[2];
                        failure.imaginary = sample.index % components != 0;
                        failure.observed  = sample.expected;
                        failure.expected  = sample.observed;
                        failure.error     = error;
                    }
                    ++failure.badCount;
                }
                first += count;
            }
            return failure;
        }

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
            const auto                  offset = tensor.layout().elementOffset(origin);
            if(!offset)
                throw std::runtime_error(offset.error().message());
            return tensor.shareStorageWithLayout(
                Layout(Shape{tensor.shape()[0], tensor.shape()[1]},
                       {tensor.layout().stride(0), tensor.layout().stride(1)},
                       *offset));
        }

        ComparisonEvidence compareMatmulOutput(const MatmulValidationCase::TensorPair& tensors,
                                               const ComparisonTolerance&              tolerance,
                                               const MatmulValidationOptions&          validation,
                                               bool specialValueConsistency,
                                               bool searchAllClose,
                                               bool computeUlp)
        {
            const auto& [expected, observed] = tensors;
            ComparisonEvidence  result;
            const auto order = roc::hostnumerics::Permutation::reverse(expected.shape().rank());

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

                result.comparison
                    = compare(observed,
                              expected,
                              options,
                              roc::hostnumerics::IndexRange::all(observed.elementCount()),
                              order);
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

                result.unitsInLastPlace
                    = compare(observed,
                              expected,
                              options,
                              roc::hostnumerics::IndexRange::all(observed.elementCount()),
                              order);
            }

            if(validation.compareNorm && expected.elementCount() != 0)
            {
                ComparisonOptions options;
                options.allClose                     = false;
                options.equalNaNs                    = true;
                options.computeElementwiseStatistics = false;
                options.computeFrobenius             = true;
                options.maxReportedMismatches        = 0;

                const size_t batches
                    = expected.shape().rank() == 3 ? expected.shape()[2] : size_t{1};
                for(size_t batch = 0; batch < batches; ++batch)
                {
                    const Tensor        batchExpected = batchView(expected, batch);
                    const auto batchOrder = roc::hostnumerics::Permutation::reverse(batchExpected.shape().rank());

                    const ComparisonReport batchReport
                        = compare(batchView(observed, batch),
                                  batchExpected,
                                  options,
                                  roc::hostnumerics::IndexRange::all(batchExpected.elementCount()),
                                  batchOrder);
                    const double batchError = std::abs(batchReport.relativeFrobeniusError);
                    result.relativeFrobeniusError += batchReport.relativeFrobeniusError;
                    if(worseError(batchError, result.worstBatchFrobeniusError))
                    {
                        result.worstBatchFrobeniusError = batchError;
                        result.worstBatch               = batch;
                    }
                }
            }

            if(searchAllClose && expected.elementCount() != 0)
            {
                ComparisonOptions options = allCloseComparisonOptions(1e-8, 1e-5, false);
                options.complexComparisonMode
                    = roc::hostnumerics::ComplexComparisonMode::Componentwise;
                options.computeElementwiseStatistics = false;
                options.computeFrobenius             = false;
                options.maxReportedMismatches        = 0;

                constexpr std::array<double, 6> candidates{
                    1e-6,
                    1e-5,
                    1e-4,
                    1e-3,
                    1e-2,
                    1e-1,
                };
                result.allCloseTolerance = findAllCloseTolerance(
                    expected,
                    observed,
                    candidates,
                    candidates,
                    options,
                    roc::hostnumerics::IndexRange::all(observed.elementCount()),
                    order);
                if(!result.allCloseTolerance)
                    result.allCloseFailure = describeAllCloseFailure(expected, observed, order);
            }
            return result;
        }
    } // namespace

    MatmulValidationSummary validateMatmulOutputs(const MatmulValidationOptions&        options,
                                                  std::span<const MatmulValidationCase> cases)
    {
        MatmulValidationSummary summary;
        size_t                  failedChecks = 0;
        const auto record   = [&](bool passed) { failedChecks += static_cast<size_t>(!passed); };
        double     ulpSum   = 0.0;
        size_t     ulpCount = 0;
        // Each output is searched independently. The componentwise maxima form
        // a tolerance that accepts every successful output; any failed search
        // dominates the aggregate regardless of output order.
        bool   allCloseCompared   = false;
        bool   allCloseFailed     = false;
        double requiredAbsolute   = 0.0;
        double requiredRelative   = 0.0;
        size_t allCloseComponents = 0;
        // All-close already rejects every non-finite mismatch. Norm-only validation needs the
        // separate consistency evidence because a Frobenius reduction cannot represent it.
        const bool needsSpecialValueStatistics = options.compareNorm && !options.compareAllClose;

        for(size_t caseIndex = 0; caseIndex < cases.size(); ++caseIndex)
        {
            const auto&                     validationCase = cases[caseIndex];
            std::vector<ComparisonEvidence> outputReports;
            outputReports.reserve(validationCase.outputs.size());
            for(const auto& output : validationCase.outputs)
            {
                outputReports.push_back(compareMatmulOutput(output,
                                                            validationCase.allCloseTolerance,
                                                            options,
                                                            needsSpecialValueStatistics,
                                                            options.searchAllClose,
                                                            options.computeUlp));
            }

            if(needsSpecialValueStatistics)
                for(const auto& report : outputReports)
                    record(report.comparison.nonFiniteMismatches == 0);
            if(options.compareAllClose)
                for(const auto& report : outputReports)
                    record(report.comparison.passed());

            if(options.compareNorm)
            {
                double worstError = 0.0;
                size_t worstBatch = 0;
                for(size_t outputIndex = 0; outputIndex < outputReports.size(); ++outputIndex)
                {
                    const auto&  report    = outputReports[outputIndex];
                    const double normError = std::abs(report.relativeFrobeniusError);
                    summary.relativeFrobeniusError += normError;
                    if(options.assertNorm)
                        record(norm_check(normError,
                                          validationCase.outputs.front().first.type(),
                                          options.computeType,
                                          options.inputTypeA,
                                          options.inputTypeB));
                    if(worseError(report.worstBatchFrobeniusError, worstError))
                    {
                        worstError = report.worstBatchFrobeniusError;
                        // Separate outputs are the batches of a pointer-array case.
                        worstBatch = validationCase.outputs[outputIndex].first.shape().rank() == 3
                                         ? report.worstBatch
                                         : outputIndex;
                    }
                }
                double limit = 0.0;
                try
                {
                    limit = norm_tolerance(validationCase.outputs.front().first.type(),
                                           options.computeType,
                                           options.inputTypeA,
                                           options.inputTypeB);
                }
                catch(const std::runtime_error&)
                {
                }
                using Kind = MatmulOutputCheckFailure::Kind;
                if(!(limit > 0.0))
                    summary.outputCheckFailures.push_back(
                        {.kind = Kind::NormUnavailable, .gemm = caseIndex});
                else if(!(worstError < limit))
                    summary.outputCheckFailures.push_back({.kind  = Kind::Norm,
                                                           .gemm  = caseIndex,
                                                           .batch = worstBatch,
                                                           .error = worstError,
                                                           .limit = limit});
            }

            if(options.searchAllClose)
            {
                bool caseFailed = false;
                for(size_t outputIndex = 0; outputIndex < outputReports.size(); ++outputIndex)
                {
                    const auto& expected = validationCase.outputs[outputIndex].first;
                    const auto& report   = outputReports[outputIndex];
                    if(expected.elementCount() == 0)
                        continue;

                    allCloseCompared = true;
                    allCloseComponents
                        += expected.elementCount()
                           * (roc::hostnumerics::scalarTypeInfo(expected.type()).category
                                      == roc::hostnumerics::ScalarCategory::Complex
                                  ? 2
                                  : 1);
                    if(report.allCloseTolerance)
                    {
                        requiredAbsolute
                            = std::max(requiredAbsolute, report.allCloseTolerance->absolute);
                        requiredRelative
                            = std::max(requiredRelative, report.allCloseTolerance->relative);
                    }
                    else
                    {
                        allCloseFailed = true;
                        caseFailed     = true;
                        auto failure   = *report.allCloseFailure;
                        failure.gemm   = caseIndex;
                        if(expected.shape().rank() != 3)
                            failure.batch = outputIndex;
                        if(!summary.allCloseFailure)
                            summary.allCloseFailure = failure;
                        else
                        {
                            const size_t totalBad
                                = summary.allCloseFailure->badCount + failure.badCount;
                            if(worseError(failure.error, summary.allCloseFailure->error))
                                summary.allCloseFailure = failure;
                            summary.allCloseFailure->badCount = totalBad;
                        }
                    }
                }
                if(caseFailed)
                    summary.outputCheckFailures.push_back(
                        {.kind = MatmulOutputCheckFailure::Kind::AllClose, .gemm = caseIndex});
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
                          const double normError      = std::abs(report.relativeFrobeniusError);
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
            if(summary.allCloseFailure)
                summary.allCloseFailure->comparedCount = allCloseComponents;
        }
        if(options.computeUlp && ulpCount != 0)
            summary.averageUlp = ulpSum / ulpCount;
        summary.passed = failedChecks == 0;
        return summary;
    }

    std::string formatAllCloseFailure(const MatmulAllCloseFailure& failure)
    {
        std::ostringstream output;
        output << std::setprecision(9) << failure.badCount << " of " << failure.comparedCount
               << " elements are outside atol 0.1 + rtol 0.1 * |result|; the worst is GEMM "
               << failure.gemm << ", batch " << failure.batch << ", row " << failure.row
               << ", column " << failure.column << (failure.imaginary ? " (imaginary part)" : "")
               << ": result " << failure.observed << ", reference " << failure.expected;
        return output.str();
    }
} // namespace hipblaslt::hostnumerics
