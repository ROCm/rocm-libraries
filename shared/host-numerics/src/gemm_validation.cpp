// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <roc/host_numerics/gemm_validation.hpp>

#include <stdexcept>
#include <utility>

#include "detail/blocked_gemm.hpp"
#include "detail/reference_gemm.hpp"

namespace roc::host_numerics {
namespace {
void remapComparisonLocations(ComparisonResult& result, const Tensor& observed,
                              const OutputSelection& selection,
                              const std::vector<size_t>& selectedIndices) {
    const auto remap = [&](Mismatch& mismatch) {
        const size_t logicalIndex = selectedIndices.at(mismatch.index);
        const auto coordinates = observed.shape().coordinates(logicalIndex, selection.indexOrder());
        const ptrdiff_t offset = observed.layout().elementOffset(coordinates);
        mismatch.index = logicalIndex;
        mismatch.coordinates = coordinates;
        mismatch.observedOffset = offset;
        mismatch.expectedOffset = offset;
    };
    for (Mismatch& mismatch : result.reportedMismatches) remap(mismatch);
    for (Mismatch& comparison : result.reportedComparisons) remap(comparison);
}

GemmRunInfo runSelectedReference(const GemmRequest& request, Tensor& selectedOutput,
                                 GemmBackend backend) {
    std::optional<std::string> fallbackReason;
    if (backend == GemmBackend::Automatic) {
        const GemmSupportInfo blockedSupport = detail::queryBlockedGemmSupport(request);
        if (blockedSupport && blockedSupport.preferredForAutomaticExecution)
            return detail::runBlockedGemmToSelectedOutput(request, selectedOutput);
        if (!blockedSupport) fallbackReason = blockedSupport.reason;
        backend = GemmBackend::Pointwise;
    }

    const GemmSupportInfo support = queryGemmSupport(request, backend);
    if (!support) throw std::invalid_argument(support.reason);

    GemmRunInfo runInfo;
    if (backend == GemmBackend::Blocked)
        runInfo = detail::runBlockedGemmToSelectedOutput(request, selectedOutput);
    else
        runInfo = detail::runPointwiseGemmToSelectedOutput(request, selectedOutput);
    runInfo.fallbackReason = std::move(fallbackReason);
    return runInfo;
}
}  // namespace

GemmValidationResult validateGemm(const GemmProblem& problem, const Tensor& observed,
                                  const GemmValidationOptions& options) {
    GemmRequest request(problem, observed, options.comparison.selection);
    const GemmSupportInfo support = queryGemmSupport(request, options.backend);
    if (!support) throw std::invalid_argument(support.reason);

    if (options.comparison.selection.selectsAll()) {
        GemmOutputOptions output;
        output.layout = observed.layout();
        output.selection = options.comparison.selection;
        GemmResult expected = referenceGemm(problem, output, options.backend);
        return {
            .reference = std::move(expected.runInfo),
            .comparison = compare(observed, expected.output, options.comparison),
        };
    }

    const std::vector<size_t> selectedIndices
        = options.comparison.selection.indices(observed.shape().elementCount());
    Tensor expected(problem.outputType, Shape{1, selectedIndices.size()});
    Tensor observedSelected = observed
                                  .copySelectedElements(
                                      selectedIndices, options.comparison.selection.indexOrder())
                                  .reshapeSharingStorage(Shape{1, selectedIndices.size()});

    GemmRunInfo runInfo
        = runSelectedReference(request, expected, options.backend);
    ComparisonOptions compactOptions = options.comparison;
    compactOptions.selection = OutputSelection::all();
    ComparisonResult comparison = compare(observedSelected, expected, compactOptions);
    remapComparisonLocations(
        comparison, observed, options.comparison.selection, selectedIndices);
    return {
        .reference = std::move(runInfo),
        .comparison = std::move(comparison),
    };
}
}  // namespace roc::host_numerics
