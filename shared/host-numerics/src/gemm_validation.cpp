// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <roc/host_numerics/gemm_validation.hpp>
#include <stdexcept>
#include <utility>

#include "detail/blocked_gemm.hpp"
#include "detail/reference_gemm.hpp"

namespace roc::host_numerics {
namespace {
void remapComparisonLocations(ComparisonReport& result, const Tensor& observed,
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

void runSelectedReference(const GemmInvocation& request, Tensor& selectedOutput,
                          GemmBackend backend) {
    if (backend == GemmBackend::Automatic) {
        const GemmSupportInfo blockedSupport = detail::queryBlockedGemmSupport(request);
        if (blockedSupport && blockedSupport.preferredForAutomaticExecution)
            return (void)detail::runBlockedGemmToSelectedOutput(request, selectedOutput);
        backend = GemmBackend::Pointwise;
    }

    const GemmSupportInfo support = queryGemmSupport(request, backend);
    if (!support) throw std::invalid_argument(support.reason);

    if (backend == GemmBackend::Blocked)
        (void)detail::runBlockedGemmToSelectedOutput(request, selectedOutput);
    else
        (void)detail::runPointwiseGemmToSelectedOutput(request, selectedOutput);
}
}  // namespace

ComparisonReport validateGemm(const Tensor& a, const Tensor& b, const Tensor& c,
                              const Tensor& observed, const GemmOptions& gemmOptions,
                              const GemmValidationOptions& validationOptions) {
    GemmOptions executionOptions = gemmOptions;
    executionOptions.outputSelection = validationOptions.comparison.selection;
    GemmInvocation request(a, b, c, observed, executionOptions);
    const GemmSupportInfo support = detail::queryGemmSupport(request, validationOptions.backend);
    if (!support) throw std::invalid_argument(support.reason);

    if (validationOptions.comparison.selection.selectsAll()) {
        Tensor expected = referenceGemm(a, b, c, observed.type(), executionOptions,
                                        observed.layout(), validationOptions.backend);
        return compare(observed, expected, validationOptions.comparison);
    }

    const std::vector<size_t> selectedIndices =
        validationOptions.comparison.selection.indices(observed.shape().elementCount());
    Tensor expected(observed.type(), Shape{1, selectedIndices.size()});
    Tensor observedSelected =
        observed
            .copySelectedElements(selectedIndices,
                                  validationOptions.comparison.selection.indexOrder())
            .reshapeSharingStorage(Shape{1, selectedIndices.size()});

    runSelectedReference(request, expected, validationOptions.backend);
    ComparisonOptions compactOptions = validationOptions.comparison;
    compactOptions.selection = OutputSelection::all();
    ComparisonReport comparison = compare(observedSelected, expected, compactOptions);
    remapComparisonLocations(comparison, observed, validationOptions.comparison.selection,
                             selectedIndices);
    return comparison;
}
}  // namespace roc::host_numerics
