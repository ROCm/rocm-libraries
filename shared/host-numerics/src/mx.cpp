// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <roc/host_numerics/mx.hpp>

#include "detail/mx_generation.hpp"

namespace roc::host_numerics {
MxDataGeneration::MxDataGeneration(GenerationRecipe recipe, MxDataQuantization quantization,
                                   std::optional<MxRepresentedValueRange> representedValueRange)
    : recipe_(std::move(recipe)),
      quantization_(quantization),
      representedValueRange_(representedValueRange) {}

MxDataGeneration MxDataGeneration::quantize(GenerationRecipe recipe) {
    return MxDataGeneration(std::move(recipe), MxDataQuantization::Nearest, std::nullopt);
}

MxDataGeneration MxDataGeneration::preserveRange(GenerationRecipe recipe,
                                                 MxRepresentedValueRange representedValueRange) {
    if (!std::isfinite(representedValueRange.lower) ||
        !std::isfinite(representedValueRange.upper) ||
        !(representedValueRange.lower < representedValueRange.upper))
        throw std::invalid_argument(
            "Range-preserving MX generation requires finite lower < upper.");
    return MxDataGeneration(std::move(recipe), MxDataQuantization::PreserveRange,
                            representedValueRange);
}

MxDataGeneration MxDataGeneration::preserveGeneratedEncoding(GenerationRecipe recipe) {
    return MxDataGeneration(std::move(recipe), MxDataQuantization::PreserveGeneratedEncoding,
                            std::nullopt);
}

const GenerationRecipe& MxDataGeneration::recipe() const noexcept {
    return recipe_;
}

MxDataQuantization MxDataGeneration::quantization() const noexcept {
    return quantization_;
}

const std::optional<MxRepresentedValueRange>& MxDataGeneration::representedValueRange()
    const noexcept {
    return representedValueRange_;
}

MxDataGeneration MxDataGeneration::withSeed(uint64_t seed) const {
    return MxDataGeneration(recipe_.withSeed(seed), quantization_, representedValueRange_);
}

MxTensor generateMx(Shape shape, MxDataGeneration generation, const MxGenerationOptions& options) {
    const MxGenerationInvocation problem(std::move(shape), std::move(generation), options);
    validateInvocation(problem);
    const size_t rows = problem.shape[0];
    const size_t columns = problem.shape[1];
    const size_t leadingDimension =
        problem.leadingDimension == 0 ? rows : static_cast<size_t>(problem.leadingDimension);
    const size_t physicalElementCount =
        detail::checkedMultiply(leadingDimension, columns, "MX physical element count overflow.");
    const size_t logicalElementCount =
        detail::checkedMultiply(rows, columns, "MX logical element count overflow.");
    if (leadingDimension > static_cast<size_t>(std::numeric_limits<ptrdiff_t>::max()))
        throw std::overflow_error("MX leading dimension exceeds ptrdiff_t.");

    const ScaleBlocking blocking(problem);
    if (blocking.scaleCount > std::numeric_limits<uint32_t>::max())
        throw std::overflow_error("MX scale count exceeds UInt32 scale-index storage.");
    std::vector<uint8_t> dataRawValues(physicalElementCount, 0);
    std::vector<uint8_t> scaleRawValues(blocking.scaleCount, 0);
    std::vector<uint32_t> scaleIndexValues(logicalElementCount, 0);
    std::vector<float> referenceValues(logicalElementCount, 0.0f);
    const int threadCount =
        detail::operationThreadCount(std::max(logicalElementCount, physicalElementCount));

    if (problem.data.quantization() == MxDataQuantization::PreserveGeneratedEncoding)
        generateUnbounded(problem, blocking, dataRawValues, scaleRawValues, scaleIndexValues,
                          referenceValues, threadCount);
    else
        generateQuantized(problem, blocking, dataRawValues, scaleRawValues, scaleIndexValues,
                          referenceValues, threadCount);

    std::vector<std::byte> dataStorage =
        packRawValues(dataRawValues, scalarTypeInfo(problem.dataType).storageBits, threadCount);
    Tensor data = Tensor::takeOwnershipOfEncodedBackingStorage(
        problem.dataType, Layout(problem.shape, {1, static_cast<ptrdiff_t>(leadingDimension)}),
        std::move(dataStorage));
    std::vector<std::byte> scaleStorage(scaleRawValues.size());
    std::memcpy(scaleStorage.data(), scaleRawValues.data(), scaleRawValues.size());
    Tensor scales = Tensor::takeOwnershipOfEncodedBackingStorage(
        problem.scaleType, Layout::contiguousLastDimensionFastest(blocking.naturalScaleShape()),
        std::move(scaleStorage));
    Tensor scaleIndices =
        Tensor::copyNativeStorage(Layout(problem.shape, {1, static_cast<ptrdiff_t>(rows)}),
                                  std::span<const uint32_t>(scaleIndexValues));
    Tensor reference =
        Tensor::copyNativeStorage(Layout(problem.shape, {1, static_cast<ptrdiff_t>(rows)}),
                                  std::span<const float>(referenceValues));
    return {std::move(data), std::move(scales), std::move(scaleIndices), std::move(reference)};
}
}  // namespace roc::host_numerics
