// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <cmath>
#include <roc/host_numerics/mx.hpp>
#include <stdexcept>
#include <utility>

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

}  // namespace roc::host_numerics
