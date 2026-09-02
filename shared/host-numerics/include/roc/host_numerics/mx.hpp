// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <roc/host_numerics/generation.hpp>

namespace roc::host_numerics {
enum class MxDataQuantization {
    Nearest,
    PreserveRange,
    PreserveGeneratedEncoding,
};

struct MxRepresentedValueRange {
    double lower = -1.0;
    double upper = 1.0;
};

// Couples a general source-value recipe to the MX-specific quantization
// contract used after per-block scale selection.
class MxDataGeneration {
   public:
    [[nodiscard]] static MxDataGeneration quantize(GenerationRecipe recipe);
    [[nodiscard]] static MxDataGeneration preserveRange(
        GenerationRecipe recipe, MxRepresentedValueRange representedValueRange);
    [[nodiscard]] static MxDataGeneration preserveGeneratedEncoding(GenerationRecipe recipe);

    [[nodiscard]] const GenerationRecipe& recipe() const noexcept;
    [[nodiscard]] MxDataQuantization quantization() const noexcept;
    [[nodiscard]] const std::optional<MxRepresentedValueRange>& representedValueRange()
        const noexcept;
    [[nodiscard]] MxDataGeneration withSeed(uint64_t seed) const;

   private:
    MxDataGeneration(GenerationRecipe recipe, MxDataQuantization quantization,
                     std::optional<MxRepresentedValueRange> representedValueRange);

    GenerationRecipe recipe_;
    MxDataQuantization quantization_;
    std::optional<MxRepresentedValueRange> representedValueRange_;
};

// Selects how block scales are produced, independently of the data recipe.
// Derived chooses a scale from each block's generated source values. Every
// other mode writes one scale-specific constant to all blocks and does not
// change the generated data values.
enum class MxScaleGenerationMode {
    Derived,
    RandomFinite,
    // Smallest encoded finite scale: numerical zero where the scale format has
    // one, otherwise its minimum positive value (for example E8M0 raw zero).
    Minimum,
    One,
    Two,
    Maximum,
    NaN,
};

// Storage and scale policy for a rank-two, block-scaled MX tensor. Data
// generation is supplied separately so scale selection cannot alter its stream.
struct MxGenerationOptions {
    // Packed data encoding and per-block scale encoding.
    ScalarType dataType = ScalarType::Float4E2M1;
    ScalarType scaleType = ScalarType::E8M0;

    // A zero leading dimension selects shape[0]; otherwise data uses strides
    // {1, leadingDimension}.
    ptrdiff_t leadingDimension = 0;

    // Scales group blockSize consecutive elements along blockAxis. The other
    // axis is the free coordinate identifying independent block sequences.
    size_t blockAxis = 0;
    size_t blockSize = 32;

    // Scale-selection policy. Derived computes scales from data; constant modes
    // use the same encoded scale for every block.
    MxScaleGenerationMode scale = MxScaleGenerationMode::Derived;
};

struct MxTensor {
    // Packed data with the requested leading dimension.
    Tensor data;

    // Rank-two scales in natural row-major [slow, fast] block order.
    Tensor scales;

    // UInt32 tensor mapping every logical data element to its entry in scales.
    Tensor scaleIndices;

    // Contiguous Float32 tensor equal to decoded data * selected scale.
    Tensor reference;
};

MxTensor generateMx(Shape shape, MxDataGeneration generation,
                    const MxGenerationOptions& options = {});
}  // namespace roc::host_numerics
