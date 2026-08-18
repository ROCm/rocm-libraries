// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <cstdint>
#include <roc/host_validation/tensor.hpp>

namespace roc::host_validation {
// Selects the source-value recipe for packed MX data. Scale selection uses the
// separate MxScaleGenerationMode policy.
enum class MxGenerationMode {
    Bounded,
    BoundedAlternatingSign,
    Unbounded,
    Identity,
    Ones,
    Zeros,
    Sequential,
    RowIndex,
    ColumnIndex,
    Checkerboard,
    ScaledDiagonal,
    Twos,
    NegativeOnes,
    Maximum,
    DenormalMinimum,
    DenormalMaximum,
    NaN,
    Infinity,
    Trigonometric,
    Normal,
    UniformInteger,
};

struct MxGenerationRecipe {
    // parameter0 and parameter1 are mode-specific operands. Examples include
    // bounded/integer endpoints and normal-distribution mean/deviation.
    MxGenerationMode mode = MxGenerationMode::Bounded;
    double parameter0 = -1.0;
    double parameter1 = 1.0;
};

// Selects how block scales are produced, independently of the data recipe.
// Derived chooses a scale from each block's generated source values. Every
// other mode writes one scale-specific constant to all blocks and does not
// change the generated data values.
enum class MxScaleGenerationMode {
    Derived,
    // Smallest encoded finite scale: numerical zero where the scale format has
    // one, otherwise its minimum positive value (for example E8M0 raw zero).
    Minimum,
    One,
    Two,
    Maximum,
    NaN,
};

// Describes a rank-two, block-scaled MX tensor in its natural host layout.
// Data generation and scale selection are independent: data controls source
// values, while scale controls only how each block scale is selected.
struct MxGenerationProblem {
    // Retained for compatibility with the pre-component MX generator.
    static constexpr uint32_t defaultSeed = 1713573849U;

    // Packed data encoding and per-block scale encoding.
    ScalarType dataType = ScalarType::Float4E2M1;
    ScalarType scaleType = ScalarType::E8M0;

    // Logical data shape. A zero leading dimension selects shape[0], otherwise
    // data uses strides {1, leadingDimension}.
    Shape shape;
    ptrdiff_t leadingDimension = 0;

    // Scales group blockSize consecutive elements along blockAxis. The other
    // axis is the free coordinate identifying independent block sequences.
    size_t blockAxis = 0;
    size_t blockSize = 32;

    // Source values to quantize into dataType.
    MxGenerationRecipe data;

    // Scale-selection policy. Derived computes scales from data; constant modes
    // use the same encoded scale for every block.
    MxScaleGenerationMode scale = MxScaleGenerationMode::Derived;

    // Seed used by stochastic data and scale choices.
    uint32_t seed = defaultSeed;
};

struct MxGenerationResult {
    // Packed data with the requested leading dimension.
    Tensor data;

    // One-dimensional scales in natural block order.
    Tensor scales;

    // UInt32 tensor mapping every logical data element to its entry in scales.
    Tensor scaleIndices;

    // Contiguous Float32 tensor equal to decoded data * selected scale.
    Tensor reference;
};

MxGenerationResult generateMx(const MxGenerationProblem& problem);
}  // namespace roc::host_validation
