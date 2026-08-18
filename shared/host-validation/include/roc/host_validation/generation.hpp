// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <cstdint>
#include <roc/host_validation/generation_primitives.hpp>
#include <roc/host_validation/tensor.hpp>
#include <span>
#include <type_traits>
#include <vector>

namespace roc::host_validation {
// Selects the base value recipe for one numerical component. Numerical recipes
// pass through the destination scalar codec; Raw* recipes instead write scalar
// encoding bits directly and do not support complex destinations.
enum class GenerationPattern {
    Zero,
    Constant,
    CandidateSet,
    UniformInteger,
    AbsoluteUniformInteger,
    UniformReal,
    Normal,
    Sine,
    Cosine,
    AbsoluteSine,
    AbsoluteCosine,
    SerialIndex,
    SerialDimension,
    AffineIndexRemainder,
    Identity,
    CheckerboardUniformInteger,
    TypeMaximum,
    TypeLowest,
    TypeDenormalMinimum,
    TypeDenormalMaximum,
    TypeNaN,
    TypeInfinity,
    TypeNegativeInfinity,
    TypeNegativeZero,
    UniformTypeRange,
    RandomEncodedExponent,
    RawConstant,
    UniformRawInteger,
    RandomRawBits,
    RawSerialDimension,
};

// Optional unary processing applied to a numerical base value before its
// affine value scale and offset.
enum class GenerationTransform {
    None,
    Absolute,
    Sine,
    Cosine,
};

struct GenerationPatternSpec {
    // Base recipe used to produce this component.
    GenerationPattern pattern = GenerationPattern::Zero;

    // Pattern-specific operands, such as a constant, interval bounds, or a
    // normal distribution's mean and standard deviation.
    double parameter0 = 0.0;
    double parameter1 = 1.0;

    // Numerical values are postprocessed as
    // transform(baseValue) * valueScale + valueOffset.
    double valueScale = 1.0;
    double valueOffset = 0.0;

    // Counter-random stream combined with GenerationOptions::seed and the
    // logical element index. Distinct streams give independent sequences.
    uint64_t stream = 0;

    // Tensor axis used by SerialDimension and RawSerialDimension.
    size_t dimension = 0;

    // Encoding used by RandomEncodedExponent. Count selects the destination's
    // real component type.
    ScalarType sourceType = ScalarType::Count;

    // Unary transform applied before valueScale and valueOffset.
    GenerationTransform transform = GenerationTransform::None;

    // AffineIndexRemainder computes
    // (affineOffset + sum(dimensionCoefficients[d] * index[d]))
    // % remainderDivisor before the common numerical postprocessing.
    std::vector<int64_t> dimensionCoefficients;
    int64_t affineOffset = 0;
    int64_t remainderDivisor = 1;

    // Values sampled by CandidateSet.
    std::vector<double> candidates;

    // After numerical postprocessing, negate the value when the XOR parity of
    // these coordinates equals the low bit of negativeParity.
    std::vector<size_t> alternatingDimensions;
    size_t negativeParity = 0;
};

struct GenerationOptions {
    // Shared seed for both components; each component can select its own stream.
    uint64_t seed = 0;

    // Defines coordinate-to-logical-index mapping for indexed recipes and
    // generateAt.
    LogicalIndexOrder indexOrder = LogicalIndexOrder::FirstDimensionFastest;

    // real is always used. imaginary is used only for complex destinations.
    GenerationPatternSpec real;
    GenerationPatternSpec imaginary;
};

struct GenerationRunInfo {
    // Number of logical destination elements written by the operation.
    size_t elementsGenerated = 0;
};

template <typename Generator>
    requires(std::is_invocable_v<Generator&, std::span<const size_t>, size_t> ||
             std::is_invocable_v<Generator&, std::span<const size_t>>)
void generate(Tensor destination, Generator&& generator) {
    detail::forEachIndex(
        destination.shape(), [&](std::span<const size_t> indices, size_t linearIndex) {
            if constexpr (std::is_invocable_v<Generator&, std::span<const size_t>, size_t>)
                destination.storeFrom(indices, generator(indices, linearIndex));
            else
                destination.storeFrom(indices, generator(indices));
        });
}

GenerationRunInfo generate(Tensor destination, const GenerationOptions& options);
Tensor generate(ScalarType type, Layout layout, const GenerationOptions& options);
Tensor generate(ScalarType type, Layout layout, const GenerationOptions& options,
                const TensorStorageAllocator& allocator);
Tensor generate(ScalarType type, Shape shape, const GenerationOptions& options);
Tensor generate(ScalarType type, Shape shape, const GenerationOptions& options,
                const TensorStorageAllocator& allocator);
GenerationRunInfo generateAt(Tensor destination, size_t logicalIndex,
                             const GenerationOptions& options);
}  // namespace roc::host_validation
