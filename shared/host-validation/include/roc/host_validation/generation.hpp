// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <cstdint>
#include <roc/host_validation/detail/tensor_views.hpp>
#include <roc/host_validation/generation_primitives.hpp>
#include <span>
#include <type_traits>
#include <vector>

namespace roc::host_validation {
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

enum class GenerationTransform {
    None,
    Absolute,
    Sine,
    Cosine,
};

struct GenerationPatternSpec {
    GenerationPattern pattern = GenerationPattern::Zero;
    double parameter0 = 0.0;
    double parameter1 = 1.0;
    double valueScale = 1.0;
    double valueOffset = 0.0;
    uint64_t stream = 0;
    size_t dimension = 0;
    ScalarType sourceType = ScalarType::Count;
    GenerationTransform transform = GenerationTransform::None;
    std::vector<int64_t> dimensionCoefficients;
    int64_t affineOffset = 0;
    int64_t remainderDivisor = 1;
    std::vector<double> candidates;
    std::vector<size_t> alternatingDimensions;
    size_t negativeParity = 0;
};

struct GenerationOptions {
    uint64_t seed = 0;
    LogicalIndexOrder indexOrder = LogicalIndexOrder::FirstDimensionFastest;
    GenerationPatternSpec real;
    GenerationPatternSpec imaginary;
};

struct GenerationRunInfo {
    size_t elementsGenerated = 0;
};

template <typename T, typename Generator>
void generate(MatrixView<T> destination, Generator&& generator) {
    for (size_t column = 0; column < destination.columns(); ++column) {
        for (size_t row = 0; row < destination.rows(); ++row)
            destination(row, column) = static_cast<T>(generator(row, column));
    }
}

template <typename Generator>
    requires(std::is_invocable_v<Generator&, std::span<const size_t>, size_t> ||
             std::is_invocable_v<Generator&, std::span<const size_t>>)
void generate(MutableTensorView destination, Generator&& generator) {
    detail::forEachIndex(
        destination.shape(), [&](std::span<const size_t> indices, size_t linearIndex) {
            if constexpr (std::is_invocable_v<Generator&, std::span<const size_t>, size_t>)
                destination.storeFrom(indices, generator(indices, linearIndex));
            else
                destination.storeFrom(indices, generator(indices));
        });
}

GenerationRunInfo generate(MutableTensorView destination, const GenerationOptions& options);
GenerationRunInfo generateAt(MutableTensorView destination, size_t logicalIndex,
                             const GenerationOptions& options);
}  // namespace roc::host_validation
