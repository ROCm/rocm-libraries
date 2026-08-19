// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// Transitional mutable generation API. New C++ code includes generation.hpp
// and uses GenerationRecipe.

#include <bit>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <roc/host_validation/generation.hpp>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <variant>
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
    std::optional<ScalarType> sourceType;
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

namespace detail {
struct GenerationRecipeCompatibilityAccess {
    using Component = GenerationRecipe::Component;

    static Component componentFromLegacy(const GenerationPatternSpec& spec) {
        Component result = [&]() {
            switch (spec.pattern) {
                case GenerationPattern::Zero:
                    return Component(Component::ZeroPattern{});
                case GenerationPattern::Constant:
                    return Component(Component::ConstantPattern{{.value = spec.parameter0}});
                case GenerationPattern::CandidateSet:
                    return Component(Component::CandidateSetPattern{{.values = spec.candidates}});
                case GenerationPattern::UniformInteger:
                    return Component(Component::UniformIntegerPattern{
                        {.lower = static_cast<int>(spec.parameter0),
                         .upper = static_cast<int>(spec.parameter1)}});
                case GenerationPattern::AbsoluteUniformInteger:
                    return Component(Component::AbsoluteUniformIntegerPattern{
                        {.lower = static_cast<int>(spec.parameter0),
                         .upper = static_cast<int>(spec.parameter1)}});
                case GenerationPattern::UniformReal:
                    return Component(Component::UniformRealPattern{
                        {.lower = spec.parameter0, .upper = spec.parameter1}});
                case GenerationPattern::Normal:
                    return Component(Component::NormalPattern{
                        {.mean = spec.parameter0, .standardDeviation = spec.parameter1}});
                case GenerationPattern::Sine:
                    return Component(Component::SinePattern{});
                case GenerationPattern::Cosine:
                    return Component(Component::CosinePattern{});
                case GenerationPattern::AbsoluteSine:
                    return Component(Component::AbsoluteSinePattern{});
                case GenerationPattern::AbsoluteCosine:
                    return Component(Component::AbsoluteCosinePattern{});
                case GenerationPattern::SerialIndex:
                    return Component(Component::SerialIndexPattern{});
                case GenerationPattern::SerialDimension:
                    return Component(
                        Component::SerialDimensionPattern{{.dimension = spec.dimension}});
                case GenerationPattern::AffineIndexRemainder:
                    return Component(Component::AffineIndexRemainderPattern{
                        {.dimensionCoefficients = spec.dimensionCoefficients,
                         .offset = spec.affineOffset,
                         .positiveDivisor = spec.remainderDivisor}});
                case GenerationPattern::Identity:
                    return Component(Component::IdentityPattern{});
                case GenerationPattern::CheckerboardUniformInteger:
                    return Component(Component::CheckerboardUniformIntegerPattern{
                        {.lower = static_cast<int>(spec.parameter0),
                         .upper = static_cast<int>(spec.parameter1)}});
                case GenerationPattern::TypeMaximum:
                    return Component(Component::TypeMaximumPattern{});
                case GenerationPattern::TypeLowest:
                    return Component(Component::TypeLowestPattern{});
                case GenerationPattern::TypeDenormalMinimum:
                    return Component(Component::TypeDenormalMinimumPattern{});
                case GenerationPattern::TypeDenormalMaximum:
                    return Component(Component::TypeDenormalMaximumPattern{});
                case GenerationPattern::TypeNaN:
                    return Component(Component::TypeNaNPattern{});
                case GenerationPattern::TypeInfinity:
                    return Component(Component::TypeInfinityPattern{});
                case GenerationPattern::TypeNegativeInfinity:
                    return Component(Component::TypeNegativeInfinityPattern{});
                case GenerationPattern::TypeNegativeZero:
                    return Component(Component::TypeNegativeZeroPattern{});
                case GenerationPattern::UniformTypeRange:
                    return Component(Component::UniformTypeRangePattern{});
                case GenerationPattern::RandomEncodedExponent: {
                    RandomEncodedExponentGenerationParameters parameters{
                        .lowerUnbiasedExponent = static_cast<int>(spec.parameter0),
                        .upperUnbiasedExponent = static_cast<int>(spec.parameter1),
                        .sourceType = std::nullopt,
                    };
                    if (spec.sourceType.has_value()) parameters.sourceType = *spec.sourceType;
                    return Component(
                        Component::RandomEncodedExponentPattern{std::move(parameters)});
                }
                case GenerationPattern::RawConstant:
                    return Component(Component::RawConstantPattern{
                        {.bits = static_cast<uint64_t>(static_cast<int64_t>(spec.parameter0))}});
                case GenerationPattern::UniformRawInteger:
                    return Component(Component::UniformRawIntegerPattern{
                        {.lower = static_cast<int>(spec.parameter0),
                         .upper = static_cast<int>(spec.parameter1)}});
                case GenerationPattern::RandomRawBits:
                    return Component(Component::RandomRawBitsPattern{});
                case GenerationPattern::RawSerialDimension:
                    return Component(
                        Component::RawSerialDimensionPattern{{.dimension = spec.dimension}});
            }
            throw std::invalid_argument("Unsupported GenerationPattern.");
        }();

        switch (spec.transform) {
            case GenerationTransform::None:
                break;
            case GenerationTransform::Absolute:
                result.unaryTransform_ = Component::UnaryTransform::Absolute;
                break;
            case GenerationTransform::Sine:
                result.unaryTransform_ = Component::UnaryTransform::Sine;
                break;
            case GenerationTransform::Cosine:
                result.unaryTransform_ = Component::UnaryTransform::Cosine;
                break;
        }
        result.affineValue_ = {.scale = spec.valueScale, .offset = spec.valueOffset};
        if (!spec.alternatingDimensions.empty()) {
            result.alternatingSign_ = AlternatingSignGenerationParameters{
                .dimensions = spec.alternatingDimensions,
                .negativeWhenOdd = (spec.negativeParity & 1U) != 0,
            };
        }
        return result;
    }

    static GenerationRecipe fromLegacy(const GenerationOptions& options) {
        return GenerationRecipe(
            {.seed = options.seed, .indexOrder = options.indexOrder},
            GenerationRecipe::CartesianPolicy{
                .real = {.component = componentFromLegacy(options.real),
                         .randomDomain = options.real.stream},
                .imaginary = {.component = componentFromLegacy(options.imaginary),
                              .randomDomain = options.imaginary.stream},
            });
    }

    static double legacyRawConstantParameter(uint64_t bits) {
        const int64_t signedBits = std::bit_cast<int64_t>(bits);
        const uint64_t magnitude = signedBits < 0 ? static_cast<uint64_t>(-(signedBits + 1)) + 1
                                                  : static_cast<uint64_t>(signedBits);
        const unsigned significantBits = std::bit_width(magnitude);
        if (significantBits > 53) {
            const unsigned discardedBits = significantBits - 53;
            const uint64_t discardedMask = (uint64_t{1} << discardedBits) - 1;
            if ((magnitude & discardedMask) != 0)
                throw std::invalid_argument(
                    "Raw constant bits are not exactly representable by legacy generation.");
        }
        return static_cast<double>(signedBits);
    }

    static GenerationPatternSpec componentToLegacy(const Component& component,
                                                   uint64_t randomDomain) {
        GenerationPatternSpec result;
        result.stream = randomDomain;
        std::visit(
            [&](const auto& pattern) {
                using Pattern = std::remove_cvref_t<decltype(pattern)>;
                if constexpr (std::is_same_v<Pattern, Component::ZeroPattern>) {
                    result.pattern = GenerationPattern::Zero;
                } else if constexpr (std::is_same_v<Pattern, Component::ConstantPattern>) {
                    result.pattern = GenerationPattern::Constant;
                    result.parameter0 = pattern.parameters.value;
                } else if constexpr (std::is_same_v<Pattern, Component::CandidateSetPattern>) {
                    result.pattern = GenerationPattern::CandidateSet;
                    result.candidates = pattern.parameters.values;
                } else if constexpr (std::is_same_v<Pattern, Component::UniformIntegerPattern>) {
                    result.pattern = GenerationPattern::UniformInteger;
                    result.parameter0 = pattern.parameters.lower;
                    result.parameter1 = pattern.parameters.upper;
                } else if constexpr (std::is_same_v<Pattern,
                                                    Component::AbsoluteUniformIntegerPattern>) {
                    result.pattern = GenerationPattern::AbsoluteUniformInteger;
                    result.parameter0 = pattern.parameters.lower;
                    result.parameter1 = pattern.parameters.upper;
                } else if constexpr (std::is_same_v<Pattern, Component::UniformRealPattern>) {
                    result.pattern = GenerationPattern::UniformReal;
                    result.parameter0 = pattern.parameters.lower;
                    result.parameter1 = pattern.parameters.upper;
                } else if constexpr (std::is_same_v<Pattern, Component::NormalPattern>) {
                    result.pattern = GenerationPattern::Normal;
                    result.parameter0 = pattern.parameters.mean;
                    result.parameter1 = pattern.parameters.standardDeviation;
                } else if constexpr (std::is_same_v<Pattern, Component::SinePattern>) {
                    result.pattern = GenerationPattern::Sine;
                } else if constexpr (std::is_same_v<Pattern, Component::CosinePattern>) {
                    result.pattern = GenerationPattern::Cosine;
                } else if constexpr (std::is_same_v<Pattern, Component::AbsoluteSinePattern>) {
                    result.pattern = GenerationPattern::AbsoluteSine;
                } else if constexpr (std::is_same_v<Pattern, Component::AbsoluteCosinePattern>) {
                    result.pattern = GenerationPattern::AbsoluteCosine;
                } else if constexpr (std::is_same_v<Pattern, Component::SerialIndexPattern>) {
                    result.pattern = GenerationPattern::SerialIndex;
                } else if constexpr (std::is_same_v<Pattern, Component::SerialDimensionPattern>) {
                    result.pattern = GenerationPattern::SerialDimension;
                    result.dimension = pattern.parameters.dimension;
                } else if constexpr (std::is_same_v<Pattern,
                                                    Component::AffineIndexRemainderPattern>) {
                    result.pattern = GenerationPattern::AffineIndexRemainder;
                    result.dimensionCoefficients = pattern.parameters.dimensionCoefficients;
                    result.affineOffset = pattern.parameters.offset;
                    result.remainderDivisor = pattern.parameters.positiveDivisor;
                } else if constexpr (std::is_same_v<Pattern, Component::IdentityPattern>) {
                    result.pattern = GenerationPattern::Identity;
                } else if constexpr (std::is_same_v<Pattern,
                                                    Component::CheckerboardUniformIntegerPattern>) {
                    result.pattern = GenerationPattern::CheckerboardUniformInteger;
                    result.parameter0 = pattern.parameters.lower;
                    result.parameter1 = pattern.parameters.upper;
                } else if constexpr (std::is_same_v<Pattern, Component::TypeMaximumPattern>) {
                    result.pattern = GenerationPattern::TypeMaximum;
                } else if constexpr (std::is_same_v<Pattern, Component::TypeLowestPattern>) {
                    result.pattern = GenerationPattern::TypeLowest;
                } else if constexpr (std::is_same_v<Pattern,
                                                    Component::TypeDenormalMinimumPattern>) {
                    result.pattern = GenerationPattern::TypeDenormalMinimum;
                } else if constexpr (std::is_same_v<Pattern,
                                                    Component::TypeDenormalMaximumPattern>) {
                    result.pattern = GenerationPattern::TypeDenormalMaximum;
                } else if constexpr (std::is_same_v<Pattern, Component::TypeNaNPattern>) {
                    result.pattern = GenerationPattern::TypeNaN;
                } else if constexpr (std::is_same_v<Pattern, Component::TypeInfinityPattern>) {
                    result.pattern = GenerationPattern::TypeInfinity;
                } else if constexpr (std::is_same_v<Pattern,
                                                    Component::TypeNegativeInfinityPattern>) {
                    result.pattern = GenerationPattern::TypeNegativeInfinity;
                } else if constexpr (std::is_same_v<Pattern, Component::TypeNegativeZeroPattern>) {
                    result.pattern = GenerationPattern::TypeNegativeZero;
                } else if constexpr (std::is_same_v<Pattern, Component::UniformTypeRangePattern>) {
                    result.pattern = GenerationPattern::UniformTypeRange;
                } else if constexpr (std::is_same_v<Pattern,
                                                    Component::RandomEncodedExponentPattern>) {
                    result.pattern = GenerationPattern::RandomEncodedExponent;
                    result.parameter0 = pattern.parameters.lowerUnbiasedExponent;
                    result.parameter1 = pattern.parameters.upperUnbiasedExponent;
                    result.sourceType = pattern.parameters.sourceType;
                } else if constexpr (std::is_same_v<Pattern, Component::RawConstantPattern>) {
                    result.pattern = GenerationPattern::RawConstant;
                    result.parameter0 = legacyRawConstantParameter(pattern.parameters.bits);
                } else if constexpr (std::is_same_v<Pattern, Component::UniformRawIntegerPattern>) {
                    result.pattern = GenerationPattern::UniformRawInteger;
                    result.parameter0 = pattern.parameters.lower;
                    result.parameter1 = pattern.parameters.upper;
                } else if constexpr (std::is_same_v<Pattern, Component::RandomRawBitsPattern>) {
                    result.pattern = GenerationPattern::RandomRawBits;
                } else {
                    result.pattern = GenerationPattern::RawSerialDimension;
                    result.dimension = pattern.parameters.dimension;
                }
            },
            component.pattern_);

        switch (component.unaryTransform_) {
            case Component::UnaryTransform::None:
                result.transform = GenerationTransform::None;
                break;
            case Component::UnaryTransform::Absolute:
                result.transform = GenerationTransform::Absolute;
                break;
            case Component::UnaryTransform::Sine:
                result.transform = GenerationTransform::Sine;
                break;
            case Component::UnaryTransform::Cosine:
                result.transform = GenerationTransform::Cosine;
                break;
        }
        result.valueScale = component.affineValue_.scale;
        result.valueOffset = component.affineValue_.offset;
        if (component.alternatingSign_.has_value()) {
            result.alternatingDimensions = component.alternatingSign_->dimensions;
            result.negativeParity =
                static_cast<size_t>(component.alternatingSign_->negativeWhenOdd);
        }
        return result;
    }

    static GenerationOptions toLegacy(const GenerationRecipe& recipe) {
        GenerationOptions result;
        result.seed = recipe.settings_.seed;
        result.indexOrder = recipe.settings_.indexOrder;
        std::visit(
            [&](const auto& policy) {
                using Policy = std::remove_cvref_t<decltype(policy)>;
                if constexpr (std::is_same_v<Policy, GenerationRecipe::RealOnlyPolicy>) {
                    result.real =
                        componentToLegacy(policy.real.component, policy.real.randomDomain);
                    result.imaginary = componentToLegacy(Component(Component::ZeroPattern{}), 0);
                } else if constexpr (std::is_same_v<Policy, GenerationRecipe::ReplicatedPolicy>) {
                    result.real =
                        componentToLegacy(policy.value.component, policy.value.randomDomain);
                    result.imaginary = result.real;
                } else {
                    result.real =
                        componentToLegacy(policy.real.component, policy.real.randomDomain);
                    result.imaginary = componentToLegacy(policy.imaginary.component,
                                                         policy.imaginary.randomDomain);
                }
            },
            recipe.complexPolicy_);
        return result;
    }
};
}  // namespace detail

// Conversion preserves generated values and random domains. Converting a typed
// raw constant fails when parameter0 cannot represent its signed bit pattern.
inline GenerationRecipe generationRecipeFromLegacyOptions(const GenerationOptions& options) {
    return detail::GenerationRecipeCompatibilityAccess::fromLegacy(options);
}

inline GenerationOptions legacyOptionsFromGenerationRecipe(const GenerationRecipe& recipe) {
    return detail::GenerationRecipeCompatibilityAccess::toLegacy(recipe);
}

inline GenerationRunInfo generate(Tensor destination, const GenerationOptions& options) {
    return generate(destination, generationRecipeFromLegacyOptions(options));
}

inline Tensor generate(ScalarType type, Layout layout, const GenerationOptions& options) {
    return generate(type, std::move(layout), generationRecipeFromLegacyOptions(options));
}

inline Tensor generate(ScalarType type, Layout layout, const GenerationOptions& options,
                       const TensorStorageAllocator& allocator) {
    return generate(type, std::move(layout), generationRecipeFromLegacyOptions(options), allocator);
}

inline Tensor generate(ScalarType type, Shape shape, const GenerationOptions& options) {
    return generate(type, std::move(shape), generationRecipeFromLegacyOptions(options));
}

inline Tensor generate(ScalarType type, Shape shape, const GenerationOptions& options,
                       const TensorStorageAllocator& allocator) {
    return generate(type, std::move(shape), generationRecipeFromLegacyOptions(options), allocator);
}

inline GenerationRunInfo generateAt(Tensor destination, size_t logicalIndex,
                                    const GenerationOptions& options) {
    return generateAt(destination, logicalIndex, generationRecipeFromLegacyOptions(options));
}
}  // namespace roc::host_validation
