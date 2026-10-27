// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "detail/generation_recipe_access.hpp"

#include <array>
#include <bit>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <limits>
#include <numeric>
#include <roc/hostnumerics/generation.hpp>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include "detail/generation_primitives.hpp"
#include "detail/generation_values.hpp"
#include "detail/layout_iteration.hpp"
#include "detail/prepared_scalar_storage.hpp"
#include "detail/threading.hpp"

namespace roc::hostnumerics::detail {
struct GenerationRecipeAccess {
    using Component = GenerationRecipe::Component;

    template <typename Pattern>
    static constexpr bool isRawPattern =
        std::is_same_v<Pattern, Component::RawConstantPattern> ||
        std::is_same_v<Pattern, Component::UniformRawIntegerPattern> ||
        std::is_same_v<Pattern, Component::UniformFiniteEncodedValuePattern> ||
        std::is_same_v<Pattern, Component::RandomRawBitsPattern> ||
        std::is_same_v<Pattern, Component::RawSerialDimensionPattern>;

    static bool isRawComponent(const Component& component) {
        return std::visit(
            [](const auto& pattern) {
                return isRawPattern<std::remove_cvref_t<decltype(pattern)>>;
            },
            component.pattern_);
    }

    template <bool Checked = true, typename Pattern>
    static uint64_t rawGenerationValueKnown(const Pattern& pattern, uint64_t seed, uint64_t domain,
                                            std::span<const size_t> indices, const Shape& shape,
                                            size_t logicalIndex, ScalarType destinationType,
                                            const PreparedGenerationValues* prepared = nullptr) {
        if constexpr (std::is_same_v<Pattern, Component::RawConstantPattern>) {
            return pattern.parameters.bits;
        } else if constexpr (std::is_same_v<Pattern, Component::UniformRawIntegerPattern>) {
            return static_cast<uint64_t>(static_cast<int64_t>(indexedUniformInteger<Checked>(
                seed, domain, logicalIndex, pattern.parameters.lower, pattern.parameters.upper)));
        } else if constexpr (std::is_same_v<Pattern, Component::UniformFiniteEncodedValuePattern>) {
            if constexpr (Checked)
                return uniformFiniteEncodedValue(destinationType, seed, domain, logicalIndex);
            else
                return prepared->finiteEncodings[counterRandom(seed, domain, logicalIndex) %
                                                 prepared->finiteEncodings.size()];
        } else if constexpr (std::is_same_v<Pattern, Component::RandomRawBitsPattern>) {
            return counterRandom(seed, domain, logicalIndex);
        } else if constexpr (std::is_same_v<Pattern, Component::RawSerialDimensionPattern>) {
            if (Checked && pattern.parameters.dimension >= shape.rank())
                throw std::runtime_error("Generation dimension exceeds tensor rank.");
            return static_cast<uint64_t>(indices[pattern.parameters.dimension]);
        } else {
            throw std::runtime_error("Requested generation pattern does not produce raw storage.");
        }
    }

    static uint64_t rawGenerationValue(const Component& component, uint64_t seed, uint64_t domain,
                                       std::span<const size_t> indices, const Shape& shape,
                                       size_t logicalIndex, ScalarType destinationType) {
        return std::visit(
            [&](const auto& pattern) -> uint64_t {
                return rawGenerationValueKnown(pattern, seed, domain, indices, shape, logicalIndex,
                                               destinationType);
            },
            component.pattern_);
    }

    static double uniformRealValue(const Component::UniformRealPattern& pattern,
                                   uint64_t randomMantissa) {
        return pattern.parameters.lower + uniformUnitFromMantissa(randomMantissa) *
                                              (pattern.parameters.upper - pattern.parameters.lower);
    }

    template <bool Checked = true, typename Pattern>
    static double baseGenerationValueKnown(const Pattern& pattern, uint64_t seed, uint64_t domain,
                                           std::span<const size_t> indices, const Shape& shape,
                                           size_t logicalIndex, ScalarType destinationType,
                                           const PreparedGenerationValues* prepared = nullptr) {
        if constexpr (std::is_same_v<Pattern, Component::ZeroPattern>) {
            return 0.0;
        } else if constexpr (std::is_same_v<Pattern, Component::ConstantPattern>) {
            return pattern.parameters.value;
        } else if constexpr (std::is_same_v<Pattern, Component::ChoicePattern>) {
            if (Checked && pattern.parameters.values.empty())
                throw std::runtime_error("Choice generation requires values.");
            return pattern.parameters.values[counterRandom(seed, domain, logicalIndex) %
                                             pattern.parameters.values.size()];
        } else if constexpr (std::is_same_v<Pattern, Component::UniformIntegerPattern>) {
            return static_cast<double>(indexedUniformInteger<Checked>(
                seed, domain, logicalIndex, pattern.parameters.lower, pattern.parameters.upper));
        } else if constexpr (std::is_same_v<Pattern, Component::AbsoluteUniformIntegerPattern>) {
            return std::abs(static_cast<double>(indexedUniformInteger<Checked>(
                seed, domain, logicalIndex, pattern.parameters.lower, pattern.parameters.upper)));
        } else if constexpr (std::is_same_v<Pattern, Component::UniformRealPattern>) {
            if (Checked && pattern.parameters.lower > pattern.parameters.upper)
                throw std::runtime_error("Uniform-real lower bound exceeds upper bound.");
            return uniformRealValue(pattern, counterRandom(seed, domain, logicalIndex) >> 11);
        } else if constexpr (std::is_same_v<Pattern, Component::BinomialHalfPattern>) {
            const uint32_t trials = pattern.parameters.trials;
            const uint64_t mask = trials == 64 ? ~uint64_t{0} : (uint64_t{1} << trials) - 1U;
            return static_cast<double>(
                std::popcount(counterRandom(seed, domain, logicalIndex) & mask));
        } else if constexpr (std::is_same_v<Pattern, Component::NormalPattern>) {
            constexpr double twoPi = 6.28318530717958647692528676655900576;
            const double first = indexedUniformUnit(seed, domain, 2 * logicalIndex);
            const double second = indexedUniformUnit(seed, domain, 2 * logicalIndex + 1);
            const double standardNormal =
                std::sqrt(-2.0 * std::log(first)) * std::cos(twoPi * second);
            return pattern.parameters.mean + pattern.parameters.standardDeviation * standardNormal;
        } else if constexpr (std::is_same_v<Pattern, Component::SinePattern>) {
            return std::sin(static_cast<double>(logicalIndex));
        } else if constexpr (std::is_same_v<Pattern, Component::CosinePattern>) {
            return std::cos(static_cast<double>(logicalIndex));
        } else if constexpr (std::is_same_v<Pattern, Component::AbsoluteSinePattern>) {
            return std::abs(std::sin(static_cast<double>(logicalIndex)));
        } else if constexpr (std::is_same_v<Pattern, Component::AbsoluteCosinePattern>) {
            return std::abs(std::cos(static_cast<double>(logicalIndex)));
        } else if constexpr (std::is_same_v<Pattern, Component::SerialIndexPattern>) {
            return static_cast<double>(logicalIndex);
        } else if constexpr (std::is_same_v<Pattern, Component::SerialDimensionPattern>) {
            if (Checked && pattern.parameters.dimension >= shape.rank())
                throw std::runtime_error("Generation dimension exceeds tensor rank.");
            return static_cast<double>(indices[pattern.parameters.dimension]);
        } else if constexpr (std::is_same_v<Pattern, Component::AffineIndexRemainderPattern>) {
            const auto& parameters = pattern.parameters;
            if (Checked && parameters.dimensionCoefficients.size() != shape.rank())
                throw std::runtime_error(
                    "Affine-index coefficient count must match the tensor rank.");
            if (Checked && parameters.positiveDivisor <= 0)
                throw std::runtime_error("Affine-index remainder divisor must be positive.");

            int64_t value = parameters.offset;
            if constexpr (!Checked) {
                for (size_t dimension = 0; dimension < shape.rank(); ++dimension)
                    value += parameters.dimensionCoefficients[dimension] *
                             static_cast<int64_t>(indices[dimension]);
            } else {
                for (size_t dimension = 0; dimension < shape.rank(); ++dimension) {
                    const int64_t coefficient = parameters.dimensionCoefficients[dimension];
                    if (indices[dimension] >
                        static_cast<size_t>(std::numeric_limits<int64_t>::max()))
                        throw std::runtime_error("Affine-index coordinate exceeds Int64.");
                    const int64_t index = static_cast<int64_t>(indices[dimension]);
                    int64_t term = 0;
                    if (coefficient > 0) {
                        if (index > std::numeric_limits<int64_t>::max() / coefficient)
                            throw std::runtime_error("Affine-index multiplication overflow.");
                        term = coefficient * index;
                    } else if (coefficient < 0) {
                        if (coefficient == std::numeric_limits<int64_t>::min()) {
                            if (index > 1)
                                throw std::runtime_error("Affine-index multiplication overflow.");
                            term = index == 0 ? 0 : std::numeric_limits<int64_t>::min();
                        } else {
                            const int64_t magnitude = -coefficient;
                            if (index > std::numeric_limits<int64_t>::max() / magnitude)
                                throw std::runtime_error("Affine-index multiplication overflow.");
                            term = -(magnitude * index);
                        }
                    }

                    if ((term > 0 && value > std::numeric_limits<int64_t>::max() - term) ||
                        (term < 0 && value < std::numeric_limits<int64_t>::min() - term))
                        throw std::runtime_error("Affine-index addition overflow.");
                    value += term;
                }
            }
            return static_cast<double>(value % parameters.positiveDivisor);
        } else if constexpr (std::is_same_v<Pattern, Component::IdentityPattern>) {
            if (Checked && shape.rank() < 2)
                throw std::runtime_error("Identity generation requires rank at least two.");
            return indices[0] == indices[1] ? 1.0 : 0.0;
        } else if constexpr (std::is_same_v<Pattern,
                                            Component::CheckerboardUniformIntegerPattern>) {
            double value = static_cast<double>(indexedUniformInteger<Checked>(
                seed, domain, logicalIndex, pattern.parameters.lower, pattern.parameters.upper));
            size_t parity = 0;
            for (const size_t index : indices) parity ^= index;
            return (parity & 1U) == 0 ? -value : value;
        } else if constexpr (std::is_same_v<Pattern, Component::TypeMaximumPattern>) {
            if constexpr (Checked)
                return typeMaximum(destinationType);
            else
                return prepared->constant;
        } else if constexpr (std::is_same_v<Pattern, Component::TypeLowestPattern>) {
            if constexpr (Checked)
                return typeLowest(destinationType);
            else
                return prepared->constant;
        } else if constexpr (std::is_same_v<Pattern, Component::TypeDenormalMinimumPattern>) {
            if constexpr (Checked)
                return typeDenormalMinimum(destinationType);
            else
                return prepared->constant;
        } else if constexpr (std::is_same_v<Pattern, Component::TypeDenormalMaximumPattern>) {
            if constexpr (Checked)
                return typeDenormalMaximum(destinationType);
            else
                return prepared->constant;
        } else if constexpr (std::is_same_v<Pattern, Component::TypeNaNPattern>) {
            if constexpr (Checked)
                return typeNaN(destinationType);
            else
                return prepared->constant;
        } else if constexpr (std::is_same_v<Pattern, Component::TypeInfinityPattern>) {
            if constexpr (Checked)
                return typeInfinity(destinationType, false);
            else
                return prepared->constant;
        } else if constexpr (std::is_same_v<Pattern, Component::TypeNegativeInfinityPattern>) {
            if constexpr (Checked)
                return typeInfinity(destinationType, true);
            else
                return prepared->constant;
        } else if constexpr (std::is_same_v<Pattern, Component::TypeNegativeZeroPattern>) {
            return -0.0;
        } else if constexpr (std::is_same_v<Pattern, Component::UniformTypeRangePattern>) {
            const double maximum = Checked ? typeMaximum(destinationType) : prepared->maximum;
            const double unit = indexedUniformUnit(seed, domain, logicalIndex);
            return maximum * (2.0 * unit - 1.0);
        } else if constexpr (std::is_same_v<Pattern, Component::RandomEncodedExponentPattern>) {
            if constexpr (Checked)
                return randomEncodedExponentValue(pattern.parameters, seed, domain, logicalIndex,
                                                  destinationType);
            else
                return prepared->exponent(counterRandom(seed, domain, logicalIndex));
        } else {
            throw std::runtime_error("Raw generation requires encoded storage output.");
        }
    }

    static double baseGenerationValue(const Component& component, uint64_t seed, uint64_t domain,
                                      std::span<const size_t> indices, const Shape& shape,
                                      size_t logicalIndex, ScalarType destinationType) {
        return std::visit(
            [&](const auto& pattern) -> double {
                return baseGenerationValueKnown(pattern, seed, domain, indices, shape, logicalIndex,
                                                destinationType);
            },
            component.pattern_);
    }

    template <bool Checked = true, typename Pattern>
    static double generationValueKnown(const Component& component, const Pattern& pattern,
                                       uint64_t seed, uint64_t domain,
                                       std::span<const size_t> indices, const Shape& shape,
                                       size_t logicalIndex, ScalarType destinationType,
                                       const PreparedGenerationValues* prepared = nullptr) {
        if (component.zeroOutsideMainDiagonal_) {
            if (Checked && shape.rank() < 2)
                throw std::runtime_error("Main-diagonal generation requires rank at least two.");
            if (indices[0] != indices[1]) return 0.0;
        }
        double value = baseGenerationValueKnown<Checked>(pattern, seed, domain, indices, shape,
                                                         logicalIndex, destinationType, prepared);
        switch (component.unaryTransform_) {
            case Component::UnaryTransform::None:
                break;
            case Component::UnaryTransform::Absolute:
                value = std::abs(value);
                break;
            case Component::UnaryTransform::Sine:
                value = std::sin(value);
                break;
            case Component::UnaryTransform::Cosine:
                value = std::cos(value);
                break;
            case Component::UnaryTransform::Exp2:
                value = std::exp2(value);
                break;
            default:
                throw std::runtime_error("Invalid generation transform.");
        }
        value = value * component.affineValue_.scale + component.affineValue_.offset;

        if (component.alternatingSign_.has_value()) {
            size_t parity = 0;
            for (const size_t dimension : component.alternatingSign_->dimensions) {
                if (Checked && dimension >= shape.rank())
                    throw std::runtime_error(
                        "Alternating-sign generation dimension exceeds tensor rank.");
                parity ^= indices[dimension];
            }
            if (((parity & 1U) != 0) == component.alternatingSign_->negativeWhenOdd) value = -value;
        }
        return value;
    }

    static double generationValue(const Component& component, uint64_t seed, uint64_t domain,
                                  std::span<const size_t> indices, const Shape& shape,
                                  size_t logicalIndex, ScalarType destinationType) {
        return std::visit(
            [&](const auto& pattern) {
                return generationValueKnown(component, pattern, seed, domain, indices, shape,
                                            logicalIndex, destinationType);
            },
            component.pattern_);
    }

    static const GenerationRecipe::BoundComponent& realComponent(const GenerationRecipe& recipe) {
        return *std::visit(
            [](const auto& policy) -> const GenerationRecipe::BoundComponent* {
                using Policy = std::remove_cvref_t<decltype(policy)>;
                if constexpr (std::is_same_v<Policy, GenerationRecipe::RealOnlyPolicy>)
                    return &policy.real;
                else if constexpr (std::is_same_v<Policy, GenerationRecipe::ReplicatedPolicy>)
                    return &policy.value;
                else
                    return &policy.real;
            },
            recipe.complexPolicy_);
    }

    static uint64_t randomDomain(const GenerationRecipe::BoundComponent& component) {
        return component.randomDomain;
    }

    static double generatedNumericalValue(const GenerationRecipe& recipe,
                                          std::span<const size_t> indices, const Shape& shape,
                                          size_t logicalIndex, ScalarType destinationType) {
        const GenerationRecipe::BoundComponent& component = realComponent(recipe);
        if (component.component.isRaw())
            throw std::runtime_error("Raw generation recipe does not produce a numerical value.");
        return generationValue(component.component, recipe.settings_.seed, randomDomain(component),
                               indices, shape, logicalIndex + recipe.settings_.indexOffset,
                               destinationType);
    }

    static uint64_t generatedRawValue(const GenerationRecipe& recipe,
                                      std::span<const size_t> indices, const Shape& shape,
                                      size_t logicalIndex, ScalarType destinationType) {
        const GenerationRecipe::BoundComponent& component = realComponent(recipe);
        if (!component.component.isRaw())
            throw std::runtime_error("Numerical generation recipe does not produce raw bits.");
        return rawGenerationValue(component.component, recipe.settings_.seed,
                                  randomDomain(component), indices, shape,
                                  logicalIndex + recipe.settings_.indexOffset, destinationType);
    }

    static void writeRawGeneration(Tensor destination, const GenerationRecipe& recipe,
                                   const GenerationRecipe::BoundComponent& bound,
                                   std::span<const size_t> indices, size_t logicalIndex) {
        const uint16_t bits = scalarTypeInfo(destination.type()).storageBits;
        if (bits > 64)
            throw std::runtime_error("Raw generation supports scalar encodings up to 64 bits.");

        const ptrdiff_t elementOffset =
            detail::preparedElementOffset(destination.layout(), indices);
        const uint64_t raw = rawGenerationValue(
            bound.component, recipe.settings_.seed, randomDomain(bound), indices,
            destination.shape(), logicalIndex + recipe.settings_.indexOffset, destination.type());
        const uint64_t offsetBits = bitOffset(destination.type(), elementOffset);
        if (bits <= 32) {
            writePackedBits(destination.rawEncodedBackingStorage(), offsetBits, bits,
                            static_cast<uint32_t>(raw));
        } else {
            writeNative<uint64_t>(destination.rawEncodedBackingStorage(),
                                  static_cast<size_t>(offsetBits / 8), raw);
        }
    }

    static void generateElement(Tensor destination, const GenerationRecipe& recipe,
                                std::span<const size_t> indices, size_t logicalIndex,
                                const ScalarConversionOptions* selectedOptions = nullptr) {
        validateRecipe(recipe, destination.shape(), destination.type());
        const auto options = selectedOptions ? *selectedOptions
                                             : implicitStorageConversionOptions(destination.type());
        const bool complexOutput =
            scalarTypeInfo(destination.type()).category == ScalarCategory::Complex;
        const GenerationRecipe::BoundComponent* real = &realComponent(recipe);

        if (!complexOutput) {
            if (real->component.isRaw()) {
                writeRawGeneration(destination, recipe, *real, indices, logicalIndex);
                return;
            }
            destination.storeFrom(indices,
                                  generatedNumericalValue(recipe, indices, destination.shape(),
                                                          logicalIndex, destination.type()),
                                  options);
            return;
        }

        logicalIndex += recipe.settings_.indexOffset;
        std::visit(
            [&](const auto& policy) {
                using Policy = std::remove_cvref_t<decltype(policy)>;
                if constexpr (std::is_same_v<Policy, GenerationRecipe::RealOnlyPolicy>) {
                    if (policy.real.component.isRaw())
                        throw std::runtime_error("Raw generation does not support complex output.");
                    const double value = generationValue(
                        policy.real.component, recipe.settings_.seed, randomDomain(policy.real),
                        indices, destination.shape(), logicalIndex, destination.type());
                    destination.storeFrom(indices, std::complex<double>(value, 0.0), options);
                } else if constexpr (std::is_same_v<Policy, GenerationRecipe::ReplicatedPolicy>) {
                    if (policy.value.component.isRaw())
                        throw std::runtime_error("Raw generation does not support complex output.");
                    const double value = generationValue(
                        policy.value.component, recipe.settings_.seed, randomDomain(policy.value),
                        indices, destination.shape(), logicalIndex, destination.type());
                    destination.storeFrom(indices, std::complex<double>(value, value), options);
                } else {
                    if (policy.real.component.isRaw() || policy.imaginary.component.isRaw())
                        throw std::runtime_error("Raw generation does not support complex output.");
                    const double realValue = generationValue(
                        policy.real.component, recipe.settings_.seed, randomDomain(policy.real),
                        indices, destination.shape(), logicalIndex, destination.type());
                    const double imaginaryValue =
                        generationValue(policy.imaginary.component, recipe.settings_.seed,
                                        randomDomain(policy.imaginary), indices,
                                        destination.shape(), logicalIndex, destination.type());
                    destination.storeFrom(indices, std::complex<double>(realValue, imaginaryValue),
                                          options);
                }
            },
            recipe.complexPolicy_);
    }

    using NumericalWriter = void (*)(std::span<std::byte>, ptrdiff_t, double);
    using ComplexWriter = void (*)(std::span<std::byte>, ptrdiff_t, std::complex<double>);
    using RawWriter = void (*)(std::span<std::byte>, ptrdiff_t, uint64_t);

    static NumericalWriter numericalWriter(ScalarType type) {
        return visitScalarType(type, []<typename Tag>() -> NumericalWriter {
            return [](std::span<std::byte> storage, ptrdiff_t offset, double value) {
                preparedEncodeScalarKnown<Tag>(storage, offset, value);
            };
        });
    }

    static ComplexWriter complexWriter(ScalarType type) {
        return visitScalarType(type, []<typename Tag>() -> ComplexWriter {
            return [](std::span<std::byte> storage, ptrdiff_t offset, std::complex<double> value) {
                preparedEncodeScalarKnown<Tag>(storage, offset, value);
            };
        });
    }

    static RawWriter rawWriter(ScalarType type) {
        return visitScalarType(type, []<typename Tag>() -> RawWriter {
            constexpr uint16_t bits = scalarTypeInfo(Tag::type).storageBits;
            if constexpr (bits > 64) {
                throw std::runtime_error("Raw generation supports scalar encodings up to 64 bits.");
            } else {
                return [](std::span<std::byte> storage, ptrdiff_t offset, uint64_t value) {
                    constexpr uint16_t bits = scalarTypeInfo(Tag::type).storageBits;
                    if constexpr (bits <= 32)
                        preparedWritePackedBits(storage, preparedBitOffset(Tag::type, offset), bits,
                                                static_cast<uint32_t>(value));
                    else
                        std::memcpy(storage.data() + static_cast<size_t>(
                                                         preparedBitOffset(Tag::type, offset) / 8),
                                    &value, sizeof(value));
                };
            }
        });
    }

    template <typename Pattern>
    static bool usesCoordinates(const Component& component) {
        constexpr bool patternUsesCoordinates =
            std::is_same_v<Pattern, Component::SerialDimensionPattern> ||
            std::is_same_v<Pattern, Component::AffineIndexRemainderPattern> ||
            std::is_same_v<Pattern, Component::IdentityPattern> ||
            std::is_same_v<Pattern, Component::CheckerboardUniformIntegerPattern> ||
            std::is_same_v<Pattern, Component::RawSerialDimensionPattern>;
        return patternUsesCoordinates || component.zeroOutsideMainDiagonal_ ||
               component.alternatingSign_.has_value();
    }

    template <typename Pattern>
    static constexpr bool isConstantPattern =
        std::is_same_v<Pattern, Component::ZeroPattern> ||
        std::is_same_v<Pattern, Component::ConstantPattern> ||
        std::is_same_v<Pattern, Component::TypeMaximumPattern> ||
        std::is_same_v<Pattern, Component::TypeLowestPattern> ||
        std::is_same_v<Pattern, Component::TypeDenormalMinimumPattern> ||
        std::is_same_v<Pattern, Component::TypeDenormalMaximumPattern> ||
        std::is_same_v<Pattern, Component::TypeNaNPattern> ||
        std::is_same_v<Pattern, Component::TypeInfinityPattern> ||
        std::is_same_v<Pattern, Component::TypeNegativeInfinityPattern> ||
        std::is_same_v<Pattern, Component::TypeNegativeZeroPattern>;

    template <typename Pattern>
    static bool isUnmodifiedComponent(const Component& component) {
        return !usesCoordinates<Pattern>(component) &&
               component.unaryTransform_ == Component::UnaryTransform::None &&
               component.affineValue_.scale == 1.0 && component.affineValue_.offset == 0.0;
    }

    template <typename Pattern>
    static PreparedGenerationValues prepareValues(const Pattern& pattern, ScalarType type) {
        PreparedGenerationValues result;
        if constexpr (isConstantPattern<Pattern>)
            result.constant = baseGenerationValueKnown(pattern, 0, 0, {}, Shape{}, 0, type);
        else if constexpr (std::is_same_v<Pattern, Component::UniformTypeRangePattern>)
            result.maximum = typeMaximum(type);
        else if constexpr (std::is_same_v<Pattern, Component::RandomEncodedExponentPattern>)
            result.exponent = prepareEncodedExponent(pattern.parameters, type);
        else if constexpr (std::is_same_v<Pattern, Component::UniformFiniteEncodedValuePattern>)
            result.finiteEncodings = finiteEncodedValues(type);
        return result;
    }

    template <typename Pattern>
    static double preparedBaseValue(const Component&, const void* erasedPattern, uint64_t seed,
                                    uint64_t domain, std::span<const size_t> indices,
                                    const Shape& shape, size_t logicalIndex,
                                    ScalarType destinationType,
                                    const PreparedGenerationValues& values) {
        return baseGenerationValueKnown<false>(*static_cast<const Pattern*>(erasedPattern), seed,
                                               domain, indices, shape, logicalIndex,
                                               destinationType, &values);
    }

    template <typename Pattern>
    static double preparedModifiedValue(const Component& component, const void* erasedPattern,
                                        uint64_t seed, uint64_t domain,
                                        std::span<const size_t> indices, const Shape& shape,
                                        size_t logicalIndex, ScalarType destinationType,
                                        const PreparedGenerationValues& values) {
        return generationValueKnown<false>(component, *static_cast<const Pattern*>(erasedPattern),
                                           seed, domain, indices, shape, logicalIndex,
                                           destinationType, &values);
    }

    template <typename Pattern>
    static uint64_t preparedRawValue(const void* erasedPattern, uint64_t seed, uint64_t domain,
                                     std::span<const size_t> indices, const Shape& shape,
                                     size_t logicalIndex, ScalarType destinationType,
                                     const PreparedGenerationValues& values) {
        return rawGenerationValueKnown<false>(*static_cast<const Pattern*>(erasedPattern), seed,
                                              domain, indices, shape, logicalIndex, destinationType,
                                              &values);
    }

    static PreparedNumericalGenerator prepareNumericalComponent(
        const GenerationRecipe::BoundComponent& bound, uint64_t seed, ScalarType destinationType) {
        if (bound.component.isRaw())
            throw std::runtime_error("Raw generation does not support complex output.");
        return std::visit(
            [&](const auto& pattern) {
                using Pattern = std::remove_cvref_t<decltype(pattern)>;
                return PreparedNumericalGenerator{
                    .component = &bound.component,
                    .pattern = &pattern,
                    .seed = seed,
                    .domain = randomDomain(bound),
                    .destinationType = destinationType,
                    .value = isUnmodifiedComponent<Pattern>(bound.component)
                                 ? preparedBaseValue<Pattern>
                                 : preparedModifiedValue<Pattern>,
                    .values = prepareValues(pattern, destinationType),
                    .usesCoordinates = usesCoordinates<Pattern>(bound.component),
                    .isConstant =
                        isConstantPattern<Pattern> && !usesCoordinates<Pattern>(bound.component),
                };
            },
            bound.component.pattern_);
    }

    static PreparedRawGenerator prepareRawComponent(const GenerationRecipe::BoundComponent& bound,
                                                    uint64_t seed, ScalarType destinationType) {
        if (!bound.component.isRaw())
            throw std::runtime_error("Numerical generation recipe does not produce raw bits.");
        return std::visit(
            [&](const auto& pattern) {
                using Pattern = std::remove_cvref_t<decltype(pattern)>;
                return PreparedRawGenerator{
                    .pattern = &pattern,
                    .seed = seed,
                    .domain = randomDomain(bound),
                    .destinationType = destinationType,
                    .value = preparedRawValue<Pattern>,
                    .values = prepareValues(pattern, destinationType),
                    .usesCoordinates = usesCoordinates<Pattern>(bound.component),
                };
            },
            bound.component.pattern_);
    }

    static void incrementCoordinates(std::vector<size_t>& indices, const Shape& shape,
                                     std::span<const size_t> order) {
        const auto incrementDimension = [&](size_t dimension) {
            if (++indices[dimension] < shape.dimensions()[dimension]) return true;
            indices[dimension] = 0;
            return false;
        };

        for (size_t position = shape.rank(); position > 0; --position) {
            const size_t axis = order[position - 1];
            if (incrementDimension(axis)) return;
        }
    }

    template <typename WriteFirst>
    static bool fillContiguous(Tensor destination, std::span<const size_t> order,
                               WriteFirst&& writeFirst) {
        const uint16_t storageBits = scalarTypeInfo(destination.type()).storageBits;
        if (!detail::isContiguous(destination.layout(), order)) return false;

        const size_t elementCount = destination.elementCount();
        if (elementCount == 0) return true;
        const auto storage = destination.rawEncodedBackingStorage();
        if (storageBits % 8 != 0) {
            if (destination.layout().offset() != 0) return false;
            const size_t elementsPerGroup = scalarElementGroupSize(destination.type());
            const size_t bytesPerGroup = elementsPerGroup * storageBits / 8;
            const size_t fullGroupCount = elementCount / elementsPerGroup;
            if (fullGroupCount == 0) {
                for (size_t logicalIndex = 0; logicalIndex < elementCount; ++logicalIndex)
                    writeFirst(storage, static_cast<ptrdiff_t>(logicalIndex));
                return true;
            }

            for (size_t logicalIndex = 0; logicalIndex < elementsPerGroup; ++logicalIndex)
                writeFirst(storage, static_cast<ptrdiff_t>(logicalIndex));
            const size_t fullGroupBytes = fullGroupCount * bytesPerGroup;
            size_t initializedBytes = bytesPerGroup;
            while (initializedBytes < fullGroupBytes) {
                const size_t copyBytes =
                    std::min(initializedBytes, fullGroupBytes - initializedBytes);
                std::memcpy(storage.data() + initializedBytes, storage.data(), copyBytes);
                initializedBytes += copyBytes;
            }
            for (size_t logicalIndex = fullGroupCount * elementsPerGroup;
                 logicalIndex < elementCount; ++logicalIndex)
                writeFirst(storage, static_cast<ptrdiff_t>(logicalIndex));
            return true;
        }

        const size_t elementBytes = storageBits / 8;
        const size_t byteOffset = static_cast<size_t>(destination.layout().offset()) * elementBytes;
        const size_t byteCount = elementCount * elementBytes;
        writeFirst(storage, destination.layout().offset());

        size_t initializedBytes = elementBytes;
        while (initializedBytes < byteCount) {
            const size_t copyBytes = std::min(initializedBytes, byteCount - initializedBytes);
            std::memcpy(storage.data() + byteOffset + initializedBytes, storage.data() + byteOffset,
                        copyBytes);
            initializedBytes += copyBytes;
        }
        return true;
    }

    template <typename Function>
    static void forEachGenerationElement(Tensor destination, std::span<const size_t> logicalOrder,
                                         bool valueUsesCoordinates, Function&& function) {
        const detail::IndexIteration logicalPlan(destination.shape(), logicalOrder);
        const size_t elementCount = logicalPlan.count();
        const bool independent = hasProvablyIndependentElements(destination);
        const auto fixedTraversalOrder = ascendingIndexOrder(destination.shape().rank());
        const std::span<const size_t> traversalOrder =
            independent ? logicalOrder : std::span<const size_t>(fixedTraversalOrder);
        const detail::IndexIteration traversalPlan(destination.shape(), traversalOrder);
        const bool sameOrder = std::ranges::equal(traversalPlan.axes(), logicalPlan.axes());
        const bool contiguous = detail::isContiguous(destination.layout(), traversalOrder);
        const bool trackCoordinates = valueUsesCoordinates || !contiguous || !sameOrder;
        const int threadCount = independent ? operationThreadCount(elementCount, 4096, 8) : 1;

        const auto runRange = [&](size_t first, size_t end) {
            if (first == end) return;
            std::vector<size_t> indices;
            if (trackCoordinates) {
                indices.resize(destination.shape().rank());
                traversalPlan.coordinates(first, indices);
            }
            for (size_t traversalIndex = first; traversalIndex < end; ++traversalIndex) {
                const size_t logicalIndex =
                    sameOrder ? traversalIndex : logicalPlan.linearIndex(indices);
                const ptrdiff_t offset =
                    contiguous
                        ? destination.layout().offset() + static_cast<ptrdiff_t>(traversalIndex)
                        : detail::preparedElementOffset(destination.layout(), indices);
                function(std::span<const size_t>(indices), logicalIndex, offset);
                if (trackCoordinates) traversalPlan.advance(indices);
            }
        };

#ifdef _OPENMP
        if (threadCount > 1 &&
            elementCount <= static_cast<size_t>(std::numeric_limits<ptrdiff_t>::max())) {
            std::exception_ptr error;
#pragma omp parallel num_threads(threadCount)
            {
                try {
                    const size_t threadIndex = static_cast<size_t>(omp_get_thread_num());
                    const size_t actualThreadCount = static_cast<size_t>(omp_get_num_threads());
                    const size_t baseCount = elementCount / actualThreadCount;
                    const size_t remainder = elementCount % actualThreadCount;
                    const size_t first = threadIndex * baseCount + std::min(threadIndex, remainder);
                    const size_t count = baseCount + static_cast<size_t>(threadIndex < remainder);
                    runRange(first, first + count);
                } catch (...) {
#pragma omp critical(hostnumerics_generation_error)
                    {
                        if (!error) error = std::current_exception();
                    }
                }
            }
            if (error) std::rethrow_exception(error);
            return;
        }
#else
        (void)threadCount;
#endif
        runRange(0, elementCount);
    }

    template <typename Tag>
    static auto encodeGeneratedInteger(auto value) {
        constexpr ScalarCategory category = scalarTypeInfo(Tag::type).category;
        static_assert(category == ScalarCategory::Boolean ||
                      category == ScalarCategory::SignedInteger ||
                      category == ScalarCategory::UnsignedInteger);
        if constexpr (category == ScalarCategory::Boolean) {
            return static_cast<uint8_t>(value != 0);
        } else if constexpr (Tag::type == ScalarType::Int4) {
            return static_cast<uint32_t>(value) & 0xfU;
        } else if constexpr (category == ScalarCategory::SignedInteger) {
            using Storage = typename Tag::Storage;
            using Unsigned = std::make_unsigned_t<Storage>;
            return std::bit_cast<Storage>(static_cast<Unsigned>(value));
        } else {
            return static_cast<typename Tag::Storage>(value);
        }
    }

    template <typename Tag, typename Pattern>
    static auto generatedEncodedValue(const Pattern& pattern, const GenerationRecipe& recipe,
                                      const GenerationRecipe::BoundComponent& bound,
                                      const Shape& shape, size_t logicalIndex,
                                      ScalarType destinationType,
                                      const PreparedGenerationValues& values) {
        constexpr ScalarCategory category = scalarTypeInfo(Tag::type).category;
        constexpr bool integerDestination = category == ScalarCategory::Boolean ||
                                            category == ScalarCategory::SignedInteger ||
                                            category == ScalarCategory::UnsignedInteger;
        if constexpr (integerDestination &&
                      std::is_same_v<Pattern, Component::UniformIntegerPattern>) {
            return encodeGeneratedInteger<Tag>(indexedUniformInteger<false>(
                recipe.settings_.seed, randomDomain(bound), logicalIndex, pattern.parameters.lower,
                pattern.parameters.upper));
        } else if constexpr (integerDestination &&
                             std::is_same_v<Pattern, Component::SerialIndexPattern>) {
            return encodeGeneratedInteger<Tag>(logicalIndex);
        } else {
            return encodeScalarValueKnown<Tag>(
                baseGenerationValueKnown<false>(pattern, recipe.settings_.seed, randomDomain(bound),
                                                {}, shape, logicalIndex, destinationType, &values));
        }
    }

    template <typename Tag>
    static auto rawStorageValue(uint64_t value) {
        using Storage = typename Tag::Storage;
        if constexpr (std::is_void_v<Storage>) {
            return static_cast<uint32_t>(value);
        } else if constexpr (sizeof(Storage) == 1) {
            return std::bit_cast<Storage>(static_cast<uint8_t>(value));
        } else if constexpr (sizeof(Storage) == 2) {
            return std::bit_cast<Storage>(static_cast<uint16_t>(value));
        } else if constexpr (sizeof(Storage) == 4) {
            return std::bit_cast<Storage>(static_cast<uint32_t>(value));
        } else {
            static_assert(sizeof(Storage) == 8);
            return std::bit_cast<Storage>(value);
        }
    }

    template <typename EncodedValue>
    static bool generateContiguousPacked(Tensor destination, std::span<const size_t> order,
                                         bool valueUsesCoordinates, EncodedValue&& encodedValue) {
        const uint16_t bits = scalarTypeInfo(destination.type()).storageBits;
        if (bits >= 8 || valueUsesCoordinates || destination.layout().offset() != 0 ||
            !detail::isContiguous(destination.layout(), order))
            return false;

        const size_t elementsPerGroup = scalarElementGroupSize(destination.type());
        const size_t bytesPerGroup = elementsPerGroup * bits / 8;
        const size_t elementCount = destination.elementCount();
        const size_t fullGroupCount = elementCount / elementsPerGroup;
        const uint32_t valueMask = (uint32_t{1} << bits) - 1U;
        const auto storage = destination.rawEncodedBackingStorage();

        forEachParallelIndex(fullGroupCount, elementCount, true, 4096, [&](size_t groupIndex) {
            uint32_t packed = 0;
            const size_t firstElement = groupIndex * elementsPerGroup;
            for (size_t element = 0; element < elementsPerGroup; ++element)
                packed |= (static_cast<uint32_t>(encodedValue(firstElement + element)) & valueMask)
                          << (element * bits);
            for (size_t byte = 0; byte < bytesPerGroup; ++byte)
                storage[groupIndex * bytesPerGroup + byte] =
                    static_cast<std::byte>((packed >> (byte * 8)) & 0xffU);
        });

        for (size_t logicalIndex = fullGroupCount * elementsPerGroup; logicalIndex < elementCount;
             ++logicalIndex)
            writePackedBits(storage, logicalIndex * bits, bits,
                            static_cast<uint32_t>(encodedValue(logicalIndex)));
        return true;
    }

    template <typename Tag, typename EncodedValue>
    static bool generateContiguousEncodedValues(Tensor destination, std::span<const size_t> order,
                                                EncodedValue&& encodedValue) {
        using Storage = typename Tag::Storage;
        if constexpr (std::is_void_v<Storage>) {
            return generateContiguousPacked(destination, order, false,
                                            std::forward<EncodedValue>(encodedValue));
        } else {
            const auto storage = destination.rawEncodedBackingStorage();
            const size_t byteOffset =
                static_cast<size_t>(destination.layout().offset()) * sizeof(Storage);
            if (reinterpret_cast<uintptr_t>(storage.data() + byteOffset) % alignof(Storage) != 0)
                return false;
            auto* output = reinterpret_cast<Storage*>(storage.data() + byteOffset);
            forEachParallelIndex(
                destination.elementCount(), destination.elementCount(), true, 4096,
                [&](size_t logicalIndex) { output[logicalIndex] = encodedValue(logicalIndex); });
            return true;
        }
    }

    static constexpr size_t maximumDiscreteLookupSize = 256;
    static constexpr size_t uniformRealLookupBits = 12;
    static constexpr size_t uniformRealLookupSize = size_t{1} << uniformRealLookupBits;
    static constexpr size_t uniformRandomMantissaBits = 53;
    static constexpr size_t uniformRealLookupShift =
        uniformRandomMantissaBits - uniformRealLookupBits;
    static constexpr uint16_t ambiguousUniformRealLookupValue = 0x100U;

    template <typename Encoded>
    struct EncodedDiscreteLookup {
        std::array<Encoded, maximumDiscreteLookupSize> values{};
        size_t size = 0;
    };

    struct AlternatingDimension {
        size_t stride;
        size_t oddExtent;
    };

    struct PreparedAlternatingSign {
        std::vector<AlternatingDimension> dimensions;
        std::vector<uint8_t> repeatingNegation;
        size_t repeatMask = 0;
        bool negativeWhenOdd;

        bool negatesFromDimensions(size_t logicalIndex) const {
            size_t parity = 0;
            for (const auto [stride, oddExtent] : dimensions) {
                size_t coordinate = stride == 1 ? logicalIndex : logicalIndex / stride;
                if (oddExtent != 0) coordinate %= oddExtent;
                parity ^= coordinate;
            }
            return ((parity & 1U) != 0) == negativeWhenOdd;
        }

        bool negates(size_t logicalIndex) const {
            if (repeatingNegation.empty()) return negatesFromDimensions(logicalIndex);
            const size_t index = repeatMask == 0 ? logicalIndex % repeatingNegation.size()
                                                 : logicalIndex & repeatMask;
            return repeatingNegation[index] != 0;
        }
    };

    static PreparedAlternatingSign prepareAlternatingSign(const Component& component,
                                                          const Shape& shape,
                                                          std::span<const size_t> order) {
        const auto& parameters = *component.alternatingSign_;
        std::vector<size_t> strides(shape.rank(), 1);
        const auto axes = detail::validatedIndexOrder(shape.rank(), order);
        size_t stride = 1;
        for (auto axis = axes.rbegin(); axis != axes.rend(); ++axis) {
            strides[*axis] = stride;
            stride *= shape.dimensions()[*axis];
        }

        PreparedAlternatingSign result{
            .dimensions = {},
            .repeatingNegation = {},
            .repeatMask = 0,
            .negativeWhenOdd = parameters.negativeWhenOdd,
        };
        result.dimensions.reserve(parameters.dimensions.size());
        constexpr size_t maximumRepeatSize = 65536;
        size_t repeatSize = shape.elementCount() == 0 ? 0 : 1;
        for (const size_t dimension : parameters.dimensions) {
            if (dimension >= shape.rank())
                throw std::runtime_error(
                    "Alternating-sign generation dimension exceeds tensor rank.");
            const size_t extent = shape[dimension];
            result.dimensions.push_back(
                {.stride = strides[dimension], .oddExtent = extent % 2 == 0 ? 0 : extent});
            if (extent <= 1 || repeatSize == 0) continue;
            const size_t coordinatePeriodFactor = extent % 2 == 0 ? 2 : extent;
            if (strides[dimension] > maximumRepeatSize / coordinatePeriodFactor) {
                repeatSize = 0;
                continue;
            }
            const size_t dimensionPeriod = strides[dimension] * coordinatePeriodFactor;
            const size_t commonDivisor = std::gcd(repeatSize, dimensionPeriod);
            if (repeatSize / commonDivisor > maximumRepeatSize / dimensionPeriod)
                repeatSize = 0;
            else
                repeatSize = repeatSize / commonDivisor * dimensionPeriod;
        }
        if (repeatSize != 0) {
            result.repeatingNegation.resize(repeatSize);
            for (size_t index = 0; index < repeatSize; ++index)
                result.repeatingNegation[index] = result.negatesFromDimensions(index);
            if (std::has_single_bit(repeatSize)) result.repeatMask = repeatSize - 1;
        }
        return result;
    }

    template <typename Tag, typename Pattern>
    static auto prepareEncodedDiscreteValues(const Pattern& pattern, bool negate) {
        // The random counter still selects one source candidate per logical element. Caching only
        // removes repeated conversion of that finite candidate set, so both the distribution and
        // the seed-to-output mapping remain unchanged.
        using Encoded = decltype(encodeScalarValueKnown<Tag>(0.0));
        EncodedDiscreteLookup<Encoded> lookup;
        if constexpr (std::is_same_v<Pattern, Component::ChoicePattern>) {
            if (pattern.parameters.values.size() > maximumDiscreteLookupSize) return lookup;
            for (const double value : pattern.parameters.values)
                lookup.values[lookup.size++] = encodeScalarValueKnown<Tag>(negate ? -value : value);
        } else {
            const int64_t lower = pattern.parameters.lower;
            const int64_t upper = pattern.parameters.upper;
            const uint64_t count = static_cast<uint64_t>(upper - lower + 1);
            if (count > maximumDiscreteLookupSize) return lookup;
            for (int64_t value = lower; value <= upper; ++value) {
                const double generated =
                    std::is_same_v<Pattern, Component::AbsoluteUniformIntegerPattern>
                        ? std::abs(static_cast<double>(value))
                        : static_cast<double>(value);
                lookup.values[lookup.size++] =
                    encodeScalarValueKnown<Tag>(negate ? -generated : generated);
            }
        }
        return lookup;
    }

    template <typename Tag>
    static auto prepareEncodedUniformRealLookup(const Component::UniformRealPattern& pattern) {
        // Uniform generation is monotonic in the 53 random mantissa bits. If both ends of a
        // bucket encode identically, every value in that bucket does too. Only buckets crossing
        // an encoding boundary need the full floating-point calculation at generation time.
        std::array<uint16_t, uniformRealLookupSize> values{};
        for (size_t bucket = 0; bucket < uniformRealLookupSize; ++bucket) {
            const uint64_t firstMantissa = static_cast<uint64_t>(bucket) << uniformRealLookupShift;
            const uint64_t lastMantissa =
                (static_cast<uint64_t>(bucket + 1) << uniformRealLookupShift) - 1U;
            const uint32_t first = static_cast<uint32_t>(
                encodeScalarValueKnown<Tag>(uniformRealValue(pattern, firstMantissa)));
            const uint32_t last = static_cast<uint32_t>(
                encodeScalarValueKnown<Tag>(uniformRealValue(pattern, lastMantissa)));
            values[bucket] =
                first == last ? static_cast<uint16_t>(first) : ambiguousUniformRealLookupValue;
        }
        return values;
    }

    template <typename Tag, typename Pattern>
    static bool generateContiguousNumericalType(Tensor destination, const GenerationRecipe& recipe,
                                                const GenerationRecipe::BoundComponent& bound,
                                                const Pattern& pattern) {
        const auto indexOrder = recipe.indexOrder(destination.shape().rank());
        if (!detail::isContiguous(destination.layout(), indexOrder)) return false;

        constexpr auto& info = scalarTypeInfo(Tag::type);
        constexpr bool signedFloatingPoint =
            info.category == ScalarCategory::FloatingPoint && info.isSigned;
        constexpr bool signedNumerical =
            signedFloatingPoint || info.category == ScalarCategory::SignedInteger;
        constexpr bool lowPrecisionFloatingPoint = signedFloatingPoint && info.storageBits <= 16;
        constexpr bool discretePattern =
            std::is_same_v<Pattern, Component::ChoicePattern> ||
            std::is_same_v<Pattern, Component::UniformIntegerPattern> ||
            std::is_same_v<Pattern, Component::AbsoluteUniformIntegerPattern>;
        const bool onlyAlternatingSign =
            bound.component.alternatingSign_.has_value() &&
            !bound.component.zeroOutsideMainDiagonal_ &&
            bound.component.unaryTransform_ == Component::UnaryTransform::None &&
            bound.component.affineValue_.scale == 1.0 && bound.component.affineValue_.offset == 0.0;
        if (!isUnmodifiedComponent<Pattern>(bound.component) &&
            !(signedNumerical && discretePattern && onlyAlternatingSign))
            return false;

        if constexpr (signedNumerical && discretePattern) {
            if (lowPrecisionFloatingPoint || onlyAlternatingSign) {
                const auto encodedValues = prepareEncodedDiscreteValues<Tag>(pattern, false);
                if (encodedValues.size != 0) {
                    const uint64_t seed = recipe.settings_.seed;
                    const uint64_t domain = randomDomain(bound);
                    if (onlyAlternatingSign) {
                        const auto negativeEncodedValues =
                            prepareEncodedDiscreteValues<Tag>(pattern, true);
                        const PreparedAlternatingSign alternating = prepareAlternatingSign(
                            bound.component, destination.shape(), indexOrder);
                        return generateContiguousEncodedValues<Tag>(
                            destination, indexOrder, [&](size_t logicalIndex) {
                                const size_t selected =
                                    counterRandom(seed, domain, logicalIndex) % encodedValues.size;
                                return alternating.negates(logicalIndex)
                                           ? negativeEncodedValues.values[selected]
                                           : encodedValues.values[selected];
                            });
                    }
                    return generateContiguousEncodedValues<Tag>(
                        destination, indexOrder, [&](size_t logicalIndex) {
                            return encodedValues.values[counterRandom(seed, domain, logicalIndex) %
                                                        encodedValues.size];
                        });
                }
                if (onlyAlternatingSign) return false;
            }
        }

        constexpr bool subByteOrByteFloatingPoint = signedFloatingPoint && info.storageBits <= 8;
        if constexpr (subByteOrByteFloatingPoint &&
                      std::is_same_v<Pattern, Component::UniformRealPattern>) {
            constexpr size_t minimumLookupElementCount = 16384;
            if (destination.elementCount() >= minimumLookupElementCount &&
                std::isfinite(pattern.parameters.lower) &&
                std::isfinite(pattern.parameters.upper - pattern.parameters.lower)) {
                const auto encodedValues = prepareEncodedUniformRealLookup<Tag>(pattern);
                const uint64_t seed = recipe.settings_.seed;
                const uint64_t domain = randomDomain(bound);
                return generateContiguousEncodedValues<Tag>(
                    destination, indexOrder, [&](size_t logicalIndex) {
                        const uint64_t mantissa = counterRandom(seed, domain, logicalIndex) >> 11;
                        const uint16_t encoded = encodedValues[mantissa >> uniformRealLookupShift];
                        return encoded == ambiguousUniformRealLookupValue
                                   ? encodeScalarValueKnown<Tag>(
                                         uniformRealValue(pattern, mantissa))
                                   : encoded;
                    });
            }
        }

        const auto values = prepareValues(pattern, destination.type());
        return generateContiguousEncodedValues<Tag>(
            destination, indexOrder, [&](size_t logicalIndex) {
                return generatedEncodedValue<Tag>(pattern, recipe, bound, destination.shape(),
                                                  logicalIndex, destination.type(), values);
            });
    }

    template <typename Pattern>
    static bool generateCommonNumericalType(Tensor destination, const GenerationRecipe& recipe,
                                            const GenerationRecipe::BoundComponent& bound,
                                            const Pattern& pattern) {
        return visitScalarType(destination.type(), [&]<typename Tag>() {
            if constexpr (scalarTypeInfo(Tag::type).category == ScalarCategory::Complex)
                return false;
            else
                return generateContiguousNumericalType<Tag>(destination, recipe, bound, pattern);
        });
    }

    template <typename Pattern>
    static bool generateCommonRawType(Tensor destination, const GenerationRecipe& recipe,
                                      const GenerationRecipe::BoundComponent& bound,
                                      const Pattern& pattern) {
        const auto indexOrder = recipe.indexOrder(destination.shape().rank());
        if (usesCoordinates<Pattern>(bound.component) ||
            !detail::isContiguous(destination.layout(), indexOrder))
            return false;

        return visitScalarType(destination.type(), [&]<typename Tag>() {
            constexpr ScalarType type = Tag::type;
            if constexpr (scalarTypeInfo(type).category == ScalarCategory::Complex ||
                          scalarTypeInfo(type).storageBits > 64) {
                return false;
            } else {
                const uint64_t seed = recipe.settings_.seed;
                const uint64_t domain = randomDomain(bound);
                if constexpr (std::is_same_v<Pattern,
                                             Component::UniformFiniteEncodedValuePattern>) {
                    const std::span<const uint8_t> candidates =
                        finiteEncodedValues(destination.type());
                    return generateContiguousEncodedValues<Tag>(
                        destination, indexOrder, [&](size_t logicalIndex) {
                            return rawStorageValue<Tag>(
                                candidates[counterRandom(seed, domain, logicalIndex) %
                                           candidates.size()]);
                        });
                }
                const auto values = prepareValues(pattern, type);
                return generateContiguousEncodedValues<Tag>(
                    destination, indexOrder, [&](size_t logicalIndex) {
                        return rawStorageValue<Tag>(rawGenerationValueKnown<false>(
                            pattern, seed, domain, {}, destination.shape(), logicalIndex, type,
                            &values));
                    });
            }
        });
    }

    static void generateNonComplex(Tensor destination, const GenerationRecipe& recipe,
                                   const GenerationRecipe::BoundComponent& bound) {
        const auto indexOrder = recipe.indexOrder(destination.shape().rank());
        const auto storage = destination.rawEncodedBackingStorage();
        std::visit(
            [&](const auto& pattern) {
                using Pattern = std::remove_cvref_t<decltype(pattern)>;
                const auto values = prepareValues(pattern, destination.type());
                if constexpr (isRawPattern<Pattern>) {
                    const RawWriter write = rawWriter(destination.type());
                    if constexpr (std::is_same_v<Pattern, Component::RawConstantPattern>) {
                        if (fillContiguous(
                                destination, indexOrder,
                                [&](std::span<std::byte> destinationStorage, ptrdiff_t offset) {
                                    write(destinationStorage, offset, pattern.parameters.bits);
                                }))
                            return;
                    }
                    if (generateCommonRawType(destination, recipe, bound, pattern)) return;
                    forEachGenerationElement(
                        destination, indexOrder, usesCoordinates<Pattern>(bound.component),
                        [&](std::span<const size_t> indices, size_t logicalIndex,
                            ptrdiff_t offset) {
                            write(storage, offset,
                                  rawGenerationValueKnown<false>(pattern, recipe.settings_.seed,
                                                                 randomDomain(bound), indices,
                                                                 destination.shape(), logicalIndex,
                                                                 destination.type(), &values));
                        });
                } else {
                    const NumericalWriter write = numericalWriter(destination.type());
                    if constexpr (isConstantPattern<Pattern>) {
                        if (!usesCoordinates<Pattern>(bound.component) &&
                            fillContiguous(
                                destination, indexOrder,
                                [&](std::span<std::byte> destinationStorage, ptrdiff_t offset) {
                                    write(destinationStorage, offset,
                                          generationValueKnown<false>(
                                              bound.component, pattern, recipe.settings_.seed,
                                              randomDomain(bound), {}, destination.shape(), 0,
                                              destination.type(), &values));
                                }))
                            return;
                    }
                    if (generateCommonNumericalType(destination, recipe, bound, pattern)) return;
                    forEachGenerationElement(
                        destination, indexOrder, usesCoordinates<Pattern>(bound.component),
                        [&](std::span<const size_t> indices, size_t logicalIndex,
                            ptrdiff_t offset) {
                            write(storage, offset,
                                  generationValueKnown<false>(
                                      bound.component, pattern, recipe.settings_.seed,
                                      randomDomain(bound), indices, destination.shape(),
                                      logicalIndex, destination.type(), &values));
                        });
                }
            },
            bound.component.pattern_);
    }

    enum class ComplexGenerationKind { RealOnly, Replicated, Cartesian };

    template <typename Storage>
    static bool generateContiguousComplexValues(Tensor destination, const GenerationRecipe& recipe,
                                                const PreparedNumericalGenerator& real,
                                                const PreparedNumericalGenerator* imaginary,
                                                ComplexGenerationKind kind) {
        const auto indexOrder = recipe.indexOrder(destination.shape().rank());
        if (!detail::isContiguous(destination.layout(), indexOrder)) return false;
        const auto storage = destination.rawEncodedBackingStorage();
        const size_t byteOffset =
            static_cast<size_t>(destination.layout().offset()) * sizeof(Storage);
        if (reinterpret_cast<uintptr_t>(storage.data() + byteOffset) % alignof(Storage) != 0)
            return false;
        auto* output = reinterpret_cast<Storage*>(storage.data() + byteOffset);
        const bool usesCoordinates =
            real.usesCoordinates || (imaginary != nullptr && imaginary->usesCoordinates);
        forEachGenerationElement(
            destination, indexOrder, usesCoordinates,
            [&](std::span<const size_t> indices, size_t logicalIndex, ptrdiff_t) {
                const double realValue = real(indices, destination.shape(), logicalIndex);
                const double imaginaryValue =
                    kind == ComplexGenerationKind::RealOnly ? 0.0
                    : kind == ComplexGenerationKind::Replicated
                        ? realValue
                        : (*imaginary)(indices, destination.shape(), logicalIndex);
                output[logicalIndex] = Storage(realValue, imaginaryValue);
            });
        return true;
    }

    static bool generateCommonComplexType(Tensor destination, const GenerationRecipe& recipe,
                                          const PreparedNumericalGenerator& real,
                                          const PreparedNumericalGenerator* imaginary,
                                          ComplexGenerationKind kind) {
        switch (destination.type()) {
            case ScalarType::ComplexFloat32:
                return generateContiguousComplexValues<std::complex<float>>(destination, recipe,
                                                                            real, imaginary, kind);
            case ScalarType::ComplexFloat64:
                return generateContiguousComplexValues<std::complex<double>>(destination, recipe,
                                                                             real, imaginary, kind);
            default:
                return false;
        }
    }

    static void generateComplex(Tensor destination, const GenerationRecipe& recipe) {
        const auto indexOrder = recipe.indexOrder(destination.shape().rank());
        const auto storage = destination.rawEncodedBackingStorage();
        const ComplexWriter write = complexWriter(destination.type());
        std::visit(
            [&](const auto& policy) {
                using Policy = std::remove_cvref_t<decltype(policy)>;
                if constexpr (std::is_same_v<Policy, GenerationRecipe::RealOnlyPolicy>) {
                    PreparedNumericalGenerator real = prepareNumericalComponent(
                        policy.real, recipe.settings_.seed, destination.type());
                    real.indexOffset = recipe.indexOffset();
                    if (real.isConstant &&
                        fillContiguous(
                            destination, indexOrder,
                            [&](std::span<std::byte> destinationStorage, ptrdiff_t offset) {
                                write(destinationStorage, offset,
                                      {real({}, destination.shape(), 0), 0.0});
                            }))
                        return;
                    if (generateCommonComplexType(destination, recipe, real, nullptr,
                                                  ComplexGenerationKind::RealOnly))
                        return;
                    forEachGenerationElement(destination, indexOrder, real.usesCoordinates,
                                             [&](std::span<const size_t> indices,
                                                 size_t logicalIndex, ptrdiff_t offset) {
                                                 const double value = real(
                                                     indices, destination.shape(), logicalIndex);
                                                 write(storage, offset, {value, 0.0});
                                             });
                } else if constexpr (std::is_same_v<Policy, GenerationRecipe::ReplicatedPolicy>) {
                    PreparedNumericalGenerator value = prepareNumericalComponent(
                        policy.value, recipe.settings_.seed, destination.type());
                    value.indexOffset = recipe.indexOffset();
                    if (value.isConstant &&
                        fillContiguous(
                            destination, indexOrder,
                            [&](std::span<std::byte> destinationStorage, ptrdiff_t offset) {
                                const double component = value({}, destination.shape(), 0);
                                write(destinationStorage, offset, {component, component});
                            }))
                        return;
                    if (generateCommonComplexType(destination, recipe, value, nullptr,
                                                  ComplexGenerationKind::Replicated))
                        return;
                    forEachGenerationElement(destination, indexOrder, value.usesCoordinates,
                                             [&](std::span<const size_t> indices,
                                                 size_t logicalIndex, ptrdiff_t offset) {
                                                 const double component = value(
                                                     indices, destination.shape(), logicalIndex);
                                                 write(storage, offset, {component, component});
                                             });
                } else {
                    PreparedNumericalGenerator real = prepareNumericalComponent(
                        policy.real, recipe.settings_.seed, destination.type());
                    real.indexOffset = recipe.indexOffset();
                    PreparedNumericalGenerator imaginary = prepareNumericalComponent(
                        policy.imaginary, recipe.settings_.seed, destination.type());
                    imaginary.indexOffset = recipe.indexOffset();
                    if (real.isConstant && imaginary.isConstant &&
                        fillContiguous(
                            destination, indexOrder,
                            [&](std::span<std::byte> destinationStorage, ptrdiff_t offset) {
                                write(destinationStorage, offset,
                                      {real({}, destination.shape(), 0),
                                       imaginary({}, destination.shape(), 0)});
                            }))
                        return;
                    if (generateCommonComplexType(destination, recipe, real, &imaginary,
                                                  ComplexGenerationKind::Cartesian))
                        return;
                    forEachGenerationElement(
                        destination, indexOrder, real.usesCoordinates || imaginary.usesCoordinates,
                        [&](std::span<const size_t> indices, size_t logicalIndex,
                            ptrdiff_t offset) {
                            write(storage, offset,
                                  {real(indices, destination.shape(), logicalIndex),
                                   imaginary(indices, destination.shape(), logicalIndex)});
                        });
                }
            },
            recipe.complexPolicy_);
    }

    static void validateComponent(const Component& component, const Shape& shape, ScalarType type) {
        std::vector<size_t> coordinates(shape.rank(), 0);
        if (component.isRaw()) {
            (void)rawGenerationValue(component, 0, 0, coordinates, shape, 0, type);
            return;
        }
        (void)generationValue(component, 0, 0, coordinates, shape, 0, type);
        std::visit(
            [&](const auto& pattern) {
                using Pattern = std::remove_cvref_t<decltype(pattern)>;
                if constexpr (std::is_same_v<Pattern, Component::AffineIndexRemainderPattern>) {
                    for (bool positive : {false, true}) {
                        for (size_t axis = 0; axis < shape.rank(); ++axis) {
                            if (shape.dimensions()[axis] - 1 >
                                static_cast<size_t>(std::numeric_limits<int64_t>::max()))
                                throw std::runtime_error("Affine-index coordinate exceeds Int64.");
                            coordinates[axis] =
                                (pattern.parameters.dimensionCoefficients[axis] > 0) == positive
                                    ? shape.dimensions()[axis] - 1
                                    : 0;
                        }
                        (void)baseGenerationValueKnown(pattern, 0, 0, coordinates, shape, 0, type);
                    }
                }
            },
            component.pattern_);
    }

    static void validateRecipe(const GenerationRecipe& recipe, const Shape& shape,
                               ScalarType type) {
        (void)validatedIndexOrder(shape.rank(), recipe.indexOrder(shape.rank()));
        if (shape.elementCount() == 0) return;
        if (recipe.settings_.indexOffset >
            std::numeric_limits<size_t>::max() - (shape.elementCount() - 1))
            throw std::runtime_error("Generation sample index offset overflows size_t.");
        std::visit(
            [&](const auto& policy) {
                using Policy = std::remove_cvref_t<decltype(policy)>;
                if constexpr (std::is_same_v<Policy, GenerationRecipe::ReplicatedPolicy>)
                    validateComponent(policy.value.component, shape, type);
                else {
                    validateComponent(policy.real.component, shape, type);
                    if constexpr (std::is_same_v<Policy, GenerationRecipe::CartesianPolicy>)
                        validateComponent(policy.imaginary.component, shape, type);
                }
            },
            recipe.complexPolicy_);
    }

    static void generateTensor(Tensor destination, const GenerationRecipe& recipe,
                               const ScalarConversionOptions& options) {
        validateRecipe(recipe, destination.shape(), destination.type());
        if (destination.elementCount() == 0) return;
        const auto order = recipe.indexOrder(destination.shape().rank());
        if (scalarTypeInfo(destination.type()).category == ScalarCategory::Complex) {
            generateComplex(destination, recipe);
        } else if (realComponent(recipe).component.isRaw()) {
            auto generator =
                prepareRawComponent(realComponent(recipe), recipe.seed(), destination.type());
            generator.indexOffset = recipe.indexOffset();
            const auto writer = rawWriter(destination.type());
            forEachGenerationElement(
                destination, order, generator.usesCoordinates,
                [&](std::span<const size_t> indices, size_t index, ptrdiff_t offset) {
                    writer(destination.rawEncodedBackingStorage(), offset,
                           generator(indices, destination.shape(), index));
                });
        } else {
            auto generator =
                prepareNumericalComponent(realComponent(recipe), recipe.seed(), destination.type());
            generator.indexOffset = recipe.indexOffset();
            using Writer =
                void (*)(std::span<std::byte>, ptrdiff_t, double, const ScalarConversionOptions&);
            const auto writer = visitScalarType(destination.type(), []<typename Tag>() -> Writer {
                return [](std::span<std::byte> storage, ptrdiff_t offset, double value,
                          const ScalarConversionOptions& conversion) {
                    preparedEncodeScalarKnown<Tag>(storage, offset, value, conversion);
                };
            });
            forEachGenerationElement(
                destination, order, generator.usesCoordinates,
                [&](std::span<const size_t> indices, size_t index, ptrdiff_t offset) {
                    writer(destination.rawEncodedBackingStorage(), offset,
                           generator(indices, destination.shape(), index), options);
                });
        }
    }

    static void generateTensor(Tensor destination, const GenerationRecipe& recipe) {
        if (recipe.indexOffset() != 0) {
            generateTensor(destination, recipe,
                           implicitStorageConversionOptions(destination.type()));
            return;
        }
        validateRecipe(recipe, destination.shape(), destination.type());
        if (destination.elementCount() == 0) return;
        if (scalarTypeInfo(destination.type()).category != ScalarCategory::Complex) {
            generateNonComplex(destination, recipe, realComponent(recipe));
            return;
        }
        generateComplex(std::move(destination), recipe);
    }
};

bool isRawGenerationComponent(const GenerationRecipe::Component& component) {
    return GenerationRecipeAccess::isRawComponent(component);
}

PreparedNumericalGenerator prepareNumericalGenerator(const GenerationRecipe& recipe,
                                                     ScalarType destinationType,
                                                     const Shape& shape) {
    GenerationRecipeAccess::validateRecipe(recipe, shape, destinationType);
    auto result = GenerationRecipeAccess::prepareNumericalComponent(
        GenerationRecipeAccess::realComponent(recipe), recipe.seed(), destinationType);
    result.indexOffset = recipe.indexOffset();
    return result;
}

PreparedRawGenerator prepareRawGenerator(const GenerationRecipe& recipe, ScalarType destinationType,
                                         const Shape& shape) {
    GenerationRecipeAccess::validateRecipe(recipe, shape, destinationType);
    auto result = GenerationRecipeAccess::prepareRawComponent(
        GenerationRecipeAccess::realComponent(recipe), recipe.seed(), destinationType);
    result.indexOffset = recipe.indexOffset();
    return result;
}

double generatedNumericalValue(const GenerationRecipe& recipe, std::span<const size_t> indices,
                               const Shape& shape, size_t logicalIndex,
                               ScalarType destinationType) {
    return GenerationRecipeAccess::generatedNumericalValue(recipe, indices, shape, logicalIndex,
                                                           destinationType);
}

uint64_t generatedRawValue(const GenerationRecipe& recipe, std::span<const size_t> indices,
                           const Shape& shape, size_t logicalIndex, ScalarType destinationType) {
    return GenerationRecipeAccess::generatedRawValue(recipe, indices, shape, logicalIndex,
                                                     destinationType);
}

void generateElement(Tensor destination, const GenerationRecipe& recipe,
                     std::span<const size_t> indices, size_t logicalIndex) {
    GenerationRecipeAccess::generateElement(destination, recipe, indices, logicalIndex);
}

void generateTensor(Tensor destination, const GenerationRecipe& recipe) {
    GenerationRecipeAccess::generateTensor(std::move(destination), recipe);
}
void generateTensor(Tensor destination, const GenerationRecipe& recipe,
                    const ScalarConversionOptions& options) {
    GenerationRecipeAccess::generateTensor(std::move(destination), recipe, options);
}
void generateElement(Tensor destination, const GenerationRecipe& recipe,
                     std::span<const size_t> indices, size_t logicalIndex,
                     const ScalarConversionOptions& options) {
    GenerationRecipeAccess::generateElement(destination, recipe, indices, logicalIndex, &options);
}
}  // namespace roc::hostnumerics::detail
