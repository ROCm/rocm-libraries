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
#include <roc/host_numerics/generation.hpp>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include "detail/generation_primitives.hpp"
#include "detail/generation_values.hpp"
#include "detail/layout_iteration.hpp"
#include "detail/threading.hpp"

namespace roc::host_numerics::detail {
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

    template <typename Pattern>
    static uint64_t rawGenerationValueKnown(const Pattern& pattern, uint64_t seed, uint64_t domain,
                                            std::span<const size_t> indices, const Shape& shape,
                                            size_t logicalIndex, ScalarType destinationType) {
        if constexpr (std::is_same_v<Pattern, Component::RawConstantPattern>) {
            return pattern.parameters.bits;
        } else if constexpr (std::is_same_v<Pattern, Component::UniformRawIntegerPattern>) {
            return static_cast<uint64_t>(static_cast<int64_t>(indexedUniformInteger(
                seed, domain, logicalIndex, pattern.parameters.lower, pattern.parameters.upper)));
        } else if constexpr (std::is_same_v<Pattern, Component::UniformFiniteEncodedValuePattern>) {
            return uniformFiniteEncodedValue(destinationType, seed, domain, logicalIndex);
        } else if constexpr (std::is_same_v<Pattern, Component::RandomRawBitsPattern>) {
            return counterRandom(seed, domain, logicalIndex);
        } else if constexpr (std::is_same_v<Pattern, Component::RawSerialDimensionPattern>) {
            if (pattern.parameters.dimension >= shape.rank())
                throw std::out_of_range("Generation dimension exceeds tensor rank.");
            return static_cast<uint64_t>(indices[pattern.parameters.dimension]);
        } else {
            throw std::invalid_argument(
                "Requested generation pattern does not produce raw storage.");
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

    template <typename Pattern>
    static double baseGenerationValueKnown(const Pattern& pattern, uint64_t seed, uint64_t domain,
                                           std::span<const size_t> indices, const Shape& shape,
                                           size_t logicalIndex, ScalarType destinationType) {
        if constexpr (std::is_same_v<Pattern, Component::ZeroPattern>) {
            return 0.0;
        } else if constexpr (std::is_same_v<Pattern, Component::ConstantPattern>) {
            return pattern.parameters.value;
        } else if constexpr (std::is_same_v<Pattern, Component::ChoicePattern>) {
            if (pattern.parameters.values.empty())
                throw std::invalid_argument("Choice generation requires values.");
            return pattern.parameters.values[counterRandom(seed, domain, logicalIndex) %
                                             pattern.parameters.values.size()];
        } else if constexpr (std::is_same_v<Pattern, Component::UniformIntegerPattern>) {
            return static_cast<double>(indexedUniformInteger(
                seed, domain, logicalIndex, pattern.parameters.lower, pattern.parameters.upper));
        } else if constexpr (std::is_same_v<Pattern, Component::AbsoluteUniformIntegerPattern>) {
            return std::abs(static_cast<double>(indexedUniformInteger(
                seed, domain, logicalIndex, pattern.parameters.lower, pattern.parameters.upper)));
        } else if constexpr (std::is_same_v<Pattern, Component::UniformRealPattern>) {
            if (pattern.parameters.lower > pattern.parameters.upper)
                throw std::invalid_argument("Uniform-real lower bound exceeds upper bound.");
            return uniformRealValue(pattern, counterRandom(seed, domain, logicalIndex) >> 11);
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
            if (pattern.parameters.dimension >= shape.rank())
                throw std::out_of_range("Generation dimension exceeds tensor rank.");
            return static_cast<double>(indices[pattern.parameters.dimension]);
        } else if constexpr (std::is_same_v<Pattern, Component::AffineIndexRemainderPattern>) {
            const auto& parameters = pattern.parameters;
            if (parameters.dimensionCoefficients.size() != shape.rank())
                throw std::invalid_argument(
                    "Affine-index coefficient count must match the tensor rank.");
            if (parameters.positiveDivisor <= 0)
                throw std::invalid_argument("Affine-index remainder divisor must be positive.");

            int64_t value = parameters.offset;
            for (size_t dimension = 0; dimension < shape.rank(); ++dimension) {
                const int64_t coefficient = parameters.dimensionCoefficients[dimension];
                if (indices[dimension] > static_cast<size_t>(std::numeric_limits<int64_t>::max()))
                    throw std::overflow_error("Affine-index coordinate exceeds Int64.");
                const int64_t index = static_cast<int64_t>(indices[dimension]);
                int64_t term = 0;
                if (coefficient > 0) {
                    if (index > std::numeric_limits<int64_t>::max() / coefficient)
                        throw std::overflow_error("Affine-index multiplication overflow.");
                    term = coefficient * index;
                } else if (coefficient < 0) {
                    if (coefficient == std::numeric_limits<int64_t>::min()) {
                        if (index > 1)
                            throw std::overflow_error("Affine-index multiplication overflow.");
                        term = index == 0 ? 0 : std::numeric_limits<int64_t>::min();
                    } else {
                        const int64_t magnitude = -coefficient;
                        if (index > std::numeric_limits<int64_t>::max() / magnitude)
                            throw std::overflow_error("Affine-index multiplication overflow.");
                        term = -(magnitude * index);
                    }
                }

                if ((term > 0 && value > std::numeric_limits<int64_t>::max() - term) ||
                    (term < 0 && value < std::numeric_limits<int64_t>::min() - term))
                    throw std::overflow_error("Affine-index addition overflow.");
                value += term;
            }
            return static_cast<double>(value % parameters.positiveDivisor);
        } else if constexpr (std::is_same_v<Pattern, Component::IdentityPattern>) {
            if (shape.rank() < 2)
                throw std::invalid_argument("Identity generation requires rank at least two.");
            return indices[0] == indices[1] ? 1.0 : 0.0;
        } else if constexpr (std::is_same_v<Pattern,
                                            Component::CheckerboardUniformIntegerPattern>) {
            double value = static_cast<double>(indexedUniformInteger(
                seed, domain, logicalIndex, pattern.parameters.lower, pattern.parameters.upper));
            size_t parity = 0;
            for (const size_t index : indices) parity ^= index;
            return (parity & 1U) == 0 ? -value : value;
        } else if constexpr (std::is_same_v<Pattern, Component::TypeMaximumPattern>) {
            return typeMaximum(destinationType);
        } else if constexpr (std::is_same_v<Pattern, Component::TypeLowestPattern>) {
            return typeLowest(destinationType);
        } else if constexpr (std::is_same_v<Pattern, Component::TypeDenormalMinimumPattern>) {
            return typeDenormalMinimum(destinationType);
        } else if constexpr (std::is_same_v<Pattern, Component::TypeDenormalMaximumPattern>) {
            return typeDenormalMaximum(destinationType);
        } else if constexpr (std::is_same_v<Pattern, Component::TypeNaNPattern>) {
            return typeNaN(destinationType);
        } else if constexpr (std::is_same_v<Pattern, Component::TypeInfinityPattern>) {
            return typeInfinity(destinationType, false);
        } else if constexpr (std::is_same_v<Pattern, Component::TypeNegativeInfinityPattern>) {
            return typeInfinity(destinationType, true);
        } else if constexpr (std::is_same_v<Pattern, Component::TypeNegativeZeroPattern>) {
            return -0.0;
        } else if constexpr (std::is_same_v<Pattern, Component::UniformTypeRangePattern>) {
            const double maximum = typeMaximum(destinationType);
            const double unit = indexedUniformUnit(seed, domain, logicalIndex);
            return maximum * (2.0 * unit - 1.0);
        } else if constexpr (std::is_same_v<Pattern, Component::RandomEncodedExponentPattern>) {
            return randomEncodedExponentValue(pattern.parameters, seed, domain, logicalIndex,
                                              destinationType);
        } else {
            throw std::invalid_argument("Raw generation requires encoded storage output.");
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

    template <typename Pattern>
    static double generationValueKnown(const Component& component, const Pattern& pattern,
                                       uint64_t seed, uint64_t domain,
                                       std::span<const size_t> indices, const Shape& shape,
                                       size_t logicalIndex, ScalarType destinationType) {
        if (component.zeroOutsideMainDiagonal_) {
            if (shape.rank() < 2)
                throw std::invalid_argument("Main-diagonal generation requires rank at least two.");
            if (indices[0] != indices[1]) return 0.0;
        }
        double value = baseGenerationValueKnown(pattern, seed, domain, indices, shape, logicalIndex,
                                                destinationType);
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
        }
        value = value * component.affineValue_.scale + component.affineValue_.offset;

        if (component.alternatingSign_.has_value()) {
            size_t parity = 0;
            for (const size_t dimension : component.alternatingSign_->dimensions) {
                if (dimension >= shape.rank())
                    throw std::out_of_range(
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
            throw std::invalid_argument(
                "Raw generation recipe does not produce a numerical value.");
        return generationValue(component.component, recipe.settings_.seed, randomDomain(component),
                               indices, shape, logicalIndex, destinationType);
    }

    static uint64_t generatedRawValue(const GenerationRecipe& recipe,
                                      std::span<const size_t> indices, const Shape& shape,
                                      size_t logicalIndex, ScalarType destinationType) {
        const GenerationRecipe::BoundComponent& component = realComponent(recipe);
        if (!component.component.isRaw())
            throw std::invalid_argument("Numerical generation recipe does not produce raw bits.");
        return rawGenerationValue(component.component, recipe.settings_.seed,
                                  randomDomain(component), indices, shape, logicalIndex,
                                  destinationType);
    }

    static void writeRawGeneration(Tensor destination, const GenerationRecipe& recipe,
                                   const GenerationRecipe::BoundComponent& bound,
                                   std::span<const size_t> indices, size_t logicalIndex) {
        const uint16_t bits = scalarTypeInfo(destination.type()).storageBits;
        if (bits > 64)
            throw std::invalid_argument("Raw generation supports scalar encodings up to 64 bits.");

        const ptrdiff_t elementOffset = destination.layout().elementOffset(indices);
        const uint64_t raw =
            rawGenerationValue(bound.component, recipe.settings_.seed, randomDomain(bound), indices,
                               destination.shape(), logicalIndex, destination.type());
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
                                std::span<const size_t> indices, size_t logicalIndex) {
        const bool complexOutput =
            scalarTypeInfo(destination.type()).category == ScalarCategory::Complex;
        const GenerationRecipe::BoundComponent* real = &realComponent(recipe);

        if (!complexOutput) {
            if (real->component.isRaw()) {
                writeRawGeneration(destination, recipe, *real, indices, logicalIndex);
                return;
            }
            destination.storeFrom(
                indices, generatedNumericalValue(recipe, indices, destination.shape(), logicalIndex,
                                                 destination.type()));
            return;
        }

        std::visit(
            [&](const auto& policy) {
                using Policy = std::remove_cvref_t<decltype(policy)>;
                if constexpr (std::is_same_v<Policy, GenerationRecipe::RealOnlyPolicy>) {
                    if (policy.real.component.isRaw())
                        throw std::invalid_argument(
                            "Raw generation does not support complex output.");
                    const double value = generationValue(
                        policy.real.component, recipe.settings_.seed, randomDomain(policy.real),
                        indices, destination.shape(), logicalIndex, destination.type());
                    destination.storeFrom(indices, std::complex<double>(value, 0.0));
                } else if constexpr (std::is_same_v<Policy, GenerationRecipe::ReplicatedPolicy>) {
                    if (policy.value.component.isRaw())
                        throw std::invalid_argument(
                            "Raw generation does not support complex output.");
                    const double value = generationValue(
                        policy.value.component, recipe.settings_.seed, randomDomain(policy.value),
                        indices, destination.shape(), logicalIndex, destination.type());
                    destination.storeFrom(indices, std::complex<double>(value, value));
                } else {
                    if (policy.real.component.isRaw() || policy.imaginary.component.isRaw())
                        throw std::invalid_argument(
                            "Raw generation does not support complex output.");
                    const double realValue = generationValue(
                        policy.real.component, recipe.settings_.seed, randomDomain(policy.real),
                        indices, destination.shape(), logicalIndex, destination.type());
                    const double imaginaryValue =
                        generationValue(policy.imaginary.component, recipe.settings_.seed,
                                        randomDomain(policy.imaginary), indices,
                                        destination.shape(), logicalIndex, destination.type());
                    destination.storeFrom(indices, std::complex<double>(realValue, imaginaryValue));
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
                encodeScalarKnown<Tag>(storage, offset, value);
            };
        });
    }

    static ComplexWriter complexWriter(ScalarType type) {
        return visitScalarType(type, []<typename Tag>() -> ComplexWriter {
            return [](std::span<std::byte> storage, ptrdiff_t offset, std::complex<double> value) {
                encodeScalarKnown<Tag>(storage, offset, value);
            };
        });
    }

    static RawWriter rawWriter(ScalarType type) {
        return visitScalarType(type, []<typename Tag>() -> RawWriter {
            return [](std::span<std::byte> storage, ptrdiff_t offset, uint64_t value) {
                constexpr uint16_t bits = scalarTypeInfo(Tag::type).storageBits;
                if constexpr (bits > 64) {
                    throw std::invalid_argument(
                        "Raw generation supports scalar encodings up to 64 bits.");
                } else if constexpr (bits <= 32) {
                    writePackedBits(storage, bitOffset(Tag::type, offset), bits,
                                    static_cast<uint32_t>(value));
                } else {
                    writeNative<uint64_t>(
                        storage, static_cast<size_t>(bitOffset(Tag::type, offset) / 8), value);
                }
            };
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
    static double preparedBaseValue(const Component&, const void* erasedPattern, uint64_t seed,
                                    uint64_t domain, std::span<const size_t> indices,
                                    const Shape& shape, size_t logicalIndex,
                                    ScalarType destinationType) {
        return baseGenerationValueKnown(*static_cast<const Pattern*>(erasedPattern), seed, domain,
                                        indices, shape, logicalIndex, destinationType);
    }

    template <typename Pattern>
    static double preparedModifiedValue(const Component& component, const void* erasedPattern,
                                        uint64_t seed, uint64_t domain,
                                        std::span<const size_t> indices, const Shape& shape,
                                        size_t logicalIndex, ScalarType destinationType) {
        return generationValueKnown(component, *static_cast<const Pattern*>(erasedPattern), seed,
                                    domain, indices, shape, logicalIndex, destinationType);
    }

    template <typename Pattern>
    static uint64_t preparedRawValue(const void* erasedPattern, uint64_t seed, uint64_t domain,
                                     std::span<const size_t> indices, const Shape& shape,
                                     size_t logicalIndex, ScalarType destinationType) {
        return rawGenerationValueKnown(*static_cast<const Pattern*>(erasedPattern), seed, domain,
                                       indices, shape, logicalIndex, destinationType);
    }

    static PreparedNumericalGenerator prepareNumericalComponent(
        const GenerationRecipe::BoundComponent& bound, uint64_t seed, ScalarType destinationType) {
        if (bound.component.isRaw())
            throw std::invalid_argument("Raw generation does not support complex output.");
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
            throw std::invalid_argument("Numerical generation recipe does not produce raw bits.");
        return std::visit(
            [&](const auto& pattern) {
                using Pattern = std::remove_cvref_t<decltype(pattern)>;
                return PreparedRawGenerator{
                    .pattern = &pattern,
                    .seed = seed,
                    .domain = randomDomain(bound),
                    .destinationType = destinationType,
                    .value = preparedRawValue<Pattern>,
                    .usesCoordinates = usesCoordinates<Pattern>(bound.component),
                };
            },
            bound.component.pattern_);
    }

    static void incrementCoordinates(std::vector<size_t>& indices, const Shape& shape,
                                     IndexOrder order) {
        const auto incrementDimension = [&](size_t dimension) {
            if (++indices[dimension] < shape[dimension]) return true;
            indices[dimension] = 0;
            return false;
        };

        if (order == IndexOrder::FirstDimensionFastest) {
            for (size_t dimension = 0; dimension < shape.rank(); ++dimension)
                if (incrementDimension(dimension)) return;
        } else {
            for (size_t dimension = shape.rank(); dimension > 0; --dimension)
                if (incrementDimension(dimension - 1)) return;
        }
    }

    template <typename WriteFirst>
    static bool fillContiguous(Tensor destination, IndexOrder order, WriteFirst&& writeFirst) {
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
    static void forEachGenerationElement(Tensor destination, IndexOrder logicalOrder,
                                         bool valueUsesCoordinates, Function&& function) {
        const size_t elementCount = destination.shape().elementCount();
        const bool independent = hasProvablyIndependentElements(destination);
        const IndexOrder traversalOrder =
            independent ? logicalOrder : IndexOrder::LastDimensionFastest;
        const bool contiguous = detail::isContiguous(destination.layout(), traversalOrder);
        const bool trackCoordinates =
            valueUsesCoordinates || !contiguous || traversalOrder != logicalOrder;
        const int threadCount = independent ? operationThreadCount(elementCount) : 1;

        const auto runRange = [&](size_t first, size_t end) {
            if (first == end) return;
            std::vector<size_t> indices;
            if (trackCoordinates) indices = destination.shape().coordinates(first, traversalOrder);
            for (size_t traversalIndex = first; traversalIndex < end; ++traversalIndex) {
                const size_t logicalIndex =
                    traversalOrder == logicalOrder
                        ? traversalIndex
                        : destination.shape().linearIndex(indices, logicalOrder);
                const ptrdiff_t offset = contiguous ? destination.layout().offset() +
                                                          static_cast<ptrdiff_t>(traversalIndex)
                                                    : destination.layout().elementOffset(indices);
                function(std::span<const size_t>(indices), logicalIndex, offset);
                if (trackCoordinates)
                    incrementCoordinates(indices, destination.shape(), traversalOrder);
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
#pragma omp critical(roc_host_numerics_generation_error)
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
                                      ScalarType destinationType) {
        constexpr ScalarCategory category = scalarTypeInfo(Tag::type).category;
        constexpr bool integerDestination = category == ScalarCategory::Boolean ||
                                            category == ScalarCategory::SignedInteger ||
                                            category == ScalarCategory::UnsignedInteger;
        if constexpr (integerDestination &&
                      std::is_same_v<Pattern, Component::UniformIntegerPattern>) {
            return encodeGeneratedInteger<Tag>(
                indexedUniformInteger(recipe.settings_.seed, randomDomain(bound), logicalIndex,
                                      pattern.parameters.lower, pattern.parameters.upper));
        } else if constexpr (integerDestination &&
                             std::is_same_v<Pattern, Component::SerialIndexPattern>) {
            return encodeGeneratedInteger<Tag>(logicalIndex);
        } else {
            return encodeScalarValueKnown<Tag>(
                baseGenerationValueKnown(pattern, recipe.settings_.seed, randomDomain(bound), {},
                                         shape, logicalIndex, destinationType));
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
    static bool generateContiguousPacked(Tensor destination, IndexOrder order,
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
    static bool generateContiguousEncodedValues(Tensor destination, IndexOrder order,
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
                                                          const Shape& shape, IndexOrder order) {
        const auto& parameters = *component.alternatingSign_;
        std::vector<size_t> strides(shape.rank(), 1);
        if (order == IndexOrder::FirstDimensionFastest) {
            for (size_t dimension = 1; dimension < shape.rank(); ++dimension)
                strides[dimension] = strides[dimension - 1] * shape[dimension - 1];
        } else {
            for (size_t dimension = shape.rank(); dimension > 1; --dimension)
                strides[dimension - 2] = strides[dimension - 1] * shape[dimension - 1];
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
                throw std::out_of_range(
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
    static auto prepareEncodedDiscreteValues(const Pattern& pattern, bool negate = false) {
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
        if (!detail::isContiguous(destination.layout(), recipe.settings_.indexOrder)) return false;

        constexpr ScalarCategory category = scalarTypeInfo(Tag::type).category;
        constexpr bool signedNumerical =
            category == ScalarCategory::FloatingPoint || category == ScalarCategory::SignedInteger;
        constexpr bool lowPrecisionFloatingPoint = category == ScalarCategory::FloatingPoint &&
                                                   scalarTypeInfo(Tag::type).storageBits <= 16;
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
                const auto encodedValues = prepareEncodedDiscreteValues<Tag>(pattern);
                if (encodedValues.size != 0) {
                    const uint64_t seed = recipe.settings_.seed;
                    const uint64_t domain = randomDomain(bound);
                    if (onlyAlternatingSign) {
                        const auto negativeEncodedValues =
                            prepareEncodedDiscreteValues<Tag>(pattern, true);
                        const PreparedAlternatingSign alternating = prepareAlternatingSign(
                            bound.component, destination.shape(), recipe.settings_.indexOrder);
                        return generateContiguousEncodedValues<Tag>(
                            destination, recipe.settings_.indexOrder, [&](size_t logicalIndex) {
                                const size_t selected =
                                    counterRandom(seed, domain, logicalIndex) % encodedValues.size;
                                return alternating.negates(logicalIndex)
                                           ? negativeEncodedValues.values[selected]
                                           : encodedValues.values[selected];
                            });
                    }
                    return generateContiguousEncodedValues<Tag>(
                        destination, recipe.settings_.indexOrder, [&](size_t logicalIndex) {
                            return encodedValues.values[counterRandom(seed, domain, logicalIndex) %
                                                        encodedValues.size];
                        });
                }
                if (onlyAlternatingSign) return false;
            }
        }

        constexpr bool subByteOrByteFloatingPoint =
            category == ScalarCategory::FloatingPoint && scalarTypeInfo(Tag::type).storageBits <= 8;
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
                    destination, recipe.settings_.indexOrder, [&](size_t logicalIndex) {
                        const uint64_t mantissa = counterRandom(seed, domain, logicalIndex) >> 11;
                        const uint16_t encoded = encodedValues[mantissa >> uniformRealLookupShift];
                        return encoded == ambiguousUniformRealLookupValue
                                   ? encodeScalarValueKnown<Tag>(
                                         uniformRealValue(pattern, mantissa))
                                   : encoded;
                    });
            }
        }

        return generateContiguousEncodedValues<Tag>(
            destination, recipe.settings_.indexOrder, [&](size_t logicalIndex) {
                return generatedEncodedValue<Tag>(pattern, recipe, bound, destination.shape(),
                                                  logicalIndex, destination.type());
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
        if (usesCoordinates<Pattern>(bound.component) ||
            !detail::isContiguous(destination.layout(), recipe.settings_.indexOrder))
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
                        destination, recipe.settings_.indexOrder, [&](size_t logicalIndex) {
                            return rawStorageValue<Tag>(
                                candidates[counterRandom(seed, domain, logicalIndex) %
                                           candidates.size()]);
                        });
                }
                return generateContiguousEncodedValues<Tag>(
                    destination, recipe.settings_.indexOrder, [&](size_t logicalIndex) {
                        return rawStorageValue<Tag>(rawGenerationValueKnown(
                            pattern, seed, domain, {}, destination.shape(), logicalIndex, type));
                    });
            }
        });
    }

    static void generateNonComplex(Tensor destination, const GenerationRecipe& recipe,
                                   const GenerationRecipe::BoundComponent& bound) {
        const auto storage = destination.rawEncodedBackingStorage();
        std::visit(
            [&](const auto& pattern) {
                using Pattern = std::remove_cvref_t<decltype(pattern)>;
                if constexpr (isRawPattern<Pattern>) {
                    const RawWriter write = rawWriter(destination.type());
                    if constexpr (std::is_same_v<Pattern, Component::RawConstantPattern>) {
                        if (fillContiguous(
                                destination, recipe.settings_.indexOrder,
                                [&](std::span<std::byte> destinationStorage, ptrdiff_t offset) {
                                    write(destinationStorage, offset, pattern.parameters.bits);
                                }))
                            return;
                    }
                    if (generateCommonRawType(destination, recipe, bound, pattern)) return;
                    forEachGenerationElement(
                        destination, recipe.settings_.indexOrder,
                        usesCoordinates<Pattern>(bound.component),
                        [&](std::span<const size_t> indices, size_t logicalIndex,
                            ptrdiff_t offset) {
                            write(storage, offset,
                                  rawGenerationValueKnown(
                                      pattern, recipe.settings_.seed, randomDomain(bound), indices,
                                      destination.shape(), logicalIndex, destination.type()));
                        });
                } else {
                    const NumericalWriter write = numericalWriter(destination.type());
                    if constexpr (isConstantPattern<Pattern>) {
                        if (!usesCoordinates<Pattern>(bound.component) &&
                            fillContiguous(
                                destination, recipe.settings_.indexOrder,
                                [&](std::span<std::byte> destinationStorage, ptrdiff_t offset) {
                                    write(destinationStorage, offset,
                                          generationValueKnown(
                                              bound.component, pattern, recipe.settings_.seed,
                                              randomDomain(bound), {}, destination.shape(), 0,
                                              destination.type()));
                                }))
                            return;
                    }
                    if (generateCommonNumericalType(destination, recipe, bound, pattern)) return;
                    forEachGenerationElement(
                        destination, recipe.settings_.indexOrder,
                        usesCoordinates<Pattern>(bound.component),
                        [&](std::span<const size_t> indices, size_t logicalIndex,
                            ptrdiff_t offset) {
                            write(storage, offset,
                                  generationValueKnown(bound.component, pattern,
                                                       recipe.settings_.seed, randomDomain(bound),
                                                       indices, destination.shape(), logicalIndex,
                                                       destination.type()));
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
        if (!detail::isContiguous(destination.layout(), recipe.settings_.indexOrder)) return false;
        const auto storage = destination.rawEncodedBackingStorage();
        const size_t byteOffset =
            static_cast<size_t>(destination.layout().offset()) * sizeof(Storage);
        if (reinterpret_cast<uintptr_t>(storage.data() + byteOffset) % alignof(Storage) != 0)
            return false;
        auto* output = reinterpret_cast<Storage*>(storage.data() + byteOffset);
        const bool usesCoordinates =
            real.usesCoordinates || (imaginary != nullptr && imaginary->usesCoordinates);
        forEachGenerationElement(
            destination, recipe.settings_.indexOrder, usesCoordinates,
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
        const auto storage = destination.rawEncodedBackingStorage();
        const ComplexWriter write = complexWriter(destination.type());
        std::visit(
            [&](const auto& policy) {
                using Policy = std::remove_cvref_t<decltype(policy)>;
                if constexpr (std::is_same_v<Policy, GenerationRecipe::RealOnlyPolicy>) {
                    const PreparedNumericalGenerator real = prepareNumericalComponent(
                        policy.real, recipe.settings_.seed, destination.type());
                    if (real.isConstant &&
                        fillContiguous(
                            destination, recipe.settings_.indexOrder,
                            [&](std::span<std::byte> destinationStorage, ptrdiff_t offset) {
                                write(destinationStorage, offset,
                                      {real({}, destination.shape(), 0), 0.0});
                            }))
                        return;
                    if (generateCommonComplexType(destination, recipe, real, nullptr,
                                                  ComplexGenerationKind::RealOnly))
                        return;
                    forEachGenerationElement(
                        destination, recipe.settings_.indexOrder, real.usesCoordinates,
                        [&](std::span<const size_t> indices, size_t logicalIndex,
                            ptrdiff_t offset) {
                            const double value = real(indices, destination.shape(), logicalIndex);
                            write(storage, offset, {value, 0.0});
                        });
                } else if constexpr (std::is_same_v<Policy, GenerationRecipe::ReplicatedPolicy>) {
                    const PreparedNumericalGenerator value = prepareNumericalComponent(
                        policy.value, recipe.settings_.seed, destination.type());
                    if (value.isConstant &&
                        fillContiguous(
                            destination, recipe.settings_.indexOrder,
                            [&](std::span<std::byte> destinationStorage, ptrdiff_t offset) {
                                const double component = value({}, destination.shape(), 0);
                                write(destinationStorage, offset, {component, component});
                            }))
                        return;
                    if (generateCommonComplexType(destination, recipe, value, nullptr,
                                                  ComplexGenerationKind::Replicated))
                        return;
                    forEachGenerationElement(
                        destination, recipe.settings_.indexOrder, value.usesCoordinates,
                        [&](std::span<const size_t> indices, size_t logicalIndex,
                            ptrdiff_t offset) {
                            const double component =
                                value(indices, destination.shape(), logicalIndex);
                            write(storage, offset, {component, component});
                        });
                } else {
                    const PreparedNumericalGenerator real = prepareNumericalComponent(
                        policy.real, recipe.settings_.seed, destination.type());
                    const PreparedNumericalGenerator imaginary = prepareNumericalComponent(
                        policy.imaginary, recipe.settings_.seed, destination.type());
                    if (real.isConstant && imaginary.isConstant &&
                        fillContiguous(
                            destination, recipe.settings_.indexOrder,
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
                        destination, recipe.settings_.indexOrder,
                        real.usesCoordinates || imaginary.usesCoordinates,
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

    static void generateTensor(Tensor destination, const GenerationRecipe& recipe) {
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
                                                     ScalarType destinationType) {
    return GenerationRecipeAccess::prepareNumericalComponent(
        GenerationRecipeAccess::realComponent(recipe), recipe.seed(), destinationType);
}

PreparedRawGenerator prepareRawGenerator(const GenerationRecipe& recipe,
                                         ScalarType destinationType) {
    return GenerationRecipeAccess::prepareRawComponent(
        GenerationRecipeAccess::realComponent(recipe), recipe.seed(), destinationType);
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
}  // namespace roc::host_numerics::detail
