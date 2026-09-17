// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "detail/generation_recipe_access.hpp"

#include <array>
#include <bit>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <roc/host_numerics/generation.hpp>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "detail/generation_primitives.hpp"
#include "detail/generation_values.hpp"
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
            const double unit = indexedUniformUnit(seed, domain, logicalIndex);
            return pattern.parameters.lower +
                   unit * (pattern.parameters.upper - pattern.parameters.lower);
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

    struct PreparedNumericalComponent {
        const Component* component;
        const void* pattern;
        uint64_t domain;
        double (*value)(const Component&, const void*, uint64_t, uint64_t, std::span<const size_t>,
                        const Shape&, size_t, ScalarType);
        bool usesCoordinates;
    };

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

    static PreparedNumericalComponent prepareNumericalComponent(
        const GenerationRecipe::BoundComponent& bound) {
        if (bound.component.isRaw())
            throw std::invalid_argument("Raw generation does not support complex output.");
        return std::visit(
            [&](const auto& pattern) {
                using Pattern = std::remove_cvref_t<decltype(pattern)>;
                return PreparedNumericalComponent{
                    .component = &bound.component,
                    .pattern = &pattern,
                    .domain = randomDomain(bound),
                    .value =
                        [](const Component& component, const void* erasedPattern, uint64_t seed,
                           uint64_t domain, std::span<const size_t> indices, const Shape& shape,
                           size_t logicalIndex, ScalarType destinationType) {
                            return generationValueKnown(
                                component, *static_cast<const Pattern*>(erasedPattern), seed,
                                domain, indices, shape, logicalIndex, destinationType);
                        },
                    .usesCoordinates = usesCoordinates<Pattern>(bound.component),
                };
            },
            bound.component.pattern_);
    }

    static double generatedPreparedValue(const PreparedNumericalComponent& component,
                                         const GenerationRecipe& recipe,
                                         std::span<const size_t> indices, const Shape& shape,
                                         size_t logicalIndex, ScalarType destinationType) {
        return component.value(*component.component, component.pattern, recipe.settings_.seed,
                               component.domain, indices, shape, logicalIndex, destinationType);
    }

    static bool isContiguous(const Layout& layout, IndexOrder order) {
        uint64_t expectedStride = 1;
        const auto matchesDimension = [&](size_t dimension) {
            const size_t extent = layout.shape()[dimension];
            if (extent > 1 && (layout.stride(dimension) < 0 ||
                               static_cast<uint64_t>(layout.stride(dimension)) != expectedStride))
                return false;
            if (extent != 0 && expectedStride > std::numeric_limits<uint64_t>::max() / extent)
                return false;
            expectedStride *= extent;
            return true;
        };

        if (order == IndexOrder::FirstDimensionFastest) {
            for (size_t dimension = 0; dimension < layout.shape().rank(); ++dimension)
                if (!matchesDimension(dimension)) return false;
        } else {
            for (size_t dimension = layout.shape().rank(); dimension > 0; --dimension)
                if (!matchesDimension(dimension - 1)) return false;
        }
        return true;
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

    template <typename Function>
    static void forEachGenerationElement(Tensor destination, IndexOrder logicalOrder,
                                         bool valueUsesCoordinates, Function&& function) {
        const size_t elementCount = destination.shape().elementCount();
        const bool independent = hasProvablyIndependentElements(destination);
        const IndexOrder traversalOrder =
            independent ? logicalOrder : IndexOrder::LastDimensionFastest;
        const bool contiguous = isContiguous(destination.layout(), traversalOrder);
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

    static void generateNonComplex(Tensor destination, const GenerationRecipe& recipe,
                                   const GenerationRecipe::BoundComponent& bound) {
        const auto storage = destination.rawEncodedBackingStorage();
        std::visit(
            [&](const auto& pattern) {
                using Pattern = std::remove_cvref_t<decltype(pattern)>;
                if constexpr (isRawPattern<Pattern>) {
                    const RawWriter write = rawWriter(destination.type());
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

    static void generateComplex(Tensor destination, const GenerationRecipe& recipe) {
        const auto storage = destination.rawEncodedBackingStorage();
        const ComplexWriter write = complexWriter(destination.type());
        std::visit(
            [&](const auto& policy) {
                using Policy = std::remove_cvref_t<decltype(policy)>;
                if constexpr (std::is_same_v<Policy, GenerationRecipe::RealOnlyPolicy>) {
                    const PreparedNumericalComponent real = prepareNumericalComponent(policy.real);
                    forEachGenerationElement(
                        destination, recipe.settings_.indexOrder, real.usesCoordinates,
                        [&](std::span<const size_t> indices, size_t logicalIndex,
                            ptrdiff_t offset) {
                            const double value =
                                generatedPreparedValue(real, recipe, indices, destination.shape(),
                                                       logicalIndex, destination.type());
                            write(storage, offset, {value, 0.0});
                        });
                } else if constexpr (std::is_same_v<Policy, GenerationRecipe::ReplicatedPolicy>) {
                    const PreparedNumericalComponent value =
                        prepareNumericalComponent(policy.value);
                    forEachGenerationElement(
                        destination, recipe.settings_.indexOrder, value.usesCoordinates,
                        [&](std::span<const size_t> indices, size_t logicalIndex,
                            ptrdiff_t offset) {
                            const double component =
                                generatedPreparedValue(value, recipe, indices, destination.shape(),
                                                       logicalIndex, destination.type());
                            write(storage, offset, {component, component});
                        });
                } else {
                    const PreparedNumericalComponent real = prepareNumericalComponent(policy.real);
                    const PreparedNumericalComponent imaginary =
                        prepareNumericalComponent(policy.imaginary);
                    forEachGenerationElement(
                        destination, recipe.settings_.indexOrder,
                        real.usesCoordinates || imaginary.usesCoordinates,
                        [&](std::span<const size_t> indices, size_t logicalIndex,
                            ptrdiff_t offset) {
                            write(
                                storage, offset,
                                {generatedPreparedValue(real, recipe, indices, destination.shape(),
                                                        logicalIndex, destination.type()),
                                 generatedPreparedValue(imaginary, recipe, indices,
                                                        destination.shape(), logicalIndex,
                                                        destination.type())});
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
