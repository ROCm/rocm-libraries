// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <roc/host_numerics/index_order.hpp>
#include <roc/host_numerics/tensor.hpp>
#include <span>
#include <type_traits>
#include <variant>
#include <vector>

namespace roc::host_numerics {
namespace detail {
struct GenerationRecipeAccess;
}  // namespace detail

struct GenerationRecipeSettings {
    // The same seed and logical element index always produce the same value,
    // regardless of traversal order or thread count.
    uint64_t seed = 0;

    // Indexed recipes and generateAt use this coordinate-to-index mapping.
    IndexOrder indexOrder = IndexOrder::FirstDimensionFastest;
};

struct ConstantGenerationParameters {
    double value = 0.0;
};

struct ChoiceGenerationParameters {
    // Values are selected deterministically from the list. Repeated entries
    // make that value appear more often.
    std::vector<double> values;
};

struct UniformIntegerGenerationParameters {
    // Inclusive int bounds.
    int lower = 0;
    int upper = 1;
};

struct UniformRealGenerationParameters {
    // Double bounds in generated numerical units. For unequal finite bounds,
    // generated values are strictly between lower and upper.
    double lower = 0.0;
    double upper = 1.0;
};

struct NormalGenerationParameters {
    // Mean and standard deviation are in generated numerical units. Sampling
    // uses one Box-Muller output per logical index.
    double mean = 0.0;
    double standardDeviation = 1.0;
};

struct DimensionGenerationParameters {
    // Zero-based tensor dimension.
    size_t dimension = 0;
};

struct AffineIndexRemainderGenerationParameters {
    // Computes
    // (offset + sum(dimensionCoefficients[d] * index[d])) % positiveDivisor
    // with signed Int64 arithmetic. One coefficient is required per tensor
    // dimension. The C++ remainder sign follows the dividend.
    std::vector<int64_t> dimensionCoefficients;
    int64_t offset = 0;
    int64_t positiveDivisor = 1;
};

struct RandomEncodedExponentGenerationParameters {
    // Bounds are unbiased binary exponents and are inclusive.
    int lowerUnbiasedExponent = 0;
    int upperUnbiasedExponent = 0;

    // No value selects the destination's real component encoding.
    std::optional<ScalarType> sourceType;
};

struct RawConstantGenerationParameters {
    // Low destination storage bits are written without numerical conversion.
    uint64_t bits = 0;
};

struct GenerationAffineValueParameters {
    // The generated numerical value becomes value * scale + offset.
    double scale = 1.0;
    double offset = 0.0;
};

struct AlternatingSignGenerationParameters {
    // Coordinate parity is the XOR of these dimensions. Repeated dimensions
    // cancel in pairs. false negates even parity; true negates odd parity.
    std::vector<size_t> dimensions;
    bool negativeWhenOdd = false;
};

// Immutable generation description. Components and recipes are copyable and
// assignable values; all modifiers return a new Component.
//
// For Shape{6}, before destination scalar conversion:
//   constant({.value = 2.0}) produces [2, 2, 2, 2, 2, 2].
//   serialIndex() produces [0, 1, 2, 3, 4, 5].
//   affineIndexRemainder({.dimensionCoefficients = {2}, .offset = 1,
//                         .positiveDivisor = 5})
//   produces [1, 3, 0, 2, 4, 1].
//
// Candidate sets and alternating-sign dimension lists must be nonempty.
// Interval bounds must be ordered and real bounds must not be NaN. A normal
// standard deviation must be nonnegative and not NaN. An affine-index divisor
// must be positive. Generation rejects dimensions outside the tensor rank,
// affine coefficient counts different from the rank, affine Int64 overflow,
// identity generation below rank two, NaN/infinity/denormal requests for
// encodings without that value, 64-bit integer extrema represented through
// double, encoded-exponent types without an exponent field, raw complex
// output, and generateAt indices outside the tensor.
class GenerationRecipe {
   public:
    class Component {
       public:
        Component(const Component&) = default;
        Component(Component&&) noexcept = default;
        Component& operator=(const Component&) = default;
        Component& operator=(Component&&) noexcept = default;
        ~Component() = default;

        // The unary transform is applied to the base numerical value before
        // the affine value mapping. Sine and cosine inputs are radians.
        [[nodiscard]] Component withAbsoluteTransform() const;
        [[nodiscard]] Component withSineTransform() const;
        [[nodiscard]] Component withCosineTransform() const;

        [[nodiscard]] Component withAffineValueMapping(
            GenerationAffineValueParameters parameters) const;

        // Sign alternation is applied after the unary transform and affine
        // value mapping.
        [[nodiscard]] Component withAlternatingSign(
            AlternatingSignGenerationParameters parameters) const;

        // Returns zero unless dimensions zero and one have the same
        // coordinate. Evaluation requires a tensor rank of at least two.
        [[nodiscard]] Component withZeroOutsideMainDiagonal() const;

       private:
        enum class UnaryTransform {
            None,
            Absolute,
            Sine,
            Cosine,
        };

        struct ZeroPattern {};
        struct ConstantPattern {
            ConstantGenerationParameters parameters;
        };
        struct ChoicePattern {
            ChoiceGenerationParameters parameters;
        };
        struct UniformIntegerPattern {
            UniformIntegerGenerationParameters parameters;
        };
        struct AbsoluteUniformIntegerPattern {
            UniformIntegerGenerationParameters parameters;
        };
        struct UniformRealPattern {
            UniformRealGenerationParameters parameters;
        };
        struct NormalPattern {
            NormalGenerationParameters parameters;
        };
        struct SinePattern {};
        struct CosinePattern {};
        struct AbsoluteSinePattern {};
        struct AbsoluteCosinePattern {};
        struct SerialIndexPattern {};
        struct SerialDimensionPattern {
            DimensionGenerationParameters parameters;
        };
        struct AffineIndexRemainderPattern {
            AffineIndexRemainderGenerationParameters parameters;
        };
        struct IdentityPattern {};
        struct CheckerboardUniformIntegerPattern {
            UniformIntegerGenerationParameters parameters;
        };
        struct TypeMaximumPattern {};
        struct TypeLowestPattern {};
        struct TypeDenormalMinimumPattern {};
        struct TypeDenormalMaximumPattern {};
        struct TypeNaNPattern {};
        struct TypeInfinityPattern {};
        struct TypeNegativeInfinityPattern {};
        struct TypeNegativeZeroPattern {};
        struct UniformTypeRangePattern {};
        struct RandomEncodedExponentPattern {
            RandomEncodedExponentGenerationParameters parameters;
        };
        struct RawConstantPattern {
            RawConstantGenerationParameters parameters;
        };
        struct UniformRawIntegerPattern {
            UniformIntegerGenerationParameters parameters;
        };
        struct UniformFiniteEncodedValuePattern {};
        struct RandomRawBitsPattern {};
        struct RawSerialDimensionPattern {
            DimensionGenerationParameters parameters;
        };

        using Pattern =
            std::variant<ZeroPattern, ConstantPattern, ChoicePattern, UniformIntegerPattern,
                         AbsoluteUniformIntegerPattern, UniformRealPattern, NormalPattern,
                         SinePattern, CosinePattern, AbsoluteSinePattern, AbsoluteCosinePattern,
                         SerialIndexPattern, SerialDimensionPattern, AffineIndexRemainderPattern,
                         IdentityPattern, CheckerboardUniformIntegerPattern, TypeMaximumPattern,
                         TypeLowestPattern, TypeDenormalMinimumPattern, TypeDenormalMaximumPattern,
                         TypeNaNPattern, TypeInfinityPattern, TypeNegativeInfinityPattern,
                         TypeNegativeZeroPattern, UniformTypeRangePattern,
                         RandomEncodedExponentPattern, RawConstantPattern, UniformRawIntegerPattern,
                         UniformFiniteEncodedValuePattern, RandomRawBitsPattern,
                         RawSerialDimensionPattern>;

        explicit Component(Pattern pattern);
        [[nodiscard]] bool isRaw() const;

        Pattern pattern_;
        UnaryTransform unaryTransform_ = UnaryTransform::None;
        GenerationAffineValueParameters affineValue_;
        std::optional<AlternatingSignGenerationParameters> alternatingSign_;
        bool zeroOutsideMainDiagonal_ = false;

        friend class GenerationRecipe;
        friend struct detail::GenerationRecipeAccess;
    };

    GenerationRecipe(const GenerationRecipe&) = default;
    GenerationRecipe(GenerationRecipe&&) noexcept = default;
    GenerationRecipe& operator=(const GenerationRecipe&) = default;
    GenerationRecipe& operator=(GenerationRecipe&&) noexcept = default;
    ~GenerationRecipe() = default;

    [[nodiscard]] static Component zero();
    [[nodiscard]] static Component constant(ConstantGenerationParameters parameters);
    [[nodiscard]] static Component choice(ChoiceGenerationParameters parameters);
    [[nodiscard]] static Component uniformInteger(UniformIntegerGenerationParameters parameters);
    [[nodiscard]] static Component absoluteUniformInteger(
        UniformIntegerGenerationParameters parameters);
    [[nodiscard]] static Component uniformReal(UniformRealGenerationParameters parameters);
    [[nodiscard]] static Component normal(NormalGenerationParameters parameters);

    // These evaluate sin(logicalIndex) and cos(logicalIndex), with the logical
    // index interpreted in radians.
    [[nodiscard]] static Component sine();
    [[nodiscard]] static Component cosine();
    [[nodiscard]] static Component absoluteSine();
    [[nodiscard]] static Component absoluteCosine();

    // serialIndex returns the logical index. serialDimension returns one
    // coordinate. identity compares dimensions zero and one.
    [[nodiscard]] static Component serialIndex();
    [[nodiscard]] static Component serialDimension(DimensionGenerationParameters parameters);
    [[nodiscard]] static Component affineIndexRemainder(
        AffineIndexRemainderGenerationParameters parameters);
    [[nodiscard]] static Component identity();

    // Samples an inclusive integer interval, then negates values at even XOR
    // parity across all tensor coordinates.
    [[nodiscard]] static Component checkerboardUniformInteger(
        UniformIntegerGenerationParameters parameters);

    // Type-derived factories use the destination's real component type.
    [[nodiscard]] static Component typeMaximum();
    [[nodiscard]] static Component typeLowest();
    [[nodiscard]] static Component typeDenormalMinimum();
    [[nodiscard]] static Component typeDenormalMaximum();
    [[nodiscard]] static Component typeNaN();
    [[nodiscard]] static Component typeInfinity();
    [[nodiscard]] static Component typeNegativeInfinity();
    [[nodiscard]] static Component typeNegativeZero();
    [[nodiscard]] static Component uniformTypeRange();
    [[nodiscard]] static Component randomEncodedExponent(
        RandomEncodedExponentGenerationParameters parameters);

    // Raw factories write encoded storage bits and support only real
    // destinations with at most 64 storage bits.
    [[nodiscard]] static Component rawConstant(RawConstantGenerationParameters parameters);
    [[nodiscard]] static Component uniformRawInteger(UniformIntegerGenerationParameters parameters);
    [[nodiscard]] static Component uniformFiniteEncodedValue();
    [[nodiscard]] static Component randomRawBits();
    [[nodiscard]] static Component rawSerialDimension(DimensionGenerationParameters parameters);

    // For a complex destination, realOnly writes zero to the imaginary
    // component. For a non-complex destination, it evaluates component.
    [[nodiscard]] static GenerationRecipe realOnly(Component component,
                                                   GenerationRecipeSettings settings = {});

    // For a complex destination, replicated evaluates one component once and
    // writes the same value to both components. Raw components are rejected.
    [[nodiscard]] static GenerationRecipe replicated(Component component,
                                                     GenerationRecipeSettings settings = {});

    // For a complex destination, cartesian evaluates real and imaginary in
    // distinct random domains. For a non-complex destination, it evaluates
    // only real. Raw components are rejected.
    [[nodiscard]] static GenerationRecipe cartesian(Component real, Component imaginary,
                                                    GenerationRecipeSettings settings = {});

    [[nodiscard]] uint64_t seed() const noexcept;
    [[nodiscard]] IndexOrder indexOrder() const noexcept;
    [[nodiscard]] GenerationRecipe withSeed(uint64_t seed) const;
    [[nodiscard]] GenerationRecipe withIndexOrder(IndexOrder order) const;

   private:
    struct BoundComponent {
        Component component;
        uint64_t randomDomain;
    };
    struct RealOnlyPolicy {
        BoundComponent real;
    };
    struct ReplicatedPolicy {
        BoundComponent value;
    };
    struct CartesianPolicy {
        BoundComponent real;
        BoundComponent imaginary;
    };
    using ComplexPolicy = std::variant<RealOnlyPolicy, ReplicatedPolicy, CartesianPolicy>;

    GenerationRecipe(GenerationRecipeSettings settings, ComplexPolicy complexPolicy);

    GenerationRecipeSettings settings_;
    ComplexPolicy complexPolicy_;

    friend struct detail::GenerationRecipeAccess;
};

// C++20 constrained callable overload. Generator may accept
// (std::span<const size_t> indices) or
// (std::span<const size_t> indices, size_t linearIndex). If both signatures
// are available, the two-argument signature is used. linearIndex traverses
// with the last tensor dimension fastest. The return value is passed to
// Tensor::storeFrom.
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

void generate(Tensor destination, const GenerationRecipe& recipe);
Tensor generate(ScalarType type, Layout layout, const GenerationRecipe& recipe);
Tensor generate(ScalarType type, Shape shape, const GenerationRecipe& recipe);
void generateAt(Tensor destination, size_t logicalIndex, const GenerationRecipe& recipe);
}  // namespace roc::host_numerics
