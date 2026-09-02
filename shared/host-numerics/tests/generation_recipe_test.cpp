// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <algorithm>
#include <array>
#include <complex>
#include <cstdint>
#include <iostream>
#include <limits>
#include <roc/host_numerics/generation.hpp>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace {
using namespace roc::host_numerics;

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

template <typename Exception, typename Function>
void requireThrows(Function&& function, const std::string& message) {
    try {
        function();
    } catch (const Exception&) {
        return;
    }
    throw std::runtime_error(message);
}

bool hasEqualStorage(const Tensor& first, const Tensor& second) {
    return first.type() == second.type() && first.layout() == second.layout() &&
           first.rawEncodedBackingStorage().size() == second.rawEncodedBackingStorage().size() &&
           std::equal(first.rawEncodedBackingStorage().begin(),
                      first.rawEncodedBackingStorage().end(),
                      second.rawEncodedBackingStorage().begin());
}

void requireEqualStorage(const Tensor& first, const Tensor& second, const std::string& message) {
    require(hasEqualStorage(first, second), message);
}

void testValueSemanticsAndExamples() {
    static_assert(std::is_copy_constructible_v<GenerationRecipe>);
    static_assert(std::is_copy_assignable_v<GenerationRecipe>);
    static_assert(std::is_move_constructible_v<GenerationRecipe>);
    static_assert(std::is_move_assignable_v<GenerationRecipe>);
    static_assert(std::is_copy_constructible_v<GenerationRecipe::Component>);
    static_assert(std::is_copy_assignable_v<GenerationRecipe::Component>);

    const GenerationRecipe::Component constant = GenerationRecipe::constant({.value = 2.0});
    const GenerationRecipe::Component mapped =
        constant.withAffineValueMapping({.scale = 3.0, .offset = 1.0});
    const Tensor constantValues =
        generate(ScalarType::Float64, Shape{6}, GenerationRecipe::realOnly(constant));
    const Tensor mappedValues =
        generate(ScalarType::Float64, Shape{6}, GenerationRecipe::realOnly(mapped));
    for (size_t index = 0; index < 6; ++index) {
        require(constantValues.loadAs<double>({index}) == 2.0,
                "A component modifier changed the source component.");
        require(mappedValues.loadAs<double>({index}) == 7.0,
                "Affine value mapping produced an incorrect value.");
    }

    const Tensor serial = generate(ScalarType::Int32, Shape{6},
                                   GenerationRecipe::realOnly(GenerationRecipe::serialIndex()));
    for (size_t index = 0; index < 6; ++index)
        require(serial.loadAs<int32_t>({index}) == static_cast<int32_t>(index),
                "Serial-index generation did not produce [0, 1, 2, 3, 4, 5].");

    const Tensor affine =
        generate(ScalarType::Int32, Shape{6},
                 GenerationRecipe::realOnly(GenerationRecipe::affineIndexRemainder(
                     {.dimensionCoefficients = {2}, .offset = 1, .positiveDivisor = 5})));
    constexpr std::array<int32_t, 6> expected{1, 3, 0, 2, 4, 1};
    for (size_t index = 0; index < expected.size(); ++index)
        require(affine.loadAs<int32_t>({index}) == expected[index],
                "Affine-index remainder generation produced an incorrect six-element sample.");

    const Tensor diagonal = generate(
        ScalarType::Int32, Shape{2, 3},
        GenerationRecipe::realOnly(GenerationRecipe::serialDimension({.dimension = 0})
                                       .withAffineValueMapping({.scale = 1.0, .offset = 1.0})
                                       .withZeroOutsideMainDiagonal()));
    constexpr std::array<int32_t, 6> expectedDiagonal{1, 0, 0, 0, 2, 0};
    for (size_t index = 0; index < expectedDiagonal.size(); ++index)
        require(diagonal.loadAs<int32_t>({index / 3, index % 3}) == expectedDiagonal[index],
                "Main-diagonal generation modifier produced an incorrect value.");

    constexpr uint64_t rawBits = 0xfedcba9876543210ULL;
    const Tensor rawTyped =
        generate(ScalarType::UInt64, Shape{2},
                 GenerationRecipe::realOnly(GenerationRecipe::rawConstant({.bits = rawBits})));
    for (size_t index = 0; index < 2; ++index)
        require(rawTyped.loadAs<uint64_t>({index}) == rawBits,
                "Typed raw-constant generation changed the requested bits.");

    const GenerationRecipe finiteEncodedRecipe =
        GenerationRecipe::realOnly(GenerationRecipe::uniformFiniteEncodedValue(), {.seed = 91});
    const Tensor finiteEncoded = generate(ScalarType::Float8E5M2, Shape{1024}, finiteEncodedRecipe);
    const Tensor finiteEncodedRepeat =
        generate(ScalarType::Float8E5M2, Shape{1024}, finiteEncodedRecipe);
    requireEqualStorage(finiteEncoded, finiteEncodedRepeat,
                        "Uniform finite encoded generation was not deterministic.");
    for (size_t index = 0; index < finiteEncoded.elementCount(); ++index)
        require(std::isfinite(finiteEncoded.loadAs<double>({index})),
                "Uniform finite encoded generation produced a non-finite value.");

    const Tensor e8ZeroExponents = generate(
        ScalarType::E8M0Zero, Shape{64},
        GenerationRecipe::realOnly(GenerationRecipe::randomEncodedExponent(
                                       {.lowerUnbiasedExponent = -3, .upperUnbiasedExponent = 3}),
                                   {.seed = 19}));
    for (size_t index = 0; index < e8ZeroExponents.elementCount(); ++index) {
        const float value = e8ZeroExponents.loadAs<float>({index});
        require(std::isfinite(value) && value >= 0.125f && value <= 8.0f,
                "E8M0Zero random-exponent generation produced an invalid scale.");
    }
}

void testSeedAndComplexPolicies() {
    constexpr uint64_t seed = 37;
    constexpr UniformIntegerGenerationParameters parameters{.lower = -100, .upper = 100};
    const GenerationRecipe::Component component = GenerationRecipe::uniformInteger(parameters);

    const GenerationRecipe realOnly = GenerationRecipe::realOnly(component, {.seed = seed});
    const GenerationRecipe replicated = GenerationRecipe::replicated(component, {.seed = seed});
    const GenerationRecipe cartesian =
        GenerationRecipe::cartesian(component, component, {.seed = seed});

    const Tensor realOnlyValues = generate(ScalarType::ComplexFloat64, Shape{16}, realOnly);
    const Tensor replicatedValues = generate(ScalarType::ComplexFloat64, Shape{16}, replicated);
    const Tensor cartesianValues = generate(ScalarType::ComplexFloat64, Shape{16}, cartesian);
    const Tensor cartesianRepeat = generate(ScalarType::ComplexFloat64, Shape{16}, cartesian);
    requireEqualStorage(cartesianValues, cartesianRepeat,
                        "Equal typed recipes and seeds produced different storage.");

    constexpr std::array<int, 16> expectedReal{63, -65, 68,  30, 88, -25, 23, 61,
                                               31, -99, -88, 35, 61, -78, 52, -100};
    constexpr std::array<int, 16> expectedImaginary{75,  23, -79, -13, 8,   99,  -9,  44,
                                                    -15, -1, -10, 64,  -42, -37, -96, -24};

    bool cartesianComponentsDiffer = false;
    for (size_t index = 0; index < 16; ++index) {
        const std::complex<double> realOnlyValue =
            realOnlyValues.loadAs<std::complex<double>>({index});
        const std::complex<double> replicatedValue =
            replicatedValues.loadAs<std::complex<double>>({index});
        const std::complex<double> cartesianValue =
            cartesianValues.loadAs<std::complex<double>>({index});

        require(realOnlyValue == std::complex<double>(expectedReal[index], 0.0),
                "Real-only generation did not zero the imaginary component.");
        require(replicatedValue == std::complex<double>(expectedReal[index], expectedReal[index]),
                "Replicated generation did not use one value for both components.");
        require(
            cartesianValue == std::complex<double>(expectedReal[index], expectedImaginary[index]),
            "Cartesian generation did not use the versioned component domains.");
        cartesianComponentsDiffer |= cartesianValue.real() != cartesianValue.imag();
    }
    require(cartesianComponentsDiffer,
            "Cartesian component domains produced identical values for the complete sample.");

    const GenerationRecipe differentSeed = realOnly.withSeed(seed + 1);
    const Tensor differentSeedValues = generate(ScalarType::Int32, Shape{16}, differentSeed);
    require(!hasEqualStorage(realOnlyValues, differentSeedValues),
            "Different caller-provided seeds produced identical storage.");
    require(differentSeed.seed() == seed + 1 && realOnly.seed() == seed,
            "Generation recipe seed modifier changed the source recipe.");

    const Tensor realDestination = generate(ScalarType::Int32, Shape{16}, cartesian);
    for (size_t index = 0; index < 16; ++index) {
        require(realDestination.loadAs<int32_t>({index}) == expectedReal[index],
                "Cartesian generation used the imaginary recipe for a real destination.");
    }
}

void testValidationFailures() {
    requireThrows<std::invalid_argument>([] { (void)GenerationRecipe::choice({}); },
                                         "An empty candidate set was accepted.");
    requireThrows<std::invalid_argument>(
        [] { (void)GenerationRecipe::uniformInteger({.lower = 2, .upper = 1}); },
        "A reversed integer interval was accepted.");
    requireThrows<std::invalid_argument>(
        [] { (void)GenerationRecipe::uniformReal({.lower = 2.0, .upper = 1.0}); },
        "A reversed real interval was accepted.");
    requireThrows<std::invalid_argument>(
        [] {
            (void)GenerationRecipe::uniformReal(
                {.lower = std::numeric_limits<double>::quiet_NaN(), .upper = 1.0});
        },
        "A NaN real interval bound was accepted.");
    requireThrows<std::invalid_argument>(
        [] {
            (void)GenerationRecipe::normal(
                {.mean = 0.0, .standardDeviation = std::numeric_limits<double>::quiet_NaN()});
        },
        "A NaN normal standard deviation was accepted.");
    requireThrows<std::invalid_argument>(
        [] {
            (void)GenerationRecipe::normal(
                {.mean = std::numeric_limits<double>::infinity(), .standardDeviation = 1.0});
        },
        "A non-finite normal mean was accepted.");
    requireThrows<std::invalid_argument>(
        [] {
            (void)GenerationRecipe::affineIndexRemainder(
                {.dimensionCoefficients = {1}, .positiveDivisor = 0});
        },
        "A nonpositive affine-index divisor was accepted.");
    requireThrows<std::invalid_argument>(
        [] {
            (void)GenerationRecipe::randomEncodedExponent({.lowerUnbiasedExponent = -2,
                                                           .upperUnbiasedExponent = 2,
                                                           .sourceType = ScalarType::Int32});
        },
        "An encoded-exponent source without an exponent field was accepted.");
    requireThrows<std::invalid_argument>(
        [] { (void)GenerationRecipe::replicated(GenerationRecipe::rawConstant({.bits = 1})); },
        "A raw replicated component was accepted.");
    requireThrows<std::invalid_argument>(
        [] {
            (void)GenerationRecipe::rawConstant({.bits = 1}).withAffineValueMapping({.scale = 2.0});
        },
        "A numerical modifier was accepted for a raw component.");
    requireThrows<std::invalid_argument>(
        [] { (void)GenerationRecipe::constant({.value = 1.0}).withAlternatingSign({}); },
        "An empty alternating-sign dimension list was accepted.");

    const GenerationRecipe invalidDimension =
        GenerationRecipe::realOnly(GenerationRecipe::serialDimension({.dimension = 1}));
    requireThrows<std::out_of_range>(
        [&] {
            Tensor destination(ScalarType::Float32, Shape{4});
            generate(destination, invalidDimension);
        },
        "A generation dimension outside the tensor rank was accepted.");

    const GenerationRecipe invalidCoefficientCount =
        GenerationRecipe::realOnly(GenerationRecipe::affineIndexRemainder(
            {.dimensionCoefficients = {1}, .positiveDivisor = 3}));
    requireThrows<std::invalid_argument>(
        [&] {
            Tensor destination(ScalarType::Float32, Shape{2, 2});
            generate(destination, invalidCoefficientCount);
        },
        "An affine coefficient count different from the tensor rank was accepted.");

    requireThrows<std::out_of_range>(
        [] {
            Tensor destination(ScalarType::Float32, Shape{4});
            generateAt(destination, 4, GenerationRecipe::realOnly(GenerationRecipe::zero()));
        },
        "generateAt accepted a logical index equal to the element count.");

    requireThrows<std::invalid_argument>(
        [] {
            Tensor destination(ScalarType::ComplexFloat32, Shape{1});
            generate(destination,
                     GenerationRecipe::realOnly(GenerationRecipe::rawConstant({.bits = 1})));
        },
        "Raw generation accepted a complex destination.");
    requireThrows<std::invalid_argument>(
        [] {
            Tensor destination(ScalarType::Float32, Shape{1});
            generate(destination,
                     GenerationRecipe::realOnly(GenerationRecipe::uniformFiniteEncodedValue()));
        },
        "Uniform finite encoded generation accepted a format wider than eight bits.");
    requireThrows<std::invalid_argument>(
        [] {
            Tensor destination(ScalarType::Float32, Shape{1});
            generate(destination,
                     GenerationRecipe::realOnly(
                         GenerationRecipe::constant({.value = 1.0}).withZeroOutsideMainDiagonal()));
        },
        "Main-diagonal generation accepted a rank-one destination.");
}

struct BothCallableSignatures {
    int operator()(std::span<const size_t>) const {
        return -1;
    }

    int operator()(std::span<const size_t>, size_t linearIndex) const {
        return static_cast<int>(linearIndex);
    }
};

void testCallableOverload() {
    Tensor destination(ScalarType::Int32, Shape{2, 3});
    generate(destination, BothCallableSignatures{});
    for (size_t first = 0; first < 2; ++first) {
        for (size_t second = 0; second < 3; ++second) {
            const int expected = static_cast<int>(first * 3 + second);
            require(destination.loadAs<int32_t>({first, second}) == expected,
                    "The C++20 callable overload did not select the two-argument signature.");
        }
    }
}
}  // namespace

int main() {
    try {
        testValueSemanticsAndExamples();
        testSeedAndComplexPolicies();
        testValidationFailures();
        testCallableOverload();
        std::cout << "generation recipe tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
