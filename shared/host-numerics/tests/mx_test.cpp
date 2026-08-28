// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <roc/host_numerics/mx.hpp>
#include <stdexcept>
#include <utility>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace roc::host_numerics;

namespace {
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

bool sameStorage(const Tensor& first, const Tensor& second) {
    return first.type() == second.type() && first.layout() == second.layout() &&
           first.rawEncodedBackingStorage().size() == second.rawEncodedBackingStorage().size() &&
           std::equal(first.rawEncodedBackingStorage().begin(),
                      first.rawEncodedBackingStorage().end(),
                      second.rawEncodedBackingStorage().begin());
}

bool sameResult(const MxGenerationResult& first, const MxGenerationResult& second) {
    return sameStorage(first.data, second.data) && sameStorage(first.scales, second.scales) &&
           sameStorage(first.scaleIndices, second.scaleIndices) &&
           sameStorage(first.reference, second.reference);
}

template <size_t N>
void requireStorage(const Tensor& tensor, const std::array<uint8_t, N>& expected,
                    const char* message) {
    const auto storage = tensor.rawEncodedBackingStorage();
    require(storage.size() == expected.size(), message);
    for (size_t index = 0; index < expected.size(); ++index)
        require(std::to_integer<uint8_t>(storage[index]) == expected[index], message);
}

size_t expectedScaleCount(const MxGenerationProblem& problem) {
    const size_t blockedExtent = problem.shape[problem.blockAxis];
    const size_t freeExtent = problem.shape[1 - problem.blockAxis];
    return ((blockedExtent + problem.blockSize - 1) / problem.blockSize) * freeExtent;
}

size_t expectedScaleIndex(const MxGenerationProblem& problem, size_t row, size_t column) {
    if (problem.blockAxis == 0) {
        const size_t blocks = (problem.shape[0] + problem.blockSize - 1) / problem.blockSize;
        return row / problem.blockSize + column * blocks;
    }
    return row + (column / problem.blockSize) * problem.shape[0];
}

void checkReference(const MxGenerationProblem& problem, const MxGenerationResult& result) {
    const Tensor data = result.data;
    const Tensor scales = result.scales;
    const Tensor scaleIndices = result.scaleIndices;
    const Tensor reference = result.reference;
    const size_t rows = problem.shape[0];
    const size_t columns = problem.shape[1];
    const size_t scaleCount = expectedScaleCount(problem);
    const Shape expectedScaleShape =
        problem.blockAxis == 0
            ? Shape{problem.shape[1],
                    (problem.shape[0] + problem.blockSize - 1) / problem.blockSize}
            : Shape{(problem.shape[1] + problem.blockSize - 1) / problem.blockSize,
                    problem.shape[0]};
    require(scales.shape() == expectedScaleShape, "MX scale shape mismatch.");
    const Tensor flatScales = scales.reshapeSharingStorage(Shape{scaleCount});
    require(scaleIndices.shape() == problem.shape, "MX scale-index shape mismatch.");
    require(reference.shape() == problem.shape, "MX reference shape mismatch.");

    for (size_t column = 0; column < columns; ++column) {
        for (size_t row = 0; row < rows; ++row) {
            const size_t scaleIndex = scaleIndices.loadAs<uint32_t>({row, column});
            require(scaleIndex == expectedScaleIndex(problem, row, column),
                    "MX scale-index mapping mismatch.");
            require(scaleIndex < scaleCount, "MX scale index exceeds scale storage.");
            const float expected =
                data.loadAs<float>({row, column}) * flatScales.loadAs<float>({scaleIndex});
            const float observed = reference.loadAs<float>({row, column});
            if (std::isnan(expected))
                require(std::isnan(observed), "MX NaN reference mismatch.");
            else
                require(expected == observed, "MX reference value mismatch.");
        }
    }
}

GenerationRecipe mxRecipe(GenerationRecipe::Component component, uint64_t seed = 12345,
                          uint64_t randomDomain = mx_generation_random_domain_version_1::data) {
    return GenerationRecipe::realOnly(std::move(component),
                                      {
                                          .seed = seed,
                                          .indexOrder = IndexOrder::FirstDimensionFastest,
                                          .randomDomain = randomDomain,
                                      });
}

MxDataGeneration defaultMxDataGeneration() {
    return MxDataGeneration::preserveRange(
        mxRecipe(GenerationRecipe::uniformReal({.lower = -1.0, .upper = 1.0}), 0),
        {.lower = -1.0, .upper = 1.0});
}

MxGenerationProblem defaultMxProblem(Shape shape) {
    return MxGenerationProblem(std::move(shape), defaultMxDataGeneration());
}

MxGenerationProblem stochasticProblem(
    GenerationRecipe::Component component,
    MxDataQuantization quantization = MxDataQuantization::Nearest,
    std::optional<MxRepresentedValueRange> representedValueRange = std::nullopt,
    MxScaleGenerationMode scale = MxScaleGenerationMode::Derived,
    uint64_t randomDomain = mx_generation_random_domain_version_1::data) {
    GenerationRecipe recipe = mxRecipe(std::move(component), 12345, randomDomain);
    MxDataGeneration data = [&] {
        if (quantization == MxDataQuantization::PreserveRange)
            return MxDataGeneration::preserveRange(std::move(recipe), *representedValueRange);
        if (quantization == MxDataQuantization::PreserveGeneratedEncoding)
            return MxDataGeneration::preserveGeneratedEncoding(std::move(recipe));
        return MxDataGeneration::quantize(std::move(recipe));
    }();
    MxGenerationProblem problem(Shape{257, 67}, std::move(data));
    problem.dataType = ScalarType::Float4E2M1;
    problem.scaleType = ScalarType::E8M0;
    problem.leadingDimension = 263;
    problem.blockAxis = 0;
    problem.blockSize = 16;
    problem.scale = scale;
    return problem;
}

MxGenerationProblem boundedProblem(double lower = -1.0, double upper = 1.0) {
    return stochasticProblem(GenerationRecipe::uniformReal({.lower = lower, .upper = upper}),
                             MxDataQuantization::PreserveRange,
                             MxRepresentedValueRange{.lower = lower, .upper = upper});
}

MxGenerationProblem unboundedProblem() {
    return stochasticProblem(GenerationRecipe::uniformFiniteEncodedValue(),
                             MxDataQuantization::PreserveGeneratedEncoding, std::nullopt,
                             MxScaleGenerationMode::RandomFinite,
                             mx_generation_random_domain_version_1::unboundedData);
}

void testLegacyRandomStreamCompatibility() {
    auto makeProblem = [](GenerationRecipe::Component component, MxDataQuantization quantization,
                          std::optional<MxRepresentedValueRange> representedValueRange,
                          MxScaleGenerationMode scale, uint64_t randomDomain) {
        GenerationRecipe recipe = mxRecipe(std::move(component), 0x10203040, randomDomain);
        MxDataGeneration data = [&] {
            if (quantization == MxDataQuantization::PreserveRange)
                return MxDataGeneration::preserveRange(std::move(recipe), *representedValueRange);
            if (quantization == MxDataQuantization::PreserveGeneratedEncoding)
                return MxDataGeneration::preserveGeneratedEncoding(std::move(recipe));
            return MxDataGeneration::quantize(std::move(recipe));
        }();
        MxGenerationProblem problem(Shape{5, 3}, std::move(data));
        problem.dataType = ScalarType::Float4E2M1;
        problem.scaleType = ScalarType::E8M0;
        problem.leadingDimension = 7;
        problem.blockAxis = 0;
        problem.blockSize = 4;
        problem.scale = scale;
        return problem;
    };

    const MxGenerationResult bounded = generateMx(makeProblem(
        GenerationRecipe::uniformReal({.lower = -1.25, .upper = 0.875}),
        MxDataQuantization::PreserveRange, MxRepresentedValueRange{.lower = -1.25, .upper = 0.875},
        MxScaleGenerationMode::Derived, mx_generation_random_domain_version_1::data));
    requireStorage(
        bounded.data,
        std::array<uint8_t, 11>{0xff, 0x77, 0x05, 0x20, 0x1e, 0xda, 0x00, 0x1c, 0x5e, 0x0b, 0x00},
        "Bounded MX data stream changed.");
    requireStorage(bounded.scales, std::array<uint8_t, 6>{0x7c, 0x7d, 0x7d, 0x7d, 0x7d, 0x7e},
                   "Bounded MX scale stream changed.");

    const MxGenerationResult normal = generateMx(
        makeProblem(GenerationRecipe::normal({.mean = 0.0, .standardDeviation = 1.25}),
                    MxDataQuantization::Nearest, std::nullopt, MxScaleGenerationMode::Derived,
                    mx_generation_random_domain_version_1::normal));
    requireStorage(
        normal.data,
        std::array<uint8_t, 11>{0xdb, 0x43, 0x0f, 0x50, 0xf6, 0xee, 0x00, 0xd4, 0xc9, 0x0e, 0x00},
        "Normal MX data stream changed.");
    requireStorage(normal.scales, std::array<uint8_t, 6>{0x7f, 0x7d, 0x7d, 0x7d, 0x7e, 0x7d},
                   "Normal MX scale stream changed.");

    const MxGenerationResult unbounded = generateMx(makeProblem(
        GenerationRecipe::uniformFiniteEncodedValue(),
        MxDataQuantization::PreserveGeneratedEncoding, std::nullopt,
        MxScaleGenerationMode::RandomFinite, mx_generation_random_domain_version_1::unboundedData));
    requireStorage(
        unbounded.data,
        std::array<uint8_t, 11>{0x3a, 0x98, 0x06, 0x40, 0x58, 0x50, 0x00, 0x13, 0x1a, 0x06, 0x00},
        "Unbounded MX data stream changed.");
    requireStorage(unbounded.scales, std::array<uint8_t, 6>{0xd2, 0xdc, 0x12, 0x24, 0x36, 0xbf},
                   "Unbounded MX scale stream changed.");
}

void testGenerationRecipeIndexOrder() {
    auto makeProblem = [](IndexOrder order) {
        MxDataGeneration data = MxDataGeneration::quantize(GenerationRecipe::realOnly(
            GenerationRecipe::uniformInteger({.lower = -4, .upper = 4}),
            {
                .seed = 123,
                .indexOrder = order,
                .randomDomain = mx_generation_random_domain_version_1::data,
            }));
        MxGenerationProblem problem(Shape{3, 5}, std::move(data));
        problem.dataType = ScalarType::Float8E4M3;
        problem.scaleType = ScalarType::E8M0;
        problem.leadingDimension = 3;
        problem.blockAxis = 0;
        problem.blockSize = 3;
        problem.scale = MxScaleGenerationMode::One;
        return problem;
    };

    const MxGenerationProblem firstDimensionFastest =
        makeProblem(IndexOrder::FirstDimensionFastest);
    const MxGenerationProblem lastDimensionFastest = makeProblem(IndexOrder::LastDimensionFastest);
    const MxGenerationResult first = generateMx(firstDimensionFastest);
    const MxGenerationResult last = generateMx(lastDimensionFastest);
    require(!sameStorage(first.data, last.data),
            "MX generation ignored the recipe's logical index order.");

    const Layout dataLayout(Shape{3, 5}, {1, 3});
    require(sameStorage(first.data, generate(ScalarType::Float8E4M3, dataLayout,
                                             firstDimensionFastest.data.recipe())),
            "MX generation changed a first-dimension-fastest recipe stream.");
    require(sameStorage(last.data, generate(ScalarType::Float8E4M3, dataLayout,
                                            lastDimensionFastest.data.recipe())),
            "MX generation changed a last-dimension-fastest recipe stream.");
}

void testPreservedRawGenerationMasksToDataWidth() {
    MxGenerationProblem problem(
        Shape{17, 3}, MxDataGeneration::preserveGeneratedEncoding(GenerationRecipe::realOnly(
                          GenerationRecipe::randomRawBits(), {.seed = 73})));
    problem.dataType = ScalarType::Float4E2M1;
    problem.scaleType = ScalarType::E8M0;
    problem.blockAxis = 0;
    problem.blockSize = 4;
    problem.scale = MxScaleGenerationMode::One;

    const MxGenerationResult result = generateMx(problem);
    checkReference(problem, result);
    for (size_t index = 0; index < result.data.elementCount(); ++index) {
        const size_t row = index % problem.shape[0];
        const size_t column = index / problem.shape[0];
        const float value = result.data.loadAs<float>({row, column});
        require(std::isfinite(value), "Preserved raw MX generation produced a non-finite value.");
    }
}
}  // namespace

int main() {
    bool rejectedInvalidConstruction = false;
    try {
        (void)MxGenerationProblem(Shape{1}, defaultMxDataGeneration());
    } catch (const std::invalid_argument&) {
        rejectedInvalidConstruction = true;
    }
    require(rejectedInvalidConstruction,
            "MX problem construction accepted a shape that was not rank two.");

    testLegacyRandomStreamCompatibility();
    testGenerationRecipeIndexOrder();
    testPreservedRawGenerationMasksToDataWidth();
    const std::array typePairs{
        std::pair{ScalarType::Float8E5M2, ScalarType::E8M0},
        std::pair{ScalarType::Float8E4M3, ScalarType::E8M0},
        std::pair{ScalarType::Float6E2M3, ScalarType::E8M0},
        std::pair{ScalarType::Float6E3M2, ScalarType::E8M0},
        std::pair{ScalarType::Float4E2M1, ScalarType::E8M0},
        std::pair{ScalarType::Float8E4M3, ScalarType::E8M0Zero},
        std::pair{ScalarType::Float4E2M1, ScalarType::E8M0Zero},
        std::pair{ScalarType::Float4E2M1, ScalarType::E4M3},
        std::pair{ScalarType::Float4E2M1, ScalarType::E5M3},
    };
    for (const auto& [dataType, scaleType] : typePairs) {
        for (const size_t blockSize : {size_t{16}, size_t{32}}) {
            MxGenerationProblem problem = defaultMxProblem(Shape{67, 5});
            problem.dataType = dataType;
            problem.scaleType = scaleType;
            problem.leadingDimension = 73;
            problem.blockAxis = 0;
            problem.blockSize = blockSize;
            const MxGenerationResult result = generateMx(problem);
            checkReference(problem, result);
            const size_t expectedBytes =
                (static_cast<size_t>(problem.leadingDimension) * problem.shape[1] *
                     scalarTypeInfo(dataType).storageBits +
                 7) /
                8;
            require(result.data.rawEncodedBackingStorage().size() == expectedBytes,
                    "MX padded data-storage size mismatch.");
        }
    }

    MxGenerationProblem blockAxisOne = defaultMxProblem(Shape{5, 37});
    blockAxisOne.dataType = ScalarType::Float6E2M3;
    blockAxisOne.scaleType = ScalarType::E8M0;
    blockAxisOne.leadingDimension = 8;
    blockAxisOne.blockAxis = 1;
    blockAxisOne.blockSize = 16;
    checkReference(blockAxisOne, generateMx(blockAxisOne));

    MxGenerationProblem paddedRegression = defaultMxProblem(Shape{64, 2});
    paddedRegression.dataType = ScalarType::Float4E2M1;
    paddedRegression.scaleType = ScalarType::E8M0;
    paddedRegression.leadingDimension = 80;
    paddedRegression.blockAxis = 0;
    paddedRegression.blockSize = 32;
    checkReference(paddedRegression, generateMx(paddedRegression));

    MxGenerationProblem formerLargeTailRegression = defaultMxProblem(Shape{2048, 514});
    formerLargeTailRegression.dataType = ScalarType::Float4E2M1;
    formerLargeTailRegression.scaleType = ScalarType::E8M0;
    formerLargeTailRegression.leadingDimension = 2048;
    formerLargeTailRegression.blockAxis = 0;
    formerLargeTailRegression.blockSize = 32;
    const MxGenerationResult largeTail = generateMx(formerLargeTailRegression);
    require(largeTail.data.rawEncodedBackingStorage().size() == 2048 * 514 / 2,
            "Large MX tail regression produced the wrong packed data size.");
    require(largeTail.scales.shape() == Shape{514, 64},
            "Large MX tail regression produced the wrong scale shape.");

    for (const ptrdiff_t leadingDimension : {ptrdiff_t{5}, ptrdiff_t{6}, ptrdiff_t{7}}) {
        MxGenerationProblem fp6PackingTail = defaultMxProblem(Shape{5, 1});
        fp6PackingTail.dataType = ScalarType::Float6E3M2;
        fp6PackingTail.scaleType = ScalarType::E8M0;
        fp6PackingTail.leadingDimension = leadingDimension;
        fp6PackingTail.blockAxis = 0;
        fp6PackingTail.blockSize = 4;
        const MxGenerationResult result = generateMx(fp6PackingTail);
        checkReference(fp6PackingTail, result);
        const size_t physicalElements = static_cast<size_t>(leadingDimension);
        require(result.data.rawEncodedBackingStorage().size() == (physicalElements * 6 + 7) / 8,
                "FP6 packing-tail storage size mismatch.");
    }

    const std::array<GenerationRecipe::Component, 15> deterministicRecipes{
        GenerationRecipe::identity(),
        GenerationRecipe::constant({.value = 1.0}),
        GenerationRecipe::zero(),
        GenerationRecipe::affineIndexRemainder(
            {.dimensionCoefficients = {67, 1}, .positiveDivisor = 256}),
        GenerationRecipe::affineIndexRemainder(
            {.dimensionCoefficients = {1, 0}, .positiveDivisor = 256}),
        GenerationRecipe::affineIndexRemainder(
            {.dimensionCoefficients = {0, 1}, .positiveDivisor = 256}),
        GenerationRecipe::affineIndexRemainder(
            {.dimensionCoefficients = {1, 1}, .positiveDivisor = 2})
            .withAffineValueMapping({.scale = -1.0, .offset = 1.0}),
        GenerationRecipe::serialDimension({.dimension = 0})
            .withAffineValueMapping({.scale = 1.0, .offset = 1.0})
            .withZeroOutsideMainDiagonal(),
        GenerationRecipe::constant({.value = 2.0}),
        GenerationRecipe::constant({.value = -1.0}),
        GenerationRecipe::typeMaximum(),
        GenerationRecipe::typeDenormalMinimum(),
        GenerationRecipe::typeDenormalMaximum(),
        GenerationRecipe::constant({.value = std::numeric_limits<double>::quiet_NaN()}),
        GenerationRecipe::uniformInteger({.lower = -4, .upper = 4}),
    };
    for (size_t recipeIndex = 0; recipeIndex < deterministicRecipes.size(); ++recipeIndex) {
        MxGenerationProblem problem = stochasticProblem(deterministicRecipes[recipeIndex]);
        const MxGenerationResult result = generateMx(problem);
        checkReference(problem, result);
        if (recipeIndex == 13) {
            for (size_t column = 0; column < problem.shape[1]; ++column)
                for (size_t row = 0; row < problem.shape[0]; ++row)
                    require(std::isnan(result.reference.loadAs<float>({row, column})),
                            "MX NaN mode produced a finite reference value.");
        }
    }

    MxGenerationProblem infinity(
        Shape{16, 3}, MxDataGeneration::quantize(mxRecipe(GenerationRecipe::typeInfinity(), 0)));
    infinity.dataType = ScalarType::Float8E5M2;
    infinity.scaleType = ScalarType::E8M0;
    infinity.blockSize = 16;
    checkReference(infinity, generateMx(infinity));

    MxGenerationProblem explicitScale = boundedProblem();
    explicitScale.scale = MxScaleGenerationMode::One;
    const MxGenerationResult explicitlyScaled = generateMx(explicitScale);
    const Tensor explicitlyScaledFlat = explicitlyScaled.scales.reshapeSharingStorage(
        Shape{explicitlyScaled.scales.elementCount()});
    for (size_t scaleIndex = 0; scaleIndex < explicitlyScaledFlat.elementCount(); ++scaleIndex)
        require(explicitlyScaledFlat.loadAs<float>({scaleIndex}) == 1.0f,
                "MX explicit unity-scale generation mismatch.");
    checkReference(explicitScale, explicitlyScaled);

    MxGenerationProblem zeroScale(
        Shape{8, 2},
        MxDataGeneration::quantize(GenerationRecipe::realOnly(GenerationRecipe::zero())));
    zeroScale.dataType = ScalarType::Float4E2M1;
    zeroScale.scaleType = ScalarType::E8M0Zero;
    zeroScale.blockAxis = 0;
    zeroScale.blockSize = 4;
    zeroScale.scale = MxScaleGenerationMode::Minimum;
    const MxGenerationResult zeroScaled = generateMx(zeroScale);
    require(std::ranges::all_of(zeroScaled.scales.rawEncodedBackingStorage(),
                                [](std::byte value) { return value == std::byte{0}; }),
            "E8M0Zero minimum scale did not use its zero encoding.");
    for (size_t index = 0; index < zeroScaled.reference.elementCount(); ++index)
        require(zeroScaled.reference.loadAs<float>({index % 8, index / 8}) == 0.0f,
                "E8M0Zero minimum scale did not produce a zero reference.");

    const std::array constantScaleModes{
        std::pair{MxScaleGenerationMode::Minimum, uint8_t{0}},
        std::pair{MxScaleGenerationMode::One, uint8_t{127}},
        std::pair{MxScaleGenerationMode::Two, uint8_t{128}},
    };
    for (const auto& [mode, expectedRaw] : constantScaleModes) {
        MxGenerationProblem constantScale =
            stochasticProblem(GenerationRecipe::affineIndexRemainder(
                {.dimensionCoefficients = {67, 1}, .positiveDivisor = 256}));
        constantScale.scale = mode;
        const MxGenerationResult result = generateMx(constantScale);
        for (size_t scaleIndex = 0; scaleIndex < result.scales.elementCount(); ++scaleIndex)
            require(std::to_integer<uint8_t>(
                        result.scales.rawEncodedBackingStorage()[scaleIndex]) == expectedRaw,
                    "MX explicit constant-scale generation mismatch.");
        checkReference(constantScale, result);
    }

    MxGenerationProblem maximumScale = stochasticProblem(GenerationRecipe::affineIndexRemainder(
        {.dimensionCoefficients = {67, 1}, .positiveDivisor = 256}));
    maximumScale.scale = MxScaleGenerationMode::Maximum;
    const MxGenerationResult maximumScaled = generateMx(maximumScale);
    for (size_t scaleIndex = 0; scaleIndex < maximumScaled.scales.elementCount(); ++scaleIndex)
        require(std::to_integer<uint8_t>(
                    maximumScaled.scales.rawEncodedBackingStorage()[scaleIndex]) == 0xfeU,
                "MX explicit maximum-scale generation mismatch.");
    checkReference(maximumScale, maximumScaled);

    MxGenerationProblem nanScale = stochasticProblem(GenerationRecipe::affineIndexRemainder(
        {.dimensionCoefficients = {67, 1}, .positiveDivisor = 256}));
    nanScale.scale = MxScaleGenerationMode::NaN;
    const MxGenerationResult nanScaled = generateMx(nanScale);
    for (size_t scaleIndex = 0; scaleIndex < nanScaled.scales.elementCount(); ++scaleIndex)
        require(std::to_integer<uint8_t>(nanScaled.scales.rawEncodedBackingStorage()[scaleIndex]) ==
                    0xffU,
                "MX explicit NaN-scale generation mismatch.");
    checkReference(nanScale, nanScaled);

    MxGenerationProblem impossibleInterval = explicitScale;
    impossibleInterval.shape = Shape{8192, 1};
    impossibleInterval.leadingDimension = 8192;
    impossibleInterval.data = MxDataGeneration::preserveRange(
        mxRecipe(GenerationRecipe::uniformReal({.lower = 0.1, .upper = 0.2})),
        {.lower = 0.1, .upper = 0.2});
    bool rejectedImpossibleInterval = false;
    try {
        (void)generateMx(impossibleInterval);
    } catch (const std::invalid_argument&) {
        rejectedImpossibleInterval = true;
    }
    require(rejectedImpossibleInterval,
            "MX parallel generation did not reject an unrepresentable bounded interval.");

    std::vector<MxGenerationProblem> stochasticProblems;
    stochasticProblems.push_back(boundedProblem());
    stochasticProblems.push_back(stochasticProblem(
        GenerationRecipe::uniformReal({.lower = 0.0, .upper = 1.0})
            .withAlternatingSign({.dimensions = {0, 1}, .negativeWhenOdd = true}),
        MxDataQuantization::PreserveRange, MxRepresentedValueRange{.lower = -1.0, .upper = 1.0}));
    stochasticProblems.push_back(unboundedProblem());
    stochasticProblems.push_back(
        stochasticProblem(GenerationRecipe::uniformReal(
                              {.lower = 0.0, .upper = 6.28318530717958647692528676655900576})
                              .withCosineTransform()));
    stochasticProblems.push_back(
        stochasticProblem(GenerationRecipe::normal({.mean = 0.0, .standardDeviation = 1.25}),
                          MxDataQuantization::Nearest, std::nullopt, MxScaleGenerationMode::Derived,
                          mx_generation_random_domain_version_1::normal));
    stochasticProblems.push_back(
        stochasticProblem(GenerationRecipe::uniformInteger({.lower = -4, .upper = 4})));
    for (MxGenerationProblem problem : stochasticProblems) {
#ifdef _OPENMP
        omp_set_dynamic(0);
        omp_set_num_threads(1);
#endif
        const MxGenerationResult oneThread = generateMx(problem);
#ifdef _OPENMP
        omp_set_num_threads(4);
#endif
        const MxGenerationResult fourThreads = generateMx(problem);
        require(sameResult(oneThread, fourThreads),
                "MX generation changed with OpenMP thread count.");
        checkReference(problem, oneThread);

        problem.data = problem.data.withSeed(problem.data.recipe().seed() + 1);
        const MxGenerationResult differentSeed = generateMx(problem);
        require(!sameStorage(oneThread.data, differentSeed.data) ||
                    !sameStorage(oneThread.scales, differentSeed.scales),
                "MX stochastic generation ignored the seed.");
    }

    return 0;
}
