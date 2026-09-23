// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "host_numerics_test_support.hpp"

namespace host_numerics_test {
void testIndexedGeneration() {
    using namespace roc::host_numerics;

    Tensor serial(ScalarType::Float32, Shape{2, 3});
    generate(serial, GenerationRecipe::realOnly(GenerationRecipe::serialIndex()));
    require(serial.loadAs<float>({0, 0}) == 0 && serial.loadAs<float>({1, 0}) == 1 &&
                serial.loadAs<float>({0, 1}) == 2 && serial.loadAs<float>({1, 2}) == 5,
            "First-dimension-fast serial generation mismatch.");

    Tensor complex(ScalarType::ComplexFloat32, Shape{2, 2});
    generate(complex,
             GenerationRecipe::cartesian(GenerationRecipe::sine(), GenerationRecipe::cosine()));
    const std::complex<float> value = complex.loadAs<std::complex<float>>({1, 0});
    require(std::abs(value.real() - std::sin(1.0f)) < 1e-6f &&
                std::abs(value.imag() - std::cos(1.0f)) < 1e-6f,
            "Complex trigonometric generation mismatch.");

    Tensor candidates(ScalarType::Float32, Shape{8});
    const std::vector<double> candidateValues{-6.0, -1.5, 0.0, 4.0};
    const std::array<double, 8> expectedCandidateValues{0.0, -1.5, 0.0, -6.0, 0.0, -6.0, -1.5, 0.0};
    constexpr uint64_t candidateSeed = 37;
    generate(candidates,
             GenerationRecipe::realOnly(GenerationRecipe::choice({.values = candidateValues}),
                                        {.seed = candidateSeed}));
    for (size_t index = 0; index < candidates.elementCount(); ++index) {
        require(candidates.loadAs<float>({index}) == expectedCandidateValues[index],
                "Choice generation mismatch.");
    }

    Tensor point(ScalarType::Float32, Shape{2, 3, 2});
    generateAt(point, 3, GenerationRecipe::realOnly(GenerationRecipe::constant({.value = 9.0})));
    require(point.loadAs<float>({1, 1, 0}) == 9.0f && point.loadAs<float>({0, 1, 1}) == 0.0f,
            "First-dimension-fast point generation mismatch.");

    generateAt(point, 3,
               GenerationRecipe::realOnly(GenerationRecipe::constant({.value = 7.0}),
                                          {.indexOrder = IndexOrder::LastDimensionFastest}));
    require(point.loadAs<float>({0, 1, 1}) == 7.0f,
            "Last-dimension-fast point generation mismatch.");

    for (const IndexOrder order :
         {IndexOrder::FirstDimensionFastest, IndexOrder::LastDimensionFastest}) {
        Tensor whole(ScalarType::Float4E2M1, Shape{2, 3, 2});
        Tensor elementwise(ScalarType::Float4E2M1, Shape{2, 3, 2});
        const GenerationRecipe exactRecipe =
            GenerationRecipe::realOnly(GenerationRecipe::uniformInteger({.lower = -2, .upper = 2}),
                                       {.seed = 0x12345678, .indexOrder = order});
        generate(whole, exactRecipe);
        for (size_t index = 0; index < whole.elementCount(); ++index)
            generateAt(elementwise, index, exactRecipe);
        require(std::equal(whole.rawEncodedBackingStorage().begin(),
                           whole.rawEncodedBackingStorage().end(),
                           elementwise.rawEncodedBackingStorage().begin(),
                           elementwise.rawEncodedBackingStorage().end()),
                "Whole-tensor and elementwise generation encodings differ.");
    }

    Tensor packedRawWhole(ScalarType::Float6E3M2, Shape{13});
    Tensor packedRawElementwise(ScalarType::Float6E3M2, Shape{13});
    const GenerationRecipe packedRawRecipe = GenerationRecipe::realOnly(
        GenerationRecipe::uniformFiniteEncodedValue(), {.seed = 0x87654321});
    generate(packedRawWhole, packedRawRecipe);
    for (size_t index = 0; index < packedRawWhole.elementCount(); ++index)
        generateAt(packedRawElementwise, index, packedRawRecipe);
    require(std::equal(packedRawWhole.rawEncodedBackingStorage().begin(),
                       packedRawWhole.rawEncodedBackingStorage().end(),
                       packedRawElementwise.rawEncodedBackingStorage().begin(),
                       packedRawElementwise.rawEncodedBackingStorage().end()),
            "Whole-tensor and elementwise packed raw generation encodings differ.");

    Tensor choiceWhole(ScalarType::Float8E4M3, Shape{257});
    Tensor choiceElementwise(ScalarType::Float8E4M3, Shape{257});
    const GenerationRecipe choiceRecipe = GenerationRecipe::realOnly(
        GenerationRecipe::choice({.values = {-3.0, -0.5, 0.0, 1.5, 6.0}}), {.seed = 0x31415926});
    generate(choiceWhole, choiceRecipe);
    for (size_t index = 0; index < choiceWhole.elementCount(); ++index)
        generateAt(choiceElementwise, index, choiceRecipe);
    require(std::equal(choiceWhole.rawEncodedBackingStorage().begin(),
                       choiceWhole.rawEncodedBackingStorage().end(),
                       choiceElementwise.rawEncodedBackingStorage().begin(),
                       choiceElementwise.rawEncodedBackingStorage().end()),
            "Pre-encoded choice generation changed the encoded sequence.");

    Tensor absoluteWhole(ScalarType::Float6E2M3, Shape{257});
    Tensor absoluteElementwise(ScalarType::Float6E2M3, Shape{257});
    const GenerationRecipe absoluteRecipe = GenerationRecipe::realOnly(
        GenerationRecipe::absoluteUniformInteger({.lower = -7, .upper = 7}), {.seed = 0x16180339});
    generate(absoluteWhole, absoluteRecipe);
    for (size_t index = 0; index < absoluteWhole.elementCount(); ++index)
        generateAt(absoluteElementwise, index, absoluteRecipe);
    require(std::equal(absoluteWhole.rawEncodedBackingStorage().begin(),
                       absoluteWhole.rawEncodedBackingStorage().end(),
                       absoluteElementwise.rawEncodedBackingStorage().begin(),
                       absoluteElementwise.rawEncodedBackingStorage().end()),
            "Pre-encoded absolute-integer generation changed the encoded sequence.");

    const Shape alternatingShape{17, 19, 2};
    for (const ScalarType type :
         {ScalarType::Float64, ScalarType::Float32, ScalarType::Float16, ScalarType::BFloat16,
          ScalarType::Float8E4M3, ScalarType::Float4E2M1, ScalarType::Int32, ScalarType::Int8,
          ScalarType::Int4}) {
        for (const IndexOrder order :
             {IndexOrder::FirstDimensionFastest, IndexOrder::LastDimensionFastest}) {
            const Layout alternatingLayout =
                order == IndexOrder::FirstDimensionFastest
                    ? Layout::contiguousFirstDimensionFastest(alternatingShape)
                    : Layout::contiguousLastDimensionFastest(alternatingShape);
            Tensor alternatingWhole(type, alternatingLayout);
            Tensor alternatingElementwise(type, alternatingLayout);
            const GenerationRecipe alternatingRecipe = GenerationRecipe::realOnly(
                GenerationRecipe::uniformInteger({.lower = -4, .upper = 4})
                    .withAlternatingSign({.dimensions = {0, 1}, .negativeWhenOdd = false}),
                {.seed = 0x14142135, .indexOrder = order});
            generate(alternatingWhole, alternatingRecipe);
            for (size_t index = 0; index < alternatingWhole.elementCount(); ++index)
                generateAt(alternatingElementwise, index, alternatingRecipe);
            require(std::equal(alternatingWhole.rawEncodedBackingStorage().begin(),
                               alternatingWhole.rawEncodedBackingStorage().end(),
                               alternatingElementwise.rawEncodedBackingStorage().begin(),
                               alternatingElementwise.rawEncodedBackingStorage().end()),
                    "Pre-encoded alternating generation changed the encoded sequence.");
        }
    }

    for (const ScalarType type :
         {ScalarType::Float8E4M3, ScalarType::Float8E5M2, ScalarType::Float8E4M3Fnuz,
          ScalarType::Float8E5M2Fnuz, ScalarType::Float6E2M3, ScalarType::Float6E3M2,
          ScalarType::Float4E2M1}) {
        for (const UniformRealGenerationParameters parameters :
             {UniformRealGenerationParameters{.lower = -0.5, .upper = 0.5},
              UniformRealGenerationParameters{.lower = -1000.0, .upper = 1000.0}}) {
            Tensor uniformWhole(type, Shape{16385});
            Tensor uniformElementwise(type, Shape{16385});
            const GenerationRecipe uniformRecipe = GenerationRecipe::realOnly(
                GenerationRecipe::uniformReal(parameters), {.seed = 0x27182818});
            generate(uniformWhole, uniformRecipe);
            for (size_t index = 0; index < uniformWhole.elementCount(); ++index)
                generateAt(uniformElementwise, index, uniformRecipe);
            require(std::equal(uniformWhole.rawEncodedBackingStorage().begin(),
                               uniformWhole.rawEncodedBackingStorage().end(),
                               uniformElementwise.rawEncodedBackingStorage().begin(),
                               uniformElementwise.rawEncodedBackingStorage().end()),
                    "Uniform-real lookup changed the encoded sequence.");
        }
    }

    std::vector<std::byte> packedStorage(4, std::byte{0});
    packedStorage.back() = std::byte{0xc0};
    Tensor packedConstant = Tensor::takeOwnershipOfEncodedBackingStorage(
        ScalarType::Float6E2M3, Layout::contiguousLastDimensionFastest(Shape{5}),
        std::move(packedStorage));
    generate(packedConstant,
             GenerationRecipe::realOnly(GenerationRecipe::constant({.value = 1.0})));
    for (size_t index = 0; index < packedConstant.elementCount(); ++index)
        require(packedConstant.loadAs<float>({index}) == 1.0f,
                "Packed constant generation produced an incorrect value.");
    require((std::to_integer<uint8_t>(packedConstant.rawEncodedBackingStorage().back()) & 0xc0U) ==
                0xc0U,
            "Packed constant generation modified tail padding bits.");

    const Shape traversalShape{3, 4, 2};
    const Layout traversalLayout = Layout::contiguousFirstDimensionFastest(traversalShape);
    Tensor whole(ScalarType::Float32, traversalLayout);
    Tensor elementwise(ScalarType::Float32, traversalLayout);
    const GenerationRecipe traversalRecipe = GenerationRecipe::realOnly(
        GenerationRecipe::uniformReal({.lower = -2.0, .upper = 3.0}), {.seed = 0xabcdef});
    generate(whole, traversalRecipe);
    for (size_t index = 0; index < whole.elementCount(); ++index)
        generateAt(elementwise, index, traversalRecipe);
    require(
        std::equal(whole.rawEncodedBackingStorage().begin(), whole.rawEncodedBackingStorage().end(),
                   elementwise.rawEncodedBackingStorage().begin()),
        "Bulk generation changed first-dimension-fast values.");

    Tensor affine(ScalarType::Float32, Shape{2, 3, 2});
    generate(affine,
             GenerationRecipe::realOnly(
                 GenerationRecipe::affineIndexRemainder(
                     {.dimensionCoefficients = {1, -1, 2}, .offset = -2, .positiveDivisor = 5})
                     .withAffineValueMapping({.offset = 1.0})));
    require(affine.loadAs<float>({0, 0, 0}) == -1.0f && affine.loadAs<float>({1, 2, 1}) == 0.0f,
            "Affine-index remainder generation mismatch.");

#ifdef HOST_NUMERICS_TEST_OPENMP
    const int originalDynamic = omp_get_dynamic();
    const int originalThreadCount = omp_get_max_threads();
    omp_set_dynamic(0);

    const Layout paddedLayout(Shape{128, 96}, {1, 137});
    Tensor oneThread(ScalarType::Float32, paddedLayout);
    Tensor fourThreads(ScalarType::Float32, paddedLayout);
    const GenerationRecipe parallelRecipe =
        GenerationRecipe::realOnly(GenerationRecipe::uniformReal({.lower = -3.0, .upper = 7.0}),
                                   {.seed = 0x1020304050607080ULL});
    omp_set_num_threads(1);
    generate(oneThread, parallelRecipe);
    omp_set_num_threads(4);
    generate(fourThreads, parallelRecipe);
    require(std::equal(oneThread.rawEncodedBackingStorage().begin(),
                       oneThread.rawEncodedBackingStorage().end(),
                       fourThreads.rawEncodedBackingStorage().begin(),
                       fourThreads.rawEncodedBackingStorage().end()),
            "Ordinary generation changed with OpenMP thread count.");

    Tensor packedOneThread(ScalarType::Float6E3M2, Shape{8192});
    Tensor packedFourThreads(ScalarType::Float6E3M2, Shape{8192});
    const GenerationRecipe packedParallelRecipe = GenerationRecipe::realOnly(
        GenerationRecipe::uniformInteger({.lower = -28, .upper = 28})
            .withAlternatingSign({.dimensions = {0}, .negativeWhenOdd = true}),
        {.seed = 0x1020304050607080ULL});
    omp_set_num_threads(1);
    generate(packedOneThread, packedParallelRecipe);
    omp_set_num_threads(4);
    generate(packedFourThreads, packedParallelRecipe);
    require(std::equal(packedOneThread.rawEncodedBackingStorage().begin(),
                       packedOneThread.rawEncodedBackingStorage().end(),
                       packedFourThreads.rawEncodedBackingStorage().begin(),
                       packedFourThreads.rawEncodedBackingStorage().end()),
            "Packed generation changed with OpenMP thread count.");

    Tensor aliased(ScalarType::Float32, Layout(Shape{8192}, {0}));
    generate(aliased, GenerationRecipe::realOnly(GenerationRecipe::serialIndex()));
    require(aliased.loadAs<float>({0}) == 8191.0f,
            "Aliased generation did not preserve deterministic traversal order.");

    omp_set_num_threads(originalThreadCount);
    omp_set_dynamic(originalDynamic);
#endif
}

}  // namespace host_numerics_test
