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

    Tensor aliased(ScalarType::Float32, Layout(Shape{8192}, {0}));
    generate(aliased, GenerationRecipe::realOnly(GenerationRecipe::serialIndex()));
    require(aliased.loadAs<float>({0}) == 8191.0f,
            "Aliased generation did not preserve deterministic traversal order.");

    omp_set_num_threads(originalThreadCount);
    omp_set_dynamic(originalDynamic);
#endif
}

}  // namespace host_numerics_test
