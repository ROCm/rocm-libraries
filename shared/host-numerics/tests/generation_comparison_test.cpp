// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "host_numerics_test_support.hpp"

namespace host_numerics_test {
void testGenerationAndComparison() {
    using namespace roc::host_numerics;

    const GenerationRecipe binaryGeneration =
        GenerationRecipe::realOnly(GenerationRecipe::choice({.values = {-1.0, 1.0}}), {.seed = 42});
    Tensor a(ScalarType::Float32, Shape{32});
    Tensor b(ScalarType::Float32, Shape{32});
    generate(a, binaryGeneration);
    generate(b, binaryGeneration);
    require(compare(b, a).passed(), "Random generation is not repeatable for equal seeds.");

    b.storeFrom({7}, b.loadAs<float>({7}) + 1.0f);
    ComparisonOptions mismatchOptions;
    mismatchOptions.absoluteTolerance = 0.0;
    mismatchOptions.relativeTolerance = 0.0;
    mismatchOptions.maxReportedMismatches = 4;
    const auto result = compare(b, a, mismatchOptions);
    require(result.mismatches == 1, "Comparison did not count one mismatch.");
    require(result.reportedMismatches.size() == 1, "Comparison did not report one mismatch.");
    require(result.reportedMismatches[0].index == 7,
            "Comparison reported the wrong mismatch index.");
    const std::string diagnostic = formatComparisonReport(result);
    require(diagnostic.starts_with("comparison failed: 1 of 32 compared elements mismatched") &&
                diagnostic.find("index 7 [7]") != std::string::npos &&
                diagnostic.find("absolute difference") != std::string::npos,
            "Comparison diagnostic omitted its result summary or mismatch coordinates.");

    const std::array<double, 2> nonFiniteA{
        std::numeric_limits<double>::infinity(),
        1.0,
    };
    const std::array<double, 2> nonFiniteB{
        std::numeric_limits<double>::infinity(),
        std::numeric_limits<double>::infinity(),
    };
    ComparisonOptions nonFiniteOptions;
    nonFiniteOptions.relativeTolerance = 1.0;
    const auto nonFiniteResult =
        compare(Tensor::copyNativeStorage(std::span<const double>(nonFiniteA)),
                Tensor::copyNativeStorage(std::span<const double>(nonFiniteB)), nonFiniteOptions);
    require(nonFiniteResult.mismatches == 1,
            "Comparison did not distinguish finite and infinite values.");

    std::array<int, 8> generatedStorage;
    generatedStorage.fill(-1);
    std::vector<std::byte> generatedBytes(sizeof(generatedStorage));
    std::memcpy(generatedBytes.data(), generatedStorage.data(), generatedBytes.size());
    Tensor generated = Tensor::takeOwnershipOfEncodedBackingStorage(
        ScalarType::Int32, Layout(Shape{2, 2}, std::vector<ptrdiff_t>{1, 3}, 1),
        std::move(generatedBytes));
    generate(generated,
             [](std::span<const size_t> indices) { return 10 * indices[1] + indices[0]; });
    require(generated.loadAs<int>({0, 0}) == 0 && generated.loadAs<int>({1, 0}) == 1 &&
                generated.loadAs<int>({0, 1}) == 10 && generated.loadAs<int>({1, 1}) == 11,
            "Matrix generation produced incorrect logical values.");
    const auto generatedStorageValue = [&generated](size_t index) {
        int value;
        std::memcpy(&value, generated.rawEncodedBackingStorage().data() + index * sizeof(int),
                    sizeof(value));
        return value;
    };
    require(generatedStorageValue(0) == -1 && generatedStorageValue(3) == -1 &&
                generatedStorageValue(7) == -1,
            "Matrix generation modified padding.");

    Tensor runtimeExpected(ScalarType::Float32, Shape{2, 3});
    const GenerationRecipe runtimeGeneration = GenerationRecipe::realOnly(
        GenerationRecipe::uniformInteger({.lower = -2, .upper = 2}), {.seed = 7});
    generate(runtimeExpected, runtimeGeneration);
    Tensor runtimeObserved = runtimeExpected.deepCopy();
    runtimeObserved.storeFrom({1, 2}, runtimeExpected.loadAs<float>({1, 2}) + 1.0f);
    ComparisonOptions runtimeComparisonOptions;
    runtimeComparisonOptions.absoluteTolerance = 0.0;
    runtimeComparisonOptions.maxReportedMismatches = 2;
    const auto runtimeComparison =
        compare(runtimeObserved, runtimeExpected, runtimeComparisonOptions);
    require(runtimeComparison.compared == 6 && runtimeComparison.mismatches == 1 &&
                runtimeComparison.reportedMismatches[0].index == 5,
            "Runtime tensor generation/comparison mismatch.");
}

}  // namespace host_numerics_test
