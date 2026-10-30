// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <numbers>
#include <roc/hostnumerics/epilogue.hpp>
#include <roc/hostnumerics/gemm.hpp>
#include <roc/hostnumerics/generation.hpp>
#include <roc/hostnumerics/layer_norm.hpp>
#include <roc/hostnumerics/reduction.hpp>
#include <roc/hostnumerics/softmax.hpp>
#include <roc/hostnumerics/structured_sparsity.hpp>
#include <roc/hostnumerics/tensor_operations.hpp>
#include <span>
#include <utility>

TEST(Package, NamedActivations) {
    using namespace roc::hostnumerics;

    const Tensor input(-2.0);
    const double sigmoidValue = 1.0 / (1.0 + std::exp(2.0));
    const std::array<std::pair<Tensor, double>, 8> cases{{
        {absolute(input), 2.0},
        {relu(input), 0.0},
        {gelu(input),
         -(1.0 + std::tanh(std::sqrt(2.0 / std::numbers::pi) * (-2.0 + 0.044715 * -8.0)))},
        {sigmoid(input), sigmoidValue},
        {tanh(input, 0.75, 1.25), std::tanh(-1.5) * 1.25},
        {silu(input), -2.0 * sigmoidValue},
        {swish(input, 1.5), -2.0 / (1.0 + std::exp(3.0))},
        {clip(input, -1.0, 1.0), -1.0},
    }};
    for (const auto& [output, expected] : cases) {
        EXPECT_EQ(output.type(), ScalarType::Float64);
        EXPECT_EQ(output.shape(), Shape{});
        EXPECT_NEAR(output.item<double>(), expected, 1e-14);
    }
}

TEST(Package, Operations) {
    using namespace roc::hostnumerics;

    const std::array<float, 1> a{2};
    const std::array<float, 1> b{3};
    Tensor d(ScalarType::Float32, Shape{1, 1});

    const Tensor operandA = Tensor::copyNativeStorage<float>(
        Layout::contiguousLastDimensionFastest(Shape{1, 1}), std::span<const float>(a));
    const Tensor operandB = Tensor::copyNativeStorage<float>(
        Layout::contiguousLastDimensionFastest(Shape{1, 1}), std::span<const float>(b));
    matmulInto(operandA, operandB, d, MatmulOptions{ScalarType::Float32},
               IndexRange::all(d.elementCount()), LastDimensionFastest{}, GemmBackend::Automatic);
    ASSERT_FALSE((d.loadAs<float>({0, 0}) != 6));
    matmulInto(operandA, operandB, d, MatmulOptions{ScalarType::Float32},
               IndexRange::all(d.elementCount()), LastDimensionFastest{}, GemmBackend::Blocked);
    ASSERT_FALSE((d.loadAs<float>({0, 0}) != 6));

    const Tensor ownedGemm =
        matmul(Tensor::copyNativeStorage<float>(Layout::contiguousLastDimensionFastest(Shape{1, 1}),
                                                std::span<const float>(a)),
               Tensor::copyNativeStorage<float>(Layout::contiguousLastDimensionFastest(Shape{1, 1}),
                                                std::span<const float>(b)),
               ScalarType::Float32, MatmulOptions{ScalarType::Float32}, std::nullopt,
               GemmBackend::Automatic);
    ASSERT_FALSE((ownedGemm.loadAs<float>({0, 0}) != 6));

    EpilogueOptions epilogueOptions(ScalarType::Float32);
    epilogueOptions.activation = ReluActivation{};
    const EpilogueOutputs epilogue = referenceEpilogue(
        Tensor::copyNativeValues<float>(Shape{1, 1}, std::array<float, 1>{-2.0f}),
        {.output = ScalarType::Float32, .amax = ScalarType::Float32}, epilogueOptions);
    ASSERT_FALSE((epilogue.output.loadAs<float>({0, 0}) != 0 || !epilogue.amax ||
                  epilogue.amax->loadAs<float>({0}) != 0));

    const std::array<float, 3> reductionInput{-1, 4, -3};
    const Tensor maximumAbsolute = referenceMaximumAbsolute(
        Tensor::copyNativeStorage<float>(Layout::contiguousLastDimensionFastest(Shape{3}),
                                         std::span<const float>(reductionInput)),
        ScalarType::Float32, ScalarType::Float32);
    ASSERT_FALSE((maximumAbsolute.loadAs<float>({}) != 4));

    Tensor generated = generate(ScalarType::Float32, Shape{4},
                                GenerationRecipe::realOnlyLastDimensionFastest(
                                    GenerationRecipe::choice({.values = {-2.0, 3.0}}), 17));
    for (size_t index = 0; index < generated.elementCount(); ++index) {
        const float value = generated.loadAs<float>({index});
        ASSERT_FALSE((value != -2.0f && value != 3.0f));
    }
    generateAt(generated, 2,
               GenerationRecipe::realOnlyLastDimensionFastest(
                   GenerationRecipe::constant({.value = 11.0}), 0));
    ASSERT_FALSE((generated.loadAs<float>({2}) != 11.0f));

    const Tensor scaledA =
        multiply(Tensor::copyNativeStorage<float>(Layout::contiguousLastDimensionFastest(Shape{1}),
                                                  std::span<const float>(a)),
                 Tensor(2.0f));
    const Tensor scaledB =
        multiply(Tensor::copyNativeStorage<float>(Layout::contiguousLastDimensionFastest(Shape{1}),
                                                  std::span<const float>(b)),
                 Tensor(-1.0f));
    ASSERT_FALSE((add(scaledA, scaledB).loadAs<float>({0}) != 1.0f));

    const std::array<float, 2> softmaxValues{1.0f, 2.0f};
    const Tensor softmaxInput =
        Tensor::copyNativeValues<float>(Shape{1, 2}, std::span<const float>(softmaxValues));
    const Tensor softmax =
        referenceSoftmax(softmaxInput, 1, ScalarType::Float32, ScalarType::Float32);
    ASSERT_FALSE(
        (std::abs(softmax.loadAs<float>({0, 0}) + softmax.loadAs<float>({0, 1}) - 1.0f) > 1e-6f));

    LayerNormOptions layerNormOptions;
    layerNormOptions.axis = 1;
    const LayerNormOutputs layerNorm = referenceLayerNorm(softmaxInput,
                                                          {.output = ScalarType::Float32,
                                                           .mean = ScalarType::Float32,
                                                           .inverseVariance = ScalarType::Float32},
                                                          layerNormOptions);
    ASSERT_FALSE((!layerNorm.mean || layerNorm.mean->loadAs<float>({0}) != 1.5f));

    StructuredSparsityPattern sparsityPattern;
    sparsityPattern.indexOrder = Permutation{0};
    sparsityPattern.axis = 0;
    sparsityPattern.fixedPositions = {0, 2};
    const std::array<float, 4> sparseValues{1, 2, 3, 4};
    const StructuredSparseTensor sparse = applyStructuredSparsity(
        Tensor::copyNativeValues<float>(Shape{4}, std::span<const float>(sparseValues)),
        sparsityPattern, {.retainedIndices = true, .twoOfFourMetadata = true});
    ASSERT_FALSE((sparse.pruned.loadAs<float>({1}) != 0 ||
                  sparse.compressed.loadAs<float>({1}) != 3 || !sparse.retainedIndices ||
                  !sparse.twoOfFourMetadata ||
                  sparse.twoOfFourMetadata->loadAs<uint8_t>({0}) != 0x08));
}
