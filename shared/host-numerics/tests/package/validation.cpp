// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <array>
#include <cmath>
#include <roc/host_numerics/epilogue.hpp>
#include <roc/host_numerics/gemm.hpp>
#include <roc/host_numerics/generation.hpp>
#include <roc/host_numerics/layer_norm.hpp>
#include <roc/host_numerics/linear_combination.hpp>
#include <roc/host_numerics/reduction.hpp>
#include <roc/host_numerics/softmax.hpp>
#include <roc/host_numerics/structured_sparsity.hpp>
#include <span>
#include <utility>

int main() {
    using namespace roc::host_numerics;

    const std::array<float, 1> a{2};
    const std::array<float, 1> b{3};
    const std::array<float, 1> c{0};
    Tensor d(ScalarType::Float32, Shape{1, 1});

    const Tensor operandA = Tensor::copyNativeStorage<float>(
        Layout::contiguousLastDimensionFastest(Shape{1, 1}), std::span<const float>(a));
    const Tensor operandB = Tensor::copyNativeStorage<float>(
        Layout::contiguousLastDimensionFastest(Shape{1, 1}), std::span<const float>(b));
    const Tensor inputC = Tensor::copyNativeStorage<float>(
        Layout::contiguousLastDimensionFastest(Shape{1, 1}), std::span<const float>(c));
    if (!queryGemmSupport(operandA, operandB, inputC, d)) return 1;
    if (referenceGemmInto(operandA, operandB, inputC, d) != GemmBackend::Pointwise) return 1;
    if (d.loadAs<float>({0, 0}) != 6) return 1;
    if (referenceGemmInto(operandA, operandB, inputC, d, GemmOptions{}, GemmBackend::Blocked) !=
            GemmBackend::Blocked ||
        d.loadAs<float>({0, 0}) != 6)
        return 1;

    const Tensor ownedGemm = referenceGemm(
        Tensor::copyNativeStorage<float>(Layout::contiguousLastDimensionFastest(Shape{1, 1}),
                                         std::span<const float>(a)),
        Tensor::copyNativeStorage<float>(Layout::contiguousLastDimensionFastest(Shape{1, 1}),
                                         std::span<const float>(b)),
        Tensor::copyNativeStorage<float>(Layout::contiguousLastDimensionFastest(Shape{1, 1}),
                                         std::span<const float>(c)),
        ScalarType::Float32);
    if (ownedGemm.loadAs<float>({0, 0}) != 6) return 1;

    EpilogueOptions epilogueOptions;
    epilogueOptions.activation = Activation::Relu;
    const EpilogueOutputs epilogue = referenceEpilogue(
        Tensor::copyNativeValues<float>(Shape{1, 1}, std::array<float, 1>{-2.0f}),
        {.output = ScalarType::Float32, .amax = ScalarType::Float32}, epilogueOptions);
    if (epilogue.output.loadAs<float>({0, 0}) != 0 || !epilogue.amax ||
        epilogue.amax->loadAs<float>({0}) != 0)
        return 1;

    const std::array<float, 3> reductionInput{-1, 4, -3};
    const Tensor maximumAbsolute = referenceMaximumAbsolute(
        Tensor::copyNativeStorage<float>(Layout::contiguousLastDimensionFastest(Shape{3}),
                                         std::span<const float>(reductionInput)),
        ScalarType::Float32, ScalarType::Float32);
    if (maximumAbsolute.loadAs<float>({}) != 4) return 1;

    Tensor generated =
        generate(ScalarType::Float32, Shape{4},
                 GenerationRecipe::realOnly(GenerationRecipe::choice({.values = {-2.0, 3.0}}),
                                            {.seed = 17}));
    for (size_t index = 0; index < generated.elementCount(); ++index) {
        const float value = generated.loadAs<float>({index});
        if (value != -2.0f && value != 3.0f) return 1;
    }
    generateAt(generated, 2,
               GenerationRecipe::realOnly(GenerationRecipe::constant({.value = 11.0})));
    if (generated.loadAs<float>({2}) != 11.0f) return 1;

    LinearCombinationOptions linearCombinationOptions(ScalarType::Float32);
    linearCombinationOptions.alpha = 2.0;
    linearCombinationOptions.beta = -1.0;
    const Tensor linearCombinationOutput = linearCombination(
        Tensor::copyNativeStorage<float>(Layout::contiguousLastDimensionFastest(Shape{1}),
                                         std::span<const float>(a)),
        Tensor::copyNativeStorage<float>(Layout::contiguousLastDimensionFastest(Shape{1}),
                                         std::span<const float>(b)),
        ScalarType::Float32, linearCombinationOptions);
    if (linearCombinationOutput.loadAs<float>({0}) != 1.0f) return 1;

    const std::array<float, 2> softmaxValues{1.0f, 2.0f};
    const Tensor softmaxInput =
        Tensor::copyNativeValues<float>(Shape{1, 2}, std::span<const float>(softmaxValues));
    const Tensor softmax =
        referenceSoftmax(softmaxInput, 1, ScalarType::Float32, ScalarType::Float32);
    if (std::abs(softmax.loadAs<float>({0, 0}) + softmax.loadAs<float>({0, 1}) - 1.0f) > 1e-6f)
        return 1;

    LayerNormOptions layerNormOptions;
    layerNormOptions.axis = 1;
    const LayerNormOutputs layerNorm = referenceLayerNorm(softmaxInput,
                                                          {.output = ScalarType::Float32,
                                                           .mean = ScalarType::Float32,
                                                           .inverseVariance = ScalarType::Float32},
                                                          layerNormOptions);
    if (!layerNorm.mean || layerNorm.mean->loadAs<float>({0}) != 1.5f) return 1;

    StructuredSparsityPattern sparsityPattern;
    sparsityPattern.axis = 0;
    sparsityPattern.fixedPositions = {0, 2};
    const std::array<float, 4> sparseValues{1, 2, 3, 4};
    const StructuredSparseTensor sparse = applyStructuredSparsity(
        Tensor::copyNativeValues<float>(Shape{4}, std::span<const float>(sparseValues)),
        sparsityPattern, {.retainedIndices = true, .twoOfFourMetadata = true});
    if (sparse.pruned.loadAs<float>({1}) != 0 || sparse.compressed.loadAs<float>({1}) != 3 ||
        !sparse.retainedIndices || !sparse.twoOfFourMetadata ||
        sparse.twoOfFourMetadata->loadAs<uint8_t>({0}) != 0x08)
        return 1;
    return 0;
}
