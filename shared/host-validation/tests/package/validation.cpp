// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <array>
#include <cmath>
#include <roc/host_validation/axpby.hpp>
#include <roc/host_validation/gemm.hpp>
#include <roc/host_validation/generation.hpp>
#include <roc/host_validation/layer_norm.hpp>
#include <roc/host_validation/reduction.hpp>
#include <roc/host_validation/softmax.hpp>
#include <span>
#include <utility>

int main() {
    using namespace roc::host_validation;

    const std::array<float, 1> a{2};
    const std::array<float, 1> b{3};
    const std::array<float, 1> c{0};
    Tensor d(ScalarType::Float32, Shape{1, 1});

    GemmRequest problem(
        GemmOperand(
            Tensor::fromNative<float>(Layout::contiguous(Shape{1, 1}), std::span<const float>(a))),
        GemmOperand(
            Tensor::fromNative<float>(Layout::contiguous(Shape{1, 1}), std::span<const float>(b))),
        Tensor::fromNative<float>(Layout::contiguous(Shape{1, 1}), std::span<const float>(c)), d,
        ScalarType::Float32);
    if (!queryGemmSupport(problem)) return 1;
    referenceGemm(problem);
    if (d.loadAs<float>({0, 0}) != 6) return 1;

    const std::array<float, 3> reductionInput{-1, 4, -3};
    Tensor maximumAbsolute(ScalarType::Float32, Shape{});
    referenceMaximumAbsolute(Tensor::fromNative<float>(Layout::contiguous(Shape{3}),
                                                       std::span<const float>(reductionInput)),
                             maximumAbsolute, ScalarType::Float32);
    if (maximumAbsolute.loadAs<float>({}) != 4) return 1;

    Tensor generated =
        generate(ScalarType::Float32, Shape{4},
                 GenerationRecipe::realOnly(GenerationRecipe::candidateSet({.values = {-2.0, 3.0}}),
                                            {.seed = 17}));
    for (size_t index = 0; index < generated.size(); ++index) {
        const float value = generated.loadAs<float>({index});
        if (value != -2.0f && value != 3.0f) return 1;
    }
    generateAt(generated, 2,
               GenerationRecipe::realOnly(GenerationRecipe::constant({.value = 11.0})));
    if (generated.loadAs<float>({2}) != 11.0f) return 1;

    AxpbyProblem axpby(
        Tensor::fromNative<float>(Layout::contiguous(Shape{1}), std::span<const float>(a)),
        Tensor::fromNative<float>(Layout::contiguous(Shape{1}), std::span<const float>(b)),
        ScalarType::Float32, ScalarType::Float32);
    axpby.alpha = 2.0;
    axpby.beta = -1.0;
    const AxpbyResult axpbyResult = referenceAxpby(axpby);
    if (axpbyResult.runInfo.outputElementsWritten != 1 ||
        axpbyResult.output.loadAs<float>({0}) != 1.0f)
        return 1;

    const std::array<float, 2> softmaxValues{1.0f, 2.0f};
    const Tensor softmaxInput =
        Tensor::fromNativeValues<float>(Shape{1, 2}, std::span<const float>(softmaxValues));
    const SoftmaxResult softmax =
        referenceSoftmax(SoftmaxProblem(softmaxInput, ScalarType::Float32, 1, ScalarType::Float32));
    if (softmax.runInfo.slicesProcessed != 1 || softmax.runInfo.outputElementsWritten != 2 ||
        std::abs(softmax.output.loadAs<float>({0, 0}) + softmax.output.loadAs<float>({0, 1}) -
                 1.0f) > 1e-6f)
        return 1;

    Tensor layerNormOutput(ScalarType::Float32, Shape{1, 2});
    Tensor layerNormMean(ScalarType::Float32, Shape{1});
    Tensor layerNormInverseVariance(ScalarType::Float32, Shape{1});
    LayerNormProblem layerNorm(softmaxInput, layerNormOutput, 1, ScalarType::Float32);
    layerNorm.mean = layerNormMean;
    layerNorm.inverseVariance = layerNormInverseVariance;
    const LayerNormRunInfo layerNormRun = referenceLayerNorm(layerNorm);
    if (layerNormRun.slicesProcessed != 1 || layerNormRun.outputElementsWritten != 2 ||
        layerNormRun.meanElementsWritten != 1 || layerNormRun.inverseVarianceElementsWritten != 1 ||
        layerNormMean.loadAs<float>({0}) != 1.5f)
        return 1;
    return 0;
}
