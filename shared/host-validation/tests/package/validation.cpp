// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <array>
#include <cmath>
#include <roc/host_validation/axpby.hpp>
#include <roc/host_validation/gemm.hpp>
#include <roc/host_validation/generation.hpp>
#include <roc/host_validation/reduction.hpp>
#include <roc/host_validation/softmax.hpp>
#include <span>
#include <utility>

int main() {
    using namespace roc::host_validation;

    const std::array<float, 1> a{2};
    const std::array<float, 1> b{3};
    const std::array<float, 1> c{0};
    std::array<float, 1> d{};

    GemmProblem problem(
        GemmOperand(TensorView::fromNative<float>(Layout::contiguous(Shape{1, 1}),
                                                  std::span<const float>(a))),
        GemmOperand(TensorView::fromNative<float>(Layout::contiguous(Shape{1, 1}),
                                                  std::span<const float>(b))),
        TensorView::fromNative<float>(Layout::contiguous(Shape{1, 1}), std::span<const float>(c)),
        MutableTensorView::fromNative<float>(Layout::contiguous(Shape{1, 1}), std::span<float>(d)),
        ScalarType::Float32);
    GemmInvocation invocation(std::move(problem));
    if (!queryGemmSupport(invocation)) return 1;
    referenceGemm(invocation);
    if (d[0] != 6) return 1;

    const std::array<float, 3> reductionInput{-1, 4, -3};
    std::array<float, 1> maximumAbsolute{};
    referenceMaximumAbsolute(TensorView::fromNative<float>(Layout::contiguous(Shape{3}),
                                                           std::span<const float>(reductionInput)),
                             MutableTensorView::fromNative<float>(
                                 Layout::contiguous(Shape{}), std::span<float>(maximumAbsolute)),
                             ScalarType::Float32);
    if (maximumAbsolute[0] != 4) return 1;

    Tensor generated(ScalarType::Float32, Shape{4});
    GenerationOptions generation;
    generation.seed = 17;
    generation.real.pattern = GenerationPattern::CandidateSet;
    generation.real.candidates = {-2.0, 3.0};
    generate(generated.mutableView(), generation);
    for (size_t index = 0; index < generated.size(); ++index) {
        const float value = generated.view().loadAs<float>({index});
        if (value != -2.0f && value != 3.0f) return 1;
    }
    generation.real.pattern = GenerationPattern::Constant;
    generation.real.parameter0 = 11.0;
    generateAt(generated.mutableView(), 2, generation);
    if (generated.view().loadAs<float>({2}) != 11.0f) return 1;

    Tensor axpbyOutput(ScalarType::Float32, Shape{1});
    AxpbyProblem axpby(
        TensorView::fromNative<float>(Layout::contiguous(Shape{1}), std::span<const float>(a)),
        TensorView::fromNative<float>(Layout::contiguous(Shape{1}), std::span<const float>(b)),
        axpbyOutput.mutableView(), ScalarType::Float32);
    axpby.alpha = 2.0;
    axpby.beta = -1.0;
    if (referenceAxpby(axpby).elementsComputed != 1 ||
        axpbyOutput.view().loadAs<float>({0}) != 1.0f)
        return 1;

    const std::array<float, 2> softmaxValues{1.0f, 2.0f};
    const Tensor softmaxInput =
        Tensor::fromNativeValues<float>(Shape{1, 2}, std::span<const float>(softmaxValues));
    Tensor softmaxOutput(ScalarType::Float32, Shape{1, 2});
    const SoftmaxRunInfo softmax = referenceSoftmax(
        SoftmaxProblem(softmaxInput.view(), softmaxOutput.mutableView(), 1, ScalarType::Float32));
    if (softmax.slicesComputed != 1 ||
        std::abs(softmaxOutput.view().loadAs<float>({0, 0}) +
                 softmaxOutput.view().loadAs<float>({0, 1}) - 1.0f) > 1e-6f)
        return 1;
    return 0;
}
