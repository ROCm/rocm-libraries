// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <array>
#include <roc/host_validation/validation.hpp>
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
    return maximumAbsolute[0] == 4 ? 0 : 1;
}
