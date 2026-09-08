// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <array>
#include <roc/host_numerics/comparison.hpp>
#include <roc/host_numerics/gemm.hpp>
#include <roc/host_numerics/generation.hpp>
#include <roc/host_numerics/tensor_operations.hpp>

int main() {
    using namespace roc::host_numerics;

    const std::array<float, 6> aValues{1, 2, 3, 4, 5, 6};
    const std::array<float, 6> bValues{7, 8, 9, 10, 11, 12};
    const Tensor a = Tensor::copyNativeValues<float>(Shape{2, 3}, aValues);
    const Tensor b = Tensor::copyNativeValues<float>(Shape{3, 2}, bValues);

    const Tensor product = matmul(a, b, ScalarType::Float32);
    const Tensor bias =
        Tensor::copyNativeValues<float>(Shape{2}, std::array<float, 2>{-100.0f, 1.0f});
    const Tensor result = relu(add(multiply(product, 0.5f), bias));

    GenerationRecipe recipe = GenerationRecipe::realOnly(
        GenerationRecipe::uniformReal({.lower = -1.0, .upper = 1.0}), {.seed = 17});
    const Tensor random = generate(ScalarType::Float32, Shape{2, 2}, recipe);
    if (random.shape() != Shape{2, 2}) return 1;

    const Tensor expected = Tensor::copyNativeValues<float>(
        Shape{2, 2}, std::array<float, 4>{0.0f, 33.0f, 0.0f, 78.0f});
    return compare(result, expected, allCloseComparisonOptions()).passed() ? 0 : 1;
}
