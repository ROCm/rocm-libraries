// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <array>
#include <roc/host_numerics/backends/blas.hpp>
#include <span>

int main() {
    using namespace roc::host_numerics;

    const std::array<float, 4> a{1, 3, 2, 4};
    const std::array<float, 4> b{5, 7, 6, 8};
    const Layout layout(Shape{2, 2}, {1, 2});
    Tensor output(ScalarType::Float32, layout);
    referenceGemmIntoWithBlasBackend(
        Tensor::copyNativeStorage<float>(layout, std::span<const float>(a)),
        Tensor::copyNativeStorage<float>(layout, std::span<const float>(b)), output, output,
        GemmOptions{}, GemmBackend::Blas);

    return output.loadAs<float>({0, 0}) == 19 && output.loadAs<float>({1, 0}) == 43 &&
                   output.loadAs<float>({0, 1}) == 22 && output.loadAs<float>({1, 1}) == 50
               ? 0
               : 1;
}
