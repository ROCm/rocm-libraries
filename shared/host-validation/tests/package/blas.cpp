// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <array>
#include <roc/host_validation/backends/blas.hpp>
#include <span>

int main() {
    using namespace roc::host_validation;

    const std::array<float, 4> a{1, 3, 2, 4};
    const std::array<float, 4> b{5, 7, 6, 8};
    const Layout layout(Shape{2, 2}, {1, 2});
    Tensor output(ScalarType::Float32, layout);
    GemmRequest problem(
        GemmOperand(Tensor::copyNativeStorage<float>(layout, std::span<const float>(a))),
        GemmOperand(Tensor::copyNativeStorage<float>(layout, std::span<const float>(b))), output,
        output, ScalarType::Float32);
    BlasGemmBackend backend;
    referenceGemm(problem,
                  {
                      .backend = GemmBackend::Blas,
                      .requireRequestedBackend = true,
                  },
                  &backend);

    return output.loadAs<float>({0, 0}) == 19 && output.loadAs<float>({1, 0}) == 43 &&
                   output.loadAs<float>({0, 1}) == 22 && output.loadAs<float>({1, 1}) == 50
               ? 0
               : 1;
}
