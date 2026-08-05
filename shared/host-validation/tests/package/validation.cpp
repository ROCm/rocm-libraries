// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <array>
#include <roc/host_validation/validation.hpp>

int main() {
    using namespace roc::host_validation;

    const std::array<float, 1> a{2};
    const std::array<float, 1> b{3};
    const std::array<float, 1> c{0};
    std::array<float, 1> d{};

    GemmInvocation<float, float, float, float, float> invocation{
        ConstMatrixView<float>(a.data(), 1, 1, 1, 1), ConstMatrixView<float>(b.data(), 1, 1, 1, 1),
        ConstMatrixView<float>(c.data(), 1, 1, 1, 1), MatrixView<float>(d.data(), 1, 1, 1, 1)};
    referenceGemm(invocation);
    return d[0] == 6 ? 0 : 1;
}
