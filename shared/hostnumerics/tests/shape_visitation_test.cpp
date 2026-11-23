// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <vector>

#include "hostnumerics_test_support.hpp"

namespace hostnumerics_test {
void testShapeVisitation() {
    using namespace roc::hostnumerics;
    const Tensor a = Tensor::copyNativeValues<float>(Shape{2, 2}, std::array<float, 4>{1, 2, 3, 4});
    const Tensor b = Tensor::copyNativeValues<float>(Shape{2, 2}, std::array<float, 4>{5, 6, 7, 8});
    const auto check = [&](const auto& indices) {
        Tensor d =
            Tensor::copyNativeValues<float>(Shape{2, 2}, std::array<float, 4>{-99, -99, -99, -99});
        matmulInto(a, b, d, MatmulOptions(ScalarType::Float32), indices, Permutation{1, 0},
                   GemmBackend::Blocked);
        require(d.loadAs<float>({1, 0}) == 43 && d.loadAs<float>({0, 1}) == -99 &&
                    d.loadAs<float>({0, 0}) == -99 && d.loadAs<float>({1, 1}) == -99,
                "Selected GEMM modified the wrong coordinates.");
    };
    check(IndexRange(1, 3, 2));
    check(std::set<size_t>{1});
}
}  // namespace hostnumerics_test
