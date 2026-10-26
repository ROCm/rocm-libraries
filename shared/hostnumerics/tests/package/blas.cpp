// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <array>
#include <roc/hostnumerics/backends/blas.hpp>
#include <span>

TEST(Package, Blas) {
    using namespace roc::hostnumerics;

    const std::array<float, 4> a{1, 3, 2, 4};
    const std::array<float, 4> b{5, 7, 6, 8};
    const Layout layout(Shape{2, 2}, {1, 2}, 0);
    Tensor output(ScalarType::Float32, layout);
    matmulIntoWithBlasBackend(Tensor::copyNativeStorage<float>(layout, std::span<const float>(a)),
                              Tensor::copyNativeStorage<float>(layout, std::span<const float>(b)),
                              output, MatmulOptions{ScalarType::Float32}, IndexRange::all(),
                              LastDimensionFastest{});

    EXPECT_EQ(output.loadAs<float>({0, 0}), 19);
    EXPECT_EQ(output.loadAs<float>({1, 0}), 43);
    EXPECT_EQ(output.loadAs<float>({0, 1}), 22);
    EXPECT_EQ(output.loadAs<float>({1, 1}), 50);
}
