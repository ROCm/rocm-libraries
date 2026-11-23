// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <array>
#include <roc/hostnumerics/tensor.hpp>
#include <span>

TEST(Package, Core) {
    using namespace roc::hostnumerics;
    Tensor tensor(ScalarType::Float32, Shape{2, 3});
    tensor.storeFrom({1, 2}, 4.0f);
    const Tensor converted = tensor.copyConvertedTo(ScalarType::Float16);
    const std::array<float, 3> native{1.0f, 2.0f, 3.0f};
    const Tensor runtimeTensor = Tensor::copyNativeStorage(std::span<const float>(native));
    const Tensor reversed = runtimeTensor.shareStorageWithLayout(Layout(Shape{3}, {-1}, 2));
    EXPECT_EQ(tensor.elementCount(), 6);
    EXPECT_EQ(tensor.loadAs<float>({1, 2}), 4.0f);
    EXPECT_EQ(converted.type(), ScalarType::Float16);
    EXPECT_EQ(converted.loadAs<float>({1, 2}), 4.0f);
    EXPECT_EQ(reversed.loadAs<float>({0}), 3.0f);
    EXPECT_EQ(reversed.loadAs<float>({2}), 1.0f);
    EXPECT_EQ(runtimeTensor.shape(), Shape{3});
}
