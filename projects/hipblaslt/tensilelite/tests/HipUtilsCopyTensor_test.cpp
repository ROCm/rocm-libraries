// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <Tensile/TensorDescriptor.hpp>
#include <Tensile/hip/HipUtils.hpp>

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

#include <array>
#include <cstdint>

TEST(HipUtilsCopyTensor, CopiesContiguousSource)
{
    std::array<int32_t, 12> const src{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    std::array<int32_t, 12>       dst;
    dst.fill(-1);

    TensileLite::TensorDescriptor const desc("copy", rocisa::DataType::Int32, {3, 2, 2});
    TensileLite::hip::CopyTensorVoid(dst.data(), src.data(), desc, hipMemcpyHostToHost);
    ASSERT_EQ(hipStreamSynchronize(nullptr), hipSuccess);

    EXPECT_EQ(dst, src);
}

TEST(HipUtilsCopyTensor, CopiesPaddedSourceWithoutChangingPadding)
{
    // Include trailing padding so an over-wide copy is visible in the comparison.
    std::array<int32_t, 26> const src{1, 2, 3, 99, 99, 4,  5,  6,  99, 99, 99, 99, 99,
                                      7, 8, 9, 99, 99, 10, 11, 12, 99, 99, 99, 99, 99};
    std::array<int32_t, 26> const expected{1, 2, 3, -1, -1, 4,  5,  6,  -1, -1, -1, -1, -1,
                                           7, 8, 9, -1, -1, 10, 11, 12, -1, -1, -1, -1, -1};
    std::array<int32_t, 26>       dst;
    dst.fill(-1);

    TensileLite::TensorDescriptor const desc(
        "copy", rocisa::DataType::Int32, {3, 2, 2}, {1, 5, 13});
    TensileLite::hip::CopyTensorVoid(dst.data(), src.data(), desc, hipMemcpyHostToHost);
    ASSERT_EQ(hipStreamSynchronize(nullptr), hipSuccess);

    EXPECT_EQ(dst, expected);
}
