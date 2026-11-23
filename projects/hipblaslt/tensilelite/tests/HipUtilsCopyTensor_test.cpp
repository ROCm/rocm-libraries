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

TEST(HipUtilsCopyTensor, TypedCopyRepeatsAZeroStrideDimension)
{
    // A 4-element bias shared by 3 batches. Coalescing the batch dimension would copy
    // 3 elements instead of 4; the last element of dst is a canary.
    std::array<int32_t, 4> const src{1, 2, 3, 4};
    std::array<int32_t, 5>       dst;
    dst.fill(-1);

    TensileLite::TensorDescriptor const desc("copy", rocisa::DataType::Int32, {4, 3}, {1, 0});
    TensileLite::hip::CopyTensor(dst.data(), src.data(), desc, hipMemcpyHostToHost);
    ASSERT_EQ(hipStreamSynchronize(nullptr), hipSuccess);

    EXPECT_EQ(dst, (std::array<int32_t, 5>{1, 2, 3, 4, -1}));
}

TEST(HipUtilsCopyTensor, TypedCopyDoesNotReadPastAZeroStrideTensor)
{
    // A 2-element tensor shared by 5 batches. Coalescing the batch dimension would copy
    // 5 elements; the elements of src past the tensor are never copied.
    std::array<int32_t, 6> const src{1, 2, 77, 77, 77, 77};
    std::array<int32_t, 6>       dst;
    dst.fill(-1);

    TensileLite::TensorDescriptor const desc("copy", rocisa::DataType::Int32, {2, 5}, {1, 0});
    TensileLite::hip::CopyTensor(dst.data(), src.data(), desc, hipMemcpyHostToHost);
    ASSERT_EQ(hipStreamSynchronize(nullptr), hipSuccess);

    EXPECT_EQ(dst, (std::array<int32_t, 6>{1, 2, -1, -1, -1, -1}));
}

TEST(HipUtilsCopyTensor, VoidCopyRepeatsAZeroStrideDimension)
{
    // TypedCopyRepeatsAZeroStrideDimension through CopyTensorVoid.
    std::array<int32_t, 4> const src{1, 2, 3, 4};
    std::array<int32_t, 5>       dst;
    dst.fill(-1);

    TensileLite::TensorDescriptor const desc("copy", rocisa::DataType::Int32, {4, 3}, {1, 0});
    TensileLite::hip::CopyTensorVoid(dst.data(), src.data(), desc, hipMemcpyHostToHost);
    ASSERT_EQ(hipStreamSynchronize(nullptr), hipSuccess);

    EXPECT_EQ(dst, (std::array<int32_t, 5>{1, 2, 3, 4, -1}));
}

TEST(HipUtilsCopyTensor, VoidCopyDoesNotReadPastAZeroStrideTensor)
{
    // TypedCopyDoesNotReadPastAZeroStrideTensor through CopyTensorVoid.
    std::array<int32_t, 6> const src{1, 2, 77, 77, 77, 77};
    std::array<int32_t, 6>       dst;
    dst.fill(-1);

    TensileLite::TensorDescriptor const desc("copy", rocisa::DataType::Int32, {2, 5}, {1, 0});
    TensileLite::hip::CopyTensorVoid(dst.data(), src.data(), desc, hipMemcpyHostToHost);
    ASSERT_EQ(hipStreamSynchronize(nullptr), hipSuccess);

    EXPECT_EQ(dst, (std::array<int32_t, 6>{1, 2, -1, -1, -1, -1}));
}

TEST(HipUtilsCopyTensor, TypedCopyOfAZeroStrideFirstDimensionCopiesOneElement)
{
    // One element shared by every coordinate of the only dimension. Coalescing that
    // dimension would copy zero bytes.
    std::array<int32_t, 4> const src{7, 77, 77, 77};
    std::array<int32_t, 4>       dst;
    dst.fill(-1);

    TensileLite::TensorDescriptor const desc("copy", rocisa::DataType::Int32, {4}, {0});
    TensileLite::hip::CopyTensor(dst.data(), src.data(), desc, hipMemcpyHostToHost);
    ASSERT_EQ(hipStreamSynchronize(nullptr), hipSuccess);

    EXPECT_EQ(dst, (std::array<int32_t, 4>{7, -1, -1, -1}));
}

TEST(HipUtilsCopyTensor, TypedCopyOfAZeroStrideFirstDimensionKeepsTheOtherStrides)
{
    // Element (i, j) is at 4 * j, so only the first of every 4 elements is read.
    // Coalescing the zero-stride first dimension would copy zero bytes.
    std::array<int32_t, 8> const src{1, 77, 77, 77, 5, 77, 77, 77};
    std::array<int32_t, 8>       dst;
    dst.fill(-1);

    TensileLite::TensorDescriptor const desc("copy", rocisa::DataType::Int32, {4, 2}, {0, 4});
    TensileLite::hip::CopyTensor(dst.data(), src.data(), desc, hipMemcpyHostToHost);
    ASSERT_EQ(hipStreamSynchronize(nullptr), hipSuccess);

    EXPECT_EQ(dst, (std::array<int32_t, 8>{1, -1, -1, -1, 5, -1, -1, -1}));
}

TEST(HipUtilsCopyTensor, VoidCopyOfAZeroStrideFirstDimensionCopiesOneElement)
{
    // One element shared by every coordinate of the only dimension. Coalescing that
    // dimension would copy zero bytes.
    std::array<int32_t, 4> const src{7, 77, 77, 77};
    std::array<int32_t, 4>       dst;
    dst.fill(-1);

    TensileLite::TensorDescriptor const desc("copy", rocisa::DataType::Int32, {4}, {0});
    TensileLite::hip::CopyTensorVoid(dst.data(), src.data(), desc, hipMemcpyHostToHost);
    ASSERT_EQ(hipStreamSynchronize(nullptr), hipSuccess);

    EXPECT_EQ(dst, (std::array<int32_t, 4>{7, -1, -1, -1}));
}

TEST(HipUtilsCopyTensor, VoidCopyOfAZeroStrideFirstDimensionKeepsTheOtherStrides)
{
    // Element (i, j) is at 4 * j, so only the first of every 4 elements is read.
    // Coalescing the zero-stride first dimension would copy zero bytes.
    std::array<int32_t, 8> const src{1, 77, 77, 77, 5, 77, 77, 77};
    std::array<int32_t, 8>       dst;
    dst.fill(-1);

    TensileLite::TensorDescriptor const desc("copy", rocisa::DataType::Int32, {4, 2}, {0, 4});
    TensileLite::hip::CopyTensorVoid(dst.data(), src.data(), desc, hipMemcpyHostToHost);
    ASSERT_EQ(hipStreamSynchronize(nullptr), hipSuccess);

    EXPECT_EQ(dst, (std::array<int32_t, 8>{1, -1, -1, -1, 5, -1, -1, -1}));
}
