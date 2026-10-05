// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "hipdnn_data_sdk/types/Half.hpp"
#include <gtest/gtest.h>

#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/resample_common_generated.h>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceResampleBwd.hpp>
#include <hipdnn_test_sdk/utilities/detail/CpuFpReferenceUtilities.hpp>

#include <cmath>
#include <limits>
#include <vector>

using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_data_sdk::types;
using namespace hipdnn_flatbuffers_sdk::data_objects;
using namespace hipdnn_test_sdk::utilities;
using hipdnn_test_sdk::detail::safeTestTypeCast;

// ============================================================================
// Rejection tests
// ============================================================================

TEST(TestCpuFpReferenceResampleBwd, RejectsUnsupportedRanks)
{
    // Rank < 4
    const Tensor<float> dy3D({1, 2, 2});
    Tensor<float> dx3D({1, 4, 4});

    EXPECT_THROW((CpuFpReferenceResampleBwd::backward<float>(dy3D,
                                                             dx3D,
                                                             {0},
                                                             {1},
                                                             {2},
                                                             ResampleMode::AVGPOOL_INCLUDE_PADDING,
                                                             PaddingMode::ZERO_PAD)),
                 std::runtime_error);

    // Rank > 5
    const Tensor<float> dy6D({1, 1, 2, 2, 2, 2});
    Tensor<float> dx6D({1, 1, 4, 4, 4, 4});

    EXPECT_THROW((CpuFpReferenceResampleBwd::backward<float>(dy6D,
                                                             dx6D,
                                                             {0, 0, 0, 0},
                                                             {1, 1, 1, 1},
                                                             {2, 2, 2, 2},
                                                             ResampleMode::AVGPOOL_INCLUDE_PADDING,
                                                             PaddingMode::ZERO_PAD)),
                 std::runtime_error);

    // Rank Mismatch
    const Tensor<float> dy4D({1, 1, 2, 2});
    Tensor<float> dx5D({1, 1, 4, 4, 4});

    EXPECT_THROW((CpuFpReferenceResampleBwd::backward<float>(dy4D,
                                                             dx5D,
                                                             {0, 0, 0},
                                                             {1, 1, 1},
                                                             {2, 2, 2},
                                                             ResampleMode::AVGPOOL_INCLUDE_PADDING,
                                                             PaddingMode::ZERO_PAD)),
                 std::runtime_error);
}

TEST(TestCpuFpReferenceResampleBwd, RejectsBatchAndChannelMismatches)
{
    // Batch size mismatch
    const Tensor<float> dyBatch({1, 1, 2, 2});
    Tensor<float> dxBatch({2, 1, 4, 4});

    EXPECT_THROW((CpuFpReferenceResampleBwd::backward<float>(dyBatch,
                                                             dxBatch,
                                                             {0, 0},
                                                             {2, 2},
                                                             {2, 2},
                                                             ResampleMode::AVGPOOL_INCLUDE_PADDING,
                                                             PaddingMode::ZERO_PAD)),
                 std::runtime_error);

    // Channel count mismatch
    const Tensor<float> dyChannel({1, 1, 2, 2});
    Tensor<float> dxChannel({1, 3, 4, 4});

    EXPECT_THROW((CpuFpReferenceResampleBwd::backward<float>(dyChannel,
                                                             dxChannel,
                                                             {0, 0},
                                                             {2, 2},
                                                             {2, 2},
                                                             ResampleMode::AVGPOOL_INCLUDE_PADDING,
                                                             PaddingMode::ZERO_PAD)),
                 std::runtime_error);
}

TEST(TestCpuFpReferenceResampleBwd, RejectsMismatchedSpatialParameterRanks)
{
    const Tensor<float> dy({1, 1, 2, 2});
    Tensor<float> dx({1, 1, 4, 4});

    // Padding rank mismatch
    EXPECT_THROW((CpuFpReferenceResampleBwd::backward<float>(dy,
                                                             dx,
                                                             {0, 0, 0},
                                                             {2, 2},
                                                             {2, 2},
                                                             ResampleMode::AVGPOOL_INCLUDE_PADDING,
                                                             PaddingMode::ZERO_PAD)),
                 std::runtime_error);

    // Stride rank mismatch
    EXPECT_THROW((CpuFpReferenceResampleBwd::backward<float>(dy,
                                                             dx,
                                                             {0, 0},
                                                             {2},
                                                             {2, 2},
                                                             ResampleMode::AVGPOOL_INCLUDE_PADDING,
                                                             PaddingMode::ZERO_PAD)),
                 std::runtime_error);

    // Window rank mismatch
    EXPECT_THROW((CpuFpReferenceResampleBwd::backward<float>(dy,
                                                             dx,
                                                             {0, 0},
                                                             {2, 2},
                                                             {2, 2, 2},
                                                             ResampleMode::AVGPOOL_INCLUDE_PADDING,
                                                             PaddingMode::ZERO_PAD)),
                 std::runtime_error);
}

TEST(TestCpuFpReferenceResampleBwd, RejectsNotSetAndUnsupportedIndexTensors)
{
    const Tensor<float> dy({1, 1, 2, 2});
    Tensor<float> dx({1, 1, 4, 4});

    // No index tensor for MAXPOOL mode
    EXPECT_THROW((CpuFpReferenceResampleBwd::backward<float>(
                     dy, dx, {0, 0}, {2, 2}, {2, 2}, ResampleMode::MAXPOOL, PaddingMode::ZERO_PAD)),
                 std::runtime_error);

    // Index tensor shape not matching dy shape
    const Tensor<int32_t> mismatchedIndex({1, 1, 4, 4});
    EXPECT_THROW((CpuFpReferenceResampleBwd::backward<float>(dy,
                                                             dx,
                                                             {0, 0},
                                                             {2, 2},
                                                             {2, 2},
                                                             ResampleMode::MAXPOOL,
                                                             PaddingMode::ZERO_PAD,
                                                             &mismatchedIndex)),
                 std::runtime_error);
}

TEST(TestCpuFpReferenceResampleBwd, RejectsUnsupportedResampleMode)
{
    const Tensor<float> dy({1, 1, 2, 2});
    Tensor<float> dx({1, 1, 4, 4});

    EXPECT_THROW((CpuFpReferenceResampleBwd::backward<float>(
                     dy, dx, {0, 0}, {2, 2}, {2, 2}, ResampleMode::NOT_SET, PaddingMode::ZERO_PAD)),
                 std::runtime_error);
}

// ============================================================================
// Verification tests
// ============================================================================

namespace
{

template <typename Type>
void fillValues(Tensor<Type>& tensor, const std::vector<float>& values)
{
    ASSERT_EQ(static_cast<size_t>(tensor.elementCount()), values.size());
    auto* data = tensor.memory().hostData();
    for(size_t i = 0; i < values.size(); ++i)
    {
        data[i] = safeTestTypeCast<Type>(values[i]);
    }
    tensor.memory().markHostModified();
}

template <typename Type>
void expectTensorValues(const Tensor<Type>& tensor, const std::vector<float>& expected)
{
    ASSERT_EQ(static_cast<size_t>(tensor.elementCount()), expected.size());
    const auto* data = tensor.memory().hostData();
    for(size_t i = 0; i < expected.size(); ++i)
    {
        EXPECT_FLOAT_EQ(static_cast<float>(data[i]), expected[i]) << "Mismatch at index " << i;
    }
}

} // namespace

TEST(TestCpuFpReferenceResampleBwd, MaxPoolScattersGradientsToIndices)
{
    Tensor<float> dy({1, 1, 2, 2});
    Tensor<float> dx({1, 1, 3, 3});
    Tensor<int32_t> index({1, 1, 2, 2});

    fillValues(dy, {1.0f, 2.0f, 3.0f, 4.0f});
    fillValues(index, {4.0f, 5.0f, 7.0f, 8.0f});

    CpuFpReferenceResampleBwd::backward<float, float, float, int32_t>(
        dy, dx, {0, 0}, {1, 1}, {2, 2}, ResampleMode::MAXPOOL, PaddingMode::ZERO_PAD, &index);

    // dx should be zero everywhere except at indices {4, 5, 7, 8}
    expectTensorValues(dx, {0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 2.0f, 0.0f, 3.0f, 4.0f});
}

TEST(TestCpuFpReferenceResampleBwd, MaxPoolAccumulatesDuplicateIndices)
{
    Tensor<float> dy({1, 1, 2, 2});
    Tensor<float> dx({1, 1, 3, 3});
    Tensor<int32_t> index({1, 1, 2, 2});

    fillValues(dy, {1.0f, 2.0f, 3.0f, 4.0f});
    // Three maxpool outputs map to the center of the 3x3 input
    // and one maps to the bottom-right corner
    fillValues(index, {4.0f, 4.0f, 4.0f, 8.0f});

    CpuFpReferenceResampleBwd::backward<float, float, float, int32_t>(
        dy, dx, {0, 0}, {1, 1}, {2, 2}, ResampleMode::MAXPOOL, PaddingMode::ZERO_PAD, &index);

    // dx[4] = 1.0 + 2.0 + 3.0 = 6.0, dx[8] = 4.0
    expectTensorValues(dx, {0.0f, 0.0f, 0.0f, 0.0f, 6.0f, 0.0f, 0.0f, 0.0f, 4.0f});
}

TEST(TestCpuFpReferenceResampleBwd, MaxPoolIgnoresNegativeIndex)
{
    Tensor<float> dy({1, 1, 2, 2});
    Tensor<float> dx({1, 1, 3, 3});
    Tensor<int32_t> index({1, 1, 2, 2});

    fillValues(dy, {1.0f, 2.0f, 3.0f, 4.0f});
    fillValues(index, {4.0f, -1.0f, -1.0f, 8.0f});

    CpuFpReferenceResampleBwd::backward<float, float, float, int32_t>(
        dy, dx, {0, 0}, {1, 1}, {2, 2}, ResampleMode::MAXPOOL, PaddingMode::ZERO_PAD, &index);

    // dx[4] = 1.0, dx[8] = 4.0
    expectTensorValues(dx, {0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 4.0f});
}

TEST(TestCpuFpReferenceResampleBwd, AverageExcludePaddingDividesByValidElementCount)
{
    Tensor<float> dy({1, 1, 2, 2});
    Tensor<float> dx({1, 1, 2, 2});

    fillValues(dy, {1.0f, 2.0f, 3.0f, 4.0f});

    // Padding = 1, Window = 2x2, Stride = 1:
    // dy[0] maps to dx[0] with 1 valid element
    // dy[1] maps to dx[0] and dx[1] with 2 valid elements
    // dy[2] maps to dx[0] and dx[2] with 2 valid elements
    // dy[3] maps to dx[0], dx[1], dx[2], and dx[3] with 4 valid elements
    CpuFpReferenceResampleBwd::backward<float, float, float>(dy,
                                                             dx,
                                                             {1, 1},
                                                             {1, 1},
                                                             {2, 2},
                                                             ResampleMode::AVGPOOL_EXCLUDE_PADDING,
                                                             PaddingMode::ZERO_PAD);

    // dx[0] = 1.0 / 1 + 2.0 / 2 + 3.0 / 2 + 4.0 / 4 = 1.0 + 1.0 + 1.5 + 1.0 = 4.5
    // dx[1] = 2.0 / 2 + 4.0 / 4 = 1.0 + 1.0 = 2.0
    // dx[2] = 3.0 / 2 + 4.0 / 4 = 1.5 + 1.0 = 2.5
    // dx[3] = 4.0 / 4 = 1.0
    expectTensorValues(dx, {4.5f, 2.0f, 2.5f, 1.0f});
}

TEST(TestCpuFpReferenceResampleBwd, AverageIncludePaddingDividesByWindowSize)
{
    Tensor<float> dy({1, 1, 2, 2});
    Tensor<float> dx({1, 1, 2, 2});

    fillValues(dy, {1.0f, 2.0f, 3.0f, 4.0f});

    // Window size = 4. Divisor is always 4 and all
    // elements contribute to the average
    CpuFpReferenceResampleBwd::backward<float, float, float>(dy,
                                                             dx,
                                                             {1, 1},
                                                             {1, 1},
                                                             {2, 2},
                                                             ResampleMode::AVGPOOL_INCLUDE_PADDING,
                                                             PaddingMode::ZERO_PAD);

    // dx[0] = 1.0 / 4 + 2.0 / 4 + 3.0 / 4 + 4.0 / 4 = 0.25 + 0.5 + 0.75 + 1.0 = 2.5
    // dx[1] = 2.0 / 4 + 4.0 / 4 = 0.5 + 1.0 = 1.5
    // dx[2] = 3.0 / 4 + 4.0 / 4 = 0.75 + 1.0 = 1.75
    // dx[3] = 4.0 / 4 = 1.0
    expectTensorValues(dx, {2.5f, 1.5f, 1.75f, 1.0f});
}

TEST(TestCpuFpReferenceResampleBwd, AverageExcludePaddingPropagatesNanAndInfinity)
{
    Tensor<float> dy({1, 1, 1, 2});
    Tensor<float> dx({1, 1, 1, 2});

    auto* dyData = dy.memory().hostData();
    dyData[0] = std::numeric_limits<float>::quiet_NaN();
    dyData[1] = std::numeric_limits<float>::infinity();
    dy.memory().markHostModified();

    CpuFpReferenceResampleBwd::backward<float, float, float>(dy,
                                                             dx,
                                                             {0, 0},
                                                             {1, 1},
                                                             {1, 1},
                                                             ResampleMode::AVGPOOL_EXCLUDE_PADDING,
                                                             PaddingMode::ZERO_PAD);

    EXPECT_TRUE(std::isnan(dx.getHostValue({0, 0, 0, 0})));
    EXPECT_EQ(dx.getHostValue({0, 0, 0, 1}), std::numeric_limits<float>::infinity());
}

TEST(TestCpuFpReferenceResampleBwd, MaxPoolSupportsChannelLastLayout)
{
    Tensor<float> dy({1, 2, 2, 2}, TensorLayout::NHWC);
    Tensor<float> dx({1, 2, 3, 3}, TensorLayout::NHWC);
    Tensor<int32_t> index({1, 2, 2, 2}, TensorLayout::NHWC);

    fillValues(dy, {1.0f, 10.0f, 2.0f, 20.0f, 3.0f, 30.0f, 4.0f, 40.0f});
    fillValues(index, {4.0f, 4.0f, 5.0f, 5.0f, 7.0f, 7.0f, 8.0f, 8.0f});

    CpuFpReferenceResampleBwd::backward<float, float, float, int32_t>(
        dy, dx, {0, 0}, {1, 1}, {2, 2}, ResampleMode::MAXPOOL, PaddingMode::ZERO_PAD, &index);

    expectTensorValues(dx,
                       {0.0f,
                        0.0f,
                        0.0f,
                        0.0f,
                        0.0f,
                        0.0f,
                        0.0f,
                        0.0f,
                        1.0f,
                        10.0f,
                        2.0f,
                        20.0f,
                        0.0f,
                        0.0f,
                        3.0f,
                        30.0f,
                        4.0f,
                        40.0f});
}

TEST(TestCpuFpReferenceResampleBwd, AverageExcludePaddingSupportsChannelLastLayout)
{
    Tensor<float> dy({1, 2, 2, 2}, TensorLayout::NHWC);
    Tensor<float> dx({1, 2, 2, 2}, TensorLayout::NHWC);

    fillValues(dy, {1.0f, 10.0f, 2.0f, 20.0f, 3.0f, 30.0f, 4.0f, 40.0f});

    CpuFpReferenceResampleBwd::backward<float, float, float>(dy,
                                                             dx,
                                                             {1, 1},
                                                             {1, 1},
                                                             {2, 2},
                                                             ResampleMode::AVGPOOL_EXCLUDE_PADDING,
                                                             PaddingMode::ZERO_PAD);

    expectTensorValues(dx, {4.5f, 45.0f, 2.0f, 20.0f, 2.5f, 25.0f, 1.0f, 10.0f});
}

TEST(TestCpuFpReferenceResampleBwd, AverageExcludePaddingNonOverlappingStride)
{
    Tensor<float> dy({1, 1, 2, 2});
    Tensor<float> dx({1, 1, 4, 4});

    fillValues(dy, {1.0f, 2.0f, 3.0f, 4.0f});

    CpuFpReferenceResampleBwd::backward<float, float, float>(dy,
                                                             dx,
                                                             {0, 0},
                                                             {2, 2},
                                                             {2, 2},
                                                             ResampleMode::AVGPOOL_EXCLUDE_PADDING,
                                                             PaddingMode::ZERO_PAD);

    expectTensorValues(dx,
                       {0.25f,
                        0.25f,
                        0.5f,
                        0.5f,
                        0.25f,
                        0.25f,
                        0.5f,
                        0.5f,
                        0.75f,
                        0.75f,
                        1.0f,
                        1.0f,
                        0.75f,
                        0.75f,
                        1.0f,
                        1.0f});
}

TEST(TestCpuFpReferenceResampleBwd, AverageExcludePaddingHandlesOverlappingStride)
{
    Tensor<float> dy({1, 1, 2, 2});
    Tensor<float> dx({1, 1, 5, 5});

    fillValues(dy, {1.0f, 2.0f, 3.0f, 4.0f});

    CpuFpReferenceResampleBwd::backward<float, float, float>(dy,
                                                             dx,
                                                             {0, 0},
                                                             {2, 2},
                                                             {3, 3},
                                                             ResampleMode::AVGPOOL_EXCLUDE_PADDING,
                                                             PaddingMode::ZERO_PAD);

    expectTensorValues(dx, {1.0f / 9.0f, 1.0f / 9.0f, 1.0f / 3.0f,  2.0f / 9.0f, 2.0f / 9.0f,
                            1.0f / 9.0f, 1.0f / 9.0f, 1.0f / 3.0f,  2.0f / 9.0f, 2.0f / 9.0f,
                            4.0f / 9.0f, 4.0f / 9.0f, 10.0f / 9.0f, 2.0f / 3.0f, 2.0f / 3.0f,
                            1.0f / 3.0f, 1.0f / 3.0f, 7.0f / 9.0f,  4.0f / 9.0f, 4.0f / 9.0f,
                            1.0f / 3.0f, 1.0f / 3.0f, 7.0f / 9.0f,  4.0f / 9.0f, 4.0f / 9.0f});
}

TEST(TestCpuFpReferenceResampleBwd, MaxPoolHandlesOverlappingStride)
{
    Tensor<float> dy({1, 1, 2, 2});
    Tensor<float> dx({1, 1, 5, 5});
    Tensor<int32_t> index({1, 1, 2, 2});

    fillValues(dy, {1.0f, 2.0f, 3.0f, 4.0f});
    fillValues(index, {6.0f, 8.0f, 16.0f, 18.0f});

    CpuFpReferenceResampleBwd::backward<float, float, float, int32_t>(
        dy, dx, {0, 0}, {2, 2}, {3, 3}, ResampleMode::MAXPOOL, PaddingMode::ZERO_PAD, &index);

    expectTensorValues(dx, {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 2.0f,
                            0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 3.0f, 0.0f,
                            4.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f});
}

// ============================================================================
// Smoke tests
// ============================================================================

TEST(TestCpuFpReferenceResampleBwdFp32, MaxPool4DNchw)
{
    Tensor<float> dy({2, 3, 4, 4});
    Tensor<float> dx({2, 3, 5, 5});
    Tensor<int32_t> index({2, 3, 4, 4});

    dy.fillWithValue(1.0f);
    index.fillWithValue(0);

    CpuFpReferenceResampleBwd::backward<float, float, float, int32_t>(
        dy, dx, {0, 0}, {1, 1}, {2, 2}, ResampleMode::MAXPOOL, PaddingMode::ZERO_PAD, &index);

    const auto* dxData = dx.memory().hostData();
    for(size_t n = 0; n < 2; ++n)
    {
        for(size_t c = 0; c < 3; ++c)
        {
            const auto base = (n * 3 + c) * 25;
            for(size_t i = 0; i < 25; ++i)
            {
                const float expected = (i == 0) ? 1.0f : 0.0f;
                EXPECT_FLOAT_EQ(dxData[base + i], expected)
                    << "Mismatch at n=" << n << " c=" << c << " spatial index " << i;
            }
        }
    }
}

TEST(TestCpuFpReferenceResampleBwdFp32, AverageExcludePadding4DNchw)
{
    Tensor<float> dy({2, 3, 4, 4});
    Tensor<float> dx({2, 3, 5, 5});

    dy.fillWithValue(1.0f);

    CpuFpReferenceResampleBwd::backward<float, float, float>(dy,
                                                             dx,
                                                             {0, 0},
                                                             {1, 1},
                                                             {2, 2},
                                                             ResampleMode::AVGPOOL_EXCLUDE_PADDING,
                                                             PaddingMode::ZERO_PAD);

    const auto* dxData = dx.memory().hostData();
    const std::array<float, 5> factors{1.0f, 2.0f, 2.0f, 2.0f, 1.0f};
    for(size_t n = 0; n < 2; ++n)
    {
        for(size_t c = 0; c < 3; ++c)
        {
            const auto base = (n * 3 + c) * 25;
            for(size_t h = 0; h < 5; ++h)
            {
                for(size_t w = 0; w < 5; ++w)
                {
                    const float expected = factors[h] * factors[w] * 1.0f / 4.0f;
                    EXPECT_FLOAT_EQ(dxData[base + h * 5 + w], expected)
                        << "Mismatch at n=" << n << " c=" << c << " h=" << h << " w=" << w;
                }
            }
        }
    }
}

TEST(TestCpuFpReferenceResampleBwdFp32, AverageIncludePadding4DNhwc)
{
    Tensor<float> dy({2, 3, 4, 4}, TensorLayout::NHWC);
    Tensor<float> dx({2, 3, 5, 5}, TensorLayout::NHWC);

    dy.fillWithValue(1.0f);

    CpuFpReferenceResampleBwd::backward<float, float, float>(dy,
                                                             dx,
                                                             {0, 0},
                                                             {1, 1},
                                                             {2, 2},
                                                             ResampleMode::AVGPOOL_INCLUDE_PADDING,
                                                             PaddingMode::ZERO_PAD);

    const auto* dxData = dx.memory().hostData();
    const std::array<float, 5> factors{1.0f, 2.0f, 2.0f, 2.0f, 1.0f};
    for(size_t n = 0; n < 2; ++n)
    {
        for(size_t h = 0; h < 5; ++h)
        {
            for(size_t w = 0; w < 5; ++w)
            {
                const float expected = factors[h] * factors[w] * 1.0f / 4.0f;
                for(size_t c = 0; c < 3; ++c)
                {
                    const size_t idx = ((n * 5 + h) * 5 + w) * 3 + c;
                    EXPECT_FLOAT_EQ(dxData[idx], expected)
                        << "Mismatch at n=" << n << " h=" << h << " w=" << w << " c=" << c;
                }
            }
        }
    }
}

TEST(TestCpuFpReferenceResampleBwdHalf, MaxPool4DNchw)
{
    Tensor<half> dy({2, 3, 4, 4});
    Tensor<half> dx({2, 3, 5, 5});
    Tensor<int32_t> index({2, 3, 4, 4});

    dy.fillWithValue(static_cast<half>(1.0f));
    index.fillWithValue(0);

    CpuFpReferenceResampleBwd::backward<half, half, float, int32_t>(
        dy, dx, {0, 0}, {1, 1}, {2, 2}, ResampleMode::MAXPOOL, PaddingMode::ZERO_PAD, &index);

    const auto* dxData = dx.memory().hostData();
    for(size_t n = 0; n < 2; ++n)
    {
        for(size_t c = 0; c < 3; ++c)
        {
            const auto base = (n * 3 + c) * 25;
            for(size_t i = 0; i < 25; ++i)
            {
                const float expected = (i == 0) ? 1.0f : 0.0f;
                EXPECT_NEAR(static_cast<float>(dxData[base + i]), expected, 1.0e-3f)
                    << "Mismatch at n=" << n << " c=" << c << " spatial index " << i;
            }
        }
    }
}

TEST(TestCpuFpReferenceResampleBwdBfloat16, AverageIncludePadding4DNchw)
{
    Tensor<bfloat16> dy({2, 3, 4, 4});
    Tensor<bfloat16> dx({2, 3, 5, 5});

    dy.fillWithValue(static_cast<bfloat16>(1.0f));

    CpuFpReferenceResampleBwd::backward<bfloat16, bfloat16, float>(
        dy,
        dx,
        {0, 0},
        {1, 1},
        {2, 2},
        ResampleMode::AVGPOOL_INCLUDE_PADDING,
        PaddingMode::ZERO_PAD);

    const auto* dxData = dx.memory().hostData();
    const std::array<float, 5> factors{1.0f, 2.0f, 2.0f, 2.0f, 1.0f};
    for(size_t n = 0; n < 2; ++n)
    {
        for(size_t c = 0; c < 3; ++c)
        {
            const auto base = (n * 3 + c) * 25;
            for(size_t h = 0; h < 5; ++h)
            {
                for(size_t w = 0; w < 5; ++w)
                {
                    const float expected = factors[h] * factors[w] * 1.0f / 4.0f;
                    EXPECT_NEAR(static_cast<float>(dxData[base + h * 5 + w]), expected, 1.0e-3f)
                        << "Mismatch at n=" << n << " c=" << c << " h=" << h << " w=" << w;
                }
            }
        }
    }
}

TEST(TestCpuFpReferenceResampleBwdDouble, AverageExcludePadding4DNchw)
{
    Tensor<double> dy({2, 3, 4, 4});
    Tensor<double> dx({2, 3, 5, 5});

    dy.fillWithValue(1.0);

    CpuFpReferenceResampleBwd::backward<double, double, double>(
        dy,
        dx,
        {0, 0},
        {1, 1},
        {2, 2},
        ResampleMode::AVGPOOL_EXCLUDE_PADDING,
        PaddingMode::ZERO_PAD);

    const auto* dxData = dx.memory().hostData();
    const std::array<double, 5> factors{1.0, 2.0, 2.0, 2.0, 1.0};
    for(size_t n = 0; n < 2; ++n)
    {
        for(size_t c = 0; c < 3; ++c)
        {
            const auto base = (n * 3 + c) * 25;
            for(size_t h = 0; h < 5; ++h)
            {
                for(size_t w = 0; w < 5; ++w)
                {
                    const double expected = factors[h] * factors[w] * 1.0 / 4.0;
                    EXPECT_DOUBLE_EQ(dxData[base + h * 5 + w], expected)
                        << "Mismatch at n=" << n << " c=" << c << " h=" << h << " w=" << w;
                }
            }
        }
    }
}

TEST(TestCpuFpReferenceResampleBwdFloatHalf, AverageExcludePadding4DNchw)
{
    Tensor<float> dy({2, 3, 4, 4});
    Tensor<half> dx({2, 3, 5, 5});

    dy.fillWithValue(1.0f);

    CpuFpReferenceResampleBwd::backward<float, half, float>(dy,
                                                            dx,
                                                            {0, 0},
                                                            {1, 1},
                                                            {2, 2},
                                                            ResampleMode::AVGPOOL_EXCLUDE_PADDING,
                                                            PaddingMode::ZERO_PAD);

    const auto* dxData = dx.memory().hostData();
    const std::array<float, 5> factors{1.0f, 2.0f, 2.0f, 2.0f, 1.0f};
    for(size_t n = 0; n < 2; ++n)
    {
        for(size_t c = 0; c < 3; ++c)
        {
            const size_t base = (n * 3 + c) * 25;
            for(size_t h = 0; h < 5; ++h)
            {
                for(size_t w = 0; w < 5; ++w)
                {
                    const float expected = factors[h] * factors[w] * 1.0f / 4.0f;
                    EXPECT_NEAR(static_cast<float>(dxData[base + h * 5 + w]), expected, 1.0e-3f)
                        << "Mismatch at n=" << n << " c=" << c << " h=" << h << " w=" << w;
                }
            }
        }
    }
}

TEST(TestCpuFpReferenceResampleBwdBfloat16Half, MaxPool4DNchw)
{
    Tensor<bfloat16> dy({2, 3, 4, 4});
    Tensor<half> dx({2, 3, 5, 5});
    Tensor<int32_t> index({2, 3, 4, 4});

    dy.fillWithValue(static_cast<bfloat16>(1.0f));
    index.fillWithValue(0);

    CpuFpReferenceResampleBwd::backward<bfloat16, half, float, int32_t>(
        dy, dx, {0, 0}, {1, 1}, {2, 2}, ResampleMode::MAXPOOL, PaddingMode::ZERO_PAD, &index);

    const auto* dxData = dx.memory().hostData();
    for(size_t n = 0; n < 2; ++n)
    {
        for(size_t c = 0; c < 3; ++c)
        {
            const auto base = (n * 3 + c) * 25;
            for(size_t i = 0; i < 25; ++i)
            {
                const float expected = (i == 0) ? 1.0f : 0.0f;
                EXPECT_NEAR(static_cast<float>(dxData[base + i]), expected, 1.0e-3f)
                    << "Mismatch at n=" << n << " c=" << c << " spatial index " << i;
            }
        }
    }
}
