// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>
#include <hipdnn-gpu-ref/GpuFpReferenceBlockScale.hpp>
#include <hipdnn-gpu-ref/ShallowGpuTensor.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceBlockScaleDequantize.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceBlockScaleQuantize.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

using hipdnn_data_sdk::types::bfloat16;
using hipdnn_data_sdk::types::fp4_e2m1;
using hipdnn_data_sdk::types::fp6_e2m3;
using hipdnn_data_sdk::types::fp6_e3m2;
using hipdnn_data_sdk::types::fp8_e4m3;
using hipdnn_data_sdk::types::fp8_e5m2;
using hipdnn_data_sdk::types::fp8_e8m0;
using hipdnn_data_sdk::types::half;
using hipdnn_data_sdk::utilities::Tensor;
using hipdnn_gpu_ref::GpuFpReferenceBlockScale;
using hipdnn_test_sdk::utilities::CpuFpReferenceBlockScaleDequantize;
using hipdnn_test_sdk::utilities::CpuFpReferenceBlockScaleQuantize;

TEST(TestGpuFpReferenceBlockScale, QuantizeMatchesCpu)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> cpuX({2, 5});
    Tensor<int8_t> cpuY({2, 5});
    Tensor<float> cpuScale({2, 3});
    Tensor<float> gpuX({2, 5});
    Tensor<int8_t> gpuY({2, 5});
    Tensor<float> gpuScale({2, 3});

    const std::vector<float> values{-3.0f, -1.0f, 2.0f, 8.0f, 0.0f, 1.0f, 7.0f, -4.0f, 2.5f, -0.5f};
    cpuX.fillWithData(values.data(), values.size() * sizeof(float));
    gpuX.fillWithData(values.data(), values.size() * sizeof(float));
    CpuFpReferenceBlockScaleQuantize::quantize(cpuX, cpuY, cpuScale, 2, 1);
    gpuX.memory().hostData();
    GpuFpReferenceBlockScale::quantize(gpuX, gpuY, gpuScale, 2, 1);
    gpuY.memory().hostData();
    gpuScale.memory().hostData();

    for(int64_t row = 0; row < 2; ++row)
    {
        for(int64_t column = 0; column < 5; ++column)
        {
            EXPECT_EQ(gpuY.getHostValue(row, column), cpuY.getHostValue(row, column));
        }
        for(int64_t block = 0; block < 3; ++block)
        {
            EXPECT_EQ(gpuScale.getHostValue(row, block), cpuScale.getHostValue(row, block));
        }
    }

    const std::vector<float> expectedScales{
        0x1.83060ep-6f, 0x1.02040ap-4f, 0.0f, 0x1.c38710p-5f, 0x1.02040ap-5f, 0x1.02040ap-8f};
    for(size_t i = 0; i < expectedScales.size(); ++i)
    {
        EXPECT_EQ(gpuScale.getHostValue(static_cast<int64_t>(i / 3), static_cast<int64_t>(i % 3)),
                  expectedScales[i]);
    }
}

TEST(TestGpuFpReferenceBlockScale, DequantizeMatchesCpu)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> cpuX({1, 5, 7});
    Tensor<float> cpuScale({1, 3, 4});
    Tensor<float> cpuY({1, 5, 7});
    Tensor<float> gpuX({1, 5, 7});
    Tensor<float> gpuScale({1, 3, 4});
    Tensor<float> gpuY({1, 5, 7});

    std::vector<float> values(35);
    for(size_t i = 0; i < values.size(); ++i)
    {
        values[i] = static_cast<float>(i) - 11.5f;
    }
    cpuX.fillWithData(values.data(), values.size() * sizeof(float));
    gpuX.fillWithData(values.data(), values.size() * sizeof(float));
    const std::vector<float> scales{
        1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};
    cpuScale.fillWithData(scales.data(), scales.size() * sizeof(float));
    gpuScale.fillWithData(scales.data(), scales.size() * sizeof(float));
    CpuFpReferenceBlockScaleDequantize::dequantize(cpuX, cpuScale, cpuY, {2, 2}, false);
    gpuX.memory().hostData();
    gpuScale.memory().hostData();
    GpuFpReferenceBlockScale::dequantize(gpuX, gpuScale, gpuY, {2, 2}, false);
    gpuY.memory().hostData();

    for(int64_t row = 0; row < 5; ++row)
    {
        for(int64_t column = 0; column < 7; ++column)
        {
            EXPECT_FLOAT_EQ(gpuY.getHostValue(0, row, column), cpuY.getHostValue(0, row, column));
        }
    }
}

TEST(TestGpuFpReferenceBlockScale, QuantizeBlockSizeOneHasExpectedValues)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> x({2});
    Tensor<int8_t> y({2});
    Tensor<float> scale({2});
    const std::vector<float> values{127.0f, -254.0f};
    x.fillWithData(values.data(), values.size() * sizeof(float));

    GpuFpReferenceBlockScale::quantize(x, y, scale, 1, 0);
    y.memory().hostData();
    scale.memory().hostData();

    EXPECT_EQ(y.getHostValue(0), 127);
    EXPECT_EQ(y.getHostValue(1), -127);
    EXPECT_EQ(scale.getHostValue(0), 1.0f);
    EXPECT_EQ(scale.getHostValue(1), 2.0f);
}

TEST(TestGpuFpReferenceBlockScale, QuantizeRejectsNonFiniteInput)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> x({2});
    Tensor<int8_t> y({2});
    Tensor<float> scale({1});
    const std::vector<float> values{1.0f, std::numeric_limits<float>::quiet_NaN()};
    x.fillWithData(values.data(), values.size() * sizeof(float));

    EXPECT_THROW(GpuFpReferenceBlockScale::quantize(x, y, scale, 2, 0), std::runtime_error);
}

TEST(TestGpuFpReferenceBlockScale, QuantizeNonLastAxisWithPermutedStridesMatchesCpu)
{
    SKIP_IF_NO_DEVICES();
    const std::vector<int64_t> dims{2, 3, 4};
    const std::vector<int64_t> scaleDims{2, 2, 4};
    Tensor<float> cpuX(dims, {12, 1, 3});
    Tensor<int8_t> cpuY(dims, {1, 8, 2});
    Tensor<float> cpuScale(scaleDims, {8, 1, 2});
    Tensor<float> gpuX(dims, {12, 1, 3});
    Tensor<int8_t> gpuY(dims, {1, 8, 2});
    Tensor<float> gpuScale(scaleDims, {8, 1, 2});

    for(int64_t batch = 0; batch < dims[0]; ++batch)
    {
        for(int64_t channel = 0; channel < dims[1]; ++channel)
        {
            for(int64_t column = 0; column < dims[2]; ++column)
            {
                const auto value = static_cast<float>(batch * 20 + channel * 5 + column - 12);
                cpuX.setHostValue(value, batch, channel, column);
                gpuX.setHostValue(value, batch, channel, column);
            }
        }
    }

    CpuFpReferenceBlockScaleQuantize::quantize(cpuX, cpuY, cpuScale, 2, 1);
    GpuFpReferenceBlockScale::quantize(gpuX, gpuY, gpuScale, 2, 1);
    gpuY.memory().hostData();
    gpuScale.memory().hostData();

    for(int64_t batch = 0; batch < dims[0]; ++batch)
    {
        for(int64_t channel = 0; channel < dims[1]; ++channel)
        {
            for(int64_t column = 0; column < dims[2]; ++column)
            {
                EXPECT_EQ(gpuY.getHostValue(batch, channel, column),
                          cpuY.getHostValue(batch, channel, column));
            }
        }
        for(int64_t block = 0; block < scaleDims[1]; ++block)
        {
            for(int64_t column = 0; column < scaleDims[2]; ++column)
            {
                EXPECT_EQ(gpuScale.getHostValue(batch, block, column),
                          cpuScale.getHostValue(batch, block, column));
            }
        }
    }
}

TEST(TestGpuFpReferenceBlockScale, DequantizeNegativeScaleHasExpectedValue)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> cpuX({1});
    Tensor<float> cpuScale({1});
    Tensor<float> cpuY({1});
    Tensor<float> gpuX({1});
    Tensor<float> gpuScale({1});
    Tensor<float> gpuY({1});
    cpuX.setHostValue(3.0f, 0);
    cpuScale.setHostValue(2.0f, 0);
    gpuX.setHostValue(3.0f, 0);
    gpuScale.setHostValue(2.0f, 0);

    CpuFpReferenceBlockScaleDequantize::dequantize(cpuX, cpuScale, cpuY, {1}, true);
    GpuFpReferenceBlockScale::dequantize(gpuX, gpuScale, gpuY, {1}, true);
    gpuY.memory().hostData();

    EXPECT_EQ(gpuY.getHostValue(0), 0.75f);
    EXPECT_EQ(gpuY.getHostValue(0), cpuY.getHostValue(0));
}

template <typename QuantizedType, typename RealType = float, typename ScaleType = fp8_e8m0>
static void runBlockScaleRoundTrip()
{
    Tensor<RealType> cpuX({1, 8});
    Tensor<QuantizedType> cpuQuantized({1, 8});
    Tensor<ScaleType> cpuScale({1, 2});
    Tensor<RealType> cpuDequantized({1, 8});
    Tensor<RealType> gpuX({1, 8});
    Tensor<QuantizedType> gpuQuantized({1, 8});
    Tensor<ScaleType> gpuScale({1, 2});
    Tensor<RealType> gpuDequantized({1, 8});

    const std::vector<float> values{-8.0f, -3.0f, -1.0f, 0.0f, 0.5f, 2.0f, 5.0f, 9.0f};
    for(int64_t column = 0; column < 8; ++column)
    {
        const auto value = static_cast<RealType>(values[static_cast<size_t>(column)]);
        cpuX.setHostValue(value, 0, column);
        gpuX.setHostValue(value, 0, column);
    }
    CpuFpReferenceBlockScaleQuantize::quantize(cpuX, cpuQuantized, cpuScale, 4, 1);
    gpuX.memory().hostData();
    GpuFpReferenceBlockScale::quantize(gpuX, gpuQuantized, gpuScale, 4, 1);
    gpuQuantized.memory().hostData();
    gpuScale.memory().hostData();

    for(int64_t column = 0; column < 8; ++column)
    {
        EXPECT_EQ(static_cast<float>(gpuQuantized.getHostValue(0, column)),
                  static_cast<float>(cpuQuantized.getHostValue(0, column)));
    }
    for(int64_t block = 0; block < 2; ++block)
    {
        EXPECT_EQ(static_cast<float>(gpuScale.getHostValue(0, block)),
                  static_cast<float>(cpuScale.getHostValue(0, block)));
    }

    CpuFpReferenceBlockScaleDequantize::dequantize(
        cpuQuantized, cpuScale, cpuDequantized, {4}, false);
    GpuFpReferenceBlockScale::dequantize(gpuQuantized, gpuScale, gpuDequantized, {4}, false);
    gpuDequantized.memory().hostData();
    for(int64_t column = 0; column < 8; ++column)
    {
        EXPECT_FLOAT_EQ(static_cast<float>(gpuDequantized.getHostValue(0, column)),
                        static_cast<float>(cpuDequantized.getHostValue(0, column)));
    }
}

TEST(TestGpuFpReferenceBlockScale, CompactE8M0RoundTripsMatchCpu)
{
    SKIP_IF_NO_DEVICES();

    runBlockScaleRoundTrip<fp4_e2m1>();
    runBlockScaleRoundTrip<fp6_e2m3>();
    runBlockScaleRoundTrip<fp6_e3m2>();
    runBlockScaleRoundTrip<fp8_e4m3>();
    runBlockScaleRoundTrip<fp8_e5m2>();
}

TEST(TestGpuFpReferenceBlockScaleFp16, RoundTripMatchesCpu)
{
    SKIP_IF_NO_DEVICES();
    runBlockScaleRoundTrip<half, half, float>();
}

TEST(TestGpuFpReferenceBlockScaleBfp16, RoundTripMatchesCpu)
{
    SKIP_IF_NO_DEVICES();
    runBlockScaleRoundTrip<bfloat16, bfloat16, float>();
}

TEST(TestGpuFpReferenceBlockScaleFp16, CompactE8M0RoundTripMatchesCpu)
{
    SKIP_IF_NO_DEVICES();
    runBlockScaleRoundTrip<fp4_e2m1, half>();
}

TEST(TestGpuFpReferenceBlockScaleBfp16, CompactE8M0RoundTripMatchesCpu)
{
    SKIP_IF_NO_DEVICES();
    runBlockScaleRoundTrip<fp4_e2m1, bfloat16>();
}

TEST(TestGpuFpReferenceBlockScale, DequantizeWithPermutedOutputStridesMatchesCpu)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> x({2, 3, 5}, {15, 5, 1});
    Tensor<float> scale({2, 2, 3}, {1, 6, 2});
    Tensor<float> cpuY({2, 3, 5}, {1, 10, 2});
    Tensor<float> gpuY({2, 3, 5}, {1, 10, 2});
    for(int64_t batch = 0; batch < 2; ++batch)
    {
        for(int64_t row = 0; row < 3; ++row)
        {
            for(int64_t column = 0; column < 5; ++column)
            {
                x.setHostValue(
                    static_cast<float>(batch * 15 + row * 5 + column - 12), batch, row, column);
            }
        }
        for(int64_t row = 0; row < 2; ++row)
        {
            for(int64_t column = 0; column < 3; ++column)
            {
                scale.setHostValue(
                    static_cast<float>(batch * 6 + row * 3 + column + 1), batch, row, column);
            }
        }
    }
    CpuFpReferenceBlockScaleDequantize::dequantize(x, scale, cpuY, {2, 2});
    GpuFpReferenceBlockScale::dequantize(x, scale, gpuY, {2, 2});
    gpuY.memory().hostData();
    for(int64_t batch = 0; batch < 2; ++batch)
    {
        for(int64_t row = 0; row < 3; ++row)
        {
            for(int64_t column = 0; column < 5; ++column)
            {
                EXPECT_FLOAT_EQ(gpuY.getHostValue(batch, row, column),
                                cpuY.getHostValue(batch, row, column));
            }
        }
    }
}

TEST(TestGpuFpReferenceBlockScale, InvalidQuantizeArgumentsThrow)
{
    using hipdnn_gpu_ref::ShallowGpuTensor;
    ShallowGpuTensor<float> x(nullptr, {2, 3, 5}, {15, 5, 1});
    ShallowGpuTensor<float> y(nullptr, {2, 3, 5}, {15, 5, 1});
    ShallowGpuTensor<float> scale(nullptr, {2, 3, 3}, {9, 3, 1});
    for(const auto blockSize : {0, -1})
    {
        EXPECT_THROW(GpuFpReferenceBlockScale::quantize(x, y, scale, blockSize),
                     std::invalid_argument);
    }
    for(const int64_t axis : {-1, 3})
    {
        EXPECT_THROW(GpuFpReferenceBlockScale::quantize(x, y, scale, 2, axis),
                     std::invalid_argument);
    }
    ShallowGpuTensor<float> wrongDims(nullptr, {2, 3, 4}, {12, 4, 1});
    EXPECT_THROW(GpuFpReferenceBlockScale::quantize(x, wrongDims, scale, 2), std::invalid_argument);
    ShallowGpuTensor<float> wrongRank(nullptr, {2, 3}, {3, 1});
    EXPECT_THROW(GpuFpReferenceBlockScale::quantize(x, y, wrongRank, 2), std::invalid_argument);
    ShallowGpuTensor<float> wrongScale(nullptr, {2, 3, 2}, {6, 2, 1});
    EXPECT_THROW(GpuFpReferenceBlockScale::quantize(x, y, wrongScale, 2), std::invalid_argument);
    ShallowGpuTensor<float> wrongStrides(nullptr, {2, 3, 5}, {15, 5, 1, 1});
    EXPECT_THROW(GpuFpReferenceBlockScale::quantize(wrongStrides, y, scale, 2),
                 std::invalid_argument);
    EXPECT_THROW(GpuFpReferenceBlockScale::quantize(x, wrongStrides, scale, 2),
                 std::invalid_argument);
    EXPECT_THROW(GpuFpReferenceBlockScale::quantize(x, y, wrongStrides, 2), std::invalid_argument);
}

TEST(TestGpuFpReferenceBlockScale, InvalidDequantizeArgumentsThrow)
{
    using hipdnn_gpu_ref::ShallowGpuTensor;
    ShallowGpuTensor<float> x(nullptr, {2, 3, 5}, {15, 5, 1});
    ShallowGpuTensor<float> y(nullptr, {2, 3, 5}, {15, 5, 1});
    ShallowGpuTensor<float> scale(nullptr, {2, 2, 3}, {6, 3, 1});
    for(const auto& blockSize : std::vector<std::vector<int32_t>>{{0, 2}, {-1, 2}, {1, 1, 2, 2}})
    {
        EXPECT_THROW(GpuFpReferenceBlockScale::dequantize(x, scale, y, blockSize),
                     std::invalid_argument);
    }
    ShallowGpuTensor<float> wrongDims(nullptr, {2, 3, 4}, {12, 4, 1});
    EXPECT_THROW(GpuFpReferenceBlockScale::dequantize(x, scale, wrongDims, {2, 2}),
                 std::invalid_argument);
    ShallowGpuTensor<float> wrongRank(nullptr, {2, 3}, {3, 1});
    EXPECT_THROW(GpuFpReferenceBlockScale::dequantize(x, wrongRank, y, {2, 2}),
                 std::invalid_argument);
    ShallowGpuTensor<float> wrongScale(nullptr, {2, 2, 2}, {4, 2, 1});
    EXPECT_THROW(GpuFpReferenceBlockScale::dequantize(x, wrongScale, y, {2, 2}),
                 std::invalid_argument);
    ShallowGpuTensor<float> wrongStrides(nullptr, {2, 3, 5}, {15, 5, 1, 1});
    EXPECT_THROW(GpuFpReferenceBlockScale::dequantize(wrongStrides, scale, y, {2, 2}),
                 std::invalid_argument);
    EXPECT_THROW(GpuFpReferenceBlockScale::dequantize(x, wrongStrides, y, {2, 2}),
                 std::invalid_argument);
    EXPECT_THROW(GpuFpReferenceBlockScale::dequantize(x, scale, wrongStrides, {2, 2}),
                 std::invalid_argument);
}
