// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>
#include <hipdnn-gpu-ref/GpuFpReferenceBlockScale.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceBlockScaleDequantize.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceBlockScaleQuantize.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

using hipdnn_data_sdk::types::fp4_e2m1;
using hipdnn_data_sdk::types::fp6_e2m3;
using hipdnn_data_sdk::types::fp6_e3m2;
using hipdnn_data_sdk::types::fp8_e4m3;
using hipdnn_data_sdk::types::fp8_e5m2;
using hipdnn_data_sdk::types::fp8_e8m0;
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

template <typename QuantizedType>
static void runCompactE8M0RoundTrip()
{
    Tensor<float> cpuX({1, 8});
    Tensor<QuantizedType> cpuQuantized({1, 8});
    Tensor<fp8_e8m0> cpuScale({1, 2});
    Tensor<float> cpuDequantized({1, 8});
    Tensor<float> gpuX({1, 8});
    Tensor<QuantizedType> gpuQuantized({1, 8});
    Tensor<fp8_e8m0> gpuScale({1, 2});
    Tensor<float> gpuDequantized({1, 8});

    const std::vector<float> values{-8.0f, -3.0f, -1.0f, 0.0f, 0.5f, 2.0f, 5.0f, 9.0f};
    cpuX.fillWithData(values.data(), values.size() * sizeof(float));
    gpuX.fillWithData(values.data(), values.size() * sizeof(float));
    CpuFpReferenceBlockScaleQuantize::quantize(cpuX, cpuQuantized, cpuScale, 4, 1);
    gpuX.memory().hostData();
    GpuFpReferenceBlockScale::quantize(gpuX, gpuQuantized, gpuScale, 4, 1);
    gpuQuantized.memory().hostData();
    gpuScale.memory().hostData();

    for(int64_t column = 0; column < 8; ++column)
    {
        EXPECT_EQ(gpuQuantized.getHostValue(0, column).data,
                  cpuQuantized.getHostValue(0, column).data);
    }
    for(int64_t block = 0; block < 2; ++block)
    {
        EXPECT_EQ(gpuScale.getHostValue(0, block).data, cpuScale.getHostValue(0, block).data);
    }

    CpuFpReferenceBlockScaleDequantize::dequantize(
        cpuQuantized, cpuScale, cpuDequantized, {4}, false);
    GpuFpReferenceBlockScale::dequantize(gpuQuantized, gpuScale, gpuDequantized, {4}, false);
    gpuDequantized.memory().hostData();
    for(int64_t column = 0; column < 8; ++column)
    {
        EXPECT_FLOAT_EQ(gpuDequantized.getHostValue(0, column),
                        cpuDequantized.getHostValue(0, column));
    }
}

TEST(TestGpuFpReferenceBlockScale, CompactE8M0RoundTripsMatchCpu)
{
    SKIP_IF_NO_DEVICES();

    runCompactE8M0RoundTrip<fp4_e2m1>();
    runCompactE8M0RoundTrip<fp6_e2m3>();
    runCompactE8M0RoundTrip<fp6_e3m2>();
    runCompactE8M0RoundTrip<fp8_e4m3>();
    runCompactE8M0RoundTrip<fp8_e5m2>();
}
