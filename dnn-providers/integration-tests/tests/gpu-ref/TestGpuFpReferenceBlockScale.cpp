// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>
#include <hipdnn-gpu-ref/GpuFpReferenceBlockScale.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceBlockScaleDequantize.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceBlockScaleQuantize.hpp>
#include <hipdnn_test_sdk/utilities/DeviceQuery.hpp>
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
            EXPECT_FLOAT_EQ(gpuScale.getHostValue(row, block), cpuScale.getHostValue(row, block));
        }
    }
}

TEST(TestGpuFpReferenceBlockScale, DequantizeMatchesCpu)
{
    Tensor<float> cpuX({1, 4, 6});
    Tensor<float> cpuScale({1, 2, 3});
    Tensor<float> cpuY({1, 4, 6});
    Tensor<float> gpuX({1, 4, 6});
    Tensor<float> gpuScale({1, 2, 3});
    Tensor<float> gpuY({1, 4, 6});

    cpuX.fillWithValue(1.5f);
    gpuX.fillWithValue(1.5f);
    const std::vector<float> scales{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    cpuScale.fillWithData(scales.data(), scales.size() * sizeof(float));
    gpuScale.fillWithData(scales.data(), scales.size() * sizeof(float));
    CpuFpReferenceBlockScaleDequantize::dequantize(cpuX, cpuScale, cpuY, {2, 2}, false);
    gpuX.memory().hostData();
    gpuScale.memory().hostData();
    GpuFpReferenceBlockScale::dequantize(gpuX, gpuScale, gpuY, {2, 2}, false);
    gpuY.memory().hostData();

    for(int64_t row = 0; row < 4; ++row)
    {
        for(int64_t column = 0; column < 6; ++column)
        {
            EXPECT_FLOAT_EQ(gpuY.getHostValue(0, row, column), cpuY.getHostValue(0, row, column));
        }
    }
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
    const auto arch = hipdnn_test_sdk::utilities::currentDeviceArch();
    if(arch.rfind("gfx950", 0) != 0 && arch.rfind("gfx1250", 0) != 0)
    {
        GTEST_SKIP() << "Native HIP compact types are not supported on " << arch;
    }

    runCompactE8M0RoundTrip<fp4_e2m1>();
    runCompactE8M0RoundTrip<fp6_e2m3>();
    runCompactE8M0RoundTrip<fp6_e3m2>();
    runCompactE8M0RoundTrip<fp8_e4m3>();
    runCompactE8M0RoundTrip<fp8_e5m2>();
}
