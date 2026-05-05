// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <cstdint>
#include <hip/hip_runtime.h>
#include <hipdnn_data_sdk/utilities/Workspace.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>
#include <unordered_map>
#include <vector>

#include "ConvolutionFwdGraphTestUtils.hpp"
#include "common/ConvolutionCommon.hpp"
#include "harness/ReferenceGraphExecutorFactory.hpp"

using hipdnn_integration_tests::ReferenceExecutorType;
using hipdnn_integration_tests::ReferenceGraphExecutorFactory;
using namespace hipdnn_flatbuffers_sdk::data_objects;
using namespace hipdnn_integration_tests::test_utils;

// NOLINTBEGIN(readability-identifier-naming) -- gtest macro-generated names

TEST(TestReferenceGraphExecutorFactory, CreateCpuExecutor)
{
    auto executor = ReferenceGraphExecutorFactory::create(ReferenceExecutorType::CPU);
    ASSERT_NE(executor, nullptr);
    EXPECT_FALSE(executor->requiresDeviceMemory());
}

TEST(TestReferenceGraphExecutorFactory, CreateDeviceExecutor)
{
    auto executor = ReferenceGraphExecutorFactory::create(ReferenceExecutorType::GPU);
    ASSERT_NE(executor, nullptr);
    EXPECT_TRUE(executor->requiresDeviceMemory());
}

// Exercises the factory's core value: both CPU and GPU executors produce
// consistent results for a conv forward graph using a canonical test shape.
TEST(TestReferenceGraphExecutorFactory, GpuAndCpuExecutorsAgreeOnConvFwd)
{
    SKIP_IF_NO_DEVICES();

    // Use case 1 from the shared catalog: 3×3 filter, no padding
    // x={1,16,16,16}, w={1,16,3,3}, y={1,1,14,14}
    auto cases = test_conv_common::getConvTestCases4D();
    const auto& tc = cases[1];

    constexpr int64_t X_UID = 1;
    constexpr int64_t W_UID = 2;
    constexpr int64_t Y_UID = 3;

    auto xStrides = generateStrides(tc.xDims);
    auto wStrides = generateStrides(tc.wDims);
    auto yStrides = generateStrides(tc.yDims);

    auto graphBuilder = createConvFwdGraph(X_UID,
                                           W_UID,
                                           Y_UID,
                                           tc.xDims,
                                           tc.wDims,
                                           tc.yDims,
                                           xStrides,
                                           wStrides,
                                           yStrides,
                                           tc.convPrePadding,
                                           tc.convStride,
                                           tc.convDilation,
                                           DataType::FLOAT);

    auto elemCount = [](const std::vector<int64_t>& dims) {
        size_t count = 1;
        for(auto d : dims)
        {
            count *= static_cast<size_t>(d);
        }
        return count;
    };

    auto xCount = elemCount(tc.xDims);
    auto wCount = elemCount(tc.wDims);
    auto yCount = elemCount(tc.yDims);

    // Fill input data
    std::vector<float> xData(xCount);
    std::vector<float> wData(wCount);
    for(size_t i = 0; i < xCount; ++i)
    {
        xData[i] = static_cast<float>(i + 1);
    }
    for(size_t i = 0; i < wCount; ++i)
    {
        wData[i] = 1.0f;
    }

    // --- Run CPU executor (host pointers) ---
    auto cpuExecutor = ReferenceGraphExecutorFactory::create(ReferenceExecutorType::CPU);
    ASSERT_FALSE(cpuExecutor->requiresDeviceMemory());

    std::vector<float> cpuY(yCount, 0.0f);
    std::unordered_map<int64_t, void*> cpuPack;
    cpuPack[X_UID] = xData.data();
    cpuPack[W_UID] = wData.data();
    cpuPack[Y_UID] = cpuY.data();

    cpuExecutor->execute(graphBuilder.GetBufferPointer(), graphBuilder.GetSize(), cpuPack);

    // --- Run GPU executor (device pointers) ---
    auto gpuExecutor = ReferenceGraphExecutorFactory::create(ReferenceExecutorType::GPU);
    ASSERT_TRUE(gpuExecutor->requiresDeviceMemory());

    const hipdnn_data_sdk::utilities::Workspace dX(xCount * sizeof(float));
    const hipdnn_data_sdk::utilities::Workspace dW(wCount * sizeof(float));
    const hipdnn_data_sdk::utilities::Workspace dY(yCount * sizeof(float));

    ASSERT_EQ(hipMemcpy(dX.get(), xData.data(), xCount * sizeof(float), hipMemcpyHostToDevice),
              hipSuccess);
    ASSERT_EQ(hipMemcpy(dW.get(), wData.data(), wCount * sizeof(float), hipMemcpyHostToDevice),
              hipSuccess);
    ASSERT_EQ(hipMemset(dY.get(), 0, yCount * sizeof(float)), hipSuccess);

    std::unordered_map<int64_t, void*> gpuPack;
    gpuPack[X_UID] = dX.get();
    gpuPack[W_UID] = dW.get();
    gpuPack[Y_UID] = dY.get();

    gpuExecutor->execute(graphBuilder.GetBufferPointer(), graphBuilder.GetSize(), gpuPack);

    std::vector<float> gpuY(yCount);
    ASSERT_EQ(hipMemcpy(gpuY.data(), dY.get(), yCount * sizeof(float), hipMemcpyDeviceToHost),
              hipSuccess);

    // --- Compare: both executors should produce the same result ---
    for(size_t i = 0; i < yCount; ++i)
    {
        EXPECT_NEAR(gpuY[i], cpuY[i], 1e-5f) << "Mismatch at index " << i;
    }
}

// NOLINTEND(readability-identifier-naming)
