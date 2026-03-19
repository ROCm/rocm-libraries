// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <array>

#include <gtest/gtest.h>

#include "HipKernelUtils.hpp"

#include <hipdnn_data_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>

using namespace hip_kernel_provider::hip_kernel_utils;

// ============================================================================
// findDeviceBuffer
// ============================================================================

TEST(TestFindDeviceBuffer, FindsBufferWithMatchingUid)
{
    int data1 = 0;
    int data2 = 0;
    std::array<hipdnnPluginDeviceBuffer_t, 2> buffers = {{{1, &data1}, {2, &data2}}};

    auto result = findDeviceBuffer(2, buffers.data(), 2);

    EXPECT_EQ(result.uid, 2);
    EXPECT_EQ(result.ptr, &data2);
}

TEST(TestFindDeviceBuffer, FindsFirstBufferInArray)
{
    int data = 0;
    std::array<hipdnnPluginDeviceBuffer_t, 1> buffers = {{{42, &data}}};

    auto result = findDeviceBuffer(42, buffers.data(), 1);

    EXPECT_EQ(result.uid, 42);
    EXPECT_EQ(result.ptr, &data);
}

TEST(TestFindDeviceBuffer, ThrowsWhenUidNotFound)
{
    int data = 0;
    std::array<hipdnnPluginDeviceBuffer_t, 1> buffers = {{{1, &data}}};

    EXPECT_THROW(findDeviceBuffer(99, buffers.data(), 1), hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestFindDeviceBuffer, ThrowsWhenBufferArrayIsEmpty)
{
    EXPECT_THROW(findDeviceBuffer(1, nullptr, 0), hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestFindDeviceBuffer, FindsFirstMatchWhenDuplicateUidsExist)
{
    int data1 = 0;
    int data2 = 0;
    std::array<hipdnnPluginDeviceBuffer_t, 2> buffers = {{{5, &data1}, {5, &data2}}};

    auto result = findDeviceBuffer(5, buffers.data(), 2);

    EXPECT_EQ(result.uid, 5);
    EXPECT_EQ(result.ptr, &data1);
}

// ============================================================================
// findTensorAttributes
// ============================================================================

TEST(TestFindTensorAttributes, FindsTensorWithMatchingUid)
{
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormInferenceGraph();
    hipdnn_data_sdk::flatbuffer_utilities::GraphWrapper graph(builder.GetBufferPointer(),
                                                              builder.GetSize());

    const auto& tensorMap = graph.getTensorMap();

    // UID 1 is the x tensor in the test graph
    const auto& attrs = findTensorAttributes(tensorMap, 1);
    EXPECT_EQ(attrs.uid(), 1);
}

TEST(TestFindTensorAttributes, ThrowsWhenUidNotFound)
{
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormInferenceGraph();
    hipdnn_data_sdk::flatbuffer_utilities::GraphWrapper graph(builder.GetBufferPointer(),
                                                              builder.GetSize());

    const auto& tensorMap = graph.getTensorMap();

    EXPECT_THROW(findTensorAttributes(tensorMap, 9999), hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestFindTensorAttributes, ThrowsWhenMapIsEmpty)
{
    std::unordered_map<int64_t, const hipdnn_data_sdk::data_objects::TensorAttributes*> emptyMap;

    EXPECT_THROW(findTensorAttributes(emptyMap, 1), hipdnn_plugin_sdk::HipdnnPluginException);
}
