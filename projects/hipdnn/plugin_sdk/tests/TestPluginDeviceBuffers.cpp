// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include <hipdnn_plugin_sdk/PluginDeviceBuffers.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>

using namespace hipdnn_plugin_sdk;

TEST(TestPluginDeviceBuffers, FindDeviceBufferReturnsCorrectBuffer)
{
    std::vector<hipdnnPluginDeviceBuffer_t> buffers
        = {{42, reinterpret_cast<void*>(0x1234)}, {99, reinterpret_cast<void*>(0x5678)}};

    auto result = findDeviceBuffer(99, buffers.data(), static_cast<uint32_t>(buffers.size()));
    EXPECT_EQ(result.uid, 99);
    EXPECT_EQ(result.ptr, reinterpret_cast<void*>(0x5678));
}

TEST(TestPluginDeviceBuffers, FindDeviceBufferThrowsIfNotFound)
{
    std::vector<hipdnnPluginDeviceBuffer_t> buffers = {{1, reinterpret_cast<void*>(0x1111)}};

    EXPECT_THROW(findDeviceBuffer(2, buffers.data(), static_cast<uint32_t>(buffers.size())),
                 HipdnnPluginException);
}

TEST(TestPluginDeviceBuffers, FindDeviceBufferThrowsWhenArrayIsEmpty)
{
    EXPECT_THROW(findDeviceBuffer(1, nullptr, 0), HipdnnPluginException);
}

TEST(TestPluginDeviceBuffers, FindDeviceBufferReturnsFirstMatchWhenDuplicateUidsExist)
{
    int data1 = 0;
    int data2 = 0;
    std::vector<hipdnnPluginDeviceBuffer_t> buffers = {{5, &data1}, {5, &data2}};

    auto result = findDeviceBuffer(5, buffers.data(), static_cast<uint32_t>(buffers.size()));
    EXPECT_EQ(result.uid, 5);
    EXPECT_EQ(result.ptr, &data1);
}
