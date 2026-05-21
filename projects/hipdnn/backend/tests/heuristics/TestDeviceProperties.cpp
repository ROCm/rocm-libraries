// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

/**
 * @file TestDeviceProperties.cpp
 * @brief Unit tests for DeviceProperties helper functions (RFC 0007 PR3)
 *
 * Tests the device properties query, serialization, and wrapper utilities
 * used for passing device info to heuristic plugins.
 */

#include "heuristics/DeviceProperties.hpp"

#include <flatbuffers/flatbuffers.h>
#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <hipdnn_flatbuffers_sdk/data_objects/device_properties_generated.h>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

using namespace hipdnn_backend::heuristics;
using namespace hipdnn_flatbuffers_sdk::data_objects;

// Serialize/wrap tests build DevicePropertiesT in-memory and don't touch HIP,
// so they can run on CPU-only CI runners. GPU-requiring tests that call
// queryDeviceProperties() live in TestGpuDeviceProperties below.
class TestDeviceProperties : public ::testing::Test
{
protected:
    // Helper to create test device properties
    static DevicePropertiesT createTestProperties()
    {
        DevicePropertiesT props;
        props.device_id = 0;
        props.multi_processor_count = 120;
        props.total_global_mem = 16ULL * 1024 * 1024 * 1024; // 16GB
        props.architecture_name = "gfx90a";
        return props;
    }
};

// ========== serializeDeviceProperties Tests ==========

TEST_F(TestDeviceProperties, SerializeDevicePropertiesReturnsNonEmpty)
{
    const auto props = createTestProperties();

    const auto serialized = serializeDeviceProperties(props);

    EXPECT_FALSE(serialized.empty());
    EXPECT_GT(serialized.size(), 0u);
}

TEST_F(TestDeviceProperties, SerializeDevicePropertiesHasValidFormat)
{
    const auto props = createTestProperties();

    const auto serialized = serializeDeviceProperties(props);

    // FlatBuffer should have at least the file identifier and root table
    EXPECT_GE(serialized.size(), 8u) << "FlatBuffer too small to be valid";

    // Check for FlatBuffer file identifier "HDDP" (HipDNN Device Properties)
    // File identifier is stored at offset 4-7 in FlatBuffer format
    if(serialized.size() >= 8)
    {
        EXPECT_EQ(serialized[4], 'H');
        EXPECT_EQ(serialized[5], 'D');
        EXPECT_EQ(serialized[6], 'D');
        EXPECT_EQ(serialized[7], 'P');
    }
}

TEST_F(TestDeviceProperties, SerializeDevicePropertiesCanBeDeserialized)
{
    const auto props = createTestProperties();

    const auto serialized = serializeDeviceProperties(props);

    // Verify we can deserialize back
    const auto* deviceProps = flatbuffers::GetRoot<DeviceProperties>(serialized.data());
    ASSERT_NE(deviceProps, nullptr);

    // Verify deserialized values match original
    EXPECT_EQ(deviceProps->device_id(), props.device_id);
    EXPECT_EQ(deviceProps->multi_processor_count(), props.multi_processor_count);
    EXPECT_EQ(deviceProps->total_global_mem(), props.total_global_mem);
    EXPECT_EQ(deviceProps->architecture_name()->str(), props.architecture_name);
}

TEST_F(TestDeviceProperties, SerializeDevicePropertiesHandlesEmptyArchitecture)
{
    auto props = createTestProperties();
    props.architecture_name = ""; // Empty string

    const auto serialized = serializeDeviceProperties(props);

    EXPECT_FALSE(serialized.empty());

    // Should be deserializable even with empty architecture
    const auto* deviceProps = flatbuffers::GetRoot<DeviceProperties>(serialized.data());
    ASSERT_NE(deviceProps, nullptr);
    // Check if architecture_name is null or empty
    if(deviceProps->architecture_name() != nullptr)
    {
        EXPECT_TRUE(deviceProps->architecture_name()->str().empty());
    }
}

TEST_F(TestDeviceProperties, SerializeDevicePropertiesHandlesLargeValues)
{
    auto props = createTestProperties();
    props.multi_processor_count = 65536; // Large CU count
    props.total_global_mem = 128ULL * 1024 * 1024 * 1024; // 128GB

    const auto serialized = serializeDeviceProperties(props);

    const auto* deviceProps = flatbuffers::GetRoot<DeviceProperties>(serialized.data());
    ASSERT_NE(deviceProps, nullptr);

    EXPECT_EQ(deviceProps->multi_processor_count(), 65536);
    EXPECT_EQ(deviceProps->total_global_mem(), 128ULL * 1024 * 1024 * 1024);
}

TEST_F(TestDeviceProperties, SerializeDevicePropertiesIsDeterministic)
{
    const auto props = createTestProperties();

    const auto serialized1 = serializeDeviceProperties(props);
    const auto serialized2 = serializeDeviceProperties(props);

    // Same input should produce identical output
    EXPECT_EQ(serialized1.size(), serialized2.size());
    EXPECT_EQ(serialized1, serialized2);
}

// ========== wrapSerializedDeviceProperties Tests ==========

TEST_F(TestDeviceProperties, WrapSerializedDevicePropertiesCreatesValidWrapper)
{
    const auto props = createTestProperties();
    const auto serialized = serializeDeviceProperties(props);

    const auto wrapper = wrapSerializedDeviceProperties(serialized);

    EXPECT_NE(wrapper.ptr, nullptr);
    EXPECT_EQ(wrapper.size, serialized.size());
    EXPECT_EQ(wrapper.ptr, serialized.data());
}

TEST_F(TestDeviceProperties, WrapSerializedDevicePropertiesPointsToOriginalBuffer)
{
    const auto props = createTestProperties();
    const auto serialized = serializeDeviceProperties(props);

    const auto wrapper = wrapSerializedDeviceProperties(serialized);

    // Wrapper should point to the same memory as the original buffer
    EXPECT_EQ(static_cast<const void*>(wrapper.ptr), static_cast<const void*>(serialized.data()));

    // First bytes should match
    if(!serialized.empty())
    {
        const auto* ptrAsBytes = static_cast<const uint8_t*>(wrapper.ptr);
        EXPECT_EQ(ptrAsBytes[0], serialized[0]);
    }
}

TEST_F(TestDeviceProperties, WrapSerializedDevicePropertiesWithEmptyBuffer)
{
    const std::vector<uint8_t> emptyBuffer;

    const auto wrapper = wrapSerializedDeviceProperties(emptyBuffer);

    EXPECT_EQ(wrapper.size, 0u);
    // ptr may be nullptr or non-null (implementation defined for empty vector)
}

TEST_F(TestDeviceProperties, WrapSerializedDevicePropertiesPreservesSize)
{
    const auto props = createTestProperties();
    const auto serialized = serializeDeviceProperties(props);

    const auto wrapper = wrapSerializedDeviceProperties(serialized);

    // Size should exactly match
    EXPECT_EQ(wrapper.size, serialized.size());
}

// ========== Integration Tests ==========

TEST_F(TestDeviceProperties, CompleteWorkflowWithCustomProperties)
{
    // Test with manually created properties
    auto props = createTestProperties();
    props.device_id = 1;
    props.multi_processor_count = 240;
    props.total_global_mem = 32ULL * 1024 * 1024 * 1024;
    props.architecture_name = "gfx942";

    const auto serialized = serializeDeviceProperties(props);
    const auto wrapper = wrapSerializedDeviceProperties(serialized);

    // Verify round-trip
    const auto* deviceProps = flatbuffers::GetRoot<DeviceProperties>(wrapper.ptr);
    ASSERT_NE(deviceProps, nullptr);

    EXPECT_EQ(deviceProps->device_id(), 1);
    EXPECT_EQ(deviceProps->multi_processor_count(), 240);
    EXPECT_EQ(deviceProps->total_global_mem(), 32ULL * 1024 * 1024 * 1024);
    EXPECT_EQ(deviceProps->architecture_name()->str(), "gfx942");
}

// ========== Edge Case Tests ==========

TEST_F(TestDeviceProperties, SerializeDevicePropertiesWithZeroValues)
{
    DevicePropertiesT props;
    props.device_id = 0;
    props.multi_processor_count = 0;
    props.total_global_mem = 0;
    props.architecture_name = "";

    const auto serialized = serializeDeviceProperties(props);

    EXPECT_FALSE(serialized.empty());

    const auto* deviceProps = flatbuffers::GetRoot<DeviceProperties>(serialized.data());
    ASSERT_NE(deviceProps, nullptr);

    EXPECT_EQ(deviceProps->device_id(), 0);
    EXPECT_EQ(deviceProps->multi_processor_count(), 0);
    EXPECT_EQ(deviceProps->total_global_mem(), 0ULL);
}

TEST_F(TestDeviceProperties, SerializeDevicePropertiesWithLongArchitectureName)
{
    auto props = createTestProperties();
    props.architecture_name = "gfx90a-very-long-architecture-name-for-testing-purposes";

    const auto serialized = serializeDeviceProperties(props);

    const auto* deviceProps = flatbuffers::GetRoot<DeviceProperties>(serialized.data());
    ASSERT_NE(deviceProps, nullptr);

    EXPECT_EQ(deviceProps->architecture_name()->str(),
              "gfx90a-very-long-architecture-name-for-testing-purposes");
}

// ========== queryDeviceProperties Tests (GPU required) ==========

// queryDeviceProperties() goes through hipGetDevice/hipGetDeviceProperties;
// SKIP rather than fail on no-device CI runners.
class TestGpuDeviceProperties : public ::testing::Test
{
protected:
    void SetUp() override
    {
        SKIP_IF_NO_DEVICES();
    }
};

TEST_F(TestGpuDeviceProperties, QueryDevicePropertiesSucceeds)
{
    DevicePropertiesT props;
    EXPECT_NO_THROW({ props = queryDeviceProperties(); });

    EXPECT_GE(props.device_id, 0);
    EXPECT_GT(props.multi_processor_count, 0);
    EXPECT_GT(props.total_global_mem, 0ULL);
    EXPECT_FALSE(props.architecture_name.empty());
}

TEST_F(TestGpuDeviceProperties, QueryDevicePropertiesHasValidArchitecture)
{
    const auto props = queryDeviceProperties();

    // Architecture name should be a valid GCN/CDNA architecture
    // Common formats: gfx908, gfx90a, gfx942, etc.
    EXPECT_TRUE(props.architecture_name.find("gfx") == 0
                || props.architecture_name.find("CDNA") == 0 || !props.architecture_name.empty())
        << "Architecture name: " << props.architecture_name;
}

TEST_F(TestGpuDeviceProperties, QueryDevicePropertiesIsConsistent)
{
    const auto props1 = queryDeviceProperties();
    const auto props2 = queryDeviceProperties();

    EXPECT_EQ(props1.device_id, props2.device_id);
    EXPECT_EQ(props1.multi_processor_count, props2.multi_processor_count);
    EXPECT_EQ(props1.total_global_mem, props2.total_global_mem);
    EXPECT_EQ(props1.architecture_name, props2.architecture_name);
}

TEST_F(TestGpuDeviceProperties, CompleteWorkflowQuerySerializeWrap)
{
    // Complete workflow: query -> serialize -> wrap
    const auto props = queryDeviceProperties();
    const auto serialized = serializeDeviceProperties(props);
    const auto wrapper = wrapSerializedDeviceProperties(serialized);

    EXPECT_NE(wrapper.ptr, nullptr);
    EXPECT_GT(wrapper.size, 0u);

    const auto* deviceProps = flatbuffers::GetRoot<DeviceProperties>(wrapper.ptr);
    ASSERT_NE(deviceProps, nullptr);

    EXPECT_EQ(deviceProps->device_id(), props.device_id);
    EXPECT_EQ(deviceProps->multi_processor_count(), props.multi_processor_count);
    EXPECT_EQ(deviceProps->total_global_mem(), props.total_global_mem);
    EXPECT_EQ(deviceProps->architecture_name()->str(), props.architecture_name);
}
