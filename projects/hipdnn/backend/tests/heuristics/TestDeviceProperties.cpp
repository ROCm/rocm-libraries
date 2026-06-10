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

#include "HipdnnException.hpp"

using namespace hipdnn_backend::heuristics;
using namespace hipdnn_flatbuffers_sdk::data_objects;

namespace
{

DevicePropertiesT createTestProperties()
{
    DevicePropertiesT props;
    props.device_id = 0;
    props.multi_processor_count = 120;
    props.total_global_mem = 16ULL * 1024 * 1024 * 1024; // 16GB
    props.architecture_name = "gfx90a";
    return props;
}

// Deserialize a serialized DevicePropertiesT from a buffer and assert every
// scalar field round-trips. architecture_name is checked separately when the
// caller cares about the empty/null distinction.
void expectMatchesProps(const void* buf, const DevicePropertiesT& expected)
{
    const auto* deviceProps = flatbuffers::GetRoot<DeviceProperties>(buf);
    ASSERT_NE(deviceProps, nullptr);

    EXPECT_EQ(deviceProps->device_id(), expected.device_id);
    EXPECT_EQ(deviceProps->multi_processor_count(), expected.multi_processor_count);
    EXPECT_EQ(deviceProps->total_global_mem(), expected.total_global_mem);

    if(!expected.architecture_name.empty())
    {
        ASSERT_NE(deviceProps->architecture_name(), nullptr);
        EXPECT_EQ(deviceProps->architecture_name()->str(), expected.architecture_name);
    }
}

} // namespace

// Serialize/wrap tests build DevicePropertiesT in-memory and don't touch HIP,
// so they can run on CPU-only CI runners. GPU-requiring tests that call
// queryDeviceProperties() live in TestGpuDeviceProperties below.
class TestDeviceProperties : public ::testing::Test
{
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

    expectMatchesProps(serialized.data(), props);
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
    // FlatBuffers may omit empty strings (null pointer) or store them as empty
    // strings; both are valid round-trips of the empty input.
    const auto* arch = deviceProps->architecture_name();
    EXPECT_TRUE(arch == nullptr || arch->str().empty())
        << "Expected null or empty architecture, got '"
        << (arch != nullptr ? arch->str() : "<null>") << "'";
}

TEST_F(TestDeviceProperties, SerializeDevicePropertiesHandlesLargeValues)
{
    auto props = createTestProperties();
    props.multi_processor_count = 65536; // Large CU count
    props.total_global_mem = 128ULL * 1024 * 1024 * 1024; // 128GB

    const auto serialized = serializeDeviceProperties(props);

    expectMatchesProps(serialized.data(), props);
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

    expectMatchesProps(wrapper.ptr, props);
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
    expectMatchesProps(serialized.data(), props);
}

TEST_F(TestDeviceProperties, SerializeDevicePropertiesWithLongArchitectureName)
{
    auto props = createTestProperties();
    props.architecture_name = "gfx90a-very-long-architecture-name-for-testing-purposes";

    const auto serialized = serializeDeviceProperties(props);

    expectMatchesProps(serialized.data(), props);
}

// ========== queryDeviceProperties(handle) null-handle Test ==========

TEST_F(TestDeviceProperties, QueryDevicePropertiesNullHandleThrows)
{
    EXPECT_THROW(queryDeviceProperties(static_cast<hipdnnHandle_t>(nullptr)),
                 hipdnn_backend::HipdnnException);
}

// ========== queryDeviceProperties Tests (GPU required) ==========

// queryDeviceProperties(int) goes through hipGetDeviceProperties; SKIP rather
// than fail on no-device CI runners. The Handle overload is exercised by
// EngineHeuristicDescriptor integration tests, which already own the handle
// infrastructure.
class TestGpuDeviceProperties : public ::testing::Test
{
protected:
    void SetUp() override
    {
        SKIP_IF_NO_DEVICES();
        ASSERT_EQ(hipGetDevice(&_currentDevice), hipSuccess);
    }

    int _currentDevice = 0;
};

TEST_F(TestGpuDeviceProperties, QueryDevicePropertiesSucceeds)
{
    DevicePropertiesT props;
    EXPECT_NO_THROW({ props = queryDeviceProperties(_currentDevice); });

    EXPECT_EQ(props.device_id, _currentDevice);
    EXPECT_GT(props.multi_processor_count, 0);
    EXPECT_GT(props.total_global_mem, 0ULL);
    EXPECT_FALSE(props.architecture_name.empty());
}

TEST_F(TestGpuDeviceProperties, QueryDevicePropertiesHasValidArchitecture)
{
    const auto props = queryDeviceProperties(_currentDevice);

    // hipDeviceProp_t::gcnArchName always reports an AMD GPU architecture name
    // starting with "gfx" (e.g. gfx908, gfx90a, gfx942).
    EXPECT_EQ(props.architecture_name.rfind("gfx", 0), 0u)
        << "Architecture name should start with 'gfx', got: " << props.architecture_name;
}

TEST_F(TestGpuDeviceProperties, QueryDevicePropertiesIsConsistent)
{
    const auto props1 = queryDeviceProperties(_currentDevice);
    const auto props2 = queryDeviceProperties(_currentDevice);

    EXPECT_EQ(props1.device_id, props2.device_id);
    EXPECT_EQ(props1.multi_processor_count, props2.multi_processor_count);
    EXPECT_EQ(props1.total_global_mem, props2.total_global_mem);
    EXPECT_EQ(props1.architecture_name, props2.architecture_name);
}

// Pins the documented "throws HipdnnException on HIP failure" contract for the
// int overload. Requires the HIP runtime, so it lives in the GPU fixture even
// though it only exercises an error path.
TEST_F(TestGpuDeviceProperties, QueryDevicePropertiesInvalidDeviceIdThrows)
{
    constexpr int INVALID_DEVICE_ID = 99999;
    EXPECT_THROW(queryDeviceProperties(INVALID_DEVICE_ID), hipdnn_backend::HipdnnException);

    // hipGetDeviceProperties leaves the invalid-device-ordinal error sticky in
    // HIP's per-thread state. HipErrorHandler::OnTestEnd will otherwise flag it
    // as a leak and fail this test. Drain both error channels here.
    (void)hipGetLastError();
    (void)hipExtGetLastError();
}

TEST_F(TestGpuDeviceProperties, CompleteWorkflowQuerySerializeWrap)
{
    // Complete workflow: query -> serialize -> wrap
    const auto props = queryDeviceProperties(_currentDevice);
    const auto serialized = serializeDeviceProperties(props);
    const auto wrapper = wrapSerializedDeviceProperties(serialized);

    EXPECT_NE(wrapper.ptr, nullptr);
    EXPECT_GT(wrapper.size, 0u);

    expectMatchesProps(wrapper.ptr, props);
}
