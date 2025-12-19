// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "TestUtil.hpp"
#include "hipdnn_backend.h"
#include <cstdlib>
#include <filesystem>
#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

namespace fs = std::filesystem;

// Test fixture that enables logging to a temp file
class IntegrationGpuLoggingPipeline : public ::testing::Test
{
protected:
    fs::path _logFile;

    void SetUp() override
    {
        // Create temp log file path
        _logFile = fs::temp_directory_path() / "hipdnn_test_log.txt";

        // Enable logging to file
        setenv("HIPDNN_LOG_LEVEL", "info", 1);
        setenv("HIPDNN_LOG_FILE", _logFile.c_str(), 1);
    }

    void TearDown() override
    {
        unsetenv("HIPDNN_LOG_LEVEL");
        unsetenv("HIPDNN_LOG_FILE");

        // Clean up temp log file
        if(fs::exists(_logFile))
        {
            fs::remove(_logFile);
        }
    }
};

// Test that handle creation/destruction logging doesn't crash
TEST_F(IntegrationGpuLoggingPipeline, HandleLogging)
{
    SKIP_IF_NO_DEVICES();

    hipdnnHandle_t handle = nullptr;

    // This should trigger logging of handle creation with toString()
    auto createStatus = hipdnnCreate(&handle);
    ASSERT_EQ(createStatus, HIPDNN_STATUS_SUCCESS);
    ASSERT_NE(handle, nullptr);

    // This should trigger logging of handle destruction
    auto destroyStatus = hipdnnDestroy(handle);
    ASSERT_EQ(destroyStatus, HIPDNN_STATUS_SUCCESS);
}

// Test that stream logging (logHipDeviceInfo) doesn't crash with real HIP stream
TEST_F(IntegrationGpuLoggingPipeline, StreamLogging)
{
    SKIP_IF_NO_DEVICES();

    hipdnnHandle_t handle = nullptr;
    ASSERT_EQ(hipdnnCreate(&handle), HIPDNN_STATUS_SUCCESS);

    hipStream_t stream;
    ASSERT_EQ(hipStreamCreate(&stream), hipSuccess) << "Failed to create HIP stream.";

    // This should trigger logHipDeviceInfo() with a real stream
    auto setStreamStatus = hipdnnSetStream(handle, stream);
    ASSERT_EQ(setStreamStatus, HIPDNN_STATUS_SUCCESS);

    ASSERT_EQ(hipdnnDestroy(handle), HIPDNN_STATUS_SUCCESS);
    ASSERT_EQ(hipStreamDestroy(stream), hipSuccess) << "Failed to destroy HIP stream.";
}

// Test that descriptor logging (toString via logPtr) doesn't crash
TEST_F(IntegrationGpuLoggingPipeline, DescriptorLogging)
{
    SKIP_IF_NO_DEVICES();

    // Test various descriptor types to exercise toString() implementations
    std::vector<hipdnnBackendDescriptorType_t> descriptorTypes = {
        HIPDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR,
        HIPDNN_BACKEND_ENGINE_DESCRIPTOR,
        HIPDNN_BACKEND_ENGINECFG_DESCRIPTOR,
        HIPDNN_BACKEND_ENGINEHEUR_DESCRIPTOR,
        HIPDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR,
        HIPDNN_BACKEND_VARIANT_PACK_DESCRIPTOR};

    for(auto type : descriptorTypes)
    {
        hipdnnBackendDescriptor_t descriptor = nullptr;

        // This should trigger logging with descriptor toString()
        auto status = hipdnnBackendCreateDescriptor(type, &descriptor);
        ASSERT_EQ(status, HIPDNN_STATUS_SUCCESS) << "Failed to create descriptor type: " << type;

        // Destroy should also log
        status = hipdnnBackendDestroyDescriptor(descriptor);
        ASSERT_EQ(status, HIPDNN_STATUS_SUCCESS) << "Failed to destroy descriptor type: " << type;
    }
}

// Test that finalize logging with descriptor details doesn't crash
TEST_F(IntegrationGpuLoggingPipeline, FinalizeLogging)
{
    SKIP_IF_NO_DEVICES();

    hipdnnHandle_t handle = nullptr;
    hipdnnBackendDescriptor_t graph = nullptr;

    test_util::createTestHandle(&handle);
    test_util::createTestGraph(&graph, handle);

    // Finalize should log descriptor details
    auto status = hipdnnBackendFinalize(graph);
    ASSERT_EQ(status, HIPDNN_STATUS_SUCCESS);

    hipdnnBackendDestroyDescriptor(graph);
    hipdnnDestroy(handle);
}

// Test that enum formatting in logs doesn't crash
TEST_F(IntegrationGpuLoggingPipeline, EnumFormatting)
{
    SKIP_IF_NO_DEVICES();

    // Exercise various enum types through API calls that log them

    // Test hipdnnBackendDescriptorType_t formatting
    hipdnnBackendDescriptor_t descriptor = nullptr;
    auto status
        = hipdnnBackendCreateDescriptor(HIPDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR, &descriptor);
    ASSERT_EQ(status, HIPDNN_STATUS_SUCCESS);

    // Test hipdnnBackendAttributeName_t and hipdnnBackendAttributeType_t formatting
    hipdnnHandle_t handle = nullptr;
    ASSERT_EQ(hipdnnCreate(&handle), HIPDNN_STATUS_SUCCESS);

    // SetAttribute logs attributeName and attributeType enums
    status = hipdnnBackendSetAttribute(
        descriptor, HIPDNN_ATTR_OPERATIONGRAPH_HANDLE, HIPDNN_TYPE_HANDLE, 1, &handle);
    ASSERT_EQ(status, HIPDNN_STATUS_SUCCESS);

    // GetAttribute also logs these enums
    int64_t elementCount = 0;
    status = hipdnnBackendGetAttribute(descriptor,
                                       HIPDNN_ATTR_OPERATIONGRAPH_HANDLE,
                                       HIPDNN_TYPE_HANDLE,
                                       0,
                                       &elementCount,
                                       nullptr);
    ASSERT_EQ(status, HIPDNN_STATUS_SUCCESS);

    hipdnnBackendDestroyDescriptor(descriptor);
    hipdnnDestroy(handle);
}

// Test that error status logging formats correctly
TEST_F(IntegrationGpuLoggingPipeline, ErrorStatusLogging)
{
    SKIP_IF_NO_DEVICES();

    // Intentionally cause errors to test error logging paths
    // These should log hipdnnStatus_t enum values

    // Null pointer error
    auto status = hipdnnCreate(nullptr);
    ASSERT_EQ(status, HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);

    // Invalid descriptor type
    hipdnnBackendDescriptor_t descriptor = nullptr;
    status = hipdnnBackendCreateDescriptor(HIPDNN_INVALID_TYPE, &descriptor);
    ASSERT_EQ(status, HIPDNN_STATUS_NOT_SUPPORTED);
}

// Test full workflow with all logging points
TEST_F(IntegrationGpuLoggingPipeline, FullWorkflowLogging)
{
    SKIP_IF_NO_DEVICES();

    // Create handle
    hipdnnHandle_t handle = nullptr;
    ASSERT_EQ(hipdnnCreate(&handle), HIPDNN_STATUS_SUCCESS);

    // Set stream (triggers device info logging)
    hipStream_t stream;
    ASSERT_EQ(hipStreamCreate(&stream), hipSuccess);
    ASSERT_EQ(hipdnnSetStream(handle, stream), HIPDNN_STATUS_SUCCESS);

    // Create and setup graph descriptor
    hipdnnBackendDescriptor_t graph = nullptr;
    test_util::createTestGraph(&graph, handle);
    ASSERT_EQ(hipdnnBackendFinalize(graph), HIPDNN_STATUS_SUCCESS);

    // Create engine descriptor
    hipdnnBackendDescriptor_t engine = nullptr;
    ASSERT_EQ(hipdnnBackendCreateDescriptor(HIPDNN_BACKEND_ENGINE_DESCRIPTOR, &engine),
              HIPDNN_STATUS_SUCCESS);

    int64_t gidx = 0;
    ASSERT_EQ(hipdnnBackendSetAttribute(
                  engine, HIPDNN_ATTR_ENGINE_OPERATION_GRAPH, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &graph),
              HIPDNN_STATUS_SUCCESS);
    ASSERT_EQ(hipdnnBackendSetAttribute(
                  engine, HIPDNN_ATTR_ENGINE_GLOBAL_INDEX, HIPDNN_TYPE_INT64, 1, &gidx),
              HIPDNN_STATUS_SUCCESS);
    ASSERT_EQ(hipdnnBackendFinalize(engine), HIPDNN_STATUS_SUCCESS);

    // Create engine config descriptor
    hipdnnBackendDescriptor_t engineConfig = nullptr;
    ASSERT_EQ(hipdnnBackendCreateDescriptor(HIPDNN_BACKEND_ENGINECFG_DESCRIPTOR, &engineConfig),
              HIPDNN_STATUS_SUCCESS);
    ASSERT_EQ(
        hipdnnBackendSetAttribute(
            engineConfig, HIPDNN_ATTR_ENGINECFG_ENGINE, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &engine),
        HIPDNN_STATUS_SUCCESS);
    ASSERT_EQ(hipdnnBackendFinalize(engineConfig), HIPDNN_STATUS_SUCCESS);

    // Cleanup
    hipdnnBackendDestroyDescriptor(engineConfig);
    hipdnnBackendDestroyDescriptor(engine);
    hipdnnBackendDestroyDescriptor(graph);
    hipdnnDestroy(handle);
    ASSERT_EQ(hipStreamDestroy(stream), hipSuccess);
}

