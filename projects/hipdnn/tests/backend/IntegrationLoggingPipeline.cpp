// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "TestUtil.hpp"
#include "hipdnn_backend.h"
#include <cstdlib>
#include <filesystem>
#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_test_sdk/utilities/ScopedEnvironmentVariableSetter.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>
#include <test_plugins/TestPluginConstants.hpp>

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

        hipdnn_data_sdk::utilities::setEnv("HIPDNN_LOG_LEVEL", "info");
        hipdnn_data_sdk::utilities::setEnv("HIPDNN_LOG_FILE", _logFile.string().c_str());
    }

    void TearDown() override
    {
        hipdnn_data_sdk::utilities::unsetEnv("HIPDNN_LOG_LEVEL");
        hipdnn_data_sdk::utilities::unsetEnv("HIPDNN_LOG_FILE");

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

    auto createStatus = hipdnnCreate(&handle);
    ASSERT_EQ(createStatus, HIPDNN_STATUS_SUCCESS);
    ASSERT_NE(handle, nullptr);

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

    const std::vector<hipdnnBackendDescriptorType_t> descriptorTypes
        = {HIPDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR,
           HIPDNN_BACKEND_ENGINE_DESCRIPTOR,
           HIPDNN_BACKEND_ENGINECFG_DESCRIPTOR,
           HIPDNN_BACKEND_ENGINEHEUR_DESCRIPTOR,
           HIPDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR,
           HIPDNN_BACKEND_VARIANT_PACK_DESCRIPTOR};

    for(auto type : descriptorTypes)
    {
        hipdnnBackendDescriptor_t descriptor = nullptr;

        auto status = hipdnnBackendCreateDescriptor(type, &descriptor);
        ASSERT_EQ(status, HIPDNN_STATUS_SUCCESS) << "Failed to create descriptor type: " << type;

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

    hipdnnBackendDescriptor_t descriptor = nullptr;
    auto status
        = hipdnnBackendCreateDescriptor(HIPDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR, &descriptor);
    ASSERT_EQ(status, HIPDNN_STATUS_SUCCESS);

    hipdnnHandle_t handle = nullptr;
    ASSERT_EQ(hipdnnCreate(&handle), HIPDNN_STATUS_SUCCESS);

    status = hipdnnBackendSetAttribute(descriptor,
                                       HIPDNN_ATTR_OPERATIONGRAPH_HANDLE,
                                       HIPDNN_TYPE_HANDLE,
                                       1,
                                       static_cast<const void*>(&handle));
    ASSERT_EQ(status, HIPDNN_STATUS_SUCCESS);

    // Note: GraphDescriptor::getAttribute is not supported - it returns NOT_SUPPORTED for all attributes.
    // We exercise GetAttribute logging through EngineHeuristicDescriptor instead.
    hipdnnBackendDescriptor_t heur = nullptr;
    ASSERT_EQ(hipdnnBackendCreateDescriptor(HIPDNN_BACKEND_ENGINEHEUR_DESCRIPTOR, &heur),
              HIPDNN_STATUS_SUCCESS);

    int64_t elementCount = 0;
    status = hipdnnBackendGetAttribute(heur,
                                       HIPDNN_ATTR_ENGINEHEUR_RESULTS,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       0,
                                       &elementCount,
                                       nullptr);
    // Returns BAD_PARAM_NOT_FINALIZED because the descriptor isn't finalized, but logs the enums
    ASSERT_EQ(status, HIPDNN_STATUS_BAD_PARAM_NOT_FINALIZED);

    hipdnnBackendDestroyDescriptor(heur);
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
    status = hipdnnBackendCreateDescriptor(HIPDNN_INVALID_TYPE_EXT, &descriptor);
    ASSERT_EQ(status, HIPDNN_STATUS_NOT_SUPPORTED);
}

// Test full workflow with all logging points
TEST_F(IntegrationGpuLoggingPipeline, FullWorkflowLogging)
{
    SKIP_IF_NO_DEVICES();

    const std::array<const char*, 1> heuristicPaths
        = {hipdnn_tests::plugin_constants::testGoodHeuristicPluginPath().c_str()};
    ASSERT_EQ(hipdnnSetHeuristicPluginPaths_ext(
                  heuristicPaths.size(), heuristicPaths.data(), HIPDNN_PLUGIN_LOADING_ABSOLUTE),
              HIPDNN_STATUS_SUCCESS);
    const hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter policyEnv(
        "HIPDNN_HEUR_POLICY_ORDER", hipdnn_tests::plugin_constants::testGoodHeuristicPolicyName());

    hipdnnHandle_t handle = nullptr;
    ASSERT_EQ(hipdnnCreate(&handle), HIPDNN_STATUS_SUCCESS);

    hipStream_t stream;
    ASSERT_EQ(hipStreamCreate(&stream), hipSuccess);
    ASSERT_EQ(hipdnnSetStream(handle, stream), HIPDNN_STATUS_SUCCESS);

    hipdnnBackendDescriptor_t graph = nullptr;
    test_util::createTestGraph(&graph, handle);
    ASSERT_EQ(hipdnnBackendFinalize(graph), HIPDNN_STATUS_SUCCESS);

    hipdnnBackendDescriptor_t heur = nullptr;
    ASSERT_EQ(hipdnnBackendCreateDescriptor(HIPDNN_BACKEND_ENGINEHEUR_DESCRIPTOR, &heur),
              HIPDNN_STATUS_SUCCESS);

    ASSERT_EQ(hipdnnBackendSetAttribute(heur,
                                        HIPDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH,
                                        HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                        1,
                                        static_cast<const void*>(&graph)),
              HIPDNN_STATUS_SUCCESS);

    auto mode = HIPDNN_HEUR_MODE_FALLBACK;
    ASSERT_EQ(hipdnnBackendSetAttribute(
                  heur, HIPDNN_ATTR_ENGINEHEUR_MODE, HIPDNN_TYPE_HEUR_MODE, 1, &mode),
              HIPDNN_STATUS_SUCCESS);
    ASSERT_EQ(hipdnnBackendFinalize(heur), HIPDNN_STATUS_SUCCESS);

    int64_t count = 0;
    ASSERT_EQ(hipdnnBackendGetAttribute(heur,
                                        HIPDNN_ATTR_ENGINEHEUR_RESULTS,
                                        HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                        0,
                                        &count,
                                        nullptr),
              HIPDNN_STATUS_SUCCESS);

    hipdnnBackendDescriptor_t engine = nullptr;
    ASSERT_EQ(hipdnnBackendCreateDescriptor(HIPDNN_BACKEND_ENGINE_DESCRIPTOR, &engine),
              HIPDNN_STATUS_SUCCESS);

    hipdnnBackendDestroyDescriptor(engine);
    hipdnnBackendDestroyDescriptor(heur);
    hipdnnBackendDestroyDescriptor(graph);
    hipdnnDestroy(handle);
    ASSERT_EQ(hipStreamDestroy(stream), hipSuccess);
}
