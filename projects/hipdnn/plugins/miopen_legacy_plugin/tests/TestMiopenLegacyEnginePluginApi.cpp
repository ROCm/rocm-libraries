// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include <hipdnn_sdk/plugin/EnginePluginApi.h>
#include <hipdnn_sdk/plugin/PluginApiDataTypes.h>
#include <hipdnn_sdk/plugin/flatbuffer_utilities/EngineDetailsWrapper.hpp>
#include <hipdnn_sdk/test_utilities/FlatbufferGraphTestUtils.hpp>
#include <hipdnn_sdk/test_utilities/TestUtilities.hpp>

#include "HipdnnEnginePluginExecutionContext.hpp"
#include "HipdnnEnginePluginHandle.hpp"
#include "mocks/MockHipdnnEnginePluginExecutionContext.hpp"

TEST(TestMiopenLegacyEnginePluginApi, GetAllEngineIdsNull)
{
    std::array<int64_t, 1> engineIds = {0};
    uint32_t numEngines = 0;

    EXPECT_EQ(hipdnnEnginePluginGetAllEngineIds(nullptr, 0, nullptr),
              HIPDNN_PLUGIN_STATUS_BAD_PARAM);

    EXPECT_EQ(hipdnnEnginePluginGetAllEngineIds(nullptr, 1, &numEngines),
              HIPDNN_PLUGIN_STATUS_BAD_PARAM);

    EXPECT_EQ(hipdnnEnginePluginGetAllEngineIds(engineIds.data(), 1, nullptr),
              HIPDNN_PLUGIN_STATUS_BAD_PARAM);
}

TEST(TestMiopenLegacyEnginePluginApi, GetAllEngineIdsValid)
{
    std::array<int64_t, 1> engineIds = {0};
    uint32_t numEngines = 0;

    //get max 1 engine
    auto status = hipdnnEnginePluginGetAllEngineIds(engineIds.data(), 1, &numEngines);

    EXPECT_EQ(status, HIPDNN_PLUGIN_STATUS_SUCCESS);
    EXPECT_EQ(numEngines, 1u);
    EXPECT_EQ(engineIds[0], 1u);

    status = hipdnnEnginePluginGetAllEngineIds(nullptr, 0, &numEngines);

    EXPECT_EQ(status, HIPDNN_PLUGIN_STATUS_SUCCESS);
    EXPECT_EQ(numEngines, 1u);
}

TEST(TestMiopenLegacyEnginePluginApi, CreateNullHandle)
{
    EXPECT_EQ(hipdnnEnginePluginCreate(nullptr), HIPDNN_PLUGIN_STATUS_BAD_PARAM);
}

TEST(TestMiopenLegacyEnginePluginApi, DestroyNullHandle)
{
    EXPECT_EQ(hipdnnEnginePluginDestroy(nullptr), HIPDNN_PLUGIN_STATUS_BAD_PARAM);
}

TEST(TestMiopenLegacyEnginePluginApi, SetStreamNullHandle)
{
    EXPECT_EQ(hipdnnEnginePluginSetStream(nullptr, nullptr), HIPDNN_PLUGIN_STATUS_BAD_PARAM);
}

TEST(TestMiopenLegacyEnginePluginApi, GetApplicableEngineIdsNull)
{
    auto handle = reinterpret_cast<hipdnnEnginePluginHandle_t>(0x1234);
    auto opGraph = reinterpret_cast<hipdnnPluginConstData_t*>(0x5678);
    std::array<int64_t, 1> engineIds = {0};
    uint32_t numEngines = 0;

    EXPECT_EQ(hipdnnEnginePluginGetApplicableEngineIds(nullptr, nullptr, nullptr, 0, nullptr),
              HIPDNN_PLUGIN_STATUS_BAD_PARAM);

    EXPECT_EQ(hipdnnEnginePluginGetApplicableEngineIds(
                  nullptr, opGraph, engineIds.data(), 1, &numEngines),
              HIPDNN_PLUGIN_STATUS_BAD_PARAM);

    EXPECT_EQ(
        hipdnnEnginePluginGetApplicableEngineIds(handle, nullptr, engineIds.data(), 1, &numEngines),
        HIPDNN_PLUGIN_STATUS_BAD_PARAM);

    EXPECT_EQ(hipdnnEnginePluginGetApplicableEngineIds(handle, opGraph, nullptr, 1, &numEngines),
              HIPDNN_PLUGIN_STATUS_BAD_PARAM);

    EXPECT_EQ(
        hipdnnEnginePluginGetApplicableEngineIds(handle, opGraph, engineIds.data(), 1, nullptr),
        HIPDNN_PLUGIN_STATUS_BAD_PARAM);
}

TEST(TestMiopenLegacyEnginePluginApi, GetEngineDetailsNull)
{
    auto handle = reinterpret_cast<hipdnnEnginePluginHandle_t>(0x1234);
    auto opGraph = reinterpret_cast<hipdnnPluginConstData_t*>(0x5678);
    hipdnnPluginConstData_t engineDetailsOut;

    EXPECT_EQ(hipdnnEnginePluginGetEngineDetails(nullptr, 1, nullptr, nullptr),
              HIPDNN_PLUGIN_STATUS_BAD_PARAM);

    EXPECT_EQ(hipdnnEnginePluginGetEngineDetails(nullptr, 1, opGraph, &engineDetailsOut),
              HIPDNN_PLUGIN_STATUS_BAD_PARAM);

    EXPECT_EQ(hipdnnEnginePluginGetEngineDetails(handle, 1, nullptr, &engineDetailsOut),
              HIPDNN_PLUGIN_STATUS_BAD_PARAM);

    EXPECT_EQ(hipdnnEnginePluginGetEngineDetails(handle, 1, opGraph, nullptr),
              HIPDNN_PLUGIN_STATUS_BAD_PARAM);
}

TEST(TestMiopenLegacyEnginePluginApi, DestroyEngineDetailsNull)
{
    auto handle = reinterpret_cast<hipdnnEnginePluginHandle_t>(0x1234);
    hipdnnPluginConstData_t engineDetailsOut;

    EXPECT_EQ(hipdnnEnginePluginDestroyEngineDetails(nullptr, &engineDetailsOut),
              HIPDNN_PLUGIN_STATUS_BAD_PARAM);

    EXPECT_EQ(hipdnnEnginePluginDestroyEngineDetails(handle, nullptr),
              HIPDNN_PLUGIN_STATUS_BAD_PARAM);

    engineDetailsOut.ptr = nullptr;
    EXPECT_EQ(hipdnnEnginePluginDestroyEngineDetails(handle, nullptr),
              HIPDNN_PLUGIN_STATUS_BAD_PARAM);
}

TEST(TestMiopenLegacyEnginePluginApi, GetWorkspaceSizeNull)
{
    auto handle = reinterpret_cast<hipdnnEnginePluginHandle_t>(0x1234);
    auto engineConfig = reinterpret_cast<hipdnnPluginConstData_t*>(0x5678);
    auto opGraph = reinterpret_cast<hipdnnPluginConstData_t*>(0x9abc);
    size_t workspaceSize = 123;

    // Null handle
    EXPECT_EQ(hipdnnEnginePluginGetWorkspaceSize(nullptr, engineConfig, opGraph, &workspaceSize),
              HIPDNN_PLUGIN_STATUS_BAD_PARAM);

    // Null engineConfig
    EXPECT_EQ(hipdnnEnginePluginGetWorkspaceSize(handle, nullptr, opGraph, &workspaceSize),
              HIPDNN_PLUGIN_STATUS_BAD_PARAM);

    // Null opGraph
    EXPECT_EQ(hipdnnEnginePluginGetWorkspaceSize(handle, engineConfig, nullptr, &workspaceSize),
              HIPDNN_PLUGIN_STATUS_BAD_PARAM);

    // Null workspaceSize
    EXPECT_EQ(hipdnnEnginePluginGetWorkspaceSize(handle, engineConfig, opGraph, nullptr),
              HIPDNN_PLUGIN_STATUS_BAD_PARAM);
}

TEST(TestMiopenLegacyEnginePluginApi, CreateExecutionContextNull)
{
    auto handle = reinterpret_cast<hipdnnEnginePluginHandle_t>(0x1234);
    auto engineConfig = reinterpret_cast<hipdnnPluginConstData_t*>(0x5678);
    auto opGraph = reinterpret_cast<hipdnnPluginConstData_t*>(0x9abc);
    hipdnnEnginePluginExecutionContext_t executionContext;

    // Null handle
    EXPECT_EQ(
        hipdnnEnginePluginCreateExecutionContext(nullptr, engineConfig, opGraph, &executionContext),
        HIPDNN_PLUGIN_STATUS_BAD_PARAM);

    // Null engineConfig
    EXPECT_EQ(hipdnnEnginePluginCreateExecutionContext(handle, nullptr, opGraph, &executionContext),
              HIPDNN_PLUGIN_STATUS_BAD_PARAM);

    // Null opGraph
    EXPECT_EQ(
        hipdnnEnginePluginCreateExecutionContext(handle, engineConfig, nullptr, &executionContext),
        HIPDNN_PLUGIN_STATUS_BAD_PARAM);

    // Null executionContext
    EXPECT_EQ(hipdnnEnginePluginCreateExecutionContext(handle, engineConfig, opGraph, nullptr),
              HIPDNN_PLUGIN_STATUS_BAD_PARAM);
}

TEST(TestMiopenLegacyEnginePluginApi, DestroyExecutionContextNull)
{
    auto handle = reinterpret_cast<hipdnnEnginePluginHandle_t>(0x1234);
    MockHipdnnEnginePluginExecutionContext executionContext;

    EXPECT_EQ(hipdnnEnginePluginDestroyExecutionContext(nullptr, &executionContext),
              HIPDNN_PLUGIN_STATUS_BAD_PARAM);

    EXPECT_EQ(hipdnnEnginePluginDestroyExecutionContext(handle, nullptr),
              HIPDNN_PLUGIN_STATUS_BAD_PARAM);
}

TEST(TestMiopenLegacyEnginePluginApi, GetWorkspaceSizeFromExecutionContextNull)
{
    auto handle = reinterpret_cast<hipdnnEnginePluginHandle_t>(0x1234);
    auto executionContext = reinterpret_cast<hipdnnEnginePluginExecutionContext_t>(0x5678);
    size_t workspaceSize = 123;

    // Null handle
    EXPECT_EQ(hipdnnEnginePluginGetWorkspaceSizeFromExecutionContext(
                  nullptr, executionContext, &workspaceSize),
              HIPDNN_PLUGIN_STATUS_BAD_PARAM);

    // Null executionContext
    EXPECT_EQ(
        hipdnnEnginePluginGetWorkspaceSizeFromExecutionContext(handle, nullptr, &workspaceSize),
        HIPDNN_PLUGIN_STATUS_BAD_PARAM);

    // Null workspaceSize
    EXPECT_EQ(
        hipdnnEnginePluginGetWorkspaceSizeFromExecutionContext(handle, executionContext, nullptr),
        HIPDNN_PLUGIN_STATUS_BAD_PARAM);
}

TEST(TestGpuMiopenLegacyEnginePluginApi, CreateAlsoCreatesMIOpenHandleOnSuccess)
{
    SKIP_IF_NO_DEVICES();
    hipdnnEnginePluginHandle_t handle = nullptr;

    auto status = hipdnnEnginePluginCreate(&handle);

    EXPECT_EQ(status, HIPDNN_PLUGIN_STATUS_SUCCESS);
    ASSERT_NE(handle, nullptr);
    ASSERT_NE(handle->miopenHandle, nullptr);

    EXPECT_EQ(hipdnnEnginePluginDestroy(handle), HIPDNN_PLUGIN_STATUS_SUCCESS);
}

TEST(TestGpuMiopenLegacyEnginePluginApi, CreateTwiceGivesTheSameContainerHandle)
{
    SKIP_IF_NO_DEVICES();
    hipdnnEnginePluginHandle_t handle1 = nullptr;
    auto status1 = hipdnnEnginePluginCreate(&handle1);

    hipdnnEnginePluginHandle_t handle2 = nullptr;
    auto status2 = hipdnnEnginePluginCreate(&handle2);

    EXPECT_EQ(status1, HIPDNN_PLUGIN_STATUS_SUCCESS);
    EXPECT_EQ(status2, HIPDNN_PLUGIN_STATUS_SUCCESS);

    EXPECT_EQ(handle1->miopenContainer, handle2->miopenContainer);

    EXPECT_EQ(hipdnnEnginePluginDestroy(handle1), HIPDNN_PLUGIN_STATUS_SUCCESS);
    EXPECT_EQ(hipdnnEnginePluginDestroy(handle2), HIPDNN_PLUGIN_STATUS_SUCCESS);
}

TEST(TestGpuMiopenLegacyEnginePluginApi, CreateNonNullHandlePointer)
{
    SKIP_IF_NO_DEVICES();
    auto handle = reinterpret_cast<hipdnnEnginePluginHandle_t>(0x1234);
    auto status = hipdnnEnginePluginCreate(&handle);
    EXPECT_EQ(status, HIPDNN_PLUGIN_STATUS_SUCCESS);
    ASSERT_NE(handle, nullptr);
    // Clean up
    EXPECT_EQ(hipdnnEnginePluginDestroy(handle), HIPDNN_PLUGIN_STATUS_SUCCESS);
}

TEST(TestGpuMiopenLegacyEnginePluginApi, SetStreamNullStream)
{
    SKIP_IF_NO_DEVICES();
    hipdnnEnginePluginHandle_t handle = nullptr;
    EXPECT_EQ(hipdnnEnginePluginCreate(&handle), HIPDNN_PLUGIN_STATUS_SUCCESS);

    EXPECT_EQ(hipdnnEnginePluginSetStream(handle, nullptr), HIPDNN_PLUGIN_STATUS_SUCCESS);
    // Clean up
    EXPECT_EQ(hipdnnEnginePluginDestroy(handle), HIPDNN_PLUGIN_STATUS_SUCCESS);
}

TEST(TestGpuMiopenLegacyEnginePluginApi, SetStreamValidStream)
{
    SKIP_IF_NO_DEVICES();
    hipdnnEnginePluginHandle_t handle = nullptr;
    EXPECT_EQ(hipdnnEnginePluginCreate(&handle), HIPDNN_PLUGIN_STATUS_SUCCESS);

    hipStream_t stream = nullptr;
    EXPECT_EQ(hipStreamCreate(&stream), hipSuccess);
    EXPECT_EQ(hipdnnEnginePluginSetStream(handle, stream), HIPDNN_PLUGIN_STATUS_SUCCESS);
    EXPECT_EQ(handle->getStream(), stream);

    EXPECT_EQ(hipdnnEnginePluginDestroy(handle), HIPDNN_PLUGIN_STATUS_SUCCESS);
    EXPECT_EQ(hipStreamDestroy(stream), hipSuccess);
}

TEST(TestGpuMiopenLegacyEnginePluginApi, GetApplicableEngineIdsValid)
{
    SKIP_IF_NO_DEVICES();
    hipdnnEnginePluginHandle_t handle = nullptr;
    ASSERT_EQ(hipdnnEnginePluginCreate(&handle), HIPDNN_PLUGIN_STATUS_SUCCESS);

    auto builder = hipdnn_sdk::test_utilities::createValidBatchnormInferenceGraph();
    auto serializedGraph = builder.Release();
    hipdnnPluginConstData_t opGraph
        = hipdnn_sdk::test_utilities::createValidConstDataGraph(serializedGraph);
    std::array<int64_t, 1> engineIds = {0};
    uint32_t numEngines = 0;

    //get max 1 engine
    auto status = hipdnnEnginePluginGetApplicableEngineIds(
        handle, &opGraph, engineIds.data(), 1, &numEngines);

    EXPECT_EQ(status, HIPDNN_PLUGIN_STATUS_SUCCESS);
    EXPECT_EQ(numEngines, 1u);
    EXPECT_EQ(engineIds[0], 1u);

    engineIds[0] = 1337;
    status = hipdnnEnginePluginGetApplicableEngineIds(
        handle, &opGraph, engineIds.data(), 0, &numEngines);

    EXPECT_EQ(status, HIPDNN_PLUGIN_STATUS_SUCCESS);
    EXPECT_EQ(numEngines, 1u);
    EXPECT_EQ(engineIds[0], 1337);

    // Clean up
    EXPECT_EQ(hipdnnEnginePluginDestroy(handle), HIPDNN_PLUGIN_STATUS_SUCCESS);
}
TEST(TestGpuMiopenLegacyEnginePluginApi, GetEngineDetailsValid)
{
    SKIP_IF_NO_DEVICES();
    hipdnnEnginePluginHandle_t handle = nullptr;
    ASSERT_EQ(hipdnnEnginePluginCreate(&handle), HIPDNN_PLUGIN_STATUS_SUCCESS);

    auto builder = hipdnn_sdk::test_utilities::createValidBatchnormInferenceGraph();
    auto serializedGraph = builder.Release();
    hipdnnPluginConstData_t opGraph
        = hipdnn_sdk::test_utilities::createValidConstDataGraph(serializedGraph);
    hipdnnPluginConstData_t engineDetailsOut;

    auto status = hipdnnEnginePluginGetEngineDetails(handle, 1, &opGraph, &engineDetailsOut);

    hipdnn_plugin::EngineDetailsWrapper engineDetails(engineDetailsOut.ptr, engineDetailsOut.size);

    EXPECT_EQ(status, HIPDNN_PLUGIN_STATUS_SUCCESS);
    EXPECT_EQ(engineDetails.engineId(), 1);

    // Clean up
    EXPECT_EQ(hipdnnEnginePluginDestroyEngineDetails(handle, &engineDetailsOut),
              HIPDNN_PLUGIN_STATUS_SUCCESS);
    EXPECT_EQ(hipdnnEnginePluginDestroy(handle), HIPDNN_PLUGIN_STATUS_SUCCESS);
}

TEST(TestGpuMiopenLegacyEnginePluginApi, GetWorkspaceSizeValid)
{
    SKIP_IF_NO_DEVICES();
    hipdnnEnginePluginHandle_t handle = nullptr;
    ASSERT_EQ(hipdnnEnginePluginCreate(&handle), HIPDNN_PLUGIN_STATUS_SUCCESS);

    // Create a valid flatbuffer graph and engine config
    auto builder = hipdnn_sdk::test_utilities::createValidBatchnormInferenceGraph();
    auto serializedGraph = builder.Release();
    hipdnnPluginConstData_t opGraph
        = hipdnn_sdk::test_utilities::createValidConstDataGraph(serializedGraph);

    auto engineConfigBuilder = hipdnn_sdk::test_utilities::createValidEngineConfig(1);
    auto serializedEngineConfig = engineConfigBuilder.Release();
    hipdnnPluginConstData_t engineConfig
        = hipdnn_sdk::test_utilities::createValidConstDataEngineConfig(serializedEngineConfig);

    size_t workspaceSize = 0;
    auto status
        = hipdnnEnginePluginGetWorkspaceSize(handle, &engineConfig, &opGraph, &workspaceSize);

    EXPECT_EQ(status, HIPDNN_PLUGIN_STATUS_SUCCESS);
    EXPECT_EQ(workspaceSize, 0u); // batchnorm workspace size is always 0

    EXPECT_EQ(hipdnnEnginePluginDestroy(handle), HIPDNN_PLUGIN_STATUS_SUCCESS);
}

TEST(TestGpuMiopenLegacyEnginePluginApi, CreateExecutionContextValid)
{
    SKIP_IF_NO_DEVICES();
    hipdnnEnginePluginHandle_t handle = nullptr;
    ASSERT_EQ(hipdnnEnginePluginCreate(&handle), HIPDNN_PLUGIN_STATUS_SUCCESS);

    auto builder = hipdnn_sdk::test_utilities::createValidBatchnormInferenceGraph();
    auto serializedGraph = builder.Release();
    hipdnnPluginConstData_t opGraph
        = hipdnn_sdk::test_utilities::createValidConstDataGraph(serializedGraph);

    auto engineConfigBuilder = hipdnn_sdk::test_utilities::createValidEngineConfig(1);
    auto serializedEngineConfig = engineConfigBuilder.Release();
    hipdnnPluginConstData_t engineConfig
        = hipdnn_sdk::test_utilities::createValidConstDataEngineConfig(serializedEngineConfig);

    hipdnnEnginePluginExecutionContext_t executionContext = nullptr;
    auto status = hipdnnEnginePluginCreateExecutionContext(
        handle, &engineConfig, &opGraph, &executionContext);

    EXPECT_EQ(status, HIPDNN_PLUGIN_STATUS_SUCCESS);
    ASSERT_NE(executionContext, nullptr);

    EXPECT_EQ(hipdnnEnginePluginDestroyExecutionContext(handle, executionContext),
              HIPDNN_PLUGIN_STATUS_SUCCESS);
    EXPECT_EQ(hipdnnEnginePluginDestroy(handle), HIPDNN_PLUGIN_STATUS_SUCCESS);
}

TEST(TestGpuMiopenLegacyEnginePluginApi, GetWorkspaceSizeFromExecutionContextValid)
{
    SKIP_IF_NO_DEVICES();
    hipdnnEnginePluginHandle_t handle = nullptr;
    ASSERT_EQ(hipdnnEnginePluginCreate(&handle), HIPDNN_PLUGIN_STATUS_SUCCESS);

    auto builder = hipdnn_sdk::test_utilities::createValidBatchnormInferenceGraph();
    auto serializedGraph = builder.Release();
    hipdnnPluginConstData_t opGraph
        = hipdnn_sdk::test_utilities::createValidConstDataGraph(serializedGraph);

    auto engineConfigBuilder = hipdnn_sdk::test_utilities::createValidEngineConfig(1);
    auto serializedEngineConfig = engineConfigBuilder.Release();
    hipdnnPluginConstData_t engineConfig
        = hipdnn_sdk::test_utilities::createValidConstDataEngineConfig(serializedEngineConfig);

    hipdnnEnginePluginExecutionContext_t executionContext = nullptr;
    auto status = hipdnnEnginePluginCreateExecutionContext(
        handle, &engineConfig, &opGraph, &executionContext);

    EXPECT_EQ(status, HIPDNN_PLUGIN_STATUS_SUCCESS);
    ASSERT_NE(executionContext, nullptr);

    size_t workspaceSize = 0;
    status = hipdnnEnginePluginGetWorkspaceSizeFromExecutionContext(
        handle, executionContext, &workspaceSize);

    EXPECT_EQ(status, HIPDNN_PLUGIN_STATUS_SUCCESS);
    EXPECT_EQ(workspaceSize, 0u); // batchnorm workspace size is always 0

    EXPECT_EQ(hipdnnEnginePluginDestroyExecutionContext(handle, executionContext),
              HIPDNN_PLUGIN_STATUS_SUCCESS);
    EXPECT_EQ(hipdnnEnginePluginDestroy(handle), HIPDNN_PLUGIN_STATUS_SUCCESS);
}
