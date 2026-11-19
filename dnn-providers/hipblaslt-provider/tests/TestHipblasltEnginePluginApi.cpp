// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <algorithm>
#include <cstdint>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include <HipblasltPlugin.hpp>
#include <algorithm>
#include <array>
#include <hipdnn_data_sdk/utilities/EngineNames.hpp>
#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/EngineDetailsWrapper.hpp>
#include <hipdnn_plugin_sdk/EnginePluginApi.h>
#include <hipdnn_plugin_sdk/NumericalNote.h>
#include <hipdnn_plugin_sdk/PluginApiDataTypes.h>
#include <hipdnn_plugin_sdk/PluginGraphTestUtils.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include "HipdnnEnginePluginExecutionContext.hpp"
#include "HipdnnEnginePluginHandle.hpp"
#include "TestWorkarounds.hpp"
#include "mocks/MockHipdnnEnginePluginExecutionContext.hpp"

using namespace hipdnn_data_sdk::utilities;

TEST(TestHipblasltEnginePluginApi, GetAllEngineIdsNull)
{
    std::array<int64_t, 1> engineIds = {0};
    uint32_t numEngines = 0;

    EXPECT_EQ(hipdnnEnginePluginGetAllEngineIdsImpl(nullptr, 0, nullptr),
              HIPDNN_PLUGIN_STATUS_BAD_PARAM);

    EXPECT_EQ(hipdnnEnginePluginGetAllEngineIdsImpl(nullptr, 1, &numEngines),
              HIPDNN_PLUGIN_STATUS_BAD_PARAM);

    EXPECT_EQ(hipdnnEnginePluginGetAllEngineIdsImpl(engineIds.data(), 1, nullptr),
              HIPDNN_PLUGIN_STATUS_BAD_PARAM);
}

TEST(TestHipblasltEnginePluginApi, GetAllEngineIdsValid)
{
    // One engine per enabled backend.
    std::vector<int64_t> expected;
#ifdef HIPBLASLTPROVIDER_ENABLE_HIPBLASLT
    expected.push_back(HIPBLASLT_ENGINE_ID);
#endif
#ifdef HIPBLASLTPROVIDER_ENABLE_ROCBLAS
    expected.push_back(ROCBLAS_MATMUL_ENGINE_ID);
#endif
#ifdef HIPBLASLTPROVIDER_ENABLE_CONV
    // The exact engines, their TF32 and reduced-precision-reduction siblings and the
    // deterministic siblings of the exact and TF32 engines, by default.
#ifdef HIPBLASLTPROVIDER_ENABLE_HIPBLASLT
    expected.push_back(HIPBLASLT_CONV_ENGINE_ID);
    expected.push_back(HIPBLASLT_CONV_ENGINE_DETERMINISTIC_ID);
    expected.push_back(HIPBLASLT_CONV_TF32_ENGINE_ID);
    expected.push_back(HIPBLASLT_CONV_TF32_ENGINE_DETERMINISTIC_ID);
    expected.push_back(HIPBLASLT_CONV_RPR_ENGINE_ID);
#endif
#ifdef HIPBLASLTPROVIDER_ENABLE_ROCBLAS
    expected.push_back(ROCBLAS_CONV_ENGINE_ID);
    expected.push_back(ROCBLAS_CONV_ENGINE_DETERMINISTIC_ID);
    expected.push_back(ROCBLAS_CONV_TF32_ENGINE_ID);
    expected.push_back(ROCBLAS_CONV_TF32_ENGINE_DETERMINISTIC_ID);
    expected.push_back(ROCBLAS_CONV_RPR_ENGINE_ID);
#endif
#endif

    std::vector<int64_t> engineIds(expected.size(), 0);
    uint32_t numEngines = 0;

    auto status = hipdnnEnginePluginGetAllEngineIdsImpl(
        engineIds.data(), static_cast<uint32_t>(engineIds.size()), &numEngines);

    EXPECT_EQ(status, HIPDNN_PLUGIN_STATUS_SUCCESS);
    EXPECT_EQ(numEngines, expected.size());
    for(const auto id : expected)
    {
        EXPECT_NE(std::find(engineIds.begin(), engineIds.end(), id), engineIds.end());
    }

    status = hipdnnEnginePluginGetAllEngineIdsImpl(nullptr, 0, &numEngines);

    EXPECT_EQ(status, HIPDNN_PLUGIN_STATUS_SUCCESS);
    EXPECT_EQ(numEngines, expected.size());
}

#if defined(HIPBLASLTPROVIDER_ENABLE_CONV) && defined(HIPBLASLTPROVIDER_ENABLE_HIPBLASLT)
TEST(TestHipblasltEnginePluginApi, LibraryActivationEngineIsOptIn)
{
    // Listed only with HIPBLASLT_PROVIDER_CONV_LIBRARY_ACTIVATIONS=1: its fused activations
    // give other special values than every other engine, and no numerical note says so.
    const auto listed = [] {
        uint32_t count = 0;
        EXPECT_EQ(hipdnnEnginePluginGetAllEngineIdsImpl(nullptr, 0, &count),
                  HIPDNN_PLUGIN_STATUS_SUCCESS);
        std::vector<int64_t> ids(count, 0);
        EXPECT_EQ(hipdnnEnginePluginGetAllEngineIdsImpl(ids.data(), count, &count),
                  HIPDNN_PLUGIN_STATUS_SUCCESS);
        return std::find(ids.begin(), ids.end(), HIPBLASLT_CONV_LIBRARY_ACTIVATION_ENGINE_ID)
               != ids.end();
    };
    const char* name = "HIPBLASLT_PROVIDER_CONV_LIBRARY_ACTIVATIONS";
    const std::string previous = getEnv(name);
    unsetEnv(name);
    EXPECT_FALSE(listed());
    setEnv(name, "0");
    EXPECT_FALSE(listed());
    setEnv(name, "1");
    EXPECT_TRUE(listed());
    if(previous.empty())
    {
        unsetEnv(name);
    }
    else
    {
        setEnv(name, previous.c_str());
    }
}
#endif

TEST(TestHipblasltEnginePluginApi, CreateNullHandle)
{
    EXPECT_EQ(hipdnnEnginePluginCreateImpl(nullptr), HIPDNN_PLUGIN_STATUS_BAD_PARAM);
}

TEST(TestHipblasltEnginePluginApi, DestroyNullHandle)
{
    EXPECT_EQ(hipdnnEnginePluginDestroyImpl(nullptr), HIPDNN_PLUGIN_STATUS_BAD_PARAM);
}

TEST(TestHipblasltEnginePluginApi, SetStreamNullHandle)
{
    EXPECT_EQ(hipdnnEnginePluginSetStreamImpl(nullptr, nullptr), HIPDNN_PLUGIN_STATUS_BAD_PARAM);
}

TEST(TestHipblasltEnginePluginApi, GetApplicableEngineIdsNull)
{
    auto handle = reinterpret_cast<hipdnnEnginePluginHandle_t>(0x1234);
    auto opGraph = reinterpret_cast<hipdnnPluginConstData_t*>(0x5678);
    std::array<int64_t, 1> engineIds = {0};
    uint32_t numEngines = 0;

    EXPECT_EQ(hipdnnEnginePluginGetApplicableEngineIdsImpl(nullptr, nullptr, nullptr, 0, nullptr),
              HIPDNN_PLUGIN_STATUS_BAD_PARAM);

    EXPECT_EQ(hipdnnEnginePluginGetApplicableEngineIdsImpl(
                  nullptr, opGraph, engineIds.data(), 1, &numEngines),
              HIPDNN_PLUGIN_STATUS_BAD_PARAM);

    EXPECT_EQ(hipdnnEnginePluginGetApplicableEngineIdsImpl(
                  handle, nullptr, engineIds.data(), 1, &numEngines),
              HIPDNN_PLUGIN_STATUS_BAD_PARAM);

    EXPECT_EQ(
        hipdnnEnginePluginGetApplicableEngineIdsImpl(handle, opGraph, nullptr, 1, &numEngines),
        HIPDNN_PLUGIN_STATUS_BAD_PARAM);

    EXPECT_EQ(
        hipdnnEnginePluginGetApplicableEngineIdsImpl(handle, opGraph, engineIds.data(), 1, nullptr),
        HIPDNN_PLUGIN_STATUS_BAD_PARAM);
}

TEST(TestHipblasltEnginePluginApi, GetEngineDetailsNull)
{
    auto handle = reinterpret_cast<hipdnnEnginePluginHandle_t>(0x1234);
    auto opGraph = reinterpret_cast<hipdnnPluginConstData_t*>(0x5678);
    hipdnnPluginConstData_t engineDetailsOut;

    EXPECT_EQ(hipdnnEnginePluginGetEngineDetailsImpl(nullptr, 1, nullptr, nullptr),
              HIPDNN_PLUGIN_STATUS_BAD_PARAM);

    EXPECT_EQ(hipdnnEnginePluginGetEngineDetailsImpl(nullptr, 1, opGraph, &engineDetailsOut),
              HIPDNN_PLUGIN_STATUS_BAD_PARAM);

    EXPECT_EQ(hipdnnEnginePluginGetEngineDetailsImpl(handle, 1, nullptr, &engineDetailsOut),
              HIPDNN_PLUGIN_STATUS_BAD_PARAM);

    EXPECT_EQ(hipdnnEnginePluginGetEngineDetailsImpl(handle, 1, opGraph, nullptr),
              HIPDNN_PLUGIN_STATUS_BAD_PARAM);
}

TEST(TestHipblasltEnginePluginApi, DestroyEngineDetailsNull)
{
    auto handle = reinterpret_cast<hipdnnEnginePluginHandle_t>(0x1234);
    hipdnnPluginConstData_t engineDetailsOut;

    EXPECT_EQ(hipdnnEnginePluginDestroyEngineDetailsImpl(nullptr, &engineDetailsOut),
              HIPDNN_PLUGIN_STATUS_BAD_PARAM);

    EXPECT_EQ(hipdnnEnginePluginDestroyEngineDetailsImpl(handle, nullptr),
              HIPDNN_PLUGIN_STATUS_BAD_PARAM);

    engineDetailsOut.ptr = nullptr;
    EXPECT_EQ(hipdnnEnginePluginDestroyEngineDetailsImpl(handle, nullptr),
              HIPDNN_PLUGIN_STATUS_BAD_PARAM);
}

TEST(TestHipblasltEnginePluginApi, GetWorkspaceSizeNull)
{
    auto handle = reinterpret_cast<hipdnnEnginePluginHandle_t>(0x1234);
    auto engineConfig = reinterpret_cast<hipdnnPluginConstData_t*>(0x5678);
    auto opGraph = reinterpret_cast<hipdnnPluginConstData_t*>(0x9abc);
    size_t workspaceSize = 123;

    // Null handle
    EXPECT_EQ(
        hipdnnEnginePluginGetWorkspaceSizeImpl(nullptr, engineConfig, opGraph, &workspaceSize),
        HIPDNN_PLUGIN_STATUS_BAD_PARAM);

    // Null engineConfig
    EXPECT_EQ(hipdnnEnginePluginGetWorkspaceSizeImpl(handle, nullptr, opGraph, &workspaceSize),
              HIPDNN_PLUGIN_STATUS_BAD_PARAM);

    // Null opGraph
    EXPECT_EQ(hipdnnEnginePluginGetWorkspaceSizeImpl(handle, engineConfig, nullptr, &workspaceSize),
              HIPDNN_PLUGIN_STATUS_BAD_PARAM);

    // Null workspaceSize
    EXPECT_EQ(hipdnnEnginePluginGetWorkspaceSizeImpl(handle, engineConfig, opGraph, nullptr),
              HIPDNN_PLUGIN_STATUS_BAD_PARAM);
}

TEST(TestHipblasltEnginePluginApi, CreateExecutionContextNull)
{
    auto handle = reinterpret_cast<hipdnnEnginePluginHandle_t>(0x1234);
    auto engineConfig = reinterpret_cast<hipdnnPluginConstData_t*>(0x5678);
    auto opGraph = reinterpret_cast<hipdnnPluginConstData_t*>(0x9abc);
    hipdnnEnginePluginExecutionContext_t executionContext;

    // Null handle
    EXPECT_EQ(hipdnnEnginePluginCreateExecutionContextImpl(
                  nullptr, engineConfig, opGraph, &executionContext),
              HIPDNN_PLUGIN_STATUS_BAD_PARAM);

    // Null engineConfig
    EXPECT_EQ(
        hipdnnEnginePluginCreateExecutionContextImpl(handle, nullptr, opGraph, &executionContext),
        HIPDNN_PLUGIN_STATUS_BAD_PARAM);

    // Null opGraph
    EXPECT_EQ(hipdnnEnginePluginCreateExecutionContextImpl(
                  handle, engineConfig, nullptr, &executionContext),
              HIPDNN_PLUGIN_STATUS_BAD_PARAM);

    // Null executionContext
    EXPECT_EQ(hipdnnEnginePluginCreateExecutionContextImpl(handle, engineConfig, opGraph, nullptr),
              HIPDNN_PLUGIN_STATUS_BAD_PARAM);
}

TEST(TestHipblasltEnginePluginApi, DestroyExecutionContextNull)
{
    auto handle = reinterpret_cast<hipdnnEnginePluginHandle_t>(0x1234);
    MockHipdnnEnginePluginExecutionContext executionContext;

    EXPECT_EQ(hipdnnEnginePluginDestroyExecutionContextImpl(nullptr, &executionContext),
              HIPDNN_PLUGIN_STATUS_BAD_PARAM);

    EXPECT_EQ(hipdnnEnginePluginDestroyExecutionContextImpl(handle, nullptr),
              HIPDNN_PLUGIN_STATUS_BAD_PARAM);
}

TEST(TestHipblasltEnginePluginApi, GetWorkspaceSizeFromExecutionContextNull)
{
    auto handle = reinterpret_cast<hipdnnEnginePluginHandle_t>(0x1234);
    auto executionContext = reinterpret_cast<hipdnnEnginePluginExecutionContext_t>(0x5678);
    size_t workspaceSize = 123;

    // Null handle
    EXPECT_EQ(hipdnnEnginePluginGetWorkspaceSizeFromExecutionContextImpl(
                  nullptr, executionContext, &workspaceSize),
              HIPDNN_PLUGIN_STATUS_BAD_PARAM);

    // Null executionContext
    EXPECT_EQ(
        hipdnnEnginePluginGetWorkspaceSizeFromExecutionContextImpl(handle, nullptr, &workspaceSize),
        HIPDNN_PLUGIN_STATUS_BAD_PARAM);

    // Null workspaceSize
    EXPECT_EQ(hipdnnEnginePluginGetWorkspaceSizeFromExecutionContextImpl(
                  handle, executionContext, nullptr),
              HIPDNN_PLUGIN_STATUS_BAD_PARAM);
}

TEST(TestGpuHipblasltEnginePluginApi, CreateAlsoCreatesHipblasltHandleOnSuccess)
{
    SKIP_IF_NO_DEVICES();
    hipdnnEnginePluginHandle_t handle = nullptr;

    auto status = hipdnnEnginePluginCreateImpl(&handle);

    EXPECT_EQ(status, HIPDNN_PLUGIN_STATUS_SUCCESS);
    ASSERT_NE(handle, nullptr);
    ASSERT_NE(handle->hipblasltHandle, nullptr);

    EXPECT_EQ(hipdnnEnginePluginDestroyImpl(handle), HIPDNN_PLUGIN_STATUS_SUCCESS);
}

TEST(TestGpuHipblasltEnginePluginApi, CreateTwiceGivesTheSameContainerHandle)
{
    SKIP_IF_NO_DEVICES();
    hipdnnEnginePluginHandle_t handle1 = nullptr;
    auto status1 = hipdnnEnginePluginCreateImpl(&handle1);

    hipdnnEnginePluginHandle_t handle2 = nullptr;
    auto status2 = hipdnnEnginePluginCreateImpl(&handle2);

    EXPECT_EQ(status1, HIPDNN_PLUGIN_STATUS_SUCCESS);
    EXPECT_EQ(status2, HIPDNN_PLUGIN_STATUS_SUCCESS);

    EXPECT_EQ(handle1->hipblasltContainer, handle2->hipblasltContainer);

    EXPECT_EQ(hipdnnEnginePluginDestroyImpl(handle1), HIPDNN_PLUGIN_STATUS_SUCCESS);
    EXPECT_EQ(hipdnnEnginePluginDestroyImpl(handle2), HIPDNN_PLUGIN_STATUS_SUCCESS);
}

TEST(TestGpuHipblasltEnginePluginApi, CreateNonNullHandlePointer)
{
    SKIP_IF_NO_DEVICES();
    auto handle = reinterpret_cast<hipdnnEnginePluginHandle_t>(0x1234);
    auto status = hipdnnEnginePluginCreateImpl(&handle);
    EXPECT_EQ(status, HIPDNN_PLUGIN_STATUS_SUCCESS);
    ASSERT_NE(handle, nullptr);
    // Clean up
    EXPECT_EQ(hipdnnEnginePluginDestroyImpl(handle), HIPDNN_PLUGIN_STATUS_SUCCESS);
}

TEST(TestGpuHipblasltEnginePluginApi, SetStreamNullStream)
{
    SKIP_IF_NO_DEVICES();
    hipdnnEnginePluginHandle_t handle = nullptr;
    EXPECT_EQ(hipdnnEnginePluginCreateImpl(&handle), HIPDNN_PLUGIN_STATUS_SUCCESS);

    EXPECT_EQ(hipdnnEnginePluginSetStreamImpl(handle, nullptr), HIPDNN_PLUGIN_STATUS_SUCCESS);
    // Clean up
    EXPECT_EQ(hipdnnEnginePluginDestroyImpl(handle), HIPDNN_PLUGIN_STATUS_SUCCESS);
}

TEST(TestGpuHipblasltEnginePluginApi, SetStreamValidStream)
{
    SKIP_IF_NO_DEVICES();
    hipdnnEnginePluginHandle_t handle = nullptr;
    EXPECT_EQ(hipdnnEnginePluginCreateImpl(&handle), HIPDNN_PLUGIN_STATUS_SUCCESS);

    hipStream_t stream = nullptr;
    EXPECT_EQ(hipStreamCreate(&stream), hipSuccess);
    EXPECT_EQ(hipdnnEnginePluginSetStreamImpl(handle, stream), HIPDNN_PLUGIN_STATUS_SUCCESS);
    EXPECT_EQ(handle->getStream(), stream);

    EXPECT_EQ(hipdnnEnginePluginDestroyImpl(handle), HIPDNN_PLUGIN_STATUS_SUCCESS);
    EXPECT_EQ(hipStreamDestroy(stream), hipSuccess);
}

TEST(TestGpuHipblasltEnginePluginApi, GetEngineDetailsValid)
{
    SKIP_IF_NO_DEVICES();
    hipdnnEnginePluginHandle_t handle = nullptr;
    ASSERT_EQ(hipdnnEnginePluginCreateImpl(&handle), HIPDNN_PLUGIN_STATUS_SUCCESS);

    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormFwdTrainingGraph();
    auto serializedGraph = builder.Release();
    hipdnnPluginConstData_t const opGraph
        = hipdnn_plugin_sdk::createValidConstDataGraph(serializedGraph);
    hipdnnPluginConstData_t engineDetailsOut;

    auto status = hipdnnEnginePluginGetEngineDetailsImpl(
        handle, HIPBLASLT_ENGINE_ID, &opGraph, &engineDetailsOut);

    hipdnn_flatbuffers_sdk::flatbuffer_utilities::EngineDetailsWrapper const engineDetails(
        engineDetailsOut.ptr, engineDetailsOut.size);

    EXPECT_EQ(status, HIPDNN_PLUGIN_STATUS_SUCCESS);
    EXPECT_EQ(engineDetails.engineId(), HIPBLASLT_ENGINE_ID);

    // Clean up
    EXPECT_EQ(hipdnnEnginePluginDestroyEngineDetailsImpl(handle, &engineDetailsOut),
              HIPDNN_PLUGIN_STATUS_SUCCESS);
    EXPECT_EQ(hipdnnEnginePluginDestroyImpl(handle), HIPDNN_PLUGIN_STATUS_SUCCESS);
}

#ifdef HIPBLASLTPROVIDER_ENABLE_CONV
// Each convolution engine's RFC 0025 numerical notes: TENSOR_CORE for TF32 (cuDNN's TF32 note),
// REDUCED_PRECISION_REDUCTION for the fp16 partial sums, and NONDETERMINISTIC for every engine
// but the *_DETERMINISTIC siblings, which claim determinism (TestGpuConvGemmDeterministic reruns
// their plans and compares the bits). Deselecting NONDETERMINISTIC therefore leaves exactly the
// deterministic siblings.
TEST(TestGpuHipblasltEnginePluginApi, ConvEnginesReportTheirNumericalNotes)
{
    SKIP_IF_NO_DEVICES();
    hipdnnEnginePluginHandle_t handle = nullptr;
    ASSERT_EQ(hipdnnEnginePluginCreateImpl(&handle), HIPDNN_PLUGIN_STATUS_SUCCESS);

    // Notes describe the engine: any graph serves the query.
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormFwdTrainingGraph();
    auto serializedGraph = builder.Release();
    hipdnnPluginConstData_t const opGraph
        = hipdnn_plugin_sdk::createValidConstDataGraph(serializedGraph);

    using Notes = std::vector<int32_t>;
    constexpr int32_t TF32 = HIPDNN_NUMERICAL_NOTE_TENSOR_CORE;
    constexpr int32_t RPR = HIPDNN_NUMERICAL_NOTE_REDUCED_PRECISION_REDUCTION;
    constexpr int32_t NONDETERMINISTIC = HIPDNN_NUMERICAL_NOTE_NONDETERMINISTIC;
    std::vector<std::pair<int64_t, Notes>> expected;
    std::set<int64_t> deterministic;
#ifdef HIPBLASLTPROVIDER_ENABLE_HIPBLASLT
    expected.emplace_back(HIPBLASLT_CONV_ENGINE_ID, Notes{NONDETERMINISTIC});
    expected.emplace_back(HIPBLASLT_CONV_ENGINE_DETERMINISTIC_ID, Notes{});
    expected.emplace_back(HIPBLASLT_CONV_TF32_ENGINE_ID, Notes{TF32, NONDETERMINISTIC});
    expected.emplace_back(HIPBLASLT_CONV_TF32_ENGINE_DETERMINISTIC_ID, Notes{TF32});
    expected.emplace_back(HIPBLASLT_CONV_RPR_ENGINE_ID, Notes{RPR, NONDETERMINISTIC});
    deterministic.insert(
        {HIPBLASLT_CONV_ENGINE_DETERMINISTIC_ID, HIPBLASLT_CONV_TF32_ENGINE_DETERMINISTIC_ID});
#endif
#ifdef HIPBLASLTPROVIDER_ENABLE_ROCBLAS
    expected.emplace_back(ROCBLAS_CONV_ENGINE_ID, Notes{NONDETERMINISTIC});
    expected.emplace_back(ROCBLAS_CONV_ENGINE_DETERMINISTIC_ID, Notes{});
    expected.emplace_back(ROCBLAS_CONV_TF32_ENGINE_ID, Notes{TF32, NONDETERMINISTIC});
    expected.emplace_back(ROCBLAS_CONV_TF32_ENGINE_DETERMINISTIC_ID, Notes{TF32});
    expected.emplace_back(ROCBLAS_CONV_RPR_ENGINE_ID, Notes{RPR, NONDETERMINISTIC});
    deterministic.insert(
        {ROCBLAS_CONV_ENGINE_DETERMINISTIC_ID, ROCBLAS_CONV_TF32_ENGINE_DETERMINISTIC_ID});
#endif
    // Deselecting NONDETERMINISTIC (cuDNN's deterministic-only filter) over every engine the
    // plugin lists keeps exactly the deterministic siblings.
    uint32_t count = 0;
    ASSERT_EQ(hipdnnEnginePluginGetAllEngineIdsImpl(nullptr, 0, &count),
              HIPDNN_PLUGIN_STATUS_SUCCESS);
    std::vector<int64_t> allIds(count, 0);
    ASSERT_EQ(hipdnnEnginePluginGetAllEngineIdsImpl(allIds.data(), count, &count),
              HIPDNN_PLUGIN_STATUS_SUCCESS);
    std::set<int64_t> kept;
    for(const auto engineId : allIds)
    {
        hipdnnPluginConstData_t details;
        ASSERT_EQ(hipdnnEnginePluginGetEngineDetailsImpl(handle, engineId, &opGraph, &details),
                  HIPDNN_PLUGIN_STATUS_SUCCESS);
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::EngineDetailsWrapper wrapper(
            details.ptr, details.size);
        // An absent list is reported NONDETERMINISTIC (RFC 0025).
        const auto notes = wrapper.numericalNotes();
        if(notes.has_value()
           && std::find(notes->begin(), notes->end(), NONDETERMINISTIC) == notes->end())
        {
            kept.insert(engineId);
        }
        EXPECT_EQ(hipdnnEnginePluginDestroyEngineDetailsImpl(handle, &details),
                  HIPDNN_PLUGIN_STATUS_SUCCESS);
    }
    EXPECT_EQ(kept, deterministic);
    for(const auto& [engineId, notes] : expected)
    {
        SCOPED_TRACE(engineId);
        hipdnnPluginConstData_t engineDetailsOut;
        ASSERT_EQ(
            hipdnnEnginePluginGetEngineDetailsImpl(handle, engineId, &opGraph, &engineDetailsOut),
            HIPDNN_PLUGIN_STATUS_SUCCESS);
        hipdnn_flatbuffers_sdk::flatbuffer_utilities::EngineDetailsWrapper const engineDetails(
            engineDetailsOut.ptr, engineDetailsOut.size);
        const auto reported = engineDetails.numericalNotes();
        ASSERT_TRUE(reported.has_value());
        EXPECT_EQ(*reported, notes);
        EXPECT_EQ(hipdnnEnginePluginDestroyEngineDetailsImpl(handle, &engineDetailsOut),
                  HIPDNN_PLUGIN_STATUS_SUCCESS);
    }
    EXPECT_EQ(hipdnnEnginePluginDestroyImpl(handle), HIPDNN_PLUGIN_STATUS_SUCCESS);
}
#endif

TEST(TestGpuHipblasltEnginePluginApi, GetWorkspaceSizeValid)
{
    SKIP_IF_NO_DEVICES();
    [[maybe_unused]] const HipdnnEnginePluginHandle probe{};
    SKIP_IF_WORKAROUND_ISSUE_9962(probe);
    hipdnnEnginePluginHandle_t handle = nullptr;
    ASSERT_EQ(hipdnnEnginePluginCreateImpl(&handle), HIPDNN_PLUGIN_STATUS_SUCCESS);

    // Create a valid flatbuffer graph and engine config
    auto builder = hipdnn_test_sdk::utilities::createValidMatmulGraph();
    auto serializedGraph = builder.Release();
    hipdnnPluginConstData_t const opGraph
        = hipdnn_plugin_sdk::createValidConstDataGraph(serializedGraph);

    auto engineConfigBuilder
        = hipdnn_test_sdk::utilities::createValidEngineConfig(HIPBLASLT_ENGINE_ID);
    auto serializedEngineConfig = engineConfigBuilder.Release();
    hipdnnPluginConstData_t const engineConfig
        = hipdnn_plugin_sdk::createValidConstDataEngineConfig(serializedEngineConfig);

    // The engine must serve the graph for the workspace query to be meaningful.
    std::array<int64_t, 8> engineIds = {};
    uint32_t numEngines = 0;
    ASSERT_EQ(hipdnnEnginePluginGetApplicableEngineIdsImpl(
                  handle, &opGraph, engineIds.data(), engineIds.size(), &numEngines),
              HIPDNN_PLUGIN_STATUS_SUCCESS);
    ASSERT_LE(numEngines, engineIds.size());
    ASSERT_NE(std::find(engineIds.begin(), engineIds.begin() + numEngines, HIPBLASLT_ENGINE_ID),
              engineIds.begin() + numEngines);

    size_t workspaceSize = 0;
    auto status
        = hipdnnEnginePluginGetWorkspaceSizeImpl(handle, &engineConfig, &opGraph, &workspaceSize);

    EXPECT_EQ(status, HIPDNN_PLUGIN_STATUS_SUCCESS);

    EXPECT_EQ(hipdnnEnginePluginDestroyImpl(handle), HIPDNN_PLUGIN_STATUS_SUCCESS);
}

TEST(TestGpuHipblasltEnginePluginApi, CreateExecutionContextValid)
{
    SKIP_IF_NO_DEVICES();
    [[maybe_unused]] const HipdnnEnginePluginHandle probe{};
    SKIP_IF_WORKAROUND_ISSUE_9962(probe);
    hipdnnEnginePluginHandle_t handle = nullptr;
    ASSERT_EQ(hipdnnEnginePluginCreateImpl(&handle), HIPDNN_PLUGIN_STATUS_SUCCESS);

    auto builder = hipdnn_test_sdk::utilities::createValidMatmulGraph();
    auto serializedGraph = builder.Release();
    hipdnnPluginConstData_t const opGraph
        = hipdnn_plugin_sdk::createValidConstDataGraph(serializedGraph);

    auto engineConfigBuilder
        = hipdnn_test_sdk::utilities::createValidEngineConfig(HIPBLASLT_ENGINE_ID);
    auto serializedEngineConfig = engineConfigBuilder.Release();
    hipdnnPluginConstData_t const engineConfig
        = hipdnn_plugin_sdk::createValidConstDataEngineConfig(serializedEngineConfig);

    hipdnnEnginePluginExecutionContext_t executionContext = nullptr;
    auto status = hipdnnEnginePluginCreateExecutionContextImpl(
        handle, &engineConfig, &opGraph, &executionContext);

    EXPECT_EQ(status, HIPDNN_PLUGIN_STATUS_SUCCESS);
    ASSERT_NE(executionContext, nullptr);

    EXPECT_EQ(hipdnnEnginePluginDestroyExecutionContextImpl(handle, executionContext),
              HIPDNN_PLUGIN_STATUS_SUCCESS);
    EXPECT_EQ(hipdnnEnginePluginDestroyImpl(handle), HIPDNN_PLUGIN_STATUS_SUCCESS);
}

TEST(TestGpuHipblasltEnginePluginApi, CreateExecutionContextReturnsInternalErrorForUnservedGraph)
{
    SKIP_IF_NO_DEVICES();
    hipdnnEnginePluginHandle_t handle = nullptr;
    ASSERT_EQ(hipdnnEnginePluginCreateImpl(&handle), HIPDNN_PLUGIN_STATUS_SUCCESS);

    // No hipBLASLt plan builder serves batchnorm, so no context is created.
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormFwdTrainingGraph();
    auto serializedGraph = builder.Release();
    hipdnnPluginConstData_t const opGraph
        = hipdnn_plugin_sdk::createValidConstDataGraph(serializedGraph);

    auto engineConfigBuilder
        = hipdnn_test_sdk::utilities::createValidEngineConfig(HIPBLASLT_ENGINE_ID);
    auto serializedEngineConfig = engineConfigBuilder.Release();
    hipdnnPluginConstData_t const engineConfig
        = hipdnn_plugin_sdk::createValidConstDataEngineConfig(serializedEngineConfig);

    hipdnnEnginePluginExecutionContext_t executionContext = nullptr;
    auto status = hipdnnEnginePluginCreateExecutionContextImpl(
        handle, &engineConfig, &opGraph, &executionContext);

    EXPECT_EQ(status, HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR);
    EXPECT_EQ(executionContext, nullptr);

    EXPECT_EQ(hipdnnEnginePluginDestroyImpl(handle), HIPDNN_PLUGIN_STATUS_SUCCESS);
}
