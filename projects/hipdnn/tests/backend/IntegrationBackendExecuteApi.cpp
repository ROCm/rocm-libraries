// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "HipdnnStatus.h"
#include "ScopedBackendDescriptor.hpp"
#include "ScopedBackendLogCapture.hpp"
#include "TestMacros.hpp"
#include "TestUtil.hpp"
#include "hipdnn_backend.h"
#include <test_plugins/TestPluginConstants.hpp>

#include <hip/hip_runtime.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/utilities/Uuid.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>
#include <hipdnn_test_sdk/utilities/LogRecorder.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <memory>
#include <regex>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

using hipdnn_backend::test_utilities::ScopedBackendDescriptor;

namespace
{
// Reads the graph id that the backend stamped into the serialized graph.
void getSerializedGraphId(hipdnnBackendDescriptor_t graph, std::string& graphId)
{
    size_t size = 0;
    ASSERT_EQ(hipdnnBackendGetSerializedBinaryGraph_ext(graph, 0, &size, nullptr),
              HIPDNN_STATUS_SUCCESS);
    std::vector<uint8_t> buffer(size);
    ASSERT_EQ(hipdnnBackendGetSerializedBinaryGraph_ext(graph, size, &size, buffer.data()),
              HIPDNN_STATUS_SUCCESS);
    const auto* id = hipdnn_flatbuffers_sdk::data_objects::GetGraph(buffer.data())->id();
    ASSERT_NE(id, nullptr);
    graphId = hipdnn_flatbuffers_sdk::utilities::formatUuid(
        hipdnn_flatbuffers_sdk::utilities::toUuidBytes(*id));
}

struct ExecuteLogFields
{
    std::string planId;
    std::string graphId;
    std::string executionCount;
};

// Parses every "API success: [hipdnnBackendExecute]" log that the recorder captured, in order.
std::vector<ExecuteLogFields>
    parseExecuteSuccessLogs(const hipdnn_test_sdk::utilities::IsolatedLogRecorder& recorder)
{
    const std::regex pattern(R"(planId=(\d+), graphId=([0-9a-f-]{36}),.*executionCount=(\d+))");
    std::vector<ExecuteLogFields> fields;
    for(const auto& log : recorder.getRecordedLogs())
    {
        if(log.message.find("API success: [hipdnnBackendExecute]") == std::string::npos)
        {
            continue;
        }
        std::smatch match;
        EXPECT_TRUE(std::regex_search(log.message, match, pattern)) << log.message;
        if(!match.empty())
        {
            fields.push_back({match[1].str(), match[2].str(), match[3].str()});
        }
    }
    return fields;
}

using ScopedStream
    = std::unique_ptr<std::remove_pointer_t<hipStream_t>, decltype(&hipStreamDestroy)>;
using ScopedHandle
    = std::unique_ptr<std::remove_pointer_t<hipdnnHandle_t>, decltype(&hipdnnDestroy)>;

ScopedStream createStream()
{
    hipStream_t stream = nullptr;
    EXPECT_EQ(hipStreamCreate(&stream), hipSuccess);
    return {stream, &hipStreamDestroy};
}

ScopedHandle createHandle(hipStream_t stream)
{
    hipdnnHandle_t handle = nullptr;
    EXPECT_EQ(hipdnnCreate(&handle), HIPDNN_STATUS_SUCCESS);
    EXPECT_EQ(hipdnnSetStream(handle, stream), HIPDNN_STATUS_SUCCESS);
    return {handle, &hipdnnDestroy};
}

std::string takeLastErrorString()
{
    std::array<char, HIPDNN_ERROR_STRING_MAX_LENGTH> buffer{};
    hipdnnGetLastErrorString(buffer.data(), buffer.size());
    return buffer.data();
}
} // namespace

class IntegrationBackendExecuteApi : public ::testing::Test
{
protected:
    static constexpr int64_t GIDX = hipdnn_tests::plugin_constants::engineId<GoodPlugin>();
    hipdnnBackendDescriptor_t _plan = nullptr;
    hipdnnHandle_t _handle = nullptr;
    hipdnnBackendDescriptor_t _engineConfig = nullptr;
    hipdnnBackendDescriptor_t _engine = nullptr;
    hipdnnBackendDescriptor_t _graphDescriptor = nullptr;

    hipdnnBackendDescriptor_t _variantPack = nullptr;

    void SetUp() override
    {
        const std::array<const char*, 1> paths
            = {hipdnn_tests::plugin_constants::testGoodPluginPath().c_str()};
        ASSERT_EQ(hipdnnSetEnginePluginPaths_ext(
                      paths.size(), paths.data(), HIPDNN_PLUGIN_LOADING_ABSOLUTE),
                  HIPDNN_STATUS_SUCCESS);

        ASSERT_EQ(hipdnnCreate(&_handle), HIPDNN_STATUS_SUCCESS);
    }

    void TearDown() override
    {
        destroyTestHandle();
        destroyTestDescriptor(_plan);
        destroyTestDescriptor(_engineConfig);
        destroyTestDescriptor(_engine);
        destroyTestDescriptor(_graphDescriptor);
        destroyTestDescriptor(_variantPack);
        for(void* tensorData : _tensorData)
        {
            test_util::freeTensorMemory(tensorData);
        }
    }

    // Creates a finalized batchnorm plan in _plan and a matching variant pack in _variantPack.
    void createExecutablePlan()
    {
        auto batchnormBuilder = test_util::createAndPopulateBatchnormNode();
        auto serializedGraph = batchnormBuilder.Release();

        std::unordered_map<int64_t, std::string> uidToNameMap;
        std::unordered_map<std::string, int64_t> nameToUidMap;
        std::unordered_map<int64_t, std::vector<int64_t>> uidToDimsMap;

        test_util::extractTensorInfoFromGraph(
            serializedGraph, uidToNameMap, nameToUidMap, uidToDimsMap);

        test_util::createAndInitializeBackendDescriptor(
            &_graphDescriptor, serializedGraph, _handle);
        test_util::createTestEngine(&_engine, &_graphDescriptor, _handle, GIDX, true);
        test_util::createTestEngineConfig(
            &_engineConfig, &_engine, &_graphDescriptor, _handle, GIDX, true);

        ASSERT_EQ(hipdnnBackendCreateDescriptor(HIPDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR, &_plan),
                  HIPDNN_STATUS_SUCCESS);

        test_util::populateTestExecutionPlan(
            &_plan, &_engineConfig, &_engine, &_graphDescriptor, _handle, GIDX, true);

        std::unordered_map<int64_t, void*> dataPtrMappings;

        for(const auto& [uid, dims] : uidToDimsMap)
        {
            void* tensorData = test_util::allocateTensorMemory(
                dims.data(), dims.size(), HIPDNN_TYPE_FLOAT, true);

            ASSERT_NE(tensorData, nullptr)
                << "Failed to allocate memory for tensor " << uidToNameMap[uid];

            _tensorData.push_back(tensorData);
            dataPtrMappings[uid] = tensorData;
        }

        ASSERT_EQ(
            hipdnnBackendCreateDescriptor(HIPDNN_BACKEND_VARIANT_PACK_DESCRIPTOR, &_variantPack),
            HIPDNN_STATUS_SUCCESS);

        test_util::populateVariantPackWithMappings(_variantPack, dataPtrMappings, nullptr);
    }

    static void destroyTestDescriptor(hipdnnBackendDescriptor_t descriptor)
    {
        if(descriptor != nullptr)
        {
            EXPECT_EQ(hipdnnBackendDestroyDescriptor(descriptor), HIPDNN_STATUS_SUCCESS);
            descriptor = nullptr;
        }
    }

private:
    std::vector<void*> _tensorData;

    void destroyTestHandle()
    {
        if(_handle != nullptr)
        {
            EXPECT_EQ(hipdnnDestroy(_handle), HIPDNN_STATUS_SUCCESS);
            _handle = nullptr;
        }
    }
};

TEST_F(IntegrationBackendExecuteApi, NullHandle)
{
    auto batchnormBuilder = test_util::createAndPopulateBatchnormNode();
    auto serializedGraph = batchnormBuilder.Release();

    test_util::createAndInitializeBackendDescriptor(&_graphDescriptor, serializedGraph, _handle);
    test_util::createTestEngine(&_engine, &_graphDescriptor, _handle, GIDX, true);
    test_util::createTestEngineConfig(
        &_engineConfig, &_engine, &_graphDescriptor, _handle, GIDX, true);

    EXPECT_EQ(hipdnnBackendCreateDescriptor(HIPDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR, &_plan),
              HIPDNN_STATUS_SUCCESS);

    test_util::populateTestExecutionPlan(
        &_plan, &_engineConfig, &_engine, &_graphDescriptor, _handle, GIDX, true);

    ASSERT_EQ(hipdnnBackendCreateDescriptor(HIPDNN_BACKEND_VARIANT_PACK_DESCRIPTOR, &_variantPack),
              HIPDNN_STATUS_SUCCESS);

    ASSERT_EQ(hipdnnBackendExecute(nullptr, _plan, _variantPack),
              HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

TEST_F(IntegrationBackendExecuteApi, NullVariantPack)
{
    auto batchnormBuilder = test_util::createAndPopulateBatchnormNode();
    auto serializedGraph = batchnormBuilder.Release();

    test_util::createAndInitializeBackendDescriptor(&_graphDescriptor, serializedGraph, _handle);
    test_util::createTestEngine(&_engine, &_graphDescriptor, _handle, GIDX, true);
    test_util::createTestEngineConfig(
        &_engineConfig, &_engine, &_graphDescriptor, _handle, GIDX, true);

    EXPECT_EQ(hipdnnBackendCreateDescriptor(HIPDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR, &_plan),
              HIPDNN_STATUS_SUCCESS);

    test_util::populateTestExecutionPlan(
        &_plan, &_engineConfig, &_engine, &_graphDescriptor, _handle, GIDX, true);

    ASSERT_EQ(hipdnnBackendExecute(_handle, _plan, nullptr), HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

TEST_F(IntegrationBackendExecuteApi, NullPlan)
{
    ASSERT_EQ(hipdnnBackendCreateDescriptor(HIPDNN_BACKEND_VARIANT_PACK_DESCRIPTOR, &_variantPack),
              HIPDNN_STATUS_SUCCESS);

    ASSERT_EQ(hipdnnBackendExecute(_handle, nullptr, _variantPack),
              HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

TEST_F(IntegrationBackendExecuteApi, UnfinalizedPlan)
{
    hipdnnBackendDescriptor_t unfinalizedPlan = nullptr;
    ASSERT_EQ(
        hipdnnBackendCreateDescriptor(HIPDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR, &unfinalizedPlan),
        HIPDNN_STATUS_SUCCESS);

    auto batchnormBuilder = test_util::createAndPopulateBatchnormNode();
    auto serializedGraph = batchnormBuilder.Release();

    test_util::createAndInitializeBackendDescriptor(&_graphDescriptor, serializedGraph, _handle);

    ASSERT_EQ(hipdnnBackendCreateDescriptor(HIPDNN_BACKEND_VARIANT_PACK_DESCRIPTOR, &_variantPack),
              HIPDNN_STATUS_SUCCESS);

    ASSERT_EQ(hipdnnBackendExecute(_handle, unfinalizedPlan, _variantPack),
              HIPDNN_STATUS_BAD_PARAM);

    destroyTestDescriptor(unfinalizedPlan);
}

TEST_F(IntegrationBackendExecuteApi, WrongDescriptorTypes)
{
    auto batchnormBuilder = test_util::createAndPopulateBatchnormNode();
    auto serializedGraph = batchnormBuilder.Release();

    test_util::createAndInitializeBackendDescriptor(&_graphDescriptor, serializedGraph, _handle);
    test_util::createTestEngine(&_engine, &_graphDescriptor, _handle, GIDX, true);

    ASSERT_EQ(hipdnnBackendCreateDescriptor(HIPDNN_BACKEND_VARIANT_PACK_DESCRIPTOR, &_variantPack),
              HIPDNN_STATUS_SUCCESS);

    ASSERT_EQ(hipdnnBackendExecute(_handle, _engine, _variantPack), HIPDNN_STATUS_BAD_PARAM);

    EXPECT_EQ(hipdnnBackendCreateDescriptor(HIPDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR, &_plan),
              HIPDNN_STATUS_SUCCESS);

    test_util::populateTestExecutionPlan(
        &_plan, &_engineConfig, &_engine, &_graphDescriptor, _handle, GIDX, true);
    ASSERT_EQ(hipdnnBackendExecute(_handle, _plan, _graphDescriptor), HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(IntegrationBackendExecuteApi, ValidExecute)
{
    ASSERT_NO_FATAL_FAILURE(createExecutablePlan());

    ASSERT_EQ(hipdnnBackendExecute(_handle, _plan, _variantPack), HIPDNN_STATUS_SUCCESS);
}

TEST_F(IntegrationBackendExecuteApi, ExecuteLogsExecutionCountPerPlan)
{
    using hipdnn_test_sdk::utilities::IsolatedLogRecorder;

    ASSERT_NO_FATAL_FAILURE(createExecutablePlan());

    hipdnnBackendDescriptor_t rawOtherPlan = nullptr;
    ASSERT_EQ(
        hipdnnBackendCreateDescriptor(HIPDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR, &rawOtherPlan),
        HIPDNN_STATUS_SUCCESS);
    const ScopedBackendDescriptor otherPlan(rawOtherPlan);
    ASSERT_NO_FATAL_FAILURE(test_util::populateTestExecutionPlan(
        &rawOtherPlan, &_engineConfig, &_engine, &_graphDescriptor, _handle, GIDX, true));

    auto recorder = IsolatedLogRecorder::withOverrideLevel(HIPDNN_SEV_INFO);
    const hipdnn_tests::ScopedBackendLogCapture logCapture(
        IsolatedLogRecorder::getIsolatedUserRecordingCallback(), HIPDNN_SEV_INFO, this);

    ASSERT_EQ(hipdnnBackendExecute(_handle, _plan, _variantPack), HIPDNN_STATUS_SUCCESS);
    ASSERT_EQ(hipdnnBackendExecute(_handle, _plan, _variantPack), HIPDNN_STATUS_SUCCESS);
    ASSERT_EQ(hipdnnBackendExecute(_handle, otherPlan.get(), _variantPack), HIPDNN_STATUS_SUCCESS);
    ASSERT_EQ(hipdnnBackendExecute(_handle, _plan, _variantPack), HIPDNN_STATUS_SUCCESS);

    const auto logs = parseExecuteSuccessLogs(recorder);
    ASSERT_EQ(logs.size(), 4u) << recorder.getRecordedLogsAsString();

    // Each log names the plan by planId.
    EXPECT_NE(logs[0].planId, logs[2].planId);
    EXPECT_EQ(logs[1].planId, logs[0].planId);
    EXPECT_EQ(logs[3].planId, logs[0].planId);

    // Both plans log the id of the graph they were finalized from.
    std::string expectedGraphId;
    ASSERT_NO_FATAL_FAILURE(getSerializedGraphId(_graphDescriptor, expectedGraphId));
    for(const auto& log : logs)
    {
        EXPECT_EQ(log.graphId, expectedGraphId);
    }

    // Counts are per plan: the other plan starts at 1 and does not advance this plan's count.
    EXPECT_EQ(logs[0].executionCount, "1");
    EXPECT_EQ(logs[1].executionCount, "2");
    EXPECT_EQ(logs[2].executionCount, "1");
    EXPECT_EQ(logs[3].executionCount, "3");
}

TEST_F(IntegrationBackendExecuteApi, FailedExecuteDoesNotAdvanceExecutionCount)
{
    using hipdnn_test_sdk::utilities::IsolatedLogRecorder;

    ASSERT_NO_FATAL_FAILURE(createExecutablePlan());

    // The plugin resource manager rejects an unfinalized variant pack, so this execute fails
    // after the API-level argument checks pass.
    hipdnnBackendDescriptor_t rawUnfinalizedVariantPack = nullptr;
    ASSERT_EQ(hipdnnBackendCreateDescriptor(HIPDNN_BACKEND_VARIANT_PACK_DESCRIPTOR,
                                            &rawUnfinalizedVariantPack),
              HIPDNN_STATUS_SUCCESS);
    const ScopedBackendDescriptor unfinalizedVariantPack(rawUnfinalizedVariantPack);

    auto recorder = IsolatedLogRecorder::withOverrideLevel(HIPDNN_SEV_INFO);
    const hipdnn_tests::ScopedBackendLogCapture logCapture(
        IsolatedLogRecorder::getIsolatedUserRecordingCallback(), HIPDNN_SEV_INFO, this);

    ASSERT_EQ(hipdnnBackendExecute(_handle, _plan, unfinalizedVariantPack.get()),
              HIPDNN_STATUS_BAD_PARAM);
    ASSERT_EQ(hipdnnBackendExecute(_handle, _plan, _variantPack), HIPDNN_STATUS_SUCCESS);

    const auto logs = parseExecuteSuccessLogs(recorder);
    ASSERT_EQ(logs.size(), 1u) << recorder.getRecordedLogsAsString();
    EXPECT_EQ(logs[0].executionCount, "1");
}

// A plan is bound to the device of the handle it was built on and may run on any handle whose
// stream is on that device. These tests switch the current device and restore it on teardown.
class IntegrationGpuBackendExecuteDevice : public IntegrationBackendExecuteApi
{
protected:
    int _deviceCount = 0;
    int _originalDevice = -1;

    void SetUp() override
    {
        SKIP_IF_NO_DEVICES();
        ASSERT_EQ(hipGetDeviceCount(&_deviceCount), hipSuccess);
        ASSERT_EQ(hipGetDevice(&_originalDevice), hipSuccess);
        IntegrationBackendExecuteApi::SetUp();
    }

    void TearDown() override
    {
        IntegrationBackendExecuteApi::TearDown();
        if(_originalDevice >= 0)
        {
            EXPECT_EQ(hipSetDevice(_originalDevice), hipSuccess);
        }
    }
};

TEST_F(IntegrationGpuBackendExecuteDevice, PlanRunsOnAnotherThreadsHandleForTheSameDevice)
{
    ASSERT_NO_FATAL_FAILURE(createExecutablePlan());

    hipdnnStatus_t status = HIPDNN_STATUS_INTERNAL_ERROR;
    std::thread worker([&] {
        ASSERT_EQ(hipSetDevice(_originalDevice), hipSuccess);
        const auto stream = createStream();
        const auto handle = createHandle(stream.get());
        status = hipdnnBackendExecute(handle.get(), _plan, _variantPack);
        EXPECT_EQ(hipStreamSynchronize(stream.get()), hipSuccess);
    });
    worker.join();

    EXPECT_EQ(status, HIPDNN_STATUS_SUCCESS);
}

TEST_F(IntegrationGpuBackendExecuteDevice, PlanRejectsHandleOnAnotherDevice)
{
    if(_deviceCount < 2)
    {
        GTEST_SKIP() << "Needs at least two devices; found " << _deviceCount << ".";
    }

    ASSERT_EQ(hipSetDevice(0), hipSuccess);
    ASSERT_NO_FATAL_FAILURE(createExecutablePlan());

    ASSERT_EQ(hipSetDevice(1), hipSuccess);
    const auto deviceOneStream = createStream();
    ASSERT_EQ(hipSetDevice(0), hipSuccess);
    const auto deviceOneHandle = createHandle(deviceOneStream.get());

    EXPECT_EQ(hipdnnBackendExecute(deviceOneHandle.get(), _plan, _variantPack),
              HIPDNN_STATUS_BAD_PARAM);
    const auto message = takeLastErrorString();
    EXPECT_NE(message.find("created for device 0"), std::string::npos) << message;
    EXPECT_NE(message.find("stream is on device 1"), std::string::npos) << message;

    // The null stream follows the calling thread's current device.
    ASSERT_EQ(hipSetDevice(1), hipSuccess);
    EXPECT_EQ(hipdnnBackendExecute(_handle, _plan, _variantPack), HIPDNN_STATUS_BAD_PARAM);
    ASSERT_EQ(hipSetDevice(0), hipSuccess);
    EXPECT_EQ(hipdnnBackendExecute(_handle, _plan, _variantPack), HIPDNN_STATUS_SUCCESS);
}
