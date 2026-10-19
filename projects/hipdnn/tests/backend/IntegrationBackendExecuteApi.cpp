// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "HipdnnStatus.h"
#include "TestMacros.hpp"
#include "TestUtil.hpp"
#include "hipdnn_backend.h"
#include <test_plugins/TestPluginConstants.hpp>

#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>
#include <hipdnn_test_sdk/utilities/LogRecorder.hpp>

#include <gtest/gtest.h>

#include <regex>
#include <string>
#include <vector>

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

    hipdnnBackendDescriptor_t otherPlan = nullptr;
    ASSERT_EQ(hipdnnBackendCreateDescriptor(HIPDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR, &otherPlan),
              HIPDNN_STATUS_SUCCESS);
    test_util::populateTestExecutionPlan(
        &otherPlan, &_engineConfig, &_engine, &_graphDescriptor, _handle, GIDX, true);

    hipdnnSeverity_t originalLevel = HIPDNN_SEV_OFF;
    ASSERT_EQ(hipdnnBackendGetGlobalLogLevel_ext(&originalLevel), HIPDNN_STATUS_SUCCESS);
    ASSERT_EQ(hipdnnSetUserLogCallback_ext(IsolatedLogRecorder::getIsolatedUserRecordingCallback(),
                                           HIPDNN_SEV_INFO,
                                           HIPDNN_LOG_CALLBACK_SYNC,
                                           this),
              HIPDNN_STATUS_SUCCESS);
    ASSERT_EQ(hipdnnBackendSetGlobalLogLevel_ext(HIPDNN_SEV_INFO), HIPDNN_STATUS_SUCCESS);

    std::vector<std::string> executeLogs;
    {
        auto recorder = IsolatedLogRecorder::withOverrideLevel(HIPDNN_SEV_INFO);

        EXPECT_EQ(hipdnnBackendExecute(_handle, _plan, _variantPack), HIPDNN_STATUS_SUCCESS);
        EXPECT_EQ(hipdnnBackendExecute(_handle, _plan, _variantPack), HIPDNN_STATUS_SUCCESS);
        EXPECT_EQ(hipdnnBackendExecute(_handle, otherPlan, _variantPack), HIPDNN_STATUS_SUCCESS);
        EXPECT_EQ(hipdnnBackendExecute(_handle, _plan, _variantPack), HIPDNN_STATUS_SUCCESS);

        for(const auto& log : recorder.getRecordedLogs())
        {
            if(log.message.find("API success: [hipdnnBackendExecute]") != std::string::npos)
            {
                executeLogs.push_back(log.message);
            }
        }
    }

    EXPECT_EQ(hipdnnSetUserLogCallback_ext(IsolatedLogRecorder::getIsolatedUserRecordingCallback(),
                                           HIPDNN_SEV_OFF,
                                           HIPDNN_LOG_CALLBACK_SYNC,
                                           this),
              HIPDNN_STATUS_SUCCESS);
    EXPECT_EQ(hipdnnBackendSetGlobalLogLevel_ext(originalLevel), HIPDNN_STATUS_SUCCESS);
    destroyTestDescriptor(otherPlan);

    ASSERT_EQ(executeLogs.size(), 4u);

    // Each log names the plan by planId and the source graph by graphId.
    const std::regex pattern(R"(planId=(\d+), graphId=([0-9a-f-]{36}),.*executionCount=(\d+))");
    std::vector<std::smatch> matches(executeLogs.size());
    for(size_t i = 0; i < executeLogs.size(); ++i)
    {
        ASSERT_TRUE(std::regex_search(executeLogs[i], matches[i], pattern)) << executeLogs[i];
    }

    const auto planId = matches[0][1].str();
    const auto otherPlanId = matches[2][1].str();
    EXPECT_NE(planId, otherPlanId);
    EXPECT_EQ(matches[1][1].str(), planId);
    EXPECT_EQ(matches[3][1].str(), planId);

    // Both plans come from the same graph.
    for(const auto& match : matches)
    {
        EXPECT_EQ(match[2].str(), matches[0][2].str());
    }

    // Counts are per plan: the other plan starts at 1 and does not advance this plan's count.
    EXPECT_EQ(matches[0][3].str(), "1");
    EXPECT_EQ(matches[1][3].str(), "2");
    EXPECT_EQ(matches[2][3].str(), "1");
    EXPECT_EQ(matches[3][3].str(), "3");
}
