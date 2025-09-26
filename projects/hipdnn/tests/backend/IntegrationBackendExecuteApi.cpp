// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "HipdnnStatus.h"
#include "TestMacros.hpp"
#include "TestUtil.hpp"
#include "hipdnn_backend.h"
#include <test_plugins/TestPluginConstants.hpp>

#include <hipdnn_sdk/test_utilities/FlatbufferGraphTestUtils.hpp>

#include <gtest/gtest.h>

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
    auto batchnormBuilder = test_util::createAndPopulateBatchnormNode();
    auto serializedGraph = batchnormBuilder.Release();

    std::unordered_map<int64_t, std::string> uidToNameMap;
    std::unordered_map<std::string, int64_t> nameToUidMap;
    std::unordered_map<int64_t, std::vector<int64_t>> uidToDimsMap;

    test_util::extractTensorInfoFromGraph(
        serializedGraph, uidToNameMap, nameToUidMap, uidToDimsMap);

    test_util::createAndInitializeBackendDescriptor(&_graphDescriptor, serializedGraph, _handle);
    test_util::createTestEngine(&_engine, &_graphDescriptor, _handle, GIDX, true);
    test_util::createTestEngineConfig(
        &_engineConfig, &_engine, &_graphDescriptor, _handle, GIDX, true);

    EXPECT_EQ(hipdnnBackendCreateDescriptor(HIPDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR, &_plan),
              HIPDNN_STATUS_SUCCESS);

    test_util::populateTestExecutionPlan(
        &_plan, &_engineConfig, &_engine, &_graphDescriptor, _handle, GIDX, true);

    std::unordered_map<int64_t, void*> dataPtrMappings;

    for(const auto& [uid, dims] : uidToDimsMap)
    {
        void* tensorData
            = test_util::allocateTensorMemory(dims.data(), dims.size(), HIPDNN_TYPE_FLOAT, true);

        ASSERT_NE(tensorData, nullptr)
            << "Failed to allocate memory for tensor " << uidToNameMap[uid];

        dataPtrMappings[uid] = tensorData;
    }

    ASSERT_EQ(hipdnnBackendCreateDescriptor(HIPDNN_BACKEND_VARIANT_PACK_DESCRIPTOR, &_variantPack),
              HIPDNN_STATUS_SUCCESS);

    test_util::populateVariantPackWithMappings(_variantPack, dataPtrMappings, nullptr);

    ASSERT_EQ(hipdnnBackendExecute(_handle, _plan, _variantPack), HIPDNN_STATUS_SUCCESS);

    for(const auto& [uid, dataPtr] : dataPtrMappings)
    {
        test_util::freeTensorMemory(dataPtr);
    }
}
