// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "DescriptorTestUtils.hpp"
#include "HipdnnException.hpp"
#include "TestMacros.hpp"
#include "descriptors/EngineConfigDescriptor.hpp"
#include "descriptors/ExecutionPlanDescriptor.hpp"
#include "descriptors/ScopedDescriptor.hpp"
#include "hipdnn_backend.h"
#include "mocks/MockDescriptor.hpp"
#include "mocks/MockEnginePluginResourceManager.hpp"
#include "mocks/MockHandle.hpp"

#include <flatbuffers/flatbuffer_builder.h>
#include <gtest/gtest.h>
#include <hipdnn_flatbuffers_sdk/data_objects/execution_plan_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>

#include <array>
#include <memory>

using namespace hipdnn_backend;
using namespace plugin;
using namespace hipdnn_backend::test_utilities;
using namespace ::testing;

using ::testing::Return;

class TestExecutionPlanDescriptor : public ::testing::Test
{
public:
    std::shared_ptr<ExecutionPlanDescriptor> getExecutionPlanDescriptor() const
    {
        return _planWrapper->asDescriptor<ExecutionPlanDescriptor>();
    }

    std::shared_ptr<MockGraphDescriptor> getMockGraph() const
    {
        return MockDescriptorUtility::asDescriptorUnsafe<MockGraphDescriptor>(
            _mockGraphWrapper.get());
    }

    std::shared_ptr<MockEngineDescriptor> getMockEngine() const
    {
        return MockDescriptorUtility::asDescriptorUnsafe<MockEngineDescriptor>(
            _mockEngineWrapper.get());
    }

    std::shared_ptr<MockEngineConfigDescriptor> getMockEngineConfig() const
    {
        return MockDescriptorUtility::asDescriptorUnsafe<MockEngineConfigDescriptor>(
            _mockEngineConfigWrapper.get());
    }

    std::shared_ptr<MockEngineConfigDescriptor> getMockEngineConfigBadType() const
    {
        return MockDescriptorUtility::asDescriptorUnsafe<MockEngineConfigDescriptor>(
            _mockEngineConfigBadTypeWrapper.get());
    }

    static hipdnnEnginePluginExecutionContext_t getExecutionContext()
    {
        return reinterpret_cast<hipdnnEnginePluginExecutionContext_t>(0xFFFFFFFF);
    }

    void setHandle()
    {
        EXPECT_CALL(*_mockEnginePluginResourceManager, createExecutionContext(_, _, _))
            .WillOnce(Return(getExecutionContext()));
        EXPECT_CALL(*_mockEnginePluginResourceManager, destroyExecutionContext(_, _));
        EXPECT_CALL(*_mockEnginePluginResourceManager, getWorkspaceSize(_, _))
            .WillOnce(Return(1024));

        EXPECT_CALL(*_mockHandle, getPluginResourceManager())
            .WillOnce(Return(_mockEnginePluginResourceManager));
        getExecutionPlanDescriptor()->setAttribute(
            HIPDNN_ATTR_EXECUTION_PLAN_HANDLE, HIPDNN_TYPE_HANDLE, 1, &_mockHandle);
    }

    void setEngineConfig()
    {
        EXPECT_CALL(*getMockGraph(), getHandle()).WillRepeatedly(Return(_mockHandle.get()));
        EXPECT_CALL(*getMockGraph(), getSerializedGraph()).WillRepeatedly(Invoke([this]() {
            return hipdnnPluginConstData_t{_serializedGraph.data(), _serializedGraph.size()};
        }));
        EXPECT_CALL(*getMockEngine(), getEngineId()).WillOnce(Return(ENGINE_ID));
        EXPECT_CALL(*getMockEngine(), getGraph()).WillOnce(Return(getMockGraph()));

        EXPECT_CALL(*getMockEngineConfig(), isFinalized()).WillOnce(Return(true));
        EXPECT_CALL(*getMockEngineConfig(), getEngine()).WillOnce(Return(getMockEngine()));
        EXPECT_CALL(*getMockEngineConfig(), getSerializedEngineConfig()).WillOnce(Invoke([]() {
            return hipdnnPluginConstData_t{nullptr, 0};
        }));

        getExecutionPlanDescriptor()->setAttribute(HIPDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG,
                                                   HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                   1,
                                                   &_mockEngineConfigWrapper);
    }

    void makeExecutionPlanFinalized()
    {
        setHandle();
        setEngineConfig();
        ASSERT_NO_THROW(getExecutionPlanDescriptor()->finalize());
    }

    flatbuffers::DetachedBuffer makeSerializedPlan(uint32_t version = 1,
                                                   int64_t workspaceSize = 1024,
                                                   bool includeTensorUids = true,
                                                   bool includePluginPayload = true,
                                                   bool emptyTensorUids = false,
                                                   bool emptyPluginPayload = false) const
    {
        flatbuffers::FlatBufferBuilder builder;
        flatbuffers::Offset<flatbuffers::Vector<int64_t>> tensorUids;
        flatbuffers::Offset<flatbuffers::Vector<uint8_t>> pluginPayload;

        if(includeTensorUids)
        {
            tensorUids = emptyTensorUids ? builder.CreateVector(std::vector<int64_t>{})
                                         : builder.CreateVector(_tensorUids);
        }
        if(includePluginPayload)
        {
            auto pluginPayloadBytes
                = emptyPluginPayload ? std::vector<uint8_t>{} : std::vector<uint8_t>{4, 5, 6};
            pluginPayload = builder.CreateVector(pluginPayloadBytes);
        }

        auto plan = hipdnn_flatbuffers_sdk::data_objects::CreateSerializedExecutionPlan(
            builder, version, ENGINE_ID, workspaceSize, tensorUids, pluginPayload);
        builder.Finish(plan);
        return builder.Release();
    }

protected:
    static constexpr int64_t ENGINE_ID = 0;

    std::unique_ptr<HipdnnBackendDescriptor> _planWrapper = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _mockGraphWrapper = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _mockEngineWrapper = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _mockEngineConfigWrapper = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _mockEngineConfigBadTypeWrapper = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _mockWrongTypeWrapper = nullptr;
    std::unique_ptr<MockHandle> _mockHandle = nullptr;
    std::shared_ptr<MockEnginePluginResourceManager> _mockEnginePluginResourceManager = nullptr;
    flatbuffers::DetachedBuffer _serializedGraph;
    std::vector<int64_t> _tensorUids{11, 22, 33};

    void SetUp() override
    {
        flatbuffers::FlatBufferBuilder builder;
        hipdnn_flatbuffers_sdk::data_objects::GraphT graph;
        for(auto uid : _tensorUids)
        {
            auto tensor
                = std::make_unique<hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT>();
            tensor->uid = uid;
            graph.tensors.push_back(std::move(tensor));
        }
        builder.Finish(hipdnn_flatbuffers_sdk::data_objects::Graph::Pack(builder, &graph));
        _serializedGraph = builder.Release();

        _planWrapper = createDescriptor<ExecutionPlanDescriptor>();
        _mockGraphWrapper = createDescriptor<MockGraphDescriptor>();
        _mockEngineWrapper = createDescriptor<MockEngineDescriptor>();
        _mockEngineConfigWrapper = createDescriptor<MockEngineConfigDescriptor>();
        _mockEngineConfigBadTypeWrapper = createDescriptor<MockEngineConfigDescriptor>();
        _mockWrongTypeWrapper = createDescriptor<MockEngineDescriptor>();
        _mockHandle = std::make_unique<MockHandle>();
        _mockEnginePluginResourceManager = std::make_shared<MockEnginePluginResourceManager>();
    }

    void TearDown() override {}
};

TEST_F(TestExecutionPlanDescriptor, CreateExecutionPlanDescriptor)
{
    auto plan = getExecutionPlanDescriptor();
    ASSERT_NE(plan, nullptr);
    ASSERT_FALSE(plan->isFinalized());
    ASSERT_EQ(plan->getType(), HIPDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR);
}

TEST_F(TestExecutionPlanDescriptor, SetAttrWhenNotFinalized)
{
    auto plan = getExecutionPlanDescriptor();
    uint64_t dummyWorkspaceSize = 0;

    ASSERT_THROW_HIPDNN_STATUS(
        plan->setAttribute(
            HIPDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE, HIPDNN_TYPE_INT64, 1, &dummyWorkspaceSize),
        HIPDNN_STATUS_NOT_SUPPORTED);

    makeExecutionPlanFinalized();

    ASSERT_THROW_HIPDNN_STATUS(
        plan->setAttribute(
            HIPDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE, HIPDNN_TYPE_INT64, 1, &dummyWorkspaceSize),
        HIPDNN_STATUS_NOT_INITIALIZED);
}

TEST_F(TestExecutionPlanDescriptor, SetHandle)
{
    auto plan = getExecutionPlanDescriptor();
    hipdnnHandle_t handle = nullptr;

    ASSERT_THROW_HIPDNN_STATUS(plan->setAttribute(HIPDNN_ATTR_EXECUTION_PLAN_HANDLE,
                                                  HIPDNN_TYPE_INT64,
                                                  1,
                                                  static_cast<const void*>(&handle)),
                               HIPDNN_STATUS_BAD_PARAM);

    ASSERT_THROW_HIPDNN_STATUS(plan->setAttribute(HIPDNN_ATTR_EXECUTION_PLAN_HANDLE,
                                                  HIPDNN_TYPE_HANDLE,
                                                  2,
                                                  static_cast<const void*>(&handle)),
                               HIPDNN_STATUS_BAD_PARAM);

    ASSERT_THROW_HIPDNN_STATUS(
        plan->setAttribute(HIPDNN_ATTR_EXECUTION_PLAN_HANDLE, HIPDNN_TYPE_HANDLE, 1, nullptr),
        HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);

    ASSERT_THROW_HIPDNN_STATUS(plan->setAttribute(HIPDNN_ATTR_EXECUTION_PLAN_HANDLE,
                                                  HIPDNN_TYPE_HANDLE,
                                                  1,
                                                  static_cast<const void*>(&handle)),
                               HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);

    handle = reinterpret_cast<hipdnnHandle_t>(0x12345678);
    plan->setAttribute(HIPDNN_ATTR_EXECUTION_PLAN_HANDLE,
                       HIPDNN_TYPE_HANDLE,
                       1,
                       static_cast<const void*>(&handle));
}

TEST_F(TestExecutionPlanDescriptor, SetEngineConfig)
{
    auto plan = getExecutionPlanDescriptor();
    auto mockEngineConfig = getMockEngineConfig();

    EXPECT_CALL(*getMockEngineConfigBadType(), isFinalized()).Times(1);
    EXPECT_CALL(*mockEngineConfig, isFinalized()).WillOnce(Return(false)).WillOnce(Return(true));

    // isFinalized()->false
    ASSERT_THROW_HIPDNN_STATUS(plan->setAttribute(HIPDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG,
                                                  HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                  1,
                                                  &_mockEngineConfigWrapper),
                               HIPDNN_STATUS_BAD_PARAM_NOT_FINALIZED);

    // isFinalized()->true
    plan->setAttribute(HIPDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &_mockEngineConfigWrapper);

    ASSERT_THROW_HIPDNN_STATUS(plan->setAttribute(HIPDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG,
                                                  HIPDNN_TYPE_HANDLE,
                                                  1,
                                                  &_mockEngineConfigWrapper),
                               HIPDNN_STATUS_BAD_PARAM);

    ASSERT_THROW_HIPDNN_STATUS(plan->setAttribute(HIPDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG,
                                                  HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                  2,
                                                  &_mockEngineConfigWrapper),
                               HIPDNN_STATUS_BAD_PARAM);

    ASSERT_THROW_HIPDNN_STATUS(
        plan->setAttribute(
            HIPDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, nullptr),
        HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);

    hipdnnBackendDescriptor_t engineConfig = nullptr;
    ASSERT_THROW_HIPDNN_STATUS(plan->setAttribute(HIPDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG,
                                                  HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                  1,
                                                  &engineConfig),
                               HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);

    ASSERT_THROW_HIPDNN_STATUS(plan->setAttribute(HIPDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG,
                                                  HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                  1,
                                                  &_mockEngineConfigBadTypeWrapper),
                               HIPDNN_STATUS_BAD_PARAM_NOT_FINALIZED);

    ASSERT_THROW_HIPDNN_STATUS(plan->setAttribute(HIPDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG,
                                                  HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                  1,
                                                  &_mockWrongTypeWrapper),
                               HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestExecutionPlanDescriptor, Finalize)
{
    auto plan = getExecutionPlanDescriptor();
    ASSERT_THROW_HIPDNN_STATUS(plan->finalize(), HIPDNN_STATUS_BAD_PARAM);

    setHandle();
    setEngineConfig();

    ASSERT_NO_THROW(plan->finalize());

    ASSERT_THROW(plan->finalize(), hipdnn_backend::HipdnnException);
}

TEST_F(TestExecutionPlanDescriptor, GetAttrWhenNotFinalized)
{
    auto plan = getExecutionPlanDescriptor();
    uint64_t dummyWorkspaceSize = 0;

    ASSERT_THROW_HIPDNN_STATUS(plan->getAttribute(HIPDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE,
                                                  HIPDNN_TYPE_INT64,
                                                  1,
                                                  nullptr,
                                                  &dummyWorkspaceSize),
                               HIPDNN_STATUS_NOT_INITIALIZED);
}

TEST_F(TestExecutionPlanDescriptor, GetWorkspaceSize)
{
    auto plan = getExecutionPlanDescriptor();
    auto mockEngineConfig = getMockEngineConfig();
    int64_t workspaceSize = 0;

    makeExecutionPlanFinalized();
    plan->getAttribute(
        HIPDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE, HIPDNN_TYPE_INT64, 1, nullptr, &workspaceSize);
    ASSERT_EQ(workspaceSize, 1024);
}

TEST_F(TestExecutionPlanDescriptor, GetTensorUids)
{
    auto plan = getExecutionPlanDescriptor();

    makeExecutionPlanFinalized();

    int64_t count = 0;
    ASSERT_NO_THROW(plan->getAttribute(
        HIPDNN_ATTR_EXECUTION_PLAN_TENSOR_UIDS_EXT, HIPDNN_TYPE_INT64, 0, &count, nullptr));
    ASSERT_EQ(count, static_cast<int64_t>(_tensorUids.size()));

    std::vector<int64_t> tensorUids(static_cast<size_t>(count));
    ASSERT_NO_THROW(plan->getAttribute(HIPDNN_ATTR_EXECUTION_PLAN_TENSOR_UIDS_EXT,
                                       HIPDNN_TYPE_INT64,
                                       count,
                                       &count,
                                       tensorUids.data()));
    ASSERT_EQ(tensorUids, _tensorUids);
    ASSERT_EQ(plan->getTensorUids(), _tensorUids);
}

TEST_F(TestExecutionPlanDescriptor, DeserializeRejectsInvalidFlatBuffer)
{
    auto plan = getExecutionPlanDescriptor();
    std::array<uint8_t, 8> invalidPlan{0, 1, 2, 3, 4, 5, 6, 7};

    ASSERT_THROW_HIPDNN_STATUS(plan->deserializeBackendPlan(_mockEnginePluginResourceManager,
                                                            invalidPlan.data(),
                                                            invalidPlan.size()),
                               HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestExecutionPlanDescriptor, DeserializeRejectsUnsupportedVersion)
{
    auto plan = getExecutionPlanDescriptor();
    auto serializedPlan = makeSerializedPlan(2);

    ASSERT_THROW_HIPDNN_STATUS(plan->deserializeBackendPlan(_mockEnginePluginResourceManager,
                                                            serializedPlan.data(),
                                                            serializedPlan.size()),
                               HIPDNN_STATUS_NOT_SUPPORTED);
}

TEST_F(TestExecutionPlanDescriptor, DeserializeRejectsMissingTensorUids)
{
    auto plan = getExecutionPlanDescriptor();
    auto serializedPlan = makeSerializedPlan(1, 1024, false);

    ASSERT_THROW_HIPDNN_STATUS(plan->deserializeBackendPlan(_mockEnginePluginResourceManager,
                                                            serializedPlan.data(),
                                                            serializedPlan.size()),
                               HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestExecutionPlanDescriptor, DeserializeRejectsEmptyTensorUids)
{
    auto plan = getExecutionPlanDescriptor();
    auto serializedPlan = makeSerializedPlan(1, 1024, true, true, true);

    ASSERT_THROW_HIPDNN_STATUS(plan->deserializeBackendPlan(_mockEnginePluginResourceManager,
                                                            serializedPlan.data(),
                                                            serializedPlan.size()),
                               HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestExecutionPlanDescriptor, DeserializeRejectsMissingPluginPayload)
{
    auto plan = getExecutionPlanDescriptor();
    auto serializedPlan = makeSerializedPlan(1, 1024, true, false);

    ASSERT_THROW_HIPDNN_STATUS(plan->deserializeBackendPlan(_mockEnginePluginResourceManager,
                                                            serializedPlan.data(),
                                                            serializedPlan.size()),
                               HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestExecutionPlanDescriptor, DeserializeRejectsEmptyPluginPayload)
{
    auto plan = getExecutionPlanDescriptor();
    auto serializedPlan = makeSerializedPlan(1, 1024, true, true, false, true);

    ASSERT_THROW_HIPDNN_STATUS(plan->deserializeBackendPlan(_mockEnginePluginResourceManager,
                                                            serializedPlan.data(),
                                                            serializedPlan.size()),
                               HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestExecutionPlanDescriptor, DeserializeRejectsNegativeWorkspaceSize)
{
    auto plan = getExecutionPlanDescriptor();
    auto serializedPlan = makeSerializedPlan(1, -1);

    ASSERT_THROW_HIPDNN_STATUS(plan->deserializeBackendPlan(_mockEnginePluginResourceManager,
                                                            serializedPlan.data(),
                                                            serializedPlan.size()),
                               HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestExecutionPlanDescriptor, DeserializeRestoresSerializedExecutionPlan)
{
    auto plan = getExecutionPlanDescriptor();
    auto serializedPlan = makeSerializedPlan();

    EXPECT_CALL(*_mockEnginePluginResourceManager,
                createExecutionContextFromSerialized(ENGINE_ID, _))
        .WillOnce([](int64_t, const hipdnnPluginConstData_t* serializedContext) {
            EXPECT_EQ(serializedContext->size, 3);
            return getExecutionContext();
        });
    EXPECT_CALL(*_mockEnginePluginResourceManager, destroyExecutionContext(_, _));

    ASSERT_NO_THROW(plan->deserializeBackendPlan(
        _mockEnginePluginResourceManager, serializedPlan.data(), serializedPlan.size()));

    ASSERT_TRUE(plan->isFinalized());
    ASSERT_EQ(plan->getTensorUids(), _tensorUids);
    ASSERT_EQ(plan->getExecutionContext(), getExecutionContext());

    int64_t workspaceSize = 0;
    ASSERT_NO_THROW(plan->getAttribute(
        HIPDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE, HIPDNN_TYPE_INT64, 1, nullptr, &workspaceSize));
    ASSERT_EQ(workspaceSize, 1024);
}

TEST_F(TestExecutionPlanDescriptor, SerializeRejectsUnfinalizedPlan)
{
    auto plan = getExecutionPlanDescriptor();
    size_t planByteSize = 0;

    ASSERT_THROW_HIPDNN_STATUS(plan->serializeBackendPlan(0, &planByteSize, nullptr),
                               HIPDNN_STATUS_NOT_INITIALIZED);
}

TEST_F(TestExecutionPlanDescriptor, SerializeRejectsNullSizePointer)
{
    auto plan = getExecutionPlanDescriptor();
    makeExecutionPlanFinalized();

    ASSERT_THROW_HIPDNN_STATUS(plan->serializeBackendPlan(0, nullptr, nullptr),
                               HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

TEST_F(TestExecutionPlanDescriptor, SerializeRoundTripsFlatBufferEnvelope)
{
    auto plan = getExecutionPlanDescriptor();
    auto serializedPlan = makeSerializedPlan();
    const std::vector<uint8_t> pluginPayload{9, 8, 7, 6};

    EXPECT_CALL(*_mockEnginePluginResourceManager,
                createExecutionContextFromSerialized(ENGINE_ID, _))
        .WillOnce(Return(getExecutionContext()));
    EXPECT_CALL(*_mockEnginePluginResourceManager, destroyExecutionContext(_, _));
    ASSERT_NO_THROW(plan->deserializeBackendPlan(
        _mockEnginePluginResourceManager, serializedPlan.data(), serializedPlan.size()));

    EXPECT_CALL(*_mockEnginePluginResourceManager,
                serializeExecutionContext(ENGINE_ID, getExecutionContext(), _))
        .Times(2)
        .WillRepeatedly([&pluginPayload](int64_t,
                                         hipdnnEnginePluginExecutionContext_t,
                                         std::vector<uint8_t>& serializedContext) {
            serializedContext = pluginPayload;
        });

    size_t planByteSize = 0;
    ASSERT_NO_THROW(plan->serializeBackendPlan(0, &planByteSize, nullptr));
    ASSERT_GT(planByteSize, 0);

    std::vector<uint8_t> serializedOutput(planByteSize);
    ASSERT_NO_THROW(
        plan->serializeBackendPlan(planByteSize, &planByteSize, serializedOutput.data()));
    serializedOutput.resize(planByteSize);

    flatbuffers::Verifier verifier(serializedOutput.data(), serializedOutput.size());
    ASSERT_TRUE(
        hipdnn_flatbuffers_sdk::data_objects::VerifySerializedExecutionPlanBuffer(verifier));

    auto executionPlan
        = hipdnn_flatbuffers_sdk::data_objects::GetSerializedExecutionPlan(serializedOutput.data());
    ASSERT_EQ(executionPlan->version(), 1);
    ASSERT_EQ(executionPlan->engine_id(), ENGINE_ID);
    ASSERT_EQ(executionPlan->workspace_size(), 1024);
    ASSERT_EQ(std::vector<int64_t>(executionPlan->tensor_uids()->begin(),
                                   executionPlan->tensor_uids()->end()),
              _tensorUids);
    ASSERT_EQ(std::vector<uint8_t>(executionPlan->plugin_payload()->begin(),
                                   executionPlan->plugin_payload()->end()),
              pluginPayload);
}

TEST_F(TestExecutionPlanDescriptor, SerializeRejectsInsufficientBuffer)
{
    auto plan = getExecutionPlanDescriptor();
    auto serializedPlan = makeSerializedPlan();
    const std::vector<uint8_t> pluginPayload{9, 8, 7, 6};

    EXPECT_CALL(*_mockEnginePluginResourceManager,
                createExecutionContextFromSerialized(ENGINE_ID, _))
        .WillOnce(Return(getExecutionContext()));
    EXPECT_CALL(*_mockEnginePluginResourceManager, destroyExecutionContext(_, _));
    ASSERT_NO_THROW(plan->deserializeBackendPlan(
        _mockEnginePluginResourceManager, serializedPlan.data(), serializedPlan.size()));

    EXPECT_CALL(*_mockEnginePluginResourceManager,
                serializeExecutionContext(ENGINE_ID, getExecutionContext(), _))
        .WillOnce([&pluginPayload](int64_t,
                                   hipdnnEnginePluginExecutionContext_t,
                                   std::vector<uint8_t>& serializedContext) {
            serializedContext = pluginPayload;
        });

    size_t planByteSize = 0;
    std::array<uint8_t, 1> serializedOutput{0};
    ASSERT_THROW_HIPDNN_STATUS(
        plan->serializeBackendPlan(serializedOutput.size(), &planByteSize, serializedOutput.data()),
        HIPDNN_STATUS_BAD_PARAM_SIZE_INSUFFICIENT);
}

TEST_F(TestExecutionPlanDescriptor, GetEngineConfig)
{
    auto plan = getExecutionPlanDescriptor();

    ScopedDescriptor returnedEngineConfig;
    ScopedDescriptor nullCountEngineConfig;
    int64_t count = 0;

    makeExecutionPlanFinalized();

    ASSERT_NO_THROW(plan->getAttribute(HIPDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &count,
                                       static_cast<void*>(returnedEngineConfig.getPtr())));

    ASSERT_EQ(count, 1);
    ASSERT_EQ(*returnedEngineConfig.get(), *(_mockEngineConfigWrapper.get()));

    ASSERT_NO_THROW(plan->getAttribute(HIPDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       nullptr,
                                       static_cast<void*>(nullCountEngineConfig.getPtr())));

    ASSERT_EQ(*nullCountEngineConfig.get(), *(_mockEngineConfigWrapper.get()));
}

TEST_F(TestExecutionPlanDescriptor, GetEngineConfigErrors)
{
    auto plan = getExecutionPlanDescriptor();
    hipdnnBackendDescriptor_t returnedEngineConfig = nullptr;
    int64_t count = 0;
    void* dummy = static_cast<void*>(&returnedEngineConfig);

    makeExecutionPlanFinalized();

    ASSERT_THROW_HIPDNN_STATUS(
        plan->getAttribute(
            HIPDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG, HIPDNN_TYPE_INT64, 1, &count, &dummy),
        HIPDNN_STATUS_BAD_PARAM);

    ASSERT_THROW_HIPDNN_STATUS(plan->getAttribute(HIPDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG,
                                                  HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                  0, // Too small (needs to be at least 1)
                                                  &count,
                                                  &dummy),
                               HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestExecutionPlanDescriptor, GetUnsupportedAttr)
{
    auto plan = getExecutionPlanDescriptor();
    int64_t count = 0;
    std::array<char, 256> dummyBuffer{};

    makeExecutionPlanFinalized();

    ASSERT_THROW_HIPDNN_STATUS(plan->getAttribute(HIPDNN_ATTR_EXECUTION_PLAN_KERNEL_CACHE,
                                                  HIPDNN_TYPE_INT64,
                                                  1,
                                                  &count,
                                                  dummyBuffer.data()),
                               HIPDNN_STATUS_NOT_SUPPORTED);
}

TEST_F(TestExecutionPlanDescriptor, GetEngineConfigThrowsIfNotFinalized)
{
    auto plan = getExecutionPlanDescriptor();
    ASSERT_THROW_HIPDNN_STATUS(plan->getEngineConfig(), HIPDNN_STATUS_INTERNAL_ERROR);
}

TEST_F(TestExecutionPlanDescriptor, GetEngineConfigReturnsPointerIfFinalized)
{
    auto plan = getExecutionPlanDescriptor();
    makeExecutionPlanFinalized();
    auto engineConfigPtr = plan->getEngineConfig();
    ASSERT_NE(engineConfigPtr, nullptr);
    ASSERT_EQ(static_cast<const IBackendDescriptor*>(engineConfigPtr.get()),
              static_cast<const IBackendDescriptor*>(getMockEngineConfig().get()));
}

TEST_F(TestExecutionPlanDescriptor, GetExecutionContext)
{
    auto plan = getExecutionPlanDescriptor();
    makeExecutionPlanFinalized();
    auto context = plan->getExecutionContext();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(context, getExecutionContext());
}
