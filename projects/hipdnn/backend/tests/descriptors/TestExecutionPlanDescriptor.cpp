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

#include <gtest/gtest.h>

#include <array>
#include <memory>

using namespace hipdnn_backend;
using namespace plugin;
using namespace hipdnn_sdk::test_utilities;
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

protected:
    std::unique_ptr<HipdnnBackendDescriptor> _planWrapper = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _mockGraphWrapper = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _mockEngineWrapper = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _mockEngineConfigWrapper = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _mockEngineConfigBadTypeWrapper = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _mockWrongTypeWrapper = nullptr;
    std::unique_ptr<MockHandle> _mockHandle = nullptr;
    std::shared_ptr<MockEnginePluginResourceManager> _mockEnginePluginResourceManager = nullptr;

    void SetUp() override
    {
        _planWrapper = createDescriptor<ExecutionPlanDescriptor>();
        _mockGraphWrapper = hipdnn_sdk::test_utilities::createDescriptor<MockGraphDescriptor>();
        _mockEngineWrapper = createDescriptor<MockEngineDescriptor>();
        _mockEngineConfigWrapper = createDescriptor<MockEngineConfigDescriptor>();
        _mockEngineConfigBadTypeWrapper = createDescriptor<MockEngineConfigDescriptor>();
        _mockWrongTypeWrapper = createDescriptor<MockEngineDescriptor>();
        _mockHandle = std::make_unique<MockHandle>();
        _mockEnginePluginResourceManager = std::make_shared<MockEnginePluginResourceManager>();
    }

    void TearDown() override {}

private:
    static constexpr int64_t ENGINE_ID = 0;
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
    uint64_t dummyWorkspaceSize;

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

    ASSERT_THROW_HIPDNN_STATUS(
        plan->setAttribute(HIPDNN_ATTR_EXECUTION_PLAN_HANDLE, HIPDNN_TYPE_INT64, 1, &handle),
        HIPDNN_STATUS_BAD_PARAM);

    ASSERT_THROW_HIPDNN_STATUS(
        plan->setAttribute(HIPDNN_ATTR_EXECUTION_PLAN_HANDLE, HIPDNN_TYPE_HANDLE, 2, &handle),
        HIPDNN_STATUS_BAD_PARAM);

    ASSERT_THROW_HIPDNN_STATUS(
        plan->setAttribute(HIPDNN_ATTR_EXECUTION_PLAN_HANDLE, HIPDNN_TYPE_HANDLE, 1, nullptr),
        HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);

    ASSERT_THROW_HIPDNN_STATUS(
        plan->setAttribute(HIPDNN_ATTR_EXECUTION_PLAN_HANDLE, HIPDNN_TYPE_HANDLE, 1, &handle),
        HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);

    handle = reinterpret_cast<hipdnnHandle_t>(0x12345678);
    plan->setAttribute(HIPDNN_ATTR_EXECUTION_PLAN_HANDLE, HIPDNN_TYPE_HANDLE, 1, &handle);
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
    uint64_t dummyWorkspaceSize;

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
                                       returnedEngineConfig.getPtr()));

    ASSERT_EQ(count, 1);
    ASSERT_EQ(*returnedEngineConfig.get(), *(_mockEngineConfigWrapper.get()));

    ASSERT_NO_THROW(plan->getAttribute(HIPDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       nullptr,
                                       nullCountEngineConfig.getPtr()));

    ASSERT_EQ(*nullCountEngineConfig.get(), *(_mockEngineConfigWrapper.get()));
}

TEST_F(TestExecutionPlanDescriptor, GetEngineConfigErrors)
{
    auto plan = getExecutionPlanDescriptor();
    hipdnnBackendDescriptor_t returnedEngineConfig = nullptr;
    int64_t count = 0;
    void* dummy = &returnedEngineConfig;

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
