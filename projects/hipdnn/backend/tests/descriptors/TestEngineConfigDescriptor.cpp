// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "DescriptorTestUtils.hpp"
#include "TestMacros.hpp"
#include "descriptors/EngineConfigDescriptor.hpp"
#include "descriptors/EngineDescriptor.hpp"
#include "descriptors/GraphDescriptor.hpp"
#include "descriptors/ScopedDescriptor.hpp"
#include "hipdnn_backend.h"
#include "mocks/MockDescriptor.hpp"
#include "mocks/MockEnginePluginResourceManager.hpp"
#include "mocks/MockHandle.hpp"

#include <gtest/gtest.h>

#include <memory>

using namespace hipdnn_backend;
using namespace plugin;
using namespace ::testing;

using ::testing::Return;

class TestEngineConfigDescriptor : public ::testing::Test
{
public:
    std::shared_ptr<EngineConfigDescriptor> getEngineConfigDescriptor() const
    {
        return _engineConfigWrapper->asDescriptor<EngineConfigDescriptor>();
    }

    std::shared_ptr<MockEngineDescriptor> getMockEngine() const
    {
        return MockDescriptorUtility::asDescriptorUnsafe<MockEngineDescriptor>(
            _mockEngineWrapper.get());
    }

    std::shared_ptr<MockEngineDescriptor> getMockEngineBadType() const
    {
        return MockDescriptorUtility::asDescriptorUnsafe<MockEngineDescriptor>(
            _mockEngineBadTypeWrapper.get());
    }

    std::shared_ptr<MockGraphDescriptor> getMockGraphDescriptor() const
    {
        return MockDescriptorUtility::asDescriptorUnsafe<MockGraphDescriptor>(
            _mockGraphWrapper.get());
    }

    void setEngine() const
    {
        EXPECT_CALL(*getMockEngine(), isFinalized()).WillOnce(Return(true));
        ASSERT_NO_THROW(getEngineConfigDescriptor()->setAttribute(
            HIPDNN_ATTR_ENGINECFG_ENGINE, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_mockEngineWrapper));
    }

    void makeEngineConfigFinalized() const
    {
        // TODO: These expectations being hidden in here are dangerous.
        // It's easy to forget they exist, and if any of these functions are called prior to this
        // call it is undefined behavior. Similarly, any expectations on functions called within
        // "finalize" or "setup" made after calling this function are UB
        EXPECT_CALL(*getMockEngine(), isFinalized()).WillRepeatedly(Return(true));
        EXPECT_CALL(*getMockEngine(), getEngineId()).WillRepeatedly(Return(1));
        EXPECT_CALL(*getMockEngine(), getGraph()).WillRepeatedly(Return(getMockGraphDescriptor()));
        EXPECT_CALL(*getMockGraphDescriptor(), getHandle()).WillOnce(Return(_mockHandle.get()));
        EXPECT_CALL(*_mockHandle, getPluginResourceManager())
            .WillOnce(Return(_mockEnginePluginResourceManager));
        EXPECT_CALL(*_mockEnginePluginResourceManager, getWorkspaceSize(_, _, _))
            .WillOnce(Return(1024));

        setEngine();
        ASSERT_NO_THROW(getEngineConfigDescriptor()->finalize());
    }

protected:
    std::unique_ptr<HipdnnBackendDescriptor> _engineConfigWrapper = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _mockEngineWrapper = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _mockEngineBadTypeWrapper = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _mockWrongTypeWrapper = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _mockGraphWrapper = nullptr;
    std::unique_ptr<MockHandle> _mockHandle = nullptr;
    std::shared_ptr<MockEnginePluginResourceManager> _mockEnginePluginResourceManager = nullptr;

    void SetUp() override
    {
        _engineConfigWrapper
            = hipdnn_sdk::test_utilities::createDescriptor<EngineConfigDescriptor>();
        _mockEngineWrapper = hipdnn_sdk::test_utilities::createDescriptor<MockEngineDescriptor>();
        _mockEngineBadTypeWrapper
            = hipdnn_sdk::test_utilities::createDescriptor<MockEngineDescriptor>();
        _mockWrongTypeWrapper = hipdnn_sdk::test_utilities::createDescriptor<
            MockDescriptor<EngineConfigDescriptor>>();
        _mockGraphWrapper = hipdnn_sdk::test_utilities::createDescriptor<MockGraphDescriptor>();
        _mockHandle = std::make_unique<MockHandle>();
        _mockEnginePluginResourceManager = std::make_shared<MockEnginePluginResourceManager>();
    }
};

TEST_F(TestEngineConfigDescriptor, CreateEngineConfigDescriptor)
{
    auto engineConfig = getEngineConfigDescriptor();
    ASSERT_NE(engineConfig, nullptr);
    ASSERT_FALSE(engineConfig->isFinalized());
    ASSERT_EQ(engineConfig->getType(), HIPDNN_BACKEND_ENGINECFG_DESCRIPTOR);
}

TEST_F(TestEngineConfigDescriptor, SetEngineConfigDescriptorEngine)
{
    auto engineConfig = getEngineConfigDescriptor();

    EXPECT_CALL(*getMockEngineBadType(), isFinalized()).Times(1);
    EXPECT_CALL(*getMockEngine(), getEngineId()).Times(AnyNumber());
    EXPECT_CALL(*getMockEngine(), isFinalized()).WillOnce(Return(false)).WillOnce(Return(true));

    ASSERT_THROW_HIPDNN_STATUS(
        engineConfig->setAttribute(
            HIPDNN_ATTR_ENGINECFG_ENGINE, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_mockEngineWrapper),
        HIPDNN_STATUS_BAD_PARAM_NOT_FINALIZED);

    ASSERT_NO_THROW(engineConfig->setAttribute(
        HIPDNN_ATTR_ENGINECFG_ENGINE, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_mockEngineWrapper));

    ASSERT_THROW_HIPDNN_STATUS(
        engineConfig->setAttribute(
            HIPDNN_ATTR_ENGINECFG_ENGINE, HIPDNN_TYPE_INT64, 1, &_mockEngineWrapper),
        HIPDNN_STATUS_BAD_PARAM);

    ASSERT_THROW_HIPDNN_STATUS(
        engineConfig->setAttribute(
            HIPDNN_ATTR_ENGINECFG_ENGINE, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 2, &_mockEngineWrapper),
        HIPDNN_STATUS_BAD_PARAM);

    ASSERT_THROW_HIPDNN_STATUS(
        engineConfig->setAttribute(
            HIPDNN_ATTR_ENGINECFG_ENGINE, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, nullptr),
        HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);

    hipdnnBackendDescriptor_t engine = nullptr;
    ASSERT_THROW_HIPDNN_STATUS(
        engineConfig->setAttribute(
            HIPDNN_ATTR_ENGINECFG_ENGINE, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &engine),
        HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);

    ASSERT_THROW_HIPDNN_STATUS(engineConfig->setAttribute(HIPDNN_ATTR_ENGINECFG_ENGINE,
                                                          HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                          1,
                                                          &_mockEngineBadTypeWrapper),
                               HIPDNN_STATUS_BAD_PARAM_NOT_FINALIZED);

    ASSERT_THROW_HIPDNN_STATUS(engineConfig->setAttribute(HIPDNN_ATTR_ENGINECFG_ENGINE,
                                                          HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                          1,
                                                          &_mockWrongTypeWrapper),
                               HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestEngineConfigDescriptor, SetAttrOnFinalizedEngineConfigDescriptor)
{
    auto engineConfig = getEngineConfigDescriptor();
    makeEngineConfigFinalized();

    ASSERT_THROW_HIPDNN_STATUS(
        engineConfig->setAttribute(
            HIPDNN_ATTR_ENGINECFG_ENGINE, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_mockEngineWrapper),
        HIPDNN_STATUS_NOT_INITIALIZED);
}

TEST_F(TestEngineConfigDescriptor, FinalizeEngineConfigDescriptor)
{
    auto engineConfig = getEngineConfigDescriptor();
    ASSERT_THROW_HIPDNN_STATUS(engineConfig->finalize(), HIPDNN_STATUS_BAD_PARAM);

    makeEngineConfigFinalized();
}

TEST_F(TestEngineConfigDescriptor, GetAttrOnUnfinalizedEngineConfigDescriptor)
{
    auto engineConfig = getEngineConfigDescriptor();
    hipdnnBackendDescriptor_t dummyEngine = nullptr;

    ASSERT_THROW_HIPDNN_STATUS(
        engineConfig->getAttribute(
            HIPDNN_ATTR_ENGINECFG_ENGINE, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, nullptr, &dummyEngine),
        HIPDNN_STATUS_NOT_INITIALIZED);
}

TEST_F(TestEngineConfigDescriptor, GetEngineConfigDescriptorUnsupportedAttr)
{
    auto engineConfig = getEngineConfigDescriptor();
    hipdnnBackendDescriptor_t dummy = nullptr;

    makeEngineConfigFinalized();

    ASSERT_THROW_HIPDNN_STATUS(engineConfig->getAttribute(HIPDNN_ATTR_ENGINECFG_INTERMEDIATE_INFO,
                                                          HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                          1,
                                                          nullptr,
                                                          &dummy),
                               HIPDNN_STATUS_NOT_SUPPORTED);
}

TEST_F(TestEngineConfigDescriptor, GetEngineConfigDescriptorEngine)
{
    auto engineConfig = getEngineConfigDescriptor();
    ScopedDescriptor engine;
    ScopedDescriptor engine2;
    makeEngineConfigFinalized();

    ASSERT_THROW_HIPDNN_STATUS(
        engineConfig->getAttribute(
            HIPDNN_ATTR_ENGINECFG_ENGINE, HIPDNN_TYPE_INT64, 1, nullptr, engine.getPtr()),
        HIPDNN_STATUS_BAD_PARAM);

    ASSERT_THROW_HIPDNN_STATUS(engineConfig->getAttribute(HIPDNN_ATTR_ENGINECFG_ENGINE,
                                                          HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                          2,
                                                          nullptr,
                                                          engine.getPtr()),
                               HIPDNN_STATUS_BAD_PARAM);

    ASSERT_THROW_HIPDNN_STATUS(
        engineConfig->getAttribute(
            HIPDNN_ATTR_ENGINECFG_ENGINE, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, nullptr, nullptr),
        HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);

    ASSERT_NO_THROW(engineConfig->getAttribute(
        HIPDNN_ATTR_ENGINECFG_ENGINE, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, nullptr, engine.getPtr()));
    ASSERT_EQ(*engine.get(), *(_mockEngineWrapper.get()));

    int64_t count;
    ASSERT_NO_THROW(engineConfig->getAttribute(
        HIPDNN_ATTR_ENGINECFG_ENGINE, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &count, engine2.getPtr()));
    ASSERT_EQ(count, 1);
}

TEST_F(TestEngineConfigDescriptor, GetEngineThrowsIfNotFinalized)
{
    auto engineConfig = getEngineConfigDescriptor();
    ASSERT_THROW_HIPDNN_STATUS(engineConfig->getEngine(), HIPDNN_STATUS_INTERNAL_ERROR);
}

TEST_F(TestEngineConfigDescriptor, GetEngineReturnsPointerIfFinalized)
{
    auto engineConfig = getEngineConfigDescriptor();
    makeEngineConfigFinalized();
    auto enginePtr = engineConfig->getEngine();
    ASSERT_NE(enginePtr, nullptr);
    ASSERT_EQ(static_cast<const IBackendDescriptor*>(enginePtr.get()),
              static_cast<const IBackendDescriptor*>(getMockEngine().get()));
}

TEST_F(TestEngineConfigDescriptor, GetEngineDescriptorMaxWorkspaceSize)
{
    auto engineConfig = getEngineConfigDescriptor();
    int64_t workspaceSize = 0;

    makeEngineConfigFinalized();

    ASSERT_THROW_HIPDNN_STATUS(engineConfig->getAttribute(HIPDNN_ATTR_ENGINECFG_WORKSPACE_SIZE,
                                                          HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                          1,
                                                          nullptr,
                                                          &workspaceSize),
                               HIPDNN_STATUS_BAD_PARAM);

    ASSERT_THROW_HIPDNN_STATUS(
        engineConfig->getAttribute(
            HIPDNN_ATTR_ENGINECFG_WORKSPACE_SIZE, HIPDNN_TYPE_INT64, 2, nullptr, &workspaceSize),
        HIPDNN_STATUS_BAD_PARAM);

    ASSERT_THROW_HIPDNN_STATUS(
        engineConfig->getAttribute(
            HIPDNN_ATTR_ENGINECFG_WORKSPACE_SIZE, HIPDNN_TYPE_INT64, 1, nullptr, nullptr),
        HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);

    ASSERT_NO_THROW(engineConfig->getAttribute(
        HIPDNN_ATTR_ENGINECFG_WORKSPACE_SIZE, HIPDNN_TYPE_INT64, 1, nullptr, &workspaceSize));
    ASSERT_EQ(workspaceSize, 1024);

    int64_t count;
    ASSERT_NO_THROW(engineConfig->getAttribute(
        HIPDNN_ATTR_ENGINECFG_WORKSPACE_SIZE, HIPDNN_TYPE_INT64, 1, &count, &workspaceSize));
    ASSERT_EQ(count, 1);
}
