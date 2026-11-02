// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "descriptors/DescriptorTestUtils.hpp"
#include "descriptors/ScopedBackendDescriptor.hpp"
#include "descriptors/mocks/MockDescriptor.hpp"
#include "descriptors/mocks/MockEnginePluginResourceManager.hpp"
#include "descriptors/mocks/MockHandle.hpp"

#include "descriptors/ExecutionPlanDescriptor.hpp"
#include "hipdnn_backend.h"

#include <flatbuffers/flatbuffer_builder.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <hipdnn_flatbuffers_sdk/data_objects/execution_plan_generated.h>

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

using namespace hipdnn_backend;
using namespace hipdnn_backend::plugin;
using namespace hipdnn_backend::test_utilities;
using namespace ::testing;

namespace
{

constexpr int64_t ENGINE_ID = 7;

auto reportDevice(int device)
{
    return DoAll(SetArgPointee<0>(device), Return(hipSuccess));
}

std::string takeLastErrorString()
{
    std::array<char, HIPDNN_ERROR_STRING_MAX_LENGTH> buffer{};
    hipdnnGetLastErrorString(buffer.data(), buffer.size());
    return buffer.data();
}

std::vector<uint8_t> makeSerializedPlan()
{
    flatbuffers::FlatBufferBuilder builder;
    const std::vector<int64_t> tensorUids{1};
    const std::vector<int64_t> tensorAlignments{16};
    const std::vector<uint8_t> pluginPayload{1, 2, 3};
    auto plan = hipdnn_flatbuffers_sdk::data_objects::CreateSerializedExecutionPlan(
        builder,
        2,
        ENGINE_ID,
        0,
        builder.CreateVector(tensorUids),
        builder.CreateVector(pluginPayload),
        false,
        builder.CreateVector(tensorAlignments));
    builder.Finish(plan);
    return {builder.GetBufferPointer(), builder.GetBufferPointer() + builder.GetSize()};
}

} // namespace

// Devices are simulated through MockHandle::getDevice, so no GPU is needed. The plan is bound to
// the device of the handle it is finalized or deserialized on; _executeHandle is a second handle
// that owns its own resource manager, as when a plan is reused across per-thread handles.
class TestBackendExecuteDeviceCheck : public ::testing::Test
{
protected:
    std::shared_ptr<NiceMock<MockEnginePluginResourceManager>> _buildResourceManager
        = std::make_shared<NiceMock<MockEnginePluginResourceManager>>();
    std::shared_ptr<MockEnginePluginResourceManager> _executeResourceManager
        = std::make_shared<MockEnginePluginResourceManager>();
    NiceMock<MockHandle> _buildHandle;
    NiceMock<MockHandle> _executeHandle;

    std::unique_ptr<HipdnnBackendDescriptor> _graphWrapper
        = createDescriptor<MockGraphDescriptor>();
    std::unique_ptr<HipdnnBackendDescriptor> _engineWrapper
        = createDescriptor<MockEngineDescriptor>();
    std::unique_ptr<HipdnnBackendDescriptor> _engineConfigWrapper
        = createDescriptor<MockEngineConfigDescriptor>();
    std::unique_ptr<HipdnnBackendDescriptor> _planWrapper
        = createDescriptor<ExecutionPlanDescriptor>();
    std::unique_ptr<HipdnnBackendDescriptor> _variantPackWrapper
        = createDescriptor<MockVariantDescriptor>();

    void SetUp() override
    {
        ON_CALL(_buildHandle, getPluginResourceManager())
            .WillByDefault(Return(_buildResourceManager));
        ON_CALL(_executeHandle, getPluginResourceManager())
            .WillByDefault(Return(_executeResourceManager));

        ON_CALL(*_buildResourceManager, createExecutionContext(_, _, _))
            .WillByDefault(Return(getExecutionContext()));
        ON_CALL(*_buildResourceManager, createExecutionContextFromSerialized(_, _))
            .WillByDefault(Return(getExecutionContext()));
        ON_CALL(*_buildResourceManager, getWorkspaceSize(_, _)).WillByDefault(Return(0));
    }

    static hipdnnEnginePluginExecutionContext_t getExecutionContext()
    {
        return reinterpret_cast<hipdnnEnginePluginExecutionContext_t>(0xC0FFEE);
    }

    // Finalizes _planWrapper from a graph whose handle is _buildHandle.
    void finalizePlan()
    {
        auto graph
            = MockDescriptorUtility::asDescriptorUnsafe<MockGraphDescriptor>(_graphWrapper.get());
        auto engine
            = MockDescriptorUtility::asDescriptorUnsafe<MockEngineDescriptor>(_engineWrapper.get());
        auto engineConfig = MockDescriptorUtility::asDescriptorUnsafe<MockEngineConfigDescriptor>(
            _engineConfigWrapper.get());

        EXPECT_CALL(*graph, getHandle()).WillRepeatedly(Return(&_buildHandle));
        EXPECT_CALL(*graph, getSerializedGraph())
            .WillRepeatedly(Return(hipdnnPluginConstData_t{nullptr, 0}));
        EXPECT_CALL(*graph, isOverrideShapeEnabled()).WillRepeatedly(Return(false));
        EXPECT_CALL(*engine, getEngineId()).WillRepeatedly(Return(ENGINE_ID));
        EXPECT_CALL(*engine, getGraph()).WillRepeatedly(Return(graph));
        EXPECT_CALL(*engineConfig, isFinalized()).WillRepeatedly(Return(true));
        EXPECT_CALL(*engineConfig, getEngine()).WillRepeatedly(Return(engine));
        EXPECT_CALL(*engineConfig, getSerializedEngineConfig())
            .WillRepeatedly(Return(hipdnnPluginConstData_t{nullptr, 0}));

        ASSERT_EQ(hipdnnBackendSetAttribute(_planWrapper.get(),
                                            HIPDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG,
                                            HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                            1,
                                            &_engineConfigWrapper),
                  HIPDNN_STATUS_SUCCESS);
        ASSERT_EQ(hipdnnBackendFinalize(_planWrapper.get()), HIPDNN_STATUS_SUCCESS);
    }

    hipdnnStatus_t execute(hipdnnBackendDescriptor_t plan)
    {
        return hipdnnBackendExecute(&_executeHandle, plan, _variantPackWrapper.get());
    }

    void expectNoProviderCall()
    {
        EXPECT_CALL(_executeHandle, getPluginResourceManager()).Times(0);
        EXPECT_CALL(*_executeResourceManager, executeOpGraph(_, _)).Times(0);
    }
};

TEST_F(TestBackendExecuteDeviceCheck, SameDeviceOnAnotherHandleReachesProvider)
{
    EXPECT_CALL(_buildHandle, getDevice(_)).WillOnce(reportDevice(1));
    ASSERT_NO_FATAL_FAILURE(finalizePlan());

    // One device query per execute, and the executing handle's resource manager runs the plan.
    EXPECT_CALL(_executeHandle, getDevice(_)).Times(2).WillRepeatedly(reportDevice(1));
    EXPECT_CALL(*_executeResourceManager,
                executeOpGraph(_planWrapper.get(), _variantPackWrapper.get()))
        .Times(2);

    EXPECT_EQ(execute(_planWrapper.get()), HIPDNN_STATUS_SUCCESS);
    EXPECT_EQ(execute(_planWrapper.get()), HIPDNN_STATUS_SUCCESS);
}

TEST_F(TestBackendExecuteDeviceCheck, DifferentDeviceReturnsBadParamBeforeProviderCall)
{
    EXPECT_CALL(_buildHandle, getDevice(_)).WillOnce(reportDevice(0));
    ASSERT_NO_FATAL_FAILURE(finalizePlan());

    EXPECT_CALL(_executeHandle, getDevice(_)).WillOnce(reportDevice(1));
    expectNoProviderCall();

    EXPECT_EQ(execute(_planWrapper.get()), HIPDNN_STATUS_BAD_PARAM);
    const auto message = takeLastErrorString();
    EXPECT_THAT(message, HasSubstr("created for device 0"));
    EXPECT_THAT(message, HasSubstr("stream is on device 1"));
}

TEST_F(TestBackendExecuteDeviceCheck, UnresolvedPlanDeviceSkipsCheck)
{
    EXPECT_CALL(_buildHandle, getDevice(_)).WillOnce(Return(hipErrorNoDevice));
    ASSERT_NO_FATAL_FAILURE(finalizePlan());

    EXPECT_CALL(_executeHandle, getDevice(_)).Times(0);
    EXPECT_CALL(*_executeResourceManager, executeOpGraph(_planWrapper.get(), _)).Times(1);

    EXPECT_EQ(execute(_planWrapper.get()), HIPDNN_STATUS_SUCCESS);
}

TEST_F(TestBackendExecuteDeviceCheck, UnresolvedExecutingDeviceReturnsInternalError)
{
    EXPECT_CALL(_buildHandle, getDevice(_)).WillOnce(reportDevice(0));
    ASSERT_NO_FATAL_FAILURE(finalizePlan());

    EXPECT_CALL(_executeHandle, getDevice(_)).WillOnce(Return(hipErrorInvalidHandle));
    expectNoProviderCall();

    EXPECT_EQ(execute(_planWrapper.get()), HIPDNN_STATUS_INTERNAL_ERROR);
    EXPECT_THAT(takeLastErrorString(), HasSubstr(hipGetErrorName(hipErrorInvalidHandle)));
}

TEST_F(TestBackendExecuteDeviceCheck, DeserializedPlanBindsToLoadHandleDevice)
{
    EXPECT_CALL(_buildHandle, getDevice(_)).WillOnce(reportDevice(3));

    const auto blob = makeSerializedPlan();
    hipdnnBackendDescriptor_t rawPlan = nullptr;
    ASSERT_EQ(hipdnnBackendCreateAndDeserializeExecutionPlan_ext(
                  &_buildHandle, &rawPlan, blob.data(), blob.size()),
              HIPDNN_STATUS_SUCCESS);
    const ScopedBackendDescriptor plan(rawPlan);

    EXPECT_CALL(_executeHandle, getDevice(_)).WillOnce(reportDevice(3)).WillOnce(reportDevice(4));
    EXPECT_CALL(*_executeResourceManager, executeOpGraph(plan.get(), _)).Times(1);

    EXPECT_EQ(execute(plan.get()), HIPDNN_STATUS_SUCCESS);
    EXPECT_EQ(execute(plan.get()), HIPDNN_STATUS_BAD_PARAM);
}
