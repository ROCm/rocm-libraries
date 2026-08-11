// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <hipdnn_plugin_sdk/PluginApiDataTypes.h>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>
#include <hipdnn_plugin_sdk/ingestor/GenericPlan.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelIngestorStateManager.hpp>
#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>

#include "IngestorMocks.hpp"
#include "KernelIngestorTestFixtures.hpp"

/**
 * @file TestGenericPlan.cpp
 * @brief Unit tests for GenericPlan.hpp: the plan every descriptor-backed engine
 *        produces, holding one selected kernel and its resolved dispatch handler.
 */
namespace
{

using namespace hipdnn_plugin_sdk::ingestor;
using namespace hipdnn_plugin_sdk::ingestor::testing;
using ::testing::_;
using ::testing::Return;

KernelDispatcher<StubHandle> makeDispatcher(const MockKernelDispatchHandler& handler)
{
    return {makeDefinition(testId(0x01), 64), &handler};
}

TEST(TestIngestorGenericPlan, ConstructorThrowsInternalErrorWhenPrepareReturnsNull)
{
    // prepare() returning null means the handler resolved no launch for this kernel: an
    // internal inconsistency between what applicability accepted and what dispatch can
    // actually build, not a caller error.
    const MockKernelDispatchHandler handler;
    EXPECT_CALL(handler, workspaceBytes(_, _, _)).WillOnce(Return(0));
    EXPECT_CALL(handler, prepare(_, _, _)).WillOnce(Return(nullptr));

    const TestGraph graph;
    const auto properties = testDeviceProperties();
    const MatchContext context{graph, 0, properties};
    const BoundTokens bound;

    try
    {
        const GenericPlan<StubHandle> plan(makeDispatcher(handler), context, bound);
        FAIL() << "expected HipdnnPluginException";
    }
    catch(const hipdnn_plugin_sdk::HipdnnPluginException& ex)
    {
        EXPECT_EQ(ex.getStatus(), HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR);
    }
}

TEST(TestIngestorGenericPlan, GetWorkspaceSizeReportsWhatTheHandlerComputedAtConstruction)
{
    const MockKernelDispatchHandler handler;
    EXPECT_CALL(handler, workspaceBytes(_, _, _)).WillOnce(Return(size_t{4096}));
    EXPECT_CALL(handler, prepare(_, _, _))
        .WillOnce(Return(::testing::ByMove(std::make_unique<PreparedDispatch>())));

    const TestGraph graph;
    const auto properties = testDeviceProperties();
    const MatchContext context{graph, 0, properties};
    const BoundTokens bound;

    const GenericPlan<StubHandle> plan(makeDispatcher(handler), context, bound);
    const StubHandle handle;

    EXPECT_EQ(plan.getWorkspaceSize(handle), 4096U);
}

TEST(TestIngestorGenericPlan, ExecuteThrowsInvalidValueWhenWorkspaceRequiredButNoneProvided)
{
    const MockKernelDispatchHandler handler;
    EXPECT_CALL(handler, workspaceBytes(_, _, _)).WillOnce(Return(size_t{1024}));
    EXPECT_CALL(handler, prepare(_, _, _))
        .WillOnce(Return(::testing::ByMove(std::make_unique<PreparedDispatch>())));
    // Workspace is missing, so launch() must never be reached.
    EXPECT_CALL(handler, launch(_, _, _, _, _)).Times(0);

    const TestGraph graph;
    const auto properties = testDeviceProperties();
    const MatchContext context{graph, 0, properties};
    const BoundTokens bound;

    const GenericPlan<StubHandle> plan(makeDispatcher(handler), context, bound);
    const StubHandle handle;

    try
    {
        plan.execute(handle, nullptr, 0, nullptr);
        FAIL() << "expected HipdnnPluginException";
    }
    catch(const hipdnn_plugin_sdk::HipdnnPluginException& ex)
    {
        EXPECT_EQ(ex.getStatus(), HIPDNN_PLUGIN_STATUS_INVALID_VALUE);
    }
}

TEST(TestIngestorGenericPlan, ExecuteForwardsDeviceBuffersAndWorkspaceToLaunchUnchanged)
{
    const MockKernelDispatchHandler handler;
    EXPECT_CALL(handler, workspaceBytes(_, _, _)).WillOnce(Return(size_t{256}));
    EXPECT_CALL(handler, prepare(_, _, _))
        .WillOnce(Return(::testing::ByMove(std::make_unique<PreparedDispatch>())));

    const std::array<hipdnnPluginDeviceBuffer_t, 1> buffers{
        {{/*uid=*/7, /*ptr=*/reinterpret_cast<void*>(0x1234)}}};
    int workspaceStorage = 0;
    void* const workspace = &workspaceStorage;

    EXPECT_CALL(handler, launch(_, _, buffers.data(), 1U, workspace)).Times(1);

    const TestGraph graph;
    const auto properties = testDeviceProperties();
    const MatchContext context{graph, 0, properties};
    const BoundTokens bound;

    const GenericPlan<StubHandle> plan(makeDispatcher(handler), context, bound);
    const StubHandle handle;

    plan.execute(handle, buffers.data(), 1, workspace);
}

TEST(TestIngestorGenericPlan, KernelReturnsTheDispatchersSelectedKernel)
{
    const MockKernelDispatchHandler handler;
    EXPECT_CALL(handler, workspaceBytes(_, _, _)).WillOnce(Return(0));
    EXPECT_CALL(handler, prepare(_, _, _))
        .WillOnce(Return(::testing::ByMove(std::make_unique<PreparedDispatch>())));

    const TestGraph graph;
    const auto properties = testDeviceProperties();
    const MatchContext context{graph, 0, properties};
    const BoundTokens bound;
    const auto kernelId = testId(0x77);

    const GenericPlan<StubHandle> plan({makeDefinition(kernelId, 64), &handler}, context, bound);

    EXPECT_EQ(plan.kernel().kernelId, kernelId);
}

} // namespace

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
