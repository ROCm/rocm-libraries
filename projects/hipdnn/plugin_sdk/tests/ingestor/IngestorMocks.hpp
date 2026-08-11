// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <memory>

#include <gmock/gmock.h>

#include <hip/hip_runtime_api.h>
#include <hipdnn_plugin_sdk/ingestor/IDeviceResolver.hpp>
#include <hipdnn_plugin_sdk/ingestor/IKernelDispatchHandler.hpp>

#include "KernelIngestorTestFixtures.hpp"

/**
 * @file IngestorMocks.hpp
 * @brief gmock doubles for the ingestor's two pure interfaces.
 *
 * IKernelDispatchHandler and IDeviceResolver have no test of their own (see the Phase 2
 * plan): they are asserted only through the generic code that consumes them --
 * GenericPlan/GenericPlanBuilder for the dispatch handler, GenericPlanBuilder::contextFor
 * for the device resolver. These mocks are what those tests set expectations against.
 */
namespace hipdnn_plugin_sdk::ingestor::testing
{

/// Mocks the native dispatch escape hatch, so GenericPlan/GenericPlanBuilder tests can
/// assert exactly what a plan asked of it (workspace queried before prepare, buffers
/// forwarded to launch unchanged) without a real kernel launch.
class MockKernelDispatchHandler : public IKernelDispatchHandler<StubHandle>
{
public:
    MOCK_METHOD(size_t,
                workspaceBytes,
                (const MatchContext& context,
                 const BoundTokens& bound,
                 const KernelDefinition& kernel),
                (const, override));
    MOCK_METHOD(std::unique_ptr<PreparedDispatch>,
                prepare,
                (const MatchContext& context,
                 const BoundTokens& bound,
                 const KernelDefinition& kernel),
                (const, override));
    MOCK_METHOD(void,
                launch,
                (const StubHandle& handle,
                 const PreparedDispatch& prepared,
                 const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                 uint32_t numDeviceBuffers,
                 void* workspace),
                (const, override));
};

/// Mocks device resolution so GenericPlanBuilder::contextFor() can be shown folding
/// per-handle device facts into the MatchContext it builds: two handles that resolve to
/// different devices must reach the matchers with different DeviceId/deviceProperties,
/// proving the resolution happens per call rather than once at construction.
class MockDeviceResolver : public IDeviceResolver<StubHandle>
{
public:
    MOCK_METHOD(DeviceId, deviceId, (const StubHandle& handle), (const, override));
    MOCK_METHOD(const hipDeviceProp_t&, deviceProperties, (DeviceId deviceId), (const, override));
};

} // namespace hipdnn_plugin_sdk::ingestor::testing

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
