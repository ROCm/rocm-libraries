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
 * @brief gmock doubles for the ingestor's two pure interfaces, IKernelDispatchHandler
 *        and IDeviceResolver.
 */
namespace hipdnn_plugin_sdk::ingestor::testing
{

/// Mocks the native dispatch escape hatch for asserting what a plan asked of it.
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

/// Resolves per-handle device facts per call, not once at construction.
class MockDeviceResolver : public IDeviceResolver<StubHandle>
{
public:
    MOCK_METHOD(DeviceId, deviceId, (const StubHandle& handle), (const, override));
    MOCK_METHOD(const DeviceProperties&, deviceProperties, (DeviceId deviceId), (const, override));
};

} // namespace hipdnn_plugin_sdk::ingestor::testing

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
