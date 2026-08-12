// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cstddef>
#include <cstdint>
#include <memory>

#include <hipdnn_plugin_sdk/PluginApiDataTypes.h>
#include <hipdnn_plugin_sdk/ingestor/KernelDefinition.hpp>
#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>

namespace hipdnn_plugin_sdk::ingestor
{

/**
 * @brief Launch state derived from the graph, resolved once at plan build.
 *
 * A provider derives from this to hold what its kernels need (argument uids, launch
 * geometry, code object, argument buffer). Must not reference the MatchContext or
 * BoundTokens it was built from (RFC 0017 §8.5).
 */
class PreparedDispatch
{
public:
    virtual ~PreparedDispatch() = default;
};

/**
 * @brief Native escape hatch for a UDD: sizes, prepares, and launches a kernel.
 *
 * Templated on THandle to match IPlanBuilder and IPlan. Also answers workspace size
 * (RFC 0017 §6), since that query arrives before a kernel is chosen.
 */
template <typename THandle>
class IKernelDispatchHandler
{
public:
    virtual ~IKernelDispatchHandler() = default;

    /**
     * @brief Global scratch this kernel requires, in bytes.
     *
     * Answered per kernel before selection; must not depend on which other kernels
     * are present (the catalog is never passed in).
     *
     * @param bound Graph state matching already resolved (RFC 0017 §8.1: not re-derived).
     */
    // NOLINTNEXTLINE(portability-template-virtual-member-function)
    virtual size_t workspaceBytes(const MatchContext& context,
                                  const BoundTokens& bound,
                                  const KernelDefinition& kernel) const
        = 0;

    /**
     * @brief Resolves everything @p kernel's launch needs from the bound token state.
     *
     * Called once at plan build, while @p context is still valid. The returned object
     * is owned by the plan and MUST NOT reference @p context or @p bound.
     */
    // NOLINTNEXTLINE(portability-template-virtual-member-function)
    virtual std::unique_ptr<PreparedDispatch> prepare(const MatchContext& context,
                                                      const BoundTokens& bound,
                                                      const KernelDefinition& kernel) const
        = 0;

    /**
     * @brief Launches a prepared kernel against the caller's device buffers.
     *
     * @param handle           Supplies the stream the launch is ordered on.
     * @param prepared         The object this handler returned from prepare().
     * @param deviceBuffers    uid/pointer pairs; resolve via
     *                         hipdnn_plugin_sdk::findDeviceBuffer.
     * @param numDeviceBuffers Length of @p deviceBuffers.
     * @param workspace        The plan's scratch, or nullptr when workspaceBytes() was 0.
     *
     * @note May run concurrently across threads; must not mutate @p prepared or the
     *       handler.
     */
    // NOLINTNEXTLINE(portability-template-virtual-member-function)
    virtual void launch(const THandle& handle,
                        const PreparedDispatch& prepared,
                        const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                        uint32_t numDeviceBuffers,
                        void* workspace) const
        = 0;
};

} // namespace hipdnn_plugin_sdk::ingestor

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
