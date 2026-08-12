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
 * @brief Everything a launch needs that was derived from the graph, resolved once.
 *
 * A provider derives from this to hold what its kernels need: argument uids, launch
 * geometry, a loaded code object, a preallocated argument buffer.
 *
 * A plan outlives the call that built it, and a MatchContext holds references into a
 * graph hipDNN owns for one plugin call only, so a plan cannot keep that state. The
 * handler reads it once at plan build and returns owned values (RFC 0017 §8.5).
 */
class PreparedDispatch
{
public:
    virtual ~PreparedDispatch() = default;
};

/**
 * @brief The native escape hatch for a UDD: how to size, prepare, and launch a kernel.
 *
 * Keeps provider machinery out of the SDK -- an implementation holds its own compiler,
 * module cache and code objects, none of which the SDK names. Templated on THandle to
 * match IPlanBuilder and IPlan.
 *
 * Workspace belongs here (RFC 0017 §6) because the workspace query arrives before any
 * kernel is chosen: anything that can launch must also be able to size its scratch.
 */
template <typename THandle>
class IKernelDispatchHandler
{
public:
    virtual ~IKernelDispatchHandler() = default;

    /**
     * @brief Global scratch this kernel requires, in bytes.
     *
     * Answered per kernel, before selection, so the engine can report the maximum across
     * the kernels the caller's knobs leave in the catalog. The catalog is never passed
     * in, so the answer cannot depend on which other kernels are present.
     *
     * @param bound What matching already resolved about this graph. A workspace size is
     *        a formula over graph dimensions as much as kernel metadata, so this is read
     *        rather than re-derived (RFC 0017 §8.1: nothing is re-matched).
     */
    // NOLINTNEXTLINE(portability-template-virtual-member-function)
    virtual size_t workspaceBytes(const MatchContext& context,
                                  const BoundTokens& bound,
                                  const KernelDefinition& kernel) const
        = 0;

    /**
     * @brief Resolves everything @p kernel's launch needs from the bound token state.
     *
     * Called once, at plan build, while @p context is still valid. The returned object is
     * owned by the plan and MUST NOT reference anything in @p context or @p bound.
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
     * @param deviceBuffers    uid/pointer pairs; resolve each argument by uid through
     *                         hipdnn_plugin_sdk::findDeviceBuffer.
     * @param numDeviceBuffers Length of @p deviceBuffers.
     * @param workspace        The plan's scratch, or nullptr when workspaceBytes() was 0.
     *
     * @note May be called concurrently from several threads with different device
     *       buffers, so it must not mutate @p prepared or the handler.
     *
     * One call, one kernel. When multi-launch UDDs land, the handler should issue the N
     * launches internally rather than the SDK calling launch() N times: launch i's
     * geometry can depend on what launch i-1 produced, and only the handler owning the
     * whole dispatch can resolve that.
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
