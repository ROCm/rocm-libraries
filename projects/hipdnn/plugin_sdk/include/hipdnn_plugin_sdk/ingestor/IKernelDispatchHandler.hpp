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
 * A provider derives from this to hold whatever its kernels need: argument uids, launch
 * geometry, a loaded code object, a preallocated argument buffer.
 *
 * It exists because a plan outlives the call that built it. The bound token state a
 * MatchContext carries is references into a graph hipDNN owns for the duration of one
 * plugin call, so a plan cannot keep it. Instead the handler reads that state once, at
 * plan build, and hands back owned values.
 *
 * This is also the shape RFC 0017 §8.5 describes: the plan build evaluates the dispatch
 * descriptor's grid, block, shared-memory and argument formulas over the bound token
 * state, and execute afterwards only resolves device pointers by uid and launches.
 * Nothing is re-matched and nothing is re-derived per execution.
 */
class PreparedDispatch
{
public:
    virtual ~PreparedDispatch() = default;
};

/**
 * @brief The native escape hatch for a UDD: how to size, prepare, and launch a kernel.
 *
 * This is the seam that keeps provider-specific machinery out of the SDK. A provider's
 * implementation holds whatever it needs to run a kernel — a compiler, a module cache, a
 * loaded code object — and the SDK never names any of it. Templating on THandle matches
 * IPlanBuilder and IPlan, which are parameterized the same way for the same reason.
 *
 * RFC 0017 §6 makes workspace part of this interface deliberately: the workspace query
 * arrives before any kernel is chosen and before a plan exists, so anything that can
 * launch must also be able to size its scratch. A handler that can launch but cannot
 * answer the workspace question is incomplete.
 *
 * The data-driven replacement evaluates the UDD's expressions instead of calling native
 * code. That is the UDD follow-up RFC; this interface is what it plugs into.
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
     * every kernel the caller's knobs leave in the catalog. The answer stays independent
     * of which other kernels are present: the catalog is never passed in.
     *
     * Takes the bound token state as well as the kernel because a workspace requirement
     * is a formula over graph dimensions as much as over kernel metadata. RFC 0017 §6's
     * worked example is `batch * num_heads * seqlen_q * 4`, which a signature over the
     * kernel alone cannot express, and the declarative evaluator that eventually replaces
     * a native handler has to.
     */
    // NOLINTNEXTLINE(portability-template-virtual-member-function)
    virtual size_t workspaceBytes(const MatchContext& context, const KernelDefinition& kernel) const
        = 0;

    /**
     * @brief Resolves everything @p kernel's launch needs from the bound token state.
     *
     * Called once, at plan build, while @p context is still valid. The returned object is
     * owned by the plan and must not reference anything in @p context.
     */
    // NOLINTNEXTLINE(portability-template-virtual-member-function)
    virtual std::unique_ptr<PreparedDispatch> prepare(const MatchContext& context,
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
