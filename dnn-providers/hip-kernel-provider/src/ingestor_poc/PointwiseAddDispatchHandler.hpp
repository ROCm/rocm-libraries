// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cstddef>
#include <cstdint>
#include <memory>

#include <hip/hip_runtime_api.h>
#include <hipdnn_plugin_sdk/ingestor/IKernelDispatchHandler.hpp>

#include "compilation/IKernelCompiler.hpp"
#include "core/Handle.hpp"

namespace hip_kernel_provider::ingestor_poc
{

/**
 * @brief The native dispatch behind this POC's UDD: sizes and launches a pointwise add.
 *
 * Holds the provider machinery a launch needs — a kernel compiler and the device
 * properties it compiles against — which is exactly why this interface exists: the SDK
 * never names either of them.
 *
 * The work splits the way RFC 0017 §8.5 describes. Everything derived from the graph and
 * the chosen kernel (which uids the arguments bind to, the launch geometry, the compiled
 * kernel itself) resolves once at plan build; execute afterwards only resolves device
 * pointers by uid and launches. A plan may execute concurrently from several threads, so
 * nothing here mutates after preparation.
 */
class PointwiseAddDispatchHandler
    : public hipdnn_plugin_sdk::ingestor::IKernelDispatchHandler<Handle>
{
public:
    /// @param kernelCompiler Owned by the engine, which outlives this handler.
    ///
    /// Device properties are not held: they arrive per call on the MatchContext, so a
    /// kernel is always compiled for the device the call is actually for rather than
    /// for whichever device was current when this handler was built.
    explicit PointwiseAddDispatchHandler(const compilation::IKernelCompiler& kernelCompiler)
        : _kernelCompiler(kernelCompiler)
    {
    }

    /**
     * @brief Scratch this kernel requires.
     *
     * A one-element add needs none. The 256-block kernel nonetheless reports a non-zero
     * requirement so the engine's "maximum across surviving kernels" is observably a
     * maximum rather than a constant zero — the workspace path is otherwise
     * indistinguishable from not being wired up at all.
     */
    size_t
        workspaceBytes(const hipdnn_plugin_sdk::ingestor::KernelDefinition& kernel) const override;

    std::unique_ptr<hipdnn_plugin_sdk::ingestor::PreparedDispatch>
        prepare(const hipdnn_plugin_sdk::ingestor::MatchContext& context,
                const hipdnn_plugin_sdk::ingestor::KernelDefinition& kernel) const override;

    void launch(const Handle& handle,
                const hipdnn_plugin_sdk::ingestor::PreparedDispatch& prepared,
                const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                uint32_t numDeviceBuffers,
                void* workspace) const override;

private:
    const compilation::IKernelCompiler& _kernelCompiler;
};

/// @brief Registers @p handler under this pack's dispatch symbol.
///
/// The handler is registered by pointer and must outlive every plan built from it.
void registerPointwiseAddDispatch(const PointwiseAddDispatchHandler& handler);

} // namespace hip_kernel_provider::ingestor_poc

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
