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

namespace hip_kernel_provider::kernel_ingestor_engine
{

/**
 * @brief The native dispatch behind this pack's UDD: sizes and launches a pointwise add.
 *
 * Holds the kernel compiler a launch needs.
 *
 * Splits per RFC 0017 §8.5: everything derived from the graph and chosen kernel
 * resolves once at plan build; execute only resolves device pointers by uid and
 * launches. A plan may execute concurrently from several threads, so nothing here
 * mutates after preparation.
 */
class PointwiseAddDispatchHandler
    : public hipdnn_plugin_sdk::ingestor::IKernelDispatchHandler<Handle>
{
public:
    /// @param kernelCompiler Owned by the engine, which outlives this handler.
    ///
    /// Device properties are not held; they arrive per call on the MatchContext, so a
    /// kernel is compiled for the device the call is actually for.
    explicit PointwiseAddDispatchHandler(const compilation::IKernelCompiler& kernelCompiler)
        : _kernelCompiler(kernelCompiler)
    {
    }

    /**
     * @brief Scratch this kernel requires.
     *
     * A one-element add needs none. The 256-block kernel reports a non-zero
     * requirement so the engine's max-across-survivors is observably a maximum.
     */
    size_t
        workspaceBytes(const hipdnn_plugin_sdk::ingestor::MatchContext& context,
                       const hipdnn_plugin_sdk::ingestor::BoundTokens& bound,
                       const hipdnn_plugin_sdk::ingestor::KernelDefinition& kernel) const override;

    std::unique_ptr<hipdnn_plugin_sdk::ingestor::PreparedDispatch>
        prepare(const hipdnn_plugin_sdk::ingestor::MatchContext& context,
                const hipdnn_plugin_sdk::ingestor::BoundTokens& bound,
                const hipdnn_plugin_sdk::ingestor::KernelDefinition& kernel) const override;

    void launch(const Handle& handle,
                const hipdnn_plugin_sdk::ingestor::PreparedDispatch& prepared,
                const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                uint32_t numDeviceBuffers,
                void* workspace) const override;

private:
    const compilation::IKernelCompiler& _kernelCompiler;
};

/**
 * @brief This pack's dispatch handler.
 *
 * Process-lifetime: the registry holds a non-owning pointer to it while a provider's
 * Container is created and destroyed per handle, so it must outlive every Container.
 * The compiler it holds is a static for the same reason.
 */
const PointwiseAddDispatchHandler& pointwiseAddDispatchHandler();

} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
