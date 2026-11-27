// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cstdint>
#include <string>
#include <vector>

#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>
#include <hipdnn_plugin_sdk/ingestor/IKernelDispatchHandler.hpp>

#include "engines/kernel_ingestor_engine/IngestorKernelCode.hpp"

namespace hip_kernel_provider::kernel_ingestor_engine
{

/// One kernel a prepared dispatch launches.
struct IngestorLaunchedKernel
{
    /// Empty if and only if the stage launches one kernel.
    std::string launchName;
    /// Owned by the prepared dispatch.
    const IngestorKernelCode* code = nullptr;
    hipdnn_plugin_sdk::ingestor::KernelSourceKind sourceKind{};
    /// The recorded signature of this launch.
    std::vector<hipdnn_plugin_sdk::ingestor::KernelArgument> signature;
};

/// A prepared dispatch of this provider's ingestor packs. It exposes the kernel code it
/// launches, so a save can read the code object without knowing the pack.
class IngestorPreparedDispatch : public hipdnn_plugin_sdk::ingestor::PreparedDispatch
{
public:
    virtual const IngestorKernelCode& kernelCode() const = 0;

    /// Empty is allowed only for a dispatch over a kernel with a single `source`: capture then
    /// reads kernelCode() and the kernel's own source and signature. A dispatch over a kernel
    /// with `sources` always returns a non-empty list, with exactly one entry and an empty
    /// launchName when a single launch runs. A non-empty list holds every kernel the dispatch
    /// launches, in launch order.
    virtual std::vector<IngestorLaunchedKernel> launchedKernels() const
    {
        return {};
    }

    /// UIDs of the scalars this dispatch reads from device_buffers as host pointers, sorted,
    /// unique.
    virtual std::vector<int64_t> userSuppliedPassByValueUids() const
    {
        return {};
    }
};

} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
