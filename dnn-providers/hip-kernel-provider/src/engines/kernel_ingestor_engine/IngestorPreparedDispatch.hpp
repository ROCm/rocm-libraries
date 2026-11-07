// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <hipdnn_plugin_sdk/ingestor/IKernelDispatchHandler.hpp>

#include "engines/kernel_ingestor_engine/IngestorKernelCode.hpp"

namespace hip_kernel_provider::kernel_ingestor_engine
{

/// A prepared dispatch of this provider's ingestor packs. It exposes the kernel code it
/// launches, so a save can read the code object without knowing the pack.
class IngestorPreparedDispatch : public hipdnn_plugin_sdk::ingestor::PreparedDispatch
{
public:
    virtual const IngestorKernelCode& kernelCode() const = 0;
};

} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
