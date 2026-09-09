// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include "KpackProgram.hpp"

#include "Kernel.hpp"

#include <utility>

#include <hipdnn_plugin_sdk/PluginException.hpp>

namespace hip_kernel_provider::compilation
{

KpackProgram::KpackProgram(std::shared_ptr<const KpackModule> module, std::string descriptorLabel)
    : _module(std::move(module))
    , _descriptorLabel(std::move(descriptorLabel))
{
}

std::unique_ptr<IRunnableKernel> KpackProgram::getKernel(const std::string& kernelName) const
{
    hipFunction_t function = nullptr;
    const hipError_t status
        = hipModuleGetFunction(&function, _module->module(), kernelName.c_str());
    if(status != hipSuccess)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
            "kpack kernel source for " + _descriptorLabel + ": symbol '" + kernelName
                + "' is not present in the loaded module: " + hipGetErrorString(status));
    }

    return std::make_unique<Kernel>(function, kernelName);
}

} // namespace hip_kernel_provider::compilation

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
