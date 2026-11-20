// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <utility>

#include <hipdnn_plugin_sdk/PluginException.hpp>

#include "Kernel.hpp"

namespace hipblaslt_plugin::compilation
{

Kernel::Kernel(hipFunction_t function, std::string name, std::shared_ptr<const void> owner)
    : _function(function)
    , _name(std::move(name))
    , _owner(std::move(owner))
{
}

void Kernel::setGridSize(unsigned int x, unsigned int y, unsigned int z)
{
    _gridX = x;
    _gridY = y;
    _gridZ = z;
}

void Kernel::setBlockSize(unsigned int x, unsigned int y, unsigned int z)
{
    _blockX = x;
    _blockY = y;
    _blockZ = z;
}

void Kernel::launchImpl(hipStream_t stream, unsigned int gridX, void** kernelParams) const
{
    const auto result = hipModuleLaunchKernel(_function,
                                              gridX,
                                              _gridY,
                                              _gridZ,
                                              _blockX,
                                              _blockY,
                                              _blockZ,
                                              /*sharedMemBytes=*/0,
                                              stream,
                                              kernelParams,
                                              nullptr);
    if(result != hipSuccess)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
            "launching kernel '" + _name + "' failed: " + hipGetErrorName(result));
    }
}

} // namespace hipblaslt_plugin::compilation
