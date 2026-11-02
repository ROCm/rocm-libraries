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

void Kernel::setSharedMemBytes(unsigned int bytes)
{
    _sharedMemBytes = bytes;
}

void Kernel::launchImpl(hipStream_t stream, void** kernelParams) const
{
    const auto result = hipModuleLaunchKernel(_function,
                                              _gridX,
                                              _gridY,
                                              _gridZ,
                                              _blockX,
                                              _blockY,
                                              _blockZ,
                                              _sharedMemBytes,
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
