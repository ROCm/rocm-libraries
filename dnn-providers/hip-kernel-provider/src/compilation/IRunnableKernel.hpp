// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <array>
#include <hip/hip_runtime_api.h>

#include <hipdnn_plugin_sdk/PluginException.hpp>

namespace hip_kernel_provider::compilation
{

class IRunnableKernel
{
public:
    virtual ~IRunnableKernel() = default;

    virtual void setBlockSize(unsigned int x, unsigned int y, unsigned int z) = 0;
    virtual void setGridSize(unsigned int x, unsigned int y, unsigned int z) = 0;
    virtual void setSharedMemBytes(unsigned int bytes) = 0;

    /// Launch as workgroup clusters of x*y*z workgroups. Only a kernel compiled with the
    /// same cluster dimensions may be launched this way. Not pure: most kernels never
    /// cluster, and one that cannot must refuse rather than launch unclustered.
    virtual void setClusterDims(unsigned int /*x*/, unsigned int /*y*/, unsigned int /*z*/)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_INVALID_VALUE, "this kernel does not support cluster launch");
    }

    template <typename... Args>
    void launch(hipStream_t stream, Args&&... args) const
    {
        std::array<void*, sizeof...(Args)> kernelParams
            = {const_cast<void*>(static_cast<const void*>(&args))...};
        launchImpl(stream, kernelParams.data());
    }

protected:
    virtual void launchImpl(hipStream_t stream, void** kernelParams) const = 0;
};

} // namespace hip_kernel_provider::compilation
