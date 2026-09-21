// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <array>
#include <hip/hip_runtime_api.h>

namespace hip_kernel_provider::compilation
{

class IRunnableKernel
{
public:
    virtual ~IRunnableKernel() = default;

    virtual void setBlockSize(unsigned int x, unsigned int y, unsigned int z) = 0;
    virtual void setGridSize(unsigned int x, unsigned int y, unsigned int z) = 0;
    virtual void setSharedMemBytes(unsigned int bytes) = 0;

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
