// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <array>
#include <memory>
#include <string>

#include <hip/hip_runtime_api.h>

namespace hipblaslt_plugin::compilation
{

// One kernel entry point and its launch shape. Ported from hip-kernel-provider's
// compilation/Kernel and IRunnableKernel.
class Kernel
{
public:
    // `function` must belong to a module that `owner` keeps loaded, such as the Program
    // it was looked up in.
    Kernel(hipFunction_t function, std::string name, std::shared_ptr<const void> owner);

    void setGridSize(unsigned int x, unsigned int y = 1, unsigned int z = 1);
    void setBlockSize(unsigned int x, unsigned int y = 1, unsigned int z = 1);
    void setSharedMemBytes(unsigned int bytes);

    // Launches on `stream`, passing each of `args` by value as a kernel argument;
    // throws when HIP refuses the launch.
    template <typename... Args>
    void launch(hipStream_t stream, Args&&... args) const
    {
        std::array<void*, sizeof...(Args)> kernelParams
            = {const_cast<void*>(static_cast<const void*>(&args))...};
        launchImpl(stream, kernelParams.data());
    }

private:
    void launchImpl(hipStream_t stream, void** kernelParams) const;

    hipFunction_t _function = nullptr;
    std::string _name;
    std::shared_ptr<const void> _owner;
    unsigned int _gridX = 1;
    unsigned int _gridY = 1;
    unsigned int _gridZ = 1;
    unsigned int _blockX = 1;
    unsigned int _blockY = 1;
    unsigned int _blockZ = 1;
    unsigned int _sharedMemBytes = 0;
};

} // namespace hipblaslt_plugin::compilation
