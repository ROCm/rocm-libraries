// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "HipRunnableKernel.hpp"

#include "HipUtils.hpp"

#include <utility>

namespace example_provider
{

HipRunnableKernel::HipRunnableKernel(hipFunction_t function, std::string kernelName)
    : _kernelName(std::move(kernelName))
    , _kernel(function)
{
}

void HipRunnableKernel::setBlockSize(unsigned int x, unsigned int y, unsigned int z)
{
    _blockX = x;
    _blockY = y;
    _blockZ = z;
}

void HipRunnableKernel::setGridSize(unsigned int x, unsigned int y, unsigned int z)
{
    _gridX = x;
    _gridY = y;
    _gridZ = z;
}

void HipRunnableKernel::setSharedMemBytes(unsigned int bytes)
{
    _sharedMemBytes = bytes;
}

void HipRunnableKernel::launchImpl(hipStream_t stream, void** kernelParams) const
{
    HIP_CHECK(hipModuleLaunchKernel(_kernel,
                                    _gridX,
                                    _gridY,
                                    _gridZ,
                                    _blockX,
                                    _blockY,
                                    _blockZ,
                                    _sharedMemBytes,
                                    stream,
                                    kernelParams,
                                    nullptr));
}

} // namespace example_provider
