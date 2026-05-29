// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "HipKernel.hpp"
#include "HipProgram.hpp"

namespace hip_kernel_provider
{

HipKernel::HipKernel(const HipProgram& program, const std::string& kernelName)
    : _kernelName(kernelName)
    , _kernel(program.getKernel(kernelName))
{
}

void HipKernel::setBlockSize(unsigned int x, unsigned int y, unsigned int z)
{
    _blockX = x;
    _blockY = y;
    _blockZ = z;
}

void HipKernel::setGridSize(unsigned int x, unsigned int y, unsigned int z)
{
    _gridX = x;
    _gridY = y;
    _gridZ = z;
}

void HipKernel::setSharedMemBytes(unsigned int bytes)
{
    _sharedMemBytes = bytes;
}

void HipKernel::launchImpl(hipStream_t stream, void** kernelParams) const
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

} // namespace hip_kernel_provider
