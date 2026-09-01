// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "Kernel.hpp"
#include "Program.hpp"
#include "Utils.hpp"

#include <utility>

namespace hip_kernel_provider::compilation
{

Kernel::Kernel(const Program& program, const std::string& kernelName)
    : _kernelName(kernelName)
    , _kernel(program.getKernel(kernelName))
{
}

Kernel::Kernel(hipFunction_t kernel, std::string kernelName)
    : _kernelName(std::move(kernelName))
    , _kernel(kernel)
{
}

void Kernel::setBlockSize(unsigned int x, unsigned int y, unsigned int z)
{
    _blockX = x;
    _blockY = y;
    _blockZ = z;
}

void Kernel::setGridSize(unsigned int x, unsigned int y, unsigned int z)
{
    _gridX = x;
    _gridY = y;
    _gridZ = z;
}

void Kernel::setSharedMemBytes(unsigned int bytes)
{
    _sharedMemBytes = bytes;
}

void Kernel::launchImpl(hipStream_t stream, void** kernelParams) const
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

} // namespace hip_kernel_provider::compilation
