// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "IRunnableKernel.hpp"

#include <hip/hip_runtime_api.h>
#include <string>

namespace example_provider
{

/// Concrete IRunnableKernel wrapping a hipFunction_t.
///
/// Stores grid/block dimensions and launches the kernel via
/// hipModuleLaunchKernel().
class HipRunnableKernel : public IRunnableKernel
{
public:
    HipRunnableKernel(hipFunction_t function, std::string kernelName);

    void setBlockSize(unsigned int x, unsigned int y, unsigned int z) override;
    void setGridSize(unsigned int x, unsigned int y, unsigned int z) override;
    void setSharedMemBytes(unsigned int bytes) override;

protected:
    void launchImpl(hipStream_t stream, void** kernelParams) const override;

private:
    std::string _kernelName;
    hipFunction_t _kernel;
    unsigned int _blockX = 1;
    unsigned int _blockY = 1;
    unsigned int _blockZ = 1;
    unsigned int _gridX = 1;
    unsigned int _gridY = 1;
    unsigned int _gridZ = 1;
    unsigned int _sharedMemBytes = 0;
};

} // namespace example_provider
