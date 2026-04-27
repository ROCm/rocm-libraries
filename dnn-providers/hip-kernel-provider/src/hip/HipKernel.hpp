// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "HipUtils.hpp"
#include "IRunnableKernel.hpp"
#include <string>

namespace hip_kernel_provider
{

class HipProgram;

class HipKernel : public IRunnableKernel
{
public:
    HipKernel(const HipProgram& program, const std::string& kernelName);

    void setBlockSize(unsigned int x, unsigned int y = 1, unsigned int z = 1) override;
    void setGridSize(unsigned int x, unsigned int y = 1, unsigned int z = 1) override;
    void setSharedMemBytes(unsigned int bytes) override;

    ~HipKernel() override = default;

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

} // namespace hip_kernel_provider
