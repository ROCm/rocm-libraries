// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "IRunnableKernel.hpp"
#include <hip/hip_runtime_api.h>
#include <string>

namespace hip_kernel_provider::compilation
{

class Program;

class Kernel : public IRunnableKernel
{
public:
    Kernel(const Program& program, const std::string& kernelName);

    /// Wraps a function handle resolved elsewhere -- by a kpack archive's module rather
    /// than by a HIPRTC compilation. The launch path is shared, not duplicated: a
    /// kpack-sourced Kernel and a HIPRTC-sourced one are the same object holding the
    /// same hipFunction_t, so launchImpl below serves both.
    ///
    /// The caller owns the module the function belongs to and must keep it alive for
    /// this Kernel's lifetime; hipFunction_t is a non-owning view into a hipModule_t.
    Kernel(hipFunction_t kernel, std::string kernelName);

    void setBlockSize(unsigned int x, unsigned int y = 1, unsigned int z = 1) override;
    void setGridSize(unsigned int x, unsigned int y = 1, unsigned int z = 1) override;
    void setSharedMemBytes(unsigned int bytes) override;

    ~Kernel() override = default;

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

} // namespace hip_kernel_provider::compilation
