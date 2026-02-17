// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "HipUtils.hpp"
#include <array>
#include <hip/hip_runtime_api.h>
#include <string>
#include <vector>

class HipProgram;

class HipKernel
{
public:
    HipKernel(const HipProgram& program, const std::string& kernelName);

    void SetBlockSize(unsigned int x, unsigned int y = 1, unsigned int z = 1);
    void SetGridSize(unsigned int x, unsigned int y = 1, unsigned int z = 1);
    void SetSharedMemBytes(unsigned int bytes);

    template <typename... Args>
    void Launch(hipStream_t stream, Args&&... args)
    {
        // Pack arguments into void* array
        std::array<void*, sizeof...(Args)> kernelParams
            = {const_cast<void*>(static_cast<const void*>(&args))...};

        HIP_CHECK(hipModuleLaunchKernel(kernel_,
                                        gridX_,
                                        gridY_,
                                        gridZ_,
                                        blockX_,
                                        blockY_,
                                        blockZ_,
                                        sharedMemBytes_,
                                        stream,
                                        kernelParams.data(),
                                        nullptr));
    }

    ~HipKernel() = default;

private:
    std::string kernelName_;
    hipFunction_t kernel_;
    unsigned int blockX_ = 1, blockY_ = 1, blockZ_ = 1;
    unsigned int gridX_ = 1, gridY_ = 1, gridZ_ = 1;
    unsigned int sharedMemBytes_ = 0;
};
