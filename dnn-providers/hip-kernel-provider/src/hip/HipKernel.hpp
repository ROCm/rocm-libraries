// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "HipUtils.hpp"
#include <array>
#include <hip/hip_runtime_api.h>
#include <string>

class HipProgram;

class HipKernel
{
public:
    HipKernel(const HipProgram& program, const std::string& kernelName);

    void setBlockSize(unsigned int x, unsigned int y = 1, unsigned int z = 1);
    void setGridSize(unsigned int x, unsigned int y = 1, unsigned int z = 1);
    void setSharedMemBytes(unsigned int bytes);

    template <typename... Args>
    void launch(hipStream_t stream, Args&&... args) const
    {
        // Pack arguments into void* array
        std::array<void*, sizeof...(Args)> kernelParams
            = {const_cast<void*>(static_cast<const void*>(&args))...};

        HIP_CHECK(hipModuleLaunchKernel(_kernel,
                                        _gridX,
                                        _gridY,
                                        _gridZ,
                                        _blockX,
                                        _blockY,
                                        _blockZ,
                                        _sharedMemBytes,
                                        stream,
                                        kernelParams.data(),
                                        nullptr));
    }

    ~HipKernel() = default;

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
