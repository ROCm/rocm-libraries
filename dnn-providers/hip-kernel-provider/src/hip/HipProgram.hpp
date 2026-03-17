// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hip/hip_runtime_api.h>
#include <string>
#include <vector>

namespace hip_kernel_provider
{

class HipProgram
{
public:
    HipProgram(std::string kernelFileName, const std::vector<std::string>& options);
    hipFunction_t getKernel(const std::string& kernelName) const;

    ~HipProgram();

    HipProgram(const HipProgram&) = delete;
    HipProgram& operator=(const HipProgram&) = delete;
    HipProgram(HipProgram&&) = default;
    HipProgram& operator=(HipProgram&&) = default;

private:
    std::string _programName;
    hipModule_t _module = nullptr;
    std::vector<char> _binary;
};

} // namespace hip_kernel_provider
