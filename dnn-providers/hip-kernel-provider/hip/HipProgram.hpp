// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hip/hip_runtime_api.h>
#include <string>
#include <vector>

class HipProgram
{
public:
    HipProgram(std::string kernelFileName, std::vector<std::string> options);
    hipFunction_t GetKernel(const std::string& kernelName) const;

    ~HipProgram();

private:
    std::string programName_;
    hipModule_t module_ = nullptr;
    std::vector<char> binary_;
};
