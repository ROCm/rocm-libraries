// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "ICompiledProgram.hpp"

#include <memory>
#include <string>
#include <vector>

namespace hip_kernel_provider::compilation
{

class IKernelCompiler
{
public:
    virtual ~IKernelCompiler() = default;
    virtual std::unique_ptr<ICompiledProgram> compile(const std::string& kernelFileName,
                                                      const std::vector<std::string>& options) const
        = 0;
};

} // namespace hip_kernel_provider::compilation
