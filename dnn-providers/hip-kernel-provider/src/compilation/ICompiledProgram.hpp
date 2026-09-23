// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "IRunnableKernel.hpp"

#include <memory>
#include <string>

namespace hip_kernel_provider::compilation
{

class ICompiledProgram
{
public:
    virtual ~ICompiledProgram() = default;
    virtual std::unique_ptr<IRunnableKernel> getKernel(const std::string& kernelName) const = 0;
};

} // namespace hip_kernel_provider::compilation
