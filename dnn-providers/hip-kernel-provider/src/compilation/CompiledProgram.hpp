// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "ICompiledProgram.hpp"
#include "Kernel.hpp"
#include "Program.hpp"

#include <memory>

namespace hip_kernel_provider::compilation
{

class CompiledProgram : public ICompiledProgram
{
public:
    explicit CompiledProgram(std::shared_ptr<Program> program)
        : _program(std::move(program))
    {
    }

    std::unique_ptr<IRunnableKernel> getKernel(const std::string& kernelName) const override
    {
        return std::make_unique<Kernel>(*_program, kernelName);
    }

private:
    std::shared_ptr<Program> _program;
};

} // namespace hip_kernel_provider::compilation
