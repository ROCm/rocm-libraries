// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "CompiledProgram.hpp"
#include "IKernelCompiler.hpp"
#include "Program.hpp"

#include <memory>

namespace hip_kernel_provider::compilation
{

class KernelCompiler : public IKernelCompiler
{
public:
    std::unique_ptr<ICompiledProgram>
        compile(const std::string& kernelFileName,
                const std::vector<std::string>& options) const override
    {
        auto program = std::make_shared<Program>(kernelFileName, options);
        return std::make_unique<CompiledProgram>(std::move(program));
    }
};

} // namespace hip_kernel_provider::compilation
