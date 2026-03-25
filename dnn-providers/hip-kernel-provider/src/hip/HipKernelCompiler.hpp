// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "HipCompiledProgram.hpp"
#include "HipProgram.hpp"
#include "IKernelCompiler.hpp"

#include <memory>

namespace hip_kernel_provider
{

class HipKernelCompiler : public IKernelCompiler
{
public:
    std::unique_ptr<ICompiledProgram>
        compile(const std::string& kernelFileName,
                const std::vector<std::string>& options) const override
    {
        auto program = std::make_shared<HipProgram>(kernelFileName, options);
        return std::make_unique<HipCompiledProgram>(std::move(program));
    }
};

} // namespace hip_kernel_provider
