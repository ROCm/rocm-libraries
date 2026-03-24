// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "HipCompiledProgram.hpp"
#include "HipProgram.hpp"
#include "IKernelCompiler.hpp"

#include <memory>
#include <string>
#include <vector>

namespace example_plugin
{

/// Concrete IKernelCompiler that compiles kernels using HIPRTC.
///
/// Creates a HipProgram (which handles HIPRTC compilation and module loading)
/// and wraps it in a HipCompiledProgram.
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

} // namespace example_plugin
