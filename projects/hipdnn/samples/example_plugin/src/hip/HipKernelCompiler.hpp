// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "HipCompiledProgram.hpp"
#include "IKernelCompiler.hpp"

#include <memory>
#include <string>
#include <vector>

namespace example_plugin
{

/// Concrete IKernelCompiler that compiles kernels using HIPRTC.
///
/// Creates a HipCompiledProgram which handles HIPRTC compilation, module loading,
/// and kernel extraction.
class HipKernelCompiler : public IKernelCompiler
{
public:
    std::unique_ptr<ICompiledProgram>
        compile(const std::string& kernelFileName,
                const std::vector<std::string>& options) const override
    {
        return std::make_unique<HipCompiledProgram>(kernelFileName, options);
    }
};

} // namespace example_plugin
