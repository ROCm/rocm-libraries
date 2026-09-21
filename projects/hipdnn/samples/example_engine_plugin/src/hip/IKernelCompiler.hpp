// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ICompiledProgram.hpp"

#include <memory>
#include <string>
#include <vector>

namespace example_provider
{

/// Interface for compiling GPU kernel source files at runtime.
///
/// Implementations load embedded kernel source by filename and compile
/// it with the specified options (e.g., --offload-arch=gfxNNN).
class IKernelCompiler
{
public:
    virtual ~IKernelCompiler() = default;

    virtual std::unique_ptr<ICompiledProgram> compile(const std::string& kernelFileName,
                                                      const std::vector<std::string>& options) const
        = 0;
};

} // namespace example_provider
