// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "IRunnableKernel.hpp"

#include <memory>
#include <string>

namespace example_provider
{

/// Interface for a compiled GPU program (module).
///
/// Represents a compiled HIP module from which kernel functions
/// can be extracted by name.
class ICompiledProgram
{
public:
    virtual ~ICompiledProgram() = default;

    virtual std::unique_ptr<IRunnableKernel>
        getRunnableKernel(const std::string& kernelFunctionName) const = 0;
};

} // namespace example_provider
