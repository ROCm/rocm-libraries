// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "HipKernel.hpp"
#include "HipProgram.hpp"
#include "ICompiledProgram.hpp"

#include <memory>
#include <string>

namespace example_plugin
{

/// Concrete ICompiledProgram wrapping a HipProgram.
///
/// Extracts kernel functions from the loaded HIP module by name.
class HipCompiledProgram : public ICompiledProgram
{
public:
    explicit HipCompiledProgram(std::unique_ptr<HipProgram> program)
        : _program(std::move(program))
    {
    }

    std::unique_ptr<IRunnableKernel> getKernel(const std::string& kernelName) const override
    {
        hipFunction_t func = _program->getKernel(kernelName);
        return std::make_unique<HipKernel>(func);
    }

private:
    std::unique_ptr<HipProgram> _program;
};

} // namespace example_plugin
