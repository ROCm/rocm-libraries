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
    explicit HipCompiledProgram(std::shared_ptr<HipProgram> program)
        : _program(std::move(program))
    {
    }

    std::unique_ptr<IRunnableKernel> getKernel(const std::string& kernelName) const override
    {
        return std::make_unique<HipKernel>(*_program, kernelName);
    }

private:
    std::shared_ptr<HipProgram> _program;
};

} // namespace example_plugin
