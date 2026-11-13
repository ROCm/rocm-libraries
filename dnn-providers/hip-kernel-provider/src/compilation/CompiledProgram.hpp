// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "DeviceResolvedKernel.hpp"
#include "ICompiledProgram.hpp"
#include "Kernel.hpp"
#include "Program.hpp"

#include <memory>
#include <string>
#include <utility>

namespace hip_kernel_provider::compilation
{

class CompiledProgram : public ICompiledProgram
{
public:
    /// A program bound to the device it was loaded on: its kernels launch there and
    /// refuse a stream from another device.
    explicit CompiledProgram(std::shared_ptr<Program> program)
        : _program(std::move(program))
    {
    }

    /// A program that follows the launch: its kernels launch on the device of the stream
    /// they are given, loading the program there through @p loader on first use.
    /// @p program is the one already loaded, which answers its own device without a load.
    ///
    /// @param builtArch The architecture the program was compiled for; a launch on a
    ///                  device of another architecture is refused.
    CompiledProgram(std::shared_ptr<Program> program, std::string builtArch, ProgramLoader loader)
        : _program(std::move(program))
        , _builtArch(std::move(builtArch))
        , _loader(std::move(loader))
    {
    }

    std::unique_ptr<IRunnableKernel> getKernel(const std::string& kernelName) const override
    {
        if(!_loader)
        {
            return std::make_unique<Kernel>(*_program, kernelName);
        }

        auto resolver = std::make_unique<ProgramKernelResolver>(_builtArch, kernelName, _loader);
        resolver->seed(_program->deviceOrdinal(),
                       DeviceResolvedKernel::Resolved{_program, _program->getKernel(kernelName)});
        return std::make_unique<DeviceResolvedKernel>(kernelName, std::move(resolver));
    }

private:
    std::shared_ptr<Program> _program;
    std::string _builtArch;
    ProgramLoader _loader;
};

} // namespace hip_kernel_provider::compilation
