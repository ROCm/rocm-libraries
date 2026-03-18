// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <rocRoller/Assemblers/Assembler.hpp>
#include <rocRoller/GPUArchitecture/GPUArchitectureTarget.hpp>

#include <functional>
#include <vector>

namespace rocRoller
{
    class SubprocessAssembler : public Assembler
    {
    public:
        using Base = Assembler;

        inline static const std::string Name = "SubprocessAssembler";

        static bool Match(Argument arg);

        static AssemblerPtr Build(Argument arg);

        std::string name() const override;

        std::vector<char> assembleMachineCode(const std::string&           machineCode,
                                              const GPUArchitectureTarget& target,
                                              const std::string&           kernelName) override;

        std::vector<char> assembleMachineCode(const std::string&           machineCode,
                                              const GPUArchitectureTarget& target) override;

    private:
        std::tuple<int, std::string> execute(std::string const& command);
        void executeChecked(std::string const& command, std::function<void()> const& cleanupCall);

        std::string makeTempFolder();
    };
}
