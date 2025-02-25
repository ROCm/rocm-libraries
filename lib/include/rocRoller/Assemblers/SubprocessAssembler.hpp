#pragma once

#include "Assembler.hpp"

#include <rocRoller/GPUArchitecture/GPUArchitectureTarget.hpp>
#include <vector>

namespace rocRoller
{
    class SubprocessAssembler : public Assembler
    {
    public:
        SubprocessAssembler();
        ~SubprocessAssembler();

        using Base = Assembler;

        static const std::string Name;

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
        void                         executeChecked(std::string const& command);

        std::string makeTempFolder();
    };
}
