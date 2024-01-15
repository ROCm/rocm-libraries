#pragma once

#include "Assembler.hpp"

#include "GPUArchitecture/GPUArchitectureTarget.hpp"
#include <vector>

namespace rocRoller
{
    class InProcessAssembler : public Assembler
    {
    public:
        InProcessAssembler();
        ~InProcessAssembler();

        using Base = Assembler;

        static const std::string Name;

        static bool Match(Argument arg);

        static AssemblerPtr Build(Argument arg);

        std::string name() const override;
        /**
         * @brief Assemble a string of machine code. The resulting object code will be returned
         * as a vector of charaters.
         *
         * If the environment variable ROCROLLER_SAVE_ASSEMBLY is set to 1, it will save the assembly
         * file to the working directory, where the file name is 'kernelName_target.s' (where all
         * colons are replaced with dashes).
         *
         * @param machineCode Machine code to assemble
         * @param target The target architecture
         * @param kernelName The name of the kernel (default is "")
         * @return std::vector<char>
         */
        std::vector<char> assembleMachineCode(const std::string&           machineCode,
                                              const GPUArchitectureTarget& target,
                                              const std::string&           kernelName) override;

        std::vector<char> assembleMachineCode(const std::string&           machineCode,
                                              const GPUArchitectureTarget& target) override;

    private:
        static void assemble(const char* machineCode,
                             const char* target,
                             const char* featureString,
                             const char* output);
        static void link(const char* input, const char* output);
    };
}
