#pragma once

#include <memory>
#include <string>
#include <vector>

#include <rocRoller/Assemblers/Assembler_fwd.hpp>
#include <rocRoller/GPUArchitecture/GPUArchitectureTarget.hpp>

namespace rocRoller
{
    std::ostream& operator<<(std::ostream&, AssemblerType);

    class Assembler
    {
    public:
        using Argument = AssemblerType;

        static const std::string Basename;

        static AssemblerPtr Get();

        virtual std::string name() const = 0;

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
        virtual std::vector<char> assembleMachineCode(const std::string&           machineCode,
                                                      const GPUArchitectureTarget& target,
                                                      const std::string&           kernelName)
            = 0;

        virtual std::vector<char> assembleMachineCode(const std::string&           machineCode,
                                                      const GPUArchitectureTarget& target)
            = 0;
    };
}
