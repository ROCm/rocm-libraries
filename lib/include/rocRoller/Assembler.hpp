#pragma once

#include "GPUArchitecture/GPUArchitectureTarget.hpp"
#include <vector>

namespace rocRoller
{
    class Assembler
    {
    public:
        Assembler();
        ~Assembler();
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
        std::vector<char> assembleMachineCode(std::string                  machineCode,
                                              const GPUArchitectureTarget& target,
                                              std::string                  kernelName = "");

    private:
        void assemble(const char* machineCode,
                      const char* target,
                      const char* features,
                      const char* output);
        void link(const char* input, const char* output);
    };
}

#include "Assembler_impl.hpp"
