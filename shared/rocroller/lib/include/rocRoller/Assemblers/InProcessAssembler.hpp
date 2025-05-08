/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2024-2025 AMD ROCm(TM) Software
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#pragma once

#include <rocRoller/Assemblers/Assembler.hpp>
#include <rocRoller/GPUArchitecture/GPUArchitectureTarget.hpp>

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
