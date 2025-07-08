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

#include <sstream>
#include <vector>

#include <rocRoller/CodeGen/Instruction_fwd.hpp>
#include <rocRoller/Context_fwd.hpp>
#include <rocRoller/ExecutableKernel.hpp>

namespace rocRoller
{
    class ScheduledInstructions
    {
    public:
        explicit ScheduledInstructions(ContextPtr ctx);
        ~ScheduledInstructions() = default;

        void schedule(const Instruction& instruction);

        std::string toString() const;
        /**
         * @brief Get the Executable Kernel object for the currently scheduled instructions.
         *
         * @return std::shared_ptr<ExecutableKernel> Returns a pointer to the Executable Kernel
         */
        std::shared_ptr<ExecutableKernel> getExecutableKernel();

        /**
         * @brief Assemble the currently scheduled instructions.
         *
         * @return std::vector<char> The binary representation of the assembled instructions.
         */
        std::vector<char> assemble() const;
        void              clear();

    private:
        std::ostringstream     m_instructionstream;
        std::weak_ptr<Context> m_context;
    };

}
