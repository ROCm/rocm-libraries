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

/**
 */

#pragma once

#include <string>
#include <unordered_map>

#include <rocRoller/AssemblyKernelArgument.hpp>
#include <rocRoller/AssemblyKernel_fwd.hpp>
#include <rocRoller/CodeGen/Instruction.hpp>
#include <rocRoller/InstructionValues/Register_fwd.hpp>
#include <rocRoller/Utilities/Generator.hpp>

namespace rocRollerTest
{
    class ArgumentLoaderTest_loadArgExtra_Test;
}

namespace rocRoller
{
    /**
     * Generates code to load argument values from the kernel argument buffer into SGPRs.
     * Supports two main strategies: all-at-once, and on-demand.
     *
     * All-at-once can be faster since we can use wider load instructions, but so far this
     * requires allocating all the argument SGPRs in one block so they can't be freed one by one
     * to allow reuse.
     *
     * On-demand is more flexible, but also potentially slower as it will require more load
     * instructions as well as possibly more synchronization.
     *
     */
    class ArgumentLoader
    {
    public:
        ArgumentLoader(AssemblyKernelPtr kernel);

        /**
         * Loads all arguments into a single allocation of SGPRs.  Uses the widest load
         * instructions possible given alignment constraints.
         */
        Generator<Instruction> loadAllArguments();

        /**
         * Obtain the `Value` for a given argument.  If the argument has not been loaded yet,
         * emits instructions to load it into SGPRs.
         *
         * @param argName
         * @param value
         */
        Generator<Instruction> getValue(std::string const& argName, Register::ValuePtr& value);

        void releaseArgument(std::string const& argName);
        void releaseAllArguments();

        Generator<Instruction>
            loadRange(int offset, int sizeBytes, Register::ValuePtr& value) const;

    private:
        friend class rocRollerTest::ArgumentLoaderTest_loadArgExtra_Test;

        std::weak_ptr<Context> m_context;

        AssemblyKernelPtr m_kernel;

        Generator<Instruction> loadArgument(std::string const& argName);
        Generator<Instruction> loadArgument(AssemblyKernelArgument const& arg);

        /// Call the one from the AssemblyKernel instead.
        Register::ValuePtr argumentPointer() const;

        std::unordered_map<std::string, Register::ValuePtr> m_loadedValues;
    };

    /**
     * Picks the widest load instruction to load some of argPtr[offset:endOffset]
     * into s[*beginReg:*endReg]. Returns the width of that load instruction in
     * bytes, or 0 if offset == endOffset.
     *
     * Reasons to decrease the width of the load:
     *  - offset is not aligned to the width of the load
     *  - endOffset-offset is less than the width of the load
     *  - *beginReg is not aligned to the width of the load
     *  - destination registers are not contiguous
     *
     * @return int Width of the load in bytes
     */
    template <typename Iter, typename End>
    inline int PickInstructionWidthBytes(int offset, int endOffset, Iter beginReg, End endReg);

}

#include <rocRoller/CodeGen/ArgumentLoader_impl.hpp>
