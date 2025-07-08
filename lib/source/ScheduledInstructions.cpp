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

#include <rocRoller/Assemblers/Assembler.hpp>
#include <rocRoller/AssemblyKernel.hpp>
#include <rocRoller/Context.hpp>
#include <rocRoller/ScheduledInstructions.hpp>

namespace rocRoller
{
    ScheduledInstructions::ScheduledInstructions(ContextPtr ctx)
        : m_context(ctx)
    {
    }

    void ScheduledInstructions::clear()
    {
        m_instructionstream = std::ostringstream();
    }

    std::shared_ptr<ExecutableKernel> ScheduledInstructions::getExecutableKernel()
    {
        auto context = m_context.lock();

        std::shared_ptr<ExecutableKernel> result = std::make_shared<ExecutableKernel>();
        result->loadKernel(
            toString(), context->targetArchitecture().target(), context->kernel()->kernelName());

        return result;
    }

    std::string ScheduledInstructions::toString() const
    {
        return m_instructionstream.str();
    }

    std::vector<char> ScheduledInstructions::assemble() const
    {
        auto context   = m_context.lock();
        auto assembler = Assembler::Get();

        return assembler->assembleMachineCode(
            toString(), context->targetArchitecture().target(), context->kernel()->kernelName());
    }

    void ScheduledInstructions::schedule(const Instruction& instruction)
    {
        auto context = m_context.lock();
        instruction.toStream(m_instructionstream, context->kernelOptions().logLevel);
    }

}
