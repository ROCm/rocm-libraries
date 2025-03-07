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

#include "TestKernels.hpp"

AssemblyTestKernel::AssemblyTestKernel(rocRoller::ContextPtr context)
    : m_context(context)
{
}

void AssemblyTestKernel::doGenerate()
{
    generate();
    m_generated = true;
}

std::vector<char> const& AssemblyTestKernel::getAssembledKernel()
{
    if(!m_generated)
        doGenerate();

    if(m_assembledKernel.empty())
    {
        m_assembledKernel = m_context->instructions()->assemble();
    }

    return m_assembledKernel;
}

std::shared_ptr<rocRoller::ExecutableKernel> AssemblyTestKernel::getExecutableKernel()
{
    if(!m_generated)
        doGenerate();

    if(!m_executableKernel)
    {
        m_executableKernel = m_context->instructions()->getExecutableKernel();
    }

    return m_executableKernel;
}

rocRoller::ContextPtr AssemblyTestKernel::getContext() const
{
    return m_context;
}
