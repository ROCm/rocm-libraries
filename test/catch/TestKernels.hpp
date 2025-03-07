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

#include <rocRoller/AssemblyKernel.hpp>
#include <rocRoller/Context.hpp>
#include <rocRoller/KernelArguments.hpp>

/**
 * A test kernel which is generated manually, usually by yielding instructions.  It should
 * not use a kernel graph or command.  When those tests are ported over, new TestKernel classes
 * should be written.
 *
 * To use this,
 * 1: Subclass AssemblyTestKernel.
 *   - Add any parameterization as fields and arguments to the constructor of the subclass.
 *   - Implement generate() which should leave m_context with a fully generated kernel.
 * 2. Instantiate subclass
 *   - Invoke the call operator to launch the kernel directly, or
 *   - check getAssembledKernel() and/or m_context.output() as appropriate.
 */
class AssemblyTestKernel
{
public:
    AssemblyTestKernel(rocRoller::ContextPtr context);

    /**
     * Launch the kernel.  Use `invocation` to determine launch bounds; Remaining
     * arguments will be passed as kernel arguments.
     */
    template <typename... Args>
    void operator()(rocRoller::KernelInvocation const& invocation, Args const&... args);

    /**
     * Get the assembled code object as bytes.
     */
    std::vector<char> const& getAssembledKernel();

    /**
     * Get an associated ExecutableKernel object which can be used to launch the kernel.
     */
    std::shared_ptr<rocRoller::ExecutableKernel> getExecutableKernel();

    rocRoller::ContextPtr getContext() const;

protected:
    virtual void generate() = 0;

    void doGenerate();

    template <int Idx = 0, typename... Args>
    void appendArgs(rocRoller::KernelArguments& kargs, std::tuple<Args...> const& args);

    rocRoller::ContextPtr                        m_context;
    bool                                         m_generated = false;
    std::vector<char>                            m_assembledKernel;
    std::shared_ptr<rocRoller::ExecutableKernel> m_executableKernel;
};

#include "TestKernels_impl.hpp"
