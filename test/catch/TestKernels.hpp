
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
