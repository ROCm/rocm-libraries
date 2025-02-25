
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
