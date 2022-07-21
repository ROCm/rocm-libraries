#include "ScheduledInstructions.hpp"
#include "Assembler.hpp"
#include "AssemblyKernel.hpp"
#include "Context.hpp"

namespace rocRoller
{
    ScheduledInstructions::ScheduledInstructions(std::shared_ptr<Context> ctx)
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
        auto      context = m_context.lock();
        Assembler assembler;
        return assembler.assembleMachineCode(
            toString(), context->targetArchitecture().target(), context->kernel()->kernelName());
    }

    void ScheduledInstructions::schedule(const Instruction& instruction)
    {
        auto context = m_context.lock();
        instruction.toStream(m_instructionstream, context->kernelOptions().logLevel);
    }

}
