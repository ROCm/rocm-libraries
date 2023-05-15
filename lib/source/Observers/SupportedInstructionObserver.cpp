#include <rocRoller/Scheduling/Observers/SupportedInstructionObserver.hpp>

namespace rocRoller
{
    namespace Scheduling
    {

        InstructionStatus SupportedInstructionObserver::peek(Instruction const& inst) const
        {
            return InstructionStatus();
        };

        void SupportedInstructionObserver::modify(Instruction& inst) const
        {
            // No modifications
        }

        void SupportedInstructionObserver::observe(Instruction const& inst)
        {
            auto instruction = inst.getOpCode();
            // If instruction has an opcode.
            if(!instruction.empty())
            {
                auto        context      = m_context.lock();
                auto const& architecture = context->targetArchitecture();

                AssertFatal(architecture.HasInstructionInfo(instruction),
                            "Instruction '" + instruction + "' is not supported.");
            }
        }
    }
}
