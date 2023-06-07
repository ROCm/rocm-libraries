
#include <concepts>
#include <string>
#include <vector>

#include <rocRoller/CodeGen/Instruction.hpp>
#include <rocRoller/GPUArchitecture/GPUArchitecture.hpp>
#include <rocRoller/Scheduling/Observers/FunctionalUnit/MFMAObserver.hpp>

namespace rocRoller
{
    namespace Scheduling
    {
        MFMAObserver::MFMAObserver() {}

        MFMAObserver::MFMAObserver(std::shared_ptr<Context> ctx)
            : m_context(ctx)
        {
        }

        bool MFMAObserver::isMFMAInstruction(Instruction const& inst) const
        {
            return inst.getOpCode().find("mfma") != std::string::npos;
        }

        InstructionStatus MFMAObserver::peek(Instruction const& inst) const
        {
            InstructionStatus rv;
            if(isMFMAInstruction(inst))
                rv.stallCycles = m_remainingCycles;
            return rv;
        }

        void MFMAObserver::modify(Instruction& inst) const
        {
            if(m_remainingCycles > 0 && !inst.isCommentOnly()
               && Settings::Get(Settings::LogLvl) >= LogLevel::Debug)
                inst.addComment(concatenate("MFMA remaining: ", m_remainingCycles));
        }

        void MFMAObserver::observe(Instruction const& inst)
        {
            if(isMFMAInstruction(inst))
            {
                auto info
                    = m_context.lock()->targetArchitecture().GetInstructionInfo(inst.getOpCode());

                m_remainingCycles = info.getLatency();
            }
            else
            {
                m_remainingCycles = std::max(0, m_remainingCycles - inst.numExecutedInstructions());
            }
        }

    }

}
