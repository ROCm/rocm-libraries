#include <concepts>
#include <string>
#include <vector>

#include <rocRoller/CodeGen/Instruction.hpp>
#include <rocRoller/GPUArchitecture/GPUArchitecture.hpp>
#include <rocRoller/GPUArchitecture/GPUInstructionInfo.hpp>
#include <rocRoller/Scheduling/Observers/FunctionalUnit/WMMAObserver.hpp>

namespace rocRoller
{
    namespace Scheduling
    {
        WMMAObserver::WMMAObserver() {}

        WMMAObserver::WMMAObserver(ContextPtr ctx)
            : m_context(ctx)
        {
        }

        bool WMMAObserver::isWMMAInstruction(const Instruction& inst) const
        {
            return GPUInstructionInfo::isWMMA(inst.getOpCode());
        }

        InstructionStatus WMMAObserver::peek(const Instruction& inst) const
        {
            InstructionStatus rv;
            if(isWMMAInstruction(inst))
                rv.stallCycles = m_remainingCycles;
            return rv;
        }

        void WMMAObserver::modify(Instruction& inst) const
        {
            if(m_remainingCycles > 0 && !inst.isCommentOnly()
               && Settings::Get(Settings::LogLvl) >= LogLevel::Debug)
                inst.addComment(concatenate("WMMA remaining: ", m_remainingCycles));
        }

        void WMMAObserver::observe(const Instruction& inst)
        {
            if(isWMMAInstruction(inst))
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
