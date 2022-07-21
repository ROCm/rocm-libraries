
#pragma once

#include "../Scheduling.hpp"

#include "../../Context.hpp"
#include "../../GPUArchitecture/GPUInstructionInfo.hpp"

namespace rocRoller
{
    namespace Scheduling
    {
        class AllocatingObserver
        {
        public:
            AllocatingObserver(std::shared_ptr<Context> context)
                : m_context(context)
            {
            }

            InstructionStatus peek(Instruction const& inst) const
            {
                return {};
            }

            //> Add any waitcnt or nop instructions needed before `inst` if it were to be scheduled now.
            //> Throw an exception if it can't be scheduled now.
            void modify(Instruction& inst) const
            {
                inst.allocateNow();
            }

            //> This instruction _will_ be scheduled now, record any side effects.
            InstructionStatus observe(Instruction const& inst)
            {
                return {};
            }

        private:
            std::weak_ptr<Context> m_context;
        };

        static_assert(CObserver<AllocatingObserver>);
    }
}
