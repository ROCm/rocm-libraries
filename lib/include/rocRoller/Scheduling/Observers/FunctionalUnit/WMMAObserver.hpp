#pragma once

#include <concepts>
#include <string>
#include <vector>

#include <rocRoller/Scheduling/Scheduling.hpp>

namespace rocRoller
{
    namespace Scheduling
    {

        class WMMAObserver
        {
        public:
            WMMAObserver();
            WMMAObserver(ContextPtr ctx);

            InstructionStatus peek(Instruction const& inst) const;

            void modify(Instruction& inst) const;

            void observe(Instruction const& inst);

            constexpr static bool required(GPUArchitectureTarget const& target)
            {
                return true;
            }

        private:
            bool isWMMAInstruction(Instruction const& inst) const;

            int m_remainingCycles = 0;

            std::weak_ptr<Context> m_context;
        };

        static_assert(CObserverConst<WMMAObserver>);

    }
}
