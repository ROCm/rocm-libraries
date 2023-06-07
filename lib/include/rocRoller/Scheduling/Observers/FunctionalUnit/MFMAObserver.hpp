
#pragma once

#include <concepts>
#include <string>
#include <vector>

#include "../../Scheduling.hpp"

namespace rocRoller
{
    namespace Scheduling
    {

        class MFMAObserver
        {
        public:
            MFMAObserver();
            MFMAObserver(std::shared_ptr<Context> ctx);

            InstructionStatus peek(Instruction const& inst) const;

            void modify(Instruction& inst) const;

            void observe(Instruction const& inst);

            static bool required(std::shared_ptr<Context>)
            {
                return true;
            }

        private:
            bool isMFMAInstruction(Instruction const& inst) const;

            int m_remainingCycles = 0;

            std::weak_ptr<Context> m_context;
        };

        static_assert(CObserver<MFMAObserver>);

    }
}
