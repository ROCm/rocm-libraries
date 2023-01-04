#pragma once

#include <rocRoller/CodeGen/Instruction.hpp>
#include <rocRoller/Scheduling/Scheduling.hpp>

namespace rocRoller
{
    namespace Scheduling
    {
        /**
         * @brief This observer maintains register map wait state hazard counters.
         *
         * This observer is responsible for maintaining the state of the wait state
         * hazard register map. Every time an instruction is scheduled, this observer
         * updates all of the entries in the map by:
         *   - decrementing existing counters by the nop amount in the instruction
         *   - if the register no longer presents the threat of a particular hazard, remove that counter from the register's collection
         *   - if the register no longer has hazard counters, remove the register's entry from the map
         *
         * Note: This observer should observe instructions before other observers that might modify the
         * register map, so that no hazard counters are prematurely decremented.
         */
        class RegisterMapObserver
        {
        public:
            RegisterMapObserver() {}
            RegisterMapObserver(std::shared_ptr<Context> context)
                : m_context(context){};

            InstructionStatus peek(Instruction const& inst) const;
            void              modify(Instruction& inst) const;
            void              observe(Instruction const& inst);

            static bool required(std::shared_ptr<Context> context)
            {
                return true;
            }

        private:
            std::weak_ptr<Context> m_context;
        };
    }
}
