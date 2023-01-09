#pragma once

#include <rocRoller/CodeGen/InstructionRef_fwd.hpp>

#include <memory>

namespace rocRoller
{
    namespace Scheduling
    {
        class WaitStateHazardCounter
        {
        public:
            WaitStateHazardCounter() {}
            WaitStateHazardCounter(int                             maxRequiredNops,
                                   std::shared_ptr<InstructionRef> inst,
                                   bool                            written)
                : m_counter(maxRequiredNops)
                , m_inst(inst)
                , m_written(written)
            {
            }

            /**
             * @brief Decrements the counter
             *
             * @param nops Number of NOPs associated with the decrement
             */
            void decrement(int nops)
            {
                m_counter--;
                m_counter -= nops;
            }

            /**
             * @brief Determine if the counter has reach the end of life
             *
             * @return true if this still represents a hazard
             * @return false if the counter can be safely removed
             */
            bool stillAlive() const
            {
                return m_counter > 0;
            }

            std::shared_ptr<InstructionRef> getInstructionRef() const
            {
                return m_inst;
            }

            int getRequiredNops() const
            {
                return m_counter;
            }

            bool regWasWritten() const
            {
                return m_written;
            }

            bool regWasRead() const
            {
                return !m_written;
            }

        private:
            int                             m_counter = 0;
            std::shared_ptr<InstructionRef> m_inst;
            bool                            m_written = false;
        };
    }
}
