#pragma once

#include <rocRoller/Scheduling/Observers/WaitState/WaitStateObserver.hpp>

namespace rocRoller
{
    namespace Scheduling
    {
        /**
         * @brief 94x rules for certain VALU Write of SGPR or VCC
         *
         * | Arch | 1st Inst                       | 2nd Inst                      | NOPs |
         * | ---- | ------------------------------ | ----------------------------- | ---- |
         * | 94x  | v_readlane write SGPR/VCC      | v_* read as constant          | 2    |
         * | 94x  | v_readlane write SGPR/VCC      | v_readlane read as laneselect | 4    |
         * | 94x  | v_readfirstlane write SGPR/VCC | v_* read as constant          | 2    |
         * | 94x  | v_readfirstlane write SGPR/VCC | v_readlane read as laneselect | 4    |
         * | 94x  | v_cmp_* write SGPR/VCC         | v_* read as constant          | 2    |
         * | 94x  | v_cmp_* write SGPR/VCC         | v_readlane read as laneselect | 4    |
         * | 94x  | v_add*_i* write SGPR/VCC       | v_* read as constant          | 2    |
         * | 94x  | v_add*_i* write SGPR/VCC       | v_readlane read as laneselect | 4    |
         * | 94x  | v_add*_u* write SGPR/VCC       | v_* read as constant          | 2    |
         * | 94x  | v_add*_u* write SGPR/VCC       | v_readlane read as laneselect | 4    |
         * | 94x  | v_sub*_i* write SGPR/VCC       | v_* read as constant          | 2    |
         * | 94x  | v_sub*_i* write SGPR/VCC       | v_readlane read as laneselect | 4    |
         * | 94x  | v_sub*_u* write SGPR/VCC       | v_* read as constant          | 2    |
         * | 94x  | v_sub*_u* write SGPR/VCC       | v_readlane read as laneselect | 4    |
         * | 94x  | v_div_scale* write SGPR/VCC    | v_* read as constant          | 2    |
         * | 94x  | v_div_scale* write SGPR/VCC    | v_readlane read as laneselect | 4    |
         *
         * NOTE: If the SGPR/VCC is read as a carry in these cases, 0 NOPs are required.
         *
         */
        class VALUWriteSGPRVCC94x : public WaitStateObserver<VALUWriteSGPRVCC94x>
        {
        public:
            VALUWriteSGPRVCC94x() {}
            VALUWriteSGPRVCC94x(ContextPtr context)
                : WaitStateObserver<VALUWriteSGPRVCC94x>(context){};

            static bool required(ContextPtr context)
            {
                auto arch = context->targetArchitecture().target().getVersionString();
                return arch == "gfx940" || arch == "gfx941" || arch == "gfx942" || arch == "gfx950";
            }

            int         getMaxNops(Instruction const& inst) const;
            bool        trigger(Instruction const& inst) const;
            bool        writeTrigger() const;
            int         getNops(Instruction const& inst) const;
            std::string getComment() const
            {
                return "VALU Write SGPR/VCC Hazard";
            }

        private:
            int const m_maxNops = 4;
        };

        static_assert(CWaitStateObserver<VALUWriteSGPRVCC94x>);
    }
}
