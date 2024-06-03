#pragma once

#include <rocRoller/Scheduling/Observers/WaitState/WaitStateObserver.hpp>

namespace rocRoller
{
    namespace Scheduling
    {
        /**
         * @brief 94x rules for v_cmpx_* writes
         *
         * | Arch | 1st Inst  | 2nd Inst          | NOPs |
         * | ---- | --------- | ----------------- | ---- |
         * | 94x  | v_cmpx_*  | v_* read EXEC     | 2    |
         * | 94x  | v_cmpx_*  | v_readlane_*      | 4    |
         * | 94x  | v_cmpx_*  | v_readfirstlane_* | 4    |
         * | 94x  | v_cmpx_*  | v_writelane_*     | 4    |
         * | 94x  | v_cmpx_*  | v_*               | 0    |
         *
         */
            void observeHazard(Instruction const& inst) override;

            int         getMaxNops(Instruction const& inst) const;
            bool        trigger(Instruction const& inst) const;
            bool        writeTrigger() const;
            int         getNops(Instruction const& inst) const;
            std::string getComment() const
            {
                return "v_cmpx Write Hazard";
            }

        private:
            int const m_maxNops = 4;
        };

        static_assert(CWaitStateObserver<VCMPXWrite94x>);
    }
}
