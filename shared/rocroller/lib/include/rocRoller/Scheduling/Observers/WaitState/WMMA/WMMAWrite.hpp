// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <rocRoller/Scheduling/Observers/WaitState/WaitStateObserver.hpp>

namespace rocRoller
{
    namespace Scheduling
    {
        /**
         * @brief 120X rules for WMMA Write Hazards.
         *
         * | Arch  | 1st Inst | 2nd Inst            | NOPs |
         * | ----- | -------- | ------------------- | ---- |
         * | 120x  | v_wmma*  | v_wmma* read SrcA/B | 1    |
         *
         */
        class WMMAWrite : public WaitStateObserver<WMMAWrite>
        {
        public:
            WMMAWrite() {}
            WMMAWrite(ContextPtr context)
                : WaitStateObserver<WMMAWrite>(context){};

            constexpr static bool required(const GPUArchitectureTarget& target)
            {
                return target.isRDNA4GPU();
            }

            int                   getMaxNops(const Instruction& inst) const;
            bool                  trigger(const Instruction& inst) const;
            static constexpr bool writeTrigger()
            {
                return true;
            }
            int         getNops(const Instruction& inst) const;
            std::string getComment() const
            {
                return "WMMA Write Hazard";
            }

        private:
            const int m_maxNops = 1;
        };

        static_assert(CWaitStateObserver<WMMAWrite>);
    }
}
