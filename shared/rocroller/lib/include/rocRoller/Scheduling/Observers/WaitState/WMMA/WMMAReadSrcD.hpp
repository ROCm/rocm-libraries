// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <rocRoller/Scheduling/Observers/WaitState/WaitStateObserver.hpp>

namespace rocRoller
{
    namespace Scheduling
    {
        /**
         * @brief 120X rules for WMMA Reads of D
         *
         * | Arch  | 1st Inst                    | 2nd Inst | NOPs |
         * | ----- | --------------------------- | -------- | ---- |
         * | 120x  | v_wmma* read SrcC (8 pass)  | v_* read | 4    |
         * | 120x  | v_wmma* read SrcC (16 pass) | v_* read | 8    |
         *
         */
        class WMMAReadSrcD : public WaitStateObserver<WMMAReadSrcD>
        {
        public:
            WMMAReadSrcD() {}
            WMMAReadSrcD(ContextPtr context)
                : WaitStateObserver<WMMAReadSrcD>(context){};

            /**
             * Overriden as we need to target src C only
             */
            void observeHazard(Instruction const& inst) override;

            constexpr static bool required(GPUArchitectureTarget const& target)
            {
                return target.isRDNA4GPU();
            }

            int                   getMaxNops(Instruction const& inst) const;
            bool                  trigger(Instruction const& inst) const;
            static constexpr bool writeTrigger()
            {
                return false;
            }
            int         getNops(Instruction const& inst) const;
            std::string getComment() const
            {
                return "WMMA Write Hazard (VALU)";
            }

        private:
            std::unordered_map<int, int> m_latencyAndNops = {{8, 4}, {16, 8}};
        };

        static_assert(CWaitStateObserver<WMMAReadSrcD>);
    }
}
