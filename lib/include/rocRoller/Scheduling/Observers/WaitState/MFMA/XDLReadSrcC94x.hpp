#pragma once

#include <rocRoller/Scheduling/Observers/WaitState/WaitStateObserver.hpp>

namespace rocRoller
{
    namespace Scheduling
    {
        /**
         * @brief 94x rules for XDLOP Reads of Src C (WAR)
         *
         * | Arch | 1st Inst                    | 2nd Inst  | NOPs |
         * | ---- | --------------------------- | --------- | ---- |
         * | 94x  | v_mfma* read SrcC (2 pass)  | v_* write | 1    |
         * | 94x  | v_mfma* read SrcC (4 pass)  | v_* write | 3    |
         * | 94x  | v_mfma* read SrcC (8 pass)  | v_* write | 7    |
         * | 94x  | v_mfma* read SrcC (16 pass) | v_* write | 15   |
         *
         */
            void observeHazard(Instruction const& inst) override;

            constexpr static bool required(GPUArchitectureTarget const& target)
            {
                return target.isCDNA3GPU() || target.isCDNA35GPU();
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
                return "XDL Read Hazard";
            }

        private:
            std::unordered_map<int, int> m_latencyAndNops = {{2, 1}, {4, 3}, {8, 7}, {16, 15}};
        };

        static_assert(CWaitStateObserver<XDLReadSrcC94x>);
    }
}
