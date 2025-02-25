#pragma once

#include <rocRoller/Scheduling/Observers/WaitState/WaitStateObserver.hpp>

namespace rocRoller
{
    namespace Scheduling
    {
        /**
         * @brief 908 rules for XDLOP Reads of Src C (WAR)
         *
         * | Arch | 1st Inst                    | 2nd Inst                         | NOPs |
         * | ---- | --------------------------- | -------------------------------- | ---- |
         * | 908  | v_mfma* read SrcC (2 pass)  | v_accvgpr_write write overlapped | 0    |
         * | 908  | v_mfma* read SrcC (8 pass)  | v_accvgpr_write write overlapped | 5    |
         * | 908  | v_mfma* read SrcC (16 pass) | v_accvgpr_write write overlapped | 13   |
         *
         */
            void observeHazard(Instruction const& inst) override;

            constexpr static bool required(GPUArchitectureTarget const& target)
            {
                return target.isCDNA1GPU();
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
            std::unordered_map<int, int> m_latencyAndNops = {{2, 0}, {8, 5}, {16, 13}};
        };

        static_assert(CWaitStateObserver<XDLReadSrcC908>);
    }
}
