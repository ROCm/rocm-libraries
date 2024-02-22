#pragma once

#include <rocRoller/Scheduling/Observers/WaitState/WaitStateObserver.hpp>

namespace rocRoller
{
    namespace Scheduling
    {
        /**
         * @brief 90a rules for XDLOP Reads of Src C (WAR)
         *
         * | Arch | 1st Inst                    | 2nd Inst  | NOPs |
         * | ---- | --------------------------- | --------- | ---- |
         * | 90a  | v_mfma* read SrcC (2 pass)  | v_* write | 1    |
         * | 90a  | v_mfma* read SrcC (8 pass)  | v_* write | 11   |
         * | 90a  | v_mfma* read SrcC (16 pass) | v_* write | 19   |
         *
         */
            void observeHazard(Instruction const& inst) override;

            static bool required(ContextPtr context)
            {
                return context->targetArchitecture().target().getVersionString() == "gfx90a";
            }

            int         getMaxNops(std::shared_ptr<InstructionRef> inst) const;
            bool        trigger(std::shared_ptr<InstructionRef> inst) const;
            bool        writeTrigger() const;
            int         getNops(Instruction const& inst) const;
            std::string getComment() const
            {
                return "XDL Read Hazard";
            }

        private:
            std::unordered_map<int, int> m_latencyAndNops = {{2, 1}, {8, 11}, {16, 19}};
        };

        static_assert(CWaitStateObserver<XDLReadSrcC90a>);
    }
}
