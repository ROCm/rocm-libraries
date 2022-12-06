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
            InstructionStatus observe(Instruction const& inst);

            static bool required(std::shared_ptr<Context> context)
            {
                return context->targetArchitecture().target().getVersionString() == "gfx908";
            }

            int  getMaxNops(std::shared_ptr<InstructionRef> inst) const;
            bool trigger(std::shared_ptr<InstructionRef> inst) const;
            bool writeTrigger() const;
            int  getNops(Instruction const& inst) const;

        private:
            std::unordered_map<int, int> m_latencyAndNops = {{2, 0}, {8, 5}, {16, 13}};
        };

        static_assert(CWaitStateObserver<XDLReadSrcC908>);
    }
}
