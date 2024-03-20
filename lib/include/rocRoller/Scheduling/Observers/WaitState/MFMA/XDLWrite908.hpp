#pragma once

#include <rocRoller/Scheduling/Observers/WaitState/WaitStateObserver.hpp>

namespace rocRoller
{
    namespace Scheduling
    {
        /**
         * @brief 908 rules for XDL Write Hazards.
         *
         * | Arch | 1st Inst                | 2nd Inst                     | NOPs |
         * | ---- | ----------------------- | ---------------------------- | ---- |
         * | 908  | v_mfma* write           | v_mfma* read SrcC same       | 0    |
         * | 908  | v_mfma* write           | v_mfma* read SrcC overlapped | 2    |
         * | 908  | v_mfma* write           | v_mfma* read SrcA/B          | 4    |
         * | 908  | v_mfma* write (2 pass)  | v_accvgpr_read read          | 4    |
         * | 908  | v_mfma* write (8 pass)  | v_accvgpr_read read          | 10   |
         * | 908  | v_mfma* write (16 pass) | v_accvgpr_read read          | 18   |
         * | 908  | v_mfma* write (2 pass)  | v_accvgpr_write write        | 1    |
         * | 908  | v_mfma* write (8 pass)  | v_accvgpr_write write        | 7    |
         * | 908  | v_mfma* write (16 pass) | v_accvgpr_write write        | 15   |
         *
         */
        class XDLWrite908 : public WaitStateObserver<XDLWrite908>
        {
        public:
            XDLWrite908() {}
            XDLWrite908(ContextPtr context)
                : WaitStateObserver<XDLWrite908>(context){};

            static bool required(ContextPtr context)
            {
                return context->targetArchitecture().target().getVersionString() == "gfx908";
            }

            int         getMaxNops(std::shared_ptr<InstructionRef> inst) const;
            bool        trigger(std::shared_ptr<InstructionRef> inst) const;
            bool        writeTrigger() const;
            int         getNops(Instruction const& inst) const;
            std::string getComment() const
            {
                return "XDL Write Hazard";
            }

        private:
            std::unordered_map<int, int> m_maxLatencyAndNops = {{2, 4}, {8, 10}, {16, 18}};
        };

        static_assert(CWaitStateObserver<XDLWrite908>);
    }
}
