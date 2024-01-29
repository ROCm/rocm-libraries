#pragma once

#include <rocRoller/Scheduling/Observers/WaitState/WaitStateObserver.hpp>

namespace rocRoller
{
    namespace Scheduling
    {
        /**
         * @brief 908/90a/94x rule for V_CMPX Write to EXEC followed by an MFMA requiring 4 NOPs
         *
         * | Arch | 1st Inst           | 2nd Inst        | NOPs |
         * | ---- | ------------------ | --------------- | ---- |
         * | 908  | v_cmpx* write EXEC | v_mfma*         | 4    |
         * | 908  | v_cmpx* write EXEC | v_accvgpr_write | 4    |
         * | 90a  | v_cmpx* write EXEC | v_mfma*         | 4    |
         * | 94x  | v_cmpx* write EXEC | v_mfma*         | 4    |
         *
         */
            void observe(Instruction const& inst);

            static bool required(ContextPtr context)
            {
                auto arch = context->targetArchitecture().target().getVersionString();
                return arch == "gfx90a" || arch == "gfx908" || arch == "gfx940" || arch == "gfx941"
                       || arch == "gfx942";
            }

            int         getMaxNops(std::shared_ptr<InstructionRef> inst) const;
            bool        trigger(std::shared_ptr<InstructionRef> inst) const;
            bool        writeTrigger() const;
            int         getNops(Instruction const& inst) const;
            std::string getComment() const
            {
                return "EXEC Write Hazard";
            }

        private:
            bool      m_checkACCVGPR;
            int const m_maxNops = 4;
        };

        static_assert(CWaitStateObserver<CMPXWriteExec>);
    }
}
