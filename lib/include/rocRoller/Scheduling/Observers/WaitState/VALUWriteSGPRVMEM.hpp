#pragma once

#include <rocRoller/Scheduling/Observers/WaitState/WaitStateObserver.hpp>

namespace rocRoller
{
    namespace Scheduling
    {
        /**
         * @brief 908/90a/94x rules for VALU Write of SGPR followed by VMEM
         *
         * | Arch | 1st Inst       | 2nd Inst       | NOPs |
         * | ---- | -------------- | -------------- | ---- |
         * | 908  | v_* write SGPR | VMEM read SGPR | 5    |
         * | 90a  | v_* write SGPR | VMEM read SGPR | 5    |
         * | 94x  | v_* write SGPR | VMEM read SGPR | 5    |
         *
         */
        class VALUWriteSGPRVMEM : public WaitStateObserver<VALUWriteSGPRVMEM>
        {
        public:
            VALUWriteSGPRVMEM() {}
            VALUWriteSGPRVMEM(ContextPtr context)
                : WaitStateObserver<VALUWriteSGPRVMEM>(context){};

            static bool required(ContextPtr context)
            {
                auto arch = context->targetArchitecture().target().getVersionString();
                return arch == "gfx90a" || arch == "gfx908" || arch == "gfx940" || arch == "gfx941"
                       || arch == "gfx942" || arch == "gfx950";
            }

            int         getMaxNops(Instruction const& inst) const;
            bool        trigger(Instruction const& inst) const;
            bool        writeTrigger() const;
            int         getNops(Instruction const& inst) const;
            std::string getComment() const
            {
                return "VALU Write SGPR VMEM Read Hazard";
            }

        private:
            int const m_maxNops = 5;
        };

        static_assert(CWaitStateObserver<VALUWriteSGPRVMEM>);
    }
}
