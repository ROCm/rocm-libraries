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

            constexpr static bool required(GPUArchitectureTarget const& target)
            {
                return target.isCDNAGPU();
            }

            int                   getMaxNops(Instruction const& inst) const;
            bool                  trigger(Instruction const& inst) const;
            static constexpr bool writeTrigger()
            {
                return true;
            }
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
