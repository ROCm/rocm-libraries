#pragma once

#include <rocRoller/Scheduling/Observers/WaitState/WaitStateObserver.hpp>

namespace rocRoller
{
    namespace Scheduling
    {
        /**
         * @brief 908/90a/94x rule for VALU Write followed by an MFMA Read requiring 2 NOPs
         *
         * | Arch | 1st Inst  | 2nd Inst             | NOPs |
         * | ---- | --------- | -------------------- | ---- |
         * | 908  | v_* write | v_mfma* read         | 2    |
         * | 908  | v_* write | v_accvgpr_write read | 2    |
         * | 90a  | v_* write | v_mfma* read         | 2    |
         * | 94x  | v_* write | v_mfma* read         | 2    |
         *
         */
        class VALUWrite : public WaitStateObserver<VALUWrite>
        {
        public:
            VALUWrite() {}
            VALUWrite(ContextPtr context)
                : WaitStateObserver<VALUWrite>(context)
            {
                m_checkACCVGPR = context->targetArchitecture().target().is908GPU();
            };

            constexpr static bool required(GPUArchitectureTarget const& target)
            {
                return target.is9XGPU();
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
                return "VALU Write Hazard";
            }

        private:
            bool      m_checkACCVGPR;
            int const m_maxNops = 2;
        };

        static_assert(CWaitStateObserver<VALUWrite>);
    }
}
