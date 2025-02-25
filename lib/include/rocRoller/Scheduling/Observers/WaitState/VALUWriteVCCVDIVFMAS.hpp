#pragma once

#include <rocRoller/Scheduling/Observers/WaitState/WaitStateObserver.hpp>

namespace rocRoller
{
    namespace Scheduling
    {
        /**
         * @brief 908/90a/94x rules for VALU Write VCC followed by v_div_fmas_*
         *
         * | Arch | 1st Inst      | 2nd Inst   | NOPs |
         * | ---- | ------------- | ---------- | ---- |
         * | 908  | v_* write VCC | v_div_fmas | 4    |
         * | 90a  | v_* write VCC | v_div_fmas | 4    |
         * | 94x  | v_* write VCC | v_div_fmas | 4    |
         *
         */
        class VALUWriteVCCVDIVFMAS : public WaitStateObserver<VALUWriteVCCVDIVFMAS>
        {
        public:
            VALUWriteVCCVDIVFMAS() {}
            VALUWriteVCCVDIVFMAS(ContextPtr context)
                : WaitStateObserver<VALUWriteVCCVDIVFMAS>(context){};

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
                return "VALU Write VCC v_div_fmas_* Hazard";
            }

        private:
            int const m_maxNops = 4;
        };

        static_assert(CWaitStateObserver<VALUWriteVCCVDIVFMAS>);
    }
}
