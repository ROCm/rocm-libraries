#pragma once

#include <rocRoller/Scheduling/Observers/WaitState/WaitStateObserver.hpp>

namespace rocRoller
{
    namespace Scheduling
    {
        /**
         * @brief 94x rules for VALU Trans Writes
         *
         * | Arch | 1st Inst (trans op)  | 2nd Inst (non-trans op)  | NOPs |
         * | ---- | -------------------- | ------------------------ | ---- |
         * | 94x  | v_exp_f32            | v_*                      | 1    |
         * | 94x  | v_log_f32            | v_*                      | 1    |
         * | 94x  | v_rcp_f32            | v_*                      | 1    |
         * | 94x  | v_rcp_iflag_f32      | v_*                      | 1    |
         * | 94x  | v_rsq_f32            | v_*                      | 1    |
         * | 94x  | v_rcp_f64            | v_*                      | 1    |
         * | 94x  | v_rsq_f64            | v_*                      | 1    |
         * | 94x  | v_sqrt_f32           | v_*                      | 1    |
         * | 94x  | v_sqrt_f64           | v_*                      | 1    |
         * | 94x  | v_sin_f32            | v_*                      | 1    |
         * | 94x  | v_cos_f32            | v_*                      | 1    |
         * | 94x  | v_rcp_f16            | v_*                      | 1    |
         * | 94x  | v_sqrt_f16           | v_*                      | 1    |
         * | 94x  | v_rsq_f16            | v_*                      | 1    |
         * | 94x  | v_log_f16            | v_*                      | 1    |
         * | 94x  | v_exp_f16            | v_*                      | 1    |
         * | 94x  | v_sin_f16            | v_*                      | 1    |
         * | 94x  | v_cos_f16            | v_*                      | 1    |
         * | 94x  | v_exp_legacy_f32     | v_*                      | 1    |
         * | 94x  | v_log_legacy_f32     | v_*                      | 1    |
         *
         */
        class VALUTransWrite94x : public WaitStateObserver<VALUTransWrite94x>
        {
        public:
            VALUTransWrite94x() {}
            VALUTransWrite94x(ContextPtr context)
                : WaitStateObserver<VALUTransWrite94x>(context){};

            static bool required(ContextPtr context)
            {
                auto arch = context->targetArchitecture().target().getVersionString();
                return arch == "gfx940" || arch == "gfx941" || arch == "gfx942" || arch == "gfx950";
            }

            int         getMaxNops(Instruction const& inst) const;
            bool        trigger(Instruction const& inst) const;
            bool        writeTrigger() const;
            int         getNops(Instruction const& inst) const;
            std::string getComment() const
            {
                return "VALU Trans Write Hazard";
            }

        private:
            int const m_maxNops = 1;
        };

        static_assert(CWaitStateObserver<VALUTransWrite94x>);
    }
}
