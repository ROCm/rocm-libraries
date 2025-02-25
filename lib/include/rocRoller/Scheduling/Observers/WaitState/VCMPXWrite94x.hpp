#pragma once

#include <rocRoller/Scheduling/Observers/WaitState/WaitStateObserver.hpp>

namespace rocRoller
{
    namespace Scheduling
    {
        /**
         * @brief 94x rules for v_cmpx_* writes
         *
         * | Arch | 1st Inst  | 2nd Inst          | NOPs |
         * | ---- | --------- | ----------------- | ---- |
         * | 94x  | v_cmpx_*  | v_* read EXEC     | 2    |
         * | 94x  | v_cmpx_*  | v_readlane_*      | 4    |
         * | 94x  | v_cmpx_*  | v_readfirstlane_* | 4    |
         * | 94x  | v_cmpx_*  | v_writelane_*     | 4    |
         * | 94x  | v_cmpx_*  | v_*               | 0    |
         * | 950  | v_cmpx_*  | v_permlane*       | 4    |
         *
         */
        class VCMPXWrite94x : public WaitStateObserver<VCMPXWrite94x>
        {
        public:
            VCMPXWrite94x() {}
            VCMPXWrite94x(ContextPtr context)
                : WaitStateObserver<VCMPXWrite94x>(context){};

            constexpr static bool required(GPUArchitectureTarget const& target)
            {
                return target.isCDNA3GPU() || target.isCDNA35GPU();
            }

            /**
             * Overriden as we need to target Exec
             */
            void observeHazard(Instruction const& inst) override;

            int                   getMaxNops(Instruction const& inst) const;
            bool                  trigger(Instruction const& inst) const;
            static constexpr bool writeTrigger()
            {
                return true;
            }
            int         getNops(Instruction const& inst) const;
            std::string getComment() const
            {
                return "v_cmpx Write Hazard";
            }

        private:
            int const m_maxNops = 4;
        };

        static_assert(CWaitStateObserver<VCMPXWrite94x>);
    }
}
