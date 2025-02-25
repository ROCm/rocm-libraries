#pragma once

#include <rocRoller/Scheduling/Observers/WaitState/WaitStateObserver.hpp>

namespace rocRoller
{
    namespace Scheduling
    {
        /**
         * @brief 94x/950 rule for VALU Write followed by a Readlane or Permlane
         *
         * | Arch | 1st Inst  | 2nd Inst             | NOPs |
         * | ---- | --------- | -------------------- | ---- |
         * | 94x  | v_* write | v_readlane_* read    | 1    |
         * | 950  | v_* write | v_permlane* read     | 2    |
         *
         */
        class VALUWriteReadlane94x : public WaitStateObserver<VALUWriteReadlane94x>
        {
        public:
            VALUWriteReadlane94x() {}
            VALUWriteReadlane94x(ContextPtr context)
                : WaitStateObserver<VALUWriteReadlane94x>(context){};

            constexpr static bool required(GPUArchitectureTarget const& target)
            {
                return target.isCDNA3GPU() || target.isCDNA35GPU();
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
                return "VALU Write Readlane Hazard";
            }

        private:
            bool      m_checkACCVGPR;
            int const m_maxNops = 2;
        };

        static_assert(CWaitStateObserver<VALUWriteReadlane94x>);
    }
}
