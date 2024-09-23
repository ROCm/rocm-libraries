#pragma once

#include <rocRoller/Scheduling/Observers/WaitState/WaitStateObserver.hpp>

namespace rocRoller
{
    namespace Scheduling
    {
        /**
         * @brief 908 rule for v_accvgpr_read Write
         *
         * | Arch | 1st Inst             | 2nd Inst                  | NOPs |
         * | ---- | -------------------- | ------------------------- | ---- |
         * | 908  | v_accvgpr_read write | VALU read as SrcA/B/C     | 0    |
         * | 908  | v_accvgpr_read write | v_mfma* read SrcA/B       | 2    |
         * | 908  | v_accvgpr_read write | v_accvgpr_write read SrcA | 2    |
         *
         */
        class ACCVGPRReadWrite : public WaitStateObserver<ACCVGPRReadWrite>
        {
        public:
            ACCVGPRReadWrite() {}
            ACCVGPRReadWrite(ContextPtr context)
                : WaitStateObserver<ACCVGPRReadWrite>(context){};

            constexpr static bool required(GPUArchitectureTarget const& target)
            {
                return target.is908GPU();
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
                return "v_accvgpr_read Write Hazard";
            }

        private:
            int const m_maxNops = 2;
        };

        static_assert(CWaitStateObserver<ACCVGPRReadWrite>);
    }
}
