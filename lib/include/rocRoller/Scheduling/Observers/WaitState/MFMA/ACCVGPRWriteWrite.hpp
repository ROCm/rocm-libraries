#pragma once

#include <rocRoller/Scheduling/Observers/WaitState/WaitStateObserver.hpp>

namespace rocRoller
{
    namespace Scheduling
    {
        /**
         * @brief 908 rule for v_accvgpr_write Write
         *
         * | Arch | 1st Inst              | 2nd Inst                  | NOPs |
         * | ---- | --------------------- | ------------------------- | ---- |
         * | 908  | v_accvgpr_write write | v_mfma* read SrcC         | 1    |
         * | 908  | v_accvgpr_write write | v_mfma* read SrcA/B       | 3    |
         * | 908  | v_accvgpr_write write | v_accvgpr_read read SrcA  | 3    |
         *
         */
        class ACCVGPRWriteWrite : public WaitStateObserver<ACCVGPRWriteWrite>
        {
        public:
            ACCVGPRWriteWrite() {}
            ACCVGPRWriteWrite(ContextPtr context)
                : WaitStateObserver<ACCVGPRWriteWrite>(context){};

            static bool required(ContextPtr context)
            {
                return context->targetArchitecture().target().getVersionString() == "gfx908";
            }

            int         getMaxNops(Instruction const& inst) const;
            bool        trigger(Instruction const& inst) const;
            bool        writeTrigger() const;
            int         getNops(Instruction const& inst) const;
            std::string getComment() const
            {
                return "v_accvgpr_write Write Hazard";
            }

        private:
            int const m_maxNops = 3;
        };

        static_assert(CWaitStateObserver<ACCVGPRWriteWrite>);
    }
}
