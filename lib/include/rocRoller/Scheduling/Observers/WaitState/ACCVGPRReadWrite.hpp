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
            ACCVGPRReadWrite(std::shared_ptr<Context> context)
                : WaitStateObserver<ACCVGPRReadWrite>(context){};

            InstructionStatus observe(Instruction const& inst)
            {
                return observe_base(inst);
            }

            static bool required(std::shared_ptr<Context> context)
            {
                return context->targetArchitecture().target().getVersionString() == "gfx908";
            }

            int         getMaxNops(std::shared_ptr<InstructionRef> inst) const;
            bool        trigger(std::shared_ptr<InstructionRef> inst) const;
            bool        writeTrigger() const;
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
