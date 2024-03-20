#pragma once

#include <rocRoller/Scheduling/Observers/WaitState/WaitStateObserver.hpp>

namespace rocRoller
{
    namespace Scheduling
    {
        /**
         * @brief 94x rule for VALU Write followed by a Readlane requiring 1 NOP
         *
         * | Arch | 1st Inst  | 2nd Inst             | NOPs |
         * | ---- | --------- | -------------------- | ---- |
         * | 94x  | v_* write | v_readlane_* read    | 1    |
         *
         */
        class VALUWriteReadlane94x : public WaitStateObserver<VALUWriteReadlane94x>
        {
        public:
            VALUWriteReadlane94x() {}
            VALUWriteReadlane94x(ContextPtr context)
                : WaitStateObserver<VALUWriteReadlane94x>(context){};

            static bool required(ContextPtr context)
            {
                auto arch = context->targetArchitecture().target().getVersionString();
                return arch == "gfx940" || arch == "gfx941" || arch == "gfx942";
            }

            int         getMaxNops(std::shared_ptr<InstructionRef> inst) const;
            bool        trigger(std::shared_ptr<InstructionRef> inst) const;
            bool        writeTrigger() const;
            int         getNops(Instruction const& inst) const;
            std::string getComment() const
            {
                return "VALU Write Readlane Hazard";
            }

        private:
            bool      m_checkACCVGPR;
            int const m_maxNops = 1;
        };

        static_assert(CWaitStateObserver<VALUWriteReadlane94x>);
    }
}
