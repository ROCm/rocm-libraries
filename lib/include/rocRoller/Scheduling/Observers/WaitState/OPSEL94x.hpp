#pragma once

#include <rocRoller/Scheduling/Observers/WaitState/WaitStateObserver.hpp>

namespace rocRoller
{
    namespace Scheduling
    {
        /**
         * @brief 94x rules for using OPSEL or SDWA
         *
         * | Arch | 1st Inst        | 2nd Inst | NOPs |
         * | ---- | --------------- | -------- | ---- |
         * | 94x  | v_* using OPSEL | v_* read | 1    |
         * | 94x  | v_* using SDWA  | v_* read | 1    |
         *
         */
        class OPSEL94x : public WaitStateObserver<OPSEL94x>
        {
        public:
            OPSEL94x() {}
            OPSEL94x(ContextPtr context)
                : WaitStateObserver<OPSEL94x>(context){};

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
                return "OPSEL/SDWA Write Hazard";
            }

        private:
            int const m_maxNops = 1;
        };

        static_assert(CWaitStateObserver<OPSEL94x>);
    }
}
