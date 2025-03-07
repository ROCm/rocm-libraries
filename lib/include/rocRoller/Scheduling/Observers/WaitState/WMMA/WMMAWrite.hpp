#pragma once

#include <rocRoller/Scheduling/Observers/WaitState/WaitStateObserver.hpp>

namespace rocRoller
{
    namespace Scheduling
    {
        class WMMAWrite : public WaitStateObserver<WMMAWrite>
        {
        public:
            WMMAWrite() {}
            WMMAWrite(ContextPtr context)
                : WaitStateObserver<WMMAWrite>(context){};

            constexpr static bool required(const GPUArchitectureTarget& target)
            {
                return target.isRDNA4GPU();
            }

            int                   getMaxNops(const Instruction& inst) const;
            bool                  trigger(const Instruction& inst) const;
            static constexpr bool writeTrigger()
            {
                return true;
            }
            int         getNops(const Instruction& inst) const;
            std::string getComment() const
            {
                return "WMMA Write Hazard";
            }

        private:
            const int m_maxNops = 1;
        };

        static_assert(CWaitStateObserver<WMMAWrite>);
    }
}
