#include <rocRoller/Scheduling/Observers/WaitState/RegisterMapObserver.hpp>

namespace rocRoller
{
    namespace Scheduling
    {

        InstructionStatus RegisterMapObserver::peek(Instruction const& inst) const
        {
            return InstructionStatus();
        };

        void RegisterMapObserver::modify(Instruction& inst) const
        {
            // No modifications
        }

        InstructionStatus RegisterMapObserver::observe(Instruction const& inst)
        {
            // If instruction is not a comment or empty
            if(!inst.getOpCode().empty())
            {
                auto regMap = m_context.lock()->getRegisterHazardMap();
                for(auto mapIt = regMap->begin(); mapIt != regMap->end();)
                {
                    for(auto hazardIt = mapIt->second.begin(); hazardIt != mapIt->second.end();)
                    {
                        hazardIt->decrement(inst.nopCount());
                        if(!hazardIt->stillAlive())
                        {
                            hazardIt = mapIt->second.erase(hazardIt);
                        }
                        else
                        {
                            hazardIt++;
                        }
                    }
                    if(mapIt->second.empty())
                    {
                        mapIt = regMap->erase(mapIt);
                    }
                    else
                    {
                        mapIt++;
                    }
                }
            }

            return InstructionStatus();
        }
    }
}
