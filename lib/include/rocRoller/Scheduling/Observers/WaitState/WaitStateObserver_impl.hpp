#pragma once

#include <rocRoller/Scheduling/Observers/WaitState/WaitStateObserver.hpp>

#include <rocRoller/CodeGen/InstructionRef.hpp>
#include <rocRoller/Context.hpp>

namespace rocRoller
{
    namespace Scheduling
    {
        template <class DerivedObserver>
        InstructionStatus WaitStateObserver<DerivedObserver>::peek(Instruction const& inst) const
        {
            return InstructionStatus::Nops(
                std::max(static_cast<DerivedObserver const*>(this)->getNops(inst), 0));
        }

        template <class DerivedObserver>
        void WaitStateObserver<DerivedObserver>::modify(Instruction& inst) const
        {
            auto const* thisDerived  = static_cast<DerivedObserver const*>(this);
            int         requiredNops = std::max(thisDerived->getNops(inst), 0);
            inst.setNopMin(requiredNops);
            if(requiredNops > 0)
            {
                inst.addComment("Wait state hazard: " + thisDerived->getComment());
            }
        }

        template <class DerivedObserver>
        InstructionStatus WaitStateObserver<DerivedObserver>::observe_base(Instruction const& inst)
        {
            auto* thisDerived = static_cast<DerivedObserver*>(this);
            auto  instRef     = std::make_shared<InstructionRef>(inst);
            if(thisDerived->trigger(instRef))
            {
                auto regMap = m_context.lock()->getRegisterHazardMap();
                for(auto iter = (thisDerived->writeTrigger() ? inst.getDsts().begin()
                                                             : inst.getSrcs().begin());
                    iter
                    != (thisDerived->writeTrigger() ? inst.getDsts().end() : inst.getSrcs().end());
                    iter++)
                {
                    auto reg = *iter;
                    if(reg)
                    {
                        for(auto const& regId : reg->getRegisterIds())
                        {
                            if(!regMap->contains(regId))
                            {
                                (*regMap)[regId] = {};
                            }
                            (*regMap)[regId].push_back(
                                WaitStateHazardCounter(thisDerived->getMaxNops(instRef),
                                                       instRef,
                                                       thisDerived->writeTrigger()));
                        }
                    }
                }
            }

            return InstructionStatus::Nops(inst.getNopCount());
        }

        template <class DerivedObserver>
        int WaitStateObserver<DerivedObserver>::getNopFromLatency(
            std::string const& opCode, std::unordered_map<int, int> const& latencyAndNops) const
        {
            auto const& architecture = m_context.lock()->targetArchitecture();
            int         passes       = architecture.GetInstructionInfo(opCode).getLatency();

            AssertFatal(latencyAndNops.contains(passes), "Unexpected number of passes");

            return latencyAndNops.at(passes);
        }

        template <class DerivedObserver>
        std::optional<int>
            WaitStateObserver<DerivedObserver>::checkRegister(Register::ValuePtr const& reg) const
        {
            if(!reg)
            {
                return std::nullopt;
            }

            auto const* thisDerived = static_cast<DerivedObserver const*>(this);
            auto        regMap      = m_context.lock()->getRegisterHazardMap();

            for(auto const& regId : reg->getRegisterIds())
            {
                if(regMap->contains(regId))
                {
                    for(auto const& hazard : regMap->at(regId))
                    {
                        bool isHazardous
                            = (thisDerived->writeTrigger() && hazard.regWasWritten())
                              || (!thisDerived->writeTrigger() && !hazard.regWasWritten());
                        if(isHazardous && thisDerived->trigger(hazard.getInstructionRef()))
                        {
                            return hazard.getRequiredNops();
                        }
                    }
                }
            }
            return std::nullopt;
        }

        template <class DerivedObserver>
        std::optional<int>
            WaitStateObserver<DerivedObserver>::checkSrcs(Instruction const& inst) const
        {
            for(auto const& src : inst.getSrcs())
            {
                if(auto val = checkRegister(src))
                {
                    return val;
                }
            }
            return std::nullopt;
        }

        template <class DerivedObserver>
        std::optional<int>
            WaitStateObserver<DerivedObserver>::checkDsts(Instruction const& inst) const
        {
            for(auto const& dst : inst.getDsts())
            {
                if(auto val = checkRegister(dst))
                {
                    return val;
                }
            }
            return std::nullopt;
        }
    }
}
