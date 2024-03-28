#include <rocRoller/Scheduling/Observers/WaitState/MFMA/DLWrite.hpp>

#include <rocRoller/CodeGen/InstructionRef.hpp>

namespace rocRoller
{
    namespace Scheduling
    {
        void DLWrite::observeHazard(Instruction const& inst)
        {
            if(trigger(inst))
            {
                m_prevOpCode = inst.getOpCode();
                for(auto iter = (writeTrigger() ? inst.getDsts().begin() : inst.getSrcs().begin());
                    iter != (writeTrigger() ? inst.getDsts().end() : inst.getSrcs().end());
                    iter++)
                {
                    auto reg = *iter;
                    if(reg)
                    {
                        for(auto const& regId : reg->getRegisterIds())
                        {
                            (*m_hazardMap)[regId].push_back(
                                WaitStateHazardCounter(getMaxNops(inst), writeTrigger()));
                        }
                    }
                }
            }
        }

        int DLWrite::getMaxNops(Instruction const& inst) const
        {
            return m_maxNops;
        }

        bool DLWrite::trigger(Instruction const& inst) const
        {
            return InstructionRef::isDLOP(inst.getOpCode());
        };

        bool DLWrite::writeTrigger() const
        {
            return true;
        }

        int DLWrite::getNops(Instruction const& inst) const
        {
            if(InstructionRef::isDLOP(inst.getOpCode()))
            {
                std::optional<int> value;

                auto const& srcs = inst.getSrcs();

                // SrcC
                AssertFatal(srcs.at(2) != nullptr, "Empty SrcC");
                for(auto const& srcId : srcs.at(2)->getRegisterIds())
                {
                    if(m_hazardMap->contains(srcId))
                    {
                        for(auto const& hazard : m_hazardMap->at(srcId))
                        {
                            if(hazard.regWasWritten() && inst.getOpCode() == m_prevOpCode)
                            {
                                // Supports same opcode of DLops back-to-back SrcC forwarding which is used for accumulation
                                return 0;
                            }
                        }
                    }
                }

                // SrcA
                AssertFatal(srcs.at(0) != nullptr, "Empty SrcA");
                if((value = checkRegister(srcs.at(0))))
                {
                    return *value;
                }

                // SrcB
                AssertFatal(srcs.at(1) != nullptr, "Empty SrcB");
                if((value = checkRegister(srcs.at(1))))
                {
                    return *value;
                }
            }

            // If the opcode is different
            {
                std::optional<int> value;

                // RAW
                if((value = checkSrcs(inst)))
                {
                    return *value;
                }

                // WAW
                if((value = checkDsts(inst)))
                {
                    return *value;
                }
            }
            return 0;
        }
    }
}
