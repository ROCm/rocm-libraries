#include <rocRoller/Scheduling/Observers/WaitState/DLWrite.hpp>

#include <rocRoller/CodeGen/InstructionRef.hpp>

namespace rocRoller
{
    namespace Scheduling
    {
        int DLWrite::getMaxNops(std::shared_ptr<InstructionRef> inst) const
        {
            return m_maxNops;
        }

        bool DLWrite::trigger(std::shared_ptr<InstructionRef> inst) const
        {
            return inst->isDLOP();
        };

        bool DLWrite::writeTrigger() const
        {
            return true;
        }

        int DLWrite::getNops(Instruction const& inst) const
        {
            InstructionRef instRef(inst);

            if(instRef.isDLOP())
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
                            if(hazard.regWasWritten() && trigger(hazard.getInstructionRef())
                               && instRef.getOpCode() == hazard.getInstructionRef()->getOpCode())
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
