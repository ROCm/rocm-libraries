#include <rocRoller/Scheduling/Observers/WaitState/MFMA/ACCVGPRWriteWrite.hpp>

#include <rocRoller/CodeGen/InstructionRef.hpp>

namespace rocRoller
{
    namespace Scheduling
    {
        int ACCVGPRWriteWrite::getMaxNops(Instruction const& inst) const
        {
            return m_maxNops;
        }

        bool ACCVGPRWriteWrite::trigger(Instruction const& inst) const
        {
            return InstructionRef::isACCVGPRWrite(inst.getOpCode());
        }

        bool ACCVGPRWriteWrite::writeTrigger() const
        {
            return true;
        }

        int ACCVGPRWriteWrite::getNops(Instruction const& inst) const
        {
            if(InstructionRef::isMFMA(inst.getOpCode()))
            {
                std::optional<int> value;

                auto const& srcs = inst.getSrcs();

                // SrcC
                AssertFatal(srcs.at(2) != nullptr, "Empty SrcC");
                if((value = checkRegister(srcs[2])))
                {
                    return *value - (m_maxNops - 1);
                }

                // SrcA
                AssertFatal(srcs.at(0) != nullptr, "Empty SrcA");
                if((value = checkRegister(srcs.at(0))))
                {
                    return *value;
                }

                // ScrB
                AssertFatal(srcs.at(1) != nullptr, "Empty SrcB");
                if((value = checkRegister(srcs.at(1))))
                {
                    return *value;
                }
            }
            else if(InstructionRef::isACCVGPRRead(inst.getOpCode()))
            {
                return checkSrcs(inst).value_or(0);
            }
            return 0;
        }
    }
}
