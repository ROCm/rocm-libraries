#include <rocRoller/Scheduling/Observers/WaitState/MFMA/ACCVGPRReadWrite.hpp>

#include <rocRoller/CodeGen/InstructionRef.hpp>

namespace rocRoller
{
    namespace Scheduling
    {
        int ACCVGPRReadWrite::getMaxNops(Instruction const& inst) const
        {
            return m_maxNops;
        }

        bool ACCVGPRReadWrite::trigger(Instruction const& inst) const
        {
            return InstructionRef::isACCVGPRRead(inst.getOpCode());
        };

        bool ACCVGPRReadWrite::writeTrigger() const
        {
            return true;
        }

        int ACCVGPRReadWrite::getNops(Instruction const& inst) const
        {
            if(InstructionRef::isMFMA(inst.getOpCode()))
            {
                auto const& srcs = inst.getSrcs();

                std::optional<int> value;

                // SrcA
                AssertFatal(srcs[0] != nullptr, "Empty SrcA");
                if((value = checkRegister(srcs[0])))
                {
                    return *value;
                }

                // ScrB
                AssertFatal(srcs[1] != nullptr, "Empty SrcB");
                if((value = checkRegister(srcs[1])))
                {
                    return *value;
                }
            }
            else if(InstructionRef::isACCVGPRWrite(inst.getOpCode()))
            {
                return checkSrcs(inst).value_or(0);
            }
            else if(InstructionRef::isVMEM(inst.getOpCode()))
            {
                return checkSrcs(inst).value_or(0);
            }
            return 0;
        }
    }
}
