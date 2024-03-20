#include <rocRoller/Scheduling/Observers/WaitState/MFMA/ACCVGPRReadWrite.hpp>

#include <rocRoller/CodeGen/InstructionRef.hpp>

namespace rocRoller
{
    namespace Scheduling
    {
        int ACCVGPRReadWrite::getMaxNops(std::shared_ptr<InstructionRef> inst) const
        {
            return m_maxNops;
        }

        bool ACCVGPRReadWrite::trigger(std::shared_ptr<InstructionRef> inst) const
        {
            return inst->isACCVGPRRead();
        };

        bool ACCVGPRReadWrite::writeTrigger() const
        {
            return true;
        }

        int ACCVGPRReadWrite::getNops(Instruction const& inst) const
        {
            InstructionRef instRef(inst);

            if(instRef.isMFMA())
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
            else if(instRef.isACCVGPRWrite())
            {
                return checkSrcs(inst).value_or(0);
            }
            else if(instRef.isVMEM())
            {
                return checkSrcs(inst).value_or(0);
            }
            return 0;
        }
    }
}
