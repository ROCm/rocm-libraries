#include <rocRoller/Scheduling/Observers/WaitState/MFMA/ACCVGPRWriteWrite.hpp>

#include <rocRoller/CodeGen/InstructionRef.hpp>

namespace rocRoller
{
    namespace Scheduling
    {
        int ACCVGPRWriteWrite::getMaxNops(std::shared_ptr<InstructionRef> inst) const
        {
            return m_maxNops;
        }

        bool ACCVGPRWriteWrite::trigger(std::shared_ptr<InstructionRef> inst) const
        {
            return inst->isACCVGPRWrite();
        }

        bool ACCVGPRWriteWrite::writeTrigger() const
        {
            return true;
        }

        int ACCVGPRWriteWrite::getNops(Instruction const& inst) const
        {
            InstructionRef instRef(inst);

            if(instRef.isMFMA())
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
            else if(instRef.isACCVGPRRead())
            {
                return checkSrcs(inst).value_or(0);
            }
            return 0;
        }
    }
}
