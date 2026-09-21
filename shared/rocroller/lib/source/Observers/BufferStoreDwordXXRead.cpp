// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <rocRoller/GPUArchitecture/GPUInstructionInfo.hpp>
#include <rocRoller/Scheduling/Observers/WaitState/BufferStoreDwordXXRead.hpp>

namespace rocRoller
{
    namespace Scheduling
    {
        void BufferStoreDwordXXRead::observeHazard(Instruction const& inst)
        {
            if(trigger(inst))
            {
                auto reg = inst.getSrcs().at(0);
                AssertFatal(reg != nullptr);
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

        int BufferStoreDwordXXRead::getMaxNops(Instruction const& inst) const
        {
            return m_maxNops;
        }

        bool BufferStoreDwordXXRead::trigger(Instruction const& inst) const
        {
            if(inst.getOpCode().starts_with("buffer_store_dwordx3")
               || inst.getOpCode().starts_with("buffer_store_dwordx4"))
            {
                auto srcs = inst.getSrcs();
                AssertFatal(srcs.size() - std::count(srcs.begin(), srcs.end(), nullptr) == 4,
                            "Unexpected arguments",
                            ShowValue(inst.toString(LogLevel::Debug)));
                return inst.getSrcs()[3]->regType() != Register::Type::Scalar;
            }
            return false;
        };

        int BufferStoreDwordXXRead::getNops(Instruction const& inst) const
        {
            if(GPUInstructionInfo::isVALU(inst.getOpCode()) && m_isCDNA1orCDNA2)
            {
                return checkDsts(inst).value_or(0) - 1;
            }
            return checkDsts(inst).value_or(0);
        }
    }
}
