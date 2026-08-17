// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <rocRoller/GPUArchitecture/GPUInstructionInfo.hpp>
#include <rocRoller/Scheduling/Observers/WaitState/OPSEL94x.hpp>

namespace rocRoller
{
    namespace Scheduling
    {
        int OPSEL94x::getMaxNops(Instruction const& inst) const
        {
            return m_maxNops;
        }

        bool OPSEL94x::trigger(Instruction const& inst) const
        {
            if(GPUInstructionInfo::isVALU(inst.getOpCode()))
            {
                for(auto const& mod : inst.getModifiers())
                {
                    if(mod.starts_with("op_sel"))
                        return true;
                }
            }
            return false;
        };

        int OPSEL94x::getNops(Instruction const& inst) const
        {
            if(GPUInstructionInfo::isVALU(inst.getOpCode()))
            {
                return checkSrcs(inst).value_or(0);
            }
            return 0;
        }
    }
}
