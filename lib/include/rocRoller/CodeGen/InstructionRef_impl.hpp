#pragma once

#include <string>

#include "Instruction_fwd.hpp"

namespace rocRoller
{
    inline std::string InstructionRef::getOpCode() const
    {
        return m_opCode;
    }

    inline bool InstructionRef::isDLOP() const
    {
        return m_opCode.rfind("v_dot", 0) == 0;
    }

    inline bool InstructionRef::isMFMA() const
    {
        return m_opCode.rfind("v_mfma", 0) == 0;
    }

    inline bool InstructionRef::isCMPX() const
    {
        return m_opCode.rfind("v_cmpx", 0) == 0;
    }

    inline bool InstructionRef::isVALU() const
    {
        return m_opCode.rfind("v_", 0) == 0;
    }

    inline bool InstructionRef::isDGEMM() const
    {
        return m_opCode.rfind("v_mfma_f64", 0) == 0;
    }

    inline bool InstructionRef::isVMEM() const
    {
        return m_opCode.rfind("buffer_", 0) == 0;
    }

    inline bool InstructionRef::isFlat() const
    {
        return m_opCode.rfind("flat_", 0) == 0;
    }

    inline bool InstructionRef::isLDS() const
    {
        return m_opCode.rfind("ds_", 0) == 0;
    }

    inline bool InstructionRef::isACCVGPRWrite() const
    {
        return m_opCode.rfind("v_accvgpr_write", 0) == 0;
    }

    inline bool InstructionRef::isACCVGPRRead() const
    {
        return m_opCode.rfind("v_accvgpr_read", 0) == 0;
    }

}
