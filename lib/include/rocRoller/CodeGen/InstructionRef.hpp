#pragma once

#include <string>

#include "Instruction_fwd.hpp"

namespace rocRoller
{
    /**
     * @brief InstructionRef represents an Instruction object as a copy, but does not carry the allocations or registers directly.
     */
    class InstructionRef
    {
    public:
        InstructionRef(Instruction const& inst);

        std::string getOpCode() const
        {
            return m_opCode;
        }

        bool isDLOP() const
        {
            return m_opCode.rfind("v_dot", 0) == 0;
        }

        bool isMFMA() const
        {
            return m_opCode.rfind("v_mfma", 0) == 0;
        }

        bool isCMPX() const
        {
            return m_opCode.rfind("v_cmpx", 0) == 0;
        }

        /**
         * @brief Whether the instruction affects the VALU
         *
         * @return Starts with a "v_" and is a VALU instruction
         */
        bool isVALU() const
        {
            return m_opCode.rfind("v_", 0) == 0;
        }

        bool isDGEMM() const
        {
            return m_opCode.rfind("v_mfma_f64", 0) == 0;
        }

        bool isVMEM() const
        {
            return m_opCode.rfind("buffer_", 0) == 0;
        }

        bool isFlat() const
        {
            return m_opCode.rfind("flat_", 0) == 0;
        }

        bool isLDS() const
        {
            return m_opCode.rfind("ds_", 0) == 0;
        }

        bool isACCVGPRWrite() const
        {
            return m_opCode.rfind("v_accvgpr_write", 0) == 0;
        }

        bool isACCVGPRRead() const
        {
            return m_opCode.rfind("v_accvgpr_read", 0) == 0;
        }

    private:
        std::string m_opCode;
    };
}
