#pragma once

#include "Instruction.hpp"
#include "InstructionValues/RegisterReference.hpp"

#include <algorithm>

namespace rocRoller
{
    /**
     * @brief InstructionReference represents an Instruction object as a copy, but does not carry the allocations or registers directly
     *
     * InstructionReference can be used to copy an instruction to refer to in the future.
     * It does not carry direct reference to its registers, so those registers' lifecycles are unaffected in the generated code.
     * This is useful in places where being able to refer to instructions that were already generated is needed, such as hazard observers.
     */
    class InstructionReference
    {
    public:
        InstructionReference() {}
        InstructionReference(Instruction const& inst)
            : m_opCode(inst.getOpCode())
        {
            auto reg = inst.getRegisters();
            for(auto src : std::get<0>(reg))
            {
                m_src.push_back(Register::RegisterReference(src));
            }
            for(auto dst : std::get<1>(reg))
            {
                m_dst.push_back(Register::RegisterReference(dst));
            }
        }

        std::string getOpCode() const
        {
            return m_opCode;
        }

        std::vector<Register::RegisterReference> getDsts() const
        {
            return m_dst;
        }
        std::vector<Register::RegisterReference> getSrcs() const
        {
            return m_src;
        }

        bool isDLOP() const
        {
            return m_opCode.rfind("v_dot", 0) == 0;
        }

        bool isMFMA() const
        {
            return m_opCode.rfind("v_mfma", 0) == 0;
        }

        bool isXDLOP() const
        {
            return isMFMA();
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

    private:
        std::string                              m_opCode;
        std::vector<Register::RegisterReference> m_src;
        std::vector<Register::RegisterReference> m_dst;
    };
}
