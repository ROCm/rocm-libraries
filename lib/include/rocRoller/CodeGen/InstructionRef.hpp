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
        explicit InstructionRef(Instruction const& inst);

        std::string getOpCode() const;

        bool isDLOP() const;
        bool isMFMA() const;
        bool isCMPX() const;

        /**
         * @brief Whether the instruction affects the VALU
         *
         * @return Starts with a "v_" and is a VALU instruction
         */
        bool isVALU() const;
        bool isDGEMM() const;
        bool isVMEM() const;
        bool isFlat() const;
        bool isLDS() const;

        bool isACCVGPRWrite() const;

        bool isACCVGPRRead() const;

    private:
        std::string m_opCode;
    };
}

#include "InstructionRef_impl.hpp"
