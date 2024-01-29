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

        bool isScalar() const;
        bool isSMEM() const;
        bool isSControl() const;
        bool isSALU() const;

        bool isVector() const;
        bool isVALU() const;
        bool isDGEMM() const;
        bool isSGEMM() const;
        bool isVMEM() const;
        bool isVMEMRead() const;
        bool isVMEMWrite() const;
        bool isFlat() const;
        bool isLDS() const;
        bool isLDSRead() const;
        bool isLDSWrite() const;

        bool isACCVGPRRead() const;
        bool isACCVGPRWrite() const;

        static bool isDLOP(Instruction const& inst);
        static bool isMFMA(Instruction const& inst);
        static bool isCMPX(Instruction const& inst);

        static bool isScalar(Instruction const& inst);
        static bool isSMEM(Instruction const& inst);
        static bool isSControl(Instruction const& inst);
        static bool isSALU(Instruction const& inst);

        static bool isVector(Instruction const& inst);
        static bool isVALU(Instruction const& inst);
        static bool isDGEMM(Instruction const& inst);
        static bool isSGEMM(Instruction const& inst);
        static bool isVMEM(Instruction const& inst);
        static bool isVMEMRead(Instruction const& inst);
        static bool isVMEMWrite(Instruction const& inst);
        static bool isFlat(Instruction const& inst);
        static bool isLDS(Instruction const& inst);
        static bool isLDSRead(Instruction const& inst);
        static bool isLDSWrite(Instruction const& inst);

        static bool isACCVGPRRead(Instruction const& inst);
        static bool isACCVGPRWrite(Instruction const& inst);

        static bool isDLOP(std::string const& inst);
        static bool isMFMA(std::string const& inst);
        static bool isCMPX(std::string const& inst);

        static bool isScalar(std::string const& inst);
        static bool isSMEM(std::string const& inst);
        static bool isSControl(std::string const& inst);
        static bool isSALU(std::string const& inst);

        static bool isVector(std::string const& inst);
        static bool isVALU(std::string const& inst);
        static bool isDGEMM(std::string const& inst);
        static bool isSGEMM(std::string const& inst);
        static bool isVMEM(std::string const& inst);
        static bool isVMEMRead(std::string const& inst);
        static bool isVMEMWrite(std::string const& inst);
        static bool isFlat(std::string const& inst);
        static bool isLDS(std::string const& inst);
        static bool isLDSRead(std::string const& inst);
        static bool isLDSWrite(std::string const& inst);

        static bool isACCVGPRRead(std::string const& inst);
        static bool isACCVGPRWrite(std::string const& inst);

    private:
        std::string m_opCode;
    };
}

#include "InstructionRef_impl.hpp"
