#pragma once

#include <string>
#include <vector>

#include "Instruction_fwd.hpp"

namespace rocRoller
{
    /**
     * @brief InstructionRef represents an Instruction object as a copy, but does not carry the allocations or registers directly.
     */
    class InstructionRef
    {
    public:
        static bool isDLOP(Instruction const& inst);
        static bool isMFMA(Instruction const& inst);
        static bool isVCMPX(Instruction const& inst);
        static bool isVCMP(Instruction const& inst);

        static bool isScalar(Instruction const& inst);
        static bool isSMEM(Instruction const& inst);
        static bool isSControl(Instruction const& inst);
        static bool isSALU(Instruction const& inst);
        static bool isIntInst(Instruction const& inst);
        static bool isUIntInst(Instruction const& inst);
        static bool isSDWA(Instruction const& inst);

        static bool isVector(Instruction const& inst);
        static bool isVALU(Instruction const& inst);
        static bool isVALUTrans(Instruction const& inst);
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
        static bool isVAddInst(Instruction const& inst);
        static bool isVSubInst(Instruction const& inst);
        static bool isVReadlane(Instruction const& inst);
        static bool isVWritelane(Instruction const& inst);
        static bool isVDivScale(Instruction const& inst);
        static bool isVDivFmas(Instruction const& inst);

        static bool isDLOP(std::string const& inst);
        static bool isMFMA(std::string const& inst);
        static bool isVCMPX(std::string const& inst);
        static bool isVCMP(std::string const& inst);

        static bool isScalar(std::string const& inst);
        static bool isSMEM(std::string const& inst);
        static bool isSControl(std::string const& inst);
        static bool isSALU(std::string const& inst);
        static bool isIntInst(std::string const& inst);
        static bool isUIntInst(std::string const& inst);
        static bool isSDWA(std::string const& inst);

        static bool isVector(std::string const& inst);
        static bool isVALU(std::string const& inst);
        static bool isVALUTrans(std::string const& inst);
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
        static bool isVAddInst(std::string const& inst);
        static bool isVSubInst(std::string const& inst);
        static bool isVReadlane(std::string const& inst);
        static bool isVWritelane(std::string const& inst);
        static bool isVDivScale(std::string const& inst);
        static bool isVDivFmas(std::string const& inst);
    };
}

#include "InstructionRef_impl.hpp"
