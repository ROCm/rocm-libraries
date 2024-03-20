#pragma once

#include <string>

#include "Instruction.hpp"

namespace rocRoller
{
    inline std::string InstructionRef::getOpCode() const
    {
        return m_opCode;
    }

    inline bool InstructionRef::isDLOP() const
    {
        return isDLOP(m_opCode);
    }

    inline bool InstructionRef::isMFMA() const
    {
        return isMFMA(m_opCode);
    }

    inline bool InstructionRef::isVCMPX() const
    {
        return isVCMPX(m_opCode);
    }

    inline bool InstructionRef::isVCMP() const
    {
        return isVCMP(m_opCode);
    }

    inline bool InstructionRef::isScalar() const
    {
        return isScalar(m_opCode);
    }

    inline bool InstructionRef::isSMEM() const
    {
        return isSMEM(m_opCode);
    }

    inline bool InstructionRef::isSControl() const
    {
        return isSControl(m_opCode);
    }

    inline bool InstructionRef::isSALU() const
    {
        return isSALU(m_opCode);
    }

    inline bool InstructionRef::isVector() const
    {
        return isVector(m_opCode);
    }

    inline bool InstructionRef::isVALU() const
    {
        return isVALU(m_opCode);
    }

    inline bool InstructionRef::isVALUTrans() const
    {
        return isVALUTrans(m_opCode);
    }

    inline bool InstructionRef::isDGEMM() const
    {
        return isDGEMM(m_opCode);
    }

    inline bool InstructionRef::isSGEMM() const
    {
        return isSGEMM(m_opCode);
    }

    inline bool InstructionRef::isVMEM() const
    {
        return isVMEM(m_opCode);
    }

    inline bool InstructionRef::isVMEMRead() const
    {
        return isVMEMRead(m_opCode);
    }

    inline bool InstructionRef::isVMEMWrite() const
    {
        return isVMEMWrite(m_opCode);
    }

    inline bool InstructionRef::isFlat() const
    {
        return isFlat(m_opCode);
    }

    inline bool InstructionRef::isLDS() const
    {
        return isLDS(m_opCode);
    }

    inline bool InstructionRef::isLDSRead() const
    {
        return isLDSRead(m_opCode);
    }

    inline bool InstructionRef::isLDSWrite() const
    {
        return isLDSWrite(m_opCode);
    }

    inline bool InstructionRef::isACCVGPRWrite() const
    {
        return isACCVGPRWrite(m_opCode);
    }

    inline bool InstructionRef::isACCVGPRRead() const
    {
        return isACCVGPRRead(m_opCode);
    }

    inline bool InstructionRef::isIntInst() const
    {
        return isIntInst(m_opCode);
    }
    inline bool InstructionRef::isUIntInst() const
    {
        return isUIntInst(m_opCode);
    }
    inline bool InstructionRef::isVAddInst() const
    {
        return isVAddInst(m_opCode);
    }
    inline bool InstructionRef::isVSubInst() const
    {
        return isVSubInst(m_opCode);
    }
    inline bool InstructionRef::isVReadlane() const
    {
        return isVReadlane(m_opCode);
    }
    inline bool InstructionRef::isVWritelane() const
    {
        return isVWritelane(m_opCode);
    }
    inline bool InstructionRef::isVDivScale() const
    {
        return isVDivScale(m_opCode);
    }
    inline bool InstructionRef::isVDivFmas() const
    {
        return isVDivFmas(m_opCode);
    }

    inline bool InstructionRef::isDLOP(Instruction const& inst)
    {
        return isDLOP(inst.getOpCode());
    }

    inline bool InstructionRef::isMFMA(Instruction const& inst)
    {
        return isMFMA(inst.getOpCode());
    }

    inline bool InstructionRef::isVCMPX(Instruction const& inst)
    {
        return isVCMPX(inst.getOpCode());
    }

    inline bool InstructionRef::isVCMP(Instruction const& inst)
    {
        return isVCMP(inst.getOpCode());
    }

    inline bool InstructionRef::isScalar(Instruction const& inst)
    {
        return isScalar(inst.getOpCode());
    }

    inline bool InstructionRef::isSMEM(Instruction const& inst)
    {
        return isSMEM(inst.getOpCode());
    }

    inline bool InstructionRef::isSControl(Instruction const& inst)
    {
        return isSControl(inst.getOpCode());
    }

    inline bool InstructionRef::isSALU(Instruction const& inst)
    {
        return isSALU(inst.getOpCode());
    }

    inline bool InstructionRef::isVector(Instruction const& inst)
    {
        return isVector(inst.getOpCode());
    }

    inline bool InstructionRef::isVALU(Instruction const& inst)
    {
        return isVALU(inst.getOpCode());
    }

    inline bool InstructionRef::isVALUTrans(Instruction const& inst)
    {
        return isVALUTrans(inst.getOpCode());
    }

    inline bool InstructionRef::isDGEMM(Instruction const& inst)
    {
        return isDGEMM(inst.getOpCode());
    }

    inline bool InstructionRef::isSGEMM(Instruction const& inst)
    {
        return isSGEMM(inst.getOpCode());
    }

    inline bool InstructionRef::isVMEM(Instruction const& inst)
    {
        return isVMEM(inst.getOpCode());
    }

    inline bool InstructionRef::isVMEMRead(Instruction const& inst)
    {
        return isVMEMRead(inst.getOpCode());
    }

    inline bool InstructionRef::isVMEMWrite(Instruction const& inst)
    {
        return isVMEMWrite(inst.getOpCode());
    }

    inline bool InstructionRef::isFlat(Instruction const& inst)
    {
        return isFlat(inst.getOpCode());
    }

    inline bool InstructionRef::isLDS(Instruction const& inst)
    {
        return isLDS(inst.getOpCode());
    }

    inline bool InstructionRef::isLDSRead(Instruction const& inst)
    {
        return isLDSRead(inst.getOpCode());
    }

    inline bool InstructionRef::isLDSWrite(Instruction const& inst)
    {
        return isLDSWrite(inst.getOpCode());
    }

    inline bool InstructionRef::isACCVGPRWrite(Instruction const& inst)
    {
        return isACCVGPRWrite(inst.getOpCode());
    }

    inline bool InstructionRef::isACCVGPRRead(Instruction const& inst)
    {
        return isACCVGPRRead(inst.getOpCode());
    }

    inline bool InstructionRef::isIntInst(Instruction const& inst)
    {
        return isIntInst(inst.getOpCode());
    }
    inline bool InstructionRef::isUIntInst(Instruction const& inst)
    {
        return InstructionRef::isUIntInst(inst.getOpCode());
    }
    inline bool InstructionRef::isVAddInst(Instruction const& inst)
    {
        return isVAddInst(inst.getOpCode());
    }
    inline bool InstructionRef::isVSubInst(Instruction const& inst)
    {
        return isVSubInst(inst.getOpCode());
    }
    inline bool InstructionRef::isVReadlane(Instruction const& inst)
    {
        return isVReadlane(inst.getOpCode());
    }
    inline bool InstructionRef::isVWritelane(Instruction const& inst)
    {
        return isVWritelane(inst.getOpCode());
    }
    inline bool InstructionRef::isVDivScale(Instruction const& inst)
    {
        return isVDivScale(inst.getOpCode());
    }
    inline bool InstructionRef::isVDivFmas(Instruction const& inst)
    {
        return isVDivFmas(inst.getOpCode());
    }

    inline bool InstructionRef::isDLOP(std::string const& opCode)
    {
        return opCode.rfind("v_dot", 0) == 0;
    }

    inline bool InstructionRef::isMFMA(std::string const& opCode)
    {
        return opCode.rfind("v_mfma", 0) == 0;
    }

    inline bool InstructionRef::isVCMPX(std::string const& opCode)
    {
        return opCode.rfind("v_cmpx_", 0) == 0;
    }

    inline bool InstructionRef::isVCMP(std::string const& opCode)
    {
        return opCode.rfind("v_cmp_", 0) == 0;
    }

    inline bool InstructionRef::isScalar(std::string const& opCode)
    {
        return opCode.rfind("s_", 0) == 0;
    }

    inline bool InstructionRef::isSMEM(std::string const& opCode)
    {
        return opCode.rfind("s_load", 0) == 0 || opCode.rfind("s_store", 0) == 0;
    }

    inline bool InstructionRef::isSControl(std::string const& opCode)
    {
        return isScalar(opCode)
               && (opCode.find("branch") != std::string::npos //
                   || opCode == "s_endpgm" || opCode == "s_barrier");
    }

    inline bool InstructionRef::isSALU(std::string const& opCode)
    {
        return isScalar(opCode) && !isSMEM(opCode) && !isSControl(opCode);
    }

    inline bool InstructionRef::isVector(std::string const& opCode)
    {
        return (opCode.rfind("v_", 0) == 0) || isVMEM(opCode) || isFlat(opCode) || isLDS(opCode);
    }

    inline bool InstructionRef::isVALU(std::string const& opCode)
    {
        return isVector(opCode) && !isVMEM(opCode) && !isFlat(opCode) && !isLDS(opCode)
               && !isMFMA(opCode) && !isDLOP(opCode);
    }

    inline bool InstructionRef::isVALUTrans(std::string const& opCode)
    {
        return isVALU(opCode)
               && (opCode.rfind("v_exp_f32", 0) == 0 || opCode.rfind("v_log_f32", 0) == 0
                   || opCode.rfind("v_rcp_f32", 0) == 0 || opCode.rfind("v_rcp_iflag_f32", 0) == 0
                   || opCode.rfind("v_rsq_f32", 0) == 0 || opCode.rfind("v_rcp_f64", 0) == 0
                   || opCode.rfind("v_rsq_f64", 0) == 0 || opCode.rfind("v_sqrt_f32", 0) == 0
                   || opCode.rfind("v_sqrt_f64", 0) == 0 || opCode.rfind("v_sin_f32", 0) == 0
                   || opCode.rfind("v_cos_f32", 0) == 0 || opCode.rfind("v_rcp_f16", 0) == 0
                   || opCode.rfind("v_sqrt_f16", 0) == 0 || opCode.rfind("v_rsq_f16", 0) == 0
                   || opCode.rfind("v_log_f16", 0) == 0 || opCode.rfind("v_exp_f16", 0) == 0
                   || opCode.rfind("v_sin_f16", 0) == 0 || opCode.rfind("v_cos_f16", 0) == 0
                   || opCode.rfind("v_exp_legacy_f32", 0) == 0
                   || opCode.rfind("v_log_legacy_f32", 0) == 0);
    }

    inline bool InstructionRef::isDGEMM(std::string const& opCode)
    {
        return opCode.rfind("v_mfma_f64", 0) == 0;
    }

    inline bool InstructionRef::isSGEMM(std::string const& opCode)
    {
        auto endPos = opCode.length() - 4;
        return opCode.rfind("v_mfma_", 0) == 0 && opCode.rfind("f32", endPos) == 0;
    }

    inline bool InstructionRef::isVMEM(std::string const& opCode)
    {
        return opCode.rfind("buffer_", 0) == 0;
    }

    inline bool InstructionRef::isVMEMRead(std::string const& opCode)
    {
        return isVMEM(opCode)
               && (opCode.find("read") != std::string::npos
                   || opCode.find("load") != std::string::npos);
    }

    inline bool InstructionRef::isVMEMWrite(std::string const& opCode)
    {
        return isVMEM(opCode)
               && (opCode.find("write") != std::string::npos
                   || opCode.find("store") != std::string::npos);
    }

    inline bool InstructionRef::isFlat(std::string const& opCode)
    {
        return opCode.rfind("flat_", 0) == 0;
    }

    inline bool InstructionRef::isLDS(std::string const& opCode)
    {
        return opCode.rfind("ds_", 0) == 0;
    }

    inline bool InstructionRef::isLDSRead(std::string const& opCode)
    {
        return isLDS(opCode)
               && (opCode.find("read") != std::string::npos
                   || opCode.find("load") != std::string::npos);
    }

    inline bool InstructionRef::isLDSWrite(std::string const& opCode)
    {
        return isLDS(opCode)
               && (opCode.find("write") != std::string::npos
                   || opCode.find("store") != std::string::npos);
    }

    inline bool InstructionRef::isACCVGPRWrite(std::string const& opCode)
    {
        return opCode.rfind("v_accvgpr_write", 0) == 0;
    }

    inline bool InstructionRef::isACCVGPRRead(std::string const& opCode)
    {
        return opCode.rfind("v_accvgpr_read", 0) == 0;
    }

    inline bool InstructionRef::isIntInst(std::string const& opCode)
    {
        // '_i' should be in the last 5 characters of the string, unless encoding specifier is present
        int offset = opCode.rfind("_e", opCode.size() - 6) == 0 ? 10 : 6;

        return opCode.rfind("_i", opCode.size() - offset) == 0;
    }

    inline bool InstructionRef::isUIntInst(std::string const& opCode)
    {
        // '_u' should be in the last 5 characters of the string, unless encoding specifier is present
        int offset = opCode.rfind("_e", opCode.size() - 6) == 0 ? 10 : 6;

        return opCode.rfind("_u", opCode.size() - offset) == 0;
    }

    inline bool InstructionRef::isVAddInst(std::string const& opCode)
    {
        return opCode.rfind("v_add", 0) == 0;
    }

    inline bool InstructionRef::isVSubInst(std::string const& opCode)
    {
        return opCode.rfind("v_sub", 0) == 0;
    }

    inline bool InstructionRef::isVReadlane(std::string const& opCode)
    {
        return opCode.rfind("v_readlane", 0) == 0 || opCode.rfind("v_readfirstlane", 0) == 0;
    }

    inline bool InstructionRef::isVWritelane(std::string const& opCode)
    {
        return opCode.rfind("v_writelane", 0) == 0;
    }

    inline bool InstructionRef::isVDivScale(std::string const& opCode)
    {
        return opCode.rfind("v_div_scale", 0) == 0;
    }

    inline bool InstructionRef::isVDivFmas(std::string const& opCode)
    {
        return opCode.rfind("v_div_fmas_", 0) == 0;
    }
}
