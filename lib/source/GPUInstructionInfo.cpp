/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2024-2025 AMD ROCm(TM) Software
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#include <rocRoller/GPUArchitecture/GPUInstructionInfo.hpp>

namespace rocRoller
{
    bool GPUInstructionInfo::isDLOP(std::string const& opCode)
    {
        return opCode.rfind("v_dot", 0) == 0;
    }

    bool GPUInstructionInfo::isMFMA(std::string const& opCode)
    {
        return opCode.rfind("v_mfma", 0) == 0;
    }

    bool GPUInstructionInfo::isWMMA(std::string const& opCode)
    {
        return opCode.rfind("v_wmma", 0) == 0;
    }

    bool GPUInstructionInfo::isVCMPX(std::string const& opCode)
    {
        return opCode.rfind("v_cmpx_", 0) == 0;
    }

    bool GPUInstructionInfo::isVCMP(std::string const& opCode)
    {
        return opCode.rfind("v_cmp_", 0) == 0;
    }

    bool GPUInstructionInfo::isScalar(std::string const& opCode)
    {
        return opCode.rfind("s_", 0) == 0;
    }

    bool GPUInstructionInfo::isSMEM(std::string const& opCode)
    {
        return opCode.rfind("s_load", 0) == 0 || opCode.rfind("s_store", 0) == 0;
    }

    bool GPUInstructionInfo::isSControl(std::string const& opCode)
    {
        return isScalar(opCode)
               && (opCode.find("branch") != std::string::npos //
                   || opCode == "s_endpgm" || opCode.rfind("s_barrier", 0) == 0);
    }

    bool GPUInstructionInfo::isSALU(std::string const& opCode)
    {
        return isScalar(opCode) && !isSMEM(opCode) && !isSControl(opCode);
    }

    bool GPUInstructionInfo::isVector(std::string const& opCode)
    {
        return (opCode.rfind("v_", 0) == 0) || isVMEM(opCode) || isFlat(opCode) || isLDS(opCode);
    }

    bool GPUInstructionInfo::isVALU(std::string const& opCode)
    {
        return isVector(opCode) && !isVMEM(opCode) && !isFlat(opCode) && !isLDS(opCode)
               && !isMFMA(opCode) && !isDLOP(opCode);
    }

    bool GPUInstructionInfo::isVALUTrans(std::string const& opCode)
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

    bool GPUInstructionInfo::isDGEMM(std::string const& opCode)
    {
        return opCode.rfind("v_mfma_f64", 0) == 0;
    }

    bool GPUInstructionInfo::isSGEMM(std::string const& opCode)
    {
        auto endPos = opCode.length() - 4;
        return opCode.rfind("v_mfma_", 0) == 0 && opCode.rfind("f32", endPos) == 0;
    }

    bool GPUInstructionInfo::isVMEM(std::string const& opCode)
    {
        return opCode.rfind("buffer_", 0) == 0;
    }

    bool GPUInstructionInfo::isVMEMRead(std::string const& opCode)
    {
        return isVMEM(opCode)
               && (opCode.find("read") != std::string::npos
                   || opCode.find("load") != std::string::npos);
    }

    bool GPUInstructionInfo::isVMEMWrite(std::string const& opCode)
    {
        return isVMEM(opCode)
               && (opCode.find("write") != std::string::npos
                   || opCode.find("store") != std::string::npos);
    }

    bool GPUInstructionInfo::isFlat(std::string const& opCode)
    {
        return opCode.rfind("flat_", 0) == 0;
    }

    bool GPUInstructionInfo::isLDS(std::string const& opCode)
    {
        return opCode.rfind("ds_", 0) == 0;
    }

    bool GPUInstructionInfo::isLDSRead(std::string const& opCode)
    {
        return isLDS(opCode)
               && (opCode.find("read") != std::string::npos
                   || opCode.find("load") != std::string::npos);
    }

    bool GPUInstructionInfo::isLDSWrite(std::string const& opCode)
    {
        return isLDS(opCode)
               && (opCode.find("write") != std::string::npos
                   || opCode.find("store") != std::string::npos);
    }

    bool GPUInstructionInfo::isACCVGPRWrite(std::string const& opCode)
    {
        return opCode.rfind("v_accvgpr_write", 0) == 0;
    }

    bool GPUInstructionInfo::isACCVGPRRead(std::string const& opCode)
    {
        return opCode.rfind("v_accvgpr_read", 0) == 0;
    }

    bool GPUInstructionInfo::isIntInst(std::string const& opCode)
    {
        return opCode.find("_i32") != std::string::npos || opCode.find("_i64") != std::string::npos;
    }

    bool GPUInstructionInfo::isUIntInst(std::string const& opCode)
    {
        return opCode.find("_u32") != std::string::npos || opCode.find("_u64") != std::string::npos;
    }

    bool GPUInstructionInfo::isVAddInst(std::string const& opCode)
    {
        return opCode.rfind("v_add", 0) == 0;
    }

    bool GPUInstructionInfo::isVSubInst(std::string const& opCode)
    {
        return opCode.rfind("v_sub", 0) == 0;
    }

    bool GPUInstructionInfo::isVReadlane(std::string const& opCode)
    {
        return opCode.rfind("v_readlane", 0) == 0 || opCode.rfind("v_readfirstlane", 0) == 0;
    }

    bool GPUInstructionInfo::isVWritelane(std::string const& opCode)
    {
        return opCode.rfind("v_writelane", 0) == 0;
    }

    bool GPUInstructionInfo::isVPermlane(std::string const& opCode)
    {
        return opCode.rfind("v_permlane", 0) == 0;
    }

    bool GPUInstructionInfo::isVDivScale(std::string const& opCode)
    {
        return opCode.rfind("v_div_scale", 0) == 0;
    }

    bool GPUInstructionInfo::isVDivFmas(std::string const& opCode)
    {
        return opCode.rfind("v_div_fmas_", 0) == 0;
    }
}
