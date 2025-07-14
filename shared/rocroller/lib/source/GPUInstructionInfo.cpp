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
        return opCode.starts_with("v_dot");
    }

    bool GPUInstructionInfo::isMFMA(std::string const& opCode)
    {
        return opCode.starts_with("v_mfma");
    }

    bool GPUInstructionInfo::isWMMA(std::string const& opCode)
    {
        return opCode.starts_with("v_wmma");
    }

    bool GPUInstructionInfo::isVCMPX(std::string const& opCode)
    {
        return opCode.starts_with("v_cmpx_");
    }

    bool GPUInstructionInfo::isVCMP(std::string const& opCode)
    {
        return opCode.starts_with("v_cmp_");
    }

    bool GPUInstructionInfo::isScalar(std::string const& opCode)
    {
        return opCode.starts_with("s_");
    }

    bool GPUInstructionInfo::isSMEM(std::string const& opCode)
    {
        return opCode.starts_with("s_load") || opCode.starts_with("s_store");
    }

    bool GPUInstructionInfo::isSBarrier(std::string const& opCode)
    {
        return opCode.starts_with("s_barrier");
    }

    bool GPUInstructionInfo::isSControl(std::string const& opCode)
    {
        return isScalar(opCode)
               && (opCode.find("branch") != std::string::npos //
                   || opCode == "s_endpgm" || isSBarrier(opCode));
    }

    bool GPUInstructionInfo::isSALU(std::string const& opCode)
    {
        return isScalar(opCode) && !isSMEM(opCode) && !isSControl(opCode);
    }

    bool GPUInstructionInfo::isVector(std::string const& opCode)
    {
        return (opCode.starts_with("v_")) || isVMEM(opCode) || isFlat(opCode) || isLDS(opCode);
    }

    bool GPUInstructionInfo::isVALU(std::string const& opCode)
    {
        return isVector(opCode) && !isVMEM(opCode) && !isFlat(opCode) && !isLDS(opCode)
               && !isMFMA(opCode) && !isDLOP(opCode);
    }

    bool GPUInstructionInfo::isVALUTrans(std::string const& opCode)
    {
        return isVALU(opCode)
               && (opCode.starts_with("v_exp_f32") || opCode.starts_with("v_log_f32")
                   || opCode.starts_with("v_rcp_f32") || opCode.starts_with("v_rcp_iflag_f32")
                   || opCode.starts_with("v_rsq_f32") || opCode.starts_with("v_rcp_f64")
                   || opCode.starts_with("v_rsq_f64") || opCode.starts_with("v_sqrt_f32")
                   || opCode.starts_with("v_sqrt_f64") || opCode.starts_with("v_sin_f32")
                   || opCode.starts_with("v_cos_f32") || opCode.starts_with("v_rcp_f16")
                   || opCode.starts_with("v_sqrt_f16") || opCode.starts_with("v_rsq_f16")
                   || opCode.starts_with("v_log_f16") || opCode.starts_with("v_exp_f16")
                   || opCode.starts_with("v_sin_f16") || opCode.starts_with("v_cos_f16")
                   || opCode.starts_with("v_exp_legacy_f32")
                   || opCode.starts_with("v_log_legacy_f32"));
    }

    bool GPUInstructionInfo::isDGEMM(std::string const& opCode)
    {
        return opCode.starts_with("v_mfma_f64");
    }

    bool GPUInstructionInfo::isSGEMM(std::string const& opCode)
    {
        auto endPos = opCode.length() - 4;
        return opCode.starts_with("v_mfma_") && opCode.rfind("f32", endPos) == 0;
    }

    bool GPUInstructionInfo::isVMEM(std::string const& opCode)
    {
        return opCode.starts_with("buffer_");
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
        return opCode.starts_with("flat_");
    }

    bool GPUInstructionInfo::isLDS(std::string const& opCode)
    {
        return opCode.starts_with("ds_");
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
        return opCode.starts_with("v_accvgpr_write");
    }

    bool GPUInstructionInfo::isACCVGPRRead(std::string const& opCode)
    {
        return opCode.starts_with("v_accvgpr_read");
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
        return opCode.starts_with("v_add");
    }

    bool GPUInstructionInfo::isVAddCarryInst(std::string const& opCode)
    {
        return opCode.starts_with("v_addc_");
    }

    bool GPUInstructionInfo::isVSubInst(std::string const& opCode)
    {
        return opCode.starts_with("v_sub");
    }

    bool GPUInstructionInfo::isVSubCarryInst(std::string const& opCode)
    {
        return opCode.starts_with("v_subb_");
    }

    bool GPUInstructionInfo::isVReadlane(std::string const& opCode)
    {
        return opCode.starts_with("v_readlane") || opCode.starts_with("v_readfirstlane");
    }

    bool GPUInstructionInfo::isVWritelane(std::string const& opCode)
    {
        return opCode.starts_with("v_writelane");
    }

    bool GPUInstructionInfo::isVPermlane(std::string const& opCode)
    {
        return opCode.starts_with("v_permlane");
    }

    bool GPUInstructionInfo::isVDivScale(std::string const& opCode)
    {
        return opCode.starts_with("v_div_scale");
    }

    bool GPUInstructionInfo::isVDivFmas(std::string const& opCode)
    {
        return opCode.starts_with("v_div_fmas_");
    }
}
