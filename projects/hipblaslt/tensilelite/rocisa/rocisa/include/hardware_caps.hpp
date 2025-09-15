/* ************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc.
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
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */
#pragma once
#include <array>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "helper.hpp"

inline bool tryAssembler(const IsaVersion&  isaVersion,
                         const std::string& assemblerPath,
                         const std::string& asmString,
                         bool               debug = false)
{
    std::vector<std::string> options;
    if(isaVersion[0] >= 10)
        options.push_back("-mwavefrontsize64");

    std::string isastr = getGfxNameTuple(isaVersion);

    std::vector<std::string> cmd
        = {assemblerPath, "-x", "assembler", "-target", "amdgcn-amdhsa", "-mcpu=" + isastr};
    for(auto o : options)
    {
        cmd.push_back(o);
    }
    cmd.push_back("-");
    std::vector<char*> args(cmd.size());
    std::transform(cmd.begin(), cmd.end(), args.begin(), [](auto& str) { return &str[0]; });
    args.push_back(nullptr);
    auto [rcode, result] = run(args, asmString, debug);

    if(debug)
    {
        std::string s;
        for(auto c : cmd)
            s += c + " ";
        std::cout << "isaVersion: " << isastr << std::endl;
        std::cout << "asm_cmd: " << s << std::endl;
        std::cout << "asmString: " << asmString << std::endl;
        std::cout << "result: " << result << std::endl;
        std::cout << "return code: " << rcode << std::endl;
    }

    if(rcode != 0)
        return false;
    return true;
}

inline int getMaxCnt(const IsaVersion& isaVersion,
                     const std::string& assemblerPath,
                     const std::string& prefix,
                     const std::string& suffix,
                     bool isDebug)
{
    for(int p = 64; p > 1; p >>= 1)
    {
        // Try ( pow(2) - 1 ) from high to low
        if(tryAssembler(isaVersion, assemblerPath, prefix + std::to_string(p - 1) + suffix, isDebug))
            return p - 1;
    }
    return 0;
}

inline std::map<std::string, int>
    initAsmCaps(const IsaVersion& isaVersion, const std::string& assemblerPath, bool isDebug)
{
    // Determine assembler capabilities by testing short instructions sequences
    std::map<std::string, int> rv;
    rv["SupportedISA"] = tryAssembler(isaVersion, assemblerPath, "", isDebug);
    rv["HasExplicitCO"]
        = tryAssembler(isaVersion, assemblerPath, "v_add_co_u32 v0,vcc,v0,1", isDebug);
    rv["HasExplicitNC"] = tryAssembler(isaVersion, assemblerPath, "v_add_nc_u32 v0,v0,1", isDebug);

    rv["HasDirectToLds"] = tryAssembler(isaVersion,
                                        assemblerPath,
                                        "buffer_load_dword v36, s[24:27], s28 offen offset:0 lds",
                                        isDebug)
                           || tryAssembler(isaVersion,
                                           assemblerPath,
                                           "buffer_load_b32 v36, s[24:27], s28 offen offset:0 lds",
                                           isDebug);
    rv["HasDirectToLdsx4"]
        = tryAssembler(isaVersion,
                       assemblerPath,
                       "buffer_load_dwordx4 v36, s[24:27], s28 offen offset:0 lds",
                       isDebug)
          || tryAssembler(isaVersion,
                          assemblerPath,
                          "buffer_load_b128 v36, s[24:27], s28 offen offset:0 lds",
                          isDebug);
    rv["HasAddLshl"]
        = tryAssembler(isaVersion, assemblerPath, "v_add_lshl_u32 v47, v36, v34, 0x2", isDebug);
    rv["HasLshlOr"]
        = tryAssembler(isaVersion, assemblerPath, "v_lshl_or_b32 v47, v36, 0x2, v34", isDebug);
    rv["HasSMulHi"]
        = tryAssembler(isaVersion, assemblerPath, "s_mul_hi_u32 s47, s36, s34", isDebug);

    rv["HasMFMA_explictB"] = tryAssembler(
        isaVersion, assemblerPath, "v_mfma_f32_32x32x1_2b_f32 a[0:31], v0, v1, a[0:31]", isDebug);
    rv["HasMFMA"] = tryAssembler(isaVersion,
                                 assemblerPath,
                                 "v_mfma_f32_32x32x2bf16 a[0:31], v32, v33, a[0:31]",
                                 isDebug)
                    || rv["HasMFMA_explictB"];
    rv["HasMFMA_f64"] = tryAssembler(isaVersion,
                                     assemblerPath,
                                     "v_mfma_f64_16x16x4f64 v[0:7], v[32:33], v[36:37], v[0:7]",
                                     isDebug)
                        || tryAssembler(isaVersion,
                                        assemblerPath,
                                        "v_mfma_f64_16x16x4_f64 v[0:7], v[32:33], v[36:37], v[0:7]",
                                        isDebug);
    rv["HasMFMA_bf16_1k"]
        = tryAssembler(isaVersion,
                       assemblerPath,
                       "v_mfma_f32_32x32x4bf16_1k a[0:31], v[32:33], v[36:37], a[0:31]",
                       isDebug);
    rv["HasMFMA_f8"] = tryAssembler(isaVersion,
                                    assemblerPath,
                                    "v_mfma_f32_16x16x32_fp8_fp8 a[0:3], v[2:3], v[4:5], a[0:3]",
                                    isDebug);
    rv["HasMFMA_b8"] = tryAssembler(isaVersion,
                                    assemblerPath,
                                    "v_mfma_f32_16x16x32_bf8_bf8 a[0:3], v[2:3], v[4:5], a[0:3]",
                                    isDebug);
    rv["HasMFMA_f8f6f4"]
        = tryAssembler(isaVersion,
                       assemblerPath,
                       "v_mfma_f32_16x16x128_f8f6f4 a[0:3], v[0:7], v[8:15], a[0:3]",
                       isDebug);

    rv["HasMFMA_xf32"]
        = tryAssembler(isaVersion,
                       assemblerPath,
                       "v_mfma_f32_32x32x4_xf32 a[0:15], v[32:33], v[36:37], a[0:15]",
                       isDebug);
    rv["HasSMFMA"] = tryAssembler(isaVersion,
                                  assemblerPath,
                                  "v_smfmac_f32_32x32x16_f16 a[0:15], v[32:33], v[36:39], v[40]",
                                  isDebug);
    rv["HasWMMA"]  = tryAssembler(isaVersion,
                                 assemblerPath,
                                 "v_wmma_f32_16x16x16_f16 v[0:3], v[8:15], v[16:23], v[0:3]",
                                 isDebug)
                    || tryAssembler(isaVersion,
                                    assemblerPath,
                                    "v_wmma_f32_16x16x16_f16 v[0:3], v[8:9], v[16:17], v[0:3]",
                                    isDebug);
    rv["HasWMMA_V1"] = tryAssembler(isaVersion,
                                    assemblerPath,
                                    "v_wmma_f32_16x16x16_f16 v[0:3], v[8:15], v[16:23], v[0:3]",
                                    isDebug);
    rv["HasWMMA_V2"] = tryAssembler(isaVersion,
                                    assemblerPath,
                                    "v_wmma_f32_16x16x16_f16 v[0:3], v[8:9], v[16:17], v[0:3]",
                                    isDebug);

    rv["v_mac_f16"] = tryAssembler(isaVersion, assemblerPath, "v_mac_f16 v47, v36, v34", isDebug);

    rv["v_fma_f16"] = tryAssembler(
        isaVersion, assemblerPath, "v_fma_f16 v47, v36, v34, v47, op_sel:[0,0,0,0]", isDebug);
    rv["v_fmac_f16"] = tryAssembler(isaVersion, assemblerPath, "v_fma_f16 v47, v36, v34", isDebug);

    rv["v_pk_fma_f16"] = tryAssembler(
        isaVersion, assemblerPath, "v_pk_fma_f16 v47, v36, v34, v47, op_sel:[0,0,0]", isDebug);
    rv["v_pk_fmac_f16"]
        = tryAssembler(isaVersion, assemblerPath, "v_pk_fma_f16 v47, v36, v34", isDebug);

    rv["v_pk_add_f32"] = tryAssembler(
        isaVersion, assemblerPath, "v_pk_add_f32 v[48:49], v[36:37], v[0:1]", isDebug);
    rv["v_pk_mul_f32"] = tryAssembler(
        isaVersion, assemblerPath, "v_pk_mul_f32 v[48:49], v[36:37], v[0:1]", isDebug);

    rv["v_pk_mul_f32"] = tryAssembler(
        isaVersion, assemblerPath, "v_pk_mul_f32 v[20:21], v[18:19], v[20:21]", isDebug);

    rv["v_mad_mix_f32"]
        = tryAssembler(isaVersion,
                       assemblerPath,
                       "v_mad_mix_f32 v47, v36, v34, v47, op_sel:[0,0,0] op_sel_hi:[1,1,0]",
                       isDebug);
    rv["v_fma_mix_f32"]
        = tryAssembler(isaVersion,
                       assemblerPath,
                       "v_fma_mix_f32 v47, v36, v34, v47, op_sel:[0,0,0] op_sel_hi:[1,1,0]",
                       isDebug);

    rv["v_dot2_f32_f16"]
        = tryAssembler(isaVersion, assemblerPath, "v_dot2_f32_f16 v20, v36, v34, v20", isDebug);
    rv["v_dot2c_f32_f16"]
        = tryAssembler(isaVersion, assemblerPath, "v_dot2c_f32_f16 v47, v36, v34", isDebug)
          || tryAssembler(isaVersion, assemblerPath, "v_dot2acc_f32_f16 v47, v36, v34", isDebug);

    rv["v_dot2_f32_bf16"]
        = tryAssembler(isaVersion, assemblerPath, "v_dot2_f32_bf16 v20, v36, v34, v20", isDebug);
    rv["v_dot2c_f32_bf16"]
        = tryAssembler(isaVersion, assemblerPath, "v_dot2c_f32_bf16 v47, v36, v34", isDebug)
          || tryAssembler(isaVersion, assemblerPath, "v_dot2acc_f32_bf16 v47, v36, v34", isDebug);

    rv["v_dot4_i32_i8"]
        = tryAssembler(isaVersion, assemblerPath, "v_dot4_i32_i8 v47, v36, v34", isDebug);
    rv["v_dot4c_i32_i8"]
        = tryAssembler(isaVersion, assemblerPath, "v_dot4c_i32_i8 v47, v36, v34", isDebug);
    rv["VOP3v_dot4_i32_i8"]
        = tryAssembler(isaVersion, assemblerPath, "v_dot4_i32_i8 v47, v36, v34, v47", isDebug);

    rv["v_mac_f32"] = tryAssembler(isaVersion, assemblerPath, "v_mac_f32 v20, v21, v22", isDebug);
    rv["v_fma_f32"]
        = tryAssembler(isaVersion, assemblerPath, "v_fma_f32 v20, v21, v22, v23", isDebug);
    rv["v_fmac_f32"] = tryAssembler(isaVersion, assemblerPath, "v_fmac_f32 v20, v21, v22", isDebug);

    rv["v_fma_f64"] = tryAssembler(
        isaVersion, assemblerPath, "v_fma_f64 v[20:21], v[22:23], v[24:25], v[20:21]", isDebug);

    rv["v_mov_b64"] = tryAssembler(isaVersion, assemblerPath, "v_mov_b64 v[0:1], v[2:3]", isDebug);

    rv["HasBF16CVT"]    = tryAssembler(isaVersion, assemblerPath, "v_cvt_f32_bf16 v0, v1", isDebug);
    rv["Hascvtfp8_f16"] = tryAssembler(isaVersion,
                                       assemblerPath,
                                       "v_cvt_scalef32_pk_fp8_f16 v[0], v[1], 0 op_sel:[0,0,0,0]",
                                       isDebug);
    rv["Hascvtf16_fp8"] = tryAssembler(isaVersion,
                                       assemblerPath,
                                       "v_cvt_scalef32_f16_fp8 v[0], v[1], 0 op_sel:[0,0,0,0]",
                                       isDebug);

    rv["HasLDSTr"] = tryAssembler(
        isaVersion, assemblerPath, "ds_read_b64_tr_b16 v[0:1], v0 offset: 0", isDebug);

    rv["HasGLTr8B64"] = tryAssembler(
        isaVersion, assemblerPath, "global_load_tr_b64 v[0], v0, s[0:1], offset:0", isDebug);

    rv["HasGLTr16B128"] = tryAssembler(
        isaVersion, assemblerPath, "global_load_tr_b128 v[0:1], v0, s[0:1], offset:0", isDebug);

    rv["v_prng_b32"] = tryAssembler(isaVersion, assemblerPath, "v_prng_b32 v47, v36", isDebug);

    rv["HasAtomicAdd"]
        = tryAssembler(isaVersion,
                       assemblerPath,
                       "buffer_atomic_add_f32 v0, v1, s[0:3], 0 offen offset:0",
                       isDebug)
          || tryAssembler(isaVersion,
                          assemblerPath,
                          "buffer_atomic_add_f32 v0, v1, s[0:3], null offen offset:0",
                          isDebug);
    rv["HasGLCModifier"]
        = tryAssembler(isaVersion,
                       assemblerPath,
                       "buffer_load_dwordx4 v[10:13], v[0], s[0:3], 0, offen offset:0, glc",
                       isDebug)
          || tryAssembler(isaVersion,
                          assemblerPath,
                          "buffer_load_dwordx4 v[10:13], v[0], s[0:3], null, offen offset:0, glc",
                          isDebug);
    rv["HasMUBUFConst"] = tryAssembler(isaVersion,
                                       assemblerPath,
                                       "buffer_load_dword v40, v36, s[24:27], 1 offen offset:0",
                                       isDebug)
                          || tryAssembler(isaVersion,
                                          assemblerPath,
                                          "buffer_load_b32 v40, v36, s[24:27], 1 offen offset:0",
                                          isDebug);
    rv["HasSCMPK"] = tryAssembler(isaVersion, assemblerPath, "s_cmpk_gt_u32 s56, 0x0", isDebug);

    rv["HasGLCModifier"]
        = tryAssembler(isaVersion,
                       assemblerPath,
                       "buffer_load_dwordx4 v[10:13], v[0], s[0:3], 0, offen offset:0, glc",
                       isDebug);

    rv["HasNTModifier"]
        = tryAssembler(isaVersion,
                       assemblerPath,
                       "buffer_load_dwordx4 v[10:13], v[0], s[0:3], 0, offen offset:0, nt",
                       isDebug);

    rv["HasNewBarrier"] = tryAssembler(isaVersion, assemblerPath, "s_barrier_wait -1", isDebug);

    rv["s_delay_alu"]
        = tryAssembler(isaVersion, assemblerPath, "s_delay_alu instid0(VALU_DEP_1)", isDebug);

    rv["SeparateVscnt"] = tryAssembler(isaVersion, assemblerPath, "s_waitcnt_vscnt null 0", isDebug);

    rv["SeparateLGKMcnt"] = tryAssembler(isaVersion, assemblerPath, "s_wait_dscnt 0", isDebug)
                        && tryAssembler(isaVersion, assemblerPath, "s_wait_kmcnt 0", isDebug);

    rv["SeparateVMcnt"] = tryAssembler(isaVersion, assemblerPath, "s_wait_loadcnt 0", isDebug)
                        && tryAssembler(isaVersion, assemblerPath, "s_wait_storecnt 0", isDebug);

    if(rv["SeparateVMcnt"])
    {
        // s_wait_loadcnt accept 16 bits immediate, but only use the lowest 6 bits are used, can't use tryAssembler
        rv["MaxLoadcnt"]  = 63;
        // s_wait_storecnt accept 16 bits immediate, but only use the lowest 6 bits are used, can't use tryAssembler
        rv["MaxStorecnt"] = 63;
    }
    else
    {
        rv["MaxVmcnt"] = getMaxCnt(isaVersion, assemblerPath, "s_waitcnt vmcnt(", ")", isDebug);
        if(rv["SeparateVscnt"])
        {
            // s_waitcnt_vscnt accept 16 bits immediate, but only use the lowest 6 bits are used, can't use tryAssembler
            rv["MaxVscnt"] = 63;
        }
    }

    if(rv["SeparateLGKMcnt"])
    {
        // s_wait_dscnt accept 16 bits immediate, but only use the lowest 6 bits are used, can't use tryAssembler
        rv["MaxDscnt"] = 63;
        // s_wait_kmcnt accept 16 bits immediate, but only use the lowest 5 bits are used, can't use tryAssembler
        rv["MaxKmcnt"] = 31;
    }
    else
    {
        rv["MaxLgkmcnt"] = getMaxCnt(isaVersion, assemblerPath, "s_waitcnt lgkmcnt(", ")", isDebug);
    }

    rv["SupportedSource"] = true;

    return rv;
}

inline std::map<std::string, int> initArchCaps(const IsaVersion& isaVersion)
{
    std::vector<std::array<int, 3>> b = {{9, 0, 6}, {9, 0, 8}, {9, 0, 10}, {9, 4, 2}};
    std::map<std::string, int>      rv;
    rv["HasEccHalf"]
        = checkInList(isaVersion, {{9, 0, 6}, {9, 0, 8}, {9, 0, 10}, {9, 4, 2}, {9, 5, 0}});
    rv["Waitcnt0Disabled"] = checkInList(isaVersion, {{9, 0, 8}, {9, 0, 10}, {9, 4, 2}, {9, 5, 0}});
    int deviceLDS          = 65536;
    if(checkInList(isaVersion, {{9, 5, 0}}))
        deviceLDS = 163840;
    rv["DeviceLDS"]          = deviceLDS;
    rv["CMPXWritesSGPR"]     = checkNotInList(isaVersion[0], {10, 11, 12});
    rv["HasWave32"]          = checkInList(isaVersion[0], {10, 11, 12});
    rv["HasSchedMode"]       = checkInList(isaVersion[0], {12});
    rv["HasAccCD"]           = checkInList(isaVersion, {{9, 0, 10}, {9, 4, 2}, {9, 5, 0}});
    rv["ArchAccUnifiedRegs"] = checkInList(isaVersion, {{9, 0, 10}, {9, 4, 2}, {9, 5, 0}});
    rv["CrosslaneWait"]      = checkInList(isaVersion, {{9, 4, 2}, {9, 5, 0}});
    rv["TransOpWait"]        = checkInList(isaVersion, {{9, 4, 2}, {9, 5, 0}});
    rv["SDWAWait"]           = checkInList(isaVersion, {{9, 4, 2}, {9, 5, 0}});
    rv["VgprBank"]           = checkInList(isaVersion[0], {10, 11, 12});
    rv["DSLow16NotPreserve"] = isaVersion[0] == 12;
    rv["WorkGroupIdFromTTM"] = isaVersion[0] == 12;
    rv["NoSDWA"]             = isaVersion[0] == 12;
    rv["VOP3ByteSel"]        = isaVersion[0] == 12;
    rv["HasFP8_OCP"]         = isaVersion[0] == 12;
    rv["HasF32XEmulation"]   = checkInList(isaVersion, {{9, 5, 0}});
    return rv;
}

inline std::map<std::string, int> initRegisterCaps(const IsaVersion&           isaVersion,
                                                   std::map<std::string, int>& archCaps)
{
    std::map<std::string, int> rv;
    rv["MaxVgpr"] = 256;
    // max allowed is 112 out of 112 , 6 is used by hardware 4 SGPRs are wasted
    rv["MaxSgpr"] = 102;

    rv["PhysicalMaxVgpr"]   = 512;
    rv["PhysicalMaxSgpr"]   = 800;
    rv["maxLDSConstOffset"] = 65536;
    if(isaVersion[0] == 10)
        rv["PhysicalMaxVgprCU"] = 1024 * 32;
    else if(isaVersion[0] == 11)
        if(isaVersion[2] == 2)
            rv["PhysicalMaxVgprCU"] = 1024 * 32;
        else
            rv["PhysicalMaxVgprCU"] = 1536 * 32;
    else if(isaVersion[0] == 12)
        rv["PhysicalMaxVgprCU"] = 1536 * 32;
    else if(isaVersion[0] == 9)
        if(archCaps["ArchAccUnifiedRegs"])
            rv["PhysicalMaxVgprCU"] = 2048 * 64;
        else
            rv["PhysicalMaxVgprCU"] = 1024 * 64;
    else if(isaVersion[0] == 8)
        rv["PhysicalMaxVgprCU"] = 1024 * 64;
    else if(isaVersion[0] == 0)
        rv["PhysicalMaxVgprCU"] = 0;
    else
        throw std::runtime_error("No valid VGPR value for this platform");

    return rv;
}

inline std::map<std::string, bool> initAsmBugs(const std::map<std::string, int>& asmCaps)
{
    std::map<std::string, bool> rv;
    rv["ExplicitCO"] = asmCaps.find("HasExplicitCO")->second;
    rv["ExplicitNC"] = asmCaps.find("HasExplicitNC")->second;

    return rv;
}
