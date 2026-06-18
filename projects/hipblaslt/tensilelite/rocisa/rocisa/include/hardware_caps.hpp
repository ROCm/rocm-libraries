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
                         bool               debug = false,
                         bool               isWave32 = false)
{
    std::vector<std::string> options;
    if(!isWave32 && isaVersion[0] >= 10)
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

inline int getMaxCnt(const IsaVersion&  isaVersion,
                     const std::string& assemblerPath,
                     const std::string& prefix,
                     const std::string& suffix,
                     bool               isDebug)
{
    for(int p = 64; p > 1; p >>= 1)
    {
        // Try ( pow(2) - 1 ) from high to low
        if(tryAssembler(
               isaVersion, assemblerPath, prefix + std::to_string(p - 1) + suffix, isDebug))
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

    rv["HasPartialOOB"] = checkNotInList(isaVersion, {{12,5,0}});

    rv["HasExplicitCO"]
        = tryAssembler(isaVersion, assemblerPath, "v_add_co_u32 v0,vcc,v0,1", isDebug)
          || tryAssembler(isaVersion, assemblerPath, "v_add_co_u32 v0,vcc_lo,v0,1", isDebug);
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
    rv["HasScalarStore"]
        = tryAssembler(isaVersion, assemblerPath, "s_store_dword s79, s[70:71], s77", isDebug)
          || tryAssembler(isaVersion, assemblerPath, "s_store_b32 s79, s[70:71], s77", isDebug);
    rv["HasSAtomic"] = tryAssembler(isaVersion, assemblerPath, "s_atomic_dec s11, s[0:1]", isDebug);
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
                                    isDebug)
                    || tryAssembler(isaVersion,
                                    assemblerPath,
                                    "v_wmma_f32_16x16x32_bf16 v[0:7], v[8:15], v[8:15], v[0:7]",
                                    isDebug)
                    || tryAssembler(isaVersion,
                                    assemblerPath,
                                    "v_wmma_f32_16x16x4_f32 v[0:7], v[8:9], v[8:9], v[0:7]",
                                    isDebug);
    rv["HasWMMA_V1"] = tryAssembler(isaVersion,
                                    assemblerPath,
                                    "v_wmma_f32_16x16x16_f16 v[0:3], v[8:15], v[16:23], v[0:3]",
                                    isDebug);
    rv["HasWMMA_V2"] = tryAssembler(isaVersion,
                                    assemblerPath,
                                    "v_wmma_f32_16x16x16_f16 v[0:3], v[8:9], v[16:17], v[0:3]",
                                    isDebug);
    rv["HasWMMA_V3"] = tryAssembler(isaVersion,
                                    assemblerPath,
                                    "v_wmma_f32_16x16x32_bf16 v[0:7], v[8:15], v[8:15], v[0:7]",
                                    isDebug)
                       || tryAssembler(isaVersion,
                                       assemblerPath,
                                       "v_wmma_f32_16x16x4_f32 v[0:7], v[8:9], v[8:9], v[0:7]",
                                       isDebug);
    // If HasWMMA_V3 is set, force HasWMMA_V2 off so the wmma_v3 path is used
    if(rv["HasWMMA_V3"])
        rv["HasWMMA_V2"] = false;
    rv["HasWMMA_V3_f64"]
        = tryAssembler(isaVersion,
                       assemblerPath,
                       "v_wmma_f64_16x16x4_f64 v[0:15], v[16:19], v[20:23], v[0:15]",
                       isDebug)
          || tryAssembler(isaVersion,
                          assemblerPath,
                          "v_wmma_f64_16x16x8_f64 v[0:15], v[16:23], v[24:31], v[0:15]",
                          isDebug);
    
    rv["HasWMMA_f8f6f4"] = tryAssembler(isaVersion,
                                        assemblerPath,
                                        "v_wmma_f32_16x16x128_f8f6f4 v[0:7], v[16:31], v[16:31], v[0:7]",
                                        isDebug);

    // InitCIterWmma replaces the v_mov initC with a WMMA using src C = immediate 0
    rv["HasWMMA_AccImmZero"] = tryAssembler(isaVersion,
                                            assemblerPath,
                                            "v_wmma_f32_16x16x32_bf16 v[0:7], v[8:15], v[8:15], 0",
                                            isDebug)
                                && tryAssembler(isaVersion,
                                                assemblerPath,
                                                "v_wmma_f32_16x16x4_f32 v[0:7], v[8:9], v[8:9], 0",
                                                isDebug);

    rv["HasAdd_PC_i64"] = false;

    rv["HasSWMMAC"] = tryAssembler(isaVersion, 
                                  assemblerPath, 
                                  "v_swmmac_f32_16x16x32_f16 v[0:3], v[32:33], v[36:39], v[44]", isDebug)
                    || tryAssembler(isaVersion, 
                                    assemblerPath, 
                                    "v_swmmac_f32_16x16x32_f16 v[0:7], v[32:35], v[36:43], v[44]", isDebug, 
                                    true)
                    || tryAssembler(isaVersion,
                                    assemblerPath,
                                    "v_swmmac_f32_16x16x64_f16 v[0:7], v[8:15], v[16:31], v[44]",
                                    isDebug, 
                                    true);
    rv["HasSWMMAC_gfx1250"] = tryAssembler(isaVersion,
                                          assemblerPath,
                                          "v_swmmac_f32_16x16x64_f16 v[0:7], v[8:15], v[16:31], v[44]",
                                          isDebug,
                                          true);

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
    rv["s_sub_u64"]
        = tryAssembler(isaVersion, assemblerPath, "s_sub_u64 s[0:1], s[0:1], s[2:3]", isDebug);
    rv["s_add_u64"]
        = tryAssembler(isaVersion, assemblerPath, "s_add_u64 s[0:1], s[0:1], s[2:3]", isDebug);
    rv["v_add_nc_u64"] = tryAssembler(isaVersion, assemblerPath, "v_add_nc_u64 v[0:1], v[2:3], v[4:5]", isDebug);

    rv["HasBF16CVT"] = tryAssembler(isaVersion, assemblerPath, "v_cvt_f32_bf16 v0, v1", isDebug);

    rv["HasPkF16CVT"] = tryAssembler(isaVersion, assemblerPath, "v_cvt_pk_f16_f32 v0, v1, v2", isDebug);

    rv["Hascvtfp8_f16"] = tryAssembler(isaVersion,
                                       assemblerPath,
                                       "v_cvt_scalef32_pk_fp8_f16 v[0], v[1], 0 op_sel:[0,0,0,0]",
                                       isDebug);
    rv["Hascvtf16_fp8_sf32"] = tryAssembler(isaVersion,
                                            assemblerPath,
                                            "v_cvt_scalef32_f16_fp8 v[0], v[1], 0 op_sel:[0,0,0,0]",
                                            isDebug);
    rv["HasCvtFP8toF16"] = tryAssembler(isaVersion,
                                        assemblerPath,
                                        "v_cvt_f16_fp8 v[0], v[1] byte_sel:2",
                                        isDebug);

    rv["HasLDSTrB64B16"] = tryAssembler(
        isaVersion, assemblerPath, "ds_read_b64_tr_b16 v[0:1], v0 offset: 0", isDebug);

    rv["HasGLTr8B64"] = tryAssembler(
        isaVersion, assemblerPath, "global_load_tr_b64 v[0], v0, s[0:1], offset:0", isDebug);

    rv["HasGLTr16B128"] = tryAssembler(
        isaVersion, assemblerPath, "global_load_tr_b128 v[0:1], v0, s[0:1], offset:0", isDebug);

    rv["HasLDSTrB128B16"] = tryAssembler(
        isaVersion, assemblerPath, "ds_load_tr16_b128 v[0:3], v0 offset: 0", isDebug);

    rv["HasLDSTrB64B8"]
        = tryAssembler(isaVersion, assemblerPath, "ds_load_tr8_b64 v[0:1], v0 offset: 0", isDebug);

    rv["HasLDSTrB64B4"] = tryAssembler(
        isaVersion, assemblerPath, "ds_load_tr4_b64 v[0:1], v0 offset: 0", isDebug);

    rv["HasLDSTrB96B6"] = tryAssembler(
        isaVersion, assemblerPath, "ds_load_tr6_b96 v[0:2], v0 offset: 0", isDebug);

    rv["HasLDSTr"] = rv["HasLDSTrB64B16"] || rv["HasLDSTrB128B16"] || rv["HasLDSTrB64B8"] || rv["HasLDSTrB64B4"];

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
    rv["HasSC0Modifier"]
        = tryAssembler(isaVersion,
                       assemblerPath,
                       "buffer_load_dwordx4 v[10:13], v[0], s[0:3], 0, offen offset:0, sc0",
                       isDebug)
          || tryAssembler(isaVersion,
                          assemblerPath,
                          "buffer_load_dwordx4 v[10:13], v[0], s[0:3], null, offen offset:0, sc0",
                          isDebug);
    rv["HasDLCModifier"]
        = tryAssembler(isaVersion,
                       assemblerPath,
                       "buffer_load_dwordx4 v[10:13], v[0], s[0:3], 0, offen offset:0, dlc",
                       isDebug)
          || tryAssembler(isaVersion,
                          assemblerPath,
                          "buffer_load_dwordx4 v[10:13], v[0], s[0:3], null, offen offset:0, dlc",
                          isDebug);
    rv["HasSCOPEModifier"]
        = tryAssembler(
              isaVersion,
              assemblerPath,
              "buffer_load_dwordx4 v[10:13], v[0], s[0:3], 0, offen offset:0, scope:SCOPE_DEV",
              isDebug)
          || tryAssembler(
              isaVersion,
              assemblerPath,
              "buffer_load_dwordx4 v[10:13], v[0], s[0:3], null offen offset:0, scope:SCOPE_DEV",
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

    // gfx1250 replaces the bare 'nt' bit with a 3-bit Temporal Hint
    // (th:TH_LOAD_*/TH_STORE_*) and adds a per-op Non-Volatile (nv) bit.
    // (Volatile and Non-Volatile Memory Access).  Probe both with the buffer_*
    // form because gfx12 supports the same modifier syntax for buffer / global / flat.
    rv["HasTHModifier"]
        = tryAssembler(
              isaVersion,
              assemblerPath,
              "buffer_load_dwordx4 v[10:13], v[0], s[0:3], 0 offen offset:0 th:TH_LOAD_NT",
              isDebug)
          || tryAssembler(
              isaVersion,
              assemblerPath,
              "buffer_load_dwordx4 v[10:13], v[0], s[0:3], null offen offset:0 th:TH_LOAD_NT",
              isDebug);
    rv["HasNVModifier"]
        = tryAssembler(isaVersion,
                       assemblerPath,
                       "buffer_load_dwordx4 v[10:13], v[0], s[0:3], 0 offen offset:0 nv",
                       isDebug)
          || tryAssembler(isaVersion,
                          assemblerPath,
                          "buffer_load_dwordx4 v[10:13], v[0], s[0:3], null offen offset:0 nv",
                          isDebug);

    rv["HasNewBarrier"] = tryAssembler(isaVersion, assemblerPath, "s_barrier_wait -1", isDebug);
    rv["HasClusterBarrier"] = tryAssembler(isaVersion, assemblerPath, "s_barrier_wait -3", isDebug);
    rv["HasTDM"] = tryAssembler(isaVersion, assemblerPath, "tensor_load_to_lds s[0:3], s[4:11]", isDebug);
    // v_movrelsd_2_b32: indirect-VGPR-write move used by CompactLoopStore's
    // per-iter "copy MI out reg" body. Only some archs (gfx1250) have it, so
    // probe the assembler and gate CompactLoopStore on this cap in Solution.py.
    rv["HasMovRelsD2B32"] = tryAssembler(isaVersion, assemblerPath, "v_movrelsd_2_b32 v0, v1", isDebug);

    rv["s_delay_alu"]
        = tryAssembler(isaVersion, assemblerPath, "s_delay_alu instid0(VALU_DEP_1)", isDebug);
    rv["HasVgprMSB"] = tryAssembler(isaVersion, assemblerPath, "s_set_vgpr_msb 0", isDebug);
    // 16-bit MSB form packs the previous-instruction MSB in the high byte
    // (e.g. 0x0101). Some assemblers accept only the 8-bit form, so probe
    // the wider encoding separately.
    rv["HasVgprMSB16"]
        = rv["HasVgprMSB"]
          && tryAssembler(isaVersion, assemblerPath, "s_set_vgpr_msb 0x0101", isDebug);
    // workaround: as we generate s_set_vgpr_msb in toString(), we can't calculate inst len correctly.
    rv["ShortBranchMaxLength"] = rv["HasVgprMSB"]? 8192 : 16384;

    rv["SeparateVscnt"]
        = tryAssembler(isaVersion, assemblerPath, "s_waitcnt_vscnt null 0", isDebug);
    rv["HasGlobalPrefetch"] = tryAssembler(isaVersion, assemblerPath, "global_prefetch_b8 v[0:1], off scope:SCOPE_SE th:TH_LOAD_NT", isDebug);

    rv["SeparateLGKMcnt"] = tryAssembler(isaVersion, assemblerPath, "s_wait_dscnt 0", isDebug)
                            && tryAssembler(isaVersion, assemblerPath, "s_wait_kmcnt 0", isDebug);

    rv["SeparateVMcnt"] = tryAssembler(isaVersion, assemblerPath, "s_wait_loadcnt 0", isDebug)
                          && tryAssembler(isaVersion, assemblerPath, "s_wait_storecnt 0", isDebug);

    if(rv["SeparateVMcnt"])
    {
        // s_wait_loadcnt accept 16 bits immediate, but only use the lowest 6 bits are used, can't use tryAssembler
        rv["MaxLoadcnt"] = 63;
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

    // s_wait_xcnt drains XNACK-replay tracking before a subsequent volatile/atomic
    // VMEM op. Probed independently because not every arch with separate vmem
    // counters supports xcnt (e.g. it is gfx1250+). The immediate accepts 16 bits
    // but hardware only consumes the lowest 6 bits, matching loadcnt/storecnt.
    rv["HasXcnt"] = tryAssembler(isaVersion, assemblerPath, "s_wait_xcnt 0", isDebug);
    if(rv["HasXcnt"])
    {
        rv["MaxXcnt"] = 63;
    }

    rv["SupportedSource"] = true;

    return rv;
}

inline std::map<std::string, int> initArchCaps(const IsaVersion& isaVersion)
{
    std::vector<std::array<int, 3>> b = {{9, 0, 6}, {9, 0, 8}, {9, 0, 10}, {9, 4, 2}};
    std::map<std::string, int>      rv;
    rv["HasEccHalf"]
        = checkInList(isaVersion, {{9, 0, 6}, {9, 0, 8}, {9, 0, 10}, {9, 4, 2}, {9, 5, 0}, {12, 5, 0}});
    rv["Waitcnt0Disabled"] = checkInList(isaVersion, {{9, 0, 8}, {9, 0, 10}, {9, 4, 2}, {9, 5, 0}});
    int deviceLDS          = 65536;
    if(checkInList(isaVersion, {{9, 5, 0}}))
        deviceLDS = 163840;
    else if(checkInList(isaVersion, {{12, 5, 0}}))
        deviceLDS = 327680;

    rv["DeviceLDS"]          = deviceLDS;
    rv["CMPXWritesSGPR"]     = checkNotInList(isaVersion[0], {10, 11, 12});
    rv["HasWave32"]          = checkInList(isaVersion[0], {10, 11, 12});
    rv["HasSchedMode"]       = checkInList(isaVersion[0], {12});
    rv["HasAccCD"]           = checkInList(isaVersion, {{9, 0, 10}, {9, 4, 2}, {9, 5, 0}});
    rv["ArchAccUnifiedRegs"] = checkInList(isaVersion, {{9, 0, 10}, {9, 4, 2}, {9, 5, 0}});
    // Max concurrent waves per SIMD: 8 for ArchAccUnifiedRegs (gfx90a/gfx942/gfx950), 10 otherwise.
    rv["MaxWavesPerSimd"]    = rv["ArchAccUnifiedRegs"] ? 8 : 10;
    rv["CrosslaneWait"]      = checkInList(isaVersion, {{9, 4, 2}, {9, 5, 0}});
    rv["TransOpWait"]        = checkInList(isaVersion, {{9, 4, 2}, {9, 5, 0}, {12, 5, 0}});
    rv["SDWAWait"]           = checkInList(isaVersion, {{9, 4, 2}, {9, 5, 0}, {12, 5, 0}});
    rv["VgprBank"]           = checkInList(isaVersion[0], {10, 11, 12});
    rv["DSLow16NotPreserve"] = isaVersion[0] == 12;
    rv["WorkGroupIdFromTTM"] = isaVersion[0] == 12;
    rv["NoSDWA"]             = checkInList(isaVersion[0], {11, 12});
    rv["VOP3ByteSel"]        = isaVersion[0] == 12;
    rv["HasFP8_OCP"]         = isaVersion[0] == 12;
    rv["HasWmmaArbStallBit"] = isaVersion[0] == 12 && isaVersion[1] == 5;
    rv["HasF32XEmulation"]   = checkInList(isaVersion, {{9, 5, 0}, {12, 5, 0}});
    rv["MaxSgprPreload"]     = checkInList(isaVersion, {{12, 5, 0}}) ? 32 : 16;
    rv["SgprPreloadPad"]     = checkInList(isaVersion, {{9, 5, 0}}) || checkInList(isaVersion, {{9, 0, 10}}) || (isaVersion[0] == 9 && isaVersion[1] == 4);

    // True on archs whose MFMA-scale path can consume a swizzled MX scale
    // layout: gfx950 (HostPreSwizzle via the subtile path) and gfx1250
    // (InMemorySwizzle via TDM). The kernel-side MXScaleFormat enum and the
    // Solution.py guards decide which concrete layout is in use.
    rv["HasMXScaleSwizzle"]            = checkInList(isaVersion, {{9, 5, 0}, {12, 5, 0}});

    // Cross-CU/L2 release+acquire fences for device-scope inter-workgroup
    // synchronization (e.g. StreamK partial-tile handshake). When set, a
    // store-release sequence must emit `s_wait_loadcnt 0; s_wait_storecnt 0;
    // global_wb scope:SCOPE_DEV` before the flag store, and a load-acquire
    // sequence must emit `global_inv scope:SCOPE_DEV; s_wait_loadcnt 0` after
    // the flag load.
    rv["HasInvWbDevFences"]            = checkInList(isaVersion, {{12, 5, 0}});

    // XNACK-replay drain. When set, in-flight VMEM ops can be replayed and
    // therefore reorder w.r.t. a subsequent volatile/atomic VMEM. An
    // `s_wait_xcnt 0` must precede the volatile/atomic VMEM op.
    rv["RequiresXCntForVolatileVMEM"]  = checkInList(isaVersion, {{12, 5, 0}});
    rv["DefaultScopeIsCULocal"]        = checkInList(isaVersion, {{12, 5, 0}});

    // LDS bank geometry — used for swizzle/rotation in subtile-based tiling.
    rv["LDSBankCount"] = 64;
    rv["LDSBankWidth"] = 4; // bytes per bank

    return rv;
}

inline std::map<std::string, int> initRegisterCaps(const IsaVersion&           isaVersion,
                                                   std::map<std::string, int>& archCaps)
{
    std::map<std::string, int> rv;
    // 1024 vgpr
    rv["MaxVgpr"] = isaVersion[0] == 12 && isaVersion[1] == 5? 1024 : 256;
    // max allowed is 112 out of 112 , 6 is used by hardware 4 SGPRs are wasted
    rv["MaxSgpr"] = 102;
    rv["PhysicalMaxVgpr"] = isaVersion[0] == 12 && isaVersion[1] == 5? 1024 : 512;
    rv["PhysicalMaxSgpr"]   = 800;
    rv["maxLDSConstOffset"] = 65536;
    rv["GlobalPrefetchSize"] = 256;

    if(isaVersion[0] == 10)
        rv["PhysicalMaxVgprCU"] = 1024 * 32;
    else if(isaVersion[0] == 11)
        // Code path for gfx11XX
        if(isaVersion[1] == 5)
        {
            // Code path for gfx115X
            if(isaVersion[2] == 0 || isaVersion[2] == 2 || isaVersion[2] == 3)
                // gfx1150, gfx1152, gfx1153
                rv["PhysicalMaxVgprCU"] = 2 /*two SIMDs per CU*/ * 1024 * 32;
            if(isaVersion[2] == 1)
                // gfx1151
                rv["PhysicalMaxVgprCU"] = 2 /*two SIMDs per CU*/ * 1536 * 32;
        }
        else if(isaVersion[2] == 2)
            rv["PhysicalMaxVgprCU"] = 1024 * 32;
        else
            rv["PhysicalMaxVgprCU"] = 1536 * 32;
    else if(isaVersion[0] == 12)
        rv["PhysicalMaxVgprCU"] = isaVersion[1] == 5? 4096 * 32 : 1536 * 32;
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
