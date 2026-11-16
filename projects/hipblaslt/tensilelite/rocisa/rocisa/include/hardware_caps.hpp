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
#include <algorithm>
#include <array>
#include <cstdlib>
#include <iostream>
#include <map>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "helper.hpp"

class AssemblerProbes
{
public:
    AssemblerProbes(const IsaVersion& isa, const std::string& assemblerPath, bool debug)
        : isaVersion(isa)
        , isDebug(debug)
    {
        // Assemble only: linking would write a.out into the working directory.
        commands[0] = {assemblerPath,
                       "-x",
                       "assembler",
                       "-target",
                       "amdgcn-amdhsa",
                       "-mcpu=" + getGfxNameTuple(isaVersion),
                       "-c",
                       "-o",
                       nullDevicePath(),
                       "-"};
        commands[1] = commands[0];
        if(isaVersion[0] >= 10)
            commands[0].insert(commands[0].end() - 1, "-mwavefrontsize64");
    }

    std::map<std::string, int> capabilities()
    {
        // Unanswered probes count as rejected until the next batch. Repeating
        // the definitions discovers probes reached through conditional expressions.
        for(;;)
        {
            auto caps = probeAsmCaps();
            if(!answerPendingProbes())
                return caps;
        }
    }

private:
    using Command = std::vector<std::string>;
    using Key     = std::pair<Command, std::string>;

    IsaVersion             isaVersion;
    bool                   isDebug;
    std::array<Command, 2> commands; // default wave size, then wave32
    std::optional<Command> rejectedEmptySourceCommand;
    // An empty optional identifies a probe to answer in the next batch.
    std::map<Key, std::optional<bool>> answers;

    // An assembler that could not be executed at all (e.g. a missing binary) is not
    // the same as one rejecting this particular ISA/instruction.
    static std::pair<int, std::string> runAssembler(std::vector<std::string> cmd,
                                                    const std::string&       source,
                                                    bool                     debug,
                                                    bool                     captureOutput = false)
    {
        std::vector<char*> args(cmd.size());
        std::transform(cmd.begin(), cmd.end(), args.begin(), [](auto& str) { return &str[0]; });
        args.push_back(nullptr);
        auto result = run(args, source, debug, captureOutput);
        if(result.first == execFailedExitCode)
            throw std::runtime_error("Assembler not found or not executable: " + cmd[0]);
        return result;
    }

    // The source lines named by "<stdin>:LINE:COLUMN: error:" diagnostics, or nothing
    // if the output reports an error without naming a line.
    static std::optional<std::set<int>> rejectedLines(const std::string& output)
    {
        const std::string  prefix = "<stdin>:";
        std::set<int>      lines;
        std::istringstream in(output);
        for(std::string line; std::getline(in, line);)
        {
            if(line.find("error:") == std::string::npos)
                continue;
            if(line.compare(0, prefix.size(), prefix) != 0)
                return std::nullopt;
            char* rest;
            long  number = std::strtol(line.c_str() + prefix.size(), &rest, 10);
            if(*rest != ':')
                return std::nullopt;
            std::strtol(rest + 1, &rest, 10);
            if(std::string(rest).compare(0, 8, ": error:") != 0)
                return std::nullopt;
            lines.insert(static_cast<int>(number));
        }
        return lines;
    }

    bool tryAssembler(const std::string& source, bool isWave32 = false)
    {
        const auto& cmd = commands[isWave32];
        if(!isDebug)
        {
            // A rejected empty source rules out this command line, but not the
            // probes using different wave-size options.
            if(rejectedEmptySourceCommand == cmd)
                return false;
            auto& answer = answers[{cmd, source}];
            if(answer.has_value())
                return *answer;
            if(!source.empty())
                return false;
        }
        // The empty source runs immediately to decide whether other probes are
        // needed. Debug mode runs every probe individually and records no state.
        auto [rcode, result] = runAssembler(cmd, source, isDebug);
        if(isDebug)
        {
            std::string commandLine;
            for(const auto& arg : cmd)
                commandLine += arg + " ";
            std::cout << "isaVersion: " << getGfxNameTuple(isaVersion) << std::endl;
            std::cout << "asm_cmd: " << commandLine << std::endl;
            std::cout << "asmString: " << source << std::endl;
            std::cout << "result: " << result << std::endl;
            std::cout << "return code: " << rcode << std::endl;
        }
        else
        {
            if(rcode == 1 && source.empty())
                rejectedEmptySourceCommand = cmd;
            answers[{cmd, source}] = rcode == 0;
        }
        return rcode == 0;
    }

    // Only complete, line-attributed rejections can answer a batch. Other
    // failures fall back to probing each source separately.
    bool answerPendingProbes()
    {
        std::map<Command, std::vector<std::string>> batches;
        for(const auto& [key, answer] : answers)
            if(!answer.has_value())
                batches[key.first].push_back(key.second);
        if(batches.empty())
            return false;
        for(auto& [cmd, sources] : batches)
        {
            std::string                      input;
            std::vector<std::pair<int, int>> spans; // first and last line of each source
            for(auto& source : sources)
            {
                int first = spans.empty() ? 1 : spans.back().second + 1;
                spans.emplace_back(first, first + std::count(source.begin(), source.end(), '\n'));
                input += source + "\n";
            }
            auto [rcode, output] = runAssembler(cmd, input, false, true);
            auto rejected        = rejectedLines(output);
            bool readable
                = rejected
                  && ((rcode == 0 && rejected->empty()) || (rcode == 1 && !rejected->empty()));
            for(size_t i = 0; i < sources.size(); ++i)
            {
                bool accepted;
                if(readable)
                {
                    auto line = rejected->lower_bound(spans[i].first);
                    accepted  = line == rejected->end() || *line > spans[i].second;
                }
                else
                    accepted = runAssembler(cmd, sources[i], false).first == 0;
                answers[{cmd, sources[i]}] = accepted;
            }
        }
        return true;
    }

    int getMaxCnt(const std::string& prefix, const std::string& suffix)
    {
        for(int p = 64; p > 1; p >>= 1)
        {
            // Try ( pow(2) - 1 ) from high to low
            if(tryAssembler(prefix + std::to_string(p - 1) + suffix))
                return p - 1;
        }
        return 0;
    }

    std::map<std::string, int> probeAsmCaps()
    {
        // Determine assembler capabilities by testing short instructions sequences
        std::map<std::string, int> rv;
        rv["SupportedISA"] = tryAssembler("");

        rv["HasPartialOOB"] = checkNotInList(isaVersion, {{12, 5, 0}});

        rv["HasExplicitCO"] = tryAssembler("v_add_co_u32 v0,vcc,v0,1")
                              || tryAssembler("v_add_co_u32 v0,vcc_lo,v0,1");
        rv["HasExplicitNC"] = tryAssembler("v_add_nc_u32 v0,v0,1");

        rv["HasDirectToLds"]
            = tryAssembler("buffer_load_dword v36, s[24:27], s28 offen offset:0 lds")
              || tryAssembler("buffer_load_b32 v36, s[24:27], s28 offen offset:0 lds");
        rv["HasDirectToLdsx4"]
            = tryAssembler("buffer_load_dwordx4 v36, s[24:27], s28 offen offset:0 lds")
              || tryAssembler("buffer_load_b128 v36, s[24:27], s28 offen offset:0 lds");
        rv["HasAddLshl"]     = tryAssembler("v_add_lshl_u32 v47, v36, v34, 0x2");
        rv["HasLshlOr"]      = tryAssembler("v_lshl_or_b32 v47, v36, 0x2, v34");
        rv["HasSMulHi"]      = tryAssembler("s_mul_hi_u32 s47, s36, s34");
        rv["HasScalarStore"] = tryAssembler("s_store_dword s79, s[70:71], s77")
                               || tryAssembler("s_store_b32 s79, s[70:71], s77");
        rv["HasSAtomic"]       = tryAssembler("s_atomic_dec s11, s[0:1]");
        rv["HasMFMA_explictB"] = tryAssembler("v_mfma_f32_32x32x1_2b_f32 a[0:31], v0, v1, a[0:31]");
        rv["HasMFMA"]          = tryAssembler("v_mfma_f32_32x32x2bf16 a[0:31], v32, v33, a[0:31]")
                        || rv["HasMFMA_explictB"];
        rv["HasMFMA_f64"]
            = tryAssembler("v_mfma_f64_16x16x4f64 v[0:7], v[32:33], v[36:37], v[0:7]")
              || tryAssembler("v_mfma_f64_16x16x4_f64 v[0:7], v[32:33], v[36:37], v[0:7]");
        rv["HasMFMA_bf16_1k"]
            = tryAssembler("v_mfma_f32_32x32x4bf16_1k a[0:31], v[32:33], v[36:37], a[0:31]");
        rv["HasMFMA_f8"]
            = tryAssembler("v_mfma_f32_16x16x32_fp8_fp8 a[0:3], v[2:3], v[4:5], a[0:3]");
        rv["HasMFMA_b8"]
            = tryAssembler("v_mfma_f32_16x16x32_bf8_bf8 a[0:3], v[2:3], v[4:5], a[0:3]");
        rv["HasMFMA_f8f6f4"]
            = tryAssembler("v_mfma_f32_16x16x128_f8f6f4 a[0:3], v[0:7], v[8:15], a[0:3]");

        rv["HasMFMA_xf32"]
            = tryAssembler("v_mfma_f32_32x32x4_xf32 a[0:15], v[32:33], v[36:37], a[0:15]");
        rv["HasSMFMA"]
            = tryAssembler("v_smfmac_f32_32x32x16_f16 a[0:15], v[32:33], v[36:39], v[40]");
        rv["HasWMMA"] = tryAssembler("v_wmma_f32_16x16x16_f16 v[0:3], v[8:15], v[16:23], v[0:3]")
                        || tryAssembler("v_wmma_f32_16x16x16_f16 v[0:3], v[8:9], v[16:17], v[0:3]")
                        || tryAssembler("v_wmma_f32_16x16x32_bf16 v[0:7], v[8:15], v[8:15], v[0:7]")
                        || tryAssembler("v_wmma_f32_16x16x4_f32 v[0:7], v[8:9], v[8:9], v[0:7]");
        rv["HasWMMA_V1"]
            = tryAssembler("v_wmma_f32_16x16x16_f16 v[0:3], v[8:15], v[16:23], v[0:3]");
        rv["HasWMMA_V2"] = tryAssembler("v_wmma_f32_16x16x16_f16 v[0:3], v[8:9], v[16:17], v[0:3]");
        rv["HasWMMA_V3"] = tryAssembler("v_wmma_f32_16x16x32_bf16 v[0:7], v[8:15], v[8:15], v[0:7]")
                           || tryAssembler("v_wmma_f32_16x16x4_f32 v[0:7], v[8:9], v[8:9], v[0:7]");
        // If HasWMMA_V3 is set, force HasWMMA_V2 off so the wmma_v3 path is used
        if(rv["HasWMMA_V3"])
            rv["HasWMMA_V2"] = false;
        rv["HasWMMA_V3_f64"]
            = tryAssembler("v_wmma_f64_16x16x4_f64 v[0:15], v[16:19], v[20:23], v[0:15]")
              || tryAssembler("v_wmma_f64_16x16x8_f64 v[0:15], v[16:23], v[24:31], v[0:15]");

        rv["HasWMMA_f8f6f4"]
            = tryAssembler("v_wmma_f32_16x16x128_f8f6f4 v[0:7], v[16:31], v[16:31], v[0:7]");

        // InitCIterWmma replaces the v_mov initC with a WMMA using src C = immediate 0
        rv["HasWMMA_AccImmZero"]
            = tryAssembler("v_wmma_f32_16x16x32_bf16 v[0:7], v[8:15], v[8:15], 0")
              && tryAssembler("v_wmma_f32_16x16x4_f32 v[0:7], v[8:9], v[8:9], 0");

        rv["HasAdd_PC_i64"] = false;

        rv["HasSWMMAC"]
            = tryAssembler("v_swmmac_f32_16x16x32_f16 v[0:3], v[32:33], v[36:39], v[44]")
              || tryAssembler("v_swmmac_f32_16x16x32_f16 v[0:7], v[32:35], v[36:43], v[44]", true)
              || tryAssembler("v_swmmac_f32_16x16x64_f16 v[0:7], v[8:15], v[16:31], v[44]", true);
        rv["HasSWMMAC_gfx1250"]
            = tryAssembler("v_swmmac_f32_16x16x64_f16 v[0:7], v[8:15], v[16:31], v[44]", true);

        rv["v_mac_f16"] = tryAssembler("v_mac_f16 v47, v36, v34");

        rv["v_fma_f16"]  = tryAssembler("v_fma_f16 v47, v36, v34, v47, op_sel:[0,0,0,0]");
        rv["v_fmac_f16"] = tryAssembler("v_fma_f16 v47, v36, v34");

        rv["v_pk_fma_f16"]  = tryAssembler("v_pk_fma_f16 v47, v36, v34, v47, op_sel:[0,0,0]");
        rv["v_pk_fmac_f16"] = tryAssembler("v_pk_fma_f16 v47, v36, v34");

        rv["v_pk_add_f32"] = tryAssembler("v_pk_add_f32 v[48:49], v[36:37], v[0:1]");
        rv["v_pk_mul_f32"] = tryAssembler("v_pk_mul_f32 v[48:49], v[36:37], v[0:1]");

        rv["v_pk_mul_f32"] = tryAssembler("v_pk_mul_f32 v[20:21], v[18:19], v[20:21]");

        rv["v_pk_fma_f32"] = tryAssembler("v_pk_fma_f32 v[0:1], v[2:3], v[4:5], v[6:7]");

        rv["v_mad_mix_f32"]
            = tryAssembler("v_mad_mix_f32 v47, v36, v34, v47, op_sel:[0,0,0] op_sel_hi:[1,1,0]");
        rv["v_fma_mix_f32"]
            = tryAssembler("v_fma_mix_f32 v47, v36, v34, v47, op_sel:[0,0,0] op_sel_hi:[1,1,0]");

        rv["v_dot2_f32_f16"]  = tryAssembler("v_dot2_f32_f16 v20, v36, v34, v20");
        rv["v_dot2c_f32_f16"] = tryAssembler("v_dot2c_f32_f16 v47, v36, v34")
                                || tryAssembler("v_dot2acc_f32_f16 v47, v36, v34");

        rv["v_dot2_f32_bf16"]  = tryAssembler("v_dot2_f32_bf16 v20, v36, v34, v20");
        rv["v_dot2c_f32_bf16"] = tryAssembler("v_dot2c_f32_bf16 v47, v36, v34")
                                 || tryAssembler("v_dot2acc_f32_bf16 v47, v36, v34");

        rv["v_dot4_i32_i8"]     = tryAssembler("v_dot4_i32_i8 v47, v36, v34");
        rv["v_dot4c_i32_i8"]    = tryAssembler("v_dot4c_i32_i8 v47, v36, v34");
        rv["VOP3v_dot4_i32_i8"] = tryAssembler("v_dot4_i32_i8 v47, v36, v34, v47");

        rv["v_mac_f32"]  = tryAssembler("v_mac_f32 v20, v21, v22");
        rv["v_fma_f32"]  = tryAssembler("v_fma_f32 v20, v21, v22, v23");
        rv["v_fmac_f32"] = tryAssembler("v_fmac_f32 v20, v21, v22");
        // VOPD dual-issue FMA (RDNA3/3.5/4 only); used to gate UseDualFMAC. VOPD is wave32-only,
        // so probe with isWave32=true (tryAssembler otherwise adds -mwavefrontsize64 for gfx10+).
        rv["v_dual_fmac_f32"]
            = tryAssembler("v_dual_fmac_f32 v0, v1, v2 :: v_dual_fmac_f32 v3, v4, v5", true);

        rv["v_fma_f64"] = tryAssembler("v_fma_f64 v[20:21], v[22:23], v[24:25], v[20:21]");

        rv["v_mov_b64"]    = tryAssembler("v_mov_b64 v[0:1], v[2:3]");
        rv["s_sub_u64"]    = tryAssembler("s_sub_u64 s[0:1], s[0:1], s[2:3]");
        rv["s_add_u64"]    = tryAssembler("s_add_u64 s[0:1], s[0:1], s[2:3]");
        rv["v_add_nc_u64"] = tryAssembler("v_add_nc_u64 v[0:1], v[2:3], v[4:5]");

        rv["HasBF16CVT"] = tryAssembler("v_cvt_f32_bf16 v0, v1");

        rv["HasPkF16CVT"] = tryAssembler("v_cvt_pk_f16_f32 v0, v1, v2");

        rv["Hascvtfp8_f16"]
            = tryAssembler("v_cvt_scalef32_pk_fp8_f16 v[0], v[1], 0 op_sel:[0,0,0,0]");
        rv["Hascvtf16_fp8_sf32"]
            = tryAssembler("v_cvt_scalef32_f16_fp8 v[0], v[1], 0 op_sel:[0,0,0,0]");
        rv["HasCvtFP8toF16"]        = tryAssembler("v_cvt_f16_fp8 v[0], v[1] byte_sel:2");
        rv["HasCvtScalePk8Fp8F32"]  = tryAssembler("v_cvt_scalef32_pk8_fp8_f32 v[0:1], v[2:9], s0");
        rv["HasCvtScalePk8Bf8F32"]  = tryAssembler("v_cvt_scalef32_pk8_bf8_f32 v[0:1], v[2:9], s0");
        rv["HasVMaximumMinimumF32"] = tryAssembler("v_maximumminimum_f32 v0, v1, v2, v3");

        rv["HasLDSTrB64B16"] = tryAssembler("ds_read_b64_tr_b16 v[0:1], v0 offset: 0");

        rv["HasGLTr8B64"] = tryAssembler("global_load_tr_b64 v[0], v0, s[0:1], offset:0");

        rv["HasGLTr16B128"] = tryAssembler("global_load_tr_b128 v[0:1], v0, s[0:1], offset:0");

        rv["HasLDSTrB128B16"] = tryAssembler("ds_load_tr16_b128 v[0:3], v0 offset: 0");

        rv["HasLDSTrB64B8"] = tryAssembler("ds_load_tr8_b64 v[0:1], v0 offset: 0");

        rv["HasLDSTrB64B4"] = tryAssembler("ds_load_tr4_b64 v[0:1], v0 offset: 0");

        rv["HasLDSTrB96B6"] = tryAssembler("ds_load_tr6_b96 v[0:2], v0 offset: 0");

        rv["HasLDSTr"] = rv["HasLDSTrB64B16"] || rv["HasLDSTrB128B16"] || rv["HasLDSTrB64B8"]
                         || rv["HasLDSTrB64B4"];

        rv["v_prng_b32"] = tryAssembler("v_prng_b32 v47, v36");

        // FP8 stochastic rounding pk8 conversion with scale
        rv["HasScaleSRPk8Cvt"]
            = tryAssembler("v_cvt_scalef32_sr_pk8_fp8_f32 v[0:1], v[0:7], v0, 1.0");

        rv["HasAtomicAdd"]
            = tryAssembler("buffer_atomic_add_f32 v0, v1, s[0:3], 0 offen offset:0")
              || tryAssembler("buffer_atomic_add_f32 v0, v1, s[0:3], null offen offset:0");
        // Packed 2xBF16 atomic add (gfx950 / gfx1250+). gfx950 takes the literal 0
        // soffset form, gfx1250 requires null, so probe both as HasAtomicAdd does.
        rv["HasAtomicPkAddBF16"]
            = tryAssembler("buffer_atomic_pk_add_bf16 v0, v1, s[0:3], 0 offen offset:0")
              || tryAssembler("buffer_atomic_pk_add_bf16 v0, v1, s[0:3], null offen offset:0");
        rv["HasGLCModifier"]
            = tryAssembler("buffer_load_dwordx4 v[10:13], v[0], s[0:3], 0, offen offset:0, glc")
              || tryAssembler(
                  "buffer_load_dwordx4 v[10:13], v[0], s[0:3], null, offen offset:0, glc");
        rv["HasSC0Modifier"]
            = tryAssembler("buffer_load_dwordx4 v[10:13], v[0], s[0:3], 0, offen offset:0, sc0")
              || tryAssembler(
                  "buffer_load_dwordx4 v[10:13], v[0], s[0:3], null, offen offset:0, sc0");
        rv["HasDLCModifier"]
            = tryAssembler("buffer_load_dwordx4 v[10:13], v[0], s[0:3], 0, offen offset:0, dlc")
              || tryAssembler(
                  "buffer_load_dwordx4 v[10:13], v[0], s[0:3], null, offen offset:0, dlc");
        rv["HasSCOPEModifier"]
            = tryAssembler(
                  "buffer_load_dwordx4 v[10:13], v[0], s[0:3], 0, offen offset:0, scope:SCOPE_DEV")
              || tryAssembler("buffer_load_dwordx4 v[10:13], v[0], s[0:3], null offen offset:0, "
                              "scope:SCOPE_DEV");
        rv["HasMUBUFConst"]
            = tryAssembler("buffer_load_dword v40, v36, s[24:27], 1 offen offset:0")
              || tryAssembler("buffer_load_b32 v40, v36, s[24:27], 1 offen offset:0");
        rv["HasSCMPK"] = tryAssembler("s_cmpk_gt_u32 s56, 0x0");

        rv["HasGLCModifier"]
            = tryAssembler("buffer_load_dwordx4 v[10:13], v[0], s[0:3], 0, offen offset:0, glc");

        rv["HasNTModifier"]
            = tryAssembler("buffer_load_dwordx4 v[10:13], v[0], s[0:3], 0, offen offset:0, nt");

        // gfx1250 replaces the bare 'nt' bit with a 3-bit Temporal Hint
        // (th:TH_LOAD_*/TH_STORE_*) and adds a per-op Non-Volatile (nv) bit.
        // (Volatile and Non-Volatile Memory Access).  Probe both with the buffer_*
        // form because gfx12 supports the same modifier syntax for buffer / global / flat.
        rv["HasTHModifier"]
            = tryAssembler(
                  "buffer_load_dwordx4 v[10:13], v[0], s[0:3], 0 offen offset:0 th:TH_LOAD_NT")
              || tryAssembler(
                  "buffer_load_dwordx4 v[10:13], v[0], s[0:3], null offen offset:0 th:TH_LOAD_NT");
        rv["HasNVModifier"]
            = tryAssembler("buffer_load_dwordx4 v[10:13], v[0], s[0:3], 0 offen offset:0 nv")
              || tryAssembler("buffer_load_dwordx4 v[10:13], v[0], s[0:3], null offen offset:0 nv");

        rv["HasNewBarrier"]     = tryAssembler("s_barrier_wait -1");
        rv["HasClusterBarrier"] = tryAssembler("s_barrier_wait -3");
        rv["HasTDM"]            = tryAssembler("tensor_load_to_lds s[0:3], s[4:11]");
        // v_movrelsd_2_b32: indirect-VGPR-write move used by CompactLoopStore's
        // per-iter "copy MI out reg" body. Only some archs (gfx1250) have it, so
        // probe the assembler and gate CompactLoopStore on this cap in Solution.py.
        rv["HasMovRelsD2B32"] = tryAssembler("v_movrelsd_2_b32 v0, v1");

        rv["s_delay_alu"] = tryAssembler("s_delay_alu instid0(VALU_DEP_1)");
        rv["HasVgprMSB"]  = tryAssembler("s_set_vgpr_msb 0");
        // 16-bit MSB form packs the previous-instruction MSB in the high byte
        // (e.g. 0x0101). Some assemblers accept only the 8-bit form, so probe
        // the wider encoding separately.
        rv["HasVgprMSB16"] = rv["HasVgprMSB"] && tryAssembler("s_set_vgpr_msb 0x0101");
        // workaround: as we generate s_set_vgpr_msb in toString(), we can't calculate inst len correctly.
        rv["ShortBranchMaxLength"] = rv["HasVgprMSB"] ? 8192 : 16384;

        rv["SeparateVscnt"] = tryAssembler("s_waitcnt_vscnt null 0");
        rv["HasGlobalPrefetch"]
            = tryAssembler("global_prefetch_b8 v[0:1], off scope:SCOPE_SE th:TH_LOAD_NT");

        rv["SeparateLGKMcnt"] = tryAssembler("s_wait_dscnt 0") && tryAssembler("s_wait_kmcnt 0");

        rv["SeparateVMcnt"] = tryAssembler("s_wait_loadcnt 0") && tryAssembler("s_wait_storecnt 0");

        if(rv["SeparateVMcnt"])
        {
            // s_wait_loadcnt accept 16 bits immediate, but only use the lowest 6 bits are used, can't use tryAssembler
            rv["MaxLoadcnt"] = 63;
            // s_wait_storecnt accept 16 bits immediate, but only use the lowest 6 bits are used, can't use tryAssembler
            rv["MaxStorecnt"] = 63;
        }
        else
        {
            rv["MaxVmcnt"] = getMaxCnt("s_waitcnt vmcnt(", ")");
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
            rv["MaxLgkmcnt"] = getMaxCnt("s_waitcnt lgkmcnt(", ")");
        }

        // s_wait_xcnt drains XNACK-replay tracking before a subsequent volatile/atomic
        // VMEM op. Probed independently because not every arch with separate vmem
        // counters supports xcnt (e.g. it is gfx1250+). The immediate accepts 16 bits
        // but hardware only consumes the lowest 6 bits, matching loadcnt/storecnt.
        rv["HasXcnt"] = tryAssembler("s_wait_xcnt 0");
        if(rv["HasXcnt"])
        {
            rv["MaxXcnt"] = 63;
        }

        rv["SupportedSource"] = true;

        return rv;
    }
};

inline std::map<std::string, int>
    initAsmCaps(const IsaVersion& isaVersion, const std::string& assemblerPath, bool isDebug)
{
    return AssemblerProbes(isaVersion, assemblerPath, isDebug).capabilities();
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
    // LDS allocation granule: a workgroup occupies a whole number of granules, so this
    // rounding can cost a resident workgroup at a boundary. gfx11 allocates 1024 bytes
    // at a time, other targets 256. TODO: gfx10/gfx12 might also need a different
    // granularity.
    rv["LdsGranularity"]     = isaVersion[0] == 11 ? 1024 : 256;
    rv["CMPXWritesSGPR"]     = checkNotInList(isaVersion[0], {10, 11, 12});
    rv["HasWave32"]          = checkInList(isaVersion[0], {10, 11, 12});
    // GFX12.5 always uses wave32 and reserves the descriptor's wave-size selector.
    // Keep HasWave32 for solution selection and validation.
    rv["HasWavefrontSize32Directive"]
        = rv["HasWave32"] && !(isaVersion[0] == 12 && isaVersion[1] == 5);
    rv["HasSchedMode"]       = checkInList(isaVersion[0], {12});
    rv["HasAccCD"]           = checkInList(isaVersion, {{9, 0, 10}, {9, 4, 2}, {9, 5, 0}});
    rv["ArchAccUnifiedRegs"] = checkInList(isaVersion, {{9, 0, 10}, {9, 4, 2}, {9, 5, 0}});
    // Max concurrent waves per SIMD: 8 for ArchAccUnifiedRegs (gfx90a/gfx942/gfx950), 10 otherwise.
    if(rv["ArchAccUnifiedRegs"])
        rv["MaxWavesPerSimd"] = 8;
    else if(isaVersion[0] == 11 || (isaVersion[0] == 12 && isaVersion[1] != 5))
        rv["MaxWavesPerSimd"] = 16;
    else
        rv["MaxWavesPerSimd"] = 10;
    rv["CrosslaneWait"]      = checkInList(isaVersion, {{9, 4, 2}, {9, 5, 0}});
    rv["TransOpWait"]        = checkInList(isaVersion, {{9, 4, 2}, {9, 5, 0}, {12, 5, 0}});
    rv["SDWAWait"]           = checkInList(isaVersion, {{9, 4, 2}, {9, 5, 0}, {12, 5, 0}});
    rv["VgprBank"]           = checkInList(isaVersion[0], {10, 11, 12});
    rv["DSLow16NotPreserve"] = isaVersion[0] == 12 || checkInList(isaVersion, {{11, 7, 0}, {11, 7, 1}});
    rv["WorkGroupIdFromTTM"] = isaVersion[0] == 12;
    rv["NoSDWA"]             = checkInList(isaVersion[0], {11, 12});
    rv["VOP3ByteSel"]        = isaVersion[0] == 12 || checkInList(isaVersion, {{11, 7, 0}, {11, 7, 1}});
    rv["HasFP8_OCP"]         = isaVersion[0] == 12 || checkInList(isaVersion, {{11, 7, 0}, {11, 7, 1}});
    rv["HasWmmaArbStallBit"] = isaVersion[0] == 12 && isaVersion[1] == 5;
    // Bit position of DISABLE_XDL_ARB_STALL within SCHED_MODE (HWREG 26).
    // -1 where the field does not exist; 0 would alias DEP_MODE's LSB.
    int arbStallBit = -1;
    if(checkInList(isaVersion, {{12, 5, 0}}))
        arbStallBit = 2;
    rv["WmmaArbStallBitOffset"] = arbStallBit;
    rv["HasF32XEmulation"]   = checkInList(isaVersion, {{9, 5, 0}, {12, 5, 0}});
    // F32X (xf32/TF32) emulation Inf handling. The 3xBF16 split makes an Inf
    // input produce Inf - Inf = NaN in the low-bits residual, so an extra
    // v_cmp_class Inf guard is required to keep Inf inputs producing Inf. This is
    // a hardware/platform behavior decision (not a perf-tuning knob), so it lives
    // in the arch caps: enabled only on archs that require correct Inf handling
    // under emulation. gfx950's MFMA emulation path leaves it off (the guard adds
    // VALU ops in the pack path and regresses perf ~10%+).
    rv["F32XEmulationInfSupport"] = checkInList(isaVersion, {{12, 5, 0}});
    rv["MaxSgprPreload"]     = checkInList(isaVersion, {{12, 5, 0}}) ? 32 : 16;
    rv["SgprPreloadPad"]     = checkInList(isaVersion, {{9, 5, 0}}) || checkInList(isaVersion, {{9, 0, 10}}) || (isaVersion[0] == 9 && isaVersion[1] == 4);
    // Separation a consumer must leave after an MFMA before reading its ACC.
    //
    // This is the *result* latency, not the issue latency: issue latency says
    // when the next instruction may issue (miIssueLatency, 1 quad-cycle here),
    // while this says when the result becomes readable. getMFMAIssueLatency
    // returns both; its first element, matrixInstM / mi_divisor, is this one,
    // in quad-cycles -- 16/4 = 4 for the gfx950 16x16 MFMA, so 16 cycles. The
    // value below is in cycles rather than quad-cycles.
    //
    // 17, not 16, because the result is readable strictly *after* those 16
    // cycles elapse; against a >= comparison that threshold is 17.
    //
    // Keyed on the architecture, but derived for matrixInstM == 16. The 32x32
    // shapes have a longer result latency, so a consumer that can see them
    // must not use this as-is.
    rv["MfmaToAccReadLatency"] = checkInList(isaVersion, {{9, 5, 0}}) ? 17 : 0;

    // True on archs whose MFMA-scale path can consume a swizzled MX scale
    // layout: gfx950 (HostPreSwizzle via the subtile path) and gfx1250
    // (InMemorySwizzle via TDM). The kernel-side MXScaleFormat enum and the
    // Solution.py guards decide which concrete layout is in use.
    rv["HasMXScaleSwizzle"]            = checkInList(isaVersion, {{9, 5, 0}, {12, 5, 0}});

    // Explicit cache fences and VMEM flags for inter-workgroup synchronization.
    // MTypeNC determines whether L2 writeback/device-scope invalidation is needed.
    rv["HasInvWbDevFences"]            = checkInList(isaVersion, {{12, 5, 0}});

    // Non-strict gfx1250 excludes NC mappings; its RW mappings have
    // hardware-coherent L2s. This ISA-only default cannot distinguish strict.
    // TensileLite overrides MTypeNC to True for gfx1250-strict, which shares
    // the ISA tuple but permits NC mappings.
    rv["MTypeNC"]                     = !checkInList(isaVersion, {{12, 5, 0}});

    // gfx950 splits L2 across 8 XCDs. StreamK partial-tile fixup needs
    // VMEM flags with glc+slc and waitcnt fences for cross-XCD coherence;
    // gfx1250 uses HasInvWbDevFences (global_wb / global_inv) instead.
    rv["HasXCDSplitL2"]                = checkInList(isaVersion, {{9, 5, 0}});

    // XNACK-replay drain. When set, in-flight VMEM ops can be replayed and
    // therefore reorder w.r.t. a subsequent volatile/atomic VMEM. An
    // `s_wait_xcnt 0` must precede the volatile/atomic VMEM op.
    rv["RequiresXCntForVolatileVMEM"]  = checkInList(isaVersion, {{12, 5, 0}});
    rv["EnableXnackReplay"]            = checkInList(isaVersion, {{12, 5, 0}});
    rv["DefaultScopeIsCULocal"]        = checkInList(isaVersion, {{12, 5, 0}});

    // LDS bank geometry — used for swizzle/rotation in subtile-based tiling.
    rv["LDSBankCount"] = 64;
    rv["LDSBankWidth"] = 4; // bytes per bank

    // Per-XCD work-queue count baked into StreamK dynamic-queue kernels. Single
    // codegen-side mirror of origami get_default_num_xcds(): gfx942/gfx950 bake
    // 8 (the MI300X value), every other arch 1. gfx942 covers BOTH MI300X (8
    // XCDs) and MI300A (6 XCDs), which codegen cannot tell apart, so it always
    // bakes 8; the host guard rejects a device whose runtime NUM_XCD != this
    // baked value (so MI300A's 6 is excluded at runtime). Power-of-two keeps the
    // StreamK queue masking (AND/shift) valid.
    rv["NumXCD"] = checkInList(isaVersion, {{9, 4, 2}, {9, 5, 0}}) ? 8 : 1;

    // Per-queue counter stride = L2 cache-line size (uniform 128B on supported archs).
    rv["CacheLineBytes"] = 128;

    return rv;
}

inline std::map<std::string, int> initRegisterCaps(const IsaVersion&           isaVersion,
                                                   std::map<std::string, int>& archCaps)
{
    std::map<std::string, int> rv;
    // 1024 vgpr
    rv["MaxVgpr"] = isaVersion[0] == 12 && isaVersion[1] == 5? 1024 : 256;
    // Highest addressable SGPR index plus one. gfx8/gfx9 stop at s101 (102); every
    // RDNA target (gfx10, gfx11, gfx12) addresses s0-s105 (106).
    rv["MaxSgpr"] = isaVersion[0] >= 10 ? 106 : 102;
    rv["PhysicalMaxVgpr"] = isaVersion[0] == 12 && isaVersion[1] == 5? 1024 : 512;
    // gfx11 (RDNA) does not have an SGPR-file occupancy limit; use a large value so it never binds.
    // TODO: gfx10/gfx12 are RDNA too and carry the same phantom limit.
    rv["PhysicalMaxSgpr"]   = isaVersion[0] == 11 ? 1696 : 800;
    rv["maxLDSConstOffset"] = 65536;
    rv["GlobalPrefetchSize"] = 256;

    if(isaVersion[0] == 10)
        rv["PhysicalMaxVgprCU"] = 1024 * 32;
    else if(isaVersion[0] == 11)
    {
        // Code path for gfx11XX (RDNA3, two SIMDs per CU, wave32).
        // gfx1100, gfx1101 and gfx1151 have a 1536-VGPR file per SIMD; every
        // other gfx11 part (e.g. gfx1102, gfx1103, gfx115X, gfx117X) has 1024.
        const bool has1536Vgpr = (isaVersion[1] == 0 && (isaVersion[2] == 0 || isaVersion[2] == 1))
                                 || (isaVersion[1] == 5 && isaVersion[2] == 1);
        const int vgprPerSimd = has1536Vgpr ? 1536 : 1024;
        rv["PhysicalMaxVgprCU"] = 2 /*two SIMDs per CU*/ * vgprPerSimd * 32;
    }
    else if(isaVersion[0] == 12)
        rv["PhysicalMaxVgprCU"] = isaVersion[1] == 5? 4096 * 32 : 2 * 1536 * 32;
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
