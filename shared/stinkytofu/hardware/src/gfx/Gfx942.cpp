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
* THE SOFTWARE IS PROVIDED "AS IS") WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
* THE SOFTWARE.
*
* ************************************************************************ */
#include <string>

#include "gfx/CommonInstsDSL.hpp"
#include "gfx/InstDefDSL.hpp"

namespace stinkytofu
{
    uint16_t computeCdna3MfmaLatency(const MFMA& a)
    {
        auto isFp8Family = [](const std::string& ty) -> bool {
            // The fp8 type is "BF8_BF8" .. etc, match the "xx8_" prefix
            if(ty.size() < 4)
            {
                assert(ty != "bf8" && ty != "fp8" && "Invalid fp8 family type!");
                return false;
            }

            std::string prefix = ty.substr(0, 4);
            return (prefix == "bf8_" || prefix == "fp8_");
        };

        auto speedup = [&](const std::string& ty) -> uint16_t {
            if(ty == "f32" || ty == "f64")
                return 1;
            if(ty == "xf32")
                return 4;
            if(ty == "f16" || ty == "bf16")
                return 8;
            if(ty == "i8" || isFp8Family(ty))
                return 16;
            if(ty == "f8f6f4")
                return 16;

            assert(false && "Unknown MFMA input type for latency computation!");
            return 1;
        };

        uint16_t speed    = speedup(a.inTy);
        uint16_t sparsity = a.sparse ? 2 : 1;
        uint16_t slowdown = 1;

        if(a.B > 1)
        {
            if(a.inTy == "f64" || a.inTy == "f16" || a.inTy == "bf16")
                slowdown = 2;
            if(a.inTy == "i8")
                slowdown = 4;
        }

        uint16_t latency = (a.M * a.N * a.K * a.B * slowdown) / (speed * sparsity) / 32;
        return latency;
    }

    // gfx942 instruction definition
    //
    // llvm tablegen style instruction definition for gfx942 architecture
    //
    // Note: This is just one of the possible ways to define instructions.
    //       Developer could also read an instruction name list from a file,
    //       parse it, and populate the registry accordingly.
    void defineGfx942Insts(GpuArch& registry)
    {
        // ============================================
        // Scalar instructions (SOP1 / SOP2 / Scalar ALU)
        // ============================================
        for(std::string op : {
                "s_max",
                "s_min",
                "s_add",
                "s_sub",
                "s_mul_hi",
            })
            for(auto ty : {"i32", "u32"})
                DEF_T(SALU, op + "_" + ty);

        for(std::string op : {"s_addc", "s_subb"})
            DEF_T(SALU, op + "_u32");

        for(std::string op : {"s_abs", "s_mul", "s_movk"})
            DEF_T(SALU, op + "_i32");

        for(std::string op : {
                "s_andn2",
                "s_cselect",
                "s_and",
                "s_or",
                "s_xor",
                "s_lshl",
                "s_lshr",
                "s_mov",
            })
            for(auto ty : {"b32", "b64"})
                DEF_T(SALU, op + "_" + ty);

        DEF_T(SALU, "s_bfm_b32");

        for(std::string ty : {"i32", "i64"})
            DEF_T(SALU, "s_ashr_" + ty);

        for(std::string op : {
                "s_lshl1",
                "s_lshl2",
                "s_lshl3",
                "s_lshl4",
            })
            DEF_T(SALU, op + "_add_u32");

        DEF_T(SALU, "s_sext_i32_i16");

        // ============================================
        // Control / Special
        // ============================================
        DEF_T(BranchInst, "s_branch");
        for(std::string op : {
                "execnz",
                "execz",
                "scc0",
                "scc1",
                "vccnz",
                "vccz",
            })
            DEF_T(BranchInst, "s_cbranch_" + op);

        DEF_T(BranchInst, "s_and_saveexec_b64");

        for(std::string ty : {"b32", "b64"})
            DEF_T(SALU, "s_cmov_" + ty);

        DEF_T(WaitCntInst, "s_waitcnt");
        DEF_T(BarrierInst, "s_barrier");

        // scalar instructions that have side effects
        for(auto op : {
                "s_nop",
                "s_dcache_wb",
                "s_delay_alu",
                "s_endpgm",
                "s_getreg_b32",
                "s_ff1_i32_b32",
                "s_getpc_b64",
                "or_saveexec_b32",
                "s_or_saveexec_b64",
                "s_set_mask",
                "s_setpc_b64",
                "s_setprio",
                "s_setreg_b32",
                "s_setreg_imm32_b32",
                "s_sleep",
                "s_swappc_b64",
            })
            DEF_T(HasSideEffectInst, op);

        // --------------------------------------------
        // SCmp: all scalar compare variants
        // SCmpK (compare with immediate)
        // --------------------------------------------
        for(std::string cmp : {"s_cmp", "s_cmpk"})
            for(auto ty : {"i32", "u32"})
                for(auto fn : {"eq", "ge", "gt", "le", "lt", "lg"})
                    DEF_T(SCmp, cmp + "_" + fn + "_" + ty);

        for(std::string fn : {"eq", "lg"})
            DEF_T(SCmp, "s_cmp_" + fn + "_u64");

        DEF_T(SCmp, "s_bitcmp1_b32");

        // ============================================
        // Vector ALU (VOP2 / VOP3 / VALU arithmetic)
        // ============================================
        for(std::string op : {"v_add", "v_mul", "v_max", "v_min", "v_fma"})
            for(auto ty : {"f16", "f32", "f64"})
                DEF_T(VALU, op + "_" + ty);

        for(std::string suffix : {"_i32", "_u32", "_co_u32", "c_co_u32", "3_u32"})
            DEF_T(VALU, "v_add" + suffix);

        for(std::string suffix : {"lo_u32", "hi_i32", "hi_u32", "i32_i24", "u32_u24"})
            DEF_T(VALU, "v_mul_" + suffix);

        for(std::string suffix : {"f16", "f32", "i32", "u32", "co_u32"})
            DEF_T(VALU, "v_sub_" + suffix);

        for(auto name : {
                "v_max_i32",
                "v_min_i32",
                "v_dot2c_f32_f16",
                "v_dot2_f32_f16",
                "v_pk_fma_f16",
                "v_pk_max_f16",
                "v_mad_i32_i24",
                "v_mad_u32_u24",
                "v_med3_i32",
                "v_med3_f32",
                "v_mac_f32",
                "v_mad_mix_f32",
            })
            DEF_T(VALU, name);

        // Other b32 VALU
        for(auto op : {"v_and_b32",
                       "v_and_or_b32",
                       "v_not_b32",
                       "v_or_b32",
                       "v_xor_b32",
                       "v_cndmask_b32",
                       "v_lshlrev_b32",
                       "v_lshrrev_b32",
                       "v_lshl_or_b32",
                       "v_mov_b32",
                       "v_swap_b32",
                       "v_bfi_b32",
                       "v_readfirstlane_b32",
                       "v_perm_b32"})
            DEF_T(VALU, op);

        DEF_T(VALU, "v_pack_b32_f16");

        for(auto op : {"v_lshlrev_b16",
                       "v_lshlrev_b64",
                       "v_lshrrev_b64",
                       "v_ashrrev_i32",
                       "v_add_lshl_u32",
                       "v_lshl_add_u32",
                       "v_mov_b64",
                       "v_bfe_i32",
                       "v_bfe_u32",
                       "v_rndne_f32"})
            DEF_T(VALU, op);

        // VTrans: vector transcendental instructions
        for(auto name :
            {"v_exp_f32", "v_log_f32", "v_rcp_f32",  "v_rcp_iflag_f32",  "v_rsq_f32",
             "v_rcp_f64", "v_rsq_f64", "v_sqrt_f32", "v_sqrt_f64",       "v_sin_f32",
             "v_cos_f32", "v_rcp_f16", "v_sqrt_f16", "v_rsq_f16",        "v_log_f16",
             "v_exp_f16", "v_sin_f16", "v_cos_f16",  "v_exp_legacy_f32", "v_log_legacy_f32"})
        {
            DEF_T(VTrans, name);
        }

        // ============================================
        // VCmp: vector compare patterns
        // VCmpX: vector compare and update EXEC mask
        //
        // amd-instinct-mi300-cdna3-instruction-set-architecture.pdf
        // Table 25. VALU Instruction Set
        // ============================================
        for(std::string cmp : {"v_cmp", "v_cmpx"})
            for(auto ty : {"i16", "i32", "i64", "u16", "u32", "u64"})
                for(auto fn : {"f", "lt", "eq", "le", "gt", "lg", "ge", "ne", "t"})
                    DEF_T(VCmp, cmp + "_" + fn + "_" + ty);

        for(std::string cmp : {"v_cmp", "v_cmpx"})
            for(auto ty : {"f16", "f32", "f64"})
                for(auto fn : {"f",
                               "lt",
                               "eq",
                               "le",
                               "gt",
                               "lg",
                               "ge",
                               "t",
                               "o",
                               "u",
                               "nge",
                               "nlg",
                               "ngt",
                               "nle",
                               "neq",
                               "nlt"})
                    DEF_T(VCmp, cmp + "_" + fn + "_" + ty);

        for(std::string cmp : {"v_cmp", "v_cmpx"})
            for(auto ty : {"f16", "f32", "f64"})
                DEF_T(VCmp, cmp + "_class_" + ty);

        // ============================================
        // VCvt: vector convert family
        // ============================================

        // convert to f32
        for(std::string src : {"bf8", "fp8", "f16", "i32", "u32"})
            DEF_T(VCvt, "v_cvt_f32_" + src);

        // convert from f32
        for(std::string dst : {"f16", "i32", "u32"})
            DEF_T(VCvt, "v_cvt_" + dst + "_f32");

        // pk pairs
        for(auto pair : {std::pair{"bf8", "f32"}, {"fp8", "f32"}})
        {
            DEF_T(VCvt, std::string("v_cvt_pk_") + pair.first + "_" + pair.second);
            DEF_T(VCvt, std::string("v_cvt_pk_") + pair.second + "_" + pair.first);
        }

        for(std::string dst : {"bf8", "fp8"})
            DEF_T(VCvt, "v_cvt_sr_" + dst + "_f32");

        // ============================================
        // DS: LDS access
        // ============================================
        for(std::string ty : {"b32", "b64", "b96", "b128", "u8", "i8", "u16", "i16"})
            DEF_T(DSRead, "ds_read_" + ty);

        for(std::string ty : {"b32", "b64", "b96", "b128", "b8", "b16"})
            DEF_T(DSWrite, "ds_write_" + ty);

        DEF_T(DSRead, "ds_read_u8_d16_hi");
        DEF_T(DSRead, "ds_read_u16_d16_hi");
        DEF_T(DSWrite, "ds_write_b8_d16_hi");
        DEF_T(DSWrite, "ds_write_b16_d16_hi");

        // Note: ds_bpermute_b32 Does not actually write any LDS memory.
        DEF_T(DSWrite, "ds_bpermute_b32");

        for(std::string ty : {"b32", "b64"})
        {
            DEF_T(DSRead, "ds_read2_" + ty);
            DEF_T(DSWrite, "ds_write2_" + ty);
        }

        // ============================================
        // Buffer / Global / Flat memory access
        // ============================================

        for(std::string ty : {"dword",
                              "dwordx2",
                              "dwordx3",
                              "dwordx4",
                              "ubyte",
                              "sbyte",
                              "ushort",
                              "sshort",
                              "ubyte_d16",
                              "ubyte_d16_hi",
                              "sbyte_d16",
                              "sbyte_d16_hi",
                              "short_d16",
                              "short_d16_hi"})
        {
            DEF_T(MUBUFLoad, "buffer_load_" + ty);
        }

        for(std::string ty : {"dword",
                              "dwordx2",
                              "dwordx3",
                              "dwordx4",
                              "byte",
                              "byte_d16_hi",
                              "short",
                              "short_d16_hi"})
        {
            DEF_T(MUBUFStore, "buffer_store_" + ty);
        }

        DEF_T(MUBUFAtomic, "buffer_atomic_add_f32");
        DEF_T(MUBUFAtomic, "buffer_atomic_cmpswap");
        DEF_T(MUBUFAtomic, "buffer_atomic_cmpswap_x2");
        DEF_T(MUBUFAtomic, "s_atomic_dec");

        for(std::string ty : {"dword", "dwordx2", "dwordx4", "dwordx8", "dwordx16"})
            DEF_T(MUBUFLoad, "s_load_" + ty);

        for(std::string ty : {"dword", "dwordx2", "dwordx4"})
            DEF_T(MUBUFLoad, "s_store_" + ty);

        for(std::string ty : {"ubyte",
                              "ubyte_d16",
                              "ubyte_d16_hi",
                              "sbyte",
                              "sbyte_d16",
                              "sbyte_d16_hi",
                              "ushort",
                              "sshort",
                              "short_d16",
                              "short_d16_hi"})
            DEF_T(FLATLoad, "flat_load_" + ty);

        for(std::string ty : {"dword", "dwordx2", "dwordx3", "dwordx4"})
        {
            DEF_T(FLATLoad, "flat_load_" + ty);
            DEF_T(FLATStore, "flat_store_" + ty);
        }

        for(std::string ty : {"byte", "byte_d16_hi", "short", "short_d16_hi"})
            DEF_T(FLATStore, "flat_store_" + ty);

        for(std::string ty : {"swap",
                              "cmpswap",
                              "add",
                              "sub",
                              "smin",
                              "umin",
                              "smax",
                              "umax",
                              "and",
                              "or",
                              "xor",
                              "inc",
                              "dec",
                              "add_f32",
                              "pk_add_f16",
                              "pk_add_bf16",
                              "add_f64",
                              "min_f64",
                              "max_f64"})
            DEF_T(FLATAtomic, "flat_atomic_" + ty);

        DEF_T(GLOBALLoad, "global_load_dword");
        DEF_T(GLOBALStore, "global_store_dword");

        // ============================================
        // MFMA / SMFMAC (matrix instructions)
        // ============================================
        const MatInstDesc mfma942[] = {
            // fp32, fp32
            {32, 32, 1, 2, "f32", "f32"},
            {16, 16, 1, 4, "f32", "f32"},
            {4, 4, 1, 16, "f32", "f32"},
            {32, 32, 2, 1, "f32", "f32"},
            {16, 16, 4, 1, "f32", "f32"},

            // fp32, fp16
            {32, 32, 4, 2, "f32", "f16"},
            {16, 16, 4, 4, "f32", "f16"},
            {4, 4, 4, 16, "f32", "f16"},
            {32, 32, 8, 1, "f32", "f16"},
            {16, 16, 16, 1, "f32", "f16"},

            // fp32, bf16
            {32, 32, 4, 2, "f32", "bf16"},
            {16, 16, 4, 4, "f32", "bf16"},
            {4, 4, 4, 16, "f32", "bf16"},
            {32, 32, 8, 1, "f32", "bf16"},
            {16, 16, 16, 1, "f32", "bf16"},

            // i32, i8
            {32, 32, 4, 2, "i32", "i8"},
            {16, 16, 4, 4, "i32", "i8"},
            {4, 4, 4, 16, "i32", "i8"},
            {32, 32, 16, 1, "i32", "i8"},
            {16, 16, 32, 1, "i32", "i8"},

            // f32, xf32
            {16, 16, 8, 1, "f32", "xf32"},
            {32, 32, 4, 1, "f32", "xf32"},

            // f64, f64
            {16, 16, 4, 1, "f64", "f64"},
            {4, 4, 4, 4, "f64", "f64"},

            // fp32, fp8 family
            {16, 16, 32, 1, "f32", "bf8_bf8"},
            {16, 16, 32, 1, "f32", "fp8_fp8"},
            {16, 16, 32, 1, "f32", "bf8_fp8"},
            {16, 16, 32, 1, "f32", "fp8_bf8"},

            {32, 32, 16, 1, "f32", "bf8_bf8"},
            {32, 32, 16, 1, "f32", "fp8_fp8"},
            {32, 32, 16, 1, "f32", "bf8_fp8"},
            {32, 32, 16, 1, "f32", "fp8_bf8"},
        };

        for(auto s : mfma942)
            GEN_MFMA(registry, s, false);

        const MatInstDesc smfmac942[] = {
            // f32, f16,  {16x16x32, 32x32x16}
            {16, 16, 32, 1, "f32", "f16"},
            {32, 32, 16, 1, "f32", "f16"},

            // f32, bf16, {16x16x32, 32x32x16}
            {16, 16, 32, 1, "f32", "bf16"},
            {32, 32, 16, 1, "f32", "bf16"},

            // i32, i8,   {16x16x64, 32x32x32}
            {16, 16, 64, 1, "i32", "i8"},
            {32, 32, 32, 1, "i32", "i8"},

            // f32, {bf8_bf8, fp8_fp8, bf8_fp8, fp8_bf8},  {16x16x64, 32x32x32}
            {16, 16, 64, 1, "f32", "bf8_bf8"},
            {16, 16, 64, 1, "f32", "fp8_fp8"},
            {16, 16, 64, 1, "f32", "bf8_fp8"},
            {16, 16, 64, 1, "f32", "fp8_bf8"},
            {32, 32, 32, 1, "f32", "bf8_bf8"},
            {32, 32, 32, 1, "f32", "fp8_fp8"},
            {32, 32, 32, 1, "f32", "bf8_fp8"},
            {32, 32, 32, 1, "f32", "fp8_bf8"},
        };

        for(auto s : smfmac942)
            GEN_MFMA(registry, s, true);

        DEF_T(VALU, "v_accvgpr_read_b32");
        DEF_T(VALU, "v_accvgpr_write_b32");
    }
} // namespace stinkytofu
