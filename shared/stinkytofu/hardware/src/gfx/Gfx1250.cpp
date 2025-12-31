/* ************************************************************************
* Copyright (C) 2025-2026 Advanced Micro Devices, Inc.
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
#include <utility>

#include "gfx/CommonInstsDSL.hpp"
#include "gfx/GpuArchManager.hpp"
#include "gfx/InstDefDSL.hpp"

namespace stinkytofu
{
    void defineGfx1250Insts(GpuArch& registry)
    {
        /*
         * Begin Import gfx942 instructions
         */

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
            DEF_T(ConditionalBranchInst, "s_cbranch_" + op);

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
        // Commutative operations
        for(std::string op : {"v_add", "v_mul", "v_max", "v_min"})
            for(auto ty : {"f16", "f32", "f64"})
                DEF_T(CommutativeVALU, op + "_" + ty);

        // FMA (first two operands commutative: a*b+c = b*a+c)
        for(auto ty : {"f16", "f32", "f64"})
            DEF_T(CommutativeVALU, "v_fma_" + std::string(ty));

        for(std::string suffix : {"_i32", "_u32", "_co_u32", "c_co_u32", "3_u32"})
            DEF_T(CommutativeVALU, "v_add" + suffix);

        for(std::string suffix : {"lo_u32", "hi_i32", "hi_u32", "i32_i24", "u32_u24"})
            DEF_T(CommutativeVALU, "v_mul_" + suffix);

        for(std::string suffix : {"f16", "f32", "i32", "u32", "co_u32"})
            DEF_T(VALU, "v_sub_" + suffix);

        // Commutative VALU
        for(auto name : {
                "v_max_i32",
                "v_min_i32",
                "v_dot2c_f32_f16",
                "v_dot2_f32_f16",
                "v_pk_max_f16",
            })
            DEF_T(CommutativeVALU, name);

        // Non-commutative VALU
        for(auto name : {
                "v_pk_fma_f16",
                "v_mad_i32_i24",
                "v_mad_u32_u24",
                "v_med3_i32",
                "v_med3_f32",
                "v_mac_f32",
                "v_mad_mix_f32",
            })
            DEF_T(VALU, name);

        // Commutative bitwise b32 VALU
        for(auto op : {"v_and_b32", "v_or_b32", "v_xor_b32"})
            DEF_T(CommutativeVALU, op);

        // Non-commutative b32 VALU
        for(auto op : {"v_and_or_b32",
                       "v_not_b32",
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

        /*
         * End Import gfx942 instructions
         */

        /*
         * Begin Import gfx950 instructions
         */

        // gfx950 removes xf32 variants of v_mfma.
        for(auto removed : {"v_mfma_f32_16x16x8_xf32", "v_mfma_f32_32x32x4_xf32"})
            registry.erase(removed);

        // 1) New plain/pack converts
        //    v_cvt_f32_bf16
        DEF_T(VCvt, "v_cvt_f32_bf16");

        //    v_cvt_pk_bf16_f32
        DEF_T(VCvt, "v_cvt_pk_bf16_f32");

        // 2) "scale" fp8/bf8 variants
        //    v_cvt_scalef32_f16_fp8
        DEF_T(VCvtScale, "v_cvt_scalef32_f16_fp8");

        //    pk_* pairs: (dst, src) ordered as they appear in mnemonics
        for(auto p : {
                std::pair{"bf8", "f16"},
                std::pair{"f16", "bf8"},
                std::pair{"f16", "fp8"},
                std::pair{"fp8", "f16"},
            })
        {
            std::string name = "v_cvt_scalef32_pk_";
            name += p.first;
            name += "_";
            name += p.second;
            DEF_T(VCvtScale, name.c_str());
        }

        //    sr_* pairs
        for(auto p : {
                std::pair{"bf8", "f16"},
                std::pair{"fp8", "f16"},
            })
        {
            std::string name = "v_cvt_scalef32_sr_";
            name += p.first;
            name += "_";
            name += p.second;
            DEF_T(VCvtScale, name.c_str());
        }

        // 3) New bf16 dot ops
        //    v_dot2_f32_bf16, v_dot2c_f32_bf16
        for(auto base : {"v_dot2_f32_", "v_dot2c_f32_"})
        {
            std::string name = std::string(base) + "bf16";
            DEF_T(VDot, name.c_str());
        }

        // 4) PRNG
        DEF_T(VALU, "v_prng_b32");

        // ds read b64 tr b16
        DEF_T(DSRead, "ds_read_b64_tr_b4");
        DEF_T(DSRead, "ds_read_b96_tr_b6");
        DEF_T(DSRead, "ds_read_b64_tr_b8");
        DEF_T(DSRead, "ds_read_b64_tr_b16");

        // ============================================
        // MFMA / SMFMAC (matrix instructions)
        // ============================================
        const MatInstDesc mfma950[] = {
            // V_MFMA_F32_16x16x128_F8F6F4
            {16, 16, 128, 1, "f32", "f8f6f4"},
            // V_MFMA_F32_32x32x64_F8F6F4
            {32, 32, 64, 1, "f32", "f8f6f4"},
        };

        for(auto s : mfma950)
            GEN_MFMA(registry, s, false);

        /*
         * End Import gfx950 instructions
         */

        // Set wavefront size for gfx1250
        registry.setWaveFrontSize(32);

        // gfx1250 removes ds_read/ds_write variant instructions.
        std::vector<std::string> removedInsts;
        for(const auto& inst : registry.getInstructions())
        {
            if(inst->is(IF_DSRead) || inst->is(IF_DSStore))
            {
                removedInsts.push_back(inst->name);
            }
        }

        for(const auto& name : removedInsts)
        {
            registry.erase(name);
        }

        // ============================================
        // Scalar ALU
        // ============================================
        DEF_T(SALU, "s_mul_lo_u32");
        DEF_T(SALU, "s_sub_u64");
        DEF_T(SALU, "s_and_saveexec_b32");
        DEF_T(SALU, "s_or_saveexec_b32");

        // ============================================
        // Vector ALU
        // ============================================
        DEF_T(VALU, "v_fma_mix_f32");
        DEF_T(VALU, "v_rsq_iflag_f32");

        // Vector control/sync instructions
        DEF_T(HasSideEffectInst, "v_nop");

        // ============================================
        // Scalar control/sync instructions
        // ============================================
        DEF_T(HasSideEffectInst, "s_wait_alu");
        DEF_T(HasSideEffectInst, "s_set_vgpr_msb");

        // ============================================
        // TDM
        // ============================================
        DEF_T(TensorLoadToLds, "tensor_load_to_lds");

        DEF_T(WaitTensorCntInst, "s_wait_tensorcnt");

        // ============================================
        // DS: LDS access
        // ============================================
        for(std::string ty : {"b32", "b64", "b96", "b128", "u8", "i8", "u16", "i16"})
            DEF_T(DSRead, "ds_load_" + ty);

        for(std::string ty : {"b8", "b16", "b32", "b64", "b96", "b128"})
            DEF_T(DSWrite, "ds_store_" + ty);

        // ============================================
        // WMMA / SWMMA (matrix instructions)
        // ============================================
        const MatInstDesc wmma1250[] = {
            // V_WMMA_F32_16X16X32_BF16
            {16, 16, 32, 1, "f32", "bf16"},
        };

        for(auto s : wmma1250)
            GEN_WMMA(registry, s, false);
    }

    void setGfx1250RocisaSimpleMap(GpuArch& registry)
    {
        // const std::unordered_map<std::type_index, Opcode>
        std::unordered_map<std::string, std::string> rocisaToHwInstMap = {
            /* branch.hpp */
            {"SBranch", "s_branch"},
            {"SCBranchSCC0", "s_cbranch_scc0"},
            {"SCBranchSCC1", "s_cbranch_scc1"},
            {"SCBranchVCCNZ", "s_cbranch_vccnz"},
            {"SCBranchVCCZ", "s_cbranch_vccz"},
            {"SSetPCB64", "s_setpc_b64"},
            {"SSwapPCB64", "s_swappc_b64"},
            {"SCBranchExecZ", "s_cbranch_execz"},
            {"SCBranchExecNZ", "s_cbranch_execnz"},

            /* cmp.hpp */
            {"SCmpEQI32", "s_cmp_eq_i32"},
            {"SCmpEQU32", "s_cmp_eq_u32"},
            {"SCmpEQU64", "s_cmp_eq_u64"},
            {"SCmpGeI32", "s_cmp_ge_i32"},
            {"SCmpGeU32", "s_cmp_ge_u32"},
            {"SCmpGtI32", "s_cmp_gt_i32"},
            {"SCmpGtU32", "s_cmp_gt_u32"},
            {"SCmpLeI32", "s_cmp_le_i32"},
            {"SCmpLeU32", "s_cmp_le_u32"},
            {"SCmpLgU32", "s_cmp_lg_u32"},
            {"SCmpLgI32", "s_cmp_lg_i32"},
            {"SCmpLgU64", "s_cmp_lg_u64"},
            {"SCmpLtI32", "s_cmp_lt_i32"},
            {"SCmpLtU32", "s_cmp_lt_u32"},
            {"SBitcmp1B32", "s_bitcmp1_b32"},
            {"SCmpKEQU32", "s_cmpk_eq_u32"},
            {"SCmpKGeU32", "s_cmpk_ge_u32"},
            {"SCmpKGtU32", "s_cmpk_gt_u32"},
            {"SCmpKLGU32", "s_cmpk_lg_u32"},
            {"VCmpEQF32", "v_cmp_eq_f32"},
            {"VCmpEQF64", "v_cmp_eq_f64"},
            {"VCmpEQU32", "v_cmp_eq_u32"},
            {"VCmpEQI32", "v_cmp_eq_i32"},
            {"VCmpGEF16", "v_cmp_ge_f16"},
            {"VCmpGTF16", "v_cmp_gt_f16"},
            {"VCmpGEF32", "v_cmp_ge_f32"},
            {"VCmpGTF32", "v_cmp_gt_f32"},
            {"VCmpGEF64", "v_cmp_ge_f64"},
            {"VCmpGTF64", "v_cmp_gt_f64"},
            {"VCmpGEI32", "v_cmp_ge_i32"},
            {"VCmpGTI32", "v_cmp_gt_i32"},
            {"VCmpGEU32", "v_cmp_ge_u32"},
            {"VCmpGtU32", "v_cmp_gt_u32"},
            {"VCmpLeU32", "v_cmp_le_u32"},
            {"VCmpLeI32", "v_cmp_le_i32"},
            {"VCmpLtI32", "v_cmp_lt_i32"},
            {"VCmpLtU32", "v_cmp_lt_u32"},
            {"VCmpUF32", "v_cmp_u_f32"},
            {"VCmpNeI32", "v_cmp_ne_i32"},
            {"VCmpNeU32", "v_cmp_ne_u32"},
            {"VCmpNeU64", "v_cmp_ne_u64"},
            {"VCmpClassF32", "v_cmp_class_f32"},
            {"VCmpXClassF32", "v_cmpx_class_f32"},
            {"VCmpXEqU32", "v_cmpx_eq_u32"},
            {"VCmpXGeU32", "v_cmpx_ge_u32"},
            {"VCmpXGtU32", "v_cmpx_gt_u32"},
            {"VCmpXLeU32", "v_cmpx_le_u32"},
            {"VCmpXLeI32", "v_cmpx_le_i32"},
            {"VCmpXLtF32", "v_cmpx_lt_f32"},
            {"VCmpXLtI32", "v_cmpx_lt_i32"},
            {"VCmpXLtU32", "v_cmpx_lt_u32"},
            {"VCmpXLtU64", "v_cmpx_lt_u64"},
            {"VCmpXNeU16", "v_cmpx_ne_u16"},
            {"VCmpXNeU32", "v_cmpx_ne_u32"},

            /* cvt.hpp */
            {"VCvtF16toF32", "v_cvt_f32_f16"},
            {"VCvtF32toF16", "v_cvt_f16_f32"},
            {"VCvtF32toU32", "v_cvt_u32_f32"},
            {"VCvtU32toF32", "v_cvt_f32_u32"},
            {"VCvtI32toF32", "v_cvt_f32_i32"},
            {"VCvtF32toI32", "v_cvt_i32_f32"},
            {"VCvtFP8toF32", "v_cvt_f32_fp8"},
            {"VCvtBF8toF32", "v_cvt_f32_bf8"},
            {"VCvtPkFP8toF32", "v_cvt_pk_f32_fp8"},
            {"VCvtPkBF8toF32", "v_cvt_pk_f32_bf8"},
            {"VCvtPkF32toFP8", "v_cvt_pk_fp8_f32"},
            {"VCvtPkF32toBF8", "v_cvt_pk_bf8_f32"},
            {"VCvtSRF32toFP8", "v_cvt_sr_fp8_f32"},
            {"VCvtSRF32toBF8", "v_cvt_sr_bf8_f32"},

            /* mem.hpp */
            {"BufferLoadU8", "buffer_load_ubyte"},
            {"BufferLoadD16HIU8", "buffer_load_ubyte_d16_hi"},
            {"BufferLoadD16U8", "buffer_load_ubyte_d16"},
            {"BufferLoadD16HIB16", "buffer_load_short_d16_hi"},
            {"BufferLoadD16B16", "buffer_load_short_d16"},
            {"BufferLoadB32", "buffer_load_dword"},
            {"BufferLoadB64", "buffer_load_dwordx2"},
            {"BufferLoadB128", "buffer_load_dwordx4"},
            {"FlatLoadD16HIU8", "flat_load_ubyte_d16_hi"},
            {"FlatLoadD16U8", "flat_load_ubyte_d16"},
            {"FlatLoadD16HIB16", "flat_load_short_d16_hi"},
            {"FlatLoadD16B16", "flat_load_short_d16"},
            {"FlatLoadB32", "flat_load_dword"},
            {"FlatLoadB64", "flat_load_dwordx2"},
            {"FlatLoadB128", "flat_load_dwordx4"},
            {"TensorLoadToLds", "tensor_load_to_lds"},
            {"BufferStoreB8", "buffer_store_byte"},
            {"BufferStoreD16HIU8", "buffer_store_byte_d16_hi"},
            {"BufferStoreD16HIB16", "buffer_store_short_d16_hi"},
            {"BufferStoreB16", "buffer_store_short"},
            {"BufferStoreB32", "buffer_store_dword"},
            {"BufferStoreB64", "buffer_store_dwordx2"},
            {"BufferStoreB128", "buffer_store_dwordx4"},
            {"BufferAtomicAddF32", "buffer_atomic_add_f32"},
            {"BufferAtomicCmpswapB32", "buffer_atomic_cmpswap"},
            {"BufferAtomicCmpswapB64", "buffer_atomic_cmpswap_x2"},
            {"FlatStoreD16HIB16", "flat_store_short_d16_hi"},
            {"FlatStoreB32", "flat_store_dword"},
            {"FlatStoreB64", "flat_store_dwordx2"},
            {"FlatStoreB128", "flat_store_dwordx4"},
            {"FlatAtomicCmpswapB32", "flat_atomic_cmpswap"},
            {"DSLoadU8", "ds_load_u8"},
            {"DSLoadU16", "ds_load_u16"},
            {"DSLoadB32", "ds_load_b32"},
            {"DSLoadB64", "ds_load_b64"},
            {"DSLoadB128", "ds_load_b128"},
            {"DSStoreB8", "ds_store_b8"},
            {"DSStoreB16", "ds_store_b16"},
            {"DSStoreB32", "ds_store_b32"},
            {"DSStoreB64", "ds_store_b64"},
            {"DSStoreB128", "ds_store_b128"},
            {"SAtomicDec", "s_atomic_dec"},
            {"SLoadB32", "s_load_dword"},
            {"SLoadB64", "s_load_dwordx2"},
            {"SLoadB128", "s_load_dwordx4"},
            {"SLoadB256", "s_load_dwordx8"},
            {"SLoadB512", "s_load_dwordx16"},
            {"SStoreB32", "s_store_dword"},
            {"SStoreB64", "s_store_dwordx2"},
            {"SStoreB128", "s_store_dwordx4"},

            /* common.hpp */
            {"SAbsI32", "s_abs_i32"},
            {"SMaxI32", "s_max_i32"},
            {"SMaxU32", "s_max_u32"},
            {"SMinI32", "s_min_i32"},
            {"SMinU32", "s_min_u32"},
            {"SAddI32", "s_add_i32"},
            {"SAddU32", "s_add_u32"},
            {"SAddCU32", "s_addc_u32"},
            {"SMulI32", "s_mul_i32"},
            {"SMulLOU32", "s_mul_lo_u32"},
            {"SMulHII32", "s_mul_hi_i32"},
            {"SMulHIU32", "s_mul_hi_u32"},
            {"SSubI32", "s_sub_i32"},
            {"SSubU32", "s_sub_u32"},
            {"SSubU64", "s_sub_u64"},
            {"SSubBU32", "s_subb_u32"},
            {"SCSelectB32", "s_cselect_b32"},
            {"SAndB32", "s_and_b32"},
            {"SAndB64", "s_and_b64"},
            {"SAndN2B32", "s_andn2_b32"},
            {"SOrB32", "s_or_b32"},
            {"SXorB32", "s_xor_b32"},
            {"SOrB64", "s_or_b64"},
            {"SGetPCB64", "s_getpc_b64"},
            {"SLShiftLeftB32", "s_lshl_b32"},
            {"SLShiftRightB32", "s_lshr_b32"},
            {"SLShiftLeftB64", "s_lshl_b64"},
            {"SLShiftRightB64", "s_lshr_b64"},
            {"SAShiftRightI32", "s_ashr_i32"},
            {"SLShiftLeft1AddU32", "s_lshl1_add_u32"},
            {"SLShiftLeft2AddU32", "s_lshl2_add_u32"},
            {"SLShiftLeft3AddU32", "s_lshl3_add_u32"},
            {"SLShiftLeft4AddU32", "s_lshl4_add_u32"},
            {"SSetMask", "s_set_mask"},
            {"SMovB32", "s_mov_b32"},
            {"SMovB64", "s_mov_b64"},
            {"SCMovB32", "s_cmov_b32"},
            {"SCMovB64", "s_cmov_b64"},
            {"SFf1B32", "s_ff1_i32_b32"},
            {"SBfmB32", "s_bfm_b32"},
            {"SMovkI32", "s_movk_i32"},
            {"SSExtI16toI32", "s_sext_i32_i16"},
            {"SAndSaveExecB32", "s_and_saveexec_b32"},
            {"SAndSaveExecB64", "s_and_saveexec_b64"},
            {"SOrSaveExecB32", "s_or_saveexec_b32"},
            {"SOrSaveExecB64", "s_or_saveexec_b64"},
            {"SSetPrior", "s_setprio"},
            {"SBarrier", "s_barrier"},
            {"SDcacheWb", "s_dcache_wb"},
            {"SNop", "s_nop"},
            {"SEndpgm", "s_endpgm"},
            {"SSleep", "s_sleep"},
            {"SGetRegB32", "s_getreg_b32"},
            {"SSetRegB32", "s_setreg_b32"},
            {"SSetRegIMM32B32", "s_setreg_imm32_b32"},
            {"SWaitCnt", "s_waitcnt"},
            {"SWaitTensorcnt", "s_wait_tensorcnt"},
            {"SDelayAlu", "s_delay_alu"},
            {"VAddF16", "v_add_f16"},
            {"VAddF32", "v_add_f32"},
            {"VAddF64", "v_add_f64"},
            {"VAddI32", "v_add_i32"},
            {"VAddU32", "v_add_u32"},
            {"VAddCOU32", "v_add_co_u32"},

            {"VMulF16", "v_mul_f16"},
            {"VMulF32", "v_mul_f32"},
            {"VMulF64", "v_mul_f64"},
            {"VMulLOU32", "v_mul_lo_u32"},
            {"VMulHII32", "v_mul_hi_i32"},
            {"VMulHIU32", "v_mul_hi_u32"},
            {"VMulI32I24", "v_mul_i32_i24"},
            {"VMulU32U24", "v_mul_u32_u24"},
            {"VSubF32", "v_sub_f32"},
            {"VSubI32", "v_sub_i32"},
            {"VSubU32", "v_sub_u32"},
            {"VSubCoU32", "v_sub_co_u32"},
            {"VMacF32", "v_mac_f32"},
            {"VDot2CF32F16", "v_dot2c_f32_f16"},
            {"VDot2F32F16", "v_dot2_f32_f16"},
            {"VFmaF16", "v_fma_f16"},
            {"VFmaF32", "v_fma_f32"},
            {"VFmaF64", "v_fma_f64"},
            {"VMadI32I24", "v_mad_i32_i24"},
            {"VFmaMixF32", "v_fma_mix_f32"},
            {"VMadU32U24", "v_mad_u32_u24"},
            {"VMadMixF32", "v_mad_mix_f32"},
            {"VExpF16", "v_exp_f16"},
            {"VExpF32", "v_exp_f32"},
            {"VRcpF16", "v_rcp_f16"},
            {"VRcpF32", "v_rcp_f32"},
            {"VRcpIFlagF32", "v_rcp_iflag_f32"},
            {"VRsqF16", "v_rsq_f16"},
            {"VRsqF32", "v_rsq_f32"},
            {"VRsqIFlagF32", "v_rsq_iflag_f32"},
            {"VMaxF16", "v_max_f16"},
            {"VMaxF32", "v_max_f32"},
            {"VMaxF64", "v_max_f64"},
            {"VMaxI32", "v_max_i32"},
            {"VMed3I32", "v_med3_i32"},
            {"VMed3F32", "v_med3_f32"},
            {"VMinF16", "v_min_f16"},
            {"VMinF32", "v_min_f32"},
            {"VMinF64", "v_min_f64"},
            {"VMinI32", "v_min_i32"},
            {"VAndB32", "v_and_b32"},
            {"VAndOrB32", "v_and_or_b32"},
            {"VNotB32", "v_not_b32"},
            {"VOrB32", "v_or_b32"},
            {"VXorB32", "v_xor_b32"},
            {"VCndMaskB32", "v_cndmask_b32"},
            {"VLShiftLeftB16", "v_lshlrev_b16"},
            {"VLShiftLeftB32", "v_lshlrev_b32"},
            {"VLShiftRightB32", "v_lshrrev_b32"},
            {"VLShiftLeftB64", "v_lshlrev_b64"},
            {"VLShiftRightB64", "v_lshrrev_b64"},
            {"VAShiftRightI32", "v_ashrrev_i32"},
            {"VLShiftLeftOrB32", "v_lshl_or_b32"},
            {"VAddLShiftLeftU32", "v_add_lshl_u32"},
            {"VLShiftLeftAddU32", "v_lshl_add_u32"},
            {"VMovB32", "v_mov_b32"},
            {"VMovB64", "v_mov_b64"},
            {"VSwapB32", "v_swap_b32"},
            {"VBfeI32", "v_bfe_i32"},
            {"VBfeU32", "v_bfe_u32"},
            {"VBfiB32", "v_bfi_b32"},
            {"VPackF16toB32", "v_pack_b32_f16"},
            {"VAccvgprReadB32", "v_accvgpr_read_b32"},
            {"VAccvgprWriteB32", "v_accvgpr_write_b32"},
            {"VReadfirstlaneB32", "v_readfirstlane_b32"},
            {"VRndneF32", "v_rndne_f32"},
            {"VPermB32", "v_perm_b32"},
        };

        registry.setRocisaSimpleMap(std::move(rocisaToHwInstMap));
    }

    void setGfx1250ConversionMap(GpuArch& registry)
    {
        std::unordered_map<std::string, std::string> conversion = {
            {"_SWaitCnt", "lowerRocisaWaitCnt" /* todo */},
            {"_SWaitCntVscnt", "" /* todo */},
            {"_SWaitStorecnt", "" /* todo */},
            {"_SWaitLoadcnt", "lowerRocisaWaitCnt"},
            {"_SWaitKMcnt", "" /* todo */},
            {"_SWaitDscnt", "lowerRocisaWaitCnt"},
            {"SWaitTensorcnt", "lowerRocisaWaitTensorcnt"},

            // {"VMulPKF32", "v_pk_mul_f32" /* todo */},
            // {"VAddPKF32", "v_pk_add_f32" /* todo */},

            // {"VPrngB32", "v_prng_b32"},
            // {"VMaxPKF16", "v_max_pk_f16"},

            /* mfma.hpp */
            {"MFMAInstruction", "lowerRocisaMFMA"},
            {"SMFMAInstruction", "lowerRocisaSMFMA"},
        };

        registry.setRocisaConversionMap(std::move(conversion));
    }

} // namespace stinkytofu
