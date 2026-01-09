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
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */

/**
 * @file GenHighLevelIR.cpp
 * @brief Generates high-level IR instruction class definitions and builder methods
 *
 * This generator creates:
 * 1. IR instruction class definitions (StinkyInstructions_generated.inc)
 * 2. Builder method forward declarations (StinkyBuilder_decls_generated.inc)
 * 3. Builder method implementations (StinkyBuilder_impls_generated.inc)
 * 4. Mnemonic-to-IR mappings for ToStinkyAsmPass (IRMnemonics_generated.inc)
 */

#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace stinkytofu
{

    // IR instruction definition
    struct IRInstDef
    {
        std::string className; // e.g., "VAddF32"
        std::string mnemonic; // e.g., "v_add_f32"
        std::string comment; // e.g., "Vector add F32: dst = src0 + src1"
        int         numSrcs; // Number of source operands
        bool        hasDest; // Whether it has a destination operand
        std::string category; // e.g., "Vector Arithmetic", "Scalar Bitwise"
        bool        supportsDPP; // Whether this instruction supports DPP modifiers
        bool        supportsSDWA; // Whether this instruction supports SDWA modifiers
        bool        hasDS; // Whether this instruction has DS modifiers (for LDS operations)

        IRInstDef(const std::string& cls,
                  const std::string& mn,
                  const std::string& cmt,
                  int                srcs,
                  bool               dest = true,
                  const std::string& cat  = "",
                  bool               dpp  = false,
                  bool               sdwa = false,
                  bool               ds   = false)
            : className(cls)
            , mnemonic(mn)
            , comment(cmt)
            , numSrcs(srcs)
            , hasDest(dest)
            , category(cat)
            , supportsDPP(dpp)
            , supportsSDWA(sdwa)
            , hasDS(ds)
        {
        }
    };

    // Define all high-level IR instructions
    // This is the "single source of truth" replacing manual class definitions
    static std::vector<IRInstDef> getIRInstructions()
    {
        return {
            {"VAddU32", "v_add_u32", "VAddU32", 2, true, "Vector Arithmetic", true, true, false},

            {"VMulF32", "v_mul_f32", "VMulF32", 2, true, "Vector Arithmetic", true, true, false},

            {"VAddF16", "v_add_f16", "VAddF16", 2, true, "Vector Arithmetic", true, true, false},

            {"VAddF32", "v_add_f32", "VAddF32", 2, true, "Vector Arithmetic", true, true, false},

            {"VAddF64", "v_add_f64", "VAddF64", 2, true, "Vector Arithmetic", true, true, false},

            {"VAddI32", "v_add_i32", "VAddI32", 2, true, "Vector Arithmetic", true, true, false},

            {"VAddCOU32",
             "v_add_c_o_u32",
             "VAddCOU32",
             2,
             true,
             "Vector Arithmetic",
             true,
             true,
             false},

            {"VAddCCOU32",
             "v_add_c_c_o_u32",
             "VAddCCOU32",
             2,
             true,
             "Vector Arithmetic",
             true,
             true,
             false},

            {"VAddPKF16",
             "v_add_p_k_f16",
             "VAddPKF16",
             2,
             true,
             "Vector Arithmetic",
             true,
             true,
             false},

            {"VAdd3U32", "v_add3_u32", "VAdd3U32", 2, true, "Vector Arithmetic", true, true, false},

            {"VSubF32", "v_sub_f32", "VSubF32", 2, true, "Vector Arithmetic", true, true, false},

            {"VSubI32", "v_sub_i32", "VSubI32", 2, true, "Vector Arithmetic", true, true, false},

            {"VSubU32", "v_sub_u32", "VSubU32", 2, true, "Vector Arithmetic", true, true, false},

            {"VSubCoU32",
             "v_sub_co_u32",
             "VSubCoU32",
             2,
             true,
             "Vector Arithmetic",
             true,
             true,
             false},

            {"VMulF16", "v_mul_f16", "VMulF16", 2, true, "Vector Arithmetic", true, true, false},

            {"VMulF64", "v_mul_f64", "VMulF64", 2, true, "Vector Arithmetic", true, true, false},

            {"VMulPKF16",
             "v_mul_p_k_f16",
             "VMulPKF16",
             2,
             true,
             "Vector Arithmetic",
             true,
             true,
             false},

            {"VMulPKF32S",
             "v_mul_p_k_f32_s",
             "VMulPKF32S",
             2,
             true,
             "Vector Arithmetic",
             true,
             true,
             false},

            {"VMulLOU32",
             "v_mul_l_o_u32",
             "VMulLOU32",
             2,
             true,
             "Vector Arithmetic",
             true,
             true,
             false},

            {"VMulHII32",
             "v_mul_h_i_i32",
             "VMulHII32",
             2,
             true,
             "Vector Arithmetic",
             true,
             true,
             false},

            {"VMulHIU32",
             "v_mul_h_i_u32",
             "VMulHIU32",
             2,
             true,
             "Vector Arithmetic",
             true,
             true,
             false},

            {"VMulI32I24",
             "v_mul_i32_i24",
             "VMulI32I24",
             2,
             true,
             "Vector Arithmetic",
             true,
             true,
             false},

            {"VMulU32U24",
             "v_mul_u32_u24",
             "VMulU32U24",
             2,
             true,
             "Vector Arithmetic",
             true,
             true,
             false},

            {"VMacF32", "v_mac_f32", "VMacF32", 2, true, "Vector Arithmetic", true, true, false},

            {"VFmaF16", "v_fma_f16", "VFmaF16", 3, true, "Vector Arithmetic", true, true, false},

            {"VFmaF32", "v_fma_f32", "VFmaF32", 3, true, "Vector Arithmetic", true, true, false},

            {"VFmaF64", "v_fma_f64", "VFmaF64", 3, true, "Vector Arithmetic", true, true, false},

            {"VFmaPKF16",
             "v_fma_p_k_f16",
             "VFmaPKF16",
             3,
             true,
             "Vector Arithmetic",
             true,
             true,
             false},

            {"VFmaMixF32",
             "v_fma_mix_f32",
             "VFmaMixF32",
             3,
             true,
             "Vector Arithmetic",
             true,
             true,
             false},

            {"VMadI32I24",
             "v_mad_i32_i24",
             "VMadI32I24",
             3,
             true,
             "Vector Arithmetic",
             true,
             true,
             false},

            {"VMadU32U24",
             "v_mad_u32_u24",
             "VMadU32U24",
             3,
             true,
             "Vector Arithmetic",
             true,
             true,
             false},

            {"VMadMixF32",
             "v_mad_mix_f32",
             "VMadMixF32",
             3,
             true,
             "Vector Arithmetic",
             true,
             true,
             false},

            {"VDot2CF32F16",
             "v_dot2_c_f32_f16",
             "VDot2CF32F16",
             3,
             true,
             "Vector Other",
             true,
             true,
             false},

            {"VDot2F32F16",
             "v_dot2_f32_f16",
             "VDot2F32F16",
             3,
             true,
             "Vector Other",
             true,
             true,
             false},

            {"VDot2F32BF16",
             "v_dot2_f32_b_f16",
             "VDot2F32BF16",
             3,
             true,
             "Vector Other",
             true,
             true,
             false},

            {"VDot2CF32BF16",
             "v_dot2_c_f32_b_f16",
             "VDot2CF32BF16",
             3,
             true,
             "Vector Other",
             true,
             true,
             false},

            {"VExpF16",
             "v_exp_f16",
             "VExpF16",
             1,
             true,
             "Vector Transcendental",
             true,
             true,
             false},

            {"VExpF32",
             "v_exp_f32",
             "VExpF32",
             1,
             true,
             "Vector Transcendental",
             true,
             true,
             false},

            {"VRcpF16",
             "v_rcp_f16",
             "VRcpF16",
             1,
             true,
             "Vector Transcendental",
             true,
             true,
             false},

            {"VRcpF32",
             "v_rcp_f32",
             "VRcpF32",
             1,
             true,
             "Vector Transcendental",
             true,
             true,
             false},

            {"VRcpIFlagF32",
             "v_rcp_i_flag_f32",
             "VRcpIFlagF32",
             1,
             true,
             "Vector Transcendental",
             true,
             true,
             false},

            {"VRsqF16",
             "v_rsq_f16",
             "VRsqF16",
             1,
             true,
             "Vector Transcendental",
             true,
             true,
             false},

            {"VRsqF32",
             "v_rsq_f32",
             "VRsqF32",
             1,
             true,
             "Vector Transcendental",
             true,
             true,
             false},

            {"VRsqIFlagF32",
             "v_rsq_i_flag_f32",
             "VRsqIFlagF32",
             1,
             true,
             "Vector Transcendental",
             true,
             true,
             false},

            {"VRndneF32",
             "v_rndne_f32",
             "VRndneF32",
             1,
             true,
             "Vector Transcendental",
             true,
             true,
             false},

            {"VMaxF16", "v_max_f16", "VMaxF16", 2, true, "Vector MinMax", true, true, false},

            {"VMaxF32", "v_max_f32", "VMaxF32", 2, true, "Vector MinMax", true, true, false},

            {"VMaxF64", "v_max_f64", "VMaxF64", 2, true, "Vector MinMax", true, true, false},

            {"VMaxI32", "v_max_i32", "VMaxI32", 2, true, "Vector MinMax", true, true, false},

            {"VMaxPKF16",
             "v_max_p_k_f16",
             "VMaxPKF16",
             2,
             true,
             "Vector MinMax",
             true,
             true,
             false},

            {"VMinF16", "v_min_f16", "VMinF16", 2, true, "Vector MinMax", true, true, false},

            {"VMinF32", "v_min_f32", "VMinF32", 2, true, "Vector MinMax", true, true, false},

            {"VMinF64", "v_min_f64", "VMinF64", 2, true, "Vector MinMax", true, true, false},

            {"VMinI32", "v_min_i32", "VMinI32", 2, true, "Vector MinMax", true, true, false},

            {"VMed3I32", "v_med3_i32", "VMed3I32", 3, true, "Vector MinMax", true, true, false},

            {"VMed3F32", "v_med3_f32", "VMed3F32", 3, true, "Vector MinMax", true, true, false},

            {"VAndB32", "v_and_b32", "VAndB32", 2, true, "Vector Bitwise", true, true, false},

            {"VAndOrB32",
             "v_and_or_b32",
             "VAndOrB32",
             3,
             true,
             "Vector Bitwise",
             true,
             true,
             false},

            {"VOrB32", "v_or_b32", "VOrB32", 2, true, "Vector Bitwise", true, true, false},

            {"VXorB32", "v_xor_b32", "VXorB32", 2, true, "Vector Bitwise", true, true, false},

            {"VNotB32", "v_not_b32", "VNotB32", 1, true, "Vector Bitwise", true, true, false},

            {"VPrngB32", "v_prng_b32", "VPrngB32", 1, true, "Vector Other", true, true, false},

            {"VCndMaskB32",
             "v_cnd_mask_b32",
             "VCndMaskB32",
             2,
             true,
             "Vector Other",
             true,
             true,
             false},

            {"VLShiftLeftB16",
             "v_l_shift_left_b16",
             "VLShiftLeftB16",
             2,
             true,
             "Vector Shift",
             true,
             true,
             false},

            {"VLShiftLeftB32",
             "v_l_shift_left_b32",
             "VLShiftLeftB32",
             2,
             true,
             "Vector Shift",
             true,
             true,
             false},

            {"VLShiftRightB32",
             "v_l_shift_right_b32",
             "VLShiftRightB32",
             2,
             true,
             "Vector Shift",
             true,
             true,
             false},

            {"VLShiftLeftB64",
             "v_l_shift_left_b64",
             "VLShiftLeftB64",
             2,
             true,
             "Vector Shift",
             true,
             true,
             false},

            {"VLShiftRightB64",
             "v_l_shift_right_b64",
             "VLShiftRightB64",
             2,
             true,
             "Vector Shift",
             true,
             true,
             false},

            {"VAShiftRightI32",
             "v_a_shift_right_i32",
             "VAShiftRightI32",
             2,
             true,
             "Vector Shift",
             true,
             true,
             false},

            {"VMovB32", "v_mov_b32", "VMovB32", 1, true, "Vector Move", false, false, false},

            {"VSwapB32", "v_swap_b32", "VSwapB32", 1, true, "Vector Move", true, true, false},

            {"VPackF16toB32",
             "v_pack_f16to_b32",
             "VPackF16toB32",
             2,
             true,
             "Vector Move",
             true,
             true,
             false},

            {"VPermB32", "v_perm_b32", "VPermB32", 3, true, "Vector Shift", true, true, false},

            {"VBfeI32", "v_bfe_i32", "VBfeI32", 3, true, "Vector Other", true, true, false},

            {"VBfeU32", "v_bfe_u32", "VBfeU32", 3, true, "Vector Other", true, true, false},

            {"VBfiB32", "v_bfi_b32", "VBfiB32", 3, true, "Vector Other", true, true, false},

            {"VAccvgprReadB32",
             "v_accvgpr_read_b32",
             "VAccvgprReadB32",
             1,
             true,
             "Vector Other",
             false,
             false,
             false},

            {"VAccvgprWrite",
             "v_accvgpr_write",
             "VAccvgprWrite",
             1,
             true,
             "Vector Other",
             false,
             false,
             false},

            {"VAccvgprWriteB32",
             "v_accvgpr_write_b32",
             "VAccvgprWriteB32",
             1,
             true,
             "Vector Other",
             false,
             false,
             false},

            {"VReadfirstlaneB32",
             "v_readfirstlane_b32",
             "VReadfirstlaneB32",
             1,
             true,
             "Vector Other",
             false,
             false,
             false},

            {"SCmpEQI32",
             "s_cmp_e_q_i32",
             "SCmpEQI32",
             2,
             false,
             "Scalar Compare",
             false,
             false,
             false},

            {"SCmpEQU32",
             "s_cmp_e_q_u32",
             "SCmpEQU32",
             2,
             false,
             "Scalar Compare",
             false,
             false,
             false},

            {"SCmpEQU64",
             "s_cmp_e_q_u64",
             "SCmpEQU64",
             2,
             false,
             "Scalar Compare",
             false,
             false,
             false},

            {"SCmpGeI32",
             "s_cmp_ge_i32",
             "SCmpGeI32",
             2,
             false,
             "Scalar Compare",
             false,
             false,
             false},

            {"SCmpGeU32",
             "s_cmp_ge_u32",
             "SCmpGeU32",
             2,
             false,
             "Scalar Compare",
             false,
             false,
             false},

            {"SCmpGtI32",
             "s_cmp_gt_i32",
             "SCmpGtI32",
             2,
             false,
             "Scalar Compare",
             false,
             false,
             false},

            {"SCmpGtU32",
             "s_cmp_gt_u32",
             "SCmpGtU32",
             2,
             false,
             "Scalar Compare",
             false,
             false,
             false},

            {"SCmpLeI32",
             "s_cmp_le_i32",
             "SCmpLeI32",
             2,
             false,
             "Scalar Compare",
             false,
             false,
             false},

            {"SCmpLeU32",
             "s_cmp_le_u32",
             "SCmpLeU32",
             2,
             false,
             "Scalar Compare",
             false,
             false,
             false},

            {"SCmpLgU32",
             "s_cmp_lg_u32",
             "SCmpLgU32",
             2,
             false,
             "Scalar Compare",
             false,
             false,
             false},

            {"SCmpLgI32",
             "s_cmp_lg_i32",
             "SCmpLgI32",
             2,
             false,
             "Scalar Compare",
             false,
             false,
             false},

            {"SCmpLgU64",
             "s_cmp_lg_u64",
             "SCmpLgU64",
             2,
             false,
             "Scalar Compare",
             false,
             false,
             false},

            {"SCmpLtI32",
             "s_cmp_lt_i32",
             "SCmpLtI32",
             2,
             false,
             "Scalar Compare",
             false,
             false,
             false},

            {"SCmpLtU32",
             "s_cmp_lt_u32",
             "SCmpLtU32",
             2,
             false,
             "Scalar Compare",
             false,
             false,
             false},

            {"SBitcmp1B32",
             "s_bitcmp1_b32",
             "SBitcmp1B32",
             2,
             false,
             "Scalar Other",
             false,
             false,
             false},

            {"VCmpEQF32",
             "v_cmp_e_q_f32",
             "VCmpEQF32",
             2,
             true,
             "Vector Compare",
             true,
             true,
             false},

            {"VCmpEQF64",
             "v_cmp_e_q_f64",
             "VCmpEQF64",
             2,
             true,
             "Vector Compare",
             true,
             true,
             false},

            {"VCmpEQU32",
             "v_cmp_e_q_u32",
             "VCmpEQU32",
             2,
             true,
             "Vector Compare",
             true,
             true,
             false},

            {"VCmpEQI32",
             "v_cmp_e_q_i32",
             "VCmpEQI32",
             2,
             true,
             "Vector Compare",
             true,
             true,
             false},

            {"VCmpGEF16",
             "v_cmp_g_e_f16",
             "VCmpGEF16",
             2,
             true,
             "Vector Compare",
             true,
             true,
             false},

            {"VCmpGTF16",
             "v_cmp_g_t_f16",
             "VCmpGTF16",
             2,
             true,
             "Vector Compare",
             true,
             true,
             false},

            {"VCmpGEF32",
             "v_cmp_g_e_f32",
             "VCmpGEF32",
             2,
             true,
             "Vector Compare",
             true,
             true,
             false},

            {"VCmpGTF32",
             "v_cmp_g_t_f32",
             "VCmpGTF32",
             2,
             true,
             "Vector Compare",
             true,
             true,
             false},

            {"VCmpGEF64",
             "v_cmp_g_e_f64",
             "VCmpGEF64",
             2,
             true,
             "Vector Compare",
             true,
             true,
             false},

            {"VCmpGTF64",
             "v_cmp_g_t_f64",
             "VCmpGTF64",
             2,
             true,
             "Vector Compare",
             true,
             true,
             false},

            {"VCmpGEI32",
             "v_cmp_g_e_i32",
             "VCmpGEI32",
             2,
             true,
             "Vector Compare",
             true,
             true,
             false},

            {"VCmpGTI32",
             "v_cmp_g_t_i32",
             "VCmpGTI32",
             2,
             true,
             "Vector Compare",
             true,
             true,
             false},

            {"VCmpGEU32",
             "v_cmp_g_e_u32",
             "VCmpGEU32",
             2,
             true,
             "Vector Compare",
             true,
             true,
             false},

            {"VCmpGtU32",
             "v_cmp_gt_u32",
             "VCmpGtU32",
             2,
             true,
             "Vector Compare",
             true,
             true,
             false},

            {"VCmpLeU32",
             "v_cmp_le_u32",
             "VCmpLeU32",
             2,
             true,
             "Vector Compare",
             true,
             true,
             false},

            {"VCmpLeI32",
             "v_cmp_le_i32",
             "VCmpLeI32",
             2,
             true,
             "Vector Compare",
             true,
             true,
             false},

            {"VCmpLtI32",
             "v_cmp_lt_i32",
             "VCmpLtI32",
             2,
             true,
             "Vector Compare",
             true,
             true,
             false},

            {"VCmpLtU32",
             "v_cmp_lt_u32",
             "VCmpLtU32",
             2,
             true,
             "Vector Compare",
             true,
             true,
             false},

            {"VCmpUF32", "v_cmp_u_f32", "VCmpUF32", 2, true, "Vector Compare", true, true, false},

            {"VCmpNeI32",
             "v_cmp_ne_i32",
             "VCmpNeI32",
             2,
             true,
             "Vector Compare",
             true,
             true,
             false},

            {"VCmpNeU32",
             "v_cmp_ne_u32",
             "VCmpNeU32",
             2,
             true,
             "Vector Compare",
             true,
             true,
             false},

            {"VCmpNeU64",
             "v_cmp_ne_u64",
             "VCmpNeU64",
             2,
             true,
             "Vector Compare",
             true,
             true,
             false},

            {"VCmpClassF32",
             "v_cmp_class_f32",
             "VCmpClassF32",
             2,
             true,
             "Vector Compare",
             true,
             true,
             false},

            {"VCmpXClassF32",
             "v_cmp_x_class_f32",
             "VCmpXClassF32",
             2,
             true,
             "Vector Compare",
             true,
             true,
             false},

            {"VCmpXEqU32",
             "v_cmp_x_eq_u32",
             "VCmpXEqU32",
             2,
             true,
             "Vector Compare",
             true,
             true,
             false},

            {"VCmpXGeU32",
             "v_cmp_x_ge_u32",
             "VCmpXGeU32",
             2,
             true,
             "Vector Compare",
             true,
             true,
             false},

            {"VCmpXGtU32",
             "v_cmp_x_gt_u32",
             "VCmpXGtU32",
             2,
             true,
             "Vector Compare",
             true,
             true,
             false},

            {"VCmpXLeU32",
             "v_cmp_x_le_u32",
             "VCmpXLeU32",
             2,
             true,
             "Vector Compare",
             true,
             true,
             false},

            {"VCmpXLeI32",
             "v_cmp_x_le_i32",
             "VCmpXLeI32",
             2,
             true,
             "Vector Compare",
             true,
             true,
             false},

            {"VCmpXLtF32",
             "v_cmp_x_lt_f32",
             "VCmpXLtF32",
             2,
             true,
             "Vector Compare",
             true,
             true,
             false},

            {"VCmpXLtI32",
             "v_cmp_x_lt_i32",
             "VCmpXLtI32",
             2,
             true,
             "Vector Compare",
             true,
             true,
             false},

            {"VCmpXLtU32",
             "v_cmp_x_lt_u32",
             "VCmpXLtU32",
             2,
             true,
             "Vector Compare",
             true,
             true,
             false},

            {"VCmpXLtU64",
             "v_cmp_x_lt_u64",
             "VCmpXLtU64",
             2,
             true,
             "Vector Compare",
             true,
             true,
             false},

            {"VCmpXNeU16",
             "v_cmp_x_ne_u16",
             "VCmpXNeU16",
             2,
             true,
             "Vector Compare",
             true,
             true,
             false},

            {"VCmpXNeU32",
             "v_cmp_x_ne_u32",
             "VCmpXNeU32",
             2,
             true,
             "Vector Compare",
             true,
             true,
             false},

            {"VCvtF16toF32",
             "v_cvt_f16to_f32",
             "VCvtF16toF32",
             1,
             true,
             "Vector Conversion",
             true,
             true,
             false},

            {"VCvtF32toF16",
             "v_cvt_f32to_f16",
             "VCvtF32toF16",
             1,
             true,
             "Vector Conversion",
             true,
             true,
             false},

            {"VCvtF32toU32",
             "v_cvt_f32to_u32",
             "VCvtF32toU32",
             1,
             true,
             "Vector Conversion",
             true,
             true,
             false},

            {"VCvtU32toF32",
             "v_cvt_u32to_f32",
             "VCvtU32toF32",
             1,
             true,
             "Vector Conversion",
             true,
             true,
             false},

            {"VCvtI32toF32",
             "v_cvt_i32to_f32",
             "VCvtI32toF32",
             1,
             true,
             "Vector Conversion",
             true,
             true,
             false},

            {"VCvtF32toI32",
             "v_cvt_f32to_i32",
             "VCvtF32toI32",
             1,
             true,
             "Vector Conversion",
             true,
             true,
             false},

            {"VCvtFP8toF32",
             "v_cvt_f_p8to_f32",
             "VCvtFP8toF32",
             1,
             true,
             "Vector Conversion",
             true,
             true,
             false},

            {"VCvtBF8toF32",
             "v_cvt_b_f8to_f32",
             "VCvtBF8toF32",
             1,
             true,
             "Vector Conversion",
             true,
             true,
             false},

            {"VCvtPkFP8toF32",
             "v_cvt_pk_f_p8to_f32",
             "VCvtPkFP8toF32",
             1,
             true,
             "Vector Conversion",
             true,
             true,
             false},

            {"VCvtPkBF8toF32",
             "v_cvt_pk_b_f8to_f32",
             "VCvtPkBF8toF32",
             1,
             true,
             "Vector Conversion",
             true,
             true,
             false},

            {"VCvtPkF32toFP8",
             "v_cvt_pk_f32to_f_p8",
             "VCvtPkF32toFP8",
             2,
             true,
             "Vector Conversion",
             true,
             true,
             false},

            {"VCvtPkF32toBF8",
             "v_cvt_pk_f32to_b_f8",
             "VCvtPkF32toBF8",
             2,
             true,
             "Vector Conversion",
             true,
             true,
             false},

            {"VCvtSRF32toFP8",
             "v_cvt_s_r_f32to_f_p8",
             "VCvtSRF32toFP8",
             2,
             true,
             "Vector Conversion",
             true,
             true,
             false},

            {"VCvtSRF32toBF8",
             "v_cvt_s_r_f32to_b_f8",
             "VCvtSRF32toBF8",
             2,
             true,
             "Vector Conversion",
             true,
             true,
             false},

            {"VCvtScalePkFP8toF16",
             "v_cvt_scale_pk_f_p8to_f16",
             "VCvtScalePkFP8toF16",
             2,
             true,
             "Vector Conversion",
             true,
             true,
             false},

            {"VCvtScalePkBF8toF16",
             "v_cvt_scale_pk_b_f8to_f16",
             "VCvtScalePkBF8toF16",
             2,
             true,
             "Vector Conversion",
             true,
             true,
             false},

            {"VCvtScaleFP8toF16",
             "v_cvt_scale_f_p8to_f16",
             "VCvtScaleFP8toF16",
             2,
             true,
             "Vector Conversion",
             true,
             true,
             false},

            {"VCvtScalePkF16toFP8",
             "v_cvt_scale_pk_f16to_f_p8",
             "VCvtScalePkF16toFP8",
             2,
             true,
             "Vector Conversion",
             true,
             true,
             false},

            {"VCvtScalePkF16toBF8",
             "v_cvt_scale_pk_f16to_b_f8",
             "VCvtScalePkF16toBF8",
             2,
             true,
             "Vector Conversion",
             true,
             true,
             false},

            {"VCvtScaleSRF16toFP8",
             "v_cvt_scale_s_r_f16to_f_p8",
             "VCvtScaleSRF16toFP8",
             2,
             true,
             "Vector Conversion",
             true,
             true,
             false},

            {"VCvtScaleSRF16toBF8",
             "v_cvt_scale_s_r_f16to_b_f8",
             "VCvtScaleSRF16toBF8",
             2,
             true,
             "Vector Conversion",
             true,
             true,
             false},

            {"VCvtPkF32toBF16",
             "v_cvt_pk_f32to_b_f16",
             "VCvtPkF32toBF16",
             2,
             true,
             "Vector Conversion",
             true,
             true,
             false},

            {"DSBPermuteB32",
             "d_s_b_permute_b32",
             "DSBPermuteB32",
             1,
             true,
             "Memory Shift",
             false,
             false,
             true},

            // DS (Local Data Share) Load instructions
            {"DSLoadU8", "ds_load_u8", "DSLoadU8", 1, true, "Memory LDS", false, false, false},
            {"DSLoadI8", "ds_load_i8", "DSLoadI8", 1, true, "Memory LDS", false, false, false},
            {"DSLoadU16", "ds_load_u16", "DSLoadU16", 1, true, "Memory LDS", false, false, false},
            {"DSLoadI16", "ds_load_i16", "DSLoadI16", 1, true, "Memory LDS", false, false, false},
            {"DSLoadB32", "ds_load_b32", "DSLoadB32", 1, true, "Memory LDS", false, false, false},
            {"DSLoadB64", "ds_load_b64", "DSLoadB64", 1, true, "Memory LDS", false, false, false},
            {"DSLoadB96", "ds_load_b96", "DSLoadB96", 1, true, "Memory LDS", false, false, false},
            {"DSLoadB128",
             "ds_load_b128",
             "DSLoadB128",
             1,
             true,
             "Memory LDS",
             false,
             false,
             false},
            {"DSLoad2B32",
             "ds_load_2_b32",
             "DSLoad2B32",
             2,
             true,
             "Memory LDS",
             false,
             false,
             false},
            {"DSLoad2B64",
             "ds_load_2_b64",
             "DSLoad2B64",
             2,
             true,
             "Memory LDS",
             false,
             false,
             false},

            // DS (Local Data Share) Store instructions
            {"DSStoreB8", "ds_store_b8", "DSStoreB8", 2, false, "Memory LDS", false, false, false},
            {"DSStoreB16",
             "ds_store_b16",
             "DSStoreB16",
             2,
             false,
             "Memory LDS",
             false,
             false,
             false},
            {"DSStoreB32",
             "ds_store_b32",
             "DSStoreB32",
             2,
             false,
             "Memory LDS",
             false,
             false,
             false},
            {"DSStoreB64",
             "ds_store_b64",
             "DSStoreB64",
             2,
             false,
             "Memory LDS",
             false,
             false,
             false},
            {"DSStoreB96",
             "ds_store_b96",
             "DSStoreB96",
             2,
             false,
             "Memory LDS",
             false,
             false,
             false},
            {"DSStoreB128",
             "ds_store_b128",
             "DSStoreB128",
             2,
             false,
             "Memory LDS",
             false,
             false,
             false},
            {"DSStore2B32",
             "ds_store_2_b32",
             "DSStore2B32",
             3,
             false,
             "Memory LDS",
             false,
             false,
             false},
            {"DSStore2B64",
             "ds_store_2_b64",
             "DSStore2B64",
             3,
             false,
             "Memory LDS",
             false,
             false,
             false},

            {"BufferLoadU8",
             "buffer_load_u8",
             "BufferLoadU8",
             1,
             true,
             "Memory Memory",
             false,
             false,
             false},

            {"BufferLoadI8",
             "buffer_load_i8",
             "BufferLoadI8",
             1,
             true,
             "Memory Memory",
             false,
             false,
             false},

            {"BufferLoadU16",
             "buffer_load_u16",
             "BufferLoadU16",
             1,
             true,
             "Memory Memory",
             false,
             false,
             false},

            {"BufferLoadI16",
             "buffer_load_i16",
             "BufferLoadI16",
             1,
             true,
             "Memory Memory",
             false,
             false,
             false},

            {"BufferLoadB32",
             "buffer_load_b32",
             "BufferLoadB32",
             1,
             true,
             "Memory Memory",
             false,
             false,
             false},

            {"BufferLoadB64",
             "buffer_load_b64",
             "BufferLoadB64",
             1,
             true,
             "Memory Memory",
             false,
             false,
             false},

            {"BufferLoadB96",
             "buffer_load_b96",
             "BufferLoadB96",
             1,
             true,
             "Memory Memory",
             false,
             false,
             false},

            {"BufferLoadB128",
             "buffer_load_b128",
             "BufferLoadB128",
             1,
             true,
             "Memory Memory",
             false,
             false,
             false},

            {"BufferLoadD16U8",
             "buffer_load_d16_u8",
             "BufferLoadD16U8",
             1,
             true,
             "Memory Memory",
             false,
             false,
             false},

            {"BufferLoadD16HIU8",
             "buffer_load_d16_h_i_u8",
             "BufferLoadD16HIU8",
             1,
             true,
             "Memory Memory",
             false,
             false,
             false},

            {"BufferLoadD16I8",
             "buffer_load_d16_i8",
             "BufferLoadD16I8",
             1,
             true,
             "Memory Memory",
             false,
             false,
             false},

            {"BufferLoadD16HII8",
             "buffer_load_d16_h_i_i8",
             "BufferLoadD16HII8",
             1,
             true,
             "Memory Memory",
             false,
             false,
             false},

            {"BufferLoadD16B16",
             "buffer_load_d16_b16",
             "BufferLoadD16B16",
             1,
             true,
             "Memory Memory",
             false,
             false,
             false},

            {"BufferLoadD16HIB16",
             "buffer_load_d16_h_i_b16",
             "BufferLoadD16HIB16",
             1,
             true,
             "Memory Memory",
             false,
             false,
             false},

            {"BufferStoreB8",
             "buffer_store_b8",
             "BufferStoreB8",
             2,
             true,
             "Memory Memory",
             false,
             false,
             false},

            {"BufferStoreD16HIU8",
             "buffer_store_d16_h_i_u8",
             "BufferStoreD16HIU8",
             2,
             true,
             "Memory Memory",
             false,
             false,
             false},

            {"BufferStoreB16",
             "buffer_store_b16",
             "BufferStoreB16",
             2,
             true,
             "Memory Memory",
             false,
             false,
             false},

            {"BufferStoreD16HIB16",
             "buffer_store_d16_h_i_b16",
             "BufferStoreD16HIB16",
             2,
             true,
             "Memory Memory",
             false,
             false,
             false},

            {"BufferStoreB32",
             "buffer_store_b32",
             "BufferStoreB32",
             2,
             true,
             "Memory Memory",
             false,
             false,
             false},

            {"BufferStoreB64",
             "buffer_store_b64",
             "BufferStoreB64",
             2,
             true,
             "Memory Memory",
             false,
             false,
             false},

            {"BufferStoreB96",
             "buffer_store_b96",
             "BufferStoreB96",
             2,
             true,
             "Memory Memory",
             false,
             false,
             false},

            {"BufferStoreB128",
             "buffer_store_b128",
             "BufferStoreB128",
             2,
             true,
             "Memory Memory",
             false,
             false,
             false},

            {"BufferAtomicAddF32",
             "buffer_atomic_add_f32",
             "BufferAtomicAddF32",
             1,
             true,
             "Memory Arithmetic",
             false,
             false,
             false},

            {"BufferAtomicCmpswapB32",
             "buffer_atomic_cmpswap_b32",
             "BufferAtomicCmpswapB32",
             2,
             true,
             "Memory Compare",
             false,
             false,
             false},

            {"BufferAtomicCmpswapB64",
             "buffer_atomic_cmpswap_b64",
             "BufferAtomicCmpswapB64",
             2,
             true,
             "Memory Compare",
             false,
             false,
             false},

            {"FlatLoadU8",
             "flat_load_u8",
             "FlatLoadU8",
             1,
             true,
             "Memory Memory",
             false,
             false,
             false},

            {"FlatLoadI8",
             "flat_load_i8",
             "FlatLoadI8",
             1,
             true,
             "Memory Memory",
             false,
             false,
             false},

            {"FlatLoadU16",
             "flat_load_u16",
             "FlatLoadU16",
             1,
             true,
             "Memory Memory",
             false,
             false,
             false},

            {"FlatLoadI16",
             "flat_load_i16",
             "FlatLoadI16",
             1,
             true,
             "Memory Memory",
             false,
             false,
             false},

            {"FlatLoadD16U8",
             "flat_load_d16_u8",
             "FlatLoadD16U8",
             1,
             true,
             "Memory Memory",
             false,
             false,
             false},

            {"FlatLoadD16HIU8",
             "flat_load_d16_h_i_u8",
             "FlatLoadD16HIU8",
             1,
             true,
             "Memory Memory",
             false,
             false,
             false},

            {"FlatLoadD16I8",
             "flat_load_d16_i8",
             "FlatLoadD16I8",
             1,
             true,
             "Memory Memory",
             false,
             false,
             false},

            {"FlatLoadD16HII8",
             "flat_load_d16_h_i_i8",
             "FlatLoadD16HII8",
             1,
             true,
             "Memory Memory",
             false,
             false,
             false},

            {"FlatLoadD16B16",
             "flat_load_d16_b16",
             "FlatLoadD16B16",
             1,
             true,
             "Memory Memory",
             false,
             false,
             false},

            {"FlatLoadD16HIB16",
             "flat_load_d16_h_i_b16",
             "FlatLoadD16HIB16",
             1,
             true,
             "Memory Memory",
             false,
             false,
             false},

            {"FlatLoadB32",
             "flat_load_b32",
             "FlatLoadB32",
             1,
             true,
             "Memory Memory",
             false,
             false,
             false},

            {"FlatLoadB64",
             "flat_load_b64",
             "FlatLoadB64",
             1,
             true,
             "Memory Memory",
             false,
             false,
             false},

            {"FlatLoadB96",
             "flat_load_b96",
             "FlatLoadB96",
             1,
             true,
             "Memory Memory",
             false,
             false,
             false},

            {"FlatLoadB128",
             "flat_load_b128",
             "FlatLoadB128",
             1,
             true,
             "Memory Memory",
             false,
             false,
             false},

            {"FlatStoreB8",
             "flat_store_b8",
             "FlatStoreB8",
             2,
             true,
             "Memory Memory",
             false,
             false,
             false},

            {"FlatStoreD16HIU8",
             "flat_store_d16_h_i_u8",
             "FlatStoreD16HIU8",
             2,
             true,
             "Memory Memory",
             false,
             false,
             false},

            {"FlatStoreB16",
             "flat_store_b16",
             "FlatStoreB16",
             2,
             true,
             "Memory Memory",
             false,
             false,
             false},

            {"FlatStoreD16HIB16",
             "flat_store_d16_h_i_b16",
             "FlatStoreD16HIB16",
             2,
             true,
             "Memory Memory",
             false,
             false,
             false},

            {"FlatStoreB32",
             "flat_store_b32",
             "FlatStoreB32",
             2,
             true,
             "Memory Memory",
             false,
             false,
             false},

            {"FlatStoreB64",
             "flat_store_b64",
             "FlatStoreB64",
             2,
             true,
             "Memory Memory",
             false,
             false,
             false},

            {"FlatStoreB96",
             "flat_store_b96",
             "FlatStoreB96",
             2,
             true,
             "Memory Memory",
             false,
             false,
             false},

            {"FlatStoreB128",
             "flat_store_b128",
             "FlatStoreB128",
             2,
             true,
             "Memory Memory",
             false,
             false,
             false},

            {"FlatAtomicCmpswapB32",
             "flat_atomic_cmpswap_b32",
             "FlatAtomicCmpswapB32",
             2,
             true,
             "Memory Compare",
             false,
             false,
             false},

            {"SAbsI32", "s_abs_i32", "SAbsI32", 1, true, "Scalar Arithmetic", false, false, false},

            {"SBarrier", "s_barrier", "SBarrier", 0, false, "Scalar Control", false, false, false},

            {"SMaxI32", "s_max_i32", "SMaxI32", 2, true, "Scalar MinMax", false, false, false},

            {"SMaxU32", "s_max_u32", "SMaxU32", 2, true, "Scalar MinMax", false, false, false},

            {"SMinI32", "s_min_i32", "SMinI32", 2, true, "Scalar MinMax", false, false, false},

            {"SMinU32", "s_min_u32", "SMinU32", 2, true, "Scalar MinMax", false, false, false},

            {"SAddI32", "s_add_i32", "SAddI32", 2, true, "Scalar Arithmetic", false, false, false},

            {"SAddU32", "s_add_u32", "SAddU32", 2, true, "Scalar Arithmetic", false, false, false},

            {"SAddCU32",
             "s_add_c_u32",
             "SAddCU32",
             2,
             true,
             "Scalar Arithmetic",
             false,
             false,
             false},

            {"SMulI32", "s_mul_i32", "SMulI32", 2, true, "Scalar Arithmetic", false, false, false},

            {"SMulHII32",
             "s_mul_h_i_i32",
             "SMulHII32",
             2,
             true,
             "Scalar Arithmetic",
             false,
             false,
             false},

            {"SMulHIU32",
             "s_mul_h_i_u32",
             "SMulHIU32",
             2,
             true,
             "Scalar Arithmetic",
             false,
             false,
             false},

            {"SMulLOU32",
             "s_mul_l_o_u32",
             "SMulLOU32",
             2,
             true,
             "Scalar Arithmetic",
             false,
             false,
             false},

            {"SSubI32", "s_sub_i32", "SSubI32", 2, true, "Scalar Arithmetic", false, false, false},

            {"SSubU32", "s_sub_u32", "SSubU32", 2, true, "Scalar Arithmetic", false, false, false},

            {"SSubBU32",
             "s_sub_b_u32",
             "SSubBU32",
             2,
             true,
             "Scalar Arithmetic",
             false,
             false,
             false},

            {"SSubU64", "s_sub_u64", "SSubU64", 2, true, "Scalar Arithmetic", false, false, false},

            {"SAndB32", "s_and_b32", "SAndB32", 2, true, "Scalar Bitwise", false, false, false},

            {"SAndB64", "s_and_b64", "SAndB64", 2, true, "Scalar Bitwise", false, false, false},

            {"SAndN2B32",
             "s_and_n2_b32",
             "SAndN2B32",
             2,
             true,
             "Scalar Bitwise",
             false,
             false,
             false},

            {"SOrB32", "s_or_b32", "SOrB32", 2, true, "Scalar Bitwise", false, false, false},

            {"SOrB64", "s_or_b64", "SOrB64", 2, true, "Scalar Bitwise", false, false, false},

            {"SXorB32", "s_xor_b32", "SXorB32", 2, true, "Scalar Bitwise", false, false, false},

            {"SLShiftLeftB32",
             "s_l_shift_left_b32",
             "SLShiftLeftB32",
             2,
             true,
             "Scalar Shift",
             false,
             false,
             false},

            {"SLShiftRightB32",
             "s_l_shift_right_b32",
             "SLShiftRightB32",
             2,
             true,
             "Scalar Shift",
             false,
             false,
             false},

            {"SLShiftLeftB64",
             "s_l_shift_left_b64",
             "SLShiftLeftB64",
             2,
             true,
             "Scalar Shift",
             false,
             false,
             false},

            {"SLShiftRightB64",
             "s_l_shift_right_b64",
             "SLShiftRightB64",
             2,
             true,
             "Scalar Shift",
             false,
             false,
             false},

            {"SAShiftRightI32",
             "s_a_shift_right_i32",
             "SAShiftRightI32",
             2,
             true,
             "Scalar Shift",
             false,
             false,
             false},

            {"SLShiftLeft1AddU32",
             "s_l_shift_left1_add_u32",
             "SLShiftLeft1AddU32",
             2,
             true,
             "Scalar Arithmetic",
             false,
             false,
             false},

            {"SLShiftLeft2AddU32",
             "s_l_shift_left2_add_u32",
             "SLShiftLeft2AddU32",
             2,
             true,
             "Scalar Arithmetic",
             false,
             false,
             false},

            {"SLShiftLeft3AddU32",
             "s_l_shift_left3_add_u32",
             "SLShiftLeft3AddU32",
             2,
             true,
             "Scalar Arithmetic",
             false,
             false,
             false},

            {"SLShiftLeft4AddU32",
             "s_l_shift_left4_add_u32",
             "SLShiftLeft4AddU32",
             2,
             true,
             "Scalar Arithmetic",
             false,
             false,
             false},

            {"SMovB32", "s_mov_b32", "SMovB32", 1, true, "Scalar Move", false, false, false},

            {"SMovB64", "s_mov_b64", "SMovB64", 1, true, "Scalar Move", false, false, false},

            {"SCMovB32", "s_c_mov_b32", "SCMovB32", 1, true, "Scalar Move", false, false, false},

            {"SCMovB64", "s_c_mov_b64", "SCMovB64", 1, true, "Scalar Move", false, false, false},

            {"SCSelectB32",
             "s_c_select_b32",
             "SCSelectB32",
             2,
             true,
             "Scalar Other",
             false,
             false,
             false},

            {"SGetPCB64",
             "s_get_p_c_b64",
             "SGetPCB64",
             0,
             true,
             "Scalar Other",
             false,
             false,
             false},

            {"SSetMask", "s_set_mask", "SSetMask", 1, true, "Scalar Other", false, false, false},

            {"SFf1B32", "s_ff1_b32", "SFf1B32", 1, true, "Scalar Other", false, false, false},

            {"SBfmB32", "s_bfm_b32", "SBfmB32", 2, true, "Scalar Other", false, false, false},

            {"SMovkI32", "s_movk_i32", "SMovkI32", 1, true, "Scalar Move", false, false, false},

            {"SSExtI16toI32",
             "s_s_ext_i16to_i32",
             "SSExtI16toI32",
             1,
             true,
             "Scalar Other",
             false,
             false,
             false},

            {"SAndSaveExecB32",
             "s_and_save_exec_b32",
             "SAndSaveExecB32",
             1,
             true,
             "Scalar Bitwise",
             false,
             false,
             false},

            {"SAndSaveExecB64",
             "s_and_save_exec_b64",
             "SAndSaveExecB64",
             1,
             true,
             "Scalar Bitwise",
             false,
             false,
             false},

            {"SOrSaveExecB32",
             "s_or_save_exec_b32",
             "SOrSaveExecB32",
             1,
             true,
             "Scalar Bitwise",
             false,
             false,
             false},

            {"SOrSaveExecB64",
             "s_or_save_exec_b64",
             "SOrSaveExecB64",
             1,
             true,
             "Scalar Bitwise",
             false,
             false,
             false},

            {"SGetRegB32",
             "s_get_reg_b32",
             "SGetRegB32",
             1,
             true,
             "Scalar Control",
             false,
             false,
             false},

            {"SSetRegB32",
             "s_set_reg_b32",
             "SSetRegB32",
             1,
             true,
             "Scalar Control",
             false,
             false,
             false},

            {"SSetRegIMM32B32",
             "s_set_reg_i_m_m32_b32",
             "SSetRegIMM32B32",
             1,
             true,
             "Scalar Control",
             false,
             false,
             false},

            // ========================================================================
            // Composite Instructions
            // ========================================================================

            {"VAddPKF32",
             "v_pk_add_f32",
             "VAddPKF32",
             2,
             true,
             "Vector Arithmetic",
             true,
             true,
             false},

            {"VMulPKF32",
             "v_pk_mul_f32",
             "VMulPKF32",
             2,
             true,
             "Vector Arithmetic",
             true,
             true,
             false},

            {"VMovB64", "v_mov_b64", "VMovB64", 1, true, "Vector Move", false, false, false},

            {"VLShiftLeftOrB32",
             "v_lshl_or_b32",
             "VLShiftLeftOrB32",
             3,
             true,
             "Vector Arithmetic",
             true,
             true,
             false},

        };
    }

    // Generate special MFMA instruction classes (these have custom constructors)
    bool genSpecialMFMAClasses(std::ofstream& out)
    {
        // MFMA class
        out << "    // "
               "========================================================================\n";
        out << "    // Special Matrix Instructions\n";
        out << "    // "
               "========================================================================\n\n";

        out << "    /**\n";
        out << "     * @brief MFMA (Matrix Fused Multiply-Add) instruction\n";
        out << "     */\n";
        out << "    class MFMA : public IRInstruction\n";
        out << "    {\n";
        out << "    public:\n";
        out << "        std::string instType;      ///< Input data type (bf16, f16, i8, etc.)\n";
        out << "        std::string accType;       ///< Accumulator type (f32, i32)\n";
        out << "        int         m;             ///< Matrix M dimension\n";
        out << "        int         n;             ///< Matrix N dimension\n";
        out << "        int         k;             ///< Matrix K dimension\n";
        out << "        int         blocks;        ///< Number of blocks\n";
        out << "        bool        mfma1k;        ///< Whether this is a _1k variant\n";
        out << "        StinkyRegister acc;        ///< Accumulator destination\n";
        out << "        StinkyRegister a;          ///< Matrix A source\n";
        out << "        StinkyRegister b;          ///< Matrix B source\n";
        out << "        std::optional<StinkyRegister> acc2; ///< Optional accumulator source\n";
        out << "        bool        neg;           ///< Negate operands\n";
        out << "\n";
        out << "        MFMA(const std::string& instType,\n";
        out << "             const std::string& accType,\n";
        out << "             int m,\n";
        out << "             int n,\n";
        out << "             int k,\n";
        out << "             int blocks,\n";
        out << "             bool mfma1k,\n";
        out << "             const StinkyRegister& acc,\n";
        out << "             const StinkyRegister& a,\n";
        out << "             const StinkyRegister& b,\n";
        out << "             const StinkyRegister* acc2 = nullptr,\n";
        out << "             bool neg = false,\n";
        out << "             const std::string& comment_ = \"\")\n";
        out << "            : IRInstruction(IRType::StinkyTofu)\n";
        out << "            , instType(instType)\n";
        out << "            , accType(accType)\n";
        out << "            , m(m)\n";
        out << "            , n(n)\n";
        out << "            , k(k)\n";
        out << "            , blocks(blocks)\n";
        out << "            , mfma1k(mfma1k)\n";
        out << "            , acc(acc)\n";
        out << "            , a(a)\n";
        out << "            , b(b)\n";
        out << "            , acc2(acc2 ? std::optional<StinkyRegister>(*acc2) : std::nullopt)\n";
        out << "            , neg(neg)\n";
        out << "        {\n";
        out << "            this->comment = comment_;\n";
        out << "        }\n";
        out << "\n";
        out << "        const char* getLogicalName() const override { return \"MFMA\"; }\n";
        out << "\n";
        out << "        void dump(std::ostream& out) const override\n";
        out << "        {\n";
        out << "            out << \"MFMA (IR)\";\n";
        out << "            if(!comment.empty())\n";
        out << "                out << \"  // \" << comment;\n";
        out << "        }\n";
        out << "    };\n\n";

        // MXMFMA class
        out << "    /**\n";
        out << "     * @brief MXMFMA (MX format MFMA with scale factors) instruction\n";
        out << "     */\n";
        out << "    class MXMFMA : public IRInstruction\n";
        out << "    {\n";
        out << "    public:\n";
        out << "        std::string instType;         ///< Input data type (f8, f4, bf8, etc.)\n";
        out << "        std::string accType;          ///< Accumulator type (f32)\n";
        out << "        std::string mxScaleATypeStr;  ///< Scale format for matrix A\n";
        out << "        std::string mxScaleBTypeStr;  ///< Scale format for matrix B\n";
        out << "        int         m;                ///< Matrix M dimension\n";
        out << "        int         n;                ///< Matrix N dimension\n";
        out << "        int         k;                ///< Matrix K dimension\n";
        out << "        int         block;            ///< Block size\n";
        out << "        StinkyRegister acc;           ///< Accumulator destination\n";
        out << "        StinkyRegister a;             ///< Matrix A source\n";
        out << "        StinkyRegister b;             ///< Matrix B source\n";
        out << "        StinkyRegister acc2;          ///< Accumulator source\n";
        out << "        StinkyRegister mxsa;          ///< Scale factor A register\n";
        out << "        StinkyRegister mxsb;          ///< Scale factor B register\n";
        out << "        bool        reuseA;           ///< Matrix A reuse flag\n";
        out << "        bool        reuseB;           ///< Matrix B reuse flag\n";
        out << "\n";
        out << "        MXMFMA(const std::string& instType,\n";
        out << "               const std::string& accType,\n";
        out << "               const std::string& mxScaleATypeStr,\n";
        out << "               const std::string& mxScaleBTypeStr,\n";
        out << "               int m,\n";
        out << "               int n,\n";
        out << "               int k,\n";
        out << "               int block,\n";
        out << "               const StinkyRegister& acc,\n";
        out << "               const StinkyRegister& a,\n";
        out << "               const StinkyRegister& b,\n";
        out << "               const StinkyRegister& acc2,\n";
        out << "               const StinkyRegister& mxsa,\n";
        out << "               const StinkyRegister& mxsb,\n";
        out << "               bool reuseA = false,\n";
        out << "               bool reuseB = false,\n";
        out << "               const std::string& comment_ = \"\")\n";
        out << "            : IRInstruction(IRType::StinkyTofu)\n";
        out << "            , instType(instType)\n";
        out << "            , accType(accType)\n";
        out << "            , mxScaleATypeStr(mxScaleATypeStr)\n";
        out << "            , mxScaleBTypeStr(mxScaleBTypeStr)\n";
        out << "            , m(m)\n";
        out << "            , n(n)\n";
        out << "            , k(k)\n";
        out << "            , block(block)\n";
        out << "            , acc(acc)\n";
        out << "            , a(a)\n";
        out << "            , b(b)\n";
        out << "            , acc2(acc2)\n";
        out << "            , mxsa(mxsa)\n";
        out << "            , mxsb(mxsb)\n";
        out << "            , reuseA(reuseA)\n";
        out << "            , reuseB(reuseB)\n";
        out << "        {\n";
        out << "            this->comment = comment_;\n";
        out << "        }\n";
        out << "\n";
        out << "        const char* getLogicalName() const override { return \"MXMFMA\"; }\n";
        out << "\n";
        out << "        void dump(std::ostream& out) const override\n";
        out << "        {\n";
        out << "            out << \"MXMFMA (IR)\";\n";
        out << "            if(!comment.empty())\n";
        out << "                out << \"  // \" << comment;\n";
        out << "        }\n";
        out << "    };\n\n";

        // SMFMA class
        out << "    /**\n";
        out << "     * @brief SMFMA (Sparse MFMA) instruction\n";
        out << "     */\n";
        out << "    class SMFMA : public IRInstruction\n";
        out << "    {\n";
        out << "    public:\n";
        out << "        std::string instType;      ///< Input data type (bf16, f16, i8, etc.)\n";
        out << "        std::string accType;       ///< Accumulator type (f32, i32)\n";
        out << "        int         m;             ///< Matrix M dimension\n";
        out << "        int         n;             ///< Matrix N dimension\n";
        out << "        int         k;             ///< Matrix K dimension\n";
        out << "        int         blocks;        ///< Number of blocks\n";
        out << "        bool        mfma1k;        ///< Whether this is a _1k variant\n";
        out << "        StinkyRegister acc;        ///< Accumulator destination\n";
        out << "        StinkyRegister a;          ///< Matrix A source\n";
        out << "        StinkyRegister b;          ///< Matrix B source\n";
        out << "        StinkyRegister metadata;   ///< Sparsity metadata register\n";
        out << "        bool        neg;           ///< Negate operands\n";
        out << "\n";
        out << "        SMFMA(const std::string& instType,\n";
        out << "              const std::string& accType,\n";
        out << "              int m,\n";
        out << "              int n,\n";
        out << "              int k,\n";
        out << "              int blocks,\n";
        out << "              bool mfma1k,\n";
        out << "              const StinkyRegister& acc,\n";
        out << "              const StinkyRegister& a,\n";
        out << "              const StinkyRegister& b,\n";
        out << "              const StinkyRegister& metadata,\n";
        out << "              bool neg = false,\n";
        out << "              const std::string& comment_ = \"\")\n";
        out << "            : IRInstruction(IRType::StinkyTofu)\n";
        out << "            , instType(instType)\n";
        out << "            , accType(accType)\n";
        out << "            , m(m)\n";
        out << "            , n(n)\n";
        out << "            , k(k)\n";
        out << "            , blocks(blocks)\n";
        out << "            , mfma1k(mfma1k)\n";
        out << "            , acc(acc)\n";
        out << "            , a(a)\n";
        out << "            , b(b)\n";
        out << "            , metadata(metadata)\n";
        out << "            , neg(neg)\n";
        out << "        {\n";
        out << "            this->comment = comment_;\n";
        out << "        }\n";
        out << "\n";
        out << "        const char* getLogicalName() const override { return \"SMFMA\"; }\n";
        out << "\n";
        out << "        void dump(std::ostream& out) const override\n";
        out << "        {\n";
        out << "            out << \"SMFMA (IR)\";\n";
        out << "            if(!comment.empty())\n";
        out << "                out << \"  // \" << comment;\n";
        out << "        }\n";
        out << "    };\n\n";

        // TensorLoadToLds class
        out << "    // "
               "========================================================================\n";
        out << "    // Tensor Memory Instructions\n";
        out << "    // "
               "========================================================================\n\n";

        out << "    /**\n";
        out << "     * @brief TensorLoadToLds instruction\n";
        out << "     * \n";
        out << "     * Loads tensor data to LDS (Local Data Share). Takes 2-4 SGPR groups.\n";
        out << "     * All groups must be scalar registers (SGPRs).\n";
        out << "     */\n";
        out << "    class TensorLoadToLds : public IRInstruction\n";
        out << "    {\n";
        out << "    public:\n";
        out << "        TensorLoadToLds(const StinkyRegister& group0,\n";
        out << "                        const StinkyRegister& group1,\n";
        out << "                        const StinkyRegister* group2 = nullptr,\n";
        out << "                        const StinkyRegister* group3 = nullptr,\n";
        out << "                        const std::string& comment_ = \"\")\n";
        out << "            : IRInstruction(IRType::StinkyTofu)\n";
        out << "        {\n";
        out << "            // No destination register for this instruction\n";
        out << "            srcs.push_back(group0);\n";
        out << "            srcs.push_back(group1);\n";
        out << "            if (group2) srcs.push_back(*group2);\n";
        out << "            if (group3) srcs.push_back(*group3);\n";
        out << "            this->comment = comment_;\n";
        out << "        }\n";
        out << "\n";
        out << "        const char* getLogicalName() const override { return "
               "\"TensorLoadToLds\"; }\n";
        out << "\n";
        out << "        void dump(std::ostream& out) const override\n";
        out << "        {\n";
        out << "            out << \"TensorLoadToLds (IR)\";\n";
        out << "            if(!comment.empty())\n";
        out << "                out << \"  // \" << comment;\n";
        out << "        }\n";
        out << "    };\n\n";

        // Label class
        out << "    // "
               "========================================================================\n";
        out << "    // Control Flow Instructions\n";
        out << "    // "
               "========================================================================\n\n";

        out << "    /**\n";
        out << "     * @brief Label instruction for control flow\n";
        out << "     * \n";
        out << "     * Defines a label that can be used as a branch target.\n";
        out << "     * Labels have no operands and do not produce output.\n";
        out << "     */\n";
        out << "    class Label : public IRInstruction\n";
        out << "    {\n";
        out << "    public:\n";
        out << "        std::string label_name;\n";
        out << "\n";
        out << "        explicit Label(const std::string& name)\n";
        out << "            : IRInstruction(IRType::StinkyTofu)\n";
        out << "            , label_name(name)\n";
        out << "        {\n";
        out << "            // Labels have no operands\n";
        out << "        }\n";
        out << "\n";
        out << "        const char* getLogicalName() const override { return \"Label\"; }\n";
        out << "\n";
        out << "        void dump(std::ostream& out) const override\n";
        out << "        {\n";
        out << "            out << label_name << \":\";\n";
        out << "        }\n";
        out << "    };\n\n";

        return true;
    }

    // Generate IR instruction class definitions
    bool genIRClasses(const std::string& outdir)
    {
        std::ofstream out(outdir + "/ir/StinkyInstructions_generated.hpp");
        if(!out)
        {
            std::cerr << "Failed to open StinkyInstructions_generated.hpp for writing\n";
            return false;
        }

        // File header with header guards
        out << "/* ************************************************************************\n";
        out << " * Copyright (C) 2025-2026 Advanced Micro Devices, Inc.\n";
        out << " *\n";
        out << " * Permission is hereby granted, free of charge, to any person obtaining a copy\n";
        out << " * of this software and associated documentation files (the \"Software\"), to "
               "deal\n";
        out << " * in the Software without restriction, including without limitation the rights\n";
        out << " * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell\n";
        out << " * copies of the Software, and to permit persons to whom the Software is\n";
        out << " * furnished to do so, subject to the following conditions:\n";
        out << " *\n";
        out << " * The above copyright notice and this permission notice shall be included in\n";
        out << " * all copies or substantial portions of the Software.\n";
        out << " *\n";
        out << " * THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR\n";
        out << " * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,\n";
        out << " * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE\n";
        out << " * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER\n";
        out << " * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,\n";
        out << " * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN\n";
        out << " * THE SOFTWARE.\n";
        out << " *\n";
        out << " * ************************************************************************ */\n\n";
        out << "// Auto-generated by TableGen - DO NOT EDIT\n";
        out << "// This file contains high-level IR instruction class definitions\n\n";
        out << "#pragma once\n\n";
        out << "#include \"ir/asm/StinkyAsmIR.hpp\"\n";
        out << "#include \"ir/asm/StinkyModifiers.hpp\"\n";
        out << "#include \"stinkytofu.hpp\"\n";
        out << "#include <iostream>\n";
        out << "#include <optional>\n";
        out << "#include <string>\n";
        out << "\n// TODO: High-level IR should not depend on assembly-level IR "
               "(StinkyAsmIR.hpp).\n";
        out << "// Extract StinkyRegister and RegType into a separate header (e.g., "
               "ir/StinkyRegister.hpp)\n";
        out << "// to fix the inverted dependency. StinkyRegister is a shared primitive used by "
               "both\n";
        out << "// high-level IR (IRInstruction) and assembly IR (StinkyInstruction).\n";
        out << "#include <vector>\n\n";
        out << "namespace stinkytofu\n";
        out << "{\n\n";
        out << "    // NOTE: IRInstruction base class must be defined before including this "
               "file\n\n";

        std::string currentCategory = "";
        for(const auto& inst : getIRInstructions())
        {
            if(inst.category != currentCategory)
            {
                if(!currentCategory.empty())
                {
                    out << "\n";
                }
                out << "    // "
                       "========================================================================\n";
                out << "    // " << inst.category << "\n";
                out << "    // "
                       "========================================================================"
                       "\n\n";
                currentCategory = inst.category;
            }

            out << "    /**\n";
            out << "     * @brief " << inst.comment << "\n";
            out << "     */\n";
            out << "    class " << inst.className << " : public IRInstruction\n";
            out << "    {\n";
            out << "    public:\n";

            // Add modifier member variables based on instruction definition
            if(inst.supportsDPP || inst.supportsSDWA)
            {
                if(inst.supportsDPP)
                    out << "        std::optional<DPPModifiers>  dpp;  ///< Optional DPP "
                           "modifier\n";
                if(inst.supportsSDWA)
                    out << "        std::optional<SDWAModifiers> sdwa; ///< Optional SDWA "
                           "modifier\n";
                out << "\n";
            }
            if(inst.hasDS)
            {
                out << "        std::optional<DSModifiers> ds; ///< Optional DS modifier\n\n";
            }

            out << "        " << inst.className << "(";

            // Constructor parameters
            if(inst.hasDest)
            {
                out << "const StinkyRegister& dst";
                if(inst.numSrcs > 0)
                    out << ",\n" << std::string(inst.className.length() + 9, ' ');
            }

            for(int i = 0; i < inst.numSrcs; ++i)
            {
                if(i > 0 || inst.hasDest)
                {
                    out << "const StinkyRegister& src" << i;
                    if(i < inst.numSrcs - 1)
                        out << ",\n" << std::string(inst.className.length() + 9, ' ');
                }
                else
                {
                    out << "const StinkyRegister& src" << i;
                    if(i < inst.numSrcs - 1)
                        out << ",\n" << std::string(inst.className.length() + 9, ' ');
                }
            }

            if(inst.hasDest || inst.numSrcs > 0)
            {
                out << ",\n" << std::string(inst.className.length() + 9, ' ');
            }

            // Add modifier parameters based on instruction definition
            if(inst.supportsDPP)
            {
                out << "std::optional<DPPModifiers> dpp_ = std::nullopt,\n"
                    << std::string(inst.className.length() + 9, ' ');
            }
            if(inst.supportsSDWA)
            {
                out << "std::optional<SDWAModifiers> sdwa_ = std::nullopt,\n"
                    << std::string(inst.className.length() + 9, ' ');
            }
            if(inst.hasDS)
            {
                out << "std::optional<DSModifiers> ds_ = std::nullopt,\n"
                    << std::string(inst.className.length() + 9, ' ');
            }

            out << "const std::string& comment = \"\")\n";
            out << "            : IRInstruction(IRType::StinkyTofu)\n";

            // Add initializer list for modifiers
            if(inst.supportsDPP)
            {
                out << "            , dpp(dpp_)\n";
            }
            if(inst.supportsSDWA)
            {
                out << "            , sdwa(sdwa_)\n";
            }
            if(inst.hasDS)
            {
                out << "            , ds(ds_)\n";
            }

            out << "        {\n";

            // Constructor body
            if(inst.hasDest)
            {
                out << "            dests.push_back(dst);\n";
            }
            for(int i = 0; i < inst.numSrcs; ++i)
            {
                out << "            srcs.push_back(src" << i << ");\n";
            }
            out << "            this->comment = comment;\n";
            out << "        }\n\n";

            // getLogicalName method
            out << "        const char* getLogicalName() const override\n";
            out << "        {\n";
            out << "            return \"" << inst.className << "\";\n";
            out << "        }\n\n";

            // Modifier accessor overrides (only if instruction has modifiers)
            if(inst.supportsDPP)
            {
                out << "        std::optional<DPPModifiers> getDPP() const override\n";
                out << "        {\n";
                out << "            return dpp;\n";
                out << "        }\n\n";
            }
            if(inst.supportsSDWA)
            {
                out << "        std::optional<SDWAModifiers> getSDWA() const override\n";
                out << "        {\n";
                out << "            return sdwa;\n";
                out << "        }\n\n";
            }
            if(inst.hasDS)
            {
                out << "        std::optional<DSModifiers> getDS() const override\n";
                out << "        {\n";
                out << "            return ds;\n";
                out << "        }\n\n";
            }

            // dump method
            out << "        void dump(std::ostream& out) const override\n";
            out << "        {\n";
            out << "            out << \"" << inst.className << " (IR)\";\n";
            out << "            if(!comment.empty())\n";
            out << "                out << \"  // \" << comment;\n";
            out << "        }\n";
            out << "    };\n\n";
        }

        // Generate special MFMA classes
        genSpecialMFMAClasses(out);

        // Close namespace
        out << "\n} // namespace stinkytofu\n";

        std::cout << "Generated " << getIRInstructions().size()
                  << " IR instruction classes + 5 special classes "
                     "(MFMA/MXMFMA/SMFMA/TensorLoadToLds/Label) -> "
                     "StinkyInstructions_generated.hpp\n";
        return true;
    }

    // Generate builder method forward declarations
    bool genBuilderDecls(const std::string& outdir)
    {
        std::ofstream out(outdir + "/StinkyBuilder_decls_generated.inc");
        if(!out)
        {
            std::cerr << "Failed to open StinkyBuilder_decls_generated.inc for writing\n";
            return false;
        }

        out << "// Auto-generated by TableGen - DO NOT EDIT\n";
        out << "// Builder method declarations for IR instruction classes\n\n";

        std::string currentCategory = "";
        for(const auto& inst : getIRInstructions())
        {
            if(inst.category != currentCategory)
            {
                if(!currentCategory.empty())
                {
                    out << "\n";
                }
                out << "        // " << inst.category << "\n";
                currentCategory = inst.category;
            }

            // Generate method declaration with proper return type
            out << "        stinkytofu::" << inst.className << "* " << inst.className << "(";

            // Parameters
            bool hasParams = false;
            if(inst.hasDest)
            {
                out << "const StinkyRegister& dst";
                hasParams = true;
            }

            for(int i = 0; i < inst.numSrcs; ++i)
            {
                if(hasParams)
                    out << ",\n" << std::string(inst.className.length() + 10, ' ');
                out << "const StinkyRegister& src" << i;
                hasParams = true;
            }

            if(hasParams)
                out << ",\n" << std::string(inst.className.length() + 10, ' ');

            out << "const std::string& comment = \"\");\n\n";
        }

        std::cout << "Generated builder method declarations -> StinkyBuilder_decls_generated.inc\n";
        return true;
    }

    // Generate builder method implementations
    bool genBuilderImpls(const std::string& outdir)
    {
        std::ofstream out(outdir + "/StinkyBuilder_impls_generated.inc");
        if(!out)
        {
            std::cerr << "Failed to open StinkyBuilder_impls_generated.inc for writing\n";
            return false;
        }

        out << "// Auto-generated by TableGen - DO NOT EDIT\n";
        out << "// Builder method implementations for IR instruction classes\n\n";

        std::string currentCategory = "";
        for(const auto& inst : getIRInstructions())
        {
            if(inst.category != currentCategory)
            {
                if(!currentCategory.empty())
                {
                    out << "\n";
                }
                out << "    // "
                       "========================================================================\n";
                out << "    // " << inst.category << "\n";
                out << "    // "
                       "========================================================================"
                       "\n\n";
                currentCategory = inst.category;
            }

            // Method signature
            out << "    stinkytofu::" << inst.className << "* StinkyTofu::" << inst.className
                << "(";

            // Parameters
            if(inst.hasDest)
            {
                out << "const StinkyRegister& dst";
                if(inst.numSrcs > 0)
                    out << ",\n" << std::string(inst.className.length() + 26, ' ');
            }

            for(int i = 0; i < inst.numSrcs; ++i)
            {
                if(i > 0 || inst.hasDest)
                {
                    out << "const StinkyRegister& src" << i;
                    if(i < inst.numSrcs - 1)
                        out << ",\n" << std::string(inst.className.length() + 26, ' ');
                }
                else
                {
                    out << "const StinkyRegister& src" << i;
                    if(i < inst.numSrcs - 1)
                        out << ",\n" << std::string(inst.className.length() + 26, ' ');
                }
            }

            if(inst.hasDest || inst.numSrcs > 0)
            {
                out << ",\n" << std::string(inst.className.length() + 26, ' ');
            }

            out << "const std::string& comment)\n";
            out << "    {\n";

            // Method body - construct and return IR instruction
            out << "        return new stinkytofu::" << inst.className << "(";

            // Arguments to IR constructor
            if(inst.hasDest)
            {
                out << "dst";
                if(inst.numSrcs > 0)
                    out << ", ";
            }

            for(int i = 0; i < inst.numSrcs; ++i)
            {
                out << "src" << i;
                if(i < inst.numSrcs - 1)
                    out << ", ";
            }

            // Add std::nullopt for modifiers
            if(inst.numSrcs > 0 || inst.hasDest)
                out << ", ";

            if(inst.supportsDPP)
            {
                out << "std::nullopt, ";
            }
            if(inst.supportsSDWA)
            {
                out << "std::nullopt, ";
            }
            if(inst.hasDS)
            {
                out << "std::nullopt, ";
            }

            out << "comment);\n";
            out << "    }\n\n";
        }

        std::cout << "Generated " << getIRInstructions().size()
                  << " builder method implementations -> StinkyBuilder_impls_generated.inc\n";
        return true;
    }

    // Generate mnemonic mappings for ToStinkyAsmPass
    bool genMnemonicMappings(const std::string& outdir)
    {
        std::ofstream out(outdir + "/ir/IRMnemonics_generated.inc");
        if(!out)
        {
            std::cerr << "Failed to open IRMnemonics_generated.inc for writing\n";
            return false;
        }

        out << "// Auto-generated by TableGen - DO NOT EDIT\n";
        out << "// Logical IR name to assembly mnemonic mappings\n\n";

        int count = 0;

        // Regular instructions
        for(const auto& inst : getIRInstructions())
        {
            out << "            else if(std::string(logicalName) == \"" << inst.className
                << "\")\n";
            out << "            {\n";
            out << "                mnemonic = \"" << inst.mnemonic << "\";\n";
            out << "            }\n";
            count++;
        }

        // Note: Special instructions (MFMA/MXMFMA/SMFMA/TensorLoadToLds/Label)
        // generate their mnemonics dynamically, not through this mapping

        std::cout << "Generated " << count << " mnemonic mappings -> IRMnemonics_generated.inc\n";
        return true;
    }

    // Generate Python bindings for all IR instructions
    bool genPythonBindings(const std::string& outdir)
    {
        std::ofstream out(outdir + "/PythonBindings_generated.inc");
        if(!out)
        {
            std::cerr << "Failed to open PythonBindings_generated.inc for writing\n";
            return false;
        }

        out << "// Auto-generated Python bindings for IR instructions\n";
        out << "// DO NOT EDIT - Generated by GenHighLevelIR.cpp\n\n";

        int count = 0;

        // Generate bindings for all regular IR instructions
        for(const auto& inst : getIRInstructions())
        {
            std::string className = inst.className;

            // Generate factory function instead of constructor
            out << "    m.def(\"" << className << "\", [](";

            // Constructor parameter types
            std::vector<std::string> paramTypes;
            std::vector<std::string> paramNames;
            std::vector<std::string> defaults;

            // Add destination register (if instruction has one)
            if(inst.hasDest)
            {
                paramTypes.push_back("const StinkyRegister&");
                paramNames.push_back("dest");
                defaults.push_back("");
            }

            // Add source operands
            for(int i = 0; i < inst.numSrcs; i++)
            {
                paramTypes.push_back("const StinkyRegister&");
                paramNames.push_back("src" + std::to_string(i));
                defaults.push_back("");
            }

            // Add optional modifiers
            if(inst.supportsDPP && inst.supportsSDWA)
            {
                paramTypes.push_back("std::optional<DPPModifiers>");
                paramNames.push_back("dpp");
                defaults.push_back("std::nullopt");

                paramTypes.push_back("std::optional<SDWAModifiers>");
                paramNames.push_back("sdwa");
                defaults.push_back("std::nullopt");
            }
            else if(inst.hasDS)
            {
                paramTypes.push_back("std::optional<DSModifiers>");
                paramNames.push_back("ds");
                defaults.push_back("std::nullopt");
            }

            // Add comment parameter
            paramTypes.push_back("const std::string&");
            paramNames.push_back("comment");
            defaults.push_back("\"\"");

            // Output lambda parameter list
            for(size_t i = 0; i < paramTypes.size(); i++)
            {
                out << paramTypes[i] << " " << paramNames[i];
                if(i + 1 < paramTypes.size())
                    out << ", ";
            }
            out << ") {\n";

            // Return std::make_shared as base class pointer
            out << "        return std::shared_ptr<IRInstruction>(std::make_shared<" << className
                << ">(";
            for(size_t i = 0; i < paramNames.size(); i++)
            {
                out << paramNames[i];
                if(i + 1 < paramNames.size())
                    out << ", ";
            }
            out << "));\n";
            out << "    },\n";

            // Output parameter names and defaults
            for(size_t i = 0; i < paramNames.size(); i++)
            {
                out << "    nb::arg(\"" << paramNames[i] << "\")";
                if(!defaults[i].empty())
                    out << " = " << defaults[i];
                if(i + 1 < paramNames.size())
                    out << ",\n";
            }
            out << ",\n";

            // Add docstring
            out << "    \"Create a " << inst.mnemonic << " instruction\");\n\n";
            count++;
        }

        std::cout << "Generated Python bindings for " << count
                  << " IR instructions -> PythonBindings_generated.inc\n";
        return true;
    }

    // Generate all high-level IR artifacts
    bool genHighLevelIR(const std::string& outdir)
    {
        bool success = true;

        std::cout << "\n=== Generating High-Level IR ===\n";

        success &= genIRClasses(outdir);
        success &= genBuilderDecls(outdir);
        success &= genBuilderImpls(outdir);
        success &= genMnemonicMappings(outdir);
        success &= genPythonBindings(outdir);

        if(success)
        {
            std::cout << "=== High-Level IR generation completed successfully ===\n\n";
        }
        else
        {
            std::cerr << "=== High-Level IR generation failed ===\n\n";
        }

        return success;
    }

} // namespace stinkytofu
