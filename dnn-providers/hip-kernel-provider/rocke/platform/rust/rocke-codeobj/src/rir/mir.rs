// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

//! Machine IR for gfx11: virtual registers, gfx11 opcodes with their
//! encoding facts, and basic blocks.

use std::fmt::Write;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Class {
    S,
    V,
}

#[derive(Clone, Copy, Debug)]
pub struct VRegInfo {
    pub class: Class,
    pub size: u8,
    pub align: u8,
    /// Precolored physical register (hardware-initialized inputs).
    pub fixed: Option<u16>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct VRef {
    pub reg: u32,
    pub off: u8,
    pub n: u8,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Phys {
    S(u16),
    V(u16),
    VccLo,
    ExecLo,
    M0,
    Null,
    Scc,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Opnd {
    R(VRef),
    P(Phys, u8),
    Imm(u32),
}

impl Opnd {
    pub fn vref(&self) -> Option<VRef> {
        match self {
            Opnd::R(v) => Some(*v),
            _ => None,
        }
    }

    pub fn dwords(&self) -> u8 {
        match self {
            Opnd::R(v) => v.n,
            Opnd::P(_, n) => *n,
            Opnd::Imm(_) => 1,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fmt {
    Sop1,
    Sop2,
    Sopc,
    Sopk,
    Sopp,
    Smem,
    Vop1,
    Vop2,
    /// Native VOP3 (no VOP1/VOP2 short form).
    Vop3,
    /// VOP3 with a scalar destination (carry, div_scale flag) in bits 14:8.
    Vop3b,
    /// Compare, always encoded as VOP3 with an SGPR mask destination.
    Vopc,
    Vop3p,
    /// Two VALU operations issued together (`v_dual_<x> :: v_dual_<y>`).
    Vopd,
    Ds,
    Global,
    Mubuf,
    Pseudo,
}

pub const F_DEF_SCC: u32 = 1 << 0;
pub const F_USE_SCC: u32 = 1 << 1;
pub const F_USE_VCC: u32 = 1 << 2;
pub const F_DEF_EXEC: u32 = 1 << 3;
pub const F_USE_EXEC: u32 = 1 << 4;
pub const F_VMEM_LOAD: u32 = 1 << 5;
pub const F_VMEM_STORE: u32 = 1 << 6;
pub const F_LDS: u32 = 1 << 7;
pub const F_SMEM: u32 = 1 << 8;
pub const F_BRANCH: u32 = 1 << 9;
pub const F_TRANS: u32 = 1 << 10;
pub const F_WMMA: u32 = 1 << 11;
/// The destination is also read (true16 half writes, fmac, d16 loads).
pub const F_TIED: u32 = 1 << 12;
pub const F_64BIT_SRC: u32 = 1 << 13;
pub const F_16BIT_DST: u32 = 1 << 14;
pub const F_16BIT_SRC: u32 = 1 << 15;
pub const F_END: u32 = 1 << 16;
/// The destination is written before the sources are fully read.
pub const F_EARLY: u32 = 1 << 17;

pub const M_NEG0: u32 = 1 << 0;
pub const M_NEG1: u32 = 1 << 1;
pub const M_NEG2: u32 = 1 << 2;
/// VOP3 op_sel bits: source 0/1/2 high half, destination high half.
pub const M_OPSEL_SRC0: u32 = 1 << 4;
pub const M_OPSEL_SRC1: u32 = 1 << 5;
pub const M_OPSEL_SRC2: u32 = 1 << 6;
pub const M_OPSEL_DST: u32 = 1 << 7;
/// Global/buffer access through saddr (SGPR base + VGPR 32-bit offset).
pub const M_SADDR: u32 = 1 << 8;
/// `v_fma_mix_f32` source 0/1/2 is f16 (VOP3P op_sel_hi); M_OPSEL_SRCn then
/// picks its high half.
pub const M_MIX_F16_0: u32 = 1 << 9;

macro_rules! opcodes {
    ($($name:ident = ($fmt:ident, $code:expr, $mn:expr, $flags:expr)),* $(,)?) => {
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
        pub enum Opc { $($name),*, VCmp(u8) }

        impl Opc {
            #[inline(always)]
            pub fn fmt(self) -> Fmt {
                match self { $(Opc::$name => Fmt::$fmt),*, Opc::VCmp(_) => Fmt::Vopc }
            }
            #[inline(always)]
            pub fn code(self) -> u16 {
                match self { $(Opc::$name => $code),*, Opc::VCmp(op) => op as u16 }
            }
            pub fn name(self) -> String {
                match self { $(Opc::$name => $mn.to_string()),*, Opc::VCmp(op) => vcmp_name(op) }
            }
            #[inline(always)]
            pub fn flags(self) -> u32 {
                match self {
                    $(Opc::$name => $flags),*,
                    Opc::VCmp(op) => F_USE_EXEC | if (op & 0xf0) == 0x50 { F_64BIT_SRC } else { 0 },
                }
            }
        }
    };
}

const VALU: u32 = F_USE_EXEC;

opcodes! {
    // SOP1
    SMovB32 = (Sop1, 0x00, "s_mov_b32", 0),
    SMovB64 = (Sop1, 0x01, "s_mov_b64", 0),
    SNotB32 = (Sop1, 0x1e, "s_not_b32", F_DEF_SCC),
    SAndSaveexecB32 = (Sop1, 0x20, "s_and_saveexec_b32", F_DEF_SCC | F_DEF_EXEC | F_USE_EXEC),
    // SOP2
    SAddU32 = (Sop2, 0, "s_add_u32", F_DEF_SCC),
    SSubU32 = (Sop2, 1, "s_sub_u32", F_DEF_SCC),
    SAddI32 = (Sop2, 2, "s_add_i32", F_DEF_SCC),
    SSubI32 = (Sop2, 3, "s_sub_i32", F_DEF_SCC),
    SAddcU32 = (Sop2, 4, "s_addc_u32", F_DEF_SCC | F_USE_SCC),
    SSubbU32 = (Sop2, 5, "s_subb_u32", F_DEF_SCC | F_USE_SCC),
    SLshlB32 = (Sop2, 8, "s_lshl_b32", F_DEF_SCC),
    SLshlB64 = (Sop2, 9, "s_lshl_b64", F_DEF_SCC),
    SLshrB32 = (Sop2, 10, "s_lshr_b32", F_DEF_SCC),
    SLshrB64 = (Sop2, 11, "s_lshr_b64", F_DEF_SCC),
    SAshrI32 = (Sop2, 12, "s_ashr_i32", F_DEF_SCC),
    SAshrI64 = (Sop2, 13, "s_ashr_i64", F_DEF_SCC),
    SLshl1AddU32 = (Sop2, 14, "s_lshl1_add_u32", F_DEF_SCC),
    SLshl2AddU32 = (Sop2, 15, "s_lshl2_add_u32", F_DEF_SCC),
    SLshl3AddU32 = (Sop2, 16, "s_lshl3_add_u32", F_DEF_SCC),
    SLshl4AddU32 = (Sop2, 17, "s_lshl4_add_u32", F_DEF_SCC),
    SMinI32 = (Sop2, 18, "s_min_i32", F_DEF_SCC),
    SMinU32 = (Sop2, 19, "s_min_u32", F_DEF_SCC),
    SMaxI32 = (Sop2, 20, "s_max_i32", F_DEF_SCC),
    SMaxU32 = (Sop2, 21, "s_max_u32", F_DEF_SCC),
    SAndB32 = (Sop2, 22, "s_and_b32", F_DEF_SCC),
    SAndB64 = (Sop2, 23, "s_and_b64", F_DEF_SCC),
    SOrB32 = (Sop2, 24, "s_or_b32", F_DEF_SCC),
    SOrB64 = (Sop2, 25, "s_or_b64", F_DEF_SCC),
    SXorB32 = (Sop2, 26, "s_xor_b32", F_DEF_SCC),
    SXorB64 = (Sop2, 27, "s_xor_b64", F_DEF_SCC),
    SAndNot1B32 = (Sop2, 34, "s_and_not1_b32", F_DEF_SCC),
    SOrNot1B32 = (Sop2, 36, "s_or_not1_b32", F_DEF_SCC),
    SBfeU32 = (Sop2, 38, "s_bfe_u32", F_DEF_SCC),
    SMulI32 = (Sop2, 44, "s_mul_i32", 0),
    SMulHiU32 = (Sop2, 45, "s_mul_hi_u32", 0),
    SMulHiI32 = (Sop2, 46, "s_mul_hi_i32", 0),
    SCselectB32 = (Sop2, 48, "s_cselect_b32", F_USE_SCC),
    SCselectB64 = (Sop2, 49, "s_cselect_b64", F_USE_SCC),
    // SOPC
    SCmpEqI32 = (Sopc, 0x00, "s_cmp_eq_i32", F_DEF_SCC),
    SCmpLgI32 = (Sopc, 0x01, "s_cmp_lg_i32", F_DEF_SCC),
    SCmpGtI32 = (Sopc, 0x02, "s_cmp_gt_i32", F_DEF_SCC),
    SCmpGeI32 = (Sopc, 0x03, "s_cmp_ge_i32", F_DEF_SCC),
    SCmpLtI32 = (Sopc, 0x04, "s_cmp_lt_i32", F_DEF_SCC),
    SCmpLeI32 = (Sopc, 0x05, "s_cmp_le_i32", F_DEF_SCC),
    SCmpEqU32 = (Sopc, 0x06, "s_cmp_eq_u32", F_DEF_SCC),
    SCmpLgU32 = (Sopc, 0x07, "s_cmp_lg_u32", F_DEF_SCC),
    SCmpGtU32 = (Sopc, 0x08, "s_cmp_gt_u32", F_DEF_SCC),
    SCmpGeU32 = (Sopc, 0x09, "s_cmp_ge_u32", F_DEF_SCC),
    SCmpLtU32 = (Sopc, 0x0a, "s_cmp_lt_u32", F_DEF_SCC),
    SCmpLeU32 = (Sopc, 0x0b, "s_cmp_le_u32", F_DEF_SCC),
    SCmpEqU64 = (Sopc, 0x10, "s_cmp_eq_u64", F_DEF_SCC),
    SCmpLgU64 = (Sopc, 0x11, "s_cmp_lg_u64", F_DEF_SCC),
    // SOPK
    SMovkI32 = (Sopk, 0x00, "s_movk_i32", 0),
    // SOPP
    SNop = (Sopp, 0x00, "s_nop", 0),
    SClause = (Sopp, 0x05, "s_clause", 0),
    SDelayAlu = (Sopp, 0x07, "s_delay_alu", 0),
    SWaitcnt = (Sopp, 0x09, "s_waitcnt", 0),
    SCodeEnd = (Sopp, 0x1f, "s_code_end", 0),
    SBranch = (Sopp, 0x20, "s_branch", F_BRANCH),
    SCbranchScc0 = (Sopp, 0x21, "s_cbranch_scc0", F_BRANCH | F_USE_SCC),
    SCbranchScc1 = (Sopp, 0x22, "s_cbranch_scc1", F_BRANCH | F_USE_SCC),
    SCbranchExecz = (Sopp, 0x25, "s_cbranch_execz", F_BRANCH | F_USE_EXEC),
    SCbranchExecnz = (Sopp, 0x26, "s_cbranch_execnz", F_BRANCH | F_USE_EXEC),
    SEndpgm = (Sopp, 0x30, "s_endpgm", F_END),
    SSendmsg = (Sopp, 0x36, "s_sendmsg", 0),
    SBarrier = (Sopp, 0x3d, "s_barrier", 0),
    // SMEM
    SLoadB32 = (Smem, 0, "s_load_b32", F_SMEM),
    SLoadB64 = (Smem, 1, "s_load_b64", F_SMEM),
    SLoadB128 = (Smem, 2, "s_load_b128", F_SMEM),
    SLoadB256 = (Smem, 3, "s_load_b256", F_SMEM),
    SLoadB512 = (Smem, 4, "s_load_b512", F_SMEM),
    // VOP1
    VNop = (Vop1, 0x00, "v_nop", VALU),
    VMovB32 = (Vop1, 0x01, "v_mov_b32", VALU),
    VReadfirstlaneB32 = (Vop1, 0x02, "v_readfirstlane_b32", VALU),
    VCvtF32I32 = (Vop1, 0x05, "v_cvt_f32_i32", VALU),
    VCvtF32U32 = (Vop1, 0x06, "v_cvt_f32_u32", VALU),
    VCvtU32F32 = (Vop1, 0x07, "v_cvt_u32_f32", VALU),
    VCvtI32F32 = (Vop1, 0x08, "v_cvt_i32_f32", VALU),
    VCvtF16F32 = (Vop1, 0x0a, "v_cvt_f16_f32", VALU | F_16BIT_DST),
    VCvtF32F16 = (Vop1, 0x0b, "v_cvt_f32_f16", VALU | F_16BIT_SRC),
    VMovB16 = (Vop1, 0x1c, "v_mov_b16", VALU | F_16BIT_DST | F_16BIT_SRC),
    VExpF32 = (Vop1, 0x25, "v_exp_f32", VALU | F_TRANS),
    VLogF32 = (Vop1, 0x27, "v_log_f32", VALU | F_TRANS),
    VRcpF32 = (Vop1, 0x2a, "v_rcp_f32", VALU | F_TRANS),
    VRcpIflagF32 = (Vop1, 0x2b, "v_rcp_iflag_f32", VALU | F_TRANS),
    VNotB32 = (Vop1, 0x37, "v_not_b32", VALU),
    // VOP2
    VCndmaskB32 = (Vop2, 0x01, "v_cndmask_b32", VALU),
    VAddF32 = (Vop2, 0x03, "v_add_f32", VALU),
    VSubF32 = (Vop2, 0x04, "v_sub_f32", VALU),
    VSubrevF32 = (Vop2, 0x05, "v_subrev_f32", VALU),
    VMulF32 = (Vop2, 0x08, "v_mul_f32", VALU),
    VMulI32I24 = (Vop2, 0x09, "v_mul_i32_i24", VALU),
    VMulU32U24 = (Vop2, 0x0b, "v_mul_u32_u24", VALU),
    VMulHiU32U24 = (Vop2, 0x0c, "v_mul_hi_u32_u24", VALU),
    VMinF32 = (Vop2, 0x0f, "v_min_f32", VALU),
    VMaxF32 = (Vop2, 0x10, "v_max_f32", VALU),
    VMinI32 = (Vop2, 0x11, "v_min_i32", VALU),
    VMaxI32 = (Vop2, 0x12, "v_max_i32", VALU),
    VMinU32 = (Vop2, 0x13, "v_min_u32", VALU),
    VMaxU32 = (Vop2, 0x14, "v_max_u32", VALU),
    VLshlrevB32 = (Vop2, 0x18, "v_lshlrev_b32", VALU),
    VLshrrevB32 = (Vop2, 0x19, "v_lshrrev_b32", VALU),
    VAshrrevI32 = (Vop2, 0x1a, "v_ashrrev_i32", VALU),
    VAndB32 = (Vop2, 0x1b, "v_and_b32", VALU),
    VOrB32 = (Vop2, 0x1c, "v_or_b32", VALU),
    VXorB32 = (Vop2, 0x1d, "v_xor_b32", VALU),
    VAddCoCiU32 = (Vop3b, 0x120, "v_add_co_ci_u32", VALU),
    VSubCoCiU32 = (Vop3b, 0x121, "v_sub_co_ci_u32", VALU),
    VAddNcU32 = (Vop2, 0x25, "v_add_nc_u32", VALU),
    VSubNcU32 = (Vop2, 0x26, "v_sub_nc_u32", VALU),
    VSubrevNcU32 = (Vop2, 0x27, "v_subrev_nc_u32", VALU),
    VFmacF32 = (Vop2, 0x2b, "v_fmac_f32", VALU | F_TIED),
    // VOP3
    VMadU32U24 = (Vop3, 0x20b, "v_mad_u32_u24", VALU),
    VBfeU32 = (Vop3, 0x210, "v_bfe_u32", VALU),
    VBfeI32 = (Vop3, 0x211, "v_bfe_i32", VALU),
    VFmaF32 = (Vop3, 0x213, "v_fma_f32", VALU),
    VMax3F32 = (Vop3, 0x21c, "v_max3_f32", VALU),
    VMed3I32 = (Vop3, 0x220, "v_med3_i32", VALU),
    VDivFixupF32 = (Vop3, 0x227, "v_div_fixup_f32", VALU),
    VDivFmasF32 = (Vop3, 0x237, "v_div_fmas_f32", VALU | F_USE_VCC),
    VPermB32 = (Vop3, 0x244, "v_perm_b32", VALU),
    VLshlAddU32 = (Vop3, 0x246, "v_lshl_add_u32", VALU),
    VAddLshlU32 = (Vop3, 0x247, "v_add_lshl_u32", VALU),
    VAdd3U32 = (Vop3, 0x255, "v_add3_u32", VALU),
    VLshlOrB32 = (Vop3, 0x256, "v_lshl_or_b32", VALU),
    VAndOrB32 = (Vop3, 0x257, "v_and_or_b32", VALU),
    VOr3B32 = (Vop3, 0x258, "v_or3_b32", VALU),
    VMaxminI32 = (Vop3, 0x264, "v_maxmin_i32", VALU),
    VMinmaxI32 = (Vop3, 0x265, "v_minmax_i32", VALU),
    VDivScaleF32 = (Vop3b, 0x2fc, "v_div_scale_f32", VALU),
    VMadU64U32 = (Vop3b, 0x2fe, "v_mad_u64_u32", VALU | F_64BIT_SRC),
    VMadI64I32 = (Vop3b, 0x2ff, "v_mad_i64_i32", VALU | F_64BIT_SRC),
    VAddCoU32 = (Vop3b, 0x300, "v_add_co_u32", VALU),
    VSubCoU32 = (Vop3b, 0x301, "v_sub_co_u32", VALU),
    VSubrevCoU32 = (Vop3b, 0x302, "v_subrev_co_u32", VALU),
    VPackB32F16 = (Vop3, 0x311, "v_pack_b32_f16", VALU),
    VLdexpF32 = (Vop3, 0x31c, "v_ldexp_f32", VALU),
    VMulLoU32 = (Vop3, 0x32c, "v_mul_lo_u32", VALU),
    VMulHiU32 = (Vop3, 0x32d, "v_mul_hi_u32", VALU),
    VMulHiI32 = (Vop3, 0x32e, "v_mul_hi_i32", VALU),
    VLshlrevB64 = (Vop3, 0x33c, "v_lshlrev_b64", VALU | F_64BIT_SRC),
    VLshrrevB64 = (Vop3, 0x33d, "v_lshrrev_b64", VALU | F_64BIT_SRC),
    VAshrrevI64 = (Vop3, 0x33e, "v_ashrrev_i64", VALU | F_64BIT_SRC),
    // VOP3P
    VFmaMixF32 = (Vop3p, 0x20, "v_fma_mix_f32", VALU),
    VWmmaF32F16 = (Vop3p, 0x40, "v_wmma_f32_16x16x16_f16", VALU | F_WMMA),
    // Three-address form: D must not overlap A, B or C.
    VWmmaF32F16Early = (Vop3p, 0x40, "v_wmma_f32_16x16x16_f16", VALU | F_WMMA | F_EARLY),
    // VOPD: defs [x, y]; the first `mods` uses are x's sources, the rest y's;
    // `imm` holds the x and y opcodes in bits 7:0 and 15:8.
    VDual = (Vopd, 0, "v_dual", VALU),
    // DS
    DsStoreB32 = (Ds, 0x0d, "ds_store_b32", F_LDS),
    DsStoreB16 = (Ds, 0x1f, "ds_store_b16", F_LDS),
    DsSwizzleB32 = (Ds, 0x35, "ds_swizzle_b32", F_LDS | F_USE_EXEC),
    DsLoadB32 = (Ds, 0x36, "ds_load_b32", F_LDS),
    DsLoadU16 = (Ds, 0x3c, "ds_load_u16", F_LDS),
    DsStoreB64 = (Ds, 0x4d, "ds_store_b64", F_LDS),
    DsLoadB64 = (Ds, 0x76, "ds_load_b64", F_LDS),
    DsStoreB16D16Hi = (Ds, 0xa1, "ds_store_b16_d16_hi", F_LDS),
    DsLoadU16D16 = (Ds, 0xa6, "ds_load_u16_d16", F_LDS | F_TIED),
    DsLoadU16D16Hi = (Ds, 0xa7, "ds_load_u16_d16_hi", F_LDS | F_TIED),
    DsStoreB128 = (Ds, 0xdf, "ds_store_b128", F_LDS),
    DsLoadB128 = (Ds, 0xff, "ds_load_b128", F_LDS),
    // GLOBAL
    GLoadU16 = (Global, 0x12, "global_load_u16", F_VMEM_LOAD),
    GLoadB32 = (Global, 0x14, "global_load_b32", F_VMEM_LOAD),
    GLoadB64 = (Global, 0x15, "global_load_b64", F_VMEM_LOAD),
    GLoadB96 = (Global, 0x16, "global_load_b96", F_VMEM_LOAD),
    GLoadB128 = (Global, 0x17, "global_load_b128", F_VMEM_LOAD),
    GStoreB16 = (Global, 0x19, "global_store_b16", F_VMEM_STORE),
    GStoreB32 = (Global, 0x1a, "global_store_b32", F_VMEM_STORE),
    GStoreB64 = (Global, 0x1b, "global_store_b64", F_VMEM_STORE),
    GStoreB96 = (Global, 0x1c, "global_store_b96", F_VMEM_STORE),
    GStoreB128 = (Global, 0x1d, "global_store_b128", F_VMEM_STORE),
    GLoadD16B16 = (Global, 0x20, "global_load_d16_b16", F_VMEM_LOAD | F_TIED),
    GLoadD16HiB16 = (Global, 0x23, "global_load_d16_hi_b16", F_VMEM_LOAD | F_TIED),
    GStoreD16HiB16 = (Global, 0x25, "global_store_d16_hi_b16", F_VMEM_STORE),
    GAtomicAddF32 = (Global, 0x56, "global_atomic_add_f32", F_VMEM_STORE),
    // MUBUF
    BufLoadB32 = (Mubuf, 0x14, "buffer_load_b32", F_VMEM_LOAD),
    BufLoadB64 = (Mubuf, 0x15, "buffer_load_b64", F_VMEM_LOAD),
    BufLoadB128 = (Mubuf, 0x17, "buffer_load_b128", F_VMEM_LOAD),
    BufStoreB16 = (Mubuf, 0x19, "buffer_store_b16", F_VMEM_STORE),
    BufStoreB32 = (Mubuf, 0x1a, "buffer_store_b32", F_VMEM_STORE),
    BufStoreB64 = (Mubuf, 0x1b, "buffer_store_b64", F_VMEM_STORE),
    BufStoreB128 = (Mubuf, 0x1d, "buffer_store_b128", F_VMEM_STORE),
    // pseudo
    Copy = (Pseudo, 0, "COPY", 0),
    // Defines a register whose contents are never read before overwritten.
    ImplicitDef = (Pseudo, 0, "IMPLICIT_DEF", 0),
    // dst <- {cvt_f16(src0), cvt_f16(src1)}; dst must not overlap the sources.
    PackF16 = (Pseudo, 0, "PACK_F16", VALU | F_EARLY),
}

/// `v_cmp_class_f32`: whether src0's class is in the mask src1.
pub const VCMP_CLASS_F32: u8 = 0x7e;

pub fn vcmp_name(op: u8) -> String {
    if op == VCMP_CLASS_F32 {
        return "v_cmp_class_f32".into();
    }
    let (ty, base) = match op & 0xf8 {
        0x10 | 0x18 => ("f32", 0x10),
        0x40 => ("i32", 0x40),
        0x48 => ("u32", 0x48),
        0x50 => ("i64", 0x50),
        0x58 => ("u64", 0x58),
        _ => ("?", op),
    };
    let pred = if ty == "f32" {
        ["f", "lt", "eq", "le", "gt", "lg", "ge", "o", "u", "nge", "nlg", "ngt", "nle", "neq", "nlt", "t"]
            .get((op - base) as usize)
            .copied()
            .unwrap_or("?")
    } else {
        ["f", "lt", "eq", "le", "gt", "ne", "ge", "t"]
            .get((op - base) as usize)
            .copied()
            .unwrap_or("?")
    };
    format!("v_cmp_{pred}_{ty}")
}

/// Mnemonic of a VOPD component opcode.
pub fn vopd_name(code: u8) -> &'static str {
    match code {
        3 => "v_dual_mul_f32",
        4 => "v_dual_add_f32",
        5 => "v_dual_sub_f32",
        6 => "v_dual_subrev_f32",
        8 => "v_dual_mov_b32",
        10 => "v_dual_max_f32",
        11 => "v_dual_min_f32",
        16 => "v_dual_add_nc_u32",
        17 => "v_dual_lshlrev_b32",
        18 => "v_dual_and_b32",
        _ => "v_dual_?",
    }
}

/// Up to `N` operands, stored inline.
#[derive(Clone, Copy)]
pub struct Opnds<const N: usize> {
    len: u8,
    items: [Opnd; N],
}

impl<const N: usize> Opnds<N> {
    pub fn new() -> Self {
        Opnds { len: 0, items: [Opnd::Imm(0); N] }
    }

    /// Appends `o`; panics when all `N` slots are taken.
    pub fn push(&mut self, o: Opnd) {
        self.items[self.len as usize] = o;
        self.len += 1;
    }
}

impl<const N: usize> Default for Opnds<N> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const N: usize> std::ops::Deref for Opnds<N> {
    type Target = [Opnd];
    fn deref(&self) -> &[Opnd] {
        &self.items[..self.len as usize]
    }
}

impl<const N: usize> std::ops::DerefMut for Opnds<N> {
    fn deref_mut(&mut self) -> &mut [Opnd] {
        &mut self.items[..self.len as usize]
    }
}

impl<'a, const N: usize> IntoIterator for &'a Opnds<N> {
    type Item = &'a Opnd;
    type IntoIter = std::slice::Iter<'a, Opnd>;
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, const N: usize> IntoIterator for &'a mut Opnds<N> {
    type Item = &'a mut Opnd;
    type IntoIter = std::slice::IterMut<'a, Opnd>;
    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

impl<const N: usize> From<&[Opnd]> for Opnds<N> {
    fn from(opnds: &[Opnd]) -> Self {
        let mut out = Self::new();
        for &o in opnds {
            out.push(o);
        }
        out
    }
}

impl<const K: usize, const N: usize> From<[Opnd; K]> for Opnds<N> {
    fn from(opnds: [Opnd; K]) -> Self {
        Self::from(&opnds[..])
    }
}

impl<const N: usize> PartialEq for Opnds<N> {
    fn eq(&self, other: &Self) -> bool {
        **self == **other
    }
}

impl<const N: usize> Eq for Opnds<N> {}

impl<const N: usize> std::fmt::Debug for Opnds<N> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_list().entries(self.iter()).finish()
    }
}

pub type Defs = Opnds<2>;
pub type Uses = Opnds<4>;

#[derive(Clone, Debug)]
pub struct MInst {
    pub op: Opc,
    pub defs: Defs,
    pub uses: Uses,
    /// Memory offset, SOPP immediate, or branch target block.
    pub imm: i64,
    pub mods: u32,
}

impl MInst {
    pub fn new(op: Opc, defs: impl Into<Defs>, uses: impl Into<Uses>) -> Self {
        Self {
            op,
            defs: defs.into(),
            uses: uses.into(),
            imm: 0,
            mods: 0,
        }
    }

    pub fn with_imm(mut self, imm: i64) -> Self {
        self.imm = imm;
        self
    }

    pub fn with_mods(mut self, mods: u32) -> Self {
        self.mods |= mods;
        self
    }

    pub fn is_branch(&self) -> bool {
        self.op.flags() & F_BRANCH != 0
    }
}

#[derive(Clone, Debug, Default)]
pub struct MBlock {
    pub insts: Vec<MInst>,
    pub loop_depth: u32,
}

#[derive(Clone, Debug, Default)]
pub struct MFunc {
    pub blocks: Vec<MBlock>,
    pub vregs: Vec<VRegInfo>,
}

impl MFunc {
    pub fn successors(&self, block: usize) -> Vec<usize> {
        let mut out = Vec::new();
        let insts = &self.blocks[block].insts;
        let mut falls_through = true;
        if let Some(last) = insts.last() {
            if last.op == Opc::SEndpgm {
                falls_through = false;
            }
            if last.op == Opc::SBranch {
                falls_through = false;
            }
        }
        for inst in insts.iter().rev().take(2) {
            if inst.is_branch() {
                out.push(inst.imm as usize);
            }
        }
        if falls_through && block + 1 < self.blocks.len() {
            out.push(block + 1);
        }
        out.sort_unstable();
        out.dedup();
        out
    }
}

pub fn phys_name(phys: Phys, n: u8) -> String {
    match phys {
        Phys::S(i) if n == 1 => format!("s{i}"),
        Phys::S(i) => format!("s[{}:{}]", i, i + n as u16 - 1),
        Phys::V(i) if n == 1 => format!("v{i}"),
        Phys::V(i) => format!("v[{}:{}]", i, i + n as u16 - 1),
        Phys::VccLo => "vcc_lo".into(),
        Phys::ExecLo => "exec_lo".into(),
        Phys::M0 => "m0".into(),
        Phys::Null => "null".into(),
        Phys::Scc => "scc".into(),
    }
}

pub fn opnd_name(func: &MFunc, opnd: &Opnd) -> String {
    match opnd {
        Opnd::R(v) => {
            let info = func.vregs[v.reg as usize];
            let prefix = if info.class == Class::S { "%s" } else { "%v" };
            if v.off == 0 && v.n == info.size {
                format!("{prefix}{}", v.reg)
            } else {
                format!("{prefix}{}[{}:{}]", v.reg, v.off, v.off + v.n - 1)
            }
        }
        Opnd::P(p, n) => phys_name(*p, *n),
        Opnd::Imm(v) => {
            let signed = *v as i32;
            if (-16..=64).contains(&signed) {
                signed.to_string()
            } else {
                format!("0x{v:x}")
            }
        }
    }
}

pub fn inst_text(func: &MFunc, inst: &MInst) -> String {
    if inst.op == Opc::VDual {
        let names = |defs: &[Opnd], uses: &[Opnd]| -> Vec<String> {
            defs.iter().chain(uses).map(|o| opnd_name(func, o)).collect()
        };
        let nx = inst.mods as usize;
        let x = names(&inst.defs[..1], &inst.uses[..nx]).join(", ");
        let y = names(&inst.defs[1..], &inst.uses[nx..]).join(", ");
        let (cx, cy) = (inst.imm as u8, (inst.imm >> 8) as u8);
        return format!("{} {x} :: {} {y}", vopd_name(cx), vopd_name(cy));
    }
    let mut text = inst.op.name();
    let mut parts: Vec<String> = inst.defs.iter().map(|o| opnd_name(func, o)).collect();
    parts.extend(inst.uses.iter().map(|o| opnd_name(func, o)));
    if !parts.is_empty() {
        text.push(' ');
        text.push_str(&parts.join(", "));
    }
    if inst.imm != 0 || inst.is_branch() {
        let _ = write!(text, " imm:{}", inst.imm);
    }
    if inst.mods != 0 {
        let _ = write!(text, " mods:0x{:x}", inst.mods);
    }
    text
}

pub fn dump(func: &MFunc) -> String {
    let mut out = String::new();
    for (index, block) in func.blocks.iter().enumerate() {
        let _ = writeln!(out, "bb{index} (depth {}) -> {:?}", block.loop_depth, func.successors(index));
        for inst in &block.insts {
            let _ = writeln!(out, "    {}", inst_text(func, inst));
        }
    }
    out
}
