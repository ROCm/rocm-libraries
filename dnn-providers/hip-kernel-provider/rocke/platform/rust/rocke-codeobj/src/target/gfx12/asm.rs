// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

//! What gfx12 encodes differently from gfx11.
//!
//! `encode` writes the instructions whose gfx12 encoding is not gfx11's and
//! leaves the rest to the gfx11 encoders (`rir::asm`), whose words gfx12
//! reads unchanged. The layouts and opcodes are `llvm-mc -mcpu=gfx1201`'s;
//! the tests pin words it printed.
//!
//! * Buffer, global and scratch accesses are three words (VBUF, VGLOBAL,
//!   VSCRATCH) with a 24-bit offset. gfx11's GLC bit, which makes an atomic
//!   return the old value, is the temporal hint `TH_ATOMIC_RETURN`.
//! * `s_load` keeps two words with the opcode, a 24-bit offset and the
//!   offset register in other bits.
//! * `v_min_f32`, `v_max_f32` and their three-operand and f64 forms are
//!   gfx12's `*_num_*` instructions, under other opcodes; `v_add_f64`,
//!   `v_mul_f64` and `v_lshlrev_b64` keep their names under other opcodes.

use super::{removed, Gfx12Op};
use crate::error::CompileError;
use crate::rir::mir::{Fmt, MInst, Opc, Opnd, Phys, F_VMEM_STORE, M_GLC, M_SADDR};

fn bad(message: impl Into<String>) -> CompileError {
    CompileError::new("ROCKE_DIRECT_ENCODE", message)
}

/// The register field value of "no register".
const NULL: u32 = 0x7c;

/// `TH_ATOMIC_RETURN` in a memory instruction's temporal hint (bits 22:20
/// of the second word).
const TH_ATOMIC_RETURN: u32 = 1 << 20;

fn vgpr(o: Option<&Opnd>) -> Result<u32, CompileError> {
    match o {
        Some(&Opnd::P(Phys::V(r), _)) => Ok(r as u32),
        other => Err(bad(format!("{other:?} is not a VGPR"))),
    }
}

fn sgpr(o: Option<&Opnd>) -> Result<u32, CompileError> {
    match o {
        Some(&Opnd::P(Phys::S(r), _)) => Ok(r as u32),
        Some(&Opnd::P(Phys::VccLo, _)) => Ok(106),
        other => Err(bad(format!("{other:?} is not an SGPR"))),
    }
}

/// The 24-bit offset field of a memory instruction's third word (or an
/// `s_load`'s second), from an offset within `range`.
fn offset24(what: &str, offset: i64, range: std::ops::Range<i64>) -> Result<u32, CompileError> {
    if !range.contains(&offset) {
        return Err(bad(format!("{what} offset {offset} out of range")));
    }
    Ok(offset as u32 & 0xff_ffff)
}

fn is_scratch(op: Opc) -> bool {
    matches!(
        op,
        Opc::ScratchLoadB32
            | Opc::ScratchLoadB64
            | Opc::ScratchLoadB96
            | Opc::ScratchLoadB128
            | Opc::ScratchStoreB32
            | Opc::ScratchStoreB64
            | Opc::ScratchStoreB96
            | Opc::ScratchStoreB128
    )
}

/// gfx12's opcode for an instruction it renumbered, by the form the gfx11
/// encoder chose: `(gfx11 VOP2, gfx12 VOP2)` and `(gfx11 VOP3, gfx12 VOP3)`.
/// The f64 and 64-bit forms are VOP3-only on gfx11 and VOP2 instructions on
/// gfx12, whose VOP3 form is 0x100 + the VOP2 opcode.
fn renumbered(op: Opc) -> Option<(Option<(u32, u32)>, (u32, u32))> {
    Some(match op {
        Opc::VMinF32 => (Some((0x0f, 0x15)), (0x10f, 0x115)),
        Opc::VMaxF32 => (Some((0x10, 0x16)), (0x110, 0x116)),
        Opc::VMin3F32 => (None, (0x219, 0x229)),
        Opc::VMax3F32 => (None, (0x21c, 0x22a)),
        Opc::VAddF64 => (None, (0x327, 0x102)),
        Opc::VMulF64 => (None, (0x328, 0x106)),
        Opc::VMinF64 => (None, (0x329, 0x10d)),
        Opc::VMaxF64 => (None, (0x32a, 0x10e)),
        Opc::VLshlrevB64 => (None, (0x33c, 0x11f)),
        _ => return None,
    })
}

/// Rewrites the opcode of the gfx11 words of a renumbered instruction in
/// place.
fn renumber(op: Opc, word: &mut u32) -> Result<(), CompileError> {
    let (vop2, vop3) = renumbered(op).expect("a renumbered instruction");
    let stale = || bad(format!("{}: the gfx11 encoding {:#010x} is not the one gfx12 renumbers", op.name(), *word));
    if *word >> 26 == 0x35 {
        if (*word >> 16) & 0x3ff != vop3.0 {
            return Err(stale());
        }
        *word = *word & !(0x3ff << 16) | vop3.1 << 16;
    } else {
        let Some((from, to)) = vop2 else { return Err(stale()) };
        if *word >> 31 != 0 || (*word >> 25) & 0x3f != from {
            return Err(stale());
        }
        *word = *word & !(0x3f << 25) | to << 25;
    }
    Ok(())
}

/// Encodes `inst` into `out` if gfx12 encodes it differently from gfx11;
/// whether it did. `gfx11` encodes an instruction as gfx11 does.
pub fn encode(
    inst: &MInst,
    out: &mut Vec<u32>,
    gfx11: impl FnOnce(&MInst, &mut Vec<u32>) -> Result<(), CompileError>,
) -> Result<bool, CompileError> {
    let op = inst.op;
    if let Opc::Gfx12(g) = op {
        out.push(match g {
            // SOP1 with the inline constant -1 as its source.
            Gfx12Op::SBarrierSignal => 0xBE80_0000 | (g.code() as u32) << 8 | 0xc1,
            _ => 0xBF80_0000 | (g.code() as u32) << 16 | (inst.imm as u32 & 0xffff),
        });
        return Ok(true);
    }
    if removed(op) {
        return Err(bad(format!("{} is not a gfx12 instruction", op.name())));
    }
    if renumbered(op).is_some() {
        let at = out.len();
        gfx11(inst, out)?;
        renumber(op, &mut out[at])?;
        return Ok(true);
    }
    let (code, defs, uses) = (op.code() as u32, &inst.defs[..], &inst.uses[..]);
    match op.fmt() {
        Fmt::Smem => {
            let sdata = match defs.first() {
                Some(Opnd::P(Phys::Null, _)) => NULL,
                d => sgpr(d)?,
            };
            let sbase = sgpr(uses.first())? >> 1;
            out.push(0xF400_0000 | code << 13 | sdata << 6 | sbase);
            // gfx11's 21-bit range: beyond it gfx11 reads the same number as a negative offset.
            out.push(NULL << 25 | offset24("scalar load", inst.imm, -(1 << 20)..1 << 20)?);
        }
        Fmt::Mubuf => {
            let (vdata, rest) = match defs.first() {
                Some(d) => (vgpr(Some(d))?, uses),
                None => (vgpr(uses.first())?, uses.get(1..).unwrap_or_default()),
            };
            let vaddr = vgpr(rest.first())?;
            let srsrc = sgpr(rest.get(1))?;
            // gfx11's inline constant 0 as the offset register is null here.
            let soffset = match rest.get(2) {
                Some(Opnd::Imm(0) | Opnd::P(Phys::Null, _)) => NULL,
                Some(Opnd::P(Phys::M0, _)) => 0x7d,
                Some(Opnd::P(Phys::ExecLo, _)) => 0x7e,
                other => sgpr(other)?,
            };
            out.push(0xC400_0000 | code << 14 | soffset);
            // offen: the address is a byte offset in a VGPR.
            out.push(1 << 30 | 1 << 23 | srsrc << 9 | vdata);
            out.push(offset24("buffer", inst.imm, 0..1 << 23)? << 8 | vaddr);
        }
        Fmt::Global if is_scratch(op) => {
            let (vdst, data) = match defs.first() {
                Some(d) => (vgpr(Some(d))?, 0),
                None => (0, vgpr(uses.first())?),
            };
            out.push(0xED00_0000 | code << 14 | NULL);
            out.push(data << 23 | vdst);
            out.push(offset24("scratch", inst.imm, -(1 << 23)..1 << 23)? << 8);
        }
        Fmt::Global => {
            let saddr_mode = inst.mods & M_SADDR != 0;
            let addr = vgpr(uses.first())?;
            let saddr = if saddr_mode { sgpr(uses.get(1))? } else { NULL };
            let next = if saddr_mode { 2 } else { 1 };
            let returns = inst.mods & M_GLC != 0;
            let (vdst, data) = match defs.first() {
                None => (0, vgpr(uses.get(next))?),
                // A returning atomic: the old value and the operand.
                Some(d) if op.flags() & F_VMEM_STORE != 0 => {
                    if !returns {
                        return Err(bad("a returning atomic without its return hint"));
                    }
                    (vgpr(Some(d))?, vgpr(uses.get(next))?)
                }
                // gfx11's GLC on a load is a cache policy; here the bit would be a temporal hint.
                Some(_) if returns => return Err(bad("a return hint on a load")),
                Some(d) => (vgpr(Some(d))?, 0),
            };
            if returns && defs.is_empty() {
                return Err(bad("a return hint on an access that returns nothing"));
            }
            out.push(0xEE00_0000 | code << 14 | saddr);
            out.push(data << 23 | u32::from(returns) * TH_ATOMIC_RETURN | vdst);
            out.push(offset24("global", inst.imm, -(1 << 23)..1 << 23)? << 8 | addr);
        }
        _ => return Ok(false),
    }
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(r: u16, n: u8) -> Opnd {
        Opnd::P(Phys::S(r), n)
    }

    fn v(r: u16, n: u8) -> Opnd {
        Opnd::P(Phys::V(r), n)
    }

    fn words(inst: MInst) -> Vec<u32> {
        crate::rir::asm::encode_words(&inst, crate::target::Target::Gfx1201).unwrap_or_else(|e| panic!("{e}"))
    }

    // The expected words are `llvm-mc -arch=amdgcn -mcpu=gfx1201 -show-encoding`'s
    // for the instruction in each comment.

    #[test]
    fn waits_and_the_barrier() {
        let g = |op, imm| words(MInst::new(Opc::Gfx12(op), [], []).with_imm(imm));
        assert_eq!(g(Gfx12Op::SWaitLoadcnt, 3), [0xbfc0_0003]); // s_wait_loadcnt 0x3
        assert_eq!(g(Gfx12Op::SWaitStorecnt, 0), [0xbfc1_0000]); // s_wait_storecnt 0x0
        assert_eq!(g(Gfx12Op::SWaitDscnt, 2), [0xbfc6_0002]); // s_wait_dscnt 0x2
        assert_eq!(g(Gfx12Op::SWaitKmcnt, 0), [0xbfc7_0000]); // s_wait_kmcnt 0x0
        assert_eq!(g(Gfx12Op::SWaitLoadcntDscnt, 0x203), [0xbfc8_0203]); // s_wait_loadcnt_dscnt 0x203
        assert_eq!(g(Gfx12Op::SBarrierSignal, 0xffff), [0xbe80_4ec1]); // s_barrier_signal -1
        assert_eq!(g(Gfx12Op::SBarrierWait, 0xffff), [0xbf94_ffff]); // s_barrier_wait 0xffff
        assert_eq!(g(Gfx12Op::SWaitAlu, 0xff9e), [0xbf88_ff9e]); // s_wait_alu depctr_sa_sdst(0)
        assert_eq!(g(Gfx12Op::SWaitAlu, 0xf19f), [0xbf88_f19f]); // s_wait_alu depctr_va_sdst(0)
        assert_eq!(g(Gfx12Op::SWaitAlu, 0xff9d), [0xbf88_ff9d]); // s_wait_alu depctr_va_vcc(0)
    }

    #[test]
    fn gfx11_sync_instructions_are_refused() {
        for op in [Opc::SWaitcnt, Opc::SBarrier, Opc::BufferGl0Inv] {
            let mut out = Vec::new();
            assert!(encode(&MInst::new(op, [], []), &mut out, |_, _| unreachable!()).is_err());
        }
    }

    #[test]
    fn buffer_accesses() {
        // buffer_load_b128 v[2:5], v2, s[4:7], null offen offset:16
        let load = MInst::new(Opc::BufLoadB128, [v(2, 4)], [v(2, 1), s(4, 4), Opnd::Imm(0)]).with_imm(16);
        assert_eq!(words(load), [0xc405_c07c, 0x4080_0802, 0x0000_1002]);
        // buffer_store_b16 v2, v3, s[4:7], null offen
        let store = MInst::new(Opc::BufStoreB16, [], [v(2, 1), v(3, 1), s(4, 4), Opnd::Imm(0)]);
        assert_eq!(words(store), [0xc406_407c, 0x4080_0802, 0x0000_0003]);
        // buffer_load_b32 v0, v0, s[100:103], s101 offen offset:8388607
        let far = MInst::new(Opc::BufLoadB32, [v(0, 1)], [v(0, 1), s(100, 4), s(101, 1)]).with_imm(8388607);
        assert_eq!(words(far), [0xc405_0065, 0x4080_c800, 0x7fff_ff00]);
    }

    #[test]
    fn global_and_scratch_accesses() {
        // global_load_b128 v[2:5], v[6:7], off offset:16
        let load = MInst::new(Opc::GLoadB128, [v(2, 4)], [v(6, 2)]).with_imm(16);
        assert_eq!(words(load), [0xee05_c07c, 0x0000_0002, 0x0000_1006]);
        // global_load_b32 v0, v[0:1], off offset:-1
        let back = MInst::new(Opc::GLoadB32, [v(0, 1)], [v(0, 2)]).with_imm(-1);
        assert_eq!(words(back), [0xee05_007c, 0x0000_0000, 0xffff_ff00]);
        // global_store_b32 v0, v9, s[100:101] offset:16
        let store = MInst::new(Opc::GStoreB32, [], [v(0, 1), s(100, 2), v(9, 1)]).with_imm(16).with_mods(M_SADDR);
        assert_eq!(words(store), [0xee06_8064, 0x0480_0000, 0x0000_1000]);
        // global_atomic_add_u32 v[0:1], v9, off
        let add = MInst::new(Opc::GAtomicAddU32, [], [v(0, 2), v(9, 1)]);
        assert_eq!(words(add), [0xee0d_407c, 0x0480_0000, 0x0000_0000]);
        // global_atomic_add_u32 v5, v[0:1], v9, off th:TH_ATOMIC_RETURN
        let rtn = MInst::new(Opc::GAtomicAddU32Rtn, [v(5, 1)], [v(0, 2), v(9, 1)]).with_mods(M_GLC);
        assert_eq!(words(rtn), [0xee0d_407c, 0x0490_0005, 0x0000_0000]);
        // scratch_load_b32 v255, off, off offset:-1
        let load = MInst::new(Opc::ScratchLoadB32, [v(255, 1)], []).with_imm(-1);
        assert_eq!(words(load), [0xed05_007c, 0x0000_00ff, 0xffff_ff00]);
        // scratch_store_b32 off, v255, off offset:1
        let store = MInst::new(Opc::ScratchStoreB32, [], [v(255, 1)]).with_imm(1);
        assert_eq!(words(store), [0xed06_807c, 0x7f80_0000, 0x0000_0100]);
    }

    #[test]
    fn scalar_loads() {
        // s_load_b64 s[4:5], s[4:5], 0x0
        let load = MInst::new(Opc::SLoadB64, [s(4, 2)], [s(4, 2)]);
        assert_eq!(words(load), [0xf400_2102, 0xf800_0000]);
        // s_load_b32 s105, s[104:105], -0x1
        let back = MInst::new(Opc::SLoadB32, [s(105, 1)], [s(104, 2)]).with_imm(-1);
        assert_eq!(words(back), [0xf400_1a74, 0xf8ff_ffff]);
        // s_load_b512 s[4:19], s[4:5], 0x0
        let wide = MInst::new(Opc::SLoadB512, [s(4, 16)], [s(4, 2)]);
        assert_eq!(words(wide), [0xf400_8102, 0xf800_0000]);
    }

    #[test]
    fn min_and_max_take_their_gfx12_opcodes() {
        // v_min_num_f32_e32 v2, v3, v2 and v_max_num_f32_e32 v2, v3, v2
        let two = |op| words(MInst::new(op, [v(2, 1)], [v(3, 1), v(2, 1)]));
        assert_eq!(two(Opc::VMinF32), [0x2a04_0503]);
        assert_eq!(two(Opc::VMaxF32), [0x2c04_0503]);
        // v_min3_num_f32 v2, v3, v3, v3 and v_max3_num_f32 v2, v3, v3, v3
        let three = |op| words(MInst::new(op, [v(2, 1)], [v(3, 1), v(3, 1), v(3, 1)]));
        assert_eq!(three(Opc::VMin3F32), [0xd629_0002, 0x040e_0703]);
        assert_eq!(three(Opc::VMax3F32), [0xd62a_0002, 0x040e_0703]);
    }

    #[test]
    fn the_64_bit_instructions_gfx12_moved_take_their_opcodes() {
        // v_add_f64_e64, v_mul_f64_e64 v[2:3], v[3:4], v[3:4] and
        // v_lshlrev_b64_e64 v[2:3], v3, v[3:4]; llvm-mc fills the unused
        // third source with 0x80.
        let wide = |op, a| words(MInst::new(op, [v(2, 2)], [a, v(3, 2)]));
        assert_eq!(wide(Opc::VAddF64, v(3, 2)), [0xd502_0002, 0x0002_0703]);
        assert_eq!(wide(Opc::VMulF64, v(3, 2)), [0xd506_0002, 0x0002_0703]);
        assert_eq!(wide(Opc::VLshlrevB64, v(3, 1)), [0xd51f_0002, 0x0002_0703]);
    }

    #[test]
    fn the_compares_gfx12_dropped_are_refused() {
        // v_cmp_f_f32, v_cmp_t_u64; v_cmp_lt_f32 and v_cmp_ge_u64 stay.
        let cmp = |code| crate::rir::asm::encode_words(&MInst::new(Opc::VCmp(code), [Opnd::P(Phys::VccLo, 1)], [v(1, 1), v(2, 1)]), crate::target::Target::Gfx1201);
        assert!(cmp(0x10).is_err() && cmp(0x5f).is_err());
        assert!(cmp(0x11).is_ok() && cmp(0x5e).is_ok());
    }

    #[test]
    fn everything_else_is_left_to_gfx11() {
        let add = MInst::new(Opc::VAddNcU32, [v(0, 1)], [v(1, 1), v(2, 1)]);
        let mut out = Vec::new();
        assert!(!encode(&add, &mut out, |_, _| unreachable!()).unwrap());
        assert!(out.is_empty());
    }
}
