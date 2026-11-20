// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

//! The gfx12 ISA, wave32: gfx1201 and gfx12-generic, which differ only in
//! what the code object states about its processor.
//!
//! gfx12 keeps most of gfx11's instructions and encodings, so a gfx12
//! kernel compiles through the gfx11 pipeline and this module supplies what
//! differs:
//!
//! * `lanes::check`, before optimization: a kernel that reads a lane of the
//!   other half of a wave64 is refused.
//! * `lower_sync`, after allocation: gfx12 has no `s_waitcnt`,
//!   `s_waitcnt_vscnt` or `s_barrier`. Loads, LDS, scalar loads and stores
//!   each have a counter with its own wait, and a barrier is a signal and a
//!   wait.
//! * `sgpr::insert_waits`: a VALU or SALU instruction must not read an SGPR
//!   whose last write is uncommitted (`s_wait_alu`).
//! * `asm::encode`: the 96-bit buffer, global and scratch memory encodings,
//!   the scalar load layout, and the opcodes gfx12 renumbered.
//!
//! The instruction facts are LLVM's for `-mcpu=gfx1201`
//! (`FeatureISAVersion12`, which `gfx12-generic` shares); `llvm-mc` is the
//! encoding reference, offline.

pub mod asm;
pub mod fence;
pub mod lanes;
pub mod sgpr;

use crate::error::CompileError;
use crate::rir::gen_isa::{lat, Lat};
use crate::rir::mir::{Fmt, MFunc, MInst, Opc, Opnd, F_BRANCH, F_END, F_FENCE, F_LDS, F_SMEM, F_VMEM_LOAD, F_VMEM_STORE};

/// Instructions specific to gfx12. Explicit fence instructions can enter
/// the MIR before scheduling; their ordering is carried by `F_FENCE`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Gfx12Op {
    /// `s_wait_loadcnt n`: at most `n` VMEM loads outstanding.
    SWaitLoadcnt,
    /// `s_wait_storecnt n`: at most `n` VMEM stores outstanding.
    SWaitStorecnt,
    /// `s_wait_dscnt n`: at most `n` LDS operations outstanding.
    SWaitDscnt,
    /// `s_wait_kmcnt n`: at most `n` scalar loads outstanding.
    SWaitKmcnt,
    /// `s_wait_loadcnt_dscnt`: both, as `loadcnt << 8 | dscnt`.
    SWaitLoadcntDscnt,
    /// `s_barrier_signal -1`: this wave has reached the workgroup barrier.
    SBarrierSignal,
    /// `s_barrier_wait -1`: wait until every wave has signalled.
    SBarrierWait,
    /// `s_wait_alu`: wait for the dependency counters whose field is 0.
    SWaitAlu,
    /// Operandless cache controls; `imm` holds the ISA scope field.
    GlobalInv,
    GlobalWb,
}

impl Gfx12Op {
    pub const ALL: [Gfx12Op; 10] = [
        Gfx12Op::SWaitLoadcnt,
        Gfx12Op::SWaitStorecnt,
        Gfx12Op::SWaitDscnt,
        Gfx12Op::SWaitKmcnt,
        Gfx12Op::SWaitLoadcntDscnt,
        Gfx12Op::SBarrierSignal,
        Gfx12Op::SBarrierWait,
        Gfx12Op::SWaitAlu,
        Gfx12Op::GlobalInv,
        Gfx12Op::GlobalWb,
    ];

    pub fn fmt(self) -> Fmt {
        match self {
            Gfx12Op::SBarrierSignal => Fmt::Sop1,
            Gfx12Op::GlobalInv | Gfx12Op::GlobalWb => Fmt::Global,
            _ => Fmt::Sopp,
        }
    }

    /// The gfx12 opcode within the format.
    pub fn code(self) -> u16 {
        match self {
            Gfx12Op::SWaitLoadcnt => 0x40,
            Gfx12Op::SWaitStorecnt => 0x41,
            Gfx12Op::SWaitDscnt => 0x46,
            Gfx12Op::SWaitKmcnt => 0x47,
            Gfx12Op::SWaitLoadcntDscnt => 0x48,
            Gfx12Op::SBarrierSignal => 0x4e,
            Gfx12Op::SBarrierWait => 0x14,
            Gfx12Op::SWaitAlu => 0x08,
            Gfx12Op::GlobalInv => 0x2b,
            Gfx12Op::GlobalWb => 0x2c,
        }
    }

    pub fn mnemonic(self) -> &'static str {
        match self {
            Gfx12Op::SWaitLoadcnt => "s_wait_loadcnt",
            Gfx12Op::SWaitStorecnt => "s_wait_storecnt",
            Gfx12Op::SWaitDscnt => "s_wait_dscnt",
            Gfx12Op::SWaitKmcnt => "s_wait_kmcnt",
            Gfx12Op::SWaitLoadcntDscnt => "s_wait_loadcnt_dscnt",
            Gfx12Op::SBarrierSignal => "s_barrier_signal",
            Gfx12Op::SBarrierWait => "s_barrier_wait",
            Gfx12Op::SWaitAlu => "s_wait_alu",
            Gfx12Op::GlobalInv => "global_inv",
            Gfx12Op::GlobalWb => "global_wb",
        }
    }

    pub fn flags(self) -> u32 {
        match self {
            Gfx12Op::SBarrierSignal | Gfx12Op::SBarrierWait | Gfx12Op::GlobalInv | Gfx12Op::GlobalWb
            | Gfx12Op::SWaitStorecnt | Gfx12Op::SWaitLoadcntDscnt => F_FENCE,
            _ => 0,
        }
    }
}

/// Whether gfx12 removed the shared (gfx11) opcode `op`: the single wait
/// word, the store wait, the one-instruction barrier, the cache
/// invalidations and the always-false and always-true compares.
/// `lower_sync` replaces the first three kinds. gfx12 fences select their
/// own scoped cache operations, never these legacy invalidations.
pub fn removed(op: Opc) -> bool {
    match op {
        Opc::SWaitcnt
        | Opc::SWaitcntExplicit
        | Opc::SWaitcntVscnt
        | Opc::SBarrier
        | Opc::SBarrierLds
        | Opc::SBarrierBare
        | Opc::BufferGl0Inv
        | Opc::BufferGl1Inv => true,
        // The F and T predicates: the first and last of sixteen float and
        // of eight integer predicates a type.
        Opc::VCmp(code @ 0x00..=0x2f) => matches!(code & 0xf, 0 | 0xf),
        Opc::VCmp(code @ 0x30..=0x5f) => matches!(code & 7, 0 | 7),
        _ => false,
    }
}

/// Set in `MInst::mods` of an `s_waitcnt` by the wait model: no LDS
/// operation (no scalar load) was outstanding at the wait. gfx11 counts both
/// in lgkmcnt; gfx12's dscnt and kmcnt each wait only for what is. A wait
/// without them (an explicit one) waits for both.
pub const M_WAIT_NO_DS: u32 = 1 << 0;
pub const M_WAIT_NO_SMEM: u32 = 1 << 1;

/// The trap temporaries the hardware delivers the workgroup ids in, as
/// SGPR numbers: x in ttmp9, y in the low half of ttmp7 and z in its high
/// half (zero unless the kernel descriptor enables z).
pub const TTMP7: u16 = 108 + 7;
pub const TTMP9: u16 = 108 + 9;

/// `compute_pgm_rsrc1` and `compute_pgm_rsrc3` of a gfx12 kernel, from
/// gfx11's `rsrc1`: gfx12 has no DX10 clamp or IEEE mode (their bits are
/// workgroup round-robin and performance-counter controls, left off), and
/// its instruction prefetch size is eight bits of 128-byte lines.
pub fn program_resources(rsrc1: u32, code_bytes: usize) -> (u32, u32) {
    const DX10_CLAMP: u32 = 1 << 21;
    const IEEE_MODE: u32 = 1 << 23;
    (rsrc1 & !(DX10_CLAMP | IEEE_MODE), (code_bytes.div_ceil(128).min(255) as u32) << 4)
}

/// The barrier instructions' all-waves operand.
const ALL_WAVES: i64 = 0xffff;

fn inst(op: Gfx12Op, imm: i64) -> MInst {
    MInst::new(Opc::Gfx12(op), [], []).with_imm(imm)
}

/// The gfx12 waits of a gfx11 wait word (`vm << 10 | lgkm << 4 | exp`, 63:
/// no wait on that counter) with the `M_WAIT_NO_*` of `idle`. gfx11's lgkm
/// counter is gfx12's dscnt for LDS and kmcnt for scalar loads; scalar
/// loads return out of order, so the model only ever waits for all of them.
fn waits(out: &mut Vec<MInst>, word: i64, idle: u32) {
    let (vm, lgkm) = ((word >> 10) & 0x3f, (word >> 4) & 0x3f);
    let ds = lgkm < 0x3f && !(lgkm == 0 && idle & M_WAIT_NO_DS != 0);
    match (vm < 0x3f, ds) {
        (true, true) => out.push(inst(Gfx12Op::SWaitLoadcntDscnt, vm << 8 | lgkm)),
        (true, false) => out.push(inst(Gfx12Op::SWaitLoadcnt, vm)),
        (false, true) => out.push(inst(Gfx12Op::SWaitDscnt, lgkm)),
        (false, false) => {}
    }
    if lgkm == 0 && idle & M_WAIT_NO_SMEM == 0 {
        out.push(inst(Gfx12Op::SWaitKmcnt, 0));
    }
}

/// Replaces the gfx11 wait and barrier instructions by gfx12's. Runs after
/// the wait model (`post::insert_waitcnt`), whose `s_waitcnt` it reads, and
/// before the passes that count instructions (`s_delay_alu`).
pub fn lower_sync(f: &mut MFunc) -> Result<(), CompileError> {
    for block in f.blocks.iter_mut() {
        if !block.insts.iter().any(|i| removed(i.op)) {
            continue;
        }
        let mut out = Vec::with_capacity(block.insts.len() + 8);
        for i in std::mem::take(&mut block.insts) {
            match i.op {
                Opc::SWaitcnt => waits(&mut out, i.imm, i.mods),
                // An explicit lgkmcnt(0) settles LDS and scalar loads.
                Opc::SWaitcntExplicit => waits(&mut out, i.imm, 0),
                Opc::SWaitcntVscnt => {
                    if !matches!(i.defs.first(), Some(Opnd::P(crate::rir::mir::Phys::Null, _))) {
                        return Err(CompileError::new("ROCKE_DIRECT_ENCODE", "s_waitcnt_vscnt takes null"));
                    }
                    out.push(inst(Gfx12Op::SWaitStorecnt, i.imm & 0x3f));
                }
                Opc::SBarrier | Opc::SBarrierLds | Opc::SBarrierBare => {
                    out.push(inst(Gfx12Op::SBarrierSignal, ALL_WAVES));
                    out.push(inst(Gfx12Op::SBarrierWait, ALL_WAVES));
                }
                Opc::BufferGl0Inv | Opc::BufferGl1Inv => {
                    return Err(CompileError::new(
                        "ROCKE_DIRECT_UNSUPPORTED",
                        format!("{} has no gfx1201 lowering", i.op.name()),
                    ));
                }
                _ => out.push(i),
            }
        }
        block.insts = out;
    }
    Ok(())
}

/// Whether `i` only computes in this wave's registers: no memory access, no
/// counter wait, no branch, no fence.
fn register_only(i: &MInst) -> bool {
    let memory_or_control = F_VMEM_LOAD | F_VMEM_STORE | F_LDS | F_SMEM | F_BRANCH | F_END | F_FENCE;
    matches!(lat(i.op), Lat::Salu | Lat::Valu | Lat::Trans | Lat::Wmma) && i.op.flags() & memory_or_control == 0
}

/// Moves each `s_barrier_wait` below the register-only instructions that
/// follow it in its block, so that they run while the other waves arrive
/// (LLVM's schedule of the address arithmetic after a barrier). The signal
/// stays, so the other waves see this wave arrive where they did; the
/// instructions passed read and write registers of this wave only, which no
/// wave's progress changes; and the first instruction that accesses memory,
/// waits for a counter, branches or fences ends the run, so every memory
/// access keeps its side of the barrier. Runs after `lower_sync`.
pub fn sink_barrier_waits(f: &mut MFunc) {
    if crate::rir::debug_flag("nobarriersink") {
        return;
    }
    for block in f.blocks.iter_mut() {
        let insts = &mut block.insts;
        let mut at = 0;
        while at < insts.len() {
            if insts[at].op == Opc::Gfx12(Gfx12Op::SBarrierWait) {
                let end = insts[at + 1..].iter().position(|i| !register_only(i)).map_or(insts.len(), |p| at + 1 + p);
                insts[at..end].rotate_left(1);
                at = end;
            } else {
                at += 1;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rir::mir::{MBlock, Phys};

    fn func(insts: Vec<MInst>) -> MFunc {
        MFunc { blocks: vec![MBlock { insts, ..Default::default() }], target: crate::target::Target::Gfx1201, ..Default::default() }
    }

    fn lowered(insts: Vec<MInst>) -> Vec<(Opc, i64)> {
        let mut f = func(insts);
        lower_sync(&mut f).unwrap();
        f.blocks[0].insts.iter().map(|i| (i.op, i.imm)).collect()
    }

    fn wait(vm: i64, lgkm: i64) -> MInst {
        MInst::new(Opc::SWaitcnt, [], []).with_imm(vm << 10 | lgkm << 4 | 7)
    }

    #[test]
    fn a_wait_word_becomes_the_waits_of_its_counters() {
        use Gfx12Op::*;
        let g = |op, imm| (Opc::Gfx12(op), imm);
        let idle = |mods| lowered(vec![wait(63, 0).with_mods(mods)]);
        assert_eq!(lowered(vec![wait(3, 63)]), [g(SWaitLoadcnt, 3)]);
        assert_eq!(lowered(vec![wait(63, 2).with_mods(M_WAIT_NO_SMEM)]), [g(SWaitDscnt, 2)]);
        // Three loads and two LDS operations may stay outstanding: LLVM's `encodeLoadcntDscnt`.
        assert_eq!(lowered(vec![wait(3, 2).with_mods(M_WAIT_NO_SMEM)]), [g(SWaitLoadcntDscnt, 0x302)]);
        // lgkm 0 is the counter of each kind that was outstanding.
        assert_eq!(idle(M_WAIT_NO_SMEM), [g(SWaitDscnt, 0)]);
        assert_eq!(idle(M_WAIT_NO_DS), [g(SWaitKmcnt, 0)]);
        assert_eq!(idle(0), [g(SWaitDscnt, 0), g(SWaitKmcnt, 0)]);
        let both = wait(1, 0).with_mods(M_WAIT_NO_DS);
        assert_eq!(lowered(vec![both]), [g(SWaitLoadcnt, 1), g(SWaitKmcnt, 0)]);
        // An explicit wait knows neither.
        let explicit = MInst::new(Opc::SWaitcntExplicit, [], []).with_imm(7);
        assert_eq!(lowered(vec![explicit]), [g(SWaitLoadcntDscnt, 0), g(SWaitKmcnt, 0)]);
    }

    #[test]
    fn barriers_and_store_waits_take_their_gfx12_forms() {
        use Gfx12Op::*;
        let g = |op, imm| (Opc::Gfx12(op), imm);
        for barrier in [Opc::SBarrier, Opc::SBarrierLds, Opc::SBarrierBare] {
            assert_eq!(lowered(vec![MInst::new(barrier, [], [])]), [g(SBarrierSignal, 0xffff), g(SBarrierWait, 0xffff)]);
        }
        let vscnt = MInst::new(Opc::SWaitcntVscnt, [Opnd::P(Phys::Null, 1)], []);
        assert_eq!(lowered(vec![vscnt]), [g(SWaitStorecnt, 0)]);
        let mut f = func(vec![MInst::new(Opc::BufferGl0Inv, [], [])]);
        assert!(lower_sync(&mut f).is_err());
    }

    #[test]
    fn register_work_after_a_barrier_runs_before_its_wait() {
        use Gfx12Op::*;
        let s = |n| Opnd::P(Phys::S(n), 1);
        let v = |n| Opnd::P(Phys::V(n), 1);
        let add = |n| MInst::new(Opc::SAddI32, [s(n)], [s(n), Opnd::Imm(1)]);
        let vadd = MInst::new(Opc::VAddNcU32, [v(4)], [v(4), v(4)]);
        let barrier = || MInst::new(Opc::SBarrier, [], []);
        let sunk = |after: Vec<MInst>| {
            let mut f = func([vec![add(1), barrier()], after].concat());
            lower_sync(&mut f).unwrap();
            sink_barrier_waits(&mut f);
            f.blocks[0].insts.iter().map(|i| i.op).collect::<Vec<Opc>>()
        };
        let g = Opc::Gfx12;
        // The work before the barrier stays before the signal; an LDS load
        // stays after the wait, with what follows it.
        let load = MInst::new(Opc::DsLoadB32, [v(3)], [v(1)]);
        assert_eq!(
            sunk(vec![vadd.clone(), add(2), load, add(3)]),
            [Opc::SAddI32, g(SBarrierSignal), Opc::VAddNcU32, Opc::SAddI32, g(SBarrierWait), Opc::DsLoadB32, Opc::SAddI32]
        );
        // So does a wait for a load: its reader may not pass the barrier.
        assert_eq!(
            sunk(vec![add(2), wait(0, 63), vadd.clone()]),
            [Opc::SAddI32, g(SBarrierSignal), Opc::SAddI32, g(SBarrierWait), g(SWaitLoadcnt), Opc::VAddNcU32]
        );
        // A second barrier, a store and a branch each end the run.
        let store = MInst::new(Opc::GStoreB32, [], [Opnd::P(Phys::V(12), 2), v(3)]);
        let branch = MInst::new(Opc::SCbranchScc1, [], []);
        for stop in [barrier(), store, branch] {
            let ops = sunk(vec![vadd.clone(), stop.clone(), add(2)]);
            assert_eq!(ops[..4], [Opc::SAddI32, g(SBarrierSignal), Opc::VAddNcU32, g(SBarrierWait)], "{:?}", stop.op);
        }
    }

    #[test]
    fn nothing_removed_survives_and_the_rest_is_untouched() {
        let keep = MInst::new(Opc::VNop, [], []);
        let out = lowered(vec![keep, wait(0, 0), MInst::new(Opc::SBarrier, [], [])]);
        assert_eq!(out[0], (Opc::VNop, 0));
        assert!(out.iter().all(|&(op, _)| !removed(op)));
    }

    #[test]
    fn the_descriptor_words_are_gfx12s() {
        // COMGR's words for a gfx1201 kernel of 208 VGPRs and 8712 bytes:
        // no clamp or IEEE bits, and a prefetch size past gfx11's six bits.
        assert_eq!(program_resources(0xe0af_0019, 8712), (0xe00f_0019, 0x450));
        assert_eq!(program_resources(0xe0af_0000, 1 << 20).1, 0xff0);
    }
}
