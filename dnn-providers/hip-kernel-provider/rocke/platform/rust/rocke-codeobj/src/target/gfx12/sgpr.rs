// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

//! The gfx12 SGPR read hazard.
//!
//! On gfx12 (before gfx1250) the hardware does not interlock every read of
//! an SGPR with its last write: once a VALU instruction has read an SGPR
//! pair, a later VALU or SALU read of a register of that pair must not issue
//! while a write of it is uncommitted. `s_wait_alu` waits for the writes:
//! `sa_sdst` for SALU writes, `va_sdst` for VALU writes, `va_vcc` for VALU
//! writes of VCC. A buffer access or a scalar load commits them all.
//!
//! This is LLVM's `AMDGPUWaitSGPRHazards` pass (`hasVALUReadSGPRHazard`),
//! rule for rule, without its optional culls; it runs where LLVM runs it,
//! after the hazard recognizer and before the `s_delay_alu` pass. Missing a
//! wait reads a stale register; an extra one only stalls.

use super::Gfx12Op;
use crate::rir::gen_isa::is_valu;
use crate::rir::mir::{Fmt, MFunc, MInst, Opc, Opnd, Phys, F_USE_VCC};

/// `s_wait_alu` waiting for nothing: every field at its maximum.
pub const DEPCTR_DEFAULT: i64 = 0xff9f;
/// The fields this pass zeroes.
pub const SA_SDST: i64 = 1 << 0;
pub const VA_VCC: i64 = 1 << 1;
pub const VA_SDST: i64 = 0x7 << 9;

/// VCC's SGPR number.
const VCC: u16 = 106;

const BY_SALU: u8 = 1;
const BY_VALU: u8 = 2;

const WAIT_VALU: u8 = 1;
const WAIT_SALU: u8 = 2;
const WAIT_VCC: u8 = 4;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct State {
    /// SGPR pairs (register / 2) a VALU instruction has read.
    tracked: u64,
    /// SGPRs with an uncommitted SALU write, and VALU write.
    salu: u128,
    valu: u128,
    /// What wrote VCC last, while uncommitted.
    vcc: u8,
}

impl State {
    /// Adds `other`'s hazards; whether that changed anything.
    fn merge(&mut self, other: &State) -> bool {
        let old = *self;
        self.tracked |= other.tracked;
        self.salu |= other.salu;
        self.valu |= other.valu;
        self.vcc |= other.vcc;
        *self != old
    }
}

/// The SGPR range of an operand. M0, EXEC, null and SCC carry no hazard.
fn sgprs(o: &Opnd) -> Option<(u16, u8)> {
    match *o {
        Opnd::P(Phys::S(r), n) => Some((r, n.max(1))),
        Opnd::P(Phys::VccLo, n) => Some((VCC, n.max(1))),
        _ => None,
    }
}

fn is_salu(op: Opc) -> bool {
    matches!(op.fmt(), Fmt::Sop1 | Fmt::Sop2 | Fmt::Sopc | Fmt::Sopk | Fmt::Sopp)
}

struct Step<'a> {
    state: &'a mut State,
    valu: bool,
    salu: bool,
    wait: u8,
}

impl Step<'_> {
    fn operand(&mut self, (reg, n): (u16, u8), is_use: bool) {
        let s = &mut *self.state;
        let (mut pairs, mut bits) = (0u64, 0u128);
        for r in reg..reg + n as u16 {
            pairs |= 1 << ((r >> 1) & 0x3f);
            bits |= 1 << (r & 0x7f);
        }
        // Reads and writes of an untracked pair are safe; a VALU read starts
        // tracking it.
        if s.tracked & pairs == 0 {
            if self.valu && is_use {
                s.tracked |= pairs;
            }
            return;
        }
        let vcc = reg == VCC;
        if is_use {
            // A SALU read commits the VALU writes.
            if self.salu {
                if vcc {
                    if s.vcc & BY_VALU != 0 {
                        s.vcc = 0;
                    }
                } else {
                    s.valu = 0;
                }
            }
            if s.salu & bits != 0 {
                self.wait |= WAIT_SALU;
            }
            if self.valu && s.valu & bits != 0 {
                self.wait |= WAIT_VALU;
            }
            if vcc {
                if s.vcc & BY_SALU != 0 {
                    self.wait |= WAIT_SALU;
                }
                if s.vcc & BY_VALU != 0 {
                    self.wait |= WAIT_VCC;
                }
            }
        } else if vcc {
            s.vcc = if self.salu { BY_SALU } else { BY_VALU };
        } else if self.salu {
            s.salu |= bits;
        } else {
            s.valu |= bits;
        }
    }
}

/// Steps `state` over `inst`; the `s_wait_alu` operand `inst` needs before
/// it, if any.
fn step(state: &mut State, inst: &MInst) -> Option<i64> {
    let op = inst.op;
    if matches!(op.fmt(), Fmt::Smem | Fmt::Mubuf) {
        (state.salu, state.valu, state.vcc) = (0, 0, 0);
        return None;
    }
    if op == Opc::Gfx12(Gfx12Op::SWaitAlu) {
        if inst.imm & VA_VCC == 0 {
            state.vcc &= !BY_VALU;
        }
        if inst.imm & SA_SDST == 0 {
            state.salu = 0;
            state.vcc &= !BY_SALU;
        }
        if inst.imm & VA_SDST == 0 {
            state.valu = 0;
        }
        return None;
    }
    let valu = is_valu(op);
    let salu = !valu && is_salu(op);
    if !valu && !salu {
        return None;
    }
    let mut step = Step { state, valu, salu, wait: 0 };
    // Each register range once, as the operands name it.
    let mut seen: [Option<(u16, u8)>; 6] = [None; 6];
    let implicit_vcc = (op.flags() & F_USE_VCC != 0).then_some((VCC, 1));
    for (i, range) in inst.uses.iter().filter_map(sgprs).chain(implicit_vcc).enumerate() {
        if !seen[..i].contains(&Some(range)) {
            step.operand(range, true);
        }
        seen[i] = Some(range);
    }
    let wait = step.wait;
    let mask = (wait != 0).then(|| {
        let s = &mut *step.state;
        let mut mask = DEPCTR_DEFAULT;
        if wait & WAIT_VCC != 0 {
            s.vcc &= !BY_VALU;
            mask &= !VA_VCC;
        }
        if wait & WAIT_SALU != 0 {
            s.salu = 0;
            s.vcc &= !BY_SALU;
            mask &= !SA_SDST;
        }
        if wait & WAIT_VALU != 0 {
            s.valu = 0;
            mask &= !VA_SDST;
        }
        mask
    });
    let mut first: Option<(u16, u8)> = None;
    for range in inst.defs.iter().filter_map(sgprs) {
        if first != Some(range) {
            step.operand(range, false);
        }
        first.get_or_insert(range);
    }
    mask
}

/// Inserts `s_wait_alu` before every instruction that would read an SGPR
/// with an uncommitted write.
pub fn insert_waits(f: &mut MFunc) {
    let n = f.blocks.len();
    let succs: Vec<Vec<usize>> = (0..n).map(|b| f.successors(b)).collect();
    let mut preds = vec![0u32; n];
    for &s in succs.iter().flatten() {
        preds[s] += 1;
    }
    // Block entry and exit states, to a fixed point. A block with one
    // predecessor takes that predecessor's exit state; a merge point (and
    // the entry block, which the launch enters) adds every predecessor's.
    let mut entry = vec![State::default(); n];
    let mut exit = vec![State::default(); n];
    let mut queued = vec![true; n];
    let mut work: Vec<usize> = (0..n).rev().collect();
    while let Some(b) = work.pop() {
        queued[b] = false;
        let mut state = entry[b];
        for inst in &f.blocks[b].insts {
            step(&mut state, inst);
        }
        if state == exit[b] {
            continue;
        }
        exit[b] = state;
        for &s in &succs[b] {
            let changed = if preds[s] == 1 && s != 0 {
                let changed = entry[s] != state;
                entry[s] = state;
                changed
            } else {
                entry[s].merge(&state)
            };
            if changed && !queued[s] {
                queued[s] = true;
                work.push(s);
            }
        }
    }
    let wait_alu = Opc::Gfx12(Gfx12Op::SWaitAlu);
    for (b, block) in f.blocks.iter_mut().enumerate() {
        let mut state = entry[b];
        let insts = std::mem::take(&mut block.insts);
        let mut out = Vec::with_capacity(insts.len() + insts.len() / 8 + 4);
        for inst in insts {
            if let Some(mask) = step(&mut state, &inst) {
                match out.last_mut() {
                    // Each field is all ones or zero, so the stricter of two
                    // waits is their conjunction.
                    Some(MInst { op, imm, .. }) if *op == wait_alu => *imm &= mask,
                    _ => out.push(MInst::new(wait_alu, [], []).with_imm(mask)),
                }
            }
            out.push(inst);
        }
        block.insts = out;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rir::mir::MBlock;
    use crate::target::Target;

    fn s(r: u16) -> Opnd {
        Opnd::P(Phys::S(r), 1)
    }

    fn v(r: u16) -> Opnd {
        Opnd::P(Phys::V(r), 1)
    }

    /// `s_add_i32 sD, sA, 1`: a SALU write of `d` reading `a`.
    fn salu(d: u16, a: u16) -> MInst {
        MInst::new(Opc::SAddI32, [s(d)], [s(a), Opnd::Imm(1)])
    }

    /// `v_add_nc_u32 vD, sA, vD`: a VALU read of SGPR `a`.
    fn valu_read(d: u16, a: u16) -> MInst {
        MInst::new(Opc::VAddNcU32, [v(d)], [s(a), v(d)])
    }

    fn func(blocks: Vec<Vec<MInst>>) -> MFunc {
        MFunc {
            blocks: blocks.into_iter().map(|insts| MBlock { insts, loop_depth: 0 }).collect(),
            target: Target::Gfx1201,
            ..Default::default()
        }
    }

    /// The `s_wait_alu` operands by the index of the instruction each one
    /// precedes, per block.
    fn waits(blocks: Vec<Vec<MInst>>) -> Vec<Vec<(usize, i64)>> {
        let mut f = func(blocks);
        insert_waits(&mut f);
        f.blocks
            .iter()
            .map(|b| {
                let mut at = 0;
                let mut out = Vec::new();
                for i in &b.insts {
                    match i.op {
                        Opc::Gfx12(Gfx12Op::SWaitAlu) => out.push((at, i.imm)),
                        _ => at += 1,
                    }
                }
                out
            })
            .collect()
    }

    const SA: i64 = DEPCTR_DEFAULT & !SA_SDST;
    const VA: i64 = DEPCTR_DEFAULT & !VA_SDST;

    #[test]
    fn the_first_valu_read_of_a_pair_is_safe_and_later_reads_wait() {
        // s4 written, read by a VALU (starts tracking), written again, read again.
        let code = vec![salu(4, 6), valu_read(0, 4), salu(4, 6), valu_read(1, 4)];
        assert_eq!(waits(vec![code]), [[(3, SA)]]);
        // The other register of the pair is tracked with it.
        let code = vec![valu_read(0, 4), salu(5, 6), valu_read(1, 5)];
        assert_eq!(waits(vec![code]), [[(2, SA)]]);
        // A register no VALU reads never waits, whoever reads it.
        let code = vec![salu(8, 6), salu(9, 8), salu(8, 9)];
        assert_eq!(waits(vec![code]), [[]]);
    }

    #[test]
    fn a_salu_read_of_a_tracked_register_waits_for_a_salu_write() {
        let code = vec![valu_read(0, 4), salu(4, 6), salu(7, 4)];
        assert_eq!(waits(vec![code]), [[(2, SA)]]);
    }

    #[test]
    fn one_wait_covers_every_write_before_it() {
        let code = vec![valu_read(0, 4), valu_read(0, 6), salu(4, 8), salu(6, 8), valu_read(1, 4), valu_read(1, 6)];
        assert_eq!(waits(vec![code]), [[(4, SA)]]);
    }

    #[test]
    fn a_buffer_access_or_scalar_load_commits_the_writes() {
        let load = MInst::new(Opc::BufLoadB32, [v(9)], [v(8), Opnd::P(Phys::S(12), 4), Opnd::Imm(0)]);
        let code = vec![valu_read(0, 4), salu(4, 6), load, valu_read(1, 4)];
        assert_eq!(waits(vec![code]), [[]]);
        let smem = MInst::new(Opc::SLoadB32, [s(20)], [Opnd::P(Phys::S(2), 2)]);
        let code = vec![valu_read(0, 4), salu(4, 6), smem, valu_read(1, 4)];
        assert_eq!(waits(vec![code]), [[]]);
        // A global access does neither.
        let global = MInst::new(Opc::GLoadB32, [v(9)], [Opnd::P(Phys::V(8), 2)]);
        let code = vec![valu_read(0, 4), salu(4, 6), global, valu_read(1, 4)];
        assert_eq!(waits(vec![code]), [[(3, SA)]]);
    }

    #[test]
    fn valu_writes_wait_for_valu_readers_only() {
        // v_readfirstlane sD, vA: a VALU write of an SGPR.
        let first = |d: u16| MInst::new(Opc::VReadfirstlaneB32, [s(d)], [v(3)]);
        let code = vec![valu_read(0, 4), first(4), valu_read(1, 4)];
        assert_eq!(waits(vec![code]), [[(2, VA)]]);
        // A SALU read needs no wait and commits the write for later readers.
        let code = vec![valu_read(0, 4), first(4), salu(7, 4), valu_read(1, 4)];
        assert_eq!(waits(vec![code]), [[]]);
    }

    #[test]
    fn tracking_reaches_a_loop_head_through_the_back_edge() {
        // bb0 falls into bb1; bb1 writes s4, reads it with a VALU and loops.
        // The read tracks the pair only in the first trip, so the wait the
        // later trips need is there from the head on.
        let branch = MInst::new(Opc::SCbranchScc1, [], []).with_imm(1);
        let blocks = vec![vec![salu(6, 6)], vec![salu(4, 6), valu_read(0, 4), branch], vec![MInst::new(Opc::SEndpgm, [], [])]];
        assert_eq!(waits(blocks), [vec![], vec![(1, SA)], vec![]]);
    }
}
