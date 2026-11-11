// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

//! One register per literal and block on gfx950. gfx9 encodes no literal in a
//! VOP3 instruction, so `legalize` reads each such literal from a register it
//! writes right before the instruction: a block that uses one literal many
//! times pays a move per use (the attention tunings' softmax selects -inf
//! and scales by 64.0 and -64 in every element). A move of an immediate into
//! a one-dword register written nowhere else goes when an earlier move of the
//! same class and value in the block wrote one too: its readers read that
//! register, which no instruction writes again and whose write comes first.
//!
//! That register then lives from the first move to the last reader, so a
//! move goes only within `SPAN` instructions of the one kept, and the next
//! one past that is kept in turn: kernels with many literals across a long
//! block (depthwise windows, sparse attention) went from three waves to two,
//! or six to five, with one register per literal for the whole block. A
//! function also keeps the shared moves only when neither class's peak
//! pressure (`regalloc::pressure`, before scheduling) reaches a higher
//! granule of eight registers.
//!
//! It runs only with `ROCKE_DIRECT_DEBUG=litshare` (or the lever `litshare`), as a measured variant:
//! with fewer instructions the schedule taken still ran one wave less in
//! eleven census objects (sparse attention, depthwise windows, grouped
//! dgrad), a trade measured only for the single-wave attention tunings.

use crate::rir::mir::*;
use crate::rir::regalloc::{pressure, Pressure};
use std::collections::HashMap;

/// The most instructions after a kept literal move that a later move of it
/// may go within.
const SPAN: usize = 32;

/// A move of an immediate into a whole one-dword virtual register.
fn literal_move(f: &MFunc, inst: &MInst) -> Option<(VRef, u32)> {
    if inst.op != Opc::Copy || inst.defs.len() != 1 || inst.uses.len() != 1 {
        return None;
    }
    match (inst.defs[0], inst.uses[0]) {
        (Opnd::R(d), Opnd::Imm(v)) => {
            let info = &f.vregs[d.reg as usize];
            (d.off == 0 && d.n == 1 && info.size == 1 && info.fixed.is_none()).then_some((d, v))
        }
        _ => None,
    }
}

/// With `ROCKE_DIRECT_DEBUG=litshare`, `share_with` of `f`.
pub fn share(f: &mut MFunc) -> usize {
    if !crate::rir::debug_flag("litshare") && !f.lever_anywhere("litshare") {
        return 0;
    }
    f.fill_scopes();
    share_with(f, crate::rir::debug_flag("litshare"))
}

/// Shares the literal moves of `f`'s blocks when that keeps its pressure's
/// granules; returns how many went.
fn share_with(f: &mut MFunc, all: bool) -> usize {
    let mut g = f.clone();
    let dropped = share_all(&mut g, all);
    if dropped == 0 {
        return 0;
    }
    let granules = |p: Pressure| (p.vgprs.div_ceil(8), p.sgprs.div_ceil(8));
    match (pressure(f), pressure(&g)) {
        (Ok(a), Ok(b)) if granules(b).0 <= granules(a).0 && granules(b).1 <= granules(a).1 => {
            *f = g;
            dropped
        }
        _ => 0,
    }
}

/// Shares the literal moves of `f`'s blocks; returns how many went.
/// `all`: every block's moves, else those whose scope has lever `litshare`.
fn share_all(f: &mut MFunc, all: bool) -> usize {
    let mut writes = vec![0u32; f.vregs.len()];
    for inst in f.blocks.iter().flat_map(|b| b.insts.iter()) {
        for d in inst.defs.iter() {
            if let Opnd::R(v) = d {
                writes[v.reg as usize] += 1;
            }
        }
    }
    // The register each dropped move's register is read as.
    let mut renames: HashMap<u32, VRef> = HashMap::new();
    let mut dropped = 0;
    for b in 0..f.blocks.len() {
        let mut kept: HashMap<(Class, u32), (VRef, usize)> = HashMap::new();
        let mut keep = Vec::with_capacity(f.blocks[b].insts.len());
        for (i, inst) in std::mem::take(&mut f.blocks[b].insts).into_iter().enumerate() {
            let shares = all || f.lever(&inst, "litshare");
            if let Some((d, v)) = literal_move(f, &inst).filter(|(d, _)| shares && writes[d.reg as usize] == 1) {
                let key = (f.vregs[d.reg as usize].class, v);
                match kept.get(&key) {
                    Some(&(r, at)) if i - at <= SPAN => {
                        renames.insert(d.reg, r);
                        dropped += 1;
                        continue;
                    }
                    _ => {
                        kept.insert(key, (d, i));
                    }
                }
            }
            keep.push(inst);
        }
        f.blocks[b].insts = keep;
    }
    if dropped == 0 {
        return 0;
    }
    for inst in f.blocks.iter_mut().flat_map(|b| b.insts.iter_mut()) {
        for u in inst.uses.iter_mut() {
            if let Opnd::R(v) = u {
                if let Some(&r) = renames.get(&v.reg) {
                    *u = Opnd::R(VRef { reg: r.reg, off: v.off, n: v.n });
                }
            }
        }
    }
    dropped
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::target::Target;

    const NEG_INF: u32 = 0xff80_0000;

    /// A block of `n` maxima `acc = max(acc, -inf)`, each reading -inf from a move of its own, as `legalize` leaves
    /// VOP3 literals; the accumulator starts from a move of 1.
    fn block(n: usize) -> MFunc {
        let mut f = MFunc { target: Target::Gfx950, ..MFunc::default() };
        let reg = |f: &mut MFunc| {
            f.vregs.push(VRegInfo { class: Class::V, size: 1, align: 1, fixed: None });
            Opnd::R(VRef { reg: f.vregs.len() as u32 - 1, off: 0, n: 1 })
        };
        let mut acc = reg(&mut f);
        let mut insts = vec![MInst::new(Opc::Copy, [acc], [Opnd::Imm(1)])];
        for _ in 0..n {
            let (lit, d) = (reg(&mut f), reg(&mut f));
            insts.push(MInst::new(Opc::Copy, [lit], [Opnd::Imm(NEG_INF)]));
            insts.push(MInst::new(Opc::VMaxF32, [d], [lit, acc]));
            acc = d;
        }
        f.blocks.push(MBlock { insts, loop_depth: 0 });
        f
    }

    fn moves(f: &MFunc) -> usize {
        f.blocks[0].insts.iter().filter(|i| i.op == Opc::Copy && i.uses[..] == [Opnd::Imm(NEG_INF)]).count()
    }

    #[test]
    fn a_block_writes_each_literal_once() {
        let mut f = block(8);
        assert_eq!(share_with(&mut f, true), 7);
        assert_eq!(moves(&f), 1);
        // Every maximum reads the one move's register.
        let kept = f.blocks[0].insts[1].defs[0];
        assert!(f.blocks[0].insts.iter().filter(|i| i.op == Opc::VMaxF32).all(|i| i.uses[0] == kept));
    }

    #[test]
    fn a_long_block_writes_a_literal_again_past_the_span() {
        let mut f = block(64);
        share_with(&mut f, true);
        let n = moves(&f);
        assert!(n > 1 && 4 * n <= 64, "{n} moves");
    }

    #[test]
    fn it_runs_only_when_asked() {
        if !crate::rir::debug_flag("litshare") && !crate::rir::lever("litshare") {
            let mut f = block(8);
            assert_eq!((share(&mut f), moves(&f)), (0, 8));
        }
    }
}
