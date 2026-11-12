// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

//! Rematerialization of loop-invariant addresses and masks on gfx950, before
//! scheduling. Front ends compute every per-lane offset a loop needs once, in
//! front of it (`base + 272`, `base + 544`, ... for each row a workgroup
//! copies, a column index per MFMA tile for a causal mask), and each of them
//! then occupies a register for the whole loop: 74 of the 309 dwords live at
//! the peak of the 256x128 dense attention kernel. LLVM computes them where
//! they are used, from the one register they share, which is why it does not
//! spill there.
//!
//! A one-dword VGPR value computed by one VALU instruction from constants,
//! scalars and VGPRs that are defined once (or never written: the thread id)
//! is computed again before each use that sits in a deeper loop than its
//! definition, when every VGPR it reads is live there anyway, and when its
//! other uses (the first tile's, in front of the loop) come before the first
//! of those. The definition goes when nothing else reads it. A chain
//! `a = t + 1; b = a + 272` is recomputed whole and costs the register of `t`
//! alone; the registers a chain reads that stay are the ones that must be
//! live at its uses. The spiller does the same
//! for what it must spill (`spill::pure_def`); this pass runs ahead of it
//! for a retry (`compile_remat`), which the caller keeps only when it spills
//! less or runs more waves.

use std::cell::Cell;

use super::mir::*;
use super::regalloc;
use crate::target::Target;

thread_local! {
    static ON: Cell<bool> = const { Cell::new(false) };
}

/// `f` with the pass on or off for compiles on this thread.
pub(super) fn with<T>(on: bool, f: impl FnOnce() -> T) -> T {
    let before = ON.with(|c| c.replace(on));
    let out = f();
    ON.with(|c| c.set(before));
    out
}

pub(super) fn enabled() -> bool {
    ON.with(|c| c.get())
}

thread_local! {
    static OFF: Cell<bool> = const { Cell::new(false) };
}

/// Runs `f` with no remat retry on this thread (tests of passes that the retry's outcome would mix in).
#[cfg(test)]
pub(crate) fn without<T>(f: impl FnOnce() -> T) -> T {
    let before = OFF.with(|c| c.replace(true));
    let out = f();
    OFF.with(|c| c.set(before));
    out
}

pub(super) fn off() -> bool {
    OFF.with(|c| c.get())
}

thread_local! {
    static CHANGED: Cell<usize> = const { Cell::new(0) };
}

/// How many recomputations `run` has made on this thread since `reset`.
pub(super) fn changed() -> usize {
    CHANGED.with(|c| c.get())
}

pub(super) fn reset() {
    CHANGED.with(|c| c.set(0));
}

/// Recomputes the candidates before their uses in deeper loops; returns how
/// many definitions went.
pub fn run(f: &mut MFunc) -> usize {
    if f.target != Target::Gfx950 || f.blocks.iter().all(|b| b.loop_depth == 0) {
        return 0;
    }
    let nv = f.vregs.len();
    let mut defs = vec![0u32; nv];
    let mut def_at: Vec<Option<(usize, usize)>> = vec![None; nv];
    let mut first = vec![0usize; f.blocks.len() + 1];
    for (b, block) in f.blocks.iter().enumerate() {
        first[b + 1] = first[b] + block.insts.len();
        for (k, inst) in block.insts.iter().enumerate() {
            for v in inst.defs.iter().filter_map(Opnd::vref) {
                defs[v.reg as usize] += 1;
                def_at[v.reg as usize] = Some((b, k));
            }
        }
    }
    // Candidates: one-dword VGPRs defined once by a pure VALU instruction.
    let mut rw = vec![false; nv];
    for r in 0..nv {
        let info = &f.vregs[r];
        if info.class != Class::V || info.size != 1 || info.fixed.is_some() || defs[r] != 1 {
            continue;
        }
        let Some((b, k)) = def_at[r] else { continue };
        let inst = &f.blocks[b].insts[k];
        rw[r] = inst.defs.len() == 1 && super::spill::pure_def(inst, Class::V, &f.vregs, &defs, true);
    }
    // Every reading instruction of each candidate (once), by (block, index).
    let mut uses_of: Vec<Vec<(usize, usize)>> = vec![Vec::new(); nv];
    for (b, block) in f.blocks.iter().enumerate() {
        for (k, inst) in block.insts.iter().enumerate() {
            for v in inst.uses.iter().filter_map(Opnd::vref) {
                let r = v.reg as usize;
                if rw[r] && uses_of[r].last() != Some(&(b, k)) {
                    uses_of[r].push((b, k));
                }
            }
        }
    }
    let trace = super::debug_flag("tracepressure");
    if trace {
        eprintln!("remat: {} structural candidates", rw.iter().filter(|&&c| c).count());
    }
    let depth = |b: usize| f.blocks[b].loop_depth;
    let at = |(b, k): (usize, usize)| first[b] + k;
    // Uses in deeper loops than the definition: where the value is recomputed.
    let deeper_of = |r: usize| -> Vec<(usize, usize)> {
        let (db, _) = def_at[r].expect("defined once");
        uses_of[r].iter().copied().filter(|&(ub, _)| depth(ub) > depth(db)).collect()
    };
    // The VGPRs a candidate's chain reads that stay (not recomputed themselves).
    let leaves = |rw: &[bool], r: usize| -> Vec<VRef> {
        let mut out: Vec<VRef> = Vec::new();
        let mut todo = vec![r];
        while let Some(c) = todo.pop() {
            let (b, k) = def_at[c].expect("defined once");
            for s in f.blocks[b].insts[k].uses.iter().filter_map(Opnd::vref) {
                if f.vregs[s.reg as usize].class != Class::V {
                    continue;
                }
                if rw[s.reg as usize] {
                    todo.push(s.reg as usize);
                } else if !out.contains(&s) {
                    out.push(s);
                }
            }
        }
        out
    };
    let Ok(live) = regalloc::analyze(f) else { return 0 };
    let live_at = |s: VRef, slot: usize| -> bool {
        (0..s.n as usize).all(|k| live.segs(live.base[s.reg as usize] + s.off as usize + k).iter().any(|&(a, e)| a as usize <= slot && slot <= e as usize))
    };
    // Where each candidate lands: its own deeper uses and, through the chains that read it, theirs.
    let sites_of = |rw: &[bool], r: usize| -> Vec<(usize, usize)> {
        let mut sites = deeper_of(r);
        let mut todo = vec![r];
        let mut seen = vec![r];
        while let Some(c) = todo.pop() {
            for &u in &uses_of[c] {
                if let [Opnd::R(d)] = f.blocks[u.0].insts[u.1].defs[..] {
                    let d = d.reg as usize;
                    if rw[d] && !seen.contains(&d) {
                        seen.push(d);
                        sites.extend(deeper_of(d));
                        todo.push(d);
                    }
                }
            }
        }
        sites
    };
    loop {
        let mut dropped = false;
        for r in 0..nv {
            if !rw[r] {
                continue;
            }
            let deeper = deeper_of(r);
            // Its other uses must come before the first one in a deeper loop, or the value is live through the loop anyway.
            let first_deeper = deeper.iter().map(|&u| at(u)).min();
            let early = first_deeper.is_some_and(|fd| uses_of[r].iter().all(|&u| depth(u.0) > depth(def_at[r].expect("defined once").0) || at(u) < fd));
            let mut ok = early;
            if ok {
                let ls = leaves(&rw, r);
                ok = sites_of(&rw, r).iter().all(|&(ub, uk)| ls.iter().all(|&s| live_at(s, 2 * at((ub, uk)))));
                if trace && !ok {
                    eprintln!("remat: v{r} dropped: a leaf is not live at a use ({ls:?})");
                }
            } else if trace {
                eprintln!("remat: v{r} dropped: a use outside deeper loops comes after the first deeper one (or there is none)");
            }
            if !ok {
                rw[r] = false;
                dropped = true;
            }
        }
        if !dropped {
            break;
        }
    }
    if !rw.iter().any(|&c| c) {
        return 0;
    }
    // A definition goes when every use not in a deeper loop is the definition of another that goes.
    let mut gone = rw.clone();
    loop {
        let mut dropped = false;
        for r in 0..nv {
            if !gone[r] {
                continue;
            }
            let (db, _) = def_at[r].expect("defined once");
            let all = uses_of[r].iter().all(|&(ub, uk)| {
                depth(ub) > depth(db) || matches!(f.blocks[ub].insts[uk].defs[..], [Opnd::R(d)] if gone[d.reg as usize])
            });
            if !all {
                gone[r] = false;
                dropped = true;
            }
        }
        if !dropped {
            break;
        }
    }
    // Rewrite: clones of the chain before each reading instruction in a deeper loop.
    let block_depths: Vec<u32> = f.blocks.iter().map(|b| b.loop_depth).collect();
    let mut def_of: Vec<Option<MInst>> = vec![None; nv];
    for r in 0..nv {
        if rw[r] {
            let (b, k) = def_at[r].expect("defined once");
            def_of[r] = Some(f.blocks[b].insts[k].clone());
        }
    }
    let mut new_regs: Vec<VRegInfo> = Vec::new();
    let mut removed = 0;
    // The recomputation of `r` in front of the current instruction, after those of what it reads.
    fn emit(
        r: usize,
        nv: usize,
        def_of: &[Option<MInst>],
        new_regs: &mut Vec<VRegInfo>,
        made: &mut Vec<(u32, u32)>,
        out: &mut Vec<MInst>,
    ) -> u32 {
        if let Some(m) = made.iter().find(|m| m.0 == r as u32) {
            return m.1;
        }
        let mut clone = def_of[r].clone().expect("candidate");
        for u in 0..clone.uses.len() {
            let Opnd::R(v) = clone.uses[u] else { continue };
            if (v.reg as usize) < nv && def_of[v.reg as usize].is_some() {
                let t = emit(v.reg as usize, nv, def_of, new_regs, made, out);
                clone.uses[u] = Opnd::R(VRef { reg: t, off: v.off, n: v.n });
            }
        }
        let t = (nv + new_regs.len()) as u32;
        new_regs.push(VRegInfo { class: Class::V, size: 1, align: 1, fixed: None });
        clone.defs = [Opnd::R(VRef { reg: t, off: 0, n: 1 })].into();
        out.push(clone);
        made.push((r as u32, t));
        t
    }
    for (b, block) in f.blocks.iter_mut().enumerate() {
        let old = std::mem::take(&mut block.insts);
        let mut out = Vec::with_capacity(old.len());
        for mut inst in old {
            if let [Opnd::R(d)] = inst.defs[..] {
                if (d.reg as usize) < nv && gone[d.reg as usize] {
                    removed += 1;
                    continue;
                }
            }
            let mut made: Vec<(u32, u32)> = Vec::new();
            for u in 0..inst.uses.len() {
                let Opnd::R(v) = inst.uses[u] else { continue };
                let r = v.reg as usize;
                // Only a use deeper than the definition is recomputed (its definition's instruction stays for the others).
                if r < nv && rw[r] && depth_of_def(&def_at, r, &block_depths) < block_depths[b] {
                    let t = emit(r, nv, &def_of, &mut new_regs, &mut made, &mut out);
                    inst.uses[u] = Opnd::R(VRef { reg: t, off: v.off, n: v.n });
                }
            }
            out.push(inst);
        }
        block.insts = out;
    }
    let new_regs_len = new_regs.len();
    f.vregs.extend(new_regs);
    let moved = new_regs_len;
    CHANGED.with(|c| c.set(c.get() + moved));
    removed
}

fn depth_of_def(def_at: &[Option<(usize, usize)>], r: usize, depths: &[u32]) -> u32 {
    depths[def_at[r].expect("defined once").0]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rir::spill::tests::effects;

    /// A gfx950 kernel: `n` addresses `tid + i` made up front, an accumulator updated in a loop (depth 1)
    /// by every one of them (a store through the thread id keeps `tid` live there), and a store after it.
    /// With `late`, the first address is also read after the loop.
    fn kernel(n: usize, late: bool) -> MFunc {
        let mut f = MFunc { target: Target::Gfx950, ..MFunc::default() };
        let reg = |f: &mut MFunc, class: Class, size: u8, fixed: Option<u16>| {
            f.vregs.push(VRegInfo { class, size, align: 1, fixed });
            VRef { reg: f.vregs.len() as u32 - 1, off: 0, n: size }
        };
        let r = Opnd::R;
        let tid = reg(&mut f, Class::V, 1, Some(0));
        let t = reg(&mut f, Class::V, 1, None);
        let mut entry = vec![MInst::new(Opc::Copy, [r(t)], [r(tid)])];
        let addrs: Vec<VRef> = (0..n)
            .map(|i| {
                let a = reg(&mut f, Class::V, 1, None);
                entry.push(MInst::new(Opc::VAddNcU32, [r(a)], [Opnd::Imm(272 * (i as u32 + 1)), r(t)]));
                a
            })
            .collect();
        let acc = reg(&mut f, Class::V, 1, None);
        entry.push(MInst::new(Opc::Copy, [r(acc)], [Opnd::Imm(0)]));
        let iv = reg(&mut f, Class::S, 1, None);
        entry.push(MInst::new(Opc::Copy, [r(iv)], [Opnd::Imm(0)]));
        let mut body = Vec::new();
        for &a in &addrs {
            let x = reg(&mut f, Class::V, 1, None);
            body.push(MInst::new(Opc::VAddF32, [r(x)], [r(a), r(acc)]));
            body.push(MInst::new(Opc::Copy, [r(acc)], [r(x)]));
        }
        body.push(MInst::new(Opc::GStoreB32, [], [r(t), r(acc)]));
        body.push(MInst::new(Opc::SAddI32, [r(iv)], [r(iv), Opnd::Imm(1)]));
        body.push(MInst::new(Opc::SCmpLtU32, [], [r(iv), Opnd::Imm(3)]));
        body.push(MInst::new(Opc::SCbranchScc1, [], []).with_imm(1));
        let mut exit = vec![MInst::new(Opc::GStoreB32, [], [r(t), r(acc)])];
        if late {
            exit.push(MInst::new(Opc::GStoreB32, [], [r(t), r(addrs[0])]));
        }
        exit.push(MInst::new(Opc::SEndpgm, [], []));
        f.blocks = vec![MBlock { insts: entry, loop_depth: 0 }, MBlock { insts: body, loop_depth: 1 }, MBlock { insts: exit, loop_depth: 0 }];
        f
    }

    #[test]
    fn addresses_read_only_in_a_loop_are_computed_there() {
        let mut f = kernel(12, false);
        let (before, live_before) = (effects(&f, 3), regalloc::pressure(&f).unwrap().vgprs);
        assert_eq!(run(&mut f), 12);
        assert_eq!(effects(&f, 3), before, "the same stores");
        let live_after = regalloc::pressure(&f).unwrap().vgprs;
        assert!(live_after + 8 <= live_before, "{live_before} -> {live_after}");
        // One recomputation per address in the loop, none left in front of it.
        assert_eq!(f.blocks[0].insts.iter().filter(|i| i.op == Opc::VAddNcU32).count(), 0);
        assert_eq!(f.blocks[1].insts.iter().filter(|i| i.op == Opc::VAddNcU32).count(), 12);
    }

    #[test]
    fn a_use_after_the_loop_keeps_the_definition_and_the_loop_still_recomputes() {
        let mut f = kernel(4, true);
        let before = effects(&f, 3);
        // The late use comes after the first one in the loop, so the value is live through it anyway.
        assert_eq!(run(&mut f), 3);
        assert_eq!(effects(&f, 3), before);
        assert_eq!(f.blocks[0].insts.iter().filter(|i| i.op == Opc::VAddNcU32).count(), 1);
    }

    #[test]
    fn a_source_that_is_not_live_in_the_loop_stops_the_recomputation() {
        let mut f = kernel(4, false);
        // Take the stores through `t` out of the loop and the exit: the address base dies in front of the loop.
        f.blocks[1].insts.retain(|i| i.op != Opc::GStoreB32);
        f.blocks[2].insts[0].uses[0] = Opnd::R(VRef { reg: 0, off: 0, n: 1 });
        let before = f.blocks[0].insts.len();
        assert_eq!(run(&mut f), 0);
        assert_eq!(f.blocks[0].insts.len(), before);
    }

    #[test]
    fn other_targets_and_loopless_functions_are_left_alone() {
        let mut f = kernel(4, false);
        f.target = Target::Gfx1151;
        assert_eq!(run(&mut f), 0);
        let mut f = kernel(4, false);
        f.blocks.iter_mut().for_each(|b| b.loop_depth = 0);
        assert_eq!(run(&mut f), 0);
    }
}
