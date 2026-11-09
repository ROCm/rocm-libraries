// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

//! The XNACK soft clause (H11, G1 design section 5.1) in gfx950's register
//! assignment. A run of VMEM (or SMEM) instructions is a clause that may
//! replay, so none of them may write a register one of them reads: the
//! hazard pass breaks the run with an `s_nop` where one does. The assignment
//! makes most of these: it gives a load's destination the lowest free
//! register, often the address register the load itself, or an earlier load
//! of its run, reads for the last time there.
//!
//! Allocation follows scheduling, so the runs are known: same-kind memory
//! instructions in a row in a block, pseudo instructions in between left out
//! (copies the assignment coalesces vanish, and so join the runs around them;
//! a copy that stays becomes a move, so a run predicted here may be broken
//! after all, which only costs a preference not needed). Waits and other
//! hazards' `s_nop`s break runs too, after allocation; the hazard pass sees
//! the same runs or shorter ones. Of a load's destination and a register its
//! run reads up to it, whichever the assignment places second (the
//! destination, in order of first definition; either, packing tuples first)
//! keeps off the other: where its first fit overlaps, it takes the lowest
//! free register below the most registers placed so far; with none, the
//! first fit stays and the hazard pass breaks the run as before. A register
//! taken that way can push later groups higher. Near the limit it can leave
//! a tuple no room: a round of spilling that does not fit with the
//! preference assigns without it.

use super::gen_isa as t;
use crate::rir::mir::*;
use crate::target::Target;
use std::sync::atomic::{AtomicBool, Ordering};

/// A register range of a clause: its group root, dword offset from the
/// root's register and dwords, the position in the clause of the
/// instruction naming it, and whether that instruction writes it.
#[derive(Clone, Copy)]
struct Range {
    root: u32,
    off: u32,
    n: u32,
    at: u32,
    write: bool,
}

/// The registers of a function's soft clauses: a register a load writes
/// should not be one the clause reads up to that load, whichever of the two
/// the assignment places first.
pub struct Clauses {
    /// Per root, the range of `entries` naming it.
    start: Vec<u32>,
    /// One of a root's ranges (its index in `ranges`) and the range of
    /// `ranges` its clause has.
    entries: Vec<(u32, u32, u32)>,
    ranges: Vec<Range>,
    /// Whether `prefer` gave a group another register than its first fit.
    moved: AtomicBool,
}

/// The kind of memory instruction the hazard pass runs clauses of, if any.
fn clause_kind(op: Opc) -> Option<bool> {
    match t::info(op).map(|i| i.fmt) {
        Some(Fmt::Global | Fmt::Mubuf) => Some(true),
        Some(Fmt::Smem) => Some(false),
        _ => None,
    }
}

/// The registers `run`, a clause of two or more instructions, names, in
/// order, each instruction's reads before its writes.
fn record(run: &[&MInst], group: &mut dyn FnMut(u32) -> (u32, u8), ranges: &mut Vec<Range>) {
    if run.len() < 2 {
        return;
    }
    for (at, inst) in run.iter().enumerate() {
        for (opnds, write) in [(&inst.uses[..], false), (&inst.defs[..], true)] {
            for o in opnds {
                if let Opnd::R(v) = *o {
                    let (root, off) = group(v.reg);
                    ranges.push(Range { root, off: off as u32 + v.off as u32, n: v.n as u32, at: at as u32, write });
                }
            }
        }
    }
}

impl Clauses {
    /// The loads of `f` (gfx950) in runs of two or more same-kind memory
    /// instructions; `group(v)` is a register's group root and dword offset.
    /// None when there are none, or with `ROCKE_DIRECT_DEBUG=noclauses`.
    pub fn of(f: &MFunc, group: &mut dyn FnMut(u32) -> (u32, u8)) -> Option<Clauses> {
        if f.target != Target::Gfx950 || crate::rir::debug_flag("noclauses") {
            return None;
        }
        let mut ranges: Vec<Range> = Vec::new();
        let mut runs: Vec<(u32, u32)> = Vec::new();
        for block in &f.blocks {
            let mut run: Vec<&MInst> = Vec::new();
            let mut kind = None;
            let mut close = |run: &mut Vec<&MInst>, ranges: &mut Vec<Range>| {
                let lo = ranges.len() as u32;
                record(run, group, ranges);
                if ranges.len() as u32 > lo {
                    runs.push((lo, ranges.len() as u32));
                }
                run.clear();
            };
            for inst in &block.insts {
                if matches!(t::info(inst.op).map(|i| i.fmt), None | Some(Fmt::Pseudo)) {
                    continue;
                }
                let k = clause_kind(inst.op);
                if k.is_none() || k != kind {
                    close(&mut run, &mut ranges);
                }
                kind = k;
                if k.is_some() {
                    run.push(inst);
                }
            }
            close(&mut run, &mut ranges);
        }
        if runs.is_empty() {
            return None;
        }
        let mut found: Vec<(u32, u32, u32, u32)> = Vec::with_capacity(ranges.len());
        for &(lo, hi) in &runs {
            found.extend((lo..hi).map(|x| (ranges[x as usize].root, x, lo, hi)));
        }
        found.sort_by_key(|e| e.0);
        let roots = found.last().map_or(0, |e| e.0 as usize + 1);
        let mut start = vec![0u32; roots + 1];
        for e in &found {
            start[e.0 as usize + 1] += 1;
        }
        for r in 0..roots {
            start[r + 1] += start[r];
        }
        let entries = found.into_iter().map(|(_, x, lo, hi)| (x, lo, hi)).collect();
        Some(Clauses { start, entries, ranges, moved: AtomicBool::new(false) })
    }

    /// Whether `prefer` has given a group another register than its first
    /// fit: until it has, an assignment is the one without the preference.
    pub fn moved(&self) -> bool {
        self.moved.load(Ordering::Relaxed)
    }

    /// The register for group `root` of `size` dwords and alignment `align`,
    /// whose first fit is `first`: `first`, or where
    /// a load would then write a register its clause reads (one of them the
    /// root's, the other a group's placed so far), the lowest register `free`
    /// allows that avoids every such overlap below `high`, the registers of
    /// the class the assignment uses so far with `first` taken. `placed` and
    /// `assign` give the groups placed so far and their registers, `class`
    /// whether a group is of the class.
    pub fn prefer(
        &self,
        (root, first, size, align, high): (u32, u32, u32, u32, u32),
        placed: &[bool],
        assign: &[u32],
        class: impl Fn(u32) -> bool,
        free: impl Fn(u32) -> bool,
    ) -> u32 {
        let entries = match self.start.get(root as usize + 1) {
            Some(&end) => &self.entries[self.start[root as usize] as usize..end as usize],
            None => return first,
        };
        if entries.is_empty() {
            return first;
        }
        let clash = |at: u32| {
            entries.iter().any(|&(x, lo, hi)| {
                let mine = self.ranges[x as usize];
                let (a, b) = (at + mine.off, at + mine.off + mine.n);
                self.ranges[lo as usize..hi as usize].iter().any(|o| {
                    let pair = if mine.write { !o.write && o.at <= mine.at } else { o.write && o.at >= mine.at };
                    let base = assign[o.root as usize] + o.off;
                    pair && o.root != root && placed[o.root as usize] && class(o.root) && base < b && a < base + o.n
                })
            })
        };
        if !clash(first) {
            return first;
        }
        let high = high.max(first + size);
        match (0..).map(|k| k * align).take_while(|&at| at + size <= high).find(|&at| free(at) && !clash(at)) {
            Some(at) => {
                self.moved.store(true, Ordering::Relaxed);
                at
            }
            None => first,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rir::regalloc::{allocate_within, Limits};
    use crate::target::gfx9::hazards::insert_nops;

    fn r(reg: u32, n: u8) -> Opnd {
        Opnd::R(VRef { reg, off: 0, n })
    }

    fn dword(reg: u32, k: u8) -> Opnd {
        Opnd::R(VRef { reg, off: k, n: 1 })
    }

    /// `pad` dwords live and dead before three 64-bit addresses are made,
    /// then two loads in a row, each reading an address that dies there, and
    /// their results summed and stored through the third. With `between`, a
    /// VALU instruction separates the loads.
    fn two_loads(pad: u8, between: bool, target: Target) -> MFunc {
        let mut f = MFunc { target, ..MFunc::default() };
        // 0 pad, 1 and 2 the loads' addresses, 3 and 4 their results, 5 the sum, 6 the store's address, 7 a pad dword.
        for (size, align) in [(pad, 2), (2, 2), (2, 2), (2, 2), (2, 2), (1, 1), (2, 2), (1, 1)] {
            f.vregs.push(VRegInfo { class: Class::V, size, align, fixed: None });
        }
        let mut insts = vec![
            MInst::new(Opc::ImplicitDef, [r(0, pad)], []),
            MInst::new(Opc::VAddF32, [r(7, 1)], [dword(0, 0), dword(0, pad - 1)]),
            MInst::new(Opc::ImplicitDef, [r(1, 2)], []),
            MInst::new(Opc::ImplicitDef, [r(2, 2)], []),
            MInst::new(Opc::ImplicitDef, [r(6, 2)], []),
            MInst::new(Opc::GLoadB64, [r(3, 2)], [r(1, 2)]),
        ];
        if between {
            insts.push(MInst::new(Opc::VAddF32, [r(7, 1)], [r(7, 1), r(7, 1)]));
        }
        insts.extend([
            MInst::new(Opc::GLoadB64, [r(4, 2)], [r(2, 2)]),
            MInst::new(Opc::VAddF32, [r(5, 1)], [dword(3, 0), dword(4, 1)]),
            MInst::new(Opc::VAddF32, [r(5, 1)], [r(5, 1), r(7, 1)]),
            MInst::new(Opc::GStoreB32, [], [r(6, 2), r(5, 1)]),
            MInst::new(Opc::SEndpgm, [], []),
        ]);
        f.blocks.push(MBlock { insts, loop_depth: 0 });
        f
    }

    /// The two loads' destinations and the `s_nop`s the hazard pass puts
    /// before the second load, after allocation.
    fn allocated(pad: u8) -> ([Opnd; 2], usize) {
        let mut f = two_loads(pad, false, Target::Gfx950);
        allocate_within(&mut f, Limits { vgprs: 64, sgprs: 32 }).unwrap_or_else(|e| panic!("{e}"));
        insert_nops(&mut f);
        let insts = &f.blocks[0].insts;
        let loads: Vec<usize> = (0..insts.len()).filter(|&i| insts[i].op == Opc::GLoadB64).collect();
        let nops = insts[loads[0]..loads[1]].iter().filter(|i| i.op == Opc::SNop).count();
        ([insts[loads[0]].defs[0], insts[loads[1]].defs[0]], nops)
    }

    fn overlap(a: Opnd, b: Opnd) -> bool {
        match (a, b) {
            (Opnd::P(Phys::V(x), n), Opnd::P(Phys::V(y), m)) => x < y + m as u16 && y < x + n as u16,
            _ => false,
        }
    }

    #[test]
    fn a_load_keeps_off_the_registers_its_clause_reads() {
        // Twelve dwords were live before: the first fit, each load's own
        // dying address, has free registers below the twelve to give way to.
        assert!(Clauses::of(&two_loads(12, false, Target::Gfx950), &mut |v| (v, 0)).is_some());
        let ([d0, d1], nops) = allocated(12);
        assert_eq!(nops, 0, "{d0:?} {d1:?}");
        assert!(!overlap(d0, d1));
    }

    #[test]
    fn with_no_register_free_below_the_count_the_clause_breaks() {
        // Eight dwords before: past the three addresses and a pad dword,
        // nothing below eight is free, so the first load takes its address
        // register and the hazard pass breaks the clause, as without this.
        let ([d0, _], nops) = allocated(8);
        assert_eq!(nops, 1, "{d0:?}");
    }

    #[test]
    fn an_address_placed_after_its_load_destination_keeps_off_it_too() {
        // The loads' destinations (3 and 4) are placed first, at 0 and 2: each
        // load's address (1 and 2), dying where its destination is born, gives
        // way to the next free register below the count (the first
        // destination is live while the second address is, so not free for
        // it), while the store's address (6, read outside the clause) takes
        // the first fit.
        let f = two_loads(12, false, Target::Gfx950);
        let c = Clauses::of(&f, &mut |v| (v, 0)).expect("a clause");
        let (mut placed, mut assign) = (vec![false; f.vregs.len()], vec![0u32; f.vregs.len()]);
        (placed[3], assign[3], placed[4], assign[4]) = (true, 0, true, 2);
        let any = |_: u32| true;
        assert_eq!(c.prefer((1, 0, 2, 2, 8), &placed, &assign, any, |p| p != 2), 4);
        assert_eq!(c.prefer((2, 2, 2, 2, 8), &placed, &assign, any, |p| p != 0), 4);
        assert_eq!(c.prefer((6, 0, 2, 2, 8), &placed, &assign, any, |_| true), 0);
        // Nothing free below the count: the first fit stays.
        assert_eq!(c.prefer((1, 0, 2, 2, 2), &placed, &assign, any, |_| true), 0);
        // A group of the other class is no clash.
        assert_eq!(c.prefer((1, 0, 2, 2, 8), &placed, &assign, |r| r != 3, |_| true), 0);
    }

    #[test]
    fn loads_apart_or_on_gfx1151_form_no_clause() {
        assert!(Clauses::of(&two_loads(12, true, Target::Gfx950), &mut |v| (v, 0)).is_none());
        assert!(Clauses::of(&two_loads(12, false, Target::Gfx1151), &mut |v| (v, 0)).is_none());
    }
}
