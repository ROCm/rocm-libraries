// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

//! Register allocation.
//!
//! The machine IR is in SSA form at dword granularity: every (virtual
//! register, dword) unit has one definition that dominates its uses, except
//! loop-carried registers (defined before the loop and again by the copy at
//! its latch, a phi web already resolved into copies) and tied operands
//! (accumulate-in-place WMMA, d16 loads), whose second definition is the
//! two-address form of a new value. Blocks are laid out with every block
//! after its dominators, so a value's first live slot is its definition.
//!
//! Liveness is per unit, as segments of that linear slot order. `pressure`
//! is the most units of a class live at one slot, a lower bound on the
//! registers of any assignment. `allocate` coalesces copies, then colors
//! register groups (a coalesced set of virtual registers at fixed dword
//! offsets) greedily in order of their first definition. For single-dword
//! values in SSA form this is coloring a chordal interference graph in a
//! perfect elimination order: whatever free register each value takes, the
//! registers in use at a definition are those of the values live there, so
//! `pressure` registers suffice and the choice among free ones is free to
//! serve dual issue. Tuples are placed to keep free space contiguous; when
//! a group still does not fit, the budget grows to the next register count
//! that keeps the occupancy, and only past the hardware limit does the
//! allocation fail.

use std::sync::atomic::{AtomicBool, Ordering};

use super::mir::*;
pub use super::spill::{SpillConfig, SpillStats};
use crate::error::CompileError;

/// SGPRs an allocation may hand out (s0..s105; VCC follows).
pub const MAX_SGPRS: u32 = 106;
pub const MAX_VGPRS: u32 = 256;

/// Instructions below which a second thread costs more than it saves.
pub(super) const PARALLEL_MIN_INSTS: usize = 4096;

#[derive(Clone)]
struct Bits(Vec<u64>);

impl Bits {
    fn new(n: usize) -> Self {
        Bits(vec![0; n.div_ceil(64)])
    }
    fn get(&self, i: usize) -> bool {
        self.0[i / 64] >> (i % 64) & 1 != 0
    }
    fn set(&mut self, i: usize) {
        self.0[i / 64] |= 1 << (i % 64);
    }
    fn clear(&mut self, i: usize) {
        self.0[i / 64] &= !(1 << (i % 64));
    }
    fn union_with(&mut self, other: &Bits) -> bool {
        let mut changed = false;
        for (a, b) in self.0.iter_mut().zip(&other.0) {
            let n = *a | *b;
            changed |= n != *a;
            *a = n;
        }
        changed
    }
    fn ones(&self) -> impl Iterator<Item = usize> + '_ {
        self.0.iter().enumerate().flat_map(|(w, &word)| {
            let mut bits = word;
            std::iter::from_fn(move || {
                (bits != 0).then(|| {
                    let b = bits.trailing_zeros() as usize;
                    bits &= bits - 1;
                    w * 64 + b
                })
            })
        })
    }
}

/// Inclusive slot range; instruction `i` reads at `2i` and writes at `2i+1`.
pub(super) type Seg = (u32, u32);

/// Appends `segs` to `out` sorted, with overlapping or adjacent ones joined.
fn normalize_into(segs: &mut [Seg], out: &mut Vec<Seg>) {
    segs.sort_unstable();
    let start = out.len();
    for &(s, e) in segs.iter() {
        let n = out.len();
        if n > start && s <= out[n - 1].1 + 1 {
            out[n - 1].1 = out[n - 1].1.max(e);
        } else {
            out.push((s, e));
        }
    }
}

fn overlaps(a: &[Seg], b: &[Seg]) -> bool {
    let (mut i, mut j) = (0, 0);
    while i < a.len() && j < b.len() {
        if a[i].1 < b[j].0 {
            i += 1;
        } else if b[j].1 < a[i].0 {
            j += 1;
        } else {
            return true;
        }
    }
    false
}

/// The live segments of one unit, normalized, as a range of the shared
/// segment table; coalescing appends merged lists to the table.
type UnitSegs = (u32, u32);

fn segs_of<'a>(table: &'a [Seg], &(start, len): &UnitSegs) -> &'a [Seg] {
    &table[start as usize..(start + len) as usize]
}

/// Slots occupied in each physical register of a class, one bit per slot.
/// Word `w` of every register is stored together, so probing successive
/// registers reads adjacent words.
struct Occupancy {
    regs: usize,
    bits: Vec<u64>,
    /// Per register, `stride` words with a bit for each word of `bits` that
    /// holds a taken slot, so long segments test 64 words at once.
    taken: Vec<u64>,
    stride: usize,
}

/// A segment's first and last word index with the masks of its bits there.
type Check = (u32, u32, u64, u64);

fn check_of((s, e): Seg) -> Check {
    (s / 64, e / 64, !0u64 << (s % 64), !0u64 >> (63 - e % 64))
}

/// Bit `j` set where `base + j` is a multiple of `align`.
fn aligned_positions(base: u32, align: u32) -> u64 {
    match align {
        0 | 1 => !0,
        2 if base % 2 == 0 => 0x5555_5555_5555_5555,
        4 if base % 4 == 0 => 0x1111_1111_1111_1111,
        _ => (0..64).filter(|&j| (base + j) % align == 0).fold(0, |m, j| m | 1 << j),
    }
}

impl Occupancy {
    fn new(regs: u32, slots: usize) -> Self {
        let words = slots.div_ceil(64).max(1);
        let stride = words.div_ceil(64);
        Occupancy {
            regs: regs as usize,
            bits: vec![0; regs as usize * words],
            taken: vec![0; regs as usize * stride],
            stride,
        }
    }

    fn free_at(&self, reg: usize, (first, last, low, high): Check) -> bool {
        let word = |w: u32| self.bits[w as usize * self.regs + reg];
        if first == last {
            return word(first) & low & high == 0;
        }
        word(first) & low == 0 && word(last) & high == 0 && self.empty_within(reg, first + 1, last)
    }

    /// Whether register `reg` holds no slot in words `from..to`.
    fn empty_within(&self, reg: usize, from: u32, to: u32) -> bool {
        if from >= to {
            return true;
        }
        let row = &self.taken[reg * self.stride..][..self.stride];
        let (a, b) = (from as usize, to as usize - 1);
        let (low, high) = (!0u64 << (a % 64), !0u64 >> (63 - b % 64));
        if a / 64 == b / 64 {
            return row[a / 64] & low & high == 0;
        }
        row[a / 64] & low == 0 && row[a / 64 + 1..b / 64].iter().all(|&x| x == 0) && row[b / 64] & high == 0
    }

    #[cfg(test)]
    fn free(&self, reg: u32, segs: &[Seg]) -> bool {
        assert!((reg as usize) < self.regs, "register {reg} out of range");
        segs.iter().all(|&seg| self.free_at(reg as usize, check_of(seg)))
    }

    /// Whether register `at + k` is free for every `(k, check)`.
    fn free_from(&self, at: u32, checks: &[(u32, Check)]) -> bool {
        checks.iter().all(|&(k, check)| self.free_at((at + k) as usize, check))
    }

    /// The lowest multiple `at` of `align` with `at + size <= limit` that is
    /// `free_from`, as trying each in turn finds it. Past the first few,
    /// positions go 64 at a time: the first and last word of every check
    /// rule most of them out together, and `free_from` decides the rest in
    /// order.
    fn first_free(&self, checks: &[(u32, Check)], size: u32, align: u32, limit: u32) -> Option<u32> {
        let last = limit.checked_sub(size)?;
        let mut at = 0;
        for _ in 0..4 {
            if at > last {
                return None;
            }
            if self.free_from(at, checks) {
                return Some(at);
            }
            at += align;
        }
        let mut base = at / 64 * 64;
        // Positions of the first chunk below `at` are tried.
        let mut untried = !0u64 << (at - base);
        while base <= last {
            let n = (last - base + 1).min(64);
            let mut ok = aligned_positions(base, align) & (!0u64 >> (64 - n)) & untried;
            untried = !0;
            for &(k, (first, end, low, high)) in checks {
                if ok == 0 {
                    break;
                }
                let start = (base + k) as usize;
                if first == end {
                    ok &= self.free_bits(first, low & high, start, n);
                } else {
                    ok &= self.free_bits(first, low, start, n);
                    if ok != 0 {
                        ok &= self.free_bits(end, high, start, n);
                    }
                }
            }
            while ok != 0 {
                let at = base + ok.trailing_zeros();
                if self.free_from(at, checks) {
                    return Some(at);
                }
                ok &= ok - 1;
            }
            base += 64;
        }
        None
    }

    /// Bit `j` (`j < n`) set where register `start + j` holds none of the
    /// slots of `mask` in word `w`.
    fn free_bits(&self, w: u32, mask: u64, start: usize, n: u32) -> u64 {
        let row = &self.bits[w as usize * self.regs + start..][..n as usize];
        let mut free = 0;
        for (j, &word) in row.iter().enumerate() {
            free |= u64::from(word & mask == 0) << j;
        }
        free
    }

    fn take(&mut self, reg: u32, segs: &[Seg]) {
        let reg = reg as usize;
        for &seg in segs {
            let (first, last, low, high) = check_of(seg);
            let regs = self.regs;
            let word = |w: u32| w as usize * regs + reg;
            if first == last {
                self.bits[word(first)] |= low & high;
            } else {
                self.bits[word(first)] |= low;
                for w in first + 1..last {
                    self.bits[word(w)] = !0;
                }
                self.bits[word(last)] |= high;
            }
            let row = &mut self.taken[reg * self.stride..][..self.stride];
            for w in first as usize..=last as usize {
                row[w / 64] |= 1 << (w % 64);
            }
        }
    }
}

/// Liveness of every (virtual register, dword) unit.
#[derive(Clone)]
pub(super) struct Live {
    /// First unit of each virtual register; the last entry is the unit count.
    pub(super) base: Vec<usize>,
    /// First instruction of each block in the linear order, then the total.
    pub(super) first: Vec<u32>,
    /// Each unit's live segments, as a range of `table`.
    unit_segs: Vec<UnitSegs>,
    table: Vec<Seg>,
}

impl Live {
    fn of(&self, o: &Opnd) -> Option<std::ops::Range<usize>> {
        match o {
            Opnd::R(v) => {
                let start = self.base[v.reg as usize] + v.off as usize;
                Some(start..start + v.n as usize)
            }
            _ => None,
        }
    }

    pub(super) fn slots(&self) -> usize {
        2 * *self.first.last().expect("non-empty") as usize
    }

    pub(super) fn segs(&self, unit: usize) -> &[Seg] {
        segs_of(&self.table, &self.unit_segs[unit])
    }

    /// Whether `other` has the same units, blocks and segments.
    pub(super) fn same_as(&self, other: &Live) -> bool {
        self.base == other.base && self.first == other.first && (0..*self.base.last().expect("non-empty")).all(|u| self.segs(u) == other.segs(u))
    }
}

/// Units live out of each block, by backward dataflow; the units and block
/// offsets come in `live`, whose segments are not yet filled.
fn dataflow(f: &MFunc) -> Result<(Live, LiveOut), CompileError> {
    let insts: usize = f.blocks.iter().map(|b| b.insts.len()).sum();
    let units: usize = f.vregs.iter().map(|r| r.size as usize).sum();
    // Sets of every unit per block cost little for a few blocks; across
    // many, only the units that can cross a block boundary get them.
    dataflow_with(f, f.blocks.len() * units.div_ceil(64) > insts)
}

/// `dataflow`, over the units that can cross a block boundary if `sparse`,
/// else over all of them; the result is the same.
fn dataflow_with(f: &MFunc, sparse: bool) -> Result<(Live, LiveOut), CompileError> {
    let nv = f.vregs.len();
    let mut base = vec![0usize; nv + 1];
    for i in 0..nv {
        base[i + 1] = base[i] + f.vregs[i].size as usize;
    }
    let nunits = base[nv];
    let nblocks = f.blocks.len();
    let mut first = vec![0u32; nblocks + 1];
    for b in 0..nblocks {
        first[b + 1] = first[b] + f.blocks[b].insts.len() as u32;
    }
    let live = Live { base, first, unit_segs: Vec::new(), table: Vec::new() };
    for block in &f.blocks {
        for inst in &block.insts {
            for o in inst.defs.iter().chain(&inst.uses) {
                if let Opnd::R(v) = o {
                    if v.off as u32 + v.n as u32 > f.vregs[v.reg as usize].size as u32 {
                        return Err(CompileError::new(
                            "ROCKE_DIRECT_INTERNAL",
                            format!("operand {}[{}:{}] is outside its register", v.reg, v.off, v.off + v.n - 1),
                        ));
                    }
                }
            }
        }
    }

    // Block-local upward-exposed uses and definitions; sparse, by unit
    // number among the crossing units (`unit_of` maps back).
    let (crossing, uses, defs, unit_of) = if sparse {
        crossing_sets(f, &live, nunits)
    } else {
        let mut uses = vec![Bits::new(nunits); nblocks];
        let mut defs = vec![Bits::new(nunits); nblocks];
        for (b, block) in f.blocks.iter().enumerate() {
            for inst in &block.insts {
                for o in &inst.uses {
                    if let Some(range) = live.of(o) {
                        for u in range {
                            if !defs[b].get(u) {
                                uses[b].set(u);
                            }
                        }
                    }
                }
                for o in &inst.defs {
                    if let Some(range) = live.of(o) {
                        for u in range {
                            defs[b].set(u);
                        }
                    }
                }
            }
        }
        (nunits, uses, defs, None)
    };
    let unit_of = |d: usize| unit_of.as_ref().map_or(d, |units| units[d] as usize);
    let succs: Vec<Vec<usize>> = (0..nblocks).map(|b| f.successors(b)).collect();
    let mut live_in = vec![Bits::new(crossing); nblocks];
    let mut live_out = vec![Bits::new(crossing); nblocks];
    let mut inb = Bits::new(crossing);
    loop {
        let mut changed = false;
        for b in (0..nblocks).rev() {
            for &s in &succs[b] {
                changed |= live_out[b].union_with(&live_in[s]);
            }
            for ((w, &out), &d) in inb.0.iter_mut().zip(&live_out[b].0).zip(&defs[b].0) {
                *w = out & !d;
            }
            inb.union_with(&uses[b]);
            if inb.0 != live_in[b].0 {
                std::mem::swap(&mut live_in[b], &mut inb);
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    if nblocks > 0 {
        for u in live_in[0].ones().map(unit_of) {
            let owner = live.base.partition_point(|&b| b <= u) - 1;
            if f.vregs[owner].fixed.is_none() {
                return Err(CompileError::new(
                    "ROCKE_DIRECT_INTERNAL",
                    format!("virtual register {owner} is used before it is defined"),
                ));
            }
        }
    }
    let mut out = LiveOut { start: Vec::with_capacity(nblocks + 1), units: Vec::new() };
    out.start.push(0);
    for bits in &live_out {
        out.units.extend(bits.ones().map(|d| unit_of(d) as u32));
        out.start.push(out.units.len() as u32);
    }
    Ok((live, out))
}

/// Per block, the upward-exposed uses and the definitions of the units some
/// block reads before writing (only those can be live across a block
/// boundary), numbered densely in unit order; and the unit of each number.
fn crossing_sets(f: &MFunc, live: &Live, nunits: usize) -> (usize, Vec<Bits>, Vec<Bits>, Option<Vec<u32>>) {
    let nblocks = f.blocks.len();
    // Block + 1 marks a unit defined, or recorded as exposed, in the block
    // at hand; the uses and definitions go in (block, unit) pairs.
    let (mut defined, mut exposed) = (vec![0u32; nunits], vec![0u32; nunits]);
    let (mut ue, mut de): (Vec<(u32, u32)>, Vec<(u32, u32)>) = (Vec::new(), Vec::new());
    for (b, block) in f.blocks.iter().enumerate() {
        let mark = b as u32 + 1;
        for inst in &block.insts {
            for o in &inst.uses {
                if let Some(range) = live.of(o) {
                    for u in range {
                        if defined[u] != mark && exposed[u] != mark {
                            exposed[u] = mark;
                            ue.push((b as u32, u as u32));
                        }
                    }
                }
            }
            for o in &inst.defs {
                if let Some(range) = live.of(o) {
                    for u in range {
                        if defined[u] != mark {
                            defined[u] = mark;
                            de.push((b as u32, u as u32));
                        }
                    }
                }
            }
        }
    }
    let mut dense = exposed;
    dense.fill(u32::MAX);
    for &(_, u) in &ue {
        dense[u as usize] = 0;
    }
    let mut unit_of = Vec::new();
    for (u, d) in dense.iter_mut().enumerate() {
        if *d == 0 {
            *d = unit_of.len() as u32;
            unit_of.push(u as u32);
        }
    }
    let crossing = unit_of.len();
    let mut uses = vec![Bits::new(crossing); nblocks];
    let mut defs = vec![Bits::new(crossing); nblocks];
    for &(b, u) in &ue {
        uses[b as usize].set(dense[u as usize] as usize);
    }
    for &(b, u) in &de {
        if dense[u as usize] != u32::MAX {
            defs[b as usize].set(dense[u as usize] as usize);
        }
    }
    (crossing, uses, defs, Some(unit_of))
}

/// The units live out of each block, in increasing order.
#[derive(Debug, PartialEq, Eq)]
pub(super) struct LiveOut {
    start: Vec<u32>,
    units: Vec<u32>,
}

impl LiveOut {
    fn of(&self, block: usize) -> impl Iterator<Item = usize> + '_ {
        self.units[self.start[block] as usize..self.start[block + 1] as usize].iter().map(|&u| u as usize)
    }

    /// These sets without the units `dropped` marks and with `unit` live out
    /// of exactly the blocks its flags mark.
    pub(super) fn edited(&self, dropped: &[bool], unit: Option<(usize, &[bool])>) -> LiveOut {
        let mut out = LiveOut { start: Vec::with_capacity(self.start.len()), units: Vec::with_capacity(self.units.len() + 8) };
        out.start.push(0);
        for b in 0..self.start.len() - 1 {
            let at = out.units.len();
            let kept = |&u: &usize| !dropped.get(u).copied().unwrap_or(false) && unit.is_none_or(|(x, _)| x != u);
            out.units.extend(self.of(b).filter(kept).map(|u| u as u32));
            if let Some((x, _)) = unit.filter(|(_, live)| live[b]) {
                let pos = out.units[at..].partition_point(|&u| (u as usize) < x);
                out.units.insert(at + pos, x as u32);
            }
            out.start.push(out.units.len() as u32);
        }
        out
    }
}

/// Buffers `analyze_reusing` fills, kept between analyses of one function as
/// it changes: refilling memory in use costs a fraction of fresh memory.
#[derive(Default)]
pub(super) struct LiveScratch {
    raw: Vec<(u32, Seg)>,
    end: Vec<u32>,
    by_unit: Vec<u32>,
    sorted: Vec<Seg>,
    fill: Vec<u32>,
    /// The tables of a liveness no longer needed.
    table: Vec<Seg>,
    unit_segs: Vec<UnitSegs>,
}

impl LiveScratch {
    /// Keeps the tables of `live` for the next analysis.
    pub(super) fn recycle(&mut self, live: Live) {
        self.table = live.table;
        self.unit_segs = live.unit_segs;
    }
}

/// Per-unit liveness by backward dataflow over the blocks, then the live
/// segments of each unit in linear slot order.
pub(super) fn analyze(f: &MFunc) -> Result<Live, CompileError> {
    analyze_reusing(f, &mut LiveScratch::default())
}

/// `analyze`, in the buffers of `scratch`.
pub(super) fn analyze_reusing(f: &MFunc, scratch: &mut LiveScratch) -> Result<Live, CompileError> {
    analyze_with_live_out(f, scratch).map(|(live, _)| live)
}

/// `analyze_reusing`, with the units live out of each block.
pub(super) fn analyze_with_live_out(f: &MFunc, scratch: &mut LiveScratch) -> Result<(Live, LiveOut), CompileError> {
    let (live, live_out) = dataflow(f)?;
    Ok((fill_segments(f, live, &live_out, scratch), live_out))
}

/// `analyze_reusing` of a function whose units live out of each block are
/// `live_out`, without the dataflow (nor its checks of the operands).
pub(super) fn analyze_given(f: &MFunc, live_out: &LiveOut, scratch: &mut LiveScratch) -> Live {
    let nv = f.vregs.len();
    let mut base = vec![0usize; nv + 1];
    for i in 0..nv {
        base[i + 1] = base[i] + f.vregs[i].size as usize;
    }
    let mut first = vec![0u32; f.blocks.len() + 1];
    for b in 0..f.blocks.len() {
        first[b + 1] = first[b] + f.blocks[b].insts.len() as u32;
    }
    fill_segments(f, Live { base, first, unit_segs: Vec::new(), table: Vec::new() }, live_out, scratch)
}

/// The live segments of each unit of `live` in linear slot order, given
/// the units live out of each block.
fn fill_segments(f: &MFunc, mut live: Live, live_out: &LiveOut, scratch: &mut LiveScratch) -> Live {
    let nunits = *live.base.last().expect("non-empty");
    let nblocks = f.blocks.len();
    // Live segments per unit, gathered as (unit, segment) and then sorted
    // into one table of normalized per-unit runs.
    let first = &live.first;
    let LiveScratch { raw, end, by_unit, sorted, fill, table, unit_segs } = scratch;
    raw.clear();
    raw.reserve(2 * first[nblocks] as usize);
    end.clear();
    end.resize(nunits, 0);
    let mut cur = Bits::new(nunits);
    for (b, block) in f.blocks.iter().enumerate() {
        if block.insts.is_empty() {
            continue;
        }
        let start_slot = 2 * first[b];
        let end_slot = 2 * first[b + 1] - 1;
        cur.0.fill(0);
        for u in live_out.of(b) {
            cur.set(u);
            end[u] = end_slot;
        }
        for (k, inst) in block.insts.iter().enumerate().rev() {
            let i = first[b] + k as u32;
            let dslot = if inst.op.flags() & F_EARLY != 0 { 2 * i } else { 2 * i + 1 };
            for o in &inst.defs {
                if let Some(range) = live.of(o) {
                    for u in range {
                        if cur.get(u) {
                            raw.push((u as u32, (dslot, end[u])));
                            cur.clear(u);
                        } else {
                            raw.push((u as u32, (dslot, dslot)));
                        }
                    }
                }
            }
            for o in &inst.uses {
                if let Some(range) = live.of(o) {
                    for u in range {
                        if !cur.get(u) {
                            cur.set(u);
                            end[u] = 2 * i;
                        }
                    }
                }
            }
        }
        for u in cur.ones() {
            raw.push((u as u32, (start_slot, end[u])));
        }
    }
    by_unit.clear();
    by_unit.resize(nunits + 1, 0);
    for &(u, _) in raw.iter() {
        by_unit[u as usize + 1] += 1;
    }
    for u in 0..nunits {
        by_unit[u + 1] += by_unit[u];
    }
    sorted.clear();
    sorted.resize(raw.len(), (0, 0));
    fill.clear();
    fill.extend_from_slice(by_unit);
    for &(u, seg) in raw.iter() {
        sorted[fill[u as usize] as usize] = seg;
        fill[u as usize] += 1;
    }
    let mut table = std::mem::take(table);
    table.clear();
    table.reserve(raw.len());
    let mut unit_segs = std::mem::take(unit_segs);
    unit_segs.clear();
    unit_segs.resize(nunits, (0, 0));
    for u in 0..nunits {
        let run = &mut sorted[by_unit[u] as usize..by_unit[u + 1] as usize];
        if !run.is_empty() {
            let start = table.len();
            normalize_into(run, &mut table);
            unit_segs[u] = (start as u32, (table.len() - start) as u32);
        }
    }
    live.unit_segs = unit_segs;
    live.table = table;
    live
}

/// The most units of each register class live at one slot.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Pressure {
    pub vgprs: u32,
    pub sgprs: u32,
}

/// Units of `class` live at each slot.
pub(super) fn profile(f: &MFunc, live: &Live, class: Class) -> Vec<u32> {
    let slots = live.slots();
    let mut delta = vec![0i32; slots + 1];
    for v in (0..f.vregs.len()).filter(|&v| f.vregs[v].class == class) {
        for u in live.base[v]..live.base[v + 1] {
            for &(s, e) in live.segs(u) {
                delta[s as usize] += 1;
                delta[e as usize + 1] -= 1;
            }
        }
    }
    let mut level = 0i32;
    delta.truncate(slots);
    delta
        .into_iter()
        .map(|d| {
            level += d;
            level as u32
        })
        .collect()
}

fn pressure_of(f: &MFunc, live: &Live) -> Pressure {
    let most = |class| profile(f, live, class).into_iter().max().unwrap_or(0);
    Pressure { vgprs: most(Class::V), sgprs: most(Class::S) }
}

/// The most VGPR and SGPR dwords live at one point of `f`: every assignment
/// uses at least that many registers. Counts the live units slot by slot,
/// as `profile` does over `analyze`'s segments, without building them.
pub fn pressure(f: &MFunc) -> Result<Pressure, CompileError> {
    Ok(Liveness::new(f)?.pressure(f, None))
}

/// What `pressure` reads of a function besides the order of each block's
/// instructions: the units of each register, the block offsets, the units
/// live out of each block, and which units are VGPRs. Scheduling and load
/// sinking change only that order, and keep the order of every unit's
/// definitions and uses within it, so each block reads the same units before
/// writing them and these hold for every schedule of the function.
pub(super) struct Liveness {
    live: Live,
    live_out: LiveOut,
    is_v: Bits,
}

impl Liveness {
    pub(super) fn new(f: &MFunc) -> Result<Liveness, CompileError> {
        let (live, live_out) = dataflow(f)?;
        let nunits = *live.base.last().expect("non-empty");
        let mut is_v = Bits::new(nunits);
        for v in (0..f.vregs.len()).filter(|&v| f.vregs[v].class == Class::V) {
            for u in live.base[v]..live.base[v + 1] {
                is_v.set(u);
            }
        }
        Ok(Liveness { live, live_out, is_v })
    }

    /// `pressure` of `f`, a reordering of the function these are of, with
    /// the instructions of each block in `orders`, or as they stand.
    pub(super) fn pressure(&self, f: &MFunc, orders: Option<&[Vec<u32>]>) -> Pressure {
        pressure_in(f, &self.live, &self.live_out, &self.is_v, orders)
    }

    /// The units each instruction of `f`, the function these are of, writes
    /// and reads.
    pub(super) fn unit_lists(&self, f: &MFunc) -> UnitLists {
        let n: usize = f.blocks.iter().map(|b| b.insts.len()).sum();
        let mut lists = UnitLists { start: Vec::with_capacity(n + 1), uses: Vec::with_capacity(n), early: Vec::with_capacity(n), units: Vec::with_capacity(3 * n) };
        let base = &self.live.base;
        // A register's units are of its class (`is_v`).
        let push = |units: &mut Vec<u32>, opnds: &[Opnd]| {
            for o in opnds {
                if let Opnd::R(v) = o {
                    let start = base[v.reg as usize] + v.off as usize;
                    let sgpr = u32::from(f.vregs[v.reg as usize].class != Class::V);
                    units.extend((start..start + v.n as usize).map(|u| (u as u32) << 1 | sgpr));
                }
            }
        };
        for inst in f.blocks.iter().flat_map(|b| b.insts.iter()) {
            lists.start.push(lists.units.len() as u32);
            push(&mut lists.units, &inst.defs);
            lists.uses.push(lists.units.len() as u32);
            push(&mut lists.units, &inst.uses);
            lists.early.push(inst.op.flags() & F_EARLY != 0);
        }
        lists.start.push(lists.units.len() as u32);
        lists
    }

    /// `pressure` of the function `lists` are of with the instructions of
    /// each block in `orders`, or as they stand: `pressure_in` over the lists.
    pub(super) fn pressure_listed(&self, lists: &UnitLists, orders: Option<&[Vec<u32>]>) -> Pressure {
        let nunits = *self.live.base.last().expect("non-empty");
        let mut cur = Bits::new(nunits);
        let mut most = Pressure::default();
        for b in 0..self.live.first.len() - 1 {
            let p = self.block_pressure_listed(lists, b, orders.map(|orders| &orders[b][..]), &mut cur);
            most = Pressure { vgprs: most.vgprs.max(p.vgprs), sgprs: most.sgprs.max(p.sgprs) };
        }
        most
    }

    /// `pressure_listed` with block `b` in `order_of(b)`, taking the blocks
    /// largest first, or `None` as soon as one has more units of a class live
    /// at once than `cap` allows: the function as a whole then has too.
    pub(super) fn pressure_capped(
        &self,
        lists: &UnitLists,
        cap: Limits,
        mut order_of: impl FnMut(usize) -> Vec<u32>,
    ) -> Option<(Vec<Vec<u32>>, Pressure)> {
        let nblocks = self.live.first.len() - 1;
        let mut by_size: Vec<usize> = (0..nblocks).collect();
        by_size.sort_by_key(|&b| std::cmp::Reverse(self.live.first[b + 1] - self.live.first[b]));
        let mut cur = Bits::new(*self.live.base.last().expect("non-empty"));
        let (mut orders, mut most) = (vec![Vec::new(); nblocks], Pressure::default());
        for b in by_size {
            orders[b] = order_of(b);
            let p = self.block_pressure_listed(lists, b, Some(&orders[b]), &mut cur);
            most = Pressure { vgprs: most.vgprs.max(p.vgprs), sgprs: most.sgprs.max(p.sgprs) };
            if most.vgprs > cap.vgprs || most.sgprs > cap.sgprs {
                return None;
            }
        }
        Some((orders, most))
    }

    /// `pressure_listed` of block `b` alone, with its instructions in
    /// `order`, or as they stand; `cur` is a set of every unit to work in.
    fn block_pressure_listed(&self, lists: &UnitLists, b: usize, order: Option<&[u32]>, cur: &mut Bits) -> Pressure {
        let (live, is_v) = (&self.live, &self.is_v);
        let mut most = Pressure::default();
        cur.0.fill(0);
        let mut n = [0u32; 2];
        for u in self.live_out.of(b) {
            cur.set(u);
            n[!is_v.get(u) as usize] += 1;
        }
        let first = live.first[b] as usize;
        for k in (0..live.first[b + 1] as usize - first).rev() {
            let i = first + order.map_or(k, |order| order[k] as usize);
            let (start, uses, end) = (lists.start[i] as usize, lists.uses[i] as usize, lists.start[i + 1] as usize);
            let (defs, used) = (&lists.units[start..uses], &lists.units[uses..end]);
            let unlive = |cur: &Bits, extra: &mut [u32; 2]| {
                for &x in defs {
                    if !cur.get((x >> 1) as usize) {
                        extra[(x & 1) as usize] += 1;
                    }
                }
            };
            // Write slot: the units live after, plus results nothing reads.
            let mut extra = [0u32; 2];
            if !lists.early[i] {
                unlive(cur, &mut extra);
            }
            most.vgprs = most.vgprs.max(n[0] + extra[0]);
            most.sgprs = most.sgprs.max(n[1] + extra[1]);
            for &x in defs {
                if cur.get((x >> 1) as usize) {
                    cur.clear((x >> 1) as usize);
                    n[(x & 1) as usize] -= 1;
                }
            }
            for &x in used {
                if !cur.get((x >> 1) as usize) {
                    cur.set((x >> 1) as usize);
                    n[(x & 1) as usize] += 1;
                }
            }
            // Read slot: the units live before, plus early-clobber results.
            let mut extra = [0u32; 2];
            if lists.early[i] {
                unlive(cur, &mut extra);
            }
            most.vgprs = most.vgprs.max(n[0] + extra[0]);
            most.sgprs = most.sgprs.max(n[1] + extra[1]);
        }
        most
    }
}

/// The units each instruction of a function writes and then reads, in
/// operand order, as `unit << 1` with the low bit set for an SGPR unit;
/// whether it writes early. Built once for the walks of all of the
/// function's schedules (`Liveness::pressure_listed`).
pub(super) struct UnitLists {
    /// Per instruction, in block order: where its units start; one more
    /// entry ends the last.
    start: Vec<u32>,
    /// Where each instruction's read units start.
    uses: Vec<u32>,
    early: Vec<bool>,
    units: Vec<u32>,
}

/// `Liveness::pressure`.
fn pressure_in(f: &MFunc, live: &Live, live_out: &LiveOut, is_v: &Bits, orders: Option<&[Vec<u32>]>) -> Pressure {
    let nunits = *live.base.last().expect("non-empty");
    let mut most = Pressure::default();
    let mut cur = Bits::new(nunits);
    for (b, block) in f.blocks.iter().enumerate() {
        cur.0.fill(0);
        let mut n = [0u32; 2];
        for u in live_out.of(b) {
            cur.set(u);
            n[!is_v.get(u) as usize] += 1;
        }
        for k in (0..block.insts.len()).rev() {
            let inst = &block.insts[orders.map_or(k, |orders| orders[b][k] as usize)];
            let early = inst.op.flags() & F_EARLY != 0;
            let units = |o: &Opnd| live.of(o).into_iter().flatten();
            // Write slot: the units live after, plus results nothing reads.
            let mut extra = [0u32; 2];
            if !early {
                for u in inst.defs.iter().flat_map(units) {
                    if !cur.get(u) {
                        extra[!is_v.get(u) as usize] += 1;
                    }
                }
            }
            most.vgprs = most.vgprs.max(n[0] + extra[0]);
            most.sgprs = most.sgprs.max(n[1] + extra[1]);
            for u in inst.defs.iter().flat_map(units) {
                if cur.get(u) {
                    cur.clear(u);
                    n[!is_v.get(u) as usize] -= 1;
                }
            }
            for u in inst.uses.iter().flat_map(units) {
                if !cur.get(u) {
                    cur.set(u);
                    n[!is_v.get(u) as usize] += 1;
                }
            }
            // Read slot: the units live before, plus early-clobber results.
            let mut extra = [0u32; 2];
            if early {
                for u in inst.defs.iter().flat_map(units) {
                    if !cur.get(u) {
                        extra[!is_v.get(u) as usize] += 1;
                    }
                }
            }
            most.vgprs = most.vgprs.max(n[0] + extra[0]);
            most.sgprs = most.sgprs.max(n[1] + extra[1]);
        }
    }
    most
}

/// Registers an allocation may use: VGPRs past `vgprs` would lower the
/// occupancy. Assignment stays within the pressure when it can and never
/// exceeds these.
#[derive(Clone, Copy, Debug)]
pub struct Limits {
    pub vgprs: u32,
    pub sgprs: u32,
}

impl Default for Limits {
    fn default() -> Self {
        Limits { vgprs: MAX_VGPRS, sgprs: MAX_SGPRS }
    }
}

impl Limits {
    /// Every register of `t` an allocation may hand out.
    pub fn for_target(t: &crate::target::TargetInfo) -> Self {
        Limits { vgprs: t.max_vgprs, sgprs: t.max_sgprs }
    }

    /// Every register an allocation of `f` may hand out: its target's, or
    /// fewer VGPRs under the function's `vgpr_limit`.
    pub fn for_func(f: &MFunc) -> Self {
        let hw = Limits::for_target(f.target.info());
        Limits { vgprs: f.vgpr_limit.map_or(hw.vgprs, |v| v.clamp(1, hw.vgprs)), ..hw }
    }
}

#[derive(Clone, Debug, Default)]
pub struct Allocation {
    pub num_sgprs: u32,
    pub num_vgprs: u32,
    /// The pressure of the function as allocated, where spilling measured
    /// it (the caller knows it otherwise).
    pub pressure: Pressure,
    pub spill: SpillStats,
    /// Only packing tuples first kept the assignment within the hardware.
    pub tight: bool,
}

impl Allocation {
    /// The registers occupancy counts: the VGPRs and, from the next
    /// multiple of four, the AGPRs spills took (gfx950).
    pub fn unified_vgprs(&self) -> u32 {
        match self.spill.agpr_dwords {
            0 => self.num_vgprs,
            agprs => self.num_vgprs.next_multiple_of(4) + agprs,
        }
    }
}

pub(super) fn find(parent: &mut [(u32, u8)], v: u32) -> (u32, u8) {
    let (p, off) = parent[v as usize];
    if p == v {
        return (v, 0);
    }
    let (root, poff) = find(parent, p);
    parent[v as usize] = (root, off + poff);
    (root, off + poff)
}

/// Coalesced register groups: each virtual register maps to a root at a
/// dword offset; a root's group is `gsize` dwords with the strictest
/// alignment and the precolor of its members.
pub(super) struct Groups {
    pub(super) parent: Vec<(u32, u8)>,
    pub(super) gsize: Vec<u32>,
    galign: Vec<u8>,
    pub(super) gfixed: Vec<Option<u16>>,
}

/// Merges copy-related groups whose units are never live at once, hottest
/// copies first; a merged group's units get the union of the segments.
pub(super) fn coalesce(f: &MFunc, live: &mut Live) -> Groups {
    let nv = f.vregs.len();
    let mut g = Groups {
        parent: (0..nv as u32).map(|v| (v, 0)).collect(),
        gsize: f.vregs.iter().map(|r| r.size as u32).collect(),
        galign: f.vregs.iter().map(|r| r.align).collect(),
        gfixed: f.vregs.iter().map(|r| r.fixed).collect(),
    };
    let mut copies: Vec<(u32, usize, usize)> = Vec::new();
    for (b, block) in f.blocks.iter().enumerate() {
        for (k, inst) in block.insts.iter().enumerate() {
            if inst.op == Opc::Copy {
                copies.push((block.loop_depth, b, k));
            }
        }
    }
    copies.sort_by(|a, b| b.0.cmp(&a.0).then((a.1, a.2).cmp(&(b.1, b.2))));
    if super::debug_flag("nocoalesce") {
        copies.clear();
    }
    let base = live.base.clone();
    let group_unit = |v: u32, k: u32| base[v as usize] + k as usize;
    let mut both: Vec<Seg> = Vec::new();
    for &(_, b, k) in &copies {
        let inst = &f.blocks[b].insts[k];
        let (Opnd::R(d), Opnd::R(s)) = (inst.defs[0], inst.uses[0]) else {
            continue;
        };
        if f.vregs[d.reg as usize].class != f.vregs[s.reg as usize].class {
            continue;
        }
        let (rd, od) = find(&mut g.parent, d.reg);
        let (rs, os) = find(&mut g.parent, s.reg);
        let od = od as i32 + d.off as i32;
        let os = os as i32 + s.off as i32;
        if rd == rs {
            continue;
        }
        let try_merge = |x: u32, y: u32, delta: i32, g: &Groups, live: &Live| -> bool {
            let sx = g.gsize[x as usize] as i32;
            let sy = g.gsize[y as usize] as i32;
            if delta < 0 || delta + sx > sy || g.gfixed[x as usize].is_some() {
                return false;
            }
            let ax = g.galign[x as usize] as i32;
            if delta % ax != 0 {
                return false;
            }
            if let Some(p) = g.gfixed[y as usize] {
                if (p as i32 + delta) % ax != 0 {
                    return false;
                }
            }
            (0..sx as u32).all(|k| {
                let a = live.segs(group_unit(x, k));
                let b = live.segs(group_unit(y, k + delta as u32));
                !overlaps(a, b)
            })
        };
        let (x, y, delta) = if try_merge(rs, rd, od - os, &g, live) {
            (rs, rd, od - os)
        } else if try_merge(rd, rs, os - od, &g, live) {
            (rd, rs, os - od)
        } else {
            continue;
        };
        for k in 0..g.gsize[x as usize] {
            let from = group_unit(x, k);
            let to = group_unit(y, k + delta as u32);
            both.clear();
            both.extend_from_slice(live.segs(to));
            both.extend_from_slice(live.segs(from));
            let start = live.table.len();
            normalize_into(&mut both, &mut live.table);
            live.unit_segs[to] = (start as u32, (live.table.len() - start) as u32);
            live.unit_segs[from] = (0, 0);
        }
        g.gsize[x as usize] = 0;
        g.galign[y as usize] = g.galign[y as usize].max(g.galign[x as usize]);
        if g.gfixed[y as usize].is_none() {
            g.gfixed[y as usize] = g.gfixed[x as usize];
        }
        g.parent[x as usize] = (y, delta as u8);
    }
    g
}

/// Allocates within the hardware limits, without spilling.
pub fn allocate(f: &mut MFunc) -> Result<Allocation, CompileError> {
    allocate_within(f, Limits::for_func(f))
}

/// Allocates registers for `f` without spilling. `budget` is the register
/// count that keeps the occupancy: when the first fit needs more, a second
/// assignment packs tuples first, and the one using fewer VGPRs is kept.
/// The result may exceed the budget; only past the hardware does this fail.
pub fn allocate_within(f: &mut MFunc, budget: Limits) -> Result<Allocation, CompileError> {
    let live = analyze(f)?;
    allocate_analyzed(f, live, budget)
}

/// `allocate_within` of `f`, a reordering of the function `known` is of,
/// without the dataflow: the units live out of each block are `known`'s.
pub(super) fn allocate_within_known(f: &mut MFunc, budget: Limits, known: &Liveness) -> Result<Allocation, CompileError> {
    let live = analyze_given(f, &known.live_out, &mut LiveScratch::default());
    allocate_analyzed(f, live, budget)
}

/// The rest of `allocate_within` once `f`'s liveness is `live`.
fn allocate_analyzed(f: &mut MFunc, mut live: Live, budget: Limits) -> Result<Allocation, CompileError> {
    let groups = coalesce(f, &mut live);
    let hw = Limits::for_func(f);
    let hard = Limits { vgprs: budget.vgprs.max(hw.vgprs), sgprs: budget.sgprs.max(hw.sgprs) };
    let clauses = clauses_of(f, &groups);
    let (phys, used, tight) = assign_best(f, &live, groups, budget, hard, 1, clauses.as_ref()).map_err(overflow_error)?;
    rewrite(f, &phys);
    let pressure = Pressure::default();
    Ok(Allocation { num_sgprs: used.sgprs, num_vgprs: used.vgprs, pressure, spill: SpillStats::default(), tight })
}

/// The registers of each class an assignment of `f` needs, with no limit:
/// at least its pressure, more where tuples fragment the register file.
pub fn assignment_need(f: &MFunc) -> Result<Pressure, CompileError> {
    let mut live = analyze(f)?;
    let groups = coalesce(f, &mut live);
    let unbounded = Limits { vgprs: 4 * MAX_VGPRS, sgprs: 4 * MAX_SGPRS };
    let (_, used, _) = assign_best(f, &live, groups, Limits::for_func(f), unbounded, 1, None).map_err(overflow_error)?;
    Ok(used)
}

/// Allocates within `limits`, spilling when more dwords of a class are live
/// at once than its limit, or when the assignment does not fit: SGPRs to
/// VGPR lanes first (which adds VGPRs), then VGPRs to LDS and scratch.
pub fn allocate_spilling(f: &mut MFunc, limits: Limits, cfg: SpillConfig) -> Result<Allocation, CompileError> {
    allocate_spilling_on(f, limits, cfg, 1)
}

/// `allocate_spilling` on up to `threads` threads; the result is the same.
pub(super) fn allocate_spilling_on(f: &mut MFunc, limits: Limits, cfg: SpillConfig, threads: usize) -> Result<Allocation, CompileError> {
    let first = pressure(f)?;
    let fits = |a: &Allocation| a.num_vgprs <= limits.vgprs && a.num_sgprs <= limits.sgprs;
    if first.vgprs <= limits.vgprs && first.sgprs <= limits.sgprs {
        let mut g = f.clone();
        if let Ok(a) = allocate_within(&mut g, limits) {
            if fits(&a) {
                *f = g;
                return Ok(Allocation { pressure: first, ..a });
            }
        }
    }
    let mut spiller = super::spill::Spiller::new(f, cfg);
    let hw = Limits::for_func(f);
    let (mut target_s, mut target_v) = (limits.sgprs.min(hw.sgprs), limits.vgprs.min(hw.vgprs));
    // The VGPR target spilling may go down to: gfx950's even-aligned tuples
    // can fragment a small file far more than gfx11's unaligned ones.
    let floor_v = if f.target == crate::target::Target::Gfx950 { 8 } else { 16 };
    // The liveness of `f` as it stands, while a step leaves `f` unchanged.
    let mut known = None;
    loop {
        spiller.reduce(f, Class::S, target_s, &mut known)?;
        spiller.reduce(f, Class::V, target_v, &mut known)?;
        let mut live = match known.take() {
            Some(live) => live,
            None => spiller.analyze(f)?,
        };
        let pressure = pressure_of(f, &live);
        // Coalescing merges segments; the next round starts from `f` as is.
        known = Some(live.clone());
        let groups = coalesce(f, &mut live);
        let hard = Limits { vgprs: limits.vgprs.min(hw.vgprs), sgprs: limits.sgprs.min(hw.sgprs) };
        // Fragmentation can need more registers than are live at once:
        // then spill further below the limit.
        let assigned = assign_best_meanwhile(f, &live, groups, hard, threads, (&mut spiller, &mut known), (target_s, target_v.saturating_sub(2)));
        spiller.recycle(live);
        match assigned {
            Ok((phys, used, tight)) if used.vgprs <= hard.vgprs && used.sgprs <= hard.sgprs => {
                rewrite(f, &phys);
                let spill = spiller.finish(f)?;
                return Ok(Allocation { num_sgprs: used.sgprs, num_vgprs: used.vgprs, pressure, spill, tight });
            }
            Ok((_, used, _)) => {
                if used.sgprs > hard.sgprs {
                    target_s = target_s.saturating_sub(used.sgprs - hard.sgprs).max(8);
                }
                if used.vgprs > hard.vgprs {
                    target_v = target_v.saturating_sub(used.vgprs - hard.vgprs).max(floor_v);
                }
            }
            Err(o) if o.class == Class::V && target_v > floor_v => target_v -= o.size.clamp(2, 8),
            Err(o) if o.class == Class::S && target_s > 8 => target_s -= o.size.clamp(1, 4),
            Err(o) => return Err(overflow_error(o)),
        }
    }
}

/// `assign_best` of `f` within `hard` for `allocate_spilling_on`. With a
/// thread to spare on a large function, the assignment runs on another
/// thread, over a copy of `f`, while this one already reduces `f` to `next`:
/// the targets of the next round after an overflow by a value of one or two
/// dwords, as most failed assignments end. The spiller keeps that reduction
/// only if the next round asks for those targets; an assignment that fits
/// gets `f` as it was. Either way the result is `assign_best`'s.
fn assign_best_meanwhile(
    f: &mut MFunc,
    live: &Live,
    groups: Groups,
    hard: Limits,
    threads: usize,
    (spiller, known): (&mut super::spill::Spiller, &mut Option<Live>),
    next: (u32, u32),
) -> Result<(Vec<u32>, Used, bool), Overflow> {
    if threads < 2 || live.slots() < 2 * PARALLEL_MIN_INSTS {
        return assign_fitting(f, live, groups, hard, threads);
    }
    let before = f.clone();
    let checkpoint = spiller.checkpoint();
    let stop = AtomicBool::new(false);
    let fits = |used: &Used| used.vgprs <= hard.vgprs && used.sgprs <= hard.sgprs;
    let (assigned, ahead) = std::thread::scope(|s| {
        let helper = s.spawn(|| {
            let assigned = assign_fitting(&before, live, groups, hard, threads - 1);
            // Outcomes after which the next round asks for other targets.
            let elsewhere = match &assigned {
                Ok((_, used, _)) => fits(used) || used.sgprs > hard.sgprs || used.vgprs != hard.vgprs + 2,
                Err(o) => o.class != Class::V || o.size > 2,
            };
            if elsewhere {
                stop.store(true, Ordering::Relaxed);
            }
            assigned
        });
        let ahead = spiller.reduce_ahead(f, next, known, &stop);
        (helper.join().unwrap_or_else(|panic| std::panic::resume_unwind(panic)), ahead)
    });
    match ahead {
        Some(reductions) if !matches!(&assigned, Ok((_, used, _)) if fits(used)) => spiller.hold(reductions, next, before, checkpoint),
        _ => {
            *f = before;
            spiller.restore(checkpoint);
            *known = None;
        }
    }
    assigned
}

/// gfx950's clause preference for assigning `f`'s groups (`gfx9::clauses`).
fn clauses_of(f: &MFunc, groups: &Groups) -> Option<crate::target::gfx9::clauses::Clauses> {
    if f.target != crate::target::Target::Gfx950 {
        return None;
    }
    let mut roots = groups.parent.clone();
    crate::target::gfx9::clauses::Clauses::of(f, &mut |v| find(&mut roots, v))
}

/// `assign_best` within `hard` for a round of spilling: with gfx950's clause
/// preference, and where that does not fit and the preference moved a
/// group, without it, so that a round fits whenever it did without it.
fn assign_fitting(f: &MFunc, live: &Live, groups: Groups, hard: Limits, threads: usize) -> Result<(Vec<u32>, Used, bool), Overflow> {
    let Some(clauses) = clauses_of(f, &groups) else {
        return assign_best(f, live, groups, hard, hard, threads, None);
    };
    let Groups { parent, gsize, galign, gfixed } = groups;
    let plain = Groups { parent: parent.clone(), gsize: gsize.clone(), galign: galign.clone(), gfixed: gfixed.clone() };
    let first = assign_best(f, live, Groups { parent, gsize, galign, gfixed }, hard, hard, threads, Some(&clauses));
    if !clauses.moved() || first.as_ref().is_ok_and(|(_, used, _)| used.vgprs <= hard.vgprs && used.sgprs <= hard.sgprs) {
        return first;
    }
    assign_best(f, live, plain, hard, hard, threads, None)
}

fn overflow_error(o: Overflow) -> CompileError {
    let what = if o.class == Class::S { "SGPRs" } else { "VGPRs" };
    CompileError::new("ROCKE_DIRECT_REGISTERS", format!("out of {what}: no room for a {}-dword value", o.size))
}

/// Registers of each class an assignment uses.
type Used = Pressure;

/// Why an assignment failed: the class and the size of the group that found
/// no room within the limit.
struct Overflow {
    class: Class,
    size: u32,
}

/// Assigns in order of first definition, each group taking its preferred
/// register or the first fit. If that needs more VGPRs than `budget`, but
/// at most a tuple's worth more (or does not fit `hard`), also with larger
/// groups first and no preferences (which packs tuples better), keeping the
/// one using fewer VGPRs. Each may use registers up to `hard`. The flag
/// tells that the first assignment did not fit `hard`. With `threads` above
/// one, a large function packs tuples first meanwhile, in case it is asked
/// for; the result is the same. With `clauses`, `assign`'s.
fn assign_best(
    f: &MFunc,
    live: &Live,
    groups: Groups,
    budget: Limits,
    hard: Limits,
    threads: usize,
    clauses: Option<&crate::target::gfx9::clauses::Clauses>,
) -> Result<(Vec<u32>, Used, bool), Overflow> {
    let fits = |r: &Result<(Vec<u32>, Used), Overflow>| r.as_ref().is_ok_and(|(_, u)| u.vgprs <= budget.vgprs);
    // Packing tuples first recovers a few registers, not more than a tuple's.
    let near = |r: &Result<(Vec<u32>, Used), Overflow>| match r {
        Err(_) => true,
        Ok((_, u)) => u.vgprs <= budget.vgprs + 8,
    };
    let (mut best, packed) = if threads > 1 && live.slots() >= 2 * PARALLEL_MIN_INSTS {
        let stop = AtomicBool::new(false);
        std::thread::scope(|s| {
            let packed = s.spawn(|| assign(f, live, &groups, hard, Order::BigFirst, Some(&stop), clauses));
            let best = assign(f, live, &groups, hard, Order::Preferred, None, clauses);
            if fits(&best) || !near(&best) {
                stop.store(true, Ordering::Relaxed);
            }
            (best, Some(packed.join().unwrap_or_else(|panic| std::panic::resume_unwind(panic))))
        })
    } else {
        (assign(f, live, &groups, hard, Order::Preferred, None, clauses), None)
    };
    let first_fit_failed = best.is_err();
    if !fits(&best) && near(&best) {
        let next = packed.unwrap_or_else(|| assign(f, live, &groups, hard, Order::BigFirst, None, clauses));
        best = match (&best, &next) {
            (Ok((_, a)), Ok((_, b))) if b.vgprs < a.vgprs => next,
            (Err(_), Ok(_)) => next,
            _ => best,
        };
    }
    best.map(|(phys, used)| (phys, used, first_fit_failed))
}

/// How `assign` orders and places groups.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Order {
    /// By first definition, each at its preferred register when free.
    Preferred,
    /// Larger groups first, each at the first fit.
    BigFirst,
}

/// Registers groups would like to share: in an accumulation chain (an
/// `v_fma_f32` whose result is the addend of another), the result with its
/// addend (where that dies there), which makes it the two-address `fmac`
/// VOPD dual-issues. Outside chains the addend's register is often rewritten
/// right away, and taking it there can put a write just behind a
/// transcendental reading the same register. Per root: (other root, offset
/// of this root's base from the other's).
fn affinities(f: &MFunc, groups: &Groups) -> Affinities {
    let mut parent = groups.parent.clone();
    let mut prefs = Vec::new();
    let plain_fma = |i: &MInst| i.op == Opc::VFmaF32 && i.mods == 0;
    let mut addend = vec![false; f.vregs.len()];
    for inst in f.blocks.iter().flat_map(|b| b.insts.iter()).filter(|i| plain_fma(i)) {
        if let Opnd::R(c) = inst.uses[2] {
            addend[c.reg as usize] = true;
        }
    }
    for inst in f.blocks.iter().flat_map(|b| b.insts.iter()) {
        if !plain_fma(inst) {
            continue;
        }
        let (Opnd::R(d), Opnd::R(c)) = (inst.defs[0], inst.uses[2]) else {
            continue;
        };
        if f.vregs[d.reg as usize].class != Class::V || f.vregs[c.reg as usize].class != Class::V || !addend[d.reg as usize] {
            continue;
        }
        let (rd, od) = find(&mut parent, d.reg);
        let (rc, oc) = find(&mut parent, c.reg);
        if rd != rc {
            let delta = (oc as i32 + c.off as i32) - (od as i32 + d.off as i32);
            prefs.push((rd, rc, delta));
            prefs.push((rc, rd, -delta));
        }
    }
    Affinities::new(f.vregs.len(), &prefs)
}

/// Each root's affinities in the order found, in one table.
struct Affinities {
    start: Vec<u32>,
    pairs: Vec<(u32, i32)>,
}

impl Affinities {
    /// From (root, other root, offset) triples.
    fn new(roots: usize, prefs: &[(u32, u32, i32)]) -> Self {
        let mut start = vec![0u32; roots + 1];
        for &(root, _, _) in prefs {
            start[root as usize + 1] += 1;
        }
        for r in 0..roots {
            start[r + 1] += start[r];
        }
        let mut next = start.clone();
        let mut pairs = vec![(0, 0); prefs.len()];
        for &(root, other, delta) in prefs {
            pairs[next[root as usize] as usize] = (other, delta);
            next[root as usize] += 1;
        }
        Affinities { start, pairs }
    }

    fn of(&self, root: u32) -> &[(u32, i32)] {
        &self.pairs[self.start[root as usize] as usize..self.start[root as usize + 1] as usize]
    }
}

/// One greedy assignment of the group roots to physical registers: each
/// takes the lowest position free over all its units' live segments or,
/// with `Order::Preferred`, first the register of an assigned group it has
/// an affinity with. Precolored groups go first, then groups by first
/// definition (with `Order::BigFirst`, by size and then first definition).
/// In order of first definition the registers taken at a definition are
/// those of the values live there, so preferring one free register over
/// another costs single-dword SSA values nothing; holes in loop-carried
/// values and tuples can still make it cost, hence the other order.
/// `stop` set ends it early with a meaningless `Err`. With `clauses`, a
/// gfx950 load and the registers its soft clause reads keep off each other
/// where another register is free (`gfx9::clauses`).
fn assign(
    f: &MFunc,
    live: &Live,
    groups: &Groups,
    limits: Limits,
    order: Order,
    stop: Option<&AtomicBool>,
    clauses: Option<&crate::target::gfx9::clauses::Clauses>,
) -> Result<(Vec<u32>, Used), Overflow> {
    let nv = f.vregs.len();
    let slots = live.slots();
    let big_first = order == Order::BigFirst;
    let prefs = if order == Order::Preferred { affinities(f, groups) } else { Affinities::new(nv, &[]) };
    let mut placed = vec![false; nv];
    let Groups { parent, gsize, galign, gfixed } = groups;
    let group_unit = |v: u32, k: u32| live.base[v as usize] + k as usize;
    let start_of = |v: u32| -> u32 {
        (0..gsize[v as usize])
            .filter_map(|k| live.segs(group_unit(v, k)).first().map(|x| x.0))
            .min()
            .unwrap_or(u32::MAX)
    };
    // Precolored groups, then the others, each by (rank, start, size
    // downward, register) packed into one integer.
    let key = |v: u32| -> u128 {
        let size = gsize[v as usize];
        let rank = if big_first { u32::MAX - size } else { 0 };
        (rank as u128) << 96 | (start_of(v) as u128) << 64 | ((u32::MAX - size) as u128) << 32 | v as u128
    };
    let (mut fixed, mut free) = (Vec::new(), Vec::new());
    for v in (0..nv as u32).filter(|&v| parent[v as usize].0 == v) {
        let bucket = if gfixed[v as usize].is_some() { &mut fixed } else { &mut free };
        bucket.push(key(v));
    }
    sort_root_keys(&mut fixed);
    sort_root_keys(&mut free);
    let roots = fixed.into_iter().chain(free).map(|key| key as u32);
    let mut occ_s = Occupancy::new(limits.sgprs.max(MAX_SGPRS), slots);
    let mut occ_v = Occupancy::new(limits.vgprs.max(MAX_VGPRS), slots);
    let mut assign = vec![0u32; nv];
    let mut used = Used::default();
    let mut checks: Vec<(u32, Check)> = Vec::new();
    for root in roots {
        if stop.is_some_and(|stop| stop.load(Ordering::Relaxed)) {
            return Err(Overflow { class: Class::V, size: 0 });
        }
        let class = f.vregs[root as usize].class;
        let size = gsize[root as usize];
        let align = galign[root as usize].max(1) as u32;
        let (occ, limit) = match class {
            Class::S => (&mut occ_s, limits.sgprs),
            Class::V => (&mut occ_v, limits.vgprs),
        };
        let segs = |k: u32| live.segs(group_unit(root, k));
        checks.clear();
        for k in 0..size {
            checks.extend(segs(k).iter().map(|&seg| (k, check_of(seg))));
        }
        let chosen = if let Some(p) = gfixed[root as usize] {
            // Precolored inputs sit at fixed hardware registers, budget or not.
            let p = p as u32;
            if !occ.free_from(p, &checks) {
                return Err(Overflow { class, size });
            }
            p
        } else {
            let preferred = prefs.of(root).iter().find_map(|&(other, delta)| {
                let at = u32::try_from(assign[other as usize] as i32 + delta).ok()?;
                (placed[other as usize] && at % align == 0 && at + size <= limit && occ.free_from(at, &checks))
                    .then_some(at)
            });
            match preferred {
                Some(at) => at,
                None => {
                    let at = occ.first_free(&checks, size, align, limit).ok_or(Overflow { class, size })?;
                    let high = match class {
                        Class::S => used.sgprs,
                        Class::V => used.vgprs,
                    };
                    let same = |r: u32| f.vregs[r as usize].class == class;
                    clauses.map_or(at, |c| c.prefer((root, at, size, align, high), &placed, &assign, same, |p| occ.free_from(p, &checks)))
                }
            }
        };
        for k in 0..size {
            let segs = segs(k);
            if !segs.is_empty() {
                occ.take(chosen + k, segs);
                match class {
                    Class::S => used.sgprs = used.sgprs.max(chosen + k + 1),
                    Class::V => used.vgprs = used.vgprs.max(chosen + k + 1),
                }
            }
        }
        assign[root as usize] = chosen;
        placed[root as usize] = true;
    }
    let mut parent = parent.clone();
    let phys = (0..nv as u32)
        .map(|v| {
            let (root, off) = find(&mut parent, v);
            assign[root as usize] + off as u32
        })
        .collect();
    Ok((phys, used))
}

/// Sorts `keys` as `sort_unstable` does, by a stable counting sort on each
/// 32-bit field from the lowest up (skipped where the keys are already in
/// that field's order), in time linear in their number while each field's
/// values, `u32::MAX` aside, span a range not much wider; a field spread
/// wider falls back to `sort_unstable`.
fn sort_root_keys(keys: &mut Vec<u128>) {
    let n = keys.len();
    let (mut out, mut counts) = (Vec::new(), Vec::<u32>::new());
    for shift in [0u32, 32, 64, 96] {
        let field = |k: u128| (k >> shift) as u32;
        if keys.windows(2).all(|w| field(w[0]) <= field(w[1])) {
            continue;
        }
        let (lo, hi) = keys
            .iter()
            .map(|&k| field(k))
            .filter(|&x| x != u32::MAX)
            .fold((u32::MAX, 0), |(lo, hi), x| (lo.min(x), hi.max(x)));
        let width = hi.saturating_sub(lo) as usize + 2;
        if width > 16 * n + 1024 {
            keys.sort_unstable();
            return;
        }
        let bucket = |x: u32| if x == u32::MAX { width - 1 } else { (x - lo) as usize };
        counts.clear();
        counts.resize(width + 1, 0);
        for &k in keys.iter() {
            counts[bucket(field(k)) + 1] += 1;
        }
        for b in 1..width {
            counts[b] += counts[b - 1];
        }
        out.resize(n, 0);
        for &k in keys.iter() {
            let b = bucket(field(k));
            out[counts[b] as usize] = k;
            counts[b] += 1;
        }
        std::mem::swap(keys, &mut out);
    }
}

/// Replaces virtual registers by `phys[reg] + off` and drops identity copies.
fn rewrite(f: &mut MFunc, phys: &[u32]) {
    let vregs = std::mem::take(&mut f.vregs);
    let phys_of = |o: &Opnd| -> Opnd {
        match o {
            Opnd::R(v) => {
                let p = phys[v.reg as usize] + v.off as u32;
                let phys = match vregs[v.reg as usize].class {
                    Class::S => Phys::S(p as u16),
                    Class::V => Phys::V(p as u16),
                };
                Opnd::P(phys, v.n)
            }
            other => *other,
        }
    };
    for block in f.blocks.iter_mut() {
        for inst in block.insts.iter_mut() {
            for o in inst.defs.iter_mut() {
                *o = phys_of(o);
            }
            for o in inst.uses.iter_mut() {
                *o = phys_of(o);
            }
        }
        block.insts.retain(|inst| !(inst.op == Opc::Copy && inst.defs[0] == inst.uses[0]));
    }
    f.vregs = vregs;
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Units live out of each block by the plain dataflow over every unit.
    fn dense_live_out(f: &MFunc) -> Vec<Vec<usize>> {
        let (live, _) = dataflow(f).expect("analyzes");
        let (units, blocks) = (*live.base.last().expect("non-empty"), f.blocks.len());
        let (mut uses, mut defs) = (vec![vec![false; units]; blocks], vec![vec![false; units]; blocks]);
        for (b, block) in f.blocks.iter().enumerate() {
            for inst in &block.insts {
                for u in inst.uses.iter().filter_map(|o| live.of(o)).flatten() {
                    uses[b][u] |= !defs[b][u];
                }
                for u in inst.defs.iter().filter_map(|o| live.of(o)).flatten() {
                    defs[b][u] = true;
                }
            }
        }
        let (mut live_in, mut live_out) = (vec![vec![false; units]; blocks], vec![vec![false; units]; blocks]);
        let mut changed = true;
        while changed {
            changed = false;
            for b in (0..blocks).rev() {
                for s in f.successors(b) {
                    for u in 0..units {
                        changed |= live_in[s][u] && !live_out[b][u];
                        live_out[b][u] |= live_in[s][u];
                    }
                }
                for u in 0..units {
                    let inside = uses[b][u] || live_out[b][u] && !defs[b][u];
                    changed |= inside != live_in[b][u];
                    live_in[b][u] = inside;
                }
            }
        }
        live_out.iter().map(|out| (0..units).filter(|&u| out[u]).collect()).collect()
    }

    #[test]
    fn dataflow_over_crossing_units_matches_every_unit() {
        for seed in 0..40 {
            let f = super::super::spill::tests::random_kernel(seed, 20 + seed as usize, 5 + seed as usize % 30);
            let want = dense_live_out(&f);
            for sparse in [false, true] {
                let (_, out) = dataflow_with(&f, sparse).expect("analyzes");
                let got: Vec<Vec<usize>> = (0..f.blocks.len()).map(|b| out.of(b).collect()).collect();
                assert_eq!(got, want, "seed {seed}, sparse {sparse}");
            }
        }
    }

    #[test]
    fn pressure_counts_what_the_segments_hold() {
        for seed in 0..40 {
            let f = super::super::spill::tests::random_kernel(seed, 20 + seed as usize, 5 + seed as usize % 30);
            let live = analyze(&f).expect("analyzes");
            assert_eq!(pressure(&f).expect("analyzes"), pressure_of(&f, &live), "seed {seed}");
        }
    }

    #[test]
    fn assignment_uses_the_pressure_for_single_dword_ssa_values() {
        // Straight-line values with arbitrary lifetimes: a chordal
        // (interval) interference graph, colored in definition order.
        let mut f = MFunc::default();
        let mut insts = Vec::new();
        let mut live: Vec<u32> = Vec::new();
        let mut seed = 7u64;
        let mut next = |n: usize| {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (seed >> 33) as usize % n
        };
        for i in 0..400u32 {
            f.vregs.push(VRegInfo { class: Class::V, size: 1, align: 1, fixed: None });
            let src = if live.is_empty() { Opnd::Imm(i) } else { Opnd::R(VRef { reg: live[next(live.len())], off: 0, n: 1 }) };
            insts.push(MInst::new(Opc::VAddNcU32, [Opnd::R(VRef { reg: i, off: 0, n: 1 })], [Opnd::Imm(1), src]));
            live.push(i);
            while live.len() > 3 && (next(5) < 2 || live.len() > 100) {
                let v = live.swap_remove(next(live.len()));
                insts.push(MInst::new(Opc::GStoreB32, [], [Opnd::R(VRef { reg: v, off: 0, n: 1 }); 2]));
            }
        }
        for v in live {
            insts.push(MInst::new(Opc::GStoreB32, [], [Opnd::R(VRef { reg: v, off: 0, n: 1 }); 2]));
        }
        f.blocks.push(MBlock { insts, loop_depth: 0 });
        let p = pressure(&f).expect("analyzes");
        let a = allocate(&mut f).expect("fits");
        assert_eq!(a.num_vgprs, p.vgprs);
    }

    #[test]
    fn occupancy_matches_segment_overlap() {
        let mut occ = Occupancy::new(2, 300);
        let held: Vec<Seg> = vec![(3, 3), (60, 70), (127, 128), (200, 290)];
        occ.take(1, &held);
        let probes: Vec<Seg> = vec![(0, 2), (3, 3), (4, 59), (71, 126), (64, 64), (129, 199), (0, 299), (291, 299), (128, 128)];
        for &p in &probes {
            assert_eq!(occ.free(1, &[p]), !overlaps(&[p], &held), "{p:?}");
            assert!(occ.free(0, &[p]));
        }
    }

    #[test]
    fn first_free_is_the_first_position_free_from() {
        let mut seed = 0x51f1_5eed_0bad_cafeu64;
        let mut next = |n: u32| {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (seed >> 33) as u32 % n
        };
        for round in 0..300 {
            let (regs, slots) = (if round % 3 == 0 { 300 } else { 256 }, 64 * (1 + next(12)));
            let mut occ = Occupancy::new(regs, slots as usize);
            for _ in 0..next(3000) {
                let reg = next(regs);
                let start = next(slots);
                let span = if next(4) == 0 { slots } else { 90 };
                let seg = (start, (start + next(span)).min(slots - 1));
                if occ.free(reg, &[seg]) {
                    occ.take(reg, &[seg]);
                }
            }
            for _ in 0..20 {
                let size = 1 + next(6);
                let align = [1, 1, 2, 4, 3][next(5) as usize];
                let limit = [regs, 256, 200, 7, size, size - 1][next(6) as usize];
                let mut checks = Vec::new();
                for k in 0..size {
                    for _ in 0..next(4) {
                        let start = next(slots);
                        let len = if next(5) == 0 { next(slots) } else { next(150) };
                        checks.push((k, check_of((start, (start + len).min(slots - 1)))));
                    }
                }
                let mut want = None;
                let mut at = 0;
                while at + size <= limit {
                    if occ.free_from(at, &checks) {
                        want = Some(at);
                        break;
                    }
                    at += align;
                }
                assert_eq!(occ.first_free(&checks, size, align, limit), want, "round {round}");
            }
        }
    }

    #[test]
    fn occupancy_matches_overlap_across_many_words() {
        let slots = 9000u32;
        let mut seed = 0x2545_f491_4f6c_dd1du64;
        let mut next = |n: u32| {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (seed >> 33) as u32 % n
        };
        let random_seg = |next: &mut dyn FnMut(u32) -> u32| {
            let start = next(slots);
            let len = match next(3) {
                0 => next(4),
                1 => next(200),
                _ => next(slots),
            };
            (start, (start + len).min(slots - 1))
        };
        let mut occ = Occupancy::new(3, slots as usize);
        let mut held: Vec<Vec<Seg>> = vec![Vec::new(); 3];
        for _ in 0..60 {
            let reg = next(3);
            let seg = random_seg(&mut next);
            if !overlaps(&[seg], &held[reg as usize]) {
                occ.take(reg, &[seg]);
                held[reg as usize].push(seg);
                held[reg as usize].sort_unstable();
            }
            for _ in 0..40 {
                let probe = random_seg(&mut next);
                for r in 0..3 {
                    assert_eq!(occ.free(r, &[probe]), !overlaps(&[probe], &held[r as usize]), "{r} {probe:?}");
                }
            }
        }
    }

    /// `pressure` before `Liveness`: the reference `Liveness::pressure` is
    /// checked against.
    fn pressure_reference(f: &MFunc) -> Result<Pressure, CompileError> {
        let (live, live_out) = dataflow(f)?;
        let nunits = *live.base.last().expect("non-empty");
        let mut is_v = Bits::new(nunits);
        for v in (0..f.vregs.len()).filter(|&v| f.vregs[v].class == Class::V) {
            for u in live.base[v]..live.base[v + 1] {
                is_v.set(u);
            }
        }
        let mut most = Pressure::default();
        let mut cur = Bits::new(nunits);
        for (b, block) in f.blocks.iter().enumerate() {
            cur.0.fill(0);
            let mut n = [0u32; 2];
            for u in live_out.of(b) {
                cur.set(u);
                n[!is_v.get(u) as usize] += 1;
            }
            for inst in block.insts.iter().rev() {
                let early = inst.op.flags() & F_EARLY != 0;
                let units = |o: &Opnd| live.of(o).into_iter().flatten();
                // Write slot: the units live after, plus results nothing reads.
                let mut extra = [0u32; 2];
                if !early {
                    for u in inst.defs.iter().flat_map(units) {
                        if !cur.get(u) {
                            extra[!is_v.get(u) as usize] += 1;
                        }
                    }
                }
                most.vgprs = most.vgprs.max(n[0] + extra[0]);
                most.sgprs = most.sgprs.max(n[1] + extra[1]);
                for u in inst.defs.iter().flat_map(units) {
                    if cur.get(u) {
                        cur.clear(u);
                        n[!is_v.get(u) as usize] -= 1;
                    }
                }
                for u in inst.uses.iter().flat_map(units) {
                    if !cur.get(u) {
                        cur.set(u);
                        n[!is_v.get(u) as usize] += 1;
                    }
                }
                // Read slot: the units live before, plus early-clobber results.
                let mut extra = [0u32; 2];
                if early {
                    for u in inst.defs.iter().flat_map(units) {
                        if !cur.get(u) {
                            extra[!is_v.get(u) as usize] += 1;
                        }
                    }
                }
                most.vgprs = most.vgprs.max(n[0] + extra[0]);
                most.sgprs = most.sgprs.max(n[1] + extra[1]);
            }
        }
        Ok(most)
    }

    #[test]
    fn liveness_gives_the_pressure_of_every_schedule() {
        let random = (0..60u64).map(|seed| super::super::post::tests::random_function(seed, 1 + seed as usize % 4, 10 + 9 * seed as usize % 300));
        let looping = (0..40u64).map(|seed| super::super::spill::tests::random_kernel(seed, 20 + seed as usize, 5 + seed as usize % 30));
        for (k, f) in random.chain(looping).enumerate() {
            let known = Liveness::new(&f).expect("analyzes");
            assert_eq!(known.pressure(&f, None), pressure_reference(&f).expect("analyzes"), "function {k}");
            let mut sunk = f.clone();
            super::super::post::sink_load_uses(&mut sunk, 32);
            assert_eq!(known.pressure(&sunk, None), pressure_reference(&sunk).expect("analyzes"), "function {k}, sunk");
            let dags = super::super::post::Dags::new(&sunk);
            for window in [1, 4, 16, 64, 512] {
                let orders = dags.orders(window);
                let scheduled = super::super::post::reordered(&sunk, &orders);
                assert_eq!(known.pressure(&sunk, Some(&orders)), pressure_reference(&scheduled).expect("analyzes"), "function {k}, window {window}");
                assert_eq!(known.pressure(&scheduled, None), pressure(&scheduled).expect("analyzes"), "function {k}, window {window}");
            }
        }
    }

    #[test]
    fn known_liveness_allocates_as_the_dataflow_does() {
        let random = (0..40u64).map(|seed| super::super::post::tests::random_function(seed, 1 + seed as usize % 4, 10 + 9 * seed as usize % 200));
        let looping = (0..30u64).map(|seed| super::super::spill::tests::random_kernel(seed, 20 + seed as usize, 5 + seed as usize % 30));
        for (k, f) in random.chain(looping).enumerate() {
            let known = Liveness::new(&f).expect("analyzes");
            let dags = super::super::post::Dags::new(&f);
            for window in [1, 16, 64] {
                let scheduled = super::super::post::reordered(&f, &dags.orders(window));
                for budget in [Limits { vgprs: 12, sgprs: 10 }, Limits::default()] {
                    let (mut a, mut b) = (scheduled.clone(), scheduled.clone());
                    let (x, y) = (allocate_within(&mut a, budget), allocate_within_known(&mut b, budget, &known));
                    assert_eq!(format!("{x:?}"), format!("{y:?}"), "function {k}, window {window}");
                    assert_eq!(format!("{:?}", a.blocks), format!("{:?}", b.blocks), "function {k}, window {window}");
                }
            }
        }
    }

    #[test]
    fn unit_lists_give_the_pressure_of_every_schedule() {
        let random = (0..60u64).map(|seed| super::super::post::tests::random_function(seed, 1 + seed as usize % 5, 10 + 9 * seed as usize % 300));
        let looping = (0..40u64).map(|seed| super::super::spill::tests::random_kernel(seed, 20 + seed as usize, 5 + seed as usize % 30));
        for (k, f) in random.chain(looping).enumerate() {
            let mut sunk = f.clone();
            super::super::post::sink_load_uses(&mut sunk, 32);
            let known = Liveness::new(&sunk).expect("analyzes");
            let lists = known.unit_lists(&sunk);
            assert_eq!(known.pressure_listed(&lists, None), pressure_reference(&sunk).expect("analyzes"), "function {k}");
            let dags = super::super::post::Dags::new(&sunk);
            for window in [1, 4, 16, 64, 512] {
                let orders = dags.orders(window);
                let scheduled = super::super::post::reordered(&sunk, &orders);
                assert_eq!(known.pressure_listed(&lists, Some(&orders)), pressure_reference(&scheduled).expect("analyzes"), "function {k}, window {window}");
            }
        }
    }

    #[test]
    fn capped_pressure_stops_only_past_the_cap() {
        let random = (0..60u64).map(|seed| super::super::post::tests::random_function(seed, 1 + seed as usize % 5, 10 + 9 * seed as usize % 300));
        let looping = (0..40u64).map(|seed| super::super::spill::tests::random_kernel(seed, 20 + seed as usize, 5 + seed as usize % 30));
        let (mut stopped, mut kept) = (0, 0);
        for (k, f) in random.chain(looping).enumerate() {
            let known = Liveness::new(&f).expect("analyzes");
            let lists = known.unit_lists(&f);
            let dags = super::super::post::Dags::new(&f);
            for window in [1, 16, 64] {
                let orders = dags.orders(window);
                let full = known.pressure_listed(&lists, Some(&orders));
                let caps = [(full.vgprs, full.sgprs), (full.vgprs.saturating_sub(1), full.sgprs), (full.vgprs, full.sgprs / 2), (0, 0)];
                for (vgprs, sgprs) in caps {
                    let cap = Limits { vgprs, sgprs };
                    match known.pressure_capped(&lists, cap, |b| dags.block_order(b, window)) {
                        Some((got, p)) => {
                            assert!(got == orders && p == full, "function {k}, window {window}");
                            kept += 1;
                        }
                        None => {
                            assert!(full.vgprs > cap.vgprs || full.sgprs > cap.sgprs, "function {k}, window {window}: stopped within {cap:?}");
                            stopped += 1;
                        }
                    }
                }
            }
        }
        assert!(stopped > 100 && kept > 100, "{stopped} stopped, {kept} kept");
    }

    #[test]
    fn root_keys_sort_as_sort_unstable_does() {
        let mut seed = 0x0dd5_0b1e_c7a5_5e7du64;
        let mut next = |n: u64| {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (seed >> 16) % n.max(1)
        };
        for round in 0..600u64 {
            let n = next(if round % 4 == 0 { 20 } else { 4000 });
            let big_first = round % 2 == 1;
            let mut keys: Vec<u128> = (0..n)
                .map(|v| {
                    let size = 1 + next(if round % 3 == 0 { 64 } else { 8 }) as u32;
                    let start = match next(12) {
                        0 => u32::MAX,
                        _ if round % 5 == 4 => next(1 << 32) as u32,
                        _ => next(2 * n + 1) as u32,
                    };
                    let rank = if big_first { u32::MAX - size } else { 0 };
                    let v = if round % 6 == 5 { next(n) } else { v } as u32;
                    (rank as u128) << 96 | (start as u128) << 64 | ((u32::MAX - size) as u128) << 32 | v as u128
                })
                .collect();
            match round % 7 {
                3 => keys.reverse(),
                6 => keys.iter_mut().for_each(|k| *k = (*k as u64 as u128) << 64 | next(1 << 40) as u128),
                _ => {}
            }
            let mut expected = keys.clone();
            expected.sort_unstable();
            sort_root_keys(&mut keys);
            assert_eq!(keys, expected, "round {round}");
        }
    }
}
