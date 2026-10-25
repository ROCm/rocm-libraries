// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

//! Register allocation: per-dword liveness, copy coalescing and first-fit
//! assignment. There is no spilling; running out of registers is an error.

use super::mir::*;
use crate::error::CompileError;

const NUM_SGPRS: u32 = 106;
const NUM_VGPRS: u32 = 256;

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
type Seg = (u32, u32);

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
}

/// A segment's first and last word index with the masks of its bits there.
type Check = (u32, u32, u64, u64);

fn check_of((s, e): Seg) -> Check {
    (s / 64, e / 64, !0u64 << (s % 64), !0u64 >> (63 - e % 64))
}

impl Occupancy {
    fn new(regs: u32, slots: usize) -> Self {
        let words = slots.div_ceil(64).max(1);
        Occupancy { regs: regs as usize, bits: vec![0; regs as usize * words] }
    }

    fn free_at(&self, reg: usize, (first, last, low, high): Check) -> bool {
        let word = |w: u32| self.bits[w as usize * self.regs + reg];
        if first == last {
            return word(first) & low & high == 0;
        }
        word(first) & low == 0 && (first + 1..last).all(|w| word(w) == 0) && word(last) & high == 0
    }

    fn free(&self, reg: u32, segs: &[Seg]) -> bool {
        assert!((reg as usize) < self.regs, "register {reg} out of range");
        segs.iter().all(|&seg| self.free_at(reg as usize, check_of(seg)))
    }

    /// Whether register `at + k` is free for every `(k, check)`.
    fn free_from(&self, at: u32, checks: &[(u32, Check)]) -> bool {
        checks.iter().all(|&(k, check)| self.free_at((at + k) as usize, check))
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
        }
    }
}

pub struct Allocation {
    pub num_sgprs: u32,
    pub num_vgprs: u32,
}

struct Units {
    base: Vec<usize>,
}

impl Units {
    fn of(&self, o: &Opnd) -> Option<std::ops::Range<usize>> {
        match o {
            Opnd::R(v) => {
                let start = self.base[v.reg as usize] + v.off as usize;
                Some(start..start + v.n as usize)
            }
            _ => None,
        }
    }
}

fn find(parent: &mut [(u32, u8)], v: u32) -> (u32, u8) {
    let (p, off) = parent[v as usize];
    if p == v {
        return (v, 0);
    }
    let (root, poff) = find(parent, p);
    parent[v as usize] = (root, off + poff);
    (root, off + poff)
}

pub fn allocate(f: &mut MFunc) -> Result<Allocation, CompileError> {
    allocate_unless(f, None).map(|a| a.expect("never skipped"))
}

/// Like `allocate`, but returns `Ok(None)`, leaving `f` as it is, when
/// `skip(n)` holds for `n`, a number of VGPRs that every assignment uses
/// at least: the most VGPR dwords live at one point.
pub fn allocate_unless(f: &mut MFunc, skip: Option<&dyn Fn(u32) -> bool>) -> Result<Option<Allocation>, CompileError> {
    let nv = f.vregs.len();
    let mut base = vec![0usize; nv + 1];
    for i in 0..nv {
        base[i + 1] = base[i] + f.vregs[i].size as usize;
    }
    let units = Units { base };
    let nunits = units.base[nv];
    let nblocks = f.blocks.len();
    let mut first = vec![0u32; nblocks + 1];
    for b in 0..nblocks {
        first[b + 1] = first[b] + f.blocks[b].insts.len() as u32;
    }

    // Block-local upward-exposed uses and definitions.
    let mut uses = vec![Bits::new(nunits); nblocks];
    let mut defs = vec![Bits::new(nunits); nblocks];
    for (b, block) in f.blocks.iter().enumerate() {
        for inst in &block.insts {
            for o in &inst.uses {
                if let Some(range) = units.of(o) {
                    for u in range {
                        if !defs[b].get(u) {
                            uses[b].set(u);
                        }
                    }
                }
            }
            for o in &inst.defs {
                if let Some(range) = units.of(o) {
                    for u in range {
                        defs[b].set(u);
                    }
                }
            }
        }
    }
    let succs: Vec<Vec<usize>> = (0..nblocks).map(|b| f.successors(b)).collect();
    let mut live_in = vec![Bits::new(nunits); nblocks];
    let mut live_out = vec![Bits::new(nunits); nblocks];
    let mut inb = Bits::new(nunits);
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
    let unit_owner = |u: usize| -> usize { units.base.partition_point(|&b| b <= u) - 1 };
    for u in live_in[0].ones() {
        let owner = unit_owner(u);
        if f.vregs[owner].fixed.is_none() {
            return Err(CompileError::new(
                "ROCKE_DIRECT_INTERNAL",
                format!("virtual register {owner} is used before it is defined"),
            ));
        }
    }

    // Live segments per unit, gathered as (unit, segment) and then sorted
    // into one table of normalized per-unit runs.
    let mut raw: Vec<(u32, Seg)> = Vec::with_capacity(2 * first[nblocks] as usize);
    let mut end = vec![0u32; nunits];
    for (b, block) in f.blocks.iter().enumerate() {
        if block.insts.is_empty() {
            continue;
        }
        let start_slot = 2 * first[b];
        let end_slot = 2 * first[b + 1] - 1;
        let mut live = live_out[b].clone();
        for u in live.ones() {
            end[u] = end_slot;
        }
        for (k, inst) in block.insts.iter().enumerate().rev() {
            let i = first[b] + k as u32;
            let dslot = if inst.op.flags() & F_EARLY != 0 { 2 * i } else { 2 * i + 1 };
            for o in &inst.defs {
                if let Some(range) = units.of(o) {
                    for u in range {
                        if live.get(u) {
                            raw.push((u as u32, (dslot, end[u])));
                            live.clear(u);
                        } else {
                            raw.push((u as u32, (dslot, dslot)));
                        }
                    }
                }
            }
            for o in &inst.uses {
                if let Some(range) = units.of(o) {
                    for u in range {
                        if !live.get(u) {
                            live.set(u);
                            end[u] = 2 * i;
                        }
                    }
                }
            }
        }
        for u in live.ones() {
            raw.push((u as u32, (start_slot, end[u])));
        }
    }
    let mut by_unit = vec![0u32; nunits + 1];
    for &(u, _) in &raw {
        by_unit[u as usize + 1] += 1;
    }
    for u in 0..nunits {
        by_unit[u + 1] += by_unit[u];
    }
    let mut sorted: Vec<Seg> = vec![(0, 0); raw.len()];
    let mut fill = by_unit.clone();
    for &(u, seg) in &raw {
        sorted[fill[u as usize] as usize] = seg;
        fill[u as usize] += 1;
    }
    let mut table: Vec<Seg> = Vec::with_capacity(raw.len());
    let mut unit_segs: Vec<UnitSegs> = vec![(0, 0); nunits];
    for u in 0..nunits {
        let run = &mut sorted[by_unit[u] as usize..by_unit[u + 1] as usize];
        if !run.is_empty() {
            let start = table.len();
            normalize_into(run, &mut table);
            unit_segs[u] = (start as u32, (table.len() - start) as u32);
        }
    }
    let slots = 2 * first[nblocks] as usize;
    if let Some(skip) = skip {
        // Units live at the same slot get distinct registers.
        let mut delta = vec![0i32; slots + 1];
        for v in (0..nv).filter(|&v| f.vregs[v].class == Class::V) {
            for u in units.base[v]..units.base[v + 1] {
                for &(s, e) in segs_of(&table, &unit_segs[u]) {
                    delta[s as usize] += 1;
                    delta[e as usize + 1] -= 1;
                }
            }
        }
        let (mut live, mut most) = (0i32, 0i32);
        for d in delta {
            live += d;
            most = most.max(live);
        }
        if skip(most as u32) {
            return Ok(None);
        }
    }

    // Coalescing: map whole groups into other groups at a dword offset. A
    // group is its root's units; merged groups are never read again.
    let mut parent: Vec<(u32, u8)> = (0..nv as u32).map(|v| (v, 0)).collect();
    let mut gsize: Vec<u32> = f.vregs.iter().map(|r| r.size as u32).collect();
    let mut galign: Vec<u8> = f.vregs.iter().map(|r| r.align).collect();
    let mut gfixed: Vec<Option<u16>> = f.vregs.iter().map(|r| r.fixed).collect();
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
    let group_unit = |v: u32, k: u32| units.base[v as usize] + k as usize;
    let mut both: Vec<Seg> = Vec::new();
    for &(_, b, k) in &copies {
        let inst = &f.blocks[b].insts[k];
        let (Opnd::R(d), Opnd::R(s)) = (inst.defs[0], inst.uses[0]) else {
            continue;
        };
        if f.vregs[d.reg as usize].class != f.vregs[s.reg as usize].class {
            continue;
        }
        let (rd, od) = find(&mut parent, d.reg);
        let (rs, os) = find(&mut parent, s.reg);
        let od = od as i32 + d.off as i32;
        let os = os as i32 + s.off as i32;
        if rd == rs {
            continue;
        }
        let try_merge = |x: u32, y: u32, delta: i32, unit_segs: &[UnitSegs], gsize: &[u32], galign: &[u8], gfixed: &[Option<u16>]| -> bool {
            let sx = gsize[x as usize] as i32;
            let sy = gsize[y as usize] as i32;
            if delta < 0 || delta + sx > sy || gfixed[x as usize].is_some() {
                return false;
            }
            let ax = galign[x as usize] as i32;
            if delta % ax != 0 {
                return false;
            }
            if let Some(p) = gfixed[y as usize] {
                if (p as i32 + delta) % ax != 0 {
                    return false;
                }
            }
            (0..sx as u32).all(|k| {
                let a = segs_of(&table, &unit_segs[group_unit(x, k)]);
                let b = segs_of(&table, &unit_segs[group_unit(y, k + delta as u32)]);
                !overlaps(a, b)
            })
        };
        let (x, y, delta) = if try_merge(rs, rd, od - os, &unit_segs, &gsize, &galign, &gfixed) {
            (rs, rd, od - os)
        } else if try_merge(rd, rs, os - od, &unit_segs, &gsize, &galign, &gfixed) {
            (rd, rs, os - od)
        } else {
            continue;
        };
        for k in 0..gsize[x as usize] {
            let from = group_unit(x, k);
            let to = group_unit(y, k + delta as u32);
            both.clear();
            both.extend_from_slice(segs_of(&table, &unit_segs[to]));
            both.extend_from_slice(segs_of(&table, &unit_segs[from]));
            let start = table.len();
            normalize_into(&mut both, &mut table);
            unit_segs[to] = (start as u32, (table.len() - start) as u32);
            unit_segs[from] = (0, 0);
        }
        gsize[x as usize] = 0;
        galign[y as usize] = galign[y as usize].max(galign[x as usize]);
        if gfixed[y as usize].is_none() {
            gfixed[y as usize] = gfixed[x as usize];
        }
        parent[x as usize] = (y, delta as u8);
    }

    // First-fit assignment of group roots.
    let start_of = |v: u32| -> u32 {
        (0..gsize[v as usize])
            .filter_map(|k| segs_of(&table, &unit_segs[group_unit(v, k)]).first().map(|x| x.0))
            .min()
            .unwrap_or(u32::MAX)
    };
    let mut roots: Vec<(bool, u32, std::cmp::Reverse<u32>, u32)> = (0..nv as u32)
        .filter(|&v| parent[v as usize].0 == v)
        .map(|v| (gfixed[v as usize].is_none(), start_of(v), std::cmp::Reverse(gsize[v as usize]), v))
        .collect();
    roots.sort_unstable();
    let mut occ_s = Occupancy::new(NUM_SGPRS, slots);
    let mut occ_v = Occupancy::new(NUM_VGPRS, slots);
    let mut assign = vec![0u32; nv];
    let mut max_s = 0u32;
    let mut max_v = 0u32;
    let mut checks: Vec<(u32, Check)> = Vec::new();
    for &(_, _, _, root) in &roots {
        let class = f.vregs[root as usize].class;
        let size = gsize[root as usize];
        let align = galign[root as usize].max(1) as u32;
        let (occ, limit) = match class {
            Class::S => (&mut occ_s, NUM_SGPRS),
            Class::V => (&mut occ_v, NUM_VGPRS),
        };
        let segs = |k: u32| segs_of(&table, &unit_segs[group_unit(root, k)]);
        let fits = |occ: &Occupancy, at: u32| -> bool { (0..size).all(|k| occ.free(at + k, segs(k))) };
        let chosen = if let Some(p) = gfixed[root as usize] {
            let p = p as u32;
            if !fits(occ, p) {
                return Err(CompileError::new(
                    "ROCKE_DIRECT_INTERNAL",
                    format!("precolored register {p} conflicts"),
                ));
            }
            p
        } else {
            checks.clear();
            for k in 0..size {
                checks.extend(segs(k).iter().map(|&seg| (k, check_of(seg))));
            }
            let mut at = 0;
            loop {
                if at + size > limit {
                    let what = if class == Class::S { "SGPRs" } else { "VGPRs" };
                    return Err(CompileError::new(
                        "ROCKE_DIRECT_REGISTERS",
                        format!("out of {what}: no room for a {size}-dword value (no spilling)"),
                    ));
                }
                if occ.free_from(at, &checks) {
                    break at;
                }
                at += align;
            }
        };
        for k in 0..size {
            let segs = segs(k);
            if !segs.is_empty() {
                occ.take(chosen + k, segs);
                match class {
                    Class::S => max_s = max_s.max(chosen + k + 1),
                    Class::V => max_v = max_v.max(chosen + k + 1),
                }
            }
        }
        assign[root as usize] = chosen;
    }

    // Rewrite to physical registers and drop identity copies.
    let phys: Vec<u32> = (0..nv as u32)
        .map(|v| {
            let (root, off) = find(&mut parent, v);
            assign[root as usize] + off as u32
        })
        .collect();
    let phys_of = |o: &Opnd| -> Opnd {
        match o {
            Opnd::R(v) => {
                let p = phys[v.reg as usize] + v.off as u32;
                let phys = match f.vregs[v.reg as usize].class {
                    Class::S => Phys::S(p as u16),
                    Class::V => Phys::V(p as u16),
                };
                Opnd::P(phys, v.n)
            }
            other => *other,
        }
    };
    let mut blocks = std::mem::take(&mut f.blocks);
    for block in blocks.iter_mut() {
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
    f.blocks = blocks;
    Ok(Some(Allocation {
        num_sgprs: max_s,
        num_vgprs: max_v,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
