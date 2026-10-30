// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

//! AGPRs as registers for MFMA accumulators on gfx950 (G2 design section 5.2,
//! step 2). When no schedule of a function fits the VGPRs it may use, its MFMA
//! accumulator webs move to AGPRs before anything spills (`allocate_banked`,
//! which `rir::spill_best` tries on each schedule, the widest window first).
//!
//! A web is the V registers that whole-register copies join to an MFMA's D,
//! which selection ties to its C (`mfma::select`). Its registers that are never
//! live at once share AGPRs, coalesced as the allocator coalesces copies; the
//! others take their own. The MFMAs then name the AGPRs (the encoder sets
//! `acc_cd` from the bank), copies within the web become `v_accvgpr_mov_b32` or
//! vanish, and every other access goes through a VGPR: `v_accvgpr_read_b32`
//! before a use, `v_accvgpr_write_b32` after a definition (an inline constant
//! is written directly). A web stays in VGPRs when such an access sits in the
//! innermost loop of its MFMAs, where it would run every iteration, or when an
//! instruction other than an MFMA reads and writes one of its registers.
//!
//! The allocator does not see the AGPRs (it tracks virtual registers), so it
//! assigns what is left within the VGPR limit as before. The AGPRs count towards
//! occupancy and the descriptor from a0, as spill slots do
//! (`SpillStats::agpr_dwords`): a function either fits with its webs in AGPRs
//! or spills as before, never both.

use super::asm::{literal_of, Lit, AGPR0};
use super::gen_isa::{self as t, Gfx9Op};
use crate::rir::mir::*;
use crate::rir::regalloc::{self, Allocation, Limits};
use crate::target::Target;

/// The AGPRs of a lane.
const AGPRS: u32 = 256;
const NONE: u32 = u32::MAX;

/// Whether `op` is an MFMA (D in `defs[0]`; A, B and C in `uses[0..3]`).
pub fn is_mfma(op: Opc) -> bool {
    matches!(op, Opc::Gfx9(_)) && t::info(op).is_some_and(|i| matches!(i.form, t::Form::Mai | t::Form::MaiScale))
}

/// The first resort before spilling on gfx950: `f` with its accumulator webs in
/// AGPRs, allocated within `limits` without spilling, when that fits.
pub fn allocate_banked(f: &MFunc, limits: Limits) -> Option<(MFunc, Allocation)> {
    if f.target != Target::Gfx950 || !f.blocks.iter().flat_map(|b| &b.insts).any(|i| is_mfma(i.op)) {
        return None;
    }
    let mut g = f.clone();
    let agprs = bank(&mut g)?;
    let pressure = regalloc::pressure(&g).ok()?;
    if pressure.vgprs > limits.vgprs || pressure.sgprs > limits.sgprs {
        return None;
    }
    let a = regalloc::allocate_within(&mut g, limits).ok()?;
    if a.num_vgprs > limits.vgprs || a.num_sgprs > limits.sgprs {
        return None;
    }
    let mut spill = a.spill;
    spill.agpr_dwords = agprs;
    Some((g, Allocation { pressure, spill, ..a }))
}

/// A set of small indices.
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
    fn or(&mut self, other: &Bits) -> bool {
        let mut changed = false;
        for (a, b) in self.0.iter_mut().zip(&other.0) {
            changed |= *a | *b != *a;
            *a |= *b;
        }
        changed
    }
    #[cfg(test)]
    fn intersects(&self, other: &Bits) -> bool {
        self.0.iter().zip(&other.0).any(|(a, b)| a & b != 0)
    }
    #[cfg(test)]
    fn ones(&self) -> impl Iterator<Item = usize> + '_ {
        self.0.iter().enumerate().flat_map(|(w, &word)| (0..64).filter(move |b| word >> b & 1 != 0).map(move |b| w * 64 + b))
    }
}

fn find(parent: &mut [u32], x: u32) -> u32 {
    let mut r = x;
    while parent[r as usize] != r {
        r = parent[r as usize];
    }
    let mut y = x;
    while parent[y as usize] != r {
        let next = parent[y as usize];
        parent[y as usize] = r;
        y = next;
    }
    r
}

/// Moves the accumulator webs of `f` (virtual registers) to AGPRs, as many as
/// fit; returns the AGPRs they take, or None when no web moves.
pub fn bank(f: &mut MFunc) -> Option<u32> {
    let (base_of, agprs) = place(f)?;
    rewrite(f, &base_of);
    Some(agprs)
}

/// Where `bank` puts each register of the webs that move (from AGPR 0, NONE
/// for the others), and the AGPRs they take.
pub(super) fn place(f: &MFunc) -> Option<(Vec<u32>, u32)> {
    place_with(f, false)
}

/// `place`; with `in_loops`, a web also moves when an access to it sits in
/// its MFMAs' innermost loop, the move running there every iteration.
pub(super) fn place_with(f: &MFunc, in_loops: bool) -> Option<(Vec<u32>, u32)> {
    let webs = Webs::of(f);
    let bad = webs.marked(f, in_loops);
    webs.placed(f, &bad)
}

/// `place_with(f, true)` and `place(f)`, from one pass over the webs, where
/// the in-loop rule keeps other webs in VGPRs than `place`'s does; None where
/// it keeps the same ones, and the two placements are then equal.
pub(super) fn place_apart(f: &MFunc) -> Option<(Option<(Vec<u32>, u32)>, Option<(Vec<u32>, u32)>)> {
    let webs = Webs::of(f);
    let (strict, in_loops) = (webs.marked(f, false), webs.marked(f, true));
    (strict != in_loops).then(|| (webs.placed(f, &in_loops), webs.placed(f, &strict)))
}

/// The register `o` names whole, if it does.
fn whole_reg(f: &MFunc, o: &Opnd) -> Option<u32> {
    match *o {
        Opnd::R(v) if v.off == 0 && v.n == f.vregs[v.reg as usize].size => Some(v.reg),
        _ => None,
    }
}

/// `rows[a]` to change and `rows[b]` to read, `a != b`.
fn pair(rows: &mut [Bits], a: usize, b: usize) -> (&mut Bits, &Bits) {
    if a < b {
        let (lo, hi) = rows.split_at_mut(b);
        (&mut lo[a], &hi[0])
    } else {
        let (lo, hi) = rows.split_at_mut(a);
        (&mut hi[0], &lo[b])
    }
}

/// The webs of a function as `place_with` finds them before its in-loop
/// rule: each register's root, which roots are webs, which of those stay in
/// VGPRs whatever that rule says, and each web's innermost MFMA loop and
/// weight.
struct Webs {
    root: Vec<u32>,
    web: Vec<bool>,
    bad: Vec<bool>,
    hot: Vec<u32>,
    value: Vec<u64>,
}

impl Webs {
    fn of(f: &MFunc) -> Webs {
        let nv = f.vregs.len();
        let whole = |o: &Opnd| whole_reg(f, o);
        // Registers only whole copies and MFMAs define (or none, undefined): the
        // others, such as a zero tuple several accumulators start from, stay out
        // of webs, and copies from them are moves into the web.
        let mut defined_inside = vec![true; nv];
        for inst in f.blocks.iter().flat_map(|b| &b.insts) {
            let inside = match inst.op {
                Opc::Copy | Opc::ImplicitDef => true,
                op if is_mfma(op) => inst.uses.get(2) == Some(&inst.defs[0]) || matches!(inst.uses.get(2), Some(Opnd::Imm(_))),
                _ => false,
            };
            for d in &inst.defs {
                if let Opnd::R(v) = d {
                    defined_inside[v.reg as usize] &= inside && whole(d).is_some();
                }
            }
        }
        let eligible = |v: u32| f.vregs[v as usize].class == Class::V && f.vregs[v as usize].fixed.is_none() && defined_inside[v as usize];
        // Webs: V registers joined by whole copies, with an MFMA's tied D and C.
        let mut parent: Vec<u32> = (0..nv as u32).collect();
        let mut seed = vec![false; nv];
        let mut pinned = vec![false; nv];
        for inst in f.blocks.iter().flat_map(|b| &b.insts) {
            if is_mfma(inst.op) {
                match (whole(&inst.defs[0]), inst.uses.get(2).and_then(whole)) {
                    (Some(d), Some(c)) if d == c && eligible(d) => seed[d as usize] = true,
                    // C an inline constant (`zero_acc`): the web starts at this D.
                    (Some(d), None) if matches!(inst.uses.get(2), Some(Opnd::Imm(_))) && eligible(d) => seed[d as usize] = true,
                    _ => {
                        for o in inst.defs.iter().chain(inst.uses.get(2)) {
                            if let Opnd::R(v) = o {
                                pinned[v.reg as usize] = true;
                            }
                        }
                    }
                }
            } else if inst.op == Opc::Copy {
                if let (Some(d), Some(s)) = (whole(&inst.defs[0]), whole(&inst.uses[0])) {
                    if eligible(d) && eligible(s) && f.vregs[d as usize].size == f.vregs[s as usize].size {
                        let (rd, rs) = (find(&mut parent, d), find(&mut parent, s));
                        parent[rd as usize] = rs;
                    }
                }
            }
        }
        let root: Vec<u32> = (0..nv as u32).map(|v| find(&mut parent, v)).collect();
        let mut web = vec![false; nv];
        let mut bad = vec![false; nv];
        for v in 0..nv {
            web[root[v] as usize] |= seed[v];
            bad[root[v] as usize] |= pinned[v];
        }
        // The innermost loop of each web's MFMAs, and their weight.
        let (mut hot, mut value) = (vec![0u32; nv], vec![0u64; nv]);
        for block in &f.blocks {
            for inst in block.insts.iter().filter(|i| is_mfma(i.op)) {
                if let Some(d) = whole(&inst.defs[0]).filter(|&d| seed[d as usize]) {
                    let r = root[d as usize] as usize;
                    hot[r] = hot[r].max(block.loop_depth);
                    value[r] += 8u64.pow(block.loop_depth.min(4));
                }
            }
        }
        Webs { root, web, bad, hot, value }
    }

    /// The roots that stay in VGPRs: those `of` keeps there, a web with a
    /// move in its MFMAs' innermost loop (unless `in_loops`), and a web an
    /// instruction other than an MFMA reads and writes in place.
    fn marked(&self, f: &MFunc, in_loops: bool) -> Vec<bool> {
        let (root, web, hot) = (&self.root, &self.web, &self.hot);
        let whole = |o: &Opnd| whole_reg(f, o);
        let mut bad = self.bad.clone();
        let in_web = |v: u32, bad: &[bool]| web[root[v as usize] as usize] && !bad[root[v as usize] as usize];
        let named = |o: &Opnd, bad: &[bool]| match o {
            Opnd::R(v) if in_web(v.reg, bad) => Some(v.reg),
            _ => None,
        };
        let (mut moved, mut defs, mut uses) = (Vec::new(), Vec::new(), Vec::new());
        // A web stays in VGPRs if a move would sit in its MFMAs' innermost loop,
        // or an instruction other than an MFMA reads and writes it in place.
        for block in &f.blocks {
            let depth = block.loop_depth;
            for inst in &block.insts {
                moved.clear();
                if is_mfma(inst.op) && whole(&inst.defs[0]).is_some_and(|d| in_web(d, &bad)) {
                    moved.extend(inst.uses.iter().enumerate().filter(|&(k, _)| k != 2).filter_map(|(_, o)| named(o, &bad)));
                } else if inst.op == Opc::Copy {
                    let (d, s) = (whole(&inst.defs[0]), whole(&inst.uses[0]));
                    let inside = matches!((d, s), (Some(d), Some(s)) if in_web(d, &bad) && in_web(s, &bad));
                    if !inside {
                        moved.extend(inst.defs.iter().chain(inst.uses.iter()).filter_map(|o| named(o, &bad)));
                    }
                } else {
                    defs.clear();
                    defs.extend(inst.defs.iter().filter_map(|o| named(o, &bad)));
                    uses.clear();
                    uses.extend(inst.uses.iter().filter_map(|o| named(o, &bad)));
                    for &d in defs.iter().filter(|d| uses.contains(d)) {
                        bad[root[d as usize] as usize] = true;
                    }
                    if inst.op != Opc::ImplicitDef {
                        moved.extend(defs.iter().chain(uses.iter()).copied());
                    }
                }
                for &v in &moved {
                    let r = root[v as usize] as usize;
                    if !in_loops && hot[r] > 0 && depth >= hot[r] {
                        bad[r] = true;
                    }
                }
            }
        }
        bad
    }

    /// Where `bank` puts each register of the webs `bad` leaves to move (from
    /// AGPR 0, NONE for the others), and the AGPRs they take.
    fn placed(&self, f: &MFunc, bad: &[bool]) -> Option<(Vec<u32>, u32)> {
        let (root, web, value) = (&self.root, &self.web, &self.value);
        let nv = f.vregs.len();
        let whole = |o: &Opnd| whole_reg(f, o);
        let in_web = |v: u32| web[root[v as usize] as usize] && !bad[root[v as usize] as usize];
        let members: Vec<u32> = (0..nv as u32).filter(|&v| in_web(v)).collect();
        if members.is_empty() {
            return None;
        }
        let mut idx = vec![NONE; nv];
        for (i, &v) in members.iter().enumerate() {
            idx[v as usize] = i as u32;
        }
        let m = members.len();
        let member = |o: &Opnd| match o {
            Opnd::R(v) if idx[v.reg as usize] != NONE => Some(idx[v.reg as usize] as usize),
            _ => None,
        };
        let whole_member = |o: &Opnd| whole(o).filter(|&v| idx[v as usize] != NONE).map(|v| idx[v as usize] as usize);
        // Liveness of the members, by block: upward-exposed uses (a partial
        // definition reads the rest) and whole definitions.
        let nb = f.blocks.len();
        let (mut gen, mut kill) = (vec![Bits::new(m); nb], vec![Bits::new(m); nb]);
        for (b, block) in f.blocks.iter().enumerate() {
            for inst in &block.insts {
                for i in inst.uses.iter().filter_map(member) {
                    if !kill[b].get(i) {
                        gen[b].set(i);
                    }
                }
                for d in &inst.defs {
                    match (whole_member(d), member(d)) {
                        (Some(i), _) => kill[b].set(i),
                        (None, Some(i)) if !kill[b].get(i) => gen[b].set(i),
                        _ => {}
                    }
                }
            }
        }
        let succs: Vec<Vec<usize>> = (0..nb).map(|b| f.successors(b)).collect();
        let mut live_out = vec![Bits::new(m); nb];
        let mut live_in = gen.clone();
        // Each block's live-in, word by word: the live-out less what it kills,
        // and what it reads first.
        let (mut out, mut inn) = (Bits::new(m), Bits::new(m));
        loop {
            let mut changed = false;
            for b in (0..nb).rev() {
                out.0.fill(0);
                for &s in &succs[b] {
                    out.or(&live_in[s]);
                }
                for (w, word) in inn.0.iter_mut().enumerate() {
                    *word = (out.0[w] & !kill[b].0[w]) | gen[b].0[w];
                }
                changed |= live_out[b].or(&out) | live_in[b].or(&inn);
            }
            if !changed {
                break;
            }
        }
        // Interference: a member defined while another is live, except the source
        // of the copy that defines it.
        let (adj, mut copies) = interference(f, m, &live_out, &member, &whole_member);
        // Groups: members joined by copies while they do not interfere, the
        // innermost copies first. Each group's members are kept at its root.
        copies.sort_by(|a, b| b.0.cmp(&a.0).then((a.1, a.2).cmp(&(b.1, b.2))));
        let mut gparent: Vec<u32> = (0..m as u32).collect();
        let mut gmem: Vec<Vec<u32>> = (0..m as u32).map(|i| vec![i]).collect();
        let mut gadj = adj;
        for &(_, _, _, d, s) in &copies {
            let (rd, rs) = (find(&mut gparent, d as u32) as usize, find(&mut gparent, s as u32) as usize);
            if rd == rs || gmem[rs].iter().any(|&j| gadj[rd].get(j as usize)) {
                continue;
            }
            let joined = std::mem::take(&mut gmem[rs]);
            gmem[rd].extend(joined);
            let (into, from) = pair(&mut gadj, rd, rs);
            into.or(from);
            gparent[rs] = rd as u32;
        }
        // AGPRs per group, the weightiest webs first; a web that does not fit
        // stays in VGPRs whole.
        let mut webs: Vec<u32> = (0..nv as u32).filter(|&r| root[r as usize] == r && web[r as usize] && !bad[r as usize]).collect();
        webs.sort_by(|&a, &b| value[b as usize].cmp(&value[a as usize]).then(a.cmp(&b)));
        let mut groups_of: Vec<Vec<usize>> = vec![Vec::new(); nv];
        for i in 0..m {
            if find(&mut gparent, i as u32) as usize == i {
                groups_of[root[members[i] as usize] as usize].push(i);
            }
        }
        let mut base = vec![NONE; m];
        let mut placed: Vec<(usize, u32, u32)> = Vec::new();
        // The AGPRs of the groups placed so far that interfere with the one placed.
        let mut near: Vec<(u32, u32)> = Vec::new();
        for &w in &webs {
            let mark = placed.len();
            for &g in &groups_of[w as usize] {
                let size = f.vregs[members[g] as usize].size as u32;
                let align = if size >= 4 { 4 } else { size.max(1) };
                near.clear();
                near.extend(placed.iter().filter(|&&(h, _, _)| gmem[h].iter().any(|&j| gadj[g].get(j as usize))).map(|&(_, hb, hs)| (hb, hs)));
                let fits = |at: u32| near.iter().all(|&(hb, hs)| at + size <= hb || hb + hs <= at);
                match (0..=AGPRS.saturating_sub(size)).step_by(align as usize).find(|&at| fits(at)) {
                    Some(at) => placed.push((g, at, size)),
                    None => {
                        placed.truncate(mark);
                        break;
                    }
                }
            }
        }
        if placed.is_empty() {
            return None;
        }
        for &(g, at, _) in &placed {
            for &i in &gmem[g] {
                base[i as usize] = at;
            }
        }
        let mut base_of = vec![NONE; nv];
        for (i, &v) in members.iter().enumerate() {
            base_of[v as usize] = base[i];
        }
        Some((base_of, placed.iter().map(|&(_, at, size)| at + size).max().unwrap_or(0)))
    }
}

/// `bank`'s interference of its `m` members (`member` and `whole_member` map
/// an operand to one) given each block's live-out members: both bits of
/// each pair where one is defined while the other is live, except the source
/// of the copy defining it; and the copies between members as (loop depth,
/// block, position, destination, source), last first in each block. Each
/// definition ORs the live set into its row a word at a time and sets its
/// column in the rows of the live members alone.
fn interference(
    f: &MFunc,
    m: usize,
    live_out: &[Bits],
    member: &dyn Fn(&Opnd) -> Option<usize>,
    whole_member: &dyn Fn(&Opnd) -> Option<usize>,
) -> (Vec<Bits>, Vec<(u32, usize, usize, usize, usize)>) {
    let mut adj = vec![Bits::new(m); m];
    let mut copies: Vec<(u32, usize, usize, usize, usize)> = Vec::new();
    let mut others = Bits::new(m);
    for (b, block) in f.blocks.iter().enumerate() {
        let mut live = live_out[b].clone();
        for (k, inst) in block.insts.iter().enumerate().rev() {
            let source = (inst.op == Opc::Copy).then(|| whole_member(&inst.uses[0])).flatten();
            if let (Some(d), Some(s)) = ((inst.op == Opc::Copy).then(|| whole_member(&inst.defs[0])).flatten(), source) {
                copies.push((block.loop_depth, b, k, d, s));
            }
            for i in inst.defs.iter().filter_map(member) {
                others.0.copy_from_slice(&live.0);
                others.clear(i);
                if let Some(s) = source {
                    others.clear(s);
                }
                adj[i].or(&others);
                for (w, &word) in others.0.iter().enumerate() {
                    let mut bits = word;
                    while bits != 0 {
                        adj[w * 64 + bits.trailing_zeros() as usize].set(i);
                        bits &= bits - 1;
                    }
                }
            }
            for i in inst.defs.iter().filter_map(whole_member) {
                live.clear(i);
            }
            for i in inst.uses.iter().filter_map(member) {
                live.set(i);
            }
        }
    }
    (adj, copies)
}

fn acc(op: Gfx9Op, def: Opnd, src: Opnd) -> MInst {
    MInst::new(Opc::Gfx9(op), [def], [src])
}

fn temp(f: &mut MFunc, n: u8) -> VRef {
    f.vregs.push(VRegInfo { class: Class::V, size: n, align: if n > 1 { 2 } else { 1 }, fixed: None });
    VRef { reg: f.vregs.len() as u32 - 1, off: 0, n }
}

/// Dword `k` of a register operand, or the operand (a constant).
fn dword(o: &Opnd, k: u8) -> Opnd {
    match *o {
        Opnd::R(v) => Opnd::R(VRef { off: v.off + k, n: 1, ..v }),
        Opnd::P(Phys::V(r), _) => Opnd::P(Phys::V(r + k as u16), 1),
        Opnd::P(Phys::S(r), _) => Opnd::P(Phys::S(r + k as u16), 1),
        other => other,
    }
}

fn width(o: &Opnd) -> u8 {
    match *o {
        Opnd::R(v) => v.n,
        Opnd::P(_, n) => n,
        Opnd::Imm(_) => 1,
    }
}

/// Puts the registers with a `base_of` in AGPRs `base_of + dword`, then drops
/// the moves whose results nothing reads any more.
pub(super) fn rewrite(f: &mut MFunc, base_of: &[u32]) {
    let banked = |o: &Opnd| matches!(o, Opnd::R(v) if base_of[v.reg as usize] != NONE);
    let agpr = |v: VRef| Opnd::P(Phys::V(AGPR0 + (base_of[v.reg as usize] + v.off as u32) as u16), v.n);
    // Dwords whose one definition is a move of an inline constant: a copy of
    // one into an AGPR writes the constant.
    let mut consts: crate::rir::fx::FxHashMap<(u32, u8), Option<Opnd>> = Default::default();
    for inst in f.blocks.iter().flat_map(|b| &b.insts) {
        for d in &inst.defs {
            let Opnd::R(v) = *d else { continue };
            for k in 0..v.n {
                let c = match (inst.op, inst.uses.first()) {
                    (Opc::VMovB32 | Opc::Copy, Some(&c @ Opnd::Imm(_))) if v.n == 1 && literal_of(&c, Lit::B32).is_none() => Some(c),
                    _ => None,
                };
                consts.entry((v.reg, v.off + k)).and_modify(|e| *e = None).or_insert(c);
            }
        }
    }
    let constant = |o: &Opnd| match *o {
        Opnd::R(v) => consts.get(&(v.reg, v.off)).copied().flatten(),
        _ => None,
    };
    for b in 0..f.blocks.len() {
        let old = std::mem::take(&mut f.blocks[b].insts);
        let mut out = Vec::with_capacity(old.len() + 8);
        for mut inst in old {
            if inst.op == Opc::Copy && (banked(&inst.defs[0]) || banked(&inst.uses[0])) {
                let (dst, src) = (inst.defs[0], inst.uses[0]);
                let side = |o: &Opnd| match *o {
                    Opnd::R(v) if base_of[v.reg as usize] != NONE => agpr(v),
                    other => other,
                };
                let (d, s) = (side(&dst), side(&src));
                for k in 0..width(&dst) {
                    let (dk, sk) = (dword(&d, k), dword(&s, k));
                    match (banked(&dst), banked(&src)) {
                        (true, true) if dk == sk => {}
                        (true, true) => out.push(acc(Gfx9Op::VAccvgprMovB32, dk, sk)),
                        (true, false) if constant(&sk).is_some() => out.push(acc(Gfx9Op::VAccvgprWriteB32, dk, constant(&sk).expect("a constant"))),
                        (true, false) if literal_of(&sk, Lit::B32).is_some() => {
                            let t = temp(f, 1);
                            out.push(MInst::new(Opc::VMovB32, [Opnd::R(t)], [sk]));
                            out.push(acc(Gfx9Op::VAccvgprWriteB32, dk, Opnd::R(t)));
                        }
                        (true, false) => out.push(acc(Gfx9Op::VAccvgprWriteB32, dk, sk)),
                        (false, _) => out.push(acc(Gfx9Op::VAccvgprReadB32, dk, sk)),
                    }
                }
                continue;
            }
            if inst.op == Opc::ImplicitDef && inst.defs.iter().any(banked) {
                let defs: Vec<Opnd> = inst.defs.iter().copied().filter(|d| !banked(d)).collect();
                if !defs.is_empty() {
                    out.push(MInst { defs: Defs::from(&defs[..]), ..inst });
                }
                continue;
            }
            if inst.op == Opc::VMovB32 && inst.defs.len() == 1 && banked(&inst.defs[0]) && width(&inst.defs[0]) == 1 {
                if let src @ Opnd::Imm(_) = inst.uses[0] {
                    if literal_of(&src, Lit::B32).is_none() {
                        let Opnd::R(v) = inst.defs[0] else { unreachable!("a banked register") };
                        out.push(acc(Gfx9Op::VAccvgprWriteB32, agpr(v), src));
                        continue;
                    }
                }
            }
            if is_mfma(inst.op) && banked(&inst.defs[0]) {
                let Opnd::R(d) = inst.defs[0] else { unreachable!("a banked register") };
                let tied = inst.uses[2] == inst.defs[0];
                inst.defs[0] = agpr(d);
                if tied {
                    inst.uses[2] = agpr(d);
                }
            }
            let mut reads: Vec<(VRef, VRef)> = Vec::new();
            for u in 0..inst.uses.len() {
                let Opnd::R(v) = inst.uses[u] else { continue };
                if base_of[v.reg as usize] == NONE {
                    continue;
                }
                let t = match reads.iter().find(|r| r.0 == v) {
                    Some(r) => r.1,
                    None => {
                        let t = temp(f, v.n);
                        for k in 0..v.n {
                            out.push(acc(Gfx9Op::VAccvgprReadB32, dword(&Opnd::R(t), k), dword(&agpr(v), k)));
                        }
                        reads.push((v, t));
                        t
                    }
                };
                inst.uses[u] = Opnd::R(t);
            }
            let mut writes = Vec::new();
            for d in 0..inst.defs.len() {
                let Opnd::R(v) = inst.defs[d] else { continue };
                if base_of[v.reg as usize] == NONE {
                    continue;
                }
                let t = temp(f, v.n);
                inst.defs[d] = Opnd::R(t);
                for k in 0..v.n {
                    writes.push(acc(Gfx9Op::VAccvgprWriteB32, dword(&agpr(v), k), dword(&Opnd::R(t), k)));
                }
            }
            out.push(inst);
            out.extend(writes);
        }
        f.blocks[b].insts = out;
    }
    loop {
        let mut used = vec![false; f.vregs.len()];
        for o in f.blocks.iter().flat_map(|b| &b.insts).flat_map(|i| i.uses.iter()) {
            if let Opnd::R(v) = o {
                used[v.reg as usize] = true;
            }
        }
        let dead = |i: &MInst| {
            matches!(i.op, Opc::VMovB32 | Opc::Copy | Opc::ImplicitDef) && i.defs.iter().all(|d| matches!(d, Opnd::R(v) if !used[v.reg as usize]))
        };
        let before: usize = f.blocks.iter().map(|b| b.insts.len()).sum();
        for block in f.blocks.iter_mut() {
            block.insts.retain(|i| !dead(i));
        }
        if f.blocks.iter().map(|b| b.insts.len()).sum::<usize>() == before {
            break;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rir::{analysis, isel, opt, parse};
    use std::collections::HashMap;

    fn select(text: &str) -> MFunc {
        let mut kernel = parse::parse(text).unwrap_or_else(|e| panic!("{e}"));
        kernel.wave = Target::Gfx950.info().wave;
        let (mut lds, lds_size) = analysis::lds_layout(&kernel);
        let facts = opt::optimize(&mut kernel, &mut lds).unwrap_or_else(|e| panic!("{e}"));
        let uniform = analysis::classify(&kernel);
        isel::select_for(Target::Gfx950, &kernel, &facts, uniform, &lds, lds_size).unwrap_or_else(|e| panic!("{e}")).func
    }

    /// A K loop of `accs` 32x32x8 f16 MFMA accumulators, each stored after it;
    /// `in_loop` also reads the first accumulator with a VALU op every iteration.
    fn kernel(accs: usize, in_loop: bool) -> String {
        let mut s = String::from(
            "ckdsl.ir v1\nkernel @acc {\n  attrs { max_workgroup_size = i:64 }\n  params {\n    %A : ptr<f16,global>\n    \
             %Y : ptr<f32,global>\n    %n : i32\n  }\n  region @entry {\n    %tid : i32 = gpu.thread_id (  ) { axis = s:\"x\" }\n    \
             %c0 : i32 = arith.constant (  ) { ity = s:\"i32\", value = i:0 }\n    %c1 : i32 = arith.constant (  ) { ity = s:\"i32\", value = i:1 }\n    \
             %c4 : i32 = arith.constant (  ) { ity = s:\"i32\", value = i:4 }\n    \
             %z : vec<f32x16> = arith.constant_vec (  ) { elem = s:\"f32\", fill = f:0.0, vec = i:16 }\n",
        );
        let results: Vec<String> = (0..accs).map(|a| format!("%r{a} : vec<f32x16>")).collect();
        let inits: Vec<&str> = (0..accs).map(|_| "%z").collect();
        let iter_args: Vec<String> = (0..accs).map(|a| format!("{{ name = s:\"%acc{a}\", type = s:\"vec<f32x16>\" }}")).collect();
        s += &format!(
            "    {} = scf.for ( %c0, %n, %c1, {} ) {{ iter_args = l:[ {} ], iv = s:\"%k\", iv_type = s:\"i32\", num_iter_args = i:{accs}, unroll = b:false }}\n      region @body {{\n",
            results.join(", "),
            inits.join(", "),
            iter_args.join(", ")
        );
        s += "        %o : i32 = arith.mul ( %k, %c4 )\n        %p : i32 = arith.add ( %o, %tid )\n        \
              %a : vec<f16x4> = memref.global_load_vN ( %A, %p ) { align = i:8, elem_type = s:\"f16\", vec = i:4 }\n";
        let mut yields = Vec::new();
        for a in 0..accs {
            s += &format!("        %d{a} : vec<f32x16> = tile.mma ( %a, %a, %acc{a} ) {{ op_id = s:\"mfma_f32_32x32x8_f16\" }}\n");
            yields.push(format!("%d{a}"));
        }
        if in_loop {
            s += "        %e : f32 = vector.extract ( %acc0 ) { index = i:3 }\n        \
                  memref.global_store_typed ( %Y, %p, %e ) { align = i:4, elem_type = s:\"f32\" }\n";
        }
        s += &format!("        scf.yield ( {} ) {{ num = i:{accs} }}\n      }}\n", yields.join(", "));
        for a in 0..accs {
            for k in [0, 15] {
                s += &format!(
                    "    %e{a}_{k} : f32 = vector.extract ( %r{a} ) {{ index = i:{k} }}\n    \
                     %i{a}_{k} : i32 = arith.constant (  ) {{ ity = s:\"i32\", value = i:{} }}\n    \
                     %q{a}_{k} : i32 = arith.add ( %tid, %i{a}_{k} )\n    \
                     memref.global_store_typed ( %Y, %q{a}_{k}, %e{a}_{k} ) {{ align = i:4, elem_type = s:\"f32\" }}\n",
                    64 * (16 * a + k)
                );
            }
        }
        s + "    cf.return (  )\n  }\n}\n"
    }

    #[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
    enum Key {
        Virt(u32, u8),
        V(u16),
        S(u16),
        Other(Phys),
    }

    fn keys(o: &Opnd) -> Vec<Key> {
        match *o {
            Opnd::R(v) => (0..v.n).map(|k| Key::Virt(v.reg, v.off + k)).collect(),
            Opnd::P(Phys::V(r), n) => (0..n as u16).map(|k| Key::V(r + k)).collect(),
            Opnd::P(Phys::S(r), n) => (0..n as u16).map(|k| Key::S(r + k)).collect(),
            Opnd::P(p, _) => vec![Key::Other(p)],
            Opnd::Imm(_) => Vec::new(),
        }
    }

    fn mix(h: u64, x: u64) -> u64 {
        (h ^ x).wrapping_mul(0x100_0000_01b3).rotate_left(29) ^ 0x9e37_79b9_7f4a_7c15
    }

    /// A number for opcode `op`.
    fn opcode(op: Opc) -> u64 {
        format!("{op:?}").bytes().fold(5u64, |h, b| mix(h, b as u64))
    }

    /// What `f` computes, as value numbers: every instruction without
    /// results (stores, branches) with the values it reads. Moves copy values,
    /// any other result is a function of its operation and the values read.
    /// A block that branches to itself runs `trips` times.
    fn effects(f: &MFunc, trips: usize) -> Vec<u64> {
        let mut regs: HashMap<Key, u64> = HashMap::new();
        let mut out = Vec::new();
        let read = |regs: &HashMap<Key, u64>, o: &Opnd| -> Vec<u64> {
            match *o {
                Opnd::Imm(v) => vec![mix(1, v as u64)],
                _ => keys(o).iter().map(|k| *regs.get(k).unwrap_or(&0xdead)).collect(),
            }
        };
        let moves = |op: Opc| {
            matches!(op, Opc::Copy | Opc::VMovB32)
                || matches!(op, Opc::Gfx9(Gfx9Op::VAccvgprReadB32 | Gfx9Op::VAccvgprWriteB32 | Gfx9Op::VAccvgprMovB32))
        };
        for (b, block) in f.blocks.iter().enumerate() {
            let runs = if f.successors(b).contains(&b) { trips } else { 1 };
            for _ in 0..runs {
                for inst in &block.insts {
                    let defs: Vec<Key> = inst.defs.iter().flat_map(keys).collect();
                    let values: Vec<u64> = inst.uses.iter().flat_map(|o| read(&regs, o)).collect();
                    if moves(inst.op) {
                        for (k, v) in defs.iter().zip(values) {
                            regs.insert(*k, v);
                        }
                    } else if defs.is_empty() {
                        out.push(values.iter().fold(mix(opcode(inst.op), inst.imm as u64), |h, &v| mix(h, v)));
                    } else {
                        let h = values.iter().fold(mix(opcode(inst.op), mix(inst.imm as u64, inst.mods as u64)), |h, &v| mix(h, v));
                        for (k, d) in defs.iter().enumerate() {
                            regs.insert(*d, mix(h, k as u64));
                        }
                    }
                }
            }
        }
        out
    }

    fn agpr_operand(o: &Opnd) -> bool {
        matches!(o, Opnd::P(Phys::V(r), _) if *r >= AGPR0)
    }

    fn insts_of(f: &MFunc) -> impl Iterator<Item = &MInst> {
        f.blocks.iter().flat_map(|b| &b.insts)
    }

    #[test]
    fn accumulators_move_to_agprs_and_compute_the_same() {
        let f = select(&kernel(3, false));
        let mut g = f.clone();
        assert_eq!(bank(&mut g), Some(48));
        assert_eq!(effects(&f, 3), effects(&g, 3));
        let insts = || g.blocks.iter().flat_map(|b| &b.insts);
        let mfmas: Vec<&MInst> = insts().filter(|i| is_mfma(i.op)).collect();
        assert_eq!(mfmas.len(), 3);
        for m in &mfmas {
            assert!(agpr_operand(&m.defs[0]) && m.defs[0] == m.uses[2], "{m:?}");
        }
        // The loop moves nothing between the banks: the accumulators are
        // written once before it (zeros, inline) and read once after it.
        let loop_moves = g.blocks.iter().filter(|b| b.loop_depth > 0).flat_map(|b| &b.insts).filter(|i| matches!(i.op, Opc::Gfx9(Gfx9Op::VAccvgprReadB32 | Gfx9Op::VAccvgprWriteB32 | Gfx9Op::VAccvgprMovB32))).count();
        assert_eq!(loop_moves, 0);
        let writes: Vec<&MInst> = insts().filter(|i| i.op == Opc::Gfx9(Gfx9Op::VAccvgprWriteB32)).collect();
        assert!(writes.iter().all(|w| w.uses[0] == Opnd::Imm(0)), "{writes:?}");
        assert_eq!(writes.len(), 48);
        assert_eq!(insts().filter(|i| i.op == Opc::Gfx9(Gfx9Op::VAccvgprReadB32)).count(), 6);
        // The zero tuple they start from is no longer built.
        let zeros = |f: &MFunc| insts_of(f).filter(|i| matches!(i.op, Opc::VMovB32 | Opc::Copy) && i.uses[0] == Opnd::Imm(0)).count();
        assert_eq!(zeros(&g), zeros(&f) - 16);
    }

    #[test]
    fn accumulators_read_in_their_loop_stay_in_vgprs() {
        let f = select(&kernel(2, true));
        let mut g = f.clone();
        // The first web has a VALU read each iteration; the second moves.
        assert_eq!(bank(&mut g), Some(16));
        assert_eq!(effects(&f, 3), effects(&g, 3));
        let banked: Vec<bool> = g.blocks.iter().flat_map(|b| &b.insts).filter(|i| is_mfma(i.op)).map(|m| agpr_operand(&m.defs[0])).collect();
        assert_eq!(banked, [false, true]);
    }

    #[test]
    fn a_function_short_of_vgprs_banks_instead_of_spilling() {
        let text = kernel(6, false);
        // 96 accumulator dwords where 64 VGPRs are allowed.
        let compiled = crate::rir::compile_any_target_limited(&text, Target::Gfx950, 64).unwrap_or_else(|e| panic!("{e}"));
        let s = &compiled.stats.spill;
        assert_eq!((s.agpr_dwords, s.lds_dwords, s.scratch_dwords), (96, 0, 0), "{s:?}");
        assert!(compiled.stats.vgprs <= 64);
        let listing = compiled.listing();
        let mfmas: Vec<&str> = listing.lines().map(str::trim).filter(|l| l.starts_with("v_mfma")).collect();
        assert_eq!(mfmas.len(), 6);
        // The listing names AGPR k as v(256 + k).
        let first_reg = |l: &str| l.split_whitespace().nth(1).and_then(|d| d.trim_start_matches("v[").split(':').next()?.parse::<u32>().ok());
        assert!(mfmas.iter().all(|l| first_reg(l).is_some_and(|r| r >= 256)), "{listing}");
        // Without the limit the accumulators stay in VGPRs.
        let plain = crate::rir::compile_any_target(&text, Target::Gfx950).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(plain.stats.spill.agpr_dwords, 0);
    }

    /// `bank`'s interference loop as it was before `interference`, kept
    /// verbatim to compare against.
    fn interference_reference(
        f: &MFunc,
        m: usize,
        live_out: &[Bits],
        member: &dyn Fn(&Opnd) -> Option<usize>,
        whole_member: &dyn Fn(&Opnd) -> Option<usize>,
    ) -> (Vec<Bits>, Vec<(u32, usize, usize, usize, usize)>) {
        let mut adj = vec![Bits::new(m); m];
        let mut copies: Vec<(u32, usize, usize, usize, usize)> = Vec::new();
        for (b, block) in f.blocks.iter().enumerate() {
            let mut live = live_out[b].clone();
            for (k, inst) in block.insts.iter().enumerate().rev() {
                let source = (inst.op == Opc::Copy).then(|| whole_member(&inst.uses[0])).flatten();
                if let (Some(d), Some(s)) = ((inst.op == Opc::Copy).then(|| whole_member(&inst.defs[0])).flatten(), source) {
                    copies.push((block.loop_depth, b, k, d, s));
                }
                for i in inst.defs.iter().filter_map(member) {
                    for j in live.ones().collect::<Vec<_>>() {
                        if j != i && Some(j) != source {
                            adj[i].set(j);
                            adj[j].set(i);
                        }
                    }
                }
                for i in inst.defs.iter().filter_map(whole_member) {
                    live.clear(i);
                }
                for i in inst.uses.iter().filter_map(member) {
                    live.set(i);
                }
            }
        }
        (adj, copies)
    }

    /// Blocks of whole and partial copies, MFMAs and VALU instructions over
    /// registers of one to sixteen dwords, a random set of them members, and
    /// random live-out members per block (past 64 and 128 members too).
    #[test]
    fn interference_marks_what_it_marked() {
        for seed in 0..300u64 {
            let mut x = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
            let mut next = |n: u64| {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                (x % n) as u32
            };
            let nv = 4 + next(200) as usize;
            let mut f = MFunc { target: Target::Gfx950, ..MFunc::default() };
            for _ in 0..nv {
                let size = [1, 2, 4, 16][next(4) as usize];
                f.vregs.push(VRegInfo { class: Class::V, size, align: 1, fixed: None });
            }
            let mut idx = vec![NONE; nv];
            let mut m = 0;
            for slot in idx.iter_mut() {
                if next(3) != 0 {
                    *slot = m as u32;
                    m += 1;
                }
            }
            let size = |r: u32| f.vregs[r as usize].size;
            let r = |reg: u32, whole: bool, k: &mut dyn FnMut(u64) -> u32| {
                let n = size(reg);
                if whole || n == 1 {
                    Opnd::R(VRef { reg, off: 0, n })
                } else {
                    Opnd::R(VRef { reg, off: k(u64::from(n)) as u8 % n, n: 1 })
                }
            };
            let mut blocks = Vec::new();
            for _ in 0..1 + next(4) {
                let mut insts = Vec::new();
                for _ in 0..next(40) {
                    let (a, b, c) = (next(nv as u64), next(nv as u64), next(nv as u64));
                    insts.push(match next(4) {
                        0 => MInst::new(Opc::Copy, [r(a, true, &mut next)], [r(b, next(4) != 0, &mut next)]),
                        1 => MInst::new(Opc::Gfx9(Gfx9Op::VMfma16x16x32F16), [r(a, true, &mut next)], [r(b, false, &mut next), r(c, false, &mut next), r(a, true, &mut next)]),
                        2 => MInst::new(Opc::VAddF32, [r(a, false, &mut next)], [r(b, false, &mut next), r(c, false, &mut next)]),
                        _ => MInst::new(Opc::VAddF32, [r(a, next(2) == 0, &mut next)], [r(a, false, &mut next), Opnd::Imm(1)]),
                    });
                }
                blocks.push(MBlock { insts, loop_depth: next(3) });
            }
            f.blocks = blocks;
            let whole = |o: &Opnd| match *o {
                Opnd::R(v) if v.off == 0 && v.n == f.vregs[v.reg as usize].size => Some(v.reg),
                _ => None,
            };
            let member = |o: &Opnd| match o {
                Opnd::R(v) if idx[v.reg as usize] != NONE => Some(idx[v.reg as usize] as usize),
                _ => None,
            };
            let whole_member = |o: &Opnd| whole(o).filter(|&v| idx[v as usize] != NONE).map(|v| idx[v as usize] as usize);
            let live_out: Vec<Bits> = (0..f.blocks.len())
                .map(|_| {
                    let mut live = Bits::new(m);
                    for i in 0..m {
                        if next(4) == 0 {
                            live.set(i);
                        }
                    }
                    live
                })
                .collect();
            let (adj, copies) = interference(&f, m, &live_out, &member, &whole_member);
            let (want, want_copies) = interference_reference(&f, m, &live_out, &member, &whole_member);
            assert_eq!(copies, want_copies, "seed {seed}");
            assert!(adj.iter().zip(&want).all(|(a, b)| a.0 == b.0), "seed {seed}");
        }
    }

    /// `place_with` before it was split into finding, marking and placing,
    /// kept verbatim to compare against.
    fn place_with_reference(f: &MFunc, in_loops: bool) -> Option<(Vec<u32>, u32)> {
        let nv = f.vregs.len();
        let whole = |o: &Opnd| -> Option<u32> {
            match *o {
                Opnd::R(v) if v.off == 0 && v.n == f.vregs[v.reg as usize].size => Some(v.reg),
                _ => None,
            }
        };
        // Registers only whole copies and MFMAs define (or none, undefined): the
        // others, such as a zero tuple several accumulators start from, stay out
        // of webs, and copies from them are moves into the web.
        let mut defined_inside = vec![true; nv];
        for inst in f.blocks.iter().flat_map(|b| &b.insts) {
            let inside = match inst.op {
                Opc::Copy | Opc::ImplicitDef => true,
                op if is_mfma(op) => inst.uses.get(2) == Some(&inst.defs[0]) || matches!(inst.uses.get(2), Some(Opnd::Imm(_))),
                _ => false,
            };
            for d in &inst.defs {
                if let Opnd::R(v) = d {
                    defined_inside[v.reg as usize] &= inside && whole(d).is_some();
                }
            }
        }
        let eligible = |v: u32| f.vregs[v as usize].class == Class::V && f.vregs[v as usize].fixed.is_none() && defined_inside[v as usize];
        // Webs: V registers joined by whole copies, with an MFMA's tied D and C.
        let mut parent: Vec<u32> = (0..nv as u32).collect();
        let mut seed = vec![false; nv];
        let mut pinned = vec![false; nv];
        for inst in f.blocks.iter().flat_map(|b| &b.insts) {
            if is_mfma(inst.op) {
                match (whole(&inst.defs[0]), inst.uses.get(2).and_then(whole)) {
                    (Some(d), Some(c)) if d == c && eligible(d) => seed[d as usize] = true,
                    // C an inline constant (`zero_acc`): the web starts at this D.
                    (Some(d), None) if matches!(inst.uses.get(2), Some(Opnd::Imm(_))) && eligible(d) => seed[d as usize] = true,
                    _ => {
                        for o in inst.defs.iter().chain(inst.uses.get(2)) {
                            if let Opnd::R(v) = o {
                                pinned[v.reg as usize] = true;
                            }
                        }
                    }
                }
            } else if inst.op == Opc::Copy {
                if let (Some(d), Some(s)) = (whole(&inst.defs[0]), whole(&inst.uses[0])) {
                    if eligible(d) && eligible(s) && f.vregs[d as usize].size == f.vregs[s as usize].size {
                        let (rd, rs) = (find(&mut parent, d), find(&mut parent, s));
                        parent[rd as usize] = rs;
                    }
                }
            }
        }
        let root: Vec<u32> = (0..nv as u32).map(|v| find(&mut parent, v)).collect();
        let mut web = vec![false; nv];
        let mut bad = vec![false; nv];
        for v in 0..nv {
            web[root[v] as usize] |= seed[v];
            bad[root[v] as usize] |= pinned[v];
        }
        // The innermost loop of each web's MFMAs, and their weight.
        let (mut hot, mut value) = (vec![0u32; nv], vec![0u64; nv]);
        for block in &f.blocks {
            for inst in block.insts.iter().filter(|i| is_mfma(i.op)) {
                if let Some(d) = whole(&inst.defs[0]).filter(|&d| seed[d as usize]) {
                    let r = root[d as usize] as usize;
                    hot[r] = hot[r].max(block.loop_depth);
                    value[r] += 8u64.pow(block.loop_depth.min(4));
                }
            }
        }
        let in_web = |v: u32, web: &[bool], bad: &[bool]| web[root[v as usize] as usize] && !bad[root[v as usize] as usize];
        // A web stays in VGPRs if a move would sit in its MFMAs' innermost loop,
        // or an instruction other than an MFMA reads and writes it in place.
        for block in &f.blocks {
            let depth = block.loop_depth;
            for inst in &block.insts {
                let mut moved: Vec<u32> = Vec::new();
                let named = |o: &Opnd| match o {
                    Opnd::R(v) if in_web(v.reg, &web, &bad) => Some(v.reg),
                    _ => None,
                };
                if is_mfma(inst.op) && whole(&inst.defs[0]).is_some_and(|d| in_web(d, &web, &bad)) {
                    moved.extend(inst.uses.iter().enumerate().filter(|&(k, _)| k != 2).filter_map(|(_, o)| named(o)));
                } else if inst.op == Opc::Copy {
                    let (d, s) = (whole(&inst.defs[0]), whole(&inst.uses[0]));
                    let inside = matches!((d, s), (Some(d), Some(s)) if in_web(d, &web, &bad) && in_web(s, &web, &bad));
                    if !inside {
                        moved.extend(inst.defs.iter().chain(inst.uses.iter()).filter_map(named));
                    }
                } else {
                    let defs: Vec<u32> = inst.defs.iter().filter_map(named).collect();
                    let uses: Vec<u32> = inst.uses.iter().filter_map(named).collect();
                    for &d in defs.iter().filter(|d| uses.contains(d)) {
                        bad[root[d as usize] as usize] = true;
                    }
                    if inst.op != Opc::ImplicitDef {
                        moved.extend(defs.into_iter().chain(uses));
                    }
                }
                for v in moved {
                    let r = root[v as usize] as usize;
                    if !in_loops && hot[r] > 0 && depth >= hot[r] {
                        bad[r] = true;
                    }
                }
            }
        }
        let members: Vec<u32> = (0..nv as u32).filter(|&v| in_web(v, &web, &bad)).collect();
        if members.is_empty() {
            return None;
        }
        let mut idx = vec![NONE; nv];
        for (i, &v) in members.iter().enumerate() {
            idx[v as usize] = i as u32;
        }
        let m = members.len();
        let member = |o: &Opnd| match o {
            Opnd::R(v) if idx[v.reg as usize] != NONE => Some(idx[v.reg as usize] as usize),
            _ => None,
        };
        let whole_member = |o: &Opnd| whole(o).filter(|&v| idx[v as usize] != NONE).map(|v| idx[v as usize] as usize);
        // Liveness of the members, by block: upward-exposed uses (a partial
        // definition reads the rest) and whole definitions.
        let nb = f.blocks.len();
        let (mut gen, mut kill) = (vec![Bits::new(m); nb], vec![Bits::new(m); nb]);
        for (b, block) in f.blocks.iter().enumerate() {
            for inst in &block.insts {
                for i in inst.uses.iter().filter_map(member) {
                    if !kill[b].get(i) {
                        gen[b].set(i);
                    }
                }
                for d in &inst.defs {
                    match (whole_member(d), member(d)) {
                        (Some(i), _) => kill[b].set(i),
                        (None, Some(i)) if !kill[b].get(i) => gen[b].set(i),
                        _ => {}
                    }
                }
            }
        }
        let succs: Vec<Vec<usize>> = (0..nb).map(|b| f.successors(b)).collect();
        let mut live_out = vec![Bits::new(m); nb];
        let mut live_in = gen.clone();
        loop {
            let mut changed = false;
            for b in (0..nb).rev() {
                let mut out = Bits::new(m);
                for &s in &succs[b] {
                    out.or(&live_in[s]);
                }
                let mut inn = out.clone();
                for i in kill[b].ones() {
                    inn.clear(i);
                }
                inn.or(&gen[b]);
                changed |= live_out[b].or(&out) | live_in[b].or(&inn);
            }
            if !changed {
                break;
            }
        }
        // Interference: a member defined while another is live, except the source
        // of the copy that defines it.
        let (adj, mut copies) = interference(f, m, &live_out, &member, &whole_member);
        // Groups: members joined by copies while they do not interfere, the
        // innermost copies first.
        copies.sort_by(|a, b| b.0.cmp(&a.0).then((a.1, a.2).cmp(&(b.1, b.2))));
        let mut gparent: Vec<u32> = (0..m as u32).collect();
        let mut gmem: Vec<Bits> = (0..m)
            .map(|i| {
                let mut s = Bits::new(m);
                s.set(i);
                s
            })
            .collect();
        let mut gadj = adj;
        for &(_, _, _, d, s) in &copies {
            let (rd, rs) = (find(&mut gparent, d as u32) as usize, find(&mut gparent, s as u32) as usize);
            if rd == rs || gadj[rd].intersects(&gmem[rs]) {
                continue;
            }
            let (mem, adjs) = (gmem[rs].clone(), gadj[rs].clone());
            gmem[rd].or(&mem);
            gadj[rd].or(&adjs);
            gparent[rs] = rd as u32;
        }
        // AGPRs per group, the weightiest webs first; a web that does not fit
        // stays in VGPRs whole.
        let mut webs: Vec<u32> = (0..nv as u32).filter(|&r| root[r as usize] == r && web[r as usize] && !bad[r as usize]).collect();
        webs.sort_by(|&a, &b| value[b as usize].cmp(&value[a as usize]).then(a.cmp(&b)));
        let mut base = vec![NONE; m];
        let mut placed: Vec<(usize, u32, u32)> = Vec::new();
        for &w in &webs {
            let groups: Vec<usize> = (0..m).filter(|&i| root[members[i] as usize] == w && find(&mut gparent, i as u32) as usize == i).collect();
            let mark = placed.len();
            for &g in &groups {
                let size = f.vregs[members[g] as usize].size as u32;
                let align = if size >= 4 { 4 } else { size.max(1) };
                let fits = |at: u32| placed.iter().all(|&(h, hb, hs)| !gadj[g].intersects(&gmem[h]) || at + size <= hb || hb + hs <= at);
                match (0..=AGPRS.saturating_sub(size)).step_by(align as usize).find(|&at| fits(at)) {
                    Some(at) => placed.push((g, at, size)),
                    None => {
                        placed.truncate(mark);
                        break;
                    }
                }
            }
        }
        if placed.is_empty() {
            return None;
        }
        for &(g, at, _) in &placed {
            for i in gmem[g].ones() {
                base[i] = at;
            }
        }
        let mut base_of = vec![NONE; nv];
        for (i, &v) in members.iter().enumerate() {
            base_of[v as usize] = base[i];
        }
        Some((base_of, placed.iter().map(|&(_, at, size)| at + size).max().unwrap_or(0)))
    }

    /// A random function for the placement: whole and partial copies, MFMAs
    /// whose C is their D, a constant or another register, VALU instructions
    /// (some reading and writing one register), implicit definitions and
    /// moves of constants, over registers of one to sixteen dwords (some
    /// fixed, some SGPRs), in blocks of loop depth 0 to 2 with branches
    /// forward and back.
    fn random_function(seed: u64) -> MFunc {
        let mut x = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
        let mut next = |n: u64| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            (x % n) as u32
        };
        let nv = 4 + next(120) as usize;
        let mut f = MFunc { target: Target::Gfx950, ..MFunc::default() };
        for _ in 0..nv {
            let size = [1, 2, 4, 16][next(4) as usize];
            let class = if next(12) == 0 { Class::S } else { Class::V };
            let fixed = (next(20) == 0).then_some(0);
            f.vregs.push(VRegInfo { class, size, align: 1, fixed });
        }
        let size = |f: &MFunc, r: u32| f.vregs[r as usize].size;
        let nb = 1 + next(6) as usize;
        for b in 0..nb {
            let mut insts = Vec::new();
            for _ in 0..next(30) {
                let (a, s, c) = (next(nv as u64), next(nv as u64), next(nv as u64));
                let r = |reg: u32, whole: bool, k: u32| {
                    let n = size(&f, reg);
                    if whole || n == 1 {
                        Opnd::R(VRef { reg, off: 0, n })
                    } else {
                        Opnd::R(VRef { reg, off: (k % n as u32) as u8, n: 1 })
                    }
                };
                let k = next(16);
                insts.push(match next(9) {
                    0 | 1 => MInst::new(Opc::Copy, [r(a, true, k)], [r(s, next(5) != 0, k)]),
                    2 | 3 => MInst::new(Opc::Gfx9(Gfx9Op::VMfma16x16x32F16), [r(a, true, k)], [r(s, false, k), r(c, false, k), r(a, true, k)]),
                    4 => MInst::new(Opc::Gfx9(Gfx9Op::VMfma16x16x32F16), [r(a, true, k)], [r(s, false, k), r(c, false, k), Opnd::Imm(0)]),
                    5 => MInst::new(Opc::Gfx9(Gfx9Op::VMfma16x16x32F16), [r(a, true, k)], [r(s, false, k), r(c, false, k), r(c, true, k)]),
                    6 => MInst::new(Opc::VAddF32, [r(a, false, k)], [r(s, false, k), r(c, false, k)]),
                    7 => MInst::new(Opc::VAddF32, [r(a, next(2) == 0, k)], [r(a, false, k), Opnd::Imm(1)]),
                    _ if next(2) == 0 => MInst::new(Opc::ImplicitDef, [r(a, true, k)], []),
                    _ => MInst::new(Opc::VMovB32, [r(a, false, k)], [Opnd::Imm(0)]),
                });
            }
            if next(3) == 0 {
                let target = next(nb as u64) as i64;
                insts.push(MInst::new(Opc::SCbranchScc1, [], []).with_imm(target));
            }
            let depth = if b == 0 || b + 1 == nb { 0 } else { next(3) };
            f.blocks.push(MBlock { insts, loop_depth: depth });
        }
        f
    }

    /// The placement agrees with the code before it was split into finding,
    /// marking and placing, kept verbatim (`place_with_reference`), with and
    /// without the in-loop rule; `place_apart` gives both where the rule
    /// marks other webs and None only where the two agree.
    #[test]
    fn placements_place_what_they_placed() {
        let (mut placed, mut apart, mut joined) = (0, 0, 0);
        for seed in 0..1500u64 {
            let f = random_function(seed);
            let want = [place_with_reference(&f, false), place_with_reference(&f, true)];
            assert_eq!(place_with(&f, false), want[0], "seed {seed}");
            assert_eq!(place_with(&f, true), want[1], "seed {seed}");
            match place_apart(&f) {
                Some((in_loops, strict)) => {
                    assert_eq!((in_loops, strict), (want[1].clone(), want[0].clone()), "seed {seed}");
                    apart += 1;
                }
                None => assert_eq!(want[0], want[1], "seed {seed}"),
            }
            if let Some((base_of, _)) = &want[1] {
                placed += 1;
                let banked: Vec<u32> = base_of.iter().copied().filter(|&b| b != NONE).collect();
                let distinct: std::collections::BTreeSet<u32> = banked.iter().copied().collect();
                joined += usize::from(distinct.len() < banked.len());
            }
        }
        // The random functions do exercise the placement, the in-loop rule
        // and the coalescing of copies.
        assert!(placed > 300 && apart > 50 && joined > 50, "{placed} placed, {apart} apart, {joined} joined");
    }

    /// The same on the selected kernels of these tests.
    #[test]
    fn placements_of_selected_kernels_place_what_they_placed() {
        for (accs, in_loop) in [(3, false), (2, true), (6, false), (4, true)] {
            let f = select(&kernel(accs, in_loop));
            for flag in [false, true] {
                assert_eq!(place_with(&f, flag), place_with_reference(&f, flag), "{accs} {in_loop} {flag}");
            }
        }
    }
}
