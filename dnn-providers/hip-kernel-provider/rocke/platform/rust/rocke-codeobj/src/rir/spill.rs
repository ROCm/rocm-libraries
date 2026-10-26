// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

//! Spilling. When more dwords of a register class are live at one point
//! than the allocation may use, whole coalesced webs leave registers, taken
//! by how much of the excess pressure they cover per weighted access (loop
//! depth weighs eight times per level). A one-dword value computed from
//! constants (and, for VGPRs, scalar registers that stay live) is recomputed
//! before each use (rematerialization); any other web is stored after each
//! definition and reloaded before each use, copies within the web vanishing.
//!
//! VGPR webs go to per-lane LDS slots while free LDS costs no occupancy,
//! then to scratch. A store right after the definition runs under the
//! definition's EXEC, which covers every lane of every later use: values
//! never flow out of a divergent region into a wider one.
//!
//! SGPR webs go to lanes of VGPRs (`v_writelane_b32` / `v_readlane_b32`,
//! 32 dwords per VGPR), which ignore EXEC.
//!
//! VGPR spill code first uses placeholder scratch accesses, marked
//! `M_SPILL` with the web's dword slot in `imm` and the LDS address as an
//! extra source so that its register stays live; once registers are
//! assigned, `finish` gives the most accessed webs LDS slots and the rest
//! scratch.

use super::mir::*;
use super::regalloc::{self, Live};
use crate::error::CompileError;

/// Marks a placeholder spill access (and the LDS address computation).
const M_SPILL: u32 = 1 << 31;

/// Where spill slots may go.
#[derive(Clone, Copy, Debug, Default)]
pub struct SpillConfig {
    /// Static LDS pool in bytes; LDS slots follow it.
    pub lds_size: u32,
    /// Work-items per workgroup at most: an LDS slot holds a dword for each.
    pub wg_size: u32,
    /// LDS slots (dwords per work-item) that leave the occupancy unchanged.
    pub lds_slots: u32,
}

/// What spilling did.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SpillStats {
    /// Values recomputed at their uses.
    pub remat: u32,
    pub lds_dwords: u32,
    pub scratch_dwords: u32,
    /// SGPR dwords kept in VGPR lanes.
    pub sgpr_dwords: u32,
    /// Bytes added to the group segment.
    pub lds_bytes: u32,
    /// Private segment bytes per work-item.
    pub scratch_bytes: u32,
    /// Spill accesses inserted (placeholders: LDS ones split per dword later).
    pub reloads: u32,
    pub stores: u32,
}

/// A spilled VGPR web: its first dword slot, size, and weighted accesses.
struct Web {
    slot: u32,
    size: u32,
    heat: u64,
}

enum Choice {
    /// Members (register, dword offset in the web), web size, weighted accesses.
    Spill(Vec<(u32, u8)>, u32, u64),
    /// A register and its defining instruction (block, index).
    Remat(u32, (usize, usize)),
}

pub(super) struct Spiller {
    cfg: SpillConfig,
    webs: Vec<Web>,
    next_slot: u32,
    /// The per-lane LDS address register, made with the first VGPR spill
    /// when LDS slots exist.
    addr: Option<VRef>,
    /// VGPRs holding spilled SGPRs, 32 dwords each.
    lanes: Vec<VRef>,
    next_lane: u32,
    /// Registers spilling created: temporaries, lanes, the LDS address.
    temp: Vec<bool>,
    stats: SpillStats,
    /// Per block, the instruction buffer the last rewrite left: rewriting
    /// into memory already in use costs a fraction of fresh memory.
    spare: Vec<Vec<MInst>>,
    /// The same for the liveness analyses between rewrites.
    scratch: regalloc::LiveScratch,
}

fn weight(depth: u32) -> u64 {
    1u64 << (3 * depth.min(6))
}

fn store_op(n: u8) -> Opc {
    match n {
        1 => Opc::ScratchStoreB32,
        2 => Opc::ScratchStoreB64,
        3 => Opc::ScratchStoreB96,
        _ => Opc::ScratchStoreB128,
    }
}

fn load_op(n: u8) -> Opc {
    match n {
        1 => Opc::ScratchLoadB32,
        2 => Opc::ScratchLoadB64,
        3 => Opc::ScratchLoadB96,
        _ => Opc::ScratchLoadB128,
    }
}

fn sub(v: VRef, off: u8, n: u8) -> VRef {
    VRef { reg: v.reg, off: v.off + off, n }
}

/// Whether `inst` computes its one result purely from constants and, for
/// a VGPR, single-definition SGPRs, so it can be repeated where those are
/// live. Scalar candidates are moves of constants only: anything else might
/// clobber SCC where the copy lands.
fn pure_def(inst: &MInst, class: Class, vregs: &[VRegInfo], defs: &[u32]) -> bool {
    if inst.mods != 0 || inst.defs.len() != 1 {
        return false;
    }
    if class == Class::S {
        return inst.op == Opc::Copy && matches!(inst.uses[..], [Opnd::Imm(_)]);
    }
    let pure = match inst.op {
        Opc::Copy | Opc::VMovB32 => true,
        op => matches!(op.fmt(), Fmt::Vop1 | Fmt::Vop2 | Fmt::Vop3) && op.flags() == F_USE_EXEC,
    };
    pure && inst.uses.iter().all(|o| match o {
        Opnd::Imm(_) => true,
        Opnd::R(s) => vregs[s.reg as usize].class == Class::S && defs[s.reg as usize] == 1,
        Opnd::P(..) => false,
    })
}

/// The linear index of the instruction of each use of each register, in
/// program order, all in one table.
struct UsePositions {
    start: Vec<u32>,
    at: Vec<u32>,
}

impl UsePositions {
    fn new(f: &MFunc, nv: usize) -> Self {
        let uses = || f.blocks.iter().flat_map(|b| &b.insts).enumerate().flat_map(|(i, inst)| {
            inst.uses.iter().filter_map(move |o| o.vref().map(|v| (i as u32, v.reg as usize)))
        });
        let mut start = vec![0u32; nv + 1];
        for (_, reg) in uses() {
            start[reg + 1] += 1;
        }
        for r in 0..nv {
            start[r + 1] += start[r];
        }
        let mut next = start.clone();
        let mut at = vec![0u32; start[nv] as usize];
        for (i, reg) in uses() {
            at[next[reg] as usize] = i;
            next[reg] += 1;
        }
        UsePositions { start, at }
    }

    fn of(&self, reg: u32) -> &[u32] {
        &self.at[self.start[reg as usize] as usize..self.start[reg as usize + 1] as usize]
    }
}

impl Spiller {
    pub(super) fn new(f: &MFunc, cfg: SpillConfig) -> Self {
        Spiller {
            cfg,
            webs: Vec::new(),
            next_slot: 0,
            addr: None,
            lanes: Vec::new(),
            next_lane: 0,
            temp: vec![false; f.vregs.len()],
            stats: SpillStats::default(),
            spare: Vec::new(),
            scratch: regalloc::LiveScratch::default(),
        }
    }

    /// The liveness of `f`, in the buffers of earlier analyses.
    pub(super) fn analyze(&mut self, f: &MFunc) -> Result<Live, CompileError> {
        regalloc::analyze_reusing(f, &mut self.scratch)
    }

    /// Keeps the buffers of a liveness no longer needed.
    pub(super) fn recycle(&mut self, live: Live) {
        self.scratch.recycle(live);
    }

    fn new_temp(&mut self, f: &mut MFunc, class: Class, n: u8) -> VRef {
        let reg = f.vregs.len() as u32;
        let align = match (class, n) {
            (Class::V, _) | (Class::S, 1) => 1,
            (Class::S, 2) => 2,
            _ => 4,
        };
        f.vregs.push(VRegInfo { class, size: n, align, fixed: None });
        self.temp.push(true);
        VRef { reg, off: 0, n }
    }

    /// The per-lane LDS address, thread id times four, computed at entry.
    fn lds_addr(&mut self, f: &mut MFunc) -> Option<VRef> {
        if self.addr.is_none() && self.cfg.lds_slots > 0 && self.cfg.wg_size > 0 {
            let tid = f.vregs.iter().position(|r| r.class == Class::V && r.fixed == Some(0))?;
            let a = self.new_temp(f, Class::V, 1);
            let t = Opnd::R(VRef { reg: tid as u32, off: 0, n: 1 });
            let inst = MInst::new(Opc::VLshlrevB32, [Opnd::R(a)], [Opnd::Imm(2), t]).with_mods(M_SPILL);
            f.blocks[0].insts.insert(0, inst);
            self.addr = Some(a);
        }
        self.addr
    }

    /// Makes the lane VGPRs of every SGPR slot taken so far, undefined at entry.
    fn make_lanes(&mut self, f: &mut MFunc) {
        while (self.lanes.len() as u32) * 32 < self.next_lane {
            let l = self.new_temp(f, Class::V, 1);
            f.blocks[0].insts.insert(0, MInst::new(Opc::ImplicitDef, [Opnd::R(l)], []));
            self.lanes.push(l);
        }
    }

    /// The lane VGPR and lane of SGPR spill slot `slot`.
    fn lane(&mut self, _f: &mut MFunc, slot: u32) -> (VRef, i64) {
        (self.lanes[(slot / 32) as usize], (slot % 32) as i64)
    }

    /// Stores `data` to the slots from `slot`: VGPRs four dwords at a time
    /// into placeholders, SGPRs a dword per lane.
    fn store(&mut self, f: &mut MFunc, out: &mut Vec<MInst>, class: Class, data: VRef, slot: u32) {
        if class == Class::S {
            for k in 0..data.n {
                let (lane, idx) = self.lane(f, slot + k as u32);
                let srcs = [Opnd::R(sub(data, k, 1)), Opnd::Imm(idx as u32), Opnd::R(lane)];
                out.push(MInst::new(Opc::VWritelaneB32, [Opnd::R(lane)], srcs));
                self.stats.stores += 1;
            }
            return;
        }
        let mut k = 0u8;
        while k < data.n {
            let n = (data.n - k).min(4);
            let mut uses = Uses::new();
            uses.push(Opnd::R(sub(data, k, n)));
            if let Some(a) = self.addr {
                uses.push(Opnd::R(a));
            }
            let at = slot + k as u32;
            out.push(MInst::new(store_op(n), Defs::new(), uses).with_imm(at as i64).with_mods(M_SPILL));
            self.stats.stores += 1;
            k += n;
        }
    }

    fn load(&mut self, f: &mut MFunc, out: &mut Vec<MInst>, class: Class, dst: VRef, slot: u32) {
        if class == Class::S {
            for k in 0..dst.n {
                let (lane, idx) = self.lane(f, slot + k as u32);
                out.push(MInst::new(Opc::VReadlaneB32, [Opnd::R(sub(dst, k, 1))], [Opnd::R(lane), Opnd::Imm(idx as u32)]));
                self.stats.reloads += 1;
            }
            return;
        }
        let mut k = 0u8;
        while k < dst.n {
            let n = (dst.n - k).min(4);
            let mut uses = Uses::new();
            if let Some(a) = self.addr {
                uses.push(Opnd::R(a));
            }
            let at = slot + k as u32;
            out.push(MInst::new(load_op(n), [Opnd::R(sub(dst, k, n))], uses).with_imm(at as i64).with_mods(M_SPILL));
            self.stats.reloads += 1;
            k += n;
        }
    }

    /// Spills webs of `class` until at most `target` of its dwords are live
    /// at any point. `known` is the liveness of `f` if the caller has it,
    /// and gets it back when this leaves `f` as it is.
    pub(super) fn reduce(&mut self, f: &mut MFunc, class: Class, target: u32, known: &mut Option<Live>) -> Result<(), CompileError> {
        let mut stalls = 0;
        let mut last_excess = u64::MAX;
        loop {
            self.temp.resize(f.vregs.len(), false);
            let mut live = match known.take() {
                Some(live) => live,
                None => self.analyze(f)?,
            };
            let prof = regalloc::profile(f, &live, class);
            let excess: u64 = prof.iter().map(|&p| p.saturating_sub(target) as u64).sum();
            let Some((peak_slot, &peak)) = prof.iter().enumerate().max_by_key(|&(s, &p)| (p, std::cmp::Reverse(s)))
            else {
                *known = Some(live);
                return Ok(());
            };
            if peak <= target {
                *known = Some(live);
                return Ok(());
            }
            // Every round removes excess; rounds that do not are spilling in
            // circles around reload temporaries.
            stalls = if excess < last_excess { 0 } else { stalls + 1 };
            last_excess = excess;
            let chosen =
                if stalls > 8 { Vec::new() } else { self.choose(f, &mut live, class, &prof, peak_slot, peak - target, target) };
            if chosen.is_empty() {
                let what = if class == Class::S { "SGPRs" } else { "VGPRs" };
                return Err(CompileError::new(
                    "ROCKE_DIRECT_REGISTERS",
                    format!("out of {what}: {peak} dwords live and no value left to spill"),
                ));
            }
            self.apply(f, class, &chosen);
            self.recycle(live);
        }
    }

    /// Webs of `class` to spill or values to rematerialize, covering `need`
    /// dwords at the peak slot, best ratio of covered excess to weighted
    /// cost first.
    #[allow(clippy::too_many_arguments)]
    fn choose(
        &mut self,
        f: &MFunc,
        live: &mut Live,
        class: Class,
        prof: &[u32],
        peak_slot: usize,
        need: u32,
        target: u32,
    ) -> Vec<Choice> {
        let nv = f.vregs.len();
        let groups = regalloc::coalesce(f, live);
        let mut parent = groups.parent.clone();
        let root: Vec<(u32, u8)> = (0..nv as u32).map(|v| regalloc::find(&mut parent, v)).collect();
        // Excess pressure summed over slots, for coverage by segments.
        let mut excess = vec![0u64; prof.len() + 1];
        for (s, &p) in prof.iter().enumerate() {
            excess[s + 1] = excess[s] + p.saturating_sub(target) as u64;
        }
        let unit = |v: u32, k: u32| live.base[v as usize] + k as usize;
        let mut members = vec![0u32; nv];
        for v in 0..nv {
            members[root[v].0 as usize] += 1;
        }
        // Weighted accesses per root (copies inside a web vanish), the
        // weighted uses of each register, definitions, and use positions.
        let mut cost = vec![0u64; nv];
        let mut use_cost = vec![0u64; nv];
        let mut defs = vec![0u32; nv];
        let mut def_inst: Vec<Option<(u32, u32)>> = vec![None; nv];
        // Units the peak instruction reads or writes stay live as temporaries.
        let mut at_peak_inst = vec![false; live.base[nv]];
        let peak_inst = (peak_slot / 2) as u32;
        let mut idx = 0u32;
        for (b, block) in f.blocks.iter().enumerate() {
            let w = weight(block.loop_depth);
            for (k, inst) in block.insts.iter().enumerate() {
                let internal = inst.op == Opc::Copy
                    && matches!((inst.defs[0], inst.uses[0]), (Opnd::R(d), Opnd::R(s))
                        if root[d.reg as usize].0 == root[s.reg as usize].0
                            && root[d.reg as usize].1 + d.off == root[s.reg as usize].1 + s.off);
                for o in inst.defs.iter().chain(&inst.uses) {
                    if let Opnd::R(v) = o {
                        if !internal {
                            cost[root[v.reg as usize].0 as usize] += w * v.n as u64;
                        }
                        if idx == peak_inst {
                            let u0 = unit(v.reg, v.off as u32);
                            at_peak_inst[u0..u0 + v.n as usize].iter_mut().for_each(|x| *x = true);
                        }
                    }
                }
                for o in &inst.defs {
                    if let Opnd::R(v) = o {
                        defs[v.reg as usize] += 1;
                        def_inst[v.reg as usize] = Some((b as u32, k as u32));
                    }
                }
                for o in &inst.uses {
                    if let Opnd::R(v) = o {
                        use_cost[v.reg as usize] += w;
                    }
                }
                idx += 1;
            }
        }
        let uses_at = UsePositions::new(f, nv);
        let live_at = |s: VRef, slot: u32| -> bool {
            (0..s.n as u32).all(|k| {
                live.segs(unit(s.reg, s.off as u32 + k)).iter().any(|&(a, e)| a <= slot && slot <= e)
            })
        };
        let mut cands: Vec<(f64, u32, bool, u32)> = Vec::new();
        for r in 0..nv as u32 {
            if root[r as usize].0 != r
                || f.vregs[r as usize].class != class
                || groups.gfixed[r as usize].is_some()
                || self.temp[r as usize]
            {
                continue;
            }
            let size = groups.gsize[r as usize];
            let freed = (0..size)
                .filter(|&k| {
                    let u = unit(r, k);
                    !at_peak_inst[u] && live.segs(u).iter().any(|&(s, e)| s as usize <= peak_slot && peak_slot <= e as usize)
                })
                .count() as u32;
            if freed == 0 {
                continue;
            }
            let coverage: u64 = (0..size)
                .flat_map(|k| live.segs(unit(r, k)).iter())
                .map(|&(s, e)| excess[e as usize + 1] - excess[s as usize])
                .sum();
            let remat = members[r as usize] == 1
                && size == 1
                && defs[r as usize] == 1
                && def_inst[r as usize].is_some_and(|(b, k)| {
                    let d = &f.blocks[b as usize].insts[k as usize];
                    pure_def(d, class, &f.vregs, &defs)
                        && d.uses.iter().all(|o| match o {
                            Opnd::R(s) => uses_at.of(r).iter().all(|&i| live_at(*s, 2 * i)),
                            _ => true,
                        })
                });
            // A recomputation is one instruction per use; a spill access is
            // a memory operation per dword (priced at four) or, for SGPRs, a
            // lane read or write (priced at two).
            let per = if class == Class::S { 2 } else { 4 };
            let c = if remat { use_cost[r as usize].max(1) } else { per * cost[r as usize] };
            cands.push((coverage as f64 / (c as f64 + 1.0), r, remat, freed));
        }
        cands.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
        let mut covered = 0u32;
        let mut out = Vec::new();
        for (_, r, remat, freed) in cands {
            if covered >= need {
                break;
            }
            covered += freed;
            if remat {
                let (b, k) = def_inst[r as usize].expect("defined once");
                out.push(Choice::Remat(r, (b as usize, k as usize)));
            } else {
                let list: Vec<(u32, u8)> =
                    (0..nv as u32).filter(|&v| root[v as usize].0 == r).map(|v| (v, root[v as usize].1)).collect();
                out.push(Choice::Spill(list, groups.gsize[r as usize], cost[r as usize]));
            }
        }
        out
    }

    fn apply(&mut self, f: &mut MFunc, class: Class, chosen: &[Choice]) {
        let nv = f.vregs.len();
        // Dword slot of each spilled register's dword 0.
        let mut slot_of: Vec<Option<u32>> = vec![None; nv];
        let mut remat: Vec<Option<MInst>> = vec![None; nv];
        for c in chosen {
            match c {
                Choice::Spill(list, size, heat) => {
                    let slot = if class == Class::S {
                        let s = self.next_lane;
                        self.next_lane += size;
                        self.stats.sgpr_dwords += size;
                        s
                    } else {
                        let s = self.next_slot;
                        self.next_slot += size;
                        self.webs.push(Web { slot: s, size: *size, heat: *heat });
                        s
                    };
                    for &(v, off) in list {
                        slot_of[v as usize] = Some(slot + off as u32);
                    }
                }
                Choice::Remat(v, (b, k)) => {
                    remat[*v as usize] = Some(f.blocks[*b].insts[*k].clone());
                    // Dropped below: an implicit def with no operands.
                    f.blocks[*b].insts[*k] = MInst::new(Opc::ImplicitDef, [], []);
                    self.stats.remat += 1;
                }
            }
        }
        // The LDS address and lanes go in at entry once the rematerialized
        // definitions (found by position) are gone, before the rewrite
        // takes the blocks.
        if class == Class::V && chosen.iter().any(|c| matches!(c, Choice::Spill(..))) {
            self.lds_addr(f);
        }
        self.make_lanes(f);
        let spilled = |o: &Opnd, slot_of: &[Option<u32>]| -> Option<(VRef, u32)> {
            match o {
                Opnd::R(v) if (v.reg as usize) < slot_of.len() => slot_of[v.reg as usize].map(|s| (*v, s + v.off as u32)),
                _ => None,
            }
        };
        let touched = |o: &Opnd| match o {
            Opnd::R(v) => slot_of.get(v.reg as usize).is_some_and(Option::is_some) || remat.get(v.reg as usize).is_some_and(Option::is_some),
            _ => false,
        };
        self.spare.resize_with(f.blocks.len(), Vec::new);
        for b in 0..f.blocks.len() {
            // A block no spilled or recomputed register appears in, and with
            // no definition to drop, stays as it is.
            let unchanged = f.blocks[b].insts.iter().all(|inst| {
                !(inst.op == Opc::ImplicitDef && inst.defs.iter().all(|d| spilled(d, &slot_of).is_some()))
                    && !inst.defs.iter().chain(&inst.uses).any(touched)
            });
            if unchanged {
                continue;
            }
            let mut old = std::mem::take(&mut f.blocks[b].insts);
            let mut out = std::mem::take(&mut self.spare[b]);
            out.clear();
            out.reserve(old.len() + 16);
            for mut inst in old.drain(..) {
                if inst.op == Opc::ImplicitDef
                    && (inst.defs.is_empty() || inst.defs.iter().all(|d| spilled(d, &slot_of).is_some()))
                {
                    continue;
                }
                if !inst.defs.iter().chain(&inst.uses).any(touched) {
                    out.push(inst);
                    continue;
                }
                // Recompute rematerialized sources; a copy of one takes the
                // constant or SGPR directly.
                for u in 0..inst.uses.len() {
                    let Opnd::R(v) = inst.uses[u] else { continue };
                    let Some(def) = remat.get(v.reg as usize).and_then(|d| d.clone()) else { continue };
                    if inst.op == Opc::Copy && matches!(def.op, Opc::Copy | Opc::VMovB32) {
                        inst.uses[u] = def.uses[0];
                        continue;
                    }
                    let t = self.new_temp(f, class, 1);
                    let mut clone = def;
                    clone.defs = [Opnd::R(t)].into();
                    out.push(clone);
                    inst.uses[u] = Opnd::R(t);
                }
                let class_of = |f: &MFunc, o: &Opnd| match o {
                    Opnd::R(v) => Some(f.vregs[v.reg as usize].class),
                    _ => None,
                };
                if inst.op == Opc::Copy {
                    let (d, s) = (inst.defs[0], inst.uses[0]);
                    match (spilled(&d, &slot_of), spilled(&s, &slot_of)) {
                        (Some((_, sd)), Some((_, ss))) if sd == ss => continue,
                        (None, Some((sv, ss))) => {
                            let Opnd::R(dv) = d else { unreachable!("a virtual register") };
                            if class_of(f, &d) == Some(class) {
                                self.load(f, &mut out, class, dv, ss);
                            } else {
                                let t = self.new_temp(f, class, sv.n);
                                self.load(f, &mut out, class, t, ss);
                                out.push(MInst::new(Opc::Copy, [d], [Opnd::R(t)]));
                            }
                            continue;
                        }
                        (Some((dv, sd)), None) => {
                            let data = match s {
                                Opnd::R(sv) if class_of(f, &s) == Some(class) => sv,
                                _ => {
                                    let t = self.new_temp(f, class, dv.n);
                                    out.push(MInst::new(Opc::Copy, [Opnd::R(t)], [s]));
                                    t
                                }
                            };
                            self.store(f, &mut out, class, data, sd);
                            continue;
                        }
                        (Some((dv, sd)), Some((_, ss))) => {
                            let t = self.new_temp(f, class, dv.n);
                            self.load(f, &mut out, class, t, ss);
                            self.store(f, &mut out, class, t, sd);
                            continue;
                        }
                        _ => {}
                    }
                }
                // Reload spilled sources into temporaries and write spilled
                // results to temporaries stored afterwards; a result tied to a
                // source of the same range shares its temporary.
                let mut reloaded: Vec<(u32, u8, VRef)> = Vec::new();
                for u in 0..inst.uses.len() {
                    if let Some((v, slot)) = spilled(&inst.uses[u], &slot_of) {
                        let t = match reloaded.iter().find(|r| r.0 == slot && r.1 == v.n) {
                            Some(r) => r.2,
                            None => {
                                let t = self.new_temp(f, class, v.n);
                                self.load(f, &mut out, class, t, slot);
                                reloaded.push((slot, v.n, t));
                                t
                            }
                        };
                        inst.uses[u] = Opnd::R(t);
                    }
                }
                let mut stores: Vec<(VRef, u32)> = Vec::new();
                for d in 0..inst.defs.len() {
                    if let Some((v, slot)) = spilled(&inst.defs[d], &slot_of) {
                        let t = match reloaded.iter().find(|r| r.0 == slot && r.1 == v.n) {
                            Some(r) => r.2,
                            None => self.new_temp(f, class, v.n),
                        };
                        inst.defs[d] = Opnd::R(t);
                        stores.push((t, slot));
                    }
                }
                out.push(inst);
                for (t, slot) in stores {
                    self.store(f, &mut out, class, t, slot);
                }
            }
            f.blocks[b].insts = out;
            self.spare[b] = old;
        }
    }

    /// Finishes the allocated function: VGPR placeholders become LDS or
    /// scratch accesses (the most accessed webs take the LDS slots), and
    /// lane writes drop the liveness-only source.
    pub(super) fn finish(mut self, f: &mut MFunc) -> Result<SpillStats, CompileError> {
        let mut order: Vec<usize> = (0..self.webs.len()).collect();
        order.sort_by(|&a, &b| self.webs[b].heat.cmp(&self.webs[a].heat).then(a.cmp(&b)));
        // Per dword slot: (in LDS, LDS slot or scratch dword).
        let mut place: Vec<(bool, u32)> = vec![(false, 0); self.next_slot as usize];
        let (mut lds, mut scratch) = (0u32, 0u32);
        for &w in &order {
            let web = &self.webs[w];
            let in_lds = self.addr.is_some() && lds + web.size <= self.cfg.lds_slots;
            for k in 0..web.size {
                place[(web.slot + k) as usize] = if in_lds { (true, lds + k) } else { (false, scratch + k) };
            }
            if in_lds {
                lds += web.size;
            } else {
                scratch += web.size;
            }
        }
        if scratch * 4 > 4096 {
            return Err(CompileError::new(
                "ROCKE_DIRECT_REGISTERS",
                format!("out of VGPRs: {scratch} spilled dwords exceed the 4 KiB scratch offset range"),
            ));
        }
        let stride = 4 * self.cfg.wg_size;
        for block in f.blocks.iter_mut() {
            let old = std::mem::take(&mut block.insts);
            let mut out = Vec::with_capacity(old.len() + 8);
            for mut inst in old {
                if inst.op == Opc::VWritelaneB32 && inst.uses.len() == 3 {
                    inst.uses = Uses::from(&inst.uses[..2]);
                }
                if inst.mods & M_SPILL == 0 {
                    out.push(inst);
                    continue;
                }
                if inst.op == Opc::VLshlrevB32 {
                    if lds > 0 {
                        out.push(MInst { mods: 0, ..inst });
                    }
                    continue;
                }
                let slot = inst.imm as u32;
                let load = !inst.defs.is_empty();
                let data = if load { inst.defs[0] } else { inst.uses[0] };
                let Opnd::P(Phys::V(r), n) = data else {
                    return Err(CompileError::new("ROCKE_DIRECT_INTERNAL", "spill access to a non-VGPR"));
                };
                let (in_lds, at) = place[slot as usize];
                if in_lds {
                    let addr = if load { inst.uses[0] } else { inst.uses[1] };
                    for k in 0..n as u32 {
                        let offset = (self.cfg.lds_size + place[(slot + k) as usize].1 * stride) as i64;
                        let reg = Opnd::P(Phys::V(r + k as u16), 1);
                        out.push(if load {
                            MInst::new(Opc::DsLoadB32, [reg], [addr]).with_imm(offset)
                        } else {
                            MInst::new(Opc::DsStoreB32, [], [addr, reg]).with_imm(offset)
                        });
                    }
                } else {
                    let offset = 4 * at as i64;
                    out.push(if load {
                        MInst::new(inst.op, [data], []).with_imm(offset)
                    } else {
                        MInst::new(inst.op, [], [data]).with_imm(offset)
                    });
                }
            }
            block.insts = out;
        }
        self.stats.lds_dwords = lds;
        self.stats.scratch_dwords = scratch;
        self.stats.lds_bytes = lds * stride;
        self.stats.scratch_bytes = scratch * 4;
        Ok(self.stats)
    }
}

#[cfg(test)]
pub(super) mod tests {
    use super::*;
    use std::collections::HashMap;

    /// A register dword in either form of a function.
    #[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
    enum Key {
        Virt(u32, u8),
        V(u16),
        S(u16),
    }

    fn mix(h: u64, x: u64) -> u64 {
        (h ^ x).wrapping_mul(0x100_0000_01b3).rotate_left(29) ^ 0x9e37_79b9_7f4a_7c15
    }

    fn keys(o: &Opnd) -> Vec<Key> {
        match *o {
            Opnd::R(v) => (0..v.n).map(|k| Key::Virt(v.reg, v.off + k)).collect(),
            Opnd::P(Phys::V(r), n) => (0..n as u16).map(|k| Key::V(r + k)).collect(),
            Opnd::P(Phys::S(r), n) => (0..n as u16).map(|k| Key::S(r + k)).collect(),
            _ => Vec::new(),
        }
    }

    /// The effects of running `f` as value numbers: every global store's
    /// operation and operand values, in order. Blocks run in layout order,
    /// a block that branches to itself `trips` times. Arithmetic results are
    /// functions of the operation and the values read; spill accesses move
    /// values through per-lane LDS and scratch memory and VGPR lanes.
    pub(in super::super) fn effects(f: &MFunc, trips: usize) -> Vec<u64> {
        let mut regs: HashMap<Key, u64> = HashMap::new();
        // The thread id, in v0 in both forms.
        if let Some(t) = f.vregs.iter().position(|r| r.class == Class::V && r.fixed == Some(0)) {
            regs.insert(Key::Virt(t as u32, 0), 12345);
        }
        regs.insert(Key::V(0), 12345);
        let (mut lds, mut scratch) = (HashMap::new(), HashMap::new());
        let mut lanes: HashMap<(Key, u32), u64> = HashMap::new();
        let mut out = Vec::new();
        let name = |op: Opc| op.name().bytes().fold(5u64, |h, b| mix(h, b as u64));
        let read = |regs: &HashMap<Key, u64>, o: &Opnd| -> Vec<u64> {
            match *o {
                Opnd::Imm(v) => vec![mix(1, v as u64)],
                _ => keys(o).iter().map(|k| *regs.get(k).unwrap_or(&0xdead)).collect(),
            }
        };
        for (b, block) in f.blocks.iter().enumerate() {
            let runs = if f.successors(b).contains(&b) { trips } else { 1 };
            for _ in 0..runs {
                for inst in &block.insts {
                    let defs: Vec<Key> = inst.defs.iter().flat_map(keys).collect();
                    match inst.op {
                        Opc::Copy | Opc::VMovB32 | Opc::SMovB32 => {
                            for (k, v) in defs.iter().zip(read(&regs, &inst.uses[0])) {
                                regs.insert(*k, v);
                            }
                        }
                        Opc::DsStoreB32 => {
                            lds.insert(inst.imm, read(&regs, &inst.uses[1])[0]);
                        }
                        Opc::DsLoadB32 => {
                            regs.insert(defs[0], *lds.get(&inst.imm).unwrap_or(&0xbad));
                        }
                        op if op.name().starts_with("scratch_store") => {
                            for (k, v) in read(&regs, &inst.uses[0]).into_iter().enumerate() {
                                scratch.insert(inst.imm + 4 * k as i64, v);
                            }
                        }
                        op if op.name().starts_with("scratch_load") => {
                            for (k, d) in defs.iter().enumerate() {
                                regs.insert(*d, *scratch.get(&(inst.imm + 4 * k as i64)).unwrap_or(&0xbad));
                            }
                        }
                        Opc::VWritelaneB32 => {
                            let Opnd::Imm(lane) = inst.uses[1] else { panic!("lane") };
                            lanes.insert((defs[0], lane), read(&regs, &inst.uses[0])[0]);
                        }
                        Opc::VReadlaneB32 => {
                            let Opnd::Imm(lane) = inst.uses[1] else { panic!("lane") };
                            let k = keys(&inst.uses[0])[0];
                            regs.insert(defs[0], *lanes.get(&(k, lane)).unwrap_or(&0xbad));
                        }
                        Opc::GStoreB32 | Opc::GStoreB128 => {
                            let h = inst.uses.iter().flat_map(|o| read(&regs, o)).fold(name(inst.op), mix);
                            out.push(h);
                        }
                        Opc::ImplicitDef | Opc::SCbranchScc1 | Opc::SCmpLtU32 | Opc::SEndpgm => {}
                        op => {
                            let h = inst.uses.iter().flat_map(|o| read(&regs, o)).fold(name(op), mix);
                            for (k, d) in defs.iter().enumerate() {
                                regs.insert(*d, mix(h, k as u64));
                            }
                        }
                    }
                }
            }
        }
        out
    }

    /// A kernel with `nv` VGPR and `ns` SGPR values defined up front, VGPR
    /// tuples built from them, loop-carried accumulators updated by a loop
    /// that reads everything, and stores of everything after it.
    pub(in super::super) fn random_kernel(seed: u64, nv: usize, ns: usize) -> MFunc {
        let mut seed = seed;
        let mut next = |n: usize| {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (seed >> 33) as usize % n
        };
        let mut f = MFunc::default();
        let reg = |f: &mut MFunc, class: Class, size: u8, fixed: Option<u16>| {
            let align = if class == Class::S && size > 1 { 2 } else { 1 };
            f.vregs.push(VRegInfo { class, size, align, fixed });
            VRef { reg: f.vregs.len() as u32 - 1, off: 0, n: size }
        };
        let r = Opnd::R;
        let tid = reg(&mut f, Class::V, 1, Some(0));
        let mut entry = Vec::new();
        let t = reg(&mut f, Class::V, 1, None);
        entry.push(MInst::new(Opc::Copy, [r(t)], [r(tid)]));
        let mut vs = vec![t];
        for i in 0..nv {
            let d = reg(&mut f, Class::V, 1, None);
            let a = vs[next(vs.len())];
            let inst = match next(3) {
                0 => MInst::new(Opc::Copy, [r(d)], [Opnd::Imm(1000 + i as u32)]),
                1 => MInst::new(Opc::VAddNcU32, [r(d)], [Opnd::Imm(i as u32), r(a)]),
                _ => MInst::new(Opc::VMulF32, [r(d)], [r(a), r(vs[next(vs.len())])]),
            };
            entry.push(inst);
            vs.push(d);
        }
        let mut ss = Vec::new();
        for i in 0..ns {
            let d = reg(&mut f, Class::S, 1, None);
            entry.push(if ss.is_empty() || next(2) == 0 {
                MInst::new(Opc::Copy, [r(d)], [Opnd::Imm(77 + i as u32)])
            } else {
                MInst::new(Opc::SAddI32, [r(d)], [r(ss[next(ss.len())]), Opnd::Imm(3)])
            });
            ss.push(d);
        }
        let mut tuples = Vec::new();
        for _ in 0..3 {
            let d = reg(&mut f, Class::V, 4, None);
            for k in 0..4 {
                entry.push(MInst::new(Opc::Copy, [r(sub(d, k, 1))], [r(vs[next(vs.len())])]));
            }
            tuples.push(d);
        }
        let mut accs = Vec::new();
        for _ in 0..4 {
            let d = reg(&mut f, Class::V, 1, None);
            entry.push(MInst::new(Opc::Copy, [r(d)], [Opnd::Imm(0)]));
            accs.push(d);
        }
        let iv = reg(&mut f, Class::S, 1, None);
        entry.push(MInst::new(Opc::Copy, [r(iv)], [Opnd::Imm(0)]));
        // The loop: every accumulator takes one more value and one scalar
        // each trip; each value is read on some trip.
        let mut body = Vec::new();
        for (j, &acc) in accs.iter().enumerate() {
            for i in (j..vs.len()).step_by(accs.len()) {
                let n = reg(&mut f, Class::V, 1, None);
                body.push(MInst::new(Opc::VAddF32, [r(n)], [r(vs[i]), r(acc)]));
                body.push(MInst::new(Opc::Copy, [r(acc)], [r(n)]));
            }
            for i in (j..ss.len()).step_by(accs.len()) {
                let n = reg(&mut f, Class::V, 1, None);
                body.push(MInst::new(Opc::VAddNcU32, [r(n)], [r(ss[i]), r(acc)]));
                body.push(MInst::new(Opc::Copy, [r(acc)], [r(n)]));
            }
        }
        body.push(MInst::new(Opc::SAddI32, [r(iv)], [r(iv), Opnd::Imm(1)]));
        body.push(MInst::new(Opc::SCmpLtU32, [], [r(iv), Opnd::Imm(3)]));
        body.push(MInst::new(Opc::SCbranchScc1, [], []).with_imm(1));
        let mut exit = Vec::new();
        for &v in vs.iter().chain(&accs) {
            exit.push(MInst::new(Opc::GStoreB32, [], [r(t), r(v)]));
        }
        for &s in &ss {
            let n = reg(&mut f, Class::V, 1, None);
            exit.push(MInst::new(Opc::VAddNcU32, [r(n)], [r(s), r(t)]));
            exit.push(MInst::new(Opc::GStoreB32, [], [r(t), r(n)]));
        }
        for &d in &tuples {
            exit.push(MInst::new(Opc::GStoreB128, [], [r(t), r(d)]));
        }
        exit.push(MInst::new(Opc::SEndpgm, [], []));
        f.blocks = vec![
            MBlock { insts: entry, loop_depth: 0 },
            MBlock { insts: body, loop_depth: 1 },
            MBlock { insts: exit, loop_depth: 0 },
        ];
        f
    }

    #[test]
    fn spilled_code_computes_the_same_values() {
        let mut total = SpillStats::default();
        for (seed, nv, ns, lds_slots) in [(1, 60, 30, 0), (2, 80, 40, 8), (3, 120, 12, 64), (4, 40, 70, 4)] {
            let f = random_kernel(seed, nv, ns);
            let want = effects(&f, 3);
            let mut g = f.clone();
            let limits = regalloc::Limits { vgprs: 24, sgprs: 20 };
            let cfg = SpillConfig { lds_size: 256, wg_size: 32, lds_slots };
            let a = regalloc::allocate_spilling(&mut g, limits, cfg).expect("allocates");
            assert!(a.num_vgprs <= 24 && a.num_sgprs <= 20, "{a:?}");
            assert!(a.spill.lds_dwords + a.spill.scratch_dwords + a.spill.remat + a.spill.sgpr_dwords > 0);
            assert!(a.spill.lds_dwords <= lds_slots, "{a:?}");
            assert_eq!(effects(&g, 3), want, "seed {seed}: {a:?}");
            // The placeholders are gone and the code encodes.
            assert!(g.blocks.iter().flat_map(|b| &b.insts).all(|i| i.mods & M_SPILL == 0));
            super::super::post::expand_pseudos(&mut g).expect("expands");
            super::super::asm::assemble(&g).expect("encodes");
            total.remat += a.spill.remat;
            total.lds_dwords += a.spill.lds_dwords;
            total.scratch_dwords += a.spill.scratch_dwords;
            total.sgpr_dwords += a.spill.sgpr_dwords;
        }
        // Every kind of spill took part.
        assert!(total.remat > 0 && total.lds_dwords > 0 && total.scratch_dwords > 0 && total.sgpr_dwords > 0, "{total:?}");
    }

    #[test]
    fn allocation_without_spilling_computes_the_same_values() {
        for seed in 10..20 {
            let f = random_kernel(seed, 30, 20);
            let want = effects(&f, 3);
            let mut g = f.clone();
            regalloc::allocate(&mut g).expect("fits");
            assert_eq!(effects(&g, 3), want, "seed {seed}");
        }
    }
}
