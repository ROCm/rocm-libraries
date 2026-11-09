// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

//! Peephole rewrites into cheaper gfx11 sequences, on virtual registers after
//! instruction selection and dead-copy removal, before VALU fusion,
//! scheduling and register allocation. Every rewrite instantiates a rule that the offline
//! superoptimizer proved equal for all inputs with bit-vector SMT, under the
//! kernels' float mode (IEEE mode, denormals kept): `PROVEN` lists the rules,
//! generated with the `v_perm_b32` composition table from the proofs.
//!
//! A rewrite replaces the last instruction of a pattern (the root) and drops
//! the producers left without uses. Producers are single-definition virtual
//! registers defined earlier in the same block, with no EXEC write in between
//! and their sources unchanged up to the root, which then reads those sources
//! itself. A rewrite that removes or adds an SCC definition needs SCC dead
//! there.

use super::isel::{is_inline_f32, is_inline_int};
use super::mir::*;

include!("peephole_table.rs");

/// Rule families, in the order they are tried; `nopeephole_<family>` in
/// `ROCKE_DIRECT_DEBUG` turns one off (`nopeephole` all of them).
pub const FAMILIES: [&str; 11] = [
    "addr", "bounds", "cselect", "perm", "pack", "minmax", "bfe", "and", "shl", "sshl", "cndmask",
];

/// How often each family fired.
#[derive(Clone, Debug, Default)]
pub struct Stats {
    pub fired: [u32; FAMILIES.len()],
}

/// The proven rules this pass instantiates, each with the precondition the
/// matcher establishes (empty when there is none).
pub fn proven_rules() -> &'static [(&'static str, &'static str)] {
    PROVEN
}

const NONE: u32 = u32::MAX;
const NULL: Opnd = Opnd::P(Phys::Null, 1);

/// Optimizes `f` in place.
pub fn optimize(f: &mut MFunc) -> Stats {
    // The flags as `debug_flag` reads them, with one lookup of the variable.
    let flags = std::env::var("ROCKE_DIRECT_DEBUG").unwrap_or_default();
    let off = |name: &str| flags.split(',').any(|f| f.trim() == name);
    if off("nopeephole") {
        return Stats::default();
    }
    let enabled: [bool; FAMILIES.len()] = std::array::from_fn(|k| !off(&format!("nopeephole_{}", FAMILIES[k])));
    run(f, enabled)
}

/// `optimize` with only the families `allowed` keeps: the rules valid on a
/// target that is not gfx11.
pub fn optimize_with(f: &mut MFunc, allowed: [bool; FAMILIES.len()]) -> Stats {
    let flags = std::env::var("ROCKE_DIRECT_DEBUG").unwrap_or_default();
    let off = |name: &str| flags.split(',').any(|f| f.trim() == name);
    if off("nopeephole") {
        return Stats::default();
    }
    let enabled: [bool; FAMILIES.len()] = std::array::from_fn(|k| allowed[k] && !off(&format!("nopeephole_{}", FAMILIES[k])));
    run(f, enabled)
}

fn run(f: &mut MFunc, enabled: [bool; FAMILIES.len()]) -> Stats {
    let mut stats = Stats::default();
    let succs: Vec<Vec<usize>> = (0..f.blocks.len()).map(|b| f.successors(b)).collect();
    let mut pass = Pass::new(f, enabled);
    for b in 0..pass.f.blocks.len() {
        pass.block(b, &succs);
    }
    stats.fired = pass.fired;
    stats
}

struct Pass<'f> {
    f: &'f mut MFunc,
    enabled: [bool; FAMILIES.len()],
    /// First unit (dword) of each virtual register; one past the end last.
    base: Vec<u32>,
    uses: Vec<u32>,
    defs: Vec<u32>,
    /// Block (number + 1) and index of each unit's latest definition seen.
    last: Vec<(u32, u32)>,
    /// The unit never holds a signaling NaN (f32 rules' precondition);
    /// computed at its first use.
    quiet: Vec<bool>,
    quiet_known: bool,
    /// Virtual registers before the pass made any.
    orig_vregs: usize,
    /// The unit's one definition is a SALU instruction.
    salu_def: Vec<bool>,
    /// Uses of a unit by `v_and_b32` with a constant, in its defining block.
    and_uses: Vec<u32>,
    /// Block of each single-definition unit's definition.
    def_block: Vec<u32>,
    /// `smax(n, 0)` registers made for the bounds rule, with the block
    /// they are confined to (none: placed after n's definition, so valid
    /// wherever n is).
    limits: Vec<(VRef, VRef, Option<u32>)>,
    stamp: u32,
    dead: Vec<bool>,
    /// Instructions to insert before an index of the current block.
    insert: Vec<(usize, MInst)>,
    /// The worklist of `kill`, kept for its allocation.
    work: Vec<usize>,
    fired: [u32; FAMILIES.len()],
}

fn vrefs(opnds: &[Opnd]) -> impl Iterator<Item = VRef> + '_ {
    opnds.iter().filter_map(Opnd::vref)
}

fn is_imm(o: &Opnd, v: u32) -> bool {
    *o == Opnd::Imm(v)
}

fn imm(o: &Opnd) -> Option<u32> {
    match *o {
        Opnd::Imm(v) => Some(v),
        _ => None,
    }
}

fn dword(v: VRef, k: u8) -> VRef {
    VRef { reg: v.reg, off: v.off + k, n: 1 }
}

fn is_snan(bits: u32) -> bool {
    bits & 0x7f80_0000 == 0x7f80_0000 && bits & 0x007f_ffff != 0 && bits & 0x0040_0000 == 0
}

/// Opcodes whose f32 result is never a signaling NaN: in IEEE mode, min/max
/// quiet a signaling source, and arithmetic returns a quiet NaN.
fn quiets(op: Opc) -> bool {
    matches!(
        op,
        Opc::VMaxF32
            | Opc::VMinF32
            | Opc::VMax3F32
            | Opc::VMin3F32
            | Opc::VAddF32
            | Opc::VSubF32
            | Opc::VSubrevF32
            | Opc::VMulF32
            | Opc::VFmaF32
            | Opc::VFmacF32
            | Opc::VFmaMixF32
            | Opc::VLdexpF32
            | Opc::VExpF32
            | Opc::VLogF32
            | Opc::VRcpF32
            | Opc::VRsqF32
            | Opc::VCvtF32I32
            | Opc::VCvtF32U32
            | Opc::VDot2F32F16
    )
}

/// The instruction computes only its register results.
fn pure(inst: &MInst) -> bool {
    let flags = inst.op.flags();
    let fmt_ok = matches!(
        inst.op.fmt(),
        Fmt::Sop1 | Fmt::Sop2 | Fmt::Sopk | Fmt::Vop1 | Fmt::Vop2 | Fmt::Vop3 | Fmt::Vop3b | Fmt::Vopc | Fmt::Vop3p
    ) || inst.op == Opc::Copy;
    fmt_ok
        && flags & (F_BRANCH | F_END | F_DEF_EXEC | F_VMEM_LOAD | F_VMEM_STORE | F_LDS | F_SMEM | F_FENCE | F_TIED) == 0
        && inst.defs.iter().all(|d| matches!(d, Opnd::R(_)) || *d == NULL)
}

/// No physical registers among the operands (a null carry-out is fine).
fn virtual_only(inst: &MInst) -> bool {
    inst.defs.iter().chain(inst.uses.iter()).all(|o| !matches!(o, Opnd::P(p, _) if *p != Phys::Null))
}

impl<'f> Pass<'f> {
    fn new(f: &'f mut MFunc, enabled: [bool; FAMILIES.len()]) -> Self {
        let mut base = Vec::with_capacity(f.vregs.len() + 1);
        let mut total = 0u32;
        for info in &f.vregs {
            base.push(total);
            total += info.size as u32;
        }
        base.push(total);
        let n = total as usize;
        let orig_vregs = f.vregs.len();
        let mut pass = Pass {
            f,
            enabled,
            base,
            uses: vec![0; n],
            defs: vec![0; n],
            last: vec![(0, NONE); n],
            quiet: vec![true; n],
            quiet_known: false,
            orig_vregs,
            salu_def: vec![false; n],
            and_uses: vec![0; n],
            def_block: Vec::new(),
            limits: Vec::new(),
            stamp: 0,
            dead: Vec::new(),
            insert: Vec::new(),
            work: Vec::new(),
            fired: [0; FAMILIES.len()],
        };
        let mut def_block = vec![NONE; n];
        for (b, block) in pass.f.blocks.iter().enumerate() {
            for inst in &block.insts {
                for v in vrefs(&inst.uses) {
                    for u in unit_range(&pass.base, v) {
                        pass.uses[u] += 1;
                    }
                }
                let salu = is_salu(inst.op);
                for v in vrefs(&inst.defs) {
                    for u in unit_range(&pass.base, v) {
                        pass.defs[u] += 1;
                        def_block[u] = b as u32;
                        pass.salu_def[u] = salu;
                    }
                }
            }
        }
        for (b, block) in pass.f.blocks.iter().enumerate() {
            for inst in &block.insts {
                if let Some(t) = and_const_source(inst) {
                    for u in unit_range(&pass.base, t) {
                        if def_block[u] == b as u32 {
                            pass.and_uses[u] += 1;
                        }
                    }
                }
            }
        }
        pass.def_block = def_block;
        pass
    }

    fn units(&self, v: VRef) -> std::ops::Range<usize> {
        unit_range(&self.base, v)
    }

    /// Greatest fixpoint: a unit is quiet unless some definition may write a
    /// signaling NaN to it. Computed when a rule first needs it, over every
    /// block, `current` being the block taken out for optimization; no
    /// rewrite changes whether a unit is quiet.
    fn ensure_quiet(&mut self, current: &[MInst]) {
        if self.quiet_known {
            return;
        }
        self.quiet_known = true;
        for u in 0..self.quiet.len() {
            self.quiet[u] = self.defs[u] > 0;
        }
        // Registers the pass made itself are integer temporaries.
        for reg in self.orig_vregs..self.f.vregs.len() {
            for u in self.units(VRef { reg: reg as u32, off: 0, n: self.f.vregs[reg].size }) {
                self.quiet[u] = false;
            }
        }
        // Copies and selects pass their sources on and are revisited until
        // nothing changes; any other instruction settles in one visit.
        let mut copies: Vec<&MInst> = Vec::new();
        for block in self.f.blocks.iter().map(|b| &b.insts[..]).chain([current]) {
            for inst in block {
                match inst.op {
                    op if quiets(op) => {}
                    Opc::Copy | Opc::VMovB32 | Opc::DsSwizzleB32 | Opc::VCndmaskB32 => copies.push(inst),
                    _ => {
                        for v in vrefs(&inst.defs) {
                            for u in unit_range(&self.base, v) {
                                self.quiet[u] = false;
                            }
                        }
                    }
                }
            }
        }
        loop {
            let mut changed = false;
            for inst in &copies {
                let src_quiet = |o: &Opnd| match *o {
                    Opnd::Imm(v) => !is_snan(v),
                    Opnd::R(v) => unit_range(&self.base, v).all(|u| self.quiet[u]),
                    Opnd::P(..) => false,
                };
                let ok = src_quiet(&inst.uses[0]) && (inst.op != Opc::VCndmaskB32 || src_quiet(&inst.uses[1]));
                if ok {
                    continue;
                }
                for v in vrefs(&inst.defs) {
                    for u in unit_range(&self.base, v) {
                        if self.quiet[u] {
                            self.quiet[u] = false;
                            changed = true;
                        }
                    }
                }
            }
            if !changed {
                break;
            }
        }
    }

    fn is_quiet(&self, o: &Opnd) -> bool {
        match *o {
            Opnd::Imm(v) => !is_snan(v),
            Opnd::R(v) => self.units(v).all(|u| self.quiet[u]),
            Opnd::P(..) => false,
        }
    }

    fn class(&self, v: VRef) -> Class {
        self.f.vregs[v.reg as usize].class
    }

    fn is_sgpr(&self, o: &Opnd) -> bool {
        matches!(o, Opnd::R(v) if self.class(*v) == Class::S)
    }

    fn new_sreg(&mut self) -> VRef {
        let reg = self.f.vregs.len() as u32;
        self.f.vregs.push(VRegInfo { class: Class::S, size: 1, align: 1, fixed: None });
        let total = *self.base.last().expect("base has an end");
        self.base.push(total + 1);
        self.uses.push(0);
        self.defs.push(1);
        self.last.push((self.stamp, NONE));
        self.quiet.push(false);
        self.salu_def.push(true);
        self.and_uses.push(0);
        self.def_block.push(NONE);
        VRef { reg, off: 0, n: 1 }
    }

    /// The index in the current block of the one instruction defining every
    /// unit of `o`, before `before`, when each unit has a single definition.
    fn producer(&self, insts: &[MInst], o: &Opnd, before: usize) -> Option<usize> {
        let Opnd::R(v) = *o else { return None };
        let mut at = None;
        for u in self.units(v) {
            let (stamp, i) = self.last[u];
            if self.defs[u] != 1 || stamp != self.stamp || i == NONE || i as usize >= before {
                return None;
            }
            if at.is_some_and(|a| a != i) {
                return None;
            }
            at = Some(i);
        }
        let p = at? as usize;
        let inst = &insts[p];
        (!self.dead[p] && inst.mods == 0 && virtual_only(inst)).then_some(p)
    }

    fn single_use(&self, o: &Opnd) -> bool {
        match *o {
            Opnd::R(v) => self.units(v).all(|u| self.uses[u] == 1),
            _ => false,
        }
    }

    /// `o` holds at `to` the value it held at `from`: no definition between.
    fn stable(&self, insts: &[MInst], o: &Opnd, from: usize, to: usize) -> bool {
        let Opnd::R(v) = *o else { return true };
        let range = self.units(v);
        if range.clone().all(|u| self.defs[u] == 1) {
            return true;
        }
        !insts[from + 1..to].iter().enumerate().any(|(k, inst)| {
            !self.dead[from + 1 + k]
                && vrefs(&inst.defs).any(|d| {
                    let r = self.units(d);
                    r.start < range.end && range.start < r.end
                })
        })
    }

    fn exec_stable(insts: &[MInst], from: usize, to: usize) -> bool {
        !insts[from + 1..to].iter().any(|k| {
            k.op.flags() & F_DEF_EXEC != 0 || k.defs.iter().any(|d| matches!(d, Opnd::P(Phys::ExecLo, _)))
        })
    }

    /// Nothing reads SCC after `at` (in the current block, then at the top
    /// of each successor) before something writes it.
    fn scc_dead_after(&self, insts: &[MInst], at: usize, b: usize, succs: &[Vec<usize>]) -> bool {
        let scan = |list: &[MInst], skip_dead: Option<usize>| -> Option<bool> {
            for (k, inst) in list.iter().enumerate() {
                if skip_dead.is_some_and(|start| self.dead.get(start + k).copied().unwrap_or(false)) {
                    continue;
                }
                let flags = inst.op.flags();
                if flags & F_USE_SCC != 0 {
                    return Some(false);
                }
                if flags & F_DEF_SCC != 0 {
                    return Some(true);
                }
            }
            None
        };
        if let Some(dead) = scan(&insts[at + 1..], Some(at + 1)) {
            return dead;
        }
        succs[b].iter().all(|&s| {
            let list = if s == b { insts } else { &self.f.blocks[s].insts[..] };
            scan(list, if s == b { Some(0) } else { None }).unwrap_or(true)
        })
    }

    fn no_scc_def_between(&self, insts: &[MInst], from: usize, to: usize) -> bool {
        !insts[from + 1..to]
            .iter()
            .enumerate()
            .any(|(k, inst)| !self.dead[from + 1 + k] && inst.op.flags() & F_DEF_SCC != 0)
    }

    fn count(&mut self, opnds: &[Opnd], delta: i32) {
        for v in vrefs(opnds) {
            for u in unit_range(&self.base, v) {
                self.uses[u] = (self.uses[u] as i32 + delta) as u32;
            }
        }
    }

    /// Replaces the root `insts[i]` by `new`.
    fn replace(&mut self, insts: &mut [MInst], i: usize, new: MInst) {
        let salu = is_salu(new.op);
        let old = std::mem::replace(&mut insts[i], new);
        self.count(&old.uses, -1);
        self.count(&insts[i].uses.clone(), 1);
        self.forget_and_use(&old);
        for v in vrefs(&insts[i].defs.clone()) {
            for u in self.units(v) {
                self.salu_def[u] = salu;
            }
        }
        self.kill_producers(insts, &old.uses, i);
    }

    fn insert_before(&mut self, i: usize, inst: MInst) {
        self.count(&inst.uses.clone(), 1);
        self.insert.push((i, inst));
    }

    fn forget_and_use(&mut self, inst: &MInst) {
        if inst.op == Opc::VAndB32 {
            if let Some(t) = and_const_source(inst) {
                for u in self.units(t) {
                    self.and_uses[u] = self.and_uses[u].saturating_sub(1);
                }
            }
        }
    }

    /// Kills the producers of `uses` (read before `at`) left without uses.
    fn kill_producers(&mut self, insts: &mut [MInst], uses: &[Opnd], at: usize) {
        let mut work = std::mem::take(&mut self.work);
        self.push_producers(insts, uses, at, &mut work);
        self.kill(insts, &mut work);
        self.work = work;
    }

    /// Pushes the producers, in the current block before `before`, of the
    /// dwords of `uses`.
    fn push_producers(&self, insts: &[MInst], uses: &[Opnd], before: usize, work: &mut Vec<usize>) {
        for v in vrefs(uses) {
            work.extend((0..v.n).filter_map(|k| self.producer(insts, &Opnd::R(dword(v, k)), before)));
        }
    }

    /// Deletes the instructions of `work` and, transitively, the producers
    /// they leave without uses, when they compute only registers nothing
    /// reads.
    fn kill(&mut self, insts: &[MInst], work: &mut Vec<usize>) {
        while let Some(p) = work.pop() {
            if self.dead[p] || !pure(&insts[p]) {
                continue;
            }
            let unused = vrefs(&insts[p].defs).all(|v| self.units(v).all(|u| self.uses[u] == 0));
            if !unused || insts[p].op.flags() & F_DEF_SCC != 0 {
                continue;
            }
            self.dead[p] = true;
            self.count(&insts[p].uses, -1);
            self.forget_and_use(&insts[p]);
            self.push_producers(insts, &insts[p].uses, p, work);
        }
    }

    /// Deletes `insts[p]`, whose definitions the root now makes itself.
    fn drop_replaced(&mut self, insts: &mut [MInst], p: usize) {
        self.dead[p] = true;
        let uses = insts[p].uses;
        self.count(&uses, -1);
        self.kill_producers(insts, &uses, p);
    }

    /// VOP3 source legality: at most one literal, and at most two scalar
    /// values (SGPRs and the literal) in all.
    fn vop3_fits(&self, srcs: &[Opnd], float: bool) -> bool {
        let literal = |o: &Opnd| matches!(*o, Opnd::Imm(v) if !(is_inline_int(v) || (float && is_inline_f32(v))));
        // A value read twice is read once.
        let first = |k: usize| !srcs[..k].contains(&srcs[k]);
        let literals = (0..srcs.len()).filter(|&k| literal(&srcs[k]) && first(k)).count();
        let scalars = (0..srcs.len()).filter(|&k| (literal(&srcs[k]) || self.is_sgpr(&srcs[k])) && first(k)).count();
        literals <= 1 && scalars <= 2
    }

    fn block(&mut self, b: usize, succs: &[Vec<usize>]) {
        self.stamp = b as u32 + 1;
        let mut insts = std::mem::take(&mut self.f.blocks[b].insts);
        self.dead.clear();
        self.dead.resize(insts.len(), false);
        self.insert.clear();
        for i in 0..insts.len() {
            if !self.dead[i] {
                self.root(&mut insts, i, b, succs);
            }
            for v in vrefs(&insts[i].defs) {
                for u in unit_range(&self.base, v) {
                    self.last[u] = (self.stamp, i as u32);
                }
            }
        }
        if self.insert.is_empty() {
            let mut k = 0;
            insts.retain(|_| {
                k += 1;
                !self.dead[k - 1]
            });
        } else {
            let mut inserts = std::mem::take(&mut self.insert);
            inserts.sort_by_key(|(i, _)| *i);
            let mut out = Vec::with_capacity(insts.len() + inserts.len());
            let mut next = inserts.into_iter().peekable();
            for (i, inst) in insts.into_iter().enumerate() {
                while next.peek().is_some_and(|(at, _)| *at == i) {
                    out.push(next.next().expect("peeked").1);
                }
                if !self.dead[i] {
                    out.push(inst);
                }
            }
            insts = out;
        }
        self.f.blocks[b].insts = insts;
    }

    fn root(&mut self, insts: &mut [MInst], i: usize, b: usize, succs: &[Vec<usize>]) {
        // The families (indices into FAMILIES) a root of this opcode can
        // belong to, in order.
        let tried: &[usize] = match insts[i].op {
            Opc::VAddCoCiU32 => &[0],
            Opc::SAndB32 => &[1, 2],
            Opc::SOrB32 => &[2],
            Opc::VPermB32 => &[3, 4],
            Opc::VMaxF32 | Opc::VMinF32 => &[5],
            Opc::VAndB32 => &[6, 7],
            Opc::VLshlrevB32 => &[8],
            Opc::SLshlB64 => &[9],
            Opc::VCndmaskB32 => &[10],
            _ => return,
        };
        if insts[i].mods != 0 || !virtual_only(&insts[i]) {
            return;
        }
        for &k in tried {
            if !self.enabled[k] {
                continue;
            }
            let fired = match k {
                0 => self.addr(insts, i),
                1 => self.bounds(insts, i, b, succs),
                2 => self.cselect(insts, i, b, succs),
                3 => self.perm(insts, i),
                4 => self.pack(insts, i),
                5 => self.minmax(insts, i),
                6 => self.bfe(insts, i),
                7 => self.and(insts, i),
                8 => self.shl(insts, i),
                9 => self.sshl(insts, i),
                _ => self.cndmask(insts, i, b, succs),
            };
            if fired {
                self.fired[k] += 1;
                return;
            }
        }
    }

    /// The value of dword `k` of `v`: the source of the copy that defines
    /// it in this block, else that dword itself; and the copy's index.
    fn dword_value(&self, insts: &[MInst], v: VRef, k: u8, before: usize) -> (Opnd, Option<usize>) {
        let d = dword(v, k);
        if let Some(p) = self.producer(insts, &Opnd::R(d), before) {
            let inst = &insts[p];
            if inst.op == Opc::Copy && inst.defs[..] == [Opnd::R(d)] && inst.uses[0].dwords() == 1 {
                if let Opnd::R(_) = inst.uses[0] {
                    return (inst.uses[0], Some(p));
                }
            }
        }
        (Opnd::R(d), None)
    }

    /// `base64 + (sext(x) << k)` as `v_mad_i64_i32 x, 1<<k, base64`, and
    /// `base64 + (zext(x) << k)` (the shift as two 32-bit halves) as
    /// `v_mad_u64_u32` (rules addr_sext_shl, addr_zext_shl and their
    /// swapped forms; k in 1..=6 keeps the multiplier inline).
    fn addr(&mut self, insts: &mut [MInst], i: usize) -> bool {
        let root = &insts[i];
        if root.op != Opc::VAddCoCiU32 || root.uses.len() != 3 {
            return false;
        }
        let (&[Opnd::R(hi), carry_out], &[u0, u1, carry]) = (&root.defs[..], &root.uses[..]) else {
            return false;
        };
        if carry_out != NULL || !self.single_use(&carry) {
            return false;
        }
        let Some(p) = self.producer(insts, &carry, i) else { return false };
        let add = &insts[p];
        let (Opnd::R(lo), [a0, a1]) = (add.defs[0], &add.uses[..]) else {
            return false;
        };
        if add.op != Opc::VAddCoU32 || add.defs[1] != carry || lo.reg != hi.reg || hi.off != lo.off + 1 {
            return false;
        }
        if lo.n != 1 || hi.n != 1 || self.class(lo) != Class::V {
            return false;
        }
        let (a0, a1) = (*a0, *a1);
        for (off_lo, off_hi, base_lo, base_hi) in [(a1, u1, a0, u0), (a0, u0, a1, u1)] {
            let (Opnd::R(bl), Opnd::R(bh)) = (base_lo, base_hi) else { continue };
            if bl.reg != bh.reg || bh.off != bl.off + 1 || bl.n != 1 || bh.n != 1 {
                continue;
            }
            if self.class(bl) == Class::S && (bl.off % 2 != 0 || self.f.vregs[bl.reg as usize].align < 2) {
                continue;
            }
            let base = Opnd::R(VRef { reg: bl.reg, off: bl.off, n: 2 });
            if !self.single_use(&off_lo) || !self.single_use(&off_hi) {
                continue;
            }
            let Some(m) = self.offset_source(insts, off_lo, off_hi, i) else { continue };
            let (op, x, k, first) = m;
            if !self.stable(insts, &x, first, i) || !self.stable(insts, &base_lo, p, i) || !Self::exec_stable(insts, first, i) {
                continue;
            }
            // `lo` is now written at the root: nothing may read it before.
            let lo_unit = self.units(lo).start;
            if insts[p + 1..i]
                .iter()
                .any(|k| vrefs(&k.uses).any(|v| self.units(v).contains(&lo_unit)))
            {
                continue;
            }
            let dst = Opnd::R(VRef { reg: lo.reg, off: lo.off, n: 2 });
            let mad = MInst::new(op, [dst, NULL], [x, Opnd::Imm(1 << k), base]);
            self.replace(insts, i, mad);
            self.drop_replaced(insts, p);
            return true;
        }
        false
    }

    /// The shifted 64-bit offset `(off_lo, off_hi)`: `(sext(x) << k)` from
    /// `v_lshlrev_b64 k, [x, v_ashrrev_i32 31, x]`, or `(zext(x) << k)` from
    /// `v_lshlrev_b32 k, x` and `v_lshrrev_b32 32-k, x`. Returns the mad
    /// opcode, x, k and the first instruction reading x.
    fn offset_source(&self, insts: &[MInst], off_lo: Opnd, off_hi: Opnd, i: usize) -> Option<(Opc, Opnd, u32, usize)> {
        let (Opnd::R(l), Opnd::R(h)) = (off_lo, off_hi) else { return None };
        let ql = self.producer(insts, &off_lo, i)?;
        let qh = self.producer(insts, &off_hi, i)?;
        if ql == qh {
            let shift = &insts[ql];
            let (Opnd::R(s), &[Opnd::Imm(k), Opnd::R(pair)]) = (shift.defs[0], &shift.uses[..]) else {
                return None;
            };
            if shift.op != Opc::VLshlrevB64 || !(1..=6).contains(&k) || pair.n != 2 {
                return None;
            }
            if s.reg != l.reg || l.off != s.off || h.off != s.off + 1 {
                return None;
            }
            let (x, cx) = self.dword_value(insts, pair, 0, ql);
            let (sign, cs) = self.dword_value(insts, pair, 1, ql);
            let a = self.producer(insts, &sign, ql)?;
            let ashr = &insts[a];
            if ashr.op != Opc::VAshrrevI32 || ashr.uses[..] != [Opnd::Imm(31), x] || x.dwords() != 1 {
                return None;
            }
            let first = [Some(a), cx, cs, Some(ql)].into_iter().flatten().min()?;
            return Some((Opc::VMadI64I32, x, k, first));
        }
        let (shl, shr) = (&insts[ql], &insts[qh]);
        let (&[Opnd::Imm(k), x], &[Opnd::Imm(k2), x2]) = (&shl.uses[..], &shr.uses[..]) else {
            return None;
        };
        if shl.op != Opc::VLshlrevB32 || shr.op != Opc::VLshrrevB32 || x != x2 || !(1..=6).contains(&k) || k2 != 32 - k {
            return None;
        }
        if !matches!(x, Opnd::R(_)) {
            return None;
        }
        Some((Opc::VMadU64U32, x, k, ql.min(qh)))
    }

    /// `(x >= 0) && (x < n)` from two VALU compares and a SALU and, with n
    /// uniform: `x <u smax(n, 0)` (rules bounds_*).
    fn bounds(&mut self, insts: &mut [MInst], i: usize, b: usize, succs: &[Vec<usize>]) -> bool {
        let root = &insts[i];
        let (Opc::SAndB32, [m], [r1, r2]) = (root.op, &root.defs[..], &root.uses[..]) else {
            return false;
        };
        let (m, r1, r2) = (*m, *r1, *r2);
        for (lower, upper) in [(r1, r2), (r2, r1)] {
            if !self.single_use(&lower) || !self.single_use(&upper) {
                continue;
            }
            let (Some(p1), Some(p2)) = (self.producer(insts, &lower, i), self.producer(insts, &upper, i)) else {
                continue;
            };
            let (c1, c2) = (&insts[p1], &insts[p2]);
            let x = match (c1.op, &c1.uses[..]) {
                (Opc::VCmp(0x46), &[x, z]) if is_imm(&z, 0) => x,
                (Opc::VCmp(0x43), &[z, x]) if is_imm(&z, 0) => x,
                (Opc::VCmp(0x44), &[x, z]) if is_imm(&z, u32::MAX) => x,
                _ => continue,
            };
            let n = match (c2.op, &c2.uses[..]) {
                (Opc::VCmp(0x41), &[y, n]) if y == x => n,
                (Opc::VCmp(0x44), &[n, y]) if y == x => n,
                _ => continue,
            };
            let uniform = matches!(n, Opnd::Imm(_)) || self.is_sgpr(&n);
            if !uniform || matches!(x, Opnd::Imm(_)) || n == x {
                continue;
            }
            let first = p1.min(p2);
            if !self.stable(insts, &x, first, i) || !self.stable(insts, &n, p2, i) || !Self::exec_stable(insts, first, i) {
                continue;
            }
            if !self.scc_dead_after(insts, i, b, succs) {
                continue;
            }
            // Two scalar sources at most (x and the limit): always encodable.
            let limit = match n {
                Opnd::Imm(c) => Opnd::Imm((c as i32).max(0) as u32),
                Opnd::R(nv) => Opnd::R(self.limit(insts, nv, i, b, succs)),
                Opnd::P(..) => continue,
            };
            self.replace(insts, i, MInst::new(Opc::VCmp(0x49), [m], [x, limit]));
            return true;
        }
        false
    }

    /// The register holding `smax(n, 0)` for a bounds rewrite at `i`: one
    /// `s_max_i32` per bound, placed after n's definition at the first point
    /// SCC is dead there (outside the loop when n is loop-invariant), shared
    /// by every bounds rewrite of n; else just before the root.
    fn limit(&mut self, insts: &mut [MInst], n: VRef, i: usize, b: usize, succs: &[Vec<usize>]) -> VRef {
        if let Some(&(_, t, _)) =
            self.limits.iter().find(|(m, _, scope)| *m == n && scope.is_none_or(|s| s == self.stamp))
        {
            return t;
        }
        let smax = |t: VRef| MInst::new(Opc::SMaxI32, [Opnd::R(t)], [Opnd::R(n), Opnd::Imm(0)]);
        let u = self.units(n).start;
        let single = n.n == 1 && self.defs[u] == 1 && self.def_block[u] != NONE;
        let bd = self.def_block[u] as usize;
        if single && bd < b {
            // n's block is final (blocks are done in order): place the max
            // there, as late as SCC allows before the block's branches, so
            // it waits for nothing a scalar load of n would make it wait for.
            let list = &self.f.blocks[bd].insts;
            let units = self.units(n);
            let def = list.iter().position(|k| vrefs(&k.defs).any(|d| self.units(d).contains(&units.start)));
            let end = list.iter().rposition(|k| !k.is_branch() && k.op != Opc::SEndpgm).map_or(0, |e| e + 1);
            let at = def.and_then(|p| (p + 1..=end).rev().find(|&q| self.scc_dead_at(list, q, bd, succs, insts, b)));
            if let Some(q) = at {
                let t = self.new_sreg();
                let inst = smax(t);
                self.count(&inst.uses.clone(), 1);
                self.f.blocks[bd].insts.insert(q, inst);
                self.limits.push((n, t, None));
                return t;
            }
        }
        // Just before the root: shared by later rewrites of the block, and
        // of every block when n is defined in this one (the root then
        // follows n's definition, which every use of n follows).
        let here = single && bd == b;
        let t = self.new_sreg();
        self.insert_before(i, smax(t));
        self.limits.push((n, t, if here { None } else { Some(self.stamp) }));
        t
    }

    /// SCC is dead just before `list[q]` of block `bl` (another block than
    /// the current one, `cur`): the next SCC access is a write.
    fn scc_dead_at(&self, list: &[MInst], q: usize, bl: usize, succs: &[Vec<usize>], cur_insts: &[MInst], cur: usize) -> bool {
        let scan = |l: &[MInst]| -> Option<bool> {
            for inst in l {
                let flags = inst.op.flags();
                if flags & F_USE_SCC != 0 {
                    return Some(false);
                }
                if flags & F_DEF_SCC != 0 {
                    return Some(true);
                }
            }
            None
        };
        if let Some(dead) = scan(&list[q..]) {
            return dead;
        }
        succs[bl].iter().all(|&s| {
            let l = if s == cur { cur_insts } else if s == bl { list } else { &self.f.blocks[s].insts[..] };
            scan(l).unwrap_or(true)
        })
    }

    /// A uniform i1 materialized as an all-lanes mask and then masked:
    /// `s_cselect_b32 s, 0` / `s_cselect_b32 -1, s` (rules cselect_*).
    fn cselect(&mut self, insts: &mut [MInst], i: usize, b: usize, succs: &[Vec<usize>]) -> bool {
        let root = &insts[i];
        let is_and = root.op == Opc::SAndB32;
        if !(is_and || root.op == Opc::SOrB32) || root.uses.len() != 2 {
            return false;
        }
        let (r, uses) = (root.defs[0], [root.uses[0], root.uses[1]]);
        for slot in 0..2 {
            let (t, s) = (uses[slot], uses[1 - slot]);
            if !self.single_use(&t) || t == s {
                continue;
            }
            let Some(q) = self.producer(insts, &t, i) else { continue };
            let sel = &insts[q];
            if sel.op != Opc::SCselectB32 || sel.uses[..] != [Opnd::Imm(u32::MAX), Opnd::Imm(0)] {
                continue;
            }
            if !self.no_scc_def_between(insts, q, i) || !self.scc_dead_after(insts, i, b, succs) {
                continue;
            }
            let srcs = if is_and { [s, Opnd::Imm(0)] } else { [Opnd::Imm(u32::MAX), s] };
            self.replace(insts, i, MInst::new(Opc::SCselectB32, [r], srcs));
            return true;
        }
        false
    }

    /// A `v_perm_b32` of a `v_perm_b32` as one, by the composition table
    /// (rules perm_perm_*).
    fn perm(&mut self, insts: &mut [MInst], i: usize) -> bool {
        let root = &insts[i];
        let (Opc::VPermB32, [r], [o0, o1, Opnd::Imm(s2)]) = (root.op, &root.defs[..], &root.uses[..]) else {
            return false;
        };
        let (r, o0, o1, s2) = (*r, *o0, *o1, *s2);
        for slot in 0..2 {
            let (t, c) = if slot == 0 { (o0, o1) } else { (o1, o0) };
            if !self.single_use(&t) {
                continue;
            }
            let Some(q) = self.producer(insts, &t, i) else { continue };
            let inner = &insts[q];
            let (Opc::VPermB32, [a, bb, Opnd::Imm(s1)]) = (inner.op, &inner.uses[..]) else { continue };
            let (a, bb, s1) = (*a, *bb, *s1);
            let Some(&(_, _, _, s3, [hi, lo])) =
                PERM_PERM.iter().find(|e| e.0 == s1 && e.1 == s2 && e.2 == (slot == 1))
            else {
                continue;
            };
            let srcs = [a, bb, c];
            let new = [srcs[hi as usize], srcs[lo as usize], Opnd::Imm(s3)];
            if !self.stable(insts, &a, q, i) || !self.stable(insts, &bb, q, i) || !Self::exec_stable(insts, q, i) {
                continue;
            }
            if !self.vop3_fits(&new, false) {
                continue;
            }
            self.replace(insts, i, MInst::new(Opc::VPermB32, [r], new));
            return true;
        }
        false
    }

    /// Two f16 conversions packed by `v_perm_b32 0x05040100` as one
    /// `PACK_F16` (rule cvt_cvt_pack).
    fn pack(&mut self, insts: &mut [MInst], i: usize) -> bool {
        let root = &insts[i];
        let (Opc::VPermB32, [r], [hi, lo, sel]) = (root.op, &root.defs[..], &root.uses[..]) else {
            return false;
        };
        let (r, hi, lo) = (*r, *hi, *lo);
        if !is_imm(sel, 0x0504_0100) || hi == lo || !self.single_use(&hi) || !self.single_use(&lo) {
            return false;
        }
        let (Some(qh), Some(ql)) = (self.producer(insts, &hi, i), self.producer(insts, &lo, i)) else {
            return false;
        };
        let (ch, cl) = (&insts[qh], &insts[ql]);
        if ch.op != Opc::VCvtF16F32 || cl.op != Opc::VCvtF16F32 {
            return false;
        }
        let (a, bsrc) = (cl.uses[0], ch.uses[0]);
        if !self.stable(insts, &a, ql, i) || !self.stable(insts, &bsrc, qh, i) || !Self::exec_stable(insts, ql.min(qh), i) {
            return false;
        }
        self.replace(insts, i, MInst::new(Opc::PackF16, [r], [a, bsrc]));
        true
    }

    /// `outer(inner(a, b), c)` of f32 min/max as one three-source
    /// instruction; `outer(c, inner(a, b))` too when no source can be a
    /// signaling NaN (rules fmaxmax_*, fminmin_*).
    fn minmax(&mut self, insts: &mut [MInst], i: usize) -> bool {
        let root = &insts[i];
        let outer = root.op;
        if !matches!(outer, Opc::VMaxF32 | Opc::VMinF32) || root.uses.len() != 2 {
            return false;
        }
        let (r, o0, o1) = (root.defs[0], root.uses[0], root.uses[1]);
        let three = if outer == Opc::VMaxF32 { Opc::VMax3F32 } else { Opc::VMin3F32 };
        for slot in 0..2 {
            let (t, c) = if slot == 0 { (o0, o1) } else { (o1, o0) };
            if !self.single_use(&t) || t == c {
                continue;
            }
            let Some(q) = self.producer(insts, &t, i) else { continue };
            let inner = &insts[q];
            if inner.op != outer || inner.uses.len() != 2 {
                continue;
            }
            let (a, bb) = (inner.uses[0], inner.uses[1]);
            let srcs = if slot == 0 {
                [a, bb, c]
            } else {
                self.ensure_quiet(insts);
                if !(self.is_quiet(&a) && self.is_quiet(&bb) && self.is_quiet(&c)) {
                    continue;
                }
                [c, a, bb]
            };
            if !self.stable(insts, &a, q, i) || !self.stable(insts, &bb, q, i) || !Self::exec_stable(insts, q, i) {
                continue;
            }
            if !self.vop3_fits(&srcs, true) {
                continue;
            }
            self.replace(insts, i, MInst::new(three, [r], srcs));
            return true;
        }
        false
    }

    /// `(x >> k) & (2^w - 1)` as `v_bfe_u32 x, k, w`, or as the shift alone
    /// when the mask keeps every bit the shift leaves (rules shr_and_*).
    fn bfe(&mut self, insts: &mut [MInst], i: usize) -> bool {
        let root = &insts[i];
        if root.op != Opc::VAndB32 || root.uses.len() != 2 {
            return false;
        }
        let r = root.defs[0];
        let (mask, t) = match (root.uses[0], root.uses[1]) {
            (Opnd::Imm(m), t @ Opnd::R(_)) | (t @ Opnd::R(_), Opnd::Imm(m)) => (m, t),
            _ => return false,
        };
        if mask == 0 || mask & mask.wrapping_add(1) != 0 || !self.single_use(&t) {
            return false;
        }
        let w = mask.count_ones();
        let Some(q) = self.producer(insts, &t, i) else { return false };
        let shr = &insts[q];
        let (Opc::VLshrrevB32, [Opnd::Imm(k), x]) = (shr.op, &shr.uses[..]) else { return false };
        let (k, x) = (*k, *x);
        if !(1..=31).contains(&k) || !self.stable(insts, &x, q, i) || !Self::exec_stable(insts, q, i) {
            return false;
        }
        let new = if k + w >= 32 {
            MInst::new(Opc::VLshrrevB32, [r], [Opnd::Imm(k), x])
        } else {
            MInst::new(Opc::VBfeU32, [r], [x, Opnd::Imm(k), Opnd::Imm(w)])
        };
        self.replace(insts, i, new);
        true
    }

    /// `(x & c1) & c2` as `x & (c1 & c2)`, when every use of `x & c1` is
    /// such an and, so it dies (rule and_and).
    fn and(&mut self, insts: &mut [MInst], i: usize) -> bool {
        let root = &insts[i];
        if root.op != Opc::VAndB32 {
            return false;
        }
        let r = root.defs[0];
        let Some(t) = and_const_source(root) else { return false };
        let c2 = root.uses.iter().find_map(imm).expect("an and with a constant");
        let tu = self.units(t);
        if tu.clone().any(|u| self.and_uses[u] != self.uses[u]) {
            return false;
        }
        let Some(q) = self.producer(insts, &Opnd::R(t), i) else { return false };
        let inner = &insts[q];
        if inner.op != Opc::VAndB32 {
            return false;
        }
        let (c1, x) = match (inner.uses[0], inner.uses[1]) {
            (Opnd::Imm(c), x @ Opnd::R(_)) | (x @ Opnd::R(_), Opnd::Imm(c)) => (c, x),
            _ => return false,
        };
        if !self.stable(insts, &x, q, i) || !Self::exec_stable(insts, q, i) {
            return false;
        }
        self.replace(insts, i, MInst::new(Opc::VAndB32, [r], [Opnd::Imm(c1 & c2), x]));
        true
    }

    /// `(x << a) << b` as `x << (a + b)` for `a + b < 32` (rule shl_shl).
    fn shl(&mut self, insts: &mut [MInst], i: usize) -> bool {
        let root = &insts[i];
        let (Opc::VLshlrevB32, [r], [Opnd::Imm(bb), t]) = (root.op, &root.defs[..], &root.uses[..]) else {
            return false;
        };
        let (r, bb, t) = (*r, *bb, *t);
        if !self.single_use(&t) {
            return false;
        }
        let Some(q) = self.producer(insts, &t, i) else { return false };
        let inner = &insts[q];
        let (Opc::VLshlrevB32, [Opnd::Imm(a), x]) = (inner.op, &inner.uses[..]) else { return false };
        let (a, x) = (*a, *x);
        if a > 31 || bb > 31 || a + bb > 31 || !self.stable(insts, &x, q, i) || !Self::exec_stable(insts, q, i) {
            return false;
        }
        self.replace(insts, i, MInst::new(Opc::VLshlrevB32, [r], [Opnd::Imm(a + bb), x]));
        true
    }

    /// `(x << a) << b` of 64-bit SGPR pairs as `x << (a + b)` for
    /// `a + b < 64`; the inner shift's SCC is never read (rule sshl64_sshl64).
    fn sshl(&mut self, insts: &mut [MInst], i: usize) -> bool {
        let root = &insts[i];
        let (Opc::SLshlB64, [r], [t, Opnd::Imm(bb)]) = (root.op, &root.defs[..], &root.uses[..]) else {
            return false;
        };
        let (r, t, bb) = (*r, *t, *bb);
        if !self.single_use(&t) {
            return false;
        }
        let Some(q) = self.producer(insts, &t, i) else { return false };
        let inner = &insts[q];
        let (Opc::SLshlB64, [x, Opnd::Imm(a)]) = (inner.op, &inner.uses[..]) else { return false };
        let (x, a) = (*x, *a);
        if a > 63 || bb > 63 || a + bb > 63 || !self.stable(insts, &x, q, i) {
            return false;
        }
        if insts[q + 1..i].iter().enumerate().any(|(k, inst)| !self.dead[q + 1 + k] && inst.op.flags() & F_USE_SCC != 0) {
            return false;
        }
        self.replace(insts, i, MInst::new(Opc::SLshlB64, [r], [x, Opnd::Imm(a + bb)]));
        // The inner shift's SCC definition is dead (the root writes SCC
        // before anything reads it), so it goes too.
        if !self.dead[q] && vrefs(&insts[q].defs).all(|v| self.units(v).all(|u| self.uses[u] == 0)) {
            self.drop_replaced(insts, q);
        }
        true
    }

    /// Two selects of one constant under uniform (SALU-made) masks as one
    /// select under their and (or) (rules cnd_cnd_*).
    fn cndmask(&mut self, insts: &mut [MInst], i: usize, b: usize, succs: &[Vec<usize>]) -> bool {
        let root = &insts[i];
        let (Opc::VCndmaskB32, [r], [f, tr, m2]) = (root.op, &root.defs[..], &root.uses[..]) else {
            return false;
        };
        let (r, f, tr, m2) = (*r, *f, *tr, *m2);
        let (konst, t, on_false) = match (f, tr) {
            (Opnd::Imm(k), t @ Opnd::R(_)) => (k, t, true),
            (t @ Opnd::R(_), Opnd::Imm(k)) => (k, t, false),
            _ => return false,
        };
        if !self.single_use(&t) {
            return false;
        }
        let Some(q) = self.producer(insts, &t, i) else { return false };
        let inner = &insts[q];
        if inner.op != Opc::VCndmaskB32 || inner.uses.len() != 3 {
            return false;
        }
        let (x, m1) = match (on_false, inner.uses[0], inner.uses[1]) {
            (true, Opnd::Imm(k), x) if k == konst => (x, inner.uses[2]),
            (false, x, Opnd::Imm(k)) if k == konst => (x, inner.uses[2]),
            _ => return false,
        };
        if !self.salu_mask(&m1) || !self.salu_mask(&m2) || m1 == m2 {
            return false;
        }
        if !self.stable(insts, &x, q, i) || !self.stable(insts, &m1, q, i) || !Self::exec_stable(insts, q, i) {
            return false;
        }
        // The and is inserted at the root: SCC must not be live across it.
        if insts[i].op.flags() & F_USE_SCC != 0 || !self.scc_dead_after(insts, i, b, succs) {
            return false;
        }
        let m = self.new_sreg();
        let op = if on_false { Opc::SAndB32 } else { Opc::SOrB32 };
        self.insert_before(i, MInst::new(op, [Opnd::R(m)], [m1, m2]));
        let srcs = if on_false { [Opnd::Imm(konst), x, Opnd::R(m)] } else { [x, Opnd::Imm(konst), Opnd::R(m)] };
        self.replace(insts, i, MInst::new(Opc::VCndmaskB32, [r], srcs));
        true
    }

    /// A lane mask written by a SALU instruction (a uniform condition), so
    /// combining it on the SALU costs no VALU-to-SALU round trip.
    fn salu_mask(&self, o: &Opnd) -> bool {
        let Opnd::R(v) = *o else { return false };
        v.n == 1 && self.class(v) == Class::S && self.units(v).all(|u| self.defs[u] == 1 && self.salu_def[u])
    }
}

fn is_salu(op: Opc) -> bool {
    matches!(op.fmt(), Fmt::Sop1 | Fmt::Sop2 | Fmt::Sopk)
}

fn unit_range(base: &[u32], v: VRef) -> std::ops::Range<usize> {
    let start = (base[v.reg as usize] + v.off as u32) as usize;
    start..start + v.n as usize
}

/// The register `t` of `v_and_b32 c, t` (either order, c an immediate).
fn and_const_source(inst: &MInst) -> Option<VRef> {
    if inst.op != Opc::VAndB32 || inst.mods != 0 || inst.uses.len() != 2 {
        return None;
    }
    match (inst.uses[0], inst.uses[1]) {
        (Opnd::Imm(_), Opnd::R(t)) | (Opnd::R(t), Opnd::Imm(_)) => Some(t),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    const ALL: [bool; FAMILIES.len()] = [true; FAMILIES.len()];

    struct Builder {
        f: MFunc,
    }

    impl Builder {
        fn new() -> Self {
            Builder { f: MFunc { blocks: vec![MBlock::default()], vregs: Vec::new(), ..Default::default() } }
        }
        fn reg(&mut self, class: Class, size: u8) -> VRef {
            let align = if class == Class::S && size >= 2 { 2 } else { 1 };
            self.f.vregs.push(VRegInfo { class, size, align, fixed: None });
            VRef { reg: self.f.vregs.len() as u32 - 1, off: 0, n: size }
        }
        fn v(&mut self) -> VRef {
            self.reg(Class::V, 1)
        }
        fn s(&mut self) -> VRef {
            self.reg(Class::S, 1)
        }
        fn push(&mut self, op: Opc, defs: &[Opnd], uses: &[Opnd]) {
            self.f.blocks[0].insts.push(MInst::new(op, defs, uses));
        }
    }

    fn r(v: VRef) -> Opnd {
        Opnd::R(v)
    }

    fn d(v: VRef, k: u8) -> Opnd {
        Opnd::R(dword(v, k))
    }

    fn imm(v: u32) -> Opnd {
        Opnd::Imm(v)
    }

    fn ops(f: &MFunc) -> Vec<Opc> {
        f.blocks[0].insts.iter().map(|i| i.op).collect()
    }

    /// The proven rule a test case instantiates.
    fn proven(name: &str) {
        assert!(PROVEN.iter().any(|(rule, _)| *rule == name), "{name} is not a proven rule");
    }

    /// The f16 conversion as some fixed function of the input that leaves
    /// garbage in the high half (the rules must not depend on either).
    fn cvt(x: u32) -> u32 {
        (x.wrapping_mul(0x9e37_79b9) >> 7) & 0xffff | 0xdead_0000
    }

    fn snan(x: u32) -> bool {
        is_snan(x)
    }

    fn nan(x: u32) -> bool {
        x & 0x7f80_0000 == 0x7f80_0000 && x & 0x007f_ffff != 0
    }

    fn key(x: u32) -> i64 {
        let mag = (x & 0x7fff_ffff) as i64;
        if x >> 31 != 0 {
            -mag
        } else {
            mag
        }
    }

    /// `v_max_f32` / `v_min_f32` in IEEE mode, from the ISA pseudo-code.
    fn fminmax(max: bool, a: u32, b: u32) -> u32 {
        let zero = |x: u32| x & 0x7fff_ffff == 0;
        let better = if max {
            key(a) > key(b) || (zero(a) && zero(b) && a >> 31 == 0 && b >> 31 != 0)
        } else {
            key(a) < key(b) || (zero(a) && zero(b) && a >> 31 != 0 && b >> 31 == 0)
        };
        if snan(a) {
            a | 0x40_0000
        } else if snan(b) {
            b | 0x40_0000
        } else if nan(b) {
            a
        } else if nan(a) {
            b
        } else if better {
            a
        } else {
            b
        }
    }

    fn perm(s0: u32, s1: u32, sel: u32) -> u32 {
        let data = (s0 as u64) << 32 | s1 as u64;
        let byte = |i: u32| (data >> (8 * i)) as u8 as u32;
        (0..4).fold(0, |acc, k| {
            let s = (sel >> (8 * k)) & 0xff;
            let b = match s {
                0..=7 => byte(s),
                8..=11 => (byte(2 * (s - 8) + 1) >> 7) * 0xff,
                12 => 0,
                _ => 0xff,
            };
            acc | b << (8 * k)
        })
    }

    /// One lane of the instructions the tests use: register dwords, SCC,
    /// and lane masks as bit 0.
    fn eval(f: &MFunc, init: &HashMap<(u32, u8), u32>, scc0: bool) -> HashMap<(u32, u8), u32> {
        let mut regs = init.clone();
        let mut scc = scc0;
        let rd = |regs: &HashMap<(u32, u8), u32>, o: &Opnd| -> u64 {
            match *o {
                Opnd::Imm(v) => v as u64,
                Opnd::R(v) => (0..v.n).fold(0u64, |acc, k| acc | (regs[&(v.reg, v.off + k)] as u64) << (32 * k)),
                Opnd::P(..) => panic!("physical operand"),
            }
        };
        for inst in &f.blocks[0].insts {
            let u: Vec<u64> = inst.uses.iter().map(|o| rd(&regs, o)).collect();
            let w = |k: usize| u[k] as u32;
            let mut out: Vec<u64> = Vec::new();
            match inst.op {
                Opc::Copy => out.push(u[0]),
                Opc::VAshrrevI32 => out.push(((w(1) as i32) >> (w(0) & 31)) as u32 as u64),
                Opc::VLshlrevB32 => out.push(((w(1) as u64) << (w(0) & 31)) as u32 as u64),
                Opc::VLshrrevB32 => out.push((w(1) >> (w(0) & 31)) as u64),
                Opc::VAndB32 => out.push((w(0) & w(1)) as u64),
                Opc::VBfeU32 => out.push(((w(0) >> (w(1) & 31)) & ((1u64 << (w(2) & 31)) - 1) as u32) as u64),
                Opc::VLshlrevB64 => out.push(u[1] << (w(0) & 63)),
                Opc::VAddCoU32 => {
                    let s = w(0) as u64 + w(1) as u64;
                    out.extend([s as u32 as u64, s >> 32]);
                }
                Opc::VAddCoCiU32 => out.push((w(0) as u64 + w(1) as u64 + (u[2] & 1)) as u32 as u64),
                Opc::VMadI64I32 => out.push((w(0) as i32 as i64 as u64).wrapping_mul(w(1) as i32 as i64 as u64).wrapping_add(u[2])),
                Opc::VMadU64U32 => out.push((w(0) as u64 * w(1) as u64).wrapping_add(u[2])),
                Opc::VPermB32 => out.push(perm(w(0), w(1), w(2)) as u64),
                Opc::VMaxF32 => out.push(fminmax(true, w(0), w(1)) as u64),
                Opc::VMinF32 => out.push(fminmax(false, w(0), w(1)) as u64),
                Opc::VMax3F32 => out.push(fminmax(true, fminmax(true, w(0), w(1)), w(2)) as u64),
                Opc::VMin3F32 => out.push(fminmax(false, fminmax(false, w(0), w(1)), w(2)) as u64),
                Opc::VCvtF16F32 => out.push(cvt(w(0)) as u64),
                Opc::PackF16 => out.push((cvt(w(0)) & 0xffff | cvt(w(1)) << 16) as u64),
                Opc::VCndmaskB32 => out.push(if u[2] & 1 != 0 { u[1] } else { u[0] }),
                Opc::VCmp(c) => {
                    let (a, b) = (w(0) as i32, w(1) as i32);
                    out.push(match c {
                        0x41 => a < b,
                        0x43 => a <= b,
                        0x44 => a > b,
                        0x46 => a >= b,
                        0x49 => w(0) < w(1),
                        _ => panic!("compare {c:#x}"),
                    } as u64);
                }
                Opc::SAndB32 => {
                    let v = w(0) & w(1);
                    scc = v != 0;
                    out.push(v as u64);
                }
                Opc::SOrB32 => {
                    let v = w(0) | w(1);
                    scc = v != 0;
                    out.push(v as u64);
                }
                Opc::SMaxI32 => {
                    scc = w(0) as i32 >= w(1) as i32;
                    out.push((w(0) as i32).max(w(1) as i32) as u32 as u64);
                }
                Opc::SCmpLgU32 => scc = w(0) != w(1),
                Opc::SCselectB32 => out.push(if scc { u[0] } else { u[1] }),
                Opc::SLshlB64 => {
                    let v = u[0] << (w(1) & 63);
                    scc = v != 0;
                    out.push(v);
                }
                op => panic!("no model of {op:?}"),
            }
            for (o, v) in inst.defs.iter().zip(out) {
                if let Opnd::R(r) = *o {
                    for k in 0..r.n {
                        regs.insert((r.reg, r.off + k), (v >> (32 * k)) as u32);
                    }
                }
            }
        }
        regs
    }

    fn rng(seed: &mut u64) -> u32 {
        *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let x = (*seed >> 32) as u32;
        // Edge values often: small, negative, NaN and infinity patterns.
        match *seed >> 61 {
            0 => x % 70,
            1 => (x % 70).wrapping_neg(),
            2 => [0x7f80_0000, 0xff80_0000, 0x7fc0_0001, 0x7f80_0001, 0xffa0_0000, 0x8000_0000, 0x7fff_ffff][x as usize % 7],
            _ => x,
        }
    }

    /// Optimizes a copy of `f` and checks that `outs` get the same values
    /// as before for random inputs (`ins`, with SCC as given).
    fn optimized(f: &MFunc, ins: &[VRef], outs: &[VRef]) -> MFunc {
        let mut g = f.clone();
        run(&mut g, ALL);
        let mut seed = 0x5eed_0001u64;
        for _ in 0..2000 {
            let mut init = HashMap::new();
            for v in ins {
                for k in 0..v.n {
                    init.insert((v.reg, v.off + k), rng(&mut seed));
                }
            }
            let scc = seed & 1 != 0;
            let (a, b) = (eval(f, &init, scc), eval(&g, &init, scc));
            for v in outs {
                for k in 0..v.n {
                    assert_eq!(a[&(v.reg, v.off + k)], b[&(v.reg, v.off + k)], "{v:?} differs for {init:?}");
                }
            }
        }
        g
    }

    /// base64 + (sext(x) << 2): sign, pair copies, 64-bit shift, 64-bit add.
    fn sext_address(b: &mut Builder, swapped: bool) -> (VRef, VRef, VRef, VRef) {
        let x = b.v();
        let base = b.reg(Class::S, 2);
        let (sign, pair, shifted, sum, carry) = (b.v(), b.reg(Class::V, 2), b.reg(Class::V, 2), b.reg(Class::V, 2), b.s());
        b.push(Opc::VAshrrevI32, &[r(sign)], &[imm(31), r(x)]);
        b.push(Opc::Copy, &[d(pair, 0)], &[r(x)]);
        b.push(Opc::Copy, &[d(pair, 1)], &[r(sign)]);
        b.push(Opc::VLshlrevB64, &[r(shifted)], &[imm(2), r(pair)]);
        let (lo, hi) = if swapped { ([d(shifted, 0), d(base, 0)], [d(shifted, 1), d(base, 1)]) } else { ([d(base, 0), d(shifted, 0)], [d(base, 1), d(shifted, 1)]) };
        b.push(Opc::VAddCoU32, &[d(sum, 0), r(carry)], &lo);
        b.push(Opc::VAddCoCiU32, &[d(sum, 1), NULL], &[hi[0], hi[1], r(carry)]);
        (x, base, sum, shifted)
    }

    #[test]
    fn sign_extended_address_becomes_mad_i64() {
        for swapped in [false, true] {
            proven(if swapped { "addr_sext_shl_swapped" } else { "addr_sext_shl" });
            let mut b = Builder::new();
            let (x, base, sum, _) = sext_address(&mut b, swapped);
            let g = optimized(&b.f, &[x, base], &[sum]);
            assert_eq!(ops(&g), [Opc::VMadI64I32]);
            assert_eq!(g.blocks[0].insts[0].uses[..], [r(x), imm(4), r(base)]);
        }
        // A shifted offset that something else reads stays as it is.
        let mut b = Builder::new();
        let (x, base, sum, shifted) = sext_address(&mut b, false);
        let other = b.v();
        b.push(Opc::VAndB32, &[r(other)], &[imm(7), d(shifted, 0)]);
        let g = optimized(&b.f, &[x, base], &[sum, other]);
        assert!(!ops(&g).contains(&Opc::VMadI64I32));
    }

    #[test]
    fn zero_extended_address_becomes_mad_u64() {
        for swapped in [false, true] {
            proven(if swapped { "addr_zext_shl_swapped" } else { "addr_zext_shl" });
            let mut b = Builder::new();
            let x = b.v();
            let base = b.reg(Class::S, 2);
            let (lo, hi, sum, carry) = (b.v(), b.v(), b.reg(Class::V, 2), b.s());
            b.push(Opc::VLshlrevB32, &[r(lo)], &[imm(3), r(x)]);
            b.push(Opc::VLshrrevB32, &[r(hi)], &[imm(29), r(x)]);
            let (l, h) = if swapped { ([r(lo), d(base, 0)], [r(hi), d(base, 1)]) } else { ([d(base, 0), r(lo)], [d(base, 1), r(hi)]) };
            b.push(Opc::VAddCoU32, &[d(sum, 0), r(carry)], &l);
            b.push(Opc::VAddCoCiU32, &[d(sum, 1), NULL], &[h[0], h[1], r(carry)]);
            let g = optimized(&b.f, &[x, base], &[sum]);
            assert_eq!(ops(&g), [Opc::VMadU64U32]);
            assert_eq!(g.blocks[0].insts[0].uses[..], [r(x), imm(8), r(base)]);
        }
        // The halves of a different shift do not match.
        let mut b = Builder::new();
        let x = b.v();
        let base = b.reg(Class::S, 2);
        let (lo, hi, sum, carry) = (b.v(), b.v(), b.reg(Class::V, 2), b.s());
        b.push(Opc::VLshlrevB32, &[r(lo)], &[imm(3), r(x)]);
        b.push(Opc::VLshrrevB32, &[r(hi)], &[imm(28), r(x)]);
        b.push(Opc::VAddCoU32, &[d(sum, 0), r(carry)], &[d(base, 0), r(lo)]);
        b.push(Opc::VAddCoCiU32, &[d(sum, 1), NULL], &[d(base, 1), r(hi), r(carry)]);
        assert!(!ops(&optimized(&b.f, &[x, base], &[sum])).contains(&Opc::VMadU64U32));
    }

    fn bounds_check(b: &mut Builder, n: Opnd) -> (VRef, VRef) {
        let x = b.v();
        let (m1, m2, m) = (b.s(), b.s(), b.s());
        b.push(Opc::VCmp(0x46), &[r(m1)], &[r(x), imm(0)]);
        b.push(Opc::VCmp(0x41), &[r(m2)], &[r(x), n]);
        b.push(Opc::SAndB32, &[r(m)], &[r(m1), r(m2)]);
        (x, m)
    }

    #[test]
    fn bounds_check_becomes_one_unsigned_compare() {
        // Every form of x >= 0 (code, whether the constant comes first, the
        // constant) with every form of x < n, in either order in the and.
        let lower: [(&str, u8, bool, u32); 3] = [("ge", 0x46, false, 0), ("le", 0x43, true, 0), ("gt", 0x44, false, u32::MAX)];
        let upper: [(&str, u8, bool); 2] = [("lt", 0x41, false), ("gt", 0x44, true)];
        for (lo, c1, swap1, z) in lower {
            for (up, c2, swap2) in upper {
                for first in [true, false] {
                    proven(&format!("bounds_{lo}_{up}_{}", if first { "m1" } else { "m2" }));
                    let mut b = Builder::new();
                    let (n, x) = (b.s(), b.v());
                    let (m1, m2, m) = (b.s(), b.s(), b.s());
                    b.push(Opc::VCmp(c1), &[r(m1)], &if swap1 { [imm(z), r(x)] } else { [r(x), imm(z)] });
                    b.push(Opc::VCmp(c2), &[r(m2)], &if swap2 { [r(n), r(x)] } else { [r(x), r(n)] });
                    b.push(Opc::SAndB32, &[r(m)], &if first { [r(m1), r(m2)] } else { [r(m2), r(m1)] });
                    let g = optimized(&b.f, &[n, x], &[m]);
                    assert_eq!(ops(&g), [Opc::SMaxI32, Opc::VCmp(0x49)]);
                }
            }
        }
        // A constant bound folds; a negative one becomes 0.
        for c in [100u32, (-5i32) as u32] {
            let mut b = Builder::new();
            let (x, m) = bounds_check(&mut b, imm(c));
            let g = optimized(&b.f, &[x], &[m]);
            assert_eq!(ops(&g), [Opc::VCmp(0x49)]);
        }
        // SCC read after the and: it stays.
        let mut b = Builder::new();
        let n = b.s();
        let (x, m) = bounds_check(&mut b, r(n));
        let t = b.s();
        b.push(Opc::SCselectB32, &[r(t)], &[imm(1), imm(2)]);
        let g = optimized(&b.f, &[n, x], &[m, t]);
        assert!(ops(&g).contains(&Opc::SAndB32));
        // A bound defined in an earlier block (a loop-invariant one): one max,
        // in that block before its compare and branch, for both checks.
        let mut b = Builder::new();
        let (p, n, k) = (b.s(), b.s(), b.s());
        b.push(Opc::SAddI32, &[r(n)], &[r(p), imm(3)]);
        b.push(Opc::SCmpLgU32, &[], &[r(k), imm(0)]);
        b.push(Opc::SCbranchScc0, &[], &[]);
        b.f.blocks[0].insts.last_mut().expect("branch").imm = 1;
        b.f.blocks.push(MBlock { insts: Vec::new(), loop_depth: 1 });
        let (x, y, m1, m2, m3, m4, o1, o2) = (b.v(), b.v(), b.s(), b.s(), b.s(), b.s(), b.s(), b.s());
        let body = [
            MInst::new(Opc::VCmp(0x46), [r(m1)], [r(x), imm(0)]),
            MInst::new(Opc::VCmp(0x41), [r(m2)], [r(x), r(n)]),
            MInst::new(Opc::SAndB32, [r(o1)], [r(m1), r(m2)]),
            MInst::new(Opc::VCmp(0x46), [r(m3)], [r(y), imm(0)]),
            MInst::new(Opc::VCmp(0x41), [r(m4)], [r(y), r(n)]),
            MInst::new(Opc::SAndB32, [r(o2)], [r(m3), r(m4)]),
        ];
        b.f.blocks[1].insts.extend(body);
        let mut g = b.f.clone();
        run(&mut g, ALL);
        let head: Vec<Opc> = g.blocks[0].insts.iter().map(|i| i.op).collect();
        assert_eq!(head, [Opc::SAddI32, Opc::SMaxI32, Opc::SCmpLgU32, Opc::SCbranchScc0]);
        assert_eq!(ops(&MFunc { blocks: vec![g.blocks[1].clone()], vregs: g.vregs.clone(), ..Default::default() }), [Opc::VCmp(0x49), Opc::VCmp(0x49)]);
    }

    #[test]
    fn masked_uniform_condition_is_one_select() {
        for (op, slot) in [(Opc::SAndB32, 0), (Opc::SAndB32, 1), (Opc::SOrB32, 0), (Opc::SOrB32, 1)] {
            let kind = if op == Opc::SAndB32 { "and" } else { "or" };
            proven(&format!("cselect_{kind}_{}", if slot == 0 { "t" } else { "s" }));
            let mut b = Builder::new();
            let (a, s, t, out) = (b.s(), b.s(), b.s(), b.s());
            b.push(Opc::SCmpLgU32, &[], &[r(a), imm(0)]);
            b.push(Opc::SCselectB32, &[r(t)], &[imm(u32::MAX), imm(0)]);
            let uses = if slot == 0 { [r(t), r(s)] } else { [r(s), r(t)] };
            b.push(op, &[r(out)], &uses);
            let g = optimized(&b.f, &[a, s], &[out]);
            assert_eq!(ops(&g), [Opc::SCmpLgU32, Opc::SCselectB32]);
        }
        // An SCC write between the select and the and.
        let mut b = Builder::new();
        let (a, s, t, out, junk) = (b.s(), b.s(), b.s(), b.s(), b.s());
        b.push(Opc::SCmpLgU32, &[], &[r(a), imm(0)]);
        b.push(Opc::SCselectB32, &[r(t)], &[imm(u32::MAX), imm(0)]);
        b.push(Opc::SMaxI32, &[r(junk)], &[r(a), imm(3)]);
        b.push(Opc::SAndB32, &[r(out)], &[r(t), r(s)]);
        assert_eq!(ops(&optimized(&b.f, &[a, s], &[out, junk])).len(), 4);
    }

    #[test]
    fn perm_of_perm_is_one_perm() {
        for &(s1, s2, second, _, _) in PERM_PERM {
            proven(&format!("perm_perm_{s1:08x}_{s2:08x}_{}", u8::from(second)));
            let mut b = Builder::new();
            let (a, c, e, t, out) = (b.v(), b.v(), b.v(), b.v(), b.v());
            b.push(Opc::VPermB32, &[r(t)], &[r(a), r(c), imm(s1)]);
            let uses = if second { [r(e), r(t), imm(s2)] } else { [r(t), r(e), imm(s2)] };
            b.push(Opc::VPermB32, &[r(out)], &uses);
            let g = optimized(&b.f, &[a, c, e], &[out]);
            assert_eq!(ops(&g), [Opc::VPermB32]);
        }
    }

    #[test]
    fn f16_conversion_pair_packs() {
        proven("cvt_cvt_pack");
        let mut b = Builder::new();
        let (a, c, t0, t1, out) = (b.v(), b.v(), b.v(), b.v(), b.v());
        b.push(Opc::VCvtF16F32, &[r(t0)], &[r(a)]);
        b.push(Opc::VCvtF16F32, &[r(t1)], &[r(c)]);
        b.push(Opc::VPermB32, &[r(out)], &[r(t1), r(t0), imm(0x0504_0100)]);
        let g = optimized(&b.f, &[a, c], &[out]);
        assert_eq!(ops(&g), [Opc::PackF16]);
        assert_eq!(g.blocks[0].insts[0].uses[..], [r(a), r(c)]);
    }

    #[test]
    fn f32_max_chain_is_max3_with_nan_payloads_kept() {
        // outer(inner(a, b), c): exact whatever the sources.
        let kinds = [("fmaxmax", Opc::VMaxF32, Opc::VMax3F32, Opc::VMinF32), ("fminmin", Opc::VMinF32, Opc::VMin3F32, Opc::VMaxF32)];
        for (name, op, three, other) in kinds {
            proven(&format!("{name}_left"));
            let mut b = Builder::new();
            let (a, c, e, t, out) = (b.v(), b.v(), b.v(), b.v(), b.v());
            b.push(op, &[r(t)], &[r(a), r(c)]);
            b.push(op, &[r(out)], &[r(t), r(e)]);
            assert_eq!(ops(&optimized(&b.f, &[a, c, e], &[out])), [three]);
            // outer(c, inner(a, b)) needs sources that cannot be signaling NaNs.
            let mut b = Builder::new();
            let (a, c, e, t, out) = (b.v(), b.v(), b.v(), b.v(), b.v());
            b.push(op, &[r(t)], &[r(a), r(c)]);
            b.push(op, &[r(out)], &[r(e), r(t)]);
            assert_eq!(ops(&optimized(&b.f, &[a, c, e], &[out])).len(), 2);
            // Quieted sources (the other of min and max of a value with itself
            // never is one, and does not fold into the chain).
            proven(&format!("{name}_right_quiet"));
            let mut b = Builder::new();
            let (a0, c0, e0) = (b.v(), b.v(), b.v());
            let (a, c, e, t, out) = (b.v(), b.v(), b.v(), b.v(), b.v());
            for (q, x) in [(a, a0), (c, c0), (e, e0)] {
                b.push(other, &[r(q)], &[r(x), r(x)]);
            }
            b.push(op, &[r(t)], &[r(a), r(c)]);
            b.push(op, &[r(out)], &[r(e), r(t)]);
            let g = optimized(&b.f, &[a0, c0, e0], &[out]);
            assert_eq!(g.blocks[0].insts.last().map(|i| (i.op, i.uses[..].to_vec())), Some((three, vec![r(e), r(a), r(c)])));
        }
    }

    #[test]
    fn shift_and_mask_is_bit_field_extract() {
        for (k, mask, want) in [(16, 1, Opc::VBfeU32), (8, 0xff, Opc::VBfeU32), (24, 0xff, Opc::VLshrrevB32), (4, u32::MAX, Opc::VLshrrevB32)] {
            for rev in [false, true] {
                let rule = if want == Opc::VBfeU32 { "shr_and_bfe" } else { "shr_and_redundant" };
                proven(&format!("{rule}{}", if rev { "_rev" } else { "" }));
                let mut b = Builder::new();
                let (x, t, out) = (b.v(), b.v(), b.v());
                b.push(Opc::VLshrrevB32, &[r(t)], &[imm(k), r(x)]);
                b.push(Opc::VAndB32, &[r(out)], &if rev { [r(t), imm(mask)] } else { [imm(mask), r(t)] });
                assert_eq!(ops(&optimized(&b.f, &[x], &[out])), [want]);
            }
        }
        // A mask that is not a low run of ones.
        let mut b = Builder::new();
        let (x, t, out) = (b.v(), b.v(), b.v());
        b.push(Opc::VLshrrevB32, &[r(t)], &[imm(8), r(x)]);
        b.push(Opc::VAndB32, &[r(out)], &[imm(0xf0), r(t)]);
        assert_eq!(ops(&optimized(&b.f, &[x], &[out])).len(), 2);
    }

    #[test]
    fn nested_masks_fold() {
        for (xc, tc) in [(false, false), (true, false), (false, true), (true, true)] {
            proven(&format!("and_and{}{}", if xc { "_xc" } else { "" }, if tc { "_tc" } else { "" }));
            let mut b = Builder::new();
            let (x, t, o1, o2) = (b.v(), b.v(), b.v(), b.v());
            b.push(Opc::VAndB32, &[r(t)], &if xc { [r(x), imm(0xff)] } else { [imm(0xff), r(x)] });
            b.push(Opc::VAndB32, &[r(o1)], &if tc { [r(t), imm(0x7f)] } else { [imm(0x7f), r(t)] });
            b.push(Opc::VAndB32, &[r(o2)], &[imm(0x80), r(t)]);
            let g = optimized(&b.f, &[x], &[o1, o2]);
            assert_eq!(ops(&g), [Opc::VAndB32, Opc::VAndB32]);
            assert!(g.blocks[0].insts.iter().all(|i| i.uses.contains(&r(x))));
        }
        // The inner and has another reader: nothing changes.
        let mut b = Builder::new();
        let (x, t, o1, o2) = (b.v(), b.v(), b.v(), b.v());
        b.push(Opc::VAndB32, &[r(t)], &[imm(0xff), r(x)]);
        b.push(Opc::VAndB32, &[r(o1)], &[imm(0x7f), r(t)]);
        b.push(Opc::VLshlrevB32, &[r(o2)], &[imm(1), r(t)]);
        assert_eq!(ops(&optimized(&b.f, &[x], &[o1, o2])).len(), 3);
    }

    #[test]
    fn constant_shifts_merge() {
        proven("shl_shl");
        let mut b = Builder::new();
        let (x, t, out) = (b.v(), b.v(), b.v());
        b.push(Opc::VLshlrevB32, &[r(t)], &[imm(6), r(x)]);
        b.push(Opc::VLshlrevB32, &[r(out)], &[imm(2), r(t)]);
        let g = optimized(&b.f, &[x], &[out]);
        assert_eq!(g.blocks[0].insts[..].iter().map(|i| (i.op, i.uses[0])).collect::<Vec<_>>(), [(Opc::VLshlrevB32, imm(8))]);
        // 64-bit scalar shifts.
        proven("sshl64_sshl64");
        let mut b = Builder::new();
        let (x, t, out) = (b.reg(Class::S, 2), b.reg(Class::S, 2), b.reg(Class::S, 2));
        b.push(Opc::SLshlB64, &[r(t)], &[r(x), imm(6)]);
        b.push(Opc::SLshlB64, &[r(out)], &[r(t), imm(2)]);
        assert_eq!(ops(&optimized(&b.f, &[x], &[out])), [Opc::SLshlB64]);
        // Too far: x << 40 << 30 is zero, not one shift.
        let mut b = Builder::new();
        let (x, t, out) = (b.reg(Class::S, 2), b.reg(Class::S, 2), b.reg(Class::S, 2));
        b.push(Opc::SLshlB64, &[r(t)], &[r(x), imm(40)]);
        b.push(Opc::SLshlB64, &[r(out)], &[r(t), imm(30)]);
        assert_eq!(ops(&optimized(&b.f, &[x], &[out])).len(), 2);
    }

    #[test]
    fn selects_under_uniform_masks_merge() {
        for (on_false, k) in [(true, 0), (true, 0xff80_0000), (false, 0), (false, 0x3f80_0000)] {
            proven(if on_false { "cnd_cnd_false" } else { "cnd_cnd_true" });
            let mut b = Builder::new();
            let (x, a, c, m1, m2, t, out) = (b.v(), b.s(), b.s(), b.s(), b.s(), b.v(), b.v());
            b.push(Opc::SCmpLgU32, &[], &[r(a), imm(0)]);
            b.push(Opc::SCselectB32, &[r(m1)], &[imm(u32::MAX), imm(0)]);
            b.push(Opc::SCmpLgU32, &[], &[r(c), imm(0)]);
            b.push(Opc::SCselectB32, &[r(m2)], &[imm(u32::MAX), imm(0)]);
            let (inner, outer) = if on_false { ([imm(k), r(x)], [imm(k), r(t)]) } else { ([r(x), imm(k)], [r(t), imm(k)]) };
            b.push(Opc::VCndmaskB32, &[r(t)], &[inner[0], inner[1], r(m1)]);
            b.push(Opc::VCndmaskB32, &[r(out)], &[outer[0], outer[1], r(m2)]);
            let g = optimized(&b.f, &[x, a, c], &[out]);
            let combine = if on_false { Opc::SAndB32 } else { Opc::SOrB32 };
            assert_eq!(ops(&g)[4..], [combine, Opc::VCndmaskB32]);
        }
        // Masks from VALU compares: the and would wait for a VALU-to-SALU
        // round trip, so the selects stay.
        let mut b = Builder::new();
        let (x, y, m1, m2, t, out) = (b.v(), b.v(), b.s(), b.s(), b.v(), b.v());
        b.push(Opc::VCmp(0x41), &[r(m1)], &[r(x), imm(5)]);
        b.push(Opc::VCmp(0x41), &[r(m2)], &[r(y), imm(9)]);
        b.push(Opc::VCndmaskB32, &[r(t)], &[imm(0), r(x), r(m1)]);
        b.push(Opc::VCndmaskB32, &[r(out)], &[imm(0), r(t), r(m2)]);
        assert_eq!(ops(&optimized(&b.f, &[x, y], &[out])).len(), 4);
    }

    #[test]
    fn every_family_instantiates_proven_rules() {
        let prefixes = [
            ("addr", "addr_"),
            ("bounds", "bounds_"),
            ("cselect", "cselect_"),
            ("perm", "perm_perm_"),
            ("pack", "cvt_cvt_pack"),
            ("minmax", "fm"),
            ("bfe", "shr_and_"),
            ("and", "and_and"),
            ("shl", "shl_shl"),
            ("sshl", "sshl64_"),
            ("cndmask", "cnd_cnd_"),
        ];
        assert_eq!(prefixes.len(), FAMILIES.len());
        for ((family, prefix), name) in prefixes.iter().zip(FAMILIES) {
            assert_eq!(*family, name);
            assert!(PROVEN.iter().any(|(rule, _)| rule.starts_with(prefix)), "{family}");
        }
        for &(s1, s2, second, _, _) in PERM_PERM {
            let name = format!("perm_perm_{s1:08x}_{s2:08x}_{}", u8::from(second));
            assert!(PROVEN.iter().any(|(rule, _)| *rule == name), "{name}");
        }
    }

    // Expected words from `llvm-mc -arch=amdgcn -mcpu=gfx1151 -show-encoding`.
    #[test]
    fn encodings_match_llvm_mc() {
        let v = |r: u16, n: u8| Opnd::P(Phys::V(r), n);
        let s = |r: u16, n: u8| Opnd::P(Phys::S(r), n);
        let cases: Vec<(MInst, Vec<u32>)> = vec![
            (MInst::new(Opc::VMadI64I32, [v(2, 2), NULL], [v(1, 1), imm(2), s(4, 2)]), vec![0xD6FF_7C02, 0x0011_0501]),
            (MInst::new(Opc::VMadI64I32, [v(2, 2), NULL], [v(1, 1), imm(64), v(6, 2)]), vec![0xD6FF_7C02, 0x0419_8101]),
            (MInst::new(Opc::VMadU64U32, [v(2, 2), NULL], [v(1, 1), imm(4), s(4, 2)]), vec![0xD6FE_7C02, 0x0011_0901]),
            (MInst::new(Opc::VMax3F32, [v(0, 1)], [v(1, 1), v(2, 1), v(3, 1)]), vec![0xD61C_0000, 0x040E_0501]),
            (MInst::new(Opc::VMax3F32, [v(0, 1)], [v(1, 1), s(2, 1), imm(0xff80_0000)]), vec![0xD61C_0000, 0x03FC_0501, 0xFF80_0000]),
            (MInst::new(Opc::VMin3F32, [v(0, 1)], [v(1, 1), v(2, 1), v(3, 1)]), vec![0xD619_0000, 0x040E_0501]),
            (MInst::new(Opc::VBfeU32, [v(0, 1)], [v(1, 1), imm(16), imm(1)]), vec![0xD610_0000, 0x0205_2101]),
            (MInst::new(Opc::VCmp(0x49), [s(0, 1)], [v(1, 1), s(2, 1)]), vec![0xD449_0000, 0x0000_0501]),
            (MInst::new(Opc::VCmp(0x49), [s(0, 1)], [v(1, 1), imm(0x1234)]), vec![0xD449_0000, 0x0001_FF01, 0x0000_1234]),
            (MInst::new(Opc::VPermB32, [v(0, 1)], [v(1, 1), v(2, 1), imm(0x0504_0100)]), vec![0xD644_0000, 0x03FE_0501, 0x0504_0100]),
            (MInst::new(Opc::SMaxI32, [s(0, 1)], [s(1, 1), imm(0)]), vec![0x8A00_8001]),
            (MInst::new(Opc::SCselectB32, [s(0, 1)], [s(1, 1), imm(0)]), vec![0x9800_8001]),
            (MInst::new(Opc::SCselectB32, [s(0, 1)], [imm(u32::MAX), s(1, 1)]), vec![0x9800_01C1]),
            (MInst::new(Opc::SLshlB64, [s(0, 2)], [s(2, 2), imm(8)]), vec![0x8480_8802]),
            (MInst::new(Opc::VLshrrevB32, [v(0, 1)], [imm(24), v(1, 1)]), vec![0x3200_0298]),
            (MInst::new(Opc::VLshlrevB32, [v(0, 1)], [imm(2), v(1, 1)]), vec![0x3000_0282]),
            (MInst::new(Opc::VAndB32, [v(0, 1)], [imm(0x7f), v(1, 1)]), vec![0x3600_02FF, 0x0000_007F]),
            (MInst::new(Opc::SAndB32, [s(0, 1)], [s(1, 1), s(2, 1)]), vec![0x8B00_0201]),
            (MInst::new(Opc::SOrB32, [s(0, 1)], [s(1, 1), s(2, 1)]), vec![0x8C00_0201]),
            (MInst::new(Opc::VCndmaskB32, [v(0, 1)], [imm(0), v(1, 1), s(3, 1)]), vec![0xD501_0000, 0x000E_0280]),
        ];
        for (inst, want) in cases {
            let f = MFunc { blocks: vec![MBlock { insts: vec![inst.clone()], loop_depth: 0 }], vregs: Vec::new(), ..Default::default() };
            let words: Vec<u32> = super::super::asm::assemble(&f)
                .expect("encodes")
                .chunks(4)
                .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            assert_eq!(words, want, "{}", inst.op.name());
        }
    }
}
