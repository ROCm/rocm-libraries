// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

//! `s_waitcnt` on gfx950 (design section 4), on physical registers after
//! allocation. The model is gfx11's (`rir::post::insert_waitcnt`): a count
//! per register of the counter ops issued since the op that writes it,
//! merged over a block's predecessors, and a wait before the first read or
//! overwrite of an outstanding result. The counters are gfx9's:
//!
//! - vmcnt counts VMEM loads and stores alike, which complete in issue
//!   order (gfx9 has no separate store counter);
//! - lgkmcnt counts LDS ops and scalar loads; scalar loads return out of
//!   order, so only lgkmcnt(0) settles one;
//! - the immediate holds vmcnt in bits 3:0 and 15:14, expcnt in 6:4 and
//!   lgkmcnt in 11:8 (4 bits). A count past what the counter tells apart
//!   is clamped as LLVM clamps it (vmcnt 62, lgkmcnt 14): waiting for fewer
//!   outstanding ops is always correct.
//!
//! LDS DMA loads write no register: each LDS range they write (their LDS
//! tag, `lds_dma`) is a key of its own on vmcnt, after the registers, and an
//! instruction that reads or writes LDS memory waits for the keys whose
//! ranges overlap its tag. An explicit `s_waitcnt` (`tile.s_waitcnt`, a
//! fence's) retires what it waits for; `s_barrier` (`tile.sync`) waits for
//! every outstanding counter, and the LDS-only barrier for lgkmcnt alone,
//! leaving DMA loads in flight.

use super::gen_isa as t;
use super::lds_dma;
use crate::rir::mir::*;

/// LDS ranges of a function's DMA loads with keys of their own; the last
/// key takes every range past them (and so all of LDS).
const DMA_KEYS: usize = 16;
/// The first LDS DMA key.
const DMA_BASE: usize = 256 + 128 + 256;

/// VGPRs 0..255, then SGPRs 0..127 (VCC 106-107, M0 124, EXEC 126-127),
/// then AGPRs 0..255, then the LDS DMA keys.
const NREGS: usize = DMA_BASE + DMA_KEYS;
const NONE: u8 = u8::MAX;
const VM_MAX: u8 = 62;
const LGKM_MAX: u8 = 14;

fn reg_keys(o: &Opnd) -> std::ops::Range<usize> {
    let (start, n) = match *o {
        Opnd::P(Phys::V(r), n) if r >= super::asm::AGPR0 => (384 + (r - super::asm::AGPR0) as usize, n as usize),
        Opnd::P(Phys::V(r), n) => (r as usize, n as usize),
        Opnd::P(Phys::S(r), n) => (256 + r as usize, n as usize),
        Opnd::P(Phys::VccLo, n) => (256 + 106, n as usize),
        Opnd::P(Phys::ExecLo, n) => (256 + 126, n as usize),
        Opnd::P(Phys::M0, _) => (256 + 124, 1),
        _ => (0, 0),
    };
    start..start + n
}

#[derive(Clone, PartialEq)]
struct State {
    /// Per register: counter ops issued since the one writing it.
    vm: [u8; NREGS],
    lgkm: [u8; NREGS],
    vm_any: bool,
    lgkm_any: bool,
    smem: bool,
}

impl State {
    fn new() -> Self {
        State { vm: [NONE; NREGS], lgkm: [NONE; NREGS], vm_any: false, lgkm_any: false, smem: false }
    }

    fn merge(&mut self, other: &State) {
        for (a, b) in self.vm.iter_mut().zip(&other.vm) {
            *a = (*a).min(*b);
        }
        for (a, b) in self.lgkm.iter_mut().zip(&other.lgkm) {
            *a = (*a).min(*b);
        }
        self.vm_any |= other.vm_any;
        self.lgkm_any |= other.lgkm_any;
        self.smem |= other.smem;
    }
}

/// One counter: an op on it ages every outstanding register by one; a wait
/// for `n` drops every register at least `n` ops old.
fn issue(counts: &mut [u8; NREGS], defs: &[Opnd]) {
    for c in counts.iter_mut() {
        if *c != NONE {
            *c = c.saturating_add(1).min(NONE - 1);
        }
    }
    for k in defs.iter().flat_map(reg_keys) {
        counts[k] = 0;
    }
}

fn wait(counts: &mut [u8; NREGS], n: u8) {
    for c in counts.iter_mut() {
        if *c != NONE && *c >= n {
            *c = NONE;
        }
    }
}

fn lds_merge(inst: &MInst) -> bool {
    inst.op.flags() & (F_LDS | F_TIED) == F_LDS | F_TIED
}

/// The use of a d16 load that is its destination's old value: its last.
/// Any other use naming the destination, such as an address in the
/// register the load overwrites, is read when the load issues.
fn tied_source(inst: &MInst) -> Option<usize> {
    let last = inst.uses.len().checked_sub(1)?;
    (inst.op.flags() & F_TIED != 0 && inst.defs.contains(&inst.uses[last])).then_some(last)
}

/// The LDS ranges (tags) of a function's DMA loads, by key: the first
/// `DMA_KEYS - 1` distinct ones in program order, then all of LDS.
struct DmaKeys {
    tags: Vec<u16>,
}

impl DmaKeys {
    fn of(f: &MFunc) -> DmaKeys {
        let mut tags = Vec::new();
        for inst in f.blocks.iter().flat_map(|b| &b.insts).filter(|i| lds_dma::is_dma(i.op)) {
            if !tags.contains(&inst.mem) {
                tags.push(inst.mem);
            }
        }
        if tags.len() > DMA_KEYS {
            tags.truncate(DMA_KEYS - 1);
            tags.push(crate::rir::alias::ANY);
        }
        DmaKeys { tags }
    }

    /// The key of a DMA load tagged `tag`.
    fn key(&self, tag: u16) -> usize {
        DMA_BASE + self.tags.iter().position(|&t| t == tag).unwrap_or(self.tags.len() - 1)
    }
}

/// The wait `inst` needs, applied to `state`, then `inst` issued.
fn step(state: &mut State, inst: &MInst, dma: &DmaKeys) -> Option<(u8, u8)> {
    let flags = inst.op.flags();
    let vmem_load = flags & F_VMEM_LOAD != 0;
    let lds_merge = lds_merge(inst);
    let (mut need_vm, mut need_lgkm) = (NONE, NONE);
    let tied_use = tied_source(inst);
    for (i, o) in inst.uses.iter().chain(inst.defs.iter()).enumerate() {
        // Loads write their results in issue order, so neither a load's own
        // destination nor a d16 load's tied source waits for an earlier load
        // of its counter.
        let tied = i >= inst.uses.len() || tied_use == Some(i);
        for k in reg_keys(o) {
            if !(vmem_load && tied) {
                need_vm = need_vm.min(state.vm[k]);
            }
            if !(lds_merge && tied) {
                need_lgkm = need_lgkm.min(state.lgkm[k]);
            }
        }
    }
    // An LDS access waits for the DMA loads into bytes it may touch; DMA
    // loads land in issue order, so one does not wait for another.
    if !dma.tags.is_empty() && lds_dma::accesses_lds_memory(inst.op) && !lds_dma::is_dma(inst.op) {
        for (k, &tag) in dma.tags.iter().enumerate() {
            if lds_dma::overlaps(tag, inst.mem) {
                need_vm = need_vm.min(state.vm[DMA_BASE + k]);
            }
        }
    }
    if inst.op == Opc::SBarrier {
        if state.vm_any {
            need_vm = 0;
        }
        if state.lgkm_any {
            need_lgkm = 0;
        }
    }
    if inst.op == Opc::SBarrierLds && state.lgkm_any {
        need_lgkm = 0;
    }
    if need_lgkm != NONE && state.smem {
        need_lgkm = 0;
    }
    let need_vm = if need_vm == NONE { NONE } else { need_vm.min(VM_MAX) };
    let need_lgkm = if need_lgkm == NONE { NONE } else { need_lgkm.min(LGKM_MAX) };
    let out = if need_vm != NONE || need_lgkm != NONE {
        if need_vm != NONE {
            wait(&mut state.vm, need_vm);
            state.vm_any &= need_vm != 0;
        }
        if need_lgkm != NONE {
            wait(&mut state.lgkm, need_lgkm);
            if need_lgkm == 0 {
                state.lgkm_any = false;
                state.smem = false;
            }
        }
        Some((need_vm, need_lgkm))
    } else {
        None
    };
    let counters = t::counters(inst.op);
    if counters & t::CNT_VM != 0 {
        issue(&mut state.vm, &inst.defs);
        state.vm_any = true;
        if lds_dma::is_dma(inst.op) {
            state.vm[dma.key(inst.mem)] = 0;
        }
    }
    if counters & t::CNT_LGKM != 0 {
        issue(&mut state.lgkm, &inst.defs);
        state.lgkm_any = true;
        state.smem |= counters & t::CNT_SMEM != 0;
    }
    if inst.op == Opc::SWaitcntExplicit {
        retire(state, inst.imm);
    }
    out
}

/// What an explicit `s_waitcnt` with immediate `imm` waits for, taken out of
/// `state`. Scalar loads return out of order: with one outstanding, only an
/// lgkmcnt of 0 settles anything on that counter.
fn retire(state: &mut State, imm: i64) {
    let vm = (imm & 0xf | (imm >> 14 & 3) << 4) as u8;
    let lgkm = (imm >> 8 & 0xf) as u8;
    if vm < 63 {
        wait(&mut state.vm, vm);
        state.vm_any &= vm != 0;
    }
    if lgkm == 0 {
        wait(&mut state.lgkm, 0);
        state.lgkm_any = false;
        state.smem = false;
    } else if lgkm < 15 && !state.smem {
        wait(&mut state.lgkm, lgkm);
    }
}

/// `s_waitcnt` immediate for vmcnt `vm` and lgkmcnt `lgkm` (`NONE`: no wait).
pub fn waitcnt_imm(vm: u8, lgkm: u8) -> i64 {
    let vm = if vm == NONE { 63 } else { vm.min(VM_MAX) as i64 };
    let lgkm = if lgkm == NONE { 15 } else { lgkm.min(LGKM_MAX) as i64 };
    (vm & 0xf) | (vm >> 4) << 14 | 7 << 4 | lgkm << 8
}

/// The immediate of `tile.s_waitcnt` as rocKE's lowering encodes it for gfx9:
/// a negative count is no wait (the field's maximum), a larger one clamps to
/// the field (vmcnt 63, expcnt 7, lgkmcnt 15).
pub fn explicit_imm(vm: i64, exp: i64, lgkm: i64) -> i64 {
    let field = |c: i64, max: i64| if c < 0 { max } else { c.min(max) };
    let (vm, exp, lgkm) = (field(vm, 63), field(exp, 7), field(lgkm, 15));
    (vm & 0xf) | exp << 4 | lgkm << 8 | (vm >> 4 & 3) << 14
}

/// Inserts `s_waitcnt` before every instruction that reads or overwrites an
/// outstanding result, and before barriers.
pub fn insert(f: &mut MFunc) {
    let n = f.blocks.len();
    let succs: Vec<Vec<usize>> = (0..n).map(|b| f.successors(b)).collect();
    let dma = DmaKeys::of(f);
    let mut entry: Vec<Option<State>> = vec![None; n];
    entry[0] = Some(State::new());
    loop {
        let mut changed = false;
        for b in 0..n {
            let Some(start) = entry[b].clone() else {
                continue;
            };
            let mut state = start;
            for inst in &f.blocks[b].insts {
                step(&mut state, inst, &dma);
            }
            for &s in &succs[b] {
                let merged = match &entry[s] {
                    None => state.clone(),
                    Some(old) => {
                        let mut m = old.clone();
                        m.merge(&state);
                        m
                    }
                };
                if entry[s].as_ref() != Some(&merged) {
                    entry[s] = Some(merged);
                    changed = true;
                }
            }
        }
        if !changed {
            break;
        }
    }
    for b in 0..n {
        let mut state = entry[b].clone().unwrap_or_else(State::new);
        let insts = std::mem::take(&mut f.blocks[b].insts);
        let mut out = Vec::with_capacity(insts.len() + 8);
        for inst in insts {
            if let Some((vm, lgkm)) = step(&mut state, &inst, &dma) {
                out.push(MInst::new(Opc::SWaitcnt, [], []).with_imm(waitcnt_imm(vm, lgkm)));
            }
            out.push(inst);
        }
        f.blocks[b].insts = out;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v(r: u16, n: u8) -> Opnd {
        Opnd::P(Phys::V(r), n)
    }

    fn s(r: u16, n: u8) -> Opnd {
        Opnd::P(Phys::S(r), n)
    }

    fn waits(insts: Vec<MInst>) -> Vec<(usize, i64)> {
        let mut f = MFunc { target: crate::target::Target::Gfx950, ..MFunc::default() };
        f.blocks = vec![MBlock { insts, loop_depth: 0 }];
        insert(&mut f);
        let mut out = Vec::new();
        let mut k = 0;
        for i in &f.blocks[0].insts {
            if i.op == Opc::SWaitcnt {
                out.push((k, i.imm));
            } else {
                k += 1;
            }
        }
        out
    }

    #[test]
    fn the_immediate_is_gfx9s() {
        // As llvm-mc -mcpu=gfx950 encodes them.
        assert_eq!(waitcnt_imm(17, 3), 0x4371);
        assert_eq!(waitcnt_imm(0, NONE), 0x0f70);
        assert_eq!(waitcnt_imm(NONE, 0), 0xc07f);
        assert_eq!(waitcnt_imm(0, 0), 0x0070);
        // Counts past the counters' range wait for their maxima.
        assert_eq!(waitcnt_imm(200, 30), waitcnt_imm(62, 14));
    }

    #[test]
    fn stores_count_on_vmcnt() {
        // load v1; store; load v2; use v1: two later vm ops, so vmcnt(2)
        // (gfx11 would wait vmcnt(1): its stores are on vscnt).
        let load = |d: Opnd| MInst::new(Opc::GLoadB32, [d], [v(10, 2)]);
        let store = MInst::new(Opc::GStoreB32, [], [v(10, 2), v(20, 1)]);
        let use1 = MInst::new(Opc::VAddF32, [v(30, 1)], [v(1, 1), v(1, 1)]);
        assert_eq!(waits(vec![load(v(1, 1)), store.clone(), load(v(2, 1)), use1.clone()]), [(3, waitcnt_imm(2, NONE))]);
        // A store's data is read at issue: overwriting it waits for nothing.
        let overwrite = MInst::new(Opc::VMovB32, [v(20, 1)], [Opnd::Imm(0)]);
        assert!(waits(vec![store, overwrite]).is_empty());
    }

    #[test]
    fn scalar_loads_settle_at_zero_and_lgkmcnt_saturates() {
        let sload = MInst::new(Opc::SLoadB64, [s(4, 2)], [s(0, 2)]).with_imm(0);
        let lds = |d: u16| MInst::new(Opc::DsLoadB32, [v(d, 1)], [v(100, 1)]);
        let use_s = MInst::new(Opc::VAddF32, [v(30, 1)], [s(4, 1), v(1, 1)]);
        assert_eq!(waits(vec![sload, lds(1), use_s]), [(2, waitcnt_imm(NONE, 0))]);
        // Twenty LDS loads, then a use of the first: 19 younger, clamped to 14.
        let mut code: Vec<MInst> = (0..20).map(|k| lds(k + 1)).collect();
        code.push(MInst::new(Opc::VAddF32, [v(60, 1)], [v(1, 1), v(1, 1)]));
        assert_eq!(waits(code), [(20, waitcnt_imm(NONE, 14))]);
    }

    #[test]
    fn barriers_wait_for_everything_outstanding() {
        let load = MInst::new(Opc::GLoadB32, [v(1, 1)], [v(10, 2)]);
        let lds = MInst::new(Opc::DsStoreB32, [], [v(11, 1), v(12, 1)]);
        assert_eq!(waits(vec![load, lds, MInst::new(Opc::SBarrier, [], [])]), [(2, waitcnt_imm(0, 0))]);
    }

    #[test]
    fn an_address_in_the_destination_waits_for_its_load() {
        // A reload into v1, then a load through v1 into v1: the address is
        // read at issue, unlike the destination, so it waits, on either
        // counter; a d16 load's kept half still does not.
        let saddr = s(0, 2);
        let code = vec![
            MInst::new(Opc::ScratchLoadB32, [v(1, 1)], []).with_imm(16),
            MInst::new(Opc::GLoadB32, [v(1, 1)], [v(1, 1), saddr]).with_mods(M_SADDR),
            MInst::new(Opc::DsLoadU16, [v(2, 1)], [v(12, 1)]),
            MInst::new(Opc::DsLoadU16D16Hi, [v(2, 1)], [v(2, 1), v(2, 1)]),
            MInst::new(Opc::GLoadB32, [v(3, 1)], [v(13, 1), saddr]).with_mods(M_SADDR),
            MInst::new(Opc::GLoadD16HiB16, [v(3, 1)], [v(14, 1), saddr, v(3, 1)]).with_mods(M_SADDR),
        ];
        assert_eq!(waits(code), [(1, waitcnt_imm(0, NONE)), (3, waitcnt_imm(NONE, 0))]);
    }

    #[test]
    fn agprs_are_not_the_sgprs_of_their_number() {
        // s_load into s[4:5], then moves through a4 and a5: nothing
        // outstanding; s4 itself still waits.
        use crate::target::gfx9::gen_isa::Gfx9Op;
        let a = |k: u16| v(super::super::asm::AGPR0 + k, 1);
        let sload = MInst::new(Opc::SLoadB64, [s(4, 2)], [s(0, 2)]).with_imm(0);
        let write = MInst::new(Opc::Gfx9(Gfx9Op::VAccvgprWriteB32), [a(4)], [v(1, 1)]);
        let read = MInst::new(Opc::Gfx9(Gfx9Op::VAccvgprReadB32), [v(2, 1)], [a(5)]);
        let use_s = MInst::new(Opc::VAddF32, [v(30, 1)], [s(4, 1), v(1, 1)]);
        assert_eq!(waits(vec![sload, write, read, use_s]), [(3, waitcnt_imm(NONE, 0))]);
    }
}
