// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

//! LDS dword pairs on gfx950 (lever `ldspairs`, `crate::rir::lever`), as
//! LLVM's `SILoadStoreOptimizer`: two `ds_read_b32` of one address register
//! become one `ds_read2_b32` and two `ds_write_b32` one `ds_write2_b32` when
//! their byte offsets are dword multiples below 1024 (the two 8-bit dword
//! offsets of the paired forms). Runs on virtual registers, before
//! allocation, inside one block:
//!
//! - loads: the pair is issued where the first load was; between the two
//!   nothing writes LDS (stores, atomics, DMA, raw-address accesses, fences
//!   and barriers end the window), EXEC or the address register; each
//!   destination is a one-dword VGPR defined only by its load, and both are
//!   renamed to the halves of one register pair;
//! - stores: the pair is issued where the second store was; between the two
//!   nothing reads or writes LDS, and neither the address nor the first
//!   store's data is written; the offsets differ.
//!
//! Every lane reads and writes the same bytes as before, in the same order
//! relative to every other LDS access of the wave, so the result is bitwise
//! the unpaired code's.

use crate::rir::fx::FxHashMap;
use crate::rir::mir::*;

use super::dpp::overlaps;
use super::gen_isa::Gfx9Op;

/// Whether `inst` ends a pairing window: it may write LDS, order other waves'
/// accesses, or change EXEC.
fn ends_window(inst: &MInst) -> bool {
    let flags = inst.op.flags();
    let lds_write = flags & F_LDS != 0 && !matches!(inst.op, Opc::DsLoadB32 | Opc::DsLoadB64 | Opc::DsLoadB128 | Opc::DsLoadU16 | Opc::DsLoadU8);
    lds_write
        || flags & (F_FENCE | F_DEF_EXEC | F_BRANCH | F_END) != 0
        || inst.defs.iter().any(|d| matches!(d, Opnd::P(Phys::ExecLo, _) | Opnd::P(Phys::M0, _)))
        || matches!(inst.op, Opc::SBarrier | Opc::SBarrierLds | Opc::SBarrierBare)
}

fn writes(inst: &MInst, o: &Opnd) -> bool {
    inst.defs.iter().any(|d| overlaps(d, o))
}

/// Defs per virtual register over the function.
fn def_counts(f: &MFunc) -> Vec<u32> {
    let mut defs = vec![0u32; f.vregs.len()];
    for inst in f.blocks.iter().flat_map(|b| &b.insts) {
        for d in &inst.defs {
            if let Opnd::R(v) = d {
                defs[v.reg as usize] += 1;
            }
        }
    }
    defs
}

fn pairable(offset: i64) -> bool {
    (0..1024).contains(&offset) && offset % 4 == 0
}

/// Pairs the eligible LDS dword loads and stores of `f`; returns how many pairs.
pub fn merge(f: &mut MFunc) -> usize {
    let defs = def_counts(f);
    let mut renames: FxHashMap<u32, VRef> = FxHashMap::default();
    let mut pairs = 0;
    for b in 0..f.blocks.len() {
        // ---- loads
        let mut i = 0;
        while i < f.blocks[b].insts.len() {
            let first = f.blocks[b].insts[i].clone();
            let one_dword = |inst: &MInst, defs: &[u32], f: &MFunc| -> bool {
                inst.op == Opc::DsLoadB32
                    && inst.mods == 0
                    && pairable(inst.imm)
                    && matches!(inst.defs.first(), Some(Opnd::R(v)) if v.n == 1 && v.off == 0 && f.vregs[v.reg as usize].size == 1 && f.vregs[v.reg as usize].class == Class::V && defs[v.reg as usize] == 1)
                    && matches!(inst.uses.first(), Some(Opnd::R(_)))
            };
            if !one_dword(&first, &defs, f) || !f.lever(&first, "ldspairs") {
                i += 1;
                continue;
            }
            let addr = first.uses[0];
            let mut partner = None;
            for k in i + 1..f.blocks[b].insts.len() {
                let x = &f.blocks[b].insts[k];
                if one_dword(x, &defs, f) && f.lever(x, "ldspairs") && x.uses[0] == addr && x.imm != first.imm {
                    partner = Some(k);
                    break;
                }
                if ends_window(x) || writes(x, &addr) {
                    break;
                }
            }
            let Some(k) = partner else {
                i += 1;
                continue;
            };
            let second = f.blocks[b].insts[k].clone();
            let t = f.vregs.len() as u32;
            f.vregs.push(VRegInfo { class: Class::V, size: 2, align: 2, fixed: None });
            let pair = VRef { reg: t, off: 0, n: 2 };
            let (Opnd::R(d1), Opnd::R(d2)) = (first.defs[0], second.defs[0]) else { unreachable!() };
            renames.insert(d1.reg, VRef { reg: t, off: 0, n: 1 });
            renames.insert(d2.reg, VRef { reg: t, off: 1, n: 1 });
            let imm = first.imm / 4 | (second.imm / 4) << 8;
            let mut merged = MInst::new(Opc::Gfx9(Gfx9Op::DsLoad2addrB32), [Opnd::R(pair)], [addr]).with_imm(imm);
            merged.mem = first.mem;
            merged.scope = first.scope;
            f.blocks[b].insts[i] = merged;
            f.blocks[b].insts.remove(k);
            pairs += 1;
            i += 1;
        }
        // ---- stores
        let mut i = 0;
        while i < f.blocks[b].insts.len() {
            let first = f.blocks[b].insts[i].clone();
            let one_dword = |inst: &MInst, f: &MFunc| -> bool {
                inst.op == Opc::DsStoreB32
                    && inst.mods == 0
                    && pairable(inst.imm)
                    && inst.uses.len() == 2
                    && matches!(inst.uses[0], Opnd::R(_))
                    && matches!(inst.uses[1], Opnd::R(v) if v.n == 1 && f.vregs[v.reg as usize].class == Class::V)
            };
            if !one_dword(&first, f) || !f.lever(&first, "ldspairs") {
                i += 1;
                continue;
            }
            let (addr, data0) = (first.uses[0], first.uses[1]);
            let mut partner = None;
            for k in i + 1..f.blocks[b].insts.len() {
                let x = &f.blocks[b].insts[k];
                if one_dword(x, f) && f.lever(x, "ldspairs") && x.uses[0] == addr && x.imm != first.imm {
                    partner = Some(k);
                    break;
                }
                // Any other LDS access (a read may see the first store), a fence, EXEC or the operands changing.
                if x.op.flags() & F_LDS != 0 || ends_window(x) || writes(x, &addr) || writes(x, &data0) {
                    break;
                }
            }
            let Some(k) = partner else {
                i += 1;
                continue;
            };
            let second = f.blocks[b].insts[k].clone();
            let imm = first.imm / 4 | (second.imm / 4) << 8;
            let mut merged = MInst::new(Opc::DsStore2addrB32, [], [addr, data0, second.uses[1]]).with_imm(imm);
            merged.mem = second.mem;
            merged.scope = second.scope;
            f.blocks[b].insts[k] = merged;
            f.blocks[b].insts.remove(i);
            pairs += 1;
        }
    }
    if !renames.is_empty() {
        for inst in f.blocks.iter_mut().flat_map(|b| b.insts.iter_mut()) {
            for o in inst.uses.iter_mut().chain(inst.defs.iter_mut()) {
                if let Opnd::R(v) = o {
                    if let Some(&to) = renames.get(&v.reg) {
                        debug_assert!(v.off == 0 && v.n == 1);
                        *o = Opnd::R(to);
                    }
                }
            }
        }
    }
    pairs
}

/// Lever `ldspairsfar` (PLAN decision 18): dword LDS loads of one address register whose byte offsets are past
/// `ds_read2_b32`'s 8-bit dword fields (the HC stages' MFMA scale reads at 45 KiB and more) pair too, through a
/// rebased address `addr + base` made once per 1 KiB window (one `v_add_u32`) and shared by every pair of that
/// window, as LLVM's `SILoadStoreOptimizer` rebases far offsets. The same rules as `merge` otherwise (one block,
/// nothing writing LDS, EXEC or the address between the two loads; the rebase is dropped at such an instruction).
/// The address each lane reads is unchanged, so the result is bitwise the unpaired code's.
pub fn merge_far(f: &mut MFunc) -> usize {
    let defs = def_counts(f);
    let mut renames: FxHashMap<u32, VRef> = FxHashMap::default();
    let mut pairs = 0;
    for b in 0..f.blocks.len() {
        // Rebased addresses made in this block and still valid: (address, base, register).
        let mut rebased: Vec<(Opnd, i64, VRef)> = Vec::new();
        let mut i = 0;
        while i < f.blocks[b].insts.len() {
            {
                let x = &f.blocks[b].insts[i];
                if ends_window(x) {
                    rebased.clear();
                } else {
                    rebased.retain(|(a, _, _)| !writes(x, a));
                }
            }
            let first = f.blocks[b].insts[i].clone();
            let one_dword = |inst: &MInst, defs: &[u32], f: &MFunc| -> bool {
                inst.op == Opc::DsLoadB32
                    && inst.mods == 0
                    && (0..65536).contains(&inst.imm)
                    && inst.imm % 4 == 0
                    && matches!(inst.defs.first(), Some(Opnd::R(v)) if v.n == 1 && v.off == 0 && f.vregs[v.reg as usize].size == 1 && f.vregs[v.reg as usize].class == Class::V && defs[v.reg as usize] == 1)
                    && matches!(inst.uses.first(), Some(Opnd::R(a)) if f.vregs[a.reg as usize].class == Class::V)
            };
            if !one_dword(&first, &defs, f) || !f.lever(&first, "ldspairsfar") {
                i += 1;
                continue;
            }
            let addr = first.uses[0];
            let mut partner = None;
            for k in i + 1..f.blocks[b].insts.len() {
                let x = &f.blocks[b].insts[k];
                if one_dword(x, &defs, f) && f.lever(x, "ldspairsfar") && x.uses[0] == addr && x.imm != first.imm && (x.imm - first.imm).abs() <= 1020 {
                    partner = Some(k);
                    break;
                }
                if ends_window(x) || writes(x, &addr) {
                    break;
                }
            }
            let Some(mut k) = partner else {
                i += 1;
                continue;
            };
            let second = f.blocks[b].insts[k].clone();
            let (lo, hi) = (first.imm.min(second.imm), first.imm.max(second.imm));
            let (base_reg, base) = match rebased.iter().find(|(a, base, _)| *a == addr && *base <= lo && hi - *base <= 1020) {
                Some(&(_, base, r)) => (Opnd::R(r), base),
                None if hi <= 1020 => (addr, 0),
                None => {
                    let t = f.vregs.len() as u32;
                    f.vregs.push(VRegInfo { class: Class::V, size: 1, align: 1, fixed: None });
                    let r = VRef { reg: t, off: 0, n: 1 };
                    let mut add = MInst::new(Opc::VAddNcU32, [Opnd::R(r)], [Opnd::Imm(lo as u32), addr]);
                    add.scope = first.scope;
                    f.blocks[b].insts.insert(i, add);
                    i += 1;
                    k += 1;
                    rebased.push((addr, lo, r));
                    (Opnd::R(r), lo)
                }
            };
            let t = f.vregs.len() as u32;
            f.vregs.push(VRegInfo { class: Class::V, size: 2, align: 2, fixed: None });
            let pair = VRef { reg: t, off: 0, n: 2 };
            let (Opnd::R(d1), Opnd::R(d2)) = (first.defs[0], second.defs[0]) else { unreachable!() };
            renames.insert(d1.reg, VRef { reg: t, off: 0, n: 1 });
            renames.insert(d2.reg, VRef { reg: t, off: 1, n: 1 });
            let imm = (first.imm - base) / 4 | ((second.imm - base) / 4) << 8;
            let mut merged = MInst::new(Opc::Gfx9(Gfx9Op::DsLoad2addrB32), [Opnd::R(pair)], [base_reg]).with_imm(imm);
            merged.mem = first.mem;
            merged.scope = first.scope;
            f.blocks[b].insts[i] = merged;
            f.blocks[b].insts.remove(k);
            pairs += 1;
            i += 1;
        }
    }
    if !renames.is_empty() {
        for inst in f.blocks.iter_mut().flat_map(|b| b.insts.iter_mut()) {
            for o in inst.uses.iter_mut().chain(inst.defs.iter_mut()) {
                if let Opnd::R(v) = o {
                    if let Some(&to) = renames.get(&v.reg) {
                        *o = Opnd::R(to);
                    }
                }
            }
        }
    }
    pairs
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vreg(f: &mut MFunc, size: u8) -> VRef {
        let reg = f.vregs.len() as u32;
        f.vregs.push(VRegInfo { class: Class::V, size, align: 1, fixed: None });
        VRef { reg, off: 0, n: size }
    }

    fn func(insts: impl FnOnce(&mut MFunc) -> Vec<MInst>) -> MFunc {
        let mut f = MFunc { target: crate::target::Target::Gfx950, ..MFunc::default() };
        let body = insts(&mut f);
        f.blocks = vec![MBlock { insts: body, loop_depth: 0 }];
        f
    }

    #[test]
    fn dword_pairs_of_one_address_merge() {
        let (mut regs, r) = (Vec::new(), Opnd::R);
        let mut f = func(|f| {
            regs = (0..5).map(|_| vreg(f, 1)).collect();
            let [a, d1, d2, x, s] = [regs[0], regs[1], regs[2], regs[3], regs[4]];
            vec![
                MInst::new(Opc::DsLoadB32, [r(d1)], [r(a)]).with_imm(4),
                MInst::new(Opc::VAddF32, [r(x)], [r(a), r(a)]),
                MInst::new(Opc::DsLoadB32, [r(d2)], [r(a)]).with_imm(40),
                MInst::new(Opc::VAddF32, [r(s)], [r(d1), r(d2)]),
                MInst::new(Opc::DsStoreB32, [], [r(a), r(x)]).with_imm(8),
                MInst::new(Opc::DsStoreB32, [], [r(a), r(s)]).with_imm(12),
            ]
        });
        assert_eq!(crate::rir::with_levers(&["ldspairs"], || merge(&mut f)), 2);
        let insts = &f.blocks[0].insts;
        assert_eq!(insts.len(), 4, "{insts:?}");
        assert_eq!(insts[0].op, Opc::Gfx9(Gfx9Op::DsLoad2addrB32));
        assert_eq!(insts[0].imm, 1 | 10 << 8);
        let Opnd::R(pair) = insts[0].defs[0] else { panic!() };
        assert_eq!((pair.n, f.vregs[pair.reg as usize].size), (2, 2));
        // The add reads the pair's halves.
        assert_eq!(insts[2].uses[..], [r(VRef { off: 0, n: 1, ..pair }), r(VRef { off: 1, n: 1, ..pair })]);
        assert_eq!(insts[3].op, Opc::DsStore2addrB32);
        assert_eq!(insts[3].imm, 2 | 3 << 8);
    }

    #[test]
    fn an_lds_write_between_keeps_the_loads_apart() {
        let r = Opnd::R;
        let mut f = func(|f| {
            let regs: Vec<VRef> = (0..4).map(|_| vreg(f, 1)).collect();
            let [a, d1, d2, x] = [regs[0], regs[1], regs[2], regs[3]];
            vec![
                MInst::new(Opc::DsLoadB32, [r(d1)], [r(a)]).with_imm(4),
                MInst::new(Opc::DsStoreB32, [], [r(a), r(x)]).with_imm(40),
                MInst::new(Opc::DsLoadB32, [r(d2)], [r(a)]).with_imm(40),
                MInst::new(Opc::VAddF32, [r(x)], [r(d1), r(d2)]),
            ]
        });
        assert_eq!(crate::rir::with_levers(&["ldspairs"], || merge(&mut f)), 0);
        // Offsets past the 8-bit dword range or misaligned do not pair either.
        let mut g = func(|f| {
            let regs: Vec<VRef> = (0..4).map(|_| vreg(f, 1)).collect();
            let [a, d1, d2, x] = [regs[0], regs[1], regs[2], regs[3]];
            vec![
                MInst::new(Opc::DsLoadB32, [r(d1)], [r(a)]).with_imm(4),
                MInst::new(Opc::DsLoadB32, [r(d2)], [r(a)]).with_imm(1024),
                MInst::new(Opc::DsLoadB32, [r(x)], [r(a)]).with_imm(6),
                MInst::new(Opc::VAddF32, [r(x)], [r(d1), r(d2)]),
            ]
        });
        assert_eq!(crate::rir::with_levers(&["ldspairs"], || merge(&mut g)), 0);
    }

    #[test]
    fn far_offsets_pair_through_one_rebase_per_window() {
        let (mut regs, r) = (Vec::new(), Opnd::R);
        let mut f = func(|f| {
            regs = (0..6).map(|_| vreg(f, 1)).collect();
            let a = regs[0];
            (0..4).map(|j| MInst::new(Opc::DsLoadB32, [r(regs[1 + j])], [r(a)]).with_imm(46352 + 64 * j as i64)).chain([MInst::new(Opc::VAddF32, [r(regs[5])], [r(regs[1]), r(regs[4])])]).collect()
        });
        assert_eq!(crate::rir::with_levers(&["ldspairsfar"], || merge_far(&mut f)), 2);
        let ops: Vec<Opc> = f.blocks[0].insts.iter().map(|i| i.op).collect();
        assert_eq!(ops, [Opc::VAddNcU32, Opc::Gfx9(Gfx9Op::DsLoad2addrB32), Opc::Gfx9(Gfx9Op::DsLoad2addrB32), Opc::VAddF32]);
        assert_eq!(f.blocks[0].insts[0].uses[0], Opnd::Imm(46352));
        assert_eq!(f.blocks[0].insts[1].imm, 16 << 8);
        assert_eq!(f.blocks[0].insts[2].imm, 32 | 48 << 8);
        // both pairs read the one rebased register
        assert_eq!(f.blocks[0].insts[1].uses[0], f.blocks[0].insts[2].uses[0]);
        // off by default
        let mut g = func(|f| {
            let a = vreg(f, 1);
            let (d1, d2) = (vreg(f, 1), vreg(f, 1));
            vec![MInst::new(Opc::DsLoadB32, [r(d1)], [r(a)]).with_imm(4096), MInst::new(Opc::DsLoadB32, [r(d2)], [r(a)]).with_imm(4160)]
        });
        assert_eq!(merge_far(&mut g), 0);
    }
}
