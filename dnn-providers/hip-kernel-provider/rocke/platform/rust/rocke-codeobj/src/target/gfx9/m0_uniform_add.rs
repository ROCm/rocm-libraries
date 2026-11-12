// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

//! The first lane of a sum with a uniform addend is the sum with the first
//! lane (gfx950 LDS DMA).
//!
//! An LDS DMA load's M0 is the first lane of its LDS address
//! (`lds_dma::select`). rocKE addresses are `base + ring * stride`: a
//! per-wave VGPR `base` (computed from the thread id, constant across the
//! kernel) plus a uniform ring offset, so each load, every loop iteration,
//! was `v_add_u32 v, s_ring, v_base` (a second one for a constant), then
//! `v_readfirstlane_b32`, then `s_mov_b32 m0`: two VALU instructions and a
//! cross-domain hazard per load, sixteen loads per K/V tile. COMGR does the
//! same.
//!
//! `readfirstlane(x + u)` is `readfirstlane(x) + u` for a uniform `u` (an
//! SGPR or a constant; `u32` wraps alike), so each such read becomes
//! `s_add_i32` of the first lane of the base, which is read once where the
//! base is defined (the kernel has no EXEC write there, so its first lane is
//! the kernel's) and held in an SGPR: the loop keeps one `s_add_i32` per
//! load, and the 16 per-load base VGPRs go.
//!
//! Applies to a `v_readfirstlane_b32` of a value that only it reads, defined
//! in its block by a `v_add_nc_u32` of a uniform and a VGPR (a chain of them
//! folds its constants into one). The rewrite stays where SCC, which the
//! `s_add_i32` writes, is dead. `ROCKE_DIRECT_DEBUG=norfladd` skips it.

use crate::rir::fx::{FxHashMap, FxHashSet};
use crate::rir::mir::*;

fn writes_exec(inst: &MInst) -> bool {
    inst.op.flags() & F_DEF_EXEC != 0 || inst.defs.iter().any(|d| matches!(d, Opnd::P(Phys::ExecLo, _)))
}

/// Whether SCC is dead after position `k` of `insts`: an instruction writing
/// it comes before any that reads it, within the block.
fn scc_dead_after(insts: &[MInst], k: usize) -> bool {
    for inst in &insts[k + 1..] {
        let flags = inst.op.flags();
        if flags & F_USE_SCC != 0 {
            return false;
        }
        if flags & F_DEF_SCC != 0 {
            return true;
        }
    }
    false
}

fn whole_v(f: &MFunc, o: &Opnd) -> Option<u32> {
    match o {
        Opnd::R(v) if v.off == 0 && v.n == 1 && f.vregs[v.reg as usize].size == 1 && f.vregs[v.reg as usize].class == Class::V => Some(v.reg),
        _ => None,
    }
}

/// An SGPR dword or a constant.
fn uniform(f: &MFunc, o: &Opnd) -> bool {
    match o {
        Opnd::Imm(_) => true,
        Opnd::R(v) => v.off == 0 && v.n == 1 && f.vregs[v.reg as usize].size == 1 && f.vregs[v.reg as usize].class == Class::S,
        _ => false,
    }
}

/// How many reads became first-lane-plus-uniform.
pub fn run(f: &mut MFunc) -> usize {
    if crate::rir::debug_flag("norfladd") {
        return 0;
    }
    if !f.blocks.iter().flat_map(|b| &b.insts).any(|i| i.op == Opc::VReadfirstlaneB32) {
        return 0;
    }
    let exec_free = !f.blocks.iter().flat_map(|b| &b.insts).any(writes_exec);
    // Definitions and uses of each register.
    let mut defs: Vec<Vec<(usize, usize)>> = vec![Vec::new(); f.vregs.len()];
    let mut uses = vec![0u32; f.vregs.len()];
    for (b, block) in f.blocks.iter().enumerate() {
        for (k, inst) in block.insts.iter().enumerate() {
            for d in &inst.defs {
                if let Opnd::R(v) = d {
                    defs[v.reg as usize].push((b, k));
                }
            }
            for u in &inst.uses {
                if let Opnd::R(v) = u {
                    uses[v.reg as usize] += 1;
                }
            }
        }
    }
    // Edits: per block, instructions to insert after a position, replacements, removals.
    let mut after: FxHashMap<(usize, usize), Vec<MInst>> = FxHashMap::default();
    let mut replace: FxHashMap<(usize, usize), Vec<MInst>> = FxHashMap::default();
    let mut remove: FxHashSet<(usize, usize)> = FxHashSet::default();
    // The SGPR holding the first lane of a base, read where it is defined.
    let mut firsts: FxHashMap<(u32, u32), u32> = FxHashMap::default();
    let new_sreg = |f: &mut MFunc| -> Opnd {
        let reg = f.vregs.len() as u32;
        f.vregs.push(VRegInfo { class: Class::S, size: 1, align: 1, fixed: None });
        Opnd::R(VRef { reg, off: 0, n: 1 })
    };
    let mut rewritten = 0;
    for b in 0..f.blocks.len() {
        for k in 0..f.blocks[b].insts.len() {
            let rfl = &f.blocks[b].insts[k];
            if rfl.op != Opc::VReadfirstlaneB32 {
                continue;
            }
            let (Some(&dst), Some(top)) = (rfl.defs.first(), rfl.uses.first().and_then(|o| whole_v(f, o))) else { continue };
            // The chain of adds, outermost first, in this block before `k`.
            let mut cur = top;
            let mut sregs: Vec<Opnd> = Vec::new();
            let mut imm: u32 = 0;
            let mut chain: Vec<usize> = Vec::new();
            loop {
                if defs[cur as usize].len() != 1 || uses[cur as usize] != 1 || f.vregs[cur as usize].fixed.is_some() {
                    break;
                }
                let (db, dk) = defs[cur as usize][0];
                if db != b || dk >= k {
                    break;
                }
                let inst = &f.blocks[b].insts[dk];
                if inst.op != Opc::VAddNcU32 || inst.uses.len() != 2 {
                    break;
                }
                let (a0, a1) = (inst.uses[0], inst.uses[1]);
                let (u, x) = if uniform(f, &a0) && whole_v(f, &a1).is_some() {
                    (a0, a1)
                } else if uniform(f, &a1) && whole_v(f, &a0).is_some() {
                    (a1, a0)
                } else {
                    break;
                };
                // A uniform register is not redefined between the add and the read.
                if let Opnd::R(r) = u {
                    if f.blocks[b].insts[dk + 1..k].iter().any(|i| i.defs.iter().any(|d| matches!(d, Opnd::R(x) if x.reg == r.reg))) {
                        break;
                    }
                }
                match u {
                    Opnd::Imm(c) => imm = imm.wrapping_add(c),
                    other => sregs.push(other),
                }
                chain.push(dk);
                cur = whole_v(f, &x).expect("a VGPR");
            }
            if chain.is_empty() || !scc_dead_after(&f.blocks[b].insts, k) {
                continue;
            }
            let base = cur;
            // Where the base's first lane is read: right after its definition when EXEC does not change
            // between that and the use, else here.
            let placed = match defs[base as usize].as_slice() {
                [(bb, dk)] if f.vregs[base as usize].fixed.is_none() => {
                    let insts = &f.blocks[*bb].insts;
                    if *bb == b && *dk < k {
                        (!insts[*dk + 1..=k].iter().any(writes_exec)).then_some((*bb, *dk))
                    } else {
                        exec_free.then_some((*bb, *dk))
                    }
                }
                _ => None,
            };
            // The constant joins the hoisted first lane: one more SGPR, no add per use.
            let mut folded_imm = false;
            let first = match placed {
                Some((bb, dk)) => {
                    let zero = match firsts.get(&(base, 0)) {
                        Some(&s) => Opnd::R(VRef { reg: s, off: 0, n: 1 }),
                        None => {
                            let s = new_sreg(f);
                            let Opnd::R(sv) = s else { unreachable!() };
                            firsts.insert((base, 0), sv.reg);
                            let read = MInst::new(Opc::VReadfirstlaneB32, [s], [Opnd::R(VRef { reg: base, off: 0, n: 1 })]);
                            after.entry((bb, dk)).or_default().push(read);
                            s
                        }
                    };
                    if imm == 0 {
                        zero
                    } else {
                        folded_imm = true;
                        match firsts.get(&(base, imm)) {
                            Some(&s) => Opnd::R(VRef { reg: s, off: 0, n: 1 }),
                            None => {
                                let s = new_sreg(f);
                                let Opnd::R(sv) = s else { unreachable!() };
                                firsts.insert((base, imm), sv.reg);
                                after.entry((bb, dk)).or_default().push(MInst::new(Opc::SAddI32, [s], [zero, Opnd::Imm(imm)]));
                                s
                            }
                        }
                    }
                }
                None => {
                    let s = new_sreg(f);
                    let read = MInst::new(Opc::VReadfirstlaneB32, [s], [Opnd::R(VRef { reg: base, off: 0, n: 1 })]);
                    replace.entry((b, k)).or_default().push(read);
                    s
                }
            };
            // first + the uniform registers + the constant, the last into the read's destination.
            let mut terms: Vec<Opnd> = sregs.into_iter().rev().collect();
            if (imm != 0 && !folded_imm) || terms.is_empty() {
                terms.push(Opnd::Imm(if folded_imm { 0 } else { imm }));
            }
            // The sum goes straight into M0 when the read feeds only the move to it, which then goes (so the
            // write can move above the address arithmetic of its load: `m0_early`).
            let m0_write = match dst {
                Opnd::R(d) if uses[d.reg as usize] == 1 => f.blocks[b].insts[k + 1..]
                    .iter()
                    .position(|i| i.op == Opc::SMovB32 && i.uses.first() == Some(&dst) && matches!(i.defs.first(), Some(Opnd::P(Phys::M0, _))))
                    .map(|j| k + 1 + j),
                _ => None,
            };
            let m0_write = m0_write.filter(|&j| {
                let (head, tail) = (&f.blocks[b].insts[k + 1..j], &f.blocks[b].insts[j]);
                let _ = tail;
                !head.iter().any(|i| i.defs.iter().chain(i.uses.iter()).any(|o| matches!(o, Opnd::P(Phys::M0, _))) || i.defs.contains(&dst))
            });
            let dst = match m0_write {
                Some(_) => Opnd::P(Phys::M0, 1),
                None => dst,
            };
            let mut seq: Vec<MInst> = Vec::new();
            let mut acc = first;
            for (i, t) in terms.iter().enumerate() {
                let out = if i + 1 == terms.len() { dst } else { new_sreg(f) };
                seq.push(MInst::new(Opc::SAddI32, [out], [acc, *t]));
                acc = out;
            }
            replace.entry((b, k)).or_default().extend(seq);
            for dk in chain {
                remove.insert((b, dk));
            }
            if let Some(j) = m0_write {
                remove.insert((b, j));
            }
            rewritten += 1;
        }
    }
    if rewritten == 0 {
        return 0;
    }
    for b in 0..f.blocks.len() {
        let old = std::mem::take(&mut f.blocks[b].insts);
        let mut out = Vec::with_capacity(old.len());
        for (k, inst) in old.into_iter().enumerate() {
            if remove.contains(&(b, k)) {
                // Dropped: its one reader is rewritten.
            } else if let Some(seq) = replace.remove(&(b, k)) {
                let start = out.len();
                out.extend(seq);
                inherit_loc(&mut out, start, inst.loc);
            } else {
                out.push(inst);
            }
            if let Some(extra) = after.remove(&(b, k)) {
                out.extend(extra);
            }
        }
        f.blocks[b].insts = out;
    }
    rewritten
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::target::Target;

    fn r(reg: u32) -> Opnd {
        Opnd::R(VRef { reg, off: 0, n: 1 })
    }

    fn m0() -> Opnd {
        Opnd::P(Phys::M0, 1)
    }

    /// v0 = tid (V), s1 = ring (S), v2 = v0 << 10, then per load: v3 = s1 + v2, s4 = first lane of v3, M0 = s4.
    fn func(extra: Vec<MInst>) -> MFunc {
        let mut f = MFunc { target: Target::Gfx950, ..MFunc::default() };
        for class in [Class::V, Class::S, Class::V, Class::V, Class::S, Class::S, Class::V, Class::S] {
            f.vregs.push(VRegInfo { class, size: 1, align: 1, fixed: None });
        }
        let mut insts = vec![
            MInst::new(Opc::VLshlrevB32, [r(2)], [Opnd::Imm(10), r(0)]),
            MInst::new(Opc::VAddNcU32, [r(3)], [r(1), r(2)]),
            MInst::new(Opc::VReadfirstlaneB32, [r(4)], [r(3)]),
            MInst::new(Opc::SMovB32, [m0()], [r(4)]),
            MInst::new(Opc::SCmpEqU32, [], [r(1), Opnd::Imm(0)]),
        ];
        insts.extend(extra);
        f.blocks.push(MBlock { insts, loop_depth: 0 });
        f
    }

    #[test]
    fn the_first_lane_of_a_sum_is_the_sum_with_the_first_lane() {
        let mut f = func(vec![]);
        assert_eq!(run(&mut f), 1);
        let ops: Vec<Opc> = f.blocks[0].insts.iter().map(|i| i.op).collect();
        // The add and the move to M0 are gone; the first lane is read after the shift, M0 gets the sum.
        assert_eq!(ops, [Opc::VLshlrevB32, Opc::VReadfirstlaneB32, Opc::SAddI32, Opc::SCmpEqU32]);
        let add = &f.blocks[0].insts[2];
        assert_eq!((add.defs[0], add.uses[1]), (m0(), r(1)));
        assert_eq!(f.blocks[0].insts[1].uses[0], r(2));
    }

    #[test]
    fn constants_fold_into_one_add_and_scc_must_be_dead() {
        // A second add of a constant folds into the hoisted first lane: one SALU add left per read.
        let mut f = func(vec![]);
        f.blocks[0].insts[1] = MInst::new(Opc::VAddNcU32, [r(3)], [Opnd::Imm(0x8800), r(2)]);
        assert_eq!(run(&mut f), 1);
        let adds: Vec<&MInst> = f.blocks[0].insts.iter().filter(|i| i.op == Opc::SAddI32).collect();
        // The constant is added to the hoisted first lane, and the last add writes M0 (of zero, which is the move).
        assert_eq!(adds.len(), 2, "{:?}", f.blocks[0].insts);
        assert_eq!(adds[0].uses[1], Opnd::Imm(0x8800));
        // SCC read after the read, before any write: the rewrite stays off.
        let mut live = func(vec![]);
        live.blocks[0].insts.insert(4, MInst::new(Opc::SCselectB32, [r(5)], [Opnd::Imm(1), Opnd::Imm(0)]));
        let before = live.blocks[0].insts.len();
        assert_eq!(run(&mut live), 0);
        assert_eq!(live.blocks[0].insts.len(), before);
    }

    #[test]
    fn a_vector_addend_or_a_second_reader_keeps_the_add() {
        // v1 divergent addend: not uniform.
        let mut f = func(vec![]);
        f.blocks[0].insts[1] = MInst::new(Opc::VAddNcU32, [r(3)], [r(6), r(2)]);
        assert_eq!(run(&mut f), 0);
        // v3 read again: the add stays.
        let mut g = func(vec![MInst::new(Opc::VAddNcU32, [r(6)], [r(3), r(0)])]);
        assert_eq!(run(&mut g), 0);
    }

    #[test]
    fn an_exec_write_between_keeps_the_read_at_the_use() {
        let exec = MInst::new(Opc::SMovB32, [Opnd::P(Phys::ExecLo, 1)], [Opnd::Imm(1)]);
        let mut f = func(vec![]);
        f.blocks[0].insts.insert(2, exec);
        assert_eq!(run(&mut f), 1);
        let rfl = f.blocks[0].insts.iter().position(|i| i.op == Opc::VReadfirstlaneB32).unwrap();
        let exec = f.blocks[0].insts.iter().position(|i| matches!(i.defs.first(), Some(Opnd::P(Phys::ExecLo, _)))).unwrap();
        assert!(rfl > exec, "{:?}", f.blocks[0].insts);
    }
}
