// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

//! Tuples built from values that are also used on their own (gfx950).
//!
//! `vector.pack` of f32 scalars is selected as a tuple of dwords, one copy
//! per element. The attention kernels with vectorized row sums
//! (`vec<f32x2>` adds) pack every pair of `exp2` results into such a tuple
//! while the scalars stay live for the bf16 conversion that follows: the
//! copy cannot be coalesced because both registers are live at once, so
//! each loop iteration pays a `v_mov_b32` per element and holds every
//! probability twice (32 extra VGPRs and 32 extra moves in the d64 persist
//! kernel; the spills of the 256x128 tiles).
//!
//! The two hold the same value, so the register allocator's liveness test
//! is too strict. Here, a tuple whose every dword is written by exactly one
//! copy of a one-dword register that has one definition takes those
//! registers as its halves: they are renamed to the tuple's dwords and the
//! copies go. Only within one block, and only when the sources die about
//! together (within `SPREAD` instructions) so the pair does not hold a
//! register hostage. `ROCKE_DIRECT_DEBUG=nopackco` skips it.

use crate::rir::fx::FxHashMap;
use crate::rir::mir::*;

const SPREAD: usize = 32;

/// How many copies went.
pub fn run(f: &mut MFunc) -> usize {
    if crate::rir::debug_flag("nopackco") {
        return 0;
    }
    let nv = f.vregs.len();
    // Per register: the number of defining instructions, the block and position of the last definition,
    // whether it is touched outside that block, and its last touch in it.
    let mut ndefs = vec![0u32; nv];
    let mut def_at: Vec<(usize, usize)> = vec![(usize::MAX, 0); nv];
    let mut early_def = vec![false; nv];
    let mut copies_into: FxHashMap<u32, Vec<(usize, usize)>> = FxHashMap::default();
    for (b, block) in f.blocks.iter().enumerate() {
        for (k, inst) in block.insts.iter().enumerate() {
            let flags = inst.op.flags();
            for d in inst.defs.iter().filter_map(Opnd::vref) {
                let r = d.reg as usize;
                ndefs[r] += 1;
                def_at[r] = (b, k);
                early_def[r] |= flags & (F_EARLY | F_TIED) != 0;
            }
            if inst.op == Opc::Copy {
                if let Some(Opnd::R(d)) = inst.defs.first() {
                    if d.n == 1 && f.vregs[d.reg as usize].size >= 2 {
                        copies_into.entry(d.reg).or_default().push((b, k));
                    }
                }
            }
        }
    }
    // Last touch within the defining block, or `FAR` for a register that outlives it (and for any other block's).
    const FAR: usize = usize::MAX / 2;
    let mut last = vec![0usize; nv];
    let mut outside = vec![false; nv];
    for (b, block) in f.blocks.iter().enumerate() {
        for (k, inst) in block.insts.iter().enumerate() {
            for o in inst.defs.iter().chain(inst.uses.iter()).filter_map(Opnd::vref) {
                let r = o.reg as usize;
                if def_at[r].0 == b {
                    last[r] = last[r].max(k);
                } else {
                    outside[r] = true;
                }
            }
        }
    }
    for r in 0..nv {
        if outside[r] {
            last[r] = FAR;
        }
    }
    let plain = |f: &MFunc, r: u32| {
        let i = &f.vregs[r as usize];
        i.class == Class::V && i.fixed.is_none()
    };
    let mut rename: FxHashMap<u32, (u32, u8)> = FxHashMap::default();
    let mut gone: Vec<(usize, usize)> = Vec::new();
    let mut tuples: Vec<u32> = copies_into.keys().copied().collect();
    tuples.sort_unstable();
    for t in tuples {
        let copies = &copies_into[&t];
        let size = f.vregs[t as usize].size as usize;
        // Written by one copy per dword and nothing else, within one block.
        if !plain(f, t) || copies.len() != size || ndefs[t as usize] as usize != size {
            continue;
        }
        let b = copies[0].0;
        // All its touches are in its block: the tuple is the pair of one iteration.
        if copies.iter().any(|c| c.0 != b) || outside[t as usize] {
            continue;
        }
        let mut halves: Vec<(u32, u8, usize)> = Vec::new();
        for &(_, k) in copies {
            let inst = &f.blocks[b].insts[k];
            let (Some(Opnd::R(d)), Some(Opnd::R(s))) = (inst.defs.first(), inst.uses.first()) else { break };
            let sr = s.reg as usize;
            let ok = s.off == 0
                && s.n == 1
                && f.vregs[sr].size == 1
                && plain(f, s.reg)
                && ndefs[sr] == 1
                && !early_def[sr]
                && def_at[sr].0 == b
                && !rename.contains_key(&s.reg)
                && s.reg != t
                && !halves.iter().any(|h| h.0 == s.reg);
            if !ok {
                break;
            }
            halves.push((s.reg, d.off, k));
        }
        if halves.len() != size {
            continue;
        }
        let spread = |it: &mut dyn Iterator<Item = usize>| {
            let v: Vec<usize> = it.collect();
            v.iter().max().unwrap() - v.iter().min().unwrap()
        };
        if spread(&mut halves.iter().map(|h| def_at[h.0 as usize].1)) > SPREAD || spread(&mut halves.iter().map(|h| last[h.0 as usize])) > SPREAD {
            continue;
        }
        for (s, off, k) in halves {
            rename.insert(s, (t, off));
            gone.push((b, k));
        }
    }
    if rename.is_empty() {
        return 0;
    }
    let removed = gone.len();
    gone.sort_unstable();
    for block in f.blocks.iter_mut() {
        for inst in block.insts.iter_mut() {
            for o in inst.defs.iter_mut().chain(inst.uses.iter_mut()) {
                if let Opnd::R(v) = o {
                    if let Some(&(t, off)) = rename.get(&v.reg) {
                        *v = VRef { reg: t, off: off + v.off, n: v.n };
                    }
                }
            }
        }
    }
    let mut by_block: FxHashMap<usize, Vec<usize>> = FxHashMap::default();
    for (b, k) in gone {
        by_block.entry(b).or_default().push(k);
    }
    for (b, ks) in by_block {
        let mut i = 0;
        let mut ks = ks.into_iter().peekable();
        f.blocks[b].insts.retain(|_| {
            let drop = ks.peek() == Some(&i);
            if drop {
                ks.next();
            }
            i += 1;
            !drop
        });
    }
    removed
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::target::Target;

    fn r(reg: u32, off: u8) -> Opnd {
        Opnd::R(VRef { reg, off, n: 1 })
    }

    /// v0 (input), v1 = exp(v0), v2 = exp(v0), v3 = pair [v1, v2] by two copies, v4 = v3.0 + v3.1, v5 = cvt(v1, v2).
    fn func(between: usize) -> MFunc {
        let mut f = MFunc { target: Target::Gfx950, ..MFunc::default() };
        for size in [1u8, 1, 1, 2, 1, 1] {
            f.vregs.push(VRegInfo { class: Class::V, size, align: size, fixed: None });
        }
        let mut insts = vec![MInst::new(Opc::VExpF32, [r(1, 0)], [r(0, 0)])];
        for _ in 0..between {
            insts.push(MInst::new(Opc::VAddF32, [r(5, 0)], [r(0, 0), r(0, 0)]));
        }
        insts.push(MInst::new(Opc::VExpF32, [r(2, 0)], [r(0, 0)]));
        insts.extend([
            MInst::new(Opc::Copy, [r(3, 0)], [r(1, 0)]),
            MInst::new(Opc::Copy, [r(3, 1)], [r(2, 0)]),
            MInst::new(Opc::VAddF32, [r(4, 0)], [r(3, 0), r(3, 1)]),
            MInst::new(Opc::VSubF32, [r(5, 0)], [r(1, 0), r(2, 0)]),
        ]);
        f.blocks.push(MBlock { insts, loop_depth: 0 });
        f
    }

    #[test]
    fn a_tuple_of_copies_takes_the_values_as_its_halves() {
        let mut f = func(0);
        assert_eq!(run(&mut f), 2);
        let insts = &f.blocks[0].insts;
        assert!(insts.iter().all(|i| i.op != Opc::Copy));
        // Both exps now write the pair, and the later read of both uses the pair.
        assert_eq!((insts[0].defs[0], insts[1].defs[0]), (r(3, 0), r(3, 1)));
        assert_eq!(insts.last().unwrap().uses.to_vec(), vec![r(3, 0), r(3, 1)]);
    }

    #[test]
    fn values_that_are_defined_far_apart_stay_copies() {
        let mut f = func(SPREAD + 2);
        assert_eq!(run(&mut f), 0);
        assert_eq!(f.blocks[0].insts.iter().filter(|i| i.op == Opc::Copy).count(), 2);
    }

    #[test]
    fn a_value_written_twice_stays_a_copy() {
        let mut f = func(0);
        f.blocks[0].insts.insert(1, MInst::new(Opc::VExpF32, [r(1, 0)], [r(0, 0)]));
        assert_eq!(run(&mut f), 0);
    }
}
