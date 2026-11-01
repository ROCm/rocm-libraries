// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

//! Wide MFMAs whose C is copied, in their block, from registers that hold
//! zero on entry to it read C as the constant 0 after allocation.
//!
//! `zero_acc::run` reads C as the constant 0 before allocation only for
//! MFMAs of at most four result dwords: a wider D must not overlap A or B,
//! and until the registers are chosen only C tied to D keeps the allocator
//! from that. A wider MFMA whose C is a copy of a zero tuple keeps the copy,
//! and where LICM moved the tuple out of the loop, as it does for each
//! attention tile's S = Q * K^T with 32x32 MFMAs, the copy runs every
//! iteration (16 `v_mov_b32` per chain) and owes the MFMA wait states.
//! `zero_acc::after_allocation` takes only zeros moved in the MFMA's own
//! block. Here the registers that hold zero on entry to each block are
//! found over the control flow (on every path into the block, the last
//! write of the register moves 0, or a register that held 0, into it), and
//! an MFMA whose C was copied in its block from such registers, and that
//! overwrites C, reads 0; the copies go. The registers and the schedule
//! stay as allocated: moving the zeros into C before allocation instead
//! freed the tuple's registers, and the schedules chosen with them waited
//! more in one-wave attention kernels that ran slower.
//! `ROCKE_DIRECT_DEBUG=nozerowide` skips it.

use super::agpr::is_mfma;
use crate::rir::mir::*;

/// One bit per VGPR (AGPRs from 256): those holding zero in every lane.
type Zeros = [u64; 8];

fn get(z: &Zeros, r: u16) -> bool {
    let r = usize::from(r);
    r < 512 && z[r / 64] >> (r % 64) & 1 == 1
}

fn set(z: &mut Zeros, r: u16, on: bool) {
    let r = usize::from(r);
    if r < 512 {
        let bit = 1u64 << (r % 64);
        z[r / 64] = if on { z[r / 64] | bit } else { z[r / 64] & !bit };
    }
}

fn covers(o: &Opnd, r: u16) -> bool {
    matches!(*o, Opnd::P(Phys::V(base), n) if (base..base + u16::from(n)).contains(&r))
}

/// A plain `v_mov_b32` (no modifiers) of one VGPR: its destination and source.
fn plain_move(inst: &MInst) -> Option<(u16, Opnd)> {
    match (inst.op, inst.mods, &inst.defs[..], &inst.uses[..]) {
        (Opc::VMovB32, 0, [Opnd::P(Phys::V(d), 1)], [src]) => Some((*d, *src)),
        _ => None,
    }
}

/// `z` after `inst`: a plain move of 0, or of a register holding 0, makes
/// its destination zero; any other write of a VGPR makes it not.
fn transfer(z: &mut Zeros, inst: &MInst) {
    if let Some((d, src)) = plain_move(inst) {
        let zero = match src {
            Opnd::Imm(0) => true,
            Opnd::P(Phys::V(s), 1) => get(z, s),
            _ => false,
        };
        set(z, d, zero);
        return;
    }
    for o in &inst.defs {
        if let Opnd::P(Phys::V(base), n) = *o {
            for r in base..base + u16::from(n) {
                set(z, r, false);
            }
        }
    }
}

/// The VGPRs holding zero on entry to each block of `f` (allocated): the
/// greatest solution of "zero on every path in", none on entry to the
/// function or to a block without predecessors.
fn entry_zeros(f: &MFunc) -> Vec<Zeros> {
    let n = f.blocks.len();
    let preds = super::hazards::predecessors(f);
    let mut ins = vec![[0u64; 8]; n];
    let mut outs = vec![[u64::MAX; 8]; n];
    loop {
        let mut changed = false;
        for b in 0..n {
            let mut z = [0u64; 8];
            if b > 0 && !preds[b].is_empty() {
                z = [u64::MAX; 8];
                for &p in &preds[b] {
                    for (w, o) in z.iter_mut().zip(&outs[p]) {
                        *w &= o;
                    }
                }
            }
            ins[b] = z;
            for inst in &f.blocks[b].insts {
                transfer(&mut z, inst);
            }
            if z != outs[b] {
                outs[b] = z;
                changed = true;
            }
        }
        if !changed {
            return ins;
        }
    }
}

/// The last instruction of `insts` (those not `gone`) before `at` that writes v`r`.
fn last_write(insts: &[MInst], gone: &[bool], at: usize, r: u16) -> Option<usize> {
    (0..at).rev().find(|&d| !gone[d] && insts[d].defs.iter().any(|o| covers(o, r)))
}

/// Whether `src` holds 0 before instruction `at`: the constant 0, or a VGPR
/// whose last write before `at` moves a value holding 0 then, or that the
/// block does not write before `at` and that holds 0 on entry (`entry`).
fn zero_at(insts: &[MInst], gone: &[bool], at: usize, src: Opnd, entry: &Zeros) -> bool {
    match src {
        Opnd::Imm(0) => true,
        Opnd::P(Phys::V(s), 1) => match last_write(insts, gone, at, s) {
            None => get(entry, s),
            Some(d) => matches!(plain_move(&insts[d]), Some((ds, next)) if ds == s && zero_at(insts, gone, d, next, entry)),
        },
        _ => false,
    }
}

/// The plain move into v`r` that instruction `k` alone reads, of a value
/// holding 0: nothing between them reads v`r`.
fn zero_copy(insts: &[MInst], gone: &[bool], k: usize, r: u16, entry: &Zeros) -> Option<usize> {
    let d = last_write(insts, gone, k, r)?;
    let (dst, src) = plain_move(&insts[d])?;
    let read = (d + 1..k).any(|j| !gone[j] && insts[j].uses.iter().any(|o| covers(o, r)));
    (dst == r && !read && zero_at(insts, gone, d, src, entry)).then_some(d)
}

/// Rewrites `f` (allocated, its pseudos expanded, after
/// `zero_acc::after_allocation`) as the module documentation says; returns
/// how many MFMAs now read C as 0.
pub fn after_allocation(f: &mut MFunc) -> usize {
    if crate::rir::debug_flag("nozerowide")
        || crate::rir::debug_flag("nozeroacc")
        || !f.blocks.iter().flat_map(|b| &b.insts).any(|i| is_mfma(i.op))
    {
        return 0;
    }
    let entry = entry_zeros(f);
    let mut rewritten = 0;
    for (block, entry) in f.blocks.iter_mut().zip(&entry) {
        let insts = &mut block.insts;
        let mut gone = vec![false; insts.len()];
        for k in 0..insts.len() {
            let inst = &insts[k];
            if !is_mfma(inst.op) {
                continue;
            }
            let (Some(&Opnd::P(Phys::V(c), n)), Some(d)) = (inst.uses.get(2), inst.defs.first()) else { continue };
            let regs = c..c + u16::from(n);
            let alone = u32::from(c) + u32::from(n) <= 256
                && regs.clone().all(|r| covers(d, r) && !inst.uses.iter().enumerate().any(|(j, o)| j != 2 && covers(o, r)));
            if !alone {
                continue;
            }
            let copies: Option<Vec<usize>> = regs.map(|r| zero_copy(insts, &gone, k, r, entry)).collect();
            if let Some(copies) = copies {
                insts[k].uses[2] = Opnd::Imm(0);
                for m in copies {
                    gone[m] = true;
                }
                rewritten += 1;
            }
        }
        let mut i = 0;
        insts.retain(|_| {
            i += 1;
            !gone[i - 1]
        });
    }
    rewritten
}

#[cfg(test)]
mod tests {
    use super::is_mfma;
    use crate::rir;
    use crate::rir::mir::*;
    use crate::target::gfx9::gen_isa::Gfx9Op;
    use crate::target::Target;

    /// Two chains of two 32x32x16 MFMAs (16 result dwords) from one zero
    /// tuple, A x A and A x B, in a loop of `%n` trips, and an element of
    /// each stored: LICM moves the tuple before the loop.
    fn kernel() -> String {
        let body = "%kb : i32 = arith.mul ( %k, %c64 )\n    %t : i32 = arith.add ( %kb, %tid )\n    %o : i32 = arith.mul ( %t, %c4 )\n    \
             %x : vec<f32x4> = memref.global_load_vN ( %X, %o ) { align = i:16, elem_type = s:\"f32\", vec = i:4 }\n    \
             %o2 : i32 = arith.add ( %o, %c256 )\n    \
             %y : vec<f32x4> = memref.global_load_vN ( %X, %o2 ) { align = i:16, elem_type = s:\"f32\", vec = i:4 }\n    \
             %a : vec<f16x8> = arith.bitcast ( %x ) { target = s:\"vec<f16x8>\" }\n    \
             %b : vec<f16x8> = arith.bitcast ( %y ) { target = s:\"vec<f16x8>\" }\n    \
             %d0 : vec<f32x16> = tile.mma ( %a, %a, %z ) { op_id = s:\"mfma_f32_32x32x16_f16\" }\n    \
             %d1 : vec<f32x16> = tile.mma ( %a, %a, %d0 ) { op_id = s:\"mfma_f32_32x32x16_f16\" }\n    \
             %e0 : vec<f32x16> = tile.mma ( %a, %b, %z ) { op_id = s:\"mfma_f32_32x32x16_f16\" }\n    \
             %e1 : vec<f32x16> = tile.mma ( %a, %b, %e0 ) { op_id = s:\"mfma_f32_32x32x16_f16\" }\n    \
             %p : f32 = vector.extract ( %d1 ) { index = i:0 }\n    \
             %q : f32 = vector.extract ( %e1 ) { index = i:15 }\n    \
             %s : f32 = arith.fadd ( %p, %q )\n    \
             memref.global_store_typed ( %Y, %t, %s ) { align = i:4, elem_type = s:\"f32\" }\n";
        format!(
            "ckdsl.ir v1\nkernel @zero {{\n  attrs {{ max_workgroup_size = i:64 }}\n  params {{\n    %X : ptr<f32,global>\n    \
             %Y : ptr<f32,global>\n    %n : i32\n  }}\n  region @entry {{\n    %tid : i32 = gpu.thread_id (  ) {{ axis = s:\"x\" }}\n    \
             %c0 : i32 = arith.constant (  ) {{ ity = s:\"i32\", value = i:0 }}\n    \
             %c1 : i32 = arith.constant (  ) {{ ity = s:\"i32\", value = i:1 }}\n    \
             %c4 : i32 = arith.constant (  ) {{ ity = s:\"i32\", value = i:4 }}\n    \
             %c64 : i32 = arith.constant (  ) {{ ity = s:\"i32\", value = i:64 }}\n    \
             %c256 : i32 = arith.constant (  ) {{ ity = s:\"i32\", value = i:256 }}\n    \
             %z : vec<f32x16> = arith.constant_vec (  ) {{ elem = s:\"f32\", fill = f:0.0, vec = i:16 }}\n    \
             %r : i32 = scf.for ( %c0, %n, %c1, %c0 ) {{ iter_args = l:[ {{ name = s:\"%it\", type = s:\"i32\" }} ], iv = s:\"%k\", \
             iv_type = s:\"i32\", num_iter_args = i:1, unroll = b:false }}\n      region @body {{\n    {body}    \
             %next : i32 = arith.add ( %it, %c1 )\n    scf.yield ( %next ) {{ num = i:1 }}\n      }}\n    cf.return (  )\n  }}\n}}\n"
        )
    }

    /// The first and last VGPR of an operand (`v5` or `v[4:19]`).
    fn span(o: &str) -> (u32, u32) {
        let o = o.trim().trim_start_matches('v');
        let o = o.trim_start_matches('[').trim_end_matches(']');
        let mut ends = o.split(':').map(|r| r.parse::<u32>().unwrap_or_else(|_| panic!("a VGPR operand: {o}")));
        let lo = ends.next().expect("a register");
        (lo, ends.next().unwrap_or(lo))
    }

    #[test]
    fn wide_chains_in_a_loop_read_zero_and_copy_nothing_there() {
        let compiled = rir::compile_any_target(&kernel(), Target::Gfx950).unwrap_or_else(|e| panic!("{e}"));
        let listing = compiled.listing();
        let text: Vec<&str> = listing.lines().map(|l| l.split("//").next().unwrap_or(l).trim()).collect();
        let mfmas: Vec<Vec<&str>> =
            text.iter().filter(|l| l.starts_with("v_mfma")).map(|l| l.split_once(' ').expect("operands").1.split(',').map(str::trim).collect()).collect();
        assert_eq!(mfmas.len(), 4, "{listing}");
        // Each chain's first MFMA reads 0, and its D overlaps neither A nor B.
        let first: Vec<&Vec<&str>> = mfmas.iter().filter(|ops| ops[3] == "0").collect();
        assert_eq!(first.len(), 2, "{listing}");
        for ops in first {
            let (d, a, b) = (span(ops[0]), span(ops[1]), span(ops[2]));
            assert!(d.1 < a.0 || a.1 < d.0, "D over A: {ops:?}");
            assert!(d.1 < b.0 || b.1 < d.0, "D over B: {ops:?}");
        }
        // The second accumulates in place.
        for ops in mfmas.iter().filter(|ops| ops[3] != "0") {
            assert_eq!(ops[0], ops[3], "{ops:?}");
        }
        // The loop (from its back edge's target to it) moves no VGPR into another.
        let addressed: Vec<(i64, &str)> = listing
            .lines()
            .filter_map(|l| {
                let (inst, comment) = l.split_once("//")?;
                let address = i64::from_str_radix(comment.trim().split(':').next()?.trim(), 16).ok()?;
                Some((address, inst.trim()))
            })
            .collect();
        let (latch, target) = addressed
            .iter()
            .enumerate()
            .find_map(|(i, (address, inst))| {
                let imm: u16 = inst.strip_prefix("s_cbranch_scc1 imm:")?.trim().parse().ok()?;
                let target = address + 4 + 4 * i64::from(imm as i16);
                (target <= *address).then_some((i, target))
            })
            .unwrap_or_else(|| panic!("a back edge\n{listing}"));
        let head = addressed.iter().position(|(address, _)| *address == target).expect("the loop head");
        for (_, l) in &addressed[head..latch] {
            let copy = l.starts_with("v_mov_b32 v") && l.split_once(',').is_some_and(|(_, s)| s.trim().starts_with('v'));
            assert!(!copy, "{l}\n{listing}");
        }
    }

    fn v(r: u16, n: u8) -> Opnd {
        Opnd::P(Phys::V(r), n)
    }

    fn mov(d: u16, s: Opnd) -> MInst {
        MInst::new(Opc::VMovB32, [v(d, 1)], [s])
    }

    fn mfma(d: u16, a: u16, b: u16, c: Opnd) -> MInst {
        MInst::new(Opc::Gfx9(Gfx9Op::VMfma32x32x16F16), [v(d, 16)], [v(a, 4), v(b, 4), c])
    }

    fn back_edge(to: usize) -> MInst {
        let mut br = MInst::new(Opc::SCbranchScc1, [], []);
        br.imm = to as i64;
        br
    }

    /// Allocated code: a zero tuple v[40:55] moved as 0 before a loop whose
    /// block copies it to v[0:15] and starts a chain there, then `rest`,
    /// then its back edge; and a block after the loop.
    fn looped(rest: Vec<MInst>) -> MFunc {
        let mut f = MFunc { target: Target::Gfx950, ..MFunc::default() };
        f.blocks.push(MBlock { insts: (40..56).map(|r| mov(r, Opnd::Imm(0))).collect(), loop_depth: 0 });
        let mut body: Vec<MInst> = (0..16).map(|r| mov(r, v(r + 40, 1))).collect();
        body.push(mfma(0, 16, 20, v(0, 16)));
        body.extend(rest);
        body.push(back_edge(1));
        f.blocks.push(MBlock { insts: body, loop_depth: 1 });
        f.blocks.push(MBlock { insts: vec![MInst::new(Opc::SEndpgm, [], [])], loop_depth: 0 });
        f
    }

    #[test]
    fn a_chain_copied_from_zeros_before_its_loop_reads_zero() {
        let mut f = looped(vec![]);
        assert_eq!(super::after_allocation(&mut f), 1);
        let insts = &f.blocks[1].insts;
        assert_eq!(insts.len(), 2, "{insts:?}");
        assert!(is_mfma(insts[0].op) && insts[0].uses[2] == Opnd::Imm(0), "{insts:?}");
    }

    #[test]
    fn zeros_a_path_overwrites_or_a_copy_another_reads_stay() {
        // The loop writes a dword of the tuple after the chain: on the back edge it is not zero.
        let mut f = looped(vec![MInst::new(Opc::VAddF32, [v(47, 1)], [v(1, 1), v(2, 1)])]);
        assert_eq!(super::after_allocation(&mut f), 0);
        assert_eq!(f.blocks[1].insts.len(), 19);
        // A copied dword is read before the MFMA.
        let mut f = looped(vec![]);
        f.blocks[1].insts.insert(16, MInst::new(Opc::VAddF32, [v(60, 1)], [v(3, 1), v(3, 1)]));
        assert_eq!(super::after_allocation(&mut f), 0);
        // D elsewhere: the MFMA reads C and writes other registers.
        let mut f = looped(vec![]);
        f.blocks[1].insts[16] = mfma(64, 16, 20, v(0, 16));
        assert_eq!(super::after_allocation(&mut f), 0);
    }
}
