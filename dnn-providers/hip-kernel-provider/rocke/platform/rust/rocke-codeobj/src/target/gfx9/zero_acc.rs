// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

//! MFMAs whose accumulator starts at zero read C as the inline constant 0.
//!
//! Selection writes C into a fresh D and updates D in place
//! (`mfma::select`). Where C is a zero tuple that other MFMAs also start
//! from, as every attention tile's S = Q * K^T and dP = dO * V^T does once
//! LICM shares the constant, the copy cannot coalesce: four `v_mov_b32`
//! per chain in the loop, and the wait states the VALU write owes the MFMA
//! that reads it (M2). COMGR names the constant instead, as here: the MFMA
//! defines D and reads A, B and 0.
//!
//! Only MFMAs of at most four result dwords: LLVM lets those write D over
//! A or B (its `NoDstOverlap` is for results wider than 128 bits), and with
//! C an immediate nothing else keeps the allocator from that overlap. And
//! only in loops, where the copies run every iteration: straight-line code
//! copies once, and there a fully unrolled GEMM (`34c42fcaa529d60d`)
//! scheduled without the copies took 70 VGPRs instead of 36, a wave less.
//!
//! After allocation (`after_allocation`), the rest: an MFMA whose C is VGPRs
//! that `v_mov_b32 v, 0`s of its block wrote for it alone, and that it
//! overwrites, reads 0 and the moves go. The schedule and the registers are
//! chosen by then, so only the moves and the wait states they owe leave: 128
//! in the unrolled grouped GEMM (`e6117daa6b3804d5`), 16 in each of the
//! small block-scale GEMMs.

use super::agpr::is_mfma;
use crate::rir::fx::FxHashMap;
use crate::rir::mir::*;

/// Whether every dword of `c` is a dword its function defines once, by a
/// copy (or move) of the immediate 0; `zero` says that per (register, dword).
fn all_zero(zero: &FxHashMap<(u32, u8), bool>, c: VRef) -> bool {
    (0..c.n).all(|k| zero.get(&(c.reg, c.off + k)).copied().unwrap_or(false))
}

/// (register, dword) of `f` -> defined exactly once, by a copy of 0.
fn zero_dwords(f: &MFunc) -> FxHashMap<(u32, u8), bool> {
    let mut zero = FxHashMap::default();
    for inst in f.blocks.iter().flat_map(|b| &b.insts) {
        let of_zero = matches!(inst.op, Opc::Copy | Opc::VMovB32) && inst.uses.first() == Some(&Opnd::Imm(0));
        for d in &inst.defs {
            let Opnd::R(v) = *d else { continue };
            for k in 0..v.n {
                zero.entry((v.reg, v.off + k)).and_modify(|z| *z = false).or_insert(of_zero && v.n == 1);
            }
        }
    }
    zero
}

/// Rewrites `f` (after selection) as the module documentation says; returns
/// how many MFMAs now read C as 0. `ROCKE_DIRECT_DEBUG=nozeroacc` skips it.
pub fn run(f: &mut MFunc) -> usize {
    if crate::rir::debug_flag("nozeroacc") || !f.blocks.iter().flat_map(|b| &b.insts).any(|i| is_mfma(i.op)) {
        return 0;
    }
    // `zero_dwords` of `f` as it comes, made at the first copy whose zeros
    // decide (before any rewrite, which takes one).
    let mut zero = None;
    let fixed = |f: &MFunc, v: VRef| f.vregs[v.reg as usize].fixed.is_some();
    let mut rewritten = 0;
    let looped: Vec<usize> = (0..f.blocks.len()).filter(|&b| f.blocks[b].loop_depth > 0).collect();
    for b in looped {
        let mut k = 1;
        while k < f.blocks[b].insts.len() {
            let (copy, mfma) = (&f.blocks[b].insts[k - 1], &f.blocks[b].insts[k]);
            let source = match (mfma.defs.first(), mfma.uses.get(2), copy.defs.first(), copy.uses.first()) {
                (Some(&Opnd::R(d)), Some(&Opnd::R(c2)), Some(&Opnd::R(cd)), Some(&Opnd::R(c)))
                    if is_mfma(mfma.op) && copy.op == Opc::Copy && d == c2 && d == cd && d.n <= 4 && c.n == d.n && !fixed(f, d) =>
                {
                    Some(c)
                }
                _ => None,
            };
            let candidate = source.is_some_and(|c| all_zero(zero.get_or_insert_with(|| zero_dwords(f)), c));
            if candidate {
                f.blocks[b].insts[k].uses[2] = Opnd::Imm(0);
                f.blocks[b].insts.remove(k - 1);
                rewritten += 1;
            } else {
                k += 1;
            }
        }
    }
    if rewritten > 0 {
        crate::rir::post::remove_dead_copies(f);
    }
    rewritten
}

fn covers(o: &Opnd, r: u16) -> bool {
    matches!(*o, Opnd::P(Phys::V(base), n) if (base..base + u16::from(n)).contains(&r))
}

/// The plain `v_mov_b32` (no modifiers) that last writes v`r` before
/// instruction `at` of `insts` (those not `gone`), with what it reads.
fn last_move(insts: &[MInst], gone: &[bool], at: usize, r: u16) -> Option<(usize, Opnd)> {
    let d = (0..at).rev().find(|&d| !gone[d] && insts[d].defs.iter().any(|o| covers(o, r)))?;
    let inst = &insts[d];
    let plain = inst.op == Opc::VMovB32 && inst.mods == 0 && inst.defs[..] == [Opnd::P(Phys::V(r), 1)] && inst.uses.len() == 1;
    plain.then(|| (d, inst.uses[0]))
}

/// Whether v`r` holds 0 before instruction `at`: moved there as 0, or from
/// a register that held 0 then.
fn zero_before(insts: &[MInst], gone: &[bool], at: usize, r: u16) -> bool {
    match last_move(insts, gone, at, r) {
        Some((_, Opnd::Imm(0))) => true,
        Some((d, Opnd::P(Phys::V(s), 1))) => zero_before(insts, gone, d, s),
        _ => false,
    }
}

/// The move of 0 into v`r` that instruction `k` alone reads: nothing
/// between them reads v`r`.
fn zero_move(insts: &[MInst], gone: &[bool], k: usize, r: u16) -> Option<usize> {
    let (d, _) = last_move(insts, gone, k, r)?;
    let read = (d + 1..k).any(|j| !gone[j] && insts[j].uses.iter().any(|o| covers(o, r)));
    (!read && zero_before(insts, gone, k, r)).then_some(d)
}

/// Rewrites `f` (allocated, its pseudos expanded) as the module
/// documentation says; returns how many MFMAs now read C as 0. A block is
/// taken again while that frees more: a zero tuple's own chain can read 0
/// once the copies of it to the other chains are gone.
/// `ROCKE_DIRECT_DEBUG=nozeroacc` skips it.
pub fn after_allocation(f: &mut MFunc) -> usize {
    if crate::rir::debug_flag("nozeroacc") {
        return 0;
    }
    let mut rewritten = 0;
    for block in &mut f.blocks {
        let insts = &mut block.insts;
        let mut gone = vec![false; insts.len()];
        loop {
            let before = rewritten;
            for k in 0..insts.len() {
                let inst = &insts[k];
                if gone[k] || !is_mfma(inst.op) {
                    continue;
                }
                let (Some(&Opnd::P(Phys::V(c), n)), Some(d)) = (inst.uses.get(2), inst.defs.first()) else { continue };
                let regs = c..c + u16::from(n);
                let alone = u32::from(c) + u32::from(n) <= 256
                    && regs.clone().all(|r| covers(d, r) && !inst.uses.iter().enumerate().any(|(j, o)| j != 2 && covers(o, r)));
                let moves: Option<Vec<usize>> = if alone { regs.map(|r| zero_move(insts, &gone, k, r)).collect() } else { None };
                if let Some(moves) = moves {
                    insts[k].uses[2] = Opnd::Imm(0);
                    for m in moves {
                        gone[m] = true;
                    }
                    rewritten += 1;
                }
            }
            if rewritten == before {
                break;
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

    /// Two chains of two MFMAs from one zero tuple (A x A and A x B) and the
    /// sum of an element of each stored, in a loop of `%n` trips (`looped`)
    /// or once.
    fn kernel(op_id: &str, ab: &str, dwords: u32, cd: u32, looped: bool) -> String {
        let (align, last) = (4 * dwords, cd - 1);
        let body = format!(
            "%kb : i32 = arith.mul ( %k, %c64 )\n    %t : i32 = arith.add ( %kb, %tid )\n    %o : i32 = arith.mul ( %t, %cn )\n    \
             %x : vec<f32x{dwords}> = memref.global_load_vN ( %X, %o ) {{ align = i:{align}, elem_type = s:\"f32\", vec = i:{dwords} }}\n    \
             %o2 : i32 = arith.add ( %o, %c256 )\n    \
             %y : vec<f32x{dwords}> = memref.global_load_vN ( %X, %o2 ) {{ align = i:{align}, elem_type = s:\"f32\", vec = i:{dwords} }}\n    \
             %a : {ab} = arith.bitcast ( %x ) {{ target = s:\"{ab}\" }}\n    \
             %b : {ab} = arith.bitcast ( %y ) {{ target = s:\"{ab}\" }}\n    \
             %d0 : vec<f32x{cd}> = tile.mma ( %a, %a, %z ) {{ op_id = s:\"{op_id}\" }}\n    \
             %d1 : vec<f32x{cd}> = tile.mma ( %a, %a, %d0 ) {{ op_id = s:\"{op_id}\" }}\n    \
             %e0 : vec<f32x{cd}> = tile.mma ( %a, %b, %z ) {{ op_id = s:\"{op_id}\" }}\n    \
             %e1 : vec<f32x{cd}> = tile.mma ( %a, %b, %e0 ) {{ op_id = s:\"{op_id}\" }}\n    \
             %p : f32 = vector.extract ( %d1 ) {{ index = i:0 }}\n    \
             %q : f32 = vector.extract ( %e1 ) {{ index = i:{last} }}\n    \
             %s : f32 = arith.fadd ( %p, %q )\n    \
             memref.global_store_typed ( %Y, %t, %s ) {{ align = i:4, elem_type = s:\"f32\" }}\n"
        );
        let body = if looped {
            format!(
                "%r : i32 = scf.for ( %c0, %n, %c1, %c0 ) {{ iter_args = l:[ {{ name = s:\"%it\", type = s:\"i32\" }} ], iv = s:\"%k\", \
                 iv_type = s:\"i32\", num_iter_args = i:1, unroll = b:false }}\n      region @body {{\n    {body}    \
                 %next : i32 = arith.add ( %it, %c1 )\n    scf.yield ( %next ) {{ num = i:1 }}\n      }}\n"
            )
        } else {
            format!("%k : i32 = arith.constant (  ) {{ ity = s:\"i32\", value = i:0 }}\n    {body}")
        };
        format!(
            "ckdsl.ir v1\nkernel @zero {{\n  attrs {{ max_workgroup_size = i:64 }}\n  params {{\n    %X : ptr<f32,global>\n    \
             %Y : ptr<f32,global>\n    %n : i32\n  }}\n  region @entry {{\n    %tid : i32 = gpu.thread_id (  ) {{ axis = s:\"x\" }}\n    \
             %c0 : i32 = arith.constant (  ) {{ ity = s:\"i32\", value = i:0 }}\n    \
             %c1 : i32 = arith.constant (  ) {{ ity = s:\"i32\", value = i:1 }}\n    \
             %c64 : i32 = arith.constant (  ) {{ ity = s:\"i32\", value = i:64 }}\n    \
             %c256 : i32 = arith.constant (  ) {{ ity = s:\"i32\", value = i:256 }}\n    \
             %cn : i32 = arith.constant (  ) {{ ity = s:\"i32\", value = i:{dwords} }}\n    \
             %z : vec<f32x{cd}> = arith.constant_vec (  ) {{ elem = s:\"f32\", fill = f:0.0, vec = i:{cd} }}\n    \
             {body}    cf.return (  )\n  }}\n}}\n"
        )
    }

    fn mfma_lines(listing: &str) -> Vec<String> {
        listing.lines().filter(|l| l.contains("v_mfma")).map(|l| l.split("//").next().unwrap_or(l).trim().to_string()).collect()
    }

    fn zeros(lines: &[String]) -> usize {
        lines.iter().filter(|l| l.ends_with(", 0")).count()
    }

    #[test]
    fn a_chain_from_zero_in_a_loop_reads_the_constant() {
        let text = kernel("mfma_f32_16x16x32_f16", "vec<f16x8>", 4, 4, true);
        let compiled = rir::compile_any_target(&text, Target::Gfx950).unwrap_or_else(|e| panic!("{e}"));
        let listing = compiled.listing();
        let lines = mfma_lines(&listing);
        assert_eq!(lines.len(), 4, "{listing}");
        // Each chain's first MFMA reads 0; the second accumulates in place.
        assert_eq!(zeros(&lines), 2, "{listing}");
        for l in lines.iter().filter(|l| !l.ends_with(", 0")) {
            let ops: Vec<&str> = l.split_once(' ').expect("operands").1.split(',').map(str::trim).collect();
            assert_eq!(ops[0], ops[3], "{l}");
        }
        // No zero tuple is materialized for them: a zero moved anywhere is
        // the high half of an address pair (`wide_add`).
        let text: Vec<&str> = listing.lines().map(|l| l.split("//").next().unwrap_or(l).trim()).collect();
        for l in text.iter().filter(|l| l.starts_with("v_mov_b32 v") && l.ends_with(", 0")) {
            let dst: u32 = l["v_mov_b32 v".len()..].split(',').next().and_then(|r| r.parse().ok()).expect("a VGPR");
            let pair = format!("v[{}:{}]", dst.saturating_sub(1), dst);
            assert!(text.iter().any(|t| t.starts_with("v_lshl_add_u64") && t.contains(&pair)), "{l}\n{listing}");
        }
    }

    #[test]
    fn straight_line_chains_read_zero_after_allocation_and_wide_ones_in_loops_keep_the_copy() {
        // Straight-line code keeps the copies before allocation; after it,
        // each chain's first MFMA reads 0 and no zero is moved anywhere.
        let once = kernel("mfma_f32_16x16x32_f16", "vec<f16x8>", 4, 4, false);
        let compiled = rir::compile_any_target(&once, Target::Gfx950).unwrap_or_else(|e| panic!("{e}"));
        let listing = compiled.listing();
        let lines = mfma_lines(&listing);
        assert_eq!((lines.len(), zeros(&lines)), (4, 2), "{lines:?}");
        assert!(!listing.lines().any(|l| l.contains("v_mov_b32") && l.split("//").next().unwrap_or(l).trim_end().ends_with(", 0")), "{listing}");
        // 32x32x16: 16 result dwords, which must not overlap A or B; in the
        // loop their copies read a zero tuple from before it.
        let wide = kernel("mfma_f32_32x32x16_f16", "vec<f16x8>", 4, 16, true);
        let compiled = rir::compile_any_target(&wide, Target::Gfx950).unwrap_or_else(|e| panic!("{e}"));
        let lines = mfma_lines(&compiled.listing());
        assert_eq!((lines.len(), zeros(&lines)), (4, 0), "{lines:?}");
    }

    fn v(r: u16, n: u8) -> Opnd {
        Opnd::P(Phys::V(r), n)
    }

    fn mov(d: u16, s: Opnd) -> MInst {
        MInst::new(Opc::VMovB32, [v(d, 1)], [s])
    }

    fn mfma(d: u16, a: u16, b: u16, c: Opnd) -> MInst {
        MInst::new(Opc::Gfx9(Gfx9Op::VMfma16x16x32F16), [v(d, 4)], [v(a, 4), v(b, 4), c])
    }

    /// One block of allocated code: a zero tuple v[8:11] moved as 0, then
    /// `between`, then `rest`.
    fn allocated(between: Vec<MInst>, rest: Vec<MInst>) -> MFunc {
        let mut f = MFunc { target: Target::Gfx950, ..MFunc::default() };
        let insts = (8..12).map(|r| mov(r, Opnd::Imm(0))).chain(between).chain(rest).collect();
        f.blocks.push(MBlock { insts, loop_depth: 0 });
        f
    }

    #[test]
    fn after_allocation_a_zero_tuple_and_its_copies_go() {
        // The tuple's copy to v[12:15] starts a second chain; each chain
        // overwrites its C.
        let copies = (12..16).map(|r| mov(r, v(r - 4, 1))).collect();
        let mut f = allocated(copies, vec![mfma(8, 0, 4, v(8, 4)), mfma(12, 0, 16, v(12, 4))]);
        assert_eq!(super::after_allocation(&mut f), 2);
        let insts = &f.blocks[0].insts;
        assert_eq!(insts.len(), 2, "{insts:?}");
        assert!(insts.iter().all(|i| is_mfma(i.op) && i.uses[2] == Opnd::Imm(0)), "{insts:?}");
    }

    #[test]
    fn after_allocation_zeros_read_elsewhere_stay() {
        // Read by another instruction before the MFMA.
        let read = MInst::new(Opc::VAndB32, [v(20, 1)], [v(9, 1), v(9, 1)]);
        let mut f = allocated(vec![read], vec![mfma(8, 0, 4, v(8, 4))]);
        assert_eq!(super::after_allocation(&mut f), 0);
        // D elsewhere: C's zeros stay live past the MFMA.
        let mut f = allocated(vec![], vec![mfma(12, 0, 4, v(8, 4))]);
        assert_eq!(super::after_allocation(&mut f), 0);
        // A reads C's registers too.
        let mut f = allocated(vec![], vec![mfma(8, 8, 4, v(8, 4))]);
        assert_eq!(super::after_allocation(&mut f), 0);
        assert_eq!(f.blocks[0].insts.len(), 5);
    }

    /// `run` as it was before its zero map was made on demand, kept verbatim
    /// to compare against.
    fn run_reference(f: &mut MFunc) -> usize {
        if crate::rir::debug_flag("nozeroacc") || !f.blocks.iter().flat_map(|b| &b.insts).any(|i| is_mfma(i.op)) {
            return 0;
        }
        let mut zero: std::collections::HashMap<(u32, u8), bool> = std::collections::HashMap::new();
        for inst in f.blocks.iter().flat_map(|b| &b.insts) {
            let of_zero = matches!(inst.op, Opc::Copy | Opc::VMovB32) && inst.uses.first() == Some(&Opnd::Imm(0));
            for d in &inst.defs {
                let Opnd::R(v) = *d else { continue };
                for k in 0..v.n {
                    zero.entry((v.reg, v.off + k)).and_modify(|z| *z = false).or_insert(of_zero && v.n == 1);
                }
            }
        }
        let all_zero = |c: VRef| (0..c.n).all(|k| zero.get(&(c.reg, c.off + k)).copied().unwrap_or(false));
        let fixed = |f: &MFunc, v: VRef| f.vregs[v.reg as usize].fixed.is_some();
        let mut rewritten = 0;
        let looped: Vec<usize> = (0..f.blocks.len()).filter(|&b| f.blocks[b].loop_depth > 0).collect();
        for b in looped {
            let mut k = 1;
            while k < f.blocks[b].insts.len() {
                let (copy, mfma) = (&f.blocks[b].insts[k - 1], &f.blocks[b].insts[k]);
                let candidate = is_mfma(mfma.op)
                    && copy.op == Opc::Copy
                    && matches!(
                        (mfma.defs.first(), mfma.uses.get(2), copy.defs.first(), copy.uses.first()),
                        (Some(&Opnd::R(d)), Some(&Opnd::R(c2)), Some(&Opnd::R(cd)), Some(&Opnd::R(c)))
                            if d == c2 && d == cd && d.n <= 4 && c.n == d.n && !fixed(f, d) && all_zero(c)
                    );
                if candidate {
                    f.blocks[b].insts[k].uses[2] = Opnd::Imm(0);
                    f.blocks[b].insts.remove(k - 1);
                    rewritten += 1;
                } else {
                    k += 1;
                }
            }
        }
        if rewritten > 0 {
            crate::rir::post::remove_dead_copies(f);
        }
        rewritten
    }

    /// A selected function: one-dword registers some defined by zeros, tuples
    /// built from them by copies, and blocks (some in loops) of MFMAs whose C
    /// is copied into D just before, from a tuple, a reused D or another
    /// value, with other writes of the dwords and fixed registers mixed in.
    fn random_function(seed: u64) -> MFunc {
        let mut s = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
        let mut next = |n: u64| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s % n) as u32
        };
        let mut f = MFunc { target: Target::Gfx950, ..MFunc::default() };
        let reg = |f: &mut MFunc, size: u8, fixed: bool| {
            f.vregs.push(VRegInfo { class: Class::V, size, align: 1, fixed: fixed.then_some(0) });
            f.vregs.len() as u32 - 1
        };
        let r = |reg: u32, off: u8, n: u8| Opnd::R(VRef { reg, off, n });
        let mut entry = Vec::new();
        let other = reg(&mut f, 1, false);
        let tuples: Vec<u32> = (0..4).map(|_| reg(&mut f, 4, false)).collect();
        for &t in &tuples {
            for k in 0..4 {
                let value = if next(12) == 0 { r(other, 0, 1) } else { Opnd::Imm(0) };
                entry.push(MInst::new(if next(2) == 0 { Opc::Copy } else { Opc::VMovB32 }, [r(t, k, 1)], [value]));
                if next(16) == 0 {
                    entry.push(MInst::new(Opc::VAddF32, [r(t, k, 1)], [r(t, k, 1), Opnd::Imm(1)]));
                }
            }
            if next(10) == 0 {
                entry.push(MInst::new(Opc::Copy, [r(t, 0, 4)], [Opnd::Imm(0)]));
            }
        }
        f.blocks.push(MBlock { insts: entry, loop_depth: 0 });
        let ab = reg(&mut f, 4, false);
        for _ in 0..1 + next(4) {
            let mut insts = Vec::new();
            for _ in 0..1 + next(6) {
                let d = reg(&mut f, if next(7) == 0 { 16 } else { 4 }, next(9) == 0);
                let n = f.vregs[d as usize].size;
                let c = match next(4) {
                    0 => r(ab, 0, 4),
                    _ => r(tuples[next(4) as usize], 0, 4),
                };
                let c = if n == 4 { c } else { r(reg(&mut f, n, false), 0, n) };
                insts.push(MInst::new(Opc::Copy, [r(d, 0, n)], [c]));
                if next(10) == 0 {
                    insts.push(MInst::new(Opc::VAddF32, [r(ab, 0, 1)], [r(ab, 0, 1), Opnd::Imm(1)]));
                }
                let op = if n == 4 { Gfx9Op::VMfma16x16x32F16 } else { Gfx9Op::VMfma32x32x16F16 };
                insts.push(MInst::new(Opc::Gfx9(op), [r(d, 0, n)], [r(ab, 0, 4), r(ab, 0, 4), r(d, 0, n)]));
                insts.push(MInst::new(Opc::GStoreB32, [], [r(ab, 0, 2), r(d, 0, 1)]));
            }
            f.blocks.push(MBlock { insts, loop_depth: next(3) });
        }
        f
    }

    #[test]
    fn zero_accumulators_rewrite_what_they_rewrote() {
        let mut rewrote = 0;
        for seed in 0..500 {
            let f = random_function(seed);
            let (mut old, mut new) = (f.clone(), f);
            let n = run_reference(&mut old);
            assert_eq!(super::run(&mut new), n, "seed {seed}");
            assert_eq!(format!("{:?}{:?}", new.blocks, new.vregs), format!("{:?}{:?}", old.blocks, old.vregs), "seed {seed}");
            rewrote += usize::from(n > 0);
        }
        assert!(rewrote > 50, "{rewrote} functions rewritten");
    }
}
