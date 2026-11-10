// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

//! Peepholes on the final instructions, after `s_waitcnt` and before the
//! hazard `s_nop`s (which then count wait states over what is left), each
//! behind a kernel attribute (`MFunc::passes`):
//!
//! - `peep_waits`: adjacent `s_waitcnt`s become one that waits for each
//!   counter's smaller count. Nothing issues between the two, so no counter
//!   can rise while the wave waits at them, and the one wait ends when the
//!   second would have. rocKE's line ring writes `lgkmcnt(0)` and its own
//!   `vmcnt(n)` as two waits before each barrier.
//! - `peep_shifts`: `s_lshl_b32 sX, y, a` then `s_lshl_b32 sX, sX, b` (an
//!   index scaled, then made a byte offset) becomes `s_lshl_b32 sX, y,
//!   a + b` when a + b < 32. The second overwrites the first's result and
//!   SCC, and the shift amounts add.

use crate::rir::mir::*;

fn is_wait(inst: &MInst) -> bool {
    matches!(inst.op, Opc::SWaitcnt | Opc::SWaitcntExplicit)
}

/// gfx9's `s_waitcnt` fields: vmcnt (bits 3:0 and 15:14), expcnt (6:4) and
/// lgkmcnt (11:8).
fn fields(imm: i64) -> (i64, i64, i64) {
    (imm & 0xf | (imm >> 14 & 3) << 4, imm >> 4 & 7, imm >> 8 & 0xf)
}

fn waitcnt(vm: i64, exp: i64, lgkm: i64) -> i64 {
    (vm & 0xf) | exp << 4 | lgkm << 8 | (vm >> 4 & 3) << 14
}

/// The wait for both `a` and `b`.
fn merged(a: i64, b: i64) -> i64 {
    let ((va, ea, la), (vb, eb, lb)) = (fields(a), fields(b));
    waitcnt(va.min(vb), ea.min(eb), la.min(lb))
}

/// `s_lshl_b32 sX, y, a`: sX, y and a.
fn shift(inst: &MInst) -> Option<(Opnd, Opnd, u32)> {
    match (inst.op, &inst.defs[..], &inst.uses[..]) {
        (Opc::SLshlB32, &[d @ Opnd::P(Phys::S(_), 1)], &[y, Opnd::Imm(a)]) if a < 32 => Some((d, y, a)),
        _ => None,
    }
}

/// Runs the peepholes the kernel's attributes ask for; returns how many
/// instructions they removed.
pub fn run(f: &mut MFunc) -> usize {
    let (waits, shifts) = (f.passes.peep_waits, f.passes.peep_shifts);
    if !waits && !shifts {
        return 0;
    }
    let mut removed = 0;
    for block in &mut f.blocks {
        let mut out: Vec<MInst> = Vec::with_capacity(block.insts.len());
        for inst in block.insts.drain(..) {
            if let Some(prev) = out.last_mut() {
                if waits && is_wait(prev) && is_wait(&inst) {
                    prev.imm = merged(prev.imm, inst.imm);
                    if inst.op == Opc::SWaitcntExplicit {
                        prev.op = Opc::SWaitcntExplicit;
                    }
                    removed += 1;
                    continue;
                }
                if let (true, Some((d, y, a)), Some((d2, y2, b))) = (shifts, shift(prev), shift(&inst)) {
                    if d2 == d && y2 == d && a + b < 32 {
                        *prev = MInst::new(Opc::SLshlB32, [d], [y, Opnd::Imm(a + b)]);
                        removed += 1;
                        continue;
                    }
                }
            }
            out.push(inst);
        }
        block.insts = out;
    }
    removed
}

#[cfg(test)]
mod tests {
    use super::super::waitcnt::{explicit_imm, waitcnt_imm};
    use super::*;
    use crate::target::Target;

    fn func(insts: Vec<MInst>, waits: bool, shifts: bool) -> MFunc {
        let mut f = MFunc { target: Target::Gfx950, ..MFunc::default() };
        f.passes.peep_waits = waits;
        f.passes.peep_shifts = shifts;
        f.blocks.push(MBlock { insts, loop_depth: 0 });
        f
    }

    fn wait(op: Opc, imm: i64) -> MInst {
        MInst::new(op, [], []).with_imm(imm)
    }

    fn s(r: u16) -> Opnd {
        Opnd::P(Phys::S(r), 1)
    }

    fn shl(d: u16, y: u16, a: u32) -> MInst {
        MInst::new(Opc::SLshlB32, [s(d)], [s(y), Opnd::Imm(a)])
    }

    #[test]
    fn adjacent_waits_merge_into_the_wait_for_both() {
        // The line ring's barrier: lgkmcnt(0), then vmcnt(15), then the barrier.
        let lgkm0 = explicit_imm(-1, -1, 0);
        let vm15 = explicit_imm(15, -1, -1);
        let barrier = MInst::new(Opc::SBarrierBare, [], []);
        let insts = vec![wait(Opc::SWaitcntExplicit, lgkm0), wait(Opc::SWaitcntExplicit, vm15), barrier.clone()];
        let mut off = func(insts.clone(), false, false);
        assert_eq!(run(&mut off), 0);
        assert_eq!(off.blocks[0].insts.len(), 3);
        let mut on = func(insts, true, false);
        assert_eq!(run(&mut on), 1);
        let out = &on.blocks[0].insts;
        assert_eq!(out.len(), 2);
        assert_eq!((out[0].op, out[0].imm), (Opc::SWaitcntExplicit, explicit_imm(15, -1, 0)));
        assert_eq!(out[1].op, Opc::SBarrierBare);
        // Each counter takes the smaller count (vmcnt past 15 in bits 15:14), a compiler wait too.
        assert_eq!(merged(waitcnt_imm(40, 3), waitcnt_imm(18, 14)), waitcnt_imm(18, 3));
        assert_eq!(merged(explicit_imm(-1, 2, -1), waitcnt_imm(5, u8::MAX)), explicit_imm(5, 2, -1));
        let mut three = func(vec![wait(Opc::SWaitcnt, waitcnt_imm(9, u8::MAX)); 3], true, false);
        assert_eq!(run(&mut three), 2);
        assert_eq!((three.blocks[0].insts.len(), three.blocks[0].insts[0].op), (1, Opc::SWaitcnt));
        // Anything between keeps them apart.
        let nop = MInst::new(Opc::SNop, [], []);
        let mut apart = func(vec![wait(Opc::SWaitcnt, lgkm0), nop, wait(Opc::SWaitcnt, vm15)], true, false);
        assert_eq!(run(&mut apart), 0);
    }

    #[test]
    fn a_shift_of_a_shift_into_the_same_sgpr_becomes_one() {
        let mut f = func(vec![shl(64, 59, 12), shl(64, 64, 2)], false, true);
        assert_eq!(run(&mut f), 1);
        assert_eq!(format!("{:?}", f.blocks[0].insts), format!("{:?}", vec![shl(64, 59, 14)]));
        // Another destination, another source, or 32 bits and more of shift stay.
        for pair in [vec![shl(64, 59, 12), shl(65, 64, 2)], vec![shl(64, 59, 12), shl(64, 63, 2)], vec![shl(64, 59, 20), shl(64, 64, 12)]] {
            let mut f = func(pair.clone(), true, true);
            assert_eq!(run(&mut f), 0);
            assert_eq!(format!("{:?}", f.blocks[0].insts), format!("{pair:?}"));
        }
        let mut off = func(vec![shl(64, 59, 12), shl(64, 64, 2)], true, false);
        assert_eq!(run(&mut off), 0);
    }

    #[test]
    fn a_kernels_two_waits_before_its_barrier_become_one() {
        let text = |attrs: &str| {
            format!(
                "ckdsl.ir v1\nkernel @k {{\n  attrs {{ max_workgroup_size = i:64{attrs} }}\n  params {{\n    %X : ptr<f32,global>\n    \
                 %Y : ptr<f32,global>\n  }}\n  region @entry {{\n    %tid : i32 = gpu.thread_id (  ) {{ axis = s:\"x\" }}\n    \
                 %lds : smem<f32, [64]> = tile.smem_alloc (  )\n    \
                 %x : f32 = memref.global_load_typed ( %X, %tid ) {{ align = i:4, elem_type = s:\"f32\" }}\n    \
                 tile.smem_store_vN_f32 ( %lds, %tid, %x ) {{ elem_type = s:\"f32\", rank = i:1, vec = i:1 }}\n    \
                 %c1 : i32 = arith.constant (  ) {{ ity = s:\"i32\", value = i:1 }}\n    %t1 : i32 = arith.xor ( %tid, %c1 )\n    \
                 %z : f32 = memref.global_load_typed ( %X, %t1 ) {{ align = i:4, elem_type = s:\"f32\" }}\n    \
                 tile.s_waitcnt (  ) {{ expcnt = i:-1, lgkmcnt = i:0, vmcnt = i:-1 }}\n    \
                 tile.s_waitcnt (  ) {{ expcnt = i:-1, lgkmcnt = i:-1, vmcnt = i:0 }}\n    \
                 tile.s_barrier_bare (  )\n    \
                 %y : vec<f32x1> = tile.smem_load_vN_f32 ( %lds, %t1 ) {{ elem_type = s:\"f32\", rank = i:1, vec = i:1 }}\n    \
                 %e : f32 = vector.extract ( %y ) {{ index = i:0 }}\n    %s : f32 = arith.fadd ( %e, %z )\n    \
                 memref.global_store_typed ( %Y, %tid, %s ) {{ align = i:4, elem_type = s:\"f32\" }}\n    cf.return (  )\n  }}\n}}\n"
            )
        };
        let listing = |attrs: &str| crate::rir::compile_any_target(&text(attrs), Target::Gfx950).unwrap_or_else(|e| panic!("{e}")).listing();
        let mnemonics = |l: &str| l.lines().filter_map(|x| x.split_whitespace().next().map(str::to_string)).collect::<Vec<_>>();
        let (off, on) = (listing(""), listing(", peep_waits = b:true"));
        let waits = |l: &str| mnemonics(l).iter().filter(|m| *m == "s_waitcnt").count();
        assert_eq!(waits(&on) + 1, waits(&off), "{off}\n{on}");
        let others = |l: &str| mnemonics(l).into_iter().filter(|m| m != "s_waitcnt").collect::<Vec<_>>();
        assert_eq!(others(&on), others(&off));
    }
}
