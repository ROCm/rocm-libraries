// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

//! DPP combine on gfx950 (lever `dppcombine`, `crate::rir::lever`), as LLVM's
//! `GCNDPPCombine`: a `v_mov_b32_dpp d, s` whose only reader is a later
//! `v_add_u32 x, d` / `v_add_u32 d, x` of the same block becomes
//! `v_add_u32_dpp x, s, x'` with the move's control, and the move goes. Kept
//! to the moves LLVM folds: full row and bank masks (every lane written) and
//! bound_ctrl (a lane whose source lane is out of the row reads 0, the value
//! the move would have written), no EXEC write and no write of `s` or the
//! other operand between the two, the other operand a VGPR (VOP2's vsrc1).
//! Runs on virtual registers, before allocation.

use crate::rir::mir::*;

use super::gen_isa::Gfx9Op;

/// Folds every eligible DPP move into the add that reads it; returns how many.
pub fn combine(f: &mut MFunc) -> usize {
    let mut uses = vec![0u32; f.vregs.len()];
    for inst in f.blocks.iter().flat_map(|b| &b.insts) {
        for o in &inst.uses {
            if let Opnd::R(v) = o {
                uses[v.reg as usize] += 1;
            }
        }
    }
    let class = |f: &MFunc, o: &Opnd| match o {
        Opnd::R(v) => Some(f.vregs[v.reg as usize].class),
        _ => None,
    };
    let mut folded = 0;
    for b in 0..f.blocks.len() {
        let mut i = 0;
        while i < f.blocks[b].insts.len() {
            let mv = f.blocks[b].insts[i].clone();
            let full = mv.imm >> 16 & 0xff == 0xff && mv.imm >> 11 & 1 == 1;
            let (Some(Opnd::R(d)), Some(&src)) = (mv.defs.first().copied(), mv.uses.first()) else {
                i += 1;
                continue;
            };
            if mv.op != Opc::VMovB32Dpp || !f.lever(&mv, "dppcombine") || !full || mv.mods != 0 || d.n != 1 || uses[d.reg as usize] != 1 || class(f, &src) != Some(Class::V) {
                i += 1;
                continue;
            }
            // The reader: the next instruction naming `d`; nothing between writes EXEC, `src`, or `d`.
            let insts = &f.blocks[b].insts;
            let mut at = None;
            for (k, x) in insts.iter().enumerate().skip(i + 1) {
                let writes = |o: &Opnd| x.defs.iter().any(|w| overlaps(w, o));
                if x.uses.iter().any(|u| overlaps(u, &Opnd::R(d))) {
                    at = Some(k);
                    break;
                }
                if x.op.flags() & F_DEF_EXEC != 0 || x.defs.iter().any(|w| matches!(w, Opnd::P(Phys::ExecLo, _))) || writes(&src) || writes(&Opnd::R(d)) {
                    break;
                }
            }
            let Some(k) = at else {
                i += 1;
                continue;
            };
            let add = &insts[k];
            let Some(fused_op) = dpp_form(add.op, f.lever(add, "dppcombinef")) else {
                i += 1;
                continue;
            };
            let other = match add.uses[..] {
                [a, o] if a == Opnd::R(d) => o,
                [o, a] if a == Opnd::R(d) => o,
                _ => {
                    i += 1;
                    continue;
                }
            };
            // `other` must not change between the move and the add (it is read at the add either way), and
            // must be a VGPR: VOP2's vsrc1.
            if add.mods != 0 || class(f, &other) != Some(Class::V) || insts[i + 1..k].iter().any(|x| x.defs.iter().any(|w| overlaps(w, &other))) {
                i += 1;
                continue;
            }
            let fused = MInst { op: fused_op, defs: add.defs, uses: Uses::from([src, other]), imm: mv.imm, mods: 0, mem: add.mem, scope: add.scope };
            f.blocks[b].insts[k] = fused;
            f.blocks[b].insts.remove(i);
            folded += 1;
        }
    }
    folded
}

/// The DPP form a move folds into: `v_add_u32` always; with lever
/// `dppcombinef` (the rust track) also the commutative f32 add, multiply and
/// max and the u32 max, as `GCNDPPCombine` folds them (the B3 and MoE wave
/// sums and scans).
fn dpp_form(op: Opc, floats: bool) -> Option<Opc> {
    let g = |x: Gfx9Op| Some(Opc::Gfx9(x));
    match op {
        Opc::VAddNcU32 => g(Gfx9Op::VAddU32Dpp),
        _ if !floats => None,
        Opc::VAddF32 => g(Gfx9Op::VAddF32Dpp),
        Opc::VMulF32 => g(Gfx9Op::VMulF32Dpp),
        Opc::VMaxF32 => g(Gfx9Op::VMaxF32Dpp),
        Opc::VMaxU32 => g(Gfx9Op::VMaxU32Dpp),
        _ => None,
    }
}

pub(crate) fn overlaps(a: &Opnd, b: &Opnd) -> bool {
    match (a, b) {
        (Opnd::R(x), Opnd::R(y)) => x.reg == y.reg && x.off < y.off + y.n && y.off < x.off + x.n,
        (Opnd::P(p, n), Opnd::P(q, m)) => match (p, q) {
            (Phys::V(x), Phys::V(y)) | (Phys::S(x), Phys::S(y)) => *x < *y + *m as u16 && *y < *x + *n as u16,
            _ => p == q,
        },
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use crate::rir;
    use crate::target::Target;

    /// The toy chain's DPP wave sum (`toy_chain._wave_sum`): xor 1, 2, 7, 15.
    fn wave_sum() -> String {
        let mut body = String::from("    %s0 : i32 = arith.add ( %tid, %n )\n");
        for (k, m) in [1, 2, 7, 15].iter().enumerate() {
            body += &format!("    %x{k} : i32 = tile.dpp_xor ( %s{k} ) {{ xor_mask = i:{m} }}\n    %s{} : i32 = arith.add ( %s{k}, %x{k} )\n", k + 1);
        }
        format!(
            "ckdsl.ir v1\nkernel @dpp {{\n  attrs {{ max_workgroup_size = i:64 }}\n  params {{\n    %Y : ptr<i32,global>\n    %n : i32\n  }}\n  \
             region @entry {{\n    %tid : i32 = gpu.thread_id (  ) {{ axis = s:\"x\" }}\n{body}    \
             memref.global_store_typed ( %Y, %tid, %s4 ) {{ align = i:4, elem_type = s:\"i32\" }}\n    cf.return (  )\n  }}\n}}\n"
        )
    }

    #[test]
    fn mirror_masks_take_dpp_and_moves_fold_into_adds() {
        use super::super::isel_hooks::{xor_shuffle_with, XorShuffle};
        assert!(matches!(xor_shuffle_with(7, false), XorShuffle::Swizzle(_)));
        assert!(matches!(xor_shuffle_with(7, true), XorShuffle::QuadPerm(0x141)));
        assert!(matches!(xor_shuffle_with(8, true), XorShuffle::QuadPerm(0x128)));
        assert!(matches!(xor_shuffle_with(15, true), XorShuffle::QuadPerm(0x140)));
        assert!(matches!(xor_shuffle_with(4, true), XorShuffle::Swizzle(_)));
        // Folding: the selected moves of the wave sum, before and after the pass.
        let mut kernel = rir::parse::parse(&wave_sum()).unwrap();
        kernel.wave = 64;
        let (mut lds, lds_size) = rir::analysis::lds_layout(&kernel);
        let facts = rir::opt::optimize(&mut kernel, &mut lds).unwrap();
        let uniform = rir::analysis::classify(&kernel);
        let mut f = rir::isel::select_for(Target::Gfx950, &kernel, &facts, uniform, &lds, lds_size).unwrap().func;
        let count = |f: &crate::rir::mir::MFunc, op: crate::rir::mir::Opc| f.blocks.iter().flat_map(|b| &b.insts).filter(|i| i.op == op).count();
        let moves = count(&f, crate::rir::mir::Opc::VMovB32Dpp);
        assert_eq!(moves, 2, "xor 1 and 2 (7 and 15 are swizzles without dppmirror)");
        assert_eq!(rir::with_levers(&["dppcombine"], || super::combine(&mut f)), 2);
        assert_eq!(count(&f, crate::rir::mir::Opc::VMovB32Dpp), 0);
        assert_eq!(count(&f, crate::rir::mir::Opc::Gfx9(super::Gfx9Op::VAddU32Dpp)), 2);
    }

    /// A block: `v_mov_b32_dpp d, y` (row_shr:1, full masks, bound_ctrl) then `op r, x, d`.
    fn moved(op: crate::rir::mir::Opc) -> crate::rir::mir::MFunc {
        use crate::rir::mir::*;
        let mut f = MFunc { target: Target::Gfx950, ..MFunc::default() };
        for _ in 0..4 {
            f.vregs.push(VRegInfo { class: Class::V, size: 1, align: 1, fixed: None });
        }
        let r = |k: u32| Opnd::R(VRef { reg: k, off: 0, n: 1 });
        f.blocks = vec![MBlock {
            insts: vec![MInst::new(Opc::VMovB32Dpp, [r(2)], [r(1)]).with_imm(0xff_0911), MInst::new(op, [r(3)], [r(0), r(2)])],
            loop_depth: 0,
        }];
        f
    }

    #[test]
    fn a_folded_move_reads_its_source_in_src0() {
        use crate::rir::mir::{Opc, Opnd, VRef};
        // x + dpp(y): src0 (the DPP operand) is y, src1 is x; distinct operands, so an operand swap shows.
        let mut f = moved(Opc::VAddNcU32);
        assert_eq!(rir::with_levers(&["dppcombine"], || super::combine(&mut f)), 1);
        let add = &f.blocks[0].insts[0];
        let r = |k: u32| Opnd::R(VRef { reg: k, off: 0, n: 1 });
        assert_eq!((add.op, add.uses[0], add.uses[1], add.imm), (Opc::Gfx9(super::Gfx9Op::VAddU32Dpp), r(1), r(0), 0xff_0911));
    }

    #[test]
    fn dppcombinef_folds_each_op_into_its_own_dpp_form() {
        use crate::rir::mir::Opc;
        use super::Gfx9Op as G;
        for (op, want) in [(Opc::VAddF32, G::VAddF32Dpp), (Opc::VMulF32, G::VMulF32Dpp), (Opc::VMaxF32, G::VMaxF32Dpp), (Opc::VMaxU32, G::VMaxU32Dpp)] {
            // Without the lever only the u32 add folds.
            let mut f = moved(op);
            assert_eq!(rir::with_levers(&["dppcombine"], || super::combine(&mut f)), 0, "{op:?}");
            let mut f = moved(op);
            assert_eq!(rir::with_levers(&["dppcombine", "dppcombinef"], || super::combine(&mut f)), 1, "{op:?}");
            assert_eq!(f.blocks[0].insts[0].op, Opc::Gfx9(want), "{op:?}");
        }
    }

    #[test]
    fn dpp_adds_encode_as_llvm_mc_does() {
        use crate::rir::mir::{MInst, Opc, Opnd, Phys};
        let v = |r: u16| Opnd::P(Phys::V(r), 1);
        // llvm-mc -mcpu=gfx950: v_add_u32_dpp v1, v1, v1 row_half_mirror row_mask:0xf bank_mask:0xf bound_ctrl:1
        let inst = MInst::new(Opc::Gfx9(super::Gfx9Op::VAddU32Dpp), [v(1)], [v(1), v(1)]).with_imm(0xff_0941);
        assert_eq!(super::super::asm::encode_words(&inst).unwrap(), vec![0x6802_02fa, 0xff09_4101]);
        let quad = MInst::new(Opc::Gfx9(super::Gfx9Op::VAddU32Dpp), [v(16)], [v(16), v(16)]).with_imm(0xff_08b1);
        assert_eq!(super::super::asm::encode_words(&quad).unwrap(), vec![0x6820_20fa, 0xff08_b110]);
        // gfx10's row_xmask is no gfx9 control.
        let bad = MInst::new(Opc::Gfx9(super::Gfx9Op::VAddU32Dpp), [v(1)], [v(1), v(1)]).with_imm(0xff_0961);
        assert!(super::super::asm::encode_words(&bad).is_err());
    }
}
