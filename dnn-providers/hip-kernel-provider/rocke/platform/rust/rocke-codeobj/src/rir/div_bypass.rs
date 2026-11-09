// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

//! The 32-bit path of a 64-bit `arith.div` / `arith.mod` by a runtime value
//! on gfx1151: LLVM's slow-division bypass (`BypassSlowDivision`, which AMDGPU
//! enables for 64-bit division). When the high halves of both operands are
//! zero, both are in [0, 2^32) and one unsigned 32-bit division gives the
//! result; otherwise the 64-bit expansion runs as before. As in LLVM, the
//! operands are tested as they come, so a negative one takes the 64-bit path,
//! and the 32-bit path needs no absolute values or sign fix-up.
//!
//! Uniform operands branch on SCC. Divergent ones branch on the whole wave:
//! the 32-bit path when every active lane fits, the 64-bit path for every lane
//! otherwise. With mixed lanes that costs one test more than the 64-bit path
//! alone, where LLVM's EXEC-masked form runs both paths; computing both and
//! selecting per lane would always pay for the 64-bit path. EXEC is not
//! touched.
//!
//! The 64-bit path comes first and the 32-bit path falls through to the join,
//! so the common case takes one branch. `ROCKE_DIRECT_DEBUG=nodivbypass`
//! turns the bypass off.

use super::{r, sub, Isel};
use crate::rir::ast::OpKind;
use crate::rir::mir::{MInst, Opc, Opnd, VRef};
use crate::target::Target;

/// `v_cmp_ne_u32`.
const VCMP_NE_U32: u8 = 0x4d;

/// A bypass between `bypass64_open` and `bypass64_close`.
pub(super) struct Bypass {
    /// The branch to the 32-bit path (block, index), patched by `bypass64_close`.
    branch: (usize, usize),
    /// The low halves: the 32-bit division's operands.
    n: Opnd,
    d: Opnd,
    /// The result halves, written by both paths.
    lo: VRef,
    hi: VRef,
    uniform: bool,
}

impl Isel<'_> {
    /// Tests whether `n` and `d` both fit in 32 bits and branches to the
    /// 32-bit path if so; what is selected next, up to `bypass64_close`, is
    /// the 64-bit path. `None`, with nothing emitted, off gfx1151, with
    /// `nodivbypass`, when a high half is a constant other than zero (it
    /// never fits), or when both are zero (no test to make).
    pub(super) fn bypass64_open(&mut self, n: (Opnd, Opnd), d: (Opnd, Opnd), uniform: bool) -> Option<Bypass> {
        if self.t.target != Target::Gfx1151 || crate::rir::debug_flag("nodivbypass") {
            return None;
        }
        let tested: Vec<Opnd> = [n.1, d.1].into_iter().filter(|&h| h != Opnd::Imm(0)).collect();
        if tested.is_empty() || tested.iter().any(|h| matches!(h, Opnd::Imm(_))) {
            return None;
        }
        if uniform || tested.iter().all(|&h| !self.is_v(h)) {
            if let [a, b] = tested[..] {
                let any = self.sreg(1);
                self.salu(Opc::SOrB32, [r(any)], [a, b]);
            } else {
                self.salu(Opc::SCmpLgU32, [], [tested[0], Opnd::Imm(0)]);
            }
        } else {
            let any = match tested[..] {
                [a, b] => self.int_bin(OpKind::Or, a, b, false, false).expect("or has a lowering"),
                _ => tested[0],
            };
            let lanes = self.mreg();
            self.valu(Opc::VCmp(VCMP_NE_U32), [r(lanes)], [any, Opnd::Imm(0)], 0);
            self.salu(self.lm(Opc::SCmpLgU32), [], [r(lanes), Opnd::Imm(0)]);
        }
        self.emit(MInst::new(Opc::SCbranchScc0, [], []));
        let branch = (self.cur, self.f.blocks[self.cur].insts.len() - 1);
        self.new_block();
        let (lo, hi) = if uniform {
            let pair = self.sreg(2);
            (sub(pair, 0, 1), sub(pair, 1, 1))
        } else {
            (self.vgpr(1), self.vgpr(1))
        };
        Some(Bypass { branch, n: n.0, d: d.0, lo, hi, uniform })
    }

    /// Ends the 64-bit path, whose result is `wide`, then selects the 32-bit
    /// path and the join; returns the result of whichever ran (`wide` itself
    /// without a bypass).
    pub(super) fn bypass64_close(&mut self, bypass: Option<Bypass>, wide: (Opnd, Opnd), rem: bool) -> (Opnd, Opnd) {
        let Some(b) = bypass else {
            return wide;
        };
        self.copy(b.lo, wide.0);
        self.copy(b.hi, wide.1);
        self.emit(MInst::new(Opc::SBranch, [], []));
        let to_join = (self.cur, self.f.blocks[self.cur].insts.len() - 1);
        let short = self.new_block();
        self.f.blocks[b.branch.0].insts[b.branch.1].imm = short as i64;
        let q = self.udiv32(b.n, b.d);
        let lo = if rem {
            let p = self.v_bin(Opc::VMulLoU32, q, b.d);
            self.v_bin(Opc::VSubNcU32, b.n, p)
        } else {
            q
        };
        if b.uniform {
            self.emit(MInst::new(Opc::VReadfirstlaneB32, [r(b.lo)], [lo]));
        } else {
            self.copy(b.lo, lo);
        }
        self.copy(b.hi, Opnd::Imm(0));
        let join = self.new_block();
        self.f.blocks[to_join.0].insts[to_join.1].imm = join as i64;
        (r(b.lo), r(b.hi))
    }
}

#[cfg(test)]
mod tests {
    use crate::rir::mir::{MBlock, MFunc, MInst, Opc, Opnd, F_DEF_EXEC, F_DEF_SCC};
    use crate::target::Target;

    fn compiled(text: &str, target: Target) -> MFunc {
        crate::rir::compile_any_target(text, target).unwrap_or_else(|e| panic!("{e}")).func
    }

    /// `%z = %x op %y` stored per lane. `x` and `y` define `%x` and `%y` from
    /// `SRC`: an i64 loaded per lane (`divergent`) or an i64 kernel argument.
    fn kernel(op: &str, divergent: bool, x: &str, y: &str) -> String {
        let load = |v: &str, p: &str| {
            format!("    %{v}0 : i64 = memref.global_load_typed ( %{p}, %tid ) {{ align = i:8, elem_type = s:\"i64\" }}\n")
        };
        let (xs, ys) = if divergent { (load("x", "X"), load("y", "Y")) } else { (String::new(), String::new()) };
        let (x0, y0) = if divergent { ("%x0", "%y0") } else { ("%a", "%b") };
        format!(
            "ckdsl.ir v1\nkernel @probe {{\n  attrs {{ max_workgroup_size = i:64 }}\n  params {{\n    \
             %X : ptr<i64,global>\n    %Y : ptr<i64,global>\n    %W : ptr<i32,global>\n    %V : ptr<i32,global>\n    \
             %Z : ptr<i64,global>\n    %a : i64\n    %b : i64\n    %i : i32\n  }}\n  \
             region @entry {{\n    %tid : i32 = gpu.thread_id (  ) {{ axis = s:\"x\" }}\n{xs}{ys}{}{}    \
             %z : i64 = arith.{op} ( %x, %y )\n    \
             memref.global_store_typed ( %Z, %tid, %z ) {{ align = i:8, elem_type = s:\"i64\" }}\n    \
             cf.return (  )\n  }}\n}}\n",
            x.replace("SRC", x0),
            y.replace("SRC", y0)
        )
    }

    const AS_IS_X: &str = "    %x : i64 = arith.add ( SRC, SRC )\n";
    const AS_IS_Y: &str = "    %y : i64 = arith.add ( SRC, SRC )\n";
    /// An i32 loaded per lane, zero-extended: a high half known to be zero.
    const ZEXT_X: &str = "    %w : i32 = memref.global_load_typed ( %W, %tid ) { align = i:4, elem_type = s:\"i32\" }\n    \
                          %x : i64 = arith.zext ( %w )\n";

    fn count(f: &MFunc, op: Opc) -> usize {
        f.blocks.iter().flat_map(|b| &b.insts).filter(|i| i.op == op).count()
    }

    /// The blocks that end in the bypass's test, and the last SCC writer of each.
    fn tests(f: &MFunc) -> Vec<(usize, &MInst)> {
        f.blocks
            .iter()
            .enumerate()
            .filter(|(_, b)| b.insts.last().is_some_and(|i| i.op == Opc::SCbranchScc0))
            .map(|(k, b)| (k, b.insts.iter().rev().skip(1).find(|i| i.op.flags() & F_DEF_SCC != 0).expect("an SCC writer")))
            .collect()
    }

    fn has(block: &MBlock, op: Opc) -> bool {
        block.insts.iter().any(|i| i.op == op)
    }

    /// The test branches to the 32-bit path; the 64-bit path falls through
    /// from it and branches over the 32-bit path to the join.
    fn check_diamond(f: &MFunc, at: usize) {
        let test = &f.blocks[at];
        let short = test.insts.last().map(|i| i.imm as usize).expect("a branch");
        let wide = &f.blocks[at + 1];
        assert!(has(wide, Opc::VRcpF32) && !has(wide, Opc::VRcpIflagF32), "the 64-bit path follows the test");
        let over = wide.insts.last().expect("the 64-bit path ends");
        assert_eq!((over.op, over.imm as usize), (Opc::SBranch, short + 1), "the 64-bit path branches to the join");
        assert!(has(&f.blocks[short], Opc::VRcpIflagF32) && !has(&f.blocks[short], Opc::VRcpF32));
        assert_eq!(f.successors(short), [short + 1], "the 32-bit path falls through to the join");
    }

    #[test]
    fn uniform_operands_that_fit_take_one_32_bit_division_behind_a_scalar_branch() {
        for op in ["div", "mod"] {
            let f = compiled(&kernel(op, false, AS_IS_X, AS_IS_Y), Target::Gfx1151);
            let found = tests(&f);
            assert_eq!(found.len(), 1, "{op}");
            assert_eq!(found[0].1.op, Opc::SOrB32, "{op}: both high halves are or-ed into SCC");
            check_diamond(&f, found[0].0);
            // The 32-bit result comes back from the VALU into the 64-bit path's SGPRs.
            let short = f.blocks[found[0].0].insts.last().map(|i| i.imm as usize).expect("a branch");
            assert!(has(&f.blocks[short], Opc::VReadfirstlaneB32), "{op}");
        }
    }

    #[test]
    fn divergent_operands_take_the_32_bit_path_only_when_every_lane_fits() {
        for op in ["div", "mod"] {
            let f = compiled(&kernel(op, true, AS_IS_X, AS_IS_Y), Target::Gfx1151);
            let found = tests(&f);
            assert_eq!(found.len(), 1, "{op}");
            // The lanes with a high half set, as a mask, tested against zero.
            let test = &f.blocks[found[0].0].insts;
            let cmp = test.iter().rposition(|i| i.op == Opc::VCmp(super::VCMP_NE_U32)).expect("a lane test");
            let scmp = test.iter().rposition(|i| i.op == Opc::SCmpLgU32).expect("a wave test");
            assert!(cmp < scmp && test[scmp].uses[0] == test[cmp].defs[0] && test[scmp].uses[1] == Opnd::Imm(0));
            check_diamond(&f, found[0].0);
            // The whole wave takes one path: EXEC is never written.
            let exec = f.blocks.iter().flat_map(|b| &b.insts).filter(|i| i.op.flags() & F_DEF_EXEC != 0).count();
            assert_eq!(exec, 0, "{op}");
        }
    }

    #[test]
    fn a_high_half_known_to_be_zero_is_not_tested() {
        // Divergent divisor: the lane test reads its high half alone.
        let f = compiled(&kernel("div", true, ZEXT_X, AS_IS_Y), Target::Gfx1151);
        let found = tests(&f);
        assert_eq!(found.len(), 1);
        let test = &f.blocks[found[0].0].insts;
        assert!(test.iter().any(|i| i.op == Opc::VCmp(super::VCMP_NE_U32)) && !test.iter().any(|i| i.op == Opc::VOrB32));
        check_diamond(&f, found[0].0);
        // Uniform divisor: a scalar compare of its high half.
        let zext = "    %x : i64 = arith.zext ( %i )\n";
        let f = compiled(&kernel("div", false, zext, AS_IS_Y), Target::Gfx1151);
        let found = tests(&f);
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].1.op, Opc::SCmpLgU32);
        check_diamond(&f, found[0].0);
    }

    #[test]
    fn divisions_the_bypass_leaves_alone() {
        let constant = |k: i64| format!("    %y : i64 = arith.constant (  ) {{ ity = s:\"i64\", value = i:{k} }}\n");
        // Constant divisors: shifts, a magic multiply, or (negative) the 64-bit
        // expansion without a test, since such a high half never fits.
        for (k, rcp) in [(7, 0), (8, 0), (1_000_000_007, 0), (-7, 1)] {
            for divergent in [false, true] {
                let f = compiled(&kernel("div", divergent, AS_IS_X, &constant(k)), Target::Gfx1151);
                assert!(tests(&f).is_empty(), "{k}");
                assert_eq!((count(&f, Opc::VRcpF32), count(&f, Opc::VRcpIflagF32)), (rcp, 0), "{k}");
            }
        }
        // Operands known to fit in 32 bits: the 32-bit division alone.
        let nonneg = "    %w : i32 = memref.global_load_typed ( %W, %tid ) { align = i:4, elem_type = s:\"i32\" }\n    \
                      %zero : i32 = arith.constant (  ) { ity = s:\"i32\", value = i:0 }\n    \
                      %p : i32 = arith.smax ( %w, %zero )\n    %x : i64 = arith.sext ( %p )\n";
        let narrow = "    %v : i32 = memref.global_load_typed ( %V, %tid ) { align = i:4, elem_type = s:\"i32\" }\n    \
                      %one : i32 = arith.constant (  ) { ity = s:\"i32\", value = i:1 }\n    \
                      %u : i32 = arith.smax ( %v, %one )\n    %y : i64 = arith.sext ( %u )\n";
        let f = compiled(&kernel("div", true, nonneg, narrow), Target::Gfx1151);
        assert!(tests(&f).is_empty());
        assert_eq!((count(&f, Opc::VRcpF32), count(&f, Opc::VRcpIflagF32)), (0, 1));
        // Both high halves known to be zero: nothing to test (the 64-bit expansion as before).
        let zext_y = "    %v : i32 = memref.global_load_typed ( %V, %tid ) { align = i:4, elem_type = s:\"i32\" }\n    \
                      %y : i64 = arith.zext ( %v )\n";
        let f = compiled(&kernel("div", true, ZEXT_X, zext_y), Target::Gfx1151);
        assert!(tests(&f).is_empty());
        // gfx950 keeps the 64-bit expansion.
        for divergent in [false, true] {
            let f = compiled(&kernel("mod", divergent, AS_IS_X, AS_IS_Y), Target::Gfx950);
            assert!(tests(&f).is_empty(), "gfx950 {divergent}");
            assert_eq!(count(&f, Opc::VRcpIflagF32), 0);
        }
    }
}
