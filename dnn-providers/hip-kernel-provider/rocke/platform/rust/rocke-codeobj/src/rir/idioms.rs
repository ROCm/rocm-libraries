//! Integer idioms the source spells out bit by bit and the hardware has one
//! instruction for (W15 climb r0 fix: the direct compiler's `route` loss,
//! PORT-K2-moe §5). LLVM's InstCombine recognizes them (`recognizePopCount`);
//! without that, a SWAR population count is 12 dependent integer ops where
//! COMGR emits one: on `route`'s serial top-k chain six of them (a 64-bit
//! count-trailing-zeros a pick) were ~190 extra SALU instructions.
//!
//! - **population count** of an i32 `x`, the classic SWAR form:
//!   ```text
//!   x1 = x - ((x >> 1) & 0x55555555)
//!   x2 = (x1 & 0x33333333) + ((x1 >> 2) & 0x33333333)
//!   x3 = (x2 + (x2 >> 4)) & 0x0F0F0F0F
//!   r  = (x3 * 0x01010101) >> 24
//!   ```
//!   (`add`, `and` and `mul` operands in either order) becomes
//!   `r = tile.bcnt(x)` (`v_bcnt_u32_b32`). Exact for every i32: both are
//!   the number of set bits. The intermediate values die in `dce`.

use super::ast::*;
use super::fx::FxHashMap;

/// What a value is: its defining op's kind and operands, or a constant.
#[derive(Clone)]
enum Def {
    Op(OpKind, Vec<ValueId>),
    Const(i64),
}

fn collect(region: &Region, defs: &mut FxHashMap<ValueId, Def>) {
    for op in &region.ops {
        if op.results.len() == 1 {
            let d = if op.kind == OpKind::Constant {
                op.int_attr("value").map(Def::Const)
            } else {
                Some(Def::Op(op.kind, op.operands.iter().copied().collect()))
            };
            if let Some(d) = d {
                defs.insert(op.results[0], d);
            }
        }
        for r in &op.regions {
            collect(r, defs);
        }
    }
}

struct M<'a> {
    defs: &'a FxHashMap<ValueId, Def>,
}

impl M<'_> {
    fn op(&self, v: ValueId, kind: OpKind) -> Option<&[ValueId]> {
        match self.defs.get(&v)? {
            Def::Op(k, ops) if *k == kind && ops.len() == 2 => Some(&ops[..]),
            _ => None,
        }
    }

    fn konst(&self, v: ValueId, c: i64) -> bool {
        matches!(self.defs.get(&v), Some(Def::Const(x)) if (*x as u32) == (c as u32))
    }

    /// `kind(a, const c)` in either operand order: `a`.
    fn with_const(&self, v: ValueId, kind: OpKind, c: i64) -> Option<ValueId> {
        let ops = self.op(v, kind)?;
        if self.konst(ops[1], c) {
            Some(ops[0])
        } else if self.konst(ops[0], c) && kind != OpKind::Lshr && kind != OpKind::Sub {
            Some(ops[1])
        } else {
            None
        }
    }

    /// `(a >> s) & m` -> `a`.
    fn shr_and(&self, v: ValueId, s: i64, m: i64) -> Option<ValueId> {
        let t = self.with_const(v, OpKind::And, m)?;
        self.with_const(t, OpKind::Lshr, s)
    }

    /// The SWAR population count `r` of the module doc: its input `x`.
    fn popcount(&self, r: ValueId) -> Option<ValueId> {
        let t8 = self.with_const(r, OpKind::Lshr, 24)?;
        let x3 = self.with_const(t8, OpKind::Mul, 0x0101_0101)?;
        let t7 = self.with_const(x3, OpKind::And, 0x0F0F_0F0F)?;
        let add = self.op(t7, OpKind::Add)?;
        // x2 + (x2 >> 4)
        let x2 = [(add[0], add[1]), (add[1], add[0])]
            .into_iter()
            .find(|&(a, b)| self.with_const(b, OpKind::Lshr, 4) == Some(a))?
            .0;
        let add2 = self.op(x2, OpKind::Add)?;
        // (x1 & 0x33333333) + ((x1 >> 2) & 0x33333333)
        let x1 = [(add2[0], add2[1]), (add2[1], add2[0])].into_iter().find_map(|(a, b)| {
            let x1 = self.with_const(a, OpKind::And, 0x3333_3333)?;
            (self.shr_and(b, 2, 0x3333_3333) == Some(x1)).then_some(x1)
        })?;
        // x - ((x >> 1) & 0x55555555)
        let sub = self.op(x1, OpKind::Sub)?;
        (self.shr_and(sub[1], 1, 0x5555_5555) == Some(sub[0])).then_some(sub[0])
    }
}

fn rewrite(region: &mut Region, m: &M, kernel_ty: &dyn Fn(ValueId) -> bool, n: &mut u32) {
    for op in region.ops.iter_mut() {
        for r in op.regions.iter_mut() {
            rewrite(r, m, kernel_ty, n);
        }
        if op.kind != OpKind::Lshr || op.results.len() != 1 || !kernel_ty(op.results[0]) {
            continue;
        }
        if let Some(x) = m.popcount(op.results[0]) {
            if kernel_ty(x) {
                op.kind = OpKind::Bcnt;
                op.operands = [x].into();
                op.attrs.clear();
                *n += 1;
            }
        }
    }
}

/// Rewrites every SWAR population count of `kernel` to `tile.bcnt`;
/// returns how many. Wave-64 (gfx9) targets only: gfx1151 has no `tile.bcnt`
/// lowering here.
pub fn popcount(kernel: &mut Kernel) -> u32 {
    if kernel.wave != 64 {
        return 0;
    }
    let mut defs = FxHashMap::default();
    collect(&kernel.body, &mut defs);
    let m = M { defs: &defs };
    let tys: Vec<bool> = kernel.values.iter().map(|v| v.ty == Type::Scalar(Scalar::I32)).collect();
    let is_i32 = |v: ValueId| tys.get(v.0 as usize).copied().unwrap_or(false);
    let mut n = 0;
    let mut body = std::mem::take(&mut kernel.body);
    rewrite(&mut body, &m, &is_i32, &mut n);
    kernel.body = body;
    n
}

#[cfg(test)]
mod tests {
    use super::*;

    const SWAR: &str = r#"ckdsl.ir v1
kernel @popc {
  attrs { max_workgroup_size = i:64 }
  params {
    %X : ptr<i32,global>
    %Y : ptr<i32,global>
  }
  region @entry {
    %tid : i32 = gpu.thread_id (  ) { axis = s:"x" }
    %x : i32 = memref.global_load_typed ( %X, %tid ) { align = i:4, elem_type = s:"i32" }
    %c1 : i32 = arith.constant (  ) { ity = s:"i32", value = i:1 }
    %c2 : i32 = arith.constant (  ) { ity = s:"i32", value = i:2 }
    %c4 : i32 = arith.constant (  ) { ity = s:"i32", value = i:4 }
    %c24 : i32 = arith.constant (  ) { ity = s:"i32", value = i:24 }
    %m55 : i32 = arith.constant (  ) { ity = s:"i32", value = i:1431655765 }
    %m33 : i32 = arith.constant (  ) { ity = s:"i32", value = i:858993459 }
    %m0f : i32 = arith.constant (  ) { ity = s:"i32", value = i:252645135 }
    %m01 : i32 = arith.constant (  ) { ity = s:"i32", value = i:16843009 }
    %t1 : i32 = arith.lshr ( %x, %c1 )
    %t2 : i32 = arith.and ( %t1, %m55 )
    %x1 : i32 = arith.sub ( %x, %t2 )
    %t3 : i32 = arith.and ( %x1, %m33 )
    %t4 : i32 = arith.lshr ( %x1, %c2 )
    %t5 : i32 = arith.and ( %t4, %m33 )
    %x2 : i32 = arith.add ( %t3, %t5 )
    %t6 : i32 = arith.lshr ( %x2, %c4 )
    %t7 : i32 = arith.add ( %x2, %t6 )
    %x3 : i32 = arith.and ( %t7, %m0f )
    %t8 : i32 = arith.mul ( %x3, %m01 )
    %r : i32 = arith.lshr ( %t8, %c24 )
    memref.global_store_typed ( %Y, %tid, %r ) { align = i:4, elem_type = s:"i32" }
    cf.return (  )
  }
}
"#;

    fn count(region: &Region, kind: OpKind) -> usize {
        region.ops.iter().map(|op| (op.kind == kind) as usize + op.regions.iter().map(|r| count(r, kind)).sum::<usize>()).sum()
    }

    #[test]
    fn swar_popcount_becomes_bcnt() {
        let mut k = super::super::parse::parse(SWAR).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(popcount(&mut k), 0, "the parser's wave 32 (gfx1151): untouched");
        k.wave = 64;
        assert_eq!(popcount(&mut k), 1);
        assert_eq!(count(&k.body, OpKind::Bcnt), 1);
    }

    #[test]
    fn a_wrong_mask_is_not_a_popcount() {
        let text = SWAR.replace("value = i:858993459", "value = i:858993458");
        let mut k = super::super::parse::parse(&text).unwrap_or_else(|e| panic!("{e}"));
        k.wave = 64;
        assert_eq!(popcount(&mut k), 0);
    }

    #[test]
    fn a_wrong_shift_is_not_a_popcount() {
        let text = SWAR.replace("%r : i32 = arith.lshr ( %t8, %c24 )", "%r : i32 = arith.lshr ( %t8, %c4 )");
        let mut k = super::super::parse::parse(&text).unwrap_or_else(|e| panic!("{e}"));
        k.wave = 64;
        assert_eq!(popcount(&mut k), 0);
    }

    #[test]
    fn the_compiled_kernel_has_one_bcnt_and_no_magic_masks() {
        let compiled = super::super::compile_for(SWAR, crate::target::Target::Gfx950).unwrap_or_else(|e| panic!("{e}"));
        let text = format!("{:?}", compiled.func.blocks);
        assert!(text.contains("VBcntU32B32"), "no v_bcnt in {text}");
        assert!(!text.contains("858993459"), "the SWAR masks survived");
    }
}
