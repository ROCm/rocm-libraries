// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

//! gfx950's choice between a compile with loops unrolled automatically and
//! the compile that leaves them rolled. The shared rule keeps the unrolled
//! code only when it spills no more and keeps the occupancy. A grouped
//! convolution whose nests of three-trip loops each issue a weight load, an
//! LDS read and an MFMA per trip then stays rolled at eight waves, waiting for
//! every load before its MFMA, where its unrolled code (the weights repeated
//! across the nests loaded once) needs more VGPRs and runs fewer waves, as
//! COMGR's does. Here, when the unrolled compile spills no more but loses
//! waves, each compile is weighed at the waves it has, by `sched::cost` with
//! MFMAs and `post::schedule_cost` without, per block, each block weighted by
//! the trips of the loops around it (a constant trip count read from the loop
//! counter, `UNKNOWN_TRIPS` for any other loop); the unrolled one is kept when
//! it costs a fifth less. The automatic unrolling (`AUTO`) also takes a loop
//! whose only op with effects that reads a carried value is an empty register
//! pin (`tile.inline_asm` without a template), which waits for nothing.
//! `ROCKE_DIRECT_DEBUG=nounrollcost` keeps the shared rule and policy.

use super::agpr::is_mfma;
use super::sched;
use crate::rir::mir::*;
use crate::rir::post;
use crate::rir::unroll::Auto;
use crate::rir::CompileStats;

/// The automatic unrolling of gfx950: `Auto::DEFAULT`, with empty register
/// pins not counted as waits on a carried value.
pub const AUTO: Auto = Auto { pins_wait: false, ..Auto::DEFAULT };

/// The automatic unrolling a gfx950 compile tries.
pub fn auto() -> Auto {
    if crate::rir::debug_flag("nounrollcost") {
        Auto::DEFAULT
    } else {
        AUTO
    }
}

/// Trips weighed for a loop whose trip count is not a constant (the shared
/// models weigh a loop level by 8).
const UNKNOWN_TRIPS: u64 = 8;

/// Whether `unrolled` (the compile with loops unrolled automatically, `u` its
/// statistics) is kept over `rolled` (`r`): it spills no more VGPR dwords and
/// keeps the waves, as the shared rule has it, or it costs a fifth less at the
/// waves it has than `rolled` at its own.
pub fn keeps(unrolled: &MFunc, u: &CompileStats, rolled: &MFunc, r: &CompileStats) -> bool {
    let spilled = |s: &CompileStats| s.spill.lds_dwords + s.spill.scratch_dwords;
    if spilled(u) > spilled(r) {
        return false;
    }
    if u.waves >= r.waves {
        return true;
    }
    !crate::rir::debug_flag("nounrollcost") && cost(unrolled, u.waves) * 5 <= cost(rolled, r.waves) * 4
}

/// Slots per wave of `f` (allocated) at `waves` waves per SIMD: each block's
/// cost model cycles there, weighted by the block's runs (`runs`).
pub fn cost(f: &MFunc, waves: u32) -> u64 {
    let mfma = f.blocks.iter().flat_map(|b| &b.insts).any(|i| is_mfma(i.op));
    let mut one = MFunc { target: f.target, ..MFunc::default() };
    runs(f)
        .into_iter()
        .zip(&f.blocks)
        .map(|(runs, block)| {
            one.blocks = vec![MBlock { insts: block.insts.clone(), loop_depth: 0 }];
            let cycles = if mfma { sched::cost(&one).cycles(waves) } else { post::schedule_cost(&one).cycles(waves) };
            runs.saturating_mul(cycles)
        })
        .fold(0u64, u64::saturating_add)
}

/// How many times each block of `f` runs per run of the function: the product
/// of the trips of the loops around it. A loop is a branch back to an earlier
/// block; its trips are known when it ends as a uniform `scf.for` does, adding
/// a constant step to an SGPR counter set to a constant before the loop and
/// comparing it with a constant bound.
pub fn runs(f: &MFunc) -> Vec<u64> {
    let mut runs = vec![1u64; f.blocks.len()];
    for (latch, block) in f.blocks.iter().enumerate() {
        let Some(last) = block.insts.last() else { continue };
        let back = matches!(last.op, Opc::SCbranchScc1 | Opc::SCbranchScc0 | Opc::SCbranchExecnz | Opc::SCbranchExecz | Opc::SBranch);
        let Ok(head) = usize::try_from(last.imm) else { continue };
        if !back || head > latch {
            continue;
        }
        let trips = if last.op == Opc::SCbranchScc1 { trips(f, head, latch) } else { None };
        for r in &mut runs[head..=latch] {
            *r = r.saturating_mul(trips.unwrap_or(UNKNOWN_TRIPS).max(1));
        }
    }
    runs
}

/// The trips of the loop from block `head` to block `latch`, which ends with
/// `s_cmp_lt_i32 counter, bound` and `s_cbranch_scc1 head`.
fn trips(f: &MFunc, head: usize, latch: usize) -> Option<u64> {
    let insts = &f.blocks[latch].insts;
    let cmp = insts.iter().rev().find(|i| i.op == Opc::SCmpLtI32)?;
    let counter = *cmp.uses.first()?;
    if !matches!(counter, Opnd::P(Phys::S(_), 1)) {
        return None;
    }
    let step = insts.iter().find_map(|i| match (i.op, i.defs.first(), &i.uses[..]) {
        (Opc::SAddI32, Some(&d), [a, Opnd::Imm(s)]) if d == counter && *a == counter => Some(i64::from(*s as i32)),
        _ => None,
    })?;
    let bound = match *cmp.uses.get(1)? {
        Opnd::Imm(b) => i64::from(b as i32),
        reg => constant_before(f, head, reg)?,
    };
    let first = constant_before(f, head, counter)?;
    if step <= 0 || bound <= first {
        return (step > 0).then_some(1);
    }
    u64::try_from((bound - first + step - 1) / step).ok()
}

/// The constant the last write of `reg` before block `head` (in block order)
/// moves into it, if that write is an `s_mov_b32` of a constant.
fn constant_before(f: &MFunc, head: usize, reg: Opnd) -> Option<i64> {
    for block in f.blocks[..head].iter().rev() {
        for inst in block.insts.iter().rev() {
            if inst.defs.contains(&reg) {
                return match (inst.op, &inst.uses[..]) {
                    (Opc::SMovB32, [Opnd::Imm(v)]) => Some(i64::from(*v as i32)),
                    _ => None,
                };
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::target::Target;

    fn s(n: u16) -> Opnd {
        Opnd::P(Phys::S(n), 1)
    }

    /// Block 0 sets two counters, blocks 1 to 3 are a loop of `outer` trips
    /// (block 3 its latch) around a loop of `inner` trips (block 2), block 4 a
    /// loop with a runtime bound.
    fn nest(outer: u32, inner: u32) -> MFunc {
        let add = |c: u16| MInst::new(Opc::SAddI32, [s(c)], [s(c), Opnd::Imm(1)]);
        let latch = |c: u16, bound: Opnd, head: i64| {
            vec![add(c), MInst::new(Opc::SCmpLtI32, [], [s(c), bound]), MInst::new(Opc::SCbranchScc1, [], []).with_imm(head)]
        };
        let blocks = vec![
            MBlock { insts: vec![MInst::new(Opc::SMovB32, [s(4)], [Opnd::Imm(0)]), MInst::new(Opc::SMovB32, [s(6)], [Opnd::Imm(0)])], loop_depth: 0 },
            MBlock { insts: vec![MInst::new(Opc::SMovB32, [s(5)], [Opnd::Imm(0)])], loop_depth: 1 },
            MBlock { insts: latch(5, Opnd::Imm(inner), 2), loop_depth: 2 },
            MBlock { insts: latch(4, Opnd::Imm(outer), 1), loop_depth: 1 },
            MBlock { insts: latch(6, s(9), 4), loop_depth: 1 },
        ];
        MFunc { blocks, target: Target::Gfx950, ..MFunc::default() }
    }

    #[test]
    fn a_block_runs_the_product_of_its_loops_trips() {
        assert_eq!(runs(&nest(3, 1)), vec![1, 3, 3, 3, UNKNOWN_TRIPS]);
        assert_eq!(runs(&nest(7, 2)), vec![1, 7, 14, 7, UNKNOWN_TRIPS]);
        // A loop whose bound does not exceed its start still runs once.
        assert_eq!(runs(&nest(0, 1)), vec![1, 1, 1, 1, UNKNOWN_TRIPS]);
    }

    /// A rolled loop of three trips whose only op reading the carried accumulator is an empty register pin,
    /// as the depthwise grouped convolutions have it.
    const PINNED: &str = "ckdsl.ir v1\nkernel @pinned {\n  attrs { max_workgroup_size = i:64 }\n  params {\n    \
        %X : ptr<f32,global> { align = i:4, noalias = b:true, readonly = b:true }\n    %Y : ptr<f32,global>\n  }\n  region @entry {\n    \
        %tid : i32 = gpu.thread_id (  ) { axis = s:\"x\" }\n    \
        %c0 : i32 = arith.constant (  ) { ity = s:\"i32\", value = i:0 }\n    \
        %c1 : i32 = arith.constant (  ) { ity = s:\"i32\", value = i:1 }\n    \
        %c3 : i32 = arith.constant (  ) { ity = s:\"i32\", value = i:3 }\n    \
        %z : f32 = arith.constant (  ) { ity = s:\"f32\", value = f:0.0 }\n    \
        %r : f32 = scf.for ( %c0, %c3, %c1, %z ) { iter_args = l:[ { name = s:\"%acc\", type = s:\"f32\" } ], iv = s:\"%k\", iv_type = s:\"i32\", num_iter_args = i:1, unroll = b:false }\n      region @body {\n        \
        %a : i32 = arith.add ( %tid, %k )\n        \
        %v : f32 = memref.global_load_typed ( %X, %a ) { align = i:4, elem_type = s:\"f32\" }\n        \
        %s : f32 = arith.fadd ( %acc, %v )\n        \
        %p : f32 = tile.inline_asm ( %s ) { constraints = s:\"=v,0\", convergent = b:false, sideeffect = b:false, template = s:\"\" }\n        \
        scf.yield ( %p ) { num = i:1 }\n      }\n    \
        memref.global_store_typed ( %Y, %tid, %r ) { align = i:4, elem_type = s:\"f32\" }\n    cf.return (  )\n  }\n}\n";

    #[test]
    fn a_loop_pinning_its_accumulator_unrolls_on_gfx950_only() {
        let unrolled = |auto: Auto| {
            let mut kernel = crate::rir::parse::parse(PINNED).unwrap_or_else(|e| panic!("{e}"));
            let (mut lds, _) = crate::rir::analysis::lds_layout(&kernel);
            crate::rir::unroll::unroll_loops_with(&mut kernel, &mut lds, Some(auto)).unwrap_or_else(|e| panic!("{e}"))
        };
        assert_eq!(unrolled(AUTO), 1, "gfx950 unrolls the pinned loop");
        assert_eq!(unrolled(Auto::DEFAULT), 0, "the shared policy keeps it rolled");
        // A pin that is a load (the gfx950 LDS read) still keeps the loop rolled.
        let mut kernel = crate::rir::parse::parse(PINNED).unwrap_or_else(|e| panic!("{e}"));
        let (mut lds, _) = crate::rir::analysis::lds_layout(&kernel);
        let mut stack = vec![&mut kernel.body];
        while let Some(region) = stack.pop() {
            for op in region.ops.iter_mut() {
                if op.kind == crate::rir::ast::OpKind::InlineAsmIdentity {
                    op.attrs.retain(|(k, _)| k != "template");
                    op.attrs.push(("template".into(), crate::rir::ast::Attr::Str("v_mov_b32 $0, $1".into())));
                }
                stack.extend(op.regions.iter_mut());
            }
        }
        assert_eq!(crate::rir::unroll::unroll_loops_with(&mut kernel, &mut lds, Some(AUTO)).expect("walks"), 0);
    }
}
