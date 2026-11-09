// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

//! rustgemm lever `sgb`: `tile.sched_group_barrier` orders the instructions before it, as LLVM's IGroupLP does.
//!
//! Without the lever the op emits nothing (the compiler ignored it before). With it, instruction selection leaves a
//! marker (`SchedBarrier` with `SGB` in its immediate: mask, count, sync group) and, before scheduling, this pass
//! builds the block's order from the groups: the markers of one scheduling region (between fences) are a pipeline
//! of groups in their program order (markers of different sync groups too: one pipeline, a simplification of
//! LLVM's independent pipelines per sync id); group `k` takes `count` instructions of the region before
//! its marker that match `mask` (LLVM's classes: 0x1 ALU, 0x2 VALU, 0x4 SALU, 0x8 MFMA, 0x10 VMEM, 0x20 VMEM read,
//! 0x40 VMEM write, 0x80 DS, 0x100 DS read, 0x200 DS write, 0x400 transcendental; mask 0 matches nothing): the
//! earliest such instructions no earlier group took. A list walk over the block's dependence graph emits, at each
//! step, the earliest ready member of the current group, else the earliest ready instruction no unfinished group
//! holds, else it closes the current group short (as LLVM leaves a group it cannot fill). A scheduling fence (`sched_barrier` mask 0) after each group keeps the
//! list scheduler from undoing the interleave; instructions inside a group (and the free ones between groups) stay
//! free to move. Only the order changes, and every dependence of the block is honoured, so values are unchanged.

use crate::rir::mir::*;

/// The immediate tag of a group marker (bit 62; bits 0-31 mask, 32-47 count, 48-55 sync group).
pub const SGB: i64 = 1 << 62;

pub fn marker(mask: i64, count: i64, group: i64) -> MInst {
    MInst::new(Opc::SchedBarrier, [], []).with_imm(SGB | (mask & 0xffff_ffff) | (count & 0xffff) << 32 | (group & 0xff) << 48)
}

pub fn is_marker(inst: &MInst) -> bool {
    inst.op == Opc::SchedBarrier && inst.imm & SGB != 0
}

fn fields(inst: &MInst) -> (u32, u32) {
    ((inst.imm & 0xffff_ffff) as u32, (inst.imm >> 32 & 0xffff) as u32)
}

/// Whether `inst` is in LLVM's instruction class `mask`.
pub fn matches(inst: &MInst, mask: u32) -> bool {
    let op = inst.op;
    // gfx950's own opcodes keep their format in the description (`Opc::fmt` is Pseudo for them).
    let fmt = super::gen_isa::info(op).map_or(Fmt::Pseudo, |i| i.fmt);
    if fmt == Fmt::Pseudo {
        return false;
    }
    let flags = op.flags();
    let mfma = super::hazards::mfma_kind(op).is_some();
    let valu = matches!(fmt, Fmt::Vop1 | Fmt::Vop2 | Fmt::Vop3 | Fmt::Vop3b | Fmt::Vopc | Fmt::Vop3p) && !mfma;
    let salu = matches!(fmt, Fmt::Sop1 | Fmt::Sop2 | Fmt::Sopc | Fmt::Sopk);
    let vmem = matches!(fmt, Fmt::Global | Fmt::Mubuf);
    let dma = super::lds_dma::is_dma(op);
    let ds = fmt == Fmt::Ds;
    (mask & 0x1 != 0 && (valu || salu || mfma))
        || (mask & 0x2 != 0 && valu)
        || (mask & 0x4 != 0 && salu)
        || (mask & 0x8 != 0 && mfma)
        || (mask & 0x10 != 0 && vmem)
        || (mask & 0x20 != 0 && vmem && flags & F_VMEM_LOAD != 0)
        || (mask & 0x40 != 0 && vmem && (flags & F_VMEM_STORE != 0 || dma))
        || (mask & 0x80 != 0 && ds)
        || (mask & 0x100 != 0 && ds && !inst.defs.is_empty())
        || (mask & 0x200 != 0 && ds && inst.defs.is_empty())
        || (mask & 0x400 != 0 && flags & F_TRANS != 0)
}

struct Group {
    mask: u32,
    left: u32,
    region: u32,
    /// The marker's position among the block's other instructions: candidates come before it.
    before: usize,
}

/// Orders every block of `f` that has group markers; returns how many groups were placed.
pub fn run(f: &mut MFunc) -> usize {
    let mut placed = 0;
    for b in 0..f.blocks.len() {
        if !f.blocks[b].insts.iter().any(is_marker) {
            continue;
        }
        let all = std::mem::take(&mut f.blocks[b].insts);
        let mut insts = Vec::with_capacity(all.len());
        let mut groups = Vec::new();
        let mut region = 0u32;
        for inst in all {
            if is_marker(&inst) {
                let (mask, count) = fields(&inst);
                if count > 0 {
                    groups.push(Group { mask, left: count, region, before: insts.len() });
                }
                continue;
            }
            let fence = inst.op.flags() & F_FENCE != 0 && !(inst.op == Opc::SchedBarrier && inst.imm != 0);
            insts.push(inst);
            if fence || matches!(insts.last().map(|i| i.op), Some(Opc::SBarrier | Opc::SBarrierBare | Opc::SBarrierLds)) {
                region += 1;
            }
        }
        f.blocks[b].insts = insts;
        placed += order_block(f, b, groups);
    }
    placed
}

fn order_block(f: &mut MFunc, b: usize, groups: Vec<Group>) -> usize {
    let succs = crate::rir::post::block_successors(f, b);
    let insts = std::mem::take(&mut f.blocks[b].insts);
    let n = insts.len();
    let mut region = vec![0u32; n];
    let mut r = 0u32;
    for (i, inst) in insts.iter().enumerate() {
        region[i] = r;
        let fence = inst.op.flags() & F_FENCE != 0 && !(inst.op == Opc::SchedBarrier && inst.imm != 0);
        if fence || matches!(inst.op, Opc::SBarrier | Opc::SBarrierBare | Opc::SBarrierLds) {
            r += 1;
        }
    }
    // Members: each group takes the earliest `count` unassigned instructions of its class before its marker.
    let mut member: Vec<Option<usize>> = vec![None; n];
    let mut left: Vec<u32> = vec![0; groups.len()];
    for (g, grp) in groups.iter().enumerate() {
        for i in 0..grp.before.min(n) {
            if left[g] == grp.left {
                break;
            }
            if member[i].is_none() && region[i] == grp.region && matches(&insts[i], grp.mask) {
                member[i] = Some(g);
                left[g] += 1;
            }
        }
    }
    let mut preds = vec![0u32; n];
    for s in &succs {
        for &j in s {
            preds[j as usize] += 1;
        }
    }
    let mut done = vec![false; n];
    let mut out: Vec<MInst> = Vec::with_capacity(n + groups.len());
    let mut fences = 0;
    let mut cur = 0usize;
    let mut in_group = false;
    while out.len() - fences < n {
        while cur < groups.len() && left[cur] == 0 {
            cur += 1;
        }
        let ready = |i: usize| !done[i] && preds[i] == 0;
        let pick = (cur < groups.len()).then(|| (0..n).find(|&i| ready(i) && member[i] == Some(cur))).flatten().or_else(|| {
            // Else an instruction no unfinished group holds.
            (0..n).find(|&i| ready(i) && member[i].is_none_or(|g| left[g] == 0 || g < cur))
        });
        let Some(i) = pick else {
            // Only members of later groups are ready: the current group closes short (its members become free).
            for m in member.iter_mut().filter(|m| **m == Some(cur)) {
                *m = None;
            }
            left[cur] = 0;
            if in_group {
                out.push(MInst::new(Opc::SchedBarrier, [], []));
                fences += 1;
                in_group = false;
            }
            continue;
        };
        done[i] = true;
        for &j in &succs[i] {
            preds[j as usize] -= 1;
        }
        out.push(insts[i].clone());
        if cur < groups.len() && member[i] == Some(cur) {
            left[cur] -= 1;
            in_group = true;
            if left[cur] == 0 {
                out.push(MInst::new(Opc::SchedBarrier, [], []));
                fences += 1;
                in_group = false;
            }
        }
    }
    f.blocks[b].insts = out;
    fences
}

#[cfg(test)]
mod tests {
    use crate::rir;
    use crate::target::Target;

    /// Four LDS reads then four global loads in program order, all summed and stored; `sgb` = the group markers
    /// after them.
    fn kernel(sgb: &str) -> String {
        let mut body = String::new();
        for k in 0..4 {
            body += &format!(
                "    %i{k} : i32 = arith.add ( %tid, %c{k} )\n    \
                 %r{k} : vec<i32x2> = tile.smem_load_vN ( %t, %i{k} ) {{ elem_type = s:\"i32\", rank = i:1, vec = i:2 }}\n    \
                 %e{k} : i32 = vector.extract ( %r{k} ) {{ index = i:0 }}\n"
            );
        }
        for k in 0..4 {
            body += &format!(
                "    %j{k} : i32 = arith.add ( %tid, %k{k} )\n    \
                 %x{k} : i32 = memref.global_load_typed ( %X, %j{k} ) {{ align = i:4, elem_type = s:\"i32\" }}\n"
            );
        }
        format!(
            "ckdsl.ir v1\nkernel @sgb {{\n  attrs {{ max_workgroup_size = i:256 }}\n  params {{\n    %X : ptr<i32,global>\n    \
             %Y : ptr<i32,global>\n  }}\n  region @entry {{\n    %tid : i32 = gpu.thread_id (  ) {{ axis = s:\"x\" }}\n    \
             %c0 : i32 = arith.constant (  ) {{ ity = s:\"i32\", value = i:0 }}\n    \
             %c1 : i32 = arith.constant (  ) {{ ity = s:\"i32\", value = i:64 }}\n    \
             %c2 : i32 = arith.constant (  ) {{ ity = s:\"i32\", value = i:128 }}\n    \
             %c3 : i32 = arith.constant (  ) {{ ity = s:\"i32\", value = i:192 }}\n    \
             %k0 : i32 = arith.constant (  ) {{ ity = s:\"i32\", value = i:1234567 }}\n    \
             %k1 : i32 = arith.constant (  ) {{ ity = s:\"i32\", value = i:7654321 }}\n    \
             %k2 : i32 = arith.constant (  ) {{ ity = s:\"i32\", value = i:5555555 }}\n    \
             %k3 : i32 = arith.constant (  ) {{ ity = s:\"i32\", value = i:3333333 }}\n    \
             %t : smem<i32, [512]> = tile.smem_alloc (  )\n    \
             %g : i32 = memref.global_load_typed ( %X, %tid ) {{ align = i:4, elem_type = s:\"i32\" }}\n    \
             tile.smem_store ( %t, %tid, %g ) {{ elem_type = s:\"i32\", rank = i:1 }}\n    \
             tile.sync (  )\n{body}{sgb}    \
             %s0 : i32 = arith.add ( %e0, %e1 )\n    %s1 : i32 = arith.add ( %e2, %e3 )\n    \
             %s2 : i32 = arith.add ( %x0, %x1 )\n    %s3 : i32 = arith.add ( %x2, %x3 )\n    \
             %s4 : i32 = arith.add ( %s0, %s1 )\n    %s5 : i32 = arith.add ( %s2, %s3 )\n    \
             %s6 : i32 = arith.add ( %s4, %s5 )\n    \
             memref.global_store_typed ( %Y, %tid, %s6 ) {{ align = i:4, elem_type = s:\"i32\" }}\n    \
             cf.return (  )\n  }}\n}}\n"
        )
    }

    fn markers(pattern: &[(u32, u32)]) -> String {
        pattern.iter().map(|&(mask, count)| format!("    tile.sched_group_barrier (  ) {{ count = i:{count}, group = i:0, mask = i:{mask} }}\n")).collect()
    }

    /// The order of the global loads (`x`) and LDS reads (`r`) in the listing.
    fn order(levers: &[&str], sgb: &str) -> String {
        let l = rir::with_levers(levers, || rir::compile_any_target(&kernel(sgb), Target::Gfx950).unwrap_or_else(|e| panic!("{e}")).listing());
        l.lines()
            .filter_map(|x| {
                let t = x.trim_start();
                if t.starts_with("global_load") {
                    Some('x')
                } else if t.starts_with("ds_read") {
                    Some('r')
                } else {
                    None
                }
            })
            .collect()
    }

    #[test]
    fn groups_interleave_the_instructions_before_them() {
        let alternate = markers(&[(0x20, 1), (0x100, 1), (0x20, 1), (0x100, 1), (0x20, 1), (0x100, 1), (0x20, 1), (0x100, 1)]);
        // (The first `x` is the load before the barrier.)
        assert_eq!(order(&["sgb"], &alternate), "xxrxrxrxr");
        let pairs = markers(&[(0x100, 2), (0x10, 2), (0x80, 2), (0x20, 2)]);
        assert_eq!(order(&["sgb"], &pairs), "xrrxxrrxx");
        // A group nothing can fill (no MFMA here) closes short; the rest keep their order.
        let short = markers(&[(0x20, 1), (0x8, 1), (0x100, 1), (0x20, 1), (0x100, 1), (0x20, 1), (0x100, 1), (0x20, 1), (0x100, 1)]);
        assert_eq!(order(&["sgb"], &short), "xxrxrxrxr");
        // Without markers (or the lever) the list scheduler keeps its own order.
        assert_ne!(order(&[], &alternate), "xxrxrxrxr");
    }

    #[test]
    fn selection_leaves_markers_only_under_the_lever() {
        let text = kernel(&markers(&[(0x20, 1), (0x100, 2)]));
        let count = |levers: &[&str]| {
            rir::with_levers(levers, || {
                let mut k = rir::parse::parse(&text).unwrap_or_else(|e| panic!("{e}"));
                k.wave = 64;
                let (mut lds, size) = rir::analysis::lds_layout(&k);
                let facts = rir::opt::optimize(&mut k, &mut lds).unwrap_or_else(|e| panic!("{e}"));
                let uniform = rir::analysis::classify(&k);
                let sel = rir::isel::select_for(Target::Gfx950, &k, &facts, uniform, &lds, size).unwrap_or_else(|e| panic!("{e}"));
                sel.func.blocks.iter().flat_map(|b| &b.insts).filter(|i| super::is_marker(i)).map(|i| i.imm & 0xffff_ffff_ffff).collect::<Vec<_>>()
            })
        };
        assert!(count(&[]).is_empty());
        assert_eq!(count(&["sgb"]), [1 << 32 | 0x20, 2 << 32 | 0x100]);
    }

    #[test]
    fn classes_are_llvms() {
        use crate::rir::mir::*;
        use crate::target::gfx9::gen_isa::Gfx9Op;
        let v = |r: u16, n: u8| Opnd::P(Phys::V(r), n);
        let read = MInst::new(Opc::DsLoadB64, [v(2, 2)], [v(0, 1)]);
        let write = MInst::new(Opc::DsStoreB32, [], [v(0, 1), v(1, 1)]);
        let gload = MInst::new(Opc::GLoadB32, [v(2, 1)], [v(0, 2)]);
        let gstore = MInst::new(Opc::GStoreB32, [], [v(0, 2), v(1, 1)]);
        let dma = MInst::new(Opc::Gfx9(Gfx9Op::BufLoadLdsB32), [], [v(0, 1), Opnd::P(Phys::S(4), 4), Opnd::Imm(0)]);
        let valu = MInst::new(Opc::VAddNcU32, [v(3, 1)], [v(1, 1), v(2, 1)]);
        let salu = MInst::new(Opc::SAddI32, [Opnd::P(Phys::S(3), 1)], [Opnd::P(Phys::S(1), 1), Opnd::Imm(1)]);
        let class = |i: &MInst| (0..11).filter(|b| super::matches(i, 1 << b)).fold(0u32, |m, b| m | 1 << b);
        assert_eq!(class(&read), 0x80 | 0x100);
        assert_eq!(class(&write), 0x80 | 0x200);
        assert_eq!(class(&gload), 0x10 | 0x20);
        assert_eq!(class(&gstore), 0x10 | 0x40);
        assert_eq!(class(&dma), 0x10 | 0x20 | 0x40);
        assert_eq!(class(&valu), 0x1 | 0x2);
        assert_eq!(class(&salu), 0x1 | 0x4);
    }

    #[test]
    fn without_the_lever_the_markers_emit_nothing() {
        let plain = rir::compile_any_target(&kernel(""), Target::Gfx950).unwrap_or_else(|e| panic!("{e}"));
        let marked = rir::compile_any_target(&kernel(&markers(&[(0x2, 1), (0x100, 1)])), Target::Gfx950).unwrap_or_else(|e| panic!("{e}"));
        assert!(plain.hsaco == marked.hsaco);
    }
}
