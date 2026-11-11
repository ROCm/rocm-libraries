// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

//! rustgemm lever `ds128`: an fp6 MFMA fragment read as three `ds_read_b64` becomes two reads.
//!
//! A 16x16x128 fp6 operand is 6 dwords per lane (24 bytes). Kernels whose LDS rows are 24 bytes (rocKE's canonical
//! MXFP6 layout, HipKittens' dwordx4 kernel) read it as three `ds_read_b64` at byte offsets `o`, `o + 8`, `o + 16` of
//! one address register, because no wider read is aligned in every lane. gfx950 runs LDS in unaligned mode under
//! ROCm (SH_MEM_CONFIG.ALIGNMENT_MODE = UNALIGNED, set by the KFD for every gfx9 queue), where `ds_read_b96` /
//! `ds_read_b128` at any dword-aligned address return the addressed bytes. With `ds128` the triple becomes
//! `ds_read_b128` at `o` and `ds_read_b64` at `o + 16` into one 6-dword register whose dword pairs replace the three
//! destinations. (Two `ds_read_b96` are no alternative: gfx950 VGPR tuples start at even registers, so the second
//! read's v[n+3:n+5] would go through a 3-dword copy.) Runs on virtual registers before
//! scheduling and allocation, inside one block, under the rules of `lds_pairs::merge`: the three loads read one
//! address register (not written between them) at those offsets, each destination is a 2-dword VGPR defined only by
//! its load, and nothing between the first and the last load writes LDS, EXEC or M0, fences (a `tile.sched_barrier`,
//! which only orders the schedule, does not end the window), branches or is a barrier. The first wide read is issued where the first load was and the second where the second was (it reads the
//! third load's bytes earlier, with no LDS write in between). Every lane reads the same bytes into the same dwords,
//! so the result is bitwise the unmerged code's on a GPU in unaligned LDS mode (on one in aligned mode a misaligned
//! lane's read is wrong: the A/B verify gates catch it).

use crate::rir::fx::FxHashMap;
use crate::rir::mir::*;

use super::dpp::overlaps;
use super::gen_isa::Gfx9Op;

fn ends_window(inst: &MInst) -> bool {
    // A `tile.sched_barrier` only orders the schedule (rocKE's `read_split` puts one between the three reads).
    if inst.op == Opc::SchedBarrier {
        return false;
    }
    let flags = inst.op.flags();
    let lds_write = flags & F_LDS != 0 && !is_plain_read(inst.op);
    lds_write
        || flags & (F_FENCE | F_DEF_EXEC | F_BRANCH | F_END) != 0
        || inst.defs.iter().any(|d| matches!(d, Opnd::P(Phys::ExecLo, _) | Opnd::P(Phys::M0, _)))
        || matches!(inst.op, Opc::SBarrier | Opc::SBarrierLds | Opc::SBarrierBare)
}

fn is_plain_read(op: Opc) -> bool {
    matches!(op, Opc::DsLoadB32 | Opc::DsLoadB64 | Opc::DsLoadB128 | Opc::DsLoadU16 | Opc::DsLoadU8 | Opc::Gfx9(Gfx9Op::DsReadB96))
}

fn def_counts(f: &MFunc) -> Vec<u32> {
    let mut defs = vec![0u32; f.vregs.len()];
    for inst in f.blocks.iter().flat_map(|b| &b.insts) {
        for d in &inst.defs {
            if let Opnd::R(v) = d {
                defs[v.reg as usize] += 1;
            }
        }
    }
    defs
}

/// A `ds_read_b64` into a whole 2-dword VGPR defined only here, from a virtual address register.
fn b64_load(inst: &MInst, defs: &[u32], f: &MFunc) -> bool {
    inst.op == Opc::DsLoadB64
        && inst.mods == 0
        && (0..65536 - 24).contains(&inst.imm)
        && matches!(inst.defs.first(), Some(Opnd::R(v)) if v.n == 2 && v.off == 0 && f.vregs[v.reg as usize].size == 2
            && f.vregs[v.reg as usize].class == Class::V && defs[v.reg as usize] == 1)
        && matches!(inst.uses.first(), Some(Opnd::R(_)))
}

/// Merges the triples of `f` under each load's lever; returns how many triples.
pub fn run(f: &mut MFunc) -> usize {
    let defs = def_counts(f);
    let mut renames: FxHashMap<u32, VRef> = FxHashMap::default();
    let mut merged = 0;
    for b in 0..f.blocks.len() {
        let mut i = 0;
        while i < f.blocks[b].insts.len() {
            let first = f.blocks[b].insts[i].clone();
            if !b64_load(&first, &defs, f) || !f.lever(&first, "ds128") {
                i += 1;
                continue;
            }
            let addr = first.uses[0];
            // Partners at +8 and +16, in either order, before the window ends.
            let mut at = [None, None];
            for k in i + 1..f.blocks[b].insts.len() {
                let x = &f.blocks[b].insts[k];
                if b64_load(x, &defs, f) && x.uses[0] == addr && x.scope == first.scope {
                    match x.imm - first.imm {
                        8 if at[0].is_none() => at[0] = Some(k),
                        16 if at[1].is_none() => at[1] = Some(k),
                        _ => {}
                    }
                    if at.iter().all(Option::is_some) {
                        break;
                    }
                }
                if ends_window(x) || x.defs.iter().any(|d| overlaps(d, &addr)) {
                    break;
                }
            }
            let [Some(k1), Some(k2)] = at else {
                i += 1;
                continue;
            };
            let t = f.vregs.len() as u32;
            f.vregs.push(VRegInfo { class: Class::V, size: 6, align: 2, fixed: None });
            let dst = |off: u8, n: u8| Opnd::R(VRef { reg: t, off, n });
            for (k, off) in [(i, 0u8), (k1, 2), (k2, 4)] {
                let Opnd::R(d) = f.blocks[b].insts[k].defs[0] else { unreachable!() };
                renames.insert(d.reg, VRef { reg: t, off, n: 2 });
            }
            let (op0, n0, op1, n1, o1) = (Opc::DsLoadB128, 4, Opc::DsLoadB64, 2, 16);
            let mut r0 = MInst::new(op0, [dst(0, n0)], [addr]).with_imm(first.imm);
            let mut r1 = MInst::new(op1, [dst(n0, n1)], [addr]).with_imm(first.imm + o1);
            (r0.mem, r0.scope) = (first.mem, first.scope);
            (r1.mem, r1.scope) = (first.mem, first.scope);
            // The second read goes where the earlier of the two partners was.
            let (lo, hi) = (k1.min(k2), k1.max(k2));
            f.blocks[b].insts[i] = r0;
            f.blocks[b].insts[lo] = r1;
            f.blocks[b].insts.remove(hi);
            merged += 1;
            i += 1;
        }
    }
    if !renames.is_empty() {
        for inst in f.blocks.iter_mut().flat_map(|b| b.insts.iter_mut()) {
            for o in inst.uses.iter_mut().chain(inst.defs.iter_mut()) {
                if let Opnd::R(v) = o {
                    if let Some(&to) = renames.get(&v.reg) {
                        *o = Opnd::R(VRef { reg: to.reg, off: to.off + v.off, n: v.n });
                    }
                }
            }
        }
    }
    merged
}

#[cfg(test)]
mod tests {
    use crate::rir;
    use crate::target::Target;

    /// A 64-lane kernel: 1 KiB of `%X` into LDS, a barrier, then each lane reads 24 bytes at `24 * lane + base` as
    /// three 8-byte loads (or `reads` instead) and stores them to `%Y`.
    fn kernel(reads: &str) -> String {
        format!(
            "ckdsl.ir v1\nkernel @k {{\n  attrs {{ max_workgroup_size = i:64 }}\n  params {{\n    %X : ptr<i32,global>\n    \
             %Y : ptr<i32,global>\n  }}\n  region @entry {{\n    %tid : i32 = gpu.thread_id (  ) {{ axis = s:\"x\" }}\n    \
             %c0 : i32 = arith.constant (  ) {{ ity = s:\"i32\", value = i:0 }}\n    \
             %c2 : i32 = arith.constant (  ) {{ ity = s:\"i32\", value = i:2 }}\n    \
             %c4 : i32 = arith.constant (  ) {{ ity = s:\"i32\", value = i:4 }}\n    \
             %c6 : i32 = arith.constant (  ) {{ ity = s:\"i32\", value = i:6 }}\n    \
             %c64 : i32 = arith.constant (  ) {{ ity = s:\"i32\", value = i:64 }}\n    \
             %t : smem<i32, [512]> = tile.smem_alloc (  )\n    \
             %x : i32 = memref.global_load_typed ( %X, %tid ) {{ align = i:4, elem_type = s:\"i32\" }}\n    \
             tile.smem_store ( %t, %tid, %x ) {{ elem_type = s:\"i32\", rank = i:1 }}\n    \
             tile.sync (  )\n    \
             %row : i32 = arith.mul ( %tid, %c6 )\n    \
             %r1 : i32 = arith.add ( %row, %c2 )\n    \
             %r2 : i32 = arith.add ( %row, %c4 )\n    \
             {reads}\n    \
             %a0 : i32 = vector.extract ( %a ) {{ index = i:0 }}\n    %a1 : i32 = vector.extract ( %a ) {{ index = i:1 }}\n    \
             %b0 : i32 = vector.extract ( %b ) {{ index = i:0 }}\n    %b1 : i32 = vector.extract ( %b ) {{ index = i:1 }}\n    \
             %d0 : i32 = vector.extract ( %c ) {{ index = i:0 }}\n    %d1 : i32 = vector.extract ( %c ) {{ index = i:1 }}\n    \
             %f : vec<i32x6> = vector.pack ( %a0, %a1, %b0, %b1, %d0, %d1 ) {{ elem = s:\"i32\", vec = i:6 }}\n    \
             %o : i32 = arith.mul ( %tid, %c6 )\n    \
             memref.global_store_vN ( %Y, %o, %f ) {{ align = i:8, elem_type = s:\"i32\", vec = i:6 }}\n    \
             cf.return (  )\n  }}\n}}\n"
        )
    }

    fn triple() -> String {
        kernel(
            "%a : vec<i32x2> = tile.smem_load_vN ( %t, %row ) { elem_type = s:\"i32\", rank = i:1, vec = i:2 }\n    \
             %b : vec<i32x2> = tile.smem_load_vN ( %t, %r1 ) { elem_type = s:\"i32\", rank = i:1, vec = i:2 }\n    \
             %c : vec<i32x2> = tile.smem_load_vN ( %t, %r2 ) { elem_type = s:\"i32\", rank = i:1, vec = i:2 }",
        )
    }

    fn listing(text: &str, levers: &[&str]) -> String {
        rir::with_levers(levers, || {
            rir::compile_any_target(text, Target::Gfx950).unwrap_or_else(|e| panic!("{e}")).listing()
        })
    }

    fn reads(listing: &str) -> Vec<String> {
        listing
            .lines()
            .filter(|l| l.trim_start().starts_with("ds_read"))
            .map(|l| l.trim().split("//").next().unwrap_or("").split_whitespace().collect::<Vec<_>>().join(" "))
            .collect()
    }

    #[test]
    fn three_b64_reads_become_a_b128_and_a_b64_into_one_tuple() {
        let base = listing(&triple(), &[]);
        let r = reads(&base);
        assert_eq!(r.iter().filter(|r| r.starts_with("ds_read_b64")).count(), 3, "{r:?}");
        let text = listing(&triple(), &["ds128"]);
        let r128 = reads(&text);
        assert_eq!(r128.len(), 2, "{r128:?}");
        assert!(r128[0].starts_with("ds_read_b128") && r128[1].starts_with("ds_read_b64") && r128[1].ends_with("imm:16"), "{r128:?}");
        // Bytes 0..16 and 16..24 of the lane's fragment land in one 6-dword tuple: no copies, no more VALU than before.
        let regs: Vec<&str> = r128.iter().map(|r| r.split(' ').nth(1).unwrap_or("")).collect();
        let lo: u32 = regs[0].trim_start_matches("v[").split(':').next().unwrap().parse().unwrap();
        assert_eq!(regs, [format!("v[{}:{}],", lo, lo + 3), format!("v[{}:{}],", lo + 4, lo + 5)]);
        let valu = |l: &str| l.lines().filter(|x| x.trim_start().starts_with("v_")).count();
        assert!(valu(&text) <= valu(&base), "{text}");
    }

    #[test]
    fn a_triple_split_by_sched_barriers_merges() {
        let split = kernel(
            "%a : vec<i32x2> = tile.smem_load_vN ( %t, %row ) { elem_type = s:\"i32\", rank = i:1, vec = i:2 }\n    \
             tile.sched_barrier (  ) { mask = i:0 }\n    \
             %b : vec<i32x2> = tile.smem_load_vN ( %t, %r1 ) { elem_type = s:\"i32\", rank = i:1, vec = i:2 }\n    \
             tile.sched_barrier (  ) { mask = i:0 }\n    \
             %c : vec<i32x2> = tile.smem_load_vN ( %t, %r2 ) { elem_type = s:\"i32\", rank = i:1, vec = i:2 }",
        );
        let r = reads(&listing(&split, &["ds128"]));
        assert_eq!(r.len(), 2, "{r:?}");
        assert!(r[0].starts_with("ds_read_b128"), "{r:?}");
    }

    #[test]
    fn a_triple_with_an_lds_write_or_a_gap_inside_stays() {
        let split = kernel(
            "%a : vec<i32x2> = tile.smem_load_vN ( %t, %row ) { elem_type = s:\"i32\", rank = i:1, vec = i:2 }\n    \
             tile.smem_store ( %t, %r1, %x ) { elem_type = s:\"i32\", rank = i:1 }\n    \
             %b : vec<i32x2> = tile.smem_load_vN ( %t, %r1 ) { elem_type = s:\"i32\", rank = i:1, vec = i:2 }\n    \
             %c : vec<i32x2> = tile.smem_load_vN ( %t, %r2 ) { elem_type = s:\"i32\", rank = i:1, vec = i:2 }",
        );
        let r = reads(&listing(&split, &["ds128"]));
        assert!(!r.iter().any(|x| x.starts_with("ds_read_b128")), "{r:?}");
        // Offsets 0, 8, 24: no +16 partner.
        let gap = kernel(
            "%c6b : i32 = arith.constant (  ) { ity = s:\"i32\", value = i:6 }\n    \
             %r3 : i32 = arith.add ( %row, %c6b )\n    \
             %a : vec<i32x2> = tile.smem_load_vN ( %t, %row ) { elem_type = s:\"i32\", rank = i:1, vec = i:2 }\n    \
             %b : vec<i32x2> = tile.smem_load_vN ( %t, %r1 ) { elem_type = s:\"i32\", rank = i:1, vec = i:2 }\n    \
             %c : vec<i32x2> = tile.smem_load_vN ( %t, %r3 ) { elem_type = s:\"i32\", rank = i:1, vec = i:2 }",
        );
        let r = reads(&listing(&gap, &["ds128"]));
        assert!(!r.iter().any(|x| x.starts_with("ds_read_b128")), "{r:?}");
    }

    #[test]
    fn without_a_lever_the_object_is_unchanged() {
        let a = rir::compile_any_target(&triple(), Target::Gfx950).unwrap_or_else(|e| panic!("{e}"));
        let b = rir::with_levers(&["ds128x"], || rir::compile_any_target(&triple(), Target::Gfx950).unwrap_or_else(|e| panic!("{e}")));
        assert!(a.hsaco == b.hsaco);
    }
}
