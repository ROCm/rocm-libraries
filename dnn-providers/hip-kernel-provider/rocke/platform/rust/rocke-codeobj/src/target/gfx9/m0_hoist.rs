// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

//! LDS DMA addresses leave VGPRs at their definition. Selection writes M0
//! for each DMA load from the first lane of its address
//! (`lds_dma::select`: `v_readfirstlane_b32`, then `s_mov_b32 m0`). An
//! unrolled GEMM computes each buffer's LDS address once and loads into it
//! every other K step, so each address stays in a VGPR across the kernel: 12
//! VGPRs in the grouped GEMM `e6117daa6b3804d5`, enough to spill at the 256
//! a 512-lane workgroup leaves. COMGR reads each address's first lane once,
//! where it is computed, and keeps it in an SGPR.
//!
//! Here likewise: a VGPR whose first lane feeds M0 (a `v_readfirstlane_b32`
//! read only by a move to M0) is read once, right after the instruction of
//! its block that defines it, and each such read becomes a copy of that
//! SGPR. The first active lane is the same lane while EXEC is, so this is
//! done only where nothing between the definition and the read writes EXEC.

use crate::rir::mir::*;

fn writes_exec(inst: &MInst) -> bool {
    inst.op.flags() & F_DEF_EXEC != 0 || inst.defs.iter().any(|d| matches!(d, Opnd::P(Phys::ExecLo, _)))
}

/// Rewrites `f` (after selection) as the module documentation says; returns
/// how many reads became copies. `ROCKE_DIRECT_DEBUG=nom0hoist` skips it.
pub fn run(f: &mut MFunc) -> usize {
    if crate::rir::debug_flag("nom0hoist") {
        return 0;
    }
    // Uses of each SGPR register, and whether each is a move to M0.
    let mut s_uses: std::collections::HashMap<u32, (usize, usize)> = std::collections::HashMap::new();
    for inst in f.blocks.iter().flat_map(|b| &b.insts) {
        let to_m0 = inst.op == Opc::SMovB32 && matches!(inst.defs.first(), Some(Opnd::P(Phys::M0, _)));
        for u in &inst.uses {
            if let Opnd::R(v) = u {
                let e = s_uses.entry(v.reg).or_default();
                e.0 += 1;
                e.1 += usize::from(to_m0);
            }
        }
    }
    let mut v_defs = vec![0u32; f.vregs.len()];
    for inst in f.blocks.iter().flat_map(|b| &b.insts) {
        for d in &inst.defs {
            if let Opnd::R(v) = d {
                v_defs[v.reg as usize] += 1;
            }
        }
    }
    let mut rewritten = 0;
    for b in 0..f.blocks.len() {
        // The M0 reads of each VGPR (one dword, defined once, in this block).
        let insts = &f.blocks[b].insts;
        let mut reads: Vec<(VRef, Vec<usize>)> = Vec::new();
        for (k, inst) in insts.iter().enumerate() {
            if inst.op != Opc::VReadfirstlaneB32 {
                continue;
            }
            let (Some(Opnd::R(s)), Some(Opnd::R(v))) = (inst.defs.first(), inst.uses.first()) else { continue };
            let m0_only = s_uses.get(&s.reg).is_some_and(|&(n, m0)| n > 0 && n == m0);
            let whole = v.off == 0 && v.n == 1 && f.vregs[v.reg as usize].size == 1 && f.vregs[v.reg as usize].class == Class::V;
            if !m0_only || !whole || v_defs[v.reg as usize] != 1 || f.vregs[v.reg as usize].fixed.is_some() {
                continue;
            }
            match reads.iter_mut().find(|(x, _)| x.reg == v.reg) {
                Some((_, ks)) => ks.push(k),
                None => reads.push((*v, vec![k])),
            }
        }
        // Each VGPR's definition in the block, and the reads EXEC does not change before.
        let mut plan: Vec<(usize, VRef, Vec<usize>)> = Vec::new();
        for (v, ks) in reads {
            let Some(d) = insts.iter().position(|i| i.defs.iter().any(|o| matches!(o, Opnd::R(x) if x.reg == v.reg))) else { continue };
            let exec_at = (d + 1..insts.len()).find(|&j| writes_exec(&insts[j])).unwrap_or(insts.len());
            let ks: Vec<usize> = ks.into_iter().filter(|&k| d < k && k < exec_at).collect();
            if !ks.is_empty() {
                plan.push((d, v, ks));
            }
        }
        if plan.is_empty() {
            continue;
        }
        let mut after: std::collections::HashMap<usize, Vec<MInst>> = std::collections::HashMap::new();
        let mut copy_of: std::collections::HashMap<usize, Opnd> = std::collections::HashMap::new();
        for (d, v, ks) in plan {
            let reg = f.vregs.len() as u32;
            f.vregs.push(VRegInfo { class: Class::S, size: 1, align: 1, fixed: None });
            let first = Opnd::R(VRef { reg, off: 0, n: 1 });
            after.entry(d).or_default().push(MInst::new(Opc::VReadfirstlaneB32, [first], [Opnd::R(v)]));
            for k in ks {
                copy_of.insert(k, first);
            }
        }
        let old = std::mem::take(&mut f.blocks[b].insts);
        let mut out = Vec::with_capacity(old.len() + after.len());
        for (k, mut inst) in old.into_iter().enumerate() {
            if let Some(&first) = copy_of.get(&k) {
                inst = MInst::new(Opc::Copy, [inst.defs[0]], [first]);
                rewritten += 1;
            }
            out.push(inst);
            if let Some(extra) = after.remove(&k) {
                out.extend(extra);
            }
        }
        f.blocks[b].insts = out;
    }
    rewritten
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::target::Target;

    fn r(reg: u32) -> Opnd {
        Opnd::R(VRef { reg, off: 0, n: 1 })
    }

    fn m0(s: u32) -> MInst {
        MInst::new(Opc::SMovB32, [Opnd::P(Phys::M0, 1)], [r(s)])
    }

    fn rfl(s: u32, v: u32) -> MInst {
        MInst::new(Opc::VReadfirstlaneB32, [r(s)], [r(v)])
    }

    /// v0 = tid, v1 = v0 << 10, then two DMA M0 writes from v1 (s2, s3), an EXEC write, and a third (s4).
    fn func() -> MFunc {
        let mut f = MFunc { target: Target::Gfx950, ..MFunc::default() };
        for class in [Class::V, Class::V, Class::S, Class::S, Class::S, Class::S] {
            f.vregs.push(VRegInfo { class, size: 1, align: 1, fixed: None });
        }
        let shl = MInst::new(Opc::VLshlrevB32, [r(1)], [Opnd::Imm(10), r(0)]);
        let exec = MInst::new(Opc::SAndSaveexecB64, [Opnd::R(VRef { reg: 5, off: 0, n: 1 })], [r(5)]);
        f.blocks.push(MBlock { insts: vec![shl, rfl(2, 1), m0(2), rfl(3, 1), m0(3), exec, rfl(4, 1), m0(4)], loop_depth: 0 });
        f
    }

    #[test]
    fn an_m0_address_is_read_once_where_it_is_defined() {
        let mut f = func();
        assert_eq!(run(&mut f), 2);
        let insts = &f.blocks[0].insts;
        // The read moves right after the shift; the two before the EXEC write copy it.
        assert_eq!(insts[1].op, Opc::VReadfirstlaneB32);
        let first = insts[1].defs[0];
        assert!(matches!(first, Opnd::R(v) if v.reg == 6));
        assert_eq!((insts[2].op, insts[2].uses[0]), (Opc::Copy, first));
        assert_eq!((insts[4].op, insts[4].uses[0]), (Opc::Copy, first));
        // Past the EXEC write the read stays.
        assert_eq!((insts[7].op, insts[7].uses[0]), (Opc::VReadfirstlaneB32, r(1)));
    }

    #[test]
    fn other_lane_reads_stay() {
        let mut f = func();
        // s2 is also read by a SALU add: not an M0-only read.
        f.blocks[0].insts.push(MInst::new(Opc::SAddI32, [r(3)], [r(2), Opnd::Imm(1)]));
        let before = f.blocks[0].insts.clone();
        run(&mut f);
        assert_eq!(f.blocks[0].insts[1].op, Opc::VReadfirstlaneB32);
        // The read of s2 is untouched; s3's moved and became a copy.
        assert!(f.blocks[0].insts.iter().any(|i| i.op == Opc::VReadfirstlaneB32 && i.defs[0] == r(2)));
        assert!(f.blocks[0].insts.len() > before.len());
    }
}
