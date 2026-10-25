// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

//! Machine passes. Before allocation, on virtual registers: dead-copy removal,
//! VALU fusion, load-use sinking and latency scheduling. After allocation, on
//! physical registers: pseudo expansion, wait counters, the WMMA hazard, ALU
//! delay hints, and the VGPR-release tail.

use super::fx::FxHashMap;
use super::mir::*;
use crate::error::CompileError;

fn internal(message: impl Into<String>) -> CompileError {
    CompileError::new("ROCKE_DIRECT_INTERNAL", message)
}

fn phys_base(o: &Opnd) -> Option<(bool, u16, u8)> {
    match o {
        Opnd::P(Phys::V(r), n) => Some((true, *r, *n)),
        Opnd::P(Phys::S(r), n) => Some((false, *r, *n)),
        _ => None,
    }
}

fn dword(o: &Opnd, k: u8) -> Opnd {
    match o {
        Opnd::P(Phys::V(r), _) => Opnd::P(Phys::V(r + k as u16), 1),
        Opnd::P(Phys::S(r), _) => Opnd::P(Phys::S(r + k as u16), 1),
        other => *other,
    }
}

fn vunits(o: &Opnd) -> impl Iterator<Item = (u32, u8)> + '_ {
    let (reg, off, n) = match o {
        Opnd::R(v) => (v.reg, v.off, v.n),
        _ => (0, 0, 0),
    };
    (off..off + n).map(move |k| (reg, k))
}

/// Dense numbering of the (virtual register, dword) units that operands of
/// some instructions name.
struct UnitIndex {
    base: Vec<u32>,
}

impl UnitIndex {
    fn new<'a>(insts: impl Iterator<Item = &'a MInst> + Clone) -> UnitIndex {
        let regs = insts
            .clone()
            .flat_map(|i| i.defs.iter().chain(&i.uses))
            .filter_map(Opnd::vref)
            .map(|v| v.reg as usize + 1)
            .max()
            .unwrap_or(0);
        let mut width = vec![0u32; regs + 1];
        for v in insts.flat_map(|i| i.defs.iter().chain(&i.uses)).filter_map(Opnd::vref) {
            let w = &mut width[v.reg as usize];
            *w = (*w).max(v.off as u32 + v.n as u32);
        }
        let mut base = width;
        let mut total = 0;
        for slot in base.iter_mut() {
            let w = *slot;
            *slot = total;
            total += w;
        }
        UnitIndex { base }
    }

    fn len(&self) -> usize {
        *self.base.last().expect("non-empty") as usize
    }

    fn of(&self, o: &Opnd) -> std::ops::Range<usize> {
        match o {
            Opnd::R(v) => {
                let start = (self.base[v.reg as usize] + v.off as u32) as usize;
                start..start + v.n as usize
            }
            _ => 0..0,
        }
    }
}

fn shares_vreg(a: &[Opnd], b: &[Opnd]) -> bool {
    a.iter().any(|x| vunits(x).any(|u| b.iter().any(|y| vunits(y).any(|w| w == u))))
}

fn has_phys(inst: &MInst) -> bool {
    inst.defs.iter().chain(inst.uses.iter()).any(|o| matches!(o, Opnd::P(p, _) if *p != Phys::Null))
}

/// What `may_sink_past` reads of an instruction: its flags, whether nothing
/// may pass it, whether it writes LDS, and 64-bit signatures of the
/// virtual-register units it defines and uses (a shared unit sets a common bit).
#[derive(Clone, Copy)]
struct SinkFacts {
    flags: u32,
    fence: bool,
    lds_write: bool,
    defs: u64,
    uses: u64,
}

fn unit_signature(opnds: &[Opnd]) -> u64 {
    opnds.iter().flat_map(vunits).fold(0, |s, (reg, k)| s | 1 << (reg.wrapping_mul(7).wrapping_add(k as u32) % 64))
}

impl SinkFacts {
    fn of(x: &MInst) -> SinkFacts {
        let flags = x.op.flags();
        SinkFacts {
            flags,
            fence: flags & (F_BRANCH | F_END | F_DEF_EXEC) != 0
                || matches!(x.op, Opc::SBarrier | Opc::SSendmsg)
                || has_phys(x),
            // LDS reads (loads, swizzles) define a register; LDS writes do not.
            lds_write: flags & F_LDS != 0 && x.defs.is_empty(),
            defs: unit_signature(&x.defs),
            uses: unit_signature(&x.uses),
        }
    }
}

/// Whether `c` may move below `x`.
fn may_sink_past(c: &MInst, cs: &SinkFacts, x: &MInst, xs: &SinkFacts) -> bool {
    let (cf, xf) = (cs.flags, xs.flags);
    if xs.fence {
        return false;
    }
    let vmem = F_VMEM_LOAD | F_VMEM_STORE;
    if (cf & F_LDS != 0 && xf & F_LDS != 0 && (cs.lds_write || xs.lds_write))
        || (cf & F_VMEM_STORE != 0 && xf & vmem != 0)
        || (cf & vmem != 0 && xf & F_VMEM_STORE != 0)
    {
        return false;
    }
    if (cs.defs & (xs.uses | xs.defs)) | (cs.uses & xs.defs) == 0 {
        return true;
    }
    !(shares_vreg(&c.defs, &x.uses) || shares_vreg(&c.uses, &x.defs) || shares_vreg(&c.defs, &x.defs))
}

/// The three-source form of `x(p(a, b), c)` for a producer `p` whose result
/// is `x`'s source `slot`: the sources in VOP3 order.
fn fused_form(p: &MInst, x: &MInst, slot: usize) -> Option<(Opc, [Opnd; 3])> {
    let (&[a, b], &[x0, x1]) = (&p.uses[..], &x.uses[..]) else {
        return None;
    };
    let c = if slot == 0 { x1 } else { x0 };
    Some(match (p.op, x.op) {
        (Opc::VAddNcU32, Opc::VLshlrevB32) if slot == 1 => (Opc::VAddLshlU32, [a, b, x0]),
        (Opc::VLshlrevB32, Opc::VAddNcU32) => (Opc::VLshlAddU32, [b, a, c]),
        (Opc::VLshlrevB32, Opc::VOrB32) => (Opc::VLshlOrB32, [b, a, c]),
        (Opc::VAddNcU32, Opc::VAddNcU32) => (Opc::VAdd3U32, [a, b, c]),
        (Opc::VMulU32U24, Opc::VAddNcU32) => (Opc::VMadU32U24, [a, b, c]),
        (Opc::VAndB32, Opc::VOrB32) => (Opc::VAndOrB32, [a, b, c]),
        (Opc::VOrB32, Opc::VOrB32) => (Opc::VOr3B32, [a, b, c]),
        (Opc::VMinI32, Opc::VMaxI32) => (Opc::VMinmaxI32, [a, b, c]),
        (Opc::VMaxI32, Opc::VMinI32) => (Opc::VMaxminI32, [a, b, c]),
        _ => return None,
    })
}

/// `x`, a `v_fma_f32` reading `v_cvt_f32_f16` results that have no other use,
/// as `v_fma_mix_f32` reading the f16 halves; the conversion is exact, so the
/// result is the same. Also returns the folded conversions.
/// Index of the last instruction of the current block that defines each
/// virtual register, tagged with the block number.
struct LastDef {
    at: Vec<(u32, u32)>,
    block: u32,
}

impl LastDef {
    fn get(&self, reg: u32) -> Option<usize> {
        let (block, at) = self.at[reg as usize];
        (block == self.block).then_some(at as usize)
    }
}

fn fma_mix_form(
    insts: &[MInst],
    x: usize,
    def_at: &LastDef,
    counts: (&[u32], &[u32]),
    vregs: &[VRegInfo],
) -> Option<(MInst, Vec<usize>)> {
    let xi = &insts[x];
    if xi.op != Opc::VFmaF32 || xi.mods != 0 || has_phys(xi) {
        return None;
    }
    let (mut srcs, mut mods, mut folded) = (xi.uses, 0, Vec::new());
    for slot in 0..srcs.len() {
        let Opnd::R(t) = srcs[slot] else {
            continue;
        };
        let Some(p) = def_at.get(t.reg) else {
            continue;
        };
        let pi = &insts[p];
        let single = counts.0[t.reg as usize] == 1 && counts.1[t.reg as usize] == 1;
        let half = matches!(pi.uses[..], [Opnd::R(s)] if vregs[s.reg as usize].class == Class::V);
        if !single
            || !half
            || pi.op != Opc::VCvtF32F16
            || pi.mods & !M_OPSEL_SRC0 != 0
            || pi.defs[..] != [Opnd::R(t)]
            || insts[p + 1..x]
                .iter()
                .any(|k| k.op.flags() & F_DEF_EXEC != 0 || shares_vreg(&k.defs, &pi.uses))
        {
            continue;
        }
        srcs[slot] = pi.uses[0];
        mods |= M_MIX_F16_0 << slot | (pi.mods & M_OPSEL_SRC0) << slot;
        folded.push(p);
    }
    // VOP3P takes no literal.
    let inline = |o: &Opnd| match *o {
        Opnd::Imm(v) => super::isel::is_inline_int(v) || super::isel::is_inline_f32(v),
        _ => true,
    };
    (!folded.is_empty() && srcs.iter().all(inline) && scalar_sources_fit(&srcs, vregs))
        .then(|| (MInst::new(Opc::VFmaMixF32, xi.defs, srcs).with_mods(mods), folded))
}

/// VOP3 reads at most two scalar values, one of which may be a literal.
fn scalar_sources_fit(srcs: &[Opnd], vregs: &[VRegInfo]) -> bool {
    let mut scalars = Uses::new();
    let mut literals = Uses::new();
    for &o in srcs {
        match o {
            Opnd::Imm(v) if !(super::isel::is_inline_int(v) || super::isel::is_inline_f32(v)) => {
                if !literals.contains(&o) {
                    literals.push(o);
                }
            }
            Opnd::R(v) if vregs[v.reg as usize].class == Class::S => {
                if !scalars.contains(&o) {
                    scalars.push(o);
                }
            }
            _ => {}
        }
    }
    literals.len() <= 1 && scalars.len() + literals.len() <= 2
}

/// Folds a VALU result with a single use into that use when the pair has a
/// three-source VOP3 form (`v_add3_u32`, `v_add_lshl_u32`, `v_minmax_i32`,
/// ...), as LLVM selects them: one instruction and one dependency fewer.
/// f16-to-f32 conversions feeding an FMA fold into `v_fma_mix_f32`.
pub fn fuse_valu(f: &mut MFunc) {
    let n = f.vregs.len();
    let (mut uses, mut defs) = (vec![0u32; n], vec![0u32; n]);
    for inst in f.blocks.iter().flat_map(|b| b.insts.iter()) {
        for o in &inst.uses {
            if let Opnd::R(v) = o {
                uses[v.reg as usize] += 1;
            }
        }
        for o in &inst.defs {
            if let Opnd::R(v) = o {
                defs[v.reg as usize] += 1;
            }
        }
    }
    let vregs = &f.vregs;
    let mut def_at = LastDef { at: vec![(0, 0); n], block: 0 };
    for block in f.blocks.iter_mut() {
        def_at.block += 1;
        let insts = &mut block.insts;
        let mut dead = vec![false; insts.len()];
        for x in 0..insts.len() {
            let fused = (0..insts[x].uses.len()).find_map(|slot| {
                let (xi, Opnd::R(t)) = (&insts[x], insts[x].uses[slot]) else {
                    return None;
                };
                let info = vregs[t.reg as usize];
                if xi.mods != 0
                    || has_phys(xi)
                    || info.class != Class::V
                    || info.size != 1
                    || uses[t.reg as usize] != 1
                    || defs[t.reg as usize] != 1
                {
                    return None;
                }
                let p = def_at.get(t.reg)?;
                let pi = &insts[p];
                if pi.mods != 0 || has_phys(pi) || pi.defs[..] != [Opnd::R(t)] {
                    return None;
                }
                let (op, srcs) = fused_form(pi, xi, slot)?;
                let moved = insts[p + 1..x]
                    .iter()
                    .any(|k| k.op.flags() & F_DEF_EXEC != 0 || shares_vreg(&k.defs, &pi.uses));
                if moved || !scalar_sources_fit(&srcs, vregs) {
                    return None;
                }
                Some((p, MInst::new(op, xi.defs, srcs)))
            });
            if let Some((inst, folded)) = fma_mix_form(insts, x, &def_at, (&uses, &defs), vregs) {
                insts[x] = inst;
                for p in folded {
                    dead[p] = true;
                }
            } else if let Some((p, inst)) = fused {
                insts[x] = inst;
                dead[p] = true;
            }
            for d in &insts[x].defs {
                if let Opnd::R(v) = d {
                    def_at.at[v.reg as usize] = (def_at.block, x as u32);
                }
            }
        }
        let mut k = 0;
        insts.retain(|_| {
            k += 1;
            !dead[k - 1]
        });
    }
}

/// Delays each instruction that reads a load's result below later
/// independent instructions, so that loads issue back to back and their
/// latencies overlap instead of each stalling at its first use. At most
/// `budget` dwords of loads move ahead of any one instruction, which bounds
/// the extra register pressure.
pub fn sink_load_uses(f: &mut MFunc, budget: u32) {
    let fixed = F_DEF_SCC | F_USE_SCC | F_USE_VCC | F_DEF_EXEC | F_BRANCH | F_END | F_SMEM;
    let units = UnitIndex::new(f.blocks.iter().flat_map(|b| b.insts.iter()));
    // Block number + 1 of the units loaded so far in that block.
    let mut loaded = vec![0u32; units.len()];
    for (b, block) in f.blocks.iter_mut().enumerate() {
        let stamp = b as u32 + 1;
        let insts = &mut block.insts;
        let mut candidate = vec![false; insts.len()];
        for (i, inst) in insts.iter().enumerate() {
            // VMEM loads write in issue order, so a VMEM load's tied
            // destination does not stall it; an LDS load's does.
            let load = inst.op.flags() & (F_VMEM_LOAD | F_LDS) != 0;
            let in_order = inst.op.flags() & F_VMEM_LOAD != 0;
            candidate[i] = inst
                .uses
                .iter()
                .any(|o| !(in_order && inst.defs.contains(o)) && units.of(o).any(|u| loaded[u] == stamp));
            if load || (inst.op == Opc::Copy && candidate[i]) {
                for d in &inst.defs {
                    for u in units.of(d) {
                        loaded[u] = stamp;
                    }
                }
            }
        }
        let mut facts: Vec<SinkFacts> = insts.iter().map(SinkFacts::of).collect();
        for i in (0..insts.len()).rev() {
            let c = &insts[i];
            if !candidate[i] || c.op.flags() & fixed != 0 || has_phys(c) || c.op == Opc::SBarrier {
                continue;
            }
            let is_load = |inst: &MInst| inst.op.flags() & (F_VMEM_LOAD | F_LDS) != 0 && !inst.defs.is_empty();
            let cs = facts[i];
            let mut j = i + 1;
            let mut passed = 0u32;
            while j < insts.len() && may_sink_past(c, &cs, &insts[j], &facts[j]) {
                if is_load(&insts[j]) {
                    // Delayed loads keep their order, so each still waits
                    // on the oldest outstanding load.
                    if is_load(c) && candidate[j] {
                        break;
                    }
                    passed += insts[j].defs.iter().map(|d| vunits(d).count() as u32).sum::<u32>();
                    if passed > budget {
                        break;
                    }
                }
                j += 1;
            }
            if j > i + 1 {
                insts[i..j].rotate_left(1);
                candidate[i..j].rotate_left(1);
                facts[i..j].rotate_left(1);
            }
        }
    }
}

/// Cycles from issue until the scheduler lets a reader of the result issue.
fn result_latency(inst: &MInst) -> u32 {
    let flags = inst.op.flags();
    match inst.op.fmt() {
        Fmt::Pseudo if flags & F_USE_EXEC == 0 => 0,
        _ if flags & F_VMEM_LOAD != 0 => 300,
        _ if flags & (F_LDS | F_SMEM) != 0 => 40,
        _ if flags & F_WMMA != 0 => 16,
        _ if flags & F_TRANS != 0 => 10,
        Fmt::Sop1 | Fmt::Sop2 | Fmt::Sopc | Fmt::Sopk => 2,
        _ => 5,
    }
}

/// Reorders each block so that instructions do not wait on results still in
/// flight: each step issues, among the unscheduled instructions at most
/// `window` places after the first, one whose predecessors have issued and
/// that can start soonest; of those, the one with the longest latency path
/// to the end of the block. An instruction moves at most `window` places,
/// which bounds the change in register pressure.
pub fn schedule_window(f: &mut MFunc, window: usize) {
    let units = UnitIndex::new(f.blocks.iter().flat_map(|b| b.insts.iter()));
    for block in f.blocks.iter_mut() {
        let insts = std::mem::take(&mut block.insts);
        block.insts = schedule_block(insts, window, &units);
    }
}

const NO_INST: u32 = u32::MAX;

fn schedule_block(insts: Vec<MInst>, window: usize, units: &UnitIndex) -> Vec<MInst> {
    let n = insts.len();
    // Dependence edges (from, to, latency); successor order does not matter.
    let mut edges: Vec<(u32, u32, u32)> = Vec::with_capacity(4 * n);
    let mut preds = vec![0u32; n];
    let mut edge = |from: usize, to: usize, latency: u32| {
        edges.push((from as u32, to as u32, latency));
        preds[to] += 1;
    };
    let mut last_def = vec![NO_INST; units.len()];
    // Readers of each unit since its last def, as linked lists of (inst, next).
    let mut reader_head = vec![NO_INST; units.len()];
    let mut readers: Vec<(u32, u32)> = Vec::with_capacity(2 * n);
    let (mut scc_def, mut scc_readers) = (None, Vec::new());
    let (mut barrier, mut since_barrier) = (None, Vec::new());
    // Last write and the reads since, of LDS and of global memory. SMEM
    // reads only kernel arguments and memory the kernel does not write.
    let mut lds = (None, Vec::new());
    let mut vmem = (None, Vec::new());
    for (i, inst) in insts.iter().enumerate() {
        let flags = inst.op.flags();
        if flags & (F_BRANCH | F_END | F_DEF_EXEC | F_USE_VCC | F_SMEM) != 0
            || matches!(inst.op, Opc::SBarrier | Opc::SSendmsg)
            || has_phys(inst)
        {
            // Fences stay in order even with nothing between them, as the
            // `s_cbranch_execz` right after its `s_and_saveexec`.
            if let Some(b) = barrier {
                edge(b, i, 0);
            }
            for j in std::mem::take(&mut since_barrier) {
                edge(j, i, 0);
            }
            barrier = Some(i);
        } else {
            if let Some(b) = barrier {
                edge(b, i, 0);
            }
            since_barrier.push(i);
        }
        let in_order = flags & F_VMEM_LOAD != 0;
        for u in inst.uses.iter().flat_map(|o| units.of(o)) {
            if last_def[u] != NO_INST {
                let d = last_def[u] as usize;
                // VMEM loads write in issue order, so a tied d16 chain does not wait.
                let ordered = in_order && insts[d].op.flags() & F_VMEM_LOAD != 0;
                edge(d, i, if ordered { 0 } else { result_latency(&insts[d]) });
            }
            readers.push((i as u32, reader_head[u]));
            reader_head[u] = readers.len() as u32 - 1;
        }
        for u in inst.defs.iter().flat_map(|o| units.of(o)) {
            if last_def[u] != NO_INST {
                edge(last_def[u] as usize, i, 0);
            }
            let mut r = std::mem::replace(&mut reader_head[u], NO_INST);
            while r != NO_INST {
                let (reader, next) = readers[r as usize];
                if reader as usize != i {
                    edge(reader as usize, i, 0);
                }
                r = next;
            }
            last_def[u] = i as u32;
        }
        if flags & F_USE_SCC != 0 {
            if let Some(d) = scc_def {
                edge(d, i, result_latency(&insts[d]));
            }
            scc_readers.push(i);
        }
        if flags & F_DEF_SCC != 0 {
            if let Some(d) = scc_def {
                edge(d, i, 0);
            }
            for r in std::mem::take(&mut scc_readers) {
                if r != i {
                    edge(r, i, 0);
                }
            }
            scc_def = Some(i);
        }
        let memory = if flags & F_LDS != 0 {
            Some((&mut lds, !inst.defs.is_empty()))
        } else if flags & (F_VMEM_LOAD | F_VMEM_STORE) != 0 {
            Some((&mut vmem, flags & F_VMEM_STORE == 0))
        } else {
            None
        };
        if let Some(((write, reads), is_read)) = memory {
            if let Some(w) = *write {
                edge(w, i, 0);
            }
            if is_read {
                reads.push(i);
            } else {
                for r in std::mem::take(reads) {
                    edge(r, i, 0);
                }
                *write = Some(i);
            }
        }
    }
    let mut succ_start = vec![0u32; n + 1];
    for &(from, _, _) in &edges {
        succ_start[from as usize + 1] += 1;
    }
    for i in 0..n {
        succ_start[i + 1] += succ_start[i];
    }
    let mut succs = vec![(0u32, 0u32); edges.len()];
    let mut fill = succ_start.clone();
    for &(from, to, latency) in &edges {
        succs[fill[from as usize] as usize] = (to, latency);
        fill[from as usize] += 1;
    }
    let mut ready_at = vec![0u32; n];
    let mut done = vec![false; n];
    // Unscheduled instructions whose predecessors have all issued.
    let mut ready = vec![0u64; n.div_ceil(64)];
    for (i, _) in preds.iter().enumerate().filter(|(_, &p)| p == 0) {
        ready[i / 64] |= 1 << (i % 64);
    }
    // Longest latency path from each instruction to the end of the block; a
    // result read only in later blocks counts its own latency.
    let mut height = vec![0u32; n];
    for i in (0..n).rev() {
        let own = if insts[i].defs.is_empty() { 0 } else { result_latency(&insts[i]) };
        height[i] = succs[succ_start[i] as usize..succ_start[i + 1] as usize]
            .iter()
            .map(|&(j, latency)| latency + height[j as usize])
            .fold(own, u32::max);
    }
    let mut order = Vec::with_capacity(n);
    let (mut cycle, mut first) = (0u32, 0usize);
    while order.len() < n {
        while done[first] {
            first += 1;
        }
        // The first unscheduled instruction's predecessors all precede it, so
        // it is always a candidate.
        let mut pick = (first, ready_at[first].max(cycle));
        let (begin, end) = (first + 1, n.min(first + window));
        for w in begin / 64..end.div_ceil(64) {
            let mut bits = ready[w];
            if w == begin / 64 {
                bits &= !0u64 << (begin % 64);
            }
            if end < (w + 1) * 64 {
                bits &= (1u64 << (end % 64)) - 1;
            }
            while bits != 0 {
                let i = w * 64 + bits.trailing_zeros() as usize;
                bits &= bits - 1;
                let start = ready_at[i].max(cycle);
                if (start, std::cmp::Reverse(height[i])) < (pick.1, std::cmp::Reverse(height[pick.0])) {
                    pick = (i, start);
                }
            }
        }
        let (i, start) = pick;
        done[i] = true;
        ready[i / 64] &= !(1 << (i % 64));
        order.push(i);
        cycle = start + u32::from(insts[i].op.fmt() != Fmt::Pseudo || insts[i].op.flags() & F_USE_EXEC != 0);
        for &(j, latency) in &succs[succ_start[i] as usize..succ_start[i + 1] as usize] {
            let j = j as usize;
            preds[j] -= 1;
            ready_at[j] = ready_at[j].max(start + latency);
            if preds[j] == 0 && !done[j] {
                ready[j / 64] |= 1 << (j % 64);
            }
        }
    }
    let mut slots: Vec<Option<MInst>> = insts.into_iter().map(Some).collect();
    order.into_iter().map(|i| slots[i].take().expect("each instruction is scheduled once")).collect()
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum AluKind {
    Valu,
    Trans,
    Salu,
    Other,
}

const VALU_MAX: u8 = 5;
const TRANS_MAX: u8 = 4;
const SALU_CYCLES_MAX: u8 = 4;

/// Outstanding ALU result of one register: cycles until it is ready and how
/// many instructions of its kind issued since, as `s_delay_alu` counts them.
#[derive(Clone, Copy)]
struct AluDelay {
    valu_cycles: u8,
    valu_num: u8,
    trans_cycles: u8,
    trans_num: u8,
    trans_num_valu: u8,
    salu_cycles: u8,
}

impl Default for AluDelay {
    fn default() -> Self {
        AluDelay {
            valu_cycles: 0,
            valu_num: VALU_MAX,
            trans_cycles: 0,
            trans_num: TRANS_MAX,
            trans_num_valu: VALU_MAX,
            salu_cycles: 0,
        }
    }
}

impl AluDelay {
    fn issued(kind: AluKind, cycles: u8) -> AluDelay {
        let mut d = AluDelay::default();
        match kind {
            AluKind::Valu => (d.valu_cycles, d.valu_num) = (cycles, 0),
            AluKind::Trans => (d.trans_cycles, d.trans_num, d.trans_num_valu) = (cycles, 0, 0),
            AluKind::Salu => d.salu_cycles = cycles.min(SALU_CYCLES_MAX),
            AluKind::Other => {}
        }
        d
    }

    fn merge(&mut self, o: &AluDelay) {
        self.valu_cycles = self.valu_cycles.max(o.valu_cycles);
        self.valu_num = self.valu_num.min(o.valu_num);
        self.trans_cycles = self.trans_cycles.max(o.trans_cycles);
        self.trans_num = self.trans_num.min(o.trans_num);
        self.trans_num_valu = self.trans_num_valu.min(o.trans_num_valu);
        self.salu_cycles = self.salu_cycles.max(o.salu_cycles);
    }

    /// Accounts for one more issued instruction; true once nothing is left
    /// to wait for.
    fn advance(&mut self, kind: AluKind, cycles: u8) -> bool {
        let mut done = true;
        self.valu_num = self.valu_num.saturating_add((kind == AluKind::Valu) as u8);
        if self.valu_num >= VALU_MAX || self.valu_cycles <= cycles {
            (self.valu_num, self.valu_cycles) = (VALU_MAX, 0);
        } else {
            self.valu_cycles -= cycles;
            done = false;
        }
        self.trans_num = self.trans_num.saturating_add((kind == AluKind::Trans) as u8);
        self.trans_num_valu = self.trans_num_valu.saturating_add((kind == AluKind::Valu) as u8);
        if self.trans_num >= TRANS_MAX || self.trans_cycles <= cycles {
            (self.trans_num, self.trans_num_valu, self.trans_cycles) = (TRANS_MAX, VALU_MAX, 0);
        } else {
            self.trans_cycles -= cycles;
            done = false;
        }
        if self.salu_cycles <= cycles {
            self.salu_cycles = 0;
        } else {
            self.salu_cycles -= cycles;
            done = false;
        }
        done
    }

    /// The `s_delay_alu` operand: instid0 in bits 3:0, instid1 in 10:7.
    fn encode(&self) -> u32 {
        let mut imm = 0;
        if self.trans_num < TRANS_MAX {
            imm |= 4 + self.trans_num as u32;
        }
        if self.valu_num < VALU_MAX && self.valu_num <= self.trans_num_valu {
            imm |= if imm & 0xf != 0 { (self.valu_num as u32) << 7 } else { self.valu_num as u32 };
        }
        if self.salu_cycles != 0 && imm & 0x780 == 0 {
            let id = self.salu_cycles as u32 + 8;
            imm |= if imm & 0xf != 0 { id << 7 } else { id };
        }
        imm
    }
}

/// Keys of the registers `o` names: SGPR numbers (VCC, M0 and EXEC at their
/// operand codes), VGPR numbers with bit 16 set.
fn alu_units(o: &Opnd) -> impl Iterator<Item = u32> {
    let (start, n, vgpr) = match *o {
        Opnd::P(Phys::V(r), n) => (r as u32, n as u32, true),
        Opnd::P(Phys::S(r), n) => (r as u32, n as u32, false),
        Opnd::P(Phys::VccLo, _) => (106, 1, false),
        Opnd::P(Phys::M0, _) => (125, 1, false),
        Opnd::P(Phys::ExecLo, _) => (126, 1, false),
        _ => (0, 0, false),
    };
    (start..start + n).map(move |u| if vgpr { 0x10000 | u } else { u })
}

/// Outstanding ALU results by register key: a table for the registers that
/// allocation hands out, with the occupied slots listed, and a map for any
/// other key.
struct AluState {
    slots: Vec<Option<AluDelay>>,
    active: Vec<u16>,
    others: FxHashMap<u32, AluDelay>,
}

impl AluState {
    fn new() -> Self {
        AluState { slots: vec![None; 512], active: Vec::new(), others: FxHashMap::default() }
    }

    fn slot(key: u32) -> Option<usize> {
        match key {
            0..=0xff => Some(key as usize),
            0x10000..=0x100ff => Some(256 + (key & 0xff) as usize),
            _ => None,
        }
    }

    fn take(&mut self, key: u32) -> Option<AluDelay> {
        let Some(u) = Self::slot(key) else {
            return self.others.remove(&key);
        };
        let d = self.slots[u].take();
        if d.is_some() {
            let i = self.active.iter().position(|&a| a as usize == u).expect("listed");
            self.active.swap_remove(i);
        }
        d
    }

    fn set(&mut self, key: u32, d: AluDelay) {
        let Some(u) = Self::slot(key) else {
            self.others.insert(key, d);
            return;
        };
        if self.slots[u].replace(d).is_none() {
            self.active.push(u as u16);
        }
    }

    fn clear(&mut self) {
        for &u in &self.active {
            self.slots[u as usize] = None;
        }
        self.active.clear();
        self.others.clear();
    }

    fn advance(&mut self, kind: AluKind, cycles: u8) {
        let slots = &mut self.slots;
        self.active.retain(|&u| {
            let done = slots[u as usize].as_mut().expect("listed").advance(kind, cycles);
            if done {
                slots[u as usize] = None;
            }
            !done
        });
        self.others.retain(|_, d| !d.advance(kind, cycles));
    }
}

/// Inserts `s_delay_alu` before ALU instructions that read a result still in
/// flight, so the wave yields its issue slot instead of stalling in the
/// pipeline. The model follows LLVM's gfx11 one: VALU results take 5 cycles,
/// transcendental 10, SALU 2; memory instructions wait for VALU results
/// themselves. A second dependency within five instructions is packed into
/// the previous `s_delay_alu`.
pub fn insert_delay_alu(f: &mut MFunc) {
    let mut state = AluState::new();
    for block in f.blocks.iter_mut() {
        state.clear();
        let mut out = Vec::with_capacity(block.insts.len() * 5 / 4);
        let mut open: Option<usize> = None;
        // Non-pseudo instructions emitted after the open `s_delay_alu`.
        let mut since_open = 0usize;
        for inst in std::mem::take(&mut block.insts) {
            let fmt = inst.op.fmt();
            if fmt == Fmt::Pseudo {
                out.push(inst);
                continue;
            }
            let kind = match fmt {
                Fmt::Vop1 | Fmt::Vop2 | Fmt::Vop3 | Fmt::Vop3b | Fmt::Vopc | Fmt::Vop3p | Fmt::Vopd => {
                    if inst.op.flags() & F_TRANS != 0 {
                        AluKind::Trans
                    } else {
                        AluKind::Valu
                    }
                }
                Fmt::Sop1 | Fmt::Sop2 | Fmt::Sopc | Fmt::Sopk | Fmt::Sopp => AluKind::Salu,
                _ => AluKind::Other,
            };
            if matches!(fmt, Fmt::Ds | Fmt::Global | Fmt::Mubuf) {
                state.clear();
            } else if kind != AluKind::Other {
                let mut need = AluDelay::default();
                for o in &inst.uses {
                    for u in alu_units(o) {
                        if let Some(d) = state.take(u) {
                            need.merge(&d);
                        }
                    }
                }
                let imm = need.encode();
                if imm != 0 {
                    match open {
                        Some(i) if imm & 0x780 == 0 && since_open < 6 => {
                            out[i].imm |= ((imm << 7) | ((since_open as u32) << 4)) as i64;
                            open = None;
                        }
                        _ => {
                            out.push(MInst::new(Opc::SDelayAlu, [], []).with_imm(imm as i64));
                            open = (imm & 0x780 == 0).then_some(out.len() - 1);
                            since_open = 0;
                        }
                    }
                }
            }
            if kind != AluKind::Other {
                let latency = match kind {
                    AluKind::Trans => 10,
                    AluKind::Valu => 5,
                    _ => 2,
                };
                for o in &inst.defs {
                    for u in alu_units(o) {
                        state.set(u, AluDelay::issued(kind, latency));
                    }
                }
            }
            let cycles = if inst.op == Opc::SNop { inst.imm as u8 + 1 } else { 1 };
            state.advance(kind, cycles);
            out.push(inst);
            since_open += 1;
        }
        block.insts = out;
    }
}

/// Removes copies and implicit defs of virtual registers nothing reads.
pub fn remove_dead_copies(f: &mut MFunc) {
    let units = UnitIndex::new(f.blocks.iter().flat_map(|b| b.insts.iter()));
    let mut used = vec![false; units.len()];
    loop {
        used.fill(false);
        for block in &f.blocks {
            for inst in &block.insts {
                for o in &inst.uses {
                    for u in units.of(o) {
                        used[u] = true;
                    }
                }
            }
        }
        let dead = |inst: &MInst| {
            matches!(inst.op, Opc::Copy | Opc::ImplicitDef)
                && inst.defs.iter().all(|d| matches!(d, Opnd::R(_) if units.of(d).all(|u| !used[u])))
        };
        let mut changed = false;
        for block in f.blocks.iter_mut() {
            let before = block.insts.len();
            block.insts.retain(|inst| !dead(inst));
            changed |= block.insts.len() != before;
        }
        if !changed {
            return;
        }
    }
}

/// Expands `Copy` and `PackF16` into real instructions.
pub fn expand_pseudos(f: &mut MFunc) -> Result<(), CompileError> {
    for block in f.blocks.iter_mut() {
        let mut out = Vec::with_capacity(block.insts.len());
        for inst in block.insts.drain(..) {
            match inst.op {
                Opc::Copy => {
                    let dst = inst.defs[0];
                    let src = inst.uses[0];
                    let (dst_v, d, n) = phys_base(&dst).ok_or_else(|| internal("copy to a non-register"))?;
                    let src_reg = phys_base(&src);
                    if !dst_v && src_reg.is_some_and(|(v, _, _)| v) {
                        return Err(internal("copy from a VGPR to an SGPR"));
                    }
                    // Copy upward ranges from the top so overlapping moves stay correct.
                    let descending = matches!(src_reg, Some((sv, s, _)) if sv == dst_v && s < d);
                    let order: Vec<u8> = if descending { (0..n).rev().collect() } else { (0..n).collect() };
                    for k in order {
                        let s = if src_reg.is_some() { dword(&src, k) } else { src };
                        let dd = dword(&dst, k);
                        if dd == s {
                            continue;
                        }
                        let op = if dst_v { Opc::VMovB32 } else { Opc::SMovB32 };
                        out.push(MInst::new(op, [dd], [s]));
                    }
                }
                Opc::PackF16 => {
                    let dst = inst.defs[0];
                    out.push(MInst::new(Opc::VCvtF16F32, [dst], [inst.uses[0]]));
                    out.push(MInst::new(Opc::VCvtF16F32, [dst], [inst.uses[1]]).with_mods(M_OPSEL_DST));
                }
                Opc::ImplicitDef => {}
                _ => out.push(inst),
            }
        }
        block.insts = out;
    }
    Ok(())
}

const NREGS: usize = 256 + 128;
const NONE: u8 = u8::MAX;

#[derive(Clone, PartialEq)]
struct WaitState {
    /// Per register: number of later ops on the counter (wait `cnt(d)`).
    vm: [u8; NREGS],
    lgkm: [u8; NREGS],
    vm_any: bool,
    lgkm_any: bool,
    smem: bool,
}

impl WaitState {
    fn new() -> Self {
        WaitState {
            vm: [NONE; NREGS],
            lgkm: [NONE; NREGS],
            vm_any: false,
            lgkm_any: false,
            smem: false,
        }
    }

    fn merge(&mut self, other: &WaitState) {
        for (a, b) in self.vm.iter_mut().zip(&other.vm) {
            *a = (*a).min(*b);
        }
        for (a, b) in self.lgkm.iter_mut().zip(&other.lgkm) {
            *a = (*a).min(*b);
        }
        self.vm_any |= other.vm_any;
        self.lgkm_any |= other.lgkm_any;
        self.smem |= other.smem;
    }
}

/// One counter of a `WaitState` within a block. An op on the counter bumps
/// every register at once by advancing `now`; a register's count is the ops
/// since its `stamp`, saturating at 62. A wait drops every register with at
/// least the waited count, which are those stamped before `floor`.
struct Counter {
    now: u32,
    floor: u32,
    stamp: [u32; NREGS],
}

impl Counter {
    fn new(counts: &[u8; NREGS]) -> Self {
        let now = 63;
        let mut stamp = [0; NREGS];
        for (s, &d) in stamp.iter_mut().zip(counts) {
            if d != NONE {
                *s = now - d as u32;
            }
        }
        Counter { now, floor: 1, stamp }
    }

    fn get(&self, k: usize) -> u8 {
        let s = self.stamp[k];
        if s < self.floor { NONE } else { (self.now - s).min(62) as u8 }
    }

    fn counts(&self) -> [u8; NREGS] {
        std::array::from_fn(|k| self.get(k))
    }

    fn wait(&mut self, need: u8) {
        self.floor = self.floor.max(self.now + 1 - need as u32);
    }

    fn issue(&mut self, defs: &[Opnd]) {
        self.now += 1;
        for k in defs.iter().flat_map(reg_keys) {
            self.stamp[k] = self.now;
        }
    }
}

struct Waits {
    vm: Counter,
    lgkm: Counter,
    vm_any: bool,
    lgkm_any: bool,
    smem: bool,
}

impl Waits {
    fn new(state: &WaitState) -> Self {
        Waits {
            vm: Counter::new(&state.vm),
            lgkm: Counter::new(&state.lgkm),
            vm_any: state.vm_any,
            lgkm_any: state.lgkm_any,
            smem: state.smem,
        }
    }

    fn state(&self) -> WaitState {
        WaitState {
            vm: self.vm.counts(),
            lgkm: self.lgkm.counts(),
            vm_any: self.vm_any,
            lgkm_any: self.lgkm_any,
            smem: self.smem,
        }
    }
}

fn reg_keys(o: &Opnd) -> impl Iterator<Item = usize> {
    let (start, n) = match o {
        Opnd::P(Phys::V(r), n) => (*r as usize, *n as usize),
        Opnd::P(Phys::S(r), n) => (256 + *r as usize, *n as usize),
        _ => (0, 0),
    };
    start..start + n
}

/// One step of the wait model; returns the wait to insert before `inst`.
fn step(state: &mut Waits, inst: &MInst) -> Option<(u8, u8)> {
    let flags = inst.op.flags();
    let vmem_load = flags & F_VMEM_LOAD != 0;
    let mut need_vm = NONE;
    let mut need_lgkm = NONE;
    for o in inst.uses.iter().chain(inst.defs.iter()) {
        // VMEM loads write VGPRs in issue order, and a d16 load merges into
        // the half it keeps when it writes, so neither a load's destination
        // nor that tied source waits for an earlier VMEM load.
        let in_order = vmem_load && inst.defs.contains(o);
        for k in reg_keys(o) {
            if !in_order {
                need_vm = need_vm.min(state.vm.get(k));
            }
            need_lgkm = need_lgkm.min(state.lgkm.get(k));
        }
    }
    if inst.op == Opc::SBarrier {
        if state.vm_any {
            need_vm = 0;
        }
        if state.lgkm_any {
            need_lgkm = 0;
        }
    }
    if need_lgkm != NONE && state.smem {
        need_lgkm = 0;
    }
    let wait = if need_vm != NONE || need_lgkm != NONE {
        if need_vm != NONE {
            state.vm.wait(need_vm);
            if need_vm == 0 {
                state.vm_any = false;
            }
        }
        if need_lgkm != NONE {
            state.lgkm.wait(need_lgkm);
            if need_lgkm == 0 {
                state.lgkm_any = false;
                state.smem = false;
            }
        }
        Some((need_vm, need_lgkm))
    } else {
        None
    };
    if flags & F_VMEM_LOAD != 0 {
        state.vm.issue(&inst.defs);
        state.vm_any = true;
    }
    if flags & (F_LDS | F_SMEM) != 0 {
        state.lgkm.issue(&inst.defs);
        state.lgkm_any = true;
        if flags & F_SMEM != 0 {
            state.smem = true;
        }
    }
    wait
}

fn waitcnt_imm(vm: u8, lgkm: u8) -> i64 {
    let vm = if vm == NONE { 0x3f } else { vm as i64 };
    let lgkm = if lgkm == NONE { 0x3f } else { lgkm as i64 };
    (vm << 10) | (lgkm << 4) | 0x7
}

/// Inserts `s_waitcnt` before every instruction that needs an outstanding
/// load, and before barriers.
pub fn insert_waitcnt(f: &mut MFunc) {
    let n = f.blocks.len();
    let succs: Vec<Vec<usize>> = (0..n).map(|b| f.successors(b)).collect();
    let mut entry: Vec<Option<WaitState>> = vec![None; n];
    entry[0] = Some(WaitState::new());
    loop {
        let mut changed = false;
        for b in 0..n {
            let Some(start) = &entry[b] else {
                continue;
            };
            let mut waits = Waits::new(start);
            for inst in &f.blocks[b].insts {
                step(&mut waits, inst);
            }
            let state = waits.state();
            for &s in &succs[b] {
                let merged = match &entry[s] {
                    None => state.clone(),
                    Some(old) => {
                        let mut m = old.clone();
                        m.merge(&state);
                        m
                    }
                };
                if entry[s].as_ref() != Some(&merged) {
                    entry[s] = Some(merged);
                    changed = true;
                }
            }
        }
        if !changed {
            break;
        }
    }
    let wait_all = super::debug_flag("waitall");
    for b in 0..n {
        let mut waits = Waits::new(entry[b].as_ref().unwrap_or(&WaitState::new()));
        let insts = std::mem::take(&mut f.blocks[b].insts);
        let mut out = Vec::with_capacity(insts.len() + 8);
        for inst in insts {
            if let Some((vm, lgkm)) = step(&mut waits, &inst) {
                out.push(MInst::new(Opc::SWaitcnt, [], []).with_imm(waitcnt_imm(vm, lgkm)));
            }
            let loads = inst.op.flags() & (F_VMEM_LOAD | F_LDS | F_SMEM) != 0;
            out.push(inst);
            if wait_all && loads {
                out.push(MInst::new(Opc::SWaitcnt, [], []).with_imm(waitcnt_imm(0, 0)));
            }
        }
        f.blocks[b].insts = out;
    }
}

fn is_valu(op: Opc) -> bool {
    matches!(
        op.fmt(),
        Fmt::Vop1 | Fmt::Vop2 | Fmt::Vop3 | Fmt::Vop3b | Fmt::Vopc | Fmt::Vop3p | Fmt::Vopd
    )
}

fn ranges_overlap(a: &Opnd, b: &Opnd) -> bool {
    match (phys_base(a), phys_base(b)) {
        (Some((av, ar, an)), Some((bv, br, bn))) => av == bv && ar < br + bn as u16 && br < ar + an as u16,
        _ => false,
    }
}

/// A WMMA must not read, as A or B, the result of the immediately preceding
/// VALU if that was a WMMA: insert `v_nop`.
pub fn fix_wmma_hazards(f: &mut MFunc) {
    let n = f.blocks.len();
    let mut preds: Vec<Vec<usize>> = vec![Vec::new(); n];
    for b in 0..n {
        for s in f.successors(b) {
            preds[s].push(b);
        }
    }
    let last_valu: Vec<Option<MInst>> = f
        .blocks
        .iter()
        .map(|b| b.insts.iter().rev().find(|i| is_valu(i.op)).cloned())
        .collect();
    for b in 0..n {
        let insts = std::mem::take(&mut f.blocks[b].insts);
        let mut out: Vec<MInst> = Vec::with_capacity(insts.len());
        for inst in insts {
            if inst.op.flags() & F_WMMA != 0 {
                let prev = out.iter().rev().find(|i| is_valu(i.op));
                let hazard = |p: &MInst| {
                    p.op.flags() & F_WMMA != 0
                        && (ranges_overlap(&p.defs[0], &inst.uses[0]) || ranges_overlap(&p.defs[0], &inst.uses[1]))
                };
                let needs = match prev {
                    Some(p) => hazard(p),
                    None => preds[b].iter().any(|&p| last_valu[p].as_ref().is_none_or(&hazard)),
                };
                if needs {
                    out.push(MInst::new(Opc::VNop, [], []));
                }
            }
            out.push(inst);
        }
        f.blocks[b].insts = out;
    }
}

/// LLVM releases VGPRs before `s_endpgm` when they limit occupancy and stores
/// may be outstanding.
pub fn add_vgpr_release(f: &mut MFunc, num_vgprs: u32) {
    let has_stores = f
        .blocks
        .iter()
        .any(|b| b.insts.iter().any(|i| i.op.flags() & F_VMEM_STORE != 0));
    if num_vgprs <= 96 || !has_stores {
        return;
    }
    for block in f.blocks.iter_mut() {
        let mut out = Vec::with_capacity(block.insts.len() + 2);
        for inst in block.insts.drain(..) {
            if inst.op == Opc::SEndpgm {
                out.push(MInst::new(Opc::SNop, [], []));
                out.push(MInst::new(Opc::SSendmsg, [], []).with_imm(3));
            }
            out.push(inst);
        }
        block.insts = out;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn r(reg: u32) -> Opnd {
        Opnd::R(VRef { reg, off: 0, n: 1 })
    }

    fn add(dst: u32, src: u32) -> MInst {
        MInst::new(Opc::VAddNcU32, [r(dst)], [r(src), Opnd::Imm(1)])
    }

    fn ops(insts: &[MInst]) -> Vec<(Opc, Vec<Opnd>)> {
        insts.iter().map(|i| (i.op, i.defs.to_vec())).collect()
    }

    fn schedule_block(insts: Vec<MInst>, window: usize) -> Vec<MInst> {
        let units = UnitIndex::new(insts.iter());
        super::schedule_block(insts, window, &units)
    }

    #[test]
    fn schedule_interleaves_independent_chains() {
        let out = schedule_block(vec![add(1, 0), add(2, 1), add(4, 3), add(5, 4)], 16);
        assert_eq!(ops(&out), ops(&[add(1, 0), add(4, 3), add(2, 1), add(5, 4)]));
    }

    #[test]
    fn schedule_keeps_scc_and_lds_order() {
        // The s_add would clobber SCC between the compare and the select.
        let scc = vec![
            MInst::new(Opc::SCmpLtI32, [], [r(0), r(1)]),
            MInst::new(Opc::SCselectB32, [r(2)], [r(3), r(4)]),
            MInst::new(Opc::SAddI32, [r(5)], [r(6), r(7)]),
        ];
        assert_eq!(ops(&schedule_block(scc.clone(), 16)), ops(&scc));
        // The load does not wait on the add, but must not pass the store.
        let lds = vec![
            add(10, 11),
            MInst::new(Opc::DsStoreB32, [], [r(12), r(10)]),
            MInst::new(Opc::DsLoadB32, [r(13)], [r(14)]),
        ];
        assert_eq!(ops(&schedule_block(lds.clone(), 16)), ops(&lds));
    }

    #[test]
    fn schedule_keeps_the_skip_branch_after_its_exec_write() {
        let exec = Opnd::P(Phys::ExecLo, 1);
        let block = vec![
            add(1, 0),
            MInst::new(Opc::VCmp(0x41), [r(2)], [r(1), Opnd::Imm(0)]),
            MInst::new(Opc::SAndSaveexecB32, [r(3), exec], [r(2), exec]),
            MInst::new(Opc::SCbranchExecz, [], []),
        ];
        assert_eq!(ops(&schedule_block(block.clone(), 16)), ops(&block));
    }

    #[test]
    fn schedule_starts_the_longest_path_first() {
        // Both adds can issue at once; the second starts the longer path (it
        // feeds a global load), so it goes first.
        let load = MInst::new(Opc::GLoadB32, [r(3)], [r(2)]);
        let insts = vec![add(1, 0), add(2, 10), load.clone(), add(4, 3)];
        let out = schedule_block(insts, 16);
        assert_eq!(ops(&out), ops(&[add(2, 10), add(1, 0), load, add(4, 3)]));
    }

    #[test]
    fn schedule_keeps_barriers_as_fences() {
        // The chain after the barrier is the longest path, but nothing passes
        // the barrier in either direction.
        let barrier = MInst::new(Opc::SBarrier, [], []);
        let load = MInst::new(Opc::GLoadB32, [r(3)], [r(2)]);
        let insts = vec![add(1, 0), barrier, add(2, 10), load, add(4, 3)];
        assert_eq!(ops(&schedule_block(insts.clone(), 16)), ops(&insts));
    }

    #[test]
    fn schedule_moves_at_most_window_places() {
        // Everything after the first add is ready, but only the next three
        // instructions may pass the stalled one.
        let mut insts = vec![add(1, 0), add(2, 1)];
        insts.extend((10..20).map(|k| add(k, 100 + k)));
        let out = schedule_block(insts, 4);
        assert_eq!(ops(&out[4..5]), ops(&[add(2, 1)]));
    }

    #[test]
    fn lazy_counter_matches_explicit_counts() {
        let mut seed = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = |n: u32| {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (seed >> 33) as u32 % n
        };
        for _ in 0..50 {
            let mut counts = [NONE; NREGS];
            for d in counts.iter_mut().take(64) {
                *d = match next(4) {
                    0 => NONE,
                    1 => 62,
                    _ => next(62) as u8,
                };
            }
            let mut lazy = Counter::new(&counts);
            for _ in 0..300 {
                if next(2) == 0 {
                    for d in counts.iter_mut() {
                        if *d != NONE {
                            *d = (*d + 1).min(62);
                        }
                    }
                    let reg = next(64);
                    let n = 1 + next(4) as u8;
                    for d in &mut counts[reg as usize..reg as usize + n as usize] {
                        *d = 0;
                    }
                    lazy.issue(&[Opnd::P(Phys::V(reg as u16), n)]);
                } else {
                    let pending: Vec<u8> = counts.iter().copied().filter(|&d| d != NONE).collect();
                    let need = if pending.is_empty() || next(5) == 0 { 0 } else { pending[next(pending.len() as u32) as usize] };
                    for d in counts.iter_mut() {
                        if *d != NONE && *d >= need {
                            *d = NONE;
                        }
                    }
                    lazy.wait(need);
                }
                assert_eq!(lazy.counts(), counts);
            }
        }
    }
}
