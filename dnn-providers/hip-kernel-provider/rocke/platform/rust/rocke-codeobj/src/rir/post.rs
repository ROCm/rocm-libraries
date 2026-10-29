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
/// may pass it, whether it writes LDS, its global memory tag, and 64-bit
/// signatures of the virtual-register units it defines and uses (a shared
/// unit sets a common bit).
#[derive(Clone, Copy)]
struct SinkFacts {
    flags: u32,
    fence: bool,
    lds_write: bool,
    mem: u16,
    defs: u64,
    uses: u64,
    /// A VMEM or LDS load with a result, and the dwords it loads, as
    /// `sink_load_uses` counts them.
    load: bool,
    load_dwords: u16,
}

fn unit_signature(opnds: &[Opnd]) -> u64 {
    opnds.iter().flat_map(vunits).fold(0, |s, (reg, k)| s | 1 << (reg.wrapping_mul(7).wrapping_add(k as u32) % 64))
}

impl SinkFacts {
    fn of(x: &MInst) -> SinkFacts {
        let flags = x.op.flags();
        SinkFacts {
            flags,
            fence: flags & (F_BRANCH | F_END | F_DEF_EXEC | F_FENCE) != 0
                || matches!(x.op, Opc::SBarrier | Opc::SSendmsg)
                || has_phys(x),
            // LDS reads (loads, swizzles) define a register; LDS writes do not.
            lds_write: flags & F_LDS != 0 && x.defs.is_empty(),
            mem: x.mem,
            defs: unit_signature(&x.defs),
            uses: unit_signature(&x.uses),
            load: flags & (F_VMEM_LOAD | F_LDS) != 0 && !x.defs.is_empty(),
            load_dwords: x.defs.iter().map(|d| vunits(d).count() as u16).sum(),
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
    let aliased = super::alias::may_alias(cs.mem, xs.mem);
    if (cf & F_LDS != 0 && xf & F_LDS != 0 && (cs.lds_write || xs.lds_write))
        || (cf & F_VMEM_STORE != 0 && xf & vmem != 0 && aliased)
        || (cf & vmem != 0 && xf & F_VMEM_STORE != 0 && aliased)
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
    // gfx950 has its own opcodes and operand rule (`target::gfx9::rules::fuses`).
    let gfx9 = f.target == crate::target::Target::Gfx950;
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
                if gfx9 && !crate::target::gfx9::rules::fuses(op, &srcs, vregs) {
                    return None;
                }
                let moved = insts[p + 1..x]
                    .iter()
                    .any(|k| k.op.flags() & F_DEF_EXEC != 0 || shares_vreg(&k.defs, &pi.uses));
                if moved || !scalar_sources_fit(&srcs, vregs) {
                    return None;
                }
                Some((p, MInst::new(op, xi.defs, srcs)))
            });
            let mix = fma_mix_form(insts, x, &def_at, (&uses, &defs), vregs);
            let mix = mix.filter(|(m, _)| !gfx9 || crate::target::gfx9::rules::fuses(m.op, &m.uses, vregs));
            if let Some((inst, folded)) = mix {
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
    let fixed = F_DEF_SCC | F_USE_SCC | F_USE_VCC | F_DEF_EXEC | F_BRANCH | F_END | F_SMEM | F_FENCE;
    let units = UnitIndex::new(f.blocks.iter().flat_map(|b| b.insts.iter()));
    // Block number + 1 of the units loaded so far in that block.
    let mut loaded = vec![0u32; units.len()];
    for (b, block) in f.blocks.iter_mut().enumerate() {
        let stamp = b as u32 + 1;
        let insts = &mut block.insts;
        let mut candidate = vec![false; insts.len()];
        for (i, inst) in insts.iter().enumerate() {
            // Loads write in issue order, so a d16 load's tied destination
            // does not stall it.
            let load = inst.op.flags() & (F_VMEM_LOAD | F_LDS) != 0;
            let in_order = inst.op.flags() & F_VMEM_LOAD != 0 || lds_merge(inst);
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
            let cs = facts[i];
            let mut j = i + 1;
            let mut passed = 0u32;
            while j < insts.len() && may_sink_past(c, &cs, &insts[j], &facts[j]) {
                if facts[j].load {
                    // Delayed loads keep their order, so each still waits
                    // on the oldest outstanding load.
                    if cs.load && candidate[j] {
                        break;
                    }
                    passed += u32::from(facts[j].load_dwords);
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
    super::gen_isa::result_latency(inst.op)
}

/// Reorders each block so that instructions do not wait on results still in
/// flight: each step issues, among the unscheduled instructions at most
/// `window` places after the first, one whose predecessors have issued and
/// that can start soonest; of those, the one with the longest latency path
/// to the end of the block. An instruction moves at most `window` places,
/// which bounds the change in register pressure.
pub fn schedule_window(f: &mut MFunc, window: usize) {
    let units = UnitIndex::new(f.blocks.iter().flat_map(|b| b.insts.iter()));
    let target = f.target;
    for block in f.blocks.iter_mut() {
        let insts = std::mem::take(&mut block.insts);
        block.insts = schedule_block(insts, window, &units, target);
    }
}

const NO_INST: u32 = u32::MAX;

fn schedule_block(insts: Vec<MInst>, window: usize, units: &UnitIndex, target: crate::target::Target) -> Vec<MInst> {
    let order = block_dag(&insts, units, target, &mut DagScratch::default()).order(window);
    order.iter().map(|&i| insts[i as usize].clone()).collect()
}

/// Tables `block_dag` fills for a block, kept for the next block of the
/// function: the per-unit ones are back to unset for every unit when it
/// returns.
#[derive(Default)]
struct DagScratch {
    edges: Vec<(u32, u32, u32)>,
    last_def: Vec<u32>,
    reader_head: Vec<u32>,
    readers: Vec<(u32, u32)>,
}

/// The dependence graph `schedule_block` schedules a block by: each
/// instruction's successors with their latencies (in no particular order),
/// its predecessor count, its longest latency path to the end of the block,
/// and whether it takes an issue cycle. It depends on the block, not on the
/// window, so one graph serves every window.
pub(super) struct BlockDag {
    succ_start: Vec<u32>,
    succs: Vec<(u32, u32)>,
    preds: Vec<u32>,
    height: Vec<u32>,
    issues: Vec<bool>,
    /// Per instruction, the slots it holds gfx950's matrix pipe for; empty
    /// when the block has no MFMA.
    pipe: Vec<u8>,
}

fn block_dag(insts: &[MInst], units: &UnitIndex, target: crate::target::Target, scratch: &mut DagScratch) -> BlockDag {
    let result_latency = |inst: &MInst| match target {
        crate::target::Target::Gfx1151 => result_latency(inst),
        crate::target::Target::Gfx950 => crate::target::gfx9::model::result_latency(inst),
    };
    let mut mfma = match target {
        crate::target::Target::Gfx1151 => None,
        crate::target::Target::Gfx950 => crate::target::gfx9::sched::MfmaEdges::new(insts),
    };
    let mut m0 = match target {
        crate::target::Target::Gfx1151 => None,
        crate::target::Target::Gfx950 => crate::target::gfx9::m0_order::M0Order::new(insts),
    };
    let n = insts.len();
    let DagScratch { edges, last_def, reader_head, readers } = scratch;
    // Dependence edges (from, to, latency); successor order does not matter.
    edges.clear();
    edges.reserve(4 * n);
    let mut preds = vec![0u32; n];
    let mut edge = |from: usize, to: usize, latency: u32| {
        edges.push((from as u32, to as u32, latency));
        preds[to] += 1;
    };
    if last_def.len() < units.len() {
        last_def.resize(units.len(), NO_INST);
        reader_head.resize(units.len(), NO_INST);
    }
    // Readers of each unit since its last def, as linked lists of (inst, next).
    readers.clear();
    readers.reserve(2 * n);
    let (mut scc_def, mut scc_readers) = (None, Vec::new());
    let (mut barrier, mut since_barrier) = (None, Vec::new());
    // Last write and the reads since, of LDS and of global memory (by
    // `alias` tag). SMEM reads only kernel arguments and memory the kernel
    // does not write.
    let mut lds = (None, Vec::new());
    let mut vmem = super::alias::GlobalOrder::default();
    for (i, inst) in insts.iter().enumerate() {
        let unfenced = m0.is_some() && crate::target::gfx9::m0_order::unfenced(inst);
        let flags = inst.op.flags() & if unfenced { !F_FENCE } else { !0 };
        if flags & (F_BRANCH | F_END | F_DEF_EXEC | F_USE_VCC | F_SMEM | F_FENCE) != 0
            || matches!(inst.op, Opc::SBarrier | Opc::SSendmsg)
            || has_phys(inst) && !unfenced
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
        // VMEM loads, and separately DS operations, write their results in
        // issue order, so a d16 load's tied half does not wait on one of its kind.
        let in_order = if flags & F_VMEM_LOAD != 0 {
            F_VMEM_LOAD
        } else if lds_merge(inst) {
            F_LDS
        } else {
            0
        };
        for (k, o) in inst.uses.iter().enumerate() {
            let tied = inst.defs.contains(o);
            for u in units.of(o) {
                if last_def[u] != NO_INST {
                    let d = last_def[u] as usize;
                    let ordered = tied && insts[d].op.flags() & in_order != 0;
                    let latency = match mfma.as_mut().and_then(|m| m.raw(insts, d, i, k)) {
                        _ if ordered => 0,
                        Some(latency) => latency,
                        None => result_latency(&insts[d]),
                    };
                    edge(d, i, latency);
                }
                readers.push((i as u32, reader_head[u]));
                reader_head[u] = readers.len() as u32 - 1;
            }
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
        if let Some(m) = m0.as_mut() {
            m.order(insts, i, &result_latency, &mut edge);
        }
        let memory = if flags & F_LDS != 0 {
            Some((&mut lds, !inst.defs.is_empty()))
        } else if flags & (F_VMEM_LOAD | F_VMEM_STORE) != 0 {
            vmem.access(i as u32, inst.mem, flags & F_VMEM_STORE == 0, |from| edge(from as usize, i, 0));
            None
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
        if unfenced && crate::target::gfx9::lds_dma::is_dma(inst.op) {
            vmem.access(i as u32, super::alias::ANY, true, |from| edge(from as usize, i, 0));
        }
    }
    for u in insts.iter().flat_map(|inst| inst.defs.iter().chain(&inst.uses)).flat_map(|o| units.of(o)) {
        last_def[u] = NO_INST;
        reader_head[u] = NO_INST;
    }
    let mut succ_start = vec![0u32; n + 1];
    for &(from, _, _) in edges.iter() {
        succ_start[from as usize + 1] += 1;
    }
    for i in 0..n {
        succ_start[i + 1] += succ_start[i];
    }
    let mut succs = vec![(0u32, 0u32); edges.len()];
    let mut fill = succ_start.clone();
    for &(from, to, latency) in edges.iter() {
        succs[fill[from as usize] as usize] = (to, latency);
        fill[from as usize] += 1;
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
    let issues = insts.iter().map(|inst| inst.op.fmt() != Fmt::Pseudo || inst.op.flags() & F_USE_EXEC != 0).collect();
    let pipe = mfma.map_or_else(Vec::new, |_| insts.iter().map(|inst| crate::target::gfx9::sched::pipe_slots(inst.op)).collect());
    BlockDag { succ_start, succs, preds, height, issues, pipe }
}

impl BlockDag {
    /// The order `schedule_block` issues the block in with `window`: each
    /// step takes the first unscheduled instruction or a ready one less than
    /// `window` places after it, whichever starts soonest, then has the
    /// longest path, then comes first.
    fn order(&self, window: usize) -> Vec<u32> {
        if self.pipe.is_empty() {
            self.order_with::<false>(window)
        } else {
            self.order_with::<true>(window)
        }
    }

    /// The rank of MFMA `i`, `k` with its start raised to `cycle`, when the
    /// matrix pipe frees at `free`: it starts no sooner, and if it can start
    /// at `cycle` it goes before any other instruction that can.
    fn piped(&self, i: usize, k: u64, free: u32, cycle: u32) -> u64 {
        if self.pipe[i] == 0 {
            return k;
        }
        let k = k.max(u64::from(free) << 32 | (k & 0xffff_ffff));
        if k >> 32 == u64::from(cycle) {
            k & !(1 << 31)
        } else {
            k
        }
    }

    /// `order`, with the matrix pipe of `pipe` when `PIPE`.
    fn order_with<const PIPE: bool>(&self, window: usize) -> Vec<u32> {
        let n = self.preds.len();
        let mut preds = self.preds.clone();
        // Each instruction's rank, smallest first, as one integer: the cycle
        // its operands are ready at, then its path, longest first. Compared
        // with its start raised to the current cycle.
        let mut key: Vec<u64> = self.height.iter().map(|&h| u64::from(!h)).collect();
        let mut done = vec![false; n];
        // Unscheduled instructions whose predecessors have all issued.
        let mut ready = vec![0u64; n.div_ceil(64)];
        for (i, _) in preds.iter().enumerate().filter(|(_, &p)| p == 0) {
            ready[i / 64] |= 1 << (i % 64);
        }
        let mut order = Vec::with_capacity(n);
        let (mut cycle, mut first) = (0u32, 0usize);
        // The cycle the matrix pipe frees at.
        let mut free = 0u32;
        while order.len() < n {
            while done[first] {
                first += 1;
            }
            let now = u64::from(cycle) << 32;
            let at = |k: u64| k.max(now | (k & 0xffff_ffff));
            let rank = |i: usize| if PIPE { self.piped(i, at(key[i]), free, cycle) } else { at(key[i]) };
            // The first unscheduled instruction's predecessors all precede
            // it, so it is always a candidate.
            let (mut pick, mut best) = (first, rank(first));
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
                    let k = rank(i);
                    if k < best {
                        (pick, best) = (i, k);
                    }
                }
            }
            let (i, start) = (pick, (best >> 32) as u32);
            done[i] = true;
            ready[i / 64] &= !(1 << (i % 64));
            order.push(i as u32);
            cycle = start + u32::from(self.issues[i]);
            if PIPE && self.pipe[i] != 0 {
                free = start + u32::from(self.pipe[i]);
            }
            for &(j, latency) in &self.succs[self.succ_start[i] as usize..self.succ_start[i + 1] as usize] {
                let j = j as usize;
                preds[j] -= 1;
                key[j] = key[j].max(u64::from(start + latency) << 32 | (key[j] & 0xffff_ffff));
                if preds[j] == 0 && !done[j] {
                    ready[j / 64] |= 1 << (j % 64);
                }
            }
        }
        order
    }
}

/// The dependence graphs of the blocks of one function, built once for all
/// the windows it is scheduled with.
pub(super) struct Dags(Vec<BlockDag>);

impl Dags {
    pub(super) fn new(f: &MFunc) -> Dags {
        let units = UnitIndex::new(f.blocks.iter().flat_map(|b| b.insts.iter()));
        let mut scratch = DagScratch::default();
        Dags(f.blocks.iter().map(|block| block_dag(&block.insts, &units, f.target, &mut scratch)).collect())
    }

    /// The order of each block's instructions in `schedule_window` with
    /// `window` of the function the graphs are of.
    pub(super) fn orders(&self, window: usize) -> Vec<Vec<u32>> {
        self.0.iter().map(|dag| dag.order(window)).collect()
    }

    /// `orders(window)` of block `block` alone.
    pub(super) fn block_order(&self, block: usize, window: usize) -> Vec<u32> {
        self.0[block].order(window)
    }

    /// The instructions of block `block` that depend on its instruction `i`.
    pub(super) fn successors(&self, block: usize, i: usize) -> impl Iterator<Item = usize> + '_ {
        let dag = &self.0[block];
        dag.succs[dag.succ_start[i] as usize..dag.succ_start[i + 1] as usize].iter().map(|&(j, _)| j as usize)
    }
}

/// `f` with the instructions of each block in `orders`.
pub(super) fn reordered(f: &MFunc, orders: &[Vec<u32>]) -> MFunc {
    let blocks = f
        .blocks
        .iter()
        .zip(orders)
        .map(|(block, order)| MBlock { insts: order.iter().map(|&i| block.insts[i as usize].clone()).collect(), loop_depth: block.loop_depth })
        .collect();
    MFunc { blocks, vregs: f.vregs.clone(), target: f.target, vgpr_limit: f.vgpr_limit, agpr_limit: f.agpr_limit }
}

/// How long a schedule keeps a SIMD busy, for choosing between schedules
/// that differ in occupancy: per block, weighted by 8 to the power of its
/// loop depth, the cycles one wave needs alone (latencies exposed) and the
/// cycles it spends issuing.
pub struct ScheduleCost {
    blocks: Vec<(u64, u64, u64)>,
}

impl ScheduleCost {
    /// Weighted cycles per wave when `waves` waves per SIMD hide each
    /// other's latency: a block takes the longer of its latency share and
    /// its issue time.
    pub fn cycles(&self, waves: u32) -> u64 {
        let w = u64::from(waves.max(1));
        self.blocks.iter().map(|&(weight, latency, issue)| weight * (latency / w).max(issue)).sum()
    }

    /// Weighted issue cycles: no schedule of these instructions takes less.
    pub fn issue(&self) -> u64 {
        self.blocks.iter().map(|&(weight, _, issue)| weight * issue).sum()
    }
}

/// The latency model of `schedule_cost`: memory results arrive in issue
/// order per counter, VMEM loads after 500 cycles, LDS after 64, SMEM after
/// 200; a WMMA occupies the VALU for 16 cycles and a transcendental for 4.
pub fn schedule_cost(f: &MFunc) -> ScheduleCost {
    let units = UnitIndex::new(f.blocks.iter().flat_map(|b| b.insts.iter()));
    // Physical registers first, then the virtual-register units.
    let keys = |o: &Opnd| reg_keys(o).chain(units.of(o).map(|u| NREGS + u));
    let mut ready = vec![0u64; NREGS + units.len()];
    let blocks = f
        .blocks
        .iter()
        .map(|block| {
            ready.fill(0);
            let (mut cycle, mut issue, mut vm_done, mut lgkm_done) = (0u64, 0u64, 0u64, 0u64);
            for inst in &block.insts {
                let flags = inst.op.flags();
                if matches!(inst.op, Opc::SBarrier | Opc::SBarrierLds) {
                    cycle = cycle.max(lgkm_done).max(if inst.op == Opc::SBarrier { vm_done } else { 0 });
                }
                let in_order = flags & F_VMEM_LOAD != 0 || lds_merge(inst);
                let operands_ready = inst
                    .uses
                    .iter()
                    .filter(|o| !(in_order && inst.defs.contains(o)))
                    .flat_map(keys)
                    .map(|k| ready[k])
                    .max()
                    .unwrap_or(0);
                let (cost, done) = if inst.op.fmt() == Fmt::Pseudo && flags & F_USE_EXEC == 0 {
                    (0, operands_ready)
                } else {
                    let start = cycle.max(operands_ready);
                    let cost = if flags & F_WMMA != 0 {
                        16
                    } else if flags & F_TRANS != 0 {
                        4
                    } else {
                        1
                    };
                    cycle = start + cost;
                    let done = if flags & F_VMEM_LOAD != 0 {
                        vm_done = vm_done.max(start + 500);
                        vm_done
                    } else if flags & (F_LDS | F_SMEM) != 0 && !inst.defs.is_empty() {
                        lgkm_done = lgkm_done.max(start + if flags & F_SMEM != 0 { 200 } else { 64 });
                        lgkm_done
                    } else {
                        cycle
                    };
                    (cost, done)
                };
                issue += cost;
                for k in inst.defs.iter().flat_map(keys) {
                    ready[k] = done;
                }
            }
            (8u64.pow(block.loop_depth.min(4)), cycle, issue)
        })
        .collect();
    ScheduleCost { blocks }
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
    insert_delay_alu_with(f, &mut Vec::new());
}

/// `insert_delay_alu`, building blocks in `spare` (see `reuse`).
pub(super) fn insert_delay_alu_with(f: &mut MFunc, spare: &mut Vec<MInst>) {
    let mut state = AluState::new();
    for block in f.blocks.iter_mut() {
        state.clear();
        let mut out = reuse(spare, block.insts.len() * 5 / 4);
        let mut open: Option<usize> = None;
        // Non-pseudo instructions emitted after the open `s_delay_alu`.
        let mut since_open = 0usize;
        let mut insts = std::mem::take(&mut block.insts);
        for inst in insts.drain(..) {
            let fmt = inst.op.fmt();
            if fmt == Fmt::Pseudo {
                out.push(inst);
                continue;
            }
            let (class, latency) = super::gen_isa::delay(inst.op);
            let kind = match class {
                super::gen_isa::Delay::Valu => AluKind::Valu,
                super::gen_isa::Delay::Trans => AluKind::Trans,
                super::gen_isa::Delay::Salu => AluKind::Salu,
                _ => AluKind::Other,
            };
            if super::gen_isa::delay_clears(inst.op) {
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
        *spare = insts;
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
    expand_pseudos_with(f, &mut Vec::new())
}

/// An empty vector with room for `len` instructions, made of `spare`: the
/// instructions of a block a pass rebuilt, which the passes after
/// allocation (`rir::finish`) hand from one to the next.
fn reuse(spare: &mut Vec<MInst>, len: usize) -> Vec<MInst> {
    let mut out = std::mem::take(spare);
    out.clear();
    out.reserve(len);
    out
}

/// `expand_pseudos`, building blocks in `spare` (see `reuse`).
pub(super) fn expand_pseudos_with(f: &mut MFunc, spare: &mut Vec<MInst>) -> Result<(), CompileError> {
    for block in f.blocks.iter_mut() {
        let mut out = reuse(spare, block.insts.len());
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
                Opc::ImplicitDef | Opc::SchedBarrier => {}
                _ => out.push(inst),
            }
        }
        *spare = std::mem::replace(&mut block.insts, out);
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

/// Whether `inst` is an LDS d16 load, which writes one half of its
/// destination and keeps the other.
fn lds_merge(inst: &MInst) -> bool {
    inst.op.flags() & (F_LDS | F_TIED) == F_LDS | F_TIED
}

/// The use of a d16 load that is its destination's old value: its last.
/// Any other use naming the destination, such as an address in the
/// register the load overwrites, is read when the load issues.
fn tied_source(inst: &MInst) -> Option<usize> {
    let last = inst.uses.len().checked_sub(1)?;
    (inst.op.flags() & F_TIED != 0 && inst.defs.contains(&inst.uses[last])).then_some(last)
}

/// One step of the wait model; returns the wait to insert before `inst`.
fn step(state: &mut Waits, inst: &MInst) -> Option<(u8, u8)> {
    let flags = inst.op.flags();
    let vmem_load = flags & F_VMEM_LOAD != 0;
    let lds_merge = lds_merge(inst);
    let mut need_vm = NONE;
    let mut need_lgkm = NONE;
    let tied_use = tied_source(inst);
    for (i, o) in inst.uses.iter().chain(inst.defs.iter()).enumerate() {
        // VMEM loads write VGPRs in issue order, and a d16 load merges into
        // the half it keeps when it writes, so neither a load's destination
        // nor that tied source waits for an earlier VMEM load. The same holds
        // for LDS: a wave's DS results are written in issue order, and an
        // LDS d16 load writes only its half when its data returns.
        let tied = i >= inst.uses.len() || tied_use == Some(i);
        for k in reg_keys(o) {
            if !(vmem_load && tied) {
                need_vm = need_vm.min(state.vm.get(k));
            }
            if !(lds_merge && tied) {
                need_lgkm = need_lgkm.min(state.lgkm.get(k));
            }
        }
    }
    if matches!(inst.op, Opc::SBarrier | Opc::SWaitcntVscnt) {
        if state.vm_any {
            need_vm = 0;
        }
        if state.lgkm_any {
            need_lgkm = 0;
        }
    }
    if inst.op == Opc::SWaitcntExplicit {
        let (vm, lgkm) = ((inst.imm >> 10) & 0x3f, (inst.imm >> 4) & 0x3f);
        if vm < 0x3f {
            state.vm.wait(vm as u8);
            state.vm_any &= vm != 0;
        }
        // Scalar loads return out of order: only lgkmcnt(0) settles them.
        if lgkm == 0 || (lgkm < 0x3f && !state.smem) {
            state.lgkm.wait(lgkm as u8);
            state.lgkm_any &= lgkm != 0;
            state.smem &= lgkm != 0;
        }
    }
    if inst.op == Opc::SBarrierLds && state.lgkm_any {
        need_lgkm = 0;
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
    let counters = super::gen_isa::counters(inst.op);
    if counters & super::gen_isa::CNT_VM != 0 {
        state.vm.issue(&inst.defs);
        state.vm_any = true;
    }
    if counters & super::gen_isa::CNT_LGKM != 0 {
        state.lgkm.issue(&inst.defs);
        state.lgkm_any = true;
        if counters & super::gen_isa::CNT_SMEM != 0 {
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
    insert_waitcnt_with(f, &mut Vec::new());
}

/// `insert_waitcnt`, building blocks in `spare` (see `reuse`).
pub(super) fn insert_waitcnt_with(f: &mut MFunc, spare: &mut Vec<MInst>) {
    let n = f.blocks.len();
    let succs: Vec<Vec<usize>> = (0..n).map(|b| f.successors(b)).collect();
    let mut entry: Vec<Option<WaitState>> = vec![None; n];
    entry[0] = Some(WaitState::new());
    // Blocks whose entry state changed since they were last stepped through:
    // stepping any other again would merge into its successors what they
    // already hold.
    let mut stale = vec![false; n];
    stale[0] = true;
    // The waits each block's last stepping found, by instruction: once the
    // entry states settle, every block was last stepped from its own.
    let mut found: Vec<Option<Vec<(u32, (u8, u8))>>> = vec![None; n];
    loop {
        let mut changed = false;
        for b in 0..n {
            if !stale[b] {
                continue;
            }
            let Some(start) = &entry[b] else {
                continue;
            };
            stale[b] = false;
            let mut waits = Waits::new(start);
            let at = found[b].get_or_insert_with(Vec::new);
            at.clear();
            for (k, inst) in f.blocks[b].insts.iter().enumerate() {
                if let Some(wait) = step(&mut waits, inst) {
                    at.push((k as u32, wait));
                }
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
                    stale[s] = true;
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
        // Blocks never stepped (unreachable ones) start from nothing pending.
        let mut known = found[b].as_deref().map(|at| at.iter().peekable());
        let mut waits = known.is_none().then(|| Waits::new(entry[b].as_ref().unwrap_or(&WaitState::new())));
        let mut insts = std::mem::take(&mut f.blocks[b].insts);
        let mut out = reuse(spare, insts.len() + insts.len() / 4 + 8);
        for (k, inst) in insts.drain(..).enumerate() {
            let wait = match (known.as_mut(), waits.as_mut()) {
                (Some(at), _) => at.next_if(|&&(i, _)| i as usize == k).map(|&(_, wait)| wait),
                (None, Some(waits)) => step(waits, &inst),
                (None, None) => unreachable!("a block is stepped or has its waits"),
            };
            if let Some((vm, lgkm)) = wait {
                out.push(MInst::new(Opc::SWaitcnt, [], []).with_imm(waitcnt_imm(vm, lgkm)));
            }
            let loads = inst.op.flags() & (F_VMEM_LOAD | F_LDS | F_SMEM) != 0;
            out.push(inst);
            if wait_all && loads {
                out.push(MInst::new(Opc::SWaitcnt, [], []).with_imm(waitcnt_imm(0, 0)));
            }
        }
        f.blocks[b].insts = out;
        *spare = insts;
    }
}

fn is_valu(op: Opc) -> bool {
    super::gen_isa::is_valu(op)
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
        if !needs_wmma_nop(&f.blocks[b].insts, &preds[b], &last_valu) {
            continue;
        }
        let insts = std::mem::take(&mut f.blocks[b].insts);
        let mut out: Vec<MInst> = Vec::with_capacity(insts.len());
        for inst in insts {
            if super::gen_isa::hazards(inst.op) & super::gen_isa::HZ_WMMA_AB != 0 {
                let prev = out.iter().rev().find(|i| is_valu(i.op));
                let hazard = |p: &MInst| {
                    super::gen_isa::hazards(p.op) & super::gen_isa::HZ_WMMA_D != 0
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

/// Whether `fix_wmma_hazards` puts a `v_nop` in the block `insts`, whose
/// predecessors are `preds`: up to its first, the instructions it has put
/// out are the block's own.
fn needs_wmma_nop(insts: &[MInst], preds: &[usize], last_valu: &[Option<MInst>]) -> bool {
    let mut prev: Option<&MInst> = None;
    for inst in insts {
        if super::gen_isa::hazards(inst.op) & super::gen_isa::HZ_WMMA_AB != 0 {
            let hazard = |p: &MInst| {
                super::gen_isa::hazards(p.op) & super::gen_isa::HZ_WMMA_D != 0
                    && (ranges_overlap(&p.defs[0], &inst.uses[0]) || ranges_overlap(&p.defs[0], &inst.uses[1]))
            };
            let needs = match prev {
                Some(p) => hazard(p),
                None => preds.iter().any(|&p| last_valu[p].as_ref().is_none_or(&hazard)),
            };
            if needs {
                return true;
            }
        }
        if is_valu(inst.op) {
            prev = Some(inst);
        }
    }
    false
}

/// What one `s_clause` may group: gfx11 hard clauses hold VMEM loads, stores
/// or atomics of one encoding only.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum ClauseKind {
    Load(Fmt),
    Store(Fmt),
    Atomic(Fmt),
}

fn clause_kind(op: Opc) -> Option<ClauseKind> {
    let (fmt, flags) = (op.fmt(), op.flags());
    if !matches!(fmt, Fmt::Global | Fmt::Mubuf) || flags & (F_VMEM_LOAD | F_VMEM_STORE) == 0 {
        return None;
    }
    let name = op.name();
    if name.starts_with("scratch_") {
        None
    } else if name.contains("atomic") {
        Some(ClauseKind::Atomic(fmt))
    } else if flags & F_VMEM_LOAD != 0 {
        Some(ClauseKind::Load(fmt))
    } else {
        Some(ClauseKind::Store(fmt))
    }
}

/// The most instructions one `s_clause` may cover on gfx11: longer clauses
/// can trigger a hardware bug (LLVM's `FeatureMaxHardClauseLength32`).
const MAX_CLAUSE: usize = 32;

/// Puts `s_clause` before each run of two or more adjacent VMEM stores of
/// one encoding, so that the run issues without other waves' memory
/// instructions in between. A wait, SALU or VALU instruction ends a run.
/// Loads and atomics stay unclaused: their clauses run no faster on gfx1151.
pub fn form_clauses(f: &mut MFunc) {
    let mut kinds = Vec::new();
    for block in f.blocks.iter_mut() {
        let kind_of: Vec<Option<ClauseKind>> = block.insts.iter().map(|i| store_kind(&mut kinds, i.op)).collect();
        if !forms_clause(&kind_of) {
            continue;
        }
        let mut out = Vec::with_capacity(block.insts.len() + 8);
        let mut run = 0;
        for (i, inst) in std::mem::take(&mut block.insts).into_iter().enumerate() {
            if run == 0 {
                if let Some(k) = kind_of[i] {
                    run = kind_of[i..].iter().take(MAX_CLAUSE).take_while(|&&x| x == Some(k)).count();
                    if run > 1 {
                        out.push(MInst::new(Opc::SClause, [], []).with_imm(run as i64 - 1));
                    }
                }
            }
            run = run.saturating_sub(1);
            out.push(inst);
        }
        block.insts = out;
    }
}

/// `clause_kind(op)` if it is a store's, as `form_clauses` reads it, through
/// `kinds`, those found so far (`clause_kind` spells the mnemonic out).
fn store_kind(kinds: &mut Vec<(Opc, Option<ClauseKind>)>, op: Opc) -> Option<ClauseKind> {
    if !matches!(op.fmt(), Fmt::Global | Fmt::Mubuf) || op.flags() & (F_VMEM_LOAD | F_VMEM_STORE) == 0 {
        return None;
    }
    if let Some(&(_, kind)) = kinds.iter().find(|(o, _)| *o == op) {
        return kind;
    }
    let kind = clause_kind(op).filter(|k| matches!(k, ClauseKind::Store(_)));
    kinds.push((op, kind));
    kind
}

/// Whether `form_clauses` puts an `s_clause` in a block whose store kinds
/// are `kind_of`: two adjacent stores of one kind.
fn forms_clause(kind_of: &[Option<ClauseKind>]) -> bool {
    kind_of.windows(2).any(|pair| pair[0].is_some() && pair[0] == pair[1])
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
        if !block.insts.iter().any(|i| i.op == Opc::SEndpgm) {
            continue;
        }
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
pub(super) mod tests {
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
        super::schedule_block(insts, window, &units, crate::target::Target::Gfx1151)
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

    fn v(reg: u16) -> Opnd {
        Opnd::P(Phys::V(reg), 1)
    }

    fn func(insts: Vec<MInst>) -> MFunc {
        MFunc { blocks: vec![MBlock { insts, loop_depth: 0 }], vregs: Vec::new(), ..Default::default() }
    }

    fn with_waits(f: &MFunc) -> Vec<(Opc, i64)> {
        f.blocks[0].insts.iter().map(|i| (i.op, if i.op == Opc::SWaitcnt { i.imm } else { 0 })).collect()
    }

    #[test]
    fn lds_d16_merge_does_not_wait_for_the_kept_half() {
        let mut f = func(vec![
            MInst::new(Opc::DsLoadU16, [v(1)], [v(10)]),
            MInst::new(Opc::DsLoadU16D16Hi, [v(1)], [v(11), v(1)]),
            MInst::new(Opc::DsBpermuteB32, [v(2)], [v(12), v(13)]),
            MInst::new(Opc::DsLoadU16D16, [v(2)], [v(14), v(2)]),
            MInst::new(Opc::VAddNcU32, [v(3)], [v(1), v(2)]),
        ]);
        insert_waitcnt(&mut f);
        assert_eq!(
            with_waits(&f),
            [
                (Opc::DsLoadU16, 0),
                (Opc::DsLoadU16D16Hi, 0),
                (Opc::DsBpermuteB32, 0),
                (Opc::DsLoadU16D16, 0),
                (Opc::SWaitcnt, waitcnt_imm(NONE, 0)),
                (Opc::VAddNcU32, 0),
            ]
        );
    }

    #[test]
    fn an_address_in_the_destination_waits_for_its_load() {
        // A reload into v1, then a load through v1 into v1 (forced spilling
        // made this common): the address is read at issue, unlike the
        // destination, so it waits; the d16 merge's kept half still does not.
        let saddr = Opnd::P(Phys::S(0), 2);
        let mut f = func(vec![
            MInst::new(Opc::ScratchLoadB32, [v(1)], []).with_imm(16),
            MInst::new(Opc::GLoadB32, [v(1)], [v(1), saddr]).with_mods(M_SADDR),
            MInst::new(Opc::DsLoadU16, [v(2)], [v(12)]),
            MInst::new(Opc::DsLoadU16D16Hi, [v(2)], [v(2), v(2)]),
            MInst::new(Opc::GLoadB32, [v(3)], [v(13), saddr]).with_mods(M_SADDR),
            MInst::new(Opc::GLoadD16HiB16, [v(3)], [v(14), saddr, v(3)]).with_mods(M_SADDR),
        ]);
        insert_waitcnt(&mut f);
        assert_eq!(
            with_waits(&f),
            [
                (Opc::ScratchLoadB32, 0),
                (Opc::SWaitcnt, waitcnt_imm(0, NONE)),
                (Opc::GLoadB32, 0),
                (Opc::DsLoadU16, 0),
                (Opc::SWaitcnt, waitcnt_imm(NONE, 0)),
                (Opc::DsLoadU16D16Hi, 0),
                (Opc::GLoadB32, 0),
                (Opc::GLoadD16HiB16, 0),
            ]
        );
    }

    #[test]
    fn d16_merges_still_wait_across_counters() {
        // A VMEM load and an LDS merge into one register complete in either order.
        let mut f = func(vec![
            MInst::new(Opc::GLoadB32, [v(1)], [Opnd::P(Phys::V(10), 2)]),
            MInst::new(Opc::DsLoadU16D16Hi, [v(1)], [v(12), v(1)]),
            MInst::new(Opc::DsLoadU16, [v(2)], [v(13)]),
            MInst::new(Opc::GLoadD16HiB16, [v(2)], [Opnd::P(Phys::V(14), 2), v(2)]),
        ]);
        insert_waitcnt(&mut f);
        assert_eq!(
            with_waits(&f),
            [
                (Opc::GLoadB32, 0),
                (Opc::SWaitcnt, waitcnt_imm(0, NONE)),
                (Opc::DsLoadU16D16Hi, 0),
                (Opc::DsLoadU16, 0),
                (Opc::SWaitcnt, waitcnt_imm(NONE, 0)),
                (Opc::GLoadD16HiB16, 0),
            ]
        );
    }

    #[test]
    fn schedule_cost_rewards_overlapped_loads_until_waves_hide_them() {
        let load = |d: u32, a: u32| MInst::new(Opc::GLoadB32, [r(d)], [r(a)]);
        let mut serial = func(vec![load(1, 0), add(2, 1), load(3, 0), add(4, 3)]);
        let mut overlapped = func(vec![load(1, 0), load(3, 0), add(2, 1), add(4, 3)]);
        serial.blocks[0].loop_depth = 1;
        overlapped.blocks[0].loop_depth = 1;
        let (s, o) = (schedule_cost(&serial), schedule_cost(&overlapped));
        // One wave alone exposes two load latencies, or one when they overlap.
        assert_eq!(s.cycles(1), 8 * 1002);
        assert_eq!(o.cycles(1), 8 * 502);
        // With enough waves both only issue: four instructions.
        assert_eq!(s.cycles(1024), 8 * 4);
        assert_eq!(o.cycles(1024), s.issue());
    }

    #[test]
    fn schedule_cost_follows_physical_registers_and_barriers() {
        let lds = MInst::new(Opc::DsLoadB32, [v(1)], [v(9)]);
        let use_it = MInst::new(Opc::VAddNcU32, [v(2)], [v(1), Opnd::Imm(1)]);
        assert_eq!(schedule_cost(&func(vec![lds.clone(), use_it.clone()])).cycles(1), 65);
        // A barrier waits for the load even when nothing reads it.
        let barrier = MInst::new(Opc::SBarrier, [], []);
        assert_eq!(schedule_cost(&func(vec![lds, barrier, add(3, 4)])).cycles(1), 66);
    }

    fn mem(op: Opc) -> MInst {
        if op.flags() & F_VMEM_LOAD != 0 {
            MInst::new(op, [v(1)], [Opnd::P(Phys::V(10), 2)])
        } else {
            MInst::new(op, [], [Opnd::P(Phys::V(10), 2), v(1)])
        }
    }

    fn clauses(insts: Vec<MInst>) -> Vec<(Opc, i64)> {
        let mut f = func(insts);
        form_clauses(&mut f);
        f.blocks[0].insts.iter().map(|i| (i.op, if i.op == Opc::SClause { i.imm } else { 0 })).collect()
    }

    #[test]
    fn clauses_cover_runs_of_adjacent_stores_only() {
        let wait = MInst::new(Opc::SWaitcnt, [], []).with_imm(waitcnt_imm(0, NONE));
        let got = clauses(vec![
            mem(Opc::GStoreB128),
            mem(Opc::GStoreB128),
            mem(Opc::GStoreD16HiB16),
            wait,
            mem(Opc::GStoreB32),
            mem(Opc::GLoadB32),
            mem(Opc::GLoadD16HiB16),
            mem(Opc::GAtomicAddF32),
            mem(Opc::GAtomicAddF32),
            mem(Opc::GStoreB32),
            mem(Opc::BufStoreB32),
            mem(Opc::ScratchStoreB32),
            mem(Opc::ScratchStoreB32),
            add(2, 1),
            mem(Opc::BufStoreB32),
            mem(Opc::BufStoreB16),
        ]);
        let ops: Vec<Opc> = got.iter().map(|&(op, _)| op).collect();
        assert_eq!(ops.iter().filter(|&&op| op == Opc::SClause).count(), 2);
        assert_eq!(got[0], (Opc::SClause, 2));
        assert_eq!(&ops[1..4], [Opc::GStoreB128, Opc::GStoreB128, Opc::GStoreD16HiB16]);
        let last = ops.len() - 3;
        assert_eq!(got[last], (Opc::SClause, 1));
        assert_eq!(&ops[last + 1..], [Opc::BufStoreB32, Opc::BufStoreB16]);
    }

    #[test]
    fn clauses_hold_at_most_32_instructions() {
        let got = clauses((0..40).map(|_| mem(Opc::GStoreB32)).collect());
        assert_eq!(got[0], (Opc::SClause, 31));
        assert_eq!(got[33], (Opc::SClause, 7));
        assert_eq!(got.len(), 42);
    }

    #[test]
    fn every_clause_is_followed_by_its_count_of_one_store_kind() {
        let pool = [
            Opc::GStoreB32,
            Opc::GStoreB128,
            Opc::BufStoreB32,
            Opc::GLoadB32,
            Opc::GAtomicAddF32,
            Opc::ScratchStoreB32,
            Opc::SWaitcnt,
            Opc::VAddNcU32,
            Opc::SNop,
        ];
        let mut seed = 0x2545_f491_4f6c_dd1du64;
        let mut next = |n: usize| {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (seed >> 33) as usize % n
        };
        for _ in 0..300 {
            let insts: Vec<MInst> = (0..next(120))
                .map(|_| match pool[next(pool.len())] {
                    Opc::VAddNcU32 => add(2, 1),
                    op @ (Opc::SWaitcnt | Opc::SNop) => MInst::new(op, [], []),
                    op => mem(op),
                })
                .collect();
            let mut f = func(insts.clone());
            form_clauses(&mut f);
            let out = &f.blocks[0].insts;
            let kept: Vec<Opc> = out.iter().map(|i| i.op).filter(|&op| op != Opc::SClause).collect();
            assert_eq!(kept, insts.iter().map(|i| i.op).collect::<Vec<_>>());
            let mut i = 0;
            while i < out.len() {
                let kind = clause_kind(out[i].op).filter(|k| matches!(k, ClauseKind::Store(_)));
                if out[i].op == Opc::SClause {
                    let n = out[i].imm as usize + 1;
                    assert!((2..=MAX_CLAUSE).contains(&n));
                    let body = &out[i + 1..i + 1 + n];
                    let first = clause_kind(body[0].op);
                    assert!(matches!(first, Some(ClauseKind::Store(_))));
                    assert!(body.iter().all(|b| clause_kind(b.op) == first));
                    i += 1 + n;
                } else {
                    // Outside clauses no two adjacent stores share a kind.
                    let next_kind = out.get(i + 1).and_then(|n| clause_kind(n.op));
                    assert!(kind.is_none() || next_kind != kind);
                    i += 1;
                }
            }
        }
    }

    /// `schedule_block` before its graph was shared between windows: the
    /// reference the shared graphs are checked against.
    fn schedule_block_reference(insts: Vec<MInst>, window: usize, units: &UnitIndex, target: crate::target::Target) -> Vec<MInst> {
        let result_latency = |inst: &MInst| match target {
            crate::target::Target::Gfx1151 => result_latency(inst),
            crate::target::Target::Gfx950 => crate::target::gfx9::model::result_latency(inst),
        };
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
        // Last write and the reads since, of LDS and of global memory (by
        // `alias` tag). SMEM reads only kernel arguments and memory the kernel
        // does not write.
        let mut lds = (None, Vec::new());
        let mut vmem = super::super::alias::GlobalOrder::default();
        for (i, inst) in insts.iter().enumerate() {
            let flags = inst.op.flags();
            if flags & (F_BRANCH | F_END | F_DEF_EXEC | F_USE_VCC | F_SMEM | F_FENCE) != 0
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
            // VMEM loads, and separately DS operations, write their results in
            // issue order, so a d16 load's tied half does not wait on one of its kind.
            let in_order = if flags & F_VMEM_LOAD != 0 {
                F_VMEM_LOAD
            } else if lds_merge(inst) {
                F_LDS
            } else {
                0
            };
            for o in &inst.uses {
                let tied = inst.defs.contains(o);
                for u in units.of(o) {
                    if last_def[u] != NO_INST {
                        let d = last_def[u] as usize;
                        let ordered = tied && insts[d].op.flags() & in_order != 0;
                        edge(d, i, if ordered { 0 } else { result_latency(&insts[d]) });
                    }
                    readers.push((i as u32, reader_head[u]));
                    reader_head[u] = readers.len() as u32 - 1;
                }
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
                vmem.access(i as u32, inst.mem, flags & F_VMEM_STORE == 0, |from| edge(from as usize, i, 0));
                None
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

    /// A random function of `blocks` blocks of `len` instructions each:
    /// VALU and SALU arithmetic, VMEM, LDS and scalar loads and stores (tied
    /// d16 halves among them) with `noalias` tags, SCC and EXEC writers and
    /// readers, barriers, waits, WMMAs, transcendentals, physical registers
    /// and pseudo instructions, on registers of one to four dwords, all
    /// defined in a first block of their own.
    pub(in super::super) fn random_function(seed: u64, blocks: usize, len: usize) -> MFunc {
        let mut seed = seed ^ 0x5bd1_e995_9e37_79b9;
        let mut next = |n: usize| {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (seed >> 33) as usize % n
        };
        let mut f = MFunc::default();
        let mut pool = |class: Class, size: u8, count: usize| -> Vec<u32> {
            (0..count)
                .map(|_| {
                    f.vregs.push(VRegInfo { class, size, align: if size > 1 && class == Class::S { 2 } else { 1 }, fixed: None });
                    f.vregs.len() as u32 - 1
                })
                .collect()
        };
        let (v1, v4, s1, s2) = (pool(Class::V, 1, 24), pool(Class::V, 4, 6), pool(Class::S, 1, 10), pool(Class::S, 2, 4));
        let whole = |reg: u32, n: u8| Opnd::R(VRef { reg, off: 0, n });
        let mut entry = Vec::new();
        for &v in v1.iter().chain(&s1) {
            entry.push(MInst::new(Opc::Copy, [whole(v, 1)], [Opnd::Imm(v)]));
        }
        for &v in &v4 {
            entry.push(MInst::new(Opc::Copy, [whole(v, 4)], [Opnd::Imm(0)]));
        }
        for &v in &s2 {
            entry.push(MInst::new(Opc::Copy, [whole(v, 2)], [Opnd::Imm(0)]));
        }
        f.blocks.push(MBlock { insts: entry, loop_depth: 0 });
        for b in 0..blocks {
            let mut insts = Vec::with_capacity(len);
            while insts.len() < len {
                let a = whole(v1[next(v1.len())], 1);
                let c = whole(v1[next(v1.len())], 1);
                let d = whole(v1[next(v1.len())], 1);
                let t = v4[next(v4.len())];
                let part = Opnd::R(VRef { reg: t, off: next(4) as u8, n: 1 });
                let (x, y) = (whole(s1[next(s1.len())], 1), whole(s1[next(s1.len())], 1));
                let pair = whole(s2[next(s2.len())], 2);
                let mem = [super::super::alias::ANY, 1, 2, 3][next(4)];
                let inst = match next(26) {
                    0 => MInst::new(Opc::VAddNcU32, [d], [a, Opnd::Imm(next(100) as u32)]),
                    1 => MInst::new(Opc::VMulF32, [d], [a, c]),
                    2 => MInst::new(Opc::VFmaF32, [d], [a, c, d]),
                    3 => MInst::new(Opc::GLoadB32, [d], [a, pair]).with_mods(M_SADDR).with_mem(mem),
                    4 => MInst::new(Opc::GLoadB128, [whole(t, 4)], [a, pair]).with_mods(M_SADDR).with_mem(mem),
                    5 => MInst::new(Opc::GStoreB32, [], [a, c, pair]).with_mods(M_SADDR).with_mem(mem),
                    6 => MInst::new(Opc::DsLoadB32, [d], [a]).with_imm(4 * next(8) as i64),
                    7 => MInst::new(Opc::DsStoreB32, [], [a, c]),
                    8 => MInst::new(Opc::DsLoadU16D16, [d], [a, d]),
                    9 => MInst::new(Opc::DsLoadU16D16Hi, [part], [a, part]),
                    10 => MInst::new(Opc::GLoadD16HiB16, [d], [a, pair, d]).with_mods(M_SADDR).with_mem(mem),
                    11 => MInst::new(Opc::SCmpLtI32, [], [x, y]),
                    12 => MInst::new(Opc::SCselectB32, [x], [y, Opnd::Imm(3)]),
                    13 => MInst::new(Opc::SAddI32, [x], [y, Opnd::Imm(1)]),
                    14 => MInst::new(Opc::SLoadB32, [x], [pair]).with_imm(4 * next(16) as i64),
                    15 => MInst::new([Opc::SBarrier, Opc::SBarrierLds][next(2)], [], []),
                    16 => MInst::new(Opc::Copy, [d], [a]),
                    17 => MInst::new(Opc::VExpF32, [d], [a]),
                    18 => MInst::new(Opc::VWmmaF32F16, [whole(t, 4)], [whole(v4[next(v4.len())], 4), whole(v4[next(v4.len())], 4), whole(t, 4)]),
                    19 => MInst::new(Opc::VMovB32, [Opnd::P(Phys::V(next(8) as u16), 1)], [a]),
                    20 => MInst::new(Opc::DsBpermuteB32, [d], [a, c]),
                    21 => MInst::new(Opc::SchedBarrier, [], []),
                    22 => MInst::new(Opc::SWaitcntExplicit, [], []).with_imm(0x3f7),
                    23 => MInst::new(Opc::VAddF32, [part], [part, a]),
                    24 => MInst::new(Opc::GAtomicAddF32, [], [a, c, pair]).with_mods(M_SADDR).with_mem(mem),
                    _ => MInst::new(Opc::VCvtF32F16, [d], [part]),
                };
                insts.push(inst);
            }
            let (x, y) = (whole(s1[next(s1.len())], 1), whole(s1[next(s1.len())], 1));
            match next(4) {
                0 => {
                    let exec = Opnd::P(Phys::ExecLo, 1);
                    insts.push(MInst::new(Opc::SAndSaveexecB32, [x, exec], [y, exec]));
                    insts.push(MInst::new(Opc::SCbranchExecz, [], []).with_imm((b as i64 + 2).min(blocks as i64)));
                }
                1 => {
                    insts.push(MInst::new(Opc::SCmpLtI32, [], [x, y]));
                    insts.push(MInst::new(Opc::SCbranchScc1, [], []).with_imm(1 + next(b + 1) as i64));
                }
                _ => {}
            }
            f.blocks.push(MBlock { insts, loop_depth: next(3) as u32 });
        }
        f.blocks.last_mut().expect("a block").insts.push(MInst::new(Opc::SEndpgm, [], []));
        f
    }

    #[test]
    fn shared_graphs_order_blocks_as_schedule_block_did() {
        for seed in 0..80u64 {
            let mut f = random_function(seed, 1 + seed as usize % 4, 10 + 9 * seed as usize % 300);
            if seed % 3 == 2 {
                f.target = crate::target::Target::Gfx950;
            }
            let dags = Dags::new(&f);
            let units = UnitIndex::new(f.blocks.iter().flat_map(|b| b.insts.iter()));
            for window in [1, 2, 3, 7, 16, 32, 63, 64, 65, 512] {
                let shared = reordered(&f, &dags.orders(window));
                let mut scheduled = f.clone();
                schedule_window(&mut scheduled, window);
                for (b, block) in f.blocks.iter().enumerate() {
                    let want = format!("{:?}", schedule_block_reference(block.insts.clone(), window, &units, f.target));
                    assert_eq!(format!("{:?}", shared.blocks[b].insts), want, "seed {seed}, window {window}, block {b}");
                    assert_eq!(format!("{:?}", scheduled.blocks[b].insts), want, "seed {seed}, window {window}, block {b}");
                    assert_eq!(shared.blocks[b].loop_depth, block.loop_depth);
                }
                assert!(super::super::same_function(&shared, &scheduled));
            }
        }
    }

    /// `insert_waitcnt` before blocks it leaves alone stayed as they were: the reference.
    fn insert_waitcnt_reference(f: &mut MFunc) {
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
        let wait_all = super::super::debug_flag("waitall");
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

    /// `fix_wmma_hazards` before blocks it leaves alone stayed as they were: the reference.
    fn fix_wmma_hazards_reference(f: &mut MFunc) {
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
                if super::super::gen_isa::hazards(inst.op) & super::super::gen_isa::HZ_WMMA_AB != 0 {
                    let prev = out.iter().rev().find(|i| is_valu(i.op));
                    let hazard = |p: &MInst| {
                        super::super::gen_isa::hazards(p.op) & super::super::gen_isa::HZ_WMMA_D != 0
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

    /// `form_clauses` before blocks it leaves alone stayed as they were: the reference.
    fn form_clauses_reference(f: &mut MFunc) {
        for block in f.blocks.iter_mut() {
            let kind_of: Vec<Option<ClauseKind>> =
                block.insts.iter().map(|i| clause_kind(i.op).filter(|k| matches!(k, ClauseKind::Store(_)))).collect();
            let mut out = Vec::with_capacity(block.insts.len() + 8);
            let mut run = 0;
            for (i, inst) in std::mem::take(&mut block.insts).into_iter().enumerate() {
                if run == 0 {
                    if let Some(k) = kind_of[i] {
                        run = kind_of[i..].iter().take(MAX_CLAUSE).take_while(|&&x| x == Some(k)).count();
                        if run > 1 {
                            out.push(MInst::new(Opc::SClause, [], []).with_imm(run as i64 - 1));
                        }
                    }
                }
                run = run.saturating_sub(1);
                out.push(inst);
            }
            block.insts = out;
        }
    }

    /// `add_vgpr_release` before blocks it leaves alone stayed as they were: the reference.
    fn add_vgpr_release_reference(f: &mut MFunc, num_vgprs: u32) {
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

    #[test]
    fn finish_passes_put_out_what_they_did() {
        let (mut checked, mut nops, mut clauses) = (0, 0, 0);
        for seed in 0..120u64 {
            let mut f = random_function(seed, 1 + seed as usize % 6, 10 + 13 * seed as usize % 250);
            if super::super::regalloc::allocate(&mut f).is_err() {
                continue;
            }
            // In `rir::finish`'s order, the rebuilding passes sharing one spare.
            let (mut spare, mut want) = (Vec::new(), f.clone());
            expand_pseudos_with(&mut f, &mut spare).expect("expands");
            expand_pseudos(&mut want).expect("expands");
            let same = |f: &MFunc, want: &MFunc, name: &str| {
                assert_eq!(format!("{:?}", f.blocks), format!("{:?}", want.blocks), "seed {seed}, {name}");
            };
            same(&f, &want, "expand");
            insert_waitcnt_with(&mut f, &mut spare);
            insert_waitcnt_reference(&mut want);
            same(&f, &want, "waits");
            super::super::vopd::form_pairs_with(&mut f, &mut spare);
            super::super::vopd::form_pairs(&mut want);
            same(&f, &want, "pairs");
            fix_wmma_hazards(&mut f);
            fix_wmma_hazards_reference(&mut want);
            same(&f, &want, "wmma");
            form_clauses(&mut f);
            form_clauses_reference(&mut want);
            same(&f, &want, "clauses");
            insert_delay_alu_with(&mut f, &mut spare);
            insert_delay_alu(&mut want);
            same(&f, &want, "delay");
            let has = |op: Opc| f.blocks.iter().any(|b| b.insts.iter().any(|i| i.op == op));
            nops += has(Opc::VNop) as u32;
            clauses += has(Opc::SClause) as u32;
            for vgprs in [64, 128] {
                let (mut got, mut want) = (f.clone(), f.clone());
                add_vgpr_release(&mut got, vgprs);
                add_vgpr_release_reference(&mut want, vgprs);
                assert_eq!(format!("{:?}", got.blocks), format!("{:?}", want.blocks), "seed {seed}, release");
            }
            checked += 1;
        }
        assert!(checked > 100 && nops > 10 && clauses > 10, "{checked} functions allocated, {nops} with v_nop, {clauses} with clauses");
    }

    /// `sink_load_uses` before its facts held the loads: the reference.
    fn sink_load_uses_reference(f: &mut MFunc, budget: u32) {
        let fixed = F_DEF_SCC | F_USE_SCC | F_USE_VCC | F_DEF_EXEC | F_BRANCH | F_END | F_SMEM | F_FENCE;
        let units = UnitIndex::new(f.blocks.iter().flat_map(|b| b.insts.iter()));
        // Block number + 1 of the units loaded so far in that block.
        let mut loaded = vec![0u32; units.len()];
        for (b, block) in f.blocks.iter_mut().enumerate() {
            let stamp = b as u32 + 1;
            let insts = &mut block.insts;
            let mut candidate = vec![false; insts.len()];
            for (i, inst) in insts.iter().enumerate() {
                // Loads write in issue order, so a d16 load's tied destination
                // does not stall it.
                let load = inst.op.flags() & (F_VMEM_LOAD | F_LDS) != 0;
                let in_order = inst.op.flags() & F_VMEM_LOAD != 0 || lds_merge(inst);
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

    #[test]
    fn load_sinking_moves_what_it_moved() {
        let mut moved = 0;
        for seed in 0..120u64 {
            let mut f = random_function(seed, 1 + seed as usize % 6, 10 + 11 * seed as usize % 400);
            if seed % 3 == 1 {
                f.target = crate::target::Target::Gfx950;
            }
            for budget in [0, 4, 32] {
                let (mut got, mut want) = (f.clone(), f.clone());
                sink_load_uses(&mut got, budget);
                sink_load_uses_reference(&mut want, budget);
                assert_eq!(format!("{:?}", got.blocks), format!("{:?}", want.blocks), "seed {seed}, budget {budget}");
                moved += (format!("{:?}", got.blocks) != format!("{:?}", f.blocks)) as u32;
            }
        }
        assert!(moved > 100, "{moved} sinkings moved something");
    }
}
