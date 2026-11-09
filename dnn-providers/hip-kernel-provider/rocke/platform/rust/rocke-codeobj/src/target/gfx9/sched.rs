// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

//! The MFMA pipe in gfx950's scheduling model (G2 design section 8), for the
//! list scheduler of `rir::post`, in its unit: one issue slot, as long as a
//! wave64 VALU instruction holds its SIMD (four cycles).
//!
//! An MFMA holds its SIMD's matrix pipe for its passes, one slot each; the
//! wave's next MFMA starts when the pipe frees, and one accumulating into the
//! result (C the same registers) needs nothing more. Other instructions issue
//! while an MFMA runs. A VALU, LDS or VMEM reader of an MFMA result waits M5's
//! states from the MFMA's start, which cover the pipe's drain; an MFMA reading
//! it as A or B waits M4's, and an MFMA reading a VALU result M2's (the hazard
//! rules of `hazards`, one slot past their wait states, as `model` counts H1).
//!
//! Selection writes an MFMA's C into a fresh D through a copy that allocation
//! coalesces (`mfma::select`): a copy of an MFMA result costs nothing here, and
//! its readers wait as for the MFMA itself. Blocks without an MFMA keep the
//! shared latencies (`MfmaEdges::new` is None for them).

use super::agpr::is_mfma;
use super::gen_isa as t;
use super::hazards;
use crate::rir::mir::*;

const NONE: u32 = u32::MAX;
/// M2: wait states between a VALU write and an MFMA reading it.
const M2: u32 = 2;

/// Slots an MFMA holds the matrix pipe for; 0 for any other instruction.
pub fn pipe_slots(op: Opc) -> u8 {
    hazards::mfma_kind(op).map_or(0, |(_, passes)| passes as u8)
}

/// A VALU instruction other than an MFMA or a DOT: M2's producers.
fn plain_valu(op: Opc) -> bool {
    matches!(t::info(op).map(|i| i.fmt), Some(Fmt::Vop1 | Fmt::Vop2 | Fmt::Vop3 | Fmt::Vop3b | Fmt::Vopc | Fmt::Vop3p))
        && !is_mfma(op)
        && !matches!(op, Opc::VDot2F32F16 | Opc::VDot2F32Bf16)
}

fn width(o: &Opnd) -> u8 {
    match *o {
        Opnd::R(v) => v.n,
        Opnd::P(_, n) => n,
        Opnd::Imm(_) => 1,
    }
}

/// How long a schedule keeps a SIMD busy, for choosing between schedules of
/// different occupancy: per block, weighted by 8 to the power of its loop
/// depth, the slots one wave needs alone (latencies exposed), the slots it
/// issues in, and the slots its MFMAs hold the matrix pipe.
pub struct Cost {
    blocks: Vec<(u64, u64, u64, u64)>,
}

impl Cost {
    /// Weighted slots per wave with `waves` waves per SIMD: a block takes the
    /// longest of its latency share, its issue and its pipe time, as the
    /// waves of a SIMD share its issue and its matrix pipe.
    pub fn cycles(&self, waves: u32) -> u64 {
        let w = u64::from(waves.max(1));
        self.blocks.iter().map(|&(weight, latency, issue, pipe)| weight * (latency / w).max(issue).max(pipe)).sum()
    }

    /// The weighted slots no number of waves gets below: each block's issue
    /// and pipe time. A reordering of the blocks issues the same instructions,
    /// so this bounds `cycles` of every schedule of them from below.
    pub fn floor(&self) -> u64 {
        self.blocks.iter().map(|&(weight, _, issue, pipe)| weight * issue.max(pipe)).sum()
    }
}

/// Keys of physical registers: VGPRs and AGPRs from 0, SGPRs after them.
const PHYS: usize = 512 + 128;

fn phys_keys(o: &Opnd) -> std::ops::Range<usize> {
    match *o {
        Opnd::P(Phys::V(r), n) => r as usize..r as usize + n as usize,
        Opnd::P(Phys::S(r), n) => 512 + r as usize..512 + r as usize + n as usize,
        _ => 0..0,
    }
}

/// The cost model of `Cost`, in slots: one wave issues each block in order;
/// an instruction starts when its operands are ready, an MFMA also when the
/// pipe frees; VMEM loads and LDS and SMEM results arrive in issue order per
/// counter after `model`'s latencies; an MFMA result reaches an MFMA reading
/// it as C when the pipe frees, any other reader M5's states later.
pub fn cost(f: &MFunc) -> Cost {
    let operands = || f.blocks.iter().flat_map(|b| &b.insts).flat_map(|i| i.defs.iter().chain(i.uses.iter())).filter_map(Opnd::vref);
    let regs = operands().map(|v| v.reg as usize + 1).max().unwrap_or(0);
    let mut base = vec![0usize; regs + 1];
    for v in operands() {
        base[v.reg as usize + 1] = base[v.reg as usize + 1].max(v.off as usize + v.n as usize);
    }
    for r in 0..regs {
        base[r + 1] += base[r];
    }
    let keys = |o: &Opnd| -> std::ops::Range<usize> {
        match *o {
            Opnd::R(v) => {
                let k = PHYS + base[v.reg as usize] + v.off as usize;
                k..k + v.n as usize
            }
            _ => phys_keys(o),
        }
    };
    let n = PHYS + base[regs];
    // When a register's value may be read by any instruction, and by an MFMA as C.
    let (mut ready, mut ready_c) = (vec![0u64; n], vec![0u64; n]);
    let blocks = f
        .blocks
        .iter()
        .map(|block| {
            ready.fill(0);
            ready_c.fill(0);
            let (mut cycle, mut issue, mut pipe, mut free) = (0u64, 0u64, 0u64, 0u64);
            let (mut vm_done, mut lgkm_done) = (0u64, 0u64);
            for inst in &block.insts {
                let flags = inst.op.flags();
                if inst.op == Opc::SBarrier {
                    cycle = cycle.max(lgkm_done).max(vm_done);
                }
                let mfma = hazards::mfma_kind(inst.op);
                let operands = inst.uses.iter().enumerate().flat_map(|(k, o)| keys(o).map(move |key| (k, key)));
                let operands_ready = operands.map(|(k, key)| if mfma.is_some() && k == 2 { ready_c[key] } else { ready[key] }).max().unwrap_or(0);
                let (done, done_c) = if inst.op.fmt() == Fmt::Pseudo && flags & F_USE_EXEC == 0 {
                    let copied = inst.uses.iter().flat_map(keys).map(|key| ready_c[key]).max().unwrap_or(0);
                    (operands_ready, copied)
                } else {
                    let mut start = cycle.max(operands_ready);
                    if let Some((_, passes)) = mfma {
                        start = start.max(free);
                        free = start + u64::from(passes);
                        pipe += u64::from(passes);
                    }
                    cycle = start + 1;
                    issue += 1;
                    let done = if let Some(kind) = mfma {
                        start + u64::from(hazards::mfma_result_waits(kind, false, false) + 1)
                    } else if flags & F_VMEM_LOAD != 0 {
                        vm_done = vm_done.max(start + u64::from(super::model::result_latency(inst)));
                        vm_done
                    } else if flags & (F_LDS | F_SMEM) != 0 && !inst.defs.is_empty() {
                        lgkm_done = lgkm_done.max(start + u64::from(super::model::result_latency(inst)));
                        lgkm_done
                    } else {
                        start + u64::from(super::model::result_latency(inst).max(1))
                    };
                    (done, if mfma.is_some() { free } else { done })
                };
                for key in inst.defs.iter().flat_map(keys) {
                    ready[key] = done;
                    ready_c[key] = done_c;
                }
            }
            (8u64.pow(block.loop_depth.min(4)), cycle, issue, pipe)
        })
        .collect();
    Cost { blocks }
}

/// The register dependence latencies of one block that has an MFMA.
pub struct MfmaEdges {
    /// Per instruction, the MFMA whose result it copies (through copies).
    origin: Vec<u32>,
}

impl MfmaEdges {
    /// None for a block without an MFMA.
    pub fn new(insts: &[MInst]) -> Option<MfmaEdges> {
        insts.iter().any(|i| is_mfma(i.op)).then(|| MfmaEdges { origin: vec![NONE; insts.len()] })
    }

    /// Slots from the start of instruction `d` until instruction `i`, which
    /// reads a register `d` defines as its operand `k`, may start; None where
    /// the shared model's result latency of `d` applies. The instructions are
    /// seen in block order, `d` before `i`.
    pub fn raw(&mut self, insts: &[MInst], d: usize, i: usize, k: usize) -> Option<u32> {
        let consumer = &insts[i];
        let m = if is_mfma(insts[d].op) {
            d
        } else if self.origin[d] != NONE {
            self.origin[d] as usize
        } else {
            return (is_mfma(consumer.op) && plain_valu(insts[d].op)).then(|| super::model::result_latency(&insts[d]).max(M2 + 1));
        };
        if consumer.op == Opc::Copy {
            self.origin[i] = m as u32;
            return Some(0);
        }
        let producer = &insts[m];
        let kind = hazards::mfma_kind(producer.op)?;
        Some(match hazards::mfma_kind(consumer.op) {
            Some((class, _)) if k == 2 => {
                let full = width(&consumer.uses[2]) == width(&producer.defs[0]);
                (hazards::mfma_srcc_waits(kind, class, full) + 1).max(kind.1)
            }
            Some(_) => hazards::mfma_srcab_waits(kind) + 1,
            None => {
                let memory = consumer.op.flags() & (F_LDS | F_VMEM_LOAD | F_VMEM_STORE) != 0;
                hazards::mfma_result_waits(kind, memory, false) + 1
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::target::gfx9::gen_isa::Gfx9Op;
    use crate::target::Target;

    fn r(reg: u32, n: u8) -> Opnd {
        Opnd::R(VRef { reg, off: 0, n })
    }

    /// D (16 dwords at `d`) = A x B + C with C copied from `c` into D first, as selection writes it.
    fn mfma(d: u32, c: u32, a: u32, b: u32) -> [MInst; 2] {
        [
            MInst::new(Opc::Copy, [r(d, 16)], [r(c, 16)]),
            MInst::new(Opc::Gfx9(Gfx9Op::VMfma32x32x16F16), [r(d, 16)], [r(a, 4), r(b, 4), r(d, 16)]),
        ]
    }

    fn add(d: u32, s: u32) -> MInst {
        MInst::new(Opc::VAddNcU32, [r(d, 1)], [r(s, 1), Opnd::Imm(1)])
    }

    #[test]
    fn mfma_results_reach_their_readers_through_copies() {
        let [c1, m1] = mfma(10, 1, 2, 3);
        let [c2, m2] = mfma(11, 10, 2, 3);
        let store = MInst::new(Opc::VAddF32, [r(20, 1)], [Opnd::R(VRef { reg: 11, off: 3, n: 1 }), Opnd::Imm(0)]);
        let valu = add(2, 4);
        let insts = vec![valu, c1, m1, c2, m2, store];
        let mut edges = MfmaEdges::new(&insts).expect("the block has MFMAs");
        let passes = u32::from(pipe_slots(insts[2].op));
        assert_eq!(passes, 8);
        // A VALU result read by an MFMA: M2's two wait states.
        assert_eq!(edges.raw(&insts, 0, 2, 0), Some(3));
        // The copy of the first result costs nothing; the MFMA reading it as C waits for the pipe only.
        assert_eq!(edges.raw(&insts, 2, 3, 0), Some(0));
        assert_eq!(edges.raw(&insts, 3, 4, 2), Some(passes));
        // A VALU reader of the second result waits M5 (8-pass XDL: 12) past the MFMA's start.
        assert_eq!(edges.raw(&insts, 4, 5, 0), Some(13));
        // Edges between instructions without an MFMA keep the shared latency.
        assert_eq!(edges.raw(&[add(1, 0), add(2, 1)], 0, 1, 0), None);
        assert!(MfmaEdges::new(&[add(1, 0), add(2, 1)]).is_none());
    }

    #[test]
    fn independent_work_issues_while_the_pipe_is_busy() {
        let mut f = MFunc { target: Target::Gfx950, ..MFunc::default() };
        let mut insts = Vec::new();
        // Four MFMAs on their own accumulators, then four independent adds.
        for k in 0..4 {
            insts.extend(mfma(10 + k, 1, 2, 3));
        }
        insts.extend((0..4).map(|k| add(30 + k, 20 + k)));
        f.blocks.push(MBlock { insts, loop_depth: 0 });
        crate::rir::post::schedule_window(&mut f, 64);
        let order: Vec<bool> = f.blocks[0].insts.iter().filter(|i| i.op != Opc::Copy).map(|i| super::super::agpr::is_mfma(i.op)).collect();
        // Each MFMA holds the pipe for eight slots: the adds go between the first two.
        assert_eq!(order, [true, false, false, false, false, true, true, true]);
    }

    #[test]
    fn the_cost_counts_the_pipe_once_per_simd_and_the_latency_per_wave() {
        // A loop of a chain of 16 MFMAs: the pipe bounds it at any occupancy.
        let mut f = MFunc { target: Target::Gfx950, ..MFunc::default() };
        let mut chain = Vec::new();
        for k in 0..16 {
            chain.extend(mfma(10 + k + 1, 10 + k, 2, 3));
        }
        f.blocks.push(MBlock { insts: chain, loop_depth: 1 });
        let c = cost(&f);
        assert_eq!(c.cycles(1), 8 * 16 * 8);
        assert_eq!(c.cycles(4), 8 * 16 * 8);
        // Twelve loads each waited for before the next: latency that more waves hide.
        let mut g = MFunc { target: Target::Gfx950, ..MFunc::default() };
        let mut serial = Vec::new();
        for k in 0..12 {
            serial.push(MInst::new(Opc::GLoadB128, [r(40 + k, 4)], [r(39 + k, 2), r(60, 2)]).with_mods(M_SADDR));
        }
        g.blocks.push(MBlock { insts: serial, loop_depth: 1 });
        let l = cost(&g);
        assert_eq!(l.cycles(1), 8 * (11 * 180 + 1));
        assert_eq!(l.cycles(2), 8 * ((11 * 180 + 1) / 2));
    }
}
