// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

//! The order of M0 and of LDS DMA loads in gfx950's dependence graphs
//! (`post::block_dag`), in place of the fences they were.
//!
//! An LDS DMA load reads M0 implicitly (its LDS address) and is a fence for
//! that (`F_FENCE`), and so is the move into M0 before it, for naming a
//! physical register. Then nothing moves past either: in paged attention,
//! each page-table load sits between two DMA loads and waits for every
//! VMEM load before it (`vmcnt(0)`), one at a time, where COMGR's code has
//! sixty in flight. Here they keep only the orders they need:
//!
//! - M0: a read after the write before it, with that write's latency (H12's
//!   wait state, filled when something else can issue), a write after the
//!   reads and the write before it;
//! - LDS: a DMA load writes it, ordered with every LDS access as any LDS
//!   write is (its `F_LDS`);
//! - global memory: a DMA load reads it at an address not traced to an
//!   argument (`ANY`), after every global write before it and before every
//!   one after it.
//!
//! Barriers, waits, EXEC writes, branches, SMEM and scheduling directives
//! stay fences, so no DMA load or M0 write moves past one.
//!
//! Only a block with another VMEM load besides its DMA loads gets these
//! orders: one of DMA loads, LDS reads and MFMAs alone keeps its fences.
//! Two implicit-GEMM convolutions (`fdb436e1446d4d9c`, `5ca126ed1e9d34fd`)
//! measured half again slower with their address arithmetic interleaved
//! across the DMA loads, at the same waits and occupancy.
//! `ROCKE_DIRECT_DEBUG=nom0order` keeps the fences everywhere.

use super::lds_dma::is_dma;
use crate::rir::mir::*;

/// Whether `inst` is a fence only for M0: an LDS DMA load, or an
/// instruction whose physical registers are all M0 (the move of a DMA
/// load's address, a lane select) and that orders nothing else.
pub fn unfenced(inst: &MInst) -> bool {
    let flags = inst.op.flags();
    if flags & (F_BRANCH | F_END | F_DEF_EXEC | F_USE_VCC | F_SMEM) != 0 || matches!(inst.op, Opc::SBarrier | Opc::SSendmsg) {
        return false;
    }
    let mut phys = inst.defs.iter().chain(&inst.uses).filter_map(|o| match o {
        Opnd::P(p, _) if *p != Phys::Null => Some(*p),
        _ => None,
    });
    if is_dma(inst.op) {
        return phys.next().is_none();
    }
    let mut any = false;
    flags & F_FENCE == 0 && phys.all(|p| {
        any = true;
        p == Phys::M0
    }) && any
}

fn names_m0(opnds: &[Opnd]) -> bool {
    opnds.iter().any(|o| matches!(o, Opnd::P(Phys::M0, _)))
}

/// The orders of a block's M0 writes and reads so far.
#[derive(Default)]
pub struct M0Order {
    write: Option<usize>,
    reads: Vec<usize>,
}

impl M0Order {
    /// The order for gfx950 block `insts` when it has a DMA load and another
    /// VMEM load; `None` with `ROCKE_DIRECT_DEBUG=nom0order`.
    pub fn new(insts: &[MInst]) -> Option<M0Order> {
        let dma = insts.iter().any(|i| is_dma(i.op));
        let other = insts.iter().any(|i| i.op.flags() & F_VMEM_LOAD != 0 && !is_dma(i.op));
        (dma && other && !crate::rir::debug_flag("nom0order")).then(M0Order::default)
    }

    /// Records `insts[i]` and calls `edge(from, i, latency)` for each earlier
    /// M0 access it must follow.
    pub fn order(&mut self, insts: &[MInst], i: usize, latency: impl Fn(&MInst) -> u32, mut edge: impl FnMut(usize, usize, u32)) {
        let inst = &insts[i];
        if is_dma(inst.op) || inst.op == Opc::SSendmsg || names_m0(&inst.uses) {
            if let Some(w) = self.write {
                edge(w, i, latency(&insts[w]));
            }
            self.reads.push(i);
        }
        if names_m0(&inst.defs) {
            if let Some(w) = self.write {
                edge(w, i, 0);
            }
            for r in std::mem::take(&mut self.reads) {
                if r != i {
                    edge(r, i, 0);
                }
            }
            self.write = Some(i);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::gen_isa::Gfx9Op;
    use super::*;
    use crate::target::Target;

    fn r(reg: u32, n: u8) -> Opnd {
        Opnd::R(VRef { reg, off: 0, n })
    }

    fn m0(from: u32) -> MInst {
        MInst::new(Opc::SMovB32, [Opnd::P(Phys::M0, 1)], [r(from, 1)])
    }

    fn dma(voffset: u32) -> MInst {
        MInst::new(Opc::Gfx9(Gfx9Op::BufLoadLdsB128), [], [r(voffset, 1), r(40, 4), Opnd::Imm(0)]).with_imm(2 << 12).with_mem(1)
    }

    fn page(dst: u32) -> MInst {
        MInst::new(Opc::GLoadB32, [r(dst, 1)], [r(dst + 100, 2)]).with_mem(crate::rir::alias::ANY)
    }

    fn add(d: u32, s: u32) -> MInst {
        MInst::new(Opc::VAddNcU32, [r(d, 1)], [r(s, 1), Opnd::Imm(1)])
    }

    /// Two DMA loads, each after its page-table load and M0 move, then an LDS read; `extra` before the second
    /// page-table load. The order of the block's non-move instructions after scheduling, as names.
    fn scheduled(target: Target, extra: Vec<MInst>) -> Vec<String> {
        let mut insts = vec![page(1), add(2, 1), m0(30), dma(2)];
        insts.extend(extra);
        insts.extend([page(3), add(4, 3), m0(31), dma(4), MInst::new(Opc::DsLoadB32, [r(5, 1)], [r(6, 1)]).with_mem(1)]);
        let mut f = MFunc { target, ..MFunc::default() };
        f.blocks.push(MBlock { insts, loop_depth: 0 });
        crate::rir::post::schedule_window(&mut f, 64);
        let name = |i: &MInst| match (i.op, i.defs.first(), i.uses.first()) {
            (Opc::GLoadB32, Some(Opnd::R(v)), _) => format!("page{}", v.reg),
            (Opc::SMovB32, _, Some(Opnd::R(v))) => format!("m0<-s{}", v.reg),
            (op, _, Some(Opnd::R(v))) if is_dma(op) => format!("dma{}", v.reg),
            (Opc::DsLoadB32, ..) => "ds".into(),
            (Opc::GStoreB32, ..) => "store".into(),
            (Opc::SWaitcntExplicit, ..) => "wait".into(),
            _ => "add".into(),
        };
        f.blocks[0].insts.iter().map(name).filter(|n| n != "add").collect()
    }

    #[test]
    fn a_page_table_load_issues_before_the_dma_load_before_it() {
        let order = scheduled(Target::Gfx950, vec![]);
        let at = |n: &str| order.iter().position(|x| x == n).unwrap_or_else(|| panic!("{n} in {order:?}"));
        assert!(at("page3") < at("dma2"), "{order:?}");
        // M0's writes and reads, the DMA loads and the LDS read after them keep their order.
        assert!(at("m0<-s30") < at("dma2") && at("dma2") < at("m0<-s31") && at("m0<-s31") < at("dma4") && at("dma4") < at("ds"), "{order:?}");
    }

    #[test]
    fn a_global_write_or_an_explicit_wait_keeps_the_dma_loads_on_their_side() {
        let store = MInst::new(Opc::GStoreB32, [], [r(7, 2), r(8, 1)]).with_mem(2);
        let order = scheduled(Target::Gfx950, vec![store]);
        let at = |n: &str| order.iter().position(|x| x == n).unwrap_or_else(|| panic!("{n} in {order:?}"));
        assert!(at("dma2") < at("store") && at("store") < at("dma4"), "{order:?}");
        let wait = MInst::new(Opc::SWaitcntExplicit, [], []).with_imm(0x3f70);
        let order = scheduled(Target::Gfx950, vec![wait]);
        let at = |n: &str| order.iter().position(|x| x == n).unwrap_or_else(|| panic!("{n} in {order:?}"));
        assert!(at("dma2") < at("wait") && at("wait") < at("page3") && at("wait") < at("dma4"), "{order:?}");
    }

    #[test]
    fn a_block_of_dma_loads_alone_keeps_its_fences() {
        let insts = vec![add(2, 1), m0(30), dma(2), add(4, 3), m0(31), dma(4)];
        assert!(M0Order::new(&insts).is_none());
        let mut f = MFunc { target: Target::Gfx950, ..MFunc::default() };
        f.blocks.push(MBlock { insts: insts.clone(), loop_depth: 0 });
        crate::rir::post::schedule_window(&mut f, 64);
        let order: Vec<String> = f.blocks[0].insts.iter().map(|i| format!("{i:?}")).collect();
        assert_eq!(order, insts.iter().map(|i| format!("{i:?}")).collect::<Vec<_>>());
    }

    #[test]
    fn what_else_names_a_physical_register_stays_a_fence() {
        assert!(unfenced(&m0(30)) && unfenced(&dma(2)));
        let exec = MInst::new(Opc::SMovB32, [Opnd::P(Phys::ExecLo, 1)], [r(1, 1)]);
        let both = MInst::new(Opc::VWritelaneB32, [Opnd::P(Phys::V(0), 1)], [r(1, 1), Opnd::P(Phys::M0, 1)]);
        let wait = MInst::new(Opc::SWaitcntExplicit, [], []);
        assert!(!unfenced(&exec) && !unfenced(&both) && !unfenced(&wait) && !unfenced(&add(1, 2)));
        // A move into EXEC between keeps the page-table load after it.
        let order = scheduled(Target::Gfx950, vec![exec]);
        let at = |n: &str| order.iter().position(|x| x == n).unwrap_or_else(|| panic!("{n} in {order:?}"));
        assert!(at("dma2") < at("page3"), "{order:?}");
    }
}
