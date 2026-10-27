// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

//! The gfx950 hazard recognizer (design section 5.1): LLVM's
//! `GCNHazardRecognizer` for the predicates gfx950 has, on physical
//! registers after allocation and `waitcnt`. Before each consumer it puts
//! the `s_nop`s that give it the wait states its rules need since their
//! producers, counted over every path into its block (a wait state is one
//! issued instruction; `s_nop N` is N+1).
//!
//! | Rule | Producer -> consumer | Wait states |
//! |---|---|---|
//! | H1 | VALU writes an SGPR (VCC, EXEC, M0 included) -> VALU reads it, explicitly or as v_div_fmas' VCC | 2 |
//! | H2 | VALU writes EXEC -> v_readlane, v_readfirstlane, v_writelane | 4 |
//! | H3 | VALU writes a VGPR -> v_readlane / v_readfirstlane reads it as src0 | 1 |
//! | H4 | VALU writes an SGPR -> v_readlane / v_writelane takes it as the lane select | 4 |
//! | H5 | VALU writes an SGPR -> VMEM reads it | 5 |
//! | H6 | VALU writes VCC -> v_div_fmas | 4 |
//! | H7 | any write of a VGPR -> DPP reads it: 2; VALU writes EXEC -> DPP: 5 | 2 / 5 |
//! | H8 | transcendental VALU -> another VALU reads its result | 1 |
//! | H9 | a forwarded destination (SDWA dst_sel, VOP3 op_sel on the destination, v_fma_mixlo/hi, a VOP3P with src0's op_sel_hi set) -> a VALU naming it | 1 |
//! | H10 | VMEM store of more than 64 bits -> VALU overwrites its data (a buffer store with an SGPR soffset: 1) | 2 |
//! | H11 | XNACK soft clause: in a run of VMEM (or SMEM) instructions, a write of a register some of them read; a store after a load | 1 (breaks the run) |
//!
//! H9's VOP3P producers are COMGR's: its LLVM reads src0's op_sel_hi bit
//! (the encoding of a VOP3 `DST_OP_SEL`) on VOP3P too, so a packed op with
//! the default op_sel_hi gets a wait state before a VALU naming its result
//! (the corpus test finds 1,209 such COMGR sites in the G1 kernels); newer
//! LLVM keeps only the mixlo/mixhi half writes, and the older rule is the
//! conservative one.
//!
//! H12 (SALU writes M0 -> LDS DMA, `s_sendmsg`, `ds_*_addtid`), H13
//! (`s_setreg`), H14 (permlane swaps) and H16 (inline asm) have no
//! consumer or producer among the instructions the gfx950 back end emits
//! (`rules_without_instructions_on_gfx950` checks the description); H15,
//! the 64-bit shift bug, is `fix_shift64`.

use super::gen_isa::{self as t, Form, Role};
use crate::error::CompileError;
use crate::rir::mir::*;

/// A register range: VGPRs from 0, SGPRs from 256 (VCC at 256+106, M0 at
/// 256+124, EXEC at 256+126), AGPRs from 384.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Regs {
    pub start: u16,
    pub n: u16,
}

impl Regs {
    pub fn overlaps(self, o: Regs) -> bool {
        self.start < o.start + o.n && o.start < self.start + self.n
    }

    pub fn is_vgpr(self) -> bool {
        self.start < 256
    }

    pub fn is_sgpr(self) -> bool {
        (256..384).contains(&self.start)
    }
}

pub const VCC: Regs = Regs { start: 256 + 106, n: 2 };
pub const M0: Regs = Regs { start: 256 + 124, n: 1 };
pub const EXEC: Regs = Regs { start: 256 + 126, n: 2 };

pub fn regs(o: &Opnd) -> Option<Regs> {
    let (start, n) = match *o {
        Opnd::P(Phys::V(r), n) => (r, n as u16),
        Opnd::P(Phys::S(r), n) => (256 + r, n as u16),
        Opnd::P(Phys::VccLo, n) => (VCC.start, n as u16),
        Opnd::P(Phys::ExecLo, n) => (EXEC.start, n as u16),
        Opnd::P(Phys::M0, _) => (M0.start, 1),
        _ => return None,
    };
    Some(Regs { start, n: n.max(1) })
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Kind {
    #[default]
    Other,
    Valu,
    Salu,
    Smem,
    Vmem,
    Lds,
}

/// What the rules read about one instruction.
#[derive(Clone, Debug, Default)]
pub struct Fact {
    pub kind: Kind,
    /// Wait states it counts for: 1, or N+1 for `s_nop N`.
    pub wait_states: u32,
    pub nop: bool,
    pub trans: bool,
    pub dpp: bool,
    /// Its destination is forwarded through dst-sel logic (H9).
    pub forwards_dst: bool,
    pub div_fmas: bool,
    /// v_readlane / v_readfirstlane: src0 is the lane read (H2, H3).
    pub lane_read: bool,
    /// v_writelane (H2).
    pub writelane: bool,
    /// v_readlane / v_writelane: src1 is the lane select (H4).
    pub lane_select: bool,
    pub store: bool,
    /// The data of a store of more than 64 bits (H10), and whether it is a
    /// buffer store with an SGPR soffset.
    pub wide_data: Option<Regs>,
    pub buffer_sgpr_soffset: bool,
    pub defs: Vec<Regs>,
    /// Register sources, src0 first.
    pub uses: Vec<Regs>,
    pub src0: Option<Regs>,
    pub src1: Option<Regs>,
    /// Reads VCC without naming it (v_div_fmas).
    pub reads_vcc: bool,
}

/// The `Fact` of one of our instructions.
pub fn fact(inst: &MInst) -> Fact {
    let op = inst.op;
    let flags = op.flags();
    let info = t::info(op);
    let fmt = info.map_or(Fmt::Pseudo, |i| i.fmt);
    let form = info.map_or(Form::Plain, |i| i.form);
    let kind = match fmt {
        Fmt::Vop1 | Fmt::Vop2 | Fmt::Vop3 | Fmt::Vop3b | Fmt::Vopc | Fmt::Vop3p => Kind::Valu,
        Fmt::Sop1 | Fmt::Sop2 | Fmt::Sopc | Fmt::Sopk => Kind::Salu,
        Fmt::Smem => Kind::Smem,
        Fmt::Global | Fmt::Mubuf => Kind::Vmem,
        Fmt::Ds => Kind::Lds,
        _ => Kind::Other,
    };
    let store = flags & F_VMEM_STORE != 0;
    let data = match fmt {
        // A scratch store (a spill) has no address register: its data comes first.
        Fmt::Global if store && form == Form::Scratch => inst.uses.first(),
        Fmt::Global if store => inst.uses.get(if inst.mods & M_SADDR != 0 { 2 } else { 1 }),
        Fmt::Mubuf if store && inst.defs.is_empty() => inst.uses.first(),
        _ => None,
    };
    let soffset = t::signature(op).and_then(|s| s.uses.iter().position(|o| o.role == Role::Soffset));
    let sgpr_soffset = fmt == Fmt::Mubuf && soffset.and_then(|k| inst.uses.get(k)).and_then(regs).is_some_and(|r| r.is_sgpr());
    let uses: Vec<Regs> = inst.uses.iter().filter_map(regs).collect();
    Fact {
        kind,
        wait_states: if op == Opc::SNop { inst.imm as u32 + 1 } else if fmt == Fmt::Pseudo { 0 } else { 1 },
        nop: op == Opc::SNop,
        trans: flags & F_TRANS != 0,
        dpp: form == Form::Dpp16,
        forwards_dst: matches!(op, Opc::VFmaMixloF16 | Opc::VFmaMixhiF16)
            || (fmt == Fmt::Vop3 && inst.mods & M_OPSEL_DST != 0)
            || (fmt == Fmt::Vop3p && (form != Form::Mix || inst.mods & M_MIX_F16_0 != 0)),
        div_fmas: matches!(op, Opc::VDivFmasF32 | Opc::VDivFmasF64),
        lane_read: matches!(op, Opc::VReadlaneB32 | Opc::VReadfirstlaneB32),
        writelane: op == Opc::VWritelaneB32,
        lane_select: matches!(op, Opc::VReadlaneB32 | Opc::VWritelaneB32),
        store,
        wide_data: data.and_then(regs).filter(|r| r.n > 2),
        buffer_sgpr_soffset: sgpr_soffset,
        defs: inst.defs.iter().filter_map(regs).collect(),
        src0: inst.uses.first().and_then(regs),
        src1: inst.uses.get(1).and_then(regs),
        uses,
        reads_vcc: flags & F_USE_VCC != 0,
    }
}

/// Wait states from just before instruction `i` of block `b` back to the
/// nearest instruction `hit` accepts, the least over every path; `limit`
/// when there is none that close.
fn since(blocks: &[Vec<Fact>], preds: &[Vec<usize>], b: usize, i: usize, limit: u32, hit: &dyn Fn(&Fact) -> bool) -> u32 {
    let mut acc = 0;
    for f in blocks[b][..i].iter().rev() {
        if hit(f) {
            return acc;
        }
        acc += f.wait_states;
        if acc >= limit {
            return limit;
        }
    }
    let mut best = limit;
    let mut entered = vec![u32::MAX; blocks.len()];
    let mut stack = vec![(b, acc)];
    while let Some((blk, at)) = stack.pop() {
        for &p in &preds[blk] {
            if at >= entered[p] || at >= best {
                continue;
            }
            entered[p] = at;
            let mut acc = at;
            let mut open = true;
            for f in blocks[p].iter().rev() {
                if hit(f) {
                    best = best.min(acc);
                    open = false;
                    break;
                }
                acc += f.wait_states;
                if acc >= best {
                    open = false;
                    break;
                }
            }
            if open {
                stack.push((p, acc));
            }
        }
    }
    best
}

/// H11: whether `mem` would join a soft clause (the run of same-kind
/// memory instructions just before it, in layout order) that a register it
/// or the run writes is read in, or that a load starts and it stores to.
fn breaks_clause(blocks: &[Vec<Fact>], b: usize, i: usize, mem: &Fact) -> bool {
    let (mut defs, mut uses) = (Vec::new(), Vec::new());
    let earlier = blocks[..b].iter().rev().flat_map(|blk| blk.iter().rev());
    for f in blocks[b][..i].iter().rev().chain(earlier) {
        if f.nop || f.kind != mem.kind {
            break;
        }
        defs.extend_from_slice(&f.defs);
        uses.extend_from_slice(&f.uses);
    }
    if defs.is_empty() {
        return false;
    }
    if mem.store {
        return true;
    }
    defs.extend_from_slice(&mem.defs);
    uses.extend_from_slice(&mem.uses);
    defs.iter().any(|d| uses.iter().any(|u| u.overlaps(*d)))
}

fn valu_writes(r: Regs) -> impl Fn(&Fact) -> bool {
    move |p: &Fact| p.kind == Kind::Valu && p.defs.iter().any(|d| d.overlaps(r))
}

/// The wait states instruction `i` of block `b` needs before it.
pub fn demand(blocks: &[Vec<Fact>], preds: &[Vec<usize>], b: usize, i: usize) -> u32 {
    let f = &blocks[b][i];
    // A clause is broken by one wait state between it and `f`.
    let mut need = u32::from(matches!(f.kind, Kind::Vmem | Kind::Smem) && breaks_clause(blocks, b, i, f));
    let mut want = |w: u32, hit: &dyn Fn(&Fact) -> bool| {
        need = need.max(w - since(blocks, preds, b, i, w, hit));
    };
    if f.kind == Kind::Vmem {
        for &u in f.uses.iter().filter(|u| u.is_sgpr()) {
            want(5, &valu_writes(u));
        }
    }
    if f.kind == Kind::Valu {
        if !f.trans {
            want(1, &|p: &Fact| p.kind == Kind::Valu && p.trans && p.defs.first().is_some_and(|d| f.uses.iter().any(|u| u.overlaps(*d))));
        }
        want(1, &|p: &Fact| {
            p.forwards_dst && p.defs.first().is_some_and(|d| f.uses.iter().chain(&f.defs).any(|u| u.overlaps(*d)))
        });
        for &u in f.uses.iter().filter(|u| u.is_sgpr()) {
            want(2, &valu_writes(u));
        }
        if f.reads_vcc {
            want(2, &valu_writes(VCC));
        }
        if f.lane_read {
            if let Some(src0) = f.src0 {
                want(1, &valu_writes(src0));
            }
        }
        if f.lane_read || f.writelane {
            want(4, &valu_writes(EXEC));
        }
        for &d in f.defs.iter().filter(|d| d.is_vgpr()) {
            want(1, &|p: &Fact| p.wide_data.is_some_and(|w| w.overlaps(d)));
            want(2, &|p: &Fact| p.wide_data.is_some_and(|w| w.overlaps(d)) && !p.buffer_sgpr_soffset);
        }
    }
    if f.dpp {
        for &u in f.uses.iter().filter(|u| u.is_vgpr()) {
            want(2, &|p: &Fact| p.defs.iter().any(|d| d.overlaps(u)));
        }
        want(5, &valu_writes(EXEC));
    }
    if f.div_fmas {
        want(4, &valu_writes(VCC));
    }
    if f.lane_select {
        if let Some(select) = f.src1.filter(|r| r.is_sgpr()) {
            want(4, &valu_writes(select));
        }
    }
    need
}

/// The predecessors of each block.
pub fn predecessors(f: &MFunc) -> Vec<Vec<usize>> {
    let mut preds = vec![Vec::new(); f.blocks.len()];
    for b in 0..f.blocks.len() {
        for s in f.successors(b) {
            preds[s].push(b);
        }
    }
    preds
}

/// Puts `s_nop`s before every instruction that needs wait states.
pub fn insert_nops(f: &mut MFunc) {
    let preds = predecessors(f);
    let mut facts: Vec<Vec<Fact>> = f.blocks.iter().map(|b| b.insts.iter().map(fact).collect()).collect();
    for b in 0..f.blocks.len() {
        let mut i = 0;
        while i < facts[b].len() {
            let mut need = demand(&facts, &preds, b, i);
            while need > 0 {
                let n = need.min(8);
                let nop = MInst::new(Opc::SNop, [], []).with_imm(n as i64 - 1);
                facts[b].insert(i, fact(&nop));
                f.blocks[b].insts.insert(i, nop);
                need -= n;
                i += 1;
            }
            i += 1;
        }
    }
}

/// H15 (gfx90a and later): a 64-bit shift whose amount is the last VGPR of
/// an allocation granule (v7, v15, ...) with the next VGPR unused computes
/// wrongly. As LLVM fixes it: the amount moves to the low half of the
/// destination first, unless the destination is the shifted value too.
pub fn fix_shift64(f: &mut MFunc, num_vgprs: u32) -> Result<(), CompileError> {
    for block in &mut f.blocks {
        let mut k = 0;
        while k < block.insts.len() {
            let inst = &block.insts[k];
            if matches!(inst.op, Opc::VLshlrevB64 | Opc::VLshrrevB64 | Opc::VAshrrevI64) {
                if let (Some(Opnd::P(Phys::V(amt), _)), Some(&dst)) = (inst.uses.first().copied(), inst.defs.first()) {
                    let next_unused = amt as u32 + 1 >= num_vgprs;
                    if amt % 8 == 7 && next_unused {
                        let (Some(d), Some(src)) = (regs(&dst), inst.uses.get(1).and_then(regs)) else {
                            return Err(CompileError::new("ROCKE_DIRECT_ENCODE", "64-bit shift with a non-register operand"));
                        };
                        if d.overlaps(src) {
                            return Err(CompileError::new(
                                "ROCKE_DIRECT_UNSUPPORTED",
                                "an in-place 64-bit shift by v(8k+7), the last VGPR, on gfx950",
                            ));
                        }
                        let Opnd::P(Phys::V(lo), _) = dst else { unreachable!("a VGPR destination") };
                        let mov = MInst::new(Opc::VMovB32, [Opnd::P(Phys::V(lo), 1)], [Opnd::P(Phys::V(amt), 1)]);
                        block.insts[k].uses[0] = Opnd::P(Phys::V(lo), 1);
                        block.insts.insert(k, mov);
                        k += 1;
                    }
                }
            }
            k += 1;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v(r: u16, n: u8) -> Opnd {
        Opnd::P(Phys::V(r), n)
    }

    fn s(r: u16, n: u8) -> Opnd {
        Opnd::P(Phys::S(r), n)
    }

    const VCC2: Opnd = Opnd::P(Phys::VccLo, 2);
    const EXEC2: Opnd = Opnd::P(Phys::ExecLo, 2);

    /// The `s_nop` immediates put in front of each instruction of one block.
    fn nops(insts: Vec<MInst>) -> Vec<(usize, i64)> {
        let mut f = MFunc { target: crate::target::Target::Gfx950, ..MFunc::default() };
        f.blocks = vec![MBlock { insts, loop_depth: 0 }];
        insert_nops(&mut f);
        let mut out = Vec::new();
        let mut k = 0;
        for i in &f.blocks[0].insts {
            if i.op == Opc::SNop {
                out.push((k, i.imm));
            } else {
                k += 1;
            }
        }
        out
    }

    fn valu(d: Opnd, a: Opnd, b: Opnd) -> MInst {
        MInst::new(Opc::VAddF32, [d], [a, b])
    }

    #[test]
    fn h1_valu_sgpr_write_then_valu_read() {
        let cmp = MInst::new(Opc::VCmp(0x14), [VCC2], [v(1, 1), v(2, 1)]);
        let sel = MInst::new(Opc::VCndmaskB32, [v(3, 1)], [v(1, 1), v(2, 1), VCC2]);
        assert_eq!(nops(vec![cmp.clone(), sel.clone()]), [(1, 1)]);
        assert_eq!(nops(vec![cmp.clone(), valu(v(9, 1), v(8, 1), v(8, 1)), sel.clone()]), [(2, 0)]);
        // A carry SGPR pair, and an SGPR source of an add.
        let add = MInst::new(Opc::VAddCoU32, [v(4, 1), s(6, 2)], [v(1, 1), v(2, 1)]);
        let addc = MInst::new(Opc::VAddCoCiU32, [v(5, 1), s(8, 2)], [v(1, 1), v(2, 1), s(6, 2)]);
        assert_eq!(nops(vec![add, addc]), [(1, 1)]);
        let rfl = MInst::new(Opc::VReadfirstlaneB32, [s(4, 1)], [v(1, 1)]);
        assert_eq!(nops(vec![rfl.clone(), valu(v(2, 1), s(4, 1), v(3, 1))]), [(1, 1)]);
        // A scalar write of the SGPR is no hazard.
        let smov = MInst::new(Opc::SMovB32, [s(4, 1)], [Opnd::Imm(1)]);
        assert!(nops(vec![smov, valu(v(2, 1), s(4, 1), v(3, 1))]).is_empty());
    }

    #[test]
    fn h2_h3_h4_lane_reads_and_writes() {
        // VALU writes EXEC, then v_readfirstlane: 4.
        let cmpx = MInst::new(Opc::VCmp(0x14), [EXEC2], [v(1, 1), v(2, 1)]);
        let rfl = MInst::new(Opc::VReadfirstlaneB32, [s(4, 1)], [v(1, 1)]);
        assert_eq!(nops(vec![cmpx, rfl.clone()]), [(1, 3)]);
        // VALU writes the lane read: 1.
        assert_eq!(nops(vec![valu(v(1, 1), v(2, 1), v(3, 1)), rfl.clone()]), [(1, 0)]);
        // VALU writes the lane select: 4 (from v_readfirstlane, a VALU).
        let rl = MInst::new(Opc::VReadlaneB32, [s(6, 1)], [v(2, 1), s(4, 1)]);
        assert_eq!(nops(vec![rfl, rl]), [(1, 3)]);
    }

    #[test]
    fn h5_valu_sgpr_write_then_vmem_read() {
        let rfl = MInst::new(Opc::VReadfirstlaneB32, [s(4, 1)], [v(1, 1)]);
        let load = MInst::new(Opc::GLoadB32, [v(2, 1)], [v(3, 1), s(4, 2)]).with_mods(M_SADDR);
        assert_eq!(nops(vec![rfl, load]), [(1, 4)]);
    }

    #[test]
    fn h6_valu_vcc_write_then_div_fmas() {
        let scale = MInst::new(Opc::VDivScaleF32, [v(1, 1), VCC2], [v(2, 1), v(2, 1), v(3, 1)]);
        let fmas = MInst::new(Opc::VDivFmasF32, [v(4, 1)], [v(1, 1), v(2, 1), v(3, 1)]);
        assert_eq!(nops(vec![scale, fmas.clone()]), [(1, 3)]);
        // VCC written by s_mov_b64 (as the compiler does it) is no hazard.
        let mov = MInst::new(Opc::SMovB64, [VCC2], [s(4, 2)]);
        assert!(nops(vec![mov, fmas]).is_empty());
    }

    #[test]
    fn h7_dpp_reads() {
        let dpp = MInst::new(Opc::VMovB32Dpp, [v(2, 1)], [v(1, 1)]).with_imm(0x00ff_08b1);
        assert_eq!(nops(vec![valu(v(1, 1), v(3, 1), v(4, 1)), dpp.clone()]), [(1, 1)]);
        // Any write counts, a load's too.
        let load = MInst::new(Opc::GLoadB32, [v(1, 1)], [v(6, 2)]);
        assert_eq!(nops(vec![load, dpp.clone()]), [(1, 1)]);
        let cmpx = MInst::new(Opc::VCmp(0x14), [EXEC2], [v(5, 1), v(6, 1)]);
        assert_eq!(nops(vec![cmpx, dpp]), [(1, 4)]);
    }

    #[test]
    fn h8_trans_result_then_valu_read() {
        let exp = MInst::new(Opc::VExpF32, [v(1, 1)], [v(2, 1)]);
        assert_eq!(nops(vec![exp.clone(), valu(v(3, 1), v(1, 1), v(4, 1))]), [(1, 0)]);
        // A transcendental reading it is no hazard; an unrelated VALU none.
        assert!(nops(vec![exp.clone(), MInst::new(Opc::VRcpF32, [v(3, 1)], [v(1, 1)])]).is_empty());
        assert!(nops(vec![exp, valu(v(3, 1), v(5, 1), v(4, 1))]).is_empty());
    }

    #[test]
    fn h9_forwarded_destination() {
        let mix = MInst::new(Opc::VFmaMixloF16, [v(1, 1)], [v(2, 1), v(3, 1), v(4, 1)]);
        assert_eq!(nops(vec![mix.clone(), valu(v(5, 1), v(1, 1), v(6, 1))]), [(1, 0)]);
        // Overwriting it is a hazard too.
        assert_eq!(nops(vec![mix, valu(v(1, 1), v(7, 1), v(6, 1))]), [(1, 0)]);
        // A packed op (op_sel_hi set, as the encoder writes it), as COMGR's LLVM has it.
        let packed = MInst::new(Opc::VPkAddF16, [v(1, 1)], [v(2, 1), v(3, 1)]);
        assert_eq!(nops(vec![packed.clone(), valu(v(5, 1), v(1, 1), v(6, 1))]), [(1, 0)]);
        assert!(nops(vec![packed, valu(v(5, 1), v(7, 1), v(6, 1))]).is_empty());
    }

    #[test]
    fn h10_wide_store_data_overwritten() {
        let store = MInst::new(Opc::GStoreB128, [], [v(0, 2), v(4, 4)]);
        assert_eq!(nops(vec![store.clone(), valu(v(5, 1), v(8, 1), v(9, 1))]), [(1, 1)]);
        assert_eq!(nops(vec![store.clone(), valu(v(9, 1), v(8, 1), v(9, 1)), valu(v(6, 1), v(8, 1), v(9, 1))]), [(2, 0)]);
        // 64 bits or less is no hazard.
        let narrow = MInst::new(Opc::GStoreB64, [], [v(0, 2), v(4, 2)]);
        assert!(nops(vec![narrow, valu(v(5, 1), v(8, 1), v(9, 1))]).is_empty());
        // A buffer store with an SGPR soffset needs one.
        let buf = MInst::new(Opc::BufStoreB128, [], [v(4, 4), v(1, 1), s(8, 4), s(12, 1)]);
        assert_eq!(nops(vec![buf, valu(v(5, 1), v(8, 1), v(9, 1))]), [(1, 0)]);
        // A spill's scratch store: its data is its only operand.
        let spill = MInst::new(Opc::ScratchStoreB128, [], [v(4, 4)]).with_imm(16);
        assert_eq!(nops(vec![spill.clone(), valu(v(7, 1), v(8, 1), v(9, 1))]), [(1, 1)]);
        assert!(nops(vec![spill, valu(v(8, 1), v(8, 1), v(9, 1))]).is_empty());
    }

    #[test]
    fn h11_soft_clauses() {
        // The second load reads what the first writes.
        let a = MInst::new(Opc::GLoadB32, [v(2, 1)], [v(2, 2)]);
        let b = MInst::new(Opc::GLoadB32, [v(5, 1)], [v(2, 2)]);
        assert_eq!(nops(vec![a.clone(), b.clone()]), [(1, 0)]);
        // A load alone, or independent loads, form no hazard.
        assert!(nops(vec![a.clone()]).is_empty());
        let c = MInst::new(Opc::GLoadB32, [v(6, 1)], [v(8, 2)]);
        let d = MInst::new(Opc::GLoadB32, [v(7, 1)], [v(10, 2)]);
        assert!(nops(vec![c.clone(), d]).is_empty());
        // A store after a load in the run starts a new clause.
        let st = MInst::new(Opc::GStoreB32, [], [v(10, 2), v(12, 1)]);
        assert_eq!(nops(vec![c, st.clone()]), [(1, 0)]);
        // Any other instruction ends the run.
        assert!(nops(vec![a, valu(v(9, 1), v(8, 1), v(8, 1)), b]).is_empty());
        // Scalar loads alike.
        let s1 = MInst::new(Opc::SLoadB64, [s(4, 2)], [s(0, 2)]);
        let s2 = MInst::new(Opc::SLoadB64, [s(6, 2)], [s(4, 2)]);
        assert_eq!(nops(vec![s1, s2]), [(1, 0)]);
    }

    #[test]
    fn waits_count_over_every_path_into_a_block() {
        // b0: v_cmp vcc; s_cbranch_execz -> b2. b1: one VALU. b2: v_cndmask vcc.
        // The path b0 -> b2 has one wait state (the branch): one more is needed.
        let mut f = MFunc { target: crate::target::Target::Gfx950, ..MFunc::default() };
        let cmp = MInst::new(Opc::VCmp(0x14), [VCC2], [v(1, 1), v(2, 1)]);
        let branch = MInst::new(Opc::SCbranchExecz, [], []).with_imm(2);
        let sel = MInst::new(Opc::VCndmaskB32, [v(3, 1)], [v(1, 1), v(2, 1), VCC2]);
        f.blocks = vec![
            MBlock { insts: vec![cmp, branch], loop_depth: 0 },
            MBlock { insts: vec![valu(v(9, 1), v(8, 1), v(8, 1))], loop_depth: 0 },
            MBlock { insts: vec![sel], loop_depth: 0 },
        ];
        insert_nops(&mut f);
        assert_eq!(f.blocks[2].insts[0].op, Opc::SNop);
        assert_eq!(f.blocks[2].insts[0].imm, 0);
    }

    #[test]
    fn one_nop_meets_every_rule_of_a_consumer() {
        // v_readfirstlane writes s4 and reads v1 just written: the load of
        // s[4:5] needs 5 (H5), the readlane before it 1 (H3).
        let add = valu(v(1, 1), v(2, 1), v(3, 1));
        let rfl = MInst::new(Opc::VReadfirstlaneB32, [s(4, 1)], [v(1, 1)]);
        let load = MInst::new(Opc::GLoadB32, [v(2, 1)], [v(3, 1), s(4, 2)]).with_mods(M_SADDR);
        assert_eq!(nops(vec![add, rfl, load]), [(1, 0), (2, 4)]);
    }

    #[test]
    fn h15_shift_amount_in_the_last_vgpr() {
        let shift = MInst::new(Opc::VLshlrevB64, [v(2, 2)], [v(7, 1), v(4, 2)]);
        let mut f = MFunc { target: crate::target::Target::Gfx950, ..MFunc::default() };
        f.blocks = vec![MBlock { insts: vec![shift.clone()], loop_depth: 0 }];
        fix_shift64(&mut f, 8).unwrap_or_else(|e| panic!("{e}"));
        let code = &f.blocks[0].insts;
        assert_eq!((code.len(), code[0].op, code[0].uses[0]), (2, Opc::VMovB32, v(7, 1)));
        assert_eq!(code[1].uses[0], v(2, 1));
        // With v8 allocated, nothing changes; in place, it is refused.
        f.blocks = vec![MBlock { insts: vec![shift], loop_depth: 0 }];
        fix_shift64(&mut f, 9).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(f.blocks[0].insts.len(), 1);
        f.blocks = vec![MBlock { insts: vec![MInst::new(Opc::VLshlrevB64, [v(4, 2)], [v(7, 1), v(4, 2)])], loop_depth: 0 }];
        assert!(fix_shift64(&mut f, 8).is_err());
    }

    /// H12, H13, H14 and H16 need instructions the gfx950 back end does not
    /// have: if a row for one appears, its rule must come with it.
    #[test]
    fn rules_without_instructions_on_gfx950() {
        let unruled = ["s_setreg", "s_getreg", "s_sendmsg", "permlane", "_addtid", "v_cmpx", "lds", "v_accvgpr", "mfma"];
        for op in t::ALL {
            let Some(name) = t::mnemonic(op) else { continue };
            let emitted = !matches!(op, Opc::SSendmsg);
            if emitted {
                assert!(!unruled.iter().any(|u| name.contains(u)), "{name} needs its hazard rule");
            }
        }
        // s_sendmsg has a row but nothing selects it for gfx950 (the VGPR
        // release message is gfx11's).
        assert!(t::info(Opc::SSendmsg).is_some());
    }
}
