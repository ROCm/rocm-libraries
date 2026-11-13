// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

//! Packed f32 VALU instructions that an MFMA reads right after: two scalar ones instead, on gfx950.
//!
//! `pack` pairs two f32 `v_mul_f32` / `v_add_f32` / `v_fma_f32` into `v_pk_mul_f32` / `v_pk_add_f32` /
//! `v_pk_fma_f32` before scheduling. A kernel that scales MFMA accumulators held in VGPRs that way (the fused KDA
//! chunk scan: the decay of its state, read as C by the next MFMA a few instructions later) got stale or torn lanes
//! out of the MFMA: about 1 element in 10^4 of the state, 8 registers x 16 lanes of the 16 the scale reached last,
//! differently on every launch (found by bisection on silicon: one `s_nop` before the MFMA, 6 to 12 wait states
//! between the two, or no packed instruction at all made the output identical launch to launch; 16 wait states
//! broke it again, so it is a timing alignment of the packed write with the matrix pipeline and no wait-state count
//! fixes it; two scalar multiplies into the same registers never failed). LLVM's recognizer has no rule for it, and
//! COMGR's code, which keeps accumulators in AGPRs that VALU cannot write, never has the sequence.
//!
//! After scheduling and allocation, a `v_pk_{mul,add,fma}_f32` whose destination an MFMA of the same block reads
//! before anything writes it again becomes the two scalar instructions that computed each lane, whatever the
//! distance: the failure came back whenever a nop count or a window kept some of them packed (a distance limit of
//! 7 instructions left the 256-thread kernel failing; the attention kernels, which have such a packed write 12 to
//! 62 instructions before the MFMA that reads it, always passed and now change by at most 3% more instructions).
//! The lanes compute exactly what they did, so nothing numeric changes. `ROCKE_DIRECT_DEBUG=nounpack` keeps every
//! packed instruction.

use super::gen_isa::Gfx9Op;
use crate::rir::mir::*;

/// Unpacks the packed f32 instructions of `f` that an MFMA of their block reads; returns how many.
pub fn run(f: &mut MFunc) -> usize {
    if crate::rir::debug_flag("nounpack") {
        return 0;
    }
    let mut n = 0;
    for block in f.blocks.iter_mut() {
        if !block.insts.iter().any(|i| matches!(i.op, Opc::Gfx9(Gfx9Op::VPkMulF32 | Gfx9Op::VPkAddF32 | Gfx9Op::VPkFmaF32))) {
            continue;
        }
        let insts = std::mem::take(&mut block.insts);
        let mut out = Vec::with_capacity(insts.len() + 8);
        for (i, inst) in insts.iter().enumerate() {
            match unpack_if_read(&insts, i) {
                Some(two) => {
                    out.extend(two);
                    n += 1;
                }
                None => out.push(inst.clone()),
            }
        }
        block.insts = out;
    }
    n
}

fn overlaps(o: &Opnd, lo: u16, n: u16) -> bool {
    matches!(o, Opnd::P(Phys::V(r), k) if *r < lo + n && lo < *r + *k as u16)
}

/// The two scalar instructions for `insts[i]` when it is a packed f32 instruction an MFMA reads.
fn unpack_if_read(insts: &[MInst], i: usize) -> Option<[MInst; 2]> {
    let inst = &insts[i];
    let op = match inst.op {
        Opc::Gfx9(Gfx9Op::VPkMulF32) => Opc::VMulF32,
        Opc::Gfx9(Gfx9Op::VPkAddF32) => Opc::VAddF32,
        Opc::Gfx9(Gfx9Op::VPkFmaF32) => Opc::VFmaF32,
        _ => return None,
    };
    let Opnd::P(Phys::V(d), 2) = inst.defs[0] else { return None };
    let read = insts[i + 1..]
        .iter()
        .find_map(|x| {
            if super::agpr::is_mfma(x.op) && x.uses.iter().any(|u| overlaps(u, d, 2)) {
                Some(true)
            } else if x.defs.iter().any(|y| overlaps(y, d, 2)) {
                Some(false)
            } else {
                None
            }
        })
        .unwrap_or(false);
    if !read {
        return None;
    }
    lanes(inst, op, d)
}

/// `inst` (a packed instruction writing the pair `d`, `d + 1`) as one scalar instruction per lane: each source is
/// the half of its pair the lane selects (`imm`: op_sel for the low lane, the inverse of bits 6:4 for the high one),
/// a one-dword source is read by both lanes, the negations apply to both. `None` for a form that has no such
/// reading (a source that is not a register, a high lane that reads what the low lane wrote).
fn lanes(inst: &MInst, op: Opc, d: u16) -> Option<[MInst; 2]> {
    let sel = inst.imm;
    if !(0..=0x77).contains(&sel) || sel & 0x08 != 0 {
        return None;
    }
    let (low, high) = (sel as u32 & 7, !(sel as u32 >> 4) & 7);
    let mut srcs: [Vec<Opnd>; 2] = [Vec::new(), Vec::new()];
    for (k, o) in inst.uses.iter().enumerate() {
        match *o {
            Opnd::P(Phys::V(r), 2) if r % 2 == 0 => {
                srcs[0].push(Opnd::P(Phys::V(r + (low >> k & 1) as u16), 1));
                srcs[1].push(Opnd::P(Phys::V(r + (high >> k & 1) as u16), 1));
            }
            Opnd::P(Phys::S(r), 2) => {
                srcs[0].push(Opnd::P(Phys::S(r + (low >> k & 1) as u16), 1));
                srcs[1].push(Opnd::P(Phys::S(r + (high >> k & 1) as u16), 1));
            }
            Opnd::P(Phys::V(_) | Phys::S(_), 1) => {
                srcs[0].push(*o);
                srcs[1].push(*o);
            }
            _ => return None,
        }
    }
    // The low lane's result is not what the high lane reads.
    let first = Opnd::P(Phys::V(d), 1);
    if srcs[1].contains(&first) {
        return None;
    }
    let mods = inst.mods & (M_NEG0 | M_NEG1 | M_NEG2);
    let scalar = |lane: usize, dst: u16| {
        let mut uses = Uses::new();
        srcs[lane].iter().for_each(|&o| uses.push(o));
        MInst { op, defs: Defs::from([Opnd::P(Phys::V(dst), 1)]), uses, mods, imm: 0, ..inst.clone() }
    };
    Some([scalar(0, d), scalar(1, d + 1)])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v(r: u16, n: u8) -> Opnd {
        Opnd::P(Phys::V(r), n)
    }

    fn pk(op: Gfx9Op, d: u16, a: Opnd, b: Opnd) -> MInst {
        MInst::new(Opc::Gfx9(op), [v(d, 2)], [a, b])
    }

    fn mfma(c: u16) -> MInst {
        MInst::new(Opc::Gfx9(Gfx9Op::VMfma32x32x16Bf16), [v(c, 16)], [v(40, 4), v(44, 4), v(c, 16)])
    }

    fn filler() -> MInst {
        MInst::new(Opc::VAddF32, [v(200, 1)], [v(201, 1), v(202, 1)])
    }

    fn unpacked(insts: Vec<MInst>) -> Vec<MInst> {
        let mut f = MFunc { target: crate::target::Target::Gfx950, ..MFunc::default() };
        f.blocks = vec![MBlock { insts, loop_depth: 0 }];
        run(&mut f);
        f.blocks.remove(0).insts
    }

    #[test]
    fn a_scaled_accumulator_pair_an_mfma_reads_becomes_two_multiplies() {
        // v[100:101] = v[100:101] * v66.hi (op_sel on the second source: both lanes read v67), read by the MFMA.
        let scale = pk(Gfx9Op::VPkMulF32, 100, v(100, 2), v(66, 2)).with_imm(0x02);
        let code = unpacked(vec![scale, filler(), mfma(100)]);
        assert_eq!(code.iter().map(|i| i.op).collect::<Vec<_>>(), [Opc::VMulF32, Opc::VMulF32, Opc::VAddF32, Opc::Gfx9(Gfx9Op::VMfma32x32x16Bf16)]);
        // lane 0 reads the low half of the first source and (op_sel) the high of the second, lane 1 the high half of
        // the first and, op_sel_hi being set, the high of the second
        assert_eq!((code[0].defs[0], &code[0].uses[..]), (v(100, 1), &[v(100, 1), v(67, 1)][..]));
        assert_eq!((code[1].defs[0], &code[1].uses[..]), (v(101, 1), &[v(101, 1), v(67, 1)][..]));
    }

    #[test]
    fn a_pair_read_however_far_off_becomes_scalar_and_one_that_is_not_read_stays() {
        let scale = || pk(Gfx9Op::VPkMulF32, 100, v(100, 2), v(66, 2));
        let packed = Opc::Gfx9(Gfx9Op::VPkMulF32);
        let mut far = vec![scale()];
        far.extend((0..40).map(|_| filler()));
        far.push(mfma(100));
        assert_eq!(unpacked(far)[0].op, Opc::VMulF32);
        assert_eq!(unpacked(vec![scale(), filler()])[0].op, packed);
        // an MFMA that reads other registers, and a write of the pair before the read
        assert_eq!(unpacked(vec![scale(), mfma(116)])[0].op, packed);
        let rewritten = MInst::new(Opc::VMovB32, [v(100, 1)], [Opnd::Imm(0)]);
        assert_eq!(unpacked(vec![scale(), rewritten, mfma(100)])[0].op, packed);
    }

    #[test]
    fn negations_and_scalar_sources_carry_over() {
        let add = pk(Gfx9Op::VPkAddF32, 100, v(100, 2), Opnd::P(Phys::S(8), 2)).with_mods(M_NEG1);
        let code = unpacked(vec![add, mfma(100)]);
        assert_eq!(code[0].op, Opc::VAddF32);
        assert_eq!((code[0].mods, &code[0].uses[..]), (M_NEG1, &[v(100, 1), Opnd::P(Phys::S(8), 1)][..]));
        assert_eq!(&code[1].uses[..], &[v(101, 1), Opnd::P(Phys::S(9), 1)][..]);
    }

    #[test]
    fn a_pair_whose_high_lane_reads_the_low_lanes_result_stays() {
        // lane 1 reads v100 (op_sel_hi cleared on the first source: bits 6:4 = 0x10), which lane 0 writes.
        let swap = pk(Gfx9Op::VPkMulF32, 100, v(100, 2), v(66, 2)).with_imm(0x10);
        assert_eq!(unpacked(vec![swap, mfma(100)])[0].op, Opc::Gfx9(Gfx9Op::VPkMulF32));
    }
}
