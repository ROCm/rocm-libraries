// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

//! VOPD dual issue: two independent VALU operations encoded as one
//! `v_dual_<x> :: v_dual_<y>` instruction, which issues in a single cycle.

use super::isel::{is_inline_f32, is_inline_int};
use super::mir::*;

/// How many places after an instruction its partner may be.
const LOOKAHEAD: usize = 4;

/// A partner that moves up past other instructions must not read a result of
/// the last `RECENT` instructions, which may still be in flight.
const RECENT: usize = 6;

/// A VALU operation in the form VOPD encodes: `dst = op(src0, vsrc1)`, or
/// `dst = src0` for a move.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Part {
    pub code: u8,
    /// The operation is only encodable as the Y half.
    pub y_only: bool,
    pub dst: u16,
    pub src0: Opnd,
    pub vsrc1: Option<u16>,
}

/// The VOPD form of `inst`, if it has one.
pub fn part(inst: &MInst) -> Option<Part> {
    let (code, y_only) = match inst.op {
        Opc::VMulF32 => (3, false),
        Opc::VAddF32 => (4, false),
        Opc::VSubF32 => (5, false),
        Opc::VSubrevF32 => (6, false),
        Opc::VMovB32 => (8, false),
        Opc::VMaxF32 => (10, false),
        Opc::VMinF32 => (11, false),
        Opc::VAddNcU32 => (16, true),
        Opc::VLshlrevB32 => (17, true),
        Opc::VAndB32 => (18, true),
        _ => return None,
    };
    let [Opnd::P(Phys::V(dst), 1)] = inst.defs[..] else {
        return None;
    };
    let (src0, vsrc1) = match (inst.op, &inst.uses[..]) {
        (Opc::VMovB32, &[src0]) => (src0, None),
        (Opc::VMovB32, _) => return None,
        (_, &[src0, Opnd::P(Phys::V(r), 1)]) => (src0, Some(r)),
        _ => return None,
    };
    let src0_ok = matches!(src0, Opnd::P(Phys::V(_) | Phys::S(_), 1) | Opnd::Imm(_));
    (inst.mods == 0 && src0_ok).then_some(Part { code, y_only, dst, src0, vsrc1 })
}

fn literal(o: &Opnd) -> Option<u32> {
    match *o {
        Opnd::Imm(v) if !is_inline_int(v) && !is_inline_f32(v) => Some(v),
        _ => None,
    }
}

fn vgpr(o: &Opnd) -> Option<u16> {
    match *o {
        Opnd::P(Phys::V(r), _) => Some(r),
        _ => None,
    }
}

/// `first` and the later `second` as the X and Y halves of one VOPD, if the
/// encoding allows it: each source slot reads its two VGPRs from different
/// banks (register number mod 4), one destination is even and the other odd,
/// both halves share at most one literal, and a Y-only operation is Y. Both
/// halves read their sources before either writes, so `second` must not read
/// `first`'s result.
pub fn pair(first: &Part, second: &Part) -> Option<(Part, Part)> {
    let banks_differ = |a: Option<u16>, b: Option<u16>| match (a, b) {
        (Some(a), Some(b)) => a % 4 != b % 4,
        _ => true,
    };
    let one_literal = match (literal(&first.src0), literal(&second.src0)) {
        (Some(a), Some(b)) => a == b,
        _ => true,
    };
    let reads_first = vgpr(&second.src0) == Some(first.dst) || second.vsrc1 == Some(first.dst);
    if reads_first
        || (first.dst ^ second.dst) & 1 == 0
        || !banks_differ(vgpr(&first.src0), vgpr(&second.src0))
        || !banks_differ(first.vsrc1, second.vsrc1)
        || !one_literal
    {
        return None;
    }
    match (first.y_only, second.y_only) {
        (false, _) => Some((*first, *second)),
        (true, false) => Some((*second, *first)),
        (true, true) => None,
    }
}

/// The VOPD instruction that issues `x` and `y`.
pub fn dual(x: &Part, y: &Part) -> MInst {
    let mut uses = Uses::new();
    for p in [x, y] {
        uses.push(p.src0);
        if let Some(r) = p.vsrc1 {
            uses.push(Opnd::P(Phys::V(r), 1));
        }
    }
    let defs = [Opnd::P(Phys::V(x.dst), 1), Opnd::P(Phys::V(y.dst), 1)];
    let mut inst = MInst::new(Opc::VDual, defs, uses).with_imm(x.code as i64 | (y.code as i64) << 8);
    inst.mods = 1 + u32::from(x.vsrc1.is_some());
    inst
}

/// Register keys of an operand: SGPRs (VCC, M0, EXEC and SCC at their
/// operand codes), VGPRs with bit 16 set.
fn keys(o: &Opnd) -> std::ops::Range<u32> {
    let (start, n) = match *o {
        Opnd::P(Phys::V(r), n) => (0x10000 | r as u32, n as u32),
        Opnd::P(Phys::S(r), n) => (r as u32, n as u32),
        Opnd::P(Phys::VccLo, _) => (106, 1),
        Opnd::P(Phys::M0, _) => (125, 1),
        Opnd::P(Phys::ExecLo, _) => (126, 1),
        Opnd::P(Phys::Scc, _) => (253, 1),
        _ => (0, 0),
    };
    start..start + n
}

fn overlap(a: &[Opnd], b: &[Opnd]) -> bool {
    a.iter().map(keys).any(|x| b.iter().map(keys).any(|y| x.start < y.end && y.start < x.end))
}

/// Whether the VALU instruction `x` may move above `k`: `k` writes nothing
/// `x` reads or writes and reads nothing `x` writes, and it is not a wait,
/// branch, barrier or other SOPP, nor a write of EXEC, which every VALU reads.
fn may_pass(x: &MInst, k: &MInst) -> bool {
    let writes_exec = k.op.flags() & F_DEF_EXEC != 0 || k.defs.iter().any(|d| matches!(d, Opnd::P(Phys::ExecLo, _)));
    k.op.fmt() != Fmt::Sopp
        && !writes_exec
        && !overlap(&k.defs, &x.uses)
        && !overlap(&k.defs, &x.defs)
        && !overlap(&k.uses, &x.defs)
}

/// Joins VALU instructions that have a VOPD form into dual-issue pairs. An
/// instruction takes as its partner the nearest later one, at most
/// `LOOKAHEAD` places on, that pairs with it and may move up to it. Runs on
/// physical registers after wait insertion, which it leaves valid because no
/// instruction moves past a wait; the hazard and ALU-delay passes follow it.
pub fn form_pairs(f: &mut MFunc) {
    for block in f.blocks.iter_mut() {
        let insts = std::mem::take(&mut block.insts);
        let mut moved = vec![false; insts.len()];
        let mut out: Vec<MInst> = Vec::with_capacity(insts.len());
        for i in 0..insts.len() {
            if moved[i] {
                continue;
            }
            let partner = part(&insts[i]).and_then(|a| {
                let end = insts.len().min(i + 1 + LOOKAHEAD);
                let mut passed = 0;
                for j in i + 1..end {
                    if moved[j] {
                        continue;
                    }
                    if let Some(xy) = part(&insts[j]).and_then(|b| pair(&a, &b)) {
                        let passes = (i + 1..j).filter(|&k| !moved[k]).all(|k| may_pass(&insts[j], &insts[k]));
                        let fresh =
                            passed > 0 && out.iter().rev().take(RECENT).any(|p| overlap(&p.defs, &insts[j].uses));
                        if passes && !fresh {
                            return Some((j, xy));
                        }
                    }
                    if insts[j].op.fmt() == Fmt::Sopp {
                        break;
                    }
                    passed += 1;
                }
                None
            });
            match partner {
                Some((j, (x, y))) => {
                    moved[j] = true;
                    out.push(dual(&x, &y));
                }
                None => out.push(insts[i].clone()),
            }
        }
        block.insts = out;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v(r: u16) -> Opnd {
        Opnd::P(Phys::V(r), 1)
    }

    fn s(r: u16) -> Opnd {
        Opnd::P(Phys::S(r), 1)
    }

    fn bin(op: Opc, dst: u16, src0: Opnd, src1: u16) -> MInst {
        MInst::new(op, [v(dst)], [src0, v(src1)])
    }

    fn mov(dst: u16, src0: Opnd) -> MInst {
        MInst::new(Opc::VMovB32, [v(dst)], [src0])
    }

    fn words(inst: &MInst) -> Vec<u32> {
        let f = MFunc { blocks: vec![MBlock { insts: vec![inst.clone()], loop_depth: 0 }], vregs: Vec::new() };
        super::super::asm::assemble(&f)
            .expect("encodes")
            .chunks(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    fn joined(first: &MInst, second: &MInst) -> Option<MInst> {
        let (x, y) = pair(&part(first)?, &part(second)?)?;
        Some(dual(&x, &y))
    }

    #[test]
    fn eligible_forms() {
        assert!(part(&bin(Opc::VMulF32, 0, v(1), 2)).is_some());
        assert!(part(&bin(Opc::VAddNcU32, 0, Opnd::Imm(0x1234_5678), 2)).is_some_and(|p| p.y_only));
        assert!(part(&mov(3, s(4))).is_some());
        // Modifiers, an SGPR or constant second source, VCC as src0 and
        // operations without a VOPD form are not encodable.
        assert!(part(&bin(Opc::VMulF32, 0, v(1), 2).with_mods(M_NEG0)).is_none());
        assert!(part(&MInst::new(Opc::VMulF32, [v(0)], [v(1), s(2)])).is_none());
        assert!(part(&MInst::new(Opc::VAddF32, [v(0)], [v(1), Opnd::Imm(1)])).is_none());
        assert!(part(&mov(0, Opnd::P(Phys::VccLo, 1))).is_none());
        assert!(part(&bin(Opc::VSubNcU32, 0, v(1), 2)).is_none());
        assert!(part(&MInst::new(Opc::VFmaF32, [v(0)], [v(1), v(2), v(3)])).is_none());
    }

    #[test]
    fn pairing_rules() {
        let mul = bin(Opc::VMulF32, 0, v(1), 2);
        assert!(joined(&mul, &bin(Opc::VAddF32, 3, v(4), 5)).is_some());
        // Same src0 bank, same vsrc1 bank, same destination parity.
        assert!(joined(&mul, &bin(Opc::VAddF32, 3, v(5), 7)).is_none());
        assert!(joined(&mul, &bin(Opc::VAddF32, 3, v(4), 6)).is_none());
        assert!(joined(&mul, &bin(Opc::VAddF32, 4, v(6), 7)).is_none());
        // The second reads the first's result; the reverse is allowed.
        assert!(joined(&mul, &bin(Opc::VAddF32, 3, v(0), 5)).is_none());
        assert!(joined(&mul, &bin(Opc::VAddF32, 3, v(4), 0)).is_none());
        assert!(joined(&mul, &bin(Opc::VAddF32, 1, v(4), 5)).is_some());
        // SGPR and constant sources have no bank; two literals must agree.
        let lit = |k| Opnd::Imm(k);
        assert!(joined(&bin(Opc::VMulF32, 0, s(1), 2), &bin(Opc::VAddF32, 3, s(1), 7)).is_some());
        assert!(joined(&bin(Opc::VMulF32, 0, lit(0x1234), 2), &bin(Opc::VAddF32, 3, lit(0x1234), 7)).is_some());
        assert!(joined(&bin(Opc::VMulF32, 0, lit(0x1234), 2), &bin(Opc::VAddF32, 3, lit(0x5678), 7)).is_none());
        assert!(joined(&bin(Opc::VMulF32, 0, lit(0x1234), 2), &bin(Opc::VAddF32, 3, lit(64), 7)).is_some());
        // A Y-only operation becomes Y whatever its order; two cannot pair.
        let add = bin(Opc::VAddNcU32, 0, v(1), 2);
        let x_first = joined(&add, &bin(Opc::VSubF32, 3, v(4), 5)).expect("pairs");
        assert_eq!(x_first.imm, 5 | 16 << 8);
        assert_eq!(x_first.defs[..], [v(3), v(0)]);
        assert!(joined(&add, &bin(Opc::VAndB32, 3, v(4), 5)).is_none());
    }

    // Expected words from `llvm-mc -arch=amdgcn -mcpu=gfx1151 -show-encoding`.
    #[test]
    fn encodings() {
        let cases: [(MInst, MInst, &[u32]); 7] = [
            // v_dual_mul_f32 v0, v1, v2 :: v_dual_add_f32 v3, v4, v5
            (bin(Opc::VMulF32, 0, v(1), 2), bin(Opc::VAddF32, 3, v(4), 5), &[0xC8C8_0501, 0x0002_0B04]),
            // v_dual_mov_b32 v0, s1 :: v_dual_add_nc_u32 v3, 5, v4
            (mov(0, s(1)), bin(Opc::VAddNcU32, 3, Opnd::Imm(5), 4), &[0xCA20_0001, 0x0002_0885]),
            // v_dual_sub_f32 v6, 0x40490fdb, v1 :: v_dual_subrev_f32 v7, 0x40490fdb, v2
            (
                bin(Opc::VSubF32, 6, Opnd::Imm(0x4049_0fdb), 1),
                bin(Opc::VSubrevF32, 7, Opnd::Imm(0x4049_0fdb), 2),
                &[0xC94C_02FF, 0x0606_04FF, 0x4049_0FDB],
            ),
            // v_dual_max_f32 v10, v11, v12 :: v_dual_min_f32 v13, v14, v15
            (bin(Opc::VMaxF32, 10, v(11), 12), bin(Opc::VMinF32, 13, v(14), 15), &[0xCA96_190B, 0x0A0C_1F0E]),
            // v_dual_mul_f32 v200, s5, v201 :: v_dual_lshlrev_b32 v255, 2, v254
            (
                bin(Opc::VMulF32, 200, s(5), 201),
                bin(Opc::VLshlrevB32, 255, Opnd::Imm(2), 254),
                &[0xC8E3_9205, 0xC8FF_FC82],
            ),
            // v_dual_add_f32 v8, 1.0, v9 :: v_dual_and_b32 v1, s3, v6
            (
                bin(Opc::VAddF32, 8, Opnd::Imm(0x3f80_0000), 9),
                bin(Opc::VAndB32, 1, s(3), 6),
                &[0xC924_12F2, 0x0800_0C03],
            ),
            // v_dual_mov_b32 v0, v1 :: v_dual_mov_b32 v3, v6
            (mov(0, v(1)), mov(3, v(6)), &[0xCA10_0101, 0x0002_0106]),
        ];
        for (first, second, expected) in cases {
            let inst = joined(&first, &second).expect("pairs");
            assert_eq!(words(&inst), expected, "{first:?} :: {second:?}");
        }
    }

    fn block(insts: Vec<MInst>) -> MFunc {
        MFunc { blocks: vec![MBlock { insts, loop_depth: 0 }], vregs: Vec::new() }
    }

    #[test]
    fn partner_moves_up_past_independent_instructions_only() {
        let mul = bin(Opc::VMulF32, 0, v(1), 2);
        let add = bin(Opc::VAddF32, 3, v(4), 5);
        let other = MInst::new(Opc::VMulLoU32, [v(20)], [v(21), v(22)]);
        let mut f = block(vec![mul.clone(), other.clone(), add.clone()]);
        form_pairs(&mut f);
        let ops: Vec<Opc> = f.blocks[0].insts.iter().map(|i| i.op).collect();
        assert_eq!(ops, [Opc::VDual, Opc::VMulLoU32]);
        // The partner reads what the instruction in between writes.
        let feeds = MInst::new(Opc::VMulLoU32, [v(4)], [v(21), v(22)]);
        let mut f = block(vec![mul.clone(), feeds, add.clone()]);
        form_pairs(&mut f);
        assert!(f.blocks[0].insts.iter().all(|i| i.op != Opc::VDual));
        // A wait in between.
        let wait = MInst::new(Opc::SWaitcnt, [], []).with_imm(0);
        let mut f = block(vec![mul.clone(), wait, add.clone()]);
        form_pairs(&mut f);
        assert!(f.blocks[0].insts.iter().all(|i| i.op != Opc::VDual));
        // A store in between reads the partner's destination.
        let store = MInst::new(Opc::DsStoreB32, [], [v(30), v(3)]);
        let mut f = block(vec![mul.clone(), store, add.clone()]);
        form_pairs(&mut f);
        assert!(f.blocks[0].insts.iter().all(|i| i.op != Opc::VDual));
        // Adjacent pairs form even when a source was just written, but a
        // partner reading a result that recent does not move up.
        let fresh = bin(Opc::VAddF32, 4, v(8), 9);
        let mut f = block(vec![fresh.clone(), mul.clone(), add.clone()]);
        form_pairs(&mut f);
        assert_eq!(f.blocks[0].insts.len(), 2);
        assert_eq!(f.blocks[0].insts[1].op, Opc::VDual);
        let mut f = block(vec![fresh, mul, other, add]);
        form_pairs(&mut f);
        assert!(f.blocks[0].insts.iter().all(|i| i.op != Opc::VDual));
    }

    fn mix(h: u64, x: u64) -> u64 {
        (h ^ x).wrapping_mul(0x100_0000_01b3).rotate_left(29) ^ 0x9e37_79b9_7f4a_7c15
    }

    const OPS: [Opc; 10] = [
        Opc::VMulF32,
        Opc::VAddF32,
        Opc::VSubF32,
        Opc::VSubrevF32,
        Opc::VMovB32,
        Opc::VMaxF32,
        Opc::VMinF32,
        Opc::VAddNcU32,
        Opc::VLshlrevB32,
        Opc::VAndB32,
    ];

    /// Final register values and the ordered side effects of a block, as
    /// value numbers: a VALU result is a function of its operation and the
    /// values it reads (EXEC included); other instructions are effects. A
    /// VOPD reads all four sources before writing either result.
    fn run(insts: &[MInst]) -> (Vec<(u32, u64)>, Vec<u64>) {
        let mut regs: std::collections::BTreeMap<u32, u64> = Default::default();
        let mut effects = Vec::new();
        let read = |regs: &std::collections::BTreeMap<u32, u64>, o: &Opnd| -> u64 {
            match o {
                Opnd::Imm(v) => mix(1, *v as u64),
                _ => keys(o).fold(2, |h, k| mix(h, *regs.get(&k).unwrap_or(&(k as u64 + 1000)))),
            }
        };
        let name = |op: Opc| op.name().bytes().fold(5, |h, b| mix(h, b as u64));
        let valu = |regs: &std::collections::BTreeMap<u32, u64>, op: Opc, srcs: &[Opnd]| -> u64 {
            let exec = *regs.get(&126).unwrap_or(&7);
            srcs.iter().fold(mix(name(op), exec), |h, o| mix(h, read(regs, o)))
        };
        for inst in insts {
            if inst.op == Opc::VDual {
                let (nx, cx, cy) = (inst.mods as usize, inst.imm as u8, (inst.imm >> 8) as u8);
                let op_of = |c: u8| *OPS.iter().find(|&&op| part_code(op) == c).expect("component");
                let x = valu(&regs, op_of(cx), &inst.uses[..nx]);
                let y = valu(&regs, op_of(cy), &inst.uses[nx..]);
                regs.insert(keys(&inst.defs[0]).start, x);
                regs.insert(keys(&inst.defs[1]).start, y);
            } else if OPS.contains(&inst.op) || inst.op == Opc::VMulLoU32 {
                let value = valu(&regs, inst.op, &inst.uses);
                regs.insert(keys(&inst.defs[0]).start, value);
            } else {
                let h = inst.uses.iter().fold(mix(name(inst.op), inst.imm as u64), |h, o| mix(h, read(&regs, o)));
                effects.push(h);
                for (k, d) in inst.defs.iter().enumerate() {
                    regs.insert(keys(d).start, mix(h, k as u64));
                }
                if inst.op.flags() & F_DEF_EXEC != 0 {
                    regs.insert(126, mix(h, 99));
                }
            }
        }
        (regs.into_iter().collect(), effects)
    }

    fn part_code(op: Opc) -> u8 {
        part(&bin(op, 0, v(1), 2)).or_else(|| part(&mov(0, v(1)))).expect("eligible").code
    }

    #[test]
    fn pairing_preserves_block_semantics() {
        let mut seed = 0x0123_4567_89ab_cdefu64;
        let mut next = |n: u32| {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (seed >> 33) as u32 % n
        };
        let mut paired = 0;
        for _ in 0..3000 {
            let mut insts = Vec::new();
            for _ in 0..24 {
                let vr = |next: &mut dyn FnMut(u32) -> u32| v(next(12) as u16);
                let src0 = |next: &mut dyn FnMut(u32) -> u32| match next(4) {
                    0 => s(next(3) as u16),
                    1 => Opnd::Imm([3, 0x3f80_0000, 0x1234, 0x5678][next(4) as usize]),
                    _ => vr(next),
                };
                let inst = match next(12) {
                    0 => MInst::new(Opc::VMulLoU32, [vr(&mut next)], [vr(&mut next), vr(&mut next)]),
                    1 => MInst::new(Opc::DsLoadB32, [vr(&mut next)], [vr(&mut next)]),
                    2 => MInst::new(Opc::DsStoreB32, [], [vr(&mut next), vr(&mut next)]),
                    3 if next(3) == 0 => MInst::new(Opc::SWaitcnt, [], []).with_imm(0),
                    3 if next(2) == 0 => MInst::new(Opc::SAndSaveexecB32, [s(next(3) as u16)], [s(next(3) as u16)]),
                    3 => MInst::new(Opc::SMovB32, [Opnd::P(Phys::ExecLo, 1)], [s(next(3) as u16)]),
                    4 => MInst::new(Opc::SAddU32, [s(next(3) as u16)], [s(next(3) as u16), Opnd::Imm(1)]),
                    _ => {
                        let op = OPS[next(OPS.len() as u32) as usize];
                        if op == Opc::VMovB32 {
                            mov(next(12) as u16, src0(&mut next))
                        } else {
                            let (dst, a) = (next(12) as u16, src0(&mut next));
                            bin(op, dst, a, next(12) as u16)
                        }
                    }
                };
                insts.push(inst);
            }
            let mut f = block(insts.clone());
            form_pairs(&mut f);
            paired += f.blocks[0].insts.iter().filter(|i| i.op == Opc::VDual).count();
            assert_eq!(run(&f.blocks[0].insts), run(&insts), "{:#?}", f.blocks[0].insts);
        }
        assert!(paired > 1000, "{paired} pairs");
    }
}
