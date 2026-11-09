// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

//! Uniform-address integer atomic adds on gfx950, one per wave, as LLVM's
//! atomic optimizer makes them in COMGR's code: the first active lane adds
//! the wave's total with one atomic, and each lane's result is the old value
//! plus the values of the active lanes below it, which is what the lanes' own
//! atomics, issued in order of lane, would have returned. A uniform value
//! totals to the lane count times it; another is summed in a loop over the
//! active lanes, which also records each lane's prefix when the result is
//! read.
//!
//! A lane's prefix is selected into it with a compare against the lane index
//! rather than written with `v_writelane_b32`, whose value and lane select gfx9
//! takes as two SGPRs only with the lane select in M0.

use super::gen_isa::Gfx9Op;
use crate::rir::mir::*;

/// The code of a wave-reduced atomic add, for the selector to append: blocks
/// in order, the first continuing the block the atomic is in, and whether each
/// is in the lane loop. A branch names its target by an index into `blocks`.
pub struct Lowered {
    pub blocks: Vec<Vec<MInst>>,
    pub in_loop: Vec<bool>,
}

/// `v_cmp_eq_u32` (the shared compare code).
const CMP_EQ_U32: u8 = 0x4a;

fn op(opc: Opc, defs: &[Opnd], uses: &[Opnd]) -> MInst {
    MInst::new(opc, defs, uses)
}

fn r(v: VRef) -> Opnd {
    Opnd::R(v)
}

fn half(v: VRef, k: u8) -> Opnd {
    Opnd::R(VRef { off: v.off + k, n: 1, ..v })
}

/// The wave's active lanes adding `value` (a VGPR, or an SGPR or constant the
/// same in every lane) at one uniform address: `atomic(data, dst)` makes the
/// atomic instruction for one lane's `data`, returning the old value into
/// `dst` when given. `result`, when given, gets each lane's old value.
/// `fresh(class, dwords)` makes a virtual register.
pub fn reduced_add(
    value: Opnd,
    uniform: bool,
    result: Option<VRef>,
    mut fresh: impl FnMut(Class, u8) -> VRef,
    atomic: impl FnOnce(Opnd, Option<VRef>) -> MInst,
) -> Lowered {
    let exec = Opnd::P(Phys::ExecLo, 2);
    let mut blocks = vec![Vec::new()];
    let mut in_loop = vec![false];
    let (total, prefix) = if uniform {
        let b = &mut blocks[0];
        let each = fresh(Class::S, 1);
        b.push(op(Opc::SMovB32, &[r(each)], &[value]));
        let lanes = fresh(Class::S, 1);
        b.push(op(Opc::Gfx9(Gfx9Op::SBcnt1I32B64), &[r(lanes)], &[exec]));
        let total = fresh(Class::S, 1);
        b.push(op(Opc::SMulI32, &[r(total)], &[r(each), r(lanes)]));
        (total, result.is_some().then_some(each))
    } else {
        let rest = fresh(Class::S, 2);
        let total = fresh(Class::S, 1);
        blocks[0].push(op(Opc::SMovB64, &[r(rest)], &[exec]));
        blocks[0].push(op(Opc::SMovB32, &[r(total)], &[Opnd::Imm(0)]));
        let lanes = result.map(|_| {
            let (lane, below) = (fresh(Class::V, 1), fresh(Class::V, 1));
            let prefix = fresh(Class::V, 1);
            blocks[0].push(op(Opc::VMbcntLoU32B32, &[r(below)], &[Opnd::Imm(u32::MAX), Opnd::Imm(0)]));
            blocks[0].push(op(Opc::VMbcntHiU32B32, &[r(lane)], &[Opnd::Imm(u32::MAX), r(below)]));
            blocks[0].push(op(Opc::VMovB32, &[r(prefix)], &[Opnd::Imm(0)]));
            (lane, prefix)
        });
        let mut body = Vec::new();
        let (l, v, bit) = (fresh(Class::S, 1), fresh(Class::S, 1), fresh(Class::S, 2));
        body.push(op(Opc::Gfx9(Gfx9Op::SFf1I32B64), &[r(l)], &[r(rest)]));
        body.push(op(Opc::VReadlaneB32, &[r(v)], &[value, r(l)]));
        if let Some((lane, prefix)) = lanes {
            let (here, sum) = (fresh(Class::S, 2), fresh(Class::V, 1));
            body.push(op(Opc::VCmp(CMP_EQ_U32), &[r(here)], &[r(l), r(lane)]));
            body.push(op(Opc::VMovB32, &[r(sum)], &[r(total)]));
            body.push(op(Opc::VCndmaskB32, &[r(prefix)], &[r(prefix), r(sum), r(here)]));
        }
        body.push(op(Opc::SAddI32, &[r(total)], &[r(total), r(v)]));
        body.push(op(Opc::SLshlB64, &[r(bit)], &[Opnd::Imm(1), r(l)]));
        body.push(op(Opc::SAndNot1B64, &[r(rest)], &[r(rest), r(bit)]));
        body.push(op(Opc::SCmpLgU64, &[], &[r(rest), Opnd::Imm(0)]));
        body.push(op(Opc::SCbranchScc1, &[], &[]).with_imm(1));
        blocks.push(body);
        in_loop.push(true);
        blocks.push(Vec::new());
        in_loop.push(false);
        (total, lanes.map(|(_, prefix)| prefix))
    };
    let b = blocks.last_mut().expect("a block");
    let (first, one, save, data) = (fresh(Class::S, 1), fresh(Class::S, 2), fresh(Class::S, 2), fresh(Class::V, 1));
    b.push(op(Opc::Gfx9(Gfx9Op::SFf1I32B64), &[r(first)], &[exec]));
    b.push(op(Opc::SLshlB64, &[r(one)], &[Opnd::Imm(1), r(first)]));
    b.push(op(Opc::VMovB32, &[r(data)], &[r(total)]));
    b.push(op(Opc::SAndSaveexecB64, &[r(save), exec], &[r(one), exec]));
    let old = result.map(|_| fresh(Class::V, 1));
    b.push(atomic(r(data), old));
    b.push(op(Opc::SMovB64, &[exec], &[r(save)]));
    if let (Some(res), Some(old), Some(prefix)) = (result, old, prefix) {
        let base = fresh(Class::S, 1);
        b.push(op(Opc::VReadlaneB32, &[r(base)], &[r(old), r(first)]));
        if uniform {
            // The lanes below this one each added the same value.
            let (mask, below, lanes, add) = (fresh(Class::S, 2), fresh(Class::V, 1), fresh(Class::V, 1), fresh(Class::V, 1));
            b.push(op(Opc::SMovB64, &[r(mask)], &[exec]));
            b.push(op(Opc::VMbcntLoU32B32, &[r(below)], &[half(mask, 0), Opnd::Imm(0)]));
            b.push(op(Opc::VMbcntHiU32B32, &[r(lanes)], &[half(mask, 1), r(below)]));
            b.push(op(Opc::VMulLoU32, &[r(add)], &[r(lanes), r(prefix)]));
            b.push(op(Opc::VAddNcU32, &[r(res)], &[r(base), r(add)]));
        } else {
            b.push(op(Opc::VAddNcU32, &[r(res)], &[r(base), r(prefix)]));
        }
    }
    Lowered { blocks, in_loop }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    /// One wave running `lowered`: SGPRs and lane masks by virtual register dword, VGPRs per lane, EXEC; the
    /// atomic adds lane data to `*memory` and returns the old value. Returns the atomics issued.
    fn run(lowered: &Lowered, exec: u64, lanes: &HashMap<u32, [u32; 64]>, memory: &mut u32) -> (u32, HashMap<u32, [u32; 64]>) {
        let (mut s, mut v, mut ex, mut scc) = (HashMap::<(u32, u8), u32>::new(), lanes.clone(), exec, false);
        let (mut block, mut i, mut atomics) = (0usize, 0usize, 0u32);
        let sval = |s: &HashMap<(u32, u8), u32>, ex: u64, o: &Opnd| -> u32 {
            match *o {
                Opnd::Imm(x) => x,
                Opnd::R(r) => s[&(r.reg, r.off)],
                Opnd::P(Phys::ExecLo, _) => ex as u32,
                _ => panic!("operand {o:?}"),
            }
        };
        let pair = |s: &HashMap<(u32, u8), u32>, ex: u64, o: &Opnd| -> u64 {
            match *o {
                Opnd::Imm(x) => u64::from(x),
                Opnd::R(r) => u64::from(s[&(r.reg, r.off)]) | u64::from(s[&(r.reg, r.off + 1)]) << 32,
                Opnd::P(Phys::ExecLo, 2) => ex,
                _ => panic!("operand {o:?}"),
            }
        };
        let reg = |o: &Opnd| match *o {
            Opnd::R(r) => r,
            _ => panic!("not a register: {o:?}"),
        };
        loop {
            let Some(inst) = lowered.blocks[block].get(i) else {
                if block + 1 == lowered.blocks.len() {
                    return (atomics, v);
                }
                (block, i) = (block + 1, 0);
                continue;
            };
            i += 1;
            let lane_v = |v: &HashMap<u32, [u32; 64]>, o: &Opnd, l: usize, s: &HashMap<(u32, u8), u32>, ex: u64| match *o {
                Opnd::R(r) if v.contains_key(&r.reg) => v[&r.reg][l],
                _ => sval(s, ex, o),
            };
            let set_pair = |s: &mut HashMap<(u32, u8), u32>, d: &Opnd, x: u64, ex: &mut u64| match *d {
                Opnd::P(Phys::ExecLo, 2) => *ex = x,
                _ => {
                    let r = reg(d);
                    s.insert((r.reg, r.off), x as u32);
                    s.insert((r.reg, r.off + 1), (x >> 32) as u32);
                }
            };
            let valu = |v: &mut HashMap<u32, [u32; 64]>, ex: u64, d: VRef, f: &dyn Fn(usize) -> u32| {
                let mut out = *v.get(&d.reg).unwrap_or(&[0; 64]);
                for (l, slot) in out.iter_mut().enumerate().filter(|(l, _)| ex >> l & 1 != 0) {
                    *slot = f(l);
                }
                v.insert(d.reg, out);
            };
            match inst.op {
                Opc::SMovB32 => {
                    let x = sval(&s, ex, &inst.uses[0]);
                    s.insert((reg(&inst.defs[0]).reg, 0), x);
                }
                Opc::SMovB64 => {
                    let x = pair(&s, ex, &inst.uses[0]);
                    set_pair(&mut s, &inst.defs[0], x, &mut ex);
                }
                Opc::Gfx9(Gfx9Op::SBcnt1I32B64) => {
                    s.insert((reg(&inst.defs[0]).reg, 0), pair(&s, ex, &inst.uses[0]).count_ones());
                }
                Opc::Gfx9(Gfx9Op::SFf1I32B64) => {
                    let m = pair(&s, ex, &inst.uses[0]);
                    s.insert((reg(&inst.defs[0]).reg, 0), if m == 0 { u32::MAX } else { m.trailing_zeros() });
                }
                Opc::SMulI32 | Opc::SAddI32 => {
                    let (a, b) = (sval(&s, ex, &inst.uses[0]), sval(&s, ex, &inst.uses[1]));
                    let x = if inst.op == Opc::SMulI32 { a.wrapping_mul(b) } else { a.wrapping_add(b) };
                    s.insert((reg(&inst.defs[0]).reg, 0), x);
                }
                Opc::SLshlB64 => {
                    let x = pair(&s, ex, &inst.uses[0]) << (sval(&s, ex, &inst.uses[1]) & 63);
                    set_pair(&mut s, &inst.defs[0], x, &mut ex);
                }
                Opc::SAndNot1B64 => {
                    let x = pair(&s, ex, &inst.uses[0]) & !pair(&s, ex, &inst.uses[1]);
                    set_pair(&mut s, &inst.defs[0], x, &mut ex);
                }
                Opc::SCmpLgU64 => scc = pair(&s, ex, &inst.uses[0]) != pair(&s, ex, &inst.uses[1]),
                Opc::SCbranchScc1 => {
                    if scc {
                        (block, i) = (inst.imm as usize, 0);
                    }
                }
                Opc::SAndSaveexecB64 => {
                    let m = pair(&s, ex, &inst.uses[0]);
                    let saved = ex;
                    set_pair(&mut s, &inst.defs[0], saved, &mut ex);
                    ex &= m;
                }
                Opc::VMbcntLoU32B32 | Opc::VMbcntHiU32B32 => {
                    let (mask, add) = (sval(&s, ex, &inst.uses[0]), inst.uses[1]);
                    let hi = inst.op == Opc::VMbcntHiU32B32;
                    let snapshot = v.clone();
                    valu(&mut v, ex, reg(&inst.defs[0]), &|l| {
                        let below = if hi { u64::from(mask) << 32 } else { u64::from(mask) } & ((1u64 << l) - 1);
                        (below.count_ones()).wrapping_add(lane_v(&snapshot, &add, l, &s, ex))
                    });
                }
                Opc::VMovB32 => {
                    let snapshot = v.clone();
                    valu(&mut v, ex, reg(&inst.defs[0]), &|l| lane_v(&snapshot, &inst.uses[0], l, &s, ex));
                }
                Opc::VReadlaneB32 => {
                    let l = (sval(&s, ex, &inst.uses[1]) & 63) as usize;
                    s.insert((reg(&inst.defs[0]).reg, 0), v[&reg(&inst.uses[0]).reg][l]);
                }
                Opc::VCmp(CMP_EQ_U32) => {
                    let mut m = 0u64;
                    for l in (0..64).filter(|l| ex >> l & 1 != 0) {
                        m |= u64::from(lane_v(&v, &inst.uses[0], l, &s, ex) == lane_v(&v, &inst.uses[1], l, &s, ex)) << l;
                    }
                    set_pair(&mut s, &inst.defs[0], m, &mut ex);
                }
                Opc::VCndmaskB32 => {
                    let (snapshot, m) = (v.clone(), pair(&s, ex, &inst.uses[2]));
                    valu(&mut v, ex, reg(&inst.defs[0]), &|l| lane_v(&snapshot, &inst.uses[usize::from(m >> l & 1 != 0)], l, &s, ex));
                }
                Opc::VAddNcU32 | Opc::VMulLoU32 => {
                    let snapshot = v.clone();
                    let mul = inst.op == Opc::VMulLoU32;
                    valu(&mut v, ex, reg(&inst.defs[0]), &|l| {
                        let (a, b) = (lane_v(&snapshot, &inst.uses[0], l, &s, ex), lane_v(&snapshot, &inst.uses[1], l, &s, ex));
                        if mul { a.wrapping_mul(b) } else { a.wrapping_add(b) }
                    });
                }
                Opc::GAtomicAddU32 | Opc::GAtomicAddU32Rtn => {
                    let data = inst.uses.last().expect("data");
                    for l in (0..64).filter(|l| ex >> l & 1 != 0) {
                        atomics += 1;
                        let old = *memory;
                        *memory = old.wrapping_add(lane_v(&v, data, l, &s, ex));
                        if let Some(d) = inst.defs.first() {
                            let mut out = *v.get(&reg(d).reg).unwrap_or(&[0; 64]);
                            out[l] = old;
                            v.insert(reg(d).reg, out);
                        }
                    }
                }
                other => panic!("unexpected {other:?}"),
            }
        }
    }

    #[test]
    fn one_atomic_per_wave_returns_each_lane_its_in_order_value() {
        let mut seed = 0x2545_f491_4f6c_dd1du64;
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        for trial in 0..40 {
            let exec = match trial % 4 {
                0 => u64::MAX,
                1 => 1 << (next() % 64),
                _ => next() | 1 << (next() % 64),
            };
            let uniform = trial % 3 == 0;
            let mut regs = 100u32;
            let mut fresh = |_: Class, _: u8| {
                regs += 1;
                VRef { reg: regs, off: 0, n: 1 }
            };
            let mut lanes = HashMap::new();
            let (value, each) = if uniform {
                (Opnd::Imm(next() as u32 % 1000), [0; 64])
            } else {
                let values: [u32; 64] = std::array::from_fn(|_| next() as u32 % 1000);
                lanes.insert(1, values);
                (Opnd::R(VRef { reg: 1, off: 0, n: 1 }), values)
            };
            let result = VRef { reg: 2, off: 0, n: 1 };
            let atomic = |data: Opnd, dst: Option<VRef>| match dst {
                Some(d) => MInst::new(Opc::GAtomicAddU32Rtn, [Opnd::R(d)], [data]),
                None => MInst::new(Opc::GAtomicAddU32, [], [data]),
            };
            let lowered = reduced_add(value, uniform, Some(result), &mut fresh, atomic);
            let start = (next() as u32) % 5000;
            let mut memory = start;
            let (atomics, v) = run(&lowered, exec, &lanes, &mut memory);
            assert_eq!(atomics, 1, "trial {trial}");
            // As each active lane's own atomic, in order of lane.
            let mut want = start;
            for l in (0..64).filter(|l| exec >> l & 1 != 0) {
                assert_eq!(v[&2][l], want, "trial {trial} lane {l}");
                want = want.wrapping_add(if uniform { match value { Opnd::Imm(x) => x, _ => unreachable!() } } else { each[l] });
            }
            assert_eq!(memory, want, "trial {trial}");
        }
    }
}
