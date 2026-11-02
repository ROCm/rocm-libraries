// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

//! Three-operand integer min / max / median on gfx950 (lever `minmax3`,
//! `crate::rir::lever`; PLAN decision 18), as LLVM's `min3` / `max3` / `med3`
//! selection: an integer `v_max` (`v_min`) whose operand is the only use of
//! an earlier `v_max` (`v_min`) of the same signedness in its block becomes
//! one `v_max3` (`v_min3`) and the inner one goes; a `v_min(v_max(x, lo), hi)`
//! or `v_max(v_min(x, hi), lo)` with constants `lo <= hi` becomes
//! `v_med3(x, lo, hi)` (the HC stages' amax scans and exponent clamps). Runs on
//! virtual registers, before allocation; the inner operands must not change
//! between the two instructions and EXEC must not change. Integer min / max
//! are associative and commutative, so the result is bitwise the pair's.

use crate::rir::mir::*;

use super::dpp::overlaps;
use super::gen_isa::Gfx9Op;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    MaxI,
    MinI,
    MaxU,
    MinU,
}

fn kind(op: Opc) -> Option<Kind> {
    match op {
        Opc::VMaxI32 => Some(Kind::MaxI),
        Opc::VMinI32 => Some(Kind::MinI),
        Opc::VMaxU32 => Some(Kind::MaxU),
        Opc::VMinU32 => Some(Kind::MinU),
        _ => None,
    }
}

fn three(k: Kind) -> Opc {
    Opc::Gfx9(match k {
        Kind::MaxI => Gfx9Op::VMax3I32,
        Kind::MinI => Gfx9Op::VMin3I32,
        Kind::MaxU => Gfx9Op::VMax3U32,
        Kind::MinU => Gfx9Op::VMin3U32,
    })
}

/// `lo <= hi` in the signedness of `k`.
fn ordered(k: Kind, lo: u32, hi: u32) -> bool {
    match k {
        Kind::MaxI | Kind::MinI => (lo as i32) <= (hi as i32),
        Kind::MaxU | Kind::MinU => lo <= hi,
    }
}

/// Folds every eligible pair; returns how many instructions went.
pub fn run(f: &mut MFunc) -> usize {
    let n = f.vregs.len();
    let mut uses = vec![0u32; n];
    let mut defs = vec![0u32; n];
    for inst in f.blocks.iter().flat_map(|b| &b.insts) {
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
    let mut folded = 0;
    for b in 0..f.blocks.len() {
        let len = f.blocks[b].insts.len();
        let mut dead = vec![false; len];
        // Where in this block each single-def VGPR is defined.
        let mut at: std::collections::HashMap<u32, usize> = std::collections::HashMap::new();
        for k in 0..len {
            let outer = f.blocks[b].insts[k].clone();
            if let [Opnd::R(d)] = outer.defs[..] {
                if defs[d.reg as usize] == 1 && d.off == 0 && d.n == 1 {
                    at.insert(d.reg, k);
                }
            }
            let Some(ko) = kind(outer.op) else { continue };
            if outer.mods != 0 || outer.uses.len() != 2 || !f.lever(&outer, "minmax3") {
                continue;
            }
            // The inner instruction: one operand of `outer`, defined once in this block, used only here.
            for j in 0..2 {
                let (x, other) = (outer.uses[j], outer.uses[1 - j]);
                let Opnd::R(xv) = x else { continue };
                if xv.off != 0 || xv.n != 1 || uses[xv.reg as usize] != 1 || defs[xv.reg as usize] != 1 {
                    continue;
                }
                let Some(&i) = at.get(&xv.reg) else { continue };
                if dead[i] {
                    continue;
                }
                let inner = &f.blocks[b].insts[i];
                let Some(ki) = kind(inner.op) else { continue };
                if inner.mods != 0 || inner.uses.len() != 2 {
                    continue;
                }
                let (a, c) = (inner.uses[0], inner.uses[1]);
                // Nothing between writes EXEC or the inner operands (they are read at `outer` instead).
                let clean = f.blocks[b].insts[i + 1..k].iter().all(|s| {
                    s.op.flags() & (F_DEF_EXEC | F_BRANCH | F_END) == 0
                        && !s.defs.iter().any(|d| overlaps(d, &a) || overlaps(d, &c) || matches!(d, Opnd::P(Phys::ExecLo, _)))
                });
                if !clean {
                    continue;
                }
                let fused = if ki == ko {
                    Some((three(ko), [a, c, other]))
                } else {
                    // med3: min(max(x, lo), hi) or max(min(x, hi), lo), lo <= hi, both constants.
                    let med = match (ki, ko) {
                        (Kind::MaxI, Kind::MinI) | (Kind::MinI, Kind::MaxI) => Some(Opc::VMed3I32),
                        (Kind::MaxU, Kind::MinU) | (Kind::MinU, Kind::MaxU) => Some(Opc::Gfx9(Gfx9Op::VMed3U32)),
                        _ => None,
                    };
                    let (inner_k, var) = match (a, c) {
                        (Opnd::Imm(v), y) if !matches!(y, Opnd::Imm(_)) => (Some(v), y),
                        (y, Opnd::Imm(v)) if !matches!(y, Opnd::Imm(_)) => (Some(v), y),
                        _ => (None, a),
                    };
                    match (med, inner_k, other) {
                        (Some(m), Some(ik), Opnd::Imm(ok)) => {
                            let (lo, hi) = if matches!(ki, Kind::MaxI | Kind::MaxU) { (ik, ok) } else { (ok, ik) };
                            ordered(ki, lo, hi).then_some((m, [var, Opnd::Imm(lo), Opnd::Imm(hi)]))
                        }
                        _ => None,
                    }
                };
                let Some((op, srcs)) = fused else { continue };
                let new = MInst { op, defs: outer.defs, uses: Uses::from(srcs), imm: 0, mods: 0, mem: outer.mem, scope: outer.scope };
                f.blocks[b].insts[k] = new;
                dead[i] = true;
                folded += 1;
                break;
            }
        }
        if dead.iter().any(|&d| d) {
            let mut k = 0;
            f.blocks[b].insts.retain(|_| {
                k += 1;
                !dead[k - 1]
            });
        }
    }
    folded
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v(reg: u32) -> Opnd {
        Opnd::R(VRef { reg, off: 0, n: 1 })
    }

    fn func(insts: Vec<MInst>, nregs: u32) -> MFunc {
        let vregs = vec![VRegInfo { class: Class::V, size: 1, align: 1, fixed: None }; nregs as usize];
        MFunc { blocks: vec![MBlock { insts, loop_depth: 0 }], vregs, target: crate::target::Target::Gfx950, ..Default::default() }
    }

    #[test]
    fn a_max_chain_pairs_into_max3() {
        // m1 = max(a, b); m2 = max(m1, c); m3 = max(m2, d); m4 = max(m3, e): (a b c) d e -> max3(max3(a,b,c), d, e)
        let insts = vec![
            MInst::new(Opc::VMaxI32, [v(10)], [v(0), v(1)]),
            MInst::new(Opc::VMaxI32, [v(11)], [v(10), v(2)]),
            MInst::new(Opc::VMaxI32, [v(12)], [v(11), v(3)]),
            MInst::new(Opc::VMaxI32, [v(13)], [v(12), v(4)]),
            MInst::new(Opc::VMovB32, [v(14)], [v(13)]),
        ];
        let mut f = func(insts, 16);
        let n = crate::rir::with_levers(&["minmax3"], || run(&mut f));
        assert_eq!(n, 2);
        let ops: Vec<Opc> = f.blocks[0].insts.iter().map(|i| i.op).collect();
        assert_eq!(ops, [Opc::Gfx9(Gfx9Op::VMax3I32), Opc::Gfx9(Gfx9Op::VMax3I32), Opc::VMovB32]);
        assert_eq!(&f.blocks[0].insts[0].uses[..], &[v(0), v(1), v(2)]);
        assert_eq!(&f.blocks[0].insts[1].uses[..], &[v(11), v(3), v(4)]);
    }

    #[test]
    fn clamp_becomes_med3_only_when_ordered() {
        let mk = |lo: u32, hi: u32| {
            vec![
                MInst::new(Opc::VMaxI32, [v(10)], [Opnd::Imm(lo), v(0)]),
                MInst::new(Opc::VMinI32, [v(11)], [Opnd::Imm(hi), v(10)]),
                MInst::new(Opc::VMovB32, [v(12)], [v(11)]),
            ]
        };
        let mut f = func(mk(1, 0xfe), 16);
        assert_eq!(crate::rir::with_levers(&["minmax3"], || run(&mut f)), 1);
        assert_eq!(f.blocks[0].insts[0].op, Opc::VMed3I32);
        assert_eq!(&f.blocks[0].insts[0].uses[..], &[v(0), Opnd::Imm(1), Opnd::Imm(0xfe)]);
        // lo > hi: min(max(x, 5), 3) is the constant 3, not med3(x, 5, 3)'s median: kept
        let mut g = func(mk(5, 3), 16);
        assert_eq!(crate::rir::with_levers(&["minmax3"], || run(&mut g)), 0);
        // signedness: max_i32(x, -1) then min_i32(., 7) is ordered as signed
        let mut h = func(mk(u32::MAX, 7), 16);
        assert_eq!(crate::rir::with_levers(&["minmax3"], || run(&mut h)), 1);
    }

    #[test]
    fn off_by_default_and_blocked_by_reuse_or_redefinition() {
        let insts = vec![
            MInst::new(Opc::VMaxI32, [v(10)], [v(0), v(1)]),
            MInst::new(Opc::VMaxI32, [v(11)], [v(10), v(2)]),
            MInst::new(Opc::VMovB32, [v(12)], [v(11)]),
        ];
        let mut f = func(insts.clone(), 16);
        assert_eq!(run(&mut f), 0);
        // the inner result read twice
        let mut two = insts.clone();
        two.push(MInst::new(Opc::VMovB32, [v(13)], [v(10)]));
        let mut f = func(two, 16);
        assert_eq!(crate::rir::with_levers(&["minmax3"], || run(&mut f)), 0);
        // an inner operand rewritten between the two
        let mut red = insts.clone();
        red.insert(1, MInst::new(Opc::VMovB32, [v(1)], [v(5)]));
        let mut f = func(red, 16);
        assert_eq!(crate::rir::with_levers(&["minmax3"], || run(&mut f)), 0);
        // mixed signedness does not fuse into max3
        let mixed = vec![
            MInst::new(Opc::VMaxU32, [v(10)], [v(0), v(1)]),
            MInst::new(Opc::VMaxI32, [v(11)], [v(10), v(2)]),
            MInst::new(Opc::VMovB32, [v(12)], [v(11)]),
        ];
        let mut f = func(mixed, 16);
        assert_eq!(crate::rir::with_levers(&["minmax3"], || run(&mut f)), 0);
    }
}
