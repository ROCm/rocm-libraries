// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

//! Integer min / max from a compare and a select on gfx950 (lever `minmax`,
//! `crate::rir::lever`), as LLVM's `select (icmp a, b), a, b` combine: a
//! `v_cmp_{gt,ge,lt,le}_{i32,u32}` whose lane mask only one
//! `v_cndmask_b32` of the same two values reads becomes `v_max` / `v_min` of
//! the matching signedness, and the compare goes (the MoE router's top-k key
//! compares). Runs on virtual registers, before allocation, in one block; the
//! two values must not change between the compare and the select (they are
//! read at the max either way) and EXEC must not change.
//!
//! The selected value is the same in every lane (ties pick equal values), so
//! the result is bitwise the select's.

use crate::rir::mir::*;

use super::dpp::overlaps;

/// (max, min) of a compare code: signed 0x41 lt, 0x43 le, 0x44 gt, 0x46 ge;
/// unsigned 0x49, 0x4b, 0x4c, 0x4e; and whether `a op b` means `a` is the
/// larger (gt / ge).
fn kind(code: u8) -> Option<(Opc, Opc, bool)> {
    match code {
        0x44 | 0x46 => Some((Opc::VMaxI32, Opc::VMinI32, true)),
        0x41 | 0x43 => Some((Opc::VMaxI32, Opc::VMinI32, false)),
        0x4c | 0x4e => Some((Opc::VMaxU32, Opc::VMinU32, true)),
        0x49 | 0x4b => Some((Opc::VMaxU32, Opc::VMinU32, false)),
        _ => None,
    }
}

/// Rewrites every eligible compare and its selects in `f`; returns how many
/// compares went. Every reader of the mask must be a `v_cndmask_b32` of the
/// compared values (the top-k compare-exchange selects both the larger and
/// the smaller), all in the compare's block with nothing writing them, EXEC or
/// the mask in between.
pub fn run(f: &mut MFunc) -> usize {
    let mut uses = vec![0u32; f.vregs.len()];
    for inst in f.blocks.iter().flat_map(|b| &b.insts) {
        for o in &inst.uses {
            if let Opnd::R(v) = o {
                uses[v.reg as usize] += 1;
            }
        }
    }
    let mut done = 0;
    for b in 0..f.blocks.len() {
        let mut i = 0;
        while i < f.blocks[b].insts.len() {
            let cmp = f.blocks[b].insts[i].clone();
            let (Opc::VCmp(code), [Opnd::R(m)], [x, y]) = (cmp.op, &cmp.defs[..], &cmp.uses[..]) else {
                i += 1;
                continue;
            };
            let (x, y, m) = (*x, *y, *m);
            let Some((max, min, a_larger)) = kind(code) else {
                i += 1;
                continue;
            };
            let want = uses[m.reg as usize] as usize;
            if cmp.mods != 0 || want == 0 || want > 4 || !f.lever(&cmp, "minmax") {
                i += 1;
                continue;
            }
            let mut found: Vec<(usize, Opc)> = Vec::new();
            let mut ok = true;
            for k in i + 1..f.blocks[b].insts.len() {
                let s = &f.blocks[b].insts[k];
                if s.uses.iter().any(|u| overlaps(u, &Opnd::R(m))) {
                    let picked = match (s.op, &s.uses[..]) {
                        (Opc::VCndmaskB32, [fv, tv, mm]) if *mm == Opnd::R(m) && s.mods == 0 => {
                            if (*tv, *fv) == (x, y) {
                                Some(a_larger)
                            } else if (*tv, *fv) == (y, x) {
                                Some(!a_larger)
                            } else {
                                None
                            }
                        }
                        _ => None,
                    };
                    let Some(larger) = picked else {
                        ok = false;
                        break;
                    };
                    found.push((k, if larger { max } else { min }));
                    if found.len() == want {
                        break;
                    }
                }
                let writes = |o: &Opnd| s.defs.iter().any(|d| overlaps(d, o));
                if s.op.flags() & (F_DEF_EXEC | F_BRANCH | F_END) != 0 || writes(&x) || writes(&y) || writes(&Opnd::R(m)) {
                    ok = false;
                    break;
                }
            }
            if !ok || found.len() != want {
                i += 1;
                continue;
            }
            for &(k, op) in &found {
                let sel = &f.blocks[b].insts[k];
                f.blocks[b].insts[k] = MInst { op, defs: sel.defs, uses: Uses::from([x, y]), imm: 0, mods: 0, mem: sel.mem, scope: sel.scope };
            }
            f.blocks[b].insts.remove(i);
            done += 1;
        }
    }
    done
}

#[cfg(test)]
mod tests {
    use super::*;

    fn func(n: usize, body: impl FnOnce(&[VRef]) -> Vec<MInst>) -> MFunc {
        let mut f = MFunc { target: crate::target::Target::Gfx950, ..MFunc::default() };
        let regs: Vec<VRef> = (0..n)
            .map(|i| {
                let class = if i == n - 1 { Class::S } else { Class::V };
                f.vregs.push(VRegInfo { class, size: if class == Class::S { 2 } else { 1 }, align: 1, fixed: None });
                VRef { reg: i as u32, off: 0, n: if class == Class::S { 2 } else { 1 } }
            })
            .collect();
        f.blocks = vec![MBlock { insts: body(&regs), loop_depth: 0 }];
        f
    }

    #[test]
    fn compare_and_select_become_max_or_min() {
        let r = Opnd::R;
        for (code, t_is_a, want) in [(0x44u8, true, Opc::VMaxI32), (0x44, false, Opc::VMinI32), (0x49, true, Opc::VMinU32), (0x4e, false, Opc::VMinU32)] {
            let mut f = func(4, |v| {
                let (a, b, d, m) = (v[0], v[1], v[2], v[3]);
                let sel = if t_is_a { [r(b), r(a), r(m)] } else { [r(a), r(b), r(m)] };
                vec![MInst::new(Opc::VCmp(code), [r(m)], [r(a), r(b)]), MInst::new(Opc::VCndmaskB32, [r(d)], sel)]
            });
            assert_eq!(crate::rir::with_levers(&["minmax"], || run(&mut f)), 1, "{code:#x}");
            assert_eq!(f.blocks[0].insts.len(), 1);
            assert_eq!(f.blocks[0].insts[0].op, want, "{code:#x} {t_is_a}");
        }
        // The compare-exchange: both selects of one mask become max and min.
        let mut f = func(5, |v| {
            let (a, b, hi, lo, m) = (v[0], v[1], v[2], v[3], v[4]);
            vec![
                MInst::new(Opc::VCmp(0x49), [r(m)], [r(a), r(b)]),
                MInst::new(Opc::VCndmaskB32, [r(hi)], [r(a), r(b), r(m)]),
                MInst::new(Opc::VCndmaskB32, [r(lo)], [r(b), r(a), r(m)]),
            ]
        });
        assert_eq!(crate::rir::with_levers(&["minmax"], || run(&mut f)), 1);
        assert_eq!(f.blocks[0].insts.iter().map(|i| i.op).collect::<Vec<_>>(), [Opc::VMaxU32, Opc::VMinU32]);
        // A select of other values keeps the compare.
        let mut f = func(5, |v| {
            let (a, b, d, e, m) = (v[0], v[1], v[2], v[3], v[4]);
            vec![
                MInst::new(Opc::VCmp(0x44), [r(m)], [r(a), r(b)]),
                MInst::new(Opc::VCndmaskB32, [r(d)], [r(b), r(e), r(m)]),
            ]
        });
        assert_eq!(crate::rir::with_levers(&["minmax"], || run(&mut f)), 0);
    }
}
