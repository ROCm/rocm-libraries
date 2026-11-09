// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

//! Trees of bitwise logic of at most three distinct values become one
//! `v_bitop3_b32` on gfx950 (lever `bitop3`, `crate::rir::lever`), as LLVM's
//! gfx950 `BITOP3` combine: the MoE router's monotone keys
//! (`(x ^ m) & -512 | t`) take one instruction instead of three.
//!
//! Runs on virtual registers before legalization, in one block. A root is any
//! `v_and` / `v_or` / `v_xor` / `v_not` / `v_and_or` / `v_or3` /
//! `v_bitop3`; an operand defined by another such instruction of the block is
//! absorbed when the root is its only reader, the leaves stay at most three,
//! and neither its operands nor EXEC change between it and the root. The
//! truth table is the tree evaluated on the canonical source bytes (`0xf0`,
//! `0xcc`, `0xaa` for sources 0, 1, 2): bitwise the same result.

use crate::rir::mir::*;

use super::dpp::overlaps;
use super::gen_isa::Gfx9Op;

/// The truth bytes of sources 0, 1 and 2.
const SRC: [u8; 3] = [0xf0, 0xcc, 0xaa];

fn logic(inst: &MInst) -> bool {
    inst.mods == 0
        && matches!(
            (inst.op, inst.uses.len()),
            (Opc::VAndB32 | Opc::VOrB32 | Opc::VXorB32, 2)
                | (Opc::VNotB32, 1)
                | (Opc::VAndOrB32 | Opc::VOr3B32, 3)
                | (Opc::Gfx9(Gfx9Op::VBitop3B32), 3)
        )
        && matches!(inst.defs.first(), Some(Opnd::R(d)) if d.off == 0 && d.n == 1)
}

/// `inst` applied to its operands' truth bytes.
fn apply(inst: &MInst, x: &[u8]) -> u8 {
    match inst.op {
        Opc::VAndB32 => x[0] & x[1],
        Opc::VOrB32 => x[0] | x[1],
        Opc::VXorB32 => x[0] ^ x[1],
        Opc::VNotB32 => !x[0],
        Opc::VAndOrB32 => (x[0] & x[1]) | x[2],
        Opc::VOr3B32 => x[0] | x[1] | x[2],
        _ => {
            // v_bitop3: bit (s0 << 2 | s1 << 1 | s2) of the table, per bit position.
            let t = inst.imm as u8;
            let mut out = 0u8;
            for bit in 0..8 {
                let k = ((x[0] >> bit) & 1) << 2 | ((x[1] >> bit) & 1) << 1 | ((x[2] >> bit) & 1);
                out |= ((t >> k) & 1) << bit;
            }
            out
        }
    }
}

/// A tree under construction: the absorbed instructions (block indices) by
/// their result register, and the leaves.
struct Tree {
    inner: Vec<(u32, usize)>,
    leaves: Vec<Opnd>,
}

impl Tree {
    fn eval(&self, insts: &[MInst], at: usize) -> u8 {
        let inst = &insts[at];
        let x: Vec<u8> = inst.uses.iter().map(|o| self.value(insts, o)).collect();
        apply(inst, &x)
    }

    fn value(&self, insts: &[MInst], o: &Opnd) -> u8 {
        if let Opnd::R(v) = o {
            if let Some(&(_, at)) = self.inner.iter().find(|(r, _)| *r == v.reg) {
                return self.eval(insts, at);
            }
        }
        let k = self.leaves.iter().position(|l| l == o).expect("a leaf of the tree");
        SRC[k]
    }
}

/// Fuses the logic trees of `f`; returns how many instructions went.
pub fn run(f: &mut MFunc) -> usize {
    let mut uses = vec![0u32; f.vregs.len()];
    let mut defs = vec![0u32; f.vregs.len()];
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
    let mut gone = 0;
    for b in 0..f.blocks.len() {
        let insts = &f.blocks[b].insts;
        // The defining index of each single-def register in this block.
        let mut def_at: std::collections::HashMap<u32, usize> = std::collections::HashMap::new();
        for (i, inst) in insts.iter().enumerate() {
            for o in &inst.defs {
                if let Opnd::R(v) = o {
                    def_at.insert(v.reg, i);
                }
            }
        }
        let mut dead = vec![false; insts.len()];
        let mut rewrites: Vec<(usize, MInst)> = Vec::new();
        for k in (0..insts.len()).rev() {
            if dead[k] || !logic(&insts[k]) || !f.lever(&insts[k], "bitop3") {
                continue;
            }
            let root = &insts[k];
            let mut tree = Tree { inner: Vec::new(), leaves: Vec::new() };
            for o in &root.uses {
                if !tree.leaves.contains(o) {
                    tree.leaves.push(*o);
                }
            }
            if tree.leaves.len() > 3 {
                continue;
            }
            loop {
                let mut grown = false;
                for li in 0..tree.leaves.len() {
                    let Opnd::R(v) = tree.leaves[li] else { continue };
                    let reg = v.reg as usize;
                    if v.off != 0 || v.n != 1 || uses[reg] != 1 || defs[reg] != 1 {
                        continue;
                    }
                    let Some(&j) = def_at.get(&v.reg) else { continue };
                    if j >= k || dead[j] || !logic(&insts[j]) {
                        continue;
                    }
                    let mut leaves: Vec<Opnd> = tree.leaves.iter().copied().filter(|l| *l != tree.leaves[li]).collect();
                    for o in &insts[j].uses {
                        if !leaves.contains(o) {
                            leaves.push(*o);
                        }
                    }
                    if leaves.len() > 3 {
                        continue;
                    }
                    // Its operands and EXEC must hold from it to the root.
                    let stable = insts[j + 1..k].iter().all(|s| {
                        s.op.flags() & (F_DEF_EXEC | F_BRANCH | F_END) == 0
                            && !s.defs.iter().any(|d| insts[j].uses.iter().any(|u| overlaps(d, u)))
                    });
                    if !stable {
                        continue;
                    }
                    tree.inner.push((v.reg, j));
                    tree.leaves = leaves;
                    grown = true;
                    break;
                }
                if !grown {
                    break;
                }
            }
            if tree.inner.is_empty() {
                continue;
            }
            // Leaves absorbed later replace earlier ones: drop leaves no longer read by any node.
            let read: Vec<Opnd> = std::iter::once(k)
                .chain(tree.inner.iter().map(|&(_, j)| j))
                .flat_map(|i| insts[i].uses.iter().copied())
                .collect();
            tree.leaves.retain(|l| read.contains(l) && !matches!(l, Opnd::R(v) if tree.inner.iter().any(|(r, _)| *r == v.reg)));
            let table = tree.eval(insts, k);
            let mut srcs = tree.leaves.clone();
            while srcs.len() < 3 {
                srcs.push(Opnd::Imm(0));
            }
            let mut fused = MInst::new(Opc::Gfx9(Gfx9Op::VBitop3B32), [root.defs[0]], [srcs[0], srcs[1], srcs[2]]).with_imm(table as i64);
            fused.mem = root.mem;
            for &(_, j) in &tree.inner {
                dead[j] = true;
            }
            gone += tree.inner.len();
            fused.scope = root.scope;
            rewrites.push((k, fused));
        }
        let block = &mut f.blocks[b];
        for (k, inst) in rewrites {
            block.insts[k] = inst;
        }
        let mut i = 0;
        block.insts.retain(|_| {
            i += 1;
            !dead[i - 1]
        });
    }
    gone
}

#[cfg(test)]
mod tests {
    use super::*;

    fn func(n: usize, body: impl FnOnce(&[VRef]) -> Vec<MInst>) -> MFunc {
        let mut f = MFunc { target: crate::target::Target::Gfx950, ..MFunc::default() };
        let regs: Vec<VRef> = (0..n)
            .map(|i| {
                f.vregs.push(VRegInfo { class: Class::V, size: 1, align: 1, fixed: None });
                VRef { reg: i as u32, off: 0, n: 1 }
            })
            .collect();
        f.blocks = vec![MBlock { insts: body(&regs), loop_depth: 0 }];
        f
    }

    /// Evaluates a block's instructions on concrete words (registers start at `init`).
    fn exec(insts: &[MInst], init: &[u32], n: usize) -> Vec<u32> {
        let mut regs = vec![0u32; n];
        regs[..init.len()].copy_from_slice(init);
        let val = |regs: &Vec<u32>, o: &Opnd| match o {
            Opnd::R(v) => regs[v.reg as usize],
            Opnd::Imm(c) => *c,
            _ => unreachable!(),
        };
        for inst in insts {
            let x: Vec<u32> = inst.uses.iter().map(|o| val(&regs, o)).collect();
            let r = match inst.op {
                Opc::VAndB32 => x[0] & x[1],
                Opc::VOrB32 => x[0] | x[1],
                Opc::VXorB32 => x[0] ^ x[1],
                Opc::VNotB32 => !x[0],
                Opc::VAndOrB32 => (x[0] & x[1]) | x[2],
                Opc::VOr3B32 => x[0] | x[1] | x[2],
                Opc::Gfx9(Gfx9Op::VBitop3B32) => {
                    let t = inst.imm as u32;
                    (0..32).fold(0u32, |acc, bit| {
                        let k = ((x[0] >> bit) & 1) << 2 | ((x[1] >> bit) & 1) << 1 | ((x[2] >> bit) & 1);
                        acc | ((t >> k) & 1) << bit
                    })
                }
                other => panic!("{other:?}"),
            };
            if let Opnd::R(d) = inst.defs[0] {
                regs[d.reg as usize] = r;
            }
        }
        regs
    }

    #[test]
    fn router_key_is_one_bitop3() {
        let r = Opnd::R;
        // key = ((x ^ m) & -512) | t  (x, m, t in v0..v2; v3, v4 inner; v5 the key)
        let body = |v: &[VRef]| {
            vec![
                MInst::new(Opc::VXorB32, [r(v[3])], [r(v[0]), r(v[1])]),
                MInst::new(Opc::VAndOrB32, [r(v[5])], [r(v[3]), Opnd::Imm(0xffff_fe00), r(v[2])]),
            ]
        };
        let mut f = func(6, body);
        let before = f.blocks[0].insts.len();
        // four leaves (x, m, the mask and t): nothing fuses
        assert_eq!(crate::rir::with_levers(&["bitop3"], || run(&mut f)), 0);
        assert_eq!(f.blocks[0].insts.len(), before);
        // ((x ^ m) & y) | t over three registers fuses, bitwise equal
        let mut f = func(6, |v| {
            vec![
                MInst::new(Opc::VXorB32, [r(v[3])], [r(v[0]), r(v[1])]),
                MInst::new(Opc::VNotB32, [r(v[4])], [r(v[3])]),
                MInst::new(Opc::VAndOrB32, [r(v[5])], [r(v[4]), r(v[0]), r(v[2])]),
            ]
        });
        let want = exec(&f.blocks[0].insts, &[0x1234_5678, 0x8000_00ff, 0x0000_01a5], 6)[5];
        assert_eq!(crate::rir::with_levers(&["bitop3"], || run(&mut f)), 2);
        assert_eq!(f.blocks[0].insts.len(), 1);
        assert_eq!(f.blocks[0].insts[0].op, Opc::Gfx9(Gfx9Op::VBitop3B32));
        for init in [[0x1234_5678u32, 0x8000_00ff, 0x0000_01a5], [0xffff_ffff, 0, 0x5555_5555], [0, 0xdead_beef, 0x0f0f_0f0f]] {
            let mut g = func(6, |v| {
                vec![
                    MInst::new(Opc::VXorB32, [r(v[3])], [r(v[0]), r(v[1])]),
                    MInst::new(Opc::VNotB32, [r(v[4])], [r(v[3])]),
                    MInst::new(Opc::VAndOrB32, [r(v[5])], [r(v[4]), r(v[0]), r(v[2])]),
                ]
            });
            let w = exec(&g.blocks[0].insts, &init, 6)[5];
            run(&mut g);
            assert_eq!(exec(&g.blocks[0].insts, &init, 6)[5], w, "{init:x?}");
        }
        assert_eq!(exec(&f.blocks[0].insts, &[0x1234_5678, 0x8000_00ff, 0x0000_01a5], 6)[5], want);
    }

    #[test]
    fn shared_or_rewritten_operands_stay() {
        let r = Opnd::R;
        // v3 read twice: not absorbed
        let mut f = func(6, |v| {
            vec![
                MInst::new(Opc::VXorB32, [r(v[3])], [r(v[0]), r(v[1])]),
                MInst::new(Opc::VAndB32, [r(v[4])], [r(v[3]), r(v[2])]),
                MInst::new(Opc::VOrB32, [r(v[5])], [r(v[3]), r(v[4])]),
            ]
        });
        let n = crate::rir::with_levers(&["bitop3"], || run(&mut f));
        assert!(f.blocks[0].insts.iter().any(|i| i.defs[0] == r(VRef { reg: 3, off: 0, n: 1 })), "{n}");
        // v0 written between the xor and its reader: not absorbed
        let mut f = func(6, |v| {
            vec![
                MInst::new(Opc::VXorB32, [r(v[3])], [r(v[0]), r(v[1])]),
                MInst::new(Opc::VMovB32, [r(v[0])], [Opnd::Imm(7)]),
                MInst::new(Opc::VAndB32, [r(v[5])], [r(v[3]), r(v[2])]),
            ]
        });
        assert_eq!(crate::rir::with_levers(&["bitop3"], || run(&mut f)), 0);
    }
}
