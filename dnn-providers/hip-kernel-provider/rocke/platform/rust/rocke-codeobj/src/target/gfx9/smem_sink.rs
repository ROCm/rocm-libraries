// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

//! Scalar ALU users of a scalar load sink below the loads that follow it
//! (lever `smemsink`).
//!
//! A table lookup such as the block table of a paged KV cache is one scalar
//! load per entry, its address a few `s_add` / `s_cmp` / `s_cselect_b32` before
//! it and a `s_cmp` + `s_cselect_b32` (clamp the value to 0 out of range)
//! after it. SCC is one register, so those compare + select pairs keep their
//! order and every pair after a load waits for it: the loads of consecutive
//! entries issue one memory round trip apart. `post::sink_load_uses` does
//! not move a user that sets SCC.
//!
//! A pair is no problem to move as a whole: here the unit is a *chunk*, an
//! instruction that sets SCC together with every instruction up to the last
//! one that reads it (one instruction when nothing does), so SCC is never
//! live across a chunk boundary. A chunk of scalar ALU instructions that
//! reads a loaded register moves down past the following chunks until one
//! of them depends on it (reads what it writes, writes what it reads or
//! writes) or is a fence, a branch or a barrier: the loads between issue back
//! to back and the wait for the values lands at their first real use. Only
//! scalar registers and constants are involved, so the effect on register
//! pressure is the loaded values living longer.
//!
//! `ROCKE_DIRECT_DEBUG=nosmemsink` skips the pass.

use crate::rir::mir::*;

use super::dpp::overlaps;

/// The farthest (instructions) a chunk moves.
const REACH: usize = 1024;
/// Chunks moved per block, at most.
const MOVES: usize = 128;

fn scalar_alu(inst: &MInst, vregs: &[VRegInfo]) -> bool {
    let flags = inst.op.flags();
    if !matches!(inst.op.fmt(), Fmt::Sop1 | Fmt::Sop2 | Fmt::Sopc | Fmt::Sopk)
        || flags & (F_FENCE | F_BRANCH | F_END | F_DEF_EXEC | F_USE_EXEC | F_USE_VCC | F_SMEM | F_LDS | F_VMEM_LOAD | F_VMEM_STORE) != 0
    {
        return false;
    }
    inst.defs.iter().chain(inst.uses.iter()).all(|o| match o {
        Opnd::R(v) => vregs[v.reg as usize].class == Class::S,
        Opnd::Imm(_) => true,
        Opnd::P(..) => false,
    })
}

/// End (exclusive) of the chunk that starts at `i`.
fn chunk_end(insts: &[MInst], i: usize) -> usize {
    if insts[i].op.flags() & F_DEF_SCC == 0 {
        return i + 1;
    }
    let mut end = i + 1;
    for (j, inst) in insts.iter().enumerate().skip(i + 1) {
        let flags = inst.op.flags();
        if flags & F_USE_SCC != 0 {
            end = j + 1;
        } else if flags & F_DEF_SCC != 0 {
            break;
        }
    }
    end
}

/// Whether anything of `moved` conflicts with `other`: a register that one
/// writes and the other reads or writes.
fn conflicts(moved: &[MInst], other: &[MInst]) -> bool {
    moved.iter().any(|x| {
        other.iter().any(|y| {
            x.defs.iter().any(|d| y.uses.iter().chain(y.defs.iter()).any(|o| overlaps(d, o)))
                || x.uses.iter().any(|u| y.defs.iter().any(|d| overlaps(u, d)))
        })
    })
}

/// Whether a chunk of `other` may be passed at all.
fn passable(other: &[MInst]) -> bool {
    other.iter().all(|y| y.op.flags() & (F_FENCE | F_BRANCH | F_END) == 0 && !matches!(y.op, Opc::SBarrier | Opc::SSendmsg | Opc::SWaitcnt))
}

/// Sinks the users of scalar loads of every block; returns the chunks moved.
pub fn run(f: &mut MFunc) -> usize {
    if crate::rir::debug_flag("nosmemsink") {
        return 0;
    }
    let mut moved = 0;
    for bi in 0..f.blocks.len() {
        let mut insts = std::mem::take(&mut f.blocks[bi].insts);
        // Registers a scalar load of this block wrote, so far.
        let mut loaded = vec![false; f.vregs.len()];
        let mut sunk = vec![false; insts.len()];
        let mut i = 0;
        let mut budget = MOVES;
        let mut n_loads = 0;
        while i < insts.len() && budget > 0 {
            if insts[i].op.flags() & F_SMEM != 0 {
                n_loads += 1;
                for d in insts[i].defs.iter().filter_map(Opnd::vref) {
                    loaded[d.reg as usize] = true;
                }
                i += 1;
                continue;
            }
            let end = chunk_end(&insts, i);
            let reads_loaded = insts[i..end].iter().any(|x| x.uses.iter().filter_map(Opnd::vref).any(|u| loaded[u.reg as usize]));
            // A reader of an SCC the block did not set (live in) stays where it is.
            let live_in = insts[i].op.flags() & (F_USE_SCC | F_DEF_SCC) == F_USE_SCC;
            if n_loads > 0 && reads_loaded && !sunk[i] && !live_in && insts[i..end].iter().all(|x| scalar_alu(x, &f.vregs)) {
                // Past whole chunks, as far as nothing stops it.
                let mut stop = end;
                while stop < insts.len() && stop - end < REACH {
                    let next = chunk_end(&insts, stop);
                    let (head, tail) = insts.split_at(stop);
                    if !passable(&tail[..next - stop]) || conflicts(&head[i..end], &tail[..next - stop]) {
                        break;
                    }
                    stop = next;
                }
                // Only past something worth it: a later load.
                if stop > end && insts[end..stop].iter().any(|x| x.op.flags() & F_SMEM != 0) {
                    insts[i..stop].rotate_left(end - i);
                    // The chunk now ends at `stop`; it is marked so that it is not moved twice.
                    let len = end - i;
                    sunk[i..stop].rotate_left(len);
                    sunk[stop - len..stop].iter_mut().for_each(|s| *s = true);
                    moved += 1;
                    budget -= 1;
                    // The instructions that took its place are looked at next.
                    continue;
                }
            }
            i = end;
        }
        f.blocks[bi].insts = insts;
    }
    moved
}

#[cfg(test)]
mod tests {
    use super::*;

    fn r(reg: u32) -> Opnd {
        Opnd::R(VRef { reg, off: 0, n: 1 })
    }

    fn func(insts: Vec<MInst>) -> MFunc {
        let vregs = vec![VRegInfo { class: Class::S, size: 1, align: 1, fixed: None }; 64];
        MFunc { blocks: vec![MBlock { insts, loop_depth: 1 }], vregs, ..Default::default() }
    }

    /// `table[i < n ? i : 0]`, then `i < n ? v : 0`: the registers are `i` (10 + k), `n` (1),
    /// the clamped index (20 + k), the loaded value (30 + k) and the select (40 + k).
    fn entry(k: u32) -> Vec<MInst> {
        vec![
            MInst::new(Opc::SCmpLtI32, [], [r(10 + k), r(1)]),
            MInst::new(Opc::SCselectB32, [r(20 + k)], [r(10 + k), Opnd::Imm(0)]),
            MInst::new(Opc::SLoadB32, [r(30 + k)], [r(20 + k), Opnd::Imm(0)]),
            MInst::new(Opc::SCmpLtI32, [], [r(10 + k), r(1)]),
            MInst::new(Opc::SCselectB32, [r(40 + k)], [r(30 + k), Opnd::Imm(0)]),
        ]
    }

    fn ops(f: &MFunc) -> Vec<(Opc, Opnd)> {
        f.blocks[0].insts.iter().map(|i| (i.op, i.defs.first().copied().unwrap_or(Opnd::Imm(0)))).collect()
    }

    #[test]
    fn the_selects_of_loaded_values_move_below_the_next_load() {
        let mut f = func((0..3).flat_map(entry).collect());
        assert_eq!(run(&mut f), 2);
        let loads: Vec<usize> = ops(&f).iter().enumerate().filter(|(_, (op, _))| *op == Opc::SLoadB32).map(|(i, _)| i).collect();
        let first_use = ops(&f).iter().position(|(op, d)| *op == Opc::SCselectB32 && *d == r(40)).unwrap();
        assert!(loads.iter().all(|&l| l < first_use), "{:?}", ops(&f));
    }

    #[test]
    fn a_value_the_next_entry_needs_stays_in_order() {
        // entry 1's index is entry 0's select
        let mut insts = entry(0);
        insts.extend(entry(1));
        insts[5] = MInst::new(Opc::SCmpLtI32, [], [r(40), r(1)]);
        insts[6] = MInst::new(Opc::SCselectB32, [r(21)], [r(40), Opnd::Imm(0)]);
        let before = ops(&func(insts.clone()));
        let mut f = func(insts);
        assert_eq!(run(&mut f), 0);
        assert_eq!(ops(&f), before);
    }

    #[test]
    fn a_barrier_stops_the_move() {
        let mut insts = entry(0);
        insts.push(MInst::new(Opc::SBarrier, [], []));
        insts.extend(entry(1));
        let mut f = func(insts);
        assert_eq!(run(&mut f), 0);
    }

    #[test]
    fn scc_stays_closed_around_a_moved_pair() {
        let mut f = func((0..2).flat_map(entry).collect());
        run(&mut f);
        let insts = &f.blocks[0].insts;
        for (i, inst) in insts.iter().enumerate() {
            if inst.op == Opc::SCselectB32 {
                assert_eq!(insts[i - 1].op, Opc::SCmpLtI32, "{:?}", ops(&f));
            }
        }
    }
}
