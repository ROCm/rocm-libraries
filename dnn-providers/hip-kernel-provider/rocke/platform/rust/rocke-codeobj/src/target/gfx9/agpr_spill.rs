// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

//! Accumulators in AGPRs beside spilled values on gfx950, as COMGR's
//! allocations have them. When a function spills (`rir::spill_best`), the
//! spiller keeps its MFMA accumulators in VGPRs and puts other webs in AGPR
//! slots, moved to a VGPR around every access: in a K loop with its global
//! loads in flight, a load's data goes out after the load and back before its
//! LDS write, every iteration. With the accumulator webs in AGPRs first
//! (`agpr::bank`: the MFMAs read C and write D there), those VGPRs hold the
//! data, and only what still does not fit spills, into the AGPR slots the
//! webs leave.
//!
//! The webs take the top of the free AGPRs while the spiller fills the slots
//! from a0 below them, and then move down to follow the slots. Two placements
//! are weighed: the webs `agpr::bank` moves, and those with accesses in their
//! MFMAs' loops too (moved there every iteration, as COMGR's attention loops
//! do). The function takes the allocation `sched::cost` gives the fewest
//! weighted slots per wave if that beats spilling alone: the cost of the
//! moves and spill accesses each adds, by instruction class and loop depth.
//! `ROCKE_DIRECT_DEBUG=nobankspill` spills alone; `nobankfits` keeps the
//! banked spills but not the banked schedules that fit without spilling.

use super::agpr::{is_mfma, place, place_with, rewrite};
use super::asm::AGPR0;
use super::sched;
use crate::error::CompileError;
use crate::rir::mir::*;
use crate::rir::regalloc::{Allocation, Limits, Pressure, SpillConfig};
use crate::target::Target;

/// The AGPRs of a lane.
const AGPRS: u32 = 256;

/// `f` (before spilling) with its accumulator webs in AGPRs and the rest
/// spilled by `allocate` into the AGPR slots of `cfg` the webs leave (then
/// LDS and scratch), within `room` unified registers: the webs `agpr::place`
/// moves, and also, when more move that way, those with accesses in their
/// MFMAs' loops (`agpr::place_with`). None of either when no web moves or the
/// webs need more AGPRs than `cfg` has.
pub fn banked_spill(
    f: &MFunc,
    cfg: SpillConfig,
    room: u32,
    allocate: impl Fn(&mut MFunc, SpillConfig) -> Result<Allocation, CompileError>,
) -> Vec<(MFunc, Allocation)> {
    if f.target != Target::Gfx950 || cfg.agpr_slots == 0 || crate::rir::debug_flag("nobankspill") {
        return Vec::new();
    }
    if !f.blocks.iter().flat_map(|b| &b.insts).any(|i| is_mfma(i.op)) {
        return Vec::new();
    }
    let strict = place(f);
    let in_loops = place_with(f, true).filter(|(_, n)| strict.as_ref().is_none_or(|(_, m)| n > m));
    [strict, in_loops].into_iter().flatten().filter_map(|placed| spill_placed(f, placed, cfg, room, &allocate)).collect()
}

/// The first of `candidates` (window, function, pressure; the widest window
/// first) that fits `hw` with its accumulator webs in AGPRs, moves in loops
/// and all, and no spill, within `room` unified registers
/// (`bank_wide::allocate_banked_in_loops`), where that moves other webs than
/// the placement `spill_best` tried first.
pub fn banked_fits(candidates: &[(usize, MFunc, Pressure)], hw: Limits, room: u32) -> Vec<(usize, MFunc, Allocation)> {
    let Some((_, first, _)) = candidates.first() else { return Vec::new() };
    let off = crate::rir::debug_flag("nobankspill") || crate::rir::debug_flag("nobankfits");
    if first.target != Target::Gfx950 || first.agpr_limit == Some(0) || off {
        return Vec::new();
    }
    if !first.blocks.iter().flat_map(|b| &b.insts).any(|i| is_mfma(i.op)) {
        return Vec::new();
    }
    candidates
        .iter()
        .filter_map(|(window, f, _)| {
            let placed = place_with(f, true)?;
            // `spill_best` already tried the webs `agpr::place` moves (`agpr::allocate_banked`).
            if place(f).is_some_and(|strict| strict == placed) {
                return None;
            }
            super::bank_wide::allocate_placed(f, placed, hw).map(|(g, a)| (*window, g, a))
        })
        .filter(|(_, _, a)| a.unified_vgprs() <= room)
        .take(1)
        .collect()
}

/// `banked_spill` with the webs placed.
fn spill_placed(
    f: &MFunc,
    (mut base_of, banked): (Vec<u32>, u32),
    cfg: SpillConfig,
    room: u32,
    allocate: &impl Fn(&mut MFunc, SpillConfig) -> Result<Allocation, CompileError>,
) -> Option<(MFunc, Allocation)> {
    // The webs' tuples start at multiples of 4 from where they go (`place`),
    // so they go and move by multiples of 4.
    let extent = banked.next_multiple_of(4);
    if extent > cfg.agpr_slots {
        return None;
    }
    let top = AGPRS - extent;
    for b in base_of.iter_mut().filter(|b| **b != u32::MAX) {
        *b += top;
    }
    let mut g = f.clone();
    rewrite(&mut g, &base_of);
    let mut a = allocate(&mut g, SpillConfig { agpr_slots: (cfg.agpr_slots - extent) & !3, ..cfg }).ok()?;
    let slots = a.spill.agpr_dwords.next_multiple_of(4);
    relocate(&mut g, AGPR0 + top as u16, (top - slots) as u16);
    a.spill.agpr_dwords = slots + banked;
    (a.unified_vgprs() <= room).then_some((g, a))
}

/// Moves every register from `from` on down by `by`.
fn relocate(f: &mut MFunc, from: u16, by: u16) {
    for inst in f.blocks.iter_mut().flat_map(|b| b.insts.iter_mut()) {
        for o in inst.defs.iter_mut().chain(inst.uses.iter_mut()) {
            if let Opnd::P(Phys::V(r), n) = *o {
                if r >= from {
                    *o = Opnd::P(Phys::V(r - by), n);
                }
            }
        }
    }
}

/// Of the `banked` candidates (window, function, allocation), the one that
/// issues in the fewest weighted slots per wave if that is fewer than
/// `spilled`'s, each at its occupancy with `lds` bytes of LDS per workgroup
/// of `wg` lanes; the first of equals.
pub fn cheapest(
    banked: Vec<(usize, MFunc, Allocation)>,
    spilled: (&MFunc, &Allocation),
    lds: u32,
    wg: u32,
) -> Option<(usize, MFunc, Allocation)> {
    if banked.is_empty() {
        return None;
    }
    let t = spilled.0.target.info();
    let waves = |a: &Allocation| t.occupancy(a.unified_vgprs().max(1), lds + a.spill.lds_bytes, wg);
    let bar = sched::cost(spilled.0).cycles(waves(spilled.1));
    banked
        .into_iter()
        .map(|b| (sched::cost(&b.1).cycles(waves(&b.2)), b))
        .filter(|(cycles, _)| *cycles < bar)
        .min_by_key(|(cycles, _)| *cycles)
        .map(|(_, b)| b)
}

#[cfg(test)]
mod tests {
    use crate::rir;
    use crate::target::Target;

    /// A K loop of `accs` 32x32x16 f16 accumulators whose A and B come from LDS, which `loads` 16-byte global
    /// loads per trip refill for the next trip (the loads issued first, their LDS writes after the MFMAs). The
    /// LDS holds one workgroup per CU, so one wave per SIMD and every AGPR free.
    fn kernel(accs: usize, loads: usize) -> String {
        let mut s = String::from(
            "ckdsl.ir v1\nkernel @bank_spill {\n  attrs { max_workgroup_size = i:64 }\n  params {\n    %A : ptr<f16,global>\n    \
             %Y : ptr<f32,global>\n    %n : i32\n  }\n  region @entry {\n    \
             %tid : i32 = gpu.thread_id (  ) { axis = s:\"x\" }\n    \
             %lds : smem<f16, [65536]> = tile.smem_alloc (  )\n    \
             %c0 : i32 = arith.constant (  ) { ity = s:\"i32\", value = i:0 }\n    %c1 : i32 = arith.constant (  ) { ity = s:\"i32\", value = i:1 }\n    \
             %c8 : i32 = arith.constant (  ) { ity = s:\"i32\", value = i:8 }\n    \
             %z : vec<f32x16> = arith.constant_vec (  ) { elem = s:\"f32\", fill = f:0.0, vec = i:16 }\n    \
             %t8 : i32 = arith.mul ( %tid, %c8 )\n",
        );
        let results: Vec<String> = (0..accs).map(|a| format!("%r{a} : vec<f32x16>")).collect();
        let inits: Vec<&str> = (0..accs).map(|_| "%z").collect();
        let iter_args: Vec<String> = (0..accs).map(|a| format!("{{ name = s:\"%acc{a}\", type = s:\"vec<f32x16>\" }}")).collect();
        s += &format!(
            "    {} = scf.for ( %c0, %n, %c1, {} ) {{ iter_args = l:[ {} ], iv = s:\"%k\", iv_type = s:\"i32\", num_iter_args = i:{accs}, unroll = b:false }}\n      region @body {{\n",
            results.join(", "),
            inits.join(", "),
            iter_args.join(", ")
        );
        s += "        %o : i32 = arith.mul ( %k, %c8 )\n";
        for l in 0..loads {
            s += &format!(
                "        %s{l} : i32 = arith.constant (  ) {{ ity = s:\"i32\", value = i:{} }}\n        %p{l} : i32 = arith.add ( %o, %s{l} )\n        \
                 %q{l} : i32 = arith.add ( %p{l}, %t8 )\n        \
                 %g{l} : vec<f16x8> = memref.global_load_vN ( %A, %q{l} ) {{ align = i:16, elem_type = s:\"f16\", vec = i:8 }}\n",
                8192 * l
            );
        }
        let mut yields = Vec::new();
        for a in 0..accs {
            s += &format!(
                "        %ia{a} : i32 = arith.constant (  ) {{ ity = s:\"i32\", value = i:{} }}\n        %la{a} : i32 = arith.add ( %t8, %ia{a} )\n        \
                 %x{a} : vec<f16x8> = tile.smem_load_vN ( %lds, %la{a} ) {{ elem_type = s:\"f16\", rank = i:1, vec = i:8 }}\n        \
                 %d{a} : vec<f32x16> = tile.mma ( %x{a}, %x{a}, %acc{a} ) {{ op_id = s:\"mfma_f32_32x32x16_f16\" }}\n",
                512 * (a % 4)
            );
            yields.push(format!("%d{a}"));
        }
        s += "        tile.sync (  )\n";
        for l in 0..loads {
            s += &format!(
                "        %iw{l} : i32 = arith.constant (  ) {{ ity = s:\"i32\", value = i:{} }}\n        %lw{l} : i32 = arith.add ( %t8, %iw{l} )\n        \
                 tile.smem_store_vN ( %lds, %lw{l}, %g{l} ) {{ align = i:16, elem_type = s:\"f16\", rank = i:1, vec = i:8 }}\n",
                (512 * l) % 4096
            );
        }
        s += "        tile.sync (  )\n";
        s += &format!("        scf.yield ( {} ) {{ num = i:{accs} }}\n      }}\n", yields.join(", "));
        for a in 0..accs {
            for k in [0, 15] {
                s += &format!(
                    "    %e{a}_{k} : f32 = vector.extract ( %r{a} ) {{ index = i:{k} }}\n    \
                     %i{a}_{k} : i32 = arith.constant (  ) {{ ity = s:\"i32\", value = i:{} }}\n    \
                     %y{a}_{k} : i32 = arith.add ( %tid, %i{a}_{k} )\n    \
                     memref.global_store_typed ( %Y, %y{a}_{k}, %e{a}_{k} ) {{ align = i:4, elem_type = s:\"f32\" }}\n",
                    64 * (16 * a + k)
                );
            }
        }
        s + "    cf.return (  )\n  }\n}\n"
    }

    /// The register numbers a listing line names (`v7`, `v[8:11]`); the listing names AGPR k as v(256 + k).
    fn regs(line: &str) -> Vec<u32> {
        line.split(|c: char| c == ',' || c == ' ')
            .filter_map(|w| w.trim().strip_prefix('v').map(|r| r.trim_start_matches('[')))
            .filter_map(|r| r.split(':').next().and_then(|n| n.trim_end_matches(']').parse::<u32>().ok()))
            .collect()
    }

    /// The accesses in the loop between VGPRs and AGPRs, and the MFMAs whose D is an AGPR, of a listing.
    fn loop_moves(listing: &str) -> (usize, usize) {
        let lines: Vec<&str> = listing.lines().map(str::trim).collect();
        let end = lines.iter().position(|l| l.starts_with("s_cbranch_scc1")).unwrap_or(lines.len());
        let start = lines[..end].iter().rposition(|l| l.ends_with(':')).unwrap_or(0);
        let body = &lines[start..end];
        let moves = body.iter().filter(|l| l.starts_with("v_accvgpr_read") || l.starts_with("v_accvgpr_write")).count();
        let in_agprs = body.iter().filter(|l| l.starts_with("v_mfma") && regs(l).first().is_some_and(|&r| r >= 256)).count();
        (moves, in_agprs)
    }

    #[test]
    fn accumulators_go_to_agprs_beside_the_spilled_values() {
        // 64 accumulator dwords and 64 dwords of loads in flight, where 64 VGPRs are allowed: banked alone, the
        // loads still do not fit, so some of them spill into the AGPRs after the accumulators.
        let compiled = rir::compile_any_target_limited(&kernel(4, 16), Target::Gfx950, 64).unwrap_or_else(|e| panic!("{e}"));
        let s = compiled.stats.spill;
        let listing = compiled.listing();
        let (moves, in_agprs) = loop_moves(&listing);
        assert!(s.agpr_dwords > 64 && s.lds_dwords + s.scratch_dwords == 0, "{s:?}");
        // The MFMAs read and write their accumulators in AGPRs; the moves in the loop are the spilled values',
        // fewer than spilled accumulators would need around their MFMAs.
        assert_eq!(in_agprs, 4, "{listing}");
        assert!(moves > 0 && moves < 2 * 64, "{listing}");
        // The webs follow the spill slots: the AGPRs in use end at the count reported.
        let mfma_end = listing
            .lines()
            .map(str::trim)
            .filter(|l| l.starts_with("v_mfma"))
            .filter_map(|l| l.split_whitespace().nth(1).and_then(|d| d.trim_end_matches(',').split(':').nth(1)))
            .filter_map(|n| n.trim_end_matches(']').parse::<u32>().ok())
            .max()
            .unwrap_or(0);
        assert_eq!(mfma_end + 1, 256 + s.agpr_dwords, "{listing}");
    }
}
