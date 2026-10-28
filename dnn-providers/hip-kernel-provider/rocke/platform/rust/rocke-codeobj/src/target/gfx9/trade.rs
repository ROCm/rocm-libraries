// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

//! The occupancy trade on gfx950 (G2 design section 8): the scheduling search
//! keeps the occupancy load sinking alone reaches and takes the widest window
//! that keeps it, which in a large MFMA tile can leave the loop's global loads
//! one at a time. Here, for functions with MFMAs, each wider window is weighed
//! at the lower occupancy it needs, with its accumulator webs in AGPRs when the
//! VGPRs cannot hold it (`agpr::allocate_banked`), against the schedule taken;
//! `sched::cost` decides, and a candidate must save at least a fifth.

use super::{agpr, sched};
use crate::error::CompileError;
use crate::rir::mir::MFunc;
use crate::rir::regalloc::{Allocation, Limits, Pressure};
use crate::target::Target;

/// The windows weighed, widest first (`rir`'s 512 window and the search's).
const WINDOWS: [usize; 4] = [512, 64, 32, 16];
/// AGPRs per lane: past the VGPR limit plus these no schedule fits, banked
/// or not, so the pressure walks stop there.
const AGPRS: u32 = 256;

/// `chosen`, the allocation and window `func` (scheduled and allocated) has,
/// or a lower-occupancy schedule that `sched::cost` gives at least a fifth
/// fewer slots per wave, the fewest of those; `func` gets the one taken.
/// `schedule(window, cap)` is the function after load sinking scheduled with
/// `window` and its pressure, or None when the pressure passes `cap`;
/// `allocate` assigns a function's registers within a budget.
pub fn trade(
    func: &mut MFunc,
    chosen: (Allocation, usize),
    schedule: impl Fn(usize, Limits) -> Option<(MFunc, Pressure)>,
    allocate: impl Fn(&mut MFunc, Limits) -> Result<Allocation, CompileError>,
    lds_size: u32,
    wg: u32,
) -> (Allocation, usize) {
    if func.target != Target::Gfx950 || !func.blocks.iter().flat_map(|b| &b.insts).any(|i| agpr::is_mfma(i.op)) {
        return chosen;
    }
    let t = func.target.info();
    let waves = |a: &Allocation| t.occupancy(a.unified_vgprs().max(1), lds_size + a.spill.lds_bytes, wg);
    let hw = Limits::for_func(func);
    // Banked AGPRs count in the unified registers, which must leave one workgroup room on a CU.
    let room = super::rules::workgroup_registers(t, wg);
    let agprs = func.agpr_limit != Some(0);
    let have = waves(&chosen.0);
    let cost = sched::cost(func);
    let bar = cost.cycles(have) * 4 / 5;
    // Every candidate is a reordering of these blocks (with AGPR moves added
    // when banked), so none issues or holds the pipe less than `floor`; and
    // none fits a workgroup at fewer waves than the fewest any fitting
    // register count gives. Past either, nothing below can pass.
    let fewest = (1..=room.min(hw.vgprs + AGPRS)).map(|v| t.occupancy(v, lds_size, wg)).min().unwrap_or(0);
    if cost.floor() >= bar || have <= fewest.max(1) {
        return chosen;
    }
    let cap = Limits { vgprs: hw.vgprs + AGPRS, sgprs: hw.sgprs };
    let mut best: Option<(u64, MFunc, Allocation, usize)> = None;
    for window in WINDOWS.into_iter().filter(|&w| w > chosen.1) {
        let Some((f, p)) = schedule(window, cap) else { continue };
        let candidate = if p.vgprs <= hw.vgprs {
            let mut g = f;
            allocate(&mut g, hw).ok().map(|a| (g, Allocation { pressure: p, ..a }))
        } else if agprs {
            agpr::allocate_banked(&f, hw)
        } else {
            None
        };
        let Some((g, a)) = candidate.filter(|(_, a)| !a.tight && a.unified_vgprs() <= room && waves(a) < have) else { continue };
        let cycles = sched::cost(&g).cycles(waves(&a));
        if cycles < bar && best.as_ref().is_none_or(|b| cycles < b.0) {
            best = Some((cycles, g, a, window));
        }
    }
    match best {
        Some((_, g, a, window)) => {
            *func = g;
            (a, window)
        }
        None => chosen,
    }
}

#[cfg(test)]
mod tests {
    use crate::rir;
    use crate::target::Target;

    /// A K loop of 14 32x32x16 f16 accumulators fed by `loads` vector loads per trip, each accumulator stored
    /// after it: 224 accumulator dwords, so the loads fit in flight beside them only with the accumulators in
    /// AGPRs.
    fn kernel(loads: usize) -> String {
        kernel_with(14, loads, false)
    }

    /// `kernel` with `accs` accumulators; with `interleave`, each load comes just before the first MFMA that
    /// reads it, so program order keeps at most two in flight.
    fn kernel_with(accs: usize, loads: usize, interleave: bool) -> String {
        let mut s = String::from(
            "ckdsl.ir v1\nkernel @trade {\n  attrs { max_workgroup_size = i:64 }\n  params {\n    %A : ptr<f16,global>\n    \
             %Y : ptr<f32,global>\n    %n : i32\n  }\n  region @entry {\n    %tid : i32 = gpu.thread_id (  ) { axis = s:\"x\" }\n    \
             %c0 : i32 = arith.constant (  ) { ity = s:\"i32\", value = i:0 }\n    %c1 : i32 = arith.constant (  ) { ity = s:\"i32\", value = i:1 }\n    \
             %c8 : i32 = arith.constant (  ) { ity = s:\"i32\", value = i:8 }\n    \
             %z : vec<f32x16> = arith.constant_vec (  ) { elem = s:\"f32\", fill = f:0.0, vec = i:16 }\n",
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
        let load = |l: usize| {
            format!(
                "        %s{l} : i32 = arith.constant (  ) {{ ity = s:\"i32\", value = i:{} }}\n        %p{l} : i32 = arith.add ( %o, %s{l} )\n        \
                 %q{l} : i32 = arith.add ( %p{l}, %tid )\n        \
                 %a{l} : vec<f16x8> = memref.global_load_vN ( %A, %q{l} ) {{ align = i:16, elem_type = s:\"f16\", vec = i:8 }}\n",
                4096 * l
            )
        };
        let mut issued = 0;
        if !interleave {
            (0..loads).for_each(|l| s += &load(l));
            issued = loads;
        }
        let mut yields = Vec::new();
        for a in 0..accs {
            while issued < loads && issued <= (a + 1).min(loads - 1) {
                s += &load(issued);
                issued += 1;
            }
            s += &format!(
                "        %d{a} : vec<f32x16> = tile.mma ( %a{}, %a{}, %acc{a} ) {{ op_id = s:\"mfma_f32_32x32x16_f16\" }}\n",
                a % loads,
                (a + 1) % loads
            );
            yields.push(format!("%d{a}"));
        }
        s += &format!("        scf.yield ( {} ) {{ num = i:{accs} }}\n      }}\n", yields.join(", "));
        for a in 0..accs {
            s += &format!(
                "    %e{a} : f32 = vector.extract ( %r{a} ) {{ index = i:5 }}\n    %i{a} : i32 = arith.constant (  ) {{ ity = s:\"i32\", value = i:{} }}\n    \
                 %y{a} : i32 = arith.add ( %tid, %i{a} )\n    memref.global_store_typed ( %Y, %y{a}, %e{a} ) {{ align = i:4, elem_type = s:\"f32\" }}\n",
                64 * a
            );
        }
        s + "    cf.return (  )\n  }\n}\n"
    }

    #[test]
    fn a_loop_keeps_its_loads_in_flight_with_its_accumulators_in_agprs() {
        let traded = rir::compile_any_target(&kernel(12), Target::Gfx950).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!((traded.stats.spill.agpr_dwords, traded.stats.waves), (224, 1), "{:?}", traded.stats.spill);
        let listing = traded.listing();
        let body: Vec<&str> = listing.lines().map(str::trim).collect();
        // The twelve loads of a trip issue before the first wait for one of them.
        let first_load = body.iter().position(|l| l.starts_with("global_load_dwordx4")).expect("a load");
        let first_wait = body[first_load..].iter().position(|l| l.starts_with("s_waitcnt")).unwrap_or_else(|| panic!("{listing}")) + first_load;
        assert!(body[first_load..first_wait].iter().filter(|l| l.starts_with("global_load_dwordx4")).count() >= 12, "{listing}");
        // Two loads a trip leave nothing to hide: the accumulators stay in VGPRs at two waves.
        let kept = rir::compile_any_target(&kernel(2), Target::Gfx950).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!((kept.stats.spill.agpr_dwords, kept.stats.waves), (0, 2), "{:?}", kept.stats.spill);
    }

    #[test]
    fn a_trade_leaves_one_workgroup_room_and_keeps_agpr_alloc() {
        let compile = |text: String| rir::compile_any_target(&text, Target::Gfx950).unwrap_or_else(|e| panic!("{e}")).stats;
        // Program order fits at two waves; the trade takes a wider window at one, its accumulators in AGPRs.
        let traded = compile(kernel_with(14, 12, true));
        assert_eq!((traded.spill.agpr_dwords, traded.waves), (224, 1), "{:?}", traded.spill);
        // A 512-lane workgroup puts two waves on each SIMD, so each wave has at most 256 unified registers.
        let wide = compile(kernel_with(14, 12, true).replacen("max_workgroup_size = i:64", "max_workgroup_size = i:512", 1));
        let unified = wide.vgprs.next_multiple_of(4) + wide.spill.agpr_dwords;
        assert!(unified <= crate::target::gfx9::rules::workgroup_registers(Target::Gfx950.info(), 512), "{unified}: {:?}", wide.spill);
        let none = compile(kernel_with(14, 12, true).replacen("attrs { ", "attrs { agpr_alloc = l:[ i:0, i:0 ], ", 1));
        assert_eq!(none.spill.agpr_dwords, 0, "{:?}", none.spill);
    }
}
