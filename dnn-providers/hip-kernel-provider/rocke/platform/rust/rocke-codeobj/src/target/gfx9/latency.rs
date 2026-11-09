// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

//! Wider schedules for latency-bound functions without MFMAs on gfx950, their
//! SGPRs past the hardware in VGPR lanes. The scheduling search keeps the SGPRs
//! the hardware has, and when nothing fits them, spilling (`rir::spill_best`)
//! takes the schedule that needs the fewest registers: load sinking alone. A
//! depthwise convolution loop that computes its row masks and offsets in SGPRs
//! then issues its loads one at a time, each waited for before the next, where
//! COMGR issues them all at once. Here, when the schedule taken leaves most of
//! its latency exposed, each wider window is allocated with its SGPRs past the
//! hardware kept in VGPR lanes, and `post::schedule_cost` weighs it against the
//! schedule taken; a candidate must keep the schedule's waves and save a
//! fifth, and the cheapest before allocation that still does after it wins.
//! Fewer waves are not weighed: measured on such kernels, this model and
//! `sched::cost` each preferred a schedule at fewer waves that ran slower, and
//! the choices between attempts after this one (alias facts, unrolling,
//! merged LDS loads) take the occupancy first.
//! Functions with MFMAs are the occupancy trade's (`trade`).
//! `ROCKE_DIRECT_DEBUG=nolatency` skips it.

use super::agpr::is_mfma;
use crate::error::CompileError;
use crate::rir::mir::MFunc;
use crate::rir::post::{self, ScheduleCost};
use crate::rir::regalloc::{self, Allocation, Limits, Pressure, SpillConfig};
use crate::target::Target;

/// The windows weighed, widest first (`rir`'s 512 window, the search's, and
/// one between its two widest).
const WINDOWS: [usize; 5] = [512, 64, 48, 32, 16];
/// The windows `spill_candidates` adds to those spilling chooses among.
const EXTRA: [usize; 2] = [512, 48];
/// Candidates allocated, cheapest before allocation first.
const TRIES: usize = 2;

/// A schedule weighed: its window, the function and its pressure.
pub type Candidate = (usize, MFunc, Pressure);

/// Whether a schedule costing `cost` leaves most of its latency exposed at
/// `waves` waves per SIMD: its cycles there are at least five fourths of its
/// issue cycles, so the SIMD would idle a fifth of the time or more.
pub fn exposed(cost: &ScheduleCost, waves: u32) -> bool {
    cost.cycles(waves) * 4 >= cost.issue() * 5
}

/// Whether `f` is weighed here: gfx950, no MFMA, and not switched off.
fn weighed(f: &MFunc) -> bool {
    f.target == Target::Gfx950
        && !crate::rir::debug_flag("nolatency")
        && !f.blocks.iter().flat_map(|b| &b.insts).any(|i| is_mfma(i.op))
}

/// No VGPR dword spilled (to LDS, scratch or AGPRs): SGPRs in VGPR lanes at most.
fn lanes_only(a: &Allocation) -> bool {
    a.spill.lds_dwords + a.spill.scratch_dwords + a.spill.agpr_dwords == 0
}

/// The waves an allocation gives with `lds` bytes of LDS and `wg` work-items.
fn waves_of(a: &Allocation, f: &MFunc, lds: u32, wg: u32) -> u32 {
    f.target.info().occupancy(a.unified_vgprs().max(1), lds + a.spill.lds_bytes, wg)
}

/// SGPRs the allocation taken may leave free for `lower` to search: a wider
/// window that fits the VGPRs is short of SGPRs only where the schedule taken
/// already holds most of them (and any other wider window that fits the
/// VGPRs was taken by the search before).
const SGPR_SLACK: u32 = 16;

/// `chosen`, the allocation and window `func` (scheduled and allocated without
/// spilling) has, or a wider window past the SGPRs at the same waves that
/// `post::schedule_cost` gives at least a fifth fewer cycles there
/// (`Weigh::first_faster`); `func` gets the one taken. `schedule(window, cap)` is the function
/// after load sinking scheduled with `window` and its pressure, or None when
/// the pressure passes `cap`; `spill` allocates a function with its SGPRs past
/// the hardware in VGPR lanes.
pub fn lower(
    func: &mut MFunc,
    chosen: (Allocation, usize),
    schedule: impl Fn(usize, Limits) -> Option<(MFunc, Pressure)>,
    spill: impl Fn(&mut MFunc, SpillConfig) -> Result<Allocation, CompileError>,
    lds_size: u32,
    wg: u32,
) -> (Allocation, usize) {
    let hw = Limits::for_func(func);
    if !weighed(func) || chosen.0.num_sgprs + SGPR_SLACK < hw.sgprs {
        return chosen;
    }
    let t = func.target.info();
    let have = waves_of(&chosen.0, func, lds_size, wg);
    let cost = post::schedule_cost(func);
    if !exposed(&cost, have) {
        return chosen;
    }
    let cap = Limits { vgprs: t.vgpr_budget(have, lds_size, wg).min(hw.vgprs), sgprs: u32::MAX / 4 };
    let candidates: Vec<Candidate> = WINDOWS
        .into_iter()
        .filter(|&w| w > chosen.1)
        .filter_map(|w| schedule(w, cap).map(|(f, p)| (w, f, p)))
        .filter(|(_, _, p)| p.sgprs > hw.sgprs)
        .collect();
    if candidates.is_empty() {
        return chosen;
    }
    let cfg = spill_config(func, lds_size, wg);
    let weigh = Weigh { bar: cost.cycles(have) * 4 / 5, have, hw, lds: lds_size, wg };
    match weigh.first_faster(candidates, &|f| spill(f, cfg)) {
        Some((window, g, a)) => {
            *func = g;
            (a, window)
        }
        None => chosen,
    }
}

/// What `spill_best` spills with on gfx950 (`agpr_slots` of `spill`).
fn spill_config(f: &MFunc, lds_size: u32, wg: u32) -> SpillConfig {
    let t = f.target.info();
    let hw = Limits::for_func(f);
    SpillConfig {
        lds_size,
        wg_size: wg.max(1),
        lds_slots: t.lds_spill_slots(lds_size, wg, hw.vgprs),
        agpr_slots: if f.agpr_limit == Some(0) { 0 } else { super::spill::agpr_slots(t, hw.vgprs, lds_size, wg) },
    }
}

/// The schedules `spilled_faster` weighs when spilling (`rir::spill_best`)
/// chooses among `candidates` (window, function, pressure; `needs` their
/// assignments' registers): a copy of each and the `EXTRA` windows of load
/// sinking alone (window 0), for a function `weighed` here whose assignments
/// all fit the VGPRs; none otherwise.
pub fn spill_candidates(candidates: &[Candidate], needs: &[Pressure], hw: Limits) -> Vec<Candidate> {
    let Some((_, first, _)) = candidates.first() else { return Vec::new() };
    if !weighed(first) || needs.iter().any(|n| n.vgprs > hw.vgprs) {
        return Vec::new();
    }
    let mut out: Vec<Candidate> = candidates.to_vec();
    if let Some((_, sunk, _)) = candidates.iter().find(|(w, _, _)| *w == 0) {
        for window in EXTRA {
            let mut wide = sunk.clone();
            post::schedule_window(&mut wide, window);
            if let Ok(p) = regalloc::pressure(&wide) {
                out.push((window, wide, p));
            }
        }
    }
    out
}

/// A faster spill than `chosen` (the function `spill_best` spilled, its
/// allocation and window) among `candidates` (`spill_candidates`): when
/// `chosen` kept only SGPRs in VGPR lanes and leaves most of its latency
/// exposed, another window at the same waves, spilled by `spill` with `cfg`,
/// that costs a fifth fewer cycles (`Weigh::first_faster`).
pub fn spilled_faster(
    candidates: Vec<Candidate>,
    chosen: (&MFunc, &Allocation, usize),
    cfg: SpillConfig,
    lds_size: u32,
    wg: u32,
    spill: impl Fn(&mut MFunc, SpillConfig) -> Result<Allocation, CompileError>,
) -> Option<(usize, MFunc, Allocation)> {
    let (f, a, window) = chosen;
    if candidates.is_empty() || !lanes_only(a) {
        return None;
    }
    let have = waves_of(a, f, lds_size, wg);
    let cost = post::schedule_cost(f);
    if !exposed(&cost, have) {
        return None;
    }
    let others = candidates.into_iter().filter(|(w, _, _)| *w != window).collect();
    let weigh = Weigh { bar: cost.cycles(have) * 4 / 5, have, hw: Limits::for_func(f), lds: lds_size, wg };
    weigh.first_faster(others, &|g| spill(g, cfg))
}

/// What a candidate is weighed against: the cycles it must stay under, the
/// waves it must keep, the hardware's registers, the LDS and workgroup size.
struct Weigh {
    bar: u64,
    have: u32,
    hw: Limits,
    lds: u32,
    wg: u32,
}

impl Weigh {
    /// The first candidate, cheapest before allocation first, that keeps
    /// `have` and costs under `bar` in `post::schedule_cost` cycles at the
    /// waves its allocation gives. Each is allocated by `spill`, its SGPRs past
    /// the hardware in VGPR lanes; one that spills VGPR dwords is not taken.
    /// Before allocation each is priced at the waves its VGPRs and SGPR lanes
    /// allow, and only the `TRIES` cheapest that keep `have` there, if under
    /// `bar`, are allocated.
    fn first_faster(
        &self,
        candidates: Vec<Candidate>,
        spill: &dyn Fn(&mut MFunc) -> Result<Allocation, CompileError>,
    ) -> Option<(usize, MFunc, Allocation)> {
        let Some((_, first, _)) = candidates.first() else { return None };
        let t = first.target.info();
        let (hw, lds, wg) = (self.hw, self.lds, self.wg);
        let kept = |waves: u32| waves >= self.have;
        let lanes = |p: &Pressure| p.sgprs.saturating_sub(hw.sgprs).div_ceil(t.wave);
        let mut priced: Vec<(u64, Candidate)> = candidates
            .into_iter()
            .filter(|(_, _, p)| p.vgprs + lanes(p) <= hw.vgprs)
            .map(|(w, f, p)| (t.occupancy((p.vgprs + lanes(&p)).max(1), lds, wg), (w, f, p)))
            .filter(|&(waves, _)| kept(waves))
            .map(|(waves, (w, f, p))| (post::schedule_cost(&f).cycles(waves), (w, f, p)))
            .filter(|(c, _)| *c < self.bar)
            .collect();
        priced.sort_by_key(|(c, (w, _, _))| (*c, std::cmp::Reverse(*w)));
        priced.into_iter().take(TRIES).find_map(|(_, (window, mut g, _))| {
            let a = spill(&mut g).ok().filter(lanes_only)?;
            let waves = waves_of(&a, &g, lds, wg);
            (kept(waves) && post::schedule_cost(&g).cycles(waves) < self.bar).then_some((window, g, a))
        })
    }
}

#[cfg(test)]
mod tests {
    use crate::rir;
    use crate::target::Target;

    /// A rolled loop (runtime trip count) of `loads` loads per iteration, each at a uniform offset
    /// `m * c + k` that its product reads again after the load, so that issuing the loads together keeps
    /// every offset live in an SGPR.
    fn kernel(loads: usize) -> String {
        let mut s = String::from(
            "ckdsl.ir v1\nkernel @lanes {\n  attrs { max_workgroup_size = i:256 }\n  params {\n    \
             %X : ptr<i32,global> { align = i:4, noalias = b:true, readonly = b:true }\n    \
             %Y : ptr<i32,global> { align = i:4, noalias = b:true }\n    %n : i32\n    %m : i32\n  }\n  region @entry {\n    \
             %tid : i32 = gpu.thread_id (  ) { axis = s:\"x\" }\n    \
             %c0 : i32 = arith.constant (  ) { ity = s:\"i32\", value = i:0 }\n    \
             %c1 : i32 = arith.constant (  ) { ity = s:\"i32\", value = i:1 }\n    \
             %r : i32 = scf.for ( %c0, %n, %c1, %c0 ) { iter_args = l:[ { name = s:\"%acc\", type = s:\"i32\" } ], iv = s:\"%k\", iv_type = s:\"i32\", num_iter_args = i:1, unroll = b:false }\n      region @body {\n",
        );
        let mut sum = String::from("%acc");
        for i in 0..loads {
            s += &format!(
                "        %c{i}x : i32 = arith.constant (  ) {{ ity = s:\"i32\", value = i:{} }}\n        \
                 %u{i} : i32 = arith.mul ( %m, %c{i}x )\n        %o{i} : i32 = arith.add ( %u{i}, %k )\n        \
                 %a{i} : i32 = arith.add ( %tid, %o{i} )\n        \
                 %v{i} : i32 = memref.global_load_typed ( %X, %a{i} ) {{ align = i:4, elem_type = s:\"i32\" }}\n        \
                 %p{i} : i32 = arith.mul ( %v{i}, %o{i} )\n        %s{i} : i32 = arith.add ( {sum}, %p{i} )\n",
                7 * i + 3
            );
            sum = format!("%s{i}");
        }
        s + &format!(
            "        scf.yield ( {sum} ) {{ num = i:1 }}\n      }}\n    \
             memref.global_store_typed ( %Y, %tid, %r ) {{ align = i:4, elem_type = s:\"i32\" }}\n    cf.return (  )\n  }}\n}}\n"
        )
    }

    /// The most global loads in flight at once in a listing: issued and not yet drained by an in-order
    /// `vmcnt` wait.
    fn most_in_flight(listing: &str) -> usize {
        let (mut outstanding, mut most) = (0usize, 0usize);
        for line in listing.lines().map(str::trim) {
            if line.starts_with("global_load") {
                outstanding += 1;
                most = most.max(outstanding);
            } else if let Some(n) = line.strip_prefix("s_waitcnt").and_then(|w| w.split("vmcnt(").nth(1)) {
                let n: usize = n.split(')').next().and_then(|n| n.parse().ok()).expect("a vmcnt count");
                outstanding = outstanding.min(n);
            }
        }
        most
    }

    #[test]
    fn a_loop_short_of_sgprs_takes_a_wider_window_with_its_sgprs_in_vgpr_lanes() {
        // Its loads together keep 120 offsets live: past the SGPRs, so spilling chose between schedules.
        let wide = rir::compile_any_target(&kernel(120), Target::Gfx950).unwrap_or_else(|e| panic!("{e}"));
        let s = &wide.stats;
        assert!(s.window >= 32 && s.spill.sgpr_dwords > 0, "window {} lanes {:?}", s.window, s.spill);
        assert_eq!(s.spill.lds_dwords + s.spill.scratch_dwords + s.spill.agpr_dwords, 0, "{:?}", s.spill);
        assert!(most_in_flight(&wide.listing()) >= 6, "{}", wide.listing());
        // With the offsets within the SGPRs nothing is kept in VGPR lanes.
        let fits = rir::compile_any_target(&kernel(40), Target::Gfx950).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(fits.stats.spill.sgpr_dwords, 0, "{:?}", fits.stats.spill);
    }
}
