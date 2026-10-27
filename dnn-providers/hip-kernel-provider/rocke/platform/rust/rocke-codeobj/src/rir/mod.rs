// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

//! Direct rocKE IR (`ck.dsl.ir/v1`) to gfx1151 HSACO compiler.

pub mod alias;
pub mod analysis;
pub mod asm;
pub mod ast;
pub mod fx;
mod fma_mix;
#[rustfmt::skip]
pub mod gen_isa;
pub mod isa;
mod hashcons;
pub mod isel;
mod loadorder;
pub mod lds_swizzle;
pub mod mir;
mod mma_order;
pub mod opt;
pub mod parse;
mod parse_par;
pub mod peephole;
pub mod post;
pub mod print;
pub mod regalloc;
mod reload;
pub mod unroll;
mod scan;
mod scan_line;
pub mod session;
pub mod validate;
mod spill;
mod stream;
mod uniform_masks;
pub mod vopd;

use std::time::{Duration, Instant};

use crate::error::CompileError;
use crate::hsaco_writer::{write_kernel_object, KernelArgMeta, KernelObjectSpec};
pub use crate::target::Target;
use ast::{Scalar, Type};

/// Conservative-mode switches for bisecting miscompiles, e.g.
/// `ROCKE_DIRECT_DEBUG=waitall,nocoalesce,nolicm,noldsreuse,noldscse,nosimplify,nofuse,nosched,nodelay,novopd,noclause,nowiden,noshrink`
/// (and `nohashcons`, which parses repeated constants as written).
/// `nopeephole` skips the proven peephole rules, `nopeephole_<family>` one family of them.
/// `nounroll` leaves rolled the loops the IR does not mark `unroll = true` (`unroll::Auto`).
/// `noalias` keeps every pair of global accesses in order, as if no kernel argument were `noalias`.
/// `nohoist` leaves spill reloads right before their uses, where the spiller puts them (`reload`).
/// `nommaorder` keeps program order when it does not fit the VGPRs (`mma_order`).
/// `nomix` keeps shared f16 conversions (`fma_mix`); `nomasks` evaluates a row's guard at each use (`uniform_masks`).
pub(crate) fn debug_flag(name: &str) -> bool {
    std::env::var("ROCKE_DIRECT_DEBUG").is_ok_and(|v| v.split(',').any(|f| f.trim() == name))
}

#[cfg(test)]
mod thread_tests {
    use super::*;

    /// `values` loads all live at once (each a short chain of products,
    /// summed in reverse): far past the registers, so it spills, and large
    /// enough for the spilling to run on several threads.
    fn spilling_kernel(values: usize) -> String {
        let mut t = String::from(
            "ckdsl.ir v1\nkernel @spills {\n  attrs { max_workgroup_size = i:64 }\n  params {\n    %X : ptr<f32,global>\n    \
             %Y : ptr<f32,global>\n  }\n  region @entry {\n    %tid : i32 = gpu.thread_id (  ) { axis = s:\"x\" }\n",
        );
        for i in 0..values {
            t += &format!("    %c{i} : i32 = arith.constant (  ) {{ ity = s:\"i32\", value = i:{} }}\n", 64 * i);
            t += &format!("    %p{i} : i32 = arith.add ( %tid, %c{i} )\n");
            t += &format!("    %v{i} : f32 = memref.global_load_typed ( %X, %p{i} ) {{ align = i:4, elem_type = s:\"f32\" }}\n");
            t += &format!("    %a{i} : f32 = arith.fmul ( %v{i}, %v{i} )\n    %b{i} : f32 = arith.fmul ( %a{i}, %v{i} )\n");
            t += &format!("    %e{i} : f32 = arith.fmul ( %b{i}, %v{i} )\n");
        }
        let mut sum = format!("%e{}", values - 1);
        for i in (0..values - 1).rev() {
            t += &format!("    %s{i} : f32 = arith.fadd ( {sum}, %e{i} )\n");
            sum = format!("%s{i}");
        }
        t + &format!(
            "    memref.global_store_typed ( %Y, %tid, {sum} ) {{ align = i:4, elem_type = s:\"f32\" }}\n    cf.return (  )\n  }}\n}}\n"
        )
    }

    #[test]
    fn one_thread_and_four_give_the_same_bytes() {
        let text = spilling_kernel(700);
        let one = compile_parsed_by_on(&text, &parse::parse, 1, Target::Gfx1151).unwrap_or_else(|e| panic!("{e}"));
        assert!(spilled(&one.stats) > 0, "spills");
        assert!(one.stats.instructions >= regalloc::PARALLEL_MIN_INSTS, "{} instructions", one.stats.instructions);
        let four = compile_parsed_by_on(&text, &parse::parse, 4, Target::Gfx1151).unwrap_or_else(|e| panic!("{e}"));
        assert!(one.hsaco == four.hsaco, "the thread count changed the code object");
    }

    /// `spilling_kernel` with `noalias` arguments and a store ahead of the
    /// loads, which the facts let the loads pass.
    fn spilling_kernel_with_facts(values: usize) -> String {
        spilling_kernel(values)
            .replace("%X : ptr<f32,global>", "%X : ptr<f32,global> { align = i:4, noalias = b:true, readonly = b:true }")
            .replace(
                "%Y : ptr<f32,global>\n",
                "%Y : ptr<f32,global> { align = i:4, noalias = b:true }\n    %Z : ptr<i32,global> { align = i:4, noalias = b:true }\n",
            )
            .replacen("    %c0 :", "    memref.global_store_typed ( %Z, %tid, %tid ) { align = i:4, elem_type = s:\"i32\" }\n    %c0 :", 1)
    }

    #[test]
    fn a_resumed_spill_gives_the_code_of_the_spilling_compile() {
        for (text, facts) in [(spilling_kernel(300), false), (spilling_kernel_with_facts(300), true)] {
            let auto = Some(unroll::Auto::DEFAULT);
            let Err(Failure::Unfit(resume)) = compile_with(&text, false, false, &parse::parse, auto, &mut 0, 0, Target::Gfx1151) else {
                panic!("fits without spilling");
            };
            assert_eq!(alias::relaxes(&resume.selected.sel.func), facts);
            let resumed = resume.spill(0, 0).unwrap_or_else(|e| panic!("{e}"));
            let full = compile_with(&text, false, true, &parse::parse, auto, &mut 0, 0, Target::Gfx1151)
                .map_err(Failure::into_error)
                .unwrap_or_else(|e| panic!("{e}"));
            assert!(spilled(&full.stats) > 0, "spills");
            assert!(resumed.hsaco == full.hsaco, "the resumed spill changed the code object");
            let compiled = compile(&text).unwrap_or_else(|e| panic!("{e}"));
            assert!(compiled.hsaco == full.hsaco, "compile does not resume the spilling compile");
        }
    }
}

/// The schedules spilling chooses among: each scheduling window, load
/// sinking alone (0) and program order (1), with their pressures.
type Candidates = Vec<(usize, mir::MFunc, regalloc::Pressure)>;

/// Why `schedule_and_allocate` gives no allocation.
enum Unallocated {
    Error(CompileError),
    /// Nothing fits without spilling, which was not asked for: the error,
    /// and the candidates spilling would choose among from here on.
    Unfit(CompileError, Candidates),
}

impl From<CompileError> for Unallocated {
    fn from(error: CompileError) -> Self {
        Unallocated::Error(error)
    }
}

/// Schedules `func` and assigns its registers. The pressure of load
/// sinking alone sets the occupancy to keep; the widest scheduling window
/// whose pressure keeps it is allocated within the VGPR budget of that
/// occupancy, and taken if the assignment keeps the occupancy too. When
/// fragmentation costs a wave, the occupancy to keep falls to what load
/// sinking alone actually assigns, as the choice by trial allocation had
/// it. When load sinking alone does not fit, program order does if it can.
/// With `spill`, a function that does not fit the hardware spills, in the
/// schedule whose assignment needs the fewest registers, on up to `threads`
/// threads (0: as the machine allows). Returns the allocation and the window
/// (0 for load sinking alone, 1 for program order).
fn schedule_and_allocate(
    func: &mut mir::MFunc,
    lds_size: u32,
    wg: u32,
    spill: bool,
    threads: usize,
) -> Result<(regalloc::Allocation, usize), Unallocated> {
    let t = func.target.info();
    let waves = |vgprs: u32| t.occupancy(vgprs, lds_size, wg);
    let hw = regalloc::Limits::for_func(func);
    let mut sunk = func.clone();
    post::sink_load_uses(&mut sunk, 32);
    let base_pressure = regalloc::pressure(&sunk)?;
    let fits_hw = |p: &regalloc::Pressure| p.vgprs <= hw.vgprs && p.sgprs <= hw.sgprs;
    let mut target = waves(base_pressure.vgprs.min(hw.vgprs));
    let budget = |target: u32| regalloc::Limits { vgprs: t.vgpr_budget(target, lds_size, wg).min(hw.vgprs), sgprs: hw.sgprs };
    let windows: &[usize] = if debug_flag("nosched") { &[] } else { &[64, 32, 16] };
    let mut base: Option<(mir::MFunc, regalloc::Allocation)> = None;
    let mut scheduled: Vec<(usize, mir::MFunc, regalloc::Pressure)> = Vec::with_capacity(windows.len());
    let mut allocated: Vec<Option<(mir::MFunc, regalloc::Allocation)>> = Vec::with_capacity(windows.len());
    let base_alloc = |target: u32| -> Option<(mir::MFunc, regalloc::Allocation)> {
        let mut b = sunk.clone();
        let a = regalloc::allocate_within(&mut b, budget(target)).ok()?;
        Some((b, a))
    };
    let mut lowered = false;
    for &window in windows {
        let mut f = sunk.clone();
        post::schedule_window(&mut f, window);
        let p = regalloc::pressure(&f)?;
        let mut tried = None;
        if !lowered && fits_hw(&p) && waves(p.vgprs) >= target {
            let mut g = f.clone();
            // A window whose first fit overflows the hardware stays out even
            // when packing tuples first squeezes it in: measured slower than
            // the narrower window at the same occupancy (v2 d256 prefill).
            if let Some(a) = regalloc::allocate_within(&mut g, budget(target)).ok().filter(|a| !a.tight) {
                if waves(a.num_vgprs) >= target {
                    *func = g;
                    return Ok((regalloc::Allocation { pressure: p, ..a }, window));
                }
                // Fragmentation cost a wave: if load sinking alone loses it
                // too, the occupancy to keep is what that reaches.
                if base.is_none() && fits_hw(&base_pressure) {
                    base = base_alloc(target);
                }
                if let Some((_, ba)) = &base {
                    if waves(ba.num_vgprs) < target {
                        target = waves(ba.num_vgprs);
                        lowered = true;
                    }
                }
                tried = Some((g, a));
            }
        }
        scheduled.push((window, f, p));
        allocated.push(tried);
    }
    if base.is_none() && fits_hw(&base_pressure) {
        base = base_alloc(target);
        if let Some((_, ba)) = &base {
            if waves(ba.num_vgprs) < target {
                target = waves(ba.num_vgprs);
                lowered = true;
            }
        }
    }
    if lowered {
        for ((window, f, p), tried) in scheduled.iter().zip(allocated) {
            if !fits_hw(p) || waves(p.vgprs) < target {
                continue;
            }
            let attempt = tried.or_else(|| {
                let mut g = f.clone();
                regalloc::allocate_within(&mut g, budget(target)).ok().filter(|a| !a.tight).map(|a| (g, a))
            });
            if let Some((g, a)) = attempt {
                if waves(a.num_vgprs) >= target {
                    *func = g;
                    return Ok((regalloc::Allocation { pressure: *p, ..a }, *window));
                }
            }
        }
    }
    if let Some((b, ba)) = base.take_if(|(_, ba)| !ba.tight) {
        *func = b;
        return Ok((regalloc::Allocation { pressure: base_pressure, ..ba }, 0));
    }
    // Load sinking alone does not fit, or only by packing tuples first: as
    // before pressure was known, program order (no sinking) gets a try.
    let program = func.clone();
    let program_pressure = regalloc::pressure(&program)?;
    if fits_hw(&program_pressure) {
        let mut g = program.clone();
        if let Ok(a) = regalloc::allocate_within(&mut g, budget(waves(program_pressure.vgprs))) {
            if base.as_ref().is_none_or(|(_, ba)| waves(a.num_vgprs) >= waves(ba.num_vgprs)) {
                *func = g;
                return Ok((regalloc::Allocation { pressure: program_pressure, ..a }, 1));
            }
        }
    }
    if let Some((b, ba)) = base {
        *func = b;
        return Ok((regalloc::Allocation { pressure: base_pressure, ..ba }, 0));
    }
    let candidates: Candidates = scheduled.into_iter().chain([(0, sunk, base_pressure), (1, program, program_pressure)]).collect();
    let error = CompileError::new(
        "ROCKE_DIRECT_REGISTERS",
        format!(
            "out of registers: {} VGPR and {} SGPR dwords live at once (spilling not tried)",
            base_pressure.vgprs, base_pressure.sgprs
        ),
    );
    if !spill {
        return Err(Unallocated::Unfit(error, candidates));
    }
    Ok(spill_best(func, candidates, lds_size, wg, threads)?)
}

/// The rest of `schedule_and_allocate` with `spill` once nothing fits
/// without spilling: spills in the candidate whose assignment needs the
/// fewest registers past the hardware, as fragmentation varies by schedule
/// more than pressure does.
fn spill_best(
    func: &mut mir::MFunc,
    candidates: Candidates,
    lds_size: u32,
    wg: u32,
    threads: usize,
) -> Result<(regalloc::Allocation, usize), CompileError> {
    let t = candidates[0].1.target.info();
    let hw = regalloc::Limits::for_func(&candidates[0].1);
    let over = |p: &regalloc::Pressure| p.vgprs.saturating_sub(hw.vgprs) + p.sgprs.saturating_sub(hw.sgprs);
    let large = candidates[0].1.blocks.iter().map(|b| b.insts.len()).sum::<usize>() >= regalloc::PARALLEL_MIN_INSTS;
    let threads = match threads {
        _ if !large => 1,
        0 => compile_threads(),
        n => n.min(4),
    };
    let needs = par_map(&candidates, threads, |(_, f, _)| {
        regalloc::assignment_need(f).unwrap_or(regalloc::Pressure { vgprs: u32::MAX / 4, sgprs: u32::MAX / 4 })
    });
    let ((window, mut f, _), _) = candidates
        .into_iter()
        .zip(needs)
        .min_by_key(|((w, _, p), need)| (over(need), over(p), *w != 0))
        .expect("load sinking alone is a candidate");
    let cfg = regalloc::SpillConfig {
        lds_size,
        wg_size: wg.max(1),
        lds_slots: t.lds_spill_slots(lds_size, wg, hw.vgprs),
    };
    let a = regalloc::allocate_spilling_on(&mut f, hw, cfg, threads)?;
    *func = f;
    Ok((a, window))
}

/// Threads one compile may run at once: `ROCKE_DIRECT_THREADS` if set, else
/// the machine's parallelism; at most four.
fn compile_threads() -> usize {
    std::env::var_os("ROCKE_DIRECT_THREADS")
        .and_then(|v| v.to_str().and_then(|v| v.trim().parse::<usize>().ok()))
        .filter(|&n| n > 0)
        .unwrap_or_else(|| std::thread::available_parallelism().map_or(1, |n| n.get()))
        .min(4)
}

/// `items.iter().map(f).collect()`, on up to `threads` scoped threads (the
/// calling one included). Each result depends only on its item, so the
/// thread count changes nothing but the time; a panic re-raises here.
fn par_map<T: Sync, R: Send>(items: &[T], threads: usize, f: impl Fn(&T) -> R + Sync) -> Vec<R> {
    let threads = threads.min(items.len());
    if threads <= 1 {
        return items.iter().map(f).collect();
    }
    let next = std::sync::atomic::AtomicUsize::new(0);
    let work = || {
        let mut done = Vec::new();
        loop {
            let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if i >= items.len() {
                return done;
            }
            done.push((i, f(&items[i])));
        }
    };
    let mut slots: Vec<Option<R>> = items.iter().map(|_| None).collect();
    std::thread::scope(|s| {
        let helpers: Vec<_> = (1..threads).map(|_| s.spawn(work)).collect();
        let mut finished = vec![work()];
        for helper in helpers {
            finished.push(helper.join().unwrap_or_else(|panic| std::panic::resume_unwind(panic)));
        }
        for (i, r) in finished.into_iter().flatten() {
            slots[i] = Some(r);
        }
    });
    slots.into_iter().map(|r| r.expect("every item is mapped")).collect()
}

/// The scheduling window of `widen_schedule`: in practice unbounded.
const WIDE_WINDOW: usize = 512;

/// Replaces the schedule `schedule_and_allocate` chose for `unscheduled`
/// by one with `WIDE_WINDOW`, at the same occupancy or one wave less, when
/// `post::schedule_cost` gives the wide one at least a twentieth less time.
/// Only a schedule whose exposed latency exceeds that twentieth is tried.
fn widen_schedule(
    unscheduled: &mir::MFunc,
    func: &mut mir::MFunc,
    (alloc, window): (regalloc::Allocation, usize),
    lds_size: u32,
    wg: u32,
) -> Result<(regalloc::Allocation, usize), CompileError> {
    let t = func.target.info();
    let waves = |vgprs: u32| t.occupancy(vgprs, lds_size, wg);
    let chosen = waves(alloc.num_vgprs);
    let cost = post::schedule_cost(func);
    let bar = cost.cycles(chosen) * 19 / 20;
    if debug_flag("nowiden") || window == 1 || bar <= cost.issue() {
        return Ok((alloc, window));
    }
    let hw = regalloc::Limits::for_func(func);
    let mut f = unscheduled.clone();
    post::sink_load_uses(&mut f, 32);
    post::schedule_window(&mut f, WIDE_WINDOW);
    let p = regalloc::pressure(&f)?;
    if p.vgprs > hw.vgprs || p.sgprs > hw.sgprs {
        return Ok((alloc, window));
    }
    let wide = post::schedule_cost(&f);
    for target in [chosen, chosen.saturating_sub(1)].into_iter().filter(|&t| t >= 1 && waves(p.vgprs) >= t) {
        if wide.cycles(target) >= bar {
            continue;
        }
        let mut g = f.clone();
        let budget = regalloc::Limits { vgprs: t.vgpr_budget(target, lds_size, wg).min(hw.vgprs), sgprs: hw.sgprs };
        let Ok(a) = regalloc::allocate_within(&mut g, budget) else {
            continue;
        };
        let reached = waves(a.num_vgprs);
        if a.tight || reached < target {
            continue;
        }
        if wide.cycles(reached) < bar {
            *func = g;
            return Ok((regalloc::Allocation { pressure: p, ..a }, WIDE_WINDOW));
        }
        if reached >= chosen {
            break;
        }
    }
    Ok((alloc, window))
}

/// The passes after allocation: pseudo expansion, wait counts, VOPD pairing,
/// the WMMA hazard, clauses, ALU delay hints, the VGPR release, FMAC
/// shrinking. Returns the VOPD pair counts.
fn finish(func: &mut mir::MFunc, num_vgprs: u32) -> Result<(u32, u64), CompileError> {
    if func.target == Target::Gfx950 {
        crate::target::gfx9::finish(func, num_vgprs)?;
        return Ok((0, 0));
    }
    post::expand_pseudos(func)?;
    post::insert_waitcnt(func);
    if !debug_flag("novopd") {
        vopd::form_pairs(func);
    }
    let pairs = vopd::count_pairs(func);
    post::fix_wmma_hazards(func);
    if !debug_flag("noclause") {
        post::form_clauses(func);
    }
    if !debug_flag("nodelay") {
        post::insert_delay_alu(func);
    }
    post::add_vgpr_release(func, num_vgprs);
    if !debug_flag("noshrink") {
        vopd::shrink_fmac(func);
    }
    Ok(pairs)
}

/// `schedule_and_allocate`, then `widen_schedule` unless it spilled.
fn schedule_and_widen(
    func: &mut mir::MFunc,
    lds_size: u32,
    wg: u32,
    spill: bool,
    threads: usize,
) -> Result<(regalloc::Allocation, usize), Unallocated> {
    let unscheduled = func.clone();
    let chosen = schedule_and_allocate(func, lds_size, wg, spill, threads)?;
    if spill || chosen.0.spill.lds_dwords + chosen.0.spill.scratch_dwords > 0 {
        return Ok(chosen);
    }
    Ok(widen_schedule(&unscheduled, func, chosen, lds_size, wg)?)
}

/// `schedule_and_widen` with the alias facts, when they relax an order, and
/// without them. The facts' schedule is taken when it spills less, or as
/// much and runs more waves, or as many and issues no more instructions
/// (after `finish`, weighted by 8 to the power of their loop depth) while
/// `post::schedule_cost` gives it at least a twentieth fewer cycles at that
/// occupancy, or no more there and at least a twentieth fewer for one wave
/// alone. Smaller modeled gains measured slower as often as faster
/// (depthwise convolutions; the bar is `widen_schedule`'s), and so did more
/// instructions with many waves resident (`fmha_appendkv` with rotary
/// embedding). The facts' schedule is not allocated when the one without
/// them spills (more loads in flight only cost registers there), nor unless,
/// before allocation, it gives one wave alone that twentieth in the widest
/// window up to the one taken without them whose pressure, plus the
/// registers that allocation lost to fragmentation without them, keeps that
/// occupancy (load sinking alone if none): more waves only hide more of the
/// latency it saves. What fits only with the facts fails as before.
fn schedule_with_facts(
    func: &mut mir::MFunc,
    lds_size: u32,
    wg: u32,
    spill: bool,
    threads: usize,
) -> Result<(regalloc::Allocation, usize), Unallocated> {
    if !alias::relaxes(func) {
        return schedule_and_widen(func, lds_size, wg, spill, threads);
    }
    let mut ordered = alias::without_facts(func);
    let strict = schedule_and_widen(&mut ordered, lds_size, wg, spill, threads)?;
    Ok(prefer_facts(func, ordered, strict, lds_size, wg, spill, threads))
}

/// The rest of `schedule_with_facts` once `ordered`, `func` without the
/// facts, has the allocation `strict`; `func` gets the schedule taken.
fn prefer_facts(
    func: &mut mir::MFunc,
    ordered: mir::MFunc,
    strict: (regalloc::Allocation, usize),
    lds_size: u32,
    wg: u32,
    spill: bool,
    threads: usize,
) -> (regalloc::Allocation, usize) {
    if strict.0.spill.lds_dwords + strict.0.spill.scratch_dwords > 0 {
        *func = ordered;
        return strict;
    }
    let faster = |a: u64, b: u64| a < b && a * 20 <= b * 19;
    let t = func.target.info();
    let waves = t.occupancy(strict.0.num_vgprs.max(1), lds_size, wg);
    let fragmented = strict.0.num_vgprs.saturating_sub(strict.0.pressure.vgprs);
    let hw = regalloc::Limits::for_func(func);
    let mut sunk = func.clone();
    post::sink_load_uses(&mut sunk, 32);
    let mut probe = None;
    for window in [WIDE_WINDOW, 64, 32, 16].into_iter().filter(|&w| w <= strict.1) {
        let mut f = sunk.clone();
        post::schedule_window(&mut f, window);
        let Ok(p) = regalloc::pressure(&f) else { continue };
        if p.vgprs <= hw.vgprs && p.sgprs <= hw.sgprs && t.occupancy(p.vgprs + fragmented, lds_size, wg) >= waves {
            probe = Some(f);
            break;
        }
    }
    let probe = probe.unwrap_or(sunk);
    if !faster(post::schedule_cost(&probe).cycles(1), post::schedule_cost(&ordered).cycles(1)) {
        *func = ordered;
        return strict;
    }
    let Ok(relaxed) = schedule_and_widen(func, lds_size, wg, spill, threads) else {
        *func = ordered;
        return strict;
    };
    // (spilled dwords, waves per SIMD, cycles at those waves, cycles of one wave).
    let rank = |f: &mir::MFunc, a: &regalloc::Allocation| {
        let waves = t.occupancy(a.num_vgprs.max(1), lds_size + a.spill.lds_bytes, wg);
        let cost = post::schedule_cost(f);
        (a.spill.lds_dwords + a.spill.scratch_dwords, waves, cost.cycles(waves), cost.cycles(1))
    };
    let (r, s) = (rank(func, &relaxed.0), rank(&ordered, &strict.0));
    let issued = |f: &mir::MFunc, a: &regalloc::Allocation| {
        let mut g = f.clone();
        finish(&mut g, a.num_vgprs).ok()?;
        Some(g.blocks.iter().map(|b| 8u64.pow(b.loop_depth.min(4)) * b.insts.len() as u64).sum::<u64>())
    };
    let fewer_issued = || matches!((issued(func, &relaxed.0), issued(&ordered, &strict.0)), (Some(a), Some(b)) if a <= b);
    let better = r.0 < s.0
        || r.0 == s.0
            && (r.1 > s.1 || r.1 == s.1 && (faster(r.2, s.2) || r.2 <= s.2 && faster(r.3, s.3)) && fewer_issued());
    if better {
        return relaxed;
    }
    *func = ordered;
    strict
}

#[derive(Debug, Default, Clone)]
pub struct CompileStats {
    pub instructions: usize,
    pub code_bytes: usize,
    pub vgprs: u32,
    pub sgprs: u32,
    pub lds_bytes: u32,
    pub kernarg_bytes: u32,
    pub phases: Vec<(&'static str, Duration)>,
    /// Most VGPR / SGPR dwords live at once in the allocated schedule.
    pub pressure_vgprs: u32,
    pub pressure_sgprs: u32,
    /// Scheduling window taken (0: load sinking alone; 1: program order).
    pub window: usize,
    /// Waves per SIMD the resources allow.
    pub waves: u32,
    /// VOPD dual-issue instructions, and those weighted by 8^loop depth.
    pub vopd_pairs: u32,
    pub vopd_weighted: u64,
    pub spill: regalloc::SpillStats,
    /// Loops unrolled beyond the IR's hints (`unroll::Auto`).
    pub unrolled: usize,
}

pub struct CompiledKernel {
    pub name: String,
    pub hsaco: Vec<u8>,
    pub stats: CompileStats,
    /// The target the code object is for.
    pub target: Target,
    func: mir::MFunc,
    /// Plain first-fit did not fit the hardware (see `compile`).
    tight: bool,
}

impl CompiledKernel {
    /// Text listing of the machine code with byte offsets and encodings.
    pub fn listing(&self) -> String {
        match self.target {
            Target::Gfx1151 => asm::listing(&self.func),
            Target::Gfx950 => crate::target::gfx9::asm::listing(&self.func),
        }
        .expect("the code assembled once")
    }
}

/// Compiles one rocKE IR kernel to a loadable gfx1151 code object.
pub fn compile(text: &str) -> Result<CompiledKernel, CompileError> {
    compile_parsed_by(text, &parse::parse, Target::Gfx1151)
}

/// Compiles one rocKE IR kernel to a loadable code object for `target`; a
/// target the compiler does not support is a `request` error.
pub fn compile_for(text: &str, target: Target) -> Result<CompiledKernel, CompileError> {
    compile_parsed_by(text, &parse::parse, target.require_supported()?)
}

/// `compile_for` without the support check, for a target the back end
/// knows but does not support yet (gfx950): its code objects are for the
/// tests and offline checks that bring the target up. Sessions and the FFI
/// never call it.
pub fn compile_any_target(text: &str, target: Target) -> Result<CompiledKernel, CompileError> {
    compile_parsed_by(text, &parse::parse, target)
}

/// `compile_any_target` with at most `vgprs` VGPRs for the allocation:
/// below what the kernel needs it spills, which is how the tests and the
/// offline checks exercise spilling where no kernel needs it. Sessions and
/// the FFI never call it.
pub fn compile_any_target_limited(text: &str, target: Target, vgprs: u32) -> Result<CompiledKernel, CompileError> {
    let limited = |text: &str| parse::parse(text).map(|kernel| ast::Kernel { vgpr_limit: Some(vgprs), ..kernel });
    compile_parsed_by(text, &limited, target)
}

/// `compile_for`, with `parse` (which gives `parse::parse`'s result)
/// parsing the text, as a session does on its own threads.
pub(crate) fn compile_parsed_by(
    text: &str,
    parse: &dyn Fn(&str) -> Result<ast::Kernel, CompileError>,
    target: Target,
) -> Result<CompiledKernel, CompileError> {
    compile_parsed_by_on(text, parse, 0, target)
}

/// `compile_parsed_by` with independent steps on up to `threads` threads
/// (0: as the machine allows, at most four); the result is the same for
/// every count.
pub(crate) fn compile_parsed_by_on(
    text: &str,
    parse: &dyn Fn(&str) -> Result<ast::Kernel, CompileError>,
    threads: usize,
    target: Target,
) -> Result<CompiledKernel, CompileError> {
    if debug_flag("nounroll") {
        return compile_retrying(text, parse, None, &mut 0, threads, target);
    }
    // Loops unrolled beyond the IR's hints stay unrolled only if that
    // spills no more than leaving them rolled and keeps its occupancy;
    // without spills at the most waves there are, nothing can do better.
    // A retry can keep an earlier attempt's result, so a result is judged by
    // its own count; `unrolled` (the last attempt's) only covers an error.
    let mut unrolled = 0;
    let first = compile_retrying(text, parse, Some(unroll::Auto::DEFAULT), &mut unrolled, threads, target);
    match &first {
        Ok(u) if u.stats.unrolled == 0 => return first,
        Err(_) if unrolled == 0 => return first,
        Ok(u) if spilled(&u.stats) == 0 && u.stats.waves >= target.info().max_waves => return first,
        _ => {}
    }
    let rolled = compile_retrying(text, parse, None, &mut 0, threads, target);
    match (first, rolled) {
        (Ok(u), Ok(r)) => Ok(if keeps_unrolled(&u.stats, &r.stats) { u } else { r }),
        (Ok(u), Err(_)) => Ok(u),
        (Err(_), rolled) => rolled,
    }
}

/// VGPR dwords spilled to LDS or scratch.
fn spilled(stats: &CompileStats) -> u32 {
    stats.spill.lds_dwords + stats.spill.scratch_dwords
}

/// Whether a compile with loops unrolled by `unroll::Auto` is kept over
/// the compile that leaves them rolled: it spills no more and keeps the
/// occupancy.
fn keeps_unrolled(unrolled: &CompileStats, rolled: &CompileStats) -> bool {
    spilled(unrolled) <= spilled(rolled) && unrolled.waves >= rolled.waves
}

/// `compile_with` and its retries; `unrolled` gets the loops `auto` unrolled.
fn compile_retrying(
    text: &str,
    parse: &dyn Fn(&str) -> Result<ast::Kernel, CompileError>,
    auto: Option<unroll::Auto>,
    unrolled: &mut usize,
    threads: usize,
    target: Target,
) -> Result<CompiledKernel, CompileError> {
    // Merged LDS loads live longer: a kernel they push out of registers
    // compiles without them, and spills only if it still does not fit. One
    // that only packing tuples first fits also tries without them (as plain
    // first fit, which did not fit, always did) and keeps the higher
    // occupancy.
    let result = match compile_with(text, true, false, parse, auto, unrolled, threads, target) {
        Err(failure) if failure.error().kind == "ROCKE_DIRECT_REGISTERS" => {
            match compile_with(text, false, false, parse, auto, unrolled, threads, target) {
                // Spilling compiles the same up to where this stopped.
                Err(Failure::Unfit(resume)) => return resume.spill(*unrolled, threads),
                Err(failure) if failure.error().kind == "ROCKE_DIRECT_REGISTERS" => {
                    compile_with(text, false, true, parse, auto, unrolled, threads, target)
                }
                result => result,
            }
        }
        Ok(packed) if packed.tight => match compile_with(text, false, false, parse, auto, unrolled, threads, target) {
            Ok(other) if other.stats.waves > packed.stats.waves => Ok(other),
            _ => Ok(packed),
        },
        result => result,
    };
    result.map_err(Failure::into_error)
}

/// Phase times of a compile so far.
struct Laps {
    phases: Vec<(&'static str, Duration)>,
    mark: Instant,
}

impl Laps {
    fn lap(&mut self, name: &'static str) {
        let now = Instant::now();
        self.phases.push((name, now - self.mark));
        self.mark = now;
    }
}

/// A kernel through instruction selection, and its phase times.
struct Selected {
    kernel: ast::Kernel,
    sel: isel::IselResult,
    laps: Laps,
}

/// Why `compile_with` gives no kernel.
enum Failure {
    Error(CompileError),
    /// Nothing fits without spilling: the error, and where a compile with
    /// spilling goes on from.
    Unfit(Box<Resume>),
}

impl Failure {
    fn error(&self) -> &CompileError {
        match self {
            Failure::Error(error) => error,
            Failure::Unfit(resume) => &resume.error,
        }
    }

    fn into_error(self) -> CompileError {
        match self {
            Failure::Error(error) => error,
            Failure::Unfit(resume) => resume.error,
        }
    }
}

impl From<CompileError> for Failure {
    fn from(error: CompileError) -> Self {
        Failure::Error(error)
    }
}

/// A compile without spilling stopped where nothing fit: the same compile
/// with spilling gets there with this state.
struct Resume {
    error: CompileError,
    selected: Selected,
    candidates: Candidates,
}

impl Resume {
    /// The compile with spilling, from where the one without stopped. With
    /// alias facts that relax an order, the candidates are those of the
    /// schedule without them, and a spill that places nothing still competes
    /// with the facts' schedule, as in `schedule_with_facts`.
    fn spill(self, unrolled: usize, threads: usize) -> Result<CompiledKernel, CompileError> {
        let Resume { selected: mut s, candidates, .. } = self;
        let (lds_size, wg) = (s.sel.lds_size, s.kernel.max_workgroup_size);
        let (alloc, window) = if alias::relaxes(&s.sel.func) {
            let mut ordered = mir::MFunc::default();
            let strict = spill_best(&mut ordered, candidates, lds_size, wg, threads)?;
            prefer_facts(&mut s.sel.func, ordered, strict, lds_size, wg, true, threads)
        } else {
            spill_best(&mut s.sel.func, candidates, lds_size, wg, threads)?
        };
        complete(s, alloc, window, unrolled)
    }
}

fn compile_with(
    text: &str,
    lds_cse: bool,
    spill: bool,
    parse: &dyn Fn(&str) -> Result<ast::Kernel, CompileError>,
    auto: Option<unroll::Auto>,
    unrolled: &mut usize,
    threads: usize,
    target: Target,
) -> Result<CompiledKernel, Failure> {
    let mut laps = Laps { phases: Vec::new(), mark: Instant::now() };
    let mut kernel = parse(text)?;
    kernel.wave = target.info().wave;
    validate::kernel(&kernel)?;
    let (mut lds, lds_size) = analysis::lds_layout(&kernel);
    validate::lds_pool(lds_size, target.info())?;
    laps.lap("parse");
    let facts = if lds_cse {
        opt::optimize(&mut kernel, &mut lds)?
    } else {
        opt::optimize_keeping_lds_loads(&mut kernel, &mut lds)?
    };
    let facts = match auto {
        Some(auto) => {
            let (facts, n) = unroll::unroll_optimized(&mut kernel, &mut lds, facts, auto, !lds_cse)?;
            *unrolled = n;
            facts
        }
        None => facts,
    };
    laps.lap("optimize");
    let (facts, lds_size) = lds_swizzle::run(&mut kernel, facts, &mut lds, lds_size);
    laps.lap("swizzle");
    let uniform = analysis::classify(&kernel);
    validate::uniformity(&kernel, &uniform)?;
    let mut sel = isel::select_for(target, &kernel, &facts, uniform, &lds, lds_size)?;
    sel.func.vgpr_limit = kernel.vgpr_limit;
    post::remove_dead_copies(&mut sel.func);
    if target == Target::Gfx950 {
        // The peephole rules, guard reuse and VALU fusion are proven for gfx11 only.
        crate::target::gfx9::legalize::run(&mut sel.func);
    } else {
        peephole::optimize(&mut sel.func);
        uniform_masks::reuse_masks(&mut sel.func);
        if !debug_flag("nofuse") {
            post::fuse_valu(&mut sel.func);
        }
    }
    laps.lap("isel");
    if !lds_cse || spill {
        mma_order::fit(&mut sel.func);
    }
    // Overlapping latencies costs registers: take the widest scheduling
    // window that keeps the occupancy of load sinking alone.
    let (alloc, window) = match fma_mix::schedule_folding(&mut sel.func, sel.lds_size, kernel.max_workgroup_size, spill, threads) {
        Ok(chosen) => chosen,
        Err(Unallocated::Error(error)) => return Err(Failure::Error(error)),
        Err(Unallocated::Unfit(error, candidates)) => {
            let selected = Selected { kernel, sel, laps };
            return Err(Failure::Unfit(Box::new(Resume { error, selected, candidates })));
        }
    };
    Ok(complete(Selected { kernel, sel, laps }, alloc, window, *unrolled)?)
}

/// The rest of a compile once registers are assigned.
fn complete(
    selected: Selected,
    alloc: regalloc::Allocation,
    window: usize,
    unrolled: usize,
) -> Result<CompiledKernel, CompileError> {
    let Selected { kernel, mut sel, mut laps } = selected;
    let target = sel.func.target;
    let lds_total = sel.lds_size + alloc.spill.lds_bytes;
    reload::hoist(&mut sel.func, &alloc, sel.lds_size, target.info().occupancy(alloc.num_vgprs.max(1), lds_total, kernel.max_workgroup_size));
    laps.lap("regalloc");
    let (vopd_pairs, vopd_weighted) = finish(&mut sel.func, alloc.num_vgprs)?;
    laps.lap("post");
    let code = match target {
        Target::Gfx1151 => asm::assemble(&sel.func)?,
        Target::Gfx950 => crate::target::gfx9::asm::assemble(&sel.func)?,
    };
    laps.lap("encode");

    let vgprs = alloc.num_vgprs.max(1);
    let rsrc1 = 0xe0af_0000 | (vgprs.div_ceil(8) - 1);
    let rsrc2 = (2 << 1)
        | (sel.wg_ids[0] as u32) << 7
        | (sel.wg_ids[1] as u32) << 8
        | (sel.wg_ids[2] as u32) << 9
        | u32::from(alloc.spill.scratch_bytes > 0);
    let rsrc3 = (code.len().div_ceil(128).min(63) as u32) << 4;
    let (rsrc1, rsrc2, rsrc3, sgpr_count) = match target {
        Target::Gfx1151 => (rsrc1, rsrc2, rsrc3, alloc.num_sgprs + 2),
        Target::Gfx950 => {
            let scratch = alloc.spill.scratch_bytes > 0;
            let r = crate::target::gfx9::program_resources(vgprs, alloc.num_sgprs, sel.wg_ids, scratch);
            (r.rsrc1, r.rsrc2, r.rsrc3, r.sgpr_count)
        }
    };
    let (offsets, kernarg_size) = opt::kernarg_layout(&kernel);
    let args = kernel
        .params
        .iter()
        .zip(&offsets)
        .map(|(p, &offset)| {
            let (size, pointer) = match p.ty {
                Type::Ptr(_) => (8, true),
                Type::Scalar(Scalar::I64) => (8, false),
                _ => (4, false),
            };
            KernelArgMeta {
                name: p.name.clone(),
                offset,
                size,
                pointer,
                access: None,
            }
        })
        .collect();
    let spec = KernelObjectSpec {
        target: sel.func.target.info().object,
        symbol: &kernel.name,
        code: &code,
        args,
        kernarg_size,
        group_segment_size: lds_total,
        max_flat_workgroup_size: kernel.max_workgroup_size,
        sgpr_count,
        vgpr_count: vgprs,
        rsrc1,
        rsrc2,
        rsrc3,
        private_segment_size: alloc.spill.scratch_bytes,
        vgpr_spill_count: alloc.spill.lds_dwords + alloc.spill.scratch_dwords,
    };
    let hsaco = write_kernel_object(&spec).map_err(|e| CompileError::new("ROCKE_DIRECT_ELF", e))?;
    laps.lap("elf");
    let instructions = sel.func.blocks.iter().map(|b| b.insts.len()).sum();
    Ok(CompiledKernel {
        name: kernel.name.clone(),
        hsaco,
        stats: CompileStats {
            instructions,
            code_bytes: code.len(),
            vgprs,
            sgprs: alloc.num_sgprs,
            lds_bytes: lds_total,
            kernarg_bytes: kernarg_size,
            phases: laps.phases,
            pressure_vgprs: alloc.pressure.vgprs,
            pressure_sgprs: alloc.pressure.sgprs,
            window,
            waves: sel.func.target.info().occupancy(vgprs, lds_total, kernel.max_workgroup_size),
            vopd_pairs,
            vopd_weighted,
            spill: alloc.spill,
            unrolled,
        },
        target: sel.func.target,
        func: sel.func,
        tight: alloc.tight && window <= 1,
    })
}
