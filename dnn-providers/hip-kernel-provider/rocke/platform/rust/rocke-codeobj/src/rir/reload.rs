// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

//! Spill-reload placement after register assignment. The spiller reloads a
//! value right before its use. VMEM loads return in order, so the wait for
//! such a reload also waits for every load issued before it: a loop that
//! reloads addresses between its gathers runs the gathers one at a time.
//! Each reload moves up its block, past instructions that leave its
//! registers and its slot alone, as far as a register stays free for it:
//! its own, or one free from there to the reload's last reader, which then
//! reads it from that register. Reloads from LDS slots move the same way:
//! their waits drain the LDS loads issued before them.

use super::mir::*;
use super::regalloc::Allocation;

/// A set of VGPRs.
type Regs = [u64; 4];

fn has(regs: &Regs, r: u32) -> bool {
    regs[(r / 64) as usize] >> (r % 64) & 1 != 0
}

fn span(r: u32, n: u32) -> Regs {
    let mut m = [0; 4];
    for k in r..(r + n).min(256) {
        m[(k / 64) as usize] |= 1 << (k % 64);
    }
    m
}

fn meets(a: &Regs, b: &Regs) -> bool {
    a.iter().zip(b).any(|(x, y)| x & y != 0)
}

fn within(a: &Regs, b: &Regs) -> bool {
    a.iter().zip(b).all(|(x, y)| x & !y == 0)
}

fn add(a: &mut Regs, b: &Regs) {
    for (x, y) in a.iter_mut().zip(b) {
        *x |= y;
    }
}

fn vgprs(o: &Opnd) -> Option<(u32, u32)> {
    match o {
        Opnd::P(Phys::V(r), n) => Some((*r as u32, *n as u32)),
        _ => None,
    }
}

/// Whether `inst` also reads the registers it writes: tied operands, half
/// writes and lane writes keep what they do not overwrite.
fn keeps_dest(inst: &MInst) -> bool {
    inst.op.flags() & (F_TIED | F_16BIT_DST) != 0
        || inst.mods & M_OPSEL_DST != 0
        || matches!(inst.op, Opc::VFmaMixloF16 | Opc::VFmaMixhiF16 | Opc::VWritelaneB32)
}

/// The VGPRs an instruction reads and writes.
#[derive(Clone, Copy, Default)]
struct Access {
    read: Regs,
    write: Regs,
}

impl Access {
    fn of(inst: &MInst) -> Access {
        let mut a = Access::default();
        for (r, n) in inst.uses.iter().filter_map(vgprs) {
            add(&mut a.read, &span(r, n));
        }
        for (r, n) in inst.defs.iter().filter_map(vgprs) {
            add(&mut a.write, &span(r, n));
        }
        if keeps_dest(inst) {
            let w = a.write;
            add(&mut a.read, &w);
        }
        a
    }

    fn any(&self) -> Regs {
        let mut m = self.read;
        add(&mut m, &self.write);
        m
    }
}

/// Where a spill access keeps its value: a range of scratch bytes, or the
/// LDS slot at an offset.
#[derive(Clone, Copy, PartialEq, Debug)]
enum Slot {
    Scratch(i64, i64),
    Lds(i64),
}

fn overlap(a: Slot, b: Slot) -> bool {
    match (a, b) {
        (Slot::Scratch(a0, a1), Slot::Scratch(b0, b1)) => a0 < b1 && b0 < a1,
        (Slot::Lds(a), Slot::Lds(b)) => a == b,
        _ => false,
    }
}

/// How spill accesses look after assignment: scratch accesses are all
/// spills; LDS slots lie past the static pool, at the per-lane address
/// register the spiller computes at entry.
struct Spills {
    lds_base: i64,
    lds_addr: Option<u32>,
}

impl Spills {
    fn new(f: &MFunc, alloc: &Allocation, lds_base: u32) -> Spills {
        let lds_base = lds_base as i64;
        Spills { lds_base, lds_addr: lds_address(f, alloc, lds_base) }
    }

    /// The slot of a spill reload or store, and whether it reloads.
    fn slot(&self, inst: &MInst) -> Option<(Slot, bool)> {
        let scratch = |o: &Opnd| Slot::Scratch(inst.imm, inst.imm + 4 * o.dwords() as i64);
        match inst.op {
            Opc::ScratchLoadB32 | Opc::ScratchLoadB64 | Opc::ScratchLoadB96 | Opc::ScratchLoadB128 => {
                Some((scratch(&inst.defs[0]), true))
            }
            Opc::ScratchStoreB32 | Opc::ScratchStoreB64 | Opc::ScratchStoreB96 | Opc::ScratchStoreB128 => {
                Some((scratch(&inst.uses[0]), false))
            }
            Opc::DsLoadB32 | Opc::DsStoreB32
                if inst.imm >= self.lds_base && self.lds_addr.is_some() && vgprs(&inst.uses[0]).map(|(r, _)| r) == self.lds_addr =>
            {
                Some((Slot::Lds(inst.imm), inst.op == Opc::DsLoadB32))
            }
            _ => None,
        }
    }
}

/// The spiller's per-lane LDS address register, `v_lshlrev_b32 a, 2, v0`
/// first at entry: written nowhere else, and the address of every LDS
/// access past the static pool.
fn lds_address(f: &MFunc, alloc: &Allocation, lds_base: i64) -> Option<u32> {
    if alloc.spill.lds_dwords == 0 {
        return None;
    }
    let first = f.blocks.first()?.insts.iter().find(|i| i.op != Opc::ImplicitDef)?;
    let a = match (first.op, &first.defs[..], &first.uses[..]) {
        (Opc::VLshlrevB32, [d], [Opnd::Imm(2), Opnd::P(Phys::V(0), 1)]) => vgprs(d).filter(|&(_, n)| n == 1)?.0,
        _ => return None,
    };
    let mut writes = 0;
    for inst in f.blocks.iter().flat_map(|b| &b.insts) {
        writes += has(&Access::of(inst).write, a) as u32;
        if inst.op.flags() & F_LDS != 0 && inst.imm >= lds_base && inst.uses.first().and_then(vgprs) != Some((a, 1)) {
            return None;
        }
    }
    (writes == 1).then_some(a)
}

/// VGPRs live at the end of each block.
fn live_out(f: &MFunc) -> Vec<Regs> {
    let n = f.blocks.len();
    let mut gen = vec![[0u64; 4]; n];
    let mut kill = vec![[0u64; 4]; n];
    for (b, block) in f.blocks.iter().enumerate() {
        for inst in &block.insts {
            let a = Access::of(inst);
            for w in 0..4 {
                gen[b][w] |= a.read[w] & !kill[b][w];
                kill[b][w] |= a.write[w];
            }
        }
    }
    let succs: Vec<Vec<usize>> = (0..n).map(|b| f.successors(b)).collect();
    let mut live_in = vec![[0u64; 4]; n];
    let mut out = vec![[0u64; 4]; n];
    loop {
        let mut changed = false;
        for b in (0..n).rev() {
            let mut o = [0u64; 4];
            for &s in &succs[b] {
                add(&mut o, &live_in[s]);
            }
            let mut i = [0u64; 4];
            for w in 0..4 {
                i[w] = gen[b][w] | (o[w] & !kill[b][w]);
            }
            changed |= o != out[b] || i != live_in[b];
            out[b] = o;
            live_in[b] = i;
        }
        if !changed {
            return out;
        }
    }
}

/// Whether `inst` changes EXEC: a saveexec, or any other write of EXEC, such
/// as the `s_or` that restores it where a divergent `if` joins.
fn writes_exec(inst: &MInst) -> bool {
    inst.op.flags() & F_DEF_EXEC != 0 || inst.defs.iter().any(|d| matches!(d, Opnd::P(Phys::ExecLo, _)))
}

/// Whether a reload may not move above `inst`: control flow, EXEC changes,
/// fences and barriers (each waits for every load), a write of a register
/// the reload reads, or a store to its slot.
fn stops(inst: &MInst, access: &Access, slot: Slot, reads: &Regs, spills: &Spills) -> bool {
    inst.op.flags() & (F_BRANCH | F_END | F_FENCE) != 0
        || writes_exec(inst)
        || matches!(inst.op, Opc::SBarrier | Opc::SSendmsg | Opc::SWaitcnt)
        || meets(&access.write, reads)
        || spills.slot(inst).is_some_and(|(s, reload)| !reload && overlap(s, slot))
}

/// The VGPRs whose values are dead after `insts[last]`: written before they
/// are read, or untouched to the end of the block and dead there. Past an
/// EXEC change a write may leave lanes alone, so nothing counts as written.
fn dead_after(insts: &[MInst], access: &[Access], last: usize, live_out: &Regs) -> Regs {
    let mut known = [0u64; 4];
    let mut dead = [0u64; 4];
    for (inst, a) in insts.iter().zip(access).skip(last + 1) {
        if writes_exec(inst) {
            return dead;
        }
        for w in 0..4 {
            let read = a.read[w] & !known[w];
            let written = a.write[w] & !a.read[w] & !known[w];
            dead[w] |= written;
            known[w] |= read | written;
        }
    }
    for w in 0..4 {
        dead[w] |= !known[w] & !live_out[w];
    }
    dead
}

/// VALU readers among `readers` whose VGPR sources would share a bank
/// (register mod 4) with the reloaded `d..d + n` read from `x` instead.
fn clashes(readers: &[MInst], d: u32, n: u32, x: u32) -> u32 {
    let mut count = 0;
    for inst in readers.iter().filter(|i| i.op.flags() & F_USE_EXEC != 0 && i.op.flags() & F_WMMA == 0) {
        let mut banks = [0u32; 4];
        for (r, _) in inst.uses.iter().filter_map(vgprs) {
            let r = if r >= d && r < d + n { x + r - d } else { r };
            banks[(r % 4) as usize] += 1;
        }
        count += banks.iter().filter(|&&k| k > 1).count() as u32;
    }
    count
}

/// Whether `inst` is one of the kernel's own loads on the counter of a
/// reload from `slot`: VMEM loads for scratch, LDS loads for LDS slots.
fn kernel_load(inst: &MInst, slot: Slot, spills: &Spills) -> bool {
    let kind = match slot {
        Slot::Scratch(..) => F_VMEM_LOAD,
        Slot::Lds(_) => F_LDS,
    };
    inst.op.flags() & kind != 0 && !inst.defs.is_empty() && spills.slot(inst).is_none()
}

/// Waves per SIMD up to which a reload moves as far as a register is free
/// for it: few waves hide little of the latency it leaves exposed. With
/// more, a reload takes another register only to pass the kernel's loads
/// still in flight before it, which its wait would drain; moving it further
/// measured slower at four waves (`conv_igemm` t256x256x32 mem, fp16).
const FEW_WAVES: u32 = 2;

/// Moves the spill reloads of one block up; returns how many moved.
fn hoist_block(insts: &mut Vec<MInst>, live_out: &Regs, num_vgprs: u32, waves: u32, spills: &Spills) -> u32 {
    let mut access: Vec<Access> = insts.iter().map(Access::of).collect();
    let mut moved = 0;
    for p in 0..insts.len() {
        let Some((slot, true)) = spills.slot(&insts[p]) else { continue };
        let Some((d, n)) = vgprs(&insts[p].defs[0]) else { continue };
        let reads = access[p].read;
        let own = span(d, n);
        let mut top = p;
        while top > 0 && !stops(&insts[top - 1], &access[top - 1], slot, &reads, spills) {
            top -= 1;
        }
        // Reloads already moved up stay first, in the order of their uses.
        while top < p && spills.slot(&insts[top]).is_some_and(|(_, reload)| reload) {
            top += 1;
        }
        if top == p {
            continue;
        }
        // The readers of the reloaded value, up to the last before a write
        // replaces it; whether they can read it from another register.
        let mut last = None;
        let mut rename = true;
        let mut end = p + 1;
        while end < insts.len() {
            let a = access[end];
            if meets(&a.read, &own) {
                last = Some(end);
                let inside = |o: &Opnd| vgprs(o).is_none_or(|(r, k)| r + k <= d || r >= d + n || (r >= d && r + k <= d + n));
                rename &= insts[end].uses.iter().all(inside) && !(meets(&a.write, &own) && keeps_dest(&insts[end]));
            }
            if meets(&a.write, &own) {
                rename &= within(&own, &a.write);
                break;
            }
            end += 1;
        }
        let Some(last) = last else { continue };
        rename &= !(end == insts.len() && meets(live_out, &own))
            && insts[p + 1..=last].iter().all(|i| !writes_exec(i));
        // Below its last access above the reload, each register is free; the
        // first kernel load above whose result nothing reads before it.
        let mut free_from = vec![top; num_vgprs as usize];
        let mut seen = [0u64; 4];
        let mut in_flight = p;
        for k in (top..p).rev() {
            let m = access[k].any();
            if kernel_load(&insts[k], slot, spills) && !meets(&access[k].write, &seen) {
                in_flight = k;
            }
            for w in 0..4 {
                let mut fresh = m[w] & !seen[w];
                while fresh != 0 {
                    let r = w * 64 + fresh.trailing_zeros() as usize;
                    fresh &= fresh - 1;
                    if let Some(at) = free_from.get_mut(r) {
                        *at = k + 1;
                    }
                }
                seen[w] |= m[w];
            }
        }
        let from = |r: u32| (r..r + n).map(|k| free_from.get(k as usize).copied().unwrap_or(p)).max().unwrap_or(p);
        let mut best = (from(d), None);
        let reach = if waves <= FEW_WAVES { top } else { in_flight };
        if rename && best.0 > reach {
            let mut busy = own;
            add(&mut busy, &reads);
            for a in &access[p + 1..=last] {
                add(&mut busy, &a.any());
            }
            let dead = dead_after(insts, &access, last, live_out);
            let mut fewest = clashes(&insts[p + 1..=last], d, n, d);
            for x in 0..num_vgprs.saturating_sub(n - 1) {
                let xs = span(x, n);
                if meets(&xs, &busy) || !within(&xs, &dead) {
                    continue;
                }
                let at = from(x).max(reach);
                if at > best.0 {
                    continue;
                }
                let c = clashes(&insts[p + 1..=last], d, n, x);
                if (at, c) < (best.0, fewest) {
                    best = (at, Some(x));
                    fewest = c;
                }
            }
        }
        let (q, to) = best;
        if q >= p {
            continue;
        }
        if let Some(x) = to {
            insts[p].defs[0] = Opnd::P(Phys::V(x as u16), n as u8);
            access[p] = Access::of(&insts[p]);
            for k in p + 1..=last {
                for o in insts[k].uses.iter_mut() {
                    if let Some((r, c)) = vgprs(o).filter(|&(r, c)| r >= d && r + c <= d + n) {
                        *o = Opnd::P(Phys::V((x + r - d) as u16), c as u8);
                    }
                }
                access[k] = Access::of(&insts[k]);
            }
        }
        insts[q..=p].rotate_right(1);
        access[q..=p].rotate_right(1);
        moved += 1;
    }
    moved
}

/// Moves each spill reload of the allocated `f` up its block (see the
/// module notes). `lds_base` is the static LDS pool, which LDS slots follow;
/// `waves` the waves per SIMD the allocation leaves room for.
pub(super) fn hoist(f: &mut MFunc, alloc: &Allocation, lds_base: u32, waves: u32) {
    if alloc.spill.lds_dwords + alloc.spill.scratch_dwords == 0 || super::debug_flag("nohoist") {
        return;
    }
    let spills = Spills::new(f, alloc, lds_base);
    let out = live_out(f);
    for (block, live) in f.blocks.iter_mut().zip(&out) {
        hoist_block(&mut block.insts, live, alloc.num_vgprs, waves, &spills);
    }
}

#[cfg(test)]
mod tests {
    use super::super::spill::tests::{effects, random_kernel};
    use super::super::{post, regalloc};
    use super::*;

    fn v(r: u16) -> Opnd {
        Opnd::P(Phys::V(r), 1)
    }

    fn s(r: u16) -> Opnd {
        Opnd::P(Phys::S(r), 4)
    }

    fn reload(dst: u16, offset: i64) -> MInst {
        MInst::new(Opc::ScratchLoadB32, [v(dst)], []).with_imm(offset)
    }

    fn gather(dst: u16, addr: u16) -> MInst {
        MInst::new(Opc::BufLoadB32, [v(dst)], [v(addr), s(0)])
    }

    fn add(dst: u16, a: u16, b: u16) -> MInst {
        MInst::new(Opc::VAddNcU32, [v(dst)], [v(a), v(b)])
    }

    fn scratch_spill() -> Allocation {
        let mut a = Allocation { num_vgprs: 16, ..Allocation::default() };
        a.spill.scratch_dwords = 4;
        a
    }

    fn run_with(insts: Vec<MInst>, live_out: &Regs, waves: u32) -> Vec<MInst> {
        let mut f = MFunc { blocks: vec![MBlock { insts, loop_depth: 1 }], vregs: Vec::new(), ..Default::default() };
        let spills = Spills::new(&f, &scratch_spill(), 0);
        hoist_block(&mut f.blocks[0].insts, live_out, 16, waves, &spills);
        f.blocks.remove(0).insts
    }

    fn run(insts: Vec<MInst>, live_out: &Regs) -> Vec<MInst> {
        run_with(insts, live_out, 1)
    }

    fn text(insts: &[MInst]) -> Vec<String> {
        let f = MFunc::default();
        insts.iter().map(|i| inst_text(&f, i)).collect()
    }

    fn gathers() -> Vec<MInst> {
        vec![
            add(1, 0, 0),
            gather(2, 1),
            reload(3, 0),
            add(4, 3, 0),
            gather(5, 4),
            reload(1, 4),
            add(6, 1, 0),
            MInst::new(Opc::BufStoreB32, [], [v(2), v(5), v(6), s(0)]),
        ]
    }

    /// A reload between two gathers moves above the first, in its own
    /// register when that is free; a reload whose register the first
    /// gather's address reads takes a free one, and its reader follows.
    /// With few waves it moves as far as that register is free; with more,
    /// just above the gathers still in flight.
    #[test]
    fn reloads_move_above_the_gathers_before_them() {
        let store = MInst::new(Opc::BufStoreB32, [], [v(2), v(5), v(6), s(0)]);
        let few = vec![reload(3, 0), reload(7, 4), add(1, 0, 0), gather(2, 1), add(4, 3, 0), gather(5, 4), add(6, 7, 0), store.clone()];
        assert_eq!(text(&run_with(gathers(), &[0; 4], 1)), text(&few));
        let many = vec![reload(3, 0), add(1, 0, 0), reload(7, 4), gather(2, 1), add(4, 3, 0), gather(5, 4), add(6, 7, 0), store];
        assert_eq!(text(&run_with(gathers(), &[0; 4], 4)), text(&many));
    }

    /// With many waves, a reload whose register is busy above it and that
    /// has no kernel load in flight to pass keeps its register: the gather
    /// before it is read already.
    #[test]
    fn reloads_keep_their_register_with_nothing_in_flight_and_many_waves() {
        let consumed = MInst::new(Opc::DsStoreB32, [], [v(9), v(2)]);
        let insts = vec![add(1, 0, 0), gather(2, 1), consumed.clone(), add(8, 1, 1), reload(1, 4), add(6, 1, 0)];
        let kept = vec![add(1, 0, 0), gather(2, 1), consumed.clone(), add(8, 1, 1), reload(1, 4), add(6, 1, 0)];
        assert_eq!(text(&run_with(insts.clone(), &[0; 4], 4)), text(&kept));
        let renamed = vec![reload(3, 4), add(1, 0, 0), gather(2, 1), consumed, add(8, 1, 1), add(6, 3, 0)];
        assert_eq!(text(&run_with(insts, &[0; 4], 2)), text(&renamed));
    }

    /// What a reload must not pass, and readers it cannot move away from.
    #[test]
    fn reloads_stay_below_stores_to_their_slot_barriers_and_exec_changes() {
        let store = MInst::new(Opc::ScratchStoreB32, [], [v(9)]).with_imm(0);
        let other = MInst::new(Opc::ScratchStoreB32, [], [v(9)]).with_imm(8);
        let exec = MInst::new(Opc::SAndSaveexecB32, [Opnd::P(Phys::S(2), 1), Opnd::P(Phys::ExecLo, 1)], [Opnd::P(Phys::S(3), 1), Opnd::P(Phys::ExecLo, 1)]);
        let barrier = MInst::new(Opc::SBarrier, [], []);
        for (stop, passes) in [(store, false), (other, true), (exec, false), (barrier, false)] {
            let out = run(vec![stop.clone(), gather(2, 1), reload(3, 0), add(4, 3, 3)], &[0; 4]);
            let first = if passes { reload(3, 0) } else { stop.clone() };
            assert_eq!(text(&out[..1]), text(&[first]), "{}", text(&[stop])[0]);
        }
        // A tied reader (fmac reads and writes its accumulator) keeps the
        // reload's register, so the reload stays below that register's use.
        let fmac = MInst::new(Opc::VFmacF32, [v(3)], [v(5), v(6), v(3)]);
        let out = run(vec![add(9, 3, 3), gather(2, 1), reload(3, 0), fmac.clone()], &[0; 4]);
        assert_eq!(text(&out), text(&[add(9, 3, 3), reload(3, 0), gather(2, 1), fmac]));
        // A value still live at the end of the block stays in its register.
        let out = run(vec![add(9, 3, 3), gather(2, 1), reload(3, 0), add(4, 3, 3)], &span(3, 1));
        assert_eq!(text(&out), text(&[add(9, 3, 3), reload(3, 0), gather(2, 1), add(4, 3, 3)]));
    }

    /// The `s_or` that restores EXEC where a divergent `if` joins stops a
    /// reload, in wave32 and wave64: above it, the reload would run in the
    /// lanes of the `if` only.
    #[test]
    fn reloads_stay_below_the_exec_restore_of_a_join() {
        for (op, w) in [(Opc::SOrB32, 1), (Opc::SOrB64, 2)] {
            let join = MInst::new(op, [Opnd::P(Phys::ExecLo, w)], [Opnd::P(Phys::ExecLo, w), Opnd::P(Phys::S(2), w)]);
            let out = run(vec![join.clone(), gather(2, 1), reload(3, 0), add(4, 3, 3)], &[0; 4]);
            assert_eq!(text(&out[..1]), text(&[join]));
        }
    }

    /// Spilled random kernels compute the same values with their reloads
    /// moved, with few waves and with many, and some reload moves.
    #[test]
    fn moved_reloads_compute_the_same_values() {
        let mut moved = 0;
        for (seed, nv, ns, lds_slots) in [(1, 60, 30, 0), (2, 80, 40, 8), (3, 120, 12, 64), (4, 40, 70, 4), (5, 90, 20, 16)] {
            let f = random_kernel(seed, nv, ns);
            let want = effects(&f, 3);
            let cfg = regalloc::SpillConfig { lds_size: 256, wg_size: 32, lds_slots };
            let mut spilled = f.clone();
            let a = regalloc::allocate_spilling(&mut spilled, regalloc::Limits { vgprs: 24, sgprs: 20 }, cfg).expect("allocates");
            for waves in [1, 8] {
                let mut g = spilled.clone();
                hoist(&mut g, &a, cfg.lds_size, waves);
                let flat = |f: &MFunc| text(&f.blocks.iter().flat_map(|b| b.insts.clone()).collect::<Vec<_>>());
                moved += flat(&spilled).iter().zip(&flat(&g)).filter(|(x, y)| x != y).count();
                assert_eq!(effects(&g, 3), want, "seed {seed}, {waves} waves");
                post::expand_pseudos(&mut g).expect("expands");
                super::super::asm::assemble(&g).expect("encodes");
            }
        }
        assert!(moved > 0);
    }
}
