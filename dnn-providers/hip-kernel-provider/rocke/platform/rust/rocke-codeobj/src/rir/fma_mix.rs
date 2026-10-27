// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

//! f16-to-f32 conversions that several FMAs read, folded into all of them.
//!
//! `post::fuse_valu` folds a `v_cvt_f32_f16` with one use into the
//! `v_fma_f32` that reads it, as `v_fma_mix_f32`. A converted input that
//! several FMAs read (a depthwise input element times each filter row) keeps
//! its conversion there. LLVM folds the conversion into every one of them,
//! and the conversion goes; this does the same when every read of it is such
//! an FMA in its block. The conversion is exact, so every result is the same.
//!
//! Load sinking (`post::sink_load_uses`) delays the readers of a load below
//! later loads of its block. The conversion was the one reader of its load;
//! folded, every FMA is one, and the block becomes all its loads and then all
//! its FMAs, which measured slower (7x7 column and 5x5 depthwise
//! convolutions). So a value of the block is folded only when no load
//! follows its first FMA.
//!
//! A loop-carried f16 (an input row kept for the next rows, a filter weight)
//! is read by its conversion near the top of the loop, so the load of the
//! value that replaces it, copied into it at the latch, may issue anywhere
//! after that and share its register. Folded, the FMAs read it until later:
//! - when that load comes below all of them in program order, it writes the
//!   carried register itself, which keeps it there through scheduling, and
//!   the latch copy goes (else the scheduler may hoist it above them, and the
//!   latch copies wait for loads issued in the same iteration);
//! - when it comes above all of them (a prefetch at the top of the loop), the
//!   two keep separate registers and a copy, which finds the load long done,
//!   as LLVM shifts its windows;
//! - otherwise the conversion is not folded.

use super::mir::*;
use super::regalloc::Allocation;
use super::Unallocated;
use crate::target::Target;

/// `super::schedule_with_facts` for `func`, and again with its shared
/// conversions folded, taking the folded schedule unless it costs the kernel.
/// Folded, the f16 inputs live until their last FMA, and the FMAs neither
/// pair for dual issue nor take the short encoding. So:
/// - at full occupancy nothing is folded: other waves already hide the
///   conversions' latency, and the fold measured slower there (9x9 tiled) or
///   on a lone wave (5x5 spatial at the census geometry);
/// - the folded schedule is taken when it spills no more, is scheduled if the
///   unfolded one is (a 13x13 spatial kernel kept its occupancy only in
///   program order, and ran slower), and keeps nine tenths of the waves (a
///   7x7 dgrad lost one of seven, a 5x5 depthwise one of eight, and ran
///   slower; the 9x9 spatial lost one of ten and ran faster).
///
/// gfx1151 only, like `post::fuse_valu`. `ROCKE_DIRECT_DEBUG=nomix` (or
/// `nofuse`) does not fold.
pub(super) fn schedule_folding(
    func: &mut MFunc,
    lds_size: u32,
    wg: u32,
    spill: bool,
    threads: usize,
) -> Result<(Allocation, usize), Unallocated> {
    let off = func.target != Target::Gfx1151 || super::debug_flag("nofuse") || super::debug_flag("nomix");
    if off || !func.blocks.iter().any(|b| b.insts.iter().any(|i| i.op == Opc::VCvtF32F16)) {
        return super::schedule_with_facts(func, lds_size, wg, spill, threads);
    }
    let mut folded = func.clone();
    if fold_shared_conversions(&mut folded) == 0 {
        return super::schedule_with_facts(func, lds_size, wg, spill, threads);
    }
    let plain = super::schedule_with_facts(func, lds_size, wg, spill, threads);
    let Ok((a, window)) = &plain else {
        return plain;
    };
    let t = func.target.info();
    let spilled = |a: &Allocation| a.spill.lds_dwords + a.spill.scratch_dwords;
    let waves = |a: &Allocation| t.occupancy(a.num_vgprs.max(1), lds_size + a.spill.lds_bytes, wg);
    let (waves_plain, spilled_plain, scheduled_plain) = (waves(a), spilled(a), *window > 1);
    if waves_plain >= t.max_waves {
        return plain;
    }
    let with = super::schedule_with_facts(&mut folded, lds_size, wg, spill, threads);
    match &with {
        Ok((b, w)) if spilled(b) <= spilled_plain && (*w > 1 || !scheduled_plain) && waves(b) * 10 >= waves_plain * 9 => {
            *func = folded;
            with
        }
        _ => plain,
    }
}

/// Fewest FMAs that must read a conversion for it to be folded. With fewer,
/// the fold gave nothing at production size, and 3x3 depthwise kernels with
/// about four readers per conversion ran slower with it on a lone wave.
const MIN_READERS: u32 = 6;

/// Most instructions a function may have for the fold. A folded FMA takes 8
/// bytes where a `v_fmac_f32` takes 4, and without branch relaxation a loop
/// body past the 128 KiB reach of a branch cannot close (LLVM folds them all
/// and relaxes the branch): this keeps every body well within it.
const MAX_INSTS: usize = 8192;

/// Folds each `v_cvt_f32_f16` of a VGPR half in a loop block whose result is
/// read only by FMAs later in its block, `MIN_READERS` or more (`v_fma_f32`,
/// or `v_fma_mix_f32` with that source still f32), into all of them, and
/// removes it. Nothing between the conversion and a reader may write its
/// source or EXEC, and no reader may end up with a literal or more scalar
/// sources than VOP3P reads. A source copied in at the latch of its block's
/// loop takes over the copied value, and a value of the block needs no load
/// after its first FMA (see the module comment). Code outside loops keeps its
/// conversions: the one such kernel the fold reached (an 11x11 tiled
/// convolution with its rows unrolled) gave up its dual issue and ran slower.
/// Returns the conversions folded.
fn fold_shared_conversions(f: &mut MFunc) -> usize {
    if f.blocks.iter().map(|b| b.insts.len()).sum::<usize>() > MAX_INSTS {
        return 0;
    }
    let n = f.vregs.len();
    let (mut uses, mut defs) = (vec![0u32; n], vec![0u32; n]);
    // Block of each register's uses: the one block they are all in, or none.
    let mut use_block: Vec<Option<u32>> = vec![None; n];
    let mut scattered = vec![false; n];
    for (b, block) in f.blocks.iter().enumerate() {
        for inst in &block.insts {
            for o in &inst.uses {
                if let Opnd::R(v) = o {
                    let r = v.reg as usize;
                    uses[r] += 1;
                    scattered[r] |= use_block[r].is_some_and(|u| u != b as u32);
                    use_block[r] = Some(b as u32);
                }
            }
            for o in &inst.defs {
                if let Opnd::R(v) = o {
                    defs[v.reg as usize] += 1;
                }
            }
        }
    }
    let in_block = |r: u32, b: usize| !scattered[r as usize] && use_block[r as usize] == Some(b as u32);
    let vregs = &f.vregs;
    let mut folded = 0;
    for b in 0..f.blocks.len() {
        if f.blocks[b].loop_depth == 0 {
            continue;
        }
        let insts = &mut f.blocks[b].insts;
        let latch = loops_back(insts, b);
        let mut dead = vec![false; insts.len()];
        let mut takeovers: Vec<(VRef, VRef)> = Vec::new();
        let mut readers = Vec::new();
        for c in 0..insts.len() {
            let Some((t, src, opsel)) = conversion(&insts[c], vregs) else {
                continue;
            };
            if defs[t.reg as usize] != 1 || uses[t.reg as usize] < MIN_READERS {
                continue;
            }
            readers.clear();
            let mut found = 0;
            for (k, inst) in insts.iter().enumerate().skip(c + 1) {
                let reads = inst.uses.iter().filter(|o| matches!(o, Opnd::R(v) if v.reg == t.reg)).count() as u32;
                if reads > 0 {
                    if !foldable(inst, t, vregs) {
                        break;
                    }
                    readers.push(k);
                    found += reads;
                    if found == uses[t.reg as usize] {
                        break;
                    }
                }
                if inst.op.flags() & F_DEF_EXEC != 0 || writes_unit_of(&inst.defs, src) {
                    break;
                }
            }
            if found != uses[t.reg as usize] {
                continue;
            }
            let loads_follow = || insts[readers[0]..].iter().any(|i| i.op.flags() & (F_VMEM_LOAD | F_LDS) != 0 && !i.defs.is_empty());
            if defs[src.reg as usize] < 2 && loads_follow() {
                continue;
            }
            if defs[src.reg as usize] > 1 {
                if !latch {
                    continue;
                }
                let (first, last) = (readers[0], *readers.last().expect("at least two readers"));
                match replacement_def(insts, src, &defs, &uses) {
                    Some(def) if def < first => {}
                    Some(def) if def > last => {
                        let Some(takeover) = takeover(insts, src, last, &dead, &defs, &uses, &in_block, b, vregs) else {
                            continue;
                        };
                        if !takeovers.contains(&takeover) {
                            takeovers.push(takeover);
                        }
                    }
                    _ => continue,
                }
            }
            for &k in &readers {
                let inst = &mut insts[k];
                let mut mods = inst.mods;
                for slot in 0..inst.uses.len() {
                    if inst.uses[slot] == Opnd::R(t) {
                        inst.uses[slot] = Opnd::R(src);
                        mods |= M_MIX_F16_0 << slot | opsel << slot;
                    }
                }
                inst.op = Opc::VFmaMixF32;
                inst.mods = mods;
            }
            dead[c] = true;
            folded += 1;
        }
        for &(carried, copied) in &takeovers {
            for (k, inst) in insts.iter_mut().enumerate() {
                if is_copy(inst, carried, copied) {
                    dead[k] = true;
                    continue;
                }
                for o in inst.defs.iter_mut().chain(inst.uses.iter_mut()) {
                    if *o == Opnd::R(copied) {
                        *o = Opnd::R(carried);
                    }
                }
            }
        }
        if dead.iter().any(|&d| d) {
            let mut k = 0;
            insts.retain(|_| {
                k += 1;
                !dead[k - 1]
            });
        }
    }
    folded
}

/// Whether the block `b` ends by branching back to itself or an earlier
/// block: a loop latch.
fn loops_back(insts: &[MInst], b: usize) -> bool {
    insts.iter().rev().take(2).any(|i| i.is_branch() && (i.imm as usize) <= b)
}

/// Copies a latch value may pass through on its way to the carried register.
const COPY_CHAIN: usize = 4;

/// Position in the block of the instruction that computes the value the
/// latch copies into `carried`, following copies of registers that are read
/// by nothing else; `None` if that is not in this block.
fn replacement_def(insts: &[MInst], carried: VRef, defs: &[u32], uses: &[u32]) -> Option<usize> {
    let defined = |r: u32| insts.iter().position(|i| i.defs.iter().any(|o| matches!(o, Opnd::R(v) if v.reg == r)));
    let at = insts.iter().position(|i| i.op == Opc::Copy && i.defs[..] == [Opnd::R(carried)])?;
    let mut value = insts[at].uses[0].vref()?;
    for _ in 0..COPY_CHAIN {
        if defs[value.reg as usize] != 1 {
            return None;
        }
        let def = defined(value.reg)?;
        match (insts[def].op, &insts[def].uses[..]) {
            (Opc::Copy, &[Opnd::R(next)]) if uses[value.reg as usize] == 1 => value = next,
            _ => return Some(def),
        }
    }
    None
}

fn is_copy(inst: &MInst, dst: VRef, src: VRef) -> bool {
    inst.op == Opc::Copy && inst.defs[..] == [Opnd::R(dst)] && inst.uses[..] == [Opnd::R(src)]
}

/// For a loop-carried `carried` whose last reader after folding is `last`:
/// `(carried, copied)` when the latch copies `copied` into it and `copied`
/// can be defined as `carried` itself. That holds when both are one-dword
/// registers of one kind, `copied` is defined once, by a plain instruction
/// of this block below `last`, and read only in this block up to that copy,
/// and every read of `carried` is in this block above that definition.
#[allow(clippy::too_many_arguments)]
fn takeover(
    insts: &[MInst],
    carried: VRef,
    last: usize,
    dead: &[bool],
    defs: &[u32],
    uses: &[u32],
    in_block: &dyn Fn(u32, usize) -> bool,
    b: usize,
    vregs: &[VRegInfo],
) -> Option<(VRef, VRef)> {
    let whole = |v: VRef| v.off == 0 && v.n == 1 && vregs[v.reg as usize].size == 1;
    if !whole(carried) || defs[carried.reg as usize] != 2 || !in_block(carried.reg, b) {
        return None;
    }
    let (at, copied) = insts.iter().enumerate().find_map(|(k, i)| match (i.op, &i.defs[..], &i.uses[..]) {
        (Opc::Copy, &[Opnd::R(d)], &[Opnd::R(s)]) if d == carried => Some((k, s)),
        _ => None,
    })?;
    let same = |a: VRegInfo, b: VRegInfo| a.class == b.class && a.size == b.size && a.align == b.align && a.fixed.is_none() && b.fixed.is_none();
    if !whole(copied) || defs[copied.reg as usize] != 1 || !in_block(copied.reg, b) || !same(vregs[carried.reg as usize], vregs[copied.reg as usize]) {
        return None;
    }
    let def = insts[..at].iter().position(|i| i.defs.iter().any(|o| matches!(o, Opnd::R(v) if v.reg == copied.reg)))?;
    let plain = insts[def].defs[..] == [Opnd::R(copied)] && insts[def].op.flags() & (F_TIED | F_EARLY) == 0 && insts[def].op != Opc::Copy;
    if !plain || def <= last {
        return None;
    }
    let reads = |i: &MInst, r: u32| i.uses.iter().filter(|o| matches!(o, Opnd::R(v) if v.reg == r)).count() as u32;
    let copied_reads: u32 = insts[def..=at].iter().map(|i| reads(i, copied.reg)).sum();
    let carried_reads_above: u32 = insts[..def].iter().zip(dead).filter(|(_, &d)| !d).map(|(i, _)| reads(i, carried.reg)).sum();
    let carried_reads_dead: u32 = insts.iter().zip(dead).filter(|(_, &d)| d).map(|(i, _)| reads(i, carried.reg)).sum();
    (copied_reads == uses[copied.reg as usize] && carried_reads_above + carried_reads_dead == uses[carried.reg as usize])
        .then_some((carried, copied))
}

/// `(t, src, op_sel)` of `v_cvt_f32_f16 t, src` converting one half of a
/// VGPR (`op_sel` is `M_OPSEL_SRC0` for the high half) into a whole one-dword
/// virtual register.
fn conversion(inst: &MInst, vregs: &[VRegInfo]) -> Option<(VRef, VRef, u32)> {
    if inst.op != Opc::VCvtF32F16 || inst.mods & !M_OPSEL_SRC0 != 0 {
        return None;
    }
    let (&[Opnd::R(t)], &[Opnd::R(src)]) = (&inst.defs[..], &inst.uses[..]) else {
        return None;
    };
    let whole = |v: VRef| v.off == 0 && v.n == 1 && vregs[v.reg as usize].size == 1;
    (whole(t) && vregs[t.reg as usize].class == Class::V && vregs[src.reg as usize].class == Class::V && src.n == 1)
        .then_some((t, src, inst.mods & M_OPSEL_SRC0))
}

/// Whether the FMA `inst` reads `t` only as a whole f32 source that may take
/// an f16 half in its place: no other modifiers than earlier mix folds, no
/// physical register, no literal and at most two scalar sources.
fn foldable(inst: &MInst, t: VRef, vregs: &[VRegInfo]) -> bool {
    let mix_bits = (M_MIX_F16_0 | M_OPSEL_SRC0) * 0b111;
    let fma = match inst.op {
        Opc::VFmaF32 => inst.mods == 0,
        Opc::VFmaMixF32 => inst.mods & !mix_bits == 0,
        _ => false,
    };
    if !fma || inst.uses.len() != 3 || inst.defs.iter().chain(inst.uses.iter()).any(|o| matches!(o, Opnd::P(..))) {
        return false;
    }
    let mut scalars: Vec<Opnd> = Vec::new();
    for (slot, &o) in inst.uses.iter().enumerate() {
        match o {
            Opnd::R(v) if v.reg == t.reg => {
                if v != t || inst.mods & (M_MIX_F16_0 << slot) != 0 {
                    return false;
                }
            }
            Opnd::R(v) if vregs[v.reg as usize].class == Class::S => {
                if !scalars.contains(&o) {
                    scalars.push(o);
                }
            }
            Opnd::Imm(v) if !(super::isel::is_inline_int(v) || super::isel::is_inline_f32(v)) => return false,
            _ => {}
        }
    }
    scalars.len() <= 2
}

/// Whether `defs` write a dword of `v`.
fn writes_unit_of(defs: &[Opnd], v: VRef) -> bool {
    defs.iter().any(|o| match o {
        Opnd::R(d) => d.reg == v.reg && d.off < v.off + v.n && v.off < d.off + d.n,
        _ => false,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn r(reg: u32) -> Opnd {
        Opnd::R(VRef { reg, off: 0, n: 1 })
    }

    fn load(dst: u32, addr: u32) -> MInst {
        MInst::new(Opc::GLoadU16, [r(dst)], [r(addr)])
    }

    fn cvt(dst: u32, src: u32) -> MInst {
        MInst::new(Opc::VCvtF32F16, [r(dst)], [r(src)])
    }

    fn fma(dst: u32, a: u32, b: u32, c: u32) -> MInst {
        MInst::new(Opc::VFmaF32, [r(dst)], [r(a), r(b), r(c)])
    }

    fn copy(dst: u32, src: u32) -> MInst {
        MInst::new(Opc::Copy, [r(dst)], [r(src)])
    }

    /// One function of VGPRs 0..64: `entry` (inside a loop when alone)
    /// then, if given, a loop block that branches back to itself.
    fn func(entry: Vec<MInst>, body: Option<Vec<MInst>>) -> MFunc {
        let vregs = vec![VRegInfo { class: Class::V, size: 1, align: 1, fixed: None }; 64];
        let mut blocks = vec![MBlock { insts: entry, loop_depth: body.is_none() as u32 }];
        if let Some(mut body) = body {
            body.push(MInst::new(Opc::SCmpLtI32, [], [Opnd::Imm(0), Opnd::Imm(1)]));
            body.push(MInst::new(Opc::SCbranchScc1, [], []).with_imm(1));
            blocks.push(MBlock { insts: body, loop_depth: 1 });
        }
        MFunc { blocks, vregs, ..Default::default() }
    }

    fn count(f: &MFunc, op: Opc) -> usize {
        f.blocks.iter().flat_map(|b| b.insts.iter()).filter(|i| i.op == op).count()
    }

    /// `n` FMAs reading `t` as their second source, into r40.. with
    /// multiplicands r3 and addends r10...
    fn fmas(t: u32, n: u32) -> Vec<MInst> {
        (0..n).map(|k| fma(40 + k, 3, t, 10 + k)).collect()
    }

    fn block(parts: &[Vec<MInst>]) -> Vec<MInst> {
        parts.concat()
    }

    #[test]
    fn a_conversion_enough_fmas_read_folds_into_all_of_them() {
        let mut entry = block(&[vec![load(1, 0), cvt(2, 1)], fmas(2, MIN_READERS - 1)]);
        entry.push(fma(50, 2, 3, 20));
        let mut f = func(entry, None);
        assert_eq!(fold_shared_conversions(&mut f), 1);
        let insts = &f.blocks[0].insts;
        assert_eq!(count(&f, Opc::VCvtF32F16), 0);
        assert_eq!(count(&f, Opc::VFmaMixF32), MIN_READERS as usize);
        // The f16 source takes the conversion's place, marked f16 in its slot.
        assert_eq!(insts[1].uses[..], [r(3), r(1), r(10)]);
        assert_eq!(insts[1].mods, M_MIX_F16_0 << 1);
        let last = MIN_READERS as usize;
        assert_eq!(insts[last].uses[..], [r(1), r(3), r(20)]);
        assert_eq!(insts[last].mods, M_MIX_F16_0);
    }

    #[test]
    fn conversions_with_other_readers_or_too_few_stay() {
        // An add reads the conversion too, or one FMA too few does.
        let add = MInst::new(Opc::VAddF32, [r(30)], [r(2), r(3)]);
        let shared = block(&[vec![load(1, 0), cvt(2, 1)], fmas(2, MIN_READERS), vec![add]]);
        let few = block(&[vec![load(1, 0), cvt(2, 1)], fmas(2, MIN_READERS - 1)]);
        for entry in [shared, few] {
            let mut f = func(entry, None);
            assert_eq!(fold_shared_conversions(&mut f), 0);
            assert_eq!(count(&f, Opc::VCvtF32F16), 1);
        }
    }

    #[test]
    fn a_load_after_the_first_fma_keeps_the_conversion() {
        // Load sinking would push the FMAs below the second load.
        let mut f = func(block(&[vec![load(1, 0), cvt(2, 1)], fmas(2, 1), vec![load(7, 8)], fmas(2, MIN_READERS)[1..].to_vec()]), None);
        assert_eq!(fold_shared_conversions(&mut f), 0);
        let mut f = func(block(&[vec![load(1, 0), load(7, 8), cvt(2, 1)], fmas(2, MIN_READERS)]), None);
        assert_eq!(fold_shared_conversions(&mut f), 1);
    }

    #[test]
    fn a_carried_value_replaced_below_its_fmas_takes_over_the_load() {
        // r1 is carried: set before the loop and copied from the load r7 at its latch.
        let body = block(&[vec![cvt(2, 1)], fmas(2, MIN_READERS), vec![load(7, 8), copy(1, 7)]]);
        let mut f = func(vec![copy(1, 20)], Some(body));
        assert_eq!(fold_shared_conversions(&mut f), 1);
        let insts = &f.blocks[1].insts;
        let at = MIN_READERS as usize;
        assert_eq!(count(&f, Opc::VFmaMixF32), at);
        assert_eq!(insts[at].op, Opc::GLoadU16);
        assert_eq!(insts[at].defs[..], [r(1)], "the load writes the carried register");
        assert!(!insts.iter().any(|i| i.op == Opc::Copy), "the latch copy is gone");
    }

    #[test]
    fn a_prefetched_carried_value_keeps_its_copy() {
        let body = block(&[vec![load(7, 8), cvt(2, 1)], fmas(2, MIN_READERS), vec![copy(1, 7)]]);
        let mut f = func(vec![copy(1, 20)], Some(body));
        assert_eq!(fold_shared_conversions(&mut f), 1);
        assert_eq!(f.blocks[1].insts[0].defs[..], [r(7)]);
        assert!(f.blocks[1].insts.iter().any(|i| i.op == Opc::Copy && i.defs[..] == [r(1)]));
    }

    #[test]
    fn a_carried_value_replaced_between_its_fmas_keeps_the_conversion() {
        let body = block(&[vec![cvt(2, 1)], fmas(2, 2), vec![load(7, 8)], fmas(2, MIN_READERS)[2..].to_vec(), vec![copy(1, 7)]]);
        let mut f = func(vec![copy(1, 20)], Some(body));
        assert_eq!(fold_shared_conversions(&mut f), 0);
        assert_eq!(count(&f, Opc::VCvtF32F16), 1);
    }

    #[test]
    fn a_carried_value_read_after_the_loop_keeps_the_conversion() {
        // The replacing load cannot take over r1 while r1 is read in another block.
        let body = block(&[vec![cvt(2, 1)], fmas(2, MIN_READERS), vec![load(7, 8), copy(1, 7)]]);
        let mut f = func(vec![copy(1, 20)], Some(body));
        f.blocks.push(MBlock { insts: vec![MInst::new(Opc::VAddF32, [r(30)], [r(1), r(1)])], loop_depth: 0 });
        assert_eq!(fold_shared_conversions(&mut f), 0);
    }

    #[test]
    fn straight_line_code_keeps_its_conversions() {
        let mut f = func(block(&[vec![load(1, 0), cvt(2, 1)], fmas(2, MIN_READERS)]), None);
        f.blocks[0].loop_depth = 0;
        assert_eq!(fold_shared_conversions(&mut f), 0);
    }

    /// A kernel whose loop loads an f16 input and adds it times seven weights
    /// to seven sums, with `live` f16 values loaded ahead of the loop through
    /// one buffer descriptor and summed after it: past 96 VGPRs, so below
    /// full occupancy, for `live` of 100.
    fn shared_input_kernel(live: usize) -> String {
        let mut t = String::from(
            "ckdsl.ir v1\nkernel @shared {\n  attrs { max_workgroup_size = i:64 }\n  params {\n    \
             %X : ptr<f16,global> { align = i:2, noalias = b:true, readonly = b:true }\n    \
             %W : ptr<f32,global> { align = i:4, noalias = b:true, readonly = b:true }\n    \
             %H : ptr<f16,global> { align = i:16, noalias = b:true, readonly = b:true }\n    \
             %Y : ptr<f32,global> { align = i:4, noalias = b:true }\n    %H_bytes : i32\n    %n : i32\n  }\n  region @entry {\n    \
             %tid : i32 = gpu.thread_id (  ) { axis = s:\"x\" }\n    \
             %zero : i32 = arith.constant (  ) { ity = s:\"i32\", value = i:0 }\n    \
             %one : i32 = arith.constant (  ) { ity = s:\"i32\", value = i:1 }\n    \
             %two : i32 = arith.constant (  ) { ity = s:\"i32\", value = i:2 }\n    \
             %row : i32 = arith.constant (  ) { ity = s:\"i32\", value = i:64 }\n    \
             %f0 : f32 = arith.constant (  ) { ity = s:\"f32\", value = f:0.0 }\n    \
             %rsrc : vec<i32x4> = tile.buffer_rsrc ( %H, %H_bytes )\n    \
             %byte : i32 = arith.mul ( %tid, %two )\n",
        );
        for i in 0..7 {
            t += &format!("    %c{i} : i32 = arith.constant (  ) {{ ity = s:\"i32\", value = i:{} }}\n", 64 * i);
            t += &format!("    %p{i} : i32 = arith.add ( %tid, %c{i} )\n");
            t += &format!("    %w{i} : f32 = memref.global_load_typed ( %W, %p{i} ) {{ align = i:4, elem_type = s:\"f32\" }}\n");
        }
        for i in 0..live {
            t += &format!("    %k{i} : i32 = arith.constant (  ) {{ ity = s:\"i32\", value = i:{} }}\n", 128 * i);
            t += &format!("    %o{i} : i32 = arith.add ( %byte, %k{i} )\n");
            t += &format!("    %h{i} : f16 = tile.buffer_load_f16 ( %rsrc, %o{i}, %zero )\n");
        }
        let results: Vec<String> = (0..7).map(|i| format!("%r{i} : f32")).collect();
        let args: Vec<String> = (0..7).map(|i| format!("{{ name = s:\"%acc{i}\", type = s:\"f32\" }}")).collect();
        t += &format!(
            "    {} = scf.for ( %zero, %n, %one{} ) {{ elide_trailing_barrier = b:false, iter_args = l:[ {} ], iv = s:\"%i\", \
             iv_type = s:\"i32\", num_iter_args = i:7, unroll = b:false }}\n      region @body {{\n",
            results.join(", "),
            ", %f0".repeat(7),
            args.join(", ")
        );
        t += "        %off : i32 = arith.mul ( %i, %row )\n        %at : i32 = arith.add ( %tid, %off )\n        \
              %x : f16 = memref.global_load_typed ( %X, %at ) { align = i:2, elem_type = s:\"f16\" }\n        \
              %xf : f32 = arith.cast_to_f32 ( %x )\n";
        for i in 0..7 {
            t += &format!("        %a{i} : f32 = arith.fma ( %w{i}, %xf, %acc{i} )\n");
        }
        let yields: Vec<String> = (0..7).map(|i| format!("%a{i}")).collect();
        t += &format!("        scf.yield ( {} ) {{ num = i:7 }}\n      }}\n", yields.join(", "));
        for i in 0..7 {
            t += &format!("    memref.global_store_typed ( %Y, %p{i}, %r{i} ) {{ align = i:4, elem_type = s:\"f32\" }}\n");
        }
        if live > 0 {
            let mut sum = String::from("%r0");
            for i in 0..live {
                t += &format!("    %hf{i} : f32 = arith.cast_to_f32 ( %h{i} )\n");
                t += &format!("    %s{i} : f32 = arith.fadd ( {sum}, %hf{i} )\n");
                sum = format!("%s{i}");
            }
            t += &format!("    memref.global_store_typed ( %Y, %tid, {sum} ) {{ align = i:4, elem_type = s:\"f32\" }}\n");
        }
        t + "    cf.return (  )\n  }\n}\n"
    }

    #[test]
    fn a_compiled_loop_reads_the_shared_f16_input_in_each_fma_below_full_occupancy() {
        let compiled = super::super::compile(&shared_input_kernel(100)).unwrap_or_else(|e| panic!("{e}"));
        assert!(compiled.stats.waves < 16, "{} waves", compiled.stats.waves);
        let listing = compiled.listing();
        let mixes = listing.matches("v_fma_mix_f32").count();
        assert!(mixes >= 7 && mixes % 7 == 0, "{mixes} folded FMAs\n{listing}");
        // Only the conversions of the summed values, each read by one add, stay.
        assert_eq!(listing.matches("v_cvt_f32_f16").count(), 100, "{listing}");
    }

    #[test]
    fn a_compiled_loop_at_full_occupancy_keeps_its_conversion() {
        let compiled = super::super::compile(&shared_input_kernel(0)).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(compiled.stats.waves, 16);
        let listing = compiled.listing();
        assert!(listing.contains("v_cvt_f32_f16"), "{listing}");
        assert!(!listing.contains("v_fma_mix_f32"), "{listing}");
    }
}
