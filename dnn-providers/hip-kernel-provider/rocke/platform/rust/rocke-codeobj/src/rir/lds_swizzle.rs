// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

//! Automatic LDS layouts. For every allocation the pass evaluates the byte
//! address of each access per (wave, lane), prices all accesses with the
//! gfx1151 bank model, and when they conflict tries the row paddings and XOR
//! swizzles that the conflicting address strides suggest. A padding that
//! leaves fewer workgroups per WGP has its cycles scaled by the loss. The
//! cheapest layout that clearly beats the current one is applied to every
//! access of the allocation. It runs on the optimized IR, where each access
//! is `(smem, addr) {offset}`. `ROCKE_DIRECT_DEBUG=noswizzle` turns it off.
//!
//! Bank model (RDNA3 ISA guide, confirmed by microbenchmarks on gfx1151):
//! each 64 KiB half of a WGP's LDS has 32 banks of 4 bytes. A wave32 access of
//! up to 4 bytes per lane is served in one phase of 32 lanes, 8 bytes in two
//! phases of 16 lanes, 12 or 16 bytes in four phases of 8 lanes. A phase takes
//! as many cycles as the most distinct dwords any one bank serves: lanes that
//! read one dword share it, and sub-dword stores to one dword merge. Under
//! load the LDS of a WGP serves 128 bytes per cycle, so these cycles are what
//! an LDS-bound kernel pays.

use std::rc::Rc;

use super::ast::{Attr, Kernel, Op, OpKind, Operands, Region, Scalar, Text, Type, ValueId};
use super::fx::{FxHashMap, FxHashSet};
use super::opt::{Facts, Konst, Range};
use crate::target::GFX1151;

pub const WAVE: usize = 32;
const BANKS: i64 = 32;
/// Trip count assumed for loops whose bounds are not constants.
const DEFAULT_TRIPS: f64 = 16.0;
/// Values of an address's unknown part that pricing samples when that part
/// is not a multiple of the period the layout depends on.
const MAX_SAMPLES: i64 = 4;
/// Nodes an address expression may have before it counts as unanalyzable.
const MAX_EXPR_NODES: usize = 4096;
/// Largest magnitude a constant or table entry may reach.
const VALUE_LIMIT: i64 = 1 << 31;
/// Symbols of loop trip indices; other symbols are value numbers.
const TRIP_SYMBOL: u32 = 1 << 31;

/// An allocation of the pool: (allocation, byte offset, bytes).
type Slot = (ValueId, u32, u32);
/// 16-bit loads that may still join a run, per (smem, addr): (offset, access).
type OpenRuns = Vec<((ValueId, ValueId), Vec<(i64, usize)>)>;
/// (pattern, width, offset modulo the period, wave) -> (cycles, conflict stride).
type CycleMemo = FxHashMap<(u32, u32, i64, u32), (u32, Option<u32>)>;
/// Per access: allocation, bytes, and its address per slot with whether it
/// has an unknown part.
type AccessAt = (ValueId, u32, Option<(Vec<i64>, bool)>);
/// Conflict strides (as powers of two) whose layouts are tried.
const STRIDES_TRIED: usize = 2;
/// A further stride is tried only while the best layout so far keeps this
/// share of the conflict cycles.
const REMAINING_EXCESS: f64 = 0.2;
/// Least share of the kernel's conflict-free LDS cycles that an
/// allocation's conflicts must cost for layouts to be priced, and that a new
/// layout must save to be applied.
const MIN_POTENTIAL: f64 = 0.10;
const MIN_GAIN: f64 = 0.03;
/// Most bases offset by a residue (see `Fold`) a layout may add to one
/// allocation: each holds a register while its accesses are in flight, and
/// register pressure is unknown here.
const MAX_RESIDUE_BASES: usize = 16;
/// Least share of the kernel's workgroups per WGP (as LDS allows them) that
/// its layouts must keep: the pricing of a lost workgroup is only measured
/// for small losses.
const MIN_GROUPS_KEPT: f64 = 0.85;

/// Lanes processed together for a per-lane access of `width` bytes.
pub fn phase_lanes(width: u32) -> usize {
    match width {
        0..=4 => 32,
        5..=8 => 16,
        _ => 8,
    }
}

/// Cycles of a conflict-free access of `width` bytes per lane.
pub fn ideal_cycles(width: u32) -> u32 {
    (WAVE / phase_lanes(width)) as u32
}

/// LDS cycles of one wave32 access of `width` bytes per lane at the given
/// byte addresses, all lanes active, and the smallest power of two (as an
/// exponent) dividing the distance between two distinct dwords that share a
/// bank in some phase (`None` without conflicts).
pub fn access_cycles(addrs: &[i64; WAVE], width: u32) -> (u32, Option<u32>) {
    let lanes = phase_lanes(width);
    let span = width.max(1) as i64;
    let mut total = 0;
    let mut stride: Option<u32> = None;
    // Per phase: the distinct 128-byte rows met, each with a mask of its
    // banks, and how many distinct dwords each bank serves.
    let mut rows = [(0i64, 0u32); 64];
    for phase in addrs.chunks(lanes) {
        let mut nrows = 0;
        let mut count = [0u32; BANKS as usize];
        for &a in phase {
            let (mut d, last) = (a.div_euclid(4), (a + span - 1).div_euclid(4));
            while d <= last {
                // The lane's dwords in this row, as a mask of banks.
                let row = d >> 5;
                let end = last.min(row * 32 + 31);
                let (lo, hi) = ((d & 31) as u32, (end & 31) as u32);
                let mask = (u32::MAX >> (31 - hi)) & (u32::MAX << lo);
                d = end + 1;
                let i = match rows[..nrows].iter().position(|r| r.0 == row) {
                    Some(i) => i,
                    None if nrows < rows.len() => {
                        rows[nrows] = (row, 0);
                        nrows += 1;
                        nrows - 1
                    }
                    None => continue,
                };
                let mut new = mask & !rows[i].1;
                rows[i].1 |= new;
                while new != 0 {
                    let bank = new.trailing_zeros();
                    new &= new - 1;
                    count[bank as usize] += 1;
                    if count[bank as usize] > 1 {
                        // Distance in rows to another row holding this bank.
                        let v = rows[..nrows]
                            .iter()
                            .filter(|r| r.0 != row && r.1 & (1 << bank) != 0)
                            .map(|r| (r.0 - row).unsigned_abs().trailing_zeros() + 7)
                            .min();
                        if let Some(v) = v {
                            stride = Some(stride.map_or(v, |s| s.min(v)));
                        }
                    }
                }
            }
        }
        total += count.iter().copied().max().unwrap_or(0).max(1);
    }
    (total, stride)
}

// ---------------------------------------------------------------------------
// Layouts
// ---------------------------------------------------------------------------

/// An address map applied to every access of one allocation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Layout {
    Identity,
    /// `2^pad` bytes after every `2^row` bytes: `a + (a >> row << pad)`.
    Pad { row: u32, pad: u32 },
    /// Address bits `[low, low + bits)` XORed with bits `[key, key + bits)`,
    /// `key >= low + bits`: `a ^ (((a >> key) & (2^bits - 1)) << low)`.
    Xor { low: u32, bits: u32, key: u32 },
}

impl Layout {
    pub fn apply(self, a: i64) -> i64 {
        match self {
            Layout::Identity => a,
            Layout::Pad { row, pad } => a + ((a >> row) << pad),
            Layout::Xor { low, bits, key } => a ^ (((a >> key) & ((1 << bits) - 1)) << low),
        }
    }

    /// The layout maps `a + k * period` to its image of `a` plus `k` times
    /// `apply(period)`, a multiple of 4, so bank conflicts repeat with it.
    pub fn period(self) -> i64 {
        match self {
            Layout::Identity => 4,
            Layout::Pad { row, .. } => 1 << row,
            Layout::Xor { bits, key, .. } => 1 << (key + bits),
        }
    }

    /// Bytes `[a, a + chunk)` with `a` a multiple of `chunk` stay contiguous.
    fn chunk(self) -> i64 {
        match self {
            Layout::Identity => 1 << 30,
            Layout::Pad { row, .. } => 1 << row,
            Layout::Xor { low, .. } => 1 << low,
        }
    }

    /// Whether `[a, a + bytes)` maps to `[apply(a), apply(a) + bytes)`.
    fn contiguous(self, a: i64, bytes: i64) -> bool {
        let c = self.chunk();
        bytes <= 1 || a.div_euclid(c) == (a + bytes - 1).div_euclid(c)
    }

    /// Largest power of two the layout keeps every address congruent modulo.
    fn keeps_align(self) -> i64 {
        match self {
            Layout::Identity => 1 << 30,
            Layout::Pad { pad, .. } => 1 << pad,
            Layout::Xor { low, .. } => 1 << low,
        }
    }

    /// Bytes the allocation needs for `size` bytes of data.
    pub fn size(self, size: u32) -> u32 {
        match self {
            Layout::Pad { .. } if size > 0 => (self.apply(size as i64 - 1) + 1) as u32,
            Layout::Xor { low, bits, .. } => size.next_multiple_of(1 << (low + bits)),
            _ => size,
        }
    }

    pub fn describe(self) -> String {
        match self {
            Layout::Identity => "none".into(),
            Layout::Pad { row, pad } => format!("pad {}B per {}B row", 1 << pad, 1 << row),
            Layout::Xor { low, bits, key } => format!("xor bits {}..{} with bits {}..{}", low, low + bits, key, key + bits),
        }
    }
}

/// Layouts worth trying against `degree`-way conflicts between dwords
/// `2^stride` bytes apart, for an allocation whose machine accesses are at
/// most `max_unit` bytes wide (and must stay aligned to it) and whose IR
/// accesses are at most `max_bytes` long (and must stay contiguous). A
/// padding of `128 / degree` bytes per conflict stride spreads the `degree`
/// rows over the banks, as XORing `log2(degree)` bits below bit 7 with the
/// stride's bits does; paddings grow the allocation by at most a quarter.
fn candidates(stride: u32, degree: u32, max_unit: u32, max_bytes: u32, out: &mut Vec<Layout>) {
    let unit = max_unit.max(4).trailing_zeros();
    let bits = degree.max(2).next_power_of_two().trailing_zeros().min(7 - unit.min(6));
    let row = stride.clamp(5, 12);
    if (1u32 << row) >= max_bytes {
        for pad in [7 - bits, unit] {
            let layout = Layout::Pad { row, pad };
            if (unit..=6).contains(&pad) && pad + 2 <= row && !out.contains(&layout) {
                out.push(layout);
            }
        }
    }
    let low = 7 - bits;
    let key = stride.max(7);
    let layout = Layout::Xor { low, bits, key };
    if bits > 0 && low >= unit && max_bytes <= 1 << low && key + bits <= 16 && !out.contains(&layout) {
        out.push(layout);
    }
}

// ---------------------------------------------------------------------------
// Address values
// ---------------------------------------------------------------------------

/// A value as `konst + lanes[wave * 32 + lane] + sum(coef * symbol)`, where a
/// symbol is unknown but the same in every lane of a wave.
#[derive(Clone, Debug)]
struct Lin {
    konst: i64,
    lanes: Option<Rc<[i64]>>,
    terms: Vec<(u32, i64)>,
}

impl Lin {
    fn konst(k: i64) -> Lin {
        Lin { konst: k, lanes: None, terms: Vec::new() }
    }

    fn sym(s: u32) -> Lin {
        Lin { konst: 0, lanes: None, terms: vec![(s, 1)] }
    }

    fn as_const(&self) -> Option<i64> {
        (self.lanes.is_none() && self.terms.is_empty()).then_some(self.konst)
    }

    /// The value in every slot, when it has no symbols.
    fn exact(&self, slots: usize) -> Option<Vec<i64>> {
        self.terms.is_empty().then(|| (0..slots).map(|i| self.at(i)).collect())
    }

    /// Largest power of two dividing every symbol coefficient.
    fn align(&self) -> i64 {
        1i64 << self.terms.iter().map(|&(_, c)| c.trailing_zeros()).min().unwrap_or(40).min(40)
    }

    /// The value in one slot with every symbol zero.
    fn at(&self, slot: usize) -> i64 {
        self.konst + self.lanes.as_ref().map_or(0, |l| l[slot])
    }

    /// Whether the value is the same in every lane of each wave.
    fn wave_uniform(&self) -> bool {
        self.lanes.as_ref().is_none_or(|l| l.chunks(WAVE).all(|w| w.iter().all(|&x| x == w[0])))
    }

    fn checked(self) -> Option<Lin> {
        let ok = self.konst.abs() < VALUE_LIMIT
            && self.lanes.as_ref().is_none_or(|l| l.iter().all(|x| x.abs() < VALUE_LIMIT))
            && self.terms.iter().all(|&(_, c)| c.abs() < (1 << 40));
        ok.then_some(self)
    }

    fn add(&self, other: &Lin, sign: i64) -> Lin {
        let lanes = match (&self.lanes, &other.lanes) {
            (None, None) => None,
            (Some(a), None) => Some(a.clone()),
            (None, Some(b)) => Some(b.iter().map(|&x| sign * x).collect()),
            (Some(a), Some(b)) => Some(a.iter().zip(b.iter()).map(|(&x, &y)| x + sign * y).collect()),
        };
        let mut terms = self.terms.clone();
        for &(s, c) in &other.terms {
            match terms.binary_search_by_key(&s, |&(t, _)| t) {
                Ok(i) => terms[i].1 += sign * c,
                Err(i) => terms.insert(i, (s, sign * c)),
            }
        }
        terms.retain(|&(_, c)| c != 0);
        Lin { konst: self.konst + sign * other.konst, lanes, terms }
    }

    fn scale(&self, k: i64) -> Lin {
        if k == 0 {
            return Lin::konst(0);
        }
        Lin {
            konst: self.konst * k,
            lanes: self.lanes.as_ref().map(|l| l.iter().map(|&x| x * k).collect()),
            terms: self.terms.iter().map(|&(s, c)| (s, c * k)).collect(),
        }
    }

    /// `f` of the symbol-free part, slot by slot, with every coefficient
    /// mapped by `g`. Callers make sure the symbols commute with `f`.
    fn map_exact(&self, f: impl Fn(i64) -> i64, g: impl Fn(i64) -> i64) -> Lin {
        let mut terms: Vec<(u32, i64)> = self.terms.iter().map(|&(s, c)| (s, g(c))).collect();
        terms.retain(|&(_, c)| c != 0);
        match &self.lanes {
            Some(l) => Lin { konst: 0, lanes: Some(l.iter().map(|&x| f(x + self.konst)).collect()), terms },
            None => Lin { konst: f(self.konst), lanes: None, terms },
        }
    }

    /// Whether the symbol-free part is non-negative in every slot.
    fn exact_nonneg(&self) -> bool {
        match &self.lanes {
            Some(l) => l.iter().all(|&t| t + self.konst >= 0),
            None => self.konst >= 0,
        }
    }
}

// ---------------------------------------------------------------------------
// Access collection
// ---------------------------------------------------------------------------

/// A loop block argument.
#[derive(Clone, Copy)]
enum Def {
    /// Induction variable, with constant start and step if known.
    Iv(Option<(i64, i64)>),
    IterArg,
}

/// One LDS load or store of the optimized IR, numbered in preorder.
#[derive(Clone, Debug)]
pub struct Access {
    pub alloc: ValueId,
    pub addr: ValueId,
    pub offset: i64,
    pub bytes: u32,
    pub align: u32,
    pub weight: f64,
    /// The loaded value of a 16-bit load (isel may fold it into a run when
    /// that value has one use).
    half: Option<ValueId>,
}

/// Machine accesses `(offset, width)` isel makes of an access of `bytes`
/// bytes at `offset` with alignment `align` (as `isel::lload`/`lstore`).
fn units(offset: i64, bytes: u32, align: u32, out: &mut Vec<(i64, u32)>) {
    if bytes <= 2 {
        out.push((offset, bytes.max(1)));
        return;
    }
    let dwords = bytes.div_ceil(4);
    let unit = match (dwords, align) {
        (4, a) if a >= 16 => 4,
        (2, a) | (4, a) if a >= 8 => 2,
        _ => 1,
    };
    let mut done = 0;
    while done < dwords {
        let n = unit.min(dwords - done);
        out.push((offset + 4 * done as i64, 4 * n));
        done += n;
    }
}

/// Machine accesses of a folded run of `bytes` bytes of 16-bit loads whose
/// first byte has alignment `align`: b128, b64 and b32 pieces (as
/// `isel::lds_chain` and the vector insert that consumes it).
fn run_units(offset: i64, bytes: u32, align: i64, out: &mut Vec<(i64, u32)>) {
    let mut done = 0u32;
    let dwords = bytes / 4;
    while done < dwords {
        let left = dwords - done;
        let n = match () {
            _ if align >= 16 && left >= 4 => 4,
            _ if align >= 8 && left >= 2 => 2,
            _ => 1,
        };
        out.push((offset + 4 * done as i64, 4 * n));
        done += n;
    }
}

/// Machine accesses of a dword quad: one b128 when its start stays 16-byte
/// aligned and its bytes contiguous, else its four b32 loads (as
/// `isel::find_dword_quads` merges them or not).
fn quad_units(offset: i64, aligned: bool, out: &mut Vec<(i64, u32)>) {
    if aligned {
        out.push((offset, 16));
    } else {
        out.extend((0..4).map(|k| (offset + 4 * k, 4)));
    }
}

struct Collector<'a> {
    kernel: &'a Kernel,
    facts: &'a Facts,
    /// The op defining each value (none for block arguments).
    defs: Vec<Option<&'a Op>>,
    /// Loop block arguments.
    args: FxHashMap<ValueId, Def>,
    /// Uses of each 16-bit load's value.
    half_uses: FxHashMap<ValueId, u32>,
    accesses: Vec<Access>,
    /// Allocations used other than as the smem operand of LLoad/LStore.
    escaped: FxHashSet<ValueId>,
    /// Weight of the region each value is defined in.
    def_weight: Vec<f32>,
    /// Runs of 16-bit loads of one address at consecutive offsets, with no
    /// LDS write or barrier between them, as access numbers.
    runs: Vec<Vec<usize>>,
    /// Every allocation with its size in bytes.
    allocs: Vec<(ValueId, u32)>,
    /// `runs` from this one on are dword quads (`isel::find_dword_quads`).
    quads_from: usize,
    quads: Vec<Vec<usize>>,
}

/// Ops past which the loads before and after cannot be one folded run.
fn breaks_runs(kind: OpKind) -> bool {
    matches!(
        kind,
        OpKind::LStore
            | OpKind::LAtomicAdd
            | OpKind::Sync
            | OpKind::SyncLdsOnly
            | OpKind::BarrierBare
            | OpKind::Fence
            | OpKind::InlineAsmIdentity
            | OpKind::SmemStore
            | OpKind::SmemStoreVec
            | OpKind::LdsAtomicAdd
    )
}

impl<'a> Collector<'a> {
    fn new(kernel: &'a Kernel, facts: &'a Facts) -> Self {
        let n = kernel.values.len();
        let mut c = Collector {
            kernel,
            facts,
            defs: vec![None; n],
            args: FxHashMap::default(),
            half_uses: FxHashMap::default(),
            accesses: Vec::new(),
            escaped: FxHashSet::default(),
            def_weight: vec![1.0; n],
            runs: Vec::new(),
            allocs: Vec::new(),
            quads_from: 0,
            quads: Vec::new(),
        };
        c.walk(&kernel.body, 1.0);
        // A folded load's value has one use; split runs at the others.
        let raw = std::mem::take(&mut c.runs);
        for run in raw {
            let mut cur: Vec<usize> = Vec::new();
            for m in run {
                let a = &c.accesses[m];
                let single = a.half.is_some_and(|r| c.half_uses.get(&r) == Some(&1));
                let next = cur.last().is_none_or(|&p| c.accesses[p].offset + 2 == a.offset);
                if single && next {
                    cur.push(m);
                } else {
                    if cur.len() >= 2 {
                        c.runs.push(std::mem::take(&mut cur));
                    }
                    cur.clear();
                    if single {
                        cur.push(m);
                    }
                }
            }
            if cur.len() >= 2 {
                c.runs.push(cur);
            }
        }
        c.quads_from = c.runs.len();
        let quads = std::mem::take(&mut c.quads);
        c.runs.extend(quads);
        c
    }

    /// Bytes per load of run `r`.
    fn run_elem(&self, r: usize) -> u32 {
        if r >= self.quads_from {
            4
        } else {
            2
        }
    }

    /// Quads of the dword loads in `open` (per address, in program order):
    /// offsets `o` to `o + 12` from a 16-byte multiple `o`, in that order.
    fn close_quads(&mut self, open: &mut OpenRuns) {
        for (_, loads) in open.drain(..) {
            let at = |o: i64| {
                let mut hits = loads.iter().filter(|l| l.0 == o);
                match (hits.next(), hits.next()) {
                    (Some(&(_, a)), None) => Some(a),
                    _ => None,
                }
            };
            let mut used = FxHashSet::default();
            for &(o, first) in &loads {
                if o % 16 != 0 || used.contains(&first) {
                    continue;
                }
                let quad: Option<Vec<usize>> = (0..4).map(|k| at(o + 4 * k)).collect();
                match quad {
                    Some(q) if q.windows(2).all(|w| w[0] < w[1]) && q.iter().all(|a| !used.contains(a)) => {
                        used.extend(q.iter().copied());
                        self.quads.push(q);
                    }
                    _ => {}
                }
            }
        }
    }

    fn konst(&self, op: &Op, i: usize) -> Option<i64> {
        op.operands.get(i).and_then(|&v| self.facts.int(v))
    }

    fn walk(&mut self, region: &'a Region, weight: f64) {
        // Open 16-bit loads per (smem, addr): (offset, access number).
        let mut open: OpenRuns = Vec::new();
        let mut open4: OpenRuns = Vec::new();
        for op in &region.ops {
            for &v in &op.results {
                self.defs[v.0 as usize] = Some(op);
                self.def_weight[v.0 as usize] = weight as f32;
            }
            // An allocation precedes its uses.
            let access = matches!(op.kind, OpKind::LLoad | OpKind::LStore);
            for (i, &v) in op.operands.iter().enumerate() {
                if !self.half_uses.is_empty() {
                    if let Some(n) = self.half_uses.get_mut(&v) {
                        *n += 1;
                    }
                }
                if !(access && i == 0) && self.allocs.iter().any(|a| a.0 == v) {
                    self.escaped.insert(v);
                }
            }
            if breaks_runs(op.kind) || !op.regions.is_empty() {
                self.close_runs(&mut open);
                self.close_quads(&mut open4);
            }
            match op.kind {
                OpKind::SmemAlloc => {
                    if let Type::Smem(elem, shape) = self.kernel.ty(op.results[0]) {
                        let bytes = elem.bytes() as u64 * shape.iter().map(|&d| d as u64).product::<u64>();
                        self.allocs.push((op.results[0], bytes.min(u32::MAX as u64) as u32));
                    }
                }
                OpKind::LLoad | OpKind::LStore => {
                    let store = op.kind == OpKind::LStore;
                    let ty = if store { self.kernel.ty(op.operands[2]) } else { self.kernel.ty(op.results[0]) };
                    let bytes = match ty {
                        Type::Vec(e, n) => e.bytes() * n,
                        other => other.elem().bytes(),
                    };
                    let half = !store && bytes == 2 && ty.elem().is_half();
                    let index = self.accesses.len();
                    let offset = op.int_attr("offset").unwrap_or(0);
                    if half {
                        self.half_uses.insert(op.results[0], 0);
                        let key = (op.operands[0], op.operands[1]);
                        match open.iter_mut().find(|(k, _)| *k == key) {
                            Some((_, loads)) => loads.push((offset, index)),
                            None => open.push((key, vec![(offset, index)])),
                        }
                    }
                    let dword = matches!(ty, Type::Scalar(Scalar::F32 | Scalar::I32) | Type::Vec(Scalar::F32 | Scalar::I32, 1));
                    if !store && dword {
                        let key = (op.operands[0], op.operands[1]);
                        match open4.iter_mut().find(|(k, _)| *k == key) {
                            Some((_, loads)) => loads.push((offset, index)),
                            None => open4.push((key, vec![(offset, index)])),
                        }
                    }
                    self.accesses.push(Access {
                        alloc: op.operands[0],
                        addr: op.operands[1],
                        offset,
                        bytes,
                        align: op.int_attr("align").unwrap_or(4).clamp(1, 1 << 16) as u32,
                        weight,
                        half: half.then(|| op.results[0]),
                    });
                }
                OpKind::For => {
                    let trips = match (self.konst(op, 0), self.konst(op, 1), self.konst(op, 2)) {
                        (Some(lo), Some(hi), Some(step)) if step > 0 => ((hi - lo).max(0) + step - 1) / step,
                        _ => DEFAULT_TRIPS as i64,
                    };
                    let inner = (weight * trips.max(1) as f64).min(1e12);
                    let iv = self.konst(op, 0).zip(self.konst(op, 2));
                    for (i, &arg) in op.block_args.iter().enumerate() {
                        self.args.insert(arg, if i == 0 { Def::Iv(iv) } else { Def::IterArg });
                        self.def_weight[arg.0 as usize] = inner as f32;
                    }
                    for r in &op.regions {
                        self.walk(r, inner);
                    }
                }
                _ => {
                    for r in &op.regions {
                        self.walk(r, weight);
                    }
                }
            }
        }
        self.close_runs(&mut open);
        self.close_quads(&mut open4);
    }

    fn close_runs(&mut self, open: &mut OpenRuns) {
        for (_, mut loads) in open.drain(..) {
            loads.sort_unstable();
            let mut i = 0;
            while i < loads.len() {
                let mut j = i + 1;
                while j < loads.len() && loads[j].0 == loads[j - 1].0 + 2 {
                    j += 1;
                }
                if j - i >= 2 {
                    self.runs.push(loads[i..j].iter().map(|&(_, a)| a).collect());
                }
                i = j;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Address evaluation
// ---------------------------------------------------------------------------

struct Evaluator<'a, 'c> {
    col: &'c Collector<'a>,
    waves: usize,
    threads: i64,
    memo: FxHashMap<ValueId, Option<Lin>>,
    /// Every unknown value is zero instead (the self-check's instance).
    zero: bool,
}

/// Arithmetic `apply` models exactly where it can.
fn modeled(kind: OpKind) -> bool {
    use OpKind::*;
    matches!(
        kind,
        Add | Sub | Mul | Shl | Lshr | Ashr | And | Or | Xor | Div | Mod | Smin | Smax | Select | Cmp | Sext | Zext | Trunc | ReadFirstLane
    )
}

fn int_type(ty: &Type) -> bool {
    matches!(ty, Type::Scalar(Scalar::I32 | Scalar::I64 | Scalar::I1))
}

impl Evaluator<'_, '_> {
    fn slots(&self) -> usize {
        self.waves * WAVE
    }

    /// Ops whose operands are evaluated: modeled arithmetic, and other pure
    /// integer ops (a result of wave-uniform operands is a symbol).
    fn evaluates_operands(&self, op: &Op) -> bool {
        op.results.len() == 1
            && int_type(self.col.kernel.ty(op.results[0]))
            && (modeled(op.kind) || op.kind.is_pure())
            && !matches!(op.kind, OpKind::ThreadId | OpKind::LaneId | OpKind::Constant)
    }

    /// Value of `v`, its operands evaluated first without recursion.
    fn eval(&mut self, v: ValueId) -> Option<Lin> {
        if let Some(a) = self.memo.get(&v) {
            return a.clone();
        }
        let mut stack = vec![v];
        let mut visited = 0usize;
        while let Some(&top) = stack.last() {
            if self.memo.contains_key(&top) {
                stack.pop();
                continue;
            }
            visited += 1;
            if visited > MAX_EXPR_NODES {
                return None;
            }
            let pending: Vec<ValueId> = match self.col.defs[top.0 as usize] {
                Some(op) if self.evaluates_operands(op) => {
                    op.operands.iter().copied().filter(|o| !self.memo.contains_key(o)).collect()
                }
                _ => Vec::new(),
            };
            if pending.is_empty() {
                let value = self.compute(top);
                self.memo.insert(top, value);
                stack.pop();
            } else {
                stack.extend(pending);
            }
        }
        self.memo.get(&v).cloned().flatten()
    }

    fn compute(&self, v: ValueId) -> Option<Lin> {
        let value = self.value(v);
        if self.zero {
            return Some(value.unwrap_or(Lin::konst(0)));
        }
        value
    }

    /// A symbol, or zero for the self-check.
    fn sym(&self, v: u32) -> Lin {
        if self.zero {
            Lin::konst(0)
        } else {
            Lin::sym(v)
        }
    }

    fn value(&self, v: ValueId) -> Option<Lin> {
        let slots = self.slots();
        let Some(op) = self.col.defs[v.0 as usize] else {
            return match self.col.args.get(&v) {
                Some(Def::Iv(Some((start, step)))) => {
                    let mut l = self.sym(TRIP_SYMBOL | v.0).scale(*step);
                    l.konst = *start;
                    l.checked()
                }
                Some(Def::Iv(None)) => Some(self.sym(v.0)),
                Some(Def::IterArg) | None => None,
            };
        };
        match op.kind {
            OpKind::Constant => match self.col.facts.konst.get(v.0 as usize).copied().flatten() {
                Some(Konst::Int(k)) => Lin::konst(k).checked(),
                _ => Some(self.sym(v.0)),
            },
            OpKind::ThreadId | OpKind::LaneId => {
                let lane = op.kind == OpKind::LaneId;
                let lanes: Rc<[i64]> =
                    (0..slots as i64).map(|t| if lane { t % WAVE as i64 } else { t.min(self.threads - 1) }).collect();
                Some(Lin { konst: 0, lanes: Some(lanes), terms: Vec::new() })
            }
            OpKind::KernArg | OpKind::BlockId => Some(self.sym(v.0)),
            _ if self.evaluates_operands(op) && op.operands.len() <= 4 => {
                let zero = Lin::konst(0);
                let mut args = [&zero; 4];
                for (slot, o) in args.iter_mut().zip(op.operands.iter()) {
                    *slot = self.memo.get(o)?.as_ref()?;
                }
                let args = &args[..op.operands.len()];
                let exact = if modeled(op.kind) { self.apply(op, args).and_then(Lin::checked) } else { None };
                exact.or_else(|| args.iter().all(|a| a.wave_uniform()).then(|| self.sym(v.0)))
            }
            _ => None,
        }
    }

    /// The result of a modeled op, or `None` when the per-lane part cannot be
    /// computed exactly.
    fn apply(&self, op: &Op, a: &[&Lin]) -> Option<Lin> {
        use OpKind::*;
        let slots = self.slots();
        let x = *a.first()?;
        let k1 = a.get(1).and_then(|l| l.as_const());
        let nonneg = |i: usize| self.col.facts.nonneg(op.operands[i]);
        let elementwise = |f: &dyn Fn(&[i64]) -> Option<i64>| -> Option<Lin> {
            let exact: Vec<Vec<i64>> = a.iter().map(|l| l.exact(slots)).collect::<Option<_>>()?;
            let mut row = vec![0; exact.len()];
            let mut lanes = Vec::with_capacity(slots);
            for i in 0..slots {
                for (j, e) in exact.iter().enumerate() {
                    row[j] = e[i];
                }
                lanes.push(f(&row)?);
            }
            Some(Lin { konst: 0, lanes: Some(lanes.into()), terms: Vec::new() })
        };
        match op.kind {
            Add => Some(x.add(a[1], 1)),
            Sub => Some(x.add(a[1], -1)),
            Mul => match (x.as_const(), k1) {
                (_, Some(k)) => Some(x.scale(k)),
                (Some(k), _) => Some(a[1].scale(k)),
                _ => elementwise(&|e| e[0].checked_mul(e[1])),
            },
            Shl => match k1 {
                Some(s) if (0..31).contains(&s) => Some(x.scale(1 << s)),
                _ => elementwise(&|e| (0..31).contains(&e[1]).then(|| e[0] << e[1])),
            },
            // Known 32-bit values shift and divide as the hardware does.
            Ashr | Lshr | Div | Mod if x.terms.is_empty() && a[1].terms.is_empty() => {
                let i32_op = op.kind;
                elementwise(&|e| {
                    let (p, q) = (e[0] as i32, e[1] as i32);
                    Some(match i32_op {
                        Ashr => p.wrapping_shr(q as u32 & 31) as i64,
                        Lshr => ((p as u32) >> (q as u32 & 31)) as i64,
                        Div => p.checked_div(q)? as i64,
                        _ => p.checked_rem(q)? as i64,
                    })
                })
            }
            // Symbols that are multiples of 2^s shift exactly (floor); a
            // logical shift needs a non-negative operand to be one.
            Ashr | Lshr => match k1 {
                Some(s) if (0..31).contains(&s) && x.align() >= (1 << s) && (op.kind == Ashr || nonneg(0)) => {
                    Some(x.map_exact(|t| t >> s, |c| c >> s))
                }
                _ => None,
            },
            Div | Mod => match k1 {
                Some(d) if d > 0 && x.terms.iter().all(|&(_, c)| c % d == 0) && nonneg(0) && x.exact_nonneg() => {
                    Some(if op.kind == Div {
                        x.map_exact(|t| t / d, |c| c / d)
                    } else {
                        x.map_exact(|t| t % d, |_| 0)
                    })
                }
                _ => None,
            },
            And => match k1 {
                Some(m) if m >= 0 && m < x.align() => Some(x.map_exact(|t| t & m, |_| 0)),
                _ => elementwise(&|e| Some(e[0] & e[1])),
            },
            Or => match k1 {
                Some(m) if m >= 0 && m < x.align() => Some(x.map_exact(|t| t | m, |c| c)),
                _ => elementwise(&|e| Some(e[0] | e[1])),
            },
            Xor => match k1 {
                Some(m) if m >= 0 && m < x.align() => Some(x.map_exact(|t| t ^ m, |c| c)),
                _ => elementwise(&|e| Some(e[0] ^ e[1])),
            },
            Smin => elementwise(&|e| Some(e[0].min(e[1]))),
            Smax => elementwise(&|e| Some(e[0].max(e[1]))),
            Select => elementwise(&|e| Some(if e[0] != 0 { e[1] } else { e[2] })),
            Cmp => {
                let pred = op.str_attr("pred").unwrap_or("lt");
                elementwise(&|e| {
                    let (p, q) = (e[0], e[1]);
                    let r = match pred {
                        "eq" => p == q,
                        "ne" => p != q,
                        "lt" => p < q,
                        "le" => p <= q,
                        "gt" => p > q,
                        "ge" => p >= q,
                        "ult" => (p as u32) < (q as u32),
                        _ => return None,
                    };
                    Some(r as i64)
                })
            }
            Sext => Some(x.clone()),
            Zext | Trunc => {
                let fits = self.col.facts.range_of(op.operands[0]).is_some_and(|r| r.lo >= 0 && r.hi <= i32::MAX as i64);
                fits.then(|| x.clone())
            }
            ReadFirstLane => Some(match &x.lanes {
                None => x.clone(),
                Some(l) => {
                    let lanes: Rc<[i64]> = (0..slots).map(|i| l[i - i % WAVE]).collect();
                    Lin { konst: x.konst, lanes: Some(lanes), terms: x.terms.clone() }
                }
            }),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// Pricing
// ---------------------------------------------------------------------------

/// The per-(wave, lane) addresses one base value gives (symbols zero), the
/// symbol part's alignment and, per wave, whether its lanes are wave 0's
/// shifted by one constant.
struct Pattern {
    at: Vec<i64>,
    align: i64,
    shifts: Vec<Option<i64>>,
}

impl Pattern {
    fn new(lin: &Lin, waves: usize) -> Pattern {
        let at: Vec<i64> = (0..waves * WAVE).map(|s| lin.at(s)).collect();
        let shifts = (0..waves)
            .map(|w| {
                let s = at[w * WAVE] - at[0];
                (0..WAVE).all(|l| at[w * WAVE + l] - at[l] == s).then_some(s)
            })
            .collect();
        Pattern { at, align: lin.align(), shifts }
    }
}

/// One machine access, or a folded run that is one or several of them
/// depending on the layout.
#[derive(Clone, Copy, Debug)]
struct Unit {
    pattern: usize,
    delta: i64,
    width: u32,
}

#[derive(Clone, Debug)]
struct Run {
    pattern: usize,
    offset: i64,
    bytes: u32,
    weight: f64,
    /// Bytes per load: 2 for a folded f16 run, 4 for a dword quad.
    elem: u32,
}

/// Conflicts seen while pricing: per stride exponent, the weighted excess
/// cycles and the worst conflict degree.
struct Conflicts {
    excess: [f64; 17],
    degree: [u32; 17],
}

impl Conflicts {
    fn new() -> Conflicts {
        Conflicts { excess: [0.0; 17], degree: [0; 17] }
    }
}

struct Pricer<'p> {
    patterns: &'p [Pattern],
    /// Weighted machine accesses that are not in folded runs, merged by
    /// (pattern, offset, width).
    units: Vec<(Unit, f64)>,
    runs: Vec<Run>,
    waves: usize,
    /// Per layout period: the units merged by offset modulo the period,
    /// with their weight and count.
    by_period: FxHashMap<i64, Vec<(Unit, f64, usize)>>,
    memo: CycleMemo,
    lanes: [i64; WAVE],
}

impl Pricer<'_> {
    /// Cycles summed over waves of one machine access, averaged over the
    /// samples of the symbol part, and the closest conflict stride seen.
    fn unit_cycles(&mut self, layout: Layout, u: Unit) -> (f64, Option<u32>) {
        let p = &self.patterns[u.pattern];
        let period = layout.period();
        let samples = if p.align >= period { 1 } else { (period / p.align).min(MAX_SAMPLES) };
        let step = if samples > 1 { (period / samples / p.align).max(1) * p.align } else { 0 };
        let mut total = 0u64;
        let mut stride: Option<u32> = None;
        for j in 0..samples {
            for w in 0..self.waves {
                let (wave_key, base) = match p.shifts[w] {
                    Some(s) => (0u32, u.delta + s + j * step),
                    None => (w as u32, u.delta + j * step),
                };
                let key = (u.pattern as u32, u.width, base.rem_euclid(period), wave_key);
                let (c, s) = match self.memo.get(&key) {
                    Some(&hit) => hit,
                    None => {
                        let from = wave_key as usize * WAVE;
                        for (l, slot) in self.lanes.iter_mut().enumerate() {
                            *slot = layout.apply(p.at[from + l] + base);
                        }
                        let found = access_cycles(&self.lanes, u.width);
                        self.memo.insert(key, found);
                        found
                    }
                };
                if let Some(s) = s {
                    stride = Some(stride.map_or(s, |t| t.min(s)));
                }
                total += c as u64;
            }
        }
        (total as f64 / samples as f64, stride)
    }

    /// The machine accesses of a folded run under a layout: vector loads
    /// while the layout keeps it contiguous and its start aligned, 16-bit
    /// loads otherwise.
    fn run_units(&self, layout: Layout, r: &Run, out: &mut Vec<(i64, u32)>) {
        let p = &self.patterns[r.pattern];
        let mut contiguous = true;
        let mut align = p.align.min(16);
        for slot in 0..self.waves * WAVE {
            let a = p.at[slot] + r.offset;
            contiguous &= layout.contiguous(a, r.bytes as i64);
            let t = layout.apply(a);
            align = align.min(if t == 0 { 16 } else { 1 << t.trailing_zeros() });
        }
        if r.elem == 4 {
            quad_units(r.offset, contiguous && align >= 16, out);
        } else if contiguous && align >= 4 {
            run_units(r.offset, r.bytes, align, out);
        } else {
            out.extend((0..r.bytes as i64 / 2).map(|i| (r.offset + 2 * i, 2)));
        }
    }

    /// Weighted cycles per wave under `layout` and the static machine
    /// accesses with a conflict; records the conflicts seen.
    fn price(&mut self, layout: Layout, seen: &mut Conflicts) -> (f64, usize) {
        self.memo.clear();
        let period = layout.period();
        let merged = match self.by_period.remove(&period) {
            Some(m) => m,
            None => {
                // Units are sorted by (pattern, width): merge each group's
                // offsets by their residue.
                let mut m: Vec<(Unit, f64, usize)> = Vec::new();
                let mut group = (usize::MAX, 0u32, 0usize);
                for &(u, w) in &self.units {
                    if (group.0, group.1) != (u.pattern, u.width) {
                        group = (u.pattern, u.width, m.len());
                    }
                    let r = u.delta.rem_euclid(period);
                    match m[group.2..].iter_mut().find(|e| e.0.delta == r) {
                        Some(e) => {
                            e.1 += w;
                            e.2 += 1;
                        }
                        None => m.push((Unit { delta: r, ..u }, w, 1)),
                    }
                }
                m
            }
        };
        let (mut cycles, mut conflicted) = (0.0, 0);
        let mut note = |c: f64, ideal: f64, weight: f64, stride: Option<u32>| {
            if let Some(s) = stride.filter(|_| c > ideal) {
                let s = s.min(16) as usize;
                seen.excess[s] += weight * (c - ideal);
                seen.degree[s] = seen.degree[s].max((c / ideal).ceil() as u32);
            }
        };
        for &(u, weight, count) in &merged {
            let (c, s) = self.unit_cycles(layout, u);
            let ideal = (ideal_cycles(u.width) as usize * self.waves) as f64;
            cycles += weight * c;
            if c > ideal {
                conflicted += count;
            }
            note(c, ideal, weight, s);
        }
        self.by_period.insert(period, merged);
        let runs = std::mem::take(&mut self.runs);
        let mut pieces = Vec::new();
        for r in &runs {
            pieces.clear();
            self.run_units(layout, r, &mut pieces);
            for &(delta, width) in &pieces {
                let (c, s) = self.unit_cycles(layout, Unit { pattern: r.pattern, delta, width });
                let ideal = (ideal_cycles(width) as usize * self.waves) as f64;
                cycles += r.weight * c;
                conflicted += (c > ideal) as usize;
                note(c, ideal, r.weight, s);
            }
        }
        self.runs = runs;
        (cycles / self.waves as f64, conflicted)
    }

    /// Weighted conflict-free cycles per wave and static machine accesses.
    fn ideal(&self) -> (f64, usize) {
        let mut cycles: f64 = self.units.iter().map(|&(u, w)| w * ideal_cycles(u.width) as f64).sum();
        let mut machine = self.units.len();
        let mut pieces = Vec::new();
        for r in &self.runs {
            pieces.clear();
            self.run_units(Layout::Identity, r, &mut pieces);
            cycles += r.weight * pieces.iter().map(|&(_, w)| ideal_cycles(w) as f64).sum::<f64>();
            machine += pieces.len();
        }
        (cycles, machine)
    }
}

// ---------------------------------------------------------------------------
// Planning
// ---------------------------------------------------------------------------

/// What the analysis found for one allocation.
#[derive(Clone, Debug)]
pub struct AllocReport {
    pub name: String,
    pub bytes: u32,
    pub accesses: usize,
    pub machine_accesses: usize,
    pub analyzable: bool,
    pub note: String,
    /// Weighted LDS cycles per wave: conflict-free, current layout, chosen.
    pub ideal: f64,
    pub before: f64,
    pub after: f64,
    /// Static machine accesses with a conflict, before and after.
    pub conflicted_before: usize,
    pub conflicted_after: usize,
    pub layout: Layout,
    pub new_bytes: u32,
    pub candidates: usize,
    /// Transformed base addresses the rewrite computes, and the accesses
    /// that use a base offset by their residue (see `Fold`).
    pub bases: usize,
    pub explicit: usize,
    /// Workgroups per WGP that LDS allows before and after (equal when the
    /// layout is unchanged).
    pub groups: (u32, u32),
}

/// How a planned access at `addr + offset` is rewritten, with `offset = q *
/// period + r` and `apply(x + q * period) = apply(x) + q * shift`: its
/// address becomes `apply(addr)` when `apply(addr + r) - apply(addr)` is one
/// constant everywhere, else `apply(addr + r)`, shared by the accesses of
/// the base with that residue; the rest is a constant offset.
#[derive(Clone, Copy, Debug)]
enum Fold {
    Offset(i64),
    Residue(i64, i64),
}

struct Plan {
    alloc: ValueId,
    layout: Layout,
    new_bytes: u32,
    folds: Vec<(usize, Fold)>,
}

/// Workgroups a WGP holds as far as its 128 KiB of LDS and its 64 wave
/// slots allow (registers may allow fewer).
fn lds_groups(lds_bytes: u32, workgroup: u32) -> u32 {
    let by_waves = (64 / workgroup.div_ceil(32).max(1)).max(1);
    match lds_bytes {
        0 => by_waves,
        lds => (128 * 1024 / lds).clamp(1, by_waves),
    }
}

/// For each access of the allocation, how the rewrite folds it, and the
/// weighted count of instructions the rewrite adds; `None` when the layout
/// would split or misalign an access, or needs more than
/// `MAX_RESIDUE_BASES` residue bases.
fn fold_plan(
    col: &Collector,
    members: &[usize],
    pattern_of: &FxHashMap<ValueId, usize>,
    patterns: &[Pattern],
    layout: Layout,
    slots: usize,
) -> Option<(Vec<(usize, Fold)>, f64)> {
    let mut folds = Vec::with_capacity(members.len());
    let mut extra = 0.0;
    let mut bases: FxHashSet<(ValueId, i64)> = FxHashSet::default();
    let period = layout.period();
    let shift = layout.apply(period) - layout.apply(0);
    // Per (base, offset mod period, bytes): whether every slot stays
    // contiguous, and the common apply(base + r) - apply(base) if any.
    let mut memo: FxHashMap<(ValueId, i64, u32), (bool, Option<i64>)> = FxHashMap::default();
    for &m in members {
        let a = &col.accesses[m];
        let p = &patterns[*pattern_of.get(&a.addr)?];
        let at = &p.at;
        if layout.keeps_align() % (a.align as i64).min(16) != 0 {
            return None;
        }
        let bytes = a.bytes as i64;
        let r = a.offset.rem_euclid(period);
        let q = a.offset.div_euclid(period);
        // Symbols that are multiples of the period shift every image alike;
        // otherwise only alignment keeps an access in one chunk.
        let (contiguous, delta) = if p.align >= period {
            *memo.entry((a.addr, r, a.bytes)).or_insert_with(|| {
                let contiguous = (0..slots).all(|s| layout.contiguous(at[s] + r, bytes));
                let d0 = layout.apply(at[0] + r) - layout.apply(at[0]);
                let same = (1..slots).all(|s| layout.apply(at[s] + r) - layout.apply(at[s]) == d0);
                (contiguous, same.then_some(d0))
            })
        } else {
            let aligned = a.align as i64 >= bytes && (bytes as u64).is_power_of_two() && layout.chunk() % bytes == 0;
            (aligned, (r == 0).then_some(0))
        };
        if !contiguous {
            return None;
        }
        // A constant address keeps its value (see `rewrite`).
        let konst = col.facts.int(a.addr);
        let fold = match (konst, delta) {
            (Some(k), _) => Fold::Offset(layout.apply(k + a.offset) - k),
            (None, Some(d)) => Fold::Offset(d + q * shift),
            (None, None) => Fold::Residue(r, q * shift),
        };
        let base = match fold {
            Fold::Offset(_) => konst.is_none().then_some((a.addr, 0)),
            Fold::Residue(r, _) => Some((a.addr, r)),
        };
        if let Some(key) = base {
            if bases.insert(key) {
                extra += if key.1 == 0 { 3.0 } else { 4.0 } * col.def_weight[a.addr.0 as usize] as f64;
            }
        }
        folds.push((m, fold));
    }
    if bases.iter().filter(|b| b.1 != 0).count() > MAX_RESIDUE_BASES {
        return None;
    }
    Some((folds, extra))
}

/// Offsets once `alloc` grows to `new_bytes`, and the new pool size. The
/// allocation grows in place when every allocation that overlaps the added
/// bytes also overlapped its old bytes (so their lifetimes are disjoint);
/// otherwise it moves past the end of the pool. Entries are (allocation,
/// offset, bytes) of the allocations that are used.
fn relayout(layout: &[Slot], alloc: ValueId, new_bytes: u32) -> Option<(Vec<Slot>, u32)> {
    let &(_, off, old) = layout.iter().find(|e| e.0 == alloc)?;
    let overlaps = |o: u32, b: u32, lo: u32, hi: u32| o < hi && lo < o + b;
    let in_place = layout
        .iter()
        .all(|&(v, o, b)| v == alloc || !overlaps(o, b, off + old, off + new_bytes) || overlaps(o, b, off, off + old));
    let end = layout.iter().filter(|e| e.0 != alloc).map(|&(_, o, b)| o + b).max().unwrap_or(0);
    let mut out = layout.to_vec();
    for e in out.iter_mut().filter(|e| e.0 == alloc) {
        e.2 = new_bytes;
        if !in_place {
            e.1 = end.next_multiple_of(128);
        }
    }
    let size = out.iter().map(|&(_, o, b)| o + b).max().unwrap_or(0).next_multiple_of(16);
    Some((out, size))
}

struct Analysis {
    reports: Vec<AllocReport>,
    plans: Vec<Plan>,
    accesses: Vec<Access>,
    /// Used allocations as (allocation, offset, bytes) before and after the
    /// plans, in the same order.
    original: Vec<Slot>,
    layout: Vec<Slot>,
    size: u32,
}

impl Analysis {
    fn new(kernel: &Kernel, facts: &Facts, lds: &FxHashMap<ValueId, u32>, lds_size: u32) -> Analysis {
        let col = Collector::new(kernel, facts);
        let waves = (kernel.max_workgroup_size.max(1) as usize).div_ceil(WAVE);
        let slots = waves * WAVE;
        let mut ev =
            Evaluator { col: &col, waves, threads: kernel.max_workgroup_size.max(1) as i64, memo: FxHashMap::default(), zero: false };
        let allocs = &col.allocs;
        let used: FxHashSet<ValueId> = col.accesses.iter().map(|a| a.alloc).chain(col.escaped.iter().copied()).collect();
        let mut layout: Vec<Slot> = allocs
            .iter()
            .filter(|(v, _)| used.contains(v))
            .filter_map(|&(v, b)| lds.get(&v).map(|&o| (v, o, b)))
            .collect();
        let original = layout.clone();
        let mut size = lds_size;
        let mut in_run = vec![usize::MAX; col.accesses.len()];
        for (r, members) in col.runs.iter().enumerate() {
            for &m in members {
                in_run[m] = r;
            }
        }
        // Conflict-free LDS cycles per wave of the whole kernel, with folded
        // runs counted as their 16-bit loads (a slight overestimate).
        let mut pieces = Vec::new();
        let kernel_ideal: f64 = col
            .accesses
            .iter()
            .map(|a| {
                pieces.clear();
                units(a.offset, a.bytes, a.align, &mut pieces);
                a.weight * pieces.iter().map(|&(_, w)| ideal_cycles(w) as f64).sum::<f64>()
            })
            .sum();
        let min_gain = MIN_GAIN * kernel_ideal;
        let mut reports: Vec<AllocReport> = Vec::new();
        let mut plans = Vec::new();
        // Allocations accessed exactly alike (double buffers, sibling tiles)
        // share one decision: (size, accesses) -> report and plan indices.
        type Signature = (u32, Vec<(ValueId, i64, u32, u32, u32, usize)>);
        let mut decided: Vec<(Signature, usize, Option<usize>)> = Vec::new();
        for &(alloc, bytes) in allocs {
            let members: Vec<usize> = (0..col.accesses.len()).filter(|&i| col.accesses[i].alloc == alloc).collect();
            let signature: Signature = (
                bytes,
                members
                    .iter()
                    .map(|&m| {
                        let a = &col.accesses[m];
                        let folded = usize::from(in_run[m] != usize::MAX);
                        (a.addr, a.offset, a.bytes, a.align, a.weight.to_bits() as u32 ^ (a.weight.to_bits() >> 32) as u32, folded)
                    })
                    .collect(),
            );
            let size_now = size;
            if let Some((_, report, plan)) = decided.iter().find(|d| d.0 == signature && !col.escaped.contains(&alloc)) {
                let mut report = reports[*report].clone();
                report.name = kernel.name_of(alloc).to_string();
                let copied = plan.map(|p| &plans[p]).map(|p: &Plan| (p.layout, p.new_bytes, p.folds.clone()));
                if let Some((layout_, new_bytes, folds)) = copied {
                    // The same folds, on this allocation's own accesses.
                    let folds: Vec<(usize, Fold)> = folds.iter().zip(&members).map(|(&(_, f), &m)| (m, f)).collect();
                    let wg = kernel.max_workgroup_size;
                    let fits = relayout(&layout, alloc, new_bytes).filter(|&(_, s)| {
                        let s = s.max(size_now);
                        super::validate::lds_pool(s, &GFX1151).is_ok() && lds_groups(s, wg) >= lds_groups(size_now, wg)
                    });
                    if let Some((l, s)) = fits {
                        layout = l;
                        size = size.max(s);
                        plans.push(Plan { alloc, layout: layout_, new_bytes, folds });
                        reports.push(report);
                        continue;
                    }
                } else {
                    reports.push(report);
                    continue;
                }
            }
            decided.push((signature, reports.len(), None));
            let mut report = AllocReport {
                name: kernel.name_of(alloc).to_string(),
                bytes,
                accesses: members.len(),
                machine_accesses: 0,
                analyzable: true,
                note: String::new(),
                ideal: 0.0,
                before: 0.0,
                after: 0.0,
                conflicted_before: 0,
                conflicted_after: 0,
                layout: Layout::Identity,
                new_bytes: bytes,
                candidates: 0,
                bases: 0,
                explicit: 0,
                groups: (0, 0),
            };
            if members.is_empty() || !lds.contains_key(&alloc) {
                report.note = "no accesses".into();
                reports.push(report);
                continue;
            }
            if col.escaped.contains(&alloc) {
                report.analyzable = false;
                report.note = "used other than by LDS loads and stores".into();
                reports.push(report);
                continue;
            }
            // Patterns per base value, and the priced accesses.
            let mut patterns: Vec<Pattern> = Vec::new();
            let mut pattern_of: FxHashMap<ValueId, usize> = FxHashMap::default();
            let mut unknown = 0;
            for &m in &members {
                let addr = col.accesses[m].addr;
                if pattern_of.contains_key(&addr) {
                    continue;
                }
                match ev.eval(addr) {
                    Some(lin) => {
                        pattern_of.insert(addr, patterns.len());
                        patterns.push(Pattern::new(&lin, waves));
                    }
                    None => unknown += 1,
                }
            }
            if unknown > 0 {
                report.analyzable = false;
                report.note = format!("{unknown} access address(es) not analyzable");
                reports.push(report);
                continue;
            }
            let mut pricer = Pricer {
                patterns: &patterns,
                units: Vec::new(),
                runs: Vec::new(),
                waves,
                by_period: FxHashMap::default(),
                memo: FxHashMap::default(),
                lanes: [0; WAVE],
            };
            let mut pieces = Vec::new();
            let (mut max_unit, mut max_bytes) = (1u32, 1u32);
            for &m in &members {
                let a = &col.accesses[m];
                let pattern = pattern_of[&a.addr];
                max_bytes = max_bytes.max(a.bytes);
                let r = in_run[m];
                if r != usize::MAX {
                    let run = &col.runs[r];
                    if run[0] == m {
                        let elem = col.run_elem(r);
                        let bytes = elem * run.len() as u32;
                        max_bytes = max_bytes.max(bytes);
                        max_unit = max_unit.max(16.min(bytes));
                        pricer.runs.push(Run { pattern, offset: a.offset, bytes, weight: a.weight, elem });
                    }
                    continue;
                }
                pieces.clear();
                units(a.offset, a.bytes, a.align, &mut pieces);
                for &(delta, width) in &pieces {
                    max_unit = max_unit.max(width);
                    pricer.units.push((Unit { pattern, delta, width }, a.weight));
                }
            }
            // Group by (pattern, width): a counting sort on both.
            let class = |u: &Unit| u.pattern * 5 + u.width.min(16).next_power_of_two().trailing_zeros() as usize;
            let mut start = vec![0usize; patterns.len() * 5 + 1];
            for (u, _) in &pricer.units {
                start[class(u) + 1] += 1;
            }
            for i in 1..start.len() {
                start[i] += start[i - 1];
            }
            let mut grouped = vec![(Unit { pattern: 0, delta: 0, width: 0 }, 0.0); pricer.units.len()];
            for &(u, w) in &pricer.units {
                let c = class(&u);
                grouped[start[c]] = (u, w);
                start[c] += 1;
            }
            pricer.units = grouped;
            let (ideal, machine) = pricer.ideal();
            let mut seen = Conflicts::new();
            let (before, conflicted_before) = pricer.price(Layout::Identity, &mut seen);
            report.ideal = ideal;
            report.before = before;
            report.after = before;
            report.conflicted_before = conflicted_before;
            report.conflicted_after = conflicted_before;
            report.machine_accesses = machine;
            if before <= ideal * (1.0 + 1e-9) {
                report.note = "conflict-free".into();
                reports.push(report);
                continue;
            }
            if before - ideal < MIN_POTENTIAL * kernel_ideal {
                report.note = "conflicts too small to matter".into();
                reports.push(report);
                continue;
            }
            // Layouts against the conflict stride that costs the most, then
            // the next one while much of the excess remains.
            let mut order: Vec<usize> = (7..17).filter(|&s| seen.excess[s] > 0.0).collect();
            order.sort_by(|&x, &y| seen.excess[y].total_cmp(&seen.excess[x]).then(x.cmp(&y)));
            let mut tried = Vec::new();
            // (cost, static conflicts, layout, new bytes, occupancy factor):
            // a layout that leaves a WGP fewer resident workgroups costs as
            // if time grew in proportion, as in a latency-bound kernel.
            let wg = kernel.max_workgroup_size;
            let mut priced: Vec<(f64, usize, Layout, u32, f64)> = Vec::new();
            for &s in order.iter().take(STRIDES_TRIED) {
                let best = priced.iter().map(|p| p.0).fold(before, f64::min);
                if best - ideal < REMAINING_EXCESS * (before - ideal) {
                    break;
                }
                let from = tried.len();
                candidates(s as u32, seen.degree[s], max_unit, max_bytes, &mut tried);
                for &candidate in &tried[from..] {
                    let new_bytes = candidate.size(bytes).next_multiple_of(16);
                    let Some((_, new_size)) = relayout(&layout, alloc, new_bytes) else { continue };
                    let new_size = new_size.max(size);
                    if super::validate::lds_pool(new_size, &GFX1151).is_err()
                        || (lds_groups(new_size, wg) as f64) < MIN_GROUPS_KEPT * lds_groups(lds_size, wg) as f64
                    {
                        continue;
                    }
                    let occupancy = (lds_groups(size, wg) as f64 / lds_groups(new_size, wg) as f64).max(1.0);
                    let (cycles, conflicted) = pricer.price(candidate, &mut Conflicts::new());
                    priced.push((cycles * occupancy, conflicted, candidate, new_bytes, occupancy));
                }
            }
            report.candidates = priced.len();
            priced.sort_by(|x, y| x.0.total_cmp(&y.0));
            let mut chosen = None;
            for &(cycles, conflicted, candidate, new_bytes, occupancy) in &priced {
                if cycles >= before {
                    break;
                }
                if let Some((folds, extra)) = fold_plan(&col, &members, &pattern_of, &patterns, candidate, slots) {
                    let cost = cycles + occupancy * extra / waves as f64;
                    if before - cost >= min_gain.max(0.02 * (before - ideal)) && cost < before * 0.995 {
                        chosen = Some((cost, conflicted, candidate, new_bytes, folds));
                    }
                    break;
                }
            }
            match chosen {
                Some((cost, conflicted, candidate, new_bytes, folds)) => {
                    report.after = cost;
                    report.conflicted_after = conflicted;
                    report.layout = candidate;
                    report.new_bytes = new_bytes;
                    report.note = candidate.describe();
                    if let Some((_, s)) = relayout(&layout, alloc, new_bytes) {
                        report.groups = (lds_groups(size, wg), lds_groups(s.max(size), wg));
                    }
                    let mut bases = FxHashSet::default();
                    for &(m, fold) in &folds {
                        let r = match fold {
                            Fold::Offset(_) => 0,
                            Fold::Residue(r, _) => {
                                report.explicit += 1;
                                r
                            }
                        };
                        bases.insert((col.accesses[m].addr, r));
                    }
                    report.bases = bases.len();
                    if let Some((l, s)) = relayout(&layout, alloc, new_bytes) {
                        layout = l;
                        size = size.max(s);
                    }
                    if let Some(d) = decided.last_mut() {
                        d.2 = Some(plans.len());
                    }
                    plans.push(Plan { alloc, layout: candidate, new_bytes, folds });
                }
                None if priced.is_empty() => report.note = "no layout fits".into(),
                None => report.note = format!("best layout ({}) gains too little", priced[0].2.describe()),
            }
            reports.push(report);
        }
        Analysis { reports, plans, accesses: col.accesses, original, layout, size }
    }
}

// ---------------------------------------------------------------------------
// Rewrite
// ---------------------------------------------------------------------------

fn mk(kind: OpKind, results: &[ValueId], operands: &[ValueId], attrs: Vec<(Text, Attr)>) -> Op {
    Op {
        kind,
        results: results.iter().copied().collect(),
        operands: operands.iter().copied().collect(),
        attrs,
        regions: Vec::new(),
        block_args: Vec::new(),
    }
}

struct Rewriter<'f> {
    facts: &'f mut Facts,
    /// The kernel's top-level i32 constants, by value: they dominate every
    /// use, so the rewrite reuses them.
    constants: FxHashMap<i64, ValueId>,
    /// `layout(base + r)` per (plan, base address, r).
    bases: FxHashMap<(usize, ValueId, i64), ValueId>,
    /// Ops that go right after the definition of a value, or at the start
    /// of the loop body whose block argument it is.
    after: FxHashMap<ValueId, Vec<Op>>,
    /// Plan and fold per access number.
    folds: FxHashMap<usize, (usize, Fold)>,
}

impl Rewriter<'_> {
    fn set_fact(&mut self, v: ValueId, konst: Option<i64>, range: Option<Range>) {
        let n = v.0 as usize + 1;
        if self.facts.konst.len() < n {
            self.facts.konst.resize(n, None);
            self.facts.range.resize(n, None);
        }
        self.facts.konst[v.0 as usize] = konst.map(Konst::Int);
        self.facts.range[v.0 as usize] = range.or(konst.map(|k| Range::new(k, k)));
    }

    fn constant(&mut self, kernel: &mut Kernel, k: i64, out: &mut Vec<Op>) -> ValueId {
        if let Some(&v) = self.constants.get(&k) {
            return v;
        }
        let v = kernel.new_value("%swz.k", Type::Scalar(Scalar::I32));
        self.set_fact(v, Some(k), None);
        let attrs = vec![("ity".into(), Attr::Str("i32".into())), ("value".into(), Attr::Int(k))];
        out.push(mk(OpKind::Constant, &[v], &[], attrs));
        v
    }

    /// A new `kind` of `a` and `b`; isel needs no facts about it.
    fn binary(&mut self, kernel: &mut Kernel, kind: OpKind, a: ValueId, b: ValueId, out: &mut Vec<Op>) -> ValueId {
        let v = kernel.new_value("%swz.t", Type::Scalar(Scalar::I32));
        if (v.0 as usize) < self.facts.konst.len() {
            self.set_fact(v, None, None);
        }
        out.push(mk(kind, &[v], &[a, b], Vec::new()));
        v
    }

    /// Ops computing `layout(x)`, appended to `out`; returns the result.
    fn transform(&mut self, kernel: &mut Kernel, layout: Layout, x: ValueId, out: &mut Vec<Op>) -> ValueId {
        match layout {
            Layout::Identity => x,
            Layout::Pad { row, pad } => {
                let kr = self.constant(kernel, row as i64, out);
                let kp = self.constant(kernel, pad as i64, out);
                let q = self.binary(kernel, OpKind::Ashr, x, kr, out);
                let s = self.binary(kernel, OpKind::Shl, q, kp, out);
                self.binary(kernel, OpKind::Add, x, s, out)
            }
            Layout::Xor { low, bits, key } => {
                let kk = self.constant(kernel, key as i64, out);
                let km = self.constant(kernel, (1 << bits) - 1, out);
                let kl = self.constant(kernel, low as i64, out);
                let q = self.binary(kernel, OpKind::Lshr, x, kk, out);
                let m = self.binary(kernel, OpKind::And, q, km, out);
                let s = self.binary(kernel, OpKind::Shl, m, kl, out);
                self.binary(kernel, OpKind::Xor, x, s, out)
            }
        }
    }

    /// Patches the planned accesses of `region` (numbered in preorder from
    /// `counter`) in place, and inserts the ops of `after` where they go.
    /// `pending` flags the values that still have ops to insert after them.
    fn region(&mut self, analysis: &Analysis, region: &mut Region, counter: &mut usize, args: &[ValueId], pending: &mut [bool]) {
        let flagged = |v: &ValueId| pending.get(v.0 as usize) == Some(&true);
        let rebuild = args.iter().any(flagged) || region.ops.iter().any(|op| op.results.iter().any(flagged));
        for op in region.ops.iter_mut() {
            if !op.regions.is_empty() {
                let block_args = std::mem::take(&mut op.block_args);
                for r in op.regions.iter_mut() {
                    self.region(analysis, r, counter, &block_args, pending);
                }
                op.block_args = block_args;
            }
            if matches!(op.kind, OpKind::LLoad | OpKind::LStore) {
                let m = *counter;
                *counter += 1;
                if let Some(&(pi, fold)) = self.folds.get(&m) {
                    let a = &analysis.accesses[m];
                    let (r, offset) = match fold {
                        Fold::Offset(d) => (0, d),
                        Fold::Residue(r, d) => (r, d),
                    };
                    if let Some(&base) = self.bases.get(&(pi, a.addr, r)) {
                        op.operands[1] = base;
                    }
                    op.set_attr("offset", Attr::Int(offset));
                }
            }
        }
        if !rebuild {
            return;
        }
        let mut take = |v: ValueId, out: &mut Vec<Op>| {
            if let Some(flag) = pending.get_mut(v.0 as usize).filter(|f| **f) {
                *flag = false;
                if let Some(ops) = self.after.remove(&v) {
                    out.extend(ops);
                }
            }
        };
        let mut out = Vec::with_capacity(region.ops.len() + 8);
        for &arg in args {
            take(arg, &mut out);
        }
        for op in std::mem::take(&mut region.ops) {
            let results: Operands = op.results.iter().copied().collect();
            out.push(op);
            for &r in results.iter() {
                take(r, &mut out);
            }
        }
        region.ops = out;
    }
}

/// Applies the plans: each planned access gets its new address and offset,
/// the ops computing a transformed base follow the base's definition, and
/// planned allocations grow.
fn rewrite(kernel: &mut Kernel, facts: &mut Facts, analysis: &Analysis) {
    let mut constants = FxHashMap::default();
    for op in kernel.body.ops.iter().filter(|op| op.kind == OpKind::Constant) {
        let v = op.results[0];
        if let (Type::Scalar(Scalar::I32), Some(k)) = (kernel.ty(v), facts.int(v)) {
            constants.entry(k).or_insert(v);
        }
    }
    let mut rw = Rewriter { facts, constants, bases: FxHashMap::default(), after: FxHashMap::default(), folds: FxHashMap::default() };
    for (pi, plan) in analysis.plans.iter().enumerate() {
        for &(m, fold) in &plan.folds {
            rw.folds.insert(m, (pi, fold));
            let addr = analysis.accesses[m].addr;
            let r = match fold {
                Fold::Offset(_) => 0,
                Fold::Residue(r, _) => r,
            };
            // A constant address keeps its value; the offset absorbs the layout.
            let constant = rw.facts.int(addr).is_some() && r == 0;
            if constant || rw.bases.contains_key(&(pi, addr, r)) {
                continue;
            }
            let mut ops = Vec::new();
            let x = if r == 0 {
                addr
            } else {
                let k = rw.constant(kernel, r, &mut ops);
                rw.binary(kernel, OpKind::Add, addr, k, &mut ops)
            };
            let y = rw.transform(kernel, plan.layout, x, &mut ops);
            rw.bases.insert((pi, addr, r), y);
            rw.after.entry(addr).or_default().extend(ops);
        }
    }
    let mut pending = vec![false; kernel.values.len()];
    for v in rw.after.keys() {
        pending[v.0 as usize] = true;
    }
    let mut body = std::mem::take(&mut kernel.body);
    rw.region(analysis, &mut body, &mut 0, &[], &mut pending);
    kernel.body = body;
    for plan in &analysis.plans {
        if let Type::Smem(elem, _) = kernel.ty(plan.alloc).clone() {
            kernel.values[plan.alloc.0 as usize].ty = Type::Smem(elem, vec![plan.new_bytes / elem.bytes()]);
        }
    }
}

// ---------------------------------------------------------------------------
// Entry points
// ---------------------------------------------------------------------------

/// Chooses and applies a layout per LDS allocation of the optimized kernel.
/// Returns the facts, extended to the values the rewrite adds, and the LDS
/// pool size. Wave64 kernels keep their layout: the bank model is gfx1151's.
pub fn run(kernel: &mut Kernel, mut facts: Facts, lds: &mut FxHashMap<ValueId, u32>, lds_size: u32) -> (Facts, u32) {
    if lds.is_empty() || kernel.wave as usize != WAVE || super::debug_flag("noswizzle") {
        return (facts, lds_size);
    }
    let analysis = Analysis::new(kernel, &facts, lds, lds_size);
    if analysis.plans.is_empty() {
        return (facts, lds_size);
    }
    rewrite(kernel, &mut facts, &analysis);
    for &(v, o, _) in &analysis.layout {
        lds.insert(v, o);
    }
    (facts, analysis.size)
}

/// Per-allocation findings for an optimized kernel, without changing it.
pub fn analyze(kernel: &Kernel, facts: &Facts, lds: &FxHashMap<ValueId, u32>, lds_size: u32) -> Vec<AllocReport> {
    Analysis::new(kernel, facts, lds, lds_size).reports
}

/// The byte address of each LDS access of an optimized kernel per (wave,
/// lane) slot, in preorder, and whether it has an unknown wave-uniform part.
/// With `zero`, every unknown value is zero and every address is known.
fn access_addresses(kernel: &Kernel, facts: &Facts, zero: bool) -> Vec<AccessAt> {
    let col = Collector::new(kernel, facts);
    let waves = (kernel.max_workgroup_size.max(1) as usize).div_ceil(WAVE);
    let threads = kernel.max_workgroup_size.max(1) as i64;
    let mut ev = Evaluator { col: &col, waves, threads, memo: FxHashMap::default(), zero };
    col.accesses
        .iter()
        .map(|a| {
            let at = ev.eval(a.addr).map(|l| ((0..waves * WAVE).map(|s| l.at(s) + a.offset).collect(), !l.terms.is_empty()));
            (a.alloc, a.bytes, at)
        })
        .collect()
}

/// Checks the pass on an optimized kernel without changing it: applied to a
/// copy, every access of a transformed allocation must land on the layout's
/// image of its old address in every (wave, lane) slot, inside the
/// allocation's new bytes when it was inside the old ones, and a grown
/// allocation may overlap only allocations it overlapped before. Addresses
/// with an unknown wave-uniform part are checked with that part zero.
/// Returns the accesses checked exactly and those checked at zero.
pub fn self_check(kernel: &Kernel, facts: &Facts, lds: &FxHashMap<ValueId, u32>, lds_size: u32) -> Result<(usize, usize), String> {
    let analysis = Analysis::new(kernel, facts, lds, lds_size);
    if analysis.plans.is_empty() {
        return Ok((0, 0));
    }
    let mut copy = kernel.clone();
    let mut offsets = lds.clone();
    let (new_facts, size) = run(&mut copy, facts.clone(), &mut offsets, lds_size);
    let (mut before, mut after) = (access_addresses(kernel, facts, false), access_addresses(&copy, &new_facts, false));
    if before.len() != after.len() {
        return Err(format!("{} LDS accesses became {}", before.len(), after.len()));
    }
    // Where an address has an unknown part, both sides are compared with
    // every unknown value zero.
    let partial = |v: &[AccessAt]| v.iter().any(|e| e.2.as_ref().is_none_or(|x| x.1));
    if partial(&before) || partial(&after) {
        let (zb, za) = (access_addresses(kernel, facts, true), access_addresses(&copy, &new_facts, true));
        for (i, (b, a)) in before.iter_mut().zip(after.iter_mut()).enumerate() {
            let unknown = |e: &Option<(Vec<i64>, bool)>| e.as_ref().is_none_or(|x| x.1);
            if unknown(&b.2) || unknown(&a.2) {
                b.2 = zb[i].2.clone().map(|(x, _)| (x, true));
                a.2 = za[i].2.clone().map(|(x, _)| (x, true));
            }
        }
    }
    let (mut checked, mut skipped) = (0, 0);
    for plan in &analysis.plans {
        let old_bytes = analysis.original.iter().find(|e| e.0 == plan.alloc).map_or(i64::MAX, |e| e.2 as i64);
        for (i, (b, a)) in before.iter().zip(&after).enumerate() {
            if b.0 != plan.alloc {
                continue;
            }
            match (&b.2, &a.2) {
                (Some((old, symbolic)), Some((new, _))) => {
                    for (slot, (&x, &y)) in old.iter().zip(new).enumerate() {
                        if plan.layout.apply(x) != y {
                            return Err(format!("access {i} slot {slot}: {x} moved to {y}, not {}", plan.layout.apply(x)));
                        }
                        let inside = x >= 0 && x + b.1 as i64 <= old_bytes;
                        if !symbolic && inside && y + b.1 as i64 > plan.new_bytes as i64 {
                            return Err(format!("access {i} slot {slot}: {y} is past the {} new bytes", plan.new_bytes));
                        }
                    }
                    if *symbolic {
                        skipped += 1;
                    } else {
                        checked += 1;
                    }
                }
                (None, _) => return Err(format!("access {i}: the address is not analyzable")),
                (Some(_), None) => return Err(format!("access {i}: the rewritten address is not analyzable")),
            }
        }
    }
    // Allocations overlap afterwards only if they did before (disjoint
    // lifetimes), and the pool holds every allocation.
    let overlap = |a: &Slot, b: &Slot| a.1 < b.1 + b.2 && b.1 < a.1 + a.2;
    for (i, x) in analysis.layout.iter().enumerate() {
        if offsets.get(&x.0) != Some(&x.1) {
            return Err(format!("allocation {} is at {:?}, planned at {}", x.0 .0, offsets.get(&x.0), x.1));
        }
        for (j, y) in analysis.layout.iter().enumerate().skip(i + 1) {
            if overlap(x, y) && !overlap(&analysis.original[i], &analysis.original[j]) {
                return Err(format!("allocations {} and {} overlap only after the pass", x.0 .0, y.0 .0));
            }
        }
    }
    if size < analysis.layout.iter().map(|&(_, o, b)| o + b).max().unwrap_or(0) || super::validate::lds_pool(size, &GFX1151).is_err() {
        return Err(format!("pool of {size} bytes does not hold its allocations"));
    }
    Ok((checked, skipped))
}

/// The kernel name and per-allocation findings for IR text, without
/// changing anything.
pub fn analyze_text(text: &str) -> Result<(String, Vec<AllocReport>), crate::error::CompileError> {
    let mut kernel = super::parse::parse(text)?;
    let (mut lds, lds_size) = super::analysis::lds_layout(&kernel);
    let facts = super::opt::optimize(&mut kernel, &mut lds)?;
    Ok((kernel.name.clone(), analyze(&kernel, &facts, &lds, lds_size)))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn wave(f: impl Fn(i64) -> i64) -> [i64; WAVE] {
        std::array::from_fn(|l| f(l as i64))
    }

    fn cycles(f: impl Fn(i64) -> i64, width: u32) -> u32 {
        access_cycles(&wave(f), width).0
    }

    #[test]
    fn bank_model_matches_gfx1151_measurements() {
        // b32 and narrower: one 32-lane phase; broadcasts are free.
        assert_eq!(cycles(|l| 4 * l, 4), 1);
        assert_eq!(cycles(|_| 0, 4), 1);
        assert_eq!(cycles(|l| 4 * (l % 16), 4), 1);
        assert_eq!(cycles(|l| 8 * l, 4), 2);
        assert_eq!(cycles(|l| 32 * l, 4), 8);
        assert_eq!(cycles(|l| 128 * l, 4), 32);
        assert_eq!(cycles(|l| 132 * l, 4), 1);
        assert_eq!(cycles(|l| 4 * (l % 16) + 256 * (l / 16), 4), 2);
        // b64: two 16-lane phases.
        assert_eq!(cycles(|l| 8 * l, 8), 2);
        assert_eq!(cycles(|l| 16 * l, 8), 4);
        assert_eq!(cycles(|l| 32 * l, 8), 8);
        assert_eq!(cycles(|l| 256 * (l % 2) + 8 * (l / 2), 8), 4);
        assert_eq!(cycles(|l| 8 * (l % 16) + 256 * (l / 16), 8), 2);
        // b128: four 8-lane phases.
        assert_eq!(cycles(|l| 16 * l, 16), 4);
        assert_eq!(cycles(|_| 0, 16), 4);
        assert_eq!(cycles(|l| 128 * (l % 16), 16), 32);
        assert_eq!(cycles(|l| 256 * (l % 16), 16), 32);
        assert_eq!(cycles(|l| 144 * (l % 16), 16), 4);
        assert_eq!(cycles(|l| 32 * (l % 16), 16), 8);
        assert_eq!(cycles(|l| 64 * (l % 16), 16), 16);
        assert_eq!(cycles(|l| 16 * (l % 4) + 128 * (l / 4), 16), 8);
        assert_eq!(cycles(|l| 16 * (l % 4) + 64 * ((l / 4) % 2) + 256 * (l / 8), 16), 4);
        // Sub-dword accesses: two lanes in one dword cost nothing extra.
        assert_eq!(cycles(|l| 2 * l, 2), 1);
        assert_eq!(cycles(|l| 128 * l, 2), 32);
        assert_eq!(cycles(|l| l, 1), 1);
        // The stride of the closest conflicting rows.
        assert_eq!(access_cycles(&wave(|l| 128 * (l % 16)), 16).1, Some(7));
        assert_eq!(access_cycles(&wave(|l| 256 * (l % 16)), 16).1, Some(8));
        assert_eq!(access_cycles(&wave(|l| 4 * (l % 16) + 256 * (l / 16)), 4).1, Some(8));
        assert_eq!(access_cycles(&wave(|l| 4 * l), 4).1, None);
    }

    /// A two-wave kernel that stores a 16-row f16 tile of `stride` elements
    /// per row (vec8 stores, eight lanes per row) and reads it back the way
    /// WMMA operands are read: lane `l` loads row `l % 16`, 16 bytes at a
    /// time. `extra` goes right after the loads.
    fn rows_kernel(stride: u32, extra: &str) -> String {
        let elems = 16 * stride;
        let mut stores = String::new();
        for j in 0..4 {
            stores.push_str(&format!(
                "    %so{j} : i32 = arith.constant (  ) {{ ity = s:\"i32\", value = i:{} }}\n\
                 \x20   %s{j} : i32 = arith.add ( %st, %so{j} )\n\
                 \x20   %g{j} : vec<f16x8> = memref.global_load_vN ( %src, %s{j} ) {{ align = i:16, elem_type = s:\"f16\", vec = i:8 }}\n\
                 \x20   tile.smem_store_vN ( %lds, %s{j}, %g{j} ) {{ align = i:16, elem_type = s:\"f16\", rank = i:1, vec = i:8 }}\n",
                4 * j * stride
            ));
        }
        let mut loads = String::new();
        for c in 0..4 {
            loads.push_str(&format!(
                "    %lo{c} : i32 = arith.constant (  ) {{ ity = s:\"i32\", value = i:{} }}\n\
                 \x20   %l{c} : i32 = arith.add ( %lr, %lo{c} )\n\
                 \x20   %v{c} : vec<f16x8> = tile.smem_load_vN ( %lds, %l{c} ) {{ elem_type = s:\"f16\", rank = i:1, vec = i:8 }}\n\
                 \x20   %d{c} : i32 = arith.add ( %out, %lo{c} )\n\
                 \x20   memref.global_store_vN ( %dst, %d{c}, %v{c} ) {{ align = i:16, elem_type = s:\"f16\", vec = i:8 }}\n",
                8 * c
            ));
        }
        format!(
            "ckdsl.ir v1\nkernel @swz_rows {{\n  attrs {{ max_workgroup_size = i:64 }}\n  params {{\n    %src : ptr<f16,global>\n    %dst : ptr<f16,global>\n  }}\n  region @entry {{\n\
             \x20   %tid : i32 = gpu.thread_id (  ) {{ axis = s:\"x\" }}\n\
             \x20   %c8 : i32 = arith.constant (  ) {{ ity = s:\"i32\", value = i:8 }}\n\
             \x20   %c16 : i32 = arith.constant (  ) {{ ity = s:\"i32\", value = i:16 }}\n\
             \x20   %c32 : i32 = arith.constant (  ) {{ ity = s:\"i32\", value = i:32 }}\n\
             \x20   %cs : i32 = arith.constant (  ) {{ ity = s:\"i32\", value = i:{stride} }}\n\
             \x20   %lds : smem<f16, [{elems}]> = tile.smem_alloc (  )\n\
             \x20   %lane : i32 = arith.mod ( %tid, %c32 )\n\
             \x20   %row : i32 = arith.div ( %lane, %c8 )\n\
             \x20   %chunk : i32 = arith.mod ( %lane, %c8 )\n\
             \x20   %rowe : i32 = arith.mul ( %row, %cs )\n\
             \x20   %che : i32 = arith.mul ( %chunk, %c8 )\n\
             \x20   %st : i32 = arith.add ( %rowe, %che )\n\
             {stores}\
             \x20   tile.sync (  )\n\
             \x20   %r : i32 = arith.mod ( %tid, %c16 )\n\
             \x20   %lr : i32 = arith.mul ( %r, %cs )\n\
             \x20   %out : i32 = arith.mul ( %tid, %c32 )\n\
             {loads}{extra}\
             \x20   cf.return (  )\n  }}\n}}\n"
        )
    }

    struct Optimized {
        kernel: Kernel,
        facts: Facts,
        lds: FxHashMap<ValueId, u32>,
        size: u32,
    }

    fn optimized(text: &str) -> Optimized {
        let mut kernel = super::super::parse::parse(text).unwrap_or_else(|e| panic!("{e}"));
        let (mut lds, size) = super::super::analysis::lds_layout(&kernel);
        let facts = super::super::opt::optimize(&mut kernel, &mut lds).unwrap_or_else(|e| panic!("{e}"));
        Optimized { kernel, facts, lds, size }
    }

    /// The byte address of every LDS access in every (wave, lane) slot.
    fn addresses(o: &Optimized) -> Vec<Vec<i64>> {
        let col = Collector::new(&o.kernel, &o.facts);
        let waves = (o.kernel.max_workgroup_size as usize).div_ceil(WAVE);
        let threads = o.kernel.max_workgroup_size as i64;
        let mut ev = Evaluator { col: &col, waves, threads, memo: FxHashMap::default(), zero: false };
        col.accesses
            .iter()
            .map(|a| {
                let lin = ev.eval(a.addr).expect("analyzable address");
                lin.exact(waves * WAVE).expect("no symbols").iter().map(|x| x + a.offset).collect()
            })
            .collect()
    }

    /// Applies the pass and checks that every access moved to the layout's
    /// image of its old address, lane by lane; returns the layout.
    fn check_rewrite(text: &str) -> (Layout, Optimized) {
        let mut o = optimized(text);
        let before = addresses(&o);
        let reports = analyze(&o.kernel, &o.facts, &o.lds, o.size);
        let layout = reports[0].layout;
        let facts = std::mem::take(&mut o.facts);
        let (facts, size) = run(&mut o.kernel, facts, &mut o.lds, o.size);
        o.facts = facts;
        assert_eq!(size, o.size.max(*o.lds.values().max().unwrap() + reports[0].new_bytes).next_multiple_of(16));
        o.size = size;
        let after = addresses(&o);
        assert_eq!(before.len(), after.len());
        for (b, a) in before.iter().zip(&after) {
            let moved: Vec<i64> = b.iter().map(|&x| layout.apply(x)).collect();
            assert_eq!(&moved, a, "{layout:?}");
        }
        (layout, o)
    }

    #[test]
    fn conflicting_rows_are_padded_and_every_access_follows() {
        // 128-byte rows: the eight lanes of a b128 phase hit the same banks.
        let text = rows_kernel(64, "");
        let o = optimized(&text);
        let reports = analyze(&o.kernel, &o.facts, &o.lds, o.size);
        assert_eq!(reports.len(), 1);
        let r = &reports[0];
        assert_eq!((r.ideal, r.before), (4.0 * 4.0 + 4.0 * 4.0, 4.0 * 4.0 + 4.0 * 32.0));
        assert_eq!(r.layout, Layout::Pad { row: 7, pad: 4 });
        assert_eq!((r.conflicted_before, r.conflicted_after), (4, 0));
        let (layout, o) = check_rewrite(&text);
        assert_eq!(layout, Layout::Pad { row: 7, pad: 4 });
        assert_eq!(o.size, 2048 + 16 * 15);
        let compiled = super::super::compile(&text).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(compiled.stats.lds_bytes, 2048 + 16 * 15);
        let wave64 = super::super::compile_for(&text, crate::target::Target::Gfx950).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(wave64.stats.lds_bytes, 2048);
    }

    #[test]
    fn conflict_free_layouts_are_left_alone() {
        // 144-byte rows (already padded): nothing to gain.
        let text = rows_kernel(72, "");
        let mut o = optimized(&text);
        let reports = analyze(&o.kernel, &o.facts, &o.lds, o.size);
        assert_eq!((reports[0].before, reports[0].note.as_str()), (reports[0].ideal, "conflict-free"));
        let ops = super::super::print::op_count(&o.kernel.body);
        let facts = std::mem::take(&mut o.facts);
        let (_, size) = run(&mut o.kernel, facts, &mut o.lds, o.size);
        assert_eq!((size, super::super::print::op_count(&o.kernel.body)), (o.size, ops));
    }

    #[test]
    fn an_lds_atomic_keeps_its_allocation_as_it_is() {
        let atomic = "    %one : i32 = arith.constant (  ) { ity = s:\"i32\", value = i:1 }\n\
                      \x20   %old : i32 = tile.lds_atomic_add ( %lds, %lr, %one ) { elem_type = s:\"i32\", ordering = s:\"monotonic\", rank = i:1 }\n\
                      \x20   memref.global_store_typed ( %dst, %out, %old ) { align = i:4, elem_type = s:\"i32\" }\n";
        let text = rows_kernel(64, atomic).replace("%dst : ptr<f16,global>", "%dst : ptr<f16,global>\n    %cnt : ptr<i32,global>");
        let text = text.replace("memref.global_store_typed ( %dst,", "memref.global_store_typed ( %cnt,");
        let mut o = optimized(&text);
        let reports = analyze(&o.kernel, &o.facts, &o.lds, o.size);
        assert!(!reports[0].analyzable && reports[0].note.contains("other than"), "{:?}", reports[0]);
        let facts = std::mem::take(&mut o.facts);
        let (_, size) = run(&mut o.kernel, facts, &mut o.lds, o.size);
        assert_eq!(size, o.size);
        super::super::compile(&text).unwrap_or_else(|e| panic!("{e}"));
    }

    #[test]
    fn folded_half_loads_stay_one_vector_load() {
        // The P tile of forward attention: 32-byte rows written by 16-bit
        // stores, read as 16 consecutive f16 loads of row `lane % 16` that
        // isel folds into two b128 loads; rows 4 apart share banks.
        let mut body = String::new();
        for i in 0..8 {
            body.push_str(&format!(
                "        %sr{i} : i32 = arith.constant (  ) {{ ity = s:\"i32\", value = i:{} }}\n\
                 \x20       %row{i} : i32 = arith.add ( %half, %sr{i} )\n\
                 \x20       %h{i} : f16 = arith.cast_f32_to ( %x ) {{ target = s:\"f16\" }}\n\
                 \x20       tile.smem_store ( %p, %row{i}, %col, %h{i} ) {{ elem_type = s:\"f16\", rank = i:2 }}\n",
                2 * i
            ));
        }
        body.push_str("        tile.sync (  )\n");
        let mut prev = "%acc".to_string();
        for t in 0..16 {
            body.push_str(&format!(
                "        %ct{t} : i32 = arith.constant (  ) {{ ity = s:\"i32\", value = i:{t} }}\n\
                 \x20       %a{t} : vec<f16x1> = tile.smem_load_vN ( %p, %col, %ct{t} ) {{ elem_type = s:\"f16\", rank = i:2, vec = i:1 }}\n\
                 \x20       %e{t} : f16 = vector.extract ( %a{t} ) {{ index = i:0 }}\n\
                 \x20       %vi{t} : vec<f16x16> = vector.insert ( {prev}, %e{t} ) {{ index = i:{t} }}\n"
            ));
            prev = format!("%vi{t}");
        }
        let text = format!(
            "ckdsl.ir v1\nkernel @swz_ptile {{\n  attrs {{ max_workgroup_size = i:32 }}\n  params {{\n    %dst : ptr<f16,global>\n    %x : f32\n    %n : i32\n  }}\n  region @entry {{\n\
             \x20   %tid : i32 = gpu.thread_id (  ) {{ axis = s:\"x\" }}\n\
             \x20   %c0 : i32 = arith.constant (  ) {{ ity = s:\"i32\", value = i:0 }}\n\
             \x20   %c1 : i32 = arith.constant (  ) {{ ity = s:\"i32\", value = i:1 }}\n\
             \x20   %c16 : i32 = arith.constant (  ) {{ ity = s:\"i32\", value = i:16 }}\n\
             \x20   %p : smem<f16, [16x16]> = tile.smem_alloc (  )\n\
             \x20   %col : i32 = arith.mod ( %tid, %c16 )\n\
             \x20   %half : i32 = arith.div ( %tid, %c16 )\n\
             \x20   %z : vec<f16x16> = arith.constant_vec (  ) {{ elem = s:\"f16\", fill = f:0.0, vec = i:16 }}\n\
             \x20   %res : vec<f16x16> = scf.for ( %c0, %n, %c1, %z ) {{ iter_args = l:[ {{ name = s:\"%acc\", type = s:\"vec<f16x16>\" }} ], iv = s:\"%it\", iv_type = s:\"i32\", num_iter_args = i:1, unroll = b:false }}\n\
             \x20     region @body {{\n\
             {body}\
             \x20       tile.sync (  )\n\
             \x20       scf.yield ( {prev} ) {{ num = i:1 }}\n\
             \x20     }}\n\
             \x20   %o : i32 = arith.mul ( %tid, %c16 )\n\
             \x20   memref.global_store_vN ( %dst, %o, %res ) {{ align = i:16, elem_type = s:\"f16\", vec = i:16 }}\n\
             \x20   cf.return (  )\n  }}\n}}\n"
        );
        let (layout, o) = check_rewrite(&text);
        assert!(matches!(layout, Layout::Pad { .. }), "{layout:?}");
        let col = Collector::new(&o.kernel, &o.facts);
        assert_eq!(col.runs.len(), 1, "the 16 loads still form one run");
        let run = &col.runs[0];
        assert!(run.iter().all(|&m| col.accesses[m].addr == col.accesses[run[0]].addr));
        let compiled = super::super::compile(&text).unwrap_or_else(|e| panic!("{e}"));
        let listing = compiled.listing();
        assert_eq!(listing.matches("ds_load_b128").count(), 2, "{listing}");
        assert!(!listing.contains("ds_load_u16"), "{listing}");
    }

    /// The block reduction of `conv_direct_wgrad_dw_cols`: 128 threads store
    /// three sums each, then three threads read 128 consecutive f32 each from
    /// rows 512 bytes apart, which share banks. The reads are aligned dword
    /// quads that isel loads 16 bytes at a time, so the layout that spreads
    /// the rows keeps them 16-byte aligned (a 4-byte pad split every quad).
    #[test]
    fn dword_quad_reads_keep_their_alignment() {
        let mut body = String::new();
        for k in 0..3 {
            body += &format!(
                "    %sk{k} : i32 = arith.constant (  ) {{ ity = s:\"i32\", value = i:{} }}\n\
                 \x20   %si{k} : i32 = arith.add ( %tid, %sk{k} )\n\
                 \x20   tile.smem_store_vN_f32 ( %red, %c0, %si{k}, %x ) {{ elem_type = s:\"f32\", rank = i:2, vec = i:1 }}\n",
                128 * k
            );
        }
        body += "    tile.sync (  )\n    %lt : i1 = arith.cmp ( %tid, %c3 ) { pred = s:\"lt\" }\n    scf.if ( %lt )\n      region @then {\n\
                 \x20       %row : i32 = arith.mul ( %tid, %c128 )\n";
        let mut sum = "%e0".to_string();
        for j in 0..128 {
            body += &format!(
                "        %k{j} : i32 = arith.constant (  ) {{ ity = s:\"i32\", value = i:{j} }}\n\
                 \x20       %i{j} : i32 = arith.add ( %row, %k{j} )\n\
                 \x20       %l{j} : vec<f32x1> = tile.smem_load_vN_f32 ( %red, %c0, %i{j} ) {{ elem_type = s:\"f32\", rank = i:2, vec = i:1 }}\n\
                 \x20       %e{j} : f32 = vector.extract ( %l{j} ) {{ index = i:0 }}\n"
            );
            if j > 0 {
                body += &format!("        %s{j} : f32 = arith.fadd ( {sum}, %e{j} )\n");
                sum = format!("%s{j}");
            }
        }
        body += &format!("        memref.global_store_typed ( %dst, %tid, {sum} ) {{ align = i:4, elem_type = s:\"f32\" }}\n      }}\n");
        let text = format!(
            "ckdsl.ir v1\nkernel @swz_reduce {{\n  attrs {{ max_workgroup_size = i:128 }}\n  params {{\n    %dst : ptr<f32,global>\n    %x : f32\n  }}\n  region @entry {{\n\
             \x20   %tid : i32 = gpu.thread_id (  ) {{ axis = s:\"x\" }}\n\
             \x20   %c0 : i32 = arith.constant (  ) {{ ity = s:\"i32\", value = i:0 }}\n\
             \x20   %c3 : i32 = arith.constant (  ) {{ ity = s:\"i32\", value = i:3 }}\n\
             \x20   %c128 : i32 = arith.constant (  ) {{ ity = s:\"i32\", value = i:128 }}\n\
             \x20   %red : smem<f32, [1x384]> = tile.smem_alloc (  )\n\
             {body}\x20   cf.return (  )\n  }}\n}}\n"
        );
        let (layout, o) = check_rewrite(&text);
        assert!(layout != Layout::Identity && layout.keeps_align() >= 16, "{layout:?}");
        let col = Collector::new(&o.kernel, &o.facts);
        assert_eq!(col.runs.len() - col.quads_from, 32, "every quad of the reads");
        let listing = super::super::compile(&text).unwrap_or_else(|e| panic!("{e}")).listing();
        assert_eq!((listing.matches("ds_load_b128").count(), listing.matches("ds_load_b32").count()), (32, 0), "{listing}");
    }

    #[test]
    fn half_wave_conflicts_get_an_xor_swizzle() {
        // b32 loads where lanes 16-31 read 128 bytes past lanes 0-15: a
        // two-way conflict that a pad small enough to keep would not move by
        // 16 banks, while XORing bit 6 with bit 7 does, at no LDS cost.
        let text = "ckdsl.ir v1\nkernel @swz_halves {\n  attrs { max_workgroup_size = i:32 }\n  params {\n    %dst : ptr<f32,global>\n    %n : i32\n  }\n  region @entry {\n\
             \x20   %tid : i32 = gpu.thread_id (  ) { axis = s:\"x\" }\n\
             \x20   %c0 : i32 = arith.constant (  ) { ity = s:\"i32\", value = i:0 }\n\
             \x20   %c1 : i32 = arith.constant (  ) { ity = s:\"i32\", value = i:1 }\n\
             \x20   %c16 : i32 = arith.constant (  ) { ity = s:\"i32\", value = i:16 }\n\
             \x20   %c32 : i32 = arith.constant (  ) { ity = s:\"i32\", value = i:32 }\n\
             \x20   %lds : smem<f32, [256]> = tile.smem_alloc (  )\n\
             \x20   %x : f32 = arith.constant (  ) { ity = s:\"f32\", value = f:1.0 }\n\
             \x20   tile.smem_store ( %lds, %tid, %x ) { elem_type = s:\"f32\", rank = i:1 }\n\
             \x20   tile.sync (  )\n\
             \x20   %lo : i32 = arith.mod ( %tid, %c16 )\n\
             \x20   %hi : i32 = arith.div ( %tid, %c16 )\n\
             \x20   %hio : i32 = arith.mul ( %hi, %c32 )\n\
             \x20   %i : i32 = arith.add ( %lo, %hio )\n\
             \x20   %r : f32 = scf.for ( %c0, %n, %c1, %x ) { iter_args = l:[ { name = s:\"%acc\", type = s:\"f32\" } ], iv = s:\"%it\", iv_type = s:\"i32\", num_iter_args = i:1, unroll = b:false }\n\
             \x20     region @body {\n\
             \x20       %v : vec<f32x1> = tile.smem_load_vN ( %lds, %i ) { elem_type = s:\"f32\", rank = i:1, vec = i:1 }\n\
             \x20       %e : f32 = vector.extract ( %v ) { index = i:0 }\n\
             \x20       %s : f32 = arith.fadd ( %acc, %e )\n\
             \x20       scf.yield ( %s ) { num = i:1 }\n\
             \x20     }\n\
             \x20   memref.global_store_typed ( %dst, %tid, %r ) { align = i:4, elem_type = s:\"f32\" }\n\
             \x20   cf.return (  )\n  }\n}\n";
        let (layout, o) = check_rewrite(text);
        assert_eq!(layout, Layout::Xor { low: 6, bits: 1, key: 7 });
        assert_eq!(o.size, 1024);
        super::super::compile(text).unwrap_or_else(|e| panic!("{e}"));
    }

    #[test]
    fn grown_allocations_never_overlap_live_neighbours() {
        let (a, b, c) = (ValueId(1), ValueId(2), ValueId(3));
        // `c` reuses the bytes of `a` and `b` (disjoint lifetimes): it grows
        // in place over them.
        let pool = [(a, 0, 2560), (b, 2560, 2560), (c, 0, 2048)];
        let (out, size) = relayout(&pool, c, 2560).unwrap();
        assert_eq!((out[2], size), ((c, 0, 2560), 5120));
        // `a` growing into `b`, which it never overlapped: `a` moves past
        // the end of the pool instead.
        let (out, size) = relayout(&pool, a, 2880).unwrap();
        assert_eq!((out[0], size), ((a, 5120, 2880), 8000));
        // The last allocation simply grows.
        let (out, size) = relayout(&pool, b, 2880).unwrap();
        assert_eq!((out[1], size), ((b, 2560, 2880), 5440));
    }

    #[test]
    fn layouts_are_bijective_and_keep_chunks() {
        let mut all = Vec::new();
        for stride in 5..=12 {
            for unit in [2, 4, 8, 16] {
                for degree in [2, 4, 8, 32] {
                    candidates(stride, degree, unit, unit, &mut all);
                }
            }
        }
        assert!(all.iter().any(|l| matches!(l, Layout::Pad { .. })) && all.iter().any(|l| matches!(l, Layout::Xor { .. })));
        for layout in all {
            let size = 8192u32;
            let mut seen = FxHashSet::default();
            let chunk = layout.chunk();
            for a in 0..size as i64 {
                let t = layout.apply(a);
                assert!(t >= 0 && t < layout.size(size) as i64, "{layout:?} maps {a} to {t}");
                assert!(seen.insert(t), "{layout:?} is not injective at {a}");
                assert_eq!(layout.apply(a + layout.period()), t + layout.apply(layout.period()), "{layout:?}");
                if a % chunk == 0 {
                    assert!(layout.contiguous(a, chunk));
                    assert!((0..chunk).all(|i| layout.apply(a + i) == t + i), "{layout:?} splits the chunk at {a}");
                }
            }
        }
    }
}
