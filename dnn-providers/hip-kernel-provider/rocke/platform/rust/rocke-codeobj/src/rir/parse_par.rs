// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

//! Chunked parallel parse of large inputs.
//!
//! On worker threads: the body is cut at line boundaries into chunks, which
//! threads claim in order. A first pass over a chunk scans its lines, builds
//! every op with its operands unresolved, and enters each definition into one
//! shared lock-free name table. A second pass resolves the chunk's operands
//! through the table, checks that each is in scope, and numbers the chunk's
//! values.
//!
//! Sequential: the header and parameters; a walk over the few structural
//! lines (region headers, closing braces, ops with regions) with an explicit
//! stack, which validates the nesting and fixes where each chunk's value
//! numbers and name bytes start; and stitching each chunk's ops into regions in
//! text order, which the calling thread does as chunks finish their second
//! pass.
//!
//! Every name must be defined once. Anything the sequential parser would have
//! to decide (a name defined again, any error, an unusual shape) makes the
//! parse return `None`, and the caller parses the whole input sequentially, so
//! the kernel or the error is always the sequential parser's.
//!
//! The shared state owns all it holds, so the workers can be scoped threads
//! (`parse`), a session's persistent threads (`parse_pooled`), or threads that
//! scanned chunks of a stream as they arrived (`stream`). The input reaches it
//! through a `Source`: one string, or a string per chunk.

use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use super::ast::{Kernel, Op, OpKind, Operands, Param, Region, Results, Scalar, Type, ValueDef, ValueId};
use super::hashcons::{key_hash, ConstTable};
use super::parse::{expected_regions, for_args, name_hash, prologue, scan_op, shareable_before, Lines};
use super::scan::Level;
use super::scan_line;

/// Operand placeholder until the second pass.
const UNRESOLVED: ValueId = ValueId(u32::MAX);
/// Attribute nesting a worker parses itself. Deeper lines go to the
/// sequential parser, which runs on the caller's stack as before.
const ATTR_DEPTH: u32 = 64;
/// Worker stack; workers recurse only through attributes, at most `ATTR_DEPTH`.
pub(super) const WORKER_STACK: usize = 512 << 10;
/// Default chunking: enough chunks per thread to balance threads that start
/// late, but not so small that per-chunk work dominates.
const CHUNKS_PER_THREAD: usize = 4;
const MIN_CHUNK_BYTES: usize = 16 << 10;
/// A table entry names a definition by chunk and index, plus one, in 32 bits.
const INDEX_BITS: u32 = 20;
pub(super) const MAX_CHUNKS: usize = (1 << (32 - INDEX_BITS)) - 1;
const MAX_DEFS: usize = (1 << INDEX_BITS) - 2;
const MAX_PROBES: usize = 1 << 12;

/// Where the parse reads each chunk: a string holding the chunk at
/// `bounds[k]` and at least every line after it up to the next one that is
/// not blank or a comment, and the offset of that string in the whole input.
/// Chunk 0 holds the header and parameters.
pub(super) trait Source: Send + Sync {
    fn segment(&self, chunk: usize) -> (&str, usize);
}

/// The whole input as one string.
impl Source for str {
    fn segment(&self, _: usize) -> (&str, usize) {
        (self, 0)
    }
}

/// One string per chunk, with the offset of each in the input.
pub(super) struct Segments(pub(super) Vec<(Arc<str>, usize)>);

impl Source for Segments {
    fn segment(&self, chunk: usize) -> (&str, usize) {
        let (text, base) = &self.0[chunk];
        (text, *base)
    }
}

/// A name, by byte range in its chunk's segment, with its hash.
#[derive(Clone, Copy)]
struct Name {
    at: u32,
    len: u32,
    hash: u32,
}

impl Name {
    fn of(text: &str, name: &str) -> Name {
        Name { at: offset(text, name), len: name.len() as u32, hash: name_hash(name) }
    }

    fn str(self, text: &str) -> &str {
        &text[self.at as usize..(self.at + self.len) as usize]
    }
}

/// Offset of `part`, a slice of `text`, in `text`.
fn offset(text: &str, part: &str) -> u32 {
    let (start, base) = (part.as_ptr() as usize, text.as_ptr() as usize);
    debug_assert!(start >= base && start + part.len() <= base + text.len());
    (start - base) as u32
}

/// `"name"` in `line`, as the slice between the quotes: where an `scf.for`
/// names a block argument without escapes.
fn quoted_in<'t>(line: &'t str, name: &str) -> Option<&'t str> {
    if name.is_empty() {
        return None;
    }
    let b = line.as_bytes();
    line.match_indices(name)
        .find(|&(at, _)| at > 0 && b[at - 1] == b'"' && b.get(at + name.len()) == Some(&b'"'))
        .map(|(at, _)| &line[at..at + name.len()])
}

/// How a definition gets its value number and its scope.
#[derive(Clone, Copy)]
enum Kind {
    /// Defined at its line: the result of an op without regions; the value is
    /// the chunk's `slot`-th such definition.
    Line(u32),
    /// A block argument of the `scf.for` that is the chunk's `structure`-th op
    /// with regions; visible in its region.
    ForArg { slot: u32, structure: u32 },
    /// Result `k` of the chunk's `structure`-th op with regions, defined after
    /// the op's last region.
    After { structure: u32, k: u32 },
    /// A repeated constant (see `hashcons`): another name, at its own line and
    /// region, for the value of the chunk's `Line` definition with this index.
    Alias(u32),
}

#[derive(Clone, Copy)]
struct Def {
    name: Name,
    /// Offset of the defining line in the input.
    pos: u32,
    /// Innermost region at the defining line: the index of a region opened in
    /// this chunk, or `-1 - k` for the region `k` levels out from the
    /// innermost one open when the chunk starts.
    region: i32,
    kind: Kind,
}

/// An op followed by a region header.
struct Structured {
    regions: usize,
    results: Vec<(Name, Type)>,
}

#[derive(Clone, Copy)]
enum Event {
    /// A `region @` line after `ops` ops of the chunk; `ok` if it ends with `{`.
    Header { ops: u32, ok: bool },
    /// A `}` line at input offset `pos`, after `ops` ops and `defs` line
    /// definitions of the chunk, and whether the next line is a region header.
    Close { pos: u32, ops: u32, defs: u32, header_next: bool },
    /// The chunk's op `op` is its `index`-th op with regions.
    Structured { index: u32, op: u32 },
}

/// What the first pass learns about a chunk, shared read-only afterwards.
#[derive(Default)]
pub(super) struct Info {
    defs: Vec<Def>,
    structs: Vec<Structured>,
    events: Vec<Event>,
    ops: u32,
    line_defs: u32,
    line_bytes: usize,
    headers: u32,
}

/// The first pass's ops and what the second pass needs to finish them.
#[derive(Default)]
pub(super) struct Work {
    ops: Vec<Op>,
    op_pos: Vec<u32>,
    op_region: Vec<i32>,
    /// End of each op's definitions in `Info::defs`.
    op_defs: Vec<u32>,
    /// Operand names, in op order.
    uses: Vec<Name>,
    /// Types of the line definitions, by slot.
    types: Vec<Type>,
}

/// A chunk's finished ops, and the names (with the chunk whose segment holds
/// each) and types of its values in value order. (A value's name range is an
/// offset into the kernel's name storage, so the values themselves are made
/// in order on the calling thread.)
struct Out {
    ops: Vec<Op>,
    values: Vec<(u32, Name, Type)>,
}

/// Lock-free open-addressing table from names to definitions. A slot holds
/// the name hash and a definition handle (chunk and index) plus one.
struct Table {
    mask: usize,
    slots: Vec<AtomicU64>,
}

impl Table {
    fn new(len: usize) -> Self {
        Table { mask: len - 1, slots: (0..len).map(|_| AtomicU64::new(0)).collect() }
    }

    /// Enters `handle` under `key`; `None` if the key is already there or the
    /// table is too full.
    fn insert<'s>(&self, key: &str, hash: u32, handle: u32, key_of: impl Fn(u32) -> Option<&'s str>) -> Option<()> {
        let entry = (hash as u64) << 32 | (handle as u64 + 1);
        let mut i = hash as usize & self.mask;
        for _ in 0..MAX_PROBES {
            let mut slot = self.slots[i].load(Ordering::Acquire);
            if slot == 0 {
                match self.slots[i].compare_exchange(0, entry, Ordering::AcqRel, Ordering::Acquire) {
                    Ok(_) => return Some(()),
                    Err(now) => slot = now,
                }
            }
            if (slot >> 32) as u32 == hash && key_of(slot as u32 - 1)? == key {
                return None;
            }
            i = (i + 1) & self.mask;
        }
        None
    }

    fn find<'s>(&self, key: &str, hash: u32, key_of: impl Fn(u32) -> Option<&'s str>) -> Option<u32> {
        let mut i = hash as usize & self.mask;
        for _ in 0..MAX_PROBES {
            let slot = self.slots[i].load(Ordering::Acquire);
            if slot == 0 {
                return None;
            }
            if (slot >> 32) as u32 == hash && key_of(slot as u32 - 1)? == key {
                return Some(slot as u32 - 1);
            }
            i = (i + 1) & self.mask;
        }
        None
    }
}

struct StructInfo {
    first_id: u32,
    /// Offset of the `}` after which the results are defined.
    end: u32,
    first_region: u32,
}

struct RegionInfo {
    /// Regions in its subtree, itself included; ids are in pre-order.
    size: u32,
    /// Ops directly in it.
    ops: u32,
}

/// The results of a chunk's `index`-th op with regions, defined at a `}` in
/// another (or the same) chunk after `defs` of that chunk's line definitions.
struct Emit {
    defs: u32,
    chunk: u32,
    index: u32,
    count: u32,
}

/// The structure of the whole body. Indexed by chunk, 0 being the parameters.
struct Walk {
    vbase: Vec<u32>,
    nbase: Vec<usize>,
    sbase: Vec<u32>,
    rbase: Vec<u32>,
    /// Regions open where each chunk starts, outermost first.
    inherited: Vec<Vec<u32>>,
    emits: Vec<Vec<Emit>>,
    structs: Vec<StructInfo>,
    regions: Vec<RegionInfo>,
}

impl Walk {
    fn region(&self, chunk: usize, region: i32) -> Option<u32> {
        if region >= 0 {
            return Some(self.rbase[chunk] + region as u32);
        }
        let open = &self.inherited[chunk];
        let up = (-1 - region) as usize;
        open.len().checked_sub(up + 1).map(|i| open[i])
    }

    /// Whether region `inner` is `outer` or nested in it.
    fn within(&self, outer: u32, inner: u32) -> bool {
        inner.wrapping_sub(outer) < self.regions[outer as usize].size
    }

    fn line_id(&self, chunk: usize, slot: u32) -> u32 {
        let before: u32 = self.emits[chunk].iter().take_while(|e| e.defs <= slot).map(|e| e.count).sum();
        self.vbase[chunk] + slot + before
    }
}

/// The parse's state, shared by the threads working on it.
pub(super) struct Shared {
    /// Byte range of each chunk in its segment; chunk 0 holds the parameters.
    bounds: Vec<(usize, usize)>,
    infos: Vec<OnceLock<Info>>,
    work: Vec<Mutex<Option<Work>>>,
    out: Vec<Mutex<Option<Out>>>,
    claimed: Vec<AtomicBool>,
    resolved: Vec<AtomicBool>,
    table: Table,
    /// Definitions enter the table when their chunk is scanned, or, for
    /// chunks scanned before the table could be sized (a stream), in a pass
    /// of their own once all are scanned.
    enter_later: bool,
    walk: OnceLock<Option<Walk>>,
    next_scan: AtomicUsize,
    scanned: AtomicUsize,
    next_enter: AtomicUsize,
    entered: AtomicUsize,
    /// Workers that have begun running.
    started: AtomicUsize,
    failed: AtomicBool,
    /// A worker panicked; the caller panics too, as with scoped threads.
    panicked: AtomicBool,
    /// Repeated constants are bound to the earlier one (see `hashcons`).
    hashcons: bool,
    /// Vector level of the line scans (`scan`).
    level: Level,
}

/// Sets `failed` (and `panicked`) if its thread unwinds, so that no thread
/// waits for work a panicked thread claimed.
struct FailOnPanic<'a>(&'a Shared);

impl Drop for FailOnPanic<'_> {
    fn drop(&mut self) {
        if std::thread::panicking() {
            self.0.panicked.store(true, Ordering::Release);
            self.0.failed.store(true, Ordering::Release);
        }
    }
}

/// The first pass over the chunk `seg[start..end]`, where `seg` is at input
/// offset `base` and holds the lines after the chunk as `Source` requires;
/// `None` if the chunk needs the sequential parser.
pub(super) fn scan_chunk<'s>(
    seg: &'s str,
    base: usize,
    start: usize,
    end: usize,
    hashcons: bool,
    level: Level,
) -> Option<(Info, Work)> {
    let mut lines = Lines::new(seg, start, end).scanning(level);
    let mut info = Info::default();
    let mut work = Work::default();
    let estimate = (end - start) / 64;
    work.ops.reserve(estimate);
    work.op_pos.reserve(estimate);
    work.op_region.reserve(estimate);
    work.op_defs.reserve(estimate);
    work.uses.reserve(estimate * 3 / 2);
    work.types.reserve(estimate);
    info.defs.reserve(estimate);
    let next_line = |lines: &mut Lines<'s>| -> Option<&'s str> {
        match lines.peek() {
            Some(next) => Some(next),
            None => Lines::new(seg, end, seg.len()).peek(),
        }
    };
    let header_next = |lines: &mut Lines<'s>| next_line(lines).is_some_and(|next| next.starts_with("region @"));
    let mut open: Vec<u32> = Vec::new();
    let mut closed_outer = 0i32;
    let mut pending: Vec<(&'s str, Type)> = Vec::new();
    // Constants of this chunk by key (see `hashcons`). A scope is a region
    // opened in the chunk (its index, open while `local_open` says so) or
    // the region `j` levels out from the one open at the chunk's start
    // (`INHERITED | j`, open until `closed_outer` passes `j`).
    const INHERITED: u32 = 1 << 31;
    let mut consts = ConstTable::new();
    let mut local_open: Vec<bool> = Vec::new();
    while let Some(line) = lines.next_line() {
        let pos = (base + offset(seg, line) as usize) as u32;
        if line == "}" {
            let next = header_next(&mut lines);
            let ops = work.ops.len() as u32;
            info.events.push(Event::Close { pos, ops, defs: info.line_defs, header_next: next });
            match open.pop() {
                Some(r) => local_open[r as usize] = false,
                None => closed_outer += 1,
            }
            continue;
        }
        if line.starts_with("region @") {
            info.events.push(Event::Header { ops: work.ops.len() as u32, ok: line.ends_with('{') });
            open.push(info.headers);
            local_open.push(true);
            info.headers += 1;
            continue;
        }
        let region = open.last().map_or(-1 - closed_outer, |&r| r as i32);
        let masks = scan_line::masks(level, line);
        let constant = if hashcons { scan_line::const_key_with(line, masks.as_ref()) } else { None };
        let constant = constant.filter(|_| shareable_before(next_line(&mut lines)));
        let scope = open.last().copied().unwrap_or(INHERITED | closed_outer as u32);
        let visible = |s: u32| match s & INHERITED {
            0 => local_open[s as usize],
            _ => (s & !INHERITED) as i32 >= closed_outer,
        };
        if let Some((name, key)) = constant {
            if let Some(target) = consts.lookup(key, key_hash(key), visible) {
                info.defs.push(Def { name: Name::of(seg, name), pos, region, kind: Kind::Alias(target.0) });
                continue;
            }
        }
        let mut operands = Operands::new();
        let uses = &mut work.uses;
        let scanned = scan_op(
            line,
            masks.as_ref(),
            &mut operands,
            &mut pending,
            |name, _| {
                uses.push(Name::of(seg, name));
                Ok(UNRESOLVED)
            },
            ATTR_DEPTH,
        )
        .ok()?;
        let structured = header_next(&mut lines);
        let op_index = work.ops.len() as u32;
        let op = Op {
            kind: scanned.kind,
            results: Results::new(),
            operands,
            attrs: scanned.attrs,
            regions: Vec::new(),
            block_args: Vec::new(),
        };
        let regions = expected_regions(op.kind);
        if structured != (regions > 0) {
            return None;
        }
        let mut line_def = |info: &mut Info, name: &'s str, kind: Kind, ty: Type| {
            info.defs.push(Def { name: Name::of(seg, name), pos, region, kind });
            info.line_defs += 1;
            info.line_bytes += name.len();
            work.types.push(ty);
        };
        if structured {
            let structure = info.structs.len() as u32;
            if op.kind == OpKind::For {
                let (iv, iter_args) = for_args(&op, pending.len()).ok()?;
                let args = std::iter::once((iv, Type::Scalar(Scalar::I32))).chain(iter_args);
                for (name, ty) in args {
                    let slot = info.line_defs;
                    line_def(&mut info, quoted_in(line, &name)?, Kind::ForArg { slot, structure }, ty);
                }
            }
            let results: Vec<(Name, Type)> = pending.drain(..).map(|(name, ty)| (Name::of(seg, name), ty)).collect();
            for (k, &(name, _)) in results.iter().enumerate() {
                info.defs.push(Def { name, pos, region, kind: Kind::After { structure, k: k as u32 } });
            }
            info.structs.push(Structured { regions, results });
            info.events.push(Event::Structured { index: structure, op: op_index });
        } else {
            for (name, ty) in pending.drain(..) {
                let slot = info.line_defs;
                line_def(&mut info, name, Kind::Line(slot), ty);
            }
        }
        work.ops.push(op);
        work.op_pos.push(pos);
        work.op_region.push(region);
        work.op_defs.push(info.defs.len() as u32);
        if let Some((_, key)) = constant {
            // The table's value is the index of the constant's definition.
            consts.insert(key, key_hash(key), ValueId(info.defs.len() as u32 - 1), scope);
        }
    }
    info.ops = work.ops.len() as u32;
    if info.defs.len() > MAX_DEFS {
        return None;
    }
    Some((info, work))
}

impl Shared {
    /// State for chunks at `bounds` (entry 0 for the parameters, whose
    /// definitions `params` holds) and a table for an input of `len` bytes
    /// after the parameters.
    pub(super) fn new(bounds: Vec<(usize, usize)>, params: Info, len: usize, enter_later: bool, hashcons: bool) -> Option<Shared> {
        let n = bounds.len() - 1;
        if n > MAX_CHUNKS || params.defs.len() > MAX_DEFS {
            return None;
        }
        // Sized for the corpus's line lengths at under half full, and kept below
        // the size the heap serves from fresh pages for bodies up to about 1 MiB.
        let table_len = (len / 36 + 2 * params.defs.len()).next_power_of_two().max(1024);
        let sh = Shared {
            bounds,
            infos: (0..=n).map(|_| OnceLock::new()).collect(),
            work: (0..=n).map(|_| Mutex::new(None)).collect(),
            out: (0..=n).map(|_| Mutex::new(None)).collect(),
            claimed: (0..=n).map(|_| AtomicBool::new(false)).collect(),
            resolved: (0..=n).map(|_| AtomicBool::new(false)).collect(),
            table: Table::new(table_len),
            enter_later,
            walk: OnceLock::new(),
            next_scan: AtomicUsize::new(1),
            scanned: AtomicUsize::new(0),
            next_enter: AtomicUsize::new(0),
            entered: AtomicUsize::new(0),
            started: AtomicUsize::new(0),
            failed: AtomicBool::new(false),
            panicked: AtomicBool::new(false),
            hashcons,
            level: Level::detect(),
        };
        sh.infos[0].set(params).ok()?;
        Some(sh)
    }

    /// Hands over the first pass of chunks `1..=scans.len()`, done elsewhere.
    pub(super) fn scanned_already(&self, scans: Vec<(Info, Work)>) -> Option<()> {
        let count = scans.len();
        for (k, (info, work)) in scans.into_iter().enumerate() {
            self.infos[k + 1].set(info).ok()?;
            *self.work[k + 1].lock().ok()? = Some(work);
        }
        self.next_scan.store(count + 1, Ordering::Relaxed);
        self.scanned.store(count, Ordering::Release);
        Some(())
    }

    fn chunks(&self) -> usize {
        self.bounds.len() - 1
    }

    pub(super) fn fail(&self) {
        self.failed.store(true, Ordering::Release);
    }

    fn failed(&self) -> bool {
        self.failed.load(Ordering::Acquire)
    }

    pub(super) fn panicked(&self) -> bool {
        self.panicked.load(Ordering::Acquire)
    }

    fn claim(&self, next: &AtomicUsize) -> Option<usize> {
        let k = next.fetch_add(1, Ordering::Relaxed);
        (k <= self.chunks()).then_some(k)
    }

    /// Resolves chunk `k` unless another thread has claimed it; `false` on failure.
    fn resolve_unclaimed<S: Source + ?Sized>(&self, src: &S, w: &Walk, k: usize) -> bool {
        if self.claimed[k].swap(true, Ordering::AcqRel) {
            return true;
        }
        if self.failed() || self.resolve(src, w, k).is_none() {
            self.fail();
            return false;
        }
        true
    }

    /// Waits until `done`, or `false` if the parse failed meanwhile.
    fn wait(&self, done: impl Fn() -> bool) -> bool {
        let mut spins = 0u32;
        loop {
            if done() {
                return true;
            }
            if self.failed() {
                return false;
            }
            spins += 1;
            if spins < 64 {
                std::hint::spin_loop();
            } else {
                std::thread::yield_now();
            }
        }
    }

    fn def(&self, handle: u32) -> Option<&Def> {
        let chunk = (handle >> INDEX_BITS) as usize;
        self.infos.get(chunk)?.get()?.defs.get((handle & ((1 << INDEX_BITS) - 1)) as usize)
    }

    /// The name a table handle stands for.
    fn key_of<'s, S: Source + ?Sized>(&self, src: &'s S, handle: u32) -> Option<&'s str> {
        let def = self.def(handle)?;
        Some(def.name.str(src.segment((handle >> INDEX_BITS) as usize).0))
    }

    fn enter<S: Source + ?Sized>(&self, src: &S, chunk: usize) -> Option<()> {
        let info = self.infos[chunk].get()?;
        let seg = src.segment(chunk).0;
        for (i, def) in info.defs.iter().enumerate() {
            let handle = (chunk as u32) << INDEX_BITS | i as u32;
            self.table.insert(def.name.str(seg), def.name.hash, handle, |h| self.key_of(src, h))?;
        }
        Some(())
    }

    /// Scans chunks until none is left, then waits for the other threads'
    /// (and, when definitions enter later, enters them); the chunks this
    /// thread scanned, or `None` if the parse failed.
    fn scan_all<S: Source + ?Sized>(&self, src: &S) -> Option<Vec<usize>> {
        let mut mine = Vec::new();
        while let Some(k) = self.claim(&self.next_scan) {
            if self.failed() {
                return None;
            }
            if self.scan(src, k).is_none() {
                self.fail();
                return None;
            }
            mine.push(k);
        }
        if !self.wait(|| self.scanned.load(Ordering::Acquire) == self.chunks()) {
            return None;
        }
        if self.enter_later {
            while let Some(k) = self.claim(&self.next_enter) {
                if self.failed() || self.enter(src, k).is_none() {
                    self.fail();
                    return None;
                }
                self.entered.fetch_add(1, Ordering::Release);
            }
            if !self.wait(|| self.entered.load(Ordering::Acquire) == self.chunks() + 1) {
                return None;
            }
        }
        Some(mine)
    }

    /// First pass over chunk `k`.
    fn scan<S: Source + ?Sized>(&self, src: &S, k: usize) -> Option<()> {
        let (seg, base) = src.segment(k);
        let (start, end) = self.bounds[k];
        let (info, work) = scan_chunk(seg, base, start, end, self.hashcons, self.level)?;
        self.infos[k].set(info).ok()?;
        if !self.enter_later {
            self.enter(src, k)?;
        }
        *self.work[k].lock().ok()? = Some(work);
        self.scanned.fetch_add(1, Ordering::Release);
        Some(())
    }

    fn walk(&self) -> Option<&Walk> {
        let walk = self.walk.get_or_init(|| self.compute_walk()).as_ref();
        if walk.is_none() {
            self.fail();
        }
        walk
    }

    /// Follows the structural events of all chunks in order, as the
    /// sequential parser follows the lines: checks that regions nest and
    /// attach to ops that take them, and numbers values and regions.
    fn compute_walk(&self) -> Option<Walk> {
        #[derive(Clone, Copy)]
        enum State {
            Start,
            InRegion,
            ExpectHeader,
            /// The body has closed after this many ops; the kernel's `}` is next.
            AfterBody(u64),
            Done(u64),
        }
        let n = self.chunks();
        let infos: Vec<&Info> = self.infos.iter().map(OnceLock::get).collect::<Option<_>>()?;
        let mut w = Walk {
            vbase: vec![0; n + 2],
            nbase: vec![0; n + 2],
            sbase: vec![0; n + 2],
            rbase: vec![0; n + 2],
            inherited: vec![Vec::new(); n + 1],
            emits: (0..=n).map(|_| Vec::new()).collect(),
            structs: Vec::new(),
            regions: Vec::new(),
        };
        for (k, info) in infos.iter().enumerate().skip(1) {
            w.sbase[k + 1] = w.sbase[k] + info.structs.len() as u32;
            w.rbase[k + 1] = w.rbase[k] + info.headers;
        }
        w.structs = (0..w.sbase[n + 1]).map(|_| StructInfo { first_id: 0, end: 0, first_region: 0 }).collect();
        w.vbase[1] = infos[0].line_defs;
        w.nbase[1] = infos[0].line_bytes;
        let mut state = State::Start;
        // Open regions with the global op index where their current run of ops started.
        let mut stack: Vec<(u32, u64)> = Vec::new();
        // Ops whose regions are being walked: global structure, chunk, index, regions so far.
        let mut owners: Vec<(u32, usize, u32, usize)> = Vec::new();
        let mut obase = 0u64;
        for k in 1..=n {
            let info = infos[k];
            w.inherited[k] = stack.iter().map(|&(r, _)| r).collect();
            let (mut emitted, mut emitted_bytes) = (0u32, 0usize);
            for &event in &info.events {
                match (state, event) {
                    (State::Start, Event::Header { ops, ok }) => {
                        if !ok || obase + ops as u64 != 0 {
                            return None;
                        }
                        w.regions.push(RegionInfo { size: 0, ops: 0 });
                        stack.push((0, 0));
                        state = State::InRegion;
                    }
                    (State::InRegion, Event::Structured { index, .. }) => {
                        owners.push((w.sbase[k] + index, k, index, 0));
                        state = State::ExpectHeader;
                    }
                    (State::ExpectHeader, Event::Header { ops, ok }) => {
                        if !ok {
                            return None;
                        }
                        let at = obase + ops as u64;
                        let (parent, start) = stack.last_mut()?;
                        w.regions[*parent as usize].ops += (at - *start) as u32;
                        let g = w.regions.len() as u32;
                        w.regions.push(RegionInfo { size: 0, ops: 0 });
                        let owner = owners.last_mut()?;
                        if owner.3 == 0 {
                            w.structs[owner.0 as usize].first_region = g;
                        }
                        owner.3 += 1;
                        stack.push((g, at));
                        state = State::InRegion;
                    }
                    (State::InRegion, Event::Close { pos, ops, defs, header_next }) => {
                        let at = obase + ops as u64;
                        let (g, start) = stack.pop()?;
                        let size = w.regions.len() as u32 - g;
                        let region = &mut w.regions[g as usize];
                        region.ops += (at - start) as u32;
                        region.size = size;
                        let Some((_, parent_start)) = stack.last_mut() else {
                            state = State::AfterBody(at);
                            continue;
                        };
                        *parent_start = at;
                        if header_next {
                            state = State::ExpectHeader;
                            continue;
                        }
                        let (structure, chunk, index, seen) = owners.pop()?;
                        let st = &infos[chunk].structs[index as usize];
                        if seen != st.regions {
                            return None;
                        }
                        let count = st.results.len() as u32;
                        let s = &mut w.structs[structure as usize];
                        s.first_id = w.vbase[k] + defs + emitted;
                        s.end = pos;
                        w.emits[k].push(Emit { defs, chunk: chunk as u32, index, count });
                        emitted += count;
                        emitted_bytes += st.results.iter().map(|(name, _)| name.len as usize).sum::<usize>();
                        state = State::InRegion;
                    }
                    (State::AfterBody(body_end), Event::Close { ops, .. }) => {
                        if obase + ops as u64 != body_end {
                            return None;
                        }
                        state = State::Done(body_end);
                    }
                    _ => return None,
                }
            }
            obase += info.ops as u64;
            w.vbase[k + 1] = w.vbase[k] + info.line_defs + emitted;
            w.nbase[k + 1] = w.nbase[k] + info.line_bytes + emitted_bytes;
        }
        match state {
            State::Done(body_end) if body_end == obase => Some(w),
            _ => None,
        }
    }

    /// Resolves `name`, used in chunk `chunk` at offset `pos` in region `at`,
    /// to the value it names there: the definition must come before the use,
    /// and the use must be inside the definition's scope.
    fn lookup<S: Source + ?Sized>(&self, src: &S, w: &Walk, chunk: usize, name: Name, pos: u32, at: u32) -> Option<ValueId> {
        let key = name.str(src.segment(chunk).0);
        let handle = self.table.find(key, name.hash, |h| self.key_of(src, h))?;
        let chunk = (handle >> INDEX_BITS) as usize;
        let def = self.def(handle)?;
        let (defined, scope, id) = match def.kind {
            Kind::Line(slot) => (def.pos, w.region(chunk, def.region)?, w.line_id(chunk, slot)),
            Kind::Alias(target) => {
                let Kind::Line(slot) = self.infos[chunk].get()?.defs.get(target as usize)?.kind else {
                    return None;
                };
                (def.pos, w.region(chunk, def.region)?, w.line_id(chunk, slot))
            }
            Kind::ForArg { slot, structure } => {
                (def.pos, w.structs[(w.sbase[chunk] + structure) as usize].first_region, w.line_id(chunk, slot))
            }
            Kind::After { structure, k } => {
                let s = &w.structs[(w.sbase[chunk] + structure) as usize];
                (s.end, w.region(chunk, def.region)?, s.first_id + k)
            }
        };
        (defined < pos && w.within(scope, at)).then_some(ValueId(id))
    }

    /// Defines the results of the op `emit` names, at their place in the
    /// chunk's values.
    fn emit(&self, emit: &Emit, values: &mut Vec<(u32, Name, Type)>) -> Option<()> {
        let st = self.infos[emit.chunk as usize].get()?.structs.get(emit.index as usize)?;
        values.extend(st.results.iter().map(|(name, ty)| (emit.chunk, *name, ty.clone())));
        Some(())
    }

    /// Second pass over chunk `k`.
    fn resolve<S: Source + ?Sized>(&self, src: &S, w: &Walk, k: usize) -> Option<()> {
        let info = self.infos[k].get()?;
        let Work { mut ops, op_pos, op_region, op_defs, uses, types } = self.work[k].lock().ok()?.take()?;
        let count = (w.vbase[k + 1] - w.vbase[k]) as usize;
        let mut values = Vec::with_capacity(count);
        let mut types = types.into_iter();
        let emits = &w.emits[k];
        let mut e = 0;
        let mut next = w.vbase[k];
        let mut used = uses.iter();
        let mut d = 0;
        for (o, op) in ops.iter_mut().enumerate() {
            let pos = op_pos[o];
            let at = w.region(k, op_region[o])?;
            for operand in op.operands.iter_mut() {
                *operand = self.lookup(src, w, k, *used.next()?, pos, at)?;
            }
            while d < op_defs[o] as usize {
                let def = &info.defs[d];
                d += 1;
                match def.kind {
                    Kind::Line(slot) | Kind::ForArg { slot, .. } => {
                        while e < emits.len() && emits[e].defs <= slot {
                            self.emit(&emits[e], &mut values)?;
                            next += emits[e].count;
                            e += 1;
                        }
                        let id = ValueId(next);
                        next += 1;
                        values.push((k as u32, def.name, types.next()?));
                        if matches!(def.kind, Kind::Line(_)) {
                            op.results.push(id);
                        } else {
                            op.block_args.push(id);
                        }
                    }
                    Kind::After { structure, k: r } => {
                        op.results.push(ValueId(w.structs[(w.sbase[k] + structure) as usize].first_id + r));
                    }
                    Kind::Alias(_) => {}
                }
            }
        }
        for emit in &emits[e..] {
            self.emit(emit, &mut values)?;
            next += emit.count;
        }
        if next != w.vbase[k + 1] || values.len() != count {
            return None;
        }
        *self.out[k].lock().ok()? = Some(Out { ops, values });
        self.resolved[k].store(true, Ordering::Release);
        Some(())
    }

    /// A worker resolves the chunks it scanned first, while their lines,
    /// names and table entries are still in its cache, then any left, in
    /// text order as the stitching needs them.
    pub(super) fn worker<S: Source + ?Sized>(&self, src: &S) {
        self.started.fetch_add(1, Ordering::Release);
        let _guard = FailOnPanic(self);
        let Some(mine) = self.scan_all(src) else { return };
        let Some(w) = self.walk() else { return };
        for k in mine.into_iter().chain(1..=self.chunks()) {
            if !self.resolve_unclaimed(src, w, k) {
                return;
            }
        }
    }

    /// The calling thread: scans like a worker, then stitches the chunks'
    /// ops into regions in text order, resolving a chunk itself when no
    /// thread has claimed it yet.
    pub(super) fn lead<S: Source + ?Sized>(&self, src: &S, header: Header) -> Option<Kernel> {
        let _guard = FailOnPanic(self);
        self.scan_all(src)?;
        let w = self.walk()?;
        let Header { name, max_workgroup_size, waves_per_eu, agpr_limit, params, mut values, mut names } = header;
        let n = self.chunks();
        values.reserve_exact(w.vbase[n + 1] as usize - values.len());
        names.reserve_exact(w.nbase[n + 1] - names.len());
        // Open regions, innermost last, each with the index of its op in the enclosing region.
        let mut stack: Vec<(Region, Option<usize>)> = Vec::new();
        let mut body = None;
        let mut owner = None;
        let mut next_region = 0;
        for k in 1..=n {
            if !self.resolve_unclaimed(src, w, k) || !self.wait(|| self.resolved[k].load(Ordering::Acquire)) {
                return None;
            }
            let out = self.out[k].lock().ok()?.take()?;
            if names.len() != w.nbase[k] {
                return None;
            }
            for (chunk, name, ty) in out.values {
                values.push(ValueDef::new(&mut names, name.str(src.segment(chunk as usize).0), ty));
            }
            let info = self.infos[k].get()?;
            let mut ops = out.ops.into_iter();
            // Moves the chunk's ops before index `upto` into `region`.
            let mut taken = 0u32;
            let mut take = |region: &mut Region, upto: u32| {
                region.ops.extend(ops.by_ref().take(upto.saturating_sub(taken) as usize));
                taken = taken.max(upto);
            };
            for &event in &info.events {
                match event {
                    Event::Structured { op, .. } => {
                        let (top, _) = stack.last_mut()?;
                        take(top, op + 1);
                        owner = Some(top.ops.len() - 1);
                    }
                    Event::Header { ops: upto, .. } => {
                        if let Some((top, _)) = stack.last_mut() {
                            take(top, upto);
                        }
                        let capacity = w.regions.get(next_region)?.ops as usize;
                        next_region += 1;
                        stack.push((Region { ops: Vec::with_capacity(capacity) }, owner));
                    }
                    Event::Close { ops: upto, .. } => {
                        let Some((mut region, region_owner)) = stack.pop() else {
                            continue;
                        };
                        take(&mut region, upto);
                        match stack.last_mut() {
                            Some((parent, _)) => {
                                let o = region_owner?;
                                parent.ops.get_mut(o)?.regions.push(region);
                                owner = Some(o);
                            }
                            None => body = Some(region),
                        }
                    }
                }
            }
            if let Some((top, _)) = stack.last_mut() {
                take(top, info.ops);
            }
            if ops.next().is_some() {
                return None;
            }
        }
        if !stack.is_empty() || values.len() != w.vbase[n + 1] as usize {
            return None;
        }
        Some(Kernel { name, max_workgroup_size, waves_per_eu, wave: 32, vgpr_limit: None, agpr_limit, params, body: body?, values, names })
    }
}

/// The kernel line, attributes and parameters, parsed on the calling thread.
pub(super) struct Header {
    name: String,
    max_workgroup_size: u32,
    waves_per_eu: (u32, u32),
    agpr_limit: Option<u32>,
    params: Vec<Param>,
    values: Vec<ValueDef>,
    names: String,
}

/// The header and parameter definitions of `text`, and where its body
/// starts; `None` when the sequential parser must decide.
pub(super) fn header(text: &str) -> Option<(Header, Info, usize)> {
    if text.len() >= u32::MAX as usize {
        return None;
    }
    let mut lines = Lines::new(text, 0, text.len());
    let prologue = prologue(&mut lines).ok()?;
    let body_start = lines.pos();
    let mut header = Header {
        name: prologue.name,
        max_workgroup_size: prologue.max_workgroup_size,
        waves_per_eu: prologue.waves_per_eu,
        agpr_limit: prologue.agpr_limit,
        params: Vec::with_capacity(prologue.params.len()),
        values: Vec::new(),
        names: String::new(),
    };
    let mut params = Info::default();
    for (pname, ty, noalias, readonly) in prologue.params {
        let at = offset(text, pname) as usize;
        if at == 0 || text.as_bytes()[at - 1] != b'%' {
            return None;
        }
        let name = &text[at - 1..at + pname.len()];
        let value = ValueId(header.values.len() as u32);
        header.values.push(ValueDef::new(&mut header.names, name, ty.clone()));
        params.defs.push(Def { name: Name::of(text, name), pos: 0, region: 0, kind: Kind::Line(params.line_defs) });
        params.line_defs += 1;
        params.line_bytes += name.len();
        header.params.push(Param { value, name: pname.to_string(), ty, noalias, readonly });
    }
    Some((header, params, body_start))
}

/// Chunk byte ranges covering `text[start..]`, each ending after a newline
/// (or at the end), preceded by an empty range for the parameters.
fn chunk_bounds(text: &str, start: usize, threads: usize, chunk_bytes: usize) -> Vec<(usize, usize)> {
    let len = text.len() - start;
    let target = match chunk_bytes {
        0 => (len / (threads * CHUNKS_PER_THREAD)).max(MIN_CHUNK_BYTES),
        n => n,
    };
    let bytes = text.as_bytes();
    let mut bounds = vec![(0, 0)];
    let mut at = start;
    while at < text.len() {
        let cut = at + target;
        let end = if cut >= text.len() {
            text.len()
        } else {
            bytes[cut..].iter().position(|&b| b == b'\n').map_or(text.len(), |i| cut + i + 1)
        };
        bounds.push((at, end));
        at = end;
    }
    if bounds.len() == 1 {
        bounds.push((start, start));
    }
    bounds
}

/// The shared state and header for parsing `text` on `threads` threads.
fn prepare(text: &str, threads: usize, chunk_bytes: usize, hashcons: bool) -> Option<(Shared, Header)> {
    let (header, params, body_start) = header(text)?;
    let bounds = chunk_bounds(text, body_start, threads, chunk_bytes);
    let sh = Shared::new(bounds, params, text.len() - body_start, false, hashcons)?;
    sh.enter(text, 0)?;
    Some((sh, header))
}

/// Parses `text` on up to `threads` scoped threads, or `None` when the
/// sequential parser must decide.
pub(super) fn parse(text: &str, threads: usize, chunk_bytes: usize, hashcons: bool) -> Option<Kernel> {
    let (sh, header) = prepare(text, threads, chunk_bytes, hashcons)?;
    let n = sh.chunks();
    std::thread::scope(|s| {
        // Thread creation is serialized and slow, and the scope must join
        // every thread it starts. Stop starting threads once the first pass
        // is all claimed (they would find nothing to scan), or once two
        // started threads have not begun running (no free CPU to run them).
        for spawned in 1..threads {
            if sh.next_scan.load(Ordering::Relaxed) > n || spawned - 1 - sh.started.load(Ordering::Acquire) >= 2 {
                break;
            }
            let worker = std::thread::Builder::new().stack_size(WORKER_STACK).spawn_scoped(s, || sh.worker(text));
            if worker.is_err() {
                break;
            }
        }
        let kernel = sh.lead(text, header);
        if kernel.is_none() {
            sh.fail();
        }
        kernel
    })
}

/// A job for a persistent worker thread.
pub(super) type Job = Box<dyn FnOnce() + Send + 'static>;

/// `parse` with up to `threads - 1` workers taken from persistent threads
/// through `spawn`, which runs a job on one of them. The calling thread does
/// not wait for the workers: once the kernel is stitched, any still running
/// only find nothing left to do.
pub(super) fn parse_pooled(
    spawn: &dyn Fn(Job),
    text: &Arc<str>,
    threads: usize,
    chunk_bytes: usize,
    hashcons: bool,
) -> Option<Kernel> {
    let (sh, header) = prepare(text, threads, chunk_bytes, hashcons)?;
    let sh = Arc::new(sh);
    run_pooled(spawn, &sh, text, threads, header)
}

/// Runs a prepared parse: workers through `spawn`, the lead on this thread.
pub(super) fn run_pooled<S: Source + ?Sized + 'static>(
    spawn: &dyn Fn(Job),
    sh: &Arc<Shared>,
    src: &Arc<S>,
    threads: usize,
    header: Header,
) -> Option<Kernel> {
    for _ in 1..threads.min(sh.chunks() + 1) {
        let (sh, src) = (Arc::clone(sh), Arc::clone(src));
        spawn(Box::new(move || sh.worker(&*src)));
    }
    let kernel = sh.lead(&**src, header);
    if kernel.is_none() {
        sh.fail();
    }
    if sh.panicked() {
        panic!("a parse worker panicked");
    }
    kernel
}

#[cfg(test)]
pub(crate) mod tests {
    use crate::rir::parse::{parse_chunked, parse_parallel, parse_with_threads};

    /// Loops with iteration arguments and several results, branches with one
    /// and two regions, results used after their regions, and names used
    /// far from where they are defined.
    pub(crate) fn kernel(blocks: usize) -> String {
        let mut t = String::from(
            "ckdsl.ir v1\nkernel @t {\n  attrs { max_workgroup_size = i:64 }\n  params {\n    %n : i32\n    \
             %p : ptr<f32,global> { align = i:4 }\n  }\n  region @entry {\n    \
             %c0 : i32 = arith.constant (  ) { ity = s:\"i32\", value = i:0 }\n    \
             %c1 : i32 = arith.constant (  ) { ity = s:\"i32\", value = i:1 }\n    \
             %f0 : f32 = arith.constant (  ) { ity = s:\"f32\", value = f:0.0 }\n    \
             %lt : i1 = arith.cmp ( %c0, %n ) { pred = s:\"lt\" }\n",
        );
        let mut prev = "%c0".to_string();
        for b in 0..blocks {
            t += &format!(
                "    %r{b} : i32, %q{b} : f32 = scf.for ( %c0, %n, %c1, {prev}, %f0 ) {{ iter_args = l:[ {{ name = \
                 s:\"%a{b}\", type = s:\"i32\" }}, {{ name = s:\"%b{b}\", type = s:\"f32\" }} ], iv = s:\"%i{b}\", \
                 iv_type = s:\"i32\", num_iter_args = i:2 }}\n    region @body {{\n      \
                 %s{b} : i32 = arith.add ( %a{b}, %i{b} )\n      scf.if ( %lt )\n      region @then {{\n        \
                 %t{b} : i32 = arith.mul ( %s{b}, %c1 ) {{ wrap = s:\"nsw\" }}\n      }}\n      \
                 scf.if_else ( %lt )\n      region @then {{\n        %u{b} : i32 = arith.sub ( %s{b}, {prev} )\n      \
                 }}\n      region @else {{\n        %w{b} : i32 = arith.add ( %s{b}, %n )\n      }}\n      \
                 %x{b} : f32 = arith.fadd ( %b{b}, %f0 )\n      scf.yield ( %s{b}, %x{b} )\n    }}\n    \
                 %y{b} : i32 = arith.add ( %r{b}, {prev} )\n    # block {b} done\n\n"
            );
            prev = format!("%y{b}");
        }
        t + "    cf.return (  )\n  }\n}\n"
    }

    pub(crate) fn outcome(r: Result<crate::rir::ast::Kernel, crate::CompileError>) -> String {
        match r {
            Ok(k) => format!("{k:?}"),
            Err(e) => format!("error: {}", e.message),
        }
    }

    #[test]
    fn every_chunking_gives_the_sequential_kernel() {
        let text = kernel(6);
        let reference = outcome(parse_with_threads(&text, 1));
        assert!(!reference.starts_with("error"), "{reference}");
        for threads in [2, 3, 5] {
            for chunk in [1, 2, 3, 5, 8, 13, 40, 100, 333, 1000, 0] {
                let kernel = parse_parallel(&text, threads, chunk).expect("the parallel path takes it");
                assert_eq!(format!("{kernel:?}"), reference, "threads {threads} chunk {chunk}");
            }
        }
    }

    /// Every one-line edit: the same kernel or the same error, and the
    /// parallel path never accepts what the sequential parser rejects.
    #[test]
    fn line_edits_give_the_sequential_result() {
        let text = kernel(3);
        let lines: Vec<&str> = text.lines().collect();
        let mut mutants = Vec::new();
        for i in 0..lines.len() {
            let mut deleted = lines.clone();
            deleted.remove(i);
            mutants.push(deleted.join("\n"));
            let mut duplicated = lines.clone();
            duplicated.insert(i, lines[i]);
            mutants.push(duplicated.join("\n"));
            if i + 1 < lines.len() {
                let mut swapped = lines.clone();
                swapped.swap(i, i + 1);
                mutants.push(swapped.join("\n"));
            }
        }
        let mut accepted = 0;
        for mutant in &mutants {
            let reference = outcome(parse_with_threads(mutant, 1));
            accepted += !reference.starts_with("error") as usize;
            for (threads, chunk) in [(2, 1), (3, 17), (4, 0)] {
                assert_eq!(outcome(parse_chunked(mutant, threads, chunk)), reference, "{mutant}");
                if let Some(kernel) = parse_parallel(mutant, threads, chunk) {
                    assert_eq!(format!("{kernel:?}"), reference, "{mutant}");
                }
            }
        }
        assert!(accepted > 0 && accepted < mutants.len(), "{accepted} of {}", mutants.len());
    }

    #[test]
    fn a_name_defined_twice_goes_to_the_sequential_parser() {
        let text = kernel(2).replace("%u1 : i32", "%s1 : i32");
        assert!(parse_parallel(&text, 2, 0).is_none());
        assert_eq!(outcome(parse_chunked(&text, 2, 0)), outcome(parse_with_threads(&text, 1)));
        assert!(parse_with_threads(&text, 1).is_ok());
    }

    #[test]
    fn deep_nesting_in_parallel() {
        let depth = 300;
        let mut text = String::from(
            "ckdsl.ir v1\nkernel @k {\nparams {\n%n : i32\n}\nregion @entry {\n\
             %c : i1 = arith.cmp ( %n, %n ) { pred = s:\"eq\" }\n",
        );
        for d in 0..depth {
            text += &format!("scf.if ( %c )\nregion @then {{\n%v{d} : i32 = arith.add ( %n, %n )\n");
        }
        for d in (0..depth).rev() {
            text += &format!("%w{d} : i32 = arith.add ( %v{d}, %n )\n}}\n");
        }
        text += "cf.return (  )\n}\n}\n";
        let reference = outcome(parse_with_threads(&text, 1));
        for chunk in [1, 64, 0] {
            let kernel = parse_parallel(&text, 4, chunk).expect("the parallel path takes it");
            assert_eq!(format!("{kernel:?}"), reference);
        }
    }
}
