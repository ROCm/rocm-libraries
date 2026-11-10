// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

//! Source attribution of code compiled from IR whose ops carry `@loc`s.
//!
//! Each op's location is its authoring call stack, `path:line:col:func`
//! frames joined by `;`, innermost first (the DSL's `join_loc`). Instruction
//! selection gives every instruction the location of the op it was selected
//! for (`MInst::loc`); an instruction a later pass made (a wait, a nop, a
//! copy, an M0 write, address arithmetic) takes the location of the next
//! located instruction of its block, which is the one it serves, or failing
//! that the previous one, and gets a synthetic innermost frame named
//! `rocke.<mnemonic>`.
//!
//! The result is written two ways. The code object gets DWARF 4 in the shape
//! the LLVM lowering produces with `ROCKE_DEBUG_LOC=1`: a line table with
//! the innermost frame of each address, and a compile unit holding the
//! kernel's `DW_TAG_subprogram` with one `DW_TAG_inlined_subroutine` per
//! inner frame (abstract origin, call file / line / column and PC ranges).
//! The outermost frame is the kernel itself, as in the LLVM lowering. Beside
//! that, `SourceMap::json` is the same mapping as plain JSON for tools.

use std::collections::HashMap;
use std::fmt::Write;

use crate::error::CompileError;
use crate::rir::mir::{MFunc, NO_LOC};
use crate::target::Target;

/// One authoring stack frame.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Frame {
    pub file: String,
    pub line: u32,
    pub col: u32,
    pub func: String,
}

/// Splits a location on unescaped `;` and unescapes each frame (`\\`, `\;`);
/// any other backslash is literal (the DSL's `split_loc`).
pub fn split_loc(loc: &str) -> Vec<String> {
    let mut parts = Vec::new();
    let mut buf = String::new();
    let mut chars = loc.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '\\' => match chars.next() {
                Some(n @ ('\\' | ';')) => buf.push(n),
                Some(n) => {
                    buf.push('\\');
                    buf.push(n);
                }
                None => buf.push('\\'),
            },
            ';' => parts.push(std::mem::take(&mut buf)),
            c => buf.push(c),
        }
    }
    parts.push(buf);
    parts
}

/// `path:line[:col[:func]]`, the fields taken from the right so a path may
/// hold colons (the lowering's `_parse_frame`).
pub fn parse_frame(text: &str) -> Option<Frame> {
    let mut parts: Vec<&str> = text.split(':').collect();
    if parts.len() < 2 {
        return None;
    }
    let digits = |s: &str| !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit());
    let mut func = String::new();
    if !digits(parts[parts.len() - 1]) {
        func = parts.pop()?.to_string();
    }
    let mut col = 0;
    if parts.len() >= 3 && digits(parts[parts.len() - 1]) && digits(parts[parts.len() - 2]) {
        col = parts.pop()?.parse().ok()?;
    }
    if parts.len() < 2 || !digits(parts[parts.len() - 1]) {
        return None;
    }
    let line = parts.pop()?.parse().ok()?;
    let file = parts.join(":");
    (!file.is_empty()).then_some(Frame { file, line, col, func })
}

/// A location's frames, innermost first; malformed frames are dropped.
pub fn parse_loc(loc: &str) -> Vec<Frame> {
    split_loc(loc).iter().filter_map(|part| parse_frame(part)).collect()
}

/// One instruction of the final code.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Row {
    /// Byte offset in `.text` and encoded size.
    pub offset: u32,
    pub size: u32,
    pub mnemonic: String,
    /// Index into `SourceMap::chains`, or `None` when no instruction of the
    /// function has a usable location.
    pub chain: Option<usize>,
    /// The location is the served instruction's: no op was selected to this one.
    pub synthetic: bool,
    /// The machine block and its loop depth.
    pub block: u32,
    pub loop_depth: u32,
}

/// Where each instruction of a compiled function came from.
#[derive(Clone, Debug)]
pub struct SourceMap {
    pub kernel: String,
    pub target: Target,
    pub code_bytes: u32,
    pub rows: Vec<Row>,
    /// The parsed `MFunc::locs`, innermost frame first.
    pub chains: Vec<Vec<Frame>>,
}

fn mnemonic(target: Target, op: crate::rir::mir::Opc) -> String {
    match target {
        Target::Gfx950 => crate::target::gfx9::gen_isa::mnemonic(op).map_or_else(|| op.name(), str::to_string),
        Target::Gfx1151 => op.name(),
    }
}

impl SourceMap {
    /// The map of `f`, encoded as `assemble` encodes it `entry` bytes into
    /// `.text` (after a kernarg preload entry); `None` when no instruction
    /// has a location (the IR had no `@loc`).
    pub fn of(f: &MFunc, kernel: &str, entry: u32) -> Result<Option<SourceMap>, CompileError> {
        if f.locs.is_empty() {
            return Ok(None);
        }
        let starts = match f.target {
            Target::Gfx950 => crate::target::gfx9::asm::inst_starts(f)?,
            Target::Gfx1151 => crate::rir::asm::inst_starts(f)?,
        };
        let chains: Vec<Vec<Frame>> = f.locs.iter().map(|loc| parse_loc(loc)).collect();
        let usable = |loc: u32| loc != NO_LOC && !chains[loc as usize - 1].is_empty();
        let mut rows = Vec::with_capacity(starts.len());
        let mut fallback = None;
        for (b, block) in f.blocks.iter().enumerate() {
            let mut next = None;
            let mut served = vec![None; block.insts.len()];
            for (i, inst) in block.insts.iter().enumerate().rev() {
                if usable(inst.loc) {
                    next = Some(inst.loc as usize - 1);
                }
                served[i] = next;
            }
            let mut prev = None;
            for (inst, served) in block.insts.iter().zip(served) {
                let own = usable(inst.loc).then(|| inst.loc as usize - 1);
                prev = own.or(prev);
                let i = rows.len();
                rows.push(Row {
                    offset: entry + 4 * starts[i],
                    size: 4 * (starts[i + 1] - starts[i]),
                    mnemonic: mnemonic(f.target, inst.op),
                    chain: own.or(served).or(prev),
                    synthetic: own.is_none(),
                    block: b as u32,
                    loop_depth: block.loop_depth,
                });
                fallback = fallback.or(own);
            }
        }
        // Blocks with no located instruction take the next located block's
        // location, the last ones the last location.
        let mut next = None;
        for row in rows.iter_mut().rev() {
            match row.chain {
                Some(c) => next = Some(c),
                None => row.chain = next,
            }
        }
        let mut prev = fallback;
        for row in &mut rows {
            match row.chain {
                Some(c) => prev = Some(c),
                None => row.chain = prev,
            }
        }
        Ok(Some(SourceMap { kernel: kernel.to_string(), target: f.target, code_bytes: entry + 4 * starts[starts.len() - 1], rows, chains }))
    }

    /// The map as JSON: `.text` at ELF virtual address `text_vaddr`.
    pub fn json(&self, text_vaddr: u64) -> String {
        let mut files = Interner::default();
        let mut funcs = Interner::default();
        let chains: Vec<Vec<[u32; 4]>> = self
            .chains
            .iter()
            .map(|chain| chain.iter().map(|f| [files.id(&f.file), f.line, f.col, funcs.id(&f.func)]).collect())
            .collect();
        let mut out = String::new();
        let _ = write!(
            out,
            "{{\"version\":1,\"schema\":\"rocke-srcmap\",\"kernel\":{},\"target\":{},\"text_vaddr\":{text_vaddr},\"code_bytes\":{},",
            quote(&self.kernel),
            quote(self.target.info().name),
            self.code_bytes
        );
        out.push_str("\"chain_format\":\"[file, line, col, func] per frame, innermost first\",");
        out.push_str("\"inst_format\":\"[offset, size, mnemonic, chain or -1, synthetic 0/1, block, loop depth]\",\"files\":[");
        join(&mut out, files.names.iter().map(|s| quote(s)));
        out.push_str("],\"funcs\":[");
        join(&mut out, funcs.names.iter().map(|s| quote(s)));
        out.push_str("],\"chains\":[");
        join(
            &mut out,
            chains.iter().map(|chain| {
                let frames: Vec<String> = chain.iter().map(|[f, l, c, n]| format!("[{f},{l},{c},{n}]")).collect();
                format!("[{}]", frames.join(","))
            }),
        );
        out.push_str("],\"insts\":[");
        join(
            &mut out,
            self.rows.iter().map(|r| {
                let chain = r.chain.map_or(-1, |c| c as i64);
                format!("[{},{},{},{chain},{},{},{}]", r.offset, r.size, quote(&r.mnemonic), u8::from(r.synthetic), r.block, r.loop_depth)
            }),
        );
        out.push_str("]}\n");
        out
    }

    /// The DWARF sections (`.debug_abbrev`, `.debug_info`, `.debug_ranges`,
    /// `.debug_str`, `.debug_line`) for `.text` at ELF virtual address `text_vaddr`.
    pub fn dwarf(&self, text_vaddr: u64) -> Vec<(&'static str, Vec<u8>)> {
        Dwarf::build(self, text_vaddr)
    }

    /// The stack of each instruction as `emit_inline_frames.py` reads it
    /// from the DWARF: outermost first, `(function, call frame)`, the kernel
    /// itself without a call frame.
    pub fn stack(&self, row: &Row) -> Vec<(String, Option<Frame>)> {
        let mut stack = vec![(self.kernel.clone(), None)];
        let Some(chain) = row.chain.map(|c| &self.chains[c]) else {
            return stack;
        };
        let outer: Vec<&Frame> = chain.iter().rev().collect();
        for pair in outer.windows(2) {
            stack.push((pair[1].func.clone(), Some(pair[0].clone())));
        }
        if row.synthetic {
            stack.push((format!("rocke.{}", row.mnemonic), outer.last().map(|f| (*f).clone())));
        }
        stack
    }
}

fn join(out: &mut String, items: impl Iterator<Item = String>) {
    for (i, item) in items.enumerate() {
        if i > 0 {
            out.push(',');
        }
        out.push_str(&item);
    }
}

fn quote(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

#[derive(Default)]
struct Interner {
    ids: HashMap<String, u32>,
    names: Vec<String>,
}

impl Interner {
    fn id(&mut self, name: &str) -> u32 {
        if let Some(&id) = self.ids.get(name) {
            return id;
        }
        let id = self.names.len() as u32;
        self.ids.insert(name.to_string(), id);
        self.names.push(name.to_string());
        id
    }
}

// DWARF constants (DWARF 4).
const DW_TAG_COMPILE_UNIT: u64 = 0x11;
const DW_TAG_SUBPROGRAM: u64 = 0x2e;
const DW_TAG_INLINED_SUBROUTINE: u64 = 0x1d;
const DW_AT_NAME: u64 = 0x03;
const DW_AT_STMT_LIST: u64 = 0x10;
const DW_AT_LOW_PC: u64 = 0x11;
const DW_AT_HIGH_PC: u64 = 0x12;
const DW_AT_LANGUAGE: u64 = 0x13;
const DW_AT_COMP_DIR: u64 = 0x1b;
const DW_AT_INLINE: u64 = 0x20;
const DW_AT_PRODUCER: u64 = 0x25;
const DW_AT_ABSTRACT_ORIGIN: u64 = 0x31;
const DW_AT_EXTERNAL: u64 = 0x3f;
const DW_AT_RANGES: u64 = 0x55;
const DW_AT_CALL_COLUMN: u64 = 0x57;
const DW_AT_CALL_FILE: u64 = 0x58;
const DW_AT_CALL_LINE: u64 = 0x59;
const DW_FORM_ADDR: u64 = 0x01;
const DW_FORM_DATA1: u64 = 0x0b;
const DW_FORM_DATA2: u64 = 0x05;
const DW_FORM_UDATA: u64 = 0x0f;
const DW_FORM_REF4: u64 = 0x13;
const DW_FORM_STRP: u64 = 0x0e;
const DW_FORM_SEC_OFFSET: u64 = 0x17;
const DW_FORM_FLAG_PRESENT: u64 = 0x19;
const DW_LANG_PYTHON: u16 = 0x14;
const DW_INL_INLINED: u8 = 1;

const ABBREV_CU: u64 = 1;
const ABBREV_ABSTRACT: u64 = 2;
const ABBREV_KERNEL: u64 = 3;
/// Inlined subroutines: one range or a range list, with or without children.
const ABBREV_INLINED_PC: u64 = 4;
const ABBREV_INLINED_RANGES: u64 = 6;

fn uleb(out: &mut Vec<u8>, mut v: u64) {
    loop {
        let byte = (v & 0x7f) as u8;
        v >>= 7;
        if v == 0 {
            out.push(byte);
            return;
        }
        out.push(byte | 0x80);
    }
}

fn sleb(out: &mut Vec<u8>, mut v: i64) {
    loop {
        let byte = (v & 0x7f) as u8;
        v >>= 7;
        let done = (v == 0 && byte & 0x40 == 0) || (v == -1 && byte & 0x40 != 0);
        if done {
            out.push(byte);
            return;
        }
        out.push(byte | 0x80);
    }
}

/// An inlined call: the function and its call site, the PC ranges it
/// covers (byte offsets in `.text`), and the calls inside it.
struct Node {
    func: String,
    call: Frame,
    ranges: Vec<(u32, u32)>,
    children: Vec<usize>,
}

struct Dwarf {
    strs: Vec<u8>,
    str_at: HashMap<String, u32>,
    /// Line-table file numbers (1-based).
    files: Interner,
}

impl Dwarf {
    fn strp(&mut self, s: &str) -> u32 {
        if let Some(&at) = self.str_at.get(s) {
            return at;
        }
        let at = self.strs.len() as u32;
        self.strs.extend_from_slice(s.as_bytes());
        self.strs.push(0);
        self.str_at.insert(s.to_string(), at);
        at
    }

    fn file(&mut self, path: &str) -> u64 {
        u64::from(self.files.id(path)) + 1
    }

    fn build(map: &SourceMap, text_vaddr: u64) -> Vec<(&'static str, Vec<u8>)> {
        let mut d = Dwarf { strs: Vec::new(), str_at: HashMap::new(), files: Interner::default() };
        let line = d.line_program(map, text_vaddr);

        // The inlining tree: node 0 is the kernel.
        let mut nodes = vec![Node { func: map.kernel.clone(), call: Frame { file: String::new(), line: 0, col: 0, func: String::new() }, ranges: Vec::new(), children: Vec::new() }];
        let mut index: HashMap<(usize, String, Frame), usize> = HashMap::new();
        for row in &map.rows {
            let mut at = 0;
            for (func, call) in map.stack(row).into_iter().skip(1) {
                let call = call.expect("an inner frame has a call site");
                let key = (at, func, call);
                at = match index.get(&key) {
                    Some(&n) => n,
                    None => {
                        let n = nodes.len();
                        nodes.push(Node { func: key.1.clone(), call: key.2.clone(), ranges: Vec::new(), children: Vec::new() });
                        nodes[key.0].children.push(n);
                        index.insert(key, n);
                        n
                    }
                };
                let ranges = &mut nodes[at].ranges;
                match ranges.last_mut() {
                    Some(last) if last.1 == row.offset => last.1 = row.offset + row.size,
                    _ => ranges.push((row.offset, row.offset + row.size)),
                }
            }
        }

        let mut abbrev = Vec::new();
        let mut decl = |code: u64, tag: u64, children: bool, attrs: &[(u64, u64)]| {
            uleb(&mut abbrev, code);
            uleb(&mut abbrev, tag);
            abbrev.push(u8::from(children));
            for &(at, form) in attrs {
                uleb(&mut abbrev, at);
                uleb(&mut abbrev, form);
            }
            abbrev.extend_from_slice(&[0, 0]);
        };
        decl(
            ABBREV_CU,
            DW_TAG_COMPILE_UNIT,
            true,
            &[
                (DW_AT_PRODUCER, DW_FORM_STRP),
                (DW_AT_LANGUAGE, DW_FORM_DATA2),
                (DW_AT_NAME, DW_FORM_STRP),
                (DW_AT_STMT_LIST, DW_FORM_SEC_OFFSET),
                (DW_AT_COMP_DIR, DW_FORM_STRP),
                (DW_AT_LOW_PC, DW_FORM_ADDR),
                (DW_AT_HIGH_PC, DW_FORM_ADDR),
            ],
        );
        decl(ABBREV_ABSTRACT, DW_TAG_SUBPROGRAM, false, &[(DW_AT_NAME, DW_FORM_STRP), (DW_AT_INLINE, DW_FORM_DATA1)]);
        decl(
            ABBREV_KERNEL,
            DW_TAG_SUBPROGRAM,
            true,
            &[(DW_AT_NAME, DW_FORM_STRP), (DW_AT_LOW_PC, DW_FORM_ADDR), (DW_AT_HIGH_PC, DW_FORM_ADDR), (DW_AT_EXTERNAL, DW_FORM_FLAG_PRESENT)],
        );
        let call = [(DW_AT_CALL_FILE, DW_FORM_UDATA), (DW_AT_CALL_LINE, DW_FORM_UDATA), (DW_AT_CALL_COLUMN, DW_FORM_UDATA)];
        let pc = [(DW_AT_ABSTRACT_ORIGIN, DW_FORM_REF4), (DW_AT_LOW_PC, DW_FORM_ADDR), (DW_AT_HIGH_PC, DW_FORM_ADDR)];
        let ranged = [(DW_AT_ABSTRACT_ORIGIN, DW_FORM_REF4), (DW_AT_RANGES, DW_FORM_SEC_OFFSET)];
        for (code, head) in [(ABBREV_INLINED_PC, &pc[..]), (ABBREV_INLINED_RANGES, &ranged[..])] {
            let attrs: Vec<(u64, u64)> = head.iter().chain(&call).copied().collect();
            decl(code, DW_TAG_INLINED_SUBROUTINE, true, &attrs);
            decl(code + 1, DW_TAG_INLINED_SUBROUTINE, false, &attrs);
        }
        abbrev.push(0);

        let outermost = map.chains.iter().find_map(|c| c.last()).map_or_else(|| map.kernel.clone(), |f| f.file.clone());
        let (producer, name, comp_dir) = (d.strp("rocke"), d.strp(&outermost), d.strp(""));
        let mut info = vec![0; 4];
        info.extend_from_slice(&4u16.to_le_bytes());
        info.extend_from_slice(&0u32.to_le_bytes());
        info.push(8);
        let end = text_vaddr + u64::from(map.code_bytes);
        uleb(&mut info, ABBREV_CU);
        info.extend_from_slice(&producer.to_le_bytes());
        info.extend_from_slice(&DW_LANG_PYTHON.to_le_bytes());
        info.extend_from_slice(&name.to_le_bytes());
        info.extend_from_slice(&0u32.to_le_bytes());
        info.extend_from_slice(&comp_dir.to_le_bytes());
        info.extend_from_slice(&text_vaddr.to_le_bytes());
        info.extend_from_slice(&end.to_le_bytes());

        // Abstract subprograms, one per inlined function.
        let mut origin: HashMap<String, u32> = HashMap::new();
        for node in &nodes[1..] {
            if !origin.contains_key(&node.func) {
                origin.insert(node.func.clone(), info.len() as u32);
                uleb(&mut info, ABBREV_ABSTRACT);
                let s = d.strp(&node.func);
                info.extend_from_slice(&s.to_le_bytes());
                info.push(DW_INL_INLINED);
            }
        }
        uleb(&mut info, ABBREV_KERNEL);
        let s = d.strp(&map.kernel);
        info.extend_from_slice(&s.to_le_bytes());
        info.extend_from_slice(&text_vaddr.to_le_bytes());
        info.extend_from_slice(&end.to_le_bytes());

        // Ranges are relative to the compile unit's low PC, the start of `.text`.
        let mut ranges = Vec::new();
        let mut stack: Vec<(usize, usize)> = vec![(0, 0)];
        while let Some(&mut (node, ref mut next)) = stack.last_mut() {
            let Some(&child) = nodes[node].children.get(*next) else {
                info.push(0);
                stack.pop();
                continue;
            };
            *next += 1;
            let n = &nodes[child];
            let has_children = !n.children.is_empty();
            if let [(lo, hi)] = n.ranges[..] {
                uleb(&mut info, ABBREV_INLINED_PC + u64::from(!has_children));
                info.extend_from_slice(&origin[&n.func].to_le_bytes());
                info.extend_from_slice(&(text_vaddr + u64::from(lo)).to_le_bytes());
                info.extend_from_slice(&(text_vaddr + u64::from(hi)).to_le_bytes());
            } else {
                uleb(&mut info, ABBREV_INLINED_RANGES + u64::from(!has_children));
                info.extend_from_slice(&origin[&n.func].to_le_bytes());
                info.extend_from_slice(&(ranges.len() as u32).to_le_bytes());
                for &(lo, hi) in &n.ranges {
                    ranges.extend_from_slice(&u64::from(lo).to_le_bytes());
                    ranges.extend_from_slice(&u64::from(hi).to_le_bytes());
                }
                ranges.extend_from_slice(&[0; 16]);
            }
            let file = d.file(&n.call.file);
            uleb(&mut info, file);
            uleb(&mut info, u64::from(n.call.line));
            uleb(&mut info, u64::from(n.call.col));
            if has_children {
                stack.push((child, 0));
            }
        }
        info.push(0);
        let length = (info.len() - 4) as u32;
        info[..4].copy_from_slice(&length.to_le_bytes());

        // Call files may add names the line program's header did not list.
        let line = if d.files.names.len() == line.1 { line.0 } else { d.line_program(map, text_vaddr).0 };
        vec![(".debug_abbrev", abbrev), (".debug_info", info), (".debug_ranges", ranges), (".debug_str", d.strs), (".debug_line", line)]
    }

    /// The line program and the file count its header lists.
    fn line_program(&mut self, map: &SourceMap, text_vaddr: u64) -> (Vec<u8>, usize) {
        let mut rows = Vec::new();
        for row in &map.rows {
            if let Some(frame) = row.chain.and_then(|c| map.chains[c].first()) {
                rows.push((row.offset, self.file(&frame.file), frame.line, frame.col));
            }
        }
        const LINE_BASE: i8 = -5;
        const LINE_RANGE: u8 = 14;
        const OPCODE_BASE: u8 = 13;
        let mut out = vec![0; 4];
        out.extend_from_slice(&4u16.to_le_bytes());
        out.extend_from_slice(&[0; 4]);
        let header = out.len();
        out.extend_from_slice(&[1, 1, 1, LINE_BASE as u8, LINE_RANGE, OPCODE_BASE]);
        out.extend_from_slice(&[0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1]);
        out.push(0);
        for name in &self.files.names {
            out.extend_from_slice(name.as_bytes());
            out.extend_from_slice(&[0, 0, 0, 0]);
        }
        out.push(0);
        let header_length = (out.len() - header) as u32;
        out[6..10].copy_from_slice(&header_length.to_le_bytes());

        out.extend_from_slice(&[0, 9, 2]);
        out.extend_from_slice(&text_vaddr.to_le_bytes());
        let (mut addr, mut file, mut line, mut col) = (0u32, 1u64, 1u32, 0u32);
        let mut last = None;
        for (offset, f, l, c) in rows {
            if last == Some((f, l, c)) {
                continue;
            }
            last = Some((f, l, c));
            if offset != addr {
                out.push(2);
                uleb(&mut out, u64::from(offset - addr));
                addr = offset;
            }
            if f != file {
                out.push(4);
                uleb(&mut out, f);
                file = f;
            }
            if c != col {
                out.push(5);
                uleb(&mut out, u64::from(c));
                col = c;
            }
            if l != line {
                out.push(3);
                sleb(&mut out, i64::from(l) - i64::from(line));
                line = l;
            }
            out.push(1);
        }
        if map.code_bytes != addr {
            out.push(2);
            uleb(&mut out, u64::from(map.code_bytes - addr));
        }
        out.extend_from_slice(&[0, 1, 1]);
        let length = (out.len() - 4) as u32;
        out[..4].copy_from_slice(&length.to_le_bytes());
        (out, self.files.names.len())
    }
}

/// `text` with a three-frame `@loc` on every op line of its regions, as the
/// DSL prints them: one path with a `;` in it, escaped twice.
#[cfg(test)]
pub(crate) fn with_test_locs(text: &str) -> String {
    let mut out = String::new();
    for (n, line) in text.lines().enumerate() {
        out += line;
        let op = line.starts_with("    ") && !line.trim_start().starts_with("region") && line.trim() != "}" && !line.contains(" : ptr<");
        if op {
            out += &format!(
                " @loc \"/w/semi\\\\;colon.py:{n}:5:helper{};/w/build.py:{}:9:stage{};/w/top.py:7:1:build\"",
                n % 3,
                100 + n / 4,
                n / 4
            );
        }
        out.push('\n');
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frames_parse_from_the_right_as_the_lowering_does() {
        let f = parse_frame("C:\\Users\\a\\b.py:12:5:helper").expect("a frame");
        assert_eq!(f, Frame { file: "C:\\Users\\a\\b.py".into(), line: 12, col: 5, func: "helper".into() });
        assert_eq!(parse_frame("a.py:3"), Some(Frame { file: "a.py".into(), line: 3, col: 0, func: String::new() }));
        assert_eq!(parse_frame("a.py:3:f"), Some(Frame { file: "a.py".into(), line: 3, col: 0, func: "f".into() }));
        assert_eq!(parse_frame("a.py"), None);
        assert_eq!(parse_frame(":3:4:f"), None);
        assert_eq!(split_loc("a\\;b.py:1:2:f;c\\\\d.py:3:4:g;x\\y.py:5"), ["a;b.py:1:2:f", "c\\d.py:3:4:g", "x\\y.py:5"]);
        assert_eq!(parse_loc("a.py:1:2:f;bad;b.py:3:4:g").len(), 2);
    }

    /// A loop with a global load in it: the load's wait and the loop's
    /// compare and branch are instructions no op was selected to alone.
    fn probe(attrs: &str) -> String {
        let body = [
            "%tid : i32 = gpu.thread_id (  ) { axis = s:\"x\" }",
            "%c0 : i32 = arith.constant (  ) { ity = s:\"i32\", value = i:0 }",
            "%c1 : i32 = arith.constant (  ) { ity = s:\"i32\", value = i:1 }",
            "%c8 : i32 = arith.constant (  ) { ity = s:\"i32\", value = i:8 }",
            "%z : f32 = arith.constant (  ) { ity = s:\"f32\", value = f:0.0 }",
            "%r : f32 = scf.for ( %c0, %c8, %c1, %z ) { iter_args = l:[ { name = s:\"%acc\", type = s:\"f32\" } ], \
             iv = s:\"%i\", iv_type = s:\"i32\", num_iter_args = i:1, unroll = b:false }",
            "region @body {",
            "%c1b : i32 = arith.constant (  ) { ity = s:\"i32\", value = i:1 }",
            "%p : i32 = arith.add ( %i, %tid )",
            "%q : i32 = arith.add ( %p, %c1b )",
            "%x : f32 = memref.global_load_typed ( %X, %q ) { align = i:4, elem_type = s:\"f32\" }",
            "%a2 : f32 = arith.fadd ( %acc, %x )",
            "scf.yield ( %a2 ) { num = i:1 }",
            "}",
            "memref.global_store_typed ( %Y, %tid, %r ) { align = i:4, elem_type = s:\"f32\" }",
            "cf.return (  )",
        ];
        let mut text = format!(
            "ckdsl.ir v1\nkernel @loc_probe {{\n  attrs {{ max_workgroup_size = i:64{attrs} }}\n  params {{\n    \
             %X : ptr<f32,global>\n    %Y : ptr<f32,global>\n  }}\n  region @entry {{\n"
        );
        for line in body {
            text += &format!("    {line}\n");
        }
        text + "  }\n}\n"
    }

    use super::with_test_locs as located;

    fn section<'a>(image: &'a [u8], want: &str) -> Option<&'a [u8]> {
        let u = |at: usize, n: usize| image[at..at + n].iter().rev().fold(0u64, |v, &b| v << 8 | u64::from(b)) as usize;
        let (table, count, names) = (u(0x28, 8), u(0x3c, 2), u(0x3e, 2));
        let names = u(table + 64 * names + 0x18, 8);
        (0..count).find_map(|i| {
            let h = table + 64 * i;
            let name = &image[names + u(h, 4)..];
            let name = &name[..name.iter().position(|&b| b == 0)?];
            (name == want.as_bytes()).then(|| &image[u(h + 0x18, 8)..u(h + 0x18, 8) + u(h + 0x20, 8)])
        })
    }

    struct Reader<'a> {
        b: &'a [u8],
        at: usize,
    }

    impl Reader<'_> {
        fn n(&mut self, n: usize) -> u64 {
            let v = self.b[self.at..self.at + n].iter().rev().fold(0u64, |v, &b| v << 8 | u64::from(b));
            self.at += n;
            v
        }
        fn uleb(&mut self) -> u64 {
            let (mut v, mut shift) = (0u64, 0);
            loop {
                let b = self.n(1);
                v |= (b & 0x7f) << shift;
                shift += 7;
                if b & 0x80 == 0 {
                    return v;
                }
            }
        }
        fn sleb(&mut self) -> i64 {
            let (mut v, mut shift) = (0i64, 0);
            loop {
                let b = self.n(1) as i64;
                v |= (b & 0x7f) << shift;
                shift += 7;
                if b & 0x80 == 0 {
                    return if shift < 64 && b & 0x40 != 0 { v | -1 << shift } else { v };
                }
            }
        }
        fn cstr(&mut self) -> String {
            let len = self.b[self.at..].iter().position(|&b| b == 0).expect("terminated");
            let s = String::from_utf8(self.b[self.at..self.at + len].to_vec()).expect("utf-8");
            self.at += len + 1;
            s
        }
    }

    /// The line table's file names and rows `(address, file, line, column)`.
    fn decode_lines(line: &[u8]) -> (Vec<String>, Vec<(u64, u64, u64, u64)>) {
        let mut r = Reader { b: line, at: 0 };
        let end = r.n(4) as usize + 4;
        assert_eq!(r.n(2), 4);
        let program = r.n(4) as usize + r.at;
        let (_min, _ops, _stmt, _base, _range, opcode_base) = (r.n(1), r.n(1), r.n(1), r.n(1), r.n(1), r.n(1));
        r.at += opcode_base as usize - 1;
        assert_eq!(r.cstr(), "", "no include directories");
        let mut files = Vec::new();
        loop {
            let name = r.cstr();
            if name.is_empty() {
                break;
            }
            files.push(name);
            (r.uleb(), r.uleb(), r.uleb());
        }
        assert_eq!(r.at, program);
        let (mut addr, mut file, mut line, mut col) = (0u64, 1u64, 1i64, 0u64);
        let mut rows = Vec::new();
        while r.at < end {
            match r.n(1) {
                0 => {
                    let len = r.uleb() as usize;
                    let next = r.at + len;
                    match r.n(1) {
                        1 => rows.push((addr, 0, 0, 0)),
                        2 => addr = r.n(8),
                        op => panic!("extended opcode {op}"),
                    }
                    assert_eq!(r.at, next);
                }
                1 => rows.push((addr, file, line as u64, col)),
                2 => addr += r.uleb(),
                3 => line += r.sleb(),
                4 => file = r.uleb(),
                5 => col = r.uleb(),
                op => panic!("standard opcode {op}"),
            }
        }
        (files, rows)
    }

    type Die = (usize, u64, std::collections::HashMap<u64, u64>);

    /// The DIEs of the one compile unit: `(depth, tag, attributes)` by offset.
    fn decode_info(info: &[u8], abbrev: &[u8]) -> std::collections::BTreeMap<usize, Die> {
        let mut table = HashMap::new();
        let mut a = Reader { b: abbrev, at: 0 };
        loop {
            let code = a.uleb();
            if code == 0 {
                break;
            }
            let (tag, children) = (a.uleb(), a.n(1) == 1);
            let mut attrs = Vec::new();
            loop {
                let (at, form) = (a.uleb(), a.uleb());
                if at == 0 {
                    break;
                }
                attrs.push((at, form));
            }
            table.insert(code, (tag, children, attrs));
        }
        let mut r = Reader { b: info, at: 0 };
        let end = r.n(4) as usize + 4;
        assert_eq!((r.n(2), r.n(4), r.n(1)), (4, 0, 8));
        let mut dies = std::collections::BTreeMap::new();
        let mut depth = 0;
        while r.at < end {
            let offset = r.at;
            let code = r.uleb();
            if code == 0 {
                depth -= 1;
                continue;
            }
            let (tag, children, attrs) = &table[&code];
            let mut values = HashMap::new();
            for &(at, form) in attrs {
                let v = match form {
                    DW_FORM_ADDR => r.n(8),
                    DW_FORM_DATA1 => r.n(1),
                    DW_FORM_DATA2 => r.n(2),
                    DW_FORM_UDATA => r.uleb(),
                    DW_FORM_REF4 | DW_FORM_STRP | DW_FORM_SEC_OFFSET => r.n(4),
                    DW_FORM_FLAG_PRESENT => 1,
                    _ => panic!("form {form:#x}"),
                };
                values.insert(at, v);
            }
            dies.insert(offset, (depth, *tag, values));
            if *children {
                depth += 1;
            }
        }
        assert_eq!(depth, 0, "every DIE with children is closed");
        dies
    }

    fn compiled(text: &str) -> crate::rir::CompiledKernel {
        crate::rir::compile_any_target(text, Target::Gfx950).unwrap_or_else(|e| panic!("{e}"))
    }

    #[test]
    fn locations_leave_the_code_alone_and_reach_every_instruction() {
        let plain = compiled(&probe(""));
        assert!(plain.srcmap.is_none() && plain.source_map.is_none());
        assert!(section(&plain.hsaco, ".debug_info").is_none());
        // `debug_info` alone changes nothing.
        assert_eq!(compiled(&probe(", debug_info = b:true")).hsaco, plain.hsaco);

        let with = compiled(&located(&probe(", debug_info = b:true")));
        assert_eq!(with.listing(), plain.listing());
        // Everything the loader reads is unchanged: past the header, all
        // before the section headers; in the header, all but where they are.
        let table = u64::from_le_bytes(plain.hsaco[0x28..0x30].try_into().unwrap()) as usize;
        assert_eq!(with.hsaco[0x40..table], plain.hsaco[0x40..table]);
        for field in [0..0x28, 0x30..0x3c, 0x3e..0x40] {
            assert_eq!(with.hsaco[field.clone()], plain.hsaco[field]);
        }
        assert_eq!(crate::hsaco_writer::text_section(&with.hsaco), crate::hsaco_writer::text_section(&plain.hsaco));
        crate::elf::validate_hsaco(&with.hsaco, &["loc_probe".to_string()]).unwrap_or_else(|e| panic!("{e}"));

        let map = with.source_map.as_ref().expect("a source map");
        let mut at = 0;
        for row in &map.rows {
            assert_eq!(row.offset, at);
            at += row.size;
            assert!(row.chain.is_some(), "{row:?}");
        }
        assert_eq!(at, map.code_bytes);
        assert_eq!(plain.stats.code_bytes as u32, map.code_bytes);
        assert!(map.chains.iter().all(|c| c.len() == 3 && c[0].file == "/w/semi;colon.py" && c[2].func == "build"));
        let load = map.rows.iter().find(|r| r.mnemonic.starts_with("global_load")).expect("the load");
        assert!(!load.synthetic);
        let wait = map.rows.iter().find(|r| r.mnemonic == "s_waitcnt").expect("a wait");
        assert!(wait.synthetic);
        let stack = map.stack(wait);
        assert_eq!(stack.len(), 4);
        assert_eq!(stack[0], ("loc_probe".to_string(), None));
        assert_eq!(stack[3].0, "rocke.s_waitcnt");

        let json = with.srcmap.as_ref().expect("the JSON map");
        assert!(json.starts_with("{\"version\":1,\"schema\":\"rocke-srcmap\",\"kernel\":\"loc_probe\",\"target\":\"gfx950\""));
        assert!(json.contains("\"files\":[\"/w/semi;colon.py\""));
        assert_eq!(json.matches("\"s_waitcnt\"").count(), map.rows.iter().filter(|r| r.mnemonic == "s_waitcnt").count());

        // The DWARF says the same, read back the way `emit_inline_frames.py`
        // reads llvm-dwarfdump: the frames covering an address, by depth.
        let (text_vaddr, _) = crate::hsaco_writer::text_section(&with.hsaco).unwrap();
        let (files, lines) = decode_lines(section(&with.hsaco, ".debug_line").unwrap());
        let dies = decode_info(section(&with.hsaco, ".debug_info").unwrap(), section(&with.hsaco, ".debug_abbrev").unwrap());
        let strs = section(&with.hsaco, ".debug_str").unwrap();
        let ranges = section(&with.hsaco, ".debug_ranges").unwrap();
        let string = |at: u64| Reader { b: strs, at: at as usize }.cstr();
        let mut frames = Vec::new();
        for (depth, tag, attrs) in dies.values() {
            if *tag != DW_TAG_SUBPROGRAM && *tag != DW_TAG_INLINED_SUBROUTINE {
                continue;
            }
            let name = match attrs.get(&DW_AT_ABSTRACT_ORIGIN) {
                Some(&origin) => string(dies[&(origin as usize)].2[&DW_AT_NAME]),
                None => string(attrs[&DW_AT_NAME]),
            };
            let mut covered = Vec::new();
            if let Some(&list) = attrs.get(&DW_AT_RANGES) {
                let mut r = Reader { b: ranges, at: list as usize };
                loop {
                    let (lo, hi) = (r.n(8), r.n(8));
                    if (lo, hi) == (0, 0) {
                        break;
                    }
                    covered.push((text_vaddr + lo, text_vaddr + hi));
                }
                assert!(covered.len() > 1);
            } else if let Some(&lo) = attrs.get(&DW_AT_LOW_PC) {
                covered.push((lo, attrs[&DW_AT_HIGH_PC]));
            }
            let call = attrs.get(&DW_AT_CALL_FILE).map(|&f| (files[f as usize - 1].clone(), attrs[&DW_AT_CALL_LINE], attrs[&DW_AT_CALL_COLUMN]));
            if !covered.is_empty() {
                frames.push((*depth, name, covered, call));
            }
        }
        for row in &map.rows {
            let addr = text_vaddr + u64::from(row.offset);
            let mut hits: Vec<_> = frames.iter().filter(|f| f.2.iter().any(|&(lo, hi)| lo <= addr && addr < hi)).collect();
            hits.sort_by_key(|f| f.0);
            let read: Vec<_> = hits.iter().map(|f| (f.1.clone(), f.3.clone())).collect();
            let want: Vec<_> = map.stack(row).into_iter().map(|(name, call)| (name, call.map(|c| (c.file, u64::from(c.line), u64::from(c.col))))).collect();
            assert_eq!(read, want, "{row:?}");
            let innermost = &map.chains[row.chain.unwrap()][0];
            let line = lines.iter().rev().find(|l| l.0 <= addr && l.1 != 0).expect("a line row");
            assert_eq!((files[line.1 as usize - 1].as_str(), line.2, line.3), (innermost.file.as_str(), u64::from(innermost.line), u64::from(innermost.col)));
        }
        assert_eq!(lines.last().map(|l| l.0), Some(text_vaddr + u64::from(map.code_bytes)), "the sequence ends at the end of the code");
    }

    #[test]
    fn a_preloading_kernel_maps_its_code_past_the_entry() {
        let preloading = probe(", debug_info = b:true").replace("global>\n", "global> { inreg = b:true }\n");
        let with = compiled(&located(&preloading));
        let map = with.source_map.as_ref().expect("a source map");
        assert_eq!(map.rows[0].offset, 256);
        assert_eq!(map.rows.last().map(|r| r.offset + r.size), Some(map.code_bytes));
        assert_eq!(with.stats.code_bytes as u32, map.code_bytes);
        let (text_vaddr, _) = crate::hsaco_writer::text_section(&with.hsaco).unwrap();
        let (_, lines) = decode_lines(section(&with.hsaco, ".debug_line").unwrap());
        assert_eq!(lines.iter().find(|l| l.1 != 0).map(|l| l.0), Some(text_vaddr + 256));
    }

    #[test]
    fn leb128_round_trips() {
        let mut out = Vec::new();
        uleb(&mut out, 624485);
        assert_eq!(out, [0xe5, 0x8e, 0x26]);
        out.clear();
        sleb(&mut out, -123456);
        assert_eq!(out, [0xc0, 0xbb, 0x78]);
        out.clear();
        sleb(&mut out, 63);
        sleb(&mut out, 64);
        sleb(&mut out, -64);
        assert_eq!(out, [0x3f, 0xc0, 0x00, 0x40]);
    }
}
