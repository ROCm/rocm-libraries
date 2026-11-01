// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

//! Strict parser for `ck.dsl.ir/v1` text (see `rocke.core.ir_serialize`).
//!
//! Anything outside the subset the direct backend implements is an error, so
//! an unsupported kernel fails loudly instead of compiling to wrong code.
//!
//! The sequential parser here is the reference. Large inputs are parsed in
//! chunks on several threads by `parse_par`; whenever that path meets anything
//! it does not handle, including every error, the whole input is parsed here
//! instead, so the kernel or the error is always this parser's.
//!
//! `ROCKE_DIRECT_PARSE_THREADS=<n>` forces the thread count (1 parses
//! sequentially); it changes speed only, never the result.

use std::borrow::Cow;

use std::hash::Hasher;

use super::ast::{Attr, Kernel, Op, OpKind, Operands, Param, Region, Results, Scalar, Text, Type, ValueDef, ValueId};
use super::hashcons::{key_hash, ConstTable};
use super::scan::{self, Level, Masks};
use super::scan_line::{canonical_with, const_key_with};
use crate::error::CompileError;

/// Inputs smaller than this are parsed on the calling thread: starting
/// threads, and the later passes reading an AST other cores wrote, cost about
/// what the threads save.
const PARALLEL_MIN_BYTES: usize = 256 << 10;
/// Input bytes per parse thread; thread start-up is serialized and costly,
/// so the count grows with the input instead of always using the cap.
const BYTES_PER_THREAD: usize = 128 << 10;
/// Most threads one parse uses. More make the parse itself a little faster
/// but not the compile, whose later passes then read an AST spread over more
/// cores' caches; and each costs a serialized thread start.
const MAX_THREADS: usize = 4;

pub(super) fn err(message: impl Into<String>) -> CompileError {
    CompileError::new("ROCKE_DIRECT_PARSE", message)
}

/// Well-formed rocKE IR this compiler does not lower: `ROCKE_DIRECT_UNSUPPORTED`,
/// on which `rocke.helpers.compile_kernel` compiles the kernel with COMGR.
/// Text that is not rocKE IR stays `ROCKE_DIRECT_PARSE`.
pub(super) fn unsupported(message: impl Into<String>) -> CompileError {
    CompileError::new("ROCKE_DIRECT_UNSUPPORTED", message)
}

/// The dialects of rocKE IR op names (`rocke.core.ir.IRBuilder`).
const DIALECTS: [&str; 8] = ["arith.", "math.", "vector.", "memref.", "tile.", "gpu.", "scf.", "cf."];

fn is_ident(text: &str) -> bool {
    !text.is_empty() && text.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_')
}

/// The error for an op name `OpKind::from_name` does not know: a name in a
/// rocKE IR dialect is an op rocKE defines (its serializer emits no other)
/// that this compiler does not lower, so it is refused as unsupported; any
/// other name is a parse error.
pub(super) fn unknown_op(opname: &str) -> CompileError {
    if DIALECTS.iter().any(|d| opname.strip_prefix(d).is_some_and(is_ident)) {
        unsupported(format!("op {opname:?} is not lowered by the direct backend"))
    } else {
        err(format!("op {opname:?} is not supported by the direct backend"))
    }
}

/// The pointee of a `ptr<T,constant>` spelling (a scalar `T`), `None` for
/// any other type text.
fn constant_pointer(text: &str) -> Option<Result<Scalar, CompileError>> {
    let inner = text.trim().strip_prefix("ptr<")?.strip_suffix(",constant>")?;
    Some(parse_scalar(inner.trim()))
}

pub fn parse_type(text: &str) -> Result<Type, CompileError> {
    let text = text.trim();
    if let Some(inner) = text.strip_prefix("vec<").and_then(|rest| rest.strip_suffix('>')) {
        let split = inner
            .rfind('x')
            .filter(|&index| inner[index + 1..].chars().all(|ch| ch.is_ascii_digit()))
            .ok_or_else(|| err(format!("malformed vector type {text:?}")))?;
        let elem = parse_scalar(&inner[..split])?;
        let len: u32 = inner[split + 1..]
            .parse()
            .map_err(|_| err(format!("malformed vector length in {text:?}")))?;
        return Ok(Type::Vec(elem, len));
    }
    if let Some(inner) = text.strip_prefix("ptr<").and_then(|rest| rest.strip_suffix('>')) {
        let comma = inner
            .rfind(',')
            .ok_or_else(|| err(format!("malformed pointer type {text:?}")))?;
        let space = inner[comma + 1..].trim();
        if space != "global" {
            let message = format!("only global pointers are supported, got {text:?}");
            return Err(if is_ident(space) { unsupported(message) } else { err(message) });
        }
        let pointee = inner[..comma].trim();
        if pointee.starts_with("vec<") || pointee.starts_with("ptr<") {
            // rocKE IR has pointers to vectors (`PtrType(VectorType)`); this compiler's pointers are to scalars.
            parse_type(pointee)?;
            return Err(unsupported(format!("a pointer to {pointee} ({text:?})")));
        }
        return Ok(Type::Ptr(parse_scalar(pointee)?));
    }
    if let Some(inner) = text.strip_prefix("smem<").and_then(|rest| rest.strip_suffix('>')) {
        let open = inner.rfind('[').ok_or_else(|| err(format!("malformed smem type {text:?}")))?;
        let close = inner.rfind(']').ok_or_else(|| err(format!("malformed smem type {text:?}")))?;
        let elem = parse_scalar(inner[..open].trim_end().trim_end_matches(',').trim())?;
        let shape_text = inner[open + 1..close].trim();
        let shape = if shape_text.is_empty() {
            Vec::new()
        } else {
            shape_text
                .split('x')
                .map(|dim| dim.parse::<u32>().map_err(|_| err(format!("bad smem shape {text:?}"))))
                .collect::<Result<Vec<_>, _>>()?
        };
        return Ok(Type::Smem(elem, shape));
    }
    Ok(Type::Scalar(parse_scalar(text)?))
}

fn parse_scalar(text: &str) -> Result<Scalar, CompileError> {
    match text.trim() {
        "i1" => Ok(Scalar::I1),
        "i32" => Ok(Scalar::I32),
        "i64" => Ok(Scalar::I64),
        "f16" => Ok(Scalar::F16),
        "f32" => Ok(Scalar::F32),
        "bf16" => Ok(Scalar::BF16),
        "i16" => Ok(Scalar::I16),
        "i8" => Ok(Scalar::I8),
        "fp8e4m3" => Ok(Scalar::FP8E4M3),
        "bf8e5m2" => Ok(Scalar::BF8E5M2),
        "f64" => Ok(Scalar::F64),
        "tf32" => Ok(Scalar::TF32),
        other if is_ident(other) => Err(unsupported(format!("unsupported element type {other:?}"))),
        other => Err(err(format!("malformed element type {other:?}"))),
    }
}

fn scalar_of(text: &[u8]) -> Option<Scalar> {
    Some(match text {
        b"i1" => Scalar::I1,
        b"i32" => Scalar::I32,
        b"i64" => Scalar::I64,
        b"f16" => Scalar::F16,
        b"f32" => Scalar::F32,
        b"bf16" => Scalar::BF16,
        b"i16" => Scalar::I16,
        b"i8" => Scalar::I8,
        b"fp8e4m3" => Scalar::FP8E4M3,
        b"bf8e5m2" => Scalar::BF8E5M2,
        b"f64" => Scalar::F64,
        b"tf32" => Scalar::TF32,
        _ => return None,
    })
}

/// `parse_type(text)` for the spellings the DSL emits for values (no spaces),
/// `None` for anything else.
pub(super) fn quick_type(text: &[u8]) -> Option<Type> {
    if let Some(scalar) = scalar_of(text) {
        return Some(Type::Scalar(scalar));
    }
    if let Some(inner) = text.strip_prefix(b"vec<").and_then(|rest| rest.strip_suffix(b">")) {
        let x = inner.iter().rposition(|&b| b == b'x')?;
        let digits = &inner[x + 1..];
        if digits.is_empty() || digits.len() > 9 || !digits.iter().all(u8::is_ascii_digit) {
            return None;
        }
        let len = digits.iter().fold(0u32, |n, &d| n * 10 + (d - b'0') as u32);
        return Some(Type::Vec(scalar_of(&inner[..x])?, len));
    }
    let inner = text.strip_prefix(b"ptr<")?.strip_suffix(b",global>")?;
    Some(Type::Ptr(scalar_of(inner)?))
}

macro_rules! interned {
    ($($text:literal),* $(,)?) => {
        /// `text` as a static string when it is one the DSL emits, so that
        /// parsing it allocates nothing.
        fn interned(text: &str) -> Text {
            match text {
                $($text => Cow::Borrowed($text),)*
                _ => Cow::Owned(text.to_owned()),
            }
        }
    };
}

interned!(
    "ity", "value", "pred", "wrap", "axis", "index", "align", "vec", "elem_type", "rank", "elem", "op_id",
    "target", "fill", "constraints", "convergent", "sideeffect", "template", "fence", "name", "type",
    "elide_trailing_barrier", "iter_args", "iv", "iv_type", "num_iter_args", "unroll", "num", "offset",
    "dwords", "xor_mask", "exclusive", "max_workgroup_size", "scheduler_strategy", "i1", "i32", "i64",
    "f16", "f32", "lt", "le", "gt", "ge", "eq", "ne", "ult", "olt", "ole", "ogt", "oge", "oeq", "one",
    "une", "x", "y", "z", "modular", "", "=v,0", "wmma_f32_16x16x16_f16",
    "bf16", "wmma_f32_16x16x16_bf16", "i16", "i8", "wmma_i32_16x16x16_iu8", "wmma_i32_16x16x16_iu4",
    "fp8e4m3", "bf8e5m2", "f64",
    "ordering", "scope", "monotonic", "acquire", "release", "agent", "workgroup",
);

fn spaced_at(bytes: &[u8], at: usize) -> bool {
    at > 0 && at + 1 < bytes.len() && bytes[at - 1] == b' ' && bytes[at + 1] == b' '
}

/// First index of ` c ` in `text`, for an ASCII `c`.
fn find_spaced(text: &str, c: char) -> Option<usize> {
    let mut from = 0;
    loop {
        let at = from + text[from..].find(c)?;
        if spaced_at(text.as_bytes(), at) {
            return Some(at - 1);
        }
        from = at + 1;
    }
}

/// Last index of ` c ` in `text`, for an ASCII `c`.
fn rfind_spaced(text: &str, c: char) -> Option<usize> {
    let mut end = text.len();
    loop {
        let at = text[..end].rfind(c)?;
        if spaced_at(text.as_bytes(), at) {
            return Some(at - 1);
        }
        end = at;
    }
}

struct Scanner<'a> {
    src: &'a str,
    text: &'a [u8],
    pos: usize,
}

impl<'a> Scanner<'a> {
    fn new(text: &'a str) -> Self {
        Self {
            src: text,
            text: text.as_bytes(),
            pos: 0,
        }
    }

    fn skip_ws(&mut self) {
        while self.pos < self.text.len() && matches!(self.text[self.pos], b' ' | b'\t') {
            self.pos += 1;
        }
    }

    fn peek(&self) -> Option<u8> {
        self.text.get(self.pos).copied()
    }

    fn starts_with(&self, prefix: &str) -> bool {
        self.text[self.pos..].starts_with(prefix.as_bytes())
    }

    fn expect(&mut self, ch: u8) -> Result<(), CompileError> {
        self.skip_ws();
        if self.peek() != Some(ch) {
            return Err(err(format!(
                "expected {:?} at offset {} in {:?}",
                ch as char,
                self.pos,
                String::from_utf8_lossy(self.text)
            )));
        }
        self.pos += 1;
        Ok(())
    }

    /// `text[start..end]`, or `or` when that is not whole characters.
    fn slice(&self, start: usize, end: usize, or: &'a str) -> &'a str {
        self.src.get(start..end).unwrap_or(or)
    }

    fn rest(&self) -> &'a str {
        self.slice(self.pos, self.text.len(), "")
    }

    fn scalar_token(&mut self) -> &'a str {
        let start = self.pos;
        while self.pos < self.text.len() && !matches!(self.text[self.pos], b',' | b' ' | b'}' | b']' | b'\t') {
            self.pos += 1;
        }
        self.slice(start, self.pos, "")
    }

    fn quoted(&mut self) -> Result<Text, CompileError> {
        let start = self.pos;
        if let Some(len) = self.text[start..].iter().position(|&ch| ch == b'"' || ch == b'\\') {
            if self.text[start + len] == b'"' {
                self.pos = start + len + 1;
                return Ok(interned(self.slice(start, start + len, "?")));
            }
        }
        let mut out = String::new();
        while self.pos < self.text.len() {
            let ch = self.text[self.pos];
            if ch == b'\\' && self.pos + 1 < self.text.len() {
                out.push(match self.text[self.pos + 1] {
                    b'n' => '\n',
                    b't' => '\t',
                    other => other as char,
                });
                self.pos += 2;
                continue;
            }
            if ch == b'"' {
                self.pos += 1;
                return Ok(Cow::Owned(out));
            }
            // Multi-byte UTF-8 sequences only occur inside strings; copy them whole.
            let width = utf8_width(ch);
            out.push_str(std::str::from_utf8(&self.text[self.pos..self.pos + width]).unwrap_or("?"));
            self.pos += width;
        }
        Err(err("unterminated string literal"))
    }

    fn ident(&mut self) -> &'a str {
        let start = self.pos;
        while self.pos < self.text.len()
            && (self.text[self.pos].is_ascii_alphanumeric() || matches!(self.text[self.pos], b'_' | b'.'))
        {
            self.pos += 1;
        }
        self.slice(start, self.pos, "")
    }
}

fn utf8_width(first: u8) -> usize {
    match first {
        0xf0..=0xff => 4,
        0xe0..=0xef => 3,
        0xc0..=0xdf => 2,
        _ => 1,
    }
}

/// Deepest nesting of attribute maps and lists the sequential parser takes.
/// Their parse recurses; the DSL nests three levels.
const MAX_ATTR_NESTING: u32 = 1024;

/// Parses `{ key = value, ... }`. Maps and lists may nest `depth` levels deep;
/// the sequential parser passes `MAX_ATTR_NESTING`, a worker thread with a
/// small stack a bound it can afford.
fn parse_attr_map(scanner: &mut Scanner<'_>, depth: u32) -> Result<Vec<(Text, Attr)>, CompileError> {
    if depth == 0 {
        return Err(err("attributes nested too deeply"));
    }
    scanner.expect(b'{')?;
    let mut out = Vec::new();
    loop {
        scanner.skip_ws();
        match scanner.peek() {
            Some(b'}') => {
                scanner.pos += 1;
                return Ok(out);
            }
            Some(b',') => {
                scanner.pos += 1;
                continue;
            }
            None => return Err(err("unterminated attribute map")),
            _ => {}
        }
        let key = scanner.ident();
        if key.is_empty() {
            return Err(err(format!("bad attribute key at {:?}", scanner.rest())));
        }
        let key = interned(key);
        scanner.expect(b'=')?;
        let value = parse_attr_value(scanner, depth)?;
        out.push((key, value));
    }
}

fn parse_float(token: &str) -> Result<f64, CompileError> {
    match token {
        "inf" => Ok(f64::INFINITY),
        "-inf" => Ok(f64::NEG_INFINITY),
        "nan" | "-nan" => Ok(f64::NAN),
        _ => token.parse::<f64>().map_err(|_| err(format!("bad float literal {token:?}"))),
    }
}

fn parse_attr_value(scanner: &mut Scanner<'_>, depth: u32) -> Result<Attr, CompileError> {
    scanner.skip_ws();
    match scanner.peek() {
        Some(b'{') => Ok(Attr::Map(parse_attr_map(scanner, depth - 1)?)),
        Some(b'l') if scanner.starts_with("l:") => {
            if depth == 1 {
                return Err(err("attributes nested too deeply"));
            }
            scanner.pos += 2;
            scanner.expect(b'[')?;
            let mut items = Vec::new();
            loop {
                scanner.skip_ws();
                match scanner.peek() {
                    Some(b']') => {
                        scanner.pos += 1;
                        return Ok(Attr::List(items));
                    }
                    Some(b',') => {
                        scanner.pos += 1;
                    }
                    None => return Err(err("unterminated list attribute")),
                    _ => items.push(parse_attr_value(scanner, depth - 1)?),
                }
            }
        }
        Some(b'b') if scanner.starts_with("b:true") => {
            scanner.pos += 6;
            Ok(Attr::Bool(true))
        }
        Some(b'b') if scanner.starts_with("b:false") => {
            scanner.pos += 7;
            Ok(Attr::Bool(false))
        }
        Some(b'i') if scanner.starts_with("i:") => {
            scanner.pos += 2;
            let token = scanner.scalar_token();
            token
                .parse::<i64>()
                .map(Attr::Int)
                .map_err(|_| err(format!("bad integer literal {token:?}")))
        }
        Some(b'f') if scanner.starts_with("f:") => {
            scanner.pos += 2;
            Ok(Attr::Float(parse_float(scanner.scalar_token())?))
        }
        Some(b's') if scanner.starts_with("s:") => {
            scanner.pos += 2;
            scanner.expect(b'"')?;
            Ok(Attr::Str(scanner.quoted()?))
        }
        _ => Err(err(format!("unknown attribute value at {:?}", scanner.rest()))),
    }
}

/// `str::trim`, with a fast path for ASCII ends.
fn trim(text: &str) -> &str {
    let is_ws = |b: u8| matches!(b, b' ' | b'\t' | b'\n' | b'\x0b' | b'\x0c' | b'\r');
    let bytes = text.as_bytes();
    let (mut start, mut end) = (0, bytes.len());
    while start < end && is_ws(bytes[start]) {
        start += 1;
    }
    while end > start && is_ws(bytes[end - 1]) {
        end -= 1;
    }
    let text = &text[start..end];
    match (text.as_bytes().first(), text.as_bytes().last()) {
        (Some(&a), Some(&b)) if a >= 0x80 || b >= 0x80 => text.trim(),
        _ => text,
    }
}

/// Trimmed lines of `text[pos..end]`, skipping blank lines and `#` comments.
/// `end` is a line start or the end of the text.
pub(super) struct Lines<'a> {
    text: &'a str,
    pos: usize,
    end: usize,
    peeked: Option<(&'a str, usize)>,
    /// Vector level of the newline search (`scan`).
    level: Level,
}

impl<'a> Lines<'a> {
    pub(super) fn new(text: &'a str, pos: usize, end: usize) -> Self {
        Self { text, pos, end, peeked: None, level: Level::Scalar }
    }

    /// The same lines, found with the vector level `level`.
    pub(super) fn scanning(self, level: Level) -> Self {
        Self { level, ..self }
    }

    pub(super) fn peek(&mut self) -> Option<&'a str> {
        if let Some((line, _)) = self.peeked {
            return Some(line);
        }
        let mut pos = self.pos;
        while pos < self.end {
            let end = match self.level {
                Level::Scalar => self.text[pos..self.end].find('\n').map_or(self.end, |n| pos + n),
                level => scan::find_newline(level, &self.text.as_bytes()[..self.end], pos).unwrap_or(self.end),
            };
            let line = trim(&self.text[pos..end]);
            pos = (end + 1).min(self.end);
            if !(line.is_empty() || line.starts_with('#')) {
                self.peeked = Some((line, pos));
                return Some(line);
            }
        }
        self.pos = pos;
        None
    }

    pub(super) fn next_line(&mut self) -> Option<&'a str> {
        let line = self.peek()?;
        let (_, after) = self.peeked.take().expect("peeked");
        self.pos = after;
        Some(line)
    }

    fn next(&mut self) -> Result<&'a str, CompileError> {
        self.next_line().ok_or_else(|| err("unexpected end of input"))
    }

    /// Byte offset just past the last line returned.
    pub(super) fn pos(&self) -> usize {
        self.pos
    }

    fn region_follows(&mut self) -> bool {
        self.peek().is_some_and(|next| next.starts_with("region @"))
    }
}

/// Top-level comma-separated entries of `text` (commas inside `<>`, `[]`,
/// `{}` and `()` do not split).
fn split_top(text: &str) -> impl Iterator<Item = &str> {
    let bytes = text.as_bytes();
    let (mut depth, mut start, mut index) = (0i32, 0, 0);
    let mut done = false;
    std::iter::from_fn(move || {
        if done {
            return None;
        }
        while index < bytes.len() {
            index += 1;
            match bytes[index - 1] {
                b'<' | b'[' | b'{' | b'(' => depth += 1,
                b'>' | b']' | b'}' | b')' => depth -= 1,
                b',' if depth == 0 => {
                    let entry = &text[start..index - 1];
                    start = index;
                    return Some(entry);
                }
                _ => {}
            }
        }
        done = true;
        Some(&text[start..])
    })
}

fn is_name_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_' || b == b'.'
}

/// The parts of a line in the shape the DSL prints,
/// `[%name : type = ]op ( [%a[, %b ...]] )[ tail]`, with a type `quick_type`
/// knows. For such a line the general scan in `scan_op` finds exactly these
/// parts: no earlier ` ( ` or later ` = ` can occur, the result entry cannot
/// split, and the first `)` closes the operand list.
pub(super) struct Canonical<'a> {
    pub(super) result: Option<(&'a str, Type)>,
    pub(super) opname: &'a str,
    /// The operand names joined by `, `.
    pub(super) operands: &'a str,
    /// Text after the closing `)`.
    pub(super) tail: &'a str,
}

pub(super) fn canonical(line: &str) -> Option<Canonical<'_>> {
    let b = line.as_bytes();
    let n = b.len();
    let mut i = 0;
    let mut result = None;
    if b.first() == Some(&b'%') {
        i = 1;
        while i < n && is_name_byte(b[i]) {
            i += 1;
        }
        if i == 1 || !b[i..].starts_with(b" : ") {
            return None;
        }
        let ty_start = i + 3;
        let mut j = ty_start;
        while j < n && b[j] != b' ' {
            j += 1;
        }
        if !b[j..].starts_with(b" = ") {
            return None;
        }
        result = Some((&line[..i], quick_type(&b[ty_start..j])?));
        i = j + 3;
    }
    let op_start = i;
    while i < n && is_name_byte(b[i]) {
        i += 1;
    }
    if i == op_start || !b[i..].starts_with(b" ( ") {
        return None;
    }
    let opname = &line[op_start..i];
    i += 3;
    let operands_start = i;
    let close = if b[i..].starts_with(b" )") {
        i + 1
    } else {
        loop {
            if b.get(i) != Some(&b'%') {
                return None;
            }
            i += 1;
            let name_start = i;
            while i < n && is_name_byte(b[i]) {
                i += 1;
            }
            if i == name_start {
                return None;
            }
            if b[i..].starts_with(b", ") {
                i += 2;
            } else if b[i..].starts_with(b" )") {
                break i + 1;
            } else {
                return None;
            }
        }
    };
    Some(Canonical {
        result,
        opname,
        operands: line.get(operands_start..close.saturating_sub(1).max(operands_start)).unwrap_or(""),
        tail: &line[close + 1..],
    })
}

#[derive(Clone, Copy)]
enum Flat {
    Str,
    Int,
    Float,
    Bool(bool),
}

/// `parse_attr_map` for a flat map as the DSL prints it,
/// `{ key = value, ... }` with strings without escapes, integers, floats and
/// booleans: the attributes and the length of text they span. `None` for
/// anything else, including values the general parser would reject.
fn flat_attrs(text: &str) -> Option<(Vec<(Text, Attr)>, usize)> {
    const MAX: usize = 8;
    let b = text.as_bytes();
    if !b.starts_with(b"{ ") {
        return None;
    }
    let mut parts = [(0usize, 0usize, Flat::Str, 0usize, 0usize); MAX];
    let mut n = 0;
    let mut i = 2;
    loop {
        if n == MAX {
            return None;
        }
        let key = i;
        while i < b.len() && is_name_byte(b[i]) {
            i += 1;
        }
        if i == key || !b[i..].starts_with(b" = ") {
            return None;
        }
        let key_end = i;
        i += 3;
        let (kind, value, value_end) = match b.get(i..i + 2)? {
            b"s:" => {
                if b.get(i + 2) != Some(&b'"') {
                    return None;
                }
                let start = i + 3;
                let len = b[start..].iter().position(|&c| c == b'"' || c == b'\\')?;
                if b[start + len] != b'"' {
                    return None;
                }
                i = start + len + 1;
                (Flat::Str, start, start + len)
            }
            prefix @ (b"i:" | b"f:") => {
                let start = i + 2;
                i = start;
                while i < b.len() && !matches!(b[i], b',' | b' ' | b'}' | b']' | b'\t') {
                    i += 1;
                }
                (if prefix == b"i:" { Flat::Int } else { Flat::Float }, start, i)
            }
            b"b:" if b[i..].starts_with(b"b:true") => {
                i += 6;
                (Flat::Bool(true), i, i)
            }
            b"b:" if b[i..].starts_with(b"b:false") => {
                i += 7;
                (Flat::Bool(false), i, i)
            }
            _ => return None,
        };
        parts[n] = (key, key_end, kind, value, value_end);
        n += 1;
        if b[i..].starts_with(b", ") {
            i += 2;
        } else if b[i..].starts_with(b" }") {
            i += 2;
            break;
        } else {
            return None;
        }
    }
    let mut attrs = Vec::with_capacity(n);
    for &(key, key_end, kind, value, value_end) in &parts[..n] {
        let token = &text[value..value_end];
        let attr = match kind {
            Flat::Str => Attr::Str(interned(token)),
            Flat::Int => Attr::Int(token.parse().ok()?),
            Flat::Float => Attr::Float(parse_float(token).ok()?),
            Flat::Bool(v) => Attr::Bool(v),
        };
        attrs.push((interned(&text[key..key_end]), attr));
    }
    Some((attrs, i))
}

/// Attributes after an operand list, and the `@loc` or nothing after them.
fn scan_tail(tail: &str, depth: u32) -> Result<Vec<(Text, Attr)>, CompileError> {
    let tail = trim(tail);
    if let Some((attrs, used)) = flat_attrs(tail) {
        let rest = trim(&tail[used..]);
        if !(rest.is_empty() || rest.starts_with("@loc")) {
            return Err(err(format!("unexpected trailing text {rest:?}")));
        }
        return Ok(attrs);
    }
    if tail.starts_with('{') {
        let mut scanner = Scanner::new(tail);
        let attrs = parse_attr_map(&mut scanner, depth)?;
        let rest = trim(scanner.rest());
        if !(rest.is_empty() || rest.starts_with("@loc")) {
            return Err(err(format!("unexpected trailing text {rest:?}")));
        }
        Ok(attrs)
    } else if !(tail.is_empty() || tail.starts_with("@loc")) {
        Err(err(format!("unexpected trailing text {tail:?}")))
    } else {
        Ok(Vec::new())
    }
}

pub(super) struct ScannedOp<'a> {
    pub kind: OpKind,
    pub opname: &'a str,
    pub attrs: Vec<(Text, Attr)>,
}

/// Scans one op line: the kind, each operand through `operand` (in order,
/// with the op name for its error), the attributes, and the result names and
/// types appended to `results`. Errors come in the order the checks are
/// written here, which is the order of the original single-pass parser.
/// `masks` are the line's byte classes (`scan`), if the caller has them.
pub(super) fn scan_op<'a>(
    line: &'a str,
    masks: Option<&Masks>,
    operands: &mut Operands,
    results: &mut Vec<(&'a str, Type)>,
    mut operand: impl FnMut(&'a str, &'a str) -> Result<ValueId, CompileError>,
    depth: u32,
) -> Result<ScannedOp<'a>, CompileError> {
    if let Some(parts) = canonical_with(line, masks) {
        let opname = parts.opname;
        let kind = OpKind::from_name(opname)
            .ok_or_else(|| unknown_op(opname))?;
        // Names hold no commas, and `canonical` checked each is followed by `, `.
        let names = parts.operands;
        let mut start = 0;
        while start < names.len() {
            let end = names.as_bytes()[start..].iter().position(|&b| b == b',').map_or(names.len(), |n| start + n);
            operands.push(operand(&names[start..end], opname)?);
            start = end + 2;
        }
        let attrs = scan_tail(parts.tail, depth)?;
        if let Some(result) = parts.result {
            results.push(result);
        }
        return Ok(ScannedOp { kind, opname, attrs });
    }

    let paren = find_spaced(line, '(').ok_or_else(|| err(format!("op line without operand list: {line:?}")))?;
    let (results_text, opname) = match rfind_spaced(&line[..paren], '=') {
        Some(eq) => (&line[..eq], trim(&line[eq + 3..paren])),
        None => ("", trim(&line[..paren])),
    };
    let kind = OpKind::from_name(opname)
        .ok_or_else(|| unknown_op(opname))?;

    let after = &line[paren + 3..];
    let close = after
        .bytes()
        .position(|b| b == b')')
        .ok_or_else(|| err(format!("unterminated operand list: {line:?}")))?;
    let operand_text = &after[..close];
    let mut start = 0;
    loop {
        let end = operand_text.as_bytes()[start..].iter().position(|&b| b == b',').map_or(close, |n| start + n);
        let name = trim(&operand_text[start..end]);
        if !name.is_empty() {
            operands.push(operand(name, opname)?);
        }
        if end == close {
            break;
        }
        start = end + 1;
    }

    let mut attrs = scan_tail(&after[close + 1..], depth)?;

    if !results_text.is_empty() {
        for entry in split_top(results_text) {
            let entry = trim(entry);
            let colon = find_spaced(entry, ':').ok_or_else(|| err(format!("bad result entry {entry:?}")))?;
            let text = &entry[colon + 3..];
            let ty = match constant_pointer(text) {
                // `ptr<T,constant>` (`ptr addrspace(4)`): only `tile.int_to_ptr`
                // makes one; the value is a plain pointer whose loads are
                // invariant (`space = "constant"`, `opt` marks them).
                Some(scalar) if kind == OpKind::IntToPtr => {
                    attrs.push(("space".into(), Attr::Str("constant".into())));
                    Type::Ptr(scalar?)
                }
                Some(_) => return Err(unsupported(format!("a constant pointer made by {opname} ({text:?})"))),
                None => parse_type(text)?,
            };
            results.push((trim(&entry[..colon]), ty));
        }
    }
    Ok(ScannedOp { kind, opname, attrs })
}

/// The induction variable name and the iteration arguments of an `scf.for`
/// with `results` results, checked in the sequential parser's order.
pub(super) fn for_args(op: &Op, results: usize) -> Result<(String, Vec<(String, Type)>), CompileError> {
    let iv_name = op
        .str_attr("iv")
        .ok_or_else(|| err("scf.for without iv"))?
        .to_string();
    let iv_type = parse_type(op.str_attr("iv_type").unwrap_or("i32"))?;
    if iv_type != Type::Scalar(Scalar::I32) {
        return Err(err("scf.for induction variables must be i32"));
    }
    let mut iter_args = Vec::new();
    if let Some(Attr::List(entries)) = op.attr("iter_args") {
        for entry in entries {
            let Attr::Map(fields) = entry else {
                return Err(err("malformed iter_args entry"));
            };
            let field = |key: &str| {
                fields.iter().find_map(|(name, value)| match value {
                    Attr::Str(text) if name == key => Some(text.to_string()),
                    _ => None,
                })
            };
            let name = field("name").ok_or_else(|| err("iter_arg without name"))?;
            let ty = parse_type(&field("type").ok_or_else(|| err("iter_arg without type"))?)?;
            iter_args.push((name, ty));
        }
    }
    if iter_args.len() + 3 != op.operands.len() || iter_args.len() != results {
        return Err(err("scf.for operand/iter_arg/result counts disagree"));
    }
    // `unroll = true` is honored by `unroll::unroll_loops` before optimization.
    Ok((iv_name, iter_args))
}

/// Number of regions an op of `kind` must have.
pub(super) fn expected_regions(kind: OpKind) -> usize {
    match kind {
        OpKind::For | OpKind::If => 1,
        OpKind::IfElse => 2,
        _ => 0,
    }
}

/// The lines before the body.
pub(super) struct Prologue<'a> {
    pub name: String,
    pub max_workgroup_size: u32,
    pub waves_per_eu: (u32, u32),
    pub agpr_limit: Option<u32>,
    /// Parameter names without the `%`, their types, and whether they are
    /// `noalias` and `readonly`.
    pub params: Vec<(&'a str, Type, bool, bool, bool)>,
}

/// `noalias`, `readonly` and `inreg` of a parameter's attribute map: a key
/// holds when every occurrence is `b:true`. A map that does not parse states
/// none; the other attributes are ignored.
fn param_facts(attrs: &str) -> (bool, bool, bool) {
    let mut scanner = Scanner::new(attrs);
    let Ok(map) = parse_attr_map(&mut scanner, MAX_ATTR_NESTING) else {
        return (false, false, false);
    };
    let holds = |key: &str| {
        let mut values = map.iter().filter(|(k, _)| k == key).map(|(_, v)| v).peekable();
        values.peek().is_some() && values.all(|v| *v == Attr::Bool(true))
    };
    (holds("noalias"), holds("readonly"), holds("inreg"))
}

/// Parses the header, the kernel line, its attributes and the parameters.
pub(super) fn prologue<'a>(lines: &mut Lines<'a>) -> Result<Prologue<'a>, CompileError> {
    let header = lines.next()?;
    let mut words = header.split_whitespace();
    if !(words.next() == Some("ckdsl.ir") && words.next() == Some("v1") && words.next().is_none()) {
        return Err(err(format!("bad header {header:?}; expected 'ckdsl.ir v1'")));
    }
    let kernel_line = lines.next()?;
    let name = kernel_line
        .strip_prefix("kernel @")
        .and_then(|rest| rest.strip_suffix('{'))
        .ok_or_else(|| err(format!("bad kernel line {kernel_line:?}")))?
        .trim()
        .to_string();
    // rocKE names kernels freely (LLVM quotes the symbol, e.g. a `-1` knob
    // value); the ELF writer emits `[A-Za-z0-9_.$-]` symbols of at most 1024
    // bytes only (`hsaco_writer::write_kernel_object`). Refuse the others here,
    // before any work, instead of failing in the writer.
    if name.is_empty() {
        return Err(err(format!("bad kernel line {kernel_line:?}")));
    }
    if name.len() > 1024 || !name.bytes().all(|b| b.is_ascii_alphanumeric() || matches!(b, b'_' | b'.' | b'$' | b'-')) {
        return Err(unsupported(format!("kernel symbol {name:?} (the ELF writer emits [A-Za-z0-9_.$-] symbols)")));
    }

    let mut max_workgroup_size = 1024u32;
    let mut waves_per_eu = (0u32, 0u32);
    let mut agpr_limit = None;
    if let Some(attrs_line) = lines.peek().filter(|line| line.starts_with("attrs ")) {
        lines.next()?;
        let mut scanner = Scanner::new(&attrs_line["attrs ".len()..]);
        for (key, value) in parse_attr_map(&mut scanner, MAX_ATTR_NESTING)? {
            match (&*key, value) {
                ("max_workgroup_size", Attr::Int(size)) => max_workgroup_size = u32::try_from(size).unwrap_or(0),
                // Selects LLVM's machine scheduler only; it carries no semantics.
                ("scheduler_strategy", Attr::Str(_)) => {}
                // COMGR compile policy (`rocke.helpers.compile.COMGR_FRESH_PROCESS_ATTR`): LLVM's
                // `iglp_opt(1)` code depends on earlier compiles in the process. This compiler's
                // output depends on the IR text only, so it carries nothing here.
                ("comgr_fresh_process", Attr::Bool(_)) => {}
                // Tells the LLVM lowering to scope LDS accesses per allocation, so that a read waits only
                // for the LDS DMA into its own allocation; the gfx950 waitcnt pass's LDS tags always do.
                ("lds_alias_scopes", Attr::Bool(_)) => {}
                // An occupancy hint, `n` or `[min, max]` waves per SIMD.
                ("waves_per_eu", Attr::Int(n)) if (0..=16).contains(&n) => waves_per_eu = (n as u32, n as u32),
                ("waves_per_eu", Attr::List(range)) => match range[..] {
                    [Attr::Int(lo), Attr::Int(hi)] if (0..=16).contains(&lo) && (0..=16).contains(&hi) => {
                        waves_per_eu = (lo as u32, hi as u32)
                    }
                    _ => return Err(err(format!("malformed kernel attribute waves_per_eu {range:?}"))),
                },
                // `[min, max]` AGPRs (LLVM's `amdgpu-agpr-alloc`); `[0, 0]`: none.
                ("agpr_alloc", Attr::List(range)) => match range[..] {
                    [Attr::Int(0), Attr::Int(0)] => agpr_limit = Some(0),
                    _ => {
                        return Err(unsupported(format!(
                            "kernel attribute agpr_alloc {range:?} is not supported by the direct backend"
                        )))
                    }
                },
                (other, _) => {
                    return Err(unsupported(format!(
                        "kernel attribute {other:?} is not supported by the direct backend"
                    )))
                }
            }
        }
    }

    if lines.next()? != "params {" {
        return Err(err("expected 'params {'"));
    }
    let mut params = Vec::new();
    loop {
        let line = lines.next()?;
        if line == "}" {
            break;
        }
        let (head, (noalias, readonly, inreg)) = match line.find('{') {
            Some(brace) => (&line[..brace], param_facts(&line[brace..])),
            None => (line, (false, false, false)),
        };
        let colon = find_spaced(head, ':').ok_or_else(|| err(format!("bad param line {line:?}")))?;
        let pname = head[..colon].trim().trim_start_matches('%');
        let ty = parse_type(&head[colon + 3..])?;
        if !matches!(ty, Type::Scalar(Scalar::I32 | Scalar::I64 | Scalar::F32) | Type::Ptr(_)) {
            return Err(unsupported(format!("unsupported kernel argument type for %{pname}")));
        }
        // Kernarg preload into user SGPRs (LLVM's `inreg` on gfx950):
        // `isel` preloads the leading `inreg` arguments that fit
        // (`opt::kernarg_preload`); gfx1151 refuses it there.
        params.push((pname, ty, noalias, readonly, inreg));
    }
    Ok(Prologue { name, max_workgroup_size, waves_per_eu, agpr_limit, params })
}

/// One definition of a name: the value, the scope it was made in, the
/// definition it shadows (index + 1, 0 for none), the name's hash, and its
/// byte range in the name storage.
#[derive(Clone, Copy)]
struct Binding {
    value: ValueId,
    scope: u32,
    shadows: u32,
    hash: u32,
    name: (u32, u32),
}

pub(super) fn name_hash(name: &str) -> u32 {
    let mut h = super::fx::FxHasher::default();
    h.write(name.as_bytes());
    h.finish() as u32
}

/// Values and the scoped table of their names. A name resolves to its latest
/// definition whose scope is still open, which is the definition in the
/// innermost open scope that has one: a scope receives definitions only while
/// it is innermost. Closing a scope only marks it closed. Scopes are regions
/// and `scf.for` ops (block arguments).
struct Names {
    values: Vec<ValueDef>,
    names: String,
    bindings: Vec<Binding>,
    /// Open-addressing table of the latest binding (index + 1) of each name;
    /// at most half full.
    slots: Vec<u32>,
    names_used: usize,
    open: Vec<bool>,
    scopes: Vec<u32>,
}

impl Names {
    fn new(text_len: usize) -> Self {
        let expected_values = text_len / 64;
        Names {
            values: Vec::with_capacity(expected_values),
            names: String::with_capacity(text_len / 6),
            bindings: Vec::with_capacity(expected_values),
            slots: vec![0; (text_len / 32).next_power_of_two().max(16)],
            names_used: 0,
            open: Vec::new(),
            scopes: Vec::new(),
        }
    }

    /// The slot holding `name`, or the empty slot where it would go.
    fn slot(&self, name: &str, hash: u32) -> Result<usize, usize> {
        let mask = self.slots.len() - 1;
        let mut i = hash as usize & mask;
        loop {
            let b = self.slots[i];
            if b == 0 {
                return Err(i);
            }
            let binding = &self.bindings[b as usize - 1];
            if binding.hash == hash && &self.names[binding.name.0 as usize..binding.name.1 as usize] == name {
                return Ok(i);
            }
            i = (i + 1) & mask;
        }
    }

    fn grow_slots(&mut self) {
        let mut slots = vec![0u32; self.slots.len() * 2];
        let mask = slots.len() - 1;
        for &b in self.slots.iter().filter(|&&b| b != 0) {
            let mut i = self.bindings[b as usize - 1].hash as usize & mask;
            while slots[i] != 0 {
                i = (i + 1) & mask;
            }
            slots[i] = b;
        }
        self.slots = slots;
    }

    fn enter(&mut self) {
        self.scopes.push(self.open.len() as u32);
        self.open.push(true);
    }

    fn leave(&mut self) {
        let scope = self.scopes.pop().expect("scope stack is never empty");
        self.open[scope as usize] = false;
    }

    /// The first definition in the chain starting at `b` whose scope is open.
    fn visible(&self, mut b: Binding) -> Option<Binding> {
        loop {
            if self.open[b.scope as usize] {
                return Some(b);
            }
            if b.shadows == 0 {
                return None;
            }
            b = self.bindings[b.shadows as usize - 1];
        }
    }

    fn define(&mut self, name: &str, ty: Type) -> ValueId {
        let value = ValueId(self.values.len() as u32);
        let hash = name_hash(name);
        let slot = self.slot(name, hash);
        let start = self.names.len() as u32;
        self.values.push(ValueDef::new(&mut self.names, name, ty));
        self.bind(value, hash, slot, start);
        value
    }

    /// Binds `name` to the existing `value`, as `define` binds a new one.
    pub(super) fn alias(&mut self, name: &str, value: ValueId) {
        let hash = name_hash(name);
        let slot = self.slot(name, hash);
        let start = self.names.len() as u32;
        self.names.push_str(name);
        self.bind(value, hash, slot, start);
    }

    /// The innermost open scope.
    pub(super) fn scope(&self) -> u32 {
        *self.scopes.last().expect("scope stack is never empty")
    }

    pub(super) fn is_open(&self, scope: u32) -> bool {
        self.open[scope as usize]
    }

    /// Binds the name just appended to `names` at `start` (whose `slot`
    /// lookup was made before) to `value` in the innermost scope.
    fn bind(&mut self, value: ValueId, hash: u32, slot: Result<usize, usize>, start: u32) {
        let scope = *self.scopes.last().expect("scope stack is never empty");
        let mut binding = Binding { value, scope, shadows: 0, hash, name: (start, self.names.len() as u32) };
        let index = self.bindings.len() as u32 + 1;
        match slot {
            Ok(i) => {
                let mut at = self.slots[i];
                let mut prev = self.bindings[at as usize - 1];
                while !self.open[prev.scope as usize] && prev.shadows != 0 {
                    at = prev.shadows;
                    prev = self.bindings[at as usize - 1];
                }
                if self.open[prev.scope as usize] {
                    binding.shadows = at;
                }
                self.slots[i] = index;
                self.bindings.push(binding);
            }
            Err(i) => {
                self.slots[i] = index;
                self.bindings.push(binding);
                self.names_used += 1;
                if 2 * self.names_used > self.slots.len() {
                    self.grow_slots();
                }
            }
        }
    }

    fn lookup(&self, name: &str, opname: &str) -> Result<ValueId, CompileError> {
        self.slot(name, name_hash(name))
            .ok()
            .and_then(|i| self.visible(self.bindings[self.slots[i] as usize - 1]))
            .map(|b| b.value)
            .ok_or_else(|| err(format!("operand {name:?} used before definition in {opname}")))
    }
}

/// An op whose regions are being parsed: its index in `Parser::ops`, and
/// where its pending results start in `Parser::pending`.
struct OpenOp<'a> {
    index: usize,
    opname: &'a str,
    pending: usize,
    results: usize,
}

/// The sequential parser. Regions nest through explicit stacks, not
/// recursion, so nesting depth is bounded by memory, not by the thread stack.
struct Parser<'a> {
    names: Names,
    /// Result names and types of the ops being parsed, innermost last.
    pending: Vec<(&'a str, Type)>,
    /// The ops of the regions being filled, each region's from its start in
    /// `starts` to the next one's, innermost last; `open[i]` owns the region
    /// from `starts[i + 1]`. A closed region takes its ops out.
    ops: Vec<Op>,
    starts: Vec<usize>,
    open: Vec<OpenOp<'a>>,
    consts: ConstTable<'a>,
    /// Repeated constants are bound to the earlier one (see `hashcons`).
    hashcons: bool,
    /// Vector level of the line scans (`scan`).
    level: Level,
}

/// Whether a constant line followed by the line `next` may be bound to an
/// earlier constant: not when a region follows (the op would own it) nor when
/// a `}` does (the op ends its region, which validation looks at).
pub(super) fn shareable_before(next: Option<&str>) -> bool {
    !next.is_some_and(|next| next == "}" || next.starts_with("region @"))
}

impl<'a> Parser<'a> {
    fn open_region(&mut self, lines: &mut Lines<'a>) -> Result<(), CompileError> {
        let header = lines.next()?;
        if !(header.starts_with("region @") && header.ends_with('{')) {
            return Err(err(format!("expected 'region @<label> {{', got {header:?}")));
        }
        self.names.enter();
        self.starts.push(self.ops.len());
        Ok(())
    }

    /// The region at the bottom of the stack: the kernel body.
    fn body(&mut self, lines: &mut Lines<'a>) -> Result<Region, CompileError> {
        self.open_region(lines)?;
        loop {
            match lines.peek() {
                None => return Err(err("unterminated region")),
                Some("}") => {
                    lines.next()?;
                    self.names.leave();
                    let start = self.starts.pop().expect("an open region");
                    let Some(owner) = self.open.last() else {
                        return Ok(Region { ops: std::mem::take(&mut self.ops) });
                    };
                    let region = Region { ops: self.ops.drain(start..).collect() };
                    self.ops[owner.index].regions.push(region);
                    if lines.region_follows() {
                        self.open_region(lines)?;
                    } else {
                        let owner = self.open.pop().expect("an open op");
                        self.finish(owner)?;
                    }
                }
                Some(_) => {
                    let line = lines.next()?;
                    let masks = super::scan_line::masks(self.level, line);
                    let constant = if self.hashcons { const_key_with(line, masks.as_ref()) } else { None };
                    if let Some((name, key)) = constant.filter(|_| shareable_before(lines.peek())) {
                        self.constant(line, masks.as_ref(), name, key)?;
                        continue;
                    }
                    let op = self.op(line, masks.as_ref())?;
                    if lines.region_follows() {
                        self.open.push(op);
                        self.open_region(lines)?;
                    } else {
                        self.finish(op)?;
                    }
                }
            }
        }
    }

    /// A constant line without regions: bound to the visible constant with the
    /// same key (see `hashcons`), or parsed and entered as one.
    fn constant(&mut self, line: &'a str, masks: Option<&Masks>, name: &'a str, key: &'a str) -> Result<(), CompileError> {
        let hash = key_hash(key);
        let names = &self.names;
        if let Some(value) = self.consts.lookup(key, hash, |scope| names.is_open(scope)) {
            self.names.alias(name, value);
            return Ok(());
        }
        let op = self.op(line, masks)?;
        self.finish(op)?;
        let value = self.ops.last().map(|op| op.results[0]).expect("the constant");
        self.consts.insert(key, hash, value, self.names.scope());
        Ok(())
    }

    /// Parses an op line into the innermost region. The loop body sees the
    /// induction variable and iteration arguments; the results become visible
    /// after the op (`finish`).
    fn op(&mut self, line: &'a str, masks: Option<&Masks>) -> Result<OpenOp<'a>, CompileError> {
        let pending = self.pending.len();
        let mut operands = Operands::new();
        let names = &self.names;
        let lookup = |name, opname| names.lookup(name, opname);
        let scanned = scan_op(line, masks, &mut operands, &mut self.pending, lookup, MAX_ATTR_NESTING)?;
        let results = self.pending.len() - pending;
        let mut op = Op {
            kind: scanned.kind,
            results: Results::new(),
            operands,
            attrs: scanned.attrs,
            regions: Vec::new(),
            block_args: Vec::new(),
        };
        if op.kind == OpKind::For {
            self.names.enter();
            let (iv_name, iter_args) = for_args(&op, results)?;
            op.block_args.push(self.names.define(&iv_name, Type::Scalar(Scalar::I32)));
            for (name, ty) in iter_args {
                op.block_args.push(self.names.define(&name, ty));
            }
        }
        self.ops.push(op);
        Ok(OpenOp { index: self.ops.len() - 1, opname: scanned.opname, pending, results })
    }

    /// Ends the op: its scope closes and its results are defined.
    fn finish(&mut self, open: OpenOp<'a>) -> Result<(), CompileError> {
        let op = &mut self.ops[open.index];
        if op.kind == OpKind::For {
            self.names.leave();
        }
        for i in open.pending..open.pending + open.results {
            let (name, ty) = std::mem::replace(&mut self.pending[i], ("", Type::Scalar(Scalar::I1)));
            op.results.push(self.names.define(name, ty));
        }
        self.pending.truncate(open.pending);
        let expected = expected_regions(op.kind);
        if op.regions.len() != expected {
            return Err(err(format!(
                "{} expects {expected} region(s), found {}",
                open.opname,
                op.regions.len()
            )));
        }
        Ok(())
    }
}

fn parse_sequential(text: &str, hashcons: bool) -> Result<Kernel, CompileError> {
    let level = Level::detect();
    let mut lines = Lines::new(text, 0, text.len()).scanning(level);
    let prologue = prologue(&mut lines)?;
    let mut parser = Parser {
        names: Names::new(text.len()),
        pending: Vec::new(),
        // About one op per 48 bytes, at most, when lines are short.
        ops: Vec::with_capacity(text.len() / 48),
        starts: Vec::new(),
        open: Vec::new(),
        consts: ConstTable::new(),
        hashcons,
        level,
    };
    parser.names.enter();
    let mut params = Vec::with_capacity(prologue.params.len());
    for (pname, ty, noalias, readonly, inreg) in prologue.params {
        let value = parser.names.define(&format!("%{pname}"), ty.clone());
        params.push(Param { value, name: pname.to_string(), ty, noalias, readonly, inreg });
    }
    let body = parser.body(&mut lines)?;
    if lines.next()? != "}" {
        return Err(err("expected closing kernel '}'"));
    }
    if lines.peek().is_some() {
        return Err(err("trailing text after kernel"));
    }
    Ok(Kernel {
        name: prologue.name,
        max_workgroup_size: prologue.max_workgroup_size,
        waves_per_eu: prologue.waves_per_eu,
        wave: 32,
        vgpr_limit: None,
        agpr_limit: prologue.agpr_limit,
        params,
        body,
        values: parser.names.values,
        names: parser.names.names,
    })
}

/// Threads to parse `len` bytes with: `ROCKE_DIRECT_PARSE_THREADS` if set,
/// else one per `BYTES_PER_THREAD` above `PARALLEL_MIN_BYTES`, at most the
/// machine's parallelism and `MAX_THREADS`.
fn threads_for(len: usize) -> usize {
    if let Some(n) = std::env::var_os("ROCKE_DIRECT_PARSE_THREADS")
        .and_then(|v| v.to_str().and_then(|v| v.trim().parse::<usize>().ok()))
        .filter(|&n| n > 0)
    {
        return n;
    }
    if len < PARALLEL_MIN_BYTES {
        return 1;
    }
    let cap = std::thread::available_parallelism().map_or(1, |n| n.get()).min(MAX_THREADS);
    (len / BYTES_PER_THREAD).clamp(2, cap.max(2)).min(cap)
}

pub fn parse(text: &str) -> Result<Kernel, CompileError> {
    parse_with_threads(text, threads_for(text.len()))
}

/// `parse` on `threads` threads (1 is the sequential parser). The result is
/// the same for every count.
#[doc(hidden)]
pub fn parse_with_threads(text: &str, threads: usize) -> Result<Kernel, CompileError> {
    parse_chunked(text, threads, 0)
}

/// `parse_with_threads` with chunks of about `chunk_bytes` (0 picks a size),
/// so tests can put chunk boundaries anywhere.
#[doc(hidden)]
pub fn parse_chunked(text: &str, threads: usize, chunk_bytes: usize) -> Result<Kernel, CompileError> {
    parse_options(text, threads, chunk_bytes, !super::debug_flag("nohashcons"))
}

/// `parse_chunked` without hash-consing constants (`hashcons`): the kernel as
/// written, which the optimizer turns into the same code.
#[doc(hidden)]
pub fn parse_plain(text: &str, threads: usize, chunk_bytes: usize) -> Result<Kernel, CompileError> {
    parse_options(text, threads, chunk_bytes, false)
}

pub(super) fn parse_options(text: &str, threads: usize, chunk_bytes: usize, hashcons: bool) -> Result<Kernel, CompileError> {
    if threads > 1 {
        if let Some(kernel) = super::parse_par::parse(text, threads, chunk_bytes, hashcons) {
            return Ok(kernel);
        }
    }
    parse_sequential(text, hashcons)
}

/// The parallel path alone: `None` where `parse_chunked` falls back to the
/// sequential parser.
#[doc(hidden)]
pub fn parse_parallel(text: &str, threads: usize, chunk_bytes: usize) -> Option<Kernel> {
    super::parse_par::parse(text, threads.max(1), chunk_bytes, true)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spaced_search_matches_str_find() {
        let cases = ["a ( b", " ( ", "((( ( (", "x = y = z", " = ", "a=b", "", " ", "  (  ", "%a : i32 = op ( %b )", "é ( é = é", "(é)"];
        for text in cases {
            for c in ['(', '=', ':'] {
                let pat = format!(" {c} ");
                assert_eq!(find_spaced(text, c), text.find(&pat), "{text:?} {c}");
                assert_eq!(rfind_spaced(text, c), text.rfind(&pat), "{text:?} {c}");
            }
        }
    }

    #[test]
    fn trim_matches_str_trim() {
        for text in ["", " ", "a", " a b ", "\t\x0b\x0c\r\n a \r", "\u{a0}x\u{a0}", " \u{2003} y \u{3000} ", "é", " é "] {
            assert_eq!(trim(text), text.trim(), "{text:?}");
        }
    }

    #[test]
    fn split_top_ignores_nested_commas() {
        let parts: Vec<&str> = split_top("%a : vec<f16,4>, %b : i32,, %c : smem<f32, [4x4]>").collect();
        assert_eq!(parts, ["%a : vec<f16,4>", " %b : i32", "", " %c : smem<f32, [4x4]>"]);
        assert_eq!(split_top("").collect::<Vec<_>>(), [""]);
    }

    #[test]
    fn quick_type_agrees_with_parse_type() {
        let cases = [
            "i1", "i32", "i64", "f16", "f32", "i8", "", "vec<f16x8>", "vec<f32x1>", "vec<i32x4>", "vec<f16x>",
            "vec<f16x08>", "vec<f16x+8>", "vec<x8>", "vec<f16x4294967296>", "vec<f16x999999999>", "vec<bf16x8>",
            "vec<f16xx8>", "vec<f16x8", "ptr<f16,global>", "ptr<f32,global>", "ptr<f16,local>", "ptr<f16, global>",
            "ptr<,global>", "smem<f16,[4]>", "vec<f16x8>>",
        ];
        for text in cases {
            if let Some(ty) = quick_type(text.as_bytes()) {
                assert_eq!(parse_type(text).ok(), Some(ty), "{text:?}");
            }
        }
        assert!(quick_type(b"vec<f16x8>").is_some() && quick_type(b"ptr<f16,global>").is_some());
    }

    /// Lines the canonical scan accepts give what the general scan gives.
    #[test]
    fn canonical_scan_matches_general_scan() {
        let lines = [
            "%a : i32 = arith.add ( %b, %c )",
            "%a : i32 = arith.add ( %b, %c ) { wrap = s:\"x\" }",
            "%a : i32 = arith.constant (  ) { ity = s:\"i32\", value = i:3 }",
            "%a.b_1 : vec<f16x8> = tile.smem_load_vN ( %s, %i ) { elem_type = s:\"f16\", rank = i:1, vec = i:8 }",
            "%p : ptr<f16,global> = tile.global_ptr_add ( %K, %o )",
            "tile.sync (  ) { fence = s:\"compiler\" }",
            "scf.yield ( %x ) @loc(\"a.py\":3)",
            "cf.return (  )",
            "%a : i32 = arith.add ( %b, %c ) )",
            "%a : i32 = arith.add ( %b, %c ) junk",
            "%a : i32 = nope.op ( %b )",
            "%a : i32 = arith.add ( %b,%c )",
            "%a : i32 = arith.add ( %b , %c )",
            "%a : i33 = arith.add ( %b )",
            "%a : smem<f16, [4]> = tile.smem_alloc (  )",
            "%a : i32, %b : i32 = scf.for ( %c )",
            "% : i32 = arith.add ( %b )",
            "%a : ( = arith.add ( %b )",
            "x = arith.add ( %b )",
            "arith.add ( %b ) { a = i:1 } extra",
            "%a : i32 = arith.add ( %b, %c ) { a = i:1 ",
        ];
        let mut fast = 0;
        for line in lines {
            if canonical(line).is_none() {
                continue;
            }
            fast += 1;
            // A doubled space after `(` keeps the meaning but not the shape.
            let slow_line = line.replacen(" ( ", " (  ", 1);
            assert!(canonical(&slow_line).is_none(), "{slow_line:?}");
            assert_eq!(scan_owned(line), scan_owned(&slow_line), "{line:?}");
        }
        assert!(fast >= 10, "{fast}");
    }

    /// Wherever the flat-map fast path answers, it answers as the general parser.
    #[test]
    fn flat_attrs_match_parse_attr_map() {
        let cases = [
            "{ ity = s:\"i32\", value = i:16 }",
            "{ ity = s:\"i32\", value = i:-7 } @loc(x)",
            "{ value = i:+7 }",
            "{ value = i:99999999999999999999 }",
            "{ value = i:1x }",
            "{ fill = f:0.0, vec = i:4, elem = s:\"f32\" }",
            "{ fill = f:nan, x = f:-inf, y = f:1e400, z = f:abc }",
            "{ fill = f:nan, x = f:-inf, y = f:1e400 }",
            "{ pred = s:\"lt\" }",
            "{ convergent = b:false, sideeffect = b:true, template = s:\"\" }",
            "{ a = b:truex }",
            "{ a = b:maybe }",
            "{ s = s:\"a\\\"b\" }",
            "{ s = s:\"é ü\", t = s:\"=v,0\" }",
            "{ a = i:1,, b = i:2 }",
            "{ a = i:1 , b = i:2 }",
            "{ a = i:1}",
            "{ }",
            "{}",
            "{ a.b_c = i:1 }",
            "{ a = l:[ i:1 ] }",
            "{ a = { b = i:1 } }",
            "{ a = i:1, b = i:2, c = i:3, d = i:4, e = i:5, f = i:6, g = i:7, h = i:8 }",
            "{ a = i:1, b = i:2, c = i:3, d = i:4, e = i:5, f = i:6, g = i:7, h = i:8, i = i:9 }",
            "{ a = s:\"unterminated }",
            "{ a = i:1 } junk",
        ];
        let mut fast = 0;
        for text in cases {
            let Some((attrs, used)) = flat_attrs(text) else { continue };
            fast += 1;
            let mut scanner = Scanner::new(text);
            let general = parse_attr_map(&mut scanner, u32::MAX).expect("the general parser accepts it too");
            assert_eq!(format!("{attrs:?}"), format!("{general:?}"), "{text:?}");
            assert_eq!(used, scanner.pos, "{text:?}");
            assert_eq!(attrs.capacity(), attrs.len());
        }
        assert!(fast >= 10, "{fast}");
    }

    type Scanned = (Result<(OpKind, String, String), String>, Vec<ValueId>, Vec<(String, Type)>, Vec<String>);

    /// What `scan_op` makes of `line`, with operands numbered in order.
    fn scan_owned(line: &str) -> Scanned {
        let mut names = Vec::new();
        let mut operands = Operands::new();
        let mut results = Vec::new();
        let scanned = scan_op(
            line,
            None,
            &mut operands,
            &mut results,
            |name, _| {
                names.push(name.to_string());
                Ok(ValueId(names.len() as u32))
            },
            u32::MAX,
        );
        let scanned = scanned.map(|s| (s.kind, s.opname.to_string(), format!("{:?}", s.attrs))).map_err(|e| e.message);
        let results = results.into_iter().map(|(name, ty)| (name.to_string(), ty)).collect();
        (scanned, operands.to_vec(), results, names)
    }

    #[test]
    fn scopes_shadow_and_restore() {
        let text = "ckdsl.ir v1\nkernel @k {\nparams {\n%n : i32\n}\nregion @entry {\n\
            %x : i32 = arith.add ( %n, %n )\n\
            %r : i32 = scf.for ( %n, %n, %n, %x ) { iv = s:\"%i\", iter_args = l:[ { name = s:\"%x\", type = s:\"i32\" } ] }\n\
            region @body {\n%x : i32 = arith.add ( %x, %i )\nscf.yield ( %x )\n}\n\
            %y : i32 = arith.add ( %x, %r )\ncf.return (  )\n}\n}\n";
        let kernel = parse(text).expect("parses");
        let ops = &kernel.body.ops;
        let body = &ops[1].regions[0].ops;
        assert_eq!(body[0].operands, [ops[1].block_args[1], ops[1].block_args[0]]);
        assert_eq!(body[1].operands, body[0].results);
        assert_eq!(ops[2].operands, [ops[0].results[0], ops[1].results[0]]);
        assert_eq!(kernel.name_of(ops[2].results[0]), "%y");
        assert!(parse(&text.replace("( %x, %r )", "( %i, %r )")).is_err());
    }

    #[test]
    fn name_table_grows() {
        let mut text = String::from("ckdsl.ir v1\nkernel @k {\nparams {\n%v0 : i32\n}\nregion @entry {\n");
        for i in 1..500 {
            text.push_str(&format!("%v{i} : i32 = arith.add ( %v{}, %v{} )\n", i - 1, i / 2));
        }
        text.push_str("cf.return (  )\n}\n}\n");
        let kernel = parse(&text).expect("parses");
        for (i, op) in kernel.body.ops.iter().take(499).enumerate() {
            let i = i + 1;
            assert_eq!(kernel.name_of(op.results[0]), format!("%v{i}"));
            assert_eq!(kernel.name_of(op.operands[0]), format!("%v{}", i - 1));
            assert_eq!(kernel.name_of(op.operands[1]), format!("%v{}", i / 2));
        }
    }

    /// Nesting far deeper than a recursive parser's stack allows.
    #[test]
    fn deep_nesting_needs_no_stack() {
        let depth = 100_000;
        let mut text = String::from(
            "ckdsl.ir v1\nkernel @k {\nparams {\n%n : i32\n}\nregion @entry {\n\
             %c : i1 = arith.cmp ( %n, %n ) { pred = s:\"eq\" }\n",
        );
        for _ in 0..depth {
            text.push_str("scf.if ( %c )\nregion @then {\n");
        }
        text.push_str("cf.return (  )\n");
        for _ in 0..depth {
            text.push_str("}\n");
        }
        text.push_str("cf.return (  )\n}\n}\n");
        let kernel = parse_with_threads(&text, 1).expect("parses");
        let mut region = &kernel.body;
        let mut levels = 0;
        while let Some(op) = region.ops.iter().find(|op| op.kind == OpKind::If) {
            region = &op.regions[0];
            levels += 1;
        }
        assert_eq!(levels, depth);
        // Kernel's derived drop recurses; take it apart level by level.
        let mut regions = vec![kernel.body];
        while let Some(mut region) = regions.pop() {
            for op in &mut region.ops {
                regions.append(&mut op.regions);
            }
        }
    }

    /// The sequential parser before its regions shared one op stack: the
    /// reference it is checked against.
    struct ParserReference<'a> {
        names: Names,
        /// Result names and types of the ops being parsed, innermost last.
        pending: Vec<(&'a str, Type)>,
        /// Regions being filled, innermost last; `open[i]` owns `regions[i + 1]`.
        regions: Vec<Region>,
        open: Vec<OpenOp<'a>>,
        consts: ConstTable<'a>,
        /// Repeated constants are bound to the earlier one (see `hashcons`).
        hashcons: bool,
        /// Vector level of the line scans (`scan`).
        level: Level,
    }

    impl<'a> ParserReference<'a> {
        fn open_region(&mut self, lines: &mut Lines<'a>) -> Result<(), CompileError> {
            let header = lines.next()?;
            if !(header.starts_with("region @") && header.ends_with('{')) {
                return Err(err(format!("expected 'region @<label> {{', got {header:?}")));
            }
            self.names.enter();
            self.regions.push(Region::default());
            Ok(())
        }

        /// The region at the bottom of the stack: the kernel body.
        fn body(&mut self, lines: &mut Lines<'a>) -> Result<Region, CompileError> {
            self.open_region(lines)?;
            loop {
                match lines.peek() {
                    None => return Err(err("unterminated region")),
                    Some("}") => {
                        lines.next()?;
                        self.names.leave();
                        let region = self.regions.pop().expect("an open region");
                        let Some(owner) = self.open.last() else {
                            return Ok(region);
                        };
                        let parent = self.regions.last_mut().expect("the owner's region");
                        parent.ops[owner.index].regions.push(region);
                        if lines.region_follows() {
                            self.open_region(lines)?;
                        } else {
                            let owner = self.open.pop().expect("an open op");
                            self.finish(owner)?;
                        }
                    }
                    Some(_) => {
                        let line = lines.next()?;
                        let masks = super::super::scan_line::masks(self.level, line);
                        let constant = if self.hashcons { const_key_with(line, masks.as_ref()) } else { None };
                        if let Some((name, key)) = constant.filter(|_| shareable_before(lines.peek())) {
                            self.constant(line, masks.as_ref(), name, key)?;
                            continue;
                        }
                        let op = self.op(line, masks.as_ref())?;
                        if lines.region_follows() {
                            self.open.push(op);
                            self.open_region(lines)?;
                        } else {
                            self.finish(op)?;
                        }
                    }
                }
            }
        }

        /// A constant line without regions: bound to the visible constant with the
        /// same key (see `hashcons`), or parsed and entered as one.
        fn constant(&mut self, line: &'a str, masks: Option<&Masks>, name: &'a str, key: &'a str) -> Result<(), CompileError> {
            let hash = key_hash(key);
            let names = &self.names;
            if let Some(value) = self.consts.lookup(key, hash, |scope| names.is_open(scope)) {
                self.names.alias(name, value);
                return Ok(());
            }
            let op = self.op(line, masks)?;
            self.finish(op)?;
            let value = self.regions.last().and_then(|r| r.ops.last()).map(|op| op.results[0]).expect("the constant");
            self.consts.insert(key, hash, value, self.names.scope());
            Ok(())
        }

        /// Parses an op line into the innermost region. The loop body sees the
        /// induction variable and iteration arguments; the results become visible
        /// after the op (`finish`).
        fn op(&mut self, line: &'a str, masks: Option<&Masks>) -> Result<OpenOp<'a>, CompileError> {
            let pending = self.pending.len();
            let mut operands = Operands::new();
            let names = &self.names;
            let lookup = |name, opname| names.lookup(name, opname);
            let scanned = scan_op(line, masks, &mut operands, &mut self.pending, lookup, MAX_ATTR_NESTING)?;
            let results = self.pending.len() - pending;
            let mut op = Op {
                kind: scanned.kind,
                results: Results::new(),
                operands,
                attrs: scanned.attrs,
                regions: Vec::new(),
                block_args: Vec::new(),
            };
            if op.kind == OpKind::For {
                self.names.enter();
                let (iv_name, iter_args) = for_args(&op, results)?;
                op.block_args.push(self.names.define(&iv_name, Type::Scalar(Scalar::I32)));
                for (name, ty) in iter_args {
                    op.block_args.push(self.names.define(&name, ty));
                }
            }
            let region = self.regions.last_mut().expect("an open region");
            region.ops.push(op);
            Ok(OpenOp { index: region.ops.len() - 1, opname: scanned.opname, pending, results })
        }

        /// Ends the op: its scope closes and its results are defined.
        fn finish(&mut self, open: OpenOp<'a>) -> Result<(), CompileError> {
            let op = &mut self.regions.last_mut().expect("an open region").ops[open.index];
            if op.kind == OpKind::For {
                self.names.leave();
            }
            for i in open.pending..open.pending + open.results {
                let (name, ty) = std::mem::replace(&mut self.pending[i], ("", Type::Scalar(Scalar::I1)));
                op.results.push(self.names.define(name, ty));
            }
            self.pending.truncate(open.pending);
            let expected = expected_regions(op.kind);
            if op.regions.len() != expected {
                return Err(err(format!(
                    "{} expects {expected} region(s), found {}",
                    open.opname,
                    op.regions.len()
                )));
            }
            Ok(())
        }
    }

    fn parse_sequential_reference(text: &str, hashcons: bool) -> Result<Kernel, CompileError> {
        let level = Level::detect();
        let mut lines = Lines::new(text, 0, text.len()).scanning(level);
        let prologue = prologue(&mut lines)?;
        let mut parser = ParserReference {
            names: Names::new(text.len()),
            pending: Vec::new(),
            regions: Vec::new(),
            open: Vec::new(),
            consts: ConstTable::new(),
            hashcons,
            level,
        };
        parser.names.enter();
        let mut params = Vec::with_capacity(prologue.params.len());
        for (pname, ty, noalias, readonly, inreg) in prologue.params {
            let value = parser.names.define(&format!("%{pname}"), ty.clone());
            params.push(Param { value, name: pname.to_string(), ty, noalias, readonly, inreg });
        }
        let body = parser.body(&mut lines)?;
        if lines.next()? != "}" {
            return Err(err("expected closing kernel '}'"));
        }
        if lines.peek().is_some() {
            return Err(err("trailing text after kernel"));
        }
        Ok(Kernel {
            name: prologue.name,
            max_workgroup_size: prologue.max_workgroup_size,
            waves_per_eu: prologue.waves_per_eu,
            wave: 32,
            vgpr_limit: None,
            agpr_limit: prologue.agpr_limit,
            params,
            body,
            values: parser.names.values,
            names: parser.names.names,
        })
    }

    /// Every `.ir` file under `ROCKE_PARSER_CHECK_DIR`, when it is set (the
    /// corpus, a census sample).
    fn parser_check_texts() -> Vec<String> {
        let Some(dir) = std::env::var_os("ROCKE_PARSER_CHECK_DIR") else {
            return Vec::new();
        };
        let mut paths: Vec<_> = std::fs::read_dir(dir).expect("reads the directory").filter_map(|e| e.ok().map(|e| e.path())).collect();
        paths.retain(|p| p.extension().is_some_and(|x| x == "ir"));
        paths.sort();
        paths.iter().map(|p| std::fs::read_to_string(p).expect("reads the file")).collect()
    }

    #[test]
    fn sequential_parser_builds_what_its_reference_built() {
        let nested = super::super::parse_par::tests::kernel(4);
        let lines: Vec<&str> = nested.lines().collect();
        let mut texts = vec![super::super::parse_par::tests::kernel(1), include_str!("../gfx1151_wmma_gemm.ir").to_string()];
        for i in 0..lines.len() {
            let mut deleted = lines.clone();
            deleted.remove(i);
            texts.push(deleted.join("\n"));
            let mut duplicated = lines.clone();
            duplicated.insert(i, lines[i]);
            texts.push(duplicated.join("\n"));
            if i + 1 < lines.len() {
                let mut swapped = lines.clone();
                swapped.swap(i, i + 1);
                texts.push(swapped.join("\n"));
            }
        }
        texts.push(nested);
        let external = parser_check_texts();
        let (mut accepted, mut rejected) = (0, 0);
        for (k, text) in texts.iter().chain(&external).enumerate() {
            for hashcons in [true, false] {
                match (parse_sequential(text, hashcons), parse_sequential_reference(text, hashcons)) {
                    (Ok(a), Ok(b)) => {
                        assert_eq!(format!("{a:?}"), format!("{b:?}"), "text {k}");
                        accepted += 1;
                    }
                    (Err(a), Err(b)) => {
                        assert_eq!((a.kind, &a.message), (b.kind, &b.message), "text {k}");
                        rejected += 1;
                    }
                    _ => panic!("text {k}: one parser accepts what the other rejects"),
                }
            }
        }
        assert!(accepted > 2 * external.len() && rejected > 0, "{accepted} accepted, {rejected} rejected");
    }
}
