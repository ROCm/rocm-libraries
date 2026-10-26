// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

//! Streaming ingestion: a producer feeds the IR text in pieces while it is
//! still writing it, and the session's threads parse completed lines
//! meanwhile.
//!
//! - **Feed.** Bytes are appended to the input. Once the header and
//!   parameters are complete, the body is cut into chunks at line ends as
//!   soon as a chunk's worth of lines is complete *and* so is the next line
//!   after it that is not blank or a comment (the first pass looks at it).
//!   Each chunk is copied, with those lines after it, into a segment of its
//!   own, and its first pass (`parse_par::scan_chunk`) is queued on the
//!   session's threads.
//! - **Finish.** The rest of the body is cut and queued the same way; the
//!   caller runs every first pass no thread has started. The name table is
//!   sized for the whole input and filled from every chunk on all threads;
//!   the walk, the second pass and the stitching are those of `parse_par`.
//!
//! The result is that of `parse::parse` on the whole input: the parallel
//! parse gives the sequential parser's kernel or declines, and any decline
//! (an unusual header, a chunk the first pass does not take, a chunk that is
//! not UTF-8) parses the whole input sequentially instead.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, PoisonError};

use super::ast::Kernel;
use super::parse::{self, Lines};
use super::parse_par::{self, Info, Segments, Shared, Work};
use super::scan::Level;
use super::session::Session;
use super::CompiledKernel;
use crate::error::CompileError;

/// Body bytes per chunk: small enough that chunks reach the threads while
/// the producer is still writing, large enough that each chunk's fixed cost
/// stays small.
pub const STREAM_CHUNK_BYTES: usize = 16 << 10;

/// The first pass of a cut chunk, run by whichever thread starts it first.
#[derive(Default)]
struct Scan {
    started: AtomicBool,
    done: AtomicBool,
    result: Mutex<Option<(Info, Work)>>,
}

impl Scan {
    fn run(&self, segment: &str, base: usize, len: usize, hashcons: bool, level: Level) {
        if !self.started.swap(true, Ordering::AcqRel) {
            let result = parse_par::scan_chunk(segment, base, 0, len, hashcons, level);
            *self.result.lock().unwrap_or_else(PoisonError::into_inner) = result;
            self.done.store(true, Ordering::Release);
        }
    }

    /// The result, waiting for the thread running it; `None` if the first
    /// pass declined the chunk.
    fn take(&self) -> Option<(Info, Work)> {
        let mut spins = 0u32;
        while !self.done.load(Ordering::Acquire) {
            spins += 1;
            if spins < 64 {
                std::hint::spin_loop();
            } else {
                std::thread::yield_now();
            }
        }
        self.result.lock().unwrap_or_else(PoisonError::into_inner).take()
    }
}

/// A chunk cut from the input: `segment[..len]` at input offset `base`.
struct Cut {
    segment: Arc<str>,
    base: usize,
    len: usize,
    scan: Arc<Scan>,
}

/// An input being streamed into a session.
pub(super) struct Stream {
    /// Every byte fed so far.
    input: Vec<u8>,
    /// Where the body starts, once the header and parameters are complete.
    body_start: Option<usize>,
    /// End of the body bytes cut into chunks so far.
    cut: usize,
    cuts: Vec<Cut>,
    /// The parallel parse is off for this input: finish parses it
    /// sequentially.
    declined: bool,
    hashcons: bool,
    level: Level,
}

/// The offset just past the line that contains `from`, if it is complete.
fn line_end(input: &[u8], from: usize) -> Option<usize> {
    input.get(from..)?.iter().position(|&b| b == b'\n').map(|i| from + i + 1)
}

/// Where the body starts if the header and parameters are complete in
/// `text`: just after the first `}` line that follows the first `params {`
/// line, where `parse::prologue` stops on a well-formed header. (The parse
/// reads the header properly at the end and declines if this was wrong.)
fn body_start(text: &str) -> Option<usize> {
    let mut lines = Lines::new(text, 0, text.len());
    while let Some(line) = lines.next_line() {
        if line == "params {" {
            while let Some(line) = lines.next_line() {
                if line == "}" {
                    return Some(lines.pos());
                }
            }
            return None;
        }
    }
    None
}

/// Whether `Lines` skips this line: blank (as `str::trim` has it) or a comment.
fn skipped(line: &[u8]) -> Option<bool> {
    let line = std::str::from_utf8(line).ok()?.trim();
    Some(line.is_empty() || line.starts_with('#'))
}

impl Stream {
    /// A stream that collects its input in `buffer` (emptied first).
    pub(super) fn new(mut buffer: Vec<u8>) -> Stream {
        buffer.clear();
        Stream {
            input: buffer,
            body_start: None,
            cut: 0,
            cuts: Vec::new(),
            declined: false,
            hashcons: !super::debug_flag("nohashcons"),
            level: Level::detect(),
        }
    }

    pub(super) fn feed(&mut self, session: &Session, bytes: &[u8]) {
        self.input.extend_from_slice(bytes);
        if self.declined || session.workers() == 0 {
            return;
        }
        if self.body_start.is_none() {
            let complete = self.input.iter().rposition(|&b| b == b'\n').map_or(0, |i| i + 1);
            let Ok(text) = std::str::from_utf8(&self.input[..complete]) else {
                self.declined = true;
                return;
            };
            match body_start(text) {
                Some(start) => {
                    self.body_start = Some(start);
                    self.cut = start;
                }
                None => return,
            }
        }
        let input = std::mem::take(&mut self.input);
        while self.cut_one(session, &input, false) {}
        self.input = input;
    }

    /// Cuts the next chunk of the body from `input` and queues its first
    /// pass, if the chunk's lines and the next line after it that is not
    /// blank or a comment are complete; with `last`, `input` is all there is.
    fn cut_one(&mut self, session: &Session, input: &[u8], last: bool) -> bool {
        let start = self.cut;
        if self.declined || start >= input.len() || self.cuts.len() + 1 >= parse_par::MAX_CHUNKS {
            return false;
        }
        let end = if start + STREAM_CHUNK_BYTES < input.len() {
            match line_end(input, start + STREAM_CHUNK_BYTES) {
                Some(end) => end,
                None if last => input.len(),
                None => return false,
            }
        } else if last {
            input.len()
        } else {
            return false;
        };
        let mut ahead = end;
        loop {
            if ahead >= input.len() {
                if last {
                    break;
                }
                return false;
            }
            let Some(next) = line_end(input, ahead) else {
                if last {
                    ahead = input.len();
                    break;
                }
                return false;
            };
            let Some(skip) = skipped(&input[ahead..next]) else {
                self.declined = true;
                return false;
            };
            ahead = next;
            if !skip {
                break;
            }
        }
        let Ok(segment) = std::str::from_utf8(&input[start..ahead]) else {
            self.declined = true;
            return false;
        };
        let cut = Cut { segment: Arc::from(segment), base: start, len: end - start, scan: Arc::default() };
        let (segment, scan, hashcons, level) = (Arc::clone(&cut.segment), Arc::clone(&cut.scan), self.hashcons, self.level);
        let (base, len) = (cut.base, cut.len);
        session.spawn(Box::new(move || scan.run(&segment, base, len, hashcons, level)));
        self.cuts.push(cut);
        self.cut = end;
        true
    }

    /// Ends the input and compiles it; also hands back the input's buffer.
    pub(super) fn finish(mut self, session: &Session) -> (Result<CompiledKernel, CompileError>, Vec<u8>) {
        let input = std::mem::take(&mut self.input);
        if input.is_empty() {
            return (Err(CompileError::new("request", "serialized_ir is empty")), input);
        }
        let text = match String::from_utf8(input) {
            Ok(text) => text,
            Err(e) => {
                let error = CompileError::new("request", format!("serialized_ir is not UTF-8: {}", e.utf8_error()));
                return (Err(error), e.into_bytes());
            }
        };
        let first = Mutex::new(Some(self));
        let compiled = super::compile_parsed_by(&text, &|t| match first.lock().unwrap_or_else(PoisonError::into_inner).take() {
            Some(stream) => stream.parse(session, t),
            None => session.parse(t),
        });
        (compiled, text.into_bytes())
    }

    /// The parse of the whole `text`, with the first passes done so far.
    fn parse(mut self, session: &Session, text: &str) -> Result<Kernel, CompileError> {
        if self.cuts.is_empty() || self.declined {
            return session.parse(text);
        }
        while self.cut_one(session, text.as_bytes(), true) {}
        match self.complete(session, text) {
            Some(kernel) => Ok(kernel),
            None => parse::parse_options(text, 1, 0, self.hashcons),
        }
    }

    /// The parallel parse of the cut chunks, or `None` to decline.
    fn complete(&self, session: &Session, text: &str) -> Option<Kernel> {
        // Chunks no thread has started are scanned here.
        for cut in &self.cuts {
            cut.scan.run(&cut.segment, cut.base, cut.len, self.hashcons, self.level);
        }
        let scans: Vec<(Info, Work)> = self.cuts.iter().map(|cut| cut.scan.take()).collect::<Option<_>>()?;
        if self.declined || self.cut != text.len() {
            return None;
        }
        let (header, params, body_start) = parse_par::header(text)?;
        if Some(body_start) != self.body_start {
            return None;
        }
        let mut segments = vec![(Arc::<str>::from(&text[..body_start]), 0)];
        let mut bounds = vec![(0, 0)];
        for cut in &self.cuts {
            segments.push((Arc::clone(&cut.segment), cut.base));
            bounds.push((0, cut.len));
        }
        let sh = Shared::new(bounds, params, text.len() - body_start, true, self.hashcons)?;
        sh.scanned_already(scans)?;
        let sh = Arc::new(sh);
        let src = Arc::new(Segments(segments));
        parse_par::run_pooled(&|job| session.spawn(job), &sh, &src, session.workers() + 1, header)
    }
}

#[cfg(test)]
mod tests {
    use super::super::session::{Session, SessionOptions};
    use crate::rir::parse_par::tests::{kernel, outcome};
    use crate::rir::{compile, parse};

    fn hsaco_or_error(r: Result<crate::rir::CompiledKernel, crate::CompileError>) -> Result<Vec<u8>, String> {
        r.map(|c| c.hsaco).map_err(|e| format!("{}: {}", e.kind, e.message))
    }

    /// Every way of feeding the input parses it as the sequential parser
    /// does, including after a previous stream on the same session.
    #[test]
    fn every_feeding_gives_the_sequential_kernel() {
        let text = kernel(120);
        assert!(text.len() > 4 * super::STREAM_CHUNK_BYTES);
        let reference = outcome(parse::parse_with_threads(&text, 1));
        for workers in [1, 3] {
            let session = Session::new(SessionOptions { worker_threads: workers });
            for piece in [1usize, 7, 333, 4096, 65536, text.len()] {
                for part in text.as_bytes().chunks(piece) {
                    session.feed(part).expect("feeds");
                }
                let mut state = session.enter().expect("free");
                let mut stream = state.stream.take().expect("a stream");
                drop(state);
                assert!(!stream.cuts.is_empty() && !stream.declined, "pieces of {piece}");
                while stream.cut_one(&session, text.as_bytes(), true) {}
                let kernel = stream.complete(&session, &text).ok_or_else(|| crate::CompileError::new("x", "declined"));
                assert_eq!(outcome(kernel), reference, "pieces of {piece}");
            }
        }
    }

    /// Streamed compiles agree with `compile`, for valid and broken inputs.
    #[test]
    fn streamed_compiles_agree_with_compile() {
        let session = Session::new(SessionOptions { worker_threads: 2 });
        let good = kernel(60).replace("    %lt : i1", "    %st : i32 = arith.add ( %c0, %n )\n    %lt : i1");
        let inputs = [
            good.clone(),
            good.replace("arith.add ( %r3,", "arith.add ( %nope,"),
            good.replace("region @body {\n", "region @body\n"),
            good[..good.len() / 2].to_string(),
            good.replace("params {", "params"),
            String::new(),
        ];
        for text in &inputs {
            for part in text.as_bytes().chunks(5000) {
                session.feed(part).expect("feeds");
            }
            let streamed = hsaco_or_error(session.finish());
            let reference = if text.is_empty() {
                Err("request: serialized_ir is empty".to_string())
            } else {
                hsaco_or_error(compile(text))
            };
            assert_eq!(streamed, reference);
        }
        session.feed(&[b'x', 0xff, b'\n']).expect("feeds");
        let error = session.finish().err().expect("not UTF-8");
        assert_eq!(error.kind, "request");
        assert!(error.message.starts_with("serialized_ir is not UTF-8"), "{}", error.message);
    }
}
