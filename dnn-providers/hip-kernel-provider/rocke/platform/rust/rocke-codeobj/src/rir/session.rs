// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

//! Caller-owned compile sessions.
//!
//! A session owns what compiles reuse: a pool of persistent worker threads
//! for the parallel parser, so that a compile pays no thread start, and the
//! state of a streamed input (`feed`, then `finish`). There is no global or
//! thread-local state; everything is in the `Session` value, and dropping it
//! stops and joins its threads.
//!
//! Thread safety: a session can be used from any thread, but calls on one
//! session must not overlap; a call that would overlap another is refused
//! with a `ROCKE_DIRECT_SESSION` error and changes nothing. Distinct sessions
//! are independent.
//!
//! A session compiles exactly what `compile` does: the parse on several
//! threads gives the sequential parser's kernel or falls back to it.

use std::collections::VecDeque;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError, TryLockError};

use super::ast::Kernel;
use super::parse_par::{self, Job, WORKER_STACK};
use super::stream::Stream;
use super::{parse, CompiledKernel};
use crate::error::CompileError;
use crate::target::Target;

/// Inputs smaller than this are parsed on the calling thread. Parallel
/// parsing starts lower than without a session (`parse::PARALLEL_MIN_BYTES`)
/// because the workers are already running; below this, the later passes'
/// slower reads of an AST built on other cores cost what the parse saves.
pub const SESSION_PARALLEL_MIN_BYTES: usize = 128 << 10;
/// Input bytes per parse thread.
const SESSION_BYTES_PER_THREAD: usize = 64 << 10;
/// Most threads a parse uses by default: more parse faster but compile no
/// faster, the AST then being spread over more cores' caches.
const MAX_SESSION_THREADS: usize = 3;
/// Worker threads a session starts by default, with the calling thread
/// `MAX_SESSION_THREADS`.
const DEFAULT_WORKERS: usize = MAX_SESSION_THREADS - 1;

/// How to set up a session.
#[derive(Clone, Copy, Debug, Default)]
pub struct SessionOptions {
    /// Worker threads to start: 0 picks a default from the machine's
    /// parallelism; 1 or more starts that many. Each costs a thread start
    /// when the session is created and nothing per compile.
    pub worker_threads: u32,
    /// The target every compile of the session is for.
    pub target: Target,
}

struct PoolState {
    jobs: VecDeque<Job>,
    closing: bool,
}

struct PoolShared {
    state: Mutex<PoolState>,
    wake: Condvar,
}

/// Persistent threads that run jobs in the order they are given.
struct Pool {
    shared: Arc<PoolShared>,
    threads: Vec<std::thread::JoinHandle<()>>,
}

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

impl Pool {
    fn new(workers: usize) -> Pool {
        let shared = Arc::new(PoolShared {
            state: Mutex::new(PoolState { jobs: VecDeque::new(), closing: false }),
            wake: Condvar::new(),
        });
        let threads = (0..workers)
            .filter_map(|i| {
                let shared = Arc::clone(&shared);
                std::thread::Builder::new()
                    .name(format!("rocke-codeobj-{i}"))
                    .stack_size(WORKER_STACK)
                    .spawn(move || Pool::run(&shared))
                    .ok()
            })
            .collect();
        Pool { shared, threads }
    }

    fn run(shared: &PoolShared) {
        loop {
            let job = {
                let mut state = lock(&shared.state);
                loop {
                    if state.closing {
                        return;
                    }
                    if let Some(job) = state.jobs.pop_front() {
                        break job;
                    }
                    state = shared.wake.wait(state).unwrap_or_else(PoisonError::into_inner);
                }
            };
            // A panicking job marks its parse failed and the caller re-raises;
            // the thread itself keeps serving.
            let _ = catch_unwind(AssertUnwindSafe(job));
        }
    }

    fn spawn(&self, job: Job) {
        lock(&self.shared.state).jobs.push_back(job);
        self.shared.wake.notify_one();
    }
}

impl Drop for Pool {
    fn drop(&mut self) {
        lock(&self.shared.state).closing = true;
        self.shared.wake.notify_all();
        for thread in self.threads.drain(..) {
            let _ = thread.join();
        }
    }
}

/// Per-call state, held for the duration of each call.
#[derive(Default)]
pub(super) struct State {
    pub(super) stream: Option<Stream>,
    /// The last stream's input buffer, emptied, for the next stream to fill
    /// without growing it again.
    spare: Vec<u8>,
}

/// A compile session; see the module documentation.
pub struct Session {
    pool: Pool,
    state: Mutex<State>,
    target: Target,
}

impl Session {
    pub fn new(options: SessionOptions) -> Session {
        let workers = match options.worker_threads {
            0 => std::thread::available_parallelism().map_or(1, |n| n.get()).saturating_sub(1).min(DEFAULT_WORKERS),
            n => n as usize,
        };
        Session { pool: Pool::new(workers), state: Mutex::new(State::default()), target: options.target }
    }

    /// Worker threads running.
    pub fn workers(&self) -> usize {
        self.pool.threads.len()
    }

    /// The target the session compiles for.
    pub fn target(&self) -> Target {
        self.target
    }

    /// The per-call state, or an error if another call holds it.
    pub(super) fn enter(&self) -> Result<MutexGuard<'_, State>, CompileError> {
        match self.state.try_lock() {
            Ok(state) => Ok(state),
            Err(TryLockError::Poisoned(state)) => Ok(state.into_inner()),
            Err(TryLockError::WouldBlock) => {
                Err(CompileError::new("ROCKE_DIRECT_SESSION", "calls on one session must not overlap"))
            }
        }
    }

    /// Runs `job` on one of the session's threads.
    pub(super) fn spawn(&self, job: Job) {
        self.pool.spawn(job);
    }

    /// Threads to parse `len` bytes with: `ROCKE_DIRECT_PARSE_THREADS` if
    /// set, else one per `SESSION_BYTES_PER_THREAD` above
    /// `SESSION_PARALLEL_MIN_BYTES`, at most `MAX_SESSION_THREADS`; never
    /// more than the workers and this thread.
    pub(super) fn threads_for(&self, len: usize) -> usize {
        let cap = self.workers() + 1;
        let forced = std::env::var_os("ROCKE_DIRECT_PARSE_THREADS")
            .and_then(|v| v.to_str().and_then(|v| v.trim().parse::<usize>().ok()))
            .filter(|&n| n > 0);
        match forced {
            Some(n) => n.min(cap),
            None if len < SESSION_PARALLEL_MIN_BYTES => 1,
            None => (len / SESSION_BYTES_PER_THREAD).clamp(2, MAX_SESSION_THREADS).min(cap),
        }
    }

    /// `parse::parse` on the session's threads.
    pub fn parse(&self, text: &str) -> Result<Kernel, CompileError> {
        self.parse_shared(None, text)
    }

    /// `parse`, with the text already in an `Arc` when the caller has one.
    fn parse_shared(&self, shared: Option<&Arc<str>>, text: &str) -> Result<Kernel, CompileError> {
        let threads = self.threads_for(text.len());
        let hashcons = !super::debug_flag("nohashcons");
        if threads > 1 {
            let owned;
            let shared = match shared {
                Some(shared) => shared,
                None => {
                    owned = Arc::<str>::from(text);
                    &owned
                }
            };
            if let Some(kernel) = parse_par::parse_pooled(&|job| self.spawn(job), shared, threads, 0, hashcons) {
                return Ok(kernel);
            }
        }
        parse::parse_options(text, 1, 0, hashcons)
    }

    /// `rir::compile_for` the session's target, with the parse on the
    /// session's threads.
    pub fn compile(&self, text: &str) -> Result<CompiledKernel, CompileError> {
        let _state = self.enter()?;
        super::compile_parsed_by(text, &|t| self.parse_shared(None, t), self.target)
    }

    /// `compile` of a text the caller already holds in an `Arc`.
    pub fn compile_shared(&self, text: &Arc<str>) -> Result<CompiledKernel, CompileError> {
        let _state = self.enter()?;
        super::compile_parsed_by(text, &|t| self.parse_shared(Some(text), t), self.target)
    }

    /// Appends `bytes` to the input being streamed, starting one if none is.
    /// Complete lines are parsed on the session's threads meanwhile.
    pub fn feed(&self, bytes: &[u8]) -> Result<(), CompileError> {
        let mut state = self.enter()?;
        let State { stream, spare } = &mut *state;
        stream.get_or_insert_with(|| Stream::new(std::mem::take(spare))).feed(self, bytes);
        Ok(())
    }

    /// Compiles the streamed input, as `compile` would compile it whole, and
    /// ends the stream. An input that is not UTF-8 is refused.
    pub fn finish(&self) -> Result<CompiledKernel, CompileError> {
        let mut state = self.enter()?;
        let stream = match state.stream.take() {
            Some(stream) => stream,
            None => Stream::new(std::mem::take(&mut state.spare)),
        };
        let (compiled, mut buffer) = stream.finish(self);
        buffer.clear();
        state.spare = buffer;
        compiled
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rir::parse_par::tests::{kernel, outcome};

    #[test]
    fn sessions_parse_like_the_sequential_parser() {
        let text = kernel(40);
        let reference = outcome(parse::parse_with_threads(&text, 1));
        for workers in [1, 2, 5] {
            let session = Session::new(SessionOptions { worker_threads: workers, ..Default::default() });
            assert_eq!(session.workers(), workers as usize);
            for _ in 0..3 {
                let shared: Arc<str> = Arc::from(text.as_str());
                for chunk in [0, 1, 700] {
                    let threads = workers as usize + 1;
                    let parsed = parse_par::parse_pooled(&|job| session.spawn(job), &shared, threads, chunk, true);
                    assert_eq!(outcome(parsed.ok_or_else(|| CompileError::new("x", "fell back"))), reference);
                }
                assert_eq!(outcome(session.parse(&text)), reference);
            }
        }
    }

    #[test]
    fn overlapping_calls_are_refused() {
        let session = Session::new(SessionOptions { worker_threads: 1, ..Default::default() });
        let held = session.enter().expect("free");
        let refused = session.compile("ckdsl.ir v1\n").err().expect("refused");
        assert_eq!(refused.kind, "ROCKE_DIRECT_SESSION");
        drop(held);
        assert_eq!(session.compile("ckdsl.ir v1\n").err().map(|e| e.kind), Some("ROCKE_DIRECT_PARSE"));
    }

    #[test]
    fn a_panicking_job_leaves_the_pool_serving() {
        let session = Session::new(SessionOptions { worker_threads: 1, ..Default::default() });
        session.spawn(Box::new(|| panic!("job panics on purpose")));
        let (tx, rx) = std::sync::mpsc::channel();
        session.spawn(Box::new(move || tx.send(7).expect("send")));
        assert_eq!(rx.recv().expect("the thread still serves"), 7);
    }
}
