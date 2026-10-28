# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Rank a sweep's kernels in two rounds, with one device drain per chunk.

:class:`~benchmarks.common.early_stop.EarlyStop` times every kernel the same
way -- warmup launches, then a timed loop -- and each of those is its own
synchronous round trip: :func:`rocke.runtime.time_launches` syncs the device,
records a pair of events, waits on the second and drains the stream. On a
sweep whose kernels run for tens of microseconds those drains, not the
kernels, are the clock. This module times a whole chunk of kernels between two
drains instead, and spends the full warmup/iters protocol only on the few
candidates that could still rank.

The protocol
------------

**Round 1, the probe.** Every candidate runs ``probe_warmup`` warmup launches
and ``probe_iters`` timed ones, all queued on one stream with no host sync in
between; a pair of HIP events brackets each candidate's timed launches and the
chunk is drained once at the end. Only the best ``finalists`` candidates are
carried forward, held with their bound launches so round 2 reloads nothing.

**Round 2, the measurement.** The finalists are timed exactly as the
per-kernel path does, with the caller's full ``warmup`` and ``iters``. Every
reported number comes from that round: a probe time ranks a candidate, it
never describes one.

Why the probe still warms up
----------------------------

Two cold effects sit on a first launch, and they need different answers.

*Per kernel*: the first launch of a code object faults its instructions in and
allocates its scratch. That cost lands on whichever launch goes first, so each
candidate gets ``probe_warmup`` launches of its own before its events are
recorded. The probe is cheap because it drops the **syncs**, not the warmup.

*Per device*: clocks ramp over the first few milliseconds of sustained work.
No per-candidate warmup can cover that one -- it penalises whoever is measured
first rather than whoever is slow, and in a sweep that is simply whoever the
cache listed first. :func:`prime_device` runs one candidate enough times to
bring the clocks up before the first measured chunk, and the chunk drains are
too short for them to fall back.

What is left is sampling noise on a one- or two-launch measurement, which is
why the cut is deliberately loose: the probe only has to get a candidate into
the top ``finalists``, not to rank it there.

Timing the finalists like hipDNN
--------------------------------

Round 2 has two protocols, chosen by ``timing``. ``block`` is
:func:`rocke.runtime.time_launches`: ``iters`` launches back to back between
one event pair, averaged. ``staged`` (the default) is what hipDNN's
``dnn-benchmark`` does by default, and what a number compared against hipDNN
or MIOpen has to be:
every call is its own sample, a start/stop event pair around one call queued
behind a host-released stall (:func:`rocke.runtime.time_launches_staged`), and
the reported time is the median. The span holds no host submission, but it
does hold the fixed device cost of one event pair around a single call: a
back-to-back loop amortises it away, while hipDNN's timing (and MIOpen's,
where it records events around a launch) pays it on every call. On small
kernels that fixed cost is a large share of the time, so the two protocols
must not be compared with each other.

Only round 2 changes. The probe stays a batched block measurement either way:
it only orders the candidates, and the event cost is the same constant on all
of them, so it does not change which reach the finalists. Staged samples drain
the device after every call, so the clocks the priming brought up are allowed
to settle between samples -- deliberately, as dnn-benchmark does.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence

# Candidates timed between two device drains. The chunk bounds three things at
# once: the HIP events live at any moment (two per candidate), how long the
# shared output buffer stays in flight, and the blast radius of a kernel that
# faults -- a chunk that raises is re-timed one candidate at a time so the
# failure is charged to the one that caused it.
DEFAULT_CHUNK = 128

# Launches issued before a case's first measured chunk; see "Per device" above.
DEFAULT_PRIMER = 50

# Candidates carried from the probe into the full-fidelity round. They are held
# with their bound launches, so this also bounds what stays bound at once.
DEFAULT_FINALISTS = 64

# The probe's own protocol. One warmup launch covers the per-kernel cold cost;
# two timed launches cost little more and halve the sampling noise.
DEFAULT_PROBE_WARMUP = 1
DEFAULT_PROBE_ITERS = 2

#: How a reported time is measured; see "Timing the finalists like hipDNN"
#: above. The sweeps' ``--timing`` flag picks one for every reported number,
#: ranked or not (:meth:`benchmarks.common.early_stop.EarlyStop.measure`).
TIMINGS = ("block", "staged")
DEFAULT_TIMING = "staged"
#: The protocol of :func:`rocke.runtime.time_launches`, and of every result
#: recorded before the choice existed.
BLOCK_TIMING = "block"


def add_timing_arg(parser) -> None:
    """Add ``--timing`` to a sweep's parser."""
    parser.add_argument(
        "--timing",
        choices=TIMINGS,
        default=DEFAULT_TIMING,
        help="how every reported time is measured. staged: each launch its "
        "own stalled event pair, median of --iters -- what hipDNN's "
        "dnn-benchmark measures, so the numbers compare with hipDNN and "
        "MIOpen. block: --iters launches back to back between one event "
        "pair, averaged. Early stopping and the --fast-rank probe still rank "
        f"with block launches (default: {DEFAULT_TIMING})",
    )


@dataclass
class Ranked:
    """What the ranking measured for one candidate.

    ``probe_ms`` is set for every candidate that ran at all; ``ms`` only for a
    finalist, and it is the number to report. ``failed`` marks a candidate
    whose launch raised -- its ``probe_ms`` is meaningless and it never ranks.
    ``passed`` is what the caller's finalist hook returned (``None`` when it
    did not check).
    """

    index: int
    probe_ms: Optional[float] = None
    ms: Optional[float] = None
    passed: Optional[bool] = None
    failed: bool = False

    @property
    def is_finalist(self) -> bool:
        return self.ms is not None


def prime_device(
    run: Callable[[], None], *, launches: int = DEFAULT_PRIMER, stream: int = 0
) -> None:
    """Bring the device's clocks up before the first measured chunk.

    ``run`` is any candidate's bound launch -- the ramp is a property of the
    device, not of the kernel, so which one does the priming does not matter.
    """
    from rocke.runtime import synchronize_and_release
    from rocke.runtime.launcher import no_fence

    if launches <= 0:
        return
    with no_fence():
        for _ in range(int(launches)):
            run()
    synchronize_and_release(stream)


def _time_chunk(
    runs: Sequence[Callable[[], None]],
    *,
    warmup: int,
    iters: int,
    stream: int,
) -> List[float]:
    """Time every callable in ``runs`` between one pair of device drains.

    The launches are queued in order on one stream, so same-stream FIFO makes
    each candidate's event pair bracket exactly its own timed launches. Only
    the last event is waited on.

    Raises whatever a launch raises, after draining: the queued launches are
    fire-and-forget, so the caller must not free their buffers until the
    device is idle.
    """
    from rocke.runtime import resolve_stream
    from rocke.runtime.hip_module import Runtime
    from rocke.runtime.launcher import no_fence

    rt = Runtime()
    resolved = resolve_stream(stream)
    events = [(rt.event(), rt.event()) for _ in runs]
    try:
        with no_fence():
            for run, (start, end) in zip(runs, events):
                for _ in range(warmup):
                    run()
                start.record(stream=resolved)
                for _ in range(iters):
                    run()
                end.record(stream=resolved)
            events[-1][1].synchronize()
            out = [start.elapsed_to(end) / iters for start, end in events]
    except Exception as exc:  # noqa: BLE001 - re-raised after the drain
        _drain_after_failure(rt, exc)
        _destroy(events)
        raise
    _destroy(events)
    rt.wait_stream(resolved)
    return out


def _destroy(events) -> None:
    for start, end in events:
        start.destroy()
        end.destroy()


def _drain_after_failure(rt, exc: BaseException) -> None:
    """Drain the device after a chunk raised, before ``exc`` propagates.

    Mirrors :func:`rocke.runtime.launcher._drain_after_failure`: a drain that
    fails is not raised, because the exception worth seeing is the launch
    error, not the drain that the same faulted kernel broke.
    """
    try:
        rt.sync()
    except Exception as drain_exc:  # noqa: BLE001 - reported on ``exc`` instead
        add_note = getattr(exc, "add_note", None)
        if add_note is not None:
            add_note(f"draining the device after this error also failed: {drain_exc}")


def _time_one_by_one(
    runs: Sequence[Callable[[], None]],
    *,
    warmup: int,
    iters: int,
    stream: int,
) -> List[Optional[float]]:
    """Fall back to one drain per candidate, to attribute a chunk's failure."""
    from rocke.runtime import synchronize_and_release, time_launches
    from rocke.runtime.hip_module import HipError

    out: List[Optional[float]] = []
    for run in runs:
        try:
            out.append(time_launches(run, warmup=warmup, iters=iters, stream=stream))
        except (HipError, RuntimeError):
            # The launch raised; time_launches drained before it did, but a
            # failed drain is not what it reports, so drain again -- a drain
            # that fails here ends the sweep rather than letting the caller
            # free buffers a queued kernel still uses.
            synchronize_and_release(stream)
            out.append(None)
    return out


def time_batch(
    runs: Sequence[Callable[[], None]],
    *,
    warmup: int,
    iters: int,
    stream: int = 0,
    chunk: int = DEFAULT_CHUNK,
) -> List[Optional[float]]:
    """Per-candidate times from ``runs``, one device drain per ``chunk``.

    ``None`` for a candidate whose launch raised. A chunk that raises is
    re-timed one candidate at a time: the chunk's event pairs are all past the
    failure point, so none of its times can be trusted or attributed.
    """
    from rocke.runtime.hip_module import HipError

    out: List[Optional[float]] = []
    for start in range(0, len(runs), max(1, chunk)):
        part = runs[start : start + max(1, chunk)]
        try:
            out.extend(_time_chunk(part, warmup=warmup, iters=iters, stream=stream))
        except (HipError, RuntimeError):
            out.extend(
                _time_one_by_one(part, warmup=warmup, iters=iters, stream=stream)
            )
    return out


def time_batch_staged(
    runs: Sequence[Callable[[], None]],
    *,
    warmup: int,
    iters: int,
    stream: int = 0,
) -> List[Optional[float]]:
    """Per-candidate median of ``iters`` stalled single-call samples.

    Every sample is :func:`rocke.runtime.time_launches_staged`'s: a start/stop
    event pair around one call, queued behind a stall the host releases once
    the call is enqueued. ``warmup`` samples of each candidate are discarded.
    The process's shared gate serves every candidate.

    ``None`` for a candidate whose launch raised, and for one whose call
    blocked the host on the stalled stream (the watchdog had to release it, so
    the sample holds host time); the second is reported, as it means the bound
    launch synchronizes when it should not.

    The gate itself (``hipStreamWaitValue32`` support) is checked once, before
    any candidate is timed: a device that lacks it would otherwise fail every
    candidate the same way, one caught exception at a time, and the case would
    report no winner with nothing in the log to say why. Falls back to
    :func:`time_batch` (block timing) with a warning rather than doing that.
    """
    from statistics import median

    from rocke.runtime.hip_module import HipError
    from rocke.runtime.stall_gate import (
        StallTimeout,
        default_gate,
        time_launches_staged,
    )

    try:
        default_gate()
    except HipError as e:
        print(
            f"  [staged] {e}; this device cannot be timed with --timing staged, "
            f"falling back to --timing block for this run",
            flush=True,
        )
        return time_batch(runs, warmup=warmup, iters=iters, stream=stream)

    out: List[Optional[float]] = []
    for i, run in enumerate(runs):
        try:
            samples = time_launches_staged(
                run, warmup=warmup, iters=iters, stream=stream
            )
        except StallTimeout as e:
            print(f"  [staged] finalist {i}: {e}; not ranked", flush=True)
            out.append(None)
            continue
        except (HipError, RuntimeError):
            # time_launches_staged drained before raising.
            out.append(None)
            continue
        out.append(median(samples) if samples else None)
    return out


def rank(
    n: int,
    make_run: Callable[[int], Optional[Callable[[], None]]],
    *,
    warmup: int,
    iters: int,
    probe_warmup: int = DEFAULT_PROBE_WARMUP,
    probe_iters: int = DEFAULT_PROBE_ITERS,
    finalists: int = DEFAULT_FINALISTS,
    chunk: int = DEFAULT_CHUNK,
    primer: int = DEFAULT_PRIMER,
    stream: int = 0,
    on_probe: Optional[Callable[[Ranked], None]] = None,
    on_finalist: Optional[Callable[[int, Callable[[], None]], Optional[bool]]] = None,
    on_drop: Optional[Callable[[int], None]] = None,
    held_bytes: Optional[Callable[[], int]] = None,
    chunk_bytes: int = 0,
    timing: str = DEFAULT_TIMING,
) -> List[Ranked]:
    """Probe all ``n`` candidates, then time the best ``finalists`` in full.

    ``make_run(i)`` binds candidate ``i``'s launch, or returns ``None`` when it
    cannot be loaded at all; it is called once per candidate, during the probe,
    and the finalists' callables are kept so nothing is bound twice.

    ``on_probe`` is called with each candidate's :class:`Ranked` as its probe
    result lands, so a long case can report progress. ``on_finalist(i, run)``
    runs just before a finalist is timed -- the hook the caller verifies in --
    and a ``False`` return times the candidate anyway (so its row still carries
    a number) but keeps it from being reported as the best.

    ``on_drop(i)`` is called for every candidate whose launch this function
    will not call again: one that failed, and one the probe ranked out. A
    caller whose bind allocates device scratch frees it there -- the probe
    holds a candidate's launch for as long as it could still rank, which is
    longer than a serial sweep holds it and is why the scratch has to be
    accounted for rather than freed before the next bind.

    ``held_bytes`` and ``chunk_bytes`` bound that scratch. When both are set,
    the chunk being bound is closed as soon as ``held_bytes()`` -- the device
    memory the caller is holding for this ranking, retained finalists included
    -- passes ``chunk_bytes``. A candidate whose own scratch is over the budget
    gets a chunk to itself, which is what a serial sweep would have done with
    it anyway. ``chunk`` still caps the count.

    The finalists' scratch is still held when this returns -- they were timed
    with it -- so the caller frees that along with the rest of the case.

    ``timing`` picks round 2's protocol (:data:`TIMINGS`): ``block`` times the
    finalists with :func:`time_batch`, ``staged`` with
    :func:`time_batch_staged`, so ``ms`` is then a median of single-call
    samples. The probe is the same either way.

    Returns one :class:`Ranked` per candidate, in candidate order.
    """
    if timing not in TIMINGS:
        raise ValueError(f"timing must be one of {TIMINGS}, got {timing!r}")
    results = [Ranked(index=i) for i in range(n)]
    # The finalists' bound launches, kept alive from the probe. A launch holds
    # its packed kernargs and its kernel handle, so keeping it is what makes
    # round 2 free; the dict is capped at ``finalists`` so a case with tens of
    # thousands of candidates still only ever holds that many.
    kept: dict = {}
    primed = False

    def drop(i: int) -> None:
        kept.pop(i, None)
        if on_drop is not None:
            on_drop(i)

    def trim() -> None:
        """Keep only the best ``finalists`` of everything probed so far."""
        if len(kept) <= finalists:
            return
        ordered = sorted(kept, key=lambda i: results[i].probe_ms)
        for i in ordered[finalists:]:
            drop(i)

    nxt = 0
    while nxt < n:
        runs = []
        live = []
        # Bind until the chunk is full by count, or until the caller is holding
        # more device scratch than it wants to; a chunk always takes at least
        # one candidate so an oversized one cannot stall the walk.
        while nxt < n and len(runs) < max(1, chunk):
            i = nxt
            nxt += 1
            run = make_run(i)
            if run is None:
                results[i].failed = True
                drop(i)
                if on_probe is not None:
                    on_probe(results[i])
                continue
            runs.append(run)
            live.append(i)
            if chunk_bytes and held_bytes is not None and held_bytes() > chunk_bytes:
                break
        if not runs:
            continue
        if not primed:
            prime_device(runs[0], launches=primer, stream=stream)
            primed = True
        times = time_batch(
            runs, warmup=probe_warmup, iters=probe_iters, stream=stream, chunk=chunk
        )
        for i, run, ms in zip(live, runs, times):
            if ms is None:
                results[i].failed = True
            else:
                results[i].probe_ms = ms
                kept[i] = run
            if on_probe is not None:
                on_probe(results[i])
        # Trim to the best ``finalists`` as we go, so a candidate that cannot
        # rank gives its launch -- and the caller its scratch -- back at the
        # end of the chunk that measured it rather than at the end of the case.
        trim()
        for i in live:
            if results[i].failed:
                drop(i)

    # Round 2. The finalists are verified first and then timed as one batch.
    # In block timing a candidate's timed launches start right after the
    # previous one's, so the clocks that the priming brought up never fall
    # back between them; staged timing drains after every sample, as hipDNN's
    # does (see the module docstring).
    order = sorted(kept, key=lambda i: results[i].probe_ms)
    if on_finalist is not None:
        for i in order:
            # A kernel that fails here is still timed, so its row carries a
            # number; the caller reads ``passed`` and keeps it from ranking.
            results[i].passed = on_finalist(i, kept[i])
    finals = [kept[i] for i in order]
    if timing == "staged":
        times = time_batch_staged(finals, warmup=warmup, iters=iters, stream=stream)
    else:
        times = time_batch(
            finals, warmup=warmup, iters=iters, stream=stream, chunk=chunk
        )
    for i, ms in zip(order, times):
        if ms is None:
            results[i].failed = True
        else:
            results[i].ms = ms
    return results
