# `SynchronizerValidator` investigation (gfx950 per-config rerun 2)

`full_runs_gfx950/per_config_rerun_2/AGGREGATE_analysis.txt` reports
`validate_warmups_listener.SynchronizerValidator` at 162,734.88 ms over 2,106,827 calls (6.3% of
wall). This note explains where that time goes, proposes moving the check to the GPU, and
describes an attribution edge case that any rework has to handle. Timing claims below come from
measurements; reading the code was only used to decide what to measure. Statements that are
inferred rather than measured are marked as such.

| Timer | Count | Total (ms) | Mean (ms) |
| --- | --- | --- | --- |
| `validate_synchronizer_check` | 2,106,827 | 162,692.31 | 0.08 |
| ├─ `validate_synchronizer_scan` | 608,873 | 106,747.41 | 0.18 |
| ├─ `validate_synchronizer_readback` | 608,873 | 55,816.84 | 0.09 |
| └─ `validate_synchronizer_inactive` | 1,497,978 | 27.90 | 0.00 |

All of the cost is in the 608,873 *active* checks; solutions that cannot use the Synchronizer cost
nothing.

## Method

- **Binary.** The measured client, `/tmp/tl_client_msgpack/tensilelite/client/tensilelite-client`
  (Release, `-O3 -DNDEBUG -std=c++20 -fPIC -mf16c -x hip --offload-arch=gfx950`). The linked
  `scanSynchronizerResidue` was disassembled with `llvm-objdump`.
- **Microbenchmarks.** A verbatim copy of `scanSynchronizerResidue` compiled with the client's
  flags; its assembly is instruction-for-instruction identical to the shipped binary. Timings are
  medians over 300–1000 iterations with `std::chrono::steady_clock`.
- **In-client measurement.** The `dtl` and `subtile_mxfp4` configs re-run exactly as
  `full_runs_gfx950/profiling/run_rerun.sh` does (cached artifacts, `TimingInstrumentation=True`,
  `taskset -c 0-24`, `ROCR_VISIBLE_DEVICES=0`), with `LD_PRELOAD` shims that intercept `hipMemcpy`:
  - *readback split:* for calls made from `SynchronizerValidator::readBuffer` (return address in
    `[0x41b1f0, 0x41b3e8)`; the binary is not PIE), time a `hipDeviceSynchronize()` (waiting for
    queued GPU work) separately from the copy. On an idle GPU that sync costs 0.07 µs, so the
    "wait" figure is real waiting and not shim overhead.
  - *reset tracking:* remember the device pointers `readBuffer` copies from, and count other
    `hipMemcpy` writes into those pointers between checks.
- **Reproduction.** The re-run reproduces the production numbers for `dtl`: scan 8,376.56 ms vs
  8,313 ms, readback 3,214.89 ms vs 3,381 ms, both over 49,995 checks.
- **Environment.** gfx950 host, EPYC 9575F. `perf` was unavailable (`perf_event_paranoid=4`).
- Scratch sources (`bench.cpp`, `scan.cpp`, `shim.cpp`, `shim_reset.cpp`, `run_cfg.sh`) and logs
  are in `/tmp/syncval_bench`.

## What the check does and when it runs

Stream-K flags, GSU/MBSK reduction counters and the amax counter all live in one shared
Synchronizer buffer of 409,600 alpha-typed elements (`ClientProblemFactory.cpp`), i.e. 1.6 MB for
a float alpha. Kernels must leave it zeroed on exit; a kernel that leaves residue can corrupt or
hang later launches. The check reads the buffer back after a launch, scans it for any nonzero
32-bit word, and if dirty logs the count and first offset, marks the solution `FAILED`, and
clears the buffer.

Per solution (`main.cpp`):

```text
prepareGPUInputs (input reset)
launch warmup 0 (async)
listeners.validateWarmups      ← SynchronizerValidator, then ReferenceValidator, then BenchmarkTimer
launch warmups 1..N-1
postWarmup
benchmark runs (syncs × enqueues; BenchmarkTimer::postEnqueues synchronizes)
postSolution
```

The check therefore sees exactly one launch per solution.

## Slowness 1: the host scan (106.7 s)

`scanSynchronizerResidue` tracks `nonzero` and `first` per element. In the shipped binary this
compiles to a scalar loop, unrolled 2×, with a loop-carried `cmov`/`add` dependency chain and no
SIMD:

```text
41b610: testq   %r8, %r8
        movq    %rcx, %r11
        cmoveq  %r9, %r11
        xorl    %ebx, %ebx
        cmpl    $0x0, (%rdi,%r9,4)
        setne   %bl
        cmoveq  %rcx, %r11
        addq    %r8, %rbx
        ...      (second element, same pattern)
        addq    $0x2, %r9
        cmpq    %r10, %r9
        jne     41b610
```

Measured (median, identical assembly):

| Variant | 1.6 MB | 3.2 MB | 6.5 MB |
| --- | --- | --- | --- |
| Current scan, warm cache | 162.9 µs | 326.2 µs | 650.5 µs |
| Current scan, right after the D2H copy | 166.0 µs | 327.9 µs | 657.7 µs |
| Current scan, after evicting 1 GB of cache, then the copy | 164.8 µs | 327.3 µs | 655.8 µs |
| OR-reduction first (auto-vectorizes to 256-bit `vorps`), warm | 11.3 µs | 21.4 µs | 41.3 µs |
| OR-reduction first, right after the D2H copy | 32.6 µs | 66.2 µs | 131.1 µs |

The current loop runs at a flat ~10 GB/s regardless of where the data is cached. It is bound by the
dependency chain (~0.4 ns per int), not by memory. In the client it measured 168 µs per check on
`dtl`, and per-config means in the aggregate are 0.165–0.175 ms for almost every config, consistent
with 1.6 MB buffers. 608,873 × ~165 µs ≈ 100 s, essentially the whole scan bucket.

The count and first offset are only needed for the error message on a dirty buffer, which should
be rare. The clean path only needs "is anything nonzero".

## Slowness 2: the readback (55.8 s)

The readback is a blocking `hipMemcpy` D2H of the whole buffer into a pinned staging buffer. Its
time has three parts:

1. **The copy itself.** ~37–40 µs per 1.6 MB in steady state, both in the microbenchmark (36.9 µs
   pinned, 37.5 µs pageable, GPU idle) and in the client (37.9 µs on `subtile_mxfp4`).
2. **Waiting for the warmup kernel (misattributed).** Warmup 0 is launched asynchronously and
   `SynchronizerValidator` is the first listener to block, so its readback absorbs the rest of the
   kernel's runtime plus launch-to-completion latency. In the microbenchmark, a copy issued right
   after launching a 0 / 50 / 200 µs kernel takes 53–58 / 104–106 / 254–255 µs.
3. **One-time and outlier copies.** The first copy in each client process is slow (up to 80 ms),
   and some later copies take over 100 µs. The cause was not isolated. This matters for configs
   split into many small client processes.

Measured in the client with the readback-split shim:

| | `dtl` (49,995 checks) | `subtile_mxfp4` (4,327 checks, 47 processes) |
| --- | --- | --- |
| Wait for queued GPU work | 1,184 ms (37%), ~20–29 µs/check | 156 ms (16%), ~36 µs/check |
| Copy, total | 2,009 ms (63%) | 800 ms |
| ├─ steady state | ~37–40 µs/copy | 151 ms (37.9 µs/copy) |
| ├─ first copy per process | small | 413 ms |
| └─ other copies > 100 µs | small | 236 ms (302 copies) |

**Inferred aggregate split:** 608,873 × ~38 µs ≈ 23 s of the 55.8 s is steady-state copying. The
remaining ~33 s is waiting plus first-copy and outlier costs. That split was measured only on these
two configs.

### The intended sync runs too late

`--sync-after-warmups` (default `true`) makes `BenchmarkTimer::validateWarmups` call
`hipEventSynchronize` on the warmup stop event, timed as `validate_gpu_sync`. Listeners run
`validateWarmups` in registration order, and `BenchmarkTimer` is registered after both validators
(`main.cpp`). The GPU is already idle when it runs:

| Run | `validate_gpu_sync` calls | Total | Mean |
| --- | --- | --- | --- |
| Aggregate | 2,106,827 | 6,867.77 ms | 3.3 µs |
| `dtl` re-run | 270,802 | 68.10 ms | 0.25 µs |

When `SynchronizerValidator` is inactive, the same wait presumably lands in `ReferenceValidator`'s
`validate_gpu_readback` instead (inferred from the ordering, not measured).

## Proposal: run the check on the GPU

Replace the full readback and host scan with an OR-reduction kernel over the buffer, followed by a
4-byte readback of the result. Only read the whole buffer back when the result is nonzero, to
produce the existing diagnostic (nonzero count, first offset).

```text
hipMemsetAsync(flag, 0, 4, stream)
orReduce<<<…, stream>>>(synchronizer, ints, flag)      // atomicOr of per-thread ORs
hipMemcpyAsync(hostFlag, flag, 4, D2H, stream)
hipStreamSynchronize(stream)
if(*hostFlag) { full readback + existing scan for the message; mark FAILED; clear }
```

Measured on the 1.6 MB buffer (median):

| Approach | Per check |
| --- | --- |
| Current: full D2H copy + scalar scan | ≈ 37 + 165 ≈ 200 µs |
| Full copy + vectorized OR-reduction on the host | ≈ 37 + 33 ≈ 70 µs |
| GPU OR-reduction + 4-byte readback | 16.8 µs |

The GPU version removes both slow parts: the scan (≈ 100 s) and the 1.6 MB D2H transfer (≈ 23 s
steady state, plus the per-process first-copy costs). It adds ~608,873 × 16.8 µs ≈ 10 s. The
expected net saving is roughly 110 s of the 162.7 s (inferred from per-check measurements, not
measured end-to-end). The host-vectorized variant is a smaller change that keeps the copy and saves
roughly 80 s.

**Separately, fix the wait attribution.** Synchronize on the warmup stop event in `main.cpp` right
after the first warmup launch, under its own timer, before `listeners.validateWarmups`. Then
neither validator absorbs kernel time. Reordering the listeners is riskier, because `postSolution`
runs them in reverse and the verdict ordering depends on that.

### Interaction with the client execution lock

None (code reading). `ClientExecutionLock` (`filelock.FileLock`, one lock file per physical GPU
under xdist) is held around the whole `tensilelite-client` subprocess in `ClientWriter.py` and
`ParallelExecution.py`. The reduction kernel is one more launch from the process that already
holds the lock. Points to watch:

- Enqueue it on the client's `stream`, so it stays ordered after the solution's launches and
  before anything that follows.
- It reads 1.6 MB through L2/MALL, which the DMA copy may not. At today's position, warmups 1..N-1
  follow it before the timed runs; at the proposed `postSolution` position (below), it runs after
  them. Its effect on cache state was not measured.
- FFM emulator runs (`HSA_MODEL_MEMFILE`) use private per-worker locks. A 409,600-element kernel
  may be much slower than a copy under emulation; a host fallback for that mode may be worth
  keeping (not measured).
- `CpuOnly` skips the client launch entirely and is unaffected.

## Edge case: residue attribution

The check runs once per solution, after the **first** warmup launch. Residue left by any later
launch of the same solution is not observed at that solution's check. What happens to it depends
on whether the next solution's input reset re-zeroes the buffer. There is no dedicated "clear the
Synchronizer between solutions" step; kernels are trusted to self-clean.

`prepareGPUInputs` (`DataInitialization.hpp`) has two paths:

- **Slow path** (bounds checking enabled, problem-dependent data, or swizzled tensors). It calls
  `copyInputs`, which copies *every* tensor from its pristine GPU copy, including the zero-filled
  Synchronizer. The buffer is re-zeroed before every solution. **Measured** on `dtl`: all 49,995
  checks were preceded by a D2D write into the Synchronizer buffer (111,153 writes in total).
- **Fast path** (otherwise). It calls only `resetOutput`, which skips non-output tensors. The
  Synchronizer is not marked as an output, so it carries over to the next solution. This is from
  the code; not measured, because none of the cached configs take the fast path.

| Path (aggregate reset counters) | Configs with checks | Checks | Residue from a launch after N's first warmup |
| --- | --- | --- | --- |
| Slow only, no fast-path resets (`dtl`, `subtile_mxfp4`, …) | 82 | 361,102 (59%) | wiped before N+1 runs; **never reported** |
| Fast (`fp16_tn`, `fp32_tn`, `bf16_tn`, `subtile_bf16`, … 22 configs) | 22 | 247,771 (41%) | carried over; **reported against the next active solution**, which is marked `FAILED` |

On the fast path, the next active solution may also be a different problem (if no remaining
solution of the current problem can use the buffer), and its own first launch runs on stale
flags. Residue that appears on the *first* launch is attributed correctly on both paths; a
deterministic self-cleaning bug is therefore caught. Only residue that first appears on later
launches is affected: race- or timing-dependent behavior, a different kernel variant in later
warmups (`kernels[i % kernels.size()]`), or behavior that depends on the buffer's starting state.
Two more gaps: the last checked solution's late residue is never examined, and configs that
launch with zero warmups (`NumWarmups` 0 and validation off) never trigger `validateWarmups`.

No dirty-buffer report ("Synchronizer left dirty") appears in any log examined (the `dtl` and
`subtile_mxfp4` re-runs, and the compressed logs under `timing_runs`). The edge case is a
correctness gap in the design, not an observed failure.

### Requirement for the GPU check

Whatever replaces the current check must observe **every** launch of a solution before the next
solution's launches, and must attribute the result to that solution. Running the GPU reduction
in `postSolution`, instead of after the first warmup, satisfies this:

- It sees all warmups and benchmark runs of the solution, on both reset paths. That also closes
  the slow-path gap, where late residue is currently wiped unseen.
- It is still one check per solution.
- The GPU is already idle there when benchmark runs happened (`BenchmarkTimer::postEnqueues`
  synchronizes), so no kernel time is misattributed. Enqueuing on `stream` keeps it ordered either
  way.
- `postSolution` receives no inputs, so the `ProblemInputs` pointer must be saved from
  `validateWarmups` or `preSolution`.
- It must run before `ReferenceValidator::postSolution` reports the verdict, as the existing
  registration order already arranges.
- Keep clearing the buffer after a dirty result: on the fast path nothing else will.

Trade-off: today a dirty first launch is cleared before the solution's timed runs. With the check
in `postSolution`, a failing solution's timed runs execute on a dirty buffer. That solution is
marked `FAILED` either way.

## Not measured / open items

- End-to-end savings of the GPU check (only per-check costs were measured).
- Fast-path carry-over in a real run. A small fast-path config such as `benchmark_runs__fp32_tn`
  (120 checks) or `streamk__sk_hgemm_race` (256 checks) would need to be built to show it with the
  reset-tracking shim.
- Where the warmup wait lands when `SynchronizerValidator` is inactive.
- The cause of the slow first copy per process and of the >100 µs outlier copies.
- Cache effects of the reduction kernel on subsequent timed runs, and its cost under the FFM
  emulator.
