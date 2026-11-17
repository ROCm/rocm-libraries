<!--
Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
SPDX-License-Identifier: MIT
-->

# fp16 implicit-GEMM wgrad from the AOT cache — gfx950 case study

Why fp16 backward-weight convolution timed from the AOT cache
(`benchmark_conv_compare.py --run-from-cache`, which runs
`benchmark_implicit_gemm_conv.py --run-from-cache`) lagged bf16 on the same
shapes, and the four changes that closed most of that gap.

Per [`platform/AGENTS.md`](../../../../platform/AGENTS.md) §Compliance this document
records **methodology and levers only**. No measured throughput, latency or
ratio appears here or in any commit message. Measured scope: gfx950, fp16,
`--direction wgrad`, the cached AOT implicit-GEMM wgrad kernels.

## Table of contents

- [What was investigated](#what-was-investigated)
- [The instrument](#the-instrument)
- [Finding 1: the packed fp16 atomic fenced every add](#finding-1-the-packed-fp16-atomic-fenced-every-add)
- [Finding 2: the split-K cost model had learned the fences](#finding-2-the-split-k-cost-model-had-learned-the-fences)
- [Finding 3: the cached two-stage kernels never ran on even C/groups](#finding-3-the-cached-two-stage-kernels-never-ran-on-even-cgroups)
- [Finding 4: no data prefetch in the synchronous K loop](#finding-4-no-data-prefetch-in-the-synchronous-k-loop)
- [Not taken: unroll_k and async_dma in the AOT grid](#not-taken-unroll_k-and-async_dma-in-the-aot-grid)
- [Correctness gate](#correctness-gate)
- [Replay](#replay)
- [Keep / revert / defer decisions](#keep--revert--defer-decisions)

## What was investigated

A per-model wgrad corpus compared against the reference implementation showed
fp16 implicit-GEMM wgrad far behind it on most shapes, while bf16 on similar
shapes was close. Nearly every slow case ran a single-stage split-K kernel
(`...-cshuffle-...-sk2-...`, packed 16-bit atomic epilogue) at a shallow degree
(2 to 8) on a reduction tens of thousands deep.

## The instrument

- Same-session A/B through `benchmark_implicit_gemm_conv.py --run-from-cache
  --fast-rank --verify` on small isolated caches (a few hundred identities
  copied from the full AOT cache, or rebuilt from the edited emitter), so a
  change is timed against exactly the binaries it replaces.
- `llvm-objdump -d --mcpu=gfx950` on the cached HSACOs, and on kernels rebuilt
  with `benchmarks.common.reproduce_kernel`.
- A standalone HIP probe of `global_atomic_pk_add_f16`,
  `global_atomic_pk_add_bf16` and `global_atomic_add_f32` throughput.

## Finding 1: the packed fp16 atomic fenced every add

The probe ran the fp16 and bf16 packed atomics at the same rate, so the
difference had to be in the kernels. Their ISA differed: every
`global_atomic_pk_add_f16` was bracketed by `buffer_wbl2 sc1`,
`s_waitcnt vmcnt(0)` and `buffer_inv sc1`, the bf16 atomic was bare.

`memref.global_atomic_add_pk_f16` lowered to the legacy
`llvm.amdgcn.global.atomic.fadd.v2f16` intrinsic, which LLVM auto-upgrades to an
agent-scope `seq_cst` `atomicrmw`. Its bf16 sibling already lowered to a
`monotonic` `atomicrmw fadd <2 x bfloat>` with the AMDGPU memory-model
metadata. Both engines now lower fp16 the same way
([`lower_llvm.py`](../../../../platform/python/rocke/core/lower_llvm.py),
[`mem.cpp`](../../../../platform/cpp/core/lower_llvm/mem.cpp)); the op's existing
`ordering` attribute (default `monotonic`) is honoured. The dgrad split-K
epilogue uses the same op and gets the same fix.

## Finding 2: the split-K cost model had learned the fences

[`split_k_planner`](../../../kernels/conv/split_k_planner.py)'s gfx950
`atomic_bytes_per_latency` had been fitted while the fences were in place, so
its fp16 entry was a small fraction of the bf16 one and the planner capped fp16
atomic splits at shallow degrees. With the fix, that constant made deeper
atomic splits look too expensive; the fp16 entry now takes the bf16 value.
Fixing the lowering alone, or the constant alone, each recovered only part of
the gap -- the constant alone makes the planner pick deep splits of a kernel that
still fences, which is slower than the old shallow ones.

## Finding 3: the cached two-stage kernels never ran on even C/groups

`--run-from-cache` applied the JIT sweep's `--two-stage auto` rule: two-stage
(f32 workspace + Stage 2) kernels only for odd C/groups, where the packed
atomic cannot be built. The AOT cache holds the two-stage binaries for every
geometry, but they were dropped before timing on every even shape, and
`benchmark_conv_compare.py` had no `--two-stage` flag to override it. A planner
free to choose the reduction preferred the workspace for most of those tiles.

Under `--run-from-cache` the cached two-stage kernels now always compete with
the packed-atomic ones (`never` still drops them), and the compare driver
forwards `--two-stage`. A filter that kept only the two-stage tiles the planner
prefers was tried first and recovered less: both reductions have to be timed.

## Finding 4: no data prefetch in the synchronous K loop

The `mem`, `compv3` and `compv4` wgrad K loops (with or without
`prefetch_addr`) were one body: issue the tile's global loads, wait for them,
write LDS, barrier, MFMA, barrier. The global latency was exposed every
iteration, and the three pipelines differed only in scheduling hints.
`unroll_k` alternates two LDS buffers but still waits for the next tile's loads
after one MFMA.

`pipeline="compv3"` with `prefetch_addr` and the new `register_prefetch` flag
now does what CK's compv3 does with one LDS buffer: a prologue stages tile
`k_lo`; each iteration issues tile `k+1`'s loads into registers, runs tile
`k`'s MFMAs out of LDS while they are in flight, then barriers and overwrites
LDS
([`conv_implicit_gemm_wgrad.py`](../../../kernels/common/conv_implicit_gemm_wgrad.py),
C++ twin `wgrad_emit_kloop_prefetch_addr` in
[`conv_implicit_gemm_wgrad.cpp`](../../../cpp/instances/common/conv_implicit_gemm_wgrad.cpp)).
The last iteration loads one tile past the slice into LDS that no MFMA reads;
the reduction-axis predicate sends it to the zero-fill sentinel past the
tensor. Unlike plain `prefetch_addr` (address math only, data dependency
unchanged) this restructures the loop body, so it is its own flag rather than
folded into the `pipeline="compv3"` + `prefetch_addr` identity: default
`register_prefetch=False` keeps `pipeline="compv3"` + `prefetch_addr` on the
same loop as `mem`/`compv4`, byte-identical to before this change and under
its existing kernel name and cached binaries. `mem` and `compv4` ignore
`register_prefetch`.

`register_prefetch` is opt-in and not yet wired into anything that ships: the
AOT generator (`_wgrad_prefetch_addr` in
[`aot_jobs.py`](../../../kernels/conv/aot_jobs.py)) and the
JIT sweep (`benchmark_implicit_gemm_conv.py`) still set only `prefetch_addr`,
so the production grid and cache hold only the plain loop; the C++ default is
`false` too. The only callers today are parity configs 55-56 and the
correctness tests below. Alone the new loop helps some shapes and hurts
others on the shapes it was tried on in this investigation; building it into
the AOT/JIT grids so the per-shape best can be picked from the cache is
deferred (see Keep/revert/defer).

## Not taken: unroll_k and async_dma in the AOT grid

Building `unroll_k` variants of the winning wgrad kernels showed it ahead on
some deep-reduction shapes, but the wgrad AOT grid does not build it and adding
a K-loop axis grows the cache by a third or more; deferred. `async_dma` wgrad
binaries cannot be cached today: the run side refuses an `async_dma` identity
without the chunk widths the generator records, and the wgrad generator does
not emit async jobs.

## Correctness gate

Every kernel timed in the A/B runs passed `--verify` against the torch
reference. On GPU: `test_conv_wgrad_correctness.py` (prefetch_addr numerics now
cover compv3 with the fp16 and bf16 packed atomics and with two-stage) and
`test_conv_dgrad_correctness.py`. Host: the wgrad parity emitters gained
configs 55-56 (compv3 + packed fp16 atomic, compv3 + two-stage), the
byte-identity gate is green at llvm20/22/23/24, and the wgrad/dgrad golden
hashes were re-blessed for the intended emission change.

## Replay

From `library/`, with `PYTHONPATH=.:../platform/python` and
`ROCKE_LLVM_FLAVOR=llvm22`:

```bash
# ISA of a cached fp16 split-K kernel: look for buffer_wbl2 / buffer_inv
python -m benchmarks.common.reproduce_kernel \
    wgrad-implicit_gemm-fp16-2d-t64x64x64-w2x2-a32x32x16-v8x8x8-mem-cshuffle-grp-kouter-sk2-pfa-wave64-gfx950-llvm22 \
    --out-dir repro
llvm-objdump -d --mcpu=gfx950 repro/*.hsaco | grep -B2 -A2 global_atomic_pk_add_f16

# Rebuild the AOT cache and compare both reductions from it
python -m benchmarks.common.benchmark_implicit_gemm_conv --compile-all \
    --arch gfx950 --direction wgrad --dtype fp16 --cache-dir kernel_cache --jobs 64
python -m benchmarks.common.benchmark_conv_compare --arch gfx950 --direction wgrad \
    --run-from-cache kernel_cache --miopen-file wgrad_fp16_cases.txt --fast-rank --verify
```

`--two-stage never` reproduces the old even-C/groups behaviour.

## Keep / revert / defer decisions

- **Keep:** the relaxed, agent-scope packed fp16 atomic lowering (pinned to
  match the HIP engine's `__builtin_amdgcn_global_atomic_fadd_v2f16`, which
  already compiles to agent scope); the bf16 constant for the fp16 atomic in
  the gfx950 cost model; both reductions competing under `--run-from-cache`;
  `--two-stage` in the compare driver; the compv3 register prefetch.
- **Defer:** refitting the whole gfx950 cost model on the fixed fp16 kernels;
  refitting `atomic_bytes_per_latency`'s fp16 entry if agent-scope throughput
  turns out to differ measurably from the system-scope atomics it was copied
  from (not re-measured in this pass); building `register_prefetch` into the
  AOT/JIT generators so both K-loop shapes compete from the cache (it is
  opt-in only today -- see Finding 4); `unroll_k` in the wgrad AOT grid; async
  wgrad AOT jobs; the remaining gap on large-output stride-1 3x3 shapes and on
  small grouped shapes, which is main-loop efficiency rather than split-K.
