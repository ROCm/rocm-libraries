# Shared 3D decode host-overhead case study

This study isolates host submission overhead in the shared gfx942/gfx950 3D
split-KV runtime. It compares ordinary graph-enabled calls, direct replay of
that captured graph, eager calls, prepared eager calls, and fixed-binding replay
scopes. It makes no general
kernel-performance claim; the emitted segment/reducer computation is unchanged.

## Ordinary eager submission

The ordinary gfx942/gfx950 3D path also reuses scratch and packed launch
arguments without requiring callers to enter a replay scope. A bounded cache
holds at most eight eager owners by resolved spec/launch geometry, segment
count, device and effective stream. Its lock
covers the complete segment/reducer submission. Stream FIFO ordering
protects sequential reuse. Capture is checked again after switching to the
selected stream and always uses graph-private allocations.

Follow the canonical [attention caller and cache contract](../../../../docs/strided-kv-decode.md#streams-and-workspace-ownership):
submit and clean up on one host thread, stop submission before drains, and use
finite asynchronous batches. Eager-cache clearing, graph clearing and prepared
execution closing are separate completion boundaries.

Every ordinary invocation retains its support and architecture checks and reads
live tensor addresses and scalar values. Unchanged bindings reuse immutable
packed segment/reducer buffers; a change constructs a new pair. The cache holds
packed addresses, scratch and compiled launchers, rather than request-local input
storage. A separate completion registry snapshots input storage on the first
submission of a pair between drains, retains each pair once, and releases it
only after a successful explicit stream/device drain or caller-confirmed
external synchronization. Eviction and partial submission failure therefore
cannot discard in-flight arguments. Default-stream retention includes device
identity so draining one device cannot release another device's queued buffers.

Validate this path by timing the ordinary saved-spec dispatcher closure with
internal graph replay disabled, using the pre-ownership parent
`b69179f7d5df831788b2c1cd5c919d63459124b2` as baseline. Keep setup and numerical
checks outside timing while including every ordinary invocation's host work.
Use identical specs and quantized inputs, independent reference checks before
and after timing, identical LLVM/HSACO and geometry, and three sequential trials
with the middle order reversed. Capture-only or scoped timing cannot substitute
for this ordinary-call gate. Configuration-specific evidence stays in its
access-controlled validation record.

## Diagnosis and narrow change

A stream-ownership repair added a device/ExternalStream context around every
call. Stable graph calls also rebuilt and hashed full tensor binding metadata.
Direct device replay and segment/reducer LLVM/HSACO digests stayed stable,
which isolated the regression to the host wrapper rather than kernel tuning.
The baseline for this investigation is develop commit
`55c7ae91080af4cbe09d5d00b350db918815a5d5`; the earlier ownership change is
PR #13110.

On gfx942/gfx950, the runtime now skips the context only when Q's device and the effective stream
are already current. It normalizes live layout metadata once, checks it and
storage addresses against the resident entry, and probes the most recent graph
before hashing its immutable identity. These graph changes add no additional
lifetime cache; ordinary eager ownership is described above.
Content changes reuse the graph; address changes drain and replace it.
Prepared layout/device/stream validation, graph-private scratch, cache bounds,
and asynchronous argument retention remain part of the contract.

The ordinary call still validates arbitrary live bindings on every invocation.
For repeated fixed-binding loops, `Attention3DExecution.replay_scope()` moves
validation and stream setup to entry, retaining either the captured graph or
bound segment/reducer launch buffers until exit drains the stream. The decode
benchmarks can explicitly time the yielded public callable with
`--dsl-timing scoped`; their default `--dsl-timing ordinary` times ordinary public
runtime calls after route selection outside timing. Binding changes go through
`rebind()` between scopes; the general `launch()` path retains its per-call
validation and outer-capture support.

Ordinary rebinding borrows the owner's submission lock without constructing a
scope-only replay operation. It therefore supports gfx1250 eager and graph
bindings while scoped replay remains restricted to gfx942/gfx950. Replacement
tensors remain reachable if submission or the final completion drain fails.
gfx1250 preserves its prior explicit contexts, storage-bearing graph key, and
cache eviction behavior; the host shortcuts are confined to gfx942/gfx950.

## Scenarios and replay procedure

The diagnostic input is BF16, B=1, Sq=1, Hq=32, Hkv=4, D=128, paged KV with
block size 16 and an identity page table, full attention, and Skv in
4096/16384/65536. Use identical seeded, quantized inputs for both arms. This is
an isolated synthetic wrapper diagnostic, not a workload-wide tuning sweep.

1. Create clean baseline and candidate checkouts. Load one release-matched
   ROCm/PyTorch environment (the validation used ROCm 7.14 and
   `torch==2.13.0+rocm7.14.0`, NumPy 2.2.6). Pin the source and compiler digests.
   Isolate COMGR compilation from Torch when their LLVM libraries conflict.
2. Reserve one gfx950 node and GPU for the entire comparison. Pin every process
   to the same CPU. Run baseline then candidate, candidate then baseline, and
   baseline then candidate sequentially; no comparison arms overlap.
3. Construct `UnifiedAttentionProblem` and invoke
   `run_unified_attention_torch` with `backend="auto"` on the current stream.
   Enable `HIPDNN_GFX950_3D_GRAPH=1` for ordinary graph calls. Warm up once before
   timing; record direct `entry.graph.replay` separately as a diagnostic control.
   Disable the environment switch for ordinary eager calls. Prepare once with
   `backend="3d", use_graph=False` for prepared eager calls.
   For candidate scoped modes, prepare outside timing and time the actual
   callable yielded by `execution.replay_scope()` with graph enabled and disabled.
   Direct graph replay is only a diagnostic control, not the production API.
4. In every arm and mode, compare output with an independent CPU FP32 reference
   after BF16 quantization: repeat KV heads for GQA, compute scaled QK,
   softmax over valid tokens, then multiply by V. Require finite output and
   `atol=0.01, rtol=0.02` before recording performance.
5. Use `rocke.runtime.time_launches` on the same stream with 20 warmups and
   300 iterations per sample. Collect seven samples per mode and use the median
   after discarding the first sample; report the median of three trial medians.
   Separately profile unfenced submission with `perf_counter_ns`, draining
   retained arguments after every batch. Preserve per-trial results and hashes
   in the private evidence record; keep absolute measurements out of git.
6. Check the ordinary-call improvement against the direct-replay control and
   the pre-ownership parent. A remaining gap to that parent is a residual host
   cost, not evidence that device execution regressed.

## Fixed-binding scope results

The scoped public callable was compared with the pre-ownership parent's ordinary
graph-enabled call at commit `b69179f7d5df831788b2c1cd5c919d63459124b2`,
using the BF16 paged configurations and release-matched method above on gfx950.
The candidate extends `2b5fa56a4ec8968edef6f8c697ed4e2f14666cb8` with
`replay_scope()`; both launch the same segment/reducer work. Seven event samples
per mode were collected, discarding the first, across three sequential trials
with the middle order reversed. Selection and preparation were outside timing.

| KV length | Scoped graph / parent graph time | Scoped eager / parent graph time |
| --- | --- | --- |
| 4096 | 0.876 | 0.845 |
| 16384 | 1.007 | 0.966 |
| 65536 | 0.999 | 0.979 |

Each measured arm passed the independent CPU FP32 reference after BF16
quantization with `atol=0.01, rtol=0.02`; paired LLVM/HSACO digests and launch
geometry matched. The complete decode numeric suite also passed on gfx942 and
gfx950, including both scoped modes, content updates and storage rebinding.
This is a historical fixed-binding synthetic decode comparison: scope entry and exit costs
are excluded, and a loop that changes bindings must pay the rebind boundary.
The ordinary validated call at that revision retained a host overhead gap. No gfx942 timing claim
or workload-wide speedup follows from these results. The release wheel's device
reference kernels were unavailable, so inputs and the reference were computed
on CPU and transported to/from the rocKE GPU kernels.

## Correctness replay and file map

From the rocKE root, with its platform and library roots on `PYTHONPATH`:

```bash
python -m pytest -q library/tests/test_attention_workspace.py \
  library/tests/test_attention_3d_replay_bindings.py \
  library/tests/test_attention_3d_replay_scope.py
ROCKE_BACKEND=python ROCKE_REQUIRE_DECODE_GPU=gfx950 python -m pytest -q \
  library/tests/test_strided_kv_decode_numeric.py
ROCKE_BACKEND=python ROCKE_REQUIRE_DECODE_GPU=gfx942 python -m pytest -q \
  library/tests/test_strided_kv_decode_numeric.py
```

Use each GPU command on its matching target with release-matched runtime
libraries. The numeric suite compares independent FP32 references and covers
paged/strided FP16/BF16 content updates and Q/output storage rebinding, plus
eager/internal/external GPU work on separate streams submitted sequentially
from one host thread. Host tests cover context
restoration, capture, bounded ownership, binding changes and LRU replacement.
These commands establish correctness; they do not establish a performance win.

A bounded gfx1250 eager rebinding case initially passed with BF16 paged D64 GQA-8,
new Q/output bindings, same-object storage updates, changing device lengths,
and a nondefault stream. It used an independent CPU FP32 reference after input
quantization. The installed ROCm 10.0 Torch wheel's device initialization kernels
failed, so validation supplied an equivalent separate HIP F32 initializer on
the active stream. A final repeat failed during preparation, before rebinding,
with illegal-address/illegal-instruction errors; the published-code control also
failed during preparation. The earlier pass is limited adapter-based evidence,
not a reproducible final hardware gate.
Graph attempts produced numerical failures and illegal-address errors, including
a pre-change runtime control; standalone initializer capture/replay was also
unstable. They do not establish gfx1250 graph correctness or a regression caused
by these changes. The prior graph behavior is retained pending a working lane.

| File (relative to rocKE) | Responsibility |
| --- | --- |
| `library/kernels/common/attention_unified.py` | Stream boundary, live binding validation and bounded paired graphs |
| `library/tests/test_attention_workspace.py` | Stream, capture and scratch ownership contracts |
| `library/tests/test_attention_3d_replay_bindings.py` | Content/address/layout changes, clear and LRU behavior |
| `library/tests/test_attention_3d_replay_scope.py` | Fixed-binding scope, thread ownership, failure drain and rebind |
| `library/tests/test_attention_3d_eager_reuse.py` | Ordinary pair reuse, storage retention, explicit-stream capture and completion cleanup |
| `library/tests/test_strided_kv_decode_numeric.py` | Independent on-GPU numerical and lifetime gate |
| `docs/strided-kv-decode.md` | Caller stream, update and completion contract |
| `platform/dsl_docs/optimization/optimization_runbook.md` §3.5 | Host-versus-device diagnosis procedure |


## Validation after integrating architecture-independent selection

Commit `2dde6147d01c31837d863191f89a56bfd1917531` integrates develop
`0868455a0554b9afb6cc88d514a151d801aaeb87`. The merge retains explicit
architecture arguments and architecture-bearing compile keys while preserving
scoped replay and graph binding ownership. It also removes strided callers'
obsolete problem-field architecture override and passes the launch architecture
to their support gate.

On gfx950, the three BF16 paged shapes above were measured in one allocation
(Slurm job 68095469), using ROCm 7.14, Torch `2.13.0+rocm7.14.0`, NumPy 2.2.6,
Python 3.12, isolated COMGR, and a pinned CPU. The source arms were the repaired
commit, prior PR head `205a989d89ae058c5fd5d988fbd73dc46e013434`, and the
pre-ownership parent `b69179f7d5df831788b2c1cd5c919d63459124b2`.
Each arm ran in a fresh process, sequentially across three trials with the middle
order reversed. Each mode used seven event samples, discarding the first, with
20 warmups and 300 iterations. The ratios below use the median of trial medians.
Preparation, scope entry/exit, and rebinding remain outside the timed callable.

| KV length | Repaired scoped graph / prior scoped graph | Repaired scoped eager / prior scoped eager | Repaired scoped graph / parent graph | Repaired scoped eager / parent graph |
| --- | --- | --- | --- | --- |
| 4096 | 0.999 | 0.998 | 0.885 | 0.844 |
| 16384 | 1.012 | 1.011 | 1.007 | 0.941 |
| 65536 | 0.999 | 1.001 | 0.998 | 0.971 |

These runs preserve the scoped timing at the measured configurations. The
ordinary validated graph call has a separate cost: repaired/prior-PR ratios
were 1.047, 1.039, and 0.998 respectively. The scope results do not establish
an improvement for arbitrary ordinary calls or workloads with changing bindings.

All 81 measured modes passed independent CPU FP32 references after BF16
quantization before and after timing, with `atol=0.01`, `rtol=0.02`, finite
outputs, and poisoned output buffers before checking. Segment/reducer LLVM and
HSACO digests and launch geometry matched across all source arms and trials.
The repaired commit's full decode numeric suite passed 92 cases on gfx950 with
no skips. Source payload SHA-256 was
`bfd8a41e8f9a405a96915ae3e67a01613b69b6ec4092c5560354371d579e07ed`;
all transferred source blobs were checked against their pinned Git trees.
The input/reference calculation ran on CPU, and Torch transported tensors to and
from the rocKE device kernels. This remains synthetic fixed-binding evidence.


## Benchmark timing boundaries

The generic gfx942/gfx950 decode benchmarks default to ordinary public
runtime calls, with route selection outside timing. Returned route names
are metadata, not concrete emitted-kernel identities. `--dsl-timing scoped` selects the prepared fixed-binding
callable for 3D entries, excluding preparation, scope entry/exit, and rebinding.
A 2D entry remains ordinary in either mode. JSON output identifies the requested
mode and actual per-entry mode; comparison baselines retain their ordinary
call boundaries. Scoped comparisons therefore include different host costs.

An external saved-spec caller can bind tuning metadata once and still call
`run_unified_attention_torch()` on every iteration. That binding alone does
not enter `Attention3DExecution.replay_scope()`. On gfx950, the internal 3D
graph heuristic is disabled by default; an ordinary graph-enabled diagnostic
is a different mode from default eager dispatch. An outer captured batch
measures device replay separately and excludes most host submission costs.
The scoped results above are separate from the ordinary eager submission gate
described at the start of this study.

The architecture-independent support gates already parsed decorated target IDs
on develop. The new shortcut returns early for canonical IDs.
The common launch path already supplies canonical `gfx942`, `gfx950`, or
`gfx1250`, so those inputs bypass repeated target parsing. Decorated and unknown
target IDs retain the original parser; compile-cache architecture agreement
checks remain in place. Host coverage checks both paths without a device probe.


## Ordinary saved executable-spec reproduction

The shared harness binds an executable `AttentionTuningSpec` through the
candidate's `bind_torch` and measures ordinary binding `.launch()` calls.
Default generic dispatch instead returns a routing `AttentionSpec`; passing
that routing label as `tuning_spec` is invalid. Scoped and graph replay remain
separate diagnostic modes.

Run the synthetic default matrix on either gfx942 or gfx950:

```bash
python -m benchmarks.common.benchmark_decode_host_overhead --arch gfx942 --source-revision <commit> --output private-results.json
```

The harness disables internal graphs, uses seeded quantized inputs, checks an
independent NumPy FP64 reference before and after timing (`atol=0.01`,
`rtol=0.02` for BF16), and records finite unfenced host-enqueue batches with
drains outside the host timer. HIP-event timings are recorded separately.
Setup, saved-spec binding, compilation and numerical checks stay outside both
timers. Input metadata remains fixed after binding; the changing-maximum mode
creates bindings outside timing for compatible maxima in one selector bucket.
The JSON includes absolute measurements and belongs in a private evidence
record. For source comparisons use identical case/spec configurations, GPU,
CPU affinity, environment and cache setup, with three rotating-order trials.
Retain source/archive/compiler/adapter hashes and compare LLVM/HSACO and actual
launch geometry before attributing changes solely to host work.
