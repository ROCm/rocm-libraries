# Shared 3D decode host-overhead case study

This study isolates host submission overhead in the shared gfx942/gfx950 3D
split-KV runtime. It compares ordinary graph-enabled calls, direct replay of
that captured graph, eager calls, prepared eager calls, and fixed-binding replay
scopes. It makes no general
kernel-performance claim; the emitted segment/reducer computation is unchanged.

## Diagnosis and narrow change

A stream-ownership repair added a device/ExternalStream context around every
call. Stable graph calls also rebuilt and hashed full tensor binding metadata.
Direct device replay and segment/reducer LLVM/HSACO digests stayed stable,
which isolated the regression to the host wrapper rather than kernel tuning.
The baseline for this investigation is develop commit
`55c7ae91080af4cbe09d5d00b350db918815a5d5`; the earlier ownership change is
PR #13110.

The runtime now skips the context only when Q's device and the effective stream
are already current. It normalizes live layout metadata once, checks it and
storage addresses against the resident entry, and probes the most recent graph
before hashing its immutable identity. No additional lifetime cache is added.
Content changes reuse the graph; address changes drain and replace it.
Prepared layout/device/stream validation, graph-private scratch, cache bounds,
and asynchronous argument retention remain part of the contract.

The ordinary call still validates arbitrary live bindings on every invocation.
For repeated fixed-binding loops, `Attention3DExecution.replay_scope()` moves
validation and stream setup to entry, retaining either the captured graph or
bound segment/reducer launch buffers until exit drains the stream. The decode
benchmarks time the yielded public callable. Binding changes go through
`rebind()` between scopes; the general `launch()` path retains its per-call
validation and outer-capture support.

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
   300 iterations per sample. Collect nine samples per mode and use the median
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
This is a fixed-binding synthetic decode comparison: scope entry and exit costs
are excluded, and a loop that changes bindings must pay the rebind boundary.
The ordinary validated call retains a host overhead gap. No gfx942 timing claim
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
existing eager/internal/external graph concurrency. Host tests cover context
restoration, capture, bounded ownership, binding changes and LRU replacement.
These commands establish correctness; they do not establish a performance win.

| File (relative to rocKE) | Responsibility |
| --- | --- |
| `library/kernels/common/attention_unified.py` | Stream boundary, live binding validation and bounded paired graphs |
| `library/tests/test_attention_workspace.py` | Stream, capture and scratch ownership contracts |
| `library/tests/test_attention_3d_replay_bindings.py` | Content/address/layout changes, clear and LRU behavior |
| `library/tests/test_attention_3d_replay_scope.py` | Fixed-binding scope, thread ownership, failure drain and rebind |
| `library/tests/test_strided_kv_decode_numeric.py` | Independent on-GPU numerical and lifetime gate |
| `docs/strided-kv-decode.md` | Caller stream, update and completion contract |
| `platform/dsl_docs/optimization/optimization_runbook.md` §3.5 | Host-versus-device diagnosis procedure |
