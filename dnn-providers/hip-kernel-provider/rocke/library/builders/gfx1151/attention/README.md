# gfx1151 WMMA FMHA-forward: an optimization case study

## Inference coverage benchmark

The current inference harness is
[`benchmarks/gfx1151/attention`](../../../benchmarks/gfx1151/attention/).
It runs a fixed, seeded 54-case matrix covering FP16/BF16, head dimensions
64/128/256, MHA/GQA/MQA, causal alignment, sequence tails, decode, paged and
ragged inputs, sliding windows, softcap, sinks, ALiBi, FP32 QQ-bias, FP8 KV
storage, and feature combinations. This is ordinary SDPA coverage, not an
absorbed-attention or backward benchmark.

- `cases.py` defines the immutable workload and an independent CPU oracle.
  The oracle reads the already-rounded input storage, computes in FP64, and
  returns FP32. The numerical gates are max-absolute error `2e-2` for FP16 and
  `4e-2` for BF16; nonfinite or unwritten outputs fail.
- `candidate.py` uses public `dispatch_attention` and its tensor binding.
  Real HIP tensor owners exercise the library path without requiring torch;
  unsupported rows remain required.
- Correctness is checked before and after timing. Timing uses one HIP stream,
  graph replays of 32 operations, seven timing batches, and the median batch
  duration. Compilation, allocation, transfers, and the oracle are not timed.
- Every run prints a per-case record, a grouped table, provenance, and
  `METRIC` lines; `coverage_passed` is the primary metric.

Run on a gfx1151 device with the platform Python package and library on
`PYTHONPATH`:

```bash
python -m benchmarks.gfx1151.attention.benchmark_sdpa
```

### Dense forward dtype support

`WmmaFmhaFwdSpec` accepts `fp16` (including the `f16` spelling) and `bf16`.
The selected WMMA atom and all Q/K/V/output pointer types follow that dtype.
FP16 kernel-cache names remain unchanged; BF16 has distinct names, preventing
cross-dtype cache collisions. Both causal and noncausal paths and the optional
V-LDS staging path support the dtype selection. Aligned query/KV tiles remain
the default specialization; the independent tail flags below handle partial
tiles. Packed and paged inputs use the explicit layout selection below.

The C `rocke_wmma_fmha_fwd_spec_t` now includes `dtype`, initialized to `"fp16"`
by `rocke_wmma_fmha_fwd_spec_default()`. Native callers must rebuild against
the updated header and archive; the pybind spec adapter also carries the field.
The parity emitters cover both dtypes, head sizes, masks, GQA, and V staging.

### Aligned FP16 transposed-QK specialization

Public dispatch selects `transposed_qk=True` for dense FP16 D64/D128 attention
with no mask or either causal alignment, query lengths divisible by 16, and
KV lengths divisible by 32. Score features, sequence tails, packed/paged
layouts, BF16, FP8 KV, and D256 retain the shared WMMA path.

The specialized helper computes `K @ Q.T`, reduces softmax over lane-local
accumulator slots and a cross-half exchange, then computes `V @ P`. The
probability transpose stays in registers; V remains ordinary row-major
`[B,S,Hkv,D]`. No host/device transpose or preprocessing is required.
The implementation specializes the transposed-QK design from
[PR #9710](https://github.com/ROCm/rocm-libraries/pull/9710), not its optional
pre-transposed-V or experimental scheduling variants.

`block_n` accepts 32 or 64; `num_waves` accepts 1 or 2. Each wave writes
16 query rows. Dispatch uses `(block_n=64, num_waves=2)` when the query length
is at least 512 and divisible by 32 and the KV length is divisible by 64;
otherwise it uses `(32,1)`. Direct spec callers must satisfy the selected
query-group and KV-tile alignment. The grid helper and tensor binding reject
partial groups/tiles rather than silently dropping them.

Bottom-right alignment shifts the visible-key bound by `Sk-Sq`. Empty query
groups skip the KV loop. True negative-infinity initialization and safe
empty-row exponential shifts preserve exact-zero fully masked prefixes;
the legacy top-left specialization is unchanged.

The new variant has a distinct cache name and preserves the base tensor ABI.
Its native spec fields and pybind mapping mirror Python; native consumers must
rebuild against the updated header. Existing non-transposed specializations
are unchanged. Numeric regressions cover both tile sizes, wave counts, masks,
GQA, independent batch/head/column coordinates, and output guards.

### Output-column tiling

`value_tile_size=0` preserves the full-head WMMA path. A nonzero value must
be a proper multiple-of-16 divisor of `head_size`; it selects that many
PV/output columns per workgroup without truncating QK's head dimension.
`value_tiles` reports the number of output partitions, and the grid's Z axis
is `batch * value_tiles`. The kernel decodes the logical batch and column
offset; callers still pass full-sized Q/K/V/O tensors and their ordinary strides.

Each partition computes the same complete attention normalization and writes
disjoint output columns. This trades repeated QK work for fewer live output
accumulators and more workgroups. V-LDS staging, when enabled, stages only the
selected columns. Dense, packed and paged addressing and FP8 KV storage retain
their existing meanings. The knob is independent of sequence tails but cannot
be combined with `transposed_qk`.

Public dispatch uses this profile for dense D256 requests with Q-matched KV
storage, no extra score features or window, at least 128 KV tokens, and at most
128 query workgroups before output tiling. Noncausal prefill uses 32-column tiles with
V-LDS staging; causal prefill uses 64-column tiles with direct V gathers.
Queries of at most 16 rows use 64-column tiles with V-LDS staging. Other
requests retain their existing policy. The tile width is part of the cache
identity; native consumers must rebuild against the extended spec header.

Numeric regressions place all QK signal in the final head dimension, so an
incorrectly shortened QK reduction fails even in the first output partition.
They cover every output column, query/KV tails, both dtypes, packed/paged FP8,
and exact-zero empty sequences.



### Bottom-right causal alignment

With `mask_mode="causal"` and `causal_bottom_right=True`, the mask admits
`key_index <= query_index + seqlen_k - seqlen_q`. This supports chunked dense
prefill and also `seqlen_q > seqlen_k`, where the fully masked query prefix is
written as zero. Both input dtypes and both V-staging choices use the same rule.
Alignment uses the logical sequence lengths, including when tail flags are set.

The bottom-right variant has a distinct kernel-cache key. It uses true negative
infinity for masked scores and the initial row maximum. An empty row uses a
zero exponential shift, avoiding `inf-inf` while retaining zero probability
mass. BF16 also uses this normalization for ordinary attention: large finite
negative logits must not underflow solely because of a finite initial maximum.
`test_wmma_fmha_fwd_numeric.py` covers constant-value preservation under those
logits and exact-zero masked prefixes. Legacy FP16 default emission remains
unchanged; the BF16 normalization change is intentional.

### Sequence tails and dense decode

Set `query_tail=True` when the query length is not divisible by 16, and
`kv_tail=True` when the KV length is not divisible by 16. Query tiling then
rounds up; invalid Q lanes load zero and do not store output. The KV-tail
variant includes the final partial tile, substitutes zero for invalid K/V
loads, and excludes those keys from softmax. The two flags are independent,
so a one-token query with an aligned cache does not pay for KV-tail masking.

No input/output padding is required. The shared WMMA helper clamps a row before
using its address callback, and both direct V gathers and V-LDS staging
zero-fill invalid values. Numeric tests use NaN-poisoned input padding and
output canaries to detect leaked padding values or out-of-range stores.
The benchmark adapter selects these flags from each case's true lengths.
Aligned configurations retain their previous code and cache names.

### Dense score features

`sliding_window=W` intersects the causal mask with the last W keys. The builder
bounds the KV tile loop to the relevant window rather than scanning masked
cache blocks. `use_softcap`, `use_sinks`, `use_alibi`, and `use_qq_bias` are
independent compile-time feature switches; they compose with both dtypes,
causal alignments, and tail specializations.

The score order is scaled QK, softcap, ALiBi, QQ-bias, then masking. Softcap
uses the stable AMDGPU-lowerable `tanh` operation. ALiBi uses
`slope[head] * (key_position - context)`, including in combinations with sinks.
QQ-bias is FP32 and uses query-local rows and context-relative key columns;
out-of-bounds bias entries contribute zero. A sink is an always-visible
softmax logit with no value-vector contribution, so it adds denominator mass
without being multiplied into the output.

Enabled features append arguments to the base ABI, in order: a positive FP32
`softcap`; an input-dtype `sink_ptr`; FP32 `alibi_slopes_ptr`; FP32 `qq_bias_ptr`
and its I32 row count, column count, and element row stride. Disabled features
add no arguments. Use `wmma_fmha_fwd_signature`, the actual `KernelDef.params`,
or the native signature API with the standard kernarg packer. Native signatures
own their names/types in the caller arena.

### Packed variable lengths and paged KV

`layout="ragged"` reads packed Q/O `[total_q,Hq,D]` and K/V
`[total_k,Hkv,D]`, with I32 `cu_seqlens_q` and `cu_seqlens_k` prefix sums.
`layout="paged"` keeps packed Q/O and reads K/V caches
`[num_pages,page_block_size,Hkv,D]` through an I32 `block_table`; per-sequence
KV lengths come from I32 `seqused_k`. Page size is a positive power of two.
The block table need not be identity, contiguous, or sequence-ordered.

Pass the maximum query length to `wmma_fmha_fwd_grid` and the sequence count
as `batch`. Each workgroup loads its sequence's real lengths on the device;
query tiles beyond that sequence return before touching Q/K/V. Both packed
layouts always enable query and KV tail bounds, independently of the dense
specialization flags. Empty-KV sequences produce zero output. Bottom-right
alignment, windows, and score features use per-sequence logical positions,
not packed offsets or physical page numbers.

Layout metadata follows the enabled score arguments in the ABI. Ragged adds
`cu_seqlens_q`, `cu_seqlens_k`. Paged adds `cu_seqlens_q`, `seqused_k`,
`block_table`, `block_table_stride`, `stride_k_block`, `stride_v_block`.
All strides are in their pointer's element units; page strides are separate
from token/head strides. K/V head elements remain contiguous. No host
densification or input/output padding is required.

The native spec and pybind conversion mirror these fields. Existing dense
configurations retain their ABI and emitted code. Numeric regressions cover
empty query/KV sequences, mixed lengths, shuffled pages, poisoned padding,
output guards, both dtypes, and both V-staging choices.

### Compiler policy and launch bounds

LLVM launch bounds use the target's wave size as the minimum, capped by the
kernel's declared maximum. A single-wave gfx1151 kernel therefore declares
`32,32`, not an inverted `64,32` range. LLVM
[discards invalid ranges](https://llvm.org/docs/doxygen/AMDGPUSubtarget_8cpp_source.html#l00157)
and substitutes its defaults, which loses the intended register-allocation
constraint. Both engines emit the corrected bounds. The GPU regression queries
the compiled function's maximum block size through HIP rather than checking
source text.

`WmmaFmhaFwdSpec.scheduler_strategy` selects the existing typed LLVM scheduler
policy: `max-ilp`, `max-memory-clause`, `iterative-ilp`, `iterative-minreg`, or
`iterative-maxocc`. `None` preserves the backend default and legacy cache names.
An explicit policy is part of the kernel name, complete spec cache key, and
serialized kernel attributes in both engines. Unsupported values are rejected.
The HIPCC alternative does not support this LLVM scheduler option. Native
callers must rebuild against the extended spec header and keep its policy
string alive while using the spec.

### FP8 KV storage

Set `kv_dtype="fp8e4m3"` for OCP E4M3FN byte storage with FP16/BF16 Q and O.
The WMMA path decodes bytes with integer/IEEE operations, multiplies by ordinary
FP32 `k_scale` and `v_scale`, then casts each operand to the Q dtype before
the existing 16-bit WMMA. It does not require native FP8 conversion or FP8 WMMA
instructions. Both scales are runtime scalars appended after layout metadata;
FNUZ and E5M2 storage are not accepted.

The decoder preserves all finite OCP values, signed zeros, subnormals, and the
two NaN encodings. Numeric regressions enumerate all 256 bytes and check
non-power-of-two scale rounding. FP8 storage composes with packed/paged layouts,
tail bounds, both V-staging choices, and the score features above.

### Head dimensions, causal tile skipping, right windows, dense strides

- **Head dimensions.** The Q/K width (`head_size`) and the V/O width
  (`v_head_size`, zero meaning equal) are independent multiples of 16 up to 256.
  The transposed-QK specialization stays restricted to equal power-of-two
  widths; every other width pair uses the standard path. Output-column tiling
  applies to the V/O width.
- **Causal tile skipping.** `causal_tile_skip` bounds the K loop at the causal
  diagonal of the standard path (either alignment), so fully masked key tiles
  are never loaded. It is a default-off spec flag; dispatch sets it for
  causal masks without an explicit window.
- **Right window.** `window_right >= 0` keeps keys with
  `k <= q + ctx + window_right`, where `ctx` is `0` for top-left and
  `Sk - Sq` for bottom-right alignment. Combined with `sliding_window` it gives
  a two-sided local window; `-1` disables it. It requires `mask_mode="none"`,
  the standard path, and no `causal_tile_skip`, and adds a `wr{N}` part to the
  kernel name. Dispatch accepts it for `NO_MASK`, `SLIDING_WINDOW` and
  `BOTTOM_RIGHT_CAUSAL` requests; `TOP_LEFT_CAUSAL` is rejected because the
  causal mask already fixes the right edge.
- **Dense strides.** When the dense batch stride equals
  `seqlen * token_stride` the batch is folded into the grid. Any other stride
  (padded or gapped batches) is served by one launch per batch with
  per-batch pointer offsets, so no layout is rejected for its batch stride.
- **Offset overflow.** The kernel computes element offsets in I32. The binding
  rejects any request whose largest element offset would not fit, rather than
  launching and silently wrapping.
- **Sweep.** `sweep_space` offers distinct valid variants, baseline first,
  gated on the device compute-unit count (`num_cus`; zero uses the reference
  part) instead of a fixed query-group limit.

Adding spec fields changes the C struct layout; the ABI string is
`rocke-attention-gfx1151/v2`. Rebuild native callers and zero-initialise with
`rocke_wmma_fmha_fwd_spec_default()`.

### Public library selection and launch

`dispatch.attention.AttentionRequest(arch="gfx1151", ...)` auto-selects
`attention_gfx1151_wmma` for supported requests. Set `layout` explicitly to
`dense`, `ragged`, or `paged`; `auto` resolves to dense on this candidate and
preserves legacy conventions on other architectures. `use_fp8`,
`use_softcap`, `use_sinks`, `use_alibi`, and `use_qq_bias` describe required
features before selection, not features inferred silently at bind time.

For FP16 D64 ragged causal requests and dense query/KV-tail requests without
additional score features or an explicit window, dispatch uses V-LDS staging
and `max-ilp`. Causal profiles use an effective window spanning both advertised
maximum sequence lengths, preserving every causally-visible key for either
alignment, including top-left `Sq > Sk`. Unmasked profiles keep the window
disabled. Those maxima must bound the device-resident sequence lengths.
The profile admits maxima up to `2**30` to keep context/window arithmetic
within I32; other requests retain their existing policy. No sequence metadata
is copied to the host to choose the profile.

`dispatch_attention(request).bind_torch(tensors, **scalars)` accepts caller-owned
`q`, `k`, `v`, `out` and the selected metadata/auxiliary tensors. Despite the
historical method name, any real device-tensor owner implementing shape, dtype,
element strides, device, and `data_ptr()` can bind. The adapter validates rank,
shape agreement, vector alignment, metadata contiguity, and the dense
batch stride (folded or per-batch launch) without reading GPU sequence contents.

The compiled launcher is cached by target and complete spec. Repeated
`binding.launch(stream=...)` calls reuse it. The default fence synchronizes only
that stream; `fence=False` permits asynchronous and graph-captured launches.
After external stream synchronization, call
`rocke.runtime.launcher.release_retained_for_stream(stream)` to release retained
tensor owners. Captured launch owners must remain alive until their graphs
are destroyed. The benchmark drains them at that boundary, outside timing.

The sections below are a historical campaign, not results for this benchmark.

The sections below record the design lessons of an earlier optimization campaign
on gfx1151 (RDNA3.5, wave32, a single 16x16x16 WMMA atom). They are qualitative.
They are not results for the comparator above, and this document deliberately
carries no absolute performance figures: re-measure with the harnesses listed
under "Files" before acting on any ordering below.
[`ALGORITHM.md`](ALGORITHM.md) derives the kernel from the math and is the place
to start if flash attention is new to you.

The study applies the
[optimization runbook](../../../../platform/dsl_docs/optimization/optimization_runbook.md)
to the native gfx1151 WMMA flash-attention forward kernel
(`kernels.gfx1151.wmma_fmha_fwd`). Every variant was correctness-gated against a
reference before it was timed.

- **Part 1** applies the runbook loop to one lever (V-LDS staging). The
  "optimization" regressed and was reverted.
- **Part 2** sweeps many levers across several kernel bodies (`fmha_singlewave`,
  `fmha_pipelined`, `fmha_blockn`, `fmha_multiwave`, `fmha_regblocked`) and
  diagnoses why the kernel is issue- and register-bound rather than FLOP-bound.
- **Part 3** surveys the kept winners across a shape matrix and records the
  campaign's one large algorithmic win, the causal early exit.

## The kernel under test

One wave32 owns 16 Q rows for a `(q_tile, head, batch)`. Per K-tile it runs
`QK^T` (WMMA), an online softmax, then `PV` (WMMA), carrying the running max
`m`, the running sum `l`, and the `<8 x f32>` PV accumulator as `scf.for`
iter-args. `BLOCK_M = BLOCK_K = 16`.

The PV matmul's B operand is the sensitive part. WMMA computes `A @ B^T`, so
`PV = P @ V` needs `V` in `(d x k)` layout: for a lane's d-column the B fragment
is `V[k, d_col]` for `k = 0..15`, an inherently column-strided gather of `V`
(stride = `head_size`).

## The loop

Each lever followed the runbook loop:

1. **Hypothesis.** State the bottleneck the lever attacks and the instruction or
   resource counter that should move.
2. **One lever.** Change exactly one thing, behind a spec field or config knob,
   so the A/B stays reproducible.
3. **Measure.** Correctness gate first (against a numpy reference), then time
   with HIP events, A/B back-to-back in one thermal window.
4. **Inspect the ISA.** Explain the result with static instruction counts and
   the VGPR/spill note from the HSACO rather than guessing.
5. **Keep or revert.** Revert anything that is not a win after repeat, and keep
   the knob so the experiment can be replayed.

## Part 1: the V-LDS staging study

**Hypothesis.** The baseline gathers the PV V operand straight from global
memory with one scalar load per `(d, k)`. Staging each K-tile's V rows into LDS
once with wide loads, then reading the B operand from LDS, should cut global
traffic and speed the kernel up.

**One lever.** `WmmaFmhaFwdSpec.v_lds_stage`: `False` is the per-element global
gather; `True` vector-loads each k-row into an LDS tile, shares the existing
P-staging barrier, and reads the B operand from LDS.

**Result.** The variants were bit-identical to each other and within tolerance of
the reference. The staged variant was consistently slower across the shapes and
head sizes tried, causal and non-causal. Static instruction counts confirmed the
hypothesis about global loads (far fewer) and refuted everything else:

1. The strided access pattern is relocated, not removed. The B operand is still a
   column gather, now as many scalar, column-strided `ds_load`s with bank
   pressure.
2. One wave per CTA leaves nothing to overlap with the barrier the staged path
   forces, so its latency is fully exposed.
3. The baseline loads are cache-resident: across the wave each k-row is read
   once, so the gather's global traffic was already close to optimal.

**Decision.** Revert the default. `v_lds_stage` defaults to `False` and the
toggle remains. Dispatch re-enables it only for profiles where a fresh
measurement favors it (for example D64 tails and tiled D256 values).

Lessons:

- Fewer global loads is not faster. Trading global loads for LDS wins only when
  LDS changes the access pattern or enables reuse the cache was not providing.
- LDS staging needs occupancy to pay for its barrier.
- The cache does real work on this part; an "obviously bad" memory pattern can
  be fine, and only measurement tells.

## Part 2: the multi-lever campaign

A single lever that loses is not proof the kernel is optimal: a lever can be
dead alone yet alive in combination, or masked by another bottleneck. A heavily
parameterized vehicle (`fmha_singlewave.py`, driven by `tune.py`) swept levers
individually and in combination.

**Name the roofline first.** The initial throughput target exceeded the part's
physical f16 WMMA peak, which no kernel can reach. That is a specification
error, caught with one line of arithmetic before spending iterations. The honest
objective is to approach the roofline, not exceed it.

**Why it is stuck: issue- and register-bound.** WMMA instructions are a tiny
fraction of issued instructions, so the kernel is limited by instruction issue
and VGPR pressure rather than FLOPs or bandwidth. At D128 the PV accumulator,
the K fragments, and the softmax temporaries saturate the VGPR budget and spill;
D64 halves the accumulator and runs spill-free. That is why the best levers
split by head size.

Levers swept and outcomes (all checked against a reference first):

| lever | outcome |
| --- | --- |
| Vectorized accumulator rescale (one vector multiply, not per-element) | kept |
| Transposed V staging in LDS | regression; the transpose scatter costs more than the gather it replaces; reverted |
| M-amplification (one wave owns several Q tiles) | regression; a smaller grid cuts occupancy on a latency-bound kernel; reverted |
| K-fragment fusion (`fuse_k`): load each K fragment inside the QK matmul | wins at D128 by relieving spills; hurts at D64 where there is nothing to relieve; auto-enabled for `head_size >= 128` |
| BLOCK_N widening (`fmha_blockn.py`) | regression at every width; spills grow and the extra V gathers and P-transpose reads outweigh the loop overhead saved |
| Software pipelining (`fmha_pipelined.py`, CK Tile `qr_ks_vs`): hoist the next tile's QK and carry the score through iter-args | wins at D64 where registers have headroom; regresses at D128 where the carried state spills; kept for D64 |
| `ds_bpermute` register P-transpose | structural dead end; the same LDS engine with more instructions |
| Lever recombination (`combo.py` cartesian sweep) | no combination beat the best single lever beyond measurement noise |
| Multi-wave cooperative K/V staging (`fmha_multiwave.py`) | correct but slower; the kernel is not memory-bound here |
| Register-blocked multi-wave rewrite (`fmha_regblocked.py`) | correct in every mode but a consistent regression against the single-wave kernels |

Two small fusions were kept because they cut static instruction count with
bit-identical output (instruction count is the currency on an issue-bound
kernel): a vectorized P-transpose read, and hoisting the epilogue reciprocal out
of the per-column loop. Both were within run-to-run noise on wall clock, as the
issue-bound model predicts.

A different WMMA intrinsic does not help on this silicon: only the 16x16x16
f16/bf16 atom exists, with an f32-only accumulator.

## Part 3: shape survey and the causal early exit

A survey of the kept winners across a matrix of shapes showed uniformly
issue-bound behavior. It also surfaced the campaign's largest win: tuning on one
shape had hidden that causal masking ran at roughly half the efficiency of
non-causal. Clamping the K loop to skip fully masked tiles recovered that, and it
was algorithmic rather than microarchitectural. The production kernel applies the
same idea through `_window_tiles` (see "Dense score features" above).

## Iteration ledger

| kernel | levers swept | outcome |
| --- | --- | --- |
| `fmha_singlewave` | `bm_tiles`, `p_mode`, `v_mode`, `q_preload`, `fuse_k`, vectorized rescale and P read, epilogue hoist | D128 winner (`fuse_k` auto on D128) |
| `fmha_pipelined` | software pipelining, `sched` hints, `p_xpose` | D64 winner |
| `fmha_blockn` | `bn_tiles` | regression (spills) |
| `fmha_multiwave` / `fmha_regblocked` | `n_waves`, then `num_warps x m_repeat x block_n` | regression |

## gfx1151 nuances (why this part behaves differently)

gfx1151 is an APU-class part, and most surprising results trace back to that:

- **Unified memory and a large last-level cache.** K/V for one `(head, batch)`
  tile is small and reused across the wave, so a column-strided global gather
  stays cache-resident. LDS staging replaces cheap cached reads with serialized
  `ds_load`s and barriers and tends to lose. On a discrete HBM part the same
  staging is mandatory.
- **wave32 and one WMMA atom.** The only matrix intrinsic is the 16x16x16 f16
  (or bf16) atom with an `<8 x f32>` accumulator. There is no larger-K variant,
  no `ds_read_tr`, and no async global-to-LDS copy.
- **VGPR cap with no slack at D128.** See "Why it is stuck" above.
- **Run-to-run drift.** Long build sweeps heat-soak the machine. Every
  keep/revert decision was made A/B back-to-back in one thermal window, never
  across sweeps, and several apparent small wins disappeared on repeat.

## Algorithmic differences from a gfx950 (CDNA) MFMA FMHA

The attention math (online softmax, causal early exit, the `qr_ks_vs` pipeline)
is identical, but the kernel structure inverts on several axes:

| axis | gfx950 (CDNA, MFMA) | gfx1151 (RDNA3.5, WMMA) |
| --- | --- | --- |
| wavefront | wave64 | wave32; the softmax row-reduction butterfly spans 16 lanes |
| matrix unit | `v_mfma_*`, many tile shapes | one 16x16x16 WMMA; low matrix-to-overhead ratio |
| operand semantics | native K accumulation, large K tiles | `A . B^T`, forcing the V column-gather and transpose dance |
| C fragment | can output packed f16 | always `<8 x f32>` |
| bottleneck | HBM bandwidth; LDS staging, double buffering, and async copy are correct | cache-resident and issue-bound; LDS staging is a net loss |
| transpose | `ds_read_tr`, async global-to-LDS | neither exists; P to A round-trips LDS or `ds_bpermute` |
| register file | large; deep pipelines fit | limited; even one-stage pipelining spills at D128 |

Porting the CDNA LDS-staging design verbatim (`fmha_multiwave`,
`fmha_regblocked`) reproduces its dataflow faithfully and loses, because the
bottleneck it relieves is not this part's bottleneck.

## Lessons ported from CK Tile

- **`qr_ks_vs` software pipelining** ports cleanly (`fmha_pipelined`) and helps
  where registers have headroom. A pipeline depth that is free on a large
  register file costs spills on a small one.
- **`PermuteWarpGemmCToA`** (register C-to-A transpose via `permlanex16` and
  `v_perm`) cannot port: CK Tile co-designs the consuming WMMA's operand
  distribution so a 2-lane permute suffices, while the DSL's `mma` has a fixed A
  map needing a 16-lane gather. A register-shuffle transpose needs the consuming
  matmul's layout to be customizable too.
- **Cooperative LDS K/V staging plus register blocking** (`MRepeat`/`NRepeat`):
  register blocking is a genuine density lever and works as designed; the LDS
  staging is a bandwidth lever and inverts here. Separate a porting source's
  levers by which bottleneck each attacks, and re-test each against the target
  part's actual bottleneck instead of adopting the bundle.
- **Online softmax plus causal early exit** is the one unambiguous shared win,
  because it skips masked work rather than tuning microarchitecture.

## Generalizable lessons

- Name the roofline before optimizing.
- Count the matmul fraction. If WMMA is a sliver of issued instructions you are
  issue-bound, and chasing the FLOP roofline is futile.
- Re-test rejected levers when the bottleneck model changes, but believe the
  measurement over the model.
- Spills are visible without a profiler: the AMDGPU msgpack note carries
  `.vgpr_count` and `.vgpr_spill_count`; `tune._resource_counts` decodes them
  from the HSACO.
- Survey before micro-optimizing. One-shape tuning hid the causal inefficiency
  until a shape matrix put causal and non-causal side by side.
- Fix the measurement before trusting a delta: compare A/B back-to-back in one
  thermal window, and gate every variant on correctness first.

## Reusing and extending the CK Tile helper layer for RDNA WMMA

The kernels are built on `rocke.helpers`: `make_global_view` and
`make_tile_window` (with `shift_by` for the K-loop step) for Q/K/V/O addressing,
`make_lds_view` and `TileWindow` for the P-transpose round trip, the
`helpers.attention` softmax primitives, and a WMMA atom for the QK/PV matmuls.

The helper layer was built for CDNA/MFMA/wave64, so these additive pieces were
added for RDNA3.5 WMMA/wave32 without changing any MFMA or f32 path:

| gap | addition | where |
|---|---|---|
| only `MfmaAtom` (wave64) existed | `WmmaAtom` (wave32, m=n=k=16, 16 operand elements and 8 accumulator elements per lane); lane-layout accessors delegate to the existing `target.mma` layout maps | `helpers/atoms.py` |
| `WarpGrid` was MFMA/wave64-only | a wave32 WMMA path accepting the WMMA atom | `helpers/geometry.py` |
| `load_tile` casts every element to f32, so its result is not directly `b.mma`-able | `load_wmma_fragment` / `store_wmma_acc`: packed fragment load/store built on `TileWindow.load_vec` | `helpers/distribution.py`, `helpers/tensor_view.py` |
| `StaticDistributedTensor` is a per-element f32 list | `WmmaTensor`: a packed distributed tensor carrying one lane's fragment or accumulator as a single SSA vector, with `load_wmma_tile` / `wmma_mma` / `store_wmma_tile` wrappers | `helpers/distribution.py` |

The WMMA operand load must be one packed vector load feeding `b.mma`, and the
accumulator must stay a packed vector (the rescale is one multiply, not eight).
Routing through the f32 `load_tile` or a per-element container inserts casts and
repacks that silently regress an issue-bound kernel.

## Files

- `ALGORITHM.md`: a from-the-math-up guide to the kernel for readers new to flash
  attention.
- `bench_v_staging.py`: Part 1 A/B harness for V staging on the production spec;
  gates each variant on a numpy reference, times with HIP events, and
  disassembles for the memory-instruction mix.
- `fmha_singlewave.py` / `tune.py`: the Part 2 vehicle (`SingleWaveCfg`:
  `bm_tiles`, `p_mode`, `v_mode`, `prefetch_k`, `q_preload`, `fuse_k`) and its
  driver, which verifies against numpy, times, and reports instruction mix and
  VGPR/SGPR/spill/LDS resources.
- `fmha_multiwave.py` / `mw_tune.py`: the multi-wave structural lever.
- `fmha_regblocked.py` / `prod_tune.py`: the register-blocked multi-wave rewrite
  (`RegBlockedCfg`: `num_warps`, `m_repeat`, `block_n`).
- `fmha_blockn.py` / `bn_tune.py`: the BLOCK_N-widened single-wave kernel
  (`BlockNCfg.bn_tiles`).
- `fmha_pipelined.py` / `sp_tune.py`: the software-pipelined single-wave kernel
  (`PipelinedCfg`) and its driver (`--sched`, `--fusek`, `--pxpose`).
- `combo.py`: the cartesian-product sweep across the single-wave levers.
- `survey.py`: the shape survey, which runs the best single-wave kernels per
  shape and compares against PyTorch SDPA.
- `wmma_fmha_fwd_bench.py`, `wmma_fmha_fwd_verify.py`,
  `wmma_fmha_fwd_sweep_profile.py`: benchmark, verification, and sweep-profile
  drivers for the production kernel.
