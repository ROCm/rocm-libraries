# RDNA (gfx11 / gfx12) WMMA SDPA forward

The rocKE SDPA forward for RDNA3/3.5 (gfx11) and RDNA4 (gfx12) is two
runtime-shaped kernels, each compiled once per RDNA generation for its LLVM
generic processor:

- **Prefill**: `kernels/gfx1151/wmma_fmha_fwd_v2.py` (`WmmaFmhaFwdV2Spec`,
  body in `rocke/helpers/rdna_attention.py`).
- **Decode**: `kernels/gfx1151/wmma_fmha_decode.py` (`WmmaFmhaDecodeSpec`), a
  split-KV, GQA-packed kernel for small query lengths, plus its split-merge
  kernel.

Both have byte-identical C mirrors (`instance_gfx1151_wmma_fmha_fwd_v2.h`,
`instance_gfx1151_wmma_fmha_decode.h`) gated by the parity pairs
`gfx1151_wmma_fmha_fwd_v2_emit` and `gfx1151_wmma_fmha_decode_emit`. They
replaced the earlier single-wave `wmma_fmha_fwd` kernel and its gfx11-only
transposed-QK body (`wmma_swapqk`), which have been removed.

## Inference coverage benchmark

The current inference harness is
[`benchmarks/gfx1151/attention`](../../../benchmarks/gfx1151/attention/).
It runs a fixed, seeded 54-case matrix covering FP16/BF16, head dimensions
64/128/256, MHA/GQA/MQA, causal alignment, sequence tails, decode, paged and
ragged inputs, sliding windows, softcap, sinks, ALiBi, FP32 QQ-bias, FP8 KV
storage, and feature combinations. Dedicated dispatcher regressions additionally
cover D96, BHSD with independently padded strides, two-sided noncausal windows,
and optional FP32 LSE output. This is ordinary SDPA coverage, not an
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

Run on any gfx11 (RDNA3/3.5) or gfx12 (RDNA4) device with the platform Python
package and library on `PYTHONPATH`. The benchmark compiles for the device's
generic target by default; `--arch gfx1151` builds processor-specific objects
on a gfx1151 device:

```bash
python -m benchmarks.gfx1151.attention.benchmark_sdpa [--arch gfx1151]
```

## Prefill kernel (`wmma_fmha_fwd_v2`)

- **CTA.** A multi-wave workgroup owns `block_m` query rows (128, or 64 when a
  wave pair splits the head) of one (batch, query head). The key loop walks
  32-key tiles (`block_n`); K and V tiles are loaded once into padded LDS
  (`k_pad` / `v_pad`) and shared by every wave.
- **Transposed scores.** The waves compute `S^T = K Q^T`, keeping a query row's
  softmax state within a lane, and `O^T = V^T P`.
- **Lazy rescale.** The O/l rescale is skipped while no lane's running maximum
  grows by more than 8 (log2 domain).
- **Masked work.** The key loop runs only over the tiles a query tile can see
  (causal diagonal and window bounds); fully visible tiles skip the mask.
  Causal launches use longest-processing-time CTA order; the default order
  (`cta_order="hqb"`) is heads, then query tile, then batch.
- **Head widths.** Q/K buckets 64, 96, 128, 192 and 256 with V/O buckets 64, 96,
  128 and 256; the actual widths are multiples of 8 inside the bucket. Buckets
  above 128 split the head across a wave pair; gfx12 keeps an unsplit Q up to
  256 while V is at most 128 (MLA).
- **Runtime arguments.** Batch, Sq, Sk, Hq, Hkv, Hv, head widths, every
  batch/token/head stride (BSHD or BHSD; token strides multiples of 8),
  top-left or bottom-right causal alignment, two-sided window bounds, LSE
  selection and the bias broadcast strides.
- **Baked variants.** dtype (FP16/BF16), bucket, mask class (`none`, or
  `causal`, whose runtime bounds also express no mask and windows), additive
  bias (none, FP32, Q dtype), and the optional score features softcap, sinks,
  ALiBi, QQ-bias, ragged/paged layouts and OCP E4M3 K/V storage.

## Decode kernel (`wmma_fmha_decode`)

- **GQA packing.** The query heads that share a K/V head and their query
  positions are packed into 16-row WMMA tiles, so each K/V head is read once per
  row tile rather than once per query head.
- **Split-KV.** A runtime `num_splits` (chosen on the host from the shape and
  the CU count) cuts the visible key range; splits are interleaved so all splits
  of a head sweep the keys as one DRAM stream. Partial O and LSE go to a
  workspace and are merged by the last split to finish (fused merge, up to 4
  splits, with a self-re-arming arrival counter) or by the separate merge kernel.
  `build_wmma_fmha_decode_module` ships both kernels in one code object.
- **Per-family codegen.** gfx12 runs the waves of a workgroup in lockstep (one
  barrier per key iteration, which keeps cold-cache reads in one stream); gfx11
  does not. D256 splits the O columns over 2 (gfx11) or 4 (gfx12) workgroups.
- **Buckets.** 64, 128 and 256, with the same runtime shapes, masks, bias kinds
  and score features as the prefill.

## Selection

Routes, priorities, the decode/prefill rule (`rdna_attention_use_decode`),
head-width buckets, the bindings and the shipped hipDNN catalog
(`hipkernel:Gfx1151WmmaAttention`) are documented in the dispatcher's
[RDNA WMMA inference](../../../dispatch/attention/README.md#rdna-wmma-inference-gfx11-and-gfx12)
section.

`dispatch_attention(request).bind_torch(tensors, **scalars)` accepts caller-owned
`q`, `k`, `v`, `out` and the selected metadata/auxiliary tensors. Despite the
historical method name, any real device-tensor owner implementing shape, dtype,
element strides, device, and `data_ptr()` can bind. The adapter validates rank,
shape agreement, vector alignment, metadata contiguity, and non-overlapping
outer strides without reading GPU sequence contents; the batch and every stride
are kernel arguments, so one launch covers the whole batch.

The compiled launcher is cached by target and complete spec. Repeated
`binding.launch(stream=...)` calls reuse it. The default fence synchronizes only
that stream; `fence=False` permits asynchronous and graph-captured launches.
After external stream synchronization, call
`rocke.runtime.launcher.release_retained_for_stream(stream)` to release retained
tensor owners. Captured launch owners must remain alive until their graphs
are destroyed. The benchmark drains them at that boundary, outside timing.

## RDNA generic targets

One kernel set serves each RDNA generation; no per-processor kernels are
authored or shipped:

| Generation | Generic target | Devices | WMMA atom |
|---|---|---|---|
| RDNA3/3.5 | `gfx11-generic` | gfx1100, gfx1101, gfx1102, gfx1103, gfx1150, gfx1151, gfx1152, gfx1153, gfx1154 | `wmma_f32_16x16x16_{f16,bf16}` (operands duplicated across lane halves) |
| RDNA4 | `gfx12-generic` | gfx1200, gfx1201 | `wmma_gfx12_f32_16x16x16_{f16,bf16}` (K split across lane halves) |

rocKE IR names no processor: the LLVM module carries only the
`amdgcn-amd-amdhsa` triple, and the comgr ISA name chooses the processor.
`rocke.core.arch.generic_arch_from_target_id()` maps a device name to its
generic target; tests, the benchmark and hip-kernel-provider packaging use it.
The dispatcher's CU count is twice HIP's `multiProcessorCount` (HIP reports
WGPs on RDNA) of the live device when it can run the object, else 40.

Toolchain and runtime requirements:

- Compile: an LLVM 20 or newer comgr, the oldest flavor rocKE supports.
  Generic processors need code object v6, which the validated LLVM 22 and 23
  comgrs emit (generic version 1).
- Load: the HIP runtime must accept generic code objects. ROCm 7.0.2, 7.2.0 and
  the 7.13 and 10.0 core builds load them; ROCm 6.4.0 on a gfx1151 rejected
  them (`hipErrorNoBinaryForGpu`). Treat ROCm 7.0 as the minimum.
- Hardware coverage: the GPU tests have run on gfx1100, gfx1101, gfx1150 and
  gfx1201. gfx1151 and gfx1200 load the same generic objects but have not been
  run.

## Historical gfx1151 optimization campaign

The sections below are a historical campaign, not results for this benchmark.

The sections below record the design lessons of an earlier optimization campaign
on gfx1151 (RDNA3.5, wave32, a single 16x16x16 WMMA atom). They are qualitative.
They are not results for the kernels above, and this document deliberately
carries no absolute performance figures. Part 1 ran on the removed production
kernel and can no longer be replayed; re-measure Parts 2 and 3 with the vehicles
listed under "Files" before acting on any ordering below.
[`ALGORITHM.md`](ALGORITHM.md) derives the kernel from the math and is the place
to start if flash attention is new to you.

The study applies the
[optimization runbook](../../../../platform/dsl_docs/optimization/optimization_runbook.md)
to the single-wave gfx1151 WMMA flash-attention forward kernel that preceded
the current kernels (since removed). Every variant was correctness-gated
against a reference before it was timed.

- **Part 1** applies the runbook loop to one lever (V-LDS staging). The
  "optimization" regressed and was reverted.
- **Part 2** sweeps many levers across several kernel bodies (`fmha_singlewave`,
  `fmha_pipelined`, `fmha_blockn`, `fmha_multiwave`, `fmha_regblocked`) and
  diagnoses why the kernel is issue- and register-bound rather than FLOP-bound.
- **Part 3** surveys the kept winners across a shape matrix and records the
  campaign's one large algorithmic win, the causal early exit.

## The kernel under test

In the campaign's kernel, one wave32 owns 16 Q rows for a `(q_tile, head, batch)`. Per K-tile it runs
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

**One lever.** The removed production kernel's `v_lds_stage` knob: `False` is
the per-element global gather; `True` vector-loads each k-row into an LDS tile,
shares the existing P-staging barrier, and reads the B operand from LDS.

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

**Decision.** Reverted: the knob stayed off by default and was enabled only
for profiles where a fresh measurement favored it. The knob and its driver
were removed with that kernel.

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
was algorithmic rather than microarchitectural. The current prefill and decode
kernels apply the same idea: their key loops visit only the tiles a query tile
can see.

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

The campaign's research vehicles are self-contained FP16 gfx1151 kernels with
their own drivers; nothing in production imports them. Their drivers replay the
Part 2 and Part 3 experiments on a gfx11 device.

- `ALGORITHM.md`: a from-the-math-up guide to the campaign's single-wave kernel
  for readers new to flash attention.
- `fmha_singlewave.py` / `tune.py`: the Part 2 vehicle (`SingleWaveCfg`:
  `bm_tiles`, `p_mode`, `v_mode`, `prefetch_k`, `q_preload`, `fuse_k`) and its
  driver, which verifies against numpy, times, and reports instruction mix and
  VGPR/SGPR/spill/LDS resources. `tune.py` also holds the numpy reference and
  `llvm-objdump` lookup the other drivers share.
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

The production kernels' tests are `library/tests/test_wmma_fmha_fwd_v2_*.py` and
`library/tests/test_wmma_fmha_decode_*.py`; their inference benchmark is
[`benchmarks/gfx1151/attention`](../../../benchmarks/gfx1151/attention/).
