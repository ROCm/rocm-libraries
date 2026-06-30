# CK DSL Optimization Runbook

This runbook is a long-form checklist for optimizing GPU kernels written
with `rocke`.

Every section ties the general optimization concept to a concrete
`rocke` primitive, helper, instance, or probe. The goal is that an
engineer with the codebase open can find the lever, change it, verify
it, measure it, and explain it — without leaving this tree.

Use it as a menu of considerations. Do not apply every item blindly.
Start from the operation contract and the measured bottleneck.

## Step 0 — Exhaust the existing lever space first (do this BEFORE touching the algorithm or structure)

**This is mandatory and comes before everything below.** Before you change the
kernel body, the tiling strategy, or the algorithm — before you conclude a gap
is "structural" — you must first prove the *current* implementation cannot
already hit the target with a different configuration. The cheapest way to be
wrong is to redesign a kernel that only needed a different knob.

For the specific shape you are optimizing:

1. **Enumerate EVERY implemented lever.** Walk
   [§12.1 the Knob Catalog](#121-knob-catalog-master-list) *and* the instance's spec
   dataclass and list every lever that applies: algorithmic variant, tile
   geometry (`tile_size`, `block_m_per_warp`), `num_warps`, `waves_per_eu`,
   MFMA atom, pipeline / prefetch-ring depth, single- vs double-buffer flags
   (K *and* V), scheduling hints (`sched_barrier`, `iglp_opt`), LDS layout
   (conflict-free / bank-pad), transposed-softmax sub-flags, epilogue variant,
   compiler backend (`llvm` vs `hipcc`), and **every default-off opt-in flag**.
   Include the levers that are currently OFF — those are exactly the ones a
   heuristic may be mis-picking.
2. **Run an EXHAUSTIVE cartesian sweep over that space for the shape.** The
   DSL's cheap comgr compile is the entire point: enumerate the legal product
   (the spec `__post_init__` rejects illegal combos), compile in **batches**
   (threaded comgr, hundreds-to-thousands of configs), correctness-prune
   against an fp32 reference, then time the survivors against the baseline you
   must beat. The autotuner funnel ([§12 Autotune](#12-autotuning-strategy)) is built for
   exactly this.
3. **Only THEN enter the optimization loops below.** If the swept ceiling
   already meets the target you are done — and you have probably found that the
   production heuristic was mis-routing (turn the swept winner into a tuned
   knob). If the swept ceiling still falls short, you now *know* the gap is
   genuinely structural, a body/algorithm redesign is justified, and the sweep
   has told you which config to redesign *from* and against which resource
   budget.

This ordering is not optional. Real failures from skipping it: a selector
routed a shape to a ~2× slower configuration that an exhaustive sweep beat
immediately; a gap assumed "structural" turned out to be a single mis-set
`num_warps`; and — conversely — a kernel-body redesign was correctly justified
*only after* an exhaustive sweep proved no existing config could close the gap
(and the sweep pinned the exact resource budget the redesign had to hit).
Identify the levers, sweep them exhaustively, then loop.

## The Loop (one-page summary)

If you take only one thing from this runbook, take this loop:

```text
1. State the hypothesis.
2. Verify correctness baseline.
3. Measure baseline with stable timing.
4. Inspect generated IR / ISA / resources (probes in §18).
5. Change one lever (catalog in §12.1).
6. Re-verify correctness.
7. Re-measure with the same harness.
8. Explain why it moved (per-iter ISA diff).
9. Keep or revert.
10. Record the result.
```

Do not batch several levers unless you are explicitly doing a coarse
search and plan to isolate the winner afterward. If stuck and no clear winning change with simple levers is evident, extend to multiple levers in smart combinations and use judgement. Whenever possible, turn a lever into a knob to be tuned with heuristics, because several failures might combine to a big win. **If correctness fails, do not report speed as a win, make sure it is fixed.**

## How to read this document

The structure mirrors the general runbook:

1. Define the problem (contract, shapes, layouts, dtypes, boundaries).
2. Establish baselines (correctness, performance, hygiene).
3. Classify the bottleneck (arithmetic intensity, profiler, IR/ISA).
4. Choose the algorithmic mapping.
5. Decompose the work (grid, block, wave, thread).
6. Optimize the memory hierarchy (global, LDS, registers, caches).
7. Pick matrix instructions and operand layouts.
8. Pipeline and schedule.
9. Optimize the epilogue.
10. Tune the compiler.
11. Inspect ISA and resources.
12. Autotune.
13. Apply operation-specific checklists (GEMM / conv / attention / …).
14. Recognize failure modes.
15. Report results.
16. Apply decision heuristics.
17. Read the empirical case studies.

Two extra sections are DSL-specific:

- **§18 DSL Probe Workflow** — when and how to use the probes under
  `utilities/tools/dsl_probes/`.
- **§19 Reproducible Commands** — exact venv / PYTHONPATH / harness
  invocations for the production workflows.

Cross references:

- `runbook_mapping.md` — section-by-section DSL-primitive table.
- `runbook_compliance.md` — empirical pass results per section.
- `measured_results.md` — last documented validation pass numbers.

If you came here looking for a specific knob, jump to **§12.1 Knob
Catalog** — it enumerates every performance lever exposed by `rocke`,
grouped by family (algorithmic variant, tile geometry, MFMA atom,
pipeline, epilogue, LDS layout, occupancy, preshuffle, persistent /
Stream-K, quantization, attention-2D micro-levers, chiplet swizzle,
compiler flags, runtime / launch, dispatcher policy, benchmark
hygiene, static probes).

If you came here looking for **what is available on a specific
architecture** (MFMA atom catalog, LDS size / banks / conflict rules,
transpose-read and cross-lane intrinsics, VGPR / AGPR / occupancy caps,
chiplet swizzle parameters, buffer-descriptor flags, fp8 / quantization
support, and compiler caveats), see
**[§21 Target Architecture Reference](#21-target-architecture-reference)** —
the single hub that lists the per-arch files (gfx950 — the DSL's
default target — and gfx942; RDNA wave32 targets gfx1151 / gfx1201 are
also supported). The base runbook itself stays arch-neutral; the
arch reference holds the concrete facts.
- `utilities/skills/` — focused skill docs (`gemm-optimization`,
  `lds-optimization`, `kernel-trace-analysis`,
  `prefetch-data-load`, `capture-kernel-trace`, `empirical-case-studies`,
  `kernel-launch-guide`, `bisect-perf-regression`).
- `utilities/tools/dsl_probes/` — the static-inspection probes
  introduced for this runbook.

---

## 1. Define The Problem Exactly

### 1.1 Operation Contract

Decide what the kernel computes before deciding how it should run.

- Name the operation precisely. The DSL has a separate spec dataclass
  for each kernel family: `UniversalGemmSpec` for GEMM, `ConvProblem` +
  `ImplicitGemmConvSpec` for implicit-GEMM conv, `DirectConvProblem` +
  `DirectConv16cSpec` / `DirectConv4cSpec` for direct grouped conv,
  `UnifiedAttentionProblem` plus `UnifiedAttention2DSpec` / `*3DSpec` /
  `*ReduceSpec` for paged attention, `Reduce2DSpec` / `LayerNorm2DSpec` /
  `RMSNorm2DSpec` / `ElementwiseSpec` / `Transpose2DSpec` for the
  small-op family, `FmhaVarlenSpec` / `FmhaHeadGroupingSpec` /
  `FmhaPagedPrefillSpec` / `FmhaSplitKvDecodeSpec` / `FmhaFwdFp8Spec`
  / `FmhaBwdSpec` / `FmhaAppendKvSpec` for the FMHA variants, and so
  on. The spec name is the contract.
- State whether the operation is forward, backward-data, backward-
  weight, inference-only, training, deterministic, or approximate.
  Backward FMHA lives in `FmhaBwdSpec`; everything else in
  `instances/` is forward.
- State whether the op is standalone or part of a graph/fusion
  boundary. Fusion is in `helpers/fuse.py` (`compile_fn`, `explain_fn`,
  `FusedEpilogue`, `_PATTERN_TABLE`); see `dsl_docs/fusion/overview.md`.
- State whether atomics are allowed (`b.global_atomic_add_f32`,
  emitted by split-K GEMM and Stream-K).
- State all side effects: workspace (`WorkspacePool`,
  `*_workspace_bytes`), scratch buffers, stream usage, global memory
  writes, persistent state.

### 1.2 Shapes

- Record dimensions with names, not values: every problem dataclass
  uses explicit field names (`ConvProblem.N`, `Hi`, `Wi`, `C`, `K`,
  `R`, `S`, `sH`, `sW`, `pH`, `pW`, `dH`, `dW`; `UnifiedAttentionProblem.
  total_q`, `num_seqs`, `num_query_heads`, `num_kv_heads`, `head_size`,
  `block_size`, `max_seqlen_q`, `max_seqlen_k`, …).
- State which dimensions are compile-time constants (the spec fields)
  and which are runtime (manifest fields, kernel args).
- Identify pathological boundary shapes: `1, 2, 3, 15, 16, 17, 31, 32,
  33`, powers of two, non-multiples of tile sizes. Add them to the
  parity harness in `examples/`.

### 1.3 Layouts

- Layouts are explicit in the spec: `UniversalGemmSpec.layout = "RCR"`
  / `"CRR"` / etc.; conv uses NHWC by default (see
  `dsl_docs/instances/convolution.md`); attention uses (block, slot)
  paging for KV.
- For non-trivial mappings (paged KV, im2col), the coordinate
  transform DAG lives in `transforms.py`. Operators are
  `pass_through`, `pad`, `pad_dynamic`, `embed`, `unmerge`, `merge`,
  `indirect`. Walkthrough: `architecture/TRANSFORM_DAG.md`.
- State alignment and contiguity at the descriptor: `TensorDescriptor`
  in `transforms.py` and the legacy `TensorView` / `TileWindow` in
  `helpers/tensor_view.py`.

### 1.4 Dtypes And Numerics

- Record input dtype, weight dtype, accumulator dtype, output dtype,
  scale dtype, bias dtype, index dtype. The spec dataclass owns these
  fields explicitly; e.g. `UnifiedAttention2DTiledSpec.dtype="bf16"`,
  `kv_storage_dtype="fp8e4m3"` for the FP8 KV cache path.
- DSL dtype catalog: `I1`, `I8`, `I16`, `I32`, `I64`, `BF16`, `F16`,
  `F32`, `FP8E4M3`, `BF8E5M2` (see `core/ir.py:34-97`).
- Quantization helpers in `helpers/quant.py` (`quant_max_abs`,
  `quantize_scalar_f32`, `dequantize_scalar_to_f32`, `quant_ir_type`,
  `ir_to_qdtype`). `QDType` is `Literal["i8", "fp8e4m3", "bf8e5m2"]`.
- MX block-scale support in `helpers/mx_scale.py` and the spec
  `MxGemmSpec`.
- Always set a tolerance policy. The `examples/common/ck_tile_parity.py`
  harness encodes the per-op tolerances we currently believe in
  (elementwise linear ops bit-exact, silu/gelu `<= 2e-4`, layer/rms
  norm `<= 5e-3` (was 2.5e-3 before noise widening, see
  `notes/PROPOSALS_IMPLEMENTATION_REPORT.md::F2`), reduce `<= 1.5e-3`,
  gemm `<= 7e-2`).
- For fp16 inputs with fp32 accumulation over O(100) terms, expect
  errors near the fp16 ULP floor for correct kernels. Errors two
  orders of magnitude higher almost always indicate structural bugs
  (wrong lane mapping, missing acc reset, stale LDS, etc.). See
  `utilities/skills/empirical-case-studies.md` (Case Study 3) for
  specific bug signatures.

### 1.5 Boundary Conditions

- Padding via `transforms.pad`, `pad_dynamic` (returns `(offset,
  valid)` per coordinate).
- Buffer descriptors with `flags=0x00027000` (= 159744) return zero
  for out-of-bounds lanes — the canonical tail-safe primitive.
  Constructed with `b.buffer_rsrc(ptr, num_bytes)`.
- Causal masks, sliding window, softcap in `helpers/attention.py`
  (`causal_mask`, `sliding_window_mask`, `apply_softcap_log2`).
- Split-K / persistent / stream-K decisions move into `streamk_gemm.py`
  + `helpers/streamk.py` + `helpers/persistent.py`.
- Page-table indirection for paged attention via
  `transforms.indirect(...) + unmerge(...)`.

---

## 2. Establish Baselines

### 2.1 Correctness Baselines

The DSL ships several layered correctness gates. Use them in order
from cheapest to most expensive:

1. `pytest test/test_rocke.py` — 245 unit tests (IR construction,
   transform DAG, helpers, instance smoke). The IR/lowering subset needs
   no GPU; ~20 harness/validation/launch-timer tests require a GPU
   (torch+HIP).
2. `python dsl_docs/development/verify_dsl_docs.py` —
   imports every symbol, exercises every IR builder method, lowers
   every spec to LLVM/HIP/CK Tile, builds HSACO, launches small
   kernels (50 checks, ~2 s).
3. `python -m rocke.run_manifest <hsaco> <manifest>.json --verify` —
   per-kernel verification: loads HSACO, builds inputs, runs the
   kernel, compares against the in-process NumPy/torch reference.
4. `python Python/rocke/examples/common/ck_tile_parity.py --op all` — small
   ops vs torch reference (20 cases, deterministic with seed=0).
5. `python Python/rocke/examples/common/parity_extended_kernels.py --op all`
   — FMHA / Sparse / Sage / MoE / Block-scale / MX correctness.
6. `python Python/rocke/examples/gfx950/attention/parity_unified_attention.py`
   — attention parity (Triton + ref vs CK DSL paths).
7. `python Python/rocke/examples/common/hip_lowering_parity.py` — production
   LLVM lowering vs HIP-debug lowering audit across every shipped
   spec.

References do not have to be torch. The conv and GEMM bake-offs use
NumPy fp32 accumulation in `run_manifest.py`. Attention has a
deliberate per-shape `ref_paged_attn` in
`examples/gfx950/attention/parity_unified_attention.py`.

### 2.2 Performance Baselines

Use the best available vendor / library baseline:

- Conv / GEMM bake-offs: CK Tile C++ via
  `lower_kernel_to_cktile`. Currently only `UniversalGemmSpec` and
  `ImplicitGemmConvSpec` lower cleanly to CK Tile C++; the rest raise
  `NotImplementedError`. See `core/lower_cktile.py`.
- Triton: AITER ships the production `unified_attention` kernel that
  vLLM and AITER use. Path comes in via `AITER_PATH` env var.
- AITER FA / FA2 for the FMHA shapes.
- Torch eager: `examples/common/ck_tile_parity.py::_bench_torch`.
- A naive scalar `_fmha_warp_body.py` (`fmha_warp_fwd_inner_body`) is
  the **correctness oracle** for the FMHA family. Several FMHA specs
  still ship with the warp-scalar body (paged_prefill, splitkv_decode,
  bwd, varlen on bf16 pre-F1, fp8 on the QK/PV paths) — these are
  intentionally NOT perf paths. The MFMA-tiled body
  (`mfma_attention_fwd_inner_body` in `helpers/mfma_attention.py`)
  replaces the scalar one in the production path.

Hard rule: **do not** report speed without correctness. The runbook's
top-level constraint applies verbatim here.

### 2.3 Benchmark Hygiene

- Use HIP events via `runtime/launcher.py::time_launches`. It owns
  warmup, the per-launch synchronize, and the iteration loop.
- For Triton interop, prefer the direct `torch.cuda.Event` window in
  `utilities/tools/dsl_probes/probe_targeted_bench.py::time_cuda_event`
  — `time_launches` calls `synchronize()` between iterations which
  perturbs Triton's autotuner.
- Run at least 3-5 invocations and report median + spread (defined as
  `(max - min) / median * 100 %`). The DSL benchmark summary owns
  this in `benchmark/summary.py::summarize_runs`.
- Some kernels are bimodal across runs even with identical code: a
  fresh process may hit one tier, steady-state runs settle to another.
  Always run at least 3-5 invocations and record both median and
  spread.
- Cold-cache vs warm-cache effects can produce single-run drops of
  more than 2× the steady-state throughput on the first run of a
  fresh process. Discard the first run unless you are explicitly
  measuring cold start.
- For multi-kernel pipelines, use `KernelLauncher` (one HSACO loaded
  once) or `PipelineLauncher` (multi-stage). The launcher amortizes
  the HIP module load.
- For graph replay timing, the launcher's `time_launches` accepts a
  `no_fence` context manager that elides per-call sync inside the
  event-timed loop.
- Pin GPU clocks if you suspect throttling:
  `sudo rocm-smi --setperflevel high && sudo rocm-smi --setsclk 7`.

### 2.4 Metadata To Record

The manifest writer (`helpers/manifest.py::make_*_manifest` →
`write_artifact`) records most of this automatically:

- GPU model and architecture (default ISA target is `amdgcn-amd-amdhsa
  --gfx950`).
- ROCm/CUDA version, driver/runtime version.
- Library commit SHA (set via `manifest.notes`).
- Kernel source commit SHA.
- Compile options (`compile_kernel` accepts `options`; default `-O3`).
- Target architecture flag.
- Build type and `NDEBUG`.
- Optimization flags.
- Environment variables.
- Binary code object metadata via `analyze_hsaco(hsaco_path).resources`.
- `timings` dict on the artifact records `ir_build`, `ir_lower_llvm`,
  `comgr_bc`, `reloc`, `exe`, `total`.

---

## 3. First Bottleneck Classification

### 3.1 Estimate Arithmetic Intensity

- `ConvFp16Problem::metrics()` returns `tflops` and `gbps` for conv.
- `UniversalGemmSpec`, `BatchedGemmSpec`, `GroupedGemmSpec` expose
  `.flops` on the spec.
- For paged attention compute the working bytes per head per token
  from the problem dims; the manifest runner prints both `TFlops` and
  `GB/s` per launch.
- Compare to the hardware balance point. Look up the architecture's
  peak FLOP/s for the relevant dtype and peak HBM bandwidth from the
  vendor spec sheet, then compute `balance = peak_flops / peak_bw`
  in FLOP / byte. Below the balance point → memory-bound; above →
  compute-bound.

### 3.1a ROCm Profiler Metrics (Hardware Counters)

Use `rocprof` / `rocprofv3` to collect hardware performance counters.
These are critical for identifying *real* bottlenecks, not theoretical
ones.

#### Essential Profiling Setup

Create `metrics.txt`:

```text
pmc: SQ_WAVES
pmc: SQ_WAVE_CYCLES
pmc: SQ_BUSY_CU_CYCLES
pmc: SQ_INSTS_MFMA
pmc: SQ_ACTIVE_INST_MFMA
pmc: TCC_EA_RDREQ_32B
pmc: TCC_EA_RDREQ_64B
pmc: TCC_EA_WRREQ_32B
pmc: TCC_EA_WRREQ_64B
pmc: TCC_HIT
pmc: TCC_MISS
pmc: SQ_WAIT_INST_LDS
pmc: TCP_PENDING_STALL_CYCLES
pmc: LDS_BANK_CONFLICT
```

Then run profiling against either the manifest runner or one of the
single-process harnesses:

```bash
sudo rocm-smi --setperflevel high
sudo rocm-smi --setsclk 7

rocprofv3 -i metrics.txt -o output.csv --stats --kernel-trace -- \
    python -m rocke.run_manifest "$HSACO" "$MANIFEST" \
    --shape "..." --warmup 5 --iters 100
```

The `--kernel-trace` flag adds VGPRs, AGPRs, LDS bytes, occupancy
percentages directly to the CSV. Pair with
`utilities/tools/utils/profile_register_usage.py` for a higher-level
view per CK DSL config.

#### Critical Metrics and Interpretation

| Metric | Meaning | Action threshold |
|---|---|---|
| `DurationNs` | wall-clock per launch | minimize |
| `TCC_EA_*` | HBM read/write transactions | >80 % peak → bandwidth-bound |
| `TCC_HIT / TCC_MISS` | L2 hit rate | <60 % hit → enlarge tiles |
| `SQ_INSTS_MFMA / SQ_ACTIVE_INST_MFMA` | MFMA issue rate | >70 % → compute-bound |
| `SQ_BUSY_CU_CYCLES` ratio | occupancy | <0.3 → fix VGPR/LDS pressure |
| `TCP_PENDING_STALL_CYCLES` | memory latency stall | >40 % → add prefetch stages |
| `SQ_WAIT_INST_LDS` | LDS stall | >20 % → swizzle / pad |
| `LDS_BANK_CONFLICT` | LDS bank conflict count | >0 → analyze with `analyze_lds_conflicts.py` |

#### Bottleneck Decision Tree (Hardware-Driven)

```python
# After collecting rocprof metrics:
PEAK_GFLOPS = ...   # peak FLOP/s for the active dtype, from spec sheet
PEAK_BW_GBS = ...   # peak HBM bandwidth, from spec sheet

compute_util = (measured_gflops / PEAK_GFLOPS) * 100
memory_util  = (measured_bandwidth_gb_s / PEAK_BW_GBS) * 100

if occupancy < 0.5:
    bottleneck = "OCCUPANCY_BOUND"
    # Action: reduce VGPR (use lean pipeline), reduce LDS (smaller tile_k)
elif memory_stall_pct > 40:
    bottleneck = "MEMORY_LATENCY_BOUND"
    # Action: increase prefetch stages, use async LDS loads
elif memory_util > 70:
    bottleneck = "MEMORY_BANDWIDTH_BOUND"
    # Action: increase tile sizes (more data reuse)
elif compute_util > 70:
    bottleneck = "COMPUTE_BOUND"
    # Action: optimize MFMA packing (e.g., K32 folding)
elif lds_stall_pct > 20:
    bottleneck = "LDS_BOUND"
    # Action: enable XOR swizzle, add LDS padding
else:
    bottleneck = "BALANCED_OR_LAUNCH_BOUND"
```

| Symptom | Likely Cause | DSL Lever |
|---|---|---|
| Occupancy < 0.5 | VGPR or LDS too high | `pipeline="lean"`, smaller `tile_k`, set `kernel.attrs["waves_per_eu"]` |
| Memory stall > 40 % | Memory latency | `pipeline="compv4"` + `AsyncTileLoader`, more stages |
| Bandwidth > 80 % peak | Bandwidth-bound | enlarge `tile_m`, `tile_n` for reuse |
| MFMA util < 50 % | Memory bottleneck | check stall metrics first |
| LDS stall > 20 % | Bank conflicts | `LdsLayout` padding or XOR swizzle (Section 6.4a) |
| L2 hit rate < 60 % | Poor locality | enlarge tile sizes |
| Arithmetic intensity < 10 FLOP/byte | Memory-bound | enlarge blocking factor |

#### When the ATT decoder is unavailable

If `rocprof-trace-decoder` is not installed on the host, `rocprofv3
ATT` and the `kernel-trace-analysis` skill cannot run. The equivalent
diagnostic signal comes from one `rocprofv3` PMC pass with the
counter set above, plus static ISA via `llvm-objdump` (or
`probe_isa_inspect.py`), plus the kernel-stats header for VGPR / LDS
/ occupancy. This combination answers every row of the §3 bottleneck
decision tree without ATT. See **§17.4** for a worked example where
this fallback identified a single-digit `MfmaUtil`, an
above-threshold `LDSBankConflict`, and a near-zero `MemUnitStalled`,
and ruled HBM out of the gap on the first profiling pass.

### 3.1b Static Inspection First — The DSL Probe Tier

Before paying for a full rocprof run, exercise the static probes under
`utilities/tools/dsl_probes/`:

| Question | Probe | Cost |
|---|---|---|
| Will it fit at the expected occupancy? | `probe_occupancy.py` | ~0.5 s/variant (compile + readelf) |
| Are the right intrinsics emitted? | `probe_intrinsic_counts.py` | ~0.05 s/variant (lower only) |
| What is the opcode mix? | `probe_isa_inspect.py` | ~0.5 s/variant (compile + objdump) |
| Does HIP-debug agree with LLVM-direct? | `probe_lowering_compare.py` | ~10 s/variant (hipcc) |
| Best variant of a sweep? | `probe_config_sweep.py` + your `run_fn` | depends on `run_fn` |
| Best vs Triton on production shapes? | `probe_targeted_bench.py` | ~0.5 s per (shape, backend) |

These give you a static, GPU-free signal in well under a second per
variant. Use rocprof to confirm, not to discover.

### 3.2 Compute-Bound Signals

- High MFMA issue density (`SQ_ACTIVE_INST_MFMA` > 70 %).
- High VALU utilization (`probe_isa_inspect.py` shows `valu` >>
  `vmem_load`).
- Low memory stalls.
- Performance scales with MFMA count reduction.
- Performance sensitive to accumulation order / K packing.
- VGPR pressure throttles occupancy
  (`probe_occupancy.py::limited_by == "VGPR"`).
- Scheduling hints change performance.
- K dimension large enough to amortize memory.

### 3.3 Memory-Bound Signals

- High global load/store throughput.
- Performance scales with vectorization and coalescing.
- Performance improves with caching, preshuffle, prepack, or LDS reuse.
- Low MFMA utilization.
- Many scalar/vector memory instructions per MFMA
  (`probe_isa_inspect.py` shows `vmem_load` and `vmem_store`
  dominating).
- Sparse/gather/paged access dominates.
- Output epilogue is slow due to stores.

### 3.4 Synchronization-Bound Signals

- Many barriers in the disassembly. `probe_isa_inspect.py` reports
  the `barrier` count and the `waitcnt` count plus the most common
  encoded operands.
- Barriers inside hot loops.
- Low occupancy and frequent LDS phases.
- Performance improves when fusing iterations between barriers.
- The `pipeline="compv4"` software pipeline collapses barriers
  by overlapping global → LDS DMA with compute.

### 3.5 Launch-Bound Signals

- Tiny kernel latency.
- Graph replay improves throughput.
- Batched problem with many small independent kernels.
- CPU overhead visible.
- Kernel time close to launch latency (~5-15 µs on AMD).
- Fusion or persistent kernels help.
- Use `KernelLauncher` (one HSACO, repeated launches) and / or
  `PipelineLauncher` (multi-stage on one stream).
- Use `WorkspacePool` to keep long-lived torch workspaces alive across
  launches.
- Use `time_launches(..., warmup, iters)` to amortize the per-launch
  HIP module load before measuring.

---

## 4. Algorithmic Mapping Choices

> **Hardware Resource Consumption Warning**: Different algorithmic
> approaches for the same operation can have dramatically different
> hardware resource requirements (VGPRs, SGPRs, LDS) even when they
> achieve similar compute throughput. Always profile resource use
> alongside performance — `probe_occupancy.py` is the per-variant
> answer.

### 4.1 GEMM Family

| Variant | Instance |
|---|---|
| Plain GEMM | `instances/gemm_universal.py::UniversalGemmSpec` |
| Batched GEMM | `instances/batched_gemm.py::BatchedGemmSpec` |
| Strided batched | same; layouts in the spec |
| Grouped GEMM | `instances/grouped_gemm.py::GroupedGemmSpec` |
| Persistent GEMM | `helpers/persistent.py` + `instances/streamk_gemm.py` |
| Stream-K | `instances/streamk_gemm.py::StreamKGemmSpec` |
| Split-K | `b.global_atomic_add_f32` available; spec field WIP |
| Preshuffled GEMM | `helpers/preshuffle.py::PreshuffleBSpec` |
| Block-scaled GEMM | `instances/block_scale_gemm.py::BlockScaleGemmSpec` |
| Sparse GEMM | not yet implemented |
| Quantized GEMM | block-scaled / mx GEMM + `helpers/quant.py` |
| Fused-epilogue GEMM | `instances/gemm_multi_d.py`, `gemm_multi_abd.py`, `helpers/fuse.py` |
| MFMA reference GEMM | `instances/mfma_gemm.py` |
| FlatMM (small decode) | `instances/flatmm.py` |
| Batched contraction (N-D) | `instances/batched_contraction.py` |

Consider:

- Does the shape fill MFMA tile dimensions? Use the
  `MfmaAtom` factory under `helpers/atoms.py` to verify the shape
  matches.
- Is M / N / K too small? Use `flatmm` or `mfma_gemm` instead of
  `gemm_universal` for tiny shapes.
- Is grouped scheduling better than padding to a common shape?
  `GroupedGemmSpec` handles this.
- Is split-K useful, or only adding atomic overhead?
- Is Stream-K needed for load balancing?
- Is weight preshuffle worth the preprocessing cost?
- Is data reused enough to justify LDS?
- Is A or B naturally contiguous? (drives `layout="RCR"` etc.)

### 4.2 Convolution

| Variant | Instance |
|---|---|
| Direct convolution | `instances/conv_direct_grouped.py::DirectConv16cSpec`, `DirectConv4cSpec` |
| Implicit GEMM | `instances/conv_implicit_gemm.py::ImplicitGemmConvSpec` |
| Implicit GEMM (auto-unrolled) | `instances/conv_implicit_gemm.py` with `unroll_k=True` |
| im2col + GEMM | `instances/img2col.py` materializes the im2col operand |
| Winograd | not yet implemented |
| FFT | not yet implemented |
| Depthwise / small-channel | `DirectConv4cSpec` (4-channel grouped) |

Consider:

- If `K` or `C * R * S` is tiny, implicit GEMM may be structurally
  weak.
- Direct conv can preserve spatial/channel reuse.
- Small-channel conv may need custom MFMA mapping: `4x4x4`,
  `16x16x16`, `16x16x32`. The DSL ships all three (see
  `helpers/atoms.py::MFMA_F16_ATOMS`).
- Use circular row accumulators for `R` rows.
  `conv_direct_grouped.py` documents the streaming-row 3-acc circular
  pipeline at lines 41-66.
- Stream input rows so one input row contributes to multiple output
  rows.
- Keep filter weights in registers when small.
- Stage weights in LDS only if cooperative load + register pressure
  justifies it.
- Prefer direct stores unless a proven LDS epilogue improves
  coalescing enough to pay for barriers (see Section 9.3 + Empirical
  Case Study 1).

#### 4.2a The Implicit GEMM VGPR Tax

Implicit GEMM convolution pays an unavoidable "VGPR tax" compared to
pure GEMM due to im2col coordinate computation overhead. Understanding
this helps set realistic performance expectations.

VGPR Breakdown Comparison (3×3 conv, 64×128 tile):

| Component | Pure GEMM | Implicit GEMM Conv | Δ |
|---|---|---|---|
| MFMA Accumulators | 32 | 32 | 0 |
| Address Computation | 5-8 | 15-20 | +10-12 |
| LDS Offsets | 5-8 | 8-12 | +3-4 |
| Temporary Vectors | 12 | 12 | 0 |
| Loop Indices | 3-5 | 5-8 | +2-3 |
| **TOTAL** | **57-65** | **72-84** | **+15-19** |

Why the extra VGPRs?

- Computing spatial coordinates (n, ho, wo) from a linearized output
  index.
- Computing input coordinates (hi, wi) from (ho, wo, y, x) with
  stride/dilation.
- Bounds checking for padding regions.
- Filter position tracking (y, x within R×S window).
- Group offset computation for grouped convolutions.

Practical consequences:

- Implicit GEMM convolution typically achieves **60-65 % of theoretical
  peak**.
- Pure GEMM typically achieves **75-85 % of theoretical peak**.
- The 15-20 VGPR overhead reduces occupancy: 72 VGPRs → 7 waves/CU vs
  60 VGPRs → 8 waves/CU.
- This is a **fundamental algorithmic constraint**, not a tuning
  failure.

**⚠ WARNING**: Do NOT chase VGPR reduction below the im2col floor
(~60 VGPRs for 3×3 conv with typical tile sizes). The coordinate
arithmetic is mathematically required. Attempting to reduce VGPRs
further will:

- Require smaller tile sizes (reducing reuse and performance).
- Force spilling to memory (catastrophic for performance).
- Not achieve GEMM-level efficiency regardless of optimization effort.

When to accept the VGPR tax:

- Your conv kernel is within 5 % of other mature frameworks.
- You are at 60-65 % of theoretical peak for compute-bound shapes.
- Further VGPR optimization attempts have failed or regressed.

When to consider alternatives:

- Very small spatial dimensions: direct conv may have lower overhead
  (use `DirectConv16cSpec` / `DirectConv4cSpec`).
- Large filter sizes (5×5, 7×7): Winograd or FFT may amortize transform
  cost.
- Extreme channel counts with small spatial: specialized layouts.

Invest optimization time in memory access patterns, LDS swizzling, and
tile sizes rather than trying to eliminate the fundamental VGPR
overhead of im2col addressing. Use `probe_occupancy.py` to confirm the
VGPR floor before deciding.

### 4.3 Attention

| Variant | Instance |
|---|---|
| Prefill + decode (unified, scalar oracle) | `instances/attention_unified.py` |
| Prefill + decode (2D MFMA) | `instances/attention_tiled_2d.py::UnifiedAttention2DTiledSpec` |
| Split-KV decode segment + reduce | `instances/attention_tiled_3d.py` |
| FMHA forward (MFMA, prefill) | `instances/fmha_mfma.py` |
| FMHA varlen | `instances/fmha_varlen.py` |
| FMHA head grouping (GQA / MQA) | `instances/fmha_head_grouping.py` |
| FMHA paged prefill | `instances/fmha_paged_prefill.py` (warp-scalar inner) |
| FMHA splitkv decode | `instances/fmha_splitkv_decode.py` (warp-scalar inner) |
| FMHA fp8 forward | `instances/fmha_fwd_fp8.py` |
| FMHA backward | `instances/fmha_bwd.py` (warp-scalar inner) |
| FMHA append KV | `instances/fmha_appendkv.py` (DMA) |
| Sage attention (per-block scaled) | `instances/sage_attention.py` |
| Block-sparse attention (Jenga / VSA) | `instances/sparse_attention.py` |

Consider:

- Q / K / V head dimension alignment to MFMA. Use
  `helpers/attention.py::mfma_16x16x16_for_dtype` to pick a valid
  atom.
- Numerically stable max/sum accumulation via
  `helpers/attention.py::OnlineSoftmaxState` +
  `warp_xor_reduce_max`/`_sum`.
- Page-table overhead — choose between
  `attention_tiled_2d` and `attention_tiled_3d`. The
  `helpers/attention.py::select_2d_config` / `select_3d_config` and
  `use_2d_kernel` heuristics decide automatically; you can override
  with `_select_2d_num_warps`/`_select_2d_tile_size` monkey-patches
  for sweeps (see `utilities/tools/dsl_probes/probe_config_sweep.py`).
- Coalescing across heads / tokens / pages.
- Shared memory capacity for Q / K / V tiles.
- Register pressure from accumulators and softmax state. The 2D path
  carries Q in LDS once and async DMAs K/V (see Section 6.3).
- Store path for output and log-sum-exp.

### 4.4 Reductions And Normalization

| Variant | Instance |
|---|---|
| Row reduce (sum/max/min/mean/prod) | `instances/reduce.py::Reduce2DSpec` |
| LayerNorm forward | `instances/layernorm2d.py::LayerNorm2DSpec` |
| RMSNorm forward | `instances/rmsnorm2d.py::RMSNorm2DSpec` |
| Add + RMSNorm + rdquant (fused) | `instances/add_rmsnorm2d_rdquant.py` |
| Smoothquant | `instances/smoothquant.py`, `moe_smoothquant.py` |
| Topk softmax | `instances/topk_softmax.py` |
| Pooling | `instances/pooling.py::Pooling2DSpec` |

Consider:

- Reduction axis length.
- Number of rows.
- Whether one block per row is enough.
- Whether multiple CTAs per row require atomics or a second pass.
- Vector width (`sweep_row_chunks` accepts `vec`).
- Numerically stable accumulation (`helpers/reduction.py::block_lds_reduce`
  has `sum / max / min / prod`).
- Fusing scale / bias / activation via the
  `FusedEpilogue` system or directly in the kernel.

### 4.5 Transpose, Permute, Copy

| Variant | Instance |
|---|---|
| 2D transpose (LDS-staged) | `instances/transpose.py::Transpose2DSpec` |
| Batched 2D transpose | `instances/batched_transpose.py` |
| In-register sub-tile transpose | `instances/transpose_bc.py` |
| N-D permutation | `instances/permute_nd.py::PermuteSpec` |
| im2col copy | `instances/img2col.py` |

- Use vectorized global loads/stores.
- Use LDS tile for non-coalesced stores.
- Avoid bank conflicts with padding or swizzle (Section 6.4a).
- Use rectangular tiles tuned to the aspect ratio.
- Avoid over-general rank logic in hot loops; `permute_nd.py` is
  one-thread-per-element with rank-N index decompose for clarity, not
  for peak throughput.

### 4.6 Gather, Scatter, Paged, Sparse

- Index load overhead may dominate.
- Coalesce metadata loads.
- Cache page tables.
- Batch by locality.
- Avoid divergent branches.
- Use buffer descriptors with proper bounds (`buffer_rsrc`).
- Precompute or compact index maps.
- Separate dense fast path from sparse fallback.
- For MoE, see `instances/fused_moe.py`, `moe_sorting.py`,
  `moe_gemm_fused.py`, `fused_moe_e2e.py`.

---

## 5. Work Decomposition

### 5.1 Grid Mapping

- Map large independent dimensions to grid axes.
- Fold small batch / head / group dimensions into the x-axis if it
  improves scheduling. Per-instance `*_grid` helpers own this
  (e.g. `gemm_universal.gemm_universal_grid`).
- Avoid putting too much work only in grid-z if it reduces scheduler
  interleaving.
- Balance blocks across CUs. The MI350X chiplet swizzle in
  `helpers/grid.py::chiplet_transform_chunked` (with
  `NUM_XCDS_MI300X/MI325X/MI350X == 8`) can help.
- Ensure enough CTAs for occupancy.
- Avoid too many tiny CTAs with high launch overhead.
- Avoid too few huge CTAs that underfill the GPU.
- Consider persistent CTAs for irregular work
  (`helpers/persistent.py::persistent_tile_loop`).

### 5.2 Block Mapping

- Choose threads per block to match wave count and occupancy.
- Common wave64 choices: 64, 128, 256, 512, 1024.
- Block size is encoded in the spec (`tile_m`, `tile_n`, `tile_k`,
  `warp_m`, `warp_n`, etc.) and verified at lowering by
  `kernel.attrs["max_workgroup_size"]`.
- More waves per block can improve data sharing but reduce occupancy.
- Fewer waves per block can improve occupancy but reduce reuse.
- Sweep wave count via `probe_config_sweep.py` with
  `num_warps ∈ {1, 2, 4, 8}` overrides.
- Sweep groups / heads per block.
- Sweep spatial columns per block.
- Sweep rows per block (`block_m_per_warp` in the 2D attention spec).
- Keep hot-loop per-block work high enough to amortize barriers.

### 5.3 Wave Mapping

- Decide what each wave owns. The `helpers/geometry.py::WarpGrid`
  type packs tile + warp grid + bound `tid / lane / warp_* / block_*_off`
  SSA into one immutable view.
- Map lanes to contiguous memory where possible.
- Map lanes to MFMA operand layout exactly. The
  `MfmaAtom.lane_to_output(b, lane, i)` helper documents the lane
  mapping per atom shape.
- Avoid lane mapping that forces many shuffles.
- Avoid wasted lanes on small dimensions.
- Consider multiple independent small problems per wave.
- For `4x4x4` MFMA, use lane batching.
- For `16x16x16` MFMA, align `lane % 16` and `lane / 16` to matrix
  dimensions.
- For `16x16x32` MFMA, understand K-lane packing before using it.

For `mfma_f32_16x16x32_f16` on AMD CDNA, lane `(c4 = lane / 16)`
holds K elements `[c4*8 : c4*8 + 8]`, not a flat concatenation of two
4-element halves. A common bug when folding two filter columns
`S=0, S=1` into K=32 is to pack `[S=0 ch 0..3, S=1 ch 0..3]` per lane.
The correct mapping is `[S=0 ch 0..7]` for `c4=0`, `[S=0 ch 8..15]`
for `c4=1`, `[S=1 ch 0..7]` for `c4=2`, `[S=1 ch 8..15]` for `c4=3`.
The wrong packing compiles, runs, and validates within `1e-2` but
fails at `1e-3` (`max_abs ~ 5e-3`, ~10 % of elements bad). Always
validate K-pack lane mapping at `1e-3`, not just at `1e-2`.

### 5.4 Thread Mapping

- Assign vectorized global loads to contiguous lanes
  (`helpers/loads.py::lane_contiguous_descriptor`).
- Assign stores to contiguous lanes.
- Separate compute lanes from store lanes only if it improves
  coalescing.
- Use inactive lanes deliberately, not accidentally.
- Avoid dynamic modulo/divide in hot loops when compile-time
  decomposition is possible. The DSL's `IRBuilder.static_for` /
  `unroll` / `static_if` are Python-time-only.
- Precompute per-thread offsets outside loops.

---

## 6. Memory Hierarchy

### 6.1 Global Memory Loads

- Coalesce loads. `helpers/loads.py::CoalescedTileLoader` owns the
  sync DRAM → VGPR → LDS pattern.
- Use 16 B or larger vector loads when aligned.
  `b.buffer_load_vN_f16`, `b.global_load_vN_f16` for N ∈ {1, 2, 4}
  dwords.
- Use buffer loads on AMD for robust bounds behavior:
  `b.buffer_rsrc(ptr, num_bytes)` constructs a buffer-resource
  descriptor with `flags=0x00027000`; OOB lanes return zero.
- Use raw pointer loads only when alias / alignment is clear.
- Avoid scalar loads where vector load is possible.
- Hoist invariant loads out of loops.
- Reuse loaded values across multiple outputs.
- Use read-only cache hints if available
  (`AUX = CACHE_ALL / CACHE_GLOBAL / CACHE_STREAM / NON_TEMPORAL`).
- Avoid replay from misalignment.
- Handle tails with masked loads or padded descriptors
  (`transforms.pad`/`pad_dynamic`).

### 6.2 Global Memory Stores

- Store contiguous vectors.
- Avoid per-element scalar stores in epilogues.
- Convert accumulator vectors to packed output vectors
  (`b.vec_trunc_f32_to_f16`).
- Use direct vector stores if already coalesced
  (`helpers/epilogues.py::DirectEpilogue`).
- Use LDS epilogue only when it meaningfully improves global
  coalescing (`helpers/epilogues.py::CShuffleEpilogue`). See Section
  9 for the trade-off.
- Avoid store barriers unless required for cross-thread staging.
- Ensure store validity checks do not introduce excessive branches.
- Check whether stores are 8 B, 16 B, or scalar in ISA via
  `probe_isa_inspect.py` (`buffer_store_dwordx2` / `_x4` vs
  `buffer_store_short`).

Vectorizing the epilogue is often the single largest optimization
for kernels that already have a good main loop. Replacing four scalar
`buffer_store` of `fp16` with one `fp16x8` `buffer_store` per lane
has been measured to roughly double the throughput on direct-conv
kernels — with no other change. Always inspect the epilogue ISA
(`probe_isa_inspect.py`) before tuning the main loop; the `vmem_store`
bucket reveals scalar-store kernels immediately.

### 6.3 LDS / Shared Memory

> Arch-specific (LDS size / banks / conflict strategy, transpose-read
> availability): see the
> [architecture reference](#21-target-architecture-reference) (§21.2).

- Use LDS to share input or weights across waves / threads.
- Avoid LDS if data is not reused enough.
- Use double buffering when global load latency matters
  (`UniversalGemmSpec.pipeline = "compv4"`).
- Use direct global-to-LDS async copies when available.
  `helpers/loads.py::AsyncTileLoader` wraps the
  `raw_ptr_buffer_load_lds_addr` intrinsic with per-wave LDS base.
- Avoid register-intermediate staging if direct DMA is possible.
- Swizzle LDS layout to reduce bank conflicts
  (`helpers/layouts.py::LdsLayout`, `lds_k_pad`).
- Add padding to avoid bank conflicts.
- Match LDS write layout to LDS read layout.
- Keep LDS footprint under occupancy thresholds (verify with
  `probe_occupancy.py`).
- Reuse LDS regions after phases when lifetimes do not overlap.
- Use one shared pool for multiple phases when safe.

Switching from `buffer_load → register → vector.store LDS` to direct
DRAM-to-LDS via the AMD `raw_ptr_buffer_load_lds` /
`amd_async_buffer_load` intrinsic is worth measuring even when
register staging looks fine. A direct-conv kernel was measured to
roughly double throughput from this change alone, before any layout
modification.

Async DRAM-to-LDS instructions on AMD generally write lane-contiguous
LDS addresses (lane `i` writes at `lds_ptr + i * size`). Arbitrary
per-lane swizzled destination pointers may compile but produce wrong
output. We tested this directly: passing a per-lane swizzled LDS
pointer to `raw_ptr_buffer_load_lds` corrupted the result, so swizzle
has to be expressed in the address arithmetic of a lane-contiguous
distribution, the way CK's tile descriptors do, not by handing the
intrinsic an arbitrary LDS pointer.

`AsyncTileLoader.choose_dwords` selects `{4, 3, 1}` only (the AMDGPU
`raw_ptr_buffer_load_lds` intrinsic on this target does not accept 2
dwords).

Staging weights through LDS is not always a win. CK does it because
its tile distribution and weight read pattern fully amortize the
cost. A naive port to an MFMA kernel that already keeps weights in
registers has been measured to *regress* throughput. Test the LDS
weight path with the actual read distribution before adopting it.

Diagnostic signature: a kernel that allocates LDS for an intermediate
tensor (P in attention, A or B in GEMM) which it immediately reads
back will show `ds_write_b<N> / outer-iter ≈ tile_bytes / threads` plus
an inner-loop `s_barrier`. If the reference implementation keeps the
same tensor in registers, that LDS allocation is structural overhead.
See **§17.4** for the P-in-LDS round-trip case: 16 KiB of LDS for the
P tile plus 64 × `ds_write_b16` per K-iter, removed by switching to
register-PV + transposed-PV reads.

### 6.4 LDS Bank Conflicts

> Arch-specific (per-opcode bank-conflict periods, read/write
> asymmetry, transpose-read availability): see the
> [architecture reference](#21-target-architecture-reference) (§21.2).

- Inspect LDS access patterns, not just total LDS bytes. Use the
  `analyze_lds_conflicts.py` tool under
  `utilities/tools/stage4_analyze/` to combine rocprof counters and
  ISA inspection.
- For AMD wave64, bank conflicts can appear with regular `(row,
  channel)` layouts.
- Try XOR swizzle on two-dimensional layouts
  (`helpers/layouts.py::LdsLayout(swizzle="xor")`).
- Try cyclic-shift swizzle when CK uses it.
- Try padding strides (`lds_k_pad` defaults `+8` sync, `0` async).
- Use transposed LDS loads where supported
  (`helpers/layouts.py::TransposeLdsReader`, `b.ds_read_tr16_b64`,
  `b.ds_read_tr_b8`).
- Count `ds_read` and `ds_write` instructions with
  `probe_isa_inspect.py`.
- Compare swizzled vs contiguous variants with
  `probe_intrinsic_counts.py`.

Cyclic-shift or XOR swizzle on a 2D tile layout only pays off in our
experiments when paired with the matching async load distribution.
With register-staged LDS writes, a manually-swizzled LDS layout has
been measured to regress slightly. Bank-conflict reduction alone is
not enough; the swizzle has to be aligned with how the load is
actually delivered to LDS.

#### 6.4a LDS Swizzle Strategy Selection: XOR vs Padding

Notation:

- **LDS_total**: Total LDS allocated by your kernel (per CU). From
  `probe_occupancy.py` or `analyze_hsaco`.
- **LDS_avail**: Available LDS per CU on the target architecture.

> Arch-specific (LDS per CU, bank count, and the resulting preferred
> swizzle): see the
> [architecture reference](#21-target-architecture-reference) (§21.2).
> The selection *method* below is arch-neutral; plug in your arch's LDS
> capacity.

Two primary approaches exist for eliminating LDS bank conflicts, with
dramatically different performance characteristics depending on GPU
architecture:

**XOR Swizzle**: Mathematically proven bank-conflict-free addressing
using bitwise XOR operations.

- Cost: higher ALU overhead (more complex address computation; 5-7
  ALU vs 1-2 ALU per LDS access).
- LDS overhead: zero — no wasted LDS bytes.
- Best for: architectures with limited LDS capacity.
- Preferred when: LDS footprint is near capacity limits and occupancy
  depends on fitting in available LDS.
- Eliminates bank conflicts completely through address computation.

**Padding Swizzle**: Simple padding to avoid conflict patterns (e.g.,
`K=64 → K_PAD=72`).

- Cost: lower ALU overhead (simple offset calculation).
- LDS overhead: small percentage of total LDS (typically negligible
  on high-capacity architectures).
- Best for: architectures with abundant LDS capacity.
- Preferred when: LDS capacity is not the bottleneck and simpler
  addressing improves throughput.
- Eliminates bank conflicts through strategic padding.

Architecture-specific selection rule:

```python
if LDS_total < 96 * 1024:
    use_xor_swizzle()      # capacity-constrained
elif LDS_avail >= 128 * 1024:
    use_padding_swizzle()  # abundant LDS
else:
    benchmark_both()       # 96 KB <= LDS_b < 128 KB
```

> Arch-specific caveat: on gfx950 the scheduling barriers
> (`s_sched_barrier` / `s_sched_group_barrier`) are compile-time fences
> that usually leave no runtime instruction in the object — so do not
> rely on a runtime `s_sched_barrier` showing up to mitigate XOR address
> overhead, and do not read its absence from the ISA as "no effect"
> (§8.4). Check the
> [architecture reference](#21-target-architecture-reference)
> (§21.3/§21.8); confirm a hint's effect by diffing the mainloop ISA,
> not by the `sched_barrier` sub-bucket count.

Critical asymmetry: conflict periods can differ between `ds_read_b128`
and `ds_write_b128` (and across arches), so reads and writes may need
separate swizzle strategies. See
`utilities/skills/empirical-case-studies.md` (Case Study 2) and the
[architecture reference](#21-target-architecture-reference) (§21.2) for
the per-opcode conflict-period table.

### 6.5 Registers

- Track VGPR and SGPR usage with `probe_occupancy.py`.
- High VGPR can reduce occupancy.
- Too-low VGPR cap can spill or reduce scheduling freedom.
- Hoist invariants but avoid keeping large temporary structures
  alive.
- Store only scalar state for tile windows / descriptors.
- Avoid persistent heavyweight objects in kernels.
- Prefer precomputed offsets over live descriptor state.
- Use compile-time constants to reduce scalar arithmetic
  (`IRBuilder.const_i32`).
- Watch register growth when adding double buffering or extra
  accumulators.
- Set `kernel.attrs["waves_per_eu"]` to override LLVM's
  occupancy hint when the compiler is wrong.

### 6.6 Caches

- Determine whether the benchmark uses warm cache or cold cache.
- Use rotating buffers if measuring bandwidth realistically.
- Use graph replay to separate cache behavior from launch overhead.
- Do not tune only for warm cache unless production has warm cache.
- Consider L2 prefetch only when the memory pattern is predictable.

---

## 7. Matrix Instructions And Compute

### 7.1 MFMA/WMMA Selection

The DSL's `MfmaAtom` catalog lives in `helpers/atoms.py::MFMA_*_ATOMS`.
**Which atoms (and which K widths) are legal depends on the target
architecture** — the wide-K and fp8 / bf8 / scaled / MX variants in
particular are not universal. Always select against your target's
MFMA-atom catalog: see the
[architecture reference](#21-target-architecture-reference) (§21.1),
which also documents the lane-layout-match property of the 32×32 atoms.

General selection guidance (then confirm against the arch catalog):

- Prefer the **widest-K atom that is legal on your target** — a wider
  K folds more contraction per MFMA (see §7.4). Do not assume a
  particular wide-K atom exists; check the arch §21.1 table first.
- `16x16x16` for standard fp16/bf16 tiles (broadly available).
- `16x16x32` for fp16/bf16 **where the arch supports it** and K
  packing is valid.
- `4x4x4` for many independent small-channel computations
  (used by `DirectConv4cSpec`).
- Scaled MFMA for fp8 / fp6 / fp4 where available.
- WMMA for RDNA / wave32 architectures (`gfx1151`, `gfx1201`): the
  `WmmaAtom` catalog in `helpers/atoms.py::WMMA_*_ATOMS` and the
  `Gfx11RdnaBackend` lowering path expose the `16x16x16` f16 / bf16 WMMA
  atoms. The base runbook stays CDNA-focused; pick the atom against your
  target's catalog.

Always confirm the emitted intrinsic with `probe_intrinsic_counts.py`.
For example, switching from the default to `mfma_f32_32x32x16_f16` is
visible immediately: the `mfma.f32.32x32x16.f16` line goes from 0 to
N, and the `mfma.f32.16x16x32.f16` line drops correspondingly.

### 7.2 Operand Layout

- Verify lane mapping with `MfmaAtom.lane_to_output(b, lane, i)`.
- Verify packed vector element order.
- Verify A/B orientation.
- Verify K packing.
- Verify output accumulator lane mapping.
- Write a tiny kernel or test if unsure.
- Compare against reference before optimizing performance.

For `mfma_f32_16x16x32_f16` on AMD CDNA, lane `(c4 = lane / 16)` holds
K elements `[c4*8 : c4*8 + 8]` (not a flat concatenation of two
4-element halves) — see Section 5.3 for the canonical bug signature.

Lane-layout matching between chained atoms matters as much as atom
shape. The `mfma_f32_16x16x32` atom's C-output lane layout does not
match its A-input lane layout, so a QK→PV chain using two 16×16 atoms
needs an LDS round-trip or `~288` cross-lane permute ops per iter to
re-pack between them. The `mfma_f32_32x32x16` atom's C-output natively
matches the A-input, so the same chain runs with zero re-pack. **§17.4**
quantifies this for unified attention: the 32×32 atom (`use_mfma_32x32`
+ `use_transposed_qk_32x32`) is structurally cheaper *because of* this
layout match, independent of the atom's compute throughput.

### 7.3 Accumulator Strategy

- Use enough accumulators to hide latency but not so many that VGPR
  pressure collapses occupancy.
- Circular accumulators can avoid shifting data.
- Reset accumulator slots at the correct lifecycle point.
- For convolution, reset phantom-row accumulators even when the
  output row is OOB. The CK direct-conv "unconditional slot reset"
  trick is non-obvious and easy to get wrong. Symptom of the bug:
  error roughly proportional to one filter coefficient times one row
  of input (we measured `max_abs ~ 1e-2`, mean ~ 1.4e-3, with ~50 %
  of elements above `1e-3`).
- For split-K, define accumulation and reduction strategy.
- For attention, keep max / sum / output accumulators consistent.
  See `helpers/attention.py::OnlineSoftmaxState`.

### 7.4 Reducing MFMA Count

- Fold small dimensions into K if operand layout supports it.
- Prefer the widest-K atom **that is legal on your target** (e.g.
  `16x16x32` over `16x16x16`, `32x32x16` over `32x32x8`) — but confirm
  the wide-K variant exists for your arch in the MFMA-atom catalog (see
  the [architecture reference](#21-target-architecture-reference)
  §21.1); do not assume it is present.
- Fuse filter positions into K for convolution when contiguous.
- Use Toeplitz-like packing for convolution only when correct and
  worth the complexity.
- Avoid K32/K64 if it changes the result beyond tolerance.
- Check whether MFMA reduction count reduction causes memory / layout
  overhead elsewhere.

---

## 8. Pipelining And Scheduling

### 8.1 Software Pipeline

- Separate prologue, steady-state loop, and epilogue. The DSL exposes
  `helpers/pipeline.py::SoftwarePipeline.run_ping_pong(...)` for the
  ping-pong staging.
- Prefetch next tile while computing current tile.
- Use ping-pong buffers (typically two LDS halves).
- Use more stages only if latency is not hidden.
- Keep pipeline state small.
- Ensure waits occur as late as safely possible.
- Avoid barrier before useful independent compute.
- Order async issues to give *all* dependent compute a chance to
  overlap. The "early-V" schedule in **§17.4** moved the V async
  copy issue from after QK to before QK so V overlaps with both QK
  and softmax, not just softmax — see §17.4 for the cohort result.

### 8.2 Async Copy

- Prefer direct global-to-LDS instructions if available
  (`AsyncTileLoader` + `b.async_buffer_load_lds`).
- Track outstanding async operations.
- Place waits just before consumers (`b.s_waitcnt(vmcnt=0)`).
- Ensure boundary lanes write zeros or that suppressed loads do not
  leave stale LDS.
- Understand whether async instruction writes lane-contiguous or
  arbitrary addresses (lane-contiguous on AMD; see Section 6.3).
- If arbitrary swizzle is needed, verify the intrinsic supports it.

### 8.3 Waits And Barriers

- Count `s_waitcnt` and `s_barrier` with `probe_isa_inspect.py`.
- Distinguish global memory waits from LDS waits.
- Use barriers only for cross-thread visibility.
- `s_waitcnt` does NOT replace a barrier when another thread reads
  your LDS write.
- Avoid barrier after direct global store unless needed.
- Collapse multiple waits where possible.
- Avoid compiler-generated over-waiting by separating phases.
- The DSL encodes the canonical `s_waitcnt(vmcnt=16, lgkmcnt=16)`
  value as the constant `20336`; cross-check in the lowered IR.

### 8.4 Scheduling Hints

> Arch-specific (which scheduling intrinsics emit a runtime instruction
> vs act as a compile-time-only fence, and which ICE the compiler —
> `sched_barrier`, `sched_group_barrier`, `iglp_opt`, named/split
> barriers): see the
> [architecture reference](#21-target-architecture-reference)
> (§21.3/§21.8). The gfx950 facts below are illustrative of one such
> arch.

- Try compiler scheduling flags only after correctness is stable.
- On AMD, experiment with `sched_group_barrier(mask, count, sync_id)`,
  `s_setprio`, and `s_waitcnt` from `helpers/schedule.py::SchedulePolicy`.
- Compare ISA with and without hints (`probe_isa_inspect.py`,
  `probe_intrinsic_counts.py`).
- Validate correctness after every scheduler flag.
- Some flags can break generated kernels even when they work for
  hand-written HIP.

On gfx950, `b.sched_barrier(mask)` lowers to `llvm.amdgcn.sched.barrier`
(so `probe_intrinsic_counts.py` *does* show a non-zero `sched.barrier`
after lowering — it is emitted). That intrinsic is a **compile-time
scheduling fence**: it constrains the post-RA scheduler during codegen
and frequently leaves **no runtime `s_sched_barrier` instruction** in the
object. So a zero `sched_barrier` sub-bucket in `probe_isa_inspect.py`
confirms the runtime instruction is absent — **not** that the directive
was dropped. The fence has a measurable effect (an early llvm20 backend
measurement recorded ~+35 % on one single-batch d128 attention cohort at
`num_warps==1`; that win is backend-specific and superseded on production
llvm22 by `iglp_opt` — arch reference §21.8). Always confirm a hint's
effect by diffing the mainloop ISA, not by perf or the ISA-bucket count
alone.

The converse is also actionable on gfx950: the schedule directives the
compv4/compv3 GEMM pipeline *emits* (`sched_group_barrier` HotLoop +
per-cluster `s_setprio` / `sched_barrier(0)`) are **net-negative** — they
over-constrain the backend, and removing them lets the hardware scheduler pack
MFMAs tighter (MfmaUtil 63% -> 68%, +1.9–2.5% on square fp16/bf16 GEMM). Take
the win with `TraitSpec.emit_sched_hints=False` (default `None` is
arch-resolved: already OFF on gfx950, ON elsewhere). Full data:
empirical-case-studies.md Case Study 7.

**`iglp_opt` — the canned backend GEMM scheduler (the one that survives).**
`b.iglp_opt(n)` lowers to `__builtin_amdgcn_iglp_opt(n)` (LLVM
`llvm.amdgcn.iglp.opt`) — a *different* mechanism from `sched_barrier`: instead
of capping reordering, it asks the AMDGPU post-RA scheduler to apply a canned
MFMA / ds_read / ds_write / VMEM interleave for the enclosing loop (the IR-level
analogue of Tensile's `ScheduleIterAlg=3`). It is what CK's own C++ uses
(`gridwise_gemm_pipeline_v2.hpp`, attention pipelines). Place it **once at the
top of the main-loop body**, and when you use it **suppress the manual
`sched_barrier` / `sched_group_barrier` hints** — mixing them fights.
- It only helps when there is **`ds_write` traffic to interleave** against the
  MFMAs. On a **direct-to-LDS path (0 `ds_write`)** or a loop bounded by the
  **barrier rendezvous** rather than in-loop instruction order, it is **neutral**
  (measured exactly neutral on the square fp16 DTL kernel — §17.5 case study). So
  before reaching for it, classify the bound (§8.6): it moves *schedule-bound*
  loops, not *barrier-bound* ones.
- Verify it actually took effect via `probe_isa_inspect.py` (the mainloop
  ds/MFMA interleave should change), not perf alone — a no-op and an
  applied-but-useless directive both read as "no change".

### 8.5 Instruction-Level Balance

- Interleave MFMA, LDS reads, and global loads.
- Avoid long runs of memory instructions followed by long runs of
  MFMA if latency is exposed.
- Avoid immediate use of a just-loaded value if independent work
  exists.
- Group enough MFMA between waits.
- Watch scalar ALU address arithmetic (`probe_isa_inspect.py`'s
  `salu` sub-bucket grows if the address math is not hoisted).

### 8.6 Diagnosing the bound: barrier-bound vs schedule-bound

Before spending effort on scheduling hints (§8.4) or deeper pipelines, classify
*what* the MFMA unit is waiting on — the lever depends entirely on this. Profile
with `rocprofv3` (`MfmaUtil`, `LDSBankConflict`, `MemUnitStalled`,
`OccupancyPercent`):

- **Schedule-bound**: `ds_write`/`ds_read` and MFMA are present but poorly
  interleaved (bank conflict > 0, or visible read→MFMA stalls). → `iglp_opt`
  (§8.4), `s_setprio` clusters, operand prefetch help here.
- **Barrier-bound**: bank conflict ≈ 0, `MemUnitStalled` ≈ 0, but `MfmaUtil`
  still ~50% at **1 WG/CU**. The MFMA idles at the per-K-tile `s_barrier`
  rendezvous with nothing to overlap it. Scheduling hints do **nothing** here
  (proven: `iglp_opt` exactly neutral). The only levers are *fewer/cheaper
  barriers* or *more independent work spanning each barrier* — and on a fused
  direct-to-LDS path those need an assembly-grade `ds_write`/MFMA interleave the
  IR→comgr path cannot place (see §17.5 square-GEMM case study).

**Validated GEMM levers from that case study (gfx950, square fp16):**
- **Single-barrier depth-2 prefetch** — a double-buffer needs only *one*
  `s_barrier_bare` per K-tile if the next-tile async write is issued *after* the
  barrier (one `s_waitcnt(vmcnt=0,lgkmcnt=0)` serves both RAW and WAR). Collapsing
  the naive two-barrier form was the single biggest win (+59%). At 1 WG/CU a
  second barrier directly halves the MFMA shadow.
- **Element-granular LDS swizzle → 0 bank conflicts** — `col ^= ((row>>R)&(2^W-1))<<L`
  with `L=log2(elem)`, `W=log2(block_k/elem)`, `R=0` (key on the *low* `m_in_atom`
  bits, swizzle whole b128 slots). The row·stride term vanishes mod 32 banks, so
  only a low-bit element permutation decorrelates the M-rows; high-bit/2-slot
  swizzles floor at 25%. Applied identically to store-source + ds_read (XOR is
  self-inverse → bit-exact).
- **`dtl_cache_b=CACHE_ALL` for reuse-heavy GEMM** (+9%) — the default
  `CACHE_STREAM` on B is right for one-shot (skinny/MoE) but wrong for square,
  which reuses all of B across M-tiles (corroborates the §17.2 item-14 cache sweep).

**Validated *negatives* (don't re-try without new information):** in-phase
prefetch-local-read (neutral / spills); smaller tiles for 2 WG/CU occupancy (the
big tile at 1 WG/CU wins — reuse beats the barrier-hiding); single-buffer at
2 WG/CU (loses the prefetch, 0.5×); LDS k-padding (occupancy loss); non-DTL
register-staged depth-2 prefetch / PGR2 (correct but 0.42× — the `ds_write` it
requires can't be hidden without assembly scheduling); `iglp_opt` (neutral, see
above). Measurement discipline: this box auto-clocks ±25–30%, so **only
same-session interleaved ratios are valid** — bench the candidate and the
reference back-to-back in one process.

### 8.7 What a good hot-loop schedule looks like

rocke declares ops + dependencies and hands instruction scheduling to the
backend; on some arches the scheduling *hints* are dropped (§8.4,
[architecture reference](#21-target-architecture-reference) §21.8), so you
cannot place instructions directly. You can still shape the schedule indirectly:
the **order, spacing, and dependency structure of the ops you emit** is the
backend's starting point, and inside a fenced region (see the "hard scheduling
fence" bullet below) it is largely what survives. The levers below are what an
assembly-grade scheduler would do for the hot loop — emit toward them, and
crucially **assert the result** with `probe_isa_inspect.py` /
`probe_intrinsic_counts.py`, because a hint that did nothing and one that helped
both read as "no perf delta".

- **Size prefetch to cover the load latency.** A global/VMEM load has far higher
  latency than an LDS read — high enough that a single unhidden one stalls the
  matrix unit across many MFMAs. Pick prefetch depth so the independent MFMA work
  between *issuing* a load and *consuming* it actually covers that latency
  (`SoftwarePipeline(num_buffers=…)` / `helpers/pipeline.py::recommend_prefetch_stages`),
  rather than sweeping blind. A depth that leaves the MFMA unit waiting on VMEM is
  the *memory-bound* signal of §3.3.
- **Never read-then-immediately-MFMA (operand prefetch).** Don't consume a
  just-loaded value in the very next matrix op — its inputs may not have landed,
  and the unit stalls. In the inner `kk` loop, emit the `smem_load_*` for fragment
  `kk+1` *before* the MFMAs of `kk`. This is §8.5 "avoid immediate use of a
  just-loaded value" made mechanical, and the same idea as the §17.4 "early-V"
  reorder one level up.
- **Spread `ds_read`s evenly; avoid long bursts.** Aim for a roughly even
  LDS-read-per-MFMA ratio across the loop body instead of front-loading every
  read, and don't emit long back-to-back runs of LDS reads. Drive
  `SchedulePolicy.emit_after_mfma_step(ds_read_count=…, mfma_count=…)` to
  distribute reads across MFMA steps; hand-authored loops should interleave a few
  reads between MFMAs. (On gfx950 the emitted `sched_group_barrier` is dropped, so
  what actually carries is the *IR emission order*, not the hint.)
- **Roughly one global/async load per MFMA.** Interleave VMEM with compute — issue
  about one `async_buffer_load_lds` per MFMA-step region so loads overlap matrix
  work, instead of clumping all loads at the top of the loop. Matches §8.5
  "interleave … global loads".
- **Budget VALU *under* MFMA latency — it is not free past the budget.** A matrix
  op opens a window where only a bounded amount of VALU co-issues alongside it;
  beyond that, VALU serializes behind the matrix unit. For fused/attention paths
  keep per-MFMA VALU (softmax rescale, fp8 cvt, scaling) under that budget — this
  is the mechanism behind the §17.x finding that *"FP8 dequant adds VALU work,
  which **was** the bottleneck"*. Verify the `valu`/`trans` sub-buckets per MFMA in
  `probe_isa_inspect.py`.
- **Balance the loop boundary.** Don't let two matrix ops abut across the backedge
  with nothing to overlap them — if the last op of the iteration is an MFMA and
  the first of the next is too, the unit stalls at the seam. Make sure the backedge
  carries independent work (the next-tile load) between them. Scheduler-side
  reading of the §8.6 "single-barrier depth-2 prefetch" win.
- **Every wait / store / barrier is a hard scheduling fence.** The backend cannot
  move anything across a store, branch, barrier, or `s_waitcnt`. An over-eager
  `b.s_waitcnt(...)` or a mid-loop store fences the scheduler; place the
  **minimum** wait **as late as legal** (§8.1, §8.3) and prefer one combined wait
  over several.
- **Wait for the count you need, not zero (partial waits).** If you need the
  result of the k-th-oldest of N in-flight loads, wait `vmcnt = N − k` and keep the
  newer loads in flight, instead of draining to `vmcnt=0` every iteration. rocke
  exposes this as `SoftwarePipeline(overlap_vmcnt=True)` →
  `s_waitcnt(vmcnt=prefetch_depth)`; use it for depth≥2 pipelines (cf. §8.6
  single-barrier depth-2; `gemm_wsp3`'s `keep_vmcnt`).
- **Emit FMA, not mul+add; keep constant chains short.** Author scale/bias as
  `b.fma(...)` and pre-multiply constant scales in Python so the chain starts
  short, rather than relying on the backend to fuse `v_mul`+`v_add` and fold the
  constants. Verify the epilogue/softmax ISA shows `v_fma_f32`, not `v_mul`+`v_add`
  pairs.

**gfx1151 (RDNA) note.** RDNA hides dependent-VALU latency with `s_delay_alu`
(inserted by the backend); long *dependent* VALU chains cost delay slots. Keep
them short and break dependencies where you can; spot-check `s_delay_alu` density
in the lowered ISA.

**Verify, don't assume.** Each lever above changes an ISA-visible quantity
(per-iter `ds_read`/`mfma` interleave, `vmcnt`/`lgkmcnt` values, `valu`-per-MFMA,
`v_fma` vs `v_mul`+`v_add`). Treat those as the *acceptance test* for the schedule
you intended, exactly as §8.4 says for `iglp_opt`: a no-op and an
applied-but-useless change both read as "no perf delta", so check the histogram,
not the clock. See §10.5 (limits vs hand assembly) and the
[architecture reference](#21-target-architecture-reference) §21.8.

---

## 9. Epilogue Optimization

### 9.1 Common Epilogues

- Bias add (`helpers/fuse.py::BiasAdd`).
- Activation (`ReLU`, `SiLU`, `GELU`, `tanh`, `sigmoid` —
  `helpers/fuse.py`, `EpilogueOp` chain).
- Residual add (`ResidualAdd`).
- Scaling (`Scale`).
- Quantization / dequantization (`helpers/quant.py`).
- Type conversion (`vec_trunc_f32_to_f16`, `cvt_pk_f32_fp8`).
- Row / column scale (`helpers/mx_scale.py` for MX).
- Clamp.
- Dropout (`helpers/rng.py::dropout_mask_pair_f32`).
- Store transpose (`instances/transpose*` + LDS reader formula).

### 9.2 Direct Epilogue

`helpers/epilogues.py::DirectEpilogue(vec_in_acc=True/False)`.

- Best when accumulator lane layout already matches the global store
  layout.
- Convert vector accumulator to vector output.
- Use packed stores (`b.buffer_store_vN_f16`, `b.global_store_vN_f16`).
- Avoid scalar stores.
- Avoid unnecessary LDS.

### 9.3 LDS / Shuffle Epilogue

`helpers/epilogues.py::CShuffleEpilogue.from_grid(...)`.

- Use when accumulator lane layout is bad for global stores.
- Write per-lane accumulators to LDS.
- Barrier.
- Have a subset of threads read contiguous vectors from LDS.
- Store wide vectors globally.
- Ensure LDS write and read mappings are exact.
- Ensure only active store threads write.
- Include OOB checks for Q/N/K tails.
- Measure whether coalescing gain beats barrier/LDS cost.

"Use LDS for the epilogue" is not the same as "match the library's
LDS epilogue distribution". A naive LDS-staged epilogue with a flat
`[q, group, k]` layout was correct for one shape but wrong for
another and never beat direct vector stores in our experiments. CK's
LDS epilogue uses the MFMA distribution for the LDS write, then a
separate wide-store distribution where only `STORE_VECS = BLOCK_Q ×
BLOCK_C8` threads issue 16-byte global stores. Either copy that exact
mapping or stay with direct vector stores; the in-between case loses
both ways.

A related caveat applies to any structural change that removes one
cost: **the cost may simply move**, not disappear. **§17.4** has a
worked example where removing a per-iter LDS-store stripe replaced it
with a much larger cross-lane permute storm — a material regression
on the same shape. Always confirm via the per-iter ISA histogram
(`probe_isa_inspect.py` or `probe_intrinsic_counts.py`) that the
savings outnumber the introduced overhead before declaring the
change a win.

### 9.4 Output Validation

- Epilogue bugs often affect only certain groups / channels /
  columns.
- Validate all channels, all groups, and both boundary and interior
  Q.
- Test `groups` values that change block count.
- Test both K tails and Q tails.

---

## 10. Compiler And Build Settings

### 10.1 Build Type

- Use Release builds for performance.
- Ensure `NDEBUG` is set for C/C++ kernels when assertions affect
  device code (the DSL passes through to `runtime/comgr.py`).
- Confirm target architecture explicitly
  (`compile_kernel(kdef, isa="amdgcn-amd-amdhsa--gfx950")`).
- Save compile commands and binary metadata.
- Avoid comparing debug and release builds.

### 10.2 Common AMD/HIP Flags

The default flag set in `runtime/comgr.py` is `-O3`. Per-spec
overrides via `compile_kernel(kdef, options=[...])`.

- `--offload-arch=gfx950` (or `gfx942`, `gfx90a`).
- `-O3`.
- `-DNDEBUG`.
- `-fno-offload-uniform-block` when required for launch assumptions
  or perf.
- `-mllvm -amdgpu-function-calls=false`.
- `-mllvm -amdgpu-early-inline-all=true`.
- `-mllvm --lsr-drop-solution=1`.
- `-mllvm -enable-post-misched=0` only if correctness is verified
  (this flag is **known** to miscompile some MLIR-generated kernels
  even when it works for hand-written HIP — see Section 10.3).

Compiler flags very rarely close large performance gaps. Rebuilding
a kernel with the full recommended flag stack including `-DNDEBUG`,
`-fno-offload-uniform-block`, `-mllvm -amdgpu-function-calls=false`,
`-mllvm -amdgpu-early-inline-all=true`, `-mllvm --lsr-drop-solution=1`,
`-mllvm -enable-post-misched=0` has been measured to move throughput
by under 1 % across multiple kernels and DSLs. If you see a multi-×
gap, do not blame compiler flags first; instrument the ISA, the
bottleneck class, and the kernel structure (§3, §11).

### 10.3 DSL-Specific Compiler Hazards

> Arch-specific backend hazards (which intrinsics the LLVM backend
> silently drops or ICEs on, which `-mllvm` flags are risky on a given
> arch): see the
> [architecture reference](#21-target-architecture-reference) (§21.8).

- The DSL ships a conservative pass pipeline:
  `core/passes.py::optimize_kernel` runs canonicalize → conservative
  integer constant fold (add/sub/mul/div/mod/and/or/zext/sext/cmp/
  select) → CSE on pure ops → DCE on unused pure ops. Up to 3
  iterations. No vectorizer, no fuser, no scalar-elision pass. Loads /
  stores / MFMA / barriers are never moved.
- `core/lower_llvm.py` does not vectorize, fuse, or elide scalar
  work. The only scalar-op optimization it applies is
  `_lower_unrolled_for` (Phase 3 unroll + Phase 4a trailing-sync
  elide).
- Specific MLIR/FlyDSL trap (not our DSL, but a related one):
  `-mllvm -enable-post-misched=0` is safe in CK's CMake-generated code
  but produces wrong outputs in MLIR-generated kernels for some
  shapes. The other CK-style flags
  (`amdgpu-function-calls=false`, `amdgpu-early-inline-all=true`,
  `lsr-drop-solution=1`, `-fno-offload-uniform-block`, `-O3`, fast/
  unsafe math) were safe. Always treat compiler flags as
  one-at-a-time experiments with a correctness check after each one.
- DSL AST-rewriter trap: in tracing-style DSLs (like FlyDSL) the AST
  rewriter classifies any `if` whose test contains an `ast.Name` as
  potentially dynamic and converts it into an `scf.if` dispatch. A
  `if loads_next is not None:` guard around a Python-time decision
  is silently rewritten and the body elided, dropping all in-loop
  LDS stores while still emitting all loads, MFMAs, and barriers (and
  producing wrong but plausible-looking outputs). The CK DSL avoids
  this by raising `TypeError` on `Value.__bool__`; you MUST use
  `IRBuilder.static_if(...)` for Python booleans and
  `IRBuilder.scf_if(...)` for runtime predicates.

### 10.4 Alias And Pointer Semantics

- Preserve `noalias` metadata when safe — the DSL `IRBuilder.param`
  accepts `noalias=True`, `readonly=True`, `writeonly=True`, `align=N`,
  `dereferenceable=N`.
- Do not disable alias analysis casually.
- Mark pointers restrict / noalias in C++ when correct.
- Use buffer descriptors with proper bounds when possible.
- Avoid hidden aliasing between input / output / workspace.

### 10.5 Known limitations vs hand-written assembly

The DSL emits LLVM IR and hands scheduling + register allocation to the AMDGPU
backend (`libamd_comgr`). That boundary is the source of the residual gap against
hand-tuned assembly (Tensile/rocBLAS, or a hand-written tile kernel). Know what
it *cannot* express so you don't burn cycles trying:

- **No per-instruction placement.** You declare ops + dependencies; the backend
  orders them. `iglp_opt` / `s_setprio` / `sched_barrier` / `sched_group_barrier`
  are *compile-time hints* — `sched_barrier` is a scheduling fence that
  constrains the post-RA scheduler and frequently leaves no runtime instruction
  in the object (so "0 in the ISA bucket" means *the runtime instruction is
  absent*, not *the directive was dropped*; see arch reference §21.8 and the
  ~+35% it produced on one attention cohort on an early llvm20 backend —
  backend-specific and since superseded). You still
  cannot say "this `ds_write` issues in *this* MFMA slot" — which is exactly how
  assembly (Tensile `ScheduleIterAlg=3`) hides LDS-write cost under the matrix
  pipe. Net effect measured: a fused direct-to-LDS GEMM tops out ~0.82× rocBLAS
  and is barrier-bound, with the last ~18% needing this placement (§8.6, §17.5).
- **No register-double-buffered operand staging across the backend's will.** You
  can carry fragments as `scf.for` iter-args (depth-N global prefetch *is*
  expressible — §17.5 built PGR2), but the backend still schedules the resulting
  `ds_write`s; without slot-level placement they cost full time, so the
  register-staged path lost (0.42×) to the fused 0-`ds_write` path that can't go
  past depth-1. Assembly gets *both*; IR gets one or the other.
- **No named/split barriers on gfx950** (`s.barrier.signal/wait` ICE the
  backend). Only the bare full-CTA `s_barrier` (`b.s_barrier_bare`) — so
  fine-grained warp-specialization handoff is unavailable on this target.
- **The backend owns the final waitcnt placement.** `b.s_waitcnt(...)` is a
  request; comgr may over-wait (observed on some DTL paths). You can shape it
  with phase separation but not pin it.

**Escape hatches when the gap actually matters** (in increasing effort): (1) try
`iglp_opt` — the one sanctioned, surviving scheduler directive (but neutral
unless there is `ds_write` traffic to interleave); (2) emit the hot loop as LLVM
inline `asm` for assembly-grade placement of the critical ~20 instructions while
keeping the rest in the DSL; (3) add a scheduled-assembly backend / custom
post-RA pass (this is what Tensile *is* — a multi-month compiler effort). The
front-end being Python is **not** the limiter — a Python→IR framework can reach
near-assembly results (cf. Triton); the work is always in the backend/scheduler.

**CK Tile is NOT a hand-assembly baseline — there is no ceiling against it.**
When measuring against CK Tile FMHA / GEMM, remember the comparison is
IR-frontend vs IR-frontend, not IR vs assembly. CK Tile is C++ templates →
`hipcc` (clang) → the **same** LLVM AMDGPU backend (same instruction selection,
register allocation, post-RA scheduler) the DSL emits IR into. The DSL's control
is a strict superset: it *states* the atom, the LDS swizzle, the partial
`lgkmcnt` waits, the direct-to-LDS async copies, and the scheduling fences that
hipcc must *infer*. So matching and beating CK Tile everywhere is the correct
expectation — a gap to CK Tile is undone engineering (a missing technique or a
selection bug), not an architectural wall. The hand-assembly ceiling above
applies **only** to full hand-asm targets that bypass the LLVM scheduler
(rocBLAS/Tensile, hand-written whole-kernel asm); do not conflate the two.

### 10.6 Occupancy methodology on a clock-throttled box

When you do not have `sudo` to lock clocks (`rocm-smi --setperflevel high
--setsclk 7`), the GPU auto-throttles ±25–30 % between runs, and *absolute*
latency/TFLOPS numbers are not trustworthy across runs. Two disciplines make
optimization possible anyway:

- **Occupancy is the throttle-free primary signal.** VGPR / AGPR / LDS are
  fixed by codegen and independent of clock state. Extract them with
  `llvm-readelf --notes` (or `probe_occupancy.py`, which wraps it):
  `.vgpr_count` / `.agpr_count` (against the 256 / 256 + 512-total file) and
  `.group_segment_fixed_size` (LDS bytes, against the per-CU LDS budget) give
  the achievable WG/CU exactly. This is also how a backend artifact gets
  caught: an early llvm20 reading of the d128 attention prefill body showed
  256 VGPR + spills / 1 WG/CU and pointed at an LDS-cut occupancy story; the
  production llvm22 reading of the same body is 213 VGPR / 0 spill / 2 WG/CU,
  so occupancy was never the wall and the lever is a schedule hint
  (`iglp_opt`), not a register/LDS redesign — see arch reference §21.4a.
  (Register/occupancy counts are backend-sensitive; always read them on the
  production backend — llvm22 + comgr 7.2, torch imported first.)
- **Perf only via same-session ratios + large effects.** Benchmark the
  candidate and its reference (flash, CK Tile, or the prior config)
  **back-to-back in one process**, and only trust the *ratio*; the throttle
  applies equally to both within a session. Treat anything under ~5 % as noise.
  Single-attempt or cross-process absolute numbers must not be cited as wins.

---

## 11. ISA And Resource Inspection

> Arch-specific (the `--mcpu` / ISA target to disassemble against, the
> resource caps the occupancy math uses, which opcodes to expect): see
> the [architecture reference](#21-target-architecture-reference)
> (§21.4/§21.9).

The DSL's static analysis layer lives at `rocke.analysis`:

| Tool | Source |
|---|---|
| `analyze_llvm_ir(text)` | `analysis/ir.py` — counts MFMA / async / raw-buffer / barriers / waitcnts in LLVM IR |
| `analyze_hsaco(hsaco_path)` | `analysis/isa.py` — uses `llvm-objdump -d` + `llvm-readelf --notes`, returns `IsaStats` + `ResourceInfo` |
| `parse_isa(text)` | `analysis/isa.py` — opcode tally |
| `parse_resources(text)` | `analysis/isa.py` — VGPR / SGPR / AGPR / LDS / spill |
| `VariantReport.from_artifact(...)` | `analysis/report.py` — joins compile + IR + ISA + bench |
| `compare_variant_reports([...])` | sorted comparison rows |

The DSL probes under `utilities/tools/dsl_probes/` are the
**runbook-aligned interactive** surface on top of this:

```text
probe_intrinsic_counts.py   → analyze_llvm_ir-like, with custom intrinsic table
probe_isa_inspect.py        → analyze_hsaco-like, with VALU/SALU sub-buckets
probe_occupancy.py          → readelf notes + occupancy estimate
```

To diff a DSL kernel against a **hand-written C++/HIP reference** (a tile-library
or research kernel you want to match), use
`utilities/tools/utils/reference_isa_diff.py`: it compiles the reference with
`hipcc`, extracts the `gfx950` code object from the fatbin (`roc-obj-ls`),
disassembles both sides, and prints a semantic histogram + run-length diff
(`reference_isa()` / `rocke_isa()` / `diff_report()`). This is the only tool that
introspects a non-DSL reference — the rest of the layer only sees `KernelDef`s.

### 11.1 What To Count

- MFMA / WMMA instructions.
- Global loads / stores.
- LDS reads / writes.
- Barriers.
- Waits.
- Scalar ALU instructions.
- Vector ALU instructions.
- Branches.
- Atomics.
- Conversion instructions (`v_cvt_*`, `cvt_pk_*`).

### 11.2 What To Extract

- VGPR count.
- SGPR count.
- LDS bytes.
- Scratch / spill usage.
- Occupancy metadata.
- Code object target.
- Workgroup size.
- Waves per workgroup.

### 11.3 How To Interpret

- High VGPR with low occupancy may need fewer accumulators or a
  smaller tile.
- High barriers with little compute per phase suggests bigger
  per-barrier work or fewer phases.
- Many scalar stores indicate an epilogue problem.
- Many address instructions indicate offset math should be hoisted /
  precomputed.
- Register spills usually invalidate performance conclusions until
  fixed (`probe_occupancy.py` flags `spill > 0`).
- If MFMA count is higher than expected, inspect loop unrolling and
  instruction choice.

Example diff (from running `probe_intrinsic_counts.py` on the
`use_mfma_32x32` lever in `UnifiedAttention2DTiledSpec`):

```text
baseline_mw16 → mfma32_transposed:
  mfma.f32.16x16x32.bf16:   17 → 0   (-17, replaced)
  mfma.f32.32x32x16.bf16:    0 → 17  (+17)
  ds.swizzle:               32 → 0   (-32, replaced)
  ds.bpermute:               0 → 67  (+67)
  fmul (LLVM IR):           53 → 113 (+60)
```

The diff makes the algorithmic change visible: the `transposed`
variant swaps an `ds.swizzle`-based softmax butterfly for an
`ds.bpermute`-based one. The next question — "is the bpermute path
faster on this shape?" — is for `probe_targeted_bench.py`.

---

## 12. Autotuning Strategy

> Arch-specific: the legal range of several knobs below (MFMA atom and
> K-pack §12.1.C, LDS layout / swizzle §12.1.F, compiler flags
> §12.1.M, chiplet swizzle §12.1.L) depends on the target — prune the
> sweep to what your arch actually supports. See the
> [architecture reference](#21-target-architecture-reference).

### 12.1 Knob Catalog (Master List)

This is the master enumeration of every performance lever exposed by
`rocke` and the surrounding workflow. Knobs are grouped by lever
family. For each knob: where it lives, what it controls, what direction
it usually moves perf, and when **not** to flip it.

To discover all knobs for a specific kernel: open the spec dataclass
under `instances/common/<kernel>.py` (arch-specialized overrides, when
they exist, live under `instances/<arch>/<kernel>.py`) and read the
`@dataclass` field list. Every field is a knob (the validator in
`__post_init__` documents the constraints). The shorthand
`instances/<kernel>.py` used throughout this runbook refers to that
`instances/common/` builder unless an arch directory is named.

#### 12.1.A Algorithmic variant (choose the kernel before the knobs)

| Lever | Where | Direction |
|---|---|---|
| GEMM family member | `instances/gemm_universal.py` / `batched_gemm.py` / `grouped_gemm.py` / `streamk_gemm.py` / `gemm_multi_d.py` / `gemm_multi_abd.py` / `mfma_gemm.py` / `flatmm.py` / `block_scale_gemm.py` / `mx_gemm.py` | small / decode shapes → `flatmm`; many small problems → `grouped_gemm` or `persistent`; tail-balance → `streamk_gemm`; fused chain → `gemm_multi_d` / `gemm_multi_abd` |
| Conv family | `conv_implicit_gemm.py` (incl. `unroll_k=True` auto-unrolled path) / `conv_direct_grouped.py` (16c, 4c) / `img2col.py` | tiny K or C*R*S → direct conv; 3×3 hero shapes → implicit GEMM; explicit im2col → if downstream stage is plain GEMM |
| Attention family | `attention_unified.py` (scalar oracle) / `attention_tiled_2d.py` / `attention_tiled_3d.py` (split-KV) | prefill → 2D; long-context decode → 3D; sliding-window — see §17.4 final policy |
| FMHA family | `fmha_mfma.py` / `fmha_varlen.py` / `fmha_head_grouping.py` / `fmha_paged_prefill.py` / `fmha_splitkv_decode.py` / `fmha_fwd_fp8.py` / `fmha_bwd.py` / `fmha_appendkv.py` / `sage_attention.py` / `sparse_attention.py` | choose based on KV layout (paged vs varlen), GQA, dtype, sparse pattern |

#### 12.1.B Tile and block geometry

| Knob | Spec | Default | What it does | When to change |
|---|---|---|---|---|
| `tile_m`, `tile_n`, `tile_k` | GEMM / Conv | 64 / 64 / 64-128 | Per-CTA macro tile in M / N / K | Larger ⇒ more reuse and bigger MFMA hot loop; bounded by LDS budget and VGPR pressure (run `probe_occupancy.py`) |
| `warp_m`, `warp_n`, `warp_k` | GEMM / Conv | 2 / 2 / 1 | Warp grid inside the CTA | `warp_m × warp_n × warp_k × wave_size` = block_size |
| `warp_tile_m`, `warp_tile_n`, `warp_tile_k` | GEMM / Conv | 32 / 32 / 16 | The MFMA atom shape (see §12.1.C) | One of the validated `_F16_WARP_TILE_SHAPES_GFX950` / `_BF16_WARP_TILE_SHAPES_GFX950` sets |
| `block_size` | GEMM / Attention | derived | Threads per CTA (must equal `num_warps × 64`) | Override only for autotune experiments |
| `num_warps` | Attention | 1 | wave64 warps per CTA — `BLOCK_M = num_warps × block_m_per_warp` | 1 (decode) / 2-4 (prefill) / 8 (Triton-style large prefill); ∈ {1,2,4,8} |
| `tile_size` | Attention | `block_size` | T = KV tokens per K-loop iter; multi-block decomposition | Larger ⇒ fewer outer iters, more LDS; requires `T × head_size ≥ num_warps × 64 × 8` and `T % block_size == 0` |
| `block_m_per_warp` | Attention | 16 | M rows per warp (16 = one MFMA atom; 32 = two stacked atoms or one 32×32) | 32 only when MFMA is 32×32 |
| `block_q` | DirectConv | 16 / 4 | Q output rows per CTA | bigger ⇒ more LDS for input rows |
| `block_groups` | DirectConv | 8 / 16 | Conv groups per CTA | Sweep ±1: 16c best was BLOCK_GROUPS=4, not 8 (§17.1) |
| `groups` | conv_implicit_gemm | 1 | Grouped conv via descriptor `unmerge("group")` | `> 1` for ResNeXt / depthwise-style |
| `k0_k1_split` | conv_implicit_gemm | False | K0/K1 inner-dim split for `CoalescedTileLoader` | When C is the natural contiguous dim |
| `n_acc_slots` (derived) | DirectConv | KH | Circular accumulators for row streaming | Conv 3×3 → 3 slots |

#### 12.1.C MFMA atom and K-pack

| Knob | Spec | Effect |
|---|---|---|
| MFMA atom shape (via `warp_tile_*`) | GEMM / Conv | `16x16x16`, `16x16x32`, `32x32x8`, `32x32x16`, `4x4x4`. Larger K-pack ⇒ half the K-loop trips. (§7.1, §7.4) |
| `use_mfma_32x32` | Attention 2D | switch QK / PV atom from 16×16×32 to 32×32×16. **Headline structural win for attention** — see §17.4. Requires `block_m_per_warp=32` and `tile_size % 32 == 0` |
| `use_transposed_qk_32x32` | Attention 2D | orient softmax with one query column per lane; eliminates a cross-lane reduction. Requires `use_mfma_32x32` |
| `kpack` (bf16/f16 atoms) | `MfmaAtom` factory | True ⇒ pick the larger-K variant of a given (M, N) atom (`16x16x32` over `16x16x16`); halves K-loop count |
| `MfmaAtom.f16_*` / `bf16_*` / `fp8_*` / `bf8_*` | `helpers/atoms.py::MFMA_*_ATOMS` | The catalog every kernel picks from |

Critical caveat: lane-layout matching between chained atoms matters as
much as atom shape (§7.2). 16×16×32's C-output doesn't match its A-input
→ QK→PV chain pays 288 cross-lane permutes per iter; 32×32×16 natively
matches → zero re-pack. See §17.4 for the quantitative reduction.

#### 12.1.D Pipeline / scheduling

| Knob | Spec | Values | Effect |
|---|---|---|---|
| `pipeline` | GEMM `TraitSpec` | `"mem"` / `"compv3"` / `"compv4"` | `mem` = single-buffer; `compv4` = double-buffered async DMA + MFMA overlap. Compv3 / compv4 trade LDS for latency hiding |
| `scheduler` | GEMM `TraitSpec` | `"intrawave"` / `"interwave"` | Where the scheduler injects waits. Intrawave keeps producers and consumers in one wave; interwave splits them |
| `async_dma` | conv_implicit_gemm | False / True | Enable direct global → LDS DMA (`raw_ptr_buffer_load_lds`). Pairs with `pipeline="compv4"` |
| `unroll_k` | conv_implicit_gemm | False | Python-time unroll of the K loop. Bigger code, fewer waits |
| `use_early_v_schedule` | Attention 2D | False | Issue current-V async copy before QK so V overlaps QK + softmax (§8.1, §17.4). Use only on no-SW prefill |
| `prefetch distance` (implicit) | `pipeline` choice | — | More stages ⇒ better latency hiding but more LDS |
| `helpers/pipeline.py::SoftwarePipeline.run_ping_pong` | helper | — | Manual prologue / steady / epilogue staging when you author your own kernel |
| `helpers/schedule.py::SchedulePolicy` | helper | — | Emit `sched_group_barrier(mask, count, sync_id)` + `s_setprio` hints |
| `b.iglp_opt(n)` | IR primitive | `0` / `1` / `2` | Canned backend GEMM MFMA/memory interleave (`llvm.amdgcn.iglp.opt`). *Survives* on gfx950. Place once at loop-body top; suppress manual sched hints when used. Helps schedule-bound loops w/ `ds_write` traffic; **neutral on direct-to-LDS / barrier-bound** (§8.4, §8.6, §17.5) |
| `b.s_barrier_bare()` | IR primitive | — | Bare WG rendezvous, no implicit waitcnt — enables single-barrier ping-pong loops (§8.6, §17.5) |

Caveat: `s_sched_barrier` / `s_sched_group_barrier` are **compile-time
scheduling fences** on gfx950 (§3.1a / §8.4) — they constrain the
post-RA scheduler at codegen and frequently leave no runtime instruction
in the object, so a zero `sched_barrier` sub-bucket in
`probe_isa_inspect.py` means the runtime instruction is absent, not that
the directive was dropped (it can have a real effect — see §8.4).
`iglp_opt` and `s_barrier_bare` (above) also survive. Confirm a hint's
effect by diffing the mainloop ISA, not by the bucket count alone.

#### 12.1.E Epilogue

| Knob | Spec | Values | Effect |
|---|---|---|---|
| `epilogue` | GEMM `TraitSpec` | `"default"` / `"cshuffle"` | `cshuffle` = LDS-staged + wide `buffer_store_dwordx{2,4}`; `default` = direct per-lane stores |
| `epilogue` | conv_implicit_gemm | `"default"` / `"cshuffle"` | Same trade-off |
| `vec_in_acc` | `helpers/epilogues.DirectEpilogue` | — | True when the accumulator's per-lane elements are already contiguous (4×4 direct conv) |
| `CShuffleEpilogue.from_grid(...)` | `helpers/epilogues.py` | — | Builds the cshuffle distribution from a `WarpGrid` |
| `FusedEpilogue` chain | `helpers/fuse.py` | — | `BiasAdd`, `ReLU`, `SiLU`, `GELU`, `Cast`, `Clamp`, `ResidualAdd`, `ResidualMul`, `Scale`, `tanh`, `sigmoid` |

Caveat: the "in-between" naive LDS epilogue (LDS layout that doesn't
match either CK's wide-store distribution or a direct contiguous
pattern) loses to both extremes. Either match the library mapping or
stay direct (§9.3, §17.4 register-PV regression analogue).

#### 12.1.F LDS layout

| Knob | Spec | Default | Effect |
|---|---|---|---|
| `lds_k_pad` | conv_implicit_gemm | None | K-pad to break bank conflicts (`+8` sync default; `0` async default) |
| `lds_layout` | conv_implicit_gemm | None | Explicit `LdsLayout` (helpers/layouts.py) — padding, packed-async, transpose-reader |
| `LdsLayout` swizzle | `helpers/layouts.py` | — | XOR swizzle (zero LDS waste, higher ALU cost) vs padding swizzle (small LDS waste, lower ALU). Architecture-specific rule §6.4a |
| `TransposeLdsReader` | `helpers/layouts.py` | — | Use `ds_read_tr16_b{64,128}` for transposed BF16/F16 loads |
| `pad_m` / `pad_n` / `pad_k` | GEMM `TraitSpec` | False | Pad operands to tile boundaries (avoids tail scalar path) |
| `Q_lds`, `K_lds`, `V_lds`, `P_lds` sizing | Attention 2D (implicit via shape) | — | The §17.4 case showed 16 KiB allocated for `P_lds` was the structural cost; `use_register_pv` removes it |

#### 12.1.G Register / occupancy

| Knob | Spec | Default | Effect |
|---|---|---|---|
| `waves_per_eu` | GEMM, conv, Attention 2D | None | `"amdgpu-waves-per-eu"` hint. 2-3 forces VGPR budget down, more waves/CU. None ⇒ LLVM heuristic |
| `wave_size` | GEMM, DirectConv | 64 | wave64 is the only path the helpers support today |
| `kernel.attrs["max_workgroup_size"]` | DSL IR | derived | `"amdgpu-flat-work-group-size"` emitted from block_size |
| `use_agpr_alloc_zero` | Attention 2D | False | Force VGPR-only MFMA — avoids AGPR↔VGPR copies on accumulator-touching paths. Already a no-op when the existing R4 path has zero AGPR moves (§17.4) |
| `IRBuilder.param(noalias, readonly, writeonly, align, dereferenceable)` | DSL IR | — | Alias + alignment hints for the LLVM backend |

#### 12.1.H Operand layout & preshuffle

| Knob | Spec | Default | Effect |
|---|---|---|---|
| `dtype_a`, `dtype_b`, `dtype_c`, `dtype_acc` | GEMM `DataSpec` | f16/f16/f16/f32 | Per-operand precision |
| `layout` | GEMM `DataSpec` | `"RCR"` | A row / B col / C row major (only "RCR" today) |
| `preshuffle_b` | GEMM `TraitSpec` | False | B operand pre-shuffled by host via `host_preshuffle_layout`; per-lane B-load uses one `buffer_load_dwordx4` per K-tile instead of strided scalar loads |
| `batched` | GEMM | False | Reads `block_id_z` as batch index, picks up `stride_{a,b,c}` args |

#### 12.1.I Persistent / Stream-K / Split-K

| Knob | Spec | Default | Effect |
|---|---|---|---|
| `persistent` | GEMM `TraitSpec`, `StreamKGemmSpec` | False | Persistent CTA loops over many macro tiles via `persistent_tile_for_each` (workspace counter) |
| `num_cus` | `StreamKGemmSpec` | 304 | Target CU count for Stream-K partition |
| `blocks_per_cu` | `StreamKGemmSpec` | 1 | Persistent dispatch density |
| `reduction` | `StreamKGemmSpec` | `Atomic` | `StreamKReductionStrategy.{Atomic, Reduction, AtomicWithFixup}` |
| split-K | GEMM (not yet a spec field; available primitives) | — | `b.global_atomic_add_f32` for atomic split-K; `helpers/streamk.py` for the Stream-K macro tile decode |
| Grouped GEMM persistent | `instances/grouped_gemm.py` | per-group launches | v2 persistent variant is a documented follow-up |

#### 12.1.J Quantization

| Knob | Spec | Values | Effect |
|---|---|---|---|
| `kv_storage_dtype` | Attention 2D | None / `"fp8e4m3"` | FP8 K/V cache — halves KV HBM bytes; dequant happens on load (sync) or in-register (`use_fp8_mfma_qk`) |
| `use_fp8_mfma_qk` | Attention 2D | False | Native fp8 MFMA for QK (K stays raw fp8 in LDS; in-register dequant). bf16 math preserved |
| `use_fp8_mfma_pv` | Attention 2D | False | Native fp8 MFMA for PV (P quantized to fp8 before PV) |
| `mantissa_dtype` | `BlockScaleGemmSpec` | — | FP8 / BF8 element + per-block scale |
| `MxMantissaDType` | `MxGemmSpec` | — | E8M0 shared exponent; MX (microscaling) |
| `QDType` | `helpers/quant.py` | `"i8"` / `"fp8e4m3"` / `"bf8e5m2"` | Quant dtype for smoothquant, MoE smoothquant, add-rmsnorm-rdquant |
| `helpers/codebook.py` | helper | — | i4 packed weight unpack (`codebook_lookup_i4_pair_to_{bf8,fp8}`) |
| `helpers/i4_dequant.py` | helper | — | i4 dequant primitives |

Caveat: FP8 KV is a HBM-bandwidth lever, not a compute lever. If the
PMC says `MemUnitStalled < 1 %` the kernel isn't HBM-bound and FP8 KV
will be neutral or negative (§17.4 worked example).

#### 12.1.K Attention-2D micro-levers (UA 2D specific)

Every flag below is exposed on `UnifiedAttention2DTiledSpec`. They are
all validated in `__post_init__` with explicit constraints —
read the validator before combining flags. Cohort impact is from §17.4.

| Flag | Effect | Best paired with |
|---|---|---|
| `use_register_pv` | Keep P in registers; remove `P_lds` (16 KiB / WG) and the 64 `ds_write_b16` per iter | Only wins when paired with `use_mfma_32x32 + use_transposed_pv_tr_read`; **net regression alone with 16×16 atom** |
| `use_transposed_qk_32x32` | Orient softmax with one query column per lane | `use_mfma_32x32` |
| `use_transposed_scalar_state` | Single m / l per query lane (not per acc reg) | `use_transposed_qk_32x32` |
| `use_transposed_invariant_hoist` | Hoist row invariants out of per-reg/per-tile score loop | `use_transposed_qk_32x32` |
| `use_transposed_mask_once` | Compute mask invariants once per KV iter | `use_transposed_qk_32x32` |
| `use_transposed_half_local_pv` | Each 32-lane half consumes K rows it owns; matching half-local V `ds_read_tr16_b64` | `use_transposed_qk_32x32`. Strongest individual lever after R4 |
| `use_mfma32_skip_legacy_qreg` | Skip dead 16x16 Q register gather and drain barrier | `use_mfma_32x32` |
| `use_transposed_mask_limit` | Collapse causal + prefix masks into one compare | Full R4_s1mask stack |
| `use_grouped_kv2_softmax` | 2 KV tiles per acc update | Smoke OK, full cohort regressed (§17.4 "did not help") |
| `use_fast_paged_kv_desc` | Specialized paged-KV byte descriptor for the hot R4 shape (bf16, h64kv8, HD=64, BS=32, T=64, nw=4) | Use only on supported shape class |
| `use_early_v_schedule` | Issue V async before QK; V overlaps QK + softmax | No-SW d64 long prefill only (d128 regresses) |
| `use_v_double_buffer` | Prefetch `V[i+1]` into a second LDS slot | **d64 short single-batch combo only**; off for d128 (the no-prefetch path is the faster pick) |
| `use_softmax_mfma_interleave` | One `iglp_opt` at the loop top so the backend interleaves the causal-mask + softmax VALU into the MFMA window (shrinks the inter-MFMA gap) | **The production d128 schedule lever** (gfx950 single-batch combo, llvm22): crosses Triton at bf16 S1024. Mutually exclusive with `use_sched_barrier` |
| `use_k_single_buffer` | One `K_lds` slot instead of two, keeping `T=2*block_size` | In the tree but **not the production lever** — it addressed an llvm20 occupancy wall that does not exist on llvm22 (the d128 body is already 213 VGPR / 2 WG/CU) |
| `use_sched_barrier` | Compile-time `llvm.amdgcn.sched.barrier` fence between the QK MFMA cluster and the post-QK prefetch issue (keeps QK MFMAs packed) | An early llvm20 measurement gave +35–36 % at `num_warps==1` only; **backend-specific and superseded** by `use_softmax_mfma_interleave` on llvm22 — off on every live cohort, kept for the mechanism |
| `use_agpr_alloc_zero` | Force VGPR-form MFMA | Bit-accurate but **not** the d128 lever on llvm22 (leaves the inter-MFMA gap unchanged); redundant on R4 (already 0 AGPR moves) |

**Selector-driven (auto-on) attention routing levers.** The single-batch d128
levers above are *selected automatically* by
`instances/common/attention_unified.py` for the qualifying gfx950 cohort
(`head_size=128`, `num_seqs=1`, bf16/fp16, no-FP8, no-SW, `max_seqlen_q>256`) —
you do not flip the spec field directly. On the production **llvm22** backend the
d128 body is already occupancy-clean (213 VGPR / 0 spill / 2 WG/CU), so the
production lever is the **MFMA-schedule hint** `use_softmax_mfma_interleave`
(paired with `use_q_direct_reg`); the selector also forces `num_warps=2`
(`_select_2d_num_warps` — `num_warps=1` is occupancy-starved on the tiny
single-seq grid) and `waves_per_eu=2` (`_select_2d_waves_per_eu`, which does the
occupancy work; the older `waves_per_eu=3`-at-long-S rule regressed this cohort),
and disables the V double-buffer (`_enable_v_double_buffer`). d64 single-batch
combo uses `num_warps=4`. The K-single-buffer / small-tile occupancy levers and
the `HIPDNN_GFX950_D128_SMALL_TILE=0` opt-out hatch remain in the tree but
address an llvm20-only occupancy wall — they are not the production lever. See
arch reference §21.4a and §17.4 for the measured occupancy and per-shape numbers.

#### 12.1.L Multi-XCD / chiplet grid swizzle

| Knob | Spec | Default | Effect |
|---|---|---|---|
| `chiplet_swizzle` | GEMM / conv | False | Remap WGIDs so contiguous stripes land on the same XCD (L2 reuse) |
| `chiplet_wgm` | GEMM / conv | 8 | Super-tile WGM grouping |
| `chiplet_num_xcds` | GEMM / conv | 8 | MI300X / MI325X / MI350X have 8 XCDs |
| `chiplet_chunk_size` | GEMM / conv | 64 | XCD round-robin chunk size |
| `helpers/grid.py::chiplet_transform_chunked` | helper | — | Pure helper if you author your own kernel |
| Constants `NUM_XCDS_MI300X / MI325X / MI350X` | `helpers/grid.py` | — | All 8 today |

#### 12.1.M Compiler flags

Default flag list from `runtime/comgr.py` is `["-O3"]`. Per-spec
overrides via `compile_kernel(kdef, options=[...])` or
`build_hsaco_from_llvm_ir(..., options=[...])`.

| Flag | Safe? | Effect |
|---|---|---|
| `-O3` | ✅ default | LLVM optimization level |
| `-DNDEBUG` | ✅ | Disable C++ assertions on device |
| `-fno-offload-uniform-block` | ✅ | Required for some launch / perf assumptions |
| `-mllvm -amdgpu-function-calls=false` | ✅ | Force inline |
| `-mllvm -amdgpu-early-inline-all=true` | ✅ | Early inlining |
| `-mllvm --lsr-drop-solution=1` | ✅ | LSR pass tweak |
| `-mllvm -enable-post-misched=0` | ⚠️ risky | Safe in CK; **miscompiled MLIR-generated kernels** in our experience |
| `--offload-arch=gfx950 / gfx942 / gfx90a` | ✅ | Target ISA |
| `compile_kernel(kdef, isa="amdgcn-amd-amdhsa--gfx950")` | ✅ | DSL-level ISA override |

Compiler flags very rarely close large gaps (§10.2). A full flag stack
moved measured throughput by under 1 % on direct-conv kernels.

#### 12.1.N Runtime / launch

| Knob | Where | Effect |
|---|---|---|
| `KernelLauncher` | `runtime/launcher.py` | One-HSACO, repeated launches — amortizes HIP module load |
| `PipelineLauncher` | `runtime/launcher.py` | Multi-stage chained launches on one stream |
| `WorkspacePool` | `runtime/launcher.py` | Keep long-lived torch workspaces alive across launches |
| `no_fence` context manager | `runtime/launcher.py` | Skip per-call sync inside an event-timed loop (graph-style) |
| `time_launches(fn, warmup, iters, stream)` | `runtime/launcher.py` | The canonical HIP-event timer |
| `StreamConfig` | `runtime/launcher.py` | Mirror of CK Tile `stream_config` |
| `resolve_stream(stream=0)` | `runtime/torch_module.py` | Substitute torch's current stream to keep allocator coherent |
| `pack_args` vs `pack_args_kernelparams` | `runtime/torch_module.py` | AMDGPU kernarg buffer vs the safer `kernelParams` path |
| HIP graph capture | torch | Amortizes launch overhead — pair with `no_fence` and many iters |
| `rocm-smi --setperflevel high && --setsclk 7` | shell | Lock clocks to avoid thermal / DVFS noise during measurement |

#### 12.1.O Dispatcher / selector policy (per-shape branching)

A single kernel variant rarely wins every shape (§12.3, §17.4). The
selectors in `attention_unified.py` are the canonical dispatch surface;
override them via monkey-patch for sweeps, then promote a stable
policy back into the source.

| Selector | What it picks |
|---|---|
| `_select_2d_num_warps(problem)` | 1 (decode) / 2-4 (prefill) / 8 (Triton-style large prefill) |
| `_select_2d_tile_size(problem)` | T = block_size (decode), 64 (prefill), 32 (sliding-window) |
| `_select_2d_block_m_per_warp(problem)` | 16 / 32 |
| `_select_2d_waves_per_eu(problem)` | None / 2 / 3 |
| `_enable_mfma_32x32(problem)` | True for bf16 long-prefill no-SW |
| `_enable_transposed_qk_32x32(problem)` | gated on dtype, head_size, seq lens, sinks, softcap, sliding_window |
| `_enable_register_pv(problem)` | currently hard-disabled; lift carefully |
| `_enable_fp8_mfma_qk(problem)` | requires `kv_storage_dtype = "fp8e4m3"` |
| `use_2d_kernel(problem)` | 2D vs 3D split-KV — depends on `num_2d_prgms` vs target |
| `select_2d_config / select_3d_config` | top-level Attention dispatch |

Best UA 2D policy (§17.4): branch on `sliding_window` to route
no-sliding-window shapes to one kernel variant (early-V schedule,
default tile size) and sliding-window shapes to a different variant
(smaller tile size matched to the useful work per tile). Two kernel
variants behind one dispatcher entry point, materially better than
either single variant.

#### 12.1.P Benchmark hygiene

| Knob | Where | Effect |
|---|---|---|
| `--attempts ≥ 5` (fresh process) | `sweep_bench.py` | Catch bimodal latency and first-run JIT effects |
| `--warmup ≥ 5`, `--iters ≥ 20` | `time_launches`, harnesses | Discard JIT / module-load cost before measuring |
| Discard first run | benchmark policy | Cold-cache vs warm-cache effect can be > 2× (§2.3) |
| Median + spread (`(max-min)/median × 100 %`) | `benchmark/summary.py::summarize_runs` | Standard quoted statistics |
| Salt kernel symbol with shape hash | per-spec | Avoid HSACO module-cache aliasing across specializations (§12.2) |
| Full cohort vs smoke set | sweep harness | Smoke runs miss shape-dependent variance — see §12.2 caveat |

#### 12.1.Q Static probes (cheap pre-bench filter)

These are not perf knobs but per-iter signal generators that determine
which knob to flip next. See §18 for the full workflow.

| Probe | Output | What it tells you |
|---|---|---|
| `probe_occupancy.py` | VGPR / AGPR / SGPR / LDS / waves-per-CU / limiter | Is the variant occupancy-bound? Which resource? |
| `probe_intrinsic_counts.py` | LLVM AMDGCN intrinsic histogram | Did the atom switch actually emit the new intrinsic? Is async DMA active? |
| `probe_isa_inspect.py` | Post-codegen opcode histogram with VALU/SALU sub-buckets | Are stores vectorized? Are waitcnts excessive? |
| `probe_lowering_compare.py` | LLVM-direct vs HIP-debug HSACO compare | Does the HIP-debug backend agree on this kernel? |
| `probe_config_sweep.py` | Build status + HSACO size across overrides | Variant build matrix |
| `probe_targeted_bench.py` | Per-shape latency vs baseline (Triton, etc.) | Final latency table per shape |
| `probe_rocprof_single.py` | One-process kernel runner for rocprof | Clean rocprof attach window |

### 12.2 Sweep Discipline

- Sweep one family at a time.
- Cache compiled variants (`sweep.py` keys by spec hash).
- Record compile failures (`BuildRecord.error`).
- Record correctness failures.
- Record resource usage (`probe_occupancy.py`).
- Use median and variance (`benchmark/summary.py::summarize_runs`).
- Keep best per shape, not only the global best
  (`sweep.py::pick_best`).
- Avoid overfitting to one shape.
- **Salt cached kernel symbols with specialization constants.** When
  two kernel specs differ only in compile-time constants (e.g.
  `num_seqs`, binary-search trip count), the runtime HSACO/module
  cache may alias them under the same display symbol — and serve
  the wrong cached blob. Append a short shape hash to the kernel
  name before compile. **§17.4** documents the failure mode and the
  fix.
- **Always run the full cohort, not a smoke set.** Several levers in
  the §17.4 pass (grouped-KV2 online softmax, register-P proxy)
  looked promising on small smoke runs and regressed materially on
  the full production-shape sweep. The smoke set under-samples the
  shape-dependent variance.

### 12.3 The Dispatcher Is A Perf Lever

A single kernel variant rarely wins every shape. Per-shape branches
in the selector close the long-tail without compromising the hero
configuration. **§17.4**'s final policy branches on `sliding_window`
to route no-sliding-window shapes to one variant (early-V schedule,
default tile size) and sliding-window shapes to a different variant
(smaller tile size matched to the useful work per tile). Two kernel
variants behind one entry point, materially better than the best
single-variant pick. When designing a selector, budget explicit cases
for: long-prefill no-SW, short-prefill, decode, sliding-window,
single-batch, plus the FP8 KV variant if applicable.

The DSL tuning stack:

- `gemm_universal.all_dispatcher_configs(...)` — pre-baked GEMM
  config catalog.
- `sweep.py::build_all_instances(specs)` — parallel build with
  content-hash caching, emits a JSON run-plan.
- `sweep_bench.py` — bench `>=3` fresh-process attempts per
  (kernel, shape), reports median + spread.
- `helpers/autotune.py::Autotuner`, `AutotuneCache`,
  `AutotuneConfig`, `AutotuneResult`.
- `probe_config_sweep.py` — interactive in-process sweep over a
  spec dataclass.

### 12.4 Variant Naming

Use names that encode the hypothesis:

- `v1_scalar` — scalar baseline.
- `v2_mfma` — added MFMA atom.
- `v3_h_pipeline` — added H-row pipeline.
- `v4_async_lds` — switched to async DRAM→LDS.
- `v5_vector_store` — wide direct epilogue.
- `v6_k32` — K=32 MFMA fold.
- `v7_wg4_nf4` — block_groups=4, n_fold=4.
- `v8_swizzle_xor` — XOR LDS swizzle.
- `v9_lds_epilogue` — CShuffle epilogue.

The DSL spec dataclasses produce one canonical name per spec via
`spec.kernel_name()` — use it as the canonical variant ID in your
sweep CSVs.

---

## 13. Operation-Specific Checklists

### 13.1 GEMM Checklist

- Is layout transposed or not? (`UniversalGemmSpec.layout`)
- Are operands contiguous along the vectorized dimension?
- Does tile shape match MFMA? (`MfmaAtom.shape`)
- Is K loop long enough?
- Are A/B staged through LDS?
- Are loads vectorized? (`probe_intrinsic_counts.py`:
  `global.load`/`buffer.load` count vs MFMA count)
- Is epilogue vectorized? (`probe_isa_inspect.py`:
  `buffer_store_dwordx2`/`_x4` vs `buffer_store_short`)
- Is split-K needed?
- Is Stream-K needed?
- Is persistent scheduling helpful?
- Are scales / bias fused efficiently? (`FusedEpilogue`)

### 13.2 Convolution Checklist

- Direct vs implicit GEMM decision (see Section 4.2).
- `N_gemm`, `K_gemm`, reduction length.
- Channel count small or large.
- Filter size (3×3 vs 5×5 vs 7×7).
- Padding / tails.
- Row streaming.
- Circular accumulators.
- Weight residency (registers vs LDS).
- Input LDS reuse.
- Q / spatial tiling.
- Group tiling.
- Output store mapping.
- Swizzle on `(spatial, C8)`.

### 13.3 Attention Checklist

- Tile Q / K / V.
- Online softmax (`OnlineSoftmaxState`).
- Mask handling (`causal_mask`, `sliding_window_mask`).
- Page table overhead (`transforms.indirect + unmerge`).
- KV cache layout.
- Split sequence / head (2D vs 3D).
- Register pressure from accumulators.
- Shared memory capacity.
- Coalesced V/O stores.
- LSE output.
- 2D vs 3D selection heuristic (`select_2d_config`, `use_2d_kernel`).
- ALiBi / QQ-bias paths exercised (per
  `notes/ATTENTION_PARITY_REPORT.md` 2D path was failing C9/C10/C11
  until the score-block fix landed).

### 13.4 Reduction Checklist

- Axis length.
- Rows per block.
- Vector width.
- Warp-level vs block-level (`block_lds_reduce` vs warp-only).
- Multi-pass.
- Numerical stability.
- Atomic or no atomic.
- Store width.

### 13.5 Fused Op Checklist

- Does fusion reduce memory traffic?
- Does fusion increase VGPR too much? (`probe_occupancy.py`)
- Can the epilogue be fused without scalarizing stores?
- Are scale / bias loads coalesced?
- Are activation approximations acceptable?
- Is intermediate precision visible?
- The DSL fusion pipeline (`helpers/fuse.py`) is the place; it
  routes through `FusionLegalizer.legalize(graph)` to verify
  LDS budget, dtype, vector width, and atomic-region rules before
  lowering.

---

## 14. Failure Modes

### 14.1 Correctness Failures

- Wrong lane mapping.
- Wrong vector element order.
- Missing boundary zero.
- Stale LDS data.
- Missing barrier.
- Wait without barrier for cross-thread data.
- Incorrect tail store.
- Wrong group / head offset.
- Wrong stride.
- Wrong K packing.
- Accumulator slot reset too early or too late.
- Compiler flag miscompile (Section 10.3).
- DSL AST rewriter elision (Section 10.3 — only relevant for
  tracing-style DSLs; CK DSL guards this via `Value.__bool__`).
- HIP-debug backend missing an op lowering (run
  `probe_lowering_compare.py` to detect).
- **`REGS_PER_LANE`-dependent invariant left over from a previous
  atom.** When switching MFMA shape (e.g. 16×16 → 32×32 quadruples
  `REGS_PER_LANE`), any code that writes only "slot 0" of a per-lane
  state tensor will silently cover 1/4 or 1/16 of the lanes. The
  sinks initialization bug in **§17.4** is the canonical example —
  it manifested as an elevated `max_abs_diff` localized to specific
  query positions while the rest of the output was bit-exact.
- **Spec-validator restriction outlives the constraint it protected.**
  A "to be safe" validator block that prevents two flags from
  combining can outlive the v1 restriction it documented; if the
  restriction is no longer real, the validator is silently blocking
  the winning code path. Audit each cross-flag restriction
  periodically.
- **HSACO module-cache aliasing across specializations.** When the
  display kernel symbol is the same across two compile-time-constant
  variants, the cache can serve the wrong HSACO. Salt the kernel
  symbol with a shape hash before compile (see §12.2).

### 14.2 Performance Failures

- Scalar stores (`probe_isa_inspect.py`: `buffer_store_short` >> 0).
- Scalar loads.
- Excessive barriers.
- Excessive waits.
- Register spills (`probe_occupancy.py`: `spill > 0`).
- Low occupancy (`probe_occupancy.py`: `waves_per_cu < 4`).
- Bank conflicts (`analyze_lds_conflicts.py`).
- Bad grid scheduling.
- Tiny per-block work.
- Too much LDS.
- Over-general runtime index math.
- Uncoalesced metadata loads.
- Over-fused epilogue.

### 14.3 Benchmark Failures

- Measuring allocation.
- Measuring initialization.
- Missing synchronization.
- Not enough iterations.
- Thermal throttling (lock with `rocm-smi --setperflevel high`).
- Clock changes.
- Cache-biased results.
- Comparing different shapes / layouts.
- Comparing different precision.
- Verification included in timing.
- First-run JIT compilation included.
- `time_launches` vs Triton autotuner interaction — use
  `probe_targeted_bench.py::time_cuda_event` instead when timing
  Triton in the same window.

---

## 15. Reporting Template

Use this table for experiment logs:

| Variant | Hypothesis | Correct | Time | Throughput | VGPR | SGPR | LDS | Notes |
|---|---|---:|---:|---:|---:|---:|---:|---|
| baseline | reference | yes | | | | | | |
| v1 | change one lever | yes/no | | | | | | |

Use this final summary:

```text
Best correct variant:
  name:
  shape:
  latency:
  throughput:
  correctness:
  resources:

Main bottleneck now:
  evidence:

Rejected ideas:
  - idea: reason

Next experiments:
  1.
  2.
  3.
```

For published numbers, attach the manifest + HSACO + `analyze_hsaco`
output (or equivalently, the `probe_occupancy.py` and
`probe_isa_inspect.py` reports). See `measured_results.md` for the
last documented validation pass.

### 15.1 Minimum Result Record

For every experiment, record:

```text
kernel/spec name
shape
dtype/layout
GPU and ROCm version
baseline commit/config
variant description
correctness status
max/mean error
latency median
latency spread
TFLOPS or GB/s
VGPR/SGPR/LDS if inspected
notable ISA changes
notes
```

If correctness fails, do not report speed as a win.

### 15.2 Done Criteria For An Optimization

An optimization is done when:

- it has a one-sentence hypothesis;
- correctness passes representative and adversarial shapes;
- benchmark improvement is stable across repeated runs;
- generated IR / ISA confirms the intended primitive changed;
- resource usage is recorded;
- docs or comments explain the new invariant if it is non-obvious;
- unsupported configurations are rejected by validation.

---

## 16. Decision Heuristics

### 16.1 Lever-direction heuristics

- If the output dimension is smaller than MFMA width, consider direct
  / specialized mapping (`DirectConv4cSpec`, `4x4x4` atom).
- If an epilogue uses scalar stores, vectorize it before deeper
  changes.
- If global-to-LDS uses register intermediate, try direct async copy
  (`pipeline = "compv4"`).
- If barriers dominate, increase work per barrier or reduce stages.
- If wider tiles slow down, inspect extra LDS passes and register
  pressure.
- If smaller workgroups speed up, occupancy was likely limiting.
- If smaller workgroups slow down, data sharing / barrier amortization
  may dominate.
- If swizzle does not help through register staging, it may still
  matter only with the matching async distribution.
- If LDS epilogue slows down, direct vector stores may already be
  good enough.
- If compiler flags produce wrong answers, remove them even if
  another codebase uses them safely.
- If the HIP-debug backend disagrees with LLVM-direct on a kernel,
  one of the two is missing an op lowering — file the issue,
  don't ignore it.

### 16.2 Anti-patterns (what to avoid)

- Chasing compiler flags before inspecting IR / ISA.
- Trusting one benchmark run.
- Comparing kernels with different math or masks.
- Comparing compile + launch time to warm launch time.
- Ignoring failed correctness because the speed number is attractive.
- Adding compatibility shims for unshipped experimental branches
  instead of fixing the builder.
- Promoting a kernel variant from a smoke set without a full-cohort
  sweep (§12.2, §17.4).
- Removing one cost without checking whether it moved to a worse
  place in the per-iter ISA composition (§9.3, §17.4).

---

## 17. Empirical Case Studies

For concrete, measured findings from real optimization experiments,
see `utilities/skills/empirical-case-studies.md`. It contains:

- Direct-conv progression from a scalar baseline to MFMA-tiled
  parity with the reference, with the exact lever per step
  (Case Study 1).
- LDS swizzle XOR vs padding comparison with measured deltas across
  multiple architectures (Case Study 2).
- Numerical tolerance signatures and bug patterns (Case Study 3).
- Stability and measurement caveats (Case Study 4).
- Closing the last few percent — what doesn't work (Case Study 5).
- Implicit GEMM vs pure GEMM overhead (Case Study 6).

Use these as reference points to set expectations and recognize bug
signatures. They are specific to particular experiments — actual
results depend on hardware, shape, dtype, and memory hierarchy
behavior. Always benchmark on your target hardware.

### 17.1 Bake-off Summary (DSL)

For the implicit-GEMM conv bake-off
(`example/ck_tile/dsl/08_bake_off_implicit_gemm`), the canonical
DSL example of applying the runbook's levers in series:

| Lever | Direction | Notes |
|---|---|---|
| baseline (single-buffer LDS, direct epilogue, smaller-K atom) | reference | `bad=0` at conv tolerance |
| + cshuffle epilogue (§9.3) | small improvement | LDS-staged fp16 + wide stores |
| + buffer-rsrc DW3 = `0x00027000` (§6.1) | correctness fix | was producing all-zero outputs |
| + larger-K MFMA atom (§7.1) | major improvement | halves the K-loop trip count |
| + K-padded LDS (§6.3) | small improvement | breaks `ds_read` bank conflicts |
| + graph mode / amortized launch (§12) | further improvement | reduces per-launch overhead |

Each lever was empirically verified with a hygienic benchmark, all
preserving `bad = 0` correctness at the conv tolerance against the
grouped NumPy reference. The cumulative result is a multi-× speedup
over the baseline, by applying five distinct runbook levers in
series.

For the direct-conv bake-off
(`example/ck_tile/dsl/09_bake_off_direct_conv_16c`, `10_…_4c`), the
two dominant levers were:

- Switching from scalar epilogue stores to vectorised stores
  (`buffer_store_dwordx2` per lane in the 4-channel path; `dwordx4`
  in the 16-channel path) — a major improvement on both shapes.
- Folding the inner K dimension into a larger-K MFMA atom for the
  16-channel path — a smaller but stable additional improvement.

These are the same two levers (epilogue vectorisation, larger-K atom)
that dominate every MFMA-tiled kernel's optimisation log.

### 17.2 Validation Pass Results

The documented validation pass at the time of writing exercises:

- The 245-test unit suite (`test_rocke.py`) — IR construction,
  transform DAG, helpers, instance smoke. Most run without a GPU; ~20
  harness/validation/timer tests require one.
- `verify_dsl_docs.py` — imports every symbol referenced by the docs,
  exercises every IR builder method, lowers every spec to LLVM / HIP
  / CK Tile, builds HSACO, launches small kernels.
- `test_rocke_examples.py` — discovers every `example/ck_tile/dsl/`
  manifest, builds it, runs `run_manifest --verify`, asserts the
  declared tolerances.
- The bake-off 08 implicit-GEMM manifest — verifies bit-level
  correctness at the conv tolerance plus the declared TFLOPS / GB/s
  lower bounds.
- `ck_tile_parity.py --op all` — small-op parity vs torch across the
  20 cases listed in §13.3.
- `parity_extended_kernels.py --op all` — FMHA / sage / sparse / MoE
  / block-scale / MX parity vs torch / NumPy references.
- `parity_unified_attention.py` — Triton + reference vs CK DSL across
  the documented attention scenarios.

See `measured_results.md` for the latest documented validation pass
numbers; it is updated each time a major verification sweep lands.

### 17.3 Attention Parity Pass

A documented attention parity sweep covering the FMHA family
(`parity_extended_kernels --op all`), the unified-attention 2D / 3D /
auto lanes (`parity_unified_attention.py`), and the full
LLVM ↔ HIP-debug lowering audit (`hip_lowering_parity --case all`)
all passed end-to-end after two material correctness fixes landed:

- **2D ALiBi / QQ-bias correctness fix.** The transposed-32x32
  softmax block was skipping ALiBi / QQ-bias addition entirely. The
  symptom was `max_abs` *of order 1* on three scenarios while every
  other scenario stayed at the expected `fp16` ULP tolerance — a
  classic structural-bug signature (the score wasn't actually
  computed; it was missing an addend entirely). The fix wires both
  into the score computation inline before the per-row max reduce.
- **HIP-debug backend ops.** The HIP-debug lowering was missing a
  handful of vector ops (`vector.add`, `vector.mul`) and packed
  conversions (`cvt_pk_{fp8,bf8}_f32x4`). Production LLVM lowering
  was unaffected; the parity gap was purely a backend coverage gap.

Take-away: the parity-of-order-1 failure mode is structural, not
numeric noise. Always check whether a missing addend / mask term in
the score path explains an order-of-magnitude max_abs jump before
suspecting MFMA / lane-layout bugs (whose signatures are in the
1e-2 — 1e-3 range, see §1.4 and Case Study 3 in
`utilities/skills/empirical-case-studies.md`).

### 17.4 Unified Attention 2D Optimization Pass

A multi-day case study of closing a substantial latency gap between
the `UnifiedAttention2DTiledSpec` kernel and a Triton reference across
a production-trace cohort of bf16 prefill-2D shapes on gfx950. The
methodology, levers, and pitfalls below are written to be transferable
to any kernel optimization that targets a known-better reference.

This is the worked example for the "profile first, change second"
methodology under §3, the LDS principles in §6.3 / §6.4, the
MFMA-atom selection in §7, the pipelining choices in §8, and the
dispatcher / autotuning discipline in §12.

#### Methodology lessons

**Profile first; PMC fallback when ATT decoder unavailable.** On a
host without `rocprof-trace-decoder`, the §3.1a ATT path is
unreachable. The equivalent diagnostic deliverables come from:

1. PMC counters in one `rocprofv3` pass:
   `MfmaUtil`, `VALUBusy`, `LDSBankConflict`, `MemUnitStalled`,
   `MeanOccupancyPerActiveCU`.
2. Static ISA from the dumped HSACO via `llvm-objdump -d --mcpu=gfx950`
   (use `probe_isa_inspect.py` or `count_instructions.py`).
3. Kernel-stats header from the same `rocprofv3` pass for VGPR / AGPR
   / SGPR / LDS / grid / workgroup (or `probe_occupancy.py` statically).

The worst-shape PMC reading on this kernel showed the classic
"compute-throttled by VALU/LDS plumbing" signature: single-digit
`MfmaUtil`, dominant `VALUBusy`, elevated `LDSBankConflict`, and
near-zero `MemUnitStalled`. HBM was definitively ruled out by the
last metric, so every bandwidth-saving lever (FP8 KV storage, larger
tiles for reuse) was demoted immediately.

**Read the reference implementation before knob-sweeping.** The
original plan had a long knob-sweep table (`num_warps`, `tile_size`,
`kv_storage_dtype`, etc.). PMC plus a side-by-side read of the
reference's source revealed that **both kernels were already running
with identical launch config** — same `BLOCK_M`, `TILE_SIZE`,
`num_warps`, `waves_per_eu`, and workgroup size. The gap was
structural — in **how the loop body was realized**, not in the
launch knobs. The original sweep would have moved nothing.

**HSACO per-iter mnemonic histogram is the design-diff gold standard.**
Once one source said "P stays in registers" and the other said
"P round-trips through LDS", the way to confirm was a per-iter
mnemonic count from `llvm-objdump`. The reference emitted half as
many (but larger-K) MFMA instructions, no `ds_write_b16` P-spill, no
AGPR↔VGPR shuffles around the accumulator, and a fraction of the
total K-loop instructions of the original DSL kernel.

The opcode delta — counted, not estimated — is what makes structural
differences explicit. Latency numbers compress too much.

#### Structural change ladder (lever family by lever family)

The pass that closed most of the gap stacked structural changes in
roughly this order. Each row is one lever; the cumulative effect is
the union, not the sum.

| Lever family | Concrete change | What it removes / unlocks | DSL knob (§12.1) |
|---|---|---|---|
| MFMA atom shape | switch QK / PV from the 16×16 atom to the 32×32 atom that natively aligns its C-output with its A-input | the cross-lane re-pack between QK and PV that the 16×16 chain needs | `use_mfma_32x32`, `use_transposed_qk_32x32` (§12.1.C) |
| Intermediate residency | keep softmax P in registers instead of spilling through LDS | the P-LDS allocation, the per-iter `ds_write_b16` stripe, and the in-loop `s_barrier` | `use_register_pv` (§12.1.K) — only wins when paired with the atom + reader changes below |
| LDS reader path | use the transposed-tile LDS reader (`ds_read_tr16_b{64,128}`) for V | the per-iter `v_readlane / v_writelane` register-side reshape | `use_transposed_pv_tr_read` (§12.1.K) |
| Lane ownership | each 32-lane half consumes only the K rows it already owns; matching half-local V reads | one cross-half reshape per iter | `use_transposed_half_local_pv` (§12.1.K) |
| Schedule | issue V async copy *before* QK so V overlaps both QK and softmax (not just softmax) | latency exposed when V was issued mid-loop | `use_early_v_schedule` (§12.1.D, §8.1) |
| Tile size matched to useful work | for sliding-window shapes, halve `tile_size` so each K-iter doesn't waste half the loop on masked-out columns | the wasted K-loop work, the wasted LDS reads, the wasted waits | `tile_size` (§12.1.B), `_select_2d_tile_size` (§12.1.O) |
| Dispatcher per-shape branch | one selector picks the early-V variant for no-sliding-window shapes, a different variant for sliding-window | the long-tail of shapes a single variant can't win | `_select_*` / `_enable_*` (§12.1.O) |

The structural insight at each step:

- **Atom shape**: the 32×32 atom is structurally cheaper *not*
  because it has higher arithmetic throughput — it has the same
  throughput as the 16×16×32 atom — but because its **C-output lane
  layout natively matches its A-input lane layout**, so the QK→PV
  chain skips the re-pack the 16×16 atom requires. (§7.2.)
- **Register-PV alone is a net loss** with the wrong atom: removing
  the P→LDS spill replaces a `ds_write_b16` stripe with a much
  larger cross-lane permute storm. The cost of a structural change
  is rarely visible until you look at the per-iter ISA composition.
  (§9.3 caveat.)
- **A structural change often needs multiple co-evolved levers** to
  actually win. The atom + register-residency + transposed reader
  trio works together; any subset of two loses to the original.
- **Cross-lane reshape can be sidestepped by reorganizing what each
  lane half owns**, not just by removing the reshape. (§5.3.)
- **Latency-hiding scheduling helps when compute is ahead of memory.**
  If the compute window is already saturated, issuing memory earlier
  just enlarges the LDS budget without helping. (§8.1.)
- **Tile size must match the useful work per tile**, not the LDS
  budget's maximum. For sparse masks (sliding window), a smaller
  tile cuts wasted work proportionally. (§13.3.)
- **The dispatcher is a perf lever.** A single kernel variant rarely
  wins every shape; per-shape branches close the long tail. (§12.3.)

#### Things that did NOT help (and why)

| Attempted lever | Outcome | Why it failed |
|---|---|---|
| FP8 KV storage (`kv_storage_dtype="fp8e4m3"`) | neutral / regression | `MemUnitStalled` was near zero; HBM was not the bottleneck. FP8 dequant adds VALU work, which **was** the bottleneck. |
| Doubling `tile_size` past the default | regression | Doubled LDS-per-WG, violated the per-CU LDS ceiling, dropped occupancy below 2 WGs/CU. |
| `num_warps` sweep | flat | Already at the reference's pick. Adding warps without fixing the LDS / AGPR plumbing only amplifies the conflicts. |
| Two KV tiles per accumulator update (`use_grouped_kv2_softmax`) | smoke set looked promising; full cohort regressed materially | Always run the full cohort, not a smoke set. |
| Specialized sliding-window prefill wrapper | flat or regression | Slower than the small-tile variant on every sliding-window shape it covered. |
| Force VGPR-form MFMA (`use_agpr_alloc_zero`) | flat | The chosen path already had zero AGPR moves; the lever was redundant. |
| Compiler hint sweep (`waves_per_eu`, `maxnreg`, scheduling modes) | sub-1 % movement | Compiler flags very rarely close large gaps (§10.2). |
| Register-P proxy variant (kept allocation removal without changing the read path) | within measurement noise | HSACO diff showed identical hot-loop ISA — the only "change" was removing an unused LDS allocation that the compiler had already dead-code-eliminated. **A change that doesn't move the per-iter ISA composition almost certainly didn't change anything.** |

#### Diagnostic signatures collected from this pass

These are the qualitative PMC + ISA patterns that mapped to each
finding. The thresholds are approximate — calibrate to your kernel
family before relying on them.

```text
"compute-throttled by VALU/LDS plumbing, not HBM-bound"
    MfmaUtil           well below the architecture's healthy range
    VALUBusy           dominant (most of the cycles)
    LDSBankConflict    above the action threshold (~5 % of LDS cycles)
    MemUnitStalled     negligible (<1 %)
    → fix the LDS / AGPR plumbing, not the launch knobs

"Intermediate tile round-trips through LDS"
    ds_write_b<N> per outer-iter ≈ (tile_bytes / threads)
    LDSBankConflict elevated for narrow stripe writes (b16 on
        wide-bank LDS)
    s_barrier present in the inner loop
    → consider register residency for the intermediate

"MFMA atom lane-layout mismatch with the next atom in the chain"
    (v_accvgpr_read + v_accvgpr_write) / MFMA elevated
    ds_swizzle_b32 or ds_bpermute_b32 in the inner loop
    cross-lane permute count per iter dominates
    → switch to an atom shape whose C-out matches the next A-in
```

#### Pitfalls and gotchas

- **Spec-validator restrictions can mask the actual win path.** The
  `attention_tiled_2d.py` validator initially raised on
  `use_register_pv` + `use_mfma_32x32` together; lifting that gate
  was the prerequisite to the winning combination. Validators added
  "to be safe" for a v1 limitation can outlive the limitation. Always
  ask whether the constraint reflects real numerics or just a
  previous caution.
- **HSACO module-cache aliasing.** When two kernel specializations
  share the same display symbol but differ in compile-time constants
  (e.g. `num_seqs`, binary-search trip count), the runtime can serve
  the wrong cached HSACO. Fix: salt the kernel symbol with a shape
  hash before compilation.
- **`REGS_PER_LANE`-dependent invariants must be audited when the
  MFMA atom changes.** Code that writes to a fixed slot index of a
  per-lane state tensor can be correct for one atom's
  `REGS_PER_LANE` and silently broken for another's. The atom switch
  in this pass surfaced an elevated `max_abs_diff` localised to
  specific query positions — caused by a sinks-init loop that
  covered only a fraction of the new per-lane state. Always re-test
  sinks / masks / initialisation paths when changing the atom.

#### What the remaining gap looks like

After the structural ladder above, the best kernel's K-loop still
emits, per outer iteration, far more AGPR↔VGPR moves, far more
`ds_bpermute_b32`, far more `s_waitcnt`, and several × the VALU ops
of the reference. These are **structural MFMA-pipeline +
softmax-realisation differences** — not knob-flippable. Closing them
requires a different lane layout for the QK output that natively
aligns with the PV A-operand (zero AGPR shuffle), plus a softmax
pattern that uses the cheap CDNA cross-lane VOP1 instead of the LDS
DMA path. This is exactly the "Closing the Last 5 %" pattern from
Case Study 5 in `utilities/skills/empirical-case-studies.md`.

#### Take-away principles (transferable to other kernels)

1. **A static probe is faster than a sweep.** A short PMC pass plus
   a `probe_isa_inspect.py` run can disprove several knob-sweep rows
   at once.
2. **Read the reference implementation's source.** It is faster than
   inferring from PMC alone what the reference is actually doing
   differently.
3. **Per-iter ISA histogram is the unit of design comparison.**
   Latency compresses too much; opcode counts make structural
   differences explicit.
4. **One lever is rarely the whole change.** Co-evolve two or three
   levers when the first lever's apparent regression is caused by
   an exposed downstream cost.
5. **The dispatcher is a perf lever.** A single kernel variant
   cannot win every shape; per-shape branches close the long tail.
6. **Validators encode assumptions, not laws.** Re-examine each
   restriction whenever you cross a major design boundary.
7. **Things you remove are not free.** Always check whether removing
   a cost (LDS write, AGPR shuffle, barrier) is being paid for
   elsewhere (cross-lane permute, VGPR pressure, occupancy drop).
8. **Full cohort, not smoke set.** A change that wins on 4-8 shapes
   may regress on the full production-shape distribution.

### 17.6 Fused MoE Active-Tile Dispatch

After Round 10's preshuffle work, an `(E ∈ {2,4,8,16}, topk=2)`
sweep on `decode_T1_H4096_I7168` showed e2e time scaling ~linearly
with `experts`, not with `min(experts, topk*tokens)`:

```text
E= 2  active=2  preshuf  262.2 us
E= 4  active=2  preshuf  281.0 us  (+18.8 / +9.4 per inactive)
E= 8  active=2  preshuf  363.5 us  (+82.5 / +13.7 per inactive)
E=16  active=2  preshuf  575.7 us  (+212.2 / +15.2 per inactive)
```

CK Tile avoids this waste with an active-tile dispatcher: each CTA
reads `sorted_expert_ids[block_id_z]` to pick its expert and
returns early when the sorted-tile id exceeds `num_sorted_tiles`.
Round 11 ports the idea into the DSL kernels.

**Implementation surface.**

1. `instances/gemm_universal.py::TraitSpec.active_tile_skip`: new
   field, default `False`. `kernel_name` picks up an `actt` flag.
2. `build_universal_gemm`: when `batched and active_tile_skip`,
   declare two extra params (`SortedTokenIds: ptr<i32>`,
   `slot_size: i32`). Compute
   `bucket_head = block_id_z * slot_size + block_m_off`,
   `do_work = SortedTokenIds[bucket_head] >= 0`. Wrap the K-loop +
   epilogue in `scf.if(do_work)`. Using `block_m_off` (already
   chiplet-swizzle-aware, tile_m-aligned) keeps the gate
   consistent with the address arithmetic the body actually uses.
3. `build_moe_interleaved_gate_up_silu_gemm`: same gate, same
   `block_m_off` form.
4. `instances/batched_gemm.py::batched_gemm_signature` and
   `moe_interleaved_gate_up_silu_gemm_signature`: append the two
   extra args when `trait.active_tile_skip`.
5. `instances/fused_moe_e2e.py::FusedMoeForwardSpec.active_tile_skip_gemms`:
   orchestrator-side knob. A parameterized launcher cache
   (`_moe_batched_gemm_launcher`,
   `_moe_interleaved_gate_up_silu_launcher`) returns the right
   HSACO for any combination of `(preshuffle_b, active_tile_skip)`
   on demand, so the four binary variants per kernel family share
   one cache.
6. `examples/gfx950/moe/test_active_tile_skip.py`: standalone parity +
   perf harness. Confirms bitwise-equal output to baseline when
   all tiles active, zero output for inactive tiles, ~14× speedup
   when every tile is inactive.

**Measured outcome.**

Standalone kernel (B=8, M=32, N=4096, K=4096, fp16):

```text
base no skip       172.31 us
att, all-active    173.85 us   (0% overhead)
att, all-inactive   13.10 us   (13× faster — kernel just exits)
```

End-to-end MoE (HIP-graph-replayed, fp16):

```text
                                baseline   preshuf_intl   +ATS    vs base
decode_T1_E8  H=4096 I=4096     269.4 us   227.5 us       166.3   1.62×
decode_T1_E8  H=4096 I=7168     412.7 us   362.8 us       264.2   1.56×
decode_T8_E8  H=4096 I=4096     488.4 us   229.1 us       227.5   2.15×
decode_T8_E8  H=4096 I=7168     382.5 us   365.0 us       351.8   1.09×
decode_T1_E16 H=4096 I=7168     620.1 us   544.9 us       264.0   2.35×
```

vs CK Tile C++ on the canonical decode (router in DSL timing):

```text
decode_T1_E8_K2_H4096_I7168 :
    rocke baseline     0.379 ms  (0.32× of cktile)
    rocke preshuf      0.357 ms  (0.34× of cktile)
    rocke preshuf+ATS  0.265 ms  (0.46× of cktile)
    ck_tile_cpp         0.121 ms
```

Closed roughly 30 % of the gap to CK Tile on the canonical
decode_T1, and ~50 % on `decode_T1_E16`. Bitwise parity
(`rel=0.00e+00`) across every standalone and e2e variant in
`test_active_tile_skip.py`, `test_fused_moe_preshuffle.py`,
`test_preshuffle_b.py`.

**Lessons reinforced.**

* **Old breakdowns lie.** The runbook §17.5 / Round 1 attribution
  ("`fused_moe_reduce` is 11 % / 46 µs") was stale. Clean
  HIP-graph timing measured the reduce kernel at ~9 µs alone and
  ~3 µs in chain. After preshuffle, the GEMM kernels (gate_up +
  down) are ~99 % of the per-replay time. Always re-measure the
  hot-kernel attribution after a meaningful structural change.
* **Launch overhead is not the problem on these shapes.** Eager
  forward and HIP-graph replay are within 2 µs of each other
  (407 µs vs 410 µs baseline; 351 µs vs 356 µs preshuf). Mega-kernel
  fusion will not help because of fewer launches; it helps
  because of fewer HBM round-trips and shared register state.
* **Active-tile dispatch is independent of preshuffle and stacks
  with it.** Both are valid levers for different reasons:
  preshuffle reduces per-K-tile B-load time; ATS reduces the
  number of CTAs doing useful work. They compose multiplicatively
  on the right shapes (`decode_T1_E8`: 1.62× combined vs ~1.07×
  preshuf alone).
* **`block_m_off` not `block_id_y * tile_m`.** When a kernel does
  any tile-id remap (chiplet swizzle, persistent grid, etc.) the
  gate must read the bucket head for the *post-remap* row, or it
  will skip a tile the body still tries to compute (or vice
  versa). Use whatever value the kernel uses to address A and C.

---

### 17.5 Fused MoE Preshuffle-B Implementation

The fused-MoE optimization pass (`examples/gfx950/moe/`) ran 9 rounds of
config-level tuning before stalling at the configuration ceiling.
Round 9 named `preshuffle_b=True` as the genuinely-untried lever
from §12.1.H but discovered the flag was a documented-but-silently-
ignored knob: declared in `gemm_universal.py::TraitSpec` but never
read by `build_universal_gemm()`. Round 10 closes that gap.

**The problem.** With the canonical row-major B layout
`(N, K)`, each per-K-tile B-load in the MFMA inner loop visits
`block_n` rows of B at offsets `K` apart. A wave's lanes hit
different rows; the GPU coalescer can only batch a partial wave
into a single `buffer_load_dwordx4`, so each K-step costs multiple
discontiguous VMEM transactions.

**The transform.** Pre-shuffle B once on the host into
`(E, k_tiles, n_tiles, block_n, block_k)` contiguous, where
`k_tiles = K / block_k` and `n_tiles = N / block_n`. The
`(block_n × block_k)` tile that the per-K-tile B-load wants is now
exactly `block_n × block_k × elem_bytes` consecutive bytes. One
wide contiguous burst per warp replaces the strided per-row loads.
The in-tile element order matches `B_smem`'s row-major
`(block_n, block_k)` layout, so the MFMA inner loop's `ds_read`
pattern is unchanged.

**Implementation surface.**

1. `gemm_universal.py::emit_load_phase`: add a
   `if spec.trait.preshuffle_b:` branch that computes
   `tile_offset = (k_tile * n_tile_count + n_tile) * (block_n *
   block_k)` and issues
   `b.global_load_vN(B, base_off + vec_idx * load_vec, dtype,
   load_vec)` directly (bypassing `TileWindow` which models the
   strided 3-D view).
2. `moe_gemm_fused.py::build_moe_interleaved_gate_up_silu_gemm`:
   same branch; the only delta vs `gemm_universal` is the
   `n_tile_count = (2*N) / block_n` (the GEMM N is `2*N` because
   gate and up are packed along the N axis).
3. `gemm_universal.py::UniversalGemmSpec.kernel_name`: append a
   `preb` flag so HSACO caches don't alias.
4. `fused_moe_e2e.py`: add three orchestrator knobs
   (`preshuffle_w_down`, `preshuffle_w_gate_up_packed`,
   `preshuffle_w_gate_up_interleaved`), three host-side
   preshuffle helpers with data-ptr-keyed caches, and pre-build
   the preshuffled tensors inside `capture_graph` *before* the
   warmup loop so the one-time `torch.cat / permute / contiguous`
   cost stays out of the captured / replayed region.
5. `examples/gfx950/moe/test_preshuffle_b.py`: standalone batched-GEMM
   parity + perf harness that flips `preshuffle_b` and confirms
   bitwise-equal kernel outputs (`max|delta|=0.0` between the
   two paths) plus per-shape timing.
6. `examples/gfx950/moe/test_fused_moe_preshuffle.py`: end-to-end harness
   that runs five FusedMoeForward configurations
   (`baseline_interleaved`, `baseline_packed`, `preshuf_down_only`,
   `preshuf_packed_full`, `preshuf_intl_full`) per scenario with
   subprocess-style `_isolate_lane()` between configs, and
   confirms parity-with-baseline (`rel=0.00e+00`) for every
   variant.

**Measured outcome.**

Standalone batched-GEMM at the production tile (32 × 128 × 64,
fp16): 1.5–2.1× speedup across the canonical decode shapes. Some
sample points:

```text
B=8  M=512  N=2048 K=4096  : 232.35 → 151.03 us  (1.54×)
B=16 M=256  N=4096 K=4096  : 613.14 → 318.12 us  (1.93×)
B=32 M=128  N=4096 K=4096  : 670.52 → 346.02 us  (1.94×)
B=8  M=128  N=2048 K=4096  : 222.69 → 105.26 us  (2.12×)
```

End-to-end (HIP-graph-replayed) on real MoE shapes (E=8 K=2,
fp16, with all three preshuffle knobs enabled at the optimum):

```text
                                  baseline    preshuf_intl   speedup
decode_T8 H=4096 I=4096           274.14 us   231.73 us      1.18×
decode_T1 H=4096 I=4096           267.98 us   226.44 us      1.18×
decode_T1 H=4096 I=7168           384.41 us   360.77 us      1.07×
decode_T8 H=4096 I=7168           385.05 us   367.78 us      1.05×
```

vs CK Tile C++ on the production-canonical decode scenario:

```text
decode_T1_E8_K2_H4096_I7168 :
    baseline 0.388 ms (3.20× of cktile)
    preshuf  0.355 ms (2.93× of cktile)
    cktile   0.121 ms
```

Closed ~7 % of the gap to CK Tile on the canonical shape. The
larger 18 % wins on the I=4096 shapes show the lever pays off in
proportion to how much per-K-tile B-load time the kernel was
spending: I=7168 has flatter K-axis utilization (more N tiles per
CTA), so the in-tile B-load is a smaller fraction of kernel time.

**Lessons reinforced for §17.6.**

* **Documented-but-unimplemented knobs are real lurkers.** `preshuffle_b`
  was a `TraitSpec` field for many releases; nothing in the build
  path read it. The first round of optimization noticed
  `preshuffle_b=True` was a no-op via HSACO byte-equality and
  IR-line equality. That is exactly the §17.4 "diff the lowered IR
  to verify a flag is honored" guidance applied to a config knob.
* **Bitwise parity is the right correctness bar.** All five
  end-to-end variants produced identical Y tensors
  (`max|Y_pre - Y_base| = 0`). This rules out a subtle layout-
  mismatch bug that an `atol`-based tolerance would have hidden.
* **Shape-dependent wins, again.** The I=4096 case sees 18 %; the
  I=7168 case sees 5–7 %. The kernel-level speedup is real on both
  but the share of e2e time spent in the GEMM B-load differs,
  which is the §17.4 corollary "a change that wins on 4-8 shapes
  may regress on the full production-shape distribution" applied
  to a *non-regressing* knob: the parity is preserved and the
  speedup magnitude varies, so the knob is a per-scenario lever in
  the §15 dispatcher rather than an unconditional default.
* **Mega-kernel fusion is still the next big lever.** With
  preshuffle_b in place across both GEMM bodies (universal +
  interleaved gate-up), the remaining 3× to CK Tile is the same
  structural gap: gate / up / SiLU / down / weighted-reduce
  living in one kernel with shared register state and an
  in-kernel grouped-GEMM dispatcher. That is multi-week kernel
  authoring, not a session-scoped pass — captured here as the
  explicit follow-up.

---

### 17.7 Skinny-M Decode GEMM (o_proj M=2)

Case study from `examples/dsl_o_proj_decode/`. The matmul is Qwen3-8B's
`o_proj` decode: `bf16`, `M=2`, `N=4096`, `K=4096`, MI355X / gfx950, 8 TB/s
HBM peak. rocBLAS sits at 10.10 µs (best) / 42 % HBM efficiency. The
exercise: do whatever it takes to match or exceed rocBLAS, end-to-end
through the runbook loop.

**Final result.** 10.29 µs / 44 % HBM / 1.02 × rocBLAS. Tied at the
HBM-saturation ceiling. Bytes-per-µs is actually *better* than rocBLAS —
rocBLAS just moves slightly fewer bytes. To go further, the lever leaves
the GEMM (fusion with the next op so C never round-trips through HBM).

**Step ladder (selected).** Each row is one runbook lever; numbers are best
of N=10 attempts × 500 iters with the §2.3 hygiene rules.

| Step | Lever | µs    | × rocBLAS | Why |
|----|---|------:|----------:|---|
| 7  | tile_k=1024 without DTLA | 58.0 | 5.74 × | Register pressure: A+B in VGPRs at 1024 K-elements blows the budget, spills, collapses occupancy to 1 wave/CU |
| 13 | + `direct_to_lds=True` | 10.51 | 1.04 × | Frees 32 VGPRs (no A/B round-trip through registers), restoring tile_k=1024 to 2 waves/WG |
| 14 | DTLA cache-hint sweep on A, B | 10.51 | 1.04 × | CACHE_ALL / CACHE_ALL wins on both operands — `CACHE_STREAM` regresses even on the 32 MB B (the 256 CTAs each see a unique N-stripe but consecutive K-tiles within a CTA do reuse L2) |
| 18 | + full DTLA ping-pong prefetch (double-buffered LDS, runtime parity via scf.for iter-arg, `s_waitcnt vmcnt(loads_per_tile)`) | 10.53 | 1.04 × | **Negative result.** Implementation is bit-exact and the prefetch fires, but perf is unchanged. We're not vmcnt-stalled per tile — we're at the system-wide outstanding-loads ceiling already at 1 wave/WG × 256 CUs |
| 19–20 | Fix multi-warp DTLA correctness (per-wave LDS base offset) + sweep wider tiles | 14.0–39 | 1.4–3.8 × | **Negative result.** Multi-warp now correct but slower: wider `tile_n` cuts grid_x proportionally; at `tile_n=32` we drop from 256 → 128 CTAs over 256 CUs and exactly half the GPU goes idle |
| 21 | + `chiplet_swizzle=True, chiplet_wgm=8, chiplet_chunk_size=16` | **10.29** | **1.02 ×** | The 2.6 % win. Consecutive WGs land on the same XCD so each XCD's L2 sees correlated traffic. The other 17 knob combinations cluster between 10.29 and 10.76 µs |

**Lessons reinforced.**

* **The DTLA validator's `cols_per_chunk = halves_per_chunk` assumption is
  attention-specific.** It breaks the moment you try a GEMM tile that isn't
  a single hardware chunk wide (here: `tile_m=16, tile_k=512+`). The fix
  is to call `async_buffer_load_lds_addr` directly with a manually
  distributed pass plan rather than going through `AsyncTileLoader`. See
  `gemm_universal.py:684-820` for the GEMM-shaped form.
* **`async_buffer_load_lds_addr` is wave-level, not lane-level.** Every
  wave writes `wave_size × BYTES_PER_LANE` lane-contiguous bytes starting
  at the wave-uniform `lds_dst`. With multiple waves per WG, each wave
  must target a different LDS slice or they stomp. Add
  `warp_id * wave_size * BYTES_PER_LANE` to the LDS base before the load
  passes (`gemm_universal.py:776-795`). Diagnostic signature: multi-warp
  output is wrong by megabits-of-absdiff, but single-warp is bit-exact and
  the non-DTLA multi-warp path is bit-exact too.
* **Cheap tricks that look like wins are often noise.** Steps 15 and 17
  tested `waves_per_eu` hints over 5 then 20 attempts at 200 then 500
  iters; the apparent 0.07 µs lift at `wpe=4` was inside the stdev. Be
  paranoid about lifts smaller than 1 %; re-run with paired attempts and
  more iters before declaring victory.
* **Prefetch only helps when something is waiting.** §8.2 software
  pipelining and §8.3 `s_waitcnt vmcnt(N)` are real levers when the kernel
  is per-tile vmcnt-stalled. They are *not* levers when the kernel is at
  the system-wide HBM-controller ceiling. Diagnostic: if best ≈ median
  with std < 1 % across the simple-loop variant, more prefetch will not
  help.
* **HBM efficiency is the cap on skinny-M.** With `M=2, N=4096, K=4096,
  bf16`, B alone is 32 MB and each CTA sees a unique N-stripe — there is
  no cross-CTA B-reuse. Total HBM ≈ 33.5 MB; sustained HBM on MI355X for
  streaming bf16 reads tops out near 3.5 TB/s ≈ 44 % of the 8 TB/s
  marketed peak. Both our kernel (44 %) and rocBLAS (42 %) live at this
  ceiling. Beating rocBLAS at this HBM efficiency means reducing total
  HBM bytes — kernel fusion territory (§4.1, §13.5), not GEMM tuning.
* **Trust your rocBLAS reference only in paired hygiene.** Step 22's
  initial back-to-back run measured rocBLAS at 9.63 µs — but only because
  it was running with a warm L2 from our prior DSL kernel. Standalone
  rocBLAS lands at 10.10 µs. Always re-bench the reference in the same
  process and hygiene as the kernel under test.
* **Multi-warp DTLA isn't necessarily a perf lever — but it's a
  correctness fix.** The per-wave-LDS-offset patch in step 20 doesn't help
  *this* shape (CTAs already saturate CUs), but it removes a latent bug
  for any future user who pairs DTLA with `warp_n > 1` (a typical mid-M
  GEMM configuration).

**Reproducible commands.**

```bash
cd examples/dsl_o_proj_decode
python scripts/13_dtl.py       # DTLA + cache-hint sweep
python scripts/18_dtl_prefetch.py
python scripts/19_multiwarp_probe.py   # isolates the wave-stomping bug
python scripts/21_chiplet.py    # the 2.6 % win
python scripts/22_confirm_winner.py    # 20 × 1000 paired vs rocBLAS
```

### 17.5 Square fp16 GEMM toward matrix peak (gfx950)

Full study: `examples/gfx950/gemm_perf_square_warpspec/README.md`. The compute-
bound counterpart to §17.4. Climbed a stock `compv4` kernel to **~0.82× rocBLAS**
(same tile geometry, **0 LDS bank conflicts**) one measured lever at a time, and
characterized the residual gap down to a single missing capability.

Distilled rules (also folded into §8.6):
- **Collapse to one barrier per K-tile.** A direct-to-LDS depth-2 ping-pong only
  needs one `s_barrier_bare` if the next-tile async write is issued *after* the
  barrier (one `s_waitcnt(vmcnt=0,lgkmcnt=0)` covers RAW+WAR). Biggest single win
  (+59%); the naive two-barrier form halves the MFMA shadow at 1 WG/CU.
- **Element-granular swizzle for 0 bank conflicts** — `col ^= ((row>>R)&(2^W-1))<<L`,
  `L=log2(elem)`, `W=log2(block_k/elem)`, `R=0`. Key on the low `m_in_atom` bits
  (the row·stride term vanishes mod 32 banks); high-bit/2-slot swizzles floor at
  25%. Bit-exact (same XOR on store-source + ds_read).
- **`dtl_cache_b=CACHE_ALL` for reuse-heavy GEMM** (+9%) — corroborates the §17.2
  item-14 cache sweep; the `CACHE_STREAM` default is for one-shot operands only.
- **Diagnose the bound first (§8.6).** At 1 WG/CU with 0 bank conflict the kernel
  is *barrier-bound*: `iglp_opt` and every IR-level scheduling hint are neutral.
  Built the rocBLAS register-staged PGR2 path to test the bound directly — it is
  *correct but 0.42×* (the `ds_write` it requires can't be hidden without
  assembly-grade scheduling). That pins the IR-path ceiling: ~0.82× rocBLAS, the
  rest needing instruction placement comgr cannot do.
- **Measure same-session.** Auto-clock swings ±25–30%; only candidate-vs-reference
  ratios benched back-to-back in one process are meaningful.

---

## 18. DSL Probe Workflow

This section is the runbook's how-to for the static-inspection probes
under `utilities/tools/dsl_probes/`. Skip rocprof, get a signal in
under one second, narrow the variant list, then pay the rocprof cost.

### 18.1 Standard Diagnostic Sequence

For a new kernel or a regression:

1. **Build smoke**. Build every variant with `probe_config_sweep.py`
   `--only-build` and look for `SPEC-FAIL` or `BUILD-FAIL`. A common
   trap is that two spec fields are coupled (e.g. `use_mfma_32x32`
   requires `block_m_per_warp=32`); the spec dataclass raises
   `ValueError` and the probe reports it as `SPEC-FAIL` without
   killing the sweep.
2. **Resource check**. Run `probe_occupancy.py` over every variant.
   Look for:
   - `spill > 0` (always fix first; it invalidates perf comparisons).
   - `limited_by` distribution. If half the variants are `VGPR` and
     half are `LDS`, the optimal tile sits at the boundary.
   - waves/CU dropping below 4 — usually means you need `pipeline =
     "lean"` or a smaller tile.
3. **Intrinsic check**. Run `probe_intrinsic_counts.py` over the two
   variants you most expect to differ. Confirm:
   - The MFMA atom changed when you flipped `use_mfma_32x32`.
   - `raw.ptr.buffer.load.lds` is non-zero when you set
     `pipeline="compv4"` (async DMA active).
   - `ds.read.tr16.b64` or `ds.read.tr16.b128` appears when you used
     a transpose-LDS path (`b.ds_read_tr16_*`).
   - `s.barrier` count went down after a pipeline change.
4. **ISA check (only if intrinsic check is inconclusive)**. Run
   `probe_isa_inspect.py` to see the post-codegen instruction mix.
   This is where you confirm that vector stores actually emit
   `buffer_store_dwordx{2,4}` (and not `buffer_store_short`), that
   the cshuffle epilogue is using wide stores, and whether a runtime
   `s_sched_barrier` instruction is present on gfx950 (usually it is
   not — `b.sched_barrier` is a compile-time fence that constrains the
   scheduler without leaving a runtime instruction; absence here does
   *not* mean the directive had no effect — see Section 8.4).
5. **Lowering parity check (only if HIP debug parity is failing)**.
   Run `probe_lowering_compare.py` — if the HSACO sizes diverge >2×
   or the VGPR/LDS deltas are large, one backend is missing an op
   lowering.
6. **Best-of-sweep on production shapes**. Once the variant list is
   narrowed by static signal, run `probe_targeted_bench.py` on the
   production-trace shape set with `candidate_fn` (your best variant)
   and `baseline_fn` (Triton / AITER / CK Tile).
7. **rocprof confirm**. Pick the top 1-2 variants from step 6, run
   `rocprofv3 -i metrics.txt -- python probe_rocprof_single.py …`,
   then parse with `analyze_lds_conflicts.py`, `compare_rocprof_stats.py`.

This sequence costs roughly:

- step 1: 100 ms × N variants → typically <1 s
- step 2: 500 ms × N → typically 5-10 s
- step 3: 50 ms × N
- step 4: 500 ms × N
- step 5: 10 s × N (hipcc dominates)
- step 6: 1-5 s × N × M shapes
- step 7: 10-60 s per kernel (rocprof)

So a full sweep over 10 variants × 8 shapes is around 5 minutes
including rocprof, versus hours if you go straight to rocprof
without filtering.

### 18.2 Programmatic Use

Every probe has a Python entry point that doesn't require argv:

```python
from probe_occupancy import probe_occupancy, ARCH_GFX950
from probe_intrinsic_counts import probe_intrinsic_counts, count_intrinsics
from probe_isa_inspect import probe_isa_inspect
from probe_lowering_compare import probe_lowering_compare
from probe_config_sweep import probe_config_sweep
from probe_targeted_bench import bench_shapes, time_cuda_event

# Example: feed a custom kernel + spec to probe_occupancy
from rocke.instances import (
    UnifiedAttention2DTiledSpec, build_unified_attention_2d_tiled,
)
spec = UnifiedAttention2DTiledSpec(
    head_size=64, block_size=32, num_query_heads=64, num_kv_heads=8,
    dtype="bf16", use_sinks=True, sliding_window=0, has_softcap=False,
    num_warps=4, tile_size=64,
)
kdef = build_unified_attention_2d_tiled(spec)
probe_occupancy([("my_variant", kdef, spec.num_warps)], arch=ARCH_GFX950)
```

### 18.3 Probe Outputs and What They Mean

| Output field | From | Meaning |
|---|---|---|
| `vgpr_count` | `probe_occupancy.py` | private VGPRs per lane, allocated in 16-VGPR slots |
| `agpr_count` | `probe_occupancy.py` | MFMA accumulator VGPRs (gfx9x0 only) |
| `sgpr_count` | `probe_occupancy.py` | scalar regs per wave |
| `vgpr_spill_count` | `probe_occupancy.py` | nonzero = spills, usually a perf bug |
| `lds_size` | `probe_occupancy.py` | static LDS bytes per workgroup |
| `waves_per_cu` | `probe_occupancy.py` | coarse occupancy estimate |
| `limited_by` | `probe_occupancy.py` | `VGPR`, `AGPR`, `LDS`, `WAVES_PER_EU_HINT`, `MAX_WAVES_PER_CU` |
| `mfma` (cat) | `probe_isa_inspect.py` | count of `v_mfma_*` instructions |
| `vmem_load` / `vmem_store` | `probe_isa_inspect.py` | `buffer_*` + `global_*` count |
| `waitcnt` patterns | `probe_isa_inspect.py` | top 6 most-frequent encoded operand strings |
| `intrinsics.mfma.f32.*` | `probe_intrinsic_counts.py` | per-atom intrinsic count in lowered IR |
| `intrinsics.raw.ptr.buffer.load.lds` | `probe_intrinsic_counts.py` | async DRAM→LDS count |
| `intrinsics.ds.bpermute` / `ds.swizzle` | `probe_intrinsic_counts.py` | cross-lane reduction primitive |
| `structural.fmul` / `fadd` | `probe_intrinsic_counts.py` | post-lowering scalar arithmetic count |

---

## 19. Reproducible Commands

All commands assume `python` is the interpreter from the team's
canonical environment (CPython 3.12 with `torch + rocm`, `aiter`
importable, GPU visible). `PYTHONPATH` is set so `rocke` is
importable — the conventional layout is to set it to the
`composablekernel/python` directory.

### 19.1 PYTHONPATH bootstrap

From the `composablekernel` checkout root:

```bash
export PYTHONPATH=Python
```

### 19.2 The single validation block

```bash
cd <composablekernel-checkout>
export PYTHONPATH=Python

PYTHONDONTWRITEBYTECODE=1 python tests/test_rocke.py
PYTHONDONTWRITEBYTECODE=1 python python/test/test_rocke_examples.py

OUT_DIR="${OUT_DIR:-$(mktemp -d)}"
python -m rocke.examples.common.bake_off_implicit_gemm --output-dir "$OUT_DIR"
python -m rocke.run_manifest "$OUT_DIR"/*.hsaco "$OUT_DIR"/manifest.json --verify

python Python/rocke/examples/common/distribution_reduce_demo.py --M 32 --N 4096
python Python/rocke/examples/common/distribution_2d_add_demo.py --H 64 --W 128
python Python/rocke/examples/common/ck_tile_parity.py --op all

export AITER_PATH=<aiter-checkout>
PYTHONPATH="Python:${AITER_PATH}" python \
  Python/rocke/examples/gfx950/attention/parity_unified_attention.py \
  --scenario decode_d128_b16 --attempts 1 --warmup 0 --paths auto,2d,3d
```

### 19.3 DSL probe quick-shots

```bash
cd <composablekernel-checkout>/python
PROBES=dsl_docs/optimization/utilities/tools/dsl_probes

python "$PROBES/probe_occupancy.py"        --demo attention_tiled_2d --arch gfx950
python "$PROBES/probe_intrinsic_counts.py" --demo attention_tiled_2d
python "$PROBES/probe_isa_inspect.py"      --demo attention_tiled_2d --mcpu gfx950
python "$PROBES/probe_lowering_compare.py" --demo attention_tiled_2d --arch gfx950
python "$PROBES/probe_config_sweep.py"     --demo attention_tiled_2d
python "$PROBES/probe_targeted_bench.py"   --dry-run
```

### 19.4 rocprof attach (matched to a single kernel)

```bash
sudo rocm-smi --setperflevel high
sudo rocm-smi --setsclk 7

cat > metrics.txt <<'EOF'
pmc: SQ_WAVES
pmc: SQ_BUSY_CU_CYCLES
pmc: SQ_INSTS_MFMA
pmc: SQ_ACTIVE_INST_MFMA
pmc: TCC_EA_RDREQ_64B
pmc: TCC_EA_WRREQ_64B
pmc: TCC_HIT
pmc: TCC_MISS
pmc: SQ_WAIT_INST_LDS
pmc: TCP_PENDING_STALL_CYCLES
pmc: LDS_BANK_CONFLICT
EOF

rocprofv3 -i metrics.txt -o run.csv --stats --kernel-trace -- \
  python dsl_docs/optimization/utilities/tools/dsl_probes/probe_rocprof_single.py \
  --builder mypkg.mymod:make_runner \
  --problem-json /tmp/problem.json \
  --iters 50 --warmup 10
```

### 19.5 LDS / occupancy follow-up tools

```bash
python dsl_docs/optimization/utilities/tools/stage4_analyze/analyze_lds_conflicts.py \
  --rocprof run_kernel_stats.csv --isa kernel.s --arch gfx950

python dsl_docs/optimization/utilities/tools/stage5_compare/compare_rocprof_stats.py
```

---

## 20. Cross References

- **Short loop**: §0 "The Loop" at the top of this runbook.
- **DSL primitive map**: `runbook_mapping.md`.
- **Compliance table with measurements**: `runbook_compliance.md`.
- **Validation pass output**: `measured_results.md`.
- **Skill briefs**: `utilities/skills/*.md` —
  - `bisect-perf-regression.md`
  - `capture-kernel-trace-rocke.md`
  - `empirical-case-studies.md`
  - `gemm-optimization-rocke.md`
  - `kernel-launch-guide.md`
  - `kernel-trace-analysis.md`
  - `lds-optimization-rocke.md`
  - `prefetch-data-load-rocke.md`
- **Static inspection probes**: `utilities/tools/dsl_probes/`
  (see also `dsl_probes/README.md` for a when-to-use index).
- **Profiling-counter tools**: `utilities/tools/stage4_analyze/`,
  `utilities/tools/stage5_compare/`, `utilities/tools/utils/`.
- **Benchmark harnesses**: `utilities/tools/stage1_benchmark/`.
- **Target architecture reference**: see
  [§21 Target Architecture Reference](#21-target-architecture-reference)
  — the single hub for per-arch MFMA atoms, LDS specs, cross-lane
  primitives, register / occupancy caps, chiplet / XCD, buffer
  descriptors, fp8 / MX support, and compiler caveats.

---

## 21. Target Architecture Reference

The runbook itself is architecture-agnostic — every principle and
lever applies to any AMDGPU CDNA target. The concrete per-architecture
facts live in per-arch references under
[`optimization/arch/`](arch/README.md). **This section is the single
hub for those facts: every "architecture reference" breadcrumb earlier
in the runbook links here, and this is the only place the base runbook
points at the arch files directly.**

Pick your target:

- **[arch/gfx950.md](arch/gfx950.md)** — CDNA4 / MI350X / MI355X (the
  DSL's default target).
- **[arch/gfx942.md](arch/gfx942.md)** — CDNA3 / MI300X / MI325X.

See [arch/README.md](arch/README.md) for the index and how to add a new
architecture. Every arch reference uses the same `21.x` layout, so a
breadcrumb that cites "§21.2" means subsection 21.2 of your target's
file:

| Subsection | Topic |
|---|---|
| 21.1 | MFMA atom catalog |
| 21.2 | LDS specifics (size / banks / conflict periods / swizzle) |
| 21.3 | Cross-lane primitives |
| 21.4 | Register / occupancy |
| 21.5 | Chiplet / XCD |
| 21.6 | Buffer descriptor (AMDGPU) |
| 21.7 | FP8 / quantization |
| 21.8 | Compiler caveats |
| 21.9 | Default ISA target |
| 21.10 | Pointers to deeper material |

## Appendix: One-Page Diagnostic Decision Tree

```text
new kernel / regression
   │
   ▼
1) Build smoke (probe_config_sweep --only-build)
      │
      ├─ SPEC-FAIL → fix coupled fields (validation in spec __post_init__)
      ├─ BUILD-FAIL → reduce / isolate failing variant; verify IR coverage
      └─ all build  → 2)
   │
   ▼
2) probe_occupancy
      │
      ├─ spill > 0     → reduce VGPR pressure (atom, accumulators, unroll)
      ├─ waves < 4     → check LDS budget; try pipeline="lean"; smaller tile_k
      ├─ limited LDS   → recheck swizzle (Section 6.4a); smaller tile or async
      └─ healthy       → 3)
   │
   ▼
3) probe_intrinsic_counts
      │
      ├─ no MFMA       → wrong atom selected; check Section 7
      ├─ no async DMA  → pipeline != "compv4" (or unsupported shape)
      ├─ s.barrier ≫   → too many phases; collapse (Section 8.3)
      └─ healthy       → 4)
   │
   ▼
4) probe_targeted_bench against baseline (Triton / CK Tile / AITER)
      │
      ├─ regression    → 5)
      └─ near-best     → ship & document
   │
   ▼
5) probe_isa_inspect
      │
      ├─ buffer_store_short ≫ → epilogue is scalar; vectorize (Section 9.2)
      ├─ ds_read non-tr ≫     → swizzle isn't paired with reader; pick the right LDS layout
      ├─ waitcnt patterns ≫   → barrier/wait timing not aligned to async DMA
      └─ unclear              → 6)
   │
   ▼
6) rocprofv3 with metrics.txt
      │
      ├─ memory_stall > 40 %  → prefetch / async LDS (Section 8.2)
      ├─ lds_stall > 20 %     → run analyze_lds_conflicts.py
      ├─ bandwidth > 80 %     → memory-bound, increase reuse (Section 6.3)
      └─ compute_util > 70 %  → optimize MFMA packing (Section 7.4)
```

This decision tree is intentionally lossy: it points at the next
section to read, not at the final answer. The runbook itself is the
canonical reference; this tree is just the dispatcher.

### Symptom-to-action table (complementary view)

When you have a known symptom and want the typical first action:

```text
Symptom: correct but slow, low MFMA count
Likely:  wrong atom / tile, scalar path, missing vectorization
Action:  inspect LLVM / ISA (§11, §18); check atom selection (§7.1)
         and the inner loop (§12.1.B/C)

Symptom: fast but incorrect only on padded / tail shapes
Likely:  invalid pointer load, bad descriptor valid, vector crosses tail
Action:  test tiny adversarial shapes (§1.5); inspect buffer-rsrc
         sentinel path (§6.1, arch ref §21.6)

Symptom: intermittent wrong answers in async path
Likely:  missing `s_waitcnt` / barrier, workspace lifetime
Action:  add / check `s_waitcnt(vmcnt=0)` (§8.2-8.3), stream sync,
         launcher keep-alive (§12.1.N)

Symptom: atom change improves ISA but regresses runtime
Likely:  VGPR / LDS occupancy loss, or epilogue bottleneck
Action:  inspect resources via `probe_occupancy.py`; try cshuffle vs
         direct epilogue alternatives (§9.2-9.3)

Symptom: direct conv close but not within tolerance
Likely:  K-packed lane order or accumulator-reset bug
Action:  compare per-lane small reference; inspect fold order
         (§5.3, §7.3)

Symptom: low MfmaUtil + high VALUBusy + low MemUnitStalled
Likely:  compute-throttled by VALU / LDS plumbing, not HBM-bound
Action:  per-iter ISA histogram (§17.4); fix the plumbing
         (epilogue / register-residency / lane layout), not the
         launch knobs

Symptom: removing a per-iter cost regressed throughput
Likely:  the cost moved to a worse place (cross-lane permute,
         VGPR pressure, occupancy drop)
Action:  per-iter ISA diff before/after the change (§9.3, §17.4)
         — never declare a structural change a win without it
