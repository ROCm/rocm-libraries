# CK DSL fused-MoE parity & benchmark harness

This folder hosts the cross-backend parity + benchmark scripts for the
CK DSL `FusedMoeForward` pipeline — the **chained-launch** fused-MoE
forward (router → sort → gather → gate/up GEMM → SiLU → down GEMM →
weighted reduce, as `5 + 3·E` launches on one HIP stream). It is the
canonical performance harness for the CK DSL MoE work, and is distinct
from the single-launch mega-kernel in
[`examples/gfx950/fused_mega_moe/`](../../fused_mega_moe/) (same math,
opposite kernel structure).

> **New to fused MoE, or want the kernel strategy?**
> [`ALGORITHM.md`](ALGORITHM.md) derives the pipeline from the math up —
> the SwiGLU-FFN + top-k routing spec, the sorted/de-padded bucket
> layout, the per-stage steps, and why preshuffled-B, active-tile skip,
> and HIP-graph replay are scheduling/layout levers that never change
> what is computed. This README is the optimization history and the
> runnable field guide.

## File map

| path | role |
|---|---|
| [`README.md`](README.md) | this document — field guide + optimization history + results |
| [`ALGORITHM.md`](ALGORITHM.md) | the math, data layout, and per-stage steps of the chained pipeline |
| [`fused_moe_e2e_perf.py`](fused_moe_e2e_perf.py) | four-way end-to-end perf + correctness harness: CK DSL vs torch eager vs Triton vs CK Tile C++ |
| [`tune_gate_up_silu.py`](tune_gate_up_silu.py) | tile-shape × activation-barrier (`packed`/`dual`/`interleaved`) sweep for the fused gate/up/SiLU GEMM |
| [`test_fused_moe_preshuffle.py`](test_fused_moe_preshuffle.py) | end-to-end parity + perf for the preshuffled-B and active-tile-skip knobs on `FusedMoeForward` (HIP-graph-replayed, 6 scenarios) |
| [`test_preshuffle_b.py`](test_preshuffle_b.py) | standalone parity + perf for `trait.preshuffle_b=True` on a single `BatchedGemm` |
| [`test_active_tile_skip.py`](test_active_tile_skip.py) | standalone parity + perf for `trait.active_tile_skip=True` on a single `BatchedGemm` (all-active = zero overhead, all-inactive ≈ 13× faster) |
| `__init__.py` | package marker |

The two `tune_*` / `fused_moe_e2e_perf.py` scripts share input
generation, scenario definitions, and the torch-eager reference from
`fused_moe_e2e_perf.py`. The three `test_*` scripts are self-contained
parity/perf checks (run directly, no extra args needed).

---

## End-to-end perf: `fused_moe_e2e_perf.py`

Drives `rocke.instances.FusedMoeForward` and three reference
implementations on a set of MoE forward shapes, reports per-backend
latency (ms), correctness vs the torch reference (max abs / mean abs
/ rel error), and the resulting speedup ratios.

### Backends

1. **CK DSL** — `rocke.instances.FusedMoeForward`. Composes
   `topk_softmax` (router) + `MoeSortingLauncher` (sort 3-chain) +
   `FusedMoeLauncher` (gather / silu_mul / topk_reduce streaming
   kernels) + `GroupedGemmLauncher` (per-expert gate / up / down
   GEMMs) into one pipeline driven by chained
   `rocke.runtime.launcher.launch_kernel` calls. In static-offset
   mode the forward is HIP-graph-capturable; the harness uses graph
   replay where possible to keep the comparison focused on kernel
   work rather than Python launch overhead.
2. **Torch eager** — vectorised per-expert mask + scatter (faster
   than a naive per-token-per-topk Python loop, but still pure
   torch ops with no fusion). Gold-standard correctness oracle.
3. **Triton** — purpose-written single-kernel fused-MoE that mirrors
   the CK DSL pipeline's algorithmic shape (per `(token, k)` program:
   gate / up / SiLU mul / down GEMMs against per-expert weights,
   atomic-add into an `f32 Y` accumulator). This is **not** a tuned
   production Triton kernel — it's a fair-baseline reference. AITER's
   tuned `e2e_moe` Triton kernel currently crashes with a
   memory-access fault on MI355X (gfx950); the script falls back to
   this purpose-written kernel until that is fixed upstream.
4. **CK Tile C++** — `build/bin/tile_example_fused_moe` invoked via
   `subprocess` with matching `-t / -e / -k / -h / -i / -prec_*`
   arguments. The C++ binary uses its own random inputs (no public
   hook to feed external tensors), so the row reports a perf-only
   number; correctness is validated against torch eager on the
   Python side only.

### Scenarios

Five default scenarios (`default_scenarios()` in the script). All use
`f16` activations unless overridden by `--dtype bf16`.

| Scenario | tokens | experts | topk | hidden | intermediate | shape class |
|---|---:|---:|---:|---:|---:|---|
| `decode_T1_E8_K2_H4096_I7168` | 1 | 8 | 2 | 4096 | 7168 | inference decode (one token, large experts) |
| `decode_T8_E8_K2_H4096_I7168` | 8 | 8 | 2 | 4096 | 7168 | small-batch decode |
| `batch32_E8_K2_H4096_I7168` | 32 | 8 | 2 | 4096 | 7168 | mid-batch |
| `prefill_T128_E8_K2_H4096_I7168` | 128 | 8 | 2 | 4096 | 7168 | prefill (many tokens) |
| `small_T32_E4_K2_H128_I256` | 32 | 4 | 2 | 128 | 256 | small validation shape (matches parity tests) |

Pass `--scenario NAME` (repeatable) to restrict to a subset. Pass
`--dtype bf16` to switch every scenario to bf16.

Scenario constraints driven by the CK DSL pipeline (and matching what
the standard MoE workloads look like in production):

- `hidden` and `intermediate` must be multiples of the GEMM tile dims
  (default `tile_n=128`, `tile_k=64`) and the streaming kernel
  `block_size` (the `FusedMoeForwardSpec.streaming_block_size` default
  is 256; the harness caps it at `min(256, hidden)` for the small
  validation shape, so `hidden` must divide that).
- `experts ≤ sort_block_size` (= 64) for the single-block scan kernel.
- `topk ≤ experts`.

### Running

```bash
cd <composablekernel-checkout>
export AITER_PATH=<aiter-checkout>
PYTHONPATH="Python:${AITER_PATH}" python \
  Python/rocke/examples/gfx950/moe/fused_moe_e2e_perf.py \
  --attempts 10 --warmup 3 \
  --report /tmp/moe_perf.json
```

Flags:

| Flag | Default | Notes |
|---|---|---|
| `--scenario NAME` (repeatable) | all | restrict to the named scenarios |
| `--attempts N` | `10` | timed iterations per backend; reported number is `elapsed_ms / N` from a single HIP-event pair recorded on torch's current stream |
| `--warmup N` | `3` | untimed warmup iterations |
| `--dtype {f16, bf16}` | `f16` | activation dtype for all backends |
| `--skip-rocke` | off | skip the CK DSL backend |
| `--skip-torch` | off | skip the torch eager reference |
| `--skip-aiter` | off | skip the Triton baseline (the flag name is historical — the actual Triton call is the purpose-written fallback documented above) |
| `--skip-cktile` | off | skip the CK Tile C++ binary |
| `--report PATH` | none | dump every measurement to JSON |

The script returns a non-zero exit code if any scenario sees a CK DSL
correctness regression (`max_abs > 0.5`). AITER / CK Tile failures are
treated as environment issues and do not gate.

### Methodology

Every row in the tables below is the **mean per-launch wall time over
10 timed iterations** after 3 untimed warmup launches, measured with
HIP events recorded on torch's current stream. Both backends use
the same timer and the same stream, so the numbers are directly
comparable. Concretely, `time_callable_ms` does:

1. 3 untimed warmup launches.
2. `torch.cuda.synchronize()` to drain.
3. Record a start HIP event on `torch.cuda.current_stream()`.
4. 10 timed launches on that same stream.
5. Record an end HIP event, synchronize on it, report `elapsed_ms / 10`.

The CK DSL forward in static-offset mode is HIP-graph-captured (one
capture, then the timed loop replays the graph); this is the realistic
inference-benchmark mode and the only way the DSL pipeline can match
the per-launch overhead of a tuned C++ reference.

Between scenarios the harness runs an `_isolate_lane()` step
(`torch.cuda.synchronize` + `synchronize_and_release` +
`gc.collect`) to drop any retained args / workspace tensors / module
caches before the next scenario allocates its inputs. This avoids a
known ROCm 7.2 / torch 2.12 edge case where the recycled pool storage
is still in flight on the GPU command processor.

#### Apples-to-apples — what each timed `call()` contains

Every backend's timed region runs the **same algorithmic work** to
make the speedup numbers honest:

- **router** (`topk` + `softmax` over the routing logits) — runs
  inside the timed window for every backend, since CK DSL's
  `forward()` includes the router as a fused `topk_softmax` kernel.
- **per-expert MoE math** (gather → gate × X → up × X → SiLU(gate) ×
  up → down × hidden → topk-weighted reduce into Y) — same algorithm
  in all three backends.
- **same precision contract** — fp16 (or bf16) inputs and weights;
  fp32 accumulator inside the matmul (implicit in torch's rocBLAS
  matmul, explicit in CK DSL's MFMA atom, explicit in Triton's
  `tl.atomic_add` accumulator); fp16 / bf16 output.
- **output buffer pre-allocated outside the timed window** for every
  backend. CK DSL takes a pre-allocated `Y` argument; torch eager and
  Triton allocate `Y_out` once outside `call()` and zero it in place
  at the start of each iteration.

Specifically **not** in any backend's timed `call()`:

- HSACO compile / module load / Triton JIT (paid in warmup or before).
- Allocation churn for the output tensor.
- Gratuitous fp32 upcasts of inputs / weights inside the per-expert
  loop. (An earlier version of this harness upcast every operand to
  fp32 inside torch eager's per-expert loop, which doubled the bytes
  through every matmul and inflated the torch number by 30-40 %;
  that has been removed — see "Optimization log" below.)

What does differ across backends is the **kernel structure** and
**timed-window scope**:

| Backend | Per-call dispatches | Router included? | Notes |
|---|---:|---|---|
| CK DSL | 8 chained launches (router + sort 3-chain + gather + 2 GEMMs + silu_mul + reduce, or fewer in static-graph mode) | **yes** — `topk_softmax` runs as the first kernel in `forward()` | one `forward()` call; in-Python launcher chain |
| torch eager | router + per-expert Python loop (4-9 small launches per expert × N experts) | **yes** — `torch.topk + softmax` are inside `call()` | each expert is `mask.nonzero` + 3 small matmuls + silu_mul + index_add — the irreducible cost of "what plain torch costs" |
| Triton | router + 1 mega-kernel | **yes** — `torch.topk + softmax` are inside `call()` | the kernel does gate/up/silu/down/atomic-add all inline; output tensor cast (fp32→fp16) inside `call()` is ~µs |
| CK Tile C++ | 2 launches (sort + GEMM mega-kernel; in subprocess) | **no** — `topk_ids_ptr` / `topk_weight_ptr` are precomputed and passed in; only `fused_moesorting + fused_moegemm` are timed | uses its own internal random inputs; perf-only number |

The CK Tile C++ row is the only one whose timed window does **not**
include the topk_softmax router. The actual CK DSL router kernel
costs ~4 µs / call (from rocprof), so subtracting it would change
the published `rocke vs ck_tile_cpp` gap by ~1 % on the H=4096
shapes (`decode_T1`: 0.30× → 0.31×) and ~10 % on `small`
(0.43× → 0.48×). **Not the cause of the 2-3× gap.** See
"Optimization log → Round 6" for the full apples-to-apples
decomposition.

CK DSL's "speedup vs torch eager" is therefore best read as **"how
much faster is an 8-launch fused-kernel pipeline than a Python-side
per-expert loop driving small per-expert matmuls"** — measured in
real conditions for both, with no fp16↔fp32 traffic that wouldn't
happen in production.

### Latest results (MI355X, gfx950, ROCm 7.2, torch 2.12)

`--attempts 10 --warmup 3 --dtype f16`. Speedup is `(other) / (CK DSL)`,
so values > 1 mean CK DSL is faster.

> The `rocke` column below is the **baseline** configuration measured
> in Rounds 1-5 (no preshuffle, active-tile-skip off). The preshuffle-B
> (Round 10) and active-tile-skip (Round 11) levers — both now shipped
> as default-on / env-toggleable knobs — improve the `rocke` numbers
> further; see those rounds for the per-shape post-lever latencies
> (e.g. `decode_T1` drops from ~0.40 ms to ~0.27 ms with active-tile
> skip on). Re-run the script to regenerate the table for your build.

| Scenario | rocke | torch | triton | ck_tile_cpp | rocke vs torch | rocke vs triton | rocke vs ck_tile_cpp |
|---|---:|---:|---:|---:|---:|---:|---:|
| decode_T1_E8_K2_H4096_I7168     | 0.401 ms | 0.504 ms | 14.960 ms | 0.122 ms | **1.26×** | **37.31×** | 0.30× |
| decode_T8_E8_K2_H4096_I7168     | 0.414 ms | 1.041 ms | 17.616 ms | (skip)   | **2.51×** | **42.50×** | — |
| batch32_E8_K2_H4096_I7168       | 0.500 ms | 1.300 ms | 19.985 ms | (skip)   | **2.60×** | **39.95×** | — |
| prefill_T128_E8_K2_H4096_I7168  | 0.833 ms | 1.311 ms | 29.986 ms | (skip)   | **1.57×** | **36.01×** | — |
| small_T32_E4_K2_H128_I256       | 0.037 ms | 0.661 ms | 0.041 ms  | 0.016 ms | **17.67×** | **1.10×** | 0.43× |

Reading the table:

- **CK DSL beats torch eager** on every realistic scenario (1.26-2.60×
  on the H=4096 shapes; 17.67× on the small validation shape, where
  torch's Python-side per-expert loop overhead dominates the
  matmul time).
- **CK DSL beats the purpose-written Triton baseline** by a wide
  margin on the four large-expert scenarios (36-42× — the Triton
  kernel is single-program-per-`(token, topk-slot)` with atomic-add
  into a shared `Y`; not tuned for these shapes), and edges it on the
  small validation shape (1.10×).
- **CK Tile C++ wins** the two scenarios where it doesn't crash. On
  this build, the binary segfaults with a memory-access fault on the
  three larger E=8 / H=4096 / I=7168 scenarios (decode_T8, batch32,
  prefill_T128) — a known issue with the C++ binary that does not
  affect the CK DSL row. The remaining 2-3× gap to CK Tile C++ is
  structural (mega-kernel with in-kernel grouped-GEMM dispatch vs
  CK DSL's chained-launch pipeline) — see "Optimization log" below.

`max_abs(CK vs torch ref)` for the CK DSL column is below 5e-4 on
every scenario, well within the f16-with-fp32-accumulator tolerance
band for an MoE forward over O(7168) reduction terms.

`max_abs(Triton vs torch ref)` is similarly below 5e-4, confirming
the purpose-written Triton baseline is correct (it is just slow on
these shapes).

### Optimization log

Recorded so future passes don't repeat the same lever sweeps. The
methodology is the optimization runbook's iteration loop
(`dsl_docs/optimization/optimization_runbook.md`, "The Loop"
preamble): hypothesis → correctness → baseline → inspect → change
one lever → re-verify → re-measure → keep / revert → record.

**Round 1: identify the bottleneck.** A `torch.profiler` capture on
`decode_T1_E8_K2_H4096_I7168` showed the CUDA-time breakdown:

| Kernel | µs / iter | % of total |
|---|---:|---:|
| `interleaved_gate_up_silu` (gate+up+SiLU GEMM) | ~186 | **43 %** |
| `batched_gemm` (down GEMM)                    | ~154 | **36 %** |
| `fused_moe_reduce` (topk-weighted reduce)     | ~46  | 11 %    |
| `hipModuleLaunchKernel` overhead × 6          | ~2 each | ~3 %  |
| `moe_sorting_scatter` / `gather` / `topk_softmax` | ~4 each | ~3 % combined |

The two batched GEMMs are 79 % of the time. The streaming kernels
(gather / silu_mul / reduce) together (~13 %) are the only category
where launch / scheduling overhead is meaningful relative to compute.

**Round 2: lever sweep on the experimental / opt-in flags.** Several
levers ship enabled by default in `FusedMoeForwardSpec`
(`use_experimental_interleaved_gate_up_silu=True`,
`use_experimental_fused_down_reduce=True`,
`active_tile_skip_gemms=True`, plus the `_default_gemm_tile()`
32×32×16 atom). Note the e2e harness deliberately pins
`use_experimental_fused_down_reduce=False` when building its CK DSL
spec (see `run_rocke()`), so the baseline numbers in the headline
table are measured with the separate `DownOut` + `topk_reduce` path,
not the fused down+reduce kernel. The remaining opt-in flag was
tried:

| Lever | Verdict |
|---|---|
| `use_experimental_fused_down_reduce` (spec default `True`) | per the spec field docstring, the native-FP-atomic fused down+reduce is ~10 % faster than the separate `topk_reduce` launch on the decode shapes and neutral on the compute-bound datacenter shape; the harness pins it `False` only to keep the published baseline on the simpler two-kernel path |
| `use_experimental_static_scatter_gather=True` | within noise (≤ 2 % on the larger shapes, tied on the small one) |

**Round 3: lever sweep on streaming-kernel and GEMM knobs.** Four
further levers tried:

| Lever | Verdict |
|---|---|
| `streaming_block_size = 128`              | **+2-9 %** (small win, divides H on every shape) |
| `streaming_block_size = 256`              | **+5-14 %** on H=4096 shapes; **fails** on small (H=128 not divisible by 256) |
| `streaming_vec = 4`                       | within noise (default 8 stays) |
| `use_experimental_fused_gate_up_silu=True` (dual-B variant) | regression on every scenario; default `interleaved` stays |

**Round 4: keep / revert.** `streaming_block_size = min(256, hidden)`
was a clean win across all five scenarios (5-14 % faster). It was
applied to the harness as `streaming_block_size = min(256, s.hidden)`
in `run_rocke()`. The previous hardcoded `streaming_block_size=64`
was a stale value — the spec default has since been bumped to 256
but the harness wasn't updated. The numbers in the table above
reflect this fix.

**Per-scenario CK DSL speedup vs the pre-fix harness:**

| Scenario | pre-fix | post-fix | Δ |
|---|---:|---:|---:|
| decode_T1_E8_K2_H4096_I7168     | 0.443 ms | 0.401 ms | **−9.5 %**  |
| decode_T8_E8_K2_H4096_I7168     | 0.463 ms | 0.414 ms | **−10.6 %** |
| batch32_E8_K2_H4096_I7168       | 0.537 ms | 0.500 ms | **−6.9 %**  |
| prefill_T128_E8_K2_H4096_I7168  | 0.887 ms | 0.833 ms | **−6.1 %**  |
| small_T32_E4_K2_H128_I256       | 0.039 ms | 0.037 ms | **−5.1 %**  |

The win shrinks as token count grows because the streaming kernels
are a fixed-cost overhead that becomes proportionally smaller as
total time grows on T=128.

**Round 5: tighten the apples-to-apples comparison.** While
inspecting the timed regions of each backend, we found two unfair
overheads inflating the torch eager and Triton baselines. They
weren't CK DSL "wins" so much as competitor "losses" caused by
unrepresentative measurement code:

| Issue | Fix | Impact on torch / triton |
|---|---|---|
| `run_torch_eager` upcast every operand to fp32 inside the per-expert loop (`.float()` on `X`, `W_gate`, `W_up`, `W_down`) — doubled the bytes through every matmul vs CK DSL / Triton's fp16-in / fp32-accum / fp16-out contract | Removed the upcasts; let torch's rocBLAS matmul use its implicit fp32 accumulator (same numerical contract as CK DSL) | torch eager **−30 to −40 %** on the H=4096 shapes |
| `run_torch_eager` allocated a fresh fp32 `Y` tensor per `call()` | Pre-allocate `Y_out` (fp16) outside; zero in place inside `call()` | small (~µs per call) |
| `run_triton` returned `Y_f32.to(inputs.X.dtype)` — allocated a fresh fp16 output tensor every call | Pre-allocate `Y_out`; cast in place via `Y_out.copy_(Y_f32)` | small (~µs per call) |

Effect on the published table:

| Scenario | rocke vs torch (pre-tighten) | (post-tighten) |
|---|---:|---:|
| decode_T1_E8_K2_H4096_I7168     | 1.88× | **1.26×** |
| decode_T8_E8_K2_H4096_I7168     | 4.30× | **2.51×** |
| batch32_E8_K2_H4096_I7168       | 4.57× | **2.60×** |
| prefill_T128_E8_K2_H4096_I7168  | 2.85× | **1.57×** |
| small_T32_E4_K2_H128_I256       | 21.35× | **17.67×** |

CK DSL is still clearly faster than torch eager on every scenario,
but the 4-5× wins on the medium-batch shapes were a measurement
artifact rather than a real speedup. The corrected numbers are the
honest comparison.

**Round 6: apples-to-apples decomposition vs CK Tile C++.** Before
declaring the gap "structural and unfixable", verify it isn't
another measurement bias like the torch / Triton issues from
Round 5. Decomposition for `decode_T1`, `decode_T8`, and `small`:

| Scenario | CK Tile cpp | CK DSL full | CK DSL − router | CK DSL graph mode |
|---|---:|---:|---:|---:|
| `small_T32_E4_K2_H128_I256`         | 16.2 µs  | 37.5 µs (2.32×)  | ~33 µs (2.04×)¹ | 37.3 µs (2.30×) |
| `decode_T1_E8_K2_H4096_I7168`       | 122.3 µs | 376.2 µs (3.08×) | ~372 µs (3.04×)¹ | 389.5 µs (3.18×) |
| `decode_T8_E8_K2_H4096_I7168`       | (skip)   | 396.5 µs         | ~392 µs           | 385.6 µs |

¹ "− router" subtracts the actual CK DSL `topk_softmax` kernel cost
(~4 µs / call from rocprof) — accounts for the scope difference (CK
Tile's timed window doesn't include the router).

Three findings:

- **Router scope difference is small.** Subtracting CK DSL's router
  changes the gap by ~1 % on `decode_T1` and ~12 % on `small`. Not
  the cause of the 2-3× gap.
- **Graph mode (`use_experimental_static_scatter_gather=True`) doesn't
  close the gap.** All three scenarios are flat vs the chained-launch
  baseline. So per-launch / scheduling overhead between CK DSL's
  chained kernels is **not** the dominant cost — the GPU is saturated
  by the GEMM kernels themselves.
- **The gap lives in the kernel body, not the harness.** From the
  Round 1 profiler breakdown, the GEMM kernels alone (`gate_up_silu` +
  `down`) sum to ~340 µs of CK DSL's ~376 µs on `decode_T1`. CK Tile
  fits the same work plus the reduce into ~110 µs. That's a ~3×
  kernel-time difference driven by structural choices that are
  CK-Tile-specific: mega-kernel sharing LDS and register state across
  gate / up / SiLU / down / weighted-reduce, in-kernel grouped-GEMM
  dispatch (no host roundtrip on `Counts` / `Offsets`), and a fused
  weighted reduce inline at the end of the down GEMM.

So the gap is real, structural, and driven by kernel design — not
measurement bias.

**Round 7: trait-flip sweep on the GEMM kernels.** Per §12.1 of the
runbook, the down `BatchedGemmSpec` and the experimental fused
gate-up-silu kernels accept a `TraitSpec` with `pipeline`,
`scheduler`, `epilogue`, `pad_m/n/k`, `persistent`, `waves_per_eu`,
`chiplet_swizzle`, and `preshuffle_b` levers. The current production
defaults are `pipeline=compv4`, `scheduler=intrawave`,
`epilogue=cshuffle`, `pad_m=True, pad_n=True`, everything else off
(`chiplet_swizzle=True + waves_per_eu=2` was tried previously per
the `to_batched_gemm_spec()` docstring — no improvement).
Re-confirmed across all four scenarios:

| Lever (vs baseline `compv4 / intrawave / cshuffle`) | small | decode_T1 | decode_T8 | batch32 |
|---|---:|---:|---:|---:|
| `scheduler=interwave` | 1.07× | 1.01× | 0.98× | 1.01× |
| `pipeline=compv3` | 1.04× | 1.00× | 1.00× | 1.00× |
| `pipeline=mem` | 1.05× | 1.02× | 0.99× | 1.00× |
| `epilogue=default` (for down GEMM) | **0.88×** | **0.54×** | **0.53×** | **0.73×** |
| `persistent=True` | 1.07× | 1.01× | 0.98× | 1.01× |
| `waves_per_eu=2` | 1.06× | 1.00× | 1.00× | 1.00× |
| `waves_per_eu=3` | 1.06× | **0.94×** | **0.91×** | 1.01× |

Findings:

- **`epilogue="default"` for the down GEMM is a major regression
  (0.53-0.88×)** — `cshuffle` (LDS-staged + wide global stores) is
  genuinely the right epilogue choice. The spec default was correct.
- **`waves_per_eu=3` regresses on the H=4096 shapes** — the implied
  smaller VGPR budget hurts the GEMM. Default (no hint, let the LLVM
  backend pick) wins.
- **Everything else is within noise** (~5-7 % on `small`, ≤ 2 % on
  the H=4096 shapes — the kind of run-to-run variance the runbook
  §2.3 warns about).

The TraitSpec config-level levers are exhausted.

The genuinely-untried lever from §12.1.H is `preshuffle_b=True` —
the host pre-shuffles `W_gate / W_up / W_down` once into the layout
`helpers/preshuffle.py::host_preshuffle_layout` produces, then the
per-lane B-load becomes one `buffer_load_dwordx4` per K-tile instead
of strided scalar loads. For MoE this is the right shape (weights
don't change between forward calls; preshuffle once at construction
time). It is a real kernel-time saver on the GEMM hot loop but it
requires orchestrator changes — `FusedMoeForward.__init__` would
need to apply the preshuffle to the weight tensors before invoking
the kernel, and the kernel itself needs to be built with the
matching `preshuffle_b=True` trait. That's a code change to the
authoring layer, not a config flip — left as an explicit follow-up.

**Round 8: tile × trait combination sweep.** Round 7 was single-lever
flips. Per the runbook §17.4 take-away principle 4 ("one lever is
rarely the whole change — co-evolve two or three"), Round 8 tested
30 combinations of (tile shape) × (trait combo) for the GEMM
kernels. New tile candidates beyond the original 5-tile tuner set:

| Tile | Shape | Hypothesis |
|---|---|---|
| `wide_n_t32n256k64_w1x4`           | M=32, N=256, K=64, warps 1×4, 32×32×16 atom | halve N-tile count → more reuse per CTA |
| `deep_k_t32n128k128_w1x2`          | M=32, N=128, K=128, warps 1×2, 32×32×16     | halve K-loop trip count |
| `wide_n_deep_k_t32n256k128_w1x4`   | M=32, N=256, K=128, warps 1×4, 32×32×16     | combined |
| `big_m_t64n128k64_w2x2`            | M=64, N=128, K=64, warps 2×2, 32×32×16      | bigger M (likely bad for low T) |
| `deep_k_atom16_t32n128k128_w1x2`   | M=32, N=128, K=128, warps 1×2, 16×16×32     | smaller atom + deeper K |

Each tile crossed with `default`, `interwave_persistent`,
`no_pad_m`, `interwave_no_pad_m`, `all_small_wins`. Headline result
on `decode_T1`:

| Tile + traits | decode_T1 (ms / vs baseline / vs CK Tile) |
|---|---|
| **tuner-best `t32n128k64_w1x2` + default**       | 405 µs (1.00× / 0.31× CK Tile) ← **best** |
| tuner-best + `all_small_wins`                    | 400 µs (1.01× / 0.31×) |
| `wide_n_t32n256k64_w1x4` (any trait)             | ~460 µs (**0.87×** / 0.27×) — regression |
| `deep_k_t32n128k128_w1x2` (any trait)            | ~462 µs (**0.87×** / 0.27×) — regression |
| `wide_n_deep_k_t32n256k128_w1x4` (any trait)     | ~635 µs (**0.64×** / 0.19×) — regression |
| `big_m_t64n128k64_w2x2` (any trait)              | ~490 µs (**0.83×** / 0.25×) — regression |
| `deep_k_atom16_t32n128k128_w1x2` (any trait)     | ~414 µs (**0.98×** / 0.30×) — regression |

Why each variant regressed:

- **Wider N** (`tile_n=256`): per-CTA LDS doubles → occupancy drops
  below 2 WGs/CU on the H=4096 shapes; the bigger reuse per CTA
  doesn't pay for the lost concurrency.
- **Deeper K** (`tile_k=128`): the K-loop trip count isn't the
  bottleneck — operand staging is. Halving the trips doesn't move
  the needle; doubling per-iteration LDS does.
- **Wide N + deep K**: stacks both regressions multiplicatively.
- **Bigger M** (`tile_m=64`): wastes more rows on padding for the
  low-T scenarios (`decode_T1` has 1-2 real tokens per active
  expert; `tile_m=64` pads to 64).
- **Smaller atom** (16×16×32): confirms the tuner's earlier finding
  that 32×32×16 atom is structurally better — the same tile with the
  smaller atom regresses 2-3 %.

The tuner's original `t32n128k64_w1x2_atom32` is genuinely the
performance optimum for this kernel class, on these scenarios.
Combinations of trait flips on top of it land within run-to-run
noise (1.01-1.06× across scenarios; the runbook §2.3 caution about
±5-7 % variance on sub-200 µs measurements applies).

This **confirms the runbook §17.4 take-away principle 1**: at the
configuration layer, generic levers stop helping. The 2.3-3.1× gap
to CK Tile C++ is a structural kernel-design difference and cannot
be closed by `TileSpec` / `TraitSpec` flips. The two real next steps
both require code changes (`preshuffle_b=True` with orchestrator
support, or single-kernel re-implementation) — see the "Remaining
gap" section below.

**Round 9: attempted `preshuffle_b=True`.** Round 8 named
`preshuffle_b` as the genuinely-untried lever from §12.1.H — host
pre-shuffles `W_gate / W_up / W_down` once into a layout
(`helpers/preshuffle.py::host_preshuffle_layout`) so the per-lane
B-load in the GEMM hot loop becomes one `buffer_load_dwordx4` per
K-tile instead of strided scalar loads. Attempted to flip it on
`BatchedGemmSpec` (the simpler integration than the experimental
fused gate-up-silu specs) and observe the change in the lowered
kernel:

```text
preshuffle_b=False : hsaco 8160B, llvm_ir 546 lines,
                     kernel_name: ..._compv4_intrawave_cshuffle_pad_bat
preshuffle_b=True  : hsaco 8144B, llvm_ir 546 lines,
                     kernel_name: ..._compv4_intrawave_cshuffle_pad_bat
```

The HSACO bytes and IR line counts are essentially identical, the
kernel name doesn't even reflect the flag — meaning **the DSL
silently ignores `preshuffle_b=True`** in the universal / batched
GEMM build path. Confirming via grep:

- `preshuffle_b` is declared in `instances/gemm_universal.py:165` as
  a `TraitSpec` field but **not read anywhere** in the
  `build_universal_gemm()` body — `grep "trait\.preshuffle_b\|spec\.preshuffle_b\|preshuffle_b" instances/gemm_universal.py`
  returns only the field declaration.
- `instances/flatmm.py:213` **rejects** `preshuffle_b=True` at build
  time with "is gated by the v2 preshuffled-B kernel which is not
  yet shipped".
- `instances/block_scale_gemm.py:163` **rejects** it with "requires
  the MFMA-based kernel body which is not yet wired up".

So `preshuffle_b` is a documented-but-unimplemented lever — the
spec field exists, the runbook §12.1.H describes the intent, but
the v2 MFMA-based preshuffled-B kernel body that would consume the
flag hasn't shipped. Implementing it requires authoring a new GEMM
inner loop that calls `helpers/preshuffle.py::emit_preshuffleb_offset`
from the per-lane B-load path — a real kernel-authoring task, not
a config flip.

That closes the configuration-level optimization surface. Both
remaining follow-ups (`preshuffle_b` and single-kernel rewrite)
require kernel-side code changes, which are out of scope for a
session-scoped optimization pass following the runbook loop.

**Round 10: implemented `preshuffle_b=True` end-to-end.** Round 9
ended with `preshuffle_b` declared as a `TraitSpec` field but never
read by the kernel-build path; HSACO bytes / IR were identical with
the flag on or off. Round 10 closes that gap:

- `instances/gemm_universal.py::build_universal_gemm` now branches
  on `spec.trait.preshuffle_b` in `emit_load_phase` and emits a
  contiguous wide-load over the per-tile preshuffled buffer instead
  of the strided per-row load. The kernel name picks up a `preb`
  flag so cached HSACOs don't alias.
- `instances/moe_gemm_fused.py::build_moe_interleaved_gate_up_silu_gemm`
  honors the same flag; the only delta vs `gemm_universal` is the
  `n_tile_count` formula, which uses `2*N` for the gate-up packed
  layout.
- `instances/fused_moe_e2e.py` adds three orchestrator-side knobs:
  `preshuffle_w_down`, `preshuffle_w_gate_up_packed`, and
  `preshuffle_w_gate_up_interleaved`. The host preshuffle helper
  (`_host_preshuffle_b`) and per-spec caches (`_w_down_preshuffled`,
  `_gu_concat_preshuffled`, `_gu_interleaved_preshuffled`) are
  built inside `capture_graph` before the warmup loop so the
  one-time `torch.cat / permute / contiguous` cost stays out of the
  captured / replayed region.

Standalone `BatchedGemm` measurements (with the production
`tile_m=32, tile_n=128, tile_k=64` tile, fp16, batched=True)
confirm the kernel-level win:

```text
B=4   M=1024 N=1024 K=1024  : baseline 59.13 us  preshuf 32.53 us  speedup 1.82x
B=8   M=512  N=2048 K=4096  : baseline 232.35 us preshuf 151.03 us speedup 1.54x
B=16  M=256  N=4096 K=4096  : baseline 613.14 us preshuf 318.12 us speedup 1.93x
B=32  M=128  N=4096 K=4096  : baseline 670.52 us preshuf 346.02 us speedup 1.94x
B=8   M=128  N=2048 K=4096  : baseline 222.69 us preshuf 105.26 us speedup 2.12x
B=16  M=128  N=4096 K=2048  : baseline 157.83 us preshuf 92.35 us  speedup 1.71x
```

End-to-end (HIP-graph-replayed, all three preshuffle knobs at the
optimum) on real MoE shapes (T8/T1, E8, K2, H4096, dtype=f16):

```text
                                  baseline    preshuf_intl   speedup   parity
decode_T8_I4096   T=8 H=4096 I=4096   274.14 us   231.73 us   1.183x    bitwise
decode_T1_I4096   T=1 H=4096 I=4096   267.98 us   226.44 us   1.183x    bitwise
decode_T1_I7168   T=1 H=4096 I=7168   384.41 us   360.77 us   1.066x    bitwise
decode_T8_I7168   T=8 H=4096 I=7168   385.05 us   367.78 us   1.047x    bitwise
```

vs CK Tile C++ on the canonical decode scenario:

```text
decode_T1_E8_K2_H4096_I7168 : rocke baseline 0.388 ms (0.31x of cktile)
                              rocke preshuf  0.355 ms (0.34x of cktile)
                              ck_tile_cpp     0.121 ms
```

Closed ~7% of the gap to CK Tile on the production-canonical decode
shape. The 18% wins on the I=4096 shapes show the lever pays off
proportional to how much per-K-tile B-load time the kernel was
spending; for I=7168 the gate-up GEMM has a flatter K-axis (more N
tiles per CTA), so the in-tile B-load is a smaller fraction of the
kernel time.

Bitwise parity holds across every test (`rel=0.00e+00` on all
six scenarios in `examples/gfx950/moe/test_fused_moe_preshuffle.py`).

The lever is now an env-toggleable knob in
`examples/gfx950/moe/fused_moe_e2e_perf.py`:

- `ROCKE_PRESHUFFLE_W_DOWN=1`
- `ROCKE_PRESHUFFLE_W_GATE_UP_INTERLEAVED=1` (production default
  path uses the interleaved kernel)
- `ROCKE_PRESHUFFLE_W_GATE_UP_PACKED=1` (forces the packed
  kernel; do not combine with the interleaved knob)

**Remaining gap (after Round 10).** The CK DSL pipeline still
trails CK Tile C++ by ~3× on `decode_T1` (0.355 ms vs 0.121 ms).
The remaining gap is the same structural one Round 9 named:
mega-kernel fusion that keeps gate / up / SiLU / down / reduce
accumulators in registers across phases, removes the
`HiddenPadded` / `DownOut` HBM round-trips, and uses an in-kernel
grouped-GEMM dispatcher with chiplet-aware swizzling. That is
multi-week kernel-authoring work, not a session-scoped pass.

**Round 11: active-tile dispatch in the GEMM kernels.** Direct
empirical evidence (`E ∈ {2,4,8,16}` sweep with topk×tokens=2)
showed that DSL e2e time grows ~linearly with `experts`, not with
`min(experts, topk*tokens)` — adding 14 inactive experts costs
~314 µs of pure waste at decode_T1. CK Tile avoids this by reading
`sorted_expert_ids[block_id_z]` and skipping the kernel body for
inactive sorted tiles. Round 11 ports the same idea into the DSL
GEMM kernels:

- New `TraitSpec.active_tile_skip` field (default `False`).
- `instances/gemm_universal.py::build_universal_gemm` and
  `instances/moe_gemm_fused.py::build_moe_interleaved_gate_up_silu_gemm`
  honor the trait by taking two extra args (`SortedTokenIds` ptr +
  `slot_size` scalar) and at CTA entry compute a wave-uniform
  `do_work = SortedTokenIds[block_id_z * slot_size + block_m_off] >= 0`.
  The K-loop + epilogue are wrapped in `scf.if(do_work)` so an
  inactive (`first_row_token == -1`) tile does one bucket-head load
  and exits — no MFMAs, no LDS reads, no HBM stores.
- The kernel name carries an `actt` flag so HSACOs don't alias.
- `instances/fused_moe_e2e.py::FusedMoeForwardSpec.active_tile_skip_gemms`
  is the orchestrator-side knob. It activates on the static and
  dynamic forward paths and composes with all three preshuffle
  knobs via a parameterized launcher cache
  (`_moe_batched_gemm_launcher`,
  `_moe_interleaved_gate_up_silu_launcher`).

Standalone BatchedGemm validation (`examples/gfx950/moe/test_active_tile_skip.py`):

```text
B=8 M=32 N=4096 K=4096, tile=(32,128,64)
[parity] all-active   max|base - att|=0.0000  (need 0.0)
[parity] half-skipped max|expected - att|=0.0000  (need 0.0)
[parity] half-skipped inactive-rows max|C|=0.0000  (need 0.0)
[time] base (no skip)      =  172.31 us
[time] att, all-active     =  173.85 us  (0% overhead when nothing skips)
[time] att, all-inactive   =   13.10 us  (13× faster — the kernel just exits)
```

End-to-end (HIP-graph-replayed, all knobs at the optimum vs
baseline interleaved):

```text
                              baseline  preshuf_intl  +ATS    speedup vs base
decode_T1 H=4096 I=4096       269.4 us  227.5 us      166.3   1.62×
decode_T1 H=4096 I=7168       412.7 us  362.8 us      264.2   1.56×
decode_T8 H=4096 I=4096       488.4 us  229.1 us      227.5   2.15×
decode_T8 H=4096 I=7168       382.5 us  365.0 us      351.8   1.09×
decode_T1 E=16 I=7168         620.1 us  544.9 us      264.0   2.35×
```

vs CK Tile C++ on the production-canonical decode (router included
in DSL timing, fp16, 30 attempts, 5 warmup):

```text
decode_T1_E8_K2_H4096_I7168 :
    rocke baseline     0.379 ms  (0.32× of cktile)
    rocke preshuf      0.357 ms  (0.34× of cktile)
    rocke preshuf+ATS  0.265 ms  (0.46× of cktile)
    ck_tile_cpp         0.121 ms
```

ATS closes ~30 % of the gap to CK Tile on the canonical decode and
nearly half the gap on `decode_T1_E16` (where 7 of 8 inactive
experts get skipped). On already-fully-active shapes (small,
prefill_T128 with high topk*tokens density), ATS is within ±1 % of
baseline — measured zero overhead. Bitwise parity (`rel=0.00e+00`)
on every test scenario in `examples/gfx950/moe/test_fused_moe_preshuffle.py`.

The lever is exposed via:

- Spec: `FusedMoeForwardSpec.active_tile_skip_gemms` (default `True`).
- Env (in `examples/gfx950/moe/fused_moe_e2e_perf.py`):
  `ROCKE_ACTIVE_TILE_SKIP_GEMMS` — the harness defaults it **on**
  (`"1"`); set `ROCKE_ACTIVE_TILE_SKIP_GEMMS=0` to force the dense
  GEMM path. Composes with `ROCKE_PRESHUFFLE_W_DOWN=1` and
  `ROCKE_PRESHUFFLE_W_GATE_UP_INTERLEAVED=1`.

**Remaining gap (after Round 11).** decode_T1 canonical at 2.18×
of CK Tile (0.265 ms vs 0.121 ms). At decode_T8 / prefill (high
active-tile density), ATS provides almost nothing because every
expert is doing real work; the remaining gap there is the same
structural lever Round 9/10 named — gate/up→down LDS bridge to
remove the `HiddenPadded` HBM round-trip, plus a CK-Tile-style
flatmm micro-kernel with wider N tiles than the LDS budget allows
today.

**Summary of the kept wins and the residual gap.** Across the rounds
above, the 4-5× wins originally claimed vs torch eager were measurement
bias (Round 5, corrected to 1.26-2.60×); the wins vs the purpose-written
Triton baseline are real (36-42× on the H=4096 shapes); and the gap to
the CK Tile C++ binary, while narrowed by the preshuffle-B (Round 10)
and active-tile-skip (Round 11) levers, remains. Both real levers are
now shipped flags — `preshuffle_w_*` and `active_tile_skip_gemms` are
default-on or env-toggleable knobs, not no-ops. Closing the residual
gap further would require kernel-authoring work beyond config flips:

- A single-kernel re-implementation in CK DSL (significant effort;
  new authoring pattern beyond the current `FusedMoeForward`
  composer) covering gate / up / SiLU / down / weighted-reduce in
  one launch with shared LDS+register state across phases.
- An in-kernel grouped-GEMM dispatcher that removes the host
  roundtrip (also significant infrastructure work).

These are explicit follow-ups, not regressions of the runbook
methodology — the runbook's §17.4 lesson "at the last few percent,
generic levers stop helping; you have to port specific
implementations that already solved the problem" applies directly.

### Caveats / known issues

- **AITER `e2e_moe` Triton kernel crashes** with a memory-access
  fault on MI355X / gfx950 regardless of torch / ROCm version (likely
  an AITER kernel-arch mismatch). The script's `run_triton` path uses
  a purpose-written single-kernel fallback instead. Re-enable AITER
  once the gfx950 issue is fixed upstream.
- **`tile_example_fused_moe` C++ binary crashes** with a
  memory-access fault on E=8 / H=4096 / I=7168 shapes; the harness
  treats this as a skip (the row prints `(skip)` and the report's
  `ck_tile_cpp` field is `null`). The CK DSL row is unaffected.
- The CK DSL pipeline does **per-expert grouped-GEMM dispatch via a
  Python loop** with a small device-to-host copy of `Counts` and
  `Offsets` per dispatch (in the dynamic path); the CK Tile C++
  mega-kernel does in-kernel grouped-GEMM dispatch with no host
  roundtrip. This is the largest known overhead in the DSL path for
  shapes where per-expert GEMMs are small. HIP graph capture (used in
  static-offset mode) hides this on shapes where the routing is
  shape-stable.

### JSON report layout

Passing `--report PATH` writes a JSON list of per-scenario records:

```jsonc
[
  {
    "scenario": {
      "name": "decode_T1_E8_K2_H4096_I7168",
      "tokens": 1, "experts": 8, "topk": 2,
      "hidden": 4096, "intermediate": 7168,
      "dtype": "f16"
    },
    "results": {
      "rocke":      { "backend": "rocke",      "ok": true,  "ms": 0.443, "max_abs": 0.000488, "mean_abs": ..., "rel_max": ... },
      "torch_eager": { "backend": "torch_eager", "ok": true,  "ms": 0.920, "max_abs": 0.0,      "mean_abs": 0.0, "rel_max": 0.0 },
      "triton":      { "backend": "triton",      "ok": true,  "ms": 14.912, "max_abs": 0.000122, ... },
      "ck_tile_cpp": { "backend": "ck_tile_cpp", "ok": true,  "ms": 0.124, "max_abs": null, "mean_abs": null, "rel_max": null,
                       "note": "C++ binary, perf-only" }
    }
  },
  ...
]
```

---

## Tile / activation-barrier tuner: `tune_gate_up_silu.py`

This script tunes the experimental fused gate / up / SiLU MoE GEMM.
It compares **three activation-barrier strategies** across **five
MFMA tile candidates** for a given scenario.

### Three activation-barrier paths

1. **`packed`** — packed gate + up batched GEMM (`N = 2 × intermediate`)
   followed by a packed `silu_mul` post-pass. Selected when both
   experimental fused flags are off. One activation barrier (`silu_mul`
   after the GEMM).
2. **`dual`** — dual-B MFMA gate + up GEMM with the SiLU epilogue
   folded into the kernel. No separate `silu_mul` pass. Drives the
   `use_experimental_fused_gate_up_silu` spec flag.
3. **`interleaved`** — interleaved single-B MFMA gate + up GEMM with
   the SiLU epilogue folded into the kernel. The two GEMMs share a
   single B-operand load path. Drives the
   `use_experimental_interleaved_gate_up_silu` spec flag, which is
   **on by default** in `FusedMoeForwardSpec` — so `interleaved` is the
   production-default activation path, not `packed`.

### Five tile candidates

Each candidate is a `TileSpec` from `instances/gemm_universal`:

| Name | tile (M×N×K) | warps (M×N) | atom |
|---|---|---:|---|
| `t16n128k64_w1x1_atom16` | 16×128×64 | 1×1 | 16×16×16 |
| `t16n256k64_w1x2_atom16` | 16×256×64 | 1×2 | 16×16×16 |
| `t32n128k64_w2x1_atom16` | 32×128×64 | 2×1 | 16×16×16 |
| `t32n256k64_w2x2_atom16` | 32×256×64 | 2×2 | 16×16×16 |
| `t32n128k64_w1x2_atom32` | 32×128×64 | 1×2 | 32×32×16 |

The grid is intentionally small: every candidate compiles a fresh
HSACO for the fused kernel and one for the batched down GEMM. The
focus is **static-offset shapes** (decode / small batch) where HIP
graph capture is valid and per-launch overhead is already amortized.

### Running

```bash
cd <composablekernel-checkout>
PYTHONPATH=Python python \
  Python/rocke/examples/gfx950/moe/tune_gate_up_silu.py \
  --scenario {small, decode1, decode8} \
  --attempts 10 --warmup 3
```

### Latest results (MI355X, gfx950, ROCm 7.2, torch 2.12)

`--attempts 10 --warmup 3`. Latency in ms (mean over 10 graph-replay
iterations); lower is better. The best per scenario is bolded.

#### Scenario `small_T32_E4_K2_H128_I256`

| Tile | packed | dual | interleaved |
|---|---:|---:|---:|
| `t16n128k64_w1x1_atom16` | 0.0469 | 0.0513 | 0.0448 |
| `t16n256k64_w1x2_atom16` | 0.0504 | 0.0571 | 0.0493 |
| `t32n128k64_w2x1_atom16` | 0.0488 | 0.0531 | 0.0477 |
| `t32n256k64_w2x2_atom16` | 0.0622 | 0.0690 | 0.0610 |
| `t32n128k64_w1x2_atom32` | 0.0391 | 0.0405 | **0.0383** |

#### Scenario `decode_T1_E8_K2_H4096_I7168`

| Tile | packed | dual | interleaved |
|---|---:|---:|---:|
| `t16n128k64_w1x1_atom16` | 0.6344 | 1.0609 | 0.6247 |
| `t16n256k64_w1x2_atom16` | 0.8072 | 1.2452 | 0.7251 |
| `t32n128k64_w2x1_atom16` | 0.9454 | 0.9313 | 0.9721 |
| `t32n256k64_w2x2_atom16` | 1.4887 | 1.2646 | 1.2431 |
| `t32n128k64_w1x2_atom32` | 0.4257 | 0.6249 | **0.4161** |

#### Scenario `decode_T8_E8_K2_H4096_I7168`

| Tile | packed | dual | interleaved |
|---|---:|---:|---:|
| `t16n128k64_w1x1_atom16` | 1.4840 | 1.0717 | 0.6347 |
| `t16n256k64_w1x2_atom16` | 0.8272 | 1.3995 | 0.7407 |
| `t32n128k64_w2x1_atom16` | 0.9639 | 0.9481 | 0.9633 |
| `t32n256k64_w2x2_atom16` | 1.2357 | 1.2772 | 1.3394 |
| `t32n128k64_w1x2_atom32` | 0.4412 | 0.4582 | **0.4225** |

### Findings

- **The 32×32 MFMA atom wins on every scenario.** The
  `t32n128k64_w1x2_atom32` tile is the best across all three
  scenarios, regardless of the activation-barrier path.
- **`interleaved` is the best path for the winning tile** on every
  scenario — it ties `packed` on small and beats `packed` / `dual`
  on both decode shapes.
- **The 16×16-atom tiles are dominated** on decode shapes — the
  best 16×16 row (`t16n128k64_w1x1_atom16` / `interleaved`) is
  ~50 % slower than the 32×32-atom winner on decode_T1 and
  ~50 % slower on decode_T8.
- **`packed` is competitive on small shapes** but loses to
  `interleaved` (the production default) once the per-expert GEMMs
  grow beyond the small-validation regime.
- **Correctness** (`max_abs` vs torch reference) is identical
  across paths on each scenario: 1.9e-6 on small, 4.9e-4 on the two
  decode scenarios — well within f16-with-fp32-accumulator tolerance.

The 32×32-atom + interleaved variant this sweep selects is exactly
what `FusedMoeForwardSpec` ships as its default
(`use_experimental_interleaved_gate_up_silu=True` plus the
`_default_gemm_tile()` 32×32×16 tile), so the tuner's finding is
already the production configuration. Re-run it after changing the
per-expert GEMM shapes to confirm the same tile still wins.

---

## Reproducibility

```bash
# E2E perf
cd <composablekernel-checkout>
export AITER_PATH=<aiter-checkout>
PYTHONPATH="Python:${AITER_PATH}" python \
  Python/rocke/examples/gfx950/moe/fused_moe_e2e_perf.py \
  --attempts 10 --warmup 3 --report /tmp/moe_perf.json

# Tuner: small / decode1 / decode8
PYTHONPATH=Python python \
  Python/rocke/examples/gfx950/moe/tune_gate_up_silu.py \
  --scenario small --attempts 10 --warmup 3
PYTHONPATH=Python python \
  Python/rocke/examples/gfx950/moe/tune_gate_up_silu.py \
  --scenario decode1 --attempts 10 --warmup 3
PYTHONPATH=Python python \
  Python/rocke/examples/gfx950/moe/tune_gate_up_silu.py \
  --scenario decode8 --attempts 10 --warmup 3
```

`AITER_PATH` only needs to be set if you want the script's
`run_triton` path to import `aiter`-bundled Triton dependencies. The
purpose-written Triton kernel itself only needs `triton` to be
importable in the active environment.

---

## Parity tests

Three self-contained parity + perf checks back the levers discussed in
the optimization log. Each runs directly (no required flags) and exits
non-zero on a parity failure; each also accepts optional shape flags
(see `--help`). All require a gfx950 GPU.

```bash
cd <composablekernel-checkout>

# End-to-end parity + perf for the preshuffle / active-tile-skip knobs
# on FusedMoeForward (HIP-graph-replayed, 6 scenarios). Optional:
# --warmup N (default 10), --iters N (default 50).
python Python/rocke/examples/gfx950/moe/test_fused_moe_preshuffle.py

# Standalone BatchedGemm parity + perf for trait.preshuffle_b=True.
# Optional: --B --M --N --K --tile_m --tile_n --tile_k
#           --pipeline --scheduler --epilogue --iters
python Python/rocke/examples/gfx950/moe/test_preshuffle_b.py

# Standalone BatchedGemm parity + perf for trait.active_tile_skip=True
# (all-active vs all-inactive timing). Optional:
# --B --M --N --K --tile_m --tile_n --tile_k
python Python/rocke/examples/gfx950/moe/test_active_tile_skip.py
```

`test_fused_moe_preshuffle.py` gates every variant at `rel ≤ 5e-2` vs
the no-preshuffle baseline (the levers are bitwise-identical in
practice — `rel = 0` is reported on every scenario). `test_preshuffle_b.py`
gates the preshuffled kernel against a torch reference (K-scaled f16
tolerance) **and** bitwise (`≤ 1e-3`) against the non-preshuffled
kernel. `test_active_tile_skip.py` gates both the all-active and the
half-skipped cases bitwise (`≤ 1e-3`), and checks that skipped output
rows stay zero.
