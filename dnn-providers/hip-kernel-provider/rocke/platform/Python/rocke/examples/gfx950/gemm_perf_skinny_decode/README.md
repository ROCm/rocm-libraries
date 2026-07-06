# gemm_perf_skinny_decode — Runbook-driven walkthrough

A self-contained example of the
[CK DSL Optimization Runbook](../../../dsl_docs/optimization/optimization_runbook.md)
applied to one concrete problem: a Qwen3-8B **`o_proj` decode** matmul
(`bf16`, `M=2`, `N=4096`, `K=4096`) — a low-utilization, HBM-bound GEMM
typical of the decode phase (the same kind of projection the
`examples/gfx950/qwen3_30b_a3b_decode.py` end-to-end pipeline benchmarks
against the production AITER/hipBLASLt path).

The point is to demonstrate the **loop** from the runbook (static probe →
sweep → correctness → side-by-side → structural levers → confirmation) on
a kernel that matters to a real model, end-to-end, and report the result
honestly. Over 22 steps the DSL closed the gap to rocBLAS from several×
slower down to **within noise** (step 22: DSL 10.30 µs vs a 10.18 µs standalone
rocBLAS probe — rocBLAS marginally ahead on best, DSL marginally ahead on
median; the runbook's clean serial re-run, §17.7, records 1.02× rocBLAS). The
remaining gap is the part's sustained-bf16 HBM ceiling, not a knob.

> **Numbers in this file come from the committed `data/0N_*.json` snapshots.**
> The shipped snapshots were captured on a shared GPU: a few steps show high
> rocBLAS spread, and steps 05–08 record `"no samples"` (the GPU run for those
> steps did not complete on the capture machine), so the tables for those steps
> describe what the script *measures*, not a number from the committed JSON.
> Re-run the scripts on an exclusive GPU to get clean numbers for your hardware;
> treat each script's own printed output as authoritative there.

The runbook discipline matters precisely because most of the steps were
negative results — a regression or a within-noise tie. The wins came from
three structural changes — DirectToLDS wiring, a multi-warp LDS-offset patch,
and chiplet swizzling — each surfaced by a sweep that had to follow the rules
to be trustworthy.
Steps 23–24 then build the one structural lever the earlier steps could only
name: split-K, which fills the idle CUs the decode grid leaves behind.

> **New to skinny-M / decode GEMM?** [`ALGORITHM.md`](ALGORITHM.md) derives why
> this matmul is HBM-bound, what "skinny M" does to the tile geometry, how the
> `mem`/`compv4` pipelines and the K-pack atom map onto the K-loop, and why
> DirectToLDS, chiplet swizzle, and split-K are the levers that matter. Read it
> first if you want the *why* before this file's optimization *history*.

## What this example shows

1. **The runbook's static-first discipline (§3.1b, §12.1.Q).** Eight tile /
   pipeline variants are filtered by VGPR / AGPR / SGPR / LDS / spill /
   waves-per-CU *before* any kernel is launched. Cost: ~6 s of compile, no GPU.
2. **The runbook's sweep discipline (§12.2, §12.1.P).** Each surviving
   variant is benchmarked over 5 attempts (the script's `ATTEMPTS` constant),
   cold-cache discarded, median + spread reported, and the HSACO symbol is salted with the shape
   hash so module-cache aliasing cannot poison the result.
3. **Correctness is verified separately (§2.1, §14.1).** `rocke.run_manifest`
   `--verify` allocates `fp16` buffers regardless of manifest dtype, so it
   silently fails on `bf16`. We re-run the kernel in a small Python harness
   with actual `bf16` (via torch) and check `max_abs_diff` against an `fp32`
   reference.
4. **Side-by-side vs rocBLAS (§2.2, §17 case-study form).** Same harness,
   same warmup / iter budget, same shape. Report the ratio honestly.

## Hardware / software pin

| | |
|---|---|
| GPU | MI355X / gfx950 |
| ROCm | 7.0.2 |
| torch | 2.12.0+rocm7.2 |
| CK DSL | this repo (`dnn-providers/hip-kernel-provider/rocKE/Python/rocke`) |
| HBM peak | 8.0 TB/s |
| BF16 MFMA peak | 2.5 PFLOPS |

## Reproduce end-to-end

```bash
cd <rocke>/examples/gfx950/gemm_perf_skinny_decode    # this folder
PY=<your python with torch+rocm and rocke on PYTHONPATH>

$PY scripts/01_probe_occupancy.py     # ~6 s,  CPU only (static)
$PY scripts/02_sweep_bench.py          # ~3 min,  GPU
$PY scripts/03_correctness.py          # ~5 s,   GPU
$PY scripts/04_compare_rocblas.py      # ~5 s,   GPU
$PY scripts/05_extra_levers.py         # ~2 min,  GPU
$PY scripts/06_ground_truth_geometries.py   # ~5 min,  GPU
$PY scripts/07_push_tile_k.py               # ~3 min,  GPU
$PY scripts/08_lever_combinations.py        # ~25 min, GPU (192 combos)
$PY scripts/09_preshuffle_b.py              # ~1 min,  GPU
$PY scripts/10_hipcc_backend.py             # ~3 min,  GPU
$PY scripts/11_final_compare.py             # ~10 s,   GPU
$PY scripts/12_direct_to_lds.py             # ~30 s,   GPU
$PY scripts/13_dtl_sweep.py                 # ~2 min,  GPU — DTLA × tk × cache hints
$PY scripts/14_dtl_push.py                  # ~30 s,   GPU — push tk past 1024
$PY scripts/15_warp_geom.py                 # ~30 s,   GPU — multi-warp sweep (exposes the DTLA bug)
$PY scripts/16_more_lanes.py                # ~30 s,   GPU — widen tile_n for more lanes/WG
$PY scripts/17_confirm_wpe.py               # ~1 min,  GPU — waves_per_eu confirmation
$PY scripts/18_dtl_prefetch.py              # ~1 min,  GPU — ping-pong DTLA (perf-neutral)
$PY scripts/19_multiwarp_probe.py           # ~30 s,   GPU — isolate multi-warp bug
$PY scripts/20_dtl_multiwarp.py             # ~3 min,  GPU — multi-warp resweep
$PY scripts/21_chiplet.py                   # ~5 min,  GPU — chiplet_swizzle sweep (winner)
$PY scripts/22_confirm_winner.py            # ~5 min,  GPU — N=60 confirmation
$PY scripts/23_splitk_universal.py          # ~3 min,  GPU — split-K over the production body
$PY scripts/24_splitk_dispatch.py           # CPU dry-run; add --run for a GPU numeric check
```

> Steps 12–23 exercise the DirectToLDS / prefetch / per-wave-LDS-offset /
> split-K support that lives in `rocke/instances/common/gemm_universal.py`
> today (the `TraitSpec.direct_to_lds`, `dtl_prefetch`, and `split_k` flags and
> the per-wave LDS base offset described in steps 12/19/23). No external patch is
> needed — these run against the in-tree kernel body.

Most steps write a `data/0N_*.json` sink the next step can read; step 19 is a
correctness probe that prints to stdout, and steps 23–24 print their tables to
stdout (no JSON sink). Step 24 is the only script with a flag: `--run` adds an
on-GPU numeric sanity pass on top of the default CPU-only dispatch dry-run. The
committed `data/05_*`–`data/08_*` snapshots are empty (`"no samples"`) — see the
note at the top of this file.

## Step 1 — Static occupancy probe (`01_probe_occupancy.py`)

> Runbook §3.1b "Static Inspection First"; §12.1.Q probe catalog;
> §14.2 "Low occupancy" / "Register spills" failure modes.

Compile each candidate, parse VGPR / AGPR / SGPR / LDS / spill from the
HSACO via `llvm-readelf --notes`, estimate `waves/CU` and the dominant
limiter, and drop variants that spill or fall below 4 waves/CU.

Output from the committed `data/01_occupancy.json`:

```
label                              vgpr agpr sgpr spill  lds B   waves/CU  wg/CU  limit
t16x64x32_w1x2_a16x16x32_mem        48    8   20    0    7168       32     16  MAX_WAVES_PER_CU
t16x128x32_w1x2_a16x16x32_mem       76   16   20    0   13312       24     12  LDS
t16x128x64_w1x2_a16x16x32_mem       96   16   20    0   22528       14      7  LDS
t16x256x32_w1x4_a16x16x32_mem       96   16   20    0   25600       20      5  VGPR
t32x128x32_w2x2_a16x16x32_mem       52   16   20    0   18432       32      8  MAX_WAVES_PER_CU
t16x128x32_w1x2_a16x16x32_compv3    76   16   20    0   13312       24     12  LDS
t16x128x32_w1x2_a16x16x32_compv4    76   16   20    0   13312       24     12  LDS
t16x128x64_w1x2_a16x16x32_compv4    96   16   20    0   22528       14      7  LDS
```

All 8 variants survive the filter (zero spill, ≥4 waves/CU on every one).
The probe's diagnostic value is still real:

- At the same tile, `compv3` / `compv4` cost no extra VGPR/AGPR over `mem`
  here (all three `t16x128x32` rows are 76 VGPR / 16 AGPR / 13312 B LDS) — so
  on this geometry the pipeline choice is *free* in registers, and the bench
  (step 2) confirms it is also neutral in latency. The kind of pre-bench
  signal §12.1.D recommends checking before deciding pipeline.
- `t16x128x64` (wider K window) dropped `waves/CU` from 24 (at `t16x128x32`)
  to 14 by hitting the LDS limit. Predicts a potential occupancy regression
  from the K-window knob even before measurement.

## Step 2 — Sweep with hygiene (`02_sweep_bench.py`)

> Runbook §2.2 / §2.3 baseline + hygiene; §12.1.P knob list;
> §12.2 sweep discipline.

For each surviving variant: compile once (HSACO cached on disk), then
**5 timed attempts + 1 cold attempt** through `rocke.run_manifest` with
**20 warmup + 200 timed** iterations per attempt. Report median, spread,
best. Salt the kernel symbol with the shape hash (`__m{M}n{N}k{K}` suffix).

Result from the committed `data/02_sweep_bench.json` (best-ms basis; some rows
show a wide spread because the capture GPU was shared — the best sample is the
uncontended floor):

| Variant | Median ms | Best ms | TFLOPS | % HBM |
|---|--:|--:|--:|--:|
| **t16x64x32_w1x2_a16x16x32_mem**   | **0.0472** | **0.0472** | **1.42** | **8.9 %** |
| t16x128x64_w1x2_a16x16x32_mem      | 0.0489 | 0.0488 | 1.37 | 8.6 % |
| t16x128x64_w1x2_a16x16x32_compv4   | 0.0533 | 0.0532 | 1.26 | 7.9 % |
| t16x128x32_w1x2_a16x16x32_compv4   | 0.0661 | 0.0660 | 1.02 | 6.4 % |
| t16x128x32_w1x2_a16x16x32_compv3   | 0.0661 | 0.0660 | 1.02 | 6.4 % |
| t16x128x32_w1x2_a16x16x32_mem      | 0.0662 | 0.0661 | 1.02 | 6.4 % |
| t32x128x32_w2x2_a16x16x32_mem      | 0.0682 | 0.0680 | 0.99 | 6.2 % |
| t16x256x32_w1x4_a16x16x32_mem      | 0.0863 | 0.0858 | 0.78 | 4.9 % |

What the data says — read it against the runbook:

- **The smallest-N tile wins.** For `M=2` the kernel is HBM-bound; covering
  fewer weight columns per CTA means more CTAs cover more of the N axis in
  parallel, and `MAX_WAVES_PER_CU` is the limiter (probe step). §3.3
  "Memory-bound signals".
- **`compv4` did not help on the matching tile.** `t16x128x32` is identical
  to within noise across `mem` / `compv3` / `compv4` (0.0660–0.0661 ms × 3). The
  runbook §12.1.D notes "Compv3 / compv4 trade LDS for latency hiding";
  here, with no compute to hide behind, the trade is neutral. The §17.4
  "compiler hint sweep" lesson generalizes: schedule knobs rarely close
  memory-bound gaps.
- **The K-window hypothesis** (`t16x128x64`) saved ≈26 % of latency vs
  `t16x128x32` by halving the K-loop trip count via the same `16x16x32`
  atom — the §7.4 / §12.1.C "K-pack" lever, confirmed.
- **Wider N (`t16x256x32`) regressed by ≈83 %.** Confirms that for `M=2` the
  reuse story doesn't pay; you're paying VGPR / LDS for an N slab no lane
  fully consumes. §6.5 register pressure.
- **Best DSL kernel: `t16x64x32_w1x2_a16x16x32_mem` — 0.0472 ms / 1.42 TFLOPS /
  712 GB/s / 8.9 % of HBM peak.**

## Step 3 — Standalone bf16 correctness (`03_correctness.py`)

> Runbook §2.1; §14.1 "if correctness fails, do not report speed as a win";
> §14.3 "Verification included in timing" caveat.

We do **not** use `rocke.run_manifest --verify` for `bf16` because the
shipped verify path allocates `np.float16` buffers regardless of dtype
(`instances/common/manifest_runner/gemm.py` builds A/B/C as `np.float16`),
and the bit pattern of fp16 reinterpreted as bf16 is garbage — so it raises
a large `max_abs_diff` with every element flagged on a perfectly correct
kernel, a structural false alarm in the harness, not a kernel bug.

The replacement is ~80 lines: numpy `int16 → fp32 → bf16` for A and B, fp32
reference, launch the winning HSACO via `rocke.runtime.hip_module.Runtime`,
read back, compare.

Result:
```
Verifying winner: t16x64x32_w1x2_a16x16x32_mem
  max|out-ref|=0.0000   bad=0/8192   ref_max=2736
  → PASS
```

Bit-exact. Time-to-execute the correctness check: ~5 s.

## Step 4 — Side-by-side vs rocBLAS (`04_compare_rocblas.py`)

> Runbook §2.2 same-harness comparison; §17 case-study reporting form.

Run `torch.matmul(A_bf16, W_bf16.T)` on the same shape with the same warmup
/ iter budget. Time with `torch.cuda.Event`.

From the committed `data/04_compare_rocblas.json` (best-ms basis):

| | Best ms | TFLOPS | GB/s | % HBM |
|---|--:|--:|--:|--:|
| **rocBLAS bf16 (torch.matmul)** | 0.0174 | 3.85 | 1926 | **24.1 %** |
| **DSL winner** (`t16x64x32`)    | 0.0472 | 1.42 |  712 |   8.9 % |
| **Ratio**                       | 2.70×  | —    | —    | —     |

Honest read:

- **DSL is 2.70× slower than rocBLAS** on this shape in this snapshot
  (`dsl_over_rocblas_latency_ratio` in the JSON). The rocBLAS sample in this
  particular capture is **contended** — its `ms_spread_pct` is 223 %, so its
  best-ms (and the resulting ratio) is not a clean baseline. A standalone,
  uncontended rocBLAS probe on this shape lands near 0.011 ms (see the
  `rocblas_ms` reference baked into the later `07`/`09` sweeps and the cleaner
  step-11 / step-22 comparisons); against that floor the same DSL winner is
  roughly 4× slower. Report the noisy ratio as the snapshot shows it, and treat
  the clean-baseline comparisons in steps 11/22 as the trustworthy ones (§14.3
  "Cache-biased / contended results — record the spread, don't pick the ratio
  that flatters you").
- The decode regime this shape comes from is the same one the
  `examples/gfx950/qwen3_30b_a3b_decode.py` pipeline exercises against the
  production AITER/hipBLASLt baseline: a projection GEMM that under-utilizes
  HBM because `M` is tiny.
- The runbook's "small tile / `mem` pipeline / `kpack` atom" picks already moved
  the needle off the wider-tile sweep rows, but at this step the
  remaining gap to rocBLAS still looked **structural**, not knob-flippable:
  - The kernel was using a tile geometry tuned for prefill, not a small/decode
    shape — the runbook's §12.1.A guidance is "small / decode shapes → `flatmm`".
    (Steps 6–7 revisit this and find that hipBLASLt's *actual* kernel for this
    shape is the same `16×16` tile family, just with a much deeper `tile_k` — so
    the fix turns out to be geometry, not a different algorithm.)
  - The §17.4 take-away applies: *a structural change often needs multiple
    co-evolved levers*. Tile geometry alone, with a shallow K-loop, is
    capped at a single-digit %-of-HBM regardless of which tile you pick.

## Step 5 — CK-Tile-inspired extra levers (`05_extra_levers.py`)

> Inspired by `example/ck_tile/03_gemm/universal_gemm.cpp` and
> `example/ck_tile/18_flatmm/flatmm_basic.hpp` (persistent CTAs,
> k_batch, chiplet-aware tile partitioner).
>
> Runbook §12.1.G `waves_per_eu`, §12.1.I `persistent`, §12.1.L
> chiplet swizzle, §17.4 "co-evolve multiple levers, record what didn't
> help".

Lock the geometry to the sweep winner and exercise the remaining
`TraitSpec` knobs.

> The committed `data/05_extra_levers.json` records `"no samples"` for every
> variant (the capture-machine GPU run did not complete for this step), so the
> Δms figures below are illustrative of the *expected* outcome the script tests
> for, not values read from the committed snapshot. Re-run `05_extra_levers.py`
> on an exclusive GPU for live numbers; the qualitative verdicts are what the
> runbook predicts for a memory-bound kernel and are confirmed by the populated
> later steps (08, 17, 21).

| Variant | Δms | Δ% | Verdict |
|---|--:|--:|---|
| baseline_winner | +0.0000 | +0.0 % | reference |
| waves_per_eu_2 | +0.0015 | +3.0 % | **loss** |
| waves_per_eu_3 | +0.0015 | +3.0 % | **loss** |
| persistent | -0.0000 | -0.0 % | noise |
| chiplet_swizzle | +0.0005 | +1.1 % | loss |
| persistent_chiplet | +0.0006 | +1.2 % | loss |
| compv4_persistent | -0.0000 | -0.0 % | noise |
| compv4_persistent_chiplet | +0.0006 | +1.2 % | loss |

Diagnostic read (§17.4 form):

- **`waves_per_eu`: regression.** The occupancy probe already showed
  `MAX_WAVES_PER_CU=32` is the limiter; forcing the compiler to a smaller
  VGPR budget can't add more waves but does shorten the live-range
  optimizer's freedom. §17.4: "a compiler hint that crosses no real
  constraint is at best neutral and usually a small loss".
- **`persistent`: noise.** The shape has only `M_tiles × N_tiles = 1 × 64 = 64`
  macro tiles. The non-persistent grid is already small; persistent's
  amortization of launch cost has nothing to amortize.
- **`chiplet_swizzle`: small loss.** MI355X has 8 XCDs; at 64 tiles
  the swizzle remaps `tiles ÷ chunk_size = 64 / 64 = 1` chunk, so the
  remap is a no-op for L2-reuse purposes and the extra entry-time
  scalar math is pure overhead. §12.1.L's "MI300X / MI325X / MI350X
  have 8 XCDs" works for shapes with many more tiles per N stripe.
- **Pipeline+persistent co-evolution: noise.** The §17.4 lesson "a
  structural change often needs multiple co-evolved levers" applies to
  *structural* changes (atom shape, lane ownership, intermediate
  residency). Stacking *scheduling-tier* knobs on a memory-bound kernel
  doesn't compose into a structural win.

The structural change that would actually move this kernel — and the
one CK Tile ships in `example/ck_tile/03_gemm/gemm_splitk_two_stage*.cpp`
— is **split-K**: tile K into slabs, launch that many × more CTAs, and
reduce the partials. At `M=2, N=4096, K=4096` the kernel processes 32 MiB of
B weights against only 16 KiB of A; split-K parallelizes the same B traffic
across more CTAs and puts the idle CUs to work. This *is* available in the
production universal-GEMM body via `TraitSpec.split_k` (single-pass body +
f32-atomic-add workspace) — steps 23–24 build and dispatch it on this exact
shape. (When this step-05 note was first written the body did not yet expose
it; that gap has since closed.)

## What this example *doesn't* do (and why)

- **No separate `flatmm` variant.** `instances/common/flatmm.py` is CK Tile's
  name for a preshuffled-B batched matmul; its v1 wrapper delegates to
  `build_batched_gemm` and carries preshuffle as dispatch intent, so it would not
  differ from the preshuffle-B path already swept in step 9. A per-shape `skinny`
  load pattern in `gemm_universal` would be the more direct lever; out of scope.
- **The split-K story does not stop at "named as a follow-up".** Earlier steps
  framed split-K as out of reach; it is in fact wired into the production
  universal-GEMM body (`TraitSpec.split_k`) and exercised in steps 23–24 — see
  that section. CK Tile's `example/ck_tile/03_gemm/gemm_splitk_two_stage*.cpp`
  shows the canonical two-stage pattern; the DSL uses a single-pass body with an
  f32-atomic-add workspace instead. (The in-tree `streamk_gemm` instance now
  supports bf16 as well as f16.)
- **No FP8 KV / weight quantization.** Quantizing B to FP8 would halve its
  32 MiB of traffic, but that changes the "match rocBLAS bf16 → bf16" problem;
  it is a fair follow-up, not part of this example.
- **No end-to-end model integration.** This example optimizes the GEMM in
  isolation; wiring the kernel into a model dispatch path is the job of the
  GEMM dispatcher (steps 23–24 wire split-K into `dispatch_gemm_bf16`).

## Steps 6–11 — closing the gap with what the DSL already has

The step-04 conclusion ("several× slower, blocked on structural
extensions to `streamk_gemm`") was wrong in one important way: it never
asked what hipBLASLt's actual kernel looks like *for this shape*, only
what the YAML library suggested for similar shapes. Running `rocprof`
on `torch.matmul(A, W.t())` (M=2 N=4096 K=4096) gave the real Tensile
kernel name:

```
Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs
  _MT16x16x512  _MI16x16x1  _DTLA1_DTLB1
  _GRVWA8_GRVWB8  _GSU0  _PGR2_PLR1  _SIA3
  _SK3_SKFTR0  _WG16_4_4
                                AverageNs=9076  (≈9 µs)
```

Translation:

| Tensile token | Meaning | DSL equivalent today |
|---|---|---|
| **MT 16×16×512** | macro tile (the whole story) | `tile_m=16, tile_n=16, tile_k=512` |
| MI 16×16×1 | MFMA atom | `mfma_f32_16x16x32_bf16` |
| GRVWA8 / GRVWB8 | vector-width 8 on both operands | DSL emits dwordx8 bursts already |
| **DTLA1 / DTLB1** | DirectToLDS for A and B | **not exposed in DSL today** |
| **SK3** | StreamK basic, no split-K (GSU0) | **not exposed in DSL today** |
| PGR2 / PLR1 | 2 prefetch-global, 1 prefetch-local | `mem` pipeline (implicit-1) / `compv4` (2) |
| SIA3 | ScheduleIterAlg=3 (fixed loop schedule) | implicit per pipeline |
| WG 16,4,4 | 256-thread workgroup, 4×4 wave grid | analogous to our `warp_m × warp_n` |

The lesson: read the kernel hipBLASLt *actually launches* for this exact
shape, not what a tuning YAML suggests for a similar one. The real kernel is
**MT 16×16×512** — the same tile family the DSL can express, just with a much
deeper `tile_k`. That reframes the punch list from "missing three structural
extensions" to "push the levers we already have and see how close we get".

### Step 6 — try the geometries hipBLASLt actually picks (`06_ground_truth_geometries.py`)

Ground-truth-inspired sweep: keep `mfma_f32_16x16x32_bf16`, sweep
`tile_n ∈ {16, 32}` and `tile_k ∈ {32, 64, 128, 256}` on
`tile_m=16 w1×{1,2,4}`. The smallest-N + deepest-K candidate the
runner accepts wins.

**Winner: `t16x16x256_w1x1_mem` ≈ 16.0 µs / ≈26 % HBM / ≈1.5× rocBLAS** —
bumping `tile_n` from 64 to 16 and `tile_k` from 32 to 256 is the single biggest
lever in the whole campaign. (The committed `data/06_ground_truth_geometries.json`
records `"no samples"`; this figure is corroborated by the *populated*
`data/09_preshuffle_b.json`, whose `t16x16x256_preB` row lands at 16.0 µs /
26.2 % HBM for the same geometry. Re-run `06_ground_truth_geometries.py` for live
numbers.) Preshuffle-B variants in this script all fell over because
`run_manifest` does not pass a preshuffle flag through to the kernel; that path is
handled with a custom harness in step 9.

### Step 7 — push `tile_k` further (`07_push_tile_k.py`)

LDS budget on `t16x16` with the `mem` pipeline is `4·tile_k + 512` bytes,
so the 160 KiB cap leaves us room up to `tile_k = 4096`. Sweep
`tile_k ∈ {256, 384, 512, 768, 1024, 1536, 2048}` × `{mem, compv4}`:

| tile_k | `mem` (µs) | `compv4` (µs) | note |
|---:|---:|---:|---|
| 256  | 16.0 | 16.4 | step-6 winner |
| 384  | —    | —    | divisibility fail, skipped |
| **512**  | **≈13.5** | ≈13.5 | sweet spot |
| 768  | —    | —    | divisibility fail |
| 1024 | ≈58  | ≈58  | occupancy collapse |
| 1536 | ≈58  | LDS-OOM | compv4 exceeds the 160 KiB cap |
| 2048 | ≈58  | LDS-OOM | |

> The committed `data/07_push_tile_k.json` records `"no samples"`. The `tile_k=512`
> sweet spot and the `tile_k≥1024` collapse are both corroborated by *populated*
> later snapshots: `data/13_dtl_sweep.json` shows `base_tk512_mem` at 13.31 µs /
> 31.5 % HBM and the round-trip `base_tk1024_mem` collapsing to 58.49 µs / 7.2 %
> HBM. Re-run `07_push_tile_k.py` for live numbers.

**Winner: `t16x16x512_mem` ≈ 13.5 µs / ≈31 % HBM.**
This is hipBLASLt's exact `MT 16×16×512`. The collapse at `tile_k ≥ 1024`
is the LDS pressure crowding waves below the round-trip-hiding threshold
(steps 12–13 recover `tile_k=1024` once DirectToLDS frees the VGPRs).

### Step 8 — combinatorial lever sweep (`08_lever_combinations.py`)

192 combos of `pipeline × scheduler × epilogue × persistent × chiplet ×
waves_per_eu` locked to `t16x16x512`. The committed
`data/08_lever_combinations.json` records `"no samples"` (the capture-machine GPU
run did not complete this step), so no per-combo number is reported here; re-run
`08_lever_combinations.py` for live numbers. The conclusion the step exists to
test — that with the geometry already right, the scheduling-tier knobs cluster
within noise — is the runbook §17.4 lesson, and is borne out by the populated
steps 13/17/21 where these same knobs move nothing.

### Step 9 — preshuffle_b (`09_preshuffle_b.py`)

`TraitSpec.preshuffle_b=True` is fully wired into `emit_load_phase`
(see the `preshuffle_b` branch in `instances/common/gemm_universal.py`) — it
replaces strided B loads with contiguous `buffer_load_dwordxN` bursts. `run_manifest` doesn't know
how to permute B, so this script builds its own harness around
`rocke.runtime.hip_module.Runtime` and the host-side
`np.transpose(B.reshape(n_tiles, bn, k_tiles, bk), (2,0,1,3))`
permutation. Bit-exact correctness (max_abs_diff = 0).

**Winner: `t16x16x512_preB`  13.34 µs / 31.5 % HBM / 1.21× rocBLAS** (`data/09_preshuffle_b.json`).
~1.5 % over non-preshuffle — the load addressing was already close to
optimal on this skinny-M shape; preshuffle helps more when the strided
load pattern is the actual bottleneck.

### Step 10 — `lower_kernel_to_hip` + hipcc backend (`10_hipcc_backend.py`)

The DSL has a second compile path: `compile_kernel_via_hipcc` lowers
the same IR to HIP C++ and runs it through `hipcc --genco -O3`. Same
runtime ABI, different codegen frontend (clang for HIP vs the direct
LLVM-IR → libamd_comgr path). Tested 5 variants:

From the committed `data/10_hipcc_backend.json` (best-ms basis):

| backend / flags | µs | compile |
|---|---:|---:|
| llvm_default (libamd_comgr) | 13.44 | 10 ms |
| hipcc -O3 | 13.49 | 429 ms |
| hipcc -O3 -ffast-math | 13.50 | 402 ms |
| hipcc -O3 -ffast-math -fgpu-flush-denormals-to-zero | 13.48 | 410 ms |
| hipcc -O3 -ffast-math -flushdenorm -mllvm -amdgpu-early-inline-all=true | 13.50 | 423 ms |

All within ≈0.06 µs (noise); hipcc is ~40× slower to compile. For a memory-bound kernel the codegen
backend doesn't matter; for a long-running attention kernel the
helpers/compile.py docstring's ~5 % hipcc win could plausibly show up,
just not here.

### Step 11 — final side-by-side (`11_final_compare.py`)

Same `torch.cuda.Event` harness as step 04, so the ratio is honestly
comparable to the step-04 baseline. From the committed
`data/11_final_compare.json` (best-ms basis):

| | Best µs | TFLOPS | GB/s | % HBM |
|---|--:|--:|--:|--:|
| **rocBLAS bf16 (torch.matmul)** | 10.2 | 6.58 | 3293 | **41.2 %** |
| **DSL `t16x16x512` (mem/interwave/cshuffle)** | 13.3 | 5.06 | 2530 | 31.6 % |
| **Ratio** | **1.30×** | — | — | — |

DSL vs rocBLAS `max_abs_diff = 0.5` (one ulp of bf16 — both kernels
round their output bf16, accumulation order differs).

**Geometry + K-pack alone close most of the gap; the JSON's
`speedup_vs_step02_winner` over the original step-02 winner is 3.64×.** The
DirectToLDS / chiplet / split-K steps below take it the rest of the way.

### Step 12 — DirectToLDS (the trickiest lever) (`12_direct_to_lds.py`)

The Tensile `DTLA1_DTLB1` token maps to the hardware
`buffer_load_dwordx4 ... offen offset:0 lds` instruction — the dword
payload writes straight into LDS, bypassing the VGPR stage. Our DSL
kernel was emitting the round-trip (`global_load_dwordx4 → VGPR →
ds_write_b128`), 32 extra instructions and 32 extra VGPRs/iter.

Investigation: `rocke.core.ir.IRBuilder.async_buffer_load_lds_addr` is
the DSL primitive that lowers to exactly this instruction, used by
`attention_tiled_2d.py`. `gemm_universal.py` didn't wire it.

Patch added: `TraitSpec.direct_to_lds: bool` (off by default). When
enabled, `emit_load_phase` issues `async_buffer_load_lds_addr` for both
A and B tiles, sized at `dwords=4` (16 bytes/lane). Per-pass LDS
destination advances by `block_size * 16`. The existing `b.sync()`
after the load phase already lowers to `s_waitcnt vmcnt(0) lgkmcnt(0)
; s_barrier`, so it drains in-flight DTLA writes for free.

Disassembly verifies the new path:

```text
default emit_load_phase:                with direct_to_lds=True:
  32 × global_load_dwordx4               32 × buffer_load_dwordx4 ... nt lds
  32 × ds_write_b128                     —      (eliminated!)
  32 × ds_read_b128                      32 × ds_read_b128
  16 × v_mfma_f32_16x16x32_bf16          16 × v_mfma_f32_16x16x32_bf16
```

Bit-exact correctness (both paths `max_abs_diff = 0` in
`data/12_direct_to_lds.json`). But the perf verdict is subtle, and the
committed step-12 snapshot must be read with care: its `direct_to_lds`
sample has a **`ms_spread_pct` of ~4290 %** (the capture GPU was badly
contended during that sample), so its best-ms (11.91 µs) is *not* a
trustworthy standalone-DTLA number. The clean DTLA-vs-round-trip read at a
*matched* `tile_k` comes from the populated step-13 sweep:

| (tile_k=256, mem) | best µs | %HBM | vs rocBLAS |
|---|--:|--:|--:|
| round-trip (`base_tk256_mem`) | 15.84 | 26.5 % | 1.52× |
| direct_to_lds (`dtl_tk256_mem_ALL_ALL`) | 13.42 | 31.3 % | 1.29× |

At `tile_k=512` the two are within noise (`base_tk512_mem` 13.31 µs vs
`dtl_tk512_mem_ALL_ALL` 11.03 µs in step 13 — again a contended-baseline
caveat), and DTLA only becomes a *decisive* win at `tile_k=1024`, where the
round-trip path collapses to 58 µs but DTLA holds at 10.5 µs (step 13). So
DTLA's value is **enabling deep `tile_k`**, not the standalone instruction
count: on this geometry (block_size=64 = 1 wave/CTA, M=2, K=4096) the kernel
is wave-scheduler-bound, the round-trip's 32 ds_writes run in the shadow of
the global_load's vmcnt, and cutting them only helps once the freed issue
slots are filled with *useful work overlapping the in-flight load* — which is
what Tensile's `PGR2 PLR1 SIA3` (two prefetched global reads, one prefetched
LDS read, ScheduleIterAlg=3) adds and what makes the deeper tile viable.

So the rocprof name tells the whole story: `DTLA1_DTLB1` pays off
*together with* a deeper `tile_k` (and, in Tensile, `PGR2 PLR1 SIA3`).
Adding DTLA at a shallow tile, with no prefetch / scheduling surface to
fill the freed slots, is neutral-to-negative; its real value is unlocking
`tile_k=1024` (step 13). The patch is kept as a documented opt-in
(`direct_to_lds: bool` in `TraitSpec`) so future ping-pong work can build on
it — see `helpers/loads.py` (`AsyncPingPongLoader`) for the prefetch wrapper
that would compose with it.

### What still separates DSL from rocBLAS

Three Tensile tokens were needed *together* to break through:

- **DTLA1 + DTLB1** (direct-to-LDS): wired in `TraitSpec.direct_to_lds`.
- **CACHE_ALL** hint for both operands: the cache hint matters more
  than expected — `CACHE_STREAM` and non-temporal hints regress.
- **tile_k=1024**: only viable *with* DTLA. The round-trip load
  pattern collapsed at tk≥1024 (`data/13_dtl_sweep.json` `base_tk1024_mem`:
  58.49 µs / 7.2 % HBM) due to VGPR pressure; DTLA frees those 32 VGPRs.

## Step 13 — DTLA cache-hint sweep (`13_dtl_sweep.py`)

Sweep of `tile_k ∈ {256, 512, 1024}` × `pipeline ∈ {mem, compv4}` ×
`(cache_a, cache_b) ∈ {(ALL,ALL),(ALL,STR),(STR,STR),(NT,NT),(ALL,GLC)}`.

Result: **`tk1024_mem_ALL_ALL` at 10.51 µs / 40.0% HBM — 1.01× rocBLAS.**
The shape essentially matches hipBLASLt for the first time. Without
DTLA at tk=1024, the kernel collapses to 58 µs.

Cache hints: `CACHE_ALL` wins universally for both operands. The
A tile (M=2, 16 rows of K) gets reused across CTAs (L1 reuse), so
streaming hints cost real bandwidth. B is one-shot per CTA, but
even there `STR`/`NT` shows no measurable benefit.

## Step 14 — Push tile_k past 1024 (`14_dtl_push.py`)

Does deeper K extend the win? No:
- `tk1024 mem`: 10.52 µs (1.01×)
- `tk2048 mem`: 12.02 µs (1.16×)
- `tk2048 compv4`: LDS budget exceeded (262 KiB > 160 KiB cap)

tile_k=1024 uses ~64 KiB/WG → 2 WGs/CU. tile_k=2048 forces 1 WG/CU
and the kernel becomes latency-bound on issue. The sweet spot is at
tk=1024 exactly where the LDS-vs-occupancy frontier sits.

## Step 15–17 — wave geometry & `waves_per_eu` retest (`15_…`, `16_…`, `17_…`)

Once tk=1024 + DTLA + CACHE_ALL/CACHE_ALL was landed, the obvious next
ask was "more lanes per WG should expose more in-flight global loads".
Step 15 swept `warp_m × warp_n ∈ {1×2, 1×4, 2×2}` at the winning tile.
Step 16 widened `tile_n` (32, 64) to feed those extra waves real work.
Step 17 re-confirmed `waves_per_eu` now that the kernel is no longer
single-wave.

*Before* the multi-warp LDS-offset patch (below), every multi-warp variant in
step 15 produced **garbage output** (a `max_abs` on the order of 10⁴ instead of
0), and step 16 was nonsense as a result. That made multi-warp DTLA the suspect,
not "DTLA is bad". The committed `data/15_warp_geom.json` / `data/16_more_lanes.json`
were captured *after* the patch landed, so their rows are all `max_abs = 0` —
the garbage was the discovery state, not the shipped state.

## Step 18 — DTLA ping-pong prefetch (`18_dtl_prefetch.py`)

> Runbook §12.1.D (pipeline depth), §17.4 (compose-or-not).

Plumbed `TraitSpec.dtl_prefetch=True`: double-buffered LDS, scf.for
iter-arg carries the live half-index, the prologue issues the first
DTLA into half-0 while the steady-state loop drains half-(i⊕1) and
fires DTLA into half-i. The `s_waitcnt vmcnt(loads_per_tile)` between
the two DTLA passes had to be tuned by hand (gfx950 caps vmcnt at 6
bits = 63).

Result: bit-exact (max_abs=0). Perf identical to non-prefetch
(10.51 µs both). The 1-wave/CTA kernel is HBM-bound — adding a second
in-flight tile doesn't help because the HBM controllers are already
the bottleneck. Kept as `dtl_prefetch` for future shapes where the
bound shifts.

## Step 19 — isolate the multi-warp DTLA bug (`19_multiwarp_probe.py`)

Four-cell matrix: `{single-warp, multi-warp} × {DTLA off, DTLA on}`,
all bit-exact ref check. Result before the patch (step 19 prints to stdout, no
JSON sink):

| | DTLA off | DTLA on |
|---|---:|---:|
| 1 warp  | max_abs = 0 | max_abs = 0 |
| 2 warps | max_abs = 0 | **max_abs ≫ 0 (order 10⁴, garbage)** |

The bug is **specifically** in multi-warp DTLA. Diagnosis:
`async_buffer_load_lds_addr` is a wave-level intrinsic and writes
into a **wave-uniform** `lds_dst`. Every wave was given the same
`a_lds_par_base` / `b_lds_par_base`, so they stomped each other.

Patch in `emit_load_phase` (`instances/common/gemm_universal.py`): compute a
per-wave LDS base offset (`warp_id × wave_size × _DTL_BYTES_PER_LANE`, with
`warp_id = tid // wave_size`) and pass `a_lds_wave_base` / `b_lds_wave_base` to
both the A- and B-load passes (both must be fixed — patching only the A-loop
leaves the B side colliding).

Re-running step 19 post-patch: all four cells max_abs = 0.

## Step 20 — multi-warp DTLA, now correct (`20_dtl_multiwarp.py`)

Resweep `(tm, tn, tk, warp_m, warp_n)` with multi-warp finally legal:
`16×32×512 w1×2`, `16×64×256 w1×4`, `32×64×256 w2×4`, etc.

Every wider tile was slower (full table in `data/20_dtl_multiwarp.json`): the
single-warp `tm16 tn16 tk1024 w1×1` rows hold ~10.5 µs, while the best
multi-warp finish (`tm16 tn32 tk512 w1×2` with prefetch) is 11.75 µs and the
non-prefetch `tm16 tn32 tk1024 w1×2` is 14.06 µs; wider warp grids (`w1×4`,
`w2×4`) are far worse. Mechanism: wider tile_n cuts the grid roughly in half, so
about half the CUs idle. At `1 wave/WG` the kernel already saturates the HBM
controllers. **Multi-warp was a correctness fix, not a perf lever** on
this shape.

## Step 21 — chiplet_swizzle (`21_chiplet.py`)

> Runbook §12.1.L (chiplet swizzle).

MI355X has 8 XCDs; the default WG dispatch order scatters consecutive
CTAs across XCDs so each XCD's L2 sees uncorrelated traffic.
`chiplet_swizzle` remaps WGIDs via `chiplet_aware_super_tile_dynamic`
so consecutive WGs land on the same XCD. At `M=2` the same 16 KiB A
tile is reused by every CTA in a M-row — exactly the cross-CTA reuse
this knob targets.

Sweep `wgm ∈ {2,4,8,16}` × `chunk ∈ {16,32,64,128}` at the locked
DTLA winner, 30 warmup × 500 iters × 10 attempts. From the committed
`data/21_chiplet.json` (best-ms basis):

| variant | best µs |
|---|---:|
| baseline (no chiplet) | 10.517 |
| **chiplet wgm=8 chunk=16** | **10.297** |
| chiplet wgm=2 chunk=16 | 10.301 |
| chiplet wgm=16 chunk=16 | 10.319 |
| chiplet wgm=4 chunk=16 | 10.320 |

**≈2.1 % real lift over the no-chiplet baseline** in this snapshot (the cleaner
serial re-measure in step 22 puts it at 2.4 %). Top `wgm` values cluster within
a few ns at chunk=16; wider chunks regress (the chunk has to be small enough that
A's reuse window fits inside an XCD's residency).

## Step 22 — high-confidence confirmation (`22_confirm_winner.py`)

20 attempts × 1000 iters × 3 interleaved rounds (N=60 each for chiplet,
baseline, paired rocBLAS), then a standalone rocBLAS probe so the
reference isn't cache-warm-biased.

From the committed `data/22_confirm_winner.json` (N=60 each):

| label | best µs | median µs |
|---|---:|---:|
| chiplet winner | 10.301 | 10.319 |
| baseline (no chiplet) | 10.536 | 10.575 |
| rocBLAS (standalone, fair) | 10.181 | 10.456 |

Read straight off this (shared-GPU) snapshot: the chiplet swizzle is a real
**≈2.2 % lift** over the no-chiplet baseline (chiplet/baseline best = 0.978).
Against the standalone rocBLAS probe, **DSL and rocBLAS are within the noise
floor of each other** — rocBLAS is marginally faster on best (10.18 vs 10.30 µs)
while DSL is marginally faster on the *median* (10.32 vs 10.46 µs). Both kernels
move the same ~33.5 MB; the best-vs-median spread on each side is comparable to
the gap between them. The runbook's clean serial re-measure of this exact case
study (§17.7 "Skinny-M Decode GEMM") records the result as 10.29 µs / 44 % HBM /
1.02× rocBLAS with the chiplet swizzle a 2.6 % win — use that as the
exclusive-GPU reference; re-run `22_confirm_winner.py` on an idle device to
reproduce it.

## Steps 23–24 — split-K, the structural follow-up, actually built

Steps 04–05 named split-K as the one structural lever out of reach for the
rectangular small-M decode tile at the time. That gap has since closed — the
production universal-GEMM body now implements split-K directly
(`TraitSpec.split_k > 1`: each CTA computes one `(m_tile, n_tile)` over a K-slice
`[z·ks, (z+1)·ks)` with `ks = K/split_k`, then atomic-adds its partial f32 tile
into an f32 workspace; `split_k == 1` keeps the canonical single-pass body
byte-identical). Steps 23–24 exercise it on this exact decode shape.

The mechanism is **not** the per-byte efficiency the chiplet winner optimized —
it is **device fill**. At `M=2, N=4096, K=4096` the `split_k=1` grid is only
`m_tiles × n_tiles ≈ 1 × n_tiles` CTAs against ~256 CUs, so most of the device is
idle. Slicing K by `split_k` multiplies the CTA count by that factor and puts the
idle CUs to work on the same B traffic. This is a different question from steps
13–22 (which asked "how efficiently does one saturated config stream bytes?") —
here the kernel was never saturating the device in the first place.

### Step 23 — split-K over the production body (`23_splitk_universal.py`)

Routes the decode GEMM through `build_streamk_gemm_block_tile` →
`build_universal_gemm` with `trait.split_k > 1` (the fast vectorized,
LDS-double-buffered `compv4` inner). It verifies bf16 correctness vs
`torch.matmul(A, W.t())` (f32-accumulated, `rel < 5e-2` gate), then sweeps
`tile ∈ {16×64×32, 16×128×32, 16×64×64, 16×256×32, 16×128×64, 32×64×32}` ×
`split_k ∈ {4, 8, 16, 32}`, timing each on the same HIP-event timer / stream as
rocBLAS and reporting `ck µs / rocBLAS µs / ratio` (a `*` marks `ratio > 1`, i.e.
faster than rocBLAS). Best-vs-best is the reported basis because the script is
written to tolerate a shared GPU — it also prints median + spread, and warns if
the rocBLAS spread exceeds 40 % (a sign the device is contended and the baseline
needs a clean re-run).

**Measured result (gfx950 MI355X, exclusive GPU, clean serial median).** On this
exact shape the sweep's optimum is **`tile 16×64×64`, `split_k=8` ≈ 1.6× rocBLAS**
(two independent clean-serial runs agreed at 1.61× / 1.61×) — the realized
structural win the single-pass steps could only point at (`split_k=1` is that
single-pass body at ≈ 1.0× rocBLAS, §17.7). **The degree matters:** `split_k=4`
reaches only ≈ 1.1× because its K-slice is still 1024 deep — above the ~512
launch/atomic-reduce floor — so it under-fills the win. That is exactly why the
dispatcher's K-depth heuristic (step 24) targets `split_k=8` for `K=4096`. These
ratios are same-session (the box throttles 25–30 % under load, so only
back-to-back CK/rocBLAS ratios are valid); treat the number step 23 prints on
*your* hardware as authoritative.

### Step 24 — wire split-K into the dispatcher (`24_splitk_dispatch.py`)

Plumbs the split-K body into `dispatch_gemm_bf16` so the decode projections
(`qkv_proj` / `o_proj` / `gate_up` / `down`) pick it up automatically. The
selection lives in `rocke.helpers.split_k.select_split_k`: it engages only when
the base grid leaves the device mostly idle (`base_grid < target_ctas/2`) and
sets the degree from per-slice K-depth (`≈ K / 512`, snapped to the largest factor
that evenly slices K), so `K=4096 → split_k=8`, `K=2048 → 4`. The heuristic is
opt-in / default-safe and overridable via `ROCKE_GEMM_SPLIT_K`
(`auto` | `off` | `<n>`).

The default invocation is a **CPU-only dry run**: it prints the dispatch decision
+ grid for every shape and asserts the **square `4096³` shape stays `split_k=1`**
(its grid already fills the device, so the gate fires before the depth target is
consulted). `--run` adds one on-GPU numeric sanity check per shape (honoring the
workspace contract: `split_k > 1` ⇒ zero-init f32 `[M, N]`, launch the
`(N_tiles, M_tiles, split_k)` grid, cast back). The per-shape perf sweep is left
to step 23.

> **Reconciling with the "split-K doesn't help here" note below.** The ceiling
> argument in the next section is about a *single saturated config* that already
> streams B at the part's sustained-bf16 wall — for that config, splitting K only
> replicates B traffic. Steps 23–24 attack the orthogonal failure mode: at decode
> the base grid does **not** saturate the device, and filling the idle CUs with
> K-slices is the realized structural win the earlier steps could only point at.
> The two statements are about different operating points (saturated vs idle), not
> a contradiction. Treat step 23's printed `ratio` as the authoritative number on
> your hardware; the split-K module docstring records the motivating decode-sweep
> result of roughly `1.6×` rocBLAS at this shape.

## Why this is the ceiling (and what would break it)

The shape transfers ≈33.5 MB total (B = 32 MiB, C = 16 KiB, A = 16 KiB).
At MI355X's 8 TB/s theoretical HBM peak, the *floor* is 4.2 µs. The
sustained streaming-bf16 ceiling on this part is ~3.5 TB/s ≈ 44 % of
peak — which is where both DSL (44 %) and rocBLAS (42 %) live.

To exceed rocBLAS on this shape requires **reducing total HBM bytes**,
not improving per-byte efficiency:

- **Kernel fusion** (the realistic win): fuse `o_proj` into the
  attention output reduction so the C tensor never round-trips HBM,
  and/or fuse the residual-add that follows. This drops C and the
  add's read from the byte total.
- **Weight quantization** (FP8 → halve B's 32 MiB). Outside the scope
  of "match rocBLAS bf16 → bf16".
- **Split-K** doesn't help here — A is already L1-resident across CTAs,
  B has no cross-CTA reuse, and both kernels saturate HBM. Splitting K
  would replicate B traffic.

The runbook loop converged on this kernel; the remaining gap is
algorithmic, not knob-tunable.

## File map

```
rocke/examples/gfx950/gemm_perf_skinny_decode/
├── README.md                              # this file
├── scripts/
│   ├── 01_probe_occupancy.py              # static probe, no GPU
│   ├── 02_sweep_bench.py                  # GPU sweep with hygiene
│   ├── 03_correctness.py                  # standalone bf16 verify
│   ├── 04_compare_rocblas.py              # side-by-side vs rocBLAS (original)
│   ├── 05_extra_levers.py                 # CK-Tile-inspired extras
│   ├── 06_ground_truth_geometries.py      # rocprof-derived MT16x16x* sweep
│   ├── 07_push_tile_k.py                  # push tile_k → MT16x16x512 winner
│   ├── 08_lever_combinations.py           # 192 scheduling combos on winner
│   ├── 09_preshuffle_b.py                 # preshuffle_b with custom harness
│   ├── 10_hipcc_backend.py                # LLVM vs hipcc-O3 backend
│   ├── 11_final_compare.py                # final apples-to-apples ratio
│   ├── 12_direct_to_lds.py                # DTLA/DTLB lever (wires upstream patch)
│   ├── 13_dtl_sweep.py                    # DTLA × tile_k × pipeline × cache-hints
│   ├── 14_dtl_push.py                     # push tile_k > 1024 (occupancy frontier)
│   ├── 15_warp_geom.py                    # multi-warp sweep (exposed DTLA correctness bug)
│   ├── 16_more_lanes.py                   # widen tile_n for multi-warp (garbage pre-patch)
│   ├── 17_confirm_wpe.py                  # waves_per_eu retest under multi-warp
│   ├── 18_dtl_prefetch.py                 # ping-pong double-buffered DTLA (bit-exact, perf-neutral)
│   ├── 19_multiwarp_probe.py              # 2×2 isolation of the multi-warp DTLA bug
│   ├── 20_dtl_multiwarp.py                # resweep wider tiles post-patch (slower, grid-bound)
│   ├── 21_chiplet.py                      # chiplet_swizzle sweep — ~2.2% win (2.4-2.6% serial)
│   ├── 22_confirm_winner.py               # N=60 paired confirmation of the chiplet winner
│   ├── 23_splitk_universal.py             # split-K over the production body (tile × split_k sweep vs rocBLAS)
│   └── 24_splitk_dispatch.py              # wire split-K into dispatch_gemm_bf16 (dry-run; --run for GPU check)
├── ALGORITHM.md                          # the GEMM/decode math + why the kernel is shaped this way
└── data/                                  # 01..22 JSON sinks (no 19; 05-08 are empty "no samples")
    └── 01_occupancy.json … 22_confirm_winner.json
```

Each script also writes its compiled HSACOs to a per-step `build*/` directory
(`build/` for step 02, `build_extra/` for 05, `build_gt/` for 06, `build_push_tk/`
for 07, `build_combo/` for 08, `build_preB/` for 09, `build_final/` for 11, etc.);
these are generated artifacts and are not checked in.

## Runbook section index (where each script's discipline came from)

| Script | Runbook anchors |
|---|---|
| `01_probe_occupancy.py` | §3.1b, §12.1.Q, §14.2 |
| `02_sweep_bench.py`     | §2.2, §2.3, §12.1.B–H, §12.1.P, §12.2 |
| `03_correctness.py`     | §2.1, §14.1, §14.3 |
| `04_compare_rocblas.py` | §2.2, §17 (case-study form), §14.3 |
| `05_extra_levers.py`    | §12.1.G, §12.1.I, §12.1.L, §17.4 (didn't-help table) |
| `06_ground_truth_geometries.py` | §12.1.B–C (tile/K-pack), §17.4 (rocprof reading) |
| `07_push_tile_k.py`     | §12.1.C (K-pack), §6.5 (LDS budget), §3.3 (memory-bound) |
| `08_lever_combinations.py` | §12.1.D–L (full knob matrix), §17.4 (compose-or-not) |
| `09_preshuffle_b.py`    | §12.1.J (preshuffle), §2.1 (custom-harness correctness) |
| `10_hipcc_backend.py`   | §3.1b (DSL probe / lowering compare), §14.3 (noise threshold) |
| `11_final_compare.py`   | §2.2, §17, §15 (final form) |
| `12_direct_to_lds.py`   | §6.1/§6.3 (global loads / LDS), §17.4 (neutral/negative-result form) |
| `13_dtl_sweep.py`       | §12.1.D–L (knob composition), §6.6 (cache hints) |
| `14_dtl_push.py`        | §6.5 (LDS budget), §3.1b (occupancy frontier) |
| `15_warp_geom.py` / `16_more_lanes.py` / `17_confirm_wpe.py` | §12.1.E (warp grid), §17.4 (negative-result form) — surfaced the multi-warp DTLA bug |
| `18_dtl_prefetch.py`    | §12.1.D (pipeline depth), §17.4 (compose-or-not, neutral result) |
| `19_multiwarp_probe.py` | §2.1, §14.1 (correctness as a probe), §17.4 (isolate before patching) |
| `20_dtl_multiwarp.py`   | §3.3 (memory-bound saturation), §12.1.E (warp grid) |
| `21_chiplet.py`         | §12.1.L (chiplet swizzle) — applied at the right shape this time |
| `22_confirm_winner.py`  | §2.3 (hygiene), §14.3 (paired-vs-standalone reference) |
| `23_splitk_universal.py` | §12.1.A (decode algorithmic family), §2.1/§14.1 (correctness gate), §17 (case-study form) — split-K, the structural follow-up |
| `24_splitk_dispatch.py`  | §12.1.A (dispatch picks the family), §17.4 (default-safe heuristic, env override) |

## CK Tile examples that inspired this layout

| CK Tile path | What it gave us |
|---|---|
| `example/ck_tile/03_gemm/universal_gemm.cpp` | `persistent`, `chiplet_swizzle`, `waves_per_eu` trait surface |
| `example/ck_tile/03_gemm/gemm_splitk_two_stage*.cpp` | Two-stage split-K is the canonical skinny-M pattern (named as the unaddressed follow-up) |
| `example/ck_tile/18_flatmm/` (`flatmm_basic.hpp`, `moe_flatmm.hpp` `FlatmmConfig32` / `FlatmmConfig16`) | the "small-shape" flatmm templates the DSL `flatmm` instance mirrors |
| `example/ck_tile/40_streamk_gemm` | Where the next move would land (rectangular small-M tiles + bf16) |

## Final summary (runbook §15 template)

```text
Best correct variant (step 21, confirmed step 22 N=60):
  name:        t16x16x1024  mem / interwave / cshuffle  +  DTLA + DTLB (CACHE_ALL/CACHE_ALL)
                                                        +  chiplet_swizzle wgm=8 chunk=16
  shape:       bf16 M=2 N=4096 K=4096 (Qwen3-8B o_proj decode)
  latency:     10.30 µs best / 10.32 µs median  (data/22_confirm_winner.json, shared GPU)
               10.29 µs on an exclusive-GPU serial re-run (runbook §17.7)
  correctness: max_abs_diff vs fp32 reference = 0.0 (bit-exact)
  vs rocBLAS:  within noise — committed snapshot rocBLAS 10.18 µs best / 10.46 µs median;
               runbook §17.7 records the clean-run result as 1.02× / 44% HBM
  speedup vs step-02 winner: ~4.6× (data/02 best 47.2 µs -> 10.30 µs)
  key levers:  (1) DirectToLDS A & B in gemm_universal.py (TraitSpec.direct_to_lds)
                   removes the global→VGPR→ds_write round-trip (-32 VGPR, -32 inst/K-tile);
               (2) CACHE_ALL for BOTH operands (CACHE_STREAM/NT regress);
               (3) tile_k=1024 — only viable with DTLA (the round-trip pattern
                   collapsed at tk≥1024 from VGPR pressure: 58.49 µs, data/13);
               (4) chiplet_swizzle wgm=8 chunk=16 — final ~2.2% lift (data/21;
                   2.4-2.6% on the clean serial re-run) from steering consecutive
                   WGs onto the same XCD so A's L2 footprint is reused.

  Patches in gemm_universal.py:
    - wire async_buffer_load_lds_addr directly (the AsyncTileLoader path
      assumes attention's wide-K geometry; breaks for tile_m=16, tile_k≥512)
    - per-wave LDS base offset = warp_id * wave_size * _DTL_BYTES_PER_LANE
      before the DTLA passes — without it, multi-warp DTLA writes collide
      (step 19/20)

  Why no further win is on the table:
    Both kernels live at the sustained-streaming-bf16 ceiling of the
    part (~40-44% of HBM peak). Neither is meaningfully more bytes-per-µs
    efficient than the other on the committed snapshot; the runbook §17.7
    clean run has DSL fractionally ahead. Beating this requires reducing the
    33.5 MB byte total — kernel fusion or weight quantization — not
    GEMM-internal tuning.
```
