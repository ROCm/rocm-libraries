# W17 K1: the K = 288 FP8 blockscale GEMM that blocks ATOM at TP8

W17, 2026-11-05. MI355X (gfx950), atom-dev image `nightly_202610171506` (aiter `2cc8c750`, ATOM image `df86c482`),
ATOM tree `dev/ATOM-dsv41`. Ratios only. Absolute times: `hackathon-prep/private-perf/dsv41-mono/w17/K1-k288-abs.md`.
Harness: `hackathon-prep/dsv41/w17/` (§10). Raw results: `hackathon-prep/dsv41/w17/results/<tag>/`.

## 0. Verdict

| question | answer | evidence |
|---|---|---|
| Which GEMM fails at TP8? | Exactly one: `ffn.shared_experts.w2` (row-parallel, per rank N = 5120, **K = 2304 / 8 = 288**), target and draft, at **every M** (27 of 27 tested, 1-16384) | probe k1p1; simulated-TP8 server k1t1 |
| Why? | ATOM stores group32 FP8 weights (16,16)-preshuffled on gfx950 and calls aiter's preshuffled group32 GEMM, whose FlyDSL kernel needs `K % 64 == 0`; 288 = 4.5 x 64. No tuned row and no heuristic config exists, so it raises `ValueError` | §1 |
| Any other TP8 shape? | **No other refusal.** All 8 other per-rank FP8 linear shapes run at all 27 M. A simulated TP8 server (ATOM's `-tp 8 --fake-eplb` on one GPU = rank 0 with exact TP8 per-rank shapes) builds the model, warms up, captures all 12 FULL graphs and serves with the knob on, and dies at the w2 GEMM with it off. Not covered: the V4.1 mono kernels and real collectives (they cannot run simulated, §2.3; need 8 real GPUs, none free) | §2 |
| rocKE kernel | `dsv41_mono/k288_gemm.py`: stage_wqkv's FP8 GEMV core (via W13's `wide_gemm`) generalised to a 32-K tail, any M, whole-K staging; fused MXFP8 quantization (ATOM's rule bit for bit) or ATOM's codes | §3 |
| Numerics | **PASS.** Bit for bit equal to ATOM's own FP8 GEMMs (aiter's row-major group32 GEMM and ATOM's local kernel) at every one of 46,352 cells; float64 elementwise bound held everywhere; 276 mutant builds of 6 kinds and a checker mutant all caught | §4 |
| Speed (search picks, confirmed, cold weights) | **0.52-0.79 of the best existing ATOM route** that can run K = 288 at every M from 1 to 16384 (0.55-0.66 at decode widths). Within 1-5 % of the pick's own no-math bound prototype at M <= 192, 5-14 % above it at prefill widths | §5, §6 |
| COMGR vs Rust direct | COMGR wins every one of 22 cells: direct is 1.05-1.43x slower at the same knobs. Defects found (no `abs` source-modifier folding, no `max3` combining, extra copies, 1.6x VGPRs at wide tiles) | §7 |
| ATOM wiring | `ATOM_DSV41_RK_K288=1` (exactly the refused calls, decided by aiter's own selection, cross-checked against aiter at install) or `=all` (every K = 288 call); default off, knob off = module never imported | §8 |
| Accuracy (GSM8K / passkey) | **Not run**: it needs a served TP8 model on 8 real GPUs; owned by W17's TP8 bring-up. The routed output is bitwise ATOM's own row-major result in every gated cell, so the knob adds no numerics of its own | §8.3 |

## 1. What fails and why

**The call chain.** `Expert.forward` (`atom/models/deepseek_v4.py`) -> `self.w2(out)` -> `LinearBase.forward`:
`native_a8_group_rows == 32` (V4.1 dense layers are `QuantType.per_1x32`, FP8 32x32 weights, `layers.py`) ->
`blockscale.native_quant_linear(x, w, ws, weight_preshuffled=w.is_shuffled)`. On gfx950 the weight is shuffled at load
(`ATOM_GROUP32_WEIGHT_PRESHUFFLE`, default on). ATOM's `quantize_fp8` (group32 E8M0) quantizes the activations, then:

1. aiter `gemm_a8w8_blockscale(isBpreshuffled=True)`;
2. `gemm_a8w8_blockscale_bpreshuffle` (E8M0 route);
3. `_gemm_mxscale_bpreshuffle`: no tuned row for `K = 288`;
4. FlyDSL `run_bmm_a8w8_mxfp8_gfx950`;
5. `pick_bmm_kernel_name` raises `ValueError: [FlyDSL gfx950 bmm] no kernel config runs B=1 N=5120 K=288 with 1x32 / 32x32 scale blocks`.

**The constraint.** The heuristic tries every looser tile (`tile_k` 512 / 256 / 128, `tile_n` 256-32, fewer stages, B
through LDS, per-stage scales, splits). `check_bmm_config` rejects all of them because it requires
`K % (MFMA_K // 2) == 0`, with `MFMA_K = 128`. The kernel handles a 64-K half step but not a 32-K one. The constraint
was read in a local aiter checkout (`ee72a95`); the container's identical error message and the install-time
cross-check (§8) confirm the container behaves the same.

**Probe (job k1p1, 68289009).** Every per-rank native FP8 linear at TP8, derived from `/model/config.json` with
`weights.py`'s split rules, was run through ATOM's own entry at 27 M (1-16384), each shape in a child process:

| projection | per rank N x K | split | preshuffled route |
|---|---|---|---|
| attn `wqkv_a` (fused) | 1792 x 5120 | replicated | 27 / 27 run |
| attn `wq_b` | 4096 x 1280 | column | 27 / 27 |
| attn `wo_a` (one group, grouped) | 1024 x 4096 | column groups | 27 / 27 |
| attn `wo_b` | 5120 x 1024 | row | 27 / 27 |
| indexer `wq_b` | 4096 x 1280 | replicated | 27 / 27 |
| shared `gate_up_proj` | 576 x 5120 | column | 27 / 27 |
| **shared `w2`** | **5120 x 288** | **row** | **0 / 27 (ValueError)** |
| engram `wkv` | 25600 x 6144 | replicated | 27 / 27 |
| draft `main_proj` | 5120 x 15360 | replicated | 27 / 27 |

ATOM's two other FP8 routes run K = 288 at all 27 M:

- `native_quant_linear` on a ROW-MAJOR weight (aiter's group32 Triton GEMM);
- ATOM's local kernel `blockscale_kernels/fp8.py gemm_fp8_local`.

So ATOM could start TP8 today by not preshuffling that one weight. They are the comparators in §6. Every route's float64
relative L2 error is 1.64-1.70e-3 (bf16 output rounding).

## 2. Is anything else broken at TP8? The dry run

### 2.1 Why simulated TP8

No MI355X node had 8 free GPUs for the whole lane:

- n07-17: W15 and W16 jobs, 4-6 GPUs;
- n14-05 and m13-25: 4-6 GPUs held by perpetual Jenkins jobs;
- m10-09: no GPU access (alola rule 265);
- n01-09: drained.

ATOM has a built-in simulated TP (`atom/distributed/simulated_tp.py`). `-tp 8 --fake-eplb` on G < 8 GPUs runs ranks
0..G-1 of the TP8 deployment:

- every shard size comes from the TP group's reported width (8), so per-rank shapes, weights and memory are exactly TP8's;
- collectives cover only the present ranks, so outputs are meaningless;
- `--fake-eplb` also spreads the routing evenly over all experts.

All TP8 ranks have the same per-rank shapes: every split is even, including the vocab, 129280 / 8 = 16160 rows. Rank 0
alone therefore covers every per-rank kernel shape. The B0 server flags were used: `--level 0`, FULL graphs, DSpark 5,
capture sizes 1-8 / 16 / 32 / 48 / 64, `--max-num-batched-tokens 16384`, prefill warm-up 2 x 8192.

### 2.2 Results (job k1t1, 68289566; control k1t2, 68289687)

| arm | knob | mono | result |
|---|---|---|---|
| simoff | off | off | **dies in warm-up** at `ffn.shared_experts.w2`: `ValueError ... K=288` (§1) |
| simk288 | `ATOM_DSV41_RK_K288=1` | off | **starts**. Every step completed: weights loaded; 43 w2 weights registered (40 target + 3 draft layers); refusal predicate and aiter agree at M = 1 and 4096; install check bitwise equal to ATOM's local GEMM at 10 M; prefill warm-up; all 12 FULL decode graphs captured (6-384 rows); 8 chat replies with 0 errors |
| simall | `=all` | off | starts; same steps |
| simmono | `=1` | **on** | the mono draft kernels (width 10) build, then the first launch faults (`hipErrorIllegalAddress`) |
| sim2mono (control) | off | on, **simulated TP2** | the same fault at the same place, in a configuration that runs at real TP2 |
| sim2nomono (control) | off | off, simulated TP2 | starts |

### 2.3 Conclusions

- **Only w2 refuses** across construction, prefill and decode capture of the native path at TP8 per-rank shapes.
- **Running but untuned at TP8.** These shapes run on aiter heuristics, without tuned rows (a performance risk for the
  TP8 baseline, not a failure):
  - `wo_b` (5120 x 1024);
  - shared `gate_up` (576 x 5120);
  - the routed `fused_moe` at the TP8 padded intermediate 384, target (384 experts, top 6) and draft (128, top 3), from
    4 to 16384 tokens.
- **The V4.1 mono cannot run under simulated TP.** Its in-kernel peer mailboxes address the absent ranks: the TP2
  control faults identically. Its TP8 builds compile (the width-10 draft set built). Launching them with real
  collectives needs a real 8-GPU node. The W17 TP8 bring-up runs that.
- **The real TP8 command** is the `tp8` phase without the simulation flag, for example:

  ```
  submit.sh k1r8 <node with 8 free GPUs> 8 00:40:00 W17_PHASES=tp8 W17_PICKFROM=k1s3 W17_PICKDIR=/job/picks/search/hsaco_idx W17_TP8_ARMS=k288:refused:1:3000
  ```

## 3. The rocKE kernel (`dsv41_mono/k288_gemm.py`, `k288_space.py`)

**Starting point.** It starts from ATOM's mono K1 FP8 GEMV core, which `stage_wqkv.py` reproduces bit for bit:

- the `gemv_fp8_loads` lane map, which reads ATOM's (16,16)-preshuffled weight directly into the MFMA A operand;
- E8M0 scales packed so `op_sel` picks the byte;
- `v_mfma_scale_f32_16x16x128_f8f6f4`.

It reaches that core through W13's `wide_gemm` (the dense form of the same core), generalised three ways:

- **The 32-K tail.** K = 288 is two 128-K chunks plus one 32-K block. In the 16x16x128 fp8 lane map, half `h` of lane
  `l` holds 32-K block `4c + 2h + l/32`.
  - A weight half whose blocks all lie past K is the constant zero (no load).
  - A half with one valid block loads lanes 0-31 only. Lanes 32-63 read an out-of-range offset (0). They would
    otherwise read the next 16-row tile's bytes.
  - The activation staging writes zero codes with scale 127 for the pad blocks, so every product past K is an exact
    0 x 0.
- **Any M.** The grid is `ceil(N / R) x ceil(M / (16 tiles))`, so one build serves decode and prefill. Rows past M are
  never read or written.
- **Whole-K staging.**
  - The M block's activations are staged once, in `xb`-iteration batches.
  - `xin=bf16` quantizes in the staging with ATOM's `quantize_fp8` rule: E8M0 ceil scale of
    `max(amax, 1e-4) x fl(1/448)`, E4M3 RNE. Gate G5 proves this bit for bit.
  - `xin=fp8` reads ATOM's codes and scales. The scale is loaded by byte: a dword load can straddle the end of the
    `[M, K/32]` tensor when K/32 is not a multiple of 4, and an out-of-range dword reads 0. The first gate run caught
    exactly this bug.

**Knobs.** All are in the search space; level 0 is the default:

- `tiles` 1-24;
- `waves` 1 / 2 / 4 / 8 and `mt` 1 / 2 / 4 (R = 16 x waves x mt weight rows a CTA);
- `depth` (weight chunks in flight);
- `wfirst` (weights issued before the staging);
- `xb` (staging batch);
- `bra` (B LDS reads per chunk or per MFMA);
- `sb` (sched_barrier fences);
- `ep`: `direct` (8-byte stores) or `lds` (token-tile pairs through LDS, 16-byte stores);
- `order` (`nfast` / `mfast` CTA order);
- `wcp` (weights `nt`);
- `xin` (`bf16` fused, or ATOM's `quantize_fp8` launch plus `fp8`);
- `backend` (COMGR / Rust direct).

**Mutants**, all must fail:

- `tail`: the 32-K tail dropped;
- `tailx`: the tail half's lanes 32-63 unmasked;
- `scale`: chunk 1 picks the next scale byte;
- `row`: token rows 0 / 1 of every tile swapped;
- `mblk`: M block b reads block b-1;
- `xq`: one quantization scale one binade high.

**Bound prototypes** (`proto`, never served):

- `nomath`: the MFMAs replaced by one add (the structure's memory and latency floor);
- `noquant`: no quantization arithmetic;
- `storeonly`: the grid and epilogue only (the output-write floor);
- `empty`: the dispatch floor.

## 4. Numerics gate (`h/k1_gate.py`; jobs k1s3 = 68289435 on the gate plan, k1gp = 68289574 on the served picks)

Cells: every build x N in {5120 (served), 544 (partial row groups)} x M in {1, 2, 3, 5-7, 12, 15-18, 30-33, 48, 96,
192, the bucket edges b-1 / b / b+1, 2b+3, 4b+7, 300, 1000, 2049, 4111}.

Data classes:

- `randn`;
- `wide`: per-block magnitudes 2^-24..2^12, zero blocks, -0, bf16 subnormals;
- `ext`: every finite E4M3 weight code, E8M0 scales 100-150;
- `nanw`: NaN codes in row tile 1's first block, for every K-tail build.

Also covered: 13 K values (the 288 builds, plus every K % 128 class: 32, 96, 128, 160, 224, 256, 320, 352, 384, 416,
480, 512), both backends, both inputs, both epilogues and both CTA orders.

| gate | rule | gate plan (202 builds, 32,464 cells) | served picks (72 builds, 13,888 cells) |
|---|---|---|---|
| G0 ATOM semantics | bitwise equal to aiter's row-major group32 GEMM AND ATOM's local FP8 GEMM, same bf16 input | 0 differing cells | 0 |
| G1 float64 | every element within `0.5 ulp_bf16(max(|ref|,|y|)) + 2^-11 sum|x w|` | worst 0.94 of the bound | worst 0.91 |
| G2 | relative L2 <= 1.25 x ATOM local's + 1e-4 | worst 0.77 | pass |
| G3 canary | 3 rows after M untouched | 0 | 0 |
| G4 determinism | two launches bitwise | 0 | 0 |
| G5 quantization parity | fused-quant build == its `xin=fp8` twin on ATOM's codes | 6,784 cells, 0 differ | 0 |
| G6 backends | Rust direct == COMGR | 8,516 cells, 0 differ | 0 |
| mutants | must fail G0-G3 somewhere | **60 / 60 caught** (6 kinds x 5 bases x 2 backends) | **216 / 216** (`tail`, `scale`, `row` of every pick) |
| checker mutant | the reference with one scale block one binade high must fail G1 | caught (501 elements) | caught |

**Why G1 uses 2^-11 rather than the IEEE f32 model.** Under `(K+2) 2^-24 sum|x w|` both rocKE and ATOM's own GEMMs
exceed the bound, by up to 10.1x, in cancelling `ext` cells and in a few `randn` cells. The outputs agree bit for bit,
so this is the scaled MFMA's own dot-product arithmetic: about 12 bits relative to the largest product, not IEEE
accumulation. The 2^-11 term is calibrated on ATOM's GEMMs: their worst ratio under it is the same 0.94. The f32-model
ratio is still reported (`strict_f32_ratio_max`).

**The `tailx` mutant** is equivalent on finite weights: the leaked bytes meet zero activations. It is caught only by the
`nanw` class, where 0 x NaN turns tile 0's rows NaN. That class was added after the first run left it uncaught.

**Bugs the gate found and that were fixed:**

- the `xin=fp8` scale dword straddle;
- a too-tight G1 binade rule (`ulp(ref)`, now `ulp(max(|ref|, |y|))`).

## 5. Search (`h/k1_search.py`, engine `rocke.dispatch.tuning.search`, PLAN decision 15; job k1s3)

**Setup.**

- *Space.* `make_strategy("portfolio")` (hill + gp + de; the hill member is the lookback climber with combination
  re-sweeps) over `k288_space` (13 params, about 85 % of random configs legal).
- *Evaluator.* The mandatory check (G0 + G1 + G3 at M = cell, 17 and 1). The session's first check plants the `tail`
  mutant and stops the run if it is not caught; it was caught in all 22 sessions.
- *Measurement.* Spill rejection. Every round of 4 candidates is timed together with the baseline arm (aiter
  row-major), in CUDA graphs of 192 calls over 192 rotating weight copies (283 MB, more than the MALL, so weights come
  from HBM as in a decode step), after a cache-wiping write. Samples are per-round ratios to the baseline.
- *Budget.* 160 evaluations, 8 rounds, warm starts from the default, its direct twin and the previous cell's best.
- *Cells.* M = 1, 2, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 1024, 2048, 4096, 8192, 16384
  (3,520 evaluations, 14-115 s a cell). Every non-ok evaluation (0-7 a cell) was a spill rejection; no candidate
  failed the numerics check.
- *Records.* Traces in `results/k1s3/search/trace_<M>.jsonl`. Picks per M range: geometric midpoints between cells, and
  the 16384 pick serves every larger M. They are in `results/k1s3/search/hsaco_idx/index.json`, the directory
  `ATOM_DSV41_RK_K288_DIR` takes.

**What the search picked.**

- *1-9 rows.* t1 waves2 `wcp=nt` `order=mfast` (160 CTAs).
- *10-55 rows.* t1 waves4 `wcp=nt`, `ep=lds` from 14 rows.
- *56-221 rows.* waves8, t1-t2.
- *222-2896 rows.* t2-t6 waves8 (mt2 from 314 rows), `depth=1`, `nowfirst`, `ep=lds`.
- *Larger.* t4-t8 waves8 mt2 `ep=lds`.
- *Every cell.* COMGR. In the search's own measure, direct's best was 1.04-1.29x COMGR's best (§7).

## 6. Confirmation (`h/k1_bench.py`, 16 rounds, same job, interleaved, copy-major loads)

### 6.1 Cold weights

Per-call time relative to aiter's row-major group32 GEMM (lower is better):

| M | pick | A-vs-A twin | pick on Rust direct | default build | floor `nomath` | floor `storeonly` | floor `empty` | ATOM local | pad K->320 | pad K->384 | **pick / best ATOM route** |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.594 | 0.593 | 0.722 | 0.663 | 0.576 | 0.398 | 0.366 | 1.015 | 1.798 | 1.802 | **0.594** |
| 2 | 0.554 | 0.553 | 0.670 | 0.619 | 0.537 | 0.371 | 0.337 | 0.936 | 1.708 | 1.680 | **0.592** |
| 5 | 0.564 | 0.566 | 0.674 | 0.614 | 0.557 | 0.371 | 0.333 | 0.945 | 1.836 | 1.755 | **0.597** |
| 6 | 0.565 | 0.567 | 0.669 | 0.620 | 0.550 | 0.365 | 0.327 | 0.937 | 1.805 | 1.793 | **0.604** |
| 12 | 0.585 | 0.582 | 0.638 | 0.619 | 0.568 | 0.367 | 0.320 | 0.936 | 1.793 | 1.799 | **0.625** |
| 18 | 0.587 | 0.590 | 0.627 | 0.596 | 0.557 | 0.353 | 0.308 | 0.895 | 1.894 | 1.909 | **0.656** |
| 24 | 0.553 | 0.548 | 0.589 | 0.589 | 0.531 | 0.349 | 0.303 | 0.882 | 1.913 | 1.920 | **0.628** |
| 30 | 0.554 | 0.555 | 0.588 | 0.586 | 0.534 | 0.348 | 0.300 | 0.882 | 1.900 | 1.949 | **0.628** |
| 48 | 0.549 | 0.548 | 0.585 | 0.580 | 0.528 | 0.344 | 0.297 | 1.043 | 1.884 | 1.905 | **0.549** |
| 96 | 0.474 | 0.474 | 0.516 | 0.543 | 0.452 | 0.299 | 0.250 | 0.905 | 1.741 | 1.759 | **0.523** |
| 192 | 0.412 | 0.412 | 0.432 | 0.517 | 0.397 | 0.256 | 0.193 | 0.753 | 1.390 | 1.403 | **0.547** |
| 384 | 0.393 | 0.392 | 0.430 | 0.548 | 0.352 | 0.235 | 0.146 | 0.654 | 1.242 | 1.236 | **0.601** |
| 768 | 0.396 | 0.401 | 0.528 | 0.596 | 0.375 | 0.240 | 0.103 | 0.683 | 0.931 | 0.932 | **0.580** |
| 1536 | 0.383 | 0.383 | 0.548 | 0.620 | 0.366 | 0.235 | 0.063 | 0.741 | 0.684 | 0.674 | **0.569** |
| 4096 | 0.315 | 0.320 | 0.360 | 0.671 | 0.276 | 0.173 | 0.029 | 0.724 | 0.565 | 0.572 | **0.558** |
| 8192 | 0.288 | 0.288 | 0.370 | 0.683 | 0.257 | 0.148 | 0.015 | 0.707 | 0.428 | 0.423 | **0.682** |
| 16384 | 0.276 | 0.277 | 0.359 | 0.677 | 0.242 | 0.134 | 0.011 | 0.702 | 0.352 | 0.350 | **0.788** |

**The baselines.**

- "Best ATOM route" is the fastest of the four existing ways ATOM could run this GEMM without rocKE:
  - aiter row-major;
  - ATOM's local kernel;
  - the weight zero-padded to K = 320 or 384 at load, plus an activation pad copy and ATOM's preshuffled path.
- The padded routes lose at decode widths: an extra copy, and aiter's untuned heuristics at those shapes. They win
  among the ATOM routes from 1536 rows up.
- A-vs-A twins (the same code object loaded twice) agree within 0.006, so every ratio above is far outside the noise.

**Decode widths: at the latency floor of a separate launch.**

- The pick is 1-5 % above its own `nomath` prototype, which keeps every load, the staging and the store but does no
  math.
- The `empty`-kernel dispatch floor is about 0.30-0.37 of aiter's time.
- The gap between `empty` and `nomath` is one dependent HBM round trip (weights + activations), then the store.
- Only fusing w2 into the stage that produces its input (the shared expert's SwiGLU) can remove it. That fusion is a
  follow-up. At C1-C2 the V4.1 mono already serves the shared expert in its K2 kernel; this GEMM runs on the native
  path, for decode at 3+ requests, the draft's wide steps and prefill.

**Prefill.**

- The pick is 1.05-1.14x its `nomath` prototype.
- It sits about 2x above the pure output-write floor (`storeonly`). The structure re-stages and re-quantizes the
  activations once per 256-row group of N.
- `xin=fp8` (ATOM's quantization launch + the GEMM) was searched as the alternative and lost at every cell, except one
  16384-row near-tie in the first search run.

### 6.2 Warm weights

With 8 rotating copies the weights stay in the caches. The picks are 0.55-0.78 of the best ATOM route there, but at up
to 48 rows the default build beats the cold-tuned picks by 15-20 % (`wcp=nt` and waves2 cost there).

**Cold is the served regime.** A layer's w2 is read once per decode step, and about 40 GB of the rank's other weights
stream through between two reads. The warm table is in `bench.json` and the private store.

## 7. COMGR vs Rust direct

**Results.**

- Every one of 101 gate-plan pairs is bitwise equal in output, and the code objects all differ.
- Direct compiles 5-7x faster: a median of about 0.006-0.011 s per kernel against 0.04-0.06 s.
- Direct's runtime is 1.05-1.43x COMGR's at the same knobs (bench "pick on Rust direct" column), and its best per cell
  in the search lost to COMGR's best in all 22 cells (1.04-1.29x).

**Defects.** `llvm-objdump` of the M = 6 and M = 4096 picks (`results/k1s3/search/hsaco_idx`) shows:

1. **No `|x|` source-modifier folding.** `fabs` becomes an explicit `v_and_b32`: 101 against 38.
2. **No `v_max3_f32` combining** in the 32-value amax reduction of the fused quantization: 96 `v_max` against 32.
3. **Extra copies.** `v_mov_b32` 41 against 25 at M = 6, and 92 against 29 at M = 4096.
4. **VGPRs.** 124 against 79 at the t4 / waves8 / mt2 prefill pick: fewer waves per SIMD.

Total VALU count: +62 % at M = 6 and +64 % at M = 4096. These are compiler defects for the Rust track (W16 C1),
recorded with the exact kernels. COMGR is auto-selected here (`backend` is a search param).

## 8. ATOM wiring (`ATOM_DSV41_RK_K288`)

### 8.1 Files

- **`atom/models/deepseek_v41/rk/k288.py`** (new). Loads code objects through the same ctypes HIP-module launcher as
  WP6 (graph-capturable). At install, before any capture, it:
  - registers every eligible native group32 FP8 preshuffled weight with K % 32 == 0, K <= 512 and no bias (target and
    draft);
  - packs their scales;
  - loads every picked kernel.
  It then patches `blockscale.native_quant_linear`, plus the DSpark module's alias.
- **`atom/model_engine/model_runner.py`.** One env-guarded install block after WP6's. With the knob unset the module is
  never imported.

### 8.2 Behaviour

**Modes.**

- `=1` / `refused`: a call is routed only if `aiter_refuses(N, K, M)`, aiter's own selection, host only: a tuned row,
  else `pick_bmm_kernel_name`. At install `verify()` runs aiter once per registered shape at M = 1 and 4096 and compares;
  on a disagreement aiter's observed behaviour wins and is logged.
- `=all`: every K = 288 call.

**Picks.** Picks per M range come from `index.json`. An `xin=fp8` pick for a bf16 call runs ATOM's `quantize_fp8`
first. A call with `x_scale` takes the `fp8` picks. Any shape without a pick falls back to ATOM's path, logged once.

**Check and mutant modes** (gate only):

- `_CHECK=1`: install-time comparison with ATOM's local GEMM on the unshuffled weight, and with float64;
- `_CHECK=2`: the same comparison per call, eager only;
- `_MUTANT=<m>`: picks swapped for their must-fail twins.

### 8.3 Tests

- **`h/k1_atom.py`** (job k1s3), one fresh process per case. **PASS**:
  - *`refused` and `all`.* The K = 288 weight is routed at every shape, 2-D and 3-D. The K = 5120 weight stays on aiter
    (not even registered). Outputs are bitwise aiter's row-major result at 7 shapes, from 1 to 3000 rows. The predicate
    refuses K = 288 and runs K = 5120 at 8 M. A CUDA-graph capture replayed on new input passes the bound.
  - *`mutant`.* `_MUTANT=tail` is caught.
  - *`off`.* `rk.k288` is not imported, `native_quant_linear` is ATOM's, and aiter still refuses K = 288.
- **`ATOM-dsv41/tests/test_deepseek_v41_rk_k288.py`** (CPU, 12 pass locally with `--noconftest`): scale packing per byte
  for 6 K values, `unshuffle_weight` as the exact inverse of aiter's shuffle, the install refusals, and that the runner
  hook is env-guarded and the only import.
- **The served path.** The simulated TP8 server (§2) runs the knob through model construction, warm-up and FULL-graph
  capture.
- **Accuracy (GSM8K 1319 McNemar, passkey 8K / 24K)** needs a real TP8 server. It is owned by W17's TP8 bring-up, which
  this K1 unblocks.

## 9. Open items

1. **Real TP8 dry run.** It covers the mono kernels at TP8 widths and real collectives (§2.3); the command is ready.
   Simulated TP cannot host the mono, as the TP2 control proved.
2. **Fusion lever.** w2 fused into the shared expert's SwiGLU / quantization stage removes the one remaining HBM round
   trip at decode widths: the pick is already within 1-5 % of its standalone `nomath` floor.
3. **Prefill.** A structure that quantizes the activations once per M block and keeps it across N row groups could
   approach the output-write floor: the pick is about 2x `storeonly`.
4. **Rust direct defects** (§7) for the Rust track.
5. **TP8 aiter shapes without tuned rows** (`wo_b` 5120x1024, shared `gate_up` 576x5120, `fused_moe` at
   intermediate 384): performance risk for the TP8 baseline.
6. **Log noise.** The refusal predicate makes aiter log one "untuned" line per distinct M of the w2 shape.

## 10. Files

- **rocKE** (new, `library/kernels/gfx950/dsv41_mono/`): `k288_gemm.py` (kernel, host packing, float64 reference),
  `k288_space.py` (search space).
- **ATOM-dsv41**: `atom/models/deepseek_v41/rk/k288.py` (new), `atom/model_engine/model_runner.py` (one guarded hook),
  `tests/test_deepseek_v41_rk_k288.py` (new).
- **Harness** `hackathon-prep/dsv41/w17/`:
  - `w17_job.sh`: checks for 200 GB free in /tmp first, removes its own node-local job dir after saving results (PLAN 17);
  - `submit.sh`, `up.sh`, `mkstage.py`;
  - `h/`: `w17_run.sh` (phases `probe`, `build`, `gate`, `search`, `bench`, `atom`, `tp8`), `k1_probe.py`,
    `k1_common.py`, `k1_build.py`, `k1_gate.py`, `k1_search.py`, `k1_bench.py`, `k1_atom.py`, `textprobe.py`.
- **Jobs** (all on n07-17, 1 GPU each):

  | job | id | what |
  |---|---|---|
  | k1p1 | 68289009 | probe |
  | k1g1 | 68289229 | first gate: found the fp8 scale-straddle bug |
  | k1g2 | 68289303 | second gate: found the G1 binade rule and the NaN-class need |
  | k1s1 | 68289329 | first search |
  | k1s3 | 68289435 | final: gate, search, bench, atom |
  | k1gp | 68289574 | gate of the picks |
  | k1t1 | 68289566 | simulated TP8 |
  | k1t2 | 68289687 | simulated TP2 control |
