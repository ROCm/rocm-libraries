# W12 probe P3: DeepSeek-V4.1-Flash at TP1 on one MI355X (gfx950)

Ratios only. Absolute tok/s, accuracy and memory logs: `hackathon-prep/private-perf/dsv41-mono/p3-tp1.abs.md` and
`raw/p3-j{1a,1c,2a,3a}/`. Memory sizes below are capacities, not performance.

## Verdict

TP1 runs. It serves DeepSeek-V4.1-Flash with DSpark on one MI355X and passes the accuracy gate:

- **How it fits.** About 26 GiB of rarely routed experts and the input embedding live in pinned host memory, read over UVA. The other memory savings come from ATOM's own flags (smaller prefill batch, fewer captured graphs, utilization 0.97).
- **Mono.** ATOM's V4.1 mono refuses TP1 (`TP 1 not in (2, 4, 8)`), so TP1 runs the native path.
- **Speed vs TP2 at the same CONC (per-request median decode tok/s).** TP1 is slower at every CONC:

  | CONC | TP1 / TP2 |
  |---|---|
  | 1 | 0.47 |
  | 4 | 0.59 |
  | 16 | 0.74 |

  The comparator is ATOM's best served path (TP2 + DSpark + mono), measured in the same job; its A-vs-A twin range is 0.94-1.04.
- **Throughput per GPU.** TP1 delivers 1.0x / 1.18x / 1.41x the aggregate tok/s per GPU of TP2 at CONC 1 / 4 / 16. So TP1 replicas beat TP2 on throughput per GPU, not on per-request latency.
- **The 3x target.** P3 is not a 3x path. TP1 reads about 1.85x the weight bytes per GPU per step, and it has no mono. The 3x work stays at TP2/TP4.

## 1. Measured HBM footprint (ATOM_DSV41_RK_MEMREPORT, every parameter / buffer, storage-deduplicated)

GiB per GPU. Torch reports 287.98 GiB per MI355X. The research note's "1.06x of HBM" mixed GB and GiB: in GiB the
weights are 0.995x of HBM. Weights alone fit, but nothing else does.

| component | TP2 rank 0 (= rank 1) | TP1 (measured) | TP1 placement |
|---|---|---|---|
| routed experts (target, 40 x 384, MXFP4) | 134.77 | 268.95 | 243.00 HBM + 25.95 pinned host (cold) |
| attention projections (FP8) | 2.78 | 4.97 | HBM |
| shared expert (FP8) | 0.66 | 1.32 | HBM |
| embed (BF16) | 0.62 | 1.23 | pinned host (`EMBED_HOST`) |
| head (BF16) | 0.62 | 1.23 | HBM |
| vision tower (loaded, unused for text; replicated) | 0.52 | 0.52 | HBM |
| Engram `wkv` + gate + mHC + indexer + compressor + norms + other (replicated) | 0.81 | 0.81 | HBM |
| **target total** | **140.77** | **279.03** | 251.85 HBM + 27.18 host |
| DSpark draft (3 stages, own 128-expert MoE; shares the target's embed and head) | 3.82 | 7.39 | HBM |
| **all GPU weights** | **144.59** | **286.43** | 259.24 HBM |
| Engram tables (always host) | 91.6 page-locked | 183.1 page-locked | host |

### Runtime memory

ATOM's KV budget is `utilization x total - (peak_torch + non_torch + graphs + 2% safety)`.

| | TP2 (recipe flags, util 0.90) | TP1 (P3 flags, util 0.97) |
|---|---|---|
| peak_torch | 154.96 (weights + 8.4 warm-up activations at 16384 tokens) | 261.87 (weights + 2.6 at 4096 tokens) |
| non_torch / graphs / safety | 5.6 / 1.07 / 5.76 | 2.04 / 0.28 / 5.76 |
| KV + state pool | 91.8 | 9.40 (27013 blocks of 128 tokens) |
| capacity at full 32K context | 64 requests (slot-bound) | 16 requests (slot-bound); the blocks would hold 105 |

The V4.1 score workspace drops from 1.98 to 0.50 GiB with `--max-num-batched-tokens 4096`.

## 2. The TP1 plan

These are the four options the task listed, as measured.

| option | frees | cost / risk | status |
|---|---|---|---|
| (a) FP8 -> MXFP4 for attention + shared expert, BF16 head -> FP8 | at most ~3.9 GiB (2.3 + 0.6 + 0.6) | non-bitwise; new dense W4A8 GEMM paths; too small alone | not built: (b) is needed anyway |
| (b) cold routed experts in host memory, chosen by expert-usage statistics | any amount; 26 GiB chosen | rare UVA reads (decode cold share about 0.03%); two fused_moe calls per layer | **built, used** |
| (c) shrink pools / workspace | warm-up activations 8.4 -> 2.6, score workspace 1.98 -> 0.50, non_torch 5.6 -> 2.0, graphs 1.07 -> 0.28 | ATOM flags only | **used** (`--max-num-batched-tokens 4096 --attn-prefill-chunk-size 4096 --max-num-seqs 16 --cudagraph-capture-sizes [1..8,16] --gpu-memory-utilization 0.97`) |
| (d) drop duplicated draft weights | 0: the drafter already shares embed and head (deduplicated storage confirms it); only its 0.12 GiB Markov head is its own | - | nothing to drop. The unused vision tower (0.52, replicated) could go; not done |
| extra: input embedding in host memory (bitwise: same bytes, a lookup reads only the step's rows) | 1.23 | none measurable | **built, used** |

### Minimum offload

At util 0.97 the HBM weights may be at most 279.34 − (2.6 activations + 2.04 + 0.28 + 5.76 + 1.5 KV for 16 x 32K)
≈ 267.2 GiB. That needs at least 18.0 GiB of experts on the host when the embed is on the host. The plan takes
26 GiB: 1482 of the 15360 (layer, expert) slots, between 10 and 69 per layer, all 40 layers. That leaves KV room for
6.6x the CONC-16 x 32K need.

### How the cold set was chosen (`p3/plan.py`)

- **Calibration counts.** Routed-expert selection counts come from `ATOM_DSV41_RK_EXPSTATS`, a device-side
  `index_add_` that also runs in captured graphs. They were collected at TP2 on 512 GSM8K **train** prompts (disjoint
  from every eval prompt).
- **Selection rule.** Take the globally least-selected decode pairs; break ties by prefill count.
- **Cold-hit share by window.** Routing on GSM8K is very skewed:

  | window | decode selections that hit a cold expert | prefill selections that hit a cold expert |
  |---|---|---|
  | calibration | 0.011% | 1.3% |
  | held-out GSM8K test prompts, TP2 | 0.019% | 1.4% |
  | TP1 serving the bench cells | 0.032% | 1.2% |

- **Plan variants (held-out decode cold share).** Concentrating the same 26 GiB into fewer layers
  (`--layers-max K`) needs fewer two-call layers but raises the hit share. One uniform per-layer count (37) keeps a
  single kernel shape. Neither variant was timed.

  | variant | held-out decode cold share |
  |---|---|
  | `--layers-max 8` | 0.34% |
  | `--layers-max 16` | 0.072% |
  | uniform 37 per layer | 0.030% |

- **Domain caveat.** A general-traffic deployment needs a broader calibration set. That changes the hit share, not
  correctness.

## 3. Implementation (ATOM-dsv41, every change behind a default-off knob)

New module: `atom/models/deepseek_v41/rk/tp1.py`. The knobs:

- **`ATOM_DSV41_RK_COLD=<plan.json>`** — a planned layer's FusedMoE parameters are allocated and loaded on the CPU. At
  post-load the loader moves one layer at a time to the GPU, runs ATOM's unchanged MXFP4 processing (shuffle, scale
  shuffle), then splits the expert dimension:
  - hot experts: HBM tensors under the original names;
  - cold experts: pinned host tensors exposed as device tensors through a DLPack `kDLROCM` capsule at the
    `hipHostGetDevicePointer` address.

  The MoE then runs aiter `fused_moe` twice on the same routing (hot `expert_mask`, cold `expert_mask`) and adds the
  results. The masked calls use aiter's heuristic FlyDSL kernels (`flydsl_moe1_afp8_wfp4_*`); aiter has no tuned rows
  for these expert counts, nor for TP1's 2304-wide experts at all. Off = no allocation, loading or dispatch change.
- **`ATOM_DSV41_RK_EMBED_HOST=1`** — the target's embedding table (shared with the drafter) goes to pinned host memory.
- **`ATOM_DSV41_RK_MEMREPORT=<dir>`** — the component table above.
- **`ATOM_DSV41_RK_EXPSTATS=<dir>`** — the selection counts (graph-safe; costs a few small kernels per layer, so it is
  off in timed arms).
- **`ATOM_DSV41_RK_COLD_STREAM=1`** — the cold call on a side stream. This is a lever, **rejected on base T1-x26**: it
  measured 0.98-0.99x of T1 at TP1 and 0.95-0.97x of the plain split at TP2. It is kept default-off.
- **`ATOM_DSV41_RK_COLD_CHECK=<n>`** and **`ATOM_DSV41_RK_COLD_MUTANT=perm|zero|permall`** — the numerics gate and its
  must-fail mutants.

Hooks in shared files, all inert when unset:
- `atom/model_ops/moe.py`: the `create_weights` device context, the routing record, the split dispatch, and a refusal
  on the modular-kernel path;
- `atom/model_loader/loader.py`: the post-load step;
- `atom/model_engine/model_runner.py`: the embed move and the memreport.

The TP2 arms of job 2 ran the same stage with every knob off and reproduced the B0 TP2 numbers (C1 1.01x of B0).

No rocKE kernel was needed for P3. The split uses aiter's kernels; `dsv41_mono/` is untouched by this probe.

## 4. Correctness

### Numerics gate for the split (`COLD_CHECK`, TP2 where the full experts also fit)

On every eager MoE call of 6 layers (layers 1 and 14 have Engram), with 30% of experts cold (seeded), the gate compares:

- **split**: hot call + cold call;
- **ref**: the original single call;
- **floor**: a single call under an all-ones expert mask (the reduction-order floor).

Gate (pre-registered): relative L2 error on the rows that hit a cold expert must be ≤ max(4 x the floor, 0.02).

| run | checks with hits | split error on hit rows: median / p90 / max | floor max | verdict |
|---|---|---|---|---|
| split | 40 of 48 | 0 / 0.0041 / 0.0175 | 0.0018 | **PASS** |
| mutant `perm` (cold experts use a neighbour's weights) | 40 | max 1.00 | 0.0015 | **FAIL** (must-fail: caught) |
| mutant `zero` (cold output dropped) | 40 | max 1.00 | 0.0026 | **FAIL** (must-fail: caught) |

The split differs from the single call by one extra bf16 rounding of the two partial sums. The worst cases are the
warm-up's 16384-token dummy batches at small output magnitudes. Real-magnitude batches stay at or below 0.0043.

### Accuracy gate: TP1 vs TP2 in the same job (job 2, pass 1)

GSM8K 1319 is the 3-shot chat setup of the B0 baseline. Passkey is a 20-sample long-context check per length.

| arm | GSM8K flexible | GSM8K strict | passkey 8K | passkey 24K |
|---|---|---|---|---|
| TP2 (mono) | baseline | baseline | 20/20 | 20/20 |
| TP2 twin | +0.0046 | −0.0106 | 20/20 | 20/20 |
| **TP1** | +0.0030 | −0.0175 | 20/20 | 20/20 |
| mutant `permall` (every expert permuted) | −0.90 (0 correct) | 0 correct | 0/5 | 0/5 |
| mutant `perm` (cold experts only) | +0.0053 | +0.0151 | 20/20 | 20/20 |

Differences are vs the TP2 arm. Verdicts:

- **Flexible-extract.** TP1 lies inside the TP2 A-vs-A pair's range. **PASS.**
- **Strict-match.** TP1 is 0.007 (9 questions) below the lower of the two TP2 runs, so it is outside the literal pair
  range. It is inside the binomial 2-sigma band of a difference at n = 1319 (0.026). The near-identical `perm` mutant
  landed 0.033 above TP1, so strict-match moves by about 0.03 between near-equal models; one twin pair understates that
  noise. Verdict: **PASS by the pre-registered binomial rule; outside the literal twin range by 9 questions** (reported,
  not hidden).
- **Passkey.** **PASS.**
- **`permall` must-fail.** It FAILs every part, so the gate can fail.
- **`perm` mutant passes the end-to-end gate.** Cold experts take about 0.03% of decode selections, so no end-to-end
  accuracy gate can see a wrong cold path. That is why the layer numerics gate above is the gate for the split itself.

## 5. Performance

### TP1 vs TP2 at the same CONC

Job 2 runs everything on one node, with 2 rotated passes and a fresh server per arm-run. The cell is the median over
passes of the per-pass ratio of per-request median decode tok/s, with [min-max] in brackets. Full table:
`j2a.ratio.md`.

| arm | C1 | C4 | C16 |
|---|---|---|---|
| TP2 twin (A-vs-A) | 1.014 [0.994-1.035] | 0.969 [0.954-0.985] | 0.965 [0.944-0.987] |
| **TP1** (26 GiB cold + embed on host) | **0.470** [0.463-0.477] | **0.593** [0.572-0.614] | **0.736** [0.688-0.783] |
| TP1 + cold side stream (rejected lever) | 0.466 | 0.579 | 0.732 |
| TP1 aggregate tok/s per GPU vs TP2 per GPU | 1.00 [0.99-1.01] | 1.18 [1.15-1.20] | 1.41 [1.38-1.44] |

- TTFT is 1.1-1.3x TP2's.
- DSpark acceptance is unchanged: AL 3.67-3.70 at C1 on both.
- Startup: the first TP1 start with cold caches took about 8x a warm start, because 40 distinct hot/cold expert
  counts are new MoE shapes for FlyDSL to JIT. Warm starts (node cache) match TP2's start time.

### What the split itself costs (job 3, TP2, comm-fused off in every arm, mono off)

| arm | C1 | C4 | C16 |
|---|---|---|---|
| A-vs-A twin | 0.978 [0.948-1.009] | 0.998 [0.948-1.048] | 1.069 [1.046-1.091] |
| plain split (S) | 0.973 [0.942-1.003] | 0.928 [0.908-0.949] | 0.971 [0.962-0.981] |
| split + side stream | 0.947 | 0.894 | 0.926 |

The split stays inside the twin band at C1 and C16 and is about 2% under it at C4. Its cost is the extra masked
launches per layer, plus the rare UVA reads. Full table: `j3a.ratio.md`.

### Why TP1 is slower per request

Ranked by size:

1. **Bytes per GPU per step.** TP1 reads about 1.85x the weight bytes of a TP2 rank at batch 1 (research
   `model.md` §7).
2. **No mono at TP1.** ATOM's mono adds 1.59x at C1 on TP2 (B0). Against B0's TP2 native+DSpark, TP1 is about 0.76x at
   C1 (cross-job estimate).
3. **Untuned FlyDSL configs** for TP1's MoE shapes (no tuned rows exist for 2304-wide experts or the split expert
   counts).
4. **The split's own cost**, at most about 5%.

The per-GPU aggregate ratio shows that TP2 spends its second GPU mostly on latency, not throughput.

### Neighbours

Jobs 2 and 3 overlapped on n01-09 from 06:06 to 06:42 UTC, plus 2-4 foreign 1-GPU jobs. Both jobs pair every
comparison in-job with rotated passes. A claimed end-to-end *win* would need both nodes, but P3 claims none.

## 6. Levers and next steps

- **Kept default-off:**
  - `COLD_STREAM` (rejected on base T1-x26; re-test after any change to the MoE kernels);
  - the `--layers-max` and uniform plans (untimed);
  - `EXPSTATS` (diagnostic).
- **To improve TP1, not done here:**
  - the mono at TP1 (dims for TP1; its MoE kernel would need the hot/cold split);
  - tuned aiter MoE rows for the TP1 shapes;
  - a uniform per-layer cold count, so one kernel shape serves every layer;
  - dropping the unused vision tower (0.52 GiB) to shrink the offload.

## Repro

Harness: `hackathon-prep/dsv41/p3/`. The stage is `mkstage.py` of `dev/ATOM-dsv41`; jobs go through `p3_job.sh` and
`submit_wait.sh`.

- `step1.sh` (job 68255484 + 68255677): TP2 memreport + calibration counts, plans, the numerics gate and its mutants,
  TP1 bring-up.
- `step2.sh` (job 68256060): TP1 vs TP2, the accuracy gate and its mutants.
- `step3.sh` (job 68256069): the split's cost at TP2.
- Plan used: `plan_x26.json` from job 68255484 (`P3_REUSE=p3-j1a`).

Ratio tables: `summ.py`.
