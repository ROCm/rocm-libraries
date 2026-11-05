# DeepSeek-V4.1-Flash B0 baseline on ATOM `main` (gfx950 x 8 server, alola `rocke` partition)

Ratios only. Absolute numbers: `hackathon-prep/private-perf/dsv41-mono/B0-ABSOLUTES.md` (+ `B0-abs.md`, raw results).
Exact repro: [HOWTO.md](HOWTO.md). Full ratio table: [ratios-vs-tp4ep.md](ratios-vs-tp4ep.md). Measured 2026-11-03 on
`smci355-ccs-aus-n01-09` (jobs 68254511 TP2, 68254624 TP4, 68255255 traces).

## What ran

- **Stack:** ATOM `origin/main` `df86c482` (2026-10-16) in `rocm/atom-dev:nightly_202610171506`, which ships that same
  ATOM commit. aiter is the image's `2cc8c750` (2026-10-16), not the recipe's `2039d2b9` (2026-09-13). The recipe's pin
  is an ancestor of the image's aiter; the baseline uses what ATOM main's CI image runs.
- **Checkpoint:** `deepseek-ai/DeepSeek-V4.1-Flash` at the recipe revision `dba1be0a`. It is ungated (MIT), 475 GiB,
  and was downloaded and sha-verified to node-local NVMe (no license step, no credentials).
- **Server flags:** the AgentX recipe's flags, with bf16 KV and FULL graphs, except:
  - `--level 0` instead of 3; level 3 stalls in inductor (below);
  - no fixed acceptance length;
  - `--max-model-len 32768`.
- **Process:** fresh server per arm-run, untimed warm-up at each cell's concurrency, fixed GSM8K prompts, greedy,
  1024-token cap.
- **Twins:** A-vs-A twins of TP2 native, TP4 native and TP4 mono, each on its own server.
- **Accuracy:** GSM8K 3-shot (lm_eval, first 100 questions, 8 concurrent).
- **Traces:** one profiled CONC-1 decode window (48 DSpark steps) per arm.

Arm names: `<tp>-<ep|noep>-<ds1|ds0>-<m1|m0>-<i8|i4>`: TP size, expert parallel, DSpark (5 draft tokens) on / off,
`ATOM_MONO_ENABLE` 1 / 0, index cache fp8 / fp4.

## Which path served (server-log proof)

The recipe says the V4.1 mono is "on by default but currently refused" (MXFP4 activations in K2b). On `main` that
text is stale; the logs show which path served each arm:

| deployment | `ATOM_MONO_ENABLE=1` log line |
|---|---|
| TP4 no-EP, DSpark (fp8 or fp4 index) | `V4.1 mono decode on` + `V4.1 mono draft on` (every rank) |
| TP4 no-EP, no speculation | `V4.1 mono decode on` |
| TP2 no-EP, DSpark / no speculation | same as TP4 |
| TP4 + EP (primary deployment) | `V4.1 mono decode off: expert parallel` (refused) |
| any arm with `ATOM_MONO_ENABLE=0` | `V4.1 mono decode off: switched off` |

The mono builds one kernel set per step width (log `V4.1 mono kernels built for width N`, up to 48 rows) during
graph capture. With DSpark it routes at most 8 requests (6 rows each), and the CONC 16 / 64 cells fall back to the
original path. So the "mono" arms are the mono at CONC <= 8 and the original path above that.

## Decode speed (per-request median decode tok/s, ratio to TP4 + EP native DSpark at the same CONC)

| arm | c1 | c2 | c4 | c8 | c16 | c64 |
|---|---|---|---|---|---|---|
| TP4 + EP, DSpark, native (reference) | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| TP4 no-EP, DSpark, native | 0.999 | 1.111 | 1.073 | 0.976 | 1.004 | 0.734 |
| TP4 no-EP, DSpark, **mono** | **1.790** | 1.923 | 1.238 | 1.089 | 0.936 | 0.902 |
| TP4 no-EP, DSpark, mono, twin | 1.795 | 1.751 | 1.244 | 1.105 | 0.998 | 0.972 |
| TP4 no-EP, DSpark, mono, fp4 index | 1.778 | 1.779 | 1.301 | 1.213 | 0.960 | 0.884 |
| TP4 no-EP, no spec, native | 0.324 | 0.396 | 0.394 | 0.479 | 0.575 | 0.505 |
| TP4 no-EP, no spec, mono | 0.653 | 0.793 | 0.688 | 0.706 | 0.740 | 0.576 |
| TP2 no-EP, DSpark, native | 0.910 | 0.980 | 0.842 | 0.867 | 0.788 | - |
| TP2 no-EP, DSpark, mono | 1.444 | 1.386 | 1.011 | 0.850 | 0.808 | - |
| TP2 no-EP, no spec, native / mono | 0.306 / 0.580 | | | | | |

**Noise.** The A-vs-A twins agree within 0.016 at CONC 1 (TP4 native 0.984, TP4 mono 1.003, TP2 native 1.006). At
CONC >= 2 they spread up to +-0.15 (16-128 prompts of uneven length, one pass). CONC 1 is the decision cell; CONC >= 2
values are indicative only.

## Derived ratios at CONC 1

- **Mono vs native, DSpark:** TP4 1.79 (twin 1.80), TP2 1.59. Without speculation: TP4 2.01, TP2 1.89.
- **DSpark vs no speculation:** native TP4 3.08, mono TP4 2.74.
- **TP4 + EP vs TP4 no-EP (both native):** 1.00. Whole-expert EP has no CONC-1 decode cost or gain, but it is the
  configuration the mono refuses.
- **TP2 vs TP4 at the same CONC:** native 0.91, mono 0.81.
- **Two TP2 servers vs one TP4 at equal GPUs** (aggregate output tok/s, 2 x TP2 at CONC c/2 against TP4 at CONC c,
  derived from single-server cells; noise +-0.15):
  - native DSpark: 1.21 (c2), 1.16 (c4), 1.46 (c8);
  - mono DSpark: 1.27 (c2), 1.34 (c4), 1.10 (c8).
  - Throughput favours 2 x TP2; per-request latency favours TP4.
- **fp4 vs fp8 index plane:** mono 0.994, native 0.975 (within or near the noise band).
- **Acceptance length:** 3.6-3.9 in every DSpark cell, mono and native alike (TP4 c1: 3.71 / 3.71). The mono does not
  change acceptance.

## Accuracy (GSM8K 3-shot, lm_eval local-chat-completions, first 100 questions, 8 concurrent)

| arm | flexible-extract | strict-match |
|---|---|---|
| TP4 + EP, DSpark, native | 0.94 | 0.20 |
| TP4 no-EP, DSpark, native | 0.93 | 0.15 |
| TP4 no-EP, DSpark, mono | 0.94 | 0.21 |
| TP2 no-EP, DSpark, native | 0.94 | 0.22 |
| TP2 no-EP, DSpark, mono | 0.94 | 0.20 |

`strict-match` is low for every arm because the thinking-mode answers do not end in the `#### N` format. Use
`flexible-extract`. With n = 100 the standard error is about 0.024, so the mono and native arms agree.

## Kernel breakdown of one decode step (CONC 1, rank 0, share of summed GPU kernel time)

Window: 48 DSpark steps (draft pass + target verify) after the first 32, on a long-generation prompt with the
profiler on (`ignore_eos`). The window's acceptance length was 2.4-3.3, lower than the benchmark's. Families are a
regex map of kernel names (`hackathon-prep/dsv41/bench/analyze_trace.py`).

| family | TP4+EP native | TP4 native | TP4 mono | TP4 native no-spec | TP4 mono no-spec | TP2 native | TP2 mono |
|---|---|---|---|---|---|---|---|
| mono K2 (`v41_layer_post`: indexer, attention, out-proj, MoE, all-reduces) | - | - | 0.641 | - | 0.665 | - | 0.689 |
| mono K1 (`v41_attn_pre`: mHC seam, norm, q/kv, indexer query) | - | - | 0.172 | - | 0.186 | - | 0.153 |
| MoE (sort, quant, gate-up, down, reduce, router, top-k) | 0.303 | 0.270 | 0.000 | 0.259 | 0.000 | 0.351 | 0.000 |
| dense GEMM (bmm a8w8 mxscale, a4w4, hgemm) | 0.166 | 0.211 | 0.058 | 0.208 | 0.053 | 0.186 | 0.060 |
| mHC / RMSNorm | 0.142 | 0.142 | 0.010 | 0.159 | 0.011 | 0.137 | 0.008 |
| RoPE / quant (`quantize_fp8_kernel` alone is 0.08) | 0.135 | 0.136 | 0.010 | 0.157 | 0.010 | 0.129 | 0.008 |
| all-reduce / all-gather | 0.105 | 0.090 | 0.017 | 0.072 | 0.013 | 0.055 | 0.014 |
| attention (sparse decode, compress-attn) | 0.074 | 0.076 | 0.007 | 0.079 | 0.006 | 0.070 | 0.005 |
| elementwise / copy | 0.035 | 0.034 | 0.017 | 0.032 | 0.014 | 0.033 | 0.013 |
| indexer / score / index write | 0.013 | 0.013 | 0.011 | 0.015 | 0.011 | 0.013 | 0.009 |
| sampling / DSpark Markov head | 0.008 | 0.008 | 0.016 | 0 | 0 | 0.008 | 0.014 |
| Engram (UVA lookup) | 0.004 | 0.004 | 0.008 | 0.005 | 0.011 | 0.004 | 0.006 |
| compressor state / KV write | 0.004 | 0.004 | 0.008 | 0.004 | 0.010 | 0.003 | 0.006 |
| other | 0.013 | 0.013 | 0.013 | 0.010 | 0.008 | 0.012 | 0.010 |

Step time and device busy share (rank 0, profiled):

| comparison | step-span ratio | device busy share |
|---|---|---|
| TP4 mono / TP4 native, DSpark | 0.504 | native 0.91, mono 0.96 |
| TP4 mono / TP4 native, no spec (per token) | 0.446 | native 0.90, mono 0.97 |
| TP2 mono / TP2 native, DSpark | 0.574 | native 0.92, mono 0.98 |
| TP4 + EP / TP4 no-EP, native | 1.003 | |

- **Native path:** a DSpark step is about 40 launches per layer-kernel class (`calls_per_step` about 38-41). The time
  spreads over MoE, dense GEMM, mHC / norm, quant (fp8 quantize kernels) and all-reduce; about 9 % of the span the
  device is idle.
- **Mono path:** 81-85 % of the kernel time is the two persistent layer kernels. K2 (`layer_post`, which includes the
  MoE and both all-reduces) alone is 64-69 %. The non-mono rest is the draft LM head / Markov sampler, compressor,
  Engram, the target LM head GEMMs and small copies, about 15-19 %.
- **Next target:** for a rocKE true mono that rest plus K2 is what to attack. Per-kernel lists (top 40 by time, calls
  per step) are in the private `raw/b0trace/trace_*.breakdown2.json`.

## Findings and caveats

1. **The recipe's mono refusal is stale.** Mono serves TP2 / TP4 no-EP with or without DSpark (fp8 or fp4 index) and
   refuses only EP. The primary deployment (TP4 + EP) therefore runs the original path. ATOM's measured mono
   speed-ups need the no-EP arrangement, which at CONC 1 decodes as fast as EP on the original path.
2. **`--level 3` (the AgentX recipe's compile level) stalled for 25+ minutes in inductor** at the first server start
   (py-spy: deep `ExternKernel.__str__` recursion while compiling `backbone_late`, GPU idle). The baseline therefore
   uses `--level 0` + FULL graphs, the recipe's development / acceptance mode and the mode of its mono table. A
   level-3 comparison is open.
3. **aiter differs from the recipe pin** (image `2cc8c750` vs pin `2039d2b9`, about a month newer). A pin-exact arm would
   need an aiter rebuild in the container. Not done.
4. **CONC >= 2 needs more prompts or passes** before any sub-15 % claim (see Noise).
5. **The trace window's acceptance length (2.4-3.3) is below the benchmark's (~3.7)** because of the long-generation
   `ignore_eos` prompt. The breakdown is per DSpark step, so it is unaffected; per-token figures from it are not
   comparable with the benchmark.
6. **Server start:** with the node's compile caches warm (`/tmp/vanantha-dsv41/cache`), a start is about a quarter
   of the first cold start; the mono's per-width builds roughly double a warm start (start times in the private notes).

## Next steps

1. Pin the reference for the rocKE work: TP4 no-EP DSpark fp8-index, mono off (native) and mono on (ATOM mono), CONC
   1 / 2 / 4 / 8, plus TP4 + EP native as the deployment reference. Re-measure CONC >= 2 with 3+ rotated passes and
   64+ prompts per cell.
2. Per-kernel targets from the traces: K2 `layer_post` (MoE + all-reduces inside), K1 `attn_pre`, then the non-mono
   rest per step (draft LM head / Markov sampler, compressor, Engram UVA lookup, LM-head GEMMs, copies).
3. Open baseline gaps: a `--level 3` arm (after the inductor stall is understood), an aiter-pin-exact (`2039d2b9`)
   arm, a real 2 x TP2 run on 4 GPUs (the derived ratios above assume no interference), longer contexts (8K-32K
   prompts), and a full GSM8K run (1,319) on the mono and native arms.
4. `ATOM_MONO_TIMELINE` stamps of K1 / K2 phases (eager) to split the 64-69 % K2 share into indexer / attention /
   out-proj / MoE / all-reduce.
