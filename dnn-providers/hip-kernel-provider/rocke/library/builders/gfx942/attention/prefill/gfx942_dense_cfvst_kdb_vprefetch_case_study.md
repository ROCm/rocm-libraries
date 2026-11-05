# gfx942 bf16 D128 causal prefill: CFVST, K LDS double buffering, and V register prefetch

This change implements opt-in K LDS Double Buffering and V register pre-fetching on the CFVST body of D128 bf16 persistent=true causal prefill attention gfx942 kernels. CFVST itself is already the D128 default for both dtypes (`use_cfvst=None` resolves on via `_use_cfvst`, since #14367); this change does not alter it. Both new levers are off by default.

## Problem

Benchmarking, profiling and tracing highlighted bottlenecks in:

1. V read from LDS: V was stored in traditional layout in LDS. Narrow reads.
2. Allocation of LDS per WG (34 KiB, 34,816 B, in the production baseline). Max limit per CU is 64 KiB. This limits WG occupancy in CU to 1 while leaving 30 KiB of LDS unused.
3. Stalls at sync point for loading K and V tiles before MFMA.

The concerned optimization points were V LDS layout, register usage, LDS usage, software pipelining.

## Step-0 Lever Sweep

Lever sweep was performed to check whether performance could be improved by tuning levers to affect optimization points: **LDS usage/layout, software pipelining/scheduling, and register pressure**. Available levers are listed below along with reasons for selection decision and values swept.

#### Configuration

| Parameter | Value |
|---|---|
| GPU | (`gfx942`, 304 CUs) |
| Repository / branch | [ROCm/rocm-libraries](https://github.com/ROCm/rocm-libraries), `develop` |
| Source commit | `ceeae550e7aa95b291aff5cb75ec4d626a072bf3`|
| Data type | **BF16** |
| Head dimension (`D`) | **128** |
| Sequence length (`Sq = Skv = S`) | **4096, 8192, 16384** |
| Batch size (`B`) | **1, 16** |
| Query heads (`Hq`) | **32** |
| KV heads (`Hkv`) | **8, 16** |
| Attention | Dense, causal prefill |

Eight `(B, S, Hkv)` workloads were swept independently. The completed sweep report records this source commit with LLVM flavor `llvm23`.

#### Lever inventory and exclusions

| Lever | Swept? | Group | Reason if excluded |
|---|---|---|---|
| `use_cfvst` | No | — | Not supported when enabled for BF16 D128 at this commit; keep the non-CFVST baseline. |
| `lds_row_pad` | **Yes** | A | — |
| `lds_k_group_pad` | No | — | Only changes the D64 K-LDS row-group layout; D128 uses `lds_row_pad`. |
| `persistent` | No | — | Selects persistent vs nonpersistent grid scheduling, not the inner K/V LDS/register pipeline; preserve production dispatch. |
| `block_m` | **Yes** | A | — |
| `block_n` | **Yes** | A | — |
| `num_persistent` | No | — | Changes persistent workgroup count and device-level utilization, not the inner software pipeline; preserve production dispatch. |
| `persist_decode` | No | — | Changes grid-level mapping of work to query blocks/heads/batches; preserve production dispatch. |
| `interleave` | No | — | Changes query-block work ordering across CTAs, not instruction interleaving within the K/V pipeline; preserve production dispatch. |
| `use_v_swizzle` | No | — | Only affects the CFVST transposed-V LDS path, which is disabled for BF16 D128. |
| `v_row_pad` | No | — | Only affects the CFVST transposed-V LDS layout; has no effective tuning impact on the selected baseline path. |
| `use_exp2_fast` | **Yes** | C | — |
| `iglp` | **Yes** | B | — |
| `iglp_mode` | **Yes** | B | — |
| `pv_sched_fence_mask` | **Yes** | B | — |
| `pv_priority` | **Yes** | B | — |
| `pv_loop_order` | **Yes** | A, B | — |
| `o_store_width` | **Yes** | D | — |
| `causal_diag_split` | **Yes** | A, C | — |
| `waves_per_eu` | **Yes** | A, B, C | — |
| `lds_num_buffers` | No | — | Only `1` is implemented/valid for the kernel at this revision; no useful sweep axis. |
| `lazy_rescale` | No | — | Inactive in the gfx942 kernel body; only changes the kernel-name token. |
| `emit_lse` | No | — | Requests an additional LSE output, rather than tuning the same output contract. |

**Count: 12 swept.** The grid knobs were **not forced off**; each workload retained its dispatch-resolved settings. CFVST, K LDS double buffering, and V register prefetch were not enabled in the baseline sweep.

#### Grouping the 12 swept knobs

| Group | Purpose | Knobs |
|---|---|---|
| **A — Geometry / LDS** | Tile size, LDS layout, occupancy-related interactions | `block_m`, `block_n`, `lds_row_pad`, `waves_per_eu`, `pv_loop_order`, `causal_diag_split` |
| **B — Instruction scheduling** | IGLP, barriers, PV ordering and priority | `iglp`, `iglp_mode`, `pv_sched_fence_mask`, `pv_priority`, `pv_loop_order`, `waves_per_eu` |
| **C — Softmax / causal computation** | Exponential policy, diagonal handling and wave interactions | `use_exp2_fast`, `causal_diag_split`, `waves_per_eu` |
| **D — Output stores** | Store vector width | `o_store_width` |

There are **12 unique knobs**, not 16: `waves_per_eu`, `pv_loop_order`, and `causal_diag_split` appear in multiple groups so their interactions can be revisited.

#### Staged sweep methodology

The sweep followed **A1 → A2 → C → B1 → B2 → D → R1 → R2**, independently for each workload.

| Stage | Procedure |
|---|---|
| **A1 — Geometry screening** | Sweep `block_m={64,128,256}`, `block_n={32,64}`, `lds_row_pad={0,4,8,16}`, `waves_per_eu={1,2,4}` (72 combinations before legality/deduplication). Shortlist promising complete configurations. |
| **A2 — Geometry interactions** | On A1 finalists, retune `waves_per_eu={1,2,3,4}`, `pv_loop_order={d_major,k_major}`, `causal_diag_split={False,True}`. |
| **C — Computation** | On A2 finalists, tune `use_exp2_fast={Auto,False,True}`, `causal_diag_split`, and `waves_per_eu`. |
| **B1 — Scheduling** | On C finalists, test legal `iglp`/`iglp_mode`/`pv_sched_fence_mask` combinations together with `pv_loop_order` and `pv_priority={0,1,2,3}`. IGLP and explicit scheduling fences are mutually exclusive. |
| **B2 — Scheduling interactions** | On B1 finalists, retune `waves_per_eu` and `pv_loop_order`. |
| **D — Output** | On B2 finalists, test `o_store_width={1,2,4}`. |
| **R1, R2 — Cross-group refinement** | Two bounded rounds of single-knob neighborhood changes around the strongest **full configurations**, allowing adjustments to earlier group choices. |

At every stage, the sweep rejects unsupported/duplicate effective specs, numerically checks legal candidates against FP32 causal SDPA (`atol=rtol=1e-2`), screens GPU latency, then remeasures promising candidates against the production baseline. **Several top full configurations** are carried forward, rather than choosing one winner per independent knob. This is a **bounded staged search**, not an exhaustive Cartesian-product sweep.

#### Production dispatch default vs winning kernels

These are the best **final A/B-tested** configurations, compared with each workload's production-dispatch baseline, as measured by the sweep at its source commit. Results table 3 remeasures the same winners at the benchmarked HEAD. Each finalist passed a BF16-output-versus-FP32 causal SDPA numerical check (`atol=rtol=1e-2`) before timing. The timing statistic is the median of five alternating paired baseline/candidate rounds, with 50 timed launches per round.

| B | S | Hkv | Speedup | Latency reduction | Paired min–max | Baseline kernel | Winning kernel |
|---:|---:|---:|---:|---:|---|---|---|
| 1 | 4096 | 8 | **1.10141×** | **9.21%** | 1.09785–1.10610× | `rocke_attention_dense_d128_bn64_bf16_causal_lazyrs_persist304_bthkvmin_gfx942_wpe2` | `rocke_attention_dense_d128_bn64_bf16_causal_lazyrs_persist304_bthkvmin_gfx942_wpe3_krowpad4_iglp1_iglpm1_pvkmaj_prio1_dsplit` |
| 1 | 4096 | 16 | **1.09689×** | **8.83%** | 1.09584–1.09764× | `rocke_attention_dense_d128_bn64_bf16_causal_lazyrs_persist304_bthkvmin_gfx942_wpe2` | `rocke_attention_dense_d128_bn64_bf16_causal_lazyrs_persist304_bthkvmin_gfx942_wpe2_krowpad4_iglp1_iglpm1_pvkmaj_prio1_dsplit` |
| 1 | 8192 | 8 | **1.10407×** | **9.43%** | 1.10167–1.10657× | `rocke_attention_dense_d128_bn64_bf16_causal_lazyrs_persist304_bthkvmin_gfx942_wpe2` | `rocke_attention_dense_d128_bn64_bf16_causal_lazyrs_persist304_bthkvmin_gfx942_wpe3_krowpad4_iglp1_iglpm1_pvkmaj_prio1_dsplit` |
| 1 | 8192 | 16 | **1.10338×** | **9.37%** | 1.10162–1.10493× | `rocke_attention_dense_d128_bn64_bf16_causal_lazyrs_persist304_bthkvmin_gfx942_wpe2` | `rocke_attention_dense_d128_bn64_bf16_causal_lazyrs_persist304_bthkvmin_gfx942_wpe3_krowpad4_iglp1_iglpm1_pvkmaj_prio2_dsplit` |
| 1 | 16384 | 8 | **1.09043×** | **8.29%** | 1.08865–1.09340× | `rocke_attention_dense_d128_bn64_bf16_causal_lazyrs_persist304_bthkvmin_gfx942_wpe2` | `rocke_attention_dense_d128_bn64_bf16_causal_lazyrs_persist304_bthkvmin_gfx942_wpe3_krowpad4_iglp1_iglpm1_pvkmaj_prio3_dsplit` |
| 16 | 4096 | 8 | **1.10144×** | **9.21%** | 1.10002–1.10216× | `rocke_attention_dense_d128_bn64_bf16_causal_lazyrs_persist304_hkvmaj_gfx942_wpe2` | `rocke_attention_dense_d128_bn64_bf16_causal_lazyrs_persist304_hkvmaj_gfx942_wpe2_krowpad4_iglp1_iglpm1_pvkmaj_prio3_dsplit` |
| 16 | 4096 | 16 | **1.08896×** | **8.17%** | 1.08857–1.08966× | `rocke_attention_dense_d128_bn64_bf16_causal_lazyrs_persist304_gfx942_wpe2` | `rocke_attention_dense_d128_bn64_bf16_causal_lazyrs_persist304_gfx942_wpe2_krowpad4_iglp1_iglpm1_pvkmaj_prio3_dsplit` |
| 16 | 8192 | 8 | — | — | — | — | **Missing files** |

This shape has no completed Step-0 result, so it is compared against production dispatch only.

## Optimization 1: Enable CFVST

V was stored in traditional layout in LDS. This caused narrow loads. The hypothesis was that transposed layout would enable wide loads. This feature was already present (CFVST) but disabled for bf16 due to register spillage concerns. Performance improved substantially with it enabled, so it is offered as an opt-in.

CFVST transposes V into a layout that enables wider LDS reads during the PV MFMA stage.

**Comparison:** Production default vs. CFVST enabled
**Workload:** B=1, S=4096, Hq=32, Hkv=8, D=128, BF16, causal, gfx942.

| Metric | Production default | CFVST enabled |
|---|---|---|
| Static `ds_read_u16` instructions | Baseline | **−100%** |
| Static `ds_read2st64_b64` instructions | Baseline | Introduced (0 → 16; percentage undefined) |
| Executed LDS instructions | Baseline | **−71.23%** |

Static counts are from the benchmark run's HSACOs; executed LDS instructions are `SQ_INSTS_LDS` per dispatch from `rocprofv3 --pmc`, both at the benchmarked HEAD.

**Conclusion:** CFVST improves performance through wider V LDS reads and reduced LDS instruction traffic. It is **not** justified as a bank-conflict reduction optimization.

### Register Usage and Static Occupancy (§R.4)

Static counts from the exact HSACOs timed in the benchmark run (`static_resources.csv`; LLVM23, BM256/BN64, persistent-304, 512 threads/WG), for B=1, S=4096, Hkv=8. Instruction columns are static counts in the code object, not executed counts.

| Arm | VGPR | SGPR | VGPR / SGPR spills | Scratch | LDS bytes | `v_mfma` | `ds_read` | `ds_write` | `buffer_load` | `s_barrier` | `s_waitcnt` | VALU | SALU |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A (Step-0 winner) | 249 | 106 | 0 / 1 | 0 B | 33792 | 128 | 288 | 0 | 0 | 5 | 86 | 1645 | 493 |
| B1 baseline | 194 | 44 | 0 / 0 | 0 B | 34816 | 64 | 144 | 0 | 0 | 3 | 86 | 1194 | 469 |
| B2 CFVST | 225 | 45 | 0 / 0 | 0 B | 33792 | 64 | 32 | 8 | 8 | 3 | 31 | 1225 | 428 |
| B3 CFVST + KDB | 230 | 78 | 0 / 0 | 0 B | 51200 | 128 | 64 | 16 | 16 | 5 | 52 | 1600 | 600 |
| B4 CFVST + VPF | 230 | 76 | 0 / 0 | 0 B | 33792 | 128 | 64 | 16 | 16 | 5 | 57 | 1606 | 573 |
| B5 CFVST + KDB + VPF | 230 | 78 | 0 / 0 | 0 B | 51200 | 128 | 64 | 16 | 16 | 5 | 59 | 1604 | 585 |
| B6 Full + pad4 | 230 | 78 | 0 / 0 | 0 B | 50176 | 128 | 64 | 16 | 16 | 5 | 59 | 1604 | 586 |
| B7 Full + pad4 + diagonal split | 232 | 81 | 0 / 0 | 0 B | 50176 | 192 | 96 | 24 | 24 | 7 | 88 | 1964 | 661 |

Every B arm is spill-free with 0 B scratch on all twelve shapes. `A` has one SGPR spill on six of its seven shapes. Every arm reports 2 waves/SIMD. Across shapes, the B arms change only in their SGPR, VALU, and SALU counts, following each shape's production block order.

The SGPR V addressing (the wave index scalarized before the V buffer offsets) is used only by the KDB and VPF pipeline paths; CFVST alone and the default kernel are unchanged. It has no separate knob, so its delta is included in the CFVST → CFVST + KDB and CFVST → CFVST + VPF rows. The gfx950 dense kernel computes its wave index in VGPRs and does not share this pattern.

## Optimization 2 and 3: K LDS Double Buffering + V register pre-fetching

Analysis of stalls using threadtrace found a large number of waiting at sync points. One such point was waiting for K and V tiles to be loaded before MFMA. This wait restricted MFMA and memory loading parallelism. The hypothesis was that K LDS Double Buffering + V register pre-fetching would introduce parallelism and latency hiding. Optimizing 1 would leave other the bottleneck. The optimization improved performance substantially, so it is offered as an opt-in.

### Synchronization-Stall Analysis

Advanced Thread Trace (ATT) was used to evaluate synchronization overhead for gfx942 BF16 D128 causal prefill (B=1, S=4096, Hq=32, Hkv=8): `rocprofv3 --att` on CU 1, one traced dispatch per kernel at the benchmarked HEAD. Stalls are the `s_barrier` and `s_waitcnt` stall cycles in the trace's `code.json`.

**CFVST + KDB + VPF vs. CFVST**

| Metric | Reduction |
|---|---:|
| BARRIER hits | 0.00% |
| BARRIER issue stalls | **10.70%** |
| WAITCNT issue stalls | **27.23%** |
| **Combined issue stalls** | **21.31%** |

The combined pipeline reorganizes K/V loading through a prologue, pipelined main loop, and epilogue. ATT records the same aggregate number of barrier hits across both variants, but fewer BARRIER- and WAITCNT-attributed issue stalls.

These results are consistent with reduced synchronization overhead and improved latency hiding through K double buffering and V register prefetch.

## Results

> **Dated results.** Every timing, static count and trace below was measured at `cc84e2621e8fd9ce83f9bf43dbc1564c690c290d`, before this branch merged #14367. At that commit production dispatch resolved BF16 D128 CFVST **off**, so B1 was production dispatch and B2 added CFVST. Since #14367 production resolves CFVST **on**: the harness now pins `use_cfvst=False` for B1 and A, and B2 *is* production dispatch. The arm definitions (resolved CFVST/KDB/VPF per arm) are unchanged, but kernel sources have changed since `cc84e262`, so these numbers are not re-measured at the PR head and `--verify-provenance` against it reports the source changes.

All workloads use **BF16, Hq=32, D=128, persistent=True, causal attention** on (gfx942).

### 1. Ablation: Each Configuration vs. Step-0 Winner

| B | S | Hkv | CFVST | CFVST + KDB | CFVST + VPF | CFVST + KDB + VPF | Full + LDS pad4 + causal split | CFVST min–max | CFVST + KDB min–max | CFVST + VPF min–max | CFVST + KDB + VPF min–max | Full + LDS pad4 + causal split min–max |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1024 | 8 | — | — | — | — | — | — | — | — | — | — |
| 1 | 1024 | 16 | — | — | — | — | — | — | — | — | — | — |
| 1 | 2048 | 8 | — | — | — | — | — | — | — | — | — | — |
| 1 | 2048 | 16 | — | — | — | — | — | — | — | — | — | — |
| 1 | 4096 | 8 | 1.37% | 3.64% | 5.98% | 7.81% | 11.44% | 1.00% to 57.80% | 3.29% to 59.00% | 5.53% to 59.98% | 7.50% to 60.61% | 11.17% to 61.87% |
| 1 | 4096 | 16 | 1.16% | 2.93% | 6.17% | 7.34% | 10.21% | 0.10% to 1.78% | 1.72% to 3.02% | 5.73% to 6.69% | 6.77% to 7.57% | 9.41% to 15.02% |
| 1 | 8192 | 8 | 1.38% | 3.78% | 6.76% | 8.43% | 12.72% | 1.11% to 1.62% | 3.39% to 3.90% | 6.71% to 7.11% | 8.27% to 8.58% | 12.53% to 13.78% |
| 1 | 8192 | 16 | 1.31% | 3.67% | 6.13% | 8.27% | 13.00% | 1.08% to 1.49% | 3.40% to 4.15% | 5.31% to 6.77% | 7.97% to 8.61% | 12.75% to 13.00% |
| 1 | 16384 | 8 | 1.40% | 2.90% | 6.80% | 7.96% | 13.43% | 0.89% to 1.83% | 2.46% to 3.57% | 6.54% to 7.03% | 7.57% to 8.32% | 13.15% to 13.63% |
| 16 | 4096 | 8 | 0.55% | 1.72% | 5.49% | 6.85% | 10.65% | 0.42% to 0.72% | 1.55% to 1.98% | 5.32% to 5.51% | 6.66% to 6.90% | 10.40% to 10.74% |
| 16 | 4096 | 16 | 0.82% | 3.43% | 6.19% | 7.63% | 10.52% | 0.62% to 0.99% | 3.30% to 3.59% | 5.78% to 6.35% | 7.39% to 7.71% | 10.26% to 10.69% |
| 16 | 8192 | 8 | — | — | — | — | — | — | — | — | — | — |

### 2. Ablation: Each Configuration vs. Production Dispatch

| B | S | Hkv | CFVST | CFVST + KDB | CFVST + VPF | CFVST + KDB + VPF | Full + LDS pad4 + causal split | CFVST min–max | CFVST + KDB min–max | CFVST + VPF min–max | CFVST + KDB + VPF min–max | Full + LDS pad4 + causal split min–max |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1024 | 8 | 8.74% | 11.92% | 15.57% | 15.96% | 18.98% | 6.51% to 11.12% | 8.96% to 12.98% | 14.32% to 16.58% | 14.02% to 18.54% | -405.47% to 20.04% |
| 1 | 1024 | 16 | 9.27% | 11.30% | 14.62% | 15.64% | 16.78% | 7.98% to 11.34% | 10.86% to 13.14% | 13.91% to 15.96% | 15.41% to 17.55% | 16.07% to 18.67% |
| 1 | 2048 | 8 | 10.94% | 13.24% | 15.70% | 17.90% | 20.99% | 10.42% to 11.35% | 12.24% to 14.11% | 15.42% to 16.29% | 17.32% to 18.37% | 19.95% to 21.62% |
| 1 | 2048 | 16 | 11.01% | 13.15% | 15.13% | 17.12% | 20.21% | 9.93% to 11.89% | 11.96% to 14.04% | 10.72% to 16.43% | 15.44% to 17.95% | 19.67% to 20.53% |
| 1 | 4096 | 8 | 9.52% | 11.61% | 13.75% | 15.43% | 18.76% | 8.74% to 10.01% | 11.20% to 12.31% | 12.92% to 14.96% | 15.02% to 18.43% | 18.42% to 19.10% |
| 1 | 4096 | 16 | 9.58% | 11.21% | 14.17% | 15.24% | 17.86% | 9.16% to 9.73% | 9.84% to 12.31% | 13.34% to 14.82% | 14.50% to 16.03% | 17.31% to 21.74% |
| 1 | 8192 | 8 | 10.12% | 12.30% | 15.02% | 16.54% | 20.46% | 9.76% to 10.33% | 11.84% to 12.48% | 14.94% to 15.27% | 16.35% to 16.64% | 20.24% to 21.35% |
| 1 | 8192 | 16 | 9.89% | 12.04% | 14.28% | 16.24% | 20.56% | 9.65% to 9.98% | 11.83% to 12.46% | 13.52% to 14.93% | 15.87% to 16.66% | 20.32% to 20.66% |
| 1 | 16384 | 8 | 10.41% | 11.77% | 15.31% | 16.37% | 21.34% | 9.89% to 10.69% | 11.32% to 12.25% | 15.03% to 15.49% | 15.97% to 16.58% | 21.11% to 21.41% |
| 16 | 4096 | 8 | 9.36% | 10.42% | 13.85% | 15.10% | 18.56% | 9.36% to 9.50% | 10.21% to 10.67% | 13.65% to 14.06% | 14.96% to 15.26% | 18.37% to 18.79% |
| 16 | 4096 | 16 | 8.87% | 11.27% | 13.80% | 15.12% | 17.78% | 8.77% to 9.07% | 11.17% to 11.45% | 13.64% to 14.01% | 15.03% to 15.37% | 17.69% to 17.84% |
| 16 | 8192 | 8 | 10.67% | 12.46% | 15.70% | 17.19% | 20.53% | 10.48% to 10.69% | 12.34% to 12.69% | 15.55% to 15.73% | 17.01% to 17.35% | 19.97% to 20.64% |

### 3. Final Pipeline: Comparison of Both Baselines

| B | S | Hkv | Step-0 vs. Production | Full Pipeline vs. Step-0 | Full Pipeline vs. Production | Step-0 vs. Production min–max | Full Pipeline vs. Step-0 min–max | Full Pipeline vs. Production min–max |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1024 | 8 | — | — | 18.98% | — | — | -405.47% to 20.04% |
| 1 | 1024 | 16 | — | — | 16.78% | — | — | 16.07% to 18.67% |
| 1 | 2048 | 8 | — | — | 20.99% | — | — | 19.95% to 21.62% |
| 1 | 2048 | 16 | — | — | 20.21% | — | — | 19.67% to 20.53% |
| 1 | 4096 | 8 | 8.26% | 11.44% | 18.76% | -113.87% to 8.93% | 11.17% to 61.87% | 18.42% to 19.10% |
| 1 | 4096 | 16 | 8.53% | 10.21% | 17.86% | 7.90% to 9.65% | 9.41% to 15.02% | 17.31% to 21.74% |
| 1 | 8192 | 8 | 8.86% | 12.72% | 20.46% | 8.75% to 8.93% | 12.53% to 13.78% | 20.24% to 21.35% |
| 1 | 8192 | 16 | 8.69% | 13.00% | 20.56% | 8.58% to 8.81% | 12.75% to 13.00% | 20.32% to 20.66% |
| 1 | 16384 | 8 | 9.13% | 13.43% | 21.34% | 9.00% to 9.35% | 13.15% to 13.63% | 21.11% to 21.41% |
| 16 | 4096 | 8 | 8.85% | 10.65% | 18.56% | 8.80% to 9.12% | 10.40% to 10.74% | 18.37% to 18.79% |
| 16 | 4096 | 16 | 8.12% | 10.52% | 17.78% | 7.94% to 8.35% | 10.26% to 10.69% | 17.69% to 17.84% |
| 16 | 8192 | 8 | — | — | 20.53% | — | — | 19.97% to 20.64% |

**Outliers retained:** in one round each, an arm ran 2.3–6.3× slower than its own median (B=1, S=1024, Hkv=8, Full + LDS pad4 + causal split; B=1, S=4096, Hkv=8, the Step-0 winner `A`), while the other arms in that round did not. The min–max ranges include those rounds as measured; the medians are unaffected.

## Results and Ablation Methodology

The benchmark compares each shape's **Step-0 existing-lever sweep winner (`A`)** against seven independently compiled configurations derived from the **current production dispatch**. It uses twelve persistent-304 BF16 D128 causal attention shapes. Seven have a Step-0 winner; B=1, S=1024/2048 (Hkv 8 and 16) and B=16, S=8192, Hkv=8 do not, so they are compared with production dispatch only.

For each of the twelve shapes (the five shapes without a Step-0 winner skip `A` and every comparison with it):

1. Start one **torch-free** rocKE/NumPy Python process with one ROCm stack, the Python compilation backend, and LLVM23. Refuse a dirty or untracked checkout. Record the GPU, the loaded HIP runtime/COMGR libraries, Git HEAD, and the SHA-256 of the kernel, spec, dispatch, builder, and harness sources.
2. Create one dedicated HIP stream using `rocke.runtime.hip_module` and use that stream for every kernel launch and timing event.
3. Build the shape's `AttentionRequest` and resolve the production specification using `attention_tuning_spec(req, "gfx942_dense").kernel_spec`. Require persistent dispatch (`persistent=True`, `num_persistent=304`); reject unsupported shapes.
4. Build the separate **`A` comparator** through `tuning_spec_with_knobs(req, "gfx942_dense", knobs)` from the recorded shape-specific Step-0 winning *existing* levers, and abort unless it compiles to the recorded winner kernel name. Keep `use_cfvst=False`, `lds_num_buffers=1`, and `v_reg_prefetch=False`. The previous winner is rebuilt using the **current checkout**, not timed using historical measurements.
5. Build seven **B-stage** specifications through `tuning_spec_with_knobs` from their knob deltas against the *current production-dispatch specification*, **not** from `A`. Preserve all non-ablation production knobs except the explicitly stated LDS-padding and causal-split overrides:

   | Arm | Variant | `use_cfvst` | `lds_num_buffers` (K) | `v_reg_prefetch` | `lds_row_pad` | `causal_diag_split` |
   |---:|---|---:|---:|---:|---|---|
   | A | `A` (Step-0 winner) | `False` | 1 | `False` | Shape-specific winner | Shape-specific winner |
   | B1 | `baseline` | `False` (pinned) | 1 | `False` | Production dispatch | Production dispatch |
   | B2 | `cfvst` | `True` (production policy since #14367) | 1 | `False` | Production dispatch | Production dispatch |
   | B3 | `cfvst_kdb` | `True` | 2 | `False` | Production dispatch | Production dispatch |
   | B4 | `cfvst_vpf` | `True` | 1 | `True` | Production dispatch | Production dispatch |
   | B5 | `pipeline` | `True` | 2 | `True` | Production dispatch | Production dispatch |
   | B6 | `pipeline_pad` | `True` | 2 | `True` | 4 | Production dispatch |
   | B7 | `pipeline_pad_dsplit` | `True` | 2 | `True` | 4 | `True` |

6. Assert that the A configuration disables the new pipeline and that production dispatch starts with the three new knobs disabled. Verify every successive B-stage change modifies exactly its advertised field(s), including resolved knob values. Record each arm's `spec_id`, `tuning_id`, canonical knobs, and dispatch-time `spec_hash`, replay each record through `dispatch_attention`, and abort on any `spec_hash` mismatch.
7. Check `supports_attention_dense` for every arm. Require distinct compile-time kernel names; retain the recorded and rebuilt Step-0 winner names for auditing.
8. Run the Python kernel emitter and compile **one independent HSACO per arm**; record their kernel names, code-object hashes, specifications, and static register, spill, LDS, and ISA instruction counts. No runtime GPU-kernel variant-selection branch is used.
9. Generate one shared BF16-rounded Q/K/V input set. Allocate a separate output buffer and construct a direct `KernelLauncher` plus `LaunchConfig` for each compiled kernel.
10. Compute one independent **chunked NumPy FP32 causal GQA SDPA** reference from the shared BF16-rounded inputs. No PyTorch import or `torch.testing.assert_close` is used.
11. **Before timing**, fill each output with `0xFF` bytes (NaN poison), invoke its kernel, synchronize, and compare **every output element** against the FP32 reference using `numpy.testing.assert_allclose(atol=1e-2, rtol=1e-2, equal_nan=False)`. Record maximum absolute and relative errors; abort on failure.
12. Perform **10 round-robin warmup launches per variant**, then synchronize the HIP stream.
13. Execute **five measurement rounds**, randomly shuffling the order of all arms within each round.
14. For each variant in each round, record a HIP start event, make **50 direct `KernelLauncher` calls**, record a HIP end event, synchronize that event, and divide elapsed GPU-event time by 50 to obtain per-launch latency.
15. After each timed batch, validate its final output against the same FP32 reference with the same tolerances, **outside the GPU-event timing interval**. Record errors and abort on failure. Validation checks the last output of each 50-launch batch, not all 50 intermediate outputs.
16. Save the shape, variant, measurement round, randomized execution order, configuration knobs, latency, and correctness result. Preserve the resolved specs, kernel names, HSACO hashes, and environment provenance.

**Correctness coverage:** Each of the seven shapes with a Step-0 winner has eight initial checks and `8 × 5 = 40` post-batch checks, or **48 checks per shape**. The five shapes without one run B1–B7 only, so each has `7 + 7 × 5 = 42`. Across the twelve shapes there are **546 checks** (91 initial and 455 post-batch).

17. Collect five per-round latency samples per variant and compute each variant's independent median:

    `T_variant = median(t_variant,1, ..., t_variant,5)`

18. Use the shape-specific **Step-0 winner (`A`) as the reference denominator** for the main tables. For each candidate B2, B3, B4, B5, or B7, compute its speedup as:

    `Speedup_A_to_candidate = T_A / T_candidate`

19. Compute the main percentage latency reduction **against `A`**, not against production dispatch:

    `Reduction_A_to_candidate (%) = 100 × (1 - T_candidate / T_A)`

20. For the five-column ablation table, report B2 (`CFVST`), B3 (`CFVST+KDB`), B4 (`CFVST+VPF`), B5 (`CFVST+KDB+VPF`), and B7 (`CFVST+KDB+VPF+LDS pad4+causal split`), **each compared directly with `A`**.
21. For the final full-optimization table, compare only **B7 versus `A`**. Calculate the percentage reduction for each matched round:

    `R_r = 100 × (1 - t_B7,r / t_A,r)`

22. Report the observed same-round range:

    `Min = min(R_1, R_2, R_3, R_4, R_5)`
    `Max = max(R_1, R_2, R_3, R_4, R_5)`

23. Report the central B7-vs-A reduction from the **ratio of the two medians**, not the median of the five round-level reductions.
24. Independently calculate *within-B-stage* comparisons to isolate pipeline effects: `baseline → cfvst`, `cfvst → cfvst_kdb`, `cfvst → cfvst_vpf`, `cfvst_kdb → pipeline`, `cfvst_vpf → pipeline`, `pipeline → pipeline_pad`, and `pipeline_pad → pipeline_pad_dsplit`.
25. Export `ablation_runs.csv`, `ablation_summary.csv`, `ablation_ratios.csv`, `ablation_comparison.csv`, `ablation_comparison.md`, `A_vs_B7.csv`, `validation.csv`, `static_resources.csv`, `missing_sweep_winners.json`, `resolved_specs.json`, and `provenance.json` for reproduction and review.
26. Run `--verify-provenance` on the output at the PR head: it re-checks the source hashes, refuses any non-Markdown change since the measured commit, and replays every arm's record.

**Interpretation:** `A` and B1–B7 are measured within the same session and compiled from the same current source, so direct **Step-0-winner-versus-optimized-production** comparisons are valid. However, the B stages **do not inherit `A`'s geometry or scheduling knobs**: the results do **not** quantify the incremental benefit of adding CFVST/KDB/VPF on top of the Step-0 winner itself. **Losses:** B=1, S=1024/2048 have no Step-0 winner in this sweep, so they are compared with production dispatch only. In the earlier twelve-shape Step-0 sweep at `c8980b69`, their winners were non-persistent configurations (`block_m=128`, `block_n=32`) that outperformed the CFVST/KDB/VPF pipeline, which runs on the persistent grid only. This is a loss at those shapes.

#### Benchmark Environment and Provenance

| Component | Configuration |
| :--- | :--- |
| Hardware target | gfx942, 304 CUs |
| Benchmarked source HEAD | `cc84e2621e8fd9ce83f9bf43dbc1564c690c290d` (clean working tree; `--verify-provenance` passed at that commit; predates the #14367 merge, see *Dated results*) |
| Kernel source SHA-256 | `8bf80337c529fdab6e3adf65d942bf423eac9623b324c6cd650d0987adb71d2e` |
| Benchmark harness SHA-256 | `888b484dab6a0dc76e5544de1660047e03956b10f9f898cf8cb3cb228ec61130` |
| ROCm stack | 10.0.0 |
| HIP | `7.15.26333` |
| Python | `3.12.3` |
| Triton | `3.8.0+git4cff872c.rocm10.0.0` |
| ROCKE compiler backend | Python |
| LLVM flavor | `llvm23` |

#### Shape-by-shape default kernel identity built by `attention_tuning_spec(req, "gfx942_dense").kernel_spec`

| dtype | B | S | Hq | Hkv | D | Exact baseline kernel name | persistent | exp2_fast (resolved) | decode (resolved) | grid | block |
|---|---:|---:|---:|---:|---:|---|---|---|---|---|---|
| bf16 | 1 | 1024 | 32 | 8 | 128 | `rocke_attention_dense_d128_bn64_bf16_causal_lazyrs_persist304_bthkvmin_gfx942_wpe2` | `true` | `true` | `bt_hkv_minor` | `[304, 1, 1]` | `[512, 1, 1]` |
| bf16 | 1 | 1024 | 32 | 16 | 128 | `rocke_attention_dense_d128_bn64_bf16_causal_lazyrs_persist304_bthkvmin_gfx942_wpe2` | `true` | `true` | `bt_hkv_minor` | `[304, 1, 1]` | `[512, 1, 1]` |
| bf16 | 1 | 2048 | 32 | 8 | 128 | `rocke_attention_dense_d128_bn64_bf16_causal_lazyrs_persist304_bthkvmin_gfx942_wpe2` | `true` | `true` | `bt_hkv_minor` | `[304, 1, 1]` | `[512, 1, 1]` |
| bf16 | 1 | 2048 | 32 | 16 | 128 | `rocke_attention_dense_d128_bn64_bf16_causal_lazyrs_persist304_bthkvmin_gfx942_wpe2` | `true` | `true` | `bt_hkv_minor` | `[304, 1, 1]` | `[512, 1, 1]` |
| bf16 | 1 | 4096 | 32 | 8 | 128 | `rocke_attention_dense_d128_bn64_bf16_causal_lazyrs_persist304_bthkvmin_gfx942_wpe2` | `true` | `true` | `bt_hkv_minor` | `[304, 1, 1]` | `[512, 1, 1]` |
| bf16 | 1 | 4096 | 32 | 16 | 128 | `rocke_attention_dense_d128_bn64_bf16_causal_lazyrs_persist304_bthkvmin_gfx942_wpe2` | `true` | `true` | `bt_hkv_minor` | `[304, 1, 1]` | `[512, 1, 1]` |
| bf16 | 1 | 8192 | 32 | 8 | 128 | `rocke_attention_dense_d128_bn64_bf16_causal_lazyrs_persist304_bthkvmin_gfx942_wpe2` | `true` | `true` | `bt_hkv_minor` | `[304, 1, 1]` | `[512, 1, 1]` |
| bf16 | 1 | 8192 | 32 | 16 | 128 | `rocke_attention_dense_d128_bn64_bf16_causal_lazyrs_persist304_bthkvmin_gfx942_wpe2` | `true` | `true` | `bt_hkv_minor` | `[304, 1, 1]` | `[512, 1, 1]` |
| bf16 | 1 | 16384 | 32 | 8 | 128 | `rocke_attention_dense_d128_bn64_bf16_causal_lazyrs_persist304_bthkvmin_gfx942_wpe2` | `true` | `true` | `bt_hkv_minor` | `[304, 1, 1]` | `[512, 1, 1]` |
| bf16 | 16 | 4096 | 32 | 8 | 128 | `rocke_attention_dense_d128_bn64_bf16_causal_lazyrs_persist304_hkvmaj_gfx942_wpe2` | `true` | `true` | `hkv_major` | `[304, 1, 1]` | `[512, 1, 1]` |
| bf16 | 16 | 4096 | 32 | 16 | 128 | `rocke_attention_dense_d128_bn64_bf16_causal_lazyrs_persist304_gfx942_wpe2` | `true` | `true` | `qb_major` | `[304, 1, 1]` | `[512, 1, 1]` |
| bf16 | 16 | 8192 | 32 | 8 | 128 | `rocke_attention_dense_d128_bn64_bf16_causal_lazyrs_persist304_hkvmaj_gfx942_wpe2` | `true` | `true` | `hkv_major` | `[304, 1, 1]` | `[512, 1, 1]` |

**Benchmark harness:** `library/benchmarks/gfx942/attention/prefill/bench_rocke_7way_specs.py`

**Kernel source:** `library/kernels/gfx942/attention_dense.py`

#### Reproducibility

Clone repo to gpu-bench/.

**`setup.sh`** — install the ROCm Python environment and rocKE:

```bash
#!/usr/bin/env bash
set -Eeuo pipefail

REPO="$(git rev-parse --show-toplevel)"
ROOT="${GPU_BENCH_ROOT:-/root/gpu-bench}"
ROCKE="$REPO/dnn-providers/hip-kernel-provider/rocke"
VENV="$ROOT/venvs"
ENV="$ROOT/env"

AMD_INDEX=https://stable.repo.amd.com/rocm/whl-next/

mkdir -p "$VENV" "$ENV"

# ----------------------------------------------------------------------
# Host dependencies
# ----------------------------------------------------------------------

apt-get update
apt-get install -y \
  git python3.12 python3.12-venv python3.12-dev python3-pip \
  cmake ninja-build build-essential pkg-config libnuma-dev

# ----------------------------------------------------------------------
# ROCKE virtual environment
# ----------------------------------------------------------------------

ROCKE_VENV="$VENV/rocke"

if [ ! -x "$ROCKE_VENV/bin/python" ] || \
   ! "$ROCKE_VENV/bin/python" -m pip --version >/dev/null 2>&1; then
  rm -rf "$ROCKE_VENV"
  python3.12 -m venv "$ROCKE_VENV"
fi

"$ROCKE_VENV/bin/python" -m pip install -U \
  pip setuptools wheel packaging ninja psutil

"$ROCKE_VENV/bin/python" -m pip install \
  --index-url "$AMD_INDEX" \
  'rocm[devel,device-gfx942,libraries]==10.0.0' \
  'torch[device-gfx942]==2.13.0+rocm10.0.0'

"$ROCKE_VENV/bin/rocm-sdk" init >/dev/null

# ----------------------------------------------------------------------
# Install ROCKE
# ----------------------------------------------------------------------

"$ROCKE_VENV/bin/python" -m pip install -e "$ROCKE/platform"
"$ROCKE_VENV/bin/python" -m pip install -e "$ROCKE/library" --no-deps

# ----------------------------------------------------------------------
# ROCKE environment activation script
# ----------------------------------------------------------------------

cat > "$ENV/rocke_env.sh" <<EOF
source "$ROCKE_VENV/bin/activate"

rocm-sdk init >/dev/null

export ROCM_HOME="\$(rocm-sdk path --root)"
export ROCM_PATH="\$ROCM_HOME"
export HIP_PATH="\$ROCM_HOME"
export PATH="\$(rocm-sdk path --bin):\$PATH"

export HIP_PLATFORM=amd
export ROCKE_BACKEND=python
export ROCKE_LLVM_FLAVOR=llvm23

LIB="\$(python -c 'import importlib.util, pathlib; s=importlib.util.find_spec("_rocm_sdk_core"); print(pathlib.Path(list(s.submodule_search_locations)[0]) / "lib")')"

export ROCKE_COMGR_LIB="\$LIB/libamd_comgr.so.3"
export ROCKE_HIP_LIB="\$LIB/libamdhip64.so.7"
export LD_LIBRARY_PATH="\$LIB:\${LD_LIBRARY_PATH:-}"

unset LIB
EOF

bash -n "$ENV/rocke_env.sh"

echo
echo "ROCKE setup complete."
echo "ROCKE env: source $ENV/rocke_env.sh"
```

Run the setup script from its directory:

```bash
chmod +x setup.sh
./setup.sh
```

Run the 7-way benchmark on a committed, clean checkout (a dirty tree is refused), then verify the run's provenance:

```bash
ROOT="${GPU_BENCH_ROOT:-/root/gpu-bench}"
source "$ROOT/env/rocke_env.sh"
cd "$(git rev-parse --show-toplevel)/dnn-providers/hip-kernel-provider/rocke"

BENCH=library/benchmarks/gfx942/attention/prefill/bench_rocke_7way_specs.py
python "$BENCH" --warmup 10 --rounds 5 --repeat 50 --atol 1e-2 --rtol 1e-2 \
  --out-dir "$ROOT/rocke_7way_specs_results"
python "$BENCH" --verify-provenance "$ROOT/rocke_7way_specs_results"
```

Results are written outside the repository, to the `--out-dir` directory.

### Global-Memory Coalescing and ISA Verification (§R.5)

Verified on **gfx942**, BF16 D128 causal attention, BM256/BN64, persistent-304, LLVM23, B=1, S=4096, Hkv=8, on the HSACOs timed in the benchmark run.

| Global access | Modeled positions | Coalesced | Cache lines | Footprint | ISA |
|---|---:|---|---:|---:|---|
| K async → LDS | 64 | Yes | 2 | 1.0× | `buffer_load_dword ... lds` (32-bit) |
| V CFVST x0 | 32 | Yes | 2 | 1.0× | `buffer_load_dword` (32-bit) |
| V CFVST x1 | 32 | Yes | 2 | 1.0× | `buffer_load_dword` (32-bit) |

`analyze_coalescing` (with `vector_transactions` and `addr_fn_from_strides`, `line_bytes_for("gfx942") = 128`) reports for every access: `load [token stride 1024, head_dim stride 1]: contiguous axis = head_dim; 1 instr, VW=2 elems (b128-ideal), lines/instr 2..2 (128B lines), footprint 1x, FULLY COALESCED`. **`assert_asm_backed` passed** for K in all seven B-arm HSACOs and for V x0/x1 in the six CFVST arms; no wider or narrower K/V global load is emitted.

x0 and x1 are separate loads. In the KDB/VPF pipeline they share one VGPR lane offset and differ only in their SGPR `soffset`, one runtime token stride apart.

**Result:** No modeled global-memory coalescing regression in the verified configurations. Evidence covers the tested K/V access paths, not every tuning configuration.

## Runbook and Architecture Compliance

The CFVST, K LDS double-buffering (KDB), and V register-prefetching (VPF) optimizations follow the CK DSL Optimization Runbook (`platform/dsl_docs/optimization/optimization_runbook.md`) and the gfx942-specific architecture constraints (`platform/dsl_docs/optimization/arch/gfx942.md`, §21).

### Per-Lever Mapping

| Lever | Implementation | Runbook sections | §13.3 Attention checklist | gfx942 architecture reference |
|---|---|---|---|---|
| `use_cfvst=True` | `perm_b32` store-path transpose into transposed V LDS, with optional XOR swizzling | §6.3 LDS, §6.4 bank conflicts, §6.5 registers, §7.2 operand layouts | Tile Q / K / V; shared memory capacity; register pressure from accumulators | §21.2 LDS geometry and banks; §21.3 `v_perm_b32` cross-lane primitive; §21.4 VGPR constraints; §21.8 transpose-read limitations |
| `lds_num_buffers=2` | Double-buffer K in LDS to overlap next-tile K loading with current-tile computation; V LDS remains single-buffered | §6.3 LDS, §8.1 software pipelining, §8.2 async copy, §8.3 waits and barriers | Tile Q / K / V; shared memory capacity; register pressure from accumulators | §21.2 64 KiB LDS capacity; §21.4 occupancy constraints; §21.6 async LDS load width; §21.8 ISA constraints |
| `v_reg_prefetch=True` | Prefetch next-tile V into registers to overlap memory loading with current-tile computation | §6.5 registers, §8.1 software pipelining, §8.2 async copy, §8.3 waits and barriers | Tile Q / K / V; register pressure from accumulators | §21.4 VGPR pressure and occupancy; §21.6 global buffer-load widths; §21.8 ISA constraints |

### Architecture and Configuration Gating

The optimizations are implemented in `library/kernels/gfx942/attention_dense.py`, using `Gfx942AttentionDenseSpec`.

Architecture eligibility is enforced by the existing `arch != "gfx942"` checks in `supports_attention_dense(spec, arch=...)` and `build_attention_dense(spec, arch=...)`. The support predicate returns `(False, reason)` for other architectures, while the builder raises `NotImplementedError`. Dispatch also restricts the gfx942 dense candidate to gfx942 through its architecture-specific registration and spec factory.

The optimization-specific eligibility conditions are enforced by `supports_attention_dense()`:

- **CFVST (`use_cfvst=True`):** For BF16, requires D128 causal attention without sliding window or attention sinks, on either grid. Causal diagonal splitting is supported. Existing FP16 CFVST behavior is preserved.
- **KDB (`lds_num_buffers=2`):** Requires BF16 D128 causal attention, CFVST enabled, `persistent=True`, `block_n=64`, no sliding window, and no attention sinks.
- **VPF (`v_reg_prefetch=True`):** Has the same eligibility conditions as KDB but is independently controlled. Enabling VPF does not require enabling KDB.

The combined K/V LDS allocation is checked against the gfx942 capacity using `attention_lds_capacity_bytes(arch)` and `_lds_bytes(spec)`. Unsupported configurations are rejected before kernel emission.

For BF16, the default configuration retains `use_cfvst=None` (policy resolves to on at D128), `lds_num_buffers=1`, and `v_reg_prefetch=False`. K double buffering and V register prefetch remain opt-in and do not change the default kernel behavior.

## gfx942 Barrier / LGKM ISA Evidence

**Kernel:** `rocke_attention_dense_d128_bn64_bf16_causal_lazyrs_persist304_bthkvmin_gfx942_wpe2_kdb1_vpf1_krowpad4_vrowpad0_cfvst1_dsplit`
**Workload:** BF16, B=1, Sq=Skv=4096, Hq=32, Hkv=8, D=128; gfx942, 304 CUs.
**Artifact:** `rocke_7way_specs_results/hsaco/b1_s4096_hq32_hkv8_d128_bf16/pipeline_pad_dsplit.hsaco`
**Evidence:** complete `.text` disassembly generated with `llvm-objdump -d` (not just emitter/IR), from the HSACO timed in the benchmark run at HEAD `cc84e2621e8fd9ce83f9bf43dbc1564c690c290d`.

### Barrier inventory

All **seven** `s_barrier` sites in this code object are accounted for:

| Region | Last relevant LDS op | `s_waitcnt` | `s_barrier` | Finding |
| :--- | :--- | :--- | :--- | :--- |
| Pre-output / reduction | `ds_read2st64_b64` @ `0x2CF0` | `lgkmcnt(0)` @ `0x2D18` | `0x2D1C` | LDS drained |
| Pre-loop / initialization entry | No LDS ops since earlier drain on either entry path | `vmcnt(0)` @ `0x3960` | `0x3964` | No adjacent `lgkmcnt(0)`; see exception below |
| Initial LDS staging | `ds_write_b32` @ `0x3BA0` | `lgkmcnt(0)` @ `0x3BA8` | `0x3BAC` | LDS writes drained |
| Main-loop barrier 1 | `ds_read2st64_b64` @ `0x49F4` | `lgkmcnt(0)` @ `0x49FC` | `0x4A00` | LDS reads drained |
| Main-loop barrier 2 | `ds_write_b32` @ `0x4A78` | `lgkmcnt(0)` @ `0x4A8C` | `0x4A90` | LDS writes drained |
| Alternate-loop barrier 1 | `ds_read2st64_b64` @ `0x59DC` | `lgkmcnt(0)` @ `0x59E4` | `0x59E8` | LDS reads drained |
| Alternate-loop barrier 2 | `ds_write_b32` @ `0x5A60` | `lgkmcnt(0)` @ `0x5A70` | `0x5A74` | LDS writes drained |

### Key `llvm-objdump` excerpts

```asm
# Main loop: read-side barrier
0x49F4: ds_read2st64_b64 v[184:187], v132 offset0:82 offset1:90
0x49FC: s_waitcnt lgkmcnt(0)
0x4A00: s_barrier
0x4A04: s_waitcnt vmcnt(0)
0x4A08: ds_write_b32 v117, v2 offset:33792

# Main loop: write-side barrier
0x4A70: ds_write_b32 v123, v2 offset:33792
0x4A78: ds_write_b32 v124, v3 offset:33920
0x4A8C: s_waitcnt lgkmcnt(0)
0x4A90: s_barrier

# Alternate loop: same synchronization sequence
0x59DC: ds_read2st64_b64 v[192:195], v132 offset0:82 offset1:90
0x59E4: s_waitcnt lgkmcnt(0)
0x59E8: s_barrier
0x59EC: s_waitcnt vmcnt(0)
0x59F0: ds_write_b32 v117, v2 offset:33792
0x5A60: ds_write_b32 v124, v4 offset:33920
0x5A70: s_waitcnt lgkmcnt(0)
0x5A74: s_barrier

# Pre-output/reduction synchronization
0x2CF0: ds_read2st64_b64 v[170:173], v132 offset0:82 offset1:90
0x2D18: s_waitcnt lgkmcnt(0)
0x2D1C: s_barrier
0x2D48: global_store_dwordx2 v[6:7], v[4:5], off

# Initialization LDS staging
0x3BA0: ds_write_b32 v124, v2 offset:33920
0x3BA8: s_waitcnt lgkmcnt(0)
0x3BAC: s_barrier
```

**Exception — `0x3964`:**

```asm
0x3960: s_waitcnt vmcnt(0)
0x3964: s_barrier
```

There is **no immediately adjacent `lgkmcnt(0)`** at `0x3964`. The two paths visible in this disassembly reach it after a prior `lgkmcnt(0)` at `0x1BE4` (prologue path branching at `0x1EFC` to `0x38D0`) or `0x2D18` (post-output path); neither intervening instruction region contains another `ds_*` operation. This supports an already-drained LGKM state on those paths.
