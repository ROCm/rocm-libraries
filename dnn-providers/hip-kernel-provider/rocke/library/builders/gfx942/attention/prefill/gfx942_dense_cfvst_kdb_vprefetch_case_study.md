# gfx942 bf16 D128 causal prefill: CFVST, K LDS double buffering, and V register prefetch

Measured scope: the opt-in `use_cfvst=True`, `lds_num_buffers=2` configuration of the
gfx942 dense prefill attention kernel, bf16, D128, causal, on gfx942 with 304 CUs.
Baseline: the `baseline` arm of the ablation harness, compiled in the same process from
the benchmarked source commit (see
[Benchmark Environment and Provenance](#benchmark-environment-and-provenance)). Not yet
recorded: the baseline arm's full spec (in particular `persistent` and `exp2_fast`, and
whether it was hand-built or resolved through dispatch) and the tolerance of the
pre-timing correctness check against the chunked FP32 reference. Result: 10.1-16.3% lower
latency at the 12 measured configurations listed under Results.

## Introduction

This optimization enables CFVST and implements K LDS Double Buffering and V register pre-fetching on D128 bf16 causal prefill attention gfx942 kernels. At the 12 measured configurations it lowers latency by 10.1-16.3%. Lever sweep does not bridge the gap.

## Problem

Benchmarking, profiling and tracing highlighted bottlenecks in:
1. V read from LDS: V was stored in traditional layout in LDS. Bank conflicts occur during access.
2. Awkward allocation of LDS per WG (33KB). Max limit per CU is 64KB. This limits WG occupancy in CU to 1 while leaving 31KB of LDS unused.
3. Stalls at sync point for loading K and V tiles before MFMA.

The concerned optimization points were V LDS layout, register usage, LDS usage, software pipelining. We first tried to solve the issues by lever sweep. We selected levers relevant to the optimization points.

## Step-0 Lever Sweep

Lever sweep was performed to check whether performance could be improved by tuning levers to affect optimization points: **LDS usage/layout, software pipelining/scheduling, and register pressure**. Available levers are listed below along with reasons for selection decision and values swept.

| Field            | Swept | Relation to LDS / software pipelining / register usage                                                                                                                         | Values swept                               |
|------------------|-------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------|
| `block_m`        | Yes   | Changes workgroup/tile size and therefore affects LDS footprint, per-wave live state, register pressure, and scheduling granularity.                                           | `32, 64, 128, 256, 512`                    |
| `block_n`        | Yes   | Changes the K/V tile width, affecting LDS usage, loop trip count, load/compute cadence, and live register state.                                                               | `32, 64, 128`                              |
| `waves_per_eu`   | Yes   | Controls occupancy/scheduling pressure and helps test whether register pressure or insufficient latency hiding is limiting performance.                                        | `1, 2, 3, 4, 5, 6, 7, 8`                   |
| `lds_row_pad`    | Yes   | Directly changes the LDS layout/padding and can affect LDS bank behavior and address layout.                                                                                   | `0, 4, 8, 12, 16, 20, 24, 28, 32`          |
| `use_cfvst`      | Yes   | Changes the V-store/LDS path and therefore directly affects LDS traffic/layout; it can also change register lifetimes.                                                         | `False, True`                              |
| `use_v_swizzle`  | Yes   | Changes the V LDS layout/access pattern on the CFVST path and is directly relevant to LDS access behavior.                                                                     | `False`; `True` only when CFVST is enabled |
| `use_exp2_fast`  | Yes   | Primarily changes softmax computation, but can indirectly affect instruction count and temporary register usage.                                                               | `False, True`                              |
| `iglp`           | Yes   | Affects instruction scheduling inside the kernel by attempting to interleave MFMA and LDS operations. Relevant when CFVST introduces in-loop LDS traffic.                      | `False`; `True` only when CFVST is enabled |
| `persistent`     | No    | Controls whether the persistent grid is used. Affects grid-level scheduling/utilization, not the per-workgroup K/V LDS layout, register buffering, or software pipeline.      | —                                          |
| `num_persistent` | No    | Controls the number of persistent workgroups. Can affect device-level utilization and work distribution, but does not modify the inner K/V LDS/register software pipeline.    | —                                          |
| `persist_decode` | No    | Controls how persistent work items map to query blocks, heads, and batches. It affects grid-level work distribution rather than the inner LDS/register pipeline.              | —                                          |
| `interleave`     | No    | Controls boustrophedon query-block ordering for persistent causal workloads. It changes CTA work distribution, not instruction interleaving inside the K/V software pipeline. | —                                          |

The sweep therefore varied the following eight axes:

`block_m × block_n × waves_per_eu × lds_row_pad × use_cfvst × use_v_swizzle × use_exp2_fast × iglp`

The default domains were defined directly in the sweep harness, with applicability pruning for `use_v_swizzle` and `iglp` when CFVST was disabled.

`persistent`, `num_persistent`, `persist_decode`, and `interleave` were retained from the production-resolved configuration and recorded in the output for reproducibility, but were not varied by the Step-0 Cartesian search. These fields primarily control persistent-grid execution and work distribution rather than the inner LDS/register software pipeline targeted by this investigation.

#### Shapes Swept
The Step-0 lever sweep was run over the following BF16 D128 causal attention shapes:

| B  | S     | Hq | Hkv | D   |
|----|-------|----|-----|-----|
| 1  | 4096  | 32 | 8   | 128 |
| 1  | 4096  | 32 | 16  | 128 |
| 1  | 8192  | 32 | 8   | 128 |
| 1  | 8192  | 32 | 16  | 128 |
| 1  | 16384 | 32 | 8   | 128 |
| 16 | 4096  | 32 | 8   | 128 |
| 16 | 4096  | 32 | 16  | 128 |
| 16 | 8192  | 32 | 8   | 128 |

#### Results
None of the configurations improved the performance satisfactorily.

## Optimization 1: Enable CFVST

V was stored in traditional layout in LDS. This caused bank conflicts during access. The hypothesis was that transposed layout would reduce bank conflicts. This feature was already present (CFVST) but disabled for bf16 due to register spillage concerns. Performance improved substantially with it enabled, so we adopted this optimization.

#### Register usage

VGPR count, spill count, and scratch bytes per thread, read from the code-object
notes (`.vgpr_count`, `.vgpr_spill_count`, `.private_segment_fixed_size`) for gfx942,
B1 S4096 Hq32 Hkv8 D128 causal, at two pinned toolchains: the `llvm20` IR flavor
compiled by ROCm 7.1 comgr, and the `llvm23` IR flavor compiled by the ROCm 10 SDK comgr
(AMD clang 23.0.0git). The `waves_per_eu=2` cap is 256 VGPRs.

| variant | llvm20 VGPR | llvm20 spills | llvm20 scratch (B) | llvm23 VGPR | llvm23 spills | llvm23 scratch (B) |
|---|---:|---:|---:|---:|---:|---:|
| bf16 default (CFVST off) | 243 | 0 | 0 | 219 | 0 | 0 |
| bf16 `use_cfvst=True` | 256 | 0 | 0 | 234 | 0 | 0 |
| + `lds_num_buffers=2` | 256 | 4 | 20 | 230 | 0 | 0 |
| + `emit_lse=True` | 256 | 5 | 24 | 230 | 0 | 0 |
| + `persistent=True`, `num_persistent=304` | 256 | 16 | 68 | 256 | 7 | 32 |

CFVST alone does not spill at either toolchain; at `llvm20` it sits exactly at the
256-VGPR cap. The K double buffer and V prefetch pipeline spills at `llvm20`, and at
`llvm23` only on the persistent grid. In every spilling variant the inner KV loop
contains no scratch instructions: the spill reloads sit outside it, and on the
persistent grid they run once per work item (16 at `llvm20`, 7 at `llvm23`). The
`llvm22` flavor was not measured.

## Optimization 2 and 3: K LDS Double Buffering + V register pre-fetching

Analysis of stalls using threadtrace found a large number of waiting at sync points. One such point was waiting for K and V tiles to be loaded before MFMA. This wait restricted MFMA and memory loading parallelism. The hypothesis was that K LDS Double Buffering + V register pre-fetching would introduce parallelism and latency hiding. Both optimizations had to be implemented together because K and V load happens in parallel to each other. Optimizing 1 would leave other the bottleneck. The optimization improved performance substantially hence it was adopted.

#### Results

**Hardware:** gfx942, 304 CUs

| dtype | B  | S     | Hq | Hkv | D   | baseline vs new | latency reduction |
|------:|---:|------:|---:|----:|----:|----------------:|------------------:|
| bf16  | 1  | 1024  | 32 | 8   | 128 | 1.1342x         | 11.83%            |
| bf16  | 1  | 1024  | 32 | 16  | 128 | 1.1406x         | 12.32%            |
| bf16  | 1  | 2048  | 32 | 8   | 128 | 1.1150x         | 10.31%            |
| bf16  | 1  | 2048  | 32 | 16  | 128 | 1.1125x         | 10.11%            |
| bf16  | 1  | 4096  | 32 | 8   | 128 | 1.1787x         | 15.16%            |
| bf16  | 1  | 4096  | 32 | 16  | 128 | 1.1771x         | 15.04%            |
| bf16  | 1  | 8192  | 32 | 8   | 128 | 1.1940x         | 16.25%            |
| bf16  | 1  | 8192  | 32 | 16  | 128 | 1.1827x         | 15.45%            |
| bf16  | 1  | 16384 | 32 | 8   | 128 | 1.1832x         | 15.48%            |
| bf16  | 16 | 4096  | 32 | 8   | 128 | 1.1579x         | 13.63%            |
| bf16  | 16 | 4096  | 32 | 16  | 128 | 1.1347x         | 11.87%            |
| bf16  | 16 | 8192  | 32 | 8   | 128 | 1.1638x         | 14.07%            |

#### Ablation studies of optimizations
We performed ablation study to isolate the optimizations and understand its individual contributions. Results of 1st shape is provided.

| Variant | Incremental effect |
|---|---:|
| CFVST  | 10.35% lower latency vs baseline |
| CFVST + KDB | 3.22% lower latency vs CFVST |
| CFVST + VPF  | 2.46% lower latency vs CFVST |
| CFVST + KDB + VPF | 5.94% lower latency vs CFVST |
| full vs baseline | 15.67% lower latency |

## Results and Ablation Methodology
The complete experiment can be summarized as follows:

```text
1. Start one ROCKE/PyTorch process using one ROCm stack.
2. Obtain one current HIP stream.
3. Construct five compile-time specs:
      baseline
      cfvst
      cfvst_klds
      cfvst_vpf
      cfvst_klds_vpf
4. Run the Python emitter five times.
5. Compile five independent KernelDefs/LLVM modules/HSACOs. No runtime variant-selection branch exists in the GPU kernels.
6. Verify all five kernel names are unique.
7. Allocate one shared Q/K/V input set.
8. Construct one direct KernelLauncher per compiled kernel.
9. Build one chunked FP32 reference.
10. Validate all five kernels against that same reference.
11. Abort if correctness fails.
12. Warm all five kernels round-robin.
13. Synchronize.
14. Perform five measurement rounds.
15. Randomize the order of the five kernels in every round.
16. For each kernel in that round:
       record start event
       issue 50 direct KernelLauncher calls
       record end event
       synchronize end event
       divide elapsed GPU time by 50
17. Collect five round-level samples per kernel.
18. Take the median of those samples.
19. Compute same-session pairwise A/B ratios:
       baseline / cfvst
       cfvst / klds
       cfvst / vpf
       klds / full
       vpf / full
       baseline / full

20. Use those ratios to attribute the contribution of CFVST,
    KDB, VPF, and their interaction.
```
To obtain results on 8 shapes, we run the ablation on 8 shapes and pick necessary fields.

#### Benchmark Environment and Provenance

| Component | Configuration |
|---|---|
| Hardware target | gfx942, 304 CUs |
| Benchmarked source commit | `39416d4e229e4c70daf52f4902cd0cc8df72751d` |
| ROCm wheel stack | 10.0.0 |
| HIP version | 7.15.26333 |
| Python | 3.12.3 |
| PyTorch | 2.13.0+rocm10.0.0 |
| Triton | 3.8.0+git4cff872c.rocm10.0.0 |

#### Test configuration

The default benchmark configuration used by the harness is:

```text
B       = 1
S_q     = 4096
S_kv    = 4096
H_q     = 32
H_kv    = 8
D       = 128
dtype   = BF16
causal  = true

block_m = 256
block_n = 64
```
---
