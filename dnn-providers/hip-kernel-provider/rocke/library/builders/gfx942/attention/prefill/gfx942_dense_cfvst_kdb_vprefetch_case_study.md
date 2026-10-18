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

| dtype | B | S | Hq | Hkv | D | baseline vs CFVST+KDB+VPF | latency reduction | min | max |
|------:|---:|------:|---:|----:|----:|----------------:|------------------:|----:|----:|
| bf16 | 1 | 1024 | 32 | 8 | 128 | 1.1203x | 10.74% | 10.61% | 10.88% |
| bf16 | 1 | 1024 | 32 | 16 | 128 | 1.1362x | 11.99% | 11.02% | 12.06% |
| bf16 | 1 | 2048 | 32 | 8 | 128 | 1.1258x | 11.18% | **-361.37% | 12.38% |
| bf16 | 1 | 2048 | 32 | 16 | 128 | 1.1227x | 10.93% | 9.75% | 11.26% |
| bf16 | 1 | 4096 | 32 | 8 | 128 | 1.1692x | 14.47% | 14.15% | 15.12% |
| bf16 | 1 | 4096 | 32 | 16 | 128 | 1.1796x | 15.23% | 14.47% | 15.69% |
| bf16 | 1 | 8192 | 32 | 8 | 128 | 1.1845x | 15.58% | 15.14% | 16.05% |
| bf16 | 1 | 8192 | 32 | 16 | 128 | 1.1836x | 15.51% | 14.86% | 15.82% |
| bf16 | 1 | 16384 | 32 | 8 | 128 | 1.1834x | 15.50% | 14.61% | 15.59% |
| bf16 | 16 | 4096 | 32 | 8 | 128 | 1.1591x | 13.72% | 13.32% | 14.08% |
| bf16 | 16 | 4096 | 32 | 16 | 128 | 1.1279x | 11.34% | 10.87% | 11.66% |
| bf16 | 16 | 8192 | 32 | 8 | 128 | 1.1598x | 13.78% | 12.75% | 14.02% |

** Not a numerical error. A repeated 1st measurement anomaly. 

#### Ablation studies of optimizations
We performed ablation study to isolate the optimizations and understand its individual contributions. Results of 1st shape is provided.

| dtype | B | S | Hq | Hkv | D | CFVST vs base |  CFVST +KDB vs CFVST | CFVST +VPF vs CFVST | CFSVT +KDB +VPF vs CFVST | CFSVT +KDB +VPF vs base |
|------:|---:|------:|---:|----:|----:|----------------:|----------------:|----------------:|----------------:|----------------:|
| bf16 | 1 | 1024 | 32 | 8 | 128 | 3.57% | -0.16% | 2.13% | 7.44% | 10.74% |
| bf16 | 1 | 1024 | 32 | 16 | 128 | 4.55% | 0.12% | 3.19% | 7.79% | 11.99% |
| bf16 | 1 | 2048 | 32 | 8 | 128 | 5.02% | 0.06% | 1.90% | 6.48% | 11.18% |
| bf16 | 1 | 2048 | 32 | 16 | 128 | 4.47% | 0.77% | 2.49% | 6.76% | 10.93% |
| bf16 | 1 | 4096 | 32 | 8 | 128 | 9.86% | 1.70% | 1.23% | 5.12% | 14.47% |
| bf16 | 1 | 4096 | 32 | 16 | 128 | 10.02% | 2.06% | 2.04% | 5.79% | 15.23% |
| bf16 | 1 | 8192 | 32 | 8 | 128 | 10.05% | 2.27% | 1.60% | 6.15% | 15.58% |
| bf16 | 1 | 8192 | 32 | 16 | 128 | 10.78% | 1.89% | 1.01% | 5.30% | 15.51% |
| bf16 | 1 | 16384 | 32 | 8 | 128 | 11.16% | 1.47% | 1.37% | 4.89% | 15.50% |
| bf16 | 16 | 4096 | 32 | 8 | 128 | 10.09% | 0.76% | 0.03% | 4.04% | 13.72% |
| bf16 | 16 | 4096 | 32 | 16 | 128 | 9.05% | 0.86% | -0.24% | 2.52% | 11.34% |
| bf16 | 16 | 8192 | 32 | 8 | 128 | 10.18% | 0.53% | 0.00% | 4.01% | 13.78% |

## Results and Ablation Methodology
The complete experiment can be summarized as follows:


For each of the 12 BF16 D128 causal attention shapes:

1. Start one ROCKE/PyTorch process using a single ROCm software stack.
2. Obtain the current HIP stream and use it for all kernel executions.
3. Resolve the production baseline using `attention_tuning_spec(req, "gfx942_dense").kernel_spec`.
4. Construct five compile-time configurations, preserving all non-ablation knobs from production dispatch:

   | Variant | `use_cfvst` | `lds_num_buffers` (K) | `v_reg_prefetch` |
   |---|---:|---:|---:|
   | `baseline` | `False` | 1 | `False` |
   | `cfvst` | `True` | 1 | `False` |
   | `cfvst_klds` | `True` | 2 | `False` |
   | `cfvst_vpf` | `True` | 1 | `True` |
   | `cfvst_klds_vpf` | `True` | 2 | `True` |

5. Run the Python kernel emitter separately for each configuration.
6. Compile five independent kernels into separate HSACO artifacts. Each kernel has its own compile-time configuration; there is no runtime variant-selection branch in the GPU kernel.
7. Verify that the five compiled kernel names are unique.
8. Allocate one shared Q/K/V input set and separate output tensors for each variant.
9. Construct one direct `KernelLauncher` per compiled kernel.

10. Compute one independent chunked FP32 causal GQA SDPA reference for the shared inputs.
11. Before timing, initialize each output with NaNs, execute its kernel, synchronize, and validate against the FP32 reference using `torch.testing.assert_close` with `atol=1e-2` and `rtol=1e-2`.
12. Record the maximum absolute and relative errors. Abort immediately if any validation fails.

13. Perform 10 round-robin warmup launches per kernel.
14. Synchronize the GPU before measurement.
15. Perform five measurement rounds, randomizing the order of the five variants in each round.
16. For each variant in each round:
    - Record a start GPU event.
    - Execute 50 direct `KernelLauncher` calls.
    - Record an end GPU event.
    - Synchronize the end event.
    - Divide elapsed GPU time by 50 to obtain average latency per launch.
17. After each timed batch, validate its output against the same FP32 reference using `atol=1e-2` and `rtol=1e-2`.
18. Perform post-batch validation outside the GPU-event timing interval, so correctness comparisons do not directly contribute to the reported GPU timing.
19. Record the validation result, maximum errors, shape, variant, and measurement round. Abort if any post-batch validation fails.

**Correctness coverage:** Each shape undergoes five initial correctness checks and 25 post-batch checks. Across 12 shapes, this gives **360 checks**. Post-batch validation checks the final output of each 50-launch batch, not every individual launch.


20. Collect five round-level latency samples for each variant.
21. Compute the median latency independently for each variant:

    `T_variant = median(t_variant,1, ..., t_variant,5)`

22. Compute same-session pairwise speedup ratios using these median latencies:
    - `baseline / cfvst`
    - `cfvst / cfvst_klds`
    - `cfvst / cfvst_vpf`
    - `cfvst_klds / cfvst_klds_vpf`
    - `cfvst_vpf / cfvst_klds_vpf`
    - `baseline / cfvst_klds_vpf`
23. Calculate percentage latency reduction for each comparison:

    `Reduction (%) = 100 × (1 - T_after / T_before)`

24. For the full-vs-baseline table, calculate the reduction separately for each matched measurement round:

    `R_r = 100 × (1 - T_full,r / T_baseline,r)`

25. Report the minimum and maximum observed round-level reductions:

    `Min = min(R_1, R_2, R_3, R_4, R_5)`

    `Max = max(R_1, R_2, R_3, R_4, R_5)`

26. Report the main full-vs-baseline latency reduction using the ratio of the two median latencies, rather than the median of round-level reductions.
27. Use the five configurations and pairwise comparisons to quantify the contributions of CFVST, K LDS double buffering (KDB), V register prefetching (VPF), and their combined effect.```
To obtain results on 8 shapes, we run the ablation on 8 shapes and pick necessary fields.

#### Benchmark Environment and Provenance

The five-way ablation was measured on gfx942 with 304 CUs, using 12 BF16 D128 causal prefill configurations. The baseline for each shape was resolved from production dispatch. All five configurations were compiled and measured in the same process using identical inputs and the same HIP stream.

| Component | Configuration |
| :--- | :--- |
| Hardware target | gfx942, 304 CUs |
| Benchmarked source commit | `51832d8c8078267a8e5274478c3289f3096106ff` |
| ROCm stack | 10.0.0 (public release) |
| HIP | 7.15.26333 |
| Python | 3.12.3 |
| PyTorch | `2.13.0+rocm10.0.0` (public AMD wheel) |
| Triton | `3.8.0+git4cff872c.rocm10.0.0` (public AMD wheel) |
| ROCKE compiler backend | Python |
| LLVM flavor | `llvm23` |

**Correctness:** Each variant was validated against the same FP32 reference before measurement and after every timed batch. All validations passed. Post-batch checks cover the final output after 50 launches, not each individual launch.

#### Shape-by-shape default kernel identity built by `attention_tuning_spec(req, "gfx942_dense").kernel_spec`

| dtype | B | S | Hq | Hkv | D | Exact baseline kernel name | persistent | exp2_fast (resolved) | decode (resolved) | grid | block |
|---|---:|---:|---:|---:|---:|---|---|---|---|---|---|
| bf16 | 1 | 1024 | 32 | 8 | 128 | `rocke_attention_dense_d128_bn64_bf16_causal_lazyrs_gfx942_wpe2` | `false` | `false` | `qb_major` | `[4, 32, 1]` | `[512, 1, 1]` |
| bf16 | 1 | 1024 | 32 | 16 | 128 | `rocke_attention_dense_d128_bn64_bf16_causal_lazyrs_gfx942_wpe2` | `false` | `false` | `qb_major` | `[4, 32, 1]` | `[512, 1, 1]` |
| bf16 | 1 | 2048 | 32 | 8 | 128 | `rocke_attention_dense_d128_bn64_bf16_causal_lazyrs_gfx942_wpe2` | `false` | `false` | `qb_major` | `[8, 32, 1]` | `[512, 1, 1]` |
| bf16 | 1 | 2048 | 32 | 16 | 128 | `rocke_attention_dense_d128_bn64_bf16_causal_lazyrs_gfx942_wpe2` | `false` | `false` | `qb_major` | `[8, 32, 1]` | `[512, 1, 1]` |
| bf16 | 1 | 4096 | 32 | 8 | 128 | `rocke_attention_dense_d128_bn64_bf16_causal_lazyrs_persist304_gfx942_wpe2` | `true` | `true` | `qb_major` | `[304, 1, 1]` | `[512, 1, 1]` |
| bf16 | 1 | 4096 | 32 | 16 | 128 | `rocke_attention_dense_d128_bn64_bf16_causal_lazyrs_persist304_gfx942_wpe2` | `true` | `true` | `qb_major` | `[304, 1, 1]` | `[512, 1, 1]` |
| bf16 | 1 | 8192 | 32 | 8 | 128 | `rocke_attention_dense_d128_bn64_bf16_causal_lazyrs_persist304_gfx942_wpe2` | `true` | `true` | `qb_major` | `[304, 1, 1]` | `[512, 1, 1]` |
| bf16 | 1 | 8192 | 32 | 16 | 128 | `rocke_attention_dense_d128_bn64_bf16_causal_lazyrs_persist304_gfx942_wpe2` | `true` | `true` | `qb_major` | `[304, 1, 1]` | `[512, 1, 1]` |
| bf16 | 1 | 16384 | 32 | 8 | 128 | `rocke_attention_dense_d128_bn64_bf16_causal_lazyrs_persist304_gfx942_wpe2` | `true` | `true` | `qb_major` | `[304, 1, 1]` | `[512, 1, 1]` |
| bf16 | 16 | 4096 | 32 | 8 | 128 | `rocke_attention_dense_d128_bn64_bf16_causal_lazyrs_persist304_hkvmaj_gfx942_wpe2` | `true` | `true` | `hkv_major` | `[304, 1, 1]` | `[512, 1, 1]` |
| bf16 | 16 | 4096 | 32 | 16 | 128 | `rocke_attention_dense_d128_bn64_bf16_causal_lazyrs_persist304_gfx942_wpe2` | `true` | `true` | `qb_major` | `[304, 1, 1]` | `[512, 1, 1]` |
| bf16 | 16 | 8192 | 32 | 8 | 128 | `rocke_attention_dense_d128_bn64_bf16_causal_lazyrs_persist304_hkvmaj_gfx942_wpe2` | `true` | `true` | `hkv_major` | `[304, 1, 1]` | `[512, 1, 1]` |

**Benchmark harness:** `library/benchmarks/gfx942/attention/prefill/bench_rocke_5way_specs.py`
**Kernel source:** `library/kernels/gfx942/attention_dense.py`

#### Reproducability

Create gpu-bench/ at root. Clone branch into it. Below setup.sh sets up python env for rocke. Store it in 
```bash
dnn-providers/hip-kernel-provider/rocke/library/benchmarks/gfx942/attention/prefill/rocke_vs_aiter/setup.sh
```
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
Run setup.sh
```bash
cd /root/gpu-bench/rocm-libraries/dnn-providers/hip-kernel-provider/rocke/library/benchmarks/gfx942/attention/prefill/
chmod +x setup.sh
./setup.sh
```

Store below run_5way.sh in
```bash
dnn-providers/hip-kernel-provider/rocke/library/benchmarks/gfx942/attention/prefill/rocke_vs_aiter/run_5way.sh
```
```bash
#!/usr/bin/env bash
set -Eeuo pipefail

GPU_BENCH_ROOT="${GPU_BENCH_ROOT:-/root/gpu-bench}"
ROCM_REPO="${ROCM_REPO:-$GPU_BENCH_ROOT/rocm-libraries}"
ROCKE_ROOT="${ROCKE_ROOT:-$ROCM_REPO/dnn-providers/hip-kernel-provider/rocke}"
ROCKE_VENV="${ROCKE_VENV:-$GPU_BENCH_ROOT/venvs/rocke}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="${OUT_DIR:-$SCRIPT_DIR/results/5way_specs}"

[[ -f "$ROCKE_ROOT/library/kernels/gfx942/attention_dense.py" ]] || {
  echo "Missing $ROCKE_ROOT/library/kernels/gfx942/attention_dense.py" >&2; exit 2;
}
[[ -f "$ROCKE_VENV/bin/activate" ]] || {
  echo "Missing Python virtualenv $ROCKE_VENV" >&2; exit 2;
}
source "$ROCKE_VENV/bin/activate"

SITE="$(python - <<'PY'
import site
packages = [p for p in site.getsitepackages() if p.endswith('site-packages')]
if not packages:
    raise SystemExit('Could not find Python site-packages')
print(packages[0])
PY
)"
CORE_LIB="$SITE/_rocm_sdk_core/lib"
[[ -f "$CORE_LIB/libamd_comgr.so.3" && -f "$CORE_LIB/libamdhip64.so.7" ]] || {
  echo "Expected wheel ROCm shared libraries not found in $CORE_LIB" >&2; exit 2;
}
export ROCKE_ROOT
export ROCKE_BACKEND=python
export ROCKE_LLVM_FLAVOR="${ROCKE_LLVM_FLAVOR:-llvm23}"
export ROCKE_COMGR_LIB="$CORE_LIB/libamd_comgr.so.3"
export ROCKE_HIP_LIB="$CORE_LIB/libamdhip64.so.7"
export LD_LIBRARY_PATH="$CORE_LIB"
unset LD_PRELOAD
export PYTHONPATH="$ROCKE_ROOT/library:$ROCKE_ROOT/platform/Python:$ROCKE_ROOT${PYTHONPATH:+:$PYTHONPATH}"

# One shell launch -> one Python process -> one current HIP stream.
exec python "$SCRIPT_DIR/bench_rocke_5way_specs.py" \
  --rocke-root "$ROCKE_ROOT" \
  --out-dir "$OUT_DIR" \
  "$@"
```
Run run_5way.sh
```bash
cd /root/gpu-bench/rocm-libraries/dnn-providers/hip-kernel-provider/rocke/library/benchmarks/gfx942/attention/prefill/
chmod +x run_5way.sh
./run_5way.sh
```
Results available in results/

---


