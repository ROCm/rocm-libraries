---
name: prefetch-data-load-rocke
description: >
  Apply prefetch optimization to CK DSL kernel loops by configuring pipeline
  stages to overlap data loads with compute. Use PrefetchStages parameter in
  pipeline configuration to enable multi-stage prefetching (2-8 stages).
  Works with 'mem' pipeline or 'compv4' for automatic double-buffering.
  Use when a CK DSL kernel has global memory bandwidth bottlenecks.
  Usage: /prefetch-data-load-rocke
allowed-tools: Read Edit Bash Grep Glob Agent
---

# Prefetch Data Load Optimization (CK DSL)

Apply pipeline-based prefetch to overlap async data loads with compute in CK DSL
GPU kernels. Unlike FlyDSL's manual loop-carried values, CK DSL uses **pipeline
configuration parameters** to control prefetch depth and staging.

## Core Principle

GPU global memory loads are asynchronous — the load instruction returns
immediately and data is fetched in the background. CK DSL pipelines can issue
loads for future iterations while compute executes on current data, hiding
memory latency behind MFMA throughput.

**Without prefetch** (single-stage, load latency exposed):

```
for i in range(N):
    data = global_load(ptr + i)     # <-- stall: wait for data
    result = mfma_compute(data)     # <-- cannot start until load completes
```

Timeline:
```
|--load--|--stall--|--compute--|--load--|--stall--|--compute--|
```

**With prefetch** (multi-stage pipeline, overlapped):

```
# Pipeline manages prefetch automatically via PrefetchStages parameter
# Stage 0: Prefetch iteration i
# Stage 1: Prefetch iteration i+1 while computing iteration i
# Stage 2: Prefetch iteration i+2 while computing iteration i+1
```

Timeline (2-stage):
```
|--load₀--|--compute₀ + load₁--|--compute₁ + load₂--|--compute₂--|
```

Timeline (3-stage):
```
|--load₀--|--load₁--|--compute₀ + load₂--|--compute₁ + load₃--|...
```

Total time drops from `N * (load + compute)` to roughly
`PrefetchStages * load + N * max(load, compute)`.

## CK DSL Implementation: Pipeline Configuration

CK DSL kernels use **pipeline types** with configurable prefetch stages:

### Pipeline Types

| Pipeline | Description | Prefetch Control | Use Case |
|----------|-------------|------------------|----------|
| `basic_v1` | Single-stage, no prefetch | N/A | Baseline, compute-bound kernels |
| `mem` | Multi-stage with configurable depth | `PrefetchStages=2-8` | Memory-bound kernels, explicit control |
| `compv4` | 2-stage optimized | Fixed 2-stage | General-purpose, good default |

### Configuration via GemmSpec

For GEMM kernels (and implicit GEMM convolutions):

```python
from rocke import GemmSpec

# Basic (no prefetch)
spec = GemmSpec(
    tile_m=128, tile_n=128, tile_k=16,
    pipeline='basic_v1',  # Single-stage
)

# Multi-stage prefetch (2-8 stages)
spec = GemmSpec(
    tile_m=128, tile_n=128, tile_k=16,
    pipeline='mem',
    prefetch_stages=3,  # Prefetch 3 iterations ahead
)

# Optimized 2-stage (recommended for most cases)
spec = GemmSpec(
    tile_m=128, tile_n=128, tile_k=16,
    pipeline='compv4',  # Automatically uses 2-stage prefetch
)
```

### Configuration via Pipeline Policy

For custom kernels using block-level GEMM operations:

```python
from ck_tile.ops.gemm.pipeline import GemmPipelineAgBgCrPolicy

policy = GemmPipelineAgBgCrPolicy(
    stages=3,  # Number of prefetch stages
    scheduler='intrawave',  # Scheduling strategy
    double_smem_buffer=True,  # Enable LDS double-buffering
)

# Use in block GEMM
from ck_tile.ops.gemm.block import BlockGemmAsyncPipeline
block_gemm = BlockGemmAsyncPipeline(policy=policy, ...)
```

## Prefetch Depth Selection

Choose the number of stages based on memory latency vs compute duration:

```python
# Rule of thumb for PrefetchStages:
stages_needed = ceil(memory_latency_cycles / compute_duration_cycles)

# Example 1: Memory-bound (long latency, short compute)
# L2 cache miss = 600 cycles, MFMA block = 200 cycles
# stages_needed = 600 / 200 = 3
spec.prefetch_stages = 3

# Example 2: Compute-bound (short latency, long compute)
# L2 cache hit = 100 cycles, MFMA block = 800 cycles
# stages_needed = 100 / 800 ≈ 0 (1 stage sufficient)
spec.pipeline = 'basic_v1'  # No need for prefetch

# Example 3: Balanced
# L2 latency = 400 cycles, MFMA block = 400 cycles
# stages_needed = 400 / 400 = 1 (2 stages for safety)
spec.pipeline = 'compv4'  # 2-stage default
```

**Empirical guidance:**
- Start with `pipeline='compv4'` (2-stage) as a good default
- If still memory-bound (low MFMA utilization), increase to `pipeline='mem', prefetch_stages=3-4`
- If compute-bound (high MFMA utilization, >90%), use `pipeline='basic_v1'` (no prefetch overhead)
- Beyond 4 stages rarely helps (diminishing returns, register pressure increases)

## LDS Double-Buffering

CK DSL pipelines can also prefetch data into **LDS (Local Data Share)** using
double-buffering to hide LDS write-to-read latency:

```python
spec = GemmSpec(
    tile_m=128, tile_n=128, tile_k=16,
    pipeline='compv4',
    double_smem_buffer=True,  # Enable LDS ping-pong
)
```

**How it works:**
- Two LDS buffers (ping/pong) allocated
- While iteration `i` reads from buffer A, iteration `i+1` writes to buffer B
- Barriers synchronize between write and read phases
- Overlaps LDS write latency with previous iteration's compute

**Trade-off:**
- Doubles LDS usage (2x tile size)
- On gfx942 (64KB LDS): may reduce occupancy if LDS is constrained
- On gfx950 (160KB LDS): recommended, LDS is abundant

## Async Copy (gfx950+)

On CDNA4 (gfx950/MI350X), use hardware async copy for maximum efficiency:

```python
spec = GemmSpec(
    tile_m=128, tile_n=128, tile_k=16,
    pipeline='compv4',
    use_async_copy=True,  # Use hardware async global->LDS DMA
)
```

**Benefits:**
- Offloads global->LDS transfers to dedicated DMA engine
- Frees up VGPRs (no intermediate buffer registers needed)
- Better overlaps with MFMA execution
- **Requirement:** ROCm 6.1+ and gfx950 ISA support

## Register Budget

Prefetch stages consume **arch_vgpr** (architectural VGPRs) for buffering:

```python
# VGPR cost estimation:
vgpr_per_tile = (tile_m * tile_n * elem_size) / (64 * 4)  # bytes per wave / bytes per VGPR
vgpr_prefetch = vgpr_per_tile * prefetch_stages

# Example: tile_m=128, tile_n=128, bf16 (2 bytes)
# vgpr_per_tile = (128 * 128 * 2) / (64 * 4) = 128 VGPRs per stage
# 2-stage: 128 * 2 = 256 VGPRs (full allocation, 1 wave/SIMD max)
# 3-stage: 128 * 3 = 384 VGPRs (SPILL! — exceeds 256 per SIMD on gfx942)
```

**Critical thresholds (gfx942/MI300X, per register file):**
| Arch VGPRs | Waves/SIMD | Status |
|------------|-----------|--------|
| ≤ 128 | 2 | Good occupancy |
| ≤ 256 | 1 | Minimum occupancy |
| > 256 | **SPILL** | Severe perf regression |

**On gfx950 (MI350X):** Async copy reduces VGPR pressure (no intermediate buffers).

**How to check current VGPR usage:**
```bash
# Extract kernel metadata from compiled .so
roc-obj-ls libdispatcher_*.so
roc-obj-extract libdispatcher_*.so
llvm-objdump -d --mcpu=gfx950 *.co | grep -A5 '.amdhsa_kernel'
# Look for: .vgpr_count, .agpr_count

# Or from rocprof trace CSV:
grep 'arch_vgpr\|accum_vgpr' rocprof_kernel_stats.csv
```

## Worked Example: Grouped Conv Optimization

**Problem:** Grouped convolution kernel with global load bottleneck (350 TFLOPS, 60% MFMA utilization).

### Step 1: Baseline (basic_v1, no prefetch)

```python
from rocke import GroupedConvKernelConfig

config = GroupedConvKernelConfig(
    variant='forward', ndim_spatial=2, dtype='bf16',
    tile_m=64, tile_n=128, tile_k=64,
    wave_m=2, wave_n=2, wave_k=1,
    pipeline='basic_v1',  # Single-stage, no prefetch
    scheduler='intrawave',
)

# Compile and run
from grouped_conv_utils import setup_multiple_grouped_conv_dispatchers
libs = setup_multiple_grouped_conv_dispatchers([config])
# Result: 350 TFLOPS, arch_vgpr=120, occupancy=2 waves/SIMD
```

**Diagnosis:** Low MFMA utilization → memory-bound.

### Step 2: Add 2-stage prefetch (compv4)

```python
config.pipeline = 'compv4'  # 2-stage optimized pipeline

# Re-compile and run
libs = setup_multiple_grouped_conv_dispatchers([config])
# Result: 420 TFLOPS (+20%), arch_vgpr=180, occupancy=1 wave/SIMD
```

**Analysis:** TFLOPS improved, but occupancy dropped. MFMA utilization now 75%.

### Step 3: Enable async copy (reduce VGPR pressure)

```python
config.use_async_copy = True  # gfx950 only

# Re-compile and run
libs = setup_multiple_grouped_conv_dispatchers([config])
# Result: 460 TFLOPS (+31%), arch_vgpr=140, occupancy=2 waves/SIMD (restored!)
```

**Analysis:** Async copy recovered occupancy while keeping prefetch benefits.

### Step 4: Add LDS double-buffer (gfx950 has 160KB LDS)

```python
config.double_smem_buffer = True  # LDS ping-pong

# Re-compile and run
libs = setup_multiple_grouped_conv_dispatchers([config])
# Result: 505 TFLOPS (+44%), LDS=96KB, occupancy=2 waves/SIMD
```

**Analysis:** LDS double-buffer overlapped LDS write latency with MFMA.

### Step 5: Optimize for gfx950 (padding swizzle)

```python
config.split_image = True  # Enables padding swizzle (lds_k_pad)
config.epilogue = 'cshuffle'  # Cross-lane shuffle for C writeback

# Re-compile and run
libs = setup_multiple_grouped_conv_dispatchers([config])
# Result: 520 TFLOPS (+48%), LDS=128KB
```

**Final config:** 520 TFLOPS vs 350 TFLOPS baseline = **+48% gain**.

## Applicable Patterns

This optimization applies when you see these signals in profiling:

| Signal | Description | Fix |
|--------|-------------|-----|
| Low MFMA utilization (<70%) | Compute units stalled waiting for data | Add prefetch stages (`pipeline='compv4'` or `'mem'`) |
| High VMEM stall cycles | Global load latency exposed | Increase `prefetch_stages=3-4` |
| High LDS bank conflict rate | LDS reads/writes serialized | Enable `double_smem_buffer=True` |
| VGPR spills | Register file overflow | Use `use_async_copy=True` (gfx950) or reduce `prefetch_stages` |
| Low occupancy (1 wave/SIMD) | Too many VGPRs allocated | Reduce tile size or use async copy |

## Comparison: CK DSL vs FlyDSL Prefetch

| Aspect | CK DSL | FlyDSL |
|--------|--------|--------|
| Mechanism | Pipeline config parameters | Manual `range(..., init=...)` loop-carried values |
| Prefetch depth | `prefetch_stages=2-8` | Manual prologue/epilogue |
| LDS double-buffer | `double_smem_buffer=True` | Manual `SmemAllocator` ping-pong |
| Async copy | `use_async_copy=True` | Manual `buffer_ops.async_*` |
| Complexity | Low (declarative) | High (imperative) |
| Flexibility | Limited to pipeline types | Full control (any pattern) |

**When to use each:**
- **CK DSL:** Most kernels (GEMM, conv, matmul) — declarative, less error-prone
- **FlyDSL:** Custom patterns (paged attention, sparse ops) — full flexibility needed

## Rules and Pitfalls

### Do
- **Start with `pipeline='compv4'`** as a good default (2-stage prefetch)
- **Increase to `prefetch_stages=3-4`** if profiling shows memory bottleneck
- **Enable `use_async_copy=True`** on gfx950 to reduce VGPR pressure
- **Enable `double_smem_buffer=True`** on gfx950 (160KB LDS available)
- **Check VGPR usage** via `llvm-objdump` or `rocprof` to avoid spills
- **Measure occupancy** (waves/SIMD) and ensure it doesn't drop below 1
- **Profile MFMA utilization** to confirm memory-bound assumption

### Don't
- **Don't use prefetch on compute-bound kernels** (>90% MFMA utilization) — adds overhead
- **Don't exceed 4 prefetch stages** — diminishing returns, register pressure increases
- **Don't enable `double_smem_buffer=True` on gfx942** if LDS is tight (<64KB available)
- **Don't combine many prefetch stages with large tiles** — VGPR spills inevitable
- **Don't assume async copy works on gfx942** — requires gfx950+ ISA
- **Don't skip verification** — prefetch should never change correctness, but always test

## Verification

After applying prefetch optimization:

1. **Correctness**: Run `run_manifest(verify=True)` to check against reference
2. **Performance**: Profile with `rocprofv3 --kernel-trace --stats`:
   - Check MFMA utilization increased
   - Check VMEM stall cycles decreased
   - Check overall kernel duration decreased
3. **Register pressure**: Verify `arch_vgpr <= 256` (no spills)
4. **Occupancy**: Verify `waves_per_simd >= 1` (minimum acceptable)

```bash
# Profile before/after
rocprofv3 --stats --kernel-trace -o before -- python run_kernel.py
# (Apply prefetch optimization)
rocprofv3 --stats --kernel-trace -o after -- python run_kernel.py

# Compare metrics
python dsl_docs/optimization/utilities/tools/stage5_compare/compare_rocprof_stats.py \
  before_kernel_stats.csv after_kernel_stats.csv
```

## When NOT To Use

- **Single-iteration loops**: No future iteration to prefetch
- **Compute-bound kernels**: MFMA utilization >90%, memory not the bottleneck
- **Very high register pressure**: If occupancy already 1 wave/SIMD and VGPR near limit
- **Small tiles**: Prefetch overhead may exceed benefit for tiny tile sizes (<32×32)

## Reference

For empirical performance data on prefetch strategies, see:
- `/gemm-optimization-rocke` skill (GEMM-specific pipeline tuning)
- `/lds-optimization-rocke` skill (LDS double-buffering trade-offs)
- `.claude/skills/empirical-case-studies/SKILL.md` (Case Study 3: pipeline comparison)
