---
name: lds-optimization-rocke
description: >
  Optimize LDS (Local Data Share / shared memory) access patterns in CK DSL
  GPU kernels. Diagnose bank conflicts and high lgkmcnt stalls from ATT trace
  data, then apply swizzle via tile descriptors or padding via lds_k_pad.
  CK DSL manages LDS allocation automatically via tile policies - optimization
  is done through configuration parameters rather than manual address computation.
  Use when trace analysis shows ds_read/ds_write/lgkmcnt as bottleneck.
  Usage: /lds-optimization-rocke
tools: Read,Edit,Bash,Grep,Glob,Agent
---

# LDS Optimization (CK DSL)

Diagnose and fix LDS (shared memory) performance issues in CK DSL kernels
on AMD CDNA GPUs (MI300X/MI308/MI350).

---

## When To Use

Run `/kernel-trace-analysis` first. Apply this skill when the trace shows:

| Signal | Threshold | Example |
|--------|-----------|---------|
| `s_waitcnt lgkmcnt(0)` with high stall | > 3000 cycles per instance | `L605: stall=4080 s_waitcnt lgkmcnt(0)` |
| `ds_write` / `ds_read` with high latency | > 500 cycles per instance | `L761: stall=960 ds_write2_b32` |
| Multiple `s_barrier` between `ds_write` and `ds_read` | Barrier stall > 5000 | `L606: stall=17024 s_barrier` |
| Total LDS-related stall > 15% of kernel stall | Sum all lgkmcnt + ds stalls | Common in GEMM K-loop |

---

## LDS Architecture (Hardware Reference)

### CDNA3 (gfx942 - MI300X)

- LDS size: **64 KB per CU**
- LDS banks: **32 banks**, each **4 bytes wide**
- Bank index = `(byte_address / 4) % 32`
- Peak throughput: **128 bytes/cycle**
- LDS latency: **~20-40 cycles** (async, hidden if enough work between write and read)

### CDNA4 (gfx950 - MI350/MI355X)

- LDS size: **160 KB per CU** (2.5× larger)
- LDS banks: **64 banks**, each **4 bytes wide**
- Bank index = `(byte_address / 4) % 64`
- Peak throughput: **256 bytes/cycle** (2× faster)
- LDS latency: **~2-64 cycles** (2 cycles best case, 64 cycles worst case with full conflict)
- LDS allocation granularity: **1280 bytes** on **1280-byte alignment**

**Key difference**: gfx950 has 64 banks instead of 32, so stride-based conflict patterns change:
- gfx942: stride = 128 bytes → full conflict (all hit same bank)
- gfx950: stride = 128 bytes → only 2-way conflict (threads alternate between 2 banks)
- gfx950: stride = 256 bytes → full conflict

See `/empirical-case-studies` Case Study 2 for measured performance data on LDS swizzle strategies.

---

## CK DSL LDS Management

### Automatic LDS Allocation

CK DSL manages LDS automatically through tile policies. You don't manually allocate LDS - instead, you configure tile descriptors that describe data layout:

```python
from rocke.instances.gemm import GemmSpec

spec = GemmSpec(
    problem=problem,
    tile_m=128,
    tile_n=128,
    tile_k=16,
    # LDS is allocated automatically for A and B tiles
    # Size = (tile_m * tile_k + tile_n * tile_k) * elem_bytes * num_stages
)
```

### LDS Layout Control

CK DSL provides built-in swizzle patterns via tile descriptor transforms:

1. **Default (XOR swizzle)**: Applied automatically by CK DSL for most configs
2. **Padding swizzle**: Controlled via `lds_k_pad` parameter
3. **Custom transforms**: Advanced - modify tile descriptor directly

---

## Diagnosing LDS Bottlenecks

### Step 1: Capture rocprof Stats

```bash
rocprofv3 --stats --kernel-trace -f csv -o stats -- python my_gemm.py
```

Check `stats_kernel_stats.csv` for LDS usage:
- `LDS_Block_Size`: Total LDS bytes per workgroup
- Compare against limits (64KB gfx942, 160KB gfx950)

### Step 2: Analyze ATT Trace

Run `/capture-kernel-trace-rocke` followed by `/kernel-trace-analysis`:

```bash
/capture-kernel-trace-rocke my_gemm.py
/kernel-trace-analysis <trace_directory>
```

Look for:
- High stall on `ds_read_*` / `ds_write_*` (bank conflicts)
- High stall on `s_waitcnt lgkmcnt(0)` after `ds_write` (write latency exposed)
- High stall on `s_barrier` (sync overhead)

### Step 3: Identify Bottleneck Type

**Type A: Bank Conflicts** (high stall on `ds_read`/`ds_write` themselves)

Signs:
- `ds_read_*` / `ds_write_*` instructions with stall > 100 cycles
- Multiple reads/writes with offsets that are multiples of bank count (32 or 64)

Example from ISA:
```
L 766  stall=  160  ds_read2_b64 v[44:47], v28 offset1:8
L 767  stall=  320  ds_read2_b64 v[36:39], v28 offset0:16 offset1:24
```

**Type B: Write-Read Latency Exposed**

Signs:
- `s_waitcnt lgkmcnt(0)` with > 2000 stall cycles immediately after `ds_write`
- Very few instructions between `ds_write` and `s_waitcnt`

Example:
```
L 761  stall=  960  ds_write2_b32 v28, v41, v43 offset0:32 offset1:48
L 764  stall= 4560  s_waitcnt lgkmcnt(0)   <-- exposed latency
L 766  stall=  160  ds_read2_b64 v[44:47], v28 offset1:8
```

---

## Optimization Method 1: Padding Swizzle (lds_k_pad)

### When to Use

- On gfx950 with abundant LDS (160KB)
- When LDS bank conflicts detected in trace
- Simple config parameter (no code changes)

### How It Works

Padding adds extra elements to the K dimension to break stride-based conflict patterns:

```python
# Without padding: tile_k = 64
# LDS stride = 64 * elem_bytes
# If elem_bytes=2 (fp16): stride = 128 bytes → hits same banks on gfx950

# With padding: tile_k_padded = 64 + 8 = 72
# LDS stride = 72 * 2 = 144 bytes → breaks alignment, eliminates conflicts
```

### CK DSL Implementation

```python
spec = GemmSpec(
    problem=problem,
    tile_m=128,
    tile_n=128,
    tile_k=64,
    lds_k_pad=8,  # Add 8 elements of padding to K dimension
    # ... other params
)
```

### Choosing Padding Amount

| Data Type | tile_k | Recommended Padding | Total K_PAD |
|-----------|--------|-------------------|-------------|
| FP16/BF16 | 64 | 8 | 72 |
| FP16/BF16 | 128 | 8-16 | 136-144 |
| FP8/INT8 | 64 | 4-8 | 68-72 |
| FP8/INT8 | 128 | 8 | 136 |

**LDS Cost**:
```
extra_lds = (tile_m + tile_n) * lds_k_pad * elem_bytes * num_stages

Example (128×128×64, FP16, 2-stage, pad=8):
  extra_lds = (128 + 128) * 8 * 2 * 2 = 8192 bytes (8 KB)
  Total LDS = (128*72 + 128*72) * 2 * 2 = 73728 bytes (~72 KB)
  → Fits in gfx950 (160KB), may not fit in gfx942 (64KB)
```

### Expected Performance Gain

From `/empirical-case-studies` Case Study 2 (gfx950, ResNet50 conv3_1):
- **Without padding**: ~494 TFLOPS (XOR swizzle, bank conflicts eliminated but ALU overhead)
- **With padding swizzle**: ~705 TFLOPS (+43%, +211 TFLOPS gain)

Padding wins on gfx950 because:
- Simpler addressing (1-2 ALU vs 5-7 ALU for XOR)
- Hardware scheduler works better with simpler instruction streams
- LDS waste negligible (1.9% of 160KB)

---

## Optimization Method 2: XOR Swizzle (Default)

### When to Use

- On gfx942 with limited LDS (64KB)
- When padding would exceed LDS budget
- Already enabled by default in CK DSL

### How It Works

CK DSL applies XOR-based swizzle automatically through tile descriptor transforms.
The swizzle XORs row index bits into column index to distribute accesses across banks.

### Verifying XOR Swizzle

Check ISA for XOR operations in LDS address computation:

```bash
python src/stage3_extract_isa/count_instructions.py kernel.s | grep -A5 "LDS"
```

Look for patterns like:
```
s_xor_b32 s_addr, row_idx, col_offset  # XOR swizzle
ds_read_b64 v[...], s_addr
```

### XOR Swizzle Parameters

CK DSL infers swizzle parameters from tile size and architecture. You typically don't need to manually configure this.

If performance is still poor after default swizzle, consider:
1. Switch to padding swizzle (`lds_k_pad`)
2. Adjust tile_k to avoid power-of-2 strides
3. Check if pipeline type affects LDS layout (try `compv4` vs `basic_v1`)

---

## Optimization Method 3: Reduce Pipeline Stages

### The Problem

Multi-stage pipelines (e.g., `prefetch_stages=4`) require more LDS:

```
LDS_total = (tile_m * tile_k + tile_n * tile_k) * elem_bytes * prefetch_stages
```

More stages → more LDS → potential conflicts if LDS layout isn't optimal.

### The Solution

Start with fewer stages and increase only if needed:

```python
# Start with 2-stage (double-buffer)
spec = GemmSpec(
    pipeline='compv4',  # 2-stage by default
    # ... other params
)

# If memory-bound and LDS budget allows, try more stages
spec = GemmSpec(
    pipeline='mem',
    prefetch_stages=4,  # 4-stage pipeline
    # ... but requires 2× more LDS than 2-stage
)
```

**Decision rule**:
```
If TFLOPS < 60% peak AND LDS < 80% capacity:
    → Try more prefetch stages

If LDS near capacity OR high bank conflicts:
    → Reduce stages, optimize layout first
```

---

## Optimization Method 4: Increase Write-Read Distance

### The Problem

When `ds_write` is immediately followed by `s_waitcnt lgkmcnt(0)`, the LDS write latency (~20-40 cycles) is fully exposed as stall.

### The Solution (CK DSL Level)

CK DSL handles instruction scheduling through pipeline selection. The `compv4` pipeline has optimized barrier placement to maximize write-read distance:

```python
# basic_v1: minimal distance
spec.pipeline = 'basic_v1'
# ds_write → barrier (short distance) → ds_read

# compv4: optimized distance
spec.pipeline = 'compv4'
# ds_write → compute → barrier → ds_read
# Write completes during compute phase
```

### Advanced: Custom Pipeline Tuning

If you need more control, use the `mem` pipeline with explicit stage configuration:

```python
spec = GemmSpec(
    pipeline='mem',
    prefetch_stages=2,
    # CK DSL will insert prefetch and compute stages
    # to maximize overlap
)
```

---

## Verification Checklist

After applying LDS optimizations:

### 1. Correctness

Run verification tests:

```python
from rocke.run_manifest import run_manifest

summary = run_manifest(
    manifest_path=manifest_path,
    hsaco_path=hsaco_path,
    verify=True  # Compare against NumPy reference
)

assert summary.max_abs_diff < 1e-2, f"Verification failed: {summary.max_abs_diff}"
```

### 2. Re-profile

```bash
rocprofv3 --stats --kernel-trace -f csv -o stats_after -- python my_gemm.py
```

Compare before/after:
- `LDS_Block_Size`: Should increase if padding added
- Kernel latency: Should decrease if optimization successful

### 3. Re-analyze ATT Trace

```bash
/capture-kernel-trace-rocke my_gemm.py --output after_optimization
/kernel-trace-analysis <after_trace_directory>
```

Check:
- `ds_read_*` / `ds_write_*` stall should decrease
- `s_waitcnt lgkmcnt(0)` stall should decrease
- No new bottlenecks introduced (e.g., VGPR spilling)

### 4. LDS Budget

Verify LDS usage from rocprof `*_kernel_stats.csv`:

```bash
# Check LDS_Block_Size column
grep "LDS_Block_Size" stats_after_kernel_stats.csv
```

Ensure:
- gfx942: LDS_Block_Size ≤ 65536 bytes (64 KB)
- gfx950: LDS_Block_Size ≤ 163840 bytes (160 KB)

---

## Common CK DSL GEMM Patterns

### Pattern 1: Bank Conflicts in K-Loop

**Symptom**: High `ds_read` stall when loading A/B tiles from LDS during MFMA compute.

**Root cause**: tile_k is power-of-2 (64, 128) causing stride-based conflicts.

**Fix**:
```python
# Before: tile_k=64 (power-of-2, conflicts on gfx950)
spec = GemmSpec(tile_k=64)

# After: Add padding
spec = GemmSpec(tile_k=64, lds_k_pad=8)  # Effective K=72, breaks alignment
```

### Pattern 2: Write-Read Latency in Pipeline

**Symptom**: High `s_waitcnt lgkmcnt(0)` stall after `ds_write` in main loop.

**Root cause**: basic_v1 pipeline has minimal write-read distance.

**Fix**:
```python
# Before: basic_v1 (simple, short distance)
spec = GemmSpec(pipeline='basic_v1')

# After: compv4 (optimized scheduling)
spec = GemmSpec(pipeline='compv4')
```

### Pattern 3: LDS Capacity Exceeded

**Symptom**: Kernel fails to launch or reports LDS allocation error.

**Root cause**: Too many pipeline stages or large tile + padding exceeds LDS limit.

**Fix**:
```python
# Before: 4-stage + padding exceeds 64KB on gfx942
spec = GemmSpec(
    tile_m=128, tile_n=128, tile_k=64,
    pipeline='mem', prefetch_stages=4,
    lds_k_pad=8,
)
# LDS = (128*72 + 128*72) * 2 * 4 = 147456 bytes > 64KB

# After: Reduce stages or remove padding
spec = GemmSpec(
    tile_m=128, tile_n=128, tile_k=64,
    pipeline='compv4',  # 2-stage, 73728 bytes < 64KB
    lds_k_pad=8,
)
```

---

## Architecture-Specific Recommendations

### gfx942 (MI300X) - 64KB LDS

**Strategy**: Minimize LDS usage, prefer XOR swizzle

```python
spec = GemmSpec(
    tile_m=128, tile_n=128, tile_k=16,
    pipeline='compv4',  # 2-stage
    # Use default XOR swizzle (no lds_k_pad)
    # LDS = (128*16 + 128*16) * 2 * 2 = 16384 bytes (16KB)
)
```

**If bank conflicts persist**:
- Try reducing tile_k (e.g., 16 → 8 for FP16/BF16)
- Try different tile shapes (e.g., 64×256×16 instead of 128×128×16)
- Switch to async copy to reduce LDS pressure

### gfx950 (MI350/MI355X) - 160KB LDS

**Strategy**: Use padding swizzle for simplicity

```python
spec = GemmSpec(
    tile_m=128, tile_n=128, tile_k=64,
    pipeline='compv4',
    lds_k_pad=8,  # Padding swizzle
    use_async_copy=True,  # Further reduce LDS pressure
)
```

**LDS headroom allows**:
- Larger tiles (e.g., 256×128×64)
- More pipeline stages (prefetch_stages=3-4)
- Both padding swizzle + multi-stage pipeline

---

## Performance Expectations

### Typical Gains from LDS Optimization

| Optimization | Expected Gain | Conditions |
|--------------|--------------|------------|
| Add lds_k_pad on gfx950 | +20-40% | If bank conflicts detected, abundant LDS |
| Switch basic_v1 → compv4 | +10-20% | If write-read latency exposed |
| Reduce stages (4→2) | +5-10% | If LDS near capacity causing conflicts |
| Async copy + padding | +30-50% | Large tiles on gfx950, was memory-bound |

### Diminishing Returns

If after LDS optimization TFLOPS is still < 70% peak:
- LDS is no longer the bottleneck
- Check global memory bandwidth (use roofline model)
- Check MFMA utilization (instruction mix from ISA)
- Check register spilling (arch_vgpr > 256)

---

## Quick Reference: Decision Tree

```
LDS optimization needed? (from /kernel-trace-analysis)
│
├─ Yes: Bank conflicts detected (ds_read/write stalls > 100 cycles)
│  │
│  ├─ Architecture?
│  │  ├─ gfx942 (64KB LDS)
│  │  │  └─ Use default XOR swizzle, avoid padding
│  │  └─ gfx950 (160KB LDS)
│  │     └─ Add lds_k_pad=8, simpler addressing
│  │
│  └─ LDS budget allows padding?
│     ├─ Yes → spec.lds_k_pad = 8
│     └─ No → Keep default XOR swizzle
│
├─ Yes: Write-read latency exposed (lgkmcnt stall > 2000 after ds_write)
│  │
│  └─ Pipeline?
│     ├─ basic_v1 → Switch to compv4
│     └─ compv4 → Already optimized, check if async copy helps
│
└─ Yes: LDS capacity exceeded
   │
   └─ Reduce stages or remove padding
      ├─ prefetch_stages > 2 → Reduce to 2
      └─ lds_k_pad > 0 → Remove padding, use XOR
```

---

## Comparison with CK Tile C++

CK Tile C++ uses `split_image=true` parameter which is roughly equivalent to CK DSL's `lds_k_pad`:

| CK Tile C++ | CK DSL Equivalent | Effect |
|-------------|------------------|--------|
| `split_image=false` | Default (XOR swizzle) | Zero LDS overhead, more ALU |
| `split_image=true` | `lds_k_pad=8` | Small LDS overhead, simpler addressing |

From empirical measurements (ResNet50 conv3_1, gfx950):
- CK Tile `split_image=true`: 705 TFLOPS
- CK DSL `lds_k_pad=8`: Similar performance expected (~680-710 TFLOPS)

---

## See Also

- `.claude/OPTIMIZATION_RUNBOOK.md` Section 6.3-6.4 - LDS theory and swizzle patterns
- `/empirical-case-studies` Case Study 2 - XOR vs Padding performance data (gfx950)
- `/capture-kernel-trace-rocke` - Profiling with rocprofv3
- `/kernel-trace-analysis` - ATT trace bottleneck analysis
- `/gemm-optimization-rocke` - GEMM-specific LDS usage patterns
- `src/stage5_compare/compare_rocprof_stats.py` - Compare LDS_Block_Size before/after
