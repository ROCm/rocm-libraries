---
name: gemm-optimization-rocke
description: >
  Comprehensive guide to optimizing GEMM (General Matrix Multiply) kernels in
  CK DSL on AMD CDNA GPUs. Covers tiling strategy, pipeline types (mem, compv4),
  LDS swizzle strategies, software pipelining, MFMA instruction atoms,
  epilogue strategies (default, cshuffle), TFLOPS/bandwidth calculation,
  register pressure management, and bottleneck identification.
  Based on CK DSL's gemm_instance framework.
  Usage: /gemm-optimization-rocke
allowed-tools: Read Edit Bash Grep Glob Agent
---

# GEMM Optimization Guide (CK DSL)

Comprehensive guide to writing and optimizing high-performance GEMM kernels in
CK DSL on AMD CDNA GPUs (MI300X gfx942, MI350 gfx950).

Based on the CK DSL `instances/gemm.py` framework.

---

## 1. CK DSL GEMM Architecture Overview

### 1.1 Key Components

CK DSL GEMM kernels are built from composable components:

```python
from rocke.instances.gemm import GemmProblem, GemmSpec, build_gemm

# Problem definition
problem = GemmProblem(M=4096, N=4096, K=4096, dtype='fp16')

# Kernel specification
spec = GemmSpec(
    problem=problem,
    name="my_gemm",
    tile_m=128,
    tile_n=128,
    tile_k=16,
    warp_m=2,
    warp_n=2,
    warp_tile_m=64,
    warp_tile_n=64,
    warp_tile_k=16,
    pipeline='mem',     # or 'compv4'
    epilogue='default', # or 'cshuffle'
)

# Build and compile
kernel = build_gemm(spec)
artifact = compile_kernel(kernel, isa="amdgcn-amd-amdhsa--gfx950")
```

### 1.2 Three-Level Tiling Hierarchy

```
C[M, N] = A[M, K] × B^T[K, N]

Workgroup level (Block):
  tile_m × tile_n output tile per workgroup
  Processes tile_k columns of A/B per K-loop iteration

Warp level (Wave):
  warp_m × warp_n waves partition the tile
  Each wave computes warp_tile_m × warp_tile_n sub-tile

Thread level (MFMA):
  MFMA atoms (16×16 or 32×32) per wave
  Each lane owns 4 output elements (f32×4 accumulator)
```

### 1.3 Thread Mapping

```
Workgroup: 256 threads = warp_m × warp_n × 64 threads/wave

Wave ID mapping:
  wave_x = wave_id % warp_n  → N dimension
  wave_y = wave_id / warp_n  → M dimension

Lane mapping (within 64-lane wave):
  lane_id ∈ [0, 63]
  For 16×16 MFMA: each lane owns 4 output elements
  For 32×32 MFMA: each lane owns 16 output elements
```

---

## 2. Tiling Strategy

### 2.1 Recommended Tile Configurations

| Scenario | tile_m | tile_n | tile_k | warp_m×n | MFMA | Notes |
|----------|--------|--------|--------|----------|------|-------|
| FP16 balanced | 128 | 128 | 16 | 2×2 | 16×16×16 | Good occupancy |
| FP16 large batch | 256 | 128 | 16 | 4×2 | 16×16×16 | More M-parallelism |
| BF16 baseline | 128 | 128 | 16 | 2×2 | 16×16×16 | Same as FP16 |
| FP8 high throughput | 128 | 128 | 32 | 2×2 | 16×16×32 | 2× K per MFMA |
| INT8 balanced | 128 | 128 | 32 | 2×2 | 16×16×32 | Same as FP8 |
| Small M (M≤64) | 64 | 256 | 16 | 1×4 | 16×16×16 | Maximize N-parallelism |
| Large tile variant | 128 | 256 | 16 | 2×4 | 16×16×16 | Needs more LDS |

### 2.2 Tile Size Constraints

- **tile_m, tile_n** must be divisible by `warp_tile_m, warp_tile_n`
- **warp_tile_m, warp_tile_n** must be divisible by MFMA atom size (16 or 32)
- **tile_k** must match MFMA K dimension (16 for fp16/bf16, 32 for fp8/int8)
- **LDS budget**: `(tile_m * tile_k + tile_n * tile_k) * elem_bytes * num_stages` must fit in LDS
  - gfx942: 64 KB LDS per CU
  - gfx950: 160 KB LDS per CU

### 2.3 Warp Tile Calculation

```python
# Number of MFMA atoms per wave
mfma_m = warp_tile_m // 16  # or // 32 for 32×32 MFMA
mfma_n = warp_tile_n // 16
total_mfma_per_wave = mfma_m * mfma_n

# Example: warp_tile 64×64 with 16×16 MFMA
mfma_m = 64 // 16 = 4
mfma_n = 64 // 16 = 4
total_mfma = 4 × 4 = 16 MFMAs per wave per K-iteration
```

---

## 3. Pipeline Types

### 3.1 Pipeline Overview

CK DSL supports multiple software pipeline strategies:

| Pipeline | Stages | LDS Usage | Async Copy | Best For |
|----------|--------|-----------|------------|----------|
| `basic_v1` | 1 | 1× tile | No | Simple baseline, low LDS |
| `mem` | Configurable (1-8) | N× tile | No | Manual staging control |
| `compv4` | 2-stage | 2× tile | Optional | Production, good balance |

### 3.2 Basic_v1 Pipeline

Simplest pipeline: load → compute → repeat

```
Iteration i:
  1. Load A[tile_k] → LDS, Load B[tile_k] → LDS
  2. Barrier (wait for LDS)
  3. Compute tile with MFMA
  4. Repeat for next tile_k
```

**Pros**: Minimal LDS (single-buffer)
**Cons**: No overlap, global load latency exposed

### 3.3 Mem Pipeline (Multi-Stage)

Configurable number of pipeline stages:

```python
spec = GemmSpec(
    pipeline='mem',
    prefetch_stages=2,  # 2-stage double-buffer
    # ... other params
)
```

**2-Stage (Double-Buffer)**:
```
Stage 0: Load tile_k=0 → LDS_A[0], LDS_B[0]
Stage 1: Load tile_k=1 → LDS_A[1], LDS_B[1], Compute tile_k=0
Stage 2: Load tile_k=2 → LDS_A[0], LDS_B[0], Compute tile_k=1 (swap buffers)
...
```

**Pros**: Hides global load latency behind compute
**Cons**: 2× LDS usage, needs more registers

### 3.4 Compv4 Pipeline

Optimized 2-stage pipeline with async copy support:

```python
spec = GemmSpec(
    pipeline='compv4',
    use_async_copy=True,  # Enable global→LDS DMA
    # ... other params
)
```

**Features**:
- Double-buffered LDS for A and B
- Optional async DMA (global → LDS direct, bypasses VGPR)
- Optimized barrier placement
- Better instruction scheduling

**LDS Layout**:
```
LDS_A: [2 stages × tile_m × tile_k × elem_bytes]
LDS_B: [2 stages × tile_n × tile_k × elem_bytes]
Total = 2 × (tile_m + tile_n) × tile_k × elem_bytes
```

---

## 4. LDS Bank Conflict Mitigation

### 4.1 Swizzle Strategies

CK DSL provides built-in swizzle support via tile descriptor transforms.

**XOR Swizzle** (default for most configs):
- Mathematically proven conflict-free
- Applied automatically by CK DSL's LDS allocator
- Zero LDS overhead, ~1-2 SALU per address
- Best for capacity-constrained scenarios (gfx942: 64KB LDS)

**Padding Swizzle**:
- Pad LDS stride to avoid conflict patterns
- Example: K=64 → K_PAD=72 (adds 12.5% padding)
- Simpler addressing (fewer ALU ops)
- Best for abundant LDS (gfx950: 160KB)

**Universal-GEMM `lds_swizzle` row shift.** The universal GEMM's element-granular XOR
swizzle `col ^= ((row >> R) % 2^W) << L` uses R = 0 by default, which is conflict-free
for 16x16x32 fragments at tile_k 64 but not for 16x16x32 at tile_k 32 or 32x32x16 at
tile_k 32 / 64 (2-way to 4-way `ds_read_b128` conflicts on gfx950). `TraitSpec.lds_swizzle_r`
sets R; predict the right value with `rocke.analysis.lds` (the lds-bank-conflict-expert
skill) for the exact fragment lane map before measuring.

### 4.2 Enabling Padding Swizzle

```python
spec = GemmSpec(
    # ... tile params
    lds_k_pad=8,  # Pad K dimension by 8 elements
)
```

**Decision Heuristic** (from empirical-case-studies):
```python
if LDS_total < 96KB:
    use_xor_swizzle()      # Capacity-constrained (gfx942)
elif LDS_avail >= 128KB:
    use_padding_swizzle()  # Abundant LDS (gfx950)
else:
    benchmark_both()       # Architecture-dependent
```

See `/empirical-case-studies` Case Study 2 for measured performance data.

---

## 5. MFMA Instruction Selection

### 5.1 Available MFMA Atoms

| Data Type | K per MFMA | Instruction | Accumulator | warp_tile_k |
|-----------|-----------|-------------|-------------|-------------|
| FP16 | 16 | `mfma_f32_16x16x16f16` | f32×4 | 16 |
| BF16 | 16 | `mfma_f32_16x16x16bf16` | f32×4 | 16 |
| FP8 | 32 | `mfma_f32_16x16x32_fp8_fp8` | f32×4 | 32 |
| INT8 | 32 | `mfma_i32_16x16x32i8` | i32×4 | 32 |
| FP32 | 4 | `mfma_f32_16x16x4f32` | f32×4 | 4 |
| FP16 (32×32) | 8 | `mfma_f32_32x32x8f16` | f32×16 | 8 |

### 5.2 Selecting MFMA Size

CK DSL infers MFMA atom from `warp_tile_m`, `warp_tile_n`, and `dtype`:

```python
# 16×16 MFMA (most common)
spec = GemmSpec(
    warp_tile_m=64,  # Divisible by 16
    warp_tile_n=64,  # Divisible by 16
    dtype='fp16',    # → mfma_f32_16x16x16f16
)

# 32×32 MFMA (larger atom, fewer MFMAs)
spec = GemmSpec(
    warp_tile_m=64,  # Divisible by 32
    warp_tile_n=64,  # Divisible by 32
    dtype='fp16',
    mfma_atom='32x32',  # Explicitly request 32×32 variant
)
```

**16×16 vs 32×32 Trade-offs**:
- **16×16**: More MFMAs per tile, better load/compute overlap
- **32×32**: Fewer MFMAs, lower VGPR pressure, may have scheduling bubbles

---

## 6. Epilogue Strategies

### 6.1 Default Epilogue

Direct accumulator → global memory write:

```python
spec = GemmSpec(
    epilogue='default',
    # Each thread writes its MFMA outputs directly to C
)
```

**Characteristics**:
- No extra LDS required
- Simple implementation
- May have non-coalesced stores for certain tile sizes
- Good for small tiles or when LDS is constrained

### 6.2 CShuffle Epilogue

Cooperative shuffle via LDS for coalesced writes:

```python
spec = GemmSpec(
    epilogue='cshuffle',
    # Threads cooperate to rearrange data for vectorized stores
)
```

**How it works**:
1. Threads write accumulator values to LDS in MFMA layout
2. Barrier synchronization
3. Threads re-read from LDS in coalesced output layout
4. Vectorized global stores (buffer_store_dwordx4)

**LDS Cost**:
```
lds_epilogue = tile_m × tile_n × 4 bytes (f32)
Example (128×128): 128 × 128 × 4 = 64 KB
```

**When to use**:
- tile_n ≥ 128 (benefit from coalesced stores)
- LDS budget allows (especially on gfx950: 160KB)
- Memory-bound workloads where output bandwidth matters

---

## 7. Async Copy (Global → LDS DMA)

### 7.1 When to Use Async Copy

Enable with `use_async_copy=True`:

```python
spec = GemmSpec(
    pipeline='compv4',
    use_async_copy=True,  # Direct global→LDS DMA
    tile_m=128,           # Larger tiles benefit more
)
```

**Benefits**:
- Bypasses VGPR (A/B data goes directly global → LDS)
- Reduces arch_vgpr pressure (more occupancy)
- Better for large tiles (≥ 128×128)

**Trade-offs**:
| Aspect | Sync Copy | Async Copy |
|--------|-----------|------------|
| Path | Global → VGPR → LDS | Global → LDS (DMA) |
| arch_vgpr | Higher (buffers in VGPR) | Lower (data bypasses VGPR) |
| Scheduling | Explicit sync points | DMA engine managed |
| Best for | Small tiles, simple flow | Large tiles, register pressure |

### 7.2 Architecture Differences

**gfx942 (MI300X)**:
- Async copy granularity: 4 bytes (dword)
- Requires multiple DMA ops for large transfers

**gfx950 (MI350/MI355X)**:
- Async copy granularity: 16 bytes (dwordx4)
- More efficient bulk transfers

---

## 8. Register Pressure Management

### 8.1 VGPR Estimation

```python
# Accumulator registers (use accum_vgpr on CDNA)
accum_vgpr = (warp_tile_m // 16) × (warp_tile_n // 16) × 4

# Tile buffers (if sync copy, arch_vgpr)
a_buffer_vgpr = (tile_m × tile_k × elem_bytes) // 4  # if not async
b_buffer_vgpr = (tile_n × tile_k × elem_bytes) // 4  # if not async

# Address/index registers
overhead_vgpr = ~20

total_arch_vgpr = a_buffer_vgpr + b_buffer_vgpr + overhead_vgpr
```

**Example (128×128×16, FP16, sync copy)**:
```
accum_vgpr = (128/16) × (128/16) × 4 = 8 × 8 × 4 = 256
a_buffer = (128 × 16 × 2) / 4 = 1024 (too large!)
b_buffer = (128 × 16 × 2) / 4 = 1024 (too large!)
→ Use async copy to eliminate buffer VGPRs
```

### 8.2 Occupancy Targets

On CDNA (256 arch_vgpr + 256 accum_vgpr per SIMD):

| arch_vgpr | accum_vgpr | Waves/SIMD | Assessment |
|-----------|-----------|------------|------------|
| ≤ 128 | ≤ 128 | 2 | Excellent |
| 129-256 | ≤ 256 | 1 | Acceptable for compute-bound |
| > 256 | any | Spills | Critical issue |

**Remedies for high VGPR**:
- Enable async copy
- Reduce tile_k
- Use smaller warp tiles
- Switch to 32×32 MFMA (fewer accumulators)

---

### 8.3 AGPR accumulators for 128x128-per-wave tiles (gfx950, `acc_agpr`)

A 4-wave 256x256 block with 128x128 per wave needs 256 f32 accumulator registers per
lane: the whole AGPR file. `TraitSpec.acc_agpr` makes the universal GEMM build it
cleanly; three compiler interactions decide whether it does (details and measurements
in `examples/gfx950/gemm_perf_square_warpspec/AGPR_CASE_STUDY.md`):

- **Constant-seeded loop-carried vectors are split.** AMDGPU CodeGenPrepare breaks a
  large PHI whose incoming value is a constant into scalar PHIs; seed accumulators
  from an opaque `asm "", "=a,0"` of the zero vector instead.
- **The MFMA form is a compiler choice.** `amdgpu-agpr-alloc` alone selects the AGPR
  destination form on some LLVM builds; newer ones select the VGPR form and copy
  between files around every MFMA. The kernel attr `mfma_agpr_form` adds
  `-mllvm -amdgpu-mfma-vgpr-form=0` to the compile options for the LLVM 21+ flavors
  (decided by the lowering flavor, not by the compiler that happens to load). Always check
  the ISA of the compiler that will actually build the kernel (`v_mfma ... a[..]`
  destinations, no `v_accvgpr_mov` in the loop, no scratch).
- **The loop exit can copy every AGPR at once.** Hand accumulators out per atom with an
  ordered `asm sideeffect "", "=a,0"` right before their conversion (or, with split-K,
  right before their atomic adds).
- Each AGPR / interleave lever is admitted only where its bodies passed GPU numerics: all of them on
  gfx942 and gfx950, none on gfx90a yet. On gfx942 the AGPR bodies pass but lose to the
  16-wave ring body.

With one wave per SIMD nothing hides a stall: interleave the next fragments' `ds_read`s
and the DTL issues one behind each MFMA and pin the order with `sched_barrier(0)`
(`mfma_interleave`, `ilv_merge`), and check that no MFMA carries two of them. At a
board power cap a higher per-clock MFMA efficiency mostly turns into a lower clock;
compare sustained throughput and clock, not only short hot runs.

### 8.4 Persistent loops, stream-K, ragged and small problems (gfx950)

Details and measurements: `examples/gfx950/gemm_perf_square_warpspec/DTL_RING_SHAPES_CASE_STUDY.md`.

- **The tile shape has to fit the problem before anything else matters.** A 256x256 tile
  that fills a quarter of the last wave wastes most of the device; a 128x128 8-wave DTL
  ring (or, for K % 64 == 32, a 128x128x32 / 256x128x32 interleave ring) took small
  problems up substantially. The dispatcher picks it when the 256x256 last-wave fill is
  below 0.75 (`cdna_dtl_ring_small`).
- **One workgroup per CU needs a deeper ring.** When the small tiles fit a single wave,
  no co-resident workgroup hides the DMA latency, and the 2-stage ring stalls on a full
  `vmcnt(0)` drain every K tile (WaveScope). The 3-stage fragment-prefetch interleave
  form of the same tile removes that stall; with more than one wave of tiles the lighter
  2-stage body is faster again.
- **Gate a new candidate on every axis its evidence did not cover.** The first
  small-tile and pad rules looked only at last-wave fill and tile count and admitted
  small-K shapes nobody had measured; several of those lost to the default. Where an
  alternative candidate exists, give the new one the same K floor as its neighbours (or
  measure the regime first); where none exists (ragged M / N), admit and measure.
- **Fit dispatch thresholds against held-out shapes.** A rule fitted to the shapes that
  motivated it (here: "big grids take the wgm-4 body") can be wrong a few tiles away;
  two refits of the big-grid rule each failed on the next round of held-out shapes before
  the third held. Write the rule, then measure shapes chosen before the measurement.
- **Ragged M / N on the DTL ring.** `pad_m` / `pad_n` only guard the epilogue stores; the
  DMA loads of rows past M / N still have to stay inside the buffer. `dtl_tight_rsrc`
  bases every DTL resource at the tile origin with the operand's remaining bytes as its
  range, so the hardware range check returns zeros there (`soffset` 0, so the result
  does not depend on whether the check includes `soffset`). `pad_fast_tile` branches once
  per tile to an unguarded epilogue for tiles fully inside M x N (the per-element compares
  are expensive at short K without it). The dispatcher's `cdna_dtl_ring_pad` candidate
  serves shapes that previously had no candidate; at small K on a moderate grid the
  small-tile ring serves them better.
- **Stream-K needs every workgroup of its pool resident.** An owner spins on partials of
  higher-index workgroups, so a pool larger than `CUs x workgroups per CU` hangs instead
  of failing: make that a validity rule (LDS and worst-case register occupancy) and check
  the live device at launch (partitioned modes have fewer CUs).
- **Stream-K's cost is its partial traffic.** With a full 256x256 f32 partial tile per
  contributing workgroup the fixup moves on the order of the operands' bytes at short K,
  and the one partition built (DP + 2-tile SK over the whole pool) lost everywhere
  measured. That is a statement about this partition, not about stream-K: SK over the
  remainder tiles only, over just the idle workgroups of the last wave, smaller SK tiles
  and narrower partials are untried.
- **Never spin on an acquire load on gfx942 / gfx950.** An agent-scope acquire is
  `buffer_inv sc1`: spinning on it invalidates the XCD's L2 every iteration and slowed the
  whole kernel severalfold. Spin with `global_flag_wait_eq(..., ordering="monotonic")`
  and put one `fence(scope="agent", ordering="acquire")` after the wait.
- **Chiplet rasterization of very large grids.** Narrower super-tile groups
  (`chiplet_wgm=4`) help on very large tile counts, at small K, once A + B outgrow most of
  the MALL, and on wide grids from a moderate K on -- through shorter DTL latency per K
  tile and fewer waits, not a higher L2 hit rate (the PMC shows more DRAM reads). The
  16-wave ring with wgm 4 + `iglp_opt(1)` is the better aligned body while A + B fit the
  MALL, the 4-wave AGPR body beyond; on moderate grids wgm 4 loses and the wgm-8 ring
  stays (`common.dtl_big_grid_body`).

### 8.5 Row-interleave dispatch classes (gfx950)

The gfx950 GEMM dispatcher (`dispatch/gemm/common.py`, `rilv_*`) serves large aligned,
short-K and ragged problems with a 4-wave AGPR body (2x2 waves of 128x128, tile_k 64,
2-stage DTL ring) whose LDS tile is row-interleaved (`lds_row_interleave`, 32-byte pad
per KiB: conflict-free `ds_read_b128`, one address VGPR per fragment stream) and whose
epilogue writes 16-byte non-temporal vector stores straight from the accumulators
(`rilv_vec_epilogue`, `rve_store_nt`; `rve_pad` 2 on ragged M / N). The K tile runs
the `hb_sched` slot schedule (front-loaded fragment reads, release / acquire barriers)
with `dtl_soff0`, `dtl_addr_spread`, `hb_dtl_delay`, `kloop_unroll`, `hb_skew`,
`hb_tile_lgkm0`, `das_late` and `mfma_serpentine`.

- **Classes.** Aligned (M, N multiples of 256, K % 64 == 0, K >= `RILV_MIN_K`, at
  least `RILV_MIN_TILES` 256x256 tiles); aligned short K (`RILV_SHORTK_MIN_K` <= K <
  `RILV_MIN_K` with a well-filled last wave); ragged M / N (`rilv_pad_admits`). The
  grid is persistent on every CU at short K with more than `RILV_PERSIST_MIN_WAVES`
  waves of tiles, else non-persistent with chiplet wgm 16 / chunk 8 (wgm 32 / chunk 2
  on the ragged long-K class).
- **Retained alternatives.** `ROCKE_GEMM_RILV_BODY` (`asm_loop`, `sia`, `base`) and
  `ROCKE_GEMM_RILV_HB_LOOP=asm_loop` select bodies that did not win; the module
  constants `RILV_ASM_LOOP_*`, `RILV_LONGK_GRID`, `RILV_PAD_HB_TILE_LGKM0` and
  `RILV_PAD_DAS_LATE` are default-off knobs (`das_late` with the exact DTL bounds
  makes LLVM spill the accumulators to scratch).
- **MFMA issue order.** `mfma_serpentine` (odd A rows walk the B fragments backwards)
  is on for the tuned, pad and short-K merged-interleave bodies (`RILV_SERPENTINE`); it
  only reorders independent MFMAs, so results are bit-identical. The alternative
  orders `mfma_order` 2 (odd K atoms reversed: forces register copies) and 3
  (serpentine columns: no better than `mfma_serpentine`) are default-off knobs.
- **XCD-local stream-K tail (not dispatched).** `rilv_streamk_tail_admits` /
  `rilv_streamk_tail_trait` turn the tuned body into a persistent `streamk` +
  `sk_xcd_local` loop for aligned long-K grids with many waves and a near-empty last
  wave. The admission constants are fitted on a few shapes and the effect varied
  between nodes, so the dispatcher does not call them; a caller must also allocate the
  stream-K workspace (`streamk_workspace_bytes`).
- **Threshold provenance.** The `RILV_*` and `SMALL_TILE_SHORTK_*` K / tile-count
  thresholds were calibrated on a fixed gfx950 shape set; several sit on K values or
  tile counts of that set (for example `RILV_HB_MIN_K`, `RILV_MIN_K`,
  `RILV_PERSIST_MAX_K`, `SMALL_TILE_SHORTK_MAX_K`, `RILV_PAD_FAST_TILE_MIN_TILES`) and
  the regions between them were not validated on held-out shapes. Treat them as policy
  defaults and re-calibrate (interleaved sustained windows, full-output numerics before
  and after timing) when the shape population changes.
- **Scope.** gfx950, packed RCR, f16 / bf16, alpha = 1 / beta = 0, non-batched when
  persistent. `is_valid_spec` rejects `rilv_vec_epilogue`, `persistent_ctas` and
  `mfma_swap_ab` with a fused epilogue (`helpers.fuse`). The levers are Python-only
  (see `architecture/engines_and_switching.md`, "Python-only universal-GEMM levers").

## 9. Performance Metrics

### 9.1 TFLOPS Calculation

```python
flops = 2 * M * N * K                    # Each FMA = 2 FLOPs
tflops = flops / (latency_us / 1e6) / 1e12

# Theoretical peak (gfx942 MI300X, single GCD):
#   FP16/BF16: ~653 TFLOPS (mfma_f32_16x16x16)
#   FP8:       ~1306 TFLOPS (mfma_f32_16x16x32)
#   INT8:      ~1306 TOPS
```

### 9.2 Bandwidth Calculation

```python
bytes_moved = (M * K * elem_bytes)     # A matrix
            + (N * K * elem_bytes)     # B matrix
            + (M * N * out_bytes)      # C output

tbps = bytes_moved / 1e12 / (latency_us / 1e6)

# HBM bandwidth (gfx942 MI300X): ~5.2 TB/s theoretical
```

### 9.3 Roofline Analysis

```
Arithmetic Intensity = flops / bytes_moved

For FP16 GEMM (elem_bytes=2, out_bytes=2):
AI = (2 * M * N * K) / (M*K*2 + N*K*2 + M*N*2)
   = 2*M*N*K / (2*K*(M+N) + 2*M*N)

If M = N >> K: AI ≈ K (increases with K)
If M = N = K:  AI = 2*K / (4*K + 2*K) = K/3

Memory-bound if AI < ~200 (for gfx942)
Compute-bound if AI > ~200
```

---

## 10. Optimization Workflow

### 10.1 Step-by-Step Process

**Step 1: Choose baseline tile**
```python
spec = GemmSpec(
    problem=problem,
    tile_m=128, tile_n=128, tile_k=16,
    warp_m=2, warp_n=2,
    warp_tile_m=64, warp_tile_n=64, warp_tile_k=16,
    pipeline='basic_v1',
    epilogue='default',
)
```

**Step 2: Enable double-buffering**
```python
spec.pipeline = 'compv4'
# Expect +10-20% performance if memory-bound
```

**Step 3: Add async copy (if tile ≥ 128)**
```python
spec.use_async_copy = True
# Expect +5-10% if VGPR-limited
```

**Step 4: Optimize LDS swizzle**
```python
# On gfx950 with abundant LDS:
spec.lds_k_pad = 8
# Expect +5-15% from reduced bank conflicts
```

**Step 5: Tune epilogue**
```python
if tile_n >= 128 and LDS_budget_allows:
    spec.epilogue = 'cshuffle'
# Expect +5-10% if output-bandwidth-bound
```

### 10.2 Profiling Checkpoints

After each change:

```bash
# 1. Benchmark performance
python benchmark_rocke_gemm.py --config my_spec.json

# 2. Extract ISA
python src/stage3_extract_isa/extract_isa.py --rocke my_gemm.py

# 3. Count instructions
python src/stage3_extract_isa/count_instructions.py kernel.s

# 4. Check occupancy from rocprof
rocprofv3 --stats --kernel-trace -- python my_gemm.py
# Look at VGPR_Count, SGPR_Count, LDS_Block_Size in *_kernel_stats.csv
```

---

## 11. Bottleneck Identification

### 11.1 Common Symptoms and Fixes

| Symptom | Likely Cause | Action |
|---------|--------------|--------|
| TFLOPS < 50% peak | Multiple issues | Start with Step 10.1 baseline |
| High `s_waitcnt vmcnt` stalls | Global load exposed | Enable double-buffer or async copy |
| High `s_waitcnt lgkmcnt` stalls | LDS latency | Check bank conflicts, add swizzle |
| High `s_barrier` stalls | Workgroup sync overhead | Reduce barriers, check pipeline |
| Low MFMA utilization | Memory-bound | Increase tile_k, add prefetch stages |
| VGPR spilling | Register pressure | Enable async copy, reduce tile_k |
| Output bandwidth low | Non-coalesced stores | Use cshuffle epilogue |

### 11.2 Using ATT Traces

Run `/capture-kernel-trace-rocke` followed by `/kernel-trace-analysis`:

```bash
# Capture trace
/capture-kernel-trace-rocke my_gemm.py

# Analyze bottlenecks
/kernel-trace-analysis <trace_directory>
```

Look for:
- **Top stall sources**: VMEM wait, LDS wait, MFMA dependency
- **Instruction mix**: MFMA ratio should be > 40% of total instructions
- **Pipeline bubbles**: s_nop between MFMAs indicates poor scheduling

---

## 12. ISA Analysis

### 12.1 Key Metrics from ISA

```bash
# Count instructions in compiled kernel
python src/stage3_extract_isa/count_instructions.py kernel.s
```

**Expected ratios for good GEMM**:
```
MFMA count / total instructions > 40%
Memory ops / total instructions < 40%
s_barrier count = small (1-2 per K-loop iteration)
s_waitcnt count = moderate (balanced with compute)
```

### 12.2 MFMA Count Verification

For tile (128, 128, 16) with 16×16 MFMA, FP16:

```
MFMAs per K-iteration = (tile_m/16) × (tile_n/16)
                      = (128/16) × (128/16)
                      = 8 × 8 = 64 MFMAs

Total MFMAs for M=4096, N=4096, K=4096:
  K-iterations = 4096 / 16 = 256
  Total = 64 × 256 = 16,384 MFMAs per tile
  Total tiles = (4096/128) × (4096/128) = 32 × 32 = 1024
  Grand total = 16,384 × 1024 = 16,777,216 MFMAs
```

Verify this matches ISA MFMA count × dispatch count from profiling.

---

## 13. Comparison with Reference Implementations

### 13.1 CK Tile C++ vs CK DSL

CK DSL should match CK Tile C++ performance when using equivalent configs:

```python
# CK DSL spec equivalent to CK Tile 128×128×16 basic_v1
spec = GemmSpec(
    tile_m=128, tile_n=128, tile_k=16,
    warp_m=2, warp_n=2,
    warp_tile_m=64, warp_tile_n=64, warp_tile_k=16,
    pipeline='basic_v1',
    epilogue='default',
)
```

Compare using:
```bash
python src/stage5_compare/compare_rocprof_stats.py \
    rocke_stats.csv cktile_stats.csv
```

### 13.2 Gap Analysis

If CK DSL < CK Tile:

1. **Check register usage**: CK Tile may have optimized register allocation
2. **Check instruction scheduling**: CK Tile may have better scheduling hints
3. **Check LDS layout**: CK Tile may use different swizzle pattern
4. **Check pipeline depth**: CK Tile may use different prefetch distance

Use `/empirical-case-studies` Case Study 1 for reference performance progression.

---

## 14. Advanced Topics

### 14.1 Mixed-Precision GEMM

FP8 input → FP32 accumulator → FP16 output:

```python
spec = GemmSpec(
    dtype_a='fp8',
    dtype_b='fp8',
    dtype_c='fp16',
    dtype_acc='fp32',  # Internal accumulator
)
```

### 14.2 Batched GEMM

Multiple independent GEMMs in parallel:

```python
problem = GemmProblem(
    M=1024, N=1024, K=1024,
    batch_count=128,  # 128 independent 1024×1024 GEMMs
)
```

Grid dispatch: `(M/tile_m) × (N/tile_n) × batch_count`

### 14.3 Strided Batched GEMM

Batch with regular stride patterns:

```python
spec = GemmSpec(
    # ... tile params
    batch_stride_a=M * K,  # A matrices stored contiguously
    batch_stride_b=N * K,
    batch_stride_c=M * N,
)
```

---

## 15. Quick Reference

### 15.1 Pipeline Selection Decision Tree

```
Is LDS < 64KB?
├─ Yes → use basic_v1 or mem (prefetch_stages=1)
└─ No  → use compv4
    ├─ tile_m < 128 → use_async_copy=False
    └─ tile_m ≥ 128 → use_async_copy=True
```

### 15.2 Epilogue Selection

```
If tile_n >= 128 AND memory-bound AND LDS_budget_allows:
    epilogue = 'cshuffle'
Else:
    epilogue = 'default'
```

### 15.3 Common Config Presets

**Balanced FP16 (gfx942)**:
```python
tile=128×128×16, warp=2×2, warp_tile=64×64×16
pipeline='compv4', epilogue='default'
→ ~450-500 TFLOPS on 4096×4096×4096
```

**High-throughput FP8 (gfx950)**:
```python
tile=128×128×32, warp=2×2, warp_tile=64×64×32
pipeline='compv4', use_async_copy=True, lds_k_pad=8, epilogue='cshuffle'
→ ~900-1000 TFLOPS on 4096×4096×4096
```

**Small-batch optimized (M=64)**:
```python
tile=64×256×16, warp=1×4, warp_tile=64×64×16
pipeline='basic_v1', epilogue='default'
→ Maximize N-parallelism for small M
```

---

## 16. Worked Example: 4096×4096×4096 FP16 GEMM

### Problem Setup

```python
from rocke.instances.gemm import GemmProblem, GemmSpec, build_gemm
from rocke.helpers import compile_kernel

problem = GemmProblem(M=4096, N=4096, K=4096, dtype='fp16')
```

### Step 1: Baseline

```python
spec = GemmSpec(
    problem=problem,
    name="gemm_baseline",
    tile_m=128, tile_n=128, tile_k=16,
    warp_m=2, warp_n=2,
    warp_tile_m=64, warp_tile_n=64, warp_tile_k=16,
    pipeline='basic_v1',
    epilogue='default',
)

kernel = build_gemm(spec)
artifact = compile_kernel(kernel, isa="amdgcn-amd-amdhsa--gfx942")
# Benchmark: ~350 TFLOPS (53% of 653 TFLOPS peak)
```

### Step 2: Add Double-Buffer

```python
spec.pipeline = 'compv4'
# Benchmark: ~420 TFLOPS (+20%)
```

### Step 3: Enable Async Copy

```python
spec.use_async_copy = True
# Benchmark: ~460 TFLOPS (+10%)
```

### Step 4: Optimize for gfx950

```python
spec.lds_k_pad = 8  # Padding swizzle
spec.epilogue = 'cshuffle'
artifact = compile_kernel(kernel, isa="amdgcn-amd-amdhsa--gfx950")
# Benchmark: ~520 TFLOPS (+13%)
```

**Final performance**: 520 TFLOPS = 80% of theoretical peak (good result)

**Analysis**:
```
flops = 2 × 4096³ = 137.4 GFLOP
latency = 137.4 GFLOP / 520 TFLOPS = 0.264 ms

bytes = (4096² × 2 × 2) + (4096² × 2) = 100.7 MB
bandwidth = 100.7 MB / 0.264 ms = 381 GB/s (7% of 5.2 TB/s peak)
→ Strongly compute-bound, excellent MFMA utilization
```

---

## 17. See Also

- `.claude/OPTIMIZATION_RUNBOOK.md` Section 7.1-7.2 - MFMA instruction details
- `.claude/OPTIMIZATION_RUNBOOK.md` Section 8 - Pipeline theory
- `/empirical-case-studies` - Real-world performance data
- `/capture-kernel-trace-rocke` - Profiling with rocprofv3
- `/kernel-trace-analysis` - ATT trace bottleneck analysis
- `src/stage1_benchmark/benchmark_rocke.py` - Benchmarking framework
- `src/stage5_compare/compare_rocprof_stats.py` - Hardware metrics comparison
