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

## Optimization Method 5: K-Outer Tile + Transpose Read (gfx950 / gfx1250)

### When to Use

- On gfx950, MFMA family (not wmma), `wave_size=64`, 16-bit operand dtype
- When an operand tile is transposed on store — the loader writes LDS with a
  per-element `ds_write_b16` scatter instead of one wide store
- `warp_tile_m` / `warp_tile_n` in (16, 32)

This is the case `lds_k_pad` **cannot** fix. Methods 1-2 fix the *read* path;
this method fixes a *write* path that padding structurally cannot reach.

### Step 1: Recognize the Transpose-on-Store Anti-Pattern

A tile needs a transpose when the global tensor is contiguous along the GEMM
**free** axis (M for A, N for B) but the MFMA fragment wants lanes contiguous
along the **reduction** axis (K).

| Tile layout | Loader `vector_axis` | LDS store | Cost |
|-------------|---------------------|-----------|------|
| Reduction axis stride-1 | `vector_axis="col"` | one wide `smem_store_vN` | free |
| Free axis stride-1 | `vector_axis="row"` | `load_vec` × `ds_write_b16` | scatter |

Concretely, in a conv implicit-GEMM: wgrad flips **both** operands (A and B are
each contiguous along the free axis and strided along the reduction axis, so both
pay the scatter). dgrad flips **B only** — its A (`dY`, NHWK) already has a
stride-1 reduction axis, because `k_out` is the innermost component of
`K_dg = Y*X*K` and is also NHWK's stride-1 tensor axis, so its loader is already
`vector_axis="col"` with one wide store.

Signs in the ISA:
```
ds_write_b16 v28, v41            # one per vector element, per chunk
v_add_u32    v28, s_stride, v28  # address add between each
```

### Step 2: Why Those Writes Are Bank-Degenerate

In `vector_axis="row"` mode the tile-local decode is `row = idx1 * load_vec`,
`col = idx0`, so adjacent lanes step the tile by `load_vec` **rows**, not one
row, at a fixed column. The inter-lane dword delta is:

```
delta_dwords = load_vec * (tile_k + lds_k_pad) / 2      # 16-bit elements
```

For every admissible `(tile_k, lds_k_pad)` pair this is an exact multiple of the
32-dword bank period, so the lanes in a run all land on the same bank — the
writes are degenerate *by construction*, not by unlucky tuning.

**`lds_k_pad` cannot fix this.** That pad is derived for a row step of **1** —
the read path. Multiplying the row step by `load_vec` re-absorbs the pad back
into a multiple of the period. Sweeping `lds_k_pad` here is wasted effort; the
fix has to remove the scatter, not re-space it.

**Note**: this derivation, and the `_KOUTER_PAD` argument in Step 4, are both
stated against a 32-dword bank period, which is the wording the instance sources
use. See "LDS Architecture (Hardware Reference)" above for the per-arch bank
count before reusing the modulus in a different context.

### Step 3: The Fix — Flip the Tile, Transpose on Read

Store the tile **K-outer** (rows are K, columns are the free axis) so the global
vector lands contiguously in one row, then recover the fragment with the CDNA4
transpose read `ds_read_tr16_b64`.

Write side, per chunk, before → after:

| Instruction | M-outer (transpose-on-store) | K-outer |
|-------------|------------------------------|---------|
| LDS store | `load_vec` × `ds_write_b16` | 1 × `smem_store_vN` (`b128` for a 16-bit 8-wide vector) |
| Address adds | `load_vec - 1` | 0 |
| `vec_extract` | `load_vec` | 0 |

Read side, before → after. The K-outer fragment feed issues `n / 4` transpose
reads, where `n` is the per-lane operand length of the atom:

| Fragment length `n` | M-outer | K-outer | Delta |
|------|---------|---------|-------|
| 4 (`16x16x16`, `32x32x8`) | 1 × `ds_read_b64` | 1 × `ds_read_tr16_b64` | 0 |
| 8 (`32x32x16`, `16x16x32`) | 1 × `ds_read_b128` | 2 × `ds_read_tr16_b64` + 1 × `vec_concat` | +1 read per fragment |

**Global loads are unchanged.** `choose_vec` tests `tile_rows` in `"row"` mode
and `tile_cols` in `"col"` mode, so the flipped call tests the same extent
against the same product — the `buffer_load` width is invariant under the flip.
Only the LDS store changes.

**Break-even**: at `n = 4` the flip is free on the read side, so it is a pure
win. At `n = 8` you trade one extra `ds_read` per fragment against a net
`3 * load_vec - 2` write-side instructions removed per chunk, *plus* the removal
of a fully degenerate bank pattern. The scatter only exists when `load_vec > 1`,
so the trade never runs the wrong way — but at `n = 8` it is a trade, not a
freebie.

### Step 4: Derive the Row Pad (`_KOUTER_PAD = 8`)

**This is the reusable part.** The K-outer row stride must not be a multiple of
the 32-dword bank period or the *transpose read* degenerates instead.

Lane `l` reads 8 bytes at:
```
(k_base + ((l%16)//4)) * stride + mn_base + ((l%MN)//16)*16 + (l%4)*4
```

The only term that walks **rows** is `((l%16)//4)`. It contributes zero bank
spread whenever:
```
(stride_elems * 2 / 4) % 32 == 0
```

Worked example, 64-wide tile, 16-bit elements:

| `stride_elems` | dwords | `% 32` | Row-group banks | Verdict |
|----------------|--------|--------|-----------------|---------|
| 64 (no pad) | 32 | 0 | 0, 0, 0, 0 | degenerate |
| 64 + 8 = 72 | 36 | 4 | 0, 4, 8, 12 | spread |

A pad of **8** elements also keeps each row 16-byte aligned, which the `b128`
store side needs — so it satisfies both constraints at once.

**Async exception (wgrad)**: the direct global→LDS path deposits lane-contiguous
*packed* bytes and cannot skip a row pad, so `async_dma` forces the K-outer row
pad to **0**. The transpose read is insensitive to the row stride, so dropping
the pad costs nothing on the read side — but it gives back exactly the bank
property K-outer was introduced to obtain, which is why `async_dma` is a swept
axis rather than a deduced one. The dependency also runs the other way: wgrad
`async_dma` *requires* `lds_k_outer=True`, because the direct load needs a
stride-1 reduction axis and wgrad only has one once the tile is stored K-outer.
On dgrad the pad is an unconditional 8, since `async_dma` is rejected outright on
the K-outer path.

### Step 5: Selection, Not a Knob

There is no `--lds-k-outer` flag and no env override. The layout is deduced from
the spec, and dispatch and the sweep driver call the same predicate:

```python
WgradConvSpec.default_lds_k_outer(
    arch, dtype_a, dtype_b, warp_tile_m, warp_tile_n, wave_size)

DgradConvSpec.default_lds_k_outer(
    arch, dtype_b, warp_tile_n, cpg, wave_size)
```

The dgrad predicate is deliberately **asymmetric** — `dtype_b` / `warp_tile_n`
only, never the A-side counterparts — because only B flips. It additionally
carries `cpg` (see "When Not to Use It" below). The `lds_k_outer` spec field
itself still defaults to `False`, so existing goldens are unmoved.

An explicit `lds_layout` or `async_dma` on the dgrad K-outer path is **rejected**
rather than silently ignored, as is an explicit `lds_k_pad` on the wgrad K-outer
path — the row stride comes from the bank analysis, not from the layout.

### When Not to Use It

**The operand is already `vector_axis="col"`**. dgrad's A (`dY`, NHWK) has a
stride-1 reduction axis, so its loader already issues one wide store and its
M-outer fragment read is already conflict-free via `lds_k_pad`. Flipping it would
put the global vector along `m = (n, hi, wi)` — stride `K` in NHWK — destroying
coalescing for zero write-side gain. Check the write side before you flip: if
there is no scatter, there is nothing to remove.

**The load width has already collapsed to 1**. When the stride-1 channel run is
odd (`cpg % 2 != 0` for a 16-bit dtype) `choose_vec` returns 1, `axis_b` is
already `"col"`, and there is no scatter. K-outer then buys nothing and still
pays the extra read at `n = 8` — a small pure regression. This is why the dgrad
predicate carries `cpg`: the saving is proportional to the B load width.

**Non-gfx950 targets**. gfx1250 is supported as a second regime; see below.

### gfx1250 (wave32 WMMA)

**gfx1250 (wave32 WMMA).** Supported, as a second regime rather than a port of
the first. gfx1250 has `ds_load_tr16_b128`, a wave32 transpose-LDS read
overloaded on result element type (`.v8bf16` / `.v8f16`); the IR op is shared
with gfx950 and `core/isa/backend.py` selects the opcode, so no new primitive
was needed. What differs is the lane mapping, and it is *simpler*: 32 lanes
over a 16-wide atom edge give two lane groups that split K rather than the free
axis, so lane `l` owns column `l % 16` and K-half `l // 16`, and a fragment is a
straight run of `n` K values at one column starting at `(l // 16) * n`. The read
returns 8 per lane, so the 16-element gfx1250 fragment is two reads. gfx950's
`((l % 16) // 4)` and `(l % 4) * 4` terms exist only because 64 lanes over that
edge create groups *within* the free axis. `_tr_frag` branches on `wave_size`;
`_LDS_K_OUTER_ARCH_WAVE` pins each arch to its wave size so a mismatched spec is
rejected rather than emitting a formula the hardware does not implement. The
wave32 regime admits one atom, `16x16x32`.

Note the interaction with `Gfx1250Backend.blocks_ds_load_tr16`: that guard marks
ordinary LDS loads volatile because the AMDGPU backend otherwise substitutes
`ds_load_tr16_b128` for a `<8 x half>` load feeding a WMMA, and that
substitution assumes a column-major tile while the M-outer path is row-major.
K-outer is the layout the substitution assumes, and the transpose read is
emitted as an explicit intrinsic call rather than left to the pass, so the guard
is unaffected and still protects the M-outer loads beside it.

**Verified on gfx1250 hardware.** The numeric A/B is green: the K-outer path is
bitwise identical to the M-outer default for dgrad, and matches the fp32
reference for wgrad.

Getting there required a fix, and the trap generalises. `ds_load_tr16_b128`
transposes an 8x8 element block *within each group of 8 lanes*: the 8 lanes each
read 8 contiguous elements, and lane `j` of the group receives element `j` from
all 8 runs. The address a lane supplies is therefore **not** the element it ends
up holding. To land the documented WMMA B layout (lane `l` holds column
`l % 16`, K-half `l // 16`), the group addresses the 8-column block and each
lane supplies a different K row of it:

```
col  = mn_base + ((l % 16) // 8) * 8
row0 = k_base  + (l // 16) * n + (l % 8)     # read r at row0 + 8*r
```

The original form addressed it as if the instruction returned a straight run of
K at the lane's own column. That was byte-identical across both engines and
still read a transposed operand -- byte-identity proves the two engines agree,
never that the formula is right. A per-lane hardware probe of the intrinsic
(fill LDS with `k*16+col`, read back, decode) settles the distribution in one
run and is worth writing before trusting any transpose-read lane map.

### Trap: Fragment Length Is Per-Atom, Not a Constant

For the wave64 **MFMA** atoms, `n` (per-lane operand length) is **8** for
`32x32x16` and `16x16x32`, and **4** for `16x16x16` and `32x32x8`. Hardcoding 8
makes the `16x16x16` atom read rows 8..27 of a 16-row tile — past the end —
returning garbage that verification may not catch on small problems. Always
derive `n` from the atom (`MfmaAtom.a_per_lane` / `.b_per_lane`). The dgrad
builder additionally rejects any `b_per_lane` that is not a multiple of 4, since
`ds_read_tr16_b64` returns exactly 4 elements per lane.

Mind the atom-name collision across families: gfx1250's **WMMA** `16x16x32` is a
*different atom* from the MFMA `16x16x32` above and has
`a_frag_len == b_frag_len == 16`, not 8. Its fragment is therefore two
`ds_load_tr16_b128` reads of 8, and `(l // 16) * n` steps by 16. Read `n` off the
resolved atom, never off the `MxNxK` name.

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
