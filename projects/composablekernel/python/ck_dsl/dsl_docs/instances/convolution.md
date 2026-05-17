# Convolution Instances

This page covers:

- `instances/conv_implicit_gemm.py`
- `instances/conv_implicit_gemm_auto.py` (experimental alternate builder, not exported from `instances/__init__.py`)
- `instances/conv_direct_grouped.py`
- `instances/img2col.py`
- `instances/pooling.py`

Two convolution strategies:

```text
Implicit GEMM:
  NHWC x KRSC -> NHWK expressed as a GEMM (m, n, k) on the implicit shape
  (m = N*Ho*Wo, n = K, k = R*S*C)

Direct grouped:
  Specialized streaming kernels for grouped small-channel cases (16c, 4c)
```

## Implicit-GEMM Convolution

Source: `instances/conv_implicit_gemm.py`.

### Contract

```text
A: NHWC fp16,   [N, Hi, Wi, C]
B: KRSC fp16,   [K, R, S, C]
D: NHWK fp16,   [N, Ho, Wo, K]
```

Implicit GEMM mapping:

```text
M_gemm = N * Ho * Wo
N_gemm = K
K_gemm = R * S * C
```

Kernel ABI (`conv_args_signature()`):

```text
A: ptr<f16, global>      8 bytes
B: ptr<f16, global>      8 bytes
D: ptr<f16, global>      8 bytes
A_bytes: i32             4 bytes   # buffer rsrc bound
B_bytes: i32             4 bytes
D_bytes: i32             4 bytes
```

The `*_bytes` args drive the AMDGPU buffer descriptor `num_records` field (DW2). With the DW3 flags `0x00027000`, OOB byte offsets silently return zero on load and are dropped on store.

### Output Shape

```text
Ho = (Hi + 2*pH - dH*(R - 1) - 1) // sH + 1
Wo = (Wi + 2*pW - dW*(S - 1) - 1) // sW + 1
```

`ConvProblem.Ho`, `Wo`, `M`, `N_gemm`, `K_gemm`, `flops` are derived properties; `.short()` returns `"N8H56W56C64_K64R3S3"`.

### Spec Defaults

```python
@dataclass(frozen=True)
class ConvProblem:
    N: int; Hi: int; Wi: int; C: int
    K: int; R: int; S: int
    sH: int = 1; sW: int = 1
    pH: int = 0; pW: int = 0
    dH: int = 1; dW: int = 1


@dataclass(frozen=True)
class ImplicitGemmConvSpec:
    problem: ConvProblem
    name: str = "conv_igemm"

    tile_m: int = 64
    tile_n: int = 64
    tile_k: int = 128

    warp_m: int = 2
    warp_n: int = 2

    warp_tile_m: int = 16
    warp_tile_n: int = 16
    warp_tile_k: int = 32

    wave_size: int = 64

    pipeline: str = "mem"             # "mem" | "compv3" | "compv4"
    epilogue: str = "default"         # "default" | "cshuffle"
    async_dma: bool = False
    unroll_k: bool = False
    lds_k_pad: Optional[int] = None
    lds_layout: Optional[LdsLayout] = None

    chiplet_swizzle: bool = False
    chiplet_wgm: int = 8
    chiplet_num_xcds: int = 8
    chiplet_chunk_size: int = 64

    waves_per_eu: Optional[int] = None
```

`lds_k_pad=None` lets the kernel pick: `+8` on sync paths when `block_k >= 16`, `+0` on async paths. Override only for sweep experiments.

The example bake-off (`example/ck_tile/dsl/08_bake_off_implicit_gemm`) uses:

```text
tile_m=64, tile_n=64, tile_k=64,
warp_m=2, warp_n=2,
warp_tile_m=32, warp_tile_n=32, warp_tile_k=16,
pipeline="mem", epilogue="cshuffle"
```

and was validated end-to-end at ~230 TFLOPS (per-launch, MI355X / gfx950) during this docs pass.

### A Descriptor: `(m, k) -> NHWC`

The cleanest part of the implicit-GEMM authoring surface; the transform DAG maps the implicit-GEMM (m, k) row/column to the NHWC linear offset and emits the padding validity predicate.

`make_a_descriptor(N, Hi, Wi, C, K, R, S, Ho, Wo, sH, sW, pH, pW, dH, dW)`:

```python
TensorDescriptor.naive("A_nhwc", lengths=[N, Hi, Wi, C],
                       coord_names=["n", "hi", "wi", "c"])
  .transform(
      unmerge("m",  into=["n", "ho", "wo"], dims=[N, Ho, Wo]),
      embed (["ho", "r"], "hi", strides=[sH, dH], offset=-pH, lo=0, hi=Hi),
      embed (["wo", "s"], "wi", strides=[sW, dW], offset=-pW, lo=0, hi=Wi),
      unmerge("k",  into=["r", "s", "c"],   dims=[R, S, C]),
      pad   ("r", lo=0, hi=R),
      pad   ("s", lo=0, hi=S),
  )
```

At use site:

```python
off_elements, valid = A_desc.offset(b, m=m_val, k=k_val)
off_bytes = b.mul(off_elements, b.const_i32(2))      # fp16 -> 2 bytes
safe = b.select(valid, off_bytes, b.const_i32((1 << 31) - 1))
v = b.buffer_load_vN_f16(a_rsrc, safe, c0, dwords=2)
```

`pad("r")` and `pad("s")` matter when `K_gemm` does not cleanly divide the K-tile: without them, the unmerge would compute valid-looking offsets outside the intended filter slice.

### B and D Descriptors

`make_b_descriptor` maps `(k_out, k_gemm) -> KRSC`:

```text
unmerge(k_gemm -> r, s, c), pad(r), pad(s)
```

`make_d_descriptor` maps `(m, k_out) -> NHWK`:

```text
unmerge(m -> n, ho, wo); offset = ((n*Ho + ho)*Wo + wo)*K + k_out
valid = m < M_gemm && k_out < K
```

### Grid

```text
grid_x = ceil_div(K, tile_n)
grid_y = ceil_div(M, tile_m)
grid_z = 1
```

Same structure as GEMM; the implicit-GEMM M dimension is `N * Ho * Wo`.

### Load Phase (sync path)

```text
for each thread's A vector chunk in tile_m x tile_k:
  (m, k) = (block_m0 + local_row, k0 + local_col)
  (off, valid) = A_desc.offset(b, m, k)
  safe = valid ? off*2 : INT32_MAX
  v = buffer_load_vN_f16(a_rsrc, safe, 0, dwords)
  smem_store_vN_f16(A_smem, [local_row, local_col], v, vec)

for each thread's B vector chunk in tile_n x tile_k:
  (n, k) = (block_n0 + local_row, k0 + local_col)
  (off, valid) = B_desc.offset(b, k_out=n, k_gemm=k)
  safe = ...
  v = buffer_load_vN_f16(b_rsrc, safe, 0, dwords)
  smem_store_vN_f16(B_smem, [local_row, local_col], v, vec)

b.sync()
```

### Load Phase (async path)

`async_dma=True` uses `AsyncTileLoader`:

```text
loader = AsyncTileLoader.from_tile(tile_rows=tile_m, tile_cols=tile_k,
                                    block_size=block_size, wave_size=64)
slot = loader.bind(b, smem_dst=A_smem, wave_id=warp_id)
slot.issue(b, tid=tid, rsrc=a_rsrc, descriptor=a_desc_fn,
           coherency=CACHE_STREAM)
# ... same for B ...
b.s_waitcnt(vmcnt=0)
b.sync()
```

Constraints:

- `dwords in {1, 3, 4}`;
- LDS layout must be packed (`lds_k_pad=0`);
- LDS bank conflict avoidance moves into the consumer read arithmetic (XOR swizzle) if it becomes the next bottleneck.

### Compute Loop

Identical skeleton to universal GEMM:

```text
acc = zero f32 accumulators

for k0 in scf_for(0, K_gemm, tile_k):
  load A/B (sync or async)
  for kk in static_for(0, tile_k, atom.k):
    for warp_m fragment, warp_n fragment, output fragment:
      A_frag = smem_load_vN_f16(A_smem, ...)
      B_frag = smem_load_vN_f16(B_smem, ...)
      acc = atom.emit(b, A_frag, B_frag, acc)
      schedule_policy.emit_after_mfma_step(...)
```

`unroll_k=True` replaces the runtime `scf_for_iter` over k0 with a Python `static_for` when `K_gemm` is a compile-time multiple of `tile_k`. This produces straight-line IR and lets the LLVM backend see the entire K-loop body for scheduling, at the cost of larger compiled code.

### Epilogue

`epilogue="default"`:

```text
for each accumulator slot:
  (row_off, col_off) = atom.lane_to_output(b, lane, i)
  m = block_m0 + warp_m_off + row_off
  k_out = block_n0 + warp_n_off + col_off
  (d_off, valid) = D_desc.offset(b, m=m, k_out=k_out)
  safe = valid ? d_off*2 : INT32_MAX
  v = b.cast_f32_to(acc_slot, F16)
  buffer_store_f16(d_rsrc, safe, 0, v)
```

`epilogue="cshuffle"`:

```text
# stage in LDS, then coalesced vector buffer stores
for each acc slot:
  (row_off, col_off) = atom.lane_to_output(b, lane, i)
  smem_store_f16(D_smem, [row_off, col_off], cast_f32_to(acc_slot, F16))
b.sync()
for each thread's coalesced output chunk:
  v = smem_load_vN_f16(D_smem, [...], n=8)
  (d_off, valid) = D_desc.offset(...)
  buffer_store_vN_f16(d_rsrc, d_off*2, 0, v, dwords=4)
```

### Step-By-Step Trace (one workgroup)

```text
 1. Read block_id_x (k_out tile) and block_id_y (m tile).
 2. Compute (block_n0, block_m0) origins.
 3. Build buffer resources for A, B, D from ptrs + *_bytes.
 4. Allocate A_smem, B_smem (and D_smem if cshuffle).
 5. Decompose tid into lane / warp_m_idx / warp_n_idx / warp_m_off / warp_n_off.
 6. Initialize all f32 accumulator vectors to zero.
 7. Enter K_gemm tile loop (runtime or Python-unrolled).
 8. Per K tile: load A chunk(s), load B chunk(s), wait/sync.
 9. For each MFMA K atom: read A/B fragments, atom.emit, scheduler hint.
10. Carry updated accumulators across the loop.
11. After last K tile: emit epilogue (direct or cshuffle).
12. Done.
```

## Direct Grouped Convolution

Source: `instances/conv_direct_grouped.py`.

These are specialized kernels for grouped direct convolution bake-off cases (`cpg=kpg in {16, 4}`), not generic implicit-GEMM conv.

### `DirectConvProblem`

Verified from `instances/conv_direct_grouped.py`:

```python
@dataclass(frozen=True)
class DirectConvProblem:
    N: int
    H: int           # input/output height (no Hi vs Ho here)
    W: int           # input/output width
    groups: int
    cpg: int         # channels per group
    kpg: int         # filters per group (= cpg in bake-off)
    KH: int = 3      # kernel height (not R)
    KW: int = 3      # kernel width  (not S)
    PAD: int = 1
    stride: int = 1
```

Note this layout is different from `ConvProblem`:

- `H`/`W` not `Hi`/`Wi` (the grouped direct conv assumes equal in/out spatial size with padding);
- `KH`/`KW` not `R`/`S`;
- single `PAD` and `stride` ints (no separate `pH`/`pW`/`sH`/`sW`/`dH`/`dW`); dilation is implicitly 1.

### 16c Kernel

`DirectConv16cSpec` / `build_direct_conv_16c`.

Contract:

```text
cpg = 16, kpg = 16
A NHWC, B KRSC (grouped), D NHWK
```

Grid:

```text
grid_x = ceil_div(W, block_q)
grid_y = groups / block_groups
grid_z = N
```

Workgroup: `block_groups * 64` threads, one wave per group.

Algorithm:

```text
1. Identify (n, group_block, w_tile_block) from block IDs.
2. Each wave owns one group's worth of channels.
3. Preload or stream weights for that group.
4. Maintain a circular accumulator pipeline of depth KH over output H rows.
5. For each input/filter row:
   a. Load needed input row/window into LDS slabs.
   b. Apply H/W padding predicates through TensorDescriptor with pad().
   c. Read input + weight fragments.
   d. If fold_k32 is True:
        combine S=0 and S=1 into mfma_f32_16x16x32_f16,
        handle residual S=2 with K=16 mfma_f32_16x16x16_f16.
      Else use 16x16x16 for each S.
6. When an output row is complete:
   a. Vector-store each lane's contiguous 4 output channels via buffer_store_vN_f16.
   b. Reset that circular accumulator slot unconditionally.
```

Key levers (per `RUNBOOK_COMPLIANCE.md`):

- K=32 MFMA fold: ~92 -> ~108 TFLOPS;
- wide direct epilogue (1 `buffer_store_dwordx2` per lane = 4 halves): ~108 -> ~210 TFLOPS;
- `BLOCK_GROUPS=4`: ~210 -> ~214 TFLOPS.

### 4c Kernel

`DirectConv4cSpec` / `build_direct_conv_4c`.

Contract:

```text
cpg = 4, kpg = 4
```

Uses `mfma_f32_4x4x4_f16`: one wave computes 16 independent 4x4x4 matmuls indexed by `batch = lane / 4`.

Algorithm:

```text
1. Pack multiple groups across the 16 wave batches.
2. For each output coordinate assigned to the wave:
   - Load needed input vectors with padding masks (no LDS row pipeline).
   - Load 4-channel weights.
   - Issue mfma_f32_4x4x4_f16.
3. Accumulate across KH*KW.
4. Vector-store the 4 output channels as one contiguous buffer_store_vN_f16.
```

Levers:

- vec2-dword epilogue (1 store/lane, 4 halves fused): ~44 -> ~48 TFLOPS.

This path avoids the implicit-GEMM LDS machinery because the channel group is tiny and direct vectorization is cleaner.

## Img2Col

Source: `instances/img2col.py`.

Materializes the implicit-GEMM A matrix `[M_gemm, K_gemm]`:

```python
@dataclass(frozen=True)
class Img2ColSpec:
    problem: ConvProblem
    tile_m: int = 64
    tile_k: int = 64
    block_size: int = 256
    name: str = "img2col"
```

Algorithm:

```text
1. One thread maps to one output element Y[m, k].
2. Reuse conv_implicit_gemm.make_a_descriptor for (m, k) -> NHWC.
3. Descriptor produces NHWC offset and validity.
4. Invalid padding lanes write zero (or skip).
5. Store Y[m, k].
```

Grid:

```text
grid_x = ceil_div(K_gemm, tile_k)
grid_y = ceil_div(M_gemm, tile_m)
grid_z = 1
```

Use for debugging, verification, and baselines. Generally not the fastest production conv path because it materializes the expanded matrix.

## Pooling

Source: `instances/pooling.py`.

```python
@dataclass(frozen=True)
class PoolingProblem:
    N: int; Hi: int; Wi: int; C: int
    Y: int; X: int        # pool window
    sH: int = 1; sW: int = 1
    pH: int = 0; pW: int = 0
    dH: int = 1; dW: int = 1


class PoolOp(Enum):
    MAX = "max"
    SUM = "sum"
    AVG = "avg"
```

Algorithm:

```text
1. One thread per NHWC output element.
2. Decompose flat output index into (n, ho, wo, c).
3. Loop over pooling window (y, x).
4. Compute input (hi, wi) via embed-style formulas.
5. Mask out invalid (hi, wi).
6. Reduce in f32: max / sum (for avg, divide by selected count or window count).
7. Cast back to output dtype and store.
```

Grid: `ceil_div(total_output_elements, block_size)`.

## Convolution Failure Modes

- Missing `pad("r")` / `pad("s")` in implicit K tails: numerically valid-looking but cross-slice offsets.
- Descriptor returns element offsets but buffer op expects byte offsets: shift left by 1 for fp16, by 2 for bf16, etc.
- False lanes are masked **after** a faulting pointer load — use the buffer-rsrc sentinel pattern instead.
- Async loader writes lane-contiguous LDS but consumer assumes padded / swizzled physical layout.
- K-packed MFMA fold packs S/C channels in the wrong order (close but not bit-correct).
- Direct conv circular accumulator slot is not reset after store.
- Output descriptor uses NHWC instead of NHWK stride order.
- Benchmark compares implicit-GEMM graph mode to direct per-launch mode without labeling launch overhead.
