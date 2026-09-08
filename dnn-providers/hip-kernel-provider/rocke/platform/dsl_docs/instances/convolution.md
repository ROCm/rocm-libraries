# Convolution Instances

This page covers:

- `instances/common/conv_implicit_gemm.py`
- `instances/common/conv_direct_grouped.py`
- `instances/common/img2col.py`
- `instances/common/pooling.py`

The implicit-GEMM tile/pipeline heuristic (formerly an experimental
`conv_implicit_gemm_auto.py` alternate builder) now lives under
`rocke/heuristics/` (ML-driven config selection for the forward
implicit-GEMM path).

Two convolution strategies:

```text
Implicit GEMM:
  NHWC x KYXC -> NHWK expressed as a GEMM (m, n, k) on the implicit shape
  (m = N*Ho*Wo, n = K, k = Y*X*C)

Direct grouped:
  Specialized streaming kernels for grouped small-channel cases (16c, 4c)
```

## Implicit-GEMM Convolution

Source: `instances/common/conv_implicit_gemm.py`.

### Contract

```text
A: NHWC fp16,   [N, Hi, Wi, C]
B: KYXC fp16,  [K, Y, X, C]
D: NHWK fp16,   [N, Ho, Wo, K]
```

Implicit GEMM mapping:

```text
M_gemm = N * Ho * Wo
N_gemm = K
K_gemm = Y * X * C
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
Ho = (Hi + 2*pH - dH*(Y - 1) - 1) // sH + 1
Wo = (Wi + 2*pW - dW*(X - 1) - 1) // sW + 1
```

`ConvProblem.Ho`, `Wo`, `M`, `N_gemm`, `K_gemm`, `flops` are derived properties; `.short()` returns `"N8H56W56C64_K64Y3X3"`.

### Spec Defaults

```python
@dataclass(frozen=True)
class ConvProblem:
    N: int; Hi: int; Wi: int; C: int
    K: int; Y: int; X: int
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

`lds_k_outer` is a backward-only field (`WgradConvSpec`, `DgradConvSpec`); the forward spec above has no counterpart. It stores the LDS tile K-outer as `T[k][mn]` and transposes inside the MFMA operand fetch instead of transposing on store. Its row stride does not come from `lds_k_pad`: it is fixed at `_KOUTER_PAD` — 8, or 0 on the wgrad async path — and wgrad rejects an explicit `lds_k_pad` under `lds_k_outer` rather than silently ignoring it. dgrad keeps its A tile M-outer, so `lds_k_pad` still applies there; dgrad instead rejects an explicit `lds_layout` or `async_dma`. Do not set the field by hand: ask `WgradConvSpec.default_lds_k_outer(...)` / `DgradConvSpec.default_lds_k_outer(...)`. The backward sections below cover the layout, the pad, and the gate.

The example bake-off (`example/ck_tile/dsl/08_bake_off_implicit_gemm`) uses:

```text
tile_m=64, tile_n=64, tile_k=64,
warp_m=2, warp_n=2,
warp_tile_m=32, warp_tile_n=32, warp_tile_k=16,
pipeline="mem", epilogue="cshuffle"
```

and was validated end-to-end on gfx950 during this docs pass.

### A Descriptor: `(m, k) -> NHWC`

The cleanest part of the implicit-GEMM authoring surface; the transform DAG maps the implicit-GEMM (m, k) row/column to the NHWC linear offset and emits the padding validity predicate.

`make_a_descriptor(N, Hi, Wi, C, K, Y, X, Ho, Wo, sH, sW, pH, pW, dH, dW)`:

```python
TensorDescriptor.naive("A_nhwc", lengths=[N, Hi, Wi, C],
                       coord_names=["n", "hi", "wi", "c"])
  .transform(
      unmerge("m",  into=["n", "ho", "wo"], dims=[N, Ho, Wo]),
      embed (["ho", "y"], "hi", strides=[sH, dH], offset=-pH, lo=0, hi=Hi),
      embed (["wo", "x"], "wi", strides=[sW, dW], offset=-pW, lo=0, hi=Wi),
      unmerge("k",  into=["y", "x", "c"],   dims=[Y, X, C]),
      pad   ("y", lo=0, hi=Y),
      pad   ("x", lo=0, hi=X),
  )
```

At use site:

```python
off_elements, valid = A_desc.offset(b, m=m_val, k=k_val)
off_bytes = b.mul(off_elements, b.const_i32(2))      # fp16 -> 2 bytes
safe = b.select(valid, off_bytes, b.const_i32((1 << 31) - 1))
v = b.buffer_load_vN_f16(a_rsrc, safe, c0, dwords=2)
```

`pad("y")` and `pad("x")` matter when `K_gemm` does not cleanly divide the K-tile: without them, the unmerge would compute valid-looking offsets outside the intended filter slice.

### B and D Descriptors

`make_b_descriptor` maps `(k_out, k_gemm) -> KYXC`:

```text
unmerge(k_gemm -> y, x, c), pad(y), pad(x)
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

### Backward Passes (wgrad / dgrad)

Source: `instances/common/conv_implicit_gemm_wgrad.py` and `instances/common/conv_implicit_gemm_dgrad.py`.

Both backward directions reuse the forward implicit-GEMM skeleton (tile loop, LDS staging, MFMA phase, epilogue) over different GEMM mappings. `kpg`/`cpg` are the per-group filter and channel counts (`K`/`C` when `groups == 1`):

```text
wgrad:  M = kpg,         N = Y*X*cpg,    K_red = N*Ho*Wo
        A = dY (NHWK),   B = X (NHWC),   D = dW (KYXC)

dgrad:  M = N*Hi*Wi,     N = cpg,        K_red = Y*X*kpg
        A = dY (NHWK),   B = W (KYXC),   D = dX (NHWC)
```

### K-Outer LDS Tile

The M-outer tile transposes **on store**. When the stride-1 global axis is the GEMM *free* axis rather than the reduction axis, the loader runs with `vector_axis="row"` and `_store_tile` emits one `ds_write_b16` per element: a `load_vec`-wide global load becomes `load_vec` narrow LDS writes plus their address arithmetic.

`lds_k_outer=True` stores the tile as `T[k][mn]` instead, contiguous in LDS along the same axis the global load is contiguous in. Per chunk, with `n` the per-lane operand length:

```text
store side (removed):   load_vec x smem_store_vN(n=1)      # ds_write_b16
                      + (load_vec - 1) address adds
                      + load_vec   vec_extracts
store side (added):     1 x smem_store_vN(n=load_vec)      # b128 for a 16-bit
                                                           # 8-wide vector

read side (was):        1 wide smem_load_vN per operand fragment
                          b64  for n = 4
                          b128 for n = 8
read side (now):        n/4 x ds_read_tr16_b64
                      + (n/4 - 1) vec_concat
```

Net read-side cost is `+1` read per fragment for the `n = 8` atoms (`32x32x16`, `16x16x32`) and **exactly zero** for the `n = 4` atoms (`16x16x16`, `32x32x8`), which trade one `ds_read_b64` for one `ds_read_tr16_b64`.

The global `buffer_load`s are **unchanged** by the flip: `choose_vec` tests `tile_rows` in `"row"` mode and `tile_cols` in `"col"` mode — the same extent against the same product — so the load width is invariant. Only the LDS store changes.

The removed writes were bank-degenerate by construction. In `"row"` mode the decode is `row = (vec_idx % rows_per_vec) * load_vec`, so adjacent lanes step the tile by `load_vec` **rows** and the inter-lane dword delta is `load_vec * (block_k + k_pad) / 2`. At the 8-wide 16-bit load and the admissible `(block_k, k_pad)` that is an exact multiple of the 32-dword bank period. `lds_k_pad` cannot fix it: that pad is derived for a row step of 1, which is the read path.

### `_KOUTER_PAD`

The K-outer row stride must **not** be a multiple of the 32-dword LDS bank period, or the transpose read degenerates. The row-walking lane term differs per regime but the condition does not: on wave64 it is `((l % 16) // 4)`, four row-groups; on wave32 it is `(l % 8)`, eight. Either contributes zero bank spread whenever `(stride_elems * 2 / 4) % 32 == 0`. A pad of 8 makes a 64-wide tile 36 dwords (`36 % 32 == 4`), spreading the row-groups across banks (`0, 4, 8, 12` on wave64; `0, 4, ..., 28` on wave32), and keeps rows 16-byte aligned for the `b128` store side.

The two directions carry the constant differently. dgrad fixes it at 8 as a module constant shared by the builder and `is_valid_dgrad_spec` so the charged and allocated shapes cannot drift (mirrored as `ROCKE_DGRAD_KOUTER_PAD`). wgrad derives it per build as `0 if spec.async_dma else 8`: the direct global→LDS load deposits packed lane-contiguous bytes and cannot skip a row pad. That is not free on the read side — a pad of 0 puts the row stride back at a whole multiple of the bank period, which is the degenerate case above, so the async leg gives up exactly the bank spread K-outer was introduced to obtain. `async_dma` trades read-side bank spread for the write-side and prefetch win of the direct path, which is why it is a swept axis rather than a deduced one.

### Fragment Length Is Per-Atom

`n` in `_tr_frag` is the per-lane operand length — `op.a_frag_len` for the A fragment, `op.b_frag_len` for B — not a constant: on wave64 it is 8 for `32x32x16` and the MFMA `16x16x32`, 4 for `16x16x16` and `32x32x8`; on wave32 the WMMA `16x16x32` carries 16. It sets the k-stride between lane groups (`k = (l // MN)*n .. +n-1`). Hardcoding it at 8 made the `16x16x16` atom read k rows 8..27 of a 16-row tile — past the end — returning garbage. Both directions reject a fragment length that is not a multiple of the width the transpose read returns per lane — 4 for `ds_read_tr16_b64` (wave64), 8 for `ds_load_tr16_b128` (wave32) — since a non-multiple builds the fragment from an empty or truncated `parts` list. dgrad checks its single flipped operand (`b_per_lane`); wgrad flips both and checks `a_per_lane` and `b_per_lane`.

### Which Operands Flip

- **wgrad — both A and B.** Neither operand has a stride-1 reduction axis: dY is contiguous in `k_out` and X in inner `C`, both of which are the GEMM free axis for their side. Both paid the scatter, so both tiles flip.
- **dgrad — B only.** A (dY, NHWK) already has a stride-1 reduction axis (`k_out` innermost in `k_dg`), so its loader is already `vector_axis="col"` with one wide `smem_store_vN`, and its M-outer fragment read is already conflict-free via `lds_k_pad`. Flipping A would put the global vector along `m = (n, hi, wi)`, stride `K` in NHWK, destroying coalescing for zero write-side gain. B (W, KYXC) is the sole scatter: `c`, the GEMM free axis, is stride-1, forcing `axis_b="row"`.

### K-Outer Selection

Selection is a pure function of arch/dtype/atom and lives in one place per direction. There is no CLI flag and no env override; the deducers are keyword-only:

```python
WgradConvSpec.default_lds_k_outer(
    *, arch, dtype_a, dtype_b, warp_tile_m, warp_tile_n, wave_size=64
) -> bool

DgradConvSpec.default_lds_k_outer(
    *, arch, dtype_b, warp_tile_n, cpg, wave_size=64, pipeline="mem"
) -> bool
```

The dgrad deducer additionally takes `pipeline` and returns `False` for `pipeline == "wavelet"`: the wavelet loader does not implement the K-outer tile (the `validate()` gate rejects the pair), so the predicate keeps that combination off the sweep. A caller that omits the keyword gets the `"mem"` default and so never sees the exclusion — wavelet callers must pass their own pipeline.

Both predicates are the single selection point for `library/dispatch/grouped_convolution.py` and the sweep driver `benchmark/benchmark_implicit_gemm_conv.py`, rather than each keeping its own copy: dispatch reaches wgrad's through `_wgrad_lds_k_outer` and dgrad's through `_dgrad_lds_k_outer`.

The dgrad predicate is deliberately **asymmetric** — it keys on `dtype_b` / `warp_tile_n` only, never their A-side counterparts, because only B flips; copying wgrad's symmetric predicate would over-reject dgrad specs whose A side differs. It additionally carries `cpg`: the saving is proportional to the B load width, which collapses to 1 on an odd channel run. There `axis_b` is already `"col"`, there is no scatter to remove, and K-outer would be a small pure regression.

The `lds_k_outer` field itself still defaults to `False` on both specs, so existing goldens are unmoved.

### K-Outer Gating

The transpose read exists in two regimes -- `ds_read_tr16_b64` on gfx950 (wave64 MFMA, 4 elements per lane) and `ds_load_tr16_b128` on gfx1250 (wave32 WMMA, 8 per lane) -- so the field is gated in `validate()` plus the arch-aware `is_valid_*_spec`. `_LDS_K_OUTER_ARCH_WAVE` pins each arch to its wave size, since the lane mapping is derived per wave size. The two gates differ, in the same asymmetry as the deducers:

```text
wgrad (WgradConvSpec.validate + is_valid_wgrad_spec):
  arch in _LDS_K_OUTER_ARCH_WAVE  # {"gfx950": 64, "gfx1250": 32}
  wave_size == _LDS_K_OUTER_ARCH_WAVE[arch]
  dtype_a and dtype_b in (bf16, fp16)
  wave64: warp_tile_m and warp_tile_n in (16, 32)
  wave32: warp_tile (m, n, k) == (16, 16, 32)   # the only WMMA atom
  lds_k_pad is None            # rejected, not ignored
  async_dma implies lds_k_outer

dgrad (DgradConvSpec.validate + is_valid_dgrad_spec):
  arch in _LDS_K_OUTER_ARCH_WAVE  # {"gfx950": 64, "gfx1250": 32}
  wave_size == _LDS_K_OUTER_ARCH_WAVE[arch]
  dtype_b in (bf16, fp16)      # A keeps the ordinary _emit_smem_load
  wave64: warp_tile_n in (16, 32)
  wave32: (warp_tile_n, warp_tile_k) == (16, 32)  # the only WMMA atom
  lds_layout is None           # rejected, not ignored
  async_dma is False
  pipeline != "wavelet"        # rejected, not ignored
```

The deducers mirror the arch/dtype/atom/wave half of each gate; the dgrad deducer adds `cpg % 2 == 0`, and neither deducer tests `family` (the arch/wave pair already implies it — `gfx950` is wave64 MFMA, `gfx1250` wave32 WMMA). The arch check lives only in `is_valid_*_spec` — `validate()` has no arch to check against — which is the whole point of the split: without it an older target builds cleanly and emits an instruction the assembler rejects far from the cause.

On the dgrad path the two rejections have different reasons. `lds_layout` is rejected because the B shape is computed straight from `(block_k, block_n + _KOUTER_PAD)` and never consults the layout object, so honouring an explicit one would be a silent lie. `async_dma` is rejected because the tilde builder has no direct global→LDS path at all.

**gfx1250 (wave32 WMMA).** Supported as a second regime. The IR op is shared and
`core/isa/backend.py` selects `ds_load_tr16_b128` (`.v8bf16` / `.v8f16`), so no new
primitive was needed; what differs is the lane mapping. The read returns 8 per
lane, so gfx1250's 16-element fragment is two reads. `_tr_frag` branches on
`spec.wave_size`; the wave32 regime admits only the `16x16x32` atom.

The distinction that matters is between the layout a lane must **end up holding**
and the address it must **supply**. The result layout is the documented WMMA B
one: lane `l` holds column `l % 16` and K-half `l // 16`. But
`ds_load_tr16_b128` transposes an 8x8 element block *within each group of 8
lanes* -- the 8 lanes of a group each read 8 contiguous elements, and lane `j` of
the group receives element `j` from all 8 of those runs. So the group addresses
the 8-column block containing its target column, and each lane supplies a
different **K row** of that block rather than its own column:

```
col  = mn_base + ((l % 16) // 8) * 8
row0 = k_base  + (l // 16) * n + (l % 8)     # read r at row0 + 8*r
```

Addressing it as though the instruction returned a straight run of `n` K values
at the lane's own column (`col = mn_base + l % 16`, `row0 = k_base + (l // 16) * n`)
reads a transposed operand. That form was byte-identical across both engines and
still numerically wrong -- two engines agreeing on the same wrong formula is
still wrong.

gfx950's `((l % 16) // 4)` and `(l % 4) * 4` terms come from the different
grouping of `ds_read_b64_tr_b16` under 64 lanes.

**Verified on gfx1250 hardware**: the K-outer path is bitwise identical to the
M-outer default for dgrad, and matches the fp32 reference for wgrad.

### Backward Knob Changes

- `pipeline="basic"` (CK `pipeline_basic`): single LDS buffer, global-read / compute overlap. wgrad and forward only; dgrad has no `basic` branch.
- `async_dma` is a swept axis on the wgrad sweep driver, not a run-level flag. Unlike `lds_k_outer` it is not deducible from `(arch, spec)`: it removes the register staging of the tile, but it also forces the K-outer row pad to 0 and coarsens the load-width ladder to the widths the intrinsic accepts, and both terms are functions of tile width and channel run, which are themselves sweep axes.
- `_MAX_UNROLLED_K_ITERS = 128` (wgrad) now caps **both** statically-unrolled loops, `pipeline="basic"` and `async_dma`. It previously guarded only `"basic"`, leaving async uncapped: a deep reduction at a low split-K degree then unrolled five figures of load+MFMA bodies into one kernel and exhausted host memory during the IR build rather than failing validation. Mirrored as `ROCKE_MAX_UNROLLED_K_ITERS` in the C engine.
- Removed: the `--lds-k-outer`, `--lds-k-pad` and `--dtype-d` CLI flags and the `ROCKE_WGRAD_LDS_K_OUTER` env override. All replaced by deduction or dropped.

## Direct Grouped Convolution

Source: `instances/common/conv_direct_grouped.py`.

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
A NHWC, B KYXC (grouped), D NHWK
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

Key levers (per `optimization/runbook_compliance.md`):

- wide direct epilogue (1 `buffer_store_dwordx2` per lane = 4 halves) -- the
  largest of the three;
- K=32 MFMA fold;
- `BLOCK_GROUPS=4` -- a small increment on top of the other two.

Measured numbers are not recorded here; see `platform/AGENTS.md` "Compliance".

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

- vec2-dword epilogue (1 store/lane, 4 halves fused).

This path avoids the implicit-GEMM LDS machinery because the channel group is tiny and direct vectorization is cleaner.

## Img2Col

Source: `instances/common/img2col.py`.

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

Source: `instances/common/pooling.py`.

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

- Missing `pad("y")` / `pad("x")` in implicit K tails: numerically valid-looking but cross-slice offsets.
- Descriptor returns element offsets but buffer op expects byte offsets: shift left by 1 for fp16, by 2 for bf16, etc.
- False lanes are masked **after** a faulting pointer load — use the buffer-rsrc sentinel pattern instead.
- Async loader writes lane-contiguous LDS but consumer assumes padded / swizzled physical layout.
- K-packed MFMA fold packs S/C channels in the wrong order (close but not bit-correct).
- Direct conv circular accumulator slot is not reset after store.
- Output descriptor uses NHWC instead of NHWK stride order.
- Benchmark compares implicit-GEMM graph mode to direct per-launch mode without labeling launch overhead.
