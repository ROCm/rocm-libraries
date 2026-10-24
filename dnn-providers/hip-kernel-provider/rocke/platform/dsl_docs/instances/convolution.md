# Convolution Instances

This page covers:

- `library/kernels/common/conv_implicit_gemm.py`
- `library/kernels/common/conv_direct_grouped.py`
- `library/kernels/common/conv_direct_nongrouped.py`
- `library/kernels/common/conv_direct_wgrad_tiled.py`,
  `library/kernels/common/conv_direct_dw_wgrad.py`,
  `library/kernels/common/conv_direct_wgrad_gather.py` (direct wgrad)
- `library/kernels/common/img2col.py`
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

Direct non-grouped (groups == 1):
  LDS-staged, halo-inclusive input tile shared by all KH*KW taps
```

## Implicit-GEMM Convolution

Source: `library/kernels/common/conv_implicit_gemm.py`.

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

Kernel ABI: conv kernels are AOT, so the problem shape travels as kernargs.
The ordered argument list is owned by the conv instances, not by the platform
helpers: `kernels.common.conv_abi` in the library (`conv_arg_names`,
`conv_args_signature`, `conv_direct_args_signature`) and its C++ twin
`rocke/instance_conv_abi.h`. Every direction opens with the same six entries,
followed by the runtime problem block (extents, strides, magic-division
pairs) and any direction-specific extras:

```text
A: ptr<f16, global>      8 bytes
B: ptr<f16, global>      8 bytes
D: ptr<f16, global>      8 bytes
A_bytes: i32             4 bytes   # buffer rsrc bound
B_bytes: i32             4 bytes
D_bytes: i32             4 bytes
p_N, p_Hi, p_Wi, ...: i32          # runtime problem block (see conv_abi)
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

`unroll_k=True` double-buffers the K-loop over two LDS tile pairs. The trip count is a kernel argument (AOT), so the loop body is unrolled twice and steps by `2 * tile_k`, binding each phase to a build-time buffer; with an odd tile count the second phase of the last step reads a zero tile (under wgrad split-K it is redirected to `wg_K`, which zero-fills, instead of the next slice's first tile). The kernel name carries an `unroll` tag.

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
 7. Enter the K_gemm tile loop (runtime trip count; double-buffered for `unroll_k` / `async_dma`).
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

Both predicates are the single selection point for the sweep driver `benchmark/benchmark_implicit_gemm_conv.py`, so no caller keeps its own copy. (The grouped-convolution dispatcher, `library/dispatch/grouped_convolution.py`, registers fixed lists of forward and dgrad AOT instances whose names already record `lds_k_outer`; it derives no layout itself.)

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

- `pipeline="basic"`: now uses a bounded runtime `scf.for` K-loop (load → sync → MFMA → sync per tile), identical to the `"mem"` path. No K-trip-count limit. dgrad has no `basic` branch.
- `async_dma` is a swept axis on the wgrad sweep driver, not a run-level flag. Unlike `lds_k_outer` it is not deducible from `(arch, spec)`: it removes the register staging of the tile, but it also forces the K-outer row pad to 0 and coarsens the load-width ladder to the widths the intrinsic accepts, and both terms are functions of tile width and channel run, which are themselves sweep axes.
- No K loop is unrolled at build time any more: `async_dma` runs `SoftwarePipeline.run_ping_pong_dynamic` and `unroll_k` a hand-rolled 2x ping-pong, both over the runtime extent, so the former `_MAX_UNROLLED_K_ITERS` / `ROCKE_MAX_UNROLLED_K_ITERS` cap is gone.
- wgrad `split_k`: `1` = no split (direct store); any value `> 1` builds the one split-K kernel (name tag `spk`), whose degree is the `ks_count` / `ks` kernargs chosen at launch. There is no `split_k=0` "runtime degree" encoding.
- Removed: the `--lds-k-outer`, `--lds-k-pad` and `--dtype-d` CLI flags and the `ROCKE_WGRAD_LDS_K_OUTER` env override. All replaced by deduction or dropped.

## Direct Grouped Convolution

Source: `library/kernels/common/conv_direct_grouped.py`.

These are specialized kernels for grouped direct convolution bake-off cases (`cpg=kpg in {16, 4}`), not generic implicit-GEMM conv.

### `DirectConvProblem`

Verified from `instances/conv_direct_grouped.py`:

```python
@dataclass(frozen=True)
class DirectConvProblem:
    N: int
    H: int           # input height
    W: int           # input width
    groups: int
    cpg: int         # channels per group
    kpg: int         # filters per group (= cpg in bake-off)
    KH: int = 3      # kernel height (not R)
    KW: int = 3      # kernel width  (not S)
    PAD: int = 1
    stride: int = 1
```

Derived output spatial properties (read-only):

```python
Ho = (H + 2*PAD - KH) // stride + 1
Wo = (W + 2*PAD - KW) // stride + 1
```

`flops` is computed from `Ho × Wo` (output positions), not `H × W`, so stride > 1
reports correctly.

Note this layout is different from `ConvProblem`:

- `H`/`W` are input spatial; `Ho`/`Wo` are derived output spatial;
- `KH`/`KW` not `R`/`S`;
- single `PAD` and `stride` ints (no separate `pH`/`pW`/`sH`/`sW`);
- dilation is `dil_h`/`dil_w` (default 1; `Ho`/`Wo` use the dilated extent
  `(KH-1)*dil_h + 1`). Only the non-grouped forward kernel computes a dilated
  convolution; every other variant rejects one (`dilation_reason`).

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

Uses `mfma_f32_4x4x4_f16` (fp16) or `mfma_f32_4x4x4_bf16` (bf16, the CDNA2+
`_1k` intrinsic, selectable on gfx942 and gfx950): one wave computes 16
independent 4x4x4 matmuls indexed by `batch = lane / 4`. The kernel name carries
a `bf16` flag for bf16 I/O.

Algorithm:

```text
1. Pack multiple groups across the 16 wave batches.
2. For each output coordinate assigned to the wave:
   - Load needed input vectors with padding masks (no LDS row pipeline).
   - Load 4-channel weights.
   - Issue mfma_f32_4x4x4_{f16,bf16}.
3. Accumulate across KH*KW.
4. Vector-store the 4 output channels as one contiguous buffer_store_vN_{f16,bf16}.
```

Levers:

- vec2-dword epilogue (1 store/lane, 4 halves fused).

This path avoids the implicit-GEMM LDS machinery because the channel group is tiny and direct vectorization is cleaner.

#### 4c dgrad (`make_dgrad_4c_spec`, AOT variant `direct_grouped_dgrad_4c`)

Stride-1 dgrad with `cpg = kpg = 4` and a "same"-padded 1x1 or 3x3 filter runs
the 4c kernel as a *forward* kernel over the transposed problem
(`dgrad_4c_transposed_problem`: `dX = conv(dY, W_T)`), launched with the
forward direct ABI over that problem. Knobs (all default off, so a forward 4c
spec is unchanged):

- `dgrad_fused_weights`: `B` is the original weight; the prologue gathers each
  lane's flipped, k<->c transposed fragment (four scalar loads per tap), so
  there is no weight pre-pass or workspace.
- `dgrad_weights_lds` (with fused weights; needs `ds_read_b64_tr_b16`,
  gfx950): the workgroup's W slice is copied to LDS with 16-byte loads and each
  tap's fragment is one transpose read. Bounded by `DCONV4C_MAX_WL_PASSES`.
- `stage_rows` / `waves_q` (with LDS weights; `block_q == 4 * waves_q`): once
  per input row the workgroup copies its row (`block_q + KW - 1` columns of
  `block_groups * cpg` channels) into a double-buffered, padded LDS row, one
  row ahead through registers, and every wave reads its B fragments with
  `ds_read_b64`. The rows stream through a runtime `scf.for` from `PAD` to
  `p_Hi + KH - 1`, unrolled `lcm(KH, 2)` rows so the buffer parity and the
  accumulator slots stay build-time constants; rows outside the image load
  zeros through the out-of-range sentinel and drop their stores, and the row
  and pixel offsets come from the stride kernargs. Bounded by
  `DCONV4C_MAX_ROW_PASSES`, `DCONV4C_MAX_THREADS` and the target's LDS.

The kernel name carries `fw` / `fwl` / `sr<waves_q>`. The AOT cache builds the
policy image (`block_q` 4/8/16, `block_groups` 16/32/64, gathered, LDS-staged
and row-staged weights) for both dtypes; `plan_for` emits one forward-ABI
launch with the buffers `(dy, w, dx)`.

Not dispatched yet (the grouped-convolution dispatcher is being reworked
separately); `benchmark_direct_conv --run-from-cache <cache> --direction dgrad
--variants direct_grouped_dgrad_4c --verify` runs the cached binaries on a
shape. Scope: `cpg = kpg = 4`, a "same" 1x1/3x3 filter and
`groups % 16 == 0`. The row-staged kernel (`dgrad_weights_lds` +
`stage_rows`) pays on larger grids and taller images; on small grids the
fused generic kernel (next section), which tiles H, or implicit GEMM is the
better choice. Case studies:
[`dgrad_4c_bf16_case_study.md`](../../python/rocke/examples/gfx950/conv_dgrad/dgrad_4c_bf16_case_study.md),
[`dgrad_fused_weights_case_study.md`](../../python/rocke/examples/gfx950/conv_dgrad/dgrad_fused_weights_case_study.md).

### Generic Kernel dgrad: fused weights (`DirectConvSpec`, AOT variant `direct_grouped_dgrad_fused`)

The generic `DirectConvSpec` row stream (any `cpg` that is a multiple of 4,
`mfma_f32_16x16x16`, or `16x16x32` with `fold_k32`) also computes stride-1
dgrad as a *forward* kernel over the transposed problem
(`make_dgrad_fprop_spec`: `cpg' = kpg`, `kpg' = cpg`, `PAD' = KH-1-PAD`),
launched with the forward direct ABI over that problem. The MFMA dgrad
pipeline (`direct_grouped_dgrad_mfma`) runs it after a weight pre-pass; these
knobs make it a single kernel (all default off, so every existing generic
kernel is unchanged):

- `preload_weights` (`waves_k == 1`): every weight fragment is loaded once in
  the prologue and kept in registers for the whole row loop.
- `dgrad_fused_weights`: `B` is the original weight `W[groups*cpg', KH, KW,
  kpg']` (transposed-problem terms); the prologue gathers each fragment with
  flipped taps and k<->c transposed addressing, so there is no pre-pass and no
  workspace. Every weight stride is a product of the baked `KH`, `KW` and
  `kpg`. Implies the preload; bounded by `PRELOAD_WEIGHT_VGPR_BUDGET`
  (`preload_weight_vgprs`).
- `dgrad_weights_lds` (needs `ds_read_b64_tr_b16`, gfx950): the workgroup's
  raw `W` slice is copied to LDS with 16-byte loads and each fragment is one
  (two with `fold_k32`) transpose read; the slice shares an LDS pool slot with
  the first row buffer (`direct_conv_lds_bytes`). Bounded by
  `DGRAD_WEIGHTS_LDS_BUDGET`.
- `waves_per_eu`: the `amdgpu-waves-per-eu` hint.
- `row_guard` (preloaded weights): the row loop is unrolled over a tile of
  `block_h` output rows (`block_h > 0` is mandatory: the image height is a
  kernarg), so a tile taller than the image -- a "whole image" pick mapped to
  a cached tile height (`dgrad_fused_block_h`) -- would multiply zeros for
  every input row past `p_Hi`. With `row_guard` each such row's MFMAs run in a
  block-uniform one-trip loop carrying the accumulators it touches; its LDS
  staging, barrier and flush are unchanged.

Row-stream knobs (also default off; like the knobs above, mirrored field for
field by the C++ engine's `rocke_build_direct_conv` and byte-identity gated):

- `prefetch_rows` (0..3): input row `y + prefetch_rows` is loaded into
  registers while row `y` is computed and published to LDS at the end of row
  `y + prefetch_rows - 1`; 0 and 1 are the one-row-ahead schedule.
- `lds_only_sync`: the row barrier drains LDS only, so prefetched loads and
  the row's output stores stay in flight across it.
- `waves_m` (preloaded or fused weights): waves per group along the output
  M-tiles; each keeps only its share of the weights and accumulators
  (`preload_weight_vgprs` divides accordingly) and all share the staged row.
- `lds_pad` (multiple of 8): extra elements per staged input column, so the
  16 q-lanes of a fragment read spread over the LDS banks.
- `stage_out` (`kpg % 16 == 0`): each finished output row goes through a
  double-buffered LDS tile and is stored with 16-byte lanes over the
  workgroup's contiguous channel span; the offsets come from the output
  stride kernargs.
- `xcd_tiles` (gfx950 only): the launch-order workgroup id is remapped with
  `chiplet_transform_chunked_runtime` so the `q_tiles * group_tiles`
  workgroups of one image row band run on one XCD; the grid and the chunk are
  computed from the kernargs, only the XCD count is baked. It needs at least
  `XCD_TILES_MIN_ROW_BANDS` row bands (`N * ceil(H / block_h)`), a
  `validate()` rule that `plan_for` re-checks on the real problem.
- `row_exit`: the workgroup leaves once the next output row to flush is past
  `p_Hi` (one block-uniform scalar check per row from the first flush on),
  so a tile taller than the image skips every dead row -- loads, staging,
  barriers and masked flushes -- where `row_guard` only skips their MFMAs.

`direct_conv_lds_bytes` accounts for the column pad and the output tiles;
`direct_conv_grid` is the launch grid.

The kernel name carries `pw` / `fw` / `fwl` / `we<n>` / `pf<n>` / `lso` /
`wm<n>` / `xc` / `lp<n>` / `so` / `rg` / `rx`.
`direct_dgrad_spec_for_problem` (kernel-level helper, not the dispatch policy)
returns the 4c spec for `cpg = kpg = 4` and this spec otherwise
(`is_valid_dgrad_fused_problem`: stride 1, "same"-padded square filter, both
channel counts multiples of 4); `direct_dgrad_launch` / `build_direct_dgrad`
launch and build it.

The AOT cache variant `direct_grouped_dgrad_fused` covers 1x1/3x3/5x5/7x7
"same"-padded filters at stride 1 with `cpg`, `kpg` in 4..32 (in steps of 4,
`cpg != kpg` included; 5x5/7x7 only where the fused weights fit the register
budget) and carries the image of the generic direct dgrad policy rather than a
product: `block_q` 16 (32 on 4/8-row tiles with wide channel groups),
`block_groups` up to the per-channel rule (1 from 32 channels, 2 from 16, else
4; halved for 5x5/7x7), tiles of 4/8/16/32 rows (4/8 for 5x5/7x7), every tile
with the row guard, `fold_k32` where `kpg % 32 == 0` and the target has the
atom, LDS-staged weights where valid (gathered otherwise) and `waves_per_eu=4`
on the 16-wide-atom forms with a small weight footprint. On gfx950 square
channel groups on 16-column strips (where `waves_m = 2` is not valid) also get
the policy's row-stream stack: `prefetch_rows=2`, `lds_only_sync`,
`xcd_tiles` and the (`lds_pad`, `stage_out`) pairs the policy chooses among
(padded and staged, padded, neither). The 16- and 32-row tiles, which the
policy reaches only through whole-image picks, also carry `row_exit`.
`plan_for` emits one forward-ABI launch with the buffers `(dy, w, dx)`.

Not dispatched yet (the grouped-convolution dispatcher is being reworked
separately); `benchmark_direct_conv --run-from-cache <cache> --direction dgrad
--variants direct_grouped_dgrad_fused --verify` runs the cached binaries on a
shape. The cache-scope rules of the knob image are `_dgrad_fused_policy_reason`
in `library/kernels/conv/direct_aot_jobs.py`; the validator still accepts what
they skip, so a JIT sweep can try it. Strided problems, filters past 7x7,
channels past 32 and 5x5/7x7 pairs over `PRELOAD_WEIGHT_VGPR_BUDGET` have no
fused binary. Case studies:
[`dgrad_fused_weights_case_study.md`](../../python/rocke/examples/gfx950/conv_dgrad/dgrad_fused_weights_case_study.md),
[`grouped_generic_row_stream_case_study.md`](../../python/rocke/examples/gfx950/conv_dgrad/grouped_generic_row_stream_case_study.md).

### Depthwise Kernel (`DirectDepthwiseSpec`)

`DirectDepthwiseSpec` / `build_direct_depthwise`. Scalar FMA, no MFMA. Each lane owns
one absolute channel for the full kernel.

```python
@dataclass(frozen=True)
class DirectDepthwiseSpec:
    problem: DirectConvProblem
    name: str = "direct_depthwise"
    block_w: int = 8       # output W positions per workgroup
    block_waves: int = 1
    wave_size: int = 64
```

Constraints (Python `is_valid_depthwise_spec` / C++ `rocke_direct_depthwise_is_valid_spec`):

- `cpg == kpg == 1`
- No divisibility requirement on `groups` — partial channel tiles are
  handled by `ch_in_range = ch < groups` predicate on loads and stores.
- `stride >= 1` supported; output uses `D[N, Ho, Wo, total_k]`.

Grid: `(ceil(W / block_w), ceil(groups / block_ch), N)` where
`block_ch = block_waves * wave_size`.

H-loop: the input height is a kernel argument (AOT), so the rows stream
through a runtime `scf_for_iter` that takes `KH` input rows per iteration with
`KH × block_w` loop-carried f32 accumulators; there is no build-time-unrolled
form, since it would bake the height into the trip count.

Flush condition:

```text
p_flush_val = y - (KH - 1)
if 0 <= p_flush_val < H and p_flush_val % stride == 0:
    ho_row = p_flush_val // stride
    if ho_row < Ho:          # bounds guard added for stride > 1
        store to D[n, ho_row, w_out, ch]
```

### Depthwise Spatial Kernel (`DirectDepthwiseSpatialSpec`)

`DirectDepthwiseSpatialSpec` / `build_direct_depthwise_spatial`. For small group
counts (`groups <= wave_size`); maps both channel and output W-position onto a
single wavefront.

```python
@dataclass(frozen=True)
class DirectDepthwiseSpatialSpec:
    problem: DirectConvProblem
    name: str = "direct_depthwise_spatial"
    block_waves: int = 1
    wave_size: int = 64
```

Thread layout within each wavefront:

```text
ch        = tid % groups          # which channel this thread owns
w_in_wave = tid // groups         # W-position offset within the wave
n_w_per_wave = wave_size // groups
block_w   = block_waves * n_w_per_wave
```

Thread utilisation: `floor(wave_size/groups) * groups / wave_size`. For
`groups=3, wave_size=64`: 63/64 = 98.4%.

Constraints (`is_valid_depthwise_spatial_spec`):

- `cpg == kpg == 1`
- `groups <= wave_size`
- `n_w_per_wave > 0` (i.e. `groups < wave_size`)

Grid: `(ceil(Wo / block_w), 1, N)` — no channel tile (all channels handled
within one wavefront via the spatial thread mapping).

Streams the input rows through the same runtime `scf_for_iter` as
`DirectDepthwiseSpec`, but accumulates over a single output W position
(no `block_w` outer loop). `stride >= 1` supported via `D[N, Ho, Wo, total_k]`.

Parity gate: configs 10 (stride=1, groups=3) and 11 (stride=2, groups=3) in
`tests/instances/parity/conv_direct_grouped_emit.{c,py}`.

### Depthwise Tiled Kernel (`DirectDepthwiseTiledSpec`)

`DirectDepthwiseTiledSpec` / `build_direct_depthwise_tiled`. Output-stationary
depthwise forward for any group count and filters up to 31x31.

```python
@dataclass(frozen=True)
class DirectDepthwiseTiledSpec:
    problem: DirectConvProblem
    name: str = "direct_depthwise_tiled"
    block_w: int = 8        # output columns per block
    block_h: int = 4        # output rows per block
    block_waves: int = 1    # waves per block (64 channels each)
    unroll_rows: bool = False
    wave_size: int = 64
```

The row-streaming `DirectDepthwiseSpec` holds all `KH x KW` weights and `KH`
accumulator slots per column and unrolls `KH` rows per loop iteration: its
registers and loop body grow with `KH^2 * KW`, so past 11x11 it spills heavily
and compiles slowly. This kernel is linear in the filter width:

- a block owns a `block_h x block_w` output tile; the grid also splits the
  output rows: `(ceil(Wo/block_w), ceil(C/block_ch), N * ceil(Ho/block_h))`;
- a runtime loop walks the filter rows `r`, holding one weight row (the next
  one prefetched) and the input rows `ho*stride - PAD + r` of the tile;
- with stride 1, iteration `r+1` reads the rows of iteration `r` shifted by
  one, so `block_h - 1` rows are carried and one row is loaded per iteration;
  stride `s` splits the filter rows into `s` phases with a window each;
- `unroll_rows` unrolls the (baked) filter-row loop so the scheduler overlaps
  rows; the validator caps the unrolled body at `_DW_TILED_UNROLL_MAX_FMAS`.

Outputs are indexed directly, so it is not tied to "same" padding. Constraints
(`is_valid_depthwise_tiled_spec` / `rocke_direct_depthwise_tiled_is_valid_spec`):
`cpg == kpg == 1`, `KH, KW <= 32`, `stride <= KH`, no dilation, `Ho, Wo >= 1`,
and `live_regs = block_h*block_w + block_h*n_cols + 2*KW <= 256`
(`n_cols = (block_w-1)*stride + KW`). The register budget and the unroll cap
keep both rolled and unrolled configurations within a bounded register and
code-size footprint.

The sweeps skip a row-streaming config that would spill
(`depthwise_stream_register_reason`) and leave its shape to this kernel.

Parity gate: configs 57-62 in `library/tests/parity/conv_direct_grouped_emit.{c,py}`.

### Windowed Depthwise Dgrad Kernel (`DirectDepthwiseDgradWindowedSpec`)

`DirectDepthwiseDgradWindowedSpec` / `build_direct_depthwise_dgrad_windowed`
(C++ `rocke_build_direct_depthwise_dgrad_win`). Backward-data for depthwise
convolution (`cpg == kpg == 1`) at stride 1, any padding (rows are bounded by
`p_Ho`, columns by `p_Wo`). AOT cache variant `direct_depthwise_dgrad_win`;
not dispatched yet (the grouped-convolution dispatcher is being reworked
separately), so it is reached through the cache (`--run-from-cache`).

```python
@dataclass(frozen=True)
class DirectDepthwiseDgradWindowedSpec:
    problem: DirectConvProblem
    name: str = "direct_depthwise_dgrad_win"
    block_w: int = 8        # dX columns per block (VALU form)
    block_waves: int = 1
    ch_per_lane: int = 1    # 1/2/4/8 adjacent channels per lane (vector I/O)
    block_h: int = 8        # dX rows per block: a tile height, must be > 0
    dot2: bool = False      # tap pairs on arith.fdot2 (gfx950 only)
    mfma: bool = False      # Toeplitz MFMA form (gfx950, groups % 8 == 0)
    w_fold: int = 1         # mfma: images per 16x16 tile (1, 2, 4)
    prefetch_rows: int = 1  # mfma: dY rows in flight
    wave_size: int = 64
```

The filter, the padding and the dtype are baked (the kernel name carries
`r<KH>s<KW>p<PAD>` and `bf16`); N, H, W and the channel count are kernargs.
`block_h` is a build-time tile height: the rows of a tile are unrolled, the
image height is the `p_Hi` kernarg and rows past it are guarded at run time,
so there is no whole-image (`block_h = 0`) form.

Algorithm of the VALU form (the rows of a tile and the taps are unrolled):

```text
for each local dY row y of the tile (ho = h0 + y + PAD - (KH-1)):
    load window[t] = dY[n, ho, wi0 + PAD - (KW-1) + t, ch..]   t < block_w + KW - 1
    (the next live row's window is issued before this row's FMAs; sched_barrier)
    for r < KH, s < KW, j < block_w:
        acc[(y + r - (KH-1)) % KH][j] += W[ch, r, s] * window[j + KW - 1 - s]
    flush dX local row y - (KH-1) from its slot
```

Addressing: each offset is a row-invariant lane part (channel + column, or
`DW_DGRAD_WIN_OOB_LANE` when the channel or column is out of range) plus a
block-uniform row part (or `DW_DGRAD_WIN_OOB_UNIFORM` for a row outside the
image). The two sentinels sum to `2**31 - 1`, so an invalid access lands past
the buffer (loads read zero, stores drop) with no per-tap select and the
signed i32 add never overflows. The tensors must therefore stay within
`DW_DGRAD_WIN_MAX_TENSOR_BYTES`, a plan-time rule (the IR does not depend on
it) that the validator checks on the real problem.

`dot2` pairs taps `(s, s+1)` into one `arith.fdot2`
(`v_dot2c_f32_{f16,bf16}`), with the window packed once per row; an odd KW
issues its tail pair `(w[KW-1], 0)` against `(x, 0)` first, so the zero
weight only ever meets a zero (an Inf one column outside the receptive field
cannot turn into a NaN).

Constraints (`is_valid_depthwise_dgrad_win_spec` /
`rocke_direct_depthwise_dgrad_win_is_valid_spec`, same reason text):
fp16/bf16, `cpg == kpg == 1`, `stride == 1`, no dilation, `block_h >= 1`,
`ch_per_lane in {1, 2, 4, 8}` dividing `groups`, `dot2` only with
`ch_per_lane == 1` and only on `DW_DGRAD_WIN_DOT2_ARCHES` (gfx950), at most
`DW_DGRAD_WIN_MAX_UNROLL` unrolled FMAs
(`block_h * KH * KW * block_w * ch_per_lane`), and `w_fold` /
`prefetch_rows` left at 1 without `mfma`.

Grid: `(ceil(W / block_w), ceil(groups / block_ch), N * ceil(H / block_h))`
with `block_ch = block_waves * 64 * ch_per_lane`.

**Toeplitz MFMA form (`mfma=True`, gfx950).** Each wave owns 8 channels and a
16x16 fp32 tile per dX row in flight: tile rows are (channel, column parity),
tile columns are column pairs of `w_fold` images (`tile_w = 32 / w_fold`
columns each). A one-hot diagonal weight fragment (A) meets 8 channels x 4
window columns (B) in one `mfma_f32_16x16x32_{f16,bf16}`, so a filter row takes
`ceil((KW + 1) / 4)` passes. dY window rows (16-byte coalesced loads,
`prefetch_rows` rows in flight) and finished dX rows go through two LDS double
buffers that share one barrier per row; the block writes whole 16-byte channel
runs per pixel. `block_w` and `ch_per_lane` do not apply (`ch_per_lane` must
be 1; `block_w` stays out of the kernel name). `block_ch = 8 * block_waves`,
grid `(ceil(W / tile_w), ceil(groups / block_ch), ceil(N / w_fold) *
ceil(H / block_h))`. Extra constraints: `groups % 8 == 0` (a plan-time rule),
`block_waves <= DW_DGRAD_MFMA_MAX_WAVES`, `w_fold in DW_DGRAD_MFMA_W_FOLDS`,
`prefetch_rows <= DW_DGRAD_MFMA_MAX_PREFETCH_ROWS`, at most
`DW_DGRAD_MFMA_MAX_UNROLL` unrolled MFMAs (`block_h * KH * passes`), the
resident-fragment estimate `mfma_frag_vgprs()` within
`DW_DGRAD_MFMA_MAX_FRAG_VGPRS`, and `mfma_lds_bytes()` within
`DW_DGRAD_MFMA_LDS_BUDGET`.

Non-finite semantics differ between the forms. The VALU forms (including
`dot2`) keep an Inf / NaN in dY inside its channel and receptive field. The
MFMA form multiplies zero off-diagonal weights against the other 7 channels of
the group, so a non-finite dY value reaches every channel of its 8-channel
group, in its receptive-field rows and over `4 * ceil((KW + 1) / 4)` columns
(the zero-weight taps of the passes; `0 * Inf = NaN`). Finite inputs agree
with the VALU form up to fp32 summation order.

AOT cache image (`library/kernels/conv/direct_aot_jobs.py`): the square
filters `_DEPTHWISE_FILTERS` with every padding; for the VALU form the `(block_w, block_h)` tiles of
`DW_WIN_TILES` with and without `dot2` at `DW_WIN_WAVES` waves and with
channel pairs at `DW_WIN_PAIR_WAVES`; for the MFMA form every
`DW_WIN_MFMA_FOLDS` x `DW_WIN_MFMA_WAVES` x `DW_WIN_MFMA_BLOCK_H` at
`prefetch_rows = DW_WIN_MFMA_PREFETCH`, filtered by `_dw_win_policy_reason`.

Choosing between the forms is left to the dispatcher rework. The MFMA form
was measured ahead of the `dot2` VALU kernel, with warm and with cold caches,
only inside a bounded region of "same"-padded large filters and moderate
image sizes; outside it, and on the memory-bound 3x3, the VALU form (with
`dot2` from 5-wide filters on gfx950, channel pairs on 3x3) is the starting
point. Stride 2 runs the row-streaming kernel (`direct_depthwise_dgrad`).

Parity gate: configs 0-19 of
`library/tests/parity/conv_direct_dgrad_ext_emit.{c,py}` (0-10 the VALU and
`dot2` forms, 11-19 the MFMA form, including the gfx942, `groups % 8 != 0` and
`block_h = 0` configs both engines reject). Case study:
[`depthwise_windowed_dgrad_case_study.md`](../../python/rocke/examples/gfx950/conv_dgrad/depthwise_windowed_dgrad_case_study.md).

## Direct Non-grouped Convolution

Source: `library/kernels/common/conv_direct_nongrouped.py`. Full write-up (every
optimization, rejected levers, gates):
`library/kernels/common/README_conv_direct_nongrouped.md`.

The `groups == 1` counterpart of the grouped direct kernels, which draw all
their parallelism from `groups` and degenerate to one wave per output row when
there is a single group. It shares `DirectConvProblem` and the AOT direct-conv
kernarg block `conv_direct_arg_names(direction="fwd")` (launch values from
`ConvArgs.from_problem(problem)`). Batch, extents, `C` (`p_total_c`), `K`
(`p_total_k`) and the strides are runtime; the filter, stride, `PAD`, dtype and
tile are baked. Unlike the grouped kernels it does not bake the channel counts.

```python
@dataclass(frozen=True)
class DirectNongroupedConvSpec:
    problem: DirectConvProblem      # groups must be 1
    tile_h: int = 16; tile_w: int = 32; tile_k: int = 128; ck: int = 16
    waves_m: int = 2; waves_n: int = 4
    atom: str = "32x32x16"          # 32x32x16 | 32x32x8 | 16x16x32 | 16x16x16
    lds_pad: int = 8
    chiplet_swizzle: bool = True; swizzle_wgm: int = 8
    double_buffer: bool = False
    iglp: int | None = None; waves_per_eu: int | None = None
```

Algorithm:

```text
for each channel chunk (runtime scf.for, accumulators + next-chunk loads carried):
    barrier; publish prefetched X (halo-inclusive, [pos][c + lds_pad]) and
             W (pre-swizzled to MFMA fragment order) into LDS; barrier
    issue global loads for chunk i+1          (software prefetch)
    for k_atom, s:  read X fragments by *input* row  ((ROWS_W-1)*stride + KH of them)
        for r:      read W fragments for tap (r, s)
                    mfma into acc[m_tile][row][col_block] from X row  row*stride + r
epilogue: each 4-channel accumulator quad -> one dwordx2 buffer store
```

Structural properties:

- activation fragments shared across the `KH` taps of a filter column;
- staging predication folded into the base offset before the loop (out-of-range
  buffer loads return zero), so the loop body has no compare/select;
- weights stored in fragment order, so the LDS store and read are both linear
  in `lane`; activation channel stride padded by `lds_pad` for conflict-free
  `ds_read_b128`;
- two barriers per chunk (one with `double_buffer`), `iglp_opt` leading the loop
  body when set;
- each LDS array carries a scratch tail for staging slots past the tile end;
- `validate()` caps accumulators at 256 registers per lane (larger tiles hang
  the backend scheduler rather than failing);
- the runtime channel-loop trip count is clamped to `max(C/ck, 1)` (a no-op
  after `validate()`): without the clamp LLVM keeps a zero-trip guard and
  schedules the prefetch loads late in the body.

Grid: `spec.grid() = (n_w_tiles * n_h_tiles * N * n_k_tiles, 1, 1)`; the kernel
derives the tile counts from its kernargs and decodes the block id through
`chiplet_aware_super_tile_dynamic` with the channel tile fast-varying.
Candidates: `nongrouped_specs(problem, arch)`;
`tile_w_candidates(Wo, atom_tile)` prefers widths that divide `Wo` exactly, and
the 16-wide atoms exist for widths that are not a multiple of 32. The AOT cache
builds the shape-independent `nongrouped_knobs(arch, dtype)` grid for 3x3 stride
1/2 and offers a problem only the widths `tile_w_candidates` would pick for it.

C++ engine twin: `rocke_build_direct_conv_nongrouped` in
`cpp/instances/common/conv_direct_nongrouped.cpp`; byte-identity gated by
`library/tests/parity/conv_direct_grouped_emit.{c,py}` (configs 33+)
(`tools/check_byte_identity.py --only conv_direct_grouped`), Python IR golden
`conv_direct_nongrouped/*` in `tests/instances/rocke_ir_parity_harness.py`, binding
`rocke_engine.conv_direct_nongrouped_*` / `rocke.core.backend.lower_conv_direct_nongrouped`.

## Direct Dgrad on WMMA (gfx1250)

`library/kernels/common/conv_direct_dgrad_wmma.py`, spec `DirectDgradWmmaSpec`,
builder `build_direct_dgrad_wmma` (C++ twin `instance_conv_direct_dgrad_wmma.h`).
A stride-1 dgrad for gfx1250 (wave32, WMMA 16x16x32, TDM), dense and grouped:

    dX[n, h, w, c] = sum_{r, s, k} dY[n, h + PAD - r, w + PAD - s, k] * W[k, r, s, c]

A workgroup owns `block_c = waves_c * tiles_c * 16` dX channels of one group
and `tile_n` images of `tile_h x tile_w` dX pixels (`waves_p * tiles_p * 16` in
all). Its loop walks the group's dY channels in chunks of `ck` through an LDS
ring of `stages` slots:

- one 3-D TDM transfer per image stages the dY halo
  `[tile_h + KH - 1][tile_w + KW - 1][ck]`, zero-filled past the image;
- one 3-D TDM transfer stages the chunk's weights for every tap as `[tap][k][c]`;
- each tap reads the weights transposed (`ds_read_tr16_b128`, the WMMA A
  operand) and a window of the halo shifted by `(KH-1-r, KW-1-s)` (B), so there
  is no weight pre-pass and no im2col.

A wave holds `tiles_c x tiles_p` 16x16 accumulators. The epilogue stages the
dX tile in LDS and stores it with one TDM transfer per image, clipped to the
image and the group's channels. `prefetch` reads the next tap's fragments
before the current tap's WMMAs; `sched` sets the machine-scheduler strategy.

Runtime (kernargs): batch, extents, the group count and both channel counts.
Build-time: filter, padding, dtype and the tile geometry. Constraints: stride
1, no dilation, `PAD` in 0..K-1, every tensor below 2 GiB (i32 byte offsets), `cpg` and `kpg` multiples of 8, and the ring and the staged dX
tile within the 320 KB LDS. The AOT grid is `DGRAD_WMMA_KNOBS`
(`kernels/conv/direct_aot_jobs.py`, variant `direct_dgrad_wmma`).

## Direct Wgrad Kernels

Sources: `library/kernels/common/conv_direct_wgrad_tiled.py`,
`conv_direct_dw_wgrad.py`, `conv_direct_wgrad_gather.py`. Weight-gradient
kernels of the direct-conv AOT sweep (`--directions wgrad`), gfx950; all take
the `conv_direct_arg_names("wgrad")` kernargs (dY on A, X on B) and accumulate
in fp32.

| Spec | Serves | Output |
|---|---|---|
| `DirectWgradTiledSpec` | dense and grouped, stride 1, `cpg`/`kpg` multiples of 8 | `global_atomic_add` into a zeroed fp32 `dW[K, KH, KW, cpg]` |
| `DwWgradLdsSpec` | depthwise, `C % 8 == 0`, 3x3, stride 1/2 | partial slots `ws[part][tap][c]` |
| `DwWgradColsSpec` | depthwise, `C <= 8` (stem layers), any filter | partial slots `ws[part][tap][c]` |
| `WgradGatherSpec` | dense, small `C` (`KW * C <= 96`), any stride | partial slots `ws[part][(r, s, c)][k]` |
| `DwWgradFoldSpec` | the fold after a partial-slot pass | `dW` written whole (no zeroing) |

- **Tiled.** A wave owns `wave_k16*16 x wave_c16*16` channels for all taps;
  the block DMAs input rows of a 32-column strip into LDS ring buffers
  (`buffer_load ... lds`, `prefetch` rows in flight) with an XOR chunk swizzle
  that `pick_swizzle` picks at build time so the `ds_read_b64_tr_b16` fragment
  reads are bank-conflict free. `unfold_n` packs several images into one strip;
  `q_groups=2` (+`pingpong`) runs two wave groups half a row apart.
- **Partial slots + fold.** Same-address fp32 atomics serialize, and a
  depthwise `dW` has only a few thousand addresses, so the depthwise and gather
  kernels store one partial slot per partition with plain stores and
  `DwWgradFoldSpec` sums the partitions. In the sweep they are two-step plans:
  the pass writes the scratch `ws`, the fold reads it (kernargs `A, D, A_bytes,
  D_bytes, parts, channels`).
- All four row-streaming kernels set the kernel attr `lds_alias_scopes`: every
  smem allocation gets its own alias scope, so the waitcnt pass does not drain
  the in-flight LDS DMA before each `ds_read`.
- The validators gate on `lds_bytes()`, which counts what the builder
  allocates: ring slots rounded up to whole DMA instructions of every wave
  (plus the gather's 16-element transpose tail) and the reduction buffers.
  The LDS-DMA kernels (tiled, `DwWgradLdsSpec`, gather) also require a wave64
  target with 16-byte `buffer_load ... lds` (`async_lds_max_dwords >= 4`).

C++ engine twins: `rocke_build_direct_wgrad_tiled` (`cpp/instances/common/conv_direct_wgrad_tiled.cpp`,
also `rocke_direct_wgrad_pick_swizzle`), `rocke_build_dw_wgrad_{fold,cols,lds}`
(`conv_direct_dw_wgrad.cpp`), `rocke_build_direct_wgrad_gather`
(`conv_direct_wgrad_gather.cpp`); byte-identity gated by
`library/tests/parity/conv_direct_grouped_emit.{c,py}` (configs 95+), Python IR
golden `conv_direct_wgrad/*` in `tests/instances/rocke_ir_parity_harness.py`.
On-GPU numeric coverage: `TestDirectWgradAotKernelsCorrectness` in
`library/tests/test_direct_conv_correctness.py` (built on the cache probe,
launched on other shapes); the two-step plans:
`library/tests/test_direct_kernel_cache.py`.

## Img2Col

Source: `library/kernels/common/img2col.py`.

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
- Staging loop sized in whole thread-passes without a scratch tail: the last pass writes past the LDS array into the neighbouring allocation (first output row correct, later rows wrong).
- C++ port folds two emitting builder calls into one argument list: C++ argument evaluation order is unspecified, so `arith.constant`s reorder and byte-identity breaks.
- Output descriptor uses NHWC instead of NHWK stride order.
- Benchmark compares implicit-GEMM graph mode to direct per-launch mode without labeling launch overhead.
- Output (or a multi-kernel pipeline's workspace) verified after a zero fill: an unwritten lane reads as 0 and hides the bug. Pre-fill dX and scratch with 0xFF bytes (NaN in fp16 and bf16), except where the kernel accumulates atomically into a zeroed output.
- Group offset applied twice: indexing a per-group tensor (last dim `kpg`) with an absolute channel index (`g*kpg + k`) passes every `groups == 1` test. Test grouped and ungrouped shapes.
- Eligibility or tiling policy fitted on a cohort that varied only some axes (size but not the channel ratio, one group count): the rule fails in the region the cohort never sampled. Sweep the channel-ratio, filter, group-count and image-size axes, and keep a disjoint hold-out set.
- A whole-image tile in an AOT kernel: the image height is a kernarg, so a build-time "whole image" is a fixed tile height that can be taller or shorter than the real image. Bound the rows at run time (`row_guard`, `row_exit`, the windowed kernel's row masks) and test images shorter and taller than the tile.
