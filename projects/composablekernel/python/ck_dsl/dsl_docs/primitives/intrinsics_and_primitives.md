# Intrinsics And Primitives

This page documents the DSL primitives that map closely to GPU hardware behavior. Everything below is verified against `helpers/atoms.py`, `helpers/loads.py`, `helpers/layouts.py`, `helpers/schedule.py`, `helpers/pipeline.py`, `helpers/epilogues.py`, `helpers/reduction.py`, `core/ir.py`, and `core/lower_llvm.py`.

## MFMA Atoms

`helpers/atoms.py::MfmaAtom` packages one MFMA intrinsic's shape, per-lane widths, accumulator width, dispatch to `IRBuilder`, and lane-to-output mapping.

| Atom | A per lane | B per lane | C per lane | Output tile | Notes |
|---------------------|-----------:|-----------:|-----------:|------------:|--------------------------------|
| `f16_16x16x16` | `<4xhalf>` | `<4xhalf>` | `<4xfloat>` | 16x16 | legacy CDNA, gfx940+ |
| `f16_16x16x32` | `<8xhalf>` | `<8xhalf>` | `<4xfloat>` | 16x16 | K-packed, gfx950 only |
| `f16_32x32x8` | `<4xhalf>` | `<4xhalf>` | `<16xfloat>` | 32x32 | canonical hero atom |
| `f16_32x32x16` | `<8xhalf>` | `<8xhalf>` | `<16xfloat>` | 32x32 | K-packed, gfx950 only |
| `f16_4x4x4` | `<4xhalf>` | `<4xhalf>` | `<4xfloat>` | 4x4 x16 batches | small-channel grouped |

Catalog: `MFMA_F16_ATOMS` and the lookup `mfma_atom("f16", m, n, k)`.

`IRBuilder` also exposes bf16 variants used by `attention_tiled_*`:

```text
mfma_f32_16x16x16_bf16 # gfx950 lowers via *_1k variant; operands as <4 x i16>
mfma_f32_16x16x32_bf16 # operands as <8 x bfloat>
```

`MFMA_F16_ATOMS` does not yet include bf16 entries; the bf16 atoms are reached directly through `IRBuilder` from the attention kernels.

### FP8 / BF8 Atoms (gfx940+)

`MFMA_FP8_ATOMS` adds four FP8 / BF8 atoms paired with the f16 set:

| Atom | A per lane | B per lane | C per lane | Output tile |
|--------------------|--------------------|--------------------|---------------|-------------|
| `fp8_16x16x32` | `<8 x fp8e4m3>` | `<8 x fp8e4m3>` | `<4 x float>` | 16x16 |
| `bf8_16x16x32` | `<8 x bf8e5m2>` | `<8 x bf8e5m2>` | `<4 x float>` | 16x16 |
| `fp8_32x32x16` | `<8 x fp8e4m3>` | `<8 x fp8e4m3>` | `<16 x float>`| 32x32 |
| `bf8_32x32x16` | `<8 x bf8e5m2>` | `<8 x bf8e5m2>` | `<16 x float>`| 32x32 |

All four pack the per-lane operand into a single 64-bit register
(8 fp8 / bf8 bytes), which lowers to `<2 x i32>` at the LLVM
intrinsic boundary. The IRBuilder methods (`mfma_f32_16x16x32_fp8`,
etc.) accept the wider `<8 x fp8e4m3>` / `<8 x bf8e5m2>` IR
operand types and the lowering bitcasts to `<2 x i32>` automatically.

Output lane layouts are identical to the f16 16x16x32 / 32x32x16
atoms, so `MfmaAtom.lane_to_output(...)` works unchanged.

The four mixed-precision variants (`fp8.bf8`, `bf8.fp8` at each tile
shape) are reachable by manually bitcasting the operand vectors and
emitting the matching tile op family; we don't expose them as
distinct atoms today since the FP8 GEMM kernels we ship only need
the homogeneous pair.

Catalog lookup: `mfma_atom("fp8", 16, 16, 32)` (also accepts the
canonical `mfma_atom("fp8e4m3", 16, 16, 32)`); see `helpers/atoms.py`
for the `_DTYPE_ALIAS` map.

### Lane Layouts

These come from `helpers/atoms.py::MfmaAtom.lane_to_output`.

`16x16x16` and `16x16x32` (`c_per_lane = 4`):

```text
m_blk = lane / 16
n_in_atom = lane % 16
output row = m_blk * 4 + i # i = 0..3
output col = n_in_atom
```

`32x32x8` and `32x32x16` (`c_per_lane = 16`):

```text
m_blk = lane / 32 # 0 or 1
n_in_atom = lane % 32
output row = (i // 4) * 8 + m_blk * 4 + (i % 4) # i = 0..15
output col = n_in_atom
```

`4x4x4` (`c_per_lane = 4`):

```text
batch_idx = lane / 4 # 16 independent 4x4 matmuls per wave
lane_in_batch = lane % 4
output row = i # i = 0..3
output col = lane_in_batch
```

This mapping is the single biggest source of correctness bugs when swapping atoms. The epilogue must agree with the atom that produced the accumulator.

### K-packed Atom Operand Packing

For `f16_16x16x32`, the per-lane K layout is:

```text
c4 = lane / 16
A lane covers K = [c4*8 : c4*8 + 8] # contiguous K slice
```

This is **not** the flat-concat `[c4*4 : c4*4 + 4] + [c4*4 + 16 : c4*4 + 20]` layout. The wrong packing compiles, runs, validates within ~1e-2, and fails at 1e-3. If you see "close but not bit-correct" on a K-packed kernel, suspect this first.

## Buffer Resources

AMDGPU buffer descriptors are a core safety and performance primitive. The DSL path:

```text
make_buffer_resource(b, ptr, num_bytes=N)
 -> b.buffer_rsrc(ptr, num_bytes)
 -> tile.buffer_rsrc op
 -> llvm.amdgcn.make.buffer.rsrc.p1(ptr, i16 0, i32 N, i32 159744)
```

`159744 = 0x00027000`. DW3 encoding: TYPE=2 (BUFFER_RESOURCE), DATA_FORMAT=4 (32-bit dword), NUM_FORMAT=4 (UINT). With these flags, out-of-range byte offsets silently return zero on load and are dropped on store.

Loads and stores:

```text
buffer_load_f16(rsrc, voffset, soffset)
buffer_load_vN_f16(rsrc, voffset, soffset, dwords) # vN in halves; dwords = N/2
buffer_store_f16(rsrc, voffset, soffset, value)
buffer_store_vN_f16(rsrc, voffset, soffset, value, dwords)
```

Canonical masked-access pattern:

```text
off_bytes = off_elems * sizeof(elem)
safe = b.select(valid, off_bytes, b.const_i32((1 << 31) - 1))
v = b.buffer_load_vN_f16(rsrc, safe, c0, dwords)
```

For stores, the higher-level `view.store_vec_at(b, elem_off, value, n, mask=...)` from `helpers/tensor_view.py` handles the OOB-routing for false-mask lanes.

## Async DRAM-to-LDS

`AsyncTileLoader` (`helpers/loads.py`) wraps `raw_ptr_buffer_load_lds`:

```text
issue(b, tid, rsrc, descriptor[, oob_sentinel=INT32_MAX, coherency=CACHE_ALL])
```

Per-lane semantics:

```text
The intrinsic writes dwords*4 bytes per lane, lane-contiguously.
Lane i in a wave writes at lds_base + i * dwords * 4.
```

Constraints (from `choose_dwords` and the lowering):

- `dwords in {1, 3, 4}` (no 2-dword form on this LLVM target).
- That corresponds to `2, 6, 8` half elements per lane.
- LDS destination base must be uniform within a wave; the loader uses `b.to_sgpr_u32(...)` to hoist it into an SGPR (saves a `v_readfirstlane_b32` round-trip per iter).
- Consumers must call `b.s_waitcnt(vmcnt=0)` before reading the LDS payload (and typically `b.sync()` to get a barrier as well).
- Swizzle in the consumer's LDS read arithmetic, not in the destination base.

Coherency hints (`core/ir.py`):

```text
CACHE_ALL = 0 # default
CACHE_GLOBAL = 1 # GLC: skip L2 (one-shot loads)
CACHE_STREAM = 2 # SLC: streaming hint (don't evict useful lines)
NON_TEMPORAL = 3 # GLC + SLC: bypass cache hierarchy
```

For K-loop streaming tile loads, `CACHE_STREAM` is the documented choice; the tile is consumed in the next iter and never re-read.

## LDS Operations

LDS allocated via `tile.smem_alloc` lowers to a module-level `addrspace(3)` global. Common ops are in `IRBuilder`:

```text
smem_alloc(elem, shape, name_hint)
smem_load_f16(smem, [row, col])
smem_load_vN_f16(smem, *indices, n in {1,2,4,8})
smem_load_v4_f16(smem, row, col)
smem_store_vN_f16(smem, [row, col], value, n in {1,2,4,8})
smem_addr_of(smem) -> i64
smem_ptr_add(lds_addr, byte_off) -> i64
```

Vector LDS stores at `n=8` lower to `ds_write_b128` when the address is 16-byte aligned. Vector LDS loads at `n=8` lower to `ds_read_b128` under the same condition.

`helpers/layouts.py::LdsLayout` centralizes:

- tile shape;
- K padding (`lds_k_pad`, default `+8` for sync paths when `block_k >= 16`, `+0` for async paths);
- packed async constraints;
- compatibility with loader and consumer read formulas.

Rule: `raw_ptr_buffer_load_lds` writes lane-contiguous bytes. A packed async LDS destination cannot itself be per-lane swizzled. If a consumer needs a swizzle to avoid bank conflicts, put it in the LDS read address arithmetic — see `TransposeLdsReader`.

## Wave / Cross-Lane Primitives

```text
lane_id() # 0..63 on wave64
readfirstlane(v) # broadcast lane 0's value across the wave (lift to SGPR)
pin_sgpr(v) # asm constraint that keeps v in an SGPR across uses
to_sgpr_u32(v) # pin_sgpr(readfirstlane(v)) - the canonical idiom
wave_all(pred) # i32 = 1 iff every lane's pred != 0
wave_any(pred) # i32 = 1 iff any lane's pred != 0
wave_ballot(pred) # i64 bitmask of lanes
ds_bpermute(addr, data) # gather: lane reads from (addr >> 2) bits [7:2]
warp_shuffle_xor(v, lane_xor) # bpermute with (lane ^ lane_xor) << 2
ds_read_tr16_b64(smem, *indices, dtype=F16)
 # 16x16 fp16 transpose-read returning the MFMA B-operand layout
```

Common patterns:

- `to_sgpr_u32(...)` for any wave-uniform i32 (LDS bases, global byte offsets, tile coords). Without it, the register allocator can re-materialize the value into VGPRs at every unrolled K-loop iteration.
- `wave_all(...)` for adaptive online-softmax rescale skip: if every lane's `max_diff < THRESHOLD`, the workgroup can skip the rescale path.
- `warp_shuffle_xor(...)` for butterfly reductions on f32 values; `helpers/attention.py::warp_xor_reduce_max/sum` wraps this for online softmax.
- `ds_read_tr16_b64(...)` for PV matmul in attention: V is in LDS row-major; this returns each lane's `<4 x half>` MFMA B-operand directly, replacing four strided `ds_read_u16` instructions.

## Atomic Operations `IRBuilder` exposes atomic-add primitives on both address spaces.
Both ops return the value at the slot **before** the add (LLVM
`atomicrmw` semantics, matching HIP's `atomicAdd`).

```text
global_atomic_add(ptr, idx, value, *, ordering="monotonic")
 -> i32 | f32
lds_atomic_add(smem, indices, value, *, ordering="monotonic")
 -> i32 | f32
```

Supported value types: `i32` and `f32`. `ordering` defaults to
`monotonic` (the relaxed model that every CK Tile reduction kernel
relies on); `seq_cst` is available for the rare case requiring full
memory ordering.

Lowering on AMDGPU:

| IR op | Address space | i32 ISA | f32 ISA (gfx940+) |
|---|---|---|---|
| `global_atomic_add` | `addrspace(1)` | `global_atomic_add` | `global_atomic_add_f32` |
| `lds_atomic_add` | `addrspace(3)` | `ds_add_u32` | `ds_pk_add_f32` |

Usage in `instances/moe_sorting.py`:

```python
# Phase 1: histogram. Each (token, topk) pair atomically increments
# the expert's count.
eid = b.global_load_i32(TopkIds, pair_idx)
b.global_atomic_add(Hist, eid, b.const_i32(1))

# Phase 3: scatter. Each pair claims the next free slot in its
# expert's bucket via a global atomic counter.
local_off = b.global_atomic_add(Counter, eid, b.const_i32(1))
```

Block-level histogram + scan helpers in `helpers/scan.py` wrap the
common patterns:

```text
lds_zero_i32(b, lds, *, tid, block_size, length)
block_histogram_i32(b, lds, keys, *, tid, block_size, num_bins)
block_exclusive_scan_i32(b, lds, *, tid, block_size, length)
```

The scan is a single-block Hillis-Steele implementation requiring
`length <= block_size`; multi-block / multi-pass scans are a future
extension.

## Persistent-Kernel Pattern `helpers/persistent.py` packages the standard pattern of "launch a
small constant number of CTAs, each pulls its work-items from a
global counter via `atomic_add(1)`":

```text
persistent_tile_for_each(b, *, counter, num_tiles, max_iters, body)
persistent_tile_loop(b, *, counter, num_tiles, max_iters,
 tile_idx_init) -> contextmanager
build_persistent_counter_init(b, counter, *, counter_idx=None,
 increment=1)
```

The helper emits one `atomic_add` at kernel entry (the CTA's initial
tile id) plus one per `scf.for` iteration (the next tile id, threaded
as a loop-carried SSA value). The `in_range` predicate skips the
tail iterations past `num_tiles` -- those over-fetches are
harmless for correctness, just bump the counter past the work item
count.

This is the foundation for the StreamK GEMM (`instances/streamk_gemm.py`)
and the persistent variant of the MoE sort fused pipeline; the
helper is shared so the pattern doesn't get re-implemented per kernel.

## StreamK Tile Partitioner `helpers/streamk.py` ships the partitioner math that pairs with the
persistent-grid pattern to implement StreamK GEMM:

```text
StreamKPartition(m_tiles, n_tiles, k_iters)
 num_macro_tiles # m_tiles * n_tiles * k_iters
 k_iters_per_output_tile # = k_iters (for the Atomic strategy)

compute_streamk_grid_size(spec, *, num_cus=304, blocks_per_cu=1)
 -> int # min(num_macro_tiles, num_cus * blocks_per_cu)

emit_streamk_decode(b, linear_id, spec)
 -> (m_tile, n_tile, k_iter, is_first, is_last) # SSA bundle

StreamKReductionStrategy.{Atomic, Reduction}
```

Layout: macro tiles are walked K-major within a fixed ``(m_tile,
n_tile)``, so consecutive ``linear_id`` values touch the same output
tile. The ``is_first`` / ``is_last`` predicates let the Reduction
strategy (v2) decide which CTA seeds + finalises the output; the
Atomic strategy (v1) ignores them and just `atomic_add` every partial
into a shared f32 workspace.

Used by ``instances/streamk_gemm.py`` v1 (scalar inner GEMM + Atomic
strategy, end-to-end); the MFMA upgrade and the Reduction strategy
land as follow-ons against the same partitioner surface.

## quantised-GEMM helpers

`helpers/i4_dequant.py`, `helpers/mx_scale.py`, and
`helpers/preshuffle.py` add the support surface the block-
scaled and MX GEMM kernels need:

```text
# Packed-i4 byte unpack (sign-extend each 4-bit nibble to i32 / f32 /
# fp8 / bf8). Lowers to ``v_bfe_i32`` (bit-field-extract signed) on
# the AMDGPU backend.
unpack_i4_byte_to_pair_i32(b, packed_byte)
unpack_i4_byte_to_pair_i8(b, packed_byte) # alias of _i32 today
unpack_i4_byte_to_pair_f32(b, packed_byte)
dequant_i4_byte_to_fp8_pair(b, packed_byte, *, inv_scale)
dequant_i4_byte_to_bf8_pair(b, packed_byte, *, inv_scale)

# OCP MX E8M0 shared-exponent decode + apply.
# decode_mx_scale_e8m0(e8m0) -> f32 multiplier ``2^(e - 127)`` with
# ``e == 0`` and ``e == 255`` mapped to ``0.0`` (matches the AMDGPU
# MX MFMA hardware path's denormal flush).
decode_mx_scale_e8m0(b, e8m0)
apply_mx_scale(b, value_f32, scale_f32)
load_and_decode_mx_scale_byte(b, scale_ptr, scale_idx)

# Preshuffled-B tile-major layout helper. The launcher uses
# ``host_preshuffle_layout`` to build the permuted-B torch tensor
# before the kernel launch; ``emit_preshuffleb_offset`` is the per-
# lane address calculator inside the kernel.
PreshuffleBSpec(block_n, block_k, elem_bytes=1)
emit_preshuffleb_offset(b, spec, *, n_tile, k_tile, n_in_tile,
 k_in_tile, n_tile_count)
host_preshuffle_layout(spec, *, n, k)
```

Used together with the FP8 / BF8 MFMA atoms, these helpers compose
into the full v2 MFMA-based block-scaled / MX GEMM body.

## fused-MoE indirect-address helpers

`helpers/gather_scatter.py` is the per-token indirect-address surface
the fused-MoE forward needs. The fused-MoE pipeline (`instances/
fused_moe.py`) has two distinct indirect patterns that the existing
tile / sweep / transform DAG abstractions don't cover:

* **Gather-by-bucket-row**: each *output row* of the per-expert input
 tensor reads from one *input row* of the original token tensor,
 selected by ``sorted_token_ids[bucket_idx]``.
* **Scatter-by-bucket-row**: each *input row* of the per-expert output
 tensor writes to one *output row* of the per-token MLP output,
 scaled by ``sorted_topk_weights[bucket_idx]`` and accumulated
 atomically via :meth:`IRBuilder.global_atomic_add`.

The helpers expose the per-element address chains:

```text
# Per-column flat offset for one row of the pre-sort token tensor,
# gathered by ``bucket_idx``:
# token_id = sorted_token_ids[bucket_idx]
# offset = token_id * hidden + col
gather_row_offset(b, sorted_token_ids, bucket_idx, *, hidden, col)

# Write-side counterpart -- symmetric with gather_row_offset:
# offset = token_id * hidden + col
scatter_token_offset(b, token_id, *, hidden, col)

# One i32 global load with align=4. Enforces ``ptr<i32>`` type so a
# misrouted ptr<f32> fails fast at IR-build time.
load_sorted_token_id(b, sorted_token_ids, bucket_idx)

# One f32 global load with align=4 for the per-bucket topk weight.
load_sorted_topk_weight(b, sorted_weights, bucket_idx)
```

These four helpers + the `global_atomic_add` primitive are the
entire delta the fused-MoE forward needs on top of the existing IR
surface. The ``GroupedInput``, ``GateOut``, ``Hidden``, ``DownOut``,
and ``Y`` tensors all live in the standard ``ptr<dtype, global>``
addressing -- there's no new pointer space, no new buffer descriptor
flavour, and no new MFMA atom.

## FMHA-expansion primitives

 adds three classes of primitives the FMHA family needs on top
of the existing tiled-attention surface:

### Packed-bf16 atomic add (FMHA-bwd dQ direct-bf16 path)

```text
b.global_atomic_add_pk_bf16(ptr, idx, vec_bf16x2)
 # -> <2 x bfloat>, the pre-add value at the slot pair
```

Lowers to `llvm.amdgcn.global.atomic.fadd.v2bf16` (gfx940+).
``vec_bf16x2`` must be a ``<2 x bf16>`` vector; the pointer must be
to a bf16 buffer with an even element index (the hardware reads /
writes a 32-bit aligned pair). Roughly 2x cheaper per element than
the cmpxchg-loop single-bf16 atomic and avoids the hot-row
contention.

Used by the FMHA-backward kernel's dQ accumulate when the caller
wants to land bf16 directly in HBM rather than running a separate
``f32 -> bf16`` cast pass over the fp32 workspace.

### Extra integer ops (for Philox4x32 RNG)

```text
b.lshr(a, b) # arith.lshr -- logical right shift
b.umul_hi_i32(a, b) # high 32 bits of u32*u32 (-> v_mul_hi_u32)
```

The high-half multiply is the kernel of Philox4x32's per-round
``mulhi`` step. The LLVM lowering uses the canonical
``zext / mul / lshr / trunc`` pattern; the AMDGPU backend folds
the whole sequence into one ``v_mul_hi_u32`` instruction.

### Rotary embedding helpers (`helpers/rotary.py`)

```text
RotaryLayout = "interleaved" | "half"
RotarySpec(head_size, layout, table_stride_pos=0)
 pair_count, stride_pos # compile-time accessors

pair_indices(spec, pair_idx) # -> (lo, hi) head-dim indices
load_cos_sin(b, cos_table, sin_table, token_pos=, pair_idx=, spec=)
 # one (cos, sin) f32 pair load
apply_rotary_pair_f32(b, lo, hi, cos_t, sin_t)
 # 2x2 rotation; 2x v_fma_f32 per pair
```

Both layouts use the same ``(cos, sin)`` tables -- only the
pair-index function differs. Interleaved (GPT-J / LLaMA-1) pairs
``(2*i, 2*i+1)``; half (LLaMA-2 / 3 / Qwen) pairs
``(i, i + H/2)``.

### Philox4x32 RNG helpers (`helpers/rng.py`)

```text
PHILOX_M0 / M1 / W0 / # round constants (match PyTorch + CK Tile)
PHILOX_ROUNDS = 10

philox_u32_quartet(b, seed_lo=, seed_hi=, subseq=, offset=)
 # -> (r0, r1, r2, r3) i32
philox_uniform_f32_quartet(b, seed_lo=, seed_hi=, subseq=, offset=)
 # -> (u0, u1, u2, u3) f32 in [0, 1) via (u >> 8) * 2^-24
dropout_mask_pair_f32(b, seed_lo=, seed_hi=, subseq=, offset=, keep_prob_f32=)
 # -> (keep0, scale, keep1, scale) -- canonical FMHA-fwd dropout pair
```

Pure functions -- no per-thread state. Reproducible bit-for-bit
from ``(seed, subseq, offset)``; FMHA fwd + bwd can regenerate the
same mask independently without explicit state hand-off.

### Extra attention masks (`helpers/attention.py` extension)

```text
alibi_bias_log2(b, key_pos=, query_pos=, head_slope_log2=)
 # closed-form slope * (k_pos - q_pos), log2 score domain
alibi_bias_matrix(b, bias_ptr=, head_idx=, key_pos=, query_pos=,
 stride_head=, stride_q=)
 # precomputed (H, Q, K) f32 bias load
custom_mask(b, mask_ptr=, query_pos=, key_pos=, stride_q=)
 # per-(Q, K) i8 bool mask -- returns i1 keep predicate
```

These compose with the existing ``causal_mask`` and
``sliding_window_mask`` (both already in this module) to cover the
five mask families CK Tile's 01_fmha exposes via its ``mask_mode``
enum.

## Sage / sparse attention primitives

ops are required (reuses the ``cvt_fp8_to_f32`` /
``cvt_bf8_to_f32`` family and the ``unpack_i4_byte_to_pair_i32``
already shipping in the IR + helper layer).

### Per-block Q+K scale loaders (`helpers/qk_scale.py`)

The Sage attention family stores Q and K in fp8 / i8 / i4 mantissa
buffers and compensates for the dynamic-range loss via per-block (or
per-head) f32 scales applied to the attention score *before* the
softmax. This module exposes two layouts behind one descriptor:

```text
QkScaleLayout = "per_head" | "per_block"
QkScaleSpec(layout, scale_block, stride_batch, stride_head, stride_block)

load_q_scale_for_block(b, q_scale_ptr, *, spec=, batch_idx=, head_idx=, q_block_idx=)
load_k_scale_for_block(b, k_scale_ptr, *, spec=, batch_idx=, head_idx=, k_block_idx=)
apply_qk_scales(b, score_log2, *, q_scale=, k_scale=)
 # -> score_log2 * (q_scale * k_scale); folds to one v_fma_f32 per score
```

### Codebook dequant for sage int variants (`helpers/codebook.py`)

The ``"i8_fp8_bf16"`` and ``"i4_fp8_bf16"`` Sage variants store K/V
as i8 / i4 integers; the inner loop re-materialises the fp8 mantissa
via a small codebook lookup (256 entries for i8, 16 entries for i4):

```text
codebook_lookup_i8_to_fp8(b, codebook_ptr, i8_value_i32, *, per_tensor_scale=)
codebook_lookup_i8_to_bf8(b, codebook_ptr, i8_value_i32, *, per_tensor_scale=)
codebook_lookup_i4_pair_to_fp8(b, codebook_ptr, packed_byte_i8, *, per_tensor_scale=)
 # unpacks both nibbles via unpack_i4_byte_to_pair_i32, returns
 # (lo_fp8, hi_fp8)
codebook_lookup_i4_pair_to_bf8(b, codebook_ptr, packed_byte_i8, *, per_tensor_scale=)
apply_per_tensor_scale(b, value_f32, scale)
```

The 256-entry i8 table is ~1 KB and usually L1-resident; the 16-entry
i4 table is 64 bytes and stays in SGPR-immediate / L1 across the loop.

### Sparse attention K-iterators (`helpers/sparse_iter.py`)

Two K-iteration strategies for the two CK Tile ``50_sparse_attn``
configurations:

```text
BlockSparseSpec(num_k_blocks, stride_q_block)
block_sparse_iter(b, mask_ptr=, q_block=, spec=, body=)
 # scf.for over k_blocks; runtime guard ``if (mask_byte != 0) { body }``
 # -- Jenga one-hot block-sparse path

VsaSparseSpec(max_blocks_per_q, stride_q_block)
load_block_count(b, block_count_ptr, q_block)
vsa_lut_iter(b, lut_ptr=, block_count_ptr=, q_block=, spec=, body=)
 # scf.for up to max_blocks_per_q; runtime guard ``slot < BlockCount[q_block]``
 # -- VSA indirect-LUT path
```

Both iterators expose the per-iteration ``k_block`` index to the body
(plus the ``slot`` for VSA). The body emits the standard per-K-block
attention work (load K/V, score, online-softmax update, V accumulate).

## Quantization / Low-Precision Conversions

`IRBuilder` exposes single-element conversions used by the FP8 / int8 paths:

```text
cvt_fp8_to_f32(v) # llvm.amdgcn.cvt.f32.fp8 (fp8e4m3 -> f32)
cvt_bf8_to_f32(v) # llvm.amdgcn.cvt.f32.bf8 (bf8e5m2 -> f32)
cvt_f32_to_fp8(v) # llvm.amdgcn.cvt.pk.fp8.f32, low-byte extract
cvt_f32_to_bf8(v) # llvm.amdgcn.cvt.pk.bf8.f32, low-byte extract
cvt_f32_to_i8_sat(v) # round-to-nearest-even + saturate to int8 [-128, 127]
clamp_f32(v, lo, hi) # folds to v_med3_f32
```

Higher-level helpers in `helpers/quant.py`:

```text
QDType
QUANT_MAX_ABS # quant range bounds
quant_ir_type(qdtype)
quant_max_abs(qdtype)
quantize_scalar_f32(b, v, scale, qdtype, ...)
dequantize_scalar_to_f32(b, v, scale, qdtype, ...)
ir_to_qdtype(ir_type)
```

These cover SmoothQuant / RDQuant-style epilogues and the FP8 K/V cache dequant path in unified attention. See `primitives/quantization.md` for the workflow.

## Schedule And Pipeline

`helpers/schedule.py::SchedulePolicy` emits named scheduler-hint policies:

- `sched_group_barrier(mask, count, sync_id)` groups instructions for the AMDGPU scheduler.
- `s_setprio(priority)` sets wave priority.
- `s_waitcnt` is exposed at the IR level for fine-grained control.

`helpers/pipeline.py::SoftwarePipeline.run_ping_pong(...)` is the prologue / steady-state / epilogue construction helper for ping-pong async pipelines. It assumes the kernel author already chose the LDS layout, atom, and tile shape; it sequences the DMA + compute calls.

General rules:

- use `b.sync()` at simple phase boundaries;
- use `b.sync_lds_only()` only when VMEM can remain in flight safely;
- use `b.s_waitcnt(vmcnt=0)` before consuming async-to-LDS data;
- use scheduler hints only when IR / ISA and benchmark numbers confirm they help;
- never place a barrier inside a divergent control flow unless every wave reaches a matching barrier.

## CShuffle Epilogue

`helpers/epilogues.py::CShuffleEpilogue`. Concept:

1. Convert per-lane f32 accumulator values to output dtype.
2. Place them in an LDS output tile at the logical output coordinates (using `MfmaAtom.lane_to_output`).
3. Synchronize.
4. Threads cooperatively load contiguous chunks from LDS.
5. Emit coalesced wide vector stores (`buffer_store_dwordx{2,4}` or `global_store_dwordx{2,4}`).

When it wins:

- MFMA accumulator ownership is scattered across output coordinates (large GEMM, implicit-GEMM conv).
- Direct stores would be scalar and poorly coalesced.

Risks:

- extra LDS budget;
- one extra barrier;
- wrong `lane_to_output` corrupts output;
- vector store width must respect output contiguity and tails.

## Direct Epilogue

`helpers/epilogues.py::DirectEpilogue`. Used when:

- per-lane outputs are naturally contiguous (e.g. `4x4x4` direct grouped conv, where lane stores 4 channels in a row);
- output tile is small;
- LDS budget is tight;
- specialized atom makes direct vector stores clean.

`vec_in_acc=True` is the lever for atoms whose per-lane elements are already contiguous; the helper emits one vector store per lane.

## Row-Wise Reduction Primitive

`helpers/reduction.py::block_lds_reduce(b, val, lds, tid, block_size, combine="sum" | "max")` is the canonical block reduction:

```text
1. each thread stages its f32 partial in LDS at offset tid
2. sync_lds_only / sync
3. tree reduce: each step halves the active stride
4. all threads (or thread 0) read the final value
```

Consumers: `reduce.py`, `layernorm2d.py`, `rmsnorm2d.py`, attention reductions, pooling reductions.

## Vector I/O Helpers

`helpers/io.py` centralizes dtype-string-tolerant I/O dispatch:

```text
io_ir_type("f16" | "fp16" | "bf16") -> Type
load_scalar(b, ptr, idx, *, dtype)
load_scalar_as_f32(b, ptr, idx, *, dtype)
load_vec(b, ptr, idx, *, dtype, n)
load_vec_as_f32(b, ptr, idx, *, dtype, n)
pack_f32_to(b, scalars_f32, *, dtype) # kwarg is `dtype`, returns <len x dtype> vector
store_scalar(b, ptr, idx, value, *, dtype)
store_scalar_from_f32(b, ptr, idx, f32_value, *, dtype)
store_vec(b, ptr, idx, value, *, n) # value must already be in target dtype
```

`store_vec` does not take a `dtype` kwarg — pass an already-typed vector value (use `pack_f32_to(b, scalars, dtype="f16")` first if you have f32 scalars).
```

Use these in small ops; do not hand-roll dtype dispatch in every kernel.

## Transform DAG Primitives

The transform DAG primitives (`pass_through`, `pad`, `pad_dynamic`, `embed`, `merge`, `unmerge`, `indirect`) emit SSA offset and validity logic. They are documented in detail in `primitives/memory_layout_and_transforms.md`.

## Primitive Selection Rules

- If the op is matrix-like and compute-bound: start with the MFMA atom and confirm `lane_to_output` matches the chosen epilogue.
- If the op has padding / tails / masks: prefer buffer resources and descriptor validity.
- If global-to-LDS traffic is in the critical path: try async loading and verify `s_waitcnt(vmcnt=0)` placement.
- If output stores are scattered: try `CShuffleEpilogue`.
- If row reductions dominate: use `block_lds_reduce` and inspect LDS bank behavior.
- If address math is non-bijective: express it as a transform DAG, not hand-expanded offsets.
- If a primitive is used by more than one instance: wrap it in `helpers/`.
- For wave-uniform addresses inside hot loops: lift to SGPR with `b.to_sgpr_u32(v)`.
- For online softmax reductions: prefer `warp_shuffle_xor` / `warp_xor_reduce_*` over LDS round-trips.
