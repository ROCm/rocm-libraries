# ck_dsl/helpers/ — High-level kernel-authoring reference

This package sits one layer above `ck_dsl.core.ir` (the SSA IR +
builder) and captures the patterns every CK Tile-style GEMM, attention,
or convolution kernel re-implements by hand.

## Why a separate layer

Writing a single GEMM kernel in raw `IRBuilder` calls is ~300-400 lines:
about 50 lines of geometry/lane decomposition, 80 lines of LDS load
plan, 60 lines of MFMA emission, 100 lines of cshuffle epilogue, plus
the K-loop scaffolding. About 80% of that is the *same* across every
GEMM, attention, and convolution kernel in this repo — only the
specific tile shape, MFMA atom, and addressing transforms vary.

The helpers here let a new kernel be expressed as a 60-80 line
top-level skeleton that names what's special about the kernel (the
shape, the atom, the descriptor for non-bijective addressing) and
leaves the boilerplate to the helpers.

## What's here

```text
ck_dsl/helpers/
├── __init__.py        # public exports
│
│  # CK Tile-inspired data abstractions (port of the C++ template family).
│  # Docs: docs/conceptual/ck_tile/tensor_views.rst, tile_window.rst,
│  #       descriptors.rst, sweep_tile.rst, tile_distribution.rst,
│  #       static_distributed_tensor.rst, load_store_traits.rst,
│  #       coordinate_movement.rst.
│
├── tensor_view.py     # TensorDescriptor + TensorView + TileWindow.
│                      # The Python analogue of make_tensor_view<addr_space::*>,
│                      # make_naive_tensor_descriptor_packed, make_tile_window.
│                      # Strides may be int (compile-time) or SSA Value (runtime).
│                      # Also: TensorCoordinate + move_tensor_coordinate
│                      # (incremental (index, offset) updates) and
│                      # view_from_transforms_descriptor (bridge to
│                      # ck_dsl.transforms).
├── distribution.py    # TileDistributionEncoding + make_static_tile_distribution
│                      # (Rs/Hs/Ps/Ys mapping; v1 = no R, 1D-2D X, flexible P/Y).
│                      # StaticDistributedTensor: thread-local register
│                      # container indexed by Y. LoadStoreTraits: auto-picks
│                      # vector_dim_y + scalar_per_vector + snake traversal.
│                      # load_tile / store_tile drive a fully-automated
│                      # window <-> register-tile pass through the distribution.
├── sweep.py           # sweep_row_chunks + pass2_row_chunks: CK Tile-style
│                      # "load X once, sweep Y positions" lambda iteration.
│                      # The simpler form used by every small-op kernel;
│                      # the distribution path is opt-in for cases where
│                      # the (Y, P) decomposition is non-trivial.
├── io.py              # io_ir_type, load_vec, load_vec_as_f32, pack_f32_to,
│                      # store_vec_from_f32; dtype-string-tolerant I/O dispatch.
├── reduction.py       # block_lds_reduce (sum/max) -- canonical LDS tree
│                      # reduction shared by norm/reduce/pool kernels.
├── spec.py            # IOSpecRule + validate_io, SignatureBuilder,
│                      # kernel_name_join, ceil_div_grid -- spec/signature/grid
│                      # scaffolding so each instance file is ~10 lines of glue.
│
│  # Kernel-shape abstractions (GEMM / conv / attention infrastructure).
│
├── atoms.py           # MfmaAtom catalog (the matrix-multiply intrinsics)
├── geometry.py        # WarpGrid: block/warp/lane decomposition
├── loads.py           # CoalescedTileLoader + AsyncTileLoader
├── layouts.py         # LdsLayout: K-padding, packed async layouts, guardrails
├── schedule.py        # SchedulePolicy: named sched_group_barrier policies
├── pipeline.py        # SoftwarePipeline: prologue/steady-state/epilogue
├── epilogues.py       # DirectEpilogue + CShuffleEpilogue
├── attention.py       # Attention2DConfig, OnlineSoftmaxState, PagedKvDescriptor,
│                      # causal/sliding-window masks, soft-cap, select_*d_config.
├── compile.py         # compile_kernel() one-shot IR -> HSACO
└── manifest.py        # make_gemm_manifest, make_conv_manifest, write_artifact
```

The two halves compose: the CK Tile-inspired layer (`tensor_view`,
`sweep`, `io`, `reduction`, `spec`) is the small-op / norm / reduce
authoring surface; the kernel-shape layer (`atoms`, `geometry`,
`loads`, `epilogues`, `pipeline`) is the GEMM / conv / attention
authoring surface. Both lower to the same `ck_dsl.core.ir` builder, so
mixing them in one kernel is fine.

### Instance coverage

Every kernel-building file in `ck_dsl/instances/` uses helpers from at
least one of the two layers; bigger kernels (`gemm_universal`,
`conv_implicit_gemm`) blend both. The current matrix:

| Instance | tensor_view | sweep | io | reduction | spec | distribution | GEMM-shape |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `elementwise.py`         | yes | -   | yes | -   | yes | -  | - |
| `layernorm2d.py`         | yes | yes | yes | yes | yes | -  | - |
| `rmsnorm2d.py`           | yes | yes | yes | yes | yes | -  | - |
| `reduce.py`              | yes | yes | yes | yes | yes | -  | - |
| `transpose.py`           | yes | -   | yes | -   | yes | -  | - |
| `gemm_universal.py`      | yes | -   | -   | -   | yes | -  | `MfmaAtom` |
| `batched_gemm.py`        | -   | -   | -   | -   | yes | -  | (wraps `gemm_universal`) |
| `grouped_gemm.py`        | -   | -   | -   | -   | yes | -  | (wraps `gemm_universal`) |
| `conv_implicit_gemm.py`  | yes (buffer) | - | - | - | yes | - | `AsyncTileLoader`, `WarpGrid`, `CoalescedTileLoader`, `CShuffleEpilogue`, `MfmaAtom` |
| `conv_direct_grouped.py` | -   | -   | -   | -   | yes | -  | - |
| `attention_unified.py`   | -   | -   | -   | -   | yes | -  | - |
| `attention_tiled_2d.py`  | -   | -   | -   | -   | yes | -  | - |
| `attention_tiled_3d.py`  | -   | -   | -   | -   | yes | -  | - |

The "spec" column (kernel_name_join, SignatureBuilder, ceil_div_grid,
IOSpecRule/validate_io) is now uniform across all 13 instances. The
"tensor_view" column now covers the GEMM and implicit-GEMM-conv
families too, after adding the buffer-view feature (`BufferResource`
+ `addr_space="buffer"` + `load_*_at` / `store_*_at` flat-offset
methods with the `mask=` OOB-sentinel idiom). Attention still
intentionally uses raw IR for its per-warp `ds_bpermute` softmax
butterfly; that pattern is not a memory access and so doesn't fit
the `TensorView` surface.

#### Buffer-view feature surface (new)

For bounds-checked AMDGPU buffer ops (the canonical lever for
padding-aware conv, attention K-tail handling, and last-tile epilogue
safety):

| API | Purpose |
| --- | --- |
| `BufferResource` | The 128-bit AMDGPU buffer descriptor + soffset wrapper |
| `make_buffer_resource(b, ptr, num_bytes)` | Builds the rsrc once per buffer |
| `make_buffer_view(rsrc, shape, dtype, strides=...)` | A `TensorView` in `addr_space="buffer"` |
| `view.load_vec_at(b, elem_off, n, mask=...)` | Flat-offset vector load with OOB-zero mask |
| `view.load_scalar_at(b, elem_off, mask=...)` | Flat-offset scalar load with OOB-zero mask |
| `view.store_vec_at(b, elem_off, value, n, mask=...)` | Flat-offset vector store; False mask drops the store |
| `view.store_scalar_at(b, elem_off, value, mask=...)` | Flat-offset scalar store |
| `view.async_load_lds_at(b, lds_ptr, elem_off, dwords=..., mask=...)` | Async DRAM→LDS via `raw_ptr_buffer_load_lds` (compv4-style pipelines) |

`mask=` works by replacing a False lane's byte offset with
``INT32_MAX``, which the buffer rsrc's bounds check turns into a
silent OOB read/write — no software masking needed. The
`view.buffer` accessor gives typed access to the underlying
`BufferResource` and raises `TypeError` when the view is not in
buffer space.

CK Tile parity (which C++ name maps to which Python helper):

| CK Tile C++ | DSL helper |
| --- | --- |
| `make_naive_tensor_descriptor_packed(shape)` | `TensorDescriptor.packed(shape, dtype)` or `make_naive_tensor_descriptor_packed(shape, dtype)` |
| `make_tensor_view<addr_space::global>(ptr, desc)` | `make_global_view(ptr, shape, dtype)` |
| `make_tensor_view<addr_space::lds>(...)` | `make_lds_view(b, dtype=..., shape=...)` |
| `make_tensor_view<addr_space::buffer>(rsrc, desc)` | `make_buffer_view(rsrc, shape, dtype)` |
| AMDGPU `llvm.amdgcn.make.buffer.rsrc.p1` | `make_buffer_resource(b, ptr, num_bytes=...)` |
| `buffer_load_dwordN` / `raw_ptr_buffer_load` | `view.load_vec_at(b, elem_off, n, mask=...)` |
| `buffer_store_dwordN` | `view.store_vec_at(b, elem_off, value, n, mask=...)` |
| `raw_ptr_buffer_load_lds` (async DMA) | `view.async_load_lds_at(b, lds_ptr, elem_off, dwords=...)` |
| `make_naive_tensor_view_packed<...>(ptr, shape)` | `make_naive_tensor_view_packed(ptr, shape, dtype)` |
| `make_tile_window(view, lengths, origin)` | `make_tile_window(view, lengths, origin)` (or `view.tile(...)`) |
| `tile_window.set_window_origin(origin)` | `tile.move_to(*origin)` / `tile.shift_by(b, *deltas)` |
| `tensor_coordinate<Desc>(idx)` | `make_tensor_coordinate(b, desc, idx)` |
| `move_tensor_coordinate(desc, coord, step)` | `move_tensor_coordinate(b, coord, step)` |
| `transform_tensor_descriptor(...)` + `make_*_transform` | `ck_dsl.transforms.TensorDescriptor.naive(...).transform(...)`; wrap with `view_from_transforms_descriptor(ptr, rich_desc)` |
| `tile_distribution_encoding<Rs, Hs, Ps2RHs..., Ys2RHs...>` | `TileDistributionEncoding(Rs=..., Hs=..., Ps2RHs_major=..., Ps2RHs_minor=..., Ys2RHs_major=..., Ys2RHs_minor=...)` |
| `make_static_tile_distribution(encoding)` | `make_static_tile_distribution(encoding)` |
| `make_static_distributed_tensor<T, Distribution>()` | `make_static_distributed_tensor(distribution, dtype=...)` |
| `load_store_traits<Distribution>` (vector_dim_y + scalar_per_vector + sfc) | `make_load_store_traits(distribution, max_vec=...)` |
| `tile_window.load() -> distributed_tensor` | `load_tile(b, window, distribution=..., ps=[[tid]])` |
| `tile_window.store(distributed)` | `store_tile(b, window, distributed, ps=[[tid]])` |
| `sweep_tile(dt, [&](auto idx){...})` | `sweep_row_chunks(b, tile, body=..., cache=...)` *(simple form)*<br/>or `distributed_tensor.sweep(lambda y, v: ...)` *(distribution form)* |
| `block_tile_reduce_*` (sum / max) | `block_lds_reduce(b, val, lds, tid, ...)` |
| `block_sync_lds()` | `b.sync()` (now emits `s_waitcnt vmcnt(0) lgkmcnt(0)` before `s_barrier`) |
| `numeric<T>::min/max/lowest`, `type_convert<DstT, SrcT>` | `cast_to_f32`, `cast_f32_to`, `io_ir_type` |

### Two ergonomic paths

The DSL exposes the small-op authoring surface twice:

1. **Simple sweep** (recommended for 1D-row patterns: norm, reduce,
   elementwise). The author writes the per-thread chunked loop with
   `sweep_row_chunks(...)` and the helper handles tid arithmetic +
   `vec`-wide load + f32 promotion. The vector dim and chunk count
   are explicit kernel-author choices, not inferred.

2. **Distribution-driven** (recommended when the (Y, P) decomposition
   is non-trivial — multi-warp tiles, multi-dim Y space, MFMA-style
   distributions, or replicated workloads). The author writes a
   `TileDistributionEncoding` describing the (Rs, Hs, Ps, Ys)
   decomposition, then calls `window.load(b, distribution=...,
   ps=[[tid]])`. The helper's `LoadStoreTraits` picks the vector
   dim + width and the snake traversal order automatically.

Both paths lower through the same `TensorView` / `TileWindow` API and
land at the same `b.global_load_vN` / `b.smem_store_vN` IR ops.

Worked examples:

* `ck_dsl/examples/distribution_reduce_demo.py` — 1D row-sum reduce
  driven by a 1D distribution. Bit-exact vs `torch.sum(dim=-1)`.
* `ck_dsl/examples/distribution_2d_add_demo.py` — 2D tile add driven
  by a 2D distribution (Hs has 2 X dims, P has 2 contributors).
  Demonstrates `make_tile_window` over a runtime-stride view +
  `window.load(...) / window.store(...)` instance methods.
* `ck_dsl/instances/reduce.py` / `ck_dsl/instances/layernorm2d.py` —
  the matching simple-sweep versions (production today).

### Distribution feature matrix

| Feature | Status |
|---|---|
| `Rs == ()` (no replication) | done |
| `Rs != ()` (replication; major=0 routes to R buckets) | done |
| 1D X (single tile dim) | done |
| 2D X (two tile dims, P with multiple contributors) | done |
| 3D+ X | encoding accepts it, untested in demos |
| `LoadStoreTraits` smart picker (scans Y dims for stride-1 in X) | done |
| `LoadStoreTraits` scalar fallback (no stride-1 Y) | done |
| Snake traversal (multi-axis Gray-code-style) | done |
| Row-major (non-snake) traversal | done (`iterate_accesses(snake=False)`) |
| `TileWindow.load(distribution=...)` / `store(...)` methods | done |
| Validity / mask threading through `load_tile` | not yet (use raw rich-descriptor for now) |

The cycle of a typical kernel author becomes:

```python
from ck_dsl import (
    IRBuilder, F16, I32, PtrType,
    WarpGrid, MfmaAtom, mfma_atom,
    LdsLayout, CoalescedTileLoader, AsyncTileLoader,
    SchedulePolicy, SoftwarePipeline,
    DirectEpilogue, CShuffleEpilogue,
    compile_kernel, make_gemm_manifest, write_artifact,
)
# IR construction + the helpers do the heavy lifting; the kernel author
# only writes the descriptor callback (the "what is this op" part).
```

### Worked example: row-wise reduce in the CK Tile style

A complete row-reduction kernel using only the small-op layer (one CTA
per row, vec-wide chunks, LDS tree reduction, scalar write per row):

```python
from ck_dsl.core.ir import F32, I32, IRBuilder, PtrType
from ck_dsl.helpers import (
    SignatureBuilder, ceil_div_grid, kernel_name_join,
    io_ir_type, store_scalar_from_f32, block_lds_reduce,
    make_lds_view, make_naive_tensor_view_packed, make_tile_window,
    sweep_row_chunks,
)

def build_row_sum(*, n_per_block: int, block_size: int = 256, vec: int = 8,
                  dtype: str = "f16"):
    io_ty = io_ir_type(dtype)
    b = IRBuilder(kernel_name_join("row_sum", dtype, f"N{n_per_block}"))
    b.kernel.attrs["max_workgroup_size"] = block_size

    X = b.param("X", PtrType(io_ty, "global"), noalias=True, readonly=True, align=16)
    Y = b.param("Y", PtrType(io_ty, "global"), noalias=True, writeonly=True, align=16)
    M = b.param("M", I32)  # noqa: F841 -- ABI symmetry with the C++ reference

    tid = b.thread_id_x()
    row = b.block_id_x()

    x_view = make_naive_tensor_view_packed(X, shape=(1, n_per_block), dtype=io_ty)
    x_tile = make_tile_window(
        x_view, lengths=(1, n_per_block), origin=(row, b.const_i32(0))
    )
    lds = make_lds_view(b, dtype=F32, shape=(block_size,)).base

    acc = b.const_f32(0.0)
    def body(_n_off, x_scalars):
        nonlocal acc
        for xi in x_scalars:
            acc = b.fadd(acc, xi)

    sweep_row_chunks(
        b, x_tile, tid=tid, block_size=block_size, vec=vec,
        elems_per_thread=n_per_block // block_size, body=body,
    )
    total = block_lds_reduce(b, acc, lds, tid, block_size=block_size, combine="sum")

    with b.scf_if(b.cmp_eq(tid, b.const_i32(0))):
        store_scalar_from_f32(b, Y, row, total, dtype=dtype)

    return b.kernel

# Grid / signature:
grid = ceil_div_grid((M, 1))  # one CTA per row
sig = SignatureBuilder().ptr("X", dtype).ptr("Y", dtype).scalar("M", "i32").build()
```

That's the entire kernel — ~25 lines of body, no `smem_alloc` / no
`global_load_vN` / no manual `b.add(b.mul(row, N), col)` offset math.
The CK Tile-inspired layer recognises the row-sweep pattern and folds
all of that in.

## Atoms — `helpers/atoms.py`

`MfmaAtom` is the dataclass for one MFMA intrinsic, with everything a
kernel author needs to know in one place:

```python
from ck_dsl import MfmaAtom, MFMA_F16_ATOMS, mfma_atom

# Fixed accessors:
atom = MfmaAtom.f16_16x16x16()   # the legacy CDNA atom
atom = MfmaAtom.f16_16x16x32()   # gfx950, K-packed
atom = MfmaAtom.f16_32x32x8()    # 32x32 dispatcher hero
atom = MfmaAtom.f16_32x32x16()   # gfx950, K-packed 32x32
atom = MfmaAtom.f16_4x4x4()      # 16 batched 4x4s per wave (direct conv 4c)

# Or look up by shape:
atom = mfma_atom("f16", 32, 32, 16)
```

What `MfmaAtom` exposes:

```python
atom.m, atom.n, atom.k              # tile shape
atom.a_per_lane                     # halves per A operand per lane on wave64
atom.b_per_lane                     # halves per B operand per lane
atom.c_per_lane                     # floats per accumulator per lane
atom.dtype_in, atom.dtype_out       # "f16", "f32"
atom.name                           # "mfma_f32_16x16x32_f16"

# Dispatch to the right IRBuilder method:
acc_out = atom.emit(b, a_vec, b_vec, acc_in)

# Allocate a fresh per-lane <c_per_lane x float> accumulator:
acc_init = atom.zero_acc(b)

# Lane -> output position within one atom (the epilogue addressing):
row_off, col_off = atom.lane_to_output(b, lane, i)
# row_off, col_off are i32 SSA values in [0, m) x [0, n).
# i is the slot index in [0, c_per_lane).
```

The `lane_to_output` mapping is the part nobody wants to derive twice:

  - **16x16 atom** (`c_per_lane=4`):
    ```
    m_blk     = lane / 16
    n_in_atom = lane % 16
    row       = m_blk * 4 + i
    col       = n_in_atom
    ```
  - **32x32 atom** (`c_per_lane=16`):
    ```
    m_blk     = lane / 32  (∈ {0, 1})
    n_in_atom = lane % 32
    row       = (i // 4) * 8 + m_blk * 4 + (i % 4)
    col       = n_in_atom
    ```
  - **4x4 atom** (`c_per_lane=4`):
    ```
    batch     = lane / 4   (∈ {0..15})  # each batch is one independent 4x4
    lane_in_b = lane % 4
    row       = i
    col       = lane_in_b
    # Caller composes `batch` separately as the "group" or output-row index.
    ```

Getting any of these wrong is a silent correctness bug — the kernel
runs, the verify shows partial outputs in random places, and the
diff usually doesn't point at the lane mapping. Centralizing it
here is one of the larger correctness wins of the helpers layer.

## Geometry — `helpers/geometry.py`

`WarpGrid` packs the boilerplate every tile-MMA kernel re-derives
into one immutable view:

```python
from ck_dsl import WarpGrid, mfma_atom

atom = mfma_atom("f16", 32, 32, 16)
grid = WarpGrid.from_atom(
    atom,
    tile_m=128, tile_n=128, tile_k=32,
    warp_m=2, warp_n=2, warp_k=1,
)
# at this point `grid` is unbound: just compile-time constants
print(grid.block_size)                # 256 (= warp_m * warp_n * 64)
print(grid.mfmas_per_warp_m)          # 2  (= tile_m / (warp_m * warp_tile_m))

# Bind to a kernel builder; this emits the SSA values
b = IRBuilder("my_kernel")
grid = grid.bind(b)
# Now grid.tid, grid.lane, grid.warp_m_idx, grid.warp_n_idx,
# grid.block_m_off, grid.block_n_off are real i32 SSA values.
```

`grid.bind(b, block_m_axis="y", block_n_axis="x")` lets you choose
which `block_id_{x,y,z}` axis carries which dimension; the default
(`block.y -> M tile`, `block.x -> N tile`) matches CK Tile + the
`gemm_universal` kernel.

`grid.warp_m_off(b)` and `grid.warp_n_off(b)` return the per-warp
M / N offsets used by the epilogue helpers.

## Loads — `helpers/loads.py`

Two loaders share the same authoring contract:

  `LDS[row, col] = global[block_row_off + row, block_col_off + col]`
  for `row ∈ [0, tile_rows), col ∈ [0, tile_cols)`.

The kernel author provides a `descriptor(b, row, col) ->
(off_elements, valid_or_None)` callback that maps tile coordinates to
the *global* linear element offset. The loader handles per-thread
chunking and LDS layout.

### `CoalescedTileLoader` — sync (compv3-grade)

Per-thread `buffer_load_vN_f16` -> register -> `smem_store_vN_f16`:

```python
from ck_dsl import CoalescedTileLoader

A_smem = b.smem_alloc(F16, [block_m, block_k], name_hint="A_smem")
A_rsrc = b.buffer_rsrc(A_ptr, A_bytes)

def a_descriptor(b, row, col):
    # row in [0, tile_m), col in [0, tile_k); compute global offset
    m_global = b.add(block_m_off, row)
    k_global = b.add(k_off, col)
    return b.add(b.mul(m_global, K), k_global), None  # always valid

loader = CoalescedTileLoader.from_tile(
    tile_rows=block_m, tile_cols=block_k, block_size=grid.block_size,
)
loader.load(b, tid=grid.tid, smem_dst=A_smem, rsrc=A_rsrc,
            descriptor=a_descriptor)
```

`from_tile` picks the widest `load_vec` (halves per thread per chunk)
that distributes evenly across the block: 8, 4, 2, or 1.

### `AsyncTileLoader` — async direct DRAM→LDS (compv4-grade, runbook §6.3)

Same authoring surface, but emits `raw_ptr_buffer_load_lds` —
the DRAM→LDS hop happens in hardware without a register intermediate.
The runbook ranks this as the single biggest lever for memory-bound
direct-conv kernels (`~110 -> ~213 TFLOPS` jump on the 16c bake-off).

```python
from ck_dsl import AsyncTileLoader

loader = AsyncTileLoader.from_tile(
    tile_rows=block_m, tile_cols=block_k,
    block_size=grid.block_size, wave_size=64,
)
slot = loader.bind(b, smem_dst=A_smem, wave_id=grid.warp_id)
slot.issue(b, tid=grid.tid, rsrc=A_rsrc, descriptor=a_descriptor)

# ... emit MFMAs / other work that does NOT consume A yet ...

b.s_waitcnt(vmcnt=0)  # drain the async LDS writes
b.sync()
# now safe to read A_smem
```

Restrictions vs the sync loader:

  - `dwords ∈ {1, 3, 4}` (i.e. 2, 6, or 8 halves per lane); the
    intrinsic does *not* accept 2 dwords. `from_tile` picks the
    widest valid value.
  - The LDS destination is wave-uniform; lane `i` in a wave writes at
    `lds_base + i * dwords * 4`. Per-lane swizzles must live in the
    *consumer's* read math, not in the loader.
  - Consumers must place an `s_waitcnt(vmcnt=0)` before reading LDS
    (the intrinsic uses VMEM counters, not LGKM counters).

## Layouts — `helpers/layouts.py`

`LdsLayout` makes LDS row stride and async compatibility explicit:

```python
from ck_dsl import LdsLayout

# Winning implicit-GEMM bake-off layout: logical K=64, physical stride=72.
sync_layout = LdsLayout.padded_k(logical_cols=64, k_pad=8)
assert sync_layout.storage_shape(rows=64) == (64, 72)

# Async DMA layout: packed, because raw_ptr_buffer_load_lds writes
# lane-contiguous LDS bytes.
async_layout = LdsLayout.packed_async(logical_cols=64)
async_layout.validate_for_async()
```

`validate_for_async()` rejects K-padding and ad hoc swizzles. That encodes
the runbook lesson: async DMA's destination has to be packed/lane-contiguous;
if a swizzle is needed, express it in the consumer's LDS read math.

## Scheduling and Software Pipelines

CK Tile distinguishes two complementary scheduling modes; this repo
exposes both through `SchedulePolicy`:

* **Intrawave** — within one wave's instruction stream, interleave
  MFMA / DS_READ groups via `__builtin_amdgcn_sched_group_barrier` so
  the AMDGPU post-RA scheduler keeps the MFMA pipe fed while ds_reads
  for the next sub-tile are still streaming. Picked by
  `pipeline='intrawave'`, `'compv3'`, or `'compv4'`.

* **Interwave (ping-pong)** — across waves in the same workgroup, use
  `s_setprio(1)` at the start of every MFMA group and `s_setprio(0)`
  after, so the dispatcher arbitrates in favor of waves that are
  computing instead of waves that are blocked on `buffer_load` /
  `buffer_load_lds`. Picked by `pipeline='interwave'` /
  `'pingpong'` / `'async_dma'` (the async-DMA scheduler defaults to
  interwave because that is what overlaps DRAM→LDS bandwidth with
  MFMA throughput in a multi-wave block).

```python
from ck_dsl import SchedulePolicy

# Intrawave only (single-buffer compute pipeline).
intra = SchedulePolicy.for_pipeline("compv4")          # mode="intrawave"

# Interwave ping-pong (double-buffer async-DMA pipeline).
inter = SchedulePolicy.for_pipeline("interwave")        # mode="interwave"

intra.emit_prologue(b)
# Inside the MFMA k-loop body:
intra.emit_after_mfma_step(b, ds_read_count=2, mfma_count=4)

# In a software-pipelined ping-pong: SoftwarePipeline.run_ping_pong
# will call inter.emit_compute_prologue / _epilogue around each
# `compute(it, ...)` invocation so MFMA-heavy waves take dispatch
# priority while neighbours are still in flight on VMEM.
```

`SoftwarePipeline` builds the prologue / steady-state ping-pong:

```python
from ck_dsl import SoftwarePipeline

pipe = SoftwarePipeline(
    num_iters=K_iters,
    double_buffer=True,
    wait_vmcnt=True,
    sync_after_wait=True,
    sync_before_issue=True,   # ABA-hazard guard between iters
    overlap_vmcnt=True,       # vmcnt(1) keeps next load in flight
)
final_accs = pipe.run_ping_pong(
    b,
    buffers=[(A_smem, B_smem), (A_smem2, B_smem2)],
    initial_state=accs,
    issue_load=lambda it, bufs: emit_load_phase(it, *bufs),
    compute=lambda it, bufs, state: emit_mfma_phase(bufs[0], bufs[1], state),
    schedule=inter,           # adds setprio bookends around compute
)
```

Key invariants the helper guarantees:

1. **ABA hazard guard.** Before re-issuing an async load into a buffer
   that was just consumed, a workgroup-wide LDS-only barrier
   (`b.sync_lds_only()`, the canonical `block_sync_lds` pattern) drains
   the previous iter's ds_reads. Without this, slow waves' ds_reads can
   race with fast waves' writes against the same LDS slot. (`compute(it)`
   reads `buffers[it&1]`; `issue_load(it+2)` writes the same slot two
   iters later — that's the ABA window.) Disabled with
   `sync_before_issue=False` for kernels that prove the hazard cannot
   fire.

2. **Real overlap, not just a fence.** `overlap_vmcnt=True` rewrites the
   per-iter VMEM drain from the blunt `vmcnt(0)` (drains everything)
   to `vmcnt(1)` (drains only the *previous* outstanding async load),
   so the load for iter `it+1` keeps streaming while `compute(it)`
   issues its ds_reads + MFMAs. The trailing barrier is also
   downgraded from `b.sync()` to `b.sync_lds_only()` so the in-flight
   VMEM isn't drained by the workgroup-wide fence.

3. **Wave-level ping-pong.** When a `schedule` is passed and
   `schedule.mode == 'interwave'`, the helper wraps every `compute(...)`
   call in `s_setprio(high)` / `s_setprio(low)`. Pairs with `overlap_vmcnt=True`
   so that the dispatcher actively prefers MFMA-doing waves over
   load-issuing waves.

This keeps the async DMA machinery reusable: the kernel supplies the
descriptor-specific `emit_load_phase` and `emit_mfma_phase`; the helper
owns ping-pong ordering, vmcnt/lgkmcnt arithmetic, and the wave-prio
bookends.

## Epilogues — `helpers/epilogues.py`

Two epilogues, both consuming the per-warp accumulator list and an
output-address callback:

### `DirectEpilogue` — per-lane vec stores

Best when the atom's per-lane output layout is contiguous in the
output's fastest dim:

```python
from ck_dsl import DirectEpilogue

epi = DirectEpilogue(atom=atom, grid=grid)
epi.store(
    b,
    accs=acc_list,                 # mfmas_per_warp_m * mfmas_per_warp_n entries
    addr_fn=lambda b, m, n: (b.add(b.mul(m, N), n), None),
    d_rsrc=d_rsrc,
    bounds=(M, N),                 # optional OOB mask
    vec_in_acc=False,              # True if acc[0..c_per_lane-1] are
                                   # contiguous in the output's fastest dim
)
```

For atom shapes where lane elements are scattered (16x16, 32x32),
use `CShuffleEpilogue` instead.

### `CShuffleEpilogue` — LDS-staged wide vec stores (runbook §9.3)

Best when the atom's per-lane output layout is *not* contiguous in the
output store layout. Three-stage pattern (mirror of CK's
`cshuffle_epilogue.hpp`):

1. Each warp writes its accumulators into LDS at the MFMA output
   layout (one `ds_write_b16` per slot).
2. `block_sync_lds`.
3. A flat distribution of `block_size` threads reads `<store_vec x
   half>` from LDS in row-major order and issues one
   `buffer_store_vN_f16` per thread.

```python
from ck_dsl import CShuffleEpilogue

epi = CShuffleEpilogue.from_grid(atom=atom, grid=grid, max_store_vec=8)
epi.store(
    b,
    accs=acc_list,
    addr_fn=lambda b, m, n: (b.add(b.mul(m, N), n), None),
    d_rsrc=d_rsrc,
    bounds=(M, N),
)
```

`from_grid` picks the widest `store_vec` that distributes the tile
evenly across the block: 8, 4, 2, or 1.

## Compile + manifest — `helpers/compile.py`, `helpers/manifest.py`

```python
from ck_dsl import compile_kernel, make_gemm_manifest, write_artifact

# 1. Compile (IR -> LLVM IR text -> HSACO via libamd_comgr).
artifact = compile_kernel(kernel, isa="amdgcn-amd-amdhsa--gfx950")
# artifact.hsaco         : bytes ready for hipModuleLoadData
# artifact.ir_text       : MLIR-style IR dump
# artifact.llvm_text     : AMDGPU LLVM IR text
# artifact.timings       : dict of {ir_build, ir_lower_llvm, comgr_bc,
#                                   comgr_relocatable, comgr_executable,
#                                   total} ms

# 2. Build the manifest.json that `ck_dsl.run_manifest` consumes.
manifest = make_gemm_manifest(
    artifact=artifact,
    block_m=128, block_n=128, block_k=32,
    threads_per_block=256,
    default_shape=(3328, 4096, 4096),
    atoms=["tile.mfma_f32_32x32x16_f16"],
)

# 3. Write the (hsaco, ir.txt, ll, manifest.json) bundle to disk.
write_artifact(artifact, out_dir, manifest)
```

For convolution: `make_conv_manifest(conv=[N, H, W, C, K, R, S,
sH, sW, pH, pW, dH, dW], groups=..., cpg=..., kpg=..., ...)`.

## Launching + workspace — `ck_dsl.runtime.launcher`

For ops that talk directly to torch tensors (instead of going through
the manifest runner), the canonical launch path is the launcher
abstractions in `ck_dsl.runtime.launcher`. They capture the
"compile-once, launch-many, workspace-survives-every-call" contract
that CK Tile's `fmha_bwd_launcher`, FlyDSL's `_TorchReduceWrapper`,
and Triton's `JITFunction` all share.

```python
from ck_dsl import compile_kernel
from ck_dsl.runtime import (
    KernelLauncher, PipelineLauncher, WorkspacePool, LaunchConfig,
)

# 1. Compile + load once per problem shape.
artifact = compile_kernel(my_op_kernel(spec))
launcher = KernelLauncher(
    hsaco=artifact.hsaco,
    kernel_name=artifact.kernel_name,
    signature=my_op_signature(),       # the same dict-list shape
                                       # `pack_args` expects
    cache_key=("my_op", spec_tuple),   # for your dispatch cache
)

# 2. Optional: stash workspace tensors that outlive every call.
pool = WorkspacePool()
scratch = pool.get(
    "scratch", (problem.M, problem.N), dtype=torch.float32, device=q.device,
)

# 3. Launch -- this is the entire hot path.
launcher(
    {"out_ptr": out, "in_ptr": q, "scratch_ptr": scratch, ...},
    config=LaunchConfig(grid=(gx, gy, gz), block=(bx, 1, 1), stream=0),
)
# stream=0 is auto-resolved to torch.cuda.current_stream() so torch's
# caching allocator sees the launch and won't reuse `scratch` mid-flight.
```

For multi-kernel pipelines (split-KV attention's seg+reduce, k-fixup
GEMM, im2col+GEMM+col2im, ...) construct a `PipelineLauncher` whose
stages share one stream:

```python
seg = KernelLauncher(hsaco=seg.hsaco, ..., cache_key=("seg",) + key)
red = KernelLauncher(hsaco=red.hsaco, ..., cache_key=("red",) + key)
pipeline = PipelineLauncher([seg, red])
pipeline(
    [seg_vals, red_vals], [seg_cfg, red_cfg], stream=int(stream),
)
```

For numpy / manifest flows (no torch in scope), use `DeviceMem` (a
RAII wrapper over `Runtime.alloc/free`) instead of `WorkspacePool`,
and `time_launches(fn, warmup=..., iters=...)` for HIP-event timing.

See `ck_dsl/instances/attention_unified.py::_get_3d_pipeline` and
`::_get_2d_launcher` for the in-tree references.

## Roadmap

Phase 1 (today):
  - All helpers shipped above are production-quality and used by the
    bake-off generators (`ck_dsl.examples.bake_off_*`).

Phase 2:
  - `MfmaAtom` extension for bf16, fp8, and `smfmac` (sparse matmul).
  - `PersistentKernel` wrapper for multi-tile-per-CTA scheduling.
  - `StreamKEpilogue` for split-K accumulation via atomics.

## CDNA / chiplet-aware helpers

A second wave of helpers covering the AMDGPU-specific scheduling and
addressing tricks that high-performance matmul / attention kernels
need on CDNA3 / CDNA4. Each plays inside the same `IRBuilder` /
helper layering as everything above.

### Chiplet-aware grid swizzle (`helpers/grid.py`)

```python
from ck_dsl import chiplet_aware_super_tile, NUM_XCDS_MI300X

# At the very top of the kernel body, before any per-block math:
result = chiplet_aware_super_tile(
    b,
    b.block_id_x(),
    num_pid_m=ceil_div(M, BLOCK_M),
    num_pid_n=ceil_div(N, BLOCK_N),
    wgm=8,
    num_xcds=NUM_XCDS_MI300X,
    chunk_size=64,
)
pid_m, pid_n = result.row, result.col  # use these instead of blockIdx
```

The composition is `chiplet_transform_chunked` (XCD round-robin
reversal) followed by `super_tile_swizzle` (WGM-style Hilbert order
inside each XCD). Together they restore L2 locality on the multi-die
MI300X / MI350X by ensuring every contiguous stripe of workgroups
shares one XCD's L2 slice.

### XOR-based LDS swizzles (`helpers/layouts.py`)

```python
from ck_dsl import LdsLayout

# Closed-form bank-conflict-free swizzle for the canonical CK Tile
# shared-tile shapes:
layout = LdsLayout.xor_swizzled(tile_rows=32, tile_cols=32, elem_bytes=2)

# Apply the swizzle to a byte offset (producer or consumer side):
swizzled_bytes = layout.apply_swizzle_bytes(off_bytes)
```

Five canonical swizzles supported: `16x16`, `16x32`, `32x16`,
`32x32`, `16x128` (fp8). The XOR permutation guarantees
bank-conflict-free ds_read_b128 for the matching MFMA atom shape;
saves the ~6 % LDS overhead of a row-pad approach.

### SGPR scalarization (`IRBuilder.to_sgpr_u32`)

```python
lds_base = b.to_sgpr_u32(
    b.cast_i32(reinterpret_cast_address(&A_smem[0]) + warp_id * elem_per_warp * 2)
)
# `lds_base` is now an SGPR; subsequent uses don't pay v_readfirstlane each iteration.
```

`to_sgpr_u32(v) = pin_sgpr(readfirstlane(v))`. The `pin_sgpr` step
emits a no-op `asm sideeffect "", "=s,s"(...)` that forces the AMDGPU
register allocator to keep the value in an SGPR — without it, an
SGPR-uniform value computed once at the top of the kernel can get
re-materialized into a VGPR every loop iteration. The canonical
AMDGPU "scalarize this wave-uniform value" idiom.

### Wave-vote primitives

```python
all_below = b.wave_all(b.cmp_lt(max_diff, threshold))
# `all_below` is a wave-uniform i32 (1 or 0); single hardware op,
# no ds_bpermute ladder.

if_branch = b.cmp_eq(all_below, b.const_i32(1))
# ... skip the per-row rescale when every lane agrees ...
```

Lowered to `llvm.amdgcn.ballot.i64` + `icmp eq i64 ... -1`. Pairs
naturally with adaptive online-softmax rescaling: skip the rescale
pass entirely when every lane's `max_diff` stayed below threshold.

### Cache-coherency hints

```python
from ck_dsl.core.ir import CACHE_STREAM, NON_TEMPORAL

b.async_buffer_load_lds_addr(
    rsrc, lds_base, voff, soff, dwords=4,
    coherency=CACHE_STREAM,  # SLC=1, won't evict useful L2 lines
)
```

Maps to the AUX-byte AMDGPU buffer-load encoding. Use `CACHE_STREAM`
for one-shot streaming GEMM loads, `NON_TEMPORAL` for tail loads that
won't be reused.

### Per-MFMA setprio bookends (interwave ping-pong)

```python
from ck_dsl import SchedulePolicy

pol = SchedulePolicy.for_pipeline("interwave")
# Inside the MFMA loop body, wrap each MFMA atom:
pol.emit_mfma_setprio_bookend(b, lambda: b.mfma_f32_32x32x16_f16(...))
```

`s_setprio(1)` before the MFMA, `s_setprio(0)` after. Tells the
dispatcher to favour MFMA-issuing waves over VMEM-issuing waves at
single-MFMA granularity. The whole-`compute(...)` bookend pattern
remains as the outer wrapper (`run_ping_pong(... schedule=pol)`);
prefer the per-MFMA variant when one wave's `compute` body issues
many MFMAs interleaved with ds_reads.

### Scheduler-group barrier helpers (intrawave)

```python
# Inside one wave's K-loop body, after each MFMA group:
pol.emit_mfma_valu_pairs(b, pairs=4, valu_per_pair=2)
pol.emit_mfma_trans_pairs(b, pairs=2, trans_per_pair=1)
```

Emits `(MFMA, VALU)` and `(MFMA, TRANS)` alternating-group
`sched_group_barrier` hints — the right shape for attention-softmax
loops where each MFMA is followed by a fixed number of VALU sub /
mul / cmp or TRANS exp2 / log2 instructions. New mask constants
`VALU = 0x002`, `TRANS = 0x400` join the existing `MFMA`, `DS_READ`,
`DS_WRITE`, `VMEM_READ` masks.

### N-buffer software pipeline (quad-buffer)

```python
pipe = SoftwarePipeline(
    num_iters=K_iters,
    num_buffers=4,         # quad-buffer for deeper VMEM-prefetch parallelism
    wait_vmcnt=True,
    sync_after_wait=True,
    sync_before_issue=True,
    overlap_vmcnt=True,    # vmcnt(prefetch_depth) keeps prefetches in flight
)
final = pipe.run_ping_pong(b, buffers=[(A0,B0),(A1,B1),(A2,B2),(A3,B3)],
                          initial_state=accs,
                          issue_load=..., compute=...,
                          schedule=SchedulePolicy.for_pipeline("interwave"))
```

The pipeline prologue now issues `num_buffers - 1` loads (so the
steady-state can immediately overlap them with compute), and
`overlap_vmcnt=True` uses `s_waitcnt(vmcnt=num_buffers-1)` so the
in-flight prefetches stay alive across each iter's compute step.
Legacy `double_buffer=True` is still supported and defaults to
`num_buffers=2`.

### Half-block barrier (8-wave ping-pong)

```python
# Top of the kernel: stagger = warp_id / (NUM_WARPS / 2)
warp = b.div(tid, b.const_i32(64))
stagger = b.div(warp, b.const_i32(NUM_WARPS // 2))

# In one cluster, only the "upper" half synchronises:
b.sync_half_block(stagger)

# Later, the "lower" half synchronises:
b.sync_half_block(b.sub(b.const_i32(1), stagger))
```

Each call emits `if (selector) __builtin_amdgcn_s_barrier();`. Pairs
must match across the cluster or the workgroup will deadlock —
half-block barriers are valid only when every wave participates in
exactly one branch of every barrier pair.

### Occupancy launch bound

```python
b.kernel.attrs["max_workgroup_size"] = 256
b.kernel.attrs["waves_per_eu"] = 2      # int or (min, max) tuple
```

Emits `"amdgpu-waves-per-eu"="2,2"` on the kernel function attribute.
Equivalent to CUDA's `__launch_bounds__(NUM_THREADS, MIN_BLOCKS_PER_SM)`
— the second number is the waves-per-EU occupancy target.

## Autotuning (`helpers/autotune.py`)

CK DSL kernels build in <2 s each (LLVM IR + comgr), so a multi-config
search-the-tile-space sweep is cheap. The :class:`Autotuner` decorator
implements the same pattern Triton uses, but specialised for the
DSL's spec-dataclass pipeline:

* configs are :class:`Spec` instances (one per point in the search
  space), so the search space is type-checked at construction time;
* timings come from :func:`time_launches` (HIP events, same timer the
  production code path uses);
* winners cache in-memory **and** on disk as a small JSON file, so
  the second Python process pays zero overhead.

```python
from ck_dsl.helpers import Autotuner, AutotuneConfig, spec_replace

# 1) Define the search space as Spec instances.
base = UniversalGemmSpec(name="ugemm",
    tile=TileSpec(tile_m=128, tile_n=128, tile_k=32, warp_m=2, warp_n=2,
                  warp_k=1, warp_tile_m=32, warp_tile_n=32, warp_tile_k=16),
    trait=TraitSpec(pipeline="compv4", epilogue="cshuffle"))

configs = [
    AutotuneConfig(name="t128x128x32",         spec=base),
    AutotuneConfig(name="t128x128x32_chiplet", spec=spec_replace(
        base, trait=dataclasses.replace(base.trait, chiplet_swizzle=True))),
    AutotuneConfig(name="t256x128x32_chiplet", spec=spec_replace(base,
        tile=dataclasses.replace(base.tile, tile_m=256, warp_m=4),
        trait=dataclasses.replace(base.trait, chiplet_swizzle=True))),
    ...
]

# 2) Wire up the three callbacks (build, bench, launch).
@functools.lru_cache(maxsize=None)  # cache the compiled launcher
def _build(cfg):
    k = build_universal_gemm(cfg.spec)
    ir = lower_kernel_to_llvm(k)
    hsaco, _ = build_hsaco_from_llvm_ir(ir)
    return KernelLauncher(hsaco=hsaco, kernel_name=k.name,
                          signature=gemm_args_signature()), cfg.spec

def bench(cfg, *, M, N, K, dtype, A, B, **_):
    launcher, spec = _build(cfg)
    C = torch.empty((M, N), dtype=torch.float16, device="cuda")
    args = {"A": A, "B": B, "C": C, "M": M, "N": N, "K": K}
    bs = spec.tile.warp_m * spec.tile.warp_n * spec.tile.warp_k * 64
    cfgL = LaunchConfig(grid=((N+spec.tile.tile_n-1)//spec.tile.tile_n,
                              (M+spec.tile.tile_m-1)//spec.tile.tile_m, 1),
                        block=(bs, 1, 1))
    return time_launches(lambda: launcher(args, config=cfgL), warmup=10, iters=50)

def launch(cfg, *, M, N, K, dtype, A, B, C, **_):
    launcher, spec = _build(cfg)
    args = {"A": A, "B": B, "C": C, "M": M, "N": N, "K": K}
    bs = spec.tile.warp_m * spec.tile.warp_n * spec.tile.warp_k * 64
    cfgL = LaunchConfig(grid=((N+spec.tile.tile_n-1)//spec.tile.tile_n,
                              (M+spec.tile.tile_m-1)//spec.tile.tile_m, 1),
                        block=(bs, 1, 1))
    launcher(args, config=cfgL)

# 3) Wrap it all in an Autotuner; key on shape+dtype.
gemm = Autotuner(
    configs=configs,
    key_fn=lambda *, M, N, K, dtype, **_: (M, N, K, dtype),
    bench_fn=bench,
    launch_fn=launch,
    cache_path="~/.cache/ck_dsl_gemm_autotune.json",
    verbose=True,
)

# 4) Call. First time per shape sweeps + caches; subsequent times zero-overhead.
gemm(M=4096, N=4096, K=4096, dtype="fp16", A=A, B=B, C=C)
```

End-to-end behaviour from a single Python process:

```
[autotune] sweeping 8 configs for key=(4096, 4096, 4096, 'fp16')
[autotune]   128x128x32                    : 261.22 us/iter  (wall: 0.15s)
[autotune]   128x128x32_chiplet            : 260.48 us/iter  (wall: 0.07s)
[autotune]   256x128x32                    : 298.13 us/iter  (wall: 0.06s)
[autotune]   256x128x32_chiplet            : 294.67 us/iter  (wall: 0.07s)
[autotune]   128x256x32                    : 301.84 us/iter  (wall: 0.07s)
[autotune]   128x256x32_chiplet            : 292.50 us/iter  (wall: 0.07s)
[autotune] winner: 128x128x32_chiplet (260.48 us/iter)

  shape 4096x4096x4096: winner = 128x128x32_chiplet  (sweep wall: 0.5s)
  shape 4096x4096x4096: cached winner = 128x128x32_chiplet  (lookup: 12 us)
```

Eight configs in 0.5 s of wall time. The second call hits the cache
(12 μs lookup); a fresh Python process loading the same on-disk JSON
also skips the sweep entirely.

For comparison: a CK Tile equivalent sweep over the same 8 configs is
∼ 10–15 minutes of template instantiation per Python process restart;
Triton's autotune doesn't persist on disk and recomputes on every
restart unless you save the autotuner object.
