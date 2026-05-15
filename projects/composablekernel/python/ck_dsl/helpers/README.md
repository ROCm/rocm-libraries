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

CK Tile parity (which C++ name maps to which Python helper):

| CK Tile C++ | DSL helper |
| --- | --- |
| `make_naive_tensor_descriptor_packed(shape)` | `TensorDescriptor.packed(shape, dtype)` or `make_naive_tensor_descriptor_packed(shape, dtype)` |
| `make_tensor_view<addr_space::global>(ptr, desc)` | `make_global_view(ptr, shape, dtype)` |
| `make_tensor_view<addr_space::lds>(...)` | `make_lds_view(b, dtype=..., shape=...)` |
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

`SchedulePolicy` centralizes scheduler hint masks:

```python
from ck_dsl import SchedulePolicy

policy = SchedulePolicy.for_pipeline("compv4")
policy.emit_prologue(b)
policy.emit_after_mfma_step(b, ds_read_count=2, mfma_count=4)
```

`SoftwarePipeline` builds a static prologue / steady-state pipeline:

```python
from ck_dsl import SoftwarePipeline

pipe = SoftwarePipeline(num_iters=K_iters, double_buffer=True,
                        wait_vmcnt=True, sync_after_wait=True)
final_accs = pipe.run_ping_pong(
    b,
    buffers=[(A_smem, B_smem), (A_smem2, B_smem2)],
    initial_state=accs,
    issue_load=lambda it, bufs: emit_load_phase(it, *bufs),
    compute=lambda it, bufs, state: emit_mfma_phase(bufs[0], bufs[1], state),
)
```

This keeps the async DMA machinery reusable: the kernel supplies the
descriptor-specific `emit_load_phase` and `emit_mfma_phase`; the helper owns
ping-pong ordering and `s_waitcnt(vmcnt=0)` placement.

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
