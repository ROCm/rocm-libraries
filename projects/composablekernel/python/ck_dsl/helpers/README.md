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
├── atoms.py           # MfmaAtom catalog (the matrix-multiply intrinsics)
├── geometry.py        # WarpGrid: block/warp/lane decomposition
├── loads.py           # CoalescedTileLoader + AsyncTileLoader
├── layouts.py         # LdsLayout: K-padding, packed async layouts, guardrails
├── schedule.py        # SchedulePolicy: named sched_group_barrier policies
├── pipeline.py        # SoftwarePipeline: prologue/steady-state/epilogue
├── epilogues.py       # DirectEpilogue + CShuffleEpilogue
├── compile.py         # compile_kernel() one-shot IR -> HSACO
└── manifest.py        # make_gemm_manifest, make_conv_manifest, write_artifact
```

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

## Roadmap

Phase 1 (today):
  - All helpers shipped above are production-quality and used by the
    bake-off generators (`ck_dsl.examples.bake_off_*`).

Phase 2:
  - `MfmaAtom` extension for bf16, fp8, and `smfmac` (sparse matmul).
  - `PersistentKernel` wrapper for multi-tile-per-CTA scheduling.
  - `StreamKEpilogue` for split-K accumulation via atomics.
