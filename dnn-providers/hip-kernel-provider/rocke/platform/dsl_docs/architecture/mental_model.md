# Mental Model

`rocke` is a Python-owned kernel authoring stack for CK Tile-style AMDGPU code. It keeps the important CK Tile concepts, but turns them into concrete Python data objects and first-class SSA IR instead of C++ templates.

The single most important shift in mental model is:

```text
Do not think "Python generates a HIP string."
Think "Python builds a typed SSA KernelDef, then a lowering engine lowers that object to AMDGPU LLVM IR
and libamd_comgr turns it into HSACO."
```

The `KernelDef` is the boundary between authoring and lowering. Instance builders and helpers emit high-level operations into `KernelDef.body`. Lowerers walk that body and produce LLVM IR (production), HIP C++ (debug), or CK Tile C++ from selected specs (parity).

There are two interchangeable lowering engines for the production LLVM-IR path: the native **Python** engine (`core/lower_llvm.py`) and a peer **C++** engine (`Cpp/`, a C99->C++20 port reached through the `rocke_engine` extension). They emit byte-identical LLVM IR. `ROCKE_BACKEND` (`cpp` | `python` | `both`) selects which one runs, resolved by `core/backend.py::resolve_backend`; the default is `cpp`, which auto-falls back to the Python engine when `rocke_engine` is not built. `both` runs both and asserts byte-identical output (the differential check).

## Layer Cake

```text
instances/
  Problem and spec dataclasses; kernel builders.

helpers/
  CK Tile-like authoring objects: tensor views, transform descriptors,
  MFMA atoms, tile loaders, epilogues, schedules, reductions, fusion,
  autotuning, manifest schema.

core/ir.py
  Typed SSA Values, Ops, Regions, Params, KernelDef, IRBuilder.

core/passes.py
  Conservative constant-fold + CSE + DCE.

core/lower_llvm.py
  Production backend (native Python engine): KernelDef -> AMDGPU LLVM IR text.
  A peer C++ engine (Cpp/, via the rocke_engine extension) lowers the same
  KernelDef to byte-identical LLVM IR and is the default backend.

runtime/comgr.py
  ctypes over libamd_comgr; LLVM IR -> bitcode -> relocatable -> HSACO.

runtime/hip_module.py, runtime/launcher.py, runtime/torch_module.py
  Load HSACO, pack args, resolve streams, launch, time, manage workspace
  and per-stream pending-args lifetime.
```

Two side paths:

- `core/lower_hip.py` — readable HIP C++ for debugging and clang repros. Narrower op coverage than LLVM lowering. Not the production runtime path.
- `core/lower_cktile.py` — CK Tile C++ emission from selected high-level specs (`UniversalGemmSpec`, `ImplicitGemmConvSpec`). Parity / reference only; does not consume `KernelDef`.

## Why It Exists

CK Tile is powerful, but several pieces are hard to iterate on in C++:

- template instantiation is slow for large dispatcher sweeps (often minutes per kernel);
- coordinate-transform DAGs are clean as algebra but verbose in C++ templates;
- non-bijective mappings (convolution padding, paged attention page tables) are not simple layout permutations;
- debugging a generated kernel is easier when the IR is small, inspectable, and generated in milliseconds (typical DSL warm compile: 5-30 ms).

`rocke` keeps the performance levers close to the hardware:

- explicit LDS allocation and layout (`tile.smem_alloc`, `LdsLayout`);
- raw AMDGPU buffer descriptors (`tile.buffer_rsrc` with DW3 = `0x00027000` on CDNA; the RDNA backends use the gfx10+ raw SRD word3 `0x31014000`);
- async DRAM-to-LDS via `raw_ptr_buffer_load_lds`;
- MFMA atoms keyed by dtype and shape (`MfmaAtom`);
- `s_waitcnt`, `s.barrier`, `sched_group_barrier`, `s_setprio`;
- cshuffle-style epilogues (`CShuffleEpilogue`);
- pointer aliasing, alignment, and `dereferenceable` metadata.

The DSL does not hide the GPU programming model. It gives names and reusable helpers to patterns kernel authors already need to control.

## Main Data Flow

```text
1.  Choose or create a spec dataclass (e.g. UniversalGemmSpec).
2.  Validate tile / dtype / layout / resource constraints (is_valid_spec).
3.  Build a KernelDef with IRBuilder (via the instance's build_* function).
4.  Optionally print MLIR-style text with print_ir(kernel) for inspection.
5.  Optionally run conservative IR passes (optimize_kernel) for cleanup.
6.  Lower KernelDef to AMDGPU LLVM IR (lower_kernel_to_llvm).
7.  Build HSACO via libamd_comgr (build_hsaco_from_llvm_ir).
8.  Launch through KernelLauncher / PipelineLauncher, or via
    `python -m rocke.run_manifest`.
9.  Verify correctness on adversarial shapes; benchmark with median + spread.
10. Inspect LLVM / ISA / VGPR / SGPR / LDS before claiming speed (analyze_*).
```

`helpers/compile.py::compile_kernel(kernel)` does steps 4-7 in one call and returns a `KernelArtifact` with the original `KernelDef`, the printed IR (optional), the LLVM IR text, the HSACO bytes, and per-stage timings.

## What An Instance Builder Owns

An instance builder owns the semantic contract of an operation. For example, `build_universal_gemm` owns:

- argument list and ABI order (`gemm_args_signature`);
- dtype / layout assumptions (`DataSpec`);
- grid shape (`block_id_y * tile_m`, `block_id_x * tile_n`, optional `block_id_z` batch);
- workgroup size (`max_workgroup_size`) and waves-per-EU hints;
- block / warp / MFMA tiling (`TileSpec`);
- LDS allocations and layout;
- global-to-LDS copy plan (sync vs async, vector width);
- K-loop structure (`scf.for_iter` with loop-carried accumulators);
- accumulator initialization (zero `<c_per_lane x f32>` per warp tile);
- epilogue mapping from per-lane accumulator slots to output coords (direct or cshuffle);
- launch manifest metadata (`make_gemm_manifest`).

The builder should not own generic primitives. Shared logic belongs in `helpers/`.

## What Helpers Own

Helpers are not a second compiler. They emit normal `IRBuilder` operations, but factor out repeated CK Tile patterns:

- `TensorDescriptor` and `TensorView` describe memory shapes and access rules.
- `TileWindow` moves a tile origin over a view.
- `TileDistributionEncoding` describes distributed register tiles.
- `MfmaAtom` describes one matrix instruction's shape and lane layout.
- `WarpGrid` decomposes block / warp / lane geometry and per-CTA offsets.
- `CoalescedTileLoader` emits classic global -> VGPR -> LDS copies.
- `AsyncTileLoader` emits `raw_ptr_buffer_load_lds` copies.
- `LdsLayout` centralizes padding and packed-async constraints.
- `SchedulePolicy` emits `sched_group_barrier`, `s_setprio`, and wait policies.
- `SoftwarePipeline` emits prologue / steady-state / epilogue staging.
- `DirectEpilogue` and `CShuffleEpilogue` emit output stores.
- `block_lds_reduce` emits canonical row / block reductions.
- `helpers/attention.py` provides `OnlineSoftmaxState`, `PagedKvDescriptor`, `warp_xor_reduce_*`, mask helpers, and 2D / 3D path selectors.
- `helpers/quant.py` provides FP8 / BF8 / I8 quantization wrappers.
- `helpers/fuse.py` provides graph-level fusion patterns (matmul + bias + activation, etc.).
- `helpers/autotune.py` provides `Autotuner` for in-process spec selection.
- `helpers/grid.py` provides chiplet-aware grid swizzles for multi-XCD locality.

The test for a good helper is simple: it should let multiple instance builders share a tricky hardware pattern without obscuring the shape-specific choices.

## What The Core IR Owns

Small and deliberate:

```text
Type, VectorType, PtrType, SmemType
Value, Op, Region, Param, KernelDef
IRBuilder
```

The IR keeps operations at the level the DSL needs to reason about. Examples (not exhaustive — see `reference/op_vocabulary.md`):

```text
tile.smem_alloc / tile.smem_load_vN / tile.smem_store_vN
tile.buffer_rsrc / tile.buffer_load_vN_f16 / tile.async_buffer_load_lds_addr
tile.mfma_f32_32x32x16_f16
tile.s_waitcnt / tile.sync / tile.sched_group_barrier
tile.readfirstlane / tile.wave_all / tile.ds_bpermute
arith.cvt_fp8_to_f32 / arith.cvt_f32_to_i8_sat
scf.for / scf.for_iter / scf.if / scf.yield
```

Lowering can then map these nearly one-to-one to AMDGPU LLVM intrinsics and control-flow blocks.

## Host-Time vs Device-Time

The most important rule in the DSL.

Host-time decisions are Python decisions. Use them for compile-time constants:

```python
for i in b.unroll(0, tile_k, atom.k):
    ...
```

Device-time decisions use SSA values and IR control flow:

```python
with b.scf_if(valid):
    ...
```

`Value.__bool__` raises `TypeError`. This prevents a runtime predicate from accidentally controlling Python code:

```python
if value:        # raises
b.scf_if(value)  # correct for runtime predicates
b.static_if(host_bool, lambda: ...)  # correct for Python-time decisions
```

The verified test `test_ssa_value_cannot_be_used_as_python_bool` pins this.

## Architecture Targets

Default target: `amdgcn-amd-amdhsa--gfx950` (CDNA3 / MI355X-class). The DSL also supports the other CDNA targets `gfx942` (and the older `gfx908 / gfx90a` for the atoms that exist there) and the RDNA WMMA targets `gfx1151` (RDNA3.5, wave32) and `gfx1201` (RDNA4, wave32). The ISA backend is selected from the gfx string by `core/isa/backend.py::backend_for` (`BACKEND_REGISTRY`). The K-packed f16/bf16 atoms (`16x16x32`, `32x32x16`) are gfx950-only; gfx942's f16/bf16 catalog is `{16x16x16, 32x32x8}`. Wave size is 64 on the CDNA backends and 32 on the RDNA (gfx11/gfx12) WMMA backends.

The `_DATALAYOUT` string in `core/lower_llvm.py` is the clang-emitted gfx950 layout. The lowerer also selects an LLVM **flavor** (`llvm20` vs `llvm22`) keyed by ROCm release (llvm22 for ROCm >= 7.2), which adjusts a small set of intrinsic signatures; see `_detect_llvm_flavor` / the `llvm_flavor=` override.

## The Most Common Failure Modes

- Treating `print_ir()` output as the compiler input. It is not; LLVM lowering walks `KernelDef` directly.
- Branching on an SSA `Value` in Python instead of emitting `scf_if`.
- Forgetting that async DRAM-to-LDS completion is guarded by VMEM waitcnt, not just LDS waitcnt.
- Swizzling an async LDS destination pointer. `raw_ptr_buffer_load_lds` writes lane-contiguous bytes; swizzles must be in consumer read arithmetic.
- Changing an MFMA atom without changing `lane_to_output` and the epilogue vectorization width.
- Pinning a non-uniform value with `pin_sgpr` (only useful after `readfirstlane`; prefer `to_sgpr_u32`).
- Comparing graph / pipeline amortized timing against per-launch cold timing without labeling the difference.
- Trusting a performance number before correctness checks (`run_manifest --verify`) and resource inspection (`analyze_hsaco`).

These each have a corresponding section in the per-area docs; see the relevant file in `instances/`, `runtime/`, or `optimization/`.
