# Authoring Model

This page explains how a kernel author moves from an operation idea to a `KernelDef`. The shape:

```text
operation contract
  -> problem/spec dataclass
  -> validation
  -> argument signature
  -> grid + block geometry
  -> descriptors / views
  -> data movement
  -> compute loop
  -> epilogue
  -> manifest / runtime metadata
```

Every shipped instance in `instances/` follows this exact shape. The shared scaffolding lives in `helpers/spec.py` (`IOSpecRule`, `validate_io`, `SignatureBuilder`, `kernel_name_join`, `ceil_div_grid`).

## 1. Define The Operation Contract

Before writing IR, write down:

- the mathematical operation in index form;
- input, output, and accumulator dtypes;
- tensor layouts and strides (which dims are stride-1, which are runtime);
- which dimensions are compile-time constants in the spec;
- which dimensions are runtime kernel arguments;
- boundary behavior (tails, padding, masking, empty rows);
- whether atomics / nondeterminism / split-K / workspace are allowed;
- a reference implementation and a tolerance policy.

In `ck_dsl`, many performance decisions are encoded in the spec and the helper choices. A vague contract bakes in accidental assumptions.

Concrete contract examples (all in `instances/`):

- `UniversalGemmSpec` — GEMM tile, trait, data, layout, scheduler, epilogue.
- `ConvProblem` — NHWC/KRSC/NHWK convolution geometry; derives `Ho`, `Wo`, `M_gemm`, `flops`.
- `UnifiedAttentionProblem` — paged-attention shape; selectors choose 2D vs 3D.
- `Reduce2DSpec`, `LayerNorm2DSpec`, `RMSNorm2DSpec`, `ElementwiseSpec` — small-op contracts.

## 2. Validate Early

`is_valid_spec(spec) -> (ok, reason)` rejects impossible or unsupported configurations before IR is built. Use `helpers/spec.py::IOSpecRule + validate_io` for the common small-op shape:

```python
ok, why = validate_io(IOSpecRule(
    dtype=spec.dtype,
    block_size=spec.block_size,
    vec=spec.vec,
    n_per_block=spec.n_per_block,
    max_elems_per_thread=64,
))
```

`IOSpecRule` defaults:

```text
allowed_dtypes      = ("f16", "fp16", "bf16")
allowed_block_sizes = (64, 128, 256, 512, 1024)
allowed_vecs        = (2, 4, 8)
```

For GEMM / conv, validation also covers:

- supported architecture (current focus: gfx950);
- supported MFMA atom shape for the dtype;
- `tile_m, tile_n` divisible by `warp_* * warp_tile_*`;
- `tile_k` divisible by `warp_tile_k`;
- block size <= hardware/lowering limit;
- `block_size = warp_m * warp_n * warp_k * wave_size` consistent;
- LDS bytes under the per-block budget;
- vector load widths divide tile shape;
- requested epilogue / pipeline / scheduler names recognized.

Validation is part of the performance story: many "optimizations" are illegal unless they preserve tile / atom / LDS invariants.

## 3. Build The ABI With IRBuilder Params

Kernel arguments are declared with `b.param(...)`:

```python
b = IRBuilder(spec.kernel_name())
b.kernel.attrs["max_workgroup_size"] = block_size
b.kernel.attrs["waves_per_eu"] = (lo, hi)   # optional

A = b.param("A", PtrType(F16, "global"), noalias=True, readonly=True, align=16)
B = b.param("B", PtrType(F16, "global"), noalias=True, readonly=True, align=16)
C = b.param("C", PtrType(F16, "global"), noalias=True, writeonly=True, align=16)
M = b.param("M", I32)
N = b.param("N", I32)
K = b.param("K", I32)
```

Pointer attributes matter. LLVM lowering preserves alias / access / alignment / dereferenceable metadata. This is one of the levers that replaces template-side compiler magic. Verified by the test `test_param_metadata_lowers_to_llvm_arg_attrs`.

`max_workgroup_size` controls the emitted AMDGPU flat-workgroup attribute (`"amdgpu-flat-work-group-size"="64,N"`). Launching with more threads than `N` triggers `hipErrorLaunchFailure`.

The signature dict list for the launcher comes from `helpers/spec.py::SignatureBuilder` (or the family-specific helper in `helpers/manifest.py`):

```python
sig = (SignatureBuilder()
       .ptr("A", spec.data.dtype_a)
       .ptr("B", spec.data.dtype_b)
       .ptr("C", spec.data.dtype_c)
       .scalar("M", "i32").scalar("N", "i32").scalar("K", "i32")
       .build())
```

## 4. Compute Grid Coordinates

Most kernels begin with:

```python
tid     = b.thread_id_x()
lane    = b.lane_id()          # 0..63, wave64
warp    = b.div(tid, b.const_i32(64))   # if block_size > 64
block_x = b.block_id_x()
block_y = b.block_id_y()
block_z = b.block_id_z()
```

`helpers/geometry.py::WarpGrid` packages this for matrix kernels and exposes `mfmas_per_warp_m / n`, `k_atoms_per_tile_k`, and per-CTA `block_m_off / block_n_off`.

Grid conventions in shipped instances:

```text
GEMM:                 (ceil(N/tile_n), ceil(M/tile_m), batch?)
implicit-GEMM conv:   (ceil(K_out/tile_n), ceil(M/tile_m), 1)
direct 16c conv:      (ceil(W/block_q), groups/block_groups, N)
direct 4c conv:       groups packed across wave lanes/batches
reduce / norm:        one CTA per row
elementwise:          1D grid over contiguous elements
attention 3D tiled:   (q_blocks, kv_heads, split_kv_segments)
```

Chiplet swizzle (`chiplet_swizzle=True`, `helpers/grid.py::super_tile_swizzle`) is available for selected GEMM / conv paths. It is a launch-grid remap (improves L2 reuse on multi-XCD GPUs); the math is unchanged.

## 5. Describe Memory Instead Of Hand-Expanded Offsets

Prefer descriptors and views over raw arithmetic:

```text
plain contiguous tensors    -> TensorView, make_global_view, make_naive_tensor_view_packed
buffer-resource guarded     -> make_buffer_resource, make_buffer_view
non-bijective addressing    -> ck_dsl.transforms.TensorDescriptor (transform DAG)
tile-local movement         -> TileWindow
distributed register tiles  -> TileDistributionEncoding + StaticDistributedTensor
```

The transform DAG is essential for:

- convolution `(m, k) -> NHWC` and `(k_out, k_gemm) -> KRSC`;
- output `(m, k_out) -> NHWK`;
- paged-KV attention table lookup (`indirect`);
- dynamic attention bounds / masks (`pad_dynamic`).

A descriptor callback consumed by loaders has the shape:

```python
def a_desc(b, row, col):
    off, valid = rich_desc.offset(b, m=row, k=col)
    return off, valid
```

Loaders and epilogues only need `(offset_in_elements, valid_or_None)`. They do not need to know whether the mapping is GEMM, conv, or attention.

## 6. Choose Data Movement

For GEMM-like tiles there are two primary patterns.

`CoalescedTileLoader` (sync, classic):

```text
global / buffer load -> VGPR vector -> LDS store -> b.sync() -> LDS reads
```

`AsyncTileLoader` (compv4-style):

```text
raw_ptr_buffer_load_lds (DRAM -> LDS directly) -> b.s_waitcnt(vmcnt=0) -> LDS reads
```

Async constraints:

- `dwords in {1, 3, 4}`;
- LDS writes are lane-contiguous;
- destination base must be uniform within a wave;
- consumers must wait on VMEM before reading;
- swizzles belong in consumer read arithmetic, not in the destination pointer.

For row-wise small ops, use `helpers/sweep.py::sweep_row_chunks` and the `helpers/io.py` dispatchers (`load_vec_as_f32`, `pack_f32_to`) instead of building tile loaders.

## 7. Emit Compute

Matrix kernels follow:

```text
allocate f32 accumulators (one vector per warp tile MFMA fragment)
for K tile in scf_for_iter:
    load A/B (sync or async)
    wait/sync
    for kk in static_for(0, tile_k, atom.k):
        for warp_m fragment:
            A_frag = smem_load_vN_f16(A_smem, ...)
        for warp_n fragment:
            B_frag = smem_load_vN_f16(B_smem, ...)
        for output fragment:
            acc = atom.emit(b, A_frag, B_frag, acc)
            schedule_policy.emit_after_mfma_step(b, ...)
    yield updated acc
```

Reduction / norm kernels follow:

```text
each thread sweeps row chunks (sweep_row_chunks)
accumulate f32 local partial
block_lds_reduce
thread 0 or pass-2 writes output
```

Attention tiled kernels combine both:

```text
stage Q to LDS
iterate K/V pages or segments through paged-KV descriptor
compute QK with MFMA
apply masks (causal, sliding, ALiBi, QQ-bias, softcap)
online softmax update (warp_xor_reduce_max/sum)
compute PV with MFMA (ds_read_tr16_b64 for V)
write final output (or segment workspace for 3D)
```

## 8. Emit Epilogue

`DirectEpilogue` and `CShuffleEpilogue` from `helpers/epilogues.py`. The epilogue must agree with `MfmaAtom.lane_to_output`.

Direct epilogue when:

- per-lane outputs are naturally contiguous (e.g. `f16_4x4x4` direct grouped conv);
- output tile is small;
- LDS budget is tight.

CShuffle epilogue when:

- MFMA accumulator ownership is scattered across output coordinates;
- direct stores would be scalar or poorly coalesced;
- you want `buffer_store_dwordx{2, 4}` on the final stores.

Never swap MFMA atom shape without revisiting `lane_to_output` and the epilogue vectorization width.

## 9. Return A Kernel, Then Compile Or Manifest

Builders return `KernelDef`. They should not normally compile inside the builder.

```python
kernel = build_universal_gemm(spec)
art    = compile_kernel(kernel)
```

For examples and benchmarkable flows, emit a manifest:

```python
manifest = make_gemm_manifest(artifact=art, block_m=..., block_n=..., block_k=...,
                              threads_per_block=spec.block_size,
                              default_shape=(3328, 4096, 4096),
                              atoms=["mfma_f32_32x32x16_f16"])
paths = write_artifact(art, Path("build/ck_dsl_example"), manifest)
```

`python -m ck_dsl.run_manifest` is the portable execution path for the resulting `(hsaco, manifest.json)`.

## 10. Authoring Checklist

Before considering a new builder done:

- contract documented in a spec dataclass with a stable `kernel_name()`;
- `is_valid_spec(spec)` rejects unsupported layouts / dtypes / tile shapes / resources;
- every runtime predicate is expressed with `scf_if` or `select`, never a Python `if value:`;
- padding / tail behavior has an OOB-safe load/store path (buffer-rsrc + sentinel);
- LDS layout and async constraints are explicit (`LdsLayout`);
- MFMA atom and epilogue lane mapping agree (verified by inspection on a tiny shape);
- manifest signature and grid helper match the kernel ABI;
- correctness is checked against a reference (`run_manifest --verify`, or a torch / numpy oracle in a parity harness);
- benchmark reports median + spread, not a single lucky run (`benchmark_manifest(..., attempts=5, discard_first=True)`);
- generated LLVM / ISA / resource summaries are inspected for the intended primitive (`analyze_llvm_ir`, `analyze_hsaco`);
- the new path is added to one of the test suites (`python/test/test_ck_dsl.py` or `test_ck_dsl_examples.py`).
