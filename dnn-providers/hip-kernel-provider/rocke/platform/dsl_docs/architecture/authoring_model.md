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

Path notation in this page is relative to three explicitly named roots:

```text
<project_root>  = the rocKE component root containing platform/ and library/
<platform_root> = <project_root>/platform
<library_root>  = <project_root>/library
```

These placeholders describe source locations; Python import names are called
out separately where they differ.

Matrix-operation choices start from the exact gfx target. Resolve
`ArchTarget.from_gfx(...)`, select an `MmaOp` from that target's `MmaCatalog`,
and validate the operation's wave and layout contract. The exact catalog is
listed in [`kernel_taxonomy.md`](./kernel_taxonomy.md).

Kernel definitions in
[`<platform_root>/python/rocke/instances/`](../../python/rocke/instances/) and
[`<library_root>/kernels/`](../../../library/kernels/) follow the same
operation-to-runtime progression. Reusable authoring mechanics belong in
[`<platform_root>/python/rocke/helpers/`](../../python/rocke/helpers/);
foundational IR, analysis, and lowering mechanisms belong in
[`<platform_root>/python/rocke/core/`](../../python/rocke/core/). For example,
the shared spec scaffolding lives in
[`<platform_root>/python/rocke/helpers/spec.py`](../../python/rocke/helpers/spec.py)
(`IOSpecRule`, `validate_io`, `SignatureBuilder`, `kernel_name_join`,
`ceil_div_grid`).

## Kernel Authoring And Optimization Outputs

Kernel authoring means producing a spec-driven kernel definition, avoid producing
a one-off script. Kernel definitions compose reusable mechanisms from
`rocke.helpers` and `rocke.core`; the instance directory must not be treated as a
reusable-code layer. Any change that affects emitted IR must include its
matching C++ mirror and byte-identity coverage in the same change.

Kernel optimization means a reproducible recipe plus a chosen implementation. If
an AI-assisted session finds a better tile, schedule, prefetch, split,
vectorization, or other lever, capture both the final code path and the evidence
that justified it.

Put experiments in the example folder for the kernel being studied.
Keep benchmark scripts, shape files, qualitative mechanism notes, and
case-study methods close to the workload, for example under
`<platform_root>/python/rocke/examples/<arch>/<kernel_or_workload>/`. Keep
measured values, generated traces, and large logs outside the source tree in an
AMD-approved, access-controlled record with the revision, toolchain, gfx target,
shape set, and replay command.

Document every accepted optimization as a public qualitative case study. Describe
the workload shape class, candidate levers, commands, selected mechanism, rejected
mechanisms, and target constraints, *never* publishing measured values or comparative
performance claims.

Promote reusable optimization knowledge into `dsl_docs/optimization/`. If the
work discovers a general tactic, decision rule, debugging skill, or reusable
performance lever, add or update the relevant optimization skill or runbook docs
so future kernels can reuse it.

Wire reusable kernels into the kernel library's registry and test path. Include
a builder in byte-identity coverage if it emits IR through both Python and C++
engines.

Do not wire one-off benchmark scripts into production dispatch by default. First
separate the reusable builder from workload-specific measurement code. Only add
dispatch, manifest, or heuristic wiring once the supported shapes, architectures,
dtype contract, and fallback behavior are documented.

Optimization evidence must be replayable. Prefer checked-in shape files, small
benchmark configs, and command snippets over prose-only claims. If raw traces or
logs are too large, summarize them and point to the exact collection command and
environment.

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

In `rocke`, many performance decisions are encoded in the spec and the helper choices. A vague contract bakes in accidental assumptions.

Concrete contract examples:

- `UniversalGemmSpec` — GEMM tile, trait, data, layout, scheduler, epilogue.
- `ConvProblem` — NHWC/KYXC/NHWK convolution geometry; derives `Ho`, `Wo`, `M_gemm`, `flops`.
- `UnifiedAttentionProblem` — paged-attention shape in
  [`<library_root>/kernels/common/attention_unified.py`](../../../library/kernels/common/attention_unified.py);
  selectors choose 2D vs 3D.
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

For GEMM and convolution, validation also covers:

- the requested gfx target resolves to an `ArchTarget`;
- that target's `MmaCatalog` contains an `MmaOp` matching the required
  MFMA/WMMA family, operand and accumulator dtypes, and warp-tile `(m, n, k)`
  shape;
- `tile_m, tile_n` divisible by `warp_* * warp_tile_*`;
- `tile_k` divisible by `warp_tile_k`;
- block size <= hardware/lowering limit;
- `block_size = warp_m * warp_n * warp_k * wave_size` consistent;
- LDS bytes under the per-block budget;
- vector load widths divide tile shape;
- requested epilogue / pipeline / scheduler names recognized.

`known_arches()` lists targets with architecture metadata; actual kernel
support is determined by each builder's validation checks. If `arch` is
omitted, the builder applies its own default.

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

Resolve the wave size during spec validation and use that compile-time value in
the coordinate decomposition:

```python
wave_size = target.wave_size
tid     = b.thread_id_x()
lane    = b.lane_id()
warp    = b.div(tid, b.const_i32(wave_size))
block_x = b.block_id_x()
block_y = b.block_id_y()
block_z = b.block_id_z()
```

[`helpers/geometry.py`](../../python/rocke/helpers/geometry.py) (`WarpGrid`)
packages this for matrix kernels. Its `from_atom` constructor can obtain the
required wave size from the selected operation. `WarpGrid` also exposes the
historically named
`mfmas_per_warp_m / n`, `k_atoms_per_tile_k`, and per-workgroup
`block_m_off / block_n_off` values for both supported matrix paths.

Grid conventions in shipped instances:

```text
GEMM:                 (ceil(N/tile_n), ceil(M/tile_m), batch?)
implicit-GEMM conv:   (ceil(K_out/tile_n), ceil(M/tile_m), 1)
direct 16c conv:      (ceil(W/block_q), groups/block_groups, N)
direct 4c conv:       groups packed across wave lanes/batches
reduce / norm:        one workgroup per row
elementwise:          1D grid over contiguous elements
attention 3D tiled:   (q_blocks, kv_heads, split_kv_segments)
```

Chiplet swizzle (`chiplet_swizzle=True`, `helpers/grid.py::super_tile_swizzle`) is available for selected GEMM / conv paths. It is a launch-grid remap (improves L2 reuse on multi-XCD GPUs); the math is unchanged.

## 5. Describe Memory Instead Of Hand-Expanded Offsets

Prefer descriptors and views over raw arithmetic:

```text
plain contiguous tensors    -> TensorView, make_global_view, make_naive_tensor_view_packed
buffer-resource guarded     -> make_buffer_resource, make_buffer_view
non-bijective addressing    -> rocke.helpers.transforms.TensorDescriptor (transform DAG)
tile-local movement         -> TileWindow
distributed register tiles  -> TileDistributionEncoding + StaticDistributedTensor
```

The transform DAG is essential for:

- convolution `(m, k) -> NHWC` and `(k_out, k_gemm) -> KYXC`;
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

For GEMM-like tiles, `CoalescedTileLoader` is the broadly applicable staged
load pattern:

`CoalescedTileLoader` (sync, classic):

```text
global / buffer load -> VGPR vector -> LDS store -> b.sync() -> LDS reads
```

The current gfx942/gfx950 MFMA universal GEMM and convolution `compv4` path can
instead use `AsyncTileLoader`:

```text
raw_ptr_buffer_load_lds (DRAM -> LDS directly) -> b.s_waitcnt(vmcnt=0) -> LDS reads
```

Async constraints:

- `dwords in {1, 3, 4}`;
- LDS writes are lane-contiguous;
- destination base must be uniform within a wave;
- consumers must wait on VMEM before reading;
- swizzles belong in consumer read arithmetic, not in the destination pointer.

The gfx1250 universal GEMM path currently uses synchronous WMMA staging. It
accepts the `mem` or `wmma_v1` pipeline with the default epilogue and does not
support `compv4`, `direct_to_lds`, or `dtl_prefetch`.

gfx1250 has a separate `global_load_async_to_lds_*` instruction family with a
dedicated async counter. That path is currently used only for optional V
prefetching in gfx1250 tiled 3D attention; it is not wired into universal GEMM
or convolution. The gfx942/gfx950 `AsyncTileLoader` uses different instructions
and cannot be reused on gfx1250.

For row-wise small ops, use `helpers/sweep.py::sweep_row_chunks` and the `helpers/io.py` dispatchers (`load_vec_as_f32`, `pack_f32_to`) instead of building tile loaders.

## 7. Emit Compute

Matrix kernels commonly follow the structure below, using the layouts of the
selected `MmaOp`:

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

The tiled-attention kernels share this structure while selecting their matrix
operation and layouts from the exact target catalog.

```text
stage Q to LDS
iterate K/V pages or segments through paged-KV descriptor
compute QK with the selected matrix operation
apply masks (causal, sliding, ALiBi, QQ-bias, softcap)
online softmax update (warp_xor_reduce_max/sum)
compute PV with the selected matrix operation and the path's V operand layout
write final output (or segment workspace for 3D)
```

The implementations differ where the gfx targets differ. gfx950 variants can
use the target's transpose LDS reads, including `ds_read_tr16_b64`, and can
select wider target-catalog atoms where the owning validator permits them.
gfx942 uses the narrow `16x16x16` atoms and ordinary strided LDS reads that
reproduce the required V operand layout; it must not inherit gfx950's transpose
read recipe.

The gfx1250 tiled-attention kernels select WMMA from its catalog. Their `MmaOp`
layouts, compile-time geometry, data movement, and epilogue are the source of
truth; do not transplant the gfx950 LDS recipe.

## 8. Emit Epilogue

For the current gfx942/gfx950 MFMA universal GEMM and convolution paths, use
`DirectEpilogue` and `CShuffleEpilogue` from
[`helpers/epilogues.py`](../../python/rocke/helpers/epilogues.py). The epilogue
must agree with `MfmaAtom.lane_to_output`. WMMA universal paths instead use the
selected target-specific `MmaOp` accumulator layout and the default direct
epilogue admitted by the owning validator.

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

For examples and benchmarkable flows, emit a manifest. This gfx950 example
resolves its operation using the shape and dtypes used for `art`:

```python
target = ArchTarget.from_gfx("gfx950")
selected_op = target.mma.op_for_shape(
    family="mma", a_dtype="f16", b_dtype="f16", c_dtype="fp32",
    m=spec.tile.warp_tile_m, n=spec.tile.warp_tile_n,
    k=spec.tile.warp_tile_k,
)
assert selected_op is not None
manifest = make_gemm_manifest(artifact=art, block_m=..., block_n=..., block_k=...,
                              threads_per_block=spec.block_size,
                              default_shape=(3328, 4096, 4096),
                              atoms=[selected_op.op_id])
paths = write_artifact(art, Path("build/rocke_example"), manifest)
```

`python -m rocke.run_manifest` is the portable execution path for the resulting `(hsaco, manifest.json)`.
For every target-specific artifact, record the exact selected operation or atom
identity; do not derive the manifest entry from wave size or a family label.

## 10. Authoring Checklist

Before considering a new builder done:

- contract documented in a spec dataclass with a stable `kernel_name()`;
- `is_valid_spec(spec)` rejects unsupported layouts / dtypes / tile shapes / resources;
- every runtime predicate is expressed with `scf_if` or `select`, never a Python `if value:`;
- padding / tail behavior has an OOB-safe load/store path (buffer-rsrc + sentinel);
- LDS layout and async constraints are explicit (`LdsLayout`);
- the selected `MmaOp` exists in the exact gfx target's catalog and its wave and
  layout contract agrees with the kernel geometry;
- for an MFMA path, the MFMA atom and epilogue lane mapping agree;
- for a WMMA path, the selected `MmaOp` layout maps and supported epilogue agree;
- manifest signature and grid helper match the kernel ABI;
- correctness is checked against a reference (`run_manifest --verify`, or a torch / numpy oracle in a parity harness);
- benchmark reports median + spread, not a single lucky run (`benchmark_manifest(..., attempts=5, discard_first=True)`);
- generated LLVM / ISA / resource summaries are inspected for the intended primitive (`analyze_llvm_ir`, `analyze_hsaco`);
- the new path is added to an owning focused test and, when appropriate, the
  curated
  [`<platform_root>/python/rocke/examples/run_all.py`](../../python/rocke/examples/run_all.py)
  registry.
