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

Matrix-operation choices start from the exact gfx target, not from an
accelerator-family label or a requested wave width. Resolve
`ArchTarget.from_gfx(...)` and select an `MmaOp` from that target's
`MmaCatalog`. Separately, choose the execution wave size at compile time and
validate it against the selected operation's layout contract. Wave size
constrains lane geometry; it does not determine whether the gfx target provides
MFMA or WMMA, nor does it identify a legal atom.

The two independent fields currently recorded by the target catalog are:

| Exact gfx target | Matrix operations in `MmaCatalog` | Current `ArchTarget.wave_size` |
|---|---|---:|
| `gfx90a`, `gfx942`, `gfx950` | MFMA | 64 |
| `gfx1151`, `gfx1201`, `gfx11-generic` | WMMA | 32 |
| `gfx1250` | WMMA | 32 |

Wave width is selected when the kernel is compiled. The current rocKE catalog
admits the value shown above for each target; adding an alternative compiler
wave mode requires explicit target, backend, layout, and validator support. It
must not be inferred from the matrix-operation column, or vice versa.

Every platform-owned kernel instance in
`<platform_root>/python/rocke/instances/` follows this
shape. An instance is a complete kernel definition, either shared across
targets under `<platform_root>/python/rocke/instances/common/` or specialized
under `<platform_root>/python/rocke/instances/<arch>/`. Library-owned kernels,
including attention, live under `<library_root>/kernels/` and follow the same
operation-to-runtime progression without belonging to the platform instance
tree. Reusable authoring mechanics belong in
`<platform_root>/python/rocke/helpers/`; foundational IR, analysis, and lowering
mechanisms belong in `<platform_root>/python/rocke/core/`. For example, the
shared spec scaffolding lives in
`<platform_root>/python/rocke/helpers/spec.py` (`IOSpecRule`, `validate_io`,
`SignatureBuilder`, `kernel_name_join`, `ceil_div_grid`).

## Kernel Authoring And Optimization Outputs

New platform kernel authoring means producing a spec-driven kernel instance,
not just a one-off script. Put target-neutral platform instances under
`<platform_root>/python/rocke/instances/common/` and target-specific platform
instances under `<platform_root>/python/rocke/instances/<arch>/`. When multiple
instances need the same mechanism, promote that mechanism to platform
`helpers/` or `core/` according to the ownership boundary above instead of
treating `instances/` as a reusable-code layer. Any platform change that
affects emitted IR must include its matching C++ mirror and byte-identity
coverage in the same change.

Kernel optimization means a reproducible recipe plus a chosen implementation. If
an AI-assisted session finds a better tile, schedule, prefetch, split,
vectorization, or other lever, capture both the final code path and the evidence
that justified it.

Put platform experiments in the example folder for the instance being studied.
Keep benchmark scripts, shape files, qualitative mechanism notes, and
case-study methods close to the workload, for example under
`<platform_root>/python/rocke/examples/<arch>/<kernel_or_workload>/`. Keep
measured values, generated traces, and large logs outside the source tree in an
AMD-approved, access-controlled record with the revision, toolchain, gfx target,
shape set, and replay command.

Document every accepted optimization as a public qualitative case study. Describe
the workload shape class, candidate levers, commands, selected mechanism, rejected
mechanisms, and target constraints without publishing measured values or comparative
performance claims.

Promote reusable optimization knowledge into `dsl_docs/optimization/`. If the
work discovers a general tactic, decision rule, debugging skill, or reusable
performance lever, add or update the relevant optimization skill or runbook docs
so future kernels can reuse it.

Wire reusable kernels into their owning registry and test path. A reusable
platform instance should be importable from `rocke.instances` and covered by
the focused tests under `<platform_root>/tests/`. A library-owned kernel should
be exported by its owning library package (for example, attention kernels are
imported from `kernels`) and covered under `<library_root>/tests/`. Include a
builder in byte-identity coverage if it emits IR through both Python and C++
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

Concrete contract examples (platform instances unless noted otherwise):

- `UniversalGemmSpec` — GEMM tile, trait, data, layout, scheduler, epilogue.
- `ConvProblem` — NHWC/KYXC/NHWK convolution geometry; derives `Ho`, `Wo`, `M_gemm`, `flops`.
- `UnifiedAttentionProblem` — library-owned paged-attention shape in
  `<library_root>/kernels/common/attention_unified.py`; selectors choose 2D vs
  3D.
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

- architecture accepted by the owning validator (gfx950 is the default;
  `known_arches()` is the platform catalog, not universal family support);
- selected `MmaOp` exists in the exact gfx target's catalog for the dtype and
  tile shape;
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

The following coordinate decomposition is a wave64 compile-time configuration:

```python
tid     = b.thread_id_x()
lane    = b.lane_id()          # 0..63, wave64
warp    = b.div(tid, b.const_i32(64))   # if block_size > 64
block_x = b.block_id_x()
block_y = b.block_id_y()
block_z = b.block_id_z()
```

The literal `64` is a compile-time execution-mode assumption, not a per-kernel
runtime choice and not an MFMA selector. Current target-polymorphic builders
validate their configured wave size against `ArchTarget.wave_size` and the
selected `MmaOp.wave_size`, then use that value for lane and warp decomposition.
`helpers/geometry.py::WarpGrid` packages this for matrix kernels. Its
`from_atom` constructor can use the selected operation's required wave size as
a convenience default; that does not choose the operation or its MFMA/WMMA
family. `WarpGrid` also exposes the historically named
`mfmas_per_warp_m / n`, `k_atoms_per_tile_k`, and per-CTA
`block_m_off / block_n_off` values for both supported matrix paths.

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

Do not generalize this `compv4` loader recipe to every target. The current WMMA
universal path admits the `mem` and `wmma_v1` pipelines with the default direct
epilogue, not `compv4`. gfx1250 also has target-specific GFX12 async
global-to-LDS operations; those are distinct from the gfx942/gfx950
`AsyncTileLoader` contract and must be selected through the owning target-aware
builder. The pipeline choice follows gfx capabilities, not the independently
configured wave width.

For row-wise small ops, use `helpers/sweep.py::sweep_row_chunks` and the `helpers/io.py` dispatchers (`load_vec_as_f32`, `pack_f32_to`) instead of building tile loaders.

## 7. Emit Compute

MFMA matrix kernels commonly follow the structure below.

Execution-mode assumption for this example: the builder is compiled with
wave64 geometry. That assumption is independent of selecting MFMA from the gfx
target's catalog.

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

The gfx942 and gfx950 tiled-attention kernels select MFMA from their target
catalogs and share this structure:

Their current, separate execution-mode choice is wave64 at compile time.

```text
stage Q to LDS
iterate K/V pages or segments through paged-KV descriptor
compute QK with MFMA
apply masks (causal, sliding, ALiBi, QQ-bias, softcap)
online softmax update (warp_xor_reduce_max/sum)
compute PV with MFMA using the target-specific V-to-operand mapping
write final output (or segment workspace for 3D)
```

The implementations differ where the gfx targets differ. gfx950 variants can
use the target's transpose LDS reads, including `ds_read_tr16_b64`, and can
select wider target-catalog atoms where the owning validator permits them.
gfx942 uses the narrow `16x16x16` atoms and ordinary strided LDS reads that
reproduce the required V operand layout; it must not inherit gfx950's transpose
read recipe.

The gfx1250 tiled-attention kernels select WMMA from the gfx1250 catalog. Their
current, separate execution-mode choice is wave32 at compile time. They live
under `<library_root>/kernels/gfx1250/`. Their selected `MmaOp` layout maps,
compile-time wave geometry, target-specific data movement, and supported
epilogue path are the source of truth; do not transplant the example's wave64
lane arithmetic or the gfx950 LDS recipe into that path. The WMMA choice comes
from gfx1250 capabilities, not from choosing wave32.

## 8. Emit Epilogue

For the current gfx942/gfx950 MFMA universal GEMM and convolution paths, use
`DirectEpilogue` and `CShuffleEpilogue` from `helpers/epilogues.py`. The
epilogue must agree with `MfmaAtom.lane_to_output`. WMMA universal paths
instead use the selected target-specific `MmaOp` accumulator layout and the
default direct epilogue admitted by the owning validator. Wave-size validation
is a separate compile-time geometry check in both cases.

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

For examples and benchmarkable flows, emit a manifest. This is a default-target
gfx950 MFMA example whose builder is independently configured for wave64.
Resolve the operation from that exact target using the same shape and dtypes
used to build `art`:

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
- the compile-time wave size is supported by the target and agrees with the
  selected `MmaOp` layout contract;
- the selected `MmaOp` exists in that exact gfx target's catalog, independently
  of the wave-size choice;
- for an MFMA path, the MFMA atom and epilogue lane mapping agree;
- for a WMMA path, the selected `MmaOp` layout maps and supported epilogue agree;
- manifest signature and grid helper match the kernel ABI;
- correctness is checked against a reference (`run_manifest --verify`, or a torch / numpy oracle in a parity harness);
- benchmark reports median + spread, not a single lucky run (`benchmark_manifest(..., attempts=5, discard_first=True)`);
- generated LLVM / ISA / resource summaries are inspected for the intended primitive (`analyze_llvm_ir`, `analyze_hsaco`);
- the new path is added to an owning focused test and, when appropriate, the
  curated `<platform_root>/python/rocke/examples/run_all.py` registry.
