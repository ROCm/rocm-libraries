# GEMM Instances

This page covers:

Core GEMM family:

- `instances/gemm_universal.py`
- `instances/batched_gemm.py`
- `instances/grouped_gemm.py`

 GEMM extensions:

- `instances/gemm_multi_d.py` -- CK Tile `19_gemm_multi_d`
- `instances/gemm_multi_abd.py` -- CK Tile `22_gemm_multi_abd`

 GEMM variants (StreamK + flat / contraction):

- `instances/streamk_gemm.py` -- CK Tile `40_streamk_gemm`
- `instances/flatmm.py` -- CK Tile `18_flatmm`
- `instances/batched_contraction.py` -- CK Tile `41_batched_contraction`

The main implementation is universal GEMM. Batched GEMM is a thin
wrapper. Grouped GEMM currently dispatches one universal GEMM kernel
per group. The / extensions are thin wrappers that
either (a) attach a `FusedEpilogue` for the multi-D variants, or
(b) wrap the existing batched body for the flatmm / contraction
variants. The StreamK GEMM is the only new kernel body --
v1 is a scalar inner-product + atomic-accumulate demonstration, with
the MFMA + cshuffle upgrade landing in a follow-on commit against
the stable `helpers.streamk` partitioner surface.

## Contract

Universal GEMM, `RCR` layout:

```text
C[M, N] = A[M, K] @ B[N, K]^T
A: row-major [M, K] (R)
B: row-major [N, K] (C means "logically column-oriented for B^T"; storage is [N, K] row-major)
C: row-major [M, N] (R)
```

Kernel ABI (`gemm_args_signature()`):

```text
A: ptr<f16, global> 8 bytes
B: ptr<f16, global> 8 bytes
C: ptr<f16, global> 8 bytes
M: i32 4 bytes
N: i32 4 bytes
K: i32 4 bytes
```

If `UniversalGemmSpec.batched=True`, the kernel reads `block_id_z` as batch and the ABI is extended:

```text
A, B, C, M, N, K, stride_a, stride_b, stride_c
```

`stride_*` are element strides (multiplied by `block_id_z` to derive the per-batch pointer).

## Spec Dataclasses

From `instances/gemm_universal.py`:

```python
@dataclass(frozen=True)
class TileSpec:
 tile_m: int
 tile_n: int
 tile_k: int
 warp_m: int
 warp_n: int
 warp_k: int = 1
 warp_tile_m: int = 32
 warp_tile_n: int = 32
 warp_tile_k: int = 16

 # Derived (raises if division is not exact):
 mfmas_per_warp_m -> tile_m // (warp_m * warp_tile_m)
 mfmas_per_warp_n -> tile_n // (warp_n * warp_tile_n)
 k_atoms_per_tile_k -> tile_k // warp_tile_k

Pipeline = Literal["mem", "compv3", "compv4"]
Scheduler = Literal["intrawave", "interwave"]
Epilogue = Literal["default", "cshuffle"]

@dataclass(frozen=True)
class TraitSpec:
 pipeline: Pipeline = "compv4"
 scheduler: Scheduler = "intrawave"
 epilogue: Epilogue = "cshuffle"
 pad_m: bool = False
 pad_n: bool = False
 pad_k: bool = False
 persistent: bool = False
 chiplet_swizzle: bool = False
 chiplet_wgm: int = 8
 chiplet_num_xcds: int = 8
 chiplet_chunk_size: int = 64
 waves_per_eu: Optional[int] = None

@dataclass(frozen=True)
class DataSpec:
 dtype_a: str = "fp16"
 dtype_b: str = "fp16"
 dtype_c: str = "fp16"
 dtype_acc: str = "fp32"
 layout: str = "RCR"

@dataclass(frozen=True)
class UniversalGemmSpec:
 name: str
 tile: TileSpec
 trait: TraitSpec = field(default_factory=TraitSpec)
 data: DataSpec = field(default_factory=DataSpec)
 wave_size: int = 64
 block_size: int = 0 # derived as warp_m * warp_n * warp_k * wave_size
 batched: bool = False
```

`TileSpec.warp_tile_*` are the MFMA atom dimensions; the chosen atom is `mfma_atom(data.dtype_a, warp_tile_m, warp_tile_n, warp_tile_k)`.

Shipped tile geometries cover `64..256 x 64..256 x 16..128` with warp grids `1x1, 2x1, 1x2, 2x2, 4x1, 1x4, 2x4, 4x2, 4x4`. MFMA atom set: 16x16x16, 16x16x32, 32x32x8, 32x32x16 f16. Pipelines: `mem`, `compv3`, `compv4`. Epilogues: `default`, `cshuffle`.

## Validation

`is_valid_spec(spec)` returns `(ok, reason)` and rejects unsupported combinations before IR build:

- target architecture support (current focus: gfx950);
- dtype combinations supported by IR lowering and the MFMA atom catalog;
- supported layout string (`RCR` is the production family);
- `tile_m`, `tile_n` divisible by `warp_* * warp_tile_*`;
- `tile_k` divisible by `warp_tile_k`;
- `block_size = warp_m * warp_n * warp_k * wave_size` consistent;
- LDS allocation under the per-block budget (~160 KiB practical cap);
- vector load widths divide tile shape;
- requested epilogue is implemented;
- scheduler and pipeline names are recognized.

Validation is part of the performance story: many "optimizations" are illegal without preserving tile / atom / LDS invariants.

## Grid

```text
grid_x = ceil_div(N, tile_n)
grid_y = ceil_div(M, tile_m)
grid_z = batch if batched else 1
```

Tile origins:

```text
block_n0 = block_id_x * tile_n
block_m0 = block_id_y * tile_m
```

If `chiplet_swizzle=True`, both block IDs are remapped through `helpers/grid.py::super_tile_swizzle` (or its compile-time variant), keeping `chunk_size` consecutive logical workgroups on the same XCD. The launch grid is unchanged; the remap happens at kernel entry from the flattened blockIdx. Constants:

```text
NUM_XCDS_MI300X = 8 # CDNA3
NUM_XCDS_MI325X = 8 # CDNA3
NUM_XCDS_MI350X = 8 # CDNA4 (physical XCDs may differ)
```

`chiplet_wgm` is the WGM super-tile group size (default 8); `chiplet_chunk_size` is the WGs-per-XCD granularity (default 64).

## Thread / Warp Geometry

`helpers/geometry.py::WarpGrid` computes per-CTA constants:

```text
tid : 0..block_size-1
lane : tid % 64
warp_id : tid / 64
warp_m_idx, warp_n_idx
block_m_off, block_n_off
mfmas_per_warp_m, mfmas_per_warp_n
k_atoms_per_tile_k
```

Each workgroup owns one `tile_m x tile_n` output tile. Each warp owns a subregion. Each lane owns `c_per_lane * mfmas_per_warp_m * mfmas_per_warp_n` accumulator slots (4 for 16x16 atoms, 16 for 32x32 atoms).

## LDS Allocation

```text
A_smem: <f16, [tile_m, tile_k + lds_k_pad]>
B_smem: <f16, [tile_n, tile_k + lds_k_pad]>
```

For `pipeline="compv4"` (double-buffer), two slabs are allocated and indexed with a ping-pong slot. `lds_k_pad` defaults to `+8` on sync paths (avoid ds_read bank conflicts) and `0` on async paths (lane-contiguous LDS requirement).

LDS allocations are part of the algorithm. They control:

- coalesced writes from global / buffer loads;
- bank-conflict behavior on MFMA feeder reads;
- vector width of stores / loads;
- compatibility with async load lane-contiguous writes.

## Data Movement

`pipeline in ("mem", "compv3")`: sync path via `CoalescedTileLoader` (`buffer/global load -> VGPR -> LDS store`).

`pipeline = "compv4"`: async path via `AsyncTileLoader` (`raw_ptr_buffer_load_lds`), double-buffered, with `s_waitcnt(vmcnt=0)` before each MFMA phase consumes the freshly-loaded slab.

GEMM descriptor (`make_global_view + make_tile_window`):

```text
A_view.tile(origin=(block_m0, k0)).load_vec(...)
B_view.tile(origin=(block_n0, k0)).load_vec(...)
```

Tail predicates guard `block_m0 + row < M`, `block_n0 + row < N`, `k0 + col < K`. The buffer-resource path routes invalid lanes to `INT32_MAX` so the DW3-bounds-checked load returns zero.

## Compute Loop

```text
acc = zero f32 accumulators (one vector per warp's MFMA fragment)

for k0 in scf_for(0, K, tile_k):
 load A tile (sync or async)
 load B tile (sync or async)
 s_waitcnt + sync as needed

 for kk in static_for(0, tile_k, atom.k):
 for warp_m fragment in static_for(...):
 A_frag = smem_load_vN_f16(A_smem, ...)
 for warp_n fragment in static_for(...):
 B_frag = smem_load_vN_f16(B_smem, ...)
 for output fragment in static_for(...):
 acc = atom.emit(b, A_frag, B_frag, acc)
 schedule_policy.emit_after_mfma_step(b, atom_idx)

 yield updated acc
```

`scf_for_iter` carries the per-lane accumulators across the K loop. `static_for` (Python-time unrolling) is used inside the K tile to keep the MFMA / DS_READ schedule visible to the scheduler.

`scheduler in ("intrawave", "interwave")` selects the `SchedulePolicy` flavor (intra-wave or inter-wave overlap groups).

## Epilogue

`epilogue = "default"` (`DirectEpilogue`):

```text
for each acc vector:
 for each slot i:
 (row_off, col_off) = atom.lane_to_output(b, lane, i)
 g_m = block_m0 + warp_m_off + row_off
 g_n = block_n0 + warp_n_off + col_off
 with scf_if(g_m < M and g_n < N):
 v = b.cast_f32_to(acc_slot, F16)
 C[g_m, g_n] = v
```

`epilogue = "cshuffle"` (`CShuffleEpilogue`):

```text
1. truncate each accumulator slot to f16 (vec_trunc_f32_to_f16 packs 4 floats -> 4 halves)
2. store into a tile_m x tile_n LDS scratch using atom.lane_to_output
3. b.sync()
4. cooperatively load contiguous chunks from LDS (one or two vectors per thread)
5. buffer_store_vN_f16 to C in coalesced order
```

Never swap atom shape without also revisiting `lane_to_output` and the epilogue vectorization width.

## Compiler Settings

Per-kernel attributes:

```text
kernel.attrs["max_workgroup_size"] = block_size
kernel.attrs["waves_per_eu"] = (lo, hi) # only when set on TraitSpec
```

Pointer params carry `noalias`, `readonly`, `writeonly`, `align`, `dereferenceable` metadata where applicable.

`compile_kernel(spec_built_kernel)` returns a `KernelArtifact` with HSACO + LLVM text + timings.

## Batched GEMM

`instances/batched_gemm.py::BatchedGemmSpec` is a thin wrapper:

```text
BatchedGemmSpec(name: str, tile: TileSpec, trait: TraitSpec = field(default_factory=TraitSpec),
 wave_size: int = 64, ...)
build_batched_gemm(spec) -> KernelDef # delegates to build_universal_gemm
batched_gemm_signature(spec) # A,B,C,M,N,K,stride_a,stride_b,stride_c
batched_gemm_grid(M, N, batch, spec) # (gridx, gridy, batch)
```

`name` is required and feeds the kernel symbol. The MFMA / LDS / epilogue body is identical to universal GEMM.

## Grouped GEMM

`instances/grouped_gemm.py`:

```text
GroupedGemmProblem(M, N, K, A_ptr, B_ptr, C_ptr)
GroupedGemmSpec(name: str, tile: TileSpec, trait: TraitSpec, wave_size: int = 64, ...)
GroupedGemmLauncher(spec, hsaco) # owns one universal GEMM launcher
grouped_gemm_problems(spec, problems) # validates shape list
```

`name` and `trait` are both required for `GroupedGemmSpec` (unlike universal/batched GEMM, which default `trait`).

Current shipped behavior:

```text
build_grouped_gemm(spec) -> universal GEMM kernel
GroupedGemmLauncher.__call__(problems) -> loop:
 for p in problems:
 launcher(per-group args, config=cfg(grid=p.grid))
```

This is multi-launch grouped GEMM: one HSACO, one persistent `KernelLauncher`, one launch per group. A planned single-launch persistent / group-indexed variant is documented but not shipped — primitives such as atomics (`global_atomic_add_f32`) exist, but the group-indexed scheduler is not implemented.

## Multi-D GEMM `instances/gemm_multi_d.py::GemmMultiDSpec` and `build_gemm_multi_d`
match CK Tile `19_gemm_multi_d`. The kernel computes
``E = f(A * B, D_0, D_1, ..., D_{n-1})`` where ``f`` is a fused
elementwise chain over the GEMM accumulator and N D tensors of the
same ``(M, N)`` shape. Implementation re-uses
`build_universal_gemm` + a `FusedEpilogue` whose op chain is one
`ResidualAdd` / `ResidualMul` per D operand.

Per-D op is chosen from ``{"add", "mul"}``; ``num_d`` up to 8.
Requires `epilogue="cshuffle"` on the base spec (the default epilogue
doesn't have the fused-op hook).

## Multi-A/B/D GEMM `instances/gemm_multi_abd.py::GemmMultiAbdSpec` matches CK Tile
`22_gemm_multi_abd`. In v1 only ``num_a == num_b == 1`` is supported
(the kernel delegates to `build_gemm_multi_d`); the load-time
`AElementWise` / `BElementWise` combine is a v2 follow-on. The spec
surface accepts ``a_operands`` / ``b_operands`` tuples today so
callers can write the intended kernel shape and pick up the v2 path
automatically when it lands.

## FlatMM `instances/flatmm.py::FlatMMSpec` matches CK Tile `18_flatmm`. CK Tile's
FlatMM is a batched matmul with preshuffled B; v1 re-uses the
`build_batched_gemm` body verbatim with a `ck_dsl_flatmm` kernel-name
prefix so a sweep / dispatcher can distinguish the two
configurations. The ``preshuffle_b`` field is on the spec surface
today and rejected at build time; it wires through with the
preshuffle-B helper (which also feeds the FP8 block-scaled GEMM).

## Batched Contraction `instances/batched_contraction.py::BatchedContractionSpec` matches CK
Tile `41_batched_contraction`. Generalises `batched_gemm` to *arbitrary
leading-batch ranks*: ``(B_0, ..., B_{r-1}, M, K) x (B_0, ..., B_{r-1},
K, N) -> (B_0, ..., B_{r-1}, M, N)``. v1 flattens the leading
batches into ``batch = product(batch_shape)`` on the host side and
delegates to `build_batched_gemm`; the launcher uses the standard
``(stride_a, stride_b, stride_c)`` arg trio.

## Block-scaled GEMM `instances/block_scale_gemm.py::BlockScaleGemmSpec` matches CK Tile
``38_block_scale_gemm``. The unified spec collapses the upstream
example's 29 preconfigured variants into one
``(quant_mode, mantissa_dtype, preshuffle_b, group_size_mnk)`` knob
set. v1 ships scalar-inner kernels for
``{aquant, bquant, abquant} x {fp8e4m3, bf8e5m2}`` (6 configurations,
all f32 output); the i4 mantissa variants and the
``preshuffle_b=True`` path require the v2 MFMA-based body
(against the same spec surface). The shipped FP8 / BF8 MFMA atoms
(``mfma_f32_16x16x32_fp8`` / ``_bf8`` and the 32x32x16 siblings) and
helpers (``i4_dequant``, ``mx_scale``, ``preshuffle``) are the
pieces the v2 body composes.

## MX GEMM `instances/mx_gemm.py::MxGemmSpec` matches CK Tile ``42_mx_gemm``. Uses
the OCP MX-spec shared-exponent format: each 32-element mantissa
block carries one 8-bit unbiased E8M0 scale. v1 supports fp8 and bf8
mantissa with ``group_k=32``; fp4 / fp6 mantissa land with the
matching unpack helpers as v2.

The decode chain
(:func:`ck_dsl.helpers.decode_mx_scale_e8m0` +
:func:`ck_dsl.helpers.apply_mx_scale`) handles the NaN / zero
sentinels (E8M0 ``e == 0`` and ``e == 255``) by returning ``0.0``,
matching the AMDGPU MX MFMA hardware path's denormal flush.

## FP8 / BF8 MFMA atoms `helpers/atoms.py` adds four FP8 / BF8 MFMA atoms paired with the
existing fp16 catalog:

| Atom | Per-lane A / B | Per-lane C | gfx target |
|---|---|---|---|
| ``fp8_16x16x32`` | ``<8 x fp8e4m3>`` | ``<4 x float>`` | gfx940+ |
| ``bf8_16x16x32`` | ``<8 x bf8e5m2>`` | ``<4 x float>`` | gfx940+ |
| ``fp8_32x32x16`` | ``<8 x fp8e4m3>`` | ``<16 x float>`` | gfx940+ |
| ``bf8_32x32x16`` | ``<8 x bf8e5m2>`` | ``<16 x float>`` | gfx940+ |

Lookup: ``mfma_atom("fp8", 16, 16, 32)`` (also accepts the alias
``mfma_atom("fp8e4m3", 16, 16, 32)``). The IRBuilder accepts the
operand vectors as ``<8 x fp8e4m3>`` / ``<8 x bf8e5m2>``; the lowering
bitcasts to ``<2 x i32>`` at the LLVM intrinsic boundary
(``llvm.amdgcn.mfma.f32.16x16x32.fp8.fp8`` and family). The mixed-
precision variants (``fp8.bf8`` / ``bf8.fp8``) are reachable by
bitcasting the operand vectors manually before calling the same
intrinsic family.

## StreamK GEMM `instances/streamk_gemm.py::StreamKGemmSpec` matches CK Tile
`40_streamk_gemm`. The kernel uses
`helpers.persistent.persistent_tile_for_each` to pull macro tiles
``(m_tile, n_tile, k_iter)`` from a global counter via
`global_atomic_add`, then `helpers.streamk.emit_streamk_decode` to
decode the linear macro-tile id. The partial K-iter contributions
to a shared ``(m_tile, n_tile)`` output land via the chosen
`StreamKReductionStrategy`:

* `Atomic` -- every CTA `global_atomic_add(Cf32, ..., partial)` into
 a shared f32 workspace. Simple, requires a finalisation pass to
 convert f32 -> output dtype. v1 implements this end-to-end.
* `Reduction` -- cooperative + flag-table reduction; the last
 contributor performs the conversion in-kernel. Spec accepts it
 today; builder rejects it until the helper lands.

v1's *inner* GEMM is a scalar per-thread inner-product (no MFMA) --
intentionally simple so the StreamK *infrastructure* (partitioner +
persistent + atomic) ships end-to-end in a reviewable kernel
(~150 LOC). The MFMA + cshuffle upgrade follows; the partitioner +
atomic surface stay stable. Workspace required:
``4 * M * N + 4`` bytes (Cf32 + Counter).

## Per-Spec Naming

`UniversalGemmSpec.kernel_name()` produces a deterministic name:

```text
<spec.name>_t<tm>x<tn>x<tk>_w<wm>x<wn>x<wk>_wt<wtm>x<wtn>x<wtk>_<pipeline>_<scheduler>_<epilogue>[_pad][_pers][_bat]
```

This name is used as the HSACO function symbol and the manifest's `kernel_name`.

## GEMM Performance Levers

Ordered roughly by impact on the bake-off path:

- atom shape (`32x32x16` over `32x32x8` halves the K-loop trip count);
- tile geometry (`tile_m, tile_n, tile_k`);
- warp grid (`warp_m x warp_n`);
- LDS K-pad and read pattern;
- epilogue (`cshuffle` vs `default`);
- async vs sync load (`compv4` vs `compv3` vs `mem`);
- scheduler policy;
- chiplet swizzle (L2 reuse on multi-die GPUs);
- waves_per_eu hint;
- launch mode (per-launch vs HIP graph amortized).

Controlled experiment rule:

```text
Change one lever, verify correctness, inspect IR/ISA/resources, then benchmark.
```

See `optimization/optimization_playbook.md` for the canonical lever loop and `RUNBOOK_COMPLIANCE.md` for the empirical conv pass that climbed 111 -> 280 TFLOPS by applying five levers in series.

## Failure Modes

- K-packed atom uses wrong A/B lane packing (numerically close but not bit-correct).
- Epilogue still assumes old atom lane layout after atom change.
- LDS allocation exceeds budget (`is_valid_spec` should reject; if it doesn't, the kernel fails at launch with `hipErrorInvalidValue`).
- `tile_k` not divisible by `atom.k` or loader vectorization assumptions.
- Direct epilogue scalarizes stores and hides compute gains in the next experiment.
- Async path consumes LDS before `s_waitcnt(vmcnt=0)` (intermittent corruption).
- Tail masks protect values but not invalid pointer loads (use buffer-rsrc, not plain pointer loads).
- Batched stride passed in elements vs bytes mismatch.
- Benchmark includes module-load overhead when comparing to warm kernels (use `time_launches` against a persistent `KernelLauncher`).
