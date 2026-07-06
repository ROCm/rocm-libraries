# Multi-Architecture And Data-Layout Plan

This document plans how `rocke` should support multiple AMDGPU device
architectures and physical data layouts while keeping architecture-specific ISA
details separate from lowering logic.

The immediate goal is not to add every GPU target at once. The goal is to
reshape the interfaces so adding a new target means adding target metadata,
layout tables, and a backend variant, not editing every instance builder or the
generic lowering walk.

## Executive Goal

Support one kernel authoring path across multiple AMDGPU targets by separating
these responsibilities:

| Concern | Owns | Must not own |
| --- | --- | --- |
| Algorithm and tiling policy | Tile sizes, pipeline shape, fusion structure | Intrinsic spellings, waitcnt bitfields |
| Logical layout | Shapes, strides, coordinate transforms | LDS bank swizzles |
| Physical layout | LDS byte layout, swizzles, lane-to-fragment maps | LLVM syntax |
| Architecture | Capabilities, limits, supported MMA ops | Control-flow lowering |
| Lowering | SSA to LLVM structure, loops, PHIs, generic memory | Per-gfx intrinsic tables |

Today the lower part of the stack mixes these concerns. `core/lower_llvm.py`
contains target datalayout, AMDGPU intrinsic declarations, waitcnt encoding, and
gfx-specific operations. `helpers/atoms.py`, `helpers/layouts.py`, and selected
instances contain architecture-shaped metadata that should be selected from a
single target description.

## Code Organization (Hybrid Layout)

The codebase is split along **two axes**: the *lower* stack (`core/`, `helpers/`)
is **polymorphic** — architecture enters only as data (`ArchTarget`,
`RocmProfile`) and pluggable behaviour (`ISABackend`, layout backends), never as
a directory name. The *upper* stack (`instances/`, `examples/`) is organized so
that genuinely arch-divergent **algorithms** live in **gfx-id subfolders**, while
shared algorithms stay in one place and parametrize over the target.

This is the authoritative layout. Where earlier drafts of this document implied
instances are *always* a single arch-polymorphic builder per family, that holds
only for **shared** kernels; see "Reconciliation" below.

```text
core/                      polymorphic — no gfx-id subfolders
  arch/                    ArchTarget, MmaCatalog, RocmProfile (data + predicates)
  isa/                     ISABackend classes (gfx950, gfx9_mfma, ...)
  layout/                  OperandLayout, Lds{Producer,Consumer}Layout
  ir.py, lower_llvm.py     generic _Lowerer walk (target-neutral)
helpers/                   polymorphic building blocks — no gfx-id subfolders
  (atoms, layouts, loads, attention reductions: select from ArchTarget)
instances/                 HYBRID
  common/                  builders whose ALGORITHM is identical across arches
                           and only swap target-selected ops (MMA op_id, waitcnt,
                           datalayout, wave primitives) — e.g. elementwise, reduce,
                           plain GEMM bodies.
  gfx950/                  builders whose algorithm GENUINELY DIVERGES on gfx950
                           (e.g. transpose-LDS attention, MX fp4 GEMM).
  gfx942/                  gfx942-specific algorithm variants.
  gfx1151/                 RDNA3.5 wave32/WMMA-specific algorithm variants
                           (e.g. wmma_fmha_fwd adapter).
  common/gemm_policy.py    per-family capability filtering (GemmPipelinePolicy,
                           AttentionPipelinePolicy) — arch-agnostic code that
                           READS ArchTarget facts. Policy modules live under
                           common/ alongside the shared builders, not in a
                           per-family subfolder.
examples/                  per-arch, shared-where-possible
  common/                  arch-neutral demos + harnesses taking --arch/--isa
                           (e.g. universal_gemm_verify).
  gfx950/                  showcases exercising gfx950-only features.
  gfx942/                  gfx942-tuned / gfx942-specific showcases.
  gfx1151/                 RDNA wave32/WMMA showcases + verify harnesses.
```

### Where does a kernel live?

- **`instances/common/`** — if its IR construction is the same across arches
  *modulo* values selected from the target (`MmaOp` op-id, waitcnt encoding,
  datalayout, `arch.wave.*`). A `common/` builder must contain **no** `if arch
  == ...:` around structural control flow; only target lookups.
- **`instances/<gfx>/`** — the moment the *algorithm itself* differs (staging
  strategy, K-loop shape, memory path, which fused phases exist). If a single
  builder would need arch branches around structural code, split it into
  per-gfx modules instead. Adding/improving one arch's variant leaves the other
  arches' files untouched.
- **`instances/common/gemm_policy.py`** (and sibling `common/*_policy` modules)
  — pipeline/scheduler/warp-tile validity and LDS-budget rules for the family.
  Policy is shared, arch-agnostic code that composes `ArchTarget` predicates; it
  lives under `common/` next to the shared builders and is *not* duplicated per
  gfx.

Examples follow the same rule: arch-neutral harnesses in `examples/common/`
(imported by arch showcases to avoid duplication); only the arch-specific spec,
config, or feature demo lives under `examples/<gfx>/`.

### Module paths (no shims)

The hybrid layout shipped without per-module re-export shims. Relocated kernels
and examples now live directly under `instances/common/` (or `instances/<gfx>/`
for arch-divergent variants), and the **single** public surface is the
`rocke.instances` package `__init__`, which re-exports every spec and builder
(`build_universal_gemm`, `UniversalGemmSpec`, `GemmPipelinePolicy`, ...). Import
from the package, e.g.

```python
from rocke.instances import build_universal_gemm, UniversalGemmSpec, GemmPipelinePolicy
# or, for a specific home, the fully-qualified module:
from rocke.instances.common.gemm_universal import build_universal_gemm
from rocke.instances.common.gemm_policy import GemmPipelinePolicy
```

There are no `rocke.instances.<flat>` (e.g. `rocke.instances.gemm_universal`,
`rocke.instances.gemm`) modules and no `DeprecationWarning` shims at the old
flat paths — those import targets do **not** exist. In-tree callers, tests, and
harnesses (`rocke.examples.common.bake_off_implicit_gemm`,
`rocke.examples.common.universal_gemm_verify`) import via the package or the
fully-qualified `common/` path.

### Reconciliation with the polymorphic-instances sections

The "Pipeline Policy (Instance-Side)", "Review Rules", and "Success Criteria"
sections were drafted assuming a single arch-polymorphic builder per family that
is *never* edited to add an arch. Under the hybrid layout that goal holds for
**shared** kernels (`instances/common/` + `<family>/policy.py`): adding an arch
is data + backend + catalog/layout rows, and shared builders pick it up through
the predicates they already compose — no edit to a shared builder or a policy.
For kernels with **genuinely arch-divergent algorithms**, adding or improving an
arch *does* add a module under `instances/<gfx>/`. That is expected and allowed;
it does not perturb other arches and adds no pipeline vocabulary to `core/`. The
review rules below are annotated accordingly.

## Current Coupling

```text
instances/*.py
  |-- helpers/atoms.py       # MFMA shape and lane layout
  |-- helpers/layouts.py     # LDS swizzles
  |-- helpers/geometry.py    # wave/lane/block mapping
  |-- transforms.py          # logical coordinate algebra
  `-- core/ir.py             # KernelDef SSA
          |
          v
     core/lower_llvm.py      # generic lowering plus gfx-specific ISA details
          |
          v
     runtime/comgr.py        # isa string, default gfx950
```

The main pain points are:

- Architecture rules are scattered across instance builders, helpers, and the
  lowerer. There is no single ROCKE-owned description of "what this gfx target
  supports."
- `lower_llvm.py` knows too much about gfx-specific intrinsics, waitcnt
  encodings, datalayout, and LDS transpose reads. The `_DATALAYOUT` string is
  also implicitly keyed by ROCm version, not just gfx.
- `IRBuilder` exposes ISA-named methods directly
  (`mfma_f32_16x16x16_f16`, `mfma_f32_16x16x32_bf8`, eleven in total). Every
  instance builder, helper, and `MfmaAtom.emit` dispatches through them, so
  hiding intrinsics behind a backend requires an `IRBuilder` migration.
- MFMA atom metadata and the lowerer must be updated together when a new
  instruction shape is added.
- Wave64 assumptions are present in lane mappings, reductions, workgroup
  attributes, and attention kernels. `wave_size: int = 64` is the default in
  about fifteen instance dataclasses, in `WarpGrid`, in `helpers/loads.py`, and
  is structurally baked into the 4-stage XOR-butterfly reductions used by
  attention. Genericising wave size is a real migration, not a flag flip.
- Physical data layout is not a first-class contract between MMA selection,
  LDS staging, epilogues, and stores. The existing graph-fusion subsystem
  (`helpers/fuse.py`, `helpers/fusion_lowering.py::LoweringRegistry`) makes
  ad-hoc assumptions about MFMA-shaped accumulators.
- `transforms.py` is genuinely target-agnostic algebra and should not change.

## Target Layering

The target design has five layers and two target-selected plugins.

```text
Logical tensor algebra
  transforms.py, TensorDescriptor, TensorView
        |
        v
Tile policy and fused-kernel builders
  instances/*.py, helpers/*
        |
        v
Device SSA IR
  core/ir.py, target-neutral tile ops
        |
        v
Generic LLVM lowerer
  CFG, PHIs, types, generic memory
        |
        +-- ISABackend
        |     intrinsics, waitcnt, kernel attrs, datalayout
        |
        +-- LayoutBackend / PhysicalLayout
              accumulator maps, LDS swizzles, fragment ownership
```

Architecture and physical layout become explicit inputs to kernel building and
lowering. Instance builders may ask the target what is legal; they should not
emit ISA-specific names or encode device-specific constants themselves.

**Pipeline policy is a Layer 2 concern, not Layer 1.** Core (`core/arch/`)
publishes hardware predicates only — wave size, LDS bytes, MMA catalog,
async/transpose-LDS bits, vector-load alignment. Each kernel family owns a
policy module (`instances/common/<family>_policy.py`) that filters pipeline names,
schedulers, warp tiles, and LDS budget factors by *composing* core
predicates. See "Pipeline Policy (Instance-Side)".

## Architecture Target

Add a target description layer under `core/arch/`.

```python
@dataclass(frozen=True)
class ArchTarget:
    gfx: str
    family: str
    target_family: str
    isa_triple: str
    wave_size: int
    lds_capacity_bytes: int
    mma: MmaCatalog
    memory: MemoryCapabilities
    limits: ResourceLimits

    @staticmethod
    def from_gfx(gfx: str) -> "ArchTarget":
        ...

    # Hardware predicates that policies compose. These speak in bytes,
    # dtypes, and shapes — never in kernel-family vocabulary.
    def fits_lds(self, bytes_in_use: int) -> bool: ...
    def supports_dtype_combo(self, a: DType, b: DType, c: DType) -> bool: ...
    def max_vector_load_dwords(self, dtype: DType) -> int: ...
```

### Standalone ROCKE source of truth

ROCKE is a standalone framework. Architecture metadata must live inside the
ROCKE package and must not import from the CK Tile C++ dispatcher
(`projects/composablekernel/dispatcher/`) or any other ck_tile asset. Concretely:

- All target metadata, MMA catalogs, layout tables, and capability flags live
  under `Python/rocke/core/arch/`.
- ROCKE does not read `dispatcher/codegen/arch_specs.json` at runtime, build
  time, or test time.
- ROCKE does not import `dispatcher/codegen/arch_filter.py` or any other
  dispatcher Python module.
- `Python/rocke/` must be installable, importable, and testable without the
  dispatcher being present on disk.

Parity with CK Tile / dispatcher capability lists is a soft goal enforced by
**optional** comparison tests that are skipped when the dispatcher tree is
absent. They are diagnostic only — they may surface drift, but they may never
gate ROCKE CI or block a ROCKE release.

The ROCKE SSOT format is a ROCKE-owned data file (`core/arch/data/*.json` or
equivalent) plus the dataclasses that load it. The format is versioned and
documented in this package; downstream consumers depend on the dataclass API,
not on a JSON path elsewhere in the monorepo.

### ArchTarget contents

Architecture target data is pure-Python metadata. It must not carry SSA values
or refer to a specific `IRBuilder` instance. Builders read it during kernel
construction; backends read it during lowering.

`ArchTarget` is a **hardware-facts surface**. It must not carry kernel-family
vocabulary (no `"compv4"`, no `"qr_ks_vs"`, no per-family warp-tile tables).
That vocabulary is pipeline policy and lives in `instances/` — see
"Pipeline Policy (Instance-Side)".

`ArchTarget` carries:

- wave size;
- LDS capacity (bytes);
- per-arch memory capability bits (`has_async_lds`, `has_ds_read_tr`,
  `buffer_load_max_dwords`, etc.) — booleans and integers, not pipeline
  names;
- supported MMA or WMMA shapes by dtype combination, including mixed-dtype
  variants (`fp8_bf8_fp32`, `bf8_fp8_fp32`) and packed-fp4
  (`pk_fp4_pk_fp4_fp32` on gfx950);
- waitcnt encoding variant (4-bit vs 6-bit VMCNT);
- kernel attribute policy;
- architecture-specific resource limits (VGPRs, AGPRs, SGPRs, occupancy
  math, max-block-size) usable by any kernel family;
- vector-load / vector-store alignment limits;
- a small set of **capability predicates** that policies compose, e.g.
  `arch.fits_lds(bytes)`, `arch.supports_dtype_combo(a, b, c)`,
  `arch.max_threads_per_block`.

What `ArchTarget` does **not** carry:

- pipeline variant names or scheduler names;
- warp-tile shape tables keyed by kernel family or dtype;
- LDS budget multipliers per pipeline;
- forbidden `(pipeline, scheduler)` combos.

Those are kernel-authoring concerns. New arches are added by updating
hardware facts only; they do not need to enumerate every existing kernel
family's pipelines.

ROCm version is a separate axis. `_DATALAYOUT` in `lower_llvm.py` is keyed by
both gfx and ROCm release. The backend interface threads ROCm version
explicitly (see "ISA Backend").

## MMA Catalog

Move the supported atom catalog from a global helper table into an
architecture-selected catalog. The catalog is the ROCKE replacement for the
hardcoded `MfmaAtom` factory methods documented in
`dsl_docs/reference/mfma_atom_catalog.md`; that file's entries become the
initial gfx9 catalog rows.

The catalog supports both **enumeration** and **best-K selection**, mirroring
the K-search behaviour kernels need today:

```python
arch.mma.enumerate(
    family="mma",                       # "mma" | "wmma"
    a_dtype="fp8e4m3", b_dtype="bf8e5m2", c_dtype="f32",
    m=32, n=32,
) -> list[MmaOp]

arch.mma.select_largest_k(
    family="mma",
    a_dtype="fp8e4m3", b_dtype="bf8e5m2", c_dtype="f32",
    m=32, n=32,
    k_max=64,
) -> MmaOp | None
```

The selection keys must support mixed-dtype operands
(`fp8_bf8_fp32`, `bf8_fp8_fp32`) and packed inputs (`pk_fp4_pk_fp4_fp32`),
matching the shape of entries we describe in
`dsl_docs/reference/mfma_atom_catalog.md`.

`MmaOp` is a pure-Python descriptor:

- shape `(m, n, k)`;
- dtypes;
- per-lane operand widths and accumulator vector width;
- an opaque `op_id: MmaOpId` consumed by the backend;
- references to the operand and accumulator layouts (see "Physical Data Layout").

`MmaOp` does **not** expose LLVM intrinsic text. If a target does not support a
requested combination, `select_*` returns `None` and the instance builder must
fail with a structured error before any IR is emitted.

## Pipeline Policy (Instance-Side)

CK Tile C++ shows that pipeline *skeleton* is generic but pipeline *validity
and performance* are architecture-bound. That does **not** mean pipeline
metadata belongs in core: it means each kernel family must filter its own
pipeline space using the hardware facts that core exposes.

This section describes the separation of concerns between core arch
filtering and kernel-authoring policy.

### Why pipeline policy is not in core

Three forces push pipeline metadata out of `ArchTarget`:

1. **Open-world over families.** Pipeline names (`compv4`, `qr_ks_vs`,
   `qr_ks_vs_async`, `whole_k_prefetch_trload`, ...) only exist inside a
   kernel family. A new kernel family must not require editing core arch
   rows, and a new arch row must not require enumerating every existing
   family.
2. **Authoring locality.** The author of `instances/gemm/` knows whether
   `compv3 + interwave` is legal for their pipeline. That knowledge belongs
   next to the pipeline implementation, not in a global table that every
   reader of `core/arch/` has to scan.
3. **Stable core surface.** Pipelines come and go; new variants ship with
   new instances. `ArchTarget` should stay quiet across those releases.

### The split

```text
+--------------------------------------------------------------+
| core/arch/                       hardware facts and predicates|
|   ArchTarget                                                  |
|     wave_size, lds_capacity_bytes, mma, memory, limits        |
|     fits_lds(bytes), supports_dtype_combo(a, b, c), ...       |
+--------------------------------------------------------------+
                            ^  (read-only)
                            |
+--------------------------------------------------------------+
| instances/common/<family>_policy.py     family pipeline policy|
|   GemmPipelinePolicy                                           |
|     valid_pipelines(target) -> ("mem", "compv3", "compv4")     |
|     valid_schedulers(target, pipeline) -> ("intrawave", ...)   |
|     invalid_combos(target) -> {("compv3", "interwave")}        |
|     lds_budget_factor(pipeline) -> int                         |
|     validate(target, spec) -> ValidationResult                 |
|     enumerate(target, dtypes) -> Iterator[GemmConfig]          |
+--------------------------------------------------------------+
```

Core publishes facts; policies compose them. No core file mentions
`"compv4"`; no policy file mentions an LLVM intrinsic.

### Policy protocol

Each kernel family ships a policy module. Today's inline `is_valid_spec`
helpers become explicit policy classes:

```python
# instances/gemm/policy.py
class GemmPipelinePolicy:
    family: ClassVar[str] = "gemm"

    def valid_pipelines(self, target: ArchTarget) -> tuple[str, ...]: ...
    def valid_schedulers(
        self, target: ArchTarget, pipeline: str
    ) -> tuple[str, ...]: ...
    def invalid_combos(
        self, target: ArchTarget
    ) -> frozenset[tuple[str, str]]: ...

    def valid_warp_tiles(
        self, target: ArchTarget, dtypes: DtypeCombo
    ) -> tuple[WarpTile, ...]: ...
    def lds_budget_factor(self, pipeline: str) -> int: ...

    def validate(
        self, target: ArchTarget, spec: GemmSpec
    ) -> ValidationResult: ...
    def enumerate(
        self, target: ArchTarget, *, dtypes: DtypeCombo
    ) -> Iterator[GemmConfig]: ...
```

`validate` composes core predicates with family-specific rules:

```python
def validate(self, target, spec):
    if spec.pipeline not in self.valid_pipelines(target):
        return ValidationResult.reject("pipeline-not-on-arch")
    if (spec.pipeline, spec.scheduler) in self.invalid_combos(target):
        return ValidationResult.reject("forbidden-combo")
    if not target.supports_dtype_combo(spec.a, spec.b, spec.c):
        return ValidationResult.reject("dtype-not-on-arch")
    bytes_lds = ab_bytes(spec) * self.lds_budget_factor(spec.pipeline)
    if not target.fits_lds(bytes_lds + c_bytes(spec)):
        return ValidationResult.reject("lds-overflow")
    ...
    return ValidationResult.ok()
```

The autotuner queries the policy for legal configs, the target for what
fits:

```python
policy = GemmPipelinePolicy()
for cfg in policy.enumerate(target, dtypes=("f16", "f16", "f32")):
    if policy.validate(target, cfg).ok:
        yield cfg
```

### Migration

Today's inline tables move from instance globals into instance-side policy
modules — not into core:

| Today (instance global) | Becomes (instance-side policy) |
| --- | --- |
| `gemm_universal.is_valid_spec` body | `GemmPipelinePolicy.validate(target, spec)` |
| `gemm_universal._F16_WARP_TILE_SHAPES_GFX950` | `GemmPipelinePolicy.valid_warp_tiles(target, ("f16","f16","f32"))` |
| Inline `if pipeline == "compv4": bytes_lds = ab_single * 2` | `GemmPipelinePolicy.lds_budget_factor("compv4")` |
| Attention "is the async variant supported here?" branch | `AttentionPipelinePolicy.valid_pipelines(target)` composed with `target.memory.has_async_lds` |

Per-arch overrides live as small dispatch methods on the policy class
(`def lds_budget_factor(self, pipeline): return 2 if pipeline == "compv4" else 1`),
or as data tables module-local to the policy. They do **not** become rows
of any core registry.

### What policies must not do

- Import LLVM intrinsic names. Policies consume `MmaOp` descriptors from
  `target.mma`; they never know the intrinsic that backs an op.
- Reach into another family's policy. Cross-family fusion (e.g. an
  attention pipeline that wraps a gemm pipeline) imports both policies
  and composes them explicitly.
- Mutate `ArchTarget`. Targets are frozen dataclasses; policies are
  cheap to construct and may be parameterised at the call site.

### What stays in core

Anything a kernel author would write the same way regardless of family:

- MMA selection (`target.mma.select_largest_k`);
- LDS fit math (`target.fits_lds`);
- dtype combo support (`target.supports_dtype_combo`);
- vector-load alignment (`target.max_vector_load_dwords`);
- async-LDS, transpose-LDS capability bits.

These are the predicates every policy composes. Adding one of these helps
every kernel family; adding a `"compv4"` row helps exactly one family.

### Honest note on "target-neutral IR"

`KernelDef` is target-neutral in its **op names**, but its *sizes are not
free choices*: block size, vector load size, LDS allocation size, and
register usage all depend on `(target, policy decision, spec)` outcomes
made by the instance builder. The IR records the chosen sizes; the target
is consulted for facts and the policy is consulted for legality before
construction. CK Tile's `GemmPipelineProblemBase` already shows this
pattern with `kBlockSize = NumWarps * get_warp_size()`. The "target-neutral
IR" claim means the IR does not name a gfx or an intrinsic, not that its
sizes are independent of the target or policy.

## Physical Data Layout

Separate three layout concepts that are currently related but not cleanly
named:

| Layout kind | Meaning | Architecture dependence |
| --- | --- | --- |
| Logical layout | Tensor shape, strides, transform DAG, masks | None |
| Distribution layout | Which lane/thread owns which logical element | Wave size and tile distribution |
| Physical layout | LDS byte layout, swizzle, MMA operand and accumulator register maps | MMA family and target memory behaviour |

The previous single-protocol sketch was too narrow. Three role-specific
contracts are needed because each one has different producers, different
consumers, and different validation rules.

### Coordinate units

All layout protocols use these unit conventions:

- `LogicalCoord` is a tuple of element-typed coordinates inside the tile
  (`row`, `col`, etc.), unsigned. Values may be Python `int` (compile-time)
  or SSA `Value` (runtime); the protocol method accepts either uniformly.
- `OperandFragmentCoord` is `(elem_index_in_lane: int, m_or_n: Value)` for the
  per-lane operand register vector.
- `AccumulatorFragmentCoord` is `(row_in_tile: Value, col_in_tile: Value)`
  matching today's `MfmaAtom.lane_to_output` return.
- LDS offsets are returned in **bytes**, not elements. This matches existing
  XOR-swizzle code in `helpers/layouts.py`.

### `OperandLayout` (A / B / C)

Per-MMA-operand mapping from a logical fragment index to a per-lane element
index, and back. Today's `MfmaAtom.lane_to_output` covers only C; A and B
mappings live implicitly inside loaders (`f16_16x16x32` K-packing is an
example). This protocol makes them first-class:

```python
class OperandLayout(Protocol):
    role: Literal["a", "b", "c"]

    def lane_to_logical(
        self, lane: Value, slot: int
    ) -> LogicalCoord:
        """Where, in the (m, n)-or-(m, k)-or-(k, n) operand tile,
        does this lane's `slot`-th register element live?"""
```

Each `MmaOp` owns three `OperandLayout` instances (`a`, `b`, `c`).

### `LdsProducerLayout`

How a global-to-LDS copy decides which thread writes which LDS byte.

```python
class LdsProducerLayout(Protocol):
    def producer_byte_offset(
        self, thread_id: Value, elem: int
    ) -> Value: ...

    def is_lane_contiguous(self) -> bool: ...
```

For the async DMA path (`raw_ptr_buffer_load_lds`), `is_lane_contiguous()` must
return `True` — the producer cannot swizzle because the intrinsic always writes
lane-contiguous bytes.

### `LdsConsumerLayout`

How an MMA operand load reads from LDS, including the swizzle that the
producer could not apply.

```python
class LdsConsumerLayout(Protocol):
    def consumer_byte_offset(
        self, logical_coord: LogicalCoord
    ) -> Value: ...

    def expected_producer(self) -> "LdsProducerLayoutKind":
        """Which producer layouts this consumer is paired with."""
```

### Async invariant

The async DMA path keeps the asymmetric pairing today:

```text
async producer writes lane-contiguous LDS
consumer applies the required swizzle or transpose address math
```

`validate_for_async(producer, consumer)` must check
`producer.is_lane_contiguous() and consumer.expected_producer() == producer.kind`
and is the only allowed way to pair the two for async paths. Sync paths may use
swizzled producers as today.

### Migration

The existing `LdsLayout`, `TransposeLdsReader`, XOR swizzle tables, and
`MfmaAtom.lane_to_output` migrate as follows:

| Today | Becomes |
| --- | --- |
| `MfmaAtom.lane_to_output` | `MmaOp.operands["c"]: OperandLayout` |
| `LdsLayout.padded_k`, `LdsLayout.xor_*` | `LdsConsumerLayout` (sync path) |
| `TransposeLdsReader` | `LdsConsumerLayout` (transpose-LDS variant) |
| `helpers/loads.py` lane-contiguous producers | `LdsProducerLayout` |
| `helpers/layouts.py::XOR_SWIZZLE_TABLE` | table inside `core/layout/lds/` keyed by `(MmaOp, dtype)` |

Layout is selected by the target plus the chosen MMA op, then consumed by
loaders, MMA operand reads, cshuffle epilogues, and stores. Layout objects
return coordinates or byte offsets only; they never emit LLVM directly.

## Target-Neutral IR

Keep `KernelDef` as the boundary between authoring and lowering, but reduce
target leakage in operation names and attributes.

`KernelDef` carries the target:

```python
@dataclass
class KernelDef:
    ...
    target: ArchTarget | None = None
```

Target-neutral tile operations stored in the IR:

```text
tile.mma          {op_id: MmaOpId, dtypes, shape}
tile.copy_global_to_lds  {mode, cache_policy}
tile.memory_fence {scope, counters}
tile.block_barrier {scope}
```

`op_id` is an opaque selected operation from `ArchTarget.mma`, not an LLVM
intrinsic string.

### `legalize_for_target`: bind, do not rewrite

The pre-lowering pass `legalize_for_target` **binds** target-neutral ops to
backend op-ids. It does not rewrite op shapes, types, or loop carries.

Specifically:

- Allowed: attaching `op_id`, `module_intrinsics`, or backend-private
  attributes to existing ops.
- Allowed: rejecting unsupported ops with a structured `LegalizationError`.
- **Forbidden**: changing an op's result type or vector width.
- **Forbidden**: splitting `tile.mma {K=32}` into two `tile.mma {K=16}` calls.
- **Forbidden**: introducing or removing `scf.for` iter_args.

These restrictions exist because attention and pipelined GEMM kernels carry
MMA results through `scf.for` iter_args. Any shape rewrite there silently
breaks PHI types in `_Lowerer`. If a target cannot satisfy a requested op,
the instance builder must select a different op before constructing the
loop — selection happens in builder code, not in legalization.

The right mental model: legalization is a `bind` step that resolves opaque
handles. Op-shape choice is the instance builder's responsibility.

### `IRBuilder` migration

`IRBuilder` currently exposes eleven ISA-named methods
(`mfma_f32_16x16x16_f16` … `mfma_f32_4x4x4_f16`). These are the actual ISA
leak that the "instances may not import intrinsic names" rule depends on.
Migration is in three steps and is part of Phase 1, not deferred:

1. **Add the neutral method.** `IRBuilder.mma(op_id, a, b, c) -> Value` becomes
   the canonical API. It produces a `tile.mma` op with the bound `op_id`.
2. **Convert existing methods to thin wrappers.** Each `mfma_f32_*` method
   stays for one release cycle but delegates to `self.mma(op_id_for(...))`
   with a `DeprecationWarning`.
3. **Delete the wrappers.** After downstream callers migrate, remove the
   ISA-named methods entirely. `MfmaAtom.emit` is rewritten to call
   `b.mma(self.op_id, a, b, c)` and stops importing intrinsic names.

`KernelDef.kernel_name()` outputs must stay byte-identical for gfx950 across
steps 1 and 2 so the autotuner's on-disk HSACO cache is not invalidated.

## ISA Backend

Split architecture-specific code out of `core/lower_llvm.py` behind an
`ISABackend` protocol.

```python
class ISABackend(Protocol):
    arch: ArchTarget
    profile: RocmProfile               # see "ROCm Version Polymorphism"

    def module_preamble(self) -> str:
        """Datalayout + target triple lines.
        Reads `profile.datalayouts[arch.gfx]` and `profile.triple`."""

    def kernel_attributes(self, max_workgroup_size: int) -> dict[str, str]:
        ...

    def declare_intrinsic(self, op_id: MmaOpId) -> str:
        """Returns `profile.intrinsic_decls[op_id]`.
        Raises if the op is not present in this ROCm profile."""

    def emit_mma(self, lowerer, op, a, b, c) -> Value:
        ...

    def encode_waitcnt(self, vmcnt: int, expcnt: int, lgkmcnt: int) -> int:
        ...

    def emit_async_copy(
        self, lowerer, src, dst, bytes_per_lane: int
    ) -> Value | None:
        """None if unsupported on this arch (e.g. gfx908) or this profile."""

    def emit_lds_transpose_read(self, lowerer, *args) -> Value | None:
        """None unless arch exposes ds_read_*_tr_* (gfx950)
        and the profile declares its intrinsic."""
```

`RocmProfile` is a data-only object covering version-specific codegen knobs;
the backend is the gfx-classed code that consumes both. See the next section
for the data shape, the registry, and the fallback rules.

`_Lowerer` remains responsible for generic lowering:

- LLVM type spelling;
- function and parameter emission;
- CFG construction;
- PHIs and `scf.for`;
- generic global, LDS, and scalar operations;
- target-neutral operation dispatch.

The backend owns:

- datalayout and target triple details (keyed by `(gfx, rocm_version)`);
- kernel attributes that depend on wave size or target limits;
- intrinsic declarations;
- MFMA or WMMA call emission;
- waitcnt encoding;
- target-specific async-copy and LDS transpose operations.

### Concrete backend classes

The earlier draft said "family-level backends with variants." That is true at
the data layout level but misleading at the capability level: the gfx9 family
fractures along multiple axes. Below is the actual capability matrix that
drives backend selection.

| gfx | async LDS | fp8/bf8 MFMA | bf16 atom | VMCNT bits | LDS KB | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| gfx908 | no | no | classic | 4 | 64 | MI100 |
| gfx90a | no | no | `.1k` | 4 | 64 | MI200 |
| gfx942 | yes | yes | `.1k` | 4 | 64 | MI300; f16/bf16 atoms 16x16x16, 32x32x8 only |
| gfx950 | yes | yes (incl. fp4) | `.1k` | 6 | 160 | MI350; `ds_read_*_tr_*`; adds k=32 f16/bf16 atoms 16x16x32, 32x32x16 |
| gfx1100 | no | no (WMMA) | WMMA | n/a | 64 | RDNA3, wave32 |
| gfx1151 | no | no (WMMA) | WMMA | n/a | 64 | RDNA3.5 (Strix Halo), wave32; shipped RDNA target |
| gfx1201 | no | no (WMMA) | WMMA | n/a | 64 | RDNA4, wave32 |

> MMA-catalog note (not IR drift): the k=32 f16/bf16 MFMA atoms
> (`16x16x32`, `32x32x16`) exist **only** on gfx950. gfx942's f16/bf16
> catalog is `{16x16x16, 32x32x8}`, and gfx942 also has the smaller 64 KB
> LDS budget. A k=32 (or oversized-LDS) GEMM/FMHA spec is therefore
> *rejected* with a structured `ValueError` on gfx942 by
> `GemmPipelinePolicy` / `is_valid_spec` before any IR is emitted. This is
> intended arch filtering, **not** a regression or byte-identity drift — the
> CDNA MFMA body is byte-identical between gfx942 and gfx950 for every config
> that is legal on both.

This collapses into the following concrete backend classes:

```text
Gfx908Backend         (no async, no fp8, classic bf16)
Gfx9MfmaBackend       (gfx90a, gfx942: 4-bit VMCNT, .1k bf16)
                      flag: async_lds = (gfx != gfx90a)
                      flag: fp8       = (gfx in {gfx942})
Gfx950Backend         (6-bit VMCNT, fp8/bf8/fp4, ds_read transpose, 160 KB LDS)
Gfx11RdnaBackend      (RDNA3, wave32, WMMA)
Gfx12RdnaBackend      (RDNA4, wave32, WMMA)
```

Four CDNA backends and two RDNA backends. `Gfx9MfmaBackend` is the only one
that uses runtime flags between gfx variants; the others are split because
their codegen actually diverges (instruction selection, waitcnt encoding,
LDS read intrinsics).

`BACKEND_REGISTRY: dict[str, Callable[[ArchTarget, RocmProfile], ISABackend]]`
maps `gfx` to its constructor. Adding a new gfx is "add a row to the table
plus, usually, a new backend class" — not "edit `_Lowerer`."

## ROCm Version Polymorphism

ROCm version is a real polymorphism axis, not a parameter the backend can
ignore. Different ROCm releases ship different LLVM versions, which means:

- `target datalayout` strings change (the gfx950 datalayout in
  `core/lower_llvm.py::_DATALAYOUT` is a snapshot of one box's `clang -emit-llvm`
  output);
- intrinsic names and signatures shift between LLVM versions
  (e.g. the `mfma.f32.16x16x16bf16.1k` legacy name vs the canonical
  `mfma.f32.16x16x16.bf16` form on newer LLVM);
- new MMA shapes appear in new releases and *do not exist* in older ones;
- buffer-resource DW3 defaults have shifted historically;
- comgr's accepted option flags evolve.

The design handles this with **two orthogonal axes composed at backend
construction**:

| Axis | Type | Lifetime | Contents |
| --- | --- | --- | --- |
| `ArchTarget` | data, per `gfx` | static | wave size, LDS bytes, MMA capability list, waitcnt encoding kind, supported features |
| `RocmProfile` | data, per `(major, minor)` | static | LLVM datalayout per gfx, target triple, intrinsic name/decl tables, buffer-rsrc defaults, comgr option fragments |
| `ISABackend` | code, per gfx family | runtime instance | composes `arch` + `profile`; emits LLVM |

```text
   ArchTarget rows (data)        RocmProfile rows (data)
   per gfx                       per ROCm release
   +---------+                   +-------+
   | gfx908  |                   |  6.0  |
   +---------+                   +-------+
   | gfx942  |                   |  6.2  |
   +---------+                   +-------+
   | gfx950  |---+         +---->|  6.4  |
   +---------+   |         |     +-------+
   | gfx1100 |   |         |
   +---------+   |         |
                 |         |
                 v         v
        +--------------------------+
        | Backend class (code)     |
        |   Gfx908Backend          |
        |   Gfx9MfmaBackend        |  <-- e.g. Gfx950Backend
        |   Gfx950Backend          |
        |   Gfx11RdnaBackend       |
        |   Gfx12RdnaBackend       |
        +-------------+------------+
                      |
                      v
        +--------------------------+
        | ISABackend instance      |
        | = Class(arch, profile)   |
        +--------------------------+
```

Adding a ROCm release is a new `RocmProfile` row; adding a gfx is a new
backend class plus an `ArchTarget` row. The example resolution above is
`gfx950` × ROCm 6.4 → `Gfx950Backend(arch, profile)`; any
`(arch row, profile row, backend class)` triple is valid as long as the
backend class accepts that gfx.

### `RocmProfile`

```python
@dataclass(frozen=True)
class RocmProfile:
    version: tuple[int, int]                  # (6, 2)
    llvm_major: int                           # 18
    triple: str                               # "amdgcn-amd-amdhsa"
    datalayouts: dict[str, str]               # keyed by gfx
    intrinsic_decls: dict[MmaOpId, str]       # `declare ... @llvm.amdgcn...`
    intrinsic_calls: dict[MmaOpId, str]       # canonical call name
    buffer_rsrc_dw3: int                      # historically shifting default
    comgr_options: tuple[str, ...]            # version-safe option flags
    capability_overlay: RocmCapabilityOverlay # per-version yes/no flags

    @staticmethod
    def from_version(major: int, minor: int) -> "RocmProfile": ...
    @staticmethod
    def latest_known() -> "RocmProfile": ...

@dataclass(frozen=True)
class RocmCapabilityOverlay:
    has_async_lds_intrinsic: bool   # raw.ptr.buffer.load.lds presence
    has_ds_read_tr_b16: bool        # gfx950 transpose read at this LLVM
    mfma_immarg_count: int          # 3 or 4 across LLVM versions
    # ... add knobs only when a release actually flips them
```

The overlay is intersected with `ArchTarget` capabilities. An op is available
to instances iff **both** the arch supports it **and** the profile declares it.
This is what stops a Phase-3 RDNA backend from accidentally emitting a 6.4
MFMA intrinsic against a 6.0 ROCm install.

### Registry and lookup

```python
ROCM_PROFILES: dict[tuple[int, int], RocmProfile] = {
    (6, 0): _PROFILE_6_0,
    (6, 2): _PROFILE_6_2,
    (6, 4): _PROFILE_6_4,
    # ...
}

def resolve_rocm_profile(
    requested: tuple[int, int] | None,
    *,
    strict: bool = False,
) -> RocmProfile:
    """Exact match, then nearest-older fallback.

    - `None` triggers detection via `runtime.comgr.get_comgr_version()`
      falling back to `runtime.hip_module.get_hip_runtime_version()`.
    - Unknown future version warns and returns `latest_known()` unless
      `strict=True` (env: `ROCKE_STRICT_ROCM=1`), in which case it raises.
    - Versions older than the oldest profile always raise — there is no
      "older fallback" below the support floor.
    """
```

Adding ROCm 6.5 is one PR: one new dict entry, no code change in `_Lowerer`
or backends in the common case. Adding a 6.5-only intrinsic is one PR: the
6.5 profile gains a `intrinsic_decls` row; older profiles do not.

### Version detection

Two new accessors land under `runtime/`:

```python
runtime.comgr.get_comgr_version() -> tuple[int, int]
runtime.hip_module.get_hip_runtime_version() -> tuple[int, int]
```

`get_comgr_version` calls `amd_comgr_get_version` via ctypes; HIP version uses
`hipRuntimeGetVersion`. Resolution order:

1. explicit `compile_kernel(rocm_version=(M, m))` — always wins;
2. `ROCKE_ROCM_VERSION=M.m` env var — for CI pinning;
3. detected comgr version — runtime default;
4. detected HIP runtime version — fallback if comgr fails to load;
5. error.

### Cross-compilation

The backend builds LLVM IR for whatever `RocmProfile` it is given, regardless
of which `libamd_comgr` is dlopened. AMD's LLVM IR is forward-compatible at
the textual level: a newer comgr can ingest IR shaped for an older LLVM. The
HSACO build step uses whichever comgr is loaded, so the artifact records both:

```python
@dataclass(frozen=True)
class KernelArtifact:
    ...
    target_rocm: tuple[int, int]      # profile used for LLVM IR
    build_rocm: tuple[int, int]       # libamd_comgr version used for HSACO
```

If `target_rocm != build_rocm`, the manifest carries both. This is required
for reproducible builds on rolling base systems and for vendoring HSACO
artifacts to deploy-time hosts.

### Cache integration

The HSACO cache key (introduced in "Backward Compatibility") becomes:

```text
key = (llvm_text_sha256, arch, target_rocm_profile_id, build_comgr_version)
```

`target_rocm_profile_id` is the profile's `(version, hash-of-fields)` tuple so
that editing a profile in place invalidates its cache entries without
requiring callers to bump a manual version.

### What stays version-independent

- `KernelDef`, `core/ir.py` ops, `transforms.py` algebra, `helpers/*` authoring
  surfaces — none of these depend on ROCm version.
- `ArchTarget` — describes hardware, not the toolchain. It does not change
  between ROCm releases for the same gfx.
- `_Lowerer` generic walk — CFG, PHIs, scf.for; ROCm-version-specific changes
  only land in profiles and backends.

### Testing

| Test | What it asserts |
| --- | --- |
| Per-profile snapshot | `RocmProfile.from_version(M, m)` emits the expected `module_preamble()` and intrinsic decls (golden) |
| Fallback semantics | `(M, m+1)` resolves to `(M, m)` profile with a warning |
| Strict mode | `ROCKE_STRICT_ROCM=1` turns the warning into an error |
| Cross-compile | Same `KernelDef` lowered against two profiles differs only in allow-listed lines |
| Capability gating | `MmaCatalog.select_*` rejects ops absent from the profile, with a structured error citing the missing intrinsic |
| Cache key | Editing a profile invalidates its cache entries; touching an unrelated profile does not |

CI runs the matrix on the oldest-supported and latest-known profiles every
build, and the middle entries on a nightly schedule.

### Support window

ROCKE declares an explicit ROCm support window in the package metadata:

```text
rocke supports ROCm M.m through M'.m'.
Older releases are not tested. Newer releases use the latest-known profile.
```

The window moves forward in release notes, never silently. Dropping an old
profile is a major version bump for the ROCKE package.

## Compile Flow

The target compile flow is:

```text
spec + arch name + optional rocm_version
  -> ArchTarget.from_gfx(gfx)
  -> instance validation and target-aware helper selection
  -> KernelDef with target-neutral tile ops + target
  -> legalize_for_target(kernel)       # bind op_ids, no rewrite
  -> backend = backend_for(arch, rocm_version)
  -> lower_kernel_to_llvm(kernel, backend)
  -> build_hsaco_from_llvm_ir(..., isa=arch.isa_triple)
```

The public API:

```python
artifact = compile_kernel(
    kernel,
    arch="gfx950",                # default until Phase 3, see BC section
    rocm_version=None,            # auto-detect via comgr; explicit for reproducibility
)
```

Interactive helpers may default to detected hardware. Reproducible builds,
manifests, and sweeps should pass `arch` and `rocm_version` explicitly so
on-disk HSACO cache keys are deterministic.

## Epilogue Fusion

This plan does not require a separate epilogue fusion framework, but it does
require epilogue fusion to consume layout contracts rather than MFMA-specific
assumptions.

Existing epilogue fusion can stay above lowering:

```text
accumulator fragment
  -> optional fused scalar/vector transforms
  -> physical layout aware store or cshuffle
```

Required interface changes:

- replace direct dependence on `MfmaAtom.lane_to_output` with
  `accumulator_layout.lane_to_output`;
- let `ArchTarget` choose legal store vector widths, cshuffle layout, and LDS
  budget;
- keep fused ops such as bias, scale, clamp, activation, residual add, and cast
  as target-neutral SSA transforms;
- let the backend lower only the primitive operations used by those transforms.

Fusion belongs above lowering, but below pure algorithm selection. A fused
kernel may ask the target what primitives and layouts are legal; it must not
emit ISA-specific intrinsic names or waitcnt encodings.

## Multi-Layer Fusion

Attention-style fusion is more demanding than epilogue fusion because it mixes
multiple producer/consumer phases inside one kernel:

```text
QK
  -> mask / scale / softcap
  -> online softmax reductions
  -> P staging
  -> PV
  -> output epilogue
```

The architecture split should preserve this style. It should not force every
phase through separate materialized kernels. The change is that fused builders
must depend on target-aware primitives:

- MMA family selection, MFMA versus WMMA;
- wave reductions for wave64 and wave32;
- async copy support and wait policy;
- LDS layout for K, V, P, and accumulator staging;
- resource modeling for LDS bytes, registers, wave count, and occupancy;
- target-specific cache and buffer policies;
- **pipeline-variant selection** — `qr_ks_vs` vs `qr_ks_vs_async` vs
  `qr_ks_vs_whole_k_prefetch` vs `qr_ks_vs_async_trload` are kernel-family
  policy decisions (see CK Tile `block_fmha_pipeline_*.hpp`). The attention
  builder consults `AttentionPipelinePolicy.valid_pipelines(target)` —
  which itself composes `target.memory.has_async_lds`,
  `target.memory.has_ds_read_tr`, and MMA shape availability — to pick a
  variant before constructing the IR. Core does not learn the variant
  names.

For attention, the main migration risk is wave-size coupling. Current online
softmax and cross-lane reductions are written around wave64-style patterns.
Those should move behind helper APIs such as:

```python
arch.wave.reduce_max(...)
arch.wave.reduce_sum(...)
arch.wave.permute(...)
```

The attention builder can remain fused and hand-scheduled, but the
wave-level and memory-level primitives must be selected from `ArchTarget`.

## Phased Rollout

### Phase 0: Target Metadata (Core Only)

- Add `core/arch/` package with ROCKE-owned data files and `ArchTarget`
  dataclasses. No imports from `dispatcher/`.
- Seed the data file from observed gfx950 behaviour; add gfx942 rows from
  hardware-verified MFMA shape lists.
- Expose hardware predicates: `fits_lds`, `supports_dtype_combo`,
  `max_vector_load_dwords`, plus `memory.has_async_lds`,
  `memory.has_ds_read_tr`.
- Thread `arch` and `rocm_version` through `compile_kernel()` and the
  manifest schema.
- Default `arch="gfx950"` and `rocm_version=None` to preserve callers.
- No pipeline vocabulary lands in core. `gemm_universal.is_valid_spec`
  stays in place this phase.

### Phase 1: MMA Catalog, IRBuilder, And Instance-Side Policy Modules

- Add `MmaCatalog` selected from `ArchTarget` with `enumerate` and
  `select_largest_k`.
- Add `IRBuilder.mma(op_id, a, b, c)` as the canonical MMA emit method.
- Convert each `mfma_f32_*` method on `IRBuilder` into a thin
  `DeprecationWarning` wrapper around `self.mma(...)`.
- Migrate `MfmaAtom.emit` to call `b.mma(self.op_id, ...)`.
- Move `MfmaAtom.lane_to_output` to `OperandLayout` on the selected `MmaOp`.
- Introduce `instances/common/<family>_policy.py` modules. Start with
  `gemm_policy.py` (shipped) and an `attention` policy. Move today's
  `is_valid_spec` bodies and warp-tile globals into the family policy class;
  the existing entry point becomes a thin wrapper that calls
  `policy.validate(target, spec)`.
- Migrate the remaining instance-side validity branches
  (`conv_implicit_gemm`, `grouped_gemm`, `flatmm`, FMHA variants) to their
  family policy classes. gfx950 output stays byte-identical.
- Inline legacy validators remain as fallbacks guarded by
  `ROCKE_LEGACY_VALIDATION=1` until Phase 3.

### Phase 2: Layout Protocols And Wave-Size Audit

- Implement `OperandLayout`, `LdsProducerLayout`, `LdsConsumerLayout` and
  migrate `LdsLayout`, `TransposeLdsReader`, and the XOR swizzle table.
- Make `WarpGrid.wave_size` derive from `ArchTarget.wave_size` while keeping
  the dataclass default for source-compat.
- Audit every `wave_size: int = 64` instance dataclass (about 15 of them) and
  every `helpers/loads.py` loader. Decide per-call site: derive from arch,
  keep explicit, or accept `Optional[int]`. This is a real migration with
  manifest-name implications; see "Backward Compatibility".
- Move `helpers/attention.py` reductions
  (`warp_xor_reduce_max`, `warp_xor_reduce_sum`) behind `arch.wave.reduce_*`.
  The current 4-stage XOR butterfly is wave64-specific and must be
  re-derived for wave32.

### Phase 3: Backend Extraction

- Add `core/isa/backend.py`, `core/isa/gfx950.py`, etc.
- Extract gfx950 behaviour from `lower_llvm.py` into `Gfx950Backend`. gfx950
  LLVM IR output must be byte-identical at the end of this phase.
- Route MFMA declarations, waitcnt, datalayout, kernel attributes, async
  copy, and `ds_read_*_tr_*` through the backend.
- Add golden tests for emitted LLVM snippets and waitcnt encodings, plus a
  differential test that compares pre- and post-refactor LLVM IR for the
  full sweep manifest.

### Phase 4: Second Target Family

- Add `Gfx9MfmaBackend` (gfx90a, gfx942) and validate gfx942 end-to-end on
  hardware if available; otherwise IR-level only.
- Add an RDNA backend (`Gfx11RdnaBackend`) and WMMA catalog entries.
- Bring up one narrow path first (elementwise or small GEMM) before
  expanding to attention.

### Phase 5: Multi-Target Artifacts

AMD's HIP runtime does not load a native "fat" code object across multiple
gfx targets in a single launch. The realistic mechanism is a
per-architecture HSACO bundle plus load-time selection:

- Sweep and manifest schemas key cached HSACOs by `(spec, arch, rocm_version)`.
- A `MultiArchLauncher` loads the bundle entry matching the active device's
  gfx string and forwards launch calls to the right `KernelLauncher`.
- Autotuning records target-specific winners; selection at run time is
  `arch -> spec -> hsaco` lookup.

Do not promise a single-artifact API. The user-visible feature is "build
once for many gfx, launch the right one"; the file layout is a directory or
zip of per-arch HSACOs plus an index.

## Backward Compatibility

ROCKE has roughly thirty instance builders consumed by sweeps, manifests,
examples, tests, `torch_backend.py`, and external scripts. The refactor must
not silently break these callers.

### Default arch

- `compile_kernel(kernel, arch=...)` defaults `arch="gfx950"` through Phase 4.
  No caller is required to change.
- Phase 5 introduces an opt-in `arch="auto"` that consults
  `runtime.hip_module` for the active device. `"auto"` is never the default
  in non-interactive contexts.
- Existing dataclass defaults (`wave_size: int = 64`) remain valid. During
  Phase 2, they additionally accept `wave_size: Optional[int] = None` meaning
  "derive from arch."

### Manifest schema

- Bump from `ck.dsl.example.manifest/v1` to `/v2` once `arch` and
  `rocm_version` are required fields. `/v1` manifests are read by treating
  missing fields as `arch="gfx950"` and `rocm_version=None`.
- `kernel_name()` outputs are byte-identical for gfx950 across Phases 0-3.
  Phase 4 may add an `arch=` suffix only when `arch != "gfx950"`.

### HSACO cache

- Cache key in `runtime/comgr.py` becomes
  `(llvm_text_sha256, arch, target_rocm_profile_id, build_comgr_version)`
  where `target_rocm_profile_id = (profile.version, hash-of-fields)`.
- Existing gfx950 cache entries continue to hit because the new key reduces
  to the old one when target and build ROCm equal the legacy box's version.
- A migration script renames legacy cache entries; we do not invalidate them.
- Editing a `RocmProfile` in place invalidates only its own cache entries.

### IRBuilder API

- `IRBuilder.mfma_f32_*` methods stay as `DeprecationWarning` wrappers for
  exactly one release after Phase 1 ships.
- Removal is announced in `dsl_docs/development/extending.md` and called
  out in the release notes for the cycle that removes them.

## Performance Budget

The whole point of ROCKE is the 5-100 ms compile loop. This refactor adds
indirection (backend dispatch, catalog lookups, legalization). The budget
below is non-negotiable.

- **Compile-time p50 must not regress more than 5 %** on the existing sweep
  manifest, measured warm (comgr cached).
- **Compile-time p95 must not regress more than 10 %** on the same manifest.
- **Cold-start (first compile) is allowed to grow up to 20 %** to cover
  arch metadata load.
- **Autotune winners must reproduce** on a ROCKE-owned benchmark manifest
  (`Python/rocke/benchmark/` fixture, committed to the repo): every config
  the autotuner picks before the refactor must remain a legal pick after,
  with measured TFLOPS within 2 %.
- **gfx950 LLVM IR must be byte-identical** at the end of Phase 3 (after
  backend extraction). This is verified by a differential test on the sweep
  manifest. Phase 0-2 may differ only in textual reordering that survives
  `llvm-as`/`llvm-dis` normalisation.
- **HSACO bytes must be bit-identical** for gfx950 sweep manifest at the end
  of Phase 3, given the same ROCm version.

CI gates these budgets on the standard sweep manifest before any phase is
declared complete.

## Testing Strategy

| Layer | Test |
| --- | --- |
| `ArchTarget` (core) | Round-trip data file, predicate queries (`fits_lds`, `supports_dtype_combo`, `max_vector_load_dwords`), version stamp |
| MMA catalog (core) | enumerate, select_largest_k; supported and rejected dtype/shape combinations |
| Instance policy: gemm | `validate(target, spec)` accepts/rejects each row of the existing gfx950 sweep manifest identically; forbidden combos return stable reason strings; LDS budget arithmetic matches old inline tables |
| Instance policy: attention | `valid_pipelines(target)` selects the same FMHA variant per gfx as today; capability gating composes `target.memory.has_async_lds` and `has_ds_read_tr` |
| Instance policy isolation | importing `core/arch/` does not transitively import any policy module; grep gate verifies no pipeline string appears under `core/` |
| Operand layout | lane → logical for A, B, C across every catalog entry |
| LDS layout | producer/consumer pairing, async invariant, byte-offset goldens |
| Backend | waitcnt encoding, intrinsic declarations, `module_preamble` per `(gfx, rocm_version)` |
| Lowerer | one generic IR lowers identically through target-selected backend |
| End-to-end | gfx950 byte-parity baseline, then narrow second-target smoke |
| Fusion | epilogue fusion via `OperandLayout`; attention via `arch.wave.*` and `AttentionPipelinePolicy.valid_pipelines(target)` |
| Optional | dispatcher parity diagnostic (skipped if `dispatcher/` absent) |

The dispatcher parity diagnostic compares ROCKE's MMA catalog against
`dispatcher/codegen/arch_specs.json` and reports drift. It is informational
only; it must never fail ROCKE CI or block a release.

## Related Subsystems

This plan integrates with existing ROCKE subsystems rather than replacing
them. Implementers should read these before touching code:

- `dsl_docs/reference/mfma_atom_catalog.md` — the documented atom list whose
  entries become the initial gfx9 `MmaCatalog` rows.
- `helpers/fuse.py`, `helpers/fusion_ir.py`, `helpers/fusion_lowering.py`,
  `helpers/fusion_memory.py`, `helpers/fusion_scheduler.py`,
  `helpers/fusion_legalize.py`, `helpers/fusion_validation.py` — the existing
  graph-fusion stack. `LoweringRegistry` entries gain a target parameter;
  `GemmEpilogueLowerer` consumes `OperandLayout` instead of `MfmaAtom`;
  `WorkspacePlanner` reads `arch.lds_capacity_bytes`;
  `fusion_legalize.py` runs before `legalize_for_target` (they do different
  things: one rewrites the fusion graph, the other binds backend op-ids).
- `helpers/layouts.py` and `helpers/loads.py` — sources of the producer and
  consumer LDS layouts that move into `core/layout/`.
- `helpers/attention.py` — source of the wave reductions that move behind
  `arch.wave.*`.
- `core/lower_llvm.py` — split into a generic `_Lowerer` plus the backend
  classes listed in "ISA Backend".
- `runtime/comgr.py` — adds `(arch, rocm_version)` to the cache key.

## Review Rules

Use these as review checks when implementing the plan:

- `Python/rocke/` does not import from `dispatcher/` at any time.
- `instances/` may query `ArchTarget`, but may not import LLVM intrinsic names
  or call ISA-named `IRBuilder` methods after Phase 1.
- `core/ir.py` operation names do not contain gfx-specific names.
- Architecture constants live under `core/arch/` or an `ISABackend`.
- Physical layout code returns coordinates or byte offsets; backends emit LLVM.
- New GPU support for **shared** capabilities adds a data row, an `MmaCatalog`
  extension, layout entries, and a backend class — and does not touch any
  `instances/common/<family>_policy.py` or any `instances/common/` builder. Existing
  policies pick up the new arch automatically through the hardware predicates
  they already compose. A **genuinely arch-divergent algorithm** may add a module
  under `instances/<gfx>/` (and a showcase under `examples/<gfx>/`); this must not
  edit another arch's files, a shared `common/` builder, or `core/`.
- Pipeline names, scheduler strings, warp-tile tables, and LDS budget
  factors live exclusively in `instances/common/<family>_policy.py`. Core files
  (`core/arch/`, `core/ir.py`, `core/lower_llvm.py`) must not contain
  pipeline vocabulary.
- After Phase 1, `instances/<family>/` files must validate specs via the
  family `PipelinePolicy` class. Inline `is_valid_spec` tables outside the
  policy module are not allowed.
- Fusion code composes target-neutral primitives; it does not encode ISA
  details.
- gfx950 byte-identical LLVM IR and HSACO are CI-enforced through Phase 3.

## Success Criteria

- ROCKE builds, installs, and tests without the `dispatcher/` tree on disk.
- Changing `arch="gfx942"` to `arch="gfx950"` does not require instance code
  edits for shared capabilities (kernels in `instances/common/`).
- Adding a target adds data, catalog rows, layout tables, and one backend
  class — never an edit to a kernel-family policy, a shared `common/` builder,
  or the generic `_Lowerer` walk. Genuinely arch-divergent algorithms may add
  modules under `instances/<gfx>/` and `examples/<gfx>/`, which leave every
  other arch untouched.
- Code organization follows the hybrid layout: polymorphic `core/` + `helpers/`
  (no gfx-id subfolders); `instances/{common,<gfx>,<family>/policy.py}`;
  `examples/{common,<gfx>}`. Relocated modules keep back-compat shims at their
  old import paths.
- Adding a kernel family or a pipeline variant edits only that family's
  `instances/common/<family>_policy.py` and the family's instance builder — never
  `core/arch/`.
- Adding a ROCm release adds a `RocmProfile` row — never an edit to backend
  code in the common case.
- Cross-compilation works: artifacts produced against `target_rocm` other
  than the host's `build_rocm` are recorded and reproducible.
- Pipeline `(pipeline, scheduler, epilogue)` combo validity comes from
  `instances/common/<family>_policy.py.validate(target, spec)`, not from inline
  tables and not from core. Forbidden combos fail with stable, structured
  reasons.
- Attention pipeline-variant choice (`qr_ks_vs`, `qr_ks_vs_async`,
  `qr_ks_vs_whole_k_prefetch`, ...) is selected via
  `AttentionPipelinePolicy.valid_pipelines(target)`, not hardcoded per
  kernel.
- Epilogue fusion uses `OperandLayout` contracts, not MFMA-specific helper
  assumptions.
- Attention-style fusion remains single-kernel where appropriate while using
  target-selected wave, MMA, LDS, and wait primitives.
- Compile-time and HSACO-byte budgets in "Performance Budget" hold.
- The optional dispatcher parity diagnostic exists, runs when the dispatcher
  tree is present, and is not on the CI critical path.
