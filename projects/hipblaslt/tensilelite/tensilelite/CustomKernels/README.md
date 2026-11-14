<!--
Copyright (C) 2025-2026 Advanced Micro Devices, Inc.
SPDX-License-Identifier: MIT
-->

# Custom kernels

Each custom kernel `.s` file embeds its Tensile-side metadata directly inside
the `.amdgpu_metadata` YAML section, under a top-level `custom.config:` key.
Co-locating provenance and build instructions with the kernel they describe
makes kernels self-describing: the same file can be loaded through a logic
file, a benchmark `CustomKernels:` list, or a CI gate without consulting any
external manifest.

## Subdirectories

| Directory     | Origin                                            |
| ------------- | ------------------------------------------------- |
| `tensile/`    | Kernels emitted by Tensile's own assembly writer. |
| `aiter/`      | External: AITER-sourced GEMM kernels.             |
| `ck/`         | External: Composable Kernel-sourced kernels.     |
| `rocblas/`    | External: rocBLAS wvSpltK skinny-GEMM kernels.   |
| `vllm/`       | External: vLLM wvSplitK skinny-GEMM kernels.     |
| `rocroller/`  | External: rocRoller-sourced kernels.              |
| `wave/`       | External: Wave (handwritten) kernels.             |
| `triton/`     | External: Triton-compiled GEMM kernels.           |
| `rocke/`      | External: rocKE GEMM kernels.                     |

Loader behavior is the same in every subdirectory; the split is purely
organizational.

## Triton-sourced kernels

The `triton/` subdirectory holds GEMM kernels compiled from Triton
(`@triton.jit`). Only the `.s` file and its test YAML are committed.
The Triton kernel `.py` comes from aiter and is a local input
to the generation step; it is not maintained in this repository.

Triton-specific concerns for the compile step / YAML argument map:

- **Static LDS**: Triton emits *dynamic* LDS (`group_segment_fixed_size: 0` +
  `sharedMemBytes` at launch), but the custom-call path launches with
  `sharedMemBytes = 0`. Rewrite the kernel's LDS into a static group segment
  (size = Triton's `metadata.shared`) so the driver allocates it, exactly like
  the other custom kernels.
- **Trailing scratch pointers**: Triton appends `global_scratch` and
  `profile_scratch` kernarg pointers; when they are unused, represent their
  16-byte footprint as trailing `padding` on the last real Triton argument.
- **Layout**: for the `Cijk_Alik_Bljk` (TN) contraction both operands are
  K-contiguous; pass packed-byte leading strides via `StrideA0Bytes` /
  `StrideB0Bytes` for sub-byte (FP4) data.
- **Native aiter signatures**: use `ConstantOne` for fixed unit-stride Triton
  arguments in the constrained custom-kernel layout. Use a named semantic such
  as `StrideCK` when the value depends on split-K. For packed FP4 kernels that
  take the summation size in bytes, use `SizeSumDiv2`.
- Strip the `.amdgcn_target` / `.amdhsa_code_object_version` directives from the
  emitted assembly so Tensile's assembler flags drive target + COV.

## What `custom.config` looks like

Tensile-generated kernels (no `Source:` block) need only the runtime
requirement:

```yaml
custom.config:
  InternalSupportParams:
    KernArgsVersion: 0
```

Their `ProblemType`, `MatrixInstruction`, and tuning state come from the
consuming logic file or test YAML and are merged on top of `custom.config` at
load time. Most Tensile-generated kernels also embed those fields for
forensic value, but they are not required.

External kernels must additionally carry full provenance and the Tensile-side
interface, because Tensile cannot regenerate them and benchmark/test YAMLs
need a stable handle:

```yaml
custom.config:
  Source:
    Origin: aiter
    Repository: https://github.com/...
  Version: 1.0.0
  Features:
    SupportsUserArgs: false
    SupportsBias: false
    SupportsActivation: false
    SupportsScaleAlpha: false
    SupportsGSU: false
  InternalSupportParams:
    KernArgsVersion: 0
  ProblemType: { ... }
  CustomKernel:
    args: [ { type: address, semantic: AddressD, padding: 8 }, ... ]
    macrotile: [256, 256, 64]
    threads: [256, 1, 1]
    grid: [TilesX, TilesY, One]
  MatrixInstruction: [16, 16, 32, 1]
  EnableMatrixInstruction: True
  MIWaveTile: [8, 8]
  WavefrontSize: 64
```

`Source`, `Version`, and `Features` are presence-checked but otherwise
provenance only — they are read by the validator, not by the runtime.

Under uniform summation order, which hipBLASLt applies unless the caller turns
it off, hipBLASLt refuses an external kernel unless its `CustomKernel` block
sets `uniformSummationOrder: true` (default `false`). Set it only for a kernel
that reduces every row of D in the same order, so that identical rows of A give
bitwise-identical rows of D, and that has no compiled-in StaggerU, which
hipBLASLt still refuses. The declaration has to hold for every problem the
kernel's predicates admit, including partial tiles, any K, and padded leading
dimensions; hipBLASLt has no way to check it. A kernel that declares it also
needs a case in `clients/tests/src/row_uniformity_gtest.cpp` (see the
`SkinnyTn` cases): a shape library logic routes to it, checked row-uniform on
the device, with the kernel asserted as the default pick on its architecture.
`AddCustomConfig` copies the key from the kernel's `CustomKernel` entry in the
test YAML.

## Adding a new external kernel

The `tensilelite.AddCustomConfig` helper extracts the Tensile-side interface from
a benchmark test YAML and injects a `custom.config` block into the `.s` file:

```bash
python -m tensilelite.AddCustomConfig \
    tensilelite/CustomKernels/aiter/<kernel>.s \
    --yaml tensilelite/Tests/custom/<test>.yaml
```

Useful flags:

| Flag             | Purpose                                                              |
| ---------------- | -------------------------------------------------------------------- |
| `--yaml <path>`  | Tensile test YAML to pull `ProblemType` / `CustomKernel` / `MI` from |
| `--origin <s>`   | Override auto-detected origin (defaults to parent directory name)    |
| `--repository`   | Source repository URL                                                |
| `--version`      | Kernel version string (defaults to `1.0.0`)                          |
| `--dry-run`      | Print the block that would be inserted without modifying the file    |

Without `--yaml` the tool injects a provenance-only block; the kernel won't
be usable through the `CustomKernels:` list path until the interface is
filled in by hand.

The tool refuses to operate on a file that already contains a `custom.config`
block; if you need to regenerate one, delete the existing block first.

If library logic references the kernel, TensileCreateLibrary links it into the
same code object as Tensile's kernels, and `ld.lld` rejects mixed ABI versions.
The kernel must then be at code object version 4, the version hipBLASLt builds
its libraries at: either strip its `.amdhsa_code_object_version` directive so the
assembler's flag applies, or emit it with `-mcode-object-version=4`.

A kernel captured from rocRoller arrives carrying rocRoller's internal view of
the kernel, in three forms that nothing here reads:

- a dump of the kernel representation inside `.amdgpu_metadata`, under the keys
  `.kernel_graph`, `.kernel_graph_dot` and `.command`;
- the launch metadata rocRoller's own host side evaluates: `.workitem_count`,
  `.dynamic_sharedmemory_bytes`, `.kernel_dimensions` and `.workgroup_size` on
  the kernel entry, and `.expression` and `.variableType` on each `.args`
  entry. hipBLASLt launches from `custom.config` instead, and the GPU runtime
  ignores these keys. That holds only when `custom.config` declares a
  `CustomKernel` block: without one, the arguments are rebuilt from
  `.expression` and `.variableType`, so the strip refuses such a kernel;
- whole-line `//` comments narrating codegen -- register allocation, expression
  lowering, liveness -- which outnumber the instructions by better than ten to
  one.

They dominate the file, and the metadata also rides into the assembled object,
so remove them before committing the kernel:

```bash
python -m tensilelite.StripKernelGraphs tensilelite/CustomKernels/rocroller/<kernel>.s
```

The launch-metadata and comment strips run only where `custom.config` says
`Origin: rocroller`, and the comment strip keeps any comment carrying a licence
or an annotation marker (`TODO`, `FIXME`, `XXX`, `HACK`, `NOTE`, `WARNING`); the
hand-written kernels elsewhere in this tree are left alone. Instructions keep
their trailing `// (op N)` tags, so the assembled `.text` is unchanged -- only
the metadata note shrinks, by exactly the deleted keys.

Pass `--check` instead to report without writing; it exits non-zero when
provenance is still present, which suits a pre-commit or continuous-integration
gate.

### Checked-in rocRoller GEMM kernels

The `RR_GEMM_*` kernels in `rocroller/` are gfx950 MX GEMMs captured from the
rocRoller JIT path, one per configuration rocRoller picks. A build without
rocRoller serves a rocRoller-routed GEMM with the kernel whose tile, unroll,
workgroup mapping and Stream-K mode equal rocRoller's pick, or with nothing. Each
kernel's `custom.config` therefore declares `unroll`, and declares
`uniformSummationOrder: true` only when the kernel is listed in
`Tests/unit/data/rocroller_uso_verified.txt`. They read packed E8M0 scales, so
each declares `MXScaleFormat: NoSwizzle`. The kernels a logic file registers
name the bare `amdgcn-amd-amdhsa--gfx950` target, so they load whatever the
SRAMECC mode.

A name ending in `_LDSSYNC` is the kernel rocRoller builds for its pick when the
first attempt leaves diverging waitcnt states at the K-loop labels: the retry
drains the LDS queue there. It serves the same tile, unroll and workgroup
mapping as the name without the suffix would.

Only a build without rocRoller selects them. A build with rocRoller serves these
problems through rocRoller itself, and its Tensile paths (the C API with
`HIPBLASLT_USE_ROCROLLER=0`, the `hipblaslt_ext` APIs, `getAllAlgos`, support
checks) must return what they return without these kernels. Each one's hardware
predicate therefore carries a `RocRollerCaptures` term, which only an `AMDGPU`
whose `rocRollerCaptures` flag is set satisfies; hipBLASLt sets the flag only
when built without rocRoller, and the TensileLite client always sets it.

A name with `_SK_WGT_` is a Stream-K kernel: its `CustomKernel` block declares
`grid: [RocRollerStreamK, One, One]` and `workspaceType: RocRollerStreamK`, and
the launch sizes the grid from its `occupancy` and allocates the partials and
flags workspace. These kernels take no workgroup-mapping kernarg and do not
declare uniform summation order; a build without rocRoller serves them under
the default setting anyway, as the rocRoller JIT path does. rocRoller's
Stream-K loop is only correct when no workgroup runs a data-parallel tile, so
every launch and selection path refuses a problem with at least twice as many
tiles as the grid has workgroups. `Tests/unit/test_rocroller_streamk_corpus.py`
pins these properties.

The GridBased logic files that register the `RR_GEMM_*` kernels are
`library/.../Logic/asm_full/gfx950/gfx950/GridBased/*_MXAE8B32_MXBE8B32_BH_UserArgs.yaml`.
They declare `DeviceNames: [Device 75a3, Device 75a2, Device 75a0]`, the list
`LibraryLogic` gives a generated gfx950 library: 75a3 and 75a2 match it
directly and every other gfx950 chip id falls back to 75a0. They give every
kernel an ExactLogic point and set no size bound: a kernel's tile, batch and
buffer predicates say where it is correct. Unit tests
pin these properties (`Tests/unit/test_rocroller_*`); a change to a kernel's
predicates shows up as a diff of `Tests/unit/data/rocroller_predicate_manifest.yaml`,
regenerated with `HIPBLASLT_UPDATE_RR_PREDICATE_MANIFEST=1`.

The layout in a kernel's name (`RR_GEMM_<A><B>_`, `T` for a transposed
operand) selects its logic file: `Cijk_Alik_Bljk` for TN, `Cijk_Ailk_Bljk` for
NN, `Cijk_Ailk_Bjlk` for NT and `Cijk_Alik_Bjlk` for TT, each with that
layout's `ProblemType`. The library offers a kernel only for problems with its
file's transposes.

## Validation

Two ways to validate:

1. **CI gate** (recommended for pull-request checks):

   ```bash
   python -m tensilelite.ValidateMetadata --strict
   ```

   Walks the `CustomKernels/` tree, validates every `.s` file, exits non-zero
   on any failure. Without `--strict`, failures are reported as warnings only
   and the exit code is `0`.

2. **Build-time** (off by default):

   Set the YAML toggle `GlobalParameters.ValidateMetadata: True` in your
   benchmark configuration, or pass `--validate-metadata` to `Tensile`. With
   the flag on, `buildAssemblyCodeObjectFiles` validates every kernel that
   participates in the build and prints `Tensile::WARNING: ...` for any
   missing or invalid `custom.config` while letting the build proceed.

### Validation rules

| Kind                 | Required in `custom.config`                                                                                          |
| -------------------- | -------------------------------------------------------------------------------------------------------------------- |
| Tensile-generated    | `InternalSupportParams.KernArgsVersion`                                                                              |
| External (`Source:`) | All of: `Source.Origin`, `Features` (mapping), `InternalSupportParams.KernArgsVersion`, `Version`, `ProblemType`, `MatrixInstruction`, `CustomKernel` with `args` / `macrotile` / `threads` / `grid` |
