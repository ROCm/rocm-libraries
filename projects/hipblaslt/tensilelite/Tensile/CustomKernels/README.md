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
| `rocroller/`  | External: rocRoller-sourced kernels.              |
| `wave/`       | External: Wave (handwritten) kernels.             |
| `triton/`     | External: Triton-compiled GEMM kernels.           |

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

## Adding a new external kernel

The `Tensile.AddCustomConfig` helper extracts the Tensile-side interface from
a benchmark test YAML and injects a `custom.config` block into the `.s` file:

```bash
python -m Tensile.AddCustomConfig \
    Tensile/CustomKernels/aiter/<kernel>.s \
    --yaml Tensile/Tests/custom/<test>.yaml
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

## Validation

Two ways to validate:

1. **CI gate** (recommended for pull-request checks):

   ```bash
   python -m Tensile.ValidateMetadata --strict
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
