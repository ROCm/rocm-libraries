<!-- Copyright Advanced Micro Devices, Inc., or its affiliates. -->
<!-- SPDX-License-Identifier: MIT -->

# ADR: gfx90c XNACK target policy

## Status

Accepted.

## Context

`gfx90c`, `gfx90c:xnack+`, and `gfx90c:xnack-` identify one GPU ISA,
`IsaVersion(9, 0, 12)`. XNACK controls recoverable memory faults in the code
object target metadata; it does not define a different instruction-set
architecture or justify a separate solution-selection database.

TensileLite's assembly pipeline currently groups kernels by numeric ISA. It
therefore cannot emit independent positive and negative XNACK object sets for
one ISA in a single `TensileCreateLibrary` invocation without colliding output
names or losing the association between a kernel and its full target ID.

## Decision

- All three target IDs use the existing gfx90c logic, metadata, lazy-loading
  database, and `library/gfx90c/` directory.
- A build requesting one target preserves it through assembly directives,
  assembler/compiler arguments, bundling, and ELF target features.
- A build requesting both explicit XNACK variants deliberately emits bare
  `gfx90c`. This fallback is sorted and deterministic rather than dependent on
  command-line, set, or dictionary ordering.
- Bare gfx90c means XNACK-unspecified. Code Object V4 or newer is required
  because its ELF flags represent `XNACK_ANY`; TensileCreateLibrary accepts
  only V4/V5 and defaults to V4.
- Sanitizer builds regard gfx90c as XNACK-capable. A bare request normalizes to
  `gfx90c:xnack+`; an explicit `gfx90c:xnack-` request fails rather than being
  silently rewritten.
- Runtime selection strips target features at the first colon for metadata and
  directory lookup. ELF compatibility remains the HIP/HSA loader's job.

## Consequences

There is one tuning and solution-selection database, avoiding duplicated logic
that could drift. A request for both explicit variants receives the broadly
compatible unspecified object rather than two specialized objects. Users who
need an explicit mode must perform a single-target build.

## Runtime compatibility test plan

Build and install the device library three times, once per target, and retain
each installation separately:

```console
cd projects/hipblaslt
invoke build -a gfx90c
invoke build -a gfx90c:xnack-
invoke build -a gfx90c:xnack+
```

On gfx90c hardware, run the `gfx90c_integration_fp32`,
`gfx90c_integration_fp16`, `gfx90c_integration_hpa_tail`, and
`gfx90c_integration_hpa` client cases from `clients/tests/data/matmul_gtest.yaml`
against the matching installation. Record `gcnArchName`, the build target,
`HSA_OVERRIDE_GFX_VERSION`, and `HSA_XNACK`. Test this matrix:

| Build target | `HSA_XNACK=0` | `HSA_XNACK=1` |
|---|---|---|
| `gfx90c` | load and pass | load and pass |
| `gfx90c:xnack-` | load and pass | loader rejection |
| `gfx90c:xnack+` | loader rejection | load and pass |

The gtests initialize inputs deterministically, synchronize the operation, and
compare the computed matrix with a CPU reference. A launch-only result,
all-zero output, unchanged output, or a performance measurement is not
correctness evidence. CI that cannot switch the machine's XNACK mode must
report the hardware cases as not run; it must not substitute a fabricated
pass.

## Future migration

Emitting both explicit variants in one invocation requires grouping by full
target ID, carrying that ID in kernel-generation structures, adding feature
suffixes to assembly code-object filenames, preventing installation
collisions, teaching lazy loading to select a compatible variant, and defining
deterministic preference/fallback rules. That redesign is deferred.
