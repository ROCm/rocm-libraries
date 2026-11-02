<!-- Copyright Advanced Micro Devices, Inc., or its affiliates. -->
<!-- SPDX-License-Identifier: MIT -->

# TensileLite C++ Naming Audit

Status: Findings recorded; no implementation changes made

Audit date: 2026-08-17

Scope: `projects/hipblaslt/tensilelite/src` and the TensileLite C++ surfaces
needed to classify names correctly

## Purpose

This record identifies legacy `Tensile` names in the TensileLite C++ runtime
and separates true TensileLite naming debt from compatibility contracts that
must not be changed by a mechanical rename.

The audit inspected the complete `src/` tree, its headers, client, and C++
tests. It temporarily materialized `shared/tensile` to compare the original
Tensile project's macro and environment-variable usage, then restored the
previous sparse-checkout paths.

## Scope and invariants

- Do not modify paths under `projects/hipblaslt/library`.
- Do not rename output paths that point to `hipblaslt/library`, or output
  directories that intentionally use the `Tensile` name there.
- Do not change original Tensile under `shared/tensile` as part of this work.
- Treat installed C++ headers, virtual methods, compile definitions, and
  environment variables as compatibility interfaces until a transition policy
  is explicitly chosen.

There are no references in `tensilelite/src` to a protected
`hipblaslt/library/Tensile` output path or output directory. The source-side
findings below can therefore be addressed without changing that output-path
contract.

## Current state

All C++ namespaces and declarations in `src` already use `TensileLite`; there
is no stale `namespace Tensile` or `class Tensile...` declaration to rename.
The remaining occurrences fall into the categories below.

### Direct, in-scope file rename

`src/Tensile.cpp` is the only `Tensile`-named file in `src`. It implements
objects in `namespace TensileLite` and is compiled only because
`src/CMakeLists.txt` lists it.

The direct rename is:

```text
src/Tensile.cpp -> src/TensileLite.cpp
```

The required companion updates are:

- `src/CMakeLists.txt`: update the `target_sources(tensilelite-host ...)`
  entry.
- `tests/CMakeLists.txt`: update its explanatory source-file comment.

Generated build artifacts under `build_tmp*` will be regenerated; they are not
source edits.

### Public solver method names

`ContractionSolution` exposes these public virtual methods:

```text
solveTensileGPU
solveTensileGroupedGemmGPU
```

Their declarations are in `include/Tensile/ContractionSolution.hpp`; their
definitions and internal call are in `src/ContractionSolution.cpp`; the only
in-tree caller is `client/main.cpp`. A consistent new spelling would be:

```text
solveTensileLiteGPU
solveTensileLiteGroupedGemmGPU
```

No caller outside TensileLite was found in this checkout, and the original
Tensile project does not define either method. These are nevertheless installed
header virtual methods, so renaming them is a public C++ API and ABI decision.
Choose either a clean break or an explicit compatibility wrapper before making
this rename.

### Header include-root name

Twenty-seven implementation files include headers through `<Tensile/...>`:

- 23 are built by the current host-library CMake graph.
- Four are dormant: `ArithmeticUnitTypes.cpp` is commented out of the source
  list, and the three `src/ocl/*.cpp` files are in a directory that is not
  added by `src/CMakeLists.txt`.

Changing these source includes to `<TensileLite/...>` requires a full header
tree migration. It cannot delete the existing `include/Tensile` tree under the
current scope because it is consumed by these out-of-scope files:

- `library/src/amd_detail/hipblaslt-ext-op-internal.hpp`
- `library/src/amd_detail/hipblaslt-ext-op.cpp`
- `library/src/amd_detail/rocblaslt/src/include/UserDrivenTuningParser.hpp`
- `library/src/amd_detail/rocblaslt/src/include/tensile_host.hpp`
- `library/src/amd_detail/rocblaslt/src/rocblaslt_auxiliary.cpp`
- `library/src/amd_detail/rocblaslt/src/rocblaslt_transform.cpp`
- `library/src/amd_detail/rocblaslt/src/tensile_host.cpp`

Two additional hipBLASLt consumers outside TensileLite use the same include
prefix: `clients/common/src/utility.cpp` and
`clients/tests/src/caching_library_gtest.cpp`.

If the TensileLite-native include root is desired, add it and update
TensileLite-owned sources, but retain the old `<Tensile/...>` headers as a
compatibility surface until the protected consumers can migrate. Do not modify
the listed `library` files in this branch.

### Dormant OpenCL target

`src/ocl/CMakeLists.txt` names a static target `TensileOcl` and emits a message
with that name. `src/CMakeLists.txt` does not add the `ocl` subdirectory, so
the target is currently unreachable. If that configuration is retained, rename
the target to `TensileLiteOcl` as part of a deliberate OpenCL cleanup; do not
treat it as an active host-library rename.

### Comments and local names

The following are safe wording-only cleanup candidates:

- `src/ContractionProblem.cpp`: `Tensile Indices for contraction problem`.
- `src/ContractionSolution.cpp`: references to a `Tensile tensor name`,
  `Tensile's transA()`, `tensileLite`, and `Tensile debugging`.
- `src/Debug.cpp`: local names such as `tensile_metric`,
  `tensile_benchmark`, and `tensile_marker`.

Use `TensileLite` in prose and neutral local names such as `metricEnv` where
the old name is only an implementation detail. Renaming a local variable does
not imply that the environment variable it reads must be renamed.

## Macro and environment-variable classification

The original project at `shared/tensile` was checked before classifying these
names. The following table is the source of truth for a future implementation
pass.

| Category | Names | Required treatment |
|---|---|---|
| Used by original Tensile and TensileLite | `TENSILE_ASSERT_EXC`, `TENSILE_YAML`, `TENSILE_MSGPACK`, `TENSILE_DB`, `TENSILE_DB2`, `TENSILE_NAIVE_SEARCH`, `TENSILE_METRIC`, `TENSILE_SOLUTION_INDEX` | Do not globally rename. Any TensileLite-only change must leave original Tensile untouched and preserve the required compatibility contract. |
| Required by protected hipBLASLt library code | `TENSILE_YAML` | Keep this compile definition while `library/src/amd_detail/rocblaslt/src/tensile_host.cpp` checks it. Renaming it requires an out-of-scope library change or an explicit compatibility definition. |
| TensileLite-only feature macros | `TENSILE_USE_FP6`, `TENSILE_USE_BF6`, `TENSILE_USE_FP4` | Good candidates for `TENSILELITE_USE_*`. They occur in 11 TensileLite files spanning headers, host source, client source, and tests, so update the component atomically. |
| TensileLite-only build knob | `Tensile_ENABLE_MARKER` | Rename to `TENSILELITE_ENABLE_MARKER` if the option remains supported. Update both the compile-time guard and its user-facing diagnostic. |
| TensileLite-only environment variables | `TENSILE_ADAPTIVE_GEMM_LOG`, `TENSILE_AUTO_GSU_ALGO`, `TENSILE_ADAPTIVE_GEMM_NTAB_ALGO`, `TENSILE_BENCHMARK`, `TENSILE_DISABLE_STAGGERU`, `TENSILE_ENABLE_MARKER`, `TENSILE_GRIDBASED_BATCH_EXP`, `TENSILE_GRIDBASED_KDTREE`, `TENSILE_MAX_DECOMPRESSED_BYTES`, `TENSILE_PREDICTION_LIB`, `TENSILE_SOLUTION_SELECTION_METHOD`, `TENSILE_STREAMK5_FORCE_MODE`, `TENSILE_STREAMK_DATA_PARALLEL` | These can become `TENSILELITE_*`, but they are user-facing runtime contracts. Decide whether to make a documented breaking change or accept both the new and legacy names during a migration window. Update the corresponding C++ tests when changing the spelling. |

`TENSILE_ASSERT_EXC` has no non-TensileLite in-tree consumer in this checkout,
but it is defined in an installed TensileLite header. A local
`TENSILELITE_ASSERT_EXC` rename is technically isolated from original Tensile;
preserving the old macro as an alias remains the safer public-header transition.

`TENSILE_MSGPACK` is private to the TensileLite host target and can be renamed
independently if desired. The original Tensile project uses the same spelling,
so its configuration must not be changed by this branch.

## Recommended implementation order

1. Make the unambiguous source-file rename, update its CMake entry and test
   comment, and correct the stale comments and local variable names.
2. Decide the virtual-method compatibility policy before renaming the solver
   methods.
3. Rename the TensileLite-only feature macros in one component-wide change,
   including headers, client code, and tests.
4. Decide whether the TensileLite-only environment variables retain legacy
   aliases; implement and test that policy consistently.
5. Only then consider a `<TensileLite/...>` header root. Preserve
   `<Tensile/...>` until all protected and external consumers have an approved
   migration path.

## Relationship to existing decisions

`docs/PackagingDecisions.md` records that the prior Python-package migration
did not rename C++ `include/Tensile`, generated-kernel names, logic fields, or
the host ABI. This audit does not override that decision. Any change to the
public solver names or header include root should first update the accepted
decision record with the chosen compatibility policy.
