# Subtile Python ↔ C++ Boundary Contract

Status: contract / no behavior change. This document defines the minimal Python
surface of the `Tensile/Components/Subtile/*` package that non-Subtile code is
allowed to import once the Subtile infrastructure becomes C++ owned. It is the
boundary that later Python-deletion tasks must satisfy: any module not listed
here as a stable external API may be deleted or moved into C++ without updating
external callers.

- Scope of the audit: `rocm-libraries/projects/hipblaslt/tensilelite` only.
- Branch: `refactor/incremental` (base `develop`), arch `gfx950`,
  Docker image `hipblaslt-dev:latest`.
- This slice deletes no Python and changes no code behavior. It records the
  contract and (optionally) guards it with a source-text import test.

## Repeatable import audit

Run from `rocm-libraries/projects/hipblaslt/tensilelite`:

```bash
# Every external importer of the Subtile package (excludes the package itself):
rg -n "(from\s+\S*Subtile|import\s+\S*Subtile)" Tensile --glob '*.py' \
  | rg -v "Tensile/Components/Subtile/"

# Production-only importers (drop the unit tests):
rg -n "(from\s+\S*Subtile|import\s+\S*Subtile)" Tensile --glob '*.py' \
  | rg -v "Tensile/Components/Subtile/" \
  | rg -v "Tensile/Tests/unit/"
```

The production filter must return exactly the two sites listed below. Any new
hit is a boundary violation that must be triaged before merge.

## Production import sites (must stay minimal)

Only two non-test, non-Subtile modules import from the Subtile package:

| Caller | Import | Allowed surface |
|---|---|---|
| `Tensile/KernelWriter.py:50` | `from .Components.Subtile.Kernel import *` | `Kernel.py` facade symbols (see below) |
| `Tensile/Components/StreamK.py:37` | `from .Subtile.SubtileLREmit import localReadResetOffsetsSubtile` | `localReadResetOffsetsSubtile` only |

### Allowed `Kernel.py` facade surface used by `KernelWriter.py`

`KernelWriter.py` uses the wildcard import, but the symbols it actually
references (verified by name search against `KernelWriter.py`) are a strict
subset:

| Symbol | Kind | Used at |
|---|---|---|
| `TileInfo` | class — runtime tile state ctor | `KernelWriter.py` 6278, 6283, 6286 |
| `selectABGeometry(kernel, tc)` | geometry selection helper | `KernelWriter.py` 6282 |
| `selectDGeometry(kernel)` | geometry selection helper | `KernelWriter.py` 6277 |
| `selectMXScaleGeometry(kernel, tc)` | geometry selection helper | `KernelWriter.py` 6285 |
| `initVgprTilesToZero(writer, kernel, tileInfo)` | setup emit helper | `KernelWriter.py` 4615 |
| `mainLoop(writer, kernel, tpA, tpB)` | main-loop orchestrator | `KernelWriter.py` 4624 |

This is the minimal production-safe `Kernel.py` surface — smaller than the
candidate set in the task. In particular, the following `Kernel.py` symbols are
**not** referenced by `KernelWriter.py` and are therefore **not** part of the
required external surface:

- `preLoop` — `KernelWriter.py` calls `skComponent.preLoop` (the StreamK
  component method), not `Kernel.preLoop`; `Kernel.preLoop` is an internal /
  placeholder helper.
- `emitMfmaCode`, `emitMfmaInstruction`, `_selectF8F6F4InstType` — internal to
  the Subtile main-loop emit; used by `InstructionEmitter.py` and unit tests
  only.
- `RegisterTileInfo` — constructed internally by `LogicalScheduler.py` via a
  local `from Tensile.Components.Subtile.Kernel import RegisterTileInfo`; not a
  `KernelWriter.py` dependency.
- The frozen geometry config instances (`AB_B16`, `AB_B8`, `AB_B4`,
  `AB_B16_2x2`, `AB_B4_2x2`, `AB_B16_TLU1`, `AB_B16_TLU1_16x1`, `CD_F32`,
  `MXSA_B4`/`MXSB_B4`, `MXSA_B8`/`MXSB_B8`) — `KernelWriter.py` reaches these
  only through the `select*Geometry` helpers, so they are an internal detail of
  the geometry selectors and need not be public.

Recommended end-state: replace the `import *` in `KernelWriter.py` with an
explicit import of the six symbols above (or expose them through an explicit
`__all__` on `Kernel.py`) so the wildcard cannot silently widen the surface.
That tightening is a follow-up task, not part of this contract slice.

## Test-only import sites (not public API)

These live under `Tensile/Tests/unit/` and exist to pin C++/Python parity and
regression values. They reach into internal Subtile modules deliberately and
**must not** be treated as a stable external API. They may need updating when
internals move to C++, and that is expected.

> **Writer-free parity moved to native C++.** The geometry, TileInfo query, and
> emit-leaf instType/load-plan golden cases that previously lived in
> `test_subtileGeometryCpp.py`, `test_tileInfoCpp.py`, and the writer-free
> portion of `test_subtileEmitLeavesCpp.py` are now native gtest under
> `cpp_migration/cpp/tests/` (`subtile_geometry_test.cpp`, `tile_info_test.cpp`,
> `emit_leaves_test.cpp`). Those Python files were deleted because they only
> compared the Python facade against the (now sole) C++ implementation. The one
> retained case — `emitMfmaInstruction` rendering a real rocisa MFMA module — is
> genuine KernelWriter/rocisa integration and lives in
> `test_subtileEmitMfmaRocisa.py`, not C++ parity.

> **Scheduler parity moved to native C++.** The LogicalScheduler value/config
> types and writer-free pass pipeline, and the InstructionScheduler slot
> placement / vmcnt golden cases, that previously lived in
> `test_logicalSchedulerCpp.py`, `test_instructionSchedulerCpp.py`, and the emit
> snapshot regressions in `test_SubtileBasedSchedulerRef.py` are now native
> gtest under `cpp_migration/cpp/tests/` (`logical_scheduler_test.cpp`,
> `logical_scheduler_passes_test.cpp`, `instruction_scheduler_test.cpp`). Those
> Python files were deleted because they only compared the Python facade against
> the (now sole) C++ implementation or duplicated emit snapshots. The genuine
> writer/rocisa integration — the Python LogicalScheduler driving the C++ pass
> pipeline through `populate_instructions`, VGPR allocation, and main/tail-loop
> rocisa emission — is retained in `test_SubtileBasedLogicalScheduler.py`.

| Test file | Subtile module(s) imported |
|---|---|
| `test_subtileMainloopE2ECpp.py` | `LogicalScheduler.LogicalScheduler` (+ helpers from `test_SubtileBasedLogicalScheduler`) |
| `test_subtileOffsetAssignCpp.py` | `Kernel` (`TileInfo`, `AB_B16`, `AB_B8`), `SubtileGREmit`, `SubtileLREmit` |
| `test_subtileEmitMfmaRocisa.py` | `Kernel.emitMfmaInstruction` (rocisa integration — see note below) |
| `test_SubtileBasedLogicalScheduler.py` | `Kernel`, `LogicalScheduler`, `InstructionScheduler.instructionSchedule`, `LogicalScheduler.WaitGROp` (writer/rocisa integration) |
| `test_gr_offset.py` | `SubtileGREmit.graTileAssignment` |
| `test_emitMfmaInstruction.py` | `Kernel.emitMfmaInstruction` |
| `test_selectMXScaleGeometry.py` | `Kernel` (MX scale selectors/configs) |
| `test_storeD_roundtrip.py` | `Kernel` (`TileInfo`, `CD_F32`) |
| `gpu_test_helpers.py` | `Kernel` (`TileInfo`, `AB_B16`, `AB_B8`), `SubtileGREmit`, `SubtileLREmit` |

## Internal-only modules (NOT stable external APIs)

The following are implementation detail of the Subtile package. Production code
outside the package must not import them; they are free to be reshaped or moved
into C++:

- `Tensile/Components/Subtile/SubtileGeometry.py` — value/query geometry layer
  (already a thin facade over the `tensile_writer.subtile.geometry` C++
  extension).
- `Tensile/Components/Subtile/InstructionScheduler.py` — `instructionSchedule`
  and scheduling internals (C++-backed; test-only callers).
- `Tensile/Components/Subtile/InstructionEmitter.py` — MFMA emit driver;
  imports `Kernel.emitMfmaInstruction` internally.
- `Tensile/Components/Subtile/LogicalScheduler.py` — Subtile logical scheduler;
  internally imports `Kernel.RegisterTileInfo`, `InstructionScheduler`,
  `InstructionEmitter`.
- GR / LR / scale emit module internals:
  - `Tensile/Components/Subtile/SubtileGREmit.py` (except as re-exported by
    `Kernel.py`)
  - `Tensile/Components/Subtile/SubtileLREmit.py` — internal, **except**
    `localReadResetOffsetsSubtile`, which is the single sanctioned symbol for
    `StreamK.py`.
  - `Tensile/Components/Subtile/SubtileScaleEmit.py`
- `cpp_migration/tensile_writer/subtile/*` (`tile_info.py`, `geometry.py`,
  `emit.py`, `logical_scheduler.py`, `instruction_scheduler.py`) — Python
  reference shims / nanobind binding glue for the compiled
  `tensile_writer.subtile` extension. These are internal migration scaffolding,
  not a public Python API.

## Summary of the stable external Python surface

```
Tensile.Components.Subtile.Kernel:
    TileInfo
    selectABGeometry
    selectDGeometry
    selectMXScaleGeometry
    initVgprTilesToZero
    mainLoop

Tensile.Components.Subtile.SubtileLREmit:
    localReadResetOffsetsSubtile
```

Everything else in `Tensile/Components/Subtile/*` and in
`cpp_migration/tensile_writer/subtile/*` is internal and may change or be
deleted without notice to external callers.
