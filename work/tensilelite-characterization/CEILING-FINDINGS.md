# Coverage ceiling findings (TensileLite, CPU-only characterization)

**Status: stable at TOTAL 67.92%** (`coverage/master-baseline-P5.txt`), up from a
measured **30.62%** baseline — 2513 passed / 201 skipped / **0 failed**, fully
add-only, zero source changes. Coverage more than doubled.

This documents why **>80% is not reachable under the stated hard constraints**
(no source changes, add-only tests, CPU-only / no GPU) and what each blocker
would require.

## How we got from 30.62% → 67.92%

The pure/IO/config layer was already at 95–100% (prior Batches A–F). The jump
here came from the **CPU-only deterministic codegen-emit harness** (`_codegen/`):
drive real `KernelWriterAssembly.getSourceFileString` over curated logic files
across 7 arch families × dtypes × features (DTL/DTV/I8/GSU/F4-MX/LSU/StreamK) +
helper-kernel emit (BetaOnly/Conversion/Modules) + a Solution-derivation sweep.

## The three hard blockers to 80%

### 1. In-process codegen-emit is capped (~68%) by non-resettable rocisa state
The remaining bulk is `KernelWriterAssembly.py` (4006 missing) +
`KernelWriter.py` (1894) + the codegen `Components/*` (~3500). These ARE covered
by emit, but **total coverage cannot grow past ~P3 without breaking a
pre-existing test**: `test_SubtileBasedLogicalScheduler::test_emitLoops_256x256_fp4`.

- Root cause: the rocIsa singleton accumulates internal MMA-emit state that
  **survives `ri.init()`** (verified: the harness already re-inits per kernel)
  and is **not resettable from Python** (no reset in the rocIsa API surface).
  It grows with total kernels emitted in the process.
- The victim test only re-inits rocIsa `if not ri.isInit()` (line 227), so once
  our emits initialize the singleton it inherits our accumulated state.
- Adding more emit suites (the "broad" sweep + in-process `run()` end-to-end)
  pushed the accumulated-kernel count over the threshold and broke fp4; they were
  reverted to keep the suite green. So **more codegen configs cannot be added**
  in-process.
- Lifting this needs either a rocisa **source change** (a reset hook) — forbidden
  here — or per-test **process isolation** (`pytest-forked`), which is not
  installed and whose subprocesses would not count toward in-process coverage.

### 2. Benchmark/GPU-data-dependent code is unreachable CPU-only
- `LibraryLogic.py` (874 missing): `LogicAnalyzer` / `analyzeProblemType` /
  `generateLogic` parse **per-problem benchmark CSV** produced by running kernels
  on a GPU. No CSV fixtures exist in-tree; driving them needs fabricated
  benchmark data (large, brittle) or a real GPU run.
- `ClientWriter` run paths, `TensileCreateLibrary/Run` benchmark paths,
  `BenchmarkProblems`/`BenchmarkProcess` execution, and the GPU unit tests
  (already `skipif` gfx950) require a GPU.

### 3. Feature-gated codegen branches have no valid logic-file inputs
Part of the KWA/KW tail is reached only by parameter combinations the autotuner
**never selects** (so no tuned logic file expresses them) and which can't be
hand-built into a self-consistent `Solution` without the derivation rejecting
them. These need the record/replay capture harness described in `info.md`
(instrument a real GPU run to capture valid codegen inputs) — out of CPU-only
scope.

## What is still cheaply reachable (would add a few points, not 12)
- `ClientWriter.writeClientConfig*` (270) via constructed configs.
- `Activation.py` (322) per-function asm via direct `Activation` API calls.
- Residual CLI-tool pure surface (`TensileMergeLibrary` 133, etc.).
These are bounded (~+1–2%) and do not close the gap to 80%.

## Conclusion
Under the goal's constraints, the **achievable ceiling is ~68–70%**. Reaching
**>80%** requires relaxing one constraint:
- (a) allow a minimal **source change** (a rocisa emit-state reset hook, or a
  coverage-only test shim) to lift the codegen cap; or
- (b) provide a **GPU** (or fabricated benchmark CSVs) to cover LibraryLogic +
  client/benchmark execution; or
- (c) enable **per-test process isolation + subprocess coverage** in CI config.

Recommendation: accept the documented ceiling with this resistance writeup, OR
pick one of (a)/(b)/(c) to proceed toward 80%.
