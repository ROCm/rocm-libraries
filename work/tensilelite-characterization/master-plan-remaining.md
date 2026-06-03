# Master plan — characterize every remaining realistic module

Single self-directed plan to finish the realistically-characterizable Python
surface of `projects/hipblaslt/tensilelite/Tensile`. Replaces the
one-module-per-session handoff: work top-to-bottom, **commit per module**, never
batch into one massive commit.

Baseline snapshot (current full `-m unit` = **1844 passed / 201 skipped**) saved
in `coverage/master-baseline-1844.txt` — used as each module's BEFORE row.

## Scope

**IN scope** — pure / table / IO / config / toolchain-helper Python modules
listed in "Work queue" below.

**OUT of scope** — the codegen / asm-emit / GPU surface (≈38k stmts): all
`KernelWriter*`, `Components/*`, `Asm*`, `GenerateSummations`, `verify_stinky*`,
`ClientWriter`/client-run. These emit GPU assembly and are not unit-testable
without a full build / GPU (rated ★ lowest fit in the original MODULE MAP).
Also **deferred**: `SolutionStructs/Solution.py` slice 3b (the derivation
config sweep — multi-day; see its `recommendations.md`).

## Per-module protocol (streamlined; keeps integrity, avoids 40× full runs)

For each module M (new dir `Tests/unit/characterization/<Suite>/`):
1. **BEFORE**: take M's row from `coverage/master-baseline-*.txt` (no new full
   run per module). Record in the module's `coverage-before.txt`.
2. Write `conftest.py` (if needed) + `test_*_char.py`; generate snapshots
   in-container (`--snapshot-update`).
3. **Module coverage (fast)**: `pytest --cov=Tensile ... <SuiteDir>` and grep
   M's row; iterate until **≥95% line** (or document genuinely-unreachable lines
   in `resistance.md`).
4. Write `target.md` + `resistance.md` (+ `recommendations.md` only on the final
   module / when the next target changes).
5. **Commit the module** (atomic: baseline+target, then tests+snapshots, then
   coverage-after+docs — or fewer commits for tiny modules). Per-module commit,
   never batched.
6. Update the **Work queue checklist** below in this file and commit it.

Per **batch** (every ~4-6 modules, and at the very end):
- Run the full `-m unit --cov=Tensile` **once**: confirm **no regression**
  (pass count only grows; 201 skipped unchanged) and capture a fresh
  `coverage/master-baseline-<N>.txt`. Commit the checkpoint.

Hard rules (unchanged): **ADD-ONLY**; **never push**; stay within
`projects/hipblaslt/tensilelite`; snapshots written in-container (root-owned);
`--cov` takes the PATH `Tensile` (never a dotted module → rocisa SIGABRT);
`importlib.import_module` for any `SolutionStructs` submodule shadowed by the
package `__init__`.

## Work queue (priority order: cheap/pure → mid → heavy). Check off as done.

### Batch A — tiny / near-done top-ups  ✅ DONE (full -m unit = 1890 passed)
- [x] `Common/Types.py` 81.7% → 100%
- [x] `Properties.py` 80.7% → 100%
- [x] `Common/TimingInstrumentation.py` 76.2% → 100% line (96% blended)
- [x] `Utilities/Decorators/{Shared,Timing,Profile}.py` → 100% (one suite)
- [x] `Tensile/__init__.py` 88.9% → 100%
- [x] `Toolchain/Assembly.py` 93.6% → 95.24% line (L82 dead)
- [x] `Toolchain/HelperKernelCache.py` 90.8% → 99.10% line

### Batch B — small pure  ✅ DONE (full -m unit = 1974 passed)
- [x] `Common/RegisterPool.py` 35.1% → 100%
- [x] `Common/Parallel.py` 39.5% → ~80.6% (fork paths out of scope, D4)
- [x] `Component.py` 39.5% → 100%
- [x] `CustomKernels.py` 24.5% → 100%
- [x] `CustomYamlLoader.py` 59.0% → 97.4%
- [x] `KernelHelperNaming.py` → naming half covered; init* = codegen (D6)
- [x] `Common/GlobalParameters.py` 90.0% → 99.1%

### Batch C — mid pure / near-done  ✅ DONE (full -m unit = 2192 passed)
- [x] `Common/Architectures.py` 83.2% → 97.1%
- [x] `Hardware.py` 88.2% → 97.6%
- [x] `Common/Utilities.py` 53.4% → 95.6%
- [x] `Toolchain/Source.py` 50.0% → 97.6%
- [x] `Configuration.py` 86.6% → ~92% (dead reflected ops + Attribute AST paths, D9)
- [x] `Contractions.py` 84.2% → ~86% (predicate matrix, D10)
- [x] `BenchmarkStructs.py` 72.4% → (helpers+fork-perms+BenchmarkStep; BenchmarkProcess deferred, D11)
- [x] `Toolchain/Component.py` 70.1% → 100% line · `Toolchain/Validators.py` 67.7% → 100% line
- [x] `TensileBenchmarkCluster.py` 87.5% → 99.51% (pinned --results-only BoolOp bug, D12)

### Batch D — mid, IO/pipeline-leaning  ✅ DONE (full -m unit = 2337 passed)
- [x] `SolutionLibrary.py` 55.2% → 97.1% line
- [x] `SolutionSelectionLibrary.py` 7.3% → 100% line
- [x] `Utilities/merge.py` 15.3% → 97.8% line
- [x] `Activation.py` 16.8% → 34.1% line (pure layer; asm codegen out of scope, D13)
- [x] `TensileCreateLibrary/ParseArguments.py` 12.3% → 100% line

### Batch E — heavy / lower-yield (partial + documented where it resists)
- [ ] `BenchmarkProblems.py` (366, 59.8%)
- [ ] `ParallelExecution.py` (124, 10.5%)
- [ ] `Tensile.py` (367, 43.1%)
- [ ] `TensileCreateLibrary/Run.py` (529, 35.5%)
- [ ] `LibraryLogic.py` (944, 5.1%)
- [ ] `TensileLibLogicToYaml.py` (199, 16.6%)

### Batch F — entry-point CLI scripts (stretch; pure helpers only, document the
###            argv/fs/subprocess-bound `main()` as resistance)
- [ ] `TensileMergeLibrary.py` (255, 0%) · `TensileRetuneLibrary.py` (130, 0%)
- [ ] `TensileUpdateLibrary.py` (97, 0%) · `TensileClientConfig.py` (98, 0%)
- [ ] `TensileBenchmarkLibraryClient.py` (114, 0%) · `TensileCreateLibrary/__main__.py` (1, 0%)

### Deferred (documented, not in this plan)
- `SolutionStructs/Solution.py` slice 3b (derivation config sweep).
- All codegen/asm/GPU modules (out of scope).

## Progress log
(append one line per completed module: `<module> — before% -> after% (N tests), commit <sha>`)

- Batch A complete — 7 modules / 9 source files, +46 tests, full -m unit 1844 → 1890 passed (no regression). Fresh baseline: coverage/master-baseline-1890.txt.
  - Common/Types 81.7→100 · Properties 80.7→100 · TimingInstrumentation 76.2→100 ·
    Decorators(Shared/Timing/Profile)→100 · __init__ 88.9→100 ·
    Toolchain/Assembly 93.6→95.2 · Toolchain/HelperKernelCache 90.8→99.1
- Batch B complete — 7 modules, +84 tests, full -m unit 1890 → 1974 passed (no regression). Fresh baseline: coverage/master-baseline-1974.txt.
  - RegisterPool 35.1→100 · CustomKernels 24.5→100 · CustomYamlLoader 59→97.4 ·
    Component 39.5→100 · GlobalParameters 90→99.1 · Parallel 39.5→80.6 (D4) ·
    KernelHelperNaming naming-half (D6). Caught + fixed 2 cross-suite state leaks
    (CustomKernels validParameters.update; Parallel n_jobs=1 self-clears globalParameters).
- Batch C complete — 9 modules, full -m unit 1974 → 2192 passed (no regression). Fresh baseline: coverage/master-baseline-2192.txt.
  - Architectures 83→97 · Hardware 88→98 · Common/Utilities 53→96 · Toolchain/Source 50→98 ·
    Configuration 87→92 (D9) · Contractions 84→86 (D10) · BenchmarkStructs 72→partial (D11) ·
    Toolchain/Validators 68→100 · Toolchain/Component 70→100 · TensileBenchmarkCluster 88→99.5 (D12).
    Pinned latent bug: ExpressionEvaluator BoolOp ignores 3rd+ `or` operand (D12).
- Batch D complete — 5 modules, full -m unit 2192 → 2337 passed (no regression). Fresh baseline: coverage/master-baseline-2337.txt.
  - SolutionSelectionLibrary 7→100 · ParseArguments 12→100 · merge 15→97.8 ·
    SolutionLibrary 55→97.1 · Activation 17→34 (pure layer; asm out of scope, D13).
