#7 -- Characterization tests for tensilelite (rocm-libraries)

GOAL (tensilelite characterization tests):
  The Python code under
  rocm-libraries/projects/hipblaslt/tensilelite gains a
  characterization-test suite that pins the current
  behaviour of a chosen module with >=95% line coverage on
  that module, built using a tool chosen from a documented
  survey and integrated with tensilelite's existing pytest
  setup (pytest.ini / tox.ini). Functions that resisted
  characterization are documented with workarounds, and a
  recommendations doc states whether to scale the pattern
  to the rest of tensilelite, with a go / no-go.

  Goal achieved when:
    - survey.md compares characterization-test tooling
      (approvaltests, syrupy, pytest-snapshot, pytest-
      recording, plain golden-file). Comparison table:
      setup cost, refactor-tolerance, determinism handling,
      diff UX, fit with tensilelite's existing pytest/tox.
      One tool picked with rationale.
    - target.md names the chosen tensilelite module +
      rationale (pure-ish, low existing coverage, plausibly
      about to be refactored).
    - projects/hipblaslt/tensilelite/tests/characterization/
      <target>/ exists with snapshots driving every public
      function of the target with representative inputs.
    - Determinism handled: RNG seeded, time frozen, and any
      generated-output snapshot normalises absolute paths,
      timestamps, build IDs, and nondeterministic ordering
      (Tensile emits kernel source / solution libraries --
      snapshot the structured form, e.g. solution dict /
      YAML, not raw build blobs, wherever feasible).
    - pytest --cov of the target module reports >=95% on
      that module.
    - resistance.md lists every function that resisted with
      reason + workaround used.
    - recommendations.md states go / no-go on scaling to the
      rest of tensilelite + effort estimate per future
      target.
    - No regression: tensilelite's pre-existing test suite
      passes as before (same pytest invocation, same result).

SCOPE:
  - Worktree: this directory
    (.claude/worktrees/tensilelite-coverage off
    rocm-libraries `develop`).
  - Default tool pick: syrupy unless survey reveals strong
    reason otherwise (syrupy is a pytest plugin -- fits the
    existing pytest.ini).
  - Target candidate pool: top-level generators
    (AMaxGenerator.py, LayerNormGenerator.py,
    SoftmaxGenerator.py, ExtOpCreateLibrary.py) and modules
    under Tensile/. Pick ONE module for the first suite;
    further modules are follow-ups.
  - Only test files are added under the component; non-test
    tensilelite source is not modified.
  - Run inside tensilelite's own Python env
    (requirements-dev.txt / uv.lock / tox), not rocMETRICS's.

CONSTRAINTS:
  - rocm-libraries boundary: never recursive-list the
    monorepo root; navigate only within
    projects/hipblaslt/tensilelite. Do not modify or
    propose changes to non-test source.
  - Snapshot generated output in structured/normalised form,
    never raw nondeterministic blobs.
  - Seed RNG / freeze time as the target module needs.
  - Push / PR: do NOT push or create PRs. Commit locally
    only.

BOUND: stop after 60 turns or 180 minutes.

AUTONOMY: do not ask for confirmation. If the chosen target
  depends on global mutable state, document in
  resistance.md (snapshot the state, not just the output)
  and continue. If tensilelite's test env cannot be
  provisioned, write
  work/tensilelite-characterization/ENV_BLOCKED.md with the
  failure and stop.

---

## ANALYSIS: MODULE MAP (for subset-based characterization)

Scope of this map: `projects/hipblaslt/tensilelite` — the top-level
`*Generator.py` files plus the `Tensile/` Python package. The C++
under `src/`, `client/`, `include/` and the existing C++ gtests under
`tests/` are out of scope. `rocisa/` is a separate native package with
its own pytest/tox and is treated as a dependency, not a target.

### Two axes dominate testability (read this first)

1. **rocisa native-extension dependency.** `rocisa` is a nanobind/C++
   extension built via scikit-build-core + CMake (needs the HIP/LLVM
   toolchain). In a bare `python3` it is NOT importable
   (`from rocisa import rocIsa` fails). `requirements-dev.txt` installs
   `./rocisa`, so **building rocisa is the env gate for almost
   everything.** Crucially the binding constraint is usually the
   *package `__init__` chain*, not the leaf file:
   `Tensile/Common/__init__.py` does `from .Utilities import *`, and
   `Utilities` imports rocisa — so even rocisa-free leaves like
   `Common/Types.py` cannot be imported via `import Tensile.Common.Types`
   without rocisa. (They CAN be loaded by file path, bypassing the
   package init — a documented workaround, but fragile.)
   Practical reading: provision the dev env (build rocisa) first; then
   the rocisa dependency stops being a target-selection axis.

2. **External side-effects** beyond rocisa: subprocess/compiler+assembler
   invocation (Toolchain), GPU/`hip-python` (client run paths),
   filesystem/YAML/msgpack IO, `subprocess` arch probing, multiprocessing
   (`joblib`/`Parallel`). These are the functions that "resist"
   characterization and need fakes/normalisation or land in resistance.md.

### Module groups

LOC ≈ source lines. "Tests" = pre-existing Python unit tests found under
`Tensile/Tests/unit`. Fit = suitability as an *early* characterization
target (★★★ easy/pure → ★ heavy I/O or GPU).

| # | Module group | Key files | ~LOC | Role | Ext deps | Existing tests | Fit |
|---|---|---|---|---|---|---|---|
| 1 | **Core types & params** | `Common/{Types,Constants,DataType,Architectures,Capabilities,GlobalParameters,ValidParameters,RequiredParameters,RegisterPool,Parallel,Utilities,TimingInstrumentation}.py`, `Configuration.py` | ~5.1k | datatypes, ISA tables, global param schema & validation, config | rocisa (Utilities/DataType/Arch/RegisterPool); `subprocess` (Architectures) | `Common/{test_Architectures,test_Utilities,test_HardwarePredicateSort}`, `test_Configuration` | ★★★ (Types/Constants/ValidParameters/RequiredParameters pure once rocisa built) |
| 2 | **Solution & problem model** | `SolutionStructs/{Problem,Solution,Naming,LdsPadding,Utilities}.py` | ~7.5k | solution/problem structs, naming, LDS padding | rocisa; heavy internal coupling | `test_LdsPadding`, `test_SolutionStructsUtilities` | ★★ (Naming/LdsPadding/Utilities pure-ish; Solution.py 5.2k is huge) |
| 3 | **Validators** | `SolutionStructs/Validators/{MatrixInstruction,MXScaleFormat,WorkGroup}.py` | ~0.6k | reject/validate MI, MX-scale, workgroup configs | rocisa.enum, DataType | `test_MatrixInstructionConversion`, `test_MXScale*` | ★★★ small, input→reject/accept, ideal char-test shape |
| 4 | **Kernel source / asm emitters** | `KernelWriter.py` (9.9k), `KernelWriterAssembly.py` (18.8k), `Asm{StoreState,AddressCalculation,Memory*}.py`, `KernelWriter{Conversion,Modules,BetaOnly,Reduction,Base,Activation*}.py`, `Activation.py`, `Component.py`, `Components/*` | ~45k | emit GPU kernel source/assembly | rocisa (core), deep state | scattered (`test_emitMfmaInstruction`, `test_gr_*`, Subtile tests) | ★ enormous + stateful; snapshot structured emit only, not first |
| 5 | **Library IO & logic** | `LibraryIO.py`, `LibraryLogic.py` (1.6k), `SolutionLibrary.py`, `SolutionSelectionLibrary.py`, `Properties.py`, `Contractions.py`, `Hardware.py` | ~4.4k | (de)serialise solution libs, logic selection | rocisa (LibraryIO), YAML/msgpack | none | ★★ LibraryIO round-trips are classic golden-file targets (normalise paths) |
| 6 | **Benchmark & client orchestration** | `BenchmarkProblems,BenchmarkStructs,BenchmarkSplitter.py`, `Client{Writer,Executable}.py`, `Tensile.py`, `ParallelExecution.py`, `GenerateSummations.py` | ~3.6k | drive benchmark runs, write client | subprocess, GPU, fs, joblib | `test_BenchmarkSplitter`, `test_ClientExecutable` | ★ orchestration/IO-heavy |
| 7 | **CLI / library lifecycle** | `TensileCreateLibrary/{Run,ParseArguments,__main__}.py` (Run 1.1k = primary build entry), `Tensile{MergeLibrary,RetuneLibrary,UpdateLibrary,LibLogicToYaml}.py`, `TensileBenchmark*.py`, `ExtOpCreateLibrary.py` | ~4.5k | end-to-end CLI entry points | fs, YAML/msgpack, subprocess | `test_TensileLibLogicToYaml`, `test_TensileBenchmarkCluster*` | ★★ for the YAML transforms; ★ for full CLI |
| 8 | **TensileLogic** | `TensileLogic/{Valid{ChipId,MatrixInstruction,WorkGroup,WorkGroupMappingXCC},KnownBugs,HandleCustomKernel,ParseArguments,Run}.py` | ~1k+ | chip/MI/workgroup validity, known-bugs table | YAML data files | `test_ValidChipId`, `test_KnownBugs` | ★★★ table-driven validators, very char-test friendly |
| 9 | **Toolchain** | `Toolchain/{Assembly,Source,Validators,Component,HelperKernelCache}.py` | ~1.1k | compiler/assembler discovery + invocation, helper-kernel cache | subprocess, fs | `test_helper_cache` | ★★ Validators/cache pure with fakes; Assembly/Source wrap subprocess |
| 10 | **Utilities (standalone)** | `Utilities/{merge,stats}.py`, `Utilities/Decorators/*`, `Utilities/tensile_generator/tensile_config_generator.py`, `Utilities/archive/merge_rocblas_yaml_files.py`, `CustomYamlLoader.py`, `EmbeddedData.py`, `CustomKernels.py`, `KernelHelperNaming.py` | ~2.5k | YAML merge, config generation, decorators, embedded data | YAML/msgpack, fs (all **rocisa-free**) | `test_EmbeddedData` | ★★★ best truly-standalone candidates; `CustomYamlLoader` imports clean in bare python |
| 11 | **Standalone op generators** (top-level) | `AMaxGenerator.py` (905), `LayerNormGenerator.py` (986), `SoftmaxGenerator.py` (757), `ExtOpCreateLibrary.py` (83) | ~2.7k | emit standalone AMax/LayerNorm/Softmax asm; ext-op msgpack lib | rocisa (first 3); msgpack/yaml only (ExtOp) | none | ★★ self-contained, single public entry each; snapshot emitted module structure |

### Recommended first-target shortlist (build ONE suite first)

Pick from the high-fit, self-contained, currently-untested set:

- **`SolutionStructs/Validators/` (group 3)** — small, pure
  input→reject/accept functions, untested as a unit family, exactly the
  shape characterization snapshots fit. Best ratio of coverage-per-effort.
- **`TensileLogic` validators (group 8)** — table-driven validity checks
  over chip-id / MI / workgroup; partial tests exist, easy to reach ≥95%.
- **`LibraryIO` (group 5)** — round-trip (de)serialise solution libs;
  classic golden-file target. Higher value (matches the goal's "snapshot
  the structured solution form") but needs path/timestamp normalisation.
- **One standalone generator, e.g. `AMaxGenerator.py` (group 11)** —
  single public entry, self-contained, snapshot the emitted `Module`
  structure (not raw asm text). Good demonstrator for "scale the pattern".

Default first pick: **`SolutionStructs/Validators/`** for the proof-of-
pattern suite (cheap ≥95%, clean snapshots), then evaluate **`LibraryIO`**
or **`AMaxGenerator`** as the higher-value second target in
recommendations.md.

### Sequencing principle for scaling

Tier A (do first): groups 3, 8, 10 — pure/table-driven, low blast radius.
Tier B (high value): groups 5, 1, 11 — structured-output snapshots, need
normalisation but central to Tensile's output contract.
Tier C (defer): groups 2, 7, 9 — large or I/O/subprocess-heavy; fake the
boundary or accept partial coverage.
Tier D (last / maybe never as characterization): groups 4, 6 — 45k LOC of
stateful emitters and GPU/subprocess orchestration; snapshot only narrow
structured slices, document the rest in resistance.md.

### Coverage accounting — does a subset = "all the source"?

In-scope total: **136 files, 101,683 LOC** (top-level `*.py` + `Tensile/`,
excluding `Tests/`). Per-group LOC (authoritative census):

| Group | Files | LOC | % of scope |
|---|---|---|---|
| 1 core types & params (incl. Configuration) | 14 | 5,117 | 5.0% |
| 2 solution & problem | 6 | 7,402 | 7.3% |
| 3 validators | 3 | 598 | 0.6% |
| 4 kernel/asm emit | 54 | 69,135 | **68.0%** |
| 5 library io & logic | 7 | 4,619 | 4.5% |
| 6 benchmark & client | 8 | 3,378 | 3.3% |
| 7 CLI/lifecycle (incl. TensileCreateLibrary) | 15 | 4,485 | 4.4% |
| 8 TensileLogic | 9 | 989 | 1.0% |
| 9 Toolchain | 6 | 1,138 | 1.1% |
| 10 utilities | 7 | 1,876 | 1.8% |
| 11 op-generators | 4 | 2,731 | 2.7% |
| init/build/tasks | 3 | 211 | 0.2% |

**Selection {3,8,5,10,4,6} = 88 files, 80,595 LOC = 79.3% of scope —
but that is misleading:** ~68 of those 79 points are group 4 (the
emitters), where ≥95% characterization coverage is least achievable. The
selection *omits the two most central modules*: group 1 (core type system,
everything depends on it) and group 2 (solution/problem model — Tensile's
behavior contract). Excluding group 4, the other five selected groups are
only ~11.5k LOC.

To approach whole-codebase coverage you would additionally need groups
**1, 2, 11, 7, 9** (~20.9k LOC). Note the goal's ≥95% bar is *per chosen
module*, not repo-wide, so full coverage is not the stated target — but if
it became the target, {3,8,5,10,4,6} is not sufficient.

### Env note (gates target selection) — DECIDED & VALIDATED

ENV STATUS: ✅ provisioned and validated end-to-end (see `env/`).
Image `tensilelite-char:dev` built; container `tl-char` runs with the
worktree mounted; `invoke rocisa` builds + imports OK (needs
`LD_LIBRARY_PATH=/opt/rocm/lib`, baked into the image); existing unit suite
= 1186 passed / 201 skipped (no-regression baseline); group-3 coverage
measurable via the include-glob invocation (see COVERAGE GOTCHA below).


- Base image: **`rocm-libs-bump-e3b:26531351157`** — ROCm/HIP toolchain +
  HIP headers + `hip-config.cmake` + cmake + Python 3.12; ROCM_PATH=/opt/rocm.
  Has the toolchain to BUILD rocisa, but **no Python deps (except yaml) and
  no rocisa**.
- Derived dev image: **`tensilelite-char:dev`** (see `env/Dockerfile` +
  `env/README.md`). Bakes only Python deps + build tools. rocisa is NOT
  baked because its CMake links the monorepo siblings `shared/origami` +
  `shared/stinkytofu` (at `rocisa/../../../../shared/`), absent from the base
  image — so rocisa is built once inside a worktree-mounted container via
  `invoke rocisa` (editable; build tree lives in the worktree, survives runs).
- Provision once, up front, then rocisa stops being a target-selection
  axis; the remaining resistance axis is external side-effects
  (subprocess/GPU/IO).
- Fallback only if the rocisa build fails in the image: the rocisa-free
  standalone files (group 10 + `ExtOpCreateLibrary`) — and record the
  failure in ENV_BLOCKED.md.

### Build & coverage tooling (verified against the tree)

**rocisa build** — native pkg (scikit-build-core + nanobind + CMake; needs
`ROCM_PATH`, default `/opt/rocm`). Four equivalent ways, pick one:
- `invoke rocisa` → editable build:
  `pip install --no-build-isolation -e rocisa/` with
  `CMAKE_ARGS=-DROCM_PATH=<rocm> -DROCISA_INCLUDE_BUILD_INFO=ON`.
  **Documented dev workflow**; C++ edits caught by a staleness check in
  `rocisa/__init__.py` (no reinstall, rebuild via cmake).
- `pip install ./rocisa` (what `requirements-dev.txt` and tox `[testenv]` do).
- `tox -e rocisa` (build + run rocisa's own tests under `rocisa/test`).
- `uv sync` (rocisa is a uv **workspace member**:
  `[tool.uv.sources] rocisa = {workspace=true}`).

**Deps** — `msgpack` is a DECLARED dep (in `requirements.txt` and
`pyproject` `dependencies`); it was only absent in the bare system python
probed during analysis. `pip install -r requirements.txt` (or `uv sync`)
provides it. `pip install ./rocisa` alone does NOT (rocisa has zero deps).
Recommended provisioning: `pip install -r requirements-dev.txt`
(= requirements.txt + `./rocisa` + memory_profiler + invoke), or `uv sync`.

**Coverage already wired up** (use it, don't reinvent):
- `pyproject.toml` `[tool.coverage.*]`: branch coverage on, omits
  `*/Tests/*`, `*/tests/*`, `test_*.py`, conftest; html/xml/json reports.
- tox envs: `coverage` (unit+common, builds client), `coverage-unit`
  (unit only, no GPU/client), `coverage-common` (needs client + GPU).
  These use `--cov=Tensile --cov=rocisa` and run from the tensilelite dir,
  where `Tensile/` and `rocisa/` both exist as directories → coverage uses
  PATH mode (no import) → they DO NOT abort. VERIFIED rc=0. No tox change
  needed; the characterization runs follow the same path-mode rule.

⚠️ **COVERAGE GOTCHA (root-caused): pass `--cov` a value that resolves to
an existing directory; do NOT pass an importable dotted name that has no
matching path.** Coverage resolves each `--cov`/`source` entry as a
filesystem path IF a matching file/dir exists relative to the run cwd;
otherwise it IMPORTS it as a module to discover its files. Importing a
rocisa-touching module re-initializes the `_rocisa` nanobind extension →
`Critical nanobind error: refusing to add duplicate key ... → abort()`
(SIGABRT, core dump, empty output).
- ✅ `--cov=Tensile/SolutionStructs/Validators` (dir exists) → path mode, OK
- ✅ `--cov=Tensile` / `--cov=rocisa` from tensilelite dir (dirs exist) → OK
- ❌ `--cov=Tensile.SolutionStructs.Validators` (no such path) → import → abort
Reproduced with editable AND non-editable installs, with/without `-n`
xdist, every COVERAGE_CORE — so it is the import path, not a build artifact.

RESOLUTION — canonical group-3 invocation (run from the tensilelite dir;
the `--cov` value is the source DIR path so it resolves in path mode; reuses
`pyproject.toml`; no GPU/client; VERIFIED rc=0, 1186 passed / 201 skipped =
no regression):
```
pytest -m unit \
  --cov=Tensile/SolutionStructs/Validators \
  --cov-config=pyproject.toml --cov-report=term-missing \
  Tensile/Tests/unit
```

Baseline coverage from the EXISTING tests (pre-characterization):
WorkGroup 100%, MXScaleFormat 92%, MatrixInstruction 69%, TOTAL 78% —
characterization tests close the MatrixInstruction gap to reach ≥95%.

**Test collection / location — DECIDED: option (b).**
Suite lives at **`Tensile/Tests/unit/characterization/Validators/`**, marked
`-m unit`. Rationale: it sits under the existing collected `testpaths`
(`Tensile/Tests`), so the default `pytest` invocation and the existing
`coverage-unit`/`unit` tox flows pick it up with zero config change —
satisfying both "≥95% via `pytest --cov`" and "no regression / same
invocation". (The literal goal path `tests/characterization/` was rejected:
not on `testpaths`, co-located with C++ gtests, not collected by default.)
Note the deviation from the goal's stated path in target.md.

**GPU/client NOT required for group 3** — only the `common`/full
`[testenv]` paths call `invoke get-gpu-arch` + `invoke build-client`
(C++ `tensilelite-client`). Unit-level characterization of Validators
runs CPU-only once rocisa is built.

### Remaining source — files OUTSIDE the initial 79% target

Initial target = groups {3, 8, 5, 10, 4, 6} (88 files, 80,595 LOC). The
48 files below (21,047 LOC) are deferred for now but every one is assigned
a module so nothing is unaccounted. These become follow-up suites if/when
characterization scales past the initial 79%. (See per-group LOC table
above for the rollup; sequencing tier in parens.)

**Module 1 — Core types & params** (14 files, 5,117 LOC; Tier B)
```
   549  Tensile/Common/Architectures.py        (subprocess arch probe)
    54  Tensile/Common/Capabilities.py
    84  Tensile/Common/Constants.py
   492  Tensile/Common/DataType.py
   767  Tensile/Common/GlobalParameters.py
   251  Tensile/Common/Parallel.py              (multiprocessing)
   123  Tensile/Common/RegisterPool.py
   132  Tensile/Common/RequiredParameters.py
    52  Tensile/Common/TimingInstrumentation.py
   117  Tensile/Common/Types.py
   415  Tensile/Common/Utilities.py             (rocisa entry: from rocisa import rocIsa)
  1136  Tensile/Common/ValidParameters.py
     4  Tensile/Common/__init__.py
   941  Tensile/Configuration.py
```

**Module 2 — Solution & problem model** (6 files, 7,402 LOC; Tier C)
```
   412  Tensile/SolutionStructs/LdsPadding.py
   239  Tensile/SolutionStructs/Naming.py
  1382  Tensile/SolutionStructs/Problem.py
  5230  Tensile/SolutionStructs/Solution.py     (largest single module)
   113  Tensile/SolutionStructs/Utilities.py
    26  Tensile/SolutionStructs/__init__.py
```

**Module 7 — CLI / library lifecycle** (16 files, 4,490 LOC; Tier C,
except YAML transforms which are Tier B)
```
   106  Tensile/CustomKernels.py
   155  Tensile/CustomYamlLoader.py             (rocisa-free, yaml only)
   211  Tensile/EmbeddedData.py                 (has test_EmbeddedData)
   486  Tensile/TensileBenchmarkCluster.py
   396  Tensile/TensileBenchmarkClusterScripts.py
   183  Tensile/TensileBenchmarkLibraryClient.py
   188  Tensile/TensileClientConfig.py
   229  Tensile/TensileCreateLibrary/ParseArguments.py
  1086  Tensile/TensileCreateLibrary/Run.py     (primary build entry point)
     1  Tensile/TensileCreateLibrary/__init__.py
     4  Tensile/TensileCreateLibrary/__main__.py
   458  Tensile/TensileLibLogicToYaml.py        (Tier B: YAML transform, has test)
   373  Tensile/TensileMergeLibrary.py
   236  Tensile/TensileRetuneLibrary.py
   165  Tensile/TensileUpdateLibrary.py
   213  Tensile/verify_stinky_comment_vs_elf_text.py
```

**Module 9 — Toolchain** (6 files, 1,138 LOC; Tier C)
```
   104  Tensile/Toolchain/Assembly.py           (subprocess: assembler)
   431  Tensile/Toolchain/Component.py
   161  Tensile/Toolchain/HelperKernelCache.py  (has test_helper_cache)
   146  Tensile/Toolchain/Source.py             (subprocess: compiler)
   273  Tensile/Toolchain/Validators.py         (toolchain discovery)
    23  Tensile/Toolchain/__init__.py
```

**Module 11 — Standalone op generators** (4 files, 2,731 LOC; Tier B)
```
   905  AMaxGenerator.py                        (rocisa)
   986  LayerNormGenerator.py                   (rocisa)
   757  SoftmaxGenerator.py                     (rocisa)
    83  ExtOpCreateLibrary.py                   (rocisa-free; msgpack/yaml)
```

**Module 0 — package init / build tooling** (2 files, 210 LOC; not a
characterization target)
```
    39  Tensile/__init__.py
   171  tasks.py                                (invoke build tasks)
```

Coverage check: 88 (selected) + 48 (above) = 136 files = full in-scope
Python source. Nothing is unassigned.
