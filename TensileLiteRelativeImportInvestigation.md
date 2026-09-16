# TensileLite absolute-to-relative import investigation

Status: investigated against `develop` on 2026-09-15, then implemented as the local `users/alvasile/AIHPBLAS-3989-namespace-split/*` stack based on local `develop` at `b607a8ae0cfda0a27a0f6a1446cb3043a88c22c4`. The stack has not been pushed.

## Conclusion

It is practical to convert imports inside the `Tensile` Python package from absolute `Tensile...` imports to explicit relative imports before the namespace move. That should be a preparatory PR ahead of the physical `Tensile/` to `tensilelite/` rename. It would remove most of the package-internal import edits from the later rename and make the moved implementation substantially less dependent on its top-level package spelling.

The conversion cannot sensibly apply to every absolute import in the repository. Tests, GEKO, `tasks.py`, standalone generators, and the scripts under `Tensile/bin/` are consumers or process entry points rather than package-internal modules. They need a public package name, a console command, or an explicit `python -m ...` target. Leading-dot imports have no package context when those files are run directly.

Relative imports also do not solve the dual-namespace identity requirement in the namespace-split proposal. A probe that exposed one relative-import implementation tree under both `Tensile` and `tensilelite` still produced two distinct `Tensile.Common.GlobalParameters` module objects and two distinct `globalParameters` objects. The bridge described in `TensileLiteNamespaceSplitProposal.md:56-88` must still alias descendant modules to one canonical object; a symlink, two package directories, or a root-only alias is insufficient.

The recommendation is therefore:

1. Repair the three tests that load package modules as anonymous files.
2. Convert package-internal imports to relative imports while the package is still named `Tensile`.
3. Keep direct launchers and external consumers on absolute imports through that preparatory PR.
4. Move the package directory and activate a one-way `Tensile` to `tensilelite` namespace bridge.
5. Migrate tests, GEKO, standalone tools, and command surfaces as external consumers.
6. Remove the compatibility namespace only after both import spellings no longer need to coexist.

## Current import topology

An AST scan of `projects/hipblaslt/tensilelite/Tensile/**/*.py`, excluding `Tensile/Tests`, found:

| Import form | Static import statements | Files |
| --- | ---: | ---: |
| Absolute imports rooted at `Tensile` | 240 | 60 |
| Explicit relative imports | 305 | 97 |

The package already uses both forms in the same modules. For example, `Tensile/ClientWriter.py:35-49` imports some dependencies through `Tensile...` and others through `.TensileCreateLibrary`, `.ParallelExecution`, and `.Contractions`. `Tensile/TensileCreateLibrary/Run.py:39-89` similarly uses absolute imports for the package root and a relative import for its sibling `ParseArguments` module. `Tensile/CustomKernels.py:25-26` has one relative and one absolute package import on adjacent lines. There is no lint or packaging rule in `pyproject.toml`, `tox.ini`, or the applicable `AGENTS.md` files that prohibits relative imports.

The proposed conversions are ordinary package-relative mappings:

```python
# Tensile/ClientWriter.py
from Tensile.Common import ensurePath
from Tensile.SolutionStructs.Problem import ProblemType

# becomes
from .Common import ensurePath
from .SolutionStructs.Problem import ProblemType
```

```python
# Tensile/TensileCreateLibrary/Run.py
from Tensile.Common import CHeader

# becomes
from ..Common import CHeader
```

```python
# Tensile/Components/CustomSchedule.py:45
import Tensile.Components.CMSValidator as cmsv

# becomes
from . import CMSValidator as cmsv
```

This changes how a module reaches another module in the same package. When the implementation is imported as `Tensile.*`, the resulting module names and objects remain `Tensile.*`, so existing tests that patch `Tensile.KernelWriterAssembly.rocIsa` or import `Tensile.TensileRetuneLibrary` do not need to change in the preparatory PR.

There is also historical precedent in this codebase. Commit `4b92c073bfc` (`Relative imports for python 3`) converted the earlier generator from bare sibling imports such as `from Common import ...` to package-relative imports such as `from .Common import ...`. The current tree has since accumulated many package-root absolute imports, but relative package loading is not a new execution model for Tensile.

## Test implications

The tests should not be converted wholesale to relative imports. They exercise the package as consumers, and their collection layout is not a package hierarchy:

- The AST scan found 1,060 static absolute `Tensile...` import statements in 286 test files and no relative import statements in the test tree.
- 149 of the 155 non-cache test directories that contain Python files have no `__init__.py`. Pytest therefore commonly collects these as top-level test modules, where leading-dot imports have no parent package.
- `Tensile/Tests/conftest.py:36-39` adds the TensileLite source root to `sys.path`, making `Tensile` importable as a public package. Common tests forward that import environment to clean subprocesses at `Tensile/Tests/common/test_config.py:68-92`.
- Many tests patch or import by dotted public name. A call-site scan found 565 dotted `Tensile...` string uses in 81 test files. Those names continue to work unchanged during a package-internal relative-import PR because the canonical package name is still `Tensile`; they belong to the later public namespace migration.

Three tests are direct blockers because they deliberately bypass package importing:

| Test | Current behavior | Why relative imports fail |
| --- | --- | --- |
| `Tensile/Tests/unit/test_KnownBugs.py:14-24` | Loads `TensileLogic/KnownBugs.py` with `spec_from_file_location("KnownBugs_under_test", ...)`. | The loaded module has no package parent, so `from ..resources import ...` raises `ImportError: attempted relative import with no known parent package`. |
| `Tensile/Tests/unit/test_valid_corpus_consistency.py:25-34` | Loads `ValidCorpusConsistency.py` as `ValidCorpusConsistency_under_test`. | A relative import of `CustomYamlLoader` cannot resolve from an anonymous top-level module. |
| `Tensile/Tests/unit/test_ValidChipId.py:33-41` | Loads `ValidChipId.py` as `ValidChipId_under_test`. | A relative import of `Common.Architectures` cannot resolve from an anonymous top-level module. The same test already acknowledges this rule for `Architectures.py` at lines 81-88 and imports that module normally because it has relative imports. |

These tests use file loading to avoid `Tensile/TensileLogic/__init__.py`, whose only line eagerly imports `.Run` and therefore pulls in heavier dependencies. Before converting these three modules, either make `TensileLogic.__init__` lazy enough for normal submodule imports or give each file-loaded test module a real package-qualified name and package context. Keeping these three production imports absolute temporarily is possible, but it leaves exactly the name coupling this preparatory work is intended to remove.

`Tensile/Tensile.py` is not a blocker. It deliberately exits before its imports when executed as a file (`Tensile/Tensile.py:25-27`), and `Tensile/Tests/unit/characterization/PublicInputSurface/test_pchaos_Tensile_L25_char.py:53-81` pins that behavior. Its supported operational path is the `Tensile/bin/Tensile` launcher. `Tensile/ExperimentalLibrary.py` documents `python -m Tensile.ExperimentalLibrary` at lines 66-97, so package-relative imports fit its documented execution mode; directly executing that `.py` file would become unsupported unless a shim is retained.

## `Tensile/bin` audit

All nine tracked entries under `projects/hipblaslt/tensilelite/Tensile/bin/` are extensionless Python scripts. None is part of a Python package: `Tensile/bin/` has no `__init__.py`, and the scripts are executed by filesystem path. Every script first attempts an absolute `Tensile...` import and, on `ImportError`, appends the directory two levels above the script to `sys.path` before retrying. A leading-dot import is therefore invalid in these files.

| Entry | Imported target | Bootstrap lines | Current in-repo use |
| --- | --- | --- | --- |
| `Tensile/bin/Tensile` | `Tensile.Tensile` | `27-35` | Primary documented source launcher (`tensilelite/README.md:93-94`, `AGENTS_reference.md:17-18`); installed by hipBLASLt CMake (`CMakeLists.txt:744-750`). |
| `Tensile/bin/TensileBenchmarkCluster` | `Tensile.TensileBenchmarkCluster` | `27-35` | Legacy source launcher; the installed console entry point is also declared at `pyproject.toml:99`. |
| `Tensile/bin/TensileCreateLibrary` | `Tensile.TensileCreateLibrary` | `29-38` | Installed by hipBLASLt CMake (`CMakeLists.txt:744-750`), called by `ClientWriter.py:317-335`, GEKO, and test configuration examples. |
| `Tensile/bin/TensileGenerateSummations` | `Tensile.GenerateSummations` | `27-35` | Referenced by the deprecated auto-build wrapper machinery. |
| `Tensile/bin/TensileLibLogicToYaml` | `Tensile.TensileLibLogicToYaml` | `27-35` | Legacy source launcher; current package code also calls the module API. |
| `Tensile/bin/TensileLogic` | `Tensile.TensileLogic` | `29-38` | Directly executed by the hipBLASLt CMake validation rule at `cmake/hipblaslt_codegen.cmake:310-319`. |
| `Tensile/bin/TensileMergeLibrary` | `Tensile.TensileMergeLibrary` | `29-38` | Called by current GEKO at `utilities/geko/geko/library/operations.py:317-326` and documented at `utilities/geko/README.md:586-596`. |
| `Tensile/bin/TensileRetuneLibrary` | `Tensile.TensileRetuneLibrary` | `27-35` | Legacy source launcher; the installed console entry point is declared at `pyproject.toml:100`. |
| `Tensile/bin/TensileUpdateLibrary` | `Tensile.TensileUpdateLibrary` | `29-37` | Legacy source launcher and auto-build wrapper input. |

The launchers can remain absolute boundary adapters while everything they import becomes relative internally. At the namespace move they should import the canonical lowercase package absolutely, or be retired in favor of installed console scripts and `python -m tensilelite ...`. The existing package metadata already declares six console entry points at `pyproject.toml:94-100`, while `MANIFEST.in:4` and the hipBLASLt test-component install currently ship only `Tensile` and `TensileCreateLibrary` from this legacy directory.

There is one additional tracked `bin` launcher in the examined consumer area: `utilities/geko/bin/geko`. It inserts the GEKO package root at the front of `sys.path` at lines 9-15 and then imports `geko.cli`. That is GEKO's own source-checkout bootstrap and is independent of whether TensileLite internals use relative imports.

## Other path and process boundaries

Several non-`bin` files also establish package lookup or launch child processes. They should be treated as external boundaries rather than converted to relative imports:

- `tensilelite/tasks.py:13-25` inserts the TensileLite source root and imports `Tensile.RocisaStatus` and `Tensile.GpuRevisionTarget`. `tasks.py` is beside the package, not inside it.
- `hipblaslt/scripts/run_tensile_logic_check.py:29-73` adds the built rocisa location and the TensileLite root to `sys.path`, then imports `Tensile.TensileLogic` at lines 126-130.
- `AMaxGenerator.py:38-43`, `LayerNormGenerator.py:39-44`, and `SoftmaxGenerator.py:41-47` are standalone files beside the package. Their imports must remain public absolute imports until those tools move inside the package or migrate to the new public name.
- `Tensile/ClientWriter.py:145-156` modifies `PYTHONPATH` for a subprocess so rocisa can be found, while `ClientWriter.py:317-335` constructs a direct `ROOT_PATH/bin/TensileCreateLibrary` command. Internal relative imports do not change either process boundary.
- CMake uses both modes: it directly executes `Tensile/bin/TensileLogic` at `cmake/hipblaslt_codegen.cmake:310-319`, but runs library creation in package mode with `python -m Tensile.TensileCreateLibrary` at lines 323-329. The latter is naturally compatible with relative imports; the former depends on the launcher's absolute bootstrap.

## GEKO impact

Current `develop` GEKO is a source-checkout consumer, not a package-relative participant. Three production files contain 9 static `Tensile...` import statements. Six production files participate in the broader import, path-injection, or direct-launcher contract. Representative paths are:

- `utilities/geko/geko/config_generator/fork_params/param_meta.py:20-26` lazily imports `Tensile.Common.GlobalParameters` and `Tensile.Common.ValidParameters`.
- `utilities/geko/geko/config_generator/fork_params/hw_profiles/gfx950/post_processor.py:207-235` first tries absolute imports, then walks parent directories looking for `tensilelite`, inserts that checkout directory into `sys.path`, and retries.
- `utilities/geko/geko/library/operations.py:369-397` optionally appends `<hipblaslt>/tensilelite` to `sys.path` before importing `Tensile.LibraryIO`, `Tensile.CustomYamlLoader`, and `Tensile.TensileMergeLibrary`.
- `utilities/geko/geko/config_generator/config_generator.py:27-60` unconditionally inserts `<hipblaslt>/tensilelite` before generating configurations.
- `utilities/geko/geko/config_generator/output_writer.py:280-307` emits shell scripts that set `PYTHONPATH=<hipblaslt>/tensilelite` and execute `Tensile/bin/Tensile`.
- `utilities/geko/geko/optim/optim.py:205-213` documents the same source-tree contract and constructs both `PYTHONPATH` and a direct `Tensile/bin/Tensile` subprocess at lines 303-318.
- `utilities/geko/geko/library/operations.py:294-367` directly invokes `TensileMergeLibrary` and `TensileCreateLibrary` by checkout path.

GEKO's integration tests reinforce that contract. `utilities/geko/tests/conftest.py:125-148` inserts `<hipblaslt>/tensilelite` into `sys.path` for the test, and `utilities/geko/tox.ini:27-59` installs rocisa from the selected hipBLASLt checkout before passing that checkout to pytest. Tests also construct `tensilelite/Tensile/bin` paths (`test_optim_core_unit.py:66-75` and `267-275`) and install fake `Tensile...` modules in `sys.modules` (`test_library_operations_unit.py:584-617`, `test_config_generator_cross_module_unit.py:53-68` and `332-348`).

None of those GEKO call sites needs to change in the package-internal relative-import PR. They continue importing `Tensile`, and the imported modules resolve their own dependencies relatively. They must change when the public package name changes.

There is already useful implementation evidence in the unmerged namespace stack. Commit `54bc6ceb0b2` (`13-refactor(geko): use installed TensileLite interfaces`) removes GEKO's source-path injection, imports `tensilelite...` from the active environment, replaces direct merge/create launchers with Python API calls, and changes tuning execution to `python -m tensilelite run`. That commit demonstrates a viable GEKO end state, but it is not part of current `develop` and should not be treated as current behavior.

## Bridge and module identity

Converting internal imports to relative imports reduces the number of modules that spell the old root, but it does not cause two roots to share `sys.modules` entries. In an isolated converted tree, a simple `tensilelite -> Tensile` directory symlink produced:

```text
module_identity False
state_identity False
old_name Tensile.Common.GlobalParameters
new_name tensilelite.Common.GlobalParameters
```

This directly confirms the concern in `TensileLiteNamespaceSplitProposal.md:58-76`: Python keys modules by fully qualified import name. Relative imports keep each imported tree internally consistent, but importing the same files once through `Tensile` and once through `tensilelite` still executes them as different module trees. The migration bridge must alias every old descendant to the canonical new descendant, or prevent the second execution through an equivalent finder/loader design.

Relative imports do improve the bridge design after the physical move. Once code is loaded under the canonical namespace, its internal imports remain under that namespace rather than reaching back through a hard-coded old root. This narrows the bridge to compatibility requests from external callers and avoids package implementation code continuously crossing the alias boundary.

## Recommended migration sequence

### 1. Make the test harness package-aware

Update the three anonymous file-load tests first. Prefer normal package imports after making `TensileLogic/__init__.py` avoid its eager `.Run` import, or preserve isolated loading with a package-qualified spec name and an explicit package context. Add a focused test that verifies those three modules import normally from an installed/source package.

This change should land separately because it improves the test contract even without the rename: package modules are tested as package modules, including their real relative-import semantics.

### 2. Convert implementation imports while the root is still `Tensile`

Convert only Python files inside `Tensile/`, excluding `Tensile/Tests` and `Tensile/bin`. Include `TensileCreateLibrary/__main__.py`: CMake invokes it with `python -m Tensile.TensileCreateLibrary`, so `from . import run` is the correct package-local form. Preserve the direct-execution guard in `Tensile.py`, and explicitly decide whether direct execution of `ExperimentalLibrary.py` is supported; its documented interface is already `python -m`.

Run the full unit suite without rewriting unrelated source formatting. Keep all existing test imports and dotted patch targets on `Tensile...` in this PR. Their continued success proves the public namespace did not change.

### 3. Keep entry points as boundary adapters

Do not put relative imports in extensionless `Tensile/bin/*` files. Keep them importing `Tensile...` until the physical move, then change them atomically to the canonical lowercase package or replace their callers with installed console commands/`python -m`. Preserve at least `Tensile`, `TensileCreateLibrary`, and `TensileLogic` until their CMake and documented callers move.

### 4. Move the package and add a real compatibility bridge

With package internals already relative, most implementation files can move byte-for-byte. Update package discovery, resource keys, CMake paths, coverage paths, and launcher locations in the physical-move PR as the proposal already requires at `TensileLiteNamespaceSplitProposal.md:90-102`.

The bridge only maps legacy `Tensile.X` requests to canonical `tensilelite.X` modules. It still has to guarantee `Tensile.X is tensilelite.X` for descendants and shared state. Validate both import orders, nested imports, `importlib.resources`, console entry points, and `python -m` as required by the proposal.

### 5. Migrate external consumers in independent slices

Migrate test imports and dotted patch strings by test area, migrate the standalone generators/tasks/scripts, and migrate GEKO as its own consumer PR. For GEKO, prefer the installed-package/API shape already demonstrated by commit `54bc6ceb0b2`: direct Python APIs for merge/create, `python -m tensilelite` for process isolation, and no checkout `PYTHONPATH` injection for package resolution.

### 6. Remove the bridge

Remove `Tensile` compatibility only after the external-import audit is clean. Keep filesystem strings that are actual C++ or logic-data contracts separate from Python import names.

## Validation performed

All experiments used a temporary copy under `/tmp`; no product source was edited.

1. Parsed every Python file under `Tensile/` with `ast` and classified absolute and relative imports. Result: 240 package-internal absolute import nodes in 60 production files, 305 relative import nodes in 97 production files, and no parse errors.
2. Mechanically converted all 240 production import nodes in the temporary copy. `python -m compileall -q Tensile` passed.
3. Imported representative high-coupling modules from the converted tree: `Tensile.Tensile`, `BenchmarkProblems`, `ClientWriter`, `KernelWriterAssembly`, `LibraryIO`, `SolutionStructs.Solution`, `TensileCreateLibrary.Run`, `TensileLogic.Run`, `backends.ductile_backend`, and `ExperimentalLibrary`. All resolved from the temporary tree.
4. Ran `Tensile/bin/Tensile --help` and `python -m Tensile.TensileCreateLibrary --help` from the converted tree. Both passed. Exercised all nine legacy launchers from outside the checkout: eight reached their command parsing/help path; `TensileGenerateSummations` failed on the environment's missing `pandas`, and the unmodified launcher failed for the same dependency reason.
5. Ran the full unit collection after the all-file conversion. Collection failed at `test_KnownBugs.py` with `attempted relative import with no known parent package`, which identified the anonymous `spec_from_file_location` blocker. The same pattern was then found in the `ValidChipId` and `ValidCorpusConsistency` tests.
6. Re-ran the full unit suite with those three direct-loaded modules left absolute. Result: 7,155 passed, 275 skipped, 18 xfailed, 1 xpassed, and 6 failed in 442.88 seconds. The six failures inspect exact source spelling or formatting and were caused by the experiment's use of `ast.unparse()` over whole files, not by import resolution. A formatting-preserving focused conversion passed 94 representative unit tests.
7. Imported the exact Tensile modules used by GEKO against the converted tree; those imports succeeded. GEKO's full suite was not run because the available TensileLite unit environment lacks GEKO dependencies such as `pandas`, and the system environment lacks `scipy`.
8. Imported `Tensile.Common.GlobalParameters` and `tensilelite.Common.GlobalParameters` through a symlink alias after conversion. Both the module identity and `globalParameters` identity comparisons were false, confirming that relative imports do not replace the bridge.

## Decision

Proceed with a package-internal relative-import PR before the namespace move. Scope it to `Tensile/` implementation modules plus the three test-harness repairs needed to load package modules correctly. Keep tests as absolute public consumers, keep `Tensile/bin/*` as absolute boundary launchers, and keep GEKO unchanged until its consumer migration. This reduces rename churn and clarifies ownership without weakening the bridge or execution requirements in the existing proposal.

## Implemented-stack validation

The final local stack uses a one-way bridge only: after the move, legacy `Tensile.*` imports resolve to canonical `tensilelite.*` modules. The bridge is removed after the in-repo consumers migrate.

- After rebasing onto local `develop` at `b607a8ae0cfda0a27a0f6a1446cb3043a88c22c4`, the final unit suite passed with `7206 passed, 230 skipped, 18 xfailed, 1 xpassed` using four workers.
- Common-test collection completed successfully with 1,252 tests collected. The GPU/client-backed common tests were not executed.
- The focused import and path suite passed with 170 tests.
- Three focused GEKO path/import tests passed.
- A wheel built from a clean `git archive` contains the lowercase package and excludes both `Tensile/__init__.py` and the temporary bridge implementation at the final stack tip.
- A wheel built from the compatibility branch maps `Tensile.resources` and `tensilelite.resources` to the same module object while preserving canonical lowercase module metadata and package-resource lookup.
