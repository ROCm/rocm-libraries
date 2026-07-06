<!-- Copyright Advanced Micro Devices, Inc., or its affiliates. -->
<!-- SPDX-License-Identifier: MIT -->

# TensileLite Python Packaging Refactor

## Problem

`projects/hipblaslt/tensilelite/Tensile` is the hipBLASLt fork of the rocBLAS
Tensile generator, but the Python package, import paths, and command names still
look like upstream Tensile:

- The distribution metadata currently publishes as `tensile`.
- The import namespace is the top-level `Tensile` package.
- Many in-repo callers execute files under `Tensile/bin` or patch `sys.path` so
  that `import Tensile` and `import rocisa` resolve from a source or build tree.
- CMake installs raw Python source trees for test artifacts instead of consuming
  an installed Python package.

This leaves no clean way for another project, such as GEKO, to declare a normal
Python dependency on the TensileLite assets. It also keeps the legacy rocBLAS
Tensile tool and the TensileLite fork ambiguous in both Python imports and
executable names.

## Goals

- Publish the TensileLite generator assets as a normal Python package.
- Use a canonical package name that cannot be confused with legacy rocBLAS
  Tensile.
- Provide import paths that identify the code as TensileLite.
- Replace source-tree executable files and `PYTHONPATH` mutation with supported
  module entry points and console scripts.
- Let downstream projects depend on TensileLite through standard packaging
  metadata.
- Keep the existing hipBLASLt CMake device-library build working throughout the
  migration.

## Non-goals

- Do not change code generation semantics.
- Do not change the C++ TensileLite host-library ABI.
- Do not rename generated kernel artifacts or library logic file formats.
- Do not require GEKO or other downstream projects to vendor the tensilelite
  source tree.

## Target Package Shape

Use these canonical names:

- Python distribution: `tensilelite`
- Python import namespace: `tensilelite`
- Required native dependency: `rocisa`
- Required Python dependency for the existing command surface: `pandas`

The package should install only the TensileLite namespace. A normal install of
`tensilelite` should not claim the top-level `Tensile` package, because that
recreates the conflict with legacy Tensile.

The package should expose stable public modules for downstream users and build
tools:

```python
import tensilelite
from tensilelite.create_library import main as create_library_main
from tensilelite.logic import main as logic_main
```

The first implementation can preserve the current CamelCase module filenames
internally if that reduces churn, but public docs and new callers should use the
`tensilelite` namespace. Over time, internal imports should be moved to relative
imports or to the new namespace so the implementation no longer depends on
`Tensile`.

## Native Dependency Strategy

TensileLite depends on `rocisa`, a HIP/nanobind native extension that currently
builds from the in-repository `rocisa/` project. For this refactor, assume
`rocisa` is installed locally into the same Python environment before
`tensilelite` is installed or exercised. Supported local setups should include:

- developer mode: build/install `rocisa` from the workspace, for example through
  the existing `invoke rocisa` flow or an editable install;
- package-test mode: build a local `rocisa` wheel first, then install the local
  `rocisa` wheel and the local `tensilelite` wheel into a clean venv.

Publishing or otherwise distributing `rocisa` as a normal resolvable dependency
is a separate prerequisite for a fully self-contained downstream dependency
chain. Until then, downstream setup instructions for projects such as GEKO must
state the local `rocisa` installation precondition.

`pandas` must be added as a runtime dependency because
`TensileGenerateSummations` is part of the existing command surface and the
current implementation imports `pandas` at module import time.

## Native Artifact Packaging Boundary

The `tensilelite/` source tree contains more than the Python generator. It also
contains the native `rocisa` extension, the C++ `tensilelite-host` runtime
library, benchmark/client executables, C++ tests, static headers copied by
codegen, and hand-written assembly custom kernels.

Use this packaging split:

- `tensilelite` Python distribution: Python codegen modules plus package data
  needed by codegen, including static source headers, `CustomKernels/*.s`, and
  `known_bugs.yaml`.
- `rocisa` native distribution: the compiled `_rocisa` Python extension and its
  Python package. This is required by codegen and should be a separate native
  wheel/package rather than bundled into the `tensilelite` wheel.
- ROCm/system packages: ROCm compiler tools and native runtime libraries,
  including `amdclang++`, `clang-offload-bundler`, `hipconfig`, HIP runtime
  libraries, `libstinkytofu`, `liborigami`, and related ROCm libraries.
- ROCm/system packages, not the Python codegen wheel: `tensilelite-host`,
  `tensilelite-client`, `cpu-gemm-driver`, and C++ tests.

This keeps the Python codegen wheel mostly Python/data-only, isolates native ABI
and rebuild churn in `rocisa`, and leaves ROCm toolchain artifacts in ROCm's
normal packaging channel.

Python package managers cannot install or express all of these native ROCm
requirements as normal Python dependencies. `pip` can install Python
distributions and native Python wheels, but it cannot install a ROCm compiler
toolchain or system shared libraries from apt/rpm-style packages. `uv` improves
the developer workflow by supporting workspaces, local/path dependency sources,
lock/sync workflows, and building distributions, but it has the same boundary:
it resolves and installs Python packages, not ROCm system packages. For local
development, `uv` can point `tensilelite` at a workspace `rocisa`; for published
metadata, local `tool.uv.sources` entries must not be the only dependency story.

For now, document the local precondition: install a matching local `rocisa`
package and have the ROCm toolchain/system libraries available. Later, when
`rocisa` is published on an internal or public Python index, `tensilelite` should
declare a normal versioned `rocisa` dependency, while still validating external
ROCm tools at CLI/configure time.

## Command-Line Interface

Replace checked-in executable files under `Tensile/bin` with `pyproject.toml`
entry points. The only new public command should be `tensilelite`, with
subcommands for the existing tools. The following checked-in `Tensile/bin` files
exist today and define the initial command surface:

| Current file | Current callable | Target command | Backward-compatible alias |
|---|---|---|---|
| `Tensile` | `Tensile.Tensile.main()` | deferred; source-tree/developer workflow until package assumptions are removed | `Tensile` |
| `TensileBenchmarkCluster` | `Tensile.TensileBenchmarkCluster.main()` | `tensilelite benchmark-cluster` | `TensileBenchmarkCluster` |
| `TensileCreateLibrary` | `Tensile.TensileCreateLibrary.run()` | `tensilelite create-library` | `TensileCreateLibrary` |
| `TensileGenerateSummations` | `Tensile.GenerateSummations.GenerateSummations(sys.argv[1:])` | `tensilelite generate-summations` | `TensileGenerateSummations` |
| `TensileLibLogicToYaml` | `Tensile.TensileLibLogicToYaml.main()` | `tensilelite liblogic-to-yaml` | `TensileLibLogicToYaml` |
| `TensileLogic` | `Tensile.TensileLogic.main()` | `tensilelite logic` | `TensileLogic` |
| `TensileMergeLibrary` | `Tensile.TensileMergeLibrary.main()` | `tensilelite merge-library` | `TensileMergeLibrary` |
| `TensileRetuneLibrary` | `Tensile.TensileRetuneLibrary.main()` | `tensilelite retune-library` | `TensileRetuneLibrary` |
| `TensileUpdateLibrary` | `Tensile.TensileUpdateLibrary.main()` | `tensilelite update-library` | `TensileUpdateLibrary` |

The current installed package also exposes two console scripts that are not
checked-in `Tensile/bin` files:

| Current console script | Current callable | Target command | Decision |
|---|---|---|---|
| `TensileGetPath` | `Tensile.PrintTensileRoot()` | `tensilelite get-path` or remove | Keep only as a temporary compatibility alias if installed-package users rely on it. |
| `TensileVerifyStinkyElfText` | `Tensile.verify_stinky_comment_vs_elf_text.main(argv=None)` | `tensilelite verify-stinky-elf-text` | Keep as a temporary compatibility alias unless the tool is declared out of scope. |

First packaged release support matrix:

| Mode | Status | Rationale |
|---|---|---|
| `tensilelite create-library` | Supported | Required by hipBLASLt device-library generation and downstream library builds. |
| `tensilelite logic` | Supported | Required for logic validation and `--check-all`. |
| `tensilelite run` | Deferred | The current workflow assumes checkout-relative configs, `build_tmp`, `Tensile/bin`, and a source-tree/prebuilt client flow. Keep `Tensile/bin/Tensile` as a developer/source-tree workflow until those assumptions are removed. |
| `tensilelite generate-summations` | Supported if retained from `Tensile/bin` | Requires `pandas` and removal of the subprocess call to `Tensile/bin/TensileCreateLibrary`. |
| `tensilelite benchmark-cluster`, `merge-library`, `retune-library`, `update-library`, `liblogic-to-yaml`, `verify-stinky-elf-text`, `get-path` | Compatibility or developer tools | Each must be classified as supported, compatibility-only, developer-only, or removed before publishing. |

The first packaged public CLI should focus on `create-library` and `logic`.
Other modes should be promoted only after their source-tree assumptions,
dependencies, and validation coverage are explicitly resolved.

Do not add new executable-per-tool names such as `tensilelite-create-library`.
Those names add a second public command family without solving a compatibility
problem.

Each tool currently has its own argv contract: most `main()` functions parse
`sys.argv`, `TensileCreateLibrary.run()` parses process-global arguments, and
`TensileGenerateSummations` takes an argv list directly. Normalizing those into
subcommand functions is required migration work. The final dispatcher should
call subcommands through explicit `argv` parameters and should only read or
rewrite `sys.argv` at the outermost compatibility boundary.

The new package metadata should have one primary entry point plus legacy aliases
for the retained compatibility commands:

```toml
[project.scripts]
tensilelite = "tensilelite.cli:main"
Tensile = "tensilelite.cli.compat:tensile"
TensileBenchmarkCluster = "tensilelite.cli.compat:benchmark_cluster"
TensileCreateLibrary = "tensilelite.cli.compat:create_library"
TensileGenerateSummations = "tensilelite.cli.compat:generate_summations"
TensileLibLogicToYaml = "tensilelite.cli.compat:liblogic_to_yaml"
TensileLogic = "tensilelite.cli.compat:logic"
TensileMergeLibrary = "tensilelite.cli.compat:merge_library"
TensileRetuneLibrary = "tensilelite.cli.compat:retune_library"
TensileUpdateLibrary = "tensilelite.cli.compat:update_library"
TensileGetPath = "tensilelite.cli.compat:get_path"
TensileVerifyStinkyElfText = "tensilelite.cli.compat:verify_stinky_elf_text"
```

The package should also provide `tensilelite/__main__.py` so
`python -m tensilelite ...` uses the same dispatcher as the installed
`tensilelite` console script.

Each compatibility alias must flow through the new CLI dispatcher instead of
calling the old implementation directly. The alias should prepend the matching
subcommand and pass a `compat_path` value into the new main path:

```python
def create_library(argv: Sequence[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else list(argv)
    return main(
        ["create-library", *args],
        compat_path=("TensileCreateLibrary", "tensilelite create-library"),
    )
```

The shared main/subcommand functions should print a deprecation warning when
`compat_path` is set:

```text
TensileCreateLibrary is deprecated and will be removed in a future release.
Use `tensilelite create-library` instead.
```

This keeps warning text, exit behavior, logging, and argument parsing on the new
code path while preserving the old command names for existing scripts.

## Build-System Integration

The hipBLASLt build should treat TensileLite as an installed Python package in
the build Python environment:

1. Install `tensilelite` and `rocisa` into the build venv, preferably
   editable during local development and as wheels in packaging tests.
2. Replace `HIPBLASLT_PYTHON_COMMAND` path injection with package execution.
3. Invoke device-library generation with:

   ```bash
   python -m tensilelite create-library ...
   ```

4. Invoke logic validation with:

   ```bash
   python -m tensilelite logic ...
   ```

5. Replace `HIPBLASLT_INSTALL_TENSILELITE_TEST_ARTIFACTS` raw source installs
   with installation of the package wheel and any required test data.

The current CMake call sites that execute `Tensile/bin/TensileLogic` or
`python -m Tensile.TensileCreateLibrary` should be converted after the new entry
points exist. This removes the build's dependence on a particular source-tree
layout.

The CMake cutover must update `HipBLASLtCodegen.cmake` and
`hipblaslt_python.cmake` together. Today, CMake constructs a Python command that
injects `PYTHONPATH`, invokes `Tensile/bin/TensileLogic`, invokes
`python -m Tensile.TensileCreateLibrary`, passes `--known-bugs` as an explicit
source-tree path, and lists that file in custom-command dependencies. The
packaged command path must choose one `known_bugs.yaml` source of truth:

- keep an explicit `--known-bugs <path>` argument in CMake until the package
  resource path can be resolved cleanly at configure time; or
- make `tensilelite logic --check-all` default to the packaged
  `known_bugs.yaml`, remove the explicit CMake argument, and update the CMake
  stamp/dependency model so edits to the installed/editable package still
  trigger validation when needed.

Do not move `known_bugs.yaml` behind package resources in one step while leaving
CMake hard-coded to the old source-tree file path.

### CMake And Source-Tree Caller Inventory

The package refactor must remove all in-repo dependencies on the legacy
`Tensile` import namespace and source-tree command paths. The following known
callers must be migrated or explicitly removed:

| Caller | Current behavior | Target behavior | Notes |
|---|---|---|---|
| `cmake/hipblaslt_python.cmake` | Builds `HIPBLASLT_PYTHON_COMMAND` by setting `PYTHONPATH` to the build `rocisa` path and source-tree `tensilelite`. | Use the selected build Python with installed/editable `tensilelite` and `rocisa`; add configure-time `import tensilelite, rocisa` validation. | Defines the command environment used by codegen steps. |
| top-level `CMakeLists.txt` Python setup | Chooses bundled Python mode or raw `Python_EXECUTABLE` and may build `_rocisa` as a dependency. | Ensure package install/editable setup happens before CMake generator steps, and validate imports with the exact build Python. | Non-bundled Python currently does not prove dependencies are importable. |
| `cmake/HipBLASLtCodegen.cmake` | Runs `${_codegen_dir}/Tensile/bin/TensileLogic` and `python -m Tensile.TensileCreateLibrary`. | Run `python -m tensilelite logic` and `python -m tensilelite create-library`. | Must resolve the `known_bugs.yaml` source-of-truth and CMake `DEPENDS` behavior. |
| `device-library/CMakeLists.txt` | Calls `hipblaslt_create_device_library` with source logic path and build output path. | Keep logic/output semantics while routing generation through installed `tensilelite`. | Integration caller for device-library generation. |
| `device-library/extops/CMakeLists.txt` | Runs `LayerNormGenerator.py`, `SoftmaxGenerator.py`, `AMaxGenerator.py`, and `ExtOpCreateLibrary.py` by source-tree path. | Package these generators as `tensilelite` modules/subcommands or declare ext-op source-tree generation out of scope. | These scripts import `Tensile.*` today and are part of the device-library build when extops are enabled. |
| top-level `CMakeLists.txt` test-artifact install | Installs raw `tensilelite/Tensile/` and selected `Tensile/bin` scripts under `share/hipblaslt/tensilelite`. | Install the package wheel and explicit test data/artifacts, not the raw legacy package tree. | Controlled by `HIPBLASLT_INSTALL_TENSILELITE_TEST_ARTIFACTS`. |
| `tensilelite/pyproject.toml` | Distribution is `tensile`; scripts point at `Tensile.*`; pytest/mutmut config names `Tensile`. | Rename distribution to `tensilelite`, discover the new namespace, update script targets, and update test/tool config paths. | Current installed scripts do not match the full `Tensile/bin` surface. |
| `tensilelite/MANIFEST.in` | Broadly includes `Tensile` files and selected `Tensile/bin` launchers. | Replace with an intentional package-data allowlist for the new namespace. | Avoid sdist/wheel drift. |
| `tensilelite/tox.ini`, `pytest.ini`, and pyproject test config | Run tests against source-tree `Tensile/Tests`, `--cov=Tensile`, and `PYTHONPATH={toxinidir}`. | Add installed-wheel environments with no source-tree `PYTHONPATH`; update coverage/test paths to `tensilelite`. | Source-tree tests can mask packaging failures. |
| pre-commit tooling | Maps affected files and imports using `Tensile` module/test paths. | Map affected files and imports using `tensilelite` paths. | Otherwise developer gates keep preserving the old namespace. |
| `scripts/run_tensile_logic_check.py` | Mutates `sys.path`, finds build-tree `rocisa`, and imports `Tensile.TensileLogic`. | Invoke the installed package API or `python -m tensilelite logic`. | Keep cross-platform behavior, but remove source-tree import assumptions. |
| `scripts/run_tensile_logic_check.sh` and `scripts/README.md` | Wrap/document the Python script and source-tree `known_bugs.yaml`. | Document the installed package command and chosen known-bugs behavior. | Keep convenience wrapper if useful. |
| `install.sh` | Installs `tensilelite/requirements.txt` into a venv rather than installing the package. | Install local `rocisa` and `tensilelite` package/editable into the venv. | Requirements-only setup can leave imports dependent on the source checkout. |
| `tensilelite/cmake/tensilelite_auto_build.cmake` | Generates deprecated wrappers around `${source}/Tensile/bin/*`. | Remove the source-tree wrapper path or regenerate wrappers around `python -m tensilelite ...`. | If kept, it must not depend on `Tensile/bin`. |
| `tensilelite/LayerNormGenerator.py`, `SoftmaxGenerator.py`, `AMaxGenerator.py` | Ext-op generator scripts import `Tensile.Common.*` and are run by CMake path. | Move under the `tensilelite` package and expose module/subcommand entry points. | Active with extops; depends on `rocisa` and toolchain globals. |
| `tensilelite/ExtOpCreateLibrary.py` | CMake runs the raw script by path. | Move under the `tensilelite` package and expose a module/subcommand. | Does not import `Tensile`, but is still a source-layout dependency. |
| `tensilelite/Tensile/bin/*` | Checked-in launchers patch `sys.path` and import `Tensile.*`. | Replace with console-script aliases routed through `tensilelite.cli`. | Preserve temporarily only for compatibility warnings. |
| `Tensile/ClientWriter.py` | Builds command strings using `ROOT_PATH + "/bin/TensileCreateLibrary"`. | Use the new dispatcher or an in-process create-library API. | Required before `Tensile/bin` can be removed. |
| `Tensile/GenerateSummations.py` | Locates `bin/TensileCreateLibrary` relative to `__file__` and shells out to it. | Use the new dispatcher or an in-process create-library API. | Also requires `pandas`. |
| `Tensile/ClientExecutable.py` | Uses `SOURCE_PATH` as a CMake source directory for client builds. | Require a prebuilt client or replace with a supported packaged client-build flow. | `Tensile/Source` has no `CMakeLists.txt` in this tree. |
| `Tensile/__init__.py` path constants | Exposes `ROOT_PATH`, `SOURCE_PATH`, and `CUSTOM_KERNEL_PATH`; internal modules consume them. | Replace with resource helpers under `tensilelite`. | Do not preserve these as default public API in the canonical wheel. |
| `Tensile/Tensile.py` | Constructs `Configs/` and `Tests/` paths relative to source file and documents `Tensile/bin/Tensile`. | Defer packaged `run`, or require external configs/prebuilt client and remove source-tree fallbacks. | `Tensile/Configs` is absent in this tree. |
| `TensileCreateLibrary.copyStaticFiles` | Copies headers from `SOURCE_PATH`. | Use a resource helper such as `copy_static_headers(output_dir)`. | Needed for library generation. |
| custom-kernel loaders | Read `CustomKernels/*.s` through `CUSTOM_KERNEL_PATH`. | Use resource helpers such as `read_custom_kernel(name)` and `list_custom_kernels()`. | Needed for custom-kernel support. |
| `Tensile/TensileBenchmarkCluster.py` | Derives `RootTensileDir` from `__file__` for script/docker paths. | Use explicit external paths or package resources if still supported. | Exposed by an existing console script. |
| `Tensile/TensileCreateLibrary/__main__.py` | Supports old `python -m Tensile.TensileCreateLibrary`. | Use `python -m tensilelite create-library`; keep old module mode only in compatibility package if needed. | CMake currently depends on old module mode. |
| `Tensile/verify_stinky_comment_vs_elf_text.py` | Documents old module CLI and is exposed by `TensileVerifyStinkyElfText`. | Move to `tensilelite verify-stinky-elf-text` or developer-only module. | Decide support level before publishing. |
| `Tensile/Tests/conftest.py`, common test helpers, and subprocess tests | Mutate `sys.path` and forward `PYTHONPATH` so subprocesses import `Tensile`. | Installed-wheel tests must import `tensilelite` without adding source root. | This is the main way tests can hide packaging breakage. |
| `Tensile/Tests/**` | Tests import, patch, or importlib-load `Tensile.*`; some load files by relative legacy paths. | Migrate tests to `tensilelite.*`; keep old-name checks only in explicit compatibility tests. | Broad churn; snapshots may pin old names/text. |
| docs, examples, readmes, and test data comments | Document commands such as `Tensile/bin/Tensile`, `Tensile/bin/TensileCreateLibrary`, `Tensile/bin/TensileLogic`, `Tensile/Tests`, and requirements-only setup. | Update to package install/editable setup and `python -m tensilelite ...` commands, or mark source-tree developer workflows explicitly. | Includes README, CONTRIBUTING, AGENTS files, custom-kernel docs, utility docs, and config readmes. |
| documentation tooling metadata | Targets `projects/hipblaslt/tensilelite/Tensile`. | Point to the new package/source layout. | Tooling-only, but otherwise docs tooling may miss new files. |

Before publishing a canonical `tensilelite`-only wheel, all in-repo imports of
`Tensile.*`, invocations of `Tensile/bin`, and source-tree `PYTHONPATH`
assumptions must be gone. The only remaining legacy-name references should be
compatibility entry-point definitions, explicit compatibility tests, and
deprecation documentation.

## Package Data

The package-data contract should support the features currently reachable from
`Tensile/bin` without copying the entire current source tree:

- Library logic YAML generation in the existing source-tree `run` pipeline
  primarily needs Python modules and its input benchmark data. It does not need
  static source headers, `known_bugs.yaml`, or CMake helper files just to
  analyze benchmark CSV/YAML data and write `3_LibraryLogic/*.yaml`.
- Library generation (`tensilelite create-library`) needs the static source
  headers copied today by `copyStaticFiles`: `TensileTypes.h`,
  `tensile_bfloat16.h`, `tensile_float8_bfloat8.h`, `KernelHeader.h`,
  `ReductionTemplate.h`, and `memory_gfx.h`.
- `CustomKernels/*.s` must be included if packaged TensileLite supports custom
  kernel workflows. Benchmark configs can request custom kernels with
  `CustomKernels`, logic files can reference `CustomKernelName`, and codegen
  reads the matching assembly file from `CustomKernels/`.
- `known_bugs.yaml` should be included if the packaged `tensilelite logic
  --check-all` command is expected to provide the same default documented-skip
  behavior as the hipBLASLt CMake validation gate. It is not needed for
  producing LibraryLogic YAMLs or code objects.
- CMake helper files from `Tensile/Source` should not be included by default for
  the `Tensile/bin` feature boundary. They appear to be legacy support for an
  older CMake-based client/source workflow; in this tree, `Tensile/Source` has
  no `CMakeLists.txt`, and the active client path expects a prebuilt client.

Package-data decisions:

- Include the static source headers required by `copyStaticFiles`.
- Include `CustomKernels/*.s`; custom kernel support is part of the current
  command surface and codegen reads these files by name.
- Include `TensileLogic/known_bugs.yaml` only as validation support for
  `tensilelite logic --check-all`; it is not a generation input.
- Exclude `Tensile/Source/EnableWarnings.cmake` and
  `Tensile/Source/FindOpenCL.cmake` from the wheel unless a supported packaged
  workflow is found that still uses them. A repository scan found no tracked
  callers outside the files themselves.
- Rewrite or remove `MANIFEST.in` as part of this step. The current manifest
  recursively includes broad `Tensile` file globs and selected `Tensile/bin`
  files; the source distribution and wheel must use the same intentional
  package-data allowlist.

Code should access these files through `importlib.resources` instead of
constructing paths from `__file__` and assuming a checkout layout. Any path that
must remain user-visible should have an explicit API. Internal callers that
currently construct source-tree paths, including `ClientWriter`,
`GenerateSummations`, `TensileCreateLibrary.copyStaticFiles`, and custom-kernel
loading, must be converted before `Tensile/bin` and the top-level `Tensile`
package are removed from the canonical wheel.

Suggested resource helper APIs:

- `copy_static_headers(output_dir)`: copy the static header set needed by
  generated libraries.
- `read_custom_kernel(name)` and `list_custom_kernels()`: read/list bundled
  custom kernel assembly resources without exposing package layout.
- `default_known_bugs_path()` or `load_default_known_bugs()`: expose
  `known_bugs.yaml` for validation. If CMake needs a real filesystem path, use a
  helper that materializes or resolves the resource intentionally.

## Legacy Import Compatibility

The canonical `tensilelite` wheel should not install a top-level `Tensile`
package once the new namespace is ready. A same-wheel shim would make every
`tensilelite` install claim the legacy `Tensile` namespace, recreate the naming
conflict this refactor is meant to remove, and keep APIs such as
`Tensile.ROOT_PATH`, `Tensile.SOURCE_PATH`, and `Tensile.CUSTOM_KERNEL_PATH`
alive by default.

If external users need a short-lived transition path for `import Tensile.*`,
ship it as a separate opt-in compatibility distribution, for example
`tensilelite-tensile-compat`. That package should:

- depend on a tightly matched `tensilelite` version;
- install the top-level `Tensile` namespace only in the compatibility package;
- warn visibly on `import Tensile`, preferably with a `FutureWarning`-based
  custom warning because `DeprecationWarning` is often hidden;
- include a specific removal release in the warning text;
- document that it is mutually exclusive with any other package that owns the
  `Tensile` import namespace.

The compatibility package must avoid loading implementation modules twice. A
naive `__path__` shim can make `Tensile.Common.DataType` and
`tensilelite.Common.DataType` become different module objects, splitting module
globals, class identities, caches, and `globalParameters`. If an import
compatibility package is required, it should use a prefix-rewriting import hook
or equivalent module-aliasing layer so lazy imports such as
`Tensile.Common.DataType` resolve to the same module objects as
`tensilelite.Common.DataType`. A top-level `sys.modules["Tensile"] =
tensilelite` alias alone is not sufficient for submodule imports that have not
yet been loaded.

The main `tensilelite` wheel should still keep temporary `Tensile*`
console-script aliases during the transition. Console-script aliases do not
claim the Python `Tensile` namespace and can safely route through
`tensilelite.cli` with `compat_path` warnings.

## Migration Plan

1. Add the new package metadata.
   Rename the distribution from `tensile` to `tensilelite`, add `rocisa` as a
   dependency, split runtime dependencies from dev/test tools, add the
   `tensilelite` console script, and add legacy `Tensile*` aliases that dispatch
   through the new CLI with `compat_path`. Single-source the package version so
   `pyproject.toml`, runtime `__version__`, and any generated/config version
   references cannot drift.

2. Add the new import namespace.
   Introduce `tensilelite` as the supported import path. Keep the implementation
   thin at first if needed, but make all new callers use the new namespace.

3. Convert internal and in-repo imports.
   Move every in-repo import and reference from `Tensile.*` to relative imports
   or `tensilelite.*`. Update tests, CMake Python invocations, helper scripts,
   docs, and test fixtures. This is the main behavioral risk. Existing
   characterization tests are useful but insufficient by themselves because many
   tests still import `Tensile` or modify `sys.path` to emulate the old
   source-tree layout. The tests must be migrated to import `tensilelite` and CI
   must include an installed-wheel test that does not place the source-tree
   `Tensile` package on `sys.path`.

4. Deprecate legacy entry points.
   Keep `Tensile`, `TensileCreateLibrary`, and related command names only as
   warnings-backed compatibility aliases for a defined release window. Do not
   require those aliases for hipBLASLt's own build, and do not add new
   `tensilelite-*` executable aliases.

5. Replace implicit source-tree asset reads.
   Move the static headers, `CustomKernels/*.s`, and `known_bugs.yaml` behind
   explicit resource-access helpers. As part of this step, verify that
   `Tensile/Source/EnableWarnings.cmake` and `Tensile/Source/FindOpenCL.cmake`
   remain unused; if so, leave them out of package data and consider removing
   them from the source tree in a follow-up cleanup. Convert internal callers
   that shell out to `Tensile/bin`, such as `ClientWriter` and
   `GenerateSummations`, to use the new dispatcher or an in-process API.

6. Confirm the supported CLI surface.
   Verify whether `tensilelite run` remains functional in this fork. The current
   `Tensile.py` references a `Configs` directory that is not present in this
   tree; either document that configs must be supplied externally, restore the
   required package data, or remove/rename the advertised mode before publishing
   the new CLI.

7. Stop installing the top-level `Tensile` package.
   Once in-repo and downstream callers have migrated, remove the `Tensile`
   package from the canonical wheel. `pip install tensilelite` should make
   `import tensilelite` work and `import Tensile` fail. If a short transition
   for legacy Python imports is required, use a separate opt-in compatibility
   package as described above; do not keep `Tensile` as a shim inside the main
   wheel. Exit criteria for this step: no in-repo CMake/script/test/doc caller
   invokes `Tensile/bin`, no in-repo code imports `Tensile.*`, no test fixture
   adds the source-tree `Tensile` package to `PYTHONPATH`, and installed-wheel
   CI passes without the source tree on `PYTHONPATH`. The only remaining
   legacy-name references should be compatibility entry-point definitions,
   explicit compatibility tests, and deprecation documentation.

## Downstream Dependency Model

GEKO and other consumers should be able to declare:

```toml
[project]
dependencies = [
  "tensilelite>=5.0",
]
```

Until `rocisa` is published or otherwise resolvable as a normal package
dependency, this downstream dependency model also requires installing a matching
local `rocisa` build into the same environment.

For monorepo or source-based development before publication, downstreams can use
a direct reference to the subdirectory:

```toml
tensilelite @ git+https://github.com/ROCm/rocm-libraries.git@develop#subdirectory=projects/hipblaslt/tensilelite
```

That dependency should provide importable APIs and command-line tools without
requiring GEKO to set `PYTHONPATH`, copy `Tensile/bin`, or know the hipBLASLt
repository layout.

## Validation

The refactor should be validated in both package and hipBLASLt build modes:

- Build or install a local `rocisa` package into the test environment first.
- Build a wheel with `python -m build`.
- Install the local `rocisa` package and the `tensilelite` wheel into a clean
  venv and verify `import tensilelite`.
- Verify a clean venv does not import top-level `Tensile` from the new package
  once the compatibility window ends.
- Run `tensilelite --help` and subcommand `--help` checks for every supported
  mode, and compare key exit-code and usage/error-message behavior against the
  legacy commands during the compatibility window.
- Run compatibility alias smoke tests and verify each one prints the deprecation
  warning with the replacement `tensilelite <subcommand>` command.
- Run `tox -e unit` from `tensilelite`.
- Run installed-wheel tests without adding the source tree or `Tensile/` to
  `PYTHONPATH`.
- Run a scoped hipBLASLt device-library build with `TENSILELITE_LOGIC_FILTER`.
- Run `scripts/run_tensile_logic_check.py` after it has been converted to the
  installed package API.
- Build or smoke-test a GEKO environment that depends on `tensilelite` without
  extra path configuration, after installing the local `rocisa` dependency.
- Run grep gates that fail on accidental legacy namespace or source-tree command
  use, excluding generated outputs. The exact patterns can evolve, but should
  cover `Tensile/bin`, `python -m Tensile`, `TensileCreateLibrary`,
  `TensileLogic`, `TensileGetPath`, `TensileVerifyStinkyElfText`, `PYTHONPATH`,
  `sys.path` mutation, `from Tensile`, `import Tensile`, `ROOT_PATH`,
  `SOURCE_PATH`, `CUSTOM_KERNEL_PATH`, `Tensile/Tests`, and `--cov=Tensile`.
- Run ext-op device-library generation if extops remain supported by the
  packaged build path.

## Open Questions

- How long should legacy `Tensile*` command aliases remain available?
- Should `TensileGetPath` remain as a compatibility alias, become
  `tensilelite get-path`, or be removed with a documented replacement?
- Is `TensileVerifyStinkyElfText` part of the supported packaged command
  surface, or should it remain an internal/developer-only tool?
- Which later release, if any, should promote `tensilelite run` from
  source-tree/developer workflow to public packaged CLI, and what exact
  external-config and prebuilt-client contract should it require?
- Should CMake continue passing an explicit `--known-bugs` path, or should
  `tensilelite logic --check-all` use the packaged `known_bugs.yaml` by default?
- Should a separate compatibility package provide the old top-level `Tensile`
  namespace for users that cannot migrate immediately?
- Should the ROCm binary package install Python wheels directly, or continue to
  place package contents under `share/hipblaslt` for test artifacts?
