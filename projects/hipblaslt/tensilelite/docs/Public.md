<!-- Copyright Advanced Micro Devices, Inc., or its affiliates. -->
<!-- SPDX-License-Identifier: MIT -->

# Proposal: Package TensileLite as `tensilelite`

## Summary

Package the TensileLite Python generator as `tensilelite`, with `import tensilelite` and `python -m tensilelite`/`tensilelite` as the supported public interface.

Today, downstream users and hipBLASLt build logic have to depend on the hipBLASLt checkout layout, `PYTHONPATH` injection, and legacy `Tensile` names to run the generator.

Move the generator modules, CLI, static headers, custom kernels, and logic validation resources into a real Python package while keeping generated kernels and host ABI unchanged.

Treat the package as part of a version-matched ROCm/TheRock artifact set: `rocisa` remains a separate native dependency, and `tensilelite-client` is discovered under `ROCM_PATH` for `tensilelite run`.

## Why This Matters

Today, downstream users cannot consume TensileLite as a normal dependency. A project such as GEKO has to know where the hipBLASLt checkout lives, how the source tree is laid out, and which source or build directories must be injected into `PYTHONPATH` before invoking the generator.

That is fragile for users and for hipBLASLt itself. It makes build scripts depend on incidental checkout structure, makes installed-artifact testing different from normal Python packaging, and keeps the hipBLASLt fork ambiguous with upstream rocBLAS Tensile.

Packaging TensileLite gives us a cleaner contract:

- Python imports resolve from an installed package.
- CMake invokes a package command instead of a source-tree script.
- Data files used by code generation are declared package resources.
- Binary dependencies are version-matched against the ROCm/TheRock build that produced them.
- The public surface distinguishes hipBLASLt TensileLite from upstream Tensile.

## Decision

Package the TensileLite Python generator as `tensilelite`.

The supported public Python name would become:

```python
import tensilelite
```

The supported command-line entry point should become:

```bash
tensilelite
python -m tensilelite
```

The packaged command surface should be defined and completed as part of this transition:

- `tensilelite benchmark-cluster`
- `tensilelite create-library`
- `tensilelite generate-summations`
- `tensilelite logic`
- `tensilelite logic-to-yaml`
- `tensilelite merge-library`
- `tensilelite retune-library`
- `tensilelite run`
- `tensilelite update-library`

`tensilelite run` covers the benchmark and tuning workflow. Its command contract should be the current `Tensile` command contract, moved behind the new package CLI rather than redesigned during this migration. Making it public means this transition must also fix the compiled client artifact contract: `tensilelite-client` should be installed under `ROCM_PATH`, the Python package should find it there, and `--prebuilt-client` should become an override rather than the normal packaged path.

## Non-Goals

This proposal should not change:

- Generated kernel semantics.
- Library logic format.
- The hipBLASLt C++ host ABI.
- ROCm compiler or runtime packaging.
- The native implementation of `rocisa`.

The refactor is about packaging boundaries, names, resource lookup, and build integration.

## Current Pain Points

TensileLite is the hipBLASLt fork of Tensile, but it still looks like upstream rocBLAS Tensile at most Python boundaries:

- The Python distribution metadata publishes as `tensile`.
- The import namespace is `Tensile`.
- Tools are launched through `Tensile/bin/*` or `python -m Tensile.*`.
- CMake and helper scripts patch `PYTHONPATH` or mutate `sys.path` so imports resolve from a checkout or build tree.
- Test artifact install paths copy raw Python source trees instead of installing packages.

This keeps two different projects ambiguous: upstream rocBLAS Tensile and the hipBLASLt TensileLite fork. Reusing the `Tensile` import name and legacy executable names makes it easy to load the wrong code, hide missing package metadata, or accidentally preserve source-tree assumptions in tests.

## Target State

The package should make the common path ordinary:

- `pip install tensilelite` provides `import tensilelite`.
- `python -m tensilelite create-library ...` works without setting `PYTHONPATH`.
- `python -m tensilelite logic ...` works without setting `PYTHONPATH`.
- `python -m tensilelite run ...` works with the version-matched `tensilelite-client` installed under `ROCM_PATH`.
- Downstream projects can depend on TensileLite through Python package metadata or a ROCm/TheRock package index.
- hipBLASLt's own CMake build uses the installed package in its build Python environment.
- The wheel contains the Python generator and the data files it needs, not a copy of the source tree.

The binary dependency path must also be explicit:

- `rocisa` is a separate package with a native `_rocisa` extension.
- The `rocisa` package must be built against the same ROCm/TheRock dependency set expected by TensileLite.
- Any compiled executable that a public TensileLite command invokes must be installed as a version-matched artifact under `ROCM_PATH`. Explicit user-provided paths can remain overrides, not the normal packaged path.
- Client-using benchmark and validation workflows should fail clearly if the
  required `ROCM_PATH` artifact lookup cannot find `tensilelite-client`; package
  import and client-free code generation must not require it.

## User-Facing Names

The public names should describe the fork people are actually using:

| Today | Proposed |
| --- | --- |
| `import Tensile` | `import tensilelite` |
| `Tensile` | `tensilelite run` |
| `TensileCreateLibrary` | `tensilelite create-library` |
| `TensileLogic` | `tensilelite logic` |
| `python -m Tensile.tensilelite_create_library ...` | `python -m tensilelite create-library ...` |

Temporary `Tensile*` command aliases are reasonable during migration, but they should be provided by `tensilelite-tensile-compat`, not by the default `tensilelite` wheel. They should route through the new CLI and print deprecation warnings.

The canonical `tensilelite` wheel should never ship a top-level `Tensile` Python package. Keeping `import Tensile` in the default wheel preserves the naming conflict this work is meant to remove.

## Binary And ROCm/TheRock Contract

The Python package needs a binary contract, not just a package rename.

At a minimum, the package set must define how these compiled or system-provided pieces are resolved:

| Artifact | Needed by | Packaging contract |
| --- | --- | --- |
| `rocisa` Python package and `_rocisa` extension | Code generation imports and assembly emission | Separate `rocisa` package, version-matched with `tensilelite` and the ROCm/TheRock build. |
| `libstinkytofu` | Runtime dependency of `_rocisa` in released builds, or vendored dependency in source-built standalone wheels | Either shipped by the same ROCm/TheRock artifact set or intentionally vendored next to `_rocisa`; do not leave it as an accidental loader-path dependency. |
| `origami`, HIP headers/runtime, and ROCm libraries | Build and runtime dependencies of native components | Resolved from the same ROCm/TheRock build; not bundled into the `tensilelite` Python wheel. |
| ROCm compiler tools such as `amdclang++` and the offload bundler | `create-library` when compiling generated kernels | Required system/toolchain dependencies; `pip` cannot provide them. |
| `tensilelite-host` | hipBLASLt host library and C++ client linkage | ROCm/TheRock library artifact when shared, or internal static link input when not shared. It is not Python package data. |
| `tensilelite-client` | Benchmark, tuning, and `run` workflows | Install under `ROCM_PATH`, preferably `$ROCM_PATH/libexec/hipblaslt/tensilelite/tensilelite-client` or the platform equivalent. `pip install tensilelite` should search `ROCM_PATH` and fail if the client is missing. `--prebuilt-client` can remain an explicit override, not the primary packaged path. |
| `cpu-gemm-driver` | Local C++ test flows only | Test-only build artifact. Do not install it into release artifacts and do not include it in the Python wheel. |

The current tree shows why this matters:

- `rocisa` is already a native package boundary. Its standalone wheel installs `_rocisa` and `rocisa/__init__.py`; source-built standalone wheels can vendor `libstinkytofu`, while released builds expect `libstinkytofu` from the ROCm library path.
- The hipBLASLt test artifact path currently copies raw `Tensile/` source, normalizes `rocisa`, installs `_rocisa`, and may co-install `stinkytofu`.
- `tensilelite-client` is built by CMake and installed into the test artifact tree only when `HIPBLASLT_INSTALL_TENSILELITE_TEST_ARTIFACTS` and `TENSILELITE_ENABLE_CLIENT` are enabled.
- `cpu-gemm-driver` is built beside `tensilelite-client`, but it is only used by TensileLite CTest coverage and should remain out of release artifacts.
- `tensilelite-host` install rules are tied to shared-library builds. In the default static-host path, it is a build/link artifact rather than an independently installed runtime library.

Those are not blockers, but they must be turned into intentional packaging choices. A package that imports successfully on a developer checkout but fails in a clean ROCm/TheRock install because `_rocisa` or `libstinkytofu` is missing has not solved the integration problem. A missing `tensilelite-client` is instead a targeted failure when a benchmark or validation workflow requests it.

## Feedback Requested

Before implementation starts, reviewers should agree on these points:

- Are there any known `Tensile` CLI behaviors that should intentionally not carry forward into `tensilelite run`?
- How do we version-lock `tensilelite` and `rocisa`: exact package versions, a shared ROCm release version, a TheRock artifact manifest, or another mechanism?
- How long should the separate `tensilelite-tensile-compat` package stay available?
- Are ext-op generators part of the packaged command surface?
- Should ROCm binary packaging install Python wheels directly, or continue installing selected files under `share/hipblaslt` for test artifacts?

## Package Boundary

The Python wheel should be intentionally small. It should include the generator and the data files needed by the generator. It should not become a dumping ground for every file under `tensilelite/`.

| Item | Where it belongs | Why |
| --- | --- | --- |
| Python codegen modules | `tensilelite` wheel | Required for logic processing and library generation. |
| Static headers copied by `create-library` | `tensilelite` package data | Generated libraries need these headers in the output tree. |
| `CustomKernels/*.s` | `tensilelite` package data | Configs and logic files can reference custom kernels by name; codegen must be able to read the matching assembly. |
| `tensilelite_logic/known_bugs.yaml` | `tensilelite` package data | Default documented exception list for `tensilelite logic --check-all`; CMake and downstream callers should not need to pass a source-tree path. |
| `rocisa` Python package and `_rocisa` extension | Separate `rocisa` package | Native extension with its own ABI and ROCm/TheRock dependency contract. |
| `libstinkytofu` | ROCm/TheRock package or vendored `rocisa` wheel dependency, depending on build mode | Runtime dependency of `_rocisa`; it must be intentionally resolved. |
| ROCm compiler tools and system libraries | ROCm/system packages | `pip` cannot install `amdclang++`, HIP runtime libraries, `hipconfig`, or similar system tools. |
| `tensilelite-host` and `tensilelite-client` | ROCm/system runtime artifacts | These are compiled artifacts, not Python generator resources. |
| `cpu-gemm-driver` and C++ tests | Source tree and CI test builds only | Test-only artifacts; do not package into release artifacts. |
| `tasks.py` Invoke tasks | Source tree and CI only, except wrappers around packaged commands | Developer build tasks such as editable `rocisa`, `build-client`, coverage, and pre-commit are not release package interfaces. |

The static header set currently copied by library generation should be packaged explicitly:

- `TensileTypes.h`
- `tensile_bfloat16.h`
- `tensile_float8_bfloat8.h`
- `KernelHeader.h`
- `ReductionTemplate.h`
- `memory_gfx.h`

The wheel should not include:

- `Tensile/bin/*` launcher files; console scripts should replace them.
- `Tensile/Tests/**`; tests belong in source, CI, or explicit test artifacts, not the runtime wheel.
- `rocisa/` source or `rocisa/build`; `rocisa` is a separate package.
- C++ executables or shared libraries unless we deliberately create a platform wheel that owns that binary contract.
- source-checkout Invoke task files such as `tasks.py`; packaged workflows should be normal `tensilelite` subcommands.
- local build outputs such as `build/`, `build_tmp/`, `CMakeCache.txt`, and `install_manifest.txt`.
- broad legacy `MANIFEST.in` globs such as `recursive-include Tensile ...` or `recursive-include rocisa ...`.
- CMake helper files from `Tensile/Source` unless a supported packaged workflow still uses them.

This boundary matters because package contents become a contract. If we accidentally ship tests, build directories, old launchers, partial native source trees, or compiled tools with unclear ABI expectations, downstream users will start depending on them.

## Implementation Plan

The migration should happen in stages so reviewers can validate each boundary independently.

1. Define the binary artifact contract.
   - Decide how `tensilelite` declares a compatible `rocisa` dependency.
   - Decide whether released `rocisa` depends on ROCm/TheRock `libstinkytofu` or vendors it for any package channel.
   - Add import-time or CLI startup diagnostics that fail clearly when `_rocisa` or its native dependencies cannot load.
   - Preserve the current `Tensile` benchmark/tuning command contract under `tensilelite run`.
   - Install `tensilelite-client` under `ROCM_PATH` as the version-matched artifact used by `tensilelite run`.
   - Make package installation search `ROCM_PATH` for `tensilelite-client` and fail clearly when it is missing.
   - Repeat the same client lookup at `tensilelite run` startup so broken or relocated installs fail with a direct diagnostic.
   - Keep `cpu-gemm-driver` as a source/CI test-only artifact, excluded from release packages.

2. Rename the Python package surface.
   - Change the distribution name from `tensile` to `tensilelite`.
   - Add the `tensilelite` import namespace.
   - Move new public imports and docs to `tensilelite.*`.
   - Convert internal imports away from `Tensile.*` over the migration.

3. Add a real CLI.
   - Add `tensilelite.cli`.
   - Add `tensilelite/__main__.py` so `python -m tensilelite ...` works.
   - Expose `create-library`, `logic`, and `run` as the supported command set.
   - Make `run` use the `ROCM_PATH` `tensilelite-client` discovery path, with `--prebuilt-client` as an explicit override.
   - Keep legacy `Tensile*` console-script aliases out of the default wheel; implement them only in `tensilelite-tensile-compat`.
   - Avoid adding a second command family such as `tensilelite-create-library`.

4. Stop relying on checkout-relative paths.
   - Replace constants such as `ROOT_PATH`, `SOURCE_PATH`, and `CUSTOM_KERNEL_PATH` with resource helpers.
   - Use `importlib.resources` for packaged files.
   - Add helpers such as `copy_static_headers(output_dir)`, `read_custom_kernel(name)`, `list_custom_kernels()`, and `load_default_known_bugs()`.
   - Make `tensilelite logic --check-all` use the packaged `known_bugs.yaml` by default, with `--known-bugs` retained as an override.
   - Replace hard-coded default client paths with `ROCM_PATH` artifact discovery and an explicit `--prebuilt-client` override.

5. Update the hipBLASLt build.
   - Install version-matched `rocisa` and `tensilelite` into the build Python environment.
   - Use editable installs for local development and wheels for packaging tests.
   - Validate `import tensilelite` and `import rocisa` with the exact Python executable CMake will use.
   - Replace `HIPBLASLT_PYTHON_COMMAND` path injection with package execution.
   - Convert device-library generation to `python -m tensilelite create-library ...`.
   - Convert logic validation to `python -m tensilelite logic ...`.

6. Clean up tests and tooling.
   - Add installed-wheel tests that do not put the source tree on `PYTHONPATH`.
   - Update `tox`, `pytest`, coverage, and pre-commit paths from `Tensile` to `tensilelite`.
   - Keep legacy-name tests only where explicitly testing compatibility behavior.

## Source Areas That Need Attention

These are the main places where the old source-tree contract or binary-artifact ambiguity shows up today:

| Area | What needs to change |
| --- | --- |
| `pyproject.toml` | Rename distribution metadata, dependencies, package discovery, scripts, and test config. Add the `rocisa` dependency policy. |
| `MANIFEST.in` | Replace broad legacy includes with an explicit allowlist. |
| `rocisa/pyproject.toml` and `rocisa/CMakeLists.txt` | Make the `_rocisa` and `libstinkytofu` runtime contract clear for each package channel. |
| top-level `CMakeLists.txt` | Replace raw source-tree test artifact installs with package installs or intentional test artifacts; install `tensilelite-client` under `ROCM_PATH`; keep `cpu-gemm-driver` out of release artifacts. |
| `cmake/hipblaslt_python.cmake` | Stop constructing a `PYTHONPATH`-patched command environment. |
| `cmake/HipBLASLtCodegen.cmake` | Call `python -m tensilelite logic` and `python -m tensilelite create-library`. |
| `device-library/extops/CMakeLists.txt` | Either package ext-op generators as `tensilelite` modules/subcommands or declare them out of scope for the packaged path. |
| `tensilelite/tasks.py` | Keep Invoke tasks as source/CI tooling rather than runtime package interfaces. |
| `tensilelite/README.md` | Document the supported package commands and source-development workflow. |
| `scripts/run_tensile_logic_check.py` | Call the installed package API or `python -m tensilelite logic`. |
| `Tensile/Tensile.py` | Remove checkout/build-tree default client paths from the public packaged path. |
| `Tensile/ClientWriter.py` and `Tensile/GenerateSummations.py` | Stop shelling out to `Tensile/bin/TensileCreateLibrary`; use the new dispatcher or an in-process API. |
| `TensileCreateLibrary.copyStaticFiles` | Copy headers from package resources instead of `SOURCE_PATH`. |
| custom-kernel loading | Read `CustomKernels/*.s` through package resources. |
| tests under `Tensile/Tests/**` | Import `tensilelite.*` unless the test is specifically about compatibility. |

## Compatibility

There are two separate compatibility packages.

The default `tensilelite` wheel should expose only the new public names. From the first packaged release, `import tensilelite` should work and `import Tensile` should fail unless a separate compatibility package is installed. The default wheel should also avoid installing legacy `Tensile*` console-script aliases.

Legacy command and import compatibility should live in a separate opt-in package such as `tensilelite-tensile-compat`, tightly versioned against `tensilelite`. That package can provide the `Tensile` import namespace and the original `Tensile*` entry points as wrappers around `tensilelite`, with clear deprecation warnings and a removal date.

Keeping `import Tensile` inside the canonical `tensilelite` wheel would recreate the naming conflict. It can also load the same implementation twice under different module names, which breaks module globals, class identity, and caches.

Binary compatibility also needs an explicit policy. `tensilelite` and `rocisa` should be tested and consumed as a matched pair from the same ROCm/TheRock release or artifact manifest. Mixing a `tensilelite` wheel from one build with a `rocisa` extension or `libstinkytofu` from another should be treated as unsupported unless we add and test a broader compatibility guarantee.

## Validation

This refactor should not be considered done until it passes package-level, binary-loading, and hipBLASLt-level checks:

- Build an sdist and wheel with `python -m build`.
- Inspect package contents and fail CI if the wheel contains `Tensile/bin`, `Tensile/Tests`, `rocisa/build`, `CMakeCache.txt`, `install_manifest.txt`, generated build artifacts, or accidental compiled binaries.
- Verify the required static headers, `CustomKernels/*.s`, and packaged `known_bugs.yaml` resource are present.
- Install version-matched local `rocisa` and the `tensilelite` wheel into a clean venv and verify `import tensilelite` and `import rocisa`.
- Verify `pip install tensilelite` fails clearly when `ROCM_PATH` does not contain the version-matched `tensilelite-client`.
- Verify `pip install tensilelite` succeeds when `ROCM_PATH` contains the version-matched `tensilelite-client`.
- Verify `_rocisa` can load its native dependencies in the clean environment that represents the target ROCm/TheRock artifact set.
- Run `tensilelite --help`, `tensilelite create-library --help`, and `tensilelite logic --help`.
- Run `tensilelite run --help` and a smoke path that proves it finds the installed `tensilelite-client` or honors an explicit `--prebuilt-client` override.
- Verify `import Tensile` fails in an environment with only the default `tensilelite` wheel installed.
- Run `tensilelite-tensile-compat` smoke tests while that package exists.
- Run unit tests and installed-wheel tests without adding the source tree to `PYTHONPATH`.
- Run a scoped hipBLASLt device-library build with the package-based generator path.
- Run installed-artifact tests that prove client-using benchmark categories can
  resolve `tensilelite-client` under `ROCM_PATH`; client-free categories must
  not require it. `cpu-gemm-driver` should not be required by release-artifact
  tests.
- Smoke-test a downstream environment, such as GEKO, that depends on `tensilelite` without path injection after installing the matching `rocisa` package.
- Add grep gates for accidental new uses of `Tensile/bin`, `python -m Tensile`, `from Tensile`, `import Tensile`, `PYTHONPATH`, `sys.path` mutation, `ROOT_PATH`, `SOURCE_PATH`, `CUSTOM_KERNEL_PATH`, and checkout-relative client paths.

## Proposed Next Step

First, agree on the package boundary, CLI names, compatibility window, and ROCm/TheRock binary artifact contract.

Then implement the migration in this order:

1. `rocisa` and compiled-artifact contract.
2. `tensilelite` package metadata and CLI.
3. CMake cutover to package execution.
4. Import/resource cleanup.
5. Installed-wheel and installed-artifact validation.
6. Separate `tensilelite-tensile-compat` package for legacy imports and entry points.
