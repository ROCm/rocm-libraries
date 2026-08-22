# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repository is

`rocm-libraries` is a super-repo (monorepo) that consolidates many previously-standalone ROCm library
repositories into one tree, to unify CI/build/test workflows. Each library still largely behaves like an
independent project with its own build system, test suite, and (often) its own `CLAUDE.md`.

**Always check for a `CLAUDE.md` in the specific project directory you're working in** (e.g.
`projects/rocsparse/CLAUDE.md`, `projects/hipblaslt/CLAUDE.md`, `shared/rocroller/CLAUDE.md`) — several
projects already have detailed guidance (build flags, test invocation, architecture) that takes precedence
over the generic info here. As of this writing, per-project `CLAUDE.md` files exist for: `rocsparse`,
`composablekernel`, `hipblaslt` (+ its `tensilelite` subdir), `hipdnn` (+ `tools/DescriptorGenerator`,
`tools/dnn-benchmarking`), `rocroller`, and `dnn-providers/miopen-provider`. `rocblas` instead has
`.cursorrules` with equivalent content.

## Repository layout

```
projects/<name>/     One library per directory; each previously its own GitHub repo with its own package.
shared/<name>/       Code used as a dependency by multiple libraries but not released as its own package
                     (rocroller, tensile, mxdatagenerator, origami, ctest, primbench, stinkytofu).
dnn-providers/       hipDNN backend plugins (miopen-provider, hipblaslt-provider, hip-kernel-provider) and
                     their integration tests — these bridge hipdnn to the concrete math libraries.
cmake/               Root superbuild CMake modules and toolchain files.
test/therock/        Cross-component smoke tests (test_<project>.py) run against a TheRock install, driven
                     by test_runner.py.
docs/                Monorepo-level docs: CI, gardening/revert process, migration process, triage checklist.
.github/             CI workflows, CODEOWNERS, repo-sync automation for the migration period.
```

Project names are the standardized/released package names (not the old repo names) — e.g. `hipblas-common`,
not `hipBLAS-common`.

Most `projects/*` directories are **not yet wired into the root superbuild** (see below) and are still built
and tested exactly as they were as standalone repos — `cd projects/<name>` and follow that project's own
build docs (`README.md`, `install.sh`/`rmake.py`, etc.) or its `CLAUDE.md` if present.

## Root superbuild (CMake)

The root `CMakeLists.txt` is a superbuild that conditionally `add_subdirectory()`s a fixed allow-list of
components, gated by `ROCM_LIBS_ENABLE_COMPONENTS` (default `"all"`, which expands to `DEFAULT_COMPONENTS`):

- **Supported today** (in `AVAILABLE_COMPONENTS`): `mxdatagenerator`, `rocroller`, `origami`,
  `hipblas-common`, `hipblaslt`, `rocprim`, `rocrand`, plus opt-in DNN components `hipdnn`,
  `hipdnn-integration-tests`, `miopen`, `miopen-provider`, `hipblaslt-provider`, `hip-kernel-provider`,
  `hipdnn-samples`.
- **Not yet supported** (in `UNSUPPORTED_COMPONENTS`): requesting these fails fast with a `FATAL_ERROR` in
  `CMakeLists.txt` — e.g. `tensile`, `composablekernel`, `rocblas`, `rocfft`, `rocsparse`, `hipsparse`,
  `hipcub`, `hipfft`, `hiprand`, `hipsolver`, `hipsparselt`, `rocsolver`, `rocthrust`. Do not try to make
  these buildable from the root without first checking whether that's an intentional, tracked limitation.

Adding a new component to the superbuild means: adding it to `AVAILABLE_COMPONENTS`, adding an
`add_subdirectory_with_message(COMPONENT ... PREFIX_PATH projects|shared EXPECT_TARGET ...)` block (this
helper wraps `add_subdirectory()` with a `CMAKE_MESSAGE_CONTEXT` scope and a fatal error if the expected
target wasn't produced), and wiring any cross-component shared third-party dependency declarations (see
`rocm_libs_declare_shared_deps()` / `cmake/modules/shared_third_party.cmake`, used to avoid `hipdnn`/`miopen`
both independently fetching `nlohmann_json`).

Basic superbuild usage:

```bash
cmake -B build -S . -D CMAKE_INSTALL_PREFIX=/opt/rocm -D CMAKE_PREFIX_PATH=/opt/rocm
cmake --build build
cmake --install build
```

Or via presets (see `CMakePresets.json` for the full list — one per supported component/combination, e.g.
`hipblaslt`, `rocprim`, `rocrand`, `miopen-with-provider`, `hipdnn-providers-all`):

```bash
cmake --list-presets=configure
cmake --preset <name>
cmake --build --preset default
```

To build an arbitrary subset directly: `-D ROCM_LIBS_ENABLE_COMPONENTS="mxdatagenerator;rocroller"`.

[TheRock](https://github.com/ROCm/TheRock) is the preferred system for a full superbuild across components
and is what `test/therock/` tests validate against; the root CMake superbuild here is a lighter-weight,
partial alternative.

## Working on a single project (the common case)

For day-to-day development on one library, work exactly as you would have in its standalone repo:

```bash
cd projects/<name>   # or shared/<name>
# build/test using that project's own scripts/CMakeLists.txt (see its README.md / CLAUDE.md)
```

Sparse-checkout is recommended when only touching one or two projects (see `CONTRIBUTING.md` for full
instructions):

```bash
git sparse-checkout init --cone
git sparse-checkout set projects/rocblas shared/tensile
```

## Branching and this checkout

`develop` is the mainline branch upstream projects sync against. Branches under `gpuep-releases/` (e.g. the
currently checked-out `gpuep-releases/therock-7.14`) are downstream integration branches for AMD's TheRock
build system, layering Windows/MSVC support on top of upstream ROCm. If a change looks unusual for a
Linux-only ROCm library (MSVC static runtime, Control Flow Guard, `amdhip64.lib` workarounds, os-agnostic
`getenv`, etc.), it's very likely there to support this Windows/TheRock track — check recent history on the
same branch across sibling projects before assuming it's a mistake.

The long-term plan (post-7.0) is trunk-based development directly off `develop`; until then, changes
continue syncing through each project's pre-monorepo model (`develop` -> `staging` -> `mainline` -> `release`).

## Pre-commit hooks

Config is in `.pre-commit-config.yaml` (trailing-whitespace/EOF/YAML checks, `black` for Python,
`clang-format` for C/C++, plus local hooks for hipDNN flatbuffers regen and MIOpen gtest naming). **Most
projects are excluded by default** — see the `exclude:` block at the top of that file. Only opt a project in
after cleaning up its existing violations incrementally (see `CONTRIBUTING.md` § "Opting a Project into
Pre-commit Checks"); don't just delete a project's exclusion line as a drive-by fix.

```bash
pre-commit install               # one-time, wires into git commit
pre-commit run --all-files       # full repo
pre-commit run --files $(git ls-files projects/<name>)   # single project
```

## Code review / ownership

`.github/CODEOWNERS` maps paths to review teams (e.g. `/projects/rocsparse/` -> `@ROCm/sparses-reviewers`).
PRs get auto-labeled and routed to reviewers based on changed files — keep changes scoped to the relevant
project directory where possible so routing stays accurate.
