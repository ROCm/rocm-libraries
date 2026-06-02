# Proposed TheRock issue: publish `_dev`/`_test` artifacts for build-only third-party libraries

> Status: draft, ready to file against [ROCm/TheRock](https://github.com/ROCm/TheRock).
> Captured 2026-06-02 from the TensileLite shared-library / hipBLASLt build-enablement investigation.

## Title

Publish `_dev`/`_test` artifacts for build-only third-party libraries (msgpack-cxx, googletest,
yaml-cpp, …) so downstream components can be built and tested from artifacts, not just run

## Problem

Several third-party libraries are declared as `EXCLUDE_FROM_ALL` / `BACKGROUND_BUILD`
`therock_cmake_subproject`s, statically consumed by their dependents, and have **no
`artifact-*.toml` descriptor**. As a result they are never published in any artifact component
(`{name}_dev_*`, `{name}_test_*`, `{name}_lib_*`, …). Confirmed examples: `msgpack-cxx`,
`googletest`, `yaml-cpp` (the "no-dep third-party libraries" group in `third-party/CMakeLists.txt`).

Consequence: a consumer that fetches CI artifacts — or uses an artifact-assembled image such as the
manylinux dev environment — gets a tree that can **run** ROCm and **consume** the published
`roc::*` packages, but **cannot configure/build** a component that `find_package`s one of these
TPLs.

Concrete failure: hipBLASLt hard-requires msgpack at configure time

```
# projects/hipblaslt/CMakeLists.txt:244
find_package(msgpackc-cxx CONFIG REQUIRED NAMES msgpackc-cxx msgpack)
```

On an artifact-built image (e.g. the `rocm-libs-bump-e3b` / manylinux dev images) this fails with
`Could NOT find msgpack-cxx`, because no msgpack artifact exists in any component to satisfy it. The
same gap blocks building **and** running tests against fetched artifacts.

### Evidence

Dry-run of the artifacts for a recent multi-arch run (the one that built our dev image) shows full
`_dev`/`_lib`/`_test` coverage for published nodes — `amd-llvm`, `core-hip`, `core-runtime`, `blas`
(rocBLAS/hipBLASLt ship here as `blas_dev`/`blas_lib`/`blas_test`), `host-blas` (OpenBLAS),
`composable-kernel`, sysdeps, etc. — but **no** `msgpack*`, `gtest*`/`googletest*`, `yaml*`, or
`tensile*` artifacts at all.

```bash
python3 build_tools/fetch_artifacts.py --run-id <RUN_ID> --dry-run | grep -iE 'msgpack|gtest|yaml'
# (no output)
```

## Why this matters

TheRock already supports satisfying build-topology nodes from prebuilt artifacts
(`fetch_artifacts.py` → `buildctl.py bootstrap` writes `<stage>.prebuilt` markers →
`therock_subproject.cmake` skips those builds). That makes "fetch artifacts, build only the leaf you
care about" a first-class flow. But it can only satisfy nodes that are *published*. The build-only
TPLs are exactly the nodes that aren't — so any downstream build that touches them is forced back to
a from-source build of TheRock's third-party tree.

## Request

Publish `_dev` (headers + `*-config.cmake`) and `_test` artifacts for these build-only TPLs — i.e.
give them `artifact-*.toml` descriptors in an appropriate group (e.g. `third-party-libs`) — so:

- the "extract artifacts over a build dir → prebuilt node" flow can satisfy these `find_package`
  requirements, and
- artifact-assembled dev images carry the dev surface needed to build + test downstream components.

Goal: be able to **build and run tests** from artifacts, not only run.

## Workaround in use meanwhile

Until these are published, downstream builds reproduce TheRock's exact pin for the missing TPL — see
the companion proposal *therock-minimal-tpl-provisioning-without-full-sources.md* — building e.g.
`therock-msgpack-cxx` (7.0.0) into its `dist/` prefix and adding it to `CMAKE_PREFIX_PATH`.
