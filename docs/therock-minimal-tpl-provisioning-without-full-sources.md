# Proposed TheRock issue: build a TPL/component subset without fetching all sources

> Status: draft, ready to file against [ROCm/TheRock](https://github.com/ROCm/TheRock).
> Captured 2026-06-02 from the TensileLite shared-library / hipBLASLt build-enablement investigation.

## Title

Support building a TPL/component subset without fetching all sources (HIP-version override +
documented minimal-provisioning mode)

## Use case

A downstream consumer wants to build/install a *subset* of TheRock subprojects from source — for
example, provisioning a couple of third-party libraries that aren't published as artifacts (e.g.
`msgpack-cxx`; see the companion proposal *therock-publish-build-only-tpls-as-artifacts.md*), or
iterating on a single component on top of an artifact-bootstrapped tree — **without cloning the full
source set** (`rocm-systems`, `llvm-project`, `rocm-libraries`).

This is already *almost* possible:

- Third-party subprojects fetch their own pinned tarballs via `therock_subproject_fetch` — they do
  **not** require git submodules.
- With `THEROCK_ENABLE_ALL=OFF`, the `compiler/` subdir (entirely under `if(THEROCK_ENABLE_COMPILER)`)
  and the `core/` subdir (each subproject gated on `THEROCK_ENABLE_*`) no-op, so neither
  `llvm-project` nor `rocm-systems` sources are needed to build, say, `therock-msgpack-cxx`.

## The one blocker

The root configure hard-`FATAL`s if the HIP `VERSION` file is missing:

```cmake
# CMakeLists.txt:196
set(VERSION_PATH "${THEROCK_ROCM_SYSTEMS_SOURCE_DIR}/projects/hip/VERSION")
if(NOT EXISTS "${VERSION_PATH}")
  message(FATAL_ERROR "Could not find HIP VERSION file: ${VERSION_PATH}")
endif()
```

This forces a `rocm-systems` checkout purely to read one version file, even when nothing being built
needs `rocm-systems` sources. (By contrast, the ROCm version is read from `version.json`, which is
always present in the TheRock root.)

The only current workaround is to point `THEROCK_ROCM_SYSTEMS_SOURCE_DIR` at a hand-made stub
directory containing a fake `projects/hip/VERSION`:

```bash
mkdir -p /tmp/rsys-stub/projects/hip && printf '7\n14\n0\n' > /tmp/rsys-stub/projects/hip/VERSION
cmake -S . -B build -GNinja \
  -DTHEROCK_ENABLE_ALL=OFF \
  -DTHEROCK_ROCM_SYSTEMS_SOURCE_DIR=/tmp/rsys-stub \
  -DTHEROCK_AMDGPU_FAMILIES=gfx94X-dcgpu
cmake --build build --target therock-msgpack-cxx
# -> build/third-party/msgpack-cxx/dist/lib/cmake/msgpack-cxx/   (consumable prefix)
```

This works and produces `msgpack-cxx` 7.0.0 with **zero submodule fetch**, but it is a hack: the
stubbed version is silently wrong, which is fragile if any consumed value actually depends on the
HIP version.

## Requests (any one helps; together they make it robust)

1. **Allow supplying the HIP version directly**, e.g. `-DTHEROCK_HIP_VERSION=X.Y.Z` (or
   major/minor/patch cache vars). Only `FATAL` on the missing `VERSION` file when neither the source
   nor an override is present. Optionally fall back to `version.json`.
2. **Document (and CI-test) a minimal-provisioning mode**: `THEROCK_ENABLE_ALL=OFF` +
   `THEROCK_AMDGPU_FAMILIES=<family>` + a `therock-<tpl>` target builds that TPL with no submodule
   fetch.
3. **Relax the dist-target requirement for target-neutral TPLs.** `USE_DIST_AMDGPU_TARGETS`
   subprojects currently `FATAL` ("requires dist AMDGPU targets but none were set",
   `cmake/therock_subproject.cmake:1637`) unless a **family** is set — a bare
   `THEROCK_AMDGPU_TARGETS=gfxNNN` is not enough. A target-neutral, host-only TPL build arguably
   shouldn't need any GPU target at all.

## What we're doing meanwhile

Stubbing `hip/VERSION` + `THEROCK_ENABLE_ALL=OFF` + `THEROCK_AMDGPU_FAMILIES=<family>` to build
`therock-msgpack-cxx` into its `dist/` prefix and adding that to `CMAKE_PREFIX_PATH` for the
downstream hipBLASLt configure. Happy to contribute the version-override patch if the approach is
acceptable.

## Notes / verification

Confirmed on `2026-06-02`: the recipe above configured (`rc=0`) and built `therock-msgpack-cxx`
inside an artifact-built ROCm image with no `rocm-systems` / `llvm-project` / `rocm-libraries`
sources present, producing `…/third-party/msgpack-cxx/dist/lib/cmake/msgpack-cxx/msgpack-cxx-config.cmake`.
