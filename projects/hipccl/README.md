# hipCCL

Working area for consolidating rocPRIM, hipCUB, rocThrust, and libhipcxx into a
single hipCCL project, mirroring the role [NVIDIA/cccl](https://github.com/NVIDIA/cccl)
plays for CUB, Thrust, and libcudacxx. 

This directory temporarily has two parallel copies while that transition is
in progress:

- **[`hipccl2/`](hipccl2/)** - a loose, unbound compatibility snapshot of
  rocPRIM/hipCUB/rocThrust exactly as they exist in `rocm-libraries` today.
  Nothing at its root binds the three together.
- **[`hipccl3/`](hipccl3/)** - the forward-looking unified hipCCL project:
  common root-level CMake, docs, and CI scaffolding across rocPRIM, hipCUB,
  rocThrust, and libhipcxx, with a single version and a single install layout
  (`<prefix>/include/hipccl/<component>`).

Both currently pin rocPRIM/hipCUB/rocThrust to the same commit (tip of
`rocm-libraries`' `develop` as of this writing); `hipccl3` additionally
includes libhipcxx, pinned to `ROCm/libhipcxx@5ac455d737937ba2dfd1a4e85ad13f19a775f692`
(`amd-develop`). `hipccl3`'s rocPRIM/hipCUB/rocThrust copies are expected to
be merged forward to align with upstream CCCL 3.0 at a later time; libhipcxx
was brought in as a starting point for that same effort.

The original `projects/rocprim`, `projects/hipcub`, `projects/rocthrust`
directories are untouched and remain fully functional - this is additive, not
a cutover. Removing them (and repointing everything that currently depends on
those paths) is a separate, later step.

`hipccl3`'s copies of rocPRIM, hipCUB, and rocThrust carry their real,
preserved commit history (via `git filter-repo`), not just a flat snapshot -
`git log`/`git blame` at their new paths show the same history they had at
`projects/rocprim` etc., no `--follow` flag required.

## hipccl3: unified build, versioning, and packaging

`hipccl3` isn't just three projects sitting in the same directory - its root
`CMakeLists.txt` makes them build, version, and package as one unit, the way
NVIDIA/cccl's `cuda-cccl` does for CUB/Thrust/libcudacxx. This section
documents how that works, since it required patching rocPRIM/hipCUB/
rocThrust's own build files in a few places, not just adding a wrapper on top.

### Unified install layout

`CMAKE_INSTALL_INCLUDEDIR` is overridden to `include/hipccl` (forced via
`CACHE ... FORCE`, set *before* rocPRIM/hipCUB/rocThrust's own
`add_subdirectory()` calls). All three already reference
`${CMAKE_INSTALL_INCLUDEDIR}` in their `install()` rules rather than a
hardcoded path, so this one override is sufficient - headers land under
`<prefix>/include/hipccl/rocprim`, `<prefix>/include/hipccl/hipcub`,
`<prefix>/include/hipccl/thrust` (rocThrust's own install rule already
appends `/thrust`).

`CMAKE_INSTALL_PREFIX` is likewise set to `/opt/rocm` *before* `project()`,
matching what rocPRIM/hipCUB already do in their own `CMakeLists.txt` -
`project()` sets this cache variable to the platform default (`/usr/local`)
first if nothing has claimed it yet, and a later `set(... CACHE ...)` without
`FORCE` is a no-op once the cache entry exists. Ordering, not the value
itself, is what makes this work.

### Unified versioning

`VERSION_STRING` is forced via a `CACHE STRING ... FORCE` at the hipccl3 root,
set to the single `HIPCCL_VERSION` (currently a placeholder `0.1.0`). Each of
rocPRIM/hipCUB/rocThrust's own `CMakeLists.txt` originally did an
unconditional `set(VERSION_STRING "4.7.0")` - each now has a
`if(NOT DEFINED VERSION_STRING) ... endif()` guard around that line, so the
hipccl3-level override actually takes effect instead of being silently
overwritten. Standalone builds of each project are unaffected (the guard is a
no-op when nothing predefines `VERSION_STRING`).

### Unified packaging (`make package` / `cpack`)

**The problem this solves:** `rocm_create_package(...)` (from the shared
`rocm-cmake` toolset) triggers `include(CPack)`, which is only meaningful
once per top-level CMake project. rocPRIM's own `CMakeLists.txt` already knew
this and guarded its packaging block behind
`if(ROCPRIM_PROJECT_IS_TOP_LEVEL)` (true only when built standalone). hipCUB's
and rocThrust's `CMakeLists.txt` did **not** have that guard - each
unconditionally called `rocm_create_package()`, so building all three
together in one configure meant whichever ran last (rocThrust, since it's
`add_subdirectory()`'d last) silently won, and hipCUB's package definition
was discarded with no warning.

**The fix:** hipCUB's and rocThrust's own `CMakeLists.txt` were given the same
top-level guard rocPRIM already had
(`if(CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)`), wrapping two
things:
- Their own `rocm_create_package(...)` call (skipped when nested).
- Their internal `rocprim-dev`/`rocprim-devel` cross-package dependency
  declarations (also skipped when nested - rocPRIM's headers are bundled into
  the *same* package now, so there's no separate `rocprim-dev` package left
  to depend on).

hipccl3's root `CMakeLists.txt` then declares **one** unified
`rocm_create_package(NAME hipccl ...)` call, re-declaring the one dependency
that's actually real (the HIP runtime version constraint, previously only
declared inside rocPRIM's own now-skipped block).

Standalone builds of rocPRIM/hipCUB/rocThrust (outside hipccl3) are
completely unaffected - the guards' `else()`/unguarded branches are
byte-for-byte the original logic.

**License aggregation:** `hipccl3/LICENSE` combines rocPRIM's MIT license,
hipCUB's BSD-3-Clause license, and rocThrust's Apache-2.0 license into one
file with clear per-component sections, and `CPACK_RPM_PACKAGE_LICENSE` is set
to `"MIT and BSD and ASL 2.0"`. **Both are best-effort placeholders, not
verified legal/compliance declarations** - get them reviewed before treating
this as more than a starting point.

### Unified CMake package config (`find_package(hipccl)`)

This is a *separate* mechanism from CPack packaging above - it controls what
`find_package(hipccl)` resolves to (a new `hipccl-config.cmake`, installed to
`<prefix>/lib/cmake/hipccl/`), not what `make package` produces. It mirrors
NVIDIA/cccl's `find_package(CCCL COMPONENTS [Thrust] [CUB] [libcudacxx])`.

rocPRIM/hipCUB/rocThrust's own `rocm_export_targets()` calls (which each
project already had) are **untouched** - they keep installing their own
independent `rocprim-config.cmake`/`hipcub-config.cmake`/
`rocthrust-config.cmake` files unconditionally, to their own distinctly-named
folders (`lib/cmake/rocprim/`, etc.), exactly as before. Unlike
`rocm_create_package()`, these don't collide with each other (different file
names, no shared global state), so `find_package(rocprim)`,
`find_package(hipcub)`, `find_package(rocthrust)` all keep working completely
standalone, with or without hipccl3.

The new `hipccl-config.cmake` (template at
`hipccl3/cmake/package/hipccl-config.cmake.in`) sits alongside those and adds
a unified entry point:

```cmake
find_package(hipccl REQUIRED)                       # all three components
target_link_libraries(my_target PRIVATE hipccl::hipccl)

find_package(hipccl REQUIRED COMPONENTS hipcub)     # just one
target_link_libraries(my_target PRIVATE hip::hipcub)
```

It works by delegating: for each requested component, it calls
`find_dependency(<component> CONFIG)`, which resolves to that component's own
already-existing config file above - it does not redefine any targets itself.
When all three components are found, it additionally defines a
`hipccl::hipccl` `INTERFACE` target linking `roc::rocprim` + `hip::hipcub` +
`roc::rocthrust`, mirroring `CCCL::CCCL`.

A standalone smoke test exercising this end-to-end (a real `hipcub::DeviceReduce::Sum`
GPU kernel launch, built against an *installed* hipccl) lives at
[`hipccl3/smoketest/`](hipccl3/smoketest/).

### A drive-by fix found along the way

While touching hipCUB's install rules, a pre-existing (harmless, cosmetic)
bug was found and fixed: `hipcub/hipcub/CMakeLists.txt` had
`DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/` (note the trailing slash) where
rocPRIM's equivalent line has no trailing slash. This produced a
barely-noticeable `include//hipcub` in install logs under the old, unmodified
`CMAKE_INSTALL_INCLUDEDIR`, but became a more visibly-wrong
`include/hipccl//hipcub` under hipccl3's override. Fixed by removing the
redundant slash (in the hipccl3 copy only).

### Known gaps

- **License aggregation** needs real legal/compliance review (see above).
- **libhipcxx** isn't part of any of this yet - not wired into the build, the
  CPack package, or the `find_package(hipccl)` config. All three would need
  updating once it is.
- **Maintainer email** (`hipccl-maintainer@amd.com`) is a placeholder, not a
  real assigned address.
- **No Windows support `rmake.py`-equivalent** exists for hipccl3 yet (each of rocPRIM/hipCUB/
  rocThrust has its own convenience wrapper script today); unifying those is
  separate follow-up work.
