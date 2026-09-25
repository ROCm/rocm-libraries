# hipCCL

Working area for consolidating rocPRIM, hipCUB, rocThrust, and libhipcxx into a
single hipCCL project, mirroring the role [NVIDIA/cccl](https://github.com/NVIDIA/cccl)
plays for CUB, Thrust, and libcudacxx. 

This directory temporarily has two parallel copies while that transition is
in progress:

- **[`hipccl2/`](hipccl2/)** - a loose, compatibility snapshot of
  rocPRIM/hipCUB/rocThrust as they exist in `rocm-libraries` today. Nothing at
  its root binds the three together - each still configures, builds, and
  packages exactly as it always has when built standalone. It has picked up a
  small number of minimal, CMake-only patches (see
  [The `projects/hipccl` layout selector](#the-projectshipccl-layout-selector)
  below) so it can also be reached through the new root-level layout selector
  without breaking; the library/algorithm code itself is untouched.
- **[`hipccl3/`](hipccl3/)** - the forward-looking unified hipCCL project:
  common root-level CMake, docs, and CI scaffolding across rocPRIM, hipCUB,
  rocThrust, and libhipcxx, with a single version and a single install layout
  (`<prefix>/include/hipccl/<component>`).

A new root-level **[`CMakeLists.txt`](CMakeLists.txt)** (in this directory)
acts as a layout selector: it lets a single `cmake` invocation build either
layout via one flag, `HIPCCL_BUILD_LEGACY` (default `OFF`, meaning "build the
unified `hipccl3` project"). See
[The `projects/hipccl` layout selector](#the-projectshipccl-layout-selector)
for details.

Both currently pin rocPRIM/hipCUB/rocThrust to the same commit (tip of
`rocm-libraries`' `develop` as of this writing); `hipccl3` additionally
includes libhipcxx as a **git submodule** (see `.gitmodules`), pointed at
`ROCm/libhipcxx`'s `amd-develop` branch and pinned to whatever commit was the
tip of that branch when the submodule was added. `hipccl3`'s rocPRIM/hipCUB/
rocThrust copies are expected to be merged forward to align with upstream
CCCL 3.0 at a later time; libhipcxx was brought in as a starting point for
that same effort. `HIPCCL_BUILD_LIBHIPCXX` (default `OFF`) builds it as a
consumable `libhipcxx::libhipcxx` target within the superbuild - see the
NOT YET DONE note in [Known gaps](#known-gaps) for what that does and doesn't
cover yet.

The original `projects/rocprim`, `projects/hipcub`, `projects/rocthrust`
directories are untouched and remain fully functional - this is additive, not
a cutover. Removing them (and repointing everything that currently depends on
those paths) is a separate, later step.

`hipccl3`'s copies of rocPRIM, hipCUB, and rocThrust carry their real,
preserved commit history (via `git filter-repo`), not just a flat snapshot -
`git log`/`git blame` at their new paths show the same history they had at
`projects/rocprim` etc., no `--follow` flag required.

## CMake deduplication

`hipccl2` and `hipccl3` share the parts of their CMake infrastructure that
don't differ between the two layouts, rather than each carrying its own copy:

- **[`cmake/modules/`](cmake/modules/)** - `add_subdirectory_with_message.cmake`
  and `fetch_rocm_cmake.cmake`, byte-identical between the two layouts.
- **[`cmake/package/hipccl-config.cmake.in`](cmake/package/hipccl-config.cmake.in)**
  - the `find_package(hipccl)` config template (see
  [Unified CMake package config](#unified-cmake-package-config-find_packagehipccl)
  below); only differed by comment wording between the two copies.
- **[`LICENSE`](LICENSE)** - the aggregated MIT + BSD-3-Clause + Apache-2.0
  license text (see [License aggregation](#unified-packaging-make-package--cpack)
  below); likewise only differed by comment wording.

Both `hipccl2/CMakeLists.txt` and `hipccl3/CMakeLists.txt` reference these via
a `../` relative path (e.g. `${CMAKE_CURRENT_SOURCE_DIR}/../cmake/modules`),
so this is transparent to anything downstream - it doesn't change install
paths, package contents, or `find_package(hipccl)` behavior, just where the
CMake source files themselves live. Anything that's actually
layout-specific (e.g. `hipccl3`'s handful of otherwise-unused legacy
`cmake/modules/` and `cmake/toolchains/` files) stays put in its own layout's
folder, not shared.

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

Each of rocPRIM/hipCUB/rocThrust's own `CMakeLists.txt` (in hipccl3) also now
sets `CMAKE_INSTALL_INCLUDEDIR` to `include/hipccl` itself, the same way,
*before* its own `project()` call. This means a fully standalone build of one
of these copies (`cd hipccl3/rocprim && cmake .. && make install`, with no
superbuild involved) installs headers to the unified
`<prefix>/include/hipccl/rocprim` path too, rather than falling back to the
old `<prefix>/include/rocprim` layout. When built through hipccl3's root
instead, that root's own `CACHE ... FORCE` already claimed the cache entry
first, so each project's own copy of this line is a harmless no-op.

### Unified versioning

`VERSION_STRING` is forced via a `CACHE STRING ... FORCE` at the hipccl3 root,
set to the single `HIPCCL_VERSION` (currently a placeholder `0.1.0`). Each of
rocPRIM/hipCUB/rocThrust's own `CMakeLists.txt` originally did an
unconditional `set(VERSION_STRING "4.7.0")` - each now has a
`if(NOT DEFINED VERSION_STRING) ... endif()` guard around that line, so the
hipccl3-level override actually takes effect instead of being silently
overwritten.

The fallback value itself was also changed, from `"4.7.0"` to `"0.1.0"`
(matching `HIPCCL_VERSION`), so that a fully standalone build of one of these
hipccl3 copies (e.g. `cd hipccl3/rocprim && cmake ..`, with no superbuild
involved at all) still reports a hipccl-consistent version instead of
rocPRIM/hipCUB/rocThrust's own upstream version number. This does duplicate
the version number in four places (`hipccl3/CMakeLists.txt`'s
`HIPCCL_VERSION`, plus each of the three projects' own fallback) that must be
kept in sync manually until real unified-versioning infrastructure exists -
see Known gaps.

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

**The fix went further than just guarding it: standalone packaging is now
unconditionally disabled** for all three hipccl3 copies, not merely skipped
when nested. Building `hipcub`/`rocthrust`/`rocprim` from inside their own
`hipccl3/<component>` folder (`cmake .. && make package`) will no longer
produce a `rocprim`/`hipcub`/`rocthrust` package at all, standalone *or*
nested - only `cmake --install` works from inside those folders now. This was
a deliberate choice, not just a side effect of the nesting fix: a standalone
`hipcub` package built from `hipccl3/hipcub` would be a *different*,
differently-named artifact than the unified `hipccl` package hipccl3's root
produces, installable side-by-side by a package manager that has no idea the
two overlap - risking duplicate/conflicting installs of the same headers.
This mirrors upstream NVIDIA CCCL, where CUB/Thrust support standalone
configure/build for dev/CI purposes only, with no standalone packaging story
at all. To get a real, distributable package, build through hipccl3's root
`CMakeLists.txt`, which calls `rocm_create_package(NAME hipccl ...)` once,
unconditionally - re-declaring the one dependency that's actually real (the
HIP runtime version constraint, previously only declared inside rocPRIM's own
now-disabled block).

Each of the three projects' `rmake.py` was updated to match: the Windows
install path (`--target package --target install`) had `--target package`
dropped, since no `package` target exists in these copies anymore.

Standalone builds of rocPRIM/hipCUB/rocThrust **outside hipccl3** (i.e.
`rocm-libraries`' own `projects/rocprim`/`hipcub`/`rocthrust`, and
`hipccl2`'s copies) are completely unaffected - none of this touches those
files at all.

**License aggregation:** `LICENSE` (shared by `hipccl2` and `hipccl3` - see
[CMake deduplication](#cmake-deduplication) above) combines rocPRIM's MIT license, hipCUB's BSD-3-Clause license, and
rocThrust's Apache-2.0 license into one file with clear per-component
sections, and `CPACK_RPM_PACKAGE_LICENSE` is set to
`"MIT and BSD and ASL 2.0"`. **Both are best-effort placeholders, not
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
`cmake/package/hipccl-config.cmake.in`, shared with `hipccl2` - see
[CMake duplication](#cmake-deduplication) below) sits alongside those and
adds a unified entry point:

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

### Drive-by fixes found along the way

While touching hipCUB's install rules, a pre-existing (harmless, cosmetic)
bug was found and fixed: `hipcub/hipcub/CMakeLists.txt` had
`DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/` (note the trailing slash) where
rocPRIM's equivalent line has no trailing slash. This produced a
barely-noticeable `include//hipcub` in install logs under the old, unmodified
`CMAKE_INSTALL_INCLUDEDIR`, but became a more visibly-wrong
`include/hipccl//hipcub` under hipccl3's override. Fixed by removing the
redundant slash (in the hipccl3 copy only).

Building the superbuild with `-DBUILD_TEST=ON` also uncovered a target-name
collision: all three of rocPRIM/hipCUB/rocThrust's `test/CMakeLists.txt`
define an executable literally named `generate_resource_spec` (a helper each
project's `testing.md` documents running as `./generate_resource_spec
resources.json` to drive multi-GPU `ctest` scheduling), and that name also
used `${CMAKE_SOURCE_DIR}`/`${CMAKE_BINARY_DIR}` for its source path and
output directory - both of which only resolved correctly because each
project had always been the top-level project until hipccl3 existed. Nesting
all three in one configure failed outright
(`add_executable cannot create target "generate_resource_spec" because
another target with the same name already exists`). Fixed, in all three
hipccl3 copies, by:
- Renaming the CMake *target* per project
  (`rocprim_generate_resource_spec`, `hipcub_generate_resource_spec`,
  `rocthrust_generate_resource_spec`), while pinning `OUTPUT_NAME
  "generate_resource_spec"` so the produced binary's filename - the thing
  `testing.md` actually documents - is unchanged.
- Switching the source path to `${CMAKE_CURRENT_SOURCE_DIR}`, so it resolves
  to each project's own file regardless of nesting.
- Keeping the output directory at `${CMAKE_BINARY_DIR}` for standalone builds
  (unchanged from today), but using `${CMAKE_CURRENT_BINARY_DIR}` when
  nested, so the three nested builds don't overwrite each other's binary.
  hipCUB and rocThrust didn't previously have a `*_PROJECT_IS_TOP_LEVEL`-style
  flag to make that distinction (only rocPRIM did) - a minimal
  `HIPCUB_PROJECT_IS_TOP_LEVEL`/`ROCTHRUST_PROJECT_IS_TOP_LEVEL` flag was
  added to each, used only for this purpose.

### The `projects/hipccl` layout selector

[`projects/hipccl/CMakeLists.txt`](CMakeLists.txt) is a small router that lets
one `cmake` invocation build either layout, without changing how either
behaves when built directly:

```sh
# Default (HIPCCL_BUILD_LEGACY=OFF): build the unified hipccl3 project.
cmake -S projects/hipccl -B build

# Build the legacy hipccl2 layout instead (independent rocPRIM/hipCUB/rocThrust).
cmake -S projects/hipccl -B build -DHIPCCL_BUILD_LEGACY=ON
```

`cd hipccl3 && cmake ..` and `cd hipccl2/rocprim && cmake ..` (etc.) continue
to work exactly as before - this router is a purely additional entry point,
not a replacement for either.

Making this work correctly required two small companion fixes:

- **`hipccl3`'s unified packaging and `find_package(hipccl)` generation are
  now unconditional.** Both were previously guarded behind
  `if(CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)`, added
  speculatively "in case hipccl3 ever gets nested under something bigger."
  Once the layout selector exists, that condition is false even when the
  selector is explicitly told to build `hipccl3` (the selector's own root is
  now the outermost `CMAKE_SOURCE_DIR`), which would have silently disabled
  both features. Since `hipccl3`'s root `CMakeLists.txt` is never meant to be
  a mere sub-component of anything else, the guard was simply removed.
- **hipCUB and rocThrust in `hipccl2` gained a top-level packaging guard they
  didn't have before.** Only rocPRIM had one
  (`ROCPRIM_PROJECT_IS_TOP_LEVEL`); hipCUB and rocThrust called
  `rocm_create_package()` (which triggers `include(CPack)`) unconditionally.
  `include(CPack)` only supports one call per configure, so the
  `HIPCCL_BUILD_LEGACY=ON` route - which nests all three - would otherwise
  fail to configure at all. Both gained the same
  `HIPCUB_PROJECT_IS_TOP_LEVEL`/`ROCTHRUST_PROJECT_IS_TOP_LEVEL` guard
  rocPRIM already had, wrapped in `if(NOT DEFINED ...)` (matching a similar
  wrapper added to rocPRIM's own flag) so a parent can still explicitly force
  one back on, e.g.:
  `cmake -S projects/hipccl -B build -DHIPCCL_BUILD_LEGACY=ON -DROCPRIM_PROJECT_IS_TOP_LEVEL=ON`.
  Standalone builds of any of the three (`cd hipccl2/rocprim && cmake ..`)
  are unaffected - nothing pre-defines the flag, so it still auto-detects
  exactly as before, and packaging still happens by default.

One residual limitation: since rocPRIM, hipCUB, and rocThrust each still call
`rocm_create_package()` independently, only **one** of the three can have its
flag forced on in a given `HIPCCL_BUILD_LEGACY=ON` configure - forcing all
three on at once just reproduces the original multi-`include(CPack)`
collision. Configure/build/install work normally for all three regardless;
only simultaneous packaging of all three through the selector is out of
scope.

### Known gaps

- **License aggregation** needs real legal/compliance review (see above).
- **libhipcxx** isn't part of any of this yet - not wired into the build, the
  CPack package, or the `find_package(hipccl)` config. All three would need
  updating once it is.
- **Maintainer email** (`hipccl-maintainer@amd.com`) is a placeholder, not a
  real assigned address.
- **No Windows support `rmake.py`-equivalent** exists for hipccl3 or the
  `projects/hipccl` layout selector yet (each of rocPRIM/hipCUB/rocThrust has
  its own convenience wrapper script today); unifying those is separate
  follow-up work.
- **Version numbers are duplicated in four places** (see Unified versioning
  above) pending real unified-versioning infrastructure.
