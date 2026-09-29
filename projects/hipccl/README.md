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

Each of rocPRIM/hipCUB/rocThrust's own `CMakeLists.txt` (in hipccl3) also now
sets `CMAKE_INSTALL_INCLUDEDIR` to `include/hipccl` itself, the same way,
*before* its own `project()` call. This means a fully standalone build of one
of these copies (`cd hipccl3/rocprim && cmake .. && make install`, with no
superbuild involved) installs headers to the unified
`<prefix>/include/hipccl/rocprim` path too, rather than falling back to the
old `<prefix>/include/rocprim` layout. When built through hipccl3's root
instead, that root's own `CACHE ... FORCE` already claimed the cache entry
first, so each project's own copy of this line is a harmless no-op.

### Versioning: umbrella vs components

hipCCL deliberately keeps **two independent version numbers**, and it is
important not to collapse them into one:

- **The hipCCL umbrella version** (`HIPCCL_VERSION`, currently a placeholder
  `0.1.0`) versions the unified CPack package and `find_package(hipccl)`.
- **The component API versions** (`VERSION_STRING`, currently `4.7.0` in
  rocPRIM/hipCUB/rocThrust) version `find_package(rocprim)`,
  `find_package(hipcub)`, and `find_package(rocthrust)`, exactly as they
  always have.

Neither root forces the components to adopt `HIPCCL_VERSION`.

**Why they must stay separate.** The plan is for `HIPCCL_VERSION` to
eventually track the CCCL version it corresponds to, so that
`find_package(hipccl 3.0.0)` lines up with `find_package(CCCL 3.0.0)`. But
rocPRIM/hipCUB/rocThrust have *already shipped* at 4.7.0. Pulling them down to
a 3.x umbrella version would be a **backwards version move on
already-released packages**: package managers would treat it as a downgrade,
and every downstream consumer with `find_package(rocprim 4.x REQUIRED)` would
break. Because the `hipccl` package name is brand new, the umbrella can start
at whatever number matches CCCL without regressing anything, while the
components keep moving forward on their own 4.x line.

hipCUB already uses this same pattern internally - its `VERSION_STRING`
(4.7.0) and its `HIPCUB_CCCL_VERSION_MAJOR/MINOR/PATCH` (2.8.2) are separate
values, because "my version" and "the CCCL version I correspond to" are
different things.

**A concrete bug this caused.** An earlier iteration *did* force
`VERSION_STRING` down to `HIPCCL_VERSION`, making rocPRIM install as `0.1.0`.
hipCUB and rocThrust ask for rocPRIM via
`find_package(rocprim ${MIN_ROCPRIM_PACKAGE_VERSION} ...)` with floors of
`4.1.0` and `4.0.0` respectively, so `0.1.0` failed the version check and both
silently fell back to downloading a separate rocPRIM copy:

```
-- No existing rocprim package meeting the minimum version requirement (4.1.0) was found.
```

Keeping the component versions on their real 4.x line makes those existing
floors work as intended, with no need to weaken them.

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
copies (both `hipccl2` and `hipccl3`), by:
- Renaming the CMake *target* per project
  (`rocprim_generate_resource_spec`, `hipcub_generate_resource_spec`,
  `rocthrust_generate_resource_spec`), while pinning `OUTPUT_NAME
  "generate_resource_spec"` so the produced binary's filename - the thing
  `testing.md` actually documents - is unchanged.
- Switching the source path to `${CMAKE_CURRENT_SOURCE_DIR}`, so it resolves
  to each project's own file regardless of nesting.
- Switching the output directory to `${PROJECT_BINARY_DIR}` (each project's
  own build subdir). This equals `${CMAKE_BINARY_DIR}` in the standalone
  case, so standalone behavior is unchanged, while nested builds each get
  their own non-colliding directory. An earlier pass instead branched on a
  `*_PROJECT_IS_TOP_LEVEL`-style flag and used `${CMAKE_CURRENT_BINARY_DIR}`
  for the nested case - one level too deep (this file's own build dir,
  e.g. `rocthrust/test`, rather than `rocthrust`), so nested builds produced
  the binary inside `<component>/test/` instead of `<component>/`. Since
  `${PROJECT_BINARY_DIR}` already matches standalone behavior on its own, no
  top-level flag is needed for this purpose at all: `ROCTHRUST_PROJECT_IS_TOP_LEVEL`
  was removed entirely (it had no other consumer), and `HIPCUB_PROJECT_IS_TOP_LEVEL`
  likewise (see below - hipCUB's copies never actually gated packaging on it
  either, unlike what the layout-selector section below currently claims).
  rocPRIM's `ROCPRIM_PROJECT_IS_TOP_LEVEL` is untouched - it has other,
  genuine consumers (test/benchmark config, packaging) - only its
  `generate_resource_spec` output directory changed.

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

Making this work correctly required a companion fix:

- **`hipccl3`'s unified packaging and `find_package(hipccl)` generation are
  now unconditional.** Both were previously guarded behind
  `if(CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)`, added
  speculatively "in case hipccl3 ever gets nested under something bigger."
  Once the layout selector exists, that condition is false even when the
  selector is explicitly told to build `hipccl3` (the selector's own root is
  now the outermost `CMAKE_SOURCE_DIR`), which would have silently disabled
  both features. Since `hipccl3`'s root `CMakeLists.txt` is never meant to be
  a mere sub-component of anything else, the guard was simply removed.

Nesting rocPRIM, hipCUB, and rocThrust together (in either layout) does *not*
hit the classic multi-`include(CPack)` collision one might expect, because
none of the three call `rocm_create_package()` from inside their own copy of
`CMakeLists.txt` at all - not even standalone. Each copy's Package section
explicitly disables it, unconditionally, regardless of
`ROCPRIM_PROJECT_IS_TOP_LEVEL`/`HIPCUB_PROJECT_IS_TOP_LEVEL`/`ROCTHRUST_PROJECT_IS_TOP_LEVEL`
(the last two of which have since been removed for having no other use - see
above). A standalone package built from inside e.g. `hipccl2/rocprim` would
be a different, differently-named artifact than the unified `hipccl` package
`hipccl2`'s (or `hipccl3`'s) own root `CMakeLists.txt` produces - installable
side-by-side by a package manager with no idea they overlap, risking
duplicate/conflicting installs of the same headers. Only the root
`CMakeLists.txt` of each layout calls `rocm_create_package()`, exactly once,
unconditionally; each component copy only supports configure/build/
`cmake --install`. rocm-libraries' own standalone `projects/rocprim`,
`projects/hipcub`, and `projects/rocthrust` are untouched and still package
normally on their own.

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
- **`HIPCCL_VERSION` is still the `0.1.0` placeholder.** Setting it to the
  real CCCL-aligned value belongs with the CCCL 3.0 merge-forward work (see
  [Versioning: umbrella vs components](#versioning-umbrella-vs-components)),
  since claiming `3.0.0` before the components actually have CCCL 3.0
  semantics would mislead downstream `#if CCCL_VERSION >= ...` checks. The
  `#define CCCL_VERSION HIPCCL_VERSION` compatibility glue downstream
  consumers have asked for should land at the same time.
