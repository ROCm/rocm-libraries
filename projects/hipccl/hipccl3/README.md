# hipCCL (hipccl3 layout)

This is the unified hipCCL project: rocPRIM, hipCUB, rocThrust, and libhipcxx built,
versioned, and packaged as a single project, mirroring the role
[NVIDIA/cccl](https://github.com/NVIDIA/cccl) plays for CUB, Thrust, and libcudacxx.

> [!IMPORTANT]
> This directory lives in `rocm-libraries` at `projects/hipccl/hipccl3/`, and
> stays there; hipCCL is not being extracted into its own repository.
> A sibling directory, [`../hipccl2/`](../hipccl2/), is a frozen pre-CCCL 3.x
> compatibility snapshot of rocPRIM/hipCUB/rocThrust. Eventually only one of
> the two will remain.

## Structure

```
hipccl3/
  rocprim/        # rocPRIM 5.1.0
  hipcub/         # hipCUB 5.1.0 (tracks CCCL 3.1.4)
  rocthrust/      # rocThrust 5.1.0
  libhipcxx/      # git submodule: ROCm/libhipcxx (amd-develop), see .gitmodules
  cmake/          # shared cmake modules/toolchains, copied from rocm-libraries root
  CMakeLists.txt  # unified superbuild (see HIPCCL_BUILD_* options below)
  CMakePresets.json
  docs/           # placeholder README only; the hipCCL docs live in ../docs/ and each component keeps its own docs/
```

rocPRIM, hipCUB, and rocThrust carry their own history and are developed in
this directory. The `hipccl2` copies are at 4.7.0.

## Building

hipCUB and rocThrust require libhipcxx 3.4 or later on the ROCm platform.
Provide it in one of two ways:

- Install libhipcxx so that `find_package(libhipcxx)` can locate it
  (`CMAKE_PREFIX_PATH` or `ROCM_ROOT`). This is the recommended way; see the
  known issue below.
- Build it from the submodule with `-DHIPCCL_BUILD_LIBHIPCXX=ON`.

```sh
# From projects/hipccl/hipccl3, with libhipcxx installed
cmake --preset hipccl-all
cmake --build build
cmake --install build

# Or build libhipcxx from the submodule instead. Initialize it from the repo root:
git submodule update --init projects/hipccl/hipccl3/libhipcxx
cmake --preset hipccl-all -DHIPCCL_BUILD_LIBHIPCXX=ON   # from projects/hipccl/hipccl3
```

`HIPCCL_BUILD_ALL` (and so the `hipccl-all` preset) does not build libhipcxx;
`HIPCCL_BUILD_LIBHIPCXX` is off unless you set it. The same applies to the
single-component presets.

> [!WARNING]
> **Known issue:** libhipcxx built from the submodule is used for the build only
> and is not installed. The installed hipCUB and rocThrust CMake packages declare
> a dependency on a `libhipcxx` package, so a project that calls
> `find_package(hipcub)`, `find_package(rocthrust)` or `find_package(hipccl)`
> needs libhipcxx 3.4 or later installed separately, in the same prefix or on its
> `CMAKE_PREFIX_PATH`. See [`../docs/install/build.rst`](../docs/install/build.rst).

This installs headers under `<prefix>/include/hipccl/<component>` (e.g.
`/opt/rocm/include/hipccl/thrust`, `/opt/rocm/include/hipccl/rocprim`) and uses
one `HIPCCL_VERSION` (set in `CMakeLists.txt`) for the hipCCL package itself;
rocprim/hipcub/rocthrust keep their own component versions.

To build a single component instead, use the matching preset (`rocprim`,
`hipcub`, `rocthrust`) or set `HIPCCL_BUILD_ROCPRIM`/`HIPCCL_BUILD_HIPCUB`/
`HIPCCL_BUILD_ROCTHRUST` directly.

For the full option list, see [`../docs/install/build.rst`](../docs/install/build.rst).

## Known gaps

- **libhipcxx packaging**: libhipcxx can be built from the superbuild, but it is
  not part of the unified `hipccl` CPack package or the `find_package(hipccl)`
  config.
- **Versioning**: `HIPCCL_VERSION` versions only the hipCCL umbrella package.
  rocprim/hipcub/rocthrust keep their own `VERSION_STRING`. libhipcxx's own
  versioning has not been looked at.
- **Unified packaging**: done - see
  [`../docs/migration-notes.md`](../docs/migration-notes.md#unified-packaging-make-package--cpack)
  for how the single `hipccl` CPack package and the `find_package(hipccl)`
  CMake config work.
