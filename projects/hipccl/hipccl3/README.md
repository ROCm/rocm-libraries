# hipCCL (hipccl3 layout)

This is the unified hipCCL project: rocPRIM, hipCUB, rocThrust, and libhipcxx built,
versioned, and packaged as a single project, mirroring the role
[NVIDIA/cccl](https://github.com/NVIDIA/cccl) plays for CUB, Thrust, and libcudacxx.

> [!IMPORTANT]
> This is a first-pass scaffold living inside `rocm-libraries` at
> `projects/hipccl/hipccl3/`, and stays there - the earlier plan to extract
> hipCCL into its own `ROCm/hipCCL` repository (see
> [`../../../docs/hipccl-repository-split-proposal.md`](../../../docs/hipccl-repository-split-proposal.md))
> has been dropped; that document is kept for historical context only.
> A sibling directory, `projects/hipccl/hipccl2/`, exists purely for backward
> compatibility (a loose, unbound copy of rocPRIM/hipCUB/rocThrust as they exist
> in `rocm-libraries` today) and is intentionally *not* unified the way this
> directory is. Eventually only one of the two will remain.

## Structure

```
hipccl3/
  rocprim/        # rocPRIM, tip of rocm-libraries develop (same commit as hipccl2)
  hipcub/         # hipCUB, tip of rocm-libraries develop (same commit as hipccl2)
  rocthrust/      # rocThrust, tip of rocm-libraries develop (same commit as hipccl2)
  libhipcxx/      # ROCm/libhipcxx @ 5ac455d737937ba2dfd1a4e85ad13f19a775f692 (amd-develop)
  cmake/          # shared cmake modules/toolchains, copied from rocm-libraries root
  CMakeLists.txt  # unified superbuild (see HIPCCL_BUILD_* options below)
  CMakePresets.json
  docs/           # hipCCL-level docs; each component keeps its own docs/ too
  .github/        # CI scaffold ported from rocm-libraries' .github (see .github/README.md)
```

`rocprim`, `hipcub`, and `rocthrust` are currently on the same commit as their
`rocm-libraries` (`develop`) counterparts and as the `hipccl2` copies. `libhipcxx`
does not yet track the same "CCCL 3.0" alignment as the other three - it was
imported as a starting point and is expected to be merged forward along with the
others once the CCCL 3.0 rebase work happens.

## Building

```sh
cmake --preset hipccl-all
cmake --build build
cmake --install build
```

This installs headers under `<prefix>/include/hipccl/<component>` (e.g.
`/opt/rocm/include/hipccl/thrust`, `/opt/rocm/include/hipccl/rocprim`) and uses
one unified `HIPCCL_VERSION` (currently a placeholder `0.1.0`, see
`CMakeLists.txt`) for all of rocprim/hipcub/rocthrust.

To build a single component instead, use the matching preset (`rocprim`,
`hipcub`, `rocthrust`) or set `HIPCCL_BUILD_ROCPRIM`/`HIPCCL_BUILD_HIPCUB`/
`HIPCCL_BUILD_ROCTHRUST` directly.

**libhipcxx is not yet wired into this superbuild** (`HIPCCL_BUILD_LIBHIPCXX`
exists as an option but errors out if enabled). It's a header-only project with
its own lit-based test harness that doesn't fit the `add_subdirectory()` pattern
the other three use; see the `NOTE(hipccl3)` comments in `CMakeLists.txt` for
what's left to do.

## Known first-pass gaps

- **libhipcxx build integration**: present in-tree, not yet buildable/testable
  from the root `CMakeLists.txt`.
- **Unified versioning**: `VERSION_STRING` is forced to `HIPCCL_VERSION` for
  rocprim/hipcub/rocthrust via a CACHE-variable override (each of their
  `CMakeLists.txt` was given a small guard for this). libhipcxx's own versioning
  has not been looked at.
- **Unified packaging**: done - see [`../README.md`](../README.md#hipccl3-unified-build-versioning-and-packaging)
  for how the single `hipccl` CPack package and the `find_package(hipccl)`
  CMake config work. License aggregation there is still a best-effort
  placeholder pending real legal review.
- **CI**: see `.github/README.md` - the workflows here are a staged copy, not
  yet live (GitHub Actions only reads `.github/` at the repository root, and
  this directory is currently nested inside `rocm-libraries`).
- **Math CI**: Math CI (Jenkins, `math-ci.amd.com`) has no in-repo config
  anywhere in `rocm-libraries` (it's configured entirely on the Jenkins side),
  so there is nothing to port here; new jobs would need to be created there
  once hipCCL exists as its own repository.
