# ADR 0001: The package version is hipBLASLt's version

Status:  Accepted
Decision:  Review A, Decision 1 (AIHPBLAS-5274)

## Context

The prototype hard-coded version `0.1.0`. The bindings are a thin layer over one hipBLASLt build, and a solution index, a default heuristic, or a bug is only meaningful for that build. A separate bindings version would tell a user nothing about which hipBLASLt they have, and the two numbers would drift.

There's a second trap. hipBLASLt's version doesn't change when its kernels are retuned. Spike 3 (AIHPBLAS-5626) loaded two nightlies that both report 1.5.0 (`100500`); only the git revision told them apart. Prep A found that one such retune moved 76% of the solution indices.

## Decision

- The import name stays `hipblaslt`.
- `hipblaslt.__version__` and the package metadata version are both hipBLASLt's `MAJOR.MINOR.PATCH`, from the `hipblaslt-version.h` of the hipBLASLt the bindings are built against.
  - The package metadata comes from `skbuild_plugins/hipblaslt_version_provider.py`, a scikit-build-core dynamic-metadata provider. It searches `hipblaslt_ROOT`, `ROCM_PATH`, `ROCM_HOME`, then `/opt/rocm`.
  - `CMakeLists.txt` stops the build if `find_package(hipblaslt)` found a different version, so the two can't disagree.
- When the first `Handle()` in a process is created, the bindings compare the loaded hipBLASLt's version and git revision (`hipblasLtGetVersion`, `hipblasLtGetGitRevision`) with the ones compiled in (`HIPBLASLT_VERSION_*`, where `TWEAK` is the git revision). They warn with `HipblasLtLoadWarning` when either differs. Review A recorded this as "at import"; it moved to the first handle because both calls need one.

## Consequences

- A wheel's name says which hipBLASLt it was built for, and `pip list` shows it.
- Two bindings builds against two nightlies have the same version string. The git revision in the import-time check is what separates them, and the warning names both.
- The runtime check needs a handle, and `hipblasLtCreate` calls `exit()` when one of its allocations fails, with no device (ROCM-33629) or with device memory exhausted. Creating a handle at import would put that risk, and a GPU allocation, in every process that imports the package. So the check rides on the first handle the caller creates anyway, and a process that never creates one is never checked.
- The provider uses the standard `[[tool.dynamic-metadata]]` table, which needs scikit-build-core 1.1.2 or later. The older `tool.scikit-build.metadata` form is deprecated, and it only accepts local plugins with `experimental = true`.
