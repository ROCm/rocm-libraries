# Lane: correctness and code quality (`CORR<k>`)

Read every changed line of your owned files, and the outside code each change depends on.

## Correctness

- Trace each change from producer to consumer and from guard to caller to test. State the smallest input that breaks it.
- Error paths: every failure returns a `miopenStatus_t` or throws `miopen::Exception` (`MIOPEN_THROW`) as the surrounding code
  does, and the C API boundary converts it (`miopen::try_`); nothing is swallowed; partial state unwinds. Handles, descriptors,
  workspace buffers, streams, `hipModule_t`/program objects and external-library handles (rocBLAS, hipBLASLt, CK) are released on
  every failure path.
- Solvers (`projects/miopen/src/solver/`):
  - `IsApplicable` accepts exactly what the kernel supports: data types, layouts (NCHW, NHWC, NCDHW, NDHWC), groups, strides,
    dilations, padding, tensor sizes and 32-bit index limits, and the architecture (`StartsWith(device_name, "gfx…")`, wave size,
    XNACK, LDS limit). A solver that returns true for a problem its kernel mis-computes is the worst outcome here: the find step
    picks it silently.
  - `GetWorkspaceSize` covers what the invoker writes; the invoker uses the workspace it was given, not one it assumes.
  - Tunable solvers (declared in `projects/miopen/src/include/miopen/conv/solvers.hpp` and the other `*/solvers.hpp`): the
    perf config's `IsValid`, `SetNextValue`, `operator==` and serialization agree with `IsValidPerformanceConfig`;
    `GetDefaultPerformanceConfig` is valid for every applicable problem; a perf-config string from an existing perf-db still
    parses or is rejected cleanly.
  - Registration: a new solver is appended at the end of `SolverRegistrar` in `projects/miopen/src/solver.cpp`, a removed one is
    replaced with `++id`, and `SolverDbId()` is unique and unchanged for existing solvers ([../policy.md](../policy.md) *Review rules*).
  - The problem description and network config (`MakeNetworkConfig`, `Serialize`) encode every field the solver branches on;
    a missing field makes two different problems share a find-db record or a cached kernel.
- Kernels (`projects/miopen/src/kernels/`): compile definitions the solver passes match what the kernel reads; buffer and index
  arithmetic does not overflow 32 bits for the sizes `IsApplicable` allows; assembly kernels' metadata matches the launch.
- Find and databases: find-db, perf-db and user-db keys, the hybrid find path and immediate mode (`projects/miopen/docs/how-to/find-and-immediate.rst`)
  return the same solver for the same problem; a cached workspace size or solution is not trusted where the code now
  recomputes it.
- Numerics: accumulation type, overflow, alpha/beta handling, tolerance, NaN/Inf, denormals, FP8 and BF16 rounding
  (`MIOPEN_USE_RNE_BFLOAT16`, `MIOPEN_FP8_*`).
- Concurrency: shared state in the handle, the kernel cache, the binary cache and databases (file locks, `MIOPEN_USER_DB_PATH`),
  lazy initialization, handle and stream lifetimes.
- Backends: code that compiles under `MIOPEN_BACKEND=HIP` also compiles and behaves under `HIPNOGPU` (offline tuning, no device),
  and does not call the device there.
- Portability: Windows paths and DLL search, path separators, 32-bit sizes.

## Code quality

- Names and structure follow the surrounding code and `projects/miopen/.clang-format`; internals stay in `namespace miopen` and
  out of `projects/miopen/include/miopen/miopen.h`.
- Dead code, unused parameters, commented-out code, unexplained magic numbers, environment switches (`MIOPEN_DECLARE_ENV_VAR_*`)
  added without a default or a documented reason.
- Complexity that a simpler construct in the same codebase already solves.

Missing or weak tests are not findings here: hand them off to `TESTQ` (weak tests) or `TESTB` (missing coverage). Test registration in
`projects/miopen/test/gtest/test_categories.yaml` and gtest naming are owned by `BUILD`: hand them off, do not file them. Record
anything else for another lane under **Handoffs**; do not file it.
