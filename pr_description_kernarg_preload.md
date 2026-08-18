JIRA ID : AIHPBLAS-4194

## Corrected support statement

The custom-kernel source reader must retain
`.amdhsa_user_sgpr_kernarg_preload_*` on later ROCm major releases even
when `hipconfig` reports a build number below the historical ROCm-6
floor. Official ROCm 7.1 and 7.2 installations can report
`7.1.25424-...` and `7.2.26015-...`, respectively; both values are below
32650 even though they are later ROCm releases.

The directives are supported by the official ROCm 6.0 compiler line on
feature-enabled GPU families. The `6.0.32650` threshold is historical
hipBLASLt compatibility policy, not a complete capability proof: target
ISA, assembler identity, and firmware also matter.

## Semantic scope

`KernelWriterAssembly` intentionally changes behavior only for reported
versions whose major number is greater than 6 and whose patch is below
32650: the old inline predicate stripped the directives, while the shared
predicate retains them. `Solution.py` uses the negation of that shared
predicate, which is equivalent to its old unsupported-toolchain condition
for integer `SemanticVersion` fields.

## Compiler support history

- AMD's LLVM change
  [`1ecadb8368e`](https://github.com/ROCm/llvm-project/commit/1ecadb8368e903c297ae7bcae96620eb1173e7dc)
  added the assembler directives and kernel-descriptor fields for
  preloaded kernargs in September 2023. The official
  [`rocm-6.0.0`](https://github.com/ROCm/llvm-project/tree/rocm-6.0.0)
  source includes that support and its gfx90a/gfx94x regression tests.
- HIP recorded version `6.0.32650` on September 29, 2023, and hipBLASLt
  adopted that value as its 6.x floor on October 6. The historical guard
  did not link the value to a package manifest or compiler revision, so
  this PR does not present it as an independently proven first-supported
  build.
- [HIP issue #3881](https://github.com/ROCm/hip/issues/3881) documents the
  non-monotonic 7.1 and 7.2 `hipconfig` build values that exposed the
  original major-version bug.

## Toolchain limitation

`rocm_version` currently comes from the default `hipconfig`, while the
selected C++ compiler can come from another prefix or launcher. A robust
fix requires an explicit toolchain-identity contract or an assembler
capability probe; this PR does not add an unsound path-adjacency heuristic.

## Pending validation

- [ ] Targeted preload unit tests, including the `Solution` consumer,
  both preload directives, and real hipconfig-style parser inputs.
- [ ] Full TensileLite unit lane.
- [ ] Affected gfx942/gfx950 shared CI coverage.
