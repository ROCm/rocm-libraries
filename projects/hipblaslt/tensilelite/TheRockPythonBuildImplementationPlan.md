<!-- Copyright Advanced Micro Devices, Inc., or its affiliates. -->
<!-- SPDX-License-Identifier: MIT -->

# TensileLite Python Build Implementation Plan: TheRock

Status: Current implementation plan
Decision record: `PythonBuildGrillingDecisions.md` Q115–Q129

## Final artifact flow

```text
TheRock build
  ├── blas_lib: hipBLASLt runtime and device libraries
  └── blas_test (non-Windows testing builds)
        ├── tensilelite-client
        ├── canonical and compatibility wheels
        ├── raw rocisa and copied package tests
        └── fresh-venv artifact test execution
```

The client is test-only for now. Windows does not build, stage, bind, or test it,
while Windows device generation continues through the canonical wheel and
build-tree rocisa. Q129 records the eventual `rocm[libraries]` destination but
does not change this current artifact flow.

## Implementation

- Keep the `blas` artifact's project-owned TensileLite requirements provisioning
  before CMake; wheel build/install remains `--no-deps`.
- Restore TheRock's non-Windows, `THEROCK_BUILD_TESTING`-gated
  `TENSILELITE_ENABLE_CLIENT` setting.
- Slice `libexec/hipblaslt/tensilelite/**` into `blas_test`, not `blas_lib`, and
  install it through CMake's `tests` component only in the test-artifact path.
- Keep the canonical and compatibility wheels in the non-Windows test artifact.
  The runner selects exactly one of each with matching wheel versions, then
  installs canonical before compatibility without querying a client version.
- Keep client-free test categories client-free. The existing `ffm-quick`
  benchmark category requests and validates the staged client only when it first
  needs a client path.

## Validation

- Update TheRock runner unit tests for wheel-pair discovery, phase ordering, and
  client-free installation.
- Update artifact-structure tests to require the client in non-Windows
  `blas_test` and reject it in `blas_lib`; assert no client on Windows.
- Retain Linux reconstructed-artifact wheel coverage and existing Windows
  build/packaging coverage without adding a Windows client-test lane.
