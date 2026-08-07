<!-- Copyright Advanced Micro Devices, Inc., or its affiliates. -->
<!-- SPDX-License-Identifier: MIT -->

# TensileLite Python Build Implementation Plan: TheRock

Status: Proposed implementation plan  
Decision record: `PythonBuildGrillingDecisions.md` in the rocm-libraries
TensileLite source tree  
Baseline: `users/alvasile/tensilelite_packaging` at `3eec13b3a`

## Summary

This plan integrates the accepted TensileLite canonical-wheel contract into
TheRock. hipBLASLt remains the single owner of its internal Python/native build
graph; TheRock provisions inputs, selects the project, slices artifacts, and
orchestrates testing of the reconstructed artifact set.

The final artifact-test flow is:

```text
build hipBLASLt and release wheels
       |
       +--> blas_lib: production tensilelite-client
       |
       +--> blas_test: wheels, raw rocisa, copied installed-wheel tests
                                |
                                v
fresh test venv and reconstructed ROCm root
       |
       +--> install canonical wheel and run normal suite
       |
       +--> install compatibility wheel and run compat-only suite
```

The separate Windows reconstructed-artifact test lane remains deferred. The
Windows production build must still build and install the client, construct and
validate the canonical wheel, bind the build-tree client, and generate device
libraries through the installed canonical package.

## Public Integration Changes

| Interface | Final contract |
|---|---|
| Build requirements | The `blas` artifact declares the project-relative TensileLite requirements file through `python_requires`; one `configure_stage.py` rocm-libraries source-root selection resolves it and emits the matching CMake source argument. |
| hipBLASLt configuration | TheRock no longer disables `tensilelite-client` on Windows or when testing is off. |
| Runtime artifact | `blas_lib` owns the fixed `libexec/hipblaslt/tensilelite/` directory containing `tensilelite-client` on non-Windows and `tensilelite-client.exe` on Windows. |
| Test artifact | `blas_test` owns wheels, raw rocisa, copied installed-wheel tests excluding `unit/source_only/`, and test configuration, but no duplicate client. |
| Test execution | A thin TensileLite runner owns wheel installation and phase order; the generic pytest runner retains pytest policy. |
| Version selection | The packaged client's GPU-less `--version` is the expected version used to select artifact wheels. |
| Build-time binding | After installing the regular wheel, hipBLASLt configures the exact target client with the command-local loader/sanitizer environment in the build user's `~/.tensilelite/bindings/<installation-id>/client.json`; each supported generator command verifies that keyed binding. Plain pip installation does not require `ROCM_PATH`. |
| Wheel handoff | rocm-libraries builds and validates canonical and compatibility wheels in separate private staging directories, then copies each exact wheel and touches its target-specific completion stamp. TheRock consumes only those published paths, following its existing copy-then-stamp convention. |
| Wheel validation | Independent rocm-libraries validator modes enforce exact dependency, normalized-name, filename/tag, binding-module, and configuration-entry-point invariants before TheRock consumes either wheel. |
| Wheel filenames | hipBLASLt composes exact filenames during CMake configuration and registers all version-authority inputs for automatic regeneration; TheRock does not supply or interpret a build-time wheel-path manifest. |
| Wheel installation | hipBLASLt installs only the two configured, validated wheel files into its stage; TheRock does not consume or copy the whole release-wheel build directory. |
| Wheel build trigger | The existing `HIPBLASLT_INSTALL_TENSILELITE_TEST_ARTIFACTS=${THEROCK_BUILD_TESTING}` setting puts the aggregate two-wheel target in `ALL`; TheRock's normal build therefore creates both install inputs before its separate install step. |
| Native version rebuild | The same automatic CMake regeneration rewrites the generated client-version header and rebuilds `tensilelite-client`; a no-clean test proves its version remains equal to the rebuilt canonical wheel. |
| Binding isolation | The per-user key derives only from the exact resolved installed package directory. Reinstalling there intentionally reuses and overwrites the same slot; different wheel environments or editable worktrees resolve to different slots without modifying `.dist-info` or `RECORD`. |

## Commit Plan

### 1. `build(blas): enable TensileLite production prerequisites`

- Add the project-relative
  `projects/hipblaslt/tensilelite/requirements.txt` to the `blas` artifact's
  `python_requires`. Extend `configure_stage.py` with one explicit
  rocm-libraries source-root input, defaulting to TheRock's submodule; resolve
  the requirements path from it and emit the matching
  `-DTHEROCK_ROCM_LIBRARIES_SOURCE_DIR=<root>` argument. Callers select the root
  once, including conventional external and arbitrary local checkouts.
- Keep both rocm-libraries wheel-build commands on the selected Python with
  `--no-build-isolation --no-deps`; TheRock provisions requirements but does not
  permit either build to resolve or download wheel dependencies.
- Remove TheRock's test-only/Windows-disabled `TENSILELITE_ENABLE_CLIENT`
  override and rely on hipBLASLt's centralized device-build invariant. Keep the
  test-artifact option tied to `THEROCK_BUILD_TESTING`.
- Let that invariant build `tensilelite-host` as the required client's existing
  transitive prerequisite; do not add a TheRock-only client/host split.
- Do not pass `TENSILELITE_ENABLE_HOST=OFF` or
  `TENSILELITE_ENABLE_CLIENT=OFF` when device generation is enabled; hipBLASLt
  rejects either contradiction at configure time rather than overriding it.
- Rely on that invariant to build the in-tree `_rocisa` for every device build;
  do not satisfy generation from an externally importable rocisa or restore
  `HIPBLASLT_BUNDLE_PYTHON_DEPS` as a TheRock-controlled escape hatch.
- After hipBLASLt force-installs the canonical wheel, let its centralized graph
  configure the exact CMake client in the build user's keyed
  `~/.tensilelite/bindings` slot. This per-user build state is not installed,
  sliced into an artifact, or transferred to the fresh test job; artifact tests
  therefore exercise the fixed client below reconstructed `ROCM_PATH`.
- Retain `ROCISA_USE_STABLE_ABI=ON` and the existing graph-owned `amd-hip`
  toolchain; do not pass another ROCm version/root cache variable or add a
  duplicate root/toolchain preflight. The metadata helper's actual
  `.info/version` read and real CMake/generator consumers remain authoritative.
- Preserve hipBLASLt's target-derived stable-ABI discovery for this setting:
  Python 3.12+ with `Interpreter`, `Development.Module`, and
  `Development.SABIModule`. Client-only and host-only configurations do not
  inherit those extension requirements.
- Extend topology/configure-stage tests to prove math-libs emits the TensileLite
  requirement once on Linux and Windows and unrelated stages do not receive
  it.

### 2. `build(blas): ship tensilelite-client in the runtime artifact`

- Add `libexec/hipblaslt/tensilelite/**` to the hipBLASLt library component and
  remove it from the test component.
- Keep `share/hipblaslt/tensilelite/**` in `blas_test`, including wheels, raw
  rocisa, copied installed-wheel tests, compatibility tests, and configuration;
  exclude the genuine source/build tests under `unit/source_only/`.
- Before finalizing that slice, audit every selected test for checkout-root,
  adjacent-source, build-metadata, checkout-script, and source-relative resource
  dependencies. The audit records inputs for classification and does not itself
  require copying more source into `blas_test`.
- Keep tests of source/build machinery in source CI and out of `blas_test`.
  Tests retained as installed production coverage must use installed APIs or
  resources; do not enlarge the artifact with source/build inputs merely to make
  path-relative tests pass.
- Require genuine source/build tests to live under
  `tensilelite/Tests/unit/source_only/`. Exclude that directory as one unit from
  `blas_test` and assert it is absent, while source CI continues collecting it.
- Do not add a separate collection-only phase. Each normal category collects its
  configured installed paths with the checkout absent and treats collection
  errors as failures; preserve category-specific paths and options.
- Extend artifact-structure tests to prove `blas_lib` independently supplies
  the standard client, `blas_test` does not duplicate it, and flattening normal
  runtime plus test artifacts produces the accepted layout on Linux.
- Validate the production client pattern on Windows without adding the deferred
  Windows test-artifact slice, including the exact `.exe` filename in the fixed
  standard directory.

### 3. `test(tensilelite): run canonical and compatibility wheel phases`

- Refactor the generic `pytest_runner.py` into a callable `main`/phase API
  without changing its ownership of categories, markers, timeouts, workers,
  pytest invocation, or JUnit creation.
- Replace the legacy `test_tensilelite.py` body with a thin TensileLite phase
  runner and point the TensileLite test configuration at it.
- Construct the final test environment first: reconstructed `ROCM_PATH`,
  platform loader paths, `PATH`, and `PYTHONPATH` containing only the raw-rocisa
  parent. Reuse it for the client-version query, wheel installs, and both pytest
  phases.
- Query the production client's no-GPU `--version` as the expected artifact
  version. Discover exactly one canonical wheel and exactly one compatibility
  wheel matching it; reject zero, duplicate, wrong-name, or wrong-version
  candidates.
- Enforce Q078 on the version subprocess: pass the final environment, apply the
  five-second timeout, require zero status, one canonical stdout line and empty
  stderr, parse with `packaging.version.Version`, and require equality with the
  selected wheels. Test distinct launch/loader, timeout, signal, nonzero,
  stderr, missing/malformed/extra-output, and mismatch failures.
- Phase 1 force-reinstalls the canonical wheel with the active interpreter and
  `--no-deps`, then delegates the selected normal category to the generic runner
  and propagates its return code.
- Only after phase 1 succeeds, phase 2 runs
  `sys.executable -m pip install --force-reinstall --no-deps
  <exact-compatibility-wheel>` and delegates `compat/tests --run-compat`.
- The compatibility wheel no longer declares pandas: rocm-libraries replaces
  its sole production use in `GenerateSummations.py` with standard-library CSV
  parsing plus existing NumPy. TheRock does not provision pandas; compatibility
  tests import the real delegate and mock only expensive execution.
- Stop before compatibility on phase-1 failure, and return the compatibility
  phase's failure when it runs. Do not add required JUnit outputs, persistence,
  upload, or reporting; retain the generic runner's existing optional support.
- Add unit tests with temporary wheel metadata and mocked subprocesses for
  discovery, both complete active-interpreter pip argument lists, environment
  construction, phase ordering, fail-fast behavior, and return-code propagation.

### 4. `chore: bump rocm-libraries for the TensileLite build contract`

- After the rocm-libraries stack is finalized, update TheRock's
  `rocm-libraries` gitlink to the corresponding merged or reviewable SHA in an
  isolated commit.
- Keep this commit free of functional edits so rebases or SHA replacements do
  not disturb the reviewed build, artifact, and runner commits.

## Validation and Acceptance

- Run topology and `configure_stage.py` tests for Linux and Windows math-libs
  stages, verifying the project requirements are installed once. Cover the
  default submodule, conventional external checkout, arbitrary local override,
  and a missing file diagnostic containing the resolved path.
- Rely on the focused hipBLASLt raw-CMake and `hipblaslt-clients` configure tests
  for the centralized device/host/client invariant; do not duplicate that
  unit-level matrix in TheRock. The existing Linux and Windows math-libs builds
  remain its integration coverage.
- Run artifact descriptor/structure tests proving client ownership moved from
  `blas_test` to `blas_lib`.
- Run generic-runner regression tests and the new thin-runner tests, including
  zero/ambiguous wheels, version mismatch, canonical failure, and compatibility
  failure.
- Build `therock-artifacts` and `therock-dist` against the local
  rocm-libraries checkout and inspect the flattened production/test layout.
- Run the TensileLite component test from a fresh venv with no rocm-libraries
  checkout on the import path. Require the canonical phase to complete before
  compatibility is installed.
- Require Linux and Windows math-libs builds to complete the canonical package
  generation path. Windows acceptance stops before reconstructed-artifact
  transfer/testing.

## Landing Order and Deferred Work

- Develop the first three commits against the local rocm-libraries checkout
  using `THEROCK_ROCM_LIBRARIES_SOURCE_DIR`.
- Finalize or land the rocm-libraries implementation first, then create the
  isolated gitlink commit.
- Do not add TensileLite requirements to TheRock's root requirements and do not
  remove or modify its existing `joblib`/`msgpack` compatibility bridge or the
  separate `requirements-test.txt` block in this change. Also do not propagate
  `THEROCK_PACKAGE_VERSION` into hipBLASLt, restore a global ambient `ROCM_PATH`,
  or move raw rocisa into a production artifact.
- Defer source/channel/SHA-bearing versions, exact source identity, rocisa wheel
  packaging, compatibility removal, and Windows artifact-test transfer to their
  separately recorded follow-ups.
