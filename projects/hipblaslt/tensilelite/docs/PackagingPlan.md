<!-- Copyright Advanced Micro Devices, Inc., or its affiliates. -->
<!-- SPDX-License-Identifier: MIT -->

# TensileLite ROCm-Coupled Packaging Implementation Plan

> **Superseded for Python build integration:** use
> `PythonBuildGrillingDecisions.md` and `PythonBuildImplementationPlan.md` for
> the canonical wheel, client-binding, device-generation, and artifact-test
> design. The phases below are historical where they conflict with those files.

Status: Implemented for TensileLite; rocisa packaging explicitly deferred
Plan date: 2026-07-29
Decision record: `PackagingDecisions.md`

## Objective

Package the TensileLite generator as `tensilelite`, make installed Python
packages and ROCm-owned native artifacts the only supported execution boundary,
and cut hipBLASLt's build from checkout-relative imports to the installed
package interface without changing generated kernels, logic formats, or the
C++ host ABI.

## Target Interfaces

### Python distributions

- `tensilelite` is a pure Python generator wheel versioned as
  `<component>+rocm<release>`.
- `tensilelite-tensile-compat` is an optional, exactly matched command-wrapper
  wheel available for the defined migration window.
- TensileLite and its compatibility wheel require Python 3.10 or newer.
- A properly packaged, importable `rocisa` is an external prerequisite.

### Public imports and commands

- `import tensilelite`
- `import rocisa`
- `tensilelite create-library ...`
- `tensilelite logic ...`
- `tensilelite run ...`
- `python -m tensilelite ...` with identical dispatch

No default or compatibility wheel provides `import Tensile`.

### ROCm native layout

Use GNUInstallDirs/platform equivalents for this logical layout:

```text
$ROCM_PATH/
  .info/version
  libexec/hipblaslt/tensilelite/tensilelite-client
```

Windows uses the corresponding ROCm SDK runtime directory for the client.

## Phase 1: Preserve Decisions and Prepare Existing Code

1. Keep `Public.md` as the original proposal and maintain
   `PackagingDecisions.md` as the accepted-decision source of truth.
2. Carry forward the current `importlib.resources` implementation and tests.
3. Complete resource conversion before the namespace move:
   - make static-header and custom-kernel access resource-only;
   - make packaged known bugs the `logic --check-all` default;
   - make a missing explicit known-bugs override an error;
   - remove production dependence on `ROOT_PATH`, `SOURCE_PATH`, and
     `CUSTOM_KERNEL_PATH`.
4. Refactor run, create-library, and logic handlers to accept `argv` directly
   and return exit status, allowing one dispatcher without modifying
   `sys.argv`.
5. Replace internal subprocess calls to `Tensile/bin/TensileCreateLibrary`
   with in-process APIs.
6. Separate the existing hardcoded generator compatibility version from the
   Python distribution version.

## Phase 2: TensileLite Runtime Validation

1. Treat rocisa as an independently supplied dependency and require only that
   `import rocisa` succeeds. Do not change its build backend, binary layout,
   Python ABI, versioning, or native dependency policy in this migration.
2. Install `tensilelite-client` as a normal ROCm runtime component at the fixed
   `libexec` path.
3. Gate `cpu-gemm-driver` behind test builds and exclude it from install rules.
4. Make `tensilelite` import verify that rocisa imports and that the installed
   client exists before exposing the package.
5. Remove `--prebuilt-client` and checkout/build-tree client defaults.

## Phase 3: Python Package and CLI Cutover

1. Physically rename the Python package directory from `Tensile` to
   `tensilelite`.
2. Mechanically update Python imports while limiting replacements to the Python
   namespace; do not alter C++ Tensile names, generated logic keys, or kernel
   naming.
3. Add `tensilelite.cli` and `tensilelite.__main__` with lazy subcommand
   dispatch and unchanged handler argument contracts.
4. Expose one console script named `tensilelite` from the canonical wheel.
5. Move AMax, layer-normalization, softmax, and ext-op library generation into
   `tensilelite._extops`, with `argv`-accepting internal entry points and no
   public console scripts.

## Phase 4: Packaging Metadata and Contents

1. Make PEP 517 metadata derive the target ROCm release exclusively from the
   selected root's `.info/version`.
2. Emit the TensileLite distribution version and declare `rocisa` without
   choosing rocisa's versioning policy.
3. Keep the generator compatibility constant independent from distribution
   metadata.
4. Reduce default runtime requirements to:
   - `rocisa`;
   - PyYAML;
   - msgpack;
   - joblib 1.4 or newer;
   - packaging;
   - NumPy;
   - filelock.
5. Keep fallback/runtime capabilities available as optional extras:
   - `profile` installs yappi;
   - `hip-query` installs hip-python;
   - `orjson`, `ujson`, and `simplejson` each install the named JSON backend.
   Preserve the JSON fallback order through the standard library, while keeping
   CMake, nanobind, setuptools, Invoke, and other build/development packages out
   of runtime extras.
6. Replace broad `MANIFEST.in` rules with an explicit resource allowlist and
   package exclusions.
7. Build both sdist and wheel from the ROCm-versioned metadata path.

## Phase 5: hipBLASLt Build Integration

1. Introduce `HIPBLASLT_TENSILELITE_PYTHON_MODE=BUILD|SYSTEM`:
   - `BUILD` creates a build-local ROCm artifact stage and private Python
     environment;
   - `SYSTEM` validates an installed TensileLite wheel/client and importable
     externally supplied rocisa.
2. Keep `HIPBLASLT_BUNDLE_PYTHON_DEPS` independent as a pre-existing rocisa
   build concern. Default it off, opt in from the rocisa-only preset, and ensure
   TensileLite's `BUILD|SYSTEM` mode does not enable it, depend on a local
   `_rocisa` target, or stage rocisa-owned native artifacts.
3. In build-local mode:
   - build/stage `tensilelite-client` first;
   - create the fixed directory layout and `.info/version` under the build
     tree;
   - create a private Python 3.10+ environment without implicit network access;
   - install TensileLite editably and inherit a proper rocisa from the selected
     Python environment;
   - make the environment/stage target a dependency of every Python codegen
     target.
4. Replace code-generation commands with:
   - `python -m tensilelite logic`;
   - `python -m tensilelite create-library`;
   - installed internal ext-op modules.
5. Remove checkout-directory calculation, `PYTHONPATH`, source known-bugs paths,
   and source launcher dependencies.
6. Keep only required environment variables such as `ROCM_PATH`, toolchain
   `PATH`, and sanitizer settings.
7. Require Python 3.10 when a build path executes TensileLite; keep host-only
   builds independent and remove the unrelated rocBLAS version edit from this
   series.
8. Replace raw Python test-artifact copies with the two TensileLite-owned
   wheels and the staged client layout. Do not build a rocisa wheel or install
   rocisa source, tests, extensions, or native dependencies from this path.

## Phase 6: Compatibility and Downstream Migration

1. Add `tensilelite-tensile-compat` with exact dependency on the matching
   canonical wheel.
2. Provide warning wrappers for existing legacy commands, including:
   - `Tensile`;
   - `TensileCreateLibrary`;
   - `TensileLogic`;
   - benchmark, merge, retune, update, summation, and logic-conversion tools;
   - `TensileVerifyStinkyElfText`;
   - `TensileGetPath`.
3. Print the removal release once per process on stderr and preserve delegated
   arguments and exit status.
4. Do not add any `Tensile` import package to the compatibility wheel.
5. Update GEKO and other in-repo callers to resolve installed commands/APIs
   rather than `tensilelite/Tensile/bin` paths.
6. Update README, tox, pytest, coverage, pre-commit, and test-category paths.

## Test and Acceptance Plan

### Package contents and resolution

- Build ROCm-tagged sdists and wheels in a clean staging directory.
- Verify the complete local versions and exact `Requires-Dist` coupling.
- Verify the TensileLite wheel declares rocisa as an external dependency without
  embedding a premature rocisa release policy.
- Verify every optional fallback is represented by `Provides-Extra` and a
  conditional `Requires-Dist` entry in wheel metadata.
- Compare wheel resources with the tracked header and custom-kernel source sets.
- Fail if a canonical wheel contains `Tensile`, tests, legacy launchers, native
  objects, CMake caches/manifests, build directories, rocisa source, or Invoke
  tooling.

### Runtime/import behavior

- Test explicit `ROCM_PATH`, platform defaults, missing roots, missing/malformed
  `.info/version`, exact matches, and release mismatches.
- Test missing/broken rocisa imports and missing clients while treating rocisa's
  internal loader diagnostics as opaque dependency failures.
- Verify `import Tensile` fails with both canonical and compatibility wheels.

### CLI and resources

- Test console and `python -m` help, argument forwarding, and exit codes for all
  three canonical commands.
- Run create-library and logic smoke tests from a wheel-only environment with no
  checkout on `sys.path`.
- Test default and overridden known-bugs behavior.
- Verify every compatibility command warns and delegates correctly.

### Build and integration

- Configure and build both build-local and system Python modes with no
  `PYTHONPATH`.
- Confirm host-only hipBLASLt configuration does not acquire a new Python floor.
- Run a scoped device-library build including default ext-ops.
- Verify the installed TensileLite tree contains `tensilelite-client` but not
  `cpu-gemm-driver`; rocisa artifacts are outside this package's install rules.
- Run existing Python unit/common tests and C++ host tests.
- Compare representative generated library metadata and kernel outputs with
  pre-migration results.
- Smoke-test a GEKO/downstream environment without checkout-path injection.
- Add targeted grep gates for new production uses of legacy imports, launchers,
  `PYTHONPATH`, `sys.path` mutation, and removed root constants.

## Delivery Sequence

1. Decision/resource/handler cleanup and native install foundations while the
   legacy source interface still operates.
2. Atomic namespace, CLI, package metadata, CMake, ext-op, and test cutover.
3. Compatibility wheel, release-index publication, installed-artifact CI,
   downstream migration, and documentation before the first packaged release.

All three sequences must land before release. The canonical wheel must never
temporarily publish both `Tensile` and `tensilelite` namespaces.

## Implementation Results

Implemented on 2026-07-29 with the following local evidence:

- CMake host-only `BUILD`, device `BUILD`, and device `SYSTEM` configurations
  completed successfully with the in-tree rocisa build disabled; their target
  graphs contained no `_rocisa` or `stinkytofu` target.
- A scoped gfx942 device-library target validated one filtered logic file and
  generated/assembled/linked its single kernel through the package-based CMake
  commands.
- The integrated Release target built/staged `tensilelite-client`, created the
  private Python environment, and imported TensileLite with the externally
  supplied rocisa successfully. The staged tree contained no rocisa-owned
  extension or shared library.
- Runtime installation contained `tensilelite-client` and no
  `cpu-gemm-driver`; the test-artifact installation contained only the two
  TensileLite-owned wheels and the client, with no rocisa package or native
  artifact.
- ROCm-tagged TensileLite and compatibility sdists built successfully, and both
  wheels rebuilt from those sdists.
- Canonical wheel metadata requires Python 3.10 or newer and declares plain
  `Requires-Dist: rocisa`, with no rocisa version or local-version constraint.
- Wheel metadata publishes `profile`, `hip-query`, `orjson`, `ujson`, and
  `simplejson` extras with conditional requirements, so every optional runtime
  fallback remains explicitly installable without enlarging the default set.
- Both wheels passed the forbidden-content checker. In the canonical wheel,
  six headers, all 119 custom kernels, known bugs, and ductile defaults were
  present; legacy `Tensile`, tests, launchers, build artifacts, native binaries,
  rocisa source, and cached bytecode were absent.
- A clean installed-artifact environment passed 40 focused package/resource/
  CLI/runtime tests outside the checkout.
- Python tests passed:
  - 2,928 non-characterization unit tests, with 230 expected skips;
  - 2,822 characterization tests, with 10 expected skips and one pre-existing
    unused snapshot reported under the warning policy;
  - rocisa remained an externally supplied prerequisite; its source and
    packaging tests are not changed by this migration;
  - 2 compatibility-wrapper tests;
  - 235 GEKO tests, with 15 expected integration skips.
- Missing client and ROCm-version mismatch diagnostics were verified from
  installed wheels.
- Every tracked file in the rocisa subtree matches the pre-migration baseline;
  rocisa packaging changes from the initial implementation were removed.

rocisa ABI and cross-Python testing are deferred to rocisa's packaging work.
