# hipDNN Feature Release Test Plan

This document is the **hipDNN milestone / release verification** plan: the procedures and expectations for validating a release build and confirming it is ready to ship. It is **not** the per-PR development workflow; for the day-to-day testing bar during development, see [Testing](../testing/TESTING.md) and its [Development Workflow](../testing/TESTING.md#development-workflow).

> [!IMPORTANT]
> ⚠️ **All prerequisites and tests in this document must pass for a successful release.**

---

## Objective

The objective of a verification run is to produce **defensible evidence** that a specific hipDNN build is correct and ready to ship. Executing this plan should answer three questions without ambiguity:

- **What was validated?** The exact build (OS, GPU family, ROCm version, source commit) that was exercised.
- **What was checked?** The prerequisites, test suites, and behaviors that were confirmed to pass.
- **Can it be trusted?** Whether an independent reader can reproduce the same result from the recorded steps.

### Why the details must be recorded

A verification run only has value if its outcome can be tied back to a specific build and reproduced later. Record the identifiers and evidence as you go, because:

- **Traceability** - a passing result is meaningless unless it is pinned to the exact artifact and source commit it came from. Recording the build identifier and source SHA proves the tested code is the code being shipped.
- **Reproducibility** - the exact commands and observed output let someone re-run the validation and get the same result, rather than trusting a summary.
- **Auditability** - a release sign-off may be revisited weeks later (for a regression, a hotfix, or a compliance check). The record is the durable proof of what was done.

Capture the run in a [Feature Release Run Template](./TestRunTemplate.md) document, which provides the structure for the identifiers, evidence, and reproducible commands described below.

---

## Prerequisites

### Test Case 1: CI Is Green 🟩

Existing checks run automatically on all PRs pre-merge and on the `develop` branch post-merge.

| CI Check | Description |
|----------|-------------|
| hipDNN Superbuild CI | Builds hipDNN and the providers via the superbuild on Linux and Windows (includes clang-tidy) |
| TheRock multi-arch CI | Builds and tests across GPU families (e.g. gfx94X, gfx950, gfx1151), with per-component test shards for `hipdnn`, `hipdnn-integration-tests`, `hipdnn-samples`, `hipdnn_install`, and each provider |
| pre-commit | Runs formatting and linting checks on changed files |
| codecov | Checks code coverage requirements |

### Test Case 2: Documentation is Current 🕒

Verify that all documentation is up to date:

1. Check version numbers throughout the documentation
2. Review instructions, explanations, and wording for clarity and accuracy
4. Verify changelog is complete and correct

> See the documentation listed in the [README](../../README.md#documentation) to identify relevant areas.

---

## Running Tests From TheRock Builds

The hipDNN library is included in ROCm development and release builds produced by [TheRock](https://github.com/ROCm/TheRock). These builds ship the hipDNN test executables, so you can validate a release without building from source. The same download also provides a complete ROCm tree, which the [source build](#running-tests-from-source-build) section below reuses as its build dependency, so a single download serves both flows.

### Obtain a ROCm Build with the hipDNN Tests

The hipDNN test executables and samples are not in the plain distribution tarball; they ship in the matching `-tests.tar.gz` variant. That variant is a superset of the plain distribution tarball (it contains the full ROCm tree plus the tests and samples), so download only the `-tests.tar.gz` and extract it into a `rocm-artifacts` folder. See [Tarballs](../Building.md#tarballs) for the filename structure and how to pick a version.

```bash
mkdir rocm-artifacts

# Replace <platform>, <group>, and <version> to match the build under test.
curl -O https://rocm.nightlies.amd.com/tarball-multi-arch/therock-dist-<platform>-<group>-tests-<version>.tar.gz

tar -C rocm-artifacts -zxf therock-dist-<platform>-<group>-tests-<version>.tar.gz
```

The commands below assume the tarball was extracted to `./rocm-artifacts`; adjust the paths if you used a different folder.

**Record what was validated.** The tarball filename encodes the OS, GPU family, and ROCm version, but the authoritative record is the manifest inside the tree (`rocm-artifacts/share/therock/therock_manifest.json`), which also carries the exact source commit the build came from. Capture the ROCm version and the rocm-libraries source commit (`pin_sha`) in your test record; the [Feature Release Run Template](./TestRunTemplate.md#4-replication-setup) gives the exact commands and confirms the delivery commit is contained in that commit's history.

### Running the hipDNN Tests

Use ctest to list the hipDNN test executables:
```
ctest --test-dir rocm-artifacts/bin/hipdnn --show-only
```
Sample output:
```
Internal ctest changing into directory: /workspace/rocm-artifacts/bin/hipdnn
Test project /workspace/rocm-artifacts/bin/hipdnn
  Test #1: hipdnn_data_sdk_tests
  Test #2: hipdnn_backend_tests
  Test #3: hipdnn_frontend_tests
  Test #4: hipdnn_test_sdk_tests
  Test #5: hipdnn_plugin_sdk_tests
  Test #6: hipdnn_public_backend_tests
  Test #7: hipdnn_public_frontend_tests

Total Tests: 7
```

Run all hipDNN tests in parallel:
```
ctest --test-dir rocm-artifacts/bin/hipdnn --output-on-failure --parallel 8 --timeout 30
```
Sample output:
```
Internal ctest changing into directory: /workspace/rocm-artifacts/bin/hipdnn
Test project /workspace/rocm-artifacts/bin/hipdnn
    Start 1: hipdnn_data_sdk_tests
    Start 2: hipdnn_backend_tests
    Start 6: hipdnn_public_backend_tests
    Start 7: hipdnn_public_frontend_tests
    Start 3: hipdnn_frontend_tests
    Start 4: hipdnn_test_sdk_tests
    Start 5: hipdnn_plugin_sdk_tests
1/7 Test #4: hipdnn_test_sdk_tests ............   Passed    0.02 sec
2/7 Test #5: hipdnn_plugin_sdk_tests ..........   Passed    0.02 sec
3/7 Test #3: hipdnn_frontend_tests ............   Passed    0.02 sec
4/7 Test #7: hipdnn_public_frontend_tests .....   Passed    0.27 sec
5/7 Test #6: hipdnn_public_backend_tests ......   Passed    0.84 sec
6/7 Test #2: hipdnn_backend_tests .............   Passed    1.33 sec
7/7 Test #1: hipdnn_data_sdk_tests ............   Passed    2.64 sec

100% tests passed, 0 tests failed out of 7

Total Test time (real) =   2.64 sec
```

Use the --verbose option for more detailed output:
```
ctest --test-dir rocm-artifacts/bin/hipdnn --parallel 8 --timeout 60 --verbose
```

---

## Running Tests From Source Build

This section builds hipDNN from source and runs the tests it produces. It can reuse the ROCm tree from the [TheRock build](#running-tests-from-therock-builds) above, so a single download serves both flows.

### Obtain ROCm (Build Dependency)

Building from source needs ROCm installed as a **build dependency** (the compiler, HIP, and libraries).

- **Reuse the TheRock build (recommended).** If you extracted a `-tests.tar.gz` above, the `rocm-artifacts` folder is already a complete ROCm tree. Add its `bin` folder to your `PATH` so the standalone presets discover it automatically:
  ```bash
  export PATH="$(pwd)/rocm-artifacts/bin:$PATH"   # Linux; on Windows add rocm-artifacts\bin to PATH
  ```
- **Or install ROCm separately.** If you did not obtain a build above, install ROCm by any method in [Obtaining ROCm](../Building.md#obtaining-rocm). You do **not** need the `-tests.tar.gz` variant here, because the tests are compiled from source.

Record the ROCm version you built against in your test record (for example the output of `hipconfig --version`), so the validated build can be identified later.

### Test Case 1: Build and Run the Automated Tests ⚙️

Build the standalone tests following the [Quick Start Guide](../Building.md#quick-start-guide); for the superbuild, see [Superbuild](../Building.md#superbuild). Then run them with ctest (the same command on Linux and Windows):

```bash
ctest --test-dir build/release
```

> On Windows, ensure the ROCm `bin` folder is on your `PATH` before running so the test executables can load the ROCm DLLs.

#### Expected Results

- **Test Status**: All tests should pass
- **GPU Test Behavior**:
  - **Without GPU**: All GPU tests should skip gracefully without failures
  - **With GPU**: hipDNN provider plugin integration tests may skip if the GPU is not supported
    - Skipped tests should provide clear messages indicating lack of ASIC support
- **Provider Support**: ASIC-specific coverage is determined by individual providers and is not a global hipDNN requirement

---

## ASAN Enabled Tests

### Test Case 1: Build and Run the Automated Tests with ASAN Enabled 🚨

> [!NOTE]
> ASAN is a manual check today (not yet in CI). The ROCm build requirement differs by platform:
> - **Linux**: requires an ASAN-enabled ROCm / TheRock build, so ASAN coverage extends into the shipped ROCm code, not just hipDNN and providers. Building TheRock with ASAN is possible but a large effort, so the Linux ASAN tests are only expected when an ASAN-enabled ROCm build is already available; building ROCm solely for ASAN testing is not expected.
> - **Windows**: does not require (or use) an ASAN-enabled ROCm build; ASAN covers only the code compiled during this build, not the installed ROCm libraries.
>
> See [Testing § ASAN/TSAN/sanitizer coverage](../testing/TESTING.md#asantsansanitizer-coverage) for more.

Build with address sanitizer enabled following the [Address Sanitizer Build](../Building.md#address-sanitizer-build) instructions, then run the `standard` tier (`ctest --test-dir <build> -L standard`).

#### Expected Results

- **Test Status**: All tests either pass or are explicitly skipped (architectures that do not support ASAN are skipped via `SKIP_IF_ASAN()` or a disabled ctest registration).
- **Memory Safety**: No memory leaks or violations should be detected.
- **Platform**: On Linux the suite is expected to complete cleanly. On Windows a fully clean ASAN run is not yet available (known issues being resolved); do not treat the remaining Windows failures as a release blocker until that work lands.
