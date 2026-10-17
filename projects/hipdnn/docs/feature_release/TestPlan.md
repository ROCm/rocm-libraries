# hipDNN Feature Release Test Plan

This document is the **hipDNN milestone / release verification** plan: the procedures and expectations for validating a release build and confirming it is ready to ship. It is **not** the per-PR development workflow; for the day-to-day testing bar during development, see [Testing](../TESTING.md).

> [!IMPORTANT]
> ⚠️ **All prerequisites and tests in this document must pass for a successful release.**

---

## Objective

A verification run produces evidence that a specific hipDNN build is correct and ready to ship. The record must show:

- **What was validated:** the exact build (OS, GPU family, ROCm version, source commit).
- **What was checked:** the prerequisites, test suites and benchmarks that passed.
- **How to reproduce it:** the commands and their observed output, so a reader can re-run the validation later, for example when investigating a regression.

Record the run in a copy of the [Feature Release Run Template](./TestRunTemplate.md).

### When and where to run it

Run this plan manually ahead of release branching. Run it on every GPU family the release supports: PR CI tests only a few GPUs, so each provider must be validated on the devices it targets ([Known Testing Gaps](../testing/KNOWN_TESTING_GAPS.md#hardware-coverage)). Release ownership and priorities are in [Release Testing](../testing/RELEASE_TESTING.md).

---

## Prerequisites

### Test Case 1: CI Is Green 🟩

Existing checks run automatically on all PRs pre-merge and on the `develop` branch post-merge.

| CI Check | Description |
|----------|-------------|
| hipDNN Superbuild CI | Builds hipDNN and the providers via the superbuild on Linux and Windows (clang-tidy on Linux only) |
| TheRock multi-arch CI | Builds the GPU families and runs the `standard` tier for the affected components |
| pre-commit | Runs formatting and linting checks on changed files |

What each check runs: [What gates a pull request](../TESTING.md#what-gates-a-pull-request).

### Test Case 2: Documentation is Current 🕒

Verify that all documentation is up to date:

1. Check version numbers throughout the documentation
2. Review instructions, explanations, and wording for clarity and accuracy
3. Verify changelog is complete and correct

> See the documentation listed in the [README](../../README.md#documentation) to identify relevant areas.

---

## Running Tests From TheRock Builds

The hipDNN library is included in ROCm development and release builds produced by [TheRock](https://github.com/ROCm/TheRock). These builds ship the hipDNN test executables, so you can validate a release without building from source. The same download is a complete ROCm tree, which the [source build](#running-tests-from-source-build) below can reuse.

### Obtain a ROCm Build with the hipDNN Tests

The hipDNN test executables and samples are not in the plain distribution tarball; they ship in the matching `-tests.tar.gz` variant. That variant is a superset of the plain distribution tarball (it contains the full ROCm tree plus the tests and samples), so download only the `-tests.tar.gz` and extract it into a `rocm-artifacts` folder. See [Tarballs](../Building.md#tarballs) for the filename structure and how to pick a version.

```bash
mkdir rocm-artifacts

# Replace <platform>, <group>, and <version> to match the build under test.
curl -O https://nightly.repo.amd.com/rocm/core/tarball/therock-dist-<platform>-<group>-tests-<version>.tar.gz

tar -C rocm-artifacts -zxf therock-dist-<platform>-<group>-tests-<version>.tar.gz
```

The commands below assume the tarball was extracted to `./rocm-artifacts`; adjust the paths if you used a different folder.

**Record what was validated.** The tarball filename encodes the OS, GPU family, and ROCm version, but the authoritative record is the manifest inside the tree (`rocm-artifacts/share/therock/therock_manifest.json`), which also carries the exact source commit the build came from. Capture the ROCm version and the rocm-libraries source commit (`pin_sha`) in your test record; the [Feature Release Run Template](./TestRunTemplate.md#4-replication-setup) gives the exact commands and confirms the delivery commit is contained in that commit's history.

### Running the hipDNN Tests

Use ctest to list the hipDNN test executables:
```
ctest --test-dir rocm-artifacts/bin/hipdnn --show-only
```
Run the `full` tier in every hipDNN test directory of the artifacts. Each component (core, integration tests, samples and each provider) has its own directory, so the `hipdnn` directory alone is not enough:

```bash
for dir in hipdnn hipdnn_integration_tests_ctest hipdnn_samples miopen_plugin hipblaslt_plugin hip_kernel_provider; do
  ctest --test-dir rocm-artifacts/bin/$dir -L '^full$' -LE ex_gpu --output-on-failure --parallel 8
done
```

Where a directory has suites labelled for the GPU under test, replace `-LE ex_gpu` with `-L '^ex_gpu_<arch>$'`, as CI does ([Building and running tests](../TESTING.md#building-and-running-tests)). Add `--verbose` for each test's full output.

#### Expected Results

- Every test passes.
- A test that cannot run on this GPU skips with a message, and the skip has a tracked issue ([Flaky tests and skips](../testing/TESTING_STRATEGY.md#flaky-tests-and-skips)). Record skip counts.

---

## Running Tests From Source Build

This section builds hipDNN from source and runs the tests it produces.

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
ctest --test-dir build/release -L '^full$' -LE ex_gpu
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
> - **Linux**: full ASAN requires an ASAN-enabled ROCm / TheRock build.
> - **Windows**: ASAN instruments code built in this project, not installed ROCm libraries.
>
> How ASAN runs today: [Testing Strategy: Sanitizers](../testing/TESTING_STRATEGY.md#sanitizers).

Build with address sanitizer enabled following the [Address Sanitizer Build](../Building.md#address-sanitizer-build) instructions, then run the `standard` tier (`ctest --test-dir <build> -L '^standard$'`).

#### Expected Results

- **Test Status**: Record the pass, skip and fail counts. A skipped test was not checked under ASAN; tests skip on architectures that do not support ASAN.
- **Memory Safety**: No memory leaks or violations should be detected.

---

## Performance Testing

### Test Case 1: Benchmark the Release Build 📈

[dnn-benchmarking](https://github.com/ROCm/dnn-benchmarking) times hipDNN graphs on each engine. Set it up by following its [setup guide](https://github.com/ROCm/dnn-benchmarking/blob/main/docs/setup.md), passing `--rocm-libraries-ref <pin_sha>` with the `pin_sha` recorded from the manifest, so it builds hipDNN and the providers from the release's source commit. Run it on each GPU family you validate.

Run the headline workloads, the curated regression set described in its [README](https://github.com/ROCm/dnn-benchmarking#workload-files), and write one results file. The set takes about an hour on one GPU:

```bash
dnn-benchmark -g Workloads/headline/*.tar.gz -o headline-<version>-<arch>.json
```

Compare it with the previous release's results file for the same GPU family ([how `compare` reports changes](https://github.com/ROCm/dnn-benchmarking/blob/main/docs/usage.md#compare)):

```bash
dnn-benchmark compare headline-<previous-version>-<arch>.json headline-<version>-<arch>.json
```

Ahead of release branching, also run the larger `Workloads/microbench/` and `Workloads/models/` sets as time allows, ordered by op importance and by risk from changes since the last release ([Release Testing](../testing/RELEASE_TESTING.md#performance-corpus-manual)).

#### Expected Results

- `dnn-benchmark` exits 0.
- Every regression that `compare` flags is explained or has a tracked issue.
- The results file is kept with the run record, as the baseline for the next release.
