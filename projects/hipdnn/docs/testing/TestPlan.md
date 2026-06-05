# hipDNN Test Plan

This document outlines the test plan for hipDNN, covering test execution procedures and expectations.

> [!IMPORTANT]
> ⚠️ **All prerequisites and tests in this document must pass for a successful release.**

---

## Prerequisites

### Test Case 1: CI Is Green 🟩

Existing checks should be running automatically on all PRs pre-merge and on `develop` branch post-merge.

| CI Stage | Description |
|----------|-------------|
| `static-analysis` | Runs linting and static analysis tools to detect code issues early |
| `precheckin` | Runs unit & integration tests |
| `codecov` | Checks code coverage requirements |
| `debug` | Runs pre-checkin checks in a debug build |

### Test Case 2: Documentation is Current 🕒

Verify that all documentation is up to date:

1. Check version numbers throughout the documentation
2. Review instructions, explanations, and wording for clarity and accuracy
4. Verify changelog is complete and correct

> See the documentation listed in the [README](../../README.md#documentation) to identify relevant areas.

---

## Running Tests From Source Build

If needed, reference the [Quick Start Guide](../Building.md#quick-start-guide) to prepare a local environment.

### Test Case 1: Build and Run the Automated Tests ⚙️

With a working copy of the [rocm-libraries/projecdts/hipdnn folder](https://github.com/ROCm/rocm-libraries/tree/develop/projects/hipdnn):

```bash
# Run the following from the projects/hipdnn/build folder:
cmake ..
ninja check
```

#### Expected Results

- **Test Status**: All tests should pass
- **GPU Test Behavior**:
  - **Without GPU**: All GPU tests should skip gracefully without failures
  - **With GPU**: Plugin integration tests may skip if the GPU is not supported
    - Skipped tests should provide clear messages indicating lack of ASIC support
- **Plugin Support**: ASIC-specific coverage is determined by individual plugins and is not a global hipDNN requirement

---

## ASAN Enabled Tests

### Test Case 1: Build and Run the Automated Tests with ASAN Enabled 🚨

With a working copy of the [rocm-libraries/projecdts/hipdnn folder](https://github.com/ROCm/rocm-libraries/tree/develop/projects/hipdnn):

```bash
# Run the following from the projects/hipdnn/build folder:
cmake .. -DBUILD_ADDRESS_SANITIZER=ON
ninja check_ctest
```

#### Expected Results

- **Test Status**: All tests should pass
- **GPU Test Behavior**: All GPU tests will be skipped due to ASAN being enabled
- **Memory Safety**: No memory leaks or violations should be detected

## Running Tests From TheRock Builds

The hipDNN library is included in ROCm development and release builds produced by [TheRock](https://github.com/ROCm/TheRock).

This procedure uses the [install_rocm_from_artifacts.py](https://github.com/ROCm/TheRock/blob/main/build_tools/install_rocm_from_artifacts.py) tool to retrieve a pre-built hipDNN library with associated test programs.

The procedure that follows below was created using information from the following documents:
* TheRock [Installing Artifacts](https://github.com/ROCm/TheRock/blob/main/docs/development/installing_artifacts.md)
* TheRock [Releases](https://github.com/ROCm/TheRock/blob/main/RELEASES.md)

### Prerequisites

The `install_rocm_from_artifacts.py` script requires the `boto3` python library, run `pip install boto3` to install this library on your system.

### Download the Install Script

The simplest way to get the script with its dependencies is to clone TheRock (without submodules). The script will be located in `./TheRock/buildtools/install_rocm_from_artifacts.py`.
```
git clone https://github.com/ROCm/TheRock.git
```

### Install ROCm with hipDNN Tests

Refer to [Installing Artifacts](https://github.com/ROCm/TheRock/blob/main/docs/development/installing_artifacts.md#finding-release-versions) for instructions on selecting the artifact to download.

Be sure to include the `--hipdnn` and `--tests` option when running the script.

As an example, from examining the gfx90X GPU builds available on the [nightly tarball S3 bucket](https://therock-nightly-tarball.s3.amazonaws.com/index.html), the most recent tarball (at the time of this writing) is `therock-dist-linux-gfx90X-dcgpu-7.11.0a20251217.tar.gz`. From this:
* The release version is `7.11.0a20251217`
* The gpu family is `gfx90X-dcgpu`

With the above values, use the following command to download and install this ROCm build _with hipDNN and the hipDNN test executables_ (in this example, TheRock was cloned to `./TheRock`):
```
python3 TheRock/build_tools/install_rocm_from_artifacts.py --release 7.11.0a20251217 --amdgpu-family gfx90X-dcgpu --hipdnn --test
```

The ROCm install will be downloaded and extracted to a folder named `therock-build` in the current directory.


### Running the hipDNN Tests

After files have been extracted to `./therock-build`, use ctest to list the hipDNN test executables:
```
ctest --test-dir therock-build/bin/hipdnn --show-only
```
Sample output:
```
Internal ctest changing into directory: /workspace/therock-build/bin/hipdnn
Test project /workspace/therock-build/bin/hipdnn
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
ctest --test-dir therock-build/bin/hipdnn --output-on-failure --parallel 8 --timeout 30
```
Sample output:
```
Internal ctest changing into directory: /workspace/therock-build/bin/hipdnn
Test project /workspace/therock-build/bin/hipdnn
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
ctest --test-dir therock-build/bin/hipdnn --output-on-failure --parallel 8 --timeout 30 --verbose
```
