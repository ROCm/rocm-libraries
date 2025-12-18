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

## Regular Tests Run From Source Build

If needed, reference the [Quick Start Guide](../Building.md#quick-start-guide) to prepare a local environment.

### Test Case 1: Build and Run the Automated Tests ⚙️

With a clone of the [rocm-libraries repository](https://github.com/ROCm/rocm-libraries/):

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

With a clone of the [rocm-libraries repository](https://github.com/ROCm/rocm-libraries/):

```bash
# Run the following from the projects/hipdnn/build folder:
cmake .. -DBUILD_ADDRESS_SANITIZER=ON
ninja check_ctest
```

#### Expected Results

- **Test Status**: All tests should pass
- **GPU Test Behavior**: All GPU tests will be skipped due to ASAN being enabled
- **Memory Safety**: No memory leaks or violations should be detected

## Regular Tests Run From TheRock Build

The hipDNN library is included in ROCm development and release builds produced by [TheRock](https://github.com/ROCm/TheRock).

This procedure uses the [install_rocm_from_artifacts.py](https://github.com/ROCm/TheRock/blob/main/build_tools/install_rocm_from_artifacts.py) to retrieve a pre-built hipDNN library with associated test programs. The procedure below was created using information from the following documents:
* TheRock [Installing Artifacts](https://github.com/ROCm/TheRock/blob/main/docs/development/installing_artifacts.md)
* TheRock [Releases](https://github.com/ROCm/TheRock/blob/main/RELEASES.md)

### Prerequisites

The install_rocm_from_artifacts.py script requires the boto3 python library, run `pip install boto3` to install this library on your system.

### Download the Install Script

The simplest way to get the script with its dependencies is to clone TheRock (without submodules). The script will be located in TheRock/buildtools/install_rocm_from_artifacts.py.
```
git clone https://github.com/ROCm/TheRock.git
cd TheRock/build_tools
```

### Install ROCm with hipDNN Tests

Refer to [Installing Artifacts](https://github.com/ROCm/TheRock/blob/main/docs/development/installing_artifacts.md) for instructions on selecting the artifact to download.

Be sure to include the `--hipdnn` and `--tests` option when running the script.

As an example, from examining the gfx90X GPU builds available on the [nightly tarball S3 bucket](https://therock-nightly-tarball.s3.amazonaws.com/index.html), the most recent tarball (at the time of this writing) is `therock-dist-linux-gfx90X-dcgpu-7.11.0a20251217.tar.gz`. From this:
* The release version is `7.11.0a20251217`
* The gpu family is `gfx90X-dcgpu`

The command to download and install this ROCm build _with hipDNN and the hipDNN test executables_ is:
```
python3 install_rocm_from_artifacts.py --release 7.11.0a20251217 --amdgpu-family gfx90X-dcgpu --hipdnn --test
```

The files will be downloaded and extracted to a folder named `therock-build` in the current directory.


### Running the hipDNN Tests

The test executables can be found by running the following command from within the `therock-build` folder (on Linux):

```
 find . \( -name '*hipdnn*tests' -o -name 'miopen*plugin_tests' \)
./bin/hipdnn_plugin_sdk_tests
./bin/hipdnn_sdk_tests
./bin/hipdnn_frontend_tests
./bin/public_hipdnn_backend_tests
./bin/public_hipdnn_frontend_tests
./bin/hipdnn_test_sdk_tests
./bin/hipdnn_backend_tests
./bin/miopen_legacy_plugin_tests
```

To run all of the test, use the `-exec` option:
```
find . \( -name '*hipdnn*tests' -o -name 'miopen*plugin_tests' \) -exec {} \;
```

To view a brief help of the gtest options that cane be used with these tests, use the --help option with the test executable:
```
./bin/hipdnn_sdk_tests --help
```

Some notable options:
```
  --gtest_list_tests
      List the names of all tests instead of running them. The name of
      TEST(Foo, Bar) is "Foo.Bar".
  --gtest_filter=POSITIVE_PATTERNS[-NEGATIVE_PATTERNS]
      Run only the tests whose name matches one of the positive patterns but
      none of the negative patterns. '?' matches any single character; '*'
      matches any substring; ':' separates two patterns.
 --gtest_brief=1
      Only print test failures.
```

Example: run all tests with brief output and print errors:
```
find . \( -name '*hipdnn*tests' -o -name 'miopen*plugin_tests' \) -exec {} --gtest_brief=1 \;
```
