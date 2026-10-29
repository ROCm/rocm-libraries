# hipDNN Release Testing

Owner: @ROCm/hipdnn-reviewers | Last Updated: 2026-10-22

[Back to Testing](../TESTING.md)

This page covers **milestone / release verification**: what the hipDNN team validates before a release build ships, and the procedure to do it. It is **not** the per-PR workflow; for that, see [Testing](../TESTING.md). Record the run in a copy of the [Release Run Template](./RELEASE_RUN_TEMPLATE.md).

> [!IMPORTANT]
> ⚠️ **All prerequisites and tests in this document must pass for a successful release.**

## What release testing covers

The hipDNN team owns release testing: it runs the `full` tier, runs the benchmark corpus, and decides what to prioritize. Each item is tagged **today** (automated), **manual** (the team runs it by hand) or **planned**.

| Item | Tag |
|---|---|
| Run the `full` tier; everything must verify | today / manual |
| Run the dnn-benchmarking corpus, prioritized by risk | manual |
| ASAN build | today (nightly) |
| ASAN tests | manual (weekly) |
| Exploratory fuzzy mode within a pre-release budget | planned |

- **`full` tier.** TheRock's prerelease flow selects `full`; that flow runs outside this repository. The team also runs `full` manually ahead of release branching. Every test must verify. A test that cannot run is skipped through a tracked issue, not dropped ([Flaky tests and skips](./TESTING_STRATEGY.md#flaky-tests-and-skips)). Tier definitions: [Test tiers](../TESTING.md#test-tiers).
- **Performance corpus.** [dnn-benchmarking](https://github.com/ROCm/dnn-benchmarking) holds the benchmark corpus. Before a release the team runs the full corpus, ordered by importance of the op and by risk, meaning the ops changed most since the last release go first. Each release's results are compared with the previous release's, which is the proposed process for tracking performance. Approach: [Performance](./TESTING_STRATEGY.md#performance). Automation direction: [Roadmap](../Roadmap.md#benchmarking--performance-testing-2).
- **Sanitizers.** Release sanitizer runs follow [Testing Strategy: Sanitizers](./TESTING_STRATEGY.md#sanitizers).
- **Exploratory fuzzy mode (planned).** An exploratory mode that searches for uncovered cases within an affordable pre-release budget. See [Roadmap: Testing](../Roadmap.md#testing-1).

## Procedure

A verification run produces evidence that a specific hipDNN build is correct and ready to ship. The record shows what was validated (the exact build: OS, GPU family, ROCm version, source commit), what was checked (prerequisites, test suites and benchmarks that passed), and how to reproduce it (the commands and their observed output).

Run the procedure manually ahead of release branching, on every GPU family the release supports: PR CI tests only a few GPUs, so each provider must be validated on the devices it targets ([Known Testing Gaps](./KNOWN_TESTING_GAPS.md#hardware-coverage)).

> [!NOTE]
> Commands are given for Linux (bash), with Windows (PowerShell) equivalents where they differ. To set up a Windows environment, see [Building hipDNN: Windows](../Building.md#windows).

### 1. Confirm CI

CI checks run automatically on pull requests and on the `develop` branch post-merge; what each runs, and when, is in [What gates a pull request](../TESTING.md#what-gates-a-pull-request) and the [tier table](../TESTING.md#test-tiers). pre-commit runs formatting and linting checks on changed files. Confirm all of them are green for the release commit.

Verify that all documentation is up to date:

1. Check version numbers throughout the documentation
2. Review instructions, explanations, and wording for clarity and accuracy
3. Verify changelog is complete and correct

> See the documentation listed in the [README](../../README.md#documentation) to identify relevant areas.

### 2. Obtain a ROCm build with the hipDNN tests

The hipDNN library is included in ROCm development and release builds produced by [TheRock](https://github.com/ROCm/TheRock). These builds ship the hipDNN test executables, so you can validate a release without building from source. The same download is a complete ROCm tree, which the [source build](#6-build-from-source) below can reuse.

The hipDNN test executables and samples are not in the plain distribution tarball; they ship in the matching `-tests.tar.gz` variant. That variant is a superset of the plain distribution tarball (it contains the full ROCm tree plus the tests and samples), so download only the `-tests.tar.gz` and extract it into a `rocm-artifacts` folder. See [Tarballs](../Building.md#tarballs) for the filename structure and how to pick a version. Use a GPU node matching the artifact family (for example `gfx94X-dcgpu` for MI300/MI325, `gfx950-dcgpu` for MI350/MI355) and run from an empty working directory.

```bash
mkdir rocm-artifacts

# Replace <platform>, <group>, and <version> to match the build under test.
curl -O https://nightly.repo.amd.com/rocm/core/tarball/therock-dist-<platform>-<group>-tests-<version>.tar.gz

tar -C rocm-artifacts -zxf therock-dist-<platform>-<group>-tests-<version>.tar.gz
```

On Windows, set `<platform>` to `windows` and run `curl.exe`; PowerShell aliases `curl` to `Invoke-WebRequest`. `mkdir` and `tar` work unchanged.

The commands below assume the tarball was extracted to `./rocm-artifacts`; adjust the paths if you used a different folder.

### 3. Record the build manifest

The tarball filename encodes the OS, GPU family, and ROCm version, but the authoritative record is the manifest inside the tree (`rocm-artifacts/share/therock/therock_manifest.json`), which also carries the exact source commit the build came from. Capture the ROCm version and the rocm-libraries source commit (`pin_sha`) in your test record:

```bash
# ROCm version/package version, the TheRock build commit, and the CI run that produced the build.
grep -E '"the_rock_commit"|"github_run_id"|"rocm_version"|"rocm_package_version"' rocm-artifacts/share/therock/therock_manifest.json
# The rocm-libraries source commit (pin_sha) the build came from, with its submodule name for context.
grep -A3 '"submodule_name": "rocm-libraries"' rocm-artifacts/share/therock/therock_manifest.json | grep -E '"submodule_name"|"pin_sha"'
```

Windows (PowerShell):

```powershell
$manifest = 'rocm-artifacts\share\therock\therock_manifest.json'
# ROCm version/package version, the TheRock build commit, and the CI run that produced the build.
(Select-String -Path $manifest -Pattern '"the_rock_commit"|"github_run_id"|"rocm_version"|"rocm_package_version"').Line
# The rocm-libraries source commit (pin_sha) the build came from, with its submodule name for context.
(Select-String -Path $manifest -Pattern '"submodule_name": "rocm-libraries"' -Context 0,3 | ForEach-Object { $_.Line; $_.Context.PostContext }) | Where-Object { $_ -match '"submodule_name"|"pin_sha"' }
```

Example output:

```text
  "the_rock_commit": "4d992b34d67c91397ecaf4483d73d01c0b5c3fa3",
  "github_run_id": "38827024577",
  "rocm_package_version": "10.2.0a20261016",
  "rocm_version": "10.2.0",
      "submodule_name": "rocm-libraries",
      "pin_sha": "3dc098eb2db3c8329b7d447bab37021d529bc23d",
```

Confirm the delivery commit is in the history of the `rocm-libraries` `pin_sha`; that proves the feature is in the validated build. The manifest lists a `pin_sha` for every submodule, so use the `rocm-libraries` entry.

### 4. Run focused checks for the delivered feature

Use ctest to list the hipDNN test executables:

```bash
ctest --test-dir rocm-artifacts/bin/hipdnn --show-only
```

Then run targeted gtest/ctest filters for the feature under test. Use the real hipDNN test binaries under `rocm-artifacts/bin/` (for example `hipdnn_frontend_tests`, `hipdnn_backend_tests`, `hipdnn_public_frontend_tests`). Integration tests in the public frontend binary need `HIPDNN_PLUGIN_DIR` pointing at the test plugins; the samples need it pointing at the engine plugins:

```bash
# Focused gtest filter
./rocm-artifacts/bin/<test_binary> --gtest_filter="<Suite1*:Suite2*>" --gtest_brief=1

# Integration tests that load test plugins
HIPDNN_PLUGIN_DIR="$(pwd)/rocm-artifacts/lib/test_plugins/custom" \
  ./rocm-artifacts/bin/hipdnn_public_frontend_tests --gtest_filter="<*IntegrationFeature*>" --gtest_brief=1

# One installed sample
HIPDNN_PLUGIN_DIR="$(pwd)/rocm-artifacts/lib/hipdnn_plugins/engines" \
  ctest --test-dir ./rocm-artifacts/bin/hipdnn_samples -R <test_name_regex> --output-on-failure
```

Windows (PowerShell):

```powershell
# Focused gtest filter
.\rocm-artifacts\bin\<test_binary>.exe --gtest_filter="<Suite1*:Suite2*>" --gtest_brief=1

# Integration tests that load test plugins
$env:HIPDNN_PLUGIN_DIR = "$PWD\rocm-artifacts\bin\test_plugins\custom"
.\rocm-artifacts\bin\hipdnn_public_frontend_tests.exe --gtest_filter="<*IntegrationFeature*>" --gtest_brief=1

# One installed sample
$env:HIPDNN_PLUGIN_DIR = "$PWD\rocm-artifacts\bin\hipdnn_plugins\engines"
ctest --test-dir .\rocm-artifacts\bin\hipdnn_samples -R <test_name_regex> --output-on-failure
```

Repeat for each focused area of the change. Numeric timings and tensor values vary by GPU and run.

### 5. Run the full tier in every test directory

The focused checks do not replace the `full` tier. Run it in every hipDNN test directory of the artifacts. Each component (core, integration tests, samples and each provider) has its own directory, so the `hipdnn` directory alone is not enough:

```bash
for dir in hipdnn hipdnn_integration_tests_ctest hipdnn_samples miopen_plugin hipblaslt_plugin hip_kernel_provider; do
  ctest --test-dir rocm-artifacts/bin/$dir -L '^full$' -LE ex_gpu --output-on-failure --parallel 8 --timeout 7200
done
```

Windows (PowerShell):

```powershell
foreach ($dir in 'hipdnn','hipdnn_integration_tests_ctest','hipdnn_samples','miopen_plugin','hipblaslt_plugin','hip_kernel_provider') {
  ctest --test-dir rocm-artifacts\bin\$dir -L '^full$' -LE ex_gpu --output-on-failure --parallel 8 --timeout 7200
}
```

The `--timeout 7200` limit per test matches CI. A directory can run for over an hour, so size the job's time limit accordingly.

Where a directory has suites labelled for the GPU under test, replace `-LE ex_gpu` with `-L '^ex_gpu_<arch>$'`, as CI does ([Building and running tests](../TESTING.md#building-and-running-tests)). Add `--verbose` for each test's full output.

Expected results:

- Every test passes.
- A test that cannot run on this GPU skips with a message, and the skip has a tracked issue ([Flaky tests and skips](./TESTING_STRATEGY.md#flaky-tests-and-skips)). Record skip counts.

### 6. Build from source

This step builds hipDNN from source and runs the tests it produces.

**Obtain ROCm (build dependency).** Building from source needs ROCm installed as a build dependency (the compiler, HIP, and libraries).

- **Reuse the TheRock build (recommended).** If you extracted a `-tests.tar.gz` in [step 2](#2-obtain-a-rocm-build-with-the-hipdnn-tests), the `rocm-artifacts` folder is already a complete ROCm tree. Add its `bin` folder to your `PATH` so the standalone presets discover it automatically:
  ```bash
  export PATH="$(pwd)/rocm-artifacts/bin:$PATH"
  ```
  On Windows (PowerShell):
  ```powershell
  $env:PATH = "$PWD\rocm-artifacts\bin;$env:PATH"
  ```
- **Or install ROCm separately.** If you did not obtain a build above, install ROCm by any method in [Obtaining ROCm](../Building.md#obtaining-rocm). You do **not** need the `-tests.tar.gz` variant here, because the tests are compiled from source.

Record the ROCm version you built against (for example the output of `hipconfig --version`), so the validated build can be identified later.

**Preset flow.** Build the standalone tests following the [Quick Start Guide](../Building.md#quick-start-guide); for the superbuild, see [Superbuild](../Building.md#superbuild). Then run them with ctest (the same command on Linux and Windows):

```bash
ctest --test-dir build/release -L '^full$' -LE ex_gpu
```

> On Windows, ensure the ROCm `bin` folder is on your `PATH` before running so the test executables can load the ROCm DLLs.

**Explicit-prefix flow (optional).** Use it only when validating the source itself, or when the artifact ships runtime/devel packages but no prebuilt test/sample binaries. This builds against the installed artifacts using a user-supplied toolchain, so paths and the CMake prefix are explicit rather than preset-driven. Check out the matching source tree (use the `pin_sha` from [step 3](#3-record-the-build-manifest), or `develop` after merge):

```bash
git clone --filter=blob:none --no-checkout https://github.com/ROCm/rocm-libraries.git rocm-libraries
cd rocm-libraries
git sparse-checkout init --cone
git sparse-checkout set projects/hipdnn dnn-providers cmake shared
git checkout <develop-or-pin-sha>
```

Configure and build against the installed artifacts (run from the `rocm-libraries` folder, sibling to `rocm-artifacts`). The configure needs clang-format 18 and the third-party libraries (GTest, spdlog); the Windows artifacts do not ship the libraries. Where they are not installed, add `-DENABLE_CLANG_FORMAT=OFF` and `-DALLOW_FETCH_DEPS=ON` ([Third-Party Libraries](../Building.md#third-party-libraries)), as the Windows block does:

```bash
cmake -S projects/hipdnn \
  -B build/hipdnn-tests \
  -GNinja \
  -DCMAKE_PREFIX_PATH="$(pwd)/../rocm-artifacts/lib/cmake" \
  -DROCM_PATH="$(pwd)/../rocm-artifacts" \
  -DCMAKE_C_COMPILER="$(pwd)/../rocm-artifacts/lib/llvm/bin/clang" \
  -DCMAKE_CXX_COMPILER="$(pwd)/../rocm-artifacts/lib/llvm/bin/clang++" \
  -DHIPDNN_SKIP_TESTS=OFF \
  -DENABLE_CLANG_TIDY=OFF

cmake --build build/hipdnn-tests
ctest --test-dir build/hipdnn-tests --output-on-failure --parallel 8
```

Windows (PowerShell):

```powershell
cmake -S projects/hipdnn `
  -B build/hipdnn-tests `
  -GNinja `
  -DCMAKE_PREFIX_PATH="$PWD/../rocm-artifacts/lib/cmake" `
  -DROCM_PATH="$PWD/../rocm-artifacts" `
  -DCMAKE_C_COMPILER="$PWD/../rocm-artifacts/lib/llvm/bin/clang.exe" `
  -DCMAKE_CXX_COMPILER="$PWD/../rocm-artifacts/lib/llvm/bin/clang++.exe" `
  -DGPU_TARGETS=<gpu-target> `
  -DHIPDNN_SKIP_TESTS=OFF `
  -DENABLE_CLANG_TIDY=OFF `
  -DENABLE_CLANG_FORMAT=OFF `
  -DALLOW_FETCH_DEPS=ON

cmake --build build/hipdnn-tests
ctest --test-dir build/hipdnn-tests --output-on-failure --parallel 8
```

To build a sample instead, point `-S` at `projects/hipdnn/samples` and build the sample target. For an ordinary standalone source build using the checked-in configure presets, follow the preset flow.

Expected results:

- **Test status:** all tests pass, including the same passing snippets as [step 4](#4-run-focused-checks-for-the-delivered-feature) and a full-suite `100% tests passed, 0 tests failed out of <N>` for the explicit-prefix flow.
- **Without a GPU:** all GPU tests skip gracefully without failures.
- **With a GPU:** hipDNN provider plugin integration tests may skip if the GPU is not supported. Skipped tests provide clear messages indicating lack of ASIC support.
- **Provider support:** ASIC-specific coverage is determined by individual providers and is not a global hipDNN requirement.

### 7. Run ASAN

> [!NOTE]
> - **Linux**: full ASAN requires an ASAN-enabled ROCm / TheRock build.
> - **Windows**: ASAN instruments code built in this project, not installed ROCm libraries.
>
> How ASAN runs today: [Testing Strategy: Sanitizers](./TESTING_STRATEGY.md#sanitizers).

Build with address sanitizer enabled following the [Address Sanitizer Build](../Building.md#address-sanitizer-build) instructions, then run the `standard` tier (`ctest --test-dir <build> -L '^standard$'`). Commands: [Sanitizers and static analysis](../TESTING.md#sanitizers-and-static-analysis).

Expected results:

- **Test status:** record the pass, skip and fail counts. A skipped test was not checked under ASAN; tests skip on architectures that do not support ASAN.
- **Memory safety:** no memory leaks or violations are detected.

### 8. Run performance benchmarks

[dnn-benchmarking](https://github.com/ROCm/dnn-benchmarking) times hipDNN graphs on each engine. Set it up by following its [setup guide](https://github.com/ROCm/dnn-benchmarking/blob/main/docs/setup.md), passing `--rocm-libraries-ref <pin_sha>` with the `pin_sha` recorded from the manifest, so it builds hipDNN and the providers from the release's source commit. Run it on each GPU family you validate.

The workload tarballs are tracked with DVC, so a fresh clone holds only `.dvc` pointers. Pull the headline set first ([workload files](https://github.com/ROCm/dnn-benchmarking#workload-files)):

```bash
dvc pull -R Workloads/headline
```

Run the headline workloads, the curated regression set described in its [README](https://github.com/ROCm/dnn-benchmarking#workload-files), and write one results file. Add `--validate pytorch` to compare each engine's output with a PyTorch reference: the `full` tier is the correctness gate, and this gives extra correctness signal on the benchmark workloads. It also adds a timed `reference` row for each graph. The set takes about an hour on one GPU without validation:

```bash
dnn-benchmark -g Workloads/headline/*.tar.gz --validate pytorch -o headline-<version>-<arch>.json
```

Windows (PowerShell) expands the workload files itself:

```powershell
dnn-benchmark -g (Get-ChildItem Workloads\headline\*.tar.gz).FullName --validate pytorch -o headline-<version>-<arch>.json
```

Compare it with the previous release's results file for the same GPU family ([how `compare` reports changes](https://github.com/ROCm/dnn-benchmarking/blob/main/docs/usage.md#compare)):

```bash
dnn-benchmark compare headline-<previous-version>-<arch>.json headline-<version>-<arch>.json
```

Ahead of release branching, also run the larger `Workloads/microbench/` and `Workloads/models/` sets as time allows, in the [order above](#what-release-testing-covers).

Expected results:

- `dnn-benchmark` exits 0. Exit code 3 means an engine row failed validation against PyTorch; explain it or track it as an issue.
- Every regression that `compare` flags is explained or has a tracked issue.
- The results file is the baseline for the next release.
