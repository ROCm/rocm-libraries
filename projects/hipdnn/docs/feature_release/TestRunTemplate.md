# hipDNN Feature Release Run Template

This is the recording artifact for a **hipDNN milestone / release verification** run. It captures the traceability, evidence, and reproducible commands that show a release build was validated and is ready to ship. It is **not** the per-PR development testing form; for the day-to-day bar during development, see [Testing](../TESTING.md).

**How to use this template:**

- Copy this file to a working notes file for your feature or milestone (e.g. `<version> <short name> validation.md`) and fill in every `<placeholder>`.
- The run this template records is a run of the [Feature Release Test Plan](./TestPlan.md). Follow the plan's procedures; record the results here.
- Commands are given for Linux (bash), each followed by its Windows (PowerShell) equivalent. For Windows environment setup, see [Building hipDNN: Windows](../Building.md#windows).
- Replace the illustrative snippets with your own observed commands and output. Keep it evidence-based: paste real command output, not a summary.
- See [hipDNN testing documentation](../../README.md#testing) for how this template and the Feature Release Test Plan fit together.

**Filling this in as you follow the Feature Release Test Plan.** The sections below follow the plan in order. Work top to bottom:

1. Record identifiers as you go (section 1) and describe what the run covers (section 2).
2. [Feature Release Test Plan → Prerequisites](./TestPlan.md#prerequisites): record the CI evidence (section 3).
3. [Feature Release Test Plan → Running Tests From TheRock Builds](./TestPlan.md#running-tests-from-therock-builds): prepare the artifacts and manifest (section 4), then run the tests and paste the output (section 5).
4. [Feature Release Test Plan → Running Tests From Source Build](./TestPlan.md#running-tests-from-source-build): build from source and record its results (section 6).
5. [Feature Release Test Plan → Performance Testing](./TestPlan.md#performance-testing): record the benchmark results and the comparison (section 7).
6. Sign off once every plan step is evidenced (section 8).

---

## 1. Header / traceability

Fill in the identifiers that let a reader trace exactly what was validated back to source.

- **Feature or milestone**: `<feature name or milestone>`
- **Tracking issue**: `<issue link>`
- **Delivery PR**: `<https://github.com/ROCm/rocm-libraries/pull/NNNN>`
- **PR head commit**: `<sha>`
- **PR merge commit**: `<https://github.com/ROCm/rocm-libraries/commit/sha>`
- **GPU / ASIC families used**: `<every family validated, e.g. gfx942 (MI300), gfx950 (MI350)>`
- **Build(s) validated** (artifact identifiers/URLs, which encode OS / ASIC / ROCm version):
  - `<https://.../therock-dist-<platform>-<gpu-family>-<version>.tar.gz>`
  - `<https://.../therock-dist-<platform>-<gpu-family>-tests-<version>.tar.gz>`

> The artifact identifier encodes the OS, ASIC family, and ROCm version, so those no longer need separate fields. Still note the physical GPU/ASIC family the run executed on above, since a multi-arch artifact can target several.

---

## 2. What was validated

One paragraph describing the scope of the change and what this run exercises.

> `<One-paragraph summary: what the feature/milestone delivers and what behavior this validation confirms.>`

Bulleted test areas covered (delete rows that do not apply):

- **Frontend unit tests**: `<suites / behaviors covered>`
- **Backend unit tests**: `<suites / behaviors covered>`
- **GPU integration tests**: `<suites / behaviors covered>`
- **Sample validation**: `<sample name and flows exercised>`

---

## 3. Passing evidence

> Records the [Feature Release Test Plan → Prerequisites](./TestPlan.md#prerequisites) step (CI is green).

Link the CI, superbuild, and release-artifact runs that back this validation.

| Evidence | Link | Status |
|---|---|---|
| TheRock multi-arch CI hipdnn test step (`<gpu-family>`) | `<job log URL>` | `<Passed; e.g. 100% tests passed, 0 failed out of N>` |
| hipDNN Superbuild CI | `<actions run URL>` | `<Passed>` |
| Final TheRock / release artifact run | `<actions run URL>` | `<Passed / TBD>` |
| Final sample run artifact / log | `<actions run URL>` | `<Passed / TBD>` |

> **Scope note:** The command/output snippets in section 5 are *focused* validation runs (targeted gtest/ctest filters for the feature under test). The CI job shows the `standard` tier only. A complete release validation runs **both** the focused filters below **and** the `full`-tier installed CTest suites on the selected artifacts.

---

## 4. Replication setup

> Records the "obtain a ROCm build" step of [Feature Release Test Plan → Running Tests From TheRock Builds](./TestPlan.md#running-tests-from-therock-builds).

How to prepare the validated artifacts on a GPU node so a reader can reproduce section 5.

Use a GPU node matching the artifact family (e.g. `gfx94X-dcgpu` for MI300/MI325, `gfx950-dcgpu` for MI350/MI355). Download the `...-tests.tar.gz` variant (a superset of the plain distribution tarball that also carries the test executables); see [Obtaining ROCm](../Building.md#obtaining-rocm) for the filename structure and version selection. Run from an empty working directory:

```bash
mkdir <feature>-validation
cd <feature>-validation

curl -O https://nightly.repo.amd.com/rocm/core/tarball/therock-dist-linux-<gpu-family>-tests-<version>.tar.gz

mkdir rocm-artifacts
tar -C rocm-artifacts -zxf therock-dist-linux-<gpu-family>-tests-<version>.tar.gz
```

Windows (PowerShell):

```powershell
mkdir <feature>-validation
cd <feature>-validation

curl.exe -O https://nightly.repo.amd.com/rocm/core/tarball/therock-dist-windows-<gpu-family>-tests-<version>.tar.gz

mkdir rocm-artifacts
tar -C rocm-artifacts -zxf therock-dist-windows-<gpu-family>-tests-<version>.tar.gz
```

Record what the manifest inside the tree reports: the ROCm version and the exact source commit the build came from. Run the following and paste the output here:

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

> Confirm the delivery commit (section 1) is in the history of the `rocm-libraries` `pin_sha`; that proves the feature is in the validated build. The manifest lists a `pin_sha` for every submodule, so use the `rocm-libraries` entry.

See the [Feature Release Test Plan](./TestPlan.md#running-tests-from-therock-builds) for more on obtaining a ROCm build with the hipDNN test executables.

---

## 5. Test commands and expected output

> Records the "running the hipDNN tests" step of [Feature Release Test Plan → Running Tests From TheRock Builds](./TestPlan.md#running-the-hipdnn-tests).

The core of the record: repeatable command → observed-output blocks. Paste the **actual** output you saw. Reference the real hipDNN test binaries under `rocm-artifacts/bin/` (e.g. `hipdnn_frontend_tests`, `hipdnn_backend_tests`, `hipdnn_public_frontend_tests`). Numeric timings and tensor values vary by GPU and run.

Example commands for a focused area. Integration tests in the public frontend binary need `HIPDNN_PLUGIN_DIR` pointing at the test plugins; the samples need it pointing at the engine plugins:

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
$env:HIPDNN_PLUGIN_DIR = "$PWD\rocm-artifacts\lib\test_plugins\custom"
.\rocm-artifacts\bin\hipdnn_public_frontend_tests.exe --gtest_filter="<*IntegrationFeature*>" --gtest_brief=1

# One installed sample
$env:HIPDNN_PLUGIN_DIR = "$PWD\rocm-artifacts\lib\hipdnn_plugins\engines"
ctest --test-dir .\rocm-artifacts\bin\hipdnn_samples -R <test_name_regex> --output-on-failure
```

Paste the result lines you observed, for example:

```text
[  PASSED  ] <N> tests.
100% tests passed, 0 tests failed out of <N>
```

> Repeat one command/output block per focused area from section 2. For the full-suite requirement in the scope note, run the `full` tier in every test directory (see the [Feature Release Test Plan](./TestPlan.md#running-the-hipdnn-tests)) and record each directory's `100% tests passed, 0 tests failed out of <N>` line here as well.

---

## 6. Optional: build from source against installed artifacts

> Records the [Feature Release Test Plan → Running Tests From Source Build](./TestPlan.md#running-tests-from-source-build) step. Reuse the `rocm-artifacts` tree from section 4 as the build's ROCm dependency, as the plan describes.

Only needed when validating the **source** itself, or when the artifact ships runtime/devel packages but no prebuilt test/sample binaries. This builds against the installed artifacts using a user-supplied toolchain, so paths and the CMake prefix are explicit rather than preset-driven.

Check out the matching source tree (use the `pin_sha` from section 4, or `develop` after merge):

```bash
git clone --filter=blob:none --no-checkout https://github.com/ROCm/rocm-libraries.git rocm-libraries
cd rocm-libraries
git sparse-checkout init --cone
git sparse-checkout set projects/hipdnn dnn-providers cmake shared
git checkout <develop-or-pin-sha>
```

Configure and build against the installed artifacts (run from the `rocm-libraries` folder, sibling to `rocm-artifacts`):

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
  -DENABLE_CLANG_TIDY=OFF

cmake --build build/hipdnn-tests
ctest --test-dir build/hipdnn-tests --output-on-failure --parallel 8
```

> To build a sample instead, point `-S` at `projects/hipdnn/samples` and build the sample target. Expected result: the same passing snippets as section 5, and a full-suite `100% tests passed, 0 tests failed out of <N>`.
>
> This flow builds against the installed artifacts with an explicit toolchain and CMake prefix. For an ordinary standalone source build using the checked-in configure presets, follow the [Quick Start Guide](../Building.md#quick-start-guide) instead.

---

## 7. Performance results

> Records [Feature Release Test Plan → Performance Testing](./TestPlan.md#performance-testing).

For each GPU family, record the results file name and paste the comparison against the previous release.

- **Results file**: `<headline-<version>-<arch>.json>`
- **dnn-benchmarking version**: `<commit or tag>`
- **Baseline**: `<previous release's results file for the same GPU family>`

```bash
dnn-benchmark compare headline-<previous-version>-<arch>.json headline-<version>-<arch>.json
```

Observed output:

```text
<paste the compare output>
```

| Flagged regression | Explanation or tracking issue |
|---|---|
| `<graph / engine>` | `<reason or issue link>` |

---

## 8. Final validation checklist

Sign-off gate. Every box must be checked with real evidence before marking the milestone verified.

- [ ] Filled section 1 with the merged release/nightly **build identifier(s)** and the physical GPU/ASIC family used.
- [ ] Confirmed the delivery commit is contained in the artifact's manifest `pin_sha` (section 4).
- [ ] Replaced every `TBD` evidence link in section 3 with the final CI / release-artifact run and sample log.
- [ ] Ran the focused gtest/ctest filters **and** the `full` tier in every installed test directory, on every supported GPU family.
- [ ] Pasted the passing snippets into section 5 (elapsed times updated if rerun on final artifacts).
- [ ] Confirmed all [Feature Release Test Plan](./TestPlan.md) prerequisites (CI green, documentation/changelog current) are satisfied.
- [ ] Recorded the headline benchmark results for each GPU family, and explained or tracked every regression `compare` flagged (section 7).

**Tips for a defensible record:**

- Paste exact output, including failing messages if any; do not summarize away detail.
- Note environmental factors (GPU family, driver/ROCm version) that could affect results; the artifact identifier captures most of this.
- If a test consistently skips (e.g. GPU-gated or ASAN-incompatible), record why rather than deleting it. See [Environment](../Environment.md#environment-variables) for enabling logging when a run needs deeper insight.
