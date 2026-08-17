# hipDNN Feature Readiness Run Template

This is the recording artifact for a **hipDNN milestone / release verification** run. It captures the traceability, evidence, and reproducible commands that show a release build was validated and is ready to ship. It is **not** the per-PR development testing form; for the day-to-day bar during development, see [Testing](../TESTING.md#development-workflow).

**How to use this template:**

- Copy this file to a working notes file for your feature or milestone (e.g. `ROCM-<ticket> <short name> validation.md`) and fill in every `<placeholder>`.
- The run this template records is a run of the [Feature Readiness Plan](./FeatureReadinessPlan.md). Follow the plan's procedures; record the results here.
- Replace the illustrative snippets with your own observed commands and output. Keep it evidence-based: paste real command output, not a summary.

**Filling this in as you follow the Feature Readiness Plan.** The sections below are ordered to match the qualifier's path through the plan. Work top to bottom:

1. Record identifiers as you go (section 1) and describe what the run covers (section 2).
2. [Feature Readiness Plan → Prerequisites](./FeatureReadinessPlan.md#prerequisites): record the CI evidence (section 3).
3. [Feature Readiness Plan → Running Tests From TheRock Builds](./FeatureReadinessPlan.md#running-tests-from-therock-builds): prepare the artifacts and manifest (section 4), then run the tests and paste the output (section 5).
4. [Feature Readiness Plan → Running Tests From Source Build](./FeatureReadinessPlan.md#running-tests-from-source-build): build from source and record its results (section 6).
5. Sign off once every plan step is evidenced (section 7).

---

## 1. Header / traceability

Fill in the identifiers that let a reader trace exactly what was validated back to source.

- **Feature or milestone**: `<feature name or milestone>`
- **Epic / ticket**: `<JIRA-EPIC>` / `<JIRA-TICKET>`
- **Delivery PR**: `<https://github.com/ROCm/rocm-libraries/pull/NNNN>`
- **PR head commit**: `<sha>`
- **PR merge commit**: `<https://github.com/ROCm/rocm-libraries/commit/sha>`
- **GPU / ASIC family used**: `<e.g. gfx942 (MI300), gfx950 (MI350)>`
- **Build(s) validated** (artifact identifiers/URLs, which encode OS / ASIC / ROCm version):
  - `<https://.../therock-dist-linux-<gpu-family>-<version>.tar.gz>`
  - `<https://.../therock-dist-linux-<gpu-family>-tests-<version>.tar.gz>`

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

> Records the [Feature Readiness Plan → Prerequisites](./FeatureReadinessPlan.md#prerequisites) step (CI is green).

Link the CI, superbuild, and release-artifact runs that back this validation.

| Evidence | Link | Status |
|---|---|---|
| hipDNN Jenkins / codecov test job (`<gpu-family>`) | `<consoleText URL>` | `<Passed; e.g. 100% tests passed, 0 failed out of N>` |
| hipDNN Superbuild CI | `<actions run URL>` | `<Passed>` |
| Final TheRock / release artifact run | `<actions run URL>` | `<Passed / TBD>` |
| Final sample run artifact / log | `<actions run URL>` | `<Passed / TBD>` |

> **Scope note:** The command/output snippets in section 5 are *focused* validation runs (targeted gtest/ctest filters for the feature under test). The *full* hipDNN CTest suite evidence comes from the linked CI job. A complete release validation runs **both** the focused filters below **and** the full installed CTest suite on the selected artifacts.

---

## 4. Replication setup

> Records the "obtain a ROCm build" step of [Feature Readiness Plan → Running Tests From TheRock Builds](./FeatureReadinessPlan.md#running-tests-from-therock-builds).

How to prepare the validated artifacts on a GPU node so a reader can reproduce section 5.

Use a GPU node matching the artifact family (e.g. `gfx94X-dcgpu` for MI300/MI325, `gfx950-dcgpu` for MI350/MI355). Download the `...-tests.tar.gz` variant (a superset of the plain distribution tarball that also carries the test executables); see [Obtaining ROCm](../Building.md#obtaining-rocm) for the filename structure and version selection. Run from an empty working directory:

```bash
mkdir <feature>-validation
cd <feature>-validation

curl -O https://rocm.nightlies.amd.com/tarball-multi-arch/therock-dist-linux-<gpu-family>-tests-<version>.tar.gz

mkdir rocm-artifacts
tar -C rocm-artifacts -zxf therock-dist-linux-<gpu-family>-tests-<version>.tar.gz
```

Record what the manifest inside the tree reports: the ROCm version and the exact source commit the build came from. Run the following and paste the output here:

```bash
# ROCm version/package version, the TheRock build commit, and the CI run that produced the build.
grep -E '"the_rock_commit"|"github_run_id"|"rocm_version"|"rocm_package_version"' rocm-artifacts/share/therock/therock_manifest.json
# The rocm-libraries source commit (pin_sha) the build came from, with its submodule name for context.
grep -A3 '"submodule_name": "rocm-libraries"' rocm-artifacts/share/therock/therock_manifest.json | grep -E '"submodule_name"|"pin_sha"'
```

Example output:

```text
  "the_rock_commit": "d9551ec95643eb12cc1fdb81a95530e105539415",
  "github_run_id": "30227121935",
  "rocm_package_version": "7.15.0a20260727",
  "rocm_version": "7.15.0",
      "submodule_name": "rocm-libraries",
      "pin_sha": "bbb68174083e83e9465b60ecd8e20e4439a8e101",
```

> Record the above output from the manifest; the `pin_sha` confirm the delivery commit (section 1) is contained in the history of that SHA. This proves the feature is actually present in the validated build. The manifest lists a `pin_sha` for every submodule, so use the `rocm-libraries` entry specifically, not the first `pin_sha` in the file.

See the [Feature Readiness Plan](./FeatureReadinessPlan.md#running-tests-from-therock-builds) for more on obtaining a ROCm build with the hipDNN test executables.

---

## 5. Test commands and expected output

> Records the "running the hipDNN tests" step of [Feature Readiness Plan → Running Tests From TheRock Builds](./FeatureReadinessPlan.md#running-the-hipdnn-tests).

The core of the record: repeatable command → observed-output blocks. Paste the **actual** output you saw. Reference the real hipDNN test binaries under `rocm-artifacts/bin/` (e.g. `hipdnn_frontend_tests`, `hipdnn_backend_tests`, `hipdnn_public_frontend_tests`). Numeric timings and tensor values vary by GPU and run.

### Example: focused gtest filter run

```bash
./rocm-artifacts/bin/<test_binary> \
  --gtest_filter="<Suite1*:Suite2*>" \
  --gtest_brief=1
```

Observed passing output:

```text
[==========] <N> tests from <M> test suites ran. (<t> ms total)
[  PASSED  ] <N> tests.
```

### Example: GPU integration filter (plugin dir required)

```bash
HIPDNN_PLUGIN_DIR="$(pwd)/rocm-artifacts/lib/test_plugins/custom" \
./rocm-artifacts/bin/hipdnn_public_frontend_tests \
  --gtest_filter="<*IntegrationFeature*>" \
  --gtest_brief=1
```

Observed passing output:

```text
[==========] <N> tests from <M> test suites ran. (<t> ms total)
[  PASSED  ] <N> tests.
```

### Example: installed CTest filter

```bash
HIPDNN_PLUGIN_DIR="$(pwd)/rocm-artifacts/lib/hipdnn_plugins/engines" \
ctest --test-dir ./rocm-artifacts/bin/hipdnn_samples \
  -R <test_name_regex> \
  --output-on-failure
```

Observed passing output:

```text
100% tests passed, 0 tests failed out of <N>
```

> Repeat one command/output block per focused area from section 2. For the full-suite requirement in the scope note, run the installed hipDNN CTest suite (see the [Feature Readiness Plan](./FeatureReadinessPlan.md#running-the-hipdnn-tests)) and record its `100% tests passed, 0 tests failed out of <N>` line here as well.

---

## 6. Optional: build from source against installed artifacts

> Records the [Feature Readiness Plan → Running Tests From Source Build](./FeatureReadinessPlan.md#running-tests-from-source-build) step. Reuse the `rocm-artifacts` tree from section 4 as the build's ROCm dependency, as the plan describes.

Only needed when validating the **source** itself, or when the artifact ships runtime/devel packages but no prebuilt test/sample binaries. This builds against the installed artifacts using a user-supplied toolchain, so paths and the CMake prefix are explicit rather than preset-driven.

Check out the matching source tree (use the `pin_sha` from section 4, or `develop` after merge):

```bash
git clone --filter=blob:none --no-checkout https://github.com/ROCm/rocm-libraries.git rocm-libraries
cd rocm-libraries
git sparse-checkout init --cone
git sparse-checkout set projects/hipdnn dnn-providers cmake test
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

cmake --build build/hipdnn-tests --target <targets>
ctest --test-dir build/hipdnn-tests --output-on-failure --parallel 8
```

> To build a sample instead, point `-S` at `projects/hipdnn/samples` and build the sample target. Expected result: the same passing snippets as section 5, and a full-suite `100% tests passed, 0 tests failed out of <N>`.
>
> This flow builds against the installed artifacts with an explicit toolchain and CMake prefix. For an ordinary standalone source build using the checked-in configure presets, follow the [Quick Start Guide](../Building.md#quick-start-guide) instead.

---

## 7. Final validation checklist

Sign-off gate. Every box must be checked with real evidence before marking the milestone verified.

- [ ] Filled section 1 with the merged release/nightly **build identifier(s)** and the physical GPU/ASIC family used.
- [ ] Confirmed the delivery commit is contained in the artifact's manifest `pin_sha` (section 4).
- [ ] Replaced every `TBD` evidence link in section 3 with the final CI / release-artifact run and sample log.
- [ ] Ran the focused gtest/ctest filters **and** the full installed hipDNN CTest suite on the final artifacts.
- [ ] Pasted the passing snippets into section 5 (elapsed times updated if rerun on final artifacts).
- [ ] Confirmed all [Feature Readiness Plan](./FeatureReadinessPlan.md) prerequisites (CI green, documentation/changelog current) are satisfied.

**Tips for a defensible record:**

- Paste exact output, including failing messages if any; do not summarize away detail.
- Note environmental factors (GPU family, driver/ROCm version) that could affect results; the artifact identifier captures most of this.
- If a test consistently skips (e.g. GPU-gated or ASAN-incompatible), record why rather than deleting it. See [Environment](../Environment.md#environment-variables) for enabling logging when a run needs deeper insight.
