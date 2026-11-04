# hipDNN Release Run Template

Owner: @ROCm/hipdnn-reviewers | Last Updated: 2026-10-22

[Back to Testing](../TESTING.md)

This is the recording artifact for a **hipDNN milestone / release verification** run. It captures the traceability and evidence that show a release build was validated and is ready to ship. It is **not** the per-PR development testing form; for the day-to-day bar during development, see [Testing](../TESTING.md).

**How to use this template:**

- Copy this file to a working notes file for your feature or milestone (e.g. `<version> <short name> validation.md`) and fill in every `<placeholder>`.
- The run this template records is a run of the [Release Testing procedure](./RELEASE_TESTING.md#procedure). Follow the procedure and its commands; record the results here. For how the documents fit together, see [Testing documentation](../TESTING.md#testing-documentation).
- Replace the illustrative snippets with your own observed output. Keep it evidence-based: paste real command output, not a summary.

The sections below follow the procedure in order. Work top to bottom:

1. Record identifiers as you go (section 1) and describe what the run covers (section 2).
2. Record the CI evidence (section 3).
3. Record the manifest (section 4), then the focused and full-tier results (section 5).
4. Record the source build results (section 6) and the ASAN results (section 7).
5. Record the benchmark results and the comparison (section 8).
6. Sign off once every procedure step is evidenced (section 9).

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

> Records [procedure step 1: Confirm CI](./RELEASE_TESTING.md#1-confirm-ci).

Link the CI, superbuild, and release-artifact runs that back this validation.

| Evidence | Link | Status |
|---|---|---|
| TheRock multi-arch CI hipdnn test step (`<gpu-family>`) | `<job log URL>` | `<Passed; e.g. 100% tests passed, 0 failed out of N>` |
| hipDNN Superbuild CI | `<actions run URL>` | `<Passed>` |
| Final TheRock / release artifact run | `<actions run URL>` | `<Passed / TBD>` |
| Final sample run artifact / log | `<actions run URL>` | `<Passed / TBD>` |

> **Scope note:** The CI job shows the `standard` tier only. Section 5 records the release runs on the selected artifacts: the focused filters **and** the `full`-tier installed CTest suites.

---

## 4. Replication setup

> Records [procedure step 2: Obtain a ROCm build](./RELEASE_TESTING.md#2-obtain-a-rocm-build-with-the-hipdnn-tests) and [step 3: Record the build manifest](./RELEASE_TESTING.md#3-record-the-build-manifest).

Paste the manifest output from step 3 (ROCm version, TheRock build commit, CI run, and the `rocm-libraries` `pin_sha`):

```text
<paste the manifest output>
```

---

## 5. Test commands and expected output

> Records [procedure step 4: focused checks](./RELEASE_TESTING.md#4-run-focused-checks-for-the-delivered-feature) and [step 5: the full tier](./RELEASE_TESTING.md#5-run-the-full-tier-in-every-test-directory).

The core of the record: command and observed-output blocks. Paste the **actual** output you saw, one block per focused area from section 2 and one per full-tier test directory.

```text
<command run>
[  PASSED  ] <N> tests.
100% tests passed, 0 tests failed out of <N>
```

---

## 6. Build from source

> Records [procedure step 6: Build from source](./RELEASE_TESTING.md#6-build-from-source).

- **ROCm version built against**: `<hipconfig --version output>`
- **Flow**: `<preset, or preset and explicit-prefix>`

```text
<paste the ctest summary, e.g. 100% tests passed, 0 tests failed out of N>
```

---

## 7. ASAN results

> Records [procedure step 7: Run ASAN](./RELEASE_TESTING.md#7-run-asan).

- **GPU family**: `<gfx...>`
- **Pass / skip / fail counts**: `<N / N / N>`

```text
<paste the ctest summary and any sanitizer report>
```

---

## 8. Performance results

> Records [procedure step 8: Run performance benchmarks](./RELEASE_TESTING.md#8-run-performance-benchmarks).

For each GPU family, record the results file name and paste the comparison against the previous release.

- **Results file**: `<headline-<version>-<arch>.json>`
- **dnn-benchmarking version**: `<commit or tag>`
- **Baseline**: `<previous release's results file for the same GPU family>`
- **PyTorch validation** (`--validate pytorch`): `<passed, or the failed rows with an explanation or issue link>`

Observed `dnn-benchmark compare` output:

```text
<paste the compare output>
```

| Flagged regression | Explanation or tracking issue |
|---|---|
| `<graph / engine>` | `<reason or issue link>` |

---

## 9. Final validation checklist

Sign-off gate. Every box must be checked with real evidence before marking the milestone verified.

- [ ] Filled section 1 with the merged release/nightly **build identifier(s)** and the physical GPU/ASIC family used.
- [ ] Confirmed the delivery commit is contained in the artifact's manifest `pin_sha` (section 4).
- [ ] Replaced every `TBD` evidence link in section 3 with the final CI / release-artifact run and sample log.
- [ ] Recorded the source build (section 6) and the ASAN pass, skip and fail counts (section 7).
- [ ] Ran the focused gtest/ctest filters **and** the `full` tier in every installed test directory, on every supported GPU family.
- [ ] Pasted the passing snippets into section 5 (elapsed times updated if rerun on final artifacts).
- [ ] Confirmed all [procedure](./RELEASE_TESTING.md#1-confirm-ci) prerequisites (CI green, documentation/changelog current) are satisfied.
- [ ] Recorded the headline benchmark results for each GPU family, and explained or tracked every regression `compare` flagged (section 8).

**Tips for a defensible record:**

- Paste exact output, including failing messages if any; do not summarize away detail.
- Note environmental factors (GPU family, driver/ROCm version) that could affect results; the artifact identifier captures most of this.
- If a test consistently skips (e.g. GPU-gated or ASAN-incompatible), record why rather than deleting it. See [Environment](../Environment.md#environment-variables) for enabling logging when a run needs deeper insight.
