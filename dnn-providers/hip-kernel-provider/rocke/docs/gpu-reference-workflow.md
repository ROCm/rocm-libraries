<!--
Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
SPDX-License-Identifier: MIT
-->

# Adding or updating a pinned GPU reference

This is the authoritative authoring and publication workflow for rocKE pinned
GPU references. Use it when adding coverage, replacing a baseline, or enrolling
an operation on another architecture. The [SDPA](sdpa-test-reference.md) and
[convolution](conv-test-reference.md) guides own their numerical contracts and
coverage descriptions; they link here for the procedure.

## Decide what must change

| Change | Required work |
|---|---|
| Add or change cases on an existing operation/architecture | Update the cohort and coverage checks, qualify the complete resulting cohort, replace that pair's archive, lock, and DVC pointer |
| Replace the frozen baseline implementation | Export the selected committed revision and independently qualify the complete cohort; replace the archive, lock, and pointer |
| Add an architecture | Implement its adapter, enroll qualification support, qualify on that target, then enroll publication and provider test selection |
| Add an operation | Define its numerical contract, oracle, adapter, and replay support before following qualification and publication below |
| Reorganize host tests, documentation, or installation without changing the qualified contract | Validate the affected code; no replacement bundle solely for that reorganization |

Each operation/architecture has its own archive and trusted lock. A coverage
change cannot be validated using the previous bundle: verification requires the
manifest's case set and contracts to match the current cohort. Do not append a
case or edit a budget in a qualified manifest to make verification pass.
Requalify the complete cohort; the CLI does not incrementally extend bundles.

## 1. Choose coverage and implement the contract

Use cases that exercise distinct production paths, rather than counting shapes
alone. Record the dtype, geometry, dispatch choice, launch grid, epilogue, and
independent oracle relevant to each addition. Preserve existing cases unless
removal is intentional and explained.

For example, the convolution coverage request from
[PR #13106](https://github.com/ROCm/rocm-libraries/pull/13106#discussion_r4200111867)
motivates output-channel coverage: K=96 exercises a second N tile and its partial tail
when tile_n=64; an odd K exercises the default epilogue. Check both FP16 and BF16.
The [convolution guide](conv-test-reference.md#coverage-and-execution) records
the current cohort; its architecture adapter is the source of exact case geometry.
Assert the intended dispatch properties in host tests and verify the exported
kernel metadata during qualification. Different runtime geometries can share a
kernel binary while still exercising distinct address calculations.

For existing operations, edit
`library/tests/<operation>_reference/architectures/<arch>/__init__.py` and its
`CASES`. The adapter exports `NAME`, `FAMILIES`, `CASES`, `CASE_BY_ID`, `prepare`,
`launch`, and `exported_kernel`. It must use the production dispatch/launch path
and freeze the complete replay ABI. Keep the independent oracle independent of
production indexing and geometry helpers.

For a new architecture, add it to the operation's `architectures/registry.json`
so offline tools can load the adapter. This enables qualification only; it does
not enroll installation. Confirm production support on the actual target before
reusing another architecture's adapter. Never transfer error bounds or output
digests across architectures.

For a new operation, implement its case model, deterministic quantized inputs,
independent oracle, fixed comparison metric and budgets, qualification CLI, and
source/replay worker. Reuse `reference_common` for archives, numerics, snapshots,
and worker isolation. Add operation support to the shared artifact CLI and
pytest selector, and add the operation to the explicit CMake installation and
CTest loops. Add host contract tests and GPU failure controls as well as passing
numerical cases. New output structures or nondeterministic kernels require a
numerical/replay contract that supports them; the current contracts cannot be
assumed to apply. See the [extension strategy](gpu-ci-pinned-rocke-test-reference-plan.md#6-extend-coverage-deliberately).

## 2. Prepare reproducible qualification inputs

Commands below use Bash and run from the rocKE root unless stated otherwise.
Choose one operation and architecture, and an unused working directory outside
the checkout. Replace the placeholder paths before running:

```bash
REFERENCE_OPERATION=conv                 # sdpa or conv
REFERENCE_ARCH=gfx942
REFERENCE_WORK='<outside-checkout-work-directory>'
REFERENCE_REPOSITORY='<rocm-libraries-checkout>'
REFERENCE_REVISION='<full-committed-baseline-sha>'
```

Run on the selected GPU through the site's supported allocation/container
workflow. Record the container identity and retain logs outside the bundle.
Qualification needs NumPy, HIP, and COMGR; test execution also needs pytest and
pytest-timeout. A login host without GPU access cannot qualify a GPU baseline.

```bash
python "library/tests/run_${REFERENCE_OPERATION}_reference.py" snapshot \
  --repository "$REFERENCE_REPOSITORY" --revision "$REFERENCE_REVISION" \
  --output "$REFERENCE_WORK/baseline"
```

The snapshot contains committed production sources and their digests. It does
not include uncommitted kernel changes. The qualification command uses the
invoking checkout's reference contract and adapter with those baseline production
sources, and freezes replay support into the bundle. Retain the authoring revision
and any pending reference changes with the evidence, as well as the baseline SHA.
Select a newer committed baseline if the added cases need newer kernel support.

## 3. Qualify against independent references

```bash
python "library/tests/run_${REFERENCE_OPERATION}_reference.py" qualify \
  --arch "$REFERENCE_ARCH" --baseline "$REFERENCE_WORK/baseline" \
  --output "$REFERENCE_WORK/bundle" --repetitions 3
```

Qualification executes the baseline repeatedly, checks deterministic output
identities, and computes independent reference answers and conservative error
bounds. The manifest records inputs, outputs, reference and compiler identities,
target, launch metadata, and payload digests. Inputs and answer tensors are not
shipped. The generated `qualification-lock.json` is a candidate trust anchor,
not an automatically approved replacement for the committed lock.

For convolution, optionally add `--torch-reference` to the qualification command
in a separate offline environment with Torch installed. This checks CPU Torch
float64 and float32 answers as well as NumPy and includes their bounds in the
certificate. A CPU Torch build suffices. SDPA has no equivalent flag. The
convolution oracle-only audit is:

```bash
python -m pytest library/tests/conv_reference/check_torch_reference.py -v
```

Review every case's target, geometry, grid, epilogue, output determinism, oracle
agreement, and remaining error budget. For coverage additions, check that the
new cases actually exercise the intended paths. Do not loosen tolerances simply
to obtain a passing certificate. Every replacement baseline needs independent
qualification; agreement with its predecessor alone is insufficient.

## 4. Verify the candidate without Torch

Switch to a verification environment without Torch, preserving the workspace
variables above. Verify both GPU outputs and the failure controls before
promoting the candidate:

```bash
python -c 'import importlib.util; assert importlib.util.find_spec("torch") is None'
python "library/tests/run_${REFERENCE_OPERATION}_reference.py" verify \
  --arch "$REFERENCE_ARCH" --bundle "$REFERENCE_WORK/bundle" \
  --lock "$REFERENCE_WORK/bundle/qualification-lock.json" --current-root .
python -m pytest library/tests/test_reference_common.py \
  "library/tests/test_${REFERENCE_OPERATION}_reference_contract.py" -q
```

For the GPU pytest suite, pass the candidate bundle and lock explicitly:

```bash
python -m pytest "library/tests/test_${REFERENCE_OPERATION}_pinned_reference.py" \
  --rocke-reference-operation "$REFERENCE_OPERATION" \
  --rocke-reference-arch "$REFERENCE_ARCH" \
  --rocke-reference-bundle "$REFERENCE_WORK/bundle" \
  --rocke-reference-lock "$REFERENCE_WORK/bundle/qualification-lock.json" -v -rs
```

Explicit operation/architecture selection makes the suite required, including
failure on a missing bundle. Bundle and lock overrides require that selection; a lock
override also requires a bundle path. Without a lock override, verification uses
the committed lock. Without either path override, tests use the standard bundle
location and committed lock. The selected operation's GPU fixture alone consumes
these paths. They do not change what CMake installs. Check that numerical cases
and the changed-current-output, changed-baseline-output, and suppressed-launch
controls all execute.

## 5. Review, pack, and publish the replacement

After reviewing the qualification evidence, copy the candidate
`qualification-lock.json` to
`library/tests/<operation>_reference/architectures/<arch>/baseline_lock.json`.
Then pack against that reviewed lock:

```bash
python library/tests/reference_common/artifact.py pack \
  --operation "$REFERENCE_OPERATION" --bundle "$REFERENCE_WORK/bundle" \
  --lock "library/tests/${REFERENCE_OPERATION}_reference/architectures/$REFERENCE_ARCH/baseline_lock.json" \
  --archive "library/tests/reference_bundles/$REFERENCE_OPERATION/$REFERENCE_ARCH.tar.gz"
```

From the rocm-libraries root, track and upload that one archive:

```bash
REFERENCE_DVC="dnn-providers/hip-kernel-provider/rocke/library/tests/reference_bundles/$REFERENCE_OPERATION/$REFERENCE_ARCH.tar.gz"
dvc add "$REFERENCE_DVC"
dvc push "$REFERENCE_DVC.dvc"
```

The `storage` remote requires write access for publication; downloads support
anonymous access. If upload is unavailable, retain the candidate and arrange
publication with an authorized maintainer before merging its pointer. A local
cache hit does not establish remote availability: pull the new pointer from an
isolated checkout with an empty DVC cache and validate the downloaded archive
against the reviewed lock. Do not delete an existing working cache to perform
this check. `dvc add` may stage metadata automatically in this repository.
Commit the pointer and lock, not the archive or extracted payload.

For a newly published architecture, add it to
`ROCKE_PUBLISHED_SDPA_ARCHITECTURES` or `ROCKE_PUBLISHED_CONV_ARCHITECTURES` in
[`PublishedGpuReferences.cmake`](../platform/cmake/PublishedGpuReferences.cmake).
For new operations, add the corresponding list alongside the CMake support from
step 1. Updating coverage on an already published pair needs no enrollment change.
CMake consumes only standard archive and baseline-lock paths. Missing or corrupt
published data fails configuration; it must not silently remove coverage.

## 6. Validate installation and provider selection

This validation can run before committing or merging the replacement, and before
uploading it to DVC. CMake reads the files in the working checkout: place the
reviewed candidate lock and packed archive at the standard paths from step 5,
and update `PublishedGpuReferences.cmake` locally if adding a new published pair.
Then use the normal configure/install path below. CMake does not require a Git
commit, a DVC pointer, or remote availability to install a local archive.

This exercises the actual installation path without candidate-install overrides.
DVC publication and an empty-cache download check are still required before
merging, so other checkouts and CI can retrieve the same bundle. Installation
checks alone do not validate TheRock's final artifact splitting; check the
assembled artifacts as described below when validating packaging.

Run the source publication/install checks:

```bash
python -m pytest platform/tests/test_gpu_references.py -q
```

Follow the existing [build procedure](../BUILDING.md) to configure and install
with `ROCKE_INSTALL_TEST_GPU_REFERENCES=ON`. Fetch all published archives needed
by that configuration. In a fresh installed tree, verify the candidate manifest
and lock match the reviewed files, then run the
[Torch-free installed reference procedure](../TESTING.md#running-installed-reference-tests-without-torch)
on matching hardware with source paths removed from the environment.

The harness and committed locks belong in the generic test artifact. Each
payload belongs under
`bin/hip_kernel_provider/engines/test_arch_content/rocke/<operation>/<arch>/` in
its architecture's test artifact, outside release library payloads. Test the
assembled generic and matching architecture artifacts when validating packaging.

New published pairs get separate `rocke_<operation>_gpu_<arch>_pytest` CTest
entries. Add each new entry to the provider's
[`ROCKE_ENGINE_test_categories_external.yaml`](../../ROCKE_ENGINE_test_categories_external.yaml)
and verify the real provider selector includes it. Adding cases to an existing
pair uses its existing entry. Preserve explicit architecture selection and check
that expected GPU cases did not skip; a passing empty selection proves nothing.

Update the operation's coverage description and the
[test inventory](../platform/tests/README.md#installed-pinned-reference-suites).
Review case changes, pointer, lock, and qualification evidence together. Report
host cases, GPU comparisons, negative controls, and skips separately. The bundle
and matching source cohort must be available together when CI runs. Keep this
guide as the single procedure; operation guides should explain their contracts
and link here rather than reproduce these steps.
