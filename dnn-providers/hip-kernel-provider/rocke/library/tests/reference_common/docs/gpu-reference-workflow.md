<!--
Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
SPDX-License-Identifier: MIT
-->

# Adding or updating a pinned GPU reference

This is the authoritative authoring and publication workflow for rocKE pinned
GPU references. Use it when adding coverage, replacing a baseline, or enrolling
an operation on another architecture. The [SDPA](../../sdpa_reference/docs/sdpa-test-reference.md) and
[convolution](../../conv_reference/docs/conv-test-reference.md) guides own their numerical contracts and
coverage descriptions; they link here for the procedure.

Read [kernel arguments and compatibility](README.md#kernel-arguments-and-compatibility)
for how the current kernel and frozen baseline keep separate launch ABIs, and
[baseline replacement](README.md#when-to-replace-the-baseline) for when a new
qualification is needed. The [quickstart](quickstart.md) gives short recipes
for common tasks, and the [glossary](README.md#glossary) defines terms. When a
pinned test fails, start at [triage](#triage).

## Prerequisites

A change that replaces a bundle, as described in the table below, needs:

- **A GPU of each processor you qualify.** Certificates do not transfer between
  processors, even ones sharing a generic code object (step 2).
- **A baseline revision reachable from `develop`.** Merge kernel changes first,
  then snapshot the merged commit. Pull requests land as squash commits, so a
  feature-branch commit is not in `develop`'s history; once its branch is
  deleted, other checkouts cannot snapshot it again and the requalification
  check below skips the pair.
- **Write access to the DVC `storage` remote, or a maintainer who has it.**
  Step 5 covers the upload and how to request publication.

## Decide what must change

| Change | Required work |
|---|---|
| Change the current kernel or its launch arguments without changing the enrolled cases or numerical contract | Update its caller and any affected test adapter; verify against the existing baseline. Update export bindings before qualifying that revision as a future baseline |
| Add or change cases on an existing operation/architecture | Update the cohort and coverage checks, qualify the complete resulting cohort, replace that pair's archive, lock, and DVC pointer |
| Replace the frozen baseline implementation | Export the selected committed revision and independently qualify the complete cohort; replace the archive, lock, and pointer |
| Change the baseline compiler target, such as gfx1201 to gfx12-generic | Recompile from a committed source revision supporting that target, independently qualify, and replace the archive, lock, and pointer |
| Add an architecture | Implement its adapter, enroll qualification support, qualify on that target, then enroll publication and provider test selection |
| Add an operation | Define its numerical contract, oracle, adapter, and replay support before following qualification and publication below |
| Reorganize host tests, documentation, or installation without changing the qualified contract | Validate the affected code; no replacement bundle solely for that reorganization |

Each operation/architecture has its own archive and trusted lock. A coverage
change cannot be validated using the previous bundle: verification requires the
manifest's case set and contracts to match the current cohort. Do not append a
case or edit a budget in a qualified manifest to make verification pass.
Requalify the complete cohort; the CLI does not incrementally extend bundles.

Before qualifying on a GPU, check what a cohort edit changes. From the rocKE
root, these commands run without a GPU or Torch:

```bash
python library/tests/run_conv_reference.py cases --arch gfx942
python library/tests/run_conv_reference.py diff --arch gfx942
```

`cases` lists the enrolled case ids with their tolerances and case contracts;
add `--json` for tools. `diff` authenticates the published bundle's manifest
against the committed lock, matches cases by contract as verification does, and
names enrolled cases the bundle does not record, recorded cases that are no
longer enrolled, and changed tolerances or budgets. It exits with status 1 on
any difference, and with status 2 and one message when it cannot read or
authenticate the lock or bundle. It reads the DVC archive, or the extracted
bundle when no archive is present; pass `--bundle` and `--lock` to check a
candidate instead. `diff` reads recorded device targets with the rocKE platform
package: install rocKE or set `PYTHONPATH=platform/python`.

`library/tests/test_reference_common.py`, which is also installed as
`rocke_reference_common_pytest`, runs the same comparison for every pair with a
committed lock. A cohort edit without a replacement bundle therefore fails host
tests, not only the matching GPU suite. The check skips a pair only when no
archive, extracted bundle, or installed bundle is present.

The same file also checks that each such pair can still be requalified at its
locked baseline. Because qualification runs the current adapter with the
baseline's production sources, the check exports the lock's `baseline_revision`
from the checkout's git history and prepares every enrolled case with the
current adapter, without compiling or launching a kernel. If you port an adapter
to a newer production API, keep the older API working or requalify the pair at
a newer baseline. The check names each case it cannot prepare, with the error.
It skips in installed trees, without git, and when the revision is absent, as in
a shallow clone.

## 1. Choose coverage and implement the contract

Use cases that exercise distinct production paths, rather than counting shapes
alone. Record the dtype, geometry, dispatch choice, launch grid, epilogue, and
independent oracle relevant to each addition. Preserve existing cases unless
removal is intentional and explained.

For example, the convolution coverage request from
[PR #13106](https://github.com/ROCm/rocm-libraries/pull/13106#discussion_r4200111867)
motivates output-channel coverage: K=96 exercises a second N tile and its partial tail
when tile_n=64; an odd K exercises the default epilogue. Check both FP16 and BF16.
The [convolution guide](../../conv_reference/docs/conv-test-reference.md#coverage-and-execution) records
the current cohort; its architecture adapter is the source of exact case geometry.
Assert the intended dispatch properties in host tests and verify the exported
kernel metadata during qualification. Different runtime geometries can share a
kernel binary while still exercising distinct address calculations.

For existing operations, edit
`library/tests/<operation>_reference/architectures/<arch>/__init__.py` and its
`CASES`. The gfx1151, gfx1200, and gfx1201 SDPA adapters instead share `CASES`
from `sdpa_reference/implementations/wmma.py`: editing that cohort changes it
on every WMMA processor, so each published WMMA pair needs requalification and
a replacement bundle. The adapter exports `NAME`, `CASES`, `CASE_BY_ID`,
`prepare`, `launch`, and `exported_kernel`. `CASE_BY_ID` indexes `CASES` with
`reference_common.cases.index_cases`, so colliding case ids fail on import. An
adapter that compiles a family code object also exports `COMPILER_TARGET`, and
its exported kernel metadata records the same `compiler_target`. It must use the
production dispatch/launch path and freeze the complete replay ABI. Keep the
independent oracle independent of production indexing and geometry helpers.

For a new architecture, add it to the operation's `architectures/registry.json`
so offline tools can load the adapter. This enables qualification only; it does
not enroll installation. Confirm production support on the actual target before
reusing another architecture's adapter. Never transfer error bounds or output
digests across architectures.

For a new operation, implement its case model, deterministic quantized inputs,
independent oracle, fixed comparison metric and budgets, and architecture
adapters. Its `Case` is a frozen dataclass with `id`, `dtype`, `tolerance`,
`input_shapes`, and `output_shape`. Its `dtype` must be `fp16` or `bf16`: the
shared worker stores the inputs and output in that 2-byte format. The worker
also requires exactly one kernel launch per repetition and a single output, and
qualification and verification require the baseline's repeated outputs to be
bit-identical. Support for multiple launches or outputs is deferred until a
concrete kernel needs it. `reference_common` provides archives, numerics,
snapshots, worker isolation, and the qualification, verification, worker, and
GPU test code that every operation shares. The package declares only its own
parts:

| File | Declares |
|---|---|
| `cli.py` | `OPERATION`, a `reference_common.cli.ReferenceOperation` naming the contract, architectures, manifest reference description, `Metric`, and an optional `TorchReference` for offline Torch cross-checks; module aliases for its `qualify`, `load_bundle`, `manifest_key`, `verify_case`, and `main` |
| `worker.py` | `run(request, output)`, which passes a `WorkerOperation` with the adapter's launch call and the frozen replay-argument binding to `reference_common.worker.run_case` |
| `test_<operation>_pinned_reference.py` | A `reference_bundle` fixture from `select_bundle` and the tests from `pinned_reference_tests`, with a negative-control perturbation that exceeds the comparison limit in the operation's metric |

Qualification freezes the package's `contract.py`, `worker.py`, `architectures/`,
and `implementations/`, with `reference_common/` except its pytest-only
`pinned_tests.py` and `pytest_support.py`, into each bundle; the worker must not
import anything else from the tests tree.

No shared list names the operations: each
`library/tests/<operation>_reference/` package with `architectures/registry.json`
is an operation for the shared artifact CLI, the `--rocke-reference-operation`
pytest option, and the CMake installation and CTest loops. Record the operation
id its bundle manifests declare in the package's `operation.json`, for example
`{"manifest_operation": "conv-fwd"}`, and add its publication list (step 5).
CTest and this workflow find `test_<operation>_reference_contract.py`,
`test_<operation>_pinned_reference.py`, and `run_<operation>_reference.py` by
name. Add host contract tests and GPU failure controls as well as passing
numerical cases. New output structures or nondeterministic kernels require a
numerical/replay contract that supports them; the current contracts cannot be
assumed to apply. See the [extension strategy](gpu-ci-pinned-rocke-test-reference-plan.md#6-extend-coverage-deliberately).

Run the host contract tests before booking a GPU. They need no GPU or Torch
and catch cohort, adapter, and frozen-ABI problems before qualification:

```bash
python -m pytest library/tests/test_reference_common.py \
  library/tests/test_<operation>_reference_contract.py -q
```

After a `CASES` edit, `test_published_bundles_record_the_enrolled_cohorts` is
expected to fail for the edited pair until step 5 replaces its bundle and lock.
Fix every other failure before qualifying.

## 2. Prepare reproducible qualification inputs

Commands below use Bash and run from the rocKE root unless stated otherwise.
Choose one operation and architecture, and an unused working directory outside
the checkout. Replace the placeholder paths before running:

```bash
REFERENCE_OPERATION=conv                 # any registered operation, such as sdpa
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

### Generic code objects and processor qualification

The gfx1200 and gfx1201 SDPA adapters compile the same WMMA cohort for
`gfx12-generic`. `--arch` still selects the **execution processor**, not the
compiler target. Qualify separately on each processor: a generic code object
being loadable does not establish identical output bits or error bounds there.

The manifest records the actual HIP target, including features, in
`device_target`, and the WMMA kernel's compile target in
`kernel.compiler_target`. Verification requires the selected processor's
certificate, its qualified output digest, matching old/current device features,
and the generic compiler target. A gfx1201 certificate cannot be relabeled as
gfx1200, and the old gfx1201-specific archive cannot serve as a generic baseline.
Use a committed source snapshot containing generic WMMA support; the original
`c6f41a64e12f8e1a51afb4ebce1b9dee256aaaed` baseline predates that support.

Keep the archive, lock, installation directory, and CTest entry keyed by the
qualified processor (`gfx1200` or `gfx1201`). The artifact splitter recognizes
those directories. When both are published, a family job can download both but
runs only the matching processor's suite. Until a processor is published, its
pinned CTest entry is absent. See [family CI selection](../../sdpa_reference/docs/sdpa-test-reference.md#gfx120x-family-ci-selection)
for the skip and failure rules. Do not install the payload under `gfx12-generic`: that name
does not identify a processor artifact in the existing splitter. Enrollment in
the qualification registry alone does not publish a processor's suite; add it
to the CMake publication list only after qualification and archive publication.

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
certificate. A CPU Torch build suffices. SDPA also accepts `--torch-reference`
for CPU Torch float64/float32 SDPA, or `--torch-reference cuda` for the matching
ROCm GPU. Both operations set B to the largest measured baseline error bound
against all enabled references. No fixed percentage margin is reserved; the
comparison limit C is rounded strictly below T-B, so D<=C implies D+B<T.
The NumPy–Torch gap remains diagnostic. Existing published bundles retain
their old authenticated limits until replaced through this workflow. See the [SDPA contract](../../sdpa_reference/docs/sdpa-test-reference.md#optional-torch-reference-agreement-during-qualification)
for scope and reference-only auditing. The
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

Until step 5 replaces the published archive and lock, the host tests report
an edited cohort as differing from the published bundle; that check must pass
after step 5.

`verify` reads an extracted bundle directory; given a `.tar.gz` archive, it
prints the `artifact.py unpack` command to run first. To iterate on a failing
case, add `--case <id>`, repeated as needed, to run only those cases. The output
states that such a subset run is not a full verification; verify the complete
cohort before promoting a candidate.

For the GPU pytest suite, pass the candidate bundle and lock explicitly:

```bash
python -m pytest "library/tests/test_${REFERENCE_OPERATION}_pinned_reference.py" \
  --rocke-reference-operation "$REFERENCE_OPERATION" \
  --rocke-reference-arch "$REFERENCE_ARCH" \
  --rocke-reference-bundle "$REFERENCE_WORK/bundle" \
  --rocke-reference-lock "$REFERENCE_WORK/bundle/qualification-lock.json" -v -rs
```

Explicit operation/architecture selection makes the matching processor's suite
required, including failure on a missing bundle. A suite for another processor
in the job's requested artifact list skips before loading its bundle. Bundle and
lock overrides require that selection; a lock
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

Re-run `python -m pytest library/tests/test_reference_common.py -q`; it now
compares the enrolled cohort with the new archive and reviewed lock.

From the rocm-libraries root, track that one archive:

```bash
REFERENCE_DVC="dnn-providers/hip-kernel-provider/rocke/library/tests/reference_bundles/$REFERENCE_OPERATION/$REFERENCE_ARCH.tar.gz"
dvc add "$REFERENCE_DVC"
```

`dvc add` may stage metadata automatically in this repository. Commit the
pointer and lock, not the archive or extracted payload. Committing a replacement
pointer does not upload its archive.

Maintainers with write access to the DVC `storage` remote upload archives.
Without that access, retain the candidate and request publication on the pull
request from the rocKE library code owners (`@ROCm/rocke-lib`, which owns
`library/` in [`CODEOWNERS`](../../../../../../../.github/CODEOWNERS)). The
archive must be uploaded before its pointer merges.

The committed `.dvc/config` sets `allow_anonymous_login = true` on its remotes
so downloads need no credentials. DVC then sends unsigned requests and ignores
any credentials, so an upload fails with `AccessDenied` even with write access.
Once per checkout, turn that off for `storage` in the git-ignored
`.dvc/config.local`:

```bash
dvc remote modify --local storage allow_anonymous_login false
```

Supply credentials through the standard AWS environment variables, such as
`AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY`, or a named profile selected with
`AWS_PROFILE`. Never write them into a DVC config file or commit them. Downloads
in that checkout then need credentials too;
`dvc remote modify --local --unset storage allow_anonymous_login` restores
anonymous reads. Reference pointers name no remote, so they upload to the default
`storage` remote. Pass the pointer, or several, so the push uploads only those
archives:

```bash
dvc push "$REFERENCE_DVC.dvc"
```

A local cache hit does not establish remote availability: pull the new pointer
from an isolated checkout with an empty DVC cache and validate the downloaded
archive against the reviewed lock. A fresh checkout has no `.dvc/config.local`,
so that pull is anonymous, as for any reader without credentials. Do not delete
an existing working cache to perform this check. CI fetches DVC objects during
source preparation, before GPU suite selection, so the replacement must be
available remotely even for a job that later skips that processor's suite.

For a newly published architecture, add it to the operation's
`ROCKE_PUBLISHED_<OPERATION>_ARCHITECTURES` list, such as
`ROCKE_PUBLISHED_CONV_ARCHITECTURES`, in
[`PublishedGpuReferences.cmake`](../../../../platform/cmake/PublishedGpuReferences.cmake).
Every operation package needs a list; a new operation that publishes nothing yet
sets its list to `""`. Updating coverage on an already published pair needs no
enrollment change. CMake consumes only standard archive and baseline-lock paths.
Missing or corrupt published data, a list without an operation package, and a
package without a list or `operation.json` fail configuration; they must not
silently remove coverage.

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

Fetch every published archive first; a missing one fails configuration. Then,
from the rocm-libraries root and with the provider's HIP and hipDNN SDK
dependencies available, configure the provider with rocKE enabled as in the
[build quick start](../../../../BUILDING.md#quick-start-cmake-managed-environment),
build, and install. Provider builds default `ROCKE_INSTALL_TEST_GPU_REFERENCES`
to ON; pass it explicitly because an existing build directory keeps its cached
value:

```bash
cmake -S dnn-providers/hip-kernel-provider -B "$REFERENCE_WORK/build" \
  -DHIPKERNELPROVIDER_ENABLE_ROCKE=ON -DROCKE_INSTALL_TEST_GPU_REFERENCES=ON
cmake --build "$REFERENCE_WORK/build"
cmake --install "$REFERENCE_WORK/build" --prefix "$REFERENCE_WORK/install"
```

The installed provider tests are under
`$REFERENCE_WORK/install/bin/hip_kernel_provider/`. In a fresh installed tree,
verify the candidate manifest and lock match the reviewed files, then run the
[Torch-free installed reference procedure](../../../../TESTING.md#running-installed-reference-tests-without-torch)
on matching hardware with source paths removed from the environment.

The harness and committed locks belong in the generic test artifact. Each
payload belongs under
`bin/hip_kernel_provider/engines/test_arch_content/rocke/<operation>/<arch>/` in
its architecture's test artifact, outside release library payloads. Test the
assembled generic and matching architecture artifacts when validating packaging.

### Processor selection in TheRock jobs

TheRock owns build-family membership and expands it into the job's
`AMDGPU_TARGETS`. Reference tests consume that comma-separated list of concrete
processors, and also accept a CMake-style semicolon list for local runs; they do
not define their own family aliases or infer qualification from
`AMDGPU_FAMILIES`. A family name describes build and artifact selection,
not a promise that every member has a qualified reference bundle.

The shared test artifact contains CTest entries for all published reference
pairs. Before accessing a bundle, each entry checks its processor against the
requested targets and the actual GPU:

| Condition | Result |
|---|---|
| The suite's processor is outside `AMDGPU_TARGETS` | Skip with exit 77 |
| The GPU is another processor in `AMDGPU_TARGETS` | Skip with exit 77, even if that GPU has no reference cohort or published bundle |
| The GPU matches the suite's processor | Run; missing or corrupt required reference data fails |
| The suite's processor is requested, but the GPU is absent or outside `AMDGPU_TARGETS` | Fail with a required-GPU error |

When `AMDGPU_TARGETS` is absent, selection uses the actual processor: a different
GPU skips the suite, and a missing GPU fails an explicitly selected suite.
Family-name spelling does not change those decisions. Source pytest without an
explicit operation/architecture request also skips processors without an enrolled
cohort or installed bundle. An explicit candidate-verification request still
requires its own qualified bundle and lock, including on qualification-only
processors such as gfx1200. A skip records absent coverage; it does not qualify
another processor or permit reuse of another processor's certificate.

### Enroll provider test selection

New published pairs get separate `rocke_<operation>_gpu_<arch>_pytest` CTest
entries, and a new operation also gets `rocke_<operation>_reference_unit_pytest`.
The provider's
[`ROCKE_ENGINE_test_categories_external.yaml`](../../../../../ROCKE_ENGINE_test_categories_external.yaml)
selects both by pattern for a lowercase operation name and a `gfx` processor, so
it needs no edit; verify that the real provider selector includes the new
entries. Adding cases to an existing pair uses its existing entry. Preserve
explicit architecture selection and check that expected GPU cases did not skip;
a passing empty selection proves nothing.

Update the operation's coverage description, and add a new operation's CTest
entries to the
[test inventory](../../../../platform/tests/README.md#installed-pinned-reference-suites).
The inventory points to code for cases and published pairs, so a new case or
architecture needs no edit there. Review case changes, pointer, lock, and
qualification evidence together. Report host cases, GPU comparisons, negative
controls, and skips separately. The bundle and matching source cohort must be
available together when CI runs. Keep this guide as the single procedure;
operation guides should explain their contracts and link here rather than
reproduce these steps.

## Triage

A pinned reference failure names the operation, architecture, and case, and
the side that failed: the frozen baseline (replay), the current kernel
(source), or the baseline snapshot (qualification). Most messages also give a
likely cause and a next step. Find the row with text from the message; only
rows for the kernel author implicate a kernel change.

| Message contains | Likely cause | Who acts | Next step |
|---|---|---|---|
| `is outside its qualified budget`, `exceeds remaining limit` | The current kernel or its dispatch changed the output by more than the case's comparison limit | Kernel author | Rerun the case with the `verify --case` command the message prints, then [check the change against the independent reference](#check-a-changed-output-against-the-independent-reference) |
| `current kernel (source) worker failed` | The adapter no longer matches a renamed or re-signed dispatch, build, or launch API, or the current kernel fails to compile, writes `non-finite or unwritten output`, does not launch once per repetition, or exceeds the request timeout | Kernel author | Read the worker log that the message includes. Update the adapter it names for a changed API, or fix the kernel |
| `reference qualification failure` | The frozen baseline's output no longer matches its qualified digest: the GPU driver, ROCm runtime, or firmware differs from the qualified environment. The current kernel did not run | GPU environment owner, not the kernel author | Rerun to confirm, then compare the environment with the qualified one. Do not change the kernel or a tolerance; an environment that is meant to change the output needs a replacement baseline qualified in it |
| `frozen baseline (replay) worker failed` | This environment cannot run the frozen runtime and code object, such as through its GPU driver, ROCm runtime, or device visibility | GPU environment owner | Read the worker log and compare the environment with the qualified one |
| `differs from the qualified target`, `used different target features` | The GPU's target features, such as xnack or sramecc, differ from the qualified GPU's or between the two workers | GPU environment owner | Run on a GPU configured like the qualified one, or qualify a bundle for this target |
| `reference bundle is missing`, `no extracted`, `does not match the pinned lock`, `has no manifest.json`, `missing, modified, or extra files`, `cannot be read`, `baseline identity mismatch`, `targets a different architecture`, `wrong reference operation` | The bundle was not fetched or extracted, is stale (extracted from an earlier archive, partly extracted, or modified), or belongs to another pair | Whoever runs the test | Run the `dvc pull` and `artifact.py unpack` commands the message prints, deleting a stale directory first, or pass the matching bundle and lock. Reinstall an installed tree from one build |
| `reference archive missing`, `reference staging failed`, `No qualified ... baseline lock`, `has no publication list` during CMake configuration | A published archive was not fetched or does not match its lock, or the publication lists, locks, and operation packages disagree | Whoever configures the build, or the author of the publication change | Fetch every published archive before configuring ([step 6](#6-validate-installation-and-provider-selection)). Publish a pair only with its lock, and give each operation package a list and an `operation.json` |
| `does not cover the complete enrolled cohort`, `the enrolled cases differ from`, `tolerance or budget changed`, `different compiler target` | The adapter's `CASES`, a shared cohort, the operation's `Case` fields, or the adapter's `COMPILER_TARGET` changed without a replacement bundle | Author of that change | Revert an unintended change. Otherwise [requalify the complete cohort](#decide-what-must-change) and publish its archive, lock, and DVC pointer. `diff` lists every difference |
| `the current adapter cannot prepare`, `outside the sources of locked baseline` | An adapter update for a newer production API dropped the API of the pair's locked baseline revision, or a new case needs newer kernel support | Author of the adapter or case change | Keep the older API working in the adapter, or requalify the pair at a newer baseline revision |
| `input digest mismatch` | The input generator or the NumPy implementation differs from qualification | Author of the contract change | Restore the generator; an intended change needs requalification of every pair of the operation |
| `GPU unavailable`, `GPU is not available` | An explicitly selected suite found no GPU, or a GPU outside `AMDGPU_TARGETS` | CI job configuration | Run on the selected processor; see [processor selection](#processor-selection-in-therock-jobs) |
| `baseline snapshot (qualification) worker failed`, `is not deterministic`, `leaves no comparison budget` | During qualification: the adapter does not support the snapshot's API, or the baseline kernel fails, is not deterministic, or exceeds the tolerance | Whoever qualifies | Select another committed baseline revision or extend the adapter. Never raise a tolerance to qualify |

A skip is not a failure, but it records missing coverage on that GPU; check
that the expected cases ran.

### Check a changed output against the independent reference

An output outside the comparison limit is not necessarily less accurate: a
result closer to the independent reference can still fall outside the region
allowed around the baseline. To tell, commit the change and qualify that
revision into a scratch directory with
[steps 2 and 3](#2-prepare-reproducible-qualification-inputs), using the commit
as `REFERENCE_REVISION`, without publishing the result. Qualification prints
each case's `old_bound`, here the changed kernel's error against the
independent reference, and fails with `leaves no comparison budget` when that
error reaches the tolerance. Fix such a regression. For an intended numerical
change within the tolerance, qualify a replacement baseline as described above
instead of widening the tolerance.
