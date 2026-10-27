<!--
Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
SPDX-License-Identifier: MIT
-->

# Pinned GPU reference quickstart

Short recipes for the common kernel-author tasks. The
[workflow](gpu-reference-workflow.md) remains the authoritative procedure, and
the [glossary](README.md#glossary) defines the terms used here. When a pinned
test fails, start at [triage](gpu-reference-workflow.md#triage).

Commands use Bash and run from the rocKE root unless marked otherwise. Select
an operation, the processor of the GPU under test, and one of its
[variants](README.md#glossary), one per kernel family, first:

```bash
REFERENCE_OPERATION=conv              # or sdpa
REFERENCE_ARCH=gfx942                 # the GPU's processor, such as gfx950
REFERENCE_VARIANT=implicit_gemm_fwd   # a variant that status lists, such as dense for sdpa
```

An operation's `architectures/registry.json` lists its enrolled processors, and
[`PublishedGpuReferences.cmake`](../../../../platform/cmake/PublishedGpuReferences.cmake)
lists the published pairs. The `run_<operation>_reference.py` command line's
`cases` command needs no GPU or Torch. Its `diff`, `status`, and `verify`
commands also read device target ids with the rocKE platform package, so the
commands below set `PYTHONPATH=platform/python`. `status` prints a line for
each enrolled variant: whether a committed lock publishes it, the other
variants whose cohort is the same, and whether its bundle matches, differs, or
is missing:

```bash
PYTHONPATH=platform/python python "library/tests/run_${REFERENCE_OPERATION}_reference.py" status
```

`qualify`, `verify`, `cases`, and `diff` take `--variant`. Without it, `cases`
lists every variant of the architecture and `diff` compares every published
one. `qualify`, `verify`, `cases --json`, `diff` of an explicit bundle or lock,
and `diff` on an architecture that publishes no variant use the architecture's
only variant; when it has several, they require `--variant` and list the
choices.

## I changed a covered kernel

1. **Know what CI compares.** On the GPU of each published variant, the pinned
   suite replays the frozen baseline for every enrolled case and requires its
   qualified output digest. It then runs the current kernel through the
   variant's adapter and requires every output to stay within the case's
   comparison limit of the replayed output. Three failure controls per variant
   check that the comparison can fail. A kernel is covered when an enrolled
   case reaches it through an adapter; list the cases of each of the
   architecture's variants with:

   ```bash
   python "library/tests/run_${REFERENCE_OPERATION}_reference.py" cases --arch "$REFERENCE_ARCH"
   ```

   A change that stays within those limits keeps the existing baseline. A
   changed launch signature needs its caller and adapter updated, not a new
   baseline; see [kernel arguments](README.md#kernel-arguments-and-compatibility).

2. **Run the host checks.** They need no GPU or Torch:

   ```bash
   python -m pytest library/tests/test_reference_common.py \
     "library/tests/test_${REFERENCE_OPERATION}_reference_contract.py" -q
   ```

   `test_adapters_prepare_against_their_locked_baselines` fails if an adapter
   update for a changed production API drops the API that the variant's locked
   baseline revision uses.

3. **Fetch and extract the published bundle.** From the rocm-libraries root:

   ```bash
   dvc pull "dnn-providers/hip-kernel-provider/rocke/library/tests/reference_bundles/$REFERENCE_OPERATION/$REFERENCE_ARCH/$REFERENCE_VARIANT.tar.gz.dvc"
   ```

   Then, from the rocKE root, extract and authenticate it where source tests
   look for it:

   ```bash
   python library/tests/reference_common/artifact.py unpack \
     --operation "$REFERENCE_OPERATION" --variant "$REFERENCE_VARIANT" \
     --archive "library/tests/reference_bundles/$REFERENCE_OPERATION/$REFERENCE_ARCH/$REFERENCE_VARIANT.tar.gz" \
     --bundle "library/tests/reference_bundles/$REFERENCE_OPERATION/$REFERENCE_ARCH/$REFERENCE_VARIANT" \
     --lock "library/tests/${REFERENCE_OPERATION}_reference/architectures/$REFERENCE_ARCH/$REFERENCE_VARIANT/baseline_lock.json"
   ```

   Repeat both for each published variant of the architecture. Unpack never
   replaces an extracted bundle. When the DVC pointer changes, delete the
   extracted directory and repeat both commands.

4. **Run the pinned suite on a matching GPU**, with NumPy, pytest,
   pytest-timeout, HIP, and COMGR, and without Torch:

   ```bash
   python -m pytest "library/tests/test_${REFERENCE_OPERATION}_pinned_reference.py" \
     --rocke-reference-operation "$REFERENCE_OPERATION" \
     --rocke-reference-arch "$REFERENCE_ARCH" -v -rs
   ```

   This runs every published variant of the architecture, with test ids
   `<arch>-<variant>-<case>`; add `--rocke-reference-variant "$REFERENCE_VARIANT"`
   to run only one. A failing comparison prints the `verify --case` command
   that repeats only that case. To run selected cases yourself:

   ```bash
   PYTHONPATH=platform/python python "library/tests/run_${REFERENCE_OPERATION}_reference.py" verify \
     --arch "$REFERENCE_ARCH" --variant "$REFERENCE_VARIANT" --current-root . --case <case-id> \
     --bundle "library/tests/reference_bundles/$REFERENCE_OPERATION/$REFERENCE_ARCH/$REFERENCE_VARIANT"
   ```

   `verify` exits with status 1 when a comparison fails and 2 when the lock or
   bundle cannot be used. A subset run is not a full verification.

5. **Run the installed suites** on a matching GPU from the installed provider
   test directory, `bin/hip_kernel_provider/`. The build must fetch the archives
   before configuring; see the
   [Torch-free installed procedure](../../../../TESTING.md#running-installed-reference-tests-without-torch).

   ```bash
   ctest -R "^rocke_(reference_common|${REFERENCE_OPERATION}_reference_unit|${REFERENCE_OPERATION}_gpu_${REFERENCE_ARCH})_pytest$" \
     --output-on-failure
   ```

6. **Read a skip as missing coverage, not a pass.** CTest records a suite for
   another processor as skipped (exit code 77). Source pytest without the two
   `--rocke-reference-*` options above skips when the GPU has no enrolled
   cohort or no extracted bundle. With them, a missing bundle or GPU fails, and
   a GPU of another processor ends the run with exit code 77. An enrolled
   variant without a committed lock skips unless `--rocke-reference-variant`
   selects it, which then requires a candidate bundle and lock. Check that
   every case `cases` lists for each published variant and its three failure
   controls ran; `-rs` prints each skip reason. See
   [processor selection](gpu-reference-workflow.md#processor-selection-in-therock-jobs).

## I want to add a case

1. **Edit the cohort.** Add the case to `CASES` in the variant's adapter,
   `library/tests/<operation>_reference/architectures/<arch>/<variant>/__init__.py`.
   Variants whose adapters take `CASES` from one tuple share a cohort: the
   gfx1151, gfx1200, and gfx1201 SDPA `wmma` adapters take theirs from
   `sdpa_reference/implementations/wmma.py`, so an edit there changes the
   cohort of every WMMA processor and each published WMMA variant needs
   requalification. To change one processor's cases alone,
   [un-share the cohort](#i-want-to-change-a-shared-cohort-on-one-processor)
   first. `cases` and `status` also name variants whose separate tuples hold
   identical cases, such as the convolution adapters; editing one of those
   tuples changes only its own variant. Choose the case and assert its
   intended dispatch properties in the host contract tests as described in
   [step 1](gpu-reference-workflow.md#1-choose-coverage-and-implement-the-contract).

2. **Review the change without a GPU.** `cases` lists the edited cohort. `diff`
   names the enrolled cases that the published bundle does not record and
   exits with status 1 on any difference. Without `--variant`, it compares
   every published variant of the architecture:

   ```bash
   python "library/tests/run_${REFERENCE_OPERATION}_reference.py" cases --arch "$REFERENCE_ARCH"
   PYTHONPATH=platform/python python "library/tests/run_${REFERENCE_OPERATION}_reference.py" diff --arch "$REFERENCE_ARCH"
   ```

   `status` reports each published variant whose committed lock pins a
   different cohort digest as differing, even where no bundle was fetched.

3. **Run the host checks** from the previous recipe. Until step 4 replaces
   each edited variant's bundle and lock, these fail by design, as
   verification on that GPU does:

   - `test_committed_locks_pin_the_enrolled_cohorts` for each edited variant,
     and `test_published_bundles_record_the_enrolled_cohorts` for each one
     whose bundle data is present, with `the enrolled cases differ from`;
   - `test_published_runners_replay_the_current_request` for each edited
     variant whose bundle data is present, with
     `does not cover the complete enrolled cohort`;
   - `test_status_reports_this_tree_without_a_gpu_or_torch` and
     `test_status_says_how_to_provide_missing_bundle_data` for the operation,
     because `status` exits with status 1 while a published variant differs.

   Fix every other failure first. If the locked baseline revision's kernels
   cannot prepare the new case,
   `test_adapters_prepare_against_their_locked_baselines` also fails; qualify
   at a newer baseline revision.

4. **Requalify and publish** the complete cohort of each edited variant through
   [steps 2 to 5](gpu-reference-workflow.md#2-prepare-reproducible-qualification-inputs).
   Check the [prerequisites](gpu-reference-workflow.md#prerequisites) first: a
   GPU of each edited processor, a baseline revision reachable from `develop`
   whose kernels support the new case, and DVC write access or a maintainer
   who has it. Commit the cohort edit with each variant's new
   `baseline_lock.json` and DVC pointer; each archive must be uploaded before
   the pull request merges.

## I want to add a variant

A kernel family that computes an operation's math on the same inputs and
outputs, with the same `Case`, independent reference, input generator, and
manifest operation id, is a new variant of that operation; anything else is a
[new operation](#i-want-to-add-an-operation). See
[a variant or a new operation](gpu-reference-workflow.md#a-variant-or-a-new-operation).
A variant has its own cohort, baseline revision, lock, and archive, so adding
one leaves the architecture's other variants and their bundles untouched. It
must fit the shared harness limits listed under
[adding an operation](#i-want-to-add-an-operation). Make these changes under
`library/tests/`:

| File | Change |
|---|---|
| `<operation>_reference/implementations/<variant>.py` | The family's launch, export, and replay mechanics that its adapters share. The module name is the variant's name: it matches `^[a-z][a-z0-9_]*$` and does not start with `gfx`. New manifests record it and a rename needs requalification, so treat it as permanent once published |
| `<operation>_reference/architectures/<arch>/<variant>/__init__.py` | The adapter for each enrolled processor that runs the family, with its own `CASES`, as for [an architecture](#i-want-to-add-an-architecture). The processor package must not import it |
| `<operation>_reference/worker.py` | Bind the family's frozen replay arguments in `replay_values` if its exported kernels record a replay ABI that the worker does not bind yet |
| `test_<operation>_reference_contract.py` | Checks of the adapter's dispatch properties. On an architecture with several variants, `get_adapter(arch)` without a variant is ambiguous, so name the variant in those tests and in other host tests that call it, such as `test_sdpa_torch_reference.py` |
| `test_reference_common.py` | Only when the architecture had one variant until now. Some shared tests still assume one variant per architecture: they call `resolve_variant(arch)` or `get_adapter(arch)` without a variant, for every architecture or for the operation's first, `ARCHITECTURES[0]`. Make them name a variant or loop over `variants(arch)` |

Then:

1. **Check it without a GPU.** `cases --arch` lists the new variant beside the
   others and `status` reports it unpublished. Run the host checks from the
   [first recipe](#i-changed-a-covered-kernel), and for SDPA also
   `library/tests/test_sdpa_torch_reference.py`, which needs no Torch. On an
   architecture that had one variant, the tests that assume one fail, most
   with `needs a selected reference variant`, until you update them as in the
   table; then the host checks pass, including
   `test_enrolled_variants_follow_the_layout_rules`.
2. **Qualify, verify, and publish it** on each processor that enrolls it
   through [steps 2 to 5](gpu-reference-workflow.md#2-prepare-reproducible-qualification-inputs),
   passing `--variant` and `--rocke-reference-variant`. Commit the reviewed
   lock as `<operation>_reference/architectures/<arch>/<variant>/baseline_lock.json`
   with its pointer, `reference_bundles/<operation>/<arch>/<variant>.tar.gz.dvc`.

An architecture already listed in `PublishedGpuReferences.cmake` installs the
new variant without a CMake edit. Its existing `rocke_<operation>_gpu_<arch>_pytest`
CTest runs it with the other variants, with a timeout of five minutes per
installed variant. Publishing a variant on an unlisted architecture also lists
the architecture, as below. Once an architecture has several variants, the
command line needs `--variant` for it, and pytest's bundle and lock overrides
need `--rocke-reference-variant`.

## I want to add an architecture

Enrolling and publishing are separate steps. `registry.json` enrolls the
processor for qualification, `verify`, `cases`, `diff`, `status`, and the
pinned test parametrization. Publication installs the processor's variants and
registers its CTest. Make these changes in order, under `library/tests/` unless
shown otherwise:

| File | Change |
|---|---|
| `<operation>_reference/architectures/<arch>/__init__.py` | The processor package: a docstring and `NAME`. Replay imports it to read the name, so it imports no variant |
| `<operation>_reference/architectures/<arch>/<variant>/__init__.py` | An adapter for each kernel family that the processor runs, in a directory named for the family's `implementations/<variant>.py`. It takes `NAME` with `from .. import NAME` and exports `CASES`, `CASE_BY_ID`, `prepare`, `launch`, and `exported_kernel`, through the production dispatch and launch path. An adapter that compiles a generic code object also exports `COMPILER_TARGET`; see [generic code objects](gpu-reference-workflow.md#generic-code-objects-and-processor-qualification) |
| `<operation>_reference/architectures/registry.json` | Add the processor |
| `test_<operation>_reference_contract.py` | Add the processor where these host tests list architectures explicitly, with checks of the adapter's dispatch properties. SDPA's `test_architecture_enrollment_and_locks` asserts the whole registry |
| `<operation>_reference/architectures/<arch>/<variant>/baseline_lock.json` | The reviewed `qualification-lock.json` from qualifying the variant on that processor ([steps 2 to 5](gpu-reference-workflow.md#2-prepare-reproducible-qualification-inputs)) |
| `reference_bundles/<operation>/<arch>/<variant>.tar.gz` and its `.dvc` pointer | Pack the archive and track it with `dvc add`. Commit the pointer, not the archive, and upload the archive. `reference_bundles/.gitignore` already ignores the archive; delete any other `.gitignore` that `dvc add` writes, because configuration rejects it |
| `platform/cmake/PublishedGpuReferences.cmake` (from the rocKE root) | Add the processor to `ROCKE_PUBLISHED_<OPERATION>_ARCHITECTURES` |
| The operation guide | Describe the new coverage; the [test inventory](../../../../platform/tests/README.md#installed-pinned-reference-suites) needs no edit for a new architecture |

A listed architecture installs every variant with a committed lock; with
references installed, configuration fails for one without a published variant.
Every configuration also fails for a listed architecture that `registry.json`
does not enroll.
The `rocke_<operation>_gpu_<arch>_pytest` CTest, the installation paths, and
the CTest's selection by the provider's category YAML follow automatically. The
shared host checks cover each variant once its lock exists; a processor that
enrolls several variants also needs the host test updates described for
[a variant](#i-want-to-add-a-variant). Running it in CI also needs a runner
with that GPU.

## I want to change a shared cohort on one processor

Variants that take `CASES` from one tuple share a cohort. The SDPA `wmma`
adapters take theirs from `sdpa_reference/implementations/wmma.py`: editing it
invalidates the lock of every published WMMA variant, and host tests fail until
each is requalified on its own processor. To change the cases of some
processors only, first give each processor that keeps its cases a tuple of its
own:

1. In that processor's adapter, replace `CASES = wmma.CASES` with a tuple of
   the same cases. The cohort digest does not change, so its lock stays valid
   without requalification; `status` keeps naming the cohort as shared until the
   cases differ.
2. Edit the shared tuple, and requalify and publish only the variants that
   still use it, as in [adding a case](#i-want-to-add-a-case).

`wmma.prepare` accepts only cases in `wmma.CASES`, so every copy must stay
within it: add new cases to the shared tuple, not to a copy, and keep each case
that a copy still enrolls. Otherwise, first make `wmma.prepare` accept the
adapter's own cohort.

## I want to add an operation

An operation keeps one numerical contract. A kernel family with other math or
another input and output model, such as convolution backward-data, is a new
operation with its own variants, such as a `conv_bwd_reference` package; see
[a variant or a new operation](gpu-reference-workflow.md#a-variant-or-a-new-operation).
Each `library/tests/<operation>_reference/` package with an
`architectures/registry.json` is an operation. `artifact.py`, the
`--rocke-reference-operation` pytest option, CMake installation, and CTest
registration find it by that convention, and the shared tests in
`test_reference_common.py` check it. `reference_common` provides qualification,
verification, archives, the worker loop, and the GPU tests; the package
declares only its own parts. The `conv_reference` package is a complete
example:

| File in the package | Declares |
|---|---|
| `__init__.py` | The package marker, which qualification freezes with `contract.py` and `worker.py` |
| `operation.json` | The operation id its bundle manifests declare, such as `{"manifest_operation": "conv-fwd"}` |
| `contract.py` | A frozen `Case` dataclass with `id`, `dtype`, `tolerance`, `input_shapes`, and `output_shape`; `SCHEMA_VERSION`; `INPUT_GENERATOR`; deterministic quantized `make_inputs(case)`; and the independent reference, `independent_reference(case, inputs)` |
| `architectures/` | `__init__.py`, which binds `ARCHITECTURES`, `get_architecture`, `variants`, `resolve_variant`, `get_adapter`, and `baseline_lock` from a `reference_common.registry.AdapterRegistry`; `registry.json`; and a processor package with its variant adapters per processor, as for a new architecture |
| `implementations/` | A module per kernel family, named as its variants are, and other code that adapters share. Qualification freezes this directory, so it must exist |
| `cli.py` | `OPERATION`, a `reference_common.cli.ReferenceOperation`, with module aliases for its `qualify`, `load_bundle`, `manifest_key`, `verify_case`, and `main` |
| `worker.py` | `run(request, output)`, which passes a `WorkerOperation` and the target to `reference_common.worker.run_case`: the requested variant's adapter in source mode, and only the processor package in replay mode |

Beside the package, add `run_<operation>_reference.py`, the host
`test_<operation>_reference_contract.py`, and `test_<operation>_pinned_reference.py`,
whose negative-control perturbation must exceed the comparison limit in the
operation's metric. Add `set(ROCKE_PUBLISHED_<OPERATION>_ARCHITECTURES "")` to
[`PublishedGpuReferences.cmake`](../../../../platform/cmake/PublishedGpuReferences.cmake);
configuration fails without a list. Then add each architecture as above. Step 1
of the workflow describes these
[parts in more detail](gpu-reference-workflow.md#1-choose-coverage-and-implement-the-contract).

The shared harness supports only kernels that:

- launch exactly once per repetition through rocKE's `KernelLauncher`, which
  counts the launches;
- write one `fp16` or `bf16` output and leave their inputs unchanged;
- produce bit-identical outputs on repeated baseline runs, which qualification
  and verification require.

Multiple launches, multiple outputs, and nondeterministic kernels need a
numerical and replay contract the harness does not provide yet.
