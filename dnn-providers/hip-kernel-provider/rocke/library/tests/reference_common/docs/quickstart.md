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
an operation and the processor of the GPU under test first:

```bash
REFERENCE_OPERATION=conv   # or sdpa
REFERENCE_ARCH=gfx942      # the GPU's processor, such as gfx950
```

An operation's `architectures/registry.json` lists its enrolled processors, and
[`PublishedGpuReferences.cmake`](../../../../platform/cmake/PublishedGpuReferences.cmake)
lists the published pairs. The `run_<operation>_reference.py` command line's
`cases` command needs no GPU or Torch. Its `diff` and `verify` commands also
read device target ids with the rocKE platform package, so the commands below
set `PYTHONPATH=platform/python`.

## I changed a covered kernel

1. **Know what CI compares.** On the GPU of each published pair, the pinned
   suite replays the frozen baseline for every enrolled case and requires its
   qualified output digest. It then runs the current kernel through the
   operation's adapter and requires every output to stay within the case's
   comparison limit of the replayed output. Three failure controls check that
   the comparison can fail. A kernel is covered when an enrolled case reaches
   it through the adapter; list the cases with:

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
   update for a changed production API drops the API that the pair's locked
   baseline revision uses.

3. **Fetch and extract the published bundle.** From the rocm-libraries root:

   ```bash
   dvc pull "dnn-providers/hip-kernel-provider/rocke/library/tests/reference_bundles/$REFERENCE_OPERATION/$REFERENCE_ARCH.tar.gz.dvc"
   ```

   Then, from the rocKE root, extract and authenticate it where source tests
   look for it:

   ```bash
   python library/tests/reference_common/artifact.py unpack --operation "$REFERENCE_OPERATION" \
     --archive "library/tests/reference_bundles/$REFERENCE_OPERATION/$REFERENCE_ARCH.tar.gz" \
     --bundle "library/tests/reference_bundles/$REFERENCE_OPERATION/$REFERENCE_ARCH" \
     --lock "library/tests/${REFERENCE_OPERATION}_reference/architectures/$REFERENCE_ARCH/baseline_lock.json"
   ```

   Unpack never replaces an extracted bundle. When the DVC pointer changes,
   delete the extracted directory and repeat both commands.

4. **Run the pinned suite on a matching GPU**, with NumPy, pytest,
   pytest-timeout, HIP, and COMGR, and without Torch:

   ```bash
   python -m pytest "library/tests/test_${REFERENCE_OPERATION}_pinned_reference.py" \
     --rocke-reference-operation "$REFERENCE_OPERATION" \
     --rocke-reference-arch "$REFERENCE_ARCH" -v -rs
   ```

   A failing comparison prints the `verify --case` command that repeats only
   that case. To run selected cases yourself:

   ```bash
   PYTHONPATH=platform/python python "library/tests/run_${REFERENCE_OPERATION}_reference.py" verify \
     --arch "$REFERENCE_ARCH" --current-root . --case <case-id> \
     --bundle "library/tests/reference_bundles/$REFERENCE_OPERATION/$REFERENCE_ARCH"
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
   a GPU of another processor ends the run with exit code 77. Check that every
   case `cases` lists and the three failure controls ran; `-rs` prints each skip
   reason. See [processor selection](gpu-reference-workflow.md#processor-selection-in-therock-jobs).

## I want to add a case

1. **Edit the cohort.** Add the case to `CASES` in the adapter,
   `library/tests/<operation>_reference/architectures/<arch>/__init__.py`. The
   gfx1151, gfx1200, and gfx1201 SDPA adapters share `CASES` from
   `sdpa_reference/implementations/wmma.py`: an edit there changes the cohort
   of every WMMA processor, so each published WMMA pair needs requalification.
   Choose the case and assert its intended dispatch properties in the host
   contract tests as described in
   [step 1](gpu-reference-workflow.md#1-choose-coverage-and-implement-the-contract).

2. **Review the change without a GPU.** `cases` lists the edited cohort. `diff`
   names the enrolled cases that the published bundle does not record and
   exits with status 1 on any difference:

   ```bash
   python "library/tests/run_${REFERENCE_OPERATION}_reference.py" cases --arch "$REFERENCE_ARCH"
   PYTHONPATH=platform/python python "library/tests/run_${REFERENCE_OPERATION}_reference.py" diff --arch "$REFERENCE_ARCH"
   ```

3. **Run the host checks** from the previous recipe.
   `test_published_bundles_record_the_enrolled_cohorts` fails for each edited
   pair until step 4 replaces its bundle; verification on that GPU rejects the
   edited cohort too. Fix every other failure first. If the locked baseline revision's kernels
   cannot prepare the new case, `test_adapters_prepare_against_their_locked_baselines`
   also fails; qualify at a newer baseline revision.

4. **Requalify and publish** the complete cohort of each edited pair through
   [steps 2 to 5](gpu-reference-workflow.md#2-prepare-reproducible-qualification-inputs).
   Check the [prerequisites](gpu-reference-workflow.md#prerequisites) first: a
   GPU of each edited processor, a baseline revision reachable from `develop`
   whose kernels support the new case, and DVC write access or a maintainer
   who has it. Commit the cohort edit with each pair's new `baseline_lock.json`
   and DVC pointer; each archive must be uploaded before the pull request
   merges.

## I want to add an architecture

Enrolling and publishing are separate steps. `registry.json` enrolls the
processor for qualification, `verify`, `cases`, `diff`, and the pinned test
parametrization. Publication installs the pair and registers its CTest. Make
these changes in order, under `library/tests/` unless shown otherwise:

| File | Change |
|---|---|
| `<operation>_reference/architectures/<arch>/__init__.py` | A new adapter exporting `NAME`, `CASES`, `CASE_BY_ID`, `prepare`, `launch`, and `exported_kernel`, through the production dispatch and launch path. An adapter that compiles a generic code object also exports `COMPILER_TARGET`; see [generic code objects](gpu-reference-workflow.md#generic-code-objects-and-processor-qualification) |
| `<operation>_reference/architectures/registry.json` | Add the processor |
| `test_<operation>_reference_contract.py` | Add the processor where these host tests list architectures explicitly, with checks of the adapter's dispatch properties. SDPA's `test_architecture_enrollment_and_locks` asserts the whole registry |
| `<operation>_reference/architectures/<arch>/baseline_lock.json` | The reviewed `qualification-lock.json` from qualifying on that processor ([steps 2 to 5](gpu-reference-workflow.md#2-prepare-reproducible-qualification-inputs)) |
| `reference_bundles/<operation>/<arch>.tar.gz` and its `.dvc` pointer | Pack the archive and track it with `dvc add`. Commit the pointer and any `.gitignore` change that `dvc add` makes, not the archive, and upload the archive |
| `platform/cmake/PublishedGpuReferences.cmake` (from the rocKE root) | Add the processor to `ROCKE_PUBLISHED_<OPERATION>_ARCHITECTURES` |
| The operation guide | Describe the new coverage; the [test inventory](../../../../platform/tests/README.md#installed-pinned-reference-suites) needs no edit for a new architecture |

The `rocke_<operation>_gpu_<arch>_pytest` CTest, its installation path, and its
selection by the provider's category YAML follow automatically from the
publication list. The shared host checks cover the pair once its lock exists.
Running it in CI also needs a runner with that GPU.

## I want to add an operation

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
| `architectures/` | `__init__.py` with `ARCHITECTURES`, `get_architecture`, and `baseline_lock`; `registry.json`; and an adapter per processor, as for a new architecture |
| `implementations/` | Code that adapters share. Qualification freezes this directory, so it must exist |
| `cli.py` | `OPERATION`, a `reference_common.cli.ReferenceOperation`, with module aliases for its `qualify`, `load_bundle`, `manifest_key`, `verify_case`, and `main` |
| `worker.py` | `run(request, output)`, which passes a `WorkerOperation` and the requested adapter to `reference_common.worker.run_case` |

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
