<!--
Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
SPDX-License-Identifier: MIT
-->

# Pinned GPU reference tests

The reference tests compare the current kernel with a frozen GPU baseline that
was independently qualified against numerical oracles. Kernel authors start
with the [quickstart](quickstart.md). A failing pinned test points to
[triage](gpu-reference-workflow.md#triage), and the [glossary](#glossary)
defines the terms these guides use. Detailed documentation lives beside the
reference harnesses:

- [Quickstart: run, extend, and add references](quickstart.md)
- [Qualification and publication workflow](gpu-reference-workflow.md)
- [Triage a failing pinned test](gpu-reference-workflow.md#triage)
- [SDPA contract, coverage, and replay](../../sdpa_reference/docs/sdpa-test-reference.md)
- [Convolution contract, coverage, and replay](../../conv_reference/docs/conv-test-reference.md)
- [Methodology and extension strategy](gpu-ci-pinned-rocke-test-reference-plan.md)
- [rocKE testing strategy and installed test commands](../../../../TESTING.md)

## Kernel arguments and compatibility

The current side already runs the current kernel through the operation's test
adapter and launch path. The baseline side runs the archived code object with
its own frozen signature, launch metadata, argument bindings, and replay
support. The two kernels can have different argument lists while implementing
the same enrolled operation and numerical contract.

Adding a required argument changes a kernel's launch ABI. Its callers must
supply that argument with the correct type and meaning; an older caller is not
automatically compatible. The reference harness does not promise general binary
compatibility between kernel revisions. It preserves each frozen code object's
own ABI and supplies the current kernel's arguments through its current launch
path.

For example, if a new dense kernel takes an extra indexing scalar, its current
launcher must supply that scalar. An existing reference bundle continues to
launch its old code object with the old signature. If the new revision is later
qualified as a baseline, its exporter must capture the new scalar binding.
New dense qualifications use `dense-v2` to freeze runtime scalars and reject
unbound signature arguments before issuing a certificate. Replay supplies live
buffer pointers and the enrolled case's scale, then restores the frozen scalars.
See [dense replay compatibility](../../sdpa_reference/docs/sdpa-test-reference.md#dense-replay-compatibility)
for the handling of older dense bundles.

A signature change is not automatically supported by qualification either. The
exporter must account for every argument. New pointers, extra outputs, or changed
argument semantics can require explicit adapter, replay, and numerical-contract
changes. Update the relevant contract tests and verify fresh qualification and
replay before publishing a bundle using that ABI.

## When to replace the baseline

A production kernel change does not by itself require a new baseline. Keep the
existing qualified baseline while the enrolled cases and numerical contract
remain valid, and run the current implementation against it. Replacing the
baseline with every kernel edit would remove that stable regression reference.

Baseline replacement is a separate change: select a committed kernel revision,
independently qualify the complete cohort, verify the candidate's frozen replay
and current-kernel comparison, then review and publish the replacement archive,
trusted lock, and DVC pointer. Agreement with the previous baseline alone is
insufficient qualification. Adding cases or changing the qualified contract also
requires a replacement covering the complete resulting cohort. Follow the
[workflow](gpu-reference-workflow.md) for those steps.

## Published replay versus historical source qualification

These are separate compatibility concerns:

- **Replaying a published bundle** uses its authenticated frozen payload and
  preserves its recorded signature and numerical limits. Current kernel changes
  should continue to work with that baseline while its contract remains valid.
- **Qualifying an older source revision again** runs the invoking checkout's
  qualification adapter against that revision's production sources. This may
  need an adapter for a historical dispatch API. For example, gfx942 supports
  the published dense baseline's older `dense_spec_for_request` interface.

Historical source adapters are a maintenance convenience, not a promise that
every old revision can be requalified forever. A future policy could require
recent source revisions for baseline refreshes and retire those adapters while
preserving published-bundle replay. The current gfx942 historical adapter remains
supported; removing it would be a separate policy and code change.

## Glossary

| Term | Meaning |
|---|---|
| Operation | A `library/tests/<operation>_reference/` package, such as `conv` or `sdpa`, with its numerical contract, adapters, and `run_<operation>_reference.py` command line |
| Adapter | `<operation>_reference/architectures/<arch>/__init__.py`. It declares the architecture's cases and prepares and launches the current kernel through the production dispatch and launch path |
| Cohort, enrolled case | An architecture is enrolled when its operation's `architectures/registry.json` lists it. Its cohort is the adapter's `CASES`, and each is an enrolled case. Verification requires a bundle to record exactly the cohort, matching each case by its complete case contract |
| Independent reference | The operation's offline oracle, `independent_reference` in `contract.py`; both operations use float64 NumPy. Only qualification computes it, optionally cross-checked with Torch |
| Baseline revision | The committed revision whose production kernels a bundle freezes. Qualification snapshots its sources, and the lock records it as `baseline_revision` |
| Qualification | The offline GPU run, `qualify`, that executes the baseline revision's kernels for every enrolled case, requires bit-identical repeated outputs, bounds their error against the independent reference, and writes a candidate bundle with its `qualification-lock.json`. It is valid only on the processor it ran on |
| Manifest | A bundle's `manifest.json`. It records the baseline revision, payload file digests, and for each case the case contract, input and output digests, budget and comparison limit, kernel and launch metadata, and device target. Other guides call a processor's qualified manifest its certificate |
| Baseline lock | `<operation>_reference/architectures/<arch>/baseline_lock.json`, the reviewed copy of a candidate's `qualification-lock.json`. It pins the schema, baseline revision, and manifest digest; a bundle whose manifest differs is rejected |
| Bundle, archive, DVC pointer | A bundle is the extracted directory: the manifest and a payload of frozen code objects, runtime, and worker, without tensors. The archive `reference_bundles/<operation>/<arch>.tar.gz` packs it. Git tracks only its DVC pointer, `<arch>.tar.gz.dvc`; `dvc pull` fetches the archive and `artifact.py unpack` extracts and authenticates it |
| Published pair | An operation and architecture with a committed lock and pointer, listed in [`PublishedGpuReferences.cmake`](../../../../platform/cmake/PublishedGpuReferences.cmake). Only published pairs are installed and get a `rocke_<operation>_gpu_<arch>_pytest` CTest |
| Replay | Running the bundle's frozen code object with its frozen runtime. Its output must match the qualified digest before the current kernel's output is compared with it |
| Frozen and current ABI | The frozen ABI is the signature, launch geometry, and argument bindings that qualification recorded for the baseline code object; replay uses it unchanged. The current ABI is whatever the current kernel's launch path takes, which the adapter supplies. They may differ; see [kernel arguments](#kernel-arguments-and-compatibility) |
| Tolerance, comparison limit | The tolerance `T` is a case's original error limit against the independent reference. Qualification bounds the baseline's error by `B`, and the comparison limit `C`, rounded strictly below `T-B`, limits the current output's distance from the replayed output. Within `C`, the current output is within `T` of the independent reference; see the [budget derivation](gpu-ci-pinned-rocke-test-reference-plan.md#1-preserve-the-original-accuracy-contract) |
| Failure controls | The three GPU tests that show the comparison can fail: a perturbed current output must exceed the comparison limit, a perturbed replayed output must fail its qualified digest, and a suppressed launch must leave poisoned output that the worker rejects |
