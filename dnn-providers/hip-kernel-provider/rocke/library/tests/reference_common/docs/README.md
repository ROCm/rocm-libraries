<!--
Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
SPDX-License-Identifier: MIT
-->

# Pinned GPU reference tests

The reference tests compare the current kernel with a frozen GPU baseline that
was independently qualified against numerical oracles. Detailed documentation
lives beside the reference harnesses:

- [Qualification and publication workflow](gpu-reference-workflow.md)
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
