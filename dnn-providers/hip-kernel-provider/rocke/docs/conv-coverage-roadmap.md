<!--
Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
SPDX-License-Identifier: MIT
-->

# Convolution implementation inventory and coverage roadmap

Assessment date: 2026-10-14. Source base: `c767fbe1af3`, including the local
convolution reference adapter refactor. The assessment below records the source
inventory and CPU selection/kernel-construction probes used to plan the work.
The older `conv-reference-coverage-analysis.md` describes an earlier checkout;
its skip counts and implementation assumptions are not a current denominator.

Delivery update: the first implicit-GEMM cohorts now pass source and installed
verification on both targets: 32 forward, 12 backward-data and 14 backward-weight
cases per architecture. Both forward bundles are published, independently
retrieved from DVC, and enrolled in provider CI categories. Gradients use live
NumPy oracles, with gfx942 backward-data explicitly testing specs outside
production selection. See the [validation record](conv-six-lane-validation.md)
for evidence and limits; the remaining implementation extensions below are
future coverage, not prerequisites for these bounded cohorts.

The central finding is that convolution coverage must distinguish direction,
implementation, architecture, and execution protocol. More shapes in the current
forward cohort will cover useful geometry, but cannot cover gradients, direct
kernels, workspace protocols, or runtime binary reuse.

## Confirmed delivery scope

The convolution team lead specified **forward, backward-data, and backward-weight
on gfx942 and gfx950**. This is the required six-lane matrix, superseding the
exploratory prioritization below. Forward expansion is an incremental milestone,
not completion of the requested convolution coverage.

| Direction | gfx942 | gfx950 |
|---|---|---|
| Forward | 32-case published cohort; source and installed verification passed | 32-case published cohort; source and installed verification passed |
| Backward-data | 12 explicit-spec cases passed; no production-dispatch candidate | 12 production-selected cases passed |
| Backward-weight | 14 cases passed, including atomic split-K and two-stage paths | 14 cases passed, including atomic split-K, two-stage and merged-group paths |

The direction/architecture scope does not specify every algorithm, dtype or
geometry. Start with the existing FP16/BF16 implicit-GEMM surface and make each
lane's selected cases explicit. Direct kernels and stream-K remain implementation
extensions to prioritize within that matrix; coverage of every variant is not
implied by six enrolled lanes. gfx1151/gfx1201, 3-D and fused operations are outside
the immediate scope.

Do not add a production gfx942 backward-data candidate merely to enable a test.
An explicit-spec reference lane can exercise its existing builder; production
selection remains a separate gap to resolve with the convolution team.

## What exists

| Path | Existing implementation | Coverage implication |
|---|---|---|
| Implicit-GEMM forward | Generic builder, grouped and depthwise geometry, multiple pipelines/epilogues, vector-width choices, merged depthwise groups, 2-D and 3-D geometry support | Current pinned harness covers selected 2-D FP16/BF16 production configurations on gfx942 and gfx950 |
| Direct grouped forward | Generic runtime-channel kernel and fixed 4/8/16/32-channel specializations | Separate preparation/spec selection; forward oracle is reusable; variant admission and dtype restrictions differ |
| Direct depthwise forward | Row-streamed, spatial, tiled, and column-streamed variants | Different schedules and admitted filter/stride/padding ranges; one depthwise case does not cover all four |
| Direct non-grouped forward | Halo-reusing tiled kernel, runtime shapes/channels, atom and tile choices | Cover channel/spatial tails, dilation, and one binary serving multiple shapes |
| Implicit-GEMM backward-data | Tilde decomposition, sub-GEMM records, uniform-K decoding, pointwise fast path, split-K options | New input-gradient oracle and auxiliary metadata bindings; production gfx950 dispatch uses split-K=1 |
| Direct backward-data | Scalar grouped, depthwise, streaming depthwise, and MFMA path with weight transformations | Some paths are single kernels; MFMA dgrad can involve three kernels and transformed-weight workspaces |
| Implicit-GEMM backward-weight | Single-store split-K=1, atomic split-K, two-stage workspace/fold, merged depthwise groups, K-outer LDS and address prefetch options | New weight-gradient oracle; production selection can require atomic and multi-stage execution |
| Stream-K backward-weight | Data-parallel/stream-K and persistent scheduling; linear, tree, atomic, and workspace reductions | Explicit partial-buffer/flag initialization, synchronization checks, reduction-specific determinism and execution plans |
| Direct backward-weight | Grouped MFMA kernel, runtime channels, atomic accumulation | Gradient oracle plus zero-initialized output/accumulation policy; current numerical suite requires gfx950 |
| Deep fused convolution/pool | Shared conv0 -> conv1 -> maxpool prototype and target wrappers | Separate composed-operation oracle, operands, and output contract; lower priority than production convolution |

Primary sources:

- [Implicit-GEMM forward](../library/kernels/common/conv_implicit_gemm.py),
  [backward-data](../library/kernels/common/conv_implicit_gemm_dgrad.py), and
  [backward-weight](../library/kernels/common/conv_implicit_gemm_wgrad.py).
- [Direct grouped/depthwise/gradient builders](../library/kernels/common/conv_direct_grouped.py)
  and [direct non-grouped builder](../library/kernels/common/conv_direct_nongrouped.py).
- [Two-stage wgrad launcher](../library/kernels/common/conv_implicit_gemm_wgrad_two_stage.py)
  and [workspace reducer](../library/kernels/common/conv_wgrad_workspace_reduce.py).
- [Fused prototype](../library/kernels/common/deep_fused_conv_pool.py).

## Architecture and selection are separate from implementation

For the immediate targets:

| Target | Grouped-convolution production dispatcher | Additional source paths | Recommended first coverage |
|---|---|---|---|
| gfx942 | Forward and backward-weight | Direct kernels; explicit gradient and stream-K configurations | Expanded forward production cohort, then direct forward and gradient contracts |
| gfx950 | Forward, backward-data, backward-weight | Direct kernels, merged groups, K-outer paths, stream-K | Forward cohort, then production backward-data and production wgrad execution protocols |
| gfx1151 | No candidate in this dispatcher | Architecture-parametric WMMA builders; fused wrapper | Explicit-spec qualification experiment before adding a reference lane |
| gfx1201 | No candidate in this dispatcher | Architecture-parametric WMMA builders; fused wrapper | Same, independently qualified on matching hardware |

This table is scoped to the immediate targets, not a complete architecture list.
A registry entry, builder admission, kernel construction, compilation, GPU
correctness, and shipped-provider execution are different evidence levels.

CPU probes built both forward and backward-data kernels for gfx1151 and gfx1201
in FP16 and BF16 using a 32x32x32 tile, 2x2 warps, 16x16x16 warp tile, wave32,
`mem`, default epilogue and scalar output stores. The geometry was N=1,
Hi=Wi=8, C=K=32, Y=X=3, padding=1. This establishes construction only, not
compilation or numeric support. Default vector stores were rejected for the
forward default epilogue; explicit scalar stores were required for this probe.
Thus the absence of a production candidate does not imply a missing builder.

The [production dispatcher](../library/dispatch/grouped_convolution.py) selects
implicit-GEMM paths. The [direct cache/planner](../library/benchmarks/common/direct_kernel_sweep.py)
is a distinct selection surface with capability records and runtime shape
matching. Coverage of either Python surface is not proof of native provider
selection or packaged kernel availability.

## What the existing tests establish

At assessment time, the published reference registry enrolled convolution on
gfx942 only: eight geometries in two dtypes, plus three failure controls. The
completed first implementation doubles the cohort and publishes it on both
gfx942 and gfx950, with separate gradient numerical tests as described above.
See [the reference contract](conv-test-reference.md) and
[publication registry](../platform/cmake/PublishedGpuReferences.cmake).

Broader numerical suites already exist for
[forward](../library/tests/test_conv_fwd_correctness.py),
[backward-data](../library/tests/test_conv_dgrad_correctness.py),
[backward-weight](../library/tests/test_conv_wgrad_correctness.py),
[direct convolution](../library/tests/test_direct_conv_correctness.py), and
[stream-K](../library/tests/test_conv_wgrad_streamk_gpu.py).
Their numerical classes require Torch. Many cases construct explicit specs or
invoke benchmark sweeps, so they test a different surface from production
selection. Subtests can contain many configurations; method counts are not
kernel counts. No current GPU skip totals are claimed here.

Host tests cover ABI packing, vector widths, admission/build agreement, group
merging, stream-K plans, direct-cache capabilities, and portions of Python/C++
parity. They are valuable but cannot replace numerical execution.

Existing structural checks should survive a Torch-removal or pinning effort:

- Direct column-depthwise and non-grouped tests reuse one binary across shapes;
  MFMA direct dgrad also tests runtime channels.
- Stream-K tests poison partials, zero flags, require repeated linear/tree output
  identity, and separately check acquire/release IR and ISA ordering.
- Admission tests compare validators and builders, including negative cases.

Pinning independent cases would not automatically retain these properties.

Two previously noted test concerns still appear in the source: the wgrad policy
check imports `library.dispatch.grouped_convolution` and converts ImportError
into a skip, and the odd-slice async-DMA test requests K-outer while gating only
on MFMA (including gfx942). Some other K-outer loops still need an execution-count
audit. Treat these as source findings to resolve with targeted tests, not newly
reproduced GPU failures.

## Immediate forward expansion

CPU production-selection probes accepted all eight previously proposed shapes
in both FP16 and BF16 on both gfx942 and gfx950:

| Additional geometry | Intended distinction |
|---|---|
| C=33, K=32 | Odd input channels and scalar operand loads |
| C=3, Hi=Wi=16 | Stem geometry, short reduction, multiple spatial tiles |
| N=2, Hi=Wi=14, C=64, K=128 | Multiple spatial and channel tiles; exact reduction tiles |
| Y=X=5, padding=2, stride=2 | Longer reduction with a tail |
| C=K=66, groups=2 | Odd per-group channels and scalar stores under cshuffle |
| C=K=64, groups=64 | Depthwise geometry and many independent groups |
| Y=1, X=7, pH=0, pW=3, sW=2 | Unequal spatial-axis behavior |
| Hi=Wi=3, padding=0 | One output pixel |

Unspecified fields use the existing reference Case defaults. The selected tile
was 64x64x64 and the epilogue cshuffle for all these probes. Subsequent GPU
qualification established numerical validity and the available error margin
for both complete cohorts.

The delivered forward cohorts contain 32 numerical cases per architecture: the
original sixteen plus these sixteen. Together they provide 64 architecture-specific
numerical cases. This count does not imply coverage of every implementation or
configuration.

All these forward selections used `group_merge=1`. The recently added merged
forward builder path therefore needs explicit-spec coverage or a production
selection change; a depthwise geometry alone will not exercise it. Pipeline,
async-DMA, unrolling and epilogue alternatives similarly need named configurations
with assertions that the intended path was actually selected.

## Contracts we will need

### Forward alternatives

Retain the forward mathematical contract, but identify the implementation and
configuration separately from the geometry. Current IDs contain only direction,
shape name and dtype. A future cohort must distinguish two implementations of
the same shape and authenticate the selected variant/knobs; do not rely on a
module filename to define baseline identity.

Direct forward can reuse the NumPy forward oracle. Add implementation-specific
spec construction, admission, ABI export and production-cache selection where
appropriate. Start with generic grouped, non-grouped and representative depthwise
schedules; add fixed-channel variants where they cover a distinct code path.

### Gradient oracles and auxiliary inputs

Backward-data takes dY and W and produces dX; backward-weight takes X and dY and
produces dW. Give each a separately validated NumPy oracle and shape/layout
contract. Validate small analytic examples and optionally cross-check Torch
offline. Include stride/dilation interactions, grouping, tails, and output dtype.
The current worker assumes one FP16/BF16 output and two forward operands; FP32
and mixed output types need explicit storage/byte-size/poisoning rules.

Implicit-GEMM dgrad needs authenticated sub-GEMM records and their pointer/count
bindings. These must be replayable without consulting current decomposition
code. Direct MFMA dgrad additionally needs a sequence of weight transformation
and compute kernels.

### Execution plans and determinism

The current artifact and worker describe one code object, one launch ABI, and a
NaN-poisoned output. Extend that through a versioned execution-plan contract when
needed: ordered kernels, immutable auxiliary data, scratch sizes, buffer aliases,
initialization/reset rules, stream ordering and expected launch counts.

This is necessary for real production wgrad. CPU probes of dense, odd-channel
and depthwise shapes resolved split-K above one on gfx942 and gfx950. Odd and
depthwise shapes selected the two-stage path; gfx950 depthwise selected merged
groups. Merely forcing split-K=1 would test an explicit kernel configuration,
not these production choices. Probed production specs kept stream-K off.

Deterministic multi-kernel execution can retain pinned replay once verified.
Atomic accumulation generally cannot satisfy the current exact baseline-output
digest rule. Do not weaken that rule globally or call a few stable runs proof of
determinism. Use a separate numerical strategy: small direct NumPy comparisons
in CI, or an explicitly reviewed stored-oracle/error-bound contract. The latter
would change the present no-output-tensors design. Repeated launches are useful
stress checks, but do not by themselves certify a bound over all atomic orders.

Deterministic stream-K linear/tree paths may be pinnable after their workspace
protocol is represented. Preserve synchronization checks: output agreement alone
cannot prove correct acquire/release behavior.

### Geometry and product execution

3-D support needs a depth-aware case schema and oracle; FP32 needs dtype-specific
budgets; fused convolution needs a composed-operation contract. Keep those as
separate milestones rather than silently broadening the 2-D forward meaning.
Native provider/kernel-pack tests should verify actual selection, ABI compatibility,
workspace planning and output on matching devices. Python reference success does
not establish that integration coverage.

## Recommended sequence

1. Finish the shared forward adapter refactor and preserve old-bundle compatibility.
   Define case identity and direction-specific contracts for the six required lanes.
2. Qualify expanded gfx942 forward and initial gfx950 forward cohorts. Treat these
   as the first deliverable while developing both gradient contracts.
3. Add backward-data oracles and immutable sub-GEMM replay records for both targets.
   Use gfx950 production selection and explicitly identified gfx942 specs. Cover
   pointwise, stride/decomposition, dilation, grouping and tails within admission.
4. Add backward-weight oracles, workspace execution plans and a separate atomic
   accuracy strategy for both targets. A deterministic split-K=1 cohort can validate
   the machinery first, but does not complete coverage of selected atomic/two-stage
   paths. Include gfx950 merged-group selection.
5. Qualify and verify all six lanes on matching hardware, including source execution,
   frozen replay where applicable, installed tests and CI artifact delivery. Record
   actual executed cases and remaining selection gaps for each lane. Follow the
   publication workflow for bundle publication and enrollment.
6. Extend the six-lane matrix with direct implementations, stream-K and additional
   pipeline variants according to the team's priorities. Preserve binary-reuse,
   initialization and synchronization assertions when adapting existing tests.
   Other architectures, 3-D and fused operations remain subsequent scope.

## Implication for `implementations/`

The folder is justified by existing code, not just future architectures. A
reasonable eventual decomposition is `implicit_gemm_fwd`, `direct_fwd`,
`implicit_gemm_dgrad`, `direct_dgrad`, `implicit_gemm_wgrad`, and `direct_wgrad`.
Depthwise schedules can initially remain variants within `direct_fwd`; split-K
and stream-K modes belong with wgrad unless their execution mechanics warrant
another module. Do not create empty modules ahead of use.

Architecture adapters should own target cohorts and selection; implementation
modules should turn selected specs into executable/exportable plans; contracts
should own mathematical operands, oracles and error policy. Shared execution
infrastructure should own buffers, initialization and replay. This keeps both
architecture expansion and genuinely different convolution algorithms possible
without making the current forward helper responsible for every future protocol.
