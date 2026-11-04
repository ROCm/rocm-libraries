<!--
Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
SPDX-License-Identifier: MIT
-->

# Convolution coverage and remaining work

The delivered scope is forward, backward-data, and backward-weight on gfx942
and gfx950, with FP16/BF16 implicit-GEMM configurations. Coverage is bounded by
the cases and execution protocols below. A registered architecture, a constructible
kernel, and successful GPU numerical execution are different evidence levels.

| Direction | gfx942 | gfx950 |
|---|---|---|
| Forward | Pinned cases; explicit 64x64x64 configuration | Pinned cases; production implicit-GEMM selection |
| Backward-data | Explicit-spec cases | Production implicit-GEMM cases, except explicit K=33 cases |
| Backward-weight | Explicit cases; atomic and two-stage split-K | Explicit cases; atomic, two-stage, and merged groups |

From the rocKE root, `python library/tests/run_conv_reference.py cases --arch <arch>`
lists the pinned forward cases. The gradient cases are in
[`gradient_cases.py`](../library/tests/conv_reference/gradient_cases.py).

The [reference guide](../library/tests/conv_reference/docs/conv-test-reference.md) owns the numerical contracts.
The [validation record](conv-six-lane-validation.md) distinguishes historical
qualification from checks of the current source. Use the
[shared workflow](../library/tests/reference_common/docs/gpu-reference-workflow.md) for reference publication.

## Production selection and explicit coverage

The grouped-convolution dispatcher registers only gfx950 forward and dgrad
instance lists. A gfx942 request or wgrad request cannot establish production
selection coverage. Their tests build explicit configurations without modifying
the production registry. No test silently falls back from a refused production
request. The explicitly enrolled gfx950 K=33 gradient cases preserve coverage
where the production list has no admitted instance.

The gfx950 forward request fixes the algorithm to implicit-GEMM. Default
selection may choose a direct depthwise kernel; such a result is a different
implementation scope. The selected instance supplies its AOT kernel identity,
LLVM flavor, signature, grid, and scalar launch values. Tests must not assume
that a selected object is an old tuning spec or reconstruct a different kernel
from its shape alone.

Weight-gradient configurations retain the earlier cohort's split degrees and
tile choices explicitly. They exercise zero-initialized atomic accumulation,
f32 workspaces with reduction, and merged depthwise groups on gfx950. The
NumPy oracle is recomputed at test time and every repetition is checked.
Nondeterministic atomic outputs do not inherit forward's exact-output digest
certificate.

## Further coverage

| Implementation or behavior | Additional work |
|---|---|
| Direct forward: grouped, nongrouped, and depthwise schedules | Explicit variant admission and ABI coverage; preserve one-binary/multiple-runtime-shape tests |
| Direct dgrad, including MFMA weight transforms | Ordered multi-kernel plans, auxiliary workspaces, and independent dX checks |
| Stream-K wgrad | Linear/tree/atomic/workspace protocols, flags and partial initialization, synchronization assertions |
| Restored production gfx942 or wgrad instance lists | Separate production enrollment and selection validation before claiming dispatch coverage |
| Additional pipelines, vector widths, and LDS layouts | Named configurations and assertions that the intended paths actually executed |
| 3-D, FP32, fused convolution/pooling | Extend case, dtype, output, and independent-oracle contracts explicitly |
| Native provider and kernel packs | Validate shipped dispatch, ABI, workspace planning, and device outputs |
| Pinned gradient replay | A versioned multi-kernel replay/state contract and a justified numerical strategy for nondeterminism |

Do not count a depthwise geometry as coverage of every depthwise implementation,
or replace indexing, guard, synchronization, or binary-reuse checks with a small
numerical distance. Preserve existing cases when extending cohorts unless a
removal is intentional and explained. A new forward cohort must be independently
qualified as a whole before replacing its archive, trusted lock, and pointer.
