<!--
Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
SPDX-License-Identifier: MIT
-->

# Initial six-lane convolution implementation and validation

Date: 2026-10-14. Branch: `users/brockhargreaves-amd/multiarch-conv-references`,
based on `c767fbe1af37126cdc75b10f8e23bcee629debb5`, with reference/test changes
that were uncommitted when submitted. Production kernel sources are unchanged.
The required scope is forward, backward-data and backward-weight on gfx942 and
gfx950.

| Lane | Implemented numerical cases | GPU result |
|---|---:|---|
| Forward / gfx942 | 32 | Independent qualification, replay/source verification and 35 GPU pytest checks passed (32 numerical + 3 controls) |
| Forward / gfx950 | 32 | Independent qualification, replay/source verification and 35 GPU pytest checks passed (32 numerical + 3 controls) |
| Backward-data / gfx942 | 12 | Passed; explicit-spec source lane |
| Backward-data / gfx950 | 12 | Passed; production-selected source lane |
| Backward-weight / gfx942 | 14 | Passed; production-selected atomic/two-stage source lane |
| Backward-weight / gfx950 | 14 | Passed; production-selected atomic/two-stage source lane, including merged groups |

Successful forward Slurm jobs were 68089661 (gfx942) and 68089647 (gfx950).
Each used one matching GPU, Python 3.12, NumPy 2.5.4 and a Torch-free environment.
Qualification ran the baseline three times per case and required identical output
digests. Both complete cohorts were qualified against the independent NumPy
float64 forward oracle. No optional Torch cross-check was run. Each GPU pytest
run deselected the other architecture's 32 cases and skipped no selected cases.

After network recovery, gradient Slurm jobs 68089875 (gfx942) and 68089876
(gfx950) each passed all 26 GPU numerical cases, with no failures or skips and
process exit status zero. Each case executed three times against its live NumPy
float64 oracle, for 156 gradient comparisons across both targets. The runs checked
input and auxiliary-buffer integrity, reset atomic workspaces before each
execution, and required the expected kernel launch count. Each job also passed
the 62 gradient host-contract tests. Both environments were Torch-free and
confirmed the requested architecture through the HIP runtime.

Returned JUnit reports and exit-status files confirm these counts. The submitted
gradient Python sources match the current working tree's AST; intervening source
changes were formatting only. These results establish source execution of the
initial six-lane cohorts, not installed or native-provider verification.

The gradient oracles accumulate in float64 over quantized inputs and have host
checks of the forward/gradient adjoint identity across grouping, stride and
dilation. Execution-plan tests build real kernels and check complete scalar and
pointer ABI bindings for all four gradient lanes. The combined convolution,
gradient and shared-reference host run passed 274 tests.

The first gfx942 forward allocation had no HIP-visible GPU and was replaced.
Initial gradient submissions did not reach numerical execution: one exposed a
missing pytest operation selector, which was corrected; node prolog delays and
subsequent Slurm controller socket timeouts prevented the corrected jobs from
allocating. The successful jobs above supersede that execution gap.

## Remaining verification and delivery

- Verify installed forward and gradient suites from a source-independent tree.
- Publish both forward archives and verify retrieval with an empty DVC cache.
  Local locks/pointers are prepared, but no DVC upload was performed.
- Enroll the newly published gfx950 forward pair in CMake/provider categories.
- Gradient tests currently use live CPU references. A frozen gradient replay
  format and a numerical certificate for nondeterministic accumulation are
  separate future work; the forward exact-output digest rule is unchanged.
- Native provider execution and final TheRock artifact delivery remain untested.

The exact submitted source archives, patches, scripts, candidate bundles and raw
job logs were retained locally. They include the working changes, not just the
baseline Git SHA. No push or DVC upload was performed during validation.
