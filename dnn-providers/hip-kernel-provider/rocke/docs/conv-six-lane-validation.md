<!--
Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
SPDX-License-Identifier: MIT
-->

# Six-lane convolution implementation and validation

The October 14 results below describe the original implementation. The current
port uses production instance selection only for gfx950 forward and gfx950 dgrad
cases other than K=33; other lanes use the explicit configurations described in
the [coverage inventory](conv-coverage-roadmap.md). Historical production-selection
claims do not describe the current registry.

## October 21 requalification on develop

Both forward bundles were requalified at develop revision
`9b4fee7d99071ca2e40ae74dfa043919fb35d90a`, the refreshed base below. Their
previous baseline, `c767fbe1af37126cdc75b10f8e23bcee629debb5`, is not reachable
from develop, and the current gfx950 adapter cannot prepare its older dispatch
API. Before GPU qualification, the current adapters prepared all 32 cases per
processor from the new snapshot on a host, with only code-object generation and
kernel launch stubbed.

Qualification used the current harness, three baseline executions per case,
and the NumPy oracle without the optional Torch cross-check; Torch was absent.
Every baseline output was deterministic and matched the previous bundle's
digest, with unchanged inputs, case contracts, device targets, and baseline
error bounds. gfx942 code objects are identical to the previous bundle's; gfx950
code objects now come from the registered production instances and their LLVM
flavor. The new manifests record `max-reference-bound-v1`, so comparison limits
no longer subtract the earlier 0.0025 reserve.

| Check | gfx942 | gfx950 |
|---|---|---|
| Qualification | 32 cases | 32 cases |
| Offline `verify` of the candidate | 32 passed; current output identical to baseline | Same |
| Pinned pytest with candidate bundle and lock | 35 passed: 32 forward, 3 controls; no skips | Same |

The packed archives authenticate against the new locks. The October 20 checks
below replayed the previous bundles.

## October 20 refresh onto develop

The convolution work from `9c63320460f`, `aa525708084`, and `c701171f3a7`
was ported onto develop after the SDPA v2 merge (#13764). The refreshed base is
`9b4fee7d99071ca2e40ae74dfa043919fb35d90a`. Validation ran with the port on
`7accefb47591c68911b51212f502a8efdc28d3cd`; the two intervening develop commits
change only rocKE comments/docstrings and unrelated hipBLASLt code. No executable
rocKE changes were introduced by advancing that base.

The port consumes the current instance build and launch-plan APIs. gfx950
forward and ten dgrad cases use production implicit-GEMM instances and preserve
their identity's LLVM flavor. Other lanes retain explicit configurations,
including atomic, two-stage, and merged-group weight gradients. A refused
production request fails instead of falling back silently.

| Check | Result |
|---|---|
| Convolution, gradient, and shared-reference host contracts | 302 passed |
| Source gfx942, Slurm job 68130754 | 61 passed: 32 forward, 3 controls, 26 gradients; 32 other-target cases deselected |
| Source gfx950, Slurm job 68130755 | 61 passed: 32 forward, 3 controls, 26 gradients; 32 other-target cases deselected |
| Installed gfx942, Slurm job 68130838 | Four selected CTests passed; other-target CTest skipped; 26 focused gradients passed |
| Installed gfx950, Slurm job 68130839 | Four selected CTests passed; other-target CTest skipped; 26 focused gradients passed |

All selected numerical cases executed with no skips, using Python 3.12 and
NumPy 2.5.4 with Torch absent. Each gradient case executed three times. Source
and installed checks each performed 156 gradient comparisons across both
processors. The installed tree alone, dependencies, and a runner were shipped
to each installed-validation job; the imported `rocke` package resolved inside
that tree. The jobs confirmed the physical processor through HIP.

Each installed run passed `rocke_reference_common_pytest` (105 tests),
`rocke_conv_reference_unit_pytest` (134 tests), and `rocke_library_conv_pytest`
(205 tests, 13 existing skips, 178 subtests). The matching forward CTest passed
35 checks and deselected the other processor's 32 cases. The other forward
CTest skipped, and a direct invocation confirmed exit 77. The provider's actual
category parser and CTest selection included both forward entries and the
gradient-containing library GPU suite in `quick`, `standard`, `comprehensive`,
`full`, `pre-commit`, `smoke`, `ffm-quick`, and `ffm-full`.

Broader CMake publication, convolution ABI, layering, and dispatcher checks
produced 231 passes and one failure:
`test_conv_abi.py::test_cache_applies_runtime_launch_limits` rejects the expected
dgrad depthwise admission. The identical assertion fails with clean develop
test files and unchanged production modules. It is recorded as a preexisting
failure, not a passing gate. Repository pre-commit hooks passed. The final
frozen-runner import check also passed for both architecture adapters.

The existing 32-case archives were downloaded afresh and authenticated against
their committed pointers and locks, including all 663 payload files per
archive. No baseline was regenerated or republished. Physical processor
selection remains independent of build-family names, and no certificate is
transferred between processors.

The installed checks used the actual platform CMake Python installation and
provider CTest staging, with all six published archives. The harness omitted
native binary/export installation rules. It therefore establishes installed
Python test execution, not a native provider build, kernel-pack execution,
final TheRock artifact delivery, or a completed downstream CI run.

## Original October 14 validation

Date: 2026-10-14. Branch: `users/brockhargreaves-amd/multiarch-conv-references`,
based on `c767fbe1af37126cdc75b10f8e23bcee629debb5`. The reference/test
implementation is committed as `12c5b5eb8d2629cb21fb66ecd324cd20fdcacb5f`;
initial source jobs included those changes before commit. Production kernel
sources are unchanged.
The required scope is forward, backward-data and backward-weight on gfx942 and
gfx950.

| Lane | Implemented numerical cases | GPU result |
|---|---:|---|
| Forward / gfx942 | 32 | Independent qualification, source and installed verification passed (32 numerical + 3 controls) |
| Forward / gfx950 | 32 | Independent qualification, source and installed verification passed (32 numerical + 3 controls) |
| Backward-data / gfx942 | 12 | Source and installed tests passed; explicit specs |
| Backward-data / gfx950 | 12 | Source and installed tests passed; production selection |
| Backward-weight / gfx942 | 14 | Source and installed tests passed; production-selected atomic/two-stage paths |
| Backward-weight / gfx950 | 14 | Source and installed tests passed; production-selected atomic/two-stage paths, including merged groups |

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
changes were formatting only. These jobs establish source execution of the
initial six-lane cohorts. Installed verification is recorded separately below.

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

## Installed verification and publication

Slurm jobs 68089961 (gfx942) and 68089963 (gfx950) each configured and installed
the Python tests through platform CMake with reference installation enabled.
Their input was implementation commit `12c5b5eb8d2629cb21fb66ecd324cd20fdcacb5f`
plus the gfx950 publication enrollment change. The source tree was moved aside,
source paths were removed from the environment, and the imported `rocke` package
was verified to reside in the installed tree. Both jobs confirmed the target
through HIP and used Python 3.12 and NumPy 2.5.4 without Torch.

Each job selected six registered CTests: five passed and the other architecture's
forward entry skipped with the expected return code. Process exit status was zero.
The results per target were:

| Installed suite | Result |
|---|---|
| `rocke_reference_common_pytest` | 82 passed |
| `rocke_conv_reference_unit_pytest` | 130 passed |
| `rocke_library_conv_pytest` | 140 passed, 13 skipped, 178 subtests passed; includes all 62 new gradient host checks |
| Matching `rocke_conv_gpu_<arch>_pytest` | 32 numerical comparisons and 3 failure controls passed; 32 other-target cases deselected |
| `rocke_library_gpu_pytest` | 119 passed, 182 skipped, 86 subtests passed; includes all 12 dgrad and 14 wgrad cases |

All new gradient numerical cases executed three times with no skips, yielding
156 additional numerical comparisons across these installed runs. The broader
library GPU suite also contains structural tests; its 119 passes are not a GPU
comparison count. The library-suite skips include existing tests requiring Torch
or the native C++ extension, tests for other architectures, and existing
`library.dispatch` import checks that do not resolve in this installed layout.

Both forward archives were published to the configured DVC storage remote.
A new isolated DVC workspace with an empty cache fetched both anonymously, with
AWS credentials disabled for the check. Downloaded archive sizes and content
hashes matched the committed pointers and the archives used for installed tests.
Extraction authenticated each manifest against its committed baseline lock and
verified every payload hash.

The publication registry now enrolls both convolution targets. The actual provider
category parser was applied to the generated installed CTest file; CTest selection
confirmed both forward entries and the gradient-containing library GPU suite in
`quick`, `standard`, `comprehensive`, `full`, `pre-commit`, `smoke`, `ffm-quick`
and `ffm-full`. All seven CMake publication/install tests passed.

## Coverage limits

- Gradient tests currently use live CPU references. A frozen gradient replay
  format and a numerical certificate for nondeterministic accumulation are
  separate future work; the forward exact-output digest rule is unchanged.
- Native provider execution and final TheRock artifact delivery remain untested.

The installation harness exercises the real Python installation and CTest
generation paths while omitting native build targets. It does not establish a
native provider build, kernel-pack execution, final artifact splitting, or a
completed downstream CI run. The exact submitted source archives, patches,
scripts, bundles, returned reports and raw job logs were retained locally.
