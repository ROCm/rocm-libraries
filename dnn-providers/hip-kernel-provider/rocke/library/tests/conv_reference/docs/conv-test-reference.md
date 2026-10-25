<!--
Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
SPDX-License-Identifier: MIT
-->

# Convolution correctness with a pinned GPU reference

The convolution harness compares current GPU kernels with precompiled baseline
kernels qualified offline against an independent NumPy reference. Ordinary
verification computes neither CPU reference answers nor replacement budgets and
requires no Torch. Each architecture in
[`PublishedGpuReferences.cmake`](../../../../platform/cmake/PublishedGpuReferences.cmake)
has a published forward bundle with its own trusted lock and DVC pointer.
Gradients use separate live NumPy checks and do not reuse the forward certificate.

See the [shared compatibility and baseline policy](../../reference_common/docs/README.md#kernel-arguments-and-compatibility)
for kernel argument changes, frozen replay, and when a baseline must be replaced.

## Coverage and execution

The forward cohorts contain FP16 and BF16 cases for these geometries:
padded 3x3, pointwise 1x1, stride-2, dilation-2, groups=2, asymmetric spatial
dimensions, K=96, K=33, C=33, stem C=3, larger spatial/channel dimensions,
5x5 reduction, odd per-group channels, depthwise, anisotropic filter, and one
output pixel. Exact geometry lives in the architecture adapters' `CASES`; from
the rocKE root, `python library/tests/run_conv_reference.py cases --arch <arch>`
lists them.
Inputs are NHWC, weights KYXC with C/groups channels, and outputs NHWK.

The gfx942 current adapter uses an explicit 64x64x64 implicit-GEMM configuration;
there is no registered production forward instance list for that processor.
The gfx950 adapter requests `algorithm="implicit_gemm"` from production dispatch,
builds the selected instance, and uses its production launch plan. This keeps
implicit-GEMM coverage even when default dispatch prefers a direct depthwise
kernel. It preserves the selected instance's LLVM lowering flavor during
compilation. Current compilers still need to support the target and that flavor.

K=96 exercises two output-channel tiles with a partial second tile and cshuffle.
K=33 exercises the default epilogue. Host checks assert these properties and
complete scalar/pointer bindings. Kernel configurations can differ between
processors and current versus baseline implementations.

Qualification freezes the code object, signature, grid, block, every scalar AOT
argument, pointer-to-operand bindings, selected configuration, and old Python
runtime/worker. Replay replaces pointers with live allocations and does not
import current convolution geometry or dispatch. Both workers poison output with
NaNs, count launches, synchronize, check all outputs, and check unchanged inputs.
Baseline and current interpreters remain separate even when reused across cases.

The separate gradient suite runs the dgrad and wgrad cases in
[`gradient_cases.py`](../gradient_cases.py) on gfx942 and gfx950, including
grouped/strided/dilated geometry and odd channels. gfx950 dgrad uses production
implicit-GEMM selection except for its K=33 cases, which, like all gfx942 dgrad
cases, are explicitly configured. All wgrad cases are explicit
configurations because no wgrad instance lists are registered. They cover atomic
split-K, two-stage workspace reduction, and gfx950 merged groups. Each execution
is checked against a live float64 NumPy oracle at a normalized max-error limit
of 0.05, with three repetitions, immutable-buffer checks, and state reset.

These bounded suites exercise Python-emitted kernels. Native provider selection,
shipped kernel packs, direct implementation sweeps, stream-K, 3-D, and additional
processors remain outside this scope. See the [coverage inventory](../../../../docs/conv-coverage-roadmap.md).

## Numerical contract

Inputs use versioned PCG64 seed-0 uniform samples in [-1, 1), generated in activation
then weight order, cast to float32 and rounded to FP16 or BF16 RNE. Exact tensor
bytes and shapes are authenticated; the seed alone is not a reproducibility claim.
No input or answer tensors are shipped in the bundle.

Offline qualification computes grouped cross-correlation using NumPy float64 on
those exact quantized values. The reference is rounded float64 -> float32 -> output
storage -> float64. Output rounding follows the existing forward test's policy,
while the accumulation and random corpus differ from Torch's implementation.
Analytic grouping and spatial cases validate the oracle independently of rocKE's
geometry helpers.

Optional `qualify --torch-reference` also runs CPU
`torch.nn.functional.conv2d` in float64 and float32, using the same quantized
inputs with explicit NHWC/KYXC to NCHW/KCYX conversion. Torch rounds its own
outputs through float32 to FP16/BF16. The NumPy–Torch gap is diagnostic.
Qualification uses the largest baseline error
against any enabled oracle, with the same frozen NumPy scale, to set the
comparison budget. Per-case Torch version, implementation, answer digests,
oracle disagreement, and baseline errors are authenticated by the manifest lock.
Torch is imported only when this option is requested; a missing installation
fails qualification rather than skipping the check. No Torch answers or
dependency are needed by replay or installed CI. Reference worker interpreters
block Torch imports even when it is installed. The optional oracle checks are
excluded from installation and normal pytest discovery.

For independent reference R, qualified baseline O, and current output N, define
S = max(max(abs(R)), 1). S is recorded in the locked manifest. Both offline and
online distances use max(abs(A - B))/S, with subtraction/division rounded
conservatively. The original forward threshold is 0.05 for both dtypes; qualification
uses no fixed reserve. Each R_j below is an enabled offline reference; S
always comes from NumPy:

```text
baseline_bound = max over enabled references R_j of upper_bound(max(abs(O - R_j)) / S)
comparison_limit + baseline_bound < 0.05
max(abs(N - O)) / S <= comparison_limit
```

The comparison limit is rounded strictly below the exact rational remainder,
so acceptance implies `max(abs(N - R_j))/S < 0.05` for every enabled reference.
A baseline bound at or above the original tolerance fails qualification.
New qualifications record `budget_policy: "max-reference-bound-v1"` with the
legacy `margin` field set to zero, as both published bundles do. A bundle
without this marker retains its authenticated 0.0025 reserve and limits until
requalified and published. No workload coverage is removed by dropping the
reserve.

CI requires the baseline output digest to match the independently qualified result
before using that certificate. A nondeterministic baseline is rejected during
qualification or replay. Atomic gradients cannot simply reuse this contract:
repeated observed errors alone do not bound unknown future baseline outputs, and
atomic workspaces need initialization rather than indiscriminate NaN poisoning.

## Qualification, coverage updates, and publication

Follow [Adding or updating a pinned GPU reference](../../reference_common/docs/gpu-reference-workflow.md)
for the authoritative snapshot, qualification, candidate verification, DVC
publication, and installation procedure. Select operation `conv`. The workflow
includes optional offline CPU Torch cross-checks and requires a separate
Torch-free verification environment.

Adding geometries changes the locked cohort and requires a replacement bundle
qualified over the complete resulting cohort. The multi-tile and odd-channel
cases address the coverage request in
[PR #13106](https://github.com/ROCm/rocm-libraries/pull/13106#discussion_r4200111867).

## Installed tests

The umbrella option defaults ON for provider artifact builds and OFF for the
rocm-libraries superbuild, which does not fetch the bundles. CMake installs each
published operation/architecture bundle separately. The
installed bundle is beneath
`engines/test_arch_content/rocke/conv/<processor>/`, preserving architecture-specific,
test-only artifact packaging. The generic artifact contains the harness and trusted
lock. Host checks register as `rocke_conv_reference_unit_pytest`; a GPU entry,
`rocke_conv_gpu_<processor>_pytest`, registers for each published processor only
when reference installation is enabled and its bundle passes validation.
The provider category YAML selects these entries by pattern.

Test a replacement candidate before publication with the offline CLI's `verify`
command, passing explicit `--bundle` and `--lock` paths. CMake installs only the
published archives and always validates them against their committed architecture
locks. After publication, check the installed layout through the normal CMake path.

`ROCKE_INSTALL_TEST_GPU_REFERENCES=OFF` registers no pinned-reference GPU entries
for any operation. A job running on another processor in `AMDGPU_TARGETS` skips
the other processor's suite, including when convolution has no reference bundle for that
processor. Without `AMDGPU_TARGETS`, a different detected processor also skips.
Selection uses physical GPU identity; family names do not imply reference
enrollment. Missing hardware or a GPU outside the requested targets fails a
requested processor suite, and missing or corrupt reference data fails on matching
hardware. See the shared
[processor-selection rules](../../reference_common/docs/gpu-reference-workflow.md#processor-selection-in-therock-jobs).

Shared numeric, digest, archive, snapshot, and interpreter-transport support lives
in `library/tests/reference_common`. Both operations import that support directly. Existing frozen SDPA bundles remain self-contained; newly qualified bundles
also freeze the shared support needed by their replay workers.

Archive packaging uses the [packing and publication workflow](../../reference_common/docs/gpu-reference-workflow.md#5-review-pack-and-publish-the-replacement).

## Validation evidence

The [six-lane record](../../../../docs/conv-six-lane-validation.md) separates the original
October 14 qualification and installed checks from validation of the current
port and the October 21 requalification of both bundles at develop revision
`9b4fee7d990`. At that requalification, each baseline was independently
qualified against NumPy over all 32 cases of its cohort, without the optional
Torch cross-check. Do not infer current GPU validation from a successful replay
of the old branch history.

## Bundle identity

Each architecture's committed `baseline_lock.json` is the authoritative source
revision and manifest identity. Its matching DVC pointer records the archive
hash and size. Keep the lock and pointer together; a branch refresh or a current
adapter change does not itself replace the frozen baseline.

Fetch the published archives from the rocm-libraries root before provider
configuration:

```bash
dvc pull --recursive dnn-providers/hip-kernel-provider/rocke/library/tests/reference_bundles/conv
```

For qualification, candidate verification, publication, and installed checks,
follow the [shared workflow](../../reference_common/docs/gpu-reference-workflow.md). A change to cases,
inputs, the numerical contract, or the baseline requires complete independent
requalification. Never edit a qualified manifest to make a new cohort pass.
