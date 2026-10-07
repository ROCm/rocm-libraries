<!--
Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
SPDX-License-Identifier: MIT
-->

# Convolution correctness with a pinned GPU reference

The convolution harness compares current GPU kernels with precompiled baseline
kernels qualified offline against an independent NumPy reference. Ordinary
verification computes neither CPU reference answers nor replacement budgets and
requires no Torch. This is infrastructure for a bounded gfx942 forward cohort;
the gfx942 baseline lock and DVC pointer identify its published bundle. The
publication registry enables default provider installation alongside SDPA;
the offline CLI accepts explicit bundle and lock paths to verify replacement candidates.

## Coverage and execution

The cohort contains sixteen cases: FP16 and BF16 for each of padded 3x3,
pointwise 1x1, stride-2, dilation-2, groups=2, asymmetric spatial dimensions,
K=96 output channels, and K=33 output channels.

The K=96 cases exercise two output-channel tiles, including a partial second
tile, with the cshuffle epilogue. K=33 exercises the default epilogue with odd
output-channel counts. Host checks assert these production dispatch properties;
GPU qualification and replay check the resulting kernels and their launch ABI.
Exact geometry is in
[`architectures/gfx942/__init__.py`](../library/tests/conv_reference/architectures/gfx942/__init__.py).
Inputs are NHWC, weights KYXC with C/groups channels, and outputs NHWK. These are
selected configurations, not a replacement for the existing convolution sweeps.

The current worker calls production `dispatch_conv_grouped`, builds the selected
forward spec with the Python engine, and launches through the convolution AOT
argument ABI. It covers Python-emitted implicit-GEMM kernels and production
selection/argument construction; it does not exercise the C++ provider or shipped
kernel-pack lookup. Direct convolution, backward-data, backward-weight, 3-D,
additional architectures, and pipeline sweeps are outside this first cohort.

Qualification freezes the code object, signature, grid, block, every scalar AOT
argument, pointer-to-operand bindings, selected dispatch spec, and old Python
runtime/worker. Replay replaces pointers with live allocations and does not
import current convolution geometry or dispatch. Both workers poison output with
NaNs, count launches, synchronize, check all outputs, and check unchanged inputs.
Baseline and current interpreters remain separate even when reused across cases.

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
outputs through float32 to FP16/BF16. Each Torch answer must agree with NumPy
within the reserved margin. Qualification uses the largest baseline error
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
reserves a 0.0025 margin. Each R_j below is an enabled offline reference; S
always comes from NumPy:

```text
baseline_bound = max over enabled references R_j of upper_bound(max(abs(O - R_j)) / S)
comparison_limit + baseline_bound + margin <= 0.05
max(abs(N - O)) / S <= comparison_limit
```

CI requires the baseline output digest to match the independently qualified result
before using that certificate. A nondeterministic baseline is rejected during
qualification or replay. Atomic gradients cannot simply reuse this contract:
repeated observed errors alone do not bound unknown future baseline outputs, and
atomic workspaces need initialization rather than indiscriminate NaN poisoning.

## Qualification, coverage updates, and publication

Follow [Adding or updating a pinned GPU reference](gpu-reference-workflow.md)
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
`engines/test_arch_content/rocke/conv/gfx942/`, preserving architecture-specific,
test-only artifact packaging. The generic artifact contains the harness and trusted
lock. Host checks register as `rocke_conv_reference_unit_pytest`; the GPU entry
`rocke_conv_gpu_gfx942_pytest` registers only when reference installation is enabled and a
published bundle passes validation. Both entries are
in the provider category YAML.

Test a replacement candidate before publication with the offline CLI's `verify`
command, passing explicit `--bundle` and `--lock` paths. CMake installs only the
published archives and always validates them against their committed architecture
locks. After publication, check the installed layout through the normal CMake path.

`ROCKE_INSTALL_TEST_GPU_REFERENCES=OFF` registers no pinned-reference GPU entries
for any operation. Required tests fail on missing gfx942 hardware or a
missing/corrupt bundle; other unenrolled architectures remain outside the cohort.

Shared numeric, digest, archive, snapshot, and interpreter-transport support lives
in `library/tests/reference_common`. Both operations import that support directly. Existing frozen SDPA bundles remain self-contained; newly qualified bundles
also freeze the shared support needed by their replay workers.

Archive packaging uses the [packing and publication workflow](gpu-reference-workflow.md#5-review-pack-and-publish-the-replacement).

## Validation evidence

On 2026-10-07, the expanded gfx942 cohort passed offline qualification and
Torch-free installed validation:

- All sixteen cases qualified with three deterministic baseline executions each,
  checked against NumPy float64 and CPU Torch float64/float32 references.
- Exported metadata confirmed two N tiles with a partial tail for K=96 and the
  default epilogue for K=33, in both FP16 and BF16.
- The source convolution GPU suite passed sixteen comparisons and three failure
  controls using the candidate bundle and lock in an environment without Torch.
- A fresh installation passed all five reference CTest suites with Torch absent:
  153 host checks, 11 SDPA GPU checks, and 19 convolution GPU checks. The source
  CMake suite also passed seven publication/install checks.
- The replacement archive was uploaded to DVC, downloaded into an empty cache,
  and authenticated against the updated committed lock.

These checks cover the installed Python reference harness and Python-emitted
kernels. They do not establish native C++ provider execution or a completed
downstream CI run. Qualification uses the committed production sources named in
the lock and the expanded reference adapter frozen in the bundle.

## Bundle identity

The authoritative baseline revision and manifest digest are in the
[committed lock](../library/tests/conv_reference/architectures/gfx942/baseline_lock.json).
The archive's content identity and size are in its
[DVC pointer](../library/tests/reference_bundles/conv/gfx942.tar.gz.dvc).
Keep those files together when updating coverage; do not maintain a second copy
of their hashes in this document. The bundle manifest records per-case bounds,
input/output identities, compiler information, and offline oracle checks.

Fetch the published archive from the rocm-libraries root before provider
configuration:

```bash
dvc pull dnn-providers/hip-kernel-provider/rocke/library/tests/reference_bundles/conv/gfx942.tar.gz.dvc
```
