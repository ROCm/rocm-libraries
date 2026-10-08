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
explicit local source and lock overrides can select a replacement candidate.

## Coverage and execution

The initial cohort contains twelve cases: FP16 and BF16 for each of padded 3x3,
pointwise 1x1, stride-2, dilation-2, groups=2, and asymmetric spatial dimensions.
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

## Qualification and verification

Run verification in an environment with NumPy, pytest, and pytest-timeout,
without Torch. HIP and COMGR are required for GPU execution. Optional Torch
qualification uses a separate offline environment.

From the rocKE root, with NumPy installed:

```bash
python library/tests/run_conv_reference.py snapshot \
  --repository <rocm-libraries-checkout> --revision <full-commit-sha> \
  --output <new-source-snapshot>
```

Use a committed baseline containing the convolution AOT ABI. On a gfx942 GPU with
HIP and COMGR available, qualify and verify it:

```bash
python library/tests/run_conv_reference.py qualify --arch gfx942 \
  --baseline <new-source-snapshot> --output <new-bundle> --repetitions 3
python library/tests/run_conv_reference.py verify --arch gfx942 \
  --bundle <new-bundle> --lock <new-bundle>/qualification-lock.json \
  --current-root .
```

To include the optional Torch CPU checks, install Torch in the offline
qualification environment and add `--torch-reference` to the qualification
command. A CPU-only Torch build suffices; rocKE still needs a GPU to execute
the baseline. The oracle-only checks can also run without a GPU; invoke this file explicitly
(it is not collected by normal pytest discovery or installed in CI):

```bash
python -m pytest library/tests/conv_reference/check_torch_reference.py -v
```

Qualification exports the committed production sources and records source,
compiler, target, oracle, input, baseline-output, and payload identities. Each
replacement baseline must be independently qualified; comparing only against its
predecessor is insufficient. Review the generated lock and qualification evidence
before promoting anything. CI never trusts the bundled lock automatically.

For local GPU pytest before promotion, explicitly select both bundle and lock:

```bash
ROCKE_TEST_CONV_REFERENCE_BUNDLE_GFX942=<new-bundle> \
ROCKE_TEST_CONV_REFERENCE_LOCK_GFX942=<reviewed-lock> \
ROCKE_TEST_REQUIRE_CONV_GPU=1 \
python -m pytest library/tests/test_conv_pinned_reference.py -v -rs
```

This runs twelve comparisons and three failure-detection tests: changed current
output, changed baseline output, and suppressed launch. Host checks need no GPU:

```bash
python -m pytest library/tests/test_conv_reference_contract.py \
  library/tests/test_reference_artifact.py -v
```

## Promotion and installed tests

After independent qualification and review:

1. Commit the reviewed lock at
   `library/tests/conv_reference/architectures/gfx942/baseline_lock.json`.
2. Pack the qualified bundle with the reviewed lock:

   ```bash
   python library/tests/reference_common/artifact.py pack --operation conv \
     --bundle <qualified-bundle> --lock <reviewed-lock> \
     --archive library/tests/reference_bundles/conv/gfx942.tar.gz
   ```

3. Track and publish that archive through DVC at
   `library/tests/reference_bundles/conv/gfx942.tar.gz.dvc`, following the repository's
   artifact publication process. Confirm a fresh download before CI enrollment.
4. Add `gfx942` to the `conv` array in
   `library/tests/reference_common/published_bundles.json`. Provider builds then
   install it alongside the other published bundles when
   `ROCKE_INSTALL_TEST_GPU_REFERENCES` is ON.

The umbrella option defaults ON for provider artifact builds and OFF for the
rocm-libraries superbuild, which does not fetch the bundles. CMake installs each
published operation/architecture bundle separately. The
installed bundle is beneath
`engines/test_arch_content/rocke/conv/gfx942/`, preserving architecture-specific,
test-only artifact packaging. The generic artifact contains the harness and trusted
lock. Host checks register as `rocke_conv_reference_unit_pytest`; the GPU entry
`rocke_conv_gpu_gfx942_pytest` registers only when reference installation is enabled and a
published or explicitly staged local bundle passes validation. Both entries are
in the provider category YAML.

For testing a replacement candidate before publishing it, configure explicitly:

```text
-DROCKE_INSTALL_TEST_GPU_REFERENCES=ON
-DROCKE_TEST_CONV_REFERENCE_INSTALL_SOURCE_gfx942=<qualified-bundle>
-DROCKE_TEST_CONV_REFERENCE_INSTALL_LOCK_gfx942=<reviewed-lock>
```

The explicit install lock is copied into the installed harness. Without that
override, CMake requires the committed architecture lock. OFF ignores bundle and
lock overrides and registers no pinned-reference GPU entries for any operation. Required tests fail on
missing gfx942 hardware or a missing/corrupt bundle; other unenrolled architectures
remain outside the cohort.

Shared numeric, digest, archive, snapshot, and interpreter-transport support lives
in `library/tests/reference_common`. Both operations import that support directly. Existing frozen SDPA bundles remain self-contained; newly qualified bundles
also freeze the shared support needed by their replay workers.

Archive packaging uses the [shared artifact command](../TESTING.md#shared-gpu-reference-artifact-command).

## Validation evidence

On 2026-10-05, local gfx942 validation established:

- All twelve cases qualified against NumPy using three baseline repetitions.
  A separate qualification also passed the optional CPU Torch float64/float32
  checks. This is offline evidence, not a Torch CI dependency.
- A fresh installation with SDPA and a local convolution candidate passed all
  five reference CTest suites in an environment that explicitly confirmed Torch
  was not installed: 137 host checks, 11 SDPA GPU checks, and 15 convolution GPU
  checks. Both current and pinned kernels executed.
- The convolution GPU checks include twelve numerical cases plus rejection of
  perturbed current output, perturbed replay output, and a missing launch.
- Repository pre-commit checks and configuration checks for default installation,
  disabled installation, local overrides, and missing/corrupt artifacts passed.

These checks cover the installed Python reference harness and Python-emitted
kernels. They do not establish native C++ provider execution or a completed
downstream CI run. The qualified NumPy baseline has since been uploaded to DVC,
retrieved through an empty cache, and validated against the source lock; the
publication registry now enables its default installation.

## Published bundle identity

The gfx942 archive is `library/tests/reference_bundles/conv/gfx942.tar.gz`,
tracked by its adjacent `.dvc` pointer. Its trusted lock is
`library/tests/conv_reference/architectures/gfx942/baseline_lock.json`.

- Baseline revision: `3c7094a26e39f4d42e1c0cbeee613f7d3f8ac4ef`.
- Manifest SHA-256: `18f1eefec57c9e64bada990260b99cd7a0bd256e2fcbe4b002507cdfac40c832`.
- Archive SHA-256: `8bbb0cad597b667b06b6d924f8a88ceaa27cf999f4edb44dbf2a42a368797fbe`.
- DVC MD5: `3e02db6084a58bae0a6c5397ca6a7247`.
- Archive size: 1,913,497 bytes.
- Qualification: NumPy float64, three baseline repetitions per case.

The separate optional Torch audit does not replace this bundle's NumPy
qualification manifest or lock. Fetch the published archive from the
rocm-libraries root before provider configuration:

```bash
dvc pull dnn-providers/hip-kernel-provider/rocke/library/tests/reference_bundles/conv/gfx942.tar.gz.dvc
```
