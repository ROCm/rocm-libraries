<!--
Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
SPDX-License-Identifier: MIT
-->

# SDPA correctness with a pinned rocKE reference

The gfx942 pilot runs current rocKE against a previously qualified rocKE kernel
on the GPU. It covers the eight base parameterizations in
[`test_attention_dense_gfx942_numeric.py`](../library/tests/test_attention_dense_gfx942_numeric.py):
fp16/bf16, D64/D128, GQA/MHA, causal/full attention, and default/persistent grids.
Current execution uses the same dispatch factory and `run_attention_dense_torch`
entry as those tests, passing raw HIP buffers through its existing pointer ABI.
The selected backend is Python, matching that entry point. This lane does not
establish C++ provider execution or Torch tensor interoperability.

The original Torch tests remain available. This pilot uses a separately
qualified, portable input corpus: NumPy normal inputs rounded to fp16 or bf16,
regenerated at runtime and checked against the qualified input digests.
No input or answer tensors are stored in the schema-2 bundle. Its independent reference
is float64 NumPy SDPA, including GQA expansion, causal masking, and the f32 scale
passed through the kernel ABI. It preserves the original maximum absolute error
tolerances (`0.02` for fp16, `0.04` for bf16); it does not claim identical Torch
random inputs or bitwise equivalence between NumPy and Torch's reference outputs.

See [GPU attention coverage](gpu-attention-test-coverage.md#the-eight-required-cases)
for the exact eight configurations and broader gaps, and the
[strategy and extension guide](gpu-ci-pinned-rocke-test-reference-plan.md) for adding
architectures and operations. This is not a drop-in replacement for the skipped
Torch suites: it covers equivalent base configurations with different samples,
not their full shape, structural, or interoperability coverage.

Inputs use the versioned `numpy-pcg64-normal-f32-v1` recipe: reset PCG64 to seed 0
for each case, generate Q/K/V standard-normal float32 arrays in order, and round
to fp16 or bf16 (round-to-nearest-even for bf16). The original tests use Torch's
GPU random generator. Their intended distribution is similar, but their sample
values and generation path differ. Multiple executions repeat this fixed corpus;
they do not add random seeds. Generator incompatibility fails the digest check.

## Accuracy contract

For each case, `R` is the independent reference, `O` is the old result generated
live, and `N` is the current result:

```text
norm(N - R) <= norm(N - O) + norm(O - R) <= comparison_limit + baseline_bound
comparison_limit + baseline_bound + margin <= original_tolerance
margin > 0
```

The norm is the maximum absolute error across every output element. Qualification
measures a conservative bound on `norm(O - R)`. Both subtractions promote values
to float64 first, and the maximum difference is rounded upward. Budget allocation
uses exact rational arithmetic and rounds the comparison limit downward. The
reserved margin is `0.001` for fp16 and `0.002` for bf16. A baseline without
positive remaining headroom fails qualification; the original tolerance cannot
be increased by its manifest.

Qualification records input and old-output digests. CI executes the old kernels
and requires those same digests before using the offline bounds. An unknown old
output is a qualification failure. Repeated qualification runs check deterministic
reproduction; their maximum is not presented as a bound on arbitrary future
outputs. The digest check limits the guarantee to the qualified outputs and
inputs. The guarantee is relative to the independent reference's represented
result, not an additional proof of its error against exact real arithmetic.

Each worker initializes output storage to NaNs, counts launches, synchronizes
HIP, and copies results back. Non-finite output, unwritten output, changed inputs,
missing hardware, corrupt artifacts, and out-of-budget results fail the required
lane. Every current repetition must pass. Baseline and current workers are reused
across cases but remain separate processes with isolated imports and caches.

This is a sufficient accuracy gate within the existing Torch-derived limits,
not a tight no-regression threshold. Accepted current-to-independent error is at
most `0.019` for fp16 and `0.038` for bf16 after reserving the margins. A result
can change within that budget and still pass; an improvement can also fail the
comparison to the baseline and require independent diagnosis.

## Developer verification

Use a Python environment outside the source tree with `numpy`, `pytest`, and
`pytest-timeout`. A working HIP runtime and COMGR are required for current kernel
compilation. Set `ROCM_PATH` or the runtime's library overrides when needed, and
select `ROCKE_LLVM_FLAVOR` to match COMGR as described in
[`platform/AGENTS.md`](../platform/AGENTS.md). Use an environment without Torch for reference verification. The shared worker
launcher blocks Torch imports in both baseline and current interpreters.

From the rocm-libraries root, fetch the existing archive with DVC installed:

```bash
dvc pull dnn-providers/hip-kernel-provider/rocke/library/tests/reference_bundles/sdpa/gfx942.tar.gz.dvc
```

Then, from the rocKE root, extract and validate it for source testing:

```bash
python library/tests/reference_common/artifact.py unpack --operation sdpa \
  --archive library/tests/reference_bundles/sdpa/gfx942.tar.gz \
  --bundle library/tests/reference_bundles/sdpa/gfx942 \
  --lock library/tests/sdpa_reference/architectures/gfx942/baseline_lock.json
```

The source default lookup uses that extracted directory. For an explicit bundle,
run from the rocKE root:

```bash
python library/tests/run_sdpa_reference.py verify --arch gfx942 \
  --bundle <qualified-bundle-directory> \
  --current-root .
```

This command requires gfx942 and runs all eight cases, with two old and two
current executions per case. It exits unsuccessfully on a missing case, missing
GPU, qualification failure, or comparison failure. It never generates independent
answers or fetches a reference during verification.

The same cases can be run through pytest:

```bash
python -m pytest library/tests/test_sdpa_pinned_reference.py \
  --rocke-reference-operation sdpa --rocke-reference-arch gfx942 \
  --rocke-reference-bundle <qualified-bundle-directory> -v -rs
```

The GPU suite also checks three failure paths: perturbing the final current
result must exceed its comparison budget, perturbing a live old result must
invalidate its qualified digest, and suppressing a launch must leave poisoned
output that the worker rejects. These checks use real HIP allocations and live
GPU results; they do not change the qualified bundle.

Run the CPU contract checks with:

```bash
python -m pytest library/tests/test_sdpa_reference_contract.py \
  library/tests/test_reference_common.py -v
```

These check analytic SDPA results, bf16 interpretation, preservation of the
existing parameter cohort, conservative error budgets, and rejection of corrupt
or unqualified data. A subprocess import check also ensures that test support
packages cannot shadow the selected production library. They do not count as
GPU correctness results.

## Offline qualification and baseline promotion

Follow [Adding or updating a pinned GPU reference](gpu-reference-workflow.md)
for the authoritative snapshot, qualification, candidate verification, DVC
publication, and installation procedure. Select operation `sdpa`; qualification
uses the independent NumPy SDPA oracle and this guide's numerical contract.
Adding cases or architectures requires qualification of the complete resulting
cohort. The workflow also covers replacement baselines and publication enrollment.

## Installed hip-kernel-provider lane

The existing TheRock CI flow builds and tests the installed provider artifact:

1. Source preparation fetches the DVC-tracked archive at
   `library/tests/reference_bundles/sdpa/gfx942.tar.gz` before provider configuration.
2. Provider CMake validates and extracts the archive against the committed
   architecture-specific baseline lock, then installs the test harness and bundle.
   `ROCKE_INSTALL_TEST_GPU_REFERENCES` defaults on for provider artifact builds
   and off in the rocm-libraries superbuild (`ROCM_LIBS_SUPERBUILD`). Setting it
   to OFF disables reference installation and GPU-test registration; host checks
   remain installed.
3. TheRock's `ml-libs/artifact-hipkernelprovider.toml` collects the harness into the
   generic test artifact. Its `**/engines/test_arch_content/**` rule collects the
   bundle installed at
   `bin/hip_kernel_provider/engines/test_arch_content/rocke/sdpa/gfx942/`;
   architecture splitting places it in the gfx942 test artifact, outside the
   release library payload.
4. The test job assembles the generic and matching architecture artifacts and
   prepares the Python/runtime dependencies. The existing provider runner selects
   CTest entries using the provider's category YAML. `rocke_sdpa_gpu_gfx942_pytest` runs
   the same pytest file used locally: eight numerical cases and three negative
   checks. `rocke_sdpa_reference_unit_pytest` runs the SDPA host contract checks;
   `rocke_reference_common_pytest` runs shared archive and worker-isolation checks.
   Both host entries run independently of GPU bundles.

This uses the existing workflows; no additional workflow is required. CI verifies
an already qualified baseline and cannot generate or approve its own replacement.

CMake installs only published archives validated against their committed locks.
For an unpublished candidate, use the offline CLI's `verify` command with explicit
`--bundle` and `--lock` paths. Once its archive, pointer, lock, and publication
entry are ready, validate installation through the normal CMake path.

Run the installed GPU entry with:

```bash
ctest --test-dir <install-prefix>/bin/hip_kernel_provider \
  -R '^rocke_sdpa_gpu_gfx942_pytest$' --output-on-failure
```

The GPU entry is installed when a qualified bundle is configured. In that required
lane, a missing bundle, missing GPU, or failed qualification check is an error;
unqualified GPU architectures remain outside this pilot and skip its cases.

Further SDPA cohorts, runtime-shape reuse, sliding windows, and gfx950 need their
own qualification and retained structural assertions. This eight-case pilot
does not replace those tests or establish coverage for them.


## Configuration controls

Normal provider CI uses the defaults. The settings are:

| Setting | Type | Purpose |
|---|---|---|
| `ROCKE_INSTALL_TEST_GPU_REFERENCES` | CMake Boolean | Install all published GPU reference bundles and enroll their GPU tests |
| `--rocke-reference-operation`, `--rocke-reference-arch` | Pytest options | Select the operation and GPU architecture |
| `--rocke-reference-bundle` | Pytest option | Select a bundle for the selected operation/architecture |
| `--rocke-reference-lock` | Pytest option | Select an explicit candidate lock instead of the committed lock |

Explicit operation/architecture selection makes the suite required, including
failure on a missing bundle. CTest supplies these options. Architecture-lane
selection still uses the detected GPU and CI's `AMDGPU_FAMILIES`/`AMDGPU_TARGETS`.
Bundle and lock overrides require explicit operation and architecture options.
A lock override also requires a bundle path. For candidate qualification and
verification, follow the [reference workflow](gpu-reference-workflow.md).
These options select what pytest reads; they do not change what CMake installs.
CMake packages the standard archives for all operation/architecture pairs in
`platform/cmake/PublishedGpuReferences.cmake`, and pytest finds the installed target
bundle by default.
There is no separate architecture-selection option or unqualified bundle alias.
Other detected, unenrolled GPU architectures remain outside the required cohort.

## Bundle contents and provenance

The DVC pointer is
[`reference_bundles/sdpa/gfx942.tar.gz.dvc`](../library/tests/reference_bundles/sdpa/gfx942.tar.gz.dvc).
Its schema-2 bundle contains:

```text
manifest.json
qualification-lock.json
payload/cases/<case-id>/kernel.hsaco
payload/runtime/rocke/
payload/runner/sdpa_reference/
```

The manifest records case definitions, input recipes/digests, baseline-output and
independent-reference digests, bounds, launch metadata, source/compiler/target
identity, Python/NumPy versions, and payload hashes. The source lock authenticates
the manifest; the bundled lock alone is not the trust anchor. Temporary NPZ files
can be used for worker transport, but tensor files are rejected from the archive.

The current baseline source revision is
`dd77374194a3ea1a7258cd54f87af6747b6362a5`; manifest SHA-256 is
`e73b01b9aacb8c7586e914fbeb8844f804287c09422c57e8364026d4b47d5be1`.
The archive's DVC MD5 is `77741d920b443a6a5537085755bafc65`.
Qualification recorded Python 3.12.3, NumPy 2.5.3, LLVM flavor `llvm23`, target
`gfx942:sramecc+:xnack-`, and COMGR library SHA-256
`6f3219087c6d2be5504f93e74afe37c28862420bbf40acb49861824371adfff2`.
The compiler hash identifies the resolved library more precisely than the LLVM
flavor alone.

The current archive was migrated from the tensor-containing qualification bundle.
Its historical `reference.input_generator` description still says “stored
quantized bits”; the schema-2 `input_generation` contract and actual payload
implement regeneration with digest checks. Migration preserved the qualified
inputs, output identities, and budgets rather than silently requalifying them.

| Extracted content | Files | Bytes |
|---|---:|---:|
| Frozen runtime | 474 | 6,816,297 |
| Compiled kernels | 8 | 103,256 |
| Manifest | 1 | 79,678 |
| Replay runner/support | 3 | 18,126 |
| Qualification lock | 1 | 174 |
| Total | 487 | 7,017,531 |

The compressed DVC archive is **1,900,516 bytes**. The former tensor-containing
archive was 25,568,429 bytes; tensor removal reduced compressed size by about
92.6%. DVC archive size and the separately compressed CI artifact size differ.

## Validation evidence

These are dated observations, not guarantees for every future environment:

| Check | Observed result |
|---|---|
| Baseline qualification | Eight cases qualified against float64 NumPy on gfx942, with repeated baseline output identity |
| Additional gfx942 audit | Three baseline/current executions per case; current-to-baseline distances zero, independent bounds/digests reproduced, boundary rejection checked; 54 host and 11 GPU checks passed |
| Independent Torch cross-check | CPU Torch 2.14.1 SDPA math backend on the same inputs agreed with NumPy: maximum differences about 1.54e-6 in float32 and 1.44e-15 in float64; this did not execute the original ROCm Torch GPU suites |
| Negative numerical audit | Wrong masks/scales were rejected for all eight cases |
| Actual artifacts, run 36897764926 | Generic harness and gfx942 payload assembled; all 487 bundle files matched DVC, eight HSACO were architecture-specific, no references in the release library artifact; 54 host checks passed |
| Supplied CI log, execution 2026-10-01 18:43 UTC | 54 host checks passed (CTest 1.47 s); eight numerical plus three negative GPU checks passed (CTest 10.05 s); all 14 selected CTest entries passed |

The observed maximum independent-reference errors were approximately
`0.001116874` for fp16 and `0.008060046` for bf16. Per-case online comparison
limits are approximately `0.0178831–0.0187090` and `0.0299400–0.0299706`,
respectively; the manifest holds the exact values.

The original GPU CTest took 34.51 s; subsequent optimized CI observations were
7.74 s and 10.05 s. A separate audit had an unexplained 408.11 s GPU-suite run;
these measurements are environment-dependent, not a performance guarantee.
Legacy Torch-dependent tests still skip in the supplied CI environment, as does
an optional jsonschema check. Passing the selected CTests does not establish
execution of those skipped numerical cohorts.

Archive packaging uses the [packing and publication workflow](gpu-reference-workflow.md#5-review-pack-and-publish-the-replacement).
