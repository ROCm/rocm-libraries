<!--
Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
SPDX-License-Identifier: MIT
-->

# SDPA correctness with a pinned rocKE reference

The gfx942 lane runs current rocKE against a previously qualified rocKE kernel
on the GPU. It covers the base parameterizations in
[`test_attention_dense_gfx942_numeric.py`](../../test_attention_dense_gfx942_numeric.py):
fp16/bf16, D64/D128, GQA/MHA, causal/full attention, and default/persistent grids.
The gfx950 lane adds causal base configurations, described below.
The gfx1151 and gfx1201 lanes add standalone FP16 WMMA cases.
From the rocKE root, `python library/tests/run_sdpa_reference.py cases --arch <arch>`
lists an architecture's enrolled cases.
A gfx1200 adapter is available for qualification; its archive and installation
enrollment remain pending on-device qualification.
For gfx942/gfx950, current execution uses the same dispatch factory and `run_attention_dense_torch`
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

See [GPU attention coverage](../../../../docs/gpu-attention-test-coverage.md#the-gfx942-required-cases)
for the gfx942 configurations and broader gaps, and the
[strategy and extension guide](../../reference_common/docs/gpu-ci-pinned-rocke-test-reference-plan.md) for adding
architectures and operations. This is not a drop-in replacement for the skipped
Torch suites: it covers equivalent base configurations with different samples,
not their full shape, structural, or interoperability coverage.

Inputs use the versioned `numpy-pcg64-normal-f32-v1` recipe: reset PCG64 to seed 0
for each case, generate Q/K/V standard-normal float32 arrays in order, and round
to fp16 or bf16 (round-to-nearest-even for bf16). The original tests use Torch's
GPU random generator. Their intended distribution is similar, but their sample
values and generation path differ. Multiple executions repeat this fixed corpus;
they do not add random seeds. Generator incompatibility fails the digest check.

## Dense replay compatibility

The [shared compatibility and baseline policy](../../reference_common/docs/README.md#kernel-arguments-and-compatibility)
explains why the current kernel and the frozen reference can have different
signatures, and why a kernel update does not automatically replace its baseline.

New gfx942/gfx950 qualifications record the `dense-v2` replay ABI and freeze the
runtime shape and indexing scalars alongside the exported signature. gfx942
uses the production runtime-argument helper, including head counts and the
persistent grid's fast-division parameters. Older dense sources expose only
batch and sequence-length scalars. Qualification rejects a signature with
unbound arguments before writing a certificate. Replay binds fresh HIP pointers,
uses the enrolled case's scale, and restores the frozen runtime scalars.

Existing bundles without an ABI marker, or with `dense-v1`, retain their original
shape-binding behavior. Their authenticated contents and numerical gates do not
change. The gfx942 qualification adapter also supports the published baseline's
older `dense_spec_for_request` API, selecting persistence explicitly through its
request; current sources continue through the tuning API.

## gfx950 base dense cohort

The gfx950 lane covers the base configurations from
[`test_attention_dense_gfx950_numeric.py`](../../test_attention_dense_gfx950_numeric.py),
using an independent gfx950 bundle and lock. All cases are causal, B=1,
Sq=Sk=512, Hq=16, with the default softmax scale:

| Dtype | D | Hkv | Grid / production candidate |
|---|---:|---:|---|
| FP16 | 128 | 4 | Default / `gfx950_dense_grid` |
| FP16 | 128 | 4 | Persistent / `gfx950_dense_persist_widedma` |
| BF16 | 128 | 4 | Persistent / `gfx950_dense_persist_widedma` |
| BF16 | 128 | 4 | Default / `gfx950_dense_grid` |
| FP16 | 64 | 16 | Default / `gfx950_dense_grid` |
| FP16 | 128 | 16 | Persistent / `gfx950_dense_persist_widedma` |
| BF16 | 64 | 4 | Persistent / `gfx950_dense_persist` |

The adapter uses named production tuning candidates, matching the original
numeric tests. It selects wide DMA only for persistent, causal D128 cases.
Grid kernels take runtime shape scalars; persistent kernels bake
the shape. The existing replay worker supports both ABIs for this cohort.
The input recipe, independent oracle, absolute tolerances, and budget policy below
apply to both architectures. Error bounds and output identities are qualified
separately on each architecture.

This cohort excludes non-default scales, sliding windows, sinks, unequal lengths,
paging, binary-reuse assertions, and output-store equality checks. Those require
additional contracts or structural assertions and their own qualification.

The gfx950 bundle was qualified from committed baseline
`c6f41a64e12f8e1a51afb4ebce1b9dee256aaaed`, using Python 3.12.3, NumPy 2.5.4,
and COMGR with LLVM 23 on `gfx950:sramecc+:xnack-`. Three baseline executions
per case reproduced identical output digests. FP16 baseline maximum absolute
error bounds ranged from 0.000981060 to 0.001102120; BF16 bounds ranged from
0.007636342 to 0.008060832 (upper endpoints rounded upward here). Exact bounds
and compiler identity are in the authenticated manifest. No tolerance was widened.

Torch-free source verification of that bundle passed all seven comparisons
and three failure controls. The installed `rocke_sdpa_gpu_gfx950_pytest` entry
also passed all ten checks with the source tree moved aside. The installed shared
and SDPA host suites passed 82 and 26 tests respectively. This validates Python
reference installation and execution, not native provider execution or final
artifact splitting.

The archive is `library/tests/reference_bundles/sdpa/gfx950.tar.gz`, with MD5
`45ecc9f281ba2e1f29998f73929a41cf` and size 2,540,019 bytes. The archive was uploaded to DVC, downloaded into a new empty cache, and
authenticated against the committed lock. Follow the
[reference workflow](../../reference_common/docs/gpu-reference-workflow.md) for future updates.

## Standalone WMMA coverage

The gfx1151 and gfx1201 bundles each cover the complete Cartesian product below,
with B=2, Sq=Sk=64, Hq=4, contiguous `[B,S,H,D]` tensors, default SDPA scale,
and a nonpersistent grid:

| Dtype | Head dimension | KV heads | Mask |
|---|---|---|---|
| FP16 | 64, 128 | 4 (MHA), 1 (GQA) | Full, causal |

Both adapters use the production `WmmaFmhaFwdSpec` and builder from
`kernels.gfx1151.wmma_fmha_fwd`, selecting the target's MMA implementation.
They exercise the standalone verifier path. They do not exercise automatic
unified attention selection, native hipDNN/provider execution, BF16, persistent
kernels, sliding windows, unequal lengths, or paged KV.

The frozen `wmma-strided-v1` ABI contains Q/K/V/O pointers, a float32 log2-scaled
softmax scale, sequence lengths, and eight element strides. The production grid
helper selects `(4,4,2)` with block `(32,1,1)` for every case. Replay uses the
exported scalars and code object. Existing dense bundles retain their original ABI.

Current-source WMMA execution uses `wmma-strided-v2`, matching the production
runtime-shape ABI. Head counts and batch strides are runtime arguments; the
adapter supplies top-left masking, disabled windows, and a null LSE pointer with
LSE writes disabled. It obtains the signature from the selected production spec.
Qualification against older source snapshots still uses `wmma-strided-v1` when
head counts are spec fields. The gfx1151 archive retains its frozen v1 runner,
runtime, and code objects. The replacement gfx1201 archive uses v2 for both
baseline replay and current-source execution.

### Generic gfx12 code and processor-specific certificates

The gfx1200 and gfx1201 adapters both compile for `gfx12-generic`. Their
qualification identity remains the physical processor: the manifest records
`device_target` separately from `kernel.compiler_target`. Verification retains
exact old-output digests and old/current device-feature equality. A generic
code object does not authorize copying a certificate between processors.

Each qualified processor has its own archive, lock, and CTest entry. Payloads
stay under concrete `gfx1200` or `gfx1201` installation directories so TheRock's
artifact splitter and downloader select them correctly. A `gfx120X-all` job
requests artifacts for both processors. A pinned suite executes only when that
processor's archive is enrolled and its hardware is present.
See the [reference workflow](../../reference_common/docs/gpu-reference-workflow.md#generic-code-objects-and-processor-qualification)
for qualification and publication.

| Processor | Compiler target | Status |
|---|---|---|
| gfx1151 | gfx1151 | Existing archive published and enrolled |
| gfx1200 | gfx12-generic | Adapter and host tests ready; GPU qualification, archive, and installation enrollment pending |
| gfx1201 | gfx12-generic | Replacement archive qualified, published, and enrolled |

The gfx1201 replacement uses committed baseline
`32bcaf47b7482ef815ce997808979e50cfa36323`, which contains generic WMMA support.
Qualification used Python 3.12.3, NumPy 2.5.4, and COMGR LLVM 23 on gfx1201,
without Torch. Three baseline executions per case reproduced identical output
digests. Independent maximum absolute error bounds ranged from 0.000332024 to
0.001056340 (rounded upward); the FP16 tolerance remains `0.02`, with the
then-current `0.001` reserved margin (retained in that published bundle).
All eight archived code objects were inspected
and target `amdgcn-amd-amdhsa--gfx12-generic`.

Source and installed verification each passed all eight comparisons and three
failure controls, with zero selected GPU skips. The installed shared, SDPA, and
convolution host suites passed 96, 84, and 63 tests respectively, with the source
tree moved aside. All eight publication/install contract checks also passed.
Installation used a CMake installation of the Python reference suites without
native binaries. This qualification does not establish execution on gfx1200,
automatic unified attention dispatch, or native provider execution.

The published replacement archive has MD5 `8635f6cf21e20ef3bba89054f54ce952`
and size 2,596,336 bytes. A fresh anonymous download matched the committed DVC
pointer and authenticated against the baseline lock, including all payload
digests and the eight-case cohort. The previous
gfx1201 archive (`0373b7f4cea9cdfc0e718440d9e6226c`) contains processor-specific
binaries and cannot satisfy the generic compiler-target contract.

The unchanged gfx1151 archive was qualified from baseline
`c6f41a64e12f8e1a51afb4ebce1b9dee256aaaed`, with Python 3.12.3, NumPy 2.5.4,
and COMGR LLVM 23. Its published archive has MD5
`9d806c3533bf4ab618d9acb1b1f6cbed` and size 2,528,096 bytes. Its original
source and installed runs passed eight comparisons and three failure controls,
with zero selected skips. That setup used HIP 6.4 with ROCm 7.14 COMGR;
explicit runtime overrides were needed because the container HIP runtime failed
device initialization. Run a device preflight when selecting another setup.

### gfx120X family CI selection

`AMDGPU_FAMILIES=gfx120X-all` and `AMDGPU_TARGETS=gfx1200,gfx1201` describe the
job's requested artifacts, not two GPUs present on the runner. Session startup
compares the physical processor with these targets before opening a reference
bundle. It preserves `gfx1200` instead of using its `gfx12-generic` catalog name.
TheRock owns the family expansion; the reference selector consumes its concrete
target list without maintaining separate family aliases. If that list is absent,
the suite selects by physical processor alone. See the shared
[processor-selection rules](../../reference_common/docs/gpu-reference-workflow.md#processor-selection-in-therock-jobs).

For a job with that target list:

| Hardware and requested suite | Behavior |
|---|---|
| gfx1200 running `rocke_sdpa_gpu_gfx1201_pytest` | pytest exits with code 77; CTest reports **Skipped** |
| gfx1201 running `rocke_sdpa_gpu_gfx1201_pytest` | Runs every enrolled comparison and the three failure controls; missing or invalid reference data fails |
| No GPU, or a processor outside the requested target list, running the gfx1201 suite | Fails with a required-GPU error |
| gfx1200 with an explicit `--rocke-reference-arch gfx1200` request | Requires its own qualified bundle and lock; missing data or a gfx1201 certificate fails |

gfx1200 is not yet enrolled for installation, so normal CTest does not register
a gfx1200 pinned suite. Skipping the gfx1201 suite on gfx1200 fixes the selection
failure but does not establish gfx1200 pinned-reference coverage. Current-source
WMMA numerical tests are separate from this pinned suite.

The selection correction itself needs no replacement archive. Generic baseline
execution requires newly compiled and qualified code objects, so the generic
gfx1201 replacement has a new DVC object and matching lock. Its publication is
required for CI to fetch this branch's new pointer even when the runner will
later skip the gfx1201 suite. See the
[publication workflow](../../reference_common/docs/gpu-reference-workflow.md#5-review-pack-and-publish-the-replacement).

## Accuracy contract

For each case, `R` is the independent reference, `O` is the old result generated
live, and `N` is the current result:

```text
norm(N - R) <= norm(N - O) + norm(O - R) <= comparison_limit + baseline_bound
comparison_limit + baseline_bound < original_tolerance
```

The norm is the maximum absolute error across every output element. Qualification
measures a conservative bound on `norm(O - R)`. Both subtractions promote values
to float64 first, and the maximum difference is rounded upward. Budget allocation
uses exact rational arithmetic and rounds the comparison limit downward. The
comparison limit is strictly below the exact remainder `T - B`; this preserves
the original strict threshold without a fixed percentage reserve. A baseline
with `B >= T` fails qualification; the original tolerance cannot be increased
by its manifest. When Torch is enabled, B is the maximum baseline error bound
against NumPy and both recorded Torch answers.

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
not a tight no-regression threshold. Accepted current-to-independent error is
strictly below `0.02` for fp16 and `0.04` for bf16. A result
can change within that budget and still pass; an improvement can also fail the
comparison to the baseline and require independent diagnosis.

New qualifications record `budget_policy: "max-reference-bound-v1"` and a zero
legacy `margin` field. Storage remains tensor-free schema 2. Existing bundles
without this policy marker keep their authenticated 5% reserve and comparison
limits; loading them does not silently widen their gate. Requalify and publish
through the normal workflow to adopt the new policy. The reserve did not add
input, shape, dtype, or architecture coverage.

## Optional Torch reference agreement during qualification

The original gfx942/gfx950 dense numeric tests use Torch FP32 SDPA. The default
pinned qualification instead measures B against NumPy float64 on a portable,
quantized input corpus. These references and input samples are not interchangeable.
Existing published certificates do not acquire Torch evidence retroactively.

To measure their agreement on exactly the qualified inputs, add
`--torch-reference` (CPU) or `--torch-reference cuda` (ROCm GPU) to the offline
`qualify` command. Both Torch float64 and float32 SDPA answers are checked, using
explicit GQA expansion and the original tests' Python `1/sqrt(D)` scale. NumPy
retains the float32 ABI scale. The measured gap therefore includes that scale
representation difference. Torch outputs retain their computation dtype; they
are not rounded to FP16/BF16 output storage. The default Torch SDPA backend
selection is used, and its device, version, precision settings, checker digest,
answer digests, and conservative disagreement bounds are recorded in the locked
manifest. Requesting unavailable Torch or a missing/wrong ROCm GPU fails.

For each enabled reference R_j, measure the pinned output directly:

```text
B = max_j upper_bound(norm(O - R_j))
C < T - B
D <= C implies norm(N - R_j) <= D + B < T for every enabled R_j
```

The measured NumPy–Torch gap is retained as diagnostic evidence, not compared
against an arbitrary margin. A large baseline-to-reference error reduces the
comparison budget; `B >= T` fails qualification. This preserves the same
reference coverage without a fixed 5% reserve. Conservative arithmetic and
authenticated replay are still required.

CPU evidence applies only to the recorded CPU Torch answers. It does not qualify
the original GPU Torch backend, a different Torch version, different inputs, or
arbitrary future Torch outputs. Use `cuda` on the corresponding ROCm processor
for GPU reference evidence. This comparison also does not prove exact-real
arithmetic accuracy or accuracy no-regression. Existing Torch tests remain useful.

A reference-only audit can run without compiling or executing rocKE:

```bash
PYTHONPATH=library/tests:platform/python python -m sdpa_reference.torch_reference \
  --arch gfx942 --device cpu --output oracle-agreement.json
python -m pytest library/tests/sdpa_reference/check_torch_reference.py -q
```

This audit is not baseline qualification and cannot replace a published bundle.
To include the evidence in a certificate, rerun complete-cohort qualification
with the option, then follow the shared candidate verification and publication
workflow. Ordinary replay does not import Torch or run the independent references;
the optional checker is not copied into the frozen replay payload.

## Developer verification

Use a Python environment outside the source tree with `numpy`, `pytest`, and
`pytest-timeout`. A working HIP runtime and COMGR are required for current kernel
compilation. Set `ROCM_PATH` or the runtime's library overrides when needed, and
select `ROCKE_LLVM_FLAVOR` to match COMGR as described in
[`platform/AGENTS.md`](../../../../platform/AGENTS.md). Use an environment without Torch for reference verification. The shared worker
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

This command requires gfx942 and runs every case that `cases --arch gfx942`
lists, with two old and two current executions per case. It exits unsuccessfully
on a missing case, missing GPU, qualification failure, or comparison failure. It
never generates independent answers or fetches a reference during verification.

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

Follow [Adding or updating a pinned GPU reference](../../reference_common/docs/gpu-reference-workflow.md)
for the authoritative snapshot, qualification, candidate verification, DVC
publication, and installation procedure. Select operation `sdpa`; qualification
uses the independent NumPy SDPA oracle and this guide's numerical contract.
Adding cases or architectures requires qualification of the complete resulting
cohort. The workflow also covers replacement baselines and publication enrollment.

## Installed hip-kernel-provider lane

The existing TheRock CI flow builds and tests the installed provider artifact:

1. Source preparation fetches the DVC-tracked archive at
   `library/tests/reference_bundles/sdpa/<arch>.tar.gz` for each published SDPA
   architecture before provider configuration.
2. Provider CMake validates and extracts the archive against the committed
   architecture-specific baseline lock, then installs the test harness and bundle.
   `ROCKE_INSTALL_TEST_GPU_REFERENCES` defaults on for provider artifact builds
   and off in the rocm-libraries superbuild (`ROCM_LIBS_SUPERBUILD`). Setting it
   to OFF disables reference installation and GPU-test registration; host checks
   remain installed.
3. TheRock's `ml-libs/artifact-hipkernelprovider.toml` collects the harness into the
   generic test artifact. Its `**/engines/test_arch_content/**` rule collects the
   bundle installed at
   `bin/hip_kernel_provider/engines/test_arch_content/rocke/sdpa/<arch>/`;
   architecture splitting places it in the matching target test artifact, outside the
   release library payload.
4. The test job assembles the generic and matching architecture artifacts and
   prepares the Python/runtime dependencies. The existing provider runner selects
   CTest entries using the provider's category YAML. The
   `rocke_sdpa_gpu_<arch>_pytest` entries run the same pytest file used locally:
   one numerical comparison for each case the architecture enrolls, plus three
   negative checks. `rocke_sdpa_reference_unit_pytest` runs the SDPA host contract checks;
   `rocke_reference_common_pytest` runs shared archive and worker-isolation checks
   and compares each installed bundle's cases with the enrolled cohort. Neither
   host entry requires a GPU bundle.

This uses the existing workflows; no additional workflow is required. CI verifies
an already qualified baseline and cannot generate or approve its own replacement.

CMake installs only published archives validated against their committed locks.
For an unpublished candidate, use the offline CLI's `verify` command with explicit
`--bundle` and `--lock` paths. Once its archive, pointer, lock, and publication
entry are ready, validate installation through the normal CMake path.

Run the entry matching the local GPU, for example on gfx1201:

```bash
ctest --test-dir <install-prefix>/bin/hip_kernel_provider \
  -R '^rocke_sdpa_gpu_gfx1201_pytest$' --output-on-failure
```

The GPU entry is installed when a published bundle is configured. On matching
hardware, a missing bundle or failed qualification check is an error. Other
processors skip only when allowed by the
[processor-selection rules](../../reference_common/docs/gpu-reference-workflow.md#processor-selection-in-therock-jobs);
a missing GPU or a processor outside the requested targets fails a requested suite.

Further SDPA cohorts, runtime-shape reuse, sliding windows, and non-default
scales need their own qualification and retained structural assertions. The
published cohorts do not establish coverage for those paths.


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
selection still uses the detected GPU and CI's concrete `AMDGPU_TARGETS`.
Bundle and lock overrides require explicit operation and architecture options.
A lock override also requires a bundle path. For candidate qualification and
verification, follow the [reference workflow](../../reference_common/docs/gpu-reference-workflow.md).
These options select what pytest reads; they do not change what CMake installs.
CMake packages the standard archives for all operation/architecture pairs in
`platform/cmake/PublishedGpuReferences.cmake`, and pytest finds the installed target
bundle by default.
CMake has no per-architecture reference-install option; its publication lists
determine which pairs are installed. The pytest options select execution from
those suites or verify an explicit candidate bundle and lock.

## Bundle contents and provenance

The gfx942 DVC pointer is
[`reference_bundles/sdpa/gfx942.tar.gz.dvc`](../../reference_bundles/sdpa/gfx942.tar.gz.dvc).
Its schema-2 bundle contains:

```text
manifest.json
qualification-lock.json
payload/cases/<case-key>/kernel.hsaco
payload/runtime/rocke/
payload/runner/sdpa_reference/
```

The manifest records case definitions, input recipes/digests, baseline-output and
independent-reference digests, bounds, launch metadata, source/compiler/target
identity, Python/NumPy versions, and payload hashes. The source lock authenticates
the manifest; the bundled lock alone is not the trust anchor. Temporary NPZ files
can be used for worker transport, but tensor files are rejected from the archive.

Each case's manifest key also names its payload directory. Loading matches
entries to the enrolled cases by their full recorded case definition, not by
key, and requires exactly one entry per enrolled case and no others. New
qualifications key cases by id, such as `fp16-d128-h16-kv4-default-causal-b1-s512`.
The published bundles keep their original keys, which omit batch (`-b`) and
sequence length (`-s`); they load and replay unchanged.

The gfx942 baseline source revision is
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

Archive packaging uses the [packing and publication workflow](../../reference_common/docs/gpu-reference-workflow.md#5-review-pack-and-publish-the-replacement).
