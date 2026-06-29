# rocKE Client

`rocKE-client` is the client API for the [rocKE project](../rocKE). This library will
allow kernel delivery teams to deliver kernels developed in rocKE to clients, including
[hipDNN](../../../projects/hipdnn/).


## AOT-compiled kernels

This library currently contains a proof of concept of compiling rocKE kernels into
AMD GPU .hsaco binaries at build time, generating kernel launch metadata, and performing
kernel launches to test this process end to end with support for multiple architectures.

> The executable kernel source currently lives in the sibling `rocKE` project.
> The expected end state is for that source to move into the appropriate
> `rocKE-client/kernels/<op>/<family>/` directory once the kernel family layout
> is ready to own it.

## Prerequisites

- Python 3.
- Python packages from `requirements.txt`: `numpy==2.5.0`,
  `jsonschema==4.26.0`, and `pytest==9.1.1`.
- The sibling rocKE source tree at `../rocKE`, or an explicit
  `-DROCKE_CLIENT_ROCKE_SOURCE_DIR=<path-to-rocKE>` / `PYTHONPATH` override.
- ROCm/HIP/comgr system libraries for real HSACO generation.
- A HIP-visible device matching the requested `--arch` for numeric tests; those
  tests skip with return code 77 when no matching device is visible.

## Layout

`rocKE-client/aot/` owns common instance parsing, JSON Schema validation,
sidecar helpers, CMake helpers, and the AOT build CLI. Kernel family directories
own checked-in concrete instances, operation-specific parsing/build/sidecar
logic, CMake registration, schemas, and family-local tests.

## Checked-in SDPA instances

The first AOT family is SDPA forward `fmha_fwd_mfma`. It has one checked-in
smoke instance for `gfx1151` and one for `gfx942`:

```text
kernels/sdpa/fmha_fwd_mfma/gfx1151/aot_list.json
kernels/sdpa/fmha_fwd_mfma/gfx942/aot_list.json
```

Each `aot_list.json` is a JSON array of instance objects; both currently hold a
single smoke instance.

Both instances use:

| field | value |
|---|---|
| `schema` | `rocke.aot.instance/v1` |
| `op` / `family` | `sdpa_fwd` / `fmha_fwd_mfma` |
| `dtype` / `canonical_layout` | `fp16` / `BSHD` |
| `seqlen_q` / `seqlen_k` | `64` / `64` |
| `num_query_heads` / `num_kv_heads` | `4` / `4` |
| `head_size` | `64` |
| `block_size_q` / `block_size_k` | `16` / `64` |
| `mask_mode` | `none` |
| `selection.batch` | `{"min": 1, "max": 64}` |
| `test_profiles` | `{"batch": 1}`, `{"batch": 2}`, `{"batch": 64}` |

Runtime attribute constraints for these smoke instances require:

- `mask_mode == "none"`
- `dropout_probability == 0.0`
- `scale_policy == "default_1_over_sqrt_d"`
- `padding_mask == false`
- `alibi_mask == false`

Artifact basenames are canonical and must match the JSON `name`:

```text
{op}_{family}_{dtype}_{layout}_{arch}_q{seqlen_q}_k{seqlen_k}_hq{num_query_heads}_hkv{num_kv_heads}_d{head_size}_{mask_mode}
```

The CMake target/output-directory names are shorter than artifact basenames:
`sdpa_fwd_fmha_mfma_gfx1151` and `sdpa_fwd_fmha_mfma_gfx942`. The artifacts
inside those directories still use the full canonical basename above.

## Instance parsing and schemas

Common instance files use schema `rocke.aot.instance/v1`. The generic envelope
schema lives at `aot/schemas/instance.schema.json` and requires top-level
`schema`, `name`, `op`, `family`, `arch`, `compile_spec`, `selection`, and
`test_profiles`.

The helper package `rocke_client_aot` exports:

- `INSTANCE_SCHEMA`
- `InstanceError`
- `KernelInstanceActions`
- `ParsedInstance`
- `attributes_match_constraints`
- `normalize_attribute_constraints`
- `parse_instance_list`

`parse_instance_list()` loads the per-architecture `aot_list.json` array, loads
the kernel-local `aot_instance.py`, validates each common envelope, normalizes
generic `selection.attribute_constraints`, enforces unique instance names, and
delegates operation-specific work to handler callables:
`parse_instance_fields`, `build_kernel`, and `emit_sidecar`.

Generic `selection.attribute_constraints` supports `equals`, `not_equals`, and
`one_of`. `attributes_match_constraints()` can match normalized runtime
attributes against those constraints.

The SDPA-specific schema overlay lives at
`kernels/sdpa/fmha_fwd_mfma/schemas/instance.schema.json`. The SDPA handler
normalizes dtype aliases `fp16`, `f16`, and `half` to provider-facing `fp16`
and rocKE-internal `f16`, then validates:

- `canonical_layout == "BSHD"`
- `mask_mode == "none"`
- `seqlen_q` and `seqlen_k` are divisible by 16
- `head_size` is one of `32`, `64`, `128`, `192`, `256`
- `block_size_q == 16`
- `num_query_heads % num_kv_heads == 0`
- JSON `name` matches the canonical artifact basename

Checked-in SDPA files use external dtype spelling `fp16`; sidecar ABI pointer
strings use rocKE signature spelling `ptr<f16, global>`.

## Python-only AOT build flow

The manual builder expects the artifact directory to already exist and contain a
copied `aot_list.json` instance array. From the repository root:

```bash
BUILD_DIR=/path/to/build
ARTIFACT_DIR="${BUILD_DIR}/rocKE-client/aot/gfx1151/sdpa_fwd_fmha_mfma_gfx1151"
KERNEL_DIR=dnn-providers/hip-kernel-provider/rocKE-client/kernels/sdpa/fmha_fwd_mfma

mkdir -p "${ARTIFACT_DIR}"
cp "${KERNEL_DIR}"/gfx1151/aot_list.json "${ARTIFACT_DIR}/"

PYTHONPATH=dnn-providers/hip-kernel-provider/rocKE/Python:\
dnn-providers/hip-kernel-provider/rocKE-client/aot/python \
  python3 dnn-providers/hip-kernel-provider/rocKE-client/aot/tools/rocke_aot_build.py \
    --artifact-dir "${ARTIFACT_DIR}" \
    --kernel-dir "${KERNEL_DIR}" \
    --arch gfx1151
```

For `gfx942`, copy that architecture's list and pass the matching `--arch`:

```bash
ARTIFACT_DIR="${BUILD_DIR}/rocKE-client/aot/gfx942/sdpa_fwd_fmha_mfma_gfx942"
mkdir -p "${ARTIFACT_DIR}"
cp "${KERNEL_DIR}"/gfx942/aot_list.json "${ARTIFACT_DIR}/"
# then rerun rocke_aot_build.py with --arch gfx942
```

`rocke_aot_build.py`:

1. rejects missing `--artifact-dir`, `--kernel-dir`, `--arch`, a missing
   `aot_list.json`, and hipcc-oriented environment overrides
   (`ROCKE_AOT_BACKEND`, `ROCKE_AOT_COMPILE_BACKEND`,
   `ROCKE_COMPILE_BACKEND`, `ROCKE_USE_HIPCC`);
2. removes stale `*.hsaco` and `*.sidecar.json` outputs while preserving the
   copied `aot_list.json` input;
3. prefers kernel-local `schemas/instance.schema.json` and
   `schemas/sidecar.schema.json`, falling back to common AOT schemas;
4. validates every instance in `aot_list.json` against the instance schema and
   enforces that each instance's `arch` matches `--arch` before parsing;
5. compiles through rocKE Python lowering plus direct LLVM/comgr assembly via
   `compile_kernel(..., backend="python", capture_ir_text=False)`;
6. writes `.hsaco` plus `.sidecar.json` named by each instance's `name`;
7. validates generated sidecars before and after writing.

The build path does not use the C++ engine, `rocke_engine`, `hipcc`,
`compile_kernel_via_hipcc()`, kpack packaging, install rules, or provider
dispatcher selection.

## CMake targets

Configure CMake from the `rocKE-client` root. The hip-kernel-provider root does
not currently add this tree as a subdirectory:

```bash
cmake -S dnn-providers/hip-kernel-provider/rocKE-client -B <build>
cmake --build <build> --target rocke_client_aot_artifacts
```

Per-architecture targets are also generated:

```bash
cmake --build <build> --target sdpa_fwd_fmha_mfma_gfx1151
cmake --build <build> --target sdpa_fwd_fmha_mfma_gfx942
```

Aggregate targets:

- `rocke_client_aot_artifacts`: copies all registered checked-in provider AOT
  instances into the build tree, validates their JSON Schemas, and builds loose
  HSACO plus schema-validated metadata sidecars.
- `rocke_client_aot_check`: builds the artifacts, then runs CTest with
  `-L rocKE-client`.

CTest entries are registered when `BUILD_TESTING` or
`HIPKERNELPROVIDER_ENABLE_TESTS` is true:

- `rocke_client_aot_pytest`
- `rocke_client_sdpa_aot_pytest`
- `rocke_client_sdpa_aot_numeric_gfx1151`
- `rocke_client_sdpa_aot_numeric_gfx942`

CMake supplies `PYTHONPATH` for both `rocKE/Python` and
`rocKE-client/aot/python`; developer shell state is not required for CTest. The
default rocKE source tree is the sibling `../rocKE`; override it with
`-DROCKE_CLIENT_ROCKE_SOURCE_DIR=<path-to-rocKE>` when needed.

Kernel family CMake files register per-architecture instance lists through:

```cmake
rocke_client_add_aot_instances(
    NAME <target-name>
    ARCH <gfx-arch>
    ARCH_DIR <per-arch-directory-with-aot_list.json>
    [PYTHON_DEPENDS <extra-python-deps>...]
)
```

`rocke_client_add_aot_instances()` reads the architecture's `aot_list.json`,
derives the generated HSACO/sidecar outputs from each instance `name`, requires a
kernel-local `aot_instance.py`, copies `aot_list.json` into
`${PROJECT_BINARY_DIR}/rocKE-client/aot/${ARCH}/${NAME}`, invokes the AOT build
tool with `--arch ${ARCH}`, creates a custom target named `${NAME}`, and wires it
into `rocke_client_aot_artifacts`. The helper-owned Python path prepends
`rocKE/Python` and `rocKE-client/aot/python`, then preserves any incoming
`PYTHONPATH`.

## Build-tree outputs

Each registered architecture copies its `aot_list.json` into the build tree and
writes the HSACO and sidecar named by each instance beside it:

```text
<build>/rocKE-client/aot/<arch>/<target-name>/
  aot_list.json
  <instance-name>.hsaco
  <instance-name>.sidecar.json
  build.stamp
```

Initial SDPA output directories are:

```text
<build>/rocKE-client/aot/gfx1151/sdpa_fwd_fmha_mfma_gfx1151/
<build>/rocKE-client/aot/gfx942/sdpa_fwd_fmha_mfma_gfx942/
```

## Sidecar metadata

Sidecar files use schema `rocke.aot.sidecar/v1`. The generic envelope schema
lives at `aot/schemas/sidecar.schema.json` and requires top-level `schema`,
`cache_key`, `artifact`, `selection`, `launch`, and `args_signature`. Common
helpers in `rocke_client_aot.sidecar` provide `SIDECAR_SCHEMA`,
`canonical_json_bytes()`, `canonical_hash()`, and `make_sidecar()`.

The SDPA-specific sidecar schema lives at
`kernels/sdpa/fmha_fwd_mfma/schemas/sidecar.schema.json`. For SDPA FMHA MFMA,
`emit_sidecar()` records:

- `cache_key` built from `sdpa_fwd`, `fmha_fwd_mfma`, candidate
  `fmha_fwd_mfma`, algorithm `dense_fmha_fwd`, spec id such as
  `fp16_bshd_blockq16_blockk64`, `arch`, ABI version
  `hipkg-sdpa-fwd-fmha-mfma/v1`, request hash, and spec hash.
- `artifact.hsaco_filename`, `artifact.symbol`, `artifact.hsaco_sha256`, and
  `artifact.hsaco_size`.
- `selection.op`, `selection.arch`, dtypes `q/k/v/o = fp16`, accumulator dtype
  `fp32`, `canonical_layout`, shape constraints, and runtime attribute
  constraints.
- `launch.shared_mem_bytes`, grid formula
  `x = ceil_div(seqlen_q, block_size_q)`, `y = num_query_heads`, `z = batch`,
  block `[wave_size, 1, 1]`, and tile sizes.
- `args_signature` enriched from `fmha_fwd_mfma_signature`.

For the current smoke instances, `launch.block` is `[32, 1, 1]` on `gfx1151`
and `[64, 1, 1]` on `gfx942`. The ABI signature starts with `Q`, `K`, `V`, `O`;
tensor pointer entries use `ptr<f16, global>` with `kind = "pointer"`,
`size_bytes = 8`, and `alignment = 8`. Supported scalar entries use 4-byte size
and alignment.

These are loose build-tree files only.

## Tests

Provider-local tests cover:

- common instance parsing and `selection.attribute_constraints`;
- JSON Schema validation for common and SDPA-specific instance/sidecar shapes;
- deterministic SDPA artifact naming;
- dtype normalization;
- sidecar cache keys, artifact metadata, launch metadata, and ABI signatures;
- direct Python/comgr lowering through the build CLI;
- CMake registration and copied build-tree outputs;
- GPU numeric execution from the copied `aot_list.json`, the matching
  `<instance-name>.sidecar.json`, and the HSACO named by
  `artifact.hsaco_filename`.

Numeric tests skip rather than fail when no matching HIP device is visible.

## kpack bundling

kpack archive creation and binary embedding are separate work. This flow
intentionally stops at the copied `aot_list.json`, `.hsaco`, and `.sidecar.json`
files so provider selection and packaging can be added independently.
