# rocKE Client AOT Kernels

`rocKE-client` contains provider-owned rocKE client kernel instances and tooling. The rocKE platform under `../rocKE/` is consumed as an imported Python package; AOT artifacts remain loose files in the build tree.

## Checked-in SDPA instances

The first AOT family is SDPA forward `fmha_fwd_mfma`. Concrete instances are checked in per architecture:

```text
dnn-providers/hip-kernel-provider/rocKE-client/kernels/sdpa/fmha_fwd_mfma/
  CMakeLists.txt
  instances/
    gfx1151/*.instance.json
    gfx942/*.instance.json
  tests/
    test_sdpa_aot_recipe.py
    sdpa_aot_numeric.py
```

Instance files use schema `rocke.aot.instance/v1`: top-level `schema`, `name`, `op`, `family`, `arch`, `compile_spec`, `selection`, and `test_profiles` are common across AOT operations. The generic JSON Schema lives at `aot/schemas/instance.schema.json`; SDPA FMHA MFMA narrows it with `kernels/sdpa/fmha_fwd_mfma/schemas/instance.schema.json`. The values inside `compile_spec`, `selection`, and `test_profiles` are operation-specific, except `selection.attribute_constraints`, which the common AOT parser validates and can match against runtime attributes. Checked-in SDPA files use external dtype spelling `fp16`; the SDPA build-side parser accepts aliases, but rocKE specs use internal `f16` and sidecar ABI pointer strings use `ptr<f16, global>`.

## Python-only SDPA AOT flow

Run the builder with both rocKE Python package roots on `PYTHONPATH` after copying the checked-in instance files into the artifact directory:

```bash
PYTHONPATH=dnn-providers/hip-kernel-provider/rocKE/Python:\
dnn-providers/hip-kernel-provider/rocKE-client/aot/python \
  python3 dnn-providers/hip-kernel-provider/rocKE-client/aot/tools/rocke_aot_build.py \
    --artifact-dir <build>/rocKE-client/aot/gfx1151/sdpa_fwd_fmha_mfma_gfx1151 \
    --kernel-dir dnn-providers/hip-kernel-provider/rocKE-client/kernels/sdpa/fmha_fwd_mfma
```

The build command reads `*.instance.json` from `--artifact-dir`, validates every instance against the kernel's JSON Schema before parsing, loads the operation-specific `aot_instance.py` from the registered kernel directory, uses rocKE's Python lowering plus direct LLVM/comgr assembly via `compile_kernel(..., backend="python", capture_ir_text=False)`, writes loose HSACO plus sidecar files beside the copied instances, and validates each generated sidecar JSON before and after writing it. It does not use the C++ engine, `hipcc`, kpack packaging, install rules, or provider dispatcher selection.

Use the matching `gfx942` artifact directory for the CDNA smoke instance.

## CMake targets

Configure CMake from the `rocKE-client` root; the hip-kernel-provider root does not add this tree as a subdirectory:

```bash
cmake -S dnn-providers/hip-kernel-provider/rocKE-client -B <build>
cmake --build <build> --target rocke_client_aot_artifacts
```

The aggregate targets are:

- `rocke_client_aot_artifacts`: copies all registered checked-in provider AOT instances into the build tree, validates their JSON Schemas, and builds their loose HSACO plus schema-validated metadata sidecars.
- `rocke_client_aot_check`: builds the artifacts and runs rocKE-client CTest coverage, including numeric tests that skip with return code 77 when the visible HIP device does not match the requested architecture.

CMake supplies `PYTHONPATH` for both `rocKE/Python` and `rocKE-client/aot/python`; developer shell state is not required for CTest. The default rocKE source tree is the sibling `../rocKE`; override it with `-DROCKE_CLIENT_ROCKE_SOURCE_DIR=<path-to-rocKE>` when needed. Family CMake files live under `rocKE-client/kernels/...` and register the source instance files for each architecture. The CMake helper copies those checked-in instances into `${PROJECT_BINARY_DIR}/rocKE-client/aot/${ARCH}/${NAME}` before invoking the AOT build tool.

## Build-tree outputs

Each registered architecture writes artifacts beside the copied normalized instance that produced them:

```text
<build>/rocKE-client/aot/<arch>/<target-name>/
  <artifact>.instance.json
  <artifact>.hsaco
  <artifact>.sidecar.json
  build.stamp
```

For the initial SDPA FMHA MFMA smoke instances, the artifact basename includes op, family, dtype, layout, arch, query/KV sequence lengths, query/KV head counts, head size, and mask mode, for example:

```text
sdpa_fwd_fmha_fwd_mfma_fp16_bshd_gfx1151_q64_k64_hq4_hkv4_d64_none
```

Sidecar files use schema `rocke.aot.sidecar/v1`: top-level `schema`, `cache_key`, `artifact`, `selection`, `launch`, and `args_signature` are common across AOT operations. The generic JSON Schema lives at `aot/schemas/sidecar.schema.json`; SDPA FMHA MFMA narrows it with `kernels/sdpa/fmha_fwd_mfma/schemas/sidecar.schema.json`. The specific field names and values inside those entries are operation-specific: SDPA records a FMHA MFMA cache key, `artifact` fields for the loose HSACO, `selection` predicates for graph/runtime matching, `launch` metadata, and the kernel ABI signature. These are loose build-tree files only.

## Tests

Provider-local tests cover instance parsing, deterministic artifact naming, dtype normalization, sidecar metadata, direct Python/comgr lowering, CMake registration, copied build-tree outputs, and GPU numeric execution. GPU numeric tests skip rather than fail when no matching HIP device is visible.

## kpack bundling

kpack archive creation and binary embedding are separate work. This flow intentionally stops at copied `.instance.json`, `.hsaco`, and `.sidecar.json` files so provider selection and packaging can be added independently.
