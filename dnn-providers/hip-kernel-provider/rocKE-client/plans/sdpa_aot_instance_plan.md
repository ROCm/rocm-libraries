# rocKE SDPA AOT Instance Plan

## Goal

Add a repeatable, build-time AOT flow for checked-in rocKE SDPA forward instance files from the hip-kernel-provider rocKE client area. The provider client consumes the rocKE platform under `rocKE/` through its Python package, but checked-in client API files, AOT tooling, and kernel instance definitions live outside `rocKE/`. The output is a loose HSACO plus metadata sidecar in the build tree. kpack bundling, install rules, and dispatcher selection are separate future work.

## Requirements

- First example is SDPA forward, not GEMM.
- First supported op and family are `sdpa_fwd` and `fmha_fwd_mfma`.
- Use `dnn-providers/hip-kernel-provider/rocKE/Python/rocke` as an imported platform dependency only; do not add provider client files under `rocKE/`.
- Do not use `rocke_core`, `rocke_engine`, C++ lowering, C++ metadata APIs, or C ABI producers.
- Do not depend on `ROCKE_BACKEND=cpp`, `ROCKE_BACKEND=both`, or `ROCKE_CPP_STRICT`.
- Do not use `hipcc` or `compile_kernel_via_hipcc()`; use direct LLVM IR lowering plus `libamd_comgr` assembly through `compile_kernel(..., backend="python", capture_ir_text=False)`.
- Start with one checked-in SDPA instance for `gfx1151` and one for `gfx942`. Additional architectures are added by checking in another per-arch instance directory and registering it in the family CMake file.
- Do not install build artifacts.
- Do not embed HSACO binaries in provider sources.
- Do not create a kpack archive in this task.
- Keep dispatcher selection out of scope; this is one checked-in smoke kernel per initial architecture.
- Keep metadata generation off lowering paths so emitted kernel code is unchanged except for the normal Python build output.

## Provider Client Layout

Create the provider-owned client area under the hip-kernel-provider root:

```text
dnn-providers/hip-kernel-provider/rocKE-client/
  aot/
    CMakeLists.txt
    cmake/
    python/rocke_client_aot/
    tools/
  kernels/
    sdpa/
      fmha_fwd_mfma/
        CMakeLists.txt
        instances/
          gfx1151/
            sdpa_fwd_fmha_fwd_mfma_fp16_bshd_gfx1151_q64_k64_hq4_hkv4_d64_none.instance.json
          gfx942/
            sdpa_fwd_fmha_fwd_mfma_fp16_bshd_gfx942_q64_k64_hq4_hkv4_d64_none.instance.json
        tests/
          sdpa_aot_numeric.py
```

`rocKE-client` is the provider-side rocKE client API area. It may later grow `include/` and `src/` API layers, but this AOT work only needs `aot/` and `kernels/`.

Keep the split strict:

- `rocKE/` contains the rocKE platform and platform tests/docs.
- `rocKE-client/aot/` contains AOT instance parsing, sidecar emission, build CLIs, CMake helpers, and the aggregate AOT CMake entry point.
- `rocKE-client/kernels/<op-kind>/<family>/` contains checked-in concrete instance declarations, the family CMake file, and family-local tests.
- Do not create `rocKE/aot`, `rocKE/tools/rocke_aot_build.py`, `rocKE/cmake/rocke_aot.cmake`, or `rocKE/Python/rocke/aot` for this flow.

## Python Import Path Policy

Do not add relative path discovery or `sys.path` mutation to `rocKE-client` Python modules. Treat `rocKE/Python` and `rocKE-client/aot/python` as local source packages supplied by the build environment:

- In CMake build rules, run Python through `${CMAKE_COMMAND} -E env`.
- Set `PYTHONPATH` to both source package roots:
  1. `${CMAKE_CURRENT_SOURCE_DIR}/rocKE/Python`
  2. `${CMAKE_CURRENT_SOURCE_DIR}/rocKE-client/aot/python`
- Use `cmake_path(CONVERT ... TO_NATIVE_PATH_LIST ...)` so the path separator is correct on Linux and Windows.
- Preserve any incoming `PYTHONPATH` by appending it after the local package roots when it is non-empty.
- For CTest-only invocations, set the same value with the test `ENVIRONMENT` or `ENVIRONMENT_MODIFICATION` property instead of relying on the developer shell.

Provider Python code should simply import `rocke` and `rocke_client_aot`.

## First SDPA Instance Family

Use dense SDPA/FMHA forward:

- Builder: `rocKE/Python/rocke/instances/common/fmha_mfma.py`
- Spec types: `FmhaMfmaSpec`, `FmhaCommonSpec`, `FmhaShape`
- Build helper: `build_fmha_fwd_mfma`
- Launch helper: `fmha_fwd_mfma_grid`
- Signature helper: `fmha_fwd_mfma_signature`
- Existing numeric reference: `rocKE/Python/rocke/examples/common/fmha_fwd_verify_hip.py`

Use `fmha_mfma` instead of `attention_unified` for the first example because it is a single dense SDPA kernel with deterministic grid and signature. `attention_unified` adds paged-KV layout, 2D/3D path selection, optional split/reduce kernels, workspace, block tables, and dynamic launch behavior that belongs in a later dispatcher task.

## Checked-in AOT Instance Files

Check in one normalized JSON file per concrete AOT kernel instance. These files are the source of truth for supported/tuned coverage. The CMake build copies the selected checked-in instance files into the build-tree artifact directory before compiling them, so a failing artifact always has the exact input instance beside its HSACO and sidecar.

Add:

```text
dnn-providers/hip-kernel-provider/rocKE-client/kernels/sdpa/fmha_fwd_mfma/CMakeLists.txt
dnn-providers/hip-kernel-provider/rocKE-client/kernels/sdpa/fmha_fwd_mfma/instances/gfx1151/sdpa_fwd_fmha_fwd_mfma_fp16_bshd_gfx1151_q64_k64_hq4_hkv4_d64_none.instance.json
dnn-providers/hip-kernel-provider/rocKE-client/kernels/sdpa/fmha_fwd_mfma/instances/gfx942/sdpa_fwd_fmha_fwd_mfma_fp16_bshd_gfx942_q64_k64_hq4_hkv4_d64_none.instance.json
dnn-providers/hip-kernel-provider/rocKE-client/kernels/sdpa/fmha_fwd_mfma/tests/sdpa_aot_numeric.py
```

Each `.instance.json` contains one concrete AOT kernel and uses the normalized checked-in schema `ck.rocke.aot.instance/v1`:

```json
{
  "schema": "ck.rocke.aot.instance/v1",
  "name": "sdpa_fwd_fmha_fwd_mfma_fp16_bshd_gfx1151_q64_k64_hq4_hkv4_d64_none",
  "op": "sdpa_fwd",
  "family": "fmha_fwd_mfma",
  "arch": "gfx1151",
  "compile_spec": {
    "dtype": "fp16",
    "canonical_layout": "BSHD",
    "seqlen_q": 64,
    "seqlen_k": 64,
    "num_query_heads": 4,
    "num_kv_heads": 4,
    "head_size": 64,
    "block_size_q": 16,
    "block_size_k": 64,
    "mask_mode": "none"
  },
  "selection": {
    "batch": {"min": 1, "max": 64},
    "attributes": {
      "mask_mode": {"equals": "none"},
      "dropout_probability": {"equals": 0.0},
      "scale_policy": {"equals": "default_1_over_sqrt_d"},
      "padding_mask": {"equals": false},
      "alibi_mask": {"equals": false}
    }
  },
  "test_profiles": [
    {"batch": 2}
  ]
}
```

The `gfx942` instance uses the same smoke shape initially, with `arch` and `name` changed to `gfx942`.

Artifact names must be deterministic and include op, family, external dtype, layout, arch, query/KV sequence lengths, query/KV head counts, head size, and mask mode:

```text
{op}_{family}_{dtype}_{layout}_{arch}_q{seqlen_q}_k{seqlen_k}_hq{num_query_heads}_hkv{num_kv_heads}_d{head_size}_{mask_mode}
```

`gfx1151` viability check:

- `rocKE/Python/rocke/instances/SUPPORT_MATRIX.md` marks `fmha_fwd_mfma` as supported on `gfx1151` and records GPU-numeric PASS for `fmha_fwd_mfma` and the native `gfx1151/wmma_fmha_fwd` path.
- Local validation for the smoke shape showed `is_valid_spec(spec, "gfx1151") == (True, "ok")`.
- Local LLVM lowering for `gfx1151` emitted a WMMA kernel (`llvm.amdgcn.wmma` present, `llvm.amdgcn.mfma` absent).
- Local `compile_kernel(..., arch="gfx1151", backend="python", capture_ir_text=False)` produced a non-empty `amdgcn-amd-amdhsa--gfx1151` HSACO for `rocke_fmha_fwd_mfma_H64_HQ4_HK4_f16_Q64_K64_none`.
- The local machine did not expose a HIP runtime device, so the implementation task must still run the checked-in-instance numeric verifier on a real `gfx1151` system before declaring GPU acceptance.

Identity policy:

- `compile_spec` contains only fields that select or affect emitted kernel code: dtype, layout, head sizes, query/KV head counts, block sizes, mask mode, and static sequence lengths while `FmhaMfmaSpec` keeps `seqlen_q` / `seqlen_k` compile-time.
- Top-level `arch`, `op`, and `family` identify the architecture and builder path.
- `selection` contains graph/runtime coverage that should not create new source files, such as accepted `batch` ranges and normalized SDPA attribute predicates.
- `test_profiles` contains concrete runtime examples used by smoke and numeric tests.
- Tensor UIDs are not instance identity. They are graph-binding data supplied by tests or the future dispatcher.

Dtype policy:

- Checked-in instance files use external dtype spelling `fp16`.
- The instance parser may accept aliases such as `f16` and `half` for compatibility, but it normalizes to rocKE internal `f16` for `FmhaCommonSpec` and IR-facing helpers.
- Sidecar emits external dtype fields as `fp16`.
- Sidecar emits kernel ABI type strings as `ptr<f16, global>` because that is the actual rocKE signature spelling.

## Python Instance Parser

Add:

```text
dnn-providers/hip-kernel-provider/rocKE-client/aot/python/rocke_client_aot/__init__.py
dnn-providers/hip-kernel-provider/rocKE-client/aot/python/rocke_client_aot/instance_schema.py
```

Instance parser responsibilities:

1. Load one checked-in or copied `.instance.json`.
2. Validate `schema == "ck.rocke.aot.instance/v1"`.
3. Require `op == "sdpa_fwd"` and `family == "fmha_fwd_mfma"`.
4. Validate that the top-level `arch` matches the requested build architecture.
5. Validate that `name` matches the deterministic artifact basename for the compile spec.
6. Validate shape requirements:
   - `seqlen_q % 16 == 0`
   - `seqlen_k % 16 == 0`
   - `head_size in {32, 64, 128, 192, 256}`
   - `num_query_heads % num_kv_heads == 0`
   - `canonical_layout == "BSHD"` initially
   - `mask_mode == "none"` initially
7. Build:

```python
from rocke.instances import FmhaCommonSpec, FmhaShape
from rocke.instances.common.fmha_mfma import FmhaMfmaSpec, is_valid_spec

common = FmhaCommonSpec(
    FmhaShape(
        head_size=head_size,
        num_query_heads=num_query_heads,
        num_kv_heads=num_kv_heads,
        block_size_q=block_size_q,
        block_size_k=block_size_k,
    ),
    dtype="f16",
    mask_mode=mask_mode,
)

spec = FmhaMfmaSpec(
    common=common,
    seqlen_q=seqlen_q,
    seqlen_k=seqlen_k,
)
```

8. Call `is_valid_spec(spec, arch)` and surface its failure reason.

## Python AOT Build CLI

Add:

```text
dnn-providers/hip-kernel-provider/rocKE-client/aot/tools/rocke_aot_build.py
```

Build CLI:

```bash
PYTHONPATH=dnn-providers/hip-kernel-provider/rocKE/Python:\
dnn-providers/hip-kernel-provider/rocKE-client/aot/python \
  python3 dnn-providers/hip-kernel-provider/rocKE-client/aot/tools/rocke_aot_build.py \
  --artifact-dir <build>/rocKE-client/aot/gfx1151/sdpa_fwd_fmha_mfma_gfx1151
```

Implementation steps:

1. Parse each `.instance.json` from `--artifact-dir`. CMake is responsible for copying checked-in instance files into that directory before invoking the build CLI.
2. For each instance file, build `FmhaMfmaSpec`.
3. Build the kernel:

```python
from rocke.instances.common.fmha_mfma import build_fmha_fwd_mfma

kernel = build_fmha_fwd_mfma(spec, arch=arch)
```

4. Compile with the direct LLVM IR -> `libamd_comgr` path:

```python
from rocke.helpers import compile_kernel

artifact = compile_kernel(
    kernel,
    arch=arch,
    backend="python",
    capture_ir_text=False,
)
```

Use `compile_kernel(..., backend="python", capture_ir_text=False)` intentionally. It lowers to AMDGPU LLVM IR with the Python lowerer and assembles HSACO through `libamd_comgr`; it does not consult `ROCKE_BACKEND`, does not use `hipcc`, and does not require `rocke_engine`.

5. Write loose build-tree outputs only:

```text
<output-dir>/sdpa_fwd_fmha_fwd_mfma_fp16_bshd_gfx1151_q64_k64_hq4_hkv4_d64_none.instance.json
<output-dir>/sdpa_fwd_fmha_fwd_mfma_fp16_bshd_gfx1151_q64_k64_hq4_hkv4_d64_none.hsaco
<output-dir>/sdpa_fwd_fmha_fwd_mfma_fp16_bshd_gfx1151_q64_k64_hq4_hkv4_d64_none.sidecar.json
<output-dir>/build.stamp
```

The build stage may clean stale outputs before rebuilding, but it must preserve `*.instance.json`. Limit cleanup to outputs such as `*.hsaco`, `*.sidecar.json`, and optional debug IR files. Do not write an artifact `index.json` in this task. Each copied `.instance.json` lives beside its `.hsaco` and `.sidecar.json`, so the exact build input for a failing artifact is local to that artifact.

Optional debug output can be added only if an existing helper exposes it cleanly:

```text
<output-dir>/sdpa_fwd_fmha_fwd_mfma_fp16_bshd_gfx1151_q64_k64_hq4_hkv4_d64_none.ll
```

## Sidecar Emitter

Add:

```text
dnn-providers/hip-kernel-provider/rocKE-client/aot/python/rocke_client_aot/sidecar.py
```

Inputs:

- normalized instance;
- `FmhaMfmaSpec`;
- `artifact.kernel_name`;
- `artifact.hsaco`;
- output HSACO filename.

Output schema:

```json
{
  "schema": "ck.rocke.aot.sidecar/v1",
  "kernel_id": {
    "op": "sdpa_fwd",
    "family": "fmha_fwd_mfma",
    "candidate": "fmha_fwd_mfma",
    "algorithm": "dense_fmha_fwd",
    "spec_id": "fp16_bshd_blockq16_blockk64",
    "arch": "gfx1151",
    "abi_version": "hipkg-sdpa-fwd-fmha-mfma/v1",
    "request_hash": "...",
    "spec_hash": "...",
    "cache_key": "sdpa_fwd:fmha_fwd_mfma:..."
  },
  "artifact": {
    "hsaco_filename": "sdpa_fwd_fmha_fwd_mfma_fp16_bshd_gfx1151_q64_k64_hq4_hkv4_d64_none.hsaco",
    "symbol": "<artifact.kernel_name>",
    "hsaco_sha256": "...",
    "hsaco_size": 12345
  },
  "selection": {
    "op": "sdpa_fwd",
    "arch": "gfx1151",
    "dtypes": {
      "q": "fp16",
      "k": "fp16",
      "v": "fp16",
      "o": "fp16",
      "acc": "fp32"
    },
    "canonical_layout": "BSHD",
    "shapes": {
      "batch": {"min": 1, "max": 64},
      "seqlen_q": {"equals": 64, "multiple_of": 16},
      "seqlen_k": {"equals": 64, "multiple_of": 16},
      "num_query_heads": {"equals": 4},
      "num_kv_heads": {"equals": 4},
      "head_size": {"equals": 64}
    },
    "attributes": {
      "mask_mode": {"equals": "none"},
      "dropout_probability": {"equals": 0.0},
      "scale_policy": {"equals": "default_1_over_sqrt_d"},
      "padding_mask": {"equals": false},
      "alibi_mask": {"equals": false}
    }
  },
  "launch": {
    "shared_mem_bytes": 0,
    "grid_formula": {
      "x": {"ceil_div": ["seqlen_q", 16]},
      "y": "num_query_heads",
      "z": "batch"
    },
    "block": [32, 1, 1],
    "tile_sizes": {
      "block_q": 16,
      "block_k": 64,
      "head_size": 64,
      "wave_size": 32
    }
  },
  "args_signature": []
}
```

Computed fields:

- `artifact.hsaco_sha256 = sha256(hsaco bytes)`
- `artifact.hsaco_size = len(hsaco bytes)`
- `launch.grid_formula` describes `fmha_fwd_mfma_grid(spec, batch=batch)` without freezing one batch into the artifact sidecar.
- `launch.block = (ArchTarget.from_gfx(arch).wave_size, 1, 1)`
- `args_signature = enrich(fmha_fwd_mfma_signature(spec))`

Expected argument order:

1. `Q`
2. `K`
3. `V`
4. `O`
5. `scale_log2`
6. `seqlen_q`
7. `seqlen_k`
8. `stride_q_token`
9. `stride_q_head`
10. `stride_k_token`
11. `stride_k_head`
12. `stride_v_token`
13. `stride_v_head`
14. `stride_o_token`
15. `stride_o_head`

Argument sources:

- `Q`, `K`, `V`, `O`: tensor pointers supplied by provider graph binding.
- `scale_log2`: `log2(e) * (attn_scale_value or 1 / sqrt(head_size))`.
- `seqlen_q`, `seqlen_k`: instance shape.
- strides: dense BSHD tensor strides.

For BSHD row-major `[B, S, H, D]`:

- dims: `[batch, seqlen, heads, head_size]`
- strides: `[seqlen * heads * head_size, heads * head_size, head_size, 1]`
- kernel token stride: `strides[1]`
- kernel head stride: `strides[2]`

## CMake Build-Tree Target

Add:

```text
dnn-providers/hip-kernel-provider/rocKE-client/aot/CMakeLists.txt
dnn-providers/hip-kernel-provider/rocKE-client/aot/cmake/rocke_aot.cmake
dnn-providers/hip-kernel-provider/rocKE-client/kernels/sdpa/fmha_fwd_mfma/CMakeLists.txt
```

Provider root wiring:

```cmake
add_subdirectory(rocKE-client/aot)
```

Provider-owned AOT aggregate file:

```cmake
# dnn-providers/hip-kernel-provider/rocKE-client/aot/CMakeLists.txt
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/rocke_aot.cmake")

add_custom_target(rocke_client_aot_artifacts)

add_subdirectory(
  "${CMAKE_CURRENT_SOURCE_DIR}/../kernels/sdpa/fmha_fwd_mfma"
  "${CMAKE_CURRENT_BINARY_DIR}/kernels/sdpa/fmha_fwd_mfma"
)

add_custom_target(rocke_client_aot_check
  COMMAND "${CMAKE_CTEST_COMMAND}" --output-on-failure -L rocKE-client
  DEPENDS rocke_client_aot_artifacts
)
```

`rocKE-client/aot/CMakeLists.txt` is the only place that lists kernel families. Adding a new family means adding one `add_subdirectory()` there; the family directory owns its checked-in instances and tests.

Kernel developer-owned family file:

```cmake
# dnn-providers/hip-kernel-provider/rocKE-client/kernels/sdpa/fmha_fwd_mfma/CMakeLists.txt
rocke_client_add_aot_instances(
  NAME sdpa_fwd_fmha_mfma_gfx1151
  ARCH gfx1151
  INSTANCES
    "${CMAKE_CURRENT_SOURCE_DIR}/instances/gfx1151/sdpa_fwd_fmha_fwd_mfma_fp16_bshd_gfx1151_q64_k64_hq4_hkv4_d64_none.instance.json"
)

rocke_client_add_aot_instances(
  NAME sdpa_fwd_fmha_mfma_gfx942
  ARCH gfx942
  INSTANCES
    "${CMAKE_CURRENT_SOURCE_DIR}/instances/gfx942/sdpa_fwd_fmha_fwd_mfma_fp16_bshd_gfx942_q64_k64_hq4_hkv4_d64_none.instance.json"
)

if(BUILD_TESTING)
  add_test(
    NAME rocke_client_sdpa_aot_numeric_gfx1151
    COMMAND "${Python3_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/sdpa_aot_numeric.py"
            --arch gfx1151
            --artifact-dir "${PROJECT_BINARY_DIR}/rocKE-client/aot/gfx1151/sdpa_fwd_fmha_mfma_gfx1151"
  )
  set_tests_properties(rocke_client_sdpa_aot_numeric_gfx1151 PROPERTIES
    LABELS "gpu;rocKE-client;sdpa"
    REQUIRED_FILES "${PROJECT_BINARY_DIR}/rocKE-client/aot/gfx1151/sdpa_fwd_fmha_mfma_gfx1151/build.stamp"
    SKIP_RETURN_CODE 77
  )

  add_test(
    NAME rocke_client_sdpa_aot_numeric_gfx942
    COMMAND "${Python3_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/sdpa_aot_numeric.py"
            --arch gfx942
            --artifact-dir "${PROJECT_BINARY_DIR}/rocKE-client/aot/gfx942/sdpa_fwd_fmha_mfma_gfx942"
  )
  set_tests_properties(rocke_client_sdpa_aot_numeric_gfx942 PROPERTIES
    LABELS "gpu;rocKE-client;sdpa"
    REQUIRED_FILES "${PROJECT_BINARY_DIR}/rocKE-client/aot/gfx942/sdpa_fwd_fmha_mfma_gfx942/build.stamp"
    SKIP_RETURN_CODE 77
  )
endif()
```

The helper takes explicit instance source paths from the family `CMakeLists.txt`. No call site should pass an output directory; the helper owns the build-tree layout.

CMake helper sketch:

```cmake
find_package(Python3 COMPONENTS Interpreter REQUIRED)

get_filename_component(_ROCKE_CLIENT_ROOT "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)
get_filename_component(_HIP_KERNEL_PROVIDER_ROOT "${_ROCKE_CLIENT_ROOT}/.." ABSOLUTE)

if(NOT TARGET rocke_client_aot_artifacts)
    message(FATAL_ERROR
        "Include rocke_aot.cmake from rocKE-client/aot/CMakeLists.txt after creating rocke_client_aot_artifacts")
endif()

function(rocke_client_add_aot_instances)
    cmake_parse_arguments(ARG "" "NAME;ARCH" "INSTANCES" ${ARGN})
    if(NOT ARG_NAME OR NOT ARG_ARCH OR NOT ARG_INSTANCES)
        message(FATAL_ERROR
            "rocke_client_add_aot_instances requires NAME, ARCH, and INSTANCES")
    endif()

    set(_ROCKE_CLIENT_AOT_PYTHONPATH
        "${_HIP_KERNEL_PROVIDER_ROOT}/rocKE/Python"
        "${_ROCKE_CLIENT_ROOT}/aot/python"
    )
    if(DEFINED ENV{PYTHONPATH} AND NOT "$ENV{PYTHONPATH}" STREQUAL "")
        cmake_path(CONVERT "$ENV{PYTHONPATH}" TO_CMAKE_PATH_LIST
                   _ROCKE_CLIENT_AOT_INCOMING_PYTHONPATH)
        list(APPEND _ROCKE_CLIENT_AOT_PYTHONPATH
             ${_ROCKE_CLIENT_AOT_INCOMING_PYTHONPATH})
    endif()
    cmake_path(CONVERT "${_ROCKE_CLIENT_AOT_PYTHONPATH}" TO_NATIVE_PATH_LIST
               _ROCKE_CLIENT_AOT_PYTHONPATH_NATIVE)

    set(_ROCKE_CLIENT_AOT_OUTPUT_ROOT "${PROJECT_BINARY_DIR}/rocKE-client/aot")
    set(_ROCKE_CLIENT_AOT_ARCH_OUTPUT_DIR
        "${_ROCKE_CLIENT_AOT_OUTPUT_ROOT}/${ARG_ARCH}/${ARG_NAME}")
    set(_ROCKE_CLIENT_AOT_BUILD_STAMP
        "${_ROCKE_CLIENT_AOT_ARCH_OUTPUT_DIR}/build.stamp")

    set(_ROCKE_CLIENT_AOT_COPIED_INSTANCES)
    foreach(_instance IN LISTS ARG_INSTANCES)
        get_filename_component(_instance_name "${_instance}" NAME)
        list(APPEND _ROCKE_CLIENT_AOT_COPIED_INSTANCES
             "${_ROCKE_CLIENT_AOT_ARCH_OUTPUT_DIR}/${_instance_name}")
    endforeach()

    add_custom_command(
        OUTPUT ${_ROCKE_CLIENT_AOT_COPIED_INSTANCES}
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${_ROCKE_CLIENT_AOT_ARCH_OUTPUT_DIR}"
        COMMAND "${CMAKE_COMMAND}" -E copy_if_different
                ${ARG_INSTANCES}
                "${_ROCKE_CLIENT_AOT_ARCH_OUTPUT_DIR}"
        DEPENDS ${ARG_INSTANCES}
        VERBATIM
    )

    add_custom_command(
        OUTPUT "${_ROCKE_CLIENT_AOT_BUILD_STAMP}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${_ROCKE_CLIENT_AOT_ARCH_OUTPUT_DIR}"
        COMMAND "${CMAKE_COMMAND}" -E env
                "PYTHONPATH=${_ROCKE_CLIENT_AOT_PYTHONPATH_NATIVE}"
                "PYTHONDONTWRITEBYTECODE=1"
                "${Python3_EXECUTABLE}"
                "${_ROCKE_CLIENT_ROOT}/aot/tools/rocke_aot_build.py"
                --artifact-dir "${_ROCKE_CLIENT_AOT_ARCH_OUTPUT_DIR}"
                --arch "${ARG_ARCH}"
        COMMAND "${CMAKE_COMMAND}" -E touch "${_ROCKE_CLIENT_AOT_BUILD_STAMP}"
        DEPENDS ${_ROCKE_CLIENT_AOT_COPIED_INSTANCES}
                "${_ROCKE_CLIENT_ROOT}/aot/tools/rocke_aot_build.py"
        VERBATIM
    )

    add_custom_target("${ARG_NAME}" DEPENDS "${_ROCKE_CLIENT_AOT_BUILD_STAMP}")
    add_dependencies(rocke_client_aot_artifacts "${ARG_NAME}")
endfunction()
```

Outputs:

```text
${PROJECT_BINARY_DIR}/rocKE-client/aot/${ARCH}/${NAME}/*.instance.json
${PROJECT_BINARY_DIR}/rocKE-client/aot/${ARCH}/${NAME}/*.hsaco
${PROJECT_BINARY_DIR}/rocKE-client/aot/${ARCH}/${NAME}/*.sidecar.json
${PROJECT_BINARY_DIR}/rocKE-client/aot/${ARCH}/${NAME}/build.stamp
```

Aggregate targets owned by `rocKE-client/aot/CMakeLists.txt`:

```text
rocke_client_aot_artifacts  # copies and builds all registered kernel-family AOT artifacts
rocke_client_aot_check      # builds artifacts, then runs rocKE-client CTest coverage
```

Dependencies:

- checked-in instance JSON files;
- `rocKE-client/aot/tools/rocke_aot_build.py`;
- provider AOT Python modules under `rocKE-client/aot/python/rocke_client_aot`;
- Python SDPA builder modules.

No `install()` calls. No provider install path. No kpack archive. No binary embedding. No `rocke_core` or `rocke_engine` dependency.

## Tests

### Python unit tests

Cover:

- valid SDPA instance parsing for `gfx1151` and `gfx942`;
- deterministic artifact basename validation;
- dtype alias normalization in the parser, while checked-in files use `fp16`;
- unsupported instance schema rejected;
- unsupported op/family rejected;
- mismatched requested architecture rejected;
- invalid shape rejected during instance parsing;
- build CLI always uses direct LLVM/comgr lowering and rejects any request to override the compile path with `hipcc`;
- sidecar required fields present;
- evaluating `launch.grid_formula` with `batch == 2` yields `[4, 4, 2]`;
- `launch.block == [32, 1, 1]` for `gfx1151`;
- `launch.block == [64, 1, 1]` for `gfx942`;
- `selection.dtypes.q == "fp16"`;
- first four args are `Q`, `K`, `V`, `O`;
- pointer arg ABI strings are `ptr<f16, global>`;
- scalar arg sizes and alignments are 4 bytes;
- HSACO SHA and size fields match the emitted bytes.

### CMake smoke

Build all registered AOT kernel families:

```bash
cmake --build <build> --target rocke_client_aot_artifacts
```

Build and run provider-local AOT tests:

```bash
cmake --build <build> --target rocke_client_aot_check
```

Expected behavior:

- checked-in `.instance.json` files are copied under `${ARCH}/${NAME}/`;
- `build.stamp` exists under `${ARCH}/${NAME}/`;
- copied `.instance.json` files live beside `.hsaco` and `.sidecar.json` outputs;
- `.hsaco` files exist and are non-empty;
- `.sidecar.json` files exist;
- provider-local `rocke_client_sdpa_aot_numeric_gfx1151` and `rocke_client_sdpa_aot_numeric_gfx942` CTest entries are registered;
- sidecar SHA fields match `.hsaco` files;
- no install tree is expected.

### GPU numeric acceptance

Add a provider-local verifier:

```text
dnn-providers/hip-kernel-provider/rocKE-client/kernels/sdpa/fmha_fwd_mfma/tests/sdpa_aot_numeric.py
```

The verifier may duplicate functionality from `rocKE/Python/rocke/examples/common/fmha_fwd_verify_hip.py`. This is intentional because the provider-side acceptance harness may later move or be deleted.

Verifier behavior:

1. Accept `--arch` and `--artifact-dir`.
2. Detect the local HIP device arch with `rocke.runtime.hip_module.get_device_arch()`.
3. Exit `77` when no HIP device is available or when the local device arch does not match `--arch`; the CTest entries use `SKIP_RETURN_CODE 77`.
4. Discover copied `.instance.json` files in `--artifact-dir`.
5. For each `.instance.json` with at least one `test_profiles[]` entry:
   - load the matching `.sidecar.json`;
   - load the `.hsaco` named by the sidecar;
   - allocate Q, K, V, O, and any required workspace buffers;
   - derive launch grid from `launch.grid_formula` and the concrete `test_profiles[]` values;
   - pack kernel args from `args_signature`;
   - run the AOT kernel through `rocke.runtime.launcher` / HIP module utilities;
   - compare output against an in-script dense SDPA fp32 reference.

Pass condition:

- same tolerance policy as `Python/rocke/examples/common/fmha_fwd_verify_hip.py`;
- `gfx1151` and `gfx942` tests pass on matching hardware;
- tests skip, not fail, on machines without matching hardware.

## Docs

Update only docs that live under `dnn-providers/hip-kernel-provider/rocKE-client/` if such docs are added with the implementation: Python-only AOT command, CMake target, checked-in instance location, loose output files, and explicit note that kpack bundling is a separate task.

Do not update docs in the core `rocKE/` project for this task.

Do not document provider install locations in `rocKE/`; keep rocKE platform docs separate from provider client docs.

## Implementation Order

1. Add checked-in instance JSON files under `rocKE-client/kernels/sdpa/fmha_fwd_mfma/instances/<arch>/` and the family `CMakeLists.txt`.
2. Add instance parser support for checked-in `ck.rocke.aot.instance/v1` files.
3. Add sidecar emitter.
4. Add Python AOT build CLI for artifact directories containing copied instance files.
5. Add CMake build-tree helper that copies checked-in instance files into `${PROJECT_BINARY_DIR}/rocKE-client/aot/${ARCH}/${NAME}` before building.
6. Add parser/build/sidecar tests.
7. Add CMake smoke test.
8. Add provider-local copied-artifact numeric verifier at `rocKE-client/kernels/sdpa/fmha_fwd_mfma/tests/sdpa_aot_numeric.py`.
9. Update docs.
