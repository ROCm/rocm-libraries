# rocKE SDPA AOT Recipe Plan

## Goal

Add a repeatable, build-time AOT recipe for one rocKE SDPA forward kernel from the hip-kernel-provider rocKE client area. The recipe consumes the rocKE platform under `rocKE/` through its Python package, but checked-in client API files, AOT tooling, and instances live outside `rocKE/`. The output is a loose HSACO plus metadata sidecar in the build tree. A later task will bundle HSACO files into a kpack archive.

## Branch refresh notes

The source branch was rebased/merged after the original plan. The relevant changes are:

- The Python package is now `rocke`, rooted at `rocKE/Python/rocke`. Old `rocKE/Python/ck_dsl` paths and `ck_dsl.*` imports are stale.
- The C++ engine naming moved from `ckc_*` to `rocke_*`: `librocke_core.a`, `rocke_engine`, `ROCKE_BACKEND`, and `ROCKE_CPP_STRICT`.
- The platform SDPA builder still lives in `rocKE/Python/rocke/instances/common/fmha_mfma.py`, but provider instance recipe declarations now live under `rocKE-client/instances/<kernel-type>/<family>/`.
- `compile_kernel()` now defaults through `ROCKE_BACKEND` and the package default is the C++ engine. This plan must not rely on that default. The SDPA AOT CLI must use the LLVM IR -> `libamd_comgr` path with `compile_kernel(..., backend="python")`, so lowering stays Python-owned and does not use `hipcc`, `rocke_engine`, or C++ metadata APIs.
- CMake helper/target names should use `rocke_` names, not `ckc_` names.

- The architectural boundary changed: `rocKE/` is the platform tree only. Provider-owned client API files, AOT tools, and checked-in instance recipes live under `dnn-providers/hip-kernel-provider/rocKE-client/`.

## Constraints

- First example is SDPA forward, not GEMM.
- Use `dnn-providers/hip-kernel-provider/rocKE/Python/rocke` as an imported platform dependency only; do not add provider client files under `rocKE/`.
- Do not use `rocke_core`, `rocke_engine`, C++ lowering, C++ metadata APIs, or C ABI producers.
- Do not depend on `ROCKE_BACKEND=cpp`, `ROCKE_BACKEND=both`, or `ROCKE_CPP_STRICT`.
- Do not use `hipcc` or `compile_kernel_via_hipcc()` for this recipe; use direct LLVM IR lowering plus `libamd_comgr` assembly through `compile_kernel(..., backend="python")`.
- Start with one generated SDPA kernel for `gfx1151` and one for `gfx942`. `gfx950` can be added later by adding another arch recipe file.
- Do not install generated artifacts.
- Do not embed HSACO binaries in provider sources.
- Do not create a kpack archive in this task.
- Keep dispatcher selection out of scope; this is one generated smoke kernel per initial architecture.
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
  instances/
    sdpa/
      fmha_fwd_mfma/
        CMakeLists.txt
        recipes/
          recipe.json
          gfx1151.json
          gfx942.json
        tests/
          sdpa_aot_numeric.py
```

Generated expanded instances are build-tree only; do not check them in.

`rocKE-client` is the provider-side rocKE client API area. It may later grow `include/` and `src/` API layers, but this AOT recipe only needs `aot/` and `instances/`.

Keep the split strict:

- `rocKE/` contains the rocKE platform and platform tests/docs.
- `rocKE-client/aot/` contains AOT recipe parsers, expanders, emitters, CLIs, CMake helpers, and the aggregate AOT CMake entry point.
- `rocKE-client/instances/<kernel-type>/<family>/` contains checked-in recipe declarations and family-local tests. Build-tree expansion produces normalized concrete instances.
- Do not create `rocKE/aot`, `rocKE/tools/rocke_aot_build.py`, `rocKE/cmake/rocke_aot.cmake`, or `rocKE/Python/rocke/aot` for this recipe.


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

## First SDPA Recipe

Use dense SDPA/FMHA forward:

- Builder: `rocKE/Python/rocke/instances/common/fmha_mfma.py`
- Spec types: `FmhaMfmaSpec`, `FmhaCommonSpec`, `FmhaShape`
- Build helper: `build_fmha_fwd_mfma`
- Launch helper: `fmha_fwd_mfma_grid`
- Signature helper: `fmha_fwd_mfma_signature`
- Existing numeric reference: `rocKE/Python/rocke/examples/common/fmha_fwd_verify_hip.py`

Use `fmha_mfma` instead of `attention_unified` for the first example because it is a single dense SDPA kernel with deterministic grid and signature. `attention_unified` adds paged-KV layout, 2D/3D path selection, optional split/reduce kernels, workspace, block tables, and dynamic launch behavior that belongs in a later dispatcher task.

## AOT Instance Recipe Files

Do not check in one JSON file per concrete kernel instance. A single kernel family can have hundreds of valid instances per architecture, and each architecture needs its own supported/tuned coverage set. Source JSON must therefore describe a recipe plus bounded expansion rules; concrete instances are generated into the build tree.

Add:

```text
dnn-providers/hip-kernel-provider/rocKE-client/instances/sdpa/fmha_fwd_mfma/CMakeLists.txt
dnn-providers/hip-kernel-provider/rocKE-client/instances/sdpa/fmha_fwd_mfma/recipes/recipe.json
dnn-providers/hip-kernel-provider/rocKE-client/instances/sdpa/fmha_fwd_mfma/recipes/gfx1151.json
dnn-providers/hip-kernel-provider/rocKE-client/instances/sdpa/fmha_fwd_mfma/recipes/gfx942.json
dnn-providers/hip-kernel-provider/rocKE-client/instances/sdpa/fmha_fwd_mfma/tests/sdpa_aot_numeric.py
```

`recipe.json` contains common family defaults:

```json
{
  "schema": "ck.rocke.aot.recipe/v1",
  "op": "sdpa_fwd",
  "family": "fmha_fwd_mfma",
  "backend": "llvm_ir_comgr",
  "algorithm": "dense_fmha_fwd",
  "abi_version": "hipkg-sdpa-fwd-fmha-mfma/v1",
  "defaults": {
    "dtype": "fp16",
    "canonical_layout": "BSHD",
    "mask_mode": "none",
    "block_size_q": 16,
    "block_size_k": 64,
    "selection": {
      "batch": {"min": 1, "max": 64}
    },
    "hipdnn_node": {
      "type": "SdpaAttributes",
      "compute_data_type": "FLOAT",
      "attn_scale_source": "default_1_over_sqrt_d",
      "dropout_probability": 0.0,
      "causal_mask": false,
      "padding_mask": false,
      "alibi_mask": false
    }
  },
  "artifact_name_template": "sdpa_fwd_{dtype}_{layout}_{arch}_q{seqlen_q}_k{seqlen_k}_hq{num_query_heads}_hkv{num_kv_heads}_d{head_size}_{mask_mode}"
}
```

`gfx1151.json` contains the first RDNA coverage set:

```json
{
  "schema": "ck.rocke.aot.arch_recipe/v1",
  "arch": "gfx1151",
  "artifacts": [
    {
      "id": "smoke",
      "instances": [
        {
          "seqlen_q": 64,
          "seqlen_k": 64,
          "num_query_heads": 4,
          "num_kv_heads": 4,
          "head_size": 64
        }
      ],
      "test_profiles": [
        {"batch": 2}
      ]
    }
  ]
}
```

`gfx942.json` contains the first CDNA coverage set. It uses the same smoke shape initially:

```json
{
  "schema": "ck.rocke.aot.arch_recipe/v1",
  "arch": "gfx942",
  "artifacts": [
    {
      "id": "smoke",
      "instances": [
        {
          "seqlen_q": 64,
          "seqlen_k": 64,
          "num_query_heads": 4,
          "num_kv_heads": 4,
          "head_size": 64
        }
      ],
      "test_profiles": [
        {"batch": 2}
      ]
    }
  ]
}
```

`gfx1151` viability check:

- `rocKE/Python/rocke/instances/SUPPORT_MATRIX.md` marks `fmha_fwd_mfma` as supported on `gfx1151` and records GPU-numeric PASS for `fmha_fwd_mfma` and the native `gfx1151/wmma_fmha_fwd` path.
- Local validation for the smoke shape showed `is_valid_spec(spec, "gfx1151") == (True, "ok")`.
- Local LLVM lowering for `gfx1151` emitted a WMMA kernel (`llvm.amdgcn.wmma` present, `llvm.amdgcn.mfma` absent).
- Local `compile_kernel(..., arch="gfx1151", backend="python")` produced a non-empty `amdgcn-amd-amdhsa--gfx1151` HSACO for `rocke_fmha_fwd_mfma_H64_HQ4_HK4_f16_Q64_K64_none`.
- The local machine did not expose a HIP runtime device, so the implementation task must still run the generated-artifact numeric verifier on a real `gfx1151` system before declaring GPU acceptance.
Expansion language policy:

- Allow `instances` for explicit hand-picked configs.
- Allow `product` for bounded Cartesian products when more coverage is added.
- Allow `same_as` for correlated dimensions such as `seqlen_k == seqlen_q`.
- Allow structured `constraints` such as `num_query_heads divisible_by num_kv_heads`.
- Allow exact or partial-match `exclude` entries.
- Do not allow Python expressions, arbitrary eval, imports, or plugin scripts inside recipes.

Identity policy:

- `compile_spec` contains only fields that select or affect emitted kernel code: `arch`, `family`, dtype, layout, head sizes, query/KV head counts, block sizes, mask mode, and static sequence lengths while `FmhaMfmaSpec` keeps `seqlen_q` / `seqlen_k` compile-time.
- `selection` contains graph/runtime coverage that should not create new source files, such as accepted `batch` ranges and hipDNN attribute constraints.
- `test_profiles` contains concrete runtime examples used by smoke and numeric tests.
- Tensor UIDs are not recipe identity. They are graph-binding data supplied by tests or the future dispatcher.

Dtype policy:

- Recipe parser accepts `fp16`, `f16`, and `half`.
- Parser normalizes to rocKE internal `f16` for `FmhaCommonSpec` and IR-facing helpers.
- Sidecar emits external dtype fields as `fp16`.
- Sidecar emits kernel ABI type strings as `ptr<f16, global>` because that is the actual rocKE signature spelling.

## Python Recipe Expander and Instance Parser

Add:

```text
dnn-providers/hip-kernel-provider/rocKE-client/aot/python/rocke_client_aot/__init__.py
dnn-providers/hip-kernel-provider/rocKE-client/aot/python/rocke_client_aot/recipe_schema.py
dnn-providers/hip-kernel-provider/rocKE-client/aot/python/rocke_client_aot/expand_recipe.py
dnn-providers/hip-kernel-provider/rocKE-client/aot/python/rocke_client_aot/instance_schema.py
```

Recipe expander responsibilities:

1. Load `recipe.json` and the requested `<arch>.json`.
2. Validate `schema == "ck.rocke.aot.recipe/v1"` and `schema == "ck.rocke.aot.arch_recipe/v1"`.
3. Require `op == "sdpa_fwd"` and `family == "fmha_fwd_mfma"`.
4. Merge family defaults, arch defaults if later added, and per-artifact entries.
5. Expand `instances` and `product` deterministically.
6. Apply `constraints` and `exclude` entries.
7. Normalize dtype aliases to internal `f16`.
8. Validate `canonical_layout == "BSHD"` for the first recipe.
9. Emit a build-tree normalized instance list:

```json
{
  "schema": "ck.rocke.aot.expanded_instances/v1",
  "recipe_hash": "...",
  "arch": "gfx1151",
  "instances": [
    {
      "name": "sdpa_fwd_fp16_bshd_gfx1151_q64_k64_hq4_hkv4_d64_none",
      "op": "sdpa_fwd",
      "family": "fmha_fwd_mfma",
      "arch": "gfx1151",
      "backend": "llvm_ir_comgr",
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
        "batch": {"min": 1, "max": 64}
      },
      "test_profiles": [
        {"batch": 2}
      ]
    }
  ]
}
```

Instance parser responsibilities:

1. Load generated JSON.
2. Validate `schema == "ck.rocke.aot.expanded_instances/v1"`.
3. For each `instances[]` entry, validate shape constraints:
   - `seqlen_q % 16 == 0`
   - `seqlen_k % 16 == 0`
   - `head_size in {32, 64, 128, 192, 256}`
   - `num_query_heads % num_kv_heads == 0`
   - `mask_mode == "none"` initially
4. Build:

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

5. Call `is_valid_spec(spec, arch)` and surface its failure reason.

## Python AOT Build CLI

Add:

```text
dnn-providers/hip-kernel-provider/rocKE-client/aot/tools/rocke_aot_expand.py
dnn-providers/hip-kernel-provider/rocKE-client/aot/tools/rocke_aot_build.py
```

CLI:

```bash
PYTHONPATH=dnn-providers/hip-kernel-provider/rocKE/Python:\
dnn-providers/hip-kernel-provider/rocKE-client/aot/python \
  python3 dnn-providers/hip-kernel-provider/rocKE-client/aot/tools/rocke_aot_build.py \
  --instances <build>/rocKE-client/aot/generated/sdpa_fwd_fmha_mfma.gfx1151.instances.json \
  --output-dir <build>/rocKE-client/aot/gfx1151
```

The manual command uses `PYTHONPATH` only to mirror what CMake sets for build-tree custom commands. Recipe expansion has a separate CLI:

```bash
PYTHONPATH=dnn-providers/hip-kernel-provider/rocKE/Python:\
dnn-providers/hip-kernel-provider/rocKE-client/aot/python \
  python3 dnn-providers/hip-kernel-provider/rocKE-client/aot/tools/rocke_aot_expand.py \
  --recipe-dir dnn-providers/hip-kernel-provider/rocKE-client/instances/sdpa/fmha_fwd_mfma/recipes \
  --arch gfx1151 \
  --output <build>/rocKE-client/aot/generated/sdpa_fwd_fmha_mfma.gfx1151.instances.json
```

Implementation steps:

1. Parse generated expanded instances JSON.
2. For each normalized instance, build `FmhaMfmaSpec`.
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

Use `compile_kernel(..., backend="python")` intentionally. It lowers to AMDGPU LLVM IR with the Python lowerer and assembles HSACO through `libamd_comgr`; it does not consult `ROCKE_BACKEND`, does not use `hipcc`, and does not require `rocke_engine`.

5. Write loose build-tree outputs only:

```text
<output-dir>/sdpa_fwd_fp16_bshd_gfx1151_q64_k64_hq4_hkv4_d64_none.hsaco
<output-dir>/sdpa_fwd_fp16_bshd_gfx1151_q64_k64_hq4_hkv4_d64_none.sidecar.json
```

Do not write an artifact `index.json` in this task. The generated `.instances.json` is the deterministic build recipe, and each `.sidecar.json` is self-contained metadata for one HSACO. An index would only duplicate those two sources of truth before the kpack bundling task exists.

Optional debug output can be added only if an existing helper exposes it cleanly:

```text
<output-dir>/sdpa_fwd_fp16_bshd_gfx1151_q64_k64_hq4_hkv4_d64_none.ll
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
    "kpack_binary_name": "sdpa_fwd_fp16_bshd_gfx1151_q64_k64_hq4_hkv4_d64_none.hsaco",
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
    "shape_constraints": {
      "batch": {"min": 1, "max": 64},
      "seqlen_q": {"equals": 64, "multiple_of": 16},
      "seqlen_k": {"equals": 64, "multiple_of": 16},
      "num_query_heads": {"equals": 4},
      "num_kv_heads": {"equals": 4},
      "head_size": {"equals": 64},
      "mask_mode": {"equals": "none"}
    },
    "hipdnn_node": {
      "type": "SdpaAttributes",
      "compute_data_type": "FLOAT",
      "attn_scale_source": "default_1_over_sqrt_d",
      "dropout_probability": 0.0,
      "causal_mask": false,
      "padding_mask": false,
      "alibi_mask": false
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

- `Q`, `K`, `V`, `O`: tensor pointers from hipDNN node tensor UIDs.
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
dnn-providers/hip-kernel-provider/rocKE-client/instances/sdpa/fmha_fwd_mfma/CMakeLists.txt
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
  "${CMAKE_CURRENT_SOURCE_DIR}/../instances/sdpa/fmha_fwd_mfma"
  "${CMAKE_CURRENT_BINARY_DIR}/instances/sdpa/fmha_fwd_mfma"
)

add_custom_target(rocke_client_aot_check
  COMMAND "${CMAKE_CTEST_COMMAND}" --output-on-failure -L rocKE-client
  DEPENDS rocke_client_aot_artifacts
)
```

`rocKE-client/aot/CMakeLists.txt` is the only place that lists kernel families. Adding a new family means adding one `add_subdirectory()` there; the family directory owns its recipes and tests.

Kernel developer-owned family file:

```cmake
# dnn-providers/hip-kernel-provider/rocKE-client/instances/sdpa/fmha_fwd_mfma/CMakeLists.txt
rocke_client_add_aot_recipe(
  NAME sdpa_fwd_fmha_mfma_gfx1151
  ARCH gfx1151
)

rocke_client_add_aot_recipe(
  NAME sdpa_fwd_fmha_mfma_gfx942
  ARCH gfx942
)

if(BUILD_TESTING)
  add_test(
    NAME rocke_client_sdpa_aot_numeric_gfx1151
    COMMAND "${Python3_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/sdpa_aot_numeric.py"
            --arch gfx1151
            --artifact-dir "${PROJECT_BINARY_DIR}/rocKE-client/aot/gfx1151"
            --instances "${PROJECT_BINARY_DIR}/rocKE-client/aot/generated/sdpa_fwd_fmha_mfma_gfx1151.instances.json"
  )
  set_tests_properties(rocke_client_sdpa_aot_numeric_gfx1151 PROPERTIES
    LABELS "gpu;rocKE-client;sdpa"
    REQUIRED_FILES "${PROJECT_BINARY_DIR}/rocKE-client/aot/generated/sdpa_fwd_fmha_mfma_gfx1151.build.stamp"
    SKIP_RETURN_CODE 77
  )

  add_test(
    NAME rocke_client_sdpa_aot_numeric_gfx942
    COMMAND "${Python3_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/sdpa_aot_numeric.py"
            --arch gfx942
            --artifact-dir "${PROJECT_BINARY_DIR}/rocKE-client/aot/gfx942"
            --instances "${PROJECT_BINARY_DIR}/rocKE-client/aot/generated/sdpa_fwd_fmha_mfma_gfx942.instances.json"
  )
  set_tests_properties(rocke_client_sdpa_aot_numeric_gfx942 PROPERTIES
    LABELS "gpu;rocKE-client;sdpa"
    REQUIRED_FILES "${PROJECT_BINARY_DIR}/rocKE-client/aot/generated/sdpa_fwd_fmha_mfma_gfx942.build.stamp"
    SKIP_RETURN_CODE 77
  )
endif()
```

The helper uses the `recipes/` directory under the calling family `CMakeLists.txt` as the recipe directory. No call site should pass a recipe path or output directory; the helper owns the build-tree layout.

CMake helper sketch:

```cmake
find_package(Python3 COMPONENTS Interpreter REQUIRED)

get_filename_component(_ROCKE_CLIENT_ROOT "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)
get_filename_component(_HIP_KERNEL_PROVIDER_ROOT "${_ROCKE_CLIENT_ROOT}/.." ABSOLUTE)

if(NOT TARGET rocke_client_aot_artifacts)
    message(FATAL_ERROR
        "Include rocke_aot.cmake from rocKE-client/aot/CMakeLists.txt after creating rocke_client_aot_artifacts")
endif()

function(rocke_client_add_aot_recipe)
    cmake_parse_arguments(ARG "" "NAME;ARCH" "" ${ARGN})
    if(NOT ARG_NAME OR NOT ARG_ARCH)
        message(FATAL_ERROR
            "rocke_client_add_aot_recipe requires NAME and ARCH")
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
    set(_ROCKE_CLIENT_AOT_RECIPE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/recipes")
    set(_ROCKE_CLIENT_AOT_GENERATED_DIR "${_ROCKE_CLIENT_AOT_OUTPUT_ROOT}/generated")
    set(_ROCKE_CLIENT_AOT_INSTANCES
        "${_ROCKE_CLIENT_AOT_GENERATED_DIR}/${ARG_NAME}.instances.json")
    set(_ROCKE_CLIENT_AOT_ARCH_OUTPUT_DIR
        "${_ROCKE_CLIENT_AOT_OUTPUT_ROOT}/${ARG_ARCH}")
    set(_ROCKE_CLIENT_AOT_BUILD_STAMP
        "${_ROCKE_CLIENT_AOT_GENERATED_DIR}/${ARG_NAME}.build.stamp")

    add_custom_command(
        OUTPUT "${_ROCKE_CLIENT_AOT_INSTANCES}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${_ROCKE_CLIENT_AOT_GENERATED_DIR}"
        COMMAND "${CMAKE_COMMAND}" -E env
                "PYTHONPATH=${_ROCKE_CLIENT_AOT_PYTHONPATH_NATIVE}"
                "PYTHONDONTWRITEBYTECODE=1"
                "${Python3_EXECUTABLE}"
                "${_ROCKE_CLIENT_ROOT}/aot/tools/rocke_aot_expand.py"
                --recipe-dir "${_ROCKE_CLIENT_AOT_RECIPE_DIR}"
                --arch "${ARG_ARCH}"
                --output "${_ROCKE_CLIENT_AOT_INSTANCES}"
        DEPENDS "${_ROCKE_CLIENT_AOT_RECIPE_DIR}/recipe.json"
                "${_ROCKE_CLIENT_AOT_RECIPE_DIR}/${ARG_ARCH}.json"
                "${_ROCKE_CLIENT_ROOT}/aot/tools/rocke_aot_expand.py"
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
                --instances "${_ROCKE_CLIENT_AOT_INSTANCES}"
                --output-dir "${_ROCKE_CLIENT_AOT_ARCH_OUTPUT_DIR}"
        COMMAND "${CMAKE_COMMAND}" -E touch "${_ROCKE_CLIENT_AOT_BUILD_STAMP}"
        DEPENDS "${_ROCKE_CLIENT_AOT_INSTANCES}"
                "${_ROCKE_CLIENT_ROOT}/aot/tools/rocke_aot_build.py"
        VERBATIM
    )

    add_custom_target("${ARG_NAME}" DEPENDS "${_ROCKE_CLIENT_AOT_BUILD_STAMP}")
    add_dependencies(rocke_client_aot_artifacts "${ARG_NAME}")
endfunction()
```

Outputs:

```text
${PROJECT_BINARY_DIR}/rocKE-client/aot/generated/${NAME}.instances.json
${PROJECT_BINARY_DIR}/rocKE-client/aot/generated/${NAME}.build.stamp
${PROJECT_BINARY_DIR}/rocKE-client/aot/${ARCH}/*.hsaco
${PROJECT_BINARY_DIR}/rocKE-client/aot/${ARCH}/*.sidecar.json
```

Aggregate targets owned by `rocKE-client/aot/CMakeLists.txt`:

```text
rocke_client_aot_artifacts  # builds all registered kernel-family AOT artifacts
rocke_client_aot_check      # builds artifacts, then runs rocKE-client CTest coverage
```

Dependencies:

- instance recipe JSON files;
- `rocKE-client/aot/tools/rocke_aot_expand.py`;
- `rocKE-client/aot/tools/rocke_aot_build.py`;
- provider AOT Python modules under `rocKE-client/aot/python/rocke_client_aot`;
- Python SDPA builder modules.

No `install()` calls. No provider install path. No kpack archive. No binary embedding. No `rocke_core` or `rocke_engine` dependency.

## Tests

### Python unit tests

Cover:

- valid SDPA recipe expansion for `gfx1151` and `gfx942`;
- deterministic generated instance ordering;
- dtype alias normalization;
- unsupported recipe schema rejected;
- unsupported arch recipe schema rejected;
- unsupported op/family rejected;
- invalid shape rejected during instance parsing;
- `product`, `same_as`, `constraints`, and `exclude` expansion behavior;
- direct LLVM/comgr backend value is accepted and `hipcc` is rejected;
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

- generated `.instances.json` exists;
- generated `.build.stamp` exists;
- `.hsaco` files exist and are non-empty;
- `.sidecar.json` files exist;
- provider-local `rocke_client_sdpa_aot_numeric_gfx1151` and `rocke_client_sdpa_aot_numeric_gfx942` CTest entries are registered;
- sidecar SHA fields match `.hsaco` files;
- no install tree is expected.

### GPU numeric acceptance

Add a provider-local verifier:

```text
dnn-providers/hip-kernel-provider/rocKE-client/instances/sdpa/fmha_fwd_mfma/tests/sdpa_aot_numeric.py
```

The verifier may duplicate functionality from `rocKE/Python/rocke/examples/common/fmha_fwd_verify_hip.py`. This is intentional because the provider-side acceptance harness may later move or be deleted.

Verifier behavior:

1. Accept `--arch`, `--artifact-dir`, and `--instances`.
2. Detect the local HIP device arch with `rocke.runtime.hip_module.get_device_arch()`.
3. Exit `77` when no HIP device is available or when the local device arch does not match `--arch`; the CTest entries use `SKIP_RETURN_CODE 77`.
4. Load generated `.instances.json`.
5. For each generated instance with at least one `test_profiles[]` entry:
   - load the matching `.sidecar.json`;
   - load the `.hsaco` named by the sidecar;
   - allocate Q, K, V, O, and any required workspace buffers;
   - derive launch grid from `launch.grid_formula` and the concrete `test_profiles[]` values;
   - pack kernel args from `args_signature`;
   - run the generated kernel through `rocke.runtime.launcher` / HIP module utilities;
   - compare output against an in-script dense SDPA fp32 reference.

Pass condition:

- same tolerance policy as `Python/rocke/examples/common/fmha_fwd_verify_hip.py`;
- `gfx1151` and `gfx942` tests pass on matching hardware;
- tests skip, not fail, on machines without matching hardware.

## Docs

Update only docs that describe this provider build-tree AOT recipe:

- Provider-side rocKE client docs under `dnn-providers/hip-kernel-provider/rocKE-client/` if such docs are added with the implementation: Python-only AOT command, CMake target, loose output files, and explicit note that kpack bundling is a separate task.
- `rocKE/dsl_docs/runtime/manifest_schema.md` only if needed to distinguish runtime manifest from provider AOT sidecar.

Do not document provider install locations in `rocKE/`; keep rocKE platform docs separate from provider client recipe docs.

## Implementation Order

1. Add recipe schema, expander, generated instance parser, checked-in instance recipe JSON under `recipes/`, and `rocKE-client/instances/sdpa/fmha_fwd_mfma/CMakeLists.txt`.
2. Add sidecar emitter.
3. Add recipe expansion CLI.
4. Add Python AOT build CLI for generated instance lists.
5. Add CMake build-tree recipe target.
6. Add parser/expander/sidecar tests.
7. Add CMake smoke test.
8. Add provider-local generated-artifact numeric verifier at `rocKE-client/instances/sdpa/fmha_fwd_mfma/tests/sdpa_aot_numeric.py`.
9. Update docs.
