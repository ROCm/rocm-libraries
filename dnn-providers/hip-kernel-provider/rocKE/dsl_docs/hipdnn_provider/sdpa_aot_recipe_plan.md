# rocKE SDPA AOT Recipe Plan

## Goal

Add a repeatable, build-time AOT recipe for one rocKE SDPA forward kernel using only the Python rocKE implementation. The output is a loose HSACO plus metadata sidecar in the build tree. A later task will bundle HSACO files into a kpack archive.

## Constraints

- First example is SDPA forward, not GEMM.
- Use only `rocKE/Python/ck_dsl` code paths.
- Do not use `ckc_core`, `ckc_engine`, C++ lowering, C++ metadata APIs, or C ABI producers.
- Do not install generated artifacts.
- Do not embed HSACO binaries in provider sources.
- Do not create a kpack archive in this task.
- Keep dispatcher selection out of scope; this is one declared instance.
- Keep metadata generation off lowering paths so emitted kernel code is unchanged except for the normal Python build output.

## First SDPA Instance

Use dense SDPA/FMHA forward:

- Builder: `Python/ck_dsl/instances/common/fmha_mfma.py`
- Spec types: `FmhaMfmaSpec`, `FmhaCommonSpec`, `FmhaShape`
- Build helper: `build_fmha_fwd_mfma`
- Launch helper: `fmha_fwd_mfma_grid`
- Signature helper: `fmha_fwd_mfma_signature`
- Existing numeric reference: `Python/ck_dsl/examples/common/fmha_fwd_verify_hip.py`

Use `fmha_mfma` instead of `attention_unified` for the first example because it is a single dense SDPA kernel with deterministic grid and signature. `attention_unified` adds paged-KV layout, 2D/3D path selection, optional split/reduce kernels, workspace, block tables, and dynamic launch behavior that belongs in a later dispatcher task.

## Instance File

Add:

```text
dnn-providers/hip-kernel-provider/rocKE/aot/instances/sdpa_fwd_f16_bshd_gfx950_q64_k64_h4_d64.json
```

Initial contents:

```json
{
  "schema": "ck.rocke.aot.instance/v1",
  "name": "sdpa_fwd_f16_bshd_gfx950_q64_k64_h4_d64",
  "op": "sdpa_fwd",
  "family": "fmha_fwd_mfma",
  "arch": "gfx950",
  "backend": "hipcc",
  "problem": {
    "batch": 2,
    "seqlen_q": 64,
    "seqlen_k": 64,
    "num_query_heads": 4,
    "num_kv_heads": 4,
    "head_size": 64,
    "dtype": "fp16",
    "mask_mode": "none",
    "canonical_layout": "BSHD"
  },
  "hipdnn_node": {
    "type": "SdpaAttributes",
    "compute_data_type": "FLOAT",
    "q_tensor_uid": 1,
    "k_tensor_uid": 2,
    "v_tensor_uid": 3,
    "o_tensor_uid": 4,
    "attn_scale_source": "default_1_over_sqrt_d",
    "dropout_probability": 0.0,
    "causal_mask": false,
    "padding_mask": false,
    "alibi_mask": false
  }
}
```

Dtype policy:

- Instance parser accepts `fp16`, `f16`, and `half`.
- Parser normalizes to rocKE internal `f16` for `FmhaCommonSpec` and IR-facing helpers.
- Sidecar emits external dtype fields as `fp16`.
- Sidecar emits kernel ABI type strings as `ptr<f16, global>` because that is the actual rocKE signature spelling.

## Python Instance Parser

Add:

```text
rocKE/Python/ck_dsl/aot/__init__.py
rocKE/Python/ck_dsl/aot/instance_schema.py
```

Parser responsibilities:

1. Load JSON.
2. Validate `schema == "ck.rocke.aot.instance/v1"`.
3. Require `op == "sdpa_fwd"`.
4. Require `family == "fmha_fwd_mfma"`.
5. Require `canonical_layout == "BSHD"` for the first instance.
6. Normalize dtype aliases to internal `f16`.
7. Validate shape constraints:
   - `seqlen_q % 16 == 0`
   - `head_size in {32, 64, 128, 192, 256}`
   - `num_query_heads % num_kv_heads == 0`
   - `mask_mode == "none"` initially
8. Build:

```python
common = FmhaCommonSpec(
    FmhaShape(
        head_size=head_size,
        num_query_heads=num_query_heads,
        num_kv_heads=num_kv_heads,
        block_size_q=16,
        block_size_k=64,
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

9. Call `is_valid_spec(spec, arch)` and surface its failure reason.

## Python AOT Build CLI

Add:

```text
rocKE/tools/rocke_aot_build.py
```

CLI:

```bash
PYTHONPATH=rocKE/Python python3 rocKE/tools/rocke_aot_build.py \
  --instance rocKE/aot/instances/sdpa_fwd_f16_bshd_gfx950_q64_k64_h4_d64.json \
  --output-dir <build>/rocKE/aot
```

Implementation steps:

1. Parse the instance JSON.
2. Build `FmhaMfmaSpec`.
3. Build the kernel:

```python
kernel = build_fmha_fwd_mfma(spec, arch=arch)
```

4. Compile with the existing Python HIP path:

```python
artifact = compile_kernel_via_hipcc(kernel, arch=arch)
```

5. Write loose build-tree outputs only:

```text
<output-dir>/sdpa_fwd_f16_bshd_gfx950_q64_k64_h4_d64.hsaco
<output-dir>/sdpa_fwd_f16_bshd_gfx950_q64_k64_h4_d64.sidecar.json
```

Optional debug output can be added only if an existing helper exposes it cleanly:

```text
<output-dir>/sdpa_fwd_f16_bshd_gfx950_q64_k64_h4_d64.hip.cpp
```

## Sidecar Emitter

Add:

```text
rocKE/Python/ck_dsl/aot/sidecar.py
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
    "arch": "gfx950",
    "abi_version": "hipkg-sdpa-fwd-fmha-mfma/v1",
    "request_hash": "...",
    "spec_hash": "...",
    "cache_key": "sdpa_fwd:fmha_fwd_mfma:..."
  },
  "artifact": {
    "kpack_binary_name": "sdpa_fwd_f16_bshd_gfx950_q64_k64_h4_d64.hsaco",
    "symbol": "<artifact.kernel_name>",
    "hsaco_sha256": "...",
    "hsaco_size": 12345
  },
  "selection": {
    "op": "sdpa_fwd",
    "arch": "gfx950",
    "dtypes": {
      "q": "fp16",
      "k": "fp16",
      "v": "fp16",
      "o": "fp16",
      "acc": "fp32"
    },
    "canonical_layout": "BSHD",
    "shape_constraints": {
      "batch": {"equals": 2},
      "seqlen_q": {"equals": 64, "multiple_of": 16},
      "seqlen_k": {"equals": 64},
      "num_query_heads": {"equals": 4},
      "num_kv_heads": {"equals": 4},
      "head_size": {"equals": 64},
      "mask_mode": {"equals": "none"}
    },
    "hipdnn_node": {
      "type": "SdpaAttributes",
      "compute_data_type": "FLOAT",
      "q_tensor_uid": 1,
      "k_tensor_uid": 2,
      "v_tensor_uid": 3,
      "o_tensor_uid": 4,
      "attn_scale_source": "default_1_over_sqrt_d",
      "dropout_probability": 0.0,
      "causal_mask": false,
      "padding_mask": false,
      "alibi_mask": false
    }
  },
  "launch": {
    "grid": [4, 4, 2],
    "block": [64, 1, 1],
    "shared_mem_bytes": 0,
    "tile_sizes": {
      "block_q": 16,
      "block_k": 64,
      "head_size": 64,
      "wave_size": 64
    }
  },
  "args_signature": []
}
```

Computed fields:

- `artifact.hsaco_sha256 = sha256(hsaco bytes)`
- `artifact.hsaco_size = len(hsaco bytes)`
- `launch.grid = fmha_fwd_mfma_grid(spec, batch=batch)`
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
rocKE/cmake/ckc_aot.cmake
```

Function:

```cmake
ckc_add_aot_instance(
  NAME sdpa_fwd_f16_bshd_gfx950_q64_k64_h4_d64
  INSTANCE aot/instances/sdpa_fwd_f16_bshd_gfx950_q64_k64_h4_d64.json
  OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/aot"
)
```

Outputs:

```text
${OUTPUT_DIR}/${NAME}.hsaco
${OUTPUT_DIR}/${NAME}.sidecar.json
```

Aggregate target:

```text
ckc_aot_artifacts
```

Dependencies:

- instance JSON;
- `tools/rocke_aot_build.py`;
- Python AOT modules;
- Python SDPA builder modules.

No `install()` calls. No provider install path. No kpack archive. No binary embedding. No `ckc_core` dependency.

## Tests

### Python unit tests

Cover:

- valid SDPA instance parse;
- dtype alias normalization;
- unsupported schema rejected;
- unsupported op/family rejected;
- invalid shape rejected;
- sidecar required fields present;
- `launch.grid == [4, 4, 2]`;
- `launch.block == [64, 1, 1]` for gfx950;
- `selection.dtypes.q == "fp16"`;
- first four args are `Q`, `K`, `V`, `O`;
- pointer arg ABI strings are `ptr<f16, global>`;
- scalar arg sizes and alignments are 4 bytes;
- HSACO SHA and size fields match the emitted bytes.

### CMake smoke

Build:

```bash
cmake --build <build> --target ckc_aot_artifacts
```

Assert:

- `.hsaco` exists and is non-empty;
- `.sidecar.json` exists;
- sidecar SHA matches `.hsaco`;
- no install tree is expected.

### GPU numeric acceptance

Extend `fmha_fwd_verify_hip.py` or add a sibling verifier that:

1. loads generated `.hsaco`;
2. reads generated `.sidecar.json`;
3. packs args from the sidecar;
4. runs the same dense attention reference check as the existing example.

Pass condition: existing tolerance from `fmha_fwd_verify_hip.py`.

## Docs

Update only docs that describe this build-tree AOT recipe:

- `rocKE/BUILD.md`: Python-only AOT command, CMake target, loose output files, and explicit note that kpack bundling is a separate task.
- `rocKE/dsl_docs/runtime/manifest_schema.md`: distinguish runtime manifest from AOT sidecar.

Do not document install locations.

## Implementation Order

1. Add SDPA instance parser and checked-in JSON.
2. Add sidecar emitter.
3. Add Python AOT CLI.
4. Add CMake build-tree target.
5. Add parser/sidecar tests.
6. Add CMake smoke test.
7. Add generated-artifact numeric verifier.
8. Update docs.
