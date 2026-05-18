# Manifest Schema (v1) Reference

Manifests are the portable description of one compiled CK DSL kernel. The runner (`python -m ck_dsl.run_manifest`), the example test harness, and the sweep / benchmark drivers all consume the same v1 schema. Source of truth: `helpers/manifest.py`.

Schema version string:

```text
"schema": "ck.dsl.example.manifest/v1"
```

## Shared Fields

Every manifest carries:

| Field                | Type    | Notes                                                        |
|----------------------|---------|--------------------------------------------------------------|
| `schema`             | string  | always `"ck.dsl.example.manifest/v1"`                        |
| `kind`               | string  | `"gemm_fp16"`, `"batched_gemm_fp16"`, `"conv_fp16"`, `"attention_unified"`, `"elementwise_fp16"`, `"reduce_fp16"`, `"layernorm_fp16"`, `"rmsnorm_fp16"`, `"transpose_fp16"` |
| `kernel_name`        | string  | matches the HSACO function name                              |
| `hsaco`              | string  | basename of the HSACO file next to `manifest.json`           |
| `threads_per_block`  | int     | matches the kernel's `max_workgroup_size`                    |
| `args_signature`     | list    | one dict per kernel arg: `{"name": ..., "type": ..., "size_bytes": ...}` |
| `sig_has_bytes`      | int 0/1 | 1 if `*_bytes` args are present in `args_signature`          |
| `warmup_iters`       | int     | default 5                                                    |
| `timed_iters`        | int     | default 100                                                  |
| `timing_ms`          | object  | copy of `KernelArtifact.timings`                             |
| `hsaco_bytes`        | int     | `len(artifact.hsaco)`                                        |
| `atoms`              | list    | optional, names of MFMA atoms used (e.g. `["mfma_f32_32x32x16_f16"]`) |
| `notes`              | string  | free text                                                    |
| `ck_dependency`      | bool    | `false` — these kernels do not link CK templates             |
| `ir_authored`        | bool    | `true`                                                       |

## Per-Kind Fields

### `gemm_fp16` / `batched_gemm_fp16`

Emitted by `make_gemm_manifest(...)`:

```json
{
  "block_m": 128, "block_n": 128, "block_k": 32,
  "default_shape": [3328, 4096, 4096],
  "grid_order": "MN"
}
```

| Field             | Type    | Notes                                                      |
|-------------------|---------|------------------------------------------------------------|
| `block_m`         | int     | tile_m                                                     |
| `block_n`         | int     | tile_n                                                     |
| `block_k`         | int     | tile_k                                                     |
| `default_shape`   | `[M,N,K]` | runner shape when `--shape` is omitted                   |
| `grid_order`      | `"MN"` or `"NM"` | how `(M_tiles, N_tiles)` map to `(gx, gy)`        |
| `args_signature`  | list    | from `gemm_args_signature()` (A, B, C, M, N, K i32s)       |

Batched GEMM appends `stride_a`, `stride_b`, `stride_c` (i32 in this signature, but the kernel reads them as element strides) and uses `block_id_z` for the batch axis.

### `conv_fp16`

Emitted by `make_conv_manifest(...)`:

```json
{
  "conv_layout": "implicit_gemm",
  "block_m": 64, "block_n": 64, "block_k": 64,
  "conv": [N, H, W, C, K, R, S, sH, sW, pH, pW, dH, dW],
  "groups": 1, "cpg": 64, "kpg": 64,
  "grid_explicit": [gx, gy, gz]   // optional
}
```

| Field           | Notes                                                                |
|-----------------|----------------------------------------------------------------------|
| `conv_layout`   | `"implicit_gemm"`, `"direct_grouped_16c"`, `"direct_grouped_4c"`     |
| `conv`          | exactly 13 ints (raises `ValueError` otherwise)                      |
| `groups/cpg/kpg`| grouping; dense conv uses `groups=1, cpg=C, kpg=K`                   |
| `grid_explicit` | bypass automatic grid derivation; required for direct conv kernels   |
| `grid_order`    | `"MN"` or `"NM"` when `grid_explicit` not given                      |
| `args_signature`| from `conv_args_signature()` (A, B, D, A_bytes, B_bytes, D_bytes)    |

`sig_has_bytes=1` for all conv manifests (buffer-resource path requires the `*_bytes` args).

### `attention_unified`

Emitted by `make_attention_manifest(...)`:

```json
{
  "attention_path": "2d" | "3d" | "reduce" | "3d_tiled",
  "grid_explicit": [gx, gy, gz],
  "block_explicit": [bx, by, bz],
  "attention_config": { ... },        // free-form per-kernel config
  "args_signature": [...]             // from attention_args_signature(path=...)
}
```

The runner does not currently launch attention manifests itself; attention runs through `run_unified_attention_torch(...)` in `instances/attention_unified.py`. The manifest is for inspection and reproducibility.

`attention_args_signature(path="2d")` (and `"3d"`, `"3d_tiled"`) returns:

```text
output_ptr, query_ptr, key_cache_ptr, value_cache_ptr, sink_ptr,
block_tables_ptr, seq_lens_ptr, alibi_slopes_ptr, qq_bias_ptr,
query_start_len_ptr, scale, k_scale, v_scale, out_scale, softcap, num_seqs
```

`attention_args_signature(path="reduce")` returns the smaller reduce-stage signature:

```text
output_ptr, segm_output_ptr, segm_max_ptr, segm_expsum_ptr,
seq_lens_ptr, query_start_len_ptr
```

### `elementwise_fp16` / `reduce_fp16` / `layernorm_fp16` / `rmsnorm_fp16` / `transpose_fp16`

Emitted by `make_simple_op_manifest(...)`:

```json
{
  "op": "add" | "sub" | "mul" | "max" | "min" | "copy" | ...,
  "dtype": "f16",
  "default_shape": [N] or [M, N],
  "elems_per_block": ...,             // optional
  "is_binary": false,                  // true for two-input elementwise
  "grid_explicit": [...],              // optional
  "eps": 1e-5                          // optional, norm-only
}
```

The runner uses `default_shape` to allocate `X`, `Gamma`/`Beta` (norms), and `Y` tensors and the matching numpy reference.

## `args_signature` Type Strings

`args_signature[*].type` is one of:

```text
"ptr<f16, global>"          # 8-byte device pointer to fp16 global memory
"ptr<bf16, global>"         # 8-byte device pointer to bf16 global memory
"ptr<f32, global>"          # 8-byte device pointer to f32 global memory
"ptr<i32, global>"          # 8-byte device pointer to i32 global memory
"i32"                       # 4-byte scalar
"i64"                       # 8-byte scalar
"f32"                       # 4-byte scalar
```

`size_bytes` is for the host arg-packer (`runtime/torch_module.py::pack_args`) and the runner. Pointers are always 8 bytes; scalars match the canonical width.

## Standard Signatures

```text
gemm_args_signature(with_bytes=False) ->
  [A: ptr<f16,global>, B: ptr<f16,global>, C: ptr<f16,global>,
   M: i32, N: i32, K: i32]

gemm_args_signature(with_bytes=True) ->
  [A, B, C, A_bytes: i32, B_bytes: i32, C_bytes: i32]   # buffer-rsrc shape

conv_args_signature() ->
  [A, B, D, A_bytes, B_bytes, D_bytes]                  # always with bytes

attention_args_signature(path="2d") -> 16 entries (see above)
attention_args_signature(path="reduce") -> 6 entries
```

## Runner Behavior

`python -m ck_dsl.run_manifest <hsaco> <manifest> [--shape M,N,K] [--verify]`:

1. Reads the manifest.
2. Loads the HSACO via `Runtime.load_module`.
3. Allocates problem buffers (`numpy` -> `Runtime.alloc` -> `hipMemcpy`).
4. Packs args from `args_signature` in declaration order using `struct.pack` (or `pack_args` if torch is present).
5. Computes the grid from `grid_explicit` if set, else `(ceil_div(N, block_n), ceil_div(M, block_m), 1)` (`gemm_fp16`) or the kind-specific helper.
6. Calls `time_launches(launch_fn, warmup=warmup_iters, iters=timed_iters)`.
7. If `--verify`, copies the output back, computes the numpy / torch reference, and prints `verify max_abs_diff=... bad=K/N`.
8. Prints `Perf: <ms>, <TFLOPS>, <GB/s>`.

`max_abs_diff = 0` is bit-exactness; conv uses a `< 1e-2` tolerance internally (fp16 sum noise on the implicit-GEMM path).

## Writing A Manifest By Hand

You usually don't need to. The minimal flow is:

```python
from ck_dsl.helpers import compile_kernel, write_artifact, make_gemm_manifest

art = compile_kernel(kernel)
manifest = make_gemm_manifest(
    artifact=art,
    block_m=128, block_n=128, block_k=32,
    threads_per_block=256,
    default_shape=(3328, 4096, 4096),
    atoms=["mfma_f32_32x32x16_f16"],
    notes="hero compv4 cshuffle 32x32x16",
)
paths = write_artifact(art, Path("build/ck_dsl_example"), manifest)
```

For custom op types not yet covered by `make_simple_op_manifest`, build the dict directly and call `write_artifact` with the constructed manifest.

## Test Harness Behavior

`python/test/test_ck_dsl_examples.py`:

1. Discovers every `example/ck_tile/dsl/<N>_*/gen.py` with an adjacent `expected.json`.
2. Runs `gen.py --output-dir <tmp>` in a subprocess.
3. Reads `manifest.json`; asserts a single HSACO file and non-zero size.
4. Runs `python -m ck_dsl.run_manifest --verify` against the manifest.
5. Asserts `max_abs_diff=0` for bit-exact kernels (GEMM hero shapes) or `bad=0` for tolerance kernels (conv).
6. Asserts measured TFLOPS / GB/s pass any lower bound declared in `expected.json`.

`expected.json` schema (per example):

```json
{
  "kind": "gemm_fp16",
  "shapes": [
    {"M": 3328, "N": 4096, "K": 4096, "tflops_lower_bound": 200.0}
  ]
}
```

The runner output `Perf: X ms, Y TFlops, Z GB/s` is parsed by `_parse_tflops` and `_parse_gbps`; the test asserts `Y >= tflops_lower_bound`.
