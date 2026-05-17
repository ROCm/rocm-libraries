# Examples Index

Shipped CK DSL examples come in two forms:

1. **Python-owned generators** in `python/ck_dsl/examples/`. They build HSACO + manifest and (for `examples/*.py` other than the bake-offs) launch + verify in one process.
2. **CMake-integrated generators** in `example/ck_tile/dsl/<NN>_*/gen.py`. Each `gen.py` wraps a Python-owned generator and adds an `expected.json` gate for `test_ck_dsl_examples.py`.

Per the validation pass on this checkout, every shipped example builds and verifies end-to-end on MI355X / gfx950 / ROCm 7.0.2 / torch 2.8.0+rocm7.0.2.

## Python-Owned Generators (`python/ck_dsl/examples/`)

| File                                              | Purpose                                                          | Reproduction command |
|---------------------------------------------------|------------------------------------------------------------------|----------------------|
| `bake_off_implicit_gemm.py`                       | Implicit-GEMM conv (bake-off 1). 230+ TFLOPS on MI355X.          | `python -m ck_dsl.examples.bake_off_implicit_gemm --output-dir /tmp/x` |
| `bake_off_direct_conv_16c.py`                     | Direct grouped 16c conv (bake-off 2).                            | `python -m ck_dsl.examples.bake_off_direct_conv_16c --output-dir /tmp/x` |
| `bake_off_direct_conv_4c.py`                      | Direct grouped 4c conv.                                          | `python -m ck_dsl.examples.bake_off_direct_conv_4c --output-dir /tmp/x` |
| `distribution_reduce_demo.py`                     | 1D distribution-driven row-reduce.                               | `python python/ck_dsl/examples/distribution_reduce_demo.py --M 32 --N 4096` |
| `distribution_2d_add_demo.py`                     | 2D distribution-driven elementwise add.                          | `python python/ck_dsl/examples/distribution_2d_add_demo.py --H 64 --W 128` |
| `ck_tile_parity.py`                               | Small-op parity harness vs torch reference. Returns non-zero if any op exceeds its tolerance gate. | `python python/ck_dsl/examples/ck_tile_parity.py --op all` |
| `attention/parity_unified_attention.py`           | Triton vs CK DSL attention parity. All paths (`auto`, `2d`, `3d`) on the AITER unified-attention contract. | `python python/ck_dsl/examples/attention/parity_unified_attention.py --attempts 10 --warmup 5` |

The bake-offs build a HSACO and manifest; you launch / verify with `python -m ck_dsl.run_manifest <hsaco> manifest.json --verify`. The other examples are self-contained and verify in-process.

## CMake-Integrated Generators (`example/ck_tile/dsl/`)

Each directory has `gen.py` (builds the artifact) and `expected.json` (correctness + lower-bound TFLOPS gate). Test harness: `python/test/test_ck_dsl_examples.py`.

| Example                              | Backing builder                                              |
|--------------------------------------|--------------------------------------------------------------|
| `02_layernorm2d`                     | `instances/layernorm2d.py`                                   |
| `05_reduce`                          | `instances/reduce.py`                                        |
| `06_gemm_universal_cshuffle`         | `instances/gemm_universal.py` (hero compv4 cshuffle)         |
| `07_gemm_universal_sweep`            | `instances/gemm_universal.py` (`all_dispatcher_configs`)     |
| `08_bake_off_implicit_gemm`          | `instances/conv_implicit_gemm.py`                            |
| `09_bake_off_direct_conv_16c`        | `instances/conv_direct_grouped.py` (16c)                      |
| `10_bake_off_direct_conv_4c`         | `instances/conv_direct_grouped.py` (4c)                       |
| `10_rmsnorm2d`                       | `instances/rmsnorm2d.py`                                     |
| `16_batched_gemm`                    | `instances/batched_gemm.py`                                  |
| `21_elementwise`                     | `instances/elementwise.py`                                   |
| `37_transpose`                       | `instances/transpose.py`                                     |

Build + verify one example by hand:

```bash
PYTHONPATH=python python example/ck_tile/dsl/08_bake_off_implicit_gemm/gen.py \
    --output-dir /tmp/ex08
PYTHONPATH=python python -m ck_dsl.run_manifest \
    /tmp/ex08/*.hsaco /tmp/ex08/manifest.json --verify
```

`expected.json` schema (per example, used by `test_ck_dsl_examples.py`):

```json
{
  "kind": "gemm_fp16",
  "shapes": [
    {"M": 3328, "N": 4096, "K": 4096, "tflops_lower_bound": 200.0}
  ]
}
```

The harness asserts `max_abs_diff = 0` for bit-exact kernels, `bad = 0` for conv tolerance, and `measured_tflops >= tflops_lower_bound`.

## Sweep Examples

```bash
# Generate dispatcher-set HSACOs for the hero subset, in parallel.
PYTHONPATH=python python example/ck_tile/dsl/07_gemm_universal_sweep/gen.py \
    --output-dir /tmp/sweep --subset compute --parallel 16

# Benchmark each entry with median + spread reporting.
PYTHONPATH=python python -m ck_dsl.sweep_bench /tmp/sweep/sweep_manifest.json \
    --attempts 3 --csv /tmp/sweep/results.csv
```

`07_gemm_universal_sweep/gen.py` accepts:

- `--subset compute` — the hero compute-bound subset of dispatcher configs;
- `--parallel N` — workers for parallel COMGR compilation;
- `--output-dir <path>` — destination for HSACOs and `sweep_manifest.json`.

The benchmark writes a CSV with `(kernel, shape, median_tflops, min_tflops, max_tflops, spread_pct, max_abs_diff)`.

## Attention Parity Methodology

`examples/attention/parity_unified_attention.py` is the canonical attention parity harness. It:

- forces Triton onto the 2D / 3D kernel per row by monkey-patching its `use_2d_kernel(...)` heuristic;
- forces CK DSL onto the matching path via `run_unified_attention_torch(..., backend="tiled" / "3d" / "auto")`;
- routes both backends through `torch.cuda.current_stream()`;
- times with one HIP-event timer (`time_launches`) for both;
- compares both backends to AITER's `ref_paged_attn`.

Scenario sets:

- `default` — 11 production scenarios (decode, prefill, mixed, sliding, softcap, bf16, ALiBi, qq-bias);
- `creative` — exploratory sweep (long-context, GQA/MQA, head_size=256, bf16, sliding extremes, bias combinations);
- `fmha` — shapes adapted from CK FMHA testing matrix;
- `all` — `default + creative`.

The published 5-run mean numbers (`auto` geomean ~1.799x, `3D vs 3D` ~1.743x) come from `--attempts 10 --warmup 5` over the `default` set; see `examples/attention/README.md`.

## Running Everything In Order

The full validation flow used during this docs pass:

```bash
cd /workspace/rocm-libraries-streaming/projects/composablekernel
VENV=/workspace/dsl_bake_off/venv/bin/python
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=python

# 1. Static unit suite.
$VENV python/test/test_ck_dsl.py

# 2. Generated example harness (all CK Tile parity examples).
$VENV python/test/test_ck_dsl_examples.py

# 3. README-style implicit-GEMM conv build + verify.
$VENV -m ck_dsl.examples.bake_off_implicit_gemm --output-dir /tmp/ex08
$VENV -m ck_dsl.run_manifest /tmp/ex08/*.hsaco /tmp/ex08/manifest.json --verify

# 4. Distribution demos.
$VENV python/ck_dsl/examples/distribution_reduce_demo.py --M 32 --N 4096
$VENV python/ck_dsl/examples/distribution_2d_add_demo.py --H 64 --W 128

# 5. Small-op parity.
$VENV python/ck_dsl/examples/ck_tile_parity.py --op all

# 6. Attention smoke.
PYTHONPATH=python:/workspace/aiter $VENV \
  python/ck_dsl/examples/attention/parity_unified_attention.py \
  --scenario decode_d128_b16 --attempts 1 --warmup 0 --paths auto,2d,3d
```

Measured outputs: `optimization/measured_results.md`.
