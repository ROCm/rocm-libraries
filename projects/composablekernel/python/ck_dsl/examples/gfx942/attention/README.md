# CK DSL gfx942 unified-attention parity & benchmark harness

This folder hosts the torch-reference parity + benchmark harness for the
**gfx942 (CDNA3 / MI300X)** unified-attention 2D tiled SDPA-fwd kernel
(`ck_dsl.instances.gfx942.attention_tiled_2d`). It is the gfx942 sibling of
[`examples/gfx950/attention/`](../../gfx950/attention/README.md), but with
**no Triton/AITER dependency**: the oracle is an fp32 torch reference, so the
example runs on any box with torch + a gfx942 GPU.

Two scripts:

* `parity_unified_attention.py` — correctness (vs fp32 torch reference) +
  per-shape latency / TFLOPS for every shape in `shapes.json`, exercising
  **both** the `wide4` and `L4` D128-fp16 configs.
* `benchmark_prefill2d.py` — perf sweep of the shipped variants over the
  `perf` shapes; `--check` compares measured TFLOPS against the shipped
  baselines in `expected_perf.csv` and flags regressions.

Shipped data files:

* `shapes.json` — the canonical correctness + perf shapes (mirrors the
  ck-dsl-provider integration-test net plus the case-study perf shapes).
* `expected_perf.csv` — measured perf baselines (one row per shape × config),
  the reference for `benchmark_prefill2d.py --check`.

## wide4 is the PROVIDER's default, not the DSL spec's — this is the key nuance

The shipped headline for gfx942 D128 fp16 is **`wide4` (WG=256 / num_warps=4
flash regime), +19.7% over the L4 fallback, ~183 TF (63% of PyTorch flash)**.
But that default lives in the **provider**, not the DSL spec:

* `dnn-providers/ck-dsl-provider/.../compile_service.py` sets `_flash_wide=4`
  for the qualifying shape and runs an accurate `_lds_bytes_transposed_x8`
  wide-tile chooser; the C++ `SdpaCandidateSelector.analyticTarget` mirrors it,
  returning `{num_warps=4, block_m_per_warp=32, tile_size=64}` for gfx942 D128
  fp16.
* The **DSL spec** `UnifiedAttention2DTiledSpec`, built with no flash knobs,
  lands on the **L4 geometry** (WG=64, K single-buffer) — the DSL-side default
  and the production kill-switch (`HIPDNN_GFX942_FLASH_WIDE=0`). A standalone
  DSL example that builds the spec "by default" therefore measures ~163 TF
  (L4), **not** ~183 TF.

So to reproduce the shipped wide4 peak, this harness constructs the spec
**explicitly** (it does not rely on a DSL default), matching what the provider
builds:

| config | spec kwargs (in addition to the shape/dtype base) | block | kernel suffix |
|--------|---------------------------------------------------|-------|---------------|
| **wide4** (peak) | `num_warps=4, block_m_per_warp=32, tile_size=64, use_mfma_32x32x8=True, use_transposed_qk_32x32=True, use_k_single_buffer=False` | `(256,1,1)` | `..._w4_mw32_mfma32x8_stqk` |
| **L4** (DSL default / kill-switch) | `num_warps=1, block_m_per_warp=32, tile_size=64, use_mfma_32x32x8=True, use_transposed_qk_32x32=True, use_k_single_buffer=True` | `(64,1,1)` | `..._mw32_mfma32x8_stqk_k1buf` |
| **narrow_d64** (D64) | `num_warps=4, block_m_per_warp=32` (16×16×16 path) | `(256,1,1)` | `..._w4_mw32` |
| **narrow** (D128 bf16) | `num_warps=2` (16×16×16 path) | `(128,1,1)` | `..._w2` |

`wide4` drops `_k1buf` by design (`BLOCK_M = 4·32 = 128 > tile=64` → K is
double-buffered, LDS = 48 KB, 1 wg/CU); L4 stays at 1 wg/CU with the single
K slot. See the case study
[`dsl_docs/architecture/attention_2d_gfx942_experiment_summary.md`](../../../dsl_docs/architecture/attention_2d_gfx942_experiment_summary.md)
(Batch 5, "wide4 SHIPPED") and the lever playbook
[`dsl_docs/optimization/gfx942_playbook.md`](../../../dsl_docs/optimization/gfx942_playbook.md)
for the full evidence trail (rocprof counters, LDS budgeting, why WG=256 wins).

In the parity harness the `HIPDNN_GFX942_FLASH_WIDE` env var only **selects
which explicit spec to build** (unset / `4` → wide4, `0` → L4); it does not
delegate the choice to a DSL default.

## Running

Needs torch + a gfx942 GPU. From the `composablekernel/` checkout:

```bash
# Parity: correctness + latency for every shape (default = wide4 for D128 fp16)
PYTHONPATH=python .venv/bin/python \
    python/ck_dsl/examples/gfx942/attention/parity_unified_attention.py \
    --scenario all --report /tmp/gfx942_attn_parity.json

# Force the L4 (WG=64) contrast instead of wide4 on the D128 perf shape
HIPDNN_GFX942_FLASH_WIDE=0 PYTHONPATH=python .venv/bin/python \
    python/ck_dsl/examples/gfx942/attention/parity_unified_attention.py \
    --scenario Fp16_Prefill_GQA_S2048_D128

# Benchmark the perf shapes (wide4 + L4 + narrow) and regression-check
PYTHONPATH=python .venv/bin/python \
    python/ck_dsl/examples/gfx942/attention/benchmark_prefill2d.py \
    --scenario perf --variants wide4 L4 narrow --check
```

`parity_unified_attention.py` flags:

| Flag | Default | Notes |
|------|---------|-------|
| `--scenario NAME` (repeatable) | all | group (`correctness`/`perf`/`all`) or an exact shape name |
| `--attempts N` | `30` | timed iterations (mean per-launch over a HIP-event pair on torch's stream) |
| `--warmup N` | `10` | untimed warmup launches |
| `--tol F` | 2e-2 fp16 / 4e-2 bf16 | abs-tolerance override |
| `--report PATH` | none | dump per-shape JSON |

`benchmark_prefill2d.py` flags:

| Flag | Default | Notes |
|------|---------|-------|
| `--scenario NAME` (repeatable) | `perf` | shape selector (same grammar as parity) |
| `--variants ...` | `wide4 L4 narrow` | subset of the shipped variants; non-applicable variants are skipped per shape |
| `--attempts N` / `--warmup N` | `50` / `10` | timing iterations |
| `--check` | off | compare to `expected_perf.csv`, exit 1 on any regression |
| `--regress-pct F` | `10.0` | regression threshold (% below baseline) |
| `--write-expected PATH` | none | dump measured rows (used to re-seed `expected_perf.csv`) |
| `--rocm-ver` / `--date` | env / empty | provenance columns written to the CSV |

## Shapes (`shapes.json`)

All cases are **causal** (the unified paged kernel is causal-only); GQA is
expressed via `kv_heads < heads`. `head_size ∈ {64, 128}` — `head_size=256` is
unsupported on gfx942 (exceeds the 64 KB LDS budget) and is intentionally
absent. The harness maps each dense `[batch, heads, seqlen, head_size]` SDPA
problem onto the kernel's paged layout with `block_size=64`.

* **correctness** (10 shapes): D64 / D128, fp16 / bf16, MHA + GQA, S64 and
  S512/S528 — mirrors `IntegrationGpuCkDslSdpaFwdFp16.cpp`.
* **perf** (2 shapes): `Fp16_Prefill_GQA_S2048_D128` (Hq32/Hkv8, the wide4
  headline shape) and `Fp16_Prefill_GQA_S2048_D64`.

## Latest results (MI300X / gfx942, ROCm 7.14 / torch 2.11)

Measured on `ctr-cx64-mi300x-4` (gfx942), 50 timed iterations after 10 warmup
launches, HIP-event timer on torch's current stream. Correctness is checked vs
an fp32 torch reference for all 10 correctness shapes + both perf shapes
(all **PASS**, max_abs ≤ 6.1e-5 fp16 / ≤ 4.9e-4 bf16; tol 2e-2 / 4e-2).

### Perf shapes — the shipped baselines (`expected_perf.csv`)

| shape | dtype | config | TFLOPS | median us | note |
|-------|-------|--------|-------:|----------:|------|
| `Fp16_Prefill_GQA_S2048_D128` | fp16 | **wide4** | **191.1** | 359.58 | shipped peak (provider default, built explicitly here) |
| `Fp16_Prefill_GQA_S2048_D128` | fp16 | L4 | 162.7 | 422.44 | DSL default / `HIPDNN_GFX942_FLASH_WIDE=0` kill-switch |
| `Fp16_Prefill_GQA_S2048_D64`  | fp16 | narrow_d64 | 149.1 | 230.40 | D64 16×16×16 narrow path |

wide4 is **+17.5%** over L4 on this box (191.1 vs 162.7), consistent with the
case study's +19.7% (153.6 → 183.8 TF) on its GQA S2048 shape — the absolute
TF differ because the example's perf shape (Hq32/Hkv8) is larger than the
case-study Batch-5 shape, but the wide4/L4 ratio holds.

### `--check` mode

`benchmark_prefill2d.py --check` reads `expected_perf.csv`, re-measures each
`(shape, config)`, and exits non-zero if any measured TFLOPS is more than
`--regress-pct` (default 10%) below its baseline. Use it as a CI/perf guard:

```bash
PYTHONPATH=python .venv/bin/python \
    python/ck_dsl/examples/gfx942/attention/benchmark_prefill2d.py \
    --scenario perf --variants wide4 L4 narrow --check
# CHECK PASSED: all measured shapes within tolerance of baseline
```

To re-baseline on a new box / ROCm stack, re-run with `--write-expected` and
copy the resulting CSV over `expected_perf.csv`:

```bash
PYTHONPATH=python .venv/bin/python \
    python/ck_dsl/examples/gfx942/attention/benchmark_prefill2d.py \
    --scenario perf --variants wide4 L4 narrow \
    --rocm-ver 7.14.0 --date 2026-06-05 \
    --write-expected python/ck_dsl/examples/gfx942/attention/expected_perf.csv
```

## How the harness maps a dense SDPA problem onto the paged kernel

Each batch element becomes one sequence with `(query_len, kv_len) =
(seqlen_q, seqlen_k)`; the per-sequence block table is a contiguous,
non-overlapping run of `block_size=64`-token cache blocks. Inputs are filled
`uniform(-0.1, 0.1)` (matching the integration test) to keep the softmax
accumulation in a numerically friendly part of the 16-bit float range. The
launch grid is recomputed exactly as the production dispatcher does
(`grid=(num_kv_heads, total_num_q_blocks, 1)`, `block=(64·num_warps, 1, 1)`),
so the example exercises the same build + launch plumbing as the provider.
