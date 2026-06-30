# gfx942 unified attention — parity + benchmark harness

A torch-reference **parity + latency** harness for the **gfx942 (CDNA3 /
MI300X)** unified-attention SDPA-forward kernels: `attention_tiled_2d` for
prefill and `attention_tiled_3d` for the split-KV decode path, both in
`rocke.instances.gfx942.attention_tiled_2d` / `..._tiled_3d`. Unlike the gfx950
attention harness it has **no Triton/AITER dependency** — the oracle is an fp32
torch reference — so it runs on any box with torch + a gfx942 GPU.

> **New to flash attention?** [`ALGORITHM.md`](ALGORITHM.md) derives what these
> kernels compute from the math up — the online-softmax recurrence, the paged-KV
> layout the harness feeds them, the causal masking, and the CDNA3-specific
> transposed-x8 (register-$P^\top$) matmul that the D128-fp16 flash regime uses.
> Read it first if you want the *why* before the *how to run*.

## What it shows

- A correct gfx942 tiled SDPA-forward over a canonical shape set
  (`shapes.json`): fp16/bf16, `head_size` 64 / 128, MHA + GQA, and the
  decode / short-prefill / long-prefill regimes — each gated against an fp32
  paged reference.
- The two shipped D128-fp16 flash geometries side by side: the provider's
  analytic-default **wide4** (workgroup 256, `num_warps=4`) and the **L4**
  (workgroup 64) contrast, selectable by environment variable.
- A CK-vs-Torch latency comparison in **two timing modes** — eager (host launch
  overhead + kernel) and CUDA-graph (kernel-only, host overhead removed) — so a
  perf gap can be attributed to kernel time vs launch overhead.

## Prerequisites

- A gfx942 (CDNA3 / MI300X) GPU and a ROCm PyTorch build (`torch.cuda` must
  resolve to the HIP device).
- The repo's `python/` on `PYTHONPATH` (commands below set it).
- No numpy required: the harness uses plain Python lists for the block tables so
  it runs on the minimal container torch build.

## File map

| file | role |
|------|------|
| `shapes.json` | the canonical shape set, grouped by regime: `decode`, `short_prefill`, `long_prefill`, and `d256_disabled` (off by default) |
| `parity_unified_attention.py` | per-shape parity + latency harness; builds the gfx942 spec **explicitly** (wide4 vs L4) and exposes the `HIPDNN_GFX942_*` lever knobs via `--scenario` |
| `final_shapes_check.py` | the definitive correctness + perf check over every shape via the **production dispatcher** (`run_unified_attention_torch`), timed eager + graph against PyTorch's flash SDPA |
| `__init__.py` | package marker + module docstring |

## Running

Both scripts need torch + a gfx942 GPU. From the repo's `composablekernel/`
directory (so `python/` is the package root):

```bash
# 1) The definitive correctness + perf check over every shape, via the
#    production dispatcher, timed eager + graph vs PyTorch flash SDPA:
PYTHONPATH=Python python Python/rocke/examples/gfx942/attention/final_shapes_check.py

# 2) The explicit parity + latency harness (builds the spec by hand; default
#    runs the whole shapes.json set):
PYTHONPATH=Python python Python/rocke/examples/gfx942/attention/parity_unified_attention.py
```

### `final_shapes_check.py` flags (all verified against `argparse`)

| flag | default | meaning |
|------|---------|---------|
| `--groups G [G ...]` | all non-`*_disabled` groups | which `shapes.json` groups to run (e.g. `--groups decode`, or `--groups d256_disabled` to include the off-by-default d256 set) |
| `--warmup N` | 10 | warm-up launches before timing |
| `--iters N` | 50 | inner timed launches per measurement |
| `--reps N` | 10 | timed measurements per cell, reduced per `--reduce` |
| `--reduce {median,mean,trimmed}` | `trimmed` | how to reduce the `--reps` measurements |
| `--trim F` | 0.2 | fraction stripped from each end for `--reduce trimmed` (must be in `[0, 0.5)`) |

It prints, per shape, the eager and graph CK times, the eager and graph Torch
times, the two CK/Torch ratios (a ratio `< 1` means CK is faster), the SDPA
backend used (`flash` or `default`), and PASS/FAIL. It exits non-zero on any
correctness failure.

### `parity_unified_attention.py` flags

| flag | default | meaning |
|------|---------|---------|
| `--scenario S` (repeatable) | all shapes | a group (`correctness` / `perf` / `decode` / `all`) **or** an exact shape name; multiple flags union together |
| `--attempts N` | 30 | timed launches |
| `--warmup N` | 10 | warm-up launches |
| `--tol F` | `2e-2` fp16 / `4e-2` bf16 | absolute-tolerance override |
| `--report PATH` | none | write a JSON results report |
| `--debug-mismatch N` | 0 | on failure, print the `N` worst mismatch samples |

> **Note on `--scenario`:** the selector special-cases the group names
> `correctness`, `perf`, `decode`, and `all`. The current `shapes.json` groups
> are `decode` / `short_prefill` / `long_prefill` / `d256_disabled`, so
> `--scenario decode` selects the decode group and `--scenario all` (or no flag)
> runs everything; any other value is matched as an **exact shape name**, e.g.
> `--scenario fp16_h32kv8_b1_s2048x2048_d128`. To target the other regimes by
> group, prefer `final_shapes_check.py --groups short_prefill long_prefill`.

```bash
# Force the L4 (WG=64) flash geometry instead of the default wide4, on one shape:
HIPDNN_GFX942_FLASH_WIDE=0 PYTHONPATH=Python python \
    Python/rocke/examples/gfx942/attention/parity_unified_attention.py \
    --scenario fp16_h32kv8_b1_s2048x2048_d128
```

## The shape set (`shapes.json`)

Grouped by regime; every shape is causal (the kernel always applies a causal
mask), so non-causal *prefill* shapes are not included.

| group | what it covers |
|-------|----------------|
| `decode` | `seqlen_q == 1`, long KV (a single query attends to all keys) — fp16, D64/D128, MHA + GQA, batch 1–64 |
| `short_prefill` | 1–2 KV tiles (`seqlen_q` 64 / 512 / 528) — fp16/bf16, D64/D128, MHA + GQA |
| `long_prefill` | square prefill `seqlen_q == seqlen_k` up to 8192 — fp16/bf16, D64/D128, MHA + GQA, batch 1–51 |
| `d256_disabled` | `head_size = 256` — **off by default** (no tiled d256 path on gfx942 → scalar fallback; see below) |

## Arch notes

- **The D128-fp16 flash regime is the CDNA3 trick.** For `head_size = 128` fp16
  the harness routes to the transposed-x8 path (`use_mfma_32x32x8` +
  `use_transposed_qk_32x32`): it computes $S^\top = K Q^\top$ via the
  gfx942-legal `mfma_f32_32x32x8_f16` atom so that $P^\top$ stays
  **register-resident** as the PV B-operand — no `P_lds` round-trip. See
  [`ALGORITHM.md`](ALGORITHM.md) §6.2.
- **`use_mfma_32x32x8` is fp16-only** (gfx942 has no bf16 `32x32x8` atom), so
  D128 **bf16** uses the narrow `16x16x16` path, not the flash regime.
- **wide4 is the *provider's* analytic default, not the spec's.** A bare spec
  with no flash knobs lands on **L4** (WG=64); `parity_unified_attention.py` sets
  `num_warps=4` explicitly to reproduce the shipped peak. `HIPDNN_GFX942_FLASH_WIDE`
  selects between them: unset / `4` → wide4, `0` → L4, `2` → wide2 (WG=64).
- **The generic LDS support gate over-counts the flash path.**
  `supports_tiled_2d` models the narrow footprint (`P_lds` present, K
  double-buffered, full `Acc_lds`), which would wrongly reject the shipped wide4
  D128 config (whose $P^\top$ is register-resident). The harness applies the gate
  only to the narrow path and lets the spec's `__post_init__` + comgr arbitrate
  the flash configs, mirroring the provider.
- **Graph replay is overhead-driven.** The dispatcher's `_recommend_graph_replay`
  graphs decode (`max_seqlen_q == 1`) and short prefill (`<= 768`) — where host
  launch overhead is a large fraction — and ungraphs long prefill (kernel-bound).
  Toggle with `HIPDNN_GFX942_2D_GRAPH` / `HIPDNN_GFX942_3D_GRAPH` (both default-on).

### Lever environment variables

`parity_unified_attention.py` exposes the spec knobs through environment
variables (each is read by a small `_*_enabled()` / `_*_setting()` helper and
echoed at startup):

| variable | effect |
|----------|--------|
| `HIPDNN_GFX942_FLASH_WIDE` | flash-tile width for D128 fp16: unset/`4` = wide4 (WG=256), `0` = L4 (WG=64), `2` = wide2 (WG=64) |
| `HIPDNN_GFX942_NUM_WARPS` | override `num_warps` in the flash regime |
| `HIPDNN_GFX942_WAVES_PER_EU` | set the `waves_per_eu` occupancy hint |
| `HIPDNN_GFX942_CFV_STORE` | opt into the experimental conflict-free V store path |
| `HIPDNN_GFX942_CFV_STORE_SPLIT` / `HIPDNN_GFX942_CFV_CK_VLDS` | sub-knobs of the conflict-free V store (default-on) |
| `HIPDNN_GFX942_CFV` | the legacy gather-fill conflict-free V diagnostic path |
| `HIPDNN_GFX942_K_SLICED_RING` | the experimental sliced-K ring for the conflict-free-V-store path |
| `HIPDNN_GFX942_K_LDSSEQ` | the CK Tile LdsSeq sliced-K variant |
| `HIPDNN_GFX942_IGLP` | the `iglp_opt(1)` scheduling hint |
| `HIPDNN_GFX942_KV_CACHE_POLICY` | the KV-cache load policy (default `stream`) |
| `HIPDNN_GFX942_Q_DIRECT` | load Q directly from global (skip the Q LDS stage) |
| `HIPDNN_GFX942_GLOBAL_LOAD_LDS_K` | use direct global→LDS loads for K |
| `HIPDNN_GFX942_Q_MAJOR_GRID` | transpose the launch grid to query-major |

## Troubleshooting

- **`head_size = 256` is off by default.** There is no tiled d256 path on gfx942,
  so the dispatcher falls back to the scalar kernel, which is much slower than
  flash, fails the tolerance on some shapes, and is slow enough at `S2048` to
  stall graph capture (it can look like a hang). The `d256_disabled` group is
  skipped unless you pass `--groups d256_disabled` (and expect failures /
  slowness). A proper tiled d256 implementation is needed to revisit it.
- **Flash-ineligible shapes fall back to Torch's default SDP backend.** In
  `final_shapes_check.py`, non-square causal shapes (`seqlen_q != seqlen_k`) and
  d256 are rejected by AOTriton flash, so those rows are timed against Torch's
  *default* SDP backend (marked `default`) and excluded from the flash win-rate.
  For non-square shapes the fp32 reference uses a bottom-right-aligned causal mask
  while Torch's `is_causal=True` is top-left-aligned, so those rows reflect
  slightly different masking and are reported for context only.
- **Full sweeps are slow.** `final_shapes_check.py` runs the full set × 4 timing
  cells × `--reps` × `--iters`, plus the fp32 reference on large (up to `S8192`)
  shapes — tens of minutes on a single MI300X. Scope it with `--groups`,
  `--iters`, and `--reps`. Cells whose spread exceeds ~10% are flagged `noisy`.
- **`SKIP (unsupported on gfx942)`.** `parity_unified_attention.py` raises
  `NotImplementedError` (and skips) when a *narrow*-path shape fails the
  `supports_tiled_2d` LDS gate; flash-path shapes bypass the gate and are
  arbitrated by comgr instead.

## How the harness maps a dense SDPA problem onto the paged kernel

Each batch element becomes one sequence with
`(query_len, kv_len) = (seqlen_q, seqlen_k)`; the per-sequence block table is a
contiguous, **non-overlapping** run of `block_size = 64`-token cache blocks, so
the KV working set is genuinely per-sequence-distinct. Inputs are filled
`uniform(-0.1, 0.1)` to keep the softmax accumulation in a numerically friendly
range. The decode group uses a length-1 query attending to all keys
(`is_causal=False`, identical to causal at `q == 1`); prefill is causal. The
launch grid (`(kv_heads, total_num_q_blocks, 1)`, or query-major under
`HIPDNN_GFX942_Q_MAJOR_GRID`) and `block = (64 * num_warps, 1, 1)` are recomputed
exactly as the production dispatcher does, so the example exercises the same build
+ launch plumbing as the provider.

## Performance

This example is a correctness-and-latency *harness*, not a published benchmark:
absolute timings depend on the GPU, ROCm/torch versions, clocks, and thermal
state of the box, and are **not benchmarked into this folder**. Run
`final_shapes_check.py` to produce the CK-vs-Torch eager/graph table for your own
machine; the script prints, and gates on, the correctness result for every shape.
