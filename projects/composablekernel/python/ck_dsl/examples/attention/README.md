# CK DSL `unified_attention` parity & benchmark harness

This folder hosts the cross-backend parity + benchmark script for AITER's
`unified_attention` kernel. It is the canonical performance harness for
the CK DSL attention work.

The script (`parity_unified_attention.py`):

1. Builds the standard AITER unified-attention inputs (paged KV cache,
   block tables, cumulative query lengths, optional sliding window,
   softcap, sinks, ALiBi slopes, QQ-bias).
2. Runs the AITER **Triton** `unified_attention` in three modes:
   `auto` (Triton's own `use_2d_kernel` selector), `2d` (force Triton's
   2D kernel), `3d` (force Triton's 3D split-KV kernel). Forcing works
   by monkey-patching the `use_2d_kernel` callable that
   `unified_attention()` consults; it does not require modifying AITER.
3. Runs the **CK DSL** `run_unified_attention_torch` in matching modes
   (`backend="auto"`, `"tiled"`, `"3d"`).
4. Compares both backends' outputs to AITER's `ref_paged_attn` reference
   and to each other.
5. Emits three apples-to-apples tables: auto-vs-auto, 2D-vs-2D, 3D-vs-3D.

### Why three tables

CK DSL and Triton ship different selectors. Triton's `use_2d_kernel`
picks 2D for short `max_seqlen_k`, sliding window, or when the 2D grid
already saturates the device; CK DSL always prefers the 3D split-KV
path when supported. Without forcing, you'd be comparing Triton-2D vs
CK-3D, which is **not** apples-to-apples. The three tables resolve that:

* **`auto vs auto`** is the production-relevant comparison — what each
  backend actually launches.
* **`3d vs 3d`** is the algorithmically-fair comparison — same split-KV
  algorithm on both sides.
* **`2d vs 2d`** is the second algorithmically-fair comparison — same
  single-warp algorithm on both sides. CK DSL's 2D kernel is a
  single-warp single-CTA-per-(qblock, kv_head) design intentionally
  kept simple; it is **never** selected by `backend="auto"` and is
  noticeably slower than Triton's multi-warp 2D kernel. We include
  the column for completeness only.

## Running

```bash
cd /workspace/rocm-libraries-streaming/projects/composablekernel
sudo -n env \
  "PYTHONPATH=$(pwd)/python:/workspace/aiter" \
  /workspace/dsl_bake_off/venv/bin/python3 \
  python/ck_dsl/examples/attention/parity_unified_attention.py \
  --attempts 10 --warmup 5 \
  --report ck/dsl/unified_attention_parity.json
```

Flags:

| Flag | Default | Notes |
|------|---------|-------|
| `--scenario NAME` (repeatable) | all | restrict to the named scenarios |
| `--paths auto,2d,3d` | `auto,2d,3d` | which apples-to-apples lanes to run |
| `--attempts N` | `10` | timed iterations per lane (`torch.cuda.Event`-based) |
| `--warmup N`   | `3`  | untimed warmup iterations |
| `--skip-ck`    | off  | only run Triton (useful when CK is unavailable) |
| `--report PATH` | none | dump every measurement to JSON |

`sudo -n` is needed because the runner uses `libamd_comgr` and HIP
modules that require KFD ioctl permissions.

## Scenarios

The script ships eleven baseline scenarios in `default_scenarios()`. All
use `fp16` unless noted otherwise. The sequence-length pairs
`(q_len, kv_len)` mirror typical paged-KV decode + prefill workloads.

| Scenario | q lens / kv lens | dtype | b | d | extras |
|----------|------------------|-------|---|---|--------|
| `decode_d128_b16`             | 4 sequences, all q=1, kv ∈ {512, 1024, 2048, 4096}     | fp16 | 16 | 128 | – |
| `decode_d128_b64`             | same as above                                          | fp16 | 64 | 128 | – |
| `decode_d256_b16`             | 2 sequences, q=1, kv ∈ {1024, 2048}                    | fp16 | 16 | 256 | – |
| `prefill_d128_b16`            | (64, 64), (128, 256), (32, 256)                        | fp16 | 16 | 128 | – |
| `mixed_d128_b16`              | (1, 1328), (5, 18), (129, 463)                         | fp16 | 16 | 128 | – |
| `sliding_d128_b16`            | (1, 2048), (1, 4096), (1, 8192)                        | fp16 | 16 | 128 | sliding_window=256 |
| `softcap_d128_b16`            | (1, 1024), (1, 2048)                                   | fp16 | 16 | 128 | softcap=50 |
| `bf16_decode_d128_b64`        | (1, 1024), (1, 2048), (1, 4096)                        | bf16 | 64 | 128 | – |
| `alibi_decode_d128_b16`       | (1, 1024), (1, 2048), (1, 4096)                        | fp16 | 16 | 128 | ALiBi |
| `alibi_mixed_d128_b16`        | (1, 1328), (5, 18), (129, 463)                         | fp16 | 16 | 128 | ALiBi |
| `qq_bias_prefill_d128_b16`    | (64, 64), (128, 256), (32, 256)                        | fp16 | 16 | 128 | QQ-bias, stride=256 |

## Latest results (MI355X, gfx950, ROCm 7.0.1)

Numbers are the average per-launch latency from 10 timed iterations
after 5 warmup launches (single shot per timed call).

### Auto vs Auto — each backend's own selector

| Scenario                  | tri-auto | ck-auto  | speedup | tri-path | max_abs(CK vs ref) |
|---------------------------|---------:|---------:|--------:|---------:|-------------------:|
| decode_d128_b16           |  87.1us  |  36.8us  | **2.36x** | 3d | 1.83e-4 |
| decode_d128_b64           |  84.8us  |  33.6us  | **2.53x** | 3d | 1.83e-4 |
| decode_d256_b16           |  86.2us  |  35.5us  | **2.43x** | 3d | 1.22e-4 |
| prefill_d128_b16          |  52.5us  |  34.7us  | **1.51x** | 2d | 1.95e-3 |
| mixed_d128_b16            |  97.5us  |  33.5us  | **2.91x** | 3d | 9.77e-4 |
| sliding_d128_b16          |  63.5us  |  34.8us  | **1.82x** | 2d | 3.05e-4 |
| softcap_d128_b16          |  84.1us  |  34.4us  | **2.45x** | 3d | 1.22e-4 |
| bf16_decode_d128_b64      |  85.2us  |  36.6us  | **2.33x** | 3d | 9.77e-4 |
| alibi_decode_d128_b16     |  85.7us  |  35.9us  | **2.39x** | 3d | 9.77e-4 |
| alibi_mixed_d128_b16      |  87.8us  |  35.6us  | **2.47x** | 3d | 1.95e-3 |
| qq_bias_prefill_d128_b16  |  73.4us  |  34.8us  | **2.11x** | 2d | 1.95e-3 |

CK DSL beats Triton on every scenario, geomean speedup **≈2.3x**.
`max_abs(CK vs ref)` is the worst per-element error against the AITER
`ref_paged_attn` reference — all rows are within fp16/bf16 ULP. The
output is bit-identical to Triton's (`max_abs(CK vs Triton) == 0` once
both are cast back to the working dtype).

### 3D vs 3D — same split-KV algorithm on both backends

Force-flag rows. This is the algorithmically-honest comparison.

| Scenario                  | tri-3d   | ck-3d    | speedup |
|---------------------------|---------:|---------:|--------:|
| decode_d128_b16           |  84.1us  |  33.8us  | **2.49x** |
| decode_d128_b64           |  83.3us  |  33.9us  | **2.46x** |
| decode_d256_b16           |  83.9us  |  33.6us  | **2.50x** |
| prefill_d128_b16          |  94.7us  |  50.7us  | **1.87x** |
| mixed_d128_b16            |  83.5us  |  33.2us  | **2.51x** |
| sliding_d128_b16          |  84.2us  |  34.4us  | **2.45x** |
| softcap_d128_b16          |  84.7us  |  36.1us  | **2.35x** |
| bf16_decode_d128_b64      |  85.2us  |  33.0us  | **2.58x** |
| alibi_decode_d128_b16     |  83.1us  |  33.5us  | **2.48x** |
| alibi_mixed_d128_b16      |  87.7us  |  33.8us  | **2.59x** |
| qq_bias_prefill_d128_b16  |  85.6us  |  35.5us  | **2.41x** |

CK DSL wins 1.87x–2.59x on 10 of 11 scenarios. The win comes from the
CK Tile lessons we ported into the segment kernel:

- `ds_read_b64_tr_b16` for the PV operand using
  `TransposeLDSLayout<16,K>` lane formulas
- `ds_bpermute` 4-stage XOR butterfly for cross-lane softmax (matches
  CK Tile's `block_tile_reduce_xor_sync`)
- async DMA K/V with current-V-first + next-K-second issue order so PV
  only has to wait on the next-K stream
- specialised binary search trip count (ceil(log2(num_seqs+1)) instead
  of a fixed 32)
- 16-tile P_lds publish + `s_waitcnt(lgkmcnt=kv_calls_per_tile)`
  partial wait so K's LDS writes can overlap softmax

See
[`ck/dsl/unified_attention_results.md`](../../../ck/dsl/unified_attention_results.md)
for the full algorithm writeup.

**Variance note.** `alibi_mixed_d128_b16` contains one tiny sequence
(5 query tokens / 18 KV tokens) alongside two larger ones; with 16
split-KV segments per sequence the per-segment work for the small
sequence is below the kernel-launch overhead floor, so individual
launches in this row routinely vary 3-4x between attempts on this
GPU. Re-run the harness a few times for a stable median.

### 2D vs 2D — same single-warp algorithm on both backends

Triton's 2D kernel is multi-warp inside the CTA; CK DSL's tiled 2D
kernel is single-warp by design and is intentionally not optimized
because the 3D path already supports every problem the 2D path
supports. We never pick this path in `backend="auto"`. The table is
shown only for honesty.

| Scenario                  | tri-2d   | ck-2d    | speedup |
|---------------------------|---------:|---------:|--------:|
| decode_d128_b16           | 113.9us  | 641.6us  |  0.18x  |
| prefill_d128_b16          |  51.0us  | 598.3us  |  0.09x  |
| mixed_d128_b16            |  55.5us  | 630.3us  |  0.09x  |
| sliding_d128_b16          |  52.6us  | 601.0us  |  0.09x  |
| softcap_d128_b16          | 100.3us  | 685.3us  |  0.15x  |
| bf16_decode_d128_b64      |  89.4us  | 610.6us  |  0.15x  |
| alibi_decode_d128_b16     | 124.9us  | 627.2us  |  0.20x  |
| qq_bias_prefill_d128_b16  |  56.1us  | 633.5us  |  0.09x  |

Notes:

- `alibi_mixed_d128_b16` produces a NaN on CK 2D. Tracked as a
  known limitation of the single-warp fallback path; the auto
  selector never uses this kernel for that scenario.
- Adding a multi-warp variant of the 2D kernel is on the follow-up
  list, but not a priority — every problem currently routed to 2D in
  AITER is faster on our 3D kernel anyway.

## JSON report layout

Passing `--report PATH` writes a list of per-scenario records:

```jsonc
[
  {
    "scenario": "decode_d128_b16",
    "dtype": "torch.float16",
    "block_size": 16,
    "head_size": 128,
    "num_seqs": 4,
    "total_q": 4,
    "triton_auto_ms":    0.0871,
    "triton_auto_vs_ref": { "max_abs": 1.83e-4, "mean_abs": 6.5e-6, ... },
    "triton_natural_path": "3d",
    "ck_auto_ms":        0.0368,
    "ck_auto_vs_ref":    { "max_abs": 1.83e-4, ... },
    "ck_auto_vs_triton": { "max_abs": 0.0,    ... },
    "speedup_auto":      2.36,
    "triton_2d_ms":      0.1139,  "ck_2d_ms": 0.6416, "speedup_2d": 0.18,
    "triton_3d_ms":      0.0841,  "ck_3d_ms": 0.0338, "speedup_3d": 2.49,
    ...
  },
  ...
]
```
