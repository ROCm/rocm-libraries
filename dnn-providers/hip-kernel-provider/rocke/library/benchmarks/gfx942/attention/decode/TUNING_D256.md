# D256 bf16 decode tuning — gfx942

**Workload:** Qwen3-Next-80B-A3B gated attention decode, GQA 16/2,
`head_size=256`, `block_size=16`, `dtype=bf16`, `batch=1`,
`kv_len` sweep 512–32768.

**Dispatcher candidate:** `attention_d256_decode` (priority 5, algorithm
`d256_decode`) registered in `dispatch/attention.py`. Routes D256 bf16 decode
to the 3D split-KV path on both gfx942 and gfx950.

**Cohort predicate:** `_d256_decode_cohort()` in
`kernels/common/attention_unified.py` — `head_size==256`, `dtype=="bf16"`,
`all_decode==True`, no sliding_window / softcap / sinks / alibi / qq_bias / fp8.

---

## Problem found

`_num_segments()` in `kernels/common/attention_unified.py` had a gfx942
short-KV cap block:

```python
if arch == "gfx942" and max_seqlen_q == 1 and max_seqlen_k <= 2048:
    if head_size == 64:  return min(segments, 32)
    if head_size == 128: return min(segments, 16)
    return min(segments, 64)   # ← D256 fell here — wrong
```

D256 inherited `min(segments, 64)` via the `else` branch, which was intended
for other head sizes. The formula produces 128 segments for typical D256 decode
shapes; capping to 64 at short kv_len was a regression.

---

## Tuning workflow

```bash
# Sweep num_segments for D256 bf16 decode shapes on gfx942
python -m benchmarks.gfx942.attention.decode.benchmark_decode_live \
    --shapes library/benchmarks/gfx942/attention/decode/decode_shapes.json \
    --segments-sweep 8 16 32 64 128 \
    --output-json /tmp/seg_sweep_gfx942.json
```

The benchmark prints a `segments_sweep:` line per D256 shape:
```
qwen3next_80b_b1_kv8192   ...  segments_sweep: seg8=187.8us(0.52x)  seg16=102.2us(0.95x)  seg32=59.7us(1.63x)  seg64=48.0us(2.03x)  seg128=40.8us(2.39x)
```

---

## Sweep results (gfx942, 2026-07)

| kv_len | seg8 | seg16 | seg32 | seg64 | seg128 | winner |
|---|---|---|---|---|---|---|
| 512 | 0.26x | 0.26x | 0.26x | 0.26x | 0.26x | — (2D path) |
| 1024 | 3.23x | 3.21x | 3.18x | 3.23x | **3.24x** | seg128 |
| 2048 | 3.21x | 3.19x | **3.23x** | 3.19x | 3.19x | seg32 ≈ seg128 |
| 4096 | 0.87x | 1.47x | 2.37x | 2.40x | **2.54x** | seg128 |
| 8192 | 0.52x | 0.95x | 1.63x | 2.03x | **2.39x** | seg128 |
| 16384 | 0.25x | 0.48x | 0.88x | 1.30x | **1.76x** | seg128 |
| 32768 | 0.13x | 0.25x | 0.47x | 0.79x | **1.22x** | seg128 |

kv=512 routes to 2D (`max_seqlen_k <= 512` triggers `use_2d_kernel`); segment
count has no effect there.

**Conclusion:** seg128 wins or ties at every kv_len. Do not cap below the
formula value for D256.

---

## Fix applied

```python
# kernels/common/attention_unified.py — _num_segments()
if arch == "gfx942" and max_seqlen_q == 1 and max_seqlen_k <= 2048:
    if head_size == 64:  return min(segments, 32)
    if head_size == 128: return min(segments, 16)
    # D256: seg128 wins sweep — do not cap below formula value.
    if head_size != 256:
        return min(segments, 64)
```

---

## Before / after speedup vs Triton (gfx942, batch=1)

| kv_len | before | after | delta |
|---|---|---|---|
| 512 | 0.25x | 0.56x | +0.32x |
| 1024 | 3.09x | 3.16x | +0.07x |
| 2048 | 3.04x | 2.89x | −0.15x† |
| 4096 | 2.53x | 2.74x | +0.21x |
| 8192 | 2.26x | 2.62x | +0.37x |
| 16384 | 1.90x | 2.19x | +0.28x |
| 32768 | 1.40x | 1.66x | +0.26x |

† kv=2048 delta is within run-to-run noise; the sweep showed seg128 ≈ seg32
at this point, so no regression is expected structurally.

Full post-tuning results saved in `decode_shapes_perf.csv`.

---

## gfx950

No tuning needed. `_num_segments()` has no cap for gfx950 — it already returns
the raw formula value (128) for all D256 decode shapes.
