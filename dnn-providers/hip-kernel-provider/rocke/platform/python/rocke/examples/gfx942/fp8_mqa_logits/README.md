# FP8 MQA logits on gfx942

This example builds, launches, verifies, and benchmarks the rocKE FP8
multi-query-attention logits instance used by the DeepSeek lightning indexer.
It targets MI300-class `gfx942` devices and uses the native
`mfma_f32_16x16x32_fp8` instruction.

For query row `m` and KV position `n` in that row's valid window:

```text
logits[m, n] =
    sum_h ReLU(dot(Q[m, h, :], KV[n, :]) * kv_scale[n]) * weights[m, h]
```

`Q` and `KV` use native gfx942 E4M3 FNUZ encoding. `kv_scale` is expected to be
nonnegative, which permits the kernel to apply it once after the weighted head
sum. Positions outside `[cu_starts[m], cu_ends[m])` are left untouched; the
runner prefills the FP32 output with `-inf`.

## File map

| Path | Purpose |
|---|---|
| `fp8_mqa_logits_verify.py` | Shared build, launch, verification, and timing runner |
| `README.md` | Usage, implementation notes, and measured results |
| `rocke/instances/gfx942/fp8_mqa_logits.py` | Python kernel builder |
| `library/benchmarks/gfx942/fp8_mqa_logits/benchmark_live.py` | Live AITER comparison using this runner |
| `library/benchmarks/gfx942/fp8_mqa_logits/fp8_mqa_logits_perf.csv` | Captured results tabulated below |

The example imports `Fp8MqaLogitsSpec`, `build_fp8_mqa_logits`,
`fp8_mqa_logits_grid`, and `fp8_mqa_logits_signature` directly from the
instance. The live comparison imports its input, variant-selection, compile, and
launch helpers from this example, ensuring both paths exercise the same builder.

## Requirements

- AMD `gfx942` GPU
- PyTorch with ROCm and `torch.float8_e4m3fnuz`
- A working rocKE Python/COMGR/HIP environment
- AITER PR #3913 and FlyDSL only for the live comparison

Run from the rocKE directory:

```bash
cd <repo>/dnn-providers/hip-kernel-provider/rocke
export PYTHONPATH="$(pwd)/platform/python:${PYTHONPATH:-}"
```

## Build and verify

The default `4x128` shape is intentionally small enough for the row-at-a-time
PyTorch reference:

```bash
python -m rocke.examples.gfx942.fp8_mqa_logits.fp8_mqa_logits_verify --verify
```

Build, verify, and measure warm launch latency:

```bash
python -m rocke.examples.gfx942.fp8_mqa_logits.fp8_mqa_logits_verify \
  --shape 128x32768 \
  --bench --warmup 10 --iters 100 --repeats 7
```

Large production shapes should normally be verified against AITER with the live
comparison below; the PyTorch reference is deliberately row-at-a-time to avoid
materializing an `M x H x N` tensor and is not intended as a fast reference.

The example emits a `PerfJSON:` record when it finishes. This makes it directly
usable as a command for `rocke.benchmark.perf.harness`, including
`--verify --bench` correctness and wall-latency fields.

## Live AITER comparison

```bash
export AITER_PATH=<checkout containing PR 3913>
PYTHONPATH="$(pwd)/platform/python:${AITER_PATH}:${PYTHONPATH:-}" \
  python library/benchmarks/gfx942/fp8_mqa_logits/benchmark_live.py \
    --warmup 10 --iters 100 --repeats 7 \
    --output-csv /tmp/fp8_mqa_logits_gfx942.csv
```

For remote MI300X execution, stage and run the command through
`rocke.benchmark.remote_test` as described in
`platform/python/rocke/benchmark/remote_test/README.md`.

## Measured performance

Measured on one MI300X (`gfx942`) using Slurm job `67690637`. Each number is the
median of seven repeats with 100 timed iterations after 10 warmups. Both
implementations consumed identical tensors, used the same stream and HIP-event
timer, and included dense `-inf` output initialization. Software was PyTorch
2.10.0 with ROCm 7.2.4 and FlyDSL 0.2.2.

| Query x KV | AITER PR #3913 (ms) | rocKE (ms) | rocKE speedup | rocKE geometry |
|---:|---:|---:|---:|---|
| 4096 x 4096 | 0.2366 | 0.2165 | **1.093x** | `b64_r4_w2_wpe2_s2` |
| 8192 x 8192 | 0.7487 | 0.7139 | **1.049x** | `b64_r4_w2_wpe2_s1` |
| 128 x 32768 | 0.1325 | 0.1134 | **1.169x** | `b128_r2_w2_wpe2_s19` |
| 671 x 131072 | 1.8947 | 1.7280 | **1.097x** | `b64_r4_w4_wpe2_s18` |

The geometric-mean speedup is **1.101x**. Similarity error against AITER ranged
from `9.99e-16` to `1.22e-15`, below the `1e-3` correctness threshold.

These are point measurements, not CI performance guarantees. Re-run the live
benchmark on the target system when changing the compiler, ROCm, AITER, or the
shape-selection policy.

## Why this version is faster

- Several query rows share each KV fragment load.
- Q fragments and head weights remain in registers across the KV loop.
- Waves own disjoint KV-column tiles, requiring no cross-wave synchronization.
- The weighted ReLU accumulation uses an explicit FMA, reducing the hot
  `b64/r4/w2` kernel from 747 to 619 VALU instructions.
- Geometry and grid-y split density are selected by shape to balance KV reuse,
  register pressure, and CU occupancy.

The implementation uses no LDS and no scratch allocation. Its main remaining
tuning constraint is VGPR pressure: measured winning variants use 168-236
VGPRs, so larger KV tiles or more rows per block can reduce occupancy.
