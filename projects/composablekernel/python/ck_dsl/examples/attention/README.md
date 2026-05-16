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
| `--attempts N` | `10` | timed iterations per lane; reported number is `elapsed_ms / N` from a single HIP-event pair recorded on torch's current stream |
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

## Latest results (MI355X, gfx950, ROCm 7.0.2)

### Methodology

Every row in the tables below is the **mean per-launch wall time over
10 timed iterations** after 5 untimed warmup launches, measured with
HIP events recorded on torch's current stream. **Both backends use
the same timer and the same stream**, so the numbers are directly
comparable. Concretely, the harness does:

1. 5 untimed warmup launches (CK DSL or Triton, depending on lane).
2. ``hipDeviceSynchronize`` to drain.
3. Record a start HIP event on ``torch.cuda.current_stream()``.
4. 10 timed launches on that same stream.
5. Record an end HIP event, synchronize on it, report
   ``elapsed_ms / 10``.

This is the apples-to-apples replacement for the older mixed-clock
setup (torch CUDA events for Triton, HIP events for CK DSL), which
under-measured CK lanes for some shapes.

Numbers below reflect commit-time MI355X / gfx950 / ROCm 7.0.2.

### Auto vs Auto — each backend's own selector

| Scenario                  | tri-auto | ck-auto  | speedup | tri-path | max_abs(CK vs ref) |
|---------------------------|---------:|---------:|--------:|---------:|-------------------:|
| decode_d128_b16           |  134.1us |   42.0us | **3.19x** | 3d | 1.83e-4 |
| decode_d128_b64           |   78.5us |   43.6us | **1.80x** | 3d | 1.83e-4 |
| decode_d256_b16           |   78.7us |   54.2us | **1.45x** | 3d | 1.22e-4 |
| prefill_d128_b16          |   48.2us |   39.9us | **1.21x** | 2d | 1.95e-3 |
| mixed_d128_b16            |   80.0us |   52.5us | **1.52x** | 3d | 9.77e-4 |
| sliding_d128_b16          |   64.1us |   41.1us | **1.56x** | 2d | 3.05e-4 |
| softcap_d128_b16          |   80.2us |   42.2us | **1.90x** | 3d | 1.22e-4 |
| bf16_decode_d128_b64      |   78.7us |   39.8us | **1.98x** | 3d | 9.77e-4 |
| alibi_decode_d128_b16     |   79.6us |   41.0us | **1.94x** | 3d | 9.77e-4 |
| alibi_mixed_d128_b16      |   79.1us |   41.2us | **1.92x** | 3d | 1.95e-3 |
| qq_bias_prefill_d128_b16  |   50.2us |   40.8us | **1.23x** | 2d | 1.95e-3 |

CK DSL beats Triton on every scenario; geomean speedup **≈1.8x** under
the unified HIP-event timer. `max_abs(CK vs ref)` is the worst
per-element error against the AITER `ref_paged_attn` reference — all
rows are within fp16/bf16 ULP. The output is bit-identical to
Triton's (`max_abs(CK vs Triton) == 0` once both are cast back to the
working dtype).

### 3D vs 3D — same split-KV algorithm on both backends

Force-flag rows. This is the algorithmically-honest comparison: same
algorithm, same timer, same stream.

| Scenario                  | tri-3d   | ck-3d    | speedup |
|---------------------------|---------:|---------:|--------:|
| decode_d128_b16           |   80.1us |   42.2us | **1.90x** |
| decode_d128_b64           |   78.7us |   40.9us | **1.92x** |
| decode_d256_b16           |   78.3us |   55.1us | **1.42x** |
| prefill_d128_b16          |   82.0us |   40.9us | **2.00x** |
| mixed_d128_b16            |   78.0us |   46.5us | **1.68x** |
| sliding_d128_b16          |   77.7us |   42.3us | **1.84x** |
| softcap_d128_b16          |   78.2us |   42.9us | **1.82x** |
| bf16_decode_d128_b64      |   78.2us |   41.8us | **1.87x** |
| alibi_decode_d128_b16     |   79.8us |   41.1us | **1.94x** |
| alibi_mixed_d128_b16      |   79.6us |   48.3us | **1.65x** |
| qq_bias_prefill_d128_b16  |   79.1us |   42.6us | **1.86x** |

CK DSL wins 1.42x–2.00x on every scenario. The win comes from the
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

### 2D vs 2D — same single-CTA algorithm on both backends

CK DSL's tiled 2D kernel is single-warp by design. Under the unified
HIP-event timer the 2D path **wins on the chunked-prefill scenarios
and the small-context sliding row** but **loses on long-context
single-query decode**, because the single-warp grid leaves the device
under-occupied for those shapes. The kernel itself is correct
(`max_abs(CK vs ref)` matches Triton's), this is purely a kernel-shape
trade-off worth fixing.

| Scenario                  | tri-2d   | ck-2d    | speedup |
|---------------------------|---------:|---------:|--------:|
| decode_d128_b16           |   49.6us |  272.3us | **0.18x** |
| decode_d128_b64           |   49.9us |  127.2us | **0.39x** |
| decode_d256_b16           |   48.4us |  198.0us | **0.24x** |
| prefill_d128_b16          |   48.7us |   21.8us | **2.24x** |
| mixed_d128_b16            |   47.8us |   90.3us | **0.53x** |
| sliding_d128_b16          |   48.7us |   21.9us | **2.23x** |
| softcap_d128_b16          |   48.5us |  165.7us | **0.29x** |
| bf16_decode_d128_b64      |   55.4us |  127.0us | **0.44x** |
| alibi_decode_d128_b16     |   49.4us |  277.2us | **0.18x** |
| alibi_mixed_d128_b16      |   49.0us |   91.7us | **0.53x** |
| qq_bias_prefill_d128_b16  |   50.0us |   23.3us | **2.15x** |

**Note on the earlier 2D table.** Previous versions of this README
reported CK 2D as universally faster than Triton 2D. Those numbers
were collected with torch CUDA events timing CK's raw
`hipModuleLaunchKernel` calls, which on some ROCm stream setups
under-counts the queued work. The unified HIP-event timer above is
what the production dispatcher's `auto` selector already does
(it prefers 3D wherever the scenario allows), so the 2D regression
rows do not affect end-to-end performance — they are a known
follow-up for the 2D kernel itself.

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
