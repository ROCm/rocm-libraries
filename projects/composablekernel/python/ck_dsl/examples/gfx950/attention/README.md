# CK DSL `unified_attention` parity & benchmark harness

This folder hosts the cross-backend parity + benchmark script for AITER's
`unified_attention` kernel. It is the canonical performance harness for
the CK DSL attention work.

> **New to flash attention or this kernel family?** [`ALGORITHM.md`](ALGORITHM.md)
> derives both kernels from the math up — the paged/varlen attention spec, the
> bias/mask order, the online-softmax core, and *why* the dispatcher chooses
> between the 2D (one CTA per q-block) and 3D split-KV (many CTAs share a
> q-block) paths on gfx950. Read it first if you want to understand *what* the
> kernels compute before reading the parity + optimization history below.

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
cd <composablekernel-checkout>
export AITER_PATH=<aiter-checkout>
PYTHONPATH="python:${AITER_PATH}" python \
  python/ck_dsl/examples/gfx950/attention/parity_unified_attention.py \
  --attempts 30 --warmup 10 \
  --report /tmp/unified_attention_parity.json
```

Flags (exactly as accepted by `parity_unified_attention.py`):

| Flag | Default | Notes |
|------|---------|-------|
| `--set {default,creative,fmha,all}` | `default` | which scenario set to use (see "Scenarios" below) |
| `--scenario NAME` (repeatable) | all | restrict to the named scenarios |
| `--paths auto,2d,3d` | `auto,2d,3d` | which apples-to-apples lanes to run |
| `--attempts N` | `10` | timed iterations per lane; reported number is `elapsed_ms / N` from a single HIP-event pair recorded on torch's current stream |
| `--warmup N`   | `3`  | untimed warmup iterations |
| `--skip-ck`    | off  | only run Triton (useful when CK is unavailable) |
| `--skip-triton` | off | only run CK DSL lanes (useful when AITER/Triton deps are unavailable) |
| `--report PATH` | none | dump every measurement to JSON |

`sudo -n` is needed because the runner uses `libamd_comgr` and HIP
modules that require KFD ioctl permissions.

## Scenarios

The `default` set in `default_scenarios()` ships **13** scenarios: the
**11 d128/d256 reference scenarios** below (all `fp16` unless noted
otherwise) plus **two bf16 d64/b32 GQA-8 "combo" cohort scenarios**
(`combo_bf16_d64_b32_gqa8_64x8`, `combo_bf16_d64_b32_gqa8_16x2`) that
exercise the transposed-32×32 combo stack — see the prefill-2D section
below. The sequence-length pairs `(q_len, kv_len)` mirror typical
paged-KV decode + prefill workloads. (Column `heads` below is the number
of query heads `num_query_heads`; every reference scenario uses
`num_query_heads=16` and `num_kv_heads=2`. The `b16`/`b64` suffix in the
scenario name refers to the paged-cache `block_size`, not the head
count.)

The 11 reference scenarios:

| Scenario | q lens / kv lens | dtype | heads | d | extras |
|----------|------------------|-------|-------|---|--------|
| `decode_d128_b16`             | 4 sequences, all q=1, kv ∈ {1024, 2048, 4096, 512}     | fp16 | 16 | 128 | – |
| `decode_d128_b64`             | same as above (block_size=64)                          | fp16 | 16 | 128 | – |
| `decode_d256_b16`             | 2 sequences, q=1, kv ∈ {1024, 2048}                    | fp16 | 16 | 256 | – |
| `prefill_d128_b16`            | (64, 64), (128, 256), (32, 256)                        | fp16 | 16 | 128 | – |
| `mixed_d128_b16`              | (1, 1328), (5, 18), (129, 463)                         | fp16 | 16 | 128 | – |
| `sliding_d128_b16`            | (1, 2048), (1, 4096), (1, 8192)                        | fp16 | 16 | 128 | sliding_window=256 |
| `softcap_d128_b16`            | (1, 1024), (1, 2048)                                   | fp16 | 16 | 128 | softcap=50 |
| `bf16_decode_d128_b64`        | (1, 1024), (1, 2048), (1, 4096)                        | bf16 | 16 | 128 | – |
| `alibi_decode_d128_b16`       | (1, 1024), (1, 2048), (1, 4096)                        | fp16 | 16 | 128 | ALiBi |
| `alibi_mixed_d128_b16`        | (1, 1328), (5, 18), (129, 463)                         | fp16 | 16 | 128 | ALiBi |
| `qq_bias_prefill_d128_b16`    | (64, 64), (128, 256), (32, 256)                        | fp16 | 16 | 128 | QQ-bias, stride=256 |

The two combo scenarios (`block_size=32`, `head_size=64`, `(512, 1024)`
× 2 sequences, bf16): `combo_bf16_d64_b32_gqa8_64x8` (64 query / 8 KV
heads) and `combo_bf16_d64_b32_gqa8_16x2` (16 query / 2 KV heads).

The results tables below cover the 11 reference scenarios. The two combo
scenarios share the d64/b32 GQA-8 trace family profiled in the
"Prefill-2D trace cohort" section.

Other scenario sets are selectable with `--set`: `creative` (21
exploratory scenarios — long-context decode up to 64K, GQA/MQA variants,
head_size=256, bf16, sliding-window extremes, bias combinations), `fmha`
(26 scenarios adapted from CK Tile's
`tile_engine/ops/fmha/ck_fmha_testing_matrix.yaml` subset that fits the
paged-attention constraints), and `all` (`default` + `creative`).

## Latest results (MI355X, gfx950, ROCm 7.2 / torch 2.12)

> **Re-baselined 2026-05-29 on this MI355X / gfx950 / ROCm 7.2 box.** The
> earlier 2026-05-28 tables reported CK DSL winning ~1.27x (auto) / ~1.28x
> (3d). On the current ROCm 7.2 / torch 2.12 stack, **AITER's Triton
> `unified_attention` has improved further and CK DSL now trails it**:
> geomean **0.88x (auto)** / **0.92x (3d)** across the eleven reference
> scenarios. CK DSL's absolute `ck-3d` times are essentially unchanged
> within noise (~56-58us); the ratio moved entirely because Triton's own
> kernels keep getting faster on newer ROCm. **Correctness is unchanged** —
> every row is still bit-exact vs Triton and within fp16/bf16 ULP vs the
> AITER reference (see the `max_abs(CK vs ref)` column).
>
> Historical context (still true): the original ROCm 7.0.2 tables reported
> ~1.8-2.8x geomean; the 2026-05-28 re-baseline noted two changes —
> (1) AITER's Triton `unified_attention` got ~2x faster on ROCm 7.2
> (decode `tri-auto` 125.8us -> ~57us), and (2) an `auto`-lane harness bug
> was fixed (`_run_ck_dsl` had force-built the 2D MFMA kernel for
> `path in ("auto","2d")`, mis-reporting decode `ck-auto` at ~282us; the
> `auto` lane now calls `run_unified_attention_torch(backend="auto")`, the
> real `select_path` dispatch). The production code was never affected;
> only the benchmark's `auto` measurement was.

### Methodology

Every row in the tables below is the **mean per-launch wall time over
30 timed iterations** after 10 untimed warmup launches, measured with
HIP events recorded on torch's current stream. **Both backends use
the same timer and the same stream**, so the numbers are directly
comparable. Concretely, the harness does:

1. 10 untimed warmup launches (CK DSL or Triton, depending on lane).
2. ``hipDeviceSynchronize`` to drain.
3. Record a start HIP event on ``torch.cuda.current_stream()``.
4. 30 timed launches on that same stream.
5. Record an end HIP event, synchronize on it, report
   ``elapsed_ms / 30``.

This is the apples-to-apples replacement for the older mixed-clock
setup (torch CUDA events for Triton, HIP events for CK DSL), which
under-measured CK lanes for some shapes.

The numbers below were produced with
``--attempts 30 --warmup 10`` and are the **per-cell median of 3 full
harness runs** on this box (the MI355X is perf-noisy under load, so the
median rejects the occasional outlier launch). Re-running may shift any
single cell by a few percent; the geomeans are stable.

### Auto vs Auto — each backend's own selector

| Scenario                  | tri-auto | ck-auto | speedup | tri-path | max_abs(CK vs ref) |
|---------------------------|---------:|--------:|--------:|---------:|-------------------:|
| decode_d128_b16           |   54.0us |  59.3us | **0.91x** | 3d | 1.83e-4 |
| decode_d128_b64           |   53.4us |  60.1us | **0.89x** | 3d | 1.83e-4 |
| decode_d256_b16           |   56.2us |  60.3us | **0.93x** | 3d | 1.22e-4 |
| prefill_d128_b16          |   28.5us |  33.5us | **0.85x** | 2d | 1.95e-3 |
| mixed_d128_b16            |   53.5us |  63.1us | **0.85x** | 3d | 9.77e-4 |
| sliding_d128_b16          |   28.8us |  33.2us | **0.87x** | 2d | 2.75e-4 |
| softcap_d128_b16          |   52.7us |  61.5us | **0.86x** | 3d | 1.22e-4 |
| bf16_decode_d128_b64      |   60.2us |  74.9us | **0.80x** | 3d | 9.77e-4 |
| alibi_decode_d128_b16     |   53.8us |  59.4us | **0.90x** | 3d | 9.77e-4 |
| alibi_mixed_d128_b16      |   53.7us |  55.6us | **0.96x** | 3d | 1.95e-3 |
| qq_bias_prefill_d128_b16  |   29.2us |  33.9us | **0.86x** | 2d | 1.95e-3 |

On this box CK DSL now **trails** Triton on every auto-selected scenario;
geomean **≈0.88x** on the current ROCm 7.2 / torch 2.12 stack. The headline
flipped from the previous report's ~1.27x win because Triton's own
`unified_attention` kept getting faster on newer ROCm while CK DSL's
absolute `ck-auto` times are unchanged within noise — see the re-baseline
note above.
`max_abs(CK vs ref)` is the worst per-element error against the AITER
`ref_paged_attn` reference — all rows are within fp16/bf16 ULP. The
output is bit-identical to Triton's (`max_abs(CK vs Triton) == 0`
once both are cast back to the working dtype).

### 3D vs 3D — same split-KV algorithm on both backends

Force-flag rows. This is the algorithmically-honest comparison: same
algorithm, same timer, same stream.

| Scenario                  | tri-3d   | ck-3d    | speedup |
|---------------------------|---------:|---------:|--------:|
| decode_d128_b16           |   54.9us |   58.0us | **0.95x** |
| decode_d128_b64           |   54.7us |   56.6us | **0.97x** |
| decode_d256_b16           |   54.4us |   57.8us | **0.94x** |
| prefill_d128_b16          |   52.9us |   57.6us | **0.92x** |
| mixed_d128_b16            |   53.5us |   61.9us | **0.87x** |
| sliding_d128_b16          |   53.0us |   57.9us | **0.92x** |
| softcap_d128_b16          |   54.6us |   57.7us | **0.95x** |
| bf16_decode_d128_b64      |   53.9us |   58.0us | **0.93x** |
| alibi_decode_d128_b16     |   55.3us |   69.3us | **0.80x** |
| alibi_mixed_d128_b16      |   54.8us |   55.1us | **0.99x** |
| qq_bias_prefill_d128_b16  |   54.1us |   56.7us | **0.95x** |

On this box CK DSL trails 0.80x–0.99x; geomean **≈0.92x** on the current
ROCm 7.2 / torch 2.12 stack (down from the old report's 1.28x purely
because Triton's own 3D kernel kept getting faster; CK DSL's `ck-3d`
absolute times are unchanged within noise, ~56-58us). The CK Tile lessons
ported into the segment kernel are still what keep CK DSL competitive:

- `ds_read_b64_tr_b16` for the PV operand using
  `TransposeLDSLayout<16,K>` lane formulas
- `ds_swizzle` (XOR-pattern immediate) intra-16-lane XOR butterfly for
  cross-lane softmax (matches CK Tile's `block_tile_reduce_xor_sync`;
  `warp_shuffle_xor` only falls back to `ds_bpermute` for the unused
  64-lane cross-half case)
- async DMA K/V with current-V-first + next-K-second issue order so PV
  only has to wait on the next-K stream
- specialised binary search trip count (ceil(log2(num_seqs+1)) instead
  of a fixed 32)
- 16-tile P_lds publish + `s_waitcnt(lgkmcnt=kv_calls_per_tile)`
  partial wait so K's LDS writes can overlap softmax

See [`ALGORITHM.md`](ALGORITHM.md) for the full kernel-strategy writeup
(the 2D vs 3D split-KV math, online softmax, bias/mask order, and the
CDNA mapping that motivates the optimizations above).

**Variance note.** `alibi_mixed_d128_b16` contains one tiny sequence
(5 query tokens / 18 KV tokens) alongside two larger ones; with 16
split-KV segments per sequence the per-segment work for the small
sequence is below the kernel-launch overhead floor, so individual
launches in this row routinely vary 3-4x between attempts on this
GPU. Re-run the harness a few times for a stable median.

### 2D vs 2D — same single-CTA algorithm on both backends

CK DSL's tiled 2D kernel is **single-warp per CTA** by design (Triton
2D uses 2-4 warps depending on the shape). Under the unified HIP-event
timer the 2D path **wins on the chunked-prefill scenarios and the
small-context sliding row** but **loses on long-context single-query
decode**, because the single-warp grid leaves the device under-occupied
for those shapes. The kernel itself is correct (`max_abs(CK vs ref)`
matches Triton's on every scenario, including the ALiBi / QQ-bias
ones — see the per-row column below) — this is purely a kernel-shape
trade-off; the auto selector already routes the slow shapes to 3D
(see the auto table).

| Scenario                  | tri-2d   | ck-2d    | speedup |
|---------------------------|---------:|---------:|--------:|
| decode_d128_b16           |   50.5us |  161.1us | **0.31x** |
| decode_d128_b64           |   53.3us |  334.3us | **0.16x** |
| decode_d256_b16           |   50.8us |  126.8us | **0.40x** |
| prefill_d128_b16          |   48.3us |   22.0us | **2.19x** |
| mixed_d128_b16            |   51.3us |   78.0us | **0.66x** |
| sliding_d128_b16          |   49.2us |   21.6us | **2.28x** |
| softcap_d128_b16          |   52.2us |  121.5us | **0.43x** |
| bf16_decode_d128_b64      |   51.5us |  332.9us | **0.15x** |
| alibi_decode_d128_b16     |   52.1us |  165.8us | **0.31x** |
| alibi_mixed_d128_b16      |   53.3us |   80.4us | **0.66x** |
| qq_bias_prefill_d128_b16  |   50.1us |   21.7us | **2.31x** |

Geomean **≈0.57x** (same as the previous report; the chunked-prefill
wins balance the decode losses). The auto-selector skirts the slow
rows by routing them to 3D where CK DSL has a clean 1.4-1.95x win.
The ALiBi / QQ-bias rows previously had a 2D-kernel correctness gap
(``max_abs(ck-2d vs ref)`` reached 2.4 in the worst case); the
transposed-32x32 softmax path now applies ALiBi (``slope * (key_pos
- context_len) * RCP_LN2``) and QQ-bias (``qq_bias[q_pos, key_pos -
context_len] * RCP_LN2``) inline before the per-row max reduce, so
all three scenarios are now within fp16 / bf16 ULP. The fix lives in the
transposed-32×32 softmax path of
``ck_dsl.instances.gfx950.attention_tiled_2d``.

**Note on the earlier 2D table.** Previous versions of this README
(pre v1) reported CK 2D as universally faster than Triton 2D. Those
numbers were collected with torch CUDA events timing CK's raw
`hipModuleLaunchKernel` calls, which on some ROCm stream setups
under-counts the queued work. The unified HIP-event timer above is
what the production dispatcher's `auto` selector already does
(it prefers 3D wherever the scenario allows), so the 2D regression
rows do not affect end-to-end performance — they are a known
follow-up for the 2D kernel itself.

## Prefill-2D trace cohort (the d64 / sinks production family)

The scenarios above are the d128 reference set. Real serving traces
(`aiter_unified_attention_*.jsonl`) hit a *different* family:
**head_size 64, block_size 32, GQA-8 (64 query / 8 KV heads), attention
sinks, sliding-window (127,0) or full, bf16 (or bf16-Q + fp8-KV)**, with
chunked prefill across 1..512 sequences. These all route to the 2D path.

A dedicated live-Triton workbench, `benchmark_prefill2d_live.py`, runs
AITER's Triton `unified_attention` (forced 2D) and the CK DSL variants on
the same stream/timer with a per-shape correctness check against Triton.

**The 2D dispatcher was substantially reworked (2026-05-28)** after this
workbench showed the production path was leaving ~40% on the table:

* The full transposed-32x32 **combo** (`s1mask` + `mask_once` +
  `half_local_pv` + `skip_legacy_qreg` + `mask_limit` + `fast_paged_kv_desc`)
  was benchmark-only — `_tiled_spec_from_problem` never set those flags. It
  is now wired into production via `_enable_combo_2d` for the validated
  family, **including attention sinks** (the transposed softmax folds the
  sink as the per-lane running-max init; the old gate refused sinks).
* A latent **mw=32 trap** was fixed: sinks shapes picked `block_m_per_warp=32`
  but then could not enable the 32x32 atoms, landing on plain 16x16 atoms
  with a doubled BLOCK_M (~1.4x slower than mw=16). mw=32 now requires the
  transposed/combo path or the fp8 path.
* **`waves_per_eu` tuning** for the combo family: the combo is VGPR-limited
  (~137 VGPR -> 3 WG/CU at the default wpe=2); wpe=3 reaches 4 WG/CU
  (a consistent **+15%**, no spills). wpe=4 adds another ~5% on
  full-attention shapes (used for no-SW combo; sliding-window keeps wpe=3
  to avoid an occupancy cliff).

Result on the bf16 trace cohort (142 deduped shapes, geomean speedup of
Triton-2d over CK-DSL-production; **>1.0 means CK DSL is faster**, all
shapes bit-accurate vs Triton, max_abs <= 3.9e-3):

| stage | geomean ck-prod speedup vs Triton-2d |
|-------|-------------------------------------:|
| before (stale dispatcher)                 | 0.44x |
| + combo wired in, mw=32 trap fixed        | 0.61x |
| + `waves_per_eu` tuning                    | 0.76x |
| + prelude-light SW combo (`nw2`/`T=BS`)    | 0.90x |
| + **measured at production paged-KV scale** | **1.11x** |

> **Benchmark-scale caveat — this is the big one.** The numbers above the
> last row used a small `cap_blocks` (8192) for the synthetic paged-KV
> cache, which makes the cache **artificially L2-resident**. Production
> caches have hundreds of thousands of blocks, so the KV working set far
> exceeds L2 and attention is **HBM-bandwidth-bound** — and that is
> exactly the regime where CK DSL's async-DMA KV loads beat Triton's. At a
> production-representative `cap_blocks=65536` the **bf16 cohort flips to
> 1.11x** (the harness default is now 65536). The small-cap regime was
> understating CK DSL by ~20%.

**At production scale (`cap_blocks=65536`), the bf16 prefill cohort is a
clean win — geomean 1.108x, 105/142 shapes beating Triton** (no-SW
1.118x, SW 1.099x), all bit-accurate. That is a **2.5x improvement** over
the original 0.44x. The advantage grows with KV working-set size (the
more HBM-bound, the bigger CK DSL's bandwidth edge). On the
**low-num-seqs** shapes CK DSL wins decisively (ns=1: **1.5–1.8x**).

### fp8 KV cache (bf16-Q + fp8-KV trace family)

The fp8 prefill cohort previously ran the plain 16×16 path (the 32×32
combo was hard-gated off for fp8) and sat at **~0.55–0.60x**. The combo
actually composes with fp8 for free: the **sync-dequant** loader already
writes bf16 into K/V LDS (k_scale folded in) — exactly what the bf16
32×32 reads expect — so the combo runs unchanged once
`use_fp8_mfma_qk` is off. Enabling it (the guard was conservative, not a
real limitation) lifts the fp8 prefill cohort to:

| fp8 prefill bucket (production scale, cap=131072) | before | after |
|--------------------|-------:|------:|
| sliding-window  | ~0.58x | **1.11x (37/37 win)** |
| full attention                    | ~0.34x | 0.87x |
| overall fp8 prefill               | ~0.50x | **0.98x (near parity)** |

(numbers from `prefill2d_fp8_triton_ckdsl_perf.csv`, 74-shape live sample
— 37 sliding-window + 37 full-attention, all bit-accurate vs Triton). The
two fp8 buckets behave very differently
with cache scale:

* **fp8 sliding-window is HBM-bound** (the window caps compute, many CTAs
  stream KV) — so like bf16 it crosses parity once the working set
  exceeds L2. fp8 KV is half the bytes of bf16, so it needs ~2x the cap
  to reach the same ~2 GB HBM-bound working set: SW = 0.97x at cap=65536
  but **1.13x at cap=131072 (33/33 shapes win)** — the proper apples-to-
  apples HBM-bound comparison with bf16-at-65536.
* **fp8 full attention is compute-bound** (q≈8000 attending causally to
  ~5000 keys = high arithmetic intensity), so it is *not* helped by cache
  scale and stays ~**0.82x** at any cap. The fp8→bf16 dequant VALU is the
  gap. We implemented and measured the obvious fix — a **native fp8×fp8
  QK MFMA** (no dequant) — but it is a lose-lose here: slower *and* less
  accurate (quantizing Q to fp8 costs ~1e-2), because even on this
  compute-bound shape the dequant is largely hidden behind the K/V load
  latency while the fp8 MFMA is no faster than bf16. So the accurate
  sync-dequant path remains the production choice; fp8 full attention is
  the one genuine holdout below parity.

fp8 SW uses `num_warps=4` (not bf16's `nw2`): fp8 SW is **dequant-bound**,
not prelude-bound, so it wants more warps to spread the fp8→bf16 dequant
(nw2 concentrates it and regresses). fp8 **decode** (`max_seqlen_q ≤ 256`)
keeps the validated 16×16 `use_fp8_mfma_qk` (K-in-LDS) path. Correctness
matches the bf16 combo (max_abs within fp8/bf16 ULP vs Triton).

## Paged-cache size: 64-bit KV addressing (resolved)

The tiled 2D kernel originally addressed the paged KV cache with a
hardware **32-bit buffer voffset**, and the per-access byte offset is
`physical_block * (block_size·num_kv_heads·head_size·dtype_bytes)`. That
product overflows i32 once the cache exceeds **2 GiB** (~65 K bf16 /
~131 K fp8 blocks); above the cap the loads wrapped and produced garbage
(verified: bf16 at 131 072 blocks gave `max_abs ≈ 1.4`). Production paged
caches are much larger (the captured traces have ~350 K blocks ≈ 11 GiB),
so this would have silently corrupted real deployments.

**Fixed, across all load paths, gated on cache size.** A `global_ptr_add`
primitive (IR + LLVM + HIP lowerings) plus two descriptor helpers
(`TensorDescriptor.offset_i64_split` for buffer loads, `offset_i64` for
flat global loads) fold `physical_block * stride` into **64-bit
addressing** so only a small within-block offset stays in the 32-bit
field:

* **bf16 no-SW** (fast paged-KV desc, buffer load) — per-block i64 buffer
  base (wave-uniform `make_buffer_rsrc`).
* **bf16 SW / general path** (`paged_kv_desc.offset_i64_split`) — same.
* **fp8 sync-dequant** (per-thread flat global load) —
  `paged_kv_desc.offset_i64` (full per-lane i64 element offset; the GEP
  index width now follows the operand type).
* **fp8-in-LDS** (decode) — per-block i64 buffer base.

The dispatcher enables it automatically (`_enable_i64_kv_addr`) only when
`num_kv_blocks × block_stride > 2³¹` (filled from `k.shape[0]`), so caches
≤2 GiB keep the exact fast i32 path (zero change, bf16 1.1x preserved) and
only larger caches pay the tiny per-block-base cost. Validated:

| cache | bf16 | fp8 |
|---|---|---|
| ≤2 GiB (i32) | 1.12x, correct | 0.98x, correct |
| >2 GiB (i64) | **correct** (was garbage), 1.06x | **correct** (was garbage), SW 1.10x |

all bit-accurate vs Triton (`max_abs ≤ 7.8e-3`). The HBM-bound speedup
now carries to production-scale (11 GiB) caches.

The sliding-window jump (0.67x → 0.91x) came from recognising SW prefill
is **prelude-bound**, not compute-bound: the window prunes the KV loop to
a handful of tiles, so the per-CTA prelude (Q→LDS load, binary search,
sink init) dominates. Switching the SW combo to a lighter geometry —
`num_warps=2` (BLOCK_M=64, half the Q-load prelude, 2x the CTAs for
latency hiding) and `tile_size = block_size` (finer window pruning) —
took SW from 0.67x to ~1.04x on the high-num-seqs bulk (bit-exact). The
no-SW combo, which is compute/occupancy-bound over a long KV loop, keeps
its `num_warps=4` / `T=2·BS` / fast-paged-KV geometry where it amortises
best.

### Kernel-body bottleneck (why the rest is hard)

Static ISA inspection (`probe_isa_inspect`) shows the combo 2D kernel is
**VALU/SALU-bound, not MFMA- or LDS-bound** — ~800 VALU + ~650 SALU vs
only **16 MFMA** per kernel, dominated by the per-element causal-mask
select (`v_cndmask`). Triton's 2D kernel is *algorithmically identical*
(same `find_seq_idx`, causal/sliding-window tile pruning, sink init), so
the gap is purely per-iteration code-gen. Two findings shaped the work:

1. **Occupancy is the right lever.** The combo is VGPR-limited (~137
   VGPR → 3 WG/CU at the default `waves_per_eu=2`); raising it lets the
   backend reach 4 WG/CU and hide the per-iter latency — hence the
   `waves_per_eu=3/4` win above.
2. **Instruction-count reduction is NOT.** Most causal no-SW KV tiles sit
   entirely below the causal limit and need no masking, so the loop was
   split into a full-tile phase (mask elided — provably bit-exact since
   `select(true, s, -inf) == s`) and a masked boundary phase. It was
   verified byte-identical but ran **~7% slower**: this kernel is
   latency/occupancy-bound, so duplicating the ~1100-line body across two
   loops cost more in I-cache / code size than the masking VALU it saved.
   Reverted. (A small algebraic masking reduction that needs no code
   duplication — folding `row_ok` into the threshold and pre-subtracting
   the compile-time row offset — was kept.)

The remaining gap is therefore an occupancy/latency problem: closing it
needs lower register pressure (fewer live PV accumulators — an
algorithmic redesign) or deeper K/V latency hiding, not fewer
instructions. CK DSL still trails Triton's well-tuned d64 2D
kernel on the high-num-seqs shapes — the remaining gap is per-iteration
code-gen efficiency in the 2D kernel body (Triton uses an algorithmically
identical kernel; the delta is scheduling/VALU, not the algorithm), which
is the open follow-up. On the **low-num-seqs** shapes CK DSL already wins
(e.g. ns=1: **1.5-1.8x**).

Sweep the live workbench over a set of shapes (best-correct CK DSL
variant per shape + bucket; writes a JSON to `--output-json`, default
`/tmp/prefill2d_live.json`):

```bash
export AITER_PATH=<path/to/aiter>
PYTHONPATH="python:${AITER_PATH}" python \
  python/ck_dsl/examples/gfx950/attention/benchmark_prefill2d_live.py \
  --shapes <path/to/unified_attention_shapes.jsonl> --variants prod combo fallback
```

Regenerate the joined cohort CSV (`prefill2d_bf16_triton_ckdsl_perf.csv`)
— this is written by `benchmark_prefill2d_traces.py`, which times the CK
DSL combo policy over the traced shapes and joins a pre-profiled Triton
CSV; the joined file is emitted to the path given by `--combined-csv`:

```bash
export AITER_PATH=<path/to/aiter>
PYTHONPATH="python:${AITER_PATH}" python \
  python/ck_dsl/examples/gfx950/attention/benchmark_prefill2d_traces.py \
  --shapes <path/to/unified_attention_shapes.jsonl> \
  --combined-csv prefill2d_bf16_triton_ckdsl_perf.csv
```

## File map

The CK DSL `unified_attention` kernels themselves live in `ck_dsl.instances`
(`gfx950/attention_tiled_2d.py`, `gfx950/attention_tiled_3d.py`,
`gfx950/attention_tiled_2d_fastkv_regp.py`, and the dispatcher
`common/attention_unified.py`). This folder holds the parity + benchmark
harnesses and their captured data.

| path | purpose |
|---|---|
| `README.md` | this document — parity methodology + prefill-2D optimization history + results |
| `ALGORITHM.md` | the math + kernel strategy (2D vs 3D split-KV, online softmax, bias/mask order, CDNA mapping) |
| `parity_unified_attention.py` | the canonical parity + benchmark harness: builds AITER paged-KV inputs, runs Triton and CK DSL in `auto`/`2d`/`3d` lanes on one shared HIP-event timer/stream, compares both to `ref_paged_attn`, emits the three apples-to-apples tables. Scenario sets: `default` (13 = 11 d128/d256 reference + 2 bf16 d64/b32 combo), `creative` (21, exploratory sweep), `fmha` (26, CK Tile testing-matrix subset), `all` (default + creative) |
| `benchmark_prefill2d_live.py` | the authoritative prefill-2D workbench: runs **live** Triton (forced 2D) vs a sweep of CK DSL 2D kernel variants (`prod`/`combo`/`fallback`/…) on the same stream, checks every variant against the Triton output, reports the best correct variant per shape and per bucket (sw/no-sw, bf16/fp8). Default `--cap-blocks 65536` (production-representative HBM-bound regime) |
| `benchmark_prefill2d_traces.py` | runs the CK DSL 2D combo policy over traced AITER prefill shapes and joins against a pre-profiled Triton CSV by `shape_signature` (the CSV-join workflow; writes `prefill2d_bf16_triton_ckdsl_perf.csv`) |
| `benchmark_prefill2d_fastkv_regp.py` | benchmarks the experimental `attention_tiled_2d_fastkv_regp` kernel (fast paged-KV + register-resident P) against the R4 / combo 2D baselines; `--smart-dispatch-policy latest` reproduces the measured-best per-shape host policy |
| `_d128_cktile_bakeoff.py` | per-shape, same-session A/B of CK DSL production `unified_attention` vs CK Tile `tile_example_fmha_fwd` (subprocess) and Triton, over a d128/d256 GQA-8 cohort; reports `cktile_ms / ckdsl_ms` (>1 = CK DSL faster) — requires a built `tile_example_fmha_fwd` binary |
| `_profile_one.py` | standalone single-shape launcher for `rocprofv3` profiling of the production-dispatched 2D combo kernel (d64/b32/GQA-8/sinks); args `<sw> <num_seqs> <iters>` |
| `prefill2d_bf16_triton_ckdsl_perf.csv` | captured bf16 prefill-2D cohort (142 deduped shapes; geomean **1.108x** vs Triton-2D at `cap_blocks=65536`, 105/142 wins) |
| `prefill2d_fp8_triton_ckdsl_perf.csv` | captured bf16-Q + fp8-KV prefill cohort (74 shapes; geomean **0.984x**; SW 1.108x 37/37, full-attention 0.874x) |
| `aiter_ua_shapes.json`, `aiter_ua_2_shapes.json`, `aiter_ua_prefill2d_allbf16.json` | captured AITER `unified_attention` call records (paged-KV shapes) used as benchmark inputs |

> The `benchmark_prefill2d_*.py` scripts load shapes via the in-tree shape
> utilities under
> `python/ck_dsl/dsl_docs/optimization/utilities/tools/stage1_benchmark`
> (`_ua_shape_utils.py`); pass `--shape-utils-path` to override.

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
    "triton_auto_ms":    0.1221,
    "triton_auto_vs_ref": { "max_abs": 1.83e-4, "mean_abs": 2.0e-5, ... },
    "triton_natural_path": "3d",
    "ck_auto_ms":        0.0435,
    "ck_auto_vs_ref":    { "max_abs": 1.83e-4, ... },
    "ck_auto_vs_triton": { "max_abs": 6.10e-5, ... },
    "speedup_auto":      2.82,
    "triton_2d_ms":      0.0517,  "ck_2d_ms": 0.2712, "speedup_2d": 0.19,
    "triton_3d_ms":      0.0793,  "ck_3d_ms": 0.0420, "speedup_3d": 1.89,
    ...
  },
  ...
]
```
