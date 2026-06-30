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
PYTHONPATH="Python:${AITER_PATH}" python \
  Python/rocke/examples/gfx950/attention/parity_unified_attention.py \
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

The results tables below cover all 13 default scenarios (the 11 d128/d256
reference scenarios plus the two bf16 d64/b32 GQA-8 combos); the combo
scenarios also share the d64/b32 GQA-8 trace family profiled in the
"Prefill-2D trace cohort" section.

Other scenario sets are selectable with `--set`: `creative` (21
exploratory scenarios — long-context decode up to 64K, GQA/MQA variants,
head_size=256, bf16, sliding-window extremes, bias combinations), `fmha`
(26 scenarios adapted from CK Tile's
`tile_engine/ops/fmha/ck_fmha_testing_matrix.yaml` subset that fits the
paged-attention constraints), and `all` (`default` + `creative`).

## Latest results (MI355X, gfx950, ROCm 7.2 / torch 2.12)

> ### Measurement conditions (read first — every number below depends on these)
>
> * **Only same-session A/B ratios are load-bearing.** This MI355X
>   auto-clocks within roughly **±25-30%** (no clock-lock is available on this
>   node), so absolute microsecond values drift run-to-run and **cross-session
>   absolute ms are NOT comparable**. Every ratio in the tables below is a
>   `baseline_us / ck_us` computed from launches timed back-to-back in the
>   **same process / same HIP-event stream**; treat the raw `us` columns as
>   illustrative of one session, not as portable absolutes.
> * **Backend = llvm22 + comgr 7.2.** All numbers below were measured with
>   `torch` (which bundles comgr 7.2) imported *before* aiter so torch's 7.2
>   wins the `dlopen`. Verify with
>   `python -c "import torch; from rocke.runtime.comgr import resolved_lib_rocm_version as v; assert v()==(7,2)"`.
>   A system `comgr 7.0.1` (llvm20) is present on some boxes and, if it is
>   `dlopen`ed first (e.g. via a dependency imported before torch), it produces
>   a *different, slower* code-gen — **llvm20 measurements are not
>   representative and must not be mixed into these tables.**
> * **Two separate baselines, never blended.** We report CK DSL against
>   **(a) torch SDPA FLASH** and **(b) AITER Triton `unified_attention`**
>   independently. They are different kernels with different strengths; a single
>   blended "win" would be misleading. Where CK DSL wins one and trails the
>   other (notably the single-batch d128 prefill cohort: wins flash, trails
>   Triton-2D), both bars are stated explicitly.
> * **Re-measured 2026-06-19** on verified llvm22, `--attempts 30 --warmup 10`,
>   median of 3 same-session runs.

> **Re-baselined 2026-06-19 on this MI355X / gfx950 box, verified llvm22 /
> comgr 7.2** (torch imported first; `resolved_lib_rocm_version()==(7,2)`),
> median of 3 same-session runs. Against AITER's Triton `unified_attention`
> **on each backend's own selector**: geomean **1.04x (auto)** / **1.20x
> (3d)** across the thirteen default scenarios. The previous 2026-05-29
> report's **0.88x (auto) / 0.92x (3d)** was a *pre-campaign* snapshot; it
> predates the d128 routing rework landed since.
>
> What moved: the **3D split-KV path wins every reference scenario**
> (1.16x–1.27x — see the 3D table), so every scenario the `auto` selector
> routes to 3D (all the decode / full-context rows) is a clean win
> (decode 1.13–1.17x, d256 1.16x, mixed 1.22x, softcap 1.19x, alibi-decode
> 1.17x, alibi-mixed 1.21x). The `auto` rows that still trail are the ones
> the selector routes to the **2D kernel** (short chunked prefill 0.87–0.94x,
> sliding window 0.91x, the d64/b32 GQA-8 combos 0.73–0.89x) — below parity
> against Triton's well-tuned multi-warp 2D kernel and the open follow-up (see
> the 2D section and the single-batch d128 cohort note). **Correctness is
> unchanged** — every row is bit-exact vs Triton and within fp16/bf16 ULP vs
> the AITER reference (see the `max_abs(CK vs ref)` column).
>
> Historical context (still true): the original ROCm 7.0.2 tables reported
> ~1.8-2.8x geomean; an intermediate 2026-05-28 re-baseline noted two
> changes — (1) AITER's Triton `unified_attention` got ~2x faster on ROCm
> 7.2 (decode `tri-auto` 125.8us -> ~55us), and (2) an `auto`-lane harness
> bug was fixed (`_run_rocke` had force-built the 2D MFMA kernel for
> `path in ("auto","2d")`, mis-reporting decode `ck-auto` at ~282us; the
> `auto` lane now calls `run_unified_attention_torch(backend="auto")`, the
> real `select_path` dispatch). The production code was never affected; only
> the benchmark's `auto` measurement was. The 2026-06-19 campaign then
> reworked the 3D segment kernel and the single-batch d128 prefill routing
> (the occupancy lever described in the single-batch cohort note below),
> taking the 3D lane back above parity.

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

The numbers below were produced on **2026-06-19** with
``--attempts 30 --warmup 10`` and are the **per-cell median of 3 full
harness runs** on this box (the MI355X is perf-noisy under load, so the
median rejects the occasional outlier launch). Re-running may shift any
single cell by a few percent; the geomeans are stable.

### Auto vs Auto — each backend's own selector

| Scenario                     | tri-auto | ck-auto | speedup | tri-path | max_abs(CK vs ref) |
|------------------------------|---------:|--------:|--------:|---------:|-------------------:|
| decode_d128_b16              |   54.5us |  48.0us | **1.13x** | 3d | 1.83e-4 |
| decode_d128_b64              |   54.4us |  47.1us | **1.17x** | 3d | 1.83e-4 |
| decode_d256_b16             |   54.3us |  46.9us | **1.16x** | 3d | 1.22e-4 |
| prefill_d128_b16           |   28.9us |  33.0us | 0.87x | 2d | 1.95e-3 |
| mixed_d128_b16              |   54.0us |  44.5us | **1.22x** | 3d | 9.77e-4 |
| sliding_d128_b16           |   28.8us |  32.1us | 0.91x | 2d | 2.75e-4 |
| softcap_d128_b16            |   54.2us |  45.4us | **1.19x** | 3d | 1.22e-4 |
| bf16_decode_d128_b64        |   54.2us |  47.6us | **1.16x** | 3d | 9.77e-4 |
| alibi_decode_d128_b16       |   55.1us |  47.1us | **1.17x** | 3d | 9.77e-4 |
| alibi_mixed_d128_b16       |   55.7us |  45.9us | **1.21x** | 3d | 1.95e-3 |
| qq_bias_prefill_d128_b16   |   29.9us |  32.0us | 0.94x | 2d | 1.95e-3 |
| combo_bf16_d64_b32_gqa8_64x8 |   35.3us |  48.6us | 0.73x | 2d | 7.81e-3 |
| combo_bf16_d64_b32_gqa8_16x2 |   29.7us |  33.4us | 0.89x | 2d | 5.86e-3 |

Geomean **≈1.04x (auto)** — re-measured 2026-06-19 on verified llvm22 /
comgr 7.2, median of 3 same-session runs. **Every scenario the selector
routes to the 3D split-KV path wins** (decode 1.13–1.17x, mixed 1.22x,
softcap 1.19x, alibi-decode 1.17x, alibi-mixed 1.21x, bf16-decode 1.16x,
d256 1.16x — see the 3D table for the algorithmically-fair view). The rows
that still trail are the ones the selector routes to the **2D kernel**
(chunked prefill 0.87–0.94x, sliding window 0.91x, the d64/b32 GQA-8 combos
0.73–0.89x) — the same 2D-kernel-body code-gen gap discussed in the 2D
section and the single-batch d128 cohort note below; the auto selector keeps
the won shapes on 3D. (Two earlier single-run cells — d256 0.97x and
alibi_mixed 0.94x — were re-measured here as 1.16x / 1.21x; they were
single-launch outliers, not real losses.)
`max_abs(CK vs ref)` is the worst per-element error against the AITER
`ref_paged_attn` reference — all rows are within fp16/bf16 ULP. The
output is bit-identical to Triton's (`max_abs(CK vs Triton) == 0`
once both are cast back to the working dtype).

### 3D vs 3D — same split-KV algorithm on both backends

Force-flag rows. This is the algorithmically-honest comparison: same
algorithm, same timer, same stream.

| Scenario                     | tri-3d   | ck-3d    | speedup |
|------------------------------|---------:|---------:|--------:|
| decode_d128_b16              |   55.7us |   47.7us | **1.17x** |
| decode_d128_b64              |   55.0us |   46.5us | **1.17x** |
| decode_d256_b16             |   56.1us |   46.9us | **1.20x** |
| prefill_d128_b16           |   53.8us |   46.3us | **1.16x** |
| mixed_d128_b16             |   56.0us |   46.1us | **1.18x** |
| sliding_d128_b16          |   54.1us |   44.8us | **1.21x** |
| softcap_d128_b16           |   54.8us |   45.2us | **1.21x** |
| bf16_decode_d128_b64       |   54.7us |   44.9us | **1.22x** |
| alibi_decode_d128_b16      |   56.4us |   45.2us | **1.25x** |
| alibi_mixed_d128_b16      |   55.7us |   44.7us | **1.27x** |
| qq_bias_prefill_d128_b16  |   55.6us |   44.5us | **1.25x** |
| combo_bf16_d64_b32_gqa8_64x8 |  164.3us |  138.9us | **1.17x** |
| combo_bf16_d64_b32_gqa8_16x2 |   54.8us |   47.9us | **1.16x** |

CK DSL **wins every row, 1.16x–1.27x**; geomean **≈1.20x** — re-measured
2026-06-19 on verified llvm22 / comgr 7.2, median of 3 same-session runs.
This is the algorithmically-honest comparison (same split-KV algorithm, same
timer, same stream), and the campaign's rework of the 3D segment kernel is
what carries it. The CK Tile lessons ported into the segment kernel are what
keep CK DSL ahead:

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

| Scenario                     | tri-2d   | ck-2d    | speedup |
|------------------------------|---------:|---------:|--------:|
| decode_d128_b16             |  112.6us |  270.8us | **0.42x** |
| decode_d128_b64             |   63.2us |  316.1us | **0.20x** |
| decode_d256_b16            |   60.9us |  258.6us | **0.24x** |
| prefill_d128_b16          |   29.3us |   23.3us | **1.26x** |
| mixed_d128_b16             |   29.8us |   92.3us | **0.32x** |
| sliding_d128_b16          |   29.5us |   22.9us | **1.29x** |
| softcap_d128_b16           |   91.6us |  222.6us | **0.41x** |
| bf16_decode_d128_b64       |   63.3us |  316.2us | **0.20x** |
| alibi_decode_d128_b16      |  102.0us |  285.2us | **0.36x** |
| alibi_mixed_d128_b16      |   31.2us |   95.7us | **0.33x** |
| qq_bias_prefill_d128_b16  |   30.8us |   27.7us | **1.11x** |
| combo_bf16_d64_b32_gqa8_64x8 |   34.4us |   49.2us | **0.70x** |
| combo_bf16_d64_b32_gqa8_16x2 |   29.9us |   42.6us | **0.70x** |

Geomean **≈0.47x** (the chunked-prefill wins balance the decode losses;
the single-warp 2D kernel is intentionally simple and never auto-selected
for the decode shapes). The auto-selector skirts the slow
rows by routing them to 3D where CK DSL has a clean 1.13–1.23x win.
The ALiBi / QQ-bias rows previously had a 2D-kernel correctness gap
(``max_abs(ck-2d vs ref)`` reached 2.4 in the worst case); the
transposed-32x32 softmax path now applies ALiBi (``slope * (key_pos
- context_len) * RCP_LN2``) and QQ-bias (``qq_bias[q_pos, key_pos -
context_len] * RCP_LN2``) inline before the per-row max reduce, so
all three scenarios are now within fp16 / bf16 ULP. The fix lives in the
transposed-32×32 softmax path of
``rocke.instances.gfx950.attention_tiled_2d``.

**Note on the earlier 2D table.** Previous versions of this README
(pre v1) reported CK 2D as universally faster than Triton 2D. Those
numbers were collected with torch CUDA events timing CK's raw
`hipModuleLaunchKernel` calls, which on some ROCm stream setups
under-counts the queued work. The unified HIP-event timer above is
what the production dispatcher's `auto` selector already does
(it prefers 3D wherever the scenario allows), so the 2D regression
rows do not affect end-to-end performance — they are a known
follow-up for the 2D kernel itself.

## Single-batch d128 long-prefill cohort (the K-single-buffer occupancy lever)

The single-batch (`num_seqs == 1`) **d128 full-prefill** family — GQA-8
(64 query / 8 KV heads), bf16 + fp16, `S = q = kv ∈ {1024, 2048, 4096}` —
was historically the weakest CK DSL path. The campaign's **occupancy lever**
reversed the loss **against torch SDPA flash** (it now wins flash at
short/mid context); it does **not** close the gap to Triton's 2D kernel,
which stays ~0.55-0.60x and is the open follow-up. The d128 transposed-32×32
combo at `num_warps=2` is purely
**LDS-bound**: at the default `T = 2·block_size = 64` the K double-buffer +
V single-buffer LDS (`K_lds[2,T,HD] + V_lds[1,T,HD] = 48 KB`) leaves only
**1 workgroup per CU** even though the register file already admits two.
Keeping the larger `T = 64` tile (good long-context per-iter amortization)
but **halving K_lds with K single-buffer** (`use_k_single_buffer`, K_lds[1]
16 KB + V_lds[1] 16 KB = **32 KB → 2 WG/CU**, VGPR=215 AGPR=0), with **V
double-buffer OFF** (a net drag at d128 that would re-inflate LDS back to
1 WG/CU), is what closes the gap. The next-K prefetch is re-issued *after*
the PV-wait barrier so the single K slot cannot WAR-race.

Re-measured 2026-06-19 on **verified llvm22 / comgr 7.2** (torch imported
first; `resolved_lib_rocm_version()==(7,2)`) with the parity-harness
methodology (same `--attempts 30 --warmup 10`, same-session HIP-event
timer/stream), median of 3 same-session runs, all bit-accurate (`max_abs`
within fp16/bf16 ULP — bf16 ≤ 1.6e-2, fp16 ≤ 2.0e-3). Absolute `us` are from
one session and are **not** cross-session comparable (±25-30% auto-clock); the
ratio columns are the load-bearing numbers.

**vs torch SDPA FLASH** (the baseline the K-single-buffer occupancy lever was
tuned against — this is the win bar for this cohort):

| Scenario (single-seq, GQA-8, d128) | flash | ck-auto | ck/flash |
|------------------------------------|------:|--------:|---------:|
| bf16 S1024 |  102.6us |   75.7us | **1.36x** |
| bf16 S2048 |  230.6us |  227.0us | **1.02x** |
| bf16 S4096 |  725.4us |  761.9us | 0.95x |
| fp16 S1024 |  102.5us |   75.6us | **1.36x** |
| fp16 S2048 |  231.1us |  226.7us | **1.02x** |
| fp16 S4096 |  726.2us |  762.1us | 0.95x |

Geomean **≈1.10x over flash.** The lever wins decisively at short prefill
(S1024 1.36x) and is at parity at S2048 (1.02x), but at the long-context
**S4096 holdout it slips just under flash (0.95x)** — flash's larger fixed Q
tile amortizes the long KV loop better than our `BLOCK_M=32` tile. That
S4096 dip is an honest loss vs flash, reproduced on llvm22.

**vs Triton `unified_attention` (forced 2D, the path `auto` routes these to)
— the honest residual (this cohort loses to Triton):**

| Scenario (single-seq, GQA-8, d128) | tri-2d | ck-auto | ck/tri-2d |
|------------------------------------|-------:|--------:|----------:|
| bf16 S1024 |   45.1us |   75.7us | 0.60x |
| bf16 S2048 |  125.6us |  227.0us | 0.55x |
| bf16 S4096 |  408.0us |  761.9us | 0.54x |
| fp16 S1024 |   45.4us |   75.6us | 0.60x |
| fp16 S2048 |  128.3us |  226.7us | 0.57x |
| fp16 S4096 |  423.8us |  761.9us | 0.56x |

Geomean **≈0.57x vs Triton — a real, persistent loss, confirmed on llvm22**
(it is *not* a stale-backend artifact). Triton's multi-warp 2D prefill kernel
is ~2x faster than torch flash on these exact shapes, so this cohort can win
against flash yet still trail Triton by ~0.55-0.60x. The K-single-buffer
occupancy lever is what reversed the *flash* loss; the Triton gap is the
2D-kernel-body VALU/SALU + MFMA-scheduling code-gen gap discussed below and in
the prefill-2D section, and it remains the open 2D-kernel follow-up. The auto
selector keeps this cohort on the K-single-buffer `BLOCK_M=32` combo because
that is the configuration that wins flash; it does not pretend to beat
Triton-2D here.

### Exhaustive microlever sweep — the swept ceiling still trails Triton

> **Conditions for this subsection and the BLOCK_M=128 one below.** The
> *timing* numbers in these two investigation subsections come from a separate
> microlever-sweep campaign (same-session HIP-event timed). The shipped-config
> result re-measured above on verified llvm22 (≈0.57x vs Triton-2D)
> **corroborates** the swept ceiling conclusion (≈0.59x): even the best
> swept config does not close the Triton gap, so the conclusion is robust on
> llvm22. The VGPR / LDS / spill / waves-per-SIMD figures in the budget tables
> are static compiler-counter facts (backend-independent), not latencies.

To test whether the shipped heuristic simply mis-picked, we ran an
**exhaustive cartesian microlever sweep** over the *entire* current
gfx950 tiled-2D lever space for this cohort (≈2 000–2 900 valid configs per
shape, batch-compiled and correctness-pruned against an fp32 paged-attn
reference, then same-session HIP-event timed vs both Triton-2D and torch
flash). The swept axes: `num_warps ∈ {1,2,4,8}` × `block_m_per_warp ∈
{16,32}` (so `BLOCK_M ∈ {16…128}`) × `tile_size ∈ {block_size … 256}` ×
`waves_per_eu ∈ {None,1,2,3,4}` × the full transposed-32×32 combo stack and
its sub-flags × `use_k_single_buffer` × `use_v_double_buffer` ×
`use_staggered_iter_wait` × `kv_ring_depth ∈ {2,3}` ×
`use_early_v_schedule` × `use_sched_barrier` × `use_register_pv` ×
`use_q_reread` × `compile_backend ∈ {llvm,hipcc}`.

The **best config the sweep can find per shape** (the achievable ceiling),
vs both baselines:

| Scenario (single-seq, GQA-8, d128) | tri-2d | flash | swept-best | best/tri2d | best/flash |
|------------------------------------|-------:|------:|-----------:|-----------:|-----------:|
| bf16 S1024 |  45.6us | 102.4us |  65.5us | 0.70x | **1.56x** |
| bf16 S2048 | 126.3us | 229.2us | 223.9us | 0.56x | **1.02x** |
| bf16 S4096 | 402.0us | 722.5us | 795.4us | 0.51x | 0.91x |
| fp16 S1024 |  46.2us | 102.3us |  65.5us | 0.71x | **1.56x** |
| fp16 S2048 | 133.1us | 229.7us | 224.1us | 0.59x | **1.03x** |
| fp16 S4096 | 417.2us | 723.2us | 803.1us | 0.52x | 0.90x |

The sweep also covered a **single-batch d256 prefill** shape (head_size
256, GQA-8, S2048, fp16) as a control: there the swept-best **crosses
Triton at 1.069x** (175.9us vs tri-2d 188.0us, also 1.02x over flash,
bit-accurate) — d256 is *not* in the same structural hole as d128, so the
d128 residual below is specific to the d128 head geometry, not a blanket
2D-kernel weakness.

All swept-best configs are bit-accurate (`max_abs ≤ 9.9e-3` vs fp32 ref).
The swept best is the **same K-single-buffer transposed-32×32 combo family
at `num_warps=1`, `BLOCK_M=32`, `tile_size=block_size`** that the shipped
heuristic already selects — the heuristic is *not* mis-picking. **The full
sweep does not close the Triton gap:** geomean ceiling **≈0.59x vs Triton**
for the d128 cohort, and the gap *widens* with sequence length (0.70x at
S1024 → 0.51x at S4096), which is the signature of a per-row-amortization
difference, not a config choice. (At S4096 the swept best even slips
slightly below flash — flash's larger fixed tile amortizes the long KV loop
that our small `BLOCK_M=32` tile cannot.)

### The missing technique: Triton's BLOCK_M=128 / 4-warp 2D geometry at sub-ceiling VGPR

Reverse-engineering Triton's gfx950 2D prefill kernel (its
`select_2d_config` + a disassembly of the compiled instance) shows that for
`max_seqlen_q ≥ 256` Triton launches **`BLOCK_M=128`, `TILE_SIZE=64`,
`num_warps=4`, `num_stages=1`** — it processes a **128-row Q tile across 4
warps** per CTA, amortizing the per-CTA prelude (Q→LDS load, `find_seq_idx`
binary search, sink/mask init) and the causal-mask VALU across **4× more
rows** than our `BLOCK_M=32` single-warp winner. The compiled Triton
instance uses the *same* core ISA primitives our 3D path already ships —
`v_mfma_f32_32x32x16_bf16` (16 QK + 16 PV MFMAs), `ds_read_b64_tr_b16`
transpose LDS reads, `v_permlane32_swap_b32` cross-lane softmax — so the
*algorithm and atoms are identical*; the delta is purely the 2D geometry.

The lever space **does** include `BLOCK_M=128` (`num_warps=4`,
`block_m_per_warp=32`), and the sweep tested it — but it **loses to
`BLOCK_M=32`**, and a direct same-session A/B (S2048 bf16) shows why:

| our config | BLOCK_M | warps | VGPR | LDS | latency |
|------------|--------:|------:|-----:|----:|--------:|
| `nw1` (swept winner) | 32 | 1 | 177 | 16 KB | 224.8us |
| `nw2` | 64 | 2 | 215 | 32 KB | 208.6us |
| `nw4` (Triton's geometry) | 128 | 4 | **256 (ceiling)** | 49 KB | 236.7us |
| **Triton** `nw4` | **128** | 4 | **210, 0 spills** | **32 KB** | **132.3us** |

The root cause is now precise: **at `BLOCK_M=128` our transposed-32×32
combo hits the 256-VGPR hardware ceiling and 49 KB LDS** (PV-accumulator +
softmax-state + Q-hold register layout does not scale to 128 rows),
so it is pinned at 1 workgroup/CU at the register limit — *slower* than our
own `BLOCK_M=32`. Triton fits the **same** `BLOCK_M=128` in **210 VGPR with
zero spills and 32 KB LDS**, reaching higher occupancy. So the gap is
**not** a missing config or scheduling flag — it is that the DSL's 2D
combo's accumulator/register layout cannot express a `BLOCK_M=128` tile
within Triton's VGPR/LDS budget.

**The named follow-up kernel feature — a VGPR-frugal `BLOCK_M=128`
transposed-2D body — was built and measured. It reaches Triton's exact
hardware budget but does *not* close the latency gap; the residual is now
isolated to the inner-loop MFMA schedule.**

The feature combines two existing levers that, together, lift the
`BLOCK_M=128` (`num_warps=4`) combo into Triton's budget:

* `use_q_direct_reg` — Q is gathered straight from global into VGPRs and
  never aliases the K LDS slot, which (a) lifts the `block_m ≤ tile_size`
  constraint that previously blocked `BLOCK_M=128` + `T=64` from using a
  single K buffer, and (b) drops peak VGPR.
* `use_k_single_buffer` — single K LDS slot, so `K_lds[1]=16 KB +
  V_lds[1]=16 KB = 32 KB`.

Measured static budget of the resulting `BLOCK_M=128` body vs Triton (same
shape):

| | BLOCK_M | warps | VGPR | spills | LDS | waves/SIMD |
|--|--:|--:|--:|--:|--:|--:|
| old `BLOCK_M=128` combo | 128 | 4 | **256 (ceiling)** | 3 | 49 KB | 2 |
| **new frugal `BLOCK_M=128`** | 128 | 4 | **213** | **0** | **32 KB** | **2** |
| Triton `BLOCK_M=128` | 128 | 4 | 210 | 0 | 32 KB | 2 |

So the new body is **register- and occupancy-matched to Triton** (213 vs
210 VGPR, 0 spills, identical 32 KB LDS and 2 waves/SIMD), and bit-accurate
(`max_abs ≤ 9.9e-3`). Two further inner-loop levers were added on top: a
**mask-phase split** (`use_mask_phase_split` — peels the causal-full tiles
into a no-mask phase so the per-element `v_cndmask` VALU is emitted only for
boundary tiles; the previously-reverted experiment, re-enabled now that the
kernel is VALU- rather than occupancy-bound) and the **softmax↔MFMA
interleave hint** (`use_softmax_mfma_interleave`).

**Honest outcome — the gap does not close.** Despite matching Triton's
budget, the best `BLOCK_M=128` config (frugal + mask-split + interleave)
measures **≈0.55–0.63x vs Triton** across S1024–S4096 — essentially tied
with the shipped `BLOCK_M=32` combo, not Triton parity:

| shape (bf16) | shipped `BM32` | new frugal `BM128` (+split+iglp) | Triton |
|--------------|---------------:|---------------------------------:|-------:|
| S1024 | 0.66x | 0.63x | 1.00x |
| S2048 | 0.54x | 0.56x | 1.00x |
| S4096 | 0.50x | 0.55x | 1.00x |

The new body edges ahead at long context (S4096: 0.55x vs 0.50x) but trails
at short context, so it is **not** a strict win and is **not** wired into
the production selector — it ships as opt-in spec flags
(`use_q_direct_reg`, `use_k_single_buffer`, `use_mask_phase_split`,
`use_softmax_mfma_interleave`), all default-OFF and golden-safe.

**The residual is the MFMA schedule, not the geometry.** With occupancy and
register budget now identical to Triton, a static ISA comparison shows the
gap is the inner-loop instruction schedule: our body's largest MFMA-idle
window (the softmax / causal-mask / exp2 VALU stretch *between* the QK-MFMA
and PV-MFMA clusters) is **≈187 instructions vs Triton's ≈155**, and our
body still emits substantially more causal-mask `v_cndmask` VALU than
Triton's. The LLVM post-RA scheduler (driven through `comgr`) does not
interleave QK/PV MFMA into that VALU window as aggressively as Triton's
backend does, and the `iglp_opt` / `sched_group_barrier` hints recover only
a fraction of it. **Closing the last ≈0.4–0.5x therefore requires either a
scheduler-level change (a tighter MFMA↔VALU software pipeline the DSL can
express but `comgr` currently won't schedule) or hand-placed MFMA
interleave in the softmax block — the same scheduler-forfeit frontier seen
on the warp-specialized GEMM and FMoE drives.** Until then the auto selector
keeps this cohort on the K-single-buffer `BLOCK_M=32` combo (which **wins
torch flash at short/mid context — geomean ≈1.10x over flash, dipping to
0.95x at S4096**), and the **≈0.57x-vs-Triton residual (re-measured on
llvm22) is a real MFMA-scheduling limitation, not a missing config, an
occupancy lever, or a stale-backend artifact.**

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
Triton-2d over rocke-production; **>1.0 means CK DSL is faster**, all
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
PYTHONPATH="Python:${AITER_PATH}" python \
  Python/rocke/examples/gfx950/attention/benchmark_prefill2d_live.py \
  --shapes <path/to/unified_attention_shapes.jsonl> --variants prod combo fallback
```

Regenerate the joined cohort CSV (`prefill2d_bf16_triton_ckdsl_perf.csv`)
— this is written by `benchmark_prefill2d_traces.py`, which times the CK
DSL combo policy over the traced shapes and joins a pre-profiled Triton
CSV; the joined file is emitted to the path given by `--combined-csv`:

```bash
export AITER_PATH=<path/to/aiter>
PYTHONPATH="Python:${AITER_PATH}" python \
  Python/rocke/examples/gfx950/attention/benchmark_prefill2d_traces.py \
  --shapes <path/to/unified_attention_shapes.jsonl> \
  --combined-csv prefill2d_bf16_triton_ckdsl_perf.csv
```

## File map

The CK DSL `unified_attention` kernels themselves live in `rocke.instances`
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
| `_d128_cktile_bakeoff.py` | per-shape, same-session A/B of CK DSL production `unified_attention` vs CK Tile `tile_example_fmha_fwd` (subprocess) and Triton, over a d128/d256 GQA-8 cohort; reports `cktile_ms / rocke_ms` (>1 = CK DSL faster) — requires a built `tile_example_fmha_fwd` binary |
| `_profile_one.py` | standalone single-shape launcher for `rocprofv3` profiling of the production-dispatched 2D combo kernel (d64/b32/GQA-8/sinks); args `<sw> <num_seqs> <iters>` |
| `prefill2d_bf16_triton_ckdsl_perf.csv` | captured bf16 prefill-2D cohort (142 deduped shapes; geomean **1.108x** vs Triton-2D at `cap_blocks=65536`, 105/142 wins) |
| `prefill2d_fp8_triton_ckdsl_perf.csv` | captured bf16-Q + fp8-KV prefill cohort (74 shapes; geomean **0.984x**; SW 1.108x 37/37, full-attention 0.874x) |
| `aiter_ua_shapes.json`, `aiter_ua_2_shapes.json`, `aiter_ua_prefill2d_allbf16.json` | captured AITER `unified_attention` call records (paged-KV shapes) used as benchmark inputs |

> The `benchmark_prefill2d_*.py` scripts load shapes via the in-tree shape
> utilities under
> `dsl_docs/optimization/utilities/tools/stage1_benchmark`
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
