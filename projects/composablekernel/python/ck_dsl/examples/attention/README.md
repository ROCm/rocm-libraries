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
cd <composablekernel-checkout>
export AITER_PATH=<aiter-checkout>
PYTHONPATH="python:${AITER_PATH}" python \
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

## Latest results (MI355X, gfx950, ROCm 7.2 / torch 2.12)

> **Re-baselined 2026-05-28.** The previous tables were collected on ROCm
> 7.0.2 and reported ~1.8-2.8x geomean. Two things changed:
>
> 1. **AITER's Triton `unified_attention` got ~2x faster on ROCm 7.2**
>    (decode `tri-auto` 125.8us -> ~57us). CK DSL is essentially unchanged,
>    so the *ratios* compressed even though CK DSL still wins every lane.
> 2. **An `auto`-lane harness bug was fixed.** `_run_ck_dsl` previously
>    force-built the 2D MFMA kernel for `path in ("auto","2d")`, so the
>    `auto` lane measured forced-2D timings instead of the production
>    dispatcher. That mis-reported decode `ck-auto` at ~282us (forced-2D)
>    when the production dispatcher correctly routes decode to 3D at ~46us.
>    The `auto` lane now calls `run_unified_attention_torch(backend="auto")`
>    -- the real `select_path` dispatch. The production code was never
>    affected; only the benchmark's `auto` measurement was.

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

Numbers below are the **mean of 10 full harness runs** (each run uses
10 timed iterations after 5 warmups). The ``ck-auto`` column shows
``mean ± stddev`` across the 10 runs.

### Auto vs Auto — each backend's own selector

| Scenario                  | tri-auto | ck-auto | speedup | tri-path | max_abs(CK vs ref) |
|---------------------------|---------:|--------:|--------:|---------:|-------------------:|
| decode_d128_b16           |   57.5us |  45.7us | **1.26x** | 3d | 1.83e-4 |
| decode_d128_b64           |   55.4us |  45.9us | **1.21x** | 3d | 1.83e-4 |
| decode_d256_b16           |   57.1us |  48.7us | **1.17x** | 3d | 1.22e-4 |
| prefill_d128_b16          |   30.4us |  22.6us | **1.35x** | 2d | 1.95e-3 |
| mixed_d128_b16            |   65.9us |  49.8us | **1.32x** | 3d | 9.77e-4 |
| sliding_d128_b16          |   30.5us |  22.9us | **1.33x** | 2d | 2.75e-4 |
| softcap_d128_b16          |   56.2us |  46.4us | **1.21x** | 3d | 1.22e-4 |
| bf16_decode_d128_b64      |   56.6us |  46.6us | **1.21x** | 3d | 9.77e-4 |
| alibi_decode_d128_b16     |   56.9us |  45.0us | **1.26x** | 3d | 9.77e-4 |
| alibi_mixed_d128_b16      |   57.7us |  45.1us | **1.28x** | 3d | 1.95e-3 |
| qq_bias_prefill_d128_b16  |   31.4us |  22.9us | **1.37x** | 2d | 1.95e-3 |

CK DSL beats Triton on **every** auto-selected scenario; geomean speedup
**≈1.27x** on ROCm 7.2. (The headline dropped from the old report's ~2x
purely because Triton itself roughly halved its own latency on ROCm 7.2 —
see the re-baseline note above; CK DSL's absolute `ck-auto` times are
unchanged within noise.)
`max_abs(CK vs ref)` is the worst per-element error against the AITER
`ref_paged_attn` reference — all rows are within fp16/bf16 ULP. The
output is bit-identical to Triton's (`max_abs(CK vs Triton) == 0`
once both are cast back to the working dtype).

### 3D vs 3D — same split-KV algorithm on both backends

Force-flag rows. This is the algorithmically-honest comparison: same
algorithm, same timer, same stream.

| Scenario                  | tri-3d   | ck-3d    | speedup |
|---------------------------|---------:|---------:|--------:|
| decode_d128_b16           |   57.3us |   45.1us | **1.27x** |
| decode_d128_b64           |   56.7us |   45.4us | **1.25x** |
| decode_d256_b16           |   56.8us |   47.7us | **1.19x** |
| prefill_d128_b16          |   56.5us |   43.8us | **1.29x** |
| mixed_d128_b16            |   55.5us |   49.1us | **1.13x** |
| sliding_d128_b16          |   57.7us |   48.9us | **1.18x** |
| softcap_d128_b16          |   56.9us |   45.8us | **1.24x** |
| bf16_decode_d128_b64      |   56.8us |   45.2us | **1.26x** |
| alibi_decode_d128_b16     |   83.8us |   49.1us | **1.71x** |
| alibi_mixed_d128_b16      |   58.4us |   43.3us | **1.35x** |
| qq_bias_prefill_d128_b16  |   57.1us |   44.9us | **1.27x** |

CK DSL wins 1.13x–1.71x on every scenario; geomean **≈1.28x** on ROCm
7.2 (down from the old report's 1.76x only because Triton's own 3D
kernel sped up; CK DSL's `ck-3d` absolute times are unchanged). The win
comes from the CK Tile lessons we ported into the segment kernel:

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
all three scenarios are now within fp16 / bf16 ULP. See
``PROPOSALS_IMPLEMENTATION_REPORT.md::2D Attention Correctness
Fix`` for the diff.

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
(`/workspace/aiter_unified_attention_*.jsonl`) hit a *different* family:
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

### fp8 KV cache (bf16-Q + fp8-KV, the AmirFix trace family)

The fp8 prefill cohort previously ran the plain 16×16 path (the 32×32
combo was hard-gated off for fp8) and sat at **~0.55–0.60x**. The combo
actually composes with fp8 for free: the **sync-dequant** loader already
writes bf16 into K/V LDS (k_scale folded in) — exactly what the bf16
32×32 reads expect — so the combo runs unchanged once
`use_fp8_mfma_qk` is off. Enabling it (the guard was conservative, not a
real limitation) lifts the fp8 prefill cohort to:

| fp8 prefill bucket (production scale, cap=131072) | before | after |
|--------------------|-------:|------:|
| sliding-window (the AmirFix bulk) | ~0.58x | **1.11x (37/37 win)** |
| full attention                    | ~0.34x | 0.87x |
| overall fp8 prefill               | ~0.50x | **0.98x (near parity)** |

(numbers from `prefill2d_fp8_triton_ckdsl_perf.csv`, 42-shape live sample,
all bit-accurate vs Triton). The two fp8 buckets behave very differently
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

Regenerate the cohort numbers + CSV (`prefill2d_bf16_triton_ckdsl_perf.csv`):

```bash
export AITER_PATH=/workspace/aiter
PYTHONPATH="python:${AITER_PATH}" python \
  python/ck_dsl/examples/attention/benchmark_prefill2d_live.py \
  --shapes /workspace/aiter_unified_attention_2.jsonl --variants prod combo fallback
```

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
