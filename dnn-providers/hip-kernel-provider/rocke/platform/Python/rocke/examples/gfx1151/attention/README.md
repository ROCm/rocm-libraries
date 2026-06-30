# gfx1151 WMMA FMHA-forward: an optimization case study

## TL;DR (executive summary)

A native WMMA flash-attention-forward kernel for **gfx1151** (Strix Halo, Radeon
8060S, RDNA3.5, wave32), tuned with the runbook loop across **four kernel bodies
and ~14 levers**, every variant correctness-gated before timing.

**Bottom line:** the best this DSL achieves on this part is **~11 TF (~18% of the
59 TF f16 WMMA peak)**, from the **single-wave gather** kernels — `fmha_pipelined`
(software-pipelined) at D64, `fmha_singlewave fuse_k` at D128. The kernel is
**issue-bound and register-bound, not FLOP- or bandwidth-bound** (WMMA is ~1% of
issued instructions; the 192-VGPR cap spills at D128). **Every structural rewrite
that the textbook CDNA design prescribes — multi-wave, cooperative LDS K/V
staging, register-blocked tiles — *loses* here** (`fmha_multiwave` ~6.3 TF,
`fmha_blockn` ~8–9 TF, `fmha_regblocked` ~6 TF), because on this large-cache APU the
single-wave global gather is already cache-resident and the cheapest way to feed
the matrix unit. The one ~2× win is **algorithmic**: the causal early-exit.

**vs PyTorch SDPA** (MATH fallback — the only SDPA backend enabled on gfx1151;
FLASH/EFFICIENT are runtime-disabled):

| metric | this kernel | notes |
|---|---|---|
| correctness vs torch SDPA | **max_abs 3e-5** (full D64/D128/GQA), **4.9e-4** (causal) | inside the 2e-2 f16-WMMA-accum gate |
| throughput vs SDPA (25-shape avg) | **13.3×** | SDPA MATH does the full quadratic |
| causal speedup vs SDPA | up to **29.5×** | our causal early-exit skips masked tiles; SDPA MATH does not |
| best absolute | **11.39 TF = 19.3% of peak** | B2 GQA Sq1024 D128 |
| 25-shape avg | **17.0% of peak** | uniformly issue-bound; 0/25 reach 25% |

**Key takeaways surfaced from the body:**
- **Name the roofline first** — the original "200 TF" target is ~3.4× the 59 TF
  physical peak; unreachable by any kernel.
- **Count the matmul fraction** — WMMA at ~1% of instructions ⇒ issue-bound;
  chasing FLOPs is futile until that ratio rises (needs gfx12 larger-K WMMA).
- **CDNA's bandwidth-optimal LDS-staging design is pessimal here** — the APU's
  cache feeds the matrix unit for free, so staging only adds barriers/`ds_load`s.
- **Believe the measurement** — ±0.6 TF thermal drift on this box; several 6%
  "wins" evaporated on back-to-back repeat. Decisions made A/B in one thermal
  window.

See the **iteration ledger**, **Strix Halo nuances**, **gfx950 (CDNA) algorithmic
differences**, and **CK Tile lessons** sections for the durable takeaways.

> **New to flash attention?** [`ALGORITHM.md`](ALGORITHM.md) derives the winning
> kernel from the math up — the attention spec, the online-softmax recurrence
> (with a correctness proof), the causal early-exit, and how every step maps onto
> what one wave32 does per K-loop iteration. Read it first if you want to
> understand *what* the kernel computes before reading *how* it was optimized.

---

A worked application of the
[optimization runbook](../../../dsl_docs/optimization/optimization_runbook.md)
to the native gfx1151 WMMA flash-attention forward kernel
(`rocke.instances.gfx1151.wmma_fmha_fwd`):

- **Part 1** follows the runbook loop on a *single* lever (V-LDS staging) and
  records a result worth keeping precisely because it is the opposite of the
  intuition: **the "optimization" is a regression, and the correct decision is to
  revert it.**
- **Part 2** is a multi-lever campaign (`fmha_singlewave.py` + `tune.py`, plus the
  `fmha_pipelined`/`fmha_blockn`/`fmha_multiwave`/`fmha_regblocked` siblings and the `combo.py`
  cartesian sweep) that pushes toward peak TFLOP/s, **names the hardware roofline
  honestly** (the ~59 TF f16 WMMA peak — a "200 TFLOP/s" target is physically
  unreachable on this part), and diagnoses *why* the kernel is stuck at ~17% of
  peak: it is issue- and register-spill-bound, not FLOP-bound.
- **Part 3** surveys the kept winners across 25 shapes and lands the campaign's
  one ~2× win (the causal early-exit).

The synthesis sections at the end — the **iteration ledger**, **Strix Halo
nuances**, **algorithmic differences from a gfx950 (CDNA) MFMA FMHA**, and
**lessons ported from CK Tile** — are the durable takeaways.

Reproduce everything here with:

```bash
PYTHONPATH=Python python3 -m rocke.examples.gfx1151.attention.bench_v_staging \
    --seqlen-q 512 --seqlen-k 512 --head-size 128 --heads 8 --batch 4
```

(Run on the Strix Halo box: Radeon 8060S, RDNA3.5, wave32, WMMA
`wmma_f32_16x16x16_f16`.)

---

## The kernel under test

One wave32 owns 16 Q rows for a `(q_tile, head, batch)`. Per K-tile it runs
`QK^T` (WMMA) → online softmax → `PV` (WMMA), carrying the running max `m`, sum
`l`, and the `<8 x f32>` PV accumulator as `scf.for` iter-args. `BLOCK_M =
BLOCK_K = 16`.

The lever lives in the **PV matmul's B-operand**. WMMA computes `A @ B^T`, so
`PV = P @ V` needs `V` in `(d × k)` layout: for this lane's d-column `d_col`,
the B-fragment is `V[k, d_col]` for `k = 0..15` — an inherently
**column-strided gather** of `V` (stride = `head_size`).

---

## The Loop

### 1. Hypothesis

> The baseline gathers the PV V-operand straight from global memory with a
> per-`(d, k)` scalar `global_load` — `n_dk * 16` scalar loads per lane per
> K-tile (128 at `head_size=128`). Staging each K-tile's V rows into a
> `16 × head_size` LDS tile once (with wide vector loads), then reading the
> B-operand from LDS, should cut global-memory traffic and speed the kernel up.

This is the obvious move: fewer, wider global loads + on-chip reuse.

### 2. One lever

`WmmaFmhaFwdSpec.v_lds_stage` toggles exactly this and nothing else:

- **`v0_vgather`** (`v_lds_stage=False`): per-`(d, k)` scalar global gather.
- **`v1_vlds`** (`v_lds_stage=True`): each lane vector-loads its k-row's full
  `head_size` slice (8-wide `global_load`) into `V_lds`, then the PV B-operand
  is a strided **LDS** read. Costs one extra `16 × head_size × 2 B` LDS tile
  and shares the existing P-staging barrier.

### 3. Measure (correctness gate, then time)

Every variant is gated against a numpy dense-attention reference before it is
timed — no speed number is reported for an incorrect kernel. Both variants are
bit-for-bit equal to each other and within `3e-5` of the reference.

| shape (B Sq Sk D, Hq=Hk=8) | variant | max_abs | µs/iter | TFLOP/s |
|---|---|---:|---:|---:|
| 4 512 512 128        | v0_vgather | 3.05e-05 | **384.8** | **11.16** |
| 4 512 512 128        | v1_vlds    | 3.05e-05 |   669.1   |   6.42   |
| 4 1024 1024 128      | v0_vgather | 3.05e-05 | **1517.1**| **11.32** |
| 4 1024 1024 128      | v1_vlds    | 3.05e-05 |  2537.3   |   6.77   |
| 4 512 512 64         | v0_vgather | 3.05e-05 | **248.2** | **8.65**  |
| 4 512 512 64         | v1_vlds    | 3.05e-05 |   442.3   |   4.86   |
| 4 512 512 128 causal | v0_vgather | 4.88e-04 | **444.1** | **4.84**  |
| 4 512 512 128 causal | v1_vlds    | 4.88e-04 |   685.3   |   3.13   |

`v1_vlds` is a **consistent 1.5–1.8× regression** across every shape.

### 4. Inspect the ISA (explain the number)

Static instruction mix from `llvm-objdump -d` over the (fully unrolled) body,
`head_size=128`:

| variant | global_load | global_store | ds_load | ds_store | wmma |
|---|---:|---:|---:|---:|---:|
| v0_vgather | 160 | 64 |   4 |  8 | 16 |
| v1_vlds    |  48 | 64 | 132 | 24 | 16 |

The hypothesis was *right about the global loads* — `v1_vlds` cuts them 160 → 48
(3.3×). It was wrong about everything that matters:

1. **The strided access pattern is preserved, just relocated.** The PV B-operand
   is fundamentally `V[k, d_col]` for fixed `d_col` across `k` — a column gather.
   Moving `V` into LDS does not make that pattern contiguous; it becomes 132
   scalar (`n=1`), column-strided `ds_load`s with LDS bank pressure. LDS bought
   nothing the gather didn't already have.
2. **The barrier has nothing to hide behind.** This kernel runs **one wave32 per
   CTA**, so there is no second wave to overlap with the `s_barrier` that the
   staged path forces between the V store and the V read. The latency is exposed
   in full.
3. **The baseline loads are cache-resident.** Across the 16 lanes of the wave,
   each k-row `V[k, :]` is read once, distributed by column — so the gather's
   global traffic is already near-optimal and the L1/L2 services the rest. The
   "expensive" scalar gather is cheap in practice.

### 5. Keep / revert decision

**Revert.** `v_lds_stage` now defaults to `False` (the measured winner) in
`WmmaFmhaFwdSpec`, `mfma_attention_fwd_inner_body`, and the WMMA inner body.
The toggle is kept so this A/B stays reproducible.

---

## Generalizable lessons

- **Fewer global loads ≠ faster.** Counting `global_load` in the ISA is a
  hypothesis, not a measurement. Trading global loads for LDS only wins when LDS
  *changes the access pattern* (gather → contiguous/broadcast) or enables reuse
  that the cache wasn't already giving you.
- **LDS staging needs occupancy to pay for its barrier.** With one wave per CTA
  the staging barrier is pure exposed latency. A staging optimization here only
  becomes viable *after* a structural change that puts multiple waves per CTA
  (a separate, larger lever).
- **The cache is doing real work.** On RDNA3.5 a column-distributed scalar
  gather of a small reused tile stays resident; "obviously bad" memory patterns
  can be fine, and the only way to know is to measure.

---

# Part 2: the multi-lever campaign (`fmha_singlewave.py` + `tune.py`)

The single-lever study above reverts cleanly, but "one lever lost" is not "the
kernel is optimal" — a lever can be dead on its own yet alive in combination, or
masked by a different bottleneck. So we built a **heavily-parameterized** vehicle
(`fmha_singlewave.py`, driven by `tune.py`) and swept levers individually and in
combination, every variant GPU-measured against the numpy reference.

```bash
PYTHONPATH=Python python3 -m rocke.examples.gfx1151.attention.tune \
    --bm 1 2 --pmode lds --vmode gather lds_t --qpreload 0 1
```

## First, the hardware ceiling (why "200 TFLOP/s" is unreachable here)

The headline target was 200+ TFLOP/s. That is **~3.4× the physical peak** of this
part, so it cannot be reached by any kernel — stating it plainly up front so the
rest of the numbers have an honest frame:

- **f16 WMMA peak, Radeon 8060S (RDNA3.5, 40 CU, ~2.9 GHz):** 40 CU × 512
  FP16 FLOP/clk × 2.9e9 ≈ **59 TFLOP/s**. That is the wall.
- **Empirical reality check:** a naive one-wave-per-16×16-tile WMMA GEMM
  (`instances.gfx1151.wmma_gemm`) sustains only **4.3 TF** at 4096³ — it is
  memory-bound, re-reading A/B from global every K-step.
- **This attention kernel sits at ~10.3 TF** — i.e. it already *beats* the naive
  GEMM (better on-chip reuse) and is within noise of the production
  `wmma_fmha_fwd` (~11 TF). It is at ~17% of the 59 TF peak.

So the realistic envelope for *this* kernel family is tens of TFLOP/s, and the
honest objective is "approach the 59 TF roofline," not "hit 200."

## Why we're stuck at ~10 TF: it's issue- and register-bound, not FLOP-bound

Static instruction mix of the unrolled K-tile body (`tune.py` ISA counter +
msgpack resource decode), `head_size=128`:

| metric | value | reading |
|---|---:|---|
| total instr / K-tile body | 1398 | the issue budget |
| `wmma` | 16 | **~1% of issued instructions** |
| `global_load` | 160 | 128 of them are the PV V-gather |
| VGPR / wave | **192** (the cap) | + **26 spills** to scratch |
| waves / CTA | 1 | no second wave to hide latency |

Two structural walls fall out of this:

1. **WMMA is ~1% of the instruction stream.** Even with infinite occupancy the
   kernel is bottlenecked on VALU/memory issue, not matmul. You cannot approach
   the FLOP roofline while 99% of issued instructions are not matmul.
2. **The register file is overcommitted.** The PV accumulator alone is
   `n_dk × <8×f32>` = 64 VGPR at `head_size=128`; with K-frags and softmax temps
   the allocator pegs at the 192-VGPR cap and **spills 26**. Confirmed by the
   `head_size=64` case (accumulator halves to 32 VGPR): **vgpr=142, spill=0** —
   no spill, cleaner code. Causal `head_size=128` spills *worst* (41).

## Levers swept (all GPU-measured, B4 Sq512 Sk512 D128 Hq8 unless noted)

| lever | result | verdict |
|---|---|---|
| **Vectorized acc-rescale** — rebuild `alpha` as a `<8×f32>` and rescale each `acc[d]` with one `vector_mul` instead of `c_frag` scalar extract/mul/insert | 9.5 → ~9.9 TF | **KEEP** |
| **Q-resident preload** — hoist the `BM·n_dk` Q-frags out of the K-loop (Q is loop-invariant) | neutral (±noise); VGPR already at the 192 cap, and Q is L1-resident so dynamic loads were already cheap | default **off** |
| **V-LDS staging, transposed (`v_mode=lds_t`)** — *revisited* the rejected V-staging idea with the new occupancy understanding: stage V *transposed* so the B-operand read is contiguous | **8.06 TF (regression)**; the transpose itself costs 136 `ds_store` scatters — staging just relocates the strided access, same lesson as Part 1, now reconfirmed against the register/occupancy hypothesis | **revert** |
| **M-amplification (`bm_tiles` 2, 4)** — one wave owns 2/4 Q-tiles to amortize the shared K/V load across more matmuls | consistent regression (D128 bm2 = 5.4 TF vs bm1 9.5; D64 bm4 = 2.9 TF) — bigger tiles shrink the grid and cut occupancy on a kernel that is already latency-bound | **revert** |
| **K-frag fusion (`fuse_k`)** — load each K-frag *inside* the QK matmul (1 live frag vs `n_dk`) instead of hoisting all `n_dk` K-frags before the loop | **the campaign's win at D128**: spills 26 → 10, 9.5 → 10.5 TF (512), 10.4 → 11.1 TF (1024). But it **hurts at D64** (8.8 → 7.4 TF) where there is no spill to relieve and the extra reloads are pure added latency. So it is **auto-enabled only when the kernel spills** (`fuse_k=None` → `head_size>=128`) | **KEEP (conditional)** |
| **BLOCK_N widening (`fmha_blockn.py`, `bn_tiles` 2/4/8)** — consume `bn_tiles` 16-key subtiles per K-loop step so each iteration issues `bn_tiles`× more WMMA, directly attacking the "WMMA is ~1% of instructions" diagnosis | the WMMA fraction *does* rise (1.1% → 2.3% at bn8) but it **loses at every width**: D128 10.8 → 8.6/9.0/8.7 TF (spills explode 10 → 44 → 68 → 130 holding `NK` score+P tiles live), and even at D64 *with* spill headroom (bn2 stays spill=0) it still regresses 8.6 → 8.3 → 7.9 → 7.1 — the extra per-subtile V-gathers and P-transpose `ds_load`s outweigh the loop-overhead saved | **revert** |
| **Software pipelining (`fmha_pipelined.py`, ck_tile `qr_ks_vs`)** — hoist the *next* K-tile's `QK` (independent of the current tile's softmax) and carry the pre-computed score through the `scf.for` iter-args, so the matrix unit runs QK of tile `i+1` while the VALU does the softmax of tile `i` | **the mirror image of `fuse_k`**: a **win at D64** (no spill) — 8.6 → **10.7 TF (512), 11.1 TF (1024)** — but a **regression at D128** where carrying the extra `<8×f32>` score + the next-QK temps pushes spills **10 → 57** and drops it to 8.6 TF. The overlap only pays when there is register headroom to hold the carried state | **KEEP (conditional, D64)** |

Three changes are kept. **Vectorized rescale** is a small unconditional win.
The other two are **complementary, register-headroom-gated** levers that each
attack the named bottleneck from the opposite side:

- **`fuse_k`** wins **when the kernel spills** (D128): it trades recomputed
  K-loads for fewer live VGPRs, relieving the spill (26 → 10) — so it is gated on
  `head_size>=128`.
- **Software pipelining** wins **when there is register headroom** (D64): it
  spends spare VGPRs on a carried next-tile score to overlap QK with softmax —
  but at D128 there is no headroom, the carry *causes* spilling (10 → 57), and it
  loses. So the per-shape best is **D64 → `fmha_pipelined`, D128 → `fmha_singlewave`
  fuse_k**, and both top out at ~11 TF.

The `sched_group_barrier` interleave hints (`PipelinedCfg.sched`) are noise-level here
(±0.5 TF) — the spill, not instruction scheduling, is the wall. Everything else
is reverted to its losing-but-reproducible toggle.

### Two more levers, both negative (ds_bpermute P-transpose; lever recombination)

After software pipelining landed as the D64 winner, two follow-ups were tried.

**ds_bpermute register P-transpose (`PipelinedCfg.p_xpose="shuffle"`)** — port ck_tile's
gfx11 `PermuteWarpGemmCToA` (`permlanex16` + `v_perm_b32`) to remap the QK-score
C-distribution into the PV A-operand *in registers*, eliminating the LDS
round-trip + `s_barrier`. The ISA confirms the LDS traffic vanishes (`ds_load`/
`ds_store` 12 → 0), but it is a **structural dead-end on this DSL** and a
**regression** (D64 10.2 → 8.9, D128 8.9 → 8.3):

- ck_tile's permute only exchanges `lane ↔ lane^16` (2 lanes) and feeds the
  result into a WMMA whose A-operand distribution is *co-designed* to consume
  that cheap permute (`MakeABlockTileDistribution`). The DSL's `mma` op has a
  **fixed standard a_map** where lane `l` needs row `l` — data gathered from *16*
  source lanes. `permlanex16` provably cannot gather across 16 lanes, so the
  cheap permute **cannot** produce the layout the DSL's WMMA requires. (The
  shuffle output scores `1.1–1.3e-2` — wrong, just coincidentally close because
  post-softmax probabilities are similar in magnitude; it slips under the 2e-2
  gate but is not correct.)
- A *correct* full-row gather needs 8–16 `ds_bpermute`s + selects, and
  `ds_bpermute` **uses the same LDS hardware unit** on RDNA — it removes the
  barrier, not the LDS access. On an issue-bound kernel, trading 12 `ds_load`/
  `ds_store` for ~113 bit-twiddle instructions (+spills) is a guaranteed loss.

  **Lesson:** a register-shuffle transpose ported from a co-designed C++ kernel
  needs the *consuming* matmul's operand distribution to be customizable too; a
  DSL with a fixed WMMA a_map can't express it, and the generic substitute
  (`ds_bpermute`) is the same LDS engine with more instructions.

**Recombining old levers with the winner** — with software pipelining as the D64
base, `fuse_k` and `sched` were swept in combination (the standing "revisit
rejected levers when the bottleneck model changes" instruction). One run showed
a tempting `fuse_k+sched = 10.93 TF` synergy, but **a repeat collapsed it into
the noise**: the whole D64 shape drifts ±0.6 TF run-to-run (thermal/clock), and
on re-measure all four `fuse_k × sched` configs cluster within ±0.1 TF.
**No robust combination win** — `fuse_k`/`sched` are noise-level at D64; the only
result above the noise floor is software pipelining itself (8.6 → ~10–11 TF, a
>1 TF lift). **Lesson:** measure twice before believing a 6% combination effect
on a box with 6% thermal variance.

## The structural lever, actually built: multi-wave (`fmha_multiwave.py`)

The hypothesis from every micro-lever was the same: the only thing that breaks
the two walls is **structural** — put **multiple wave32s in one workgroup**,
**cooperatively stage K/V into LDS once per CTA** (shared across all waves), and
let the second wave hide the barrier latency that killed every single-wave
staging attempt. So we built it: `fmha_multiwave.py` (`MultiWaveCfg`, driven by
`mw_tune.py`). `n_waves` wave32s per CTA, each owns one 16-row Q-tile; `K_lds`
(`[16][hs]`) and the transposed `V_lds_t` (`[hs][16]`) are loaded cooperatively
by all threads, both WMMA B-operands become contiguous LDS reads, and the
P-transpose uses a per-wave LDS slab with an intra-wave `s_waitcnt` (no
cross-wave barrier).

```bash
PYTHONPATH=Python python3 -m rocke.examples.gfx1151.attention.mw_tune --waves 2 4
```

**It is correct (max_abs 3.05e-05) but slower: w4 = 6.3 TF vs the single-wave
10.5 TF.** This is the decisive result of the campaign, not a disappointment: it
**confirms the kernel is not memory-bound.** Cooperative LDS staging only pays
when global traffic is the bottleneck; here the single-wave gather was already
cache-resident, so moving K/V through LDS adds barriers and `ds_load`/`ds_store`
overhead that the cache was absorbing for free — the same lesson as Part 1, now
proven at the structural scale that was supposed to rescue it. The win would
require *also* register-blocking larger output tiles per wave to raise the
WMMA:overhead ratio (production CK GEMM design), a substantially larger rewrite;
even then the ceiling is **~25–35 TF (40–60% of the 59 TF peak)**, not 200.

## The full production rewrite, actually built: `fmha_regblocked.py`

So we built that substantially larger rewrite — the one design the campaign
kept naming as the only remaining path. `fmha_regblocked.py` (`RegBlockedCfg`, driven by
`prod_tune.py`) does **both** structural levers at once, the two that each lost
*alone*:

- **multi-wave + shared K/V in LDS** (the `fmha_multiwave` lever) — amortize the
  staging barrier and global K/V traffic across `num_warps` query-row tiles;
- **register-blocked larger output tiles per wave** (the `fmha_blockn` lever) —
  each wave owns `m_repeat` M-atoms × `n_repeat = block_n/16` N-atoms, issuing
  `2·m_repeat·n_repeat·n_dk` WMMAs per K-tile (m1 n4 D128 = 64 vs single-wave
  16) to raise the WMMA:overhead ratio.

The bet: the density rise pays for the staging barrier, and the extra waves give
the barrier (and the spills) latency to hide behind. Warps partition along **M
only** (`warp_n=1`), so softmax stays intra-warp (`wave_reduce_*` over 16 lanes
+ an in-register combine across the `n_repeat` score fragments); K is staged
`[n][d]`, V transposed `[d][n]`, P→A via the per-wave LDS slab (not ds_bpermute).

```bash
PYTHONPATH=Python python3 -m rocke.examples.gfx1151.attention.prod_tune \
  --head-size 128 --num-warps 2 4 --m-repeat 1 2 --block-n 32 64
```

**It is correct in every mode (max_abs 1.5e-5 D128, 3.05e-5 D64, 4.9e-4 causal)
but a consistent ~2× REGRESSION: ~6 TF across the entire D64 and D128 sweep**
(`num_warps∈{2,4} × m_repeat∈{1,2} × block_n∈{32,64}`), vs the single-wave wall
(~11 TF D64 `fmha_pipelined`, ~11 TF D128 `fmha_singlewave fuse_k`). This is the decisive
close of the structural campaign — **all three structural variants lose to the
single-wave gather kernels**:

| variant | what it does | best TF | vs single-wave |
| --- | --- | --- | --- |
| single-wave gather | no LDS K/V, no blocking | **~11** | baseline |
| `fmha_blockn` | register-block, global gather | ~8–9 | regression |
| `fmha_multiwave` | multi-wave, LDS K/V, no blocking | ~6.3 | regression |
| `fmha_regblocked` | multi-wave + LDS K/V + register-block | ~6 | bigger regression |

**Register blocking *did* behave as designed once the B-operand reuse was made
real.** The first cut reloaded each K/V LDS B-operand *inside* the `m_repeat`
loop (and Q inside the `n_repeat` loop), so one LDS read fed only one matmul —
`m_repeat=2` was *slower* than `m_repeat=1`. Hoisting the loads so one `K_lds`
read feeds all `m_repeat` QK matmuls (and one `V_lds_t` read all the PV matmuls)
flipped it: D64 `w2 m2 n32` rose 5.49 → **6.19 TF** over `m1`, WMMA density
doubled (16 → 32), **spill stayed 0**. The lever works — it just isn't enough.

**Each structural addition still makes it worse, monotonically, for one reason:
the kernel is issue-bound and every scheme adds instructions.** The ISA confirms
it — even with proper reuse the WMMA fraction stays ~1–2% (`wmma=32–64` of
`instr=1700–4400`), because the cooperative LDS staging + the per-`(mr,nr)`
P-transpose round-trips add instructions *in proportion to* the WMMA density they
buy, leaving the ratio flat. Spills were even relieved (D128 `spill=0` here vs
single-wave `spill=10/11`) and it *still* lost — proving the wall was never
spills or memory traffic but raw instruction issue. The production-CK LDS-staging
design was built for an HBM-bandwidth-bound CDNA part with dense MFMA; on this
large-cache, issue-bound RDNA3.5 APU those assumptions invert — the single-wave
global gather is cache-resident and the cheapest possible way to feed the matrix
unit, so any LDS staging is a net loss no matter how many waves amortize the
barrier. (The B-operand LDS reads still use the `vec_concat`-of-two-`<8×f16>`
pattern — two `ds_read_b128`s instead of 16 scalar `ds_load`s — so the loss is
*not* an artifact of naive LDS reads; the structural overhead is intrinsic.)

**Conclusion of the campaign: the single-wave gather kernels (`fmha_pipelined` at D64,
`fmha_singlewave fuse_k` at D128) at ~11 TF (~18% of the 59 TF peak) are the best
this DSL achieves on gfx1151.** The remaining ~5× gap to peak is not reachable
by any structural reorganization expressible here; it would need RDNA4/gfx12
intrinsics (larger-K WMMA, `ds_read_tr`, async-LDS — see below).

## Different WMMA intrinsics? A dead-end on this silicon

A natural question: would a *different* WMMA atom relieve the register/issue
walls? Investigated against `core/arch/target.py` (`_MMA_FRAGMENT_INFO`) and
`core/isa/backend.py` (`_RDNA_WMMA`, `MemoryCapabilities` for gfx1151):

- **f16-accumulate WMMA** (vs the f32 accumulator) would halve the PV
  accumulator VGPRs in principle — but on RDNA3.5 the f16-output WMMA still
  writes a `<8 x float>`-shaped C fragment (8 VGPR); **packed-f16 C output is a
  gfx12/RDNA4 feature**, not available here. No register saving.
- **Larger-K WMMA (`16x16x32`)** would cut the QK/PV instruction count per
  K-tile (better WMMA:overhead ratio) — **gfx12 only**.
- **`ds_read_tr` transpose-load** would remove the P-transpose shuffle/LDS
  round-trip — `MemoryCapabilities.has_ds_read_tr=False` on gfx1151; also gfx12.
- **`has_async_lds=False`** too, so no async-copy overlap for the staging path.

The only WMMA atom on gfx1151 is `wmma_f32_16x16x16_f16` (`_bf16` is the same
shape). **Different intrinsics are a dead-end lever on this part** — the relevant
ones are all RDNA4/gfx12. The realistic path remains the structural rewrite
above, capped by the 59 TF roofline.

## Part 3: the 25-shape survey + the causal early-exit win (`survey.py`)

The micro/structural campaign above all measured one shape (B4 Sq512 D128). To
see where the kernels actually land across the workload space, `survey.py` runs
25 shapes — D64/D128 × seqlen {512…4096} × batch/heads/GQA/causal — keeping the
faster of `fmha_pipelined` and `fmha_singlewave fuse_k` per shape, and timing PyTorch SDPA
(only the MATH fallback exists on this APU; FLASH/EFFICIENT are runtime-disabled
on gfx1151) as the reference.

The survey exposed one large, un-swept inefficiency: **causal shapes ran at half
the efficiency of non-causal** (~7.5–8.4% of peak vs ~15–19%). The cause was not
a micro-lever — the causal kernels looped **every** K-tile and masked the upper
triangle, doing ~2× the matmul work the math requires. The fix is the standard
flash-attention **causal early-exit**: a CTA owning query rows
`[group_row0, group_row0+BM·16-1]` only needs K-tiles whose lowest key index is
`≤` its max query position, i.e. `kt < group_row0/16 + BM`. Clamp the K-loop
bound to that (one `cmp_lt` + `select`, gated on `mask_mode=="causal"`):

```python
loop_stop = seqlen_k // 16
if causal:
    causal_stop = group_row0 // 16 + BM        # last needed tile + 1
    loop_stop = select(causal_stop < loop_stop, causal_stop, loop_stop)
```

Added to both `fmha_pipelined.py` and `fmha_singlewave.py`. `fmha_pipelined`'s software pipeline
stays correct: its prologue computes tile-0 QK (always needed, `loop_stop≥1`) and
its next-tile prefetch is already clamped against `loop_stop`, so the carried
`score_next` on the final iteration is computed-but-never-consumed as before.

**Result — causal throughput ~doubled, in line with non-causal:**

| causal shape (Sq=Sk)        | before | after  |
|-----------------------------|-------:|-------:|
| B4 H8 S1024 D64             | ~4.7 TF | 8.92 TF |
| B4 H8 S2048 D64            | ~5.0 TF | 9.34 TF |
| B4 H8 S1024 D128            | ~5.0 TF | 8.44 TF |
| B4 H8 S2048 D128           | ~5.2 TF | 8.89 TF |

This is the rare ~2× win in the whole campaign — and it is *algorithmic*, not a
hardware-roofline play: we stopped issuing matmuls whose output is masked to
zero. It does not move the issue-bound non-causal ceiling.

**Survey aggregate (25 shapes, after the fix):** best 11.39 TF (19.3% of peak, B2
GQA Sq1024 D128); **avg 17.0% of peak** (was 14.4% before the causal fix); **13.3×
vs SDPA** (causal shapes up to 29.5×, since SDPA's MATH path also does the full
quadratic). Non-causal sits at 15–19% and causal now at 14–16% — uniformly
~17%, the issue-bound single-wave wall. **0/25 reach 25% of peak**: that target
needs the matmul:overhead ratio to rise, which on this part requires gfx12
larger-K WMMA, not another single-wave lever (see "Different WMMA intrinsics"
above).

## Iteration ledger (what was tried, and how much)

The campaign ran the runbook loop — *hypothesis → one lever → GPU-measure against
the numpy/torch reference → inspect ISA → keep/revert* — across **four kernel
bodies** and ~14 distinct levers, every variant correctness-gated before timing:

| kernel | levers swept | outcome |
| --- | --- | --- |
| `fmha_singlewave` | `bm_tiles`, `p_mode`, `v_mode` (gather/lds_t), `q_preload`, `fuse_k`, vectorized acc-rescale, vectorized P-read (`vec_concat`), epilogue `inv_l` hoist | **D128 winner ~11 TF** (`fuse_k` auto on D128) |
| `fmha_pipelined` | software pipelining (`qr_ks_vs`), `sched` hints, `p_xpose` (lds/shuffle) | **D64 winner ~11 TF** |
| `fmha_blockn` | `bn_tiles` 2/4/8 (BLOCK_N widening) | regression (spills) |
| `fmha_multiwave` / `fmha_regblocked` | `n_waves`, then `num_warps × m_repeat × block_n` (+ register-blocked B-operand reuse, double-buffer stub) | regression (~6 TF) |

`combo.py` then ran the **full cartesian product** of every compatible single-wave
lever across `fmha_singlewave`/`fmha_pipelined`/`fmha_blockn` per shape, to answer "did any
*combination* beat what each lever scores alone?" — it did not: the per-shape best
is always a single lever (D128 `opt bm1 fuse_k` 19.3%, D64 `sp sched0` 16.9%),
and **0 combinations clear 20% robustly** (best runs straddle it at 19–20%, the
remainder thermal noise). `survey.py` then measured the kept winners across 25
shapes (avg 17% of peak, 13.3× vs SDPA).

Two micro-fusions from the final pass are kept because they cut static
instructions with bit-identical output (the kernel is issue-bound, so instruction
count is the currency): the **vectorized P-transpose read** (two `ds_read_b128` +
`vec_concat` instead of 16 scalar `ds_load` + `vec_insert`, 1407 → 1395 instr) and
the **epilogue `inv_l` hoist** (`rcp`/`select` computed once per row instead of per
`d`). Both are wall-clock-neutral within the ±0.6 TF thermal noise — expected, and
consistent with the issue-bound model: removing a handful of the ~1400 body
instructions is real but small.

**Correctness is verified against PyTorch SDPA** (MATH backend, the only one
enabled on gfx1151), not just the numpy reference: on identical inputs the kernel
matches torch to `max_abs 3e-5` (non-causal D64/D128/GQA) and `4.9e-4` (causal) —
well inside the `2e-2` f16-WMMA-accumulation-order gate.

## Strix Halo nuances (why this part behaves differently)

gfx1151 is an **APU**, and almost every surprising result traces back to that:

- **Unified LPDDR5X memory + a large last-level cache.** K/V for a single
  `(head, batch)` flash-attention tile is small and reused across the wave's 16
  lanes, so the "expensive" column-strided global gather stays **cache-resident**.
  This is the root cause of the whole campaign: LDS staging (Part 1, `fmha_multiwave`,
  `fmha_regblocked`) replaces cheap cached reads with serialized `ds_load`s + barriers
  and *always* loses. On a discrete HBM part the same staging is mandatory.
- **wave32, one WMMA atom.** The only matrix intrinsic is
  `wmma_f32_16x16x16_f16` (`_bf16` is the same shape) — K=16, **f32-only C
  accumulator** (`<8×f32>`, 8 VGPR even for f16 output; packed-f16 C is gfx12).
  No `16x16x32`, no `ds_read_tr`, no async-LDS (`MemoryCapabilities` all `False`).
- **192-VGPR cap with no slack at D128.** The PV accumulator alone is
  `n_dk × <8×f32>` = 64 VGPR at D128; with K-frags + softmax temps the allocator
  pegs the cap and **spills**. D64 halves the accumulator and runs spill-free —
  which is *why* the two best levers split by head_size (`fuse_k` relieves D128
  spills; software pipelining spends D64's spare registers).
- **~59 TF f16 WMMA peak** (40 CU × 512 FLOP/clk × ~2.9 GHz). The original
  "200 TF" target is ~3.4× the physical wall — unreachable by any kernel.
- **Thermal drift ±0.6 TF run-to-run**; the box heat-soaks during long build
  sweeps, lowering absolute numbers. Every keep/revert decision was made A/B
  *back-to-back in one thermal window*, never across sweeps — several apparent
  6% "wins" evaporated on repeat.
- **No FLASH/EFFICIENT SDPA backend** on gfx1151; only the MATH fallback exists,
  so PyTorch is a correctness reference and a (quadratic) throughput floor, not a
  fair flash-vs-flash comparison.

## Algorithmic differences from a gfx950 (CDNA) MFMA FMHA

The production CK / CK-Tile FMHA kernels target CDNA (gfx9xx, e.g. gfx950). The
flash-attention *math* is identical (online softmax, causal early-exit, the
`qr_ks_vs` pipeline), but the kernel structure inverts on several axes:

| axis | gfx950 (CDNA, MFMA) | gfx1151 (RDNA3.5, WMMA) |
| --- | --- | --- |
| wavefront | wave64 | **wave32** — softmax row-reduction butterfly spans 16 lanes, half the lane masks |
| matrix unit | `v_mfma_*`, many tile shapes (`16x16x16`, `16x16x32`, `32x32x8`) → high arithmetic intensity per instr | **one** `16x16x16` WMMA → low WMMA:overhead ratio, the issue-bound wall |
| operand semantics | MFMA `A·B` with native K-accumulation; large K tiles | WMMA `A·B^T` — forces the **V column-gather / transpose** dance the campaign fought |
| C fragment | can output packed f16 | **always `<8×f32>`** — no accumulator VGPR relief |
| memory bound | **HBM-bandwidth-bound** → LDS staging + double-buffer + async-copy is the *correct* design | **cache-resident / issue-bound** → LDS staging is a net *loss* |
| transpose | `ds_read_tr` transpose-loads, async global→LDS | neither exists → P→A must round-trip LDS or `ds_bpermute` (both costly) |
| register file | large; deep pipelines fit | 192-VGPR cap; even one-stage pipelining spills at D128 |

The headline inversion: **the production LDS-staging FMHA design is bandwidth-
optimal on CDNA and pessimal here.** Porting its structure verbatim (`fmha_multiwave`/
`fmha_regblocked`) reproduces the CDNA dataflow faithfully and loses, because the
bottleneck it was built to relieve (HBM bandwidth) is not this part's bottleneck
(instruction issue, against a cache that already feeds the matrix unit for free).

## Lessons ported from CK Tile

- **`qr_ks_vs` software pipelining** (hoist next-tile QK, overlap with current
  softmax) → `fmha_pipelined`. Ports cleanly and is the **D64 win**; at D128 the
  carried score state spills (10 → 57) and it loses. *Lesson: a pipeline depth
  that's free on a large register file costs spills on a 192-VGPR part.*
- **`PermuteWarpGemmCToA`** (register C→A transpose via `permlanex16` + `v_perm`)
  → attempted as `p_xpose="shuffle"`. **Cannot port**: CK Tile *co-designs* the
  consuming WMMA's operand distribution (`MakeABlockTileDistribution`) so a
  2-lane permute suffices; the DSL's `mma` has a **fixed a_map** needing a 16-lane
  gather, which `permlanex16` provably can't do, and the generic substitute
  `ds_bpermute` is the same LDS engine with more instructions. *Lesson: a
  register-shuffle transpose needs the consuming matmul's layout to be
  customizable too.*
- **Cooperative LDS K/V staging + register blocking (`MRepeat`/`NRepeat`)** →
  `fmha_multiwave` + `fmha_blockn` + `fmha_regblocked`. The register blocking is genuinely
  CK's density lever and works as designed (WMMA 16 → 32, spill 0); the LDS
  staging is the bandwidth lever and inverts here. *Lesson: separate a porting
  source's levers by *which bottleneck* each attacks, and re-test each against the
  target part's actual bottleneck rather than adopting the bundle.*
- **Online-softmax + causal early-exit** → the one unambiguous shared win (~2× on
  causal), because it's algorithmic (skip masked matmuls), not microarchitectural.

## Generalizable lessons (campaign)

- **Name the roofline before optimizing.** A target above the hardware peak
  (200 vs 59 TF) is a spec bug, not a kernel bug — catch it with one line of
  arithmetic before burning iterations.
- **Count the matmul fraction.** If WMMA is 1% of issued instructions, you are
  issue-bound; chasing the FLOP roofline is futile until that ratio changes.
- **Re-test rejected levers when the bottleneck model changes — but believe the
  measurement.** We revisited V-staging under the register/occupancy lens (the
  user's standing instruction); the *transposed* variant is new, yet it still
  loses, for a now-better-understood reason (the transpose scatter, plus a
  barrier with no second wave to hide it). Revisiting was right; overriding the
  GPU number would not be.
- **Spills are visible without a profiler.** The AMDGPU msgpack note carries
  `.vgpr_count` / `.vgpr_spill_count`; decode it straight from the HSACO
  (`tune._resource_counts`) — no `llvm-readelf` needed.
- **Survey before you micro-optimize.** One-shape tuning hid that causal ran at
  half efficiency. The biggest single win in the campaign (~2× on causal) was an
  *algorithmic* skip of masked work, not a hardware lever — and it was only
  obvious once the 25-shape sweep put causal and non-causal side by side.

## Reusing — and extending — the CK Tile helper layer for RDNA WMMA

All five kernels are built on the CK Tile helper layer (`rocke.helpers`):
`make_global_view` + `make_tile_window` (with `shift_by` for the K-loop step) for
Q/K/V/O addressing, `make_lds_view` + `TileWindow` for the P-transpose round-trip,
the `helpers.attention` softmax primitives (`wave_reduce_max/sum`,
`apply_attention_mask`), and a WMMA atom for the QK/PV matmuls.

The helper layer was built for **CDNA / MFMA / wave64**, so three pieces were
added for RDNA3.5 WMMA / wave32. All three are **additive** — no existing MFMA or
f32 path changed; the full `test_rocke` suite stays green:

| gap | what was added | where |
|---|---|---|
| only `MfmaAtom` (wave64, MFMA) existed | `WmmaAtom` (+ `wmma_atom`, `WMMA_F16/BF16_ATOMS`): wave32, m=n=k=16, a/b_per_lane=16, c_per_lane=8 (`<8×f32>`), `name="wmma_f32_16x16x16_f16"`. Lane-layout accessors **delegate to the existing `target.mma` LayoutMaps** so there is one source of truth; `emit` routes through `b.mma(name,…)` | `helpers/atoms.py` |
| `WarpGrid` was MFMA/wave64-only | a wave32 WMMA path accepting the WMMA atom | `helpers/geometry.py` |
| `load_tile` is f32-only (casts every element via `load_vec_as_f32`) — not a packed `<16×f16>` fragment, so not directly `b.mma`-able | `load_wmma_fragment` / `store_wmma_acc`: a **packed** fragment load/store built on `TileWindow.load_vec(n=16)` (raw packed vector → directly `b.mma`-able), matching the op's layout maps | `helpers/distribution.py`, `helpers/tensor_view.py` |
| `StaticDistributedTensor` stores a per-element f32 list (the right shape for elementwise/reduce ops, the wrong one for an issue-bound WMMA fragment) | `WmmaTensor`: a **packed** distributed tensor carrying one lane's fragment/accumulator as a *single SSA vector*, with tile-level `load_wmma_tile` / `wmma_mma` / `store_wmma_tile` wrappers and `tile.scale` (one `v_mul`) / `tile.coord` (off the verified layout map). Each is 1:1 over the underlying op, so the kernel body reads in tile terms with **zero** added instructions | `helpers/distribution.py` |

**Why the packed fragment path matters.** This kernel is issue-bound — static
instruction count is the perf currency — so the WMMA operand load must be a single
packed vector-load feeding `b.mma`, and the accumulator must carry as a packed
vector (the rescale is one `v_mul`, not eight scalar muls). Routing a WMMA operand
through the f32 `load_tile`, or a fragment through the per-element
`StaticDistributedTensor`, would insert per-element casts/repacks and regress
TFLOP/s. So `load_wmma_fragment`/`WmmaTensor` stay packed (built on the raw
`load_vec`, not `load_vec_as_f32`); all five kernels read in tile terms
(`load_wmma_tile` → `wmma_mma` → `acc.scale` → `store_wmma_tile`) while lowering to
the identical single load / `b.mma` / `v_mul`. *Lesson: on an issue-bound
kernel a convenience cast — or a per-element container — in a generic path is a
silent regression; the fragment and the accumulator have to stay packed.*

## Files

- `ALGORITHM.md` — a from-the-math-up guide to the winning kernel for readers new
  to flash attention: the attention spec, the online-softmax recurrence with a
  correctness proof, the base-2 `exp` trick, the causal early-exit, and a
  step-by-step map from each line of the recurrence to the kernel body. Read
  before the case study below.
- `bench_v_staging.py` — Part 1 A/B harness (V-staging on the production spec):
  builds both variants, gates each on a numpy reference, times with HIP events
  (`runtime.launcher.time_launches`), disassembles for the memory-instruction
  mix, writes `v_staging_perf.csv`.
- `v_staging_perf.csv` — Part 1 results.
- `fmha_singlewave.py` — Part 2 vehicle: a self-contained, heavily-parameterized WMMA
  FMHA-fwd kernel (`SingleWaveCfg`: `bm_tiles`, `p_mode`, `v_mode`, `prefetch_k`,
  `q_preload`, `fuse_k`). Same WMMA contract/ABI as production; exposes the swept
  levers. `fuse_k=None` auto-enables K-frag fusion when `head_size>=128`. Keeps
  the vectorized acc-rescale, the `vec_concat` P-transpose read, and the epilogue
  `inv_l` hoist (the issue-bound instruction-cut micro-fusions).
- `tune.py` — Part 2 driver: `verify_and_time(cfg, shape)` gates on numpy, times
  with HIP events, and tallies both the disassembly instruction mix and the
  VGPR/SGPR/spill/LDS resource note. `main()` sweeps `--bm/--pmode/--vmode/
  --qpreload/--fusek` and prints the running best.
- `fmha_multiwave.py` — the structural lever, built: a **multi-wave** WMMA FMHA-fwd
  kernel (`MultiWaveCfg`: `n_waves`). `n_waves` wave32s per CTA cooperatively stage K/V
  into LDS. Correct but slower (confirms not memory-bound) — see "The structural
  lever, actually built" above.
- `mw_tune.py` — driver for `fmha_multiwave.py`; `--waves` axis, same verify+time+ISA
  harness as `tune.py`.
- `fmha_regblocked.py` / `prod_tune.py` — the **full production rewrite**: a
  register-blocked multi-wave kernel (`RegBlockedCfg`: `num_warps`, `m_repeat`,
  `block_n`) combining the `fmha_multiwave` and `fmha_blockn` levers. Correct in every
  mode but a ~2× regression (~5 TF) — the decisive close of the structural
  campaign (see "The full production rewrite, actually built" above). Driver
  sweeps `--num-warps/--m-repeat/--block-n`.
- `fmha_blockn.py` / `bn_tune.py` — the WMMA-density lever: a BLOCK_N-widened
  single-wave kernel (`BlockNCfg.bn_tiles`) that processes `bn_tiles` 16-key subtiles
  per K-loop step. Correct but a regression at every width (see the levers table)
  — confirms the kernel is register/issue-bound, not loop-overhead-bound.
- `fmha_pipelined.py` / `sp_tune.py` — the software-pipelining lever (ck_tile
  `qr_ks_vs`): a single-wave kernel (`PipelinedCfg`) that hoists the next K-tile's `QK`
  and carries the score through the `scf.for` iter-args to overlap QK with the
  current tile's softmax. The genuine **D64 win** (8.6 → 11 TF) and a D128
  regression (carry causes spilling) — the register-headroom mirror of `fuse_k`.
  Driver axes `--sched 0 1` (`sched_group_barrier` hints, noise-level),
  `--fusek auto 0 1`, and `--pxpose lds shuffle` (the ds_bpermute register
  P-transpose; structural dead-end, see above). `PipelinedCfg.p_xpose` defaults to the
  measured winner `"lds"`.
- `combo.py` — the full cartesian-product sweep: every compatible single-wave
  lever across `fmha_singlewave`/`fmha_pipelined`/`fmha_blockn`, per shape, each GPU-verified
  and timed; reports the best combination per shape and globally. Answered "did
  any *combination* beat each lever alone?" — no (best is always a single lever;
  0 combinations robustly clear 20%). `--quick` for a 2-shape smoke.
- `survey.py` — Part 3: the 25-shape survey. Runs both best single-wave kernels
  per shape (keeps the faster verified one), times PyTorch SDPA (MATH fallback),
  reports TFLOP/s, % of the 59 TF peak, and speedup. The vehicle that surfaced
  the causal early-exit win (`fmha_pipelined`/`fmha_singlewave` `mask_mode=="causal"` clamp
  the K-loop to skip fully-masked tiles).
