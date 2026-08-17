# CK DSL Unified Attention 2D — gfx942 (MI300X) Experiment Summary

Date: 2026-06-05 (consolidated end-to-end arc; supersedes the per-batch living-document drafts)
Scope: fp16/bf16 prefill-2D unified attention SDPA-fwd on **gfx942 (CDNA3 / MI300X)**
(`instances/gfx942/attention_tiled_2d.py` narrow path; the wide flash regime lives in the
provider `compile_service.py`). The gfx950 study `attention_2d_experiment_summary.md` is a
separate effort; its wide-K `16x16x32` / `32x32x16` atoms and transposed/combo ladder are
gfx942-illegal (rejected in `__post_init__`) — the gfx950 playbook does not transfer, so this
document states gfx942 facts directly rather than by contrast.

This file is the full, self-contained per-experiment log: the distilled
arch-specific lessons (ISA constraints, no-transpose-read operand feed,
occupancy + compile-time budgets) and the kept/reverted-lever ledger both live
here. The doc-grounded ISA facts are summarized in the architecture reference
[`optimization/arch/gfx942.md`](../optimization/arch/gfx942.md).

**Final state (TL;DR).** The arc runs baselines → bound (LDS-latency-bound at WG=64) → a ladder
of levers. **Shipped peak: `wide4` (WG=256 / num_warps=4 wide-tile), default-on for gfx942 D128
fp16, +19.7% over L4 (153.6 → 183.8 TF), 63% of PyTorch flash, gfx950 byte-identical.** The last
open lever — **conflict-free-V via a store-path transpose (`cfv-store`) — is PARKED as a proven-
negative**, blocked on a ~1/8 D128 sign-flip store-mapping bug (an implementation bug in the
per-lane V distribution, NOT an ISA ceiling — flash and CK-Tile are existence proofs at ~290 TF).

**Status legend:** `SHIPPED`/`KEPT` (merged, net-positive) · `REVERTED` (measured, net-negative) ·
`PARKED` (proven-negative as implemented, resumable) · `DEFERRED` (out of scope / blocked on a
precondition).

> **Backend-validity note (read first).** Perf and register/occupancy numbers in
> this log were measured on an **llvm20 + comgr 7.0.1** backend. The production
> backend is **llvm22 + comgr 7.2** (selected by importing `torch` first). On
> gfx950 the same llvm20→llvm22 switch erased an apparent d128 register/occupancy
> wall (256 VGPR + spills / 1 WG/CU on llvm20 → 213 VGPR / 0 spill / 2 WG/CU on
> llvm22) and moved the prefill gap from ~0.5–0.7× to ~0.95–1.01× Triton. The
> gfx942 **AGPR-residency / occupancy / vs-flash** numbers below (e.g. the
> `archVGPR 128 / accumVGPR 384`, the `~1 wg/CU`, the `340` V+A count, the
> `63 % of flash`, and the `~0.53–0.90× vs flash` standing) are therefore
> flagged as **backend-dependent and pending production-llvm22 re-validation**.
> mfma-vgpr-form may eliminate the AGPR spill here too, as it did on gfx950.
> ISA-availability and correctness facts are backend-independent and stand.

---

## Baseline

The gfx942 path was implemented + correctness-validated separately (see
`project-gfx942-tiled-attention`). This study covers the **performance** push on top of that:
correctness must stay **6 PASS / 2 SKIP (D256) / 0 FAIL** vs `CpuFpReferenceSdpa`, and gfx950
stays byte-identical (all kernel edits live in `instances/gfx942/`).

### The binding constraint (Phase 0 diagnosis)

gfx942 = 64 KB LDS / CU. `probe_occupancy --arch gfx942` confirms the entire admitted config
family is **LDS-bound at exactly 1 CTA/CU**; spill=0 everywhere, VGPR/AGPR/SGPR never bind.

| config | VGPR/lane | LDS B | CTAs/CU | limited_by |
|---|---:|---:|---:|---|
| D128 nw2/mw16/t64 (D128 ship cfg) | 231 | 61,952 | 1 | LDS |
| D64 nw4/mw32/t64 (D64 ship cfg) | 134 | 51,200 | 1 | LDS |
| D64 nw2/mw32/t32 (B32 ship cfg) | ~ | ~30,720 | 2 | LDS |

- **2-CTA/CU threshold on gfx942 = ≤ 32,768 B/CTA** (64 KB / 2).
- LDS blocks (bpe=2): `K_lds = 2·T·HD·2` (double-buffered, dominant on D128 ~32 KB),
  `V_lds = T·HD·2`, `P_lds = BLOCK_M·(T+8)·2`, `Acc_lds = BLOCK_M·OUT_STRIPE·2`.
- `register_pv` removes `P_lds`: at D64 nw4/mw32/t64 that is 18,432 B → 51,200 − 18,432 =
  **32,768 = exactly the 2-CTA line.** This is the cleanest single path to 2 CTAs/CU.
- D128 maxes at nw2/t64 (nw4/nw8 D128 are LDS-rejected by the dispatcher gate); reaching 2
  CTAs/CU on D128 would require K single-buffering *and* `register_pv` — the hard case.

### Scoreboard at baseline (pre-fix → after the kept selector fixes)

TFLOPS, causal, `4·B·Hq·S²·D`, median/50. `analytic` is what ships on gfx942 (the ML model is
gfx950-trained → `ml=UNAVAILABLE`; `selectAnalyticFallback` is the gfx942 path). `oracle` = the
best enumerated config per shape; `PyTorch` = aotriton flash (TheRock gfx94X nightly).

| Shape | dtype | analytic (pre) | analytic (now) | oracle | PyTorch | now %PT |
|---|---|---:|---:|---:|---:|---:|
| GQA S2048 D64 | f16 | 83.8 | **142.6** | 142.6 | 260 | 55% |
| InFam GQA8 D64 S2048 | bf16 | 84.2 | **165.3** | 165.3 | 303 | 55% |
| InFam GQA8 D64 S2016 B32 | bf16 | 124.3 | **165.1** | 165.4 | 256 | 64% |
| GQA S2048 D128 | f16 | 111.6 | **115.7** | 115.8 | 293 | 40% |
| GQA S4096 D128 | f16 | 124.8 | **127.4** | 127.6 | 280 | 46% |
| MHA S2048 D128 | f16 | 108.5 | 108.3 | 108.2 | 292 | 37% |
| GQA B4 S2048 D128 | f16 | 123.6 | **127.6** | 127.6 | 422 | 30% |
| Bf16 GQA S2048 D128 | bf16 | 107.9 | **111.3** | 111.4 | 289 | 39% |
| GQA B8 S2048 D128 | f16 | 125.4 | **128.5** | 128.5 | 421 | 31% |
| GQA B4 S4096 D128 | f16 | 127.3 | 127.3 | 127.5 | 301 | 42% |
| GQA S8192 D128 | f16 | 128.2 | 128.5 | 128.8 | 532 | 24% |

Geomean analytic ≈ **35% → 41% of PyTorch** after the two selector fixes; analytic now tracks
the oracle ceiling cohort-wide. **The selection gap is closed; all remaining headroom is the
kernel ceiling**, and the ceiling gap is widest on the large-S D128 shapes (S8192 = 24% of PT).

---

## Methodology — the optimization loop

Two nested loops. The **inner loop** runs one experiment; the **outer loop** generates the next set.

**Inner loop (one lever):**
`hypothesize → verify correctness (6/2/0) → measure (probe_occupancy + oracle vs PyTorch) →
keep or revert → explain via probe/ISA diff → record here.`
Non-negotiable: correctness gates before perf; no verdict is recorded without a mechanistic
explanation (the *why* is what feeds the outer loop). One lever per change.

**Outer loop (what to try next) — the bound is the generator:**
1. **Re-diagnose the bound after every kept change** (`probe_occupancy`, then rocprof). The bound
   names the eligible levers: while `limited_by == LDS`, only LDS-cut / occupancy levers move the
   ceiling — scheduling hints are a priori dead.
2. **Mine the last experiment's evidence — especially negatives.** Each measurement exposes a new
   axis (e.g. swizzle exposed that VGPR is tight on nw4/mw32; that now informs every nw4/mw32 lever).
3. **Consult the gap map** (oracle vs PyTorch) to bias toward where headroom is concentrated.
4. **Completeness critic:** what shape / knob / modality is untested? (deferred levers re-enter when
   their precondition is met).
5. **Rank survivors by `impact × independence ÷ risk`; batch to the 3-worktree budget.**

Measurement mechanics (gfx942): static probes (`probe_occupancy`, `probe_isa_inspect`) run
per-worktree with no build (they import `rocke` from cwd). The correctness/oracle gtest binary
bakes the **feature** worktree's `rocke` path (`CkDslProviderPaths.cmake`; `PYTHONPATH` ignored),
so authoritative on-device measurement of a WIP kernel edit is done by copying the edited file into
the feature tree, running, then `git checkout --` to restore — inherently serial on the one GPU.

---

## Master lever ledger (the whole arc)

The complete kept/killed tally. Selector levers (S1/S2) and kernel levers below; the per-experiment
detail is in the batch sections. **Final tally: 5 wins shipped, 9 levers killed/parked with proof.**

| Lever | Kind | Status | Headline result | Mechanism / why |
|---|---|---|---|---|
| S1 D64 mw=32 + 1× tile | selector | **SHIPPED** | 3 D64 shapes → oracle (1.33–1.96×) | oracle-best D64 candidate was un-selectable; mw=32+1×-tile fits 64 KB, 2×-tile doesn't |
| S2 D128 GQA early-V | selector | **SHIPPED** | +2–4% on 6 GQA D128 | early-V hides V load at no LDS cost; gated GQA + seqlen≤4096 (inverts at S8192, MHA) |
| V_lds padding (+8) | kernel | **SHIPPED** | +2–5% every D128 shape | only DMA-compatible de-conflict; drops D128 V read 4-way → 2-way |
| L4 transposed-x8 + K-1buf | kernel | **SHIPPED** | 148–178 TF, +18–33% over narrow base | wide `32×32×8` atom + register-P^T (drop P_lds) + K single-buffer; banked floor before wide4 |
| **wide4 (WG=256 / nw=4)** | kernel | **SHIPPED (PEAK)** | **+19.7% over L4 (153.6→183.8 TF), 63% of flash, default-on** | 4 resident waves/WG hide LDS latency (wave-slot fill 4.59→7.59%); flash's WG regime |
| X P/Acc XOR swizzle | kernel | **REVERTED** | D128 noise; bf16 D64 −14.7% | per-row address recompute blew VGPR 143→228; freed LDS doesn't cross 2-CTA line |
| R register-PV (bf16 D64) | kernel | **REVERTED (marginal)** | 2 CTA/CU reached, only +2.6% (1 shape) | reshape (+129 bpermute/+384 swizzle) eats the occupancy gain |
| A async LDS-DMA 1→2 dword | kernel | **REVERTED (dead)** | b64 IR-illegal; b96/b128 abort comgr | no wider legal async-LDS-DMA on CDNA3 |
| L1 conflict-free V LDS layout | kernel | **REVERTED** | no full design beats the committed pad | async DMA can't make col-major; gfx942 has no `ds_read_tr_b16`; 2/bank floor |
| L2 register-V shuffle transpose | kernel | **REVERTED (flat)** | conflict-free read but flat to −1.4% | +256 `ds_swizzle` are LDS-port ops, serialize on `lgkmcnt` (18→234) |
| FA-4 rescale-skip (Stream D) | kernel | **REVERTED (flat)** | ±1% on 5/7, slightly neg on 2 | DSL has no warp-ballot / predicated `scf.if`-yield; branch-free select still issues the rescale |
| Stream B cfv read/fill-path | kernel | **REVERTED (superseded)** | bug-prone fill; net regression | read/fill-path transposed-V (OOB scatter, col≥8, causal NaN); superseded by store-path vehicle |
| **cfv-store (store-path xpose)** | kernel | **PARKED (proven-negative)** | blocked on ~1/8 D128 sign-flip store bug | right vehicle (perm proven, layout/consumer proven) but a store per-lane distribution bug; NOT an ISA ceiling |
| G agpr-alloc / D128 K-1buf (early) | kernel | **DEFERRED** | never triggered | bound never shifted to VGPR; occupancy plays retired by R (later re-opened by wide4 evidence) |

---

## Experiment results

### S1 — D64 plain-atom mw=32 + 1× tile selection [KEPT]

Goal: the analytic selector hardcoded `block_m_per_warp=16` + 2×-tile, so the oracle-best D64
`nw4/mw32/t64` plain-16x16 candidate (enumerated) never won.

Result: KEPT. All three D64 shapes move to oracle. Commit `660d6e2c0ff`.

Findings:
- On gfx942 D64, oracle is **always mw=32** with a **1× tile** (`tile_size == block_size`). The
  mw=32 + 2×-tile combo (nw4/mw32/t128 ≈ 70 KB) is LDS-rejected; the 1× combo (≈ 51 KB) fits.
- `num_warps` keys on **block_size, not batch**: bs≥64 → nw4 (BLOCK_M=128 amortizes T=64 KV/iter),
  bs=32 → nw2. (The oracle case name `…S2016_B32` is mislabeled — its `batch` field is 1.)
- The gfx950 comment that plain-atom mw=32 is a "latent trap" is a gfx950 fact (4→2 CTA/CU loss);
  on gfx942 we are already at 1 CTA/CU, so that reasoning does not apply.

Evidence: oracle `analytic=` jumped 84.3→142.6 / 84.2→165.3 / 124.3→165.1, each landing on the
shape's `best=`. Arch+head gated; gfx950 / D128 untouched.

Action: shipped in `SdpaCandidateSelector.cpp analyticTarget` + mirror in DSL
`_select_2d_block_m_per_warp`.

### S2 — D128 GQA early-V schedule selection [KEPT]

Goal: `early_v` (overlaps V load with QK+softmax, no LDS cost) is the oracle-best flag on the GQA
D128 shapes, but `analyticCloseness` penalized it, so the analytic picked the plain variant.

Result: KEPT. +2–4% on 6 GQA D128 shapes. Commit `5e1ecbe45cb`.

Findings:
- Two carve-outs from oracle data: **MHA** D128 oracle-best keeps the plain schedule (gate to
  GQA, `num_queries_per_kv>1`); and the benefit **inverts at S8192** (V load already well-hidden) —
  the first cut regressed S8192 ~1%, fixed by gating `seqlen_q ≤ 4096`.

Evidence (analytic, post-S1 → post-S2): S2048 D128 111.6→115.7, S4096 124.8→127.4, B4_S2048
123.6→127.6, bf16 S2048 107.9→111.3, B8_S2048 125.4→128.5. MHA and S8192 unchanged.

Action: `prefer_early_v` target field set for gfx942 D128 GQA seqlen≤4096; closeness rewards
early-V there, penalizes it elsewhere.

### X — P_lds / Acc_lds XOR swizzle [REVERTED]

Goal: remove LDS bank conflicts (and the `+8` P_lds pad) via XOR swizzle on the 64 KB part.

Result: REVERTED — net-negative. WIP-X not merged.

Findings:
- The epilogue **global store was already wide** (`global_store_dwordx4` / fp16x8) — the Phase-0a
  "narrow vmem_store=4" was the count, not a width problem. Epilogue vectorization is a no-op here.
- The swizzle's per-row address recompute pushed **VGPR 143 → 228** on the bf16 D64 nw4/mw32 path
  and tanked it; the freed ~2 KB LDS does not cross the 2-CTA threshold, so no occupancy offset.

| Shape | swizzle off → on | verdict |
|---|---|---|
| Bf16 InFam GQA8 D64 S2048 | 165.3 → 141.0 | **−14.7%** |
| D128 cohort | within ±0.8% | noise |

Action: reverted. Carries forward the insight that **VGPR is tight on nw4/mw32** — a constraint
for every nw4/mw32 lever (and a reason `agpr` (G) is likely worthless while LDS-bound).

### R — register-PV on bf16 D64 [MARGINAL — measured, not shipped]

Goal: drop `P_lds` to reach **2 CTAs/CU** on bf16 D64 nw4/mw32/t64 (51,200 → 32,768 B) and test
whether 1→2 CTA/CU converts to TFLOPS or the `_permute_p_c_to_a16` cross-lane reshape eats it.

Result: **2 CTAs/CU confirmed, but the win is only +2.6% on one shape.** Measured by adding a
bf16 `register_pv` candidate to the C++ enumerator and running the oracle (change reverted after
measurement — not shipped).

| bf16 D64 shape | regpv=0 best | regpv=1 best | verdict |
|---|---:|---:|---|
| InFam GQA8 D64 S2048 (nw4/mw32/t64, was 1 CTA/CU) | 165.0 | **169.3** | regpv wins **+2.6%** |
| InFam GQA8 D64 S2016 B32 (nw2/mw32/t32, already 3 CTA/CU) | 166.6 | not picked | regpv loses |

Findings:
- Occupancy probe confirmed the LDS math exactly: nw4/mw32/t64 51,200 → 32,768 B, **1 → 2 CTA/CU**,
  VGPR unchanged at 143 (no blowup, unlike swizzle).
- But the reshape is heavy: register_pv adds **+129 `ds.bpermute` + ~384 `ds.swizzle`** (IR doubles,
  3,241 → 6,475 lines). The shuffle-ALU consumes most of the 2× occupancy → net +2.6%.
- register_pv only helps configs actually stuck at 1 CTA/CU (nw4/mw32/t64). For nw2/mw32/t32 (already
  3 CTA/CU) the occupancy gain is moot and the reshape is pure cost → it loses.

**Decisive diagnostic:** going from 1→2 CTAs/CU buys only ~2.6%, so the kernel is **compute/reshape-
bound within the CTA, not occupancy/latency-bound**. This retires the "raise occupancy" thesis as
the dominant lever and keeps D128 K-single-buffering dead (it only mattered if 2-CTA paid big).

Action: not shipped (the +2.6% on one shape requires selector plumbing + `_permute_p_c_to_a16`
correctness verification on the gfx942 narrow lane map + unit-test churn — poor effort/reward).
Reproducible via the enumerator one-liner if revisited.

### A — async LDS-DMA width 1 → 2 dword [REVERTED — architectural dead-end]

Goal: if gfx942 legalizes **b64** (2-dword) load-to-LDS, halve the async-DMA call count
(≈4× heavier on D128).

Result: **DEAD.** The IR builder rejects 2-dword outright:
`async_buffer_load_lds_addr dwords must be 1, 3, or 4 (got 2)`. b64 is not expressible; the 3/4-dword
(b96/b128) widths are exactly the documented CDNA3 comgr-abort cases. There is no wider legal
async-LDS-DMA on gfx942 — width=1 is the only viable setting. No code changed.

---

## Batch 1 outcome — the selector pivot (interim; superseded by later batches)

**The cheap selector levers delivered the early gains** (S1+S2: analytic 35% → 41% of PyTorch, now
tracking the oracle ceiling cohort-wide). The **kernel** levers in this batch did not move the
ceiling: X (swizzle) net-negative, A (async-DMA) architecturally dead, R (register-PV) only +2.6% on
one shape — R's verdict reads as "1→2 CTA/CU buys ~2.6%, so the kernel is compute-bound within the
CTA." This batch ended believing the remaining D128 gap was a *genuine architectural limit*.

> **Both that "compute-bound" read AND the "architectural limit" conclusion are later corrected.**
> Batch 2.1's rocprof re-diagnoses the bound as **LDS bank conflicts** (not compute), and Batches 4–5
> prove the limit is an *implementation* gap, not architecture (wide4 ships +19.7%; flash/CK-Tile are
> existence proofs at ~290 TF on this silicon). The authoritative final policy + gaps live in
> **Batch 5 → "Current best policy (final)"** and the **master lever ledger** at the top of this doc.
> The next-lever discipline that survived from here: re-diagnose the bound after every kept change,
> then mine the negatives (rocprof seeds the next batch with evidence, not hypothesis).

---

## Batch 2.1 — rocprof the bound [done; supersedes the Batch-1 "compute-bound" framing]

rocprofv3 on the ship configs (logs `WIP/.../rocprof/`). The bound is **LDS bank conflicts**, not
compute — correcting the post-R wording: the MFMA pipe is ~95% idle because warps stall on LDS.

| metric | D128 (40% PT) | D64 (55% PT) |
|---|---:|---:|
| MFMA pipe busy | **4.63%** | 6.64% |
| busy cycles waiting on LDS (`SQ_WAIT_INST_LDS/BUSY`) | **0.574** | 1.580 |
| bank conflicts / LDS instr | **11.34** | 6.61 |
| LDSBankConflict (derived) | 40.3% | 30.0% |
| MemUnitStalled | ~0% | ~0% |
| inst mix | VALU 54% / LDS 20% / MFMA 7% | VALU 70% / LDS 14% / MFMA 5% |

Diagnosis: **LDS-bank-conflict bound** (D128: 11.3 conflicts/LDS inst, ~57% of busy spent waiting on
LDS), MFMA-starved, softmax-VALU-heavy, NOT memory- or occupancy-bound. This also explains R
(2 CTA/CU doesn't relieve per-access conflict serialization) and X (it swizzled `P_lds`/`Acc_lds`,
but the hot conflict is the **strided-V B-operand LDS read** — the gfx942 narrow path emulates the
`ds_read_tr16` lane map with strided loads; X targeted the wrong tile, hence D128 unmoved).

**Next experiment (Batch 2.2):** targeted de-conflict of the strided-V (and K_lds) read — a swizzle
or layout change on that specific access, with cheap addressing to avoid the VGPR blowup that sank
the blanket swizzle (X). Validate with the same rocprof counters: `SQ_LDS_BANK_CONFLICT` per LDS
inst and MFMA-busy should rise on D128.

---

## Batch 3 — the flash target, the first kernel win, and the LDS dead-end

**Flash is the proof there is huge room — and the gap is one design decision.** rocprofv3 on PyTorch
aotriton `attn_fwd`, D128, same counters as ours:

| D128 | flash `attn_fwd` | ours |
|---|---:|---:|
| bank conflicts / LDS-inst | **0.58** | 11.34 |
| LDSBankConflict (derived) | **1.0%** | 40.3% |
| MFMA-util | 14.3% | 4.6% |
| LDS-wait / busy | 0.12 | 0.57 |
| MemUnitStalled | 20.9% (HBM-bound — the *right* bound) | ~0% |

Flash is essentially conflict-free and near the HBM roofline; we stall ~57% on self-inflicted V-read
conflicts. The entire gap is the strided-V emulation. (Attention is softmax-VALU-heavy, so even flash's
MFMA-util is only 14% — MFMA was never the prize; *not stalling on LDS* is.)

**V_lds padding [KEPT, commit `a7dcd8af487`] — first kernel win.** A read-side XOR is infeasible (async
DMA writes lane-contiguous; no per-lane dest). Padding the V_lds row stride (+8 halves) — the only
DMA-compatible lever — drops the D128 read 4-way → 2-way. **+2–5% on every D128 shape** (GQA S2048
+4.6%), VGPR-neutral (226→230), correct 16/2/0 on the expanded net. D64 auto-disabled (can't help:
8 rows/DMA-call). Env-gated `HIPDNN_GFX942_SWIZZLE_VLDS` (default on); gfx950 byte-identical.

**L1 conflict-free V LDS layout [REVERTED — proven net regression, not shipped].** Goal was a wide
bank-spread `ds_read_b64` from a re-laid-out V_lds. Conflict-cycle accounting (store+read/kv-iter):
baseline 256 → committed pad **128** → best wide-layout 192 → reshape 704. **No full design beats the
committed pad**, because (1) the async DMA can't produce a col-major layout (only the wave-uniform base
is free — which the pad already uses), and (2) the transpose is symmetric and **gfx942 has no
`ds_read_tr_b16`** (the fp8 stripe precedent works only because `ds_read_tr_b8` exists). Element-degree-1
is also mathematically impossible (64 words / 32 banks = 2/bank floor; CK's `MakeVLdsBlockDescriptor`
pays it too). **The committed padding is the Pareto-optimal LDS-only lever on this ISA.** Analysis:
`WIP/.../WIP-L1/HANDOFF_FINDINGS.txt`.

**Corrected understanding:** flash's advantage is NOT a clever V *LDS* layout — it's **not round-tripping
V through LDS at all.** Closing the gap requires **L2: register-resident V for PV** (CK `vr` pipeline) —
removes the V LDS read entirely (no transpose, no store conflict). Favorable: we're LDS-bound at 1
CTA/CU with VGPR headroom (~176/512), and dropping V_lds also frees LDS. Risk: VGPR pressure could wall
it. **Next: feasibility-check L2's VGPR budget before committing to the rewrite** (same discipline that
caught L1). L3 (32x32x8 atom) and L4 (prefetch) remain second-order until V is off the LDS-stall path.

**L2 register-V + in-register shuffle transpose [REVERTED — flat, lgkmcnt serialization].** Feasibility
passed (VGPR *lower* 240→176, V is the whole bound, partial-rewrite scope). Prototype (D128, flag
`HIPDNN_GFX942_REGISTER_V`) made the V read conflict-free (1 `ds_read_b64` at the floor, `ds_read` −73%)
and is correct (16/2/0 on the expanded net). BUT the in-register transpose's **+256 `ds_swizzle` are
LDS-port ops that serialize on `lgkmcnt` (18→234)** — on-device D128 = **flat to −1.4% vs the committed
padding.** The static red flag was correct: the swizzle serialization costs what the conflict-removal
saves. (Strategy-2 — half the swizzle moved to VALU `v_perm` — is the only untried refinement; bounded
upside.)

**Batch-3 conclusion (LATER RETRACTED): "the V-read bound is extracted to its gfx942 ceiling."** The
reasoning held *for the LDS-only V-transpose lever*: with no `ds_read_tr_b16`, every LDS-resident
transpose hits *either* bank conflict (strided) *or* `lgkmcnt` serialization (swizzle), so the committed
V_lds padding (+2–5%) is the Pareto frontier *for that lever class*. **But the leap from "this lever is
maxed" to "the kernel is at a gfx942 ceiling" was wrong** — Batch 4 rewrites to the wide `32×32×8`
flash regime (L4 ships +18–33%), Batch 5 ships wide4 (+19.7% via WG=256), and aotriton/CK-Tile run
~290 TF on this exact silicon (existence proofs). The true remaining lever is **conflict-free V via a
*store*-path transpose + more waves/WG**, not a V *read* lever and not an ISA wall. **Ledger at end of
Batch 3: 3 wins shipped (D64 sel, D128 early-V, V padding → analytic 35%→41% of PT); levers killed with
proof (X, A, R, L1, L2).**

---

## Batch 4 — the flash-pipeline rewrite: L4 shipped (transposed-x8 + K-single-buffer)

Batch 3 concluded the gap to flash was "structural pipelining, a larger rewrite." Batch 4 **is**
that rewrite: a fresh `32×32×8` wide-atom flash pipeline on the gfx942 path, built correctness-first
(Stream A foundation) then layered. The wide regime lives in the provider `compile_service.py`
(`HIPDNN_GFX942_FLASH_PIPELINE` level 0–5, default OFF); gfx950 byte-identical by construction.
L4 merged at commit `9cff43bdd2d`.

### Phase-A correction — what flash actually does (re-profiled)

**Batch 3's claim that "flash never round-trips V through LDS at all" is WRONG — corrected here.**
Fresh kernel-trace + counters of PyTorch aotriton `attn_fwd` on gfx942 (MI300X), D128 fp16:

| flash `attn_fwd` (D128 fp16) | value |
|---|---:|
| LDS (dynamic) | **32,768 B** (V *is* staged in LDS) |
| bank conflicts / LDS-inst | **0.58** (conflict-free) |
| LDSBankConflict (derived) | 1.0% |
| archVGPR / accumVGPR | **128 / 384** (acc lives in AGPR) |
| SGPR / scratch / WG | 112 / 64 / **256** (num_warps=4 → 4 waves/WG) |
| num_stages | 1 |
| occupancy | ~1 wg/CU (32 KB LDS, high AGPR) |
| MFMA util | 14.1% |
| MemUnitStalled | 21.8% (HBM-bound — the *right* bound) |

(D64 variant: 16,384 B LDS, archVGPR 128 / accumVGPR 128 — same num_warps=4, smaller-tile/low-AGPR.)

So flash **does** put V in LDS. Its edge is not deep prefetch (num_stages=1) and not high wg/CU
occupancy (~1 wg/CU, same as ours): it is **(a) a conflict-free V LDS read** (0.58 vs our 7–11
conflicts/inst), **(b) operand residency** (P and acc kept off LDS — acc in AGPR, no P_lds bridge),
and **(c) WG=256 = 4 waves/WG** (vs our WG=64 = 1 wave). The kernel is HBM-bound, the correct
roofline. Items (a) and (b) drove Batch 4 (L4); item (c) — the one Batch 4 missed — drove the
wide4 win in Batch 5.

### Lever ledger (Batch 4)

| Lever | Kind | Status | Headline result | Mechanism / why |
|---|---|---|---|---|
| 32x32x8 base (Stream A) | kernel | **KEPT (foundation)** | correctness-first wide-atom; ~2× *slower* than narrow alone (naive V) | wider K=8 atom alone cut conflicts 11.34→7.25/LDS-inst, but the naive strided V keeps it LDS-bound (MFMA 7%) |
| transposed-x8 / register-P^T (Stream C) | kernel | **KEPT** | **+1.57×** over plain-x8 (up to 1.82× @ S8192); −62% LDS insts | S^T = K·Q^T so P^T is the PV B-operand *direct from registers* → P_lds round-trip eliminated (SQ_INSTS_LDS 767K→290K) |
| FA-4 rescale-skip (Stream D) | kernel | **REVERTED (flat)** | within ±1% on 5/7, slightly neg on 2 | DSL has no warp-ballot / predicated `scf.if`-with-results → the `o_acc *= alpha` rescale still issues (runtime `alpha`, backend can't fold `x*1.0`); only P's frame changes. Branch-free select form gated to `use_mfma_32x32x8`; commit `45c9ac63179` |
| Stream B cfv read/fill-path | kernel | **REVERTED (superseded)** | bug-prone fill; never net-positive | the read/fill-path transposed-V approach (store V transposed via a scatter/sync fill, read one wide `ds_read_b64`) hit a chain of fill bugs: OOB scatter on `T<THREADS` (`97e1f76621e`), col≥8 corruption (fixed via compile-time col, `70810e963b2`), causal-tile NaN (`127aa493d52`), and needed the `TV_NARROW_READ` diagnostic (`7bf57e71ad9`) to isolate read-width from fill. **Lesson: the read/fill-path transpose is fragile; the right vehicle is the *store*-path transpose (revisited in Batch 5 as cfv-store).** |
| acc→AGPR / drop Acc_lds | kernel | **REJECTED (evidence gate)** | not implemented | STEP-0 proved the 48 KB peak = K_lds+V_lds only; the backend LDS-coalescing pass already aliases Acc_lds into the loop-dead K/V region → dropping it is moot for occupancy |
| K single-buffer (prong2) | kernel | **KEPT → part of L4** | crosses the 32 KB threshold | K double→single buffer drops LDS 48→32 KB → 1→2 wg/CU at WG=64 |

### The valid perf ladder (fp16 D128)

Levels are cumulative env settings. base = the narrow 16x16x16 ship path (L0). Geomean over the
7 flagged fp16-D128 shapes (Stream C job 354226 / prong2 job 354431); flash ≈ 290 TF.

| level | path | geomean / representative TFLOPS | vs base | vs flash |
|---|---|---:|---:|---:|
| L0 | narrow 16x16x16 baseline (ship) | **126.6** | 1.00× | 0.44× |
| L1 | plain-x8 (32x32x8, P_lds bridge, naive V) | 59.5 | 0.47× | 0.21× |
| L2 | transposed-x8 (register-P^T, naive V) | 93.3 | 0.74× | 0.32× |
| **L4** | **transposed-x8 + K single-buffer** | **156–178** | **+18–33%** | **~0.62×** |

Plain-x8 (L1) is the correctness-first foundation and is *below* the narrow baseline — the naive
strided V plus the P_lds bridge dominate. The transposed register-P^T handoff (L2) recovers most of
that deficit (+57% over L1), and K single-buffering (L4) crosses the occupancy threshold to land
**above** the narrow baseline. Per-shape L4 (prong2/prong3 medians):

| shape (D128 fp16) | L0 base | L2 (x8) | **L4 (x8+k1buf)** | L4 vs base |
|---|---:|---:|---:|---:|
| GQA S2048 | 120.9 | 81.8 | **147.7** | +22% |
| GQA S4096 | 134.4 | 91.2 | **165.8** | +24% |
| MHA S2048 | 113.3 | 91.3 | **132.9** | +17% |
| GQA B4 S2048 | 133.7 | 90.2 | **171.5** | +28% |
| GQA S8192 | 135.0 | 113.7 | **178.9** | +33% |
| GQA B8 S2048 | 134.9 | 91.9 | **174.6** | +29% |
| GQA B4 S4096 | 137.3 | 95.6 | **173.0** | +26% |

Best observed at the close of Batch 4: **178.9 TF (GQA S8192) ≈ 62% of flash.** D64 fp16 and all
bf16 are controls (flag inert) and stay flat across levels — measurement validated. **L4 ships:
the analytic selector auto-selects it for gfx942 D128 fp16 (commit `9cff43bdd2d`).**

### Mechanism for L4 (K single-buffer → 2 wg/CU at WG=64)

STEP-0 (static gate) decomposed the 48 KB peak exactly: `K_lds = 2·T·HD·2 = 32 KB` (double-buffered),
`V_lds = T·HD·2 = 16 KB` (already single-buffered), `P_lds = 0` (register-P^T), `Acc_lds` aliased
(see rejected lever above). The only lever that crosses the 32 KB / 2-wg threshold is cutting loop LDS,
and V is already single-buffered — so **K single-buffer** (32→16 KB) → loop LDS = 16+16 = **32 KB →
2 wg/CU** (rocprof-confirmed: `..._earlyv` 49,152 B / 1 wg/CU → `..._earlyv_k1buf` 32,768 B / 2 wg/CU).
The single-slot K introduces a read-race (next-tile async DMA write `K[i+1]` vs the tail of `QK[i]`'s
LDS reads); **closed by a post-QK `s_waitcnt(lgkmcnt=0)` + barrier** before the next-K DMA. Numerics
unchanged; the fitter enforces `BLOCK_M ≤ tile_size` for L4 (fixes the S528 multi-tile edge case).
The prior R-lever floor (1→2 wg/CU = +2.6% on a conflict-bound kernel) did *not* hold here: on these
latency-bound long-prefill shapes the doubled occupancy buys +17–33% over baseline. **Note: L4 is
2 wg/CU at WG=64 (1 wave/WG) — this is the regime wide4 later overturns by going wide instead of deep.**

---

## Batch 5 — wide4 SHIPPED (WG=256), and cfv-store proven-negative

Batch 5 closed the arc with two decisive results: the **shipped peak (wide4)** and the **final parked
proven-negative (cfv-store)**. The framing throughout: there is **NO gfx942/ISA ceiling** — two working
kernels run ~290 TF (D128 fp16) on this exact silicon (**aotriton `attn_fwd`** and CK-Tile
**`block_fmha_pipeline_qr_ks_vs`**), using only primitives we already have (`32×32×8` via 2×-iterateK,
register-resident operands, async DMA, `v_perm` register transpose). The remaining gap is an
implementation gap, not architecture. Any "62% is the ceiling" reading (including Batch 3's "V-read
bound extracted to its gfx942 ceiling" line) is **retracted** by these existence proofs.

### 5a — The cfv READ-path attempt: correct but perf-negative (the WG=64 diagnosis)

The first Batch-5 cfv attempt kept the V transpose on the **read/load** side: register-load V →
in-register `v_perm` transpose → `smem_store` (loop-rolled to fix a JIT IR-explosion). It is **correct
16/2/0 and gather-free**, but measured **83.3 TF vs shipped L4's 140.6 TF** (Fp16 GQA S2048 D128) —
*below* L4. rocprofv3 (job 354640) answered which hypothesis — (a) occupancy / (b) not-conflict-free /
(c) VALU-bound:

| counter (mean/attn dispatch) | cfv (read-path) | L4 | flash (aotriton) | read |
|---|---:|---:|---:|---|
| LDSBankConflict (% cyc) | **19.4%** | 21.8% | **1.0%** | cfv NOT conflict-free — ~same as L4, 19× flash |
| **LdsLatency** (cyc) | **356** | 118 | — | **cfv 3.0× L4 — the smoking gun** |
| MfmaUtil (%) | 3.3% | 5.6% | 14.1% | cfv MFMA *half* L4 — compute idle |
| VALUBusy (%) | 7.0% | 10.6% | 21.3% | every engine idle — **NOT** VALU-bound |
| MemUnitStalled (%) | 4.2% | 8.0% | 21.8% | far from HBM-bound |
| MeanOccupancyPerActiveCU (wg/CU) | **1.0** | **1.0** | ~1 | **both 1 wg/CU — L4 is NOT 2 here at this shape** |
| OccupancyPercent (wave-slot fill) | 2.4% | 4.3% | — | cfv ~half L4's resident waves |
| WG size | 64 (1 wave) | 64 (1 wave) | **256** (4 waves) | flash = 4 waves/WG |

**Verdict: (b) NOT-conflict-free, manifesting as an LDS-latency blow-up** — the rolled read-path
transpose still collides (19.4% conflicts), driving LdsLatency to 3× L4, so MFMA collapses to 3.3%.
**Crucially (a) was a wrong premise: both cfv and L4 are 1 wg/CU at WG=64 (single wave) at this shape**
— the earlier "L4 = 2 wg/CU" belief does not hold here. (c) ruled out (VALU idle). **The decisive
re-frame:** single-wave-per-WG kernels cannot hide latency *regardless of V layout*; flash is WG=256
(4 waves). This pointed at the **wave-slot fill** (cfv 2.4% / L4 4.3% / flash much higher) as the
dominant lever — and seeded wide4. (Full: `WIP-flash-async/VERDICT_ROCPROF.md`.)

### 5b — wide4: more waves per WG [SHIPPED — the peak, +19.7%]

The hypothesis from 5a: widen the WG to put more resident waves on the CU and hide the LDS latency the
read-path transpose couldn't remove. The lever almost died on a **wiring trap** first: the perf/
correctness path is driven by the **provider** `compile_service.py` (which reimplements the gfx942 L4
spec + num_warps/tile chooser independently of the DSL `_select_2d_*`), and the provider's `_lds_bytes`
model was *triply* over-conservative for the transposed-x8 path (counting `P_lds` that the path drops,
K double-buffered when k1buf halves it, full `Acc_lds` that is backend-aliased) → it rejected nw>1 and
fell back to nw=1/WG=64. A bare env A/B also aliased the JitCache (num_warps is folded into the
`GraphSignature` key, so the env path served the cached WG=64 kernel — the same trap the cfv work hit).
**Fix: teach the provider chooser an accurate `_lds_bytes_transposed_x8` LDS model so nw>1 actually
fits, and route num_warps through the signature.** Then WG actually widened. Result (job 354653,
D128 fp16 GQA S2048):

| config | WG (block) | TFLOPS | vs L4 | %-flash(290) |
|---|---|---:|---:|---:|
| L4 (FLASH_WIDE unset) | (64,1,1) WG=64 | 153.6 | 1.00× | 53% |
| wide2 (FLASH_WIDE=2) | (128,1,1) WG=128 | 181.2 | +18.0% | 62% |
| **wide4 (FLASH_WIDE=4)** | **(256,1,1) WG=256** | **183.8** | **+19.7%** | **63%** |

wide4 drops `_k1buf` by design (`BLOCK_M=128 > T=64` → K double-buffered, LDS=48 KB, 1 wg/CU). wide2 is
the safe 2-wg/CU variant (keeps k1buf); wide4 edges it +1.7pp at 1 wg/CU. **rocprof confirmed the
mechanism exactly** (job 354659):

| counter | L4 (WG=64) | wide4 (WG=256) | read |
|---|---:|---:|---|
| **OccupancyPercent** (wave-slot fill) | 4.59% | **7.59%** | **+65% resident waves — the latency-hider** |
| MfmaUtil (%) | 6.37 | **7.48** | +17% — MFMA less starved |
| MeanOccupancyPerActiveCU (wg/CU) | 1.00 | 1.00 | both 1 wg/CU (= flash) |
| LDSBankConflict (/inst proxy) | 24.6 | 28.9 | conflicts UP (more lanes) but now HIDDEN by waves |
| LdsLatency (cyc) | 140 | 247 | per-op latency up, now OVERLAPPED by 4 waves |
| VALUBusy / MemUnitStalled (%) | 12.2 / 0.12 | 12.3 / 0.09 | not VALU- and not HBM-bound; headroom remains |

The win is **waves-PER-WG, not wg/CU** (both stay 1 wg/CU = flash's regime). Widening 1→4 waves raises
wave-slot fill +65%, overlaps the (now higher) LDS latency, MFMA climbs, perf follows. **wide4 IS
flash's WG regime.**

**Ship (commit-banked `a48bb04ac68`, env-gated; default-on promote on `users/bharriso/
gfx942-flash-regime-wip`).** Three coherent edits mirror the L4-ship precedent:
1. `compile_service.py`: `_flash_wide` defaults to 4 for the qualifying gfx942 D128 fp16 shape (no env);
   accurate `_lds_bytes_transposed_x8` picks nw=4/tile=64.
2. `SdpaCandidateSelector.cpp analyticTarget`: gfx942 D128 fp16 returns `{num_warps=4, mw=32,
   tile=64}` (was nw=1) so the folded `num_warps` cache key is coherent with the built `_w4` kernel.
3. `SdpaCandidateSelector.cpp supportsTiled2d`: accurate transposed-x8 LDS model so the nw4 candidate
   survives enumeration. Strictly gated gfx942+D128+fp16+plain-mw32.

Kill-switch `HIPDNN_GFX942_FLASH_WIDE=0` reverts the BUILT kernel to L4 (WG=64); `=2`/`=4` force a
width for A/B. **GATE 1 (job 355109):** default-on correctness **16 PASS / 2 SKIP / 0 FAIL**, D128 fp16
GQA S2048 = **182.71 TF** (kernel `..._w4_mw32_mfma32x8_stqk`, WG=256, no `_k1buf`); kill-switch reverts
to L4 156.27 TF, 16/2/0. D64 control 142.5 TF unchanged. **GATE 2 (gfx950):** 4 representative gfx950
SDPA-fwd HSACOs `cmp`/sha256 **byte-identical** before vs after (deterministic host comgr compile;
change is `arch=="gfx942"`-gated in both Python and C++). **Decision: SHIP.**

### 5c — cfv-store (store-path transpose) [PARKED — proven-negative, NOT an ISA ceiling]

With wide4 shipped, the last lever is genuine conflict-free V (flash's 0.58 vs our 28.9 conflicts/inst,
which the extra waves now *tolerate* but don't remove). 5a proved the **read-path** transpose is the
wrong place; the correct vehicle is the gfx942 playbook's **vehicle (c) = transpose V once on the
STORE path**: register `buffer_load` V in NATURAL `[token,dim]` (coalesced, pipelined) → in-thread
`perm_b32` 2×2 f16 transpose (CK `transpose_vectors` masks `0x01000504`/`0x03020706`, **no cross-lane,
no `lgkmcnt`**) → single contiguous `ds_write` into transposed `V_lds[HD,T+pad]` → conflict-free
`ds_read_b64` consumer (reused). Built off wide4 on `users/bharriso/gfx942-cfv-store-wip` (pushed),
default OFF, distinct cache key + dedicated `HIPDNN_GFX942_CFV_STORE` env (does not overload
FLASH_PIPELINE). The `perm_b32` op was cherry-picked clean (`5f5dd0c5be3`); transpose loop-rolled to
emit exactly 2 `perm_b32` (avoids the 45-min JIT IR-explosion).

**HARD GATE FAILED: cfvst is correct everywhere EXCEPT fp16-D128, where ~1/8 of elements are
sign-flipped** (4 D128 fp16 tests fail; D64 + bf16 + wide4 baseline all pass; wide4 baseline reproduced
on this branch at 183.2 TF). The bug is **real but not root-caused** despite an exhaustive isolation
matrix — and every building block is **independently proven correct**:

- **`perm_b32` transpose: PROVEN CORRECT** by a standalone gfx942 GPU micro-test (masks + src order give
  exactly `V_lds[d,t] = V[t,d]`).
- **layout + consumer: PROVEN CORRECT** — vehicle (a) (`FLASH_PIPELINE=3`) on this branch passes 4/4
  D128, so `[HD,T+pad]` V_lds + the `ds_read_b64` consumer are sound; only the STORE differs.
- **Ruled out** (all still ~12% wrong): vectorized vs scalar HBM load, vectorized vs scalar consumer
  read, both-scalar, coverage gap (prezero removes an `inf` but the sign-flip persists → all needed
  slots ARE written), dim-contiguity / separate per-dim offset, and element-wise scatter store w/o perm
  (so the `(dim,token)→slot` *mapping itself*, not the perm, is where the residual lives).
- **Consistent signature:** ~12.5% = exactly **1/8** of elements, sign-flipped, fp16-D128-cfvst ONLY.

**Strongest unresolved hypothesis (the resume point):** the consumer is the WG=256 / 4-wave wide tile;
for the transposed PV the MFMA-**A** operand is `V^T[M=dim, K=token]`, and the per-lane A-operand
register layout for the `32×32×8` atom may require a **lane-interleaved** token assignment that vehicle
(a)'s store happens to satisfy but the cfvst store (tokens laid 0,1,2,3… strictly contiguous per dim
row) does not. **Concrete next probe:** dump `V_lds` for a tiny single-tile case (S=64) from BOTH
vehicle (a) and cfvst and diff slot-by-slot — the 1/8 of slots that differ name the exact `(dim,token)`
the store mis-fills. (Full: `gfx942-cfv-store/{CFV_VERDICT.md,HANDOFF.md}`; diagnostic env knobs
`CFV_STORE_{SCALAR_LOAD,SCATTER,PREZERO,SEPOFF}` left gated-off in the kernel.)

**Why this is PARKED, not killed-as-ceiling:** the vehicle is the right shape (playbook + CK
`qr_ks_vs` confirm it), the perm primitive works, and the layout/consumer are proven — the blocker is
an **implementation bug in the store's per-lane V distribution**, fixable, just not localized within the
session budget. rocprof of cfvst was **BLOCKED** (the cfvst kernel ran >37 min under rocprofv3 without
emitting a CSV, walled at 45 min — a yellow flag, but moot while correctness gates it). If the 1/8
mismatch ever proves to be an inherent conflict between the 4-wave MFMA-A lane layout and any in-thread
store-transpose tiling, *that* would be the real mechanism-bearing negative — but it is not proven so
today.

### Current best policy (final)

- **D128 fp16**: **wide4 (WG=256 / nw=4) ships default-on** (analytic auto-selects on gfx942 D128 fp16;
  +19.7% over L4, 182.7 TF, 63% of flash, 1 wg/CU / 4 waves). Kill-switch `HIPDNN_GFX942_FLASH_WIDE=0`
  → L4 (WG=64). FLASH_PIPELINE levels 0–3/5, the FA-4 path, and cfv-store remain gated-OFF building
  blocks / diagnostics.
- **D128 bf16**: stays on the narrow 16x16x16 path (the x8 flash atom is fp16-only); selector unchanged
  from Batch 1–2 (plain nw2/t64 + GQA early-V seqlen≤4096).
- **D64** (fp16/bf16): unchanged from Batch 1 (`mw=32`, 1× tile, `nw = bs≥64?4:2`).
- D256 unsupported (LDS). gfx950 byte-identical.

### Current gaps to PyTorch (final)

- **D128 fp16: ~63% of flash** (was 24–46% at end of Batch 3, ~62% at L4). The residual ~37% is the
  conflict-free-V store (flash ~0.58 vs our ~29 conflicts/inst) — parked on the cfv-store 1/8 store-
  mapping bug above. The extra waves (wide4) make the kernel *tolerant* of the conflicts but do not
  remove them, so cfv-store is still the live next lever once the store bug is root-caused.
- **D128 bf16: unchanged (~39% of flash)** — no wide x8 atom for bf16 yet; the flash ladder is fp16-only.
- **D64: unchanged (~55–64% of PyTorch)** — selection gap closed in Batch 1; the flash rewrite is D128-only.

### Documented resume hypothesis (for whoever picks this up)

1. **Root-cause the cfv-store 1/8 sign-flip** via the slot-by-slot V_lds diff (5c) → ship genuine
   conflict-free V → target ~1% LDSBankConflict (≈29× cut) → MFMA off ~7.5% → toward flash's ~290 TF.
2. If conflict-free V lands, the next structural gap is **accumulator residency**: flash carries
   384 AGPR (deeper SW pipeline / larger acc tiles) vs our 152 — independent of the V fix.
3. bf16 D128 has no wide flash atom yet — a separate enablement, not a tuning lever.

---

## Update (2026-06-19) — D128 is already 2 WG/CU; the residual is scheduling, not occupancy

> **Backend caveat.** The register/occupancy and vs-flash numbers in this
> update were measured on **llvm20 + comgr 7.0.1**, not production **llvm22 +
> comgr 7.2**. On gfx950 the analogous d128 body's register/occupancy reading
> flipped between those backends; the `340` V+A count and the `~0.53–0.90×
> vs flash` standing here are **backend-dependent and pending production-llvm22
> re-validation**. The qualitative conclusion (occupancy is not the wall; the
> residual is inner-loop scheduling) is the right framing regardless of backend.

A follow-up occupancy re-check (`llvm-readelf`-exact, throttle-free)
**overturns the occupancy framing** that ran through Batches 1–4. The shipped
gfx942 D128 wide config (transposed-x8 with K **single-buffered**) read as
**already 2 WG/CU** on llvm20: LDS = 32 KB (`K_lds[1]` 16 KB + `V_lds[1]` 16 KB),
V+A = 340 (a large fraction of which is accumulator residency in the AGPR file —
exactly what the llvm22 backend reorganizes, so re-measure before citing). The
earlier "D128 wide = 48 KB → 1 WG/CU" lines (and the "narrower-N accumulator
redesign" idea) are **superseded** — K single-buffering already halved `K_lds`,
so there is **no occupancy wall left to break**. Consequently the gfx950
"small-tile breaks the 1-WG wall" thesis (see that arch's experiment summary)
**does not transfer** to gfx942.

A small-tile `T=32` **double-K** `num_warps=2` variant (LDS = 24 KB, still 2
WG/CU) does still give a robust **+11–14 %** at S≤2048 (both dtypes; fp16 S4096
0.814 → 0.903) — but via a *different* mechanism than gfx950's: it **restores the
K double-buffer prefetch** that the shipped `T=64` path gives up to fit LDS. That
is a prefetch/scheduling win, not occupancy. The literal LDS-halving
`num_warps=1` / `T=32` / K-single (4 WG/CU at 16 KB) is a **trap** (regresses
long-S; `BLOCK_M=32` starves the grid; occupancy past 2 WG/CU is useless here).

**Caveat for landing the small-tile win:** `T=32` requires a paged-KV cache
`block_size=32` (`tile_size` must be a multiple of `block_size`). If production
cache is `block_size=64`, `T=32` is illegal and the win must be re-validated
(`T=64` double-K = 48 KB → back to 1 WG/CU, likely loses it).

**Standing (llvm20-measured, re-validation pending):** on the llvm20 backend,
even *with* the small-tile win, gfx942 D128 prefill **still lost to flash**
(best ~0.53–0.67× bf16, ~0.67–0.90× fp16). Given the gfx950 llvm20→llvm22
reversal, **these ratios must be re-measured on production llvm22 before being
treated as the gfx942 gap.** The residual was attributed to **algorithmic /
inner-loop scheduling** — the same lever family as the parked cfv-store (conflict-free V via
a store-path transpose) and gfx950's S4096 push (direct-to-LDS async, staggered
partial `lgkmcnt` waits, conflict-free V). gfx942's wide bf16/fp16 atom is
`mfma_f32_32x32x8` (K=8); the K=16 atom and `ds_read_tr16` gfx950 uses for
conflict-free V are not available here.

---

## Appendix — measured arch facts (re-homed from the gfx942 arch reference)

The lean `optimization/arch/gfx942.md` keeps only AMD-doc-grounded ISA constraints; these
**measured / project-specific** numbers from this push live here so nothing is lost.

### Compile-time budget (cold JIT, gfx942)

- **≈ 250–290 s per shape-signature** for this attention path (e.g. bf16 narrow D64 ≈ 245 s;
  cfv GqaD128 ≈ 293 s). The cold-compile is **per-shape-signature** — there is no cross-shape
  cache, so every new shape pays the full cost. Treat 250–290 s as the budget to respect when
  enumerating shapes.
- **Fully-unrolled transposes explode compile time** — a fully-unrolled per-element register
  transpose hit a **45-minute comgr/JIT timeout** (also seen as the cfv-store / read-path JIT
  IR-explosion in 5a/5c). The DSL never re-rolls loops, so a Python-time `static_for` / `unroll`
  over a whole tile emits O(tile) straight-line IR. **Fix: ALWAYS loop-roll** — wrap whole-tile
  reshapes in a runtime `scf_for` over micro-tiles, keep only the tiny inner block unrolled (the
  `perm_b32` transpose in 5c is loop-rolled to emit exactly 2 `perm_b32`); the IR collapses and
  the build drops back to seconds with bit-identical numerics.
- **Prefer native vector ops** (`v_perm` / `perm_b32`, vector loads/stores) over scalar
  `vec_extract` / `vec_pack` chains — the scalar chains balloon IR for the same numerics.

### Per-opcode LDS bank-conflict period (gfx942, measured)

AMD documents LDS conflicts only qualitatively; these per-opcode periods were measured on this
push (the underlying ISA fact — 64 KB LDS / 32 banks, 2/bank floor — is in the arch reference).

| Access | Conflict period |
|---|---|
| `ds_read_b128` | 32 dwords |
| `ds_write_b128` | 32 dwords |
| Other `ds_read_b{32,64}` / `ds_write_*` | 32 dwords |
| Intra-dword sub-dword accesses | conflict-free |

### In-register reshape vehicle (gfx942)

The gfx942 in-register transpose vehicle in the DSL is **`perm_b32` (`v_perm_b32`,
`__builtin_amdgcn_perm`)** driven by CK `transpose_vectors` **mask values** (2×2 f16 transpose via
masks `0x01000504` / `0x03020706`) — not a cross-lane `ds_swizzle` / `ds_bpermute` (those are
LDS-port ops that cost `lgkmcnt`; see 5a/L2) and not `permlane32_swap` (gfx950-only). `perm_b32` has
no LDS port and no `lgkmcnt`, so it is the preferred in-register reshape on this arch.
