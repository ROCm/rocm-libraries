# gfx1151 FMHA forward — next-step plan (L=16K, closing on 25 TF)

**Scope.** Next steps for `library/kernels/gfx1151/wmma_fmha_swapqk.py` at L=16384,
D=128, H=24, B=1 on gfx1151 (Strix Halo / Radeon 8060S, RDNA3.5, wave32,
`v_wmma_f32_16x16x16_f16`). Everything below is measured on the live board; the
superseded CK-gap plan from the ~10 TF era is kept in the appendix.

---

## 0. Where we are

Best config: `mq1 of32 bn64 w2 vt dual qk_douter`, i.e.
`--mq 1 --of16 0 --block-n 64 --vt 1 --qkdo 1`.

| | value |
|---|---|
| L=16384 wall clock | **22.80 TF** mean of 3 interleaved reps (22.61 / 22.81 / 22.97), peak 23.39 |
| GRBM cycles | **15.16 Mcyc** → 21.75 TF at a pinned 2.4 GHz |
| registers | 199 VGPR, **0 spill**, 0 scratch → 208 granule → **7 waves/SIMD** |
| measured occupancy | 12.6 waves/CU |
| K-loop | 1082 instructions, 64 WMMA, 96 loads (64 global + 32 buffer) |

Cycle budget at that config (per SIMD, from `prof_full.py`):

| bucket | Mcyc | share |
|---|---:|---:|
| WMMA issue | 7.58 | 50.0% |
| other VALU issue | 2.89 | 19.0% |
| `v_exp` issue | 0.42 | 2.8% |
| **total issue** | **10.89** | **71.8%** |
| stall / unaccounted | 4.27 | 28.2% |

Roofline anchors, measured with `peaks.py` (cycles/instr/SIMD, immune to the clock
drift): WMMA 36.15 → **43.51 TF** ceiling, VALU 1.31, `v_exp` 4.00. Pricing the
loop histogram with those gives an **issue floor of 30.28 TF** — so the kernel is
issue-bound first and latency-bound second, and instruction count is the currency.

**Gap to close:** 25 TF needs 15.16 → **13.77 Mcyc, −9.2%**.

---

## 1. Step 1 — `bcast_group`: phase the gather broadcast to unlock VOPD

**The cost.** All of it is in `dual_gather_finish`. Per gather it converts 8 loaded
dwords into two 8-dword fragments, paying per dword one `permlanex16` plus two
`v_cndmask`:

```python
p = b.permlanex16(e)                  # value held by lane^16 (the other subtile)
fd.append(b.select(lane_lt16, e, p))  # subtile d   in both halves
fd1.append(b.select(lane_lt16, p, e)) # subtile d+1 in both halves
```

At bn64/D128 that runs 16× per K-iteration (`n_kv_sub=4` × 4 subtile pairs),
producing 128 of the loop's 146 `permlanex16` (the other 18 are the softmax
cross-half exchange) and 256 of its 248 `v_cndmask`. Priced at 1.31 cyc: **516 of
3325 cycles per iteration, 15.5%** — the largest addressable block after WMMA.

**Why the shuffles themselves are a floor.** Each of the 8 dwords needs its own
cross-half value and both output fragments consume it, so 8 `permlanex16` per
gather is irreducible. The reducible half is the 16 selects.

**Why VOPD is the lever, and what actually blocks it** (measured with
`shufdbg.py`):

- `v_dual_cndmask_b32 :: v_dual_cndmask_b32` **assembles for gfx1151** — checked
  with `llvm-mc`.
- All **248 selects read one mask** (`vcc_lo`, a single distinct operand), so
  VOPD's shared-VCC requirement holds by construction.
- **120 selects already sit adjacent** to another select.
- Yet the backend forms only **4** `v_dual_cndmask_b32`. `gcn-create-vopd` is in
  the pipeline and forcing `-amdgpu-enable-vopd` either way is byte-identical, so
  the blocker is VOPD's **operand rules (src bank / dst parity), not eligibility**.

The emission order explains it: the only adjacent select pair today is `A_i` with
`B_i`, and those read **the same register pair** `(e_i, p_i)` with operands
swapped. They are the def-use partners the allocator most likes to place together,
so the bank rule is usually violated. Four hits out of 64 is what "only when the
banks happen to line up" looks like.

**The fix is emission order, not new math.** Phase each gather's broadcast into
passes over the dwords instead of interleaving per dword: all `permlanex16` first,
then all `A_i`, then all `B_i`. Neighbouring selects then come from *different*
dwords and touch disjoint register pairs, giving the allocator eight independent
chances to satisfy the bank rule instead of one forced collision, and turning
eight runs of length 2 into two runs of length 8. Interleaving further as
`A_0, B_1, A_2, B_3, …` leaves neighbours sharing no operand at all.

**Knob.** `bcast_group: int = 0` — 0 keeps today's order, N phases the broadcast in
groups of N dwords. One parameter covers the risk: phasing all 8 keeps 8 shuffle
results live before any select consumes them, so the register peak rises;
`bcast_group=4` keeps 4 live and still yields runs of 4.

**Gates, in order.** Do not believe a win until all four pass:

1. `shufdbg.py`: `v_dual_cndmask_b32` climbs 4 → 60+, select count still 248 at
   the DSL level.
2. `isa_deep.py`: `vgpr <= 208` (**199 today; crossing 208 drops 7 waves/SIMD to
   6 and swamps the gain**), `spill=0`, `scratch=0`, and loop instruction count
   *down* by the number of pairs formed.
3. `chunk_sweep.py --verify`: error exactly 1.53e-05 at L=2048 and 7.63e-06 at
   L=16384. The broadcast is a pure rearrangement, so any change means a
   lane-mapping bug.
4. Perf: interleaved 3-rep A/B at L=16K plus `prof_full.py` for the
   clock-independent cycle number.

**Expected.** Full pairing takes the selects from 248 to 128 issue slots: −120
slots, ~157 cyc/iter, **4.7% of the issue floor ≈ −0.51 Mcyc ≈ +3%** → ~23.6 TF.
If the pair count rises but cycles do not fall, the selects were sitting in stall
shadow rather than on the issue path — worth knowing, with 28% of cycles stalled.

---

## 2. Step 2 — re-measure MQ=2 at D=128: the register wall may have moved

**Why now.** The other way to cut broadcast cost is to amortize it. At MQ=2 each
broadcast feeds **two** WMMAs, halving the per-WMMA cost of the shuffles *and* of
the K/Q loads, because the `q_block>1` path already hoists the K and V fragments
across query groups. That is a bigger prize than VOPD: it attacks the 19% "other
VALU" and part of the memory path at once.

It was ruled out at D=128 because MQ=2 spilled hard (vgpr 256, spill 235 → 11.6 TF
at L=4096). But `qk_douter` just deleted the `acc_ilp` accumulators and the tail
reduction from the QK loop, so the peak that wall was measured against has moved.

**Action.** Measure the register footprint *before* any perf run: `isa_deep.py
--mq 2 --block-n 64 --vt 1` and read `vgpr`/`spill`. Note `qk_douter` is
deliberately rejected under `q_block>1` (that path has its own K-hoisting loop and
the knob would silently no-op), so this is the MQ=2 loop on its own merits. If
spill is still in the hundreds, stop there and record it; if it is near zero, run
the full gate ladder from Step 1.

**Also worth a cheap look:** bn32 at MQ=2 (n_kv_sub=2 halves the live V fragments)
and the D=64 shapes, where MQ=2 is already the known win.

---

## 3. Ruled out — do not re-litigate without new evidence

Each of these was measured on this board; the kernel docstrings carry the numbers.

| lever | verdict |
|---|---|
| `v_permlane16_swap_b32` (2-op broadcast) | **not on gfx1151** — `llvm-mc` rejects it; gfx950/gfx125x only |
| exec-masked tied-`vdst` `permlanex16` | wash: still 3 ops (mov + 2 permlane) |
| forcing `-amdgpu-enable-vopd` | no-op, byte-identical code |
| `dual_gather=0` (no broadcast at all) | 34% slower — doubles MUBUF 8.4M→16.8M on a request-bound TA path |
| `waves_per_eu=8` (for the 8th wave) | reaches 190 VGPR but 24 spills / 100 B scratch, unstable 9.2–22.8 TF |
| `o_f16` (for the 8th wave) | 183 VGPR spill-free, but **2.7–7.3 TF at L=16384** (fine at L2048); unexplained cliff |
| `q_hoist` | 157 spills, 440 B scratch |
| `q_lds` / `kv_lds` | 2.5× / 0.29–0.38× — L1 hits beat LDS round-trips here |
| `v_prefetch` depth≥3 | L2 misses 0.3M → 12.4M; MLP cannot pay for lost residency |
| `iglp_opt` any level | −3.5%, conflicts with the pingpong `s_setprio` tuning |
| `v_kblock` (fewer cache lines) | MemUnitBusy 96.9% → 85% but cycles *up*: line count is not the constraint |

---

## 4. Measurement protocol (the discipline that made the last result trustworthy)

- **Interleave A/B reps.** The board decays 2405 MHz / 87 W → ~1600 MHz / 43 W
  within seconds. A sequential A/A/B/B read the `qk_douter` result **backwards**;
  three interleaved reps read it correctly. Always alternate and report all reps.
- **Prefer GRBM_GUI_ACTIVE cycles** (`prof_full.py`) for keep/revert decisions —
  clock-independent. Use wall-clock TF only as the interleaved confirmation.
- **Check the ISA before believing a mechanism.** `qk_douter` was built to delete
  24 redundant Q loads per iteration; the load count never moved, and the real
  gain was duplicate-address elimination visible only in TA_TA_BUSY. Confirm the
  intended instruction actually changed before attributing a win to it.
- **`MemUnitBusy` is misleading here** — it sits at ~96% in every config,
  including ones that are slower. Trust TA_TA_BUSY and request counts.

**Tooling:** `peaks.py` (issue-cost anchors), `isa_deep.py` (loop histogram priced
by those anchors, VGPR-granule occupancy, in-loop load mix), `prof_full.py`
(counter cycle budget), `shufdbg.py` (select masks, adjacency runs, formed VOPD
pairs), `loaddbg.py` (loads grouped by address operand), `chunk_sweep.py`
(correctness + wall clock).

---

## 5. Honest ceiling

WMMA issue alone is 7.58 Mcyc = 50% of the 15.16 Mcyc budget, and the measured
WMMA ceiling is 43.51 TF. Step 1 in full is worth ~+3% (→ ~23.6 TF), which is
**not enough for 25 TF on its own** — it needs Step 2's amortization, or a
reduction in the 28% stall, to finish. If MQ=2 does not fit at D=128 even after
`qk_douter`, then 25 TF at L=16K likely needs the KV-stationary persistent kernel
rather than more tuning of this one.

---
---

# Appendix (superseded) — Closing the CK-Tile FMHA gap on gfx1151

> **Superseded.** This plan is from the ~9–11 TF era, before the swapqk kernel and
> the transposed-V layout; its levers landed and its "do not expect >~17 TF"
> ceiling was passed (22.8 TF measured). Kept for the CK provenance and the ASM
> comparison method, not as live guidance.

**Scope.** Why CK Tile's forward FMHA runs ~17 TF on gfx1151 (Strix Halo, RDNA3.5,
WMMA `f32_16x16x16_f16`, wave32) while the rocke DSL kernels top out at ~9–11 TF,
and a concrete, evidence-based plan to replay CK's approach on rocke.

**Method.** Everything below is measured on a live gfx1151 board (node
`ctr-halo-b47-02`) or read from source with `file:line` provenance. CK numbers are
from the actual dispatched instance
`fmha_fwd_d128_fp16_batch_b64x64x32x128x32x128_r4x1x1_..._qr_vr_..._gfx115`,
compiled to gfx1151 device assembly (`hipcc --cuda-device-only -S`) and its
compiler resource comments. rocke numbers are from `builders.gfx1151.attention.tune`
(msgpack resource decode) for the same B4 Sq512 Sk512 D128 H8 shape.

---

## 0. TL;DR — the premise is (mostly) a no-op, the gap is structural

The task was "investigate CK's ASM + scheduler hints and replay them as-is on rocke."
The evidence says:

1. **CK uses almost no explicit scheduler hints on this exact path.** On
   gfx1151 / `qr` / fp16 / hdim128 / wave32: `sched_group_barrier` is compiled
   **OFF** (guarded `warp_size==64 && hdim==256`), `sched_barrier` is **OFF**
   (guarded `ELEMENTWISE_BIAS`), and there is **no `s_setprio`**. The dense
   `s_delay_alu` stream in CK's ISA is **emitted by the RDNA3 LLVM backend
   automatically**, not by CK source hints. So "replay the hints as-is" ≈ replay
   *nothing* — there is no secret hint sauce on this path.
2. **rocke can already emit everything CK can** (and more). `sched_barrier`,
   `sched_group_barrier`, `s_setprio`, `s_waitcnt`, `iglp_opt`, plus occupancy via
   `amdgpu-waves-per-eu` / `amdgpu-flat-work-group-size` are all first-class
   IRBuilder ops today. "rocke can do everything CK can" is **TRUE** at the
   emission layer — parity is not blocked by capability.
3. **The ~40% gap is structural + register-allocation, not scheduling:**
   - **VGPR budget.** CK uses **256 VGPR/wave, spill ≈ 1**. rocke is pinned at
     **192 VGPR, spill 29** (single-wave) and **spill 486** the moment it adds
     density (bm2). rocke is leaving the top 64 VGPR of the RDNA3 wave32 file
     unused *and spilling instead*.
   - **4-warp accumulator fan-out.** CK spreads one 64×128 output tile across
     **4 warps × 32 lanes = 128 threads**, so each lane owns only ~64 f32 of
     `o_acc` (+32 of `s_acc`). A single wave32 covering the same tile needs
     **256 f32 of accumulator alone → guaranteed spill**. This fan-out — not any
     hint — is why CK stays resident at 64 WMMA/body.

**Bottom line.** To close the gap we replay CK's *structure* (4-warp cooperative
tile + let the compiler use 256 VGPR by relaxing the occupancy target), not its
(absent) scheduler hints.

---

## 1. Side-by-side ASM evidence (D128, B4 Sq512 Sk512 H8, fp16, non-causal)

| Metric | **rocke** single-wave (9.3 TF) | **rocke** bm2 density (3.6 TF) | **CK Tile** `qr` 4-warp (17.5 TF) |
|---|---:|---:|---:|
| VGPR / wave | **192 (pinned)** | **192 (pinned)** | **256** |
| VGPR spill | **29** | **486** 💥 | **≈1** (ScratchSize 1040) |
| Occupancy (waves/SIMD) | high (over-constrained) | high | **5** |
| WMMA / body | 16 | 32 | **64** |
| Total instr | 1738 | 3122 | ~ (codeLen 10212 B) |
| LDS (group segment) | ~small (P tile) | ~small | **9216 B** (shared K+V+P) |
| Warps / block | 1 (32 thr) | 1 (32 thr) | **4 (128 thr)** |
| `s_delay_alu` (backend) | few | few | **100+** (auto) |
| `s_setprio` / `sched_group_barrier` | none | none | **none** |
| launch bound | `flat-work-group-size 64,32` | same | `__launch_bounds__(128, 2)` |

**Reading it.** CK is *not* faster because of hints or higher occupancy — it runs
at occupancy 5 with a fat 256-VGPR, essentially spill-free footprint and 64 WMMA
of useful work per body. rocke does 16 WMMA, is jammed under a 192-VGPR ceiling,
and *spills* — and every attempt to add density (bm2) explodes to 486 spills
because a single wave can't hold the bigger accumulator.

CK source facts behind the table:
- 4-warp tile `F_occupancy = -1` → pipeline picks `kBlockPerCu = 2` →
  `__launch_bounds__(128, 2)` (`codegen/ops/fmha_fwd.py:1213`,
  `block_fmha_pipeline_qr_ks_vs.hpp:120-139`, `host/kernel_launch.hpp:69`).
  (The `occ=6` value belongs to the *8-warp* `b128x64` tile, used only for
  `max_seqlen_q >= 2048` — not the instance benchmarked here.)
- Per-warp iteration counts (`block_gemm_areg_bsmem_creg_v1.hpp:53-55`): gemm0
  MIter=1/NIter=4/KIter=2, gemm1 MIter=1/NIter=8/KIter=2. C fragment is 8 f32
  per 16×16 tile (`warp_gemm_attribute_wmma_impl_base_traits.hpp:89-94,111`) →
  gemm1 `o_acc` = 8 tiles × 8 f32 = **64 VGPR/lane**, gemm0 `s_acc` = 32.

---

## 2. The scheduler-hint audit (what "replay the hints" actually means)

On the gfx1151 `qr` fp16 hdim128 4-warp path:

| CK mechanism | Active here? | Evidence |
|---|---|---|
| `sched_group_barrier` DS/MFMA interleave | **NO** — guarded `get_warp_size()==64 && kQKHeaddim==256` | `block_fmha_pipeline_qr_ks_vs.hpp:492-503` |
| `sched_barrier(0)` | **NO** — guarded `BiasEnum==ELEMENTWISE_BIAS` | `qr_ks_vs.hpp:564,570` |
| `s_setprio` | **NO** — absent from qr path entirely | (only in some gemm/gemm_quant pipelines) |
| `s_barrier` + `s_waitcnt lgkmcnt(0)` | **YES** — generic `block_sync_lds()`, ~12×/K-loop | `core/arch/arch.hpp:1382-1392` |
| `PermuteWarpGemmCToA` (C→A relayout) | **YES** — gfx11-only *overhead*, not a hint | `qr_ks_vs.hpp:1093`, `warp_wmma_gemm_gfx11_utils.hpp:15-43` |
| `s_delay_alu` dependency latency | **YES but backend-auto** — not a CK source hint | seen only in compiled ISA |
| explicit `amdgpu-waves-per-eu` / VGPR cap | **NO** — CK sets none; occupancy only via `__launch_bounds__` | (grep of ck_tile tree: none) |

**Conclusion:** there are no CK-authored scheduler hints to port on this path. The
useful `s_delay_alu` interleave is produced by the shared RDNA3 backend and will
appear in rocke's output too *given the same instruction-level parallelism* — i.e.
once rocke has the 4-warp density and register headroom to expose independent
WMMAs. The scheduling win is a *consequence* of the structural fix, not a separate
lever.

---

## 3. rocke capability confirmation (can it replay CK?)

Audited `rocke/core/ir.py` (IRBuilder) and `rocke/core/lower_llvm.py`:

| Capability | rocke status | Where |
|---|---|---|
| `sched_barrier(mask)` | **exposed** | `ir.py:2828` → `lower_llvm.py:3181` → `@llvm.amdgcn.sched.barrier` |
| `sched_group_barrier(mask,count,group)` | **exposed** | `ir.py:2837` → `lower_llvm.py:3186` |
| `s_setprio(level)` | **exposed** | `ir.py:2857` → `lower_llvm.py:3196` |
| `s_waitcnt` / `sync` / `sync_lds_only` | **exposed** | `ir.py:2672-2755` → `lower_llvm.py:3120-3175` |
| `iglp_opt(level)` | **exposed** (CK barely uses) | `ir.py:2816` → `lower_llvm.py:3176` |
| occupancy: `amdgpu-flat-work-group-size` | **always emitted** from `max_workgroup_size` | `lower_llvm.py:4405-4409` |
| occupancy: `amdgpu-waves-per-eu` | **emitted when** `attrs["waves_per_eu"]` set | `lower_llvm.py:4415-4422` (used by gfx1151 `deep_fused_conv_pool.py`) |
| hard VGPR cap `amdgpu-num-vgpr` | **NOT emitted** (CK also lacks it) | would add at `lower_llvm.py:4423` |

**Verdict:** rocke is a *superset* of CK's scheduling/occupancy surface. The only
missing knob (`amdgpu-num-vgpr`) is a trivial one-line add and CK doesn't use it
either. The gap is authoring/tuning, not expressiveness.

> Dual-engine note: the scheduling ops already exist in both engines. New emission
> must be mirrored in the C++ engine and re-blessed against the byte-identity gate
> (`rocke/platform/AGENTS.md`). The occupancy `attrs` and a `waves_per_eu` spec knob
> in a standalone `builders/gfx1151/attention/` vehicle are *not* byte-identity
> gated, so the experiments below stay in those vehicles.

---

## 4. Replay plan — prioritized, each with a measurable checkpoint

Target: the CK numbers on the same board (D128 S512 ~17.5, S1024 ~16.6, D64 S512
~16.5, D128 causal ~12.5), correctness-gated (`max_abs ≤ 2e-2`) on every run.
Baseline: rocke best ~10–11 TF (single-wave / multiwave-gather).

### Lever 1 — Unpin VGPR: relax the occupancy target so the backend uses 256 VGPR (highest value, lowest effort)
**Evidence:** rocke pinned at 192 VGPR + spill 29/486; CK at 256 VGPR + spill ~1
via `__launch_bounds__(128,2)`.
**Action:** set an explicit occupancy hint on the attention vehicles so the AMDGPU
backend stops maximizing waves and spilling. Add `waves_per_eu` to the
`fmha_singlewave` / `fmha_multiwave` spec and pass it through to
`b.kernel.attrs["waves_per_eu"]` (lowering already supports it,
`lower_llvm.py:4415`). Sweep `waves_per_eu ∈ {2,3,4}` for the 4-warp block
(equivalent to CK's `launch_bounds(128,2)`), and for the single-wave block too.
**Expected:** single-wave D128 spill 29 → ~0 (small TF bump); more importantly it
*unblocks* Lever 2 (density stops spilling).
**Checkpoint:** re-run `tune`/`mw_tune`, confirm `vgpr` rises toward 256 and
`spill` drops; record TF.

### Lever 2 — 4-warp cooperative single-output-tile with distributed accumulator (the structural core)
**Evidence:** CK fans one 64×128 tile across 4 warps (MWarp=4, NWarp=1; MIter=1,
gemm0 NIter=4 / gemm1 NIter=8 → 64 WMMA) so each lane holds only ~96 VGPR of
accumulator. rocke's bm2 keeps the whole enlarged accumulator in *one* wave → 486
spills.
**Action:** make `fmha_multiwave` the CK shape precisely — 4 waves cover one
`64 (M) × 128 (N=hdim_v)` output tile, `MWarp=4` splitting the 64 query rows
(16 rows/warp, `MIter=1`), each warp spanning the full N with `NIter` register
blocking; K and V staged once per CTA in shared LDS (9 KB, matches CK's group
segment). Combine with Lever 1 so the distributed 64-VGPR `o_acc` + fragments fit
256 VGPR spill-free.
**Expected:** WMMA/body 16 → ~64 at spill ≈ 0; the density is what raises the
WMMA:overhead ratio toward CK's.
**Checkpoint:** `mw_tune --waves 4 --kv-source lds`, verify `wmma≈64`, `spill≈0`,
`vgpr≈256`; target ≥14 TF non-causal.

### Lever 3 — Re-run the LDS-vs-gather A/B *under the new register budget*
**Evidence:** our earlier finding (gather beats LDS staging) was measured at the
*spilling* 192-VGPR / low-density regime. CK wins *with* LDS staging at 256 VGPR /
4-warp. The calculus may invert once Levers 1–2 remove the spill and the barrier
has 4 waves to hide behind.
**Action:** with Levers 1–2 in place, A/B `kv_source ∈ {lds, gather}` again at
`waves=4`. Keep whichever verifies faster per shape.
**Checkpoint:** direct A/B in one thermal window; record both.

### Lever 4 — Only if the backend schedule is measurably poor: add explicit interleave hints
**Evidence:** CK emits *no* `sched_group_barrier`/`s_setprio` here; `s_delay_alu`
is backend-auto. So this is a *fallback*, not expected to be needed.
**Action:** if the compiled rocke ISA (post Levers 1–2) shows poor WMMA/VALU/DS
overlap vs CK's `s_delay_alu` density, insert `b.sched_group_barrier(...)` around
the QK/softmax/PV groups (rocke supports it, `ir.py:2837`) — mirroring the
wave64/hdim256 pattern CK keeps for CDNA. Measure; revert if noise.
**Checkpoint:** disassemble rocke kernel (`--cuda-device-only -S`), compare
`s_delay_alu`/WMMA interleave and TF to CK.

### Lever 5 — Match the fixed per-iteration costs
- Causal early-exit: **already ported** (K-loop clamp) — keep.
- `exp2` fast-softmax: already present.
- `PermuteWarpGemmCToA` C→A relayout: CK pays this on gfx11; rocke's P-transpose
  via LDS is the equivalent. Confirm rocke isn't paying *more* than CK's permute
  (compare ds_read/ds_write counts); if so, port the `permlanex16`+`perm` register
  relayout (rocke has both builders).

### Sequencing
Lever 1 → Lever 2 (together they are the ~2× structural fix) → Lever 3 (pick
staging mode) → Levers 4/5 (polish). Gate correctness on every step; make all
keep/revert decisions A/B back-to-back in one thermal window (±0.6 TF drift on
this box).

---

## 5. Honest ceiling & risks

- **Issue-bound wall persists.** Even CK sits at ~29% of the 59 TF peak: WMMA is
  ~1% of issued instructions. Levers 1–3 target the rocke↔CK gap (10→~17), *not*
  the CK↔peak gap. Do not expect >~17 TF from this kernel family on gfx1151; the
  rest needs gfx12 (larger-K WMMA, `ds_read_tr`, packed-f16 C).
- **Risk that Lever 1 alone under-delivers.** Unpinning VGPR helps the single-wave
  spill but the single wave still can't reach 64-WMMA density — Lever 2 is the
  load-bearing one. They must land together.
- **Byte-identity.** Keep all changes in the `builders/gfx1151/attention/`
  vehicles; do not touch the shared `helpers/mfma_attention.py` or `core/` lowering
  (except the optional trivial `amdgpu-num-vgpr` add, which must be mirrored + gated).
- **`flat-work-group-size "64,32"` smell.** For single-wave (`max_workgroup_size=32`)
  the lowering emits `"amdgpu-flat-work-group-size"="64,32"` (lo>hi). Worth checking
  whether that malformed range perturbs the single-wave allocation; the 4-warp path
  (`"64,128"`) is well-formed.

---

## Appendix — provenance
- CK device ASM: `hipcc --cuda-device-only -S` of the dispatched D128 `qr` instance
  on `ctr-halo-b47-02`; compiler comments `NumVgprs: 256 / ScratchSize: 1040 /
  Occupancy: 5`, `.amdhsa_group_segment_fixed_size 9216`, 64× `v_wmma_f32_16x16x16_f16`,
  100+ `s_delay_alu`, 12 `s_barrier`, no `s_setprio`/`sched_group_barrier`.
- rocke resources: `builders.gfx1151.attention.tune` D128 → `vgpr=192 spill=29 wmma=16`
  (bm1), `vgpr=192 spill=486 wmma=32` (bm2).
- CK source refs: `block_fmha_pipeline_qr_ks_vs.hpp:120-139,492-503,564-570,1093`;
  `codegen/ops/fmha_fwd.py:153-156,611-629,1213-1214,1246-1255`;
  `host/kernel_launch.hpp:68-69`; `core/arch/arch.hpp:1382-1392`;
  `block_gemm_areg_bsmem_creg_v1.hpp:53-55`;
  `warp_gemm_attribute_wmma_impl_base_traits.hpp:89-94,111`.
- rocke capability refs: `core/ir.py:2672-2861`; `core/lower_llvm.py:536-543,3120-3199,4405-4425`.
