# gfx1151 Deep-Fusion Optimization Case Study

**Kernel:** single-kernel fusion of `encoder_0` —
`concat → conv0 3×3 pad1 → quant/ReLU → quant → conv1 1×1 → quant/ReLU →
2×2/s2 maxpool → packed-int4 out`.
**Target shape:** `N=1, H=2160, W=3840, C=8, K0=32, K1=24` → pooled
`(1080,1920,24)`, ~50.9 GFLOP (51.0 GOP int).
**HW:** Strix Halo iGPU, RDNA3.5 (gfx1151 / built `gfx11-generic`), wave32,
virtualized **8-CU / 4-WGP** slice, ~64 KB LDS/CU, no clock control.
**Status (2026-06-07):** native int8/int4 pipeline, bit-exact, best ≈ **11.68 ms**
(board2) = the default lever stack **plus `--conv1-int8 --pk-maxpool`** (+3.9 % over
the prior 12.14 ms default). The conv1 GEMM now runs **iu8 fragments over int4-range
codes** (conv0 stays iu8), which deletes the iu4 nibble-pack bitwork and is bit-exact.
Persistent grid-stride variant is **not competitive with the i8+pk winner — board-
sensitive, lower priority** (§3).

> **Reading the numbers.** The board auto-clocks ±25–30% and the partition slice
> drifts between sessions, so **absolute ms / TFLOP are only comparable within
> one interleaved A/B session.** Every "+x%" below is from same-session
> interleaved A/B (`compare_configs.py` / `compare_prebuilt.py`). Do **not**
> compare a 6.7 ms fp16 number against a 14 ms native-int number — they are
> different sessions / regimes. The honest story is the *relative* win of each
> lever and the evolution of the bottleneck model, not a single monotone curve.

---

## 1. The one fact that governs everything

This is a **tiny-GEMM, latency/overhead-bound** workload, not bandwidth-bound and
not matrix-throughput-bound:

- Arithmetic intensity ≈ **560 OP/byte** (~91 MB HBM for 51 GOP). Memory never binds.
- conv0 K=72 (5 K-atoms over a 16-wide WMMA), conv1 K=32 (**2 K-atoms**). K is the
  channel count — it **cannot be deepened**. The MMA/LDS pipeline can't be amortized.
- Ablation (2026-06-06) pins **89.7 % of wall-clock in the conv1 iu4 GEMM
  pipeline** (operand delivery + C0→C1 handoff + WMMA), ~12.77 ms of 14.2 ms.
  Everything else — conv0 GEMM, both requant epilogues, maxpool, output store —
  is only ~1.47 ms (10.3 %).

Consequence: levers that **hide/remove latency and redundant LDS traffic win**;
levers that **add occupancy or merely cut VALU regress or barely move**. This is
the lens for every result below.

The bottleneck is still the **short-K conv1 operand pipeline, not HBM bandwidth.**
The updated lesson is narrower than "cut VALU": static VALU counts alone are **not**
predictive, but removing VALU that sits **directly on conv1 operand formation** can
convert. The `conv1_int8 + pk_maxpool` winner keeps the same straight-line structure
and the same multiply count, but swaps the conv1 iu4 path for **iu8 fragments**,
materially cutting the fragment-pack bitwork (§2.G).

---

## 2. What worked (validated wins)

### A. fp16-emulation era (the 3.8× foundation)
Each verified bit-exact, same-session A/B.

| Lever | Win | Why it worked |
|---|---|---|
| **Warp count** (bs64→bs256) | big (largest single lever) | one WG resident/CU at ~43 KB LDS ⇒ more warps = free latency hiding, not an occupancy trade |
| **direct-conv0 footprint cache** (vs im2col) | **+38 %** (→6.7–6.9 ms) | im2col re-stages the 3×3 R·S footprint to LDS redundantly; direct loads each input once. Anti-redundant-staging is the core thesis |
| conv0-A vectorize | +20.3 % | wide LDS/VMEM fragment loads |
| maxpool vectorize | +9.8 % | coalesced gather |
| pool-tile geometry | +4.5 % | wide-W tiles coalesce conv writes + pool reads |
| barrier merge | +2.9 % | one fewer full-WG sync |
| early-W1 load | small + | hoist W1 HBM load to overlap conv0 |
| sched hints `compv3/compv4` | +2 % | DS_READ/MFMA interleave; small because K-atom counts are tiny |

Net fp16: 35.78 ms → 9.31 ms (one-lever campaign) → 6.7–6.9 ms (direct-conv0).

### B. Native int8/int4 pipeline (the precision switch)
Wired `wmma_i32_16x16x16_iu8` (conv0) + `iu4` (conv1) end-to-end: arch specs,
fragment maps, LLVM intrinsics, backend (i1 **is-signed** flag = 1, clamp = 0),
packed fragment helpers, integer requant/ReLU/maxpool. **Bit-exact**
(`max_abs_diff=0`, 0/49,766,400) and now the default path. Deletes all fp16
dequant/`rint` overhead and halves/quarters LDS staging density (i8/i4 vs f16).

The **conv1_int8 sub-path** — running the conv1 GEMM as the **iu8 atom over
int4-range codes** (conv0 codes stay int4-range so dot products are bit-identical to
the iu4 path) — is now **validated and bit-exact**, promoted from candidate to part
of the shipped winner (§2.G).

### C. Pool-tile **shape** (not size)
**2×32 tile @ tile_m=256 = 16.14 ms vs 8×8 @ tile_m=256 = 18.51 ms (~13 %), zero
extra code.** Same `tile_m`, different shape ⇒ isolates *access pattern* as the
cause. Wide-W tiles coalesce conv output writes and the maxpool read footprint
along the fast axis; tall-square tiles fragment it. Now default.

### D. fused_c0a1 (Route B — in-register C0→C1 handoff)  **+10 %**
Reorient conv0 (W0 as WMMA-A) so its accumulator lands lane=m / slot=k0, then
transpose k0↔m to the conv1 A-fragment (iu4 old path, **iu8 winner path**)
**in-register** via `permlanex16` +
`v_perm_b32` (the gfx11 FMHA C→A pattern). **Deletes `c0_smem` and a full-WG
barrier.** Bit-exact by construction. fused t768 beats non-fused t512 by ~10 %
same-session. Requires `--native-int --direct`, K0%16==0; mutually exclusive
with `repack_c0`/`packed_c0_handoff`.

### E. conv1 ILP: `prefetch_k` + `sched_fuse` — **MANDATORY COMBINATION**  **+6–7 %**
The genuine remaining lever from the roofline. In the conv1 A-register-resident
GEMM (`_wmma_gemm_conv1_i4_from_regs`, or `_i8_from_regs` on the iu8 winner path):
- `conv1_prefetch_k`: hoist **all** k-step W1 B-fragment `ds_read`s before any MMA.
- `conv1_sched_fuse`: replace per-k-step `sched_group_barrier` with one combined
  group after the loop.

ISA confirms the combo keeps all 4 W1 B-fragment register pairs live and
**interleaves k0/k1 MMAs** instead of issuing all 12 k=0 then k=1. Robust
**+6.9 %/+5.6 %/+7.0 %** across runs, bit-exact, **no LDS/VGPR/occupancy cost**
(LDS 35968 B unchanged, VGPR 138→139, 84 v_wmma all variants).

> **Why the combination is necessary** (validated): `sched_fuse` *alone* swung
> +7.0 %/−0.4 %/+6.8 % run-to-run — unstable. `prefetch_k` makes the fragments
> available; `sched_fuse` lets the scheduler actually interleave them. Either
> alone leaves the scheduler without enough freedom *or* without the data. This
> matched the user's "a combination necessary for the ILP to work" intuition.

**Best invocation.** The structural levers below are default ON in the verify CLI;
`--conv1-int8 --pk-maxpool` are the new winner additions and are **NOT yet default-on**
(flipping them to default is a pending follow-up), so they must be passed explicitly:
```
--native-int --direct --fused-c0a1 --warp-m 16 --warp-n 1 --sched compv4 \
--pool-tile-h 2 --pool-tile-w 64 --conv1-prefetch-k --conv1-sched-fuse \
--conv1-int8 --pk-maxpool
```
Escape hatches: `--no-native-int` (fp16 path), `--no-fused-c0a1`,
`--no-conv1-prefetch-k`, `--no-conv1-sched-fuse`, `--warp-m N`. `--conv1-int8` and
`--pk-maxpool` are plain default-off switches (no `--no-` form) — simply omit them
to fall back to the iu4 conv1 GEMM and the per-channel cmp/select maxpool.

### F. Warp geometry: max waves/WG (warp 16×1)  **+23.8 %** (biggest native-int lever)
`warp_m` = #waves along M (= block_size/32 at warp_n=1). The conv1 iu4 GEMM
(~90 % of wall) hides its exposed LDS-read latency with **more resident waves**
— one WG/CU resident, so warps are free latency-hiding (the fp16-era lesson,
re-confirmed in the native-int regime). Rotated 4-round bit-exact board A/B
(2026-06-06, full shape, **with the full lever stack on**, pt2x64/tile_m=512):

| warp | block | median | vs 4×1 |
|---|---|---|---|
| 4×1 | 128 thr | 15.131 ms | +0.0 % |
| 8×1 | 256 thr | 13.661 ms | +10.8 % |
| **16×1** | 512 thr | **12.223 ms** | **+23.8 %** |

> **This reverses the earlier committed result** (`3da274e9a1`: "warp 4×1 +10 %,
> 8×1 +7 %"). That batch was measured **without** `fused_c0a1`/`prefetch_k`/
> `sched_fuse`. With the fused + ILP levers on, the optimum flips to *maximum*
> waves — more independent latency to hide ⇒ more waves to hide it. Lesson:
> **re-tune geometry after changing the pipeline; geometry optima are not
> portable across lever sets.** 16×1 is now the default.

### G. conv1_int8 + pk_maxpool (the i8+pk winner)  **+3.9 %** (new best, 11.68 ms)
Stacked on the §2.E default, this pair is the current best, **bit-exact**, confirmed
by board1 **and** board2 rotated interleaved A/B (2026-06-07):

| Lever | Alone vs default | On top of the other | Bit-exact |
|---|---|---|---|
| **pk_maxpool** | +3.5 % | (drives the win) | yes |
| **conv1_int8** | +1.0 % | +0.4 % incremental | yes |
| **both** | — | **+3.9 % (12.141 → 11.681 ms, board2)** | yes |

- **conv1_int8:** run the conv1 GEMM as the **iu8 atom over int4-range codes**
  (`_wmma_gemm_conv1_i8_from_regs`): A-fragments = unsqueezed `<4 x i32>` byte codes,
  W1 = one `ds_read_b128` per n-atom; conv0 codes stay int4-range so dot products are
  **bit-identical** to the iu4 path. It **deletes the nibble-pack bitwork** the iu4
  path needs — this is targeted operand-formation VALU, so it converts (§1, §3).
- **pk_maxpool:** widen conv1 codes to i16 and reduce 2×2 max with `vector.smax`
  (`llvm.smax.v<N>i16`) instead of cmp/select.

> **Why pk_maxpool now reads +3.5 % (it was logged +1.3 % at commit `acb538cdd5`).**
> The implementation is **unchanged** since that commit (git log confirms — no code
> diff). The delta is a **session/stack difference**, not an implementation change:
> per the "absolutes only comparable within one session" discipline (header note), the
> +1.3 % and +3.5 % are different sessions / lever stacks, not a real regression-then-
> gain. It is still **not truly packed** (`v_pk_max_i16`=0 — would need 2×i16
> co-located end-to-end).

---

## 3. What did NOT work (non-levers, with the reason)

| Lever | Result | Why it failed |
|---|---|---|
| **waves_per_eu=2** | fp16 −5.9 %, native **24.26 ms** | Occupancy is **VGPR-bound, not LDS-bound**. Forcing 2 WG/CU caps VGPRs; lost per-wave registers cost more latency-hiding than the 2nd WG buys. The textbook "free 2nd WG from spare LDS" doesn't apply. |
| **mask_maxpool** | fp16 −11.5 % | Pool tile = exactly one wave (32 pooled px), so the `scf_if` was already warp-uniform. Masking instead wakes 7 warps to compute discarded work — pure loss. |
| **butterfly_conv01** (fp16) | rejected pre-codegen | conv0 N→conv1 K is a true cross-lane 16×16 transpose; the only wave32 vehicle (`ds_bpermute`) is itself an LDS-unit op and needs ~8 bpermutes/slot (~64–128 LDS ops/warp) to replace ~4 ds_reads + one uniform barrier. **The LDS round-trip is the cheap path.** |
| **repack_c0** (native) | 18.11 → 20.00 ms (−10 %) | Removed ~12.5 % static VALU (v_and 84→4, no ds_bpermute) but that VALU lived *inside* the conv1 k-loop already overlapped with WMMA/LDS; the repack added a mandatory full-WG barrier + extra C0 LDS round-trip on the critical path. |
| **"make conv1-A a free bitcast like W1"** | dead | C0 is a WMMA output with HW-fixed lane layout (col = k0 = lane%16); the k0↔m transpose is mandatory. W1 is free only because it's staged from HBM (pre-packable). |
| **"wider N / fill the WMMA N tile"** | ~0 | The ~1000-VALU nibble-pack cluster is indexed by M,K — not N. Padded N adds only B-loads (free bitcasts) + WMMA, both in the cheap 1.8 % slice. |
| **deepen K** | impossible | conv1 K = channel count (32, padded from 24). 2-step loop is fixed by the problem. |
| **cut static VALU *indiscriminately*** | sub-linear, capped | Ablation: nibble-pack = 3.9 % of wall; generic VALU micro-opts (clamp, off-path bitwork) cap in low single digits because ~86 % of wall is the conv1 WMMA/LDS-load region. Conversion ratios: pk_maxpool 0.37×, ablate_pack 0.17×. **But the lesson is targeted, not absolute:** VALU sitting **directly on conv1 operand formation DOES convert** — conv1_int8 removed the nibble-pack bitwork and bought +1.0 % alone (§2.G). Cut operand-path VALU; don't bother with off-path VALU. |
| sched `mem` policy | −worse (20.7 ms) | reordering can't help once latency, not issue, is the wall; compv3/v4/intrawave all saturate ~18.2 ms at 8×8. |
| **persistent kernel** (grid-stride, W0/W1 hoisted) | **board-sensitive; loses to i8+pk** | On board2 it regressed −2.0 % to −3.5 % (ctas 8/16/24/32/64) vs the old default; on board1 a persistent variant can beat the *old* baseline but still **loses to the i8+pk winner**. Trades away the HW's *free* inter-CTA overlap of the 16,200-deep CTA queue; the W0/W1 hoist it adds was already hidden by `prefetch_k`. Occupancy sweep flat-to-worse ⇒ **occupancy is not the recovery knob.** Lower priority — flag kept off as documented infra. |

> **Persistent kernel deep-dive (2026-06-07, bit-exact, board2 rotated A/B).**
> Hypothesis: the 16,200-CTA grid (one tile/CTA) re-stages tile-invariant W0/W1
> global→LDS every tile; a persistent kernel (launch `persistent_ctas`, grid-stride
> the tile strip) hoists that staging above the loop — read W once per CTA. Built
> behind a `--persistent` flag; winner codegen stays **byte-identical** (flag off).
>
> *The hoist worked exactly as designed.* IR/ISA confirm per-tile global loads drop
> 4→2 (W0+W1 leave the body for the preheader: `persist.s` `global_load_b64`@28 = W0,
> `global_load_b128`@43 = W1, then `s_barrier`; loop body keeps only the 2 footprint
> loads). Per-tile WMMA (20 iu8 + 8 iu4) and `ds_read` counts unchanged, **zero spills**.
>
> *On board2 it regressed — every grid size, monotonically worse as ctas grew:*
> p8 −2.0 %, p16 −2.9 %, p24 −2.8 %, p32 −2.8 %, p64 −3.5 % vs the 12.13 ms winner.
> (On board1 a persistent variant can beat the *old* baseline, but still loses to the
> i8+pk winner — the result is board-sensitive, not a clean regression everywhere.)
> This is the §1 thesis biting back: on the latency-bound 8-CU slice the deep CTA
> queue *is* the latency-hiding mechanism. The persistent loop welds the two long
> events — footprint `global_load` and `global_store Y` — across tiles via the
> back-edge `s_barrier` (tile N+1's load can't issue until tile N's epilogue passes
> the a0/c1_smem-reuse barrier), so the load latency the queue hides for free is
> exposed serially. The W-staging it removes was cheap and prefetch-hidden; the
> overlap it loses is not. More resident CTAs on the fixed slice just add contention.
>
> *What software pipelining would buy — and its ceiling.* The only untried recovery
> lever: double-buffer `a0_smem` and issue tile N+1's footprint loads *during* tile
> N's 28 WMMAs (independent addr + LDS slot), removing the back-edge barrier from the
> load's critical path. This **re-creates by hand the inter-CTA overlap the winner
> gets for free.** Ceiling = the exposed footprint-load/store latency = exactly the
> 2–3.5 % lost — so it can reach **~parity, not beat** the winner (which also has the
> overlap *and* zero loop overhead). It also costs a 2nd LDS buffer + carry-VGPRs
> (currently zero spills) on an occupancy-limited slice — the very pressure that
> hides latency. Verdict: a break-even play with downside, **board-sensitive and not
> competitive with the i8+pk winner.** Lower priority; flag kept off as documented
> infra. Revisit only with a software-pipelined tile loop, and even then expect
> ~parity at best on any GPU where 16,200 tiles keep the baseline queue deep.

**The bottleneck model itself was twice corrected** — a key meta-lesson:
1. "memory/latency-bound" (early) →
2. "static-VALU-bound, cut VALU for <10 ms" (static ISA: 82.7 % VALU) →
3. **overturned by measurement**: more waves slower + VALU cuts don't convert →
4. **ablation-proven: conv1 iu4 GEMM pipeline = 89.7 % of wall** (operand
   delivery + mandatory C0→C1 transpose + WMMA latency), NOT the epilogue VALU.

Static instruction counts lie about dynamic cycles when the hot region is
latency-overlapped. **Always confirm a static hypothesis with an interleaved
board A/B before chasing it.**

---

## 4. Lever reference — when each is useful in future

- **direct footprint cache, warp-count, pool-tile shape, vectorized A/pool** —
  the durable structural wins. Apply first to any new fused conv tile on this
  arch; they attack redundant staging and coalescing, which are precision- and
  shape-independent.
- **fused_c0a1 (in-register handoff)** — use whenever a producer WMMA feeds a
  consumer WMMA *and* you can reorient the producer so its accumulator lane
  layout matches the consumer A-fragment with a cheap `permlanex16`/`v_perm_b32`
  transpose (FMHA C→A pattern). Pays only when it deletes an LDS buffer + barrier.
  Needs K0%16==0.
- **conv1 ILP (prefetch_k + sched_fuse)** — use on any short-K WMMA loop
  (2–3 K-atoms) where latency, not VGPR pressure, is the wall. **Always ship the
  pair, never one alone.** It buys in-flight depth for free until VGPRs bind.
- **sched hints (compv3/compv4)** — cheap, correctness-neutral ~+2 %; adopt as
  default candidate, but expect little when K-atom counts are tiny.
- **pk_maxpool / VALU micro-opts** — only worth it *after* the GEMM-pipeline
  work, and only if the epilogue is on your critical path (here it's ~10 %).
  Ceiling ~2 %.
- **AVOID by default:** waves_per_eu=2 (VGPR-bound occupancy), mask_maxpool
  (uniform control flow), butterfly/repack_c0 (cross-lane shuffle > LDS round
  trip). **persistent grid-stride** is **board-sensitive and not competitive with
  the i8+pk winner** (board2 −2–3.5 %; board1 beats old baseline but still loses to
  i8+pk) — **lower priority**, not a hard AVOID. Revisit waves_per_eu only if you
  first trim LDS *and* VGPR enough to genuinely land 2 WG/CU. Revisit persistent
  **only** with a software-pipelined tile loop (double-buffered footprint), and even
  then expect ~parity at best — see §3 deep-dive.

---

## 5. Future plan: getting under 10 ms

~86 % of wall is **conv1 operand-delivery + mandatory transpose + WMMA
latency**. The path to <10 ms is hiding that latency, since the transpose itself
can't be removed and K can't grow. Ordered by expected payoff / risk:

1. **Push conv1 MMA in-flight depth further (extend the ILP win).** The
   prefetch_k+sched_fuse combo proved the scheduler can interleave when fed, and
   the warp-16×1 retune (§2.F, +23.8 %) showed max-waves is the dominant
   latency-hiding lever once the levers are on. Next: explicit **double-buffer of
   the k=1 A/B fragments in VGPR under the k=0 MMAs** (global→VGPR→LDS only;
   **no async-to-LDS on RDNA3.5**); also probe warp_m beyond 16 / warp_n>1 now
   that 16×1 won — the optimum moved, so the next retune may move it again.
   Bounded by VGPR — measure VGPR each step. *The single most promising lever.*

2. **Land 2 WG/CU honestly.** Today one WG/CU at ~36–42 KB LDS, VGPR-bound. A
   real 2nd resident wave would hide conv1 latency the way warp-count did in
   fp16. Requires *simultaneously* trimming LDS (e.g. the shelved
   `C1_pool_smem[pool_px,4,K1]` reorg: 42→38 KB — not enough alone) **and** VGPR
   below the 2-WG threshold. Only pays as a package; neither half alone helped.

3. **Preshuffled / packed W1 (and W0) layout.** W1 is static — pre-pack it
   one-time into the exact iu4 B-fragment ABI so conv1 B-loads are pure bitcasts
   with minimal address math (HipKittens/AITER "B already kernel-friendly"
   pattern). Reduces the LDS-load latency that *is* the exposed cost.

4. **Tighter operand-load scheduling around the transpose.** Place
   `s_waitcnt`/`sched_barrier`/`s_setprio` deliberately around the
   permlanex16/v_perm_b32 transpose + ds_reads so the WMMA units never stall on
   operand delivery (HipKittens deliberate-placement principle).

5. **Retune geometry from the new i8+pk baseline first.** conv1_int8 + pk_maxpool
   changed the per-tile pipeline (deleted nibble-pack bitwork, i16 maxpool), and
   geometry optima are **not portable across lever sets** (§2.F). Re-sweep warp_m /
   warp_n / pool-tile shape on top of the i8+pk winner before chasing levers 1–4 —
   the optimum may have moved again.

> *Landed since the last revision (no longer "future"):* conv1_int8 (iu8 over
> int4-range codes, deletes nibble-pack), pk_maxpool (i16 `vector.smax`), and the
> nibble-pack cleanup — all shipped in the i8+pk winner (§2.G). Remaining future
> work is W1 staging/preshuffle (lever 3), deeper conv1 operand prefetch (lever 1),
> and the geometry retune (lever 5).

**Floor check:** padded matrix ceiling ≈ 2.2 ms; realistic near-term band
3–5 ms once latency is hidden. Sub-10 ms is a ~1.2× improvement on the current
~11.68 ms i8+pk best — achievable from levers 1–3 stacked, all latency-side, none
VALU-side.

---

## 6. Library findings (what to borrow, what to skip)

From inspecting CK Tile grouped-conv, HipKittens, AITER/opus_gemm, rocBLAS for
this fixed-shape fused op:

- **CK Tile grouped conv** — *borrow* the implicit-GEMM "specialized descriptors
  + GEMM-like inner loop" shape, bank-aware XOR LDS permutations, and up-front
  vector-access validation; *skip* the runtime split-image/split-N transform
  machinery — it adds exactly the coordinate arithmetic we're removing. Its async
  global→LDS path is gfx950-only.
- **HipKittens** — *borrow*: precompute global buffer resources + LDS byte bases,
  minimal per-iter address math; small ping-pong LDS with deliberately placed
  `s_waitcnt`/`s_barrier`/`sched_barrier`/`s_setprio` around load/DS-read/WMMA
  clusters; size tile mem ops to the matrix layout with pre-swizzled offsets.
  *Confirms* the butterfly rejection: don't replace an LDS handoff with cross-lane
  shuffles unless the shuffle count is demonstrably lower.
- **AITER / opus_gemm** — *borrow*: keep A/B layouts explicit as separate
  compile-time objects (global/shared/register/accumulator); assume B is in a
  kernel-friendly layout — for our static W0/W1 this means a **one-time
  packed/preshuffled weight layout** (future-plan lever 3). *Skip* the full deep
  multi-buffer pipeline — our tiny K counts won't amortize it.
- **rocBLAS** — lesson only: specialize + autotune concrete low-bit problems
  rather than building one generic conv kernel; the cloned repo doesn't expose
  reusable Tensile int8 kernels.

---

## 7. Reproduce

From the `rocke` Python root, `dnn-providers/hip-kernel-provider/rocKE/Python` (dev host =
gfx950, builds gfx11 ELFs but can't run them; full-shape timing happens on the
gfx1151 board). `export PYTHONPATH=$(pwd)` and set `VENV` to your interpreter.

```text
# Correctness (bit-exact) — toy + multi-CTA, runs anywhere that can build.
# With the default 2x64 pool tile, W must be a multiple of 128 (H of 4); a
# too-small shape exits "invalid spec: pool dims ... must be divisible ...".
PYTHONPATH=$(pwd) $VENV \
  -m rocke.examples.gfx1151.deep_conv_fusion.deep_fused_conv_pool_verify \
  --arch gfx1151 --verify --native-int --direct --h 4 --w 128 --c 8 --k0 32 --k1 24
#   also --h 32 --w 128 for the multi-CTA grid (1,8,1)

# Best config = the structural DEFAULTS (warp 16x1 + native_int + direct +
# fused_c0a1 + prefetch_k + sched_fuse, pt2x64, compv4) PLUS the i8+pk winner
# additions --conv1-int8 --pk-maxpool, which are NOT yet default-on and must be
# passed explicitly. Build hsaco + manifest for the board (gfx11-generic, NOT gfx1151):
PYTHONPATH=$(pwd) $VENV \
  -m rocke.examples.gfx1151.deep_conv_fusion.deep_fused_conv_pool_verify \
  --arch gfx11-generic --n 1 --h 2160 --w 3840 --c 8 --k0 32 --k1 24 \
  --conv1-int8 --pk-maxpool \
  --emit-hsaco /tmp/deep/deep.hsaco

# Same-session interleaved A/B at full shape (ONLY valid way to compare on the
# auto-clocking board): compare_configs.py (in-process builds) or
# compare_prebuilt.py (manifest-driven over prebuilt hsacos, with --rotate):
PYTHONPATH=$(pwd) $VENV \
  -m rocke.examples.gfx1151.deep_conv_fusion.compare_configs \
  --h 2160 --w 3840 --rounds 8 --iters 50 --warmup 200
```

Discipline: every lever behind a correctness-neutral spec toggle; verify
`max_abs_diff=0` at toy / multi-CTA / full before any speed quote; speed only
from same-session interleaved A/B. rocprofv3/ATT unavailable on the Windows
board → interpretation is bench + static-ISA reasoning, always confirmed by A/B.

---

## 8. Other docs in this folder
This case study is the optimization record. The other files are operational
reference, not analysis: `README.md` (overview, quant explanation, run commands),
`ALGORITHM.md` (the algorithm and per-CTA steps from the math up), and
`inputs.md` (the `encoder_0` problem spec).
