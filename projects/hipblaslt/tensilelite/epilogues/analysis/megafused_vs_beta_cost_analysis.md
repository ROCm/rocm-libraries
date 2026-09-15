<!-- Copyright Advanced Micro Devices, Inc., or its affiliates. -->
<!-- SPDX-License-Identifier: MIT -->

# MegaFused epilogue cost analysis: why it outpaces beta\*C and why tile-area correlation fails

**Problem:** GEMM 8192×8192×8192, bf16/bf16, HPA, TN, batched, gfx950 wave64.
**Winner tile:** MT384×256, MI[16,16,32,1,1,12,8,2,2], DU64, PGR2, StreamK3, UseSubtileImpl.
**Emitter commit:** 458b9f0c24d ("hoist ResidualOut token\*N\_hidden multiply out of MegaFused store").
**Data source:** single Tensile run `/tmp/baselines-sweep` via `epilogues/YAMLs/benchmark_baselines_sweep.yaml`; parser `epilogues/analysis/parse_baselines.py` → `epilogues/analysis/parse_baselines.out`; disassembly from `.s` files under `/tmp/baselines-sweep/1_BenchmarkProblems/.../assembly/`.

---

## 1. Corrected baselines (winner tile MT384×256, median of 10 samples)

> **Inversion notice:** the original task framing ("MegaFused ~573 µs, faster than the ~753 µs no-epi baseline") was inverted. The fresh measurement shows MegaFused is the **slowest** configuration, not the fastest.

| Group | Config | median µs | delta vs no-epi |
|-------|--------|----------:|----------------:|
| A | no-epilogue (pure GEMM, BBS\_H) | 751.2 | — (baseline) |
| D | beta\*C (BBS\_BH, UseBeta) | 787.3 | +36.1 |
| B | chain epilogue (split emitters, MFE0, PRMS+RA+bf16D) | 833.6 | +82.4 |
| C | MegaFused (MFE1, PRMS+RA+bf16D) | 874.4 | +123.3 |

ratio\_fused\_to\_beta = delta\_fused / delta\_beta = 123.3 / 36.1 = **3.41×**.
Best-sample deltas were +34.6 / +79.5 / +119.3 (ratio 3.45×) — fully consistent with the median ranking.

Note that the chain (split three-emitter) path sits between beta\*C and MegaFused: MegaFused is costlier than even the serialized split-emitter path, confirming the overhead is intrinsic to the MegaFused store body, not to emitter dispatch or kernel launch.

---

## 2. Tile-area correlation: MegaFused vs beta\*C

Over 10 MegaFused tiles that survived codegen (each paired with the same-run no-epi baseline at the same tile):

| Metric | MegaFused | beta\*C (prior 204-config sweep) |
|--------|----------:|--------------------------------:|
| corr(tile area M×N, delta) | +0.085 | +0.40 |
| corr(tile area M×N, absolute µs) | −0.708 | — |
| corr(workgroup count, delta) | −0.186 | −0.36 |

The beta\*C mechanism is clear: C-loads = 6×waveTileArea per workgroup, so overhead grows with area and shrinks with workgroup count. No such mechanism exists for MegaFused.

> **Caveat:** only 10 MegaFused tiles survived (many large tiles were rejected for occupancy/VGPR limits) versus 204 beta configs, so +0.085 is a weak-sample estimate. The qualitative conclusion — MegaFused overhead does not scale with output-tile area the way beta\*C does — is nonetheless clear from the per-tile spread below.

**Per-tile breakdown (sorted by area):**

| Tile | area (M×N) | workgroups | no-epi µs | MegaFused µs | delta µs |
|------|-----------:|-----------:|----------:|-------------:|---------:|
| MT576×64 | 36,864 | 1,920 | 1,320.4 | 1,397.0 | +76.6 |
| MT128×384 | 49,152 | 1,408 | 895.9 | 1,026.2 | +130.3 |
| MT448×128 | 57,344 | 1,216 | 1,010.3 | 1,239.0 | +228.7 |
| MT128×512 | 65,536 | 1,024 | 774.8 | 914.8 | +140.0 |
| MT512×128 | 65,536 | 1,024 | 780.2 | 875.0 | +94.7 |
| MT384×192 | 73,728 | 946 | 802.9 | 879.5 | +76.6 |
| MT192×448 | 86,016 | 817 | 855.2 | 1,000.3 | +145.1 |
| MT448×192 | 86,016 | 817 | 846.0 | 987.9 | +142.0 |
| MT256×384 | 98,304 | 704 | 763.0 | 892.8 | +129.8 |
| MT384×256 | 98,304 | 704 | 751.2 | 874.4 | +123.3 |

The two largest-area tiles (98,304) produce middling deltas (+123–130 µs), while the mid-area MT448×128 (57,344) produces the largest delta (+228.7 µs). There is no monotone area trend.

---

## 3. Instruction-count comparison (winner tile MT384×256)

> **Critical methodology note — assembled vs executed:** the counts below are **static assembled** instructions over each kernel's whole Global-Write region. That region always assembles edge-remainder and Beta==0/Beta!=0 fallback paths that **do not execute** for this aligned problem: M=8192, MT\_M=384 → last-WG M-remainder is 128; 128 % 8 = 0 so the M-edge path is permanently dead. These totals therefore overstate executed work and must not be read as executed cost. Reviewer-verified dead-code volume: beta's Beta!=0 region contains approximately 14,126 assembled-but-never-executed edge instructions; MegaFused's epilogue contains approximately 5,614 dead edge instructions.

**Assembled instruction counts, whole Global-Write region** (ranges: beta 8,772–31,692 / chain 2,937–28,860 / MegaFused 2,937–24,826):

| Category | beta\*C | chain | MegaFused |
|----------|--------:|------:|----------:|
| buffer\_load (all) | 576 | 108 | 108 |
| buffer\_store (all) | 1,056 | 1,592 | 1,064 |
| v\_pk\_add | 0 | 0 | 192 |
| v\_fma / v\_fmac | 1,152 | 376 | 384 |
| v\_pk\_mul | 941 | 568 | 760 |
| v\_cmp | 1,920 | 1,191 | 2,257 |
| v\_cndmask | 1,448 | 2,885 | 1,636 |
| s\_waitcnt | 163 | 166 | 161 |
| s\_and\_saveexec | 0 | 1 | 97 |
| s\_mov\_b64 exec | 192 | 193 | 385 |
| ds\_bpermute | 384 | 784 | 208 |
| **Total (assembled epilogue instrs)** | **20,007** | **23,856** | **19,792** |

**Executed-path figures** (the ones the conclusions rest on):

- **Executed beta\*C epilogue** = NonEdge deferred store batch only ≈ **5,881 instructions**; executes 96 `buffer_load_dwordx2` C-loads.
- **Executed MegaFused epilogue** = pre-Global-Write fused pass + NonEdge store ≈ **14,178 instructions**; executes 108 `buffer_load_dwordx2` residual+gamma loads.
- Executed MegaFused epilogue is **~2.4× beta's executed epilogue** (14,178 vs 5,881).
- Residual/gamma load volume (108) is essentially the same as beta's C-load volume (96) — loads are **not** the differentiator.
- Note that the residual and gamma tensors MegaFused reads are distinct input tensors from the layernorm/RMS path — not the GEMM C matrix that beta\*C reads; the point is only that the two epilogues issue comparable load volumes (108 vs 96 dwordx2), so load count is not what separates their cost.

**Whole-kernel assembled instruction-line totals** (approximately 0.15% counting tolerance): beta 26,842, chain 26,105, MegaFused 22,041. MegaFused has the **smallest** whole-kernel footprint yet the **largest** epilogue cost, ruling out an instruction-cache footprint explanation.

---

## 4. Why MegaFused cost does not track tile area (structural explanation)

**beta\*C mechanism.** The overhead is dominated by C-element loads: assembled count = 6×waveTileArea per workgroup, a quantity that grows with tile area and shrinks the workgroup count — producing corr(area, delta\_beta) = +0.40 and corr(numWG, delta\_beta) = −0.36.

**MegaFused mechanism.** The overhead is dominated instead by (i) a per-tile store body that is structurally heavy regardless of how many C-elements are loaded, and (ii) fixed-shape fused work. Its executed load volume (108) is comparable to beta's (96) and does not drive the 3.4× gap. Because the dominant cost is store-body structure, fused math, and a fixed cross-wave RMS reduction rather than load volume, the overhead does not scale with output-tile area, producing corr(area, delta\_fused) = +0.085.

**Evidence the fused overhead has a fixed/near-fixed-shape component.** The cross-wave RMS reduction block (MegaFused `.s` approximately lines 12,565–12,670, approximately 99 instructions, 2 `s_barrier`, ds\_bpermute butterfly) is sized by the wave-group layout (wg\_m), not by M×N. It is therefore roughly tile-area-independent. corr(wgCount, delta\_fused) = −0.186 shows tiles with more workgroups pay slightly more aggregate barrier latency, but the effect is weak — consistent with a mostly-fixed per-tile cost rather than a load-proportional one.

---

## 5. Ranked cost drivers for the ~123 µs MegaFused overhead

1. **Heavier executed store body and fused epilogue.** MegaFused executes approximately 14,178 epilogue instructions vs beta's approximately 5,881 (~2.4×). This is the dominant, roughly tile-area-independent serial post-loop tail.

2. **Per-tile store-body structure the beta path avoids.** 97 `s_and_saveexec` + 385 `s_mov_b64 exec` (vs 0 and 192 for beta), per-element `v_cmp`/`v_cndmask` masking, narrow `buffer_store_dwordx2` (4×bf16) instead of beta's cooperative `ds_bpermute` + `buffer_store_dwordx4` (8×bf16) under a scalar block mask.

3. **Genuinely extra fused math with no beta analog.** 192 `v_pk_add` residual add, 384 `v_fma` ΣH², 760 `v_pk_mul` gamma scaling, plus the fixed cross-wave RMS reduction (2 barriers).

4. **NOT drivers.** Residual/gamma load volume (108 ≈ beta's 96 C-loads) and instruction-cache footprint (MegaFused whole-kernel is the smallest of the three).

---

## 6. Recommended next action (highest leverage)

Adopt the beta\*C subtile GWB store idiom inside the MegaFused ResidualOut store path. The relevant emitters are `_storeResidualOutRow`, `_issueResidualOutWide`, `_issueResidualOutStraddle`, and `_computeBf16Addr` in `Tensile/Components/Subtile/SubtileMegaFusedEmit.py`.

Concretely: replace per-tile `token_n*N_hidden` addressing + per-element `v_cmp`/`v_cndmask` masking + `s_and_saveexec` + narrow `dwordx2` + always-emitted straddle fallback with (a) a precomputed column-address VGPR advanced by SRD/immediate row offsets, (b) the scalar block OOB mask beta uses (SubtileMGuard/NGuard) applied via a plain `s_mov_b64 exec`, and (c) a cooperative `ds_bpermute`-packed `buffer_store_dwordx4` (8×bf16).

Expected effect: removes the approximately 97 per-tile saveexec instructions, eliminates the per-element masking, and halves store-width overhead — attacking driver #2 directly, the largest addressable chunk of the ~123 µs gap. The fused math in driver #3 and the RMS reduction are largely irreducible feature cost. This recommendation matches the analysis in `beta_c_vs_megafused_bf16_schedule.md` and is the highest-leverage structural change available.

---

## Methodology / provenance

All timing data come from a single Tensile run at `/tmp/baselines-sweep` using `epilogues/YAMLs/benchmark_baselines_sweep.yaml` with NumBenchmarks=10; medians were extracted by `epilogues/analysis/parse_baselines.py` (full output in `epilogues/analysis/parse_baselines.out`; run log in `epilogues/analysis/baselines_sweep_run.log`). Instruction counts were derived by awk-slice and grep over the epilogue line ranges of the winner-tile `.s` files under `/tmp/baselines-sweep/1_BenchmarkProblems/.../assembly/`. The assembled-vs-executed distinction is reviewer-verified: the dead-code volumes quoted in Section 3 were confirmed by tracing edge-predicate logic against the aligned problem dimensions. The 10-tile correlation figures in Section 2 are weak-sample estimates due to the small surviving population; the directional conclusion is robust but the precise values should not be over-interpreted.
