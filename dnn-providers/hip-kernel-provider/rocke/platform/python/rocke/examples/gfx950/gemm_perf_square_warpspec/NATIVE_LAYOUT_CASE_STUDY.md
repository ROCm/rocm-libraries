# Native NN / TN / TT layouts for the universal GEMM on gfx950 (case study)

Measured scope: the dispatched universal-GEMM bodies on gfx950 (f16 and bf16 data, f32 accumulation) for the
BLAS row-major layouts NN (`RRR`: B stored `[K, N]`), TN (`CRR`: A stored `[K, M]`, B `[K, N]`) and TT (`CCR`:
A stored `[K, M]`), compared with (a) the previous rocKE method, a transpose pass per MN-contiguous operand
(the faster of the tuned rocKE `transpose2d` and the framework copy) followed by the `RCR` dispatcher pick, and
(b) the hipBLASLt heuristic top-8 solutions for the same layout, all in one process, interleaved, in seeded random
order (5 rounds, median of hot batches). Shapes: a quasi-random sample of 64 points per dtype from the reference
shape measure of the dispatcher sweep (M, N log-uniform in [128, 32768], K in [64, 32768], alignment mix 40 / 25 /
25 / 10 % of multiples of 256 / 64 / 8 / unaligned), each in all four layouts; 72 large aligned and padded cells;
a 480-cell edge-shape set (M, N, K from 1 to 8192, odd, tiny, skinny). Every kernel of every cell passed an on-device
full-output check against an f64 reference on random data before and after its timing (tolerance: one output
rounding plus f32 accumulation). Relative results only; hot-batch timing is not a sustained power-capped window,
so ratios within about 2 % of 1.0 are noise.

## What was built

| piece | what it does |
|---|---|
| `DataSpec.layout` `RRR` / `CRR` / `CCR` | an operand stored MN-contiguous (A as `[K, M]`, B as `[K, N]`); the kernel signature (A, B, C, M, N, K) is unchanged; the leading dimension of an MN-contiguous operand is M (A) or N (B) |
| `instances/common/gemm_mn_operand.py` (DTL ring) | the LDS stage of an MN-contiguous operand is a sequence of 1 KiB blocks (one 16-byte-per-lane DTL instruction per wave fills one block: R = 512 / W K rows of W consecutive M / N elements) with `mn_pad` bytes after each block; the K bits read together by one half-wave of `ds_read_b64_tr_b16` select 8 distinct blocks, so the transpose reads are bank-conflict free; every fragment address is one lane value per (operand, stage) plus an immediate. The fragment equals the K-contiguous `ds_read_b128` fragment, so MFMA issue and epilogues are unchanged |
| VGPR-staged loader (no DTL ring) | `mn_vec` consecutive M / N elements per global load, written transposed into the K-contiguous LDS tile; serves the generic bodies (cshuffle, mem, decode) and every arch without transpose reads (gfx942) |
| `kc_row_guard` | the VGPR-staged loader zero-fills K-contiguous rows past M / N; always on for the non-RCR layouts |
| `sched_strategy` | the AMDGPU machine-scheduler strategy as a spec field |
| dispatcher | `native_layout_alternatives` derives the native form from the `RCR` candidate (the same body, the LDS-row-interleave body without its vector epilogue, `mfma_swap_ab`, a flipped 192 / 128 tile, the B3 ring, the 128x128 ring, the padded generic body) and the candidates admit the three layouts; a lowest-priority padded universal fallback candidate (all layouts) |

## Results (relative)

* Coverage: the dispatcher serves every point of the 19,328-point reference grid in every layout on gfx950 and
  gfx942 (exact CPU dispatch); before, NN / TN / TT ran only through the transpose pass and the `RCR` dispatcher
  refused about 40 % of the measure.
* Quasi-random sample (384 non-NT cells): the native pick is at least as fast as the best hipBLASLt solution on
  about 30 % of the NN / TN / TT cells (90 % bootstrap interval 26-34 %), against about 12 % for the transpose-pass
  method in the same runs and about 21 % for NT. The cells served by a DTL ring body win about two thirds of the
  time; the remaining losses are the generic-body classes NT shares (skinny, ragged K, few tiles).
* Large aligned and padded shapes (72 cells): the native pick wins 60 of 72; geomean speedup over the best
  hipBLASLt solution of the same layout 1.04-1.27 per (dtype, layout).
* The hipBLASLt solutions for NN / TN / TT were slower than its NT solutions on most shapes, while the native rocKE
  bodies ran within a few percent of the `RCR` kernel on the same shape.

## Findings worth reusing

1. **Two 64-bit LDS reads assembled into one 128-bit MFMA operand raise VGPR demand** in the unrolled
   LDS-row-interleave main loop (accumulators shuffled through VGPRs, spills); a single 128-bit read does not. The
   cause is the operand assembly, not the transpose instruction (plain 64-bit reads behave the same). The
   `iterative-minreg` scheduler strategy removes the shuffles when one operand is MN-contiguous; with both operands
   MN-contiguous the body still spills, so TN uses the 32-deep B3 ring instead.
2. **With the default scheduler, one such kernel computed wrong results** (deterministic, rows of the tiles a
   persistent workgroup processes after its first one). The validity gate admits the interleave bodies with an
   MN-contiguous operand only with `iterative-minreg`.
3. **16-byte LDS-DMA loads need 16-byte-aligned rows.** Rows starting 2-byte aligned (an odd leading dimension)
   returned wrong elements; such operands take the VGPR-staged loader.
4. **The persistent 16-wave ring with an N-contiguous B spilled VGPRs to scratch** (128-VGPR budget per wave) at
   the fragment assembly of finding 1; a WaveScope capture showed the scratch reloads' `s_waitcnt vmcnt(0)` as the
   largest stall class, because each also drains the in-flight DTL loads. The native form uses one workgroup per
   tile there, which does not spill.
5. **The MN chunk width should follow the wave's MN extent on 256-wide tiles** (128 for 2x2 waves, 64 for 4x4):
   the widest chunk was the slowest form in paired screens on the 256-wide bodies; 128-wide tiles keep 128.
6. **NN on the one-workgroup-per-tile interleave body:** the LDS-staged cshuffle epilogue beat the scattered
   default epilogue at long K (the 16-byte vector epilogue needs B's interleaved column order, which transpose
   reads cannot produce).
7. **Padded rows of the VGPR-staged loader read past the operand** (and faulted on the GPU for M = 21 / 30 with a
   16-row tile); `kc_row_guard` zero-fills them and the dispatcher only admits ragged shapes on guarded bodies.

## Knobs (all kept; defaults first)

`TraitSpec`: `mn_pad` [32] / 0 / 16 / 64, `mn_chunk` [auto: the wave extent on 256-wide tiles] / 64 / 128 / 256, `mn_vec` [8] / 4 / 2 / 1,
`sched_strategy` [""] / iterative-minreg / iterative-maxocc / max-ilp / max-memory-clause / iterative-ilp,
`kc_row_guard` [False]. Dispatcher (`dispatch/gemm/common.py`): `NATIVE_LAYOUT_ORDER`,
`NATIVE_LAYOUT_B3_FALLBACK`, `NATIVE_LAYOUT_B3_PERSISTENT`, `NATIVE_LAYOUT_RILV_SCHED`,
`NATIVE_LAYOUT_PERSISTENT_B_MN`, `NATIVE_LAYOUT_NN_CSHUFFLE`, `UNIVERSAL_FALLBACK`.

All new `TraitSpec` fields and the non-RCR layouts are Python-engine-only (`core/backend.py` refuses them on the
C++ spec converter); every default-RCR kernel is byte-identical to the previous tree.

## Learned selection among native forms, plan-level pre-pass, 16-byte swapped epilogue

Measured scope: gfx950, f16 / bf16, NN / TN / TT; held-out cells drawn uniformly from the NN / TN / TT Sobol points
of the same reference shape measure (excluding every shape any training run measured), 192 + 192 + 384 + 384 cells
in four successive evaluation batches; per cell the dispatcher pick (learned selection on), the rule pick
(`ROCKE_GEMM_SELECTOR=0`), the selector's top-6, every native form and knob variant of the `RCR` pick, the
transpose pass + `RCR` pick, the plan arm (below) and the hipBLASLt heuristic top-8, timed in one process as above;
every kernel passed the full-output f64 check before and after timing. Relative results only.

* **Native-form selection** (`dispatch/gemm/selector.py`, `gemm_native_<arch>` models): the RCR selector's
  gradient-boosted model, retrained on every measured native arm with data-layout features, ranks the registered
  candidates' native specs, every native form of every registered `RCR` spec and one-step mutations (MN chunk
  width, block pad, ring depth, swizzle, persistent grid, the RCR levers); the rule order stays the fallback
  (`ROCKE_GEMM_SELECTOR_NATIVE=0`). Pick at least as fast as the best hipBLASLt solution: about 61-62 % of the cells
  in the last three batches vs about 51-52 % for the rule pick in the same runs (geomean pick / rule 1.09-1.12; about
  a third of the cells faster by more than 3 %, about 3-4 % slower by more than 3 %).
* **Plan-level pre-pass choice** (`dispatch/gemm/plan.py`): the native model's time prediction of the native pick vs
  the `RCR` pick plus a transpose cost model (launch floor + bytes / sustained bandwidth, fitted on the measured
  rocKE `transpose2d` arms); eligible when both transposed extents are multiples of 64. With rocKE transposes only,
  the plan adds about 1-1.6 points of win share (NN about 2-4.5); it never chose a slower pre-pass in the offline
  check.
* **16-byte swapped-accumulator epilogue** (`TraitSpec.sab_vec16`): with `mfma_swap_ab` each lane holds 4
  consecutive N columns of one C row; the N-contiguous B's transpose reads address their 4-element MN quads in the
  order (0, 2, 1, 3) (same address set per 16-lane group, so the same banks), which puts the two halves of 8
  consecutive columns in lanes L and L + 32; one `v_permlane32_swap` per dword gives each lane one M atom's 8
  columns, written with one 16-byte store per atom pair. All arms passed the f64 check. Timing: faster than the
  8-byte `mfma_swap_ab` form, at parity with the scattered and cshuffle forms on the interleave body, slower on the
  B3 ring; streaming stores slower everywhere.

### Findings worth reusing (learned selection)

8. **Selection among existing native forms mattered more than any single body change:** the native form of the
   best `RCR` candidate (the RCR selector's pick) was often much faster than the native form of the rule layer's
   own priority pick; a model that scores native forms with layout features closed most of that gap.
9. **The C store width was not the NN / TN bottleneck.** A WaveScope capture of a short-K NN / TN shape showed the
   store stall share of the native forms close to NT's although the NT interleave body issues 8x fewer, 8x wider
   stores; with the accumulator transposed (`mfma_swap_ab`) one store instruction covers 32 rows x 32 bytes vs
   4 rows x 256 bytes for the interleave vector epilogue, and the swapped MFMA operand order costs the B3 main loop
   more than the wider stores save. TT ran at NT parity in the same capture set.
10. **An odd leading dimension of an MN-contiguous operand** (about an eighth of the NN / TN / TT measure) leaves
   only the VGPR-staged loader with 1- or 2-element loads; no native form won there, and a pre-pass with ideal
   transposes would win only a few such cells (the `RCR` GEMM on those ragged shapes is near parity itself).

Knobs (learned selection): `ROCKE_GEMM_SELECTOR_NATIVE`, `SELECTOR_NATIVE_SOURCES`, `SELECTOR_NATIVE_NEIGHBOUR_LEVERS`,
`SELECTOR_NATIVE_NEIGHBOUR_BASES` [8], `ROCKE_GEMM_PLAN_PREPASS` [auto] / off / always,
`GEMM_PLAN_PREPASS_MARGIN` [0.05], `TRANSPOSE_COST`, `TraitSpec.sab_vec16` / `sab_store_nt`,
`NATIVE_LAYOUT_SAB_VEC16` [True], `NATIVE_LAYOUT_SAB_VEC16_RINGS` [False], `NATIVE_LAYOUT_SV16_NT` [False]. The new
fields are Python-engine-only; default `RCR` kernels are unchanged.
