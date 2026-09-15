# LibraryLogic winner vs. benchmark top performer — mismatch analysis

Fused bf16 PartialRMS + ResidualAdd + StoreBf16D (MegaFusedEpilogue path).

YAML: `epilogues/YAMLs/benchmark_fused_bf16_partialrms_residualadd.yaml`
Run output: `/tmp/tmp-bf16` — Benchmark log: `epilogues/analysis/fused_bf16_run.log`

## TL;DR

This is **not a LibraryLogic bug and not a data-pipeline bug**. The YAML declares
`LibraryType: Prediction`. For a Prediction library, LibraryLogic **deliberately discards
the benchmark-measured winner** and instead ships all candidate solutions; at runtime the
**origami analytical model** ranks them. Origami's model is structurally blind to (a) the
fused epilogue and (b) `PrefetchGlobalRead` / wave-group shape — the exact knobs that
separate the top performers here. So the origami winner differing from the measured top
performer is **expected behavior of the prediction pipeline**, not a selection error.

Note on the problem size: the task brief referenced `[8192, 8192, 1, 8192]`, but the YAML
benchmarks a single exact size `[16384, 2048, 1, 2048]` (M=16384, N=2048, K=2048, batch=1).
There is no 8192 size in this config; the analysis below uses the actually-benchmarked size.

## 1. The two solutions

### Benchmark top performer (measured)
- **Solution index 45**, GFlops = **808070** (highest of 55 candidates).
- Kernel: `...MT128x512x64_MI16x16x1..._PGR2..._WG16_16_1...`
- Params: MacroTile **128x512**, DepthU 64, MatrixInstruction [16,16,32,1] (kernel-name
  tag `MI16x16x1` = M x N x MIBlock; K=32 is the MFMA K dim), **MIWaveGroup [1,4]**,
  MIWaveTile [8,8], **PrefetchGlobalRead 2**, StreamK 3, SubtileImpl True.
- Confirmed as the exact winner by the analyzer itself — log line 326:
  `ExactWinners: {(16384, 2048, 1, 2048, ...): [45, 367.3]}` (367.3 ≈ 808070 GFlops / device
  max-freq, the `UseEffLike` metric).

### LibraryLogic "winner" (origami, runtime)
- The generated logic YAML
  `3_LibraryLogic/bench_fused_bf16_partialrms_residualadd_..._MFE_UserArgs.yaml` is a
  **Prediction** library: `LibraryType: Prediction`, `ExactLogic: []`, `RangeLogic: null`,
  plus a placeholder `DefaultSolution` (MatrixInstruction `[]`, DepthU -1) and all **55**
  solutions serialized verbatim.
- There is **no baked-in per-size winner**. The runtime picks via
  `ProblemPredictionLibrary::findTopSolutions` →
  `origami::rank_configs(problem, hardware, origami_config_list)`
  (`include/Tensile/PredictionLibrary.hpp:143-237`). Reproducing origami's exact pick here
  would need a full host build (the origami Python bindings are not installed in `~/.tensile`,
  and there is no standalone origami CLI — only the `origami-tests` binary and the bindings).
  It is not needed to explain the mismatch: the mechanism below shows origami cannot
  reliably reach solution 45.

## 2. Why the benchmark winner is thrown away (LibraryLogic pipeline)

`Tensile/LibraryLogic.py`:
- The exact size populates `exactWinners` with solution 45 (`addFromCSV`, lines 473-521;
  log line 326).
- For `LibraryType == "Prediction"` the analyzer takes the `deReferenceSolutions()` branch
  (lines 101-102) and then **skips range-logic and exact-logic emission entirely** — both are
  set to `None` (lines 200-228). No removals occurred (`removeInvalidSolutions` is a no-op
  here because `NumProblemSizes` is all-zero: there are only exacts, no ranges — log line 324).
- `createLibraryLogic` therefore writes `ExactLogic: []` / `RangeLogic: null`. **The measured
  winner (sol 45) never reaches the library.**

So the benchmark GFlops are used only to enumerate/validate candidate solutions, never to
select the shipped winner. This is by design for Prediction libraries.

## 3. How origami actually selects (runtime model)

- `origami::rank_configs` ranks configs by **predicted latency (lower is better)**
  (`shared/origami/src/origami/origami.cpp:657-815`, `stable_sort` on `latency`). Tie-breaks
  (within variance) use tile arithmetic intensity, then problem shape (M>N → prefer larger
  MT_M), then a deterministic larger-MT_M/N/K rule. It does **not** rank by GFlops or by any
  measured data.
- The Formocast simulator / estimation model
  (`shared/origami/src/simulator/tensilelite/formocast_simulator.cpp:493-740`;
  `src/origami/gemm.cpp`) models only **GEMM A·B compute + A/B global loads + the plain D
  store** (with cache modeling and occupancy). It has **no term** for bias, activation,
  residual/skip add, RMS/layer-norm, or beta·C — the estimation epilogue explicitly comments
  that items 2–5 (alpha/beta, bias, activation, conversions) are skipped
  (`gemm.cpp:1514-1526`). `Formocast::ProblemInfo` carries no fusion/epilogue fields at all.

## 4. The decisive gap — origami's config cannot see the winning knobs

When tensilelite builds each `origami::config_t` at library load
(`include/Tensile/Serialization/PredictionLibrary.hpp:116-134`), it sets only:

```
mt = {macroTile.x, macroTile.y, depthU},  mi = matrixInstruction,
hand_optimized_main_loop, subtile, occupancy, workgroup_mapping,
cache_hints_a/b, workspace..., stream_k, index
```

It **does not populate `tensile_params_t`** (the `backend` variant that holds
`prefetch_global_read`, `wave_group_m/n`, `depth_u`, …). Consequently:

- **`PrefetchGlobalRead` is invisible to origami.** For the same macro tile, PGR swings
  measured performance enormously. The six MT128x512 candidates:

  | sol | PGR | WaveGroup | GFlops |
  |----:|----:|:---------:|-------:|
  | 45  | 2   | [1,4]     | **808070** (measured winner) |
  | 44  | 1   | [1,4]     | 690499 |
  | 43  | 0   | [1,4]     | 545817 |
  | 19  | 2   | [2,2]     | 766603 |
  | 18  | 1   | [2,2]     | 672126 |
  | 17  | 0   | [2,2]     | 515438 |

  These six collapse to essentially **one origami config** (identical `mt`, `mi`,
  `workgroup_mapping=WGM8`, `stream_k`, `subtile`; only `occupancy` — derived independently per solution as `CUOccupancy` — can differ). Origami
  assigns them the same predicted latency and cannot prefer PGR2 over PGR0 — a **36% spread**
  it is blind to. It will emit whichever survives predicate filtering first, which need not be
  sol 45.

- Within-group spread is large across **every** tile family (origami treats each family as one
  config): 128x512 36%, 256x256 34%, 128x384 32%, 448x192 33%, 384x256 30%, 256x384 26%,
  192x448 28%, 448x128 27%, 384x192 24%, 512x128 21%, 256x192 17%, 576x64 5%.

- **The fused epilogue is unmodeled.** This size is memory-bound in its epilogue (residual-add
  read + RMS + bf16 D store on a 16384x2048 output). The extra input/output traffic shifts the
  real optimum toward N-heavy tiles like 128x512 that stream the wide-N output efficiently.
  Origami, modeling only a bare GEMM D store, has a different notion of the optimum and — via
  its M>N shape tie-breaker (M=16384 ≫ N=2048) — is actually biased toward **larger-MT_M**
  tiles (e.g. 512x128 / 448x192), i.e. the opposite of the measured winner's 128x512.

## 5. Verdict and recommendation

**Verdict: expected behavior of the `Prediction` library type — not a LibraryLogic bug and
not a data-pipeline bug.** The measured winner (sol 45) is correctly identified during
analysis but intentionally not shipped; origami re-selects at runtime with an analytical model
that (a) ignores the fused epilogue and (b) never receives `PrefetchGlobalRead` or wave-group
shape, so it cannot converge on sol 45's specific PGR2 / WaveGroup[1,4] configuration.

Recommendations (in increasing scope):
1. **If the goal is to pin the measured winner**, do not ship this as `Prediction`. Use a
   size-indexed library type (e.g. the default Exact/Range/`Matching` logic) so the benchmarked
   exact winner (sol 45) is baked in for `[16384, 2048, 1, 2048]`.
2. **Reduce origami's blind spots at library-build time**: populate `tensile_params_t`
   (`prefetch_global_read`, `wave_group_m/n`, `depth_u`) in the `origami::config_t` built in
   `Serialization/PredictionLibrary.hpp` so at least Formocast can differentiate the PGR /
   wave-group variants instead of collapsing them.
3. **Prune redundant candidates for Prediction runs**: since origami cannot distinguish PGR /
   wave-group within a tile family, offering all three PGR values per tile only lets it pick a
   worse one. For a Prediction library, keep one (best-known) PGR/wave-group per macro tile.
4. **Longer term**, extend the origami model to account for fused-epilogue traffic
   (residual-add read + norm + narrow-precision store) so its latency ranking reflects the
   MegaFused epilogue cost this YAML actually exercises.

## Appendix — reproduction
- Run: `source ~/.tensile/bin/activate && LD_LIBRARY_PATH=/opt/rocm/lib ./Tensile/bin/Tensile epilogues/YAMLs/benchmark_fused_bf16_partialrms_residualadd.yaml /tmp/tmp-bf16`
- Measured CSV: `/tmp/tmp-bf16/2_BenchmarkData/Cijk_Alik_Bljk_BBS_H_PRMS_RA_MFE_UserArgs_00.csv` (1 data row, 55 solution columns; solution columns start at CSV field index 11 (0-based), so field 56 = solution index 45).
- Generated logic: `/tmp/tmp-bf16/3_LibraryLogic/bench_fused_bf16_partialrms_residualadd_Cijk_Alik_Bljk_BBS_H_PRMS_RA_MFE_UserArgs.yaml` (`ExactLogic: []`, `LibraryType: Prediction`, 55 solutions).
