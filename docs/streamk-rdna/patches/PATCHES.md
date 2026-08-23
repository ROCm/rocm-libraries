# Patch set — what each file contains, and whether it is a fix or an experiment

All were applied to `~/exp/stock` (a TheRock/rocm-libraries worktree). Verified against the
working tree 2026-08-22: `rocisa_glc_fix.patch` and `sk4_clamp_fix.patch` still match byte
for byte.

## Bug fixes — proposed upstream, independent of what ships

| file | what | writeup |
|---|---|---|
| `rocisa_glc_fix.patch` | A gfx12-only `th:TH_ATOMIC_RETURN` emitted unconditionally where gfx11 spells it `glc`. **Two independent sites**, both fixed: `GlobalAtomicIncU32Saddr` (broke `StreamK: 4`/`5`) and `FlatAtomicDecU32` (broke GSU `MultipleBufferSingleKernel`). Both validated PASSED on gfx1100. | `../UPSTREAM_PR_1_glc.md` |
| `sk4_clamp_fix.patch` | `TENSILE_STREAMK_TILES > tiles` wraps a `uint32_t` and crashes the GPU (25 launch failures reproduced). The SK3 path already had the clamp; SK4 was missing it. One line. | `../UPSTREAM_PR_2_clamp.md` |

## Experiments — env-gated, all default to byte-identical-to-stock

`origami_streamk_experiments.patch` — **contains all three** knobs in
`shared/origami/src/origami/streamk.cpp`, not just one. Do not read the filename as scoping
it to the gate:

| env var | default | what it does | added |
|---|---|---|---|
| `ORIGAMI_THRESHOLD_MULT` | 1 = stock | scales the `tiles > cu_count` branch threshold without moving the per-branch grid ceiling | earlier campaign |
| `ORIGAMI_SMALL_CU_MULT` | 1 = stock | raises the small-problem branch's grid ceiling without moving the threshold | earlier campaign |
| `ORIGAMI_MN_GATE` | 0 = **disabled** | above `M*N >= gate`, `grid_k_split_aware` returns `tiles` — exactly what `grid_data_parallel()` returns, so the launch becomes data-parallel-equivalent | **2026-08-22** |

`ORIGAMI_MN_GATE` is the one measured in `../GATE_RESULT.md` (+1.34% suite wall-clock,
9000 measurements) and `../SHIP_TEST.md` (+2.39% within-catalog, 12000 measurements).

Also patched but **not** in this set: `shared/origami/src/origami/hardware.cpp` carries
`ORIGAMI_RDNA_CU_MULT` (default 2 = stock); its pre-patch copy is kept beside it as
`hardware.cpp.pre-mult-experiment`.

## Baseline

`P0_baseline.patch` is the revert point captured before any of this campaign's changes —
`exp/stock` was **already dirty** with the origami experiments when the StreamK campaign
started, so this records what "clean" meant at that time.

## Caveat that governs all of it

A default gfx1100 build ships **no StreamK kernels** (2560 `StreamK: 0` vs 22 `StreamK: 3`,
all 22 in `Experimental/`, excluded by `tasks.py` default). The two bug fixes matter
regardless — they are correctness. The gate cannot fire on a default build. See
`../GATE_RESULT.md` "SCOPE".
