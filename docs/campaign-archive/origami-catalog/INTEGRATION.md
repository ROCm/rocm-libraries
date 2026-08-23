# Integration into `users/vmijovic/gfx1100-hhs-tn-v6-stock-ship`

Worktree: `/home/vmijovic/exp/shipv6`. **Committed locally, not pushed.**

| commit | change |
|---|---|
| `018e2935366` | add HSS-TN v3 catalog (72 entries), delete the 3-solution GridBased logic it replaces |
| `6bacbd2b1f6` | record which traffic reaches it, and prove selection |

Branch now carries two stock-Origami catalogs: **v6** (HHS-TN, 58 entries, pre-existing) and
**v3** (HSS-TN, 72 entries, new).

## What was integrated, and what was not

| target | catalog | measured | integrated? | why |
|---|---|---|---|---|
| HSS-TN | v3, 72 kernels | **1.177** geomean, P10 0.862 | **yes** | all gates pass |
| HHS-NN | v3, 72 kernels | 1.029 geomean, P10 0.853 | **no** | **K6 fails** |
| HSS-NN | — | — | no | not built yet |

**HHS-NN was deliberately not integrated.** It passes K4 (+3.10% off-table, bar 0.5%) but fails
K6: P90 falls 60,836 → 54,304 GFLOP/s, per-tier P10 drops in `large`, `medium` and `tiny`, and
39% of shapes regress. Integrating it would also require **deleting production's 471-point exact
table** for that ProblemType — the very thing that causes the regression, since a `Prediction`
catalog sets `[7]` to null. Shipping it would trade a well-tuned 70-solution library for a 3%
mean gain paid for out of the tail.

The logic file is built and validated (`/home/vmijovic/catalog_hhs_nn/logic/v3/`), so it can be
integrated later if the two-row approach below is implemented.

## Why the ProblemTypes collide

`ProblemType` is byte-identical between the new catalog and the G0 logic it replaces, so the two
**cannot coexist** — whichever library loads first answers every query. The 3-solution
`navi31_Cijk_Alik_Bljk_HSS_BH_Bias_HAS_SAV_UserArgs.yaml` is therefore deleted, exactly as the
298-solution HHS-TN logic was in `421797d0464`.

## Which traffic reaches the catalog

This is the part most likely to be misread. The catalog serves
`..._Bias_HA_S_SAV_UserArgs`, and in a full product library that path is reached **only when
ScaleAlphaVec is requested**:

```
plain GEMM            -> Cijk_Alik_Bljk_HSS_BH_Bias_HA_S_UserArgs_MT32x32x64      (NOT this catalog)
--bias_vector         -> Cijk_Alik_Bljk_HSS_BH_Bias_HA_S_UserArgs_MT32x32x64      (NOT this catalog)
--scaleAlpha_vector   -> Cijk_Alik_Bljk_HSS_BH_Bias_HA_S_SAV_UserArgs_MT128x96x32 (this catalog)
```

A plain call is served by a separate 4-solution non-SAV library, untouched by this change. The
deleted logic served the same ScaleAlphaVec path, so the swap is like-for-like and the A/B is
the right comparison — but every measurement in this campaign was taken against libraries
holding **only** that ProblemType, where a plain call has nowhere else to go.

**Therefore 1.177 is the speedup of the ScaleAlphaVec path, not a product-level speedup.**

## Verification performed

1. **Whole branch builds** — `TensileCreateLibrary` over the full `asm_full` set: rc=0, 1,620
   kernel objects, 40 serialized libraries, no chip-ID validation error.
2. **Chip-ID validation passes** — the emitter was fixed to write the device list in flow style
   (`- [Device 73f0]`). The block form `- - Device 73f0` parses to the identical object but fails
   `_LIST_DEVICE_LINE_RE`, which reads the header as text. This is now enforced in
   `emit_pool_logic.py` with an assertion, not left to chance.
3. **Selection proven** — 8/8 sampled shapes across regimes are served by a v3 entry when
   ScaleAlphaVec is requested (5 SK3, 3 SK0).

> **Normalise `SKWS\d+` before comparing kernel names.** Tensile added that token after these
> catalogs were serialized, so comparing the runtime's `--Solution name:` literally against the
> stored `SolutionNameMin` reports **0/8** for kernels that are in fact the catalog's. A naive
> check here produces a confident false negative.

## Not done

- **Not pushed.** No remote was touched.
- HHS-NN two-row library (`Prediction` gated by a size predicate + `Matching` catch-all) that
  would preserve the 471 exact points — the only path that makes it shippable.
- HSS-NN (third requested target) not started.
