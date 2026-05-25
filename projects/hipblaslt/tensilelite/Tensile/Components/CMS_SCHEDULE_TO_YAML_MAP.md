# CMS Schedule → YAML Configuration Map

**Purpose:** Audit of which gfx950 CMS schedule functions are exercised by which YAML config files.
**Audience:** Team migrating kernel-flag declarations from schedule function bodies into YAML/schedule-metadata.
**Date:** 2026-05-22

---

## 1. Summary

| Count | Category |
|-------|----------|
| 38 | Total CMS schedule functions in `gfx950/` |
| 36 | Schedules with ≥1 YAML match |
| 2 | Orphaned schedules (no YAML match found) |
| 36 | Schedules matched by `custom_mainloop_scheduling.yaml` and/or `custom_mainloop_scheduling_tf32.yaml` |
| 1 | Schedule matched only by `example.yaml` (not a test YAML) |
| 1 | Schedule matched by `gfx950_Cijk_Alik_Bljk_BBS_BH_UserArgs.yaml` (library-logic YAML) |

**YAML files that exercise CMS schedules (primary):**
1. `Tensile/Tests/common/gemm/gfx950/custom_mainloop_scheduling.yaml` — 16-bit and 8-bit schedules, multiple layouts
2. `Tensile/Tests/common/gemm/gfx950/custom_mainloop_scheduling_tf32.yaml` — TF32 schedules, multiple layouts
3. `Tensile/Components/example.yaml` — development/exploration YAML, also sets `UseCustomMainLoopSchedule: 1`
4. `tests/configs/mixed_configs/gfx950_Cijk_Alik_Bljk_BBS_BH_UserArgs.yaml` — library-logic file (already-solved solution; no `UseCustomMainLoopSchedule` flag directly — it is a pre-tuned config for 256x192x64, 16-bit)

**Notes on methodology:**
- The YAML `MatrixInstruction` field format is `[MIm, MIn, MIk, MIb, InnerUnroll, MIWaveTile0, MIWaveTile1, MIWaveGroup0, MIWaveGroup1]`.
- `MacroTile0 = MIm * MIWaveTile0 * MIWaveGroup0`, `MacroTile1 = MIn * MIWaveTile1 * MIWaveGroup1`.
- The CMS dispatcher matches on: `MacroTile0/1`, `DepthU`, `PGR`, `PLR`, `DirectToLds`, `DtlPlusLdsBuf`, `WaveSeparateGlobalReadA/B`, `ISA`, then `GlobalReadVectorWidth A/B`, `LocalReadVectorWidth`, `MatrixInstruction[0:4]`, `MIWaveGroup`.
- "Exact match" below means the YAML explicitly exercises this schedule's TileConfig+MI+VectorWidths.
- "Approximate" means the YAML uses `UseCustomMainLoopSchedule: 1` with problem sizes matching the MacroTile, but does not explicitly lock down every CMS field; the dispatcher runtime selects which schedule applies.
- The DPLB field in TileConfig (7th positional arg, `dtl_plus_lds_buf`) is `True` only for `_256x96x64_16bit_DPLB`; all others have `False`.

---

## 2. Per-Schedule Table

The TileConfig positional args are: `(MT0, MT1, DU, PGR, PLR, DirectToLds, dtl_plus_lds_buf, WSGRA, WSGRB, isa=(9,5,0))`.
All schedules have `isa=(9, 5, 0)`, `WSGRA=0`, `WSGRB=0`, and (except DPLB) `dtl_plus_lds_buf=False`.

### TF32 Schedules (`dtype_predicate=isTF32`)
All TF32 schedules: `vector_widths=[4, 4, 4]` (GRVWA=4, GRVWB=4, LRVW=4), `matrix_inst=[16, 16, 32, 1]` (except `_128x128x32_TF32_plr1` which uses `[32, 32, 16, 1]`), `mfma_wave_group=[2, 2]`.

| Schedule Function | TileConfig (MT0,MT1,DU,PGR,PLR,DTL,DPLB) | matrix_inst | wave_group | Matching YAMLs | Confidence |
|---|---|---|---|---|---|
| `_128x128x32_TF32` | (128,128,32,2,1,1,F) | [16,16,32,1] | [2,2] | `custom_mainloop_scheduling_tf32.yaml` (multiple groups: TN+NN+NT), `example.yaml` | Exact |
| `_128x128x32_TF32_plr1` | (128,128,32,2,1,1,F) | [32,32,16,1] | [2,2] | `custom_mainloop_scheduling_tf32.yaml` (TN group: MI=32x32x16, NN group: MI=32x32x16) | Exact |
| `_128x128x64_TF32` | (128,128,64,2,1,1,F) | [16,16,32,1] | [2,2] | `custom_mainloop_scheduling_tf32.yaml` (TN group: DepthU=64, NN group: DepthU=64) | Exact |
| `_128x160x64_TF32` | (128,160,64,2,1,1,F) | [16,16,32,1] | [2,2] | `custom_mainloop_scheduling_tf32.yaml` (TN group: MI [16,16,32,1,1,4,5,2,2]→MT128x160) | Exact |
| `_128x192x32_TF32` | (128,192,32,2,1,1,F) | [16,16,32,1] | [2,2] | `custom_mainloop_scheduling_tf32.yaml` (TN group: MI [16,16,32,1,1,4,6,2,2]→MT128x192 DU32; NN group: same MI) | Exact |
| `_128x256x32_TF32` | (128,256,32,2,1,1,F) | [16,16,32,1] | [2,2] | `custom_mainloop_scheduling_tf32.yaml` (TN group: MI [16,16,32,1,1,4,8,2,2]→MT128x256) | Exact |
| `_128x64x64_TF32` | (128,64,64,2,1,1,F) | [16,16,32,1] | [2,2] | `custom_mainloop_scheduling_tf32.yaml` (TN group: MI [16,16,32,1,1,4,2,2,2]→MT128x64) | Exact |
| `_160x128x64_TF32` | (160,128,64,2,1,1,F) | [16,16,32,1] | [2,2] | `custom_mainloop_scheduling_tf32.yaml` (TN group: MI [16,16,32,1,1,5,4,2,2]→MT160x128; NN group: same) | Exact |
| `_192x128x32_TF32` | (192,128,32,2,1,1,F) | [16,16,32,1] | [2,2] | `custom_mainloop_scheduling_tf32.yaml` (TN group: MI [16,16,32,1,1,6,4,2,2]→MT192x128 DU32; LDSTrInst=1) | Exact |
| `_192x256x32_TF32` | (192,256,32,2,1,1,F) | [16,16,32,1] | [2,2] | `custom_mainloop_scheduling_tf32.yaml` (TN group: MI [16,16,32,1,1,6,8,2,2]→MT192x256 DU32; NN group same) | Exact |
| `_256x128x32_TF32` | (256,128,32,2,1,1,F) | [16,16,32,1] | [2,2] | `custom_mainloop_scheduling_tf32.yaml` (TN group: MI [16,16,32,1,1,8,4,2,2]→MT256x128 DU32; LDSTrInst=1) | Exact |
| `_256x192x32_TF32` | (256,192,32,2,1,1,F) | [16,16,32,1] | [2,2] | `custom_mainloop_scheduling_tf32.yaml` (TN group: MI [16,16,32,1,1,8,6,2,2]→MT256x192 DU32; NN group same) | Exact |
| `_256x256x32_TF32` | (256,256,32,2,1,1,F) | [16,16,32,1] | [2,2] | `custom_mainloop_scheduling_tf32.yaml` (TN group: MI [16,16,32,1,1,8,8,2,2]→MT256x256 DU32; NN group same) | Exact |
| `_64x128x64_TF32` | (64,128,64,2,1,1,F) | [16,16,32,1] | [2,2] | `custom_mainloop_scheduling_tf32.yaml` (TN group: MI [16,16,32,1,1,2,4,2,2]→MT64x128) | Exact |

### 16-bit Schedules (`dtype_predicate=is16bit`)
Most 16-bit schedules: `vector_widths=[8, 8, 8]` (GRVWA=8, GRVWB=8, LRVW=8), `matrix_inst=[16, 16, 32, 1]`, `mfma_wave_group=[2, 2]` (with exceptions noted).

| Schedule Function | TileConfig (MT0,MT1,DU,PGR,PLR,DTL,DPLB) | matrix_inst | wave_group | Matching YAMLs | Confidence |
|---|---|---|---|---|---|
| `_96x256x64_16bit` | (96,256,64,2,1,1,F) | [16,16,32,1] | [2,2] | `custom_mainloop_scheduling.yaml` — NN (MI[16,16,32,1,1,3,8,2,2]), TN (MI same, LDSTrInst=1 and 0 variants), NT | Exact |
| `_128x192x64_16bit` | (128,192,64,2,1,1,F) | [16,16,32,1] | [2,2] | `custom_mainloop_scheduling.yaml` — TN (MI[16,16,32,1,1,4,6,2,2]), TN (MI same): two groups | Exact |
| `_128x224x64_16bit` | (128,224,64,2,1,1,F) | [16,16,32,1] | [2,2] | `custom_mainloop_scheduling.yaml` — NN (MI[16,16,32,1,1,4,7,2,2]→MT128x224, LDSTrInst=1), TN (same), NT | Exact |
| `_128x256x64_16bit` | (128,256,64,2,1,1,F) | [16,16,32,1] | [2,2] | `custom_mainloop_scheduling.yaml` — NN (MI[16,16,32,1,1,4,8,2,2]→MT128x256, LDSTrInst=1), TN (same) | Exact |
| `_160x256x64_16bit` | (160,256,64,2,1,1,F) | [16,16,32,1] | [2,2] | `custom_mainloop_scheduling.yaml` — NN (MI[16,16,32,1,1,5,8,2,2]→MT160x256, LDSTrInst=1), TN (same) | Exact |
| `_192x128x64_16bit` | (192,128,64,2,1,1,F) | [16,16,32,1] | [2,2] | `custom_mainloop_scheduling.yaml` — TN (MI[16,16,32,1,1,6,4,2,2]→MT192x128, LDSTrInst=0) | Exact |
| `_192x256x64_16bit` | (192,256,64,2,1,1,F) | [16,16,32,1] | [2,2] | `custom_mainloop_scheduling.yaml` — TN (MI[16,16,32,1,1,6,8,2,2]→MT192x256), TN (MI[16,16,32,1,1,6,8,2,2] 2nd variant) | Exact |
| `_192x320x64_16bit` | (192,320,64,2,1,1,F) | [16,16,32,1] | [2,2] | `custom_mainloop_scheduling.yaml` — NN (MI[16,16,32,1,1,6,10,2,2]), TN (same, both LDSTrInst=1) | Exact |
| `_224x128x64_16bit` | (224,128,64,2,1,1,F) | [16,16,32,1] | [2,2] | `custom_mainloop_scheduling.yaml` — NN (MI[16,16,32,1,1,7,4,2,2]→MT224x128), TN (same), NT | Exact |
| `_224x256x64_16bit` | (224,256,64,2,1,1,F) | [16,16,32,1] | [2,2] | `custom_mainloop_scheduling.yaml` — NN (MI[16,16,32,1,1,7,8,2,2]→MT224x256), TN (same) | Exact |
| `_224x320x64_16bit` | (224,320,64,2,1,1,F) | [16,16,32,1] | [2,2] | `custom_mainloop_scheduling.yaml` — TN (MI[16,16,32,1,1,7,10,2,2]→MT224x320, LDSTrInst=1) | Exact |
| `_256x96x64_16bit` | (256,96,64,2,1,1,F) | [16,16,32,1] | [2,2] | `custom_mainloop_scheduling.yaml` — NN (MI[16,16,32,1,1,8,3,2,2], LDSTrInst=1), TN (same, LDSTrInst=1), NT | Exact |
| `_256x96x64_16bit_DPLB` | (256,96,64,2,1,1,**T**) | [16,16,32,1] | [2,2] | **ORPHAN** — no YAML sets `DtlPlusLdsBuf=True` (or the equivalent `1LDSBuffer`) alongside this MI config. See §4. | N/A |
| `_256x160x64_16bit` | (256,160,64,2,1,1,F) | [16,16,32,1] | [2,2] | `custom_mainloop_scheduling.yaml` — NN (MI[16,16,32,1,1,8,5,2,2]→MT256x160, LDSTrInst=1), TN (same) | Exact |
| `_256x192x64_16bit` | (256,192,64,2,1,1,F) | [16,16,32,1] | [2,2] | `custom_mainloop_scheduling.yaml` — NN (MI[16,16,32,1,1,8,6,2,2], multiple entries), TN (same); also matched by `gfx950_Cijk_Alik_Bljk_BBS_BH_UserArgs.yaml` | Exact |
| `_256x208x64_16bit` | (256,208,64,2,1,1,F) | [16,16,32,1] | [**4,1**] | `custom_mainloop_scheduling.yaml` — NN (MI[16,16,32,1,1,4,13,4,1]→MT256x208, LDSTrInst=1), TN (same) | Exact |
| `_256x224x64_16bit` | (256,224,64,2,1,1,F) | [16,16,32,1] | [2,2] | `custom_mainloop_scheduling.yaml` — NN (MI[16,16,32,1,1,8,7,2,2]→MT256x224), TN (MI[16,16,32,1,1,8,7,2,2]) | Exact |
| `_256x240x64_16bit` | (256,240,64,2,1,1,F) | [16,16,32,1] | [**4,1**] | `custom_mainloop_scheduling.yaml` — NN (MI[16,16,32,1,1,4,15,4,1]→MT256x240, LDSTrInst=1), TN (same) | Exact |
| `_256x256x64_16bit` | (256,256,64,2,1,1,F) | [16,16,32,1] | [2,2] | `custom_mainloop_scheduling.yaml` — NN (MI[16,16,32,1,1,8,8,2,2]→MT256x256) and generic 256x256 groups; TN (same) | Exact |
| `_320x192x64_16bit` | (320,192,64,2,1,1,F) | [16,16,32,1] | [2,2] | `custom_mainloop_scheduling.yaml` — NN (MI[16,16,32,1,1,10,6,2,2]→MT320x192, LDSTrInst=1), TN (same) | Exact |
| `_352x192x64_16bit` | (352,192,64,2,1,1,F) | [16,16,32,1] | [2,2] | `custom_mainloop_scheduling.yaml` — TN (MI[16,16,32,1,1,11,6,2,2]→MT352x192, LDSTrInst=1) | Exact |
| `_208x256x64_16bit` | (208,256,64,2,1,1,F) | [16,16,32,1] | [**1,4**] | `custom_mainloop_scheduling.yaml` — NN (MI[16,16,32,1,1,13,4,1,4]→MT208x256, LDSTrInst=1), TN (same) | Exact |
| `_240x256x64_16bit` | (240,256,64,2,1,1,F) | [16,16,32,1] | [**1,4**] | `custom_mainloop_scheduling.yaml` — NN (MI[16,16,32,1,1,15,4,1,4]→MT240x256, LDSTrInst=1, comprehensive ForkParams block), TN (same) | Exact |

### 8-bit Schedule (`dtype_predicate=is8bit`)

| Schedule Function | TileConfig (MT0,MT1,DU,PGR,PLR,DTL,DPLB) | matrix_inst | wave_group | Matching YAMLs | Confidence |
|---|---|---|---|---|---|
| `_256x256x128_8bit` | (256,256,128,2,**0**,1,F) | [16,16,128,1] | [2,2] | `custom_mainloop_scheduling.yaml` — F8BS TN section (MI[16,16,128,1,1,8,8,2,2], DU=128, PGR=2) | Approximate (YAML has PGR=2; TileConfig has PLR=0; see §5) |

---

## 3. Reverse Index — YAML → Schedules

### `Tensile/Tests/common/gemm/gfx950/custom_mainloop_scheduling_tf32.yaml`

All TF32 schedules are exercised here. Layout coverage per schedule:

| Schedule | TN | NN | NT |
|---|---|---|---|
| `_128x128x32_TF32` | Yes | Yes | Yes (MI=32x32x16) |
| `_128x128x32_TF32_plr1` | Yes (MI=32x32x16) | Yes | Yes (MI=32x32x16) |
| `_128x128x64_TF32` | Yes | Yes | — |
| `_128x160x64_TF32` | Yes | — | — |
| `_128x192x32_TF32` | Yes | Yes | — |
| `_128x256x32_TF32` | Yes | — | — |
| `_128x64x64_TF32` | Yes | — | — |
| `_160x128x64_TF32` | Yes | Yes | — |
| `_192x128x32_TF32` | Yes (LDSTr) | — | — |
| `_192x256x32_TF32` | Yes | Yes | — |
| `_256x128x32_TF32` | Yes (LDSTr) | — | — |
| `_256x192x32_TF32` | Yes | Yes | — |
| `_256x256x32_TF32` | Yes | Yes | — |
| `_64x128x64_TF32` | Yes | — | — |

### `Tensile/Tests/common/gemm/gfx950/custom_mainloop_scheduling.yaml`

All 16-bit and 8-bit schedules (except `_256x96x64_16bit_DPLB`) are exercised here.
Multiple data-type layouts are covered: BBS (NN, TN, TT, NT), HHS (NN, TT, TN), F8BS (TN).

Notable multi-YAML entries for individual schedules:
- `_256x192x64_16bit`: appears in NN (multiple groups), TN (multiple groups, various LdsPad settings) — the most multiply-covered.
- `_96x256x64_16bit`: NN, TN, NT layouts.
- `_96x256x64_16bit` (DPLB=False): covered; `_256x96x64_16bit_DPLB` (DPLB=True): **NOT covered**.

### `Tensile/Components/example.yaml`

Sets `UseCustomMainLoopSchedule: 1` with `MatrixInstruction: [16, 16, 32, 1, 1, 5, 4, 2, 2]` (→ MT0=160, MT1=128) and `DepthU: [64]` — this is `_160x128x64_TF32` (approximate, since it's TF32/XF32 DataType=S+F32XdlMathOp=X).

Also has a commented-out line for `[16, 16, 32, 1, 1, 8, 2, 1, 4]` (never active).

### `tests/configs/mixed_configs/gfx950_Cijk_Alik_Bljk_BBS_BH_UserArgs.yaml`

Library-logic (already-solved solution, not a benchmark YAML). Contains one kernel entry for `_256x192x64_16bit`:
- `MacroTile0: 256`, `MacroTile1: 192`, `DepthU: 64`, `MatrixInstruction: [16, 16, 32, 1]`, `MIWaveGroup: [4, 1]` — however, `MIWaveGroup: [4, 1]` does NOT match `_256x192x64_16bit` which expects `mfma_wave_group=[2, 2]`. This kernel instead would match `_256x208x64_16bit` or `_256x240x64_16bit` (which have non-[2,2] wave groups), but the MT1=192 doesn't match those. This is an **ambiguity** — see §5.

---

## 4. Orphans

### Orphan Schedules (no matching YAML)

#### `_256x96x64_16bit_DPLB`
- **TileConfig:** `(256, 96, 64, 2, 1, 1, True, 0, 0, isa=(9,5,0))`
- `dtl_plus_lds_buf=True` means it expects `kernel["DtlPlusLdsBuf"] == True`.
- In Tensile YAML terms, this would require `1LDSBuffer: [1]` (enabling the DTL+LDS-buffer mode) combined with the 256x96x64 MI config.
- All YAML groups for 256x96x64 use `1LDSBuffer: [0]`. No YAML exercises the DPLB variant.
- **Status:** Schedule function exists, is registered, but is unreachable with any current YAML config.

### YAMLs That Set `UseCustomMainLoopSchedule: 1` But May Not Match Any Specific Schedule

#### `Tensile/Tests/common/gemm/gfx950/general_wgm.yaml`
- Uses `UseCustomMainLoopSchedule: [0]` (disabled, no CMS invoked).
- Not an orphan YAML — just not CMS-flavoured.

#### `Tensile/Components/example.yaml`
- Sets `UseCustomMainLoopSchedule: [1]`.
- MI = `[16, 16, 32, 1, 1, 5, 4, 2, 2]` → MT0=5×2×16=160, MT1=4×2×16=128, DU=64. Matches `_160x128x64_TF32`.
- DataType=S + F32XdlMathOp=X qualifies as TF32 (`isTF32` check uses `UseF32XEmulation`). This is a development YAML, not a CI test.

---

## 5. Ambiguities / Open Questions

### A. `_256x256x128_8bit`: PLR mismatch in YAML
- **TileConfig:** `PLR=0` (PrefetchLocalRead=0).
- **YAML (custom_mainloop_scheduling.yaml, F8BS TN section):** `PrefetchLocalRead: [1]`. This would make TileConfig not match (PLR field differs). The YAML explicitly sets `PrefetchGlobalRead: [2]` and `PrefetchLocalRead: [1]`.
- **Confidence: Approximate.** The YAML problem size range (`17–750 × 2–400, DU=128`) could produce 256×256×128 kernels, but PLR=1 in the YAML means the CMS dispatcher would fail the TileConfig equality check at the PLR field. Either: (a) the YAML is wrong and needs `PrefetchLocalRead: [0]`, (b) the TileConfig PLR=0 is wrong, or (c) there is a different YAML for 8-bit that was not found. **This is the most significant discrepancy and should be investigated.**

### B. `gfx950_Cijk_Alik_Bljk_BBS_BH_UserArgs.yaml` MIWaveGroup mismatch
- Contains `MIWaveGroup: [4, 1]` with `MacroTile0: 256`, `MacroTile1: 192`. No gfx950 CMS schedule is registered for TileConfig(256, 192, 64, ...) with `mfma_wave_group=[4, 1]`. The schedule `_256x192x64_16bit` has `mfma_wave_group=[2, 2]`.
- This library-logic file describes a **pre-solved kernel that does NOT go through CMS dispatch** (it's a concrete kernel description, not a search YAML). CMS applies at kernel-generation time. This file does not cause CMS to be invoked.

### C. NT layout coverage for TF32 is thin
- Only `_128x128x32_TF32` and `_128x128x32_TF32_plr1` appear in the NT section of `custom_mainloop_scheduling_tf32.yaml` (MI=32x32x16, NT, `UseCustomMainLoopSchedule: [1]`). Most TF32 schedules have no NT coverage.

### D. Some YAML groups use `UseCustomMainLoopSchedule: [0, 1]`
- Several groups in `custom_mainloop_scheduling.yaml` set `UseCustomMainLoopSchedule: [0, 1]`, meaning both CMS and non-CMS kernels are benchmarked. These groups include: BBS NN 256×256×64 (MI=8,8 and 6,8 entries), HHS NN, BBS/HHS TT, HHS TN, F8BS TN, BBS NT generic 256×256×64. The CMS=1 fork of these groups still exercises the schedule, but the CMS=0 fork is a control/comparison. These are marked as "Exact" since the CMS=1 fork is present.

### E. Schedules not covered by any explicitly CMS=1 YAML with matching DepthU/PGR combination
- `_352x192x64_16bit`: Only in TN (BBS TN section). No NN or NT coverage in any YAML.
- `_192x320x64_16bit`: Only in NN and TN (BBS sections). No HHS coverage.
- `_224x320x64_16bit`: Only TN coverage; MI requires `[16,16,32,1,1,7,10,2,2]`.

### F. `custom_mainloop_scheduling.yaml` 2150+ range
The file continues past line 2150 (file was truncated in reading at ~2150). There may be additional NT entries for more 16-bit schedules. The confirmed NT entries seen: 96x256, 224x128, 128x224, generic 256x256 BBS/HHS NT. There may be additional NT entries for 192x256, 256x192, etc. not captured in this audit.

---

## 6. File Paths Reference

| File | Role |
|---|---|
| `Tensile/Components/CustomSchedule/gfx950/_*.py` | CMS schedule function definitions |
| `Tensile/Components/CustomSchedule/shared.py` | `TileConfig` dataclass definition |
| `Tensile/Components/CustomSchedule/dispatch.py` | Dispatcher logic; `TileConfig` matching against kernel params |
| `Tensile/Tests/common/gemm/gfx950/custom_mainloop_scheduling.yaml` | Primary test YAML for 16-bit + 8-bit CMS schedules |
| `Tensile/Tests/common/gemm/gfx950/custom_mainloop_scheduling_tf32.yaml` | Primary test YAML for TF32 CMS schedules |
| `Tensile/Components/example.yaml` | Dev/exploration YAML; not a CI test |
| `tests/configs/mixed_configs/gfx950_Cijk_Alik_Bljk_BBS_BH_UserArgs.yaml` | Library-logic file; not a CMS-invocation config |
| `Tensile/Tests/common/gemm/gfx950/general_wgm.yaml` | CMS disabled (`UseCustomMainLoopSchedule: [0]`); not relevant |

---

## Appendix: TileConfig Quick-Reference for YAML Migration

When migrating a schedule's kernel-flag declarations to YAML, locate the YAML group(s) that match **all** of:
1. `MatrixInstruction: [MIm, MIn, MIk, MIb, InnerUnroll, MIWaveTile0, MIWaveTile1, MIWaveGroup0, MIWaveGroup1]` where `MIm*MIWaveTile0*MIWaveGroup0 == MT0` and `MIn*MIWaveTile1*MIWaveGroup1 == MT1`.
2. `DepthU == TileConfig.depth_u`.
3. `PrefetchGlobalRead == TileConfig.prefetch_global_read`.
4. `PrefetchLocalRead == TileConfig.prefetch_local_read`.
5. `DirectToLds == TileConfig.direct_to_lds`.
6. `1LDSBuffer == (1 if TileConfig.dtl_plus_lds_buf else 0)`.
7. `GlobalReadVectorWidthA == decorator.vector_widths[0]`, same for B and LocalReadVectorWidth.
8. `UseCustomMainLoopSchedule: [1]` is set.
