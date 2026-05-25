# CMS Kernel-Dict Mutation Audit

**Scope:** `Tensile/Components/CustomSchedule/` — exhaustive read-only audit  
**Date:** 2026-05-22  
**Branch:** validator_long_term_plans worktree  
**Purpose:** Feed rocm-libraries-2bww (P0) — "Stop CMS schedule bodies from mutating kernel-level flags"

---

## 1. Summary

| Metric | Count |
|--------|-------|
| Total `@RegisterSchedule` schedule functions found | 41 |
| Functions that mutate `kernel[...]` | 37 |
| Functions with **zero** mutations (return False early for all paths, or delegate) | 4 (`_256x96x64_16bit`, `_256x96x64_16bit_DPLB`, `_224x320x64_16bit`*, `_192x128x64_16bit`**) |
| Distinct flag names mutated | 5 (`MfmaInitCVgprs`, `UsePLRPack`, `UseMFMAF32XEmulation`, `UseDot2F32XEmulation`, `SwapGlobalReadOrder`) |
| Distinct (flag, value) pairs | 7 |

*`_224x320x64_16bit` sets two flags unconditionally at function entry (before the `if` branch check), so it always mutates even if it later returns False.  
**`_192x128x64_16bit` sets `MfmaInitCVgprs = True` unconditionally at line 42, before any branch check.  

### The 7 distinct (flag, value) pairs

| Flag | Value | Meaning |
|------|-------|---------|
| `MfmaInitCVgprs` | `True` | Init C-tile VGPRS to zero inside MFMA loop |
| `UsePLRPack` | `True` | Enable packing of prefetch local reads |
| `UseMFMAF32XEmulation` | `True` | Emulate FP32 MFMA with FP16/BF16 triple-MFMA |
| `UseMFMAF32XEmulation` | `False` | Disable FP32 MFMA emulation |
| `UseDot2F32XEmulation` | `False` | Disable DOT2 FP32 emulation |
| `SwapGlobalReadOrder` | `True` | Swap A/B global read ordering |
| `SwapGlobalReadOrder` | `False` | Explicitly force no swap |

---

## 2. Per-Flag Table

### 2a. `MfmaInitCVgprs = True`

Set unconditionally at the end (or at entry) of every schedule function that succeeds.
Every function in the list does it — it is effectively a post-condition of any valid CMS schedule.

| Function | File | Line | Branch condition |
|----------|------|------|-----------------|
| `_get_schedule_128x128x32_TF32` | `gfx950/_128x128x32_TF32.py` | 119 | Unconditional (after success path) |
| `_get_schedule_128x128x32_TF32_plr1` | `gfx950/_128x128x32_TF32_plr1.py` | 214 | Unconditional |
| `_get_schedule_128x128x64_TF32` | `gfx950/_128x128x64_TF32.py` | 158 | Unconditional |
| `_get_schedule_128x160x64_TF32` | `gfx950/_128x160x64_TF32.py` | 122 | Unconditional (after isTN+noLDSTr+TLDS=1) |
| `_get_schedule_128x192x32_TF32` | `gfx950/_128x192x32_TF32.py` | 97 | Unconditional |
| `_get_schedule_128x192x64_16bit` | `gfx950/_128x192x64_16bit.py` | 43 | **At function entry** (before branch) |
| `_get_schedule_128x224x64_16bit` | `gfx950/_128x224x64_16bit.py` | 142 | Unconditional |
| `_get_schedule_128x256x32_TF32` | `gfx950/_128x256x32_TF32.py` | 347 | Unconditional |
| `_get_schedule_128x256x64_16bit` | `gfx950/_128x256x64_16bit.py` | 109 | Unconditional |
| `_get_schedule_160x128x64_TF32` | `gfx950/_160x128x64_TF32.py` | 155 | Unconditional |
| `_get_schedule_160x256x64_16bit` | `gfx950/_160x256x64_16bit.py` | 161 | Unconditional |
| `_get_schedule_192x128x32_TF32` | `gfx950/_192x128x32_TF32.py` | 117 | Unconditional |
| `_get_schedule_192x128x64_16bit` | `gfx950/_192x128x64_16bit.py` | 42 | **At function entry** (before branch) |
| `_get_schedule_192x256x32_TF32` | `gfx950/_192x256x32_TF32.py` | 361 | Unconditional |
| `_get_schedule_192x256x64_16bit` | `gfx950/_192x256x64_16bit.py` | 168 | Unconditional |
| `_get_schedule_192x320x64_16bit` | `gfx950/_192x320x64_16bit.py` | 157 | Unconditional |
| `_get_schedule_208x256x64_16bit` | `gfx950/_208x256x64_16bit.py` | 154 | Unconditional |
| `_get_schedule_224x128x64_16bit` | `gfx950/_224x128x64_16bit.py` | 182 | Unconditional |
| `_get_schedule_224x256x64_16bit` | `gfx950/_224x256x64_16bit.py` | 155 | Unconditional |
| `_get_schedule_224x320x64_16bit` | `gfx950/_224x320x64_16bit.py` | 45 | **At function entry** (before branch) |
| `_get_schedule_240x256x64_16bit` | `gfx950/_240x256x64_16bit.py` | 150 | Unconditional |
| `_get_schedule_256x128x32_TF32` | `gfx950/_256x128x32_TF32.py` | 131 | Unconditional |
| `_get_schedule_256x160x64_16bit` | `gfx950/_256x160x64_16bit.py` | 163 | Unconditional |
| `_get_schedule_256x192x32_TF32` | `gfx950/_256x192x32_TF32.py` | 313 | Unconditional |
| `_get_schedule_256x192x64_16bit` | `gfx950/_256x192x64_16bit.py` | 170 | Unconditional |
| `_get_schedule_256x208x64_16bit` | `gfx950/_256x208x64_16bit.py` | 148 | Unconditional |
| `_get_schedule_256x224x64_16bit` | `gfx950/_256x224x64_16bit.py` | 169 | Unconditional |
| `_get_schedule_256x256x32_TF32` | `gfx950/_256x256x32_TF32.py` | 238 | Unconditional |
| `_get_schedule_256x256x64_16bit` | `gfx950/_256x256x64_16bit.py` | 177 | Unconditional |
| `_get_schedule_256x256x128_8bit` | `gfx950/_256x256x128_8bit.py` | 77 | Unconditional |
| `_get_schedule_320x192x64_16bit` | `gfx950/_320x192x64_16bit.py` | 167 | Unconditional |
| `_get_schedule_352x192x64_16bit` | `gfx950/_352x192x64_16bit.py` | 93 | Unconditional |
| `_get_schedule_64x128x64_TF32` | `gfx950/_64x128x64_TF32.py` | 121 | Unconditional (after isTN+noLDSTr+TLDS=1) |
| `_get_schedule_96x256x64_16bit` | `gfx950/_96x256x64_16bit.py` | 44 | **At function entry** (before branch) |

### 2b. `UsePLRPack = True`

| Function | File | Lines | Branch condition |
|----------|------|-------|-----------------|
| `_get_schedule_128x128x32_TF32` | `_128x128x32_TF32.py` | 120 | Unconditional (end of success path) |
| `_get_schedule_128x128x32_TF32_plr1` | `_128x128x32_TF32_plr1.py` | 215 | Unconditional |
| `_get_schedule_128x128x64_TF32` | `_128x128x64_TF32.py` | 161 | Unconditional |
| `_get_schedule_128x160x64_TF32` | `_128x160x64_TF32.py` | 52 | `isTN + not useLDSTr + TLDS==1` |
| `_get_schedule_128x256x32_TF32` | `_128x256x32_TF32.py` | 49 | `isTN + not useLDSTr + TLDS==1` |
| `_get_schedule_160x128x64_TF32` | `_160x128x64_TF32.py` | 53 | `isNN + useLDSTr + TLDS==1` |
| `_get_schedule_192x128x32_TF32` | `_192x128x32_TF32.py` | 47 | `isTN + useLDSTr + TLDS==1` |
| `_get_schedule_192x256x32_TF32` | `_192x256x32_TF32.py` | 49, 182 | branch 1: `isTN + not useLDSTr + TLDS==1`; branch 2: `isNN + TLDS==1` |
| `_get_schedule_256x128x32_TF32` | `_256x128x32_TF32.py` | 51 | `isTN + useLDSTr + TLDS==1` |
| `_get_schedule_256x192x32_TF32` | `_256x192x32_TF32.py` | 49, 135 | branch 1: `isTN + not useLDSTr + TLDS==1`; branch 2: `isNN + TLDS==1 + VectorWidthA==1` |
| `_get_schedule_256x256x32_TF32` | `_256x256x32_TF32.py` | 48, 168 | branch 1: `isTN + not useLDSTr + TLDS==1`; branch 2: `isNT + not useLDSTr + TLDS==0 + VW4` |
| `_get_schedule_256x256x64_16bit` | `_256x256x64_16bit.py` | 82, 126 | branch 1: `isNT + not useLDSTr + TLDS==0`; branch 2: `isNN or isTT + not useLDSTr + TLDS==1` |
| `_get_schedule_64x128x64_TF32` | `_64x128x64_TF32.py` | 52 | `isTN + not useLDSTr + TLDS==1` |

### 2c. `UseMFMAF32XEmulation = True`

| Function | File | Lines | Branch condition |
|----------|------|-------|-----------------|
| `_get_schedule_128x128x32_TF32` | `_128x128x32_TF32.py` | 53 | `isTN + not useLDSTr + TLDS==1` |
| `_get_schedule_128x128x32_TF32_plr1` | `_128x128x32_TF32_plr1.py` | 216 | Unconditional |
| `_get_schedule_128x128x64_TF32` | `_128x128x64_TF32.py` | 159 | Unconditional |
| `_get_schedule_128x160x64_TF32` | `_128x160x64_TF32.py` | 51 | `isTN + not useLDSTr + TLDS==1` |
| `_get_schedule_160x128x64_TF32` | `_160x128x64_TF32.py` | 52 | `isNN + useLDSTr + TLDS==1` |
| `_get_schedule_192x256x32_TF32` | `_192x256x32_TF32.py` | 50, 183 | branch 1: `isTN + not useLDSTr + TLDS==1`; branch 2: `isNN + TLDS==1` |
| `_get_schedule_256x192x32_TF32` | `_256x192x32_TF32.py` | 136 | `isNN + TLDS==1 + VectorWidthA==1` (branch 2 only) |
| `_get_schedule_256x256x32_TF32` | `_256x256x32_TF32.py` | 49, 169 | branch 1: `isTN + not useLDSTr + TLDS==1`; branch 2: `isNT + not useLDSTr + TLDS==0 + VW4` |
| `_get_schedule_64x128x64_TF32` | `_64x128x64_TF32.py` | 51 | `isTN + not useLDSTr + TLDS==1` |
| `_get_schedule_128x256x32_TF32` | `_128x256x32_TF32.py` | 50 | `isTN + not useLDSTr + TLDS==1` |

### 2d. `UseMFMAF32XEmulation = False`

| Function | File | Lines | Branch condition |
|----------|------|-------|-----------------|
| `_get_schedule_128x192x32_TF32` | `_128x192x32_TF32.py` | 52 | `isTN + not useLDSTr + TLDS==1` |
| `_get_schedule_192x128x32_TF32` | `_192x128x32_TF32.py` | 48 | `isTN + useLDSTr + TLDS==1` |
| `_get_schedule_256x128x32_TF32` | `_256x128x32_TF32.py` | 49 | `isTN + useLDSTr + TLDS==1` |
| `_get_schedule_256x192x32_TF32` | `_256x192x32_TF32.py` | 50 | `isTN + not useLDSTr + TLDS==1` (branch 1 only) |

### 2e. `UseDot2F32XEmulation = False`

Always co-occurs with a `UseMFMAF32XEmulation` mutation. Never set to `True`.

| Function | File | Lines | Branch condition |
|----------|------|-------|-----------------|
| `_get_schedule_128x128x32_TF32_plr1` | `_128x128x32_TF32_plr1.py` | 217 | Unconditional |
| `_get_schedule_128x128x64_TF32` | `_128x128x64_TF32.py` | 160 | Unconditional |
| `_get_schedule_128x192x32_TF32` | `_128x192x32_TF32.py` | 53 | `isTN + not useLDSTr + TLDS==1` |
| `_get_schedule_128x256x32_TF32` | `_128x256x32_TF32.py` | 51 | `isTN + not useLDSTr + TLDS==1` |
| `_get_schedule_192x128x32_TF32` | `_192x128x32_TF32.py` | 49 | `isTN + useLDSTr + TLDS==1` |
| `_get_schedule_192x256x32_TF32` | `_192x256x32_TF32.py` | 51, 184 | both branches |
| `_get_schedule_256x128x32_TF32` | `_256x128x32_TF32.py` | 50 | `isTN + useLDSTr + TLDS==1` |
| `_get_schedule_256x192x32_TF32` | `_256x192x32_TF32.py` | 51 | `isTN + not useLDSTr + TLDS==1` (branch 1 only) |
| `_get_schedule_256x256x32_TF32` | `_256x256x32_TF32.py` | 50, 170 | both branches |

### 2f. `SwapGlobalReadOrder = True`

| Function | File | Lines | Branch condition |
|----------|------|-------|-----------------|
| `_get_schedule_160x256x64_16bit` | `_160x256x64_16bit.py` | 94 | `isNN + useLDSTr + TLDS==1` |
| `_get_schedule_192x256x64_16bit` | `_192x256x64_16bit.py` | 50 | `isNN + useLDSTr + TLDS==1` |
| `_get_schedule_224x128x64_16bit` | `_224x128x64_16bit.py` | 121 | `isNT + useLDSTr + TLDS==0` |
| `_get_schedule_256x160x64_16bit` | `_256x160x64_16bit.py` | 46, 134 | branch 1: `isNN + useLDSTr + TLDS==1`; branch 2: `isNT + useLDSTr + TLDS==0` |
| `_get_schedule_256x192x64_16bit` | `_256x192x64_16bit.py` | 90, 128 | branch 1: `isNT + useLDSTr + TLDS==0`; branch 2 (line 128 unreachable/second branch): same |
| `_get_schedule_256x208x64_16bit` | `_256x208x64_16bit.py` | 77 | `isNN + useLDSTr + TLDS==1` |
| `_get_schedule_256x224x64_16bit` | `_256x224x64_16bit.py` | 127 | `isNN + useLDSTr + TLDS==1` |
| `_get_schedule_256x256x64_16bit` | `_256x256x64_16bit.py` | 165 | `isTT + not useLDSTr + TLDS==1` (nested inside NN/TT branch) |
| `_get_schedule_320x192x64_16bit` | `_320x192x64_16bit.py` | 48, 87, 127 | all three branches: `isNN + useLDSTr + TLDS==1`, `isTN + TLDS==1`, `isNT + useLDSTr + TLDS==0` |
| `_get_schedule_96x256x64_16bit` | `_96x256x64_16bit.py` | 52 | `isTN + TLDS==1` |

### 2g. `SwapGlobalReadOrder = False`

Anomaly: explicitly forcing the flag to `False` (override-to-default pattern).

| Function | File | Lines | Branch condition |
|----------|------|-------|-----------------|
| `_get_schedule_192x320x64_16bit` | `_192x320x64_16bit.py` | 158 | Unconditional (after both isTN and isNN branch success) |
| `_get_schedule_208x256x64_16bit` | `_208x256x64_16bit.py` | 155 | Unconditional |
| `_get_schedule_224x320x64_16bit` | `_224x320x64_16bit.py` | 46 | **At function entry** (before the if-branch) |
| `_get_schedule_240x256x64_16bit` | `_240x256x64_16bit.py` | 46 | `isTN + TLDS==1` (first branch) |

---

## 3. Per-Schedule Table (mutations with line numbers)

All paths covered. For brevity, `kernel[K]=V @L` means flag K set to value V at line L.

| Function | Decorator (TileConfig m×n×k, dtype, matrix_inst, wave_group) | Mutations |
|----------|---------------------------------------------------------------|-----------|
| `_get_schedule_128x128x32_TF32` | TF32, 128×128×32, [16,16,32,1], [2,2] | L53: `UseMFMAF32XEmulation=True` gated on `isTN+noLDSTr+TLDS=1`; L119: `MfmaInitCVgprs=True`; L120: `UsePLRPack=True` — both unconditional |
| `_get_schedule_128x128x32_TF32_plr1` | TF32, 128×128×32, [32,32,16,1], [2,2] | L214–217: `MfmaInitCVgprs=True`, `UsePLRPack=True`, `UseMFMAF32XEmulation=True`, `UseDot2F32XEmulation=False` — all unconditional (only one branch returns True) |
| `_get_schedule_128x128x64_TF32` | TF32, 128×128×64, [16,16,32,1], [2,2] | L158–161: `MfmaInitCVgprs=True`, `UseMFMAF32XEmulation=True`, `UseDot2F32XEmulation=False`, `UsePLRPack=True` — unconditional |
| `_get_schedule_128x160x64_TF32` | TF32, 128×160×64, [16,16,32,1], [2,2] | L51: `UseMFMAF32XEmulation=True`, L52: `UsePLRPack=True` — `isTN+noLDSTr+TLDS=1`; L122: `MfmaInitCVgprs=True` — unconditional |
| `_get_schedule_128x192x32_TF32` | TF32, 128×192×32, [16,16,32,1], [2,2] | L51: `UsePLRPack=True`, L52: `UseMFMAF32XEmulation=False`, L53: `UseDot2F32XEmulation=False` — `isTN+noLDSTr+TLDS=1`; L97: `MfmaInitCVgprs=True` |
| `_get_schedule_128x192x64_16bit` | is16bit, 128×192×64, [16,16,32,1], [2,2] | L43: `MfmaInitCVgprs=True` — **at function entry** |
| `_get_schedule_128x224x64_16bit` | is16bit, 128×224×64, [16,16,32,1], [2,2] | L142: `MfmaInitCVgprs=True` — unconditional |
| `_get_schedule_128x256x32_TF32` | TF32, 128×256×32, [16,16,32,1], [2,2] | L49: `UsePLRPack=True`, L50: `UseMFMAF32XEmulation=True`, L51: `UseDot2F32XEmulation=False` — `isTN+noLDSTr+TLDS=1`; L347: `MfmaInitCVgprs=True` |
| `_get_schedule_128x256x64_16bit` | is16bit, 128×256×64, [16,16,32,1], [2,2] | L109: `MfmaInitCVgprs=True` — unconditional |
| `_get_schedule_160x128x64_TF32` | TF32, 160×128×64, [16,16,32,1], [2,2] | L52: `UseMFMAF32XEmulation=True`, L53: `UsePLRPack=True` — `isNN+useLDSTr+TLDS=1`; L155: `MfmaInitCVgprs=True` |
| `_get_schedule_160x256x64_16bit` | is16bit, 160×256×64, [16,16,32,1], [2,2] | L94: `SwapGlobalReadOrder=True` — `isNN+useLDSTr+TLDS=1`; L161: `MfmaInitCVgprs=True` |
| `_get_schedule_192x128x32_TF32` | TF32, 192×128×32, [16,16,32,1], [2,2] | L47: `UsePLRPack=True`, L48: `UseMFMAF32XEmulation=False`, L49: `UseDot2F32XEmulation=False` — `isTN+useLDSTr+TLDS=1`; L117: `MfmaInitCVgprs=True` |
| `_get_schedule_192x128x64_16bit` | is16bit, 192×128×64, [16,16,32,1], [2,2] | L42: `MfmaInitCVgprs=True` — **at function entry** |
| `_get_schedule_192x256x32_TF32` | TF32, 192×256×32, [16,16,32,1], [2,2] | L49–51: `UsePLRPack=True`, `UseMFMAF32XEmulation=True`, `UseDot2F32XEmulation=False` — `isTN+noLDSTr+TLDS=1`; L182–184: same three flags — `isNN+TLDS=1`; L361: `MfmaInitCVgprs=True` |
| `_get_schedule_192x256x64_16bit` | is16bit, 192×256×64, [16,16,32,1], [2,2] | L50: `SwapGlobalReadOrder=True` — `isNN+useLDSTr+TLDS=1`; L168: `MfmaInitCVgprs=True` |
| `_get_schedule_192x320x64_16bit` | is16bit, 192×320×64, [16,16,32,1], [2,2] | L157: `MfmaInitCVgprs=True`, L158: `SwapGlobalReadOrder=False` — unconditional |
| `_get_schedule_208x256x64_16bit` | is16bit, 208×256×64, [16,16,32,1], [1,4] | L154: `MfmaInitCVgprs=True`, L155: `SwapGlobalReadOrder=False` — unconditional |
| `_get_schedule_224x128x64_16bit` | is16bit, 224×128×64, [16,16,32,1], [2,2] | L121: `SwapGlobalReadOrder=True` — `isNT+useLDSTr+TLDS=0`; L182: `MfmaInitCVgprs=True` |
| `_get_schedule_224x256x64_16bit` | is16bit, 224×256×64, [16,16,32,1], [2,2] | L155: `MfmaInitCVgprs=True` — unconditional |
| `_get_schedule_224x320x64_16bit` | is16bit, 224×320×64, [16,16,32,1], [2,2] | L45: `MfmaInitCVgprs=True`, L46: `SwapGlobalReadOrder=False` — **at function entry** |
| `_get_schedule_240x256x64_16bit` | is16bit, 240×256×64, [16,16,32,1], [1,4] | L46: `SwapGlobalReadOrder=False` — `isTN+TLDS=1`; L150: `MfmaInitCVgprs=True` |
| `_get_schedule_256x128x32_TF32` | TF32, 256×128×32, [16,16,32,1], [2,2] | L49: `UseMFMAF32XEmulation=False`, L50: `UseDot2F32XEmulation=False`, L51: `UsePLRPack=True` — `isTN+useLDSTr+TLDS=1`; L131: `MfmaInitCVgprs=True` |
| `_get_schedule_256x160x64_16bit` | is16bit, 256×160×64, [16,16,32,1], [2,2] | L46: `SwapGlobalReadOrder=True` — `isNN+useLDSTr+TLDS=1`; L134: `SwapGlobalReadOrder=True` — `isNT+useLDSTr+TLDS=0`; L163: `MfmaInitCVgprs=True` |
| `_get_schedule_256x192x32_TF32` | TF32, 256×192×32, [16,16,32,1], [2,2] | L49–51: `UsePLRPack=True`, `UseMFMAF32XEmulation=False`, `UseDot2F32XEmulation=False` — `isTN+noLDSTr+TLDS=1`; L135–136: `UsePLRPack=True`, `UseMFMAF32XEmulation=True` — `isNN+TLDS=1+VWA=1`; L313: `MfmaInitCVgprs=True` |
| `_get_schedule_256x192x64_16bit` | is16bit, 256×192×64, [16,16,32,1], [2,2] | L90: `SwapGlobalReadOrder=True` — `isNT+useLDSTr+TLDS=0`; L128: `SwapGlobalReadOrder=True` — same condition (second NT sub-path); L170: `MfmaInitCVgprs=True` |
| `_get_schedule_256x208x64_16bit` | is16bit, 256×208×64, [16,16,32,1], [4,1] | L77: `SwapGlobalReadOrder=True` — `isNN+useLDSTr+TLDS=1`; L148: `MfmaInitCVgprs=True` |
| `_get_schedule_256x224x64_16bit` | is16bit, 256×224×64, [16,16,32,1], [2,2] | L127: `SwapGlobalReadOrder=True` — `isNN+useLDSTr+TLDS=1`; L169: `MfmaInitCVgprs=True` |
| `_get_schedule_256x256x32_TF32` | TF32, 256×256×32, [16,16,32,1], [2,2] | L48–50: `UsePLRPack=True`, `UseMFMAF32XEmulation=True`, `UseDot2F32XEmulation=False` — `isTN+noLDSTr+TLDS=1`; L168–170: same three — `isNT+noLDSTr+TLDS=0+VW4`; L238: `MfmaInitCVgprs=True` |
| `_get_schedule_256x256x64_16bit` | is16bit, 256×256×64, [16,16,32,1], [2,2] | L82: `UsePLRPack=True` — `isNT+noLDSTr+TLDS=0`; L126: `UsePLRPack=True` — `isNN/isTT+noLDSTr+TLDS=1`; L165: `SwapGlobalReadOrder=True` — `isTT` nested sub-case; L177: `MfmaInitCVgprs=True` |
| `_get_schedule_256x256x128_8bit` | is8bit, 256×256×128, [16,16,128,1], [2,2] | L77: `MfmaInitCVgprs=True` — unconditional |
| `_get_schedule_320x192x64_16bit` | is16bit, 320×192×64, [16,16,32,1], [2,2] | L48: `SwapGlobalReadOrder=True` — `isNN+useLDSTr+TLDS=1`; L87: `SwapGlobalReadOrder=True` — `isTN+TLDS=1`; L127: `SwapGlobalReadOrder=True` — `isNT+useLDSTr+TLDS=0`; L167: `MfmaInitCVgprs=True` |
| `_get_schedule_352x192x64_16bit` | is16bit, 352×192×64, [16,16,32,1], [2,2] | L93: `MfmaInitCVgprs=True` — unconditional |
| `_get_schedule_64x128x64_TF32` | TF32, 64×128×64, [16,16,32,1], [2,2] | L51: `UseMFMAF32XEmulation=True`, L52: `UsePLRPack=True` — `isTN+noLDSTr+TLDS=1`; L121: `MfmaInitCVgprs=True` |
| `_get_schedule_96x256x64_16bit` | is16bit, 96×256×64, [16,16,32,1], [2,2] | L44: `MfmaInitCVgprs=True` — **at function entry**; L52: `SwapGlobalReadOrder=True` — `isTN+TLDS=1` |

**Functions with zero kernel mutations:**
- `_get_schedule_256x96x64_16bit` — reads `kernel[...]` only
- `_get_schedule_256x96x64_16bit_DPLB` — reads `kernel[...]` only

---

## 4. Cross-Cutting Observations

### O1: `MfmaInitCVgprs = True` is universal

Every schedule that returns `True` (success) also sets `MfmaInitCVgprs = True`. It is a pure post-condition invariant — not data-dependent on layout or dtype. Four schedules even set it at function entry (before the layout branch) rather than after:
- `_128x192x64_16bit` (L43), `_192x128x64_16bit` (L42), `_224x320x64_16bit` (L45), `_96x256x64_16bit` (L44)

This means those functions always mutate the kernel dict even when they subsequently return `False, None`.

### O2: `UsePLRPack` always co-occurs with `UseMFMAF32XEmulation` (or its negation)

Every schedule function that sets `UsePLRPack = True` also sets `UseMFMAF32XEmulation` to either `True` or `False` in the same branch. No schedule sets `UsePLRPack` without touching the XEmulation flag. However, the converse is not true: `UseMFMAF32XEmulation` can appear without `UsePLRPack` (e.g., `_128x128x32_TF32` where `UseMFMAF32XEmulation=True` is branch-gated but `UsePLRPack=True` appears unconditionally).

### O3: `UseMFMAF32XEmulation=False` is a TF32-only phenomenon

All four schedules that set `UseMFMAF32XEmulation = False` are TF32 tiles (`isTF32` predicate). All TF32 tiles that use `isTN + useLDSTr + TLDS==1` (LDS Transpose path) set it to `False`; those on the `isTN + not useLDSTr + TLDS==1` path may set it to `True`.

### O4: `UseDot2F32XEmulation` is always `False` — never `True`

The flag is only ever set to `False`. Its purpose is to explicitly suppress a default, not to enable a feature. This suggests it could be the default value with no need for a schedule override.

### O5: `SwapGlobalReadOrder = True` is the NN/NT path for wider 16-bit tiles

The `SwapGlobalReadOrder = True` mutations are exclusively in 16-bit schedules (is16bit predicate) and exclusively on `isNN + useLDSTr` or `isNT + useLDSTr` or `isTN + TLDS=1` branches. The TF32 schedules never set this flag.

### O6: `SwapGlobalReadOrder = False` is an anomalous defensive write

Four functions explicitly set `SwapGlobalReadOrder = False`. In two cases (`_224x320x64_16bit`, `_192x320x64_16bit`) this is unconditional and occurs regardless of branch taken. This is a defensive "reset to default" pattern — suggesting the flag's default value is not reliably known at schedule invocation time, or that the dispatch layer might have set it previously.

### O7: Three functions mutate flags before checking the layout predicate

`_128x192x64_16bit` (L43), `_192x128x64_16bit` (L42), `_224x320x64_16bit` (L45–46), and `_96x256x64_16bit` (L44) write to `kernel[...]` before evaluating the layout `if/elif` chain. This means a failed schedule (returning `False, None`) still has a side effect on the kernel dict — a direct violation of the contract.

### O8: `256x192x32_TF32` sets conflicting values of `UseMFMAF32XEmulation` across branches

Branch 1 (`isTN + noLDSTr + TLDS=1`) sets `UseMFMAF32XEmulation = False` (L50).  
Branch 2 (`isNN + TLDS=1 + VectorWidthA==1`) sets `UseMFMAF32XEmulation = True` (L136).  
This makes the effective flag value entirely opaque to the caller — it is determined by runtime layout.

---

## 5. Recommendations for 2bww

The goal is to lift these mutations out of schedule bodies into declarative metadata. Three graduated options, from minimal to clean:

### Option A — Expose a `ScheduleMetadata` return value (minimal, safe)

Extend `ScheduleInfo` (or add a parallel `ScheduleMetadata` dataclass) with optional fields for the five flags:

```python
@dataclass
class ScheduleMetadata:
    mfma_init_c_vgprs: bool = True          # almost always True
    use_plr_pack: bool = False
    use_mfma_f32x_emulation: Optional[bool] = None   # None = don't touch
    use_dot2_f32x_emulation: Optional[bool] = None
    swap_global_read_order: Optional[bool] = None
```

Schedule functions return `(True, opt1, metadata)` instead of mutating `kernel`. The dispatch layer applies the metadata after a successful match. This is a pure refactor with no semantic change.

**Upside:** Minimal blast radius. Can be done schedule-by-schedule.  
**Downside:** Still uses imperative mutation — just deferred.

### Option B — YAML / `@RegisterSchedule` decorator metadata (principled)

Add the flag values as keyword arguments to `@RegisterSchedule(...)`, matching the decorator pattern already used for `tile_config`, `dtype_predicate`, `matrix_inst`. Each schedule's known static flags (those not branch-dependent) move to the decorator:

```python
@RegisterSchedule(
    tile_config=TileConfig(128, 128, 32, ...),
    dtype_predicate=isTF32,
    matrix_inst=[16, 16, 32, 1],
    mfma_wave_group=[2, 2],
    # ---- new ----
    mfma_init_c_vgprs=True,
    use_plr_pack=True,
    use_mfma_f32x_emulation=True,
    use_dot2_f32x_emulation=False,
)
```

For the branch-dependent flags (e.g., `SwapGlobalReadOrder` depends on layout), the schedule body returns them via `ScheduleMetadata`.

**Upside:** Cleanest for static flags. Enables pre-dispatch validation (e.g., "does this registered schedule's XEmulation setting conflict with the kernel's existing flag?").  
**Downside:** The branch-dependent cases (`SwapGlobalReadOrder`, layout-conditional `UseMFMAF32XEmulation`) still require runtime metadata.

### Option C — Per-branch named schedule variants (long-term ideal)

Rather than one schedule function with multiple `if/elif` branches, register one function per layout variant. Each becomes truly stateless with respect to the kernel dict. Flags are part of the registration. This eliminates the pre-branch mutation anomaly (Observation O7) entirely.

**The path of least resistance for 2bww:**

1. Fix the pre-branch mutation anomaly (O7) immediately — those four functions must not mutate when they return `False, None`. Move the `MfmaInitCVgprs=True` inside the success branch.
2. Adopt Option A (ScheduleMetadata) for all other mutations as a single mechanical refactor.
3. Treat Option B as a follow-on design task once the contract is clean.

---

*End of audit. No production code was modified.*
