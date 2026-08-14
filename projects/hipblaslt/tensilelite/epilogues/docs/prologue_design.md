<!-- Copyright Advanced Micro Devices, Inc., or its affiliates. -->
<!-- SPDX-License-Identifier: MIT -->

# Prologue Design: Mechanism for Element-Wise Operations on A/B Tiles Before MFMA

## Overview

This document describes the structural mechanism for adding a *prologue* to the TensileLite
subtile kernel path. A prologue is an element-wise operation applied to the A or B input tile
VGPRs after the local-read (LR) stage and before the MFMA instruction that consumes them.
This document covers the injection point, tile layout, scheduling constraints, codegen
infrastructure, and register pressure headroom. It does not prescribe any specific operation.

All file paths are relative to
`/home/fmoracor/rocm-libraries/projects/hipblaslt/tensilelite/` unless otherwise noted.

---

## 1. Injection Point: Where A and B Tiles Are Live Before MFMA

### 1.1 Data flow: GR → LDS → LR → VGPRs → MFMA

The subtile kernel separates global read (GR) from local read (LR). After GR completes, data
lives in LDS. After LR (`ds_read_b128`), data lands in VGPRs held by the scheduler's tile
lists.

The relevant register objects are:

- **`scheduler.vgprTilesA`** and **`scheduler.vgprTilesB`** (allocated in
  `LogicalScheduler.allocVgprTiles`,
  `Tensile/Components/Subtile/LogicalScheduler.py` line 3820): lists of `RegisterTileInfo`
  objects indexed by `vgprTileId`.
- Each `RegisterTileInfo` wraps a `RegList` (from
  `Tensile/Components/Subtile/SubtileGeometry.py` line 28) that carries a list of physical
  VGPR indices (`regList.indices`) and the pool they were checked out from.

### 1.2 The primary injection point: `InstructionEmitter.emit_mfma`

The primary call site where A and B tile VGPRs are live and MFMA has not yet been issued is
`Tensile/Components/Subtile/InstructionEmitter.py`, inside `InstructionEmitter.emit_mfma`
(method starts line 149). The relevant section (around line 172–200):

```python
for a, b in abPairs:
    groupA = (a // self.config.lrA.mn) * self.config.lrA.mn
    groupB = (b // self.config.lrB.mn) * self.config.lrB.mn
    aTile = self.vgprTilesA[tile_maps['A'][groupA]]   # RegisterTileInfo
    bTile = self.vgprTilesB[tile_maps['B'][groupB]]   # RegisterTileInfo
    dTile = self.dtileInfo.vgprTiles[a + b * self.dtileInfo.localMMATileGrid[0]]
    ...
    module.add(emitMfmaInstruction(
        self.writer, self.kernel, aTile, bTile, dTile, dTile, ...))
```

The window between constructing `aTile`/`bTile` and appending `emitMfmaInstruction` is the
natural prologue slot. A prologue emitter receives `aTile` (or `bTile`) and modifies its
VGPRs in-place; `emitMfmaInstruction` then consumes the modified values.

`emitMfmaInstruction` itself (`Tensile/Components/Subtile/Kernel.py` line 1067) reads:
```python
vgprAStart = vgprTileA.regList.indices[0]
opASize    = len(vgprTileA.regList.indices)    # 4 for bf16 AB_B16
```
and emits `v_mfma_f32_16x16x32_bf16 acc[D], v[A:A+opASize-1], v[B:B+opBSize-1], acc[C]`.

A secondary (legacy, non-scheduler) call site exists in `emitMfmaCode` in
`Tensile/Components/Subtile/Kernel.py` line 1209, but that path is not used by the
production scheduler.

---

## 2. bf16 Tile Layout (`AB_B16`, TN Kernels)

### 2.1 Geometry constants

The `AB_B16` tile pair is declared in `Tensile/Components/Subtile/Kernel.py` lines 284–287:

```python
_B16 = dict(mmaLayout=MFMA_16x16_1B_4K_4V, instK=32, bpe=2, supportedTypes=('bf16', 'fp16'))

AB_B16 = ABTilePair(
    gr=ABGRGeometry(tag=GRTag_1x2(), **_B16, subtileShape=(1, 2), loadShape=LoadShape(m=1, k=8)),
    lr=ABLRGeometry(tag=LRTag_1x2(), **_B16, subtileShape=(1, 2), loadShape=LoadShape(m=1, k=8)),
)
```

`MFMA_16x16_1B_4K_4V` is defined in
`Tensile/Components/Subtile/SubtileGeometry.py` line 160:
```python
MFMA_16x16_1B_4K_4V = MMALayout(instM=16, blocks=1, vgprs=4, waveSize=64)
```

Key derived quantities (`SubtileGeometry.py` `ABInputGeometry.__post_init__`, lines 255–260):
- `mmaTileShape = (16, 32)` — 16 elements in M, 32 elements in K.
- `mmaTileRegCount = float(mmaLayout.vgprs) = 4.0` — **4 VGPRs per lane per MMA tile**.

### 2.2 VGPR packing per lane

For bf16 (`bpe=2`): 16×32 = 512 elements total. With 64 lanes, each lane holds 512/64 = 8
bf16 elements. These pack into 4 × 32-bit VGPRs: **2 bf16 elements per VGPR** (packed).

Each `RegisterTileInfo` for A or B holds exactly 4 VGPR indices (one `ds_read_b128` fills
all 4). The ISA packing convention: two bf16 per 32-bit VGPR, with the low 16-bit half and
the high 16-bit half carrying adjacent K-position elements.

**Any prologue that operates on individual bf16 values must unpack, operate, and repack.**
The lane-to-element mapping follows the standard 16×16×32 MFMA bf16 ISA layout:
- `kGroups = waveSize / (instM * blocks) = 64 / 16 = 4`.
- `elementsPerLaneNonK = instM / kGroups = 4`.
- VGPR[i] holds the two bf16 elements at K offsets `2i` and `2i+1` for the lane's M slice.

### 2.3 LR subtile shape

`ABLRGeometry.subtileShape = (1, 2)` means the LR stride covers 1 MMA tile in M and 2 in K.
A single LR pass on A issues 2 `ds_read_b128` instructions (one per `subIterK_within`). Each
reads into a separate 4-VGPR `RegisterTileInfo`.

---

## 3. Scheduling and Pipeline Constraints

### 3.1 The software pipeline

`UseSubtileImpl=1` kernels do not use `ScheduleIterAlg` (rejected by `Solution.py`
lines 1192–1193). Instead they use the `LogicalScheduler` (`Kernel.py` `mainLoop`,
line 1325+), which implements its own PGR=0/1/2 pipeline through `emitMainAndExitLoops`.

The pipeline phases for PGR=1 are:
- **Prefetch phase (preloop):** one round of GR + LR for the first macro-tile iteration.
- **Main loop body:** each subIterK slot — LR(n+1) interleaved with MFMAs consuming LR(n),
  plus GR(n+1) spread among free MFMA slots.
- **NLL variant (tail):** LR(n) only, no new GR.

### 3.2 Ordering constraint

A prologue must execute:
1. **After** the `ds_read_b128` (LR) that fills the tile VGPRs.
2. **After** the `s_waitcnt lgkmcnt(...)` that waits for the LR to complete.
3. **Before** the `v_mfma_f32_16x16x32_bf16` that consumes those VGPRs.

### 3.3 Placement inside `emit_mfma`

The `instructionSchedule` in `Tensile/Components/Subtile/InstructionScheduler.py` (line 455)
interleaves non-MFMA ops (LR, GR, waits) into 2 slots per MFMA interval. The 2-slot model
is tight; a prologue that emits more than a handful of instructions does not fit within the
existing slot placer.

The practical placement is immediately before each call to `emitMfmaInstruction` inside
`InstructionEmitter.emit_mfma` (InstructionEmitter.py line 196). The prologue modifies
`aTile`/`bTile` VGPRs **in-place** before the MFMA instruction is appended to the module.
Since `emit_mfma` returns `list(module.flatitems())`, the prologue instructions naturally
precede the MFMA in the emitted sequence, and the LR wait is already in scope.

**Latency headroom:** MFMA latency on gfx950 is approximately 64 cycles. A prologue that
requires fewer instructions than that fits inside the latency window without stalling. For
example, the convert/operate/repack sequence for a 4-VGPR tile takes approximately 24
instructions, well within the MFMA latency budget.

**PGR=2 (double-buffered GR):** the same placement holds — the prologue executes between
LR(n) completion and MFMA(n), which is already the window where GR(n+2) is being issued.
There is no WAW hazard because the prologue overwrites the input VGPR tile, not the output
accumulator.

### 3.4 Tail loop

The tail loop (emitted by `LogicalScheduler.emitTailLoop`) re-uses the same
`InstructionEmitter.emit_mfma` call path. A prologue guard placed inside `emit_mfma` applies
automatically to the tail loop with no extra work.

---

## 4. Codegen Infrastructure for an Arbitrary Prologue

### 4.1 Parameter guard

Following the `PartialRMS` pattern, a prologue feature is gated by a kernel parameter:

1. **`Tensile/Common/ValidParameters.py`** (line 462): add a boolean parameter entry (e.g.
   `"UsePrologueA": [False, True]`) to the `validParameters` dictionary.

2. **`Tensile/SolutionStructs/Solution.py`** (near line 1215, where `_validatePartialRMS`
   is called): add a validation function that enforces:
   - `state["UseSubtileImpl"]` is True.
   - The data type and architecture are supported (e.g. bf16 on gfx950, mirroring
     `_validateSubtileEpiloguePrereqs` at line 231).
   - `MacroTile0` satisfies any required alignment.

### 4.2 Hook point in `InstructionEmitter.emit_mfma`

The hook is added in `Tensile/Components/Subtile/InstructionEmitter.py` inside `emit_mfma`,
after `aTile`/`bTile` are constructed and before `emitMfmaInstruction` is appended:

```python
# After constructing aTile, before emitMfmaInstruction:
if self.kernel.get("UsePrologueA"):
    module.extend(self._emitPrologueA(aTile, ...))
```

`_emitPrologueA` is a method on `InstructionEmitter` (or a standalone emitter class in a new
file, e.g. `Tensile/Components/Subtile/SubtilePrologueEmit.py`) that:
1. Checks out temporary VGPRs.
2. Emits the element-wise transformation on `aTile.regList.indices`.
3. Writes results back into the same VGPR slots.
4. Checks in temporaries.
5. Returns the instructions as a flat list or module.

### 4.3 `UseSubtileImpl` guard analogy

`UseSubtileImpl` guards its code path at multiple points in `KernelWriter.py` (e.g. lines
5146, 5173). For a prologue, the analogous guard is:

```python
if kernel.get("UseSubtileImpl") and kernel.get("UsePrologueA"):
```

This mirrors how `PartialRMS` checks `kernel["PartialRMS"]`, which itself requires
`UseSubtileImpl` via `_validateSubtileEpiloguePrereqs` (Solution.py lines 231–232).

### 4.4 Epilogue-side injection (for comparison)

The existing `PartialRMS` epilogue is injected in `KernelWriter.py` after the main loop
completes:

```python
if kernel["PartialRMS"]:
    from .Components.Subtile.SubtilePartialRMSEmit import SubtilePartialRMSEmitter
    ...
```

A prologue differs: it must hook inside the MFMA loop body (inside `emit_mfma`), not after
the loop. This is the key structural difference from the existing epilogue injection pattern.

### 4.5 New prologue emitter file

A new file `Tensile/Components/Subtile/SubtilePrologueEmit.py` should follow the SPDX
license header convention:

```python
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
```

It should expose a class or function analogous to `SubtilePartialRMSEmitter`, but scoped to
a single `aTile`/`bTile` call rather than a post-loop pass over accumulator tiles.

---

## 5. Register Pressure Headroom

### 5.1 Current VGPR budget

In `mainLoop` (`Tensile/Components/Subtile/Kernel.py` lines 1358–1391), before calling
`allocVgprTiles`, the code checks VGPR headroom:

```python
vgprBudget = writer.states.regCaps["MaxVgpr"]
vgprUsed   = writer.vgprPool.size() - writer.vgprPool.available()
...
numVgpr = scheduler.getNumVgpr(tiA, tiB, scaleTiA, scaleTiB)
if vgprUsed + numVgpr <= vgprBudget:
    break
```

For a typical gfx950 bf16 kernel with `MT256x256`, `MIWaveGroup=[4,2]`:
- 4 VGPRs per tile × 4 A-tiles × 2 buffered = ~32 VGPRs for A/B inputs.
- D accumulator: 16 fp32 VGPRs (AGPRs on gfx950 with `MIArchVgpr=False`).
- ~64 VGPRs for addressing, SRDs, loop counters, and output.
- Typical remaining headroom: 100–130 VGPRs before hitting the 256-VGPR occupancy limit.

### 5.2 Temporary VGPR cost for a prologue

A prologue that unpacks packed elements, operates, and repacks needs a small number of
transient temporary VGPRs — typically 2–3 per call. These are live only for the duration of
the prologue instructions within a single `emit_mfma` invocation.

Because the temporaries are checked in before `emitMfmaInstruction` is appended, the peak
register pressure during the MFMA itself is unchanged. The pool pressure is transient and
does not affect the tile budget check in `mainLoop`.

### 5.3 VGPR checkout pattern

The established pattern in the subtile emitters (e.g. `SubtilePartialRMSEmit.emit`,
lines 179–186) is:

```python
tmpVgpr = writer.vgprPool.checkOut(n, tag="prologue_tmp")
# emit instructions using vgpr(tmpVgpr) ... vgpr(tmpVgpr + n - 1)
writer.vgprPool.checkIn(tmpVgpr)
```

Use `checkOutAligned` when the target instruction requires aligned VGPR pairs (e.g.
`ds_load_b128`). For a prologue whose temporaries are scalar within each `emit_mfma` call,
short-lived unaligned checkouts with a per-call `checkIn` are sufficient.

---

## 6. Key File and Line Reference Table

| Symbol | File | Line |
|--------|------|------|
| `AB_B16` definition | `Tensile/Components/Subtile/Kernel.py` | 284 |
| `MFMA_16x16_1B_4K_4V` | `Tensile/Components/Subtile/SubtileGeometry.py` | 160 |
| `emitMfmaInstruction` | `Tensile/Components/Subtile/Kernel.py` | 1067 |
| `emitMfmaCode` (legacy) | `Tensile/Components/Subtile/Kernel.py` | 1135 |
| `InstructionEmitter.emit_mfma` | `Tensile/Components/Subtile/InstructionEmitter.py` | 149 |
| `emit_mfma` calls `emitMfmaInstruction` | `Tensile/Components/Subtile/InstructionEmitter.py` | 196 |
| `allocVgprTiles` | `Tensile/Components/Subtile/LogicalScheduler.py` | 3820 |
| `getNumVgpr` | `Tensile/Components/Subtile/LogicalScheduler.py` | 3789 |
| `mainLoop` (pipeline entry) | `Tensile/Components/Subtile/Kernel.py` | 1325 |
| VGPR budget check in `mainLoop` | `Tensile/Components/Subtile/Kernel.py` | 1358–1391 |
| `TileInfo.allocVgprTileRegisters_legacy` | `Tensile/Components/Subtile/Kernel.py` | 687 |
| `emitSingleDsRead` | `Tensile/Components/Subtile/SubtileLREmit.py` | 668 |
| `SubtilePartialRMSEmitter.emit` (checkout pattern) | `Tensile/Components/Subtile/SubtilePartialRMSEmit.py` | 160 |
| PartialRMS epilogue injection | `Tensile/KernelWriter.py` | 5199–5209 |
| `ValidParameters` epilogue flags | `Tensile/Common/ValidParameters.py` | 462–465 |
| `_validateSubtileEpiloguePrereqs` | `Tensile/SolutionStructs/Solution.py` | 231 |
| `_validatePartialRMS` call site | `Tensile/SolutionStructs/Solution.py` | 1215 |
| `instructionSchedule` (2-slot placer) | `Tensile/Components/Subtile/InstructionScheduler.py` | 455 |
| `mmaTileRegCount` derivation | `Tensile/Components/Subtile/SubtileGeometry.py` | 260 |
| `ABLRGeometry.subtileShape` for B16 | `Tensile/Components/Subtile/Kernel.py` | 286 |
| `UseSubtileImpl` guard in KernelWriter | `Tensile/KernelWriter.py` | 5146, 5173 |
