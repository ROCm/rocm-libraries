# Device Alpha Scalar Mode — Implementation Summary

## Overview

Extended `UseDeviceAlpha` (formerly `UseScaleAlphaVec`) from a 2-bit field (values 0–3) to a 3-bit bitfield (values 0–7) by adding **bit 2 (value 4)** for "device alpha scalar" mode. This enables a single compiled kernel to support M-vector, N-vector, and scalar alpha scaling, selected at runtime via the `factorDim` kernel argument.

The full rename from `ScaleAlphaVec` → `DeviceAlpha` was applied across the entire codebase (Python, C++, YAML) to unify naming around the "device alpha" concept.

### Bitfield Definition

| Bit | Value | Mode | Description |
|-----|-------|------|-------------|
| 0 | 1 | M-vector | Scale each output row by `deviceAlpha[m]` |
| 1 | 2 | N-vector | Scale each output column by `deviceAlpha[n]` |
| 2 | 4 | Device scalar | Scale all outputs by `deviceAlpha[0]` (a single float) |

Combined values: 5 = M-vector + scalar, 7 = all three modes.

### Runtime Selection

The `factorDim` kernel argument selects the active mode at runtime:
- `factorDim=0` → M-vector path
- `factorDim=1` → N-vector path
- `factorDim=2` → Device scalar path

---

## Kernel Arguments and SGPR Resources per UseDeviceAlpha Value

### Kernel Arguments

| UseDeviceAlpha | AddressDeviceAlpha (8B) | factorDim (4B) | Total added args |
|----------------|------------------------|-----------------|------------------|
| 0 | No | No | 0 B |
| 1 (M-vec) | Yes | No | 8 B |
| 2 (N-vec) | Yes | No | 8 B |
| 3 (M+N vec) | Yes | Yes | 12 B |
| 4 (scalar) | Yes | No | 8 B |
| 5 (M-vec+scalar) | Yes | Yes | 12 B |
| 7 (all modes) | Yes | Yes | 12 B |

- `AddressDeviceAlpha`: 64-bit pointer to the device alpha buffer (always present when UseDeviceAlpha > 0)
- `factorDim`: 32-bit runtime mode selector (only when UseDeviceAlpha ≥ 3, i.e. multiple modes need disambiguation)

### SGPR Resources

| Resource | Count | Condition | Lifetime |
|----------|-------|-----------|----------|
| `AddressDeviceAlpha` | 2 | UseDeviceAlpha > 0 | Kernel arg → post-loop cleanup |
| `SrdDeviceAlpha` | 4 | UseDeviceAlpha > 0 | Post-loop SRD setup → cleanup |
| `FactorDim` | 1 | UseDeviceAlpha ≥ 3 | Kernel arg (shared w/ Bias) |
| Temp SGPR (scalar load) | 1 | UseDeviceAlpha & 4 | Post-loop scalar branch only (immediately freed) |

#### SGPR totals by UseDeviceAlpha value

| UseDeviceAlpha | AddressDeviceAlpha | SrdDeviceAlpha | FactorDim | Temp scalar | Total SGPRs |
|----------------|-------------------|----------------|-----------|-------------|-------------|
| 0 | 0 | 0 | 0 | 0 | **0** |
| 1 | 2 | 4 | 0 | 0 | **6** |
| 2 | 2 | 4 | 0 | 0 | **6** |
| 3 | 2 | 4 | 1 | 0 | **7** |
| 4 | 2 | 4 | 0 | 1 (temp) | **6** + 1 temp |
| 5 | 2 | 4 | 1 | 1 (temp) | **7** + 1 temp |
| 7 | 2 | 4 | 1 | 1 (temp) | **7** + 1 temp |

The temp SGPR for scalar load exists only within a `with self.allocTmpSgpr(1, 1)` scope in the scalar branch — it is allocated, used for `SLoadB32` + `VMulF32` fold into Alpha, then immediately released.

### Original vs Optimized Design (Scalar Path)

#### Original Design — Dedicated SGPR (`SgprDeviceAlphaScalar`)

```
Post-loop (SRD setup):
  SLoadB32  SgprDeviceAlphaScalar, [AddressDeviceAlpha]  ; load scalar
  SWaitCnt  kmcnt=0                                       ; wait

  ... hold SgprDeviceAlphaScalar across entire global write ...

Global write (per element batch):
  VMulF32   ValuC[0], SgprDeviceAlphaScalar, ValuC[0]    ; *= scalar
  VMulF32   ValuC[1], SgprDeviceAlphaScalar, ValuC[1]    ; *= scalar
  ...                                                     ; gwvw times
```

- **SGPR cost**: 1 dedicated SGPR (`SgprDeviceAlphaScalar`) held from post-loop through entire global write
- **ALU cost**: `gwvw` × `VMulF32` per element batch (e.g. gwvw=8 → 8 extra VMulF32 instructions per batch)
- **Function**: `applyDeviceAlphaScalar()` in GlobalWriteBatch.py

#### Optimized Design — Fold into sgprAlpha

```
Post-loop (SRD setup, scalar branch):
  SLoadB32  tmpSgpr, [AddressDeviceAlpha]     ; load scalar into temp
  SWaitCnt  kmcnt=0                            ; wait
  VMovB32   tmpVgpr, sgprAlpha                 ; move Alpha to VGPR
  VMulF32   tmpVgpr, tmpVgpr, tmpSgpr          ; alpha *= deviceAlphaScalar
  SNop      0                                  ; hazard wait
  VReadfirstlaneB32  sgprAlpha, tmpVgpr        ; write back to SGPR
  ; tmpSgpr freed, tmpVgpr freed — nothing held

Global write (per element batch):
  ; No extra instructions — scalar already baked into Alpha
```

- **SGPR cost**: 0 dedicated SGPRs (1 temp SGPR borrowed and immediately freed)
- **ALU cost**: 0 extra instructions in global write (3 instructions once in post-loop: `VMovB32`, `VMulF32`, `VReadfirstlaneB32`)
- **Function**: `applyDeviceAlphaScalar()` removed from GlobalWriteBatch.py

#### Comparison Summary

| Aspect | Original | Optimized |
|--------|----------|-----------|
| Dedicated SGPRs | 1 (`SgprDeviceAlphaScalar`) | 0 |
| SGPR lifetime | Entire global write | None (temp, immediately freed) |
| Global write ALU | `gwvw` × VMulF32 per batch | 0 |
| Post-loop ALU | SLoadB32 + SWaitCnt | SLoadB32 + SWaitCnt + VMovB32 + VMulF32 + SNop + VReadfirstlane |
| Total extra instructions | `gwvw × N_batches` | 3 (one-time) |
| Code complexity | `applyDeviceAlphaScalar` function | No function needed |

The optimization follows the same pattern already used by `UseScaleAB == "Scalar"` — fold the scalar into `sgprAlpha` via a VGPR round-trip (`VMovB32 → VMulF32 → VReadfirstlaneB32`) because AMDGPU has no scalar float multiply instruction.

---

## Commits

Four commits implement this feature:

1. **`[tensilelite] support device alpha`** — Core scalar mode implementation (12 Python + C++ files)
2. **`rename scale alpha vector as device alpha`** — Full rename of `UseScaleAlphaVec` → `UseDeviceAlpha` and all related identifiers across 206 files (35 code files + 171 YAML test configs)
3. **`rename SAV kernel abbreviation as DA`** — Kernel naming abbreviation `_SAV` → `_DA{N}` (value-encoded), plus file renames for 99 CustomKernels and 16 YAML configs
4. **`optimize device alpha scalar: fold into sgprAlpha`** — Remove `SgprDeviceAlphaScalar`, fold scalar into Alpha in post-loop, remove `applyDeviceAlphaScalar`, fix scalar-only mode vector guard

---

## Files Modified

### Commit 1: Core Implementation (12 code files)

#### 1. `Tensile/SolutionStructs/Problem.py`

Updated the parameter definition and comment:

```python
"UseDeviceAlpha": 0,  # =1 M-vector, =2 N-vector, =3 M+N vector, =4 device scalar, =7 all modes
```

#### 2. `Tensile/Components/Signature.py`

Changed the condition for adding `factorDimSize` to the kernel signature from exact equality to a range check, so values 4, 5, 6, 7 also get the factorDim argument:

```python
# Before:
if kernel["ProblemType"]["UseDeviceAlpha"] == 3:
# After:
if kernel["ProblemType"]["UseDeviceAlpha"] >= 3:
```

#### 3. `Tensile/KernelWriter.py`

Changed `FactorDim` state initialization and `numStoreSgprNames` guard from `== 3` to `>= 3`:

```python
# Before:
if self.states.FactorDim == 3:
# After:
if self.states.FactorDim >= 3:
```

#### 4. `Tensile/KernelWriterAssembly.py` — Primary assembly generation (~18K lines)

This file received the most changes. Key modifications:

##### 4a. SGPR Allocation

Added SRD SGPRs for the device alpha feature:

```python
if kernel["ProblemType"]["UseDeviceAlpha"]:
    self.defineSgpr("SrdDeviceAlpha", 4, 4)          # SRD for buffer loads
    module.add(RegSet("s", "sgprSrdDeviceAlpha", self.sgprs["SrdDeviceAlpha"]))
# No dedicated SGPR for scalar — it is folded into sgprAlpha in the post-loop
```

##### 4b. Post-Loop factorDims Computation

Extended the `factorDims` list to include dim 2 when bit 2 is set:

```python
useDeviceAlpha = kernel["ProblemType"]["UseDeviceAlpha"]
useBias = kernel["ProblemType"]["UseBias"]
needDim0 = (useDeviceAlpha & 1) or (useBias & 1)
needDim1 = (useDeviceAlpha & 2) or (useBias & 2)
if needDim0 and needDim1:
    factorDims = [0, 1]
elif needDim1:
    factorDims = [1]
elif needDim0:
    factorDims = [0]
else:
    factorDims = [0]
if useDeviceAlpha & 4:
    factorDims.append(2)    # NEW: add scalar dimension
```

##### 4c. Post-Loop SRD Setup with 3-Way Branching

Added branching logic for the scalar path in SRD setup. When `UseDeviceAlpha` has both vector and scalar bits set, the code generates a branch:
1. Check if `factorDim == 2` → branch to scalar path
2. Vector path: set up SRD with vector length (SizeI or SizeJ), scale by BPE
3. Scalar path: set up SRD with length 1, load scalar into temp SGPR, fold into Alpha via VGPR round-trip

```python
if (useDeviceAlpha & 4) and (useDeviceAlpha & 3):
    # Branch to scalar path if factorDim == 2
    module.add(self.getSCMPKInstruction("EQU32", "FactorDim", 2, comment="FactorDim == 2 (scalar)?"))
    module.add(SCBranchSCC1(deviceAlphaScalarLabel.getLabelName()))

# ... vector SRD setup using allocPostLoopSrdSuppress("DeviceAlpha", ...) ...

if useDeviceAlpha & 4:
    if useDeviceAlpha & 3:
        module.add(SBranch(labelName=deviceAlphaSrdEndLabel.getLabelName(), comment="Skip scalar path"))
        module.add(deviceAlphaScalarLabel)
    # Scalar path: SRD with length 1, load and fold into Alpha
    module.add(self.allocPostLoopSrdSuppress("DeviceAlpha", labelStrScalar, sgprLength=1))
    module.add(SMulI32(dst=sgpr("SrdDeviceAlpha+2"), ...))
    with self.allocTmpSgpr(1, 1) as tmpSgprRes:
        tmpSgprScalar = tmpSgprRes.idx
        module.add(SLoadB32(dst=sgpr(tmpSgprScalar), base=sgpr("AddressDeviceAlpha",2), soffset=0))
        module.add(SWaitCnt(kmcnt=0, comment="wait for device alpha scalar load"))
        newAlphaVgpr = self.vgprPool.checkOut(1)
        module.add(VMovB32(dst=vgpr(newAlphaVgpr), src=sgpr("Alpha")))
        module.add(VMulF32(dst=vgpr(newAlphaVgpr), src0=vgpr(newAlphaVgpr), src1=sgpr(tmpSgprScalar),
                           comment="alpha *= device alpha scalar"))
        module.add(SNop(waitState=0, comment="1 wait states"))
        module.add(VReadfirstlaneB32(dst=sgpr("Alpha"), src=vgpr(newAlphaVgpr),
                                     comment="Update Alpha with device alpha scalar"))
        self.vgprPool.checkIn(newAlphaVgpr)
```

##### 4d. `checkFactorDimValue` Function

New helper function to branch on a specific factorDim value (generalization of the existing `checkIsFactorDimZero`):

```python
def checkFactorDimValue(self, kernel, tmpSgprInfo, value, label, isLongBranch=False):
    module = Module("checkFactorDimValue_%d" % value)
    module.add(self.getSCMPKInstruction("EQU32", "FactorDim", value, comment="FactorDim == %d?" % value))
    module.add(SCBranchSCC1(label, comment="Branch if factorDim == %d" % value))
    return module
```

##### 4e. readVectorToLDS Scalar Skip

When `factorDim == 2` (scalar mode), the vector load into LDS is skipped entirely since the scalar is already folded into Alpha:

```python
vectorFactorDims = [fd for fd in factorDims if fd < 2]  # Exclude scalar dim
# Only generate LDS load code for vector dimensions
```

##### 4f. Global Write factorDim Dispatch

**This was the location of Bug 1.** The branching at the top of each global write batch dispatches to the correct factorDim-specific code path:

```python
if len(factorDims) >= 2:
    isLongBranch = True if currentInstLength >= self.states.asmCaps["ShortBranchMaxLength"] else False
    with self.allocTmpSgpr(3) as tmpSgprInfo:
        # INSERT ORDER MATTERS: pos=0 is LIFO
        # Insert checkIsFactorDimZero FIRST (ends up second in instruction stream)
        if len([fd for fd in factorDims if fd < 2]) >= 2:
            checkIsFactorDimZero = betaModule.add(self.checkIsFactorDimZero(...), pos=0)
            currentInstLength += countInstruction(checkIsFactorDimZero)
        # Insert checkFactorDimScalar SECOND (ends up first in instruction stream)
        if 2 in factorDims:
            checkFactorDimScalar = betaModule.add(self.checkFactorDimValue(kernel, tmpSgprInfo, 2, ...), pos=0)
            currentInstLength += countInstruction(checkFactorDimScalar)
```

The resulting instruction stream executes in order:
1. `factorDim == 2?` → branch to scalar write path
2. `factorDim != 0?` → branch to N-vector write path
3. Fall through → M-vector write path

##### 4g. SGPR Free/Undefine

Updated references for the renamed SGPRs:

```python
module.add(self.setSgprToFreeState("SrdDeviceAlpha"))
module.add(self.undefineSgpr("SrdDeviceAlpha"))
module.add(self.setSgprToFreeState("AddressDeviceAlpha"))
module.add(self.undefineSgpr("AddressDeviceAlpha"))
```

#### 5. `Tensile/Components/GlobalWriteBatch.py`

##### 5a. Vector Load Guarded by Vector Bits

The vector load and apply are guarded by `UseDeviceAlpha & 3` (has vector bits), not just `UseDeviceAlpha`. This correctly skips vector operations for scalar-only mode (`UseDeviceAlpha=4`):

```python
# Vector load — only when vector bits are set
if (self.kernel["ProblemType"]["UseDeviceAlpha"] & 3) and isSingleKernel and self.factorDim != 2:
    # load deviceAlpha vector

# Vector apply — only when vector bits are set
if (self.kernel["ProblemType"]["UseDeviceAlpha"] & 3) and isSingleKernel:
    if self.factorDim != 2:
        applyScaleVec(deviceAlphaModule, "DeviceAlpha", dataDeviceAlpha, self.factorDim, isGlobal=False)
```

For scalar mode (`factorDim==2`), both load and apply are skipped — the scalar is already folded into Alpha.

#### 6. `Tensile/AsmStoreState.py`

##### 6a. VectorDataTypes Dataclass

Fields renamed: `deviceAlphaM`, `deviceAlphaN`, method `deviceAlpha(dim)`.

##### 6b. referenceVgprDim Extended

Extended from 2 slots to 3 slots. The third slot (index 2) is for scalar mode and always empty (no VGPRs needed):

```python
self.referenceVgprDim = [[], [], []]  # was [[], []]
```

Added guard to skip VGPR allocation for DeviceAlpha when `factorDim >= 2`:

```python
if factorDim < 2:
    # allocate VGPRs for DeviceAlpha vector load
```

#### 7. `Tensile/AsmAddressCalculation.py`

Extended offset arrays from 2 to 3 slots:

```python
self.biasOffset = [0, 0, 0]            # was [0, 0]
self.deviceAlphaOffset = [0, 0, 0]     # was [0, 0]
```

#### 8. `Tensile/KernelWriterConversion.py`

Changed `== 3` to `>= 3` for enableFactorDim, and added factorDim==2 scalar branch in the conversion kernel (HIP C++ code generation):

```cpp
}else if(arg.factorDim == 2){
    // scalar mode: multiply all elements by DeviceAlpha[0]
    for(int vIdx = 0; vIdx < gwvw; vIdx++){
        accum[vIdx] *= (float)arg.DeviceAlpha[0];
    }
}
```

#### 9. `src/ContractionSolution.cpp`

- **factorDim argument guard**: `useDeviceAlpha >= 3` (was `== 3`)
- **enableFactorDim guard**: `useDeviceAlpha >= 3` (was `== 3`)
- **Kernel naming**: Extended naming logic for `factorDim >= 4` to compute factorDims from the bitfield

#### 10. `client/src/Reference.cpp`

##### 10a. GSU Path factorDim==2 Branch

```cpp
else if(factorDim == 2)
{
    alpha *= shadowDeviceAlpha[0];   // scalar mode: fold into alpha
}
```

##### 10b. Standard Path factorDim==2 Branch

```cpp
Accumulator deviceAlpha = GetValue<Accumulator>(
    problem.alphaType(), inputs.deviceAlpha, pos, aConjugate);
resultD *= deviceAlpha;
```

#### 11. `client/src/ClientProblemFactory.cpp`

Changed `== 3` to `>= 3` for factorDimSize computation and factorDim iteration, and updated factorDim computation to handle values 4–7.

#### 12. `Tensile/SolutionStructs/Solution.py`

Fixed LDS allocation to use bitfield checks instead of exact equality:

```python
# Before:
if savDim == 1:
    maxTurn = calcEpilogueTurns([0])
elif savDim == 2:
    maxTurn = calcEpilogueTurns([1])
elif savDim == 3:
    maxTurn = calcEpilogueTurns([0, 1])

# After:
savVecBits = savDim & 3   # extract vector bits (0-3)
if savVecBits == 1:
    maxTurn = calcEpilogueTurns([0])
elif savVecBits == 2:
    maxTurn = calcEpilogueTurns([1])
elif savVecBits == 3:
    maxTurn = calcEpilogueTurns([0, 1])
# Bit 2 (scalar) doesn't need LDS — no additional handling needed
```

---

### Commit 2: Full Rename (35 code files + 171 YAML configs)

Comprehensive rename of `UseScaleAlphaVec` → `UseDeviceAlpha` and all related identifiers throughout the codebase. This includes:

#### Python Files (14 files)

| File | Key Renames |
|------|-------------|
| `Tensile/SolutionStructs/Problem.py` | `UseScaleAlphaVec` → `UseDeviceAlpha` |
| `Tensile/Components/Signature.py` | `AddressScaleAlphaVec` → `AddressDeviceAlpha` |
| `Tensile/KernelWriter.py` | `ScaleAlphaVec` references → `DeviceAlpha` |
| `Tensile/KernelWriterAssembly.py` | `SrdScaleAlphaVec` → `SrdDeviceAlpha`, `AddressScaleAlphaVec` → `AddressDeviceAlpha`, all local variables |
| `Tensile/KernelWriterConversion.py` | `ScaleAlphaVec` → `DeviceAlpha` in generated C++ |
| `Tensile/Components/GlobalWriteBatch.py` | `addrScaleAlphaVec` → `addrDeviceAlpha`, `modGwvwScaleAlpha` → `modGwvwDeviceAlpha` |
| `Tensile/AsmStoreState.py` | `scaleAlphaM/N` → `deviceAlphaM/N`, `scaleAlpha()` → `deviceAlpha()`, `sharedColScaleAlphaVgprs` → `sharedColDeviceAlphaVgprs` |
| `Tensile/AsmAddressCalculation.py` | `scaleAlphaVecOffset` → `deviceAlphaOffset` |
| `Tensile/Components/LSU.py` | `ScaleAlphaVec` → `DeviceAlpha` |
| `Tensile/Contractions.py` | `ScaleAlphaVec` → `DeviceAlpha` |
| `Tensile/SolutionLibrary.py` | `ScaleAlphaVec` → `DeviceAlpha` |
| `Tensile/ClientWriter.py` | `ScaleAlphaVec` → `DeviceAlpha` |
| `Tensile/Common/GlobalParameters.py` | `DataInitTypeScaleAlphaVec` → `DataInitTypeDeviceAlpha` |
| `Tensile/SolutionStructs/Solution.py` | `ScaleAlphaVec` → `DeviceAlpha` |

#### C++ Files (18 files)

| File | Key Renames |
|------|-------------|
| `include/Tensile/ContractionProblem.hpp` | `SCALEALPHAVEC` → `DEVICEALPHA`, `useScaleAlphaVec()` → `useDeviceAlpha()`, `setScaleAlphaVec()` → `setDeviceAlpha()`, `m_scaleAlphaVecType` → `m_deviceAlphaType` |
| `include/Tensile/ContractionProblemPredicates.hpp` | `UseScaleAlphaVecCheck` → `UseDeviceAlphaCheck` |
| `include/Tensile/ContractionProblem_Detail.hpp` | `scaleAlphaVec` → `deviceAlpha` |
| `include/Tensile/ContractionSolution.hpp` | `scaleAlphaVec` → `deviceAlpha`, `useScaleAlphaVec` → `useDeviceAlpha` |
| `include/Tensile/Serialization/ContractionPredicates.hpp` | `UseScaleAlphaVec` → `UseDeviceAlpha` |
| `include/Tensile/Serialization/ContractionSolution.hpp` | `scaleAlphaVec` → `deviceAlpha` |
| `src/ContractionProblem.cpp` | `scaleAlpha` descriptor → `deviceAlpha` |
| `src/ContractionSolution.cpp` | `scaleAlphaVec` → `deviceAlpha`, `useScaleAlphaVec` → `useDeviceAlpha` |
| `client/src/Reference.cpp` | `shadowAlphaVec` → `shadowDeviceAlpha` |
| `client/src/ClientProblemFactory.cpp` | `useScaleAlphaVec` → `useDeviceAlpha` |
| `client/include/ClientProblemFactory.hpp` | `ScaleAlphaVec` → `DeviceAlpha` |
| `client/src/DataInitialization.cpp` | `ScaleAlphaVec` → `DeviceAlpha` |
| `client/src/ReferenceValidator.cpp` | `ScaleAlphaVec` → `DeviceAlpha` |
| `client/include/ReferenceValidator.hpp` | `ScaleAlphaVec` → `DeviceAlpha` |
| `client/src/ProgressListener.cpp` | `ScaleAlphaVec` → `DeviceAlpha` |
| `client/main.cpp` | `ScaleAlphaVec` → `DeviceAlpha` |
| `client/cpu_gemm_driver.cpp` | `ScaleAlphaVec` → `DeviceAlpha` |
| `Tensile/Utilities/tensile_generator/tensile_config_generator.py` | `ScaleAlphaVec` → `DeviceAlpha` |

#### Unit Test Files (3 files)

- `Tensile/Tests/unit/test_MatrixInstructionConversion.py`
- `Tensile/Tests/unit/test_TensileLibLogicToYaml.py`
- `Tensile/Tests/unit/test_storeD_roundtrip.py`

#### YAML Test Configs (171 files)

All YAML test files referencing `UseScaleAlphaVec` or `DataInitTypeScaleAlphaVec` were updated to use `UseDeviceAlpha` and `DataInitTypeDeviceAlpha` respectively.

---

### Commit 3: Kernel Abbreviation Rename `_SAV` → `_DA{N}` (128 files)

The kernel naming abbreviation `SAV` (from "ScaleAlphaVec") was replaced with `DA{N}` where `{N}` is the `UseDeviceAlpha` bitfield value. This makes the kernel name encode the actual mode (e.g. `_DA1_` for M-vector, `_DA3_` for M+N vector, `_DA7_` for all modes).

#### Naming Code Changes

| File | Change |
|------|--------|
| `Tensile/SolutionStructs/Problem.py` | `name.append("SAV")` → `name.append("DA%d" % self["UseDeviceAlpha"])` |
| `Tensile/SolutionLibrary.py` | `placeholderName += '_SAV'` → `placeholderName += '_DA'` |
| `clients/tests/src/matmul_gtest.cpp` | `name << "_SAV"` → `name << "_DA"` (scaleAlpha_vector test naming) |
| `clients/tests/data/smoke_gtest.yaml` | Test name: `matmul_bias_relu_SAV_smoke` → `matmul_bias_relu_DA_smoke` |

#### File Renames (115 files)

- **99 CustomKernel `.s` files**: `_SAV_` → `_DA1_` in both filenames and internal kernel symbols
- **16 YAML config files** under `tests/configs/mixed_configs/`: `_SAV` → `_DA1` in filenames

#### Not Changed (out of scope)

- `matmul_gtest.cpp` line 129: `_SAV` for "Scale A Vector" (`scaleA` format) — different feature, not related to DeviceAlpha
- Logic library files under `library/src/amd_detail/rocblaslt/src/Tensile/Logic/` — pre-built device library; needs regeneration

---

### Commit 4: Optimize Scalar Path — Fold into sgprAlpha (3 files)

Eliminated the dedicated `SgprDeviceAlphaScalar` SGPR and the per-element `applyDeviceAlphaScalar` function. The device alpha scalar is now multiplied into `sgprAlpha` once in the post-loop, making the scalar path zero-cost in global write.

#### Files Modified

| File | Change |
|------|--------|
| `Tensile/KernelWriterAssembly.py` | Removed `SgprDeviceAlphaScalar` SGPR definition. Scalar branch now loads into temp SGPR, folds into Alpha via `VMovB32 → VMulF32 → VReadfirstlaneB32`, releases temp. |
| `Tensile/Components/GlobalWriteBatch.py` | Removed `applyDeviceAlphaScalar()` function. Changed vector load/apply guards from `UseDeviceAlpha` to `UseDeviceAlpha & 3` to correctly skip vector operations for scalar-only mode. |

#### Bug Fixed: Scalar-Only Mode Vector Guard

**Symptom**: `UseDeviceAlpha=4` (scalar-only) with runtime `factorDim=0` caused validation failure.

**Root Cause**: The vector load and apply in GlobalWriteBatch.py were guarded by `self.kernel["ProblemType"]["UseDeviceAlpha"]` (any nonzero value), not `UseDeviceAlpha & 3` (has vector bits). For `UseDeviceAlpha=4`, the code attempted to load DeviceAlpha as a vector even though no vector SRD was configured.

**Fix**: Changed both guards to `UseDeviceAlpha & 3`, so scalar-only mode correctly skips all vector operations regardless of the runtime factorDim value.

---

## Bugs Found and Fixed

### Bug 1 (Critical): Wrong Branching Order in Global Write Dispatch

**Symptom**: UseDeviceAlpha=7 with factorDim=2 produced fundamentally wrong results (opposite sign, wrong magnitude) with random A/B data.

**Root Cause**: In the global write section, `checkFactorDimScalar` and `checkIsFactorDimZero` were both inserted at `pos=0` (LIFO). The `factorDim!=0` check executed before `factorDim==2`, routing scalar to the N-vector path.

**Fix**: Swapped insertion order so scalar check executes first.

### Bug 2: LDS Allocation Missing for UseDeviceAlpha > 3

**Symptom**: `calcEpilogueTurns` used exact equality (`savDim == 1/2/3`), so values 4–7 got zero LDS allocation.

**Fix**: Changed to `savVecBits = savDim & 3` to extract vector bits.

### Bug 3: Scalar-Only Mode Vector Guard

**Symptom**: `UseDeviceAlpha=4` validation failure when runtime sends `factorDim=0`.

**Root Cause**: Vector load/apply guarded by `UseDeviceAlpha` (truthy for 4) instead of `UseDeviceAlpha & 3` (zero for 4).

**Fix**: Changed guard to `UseDeviceAlpha & 3`.

---

## Test Results

All tests pass with random A/B data initialization:

| UseDeviceAlpha | FactorDimArgs | factorDim=0 | factorDim=1 | factorDim=2 |
|----------------|---------------|-------------|-------------|-------------|
| 3 (baseline) | [0, 1] | PASSED | PASSED | N/A |
| 4 (scalar only) | (none) | PASSED | N/A | N/A |
| 5 (M-vec + scalar) | [0, 2] | PASSED | N/A | PASSED |
| 7 (all modes) | [0, 1, 2] | PASSED | PASSED | PASSED |

Test configuration:
- GPU: gfx1250 (ISA: [[12, 5, 0]])
- Data types: BF16 input, BF16 output, FP32 compute
- Matrix: 128x128x1x64 (MxNxBatchxK)
- Wave size: 32
- MatrixInstruction: [16, 16, 32, 1, 1, 2, 1, 1, 2]
- Random initialization for A, B, C matrices and deviceAlpha

---

## Test YAML Files Created

| File | UseDeviceAlpha | FactorDimArgs | Purpose |
|------|----------------|---------------|---------|
| `test_scale_alpha_scalar.yaml` | 7 | [0, 1, 2] | Full 3-way test |
| `test_scale_alpha_scalar_debug.yaml` | 7 | [0, 1, 2] | Full 3-way test (debug settings) |
| `test_scale_alpha_scalar_only.yaml` | 4 | (none) | Scalar-only mode |
| `test_scale_alpha_scalar_5.yaml` | 5 | [0, 2] | M-vector + scalar |
| `test_scale_alpha_scalar_7_fd2only.yaml` | 7 | [2] | All modes, scalar-only runtime |
| `test_scale_alpha_vec3.yaml` | 3 | [0, 1] | Baseline regression |

---

## Key Technical Details

### Assembly Flow for factorDim=2 (Scalar Path) — Optimized

1. **Post-loop SRD setup**: Branch on `factorDim == 2`, set up minimal SRD (length 1), load scalar via `SLoadB32` into temp SGPR, wait with `SWaitCnt(kmcnt=0)`, fold into `sgprAlpha` via `VMovB32 → VMulF32 → VReadfirstlaneB32`, release temp
2. **readVectorToLDS**: Skipped entirely for scalar path (no vector to load)
3. **Global write dispatch**: `checkFactorDimValue(2)` branches to scalar-specific global write batch
4. **Global write batch**: No extra DeviceAlpha operations — the scalar is already baked into Alpha, which is applied as part of normal alpha multiplication

### Critical Architecture Insight: pos=0 Insertion Semantics

`module.add(instructions, pos=0)` inserts at position 0 (beginning) of the module. When multiple items use `pos=0`:
- The **first** `pos=0` insert goes to position 0
- The **second** `pos=0` insert pushes the first one down, taking position 0 itself
- Result: **LIFO order** — the last inserted item executes first

This is critical for branching logic where execution order matters. The factorDim=2 check must execute before the factorDim!=0 check to prevent the scalar value from being caught by the vector branch.

### Why VGPR Round-Trip for Scalar Multiply

AMDGPU has no scalar float multiply instruction (`s_mul_f32` does not exist). To multiply two SGPRs as floats:
1. `VMovB32` — move SGPR to VGPR
2. `VMulF32` — VALU float multiply (VGPR × SGPR → VGPR)
3. `SNop` — hazard wait (1 cycle)
4. `VReadfirstlaneB32` — move VGPR lane 0 back to SGPR

This is the same pattern used by `UseScaleAB == "Scalar"` to fold scaleA/scaleB into Alpha.

### gfx1250 Notes

- `SWaitCnt` for scalar loads uses `kmcnt=0`, **not** `lgkmcnt=0` (older architectures use `lgkmcnt`)
- Wave size: 32 (wave32)
- ISA specification: `[[12, 5, 0]]`

### Cache Clearing

Three cache layers must be cleared during development to avoid using stale kernels:

```bash
rm -rf build_tmp/1_BenchmarkProblems build_tmp/2_BenchmarkData ~/.tensile/helper_cache
```
