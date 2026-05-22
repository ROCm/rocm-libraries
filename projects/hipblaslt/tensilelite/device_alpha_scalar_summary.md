# Device Alpha Scalar Mode — Implementation Summary

## Overview

Extended `UseScaleAlphaVec` from a 2-bit field (values 0–3) to a 3-bit bitfield (values 0–7) by adding **bit 2 (value 4)** for "device alpha scalar" mode. This enables a single compiled kernel to support M-vector, N-vector, and scalar alpha scaling, selected at runtime via the `factorDim` kernel argument.

### Bitfield Definition

| Bit | Value | Mode | Description |
|-----|-------|------|-------------|
| 0 | 1 | M-vector | Scale each output row by `scaleAlphaVec[m]` |
| 1 | 2 | N-vector | Scale each output column by `scaleAlphaVec[n]` |
| 2 | 4 | Device scalar | Scale all outputs by `scaleAlphaVec[0]` (a single float) |

Combined values: 5 = M-vector + scalar, 7 = all three modes.

### Runtime Selection

The `factorDim` kernel argument selects the active mode at runtime:
- `factorDim=0` → M-vector path
- `factorDim=1` → N-vector path
- `factorDim=2` → Device scalar path

---

## Files Modified (12 files)

### 1. `Tensile/SolutionStructs/Problem.py` — Line 430

Updated the comment documenting the `UseScaleAlphaVec` parameter to reflect the new bitfield:

```python
"UseScaleAlphaVec": 0,  # =1 M-vector, =2 N-vector, =3 M+N vector, =4 device scalar, =7 all modes
```

### 2. `Tensile/Components/Signature.py` — Lines 242–243

Changed the condition for adding `factorDimSize` to the kernel signature from exact equality to a range check, so values 4, 5, 6, 7 also get the factorDim argument:

```python
# Before:
if kernel["ProblemType"]["UseScaleAlphaVec"] == 3:
# After:
if kernel["ProblemType"]["UseScaleAlphaVec"] >= 3:
```

### 3. `Tensile/KernelWriter.py` — Lines 3573, 8726–8728

Changed `FactorDim` state initialization and `numStoreSgprNames` guard from `== 3` to `>= 3`:

```python
# Before:
if self.states.FactorDim == 3:
# After:
if self.states.FactorDim >= 3:
```

### 4. `Tensile/KernelWriterAssembly.py` — Primary assembly generation (~18K lines)

This file received the most changes. Key modifications:

#### 4a. SGPR Allocation (lines 7383–7388)

Added new SGPRs for the device alpha feature:

```python
if kernel["ProblemType"]["UseScaleAlphaVec"]:
    self.defineSgpr("SrdDeviceAlpha", 4, 4)          # SRD for buffer loads (renamed from SrdScaleAlphaVec)
    module.add(RegSet("s", "sgprSrdDeviceAlpha", self.sgprs["SrdDeviceAlpha"]))
if kernel["ProblemType"]["UseScaleAlphaVec"] & 4:
    self.defineSgpr("SgprDeviceAlphaScalar", 1)       # Holds the loaded scalar value (renamed from SgprScaleAlphaScalar)
    module.add(RegSet("s", "sgprSgprDeviceAlphaScalar", self.sgprs["SgprDeviceAlphaScalar"]))
```

#### 4b. Post-Loop factorDims Computation (lines 14025–14036)

Extended the `factorDims` list to include dim 2 when bit 2 is set:

```python
useScaleAlphaVec = kernel["ProblemType"]["UseScaleAlphaVec"]
useBias = kernel["ProblemType"]["UseBias"]
needDim0 = (useScaleAlphaVec & 1) or (useBias & 1)
needDim1 = (useScaleAlphaVec & 2) or (useBias & 2)
if needDim0 and needDim1:
    factorDims = [0, 1]
elif needDim1:
    factorDims = [1]
else:
    factorDims = [0]
if useScaleAlphaVec & 4:
    factorDims.append(2)    # NEW: add scalar dimension
```

#### 4c. Post-Loop SRD Setup with 3-Way Branching (lines 14039–14069)

Added branching logic for the scalar path in SRD setup. When `UseScaleAlphaVec` has both vector and scalar bits set, the code generates a branch:
1. Check if `factorDim == 2` → branch to scalar path
2. Vector path: set up SRD with vector length (SizeI or SizeJ), scale by BPE
3. Scalar path: set up SRD with length 1, load scalar via `SLoadB32`, wait with `SWaitCnt(kmcnt=0)`

```python
if (useScaleAlphaVec & 4) and (useScaleAlphaVec & 3):
    # Branch to scalar path if factorDim == 2
    scaleAlphaScalarLabel = Label(...)
    scaleAlphaVecSrdEndLabel = Label(...)
    module.add(self.getSCMPKInstruction("EQU32", "FactorDim", 2, comment="FactorDim == 2 (scalar)?"))
    module.add(SCBranchSCC1(scaleAlphaScalarLabel.getLabelName()))

# ... vector SRD setup using allocPostLoopSrdSuppressRaw("DeviceAlpha", "ScaleAlphaVec", ...) ...

if useScaleAlphaVec & 4:
    if useScaleAlphaVec & 3:
        module.add(SBranch(labelName=scaleAlphaVecSrdEndLabel.getLabelName(), comment="Skip scalar path"))
        module.add(scaleAlphaScalarLabel)
    # Scalar path: SRD with length 1, load the scalar
    module.add(self.allocPostLoopSrdSuppressRaw("DeviceAlpha", "ScaleAlphaVec", labelStrScalar, sgprLength=1))
    module.add(SMulI32(dst=sgpr("SrdDeviceAlpha+2"), ...))
    module.add(SLoadB32(dst=sgpr("SgprDeviceAlphaScalar"), base=sgpr("AddressScaleAlphaVec",2), soffset=0, comment="load device alpha scalar"))
    module.add(SWaitCnt(kmcnt=0, comment="wait for device alpha scalar load"))
```

#### 4d. `checkFactorDimValue` Function (lines 13781–13792)

New helper function to branch on a specific factorDim value (generalization of the existing `checkIsFactorDimZero`):

```python
def checkFactorDimValue(self, kernel, tmpSgprInfo, value, label, isLongBranch=False):
    module = Module("checkFactorDimValue_%d" % value)
    module.add(self.getSCMPKInstruction("EQU32", "FactorDim", value, comment="FactorDim == %d?" % value))
    module.add(SCBranchSCC1(label, comment="Branch if factorDim == %d" % value))
    return module
```

#### 4e. readVectorToLDS Scalar Skip (lines 14222–14255)

When `factorDim == 2` (scalar mode), the vector load into LDS is skipped entirely since we only need the single scalar value already loaded into an SGPR:

```python
vectorFactorDims = [fd for fd in factorDims if fd < 2]  # Exclude scalar dim
# Only generate LDS load code for vector dimensions
```

#### 4f. Global Write factorDim Dispatch (lines 13401–13415)

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

#### 4g. Global Write factorDims Bitfield Computation (lines 14340–14365)

Extended to compute `factorDims` from the bitfield when `FactorDim >= 4`:

```python
if kernel["ProblemType"]["UseScaleAlphaVec"] or kernel["ProblemType"]["UseBias"]:
    useScaleAlphaVec = kernel["ProblemType"]["UseScaleAlphaVec"]
    if self.states.FactorDim >= 4:
        # Bitfield-based computation
        needDim0 = (useScaleAlphaVec & 1) or (useBias & 1)
        needDim1 = (useScaleAlphaVec & 2) or (useBias & 2)
        if needDim0 and needDim1:
            factorDims = [0, 1]
        elif needDim1:
            factorDims = [1]
        else:
            factorDims = [0]
        if useScaleAlphaVec & 4:
            factorDims.append(2)
```

#### 4h. SRD Rename: allocPostLoopSrdSuppressRaw Usage

Changed from `allocPostLoopSrdSuppress("ScaleAlphaVec", ...)` (which constructs `SrdScaleAlphaVec`) to `allocPostLoopSrdSuppressRaw("DeviceAlpha", "ScaleAlphaVec", ...)` (which constructs `SrdDeviceAlpha` for the SRD but still uses `AddressScaleAlphaVec` for the base address).

#### 4i. addVectorGlobalLoad Call Site (line 15909)

Changed SRD name parameter from `"ScaleAlphaVec"` to `"DeviceAlpha"`:

```python
# Before:
self.addVectorGlobalLoad(kernel, "ScaleAlphaVec", ...)
# After:
self.addVectorGlobalLoad(kernel, "DeviceAlpha", ...)
```

#### 4j. addVectorLocalStore Enhancement (lines 15737–15744)

Added optional `srdName` parameter to decouple the SRD name from the address name:

```python
# Before:
def addVectorLocalStore(self, kernel, addressStr, ...):
    # Used addressStr for both Address{addressStr} and Srd{addressStr}

# After:
def addVectorLocalStore(self, kernel, addressStr, ..., srdName=None):
    srdStr = srdName if srdName else addressStr
    # Uses addressStr for Address{addressStr}, srdStr for Srd{srdStr}
```

Call site updated:
```python
self.addVectorLocalStore(kernel, "ScaleAlphaVec", ..., srdName="DeviceAlpha")
```

#### 4k. SGPR Free/Undefine (lines 14271–14286)

Updated references from `SrdScaleAlphaVec` to `SrdDeviceAlpha`:

```python
module.add(self.setSgprToFreeState("SrdDeviceAlpha"))   # was SrdScaleAlphaVec
module.add(self.undefineSgpr("SrdDeviceAlpha"))          # was SrdScaleAlphaVec
```

### 5. `Tensile/Components/GlobalWriteBatch.py`

#### 5a. Skip Vector Load for Scalar Mode (line 554)

When `factorDim == 2`, skip the buffer_load for scaleAlphaVec since the scalar is already in an SGPR:

```python
if self.kernel["ProblemType"]["UseScaleAlphaVec"] and isSingleKernel and self.factorDim != 2:
    # load scaleAlphaVec vector
```

#### 5b. `applyScaleAlphaScalar` Function (lines 1002–1019)

New function that multiplies all output elements by the device alpha scalar stored in `SgprDeviceAlphaScalar`:

```python
def applyScaleAlphaScalar(vecModule):
    # Convert int32 accumulators to float if needed
    for vi in range(0, self.gwvw):
        sumIdxV = self.ss.elementSumIdx[elementIdx] + vi
        if self.kernel["ProblemType"]["ComputeDataType"].isSingle():
            vgprIdx = sumIdxV - self.parentWriter.states.c.startVgprValu
            vecModule.add(VMulF32(dst=vgpr("ValuC+%d"%vgprIdx),
                                  src0=sgpr("SgprDeviceAlphaScalar"),
                                  src1=vgpr("ValuC+%d"%vgprIdx),
                                  comment="*= device alpha scalar"))
        elif self.kernel["ProblemType"]["ComputeDataType"].isInt32():
            # Similar with VMulLOU32
```

#### 5c. Dispatch Between Scalar and Vector Apply (lines 1084–1089)

```python
if self.factorDim == 2:
    applyScaleAlphaScalar(vecModule)
else:
    applyScaleVec(vecModule, addressStr, dataScaleVec, factorDim, ...)
```

### 6. `Tensile/AsmStoreState.py`

#### 6a. referenceVgprDim Extended (lines 226–230)

Extended from 2 slots to 3 slots. The third slot (index 2) is for scalar mode and always empty (no VGPRs needed):

```python
self.referenceVgprDim = [[], [], []]  # was [[], []]
```

Added guard to skip VGPR allocation for ScaleAlpha when `factorDim >= 2`:

```python
if factorDim < 2:
    # allocate VGPRs for ScaleAlpha vector load
```

### 7. `Tensile/AsmAddressCalculation.py` — Lines 65, 68

Extended offset arrays from 2 to 3 slots:

```python
self.biasOffset = [0, 0, 0]            # was [0, 0]
self.scaleAlphaVecOffset = [0, 0, 0]   # was [0, 0]
```

### 8. `Tensile/KernelWriterConversion.py` — Lines 139, 693–703

Changed `== 3` to `>= 3` for enableFactorDim, and added factorDim==2 scalar branch in the conversion kernel (HIP C++ code generation):

```cpp
}else if(arg.factorDim == 2){
    // scalar mode: multiply all elements by ScaleAlphaVec[0]
    for(int vIdx = 0; vIdx < gwvw; vIdx++){
        accum[vIdx] *= (float)arg.ScaleAlphaVec[0];
    }
}
```

### 9. `src/ContractionSolution.cpp`

#### 9a. factorDim Argument Guard (line 905)

```cpp
// Before:
if (useScaleAlphaVec == 3)
// After:
if (useScaleAlphaVec >= 3)
```

#### 9b. enableFactorDim Guard (line 1856)

```cpp
// Before:
if (useScaleAlphaVec == 3)
// After:
if (useScaleAlphaVec >= 3)
```

#### 9c. Kernel Naming (lines 2506–2510)

Extended naming logic for `factorDim >= 4` to compute factorDims from the bitfield.

### 10. `client/src/Reference.cpp`

#### 10a. GSU Path factorDim==2 Branch (lines 1441–1443)

```cpp
else if(factorDim == 2)
{
    alpha *= shadowAlphaVec[0];   // scalar mode: use first element as scalar
}
```

#### 10b. Standard Path factorDim==2 Branch (lines 1837–1845)

Similar scalar multiplication branch in the non-GSU reference path.

#### 10c. ShadowBuffer vecLen for Scalar (line 1275)

When `factorDim == 2`, the shadow buffer uses `problem.freeSizeB(0)` for vector length (matches N-dim, but only element [0] is used).

### 11. `client/src/ClientProblemFactory.cpp` — Lines 277, 328, 453–456

Changed `== 3` to `>= 3` for factorDimSize computation and factorDim iteration:

```cpp
// Before:
if (useScaleAlphaVec == 3)
    factorDimSize = ...;
// After:
if (useScaleAlphaVec >= 3)
    factorDimSize = ...;
```

Also updated factorDim computation to handle values 4–7.

### 12. `Tensile/SolutionStructs/Solution.py` — Lines 4421–4430

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

## Bugs Found and Fixed

### Bug 1 (Critical): Wrong Branching Order in Global Write Dispatch

**Symptom**: UseScaleAlphaVec=7 with factorDim=2 produced fundamentally wrong results (opposite sign, wrong magnitude, e.g. GPU=-38.25 vs ref=82) with random A/B data. Passed with constant data.

**Root Cause**: In the global write section, both `checkFactorDimScalar` (checks `factorDim==2`) and `checkIsFactorDimZero` (checks `factorDim!=0`) were inserted using `pos=0`. Due to LIFO semantics of `pos=0` insertion, the `factorDim!=0` check was executing before the `factorDim==2` check. This meant `factorDim=2` was caught by the `factorDim!=0` branch and incorrectly routed to the N-vector (factorDim=1) code path instead of the scalar path.

**Fix**: Swapped the insertion order so `checkIsFactorDimZero` is inserted at `pos=0` first (ending up second in the stream), then `checkFactorDimScalar` is inserted at `pos=0` second (ending up first in the stream). This ensures the scalar check executes before the vector check.

**Verification Method**: Disassembled the kernel binary with `llvm-objdump -d --mcpu=gfx1250` and traced the branch instruction order, confirming the `s_cmp_eq_u32 s30, 2` (scalar check) now precedes the `s_cmp_eq_u32 s30, 0` (vector check).

### Bug 2: LDS Allocation Missing for UseScaleAlphaVec > 3

**Symptom**: `calcEpilogueTurns` in Solution.py used exact equality checks (`savDim == 1`, `savDim == 2`, `savDim == 3`), so values 4–7 resulted in zero LDS allocation for ScaleAlphaVec vectors.

**Root Cause**: The LDS size calculation didn't handle the new bitfield values.

**Fix**: Changed to `savVecBits = savDim & 3` to extract just the vector bits, then applied the same conditions on `savVecBits`.

**Impact**: This would have caused LDS corruption for vector paths when UseScaleAlphaVec=5 or 7 with factorDim=0 or 1. The scalar path (factorDim=2) doesn't use LDS so wasn't directly affected.

---

## SGPR Rename: SrdScaleAlphaVec → SrdDeviceAlpha

Renamed internal SGPR register names to better reflect the "device alpha" concept:

| Old Name | New Name | Scope |
|----------|----------|-------|
| `SrdScaleAlphaVec` | `SrdDeviceAlpha` | KernelWriterAssembly.py (define, use, free, undefine) |
| `SgprScaleAlphaScalar` | `SgprDeviceAlphaScalar` | KernelWriterAssembly.py + GlobalWriteBatch.py |

**Not renamed** (tied to kernel argument API):
- `AddressScaleAlphaVec` — Kernel argument name, used in Signature.py, KernelWriter.py, LSU.py, and multiple KWA sections

**Implementation approach**:
- Used existing `allocPostLoopSrdSuppressRaw(ch, chAddress, ...)` which takes separate SRD name (`ch`) and address name (`chAddress`), allowing `SrdDeviceAlpha` to be set up from `AddressScaleAlphaVec`
- Changed `addVectorGlobalLoad` call site from `"ScaleAlphaVec"` to `"DeviceAlpha"` (it constructs `Srd{name}`)
- Added optional `srdName` parameter to `addVectorLocalStore` to decouple SRD name from address name; call site passes `srdName="DeviceAlpha"`

---

## Test Results

All tests pass with random A/B data initialization:

| UseScaleAlphaVec | FactorDimArgs | factorDim=0 | factorDim=1 | factorDim=2 |
|------------------|---------------|-------------|-------------|-------------|
| 3 (baseline) | [0, 1] | PASSED | PASSED | N/A |
| 4 (scalar only) | [2] | N/A | N/A | PASSED |
| 5 (M-vec + scalar) | [0, 2] | PASSED | N/A | PASSED |
| 7 (all modes) | [0, 1, 2] | PASSED | PASSED | PASSED |

Test configuration:
- GPU: gfx1250 (ISA: [[12, 5, 0]])
- Data types: BF16 input, BF16 output, FP32 compute
- Matrix: 128×128×1×64 (M×N×Batch×K)
- Wave size: 32
- MatrixInstruction: [16, 16, 32, 1, 1, 2, 1, 1, 2]
- Random initialization for A, B, C matrices and scaleAlphaVec

---

## Test YAML Files Created

| File | UseScaleAlphaVec | FactorDimArgs | Purpose |
|------|------------------|---------------|---------|
| `test_scale_alpha_scalar_debug.yaml` | 7 | [0, 1, 2] | Full 3-way test |
| `test_scale_alpha_scalar_only.yaml` | 4 | (none) | Scalar-only mode |
| `test_scale_alpha_scalar_5.yaml` | 5 | [0, 2] | M-vector + scalar |
| `test_scale_alpha_scalar_7_fd2only.yaml` | 7 | [2] | All modes, scalar-only runtime |
| `test_scale_alpha_vec3.yaml` | 3 | [0, 1] | Baseline regression |

---

## Key Technical Details

### Assembly Flow for factorDim=2 (Scalar Path)

1. **Post-loop SRD setup**: Branch on `factorDim == 2`, set up minimal SRD (length 1), load scalar via `SLoadB32` into `SgprDeviceAlphaScalar`, wait with `SWaitCnt(kmcnt=0)` (gfx1250-specific: uses `kmcnt`, not `lgkmcnt`)
2. **readVectorToLDS**: Skipped entirely for scalar path (no vector to load)
3. **Global write dispatch**: `checkFactorDimValue(2)` branches to scalar-specific global write batch
4. **Global write batch**: Skips buffer_load for scaleAlphaVec, calls `applyScaleAlphaScalar` which broadcasts `SgprDeviceAlphaScalar` via `VMulF32` across all output VGPRs

### Critical Architecture Insight: pos=0 Insertion Semantics

`module.add(instructions, pos=0)` inserts at position 0 (beginning) of the module. When multiple items use `pos=0`:
- The **first** `pos=0` insert goes to position 0
- The **second** `pos=0` insert pushes the first one down, taking position 0 itself
- Result: **LIFO order** — the last inserted item executes first

This is critical for branching logic where execution order matters. The factorDim=2 check must execute before the factorDim!=0 check to prevent the scalar value from being caught by the vector branch.
