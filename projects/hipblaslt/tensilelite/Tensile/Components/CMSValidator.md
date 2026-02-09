# CMSValidator - Custom Memory Scheduling Validator

This document describes the CMSValidator module, which validates Custom Memory Scheduling (CMS) configurations for GPU kernel code generation in TensileLite.

## Overview

The CMSValidator ensures that custom memory scheduling configurations are correct before kernel code is generated. It validates timing constraints, instruction ordering, and memory access patterns to prevent runtime errors in GPU kernels.

The validator operates on a **Timeline** abstraction that represents instruction scheduling across multiple loop iterations:
- **ML-1**: Previous main loop iteration
- **ML**: Current main loop iteration
- **NGL**: No Global Load loop (iteration N+1)
- **NLL**: No Local Load loop (iteration N+2)

## Core Data Structures

### ValidatorInstruction (Abstract Base)

Base class for all instruction types with common fields:
- `name`: Instruction identifier
- `issued_at`: VMFMA index (with sub-index resolution for multiple instructions at same index)
- `done_idx()`: Index after which the instruction's result is available

### Instruction Types

| Type | Description |
|------|-------------|
| `LocalRead` | Reads data from LDS to VGPR (LRA0, LRB0, LRA1, LRB1, LRA3, LRB3) |
| `GlobalRead` | Reads data from global memory to LDS (GRA, GRB) |
| `Pack` | Data manipulation instructions preparing data for MFMA |
| `MFMA` | Matrix Fused Multiply-Add instructions |
| `SWait` | Wait instructions (SWaitCnt) that guarantee previous operations complete |
| `Barrier` | Synchronization barriers (SBarrier) across waves |
| `SNop` | No-op instructions for timing gaps |

## Validation Passes

The validator runs 8 passes for each code path. All must pass for a schedule to be considered valid.

---

### 1. `verify_correct_number_of_instructions`

**Purpose**: Ensures the schedule contains the expected number of each instruction type.

**How it works**:
- Compares the count of instructions in the schedule against the expected count from `context["idMap"]`
- Fails if any instruction type has a mismatched count

---

### 2. `verify_ascending_order`

**Purpose**: Ensures all instruction sequences are non-decreasing in their VMFMA indices.

**How it works**:
- For each instruction type (except Packs which have custom ordering rules)
- Verifies that each instruction's VMFMA index is >= the previous instruction's index
- This ensures instructions are issued in a valid order

**Example**:
```
GRIncA: [0, 1, 1, 3]  ✓ Valid (non-decreasing)
GRIncA: [0, 2, 1, 3]  ✗ Invalid (2 > 1)
```

---

### 3. `verify_lrs_finished_before_vmfma`

**Purpose**: Ensures LocalReads complete before the MFMA instructions that need their data.

**How it works**:
1. Creates a Timeline with LR and SYNC instructions
2. Sets `needed_by` field on each LR based on which MFMA consumes it:
   - LR0 data needed at MFMA index offset by `num_vmfma // 2` (second half of iteration)
   - LR1/LR3 data needed at MFMA index offset by `num_vmfma` (next iteration)
3. Applies SWaitCnt effects to set `guaranteed_by` on each LR
4. Validates that each LR is guaranteed before it's needed

**Key constraint**: `LR.guaranteed_by < LR.needed_by.issued_at`

---

### 4. `verify_packs_start_and_end_at_correct_indices`

**Purpose**: Validates Pack instruction timing for data preparation before MFMAs.

**How it works**:
1. Creates Timeline with Pack, LR, SYNC, and SNOP instructions
2. Hooks up Pack dependencies:
   - `must_start_after`: LR that provides the source data must be done
   - `needed_by`: MFMA that consumes the result
3. For TF32 emulation, additional constraints:
   - Pack grouping (24 packs per group for regular TF32, 10 for 4x4 MFMA TF32)
   - Middle-16 pair constraints (temporary register reuse)
   - Minimum quad-cycle gaps (section 7.6 of CDNA 4 ISA)

**Pack constraints validated**:
```
must_start_after.done_idx() < issued_at < needed_by.done_idx()
pair_consumer == next_scheduled_middle_16  (for TF32 middle-16 pairs)
min_quad_cycles <= estimated_quad_cycles   (for CVT instructions)
```

---

### 5. `verify_grs_not_too_early`

**Purpose**: Ensures GlobalReads don't overwrite LDS data before LocalReads finish reading it.

**How it works**:
1. GRs in iteration N write to LDS that LR0s of iteration N read from
2. Required ordering: `last LR0 → SWaitCnt → SBarrier → first GR`
3. Validates:
   - LR0 must be guaranteed done (via SWaitCnt) before GR issues
   - An SBarrier must exist between the SWaitCnt and GR (synchronizes all waves)

**Constraint**: GR.must_start_after is the last LR0 of the same type (respecting SwapGlobalReadOrder)

---

### 6. `verify_grs_finish_before_lrs`

**Purpose**: Ensures GlobalReads from previous iteration complete before corresponding LR1/LR3 instructions.

**How it works**:
1. Creates Timeline with GR, LR1/LR3, and SYNC instructions
2. Sets GR.needed_by to the first corresponding LR1/LR3 of the next iteration
3. Applies SWaitCnt effects to set GR.guaranteed_by
4. Applies SBarrier effects
5. Validates ordering: `GR → SWait → SBarrier → LR1`

**Required chain**: GR issues → SWaitCnt guarantees completion → SBarrier synchronizes waves → LR1 can safely read

---

### 7. `verify_scc_overlap`

**Purpose**: Ensures scalar instructions that modify SCC (Status/Condition Code) don't overlap improperly.

**How it works**:
- GRInc (Global Read Increment) instructions use multiple scalar ops that depend on SCC
- These form distinct intervals where SCC must not be modified:
  - `s_cmp_eq_u32, s_cselect_b32, s_cselect_b32` (3 instructions)
  - `s_add_u32, s_addc_u32` (2 instructions)
  - `s_sub_u32, s_subb_u32` (2 instructions, or just 1 without ShadowLimit)
  - `s_cmp_eq_u32, s_cselect_b32` (2 instructions)

**Checked for conflicts**:
- GRIncA vs GRIncB
- GRIncA/B vs GRA/GRB (when DirectToLds enabled)
- GRIncA/B vs LWSA/LWSB (Local Write Store)

---

### 8. `verify_gr_inc_order`

**Purpose**: Ensures GRInc instructions complete before their corresponding GlobalReads.

**How it works**:
- Validates `max(GRIncA) < min(GRA)` (and similarly for B)
- Respects SwapGlobalReadOrder: when enabled, GRIncA must finish before GRB

---

## Timeline Construction

The Timeline class creates a unified view of instruction scheduling:

1. **Instruction Population**: Deep copies instructions into 4 loops (ML-1, ML, NGL, NLL)
2. **Index Resolution**: Assigns sub-indices for multiple instructions at same VMFMA index
   - E.g., `[GRA, SWaitCnt, SBarrier, LRA1]` at index 5 → `[5.0, 5.25, 5.5, 5.75]`
3. **Linearization**: Creates searchable lookup tables for instructions by name and loop

## MFMA Reordering Support

When `mfmaReorder` is specified, the validator accounts for execution order differences:
- `mfmaReorder[new_position] = original_position`
- Uses `invert_mfma_reorder()` to find when a logical MFMA actually executes
- Affects LR→MFMA and Pack→MFMA dependency calculations

## TF32 Emulation Support

Special handling for TF32 (BF16 emulation of FP32):

### Regular TF32 (24 packs per group)
- First 4 packs: `v_cvt_pk_bf16_f32` (bf16 approximations)
- Middle 16 packs: `v_cvt_f32_bf16` + `v_sub_f32` pairs (error terms)
- Last 4 packs: `v_cvt_pk_bf16_f32` (pack error terms)

### 4x4 MFMA TF32 (10 packs per group)
- First 4 packs: CVT0 instructions
- Middle 2 packs: `v_mfma_f32_4x4x4_16b_bf16` instructions
- Last 4 packs: CVT1 instructions

### Quad-Cycle Timing
Per CDNA 4 ISA section 7.6:
- CVT packs need 2 quad-cycles before MFMA can use results
- 4x4 MFMAs need 5 quad-cycles before CVT1 can use results

## Entry Point

```python
def isValid(scheduleInfo: 'ScheduleInfo', context: dict) -> tuple[bool, str]:
```

- Returns `(True, "")` if all validations pass
- Returns `(False, "error message")` if any validation fails
- Can be explicitly disabled via `scheduleInfo.isValidationDisabled()`

## Validation Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         isValid()                                │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
              ┌───────────────────────────────┐
              │  For each code_path (0..N):   │
              └───────────────────────────────┘
                              │
    ┌─────────────────────────┼─────────────────────────┐
    ▼                         ▼                         ▼
┌────────┐              ┌────────────┐           ┌────────────┐
│ Pass 1 │──────────────│  Pass 2-8  │───────────│   ...      │
└────────┘              └────────────┘           └────────────┘
    │                         │                         │
    ▼                         ▼                         ▼
  Fail?──────────────────── Fail?────────────────── Fail?
    │ Yes                     │ Yes                     │ Yes
    ▼                         ▼                         ▼
  Return                    Return                    Return
  (False, msg)              (False, msg)              (False, msg)
    │ No                      │ No                      │ No
    └─────────────────────────┴─────────────────────────┘
                              │
                              ▼
                    Return (True, "")
```
