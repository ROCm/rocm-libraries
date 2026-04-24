# CMS Validator Major Refactor: idMap-Driven Instruction Resolution

## Problem Statement

The CMS Validator conflates instruction **names** (schedule keys like `"PackA0"`) with instruction **kinds** (the actual assembly instruction being scheduled). Every `ValidatorInstruction` carries a `name` field set to the schedule key, but multiple physically different assembly instructions share the same name. For example, `"PackA0"` can be any of:

- `v_perm_b32` (BF16 pack)
- `v_cvt_pk_bf16_f32` (TF32 CVT0/CVT1)
- `v_cvt_f32_bf16` + `v_sub_f32` (TF32 middle-16 pairs)
- `v_dot2c_f32_bf16` (alternative middle-16)
- `v_swap_b32` (register transpose for VW > 1)
- `v_mfma_f32_4x4x4_16b_bf16` (4x4 MFMA in TF32 emulation)

The validator distinguishes these only by **positional index** within the flat pack list, using hardcoded constants (`PACK_GROUP_SIZE_TF32 = 24`, `TF32_MIDDLE_16_START = 4`, etc.) and kernel emulation flags. This is brittle: if code generation ever changes the order or composition of pack instructions, the validator silently produces wrong results.

The same pattern appears in GlobalRead handling, where even/odd index checks (`idx_GR % 2 == 0: continue`) are used to distinguish m0 pointer writes from actual buffer loads.

Meanwhile, the `idMap` — built in `CustomSchedule.py` — already contains the actual rocisa instruction objects at each index. The validator could inspect these instead of guessing from position.

### Concrete Brittleness Examples

1. `_populate_instructions` (CMSValidator.py:648-676): Uses `idx_pack % PACK_GROUP_SIZE_TF32` to decide between CVTPack, MiddlePack, etc. If the code generator adds or removes an instruction from a group, every subsequent index is wrong.

2. `_hook_up_packs_f32` (CMSValidator.py:1535-1548): Hardcodes a 24-entry dependency graph `{0: [], 1: [], ..., 23: [17, 19]}` mapping pack indices to their dependencies. Cannot adapt if group composition changes.

3. `_hook_up_packs_f32_mfma` (CMSValidator.py:1660-1671): Same pattern, hardcoded 10-entry dependency graph for 4x4 MFMA groups.

4. GR even/odd hack (CMSValidator.py:643): `if idx_GR % 2 == 0: continue` — assumes m0 writes and buffer loads always alternate starting at even indices.

## Goals

1. **Use idMap to determine instruction kind** — no positional arithmetic, no kernel-flag branching for type selection.
2. **Preserve index-level error reporting** — error messages must still say which schedule name and index is wrong, but now also include the actual assembly instruction type (e.g., `"PackA0 idx=18 (v_cvt_pk_bf16_f32) issued too early"`).
3. **Derive dependencies from register operands** — replace hardcoded `pack_dependencies` dicts with operand-based analysis.
4. **Thread idMap everywhere** — including unit tests, which must construct mock idMaps.
5. **All instruction types** — Pack, GlobalRead, SYNC, SNOP, LocalRead, GRInc.

## Non-Goals

- Changing the schedule format (the YAML or optSchedule dict structure stays the same).
- Changing how CMS schedules are authored or registered.
- Changing the code generation in LocalRead.py or KernelWriter.py.

---

## Implementation Status

| Stage    | Description                                        | Status          | Notes                                                                          |
|----------|----------------------------------------------------|-----------------|--------------------------------------------------------------------------------|
| **00a**  | Eliminate TimedPack                                | **Done**        | Moved timing fields into Pack, deleted TimedPack class                         |
| **00b**  | Unify `needed_by` type                             | **Already done** | GlobalRead.needed_by was already `ValidatorInstruction`                        |
| **00c**  | Typed context (`ValidationContext` dataclass)      | **Done**        | Replaced `context: dict` and absorbed `ValidatorPassContext`                   |
| **00d**  | Document limitations                               | **Done**        | Created `CMSValidator_LIMITATIONS.md`                                          |
| **01**   | Thread idMap/mfmaCode, add `rocisa_inst`           | **Done**        | Bug fixed by review: mfma_code indexing with mfmaReorder                       |
| **02**   | Type resolution (PACK_TYPE_MAP, resolve_pack_type) | **Done**        | Bug fixed by review: detect_pack_groups CVT1→CVT0 boundary                    |
| **03**   | Register utilities                                 | **Done**        | `get_reg_range`, `reg_ranges_overlap`, `get_dst_range`, `get_src_ranges`       |
| **04**   | Register-based pack dependency derivation          | **Done**        | `derive_pack_must_start_after` reduces to latest dep; dual-path validated      |
| **05**   | Switch `_populate_instructions` to idMap-driven    | **Done**        | Positional fallback kept as else branch (removed fully in 07)                  |
| **06**   | Register-based LR→MFMA needed_by                   | **Done**        | `set_lr_needed_by_from_mfma_operands` — not yet wired into validation loop     |
| **07**   | Make idMap/mfmaCode mandatory, delete dead code    | **Done**        | All fallbacks removed, mock test infra built, all 421 tests pass               |
| **08**   | Register-based Pack→MFMA needed_by                 | **Done**        | `set_pack_needed_by_from_mfma_operands` — not yet wired into validation loop   |
| **09**   | Tile-math function deletion                        | **Blocked**     | must_start_after dual-path validated; see Known Issues 1, 3, 6 for remaining   |
| **10**   | SWaitCnt counter verification                      | **Done**        | `verify_swaitcnt_counters` — not yet wired as ValidatorPass                    |
| **11**   | Multi-ISA framework (ValidationConcern, catalog)   | **Done**        | Wired into `isValid`: `ValidationRule`/`StructuralRule` dispatch via `active_concerns()` |
| **12**   | gfx1151 rules                                      | **Not started** | Unblocked: needs `LWAfterLRRule`, `LWBeforeLRRule`, `GRVgprReadyRule`          |
| **13**   | findValidPositions query                           | **Done**        | Brute-force query wrapping isValid                                             |
| **14**   | PipelineStage model for PGR                        | **Done**        | `PipelineStage`, `build_pipeline_stages()` — not yet wired into Timeline       |
| **15**   | Split into modules                                 | **Not started** | Deferred: do after functional changes stabilize                                |
| **16**   | Separate Timeline                                  | **Not started** | No dependency on 15                                                            |

### Known Issues

1. **Mock test infrastructure does not scale to multiple ISAs**: The current `cms_test_utils.py` mock builders (`make_mock_id_map`, `make_mock_mfma_code`) use hardcoded CDNA 4 rocisa instruction types (VPermB32, VCvtPkF32toBF16, MFMAInstruction, BufferLoadB128, etc.) and hardcoded register layouts. When gfx1151 WMMA schedules or future ISAs are added, those schedules have different instruction types (WMMA instructions, different pack patterns, DTL=0 with explicit LW instructions, different GR patterns without m0 pointer interleaving) that the mocks don't handle. The mock builders are a parallel reimplementation of kernel writer output logic — they will drift as new ISAs are added. See "Future Work: Mock Infrastructure Scalability" section below.

2. **Mock register layouts don't connect LR.dst → Pack.src**: The mock instructions in `cms_test_utils.py` create LR and Pack objects with register ranges that don't form proper def-use chains. `derive_pack_must_start_after` returns empty dependency lists for 89 of 421 tests because it can't trace the LR→Pack register connection. This prevents replacing the positional `_hook_up_packs_*` functions with the register-based `derive_pack_must_start_after`. Must be fixed before stage 09 can delete the positional must_start_after code.

3. **`set_lr_needed_by_from_mfma_operands` doesn't propagate through pack chains**: The function builds `pack_to_mfma` starting from direct MFMA consumers but never propagates backwards through the Pack chain (LR→Pack→MFMA). It only finds LR→MFMA connections when LR.dst directly maps to an MFMA source register, which doesn't happen for BF16/TF32 data paths where packing is involved. The function is effectively a no-op for all current test cases. Must be fixed before stage 09 can delete `set_lr_needed_by_for_VMFMA`.

4. **`set_pack_needed_by_from_mfma_operands` doesn't trace through intermediate packs**: For TF32 4x4 paths, CVT0→MFMAPack→CVT1→MFMA is a multi-hop chain. The function only matches Pack.dst against direct MFMA.a/b registers. CVT0 packs (whose needed_by should be the MFMAPack) are not found because MFMAPacks are explicitly skipped. Must be fixed before stage 09 can delete `_set_pack_needed_by`.

5. **PipelineStage model (14) not yet wired into Timeline**: `build_pipeline_stages()` and `PipelineStage` are implemented but the Timeline still uses the hardcoded `[ML-1, ML, NGL, NLL]` loop list and string-based `_should_add`.

---

## Design

### 1. Type Resolution via idMap Inspection

A mapping from rocisa instruction types to ValidatorInstruction classes:

```python
from rocisa.instruction import (
    VPermB32, VCvtPkF32toBF16, PVCvtBF16toFP32, VCvtBF16toFP32,
    VSubF32, VDot2CF32BF16, VSwapB32, MFMAInstruction, SMovB32,
    SWaitCnt, SBarrier, SNop,
    VOrB32, VLShiftLeftOrB32,
)

# Maps rocisa instruction type -> (ValidatorInstruction class, human-readable assembly label)
#
# Lookup semantics:
#   1. Exact type(instruction) match.
#   2. If no exact match, isinstance fallback for subclass chains (e.g. MFMAInstruction subclasses).
#   3. If still no match, raise ValueError with the unknown type for immediate diagnosis.
PACK_TYPE_MAP: dict[type, tuple[type[ValidatorInstruction], str]] = {
    VPermB32:          (Pack,       "v_perm_b32"),
    VOrB32:            (Pack,       "v_or_b32"),
    VLShiftLeftOrB32:  (Pack,       "v_lshlrev_or_b32"),
    VCvtPkF32toBF16:   (CVTPack,    "v_cvt_pk_bf16_f32"),
    PVCvtBF16toFP32:   (MiddlePack, "p_v_cvt_f32_bf16"),
    VCvtBF16toFP32:    (MiddlePack, "v_cvt_f32_bf16"),
    VSubF32:           (MiddlePack, "v_sub_f32"),
    VDot2CF32BF16:     (MiddlePack, "v_dot2c_f32_bf16"),
    VSwapB32:          (SwapPack,   "v_swap_b32"),
    MFMAInstruction:   (MFMAPack,   "v_mfma_*"),
}
```

`_populate_instructions` consults this map for each entry in the idMap, replacing all positional branching:

```python
# Current (brittle):
if is_4x4mfma_tf32:
    if idx_pack < n_swaps:
        pack = SwapPack(name=name, ...)
    else:
        adjusted_idx = idx_pack - n_swaps
        idx_in_group = adjusted_idx % PACK_GROUP_SIZE_TF32_4X4
        ...

# New (robust):
rocisa_inst = idmap_items[idx_pack]
validator_cls, asm_label = resolve_type(rocisa_inst, PACK_TYPE_MAP)
pack = validator_cls(name=name, asm_label=asm_label, issued_at=..., issue_index=idx_pack, ...)
```

For GlobalRead with DirectToLds, instead of the even/odd hack:

```python
# Current (brittle):
if idx_GR % 2 == 0:
    continue  # Skip m0 writes

# New (robust):
rocisa_inst = idmap_items[idx_GR]
if isinstance(rocisa_inst, SMovB32):
    continue  # This is an m0 pointer update, not a load
global_read = GlobalRead(name=name, asm_label=type(rocisa_inst).__name__, ...)
```

### 2. Enhanced Error Messages

Add an `asm_label: str` field to `ValidatorInstruction`:

```python
@dataclass
class ValidatorInstruction(ABC):
    name: str          # Schedule key, e.g. "PackA0"
    asm_label: str     # Assembly instruction, e.g. "v_cvt_pk_bf16_f32"
    issued_at: SchedulePosition
    ...
```

All error messages include both:

```
PackA0 idx=18 (v_cvt_pk_bf16_f32) issued too early, must be issued after idx=12
(because of PackA0 idx=5 (v_sub_f32) issued @ idx=12).
```

This gives CMS developers precise information about which physical instruction has the constraint violation.

### 3. Group Index Derivation from Type Sequence

Replace hardcoded group size constants with a function that walks the idMap type sequence and detects group boundaries:

```python
def detect_pack_groups(idmap_items: list, type_map: dict) -> list[PackGroupInfo]:
    """
    Walk the idMap entries for a pack name and identify group boundaries
    from the type pattern.

    Returns a list of PackGroupInfo, each containing:
      - group_index: int
      - instructions: list of (index, rocisa_inst, validator_cls, asm_label)

    Group detection rules:
      - SwapPack instructions at the start are ungrouped (group_index=None).
      - After swaps, groups begin at each CVTPack after a non-CVTPack (or at the first CVTPack).
      - Within a group, the type sequence determines the group kind:
        - [CVTPack, MiddlePack, CVTPack] -> TF32 group of 24
        - [CVTPack, MFMAPack, CVTPack]   -> 4x4 MFMA TF32 group of 10
        - [Pack only]                    -> BF16 (no grouping needed)
    """
```

This eliminates `PACK_GROUP_SIZE_TF32`, `PACK_GROUP_SIZE_TF32_4X4`, and all the `TF32_*_START`/`TF32_*_END` constants.

### 4. Dependency Derivation from Register Operands

Replace the hardcoded `pack_dependencies` dicts with operand-based dependency analysis.

rocisa instructions expose through Python bindings:
- `instruction.dst` → `RegisterContainer` with `.regIdx`, `.regNum`, `.regType`
- `instruction.srcs` → list of `InstructionInput` (can be `RegisterContainer` or literal)

The algorithm:

```python
def derive_dependencies(
    validator_insts: list[ValidatorInstruction],
    rocisa_insts: list,
    prior_producers: dict[int, ValidatorInstruction],  # reg_idx -> producer
) -> None:
    """
    For each instruction, inspect its source register operands and find
    which prior instruction wrote to those registers. Set must_start_after
    accordingly.

    Args:
        validator_insts: The ValidatorInstruction objects in issue_index order.
        rocisa_insts:    The corresponding rocisa instruction objects.
        prior_producers: Maps VGPR index -> the ValidatorInstruction that last wrote it.
                         Pre-populated with LocalRead producers for the group's input registers.
    """
    producers = dict(prior_producers)  # Copy, don't mutate caller's map

    for v_inst, r_inst in zip(validator_insts, rocisa_insts):
        # Find dependencies: which producers wrote to our source registers?
        deps = set()
        for src in _extract_register_containers(r_inst.srcs):
            for reg in range(src.regIdx, src.regIdx + int(src.regNum)):
                if reg in producers and producers[reg] is not v_inst:
                    deps.add(producers[reg])

        v_inst.must_start_after = list(deps)

        # Update producer map: this instruction now owns its destination registers
        dst = _extract_register_container(r_inst.dst)
        if dst:
            for reg in range(dst.regIdx, dst.regIdx + int(dst.regNum)):
                producers[reg] = v_inst
```

This handles all cases automatically:
- **CVT0 → LocalRead dependency**: CVT0's source registers were written by LocalReads. The `prior_producers` map is pre-populated with LR → register mappings.
- **Middle-16 → CVT0 dependency**: Middle-16 reads from registers that CVT0 wrote to.
- **CVT1 → Middle-16/MFMAPack dependency**: CVT1 reads from registers that middle-16 or MFMAPack wrote to.
- **SwapPack → LocalRead dependency**: SwapPack sources are registers loaded by LRs.
- **MiddlePack pair constraint**: Two consecutive MiddlePacks sharing a temp register means the second one reads the register the first one wrote. This falls out naturally from the register overlap analysis.

#### What the Hardcoded Graphs Looked Like (for reference)

TF32 groups of 24 (`_hook_up_packs_f32`):
```python
pack_dependencies = {
    0: [], 1: [], 2: [], 3: [],           # CVT0: depend on LRs only
    4: [0], 5: [4], 6: [0], 7: [6],       # Middle-16 pairs
    8: [1], 9: [8], 10: [1], 11: [10],
    12: [2], 13: [12], 14: [2], 15: [14],
    16: [3], 17: [16], 18: [3], 19: [18],
    20: [17, 19],                          # CVT1: depend on middle-16
    21: [13, 15, 20],
    22: [9, 11, 21],
    23: [5, 7, 22],
}
```

4x4 MFMA TF32 groups of 10 (`_hook_up_packs_f32_mfma`):
```python
pack_dependencies = {
    0: [], 1: [], 2: [], 3: [],   # CVT0
    4: [0, 1], 5: [2, 3],        # MFMAPack
    6: [5], 7: [5, 6],           # CVT1
    8: [4, 7], 9: [4, 8],
}
```

All of these will be derived from register operands instead.

### 5. Changes to Data Flow

#### Current Flow

```
CustomSchedule.py:
  idMap = { "PackA0": PackCodeA[0], "GRA": globalReadA, ... }
  context = {"kernel": kernel, "idMap": idMap}
  cmsv.isValid(scheduleInfo, context)

CMSValidator.py:
  isValid(scheduleInfo, context):
    # idMap used ONLY in verify_correct_number_of_instructions for count check
    # Timeline._populate_instructions does NOT receive idMap
    # Instead uses positional arithmetic + kernel flags
```

#### New Flow

```
CustomSchedule.py:
  idMap = { "PackA0": PackCodeA[0], "GRA": globalReadA, ... }
  context = {"kernel": kernel, "idMap": idMap}
  cmsv.isValid(scheduleInfo, context)

CMSValidator.py:
  isValid(scheduleInfo, context):
    idMap = context["idMap"]
    # idMap flows into Timeline constructor
    timeline = create_unified_timeline(scheduleInfo, kernel, codePath, idMap)

  Timeline.__init__(..., idMap):
    self.idMap = idMap

  Timeline._populate_instructions(...):
    # For each schedule key, get the rocisa instructions from self.idMap
    # Use type inspection to create the correct ValidatorInstruction subclass
    # No positional arithmetic, no kernel flag branching for type selection

  hook_up_packs(timeline, kernel, mfma_reorder):
    # For each pack group, derive dependencies from register operands
    # using the rocisa instructions stored alongside validator instructions
```

### 6. ValidatorInstruction Changes

```python
@dataclass
class ValidatorInstruction(ABC):
    name: str              # Schedule key (e.g. "PackA0") — UNCHANGED
    asm_label: str         # NEW: assembly instruction name (e.g. "v_cvt_pk_bf16_f32")
    issued_at: SchedulePosition
    min_issue_quad_cycles_base: ClassVar[int] = 1
```

All subclasses gain `asm_label`. The `validate()` methods include `asm_label` in error strings.

#### Pack Subclass Changes

```python
@dataclass
class Pack(ValidatorInstruction):
    issue_index: int
    group_index: Optional[int] = None      # Now computed by detect_pack_groups, not positional math
    rocisa_inst: Optional[object] = None   # NEW: reference to the rocisa instruction for operand inspection
    needed_by: ...
    must_start_after: ...
```

The `rocisa_inst` field is stored so that `hook_up_packs` can inspect operands without needing a separate data structure.

### 7. Unit Test Infrastructure

Unit tests must construct mock idMaps. A helper module provides:

```python
def make_mock_pack_idmap_bf16(n_packs: int) -> list:
    """Create a list of mock VPermB32 instructions with sequential register assignments."""

def make_mock_pack_idmap_tf32(n_groups: int) -> list:
    """Create a list of mock instructions matching the TF32 group pattern:
    [VCvtPkF32toBF16 x4, PVCvtBF16toFP32/VSubF32 x16, VCvtPkF32toBF16 x4] per group.
    Each mock instruction has proper dst/srcs RegisterContainers with register indices
    that produce the correct dependency graph when analyzed by derive_dependencies."""

def make_mock_pack_idmap_tf32_4x4(n_groups: int, n_swaps: int = 0) -> list:
    """Same for 4x4 MFMA TF32: [VSwapB32 x n_swaps] + 
    [VCvtPkF32toBF16 x4, MFMAInstruction x2, VCvtPkF32toBF16 x4] per group."""

def make_mock_gr_idmap_dtl(n_loads: int) -> list:
    """Create interleaved [SMovB32, BufferLoad] pairs for DirectToLds GlobalReads."""

def make_mock_sync_idmap(syncs: list) -> list:
    """Wrap SWaitCnt/SBarrier objects for SYNC idMap entries."""
```

The existing `cms_validation_base.py` test infrastructure will be updated to accept and pass idMap through to `create_unified_timeline`.

### 8. Constants and Code to Delete

The following become dead code after this refactor:

```python
# CMSValidator.py — DELETE these constants:
PACK_GROUP_SIZE_TF32 = 24
PACK_GROUP_SIZE_TF32_4X4 = 10
TF32_CVT0_END = 4
TF32_MIDDLE_16_START = 4
TF32_MIDDLE_16_END = 20
TF32_4X4_MFMA_START = 4
TF32_4X4_MFMA_END = 6

# CMSValidator.py — DELETE these hardcoded dependency graphs:
# In _hook_up_packs_f32: pack_dependencies dict (lines 1535-1548)
# In _hook_up_packs_f32_mfma: pack_dependencies dict (lines 1660-1671)

# CMSValidator.py — DELETE/REPLACE:
# _compute_swap_pack_count (no longer needed — SwapPacks identified by type)
# The is_tf32_emulation / is_4x4mfma_tf32 branching in _populate_instructions for Pack
# The even/odd GR index check (idx_GR % 2 == 0: continue)
```

### 9. What Stays the Same

- **Schedule format**: `optSchedule` dict structure unchanged. CMS authors still write `PackA0: [5, 5, 7, 9, ...]`.
- **Schedule names**: `"PackA0"`, `"GRA"`, `"SYNC"`, etc. — unchanged.
- **idMap construction** in `CustomSchedule.py` — unchanged (it already builds the right data).
- **Structural checks** (`verify_correct_number_of_instructions`, `verify_ascending_order`, `verify_scc_overlap`) — unchanged, they already work correctly.
- **Timeline loop structure** (ML-1, ML, NGL, NLL) — unchanged.
- **MFMA handling** — MFMAs are explicitly added to the timeline, not from idMap. Unchanged.
- **`_should_add` filtering** — still uses name strings for loop membership. Unchanged.
- **Quad-cycle timing constants** (`QUAD_CYCLES_CVT_BEFORE_MFMA`, etc.) — these are ISA constraints, not layout assumptions. Unchanged.
- **MFMA type-switch thresholds** — unchanged.
- **`_set_pack_needed_by`** — links packs to their consuming MFMA. This uses `group_index` and `isinstance` checks that will continue to work correctly with the new group detection.
- **MiddlePack pair constraint** — the `pair_consumer` / `next_scheduled_middle_16` mechanism stays. Pairs are still identified (two consecutive MiddlePacks in a group), but now from the type sequence rather than positional arithmetic.

### 10. Migration and Backwards Compatibility

- No backwards compatibility shims. This is a clean refactor.
- The idMap is always present during real kernel generation (it's built in `customMainLoopSchedule` before calling `isValid`).
- Unit tests are updated to provide mock idMaps.
- The `if "idMap" not in context` guard in `verify_correct_number_of_instructions` can be tightened or removed once all tests provide idMaps.

---

## Execution Order

Each step below maps to a branch that can be reviewed and landed independently (or as a stack where noted). Steps are ordered by the split-branch landing convention: groundwork first, then refactors, then cleanup, then features. Tests travel with the code they test.

This plan incorporates items from `CMSValidator_Refactoring_Plan.md` and `docs/CMSValidatorConstraints.md`. Those documents' items map to branches as follows:

| Refactoring Item | Disposition |
|---|---|
| R14: Eliminate TimedPack | Branch 00a — lands first, simplifies hierarchy |
| R13: Unify `needed_by` type | Branch 00b — lands second, standardizes interfaces |
| R5: Typed Context | Branch 00c — lands third, replaces `context: dict` |
| R11: Test Infrastructure | Absorbed into 01-plumbing |
| R6: Registry Pattern for Packs | **Superseded** — `derive_pack_dependencies` (branch 04) replaces all three `_hook_up_packs_*` with register-based tracing; the registry pattern is not needed |
| R9: Clarify Validation Logic | Already done — passes set constraints, `validate_timeline()` delegates to instruction `validate()` methods |
| R12: Document Limitations | Branch 00d — standalone, anytime |
| R1: Split File Into Modules | Branch 15 — do after functional changes stabilize |
| R10: Separate Timeline | Branch 16 — do after file split |
| `findValidPositions()` query | Branch 13 — feature, stacked on 01-plumbing. From `docs/CMSValidatorConstraints.md` |
| PGR generalization | Branch 14 — refactor, stacked on 11-multi-isa-framework. From `CMSVALIDATORPGR.md` |

```
Dependency DAG:

  base (develop)
  ├── 00a-eliminate-timedpack    (cleanup, sibling)
  ├── 00b-unify-needed-by       (refactor, sibling)
  ├── 00c-typed-context          (refactor, sibling)
  ├── 00d-document-limitations   (docs, sibling)
  ├── 01-plumbing                (groundwork, merges 00a+00b+00c) ◆ diamond
  │   ├── 02-type-resolution     (refactor, stacked on 01)
  │   │   └── 05-populate-cutover  (refactor, stacked on 02)
  │   │       └── 07-dead-code-deletion  (cleanup, stacked on 05)
  │   ├── 03-register-utilities  (groundwork, stacked on 01)
  │   │   └── 04-pack-deps-from-registers  (refactor, stacked on 03, merges 02) ◆ diamond
  │   │       ├── 06-lr-needed-by-from-registers  (refactor, stacked on 04)
  │   │       │   └── 08-pack-needed-by-from-registers  (refactor, stacked on 06)
  │   │       │       └── 09-tile-math-deletion  (cleanup, stacked on 08, merges 07) ◆ diamond
  │   │       └── 10-swaitcnt-verification  (feature, stacked on 04)
  │   ├── 11-multi-isa-framework  (groundwork, stacked on 01)
  │   │   ├── 12-gfx1151-rules    (feature, stacked on 11)
  │   │   └── 14-pgr-generalization  (refactor, stacked on 11)
  │   └── 13-valid-position-query  (feature, stacked on 01)
  ├── 15-split-into-modules      (refactor, after 09 lands)
  └── 16-separate-timeline       (refactor, stacked on 01)
  └── reconstruction             (verification)
```

### Branch descriptions

**00a-eliminate-timedpack** (cleanup, sibling)
Move `min_quad_cycles_before_result_used` and `estimated_quad_cycles_before_result_used` from `TimedPack` into `Pack` (default 0 = no constraint). Delete the `TimedPack` class. Re-parent `CVTPack(Pack)` and `MFMAPack(Pack, MFMA)`. Update `estimate_quad_cycles()` to check `isinstance(_, Pack)` instead of `isinstance(_, TimedPack)`. No behavior change — the `> 0` guard makes timing validation a no-op for packs that don't need it. (From CMSValidator_Refactoring_Plan.md R14.)

**00b-unify-needed-by** (refactor, sibling)
Change `GlobalRead.needed_by` from `SchedulePosition` to `ValidatorInstruction` (matching `LocalRead` and `Pack`). Update `set_gr_needed_by_from_lrs()` to assign the LR instruction object rather than its `issued_at`. Update `GlobalRead._validate_needed_by()` to use `self.needed_by.issued_at` and `self.needed_by.name`. (From CMSValidator_Refactoring_Plan.md R13.)

**00c-typed-context** (refactor, sibling)
Replace the `context: dict` parameter in `isValid()` and structural checks with a `ValidationContext` dataclass. Add `from_dict()` class method for backward compatibility during transition. Merge `ValidatorPassContext` into `ValidationContext` or make it a sub-view. Add convenience properties (`swap_global_read_order`, `direct_to_lds`, `n_tiles_a`, etc.) to avoid scattered `kernel.get(...)` calls. (From CMSValidator_Refactoring_Plan.md R5.)

**00d-document-limitations** (docs, sibling)
Create `CMSValidator/LIMITATIONS.md` documenting unsupported configurations, validation gaps, false negatives/positives, and architecture-specific behavior. Consolidate scattered TODOs. (From CMSValidator_Refactoring_Plan.md R12.)

**01-plumbing** (groundwork, merges 00a+00b+00c)
Pass `mfmaCode` into the validator alongside `idMap`. Update `isValid` → `create_unified_timeline` → `Timeline.__init__` signatures. Both `idMap` and `mfmaCode` are mandatory fields on `ValidationContext`. Add `rocisa_inst` and `asm_label` fields to `ValidatorInstruction` and all subclasses. Populate `rocisa_inst` during `_populate_instructions` from `idMap` and `mfmaCode`. Update error messages to include `asm_label`. Build mock `idMap`/`mfmaCode` test infrastructure and update all existing unit tests to provide them. (Incorporates CMSValidator_Refactoring_Plan.md R11.)

**02-type-resolution** (refactor, stacked on 01)
Implement `resolve_type`, `PACK_TYPE_MAP`, and `detect_pack_groups`. Initially dual-path: run type resolution alongside existing positional logic with assertions that both agree. Does not yet switch `_populate_instructions` — the dual-path validation runs in parallel with the old code.

**03-register-utilities** (groundwork, stacked on 01)
Implement `_get_vgpr_range` and `_vgpr_ranges_overlap` helpers for numeric register range extraction and overlap detection. Unit tests against known register layouts from snapshot fixtures.

**04-pack-deps-from-registers** (refactor, stacked on 03, merges 02)
Implement `derive_pack_dependencies` — the single function replacing `_hook_up_packs_bf16`, `_hook_up_packs_f32`, and `_hook_up_packs_f32_mfma`. Dual-path: compute dependencies from both registers and existing hardcoded functions, assert agreement. Merges 02 because it needs type-resolved `ValidatorInstruction` subclasses to distinguish CVTPack/MiddlePack/MFMAPack. (Supersedes CMSValidator_Refactoring_Plan.md R6 — registry pattern not needed when dependencies are register-derived.)

**05-populate-cutover** (refactor, stacked on 02)
Switch `_populate_instructions` to use type resolution from `idMap`. Replace GR even/odd hack with `isinstance(rocisa_inst, SMovB32)`. Remove positional branching for Packs. Remove dual-path assertions from 02 (the type resolution is now the only path).

**06-lr-needed-by-from-registers** (refactor, stacked on 04)
Implement `set_lr_needed_by_from_mfma_operands` — traces LR.dst → Pack chain → MFMA.a/b to find the consuming MFMA. Dual-path validation against existing `set_lr_needed_by_for_VMFMA` / `lr_needed_by_mfma`.

**07-dead-code-deletion** (cleanup, stacked on 05) — **SCOPE EXPANDED**
As implemented: make `id_map` and `mfma_code` mandatory parameters (no Optional, no fallbacks). Remove all positional fallback else-blocks. Delete dead constants (`TF32_CVT0_END`, `TF32_MIDDLE_16_START/END`, `TF32_4X4_MFMA_START/END`) and `_compute_swap_pack_count`. Keep `PACK_GROUP_SIZE_TF32/4X4` (still used by `detect_pack_groups`). Create `cms_test_utils.py` with `make_mock_id_map()` and `make_mock_mfma_code()`. Update all 36 `isValid()` calls in tests. Add empty-data guards to `_hook_up_packs_f32` and `_hook_up_packs_f32_mfma`.

**08-pack-needed-by-from-registers** (refactor, stacked on 06)
Implement `set_pack_needed_by_from_mfma_operands` — scans `mfma_code` for the earliest MFMA consuming each pack's output. Dual-path validation against `_set_pack_needed_by` / `find_earliest_mfma_execution`.

**09-tile-math-deletion** (cleanup, stacked on 08, merges 07)
Remove dual-path assertions from 04, 06, 08. Delete replaced functions: `lr_needed_by_mfma`, `_transform_index_standard`, `_transform_index_with_force_unroll_sub_iter`, `find_earliest_mfma_execution`, `_set_pack_needed_by`, `_hook_up_packs_bf16`, `_hook_up_packs_f32`, `_hook_up_packs_f32_mfma`, `set_lr_needed_by_for_VMFMA`, `_get_lrs_for_pack`, `invert_mfma_reorder`.

**10-swaitcnt-verification** (feature, stacked on 04)
New validation pass: count actual `DSLoad*` and `BufferLoad*` instructions between SWaitCnts and verify `dscnt`/`vlcnt` values are correct. Additive — can be individually enabled/disabled.

**11-multi-isa-framework** (groundwork, stacked on 01)
Define `ValidationConcern` enum and `ISA_CONCERN_CATALOG`. Implement `active_concerns`. Convert existing passes to `ValidationRule` / `StructuralRule` objects. Replace `TIMELINE_PASSES`/`STRUCTURAL_CHECKS` dicts. Implement coverage check in `isValid`. Remove `disableValidationPass` machinery and `_disable_cdna4_only_passes_for_gfx1151`.

**12-gfx1151-rules** (feature, stacked on 11)
Add `LocalWrite` ValidatorInstruction. Write gfx1151-specific rules: `LWAfterLRRule`, `LWBeforeLRRule`, `GRVgprReadyRule`. Add gfx1151 concerns to catalog. gfx1151 schedules now get real validation.

**13-valid-position-query** (feature, stacked on 01)
Implement `findValidPositions(scheduleInfo, context, inst_name, inst_issue_idx)` — a brute-force query that returns the set of valid vmfma indices (as ranges) for moving a given instruction. For each candidate position in `[-1, numMfma-1]`, mutates `optSchedule` in-place, calls `isValid()`, and restores. Automatically benefits from all validator improvements since it delegates entirely to `isValid()`. Measured query times: 43-370ms for 256x256x64 16-bit (numMfma=128). Consumers: human developers tuning schedules and automated optimizers. Independent of all other tracks — only needs `isValid()` and `ValidationContext` from 01-plumbing. (From `docs/CMSValidatorConstraints.md`.)

**14-pgr-generalization** (refactor, stacked on 11)
Replace hardcoded 4-loop Timeline (`[ML-1, ML, NGL, NLL]`) with a `PipelineStage` model that generates the correct number of loops based on PGR. Add `build_pipeline_stages(pgr, nglshift, nllshift)` which produces: PGR=1 → 2 stages, PGR=2 → 4 stages (same as today), PGR=3 → 6 stages. Replace `_should_add` string checks with `PipelineStage` attribute checks (`has_global_reads`, `has_local_reads_lr0_only`). Replace the vlcnt_shift dict with per-stage shifts. Rewrite `nllvmcntHandling` in `CustomSchedule.py` to generate per-NGLL vlcnt branches using `remainPgr` instead of the 3-branch `useGR`/`usePLR`/`useGRInc` system. Replace MAINLOOP macro signature `(ID, useGR, usePLR, useGRInc, useLoop)` with `(ID, remainPgr)`. Update `noLoadLoopBody` in `KernelWriter.py` for per-NGLL dispatch. For PGR=2 this produces identical output. No PGR>=3 CMS schedules exist today — this is forward-looking. Stacks on 11-multi-isa-framework because both modify how loops are configured and the PipelineStage model should integrate with the ValidationRule dispatch. (From `CMSVALIDATORPGR.md`.)

**15-split-into-modules** (refactor, after 09 lands)
Split `CMSValidator.py` into a package: `instructions.py`, `timeline.py`, `constants.py`, `context.py`, `passes.py`, `utils/`. Create backward-compatible `CMSValidator.py` that re-exports the public API. (From CMSValidator_Refactoring_Plan.md R1. Deferred until after functional changes stabilize to avoid constant merge conflicts.)

**16-separate-timeline** (refactor, stacked on 01)
Extract `ScheduleParser` and `LoopManager` from `Timeline`. Timeline becomes an immutable query view over the loop manager. (From CMSValidator_Refactoring_Plan.md R10. No dependency on 15-split-into-modules — the separation makes the later file split easier, not the other way around.)

---

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| rocisa instruction types change or new types added | `resolve_type` raises `ValueError` for unknown types — fails loud at generation time, not silently at validation time |
| Register operand inspection produces different dependency graphs than hardcoded ones | Steps 7-9 use dual-path assertions to validate before cutting over. Any discrepancy is investigated before the hardcoded graphs are removed |
| Mock idMap instructions in tests don't faithfully represent real instruction patterns | Compare mock idMap type sequences against real ones from kernel generation runs |
| Performance of operand-based dependency derivation | The number of instructions per timeline is O(100-300). Register comparisons are numeric range overlaps. Total overhead is microseconds — negligible vs. kernel compilation (seconds) |
| Symbolic register names fail to resolve to numeric indices | `regName.getTotalIdx()` uses the `rocIsa` singleton's `vgprIdx` map, populated by the kernel writer before CMS runs. If resolution fails, it's a kernel writer bug — fail loud |
| TF32 emulation's 4x4 MFMAs in pack chains create a non-obvious register dependency path | The 4x4 MFMAs have `.a`, `.b`, `.acc`, `.acc2` just like regular MFMAs. Register tracing through CVT0.dst→4x4MFMA.a→4x4MFMA.acc→CVT1.src works identically. Validated by dual-path in step 7 |
| mfmaCode contains non-MFMA objects after removeComments | `removeComments` strips TextBlock and SNop. Add assertion: `assert all(isinstance(m, (MFMAInstruction, SMFMAInstruction, MXMFMAInstruction)) for m in mfmaCode)` |

---

## Future Work: Mock Infrastructure Scalability

### Problem

The current mock test infrastructure (`Tensile/Tests/unit/cms_test_utils.py`) creates mock rocisa instruction objects by hardcoding CDNA 4 instruction types and register layouts:

- **Pack mocks**: `VPermB32` for BF16, `VCvtPkF32toBF16` + `VSubF32` for TF32, `VCvtPkF32toBF16` + `MFMAInstruction` for 4x4 TF32
- **GR mocks**: interleaved `SMovB32`/`SAddU32` + `BufferLoadB128` for DTL=1, all `BufferLoadB128` for DTL=0
- **LR mocks**: `DSLoadB128` with sequential register indices
- **SwapPack mocks**: `VSwapB32` for VW > 1

This is a parallel reimplementation of what the kernel writer produces. It works for gfx950 CDNA 4 today but will not scale to:

- **gfx1151 (RDNA 3.5)**: Uses WMMA instructions (not MFMA), no packing (no VPermB32/VCvtPk), DTL=0 with explicit LW instructions, different GR patterns without m0 pointer interleaving
- **Future ISAs**: May introduce new instruction types, new pack patterns, new data paths

### Approaches to Investigate

1. **Snapshot fixtures**: Capture real `idMap` and `mfmaCode` from kernel generation runs (as done during investigation with `TENSILE_LOG_IDMAP`). Serialize to fixture files. Tests load fixtures. Pros: mocks match reality by construction. Cons: fixtures must be regenerated when kernel writer changes; fixtures are opaque binary blobs.

2. **Kernel writer export function**: Have the kernel writer expose a function that produces just the `idMap` (and `mfmaCode`) for a given kernel config without doing full code generation. Pros: always correct, no parallel implementation. Cons: requires kernel writer changes; may be slow for unit tests.

3. **Minimal type stubs**: The validator only inspects a few properties of idMap entries: `type()` for `resolve_pack_type`/`is_gr_load`, `.dst`/`.srcs`/`.a`/`.b` for register extraction, `.dscnt`/`.vlcnt`/`.vscnt` for SWaitCnt, `.getParams()` for SNop. Create a thin stub protocol that satisfies these checks without importing specific rocisa instruction classes. Pros: ISA-agnostic. Cons: stubs may diverge from real objects.

4. **Per-ISA mock registries**: Each ISA registers its mock builders, similar to how CMS schedules are registered. When a test needs a mock idMap for gfx1151, it calls the gfx1151-specific builder. Pros: cleanly extensible. Cons: still requires maintaining mock builders per ISA.

### Recommendation

Investigate option 2 (kernel writer export function) first — it's the only approach where the mock is guaranteed correct without maintenance. If that's too expensive for unit test speed, fall back to option 1 (snapshot fixtures) for the common cases and option 4 (per-ISA mock registries) for edge-case tests.

---

## rocisa-Driven Dependency Resolution and MFMA Plumbing

### Motivation

The type-resolution design (above) replaces positional arithmetic with type inspection from `idMap`. This section goes further: instead of reconstructing LR→Pack→MFMA dependency chains through tile math and kernel parameters, **derive them from register operands** on the actual rocisa instruction objects.

The current validator has ~700 lines of functions that reconstruct "which instruction consumes which other instruction's output" from first principles:

| Function | Lines | What it reconstructs |
|----------|-------|---------------------|
| `lr_needed_by_mfma` | ~70 | Which MFMA slot consumes a given LR's data |
| `_transform_index_standard` | ~50 | Sub-iteration index mapping for LR→MFMA |
| `_transform_index_with_force_unroll_sub_iter` | ~65 | Same, for ForceUnrollSubIter mode |
| `find_earliest_mfma_execution` | ~60 | Earliest MFMA consuming a Pack's output |
| `_set_pack_needed_by` | ~80 | Pack→MFMA link via tile arithmetic |
| `_hook_up_packs_bf16` | ~30 | Pack→LR dependency via index formula |
| `_hook_up_packs_f32` | ~95 | 24-entry hardcoded dependency table |
| `_hook_up_packs_f32_mfma` | ~100 | 10-entry hardcoded dependency table |
| `set_lr_needed_by_for_VMFMA` | ~45 | LR→MFMA using tile counts and mfmaReorder |
| `set_gr_needed_by_from_lrs` | ~40 | GR→LR1/3 link |
| `set_gr_must_start_after_from_lr0s` | ~50 | GR must start after last LR0 |
| `set_gr_must_start_after_from_grinc` | ~40 | GR must start after GRInc |

All of these reconstruct information that is already encoded in the register operands of the rocisa instructions. A `v_perm_b32`'s `srcs[0]` and `srcs[1]` directly name the LR destination registers it depends on. An MFMA's `.a` and `.b` directly name the packed register it consumes. There is no need for tile math.

### Prerequisites

1. **mfmaCode must be passed into the validator.** Currently, MFMAs are not in `idMap` — they're handled separately in `customMainLoopSchedule`. The MFMA list (already reordered by `mfmaReorder` at line 374 of `CustomSchedule.py`) must be passed alongside `idMap`.

2. **idMap must be mandatory.** The existing `if "idMap" not in context` guard (which skips `verify_correct_number_of_instructions`) will be removed. All call sites — including unit tests — must supply an `idMap`.

### Data Flow Changes

#### customMainLoopSchedule (CustomSchedule.py)

```python
# Current (line 400):
status, message = cmsv.isValid(opt1, {'kernel': kernel, "idMap": idMap})

# New:
status, message = cmsv.isValid(opt1, {
    'kernel': kernel,
    "idMap": idMap,
    "mfmaCode": mfmaCode,  # already reordered by mfmaReorder at line 374
})
```

One line added. `mfmaCode` at this point is a flat list of `MFMAInstruction` objects in execution order.

#### isValid (CMSValidator.py)

```python
# Current:
timeline = create_unified_timeline(scheduleInfo, kernel, code_path)

# New:
timeline = create_unified_timeline(
    scheduleInfo, kernel, code_path,
    id_map=context["idMap"],
    mfma_code=context["mfmaCode"],
)
```

#### create_unified_timeline signature

```python
def create_unified_timeline(
    schedule_info: 'ScheduleInfo',
    kernel: 'Solution',
    code_path: int,
    id_map: dict[str, list],
    mfma_code: list['MFMAInstruction'],
) -> 'Timeline':
```

Both parameters are mandatory (no Optional).

#### Timeline.__init__

The Timeline stores `id_map` and `mfma_code` as instance attributes. During `_populate_instructions`, each `ValidatorInstruction` gets a `rocisa_inst` field pointing to the corresponding rocisa instruction object.

#### ValidatorInstruction changes

```python
@dataclass
class ValidatorInstruction(ABC):
    name: str
    issued_at: SchedulePosition
    rocisa_inst: object = None   # The rocisa instruction object, or None in legacy tests
```

All subclasses inherit this. The `MFMA` dataclass stores the actual `MFMAInstruction`:

```python
@dataclass
class MFMA(ValidatorInstruction):
    mfma_finish_cycles: ClassVar[int] = QUAD_CYCLES_STANDARD_MFMA_FINISH
    # rocisa_inst is an MFMAInstruction with .a, .b, .acc, .acc2 register operands
```

### Register-Based Dependency Resolution

#### Core utility: register overlap detection

```python
def _get_vgpr_range(rc: 'RegisterContainer') -> tuple[int, int]:
    """
    Extract the (start_idx, end_idx_exclusive) VGPR range from a RegisterContainer.
    
    For HolderContainers with symbolic names (e.g. vgprValuA_X1_I0_D0+0),
    resolves to a numeric index via regName.getTotalIdx() which uses the
    rocIsa singleton's vgprIdx map.
    """

def _vgpr_ranges_overlap(a: tuple[int, int], b: tuple[int, int]) -> bool:
    """Check if two VGPR ranges overlap."""
    return a[0] < b[1] and b[0] < a[1]
```

Register resolution always uses numeric indices. The symbolic names (e.g., `vgprValuA_X1_I0_D0+0`) are resolved to numeric VGPR indices at construction time by the kernel writer. The `HolderContainer.regName.getTotalIdx()` call does this resolution. All dependency comparisons are numeric range overlaps — no string matching.

#### Cross-Iteration Register Double-Buffering

The MAINLOOP macro is invoked 4 times: ML-1 (ID=0), ML (ID=1), NGL, NLL. The register naming encodes double-buffering directly — **`X0` and `X1` are separate physical VGPR ranges**, not aliases:

```asm
.set vgprValuA_X0_I0, vgprValuA_X0_I0_BASE+0
.set vgprValuA_X1_I0, vgprValuA_X0_I0_BASE+32   ; X1 is at offset +32 from X0
```

Within one iteration of the macro:
- **First half (mfmaIndex 0..N/2-1)**: MFMAs consume `X0` registers. LR0 loads into `X1` registers (prefetch for second half).
- **Second half (mfmaIndex N/2..N-1)**: MFMAs consume `X1` registers. LR1 loads into `X0` registers (prefetch for next iteration).

The LDS address swap (`v_xor_b32 ... 0x10000`) handles the write side (global read → LDS double buffer). The register-side double buffering uses `X0`/`X1` with no aliasing.

This means register-based dependency tracing works correctly across iterations:
- LR0 dst `X1_I0_D0+0` → Pack0 src `X1_I0_D0+0` → Pack0 dst `X1_I0+0` → MFMA(N/2..N-1) src `X1_I0+0` (same iteration, second half)
- LR1 dst `X0_I0_D0+0` → Pack1 src `X0_I0_D0+0` → Pack1 dst `X0_I0+0` → MFMA(0..N/2-1) src `X0_I0+0` (next iteration, first half)

Since `X0` and `X1` never overlap in numeric VGPR space, there is no aliasing ambiguity. The validator's 4-loop timeline (ML-1, ML, NGL, NLL) uses the same `idMap` instruction objects for each loop copy — and they reference the same `X0`/`X1` register containers. Cross-iteration dependencies are correctly captured because the register indices are distinct.

**TF32 exception**: TF32 kernels with `ExpandPointerSwap=0` use only `X0` (and `T0` for temporaries). Double-buffering is done entirely through LDS address swapping. In this case there is only one sub-iteration (LR0/LR3 instead of LR0/LR1), and the register dependency chain stays within a single buffer. The `idMap` entries `LRA0`, `LRA3`, `PackA0`, `PackA3` have distinct `T0`/`X0` register ranges that don't alias within an iteration.

#### Replacing `_hook_up_packs_bf16` / `_hook_up_packs_f32` / `_hook_up_packs_f32_mfma`

All three become a single function:

```python
def derive_pack_dependencies(
    packs: list[Pack],
    local_reads: list[LocalRead],
    id_map_packs: list,      # rocisa instructions for this pack name
    id_map_lrs: list,        # rocisa instructions for the corresponding LR name
) -> None:
    """
    Set must_start_after on each Pack by tracing register operands.
    
    Algorithm:
      1. Build a producer map: register_name -> ValidatorInstruction
         Pre-populate with LR destinations -> LocalRead
      2. Walk packs in issue_index order
      3. For each pack's rocisa instruction, inspect srcs
      4. Find which producer wrote to those registers
      5. Set pack.must_start_after to the latest such producer
      6. Update producer map with this pack's dst
    """
```

This replaces:
- `_hook_up_packs_bf16` (BF16 case: `v_perm_b32` srcs are LR dsts)
- `_hook_up_packs_f32` (TF32 24-group: CVT→Middle→CVT chain via registers)
- `_hook_up_packs_f32_mfma` (TF32 4x4: CVT→MFMA→CVT chain via registers)
- The hardcoded `pack_dependencies` dicts in both functions
- The `num_element_pairs` and `element_idx` index formulas

No special-casing needed per data type — the register operands encode the correct dependencies regardless of instruction mix.

#### Replacing `lr_needed_by_mfma` / `_transform_index_*` / `set_lr_needed_by_for_VMFMA`

```python
def set_lr_needed_by_from_mfma_operands(
    timeline: Timeline,
    mfma_code: list['MFMAInstruction'],
    id_map: dict[str, list],
) -> None:
    """
    Set needed_by on each LocalRead by finding the earliest MFMA
    whose .a or .b operand overlaps with the LR's destination register
    (either directly for non-pack cases, or transitively through the
    pack chain for BF16/TF32).
    
    For packed data paths (BF16, TF32):
      LR.dst -> Pack.src -> Pack.dst -> MFMA.a/b
      The LR is "needed by" the earliest MFMA in the chain.
    
    For non-packed data paths (rare):
      LR.dst -> MFMA.a/b directly
    """
```

This replaces:
- `lr_needed_by_mfma()` (70 lines of tile math)
- `_transform_index_standard()` (50 lines)
- `_transform_index_with_force_unroll_sub_iter()` (65 lines)
- `set_lr_needed_by_for_VMFMA()` (45 lines wrapper)
- All `mfmaReorder` threading through these functions (the reorder is already baked into `mfma_code` at this point)

#### Replacing `_set_pack_needed_by` / `find_earliest_mfma_execution`

```python
def set_pack_needed_by_from_mfma_operands(
    timeline: Timeline,
    mfma_code: list['MFMAInstruction'],
    id_map: dict[str, list],
) -> None:
    """
    Set needed_by on each Pack by finding the earliest MFMA 
    whose .a or .b operand overlaps with the pack's destination register.
    
    For TF32 4x4 packs, MFMAPacks (v_mfma_f32_4x4x4) are intermediate
    consumers — they appear as needed_by for CVT0 packs, and their results
    feed into CVT1 packs. The chain is:
      CVT0.dst -> MFMAPack.a (or .b)   [MFMAPack is needed_by for CVT0]
      MFMAPack.acc -> CVT1.src          [CVT1 has must_start_after MFMAPack]
      CVT1.dst -> real_MFMA.a (or .b)  [real MFMA is needed_by for CVT1]
    """
```

This replaces:
- `_set_pack_needed_by()` (80 lines of tile math and branching)
- `find_earliest_mfma_execution()` (60 lines of column-major grid search)
- All the `n_tiles_a`, `n_tiles_b`, `packs_per_tile` arithmetic
- The `ForceUnrollSubIter` / `UsePLRPack` special cases

#### GR constraints (`set_gr_needed_by_from_lrs`, `set_gr_must_start_after_from_lr0s`)

These remain **structurally similar** but gain verification. The constraint "GRs in iteration N must finish before LR1/3s in iteration N+1" is a scheduling invariant, not a register dependency. The rocisa instructions confirm *which* LDS block a GR writes to and which block an LR reads from via their offsets, but the cross-iteration dependency is fundamentally about LDS buffer aliasing, not register overlap.

What rocisa adds here:
- **SWaitCnt counter verification**: Count actual `BufferLoadB128` instructions (which consume vmcnt slots) and verify the `SWaitCnt.vlcnt` value is correct.
- **DTL pointer sequence verification**: Instead of the even/odd GR index hack (Phase 1), verify that `SMovB32`/`SAddU32` m0-writes and `BufferLoadB128` LDS loads form a valid DirectToLds sequence.

### Per-Pass Rewrite Details

#### Pass 1: `add_local_read_constraints`

Currently calls three functions:
- `set_lr_needed_by_for_VMFMA` → **replaced** by `set_lr_needed_by_from_mfma_operands`
- `apply_swaits` → **kept**, gains counter verification
- `apply_barriers` → **kept as-is**

The rewrite for `set_lr_needed_by`: instead of computing `lr_needed_by_mfma(lr_idx, n_tiles_a, n_tiles_b, mfma_reorder, ...)`, trace the register chain:

1. Get `lr.rocisa_inst.dst` → e.g., `vgprValuA_X1_I0_D0+0:+3`
2. Find which Pack instruction reads that register (scan Pack's `rocisa_inst.srcs`)
3. Get that Pack's `rocisa_inst.dst` → e.g., `vgprValuA_X1_I0+0`
4. Find which MFMA's `.a` or `.b` reads that register
5. Set `lr.needed_by = that_mfma`

For TF32, the chain is longer (LR→CVT0→4x4MFMA→CVT1→real MFMA) but the same algorithm works — keep following dst→src chains until hitting an MFMA in the timeline.

**Concrete register chain (BF16 256x256x64)**:
```
LRA0[0]: ds_read_b128 → dst = vgprValuA_X1_I0_D0+0:+3
                                        ↓ (src match)
PackA0[0]: v_perm_b32 → srcs = [vgprValuA_X1_I0_D1+0, vgprValuA_X1_I0_D0+0]
                         dst  = vgprValuA_X1_I0+0
                                        ↓ (src match via MFMA.a)
MFMA[0]: v_mfma_f32_16x16x32_bf16 → a = vgprValuA_X1_I0+0:+N
```

**Concrete register chain (TF32 128x128x32)**:
```
LRA0[0]: ds_read_b128 → dst = vgprValuA_T0_I0+8:+11
                                        ↓
PackA0[0]: v_cvt_pk_bf16_f32 → srcs = [vgprValuA_T0_I0+8, vgprValuA_T0_I0+9]
                                dst  = vgprValuA_X0_I0+16
                                        ↓
PackA0[4]: v_mfma_f32_4x4x4 → b = vgprValuA_X0_I0+16:+17, acc = vgprValuA_T0_I0+8:+11
                                        ↓ (acc output)
PackA0[9]: v_cvt_pk_bf16_f32 → srcs = [vgprValuA_T0_I0+8, vgprValuA_T0_I0+9]
                                dst  = vgprValuA_X0_I0+20
                                        ↓
MFMA[N]: v_mfma_f32_16x16x32 → a = vgprValuA_X0_I0+20:+M
```

**Register resolution**: Symbolic register names (e.g., `vgprValuA_X1_I0_D0+0`) are resolved to numeric VGPR indices via `HolderContainer.regName.getTotalIdx()`. All dependency matching is via numeric range overlap.

#### Pass 2: `add_pack_constraints`

Currently calls:
- `apply_swaits` → **kept**
- `hook_up_packs` → **replaced** by `derive_pack_dependencies` + `set_pack_needed_by_from_mfma_operands`

This is the biggest win. The three separate functions (`_hook_up_packs_bf16`, `_hook_up_packs_f32`, `_hook_up_packs_f32_mfma`) plus the hardcoded dependency dicts become a single `derive_pack_dependencies` that walks registers:

- **BF16**: `VPermB32 dst=X1_I0+0, src0=X1_I0_D1+0, src1=X1_I0_D0+0` — sources match `DSLoadB128 dst=X1_I0_D0+0:+3` and `DSLoadB128 dst=X1_I0_D1+0:+3`. Dependency falls out from register overlap.

- **TF32 4x4**: `VCvtPkF32toBF16 dst=X0_I0+16, src0=T0_I0+8, src1=T0_I0+9` (CVT0) → `v_mfma_f32_4x4x4 acc=T0_I0+8:+11, b=X0_I0+16:+17, acc2=T0_I0+8:+11` (reads CVT0 output) → `VCvtPkF32toBF16 dst=X0_I0+20, src0=T0_I0+8, src1=T0_I0+9` (reads MFMA output via `T0_I0+8`). Every link is a register overlap.

The quad-cycle timing (`_handle_min_pack_quad_cycles`, `estimate_quad_cycles`) stays unchanged — it operates on `ValidatorInstruction` positions and types, not register names.

#### Pass 3: `add_gr_not_too_early_constraints`

Currently calls:
- `apply_swaits` → **kept**
- `set_gr_must_start_after_from_lr0s` → **kept structurally** (LDS buffer aliasing, not a register dependency)
- `set_gr_must_start_after_from_grinc` → **kept structurally**
- `apply_must_start_after_barriers` → **kept**

These are about LDS buffer lifetime, not register data flow. rocisa doesn't fundamentally change the logic. What it adds:
- **DTL sequence verification**: Check that `SMovB32`→`BufferLoadB128`→`SAddU32` patterns in GR instruction lists are structurally valid (replaces the `idx_GR % 2 == 0` hack from Phase 1).
- **SWaitCnt vlcnt verification**: Count actual `BufferLoad*` instructions outstanding at each `SWaitCnt` and verify the `vlcnt` value is correct.

#### Pass 4: `add_gr_finish_before_lr_constraints`

Currently calls:
- `apply_swaits` → **kept**
- `set_gr_needed_by_from_lrs` → **kept structurally** (cross-iteration LDS dependency)
- `apply_barriers` → **kept**

Same reasoning as Pass 3 — LDS buffer timing across iterations. Not a register data flow problem.

### What Needs to Be Built (Summary)

| Component | Estimated lines | Replaces |
|-----------|----------------|----------|
| Register matching utilities (`_get_reg_name`, `_regs_overlap`) | ~50 | N/A (new) |
| Producer map builder | ~30 | N/A (new) |
| `derive_pack_dependencies` | ~60 | `_hook_up_packs_bf16` + `_hook_up_packs_f32` + `_hook_up_packs_f32_mfma` (225 lines) |
| `set_lr_needed_by_from_mfma_operands` | ~50 | `set_lr_needed_by_for_VMFMA` + `lr_needed_by_mfma` + transforms (230 lines) |
| `set_pack_needed_by_from_mfma_operands` | ~40 | `_set_pack_needed_by` + `find_earliest_mfma_execution` (140 lines) |
| Mock instruction builders for tests | ~80 | N/A (new) |
| SWaitCnt counter verification pass | ~60 | N/A (new capability) |
| **Total new** | **~370** | **~645 deleted** |

### What This Eliminates

After both phases, the following functions and constants are deleted:

```python
# Functions deleted:
lr_needed_by_mfma()                           # ~70 lines
_transform_index_standard()                    # ~50 lines
_transform_index_with_force_unroll_sub_iter()  # ~65 lines
find_earliest_mfma_execution()                 # ~60 lines
_set_pack_needed_by()                          # ~80 lines
_hook_up_packs_bf16()                          # ~30 lines
_hook_up_packs_f32()                           # ~95 lines
_hook_up_packs_f32_mfma()                      # ~100 lines
set_lr_needed_by_for_VMFMA()                   # ~45 lines (replaced)
_compute_swap_pack_count()                     # ~15 lines
_get_lrs_for_pack()                            # ~30 lines (simplified)
invert_mfma_reorder()                          # ~5 lines (no longer needed)
# Total: ~645 lines deleted

# Constants deleted (from Phase 1):
PACK_GROUP_SIZE_TF32, PACK_GROUP_SIZE_TF32_4X4
TF32_CVT0_END, TF32_MIDDLE_16_START, TF32_MIDDLE_16_END
TF32_4X4_MFMA_START, TF32_4X4_MFMA_END
VGPRS_PER_CONVERSION_GROUP
MFMAS_PER_TILE_TF32, MFMAS_PER_TILE_BF16

# Constants kept (ISA timing — not derivable from registers):
QUAD_CYCLES_CVT_BEFORE_MFMA
QUAD_CYCLES_MFMA_4X4_BEFORE_CVT1
QUAD_CYCLES_STANDARD_MFMA_FINISH
QUAD_CYCLES_MFMA_4X4_FINISH
MFMA_TYPE_SWITCH_THRESHOLD_FROM_STANDARD
MFMA_TYPE_SWITCH_THRESHOLD_FROM_4X4
```

### What Stays the Same

- **Quad-cycle timing model** (`precompute_issue_times`, `estimate_quad_cycles`, `estimate_quad_cycles_precomputed`): Instruction latencies are ISA properties, not derivable from register names.
- **`min_quad_cycles_before_result_used`** checks on `TimedPack`: ISA timing constraints.
- **`_handle_min_pack_quad_cycles`**: Sets the timing thresholds. Still needed.
- **`apply_swaits`**: Walks SWaitCnts backward through the timeline to set `guaranteed_by`. The structure stays, but gains a new **counter verification** sub-pass.
- **`apply_barriers` / `apply_must_start_after_barriers`**: SBarrier constraint logic. Unchanged.
- **Structural checks** (`verify_correct_number_of_instructions`, `verify_ascending_order`, `verify_scc_overlap`): Unchanged.
- **Timeline loop structure** (ML-1, ML, NGL, NLL) and `_should_add` filtering: Unchanged.
- **MiddlePack pair constraint** (`pair_consumer` / `next_scheduled_middle_16`): The constraint is about shared temp VGPRs. With rocisa registers, the pair detection becomes register-based instead of positional, but the validation logic stays.

### New Validation Capabilities

These are validations that become possible only with rocisa instruction access:

| Pass | What it validates | How |
|------|------------------|-----|
| **SWaitCnt counter verification** | `vlcnt` matches outstanding `BufferLoad*` count; `dscnt` matches outstanding `DSLoad*` count | Count actual memory instructions between SWaitCnts |
| **DTL sequence validation** | m0 setup → buffer_load LDS → m0 increment pattern is correct | Check `SMovB32`→`BufferLoadB128`→`SAddU32` type sequence in GR lists |
| **Register liveness** | No VGPR is read before it's written, or overwritten before it's consumed | Full def-use analysis across the timeline |
| **Pack chain type consistency** | CVT0 outputs feed into the right subsequent instruction type | Verify `VCvtPkF32toBF16.dst` → `MFMAInstruction.a/b` register match |
| **MFMA operand validation** | MFMA reads from the correct packed/converted registers, not stale data | Match `MFMA.a`/`.b` against the pack chain's final output registers |

### Impact on mfmaReorder

With `mfmaCode` passed in (already reordered at line 374 of `CustomSchedule.py`), `mfmaReorder` is no longer threaded through any dependency resolution function. The reorder is baked into the instruction list: `mfmaCode[i]` is the MFMA at execution slot `i`, and its `.a`/`.b` registers directly encode what it consumes.

Functions that currently accept `mfma_reorder` as a parameter:
- `set_lr_needed_by_for_VMFMA` → replaced
- `lr_needed_by_mfma` → replaced
- `_transform_index_standard` / `_transform_index_with_force_unroll_sub_iter` → deleted
- `hook_up_packs` → replaced
- `_set_pack_needed_by` → replaced
- `find_earliest_mfma_execution` → deleted

The `mfmaReorder` field on `ScheduleInfo` is still needed for Timeline MFMA placement (which vmfma slot each MFMA occupies), but it no longer flows into any constraint-resolution function.

### Unit Test Strategy

Unit tests must now provide `idMap` and `mfmaCode`. Two approaches, not mutually exclusive:

**1. Mock instruction builders** (for fast, focused tests):

```python
def make_mock_mfma(a_reg_name: str, b_reg_name: str, acc_reg_idx: int, n_acc: int) -> MFMAInstruction:
    """Create an MFMAInstruction with the given register operands."""

def make_mock_ds_load(dst_reg_name: str, n_regs: int) -> DSLoadB128:
    """Create a DSLoadB128 with the given destination register."""

def make_mock_pack_bf16(dst_reg_name: str, src0_reg_name: str, src1_reg_name: str) -> VPermB32:
    """Create a VPermB32 with the given operands."""
```

These mock instructions have real register containers but no ISA state. They test the register-tracing logic without a full kernel writer.

**2. Snapshot-based tests** (for correctness against real kernels):

Capture `idMap` and `mfmaCode` from real Tensile runs (as done in this investigation with `TENSILE_LOG_IDMAP`). Serialize to fixture files. Tests load fixtures and run the full validator, comparing results against known-good baselines.

### Execution Order and Risks

See the consolidated **Execution Order** dependency DAG and **Risks and Mitigations** table above. The register-based work spans branches 03 through 09. Branches 04, 06, 08 each introduce a new register-based function with dual-path validation against the old tile-math function. Branch 09 is the cleanup that removes the old functions once all dual-path assertions pass.

---

# Multi-ISA Validator Support

## Problem Statement

The CMS Validator was written for CDNA 4 (gfx950) MFMA kernels with DirectToLds=1. Its timeline passes embed quad-cycle timings from the CDNA 4 ISA, its Timeline constructor hard-asserts `DTL=1`, and its instruction names are hardcoded to a CDNA-specific subset.

With gfx1151 (RDNA 3.5) WMMA schedules now being added (commit 1c92b5a), the current approach is to disable all timeline passes for gfx1151 and only run the ISA-agnostic structural check `VERIFY_ASCENDING_ORDER`. This means gfx1151 schedules get almost no validation — silence means "unchecked," not "correct."

The validator needs to support multiple ISAs without:
- Long if-chains switching on ISA inside functions
- Monolithic pass-disable helpers per ISA (`_disable_cdna4_only_passes_for_gfx1151`)
- Silent gaps in validation coverage

### Structural Differences Between CDNA 4 and RDNA 3.5

| Property | CDNA 4 (gfx950) | RDNA 3.5 (gfx1151) |
|---|---|---|
| Math instruction | MFMA | WMMA |
| Wave size | 64 | 32 |
| DirectToLds | 1 (GR→LDS directly) | 0 (GR→VGPR→LW→LDS) |
| Pack instructions | Yes (BF16 v_perm, TF32 v_cvt/v_sub/v_mfma_4x4) | No (WMMA bf16 has no packing) |
| GRInc (separate pointer increment) | Yes (9 SALU per matrix) | Not always (some schedules bake increments into GR) |
| LW in schedule | No (not in `ALL_INSTRUCTION_NAMES`) | Yes (explicitly scheduled) |
| LCC in schedule | No | Yes |
| Loop sub-iterations | LR0, LR1, LR3 | LR0 through LR7 |
| LRSA/LRSB/LWSA/LWSB | Present | Not always present |
| Quad-cycle timing | CDNA 4 ISA section 7.6 | Different (RDNA 3.5 timing TBD) |

## Design: Composable Rules with Coverage Manifest

### Validation Concerns (Enum)

Each concern is an abstract correctness property that the validator must check. The enum avoids string typos and is the unit of coverage tracking.

```python
class ValidationConcern(Enum):
    # --- Universal (any ISA) ---
    INSTRUCTION_ORDERING = auto()       # vmfmaIndex sequences are non-decreasing
    SCHEDULE_COMPLETENESS = auto()      # schedule slot count matches idMap instruction count

    # --- Data readiness ---
    LR_DATA_READY = auto()              # LR data guaranteed available before consuming math instruction
    PACK_DATA_READY = auto()            # pack output ready before consuming math instruction

    # --- LDS coherence ---
    LDS_WRITE_AFTER_READ = auto()       # new data not written to LDS until current reads done
    LDS_READ_AFTER_WRITE = auto()       # data arrives in LDS (visible via barrier) before next reads

    # --- DTL=0 specific ---
    LW_ORDERING = auto()                # LW completes and is barrier-visible before next LR
    GR_VGPR_READY = auto()              # GR completes (vmcnt) before LW consumes VGPR data

    # --- Scalar safety ---
    SCALAR_REGISTER_SAFETY = auto()     # SCC-modifying instructions don't interleave unsafely

    # --- Timing ---
    QUAD_CYCLE_TIMING = auto()          # ISA-specific timing between dependent instructions
```

### ISA Concern Catalog

A single data structure declares which concerns each ISA family can have. The ISA key is the `(major, minor, patch)` tuple from the kernel's `ISA` field.

```python
# All concerns that kernels on this ISA can require, depending on kernel config.
# Indexed by ISA version tuple.
ISA_CONCERN_CATALOG: dict[tuple, set[ValidationConcern]] = {
    (9, 5, 0): {    # CDNA 4 (gfx950)
        ValidationConcern.INSTRUCTION_ORDERING,
        ValidationConcern.SCHEDULE_COMPLETENESS,
        ValidationConcern.LR_DATA_READY,
        ValidationConcern.PACK_DATA_READY,
        ValidationConcern.LDS_WRITE_AFTER_READ,
        ValidationConcern.LDS_READ_AFTER_WRITE,
        ValidationConcern.SCALAR_REGISTER_SAFETY,
        ValidationConcern.QUAD_CYCLE_TIMING,
    },
    (11, 5, 1): {   # RDNA 3.5 (gfx1151)
        ValidationConcern.INSTRUCTION_ORDERING,
        ValidationConcern.SCHEDULE_COMPLETENESS,
        ValidationConcern.LR_DATA_READY,
        ValidationConcern.LDS_WRITE_AFTER_READ,
        ValidationConcern.LDS_READ_AFTER_WRITE,
        ValidationConcern.LW_ORDERING,
        ValidationConcern.GR_VGPR_READY,
        # No PACK_DATA_READY: WMMA bf16 has no packing
        # No QUAD_CYCLE_TIMING: RDNA 3.5 timing not yet characterized
        # No SCALAR_REGISTER_SAFETY: not yet audited for RDNA 3.5 wave32
    },
}
```

Adding a new ISA is one entry in this dict. The catalog is the single source of truth for "what can this ISA need checked."

### Kernel-Level Concern Derivation

The kernel configuration narrows the ISA catalog to the subset that actually applies:

```python
def active_concerns(kernel: dict, idmap: dict) -> set[ValidationConcern]:
    """
    Determine which concerns must be covered for this specific kernel.

    Intersects kernel-derived requirements with the ISA's concern catalog.
    Raises ValueError if the kernel's ISA is not in the catalog.
    """
    isa = tuple(kernel["ISA"])
    if isa not in ISA_CONCERN_CATALOG:
        raise ValueError(
            f"ISA {isa} has no entry in ISA_CONCERN_CATALOG. "
            f"Add one before registering CMS schedules for this ISA."
        )
    isa_concerns = ISA_CONCERN_CATALOG[isa]

    # Universal concerns always active
    active = {
        ValidationConcern.INSTRUCTION_ORDERING,
        ValidationConcern.SCHEDULE_COMPLETENESS,
    }

    # Data readiness
    has_local_reads = any(k.startswith("LR") and not k.startswith("LRS") for k in idmap)
    if has_local_reads:
        active.add(ValidationConcern.LR_DATA_READY)

    has_packs = any(k.startswith("Pack") for k in idmap)
    if has_packs:
        active.add(ValidationConcern.PACK_DATA_READY)

    # LDS coherence depends on data path
    dtl = kernel.get("DirectToLds", 0)
    if dtl:
        # DTL=1: GR writes directly to LDS
        active.add(ValidationConcern.LDS_WRITE_AFTER_READ)
        active.add(ValidationConcern.LDS_READ_AFTER_WRITE)
    else:
        # DTL=0: GR→VGPR→LW→LDS
        active.add(ValidationConcern.LW_ORDERING)
        active.add(ValidationConcern.GR_VGPR_READY)
        active.add(ValidationConcern.LDS_WRITE_AFTER_READ)
        active.add(ValidationConcern.LDS_READ_AFTER_WRITE)

    # Scalar safety (only when GRInc is in the schedule)
    has_grinc = any(k.startswith("GRInc") for k in idmap)
    if has_grinc:
        active.add(ValidationConcern.SCALAR_REGISTER_SAFETY)

    # Timing (only when timed packs exist)
    if has_packs and kernel.get("UseF32XEmulation", False):
        active.add(ValidationConcern.QUAD_CYCLE_TIMING)

    # Intersect with what the ISA supports
    return active & isa_concerns
```

### Validation Rules

Each rule is a self-contained object that:
1. Declares which concern(s) it covers
2. Declares what instruction types it needs in the timeline (optional — for auto-skip)
3. Implements the check

```python
@dataclass
class ValidationRule(ABC):
    """Base class for all composable validation rules."""

    @abstractmethod
    def concerns(self) -> set[ValidationConcern]:
        """Which concerns this rule covers. A rule may cover multiple concerns."""
        ...

    @abstractmethod
    def run(self, timeline: Timeline, ctx: ValidatorPassContext) -> Optional[str]:
        """
        Execute the rule. Return None if valid, error message string if not.
        Only called if the rule's concerns are active for this kernel.
        """
        ...
```

Timeline-based rules operate on the Timeline. Structural rules get a separate base:

```python
@dataclass
class StructuralRule(ABC):
    """Rules that don't need a Timeline — operate on raw schedule data."""

    @abstractmethod
    def concerns(self) -> set[ValidationConcern]:
        ...

    @abstractmethod
    def run(self, schedule_info: ScheduleInfo, context: dict,
            code_path: int) -> tuple[bool, str]:
        ...
```

### Mapping Existing Passes to Rules

Each existing pass becomes one or more rules:

| Current Pass | Rule(s) | Concerns Covered |
|---|---|---|
| `VERIFY_ASCENDING_ORDER` | `AscendingOrderRule` | `{INSTRUCTION_ORDERING}` |
| `VERIFY_CORRECT_NUMBER_OF_INSTRUCTIONS` | `InstructionCountRule` | `{SCHEDULE_COMPLETENESS}` |
| `VERIFY_SCC_OVERLAP` | `SCCOverlapRule` | `{SCALAR_REGISTER_SAFETY}` |
| `ADD_LOCAL_READ_CONSTRAINTS` | `LRDataReadyRule` | `{LR_DATA_READY}` |
| `ADD_PACK_CONSTRAINTS` | `PackDataReadyRule` | `{PACK_DATA_READY, QUAD_CYCLE_TIMING}` |
| `ADD_GR_NOT_TOO_EARLY_CONSTRAINTS` | `GRAfterLRRule` (DTL=1 variant) | `{LDS_WRITE_AFTER_READ}` |
| `ADD_GR_FINISH_BEFORE_LR_CONSTRAINTS` | `GRBeforeLRRule` (DTL=1 variant) | `{LDS_READ_AFTER_WRITE}` |

New rules for gfx1151:

| Rule | Concerns Covered |
|---|---|
| `LWAfterLRRule` | `{LDS_WRITE_AFTER_READ}` (DTL=0 variant) |
| `LWBeforeLRRule` | `{LDS_READ_AFTER_WRITE}` (DTL=0 variant) |
| `GRVgprReadyRule` | `{GR_VGPR_READY}` |

Note: `LDS_WRITE_AFTER_READ` has two rules — one for DTL=1 (GR-based) and one for DTL=0 (LW-based). Only one will be applicable for a given kernel, determined by `active_concerns`. The coverage check passes as long as the concern is covered by *at least one* rule that ran.

### Rule Registry

Rules are registered globally. The framework collects them automatically:

```python
TIMELINE_RULES: list[ValidationRule] = [
    LRDataReadyRule(),
    PackDataReadyRule(),
    GRAfterLRRule(),         # DTL=1 LDS coherence
    GRBeforeLRRule(),        # DTL=1 LDS coherence
    LWAfterLRRule(),         # DTL=0 LDS coherence
    LWBeforeLRRule(),        # DTL=0 LDS coherence
    GRVgprReadyRule(),       # DTL=0 GR→VGPR readiness
]

STRUCTURAL_RULES: list[StructuralRule] = [
    AscendingOrderRule(),
    InstructionCountRule(),
    SCCOverlapRule(),
]
```

### The Validation Loop

```python
def isValid(scheduleInfo: ScheduleInfo, context: dict) -> tuple[bool, str]:
    kernel = context["kernel"]
    idmap = context.get("idMap", {})

    required = active_concerns(kernel, idmap)
    covered: set[ValidationConcern] = set()

    for code_path in range(scheduleInfo.numCodePaths):

        # --- Structural rules ---
        for rule in STRUCTURAL_RULES:
            rule_concerns = rule.concerns()
            if not (rule_concerns & required):
                continue  # none of this rule's concerns are needed
            status, message = rule.run(scheduleInfo, context, code_path)
            if not status:
                return False, f"Code path {code_path}: {message}"
            covered |= rule_concerns

        # --- Timeline rules ---
        # Only build timeline if any timeline rules are needed
        timeline_concerns = set()
        for rule in TIMELINE_RULES:
            timeline_concerns |= rule.concerns()
        if not (timeline_concerns & required):
            continue  # no timeline rules needed for this kernel

        timeline = create_unified_timeline(scheduleInfo, kernel, code_path, idmap)
        for rule in TIMELINE_RULES:
            rule_concerns = rule.concerns()
            if not (rule_concerns & required):
                continue
            error = rule.run(timeline, ctx)
            if error:
                return False, f"Code path {code_path}: {error}"
            covered |= rule_concerns

    # --- Coverage check ---
    uncovered = required - covered
    if uncovered:
        concern_names = ", ".join(c.name for c in sorted(uncovered, key=lambda c: c.value))
        return False, (
            f"Coverage gap: the following concerns are required for this kernel "
            f"but no validation rule covered them: {concern_names}. "
            f"This is a validator bug — add rules for these concerns."
        )

    return True, ""
```

The key property: **if `isValid` returns `(True, "")`, every required concern was actually checked by at least one rule.** Silence means correctness, not absence.

### What Gets Deleted

- `_disable_cdna4_only_passes_for_gfx1151` — replaced by the concern-based auto-skip
- `ScheduleInfo.disableValidationPass` / `reasonForDisablingValidationPass` — replaced by the concern system
- `ScheduleInfo._disabledPasses` dict — no longer needed
- The `all_timeline_disabled` check in `isValid` — replaced by the timeline-concerns intersection
- The `ValidatorPass` enum — replaced by `ValidationConcern`
- The `TIMELINE_PASSES` and `STRUCTURAL_CHECKS` dicts — replaced by `TIMELINE_RULES` and `STRUCTURAL_RULES`
- The DTL=1 assertion in the Timeline constructor — replaced by conditional timeline construction based on which rules need it

### What Stays the Same

- The `Timeline` class itself (loop structure, instruction scheduling, linearization)
- The `ValidatorInstruction` hierarchy (Pack, LocalRead, GlobalRead, etc.)
- The `ScheduleInfo` data structure (minus the disabled-passes machinery)
- The `ValidatorPassContext` (kernel, mfma_reorder, swap_global_read_order)
- All existing constraint-setting functions (`set_lr_needed_by_for_VMFMA`, `apply_swaits`, `apply_barriers`, etc.) — they become the implementation bodies of rules

### Adding a New ISA

To add support for a new ISA (e.g. gfx1250):

1. Add an entry to `ISA_CONCERN_CATALOG` listing which concerns apply
2. If the ISA has new pipeline patterns (e.g. a new data path), add a `ValidationConcern` enum value
3. Update `active_concerns` if the new concern has kernel-config dependencies
4. Write rule classes that cover the new concerns
5. Register them in `TIMELINE_RULES` or `STRUCTURAL_RULES`

No if-chains. No touching existing rules. No pass-disable helpers.

### Adding a New Rule to an Existing ISA

1. Write the rule class
2. Register it in the appropriate list
3. If it covers a concern that's already covered by another rule, both will run (belt and suspenders)
4. If it covers a new concern, add the concern to `ValidationConcern` and to the ISA's catalog entry

### Timeline Construction for Different ISAs

The current Timeline constructor asserts `DTL=1` and assumes specific instruction name patterns. With multi-ISA support:

- The `DTL=1` assertion is removed
- `ALL_INSTRUCTION_NAMES` becomes dynamic — derived from the schedule's `optSchedule.keys()` intersected with names the timeline knows how to handle
- The timeline construction is gated by whether any timeline rules need to run (via the concern intersection check)
- Rules that need LW instructions in the timeline (gfx1151) will need corresponding `ValidatorInstruction` subclasses (e.g. `LocalWrite`) and `_populate_instructions` support

### Execution Order (Multi-ISA additions)

See branches **11-multi-isa-framework** and **12-gfx1151-rules** in the consolidated dependency DAG above. Branch 11 depends only on 01-plumbing (the `rocisa_inst` / `asm_label` fields and mandatory `idMap`). It is a sibling of the register-based dependency branches (03-10) — they can be developed and reviewed in parallel.

### Risks and Mitigations (Multi-ISA)

| Risk | Mitigation |
|---|---|
| Coverage check is too strict — flags gaps for concerns that are technically inapplicable | `active_concerns` already intersects with ISA catalog. If a concern is in the catalog, the ISA *should* have a rule for it. Gaps are real bugs. |
| A rule covers a concern but its implementation is wrong for a particular ISA | Rules should be ISA-scoped (separate DTL=1 and DTL=0 LDS coherence rules, not one rule with an if-chain). The concern system tracks that *something* ran, not that it ran *correctly* — correctness is ensured by testing. |
| `active_concerns` derivation logic gets complex | Keep it flat and explicit. Each `if` block adds one concern based on one kernel property. No nested logic. |
| New ISA added without catalog entry | `active_concerns` raises `ValueError` — hard failure at schedule registration time, not silent at validation time. |
