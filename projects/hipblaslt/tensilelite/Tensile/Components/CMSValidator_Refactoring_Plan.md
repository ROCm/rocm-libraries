# CMSValidator Refactoring Plan

This document outlines architectural improvements for the CMSValidator module and provides step-by-step implementation plans for each recommendation.
---

## Table of Contents

1. [Issue Summary](#issue-summary)
2. [Recommendations](#recommendations)
   - [R1: Split File Into Modules](#r1-split-file-into-modules)
   - [R5: Define Typed Context](#r5-define-typed-context)
   - [R6: Use Registry Pattern for Pack Handling](#r6-use-registry-pattern-for-pack-handling)
   - [R9: Clarify Validation Logic Location](#r9-clarify-validation-logic-location)
   - [R10: Separate Timeline Responsibilities](#r10-separate-timeline-responsibilities)
   - [R11: Improve Test Infrastructure](#r11-improve-test-infrastructure)
   - [R12: Document Limitations Formally](#r12-document-limitations-formally)
   - [R13: Standardize ValidatorInstruction Class Hierarchy](#r13-standardize-validatorinstruction-class-hierarchy)
3. [Implementation Plans](#implementation-plans)

---

## Issue Summary

| # | Issue | Severity | Category |
|---|-------|----------|----------|
| 1 | 2100+ line file with mixed responsibilities | High | Maintainability |
| 5 | Untyped `context: dict` | Medium | Type Safety |
| 6 | Nested conditionals for pack modes | Medium | Extensibility |
| 9 | Mixed validation logic locations | Medium | Clarity |
| 10 | Timeline class has too many jobs | Medium | Maintainability |
| 11 | Testing infrastructure gaps | Medium | Testability |
| 12 | Undocumented limitations | Low | Documentation |
| 13 | Inconsistent instruction class interfaces | Medium | Type Safety / Maintainability |

---

## Recommendations

### R1: Split File Into Modules

**Current State**: Single 2100+ line file containing instruction classes, timeline management, 8 validation passes, pack handling, and utility functions. Named constants are already extracted to module-level variables (R4 complete). `SchedulePosition` and sentinel constants are also at module level.

**Target State**:
```
Tensile/Components/CMSValidator/
├── __init__.py              # Public API: isValid(), TIMELINE_PASSES, structural_checks
├── instructions.py          # SchedulePosition, ValidatorInstruction base + all subclasses
├── timeline.py              # Timeline class + create_unified_timeline()
├── constants.py             # Named constants + POSITION_INF/POSITION_NEG_INF
├── context.py               # ValidatorPassContext + ValidationContext dataclasses
├── passes.py                # add_*_constraints() functions, verify_*() structural checks
├── pack_handlers/
│   ├── __init__.py          # get_pack_handler() factory
│   ├── base.py              # PackHandler ABC
│   ├── bf16.py
│   ├── tf32.py
│   └── tf32_4x4mfma.py
└── utils/
    ├── __init__.py
    ├── mfma_reorder.py      # invert_mfma_reorder, find_earliest_mfma_execution
    └── index_transforms.py  # lr_needed_by_mfma, index_for_force_unroll_sub_iter
```

**Benefits**:
- Each file has single responsibility
- Easier to navigate and understand
- Enables parallel development
- Improves test isolation

---

### R5: Define Typed Context

> **Note**: `ValidatorPassContext` already provides typed access for timeline-based passes. The remaining work is to replace the top-level `context: dict` parameter in `isValid()` and the structural check functions with a full `ValidationContext`.

**Current State**:
```python
# isValid() still takes an untyped dict:
def isValid(scheduleInfo: 'ScheduleInfo', context: dict) -> tuple[bool, str]:
    kernel = context["kernel"]
    idMap = context.get("idMap")  # Optional? Unknown structure

# But timeline passes already use a typed context:
@dataclass
class ValidatorPassContext:
    kernel: 'Solution'
    mfma_reorder: list[int]
    swap_global_read_order: bool
```

**Target State**:
```python
@dataclass
class ValidationContext:
    """Typed context for CMS validation."""
    kernel: 'Solution'
    id_map: Optional[dict[str, list[Any]]] = None

    # Convenience properties to avoid repeated kernel lookups
    @property
    def swap_global_read_order(self) -> bool:
        return self.kernel.get("SwapGlobalReadOrder", False)

    @property
    def direct_to_lds(self) -> bool:
        return self.kernel.get("DirectToLds", False)

    @property
    def use_f32x_emulation(self) -> bool:
        return self.kernel.get("UseF32XEmulation", False)

    @property
    def use_mfma_f32x_emulation(self) -> bool:
        return self.kernel.get("UseMFMAF32XEmulation", False)

    @property
    def use_plr_pack(self) -> bool:
        return self.kernel.get("UsePLRPack", False)

    @property
    def force_unroll_sub_iter(self) -> bool:
        return self.kernel.get("ForceUnrollSubIter", False)

    @property
    def n_tiles_a(self) -> int:
        return self.kernel["MIWaveTileA"]

    @property
    def n_tiles_b(self) -> int:
        return self.kernel["MIWaveTileB"]
```

---

### R6: Use Registry Pattern for Pack Handling

**Current State**:
```python
def hook_up_packs(timeline, kernel, mfma_reorder):
    if is_tf32_emulation:
        if is_4x4mfma_tf32:
            _hook_up_packs_f32_mfma(packs, local_reads)
        else:
            _hook_up_packs_f32(packs, all_middle_16_packs, local_reads)
        _handle_min_pack_quad_cycles(packs, is_4x4mfma_tf32)
    else:
        _hook_up_packs_bf16(packs, local_reads)
```

**Problem**: Nested conditionals become unwieldy as pack modes grow. Adding a new mode requires modifying the central dispatch logic.

**Target State** (Registry with Decorators):
```python
from dataclasses import dataclass
from typing import Callable

@dataclass
class PackContext:
    """Everything a pack handler might need."""
    packs: list[Pack | MFMAPack]
    local_reads: list[LocalRead]
    all_packs_in_loop: list[Pack | MFMAPack]  # For middle-16 lookups
    kernel: dict
    mfmas_by_index: dict[int, MFMA]
    mfma_reorder: list[int]
    num_vmfma: int
    loop_index: int

# Registry
_PACK_HANDLERS: dict[str, Callable[[PackContext], None]] = {}

def pack_handler(mode: str):
    """Register a function as a pack handler for a given mode."""
    def decorator(fn: Callable[[PackContext], None]):
        _PACK_HANDLERS[mode] = fn
        return fn
    return decorator

def get_pack_mode(kernel: dict) -> str:
    """Determine pack mode from kernel configuration."""
    if kernel.get("UseMFMAF32XEmulation"):
        return "tf32_4x4mfma"
    if kernel.get("UseF32XEmulation"):
        return "tf32"
    return "bf16"

def handle_packs(ctx: PackContext) -> None:
    """Dispatch to the appropriate pack handler."""
    mode = get_pack_mode(ctx.kernel)
    if mode not in _PACK_HANDLERS:
        raise ValueError(f"Unknown pack mode: {mode}")
    _PACK_HANDLERS[mode](ctx)
```

**Handler implementations** (each is a decorated function):
```python
@pack_handler("bf16")
def _handle_bf16(ctx: PackContext) -> None:
    """BF16: each pack depends on 2 consecutive LRs."""
    num_element_pairs = len(ctx.local_reads) // 2
    for pack in ctx.packs:
        element_idx = pack.issue_index % num_element_pairs
        lr1 = ctx.local_reads[element_idx * 2]
        lr2 = ctx.local_reads[element_idx * 2 + 1]
        pack.must_start_after = max(lr1, lr2, key=lambda lr: lr.issued_at)
    _set_pack_needed_by(ctx.packs, ctx.mfmas_by_index, ...)

@pack_handler("tf32")
def _handle_tf32(ctx: PackContext) -> None:
    """TF32: groups of PACK_GROUP_SIZE_TF32 with CVT0 -> middle-16 -> CVT1 chain."""
    _hook_up_packs_f32(ctx.packs, ctx.all_packs_in_loop, ctx.local_reads)
    _set_min_quad_cycles(ctx.packs, cycles=QUAD_CYCLES_CVT_BEFORE_MFMA)
    _set_pack_needed_by(ctx.packs, ctx.mfmas_by_index, ...)

@pack_handler("tf32_4x4mfma")
def _handle_tf32_4x4mfma(ctx: PackContext) -> None:
    """TF32 4x4 MFMA: groups of PACK_GROUP_SIZE_TF32_4X4 with CVT0 -> MFMAPack -> CVT1 chain.

    MFMAPack objects already have min_quad_cycles_before_result_used set at construction time,
    so no _set_min_quad_cycles call is needed here.
    """
    _hook_up_packs_f32_mfma(ctx.packs, ctx.local_reads)
    _set_pack_needed_by(ctx.packs, ctx.mfmas_by_index, ...)
```

**Adding a new mode** requires only a new decorated function:
```python
@pack_handler("fp8")
def _handle_fp8(ctx: PackContext) -> None:
    """FP8: whatever FP8 needs."""
    ...
```

**Benefits**:
- No class hierarchy or ABC boilerplate
- Self-registering handlers (decorator makes intent clear)
- Handlers can live in separate files and register on import
- Easy to discover all modes: `_PACK_HANDLERS.keys()`
- Easy to test: call function directly with mock PackContext
- Pythonic pattern (same as Flask routes, pytest fixtures, Click commands)

**Comparison to class-based Strategy Pattern**:

| Aspect | Classes | Registry |
|--------|---------|----------|
| Add new mode | New class + factory update | Add decorated function |
| Boilerplate | ABC, abstractmethod, inheritance | One decorator |
| Discoverability | Grep for subclasses | `_PACK_HANDLERS.keys()` |
| Testing | Instantiate class, call methods | Call function with mock context |
| IDE support | Good | Good (dataclass + type hints) |

---

### R9: Clarify Validation Logic Location

> **Note**: The unified timeline already achieves the key structural goal: passes only set up constraints and call `validate_timeline()`, which delegates to each instruction's `validate()` method. The remaining work is ensuring all validation logic lives in instruction classes (not in standalone functions) and adding helper methods.

**Current State**: Mixed - some validation in `Instruction.validate()`, some in standalone functions.

**Target State**: All validation in instruction classes (self-validating objects):

```python
@dataclass
class LocalRead(ValidatorInstruction):
    # ... fields ...

    def validate(self) -> Optional[str]:
        """Validate this instruction against its constraints."""
        if not self._has_constraints():
            return None

        if not self._is_guaranteed_before_needed():
            return self._format_timing_error()

        return None

    def _has_constraints(self) -> bool:
        return self.needed_by.issued_at != POSITION_INF

    def _is_guaranteed_before_needed(self) -> bool:
        return self.guaranteed_by < self.needed_by.issued_at

    def _format_timing_error(self) -> str:
        return _error_issued_too_late(
            name=self.name,
            issued_at=self.issued_at.vmfma_index,
            needed_by_name=self.needed_by.name,
            needed_by_at=self.needed_by.issued_at.vmfma_index
        )
```

**Validation passes** then only:
1. Build the timeline
2. Set up constraints (needed_by, guaranteed_by, etc.)
3. Call `validate_timeline()` which iterates and calls each instruction's `validate()`

---

### R10: Separate Timeline Responsibilities

**Current State**: Timeline handles parsing, storage, iteration management, and lookups. (Index resolution has been eliminated by `SchedulePosition` — positions are now computed in one shot at insert time.)

**Target State**:

```python
# schedule_parser.py
class ScheduleParser:
    """Parses ScheduleInfo into ValidatorInstructions."""

    def parse(
        self,
        schedule_info: 'ScheduleInfo',
        instruction_names: list[str],
        code_path: int,
        context: ValidationContext
    ) -> list[ValidatorInstruction]:
        """Parse schedule into instruction list."""
        ...

# loop_manager.py
class LoopManager:
    """Manages loop iterations (MAIN_LOOP_PREV, MAIN_LOOP, NO_GLOBAL_LOAD_LOOP, NO_LOCAL_LOAD_LOOP)."""

    def __init__(self, base_instructions: list[ValidatorInstruction], context: ValidationContext):
        self._loops = self._create_loops(base_instructions, context)

    def get_loop(self, loop_name: str) -> 'LoopIteration':
        ...

    def all_instructions(self) -> Iterator[ValidatorInstruction]:
        """Iterate all instructions across all loops."""
        ...

# timeline.py
class Timeline:
    """Immutable view of scheduled instructions."""

    def __init__(self, loop_manager: LoopManager):
        self._loop_manager = loop_manager
        self._index = self._build_index()

    def get_instructions(self, name: str, loop: str) -> Sequence[ValidatorInstruction]:
        ...

    def get_instructions_combined(self, name: str) -> Sequence[ValidatorInstruction]:
        ...

    def get_at_vmfma_index(self, index: int, loop: str) -> Sequence[ValidatorInstruction]:
        ...
```

---

### R11: Improve Test Infrastructure

**Current State**:
```python
if "idMap" not in context:
    printWarning("idMap not found in context. Skipping...")
    return True, ""
```

**Target State**:

```python
# tests/fixtures.py
class ValidationContextFactory:
    """Factory for creating test ValidationContext objects."""

    @staticmethod
    def default() -> ValidationContext:
        return ValidationContext(
            kernel=KernelFixtures.default_kernel(),
            id_map=None
        )

    @staticmethod
    def with_tf32() -> ValidationContext:
        kernel = KernelFixtures.default_kernel()
        kernel["UseF32XEmulation"] = True
        kernel["UseDirect32XEmulation"] = True
        return ValidationContext(kernel=kernel)

    @staticmethod
    def with_full_id_map(schedule_info: 'ScheduleInfo') -> ValidationContext:
        return ValidationContext(
            kernel=KernelFixtures.default_kernel(),
            id_map=IdMapBuilder.from_schedule(schedule_info)
        )

class ScheduleInfoBuilder:
    """Builder for creating test ScheduleInfo objects."""

    def __init__(self):
        self._opt_schedule = {}
        self._num_mfma = 16
        self._mfma_reorder = []

    def with_local_reads(self, lra0: list[int], lrb0: list[int]) -> 'ScheduleInfoBuilder':
        self._opt_schedule["LRA0"] = [lra0]
        self._opt_schedule["LRB0"] = [lrb0]
        return self

    def with_global_reads(self, gra: list[int], grb: list[int]) -> 'ScheduleInfoBuilder':
        self._opt_schedule["GRA"] = [gra]
        self._opt_schedule["GRB"] = [grb]
        return self

    def build(self) -> 'ScheduleInfo':
        ...

# In tests
def test_lr_finished_before_vmfma():
    schedule = (ScheduleInfoBuilder()
        .with_local_reads([0, 1, 2], [0, 1, 2])
        .with_sync_at([3])
        .build())
    context = ValidationContextFactory.default()

    result = verify_lrs_finished_before_vmfma(schedule, context, code_path=0)

    assert result.passed
```

---

### R12: Document Limitations Formally

**Current State**: TODOs and limitations scattered in code comments.

**Target State**: Create `LIMITATIONS.md`:

```markdown
# CMSValidator Known Limitations

## Unsupported Configurations

### UseDirect32XEmulation = False with UseF32XEmulation = True
- **Status**: Not supported
- **Error**: Raises ValueError
- **Reason**: Non-direct TF32 emulation uses different pack sequences not yet modeled

### ForceUnrollSubIter with Register Reuse
- **Status**: Partially supported
- **Issue**: Does not fully account for VGPR reuse patterns
- **Tracking**: TODO in timeline.py line 489

## Validation Gaps

### False Negatives (schedule appears valid but may fail)
1. Quad-cycle estimation is conservative (doesn't model all stalls)
2. SBarrier timing assumes instant synchronization

### False Positives (schedule appears invalid but works)
1. None currently known

## Architecture-Specific Behavior

### CDNA 4 (gfx950)
- Quad-cycle requirements from ISA section 7.6 are enforced
- 4x4 MFMA TF32 path is validated

### CDNA 3 and earlier
- Quad-cycle requirements not enforced (different ISA)
```

---

### R13: Standardize ValidatorInstruction Class Hierarchy

**Current State**: The `ValidatorInstruction` base class is minimal (just `name`, `issued_at`, `validate()`, `done_idx()`) and subclasses diverge significantly in their field types, constraint patterns, and error message formatting. With `SchedulePosition` in place (R2 done), `num_vmfma` has been removed from instruction classes and display formatting uses `self.issued_at.vmfma_index` directly. The remaining problems are:

**Problem 1: `needed_by` type mismatch across subclasses**
```python
class LocalRead(ValidatorInstruction):
    needed_by: ValidatorInstruction = ...   # An instruction object

class Pack(ValidatorInstruction):
    needed_by: ValidatorInstruction = ...   # An instruction object

class GlobalRead(ValidatorInstruction):
    needed_by: SchedulePosition = POSITION_INF  # A bare position!
```
`GlobalRead.needed_by` is a `SchedulePosition` (the `issued_at` of the first LR1/3), while `LocalRead.needed_by` and `Pack.needed_by` are `ValidatorInstruction` references. This means:
- Error formatting code cannot be shared (one does `self.needed_by.name`, the other can't).
- `validate_timeline()` can't make any assumptions about constraint fields.
- `estimate_quad_cycles()` must use `hasattr()` checks instead of type-safe access.

**Problem 2: Inconsistent error message formats**

Each class formats errors differently, making them hard to parse programmatically or visually:
```python
# LocalRead:
f"{self.name} @ idx={issued_at} is not valid. There are no guarantees on when it will be done."
f"{self.name} @ idx={issued_at} issued too late, must be guaranteed before {self.needed_by.name} @ idx={needed_by}{context_str} but only guaranteed @ idx={guaranteed_by}."

# Pack:
f"{self.name} @ idx={issued_at} issued too early, must be issued after idx={must_start_after_at} (because of {self.must_start_after.name} issued @ idx={must_start_after_issued_at})."
f"{self.name} @ idx={issued_at} issued too late, must be issued before {self.needed_by.name} @ idx={needed_by_at}."
f"{self.name} @ idx={issued_at} has wrong interleaving. Should have been followed by ..."
f"{self.name} @ idx={issued_at} has too little gap between it and ..."
f"{self.name} at index {issued_at} is not valid."  # Note: "at index" not "@ idx="!

# GlobalRead._validate_must_start_after():
f"{name} @ idx={issued_at} is issued too early. Must be issued after idx=..."
f"There is an SBarrier missing between the SWaitCnt @ idx=..."

# GlobalRead._validate_needed_by():
f"{name} @ idx={issued_at} is not valid. There are no guarantees on when it will be done."
f"{name} @ idx={issued_at} is not valid. There is no SBarrier acting on it."
f"{name} @ idx={issued_at} is not valid. It is guaranteed by the SWait @ idx=..."

# SWait:
f"SWait at index {self.issued_at.vmfma_index} is invalid: ..."  # Uses "at index"

# Barrier:
f"Barrier at index {self.issued_at.vmfma_index} is not valid. Must be >= -1."  # Uses "at index"
```

Note the inconsistencies: "at index" vs "@ idx=", "issued too early" vs "is issued too early", "is not valid" appearing in different positions, some messages explaining the fix ("Order must be X") and others not.

**Problem 3: `estimate_quad_cycles()` uses `hasattr()` checks**
```python
# Current: runtime duck-typing
if not hasattr(instruction, "needed_by") or instruction.needed_by is None:
    continue
if not hasattr(instruction, "min_quad_cycles_before_result_used"):
    continue
```
These `hasattr()` checks exist because the base class doesn't define `needed_by` or `min_quad_cycles_before_result_used`, so there's no type-safe way to check if an instruction has constraints.

**Target State**:

1. **Unify `GlobalRead.needed_by` to `ValidatorInstruction`** (matching `LocalRead` and `Pack`):
```python
class GlobalRead(ValidatorInstruction):
    needed_by: ValidatorInstruction = field(default_factory=lambda: MFMA(POSITION_INF))
    # Instead of: needed_by: SchedulePosition = POSITION_INF
```
Update `set_gr_needed_by_from_lrs()` to assign the LR1/3 instruction object rather than its `issued_at`:
```python
# Before:
for _, gr in grs:
    gr.needed_by = LR_target.issued_at   # SchedulePosition

# After:
for _, gr in grs:
    gr.needed_by = LR_target              # ValidatorInstruction
```

**Benefits**:
- `needed_by` has a single type across all instruction classes, enabling shared code
- Error messages are consistent and testable
- `hasattr()` checks in `estimate_quad_cycles()` are replaced with type-safe access
- Cross-iteration detection logic is standardized

**Relationship to other recommendations**:
- **Enables R9** (clarify validation logic) by making instruction interfaces consistent

---

## Implementation Plans

### Plan for R1: Split File Into Modules

**Estimated Effort**: Large (2-3 days)

**Step 1**: Create directory structure
```bash
mkdir -p Tensile/Components/CMSValidator/{passes,pack_handlers,utils}
touch Tensile/Components/CMSValidator/{__init__,instructions,timeline,constants,context}.py
touch Tensile/Components/CMSValidator/passes/{__init__,base}.py
touch Tensile/Components/CMSValidator/pack_handlers/{__init__,base}.py
touch Tensile/Components/CMSValidator/utils/__init__.py
```

**Step 2**: Extract constants (mostly done — move existing module-level constants)
- Move `MAIN_LOOP_PREV`, `MAIN_LOOP`, `NO_GLOBAL_LOAD_LOOP`, `NO_LOCAL_LOAD_LOOP` to `constants.py`
- Move `PACK_GROUP_SIZE_TF32`, `PACK_GROUP_SIZE_TF32_4X4` to `constants.py`
- Move `TF32_CVT0_END`, `TF32_MIDDLE_16_START`, `TF32_MIDDLE_16_END`, `TF32_4X4_MFMA_START`, `TF32_4X4_MFMA_END` to `constants.py`
- Move `QUAD_CYCLES_*`, `MFMA_TYPE_SWITCH_THRESHOLD_*`, `MFMAS_PER_TILE_*`, `VGPRS_PER_CONVERSION_GROUP` to `constants.py`
- Move `POSITION_INF`, `POSITION_NEG_INF` to `constants.py`
- Update imports in original file

**Step 3**: Extract `SchedulePosition` and instruction classes
- Move `SchedulePosition`, `ValidatorInstruction`, `LocalRead`, `GlobalRead`, `Pack`, `MFMAPack`, `MFMA`, `SWait`, `Barrier`, `SNop` to `instructions.py`
- Update imports

**Step 4**: Extract utility functions
- Move `invert_mfma_reorder`, `find_earliest_mfma_execution` to `utils/mfma_reorder.py`
- Move `lr_needed_by_mfma`, `index_for_force_unroll_sub_iter`, `_transform_index_*` to `utils/index_transforms.py`
- Move `schedule_get` to `utils/__init__.py`

**Step 5**: Extract Timeline class
- Move `Timeline`, `apply_barriers`, `apply_swaits`, etc. to `timeline.py`
- Keep transformation functions with Timeline for now

**Step 6**: Extract validation passes
- Move `add_*_constraints()` functions, `TIMELINE_PASSES`, and `verify_*` structural checks to `passes.py`
- Move `ValidatorPassContext` to `context.py`

**Step 7**: Update main module
- `CMSValidator/__init__.py` exports `isValid`, `ValidationResult`
- Create backward-compatible `CMSValidator.py` that imports from package

**Step 8**: Update all imports throughout codebase
- Search for `from Tensile.Components.CMSValidator import`
- Update to new paths

**Step 9**: Run tests and fix any issues

---

### Plan for R5: Define Typed Context

**Estimated Effort**: Small (2-4 hours)

> **Note**: `ValidatorPassContext` already provides typed access for timeline-based passes. This plan extends typed context to the `isValid()` entry point and structural checks. Consider whether `ValidationContext` should subsume or wrap `ValidatorPassContext`.

**Step 1**: Create ValidationContext dataclass (see R5 above)
- Decide relationship to `ValidatorPassContext`: either `ValidationContext` contains a `ValidatorPassContext`, or `ValidatorPassContext` is merged into `ValidationContext`

**Step 2**: Update isValid() signature
```python
def isValid(scheduleInfo: 'ScheduleInfo', context: Union[dict, ValidationContext]) -> tuple[bool, str]:
    if isinstance(context, dict):
        context = ValidationContext.from_dict(context)
    ...
```

**Step 3**: Add from_dict() class method for backward compatibility
```python
@classmethod
def from_dict(cls, d: dict) -> 'ValidationContext':
    return cls(
        kernel=d["kernel"],
        id_map=d.get("idMap")
    )
```

**Step 4**: Update structural check functions to use ValidationContext
- Replace `context["kernel"]` with `context.kernel`
- Replace `context.get("kernel", {}).get(...)` with `context.kernel.get(...)`

**Step 5**: Run tests

---

### Plan for R6: Use Registry Pattern for Pack Handling

**Estimated Effort**: Medium (1 day)

**Step 1**: Define PackContext dataclass and registry infrastructure
```python
# In CMSValidator.py or new pack_handlers.py
@dataclass
class PackContext:
    packs: list[Pack]
    local_reads: list[LocalRead]
    all_packs_in_loop: list[Pack]
    kernel: dict
    mfmas_by_index: dict[int, MFMA]
    mfma_reorder: list[int]
    num_vmfma: int
    loop_index: int

_PACK_HANDLERS: dict[str, Callable[[PackContext], None]] = {}

def pack_handler(mode: str):
    def decorator(fn):
        _PACK_HANDLERS[mode] = fn
        return fn
    return decorator

def get_pack_mode(kernel: dict) -> str:
    if kernel.get("UseMFMAF32XEmulation"):
        return "tf32_4x4mfma"
    if kernel.get("UseF32XEmulation"):
        return "tf32"
    return "bf16"
```

**Step 2**: Convert existing handler functions to decorated handlers
- Add `@pack_handler("bf16")` to `_hook_up_packs_bf16` (or wrapper)
- Add `@pack_handler("tf32")` to `_hook_up_packs_f32` (or wrapper)
- Add `@pack_handler("tf32_4x4mfma")` to `_hook_up_packs_f32_mfma` (or wrapper)
- Update function signatures to accept `PackContext`

**Step 3**: Update `hook_up_packs()` to use registry
```python
def hook_up_packs(timeline: Timeline, kernel: dict, mfma_reorder: list[int]) -> None:
    mode = get_pack_mode(kernel)
    if mode not in _PACK_HANDLERS:
        raise ValueError(f"Unknown pack mode: {mode}")

    mfmas_by_index = {int(m.issued_at): m for _, m in timeline.get_instructions_combined("MFMA")}

    for i_loop, loop in enumerate(timeline.loops):
        packs_by_name = _gather_packs(timeline, loop)
        all_packs_in_loop = [p for packs in packs_by_name.values() for p in packs]

        for pack_name, packs in packs_by_name.items():
            local_reads = _get_lrs_for_pack(timeline, kernel.get("UsePLRPack"), pack_name, loop)
            if not local_reads:
                continue

            ctx = PackContext(
                packs=packs,
                local_reads=local_reads,
                all_packs_in_loop=all_packs_in_loop,
                kernel=kernel,
                mfmas_by_index=mfmas_by_index,
                mfma_reorder=mfma_reorder,
                num_vmfma=timeline.num_vmfma,
                loop_index=i_loop,
            )
            _PACK_HANDLERS[mode](ctx)
```

**Step 4**: Run tests to verify no regressions

**Step 5**: (Optional) If file splitting (R1) is done, move handlers to separate files
- Each handler file imports `pack_handler` decorator and self-registers on import
- Main module imports handler files to trigger registration

---

### Plan for R9: Clarify Validation Logic Location

**Estimated Effort**: Small (half day) — Step 3 already achieved.

**Step 1**: Document the chosen pattern (all validation in instruction classes)

**Step 2**: Review each instruction's validate() method
- Ensure it handles all constraints for that instruction type
- Move any external validation logic into the class

**Step 3**: ~~Simplify standalone validation passes~~
- ~~They should only: build timeline, set constraints, call validate_timeline()~~
- **Done** — the `add_*_constraints()` functions already follow this pattern.

**Step 4**: Add helper methods to instruction classes for constraint checking

**Step 5**: Run tests

---

### Plan for R10: Separate Timeline Responsibilities

**Estimated Effort**: Large (2-3 days)

**Step 1**: Create ScheduleParser class
- Extract instruction parsing logic from Timeline.__init__
- Handle DirectToLds, mfmaReorder, etc.

**Step 2**: Create LoopManager class
- Handle MAIN_LOOP_PREV, MAIN_LOOP, NO_GLOBAL_LOAD_LOOP, NO_LOCAL_LOAD_LOOP iteration creation
- Handle instruction filtering per loop type

**Step 3**: Simplify Timeline class
- Accept LoopManager in constructor
- Focus on query/lookup functionality only
- Note: `_insert()` now creates `SchedulePosition` in one shot, so there is no separate index-resolution phase to extract

**Step 4**: Update all Timeline usages

**Step 5**: Run tests

---

### Plan for R11: Improve Test Infrastructure

**Estimated Effort**: Medium (1-2 days)

**Step 1**: Create tests/fixtures.py with factory classes

**Step 2**: Create ScheduleInfoBuilder for test data

**Step 3**: Create KernelFixtures with common kernel configurations

**Step 4**: Remove printWarning + skip pattern
```python
# Before
if "idMap" not in context:
    printWarning("...")
    return True, ""

# After
if context.id_map is None:
    raise ValueError("id_map required for instruction count validation")
```

**Step 5**: Update existing tests to use new fixtures

**Step 6**: Add tests for edge cases that were previously skipped

---

### Plan for R12: Document Limitations Formally

**Estimated Effort**: Small (1-2 hours)

**Step 1**: Create LIMITATIONS.md in CMSValidator directory

**Step 2**: Search for TODOs and limitations in code
```bash
grep -n "TODO\|FIXME\|not supported\|skip" CMSValidator.py
```

**Step 3**: Document each limitation with:
- Status (not supported, partially supported, known issue)
- Description
- Workaround if any
- Tracking reference

**Step 4**: Add reference to LIMITATIONS.md in module docstring

---

### Plan for R13: Standardize ValidatorInstruction Class Hierarchy

**Estimated Effort**: Medium (1 day)

**Important**: With R2 done, `num_vmfma` has already been removed from instruction classes and display formatting already uses `self.issued_at.vmfma_index`. The remaining work focuses on unifying `needed_by` types and error message consistency.

---

#### Phase 1: Unify `needed_by` type on GlobalRead (can be done standalone)

**Step 1**: Change `GlobalRead.needed_by` from `SchedulePosition` to `ValidatorInstruction`
```python
# Before:
needed_by: SchedulePosition = POSITION_INF

# After:
needed_by: ValidatorInstruction = field(default_factory=lambda: MFMA(POSITION_INF))
```

**Step 2**: Update `set_gr_needed_by_from_lrs()` to assign the instruction object
```python
# Before:
for _, gr in grs:
    gr.needed_by = LR_target.issued_at

# After:
for _, gr in grs:
    gr.needed_by = LR_target
```

**Step 3**: Update `GlobalRead._validate_needed_by()` to use `self.needed_by.issued_at` and `self.needed_by.name` instead of using `self.needed_by` directly as a `SchedulePosition`
```python
# Before:
if self.needed_by == POSITION_INF:
if self.issued_at < self.guaranteed_by < self.needed_by:
needed_by = self.needed_by.vmfma_index

# After:
if self.needed_by.issued_at == POSITION_INF:
if self.issued_at < self.guaranteed_by < self.needed_by.issued_at:
needed_by = self.needed_by.issued_at.vmfma_index
```

**Step 4**: Update error messages in `_validate_needed_by()` to use `self.needed_by.name` instead of hardcoded "LR1"
```python
# Before:
f"... which is after the first corresponding LR1 @ idx={needed_by}. Order must be {name} -> SWait -> SBarrier -> LR1."

# After:
f"... which is after {self.needed_by.name} @ idx={needed_by}. Order must be {name} -> SWait -> SBarrier -> {self.needed_by.name}."
```

**Step 5**: Run tests
```bash
pytest Tensile/Tests/unit/test_CMSValidator*.py -v
```

---

## Recommended Implementation Order

1. **R5: Define Typed Context** (quick win, improves IDE support; `ValidatorPassContext` is a partial step)
2. **R13: Standardize Class Hierarchy** (unify `needed_by` type — standalone, no dependencies)
3. **R12: Document Limitations** (quick win, documentation only)
4. **R9: Clarify Validation Logic** (partially done, further enabled by R13)
5. **R1: Split File Into Modules** (large effort, do after other changes stabilize; constants already extracted to module-level)
6. **R6: Registry Pattern for Packs** (medium effort, can do standalone or with R1)
7. **R10: Separate Timeline** (large effort, do last)
8. **R11: Improve Test Infrastructure** (ongoing, do incrementally)
