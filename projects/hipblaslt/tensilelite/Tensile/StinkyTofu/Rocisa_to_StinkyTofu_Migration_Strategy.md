# Rocisa to StinkyTofu Migration Strategy

This document outlines the strategy for migrating from the legacy `rocisa` assembly generation backend to the new `StinkyTofu` backend in the Tensile codebase. The migration follows a phased approach that maintains backward compatibility while progressively adopting the new infrastructure.

## Table of Contents

- [Goals](#goals)
- [Architecture](#architecture)
- [Migration Phases](#migration-phases)

## Goals

### Primary Objectives

1. **Zero Disruption**: Existing code continues to work without modifications
2. **Gradual Migration**: Migrate components incrementally rather than big-bang rewrites
3. **Dual Validation**: Run both backends in parallel to validate correctness

### Success Criteria

- All kernels generate identical assembly output between backends
- 100% test pass rate with StinkyTofu enabled

## Architecture

### Component Overview

```
┌─────────────────────────────────────────────────────┐
│                  Tensile Codebase                   │
│                                                     │
│  ┌──────────────────────────────────────────────┐   │
│  │          KernelWriter (existing)             │   │
│  └────────────────┬─────────────────────────────┘   │
│                   │                                 │
│                   ▼                                 │
│  ┌──────────────────────────────────────────────┐   │
│  │         ModuleAdapter (NEW)                  │   │
│  │  - Composition over inheritance              │   │
│  │  - Dual-backend routing                      │   │
│  │  - Statistics tracking                       │   │
│  └─────┬────────────────────────────────┬───────┘   │
│        │                                │           │
│        ▼                                ▼           │
│  ┌──────────┐                    ┌──────────────┐   │
│  │  rocisa  │                    │  StinkyTofu  │   │
│  │  Module  │                    │  IRListModule│   │
│  └──────────┘                    └──────────────┘   │
└─────────────────────────────────────────────────────┘
```

### Key Components

#### 1. ModuleAdapter (`ModuleAdapter.py`)

**Purpose**: Drop-in replacement for `rocisa.code.Module` that supports dual backends.

**Design Pattern**: Composition over inheritance
- Contains `_rocisa_module` member (instead of inheriting)
- Contains `_stinky_module` member for StinkyTofu backend
- Delegates all `rocisa::Module` API calls to internal member

**Key Features**:
```python
class Module:
    USE_STINKYTOFU = os.getenv("USE_STINKYTOFU", "1") == "1"
    DEBUG_CONVERSION = os.getenv("DEBUG_STINKYTOFU", "0") == "1"

    def __init__(self, name: str = ""):
        self._rocisa_module = rocisa_code.Module(name)  # Composition
        self._stinky_module = st.createIRList(name)

    def add(self, item, pos=-1):
        # Route to both backends
        self._rocisa_module.add(item, pos)
        self._add_to_stinky(item)
```

**Why Composition?**
- Avoids C++ binding memory leaks when passing Python subclasses
- Enables independent lifecycle management
- Cleaner delegation pattern with `__getattr__`

#### 2. Item Adapter (`ItemAdapters.py`)

**Purpose**: Decorator to create dual-backend wrappers for rocisa Item classes (instructions, directives, etc.).

**Design Pattern**: Decorator with dual instantiation

```python
@instruction_wrapper("VMovB32")
class VMovB32:
    def __init__(self, dst, src, comment = ""):
        # Decorator creates _rocisa_inst automatically with these exact parameters
        # Parameters must match rocisa.instruction.VMovB32 signature
        self.dst = dst
        self.src = src
        self.comment = comment

    def create(self, builder):
        # Create StinkyTofu instruction
        return builder.VMovB32(self.dst, self.src, self.comment)
```

**Key Features**:
- Automatic rocisa instance creation
- Attribute delegation via `__getattr__`
- Lazy StinkyTofu creation (only when needed)

## Migration Phases

### Phase 1: Foundation (Current)

**Status**: ✅ Complete

**Objectives**:
- [x] Create *ModuleAdapter* with composition pattern
- [x] Implement `instruction_wrapper` decorator
- [x] Add dual-backend routing in Module.add()
- [-] Implement `IRListModule::addModule()` with proper cloning (TODO checking)

**Key Changes**:
```python
# Old code (still works)
from rocisa.code import Module

# New code (drop-in replacement)
from Tensile.StinkyTofu.ModuleAdapter import Module
```

**Validation**:
- Statistics tracking shows conversion coverage
- No functional changes to existing code

### Phase 2: Wrapper Migration (In Progress)

**Status**: 🟡 In Progress

**Objectives**:
- [ ] Wrap rocisa Items using `instruction_wrapper` decorator in `ItemAdapters.py`
  - [ ] Common instructions (MFMA, DS_*, VMEM, SMEM, VOP3, VALU)
  - [ ] Directives (SWaitCnt, SBarrier, SBranch, etc.)
  - [ ] Special items (Labels, Comments, Macros)
- [ ] Add StinkyTofu implementations for all wrapped items (The `create(self, builder)` method)
- [ ] Gradually migrate *Tensile* codebase to use wrapped versions
- [ ] Validate assembly output matches byte-for-byte

**Compatibility Rules**:

| Item Type | Can add to rocisa.Module? | Can add to ModuleAdapter? |
|-----------|---------------------------|---------------------------|
| Original rocisa Item | ✅ Yes | ✅ Yes |
| Wrapped Item | ❌ No | ✅ Yes |

**Important**:
- **Wrapped items** (created with `@instruction_wrapper`) can ONLY be added to `.StinkyTofu.ModuleAdapter.Module`
- **ModuleAdapter** accepts BOTH original rocisa items AND wrapped items
- This allows gradual migration without breaking existing code

**Migration Approach**:

**Strategy**: Top-down module replacement with function-scope focus

1. **Start at Module level** (Critical First Step):
   - Replace `rocisa.code.Module` → `Tensile.StinkyTofu.ModuleAdapter.Module` first
   - **Why**: ModuleAdapter accepts BOTH backends, but `rocisa.Module` cannot accept wrapped items
   - Focus on function scope (most rocisa modules in Tensile are created in functions), eg.
     - `openLoop()` in *Tensile/KernelWriterAssembly.py*
     - `noLoadLoop()` in *Tensile/KernelWriter.py*
     - ...

    **Example**:

    - `openLoop()` function before and after migration using ModuleAdapter:
        ```diff
        diff --git a/projects/hipblaslt/tensilelite/Tensile/KernelWriterAssembly.py b/projects/hipblaslt/tensilelite/Tensile/KernelWriterAssembly.py
        index 43820ac111..705f93b007 100644
        --- a/projects/hipblaslt/tensilelite/Tensile/KernelWriterAssembly.py
        +++ b/projects/hipblaslt/tensilelite/Tensile/KernelWriterAssembly.py
        @@ -5890,7 +5890,10 @@ class KernelWriterAssembly(KernelWriter):
        # Open Loop
        ##############################################################################
        def openLoop(self, kernel, tPA, tPB, loopIdx, noLabelGen=False, beginLabelOnly=False):
        -    module = Module("openLoop")
        +    from .StinkyTofu.ModuleAdapter import Module as stModule
        +    from .StinkyTofu.ItemAdapters import VMovB32, VMovB64, Label
        +
        +    module = stModule("openLoop")

            if bool(kernel["ProblemType"]["MXBlockA"]) ^ bool(kernel["ProblemType"]["MXBlockB"]):
            block = max(kernel["ProblemType"]["MXBlockA"], kernel["ProblemType"]["MXBlockB"])
        ```

    - In `KernelWriter.py`,
        - the original `rocisa.Module` only could add rocisa items (therefore the ModuleAdapter.Module rocisa() method is called)
        - `pgr = Module("loopWithPrefetch")` is migrated with `pgr = stModule("loopWithPrefetch")`
        ```diff
        diff --git a/projects/hipblaslt/tensilelite/Tensile/KernelWriter.py b/projects/hipblaslt/tensilelite/Tensile/KernelWriter.py
        index a7b69f4191..c3e40a675d 100644
        --- a/projects/hipblaslt/tensilelite/Tensile/KernelWriter.py
        +++ b/projects/hipblaslt/tensilelite/Tensile/KernelWriter.py
        @@ -2259,7 +2259,7 @@ class KernelWriter(metaclass=abc.ABCMeta):
                module.addComment1("summation loop %u"%i)
                module.add(self.calculateLoopNumIter(kernel, tensorParametersA, tensorParametersB, i))
                if self.states.actualSummationLoops>1:
        -          module.add(self.openLoop(kernel, tensorParametersA, tensorParametersB, i))
        +          module.add(self.openLoop(kernel, tensorParametersA, tensorParametersB, i).rocisa())
            module.add(self.calculateLoopNumIter(kernel, tensorParametersA, tensorParametersB, self.states.unrollIdx))

            if not forceNoTileCode:
        @@ -3301,7 +3301,8 @@ class KernelWriter(metaclass=abc.ABCMeta):

            pack = [ Module() for i in range (self.states.numVgprBuffer) ]

        -    pgr = Module("loopWithPrefetch")
        +    from .StinkyTofu.ModuleAdapter import Module as stModule
        +    pgr = stModule("loopWithPrefetch")
            if kernel["PrefetchGlobalRead"]:
            if self.states.doShadowInit:
                module.add(self.openShadowInit())
        @@ -3527,7 +3528,7 @@ class KernelWriter(metaclass=abc.ABCMeta):
                finalLoop = lc == loopCopies - 1
                loop.add(self._loopBody( kernel, tensorParametersA, tensorParametersB, pack, lc, loopCopies, finalLoop, isDTVGRSecondBuf=isDTVGRSecondBuf ))
            pgr.add(loop)
        -    module.add(pgr)
        +    module.add(pgr.rocisa())

            if kernel["ExpertSchedulingMode"] > 0:
            module.add(SSetRegIMM32B32(dst=HWRegContainer(reg="26", value=[0,2]), src=0x0, comment="enable hardware dependency checking"))
        @@ -3873,7 +3874,7 @@ class KernelWriter(metaclass=abc.ABCMeta):

            # tail: macs
            module.addComment1("tail loop: macs")
        -      module.add(self.openLoop(kernel, tensorParametersA, tensorParametersB, -1, None))
        +      module.add(self.openLoop(kernel, tensorParametersA, tensorParametersB, -1, None).rocisa())

            # Try to use InnerUnroll in the tail loop if allowed:
            KinInnerUnroll = kernel["InnerUnroll"]
        ```

2. **Function-by-function migration**:
   ```python
   # Before: rocisa.Module cannot accept wrapped items
   from rocisa.code import Module
   def buildKernel():
       module = Module("kernel")  # rocisa only!
       module.add(SWaitAlu(...))  # Must use rocisa items

   # After: ModuleAdapter accepts both backends
   def buildKernel():
       from Tensile.StinkyTofu.ModuleAdapter import Module
       module = Module("kernel")  # Can accept both!
       module.add(SWaitAlu(...))  # Can use rocisa OR wrapped
   ```

3. **Gradual item wrapping within each function**:
   - Once function uses ModuleAdapter, gradually replace items:
     - Start with high-frequency items in that function
     - Replace rocisa items with wrapped versions incrementally
     - Both work during transition

4. **Prioritize wrapping by usage frequency**:
   - Phase 2a: Core instructions (MFMA, global/local memory ops)
   - Phase 2b: Synchronization (SWaitCnt, SBarrier, SWaitAlu)
   - Phase 2c: Control flow (branches, labels)
   - Phase 2d: Remaining instructions and directives

5. **Add comprehensive tests** for each module replacement:
   - Test each migrated function/module independently
   - Validate assembly output matches between rocisa-only and dual-backend versions (using `Module.dump()` stats)
   - Verify wrapped items produce correct StinkyTofu IR
   - Ensure no regression in existing functionality

**Function-Scope Migration Pattern**:
```python
# Step 1: Replace Module only (all items still rocisa)
def generateMatrixMul():
    from Tensile.StinkyTofu.ModuleAdapter import Module  # ← Module change only
    from rocisa.instruction import VMovB32, SWaitCnt    # ← Still rocisa items

    module = Module("matmul")
    module.add(VMovB32(...))   # rocisa item - works!
    module.add(SWaitCnt(...))  # rocisa item - works!
    return module

# Step 2: Gradually replace items within function
def generateMatrixMul():
    from Tensile.StinkyTofu.ModuleAdapter import Module
    from Tensile.StinkyTofu.ItemAdapters import VMovB32  # ← Wrapped
    from rocisa.instruction import SWaitCnt              # ← Still rocisa

    module = Module("matmul")
    module.add(VMovB32(...))   # Wrapped - dual backend!
    module.add(SWaitCnt(...))  # rocisa - still works!
    return module

# Step 3: Eventually all wrapped
def generateMatrixMul():
    from Tensile.StinkyTofu.ModuleAdapter import Module
    from Tensile.StinkyTofu.ItemAdapters import VMovB32, SWaitCnt  # ← All wrapped

    module = Module("matmul")
    module.add(VMovB32(...))   # Wrapped - dual backend
    module.add(SWaitCnt(...))  # Wrapped - dual backend
    return module
```

**Example Migration**:
```python
# CRITICAL: Must migrate Module first, then items
# rocisa.Module CANNOT accept wrapped items!

# ❌ WRONG: Wrapped item with rocisa.Module
from rocisa.code import Module
from Tensile.StinkyTofu.ItemAdapters import SWaitAlu  # Wrapped

module = Module("kernel")
module.add(SWaitAlu(vm_vsrc=0))  # ERROR! rocisa.Module rejects wrapped items

# ✅ CORRECT: ModuleAdapter first, then migrate items gradually
from Tensile.StinkyTofu.ModuleAdapter import Module
from Tensile.StinkyTofu.ItemAdapters import SWaitAlu  # Wrapped
from rocisa.instruction import VMovB32  # Still rocisa

module = Module("kernel")
module.add(SWaitAlu(vm_vsrc=0))         # Wrapped - dual backend!
module.add(VMovB32(vgpr(0), vgpr(1)))   # rocisa - still works!
```

**Progress Tracking**:
```python
# Check statistics after migration of a module
module.dump()

# Output shows:

# ======================================================================
# Module: openLoop
# ======================================================================

# --- RocISA Module ---
# rocisa::Module "openLoop"
# |--Label label_openLoopL:
# |--SCmpLeU32 s_cmp_le_u32 s[sgprLoopCounterL], 0x1              // LoopCounterL < EndCounter
# |--SCBranchSCC1 s_cbranch_scc1 label_LoopEndL                      // do not enter LoopL
# |--Label label_LoopBeginL:


# --- StinkyTofu IRList ---
# IRListModule: openLoop (arch 2)
#   openLoopL:
#   LoopBeginL:


# --- Conversion Statistics ---
# Rocisa items:     4
# StinkyTofu items: 2
# Failed:           0
# Coverage:               50.0%
# ======================================================================
```

### Phase 3: Validation & Optimization

**Status**: ⬜ Planned

**Objectives**:
- [ ] Enable assembly comparison mode
- [ ] Identify and fix assembly differences

### Phase 4: Cutover

**Status**: ⬜ Future

**Objectives**:
- [ ] Make StinkyTofu the default backend
- [ ] Deprecate rocisa dependencies
- [ ] Remove rocisa fallback code
- [ ] Clean up adapter layer
